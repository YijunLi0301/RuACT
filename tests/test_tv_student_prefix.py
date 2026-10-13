import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
from typing import Optional
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.distillation_context import find_over_context_rows
from verl.trainer.ppo.distillation import build_distillation_plan
from verl.trainer.ppo.distillation_teacher import (
    pad_indices_to_multiple,
    split_thinking_teacher_response,
)
from verl.trainer.ppo import distillation_teacher
from verl.trainer.ppo.distillation_teacher_gate import (
    filter_anchors_by_teacher_verification,
    select_teacher_reward_feedback,
)
from verl.workers.actor.distillation_loss import compose_distillation_loss
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config.actor import DistillationConfig

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "verl/trainer/ppo/ray_trainer.py"
ACTOR = ROOT / "verl/workers/actor/dp_actor.py"


def trainer_method():
    tree = ast.parse(TRAINER.read_text())
    node = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_maybe_build_distillation_batch"
        )
    )
    scope = dict(globals())
    if hasattr(distillation_teacher, "build_final_only_distillation_tensors"):
        scope["build_final_only_distillation_tensors"] = (
            distillation_teacher.build_final_only_distillation_tensors
        )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER), "exec"), scope)
    return scope[node.name]


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]

    def decode(self, ids, **kwargs):
        return "".join((chr(i - 2) for i in ids if i > 1))

    def apply_chat_template(self, messages, **kwargs):
        if messages and isinstance(messages[0], dict):
            return self.encode(
                "|".join((m["content"] for m in messages)) + "|assistant:"
            )
        rows = [self.apply_chat_template(ms) for ms in messages]
        width = max(map(len, rows))
        ids = torch.tensor([[0] * (width - len(row)) + row for row in rows])
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


def make_case(
    tmp_path,
    *,
    teacher_text="teacher accepted",
    gate=True,
    final_only=True,
    k=1,
    async_mode=False,
    no_failed=False,
):
    tokenizer = Tokenizer()
    responses = torch.tensor(
        [[10, 11, 12, 1], [20, 1, 0, 0], [30, 31, 1, 0], [40, 41, 42, 1]]
    )
    mask = responses.ne(0).long()
    uids = np.array(["a", "a", "b", "b"], dtype=object)
    advantages = torch.tensor([1.0, 1.5, -1.0, 2.0])[:, None] * mask
    raw_prompt = np.empty(4, dtype=object)
    raw_prompt[:] = [[{"role": "user", "content": "query " + uid}] for uid in uids]
    prompts = tokenizer.apply_chat_template(list(raw_prompt))
    inputs = torch.cat([prompts["input_ids"], responses], -1)
    attention = torch.cat([prompts["attention_mask"], mask], -1)
    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "response_mask": mask,
            "advantages": advantages,
            "input_ids": inputs,
            "attention_mask": attention,
            "position_ids": compute_position_id_with_mask(attention),
            "rm_scores": torch.tensor([[0.5], [0.25], [0.5], [0.25]]),
            "token_level_scores": torch.tensor([[1.0], [1.5], [-1.0], [2.0]]),
            "token_level_rewards": torch.tensor([[1.0], [1.5], [-1.0], [2.0]]),
        },
        non_tensors={"uid": uids, "raw_prompt": raw_prompt},
    )
    feedback = {
        "judge_valid": [True] * 4,
        "rubric_scores": [[1 if no_failed else 0, v] for v in [1, 0, 1, 0]],
        "rubric_ids": [["hard", "easy"]] * 4,
        "rubric_descriptions": [["hard requirement", "easy requirement"]] * 4,
        "rubric_weights": [[1, 1]] * 4,
    }
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": "distillation"},
                    "distillation": {
                        "max_rubrics": 3,
                        "teacher_gate": gate,
                        "teacher_final_only": final_only,
                        "teacher_k": k,
                        "teacher_max_response_length": 100,
                        "teacher_enable_thinking": False,
                    },
                },
                "rollout": {"n": 2, "max_model_len": None, "agent": {"num_workers": 3}},
            },
            "data": {
                "max_prompt_length": 1024,
                "max_response_length": 512,
                "apply_chat_template_kwargs": {"enable_thinking": False},
            },
            "trainer": {"default_local_dir": str(tmp_path / "checkpoints")},
        }
    )
    (tmp_path / "diagnostics").mkdir(parents=True, exist_ok=True)
    calls = []

    def generate(batch):
        calls.append(batch)
        texts = teacher_text if isinstance(teacher_text, list) else [teacher_text]
        rows = [
            tokenizer.encode(texts[i % len(texts)]) + [1] for i in range(len(batch))
        ]
        width = max(map(len, rows))
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return DataProto.from_dict(
            tensors={"responses": ids, "response_mask": ids.ne(0).long()}
        )

    reward = SimpleNamespace(
        _candidate_final_answer=lambda text: (text, "ok"),
        _call_judge=lambda **kw: SimpleNamespace(
            scores=[0, 0] if "rejected" in kw["final_answer"] else [1, 1]
        ),
        JudgeGiveUpError=TimeoutError,
        _build_judge_prompt=lambda **kw: "test prompt",
        CANONICAL_JUDGE_MODEL="",
    )
    aggregate = SimpleNamespace(
        aggregate_weighted_reward=lambda scores, weights: (
            sum((s * w for s, w in zip(scores, weights)))
            / sum((w for w in weights if w > 0))
        )
    )
    worker = SimpleNamespace(generate_sequences=generate)
    trainer = SimpleNamespace(
        config=config,
        tokenizer=tokenizer,
        global_steps=1,
        async_rollout_mode=async_mode,
        actor_rollout_wg=worker,
        async_rollout_manager=worker,
        _load_teacher_gate_modules=lambda: (reward, aggregate),
    )
    trainer.build = MethodType(trainer_method(), trainer)
    return (trainer, batch, feedback, calls)


@pytest.mark.parametrize("final_only", [True, False])
@pytest.mark.parametrize("async_mode", [True, False])
def test_gate_uses_teacher_but_distillation_keeps_best_student(
    tmp_path, final_only, async_mode
):
    t, batch, feedback, calls = make_case(
        tmp_path, final_only=final_only, async_mode=async_mode
    )
    original = {k: v.clone() for k, v in batch.batch.items()}
    data, metrics = t.build(batch, feedback)
    assert data.batch["distillation_anchor_mask"].tolist() == [0, 1, 0, 1]
    assert "distillation_target_responses" not in data.batch
    assert torch.equal(
        data.batch["privileged_input_ids"][:, -4:], batch.batch["responses"]
    )
    assert torch.equal(
        data.batch["privileged_attention_mask"][:, -4:], batch.batch["response_mask"]
    )
    assert len(calls) == 1
    assert metrics["distillation/teacher_gate_passed"] == 2
    for key, value in original.items():
        assert torch.equal(batch.batch[key], value), key


def test_accepted_teacher_text_cannot_change_opsd_inputs(tmp_path):
    results = []
    for text in ["teacher accepted", "entirely different accepted teacher text"]:
        t, batch, feedback, _ = make_case(tmp_path, teacher_text=text)
        results.append(t.build(batch, feedback)[0])
    assert set(results[0].batch.keys()) == set(results[1].batch.keys())
    for key in results[0].batch.keys():
        assert torch.equal(results[0].batch[key], results[1].batch[key]), key


@pytest.mark.parametrize(
    "text,k,expected",
    [
        ("teacher rejected", 1, [0, 0, 0, 0]),
        (["teacher accepted", "teacher rejected"], 2, [0, 0, 0, 0]),
        ("teacher accepted", 2, [0, 1, 0, 1]),
    ],
)
def test_gate_rejection_and_k2(tmp_path, text, k, expected):
    t, batch, feedback, _ = make_case(tmp_path, teacher_text=text, k=k)
    result, _ = t.build(batch, feedback)
    assert result.batch["distillation_anchor_mask"].tolist() == expected
    assert "distillation_target_responses" not in result.batch


@pytest.mark.parametrize(
    "gate,no_failed,expected",
    [(False, False, [0, 1, 0, 1]), (True, True, [0, 0, 0, 0])],
)
def test_no_unnecessary_teacher_generation(tmp_path, gate, no_failed, expected):
    t, batch, feedback, calls = make_case(tmp_path, gate=gate, no_failed=no_failed)
    result, _ = t.build(batch, feedback)
    assert calls == []
    assert result.batch["distillation_anchor_mask"].tolist() == expected


def test_opsd_uses_prepared_rewards_before_advantage_computation():
    tree = ast.parse(TRAINER.read_text())
    fit = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "fit"
        )
    )
    calls = [n for n in ast.walk(fit) if isinstance(n, ast.Call)]
    advantage_line = next(
        (
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Name) and n.func.id == "compute_advantage"
        )
    )
    opsd_line = next(
        (
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Attribute)
            and n.func.attr == "_maybe_build_distillation_batch"
        )
    )
    reward_lines = [
        n.lineno
        for n in ast.walk(fit)
        if isinstance(n, ast.Assign)
        and any(
            (
                isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and (t.slice.value == "token_level_rewards")
                for t in n.targets
            )
        )
    ]
    assert max(reward_lines) < opsd_line < advantage_line


def test_gate_does_not_require_legacy_final_only_flag():
    DistillationConfig(teacher_gate=True, teacher_final_only=False)


def test_selection_does_not_need_advantages(tmp_path):
    t, batch, feedback, _ = make_case(tmp_path)
    del batch.batch["advantages"]
    result, _ = t.build(batch, feedback)
    assert result.batch["distillation_anchor_mask"].tolist() == [0, 1, 0, 1]


def test_actor_has_no_teacher_trajectory_or_extra_public_forward():
    tree = ast.parse(ACTOR.read_text())
    update = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "update_policy"
        )
    )
    constants = {
        n.value
        for n in ast.walk(update)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert (
        not {
            "distillation_target_responses",
            "distillation_public_input_ids",
            "distillation_target_mask",
        }
        & constants
    )


def test_privileged_prompt_keeps_previous_total_context_budget(tmp_path):
    t, batch, feedback, _ = make_case(tmp_path)
    feedback["rubric_descriptions"] = [["hard " * 180, "easy"]] * 4
    result, _ = t.build(batch, feedback)
    assert "privileged_input_ids" in result.batch
    assert result.batch["privileged_input_ids"].shape[-1] > 1024


def test_interleaved_groups_and_reward_ties(tmp_path):
    t, batch, feedback, _ = make_case(tmp_path)
    order = [0, 2, 1, 3]
    batch = batch[order]
    feedback = {key: [value[i] for i in order] for key, value in feedback.items()}
    result, _ = t.build(batch, feedback)
    assert result.batch["distillation_anchor_mask"].tolist() == [0, 0, 1, 1]
    batch.batch["token_level_rewards"].zero_()
    result, _ = t.build(batch, feedback)
    assert result.batch["distillation_anchor_mask"].tolist() == [1, 1, 0, 0]


def test_eight_rollout_planner_uses_reward_not_positive_rubric_proxy():
    result = build_distillation_plan(
        uids=["q"] * 8,
        rubric_scores=[[0, 1, 1]] * 8,
        rubric_ids=[["hard", "easy", "pitfall"]] * 8,
        rubric_descriptions=[["hard", "easy", "pitfall"]] * 8,
        rubric_weights=[[5, 2, -3]] * 8,
        sequence_rewards=[-1, -2, 0, 3, 1, 0, 2, -3],
        group_size=8,
        max_rubrics=3,
        include_negative_weight_failures=True,
    )
    assert result.groups[0].anchor_batch_index == 3
    assert result.groups[0].injected_indices == (0, 2)


@pytest.mark.parametrize("accepted", [True, False])
def test_actual_actor_forward_block_and_gradient(tmp_path, accepted):
    t, batch, feedback, _ = make_case(
        tmp_path, teacher_text="accepted" if accepted else "rejected"
    )
    extra, _ = t.build(batch, feedback)
    batch = batch.union(extra)
    inputs = dict(batch.batch.items())
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Embedding(256, 8), torch.nn.Linear(8, 256))
    calls = []

    def forward(model_inputs, *, topk_indices=None, distill_topk=None, **kwargs):
        calls.append((torch.is_grad_enabled(), model_inputs))
        hidden = model[0](model_inputs["input_ids"])
        hidden = (hidden * model_inputs["attention_mask"].unsqueeze(-1)).cumsum(1)
        n = model_inputs["responses"].shape[-1]
        logps = model[1](hidden[:, -n - 1 : -1] * 0.01).log_softmax(-1)
        if topk_indices is None:
            topk_indices = logps.topk(distill_topk, dim=-1).indices
        return {
            "topk_logps": logps.gather(-1, topk_indices),
            "topk_indices": topk_indices,
            "log_probs": logps.gather(
                -1, model_inputs["responses"].unsqueeze(-1)
            ).squeeze(-1),
        }

    actor = SimpleNamespace(
        _forward_micro_batch=forward, teacher_module=None, actor_module=model
    )
    tree = ast.parse(ACTOR.read_text())
    update = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "update_policy"
        )
    )
    block = next(
        (
            n.body
            for n in ast.walk(update)
            if isinstance(n, ast.For)
            and any(
                (
                    isinstance(x, ast.Assign)
                    and any(
                        (
                            isinstance(t, ast.Name)
                            and t.id == "privileged_top_log_probs"
                            for t in x.targets
                        )
                    )
                    for x in n.body
                )
            )
        )
    )

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            (isinstance(x, ast.Name) and x.id == name for x in node.targets)
        )

    start = next(
        (i for i, n in enumerate(block) if assigns(n, "privileged_top_log_probs"))
    )
    end = (
        next((i for i, n in enumerate(block) if assigns(n, "student_topk_indices"))) + 1
    )
    scope = {
        "torch": torch,
        "self": actor,
        "model_inputs": inputs,
        "distillation_enabled": True,
        "distillation_config": OmegaConf.create(
            {"top_k": 8, "entropy_weighting": False, "teacher_final_only": True}
        ),
        "distillation_ema_active": False,
        "distillation_topk_temperature": 1.0,
        "temperature": 1.0,
        "calculate_entropy": False,
        "return_all_logps": False,
        "distill_topk": None,
    }
    exec(
        compile(ast.Module(body=block[start:end], type_ignores=[]), str(ACTOR), "exec"),
        scope,
    )
    assert len(calls) == 2
    assert [grad for grad, _ in calls] == [False, True]
    assert all((torch.equal(inp["responses"], inputs["responses"]) for _, inp in calls))
    assert torch.equal(calls[1][1]["input_ids"], inputs["input_ids"])
    assert scope["public_top_log_probs"].requires_grad
    assert not scope["privileged_top_log_probs"].requires_grad
    output = compose_distillation_loss(
        vanilla_policy_loss=scope["log_prob"].sum() * 0,
        public_top_log_probs=scope["public_top_log_probs"],
        privileged_top_log_probs=scope["privileged_top_log_probs"],
        response_mask=inputs["response_mask"],
        anchor_mask=inputs["distillation_anchor_mask"],
        beta=0.0125,
        group_size=8,
        normalization="valid",
        token_kl_clip=10,
        negative_kl_tolerance=0.0001,
    )
    output.total_loss.backward()
    grad_norm = sum((p.grad.abs().sum().item() for p in model.parameters()))
    assert grad_norm > 0 if accepted else grad_norm == 0
