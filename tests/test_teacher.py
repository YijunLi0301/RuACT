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
from ruact.teacher.distillation_context import find_over_context_rows
from ruact.teacher.distillation import build_distillation_plan
from ruact.teacher.distillation_teacher import (
    pad_indices_to_multiple,
    split_thinking_teacher_response,
)
from ruact.teacher import distillation_teacher
from ruact.teacher.distillation_teacher_gate import (
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


def test_teacher_selectors_and_all_rubric_control():
    args = dict(
        uids=["q"] * 8,
        rubric_scores=[[0, 1, int(i < 4), 1] for i in range(8)],
        rubric_ids=[["zero", "full", "mixed", "pitfall"]] * 8,
        rubric_descriptions=[["z", "f", "m", "p"]] * 8,
        rubric_weights=[[1, 2, 3, -4]] * 8,
        sequence_rewards=list(range(8)),
        group_size=8,
        max_rubrics=3,
        include_negative_weight_failures=True,
    )
    default = build_distillation_plan(**args)
    assert default.groups[0].injected_indices == (3, 0)
    saturated = build_distillation_plan(**args, selection_mode="saturated")
    assert saturated.groups[0].injected_indices == (1,)
    all_items = build_distillation_plan(**args, all_rubrics=True)
    assert all_items.groups[0].injected_indices == (0, 1, 2, 3)
    assert all_items.anchor_mask[-1] is True
    assert "[AVOID] p" in all_items.privileged_instructions[-1]


def test_teacher_invalid_selector_rejected():
    with pytest.raises(ValueError, match="selection"):
        build_distillation_plan(
            uids=[],
            rubric_scores=[],
            rubric_ids=[],
            rubric_descriptions=[],
            rubric_weights=[],
            sequence_rewards=[],
            group_size=8,
            max_rubrics=3,
            selection_mode="unknown",
        )


@pytest.mark.parametrize("accepted", [True, False])
def test_teacher_is_detached_and_only_accepted_anchors_train(accepted):
    student = torch.tensor([[[0.2, 0.8]]], requires_grad=True)
    teacher = torch.tensor([[[0.7, 0.3]]], requires_grad=True)
    result = compose_distillation_loss(
        vanilla_policy_loss=student.sum() * 0,
        public_top_log_probs=student.log(),
        privileged_top_log_probs=teacher.log(),
        response_mask=torch.ones(1, 1),
        anchor_mask=torch.tensor([accepted]),
        beta=0.05 / 8,
        group_size=8,
    )
    result.total_loss.backward()
    assert teacher.grad is None
    assert (student.grad.abs().sum() > 0).item() == accepted


def test_privileged_context_budget():
    assert find_over_context_rows([2048, 2049], [4096, 4096], 6144) == (1,)
