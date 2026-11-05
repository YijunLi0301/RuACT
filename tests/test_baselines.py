import ast
import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from ruact.baselines import ruscarl
from ruact.baselines.simple_reweight import rewards
from verl import DataProto
from verl.trainer.ppo.dynamic_sampling import DynamicSamplingBuffer

ROOT = Path(__file__).resolve().parents[1]


def test_reweight_polarity_and_constant_columns():
    scores = [[0, 1, 1], [1, 1, 1]]
    result, baseline, factors = rewards(scores, [2, -1, 1], 0.5)
    np.testing.assert_allclose(factors, [1.5, 1, 1])
    np.testing.assert_allclose(result, [1 / 5, 4 / 5])
    np.testing.assert_allclose(baseline, [1 / 4, 3 / 4])


def test_scaffold_schedule_and_zero_reveal():
    a = ruscarl._calculate_rubric_ratio(
        0,
        8,
        "step_sigmoid",
        global_steps=1,
        total_training_steps=200,
        step_sigmoid_start_point=0.2,
        step_sigmoid_steepness=125,
    )
    b = ruscarl._calculate_rubric_ratio(
        0,
        8,
        "step_sigmoid",
        global_steps=200,
        total_training_steps=200,
        step_sigmoid_start_point=0.2,
        step_sigmoid_steepness=125,
    )
    assert a > 0.999 and b < 1e-20

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

    truth = {
        "ground_truth": {"rubrics": [{"description": "Required detail", "weight": 1}]}
    }
    original = [1, 2, 3]
    prompt, metrics = ruscarl.scaffold_prompt(
        Tokenizer(),
        original,
        truth,
        sample_index=7,
        group_size=8,
        step=1,
        total_steps=200,
    )
    assert prompt == original and metrics["rubric_count"] == 0
    prompt, metrics = ruscarl.scaffold_prompt(
        Tokenizer(),
        original,
        truth,
        sample_index=0,
        group_size=8,
        step=1,
        total_steps=200,
    )
    assert prompt[-3:] == original and len(prompt) > 3
    assert original == [1, 2, 3]


def test_scaffold_generation_does_not_change_training_prompt():
    source = ast.parse(
        (ROOT / "verl/experimental/agent_loop/single_turn_agent_loop.py").read_text()
    )
    fn = next(
        n
        for n in ast.walk(source)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
    )
    scope = {
        "Any": object,
        "AgentLoopOutput": SimpleNamespace,
        "uuid4": uuid4,
        "simple_timer": lambda *a: nullcontext(),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<agent>", "exec"), scope)
    sent = []

    async def vision(messages):
        return {}

    async def template(*a, **kw):
        return [10, 20]

    async def generate(**kwargs):
        sent.append(kwargs["prompt_ids"])
        return SimpleNamespace(token_ids=[30, 40], log_probs=None, routed_experts=None)

    agent = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"total_training_steps": 200, "ruscarl": {"enabled": True}},
                "actor_rollout_ref": {"rollout": {"n": 8, "max_model_len": 16000}},
            }
        ),
        response_length=10,
        tool_schemas=[],
        process_vision_info=vision,
        apply_chat_template=template,
        tokenizer=SimpleNamespace(encode=lambda text, **kw: list(text.encode())),
        server_manager=SimpleNamespace(generate=generate),
    )
    kwargs = {
        "raw_prompt": [{"role": "user", "content": "Question"}],
        "reward_model": {"rubrics": [{"criterion": "Important", "points": 1}]},
        "rollout_index": 0,
        "training_step": 1,
    }
    out = asyncio.run(scope["run"](agent, {}, **kwargs))
    assert out.prompt_ids == [10, 20]
    assert out.response_ids == [30, 40]
    assert sent[0] != out.prompt_ids and sent[0][-2:] == out.prompt_ids
    kwargs.pop("training_step")
    out = asyncio.run(scope["run"](agent, {}, **kwargs))
    assert sent[-1] == [10, 20]


def batch_for(group_values, prefix="g"):
    values = [v for group in group_values for v in group]
    count = len(values)
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(count, 1),
            "attention_mask": torch.ones(count, 2),
        },
        non_tensors={
            "uid": np.array(
                [prefix + str(i) for i, g in enumerate(group_values) for _ in g]
            ),
            "rollout_index": np.array([j for g in group_values for j in range(len(g))]),
            "judge_valid": np.array([True] * count),
            "rubric_scores": np.array([[int(v > 0)] for v in values], dtype=object),
        },
    )
    batch.meta_info["reward_extra_keys"] = ("judge_valid", "rubric_scores")
    return batch, torch.tensor(values, dtype=torch.float32).reshape(-1, 1)


def test_dynamic_sampling_refills_exactly_with_same_group_order():
    pool = DynamicSamplingBuffer(target_groups=2, group_size=2, max_rounds=3)
    first, reward = batch_for([[0, 0], [0, 1]])
    assert pool.add(first, reward) is False
    second, reward = batch_for([[1, 1], [2, 3], [4, 5]], prefix="h")
    assert pool.add(second, reward) is True
    batch, reward, extras = pool.take()
    assert batch.non_tensor_batch["uid"].tolist() == ["g1", "g1", "h1", "h1"]
    assert reward.flatten().tolist() == [0, 1, 2, 3]
    assert len(extras["judge_valid"]) == 4
    assert pool.metrics()["dynamic_sampling/rejected_groups"] == 2
    assert pool.metrics()["dynamic_sampling/overflow_groups"] == 1
    with pytest.raises(RuntimeError):
        pool.take()


def test_dynamic_sampling_round_limit_fails_without_partial_update():
    pool = DynamicSamplingBuffer(target_groups=2, group_size=2, max_rounds=1)
    batch, reward = batch_for([[0, 0], [0, 1]])
    with pytest.raises(RuntimeError, match="round limit"):
        pool.add(batch, reward)
    with pytest.raises(RuntimeError, match="incomplete"):
        pool.take()
