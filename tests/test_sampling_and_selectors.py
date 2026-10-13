import numpy as np
import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.dynamic_sampling import DynamicSamplingBuffer
from verl.trainer.ppo.distillation import build_distillation_plan


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
