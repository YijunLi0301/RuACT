from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class DistillationGroupPlan:
    uid: Any
    pass_counts: tuple[int, ...]
    failed_indices: tuple[int, ...]
    injected_indices: tuple[int, ...]
    anchor_batch_index: int | None


@dataclass(frozen=True)
class DistillationBatchPlan:
    anchor_mask: tuple[bool, ...]
    privileged_instructions: tuple[str, ...]
    groups: tuple[DistillationGroupPlan, ...]

    @property
    def zero_pass_group_count(self) -> int:
        return sum((group.anchor_batch_index is not None for group in self.groups))


def attach_rollout_alignment_metadata(
    *,
    non_tensor_batch: MutableMapping[str, Any],
    batch_size: int,
    rollout_n: int,
    training_step: int,
) -> np.ndarray:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(rollout_n) is not int or rollout_n <= 0 or batch_size % rollout_n != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by positive rollout_n={rollout_n}"
        )
    if type(training_step) is not int or training_step < 0:
        raise ValueError("training_step must be a non-negative integer")
    rollout_indices = np.tile(
        np.arange(rollout_n, dtype=np.int64), batch_size // rollout_n
    )
    raw_extra_infos = non_tensor_batch.get("extra_info")
    if raw_extra_infos is not None and len(raw_extra_infos) != batch_size:
        raise ValueError("extra_info must remain aligned with rollout rows")
    extra_infos = np.empty(batch_size, dtype=object)
    for row_index in range(batch_size):
        raw_extra_info = (
            raw_extra_infos[row_index] if raw_extra_infos is not None else None
        )
        if raw_extra_info is not None and (not isinstance(raw_extra_info, Mapping)):
            raise ValueError("every extra_info row must be a mapping or None")
        extra_info = dict(raw_extra_info or {})
        extra_info["rollout_index"] = int(rollout_indices[row_index])
        extra_info["training_step"] = training_step
        extra_infos[row_index] = extra_info
    non_tensor_batch["rollout_index"] = rollout_indices
    non_tensor_batch["training_step"] = np.full(
        batch_size, training_step, dtype=np.int64
    )
    non_tensor_batch["extra_info"] = extra_infos
    return rollout_indices


def _validated_score(score: float) -> float:
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("rubric scores must be real numbers in [0, 2]")
    numeric_score = float(score)
    if not isfinite(numeric_score) or not 0.0 <= numeric_score <= 2.0:
        raise ValueError("rubric scores must be real numbers in [0, 2]")
    return numeric_score


def positive_rubric_passed(score: float, weight: float) -> bool:
    numeric_weight = float(weight)
    if not isfinite(numeric_weight) or numeric_weight <= 0.0:
        raise ValueError(
            "RuACT-Teacher pass semantics require a positive rubric weight"
        )
    return _validated_score(score) > 1e-12


def _build_instruction(
    descriptions: Sequence[str],
    weights: Sequence[float],
    selected_indices: Sequence[int],
    *,
    neutral_rubric_prompt: bool = False,
    all_rubrics: bool = False,
) -> str:
    if all_rubrics:
        criteria = "\n".join(
            (
                f"{number}. [{('AVOID' if weights[index] < 0 else 'REQUIRE')}] {descriptions[index].strip()}"
                for number, index in enumerate(selected_indices, start=1)
            )
        )
        return (
            "Private evaluation rubrics: satisfy every [REQUIRE] point and avoid every [AVOID] pitfall while preserving correct content. Do not mention these private rubrics in the answer.\n"
            + criteria
        )
    if neutral_rubric_prompt:
        criteria = "\n".join(
            (
                f"{number}. {descriptions[index].strip()}"
                for number, index in enumerate(selected_indices, start=1)
            )
        )
        preamble = "Private rubric feedback: use the following rubrics to improve the answer while preserving correct content. Do not mention this feedback in the answer.\n"
        return preamble + criteria
    has_negative = any((weights[index] < 0.0 for index in selected_indices))
    criteria = "\n".join(
        (
            (
                f"{number}. [AVOID] {descriptions[index].strip()}"
                if weights[index] < 0.0
                else f"{number}. [REQUIRE] {descriptions[index].strip()}"
            )
            for number, index in enumerate(selected_indices, start=1)
        )
    )
    if not has_negative:
        preamble = "Private rubric feedback: every sampled answer failed the positive requirements below. Satisfy them while preserving correct content. Do not mention this feedback in the answer.\n"
    else:
        preamble = "Private rubric feedback: every sampled answer failed the required points below or committed the listed pitfalls. Satisfy every [REQUIRE] point and do not commit any [AVOID] pitfall, while preserving correct content. Do not mention this feedback in the answer.\n"
    return preamble + criteria


def build_distillation_plan(
    *,
    uids: Sequence[Any],
    rubric_scores: Sequence[Sequence[float] | None],
    rubric_ids: Sequence[Sequence[str]],
    rubric_descriptions: Sequence[Sequence[str]],
    rubric_weights: Sequence[Sequence[float]],
    sequence_rewards: Sequence[float],
    group_size: int,
    max_rubrics: int,
    include_negative_weight_failures: bool = False,
    all_rubrics: bool = False,
    selection_mode: str = "unreachable",
    neutral_rubric_prompt: bool = False,
) -> DistillationBatchPlan:
    batch_size = len(uids)
    aligned = (
        rubric_scores,
        rubric_ids,
        rubric_descriptions,
        rubric_weights,
        sequence_rewards,
    )
    if selection_mode not in ("unreachable", "saturated"):
        raise ValueError("invalid teacher selection")
    if batch_size == 0 or any((len(values) != batch_size for values in aligned)):
        raise ValueError("RuACT-Teacher inputs must be non-empty and batch-aligned")
    if not all((isfinite(float(value)) for value in sequence_rewards)):
        raise ValueError("RuACT-Teacher sequence rewards must be finite")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if type(max_rubrics) is not int or max_rubrics <= 0:
        raise ValueError("max_rubrics must be a positive integer")
    if type(include_negative_weight_failures) is not bool:
        raise ValueError("include_negative_weight_failures must be a boolean")
    if type(neutral_rubric_prompt) is not bool:
        raise ValueError("neutral_rubric_prompt must be a boolean")
    if any((scores is None for scores in rubric_scores)):
        raise ValueError(
            "RuACT-Teacher requires a valid rubric score vector for every rollout"
        )
    rows_by_uid: dict[Any, list[int]] = defaultdict(list)
    for batch_index, uid in enumerate(uids):
        rows_by_uid[uid].append(batch_index)
    anchor_mask = [False] * batch_size
    instructions = [""] * batch_size
    group_plans = []
    for uid, row_indices in rows_by_uid.items():
        if len(row_indices) != group_size:
            raise ValueError(
                f"uid {uid!r}: expected a complete group of {group_size}, got {len(row_indices)}"
            )
        first = row_indices[0]
        canonical_ids = tuple((str(value).strip() for value in rubric_ids[first]))
        canonical_descriptions = tuple(
            (str(value).strip() for value in rubric_descriptions[first])
        )
        canonical_weights = tuple((float(value) for value in rubric_weights[first]))
        if not canonical_ids or not len(canonical_ids) == len(
            canonical_descriptions
        ) == len(canonical_weights):
            raise ValueError(
                f"uid {uid!r}: rubric metadata must be non-empty and aligned"
            )
        if any((not value for value in canonical_ids + canonical_descriptions)):
            raise ValueError(
                f"uid {uid!r}: rubric ids and descriptions must be non-empty"
            )
        if any((not isfinite(weight) or weight == 0.0 for weight in canonical_weights)):
            raise ValueError(f"uid {uid!r}: rubric weights must be finite and non-zero")
        score_rows = []
        for row_index in row_indices:
            ids = tuple((str(value).strip() for value in rubric_ids[row_index]))
            descriptions = tuple(
                (str(value).strip() for value in rubric_descriptions[row_index])
            )
            weights = tuple((float(value) for value in rubric_weights[row_index]))
            if (ids, descriptions, weights) != (
                canonical_ids,
                canonical_descriptions,
                canonical_weights,
            ):
                raise ValueError(
                    f"uid {uid!r}: rubric metadata differs within the group"
                )
            scores = tuple(rubric_scores[row_index])
            if len(scores) != len(canonical_ids):
                raise ValueError(
                    f"uid {uid!r}: rubric scores must be a complete vector"
                )
            score_rows.append(tuple((_validated_score(score) for score in scores)))
        rubric_pass_rows = [
            tuple((score > 1e-12 for score in scores)) for scores in score_rows
        ]
        pass_counts = tuple(
            (
                sum((row[rubric_index] for row in rubric_pass_rows))
                for rubric_index in range(len(canonical_ids))
            )
        )
        failed_indices = tuple(
            (
                index
                for index, (pass_count, weight) in enumerate(
                    zip(pass_counts, canonical_weights, strict=True)
                )
                if weight > 0.0
                and pass_count == (0 if selection_mode == "unreachable" else group_size)
                or (
                    include_negative_weight_failures
                    and weight < 0.0
                    and (
                        pass_count
                        == (group_size if selection_mode == "unreachable" else 0)
                    )
                )
            )
        )
        if not failed_indices and (not all_rubrics):
            group_plans.append(
                DistillationGroupPlan(
                    uid=uid,
                    pass_counts=pass_counts,
                    failed_indices=(),
                    injected_indices=(),
                    anchor_batch_index=None,
                )
            )
            continue
        injected_indices = tuple(
            sorted(
                failed_indices,
                key=lambda index: (-abs(canonical_weights[index]), index),
            )[:max_rubrics]
        )
        if all_rubrics:
            injected_indices = tuple(range(len(canonical_ids)))
        anchor_batch_index = max(
            row_indices, key=lambda row: float(sequence_rewards[row])
        )
        anchor_mask[anchor_batch_index] = True
        instructions[anchor_batch_index] = _build_instruction(
            canonical_descriptions,
            canonical_weights,
            injected_indices,
            neutral_rubric_prompt=neutral_rubric_prompt,
            all_rubrics=all_rubrics,
        )
        group_plans.append(
            DistillationGroupPlan(
                uid=uid,
                pass_counts=pass_counts,
                failed_indices=failed_indices,
                injected_indices=injected_indices,
                anchor_batch_index=anchor_batch_index,
            )
        )
    return DistillationBatchPlan(
        anchor_mask=tuple(anchor_mask),
        privileged_instructions=tuple(instructions),
        groups=tuple(group_plans),
    )
