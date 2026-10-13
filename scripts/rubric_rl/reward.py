from __future__ import annotations
from dataclasses import dataclass
from math import isfinite


class RewardContractError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedScores:
    scores: tuple[int, ...]


def parse_batched_scores(
    raw_scores, num_rubrics: int, *, score_scale: float = 1.0
) -> ParsedScores:
    if (
        isinstance(num_rubrics, bool)
        or not isinstance(num_rubrics, int)
        or num_rubrics <= 0
    ):
        raise RewardContractError("num_rubrics must be a positive integer")
    if float(score_scale) != 1.0:
        raise RewardContractError(
            "the RuACT judge protocol uses unscaled binary scores"
        )
    if not isinstance(raw_scores, list) or len(raw_scores) != num_rubrics:
        raise RewardContractError("judge scores must cover every rubric exactly once")
    scores: list[int | None] = [None] * num_rubrics
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != {"id", "score"}:
            raise RewardContractError(
                "each judge score must contain exactly id and score"
            )
        index = item["id"]
        score = item["score"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or (not 0 <= index < num_rubrics)
        ):
            raise RewardContractError("judge score id is out of range")
        if scores[index] is not None:
            raise RewardContractError("judge score ids must be unique")
        if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1):
            raise RewardContractError("judge scores must be binary integers")
        scores[index] = score
    if any((score is None for score in scores)):
        raise RewardContractError("judge scores are incomplete")
    return ParsedScores(tuple((int(score) for score in scores)))


def rubric_passed(weight: float, judge_score: float) -> bool:
    numeric_weight = float(weight)
    numeric_score = float(judge_score)
    if not isfinite(numeric_weight) or numeric_weight == 0.0:
        raise RewardContractError("rubric weight must be finite and non-zero")
    if not isfinite(numeric_score) or numeric_score not in (0.0, 1.0):
        raise RewardContractError("judge score must be binary")
    return numeric_score == 1.0 if numeric_weight > 0.0 else numeric_score == 0.0


def aggregate_weighted_reward(
    scores: list[int] | tuple[int, ...], weights: list[float]
) -> float:
    if not scores or len(scores) != len(weights):
        raise RewardContractError("scores and weights must be non-empty and aligned")
    if any(
        (
            isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1)
            for score in scores
        )
    ):
        raise RewardContractError("scores must be binary integers")
    numeric_weights = [float(weight) for weight in weights]
    if any((not isfinite(weight) or weight == 0.0 for weight in numeric_weights)):
        raise RewardContractError("weights must be finite and non-zero")
    denominator = sum((weight for weight in numeric_weights if weight > 0.0))
    if denominator <= 0.0:
        raise RewardContractError("at least one positive rubric weight is required")
    return (
        sum(
            (
                score * weight
                for score, weight in zip(scores, numeric_weights, strict=True)
            )
        )
        / denominator
    )
