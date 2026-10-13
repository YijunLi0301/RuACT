from __future__ import annotations
from dataclasses import dataclass


class RewardContractError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedScores:
    scores: list[float]
    fallback: str


def parse_batched_scores(
    raw_scores, num_rubrics: int, *, score_scale: float = 2.0
) -> ParsedScores:
    if raw_scores is None:
        return ParsedScores([0.5] * num_rubrics, "neutral")
    scores: list[float | None] = [None] * num_rubrics
    seen: set[int] = set()
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id"))
            raw = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < num_rubrics:
            scores[index] = max(0.0, min(1.0, raw / score_scale))
            seen.add(index)
    if not seen:
        return ParsedScores([0.5] * num_rubrics, "neutral")
    if len(seen) == num_rubrics:
        return ParsedScores([float(value) for value in scores], "none")
    valid = [float(value) for value in scores if value is not None]
    fill = sum(valid) / len(valid) if valid else 0.5
    return ParsedScores(
        [fill if value is None else float(value) for value in scores], "partial_mean"
    )


def rubric_passed(weight: float, judge_score: float) -> bool:
    if weight > 0:
        return judge_score > 0
    if weight < 0:
        return abs(judge_score) < 1e-12
    raise RewardContractError("zero-weight rubric has no pass semantics")


def aggregate_weighted_reward(scores: list[float], weights: list[float]) -> float:
    if len(scores) != len(weights):
        raise RewardContractError("score/weight length mismatch")
    denominator = sum((weight for weight in weights if weight > 0))
    return sum((score * weight for score, weight in zip(scores, weights))) / max(
        denominator, 1.0
    )


def verify_fallback_rates(
    counts: dict[str, int], max_neutral: float, max_partial: float
) -> None:
    if not 0 <= max_neutral <= 1 or not 0 <= max_partial <= 1:
        raise RewardContractError("fallback bounds must be in [0,1]")
    total = counts.get("total", 0)
    if total <= 0:
        raise RewardContractError("fallback counters require positive total")
    neutral = counts.get("neutral", 0) / total
    partial = counts.get("partial_mean", 0) / total
    if neutral > max_neutral:
        raise RewardContractError("neutral fallback fraction exceeds contract")
    if partial > max_partial:
        raise RewardContractError("partial_mean fallback fraction exceeds contract")
