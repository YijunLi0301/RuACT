from __future__ import annotations
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 7
REWARD_MODE = "parent_preserved_multi_child_50_50"
_BLEND_LAMBDA_RAW = float(os.environ.get("DISTILLATION_POSITIVE_BLEND_LAMBDA", "0.5"))
if not 0.0 < _BLEND_LAMBDA_RAW <= 1.0:
    raise SystemExit(
        f"DISTILLATION_POSITIVE_BLEND_LAMBDA must be in (0,1], got {_BLEND_LAMBDA_RAW!r}"
    )
BLEND_LAMBDA = _BLEND_LAMBDA_RAW


class PositiveRefinementProtocolError(ValueError):
    pass


def _cfg_get(config: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else default


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PositiveRefinementProtocolError(f"{field} must be a non-empty string")
    return value.strip()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parent_fingerprint(
    rubric_id: str, title: str, description: str, weight: float
) -> str:
    numeric_weight = float(weight)
    if not math.isfinite(numeric_weight) or numeric_weight == 0.0:
        raise PositiveRefinementProtocolError(
            "parent weight must be finite and non-zero"
        )
    return _stable_hash(
        {
            "rubric_id": _required_text(rubric_id, "rubric_id"),
            "title": _required_text(title, "title"),
            "description": _required_text(description, "description"),
            "weight": numeric_weight,
        }
    )


def ordered_response_set_hash(members: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash(
        [
            {
                "id": _required_text(item.get("id"), "response.id"),
                "text": str(item.get("text") or ""),
            }
            for item in members
        ]
    )


def seed_bundle_hash(selected_parents: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash(
        [
            {
                "parent_rubric_id": _required_text(
                    parent.get("rubric_id"), "parent.rubric_id"
                ),
                "parent_fingerprint": _required_text(
                    parent.get("parent_fingerprint"), "parent.parent_fingerprint"
                ),
                "candidates": list(parent.get("seed_candidates") or ()),
            }
            for parent in selected_parents
        ]
    )


def refinement_transaction_id(
    *,
    run_id: str,
    global_step: int,
    group_id: str,
    query_id: str,
    query_version: int,
    response_set_hash: str,
    parent_fingerprints: Sequence[str],
    seed_bundle_hash: str,
    blend_lambda: float,
) -> str:
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
        or isinstance(query_version, bool)
        or (not isinstance(query_version, int))
        or (query_version < 0)
    ):
        raise PositiveRefinementProtocolError(
            "transaction step and query version must be non-negative integers"
        )
    if not 1 <= len(parent_fingerprints) <= 3:
        raise PositiveRefinementProtocolError(
            "transaction requires one to three parent fingerprints"
        )
    numeric_lambda = float(blend_lambda)
    if not math.isfinite(numeric_lambda) or numeric_lambda != BLEND_LAMBDA:
        raise PositiveRefinementProtocolError(
            f"parent-preserved refinement requires blend_lambda={BLEND_LAMBDA:g}"
        )
    return _stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": _required_text(run_id, "run_id"),
            "global_step": int(global_step),
            "group_id": _required_text(group_id, "group_id"),
            "query_id": _required_text(query_id, "query_id"),
            "query_version": int(query_version),
            "response_set_hash": _required_text(response_set_hash, "response_set_hash"),
            "parent_fingerprints": [
                _required_text(value, "parent_fingerprint")
                for value in parent_fingerprints
            ],
            "seed_bundle_hash": _required_text(seed_bundle_hash, "seed_bundle_hash"),
            "blend_lambda": numeric_lambda,
        }
    )


@dataclass(frozen=True)
class PositiveRefinementConfig:
    enabled: bool = False
    mode: str = "active"
    state_dir: Path | None = None
    seed_library_path: Path | None = None
    seed_candidate_cap: int = 5
    expected_group_size: int = 8
    max_queries_per_step: int = 32
    max_parents_per_query: int = 3
    min_candidate_count: int = 5
    max_candidate_count: int = 5
    min_children_per_parent: int = 1
    max_children_per_parent: int = 3
    zero_std_epsilon: float = 1e-08
    blend_lambda: float = BLEND_LAMBDA
    include_negative_weight: bool = False
    result_poll_seconds: float = 1.0
    result_wait_timeout_seconds: float = 7200.0
    sidecar_heartbeat_timeout_seconds: float = 30.0

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any] | Any | None
    ) -> PositiveRefinementConfig:
        if config is None:
            return cls()
        raw_state_dir = _cfg_get(config, "state_dir", None)
        raw_seed_library_path = _cfg_get(config, "seed_library_path", None)
        value = cls(
            enabled=bool(_cfg_get(config, "enabled", False)),
            mode=str(_cfg_get(config, "mode", "active")),
            state_dir=Path(str(raw_state_dir)) if raw_state_dir else None,
            seed_library_path=(
                Path(str(raw_seed_library_path)) if raw_seed_library_path else None
            ),
            seed_candidate_cap=int(_cfg_get(config, "seed_candidate_cap", 5)),
            expected_group_size=int(_cfg_get(config, "expected_group_size", 8)),
            max_queries_per_step=int(_cfg_get(config, "max_queries_per_step", 32)),
            max_parents_per_query=int(_cfg_get(config, "max_parents_per_query", 3)),
            min_candidate_count=int(_cfg_get(config, "min_candidate_count", 5)),
            max_candidate_count=int(_cfg_get(config, "max_candidate_count", 5)),
            min_children_per_parent=int(_cfg_get(config, "min_children_per_parent", 1)),
            max_children_per_parent=int(_cfg_get(config, "max_children_per_parent", 3)),
            zero_std_epsilon=float(_cfg_get(config, "zero_std_epsilon", 1e-08)),
            blend_lambda=float(_cfg_get(config, "blend_lambda", BLEND_LAMBDA)),
            include_negative_weight=bool(
                _cfg_get(config, "include_negative_weight", False)
            ),
            result_poll_seconds=float(_cfg_get(config, "result_poll_seconds", 1.0)),
            result_wait_timeout_seconds=float(
                _cfg_get(config, "result_wait_timeout_seconds", 7200.0)
            ),
            sidecar_heartbeat_timeout_seconds=float(
                _cfg_get(config, "sidecar_heartbeat_timeout_seconds", 30.0)
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.mode not in {"active", "shadow"}:
            raise PositiveRefinementProtocolError(
                "positive refinement mode must be active or shadow"
            )
        if self.enabled and self.state_dir is None:
            raise PositiveRefinementProtocolError(
                "enabled positive refinement requires state_dir"
            )
        for name, value in (
            ("expected_group_size", self.expected_group_size),
            ("max_queries_per_step", self.max_queries_per_step),
            ("max_parents_per_query", self.max_parents_per_query),
            ("min_candidate_count", self.min_candidate_count),
            ("max_candidate_count", self.max_candidate_count),
            ("min_children_per_parent", self.min_children_per_parent),
            ("max_children_per_parent", self.max_children_per_parent),
            ("seed_candidate_cap", self.seed_candidate_cap),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PositiveRefinementProtocolError(
                    f"{name} must be a positive integer"
                )
        if self.expected_group_size != 8:
            raise PositiveRefinementProtocolError(
                "the refinement protocol requires K=8"
            )
        if self.max_parents_per_query > 3:
            raise PositiveRefinementProtocolError(
                "at most three saturated parents may be refined per query"
            )
        if not 1 <= self.min_candidate_count <= self.max_candidate_count:
            raise PositiveRefinementProtocolError(
                "candidate bounds must satisfy 1 <= min <= max"
            )
        if not 2 <= self.seed_candidate_cap <= 12:
            raise PositiveRefinementProtocolError(
                "seed_candidate_cap must be between two and twelve"
            )
        if not 1 <= self.min_children_per_parent <= self.max_children_per_parent <= 3:
            raise PositiveRefinementProtocolError(
                "child bounds must satisfy 1 <= min <= max <= 3"
            )
        if not math.isfinite(self.blend_lambda) or self.blend_lambda != BLEND_LAMBDA:
            raise PositiveRefinementProtocolError(
                f"parent-preserved refinement requires blend_lambda={BLEND_LAMBDA:g}"
            )
        if not math.isfinite(self.zero_std_epsilon) or self.zero_std_epsilon < 0.0:
            raise PositiveRefinementProtocolError(
                "zero_std_epsilon must be non-negative"
            )
        for name, value in (
            ("result_poll_seconds", self.result_poll_seconds),
            ("result_wait_timeout_seconds", self.result_wait_timeout_seconds),
            (
                "sidecar_heartbeat_timeout_seconds",
                self.sidecar_heartbeat_timeout_seconds,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise PositiveRefinementProtocolError(f"{name} must be positive")
        if self.result_wait_timeout_seconds <= self.sidecar_heartbeat_timeout_seconds:
            raise PositiveRefinementProtocolError(
                "result wait timeout must exceed the sidecar heartbeat timeout"
            )


def _validated_group(
    rows: Sequence[Mapping[str, Any]], expected_group_size: int
) -> list[Mapping[str, Any]]:
    if len(rows) != expected_group_size:
        raise PositiveRefinementProtocolError(
            f"complete group requires {expected_group_size} rows, got {len(rows)}"
        )
    rollout_indices: list[int] = []
    original_indices: list[int] = []
    response_ids: list[str] = []
    for row in rows:
        rollout_index = row.get("rollout_index")
        original_index = row.get("original_index")
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
            raise PositiveRefinementProtocolError("rollout_index must be an integer")
        if isinstance(original_index, bool) or not isinstance(original_index, int):
            raise PositiveRefinementProtocolError("original_index must be an integer")
        rollout_indices.append(rollout_index)
        original_indices.append(original_index)
        response_ids.append(_required_text(row.get("response_id"), "response_id"))
    if sorted(rollout_indices) != list(range(expected_group_size)):
        raise PositiveRefinementProtocolError(
            "group must contain rollout_index 0..K-1 exactly once"
        )
    if len(set(original_indices)) != expected_group_size:
        raise PositiveRefinementProtocolError("group contains duplicate original_index")
    if len(set(response_ids)) != expected_group_size:
        raise PositiveRefinementProtocolError("group contains duplicate response_id")
    return sorted(rows, key=lambda row: int(row["rollout_index"]))


def _raw_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str], list[float], list[list[int]]]:
    first = rows[0]
    rubric_ids = [str(value) for value in first.get("rubric_ids") or []]
    titles = [str(value) for value in first.get("rubric_titles") or []]
    descriptions = [str(value) for value in first.get("rubric_descriptions") or []]
    weights = [float(value) for value in first.get("rubric_weights") or []]
    if not rubric_ids or not len(rubric_ids) == len(titles) == len(descriptions) == len(
        weights
    ):
        raise PositiveRefinementProtocolError(
            "rubric metadata must be non-empty and aligned"
        )
    if any((not title.strip() for title in titles)):
        raise PositiveRefinementProtocolError("rubric titles must be non-empty")
    if len(set(rubric_ids)) != len(rubric_ids):
        raise PositiveRefinementProtocolError("rubric ids must be unique")
    if any((not math.isfinite(weight) or weight == 0.0 for weight in weights)):
        raise PositiveRefinementProtocolError(
            "rubric weights must be finite and non-zero"
        )
    matrix: list[list[int]] = []
    for row in rows:
        if [str(value) for value in row.get("rubric_ids") or []] != rubric_ids:
            raise PositiveRefinementProtocolError(
                "rubric ids changed within one rollout group"
            )
        if [str(value) for value in row.get("rubric_titles") or []] != titles:
            raise PositiveRefinementProtocolError(
                "rubric titles changed within one rollout group"
            )
        if [
            str(value) for value in row.get("rubric_descriptions") or []
        ] != descriptions:
            raise PositiveRefinementProtocolError(
                "rubric descriptions changed within one rollout group"
            )
        if [float(value) for value in row.get("rubric_weights") or []] != weights:
            raise PositiveRefinementProtocolError(
                "rubric weights changed within one rollout group"
            )
        scores = list(row.get("rubric_scores") or [])
        if len(scores) != len(rubric_ids) or any(
            (isinstance(score, bool) or score not in (0, 1) for score in scores)
        ):
            raise PositiveRefinementProtocolError(
                "rubric score vector is incomplete or non-binary"
            )
        matrix.append([int(score) for score in scores])
    return (rubric_ids, titles, descriptions, weights, matrix)


def build_refinement_jobs(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    global_step: int,
    expected_group_size: int = 8,
    max_queries: int = 32,
    max_parents_per_query: int = 3,
    min_candidate_count: int = 5,
    max_candidate_count: int = 5,
    min_children_per_parent: int = 1,
    max_children_per_parent: int = 3,
    zero_std_epsilon: float = 1e-08,
    blend_lambda: float = BLEND_LAMBDA,
    seed_candidates_by_parent: (
        Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]] | None
    ) = None,
    seed_candidate_cap: int = 5,
    blocked_query_ids: set[str] | frozenset[str] | None = None,
    include_negative_weight: bool = False,
) -> tuple[dict[str, Any], ...]:
    _required_text(run_id, "run_id")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise PositiveRefinementProtocolError(
            "global_step must be a non-negative integer"
        )
    if (
        expected_group_size != 8
        or max_queries <= 0
        or (not 1 <= max_parents_per_query <= 3)
        or (not 1 <= min_candidate_count <= max_candidate_count)
        or (not 1 <= min_children_per_parent <= max_children_per_parent <= 3)
        or (not math.isfinite(float(zero_std_epsilon)))
        or (float(zero_std_epsilon) < 0.0)
        or (not 2 <= seed_candidate_cap <= 12)
        or (not math.isfinite(float(blend_lambda)))
        or (float(blend_lambda) != BLEND_LAMBDA)
    ):
        raise PositiveRefinementProtocolError("invalid refinement job settings")
    blocked = set(blocked_query_ids or ())
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_required_text(row.get("group_id"), "group_id")].append(row)
    candidates: list[tuple[float, float, float, str, str, dict[str, Any]]] = []
    for group_id in sorted(grouped):
        group = _validated_group(grouped[group_id], expected_group_size)
        query_ids = {_required_text(row.get("query_id"), "query_id") for row in group}
        queries = {_required_text(row.get("query"), "query") for row in group}
        versions = {int(row.get("rubric_query_version", 0)) for row in group}
        if len(query_ids) != 1 or len(queries) != 1 or len(versions) != 1:
            raise PositiveRefinementProtocolError(
                "query identity or registry version changed within one group"
            )
        query_id = next(iter(query_ids))
        if query_id in blocked:
            continue
        rubric_ids, titles, descriptions, weights, matrix = _raw_matrix(group)
        pass_counts = [
            sum((row[index] for row in matrix)) for index in range(len(rubric_ids))
        ]
        positive_weight_total = sum((weight for weight in weights if weight > 0.0))
        if positive_weight_total <= 0.0:
            raise PositiveRefinementProtocolError(
                "group must contain at least one positive rubric"
            )
        saturated_positive_weight = sum(
            (
                weights[index]
                for index in range(len(rubric_ids))
                if weights[index] > 0.0
                and pass_counts[index]
                == (
                    0
                    if os.environ.get("RUACT_CRITIC_SELECTION") == "unreachable"
                    else expected_group_size
                )
            )
        )
        saturated_positive_fraction = (
            saturated_positive_weight / positive_weight_total
            if positive_weight_total > 0.0
            else 0.0
        )
        include_negative = bool(include_negative_weight)
        saturated_negative_weight = (
            sum(
                (
                    -weights[index]
                    for index in range(len(rubric_ids))
                    if weights[index] < 0.0
                    and pass_counts[index]
                    == (
                        0
                        if os.environ.get("RUACT_CRITIC_SELECTION") == "unreachable"
                        else expected_group_size
                    )
                )
            )
            if include_negative
            else 0.0
        )
        saturated_weight_fraction = (
            (saturated_positive_weight + saturated_negative_weight)
            / positive_weight_total
            if positive_weight_total > 0.0
            else 0.0
        )
        saturated_negative_count = (
            sum(
                (
                    1
                    for index in range(len(rubric_ids))
                    if weights[index] < 0.0
                    and pass_counts[index]
                    == (
                        0
                        if os.environ.get("RUACT_CRITIC_SELECTION") == "unreachable"
                        else expected_group_size
                    )
                )
            )
            if include_negative
            else 0
        )
        group_rewards = [
            sum((score * weight for score, weight in zip(row, weights)))
            / positive_weight_total
            for row in matrix
        ]
        group_mean = sum(group_rewards) / len(group_rewards)
        group_std = math.sqrt(
            sum(((value - group_mean) ** 2 for value in group_rewards))
            / len(group_rewards)
        )
        zero_std = group_std <= float(zero_std_epsilon)
        eligible = [
            {
                "rubric_index": index,
                "rubric_id": rubric_ids[index],
                "title": titles[index],
                "description": descriptions[index],
                "weight": weights[index],
                "pass_count": pass_counts[index],
                "parent_fingerprint": parent_fingerprint(
                    rubric_ids[index],
                    titles[index],
                    descriptions[index],
                    weights[index],
                ),
                **(
                    {"polarity": "negative" if weights[index] < 0.0 else "positive"}
                    if include_negative
                    else {}
                ),
            }
            for index in range(len(rubric_ids))
            if (weights[index] > 0.0 or (include_negative and weights[index] < 0.0))
            and pass_counts[index]
            == (
                0
                if os.environ.get("RUACT_CRITIC_SELECTION") == "unreachable"
                else expected_group_size
            )
        ]
        eligible.sort(
            key=lambda item: (
                -abs(float(item["weight"])),
                int(item["rubric_index"]),
                str(item["rubric_id"]),
            )
        )
        selected_parents = eligible[:max_parents_per_query]
        if not selected_parents:
            continue
        seed_lookup = seed_candidates_by_parent or {}
        for parent in selected_parents:
            lookup_key = (
                query_id,
                str(parent["rubric_id"]),
                str(parent["parent_fingerprint"]),
            )
            raw_seed_candidates = list(seed_lookup.get(lookup_key, ()))[
                :seed_candidate_cap
            ]
            seed_candidates: list[dict[str, Any]] = []
            seen_seed_points: set[str] = set()
            for raw_seed in raw_seed_candidates:
                title = _required_text(raw_seed.get("title"), "seed_candidate.title")
                completion_point = _required_text(
                    raw_seed.get("completion_point"), "seed_candidate.completion_point"
                )
                canonical_point = " ".join(completion_point.casefold().split())
                if canonical_point in seen_seed_points:
                    continue
                seen_seed_points.add(canonical_point)
                seed_candidates.append(
                    {
                        "title": title,
                        "completion_point": completion_point,
                        "source_job_id": _required_text(
                            raw_seed.get("source_job_id"),
                            "seed_candidate.source_job_id",
                        ),
                        "source_step": int(raw_seed.get("source_step", 0)),
                        "source_pass_count": int(raw_seed.get("source_pass_count", -1)),
                        **(
                            {"polarity": str(raw_seed["polarity"])}
                            if raw_seed.get("polarity") in {"positive", "negative"}
                            else {}
                        ),
                    }
                )
            parent["seed_candidates"] = seed_candidates
        members = [
            {
                "id": str(row["response_id"]),
                "rollout_index": int(row["rollout_index"]),
                "original_index": int(row["original_index"]),
                "text": str(row.get("response_text") or ""),
            }
            for row in group
        ]
        response_set_hash = ordered_response_set_hash(members)
        seed_hash = seed_bundle_hash(selected_parents)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "global_step": global_step,
            "group_id": group_id,
            "query_id": query_id,
            "query_version": next(iter(versions)),
            "response_set_hash": response_set_hash,
            "parent_fingerprints": [
                item["parent_fingerprint"] for item in selected_parents
            ],
            "seed_bundle_hash": seed_hash,
            "blend_lambda": float(blend_lambda),
        }
        job_id = refinement_transaction_id(
            run_id=run_id,
            global_step=global_step,
            group_id=group_id,
            query_id=query_id,
            query_version=next(iter(versions)),
            response_set_hash=response_set_hash,
            parent_fingerprints=[
                str(item["parent_fingerprint"]) for item in selected_parents
            ],
            seed_bundle_hash=seed_hash,
            blend_lambda=float(blend_lambda),
        )
        job = {
            **identity,
            "job_id": job_id,
            "transaction_id": job_id,
            "query": next(iter(queries)),
            "group_members": members,
            "selected_parents": selected_parents,
            "positive_saturated_count": len(eligible) - saturated_negative_count,
            "saturated_positive_weight_fraction": float(saturated_positive_fraction),
            **(
                {
                    "include_negative_weight": True,
                    "negative_saturated_count": saturated_negative_count,
                    "saturated_weight_fraction": float(saturated_weight_fraction),
                }
                if include_negative
                else {}
            ),
            "min_candidate_count": min_candidate_count,
            "max_candidate_count": max_candidate_count,
            "min_children_per_parent": min_children_per_parent,
            "max_children_per_parent": max_children_per_parent,
            "blend_lambda": float(blend_lambda),
            "gate_mode": (
                "saturated_signed_weight_fraction"
                if include_negative
                else "saturated_positive_weight_fraction"
            ),
            "generator_input_mode": "query_and_parent_only",
            "reward_mode": REWARD_MODE,
            "child_activation_mode": "all_meta_valid_fixed_weights",
            "rollout_pass_rate_filtering": False,
            "group_reward_std": float(group_std),
            "zero_std_gate": bool(zero_std),
            "zero_std_epsilon": float(zero_std_epsilon),
        }
        candidates.append(
            (
                float(saturated_weight_fraction),
                max((abs(float(item["weight"])) for item in selected_parents)),
                -float(group_std),
                query_id,
                group_id,
                job,
            )
        )
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    selected: list[tuple[float, float, float, str, str, dict[str, Any]]] = []
    seen_groups: set[str] = set()
    for item in candidates:
        if item[4] in seen_groups:
            continue
        seen_groups.add(item[4])
        selected.append(item)
        if len(selected) >= max_queries:
            break
    return tuple((item[5] for item in selected))
