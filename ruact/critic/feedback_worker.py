from __future__ import annotations
import asyncio
import json
import os
import time
from typing import Any, Mapping
from . import worker
from .bounded_retry import prepare_parent, RefinementBudgetExhausted
from .protocol import SCHEMA_VERSION, PositiveRefinementProtocolError
from .registry import META_CHECKS

CANDIDATE_BUDGET = 5


def _decomposition(job, parent, children):
    return {
        "parent_rubric_id": str(parent["rubric_id"]),
        "parent_fingerprint": str(parent["parent_fingerprint"]),
        "parent_title": str(parent["title"]),
        "parent_description": str(parent["description"]),
        "parent_weight": float(parent["weight"]),
        "blend_lambda": float(job["blend_lambda"]),
        "children": [
            {
                "title": c["title"],
                "description": c["description"],
                "completion_point": c["completion_point"],
                "necessity": c["meta_verification"]["necessity"],
                "semantic_family": c["meta_verification"]["semantic_family"],
                "meta_verification": dict(c["meta_verification"]),
                "judge_scores": c["judge_scores"],
                "pass_count": c["pass_count"],
                "generator_candidate_index": c["candidate_index"],
                "candidate_source": c["candidate_source"],
                "source_candidate_index": c["source_candidate_index"],
                **({"polarity": c["polarity"]} if float(parent["weight"]) < 0 else {}),
                **(
                    {
                        k: c[k]
                        for k in ("source_job_id", "source_step", "source_pass_count")
                    }
                    if c["candidate_source"] == "historical_seed"
                    else {}
                ),
            }
            for c in children
        ],
    }


def _meta_schema(count):
    properties = {
        "candidate_index": {"type": "integer"},
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        **{check: {"type": "boolean"} for check in META_CHECKS},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(properties),
                    "properties": properties,
                },
            }
        },
    }


def _validate_meta(value, candidates):
    results = value.get("results") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"results"}
        or not isinstance(results, list)
        or len(results) != len(candidates)
    ):
        raise PositiveRefinementProtocolError(
            "Semantic Meta Judge returned a misaligned batch"
        )
    expected = {c["candidate_index"] for c in candidates}
    by_index = {}
    for verdict in results:
        if not isinstance(verdict, Mapping) or set(verdict) != {
            "candidate_index",
            "decision",
            *META_CHECKS,
        }:
            raise PositiveRefinementProtocolError(
                "Semantic Meta Judge verdict fields are invalid"
            )
        index = verdict["candidate_index"]
        if type(index) is not int or index not in expected or index in by_index:
            raise PositiveRefinementProtocolError(
                "Semantic Meta Judge candidate indices are misaligned"
            )
        if verdict["decision"] not in {"accept", "reject"} or any(
            type(verdict[check]) is not bool for check in META_CHECKS
        ):
            raise PositiveRefinementProtocolError(
                "Semantic Meta Judge requires a decision and boolean checks"
            )
        if (verdict["decision"] == "accept") != all(
            verdict[check] for check in META_CHECKS
        ):
            raise PositiveRefinementProtocolError(
                "Semantic Meta Judge decision contradicts its checks"
            )
        by_index[index] = {**verdict, "executed": True}
    return by_index


async def _semantic_validate(job, parent, pool, config, generate):
    if not config.semantic_meta_enabled:
        return [
            {
                **c,
                "meta_verification": {
                    "executed": False,
                    "decision": "disabled",
                    **{check: None for check in META_CHECKS},
                    "necessity": "optional",
                    "semantic_family": worker._canonical_completion_point(c),
                },
            }
            for c in pool
        ], None
    if not pool:
        return [], None
    views = [
        {k: c[k] for k in ("candidate_index", "title", "completion_point", "polarity")}
        for c in pool
    ]
    prompt = (
        "You are an independent semantic Meta Judge for rubric refinement. Evaluate every candidate using only the query and parent. No rollout answers or statistics are available.\n"
        f"User query:\n{job['query']}\nParent title: {parent['title']}\nParent description: {parent['description']}\nCandidates:\n{json.dumps(views, ensure_ascii=False)}\n"
        "Check factually_valid (factually defensible; for a negative-polarity candidate, correctly identifies a real error), preserves_parent (same objective and polarity direction), binary_judgeable (decidable from a query and one answer), relevant_to_query (helps evaluate the requested answer), and substantive (affects correctness, not cosmetic style). Accept exactly when all five checks are true. Do not add any other gate. Return only JSON with the key results containing exactly one verdict for every provided candidate_index, the decision accept or reject, and all five boolean checks."
    )
    value, metadata = await generate(
        prompt, _meta_schema(len(pool)), config=config.for_meta()
    )
    verdicts = _validate_meta(value, pool)
    accepted = [
        {
            **c,
            "meta_verification": {
                **verdicts[c["candidate_index"]],
                "necessity": "optional",
                "semantic_family": worker._canonical_completion_point(c),
            },
        }
        for c in pool
        if verdicts[c["candidate_index"]]["decision"] == "accept"
    ]
    return accepted, {
        "executed": True,
        "worker": metadata,
        "decisions": list(verdicts.values()),
    }


async def _prepare_parent(
    job,
    parent,
    config,
    *,
    generator_call=None,
    previous=(),
    rejection_feedback=(),
    direction=None,
):
    generate = generator_call or worker._generate_json
    triage = None
    if direction is None:
        direction = "positive"
        if float(parent["weight"]) < 0:
            value, triage = await generate(
                worker._polarity_triage_prompt(job, parent),
                worker._POLARITY_TRIAGE_SCHEMA,
                config=config.for_meta(),
            )
            direction = value.get("direction")
            if direction not in {"convert_to_positive", "keep_negative"}:
                raise PositiveRefinementProtocolError(
                    "Invalid polarity triage direction"
                )
    polarity = "negative" if direction == "keep_negative" else "positive"
    seeds = (
        []
        if previous
        else [
            c
            for c in worker._validate_seed_candidates(parent)
            if c["polarity"] == polarity
        ][:CANDIDATE_BUDGET]
    )
    generated, generator_meta = [], None
    needed = CANDIDATE_BUDGET - len(seeds)
    if needed:
        prompt = worker._generator_prompt(
            {**job, "min_candidate_count": needed, "max_candidate_count": needed},
            parent,
            polarity_direction=direction,
            rejection_feedback=rejection_feedback,
        )
        existing = [*previous, *seeds]
        if existing:
            prompt += (
                "\nExisting completion points; generate distinct additional points:\n"
                + "\n".join("- " + c["completion_point"] for c in existing)
            )
        value, generator_meta = await generate(
            prompt, worker._generator_schema(needed, needed), config=config
        )
        generated = worker._validate_generated(
            value,
            min_candidate_count=needed,
            max_candidate_count=needed,
            parent=parent,
            polarity=polarity,
        )
    offset = max((c["candidate_index"] for c in previous), default=-1) + 1
    raw_pool = [
        {**c, "candidate_index": i + offset} for i, c in enumerate([*seeds, *generated])
    ]
    seen = {worker._canonical_completion_point(c) for c in previous}
    pool, duplicates = [], []
    for candidate in raw_pool:
        canonical = worker._canonical_completion_point(candidate)
        if canonical in seen:
            duplicates.append(
                {
                    "candidate_index": candidate["candidate_index"],
                    "candidate_source": candidate["candidate_source"],
                    "reason": "canonical_completion_point_duplicate",
                }
            )
        else:
            seen.add(canonical)
            pool.append(candidate)
    pool, meta = await _semantic_validate(job, parent, pool, config, generate)
    return pool, {
        "parent_rubric_id": str(parent["rubric_id"]),
        "generator": generator_meta,
        "generation_skipped": generator_meta is None,
        "generated_candidate_count": len(generated),
        "seed_candidate_count": len(seeds),
        "preverified_reused_count": sum(
            c["candidate_source"] == "historical_seed" for c in pool
        ),
        "seed_meta": None,
        "joint_meta": meta,
        "semantic_meta_check_performed": meta is not None,
        "polarity_direction": direction,
        "polarity_triage": triage,
        "duplicate_rejections": duplicates,
        "fixed_child_count": len(pool),
        "candidate_budget": CANDIDATE_BUDGET,
        "topup_rounds": 0,
        "all_candidates": raw_pool,
    }


def _select(job, pool, grid):
    selected, selection = worker._attach_judge_scores(
        pool,
        grid,
        response_ids=[str(m["id"]) for m in job["group_members"]],
        max_children=len(pool),
    )
    selected = selected[: int(job["max_children_per_parent"])]
    retained = {c["candidate_index"] for c in selected}
    for entry in selection:
        if entry["retained"] and entry["candidate_index"] not in retained:
            entry.update(retained=False, rejection_reason="child_cap")
    return selected, selection


async def process_feedback_job(
    job, *, config, parent_workers, execution_mode="feedback_batched"
):
    from . import bounded_retry

    if type(config.topup_round_limit) is not int or config.topup_round_limit not in (
        0,
        1,
    ):
        raise PositiveRefinementProtocolError("topup_round_limit must be zero or one")
    if parent_workers < 1:
        raise PositiveRefinementProtocolError("parent_workers must be positive")
    started = time.monotonic()
    parents = list(job["selected_parents"])
    if len({p["rubric_id"] for p in parents}) != len(parents):
        raise PositiveRefinementProtocolError("selected parent IDs must be unique")
    limit = asyncio.Semaphore(parent_workers)
    prepared = await asyncio.gather(
        *(prepare_parent(job, p, config, _prepare_parent, limit) for p in parents)
    )
    flat, mapping, spans = [], [], []
    for parent, (pool, _) in zip(parents, prepared):
        start = len(flat)
        for c in pool:
            mapping.append(
                {
                    "column": len(flat),
                    "parent_rubric_id": parent["rubric_id"],
                    "candidate_index": c["candidate_index"],
                }
            )
            flat.append(c)
        spans.append((start, len(flat)))
    grid, judge_audit = await bounded_retry.judge_candidates(
        job, flat, workers=config.judge_workers, attempts=config.judge_attempts
    )
    if len(grid) != 8 or any(len(row) != len(flat) for row in grid):
        raise PositiveRefinementProtocolError("batched Judge grid is misaligned")
    topup_enabled = (
        execution_mode != "single_pass_batched"
        and config.topup_enabled
        and config.topup_round_limit == 1
    )

    async def finish(parent, prepared_parent, span):
        pool, audit = prepared_parent
        start, end = span
        selected, selection = _select(job, pool, [row[start:end] for row in grid])
        minimum = int(job["min_children_per_parent"])
        audit.update(
            fixed_judge_scope="query_wide",
            judge_columns=[start, end],
            judged_candidates=list(pool),
        )
        if len(selected) < minimum and topup_enabled:
            audit["topup_rounds"] = 1
            audit["topup_reason"] = "below_min_children"
            reasons = {
                entry["candidate_index"]: entry["rejection_reason"]
                for entry in selection
                if not entry["retained"]
            }
            meta = audit.get("joint_meta") or {}
            reasons.update(
                {
                    v["candidate_index"]: "semantic_meta_rejection"
                    for v in meta.get("decisions", [])
                    if v["decision"] == "reject"
                }
            )
            reasons.update(
                {
                    v["candidate_index"]: v["reason"]
                    for v in audit["duplicate_rejections"]
                }
            )
            previous = audit["all_candidates"]
            feedback = [
                f"{c['completion_point']} (rejected)"
                for c in previous
                if c["candidate_index"] in reasons
            ]

            async def prepare_topup(job, parent, config, *, generator_call):
                return await _prepare_parent(
                    job,
                    parent,
                    config,
                    generator_call=generator_call,
                    previous=previous,
                    rejection_feedback=feedback,
                    direction=audit["polarity_direction"],
                )

            try:
                extra, topup = await prepare_parent(
                    job, parent, config, prepare_topup, limit
                )
                topup["status"] = "completed"
            except RefinementBudgetExhausted as error:
                extra, topup = [], {
                    **error.audit,
                    "status": "generation_failed",
                    "error": str(error),
                }
            audit["topup"] = topup
            audit["generated_candidate_count"] += topup.get(
                "generated_candidate_count", 0
            )
            if extra:
                topup["judge_attempted"] = True
                try:
                    extra_grid, extra_judge = await bounded_retry.judge_candidates(
                        job,
                        extra,
                        workers=config.judge_workers,
                        attempts=config.judge_attempts,
                    )
                    if len(extra_grid) != 8 or any(
                        len(row) != len(extra) for row in extra_grid
                    ):
                        raise PositiveRefinementProtocolError(
                            "topup Judge grid is misaligned"
                        )
                    extra_selected, extra_selection = _select(job, extra, extra_grid)
                except (
                    PositiveRefinementProtocolError,
                    RefinementBudgetExhausted,
                    TimeoutError,
                    ConnectionError,
                    OSError,
                ) as error:
                    topup.update(
                        status="judge_failed",
                        judge_scores_available=False,
                        judge_error={
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                else:
                    selected = [*selected, *extra_selected][
                        : int(job["max_children_per_parent"])
                    ]
                    selection.extend(extra_selection)
                    audit["judged_candidates"].extend(extra)
                    topup.update(
                        fixed_judge=extra_judge,
                        judge_grid=extra_grid,
                        judged_candidates=extra,
                        candidate_pass_counts=extra_selection,
                        judge_column_map=[c["candidate_index"] for c in extra],
                        judge_scores_available=True,
                    )
            audit["semantic_meta_check_performed"] |= topup.get(
                "semantic_meta_check_performed", False
            )
            for key in (
                "generator_call_count",
                "semantic_meta_call_count",
                "polarity_triage_call_count",
                "model_call_count",
                "generator_prompt_character_count",
                "generator_prompt_utf8_bytes",
            ):
                audit[key] += topup.get(key, 0)
        else:
            audit["topup_reason"] = (
                "enough_children" if len(selected) >= minimum else "disabled"
            )
        retained = {c["candidate_index"] for c in selected}
        for entry in selection:
            if entry["retained"] and entry["candidate_index"] not in retained:
                entry.update(retained=False, rejection_reason="child_cap")
        audit.update(
            status="selected" if selected else "no_nonconstant_child_pattern",
            minimum_satisfied=len(selected) >= minimum,
            candidate_pass_counts=selection,
            selected_candidate_indices=[c["candidate_index"] for c in selected],
            selected_pass_counts=[c["pass_count"] for c in selected],
        )
        return (_decomposition(job, parent, selected) if selected else None), audit

    finished = await asyncio.gather(
        *(finish(p, item, span) for p, item, span in zip(parents, prepared, spans))
    )
    decompositions = [d for d, _ in finished if d is not None]
    audits = [a for _, a in finished]
    no_filter = os.environ.get("RUACT_CRITIC_NO_FILTER") == "1"
    judge_rounds = [
        judge_audit,
        *[
            a["topup"]["fixed_judge"]
            for a in audits
            if "fixed_judge" in a.get("topup", {})
        ],
    ]
    failed_judge_rounds = sum(
        a.get("topup", {}).get("status") == "judge_failed" for a in audits
    )
    combined_judge = {
        **judge_audit,
        "rounds": judge_rounds,
        "response_task_attempts": (
            None
            if failed_judge_rounds
            else sum(item.get("response_task_attempts", 0) for item in judge_rounds)
        ),
        "known_response_task_attempts": sum(
            item.get("response_task_attempts", 0) for item in judge_rounds
        ),
        "failed_round_count": failed_judge_rounds,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "job_id": str(job["job_id"]),
        "transaction_id": str(job["transaction_id"]),
        "query_id": str(job["query_id"]),
        "source_query_version": int(job["query_version"]),
        "response_set_hash": str(job["response_set_hash"]),
        "seed_bundle_hash": str(job["seed_bundle_hash"]),
        "decompositions": decompositions,
        "audit": {
            "execution_mode": execution_mode,
            "candidate_budget_per_parent": CANDIDATE_BUDGET,
            "seed_policy": "exact_parent_first_five_current_response_rescore",
            "selected_parent_count": len(parents),
            "decomposition_count": len(decompositions),
            "selected_child_count": sum(len(d["children"]) for d in decompositions),
            "parents": audits,
            "parent_workers": min(parent_workers, len(parents)),
            "processing_seconds": round(time.monotonic() - started, 3),
            **{
                k: sum(a[k] for a in audits)
                for k in (
                    "generator_call_count",
                    "semantic_meta_call_count",
                    "polarity_triage_call_count",
                    "model_call_count",
                    "generator_prompt_character_count",
                    "generator_prompt_utf8_bytes",
                )
            },
            "seed_pre_meta_skipped_count": 0,
            "generator_model": config.model,
            "generator_enable_thinking": config.enable_thinking,
            "meta_model": (
                (config.meta_model or config.model)
                if config.semantic_meta_enabled
                else None
            ),
            "semantic_meta_enabled": config.semantic_meta_enabled,
            "semantic_meta_check_performed": any(
                a["semantic_meta_check_performed"] for a in audits
            ),
            "fixed_judge_model": os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
            "fixed_judge_input_mode": "query_answer_descriptions_only",
            "fixed_judge": combined_judge,
            "judge_column_map": mapping,
            "judge_grid": grid,
            "generator_received_rollouts": False,
            "topup_enabled": topup_enabled,
            "topup_round_limit": config.topup_round_limit if topup_enabled else 0,
            "topup_rounds": sum(a["topup_rounds"] for a in audits),
            "child_pass_count_range": [0, 8] if no_filter else [2, 6],
            "child_cap_selection": "candidate_order",
            "rollout_pass_rate_filtering": not no_filter,
            "constant_children_retained": no_filter,
            "constant_children_rejected": not no_filter,
        },
    }
