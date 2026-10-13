from __future__ import annotations
import asyncio
import hashlib
import json
import os
import time
from typing import Any, Mapping
from . import worker as refinement_worker
from .protocol import SCHEMA_VERSION, PositiveRefinementProtocolError
from .registry import META_CHECKS

EXECUTION_MODE = "single_pass_batched"
CANDIDATE_BUDGET = 5
MIN_EFFECTIVE_CANDIDATES = 5
PROMPT_NAME = "refinement"


async def _prepare_parent(job, parent, config, *, generator_call=None):
    started = time.monotonic()
    direction, triage = ("positive", None)
    if float(parent["weight"]) < 0:
        direction, triage = await refinement_worker._triage_polarity(
            job, parent, config=config
        )
    polarity = "negative" if direction == "keep_negative" else "positive"
    seeds = [
        c
        for c in refinement_worker._validate_seed_candidates(parent)
        if c["polarity"] == polarity
    ]
    pool = seeds[:CANDIDATE_BUDGET]
    generated, generator_meta = ([], None)
    needed = CANDIDATE_BUDGET - len(pool)
    minimum_needed = max(0, MIN_EFFECTIVE_CANDIDATES - len(pool))
    if minimum_needed:
        prompt_job = {
            **job,
            "min_candidate_count": minimum_needed,
            "max_candidate_count": needed,
        }
        prompt = refinement_worker._generator_prompt(
            prompt_job, parent, polarity_direction=direction
        )
        prompt = prompt.replace(
            "The parent\nand the entire checklist split the parent's original weight budget equally.",
            "The selected checklist shares a fixed portion of the parent's original weight budget.",
        )
        if pool:
            prompt += (
                "\nExisting completion points; generate distinct additional points:\n"
                + "\n".join(("- " + c["completion_point"] for c in pool))
            )
        generated_value, generator_meta = await (
            generator_call or refinement_worker._generate_json_with_retry
        )(
            prompt,
            refinement_worker._generator_schema(minimum_needed, needed),
            config=config,
        )
        generator_meta = {
            **generator_meta,
            "prompt_name": PROMPT_NAME,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_character_count": len(prompt),
            "prompt_utf8_bytes": len(prompt.encode()),
        }
        generated = refinement_worker._validate_generated(
            generated_value,
            min_candidate_count=minimum_needed,
            max_candidate_count=needed,
            parent=parent,
            polarity=polarity,
        )
        pool = pool + generated
    pool = [
        {
            **c,
            "meta_verification": {
                "decision": "accept",
                **{check: True for check in META_CHECKS},
                "necessity": "optional",
                "semantic_family": refinement_worker._canonical_completion_point(c),
                "source": (
                    "preverified_library"
                    if c["candidate_source"] == "historical_seed"
                    else "online_generation"
                ),
            },
        }
        for c in refinement_worker._reindex_candidates(pool)
    ]
    pool, duplicates = refinement_worker._fixed_meta_valid_children(
        pool, max_children=CANDIDATE_BUDGET
    )
    call_audits = [m for m in (generator_meta, triage) if m is not None]
    audit = {
        "parent_rubric_id": str(parent["rubric_id"]),
        "generator": generator_meta,
        "generation_skipped": generator_meta is None,
        "generated_candidate_count": len(generated),
        "seed_candidate_count": len(seeds),
        "preverified_reused_count": sum(
            (c["candidate_source"] == "historical_seed" for c in pool)
        ),
        "seed_meta": None,
        "joint_meta": None,
        "semantic_meta_check_performed": False,
        "polarity_direction": direction,
        "polarity_triage": triage,
        "duplicate_rejections": duplicates,
        "fixed_child_count": len(pool),
        "candidate_budget": CANDIDATE_BUDGET,
        "topup_rounds": 0,
        "generator_call_count": len(call_audits),
        "generator_prompt_character_count": sum(
            (
                int(
                    m.get("prompt_character_count")
                    or m.get("triage_prompt_character_count")
                    or 0
                )
                for m in call_audits
            )
        ),
        "generator_prompt_utf8_bytes": sum(
            (
                int(
                    m.get("prompt_utf8_bytes") or m.get("triage_prompt_utf8_bytes") or 0
                )
                for m in call_audits
            )
        ),
        "preparation_seconds": time.monotonic() - started,
    }
    return (pool, audit)


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


async def process_single_pass_job(
    job: Mapping[str, Any],
    *,
    config: refinement_worker.WorkerConfig,
    parent_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    parents = list(job["selected_parents"])
    if len({p["rubric_id"] for p in parents}) != len(parents):
        raise PositiveRefinementProtocolError("selected parent IDs must be unique")
    per_job_limit = asyncio.Semaphore(parent_workers)

    async def prepare(parent):
        from .bounded_retry import prepare_parent

        return await prepare_parent(job, parent, config, _prepare_parent, per_job_limit)

    prepared = await asyncio.gather(*(prepare(p) for p in parents))
    flat, mapping, spans = ([], [], [])
    for parent, (pool, audit) in zip(parents, prepared):
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
    from .bounded_retry import judge_candidates

    grid, judge_audit = await judge_candidates(
        job, flat, workers=config.judge_workers, attempts=config.judge_attempts
    )
    if len(grid) != 8 or any((len(row) != len(flat) for row in grid)):
        raise PositiveRefinementProtocolError("batched Judge grid is misaligned")
    response_ids = [str(m["id"]) for m in job["group_members"]]
    decompositions, parent_audits = ([], [])
    for parent, (pool, audit), (start, end) in zip(parents, prepared, spans):
        if not pool:
            selected, selection = ([], [])
        else:
            selected, selection = refinement_worker._attach_judge_scores(
                pool,
                [row[start:end] for row in grid],
                response_ids=response_ids,
                max_children=CANDIDATE_BUDGET,
            )
        selected = selected[: int(job["max_children_per_parent"])]
        retained_ids = {c["candidate_index"] for c in selected}
        for item in selection:
            if item["retained"] and item["candidate_index"] not in retained_ids:
                item.update(retained=False, rejection_reason="child_cap")
        enough = len(selected) >= int(job["min_children_per_parent"])
        if not enough:
            for item in selection:
                if item["retained"]:
                    item.update(retained=False, rejection_reason="below_min_children")
            selected = []
        audit.update(
            status=(
                "selected"
                if enough
                else audit.get("status", "no_nonconstant_child_pattern")
            ),
            candidate_pass_counts=selection,
            selected_candidate_indices=[c["candidate_index"] for c in selected],
            selected_pass_counts=[c["pass_count"] for c in selected],
            fixed_judge_scope="query_wide",
            judge_columns=[start, end],
            judged_candidates=pool,
        )
        parent_audits.append(audit)
        if enough:
            decompositions.append(_decomposition(job, parent, selected))
    print(
        json.dumps(
            {
                "event": "refinement_job_completed",
                "job_id": job["job_id"],
                "selected_parents": len(parents),
                "effective_parents": len(decompositions),
                "skipped_parents": len(parents) - len(decompositions),
                "seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )
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
            "execution_mode": EXECUTION_MODE,
            "candidate_budget_per_parent": CANDIDATE_BUDGET,
            "judge_route": {
                "model": os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
                "primary_base": os.environ.get("HEALTHBENCH_JUDGE_API_BASE"),
                "backup_base": os.environ.get("HEALTHBENCH_JUDGE_API_BASE_BACKUP"),
                "auxiliary_enabled": os.environ.get("AUXILIARY_JUDGE_ENABLED") == "1",
            },
            "seed_policy": "exact_parent_first_five_current_response_rescore",
            "selected_parent_count": len(parents),
            "decomposition_count": len(decompositions),
            "selected_child_count": sum((len(d["children"]) for d in decompositions)),
            "parents": parent_audits,
            "parent_workers": min(parent_workers, len(parents)),
            "processing_seconds": round(time.monotonic() - started, 3),
            **{
                k: sum((int(a[k]) for a in parent_audits))
                for k in (
                    "generator_call_count",
                    "generator_prompt_character_count",
                    "generator_prompt_utf8_bytes",
                )
            },
            "seed_pre_meta_skipped_count": 0,
            "generator_model": config.model,
            "generator_enable_thinking": config.enable_thinking,
            "meta_model": None,
            "semantic_meta_check_performed": False,
            "fixed_judge_model": os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
            "fixed_judge_input_mode": "query_answer_descriptions_only",
            "fixed_judge": judge_audit,
            "judge_column_map": mapping,
            "judge_grid": grid,
            "generator_received_rollouts": False,
            "topup_enabled": False,
            "child_pass_count_range": (
                [0, 8] if os.environ.get("RUACT_CRITIC_NO_FILTER") == "1" else [2, 6]
            ),
            "child_cap_selection": "candidate_order",
            "rollout_pass_rate_filtering": False,
            "constant_children_retained": os.environ.get("RUACT_CRITIC_NO_FILTER")
            == "1",
            "constant_children_rejected": os.environ.get("RUACT_CRITIC_NO_FILTER")
            != "1",
        },
    }
