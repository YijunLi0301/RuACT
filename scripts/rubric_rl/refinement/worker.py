from __future__ import annotations
import asyncio
import hashlib
import os
import time
from typing import Any, Mapping, Sequence
from rubric_rl.refinement.transport import (
    WorkerConfig,
    _judge_candidates_forever as _judge_candidates_impl,
    _generate_json_with_retry as _generate_json,
    configure_generator_concurrency,
    configure_judge_concurrency,
)
from .protocol import (
    BLEND_LAMBDA,
    REWARD_MODE,
    SCHEMA_VERSION,
    PositiveRefinementProtocolError,
    seed_bundle_hash,
)
from .registry import META_CHECKS

GENERATOR_CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "completion_point"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "completion_point": {"type": "string", "minLength": 1},
    },
}
_GENERATOR_FEW_SHOT = 'Few-shot demonstrations (format and specificity only; do not copy their medical content):\n\nExample 1\nQuery: A patient has acute chest pain. What is the initial evaluation?\nParent rubric: Identifies time-sensitive evaluation for life-threatening cardiac causes.\nValid output:\n{"candidates":[\n  {"title":"Serial ischemia testing","completion_point":"States that serial ECG and troponin assessment are needed when acute coronary syndrome remains a concern."},\n  {"title":"Immediate instability check","completion_point":"Checks for hemodynamic instability before proceeding with routine evaluation."}\n]}\n\nExample 2\nQuery: How should suspected bacterial meningitis be managed initially?\nParent rubric: Recommends prompt empiric treatment for suspected bacterial meningitis.\nValid output:\n{"candidates":[\n  {"title":"Do not delay treatment","completion_point":"States that empiric antibiotics should not be delayed when lumbar puncture or imaging would postpone treatment."},\n  {"title":"Blood cultures first when feasible","completion_point":"Requests blood cultures before antibiotics when this can be done without delaying treatment."}\n]}\n\nExample 3\nQuery: What discharge advice is appropriate after a new medication is started?\nParent rubric: Provides safe, actionable follow-up advice.\nValid output:\n{"candidates":[\n  {"title":"Specific warning signs","completion_point":"Names concrete symptoms that require urgent reassessment rather than giving only generic follow-up advice."},\n  {"title":"Defined follow-up timing","completion_point":"Specifies when and with whom the patient should follow up."}\n]}\n\n'


def _generator_schema(min_count: int, max_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": min_count,
                "maxItems": max_count,
                "items": GENERATOR_CANDIDATE_SCHEMA,
            }
        },
    }


_parent_semaphore: asyncio.Semaphore | None = None
_preverified_library: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]] = {}


def configure_preverified_library(
    path: str | os.PathLike[str], *, limit: int = 12
) -> dict[str, int | str]:
    from .seed_library import PositiveRefinementSeedLibrary

    library = PositiveRefinementSeedLibrary(path)
    mapping = library.as_lookup_mapping(limit=limit)
    global _preverified_library
    _preverified_library = mapping
    return {
        "path": str(path),
        "entry_count": len(library),
        "candidate_count": sum((len(values) for values in mapping.values())),
    }


def configure_parent_concurrency(limit: int) -> None:
    global _parent_semaphore
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise PositiveRefinementProtocolError(
            "global parent concurrency must be positive"
        )
    _parent_semaphore = asyncio.Semaphore(limit)


async def _generate_json_with_retry(
    prompt: str, schema: Mapping[str, Any], *, config: WorkerConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt = 0
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    while True:
        try:
            return await _generate_json(prompt, schema, config=config)
        except Exception as error:
            attempt += 1
            delay = min(
                60.0, max(1.0, config.retry_backoff_seconds) * 2 ** min(attempt - 1, 5)
            )
            print(
                f"positive-refinement generator retry {attempt} prompt_sha256={prompt_sha256}: {type(error).__name__}: {error}; sleeping {delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


async def _judge_candidates_forever(
    job: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    attempts: int = 3,
    descriptions_only: bool = False,
) -> tuple[list[list[int]], dict[str, Any]]:
    retry = 0
    while True:
        try:
            return await asyncio.to_thread(
                _judge_candidates_impl,
                job,
                candidates,
                workers=workers,
                attempts=attempts,
                descriptions_only=descriptions_only,
            )
        except Exception as error:
            retry += 1
            delay = min(60.0, 2.0 ** min(retry - 1, 5))
            print(
                f"positive-refinement fixed-Judge retry {retry}: {type(error).__name__}: {error}; sleeping {delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


def validate_refinement_job(job: Mapping[str, Any]) -> None:
    if job.get("schema_version") != SCHEMA_VERSION:
        raise PositiveRefinementProtocolError("unsupported refinement job schema")
    for field in (
        "job_id",
        "transaction_id",
        "query_id",
        "query",
        "group_id",
        "response_set_hash",
    ):
        if not isinstance(job.get(field), str) or not str(job[field]).strip():
            raise PositiveRefinementProtocolError(f"job.{field} must be non-empty")
    if job["job_id"] != job["transaction_id"]:
        raise PositiveRefinementProtocolError(
            "job id and lightweight transaction id must match"
        )
    include_negative = bool(job.get("include_negative_weight"))
    if job.get("gate_mode") not in {
        "saturated_positive_weight_fraction",
        "saturated_signed_weight_fraction",
    }:
        raise PositiveRefinementProtocolError("unsupported refinement gate")
    if job.get("generator_input_mode") != "query_and_parent_only":
        raise PositiveRefinementProtocolError(
            "generator must receive only the query and one parent"
        )
    if job.get("reward_mode") != REWARD_MODE:
        raise PositiveRefinementProtocolError(
            "unsupported completion decomposition reward mode"
        )
    if job.get("child_activation_mode") != "all_meta_valid_fixed_weights":
        raise PositiveRefinementProtocolError(
            "children must be activated independently of rollout pass rates"
        )
    if job.get("rollout_pass_rate_filtering") is not False:
        raise PositiveRefinementProtocolError(
            "rollout pass-rate filtering must be disabled"
        )
    members = job.get("group_members")
    if not isinstance(members, list) or len(members) != 8:
        raise PositiveRefinementProtocolError(
            "refinement job must contain exactly eight responses"
        )
    rollout_indices = [int(item["rollout_index"]) for item in members]
    response_ids = [str(item["id"]) for item in members]
    if rollout_indices != list(range(8)) or len(set(response_ids)) != 8:
        raise PositiveRefinementProtocolError(
            "refinement responses must be unique and in rollout order"
        )
    parents = job.get("selected_parents")
    if not isinstance(parents, list) or not 1 <= len(parents) <= 3:
        raise PositiveRefinementProtocolError(
            "refinement job must contain one to three parents"
        )
    for parent in parents:
        parent_weight = float(parent.get("weight", 0.0))
        valid_weight = parent_weight > 0.0 or (include_negative and parent_weight < 0.0)
        if (
            not isinstance(parent, Mapping)
            or not valid_weight
            or int(parent.get("pass_count", -1))
            != (0 if os.environ.get("RUACT_CRITIC_SELECTION") == "unreachable" else 8)
        ):
            raise PositiveRefinementProtocolError(
                "every selected parent must match the configured selection and weight polarity"
            )
        for field in ("rubric_id", "title", "description", "parent_fingerprint"):
            if not isinstance(parent.get(field), str) or not str(parent[field]).strip():
                raise PositiveRefinementProtocolError(
                    f"parent.{field} must be non-empty"
                )
        seeds = parent.get("seed_candidates")
        if not isinstance(seeds, list):
            raise PositiveRefinementProtocolError(
                "parent.seed_candidates must be a list"
            )
        seen_seed_points: set[str] = set()
        for seed in seeds:
            if not isinstance(seed, Mapping):
                raise PositiveRefinementProtocolError(
                    "seed candidate must be an object"
                )
            for field in ("title", "completion_point", "source_job_id"):
                if not isinstance(seed.get(field), str) or not str(seed[field]).strip():
                    raise PositiveRefinementProtocolError(
                        f"seed.{field} must be non-empty"
                    )
            canonical = " ".join(str(seed["completion_point"]).casefold().split())
            if canonical in seen_seed_points:
                raise PositiveRefinementProtocolError(
                    "parent contains duplicate seed completion points"
                )
            seen_seed_points.add(canonical)
    if job.get("seed_bundle_hash") != seed_bundle_hash(parents):
        raise PositiveRefinementProtocolError("refinement seed bundle hash is invalid")
    min_candidate_count = int(job.get("min_candidate_count", 0))
    max_candidate_count = int(job.get("max_candidate_count", 0))
    if not 1 <= min_candidate_count <= max_candidate_count:
        raise PositiveRefinementProtocolError(
            "refinement candidate bounds must satisfy 1 <= min <= max"
        )
    child_cap = int(job.get("max_children_per_parent", 0))
    child_min = int(job.get("min_children_per_parent", 0))
    if not 1 <= child_min <= child_cap <= 3:
        raise PositiveRefinementProtocolError(
            "child bounds must satisfy 1 <= min <= max <= 3"
        )
    blend_lambda = float(job.get("blend_lambda", 0.0))
    if blend_lambda != BLEND_LAMBDA:
        raise PositiveRefinementProtocolError(
            f"parent-preserved refinement requires blend_lambda={BLEND_LAMBDA:g}"
        )
    group_std = float(job.get("group_reward_std", -1.0))
    epsilon = float(job.get("zero_std_epsilon", -1.0))
    if group_std < 0.0 or epsilon < 0.0:
        raise PositiveRefinementProtocolError(
            "zero-standard-deviation gate metadata is invalid"
        )
    if bool(job.get("zero_std_gate")) != (group_std <= epsilon):
        raise PositiveRefinementProtocolError(
            "zero-standard-deviation gate metadata is inconsistent"
        )


def _generator_prompt(
    job: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    rejection_feedback: Sequence[str] | None = None,
    polarity_direction: str = "positive",
) -> str:
    min_candidate_count = int(job["min_candidate_count"])
    max_candidate_count = int(job["max_candidate_count"])
    feedback_block = ""
    if rejection_feedback:
        points = "\n".join((f"- {point}" for point in rejection_feedback))
        feedback_block = f"Previously rejected completion points (do NOT repeat these or any semantic equivalent; generate completion points on a DIFFERENT refinement operator or dimension):\n{points}\n\n"
    polarity_block = ""
    if polarity_direction == "convert_to_positive":
        polarity_block = '\nParent polarity: this rubric penalizes answers for MISSING one positive behavior (a negation shell such as "Does not mention / Does not state / Without"). Flip the shell and decompose positively: every candidate must be a POSITIVE completion point that specifies what a strong answer should state or do to cover the missing behavior. A fixed Judge will score each candidate as satisfied(1) or not(0) from the query and one answer.\n'
    elif polarity_direction == "keep_negative":
        polarity_block = '\nParent polarity: this rubric describes an ERRONEOUS action or claim that an answer commits (for example "Recommends wrong / Incorrectly states / Suggests a contraindicated"). Decompose the error negatively: every candidate must name ONE concrete sub-error or wrong variant of the parent mistake, described in the same observable direction (the error itself, not a prohibition, and never a "Does not mention" shell). A fixed Judge will score each candidate as committed(1) or not(0) from the query and one answer.\n'
    return f"You are refining one rubric for a RaR training example.\n\n{_GENERATOR_FEW_SHOT}Now solve the target example. Follow the demonstrated JSON shape, but derive every completion point only from the target query and parent rubric.\n\nUser query:\n{job['query']}\n\nParent rubric:\nTitle: {parent['title']}\nDescription: {parent['description']}\n{polarity_block}\n{feedback_block}Generate between {min_candidate_count} and {max_candidate_count} atomic completion-point candidates. Do not use, request, infer, or mention\nany model responses or rollout statistics. The response texts are deliberately unavailable.\n\nFor every candidate:\n- Stay inside the exact objective and direction of the parent rubric.\n- Express exactly one objective, factually defensible, query-relevant detail that a stronger answer\n  should include to complete the parent requirement.\n- The point may operationalize or add detail to the parent, but must not introduce a different\n  evaluation objective.\n- Prefer distinct refinement operators when appropriate, such as mechanism, evidence, boundary\n  condition, consequence, management implication, operationalization, or a concrete example.\n- Keep the requirement atomic and binary-judgeable from the query and one answer.\n- Do not add vague demands such as being more comprehensive, detailed, clear, or accurate.\n- Do not introduce unrelated topics, style preferences, response IDs, weights, polarity labels,\n  scoring rules, or references to the parent.\n- Make the candidates semantically distinct, not paraphrases of one another.\n- Match the language of the parent rubric.\n\nReturn only a short standalone title and the single completion_point for each candidate. Accepted\ncompletion points will be used together as a fixed checklist alongside the retained parent. The parent\nand the entire checklist split the parent's original weight budget equally.\n"


def _compose_child_description(parent_description: str, completion_point: str) -> str:
    del parent_description
    return completion_point.strip()


def _validate_generated(
    value: Mapping[str, Any],
    *,
    min_candidate_count: int,
    max_candidate_count: int,
    parent: Mapping[str, Any],
    polarity: str = "positive",
) -> list[dict[str, Any]]:
    candidates = value.get("candidates")
    if (
        not isinstance(candidates, list)
        or not min_candidate_count <= len(candidates) <= max_candidate_count
    ):
        raise PositiveRefinementProtocolError(
            f"generator must return between the requested minimum and maximum candidate count; expected={min_candidate_count}..{max_candidate_count}, observed={(len(candidates) if isinstance(candidates, list) else type(candidates).__name__)}"
        )
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise PositiveRefinementProtocolError(
                "generator candidate must be an object"
            )
        title = str(raw.get("title") or "").strip()
        completion_point = str(raw.get("completion_point") or "").strip()
        if not title or not completion_point:
            raise PositiveRefinementProtocolError(
                "generator title and completion point must be non-empty"
            )
        key = (
            " ".join(title.casefold().split()),
            " ".join(completion_point.casefold().split()),
        )
        if key in seen:
            raise PositiveRefinementProtocolError(
                "generator returned duplicate refinement candidates"
            )
        seen.add(key)
        validated.append(
            {
                "candidate_index": index,
                "title": title,
                "completion_point": completion_point,
                "description": _compose_child_description(
                    str(parent["description"]), completion_point
                ),
                "polarity": polarity,
                "candidate_source": "online_generation",
                "source_candidate_index": index,
            }
        )
    return validated


def _validate_seed_candidates(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(parent.get("seed_candidates") or ()):
        title = str(raw.get("title") or "").strip()
        completion_point = str(raw.get("completion_point") or "").strip()
        source_job_id = str(raw.get("source_job_id") or "").strip()
        if not title or not completion_point or (not source_job_id):
            raise PositiveRefinementProtocolError(
                "historical seed candidate is incomplete"
            )
        raw_polarity = raw.get("polarity")
        seed_polarity = (
            str(raw_polarity)
            if raw_polarity in {"positive", "negative"}
            else "positive"
        )
        candidates.append(
            {
                "candidate_index": index,
                "title": title,
                "completion_point": completion_point,
                "description": _compose_child_description(
                    str(parent["description"]), completion_point
                ),
                "polarity": seed_polarity,
                "candidate_source": "historical_seed",
                "source_candidate_index": index,
                "source_job_id": source_job_id,
                "source_step": int(raw.get("source_step", 0)),
                "source_pass_count": int(raw.get("source_pass_count", -1)),
            }
        )
    return candidates


def _reindex_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {**dict(candidate), "candidate_index": index}
        for index, candidate in enumerate(candidates)
    ]


def _canonical_completion_point(candidate: Mapping[str, Any]) -> str:
    return " ".join(str(candidate.get("completion_point") or "").casefold().split())


_POLARITY_TRIAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["direction"],
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["convert_to_positive", "keep_negative"],
        }
    },
}


def _polarity_triage_prompt(job: Mapping[str, Any], parent: Mapping[str, Any]) -> str:
    return f"""You are classifying one negative-weight rubric (a known answer pitfall) before decomposing it. You never see any model responses or rollout statistics.\n\nUser query:\n{job['query']}\n\nPitfall rubric (negative weight; an answer is penalized when the rubric text matches it):\nTitle: {parent['title']}\nDescription: {parent['description']}\n\nClassify the pitfall into exactly one direction:\n- convert_to_positive: the rubric is a NEGATION SHELL — it penalizes answers for MISSING one single positive behavior, and its wording is only the absence of that behavior (typical markers: "Does not mention", "Does not state", "Does not provide", "Without", "Fails to mention", "Omits", "No discussion of", and similar absence-of-X phrasing). After stripping the negation shell there is exactly one positive completion point to refine further.\n- keep_negative: the rubric describes an ERRONEOUS ACTION or wrong claim the answer actively commits (typical markers: "Recommends wrong", "Incorrectly states", "Recommends against", "Suggests a contraindicated", "Administered incorrectly", "Misidentifies", and similar wrong-doing phrasing). The error itself, not an absence, is the penalized behavior.\n\nDeciding hints:\n- Ask whether the answer that gets penalized could be repaired by merely ADDING one missing behavior (shell) versus by REPLACING a stated wrong behavior with the right one (erroneous action).\n- A shell usually has a single negated verb and no wrong content of its own; an erroneous action names concrete wrong content.\n- When a rubric contains both an absence marker and concrete wrong content, prefer keep_negative (the wrong content is the actionable error).\n\nReturn only the direction value.\n"""


async def _triage_polarity(
    job: Mapping[str, Any], parent: Mapping[str, Any], *, config: WorkerConfig
) -> tuple[str, dict[str, Any]]:
    prompt = _polarity_triage_prompt(job, parent)
    value, worker_meta = await _generate_json_with_retry(
        prompt, _POLARITY_TRIAGE_SCHEMA, config=config.for_meta()
    )
    worker_meta = {
        **worker_meta,
        "triage_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "triage_prompt_character_count": len(prompt),
        "triage_prompt_utf8_bytes": len(prompt.encode("utf-8")),
    }
    direction = str(value.get("direction") or "keep_negative")
    if direction != "convert_to_positive":
        direction = "keep_negative"
    return (direction, worker_meta)


def _attach_judge_scores(
    candidates: Sequence[Mapping[str, Any]],
    grid: Sequence[Sequence[int]],
    *,
    response_ids: Sequence[str],
    max_children: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(grid) != len(response_ids) or any(
        (len(row) != len(candidates) for row in grid)
    ):
        raise PositiveRefinementProtocolError("candidate Judge grid is misaligned")
    if len(candidates) > max_children:
        raise PositiveRefinementProtocolError(
            "fixed child set exceeds the configured cap"
        )
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for column, raw_candidate in enumerate(candidates):
        candidate = dict(raw_candidate)
        pattern = [int(grid[row][column]) for row in range(len(response_ids))]
        if any((value not in (0, 1) for value in pattern)):
            raise PositiveRefinementProtocolError(
                "candidate Judge returned a non-binary score"
            )
        pass_count = sum(pattern)
        group_size = len(response_ids)
        constant = pass_count in {0, group_size}
        weak = not constant and pass_count in {1, group_size - 1}
        if os.environ.get("RUACT_CRITIC_NO_FILTER") == "1":
            rejection_reason = None
        elif constant:
            rejection_reason = (
                "zero_pass_pattern" if pass_count == 0 else "full_pass_pattern"
            )
        elif weak:
            rejection_reason = "weak_imbalanced_pattern"
        else:
            rejection_reason = None
        audit.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "pass_count": pass_count,
                "pass_rate": pass_count / group_size,
                "is_constant": constant,
                "is_weak": weak,
                "retained": os.environ.get("RUACT_CRITIC_NO_FILTER") == "1"
                or (not constant and (not weak)),
                "rejection_reason": rejection_reason,
            }
        )
        candidate["pass_count"] = pass_count
        candidate["judge_scores"] = [
            {"response_id": response_id, "score": pattern[index]}
            for index, response_id in enumerate(response_ids)
        ]
        if os.environ.get("RUACT_CRITIC_NO_FILTER") == "1" or (
            not constant and (not weak)
        ):
            selected.append(candidate)
    return (selected, audit)


def _fixed_meta_valid_children(
    candidates: Sequence[Mapping[str, Any]], *, max_children: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    seen_completion_points: set[str] = set()
    rejected_duplicates: list[dict[str, Any]] = []
    for raw in sorted(candidates, key=lambda item: int(item["candidate_index"])):
        candidate = dict(raw)
        completion_point = _canonical_completion_point(candidate)
        if not completion_point:
            raise PositiveRefinementProtocolError(
                "Meta-valid child has an empty completion point"
            )
        if completion_point in seen_completion_points:
            rejected_duplicates.append(
                {
                    "candidate_index": int(candidate["candidate_index"]),
                    "candidate_source": str(candidate["candidate_source"]),
                    "reason": "canonical_completion_point_duplicate",
                }
            )
            continue
        seen_completion_points.add(completion_point)
        selected.append(candidate)
        if len(selected) == max_children:
            break
    return (selected, rejected_duplicates)


async def process_refinement_job(
    job, *, config, parent_workers=1, execution_mode="single_pass_batched"
):
    validate_refinement_job(job)
    if execution_mode != "single_pass_batched":
        raise ValueError("Only single_pass_batched refinement is supported")
    from .single_pass_worker import process_single_pass_job

    return await process_single_pass_job(
        job, config=config, parent_workers=parent_workers
    )
