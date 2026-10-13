from __future__ import annotations
import asyncio
from dataclasses import replace
import json
import time
from rubric_rl import verl_reward as judge_api
from . import transport as dynamic
from . import worker as legacy
from .protocol import PositiveRefinementProtocolError

PARENT_ATTEMPTS = 3
PARENT_BUDGET_SECONDS = 120.0
JUDGE_BUDGET_SECONDS = 120.0


class RefinementBudgetExhausted(RuntimeError):
    pass


async def prepare_parent(job, parent, config, prepare, per_job_limit):
    started = time.monotonic()
    errors = []
    attempts = 0
    bounded_config = replace(
        config, max_attempts=1, timeout_seconds=min(config.timeout_seconds, 40.0)
    )

    async def generate(prompt, schema, *, config):
        if errors:
            prompt += (
                "\nThe previous attempt failed structural validation: "
                + errors[-1]
                + "\nReturn exactly the requested number of distinct candidates in the candidates array. Do not return an explanation or a different JSON shape."
            )
        return await legacy._generate_json(prompt, schema, config=config)

    async def run():
        nonlocal attempts
        for attempt in range(1, PARENT_ATTEMPTS + 1):
            attempts = attempt
            try:
                async with per_job_limit:
                    if legacy._parent_semaphore is None:
                        result = await prepare(
                            job, parent, bounded_config, generator_call=generate
                        )
                    else:
                        async with legacy._parent_semaphore:
                            result = await prepare(
                                job, parent, bounded_config, generator_call=generate
                            )
                result[1].update(
                    preparation_attempts=attempts,
                    preparation_errors=errors,
                    preparation_seconds=time.monotonic() - started,
                )
                result[1]["generator_call_count"] += attempts - 1
                return result
            except (dynamic.WorkerError, PositiveRefinementProtocolError) as error:
                errors.append(str(error)[:500])
                print(
                    json.dumps(
                        {
                            "event": "refinement_parent_retry",
                            "job_id": job["job_id"],
                            "parent_rubric_id": parent["rubric_id"],
                            "attempt": attempt,
                            "max_attempts": PARENT_ATTEMPTS,
                            "error_type": type(error).__name__,
                            "error": errors[-1],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if attempt < PARENT_ATTEMPTS:
                    await asyncio.sleep(min(config.retry_backoff_seconds, 2.0))
        raise RefinementBudgetExhausted("parent attempt budget exhausted")

    try:
        return await asyncio.wait_for(run(), timeout=PARENT_BUDGET_SECONDS)
    except (asyncio.TimeoutError, RefinementBudgetExhausted) as error:
        print(
            json.dumps(
                {
                    "event": "refinement_parent_terminal",
                    "job_id": job["job_id"],
                    "parent_rubric_id": str(parent["rubric_id"]),
                    "preparation_attempts": attempts,
                    "error_type": type(error).__name__,
                    "preserve_parent": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise RefinementBudgetExhausted(
            "Rubric generation failed; refusing parent fallback"
        ) from error


def _strict_scores(content, count):
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].strip().lower() in ("```", "```json"):
            text = "\n".join(lines[1:-1]).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {str(i) for i in range(1, count + 1)}:
        raise ValueError("missing or extra rubric keys")
    if any((v not in ("PRESENT", "NOT_PRESENT") for v in obj.values())):
        raise ValueError("invalid rubric label")
    return [int(obj[str(i)] == "PRESENT") for i in range(1, count + 1)]


async def judge_candidates(job, candidates, *, workers, attempts):
    if not candidates:
        return (
            [[] for _ in job["group_members"]],
            {"response_task_attempts": 0, "latency_seconds": 0.0},
        )
    started = time.monotonic()
    semaphore = asyncio.Semaphore(max(1, workers))
    descriptions = tuple((str(c["description"]) for c in candidates))
    members = list(job["group_members"])
    counts = [0] * len(members)
    split_counts = [0] * len(members)

    def request(index, indices, correction, budget):
        prompt = judge_api._build_judge_prompt(
            query=str(job["query"]),
            final_answer=str(members[index]["text"]),
            descriptions=tuple((descriptions[i] for i in indices)),
            titles=None,
            polarities=None,
        )
        if correction:
            prompt += "\nReturn only the complete JSON object specified above. Include every required key exactly once. No markdown or explanations."
        limit = dynamic._judge_semaphore
        if limit is not None:
            limit.acquire()
        metadata = {}
        try:
            base, backup = judge_api._judge_current_base(None)
            content = judge_api._request_judge_once(
                prompt,
                rubric_count=len(indices),
                base_url=base,
                use_response_format=not backup,
                response_metadata=metadata,
                max_tokens_override=budget,
            )
            judge_api._judge_breaker_on_success(backup, base)
            return (content, metadata)
        finally:
            if limit is not None:
                limit.release()

    async def score_part(index, indices):
        formats = 0
        transient = 0
        budget = 512
        while True:
            counts[index] += 1
            try:
                async with semaphore:
                    content, meta = await asyncio.to_thread(
                        request, index, indices, formats > 0, budget
                    )
            except (
                TimeoutError,
                ConnectionError,
                OSError,
                judge_api.JudgeProtocolError,
            ) as error:
                transient += 1
                print(
                    json.dumps(
                        {
                            "event": "critic_transport_retry",
                            "job_id": job["job_id"],
                            "response_id": members[index]["id"],
                            "attempt": counts[index],
                            "error_type": type(error).__name__,
                            "http_status": getattr(error, "code", None),
                        }
                    ),
                    flush=True,
                )
                await asyncio.sleep(min(60, 2 ** min(transient, 6)))
                continue
            try:
                if meta.get("finish_reason") == "length":
                    raise ValueError("judge output truncated")
                row = _strict_scores(content, len(indices))
                print(
                    json.dumps(
                        {
                            "event": "critic_scores_recovered",
                            "job_id": job["job_id"],
                            "response_id": members[index]["id"],
                            "columns": indices,
                            "attempts": counts[index],
                        }
                    ),
                    flush=True,
                )
                return row
            except (ValueError, TypeError) as error:
                formats += 1
                print(
                    json.dumps(
                        {
                            "event": "critic_format_retry",
                            "job_id": job["job_id"],
                            "response_id": members[index]["id"],
                            "columns": indices,
                            "format_attempt": formats,
                            "finish_reason": meta.get("finish_reason"),
                            "usage": meta.get("usage"),
                            "error": str(error),
                            "reply_excerpt": content[:2000],
                        }
                    ),
                    flush=True,
                )
                if meta.get("finish_reason") == "length" and budget < 2048:
                    budget *= 2
                    continue
                if formats >= 2 and len(indices) > 1:
                    split_counts[index] += 1
                    size = 5 if len(indices) > 5 else 1
                    rows = []
                    for start in range(0, len(indices), size):
                        rows.extend(
                            await score_part(index, indices[start : start + size])
                        )
                    return rows
                await asyncio.sleep(min(30, 2 ** min(formats, 5)))

    rows = await asyncio.gather(
        *(score_part(i, list(range(len(descriptions)))) for i in range(len(members)))
    )
    return (
        rows,
        {
            "response_task_attempts": sum(counts),
            "per_response": [
                {"response_id": m["id"], "task_attempts": n, "split_count": s}
                for m, n, s in zip(members, counts, split_counts)
            ],
            "latency_seconds": round(time.monotonic() - started, 3),
            "retry_until_success": True,
            "recovery_policy": "strict_json_chunk_recovery",
        },
    )
