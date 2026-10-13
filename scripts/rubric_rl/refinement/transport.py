from __future__ import annotations
from .protocol import PositiveRefinementProtocolError as WorkerError
import asyncio
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import httpx
from rubric_rl.verl_reward import _call_judge


@dataclass(frozen=True)
class WorkerConfig:
    base_url: str = ""
    api_key: str = ""
    backend: str = "litellm_http"
    model: str = ""
    enable_thinking: bool = False
    max_tokens: int = 4096
    temperature: float = 0.8
    meta_model: str | None = None
    max_attempts: int = 3
    timeout_seconds: float = 180.0
    retry_backoff_seconds: float = 2.0
    judge_workers: int = 8
    judge_attempts: int = 3
    generator_concurrency: int = 64
    topup_enabled: bool = True

    def validate(self) -> None:
        if not self.base_url.strip():
            raise WorkerError("LiteLLM base_url must be non-empty")
        if not self.model.strip():
            raise WorkerError("LiteLLM model must be non-empty")
        if self.timeout_seconds <= 0 or self.retry_backoff_seconds < 0:
            raise WorkerError("generator timeout/backoff is invalid")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or (not 1 <= self.max_attempts <= 10)
        ):
            raise WorkerError("generator attempts must be between one and ten")
        if self.judge_workers <= 0 or not 1 <= self.judge_attempts <= 3:
            raise WorkerError("Judge workers/attempts are invalid")
        if self.meta_model is not None and (not self.meta_model.strip()):
            raise WorkerError("Meta model override must be non-empty")
        if self.max_tokens <= 0 or not 0 <= self.temperature <= 2:
            raise WorkerError("generator max_tokens/temperature is invalid")
        if (
            isinstance(self.generator_concurrency, bool)
            or not isinstance(self.generator_concurrency, int)
            or self.generator_concurrency <= 0
        ):
            raise WorkerError("generator concurrency must be positive")

    def for_meta(self) -> "WorkerConfig":
        return replace(self, model=self.meta_model or self.model)


_judge_semaphore: threading.BoundedSemaphore | None = None


def configure_judge_concurrency(limit: int) -> None:
    global _judge_semaphore
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise WorkerError("global Judge concurrency must be positive")
    _judge_semaphore = threading.BoundedSemaphore(limit)


_generator_semaphore: asyncio.Semaphore | None = None


def configure_generator_concurrency(limit: int) -> None:
    global _generator_semaphore
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise WorkerError("global generator concurrency must be positive")
    _generator_semaphore = asyncio.Semaphore(limit)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _content_diagnostic(
    content: Any, error: BaseException, *, finish_reason: Any
) -> str:
    if not isinstance(content, str):
        return f"generator assistant content is not text (type={type(content).__name__}, finish_reason={finish_reason!r})"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    location = ""
    if isinstance(error, json.JSONDecodeError):
        location = f", line={error.lineno}, column={error.colno}, position={error.pos}"
    return f"generator assistant content is not valid JSON (length={len(content)}, sha256={digest}, finish_reason={finish_reason!r}{location})"


def _parse_generator_json(
    content: Any, *, finish_reason: Any
) -> tuple[dict[str, Any], bool]:
    if not isinstance(content, str):
        error = TypeError("assistant content is not text")
        raise WorkerError(
            _content_diagnostic(content, error, finish_reason=finish_reason)
        ) from error
    stripped = _strip_code_fence(content)
    try:
        value = json.loads(stripped)
        recovered = False
    except json.JSONDecodeError as initial_error:
        object_start = stripped.find("{")
        if object_start < 0:
            raise WorkerError(
                _content_diagnostic(content, initial_error, finish_reason=finish_reason)
            ) from initial_error
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[object_start:])
            recovered = True
        except json.JSONDecodeError as recovery_error:
            raise WorkerError(
                _content_diagnostic(
                    content, recovery_error, finish_reason=finish_reason
                )
            ) from recovery_error
    if not isinstance(value, dict):
        error = ValueError("generator response is not a JSON object")
        raise WorkerError(
            _content_diagnostic(content, error, finish_reason=finish_reason)
        ) from error
    return (value, recovered)


def _repair_messages(prompt: str, invalid_content: Any) -> list[dict[str, str]]:
    prior = invalid_content if isinstance(invalid_content, str) else ""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prior[:8192]},
        {
            "role": "user",
            "content": "The previous reply was not valid JSON. Return one corrected JSON object that strictly matches the response_format schema. Output JSON only: no markdown fence, explanation, or text before or after the object.",
        },
    ]


async def _generate_json_with_retry(
    prompt: str, schema: Mapping[str, Any], *, config: WorkerConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    config.validate()
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_generation",
                "strict": True,
                "schema": schema,
            },
        },
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    started = time.monotonic()
    last_error: BaseException | None = None
    repair_attempts = 0
    recovered_json = False
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=30.0),
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=0),
    ) as client:
        for attempt in range(1, config.max_attempts + 1):
            try:
                if _generator_semaphore is not None:
                    async with _generator_semaphore:
                        response = await client.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=config.timeout_seconds,
                        )
                else:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=config.timeout_seconds,
                    )
                response.raise_for_status()
                try:
                    body = response.json()
                except json.JSONDecodeError as error:
                    response_text = response.text
                    digest = hashlib.sha256(
                        response_text.encode("utf-8", errors="replace")
                    ).hexdigest()
                    raise WorkerError(
                        f"generator HTTP body is not JSON (status={response.status_code}, length={len(response_text)}, sha256={digest}, position={error.pos})"
                    ) from error
                choice = body["choices"][0]
                message = choice["message"]
                content = message.get("content")
                try:
                    value, recovered_json = _parse_generator_json(
                        content, finish_reason=choice.get("finish_reason")
                    )
                except WorkerError:
                    if attempt < config.max_attempts:
                        repair_attempts += 1
                        payload["messages"] = _repair_messages(prompt, content)
                    raise
                return (
                    value,
                    {
                        "backend": "litellm_http",
                        "model": config.model,
                        "enable_thinking": config.enable_thinking,
                        "temperature": config.temperature,
                        "attempts": attempt,
                        "repair_attempts": repair_attempts,
                        "json_wrapper_recovered": recovered_json,
                        "latency_seconds": round(time.monotonic() - started, 3),
                    },
                )
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                WorkerError,
            ) as error:
                last_error = error
                if attempt < config.max_attempts:
                    await asyncio.sleep(
                        min(
                            30.0,
                            config.retry_backoff_seconds * 2 ** min(attempt - 1, 4),
                        )
                    )
    error_detail = (
        f"{type(last_error).__name__}: {last_error}"
        if last_error is not None
        else "unknown"
    )
    raise WorkerError(
        f"generator HTTP failed after {config.max_attempts} attempts ({error_detail})"
    )


def _judge_candidates_forever(
    job: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    attempts: int = 3,
    descriptions_only: bool = False,
) -> tuple[list[list[int]], dict[str, Any]]:
    if not 1 <= attempts <= 3:
        raise WorkerError("fixed Judge attempts must be between one and three")
    if not candidates:
        return (
            [[] for _ in job["group_members"]],
            {
                "complete_group_attempts": 0,
                "response_task_attempts": 0,
                "successful_http_attempts": 0,
                "retried_response_count": 0,
                "latency_seconds": 0.0,
                "per_response": [],
            },
        )
    titles = tuple((str(item["title"]) for item in candidates))
    descriptions = tuple((str(item["description"]) for item in candidates))
    polarities = tuple((str(item["polarity"]) for item in candidates))
    members = list(job["group_members"])
    rows: list[list[int] | None] = [None] * len(members)
    response_task_attempts = [0] * len(members)
    successful_http_attempts = [0] * len(members)
    response_latency = [0.0] * len(members)
    last_errors: list[str | None] = [None] * len(members)
    group_attempt = 0
    started = time.monotonic()

    def score(index: int) -> tuple[int, list[int], int, float]:
        request_started = time.monotonic()
        kwargs: dict[str, Any] = {
            "query": str(job["query"]),
            "final_answer": str(members[index]["text"]),
            "descriptions": descriptions,
        }
        if not descriptions_only:
            kwargs.update({"titles": titles, "polarities": polarities})
        if _judge_semaphore is None:
            result = _call_judge(**kwargs)
        else:
            with _judge_semaphore:
                result = _call_judge(**kwargs)
        return (
            index,
            list(result.scores),
            int(result.attempts),
            time.monotonic() - request_started,
        )

    retry_until_success = os.environ.get("Refinement_JUDGE_RETRY_UNTIL_SUCCESS") == "1"
    while any((row is None for row in rows)) and (
        retry_until_success or group_attempt < attempts
    ):
        group_attempt += 1
        missing = [index for index, row in enumerate(rows) if row is None]
        failed = False
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = {}
            for index in missing:
                response_task_attempts[index] += 1
                futures[pool.submit(score, index)] = index
            for future in as_completed(futures):
                try:
                    index, scores, http_attempts, latency = future.result()
                    if len(scores) != len(candidates):
                        raise WorkerError(
                            "fixed Judge returned a misaligned candidate row"
                        )
                    rows[index] = scores
                    successful_http_attempts[index] += http_attempts
                    response_latency[index] += latency
                except Exception as error:
                    index = futures[future]
                    last_errors[index] = (
                        f"{type(error).__name__}: {' '.join(str(error).split())[:1000]}"
                    )
                    failed = True
        if (
            failed
            and any((row is None for row in rows))
            and (retry_until_success or group_attempt < attempts)
        ):
            print(
                json.dumps(
                    {
                        "event": "refinement_judge_missing_rows_retry",
                        "job_id": job["job_id"],
                        "round": group_attempt,
                        "missing_rows": [
                            i for i, row in enumerate(rows) if row is None
                        ],
                        "kept_rows": sum((row is not None for row in rows)),
                    }
                ),
                flush=True,
            )
            time.sleep(min(30.0, 2 ** min(group_attempt - 1, 4)))
    if any((row is None for row in rows)):
        missing = [index for index, row in enumerate(rows) if row is None]
        raise WorkerError(
            f"fixed Judge left response rows {missing} incomplete after {attempts} attempts; last_errors={{{', '.join((f'{index}: {last_errors[index]}' for index in missing))}}}"
        )
    completed_rows = [list(row or []) for row in rows]
    if len(completed_rows) != 8:
        raise WorkerError("fixed Judge returned an incomplete candidate grid")
    return (
        completed_rows,
        {
            "complete_group_attempts": group_attempt,
            "response_task_attempts": sum(response_task_attempts),
            "successful_http_attempts": sum(successful_http_attempts),
            "retried_response_count": sum(
                (value > 1 for value in response_task_attempts)
            ),
            "latency_seconds": round(time.monotonic() - started, 3),
            "per_response": [
                {
                    "response_id": str(member["id"]),
                    "task_attempts": response_task_attempts[index],
                    "successful_http_attempts": successful_http_attempts[index],
                    "latency_seconds": round(response_latency[index], 3),
                }
                for index, member in enumerate(members)
            ],
        },
    )
