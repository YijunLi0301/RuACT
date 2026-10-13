from __future__ import annotations
import hashlib
import threading
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import isfinite
from typing import Any, NamedTuple
from rubric_rl.reward import aggregate_weighted_reward

SUPPORTED_SOURCES = frozenset(
    {"rar_medicine_static_rubric", "rar_science_static_rubric"}
)
CANONICAL_JUDGE_MODEL = os.environ.get("HEALTHBENCH_JUDGE_MODEL", "")
CANONICAL_JUDGE_API_BASE = os.environ.get(
    "HEALTHBENCH_JUDGE_CANONICAL_API_BASE", ""
).rstrip("/")
JUDGE_BACKUP_API_BASE = os.environ.get("HEALTHBENCH_JUDGE_API_BASE_BACKUP", "").rstrip(
    "/"
)
JUDGE_SPLIT_API_BASE = os.environ.get(
    "HEALTHBENCH_JUDGE_SPLIT_SECONDARY_API_BASE", ""
).rstrip("/")
_JUDGE_PRIMARY_BASES = tuple(
    (base for base in (CANONICAL_JUDGE_API_BASE, JUDGE_SPLIT_API_BASE) if base)
)
_JUDGE_SPLIT_ENABLED = len(_JUDGE_PRIMARY_BASES) > 1
_JUDGE_BASE_ALLOWED = frozenset((*_JUDGE_PRIMARY_BASES, JUDGE_BACKUP_API_BASE))
_JUDGE_TRIPPED: dict[str, bool] = {base: False for base in _JUDGE_PRIMARY_BASES}
_JUDGE_STATE: dict[str, dict[str, Any]] = {
    base: {"fails": 0, "oks": 0, "since": time.monotonic()}
    for base in _JUDGE_PRIMARY_BASES
}
_JUDGE_BACKUP_STATE: dict[str, Any] = {"fails": 0, "oks": 0, "since": time.monotonic()}
_JUDGE_FLIP_THRESHOLD = random.randint(
    2, max(2, int(os.environ.get("HEALTHBENCH_JUDGE_FAILOVER_FLIP_AFTER", "3")))
)


def _judge_min_hold() -> float:
    return float(os.environ.get("HEALTHBENCH_JUDGE_FAILOVER_MIN_HOLD_SECONDS", "20"))


def _judge_backup_escape() -> int:
    return max(3, int(os.environ.get("HEALTHBENCH_JUDGE_FAILOVER_ESCAPE_AFTER", "4")))


def _judge_probe_successes() -> int:
    return max(
        8, int(os.environ.get("HEALTHBENCH_JUDGE_FAILOVER_PROBE_SUCCESSES", "32"))
    )


def _judge_probe_interval() -> float:
    return float(
        os.environ.get("HEALTHBENCH_JUDGE_FAILOVER_PROBE_INTERVAL_SECONDS", "240")
    )


def _judge_group_base(prompt_id: str, training_step: int) -> str | None:
    if not _JUDGE_SPLIT_ENABLED:
        return None
    digest = hashlib.sha256(
        f"{prompt_id}\x1f{training_step}".encode("utf-8")
    ).hexdigest()
    return _JUDGE_PRIMARY_BASES[0 if int(digest[:2], 16) < 128 else 1]


def _judge_current_base(assigned_base: str | None = None) -> tuple[str, bool]:
    target = assigned_base if assigned_base is not None else _JUDGE_PRIMARY_BASES[0]
    if _JUDGE_TRIPPED.get(target, False):
        return (JUDGE_BACKUP_API_BASE, True)
    return (target, False)


def _judge_untrip_all() -> None:
    now = time.monotonic()
    for base in _JUDGE_PRIMARY_BASES:
        _JUDGE_TRIPPED[base] = False
        st = _JUDGE_STATE[base]
        st["fails"] = 0
        st["oks"] = 0
        st["since"] = now
    bst = _JUDGE_BACKUP_STATE
    bst["fails"] = 0
    bst["oks"] = 0
    bst["since"] = now


def _judge_breaker_on_failure(is_backup: bool, base_url: str | None = None) -> None:
    now = time.monotonic()
    if is_backup:
        if not any(_JUDGE_TRIPPED.values()):
            return
        bst = _JUDGE_BACKUP_STATE
        bst["fails"] += 1
        bst["oks"] = 0
        if (
            now - bst["since"] >= _judge_min_hold()
            and bst["fails"] >= _judge_backup_escape()
        ):
            _judge_untrip_all()
            suffix = ""
            if _JUDGE_SPLIT_ENABLED:
                suffix = " primaries=" + "|".join(_JUDGE_PRIMARY_BASES)
            print(
                f"JUDGE_FAILOVER action=backup_escape_to_primary base={CANONICAL_JUDGE_API_BASE} consecutive_failures={_judge_backup_escape()}{suffix}",
                flush=True,
            )
        return
    target = base_url if base_url is not None else _JUDGE_PRIMARY_BASES[0]
    if _JUDGE_TRIPPED.get(target, False):
        return
    st = _JUDGE_STATE[target]
    st["fails"] += 1
    st["oks"] = 0
    if now - st["since"] >= _judge_min_hold() and st["fails"] >= _JUDGE_FLIP_THRESHOLD:
        _JUDGE_TRIPPED[target] = True
        st["fails"] = 0
        st["since"] = now
        _JUDGE_BACKUP_STATE["since"] = now
        extra = ""
        if _JUDGE_SPLIT_ENABLED:
            extra = f" from_base={target}"
        print(
            f"JUDGE_FAILOVER action=primary_to_backup base={JUDGE_BACKUP_API_BASE} consecutive_failures={_JUDGE_FLIP_THRESHOLD}{extra}",
            flush=True,
        )


def _judge_breaker_on_success(is_backup: bool, base_url: str | None = None) -> None:
    now = time.monotonic()
    if is_backup:
        if not any(_JUDGE_TRIPPED.values()):
            return
        bst = _JUDGE_BACKUP_STATE
        bst["oks"] += 1
        bst["fails"] = 0
        if (
            bst["oks"] >= _judge_probe_successes()
            and now - bst["since"] >= _judge_probe_interval()
        ):
            _judge_untrip_all()
            suffix = ""
            if _JUDGE_SPLIT_ENABLED:
                suffix = " primaries=" + "|".join(_JUDGE_PRIMARY_BASES)
            print(
                f"JUDGE_FAILOVER action=backup_probe_to_primary base={CANONICAL_JUDGE_API_BASE} consecutive_successes={_judge_probe_successes()}{suffix}",
                flush=True,
            )
        return
    target = base_url if base_url is not None else _JUDGE_PRIMARY_BASES[0]
    if _JUDGE_TRIPPED.get(target, False):
        return
    st = _JUDGE_STATE[target]
    st["oks"] += 1
    st["fails"] = 0


RUACT_JUDGE_INSTRUCTIONS = (
    "Given a user prompt, a generated response, and a list of quality rubrics, e"
    'valuate the response against EACH rubric.\nFor each rubric:\n- Mark "PRESENT"'
    ' if the criterion is satisfied, or "NOT_PRESENT" if it is not. For example,'
    ' given the response "Apples are red", the rubric "Mentions apples" is PRESE'
    'NT, "Does not mention strawberries" is also PRESENT since the response does'
    ' not mention strawberries, and "Mentions oranges" is NOT_PRESENT. Likewise,'
    ' "Avoids mentioning strawberries" is PRESENT, while "Avoids mentioning appl'
    'es" is NOT_PRESENT.\n- If a rubric contains multiple sentences or criteria, '
    "consider all of them. If any required part is not met, mark it NOT_PRESENT."
    " Return PRESENT only if the entire rubric is met.\n- When a rubric gives exa"
    'mples with wording such as "such as", "for example", or "including", the re'
    "sponse does not need to include every example. Satisfying one valid listed "
    "example can satisfy the criterion. There is no partial credit.\nReturn only "
    "one strict JSON object. Its keys must be the rubric numbers and its values "
    'must be exactly "PRESENT" or "NOT_PRESENT". Do not return Markdown, reasons'
    ", or extra keys."
)


class JudgeProtocolError(RuntimeError):
    pass


class JudgeGiveUpError(JudgeProtocolError):
    pass


class JudgeResult(NamedTuple):
    scores: tuple[int, ...]
    attempts: int
    prompt_sha256: str


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JudgeProtocolError(f"{field} must be a non-empty string")
    return value


def _candidate_final_answer(solution: Any) -> tuple[str | None, str]:
    text = str(solution or "").strip()
    if not text:
        return (None, "missing_final_answer")
    close_matches = tuple(re.finditer("</think>", text, flags=re.IGNORECASE))
    if close_matches:
        final_answer = text[close_matches[-1].end() :].strip()
        if final_answer:
            return (final_answer, "after_think_close")
        return (None, "missing_final_answer")
    if re.search("<think>", text, flags=re.IGNORECASE):
        explicit_final = tuple(
            re.finditer(
                "(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:final\\s+answer|\u6700\u7ec8\u7b54\u6848|\u7b54\u6848)\\s*[:\uff1a]\\s*",
                text,
                flags=re.IGNORECASE,
            )
        )
        if explicit_final:
            final_answer = text[explicit_final[-1].end() :].strip()
            if final_answer:
                return (final_answer, "explicit_final_after_malformed_thinking")
        return (None, "missing_final_answer")
    return (text, "plain_final_answer")


def _validate_rubrics(
    rubrics: Any,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(rubrics, list) or not rubrics:
        raise JudgeProtocolError("rubrics must be a non-empty list")
    descriptions: list[str] = []
    weights: list[float] = []
    rubric_ids: list[str] = []
    titles: list[str] = []
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            raise JudgeProtocolError("each rubric must be a mapping")
        description = _require_nonempty_text(
            rubric.get("description"), f"rubric {index} description"
        )
        title = _require_nonempty_text(rubric.get("title"), f"rubric {index} title")
        try:
            weight = float(rubric["weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise JudgeProtocolError(
                f"rubric {index} weight must be numeric"
            ) from error
        if not isfinite(weight) or weight == 0.0:
            raise JudgeProtocolError(
                f"rubric {index} weight must be finite and non-zero"
            )
        rubric_id = _require_nonempty_text(
            rubric.get("rubric_key"), f"rubric {index} rubric_key"
        )
        descriptions.append(description)
        weights.append(weight)
        rubric_ids.append(rubric_id)
        titles.append(title)
    if len(set(rubric_ids)) != len(rubric_ids):
        raise JudgeProtocolError("rubric keys must be unique")
    return (tuple(descriptions), tuple(weights), tuple(rubric_ids), tuple(titles))


def _rubric_channel(rubric: Any, index: int) -> str:
    if not isinstance(rubric, dict):
        raise JudgeProtocolError(f"rubric {index} must be a mapping")
    source = str(rubric.get("source") or "static").strip()
    return "dynamic" if source in {"dynamic_harness", "dynamic_rubric"} else "static"


def _build_judge_prompt(
    *,
    query: str,
    final_answer: str,
    titles: tuple[str, ...] | None,
    descriptions: tuple[str, ...],
    polarities: tuple[str, ...] | None,
) -> str:
    query = _require_nonempty_text(query, "query")
    final_answer = _require_nonempty_text(final_answer, "final answer")
    if not descriptions:
        raise JudgeProtocolError("judge prompt requires rubrics")
    rubrics_text = "\n".join(
        (
            f"{index}. {description}"
            for index, description in enumerate(descriptions, start=1)
        )
    )
    exact_keys = ", ".join((f'"{index}"' for index in range(1, len(descriptions) + 1)))
    return f"{RUACT_JUDGE_INSTRUCTIONS}\n\nThe output object must contain exactly these keys, with no omissions or extras: {exact_keys}. Verify the key count before responding.\n\n<Prompt>\n{query}\n</Prompt>\n\n<Response>\n{final_answer}\n</Response>\n\n<Rubrics>\n{rubrics_text}\n</Rubrics>"


def _parse_judge_reply(
    content: Any, *, rubric_count: int, prompt_sha256: str | None = None
) -> tuple[int, ...]:
    expected_keys = tuple((str(index) for index in range(1, rubric_count + 1)))

    def _score_value(value: Any) -> int | None:
        if isinstance(value, str):
            normalized = re.sub("\\s+", "_", value.strip().lower())
            if (
                "not_present" in normalized
                or normalized.startswith("not_")
                or "absent" in normalized
            ):
                return 0
            if "present" in normalized:
                return 1
            return None
        if value is True:
            return 1
        if value is False or value is None:
            return 0
        if isinstance(value, int) and value in (0, 1):
            return value
        return None

    strict_payload: Any = None
    if isinstance(content, str) and content.strip():
        try:
            strict_payload = json.loads(content)
        except json.JSONDecodeError:
            strict_payload = None
        if isinstance(strict_payload, dict) and set(strict_payload) == set(
            expected_keys
        ):
            values = [strict_payload[key] for key in expected_keys]
            if all((v == "PRESENT" or v == "NOT_PRESENT" for v in values)):
                return tuple((1 if v == "PRESENT" else 0 for v in values))
    payload: Any = None
    if isinstance(content, str):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        scores: list[int] = []
        read = 0
        for key in expected_keys:
            value = _score_value(payload.get(key))
            scores.append(1 if value == 1 else 0)
            read += 1 if value is not None else 0
        if prompt_sha256:
            print(
                f"JUDGE_LENIENT_RECOVERED prompt_sha256={prompt_sha256} expected_count={rubric_count} abstained={rubric_count - read}",
                flush=True,
            )
        return tuple(scores)
    if prompt_sha256:
        print(
            f"JUDGE_ABSTAIN_ALL prompt_sha256={prompt_sha256} expected_count={rubric_count} reason=unparseable_content",
            flush=True,
        )
    return tuple([0] * rubric_count)


_AUXILIARY_COOLDOWN_UNTIL = 0.0
_AUXILIARY_ROUTE_LOCAL = threading.local()


def _request_auxiliary_judge(
    request_body: dict, *, rubric_count: int, deadline: float | None = None
) -> dict | None:
    global _AUXILIARY_COOLDOWN_UNTIL
    if (
        os.environ.get("AUXILIARY_JUDGE_ENABLED") != "1"
        or time.monotonic() < _AUXILIARY_COOLDOWN_UNTIL
    ):
        return None
    if os.environ.get("Refinement_AUXILIARY_SPLIT_BY_QUERY") == "1" and (
        not getattr(_AUXILIARY_ROUTE_LOCAL, "use_auxiliary", True)
    ):
        return None
    base = _require_nonempty_text(
        os.environ.get("AUXILIARY_JUDGE_API_BASE"), "auxiliary base"
    ).rstrip("/")
    key = _require_nonempty_text(
        os.environ.get("AUXILIARY_JUDGE_API_KEY"), "auxiliary key"
    )
    model = _require_nonempty_text(
        os.environ.get("AUXILIARY_JUDGE_MODEL", ""), "auxiliary model"
    )
    payload = {
        **request_body,
        "model": model,
        "max_tokens": max(512, int(request_body["max_tokens"])),
    }
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        timeout = float(os.environ.get("AUXILIARY_JUDGE_TIMEOUT_SECONDS", "45"))
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("auxiliary judge deadline exhausted")
            timeout = min(timeout, remaining)
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            body = json.load(reply)
        choice = body["choices"][0]
        content = choice["message"]["content"].strip()
        fence = chr(96) * 3
        if content.startswith(fence):
            content = re.sub(
                "^" + fence + "(?:json)?\\s*|\\s*" + fence + "$", "", content
            ).strip()
        scores = json.loads(content)
        expected = {str(i) for i in range(1, rubric_count + 1)}
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(scores, dict)
            or set(scores) != expected
            or any((v not in ("PRESENT", "NOT_PRESENT") for v in scores.values()))
        ):
            raise ValueError("incomplete auxiliary Judge result")
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        _AUXILIARY_COOLDOWN_UNTIL = time.monotonic() + 60.0
        print(
            json.dumps(
                {
                    "event": "auxiliary_judge_fallback",
                    "error_type": type(error).__name__,
                    "status": getattr(error, "code", None),
                    "cooldown_seconds": 60,
                }
            ),
            flush=True,
        )
        return None
    print(
        json.dumps(
            {
                "event": "auxiliary_judge_success",
                "base": base,
                "model": model,
                "rubric_count": rubric_count,
                "seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )
    body["choices"][0]["message"]["content"] = json.dumps(scores, separators=(",", ":"))
    return body


def _request_judge_once(
    prompt: str,
    *,
    rubric_count: int,
    previous_invalid_reply: str | None = None,
    base_url: str | None = None,
    use_response_format: bool = True,
    deadline: float | None = None,
    response_metadata: dict | None = None,
    max_tokens_override: int | None = None,
) -> str:
    if base_url is None:
        base_url = _require_nonempty_text(
            os.environ.get("HEALTHBENCH_JUDGE_API_BASE"), "HEALTHBENCH_JUDGE_API_BASE"
        ).rstrip("/")
    api_key = _require_nonempty_text(
        os.environ.get("HEALTHBENCH_JUDGE_API_KEY"), "HEALTHBENCH_JUDGE_API_KEY"
    )
    model = _require_nonempty_text(
        os.environ.get("HEALTHBENCH_JUDGE_MODEL"), "HEALTHBENCH_JUDGE_MODEL"
    )
    if base_url not in _JUDGE_BASE_ALLOWED:
        raise JudgeProtocolError(
            f"judge base_url must be one of {sorted(_JUDGE_BASE_ALLOWED)!r}"
        )
    if model != CANONICAL_JUDGE_MODEL:
        raise JudgeProtocolError(
            f"HEALTHBENCH_JUDGE_MODEL must be {CANONICAL_JUDGE_MODEL!r}"
        )
    expected_keys = tuple((str(index) for index in range(1, rubric_count + 1)))
    score_schema = {
        "type": "object",
        "properties": {
            key: {"type": "string", "enum": ["PRESENT", "NOT_PRESENT"]}
            for key in expected_keys
        },
        "required": list(expected_keys),
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": prompt}]
    if previous_invalid_reply is not None:
        messages.extend(
            [
                {"role": "assistant", "content": previous_invalid_reply[:8192]},
                {
                    "role": "user",
                    "content": f"The previous object violated the required key schema. Return a corrected JSON object containing exactly these keys and no others: {', '.join(expected_keys)}. Do not add explanations.",
                },
            ]
        )
    request_body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": (
            max_tokens_override
            if max_tokens_override is not None
            else int(os.environ.get("HEALTHBENCH_JUDGE_MAX_TOKENS", "4096"))
        ),
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if use_response_format:
        request_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_scores",
                "strict": True,
                "schema": score_schema,
            },
        }
    request_payload = json.dumps(
        request_body, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=request_payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("HEALTHBENCH_JUDGE_TIMEOUT_SECONDS", "180"))
    body = _request_auxiliary_judge(
        request_body, rubric_count=rubric_count, deadline=deadline
    )
    if body is None:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("judge deadline exhausted before primary request")
            timeout = min(timeout, remaining)
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            body = json.load(reply)
    if response_metadata is not None:
        response_metadata.update(
            finish_reason=(body.get("choices") or [{}])[0].get("finish_reason"),
            usage=body.get("usage"),
        )
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            (part.get("text", "") for part in content if isinstance(part, dict))
        )
    if not isinstance(content, str):
        content = ""
    return content


def _call_judge(
    *,
    query: str,
    final_answer: str,
    descriptions: tuple[str, ...],
    titles: tuple[str, ...] | None = None,
    polarities: tuple[str, ...] | None = None,
    group_base: str | None = None,
    max_total_attempts: int | None = None,
    deadline: float | None = None,
    strict_response: bool = False,
) -> JudgeResult:
    _AUXILIARY_ROUTE_LOCAL.use_auxiliary = (
        int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16) % 2 == 0
    )
    prompt = _build_judge_prompt(
        query=query,
        final_answer=final_answer,
        titles=titles,
        descriptions=descriptions,
        polarities=polarities,
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    attempts = int(os.environ.get("HEALTHBENCH_JUDGE_MAX_ATTEMPTS", "8"))
    if attempts <= 0:
        raise JudgeProtocolError("HEALTHBENCH_JUDGE_MAX_ATTEMPTS must be positive")
    protocol_failures = 0
    attempt = 0
    previous_invalid_reply = None
    schema_failures = 0
    while True:
        if (
            max_total_attempts is not None
            and attempt >= max_total_attempts
            or (deadline is not None and time.monotonic() >= deadline)
        ):
            raise JudgeGiveUpError("judge total attempt/time budget exhausted")
        attempt += 1
        base_url, is_backup = _judge_current_base(group_base)
        try:
            content = _request_judge_once(
                prompt,
                rubric_count=len(descriptions),
                previous_invalid_reply=previous_invalid_reply,
                base_url=base_url,
                use_response_format=not is_backup,
                deadline=deadline,
            )
            _judge_breaker_on_success(is_backup, base_url)
            try:
                if strict_response:
                    try:
                        parsed = json.loads(content)
                    except (ValueError, TypeError) as error:
                        raise JudgeProtocolError(
                            "strict judge response is not JSON"
                        ) from error
                    if (
                        not isinstance(parsed, dict)
                        or set(parsed)
                        != {str(i) for i in range(1, len(descriptions) + 1)}
                        or any(
                            (
                                value not in ("PRESENT", "NOT_PRESENT")
                                for value in parsed.values()
                            )
                        )
                    ):
                        raise JudgeProtocolError(
                            "strict judge response has missing/invalid labels"
                        )
                scores = _parse_judge_reply(
                    content, rubric_count=len(descriptions), prompt_sha256=prompt_sha256
                )
            except JudgeProtocolError:
                protocol_failures += 1
                schema_failures += 1
                if (
                    max_total_attempts is not None
                    and attempt >= max_total_attempts
                    or schema_failures
                    >= int(
                        os.environ.get("HEALTHBENCH_JUDGE_MAX_PROTOCOL_ATTEMPTS", "5")
                    )
                ):
                    print(
                        f"JUDGE_GIVEUP prompt_sha256={prompt_sha256} attempts={attempt} schema_failures={schema_failures} expected_count={len(descriptions)}",
                        flush=True,
                    )
                    raise JudgeGiveUpError(
                        f"judge protocol give up after {schema_failures} schema failures"
                    )
                previous_invalid_reply = content
                try:
                    observed = json.loads(content)
                    observed_keys = (
                        sorted(observed)
                        if isinstance(observed, dict)
                        else [type(observed).__name__]
                    )
                except (json.JSONDecodeError, TypeError):
                    observed_keys = ["non_json"]
                print(
                    f"JUDGE_PROTOCOL_RETRY prompt_sha256={prompt_sha256} attempt={attempt} expected_count={len(descriptions)} observed_keys={observed_keys}",
                    flush=True,
                )
                backoff_cap = float(
                    os.environ.get("HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "30")
                )
                time.sleep(
                    min(max(1.0, backoff_cap), 1.5 * attempt) * random.uniform(0.5, 1.5)
                )
                continue
            return JudgeResult(
                scores=scores, attempts=attempt, prompt_sha256=prompt_sha256
            )
        except JudgeGiveUpError:
            raise
        except urllib.error.HTTPError as error:
            protocol_failures += 1
            print(
                f"JUDGE_API_RETRY prompt_sha256={prompt_sha256} attempt={attempt} status={error.code} consecutive_failures={protocol_failures}",
                flush=True,
            )
            if error.code == 429 or not 400 <= error.code < 500:
                _judge_breaker_on_failure(is_backup, base_url)
            elif protocol_failures >= int(
                os.environ.get("HEALTHBENCH_JUDGE_MAX_PERMANENT_4XX", "5")
            ):
                print(
                    f"JUDGE_GIVEUP prompt_sha256={prompt_sha256} attempts={attempt} status={error.code} reason=permanent_client_error",
                    flush=True,
                )
                raise JudgeGiveUpError(
                    f"judge permanent client error {error.code} after {attempt} attempts"
                )
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError) as error:
            protocol_failures += 1
            print(
                f"JUDGE_API_RETRY prompt_sha256={prompt_sha256} attempt={attempt} error={type(error).__name__} consecutive_failures={protocol_failures}",
                flush=True,
            )
            _judge_breaker_on_failure(is_backup, base_url)
        except JudgeProtocolError as error:
            protocol_failures += 1
            print(
                f"JUDGE_PROTOCOL_RETRY prompt_sha256={prompt_sha256} attempt={attempt} expected_count={len(descriptions)} error={type(error).__name__}",
                flush=True,
            )
        except Exception as error:
            protocol_failures += 1
            print(
                f"JUDGE_RETRY prompt_sha256={prompt_sha256} attempt={attempt} error={type(error).__name__} consecutive_failures={protocol_failures}",
                flush=True,
            )
            _judge_breaker_on_failure(is_backup, base_url)
        if max_total_attempts is not None and attempt >= max_total_attempts:
            raise JudgeGiveUpError("judge total attempt budget exhausted")
        if deadline is not None and time.monotonic() >= deadline:
            raise JudgeGiveUpError("judge time budget exhausted")
        backoff_cap = float(
            os.environ.get("HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "30")
        )
        time.sleep(min(max(1.0, backoff_cap), 1.5 * attempt) * random.uniform(0.5, 1.5))


def compute_score_stream_single(data_source, solution_str, ground_truth, extra_info):
    if data_source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported data_source: {data_source}")
    if not isinstance(ground_truth, dict) or not isinstance(extra_info, dict):
        raise JudgeProtocolError("ground_truth and extra_info must be mappings")
    query = _require_nonempty_text(ground_truth.get("query"), "query")
    prompt_id = _require_nonempty_text(ground_truth.get("prompt_id"), "prompt_id")
    routing_step = extra_info.get("training_step")
    if (
        isinstance(routing_step, bool)
        or not isinstance(routing_step, int)
        or routing_step < 0
    ):
        raise JudgeProtocolError("training_step must be a non-negative integer")
    group_base = _judge_group_base(prompt_id, routing_step)
    descriptions, weights, rubric_ids, titles = _validate_rubrics(
        ground_truth.get("rubrics")
    )
    rubrics = ground_truth["rubrics"]
    rubric_channels = tuple(
        (_rubric_channel(rubric, index) for index, rubric in enumerate(rubrics))
    )
    static_indices = tuple(
        (index for index, channel in enumerate(rubric_channels) if channel == "static")
    )
    dynamic_indices = tuple(
        (index for index, channel in enumerate(rubric_channels) if channel == "dynamic")
    )
    if not static_indices and (not dynamic_indices):
        raise JudgeProtocolError("at least one rubric is required")
    static_descriptions = tuple((descriptions[index] for index in static_indices))
    static_weights = tuple((weights[index] for index in static_indices))
    static_rubric_ids = tuple((rubric_ids[index] for index in static_indices))
    static_titles = tuple((titles[index] for index in static_indices))
    dynamic_descriptions = tuple((descriptions[index] for index in dynamic_indices))
    dynamic_weights = tuple((weights[index] for index in dynamic_indices))
    dynamic_rubric_ids = tuple((rubric_ids[index] for index in dynamic_indices))
    dynamic_titles = tuple((titles[index] for index in dynamic_indices))
    static_polarities = tuple(
        ("positive" if weight > 0.0 else "negative" for weight in static_weights)
    )
    dynamic_polarities = tuple(
        ("positive" if weight > 0.0 else "negative" for weight in dynamic_weights)
    )
    public_messages = extra_info.get("student_messages")
    if not isinstance(public_messages, list) or not public_messages:
        raise JudgeProtocolError(
            "student_messages must be a non-empty public conversation"
        )
    if any(
        (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or (not isinstance(message["role"], str))
            or (not isinstance(message["content"], str))
            for message in public_messages
        )
    ):
        raise JudgeProtocolError("student_messages contain an invalid public message")
    final_answer, extraction_status = _candidate_final_answer(solution_str)
    judge_gave_up = False
    started = time.monotonic()
    if final_answer is None:
        static_result = JudgeResult(
            scores=(0,) * len(static_descriptions), attempts=0, prompt_sha256=""
        )
        dynamic_result = JudgeResult(
            scores=(0,) * len(dynamic_descriptions), attempts=0, prompt_sha256=""
        )
        judge_called = False
    else:
        if static_descriptions:
            try:
                static_result = _call_judge(
                    query=query,
                    final_answer=final_answer,
                    titles=static_titles,
                    descriptions=static_descriptions,
                    polarities=static_polarities,
                    group_base=group_base,
                )
            except JudgeGiveUpError:
                judge_gave_up = True
                static_result = JudgeResult(
                    scores=(0,) * len(static_descriptions), attempts=0, prompt_sha256=""
                )
        else:
            static_result = JudgeResult(scores=(), attempts=0, prompt_sha256="")
        if dynamic_descriptions:
            try:
                dynamic_result = _call_judge(
                    query=query,
                    final_answer=final_answer,
                    descriptions=dynamic_descriptions,
                    polarities=dynamic_polarities,
                    group_base=group_base,
                )
            except JudgeGiveUpError:
                judge_gave_up = True
                dynamic_result = JudgeResult(
                    scores=(0,) * len(dynamic_descriptions),
                    attempts=0,
                    prompt_sha256="",
                )
        else:
            dynamic_result = JudgeResult(scores=(), attempts=0, prompt_sha256="")
        judge_called = bool(static_descriptions)
    if len(static_result.scores) != len(static_descriptions) or len(
        dynamic_result.scores
    ) != len(dynamic_descriptions):
        raise JudgeProtocolError(
            "judge score channels are not aligned with their rubrics"
        )
    latency = time.monotonic() - started
    rollout_index = extra_info.get("rollout_index")
    if (
        isinstance(rollout_index, bool)
        or not isinstance(rollout_index, int)
        or rollout_index < 0
    ):
        raise JudgeProtocolError("rollout_index must be a non-negative integer")
    training_step = extra_info.get("training_step")
    if (
        isinstance(training_step, bool)
        or not isinstance(training_step, int)
        or training_step < 0
    ):
        raise JudgeProtocolError("training_step must be a non-negative integer")
    run_id = _require_nonempty_text(
        extra_info.get("run_id") or os.environ.get("HEALTHBENCH_RUN_ID"), "run_id"
    )
    static_scores = tuple(static_result.scores)
    dynamic_scores = tuple(dynamic_result.scores)
    combined_scores = [0] * len(rubrics)
    for index, score in zip(static_indices, static_scores, strict=True):
        combined_scores[index] = score
    for index, score in zip(dynamic_indices, dynamic_scores, strict=True):
        combined_scores[index] = score
    evidence = [
        {**rubric, "judge_score": score}
        for rubric, score in zip(rubrics, combined_scores, strict=True)
    ]
    finish_reason = str(extra_info.get("finish_reason") or "unknown")
    if finish_reason == "unknown" and extra_info.get("truncated") is True:
        finish_reason = "length"
    return {
        "score": aggregate_weighted_reward(
            static_scores + dynamic_scores, list(static_weights + dynamic_weights)
        ),
        "query": query,
        "prompt_id": prompt_id,
        "run_id": run_id,
        "training_step": training_step,
        "occurrence_id": f"{prompt_id}:{training_step}:{rollout_index}",
        "response_id": f"{prompt_id}:{training_step}:{rollout_index}",
        "rollout_index": rollout_index,
        "rubrics": evidence,
        "rubric_scores": list(static_scores),
        "rubric_ids": list(static_rubric_ids),
        "rubric_titles": list(static_titles),
        "rubric_descriptions": list(static_descriptions),
        "rubric_weights": list(static_weights),
        "dynamic_rubric_scores": list(dynamic_scores),
        "dynamic_rubric_ids": list(dynamic_rubric_ids),
        "dynamic_rubric_titles": list(dynamic_titles),
        "dynamic_rubric_descriptions": list(dynamic_descriptions),
        "dynamic_rubric_weights": list(dynamic_weights),
        "rubric_snapshot_version": int(ground_truth.get("rubric_snapshot_version", 0)),
        "rubric_query_version": int(ground_truth.get("rubric_query_version", 0)),
        "judge_valid": not judge_gave_up,
        "judge_dropped": judge_gave_up,
        "judge_drop_reason": "protocol_give_up" if judge_gave_up else "",
        "judge_called": judge_called,
        "judge_model": os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
        "judge_attempts": static_result.attempts,
        "judge_prompt_sha256": static_result.prompt_sha256,
        "dynamic_judge_called": bool(final_answer is not None and dynamic_descriptions),
        "dynamic_judge_attempts": dynamic_result.attempts,
        "dynamic_judge_prompt_sha256": dynamic_result.prompt_sha256,
        "judge_latency_seconds": latency,
        "judge_mode": "ruact_binary_independent_response",
        "student_messages": deepcopy(public_messages),
        "candidate_final_answer": final_answer or "",
        "answer_extraction_status": extraction_status,
        "finish_reason": finish_reason,
        "split": _require_nonempty_text(extra_info.get("split"), "split"),
    }


def _compute_independent_rows(data_sources, solution_strs, ground_truths, extra_infos):
    lengths = {
        len(data_sources),
        len(solution_strs),
        len(ground_truths),
        len(extra_infos),
    }
    if len(lengths) != 1 or not solution_strs:
        raise JudgeProtocolError("reward inputs must be non-empty and aligned")

    def score(index: int):
        return compute_score_stream_single(
            data_sources[index],
            solution_strs[index],
            ground_truths[index],
            extra_infos[index],
        )

    workers = max(1, int(os.environ.get("HEALTHBENCH_JUDGE_WORKERS", "32")))
    with ThreadPoolExecutor(max_workers=min(workers, len(solution_strs))) as pool:
        return list(pool.map(score, range(len(solution_strs))))


def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
    expected_group_size = int(os.environ.get("HEALTHBENCH_ROLLOUT_GROUP_SIZE", "8"))
    keys = []
    for ground_truth, extra_info in zip(ground_truths, extra_infos, strict=True):
        keys.append(
            (
                (
                    ground_truth.get("prompt_id")
                    if isinstance(ground_truth, dict)
                    else None
                ),
                (
                    extra_info.get("training_step")
                    if isinstance(extra_info, dict)
                    else None
                ),
            )
        )
    counts = Counter(keys)
    if not counts or any((count != expected_group_size for count in counts.values())):
        raise JudgeProtocolError(
            f"each query must contain exactly {expected_group_size} rollout rows"
        )
    return _compute_independent_rows(
        data_sources, solution_strs, ground_truths, extra_infos
    )


def compute_score_stream_chunk(data_sources, solution_strs, ground_truths, extra_infos):
    return _compute_independent_rows(
        data_sources, solution_strs, ground_truths, extra_infos
    )
