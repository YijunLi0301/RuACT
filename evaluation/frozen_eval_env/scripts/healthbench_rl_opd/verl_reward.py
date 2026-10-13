from __future__ import annotations
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import isfinite
from typing import Any, NamedTuple
from healthbench_rl_opd.reward import aggregate_weighted_reward

SUPPORTED_SOURCES = frozenset(
    {
        "rar_medicine_static_rubric",
        "healthbench_static_rubric",
        "rar_science_static_rubric",
    }
)
CANONICAL_JUDGE_MODEL = os.environ.get("HEALTHBENCH_JUDGE_MODEL", "")
CANONICAL_JUDGE_API_BASE = ""
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
JudgeProtocolError = RuntimeError


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
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    if not isinstance(rubrics, list) or not rubrics:
        raise JudgeProtocolError("rubrics must be a non-empty list")
    descriptions: list[str] = []
    weights: list[float] = []
    rubric_ids: list[str] = []
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            raise JudgeProtocolError("each rubric must be a mapping")
        description = _require_nonempty_text(
            rubric.get("description"), f"rubric {index} description"
        )
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
    if len(set(rubric_ids)) != len(rubric_ids):
        raise JudgeProtocolError("rubric keys must be unique")
    return (tuple(descriptions), tuple(weights), tuple(rubric_ids))


def _build_judge_prompt(
    *, query: str, final_answer: str, descriptions: tuple[str, ...]
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


def _parse_judge_reply(content: Any, *, rubric_count: int) -> tuple[int, ...]:
    if not isinstance(content, str) or not content.strip():
        raise JudgeProtocolError("judge reply must be a non-empty JSON object")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise JudgeProtocolError("judge reply is not strict JSON") from error
    expected_keys = tuple((str(index) for index in range(1, rubric_count + 1)))
    if not isinstance(payload, dict) or set(payload) != set(expected_keys):
        raise JudgeProtocolError(
            "judge reply must contain every rubric number exactly once"
        )
    scores = []
    for key in expected_keys:
        value = payload[key]
        if value == "PRESENT":
            scores.append(1)
        elif value == "NOT_PRESENT":
            scores.append(0)
        else:
            raise JudgeProtocolError("judge values must be PRESENT or NOT_PRESENT")
    return tuple(scores)


def _request_judge_once(
    prompt: str, *, rubric_count: int, previous_invalid_reply: str | None = None
) -> str:
    base_url = _require_nonempty_text(
        os.environ.get("HEALTHBENCH_JUDGE_API_BASE"), "HEALTHBENCH_JUDGE_API_BASE"
    ).rstrip("/")
    api_key = _require_nonempty_text(
        os.environ.get("HEALTHBENCH_JUDGE_API_KEY"), "HEALTHBENCH_JUDGE_API_KEY"
    )
    model = _require_nonempty_text(
        os.environ.get("HEALTHBENCH_JUDGE_MODEL"), "HEALTHBENCH_JUDGE_MODEL"
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
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": int(os.environ.get("HEALTHBENCH_JUDGE_MAX_TOKENS", "4096")),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_scores",
                "strict": True,
                "schema": score_schema,
            },
        },
    }
    if os.environ.get("HEALTHBENCH_JUDGE_OMIT_THINKING_FIELDS") != "1":
        body["enable_thinking"] = False
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request_payload = json.dumps(
        body,
        ensure_ascii=os.environ.get("HEALTHBENCH_JUDGE_ASCII_ESCAPE") == "1",
        separators=(",", ":"),
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
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        body = json.load(reply)
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeProtocolError(
            "judge HTTP response has no assistant content"
        ) from error


def _call_judge(
    *, query: str, final_answer: str, descriptions: tuple[str, ...]
) -> JudgeResult:
    prompt = _build_judge_prompt(
        query=query, final_answer=final_answer, descriptions=descriptions
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    attempts = int(os.environ.get("HEALTHBENCH_JUDGE_MAX_ATTEMPTS", "4"))
    if attempts <= 0:
        raise JudgeProtocolError("HEALTHBENCH_JUDGE_MAX_ATTEMPTS must be positive")
    protocol_failures = 0
    attempt = 0
    previous_invalid_reply = None
    while True:
        attempt += 1
        try:
            content = _request_judge_once(
                prompt,
                rubric_count=len(descriptions),
                previous_invalid_reply=previous_invalid_reply,
            )
            try:
                scores = _parse_judge_reply(content, rubric_count=len(descriptions))
            except JudgeProtocolError:
                protocol_failures += 1
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
                time.sleep(min(max(1.0, backoff_cap), 1.5 * attempt))
                continue
            return JudgeResult(
                scores=scores, attempts=attempt, prompt_sha256=prompt_sha256
            )
        except urllib.error.HTTPError as error:
            protocol_failures += 1
            print(
                f"JUDGE_API_RETRY prompt_sha256={prompt_sha256} attempt={attempt} status={error.code} consecutive_failures={protocol_failures}",
                flush=True,
            )
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError) as error:
            protocol_failures += 1
            print(
                f"JUDGE_API_RETRY prompt_sha256={prompt_sha256} attempt={attempt} error={type(error).__name__} consecutive_failures={protocol_failures}",
                flush=True,
            )
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
        backoff_cap = float(
            os.environ.get("HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "30")
        )
        time.sleep(min(max(1.0, backoff_cap), 1.5 * attempt))


def compute_score_stream_single(data_source, solution_str, ground_truth, extra_info):
    if data_source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported data_source: {data_source}")
    if not isinstance(ground_truth, dict) or not isinstance(extra_info, dict):
        raise JudgeProtocolError("ground_truth and extra_info must be mappings")
    query = _require_nonempty_text(ground_truth.get("query"), "query")
    prompt_id = _require_nonempty_text(ground_truth.get("prompt_id"), "prompt_id")
    descriptions, weights, rubric_ids = _validate_rubrics(ground_truth.get("rubrics"))
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
    started = time.monotonic()
    if final_answer is None:
        result = JudgeResult(
            scores=(0,) * len(descriptions), attempts=0, prompt_sha256=""
        )
        judge_called = False
    else:
        result = _call_judge(
            query=query, final_answer=final_answer, descriptions=descriptions
        )
        judge_called = True
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
    rubrics = ground_truth["rubrics"]
    evidence = [
        {**rubric, "judge_score": score}
        for rubric, score in zip(rubrics, result.scores, strict=True)
    ]
    return {
        "score": aggregate_weighted_reward(result.scores, list(weights)),
        "query": query,
        "prompt_id": prompt_id,
        "run_id": run_id,
        "training_step": training_step,
        "occurrence_id": f"{prompt_id}:{training_step}:{rollout_index}",
        "rollout_index": rollout_index,
        "rubrics": evidence,
        "rubric_scores": list(result.scores),
        "rubric_ids": list(rubric_ids),
        "rubric_descriptions": list(descriptions),
        "rubric_weights": list(weights),
        "judge_valid": True,
        "judge_called": judge_called,
        "judge_model": os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
        "judge_attempts": result.attempts,
        "judge_prompt_sha256": result.prompt_sha256,
        "judge_latency_seconds": latency,
        "judge_mode": "ruact_binary_independent_response",
        "student_messages": deepcopy(public_messages),
        "candidate_final_answer": final_answer or "",
        "answer_extraction_status": extraction_status,
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
