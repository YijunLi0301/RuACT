from __future__ import annotations
import hashlib
from numbers import Integral
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TeacherGateStats:
    total: int = 0
    passed: int = 0
    rejected_not_better: int = 0
    rejected_judge_give_up: int = 0
    invalid_final_answer: int = 0
    reused_samples: int = 0
    rescored_samples: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class _AnchorVerdict:
    keep: bool
    kind: str
    row_index: int
    text: str
    detail: str = ""
    score: float = float("-inf")
    reused: bool = False


def select_teacher_reward_feedback(metadata, slots):
    keys = (
        "judge_valid",
        "judge_model",
        "query",
        "candidate_final_answer",
        "rubric_descriptions",
        "rubric_weights",
        "rubric_scores",
        "judge_prompt_sha256",
        "dynamic_rubric_descriptions",
        "dynamic_rubric_weights",
        "dynamic_rubric_scores",
        "dynamic_judge_prompt_sha256",
    )
    return [
        {key: metadata[key][slot] for key in keys if key in metadata} for slot in slots
    ]


def _cached_teacher_scores(
    feedback,
    *,
    query,
    final_answer,
    static_descriptions,
    static_weights,
    dynamic_descriptions,
    dynamic_weights,
    build_judge_prompt,
    judge_model,
):
    if feedback is None or build_judge_prompt is None or (not judge_model):
        return None
    try:
        if (
            not feedback.get("judge_valid")
            or feedback.get("judge_model") != judge_model
            or feedback.get("query") != query
            or (feedback.get("candidate_final_answer") != final_answer)
        ):
            return None
        channels = []
        for prefix, descriptions, weights in (
            ("", static_descriptions, static_weights),
            ("dynamic_", dynamic_descriptions, dynamic_weights),
        ):
            if (
                tuple(feedback.get(prefix + "rubric_descriptions", ())) != descriptions
                or tuple(feedback.get(prefix + "rubric_weights", ())) != weights
            ):
                return None
            scores = tuple(feedback.get(prefix + "rubric_scores", ()))
            if len(scores) != len(descriptions) or any(
                (
                    isinstance(score, bool)
                    or not isinstance(score, Integral)
                    or score not in (0, 1)
                    for score in scores
                )
            ):
                return None
            if descriptions:
                prompt = build_judge_prompt(
                    query=query,
                    final_answer=final_answer,
                    titles=None,
                    descriptions=descriptions,
                    polarities=tuple(
                        ("positive" if w > 0 else "negative" for w in weights)
                    ),
                )
                if (
                    feedback.get(prefix + "judge_prompt_sha256")
                    != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                ):
                    return None
            channels.append(tuple((int(score) for score in scores)))
        return tuple(channels)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def compute_group_best(
    uids: Sequence[Any], raw_scores: Sequence[float]
) -> dict[Any, float]:
    if len(uids) != len(raw_scores) or not uids:
        raise ValueError("uids and raw_scores must be non-empty and aligned")
    group_best: dict[Any, float] = defaultdict(lambda: float("-inf"))
    for uid, score in zip(uids, raw_scores, strict=True):
        numeric = float(score)
        if numeric > group_best[uid]:
            group_best[uid] = numeric
    return dict(group_best)


def _query_from_raw_prompt(messages: Any) -> str:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("raw_prompt must be a non-empty message list")
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise ValueError("raw_prompt must end with a user message")
    content = last.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("raw_prompt final user content must be a non-empty string")
    return content


def _verify_one_anchor(
    *,
    row_index: int,
    text: str,
    group_best: float,
    query: str,
    static_descriptions: tuple[str, ...],
    static_weights: tuple[float, ...],
    dynamic_descriptions: tuple[str, ...],
    dynamic_weights: tuple[float, ...],
    candidate_final_answer: Callable[[Any], tuple[str | None, str]],
    call_judge: Callable[..., Any],
    aggregate_reward: Callable[[Sequence[int], Sequence[float]], float],
    give_up_exception: type[BaseException],
    log: Callable[[str], None],
    teacher_reward_feedback: dict | None = None,
    build_judge_prompt: Callable[..., str] | None = None,
    judge_model: str | None = None,
) -> _AnchorVerdict:
    final_answer, extraction_status = candidate_final_answer(text)
    if final_answer is None:
        return _AnchorVerdict(
            keep=False,
            kind="invalid",
            row_index=row_index,
            text=text,
            detail=f"extraction_status={extraction_status}",
        )
    static_scores: tuple[int, ...] = ()
    dynamic_scores: tuple[int, ...] = ()
    cached = _cached_teacher_scores(
        teacher_reward_feedback,
        query=query,
        final_answer=final_answer,
        static_descriptions=static_descriptions,
        static_weights=static_weights,
        dynamic_descriptions=dynamic_descriptions,
        dynamic_weights=dynamic_weights,
        build_judge_prompt=build_judge_prompt,
        judge_model=judge_model,
    )
    if cached is not None:
        static_scores, dynamic_scores = cached
    try:
        if cached is None and static_descriptions:
            static_scores = tuple(
                call_judge(
                    query=query,
                    final_answer=final_answer,
                    titles=None,
                    descriptions=static_descriptions,
                    polarities=tuple(
                        (
                            "positive" if weight > 0.0 else "negative"
                            for weight in static_weights
                        )
                    ),
                ).scores
            )
        if cached is None and dynamic_descriptions:
            dynamic_scores = tuple(
                call_judge(
                    query=query,
                    final_answer=final_answer,
                    titles=None,
                    descriptions=dynamic_descriptions,
                    polarities=tuple(
                        (
                            "positive" if weight > 0.0 else "negative"
                            for weight in dynamic_weights
                        )
                    ),
                ).scores
            )
    except give_up_exception as error:
        return _AnchorVerdict(
            keep=False,
            kind="give_up",
            row_index=row_index,
            text=text,
            detail=f"give_up: {type(error).__name__}",
        )
    except Exception as error:
        log(
            f"TEACHER_GATE_ERROR uid_row={row_index} error={type(error).__name__}: {error}"
        )
        return _AnchorVerdict(
            keep=False,
            kind="give_up",
            row_index=row_index,
            text=text,
            detail=f"unexpected: {type(error).__name__}",
        )
    total = aggregate_reward(
        list(static_scores) + list(dynamic_scores),
        list(static_weights) + list(dynamic_weights),
    )
    if total > group_best:
        return _AnchorVerdict(
            keep=True,
            kind="pass",
            row_index=row_index,
            text=text,
            detail=f"teacher_total={total:.6g} > group_best={group_best:.6g}",
            score=total,
            reused=cached is not None,
        )
    return _AnchorVerdict(
        keep=False,
        kind="reject_not_better",
        row_index=row_index,
        text=text,
        detail=f"teacher_total={total:.6g} <= group_best={group_best:.6g}",
        score=total,
        reused=cached is not None,
    )


def filter_anchors_by_teacher_verification(
    *,
    anchor_indices: Sequence[int],
    teacher_response_texts: Sequence[str],
    uids: Sequence[Any],
    raw_prompts: Sequence[Any],
    rubric_descriptions: Sequence[Sequence[str]],
    rubric_weights: Sequence[Sequence[float]],
    raw_scores: Sequence[float],
    dynamic_rubric_descriptions: Sequence[Sequence[str]] | None = None,
    dynamic_rubric_weights: Sequence[Sequence[float]] | None = None,
    candidate_final_answer: Callable[[Any], tuple[str | None, str]],
    call_judge: Callable[..., Any],
    aggregate_reward: Callable[[Sequence[int], Sequence[float]], float],
    give_up_exception: type[BaseException],
    teacher_k: int = 1,
    max_workers: int = 8,
    log: Callable[[str], None] = print,
    teacher_reward_feedback: Sequence[dict | None] | None = None,
    build_judge_prompt: Callable[..., str] | None = None,
    judge_model: str | None = None,
) -> tuple[list[int], list[str], TeacherGateStats, list[dict[str, object]]]:
    lengths = {
        len(uids),
        len(raw_prompts),
        len(rubric_descriptions),
        len(rubric_weights),
        len(raw_scores),
    }
    if len(lengths) != 1 or not uids:
        raise ValueError("batch-aligned gate inputs must be non-empty and aligned")
    if len(anchor_indices) != len(teacher_response_texts):
        raise ValueError("anchor_indices and teacher_response_texts must be parallel")
    if teacher_reward_feedback is not None and len(teacher_reward_feedback) != len(
        anchor_indices
    ):
        raise ValueError(
            "teacher feedback must align with the retained teacher samples"
        )
    if type(teacher_k) is not int or teacher_k < 1:
        raise ValueError("teacher_k must be a positive integer")
    if teacher_k > 1 and len(anchor_indices) % teacher_k:
        raise ValueError(
            f"K-major gate inputs must be K-aligned (teacher_k={teacher_k}, got {len(anchor_indices)} samples)"
        )
    if dynamic_rubric_descriptions is not None and (
        len(dynamic_rubric_descriptions) != len(uids)
        or dynamic_rubric_weights is None
        or len(dynamic_rubric_weights) != len(uids)
    ):
        raise ValueError("dynamic rubric inputs must be batch-aligned with uids")
    group_best = compute_group_best(uids, raw_scores)

    def verify(index_in_list: int) -> _AnchorVerdict:
        row_index = anchor_indices[index_in_list]
        text = teacher_response_texts[index_in_list]
        uid = uids[row_index]
        try:
            query = _query_from_raw_prompt(raw_prompts[row_index])
        except (TypeError, ValueError) as error:
            return _AnchorVerdict(
                keep=False,
                kind="invalid",
                row_index=row_index,
                text=text,
                detail=f"query: {type(error).__name__}",
            )
        dynamic_desc = (
            tuple(dynamic_rubric_descriptions[row_index])
            if dynamic_rubric_descriptions is not None
            else ()
        )
        dynamic_weights = (
            tuple(dynamic_rubric_weights[row_index])
            if dynamic_rubric_weights is not None
            else ()
        )
        if bool(dynamic_desc) != bool(dynamic_weights):
            return _AnchorVerdict(
                keep=False,
                kind="invalid",
                row_index=row_index,
                text=text,
                detail="dynamic rubric desc/weight channels misaligned",
            )
        verdict = _verify_one_anchor(
            row_index=row_index,
            text=text,
            group_best=group_best[uid],
            query=query,
            static_descriptions=tuple(rubric_descriptions[row_index]),
            static_weights=tuple(rubric_weights[row_index]),
            dynamic_descriptions=dynamic_desc,
            dynamic_weights=dynamic_weights,
            candidate_final_answer=candidate_final_answer,
            call_judge=call_judge,
            aggregate_reward=aggregate_reward,
            give_up_exception=give_up_exception,
            log=log,
            teacher_reward_feedback=(
                teacher_reward_feedback[index_in_list]
                if teacher_reward_feedback is not None
                else None
            ),
            build_judge_prompt=build_judge_prompt,
            judge_model=judge_model,
        )
        if not verdict.keep:
            log(
                f"TEACHER_GATE reject uid={uid!r} row={row_index} reason={verdict.kind} {verdict.detail}"
            )
        return verdict

    verdicts: list[_AnchorVerdict | None] = [None] * len(anchor_indices)
    if anchor_indices:
        with ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="teacher_gate"
        ) as executor:
            verdicts = list(executor.map(verify, range(len(anchor_indices))))
    kept_indices: list[int] = []
    kept_texts: list[str] = []
    counts = {"pass": 0, "reject_not_better": 0, "give_up": 0, "invalid": 0}
    pass_details: list[dict[str, object]] = []
    for group_start in range(0, len(verdicts), teacher_k):
        group = verdicts[group_start : group_start + teacher_k]
        if any((verdict is None or not verdict.keep for verdict in group)):
            if any(
                (v is not None and v.kind == "invalid" and (not v.keep) for v in group)
            ):
                counts["invalid"] += 1
            elif any(
                (v is not None and v.kind == "give_up" and (not v.keep) for v in group)
            ):
                counts["give_up"] += 1
            else:
                counts["reject_not_better"] += 1
            continue
        best = max(group, key=lambda verdict: verdict.score)
        kept_indices.append(best.row_index)
        kept_texts.append(best.text)
        counts["pass"] += 1
        pass_details.append(
            {
                "uid": uids[best.row_index],
                "row": best.row_index,
                "teacher_total": best.score,
                "group_best": group_best[uids[best.row_index]],
                "teacher_k": teacher_k,
                "kept_text": best.text,
                "score_source": "reward_loop" if best.reused else "gate_judge",
            }
        )
    stats = TeacherGateStats(
        total=len(verdicts) // teacher_k,
        passed=counts["pass"],
        rejected_not_better=counts["reject_not_better"],
        rejected_judge_give_up=counts["give_up"],
        invalid_final_answer=counts["invalid"],
        reused_samples=sum((v.reused for v in verdicts if v is not None)),
        rescored_samples=sum(
            (not v.reused and v.kind != "invalid" for v in verdicts if v is not None)
        ),
    )
    return (kept_indices, kept_texts, stats, pass_details)
