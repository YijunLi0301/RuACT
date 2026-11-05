from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThinkingTeacherTarget:
    thinking_prefix: str
    final_answer: str


def pad_indices_to_multiple(
    indices: Sequence[int], multiple: int
) -> tuple[list[int], int]:
    if type(multiple) is not int or multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    padded = list(indices)
    if not padded:
        return (padded, 0)
    padding_count = -len(padded) % multiple
    padded.extend([padded[-1]] * padding_count)
    return (padded, padding_count)


def split_thinking_teacher_response(text: str) -> ThinkingTeacherTarget:
    if not isinstance(text, str):
        raise TypeError("teacher response must be a string")
    close_tag = "</think>"
    close_index = text.find(close_tag)
    if close_index < 0:
        final_answer = text.strip()
        if not final_answer:
            raise ValueError("teacher response is empty")
        return ThinkingTeacherTarget(thinking_prefix="", final_answer=final_answer)
    if text.find(close_tag, close_index + len(close_tag)) >= 0:
        raise ValueError("thinking teacher response contains multiple </think> tags")
    prefix_end = close_index + len(close_tag)
    thinking_prefix = text[:prefix_end].strip()
    final_answer = text[prefix_end:].strip()
    if not thinking_prefix.startswith("<think>"):
        raise ValueError("thinking teacher response must start with <think>")
    if not final_answer:
        raise ValueError("thinking teacher response has an empty final answer")
    if "<think>" in final_answer or "</think>" in final_answer:
        raise ValueError("final-only target still contains thinking tags")
    return ThinkingTeacherTarget(
        thinking_prefix=f"{thinking_prefix}\n\n", final_answer=final_answer
    )
