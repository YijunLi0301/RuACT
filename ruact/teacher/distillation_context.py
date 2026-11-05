from collections.abc import Sequence


def find_over_context_rows(
    prompt_lengths: Sequence[int], response_lengths: Sequence[int], context_limit: int
) -> tuple[int, ...]:
    if context_limit <= 0:
        raise ValueError(f"context_limit must be positive, got {context_limit}")
    if len(prompt_lengths) != len(response_lengths):
        raise ValueError("prompt_lengths and response_lengths must have equal length")
    return tuple(
        (
            index
            for index, (prompt_length, response_length) in enumerate(
                zip(prompt_lengths, response_lengths, strict=True)
            )
            if int(prompt_length) + int(response_length) > context_limit
        )
    )
