from dataclasses import dataclass
from typing import List
import random
import numpy as np


@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: List[str]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        return cls(criterion=d["criterion"], points=d["points"], tags=d.get("tags", []))

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "points": self.points, "tags": self.tags}


def _generate_rubric_system_message(
    rubric_items: List[RubricItem],
    rubric_ratio: float,
    include_evaluation_warning: bool = True,
    rule_type: str = "linear",
) -> tuple[str, int]:
    if not rubric_items:
        return ("You are a helpful medical assistant.", 0)
    if rule_type == "math_linear":
        total_criteria = len(rubric_items)
        selected_rubric_items = []
        if total_criteria > 0:
            selected_rubric_items.append(rubric_items[-2])
        if rubric_ratio >= 1.0 and total_criteria > 1:
            selected_rubric_items.insert(0, rubric_items[-1])
        if total_criteria > 2:
            other_items = rubric_items[:-2]
            num_other_to_show = round(len(other_items) * rubric_ratio)
            if num_other_to_show > 0:
                selected_other_items = other_items[:num_other_to_show]
                selected_rubric_items = selected_other_items + selected_rubric_items
    elif rubric_ratio < 1.0:
        total_criteria = len(rubric_items)
        num_to_show = round(total_criteria * rubric_ratio)
        if num_to_show > 0:
            selected_rubric_items = random.sample(rubric_items, num_to_show)
        else:
            selected_rubric_items = []
    else:
        selected_rubric_items = rubric_items
    if not selected_rubric_items:
        return ("You are a helpful medical assistant.", 0)
    positive_points = []
    negative_points = []
    pos_count = 1
    neg_count = 1
    for rubric_item in selected_rubric_items:
        criterion = rubric_item.criterion
        points = rubric_item.points
        if points > 0:
            positive_points.append(
                f"Criterion {pos_count}: {criterion} (worth {points} points)"
            )
            pos_count += 1
        elif points < 0:
            negative_points.append(
                f"Criterion {neg_count}: {criterion} (penalty: {abs(points)} points)"
            )
            neg_count += 1
    rubric_message = "You are a helpful assistant. For this question, please consider the following evaluation criteria:\n\n"
    if positive_points:
        rubric_message += (
            "IMPORTANT POINTS TO INCLUDE (you should aim to address these):\n"
        )
        rubric_message += "\n".join(positive_points)
        rubric_message += "\n\n"
    if negative_points:
        rubric_message += "IMPORTANT POINTS TO AVOID (you should not do these):\n"
        rubric_message += "\n".join(negative_points)
        rubric_message += "\n\n"
    rubric_message += "Please provide a comprehensive and helpful response that addresses the user's concerns while following the above guidelines.\n\n"
    if include_evaluation_warning:
        rubric_message += "IMPORTANT: Do not mention or reference these evaluation criteria in your response. Do not indicate that you have seen any scoring rubric or evaluation guidelines. Your response should appear natural and spontaneous. Revealing that you have access to evaluation criteria would be considered cheating and is strictly prohibited."
    return (rubric_message, len(selected_rubric_items))


def _calculate_rubric_ratio(
    sample_idx: int,
    total_samples: int,
    rule_type: str = "linear",
    full_reveal_count: int = 1,
    global_steps: int = None,
    total_training_steps: int = None,
    step_power_n: float = 1.0,
    step_sigmoid_start_point: float = 0.3,
    step_sigmoid_steepness: float = 10.0,
    group_decay_type: str = "linear",
) -> float:
    if total_samples == 1:
        return 1.0
    if rule_type == "linear":
        return max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
    elif rule_type == "binary":
        full_reveal_count = min(full_reveal_count, total_samples)
        return 1.0 if sample_idx < full_reveal_count else 0.0
    elif rule_type == "math_linear":
        return max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
    elif rule_type == "step_linear":
        if group_decay_type == "binary":
            group_ratio = 1.0 if sample_idx < full_reveal_count else 0.0
        else:
            group_ratio = max(
                0.0, (total_samples - 1 - sample_idx) / (total_samples - 1)
            )
        if (
            global_steps is None
            or total_training_steps is None
            or total_training_steps <= 1
        ):
            step_ratio = 1.0
        else:
            step_ratio = max(
                0.0, (total_training_steps - global_steps) / (total_training_steps - 1)
            )
        combined_ratio = group_ratio * step_ratio
        return combined_ratio
    elif rule_type == "step_power":
        if group_decay_type == "binary":
            group_ratio = 1.0 if sample_idx < full_reveal_count else 0.0
        else:
            group_ratio = max(
                0.0, (total_samples - 1 - sample_idx) / (total_samples - 1)
            )
        if (
            global_steps is None
            or total_training_steps is None
            or total_training_steps <= 1
        ):
            step_ratio = 1.0
        else:
            x = min(1.0, max(0.0, (global_steps - 1) / (total_training_steps - 1)))
            step_ratio = max(0.0, (1.0 - x) ** step_power_n)
        combined_ratio = group_ratio * step_ratio
        return combined_ratio
    elif rule_type == "step_sigmoid":
        if group_decay_type == "binary":
            group_ratio = 1.0 if sample_idx < full_reveal_count else 0.0
        else:
            group_ratio = max(
                0.0, (total_samples - 1 - sample_idx) / (total_samples - 1)
            )
        if (
            global_steps is None
            or total_training_steps is None
            or total_training_steps <= 1
        ):
            step_ratio = 1.0
        else:
            x = min(1.0, max(0.0, (global_steps - 1) / (total_training_steps - 1)))
            shifted_x = (x - step_sigmoid_start_point) * step_sigmoid_steepness
            step_ratio = max(0.0, min(1.0, 1.0 / (1.0 + np.exp(shifted_x))))
        combined_ratio = group_ratio * step_ratio
        return combined_ratio
    else:
        return max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))


def scaffold_prompt(
    tokenizer,
    prompt_ids,
    reward_model,
    *,
    sample_index,
    group_size,
    step,
    total_steps,
    start=0.2,
    steepness=125.0,
):
    if isinstance(reward_model, dict) and "rubrics" not in reward_model:
        reward_model = reward_model.get("ground_truth", {})
    if not isinstance(reward_model, dict) or not isinstance(
        reward_model.get("rubrics"), (list, tuple, np.ndarray)
    ):
        raise ValueError("RuscaRL requires reward_model.rubrics")
    items = [
        RubricItem(
            criterion=str(
                item.get("criterion")
                or item.get("description")
                or item.get("title")
                or ""
            ),
            points=float(item.get("points", item.get("weight", 0))),
            tags=list(item.get("tags", [])),
        )
        for item in reward_model["rubrics"]
    ]
    ratio = _calculate_rubric_ratio(
        sample_index,
        group_size,
        "step_sigmoid",
        global_steps=step,
        total_training_steps=total_steps,
        step_sigmoid_start_point=start,
        step_sigmoid_steepness=steepness,
        group_decay_type="linear",
    )
    message, count = _generate_rubric_system_message(items, ratio, True, "step_sigmoid")
    if count == 0:
        return (list(prompt_ids), {"rubric_ratio": ratio, "rubric_count": 0})
    system_tokens = tokenizer.encode(
        f"<|im_start|>system\n{message}<|im_end|>\n", add_special_tokens=False
    )
    return (
        system_tokens + list(prompt_ids),
        {"rubric_ratio": ratio, "rubric_count": count},
    )
