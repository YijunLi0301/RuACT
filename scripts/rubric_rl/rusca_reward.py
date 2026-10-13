from functools import partial
from rubric_rl import verl_reward as common

common._call_judge = partial(
    common._call_judge, max_total_attempts=3, strict_response=True
)


def compute_score_stream_single(data_source, solution_str, ground_truth, extra_info):
    row = common.compute_score_stream_single(
        data_source, solution_str, ground_truth, extra_info
    )
    if not row["judge_valid"]:
        raise RuntimeError("RuscaRL judge failed; refusing synthetic rewards")
    return row
