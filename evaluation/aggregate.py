import argparse
import hashlib
import json
import statistics
from pathlib import Path
from ruact.reward import aggregate_weighted_reward
from evaluation.contract import validate_scored_row


def aggregate(dataset, results):
    import pyarrow.parquet as pq

    source = pq.read_table(dataset).to_pylist()
    ground = {
        str(r["reward_model"]["ground_truth"]["prompt_id"]): r["reward_model"][
            "ground_truth"
        ]
        for r in source
    }
    if not ground or len(ground) != len(source):
        raise ValueError("dataset IDs must be non-empty and unique")
    rows = [
        json.loads(line) for line in results.read_text().splitlines() if line.strip()
    ]
    ids = [str(row["prompt_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(ground):
        raise ValueError("scored results must match the complete frozen cohort")
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    step, model, judge = (
        rows[0].get(key) for key in ("checkpoint_step", "model_path", "judge_model")
    )
    values = []
    for row in rows:
        truth = ground[str(row["prompt_id"])]
        validate_scored_row(
            row, truth, dataset_sha, step=step, model=model, judge=judge
        )
        rubrics = truth["rubrics"]
        values.append(
            aggregate_weighted_reward(
                row["rubric_scores"], [r["weight"] for r in rubrics]
            )
        )
    return {
        "rows": len(values),
        "checkpoint_step": step,
        "model_path": model,
        "judge_model": judge,
        "weighting": "explicit",
        "score_mean": statistics.fmean(values),
        "score_percent": 100 * statistics.fmean(values),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.dataset, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
