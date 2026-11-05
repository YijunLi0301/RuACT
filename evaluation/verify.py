import argparse
import hashlib
import json
from pathlib import Path
from evaluation.contract import validate_rollout_row


def verify(dataset, responses, summary, step):
    import pyarrow.parquet as pq

    source = pq.read_table(dataset).to_pylist()
    expected = [str(row["reward_model"]["ground_truth"]["prompt_id"]) for row in source]
    rows = [
        json.loads(line) for line in responses.read_text().splitlines() if line.strip()
    ]
    ids = [str(row["prompt_id"]) for row in rows]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or len(set(ids)) != len(ids)
        or set(ids) != set(expected)
    ):
        raise ValueError("rollout coverage or uniqueness mismatch")
    if any(
        row.get("checkpoint_step") != step or not str(row.get("response", "")).strip()
        for row in rows
    ):
        raise ValueError("checkpoint mismatch or empty response")
    metadata = json.loads(summary.read_text())
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if (
        metadata.get("input_sha256") != digest
        or metadata.get("pending_rows") != 0
        or metadata.get("covered_rows") != len(expected)
    ):
        raise ValueError("rollout manifest mismatch")
    if len({row["model_path"] for row in rows}) != 1 or rows[0][
        "model_path"
    ] != metadata.get("model_path"):
        raise ValueError("rollout model mismatch")
    generation = metadata["generation"]
    ground = {
        str(row["reward_model"]["ground_truth"]["prompt_id"]): row["reward_model"][
            "ground_truth"
        ]
        for row in source
    }
    if metadata.get("checkpoint_step") != step:
        raise ValueError("rollout manifest checkpoint mismatch")
    for row in rows:
        validate_rollout_row(
            row,
            ground[str(row["prompt_id"])],
            digest,
            step=step,
            model=metadata["model_path"],
            generation=generation,
        )
    if any(
        row["prompt_token_count"] > generation["max_prompt_tokens"]
        or row["response_token_count"] > generation["max_tokens"]
        for row in rows
    ):
        raise ValueError("record exceeds the declared token budget")
    truncated = sum(row.get("finish_reason") == "length" for row in rows)
    return {
        "rows": len(rows),
        "checkpoint_step": step,
        "dataset_sha256": digest,
        "responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        "length_terminated": truncated,
        "generation": generation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.dataset, args.responses, args.summary, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
