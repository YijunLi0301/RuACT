from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from evaluation.contract import validate_scored_row, validate_rollout_row


def read_map(paths: list[Path]) -> dict[str, dict]:
    records = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt_id = str(record["prompt_id"])
                if prompt_id in records:
                    raise ValueError(
                        f"duplicate response {prompt_id} at {path}:{line_number}"
                    )
                records[prompt_id] = record
    return records


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return (
        ordered[lo]
        if lo == hi
        else ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo)
    )


def write_summary(output: Path, summary_path: Path, metadata: dict) -> dict:
    rows = [
        json.loads(line) for line in output.read_text().splitlines() if line.strip()
    ]
    scores = [float(x["score"]) for x in rows]
    tokens = [int(x["response_token_count"]) for x in rows]
    extraction = {}
    finish = {}
    for row in rows:
        extraction[str(row["answer_extraction_status"])] = (
            extraction.get(str(row["answer_extraction_status"]), 0) + 1
        )
        finish[str(row.get("finish_reason"))] = (
            finish.get(str(row.get("finish_reason")), 0) + 1
        )
    result = {
        **metadata,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_rows": len(rows),
        "score_mean": statistics.fmean(scores) if scores else None,
        "score_std_population": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "score_median": statistics.median(scores) if scores else None,
        "score_q25": quantile(scores, 0.25) if scores else None,
        "score_q75": quantile(scores, 0.75) if scores else None,
        "score_zero_rate": (
            sum((x == 0 for x in scores)) / len(scores) if scores else None
        ),
        "score_perfect_rate": (
            sum((x == 1 for x in scores)) / len(scores) if scores else None
        ),
        "response_tokens_mean": statistics.fmean(tokens) if tokens else None,
        "response_tokens_total": sum(tokens),
        "judge_calls": sum((bool(x["judge_called"]) for x in rows)),
        "judge_attempts": sum((int(x["judge_attempts"]) for x in rows)),
        "extraction_status_counts": extraction,
        "finish_reason_counts": finish,
    }
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, summary_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--responses-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-summary", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from evaluation.judge import compute_score_stream_chunk

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    rollout_summary = json.loads(args.rollout_summary.read_text())
    dataset_sha = hashlib.sha256(args.input_parquet.read_bytes()).hexdigest()
    if rollout_summary.get("input_sha256") != dataset_sha:
        raise RuntimeError("rollout dataset hash mismatch")
    source = pq.read_table(args.input_parquet).to_pylist()
    ground = {
        str(row["reward_model"]["ground_truth"]["prompt_id"]): row["reward_model"][
            "ground_truth"
        ]
        for row in source
    }
    responses = read_map(args.responses_jsonl)
    ids = [str(r["reward_model"]["ground_truth"]["prompt_id"]) for r in source]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(responses):
        raise RuntimeError(
            f"response coverage mismatch source={len(ids)} responses={len(responses)} missing={len(set(ids) - responses)} extra={len(set(responses) - set(ids))}"
        )
    if (
        rollout_summary.get("covered_rows") != len(ids)
        or rollout_summary.get("pending_rows") != 0
    ):
        raise RuntimeError("rollout manifest is incomplete")
    if any(not str(row.get("response", "")).strip() for row in responses.values()):
        raise RuntimeError("rollout contains empty responses")
    steps = {int(r["checkpoint_step"]) for r in responses.values()}
    models = {str(r["model_path"]) for r in responses.values()}
    if steps != {args.checkpoint_step} or len(models) != 1:
        raise RuntimeError("response model identity mismatch")
    if (
        rollout_summary.get("checkpoint_step") != args.checkpoint_step
        or rollout_summary.get("model_path") not in models
    ):
        raise RuntimeError("rollout manifest identity mismatch")
    for pid, record in responses.items():
        validate_rollout_row(
            record,
            ground[pid],
            dataset_sha,
            step=args.checkpoint_step,
            model=next(iter(models)),
            generation=rollout_summary["generation"],
        )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output_jsonl.exists():
        existing = read_map([args.output_jsonl])
        for pid, row in existing.items():
            if pid not in ground:
                raise RuntimeError("existing score contains an unknown prompt")
            validate_scored_row(
                row,
                ground[pid],
                dataset_sha,
                step=args.checkpoint_step,
                model=next(iter(models)),
                judge=os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
            )
            if (
                pid not in responses
                or row.get("response") != responses[pid]["response"]
            ):
                raise RuntimeError("existing score does not match frozen responses")
            if row.get("checkpoint_step") != args.checkpoint_step or row.get(
                "judge_model"
            ) != os.environ.get("HEALTHBENCH_JUDGE_MODEL"):
                raise RuntimeError("existing score identity mismatch")
        completed = set(existing)
    pending = [
        row
        for row in source
        if str(row["reward_model"]["ground_truth"]["prompt_id"]) not in completed
    ]
    metadata = {
        "schema_version": 3,
        "stage": "unified_judge_after_rollout",
        "protocol": "rubric_binary",
        "run_id": args.run_id,
        "checkpoint_step": args.checkpoint_step,
        "model_path": next(iter(models)),
        "input_parquet": str(args.input_parquet),
        "input_sha256": hashlib.sha256(args.input_parquet.read_bytes()).hexdigest(),
        "input_rows_selected": len(source),
        "eligible_rows": len(source),
        "overlong_rows": 0,
        "generation": rollout_summary["generation"],
        "judge": {
            "model": os.environ.get("HEALTHBENCH_JUDGE_MODEL"),
            "enable_thinking": False,
            "mode": "ruact_binary_independent_response",
        },
    }
    print(
        json.dumps(
            {
                "event": "judge_start",
                "already_completed": len(completed),
                "pending": len(pending),
                **metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            scoring = []
            for row in batch:
                pid = str(row["reward_model"]["ground_truth"]["prompt_id"])
                response = responses[pid]
                extra = dict(row["extra_info"])
                extra.update(
                    {
                        "rollout_index": 0,
                        "training_step": args.checkpoint_step,
                        "run_id": args.run_id,
                    }
                )
                scoring.append((row, response, extra))
            rewards = compute_score_stream_chunk(
                [x[0]["data_source"] for x in scoring],
                [x[1]["response"] for x in scoring],
                [x[0]["reward_model"]["ground_truth"] for x in scoring],
                [x[2] for x in scoring],
            )
            for (row, response, extra), reward in zip(scoring, rewards, strict=True):
                record = {
                    "schema_version": 3,
                    "checkpoint_step": args.checkpoint_step,
                    "input_sha256": dataset_sha,
                    "model_path": response["model_path"],
                    "prompt_id": reward["prompt_id"],
                    "query": reward["query"],
                    "rendered_prompt_sha256": response["rendered_prompt_sha256"],
                    "response": response["response"],
                    "candidate_final_answer": reward["candidate_final_answer"],
                    "answer_extraction_status": reward["answer_extraction_status"],
                    "score": reward["score"],
                    "rubric_scores": reward["rubric_scores"],
                    "rubric_ids": reward["rubric_ids"],
                    "rubric_weights": reward["rubric_weights"],
                    "rubrics": reward["rubrics"],
                    "judge_valid": reward["judge_valid"],
                    "judge_called": reward["judge_called"],
                    "judge_model": reward["judge_model"],
                    "judge_attempts": reward["judge_attempts"],
                    "judge_prompt_sha256": reward["judge_prompt_sha256"],
                    "judge_latency_seconds": reward["judge_latency_seconds"],
                    "judge_mode": reward["judge_mode"],
                    "prompt_token_count": response["prompt_token_count"],
                    "response_token_count": response["response_token_count"],
                    "finish_reason": response.get("finish_reason"),
                    "stop_reason": response.get("stop_reason"),
                    "reference_answer": response.get("reference_answer"),
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
            summary = write_summary(args.output_jsonl, args.summary_json, metadata)
            print(
                json.dumps(
                    {
                        "event": "judge_progress",
                        "completed_rows": summary["completed_rows"],
                        "score_mean": summary["score_mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "event": "judge_complete",
                **write_summary(args.output_jsonl, args.summary_json, metadata),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
