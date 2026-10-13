from __future__ import annotations
import argparse
import hashlib
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_DATASET_SHA256 = __import__("os").environ.get(
    "VERIFY_DATASET_SHA",
    "e2b2321149d9b095c4b4af565d1c7b470b86dce872b7f64f31a6195ed549786d",
)
VERIFY_PROMPT_TOKEN_CAP = int(
    __import__("os").environ.get("VERIFY_MAX_PROMPT_TOKENS", "1024")
)
VERIFY_PROTOCOL = __import__("os").environ.get(
    "VERIFY_PROTOCOL", "rar_medicine_ruact_sampling_v2"
)
EXPECTED_GENERATION = {
    "enable_thinking": False,
    "n": 1,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": -1,
    "max_prompt_tokens": VERIFY_PROMPT_TOKEN_CAP,
    "max_tokens": 4096,
    "max_model_len": int(os.environ.get("VERIFY_MAX_MODEL_LEN", "6144")),
    "seed": 42,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(model: Path) -> dict:
    weight_paths = sorted(model.glob("*.safetensors"))
    if not weight_paths:
        raise RuntimeError("model has no safetensors weights")
    files = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        for path in weight_paths
    ]
    digest = hashlib.sha256()
    for item in files:
        digest.update(
            f"{item['name']}\x00{item['size']}\x00{item['sha256']}\n".encode()
        )
    return {"files": files, "manifest_sha256": digest.hexdigest()}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def expected_ids(dataset: Path) -> set[str]:
    import pyarrow.parquet as pq

    if sha256(dataset) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("unexpected frozen dataset SHA256")
    source = pq.read_table(dataset).to_pylist()
    ids = [str(row["reward_model"]["ground_truth"]["prompt_id"]) for row in source]
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise RuntimeError(
            f"frozen dataset must contain unique prompt IDs (got {len(ids)})"
        )
    return set(ids)


def verify_rollout(args: argparse.Namespace) -> None:
    ids_expected = expected_ids(args.dataset)
    rows = read_jsonl(args.responses)
    ids = [str(row["prompt_id"]) for row in rows]
    if (
        len(rows) != len(ids_expected)
        or len(set(ids)) != len(rows)
        or set(ids) != ids_expected
    ):
        raise RuntimeError(
            f"rollout coverage mismatch rows={len(rows)} unique={len(set(ids))}"
        )
    if {int(row["checkpoint_step"]) for row in rows} != {args.step}:
        raise RuntimeError("checkpoint identity mismatch")
    if {str(row["model_path"]) for row in rows} != {str(args.model)}:
        raise RuntimeError("model path mismatch")
    if any((int(row["prompt_token_count"]) > VERIFY_PROMPT_TOKEN_CAP for row in rows)):
        raise RuntimeError("rollout contains an overlong prompt")
    summary = json.loads(args.summary.read_text())
    if (
        int(summary.get("covered_rows", -1)) != len(ids_expected)
        or int(summary.get("pending_rows", -1)) != 0
    ):
        raise RuntimeError("rollout summary is incomplete")
    expected_generation = {**EXPECTED_GENERATION, "seed": args.seed}
    if summary.get("generation") != expected_generation:
        raise RuntimeError(f"generation contract mismatch: {summary.get('generation')}")
    if summary.get("input_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("rollout dataset hash mismatch")
    finish_reasons: dict[str, int] = {}
    for row in rows:
        key = str(row.get("finish_reason"))
        finish_reasons[key] = finish_reasons.get(key, 0) + 1
    report = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "stage": "rollout_verified_before_judge",
        "protocol": VERIFY_PROTOCOL,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "generation": expected_generation,
        "step": args.step,
        "rows": len(rows),
        "unique_prompt_ids": len(set(ids)),
        "model_path": str(args.model),
        "model_weights": model_manifest(args.model),
        "responses_sha256": sha256(args.responses),
        "response_tokens_total": sum(
            (int(row["response_token_count"]) for row in rows)
        ),
        "finish_reason_counts": finish_reasons,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


def metric_domain(protocol: str, domain: str | None = None) -> str:
    if domain not in (None, "medicine", "science"):
        raise RuntimeError(f"unsupported metric domain: {domain!r}")
    return domain or (
        "science" if "science" in protocol.lower().split("_") else "medicine"
    )


def select_primary_metric(
    summary: dict, *, protocol: str, domain: str | None = None
) -> tuple[str, float]:
    domain = metric_domain(protocol, domain)
    if "score_mean_original" in summary:
        name = "score_mean_original"
    elif domain == "science":
        name = "score_mean"
    else:
        raise RuntimeError(
            "medicine final audit requires score_mean_original; run official aggregation first"
        )
    value = summary.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
    ):
        raise RuntimeError(f"invalid or missing primary metric {name}: {value!r}")
    return (name, float(value))


def verified_generation(rollout_audit: dict, *, step: int, rows: int) -> dict:
    identity = {
        "stage": "rollout_verified_before_judge",
        "protocol": VERIFY_PROTOCOL,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "step": step,
        "rows": rows,
        "unique_prompt_ids": rows,
    }
    for name, expected in identity.items():
        if rollout_audit.get(name) != expected:
            raise RuntimeError(
                f"rollout audit {name} mismatch: {rollout_audit.get(name)!r}"
            )
    generation = rollout_audit.get("generation")
    if not isinstance(generation, dict) or type(generation.get("seed")) is not int:
        raise RuntimeError("rollout audit generation must contain an integer seed")
    if generation != {**EXPECTED_GENERATION, "seed": generation["seed"]}:
        raise RuntimeError(
            f"rollout audit generation contract mismatch: {generation!r}"
        )
    return dict(generation)


def verify_final(args: argparse.Namespace) -> None:
    ids_expected = expected_ids(args.dataset)
    rows = read_jsonl(args.results)
    ids = [str(row["prompt_id"]) for row in rows]
    if (
        len(rows) != len(ids_expected)
        or len(set(ids)) != len(rows)
        or set(ids) != ids_expected
    ):
        raise RuntimeError(
            f"result coverage mismatch rows={len(rows)} unique={len(set(ids))}"
        )
    if {int(row["checkpoint_step"]) for row in rows} != {args.step}:
        raise RuntimeError("result checkpoint identity mismatch")
    if {str(row["judge_model"]) for row in rows} != {args.judge_model}:
        raise RuntimeError("Judge identity mismatch")
    if any((not bool(row["judge_valid"]) for row in rows)):
        raise RuntimeError("results contain invalid Judge rows")
    for row in rows:
        rubrics = row["rubrics"]
        rubric_ids = row["rubric_ids"]
        scores = row["rubric_scores"]
        weights = row["rubric_weights"]
        if not len(rubrics) == len(rubric_ids) == len(scores) == len(weights):
            raise RuntimeError("rubric vector length mismatch")
        allow_negative = os.environ.get("VERIFY_ALLOW_NEGATIVE_RUBRIC_WEIGHTS") == "1"
        if not weights or any(
            (
                not math.isfinite(float(weight))
                or float(weight) == 0
                or (float(weight) < 0 and (not allow_negative))
                for weight in weights
            )
        ):
            raise RuntimeError("invalid rubric weights")
        if any((int(score) not in (0, 1) for score in scores)):
            raise RuntimeError("non-binary rubric decision")
    summary = json.loads(args.summary.read_text())
    if int(summary.get("completed_rows", -1)) != len(ids_expected):
        raise RuntimeError("Judge summary is incomplete")
    if summary.get("input_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Judge summary dataset mismatch")
    judge = summary.get("judge") or {}
    if (
        judge.get("model") != args.judge_model
        or judge.get("enable_thinking") is not False
    ):
        raise RuntimeError("Judge contract mismatch")
    rollout_audit = json.loads(args.rollout_audit.read_text())
    generation = verified_generation(rollout_audit, step=args.step, rows=len(rows))
    domain = metric_domain(VERIFY_PROTOCOL, getattr(args, "metric_domain", None))
    primary_name, primary_value = select_primary_metric(
        summary, protocol=VERIFY_PROTOCOL, domain=domain
    )
    report = {
        "schema_version": 2,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "protocol": VERIFY_PROTOCOL,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "generation": generation,
        "judge": {
            "model": args.judge_model,
            "enable_thinking": False,
            "mode": "ruact_binary_independent_response",
        },
        "step": args.step,
        "rows": len(rows),
        "unique_prompt_ids": len(set(ids)),
        "metric_domain": domain,
        "primary_metric_name": primary_name,
        "primary_metric_value": primary_value,
        "formal_score_scale": 100.0,
        "formal_score": primary_value * 100.0,
        "judge_attempts": int(summary["judge_attempts"]),
        "judge_calls": int(summary["judge_calls"]),
        "finish_reason_counts": summary["finish_reason_counts"],
        "extraction_status_counts": summary["extraction_status_counts"],
        "response_tokens_mean": float(summary["response_tokens_mean"]),
        "results_sha256": sha256(args.results),
        "rollout_audit": rollout_audit,
    }
    if domain == "medicine":
        report["same_protocol_references_metric_name"] = "score_mean"
        report["same_protocol_references_scale"] = 100.0
        report["same_protocol_references"] = {
            "vanilla_grpo_step50": 34.608034440939825,
            "vanilla_grpo_step70": 35.16795062030673,
            "vanilla_grpo_step100": 38.11607090210171,
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--dataset", type=Path, required=True)
    rollout.add_argument("--step", type=int, required=True)
    rollout.add_argument("--model", type=Path, required=True)
    rollout.add_argument("--responses", type=Path, required=True)
    rollout.add_argument("--summary", type=Path, required=True)
    rollout.add_argument("--seed", type=int, default=42)
    rollout.add_argument("--output", type=Path, required=True)
    rollout.set_defaults(func=verify_rollout)
    final = subparsers.add_parser("final")
    final.add_argument("--dataset", type=Path, required=True)
    final.add_argument("--step", type=int, required=True)
    final.add_argument("--results", type=Path, required=True)
    final.add_argument("--summary", type=Path, required=True)
    final.add_argument("--rollout-audit", type=Path, required=True)
    final.add_argument("--judge-model", required=True)
    final.add_argument(
        "--metric-domain",
        choices=("medicine", "science"),
        default=os.environ.get("VERIFY_METRIC_DOMAIN"),
        help="Override domain detection for legacy protocol labels; medicine requires original scores",
    )
    final.add_argument("--output", type=Path, required=True)
    final.set_defaults(func=verify_final)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
