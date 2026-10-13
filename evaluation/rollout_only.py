from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def read_ids(paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prompt_id = str(json.loads(line)["prompt_id"])
                if prompt_id in seen:
                    raise ValueError(
                        f"duplicate prompt_id {prompt_id} in inputs through {path}:{line_number}"
                    )
                seen.add(prompt_id)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--existing-results", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = pq.read_table(args.input_parquet).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eligible = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if prompt_tokens > args.max_prompt_tokens:
            raise RuntimeError(
                f"overlong prompt {row['reward_model']['ground_truth']['prompt_id']}"
            )
        eligible.append((row, rendered, prompt_tokens))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = read_ids([*args.existing_results, args.output_jsonl])
    cohort_ids = {
        str(row[0]["reward_model"]["ground_truth"]["prompt_id"]) for row in eligible
    }
    if not completed <= cohort_ids:
        raise RuntimeError("existing results contain IDs outside this rollout cohort")
    pending = [
        item
        for item in eligible
        if str(item[0]["reward_model"]["ground_truth"]["prompt_id"]) not in completed
    ]
    print(
        json.dumps(
            {
                "event": "rollout_start",
                "cohort_rows": len(eligible),
                "already_completed": len(completed),
                "pending": len(pending),
                "enable_thinking": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        return 0
    llm = LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        min_tokens=1,
        seed=args.seed,
    )
    started = time.monotonic()
    generated = 0
    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            outputs = llm.generate(
                [item[1] for item in batch], sampling, use_tqdm=False
            )
            for (row, rendered, prompt_tokens), request_output in zip(
                batch, outputs, strict=True
            ):
                if len(request_output.outputs) != 1:
                    raise RuntimeError("evaluation requires exactly one completion")
                completion = request_output.outputs[0]
                ground_truth = row["reward_model"]["ground_truth"]
                record = {
                    "schema_version": 1,
                    "stage": "rollout_only",
                    "checkpoint_step": args.checkpoint_step,
                    "model_path": str(args.model_path),
                    "prompt_id": str(ground_truth["prompt_id"]),
                    "query": str(ground_truth["query"]),
                    "rendered_prompt_sha256": hashlib.sha256(
                        rendered.encode()
                    ).hexdigest(),
                    "response": completion.text,
                    "prompt_token_count": prompt_tokens,
                    "response_token_count": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "reference_answer": row["extra_info"].get("reference_answer"),
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                generated += 1
            summary = {
                "schema_version": 1,
                "stage": "rollout_only",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": args.run_id,
                "input_parquet": str(args.input_parquet),
                "input_sha256": hashlib.sha256(
                    args.input_parquet.read_bytes()
                ).hexdigest(),
                "model_path": str(args.model_path),
                "checkpoint_step": args.checkpoint_step,
                "cohort_rows": len(eligible),
                "preexisting_rows": len(completed),
                "generated_rows": generated,
                "covered_rows": len(completed) + generated,
                "pending_rows": len(eligible) - len(completed) - generated,
                "generation": {
                    "enable_thinking": False,
                    "n": 1,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                    "seed": args.seed,
                },
            }
            temporary = args.summary_json.with_suffix(args.summary_json.suffix + ".tmp")
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, args.summary_json)
            print(
                json.dumps(
                    {
                        "event": "rollout_progress",
                        "elapsed_seconds": time.monotonic() - started,
                        **summary,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {"event": "rollout_complete", "generated_rows": generated}, sort_keys=True
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
