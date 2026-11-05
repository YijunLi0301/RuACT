# Evaluation

Run from the repository root after installation. Supply a prepared test Parquet file following the data contract, an inference checkpoint with tokenizer files, and an explicitly selected judge service. Training checkpoints may need merging before inference; an inference export is not a complete optimizer checkpoint.

## Frozen rollout

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m evaluation.rollout \
  --input-parquet test.parquet --model-path model \
  --output-jsonl outputs/eval/responses.jsonl --summary-json outputs/eval/rollout.json \
  --checkpoint-step 200 --run-id evaluation --tensor-parallel-size 2

python -m evaluation.verify \
  --dataset test.parquet --responses outputs/eval/responses.jsonl \
  --summary outputs/eval/rollout.json --step 200 --output outputs/eval/coverage.json
```

The default generation contract is one response, no thinking, temperature 0.7, top-p 0.8, unrestricted top-k and seed 42. Prompt, response and context caps are 1,024, 4,096 and 6,144 tokens; change them explicitly for a different protocol. Overlong prompts fail rather than being silently dropped. Length-terminated responses are recorded, counted and retained. The rollout manifest and individual response records carry the actual parameters and dataset identity. Resuming rejects records from a different dataset, checkpoint or sampling contract; source paths must refer to immutable model exports. Older records without this identity information require a separate, explicit conversion rather than silent reuse.

## Judge and aggregate

Set `HEALTHBENCH_JUDGE_API_BASE`, `HEALTHBENCH_JUDGE_API_KEY` and `HEALTHBENCH_JUDGE_MODEL` privately. The API base ends in the compatible API prefix, not `/chat/completions`. `evaluation/judge.py` contains the evaluation prompt and binary-output parser; it is separate from training's routing and refinement logic.

```bash
python -m evaluation.score \
  --input-parquet test.parquet --responses-jsonl outputs/eval/responses.jsonl \
  --rollout-summary outputs/eval/rollout.json \
  --output-jsonl outputs/eval/judgments.jsonl --summary-json outputs/eval/judge.json \
  --checkpoint-step 200 --run-id evaluation

python -m evaluation.aggregate \
  --dataset test.parquet --results outputs/eval/judgments.jsonl \
  --output outputs/eval/metric.json
```

Scoring uses the frozen responses without new generation. Every rubric is independently marked PRESENT or NOT_PRESENT in a strict structured response. A negative-weight criterion records occurrence of the undesirable behavior, not its avoidance. Aggregation is `sum(score_i * weight_i) / sum(positive weights)`, averaged across questions without clipping. The explicit source weights, criterion order, complete coverage and unique IDs are checked; missing judgments cannot silently shorten the denominator.

The score summary preserves the rollout's actual generation settings. The final metric report identifies its weights and hashes. Judge request retries and response parsing failures are distinct from model rollout truncation. An empty final answer is explicitly recorded with no judge call. A failed API call must not be interpreted as successful zero scoring.

The released figure tables preserve the metrics of their historical evaluation records. Some used distinct aggregation or output-acceptance settings; see `analysis/README.md`. Running these commands is a new evaluation under the declared current protocol, not an assertion that every archived score was produced by this exact consolidated entrypoint.
