# Data

The repository contains a format example and a preparation utility, not the full datasets or historical child-rubric collections. Do not use the example as experimental data.

The experiment profiles require Medicine and Science training and validation Parquet files. Preserve the original split assignment and sample IDs. A separate DeepRubric collection can use the same query/parent/child linkage, but no validated DeepRubric-7B launch recipe is shipped here.

## Preparing a split

The normalized JSONL contract is one object per query with `id`, `query`, and `rubrics`. Each rubric contains `description`, nonzero signed `weight`, and optional `title`. Optional `reference_answer` is preserved as metadata. `data/examples/format.jsonl` illustrates the format.

```bash
python data/prepare.py --input train.jsonl --output train.parquet --domain medicine --split train
python data/prepare.py --input validation.jsonl --output validation.parquet --domain medicine --split validation
```

Use `--domain science` for Science. The utility validates identities and weights, creates stable parent IDs, and packages the query as the public conversation. It does not deduplicate, resplit, truncate, download or synthesize dataset records. For already-prepared historical Parquet files, use them directly; changing IDs would break the frozen candidate-library mapping.

## Child-rubric records

Retain the query ID, parent rubric ID, parent description and weight, parent fingerprint, child completion point, source step and source response set. Accepted results also retain eight child judgments, pass counts, semantic-check status and regeneration audit. Distinguish all proposed candidates from children actually used in rewards; rejected or unscored proposals are not accepted children.

The frozen seed library consumed by `ruact/critic/seed_library.py` is JSON with `library_schema: "refinement-seed-library"` and an `entries` array. Each entry includes `query_id`, `parent_rubric_id`, `parent_title`, `parent_description`, `parent_weight`, `parent_fingerprint`, and `children`. Each child includes `title` and `completion_point`, with optional source metadata. Compute the fingerprint with `ruact.critic.protocol.parent_fingerprint`; do not substitute a title-only match. Libraries must be frozen before training and must not draw from evaluation splits.

To train without an existing candidate library, explicitly set `allow_empty_seed_library: true`. This changes initialization and must be reported accordingly.

## Evaluation data

Prepare evaluation data separately and preserve explicit signed weights. Training-time transformed weights and evaluation-time original weights are different artifacts and must not be substituted silently. The evaluation tools record dataset hashes and validate complete ID coverage. No split size or source-machine path is hard-coded into those checks.
