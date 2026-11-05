def validate_scored_row(row, truth, dataset_sha, *, step, model, judge):
    if (
        type(step) is not int
        or step < 0
        or not isinstance(model, str)
        or not model
        or not isinstance(judge, str)
        or not judge
    ):
        raise ValueError("checkpoint, model and judge identities are required")
    if (
        row.get("checkpoint_step") != step
        or row.get("model_path") != model
        or row.get("judge_model") != judge
    ):
        raise ValueError("scored result identity mismatch")
    if row.get("input_sha256") != dataset_sha or row.get("query") != truth["query"]:
        raise ValueError("scored result dataset identity mismatch")
    rubrics = truth["rubrics"]
    for field, key in [("rubric_ids", "rubric_key"), ("rubric_weights", "weight")]:
        if row.get(field) != [r[key] for r in rubrics]:
            raise ValueError("scored rubric identities or weights changed")
    if not isinstance(row.get("rubrics"), list) or [
        r.get("description") for r in row["rubrics"]
    ] != [r["description"] for r in rubrics]:
        raise ValueError("scored rubric order or descriptions changed")
    if row.get("judge_valid") is not True:
        raise ValueError("invalid judgment cannot enter the aggregate")


def validate_rollout_row(row, truth, dataset_sha, *, step, model, generation):
    if row.get("input_sha256") != dataset_sha or row.get("query") != truth["query"]:
        raise ValueError("rollout dataset identity mismatch")
    if row.get("checkpoint_step") != step or row.get("model_path") != model:
        raise ValueError("rollout checkpoint identity mismatch")
    if row.get("generation") != generation:
        raise ValueError("rollout generation settings mismatch")
    if not isinstance(row.get("response"), str) or not row["response"].strip():
        raise ValueError("rollout contains an empty response")
