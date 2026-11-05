import pytest
from ruact.reward import (
    aggregate_weighted_reward,
    parse_batched_scores,
    rubric_passed,
    RewardContractError,
)


def test_signed_reward_uses_positive_weight_denominator():
    assert aggregate_weighted_reward([1, 1], [2, -1]) == 0.5
    assert aggregate_weighted_reward([0, 1], [2, -1]) == -0.5
    assert rubric_passed(-1, 0)
    assert not rubric_passed(-1, 1)


@pytest.mark.parametrize(
    "scores",
    [None, [{"id": 0, "score": 0.5}], [{"id": 0, "score": 1}, {"id": 0, "score": 0}]],
)
def test_missing_or_invalid_judgments_are_not_zero_rewards(scores):
    with pytest.raises(RewardContractError):
        parse_batched_scores(scores, 2)


def test_judgments_are_aligned_by_rubric_id():
    result = parse_batched_scores([{"id": 1, "score": 0}, {"id": 0, "score": 1}], 2)
    assert result.scores == (1, 0)


def test_prepared_rubrics_are_consumable_by_training_and_evaluation():
    from pathlib import Path
    from data.prepare import prepare
    from ruact.verl_reward import _validate_rubrics as training_contract
    from evaluation.judge import _validate_rubrics as evaluation_contract

    row = prepare(
        Path(__file__).resolve().parents[1] / "data/examples/format.jsonl",
        "medicine",
        "test",
    )[0]
    rubrics = row["reward_model"]["ground_truth"]["rubrics"]
    assert training_contract(rubrics)[:3] == evaluation_contract(rubrics)


def evaluation_fixture(tmp_path):
    import hashlib
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset, results = tmp_path / "test.parquet", tmp_path / "results.jsonl"
    source, scored = [], []
    for pid in ["q1", "q2"]:
        rubrics = [
            {"rubric_key": pid + ":r0", "description": "Required", "weight": 2},
            {"rubric_key": pid + ":r1", "description": "Undesired", "weight": -1},
        ]
        truth = {"prompt_id": pid, "query": "Question " + pid, "rubrics": rubrics}
        source.append({"reward_model": {"ground_truth": truth}})
        scored.append(
            {
                "prompt_id": pid,
                "query": truth["query"],
                "rubrics": rubrics,
                "rubric_ids": [r["rubric_key"] for r in rubrics],
                "rubric_weights": [2, -1],
                "rubric_scores": [1, 1],
                "judge_valid": True,
                "checkpoint_step": 200,
                "model_path": "model",
                "judge_model": "judge",
            }
        )
    pq.write_table(pa.Table.from_pylist(source), dataset)
    for row in scored:
        row["input_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    results.write_text("".join(json.dumps(row) + "\n" for row in scored))
    return dataset, results, source, scored


def test_evaluation_uses_explicit_weights_and_rejects_partial_cohorts(tmp_path):
    from evaluation.aggregate import aggregate

    dataset, results, _, _ = evaluation_fixture(tmp_path)
    assert aggregate(dataset, results)["score_percent"] == 50
    results.write_text("")
    with pytest.raises(ValueError, match="complete frozen cohort"):
        aggregate(dataset, results)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_step", 20),
        ("model_path", "other-model"),
        ("judge_model", "other-judge"),
        ("input_sha256", "other-dataset"),
        ("query", "Different question"),
        ("rubric_weights", [1, -1]),
    ],
)
def test_scoring_rejects_changed_source_and_mixed_identities(tmp_path, field, value):
    import json
    from evaluation.aggregate import aggregate
    from evaluation.contract import validate_scored_row

    dataset, results, source, scored = evaluation_fixture(tmp_path)
    digest = scored[0]["input_sha256"]
    scored[1][field] = value
    with pytest.raises(ValueError):
        validate_scored_row(
            scored[1],
            source[1]["reward_model"]["ground_truth"],
            digest,
            step=200,
            model="model",
            judge="judge",
        )
    results.write_text("".join(json.dumps(row) + "\n" for row in scored))
    with pytest.raises(ValueError):
        aggregate(dataset, results)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "changed"),
        ("query", "changed"),
        ("checkpoint_step", 20),
        ("model_path", "changed"),
        ("generation", {"seed": 43}),
    ],
)
def test_rollout_resume_rejects_changed_contract(tmp_path, field, value):
    import json
    from evaluation.rollout import read_ids

    row = {
        "prompt_id": "q",
        "query": "Question",
        "response": "Answer",
        "input_sha256": "dataset",
        "checkpoint_step": 200,
        "model_path": "model",
        "generation": {"seed": 42},
    }
    path = tmp_path / "responses.jsonl"
    ground = {"q": {"query": "Question"}}
    path.write_text(json.dumps(row) + "\n")
    assert read_ids([path], ground, "dataset", 200, "model", {"seed": 42}) == {"q"}
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        read_ids([path], ground, "dataset", 200, "model", {"seed": 42})
