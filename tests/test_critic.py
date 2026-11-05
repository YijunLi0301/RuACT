import asyncio
import copy
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from ruact.critic import bounded_retry, worker
from ruact.critic.protocol import (
    BLEND_LAMBDA,
    REWARD_MODE,
    SCHEMA_VERSION,
    seed_bundle_hash,
    parent_fingerprint,
    ordered_response_set_hash,
    refinement_transaction_id,
    PositiveRefinementProtocolError,
)
from ruact.critic.registry import META_CHECKS, PositiveRefinementRegistry


def make_job():
    parents = [
        dict(
            rubric_id=f"p{i}",
            title=f"Parent{i}",
            description=f"Requirement{i}",
            parent_fingerprint=f"fingerprint{i}",
            weight=i + 1,
            pass_count=8,
            seed_candidates=[],
        )
        for i in range(3)
    ]
    job = dict(
        schema_version=SCHEMA_VERSION,
        job_id="job",
        transaction_id="job",
        query_id="query",
        query="Explain the mechanism.",
        group_id="group",
        response_set_hash="response-hash",
        query_version=0,
        gate_mode="saturated_positive_weight_fraction",
        generator_input_mode="query_and_parent_only",
        reward_mode=REWARD_MODE,
        child_activation_mode="all_meta_valid_fixed_weights",
        rollout_pass_rate_filtering=False,
        group_members=[
            dict(id=f"r{i}", rollout_index=i, text=f"ANSWER_{i}") for i in range(8)
        ],
        selected_parents=parents,
        seed_bundle_hash=seed_bundle_hash(parents),
        min_candidate_count=2,
        max_candidate_count=5,
        min_children_per_parent=1,
        max_children_per_parent=3,
        blend_lambda=BLEND_LAMBDA,
        group_reward_std=0.1,
        zero_std_epsilon=1e-08,
        zero_std_gate=False,
    )
    return freeze_job(job)


def freeze_job(job):
    job.update(run_id="test", global_step=1)
    for parent in job["selected_parents"]:
        parent["parent_fingerprint"] = parent_fingerprint(
            parent["rubric_id"],
            parent["title"],
            parent["description"],
            parent["weight"],
        )
    job["response_set_hash"] = ordered_response_set_hash(job["group_members"])
    job["seed_bundle_hash"] = seed_bundle_hash(job["selected_parents"])
    job["job_id"] = job["transaction_id"] = refinement_transaction_id(
        run_id=job["run_id"],
        global_step=job["global_step"],
        group_id=job["group_id"],
        query_id=job["query_id"],
        query_version=job["query_version"],
        response_set_hash=job["response_set_hash"],
        parent_fingerprints=[p["parent_fingerprint"] for p in job["selected_parents"]],
        seed_bundle_hash=job["seed_bundle_hash"],
        blend_lambda=job["blend_lambda"],
    )
    return job


def one_parent_job(minimum=1):
    job = make_job()
    job["selected_parents"] = job["selected_parents"][:1]
    job["min_children_per_parent"] = minimum
    return freeze_job(job)


def run_job(
    monkeypatch,
    *,
    patterns=None,
    job=None,
    config=None,
    fail_topup=False,
    fail_topup_judge=False,
    meta_reject=(),
    mode=None,
    repeat=False,
):
    job = job or make_job()
    patterns = patterns or [[2, 3, 4, 5, 6]]
    calls, judged, counts = [], [], {}

    async def generate(prompt, schema, *, config):
        if "results" in schema["properties"]:
            calls.append(("meta", prompt))
            candidates, _ = json.JSONDecoder().raw_decode(
                prompt.split("Candidates:\n", 1)[1]
            )
            indices = [c["candidate_index"] for c in candidates]
            return {
                "results": [
                    dict(
                        candidate_index=i,
                        decision="reject" if i in meta_reject else "accept",
                        **{c: i not in meta_reject for c in META_CHECKS},
                    )
                    for i in indices
                ]
            }, {"attempts": 1}
        parent = next(
            p for p in job["selected_parents"] if "Title: " + p["title"] in prompt
        )
        key = parent["rubric_id"]
        counts[key] = counts.get(key, 0) + 1
        round_index = counts[key] - 1
        calls.append(("generate", prompt))
        if fail_topup and round_index:
            raise PositiveRefinementProtocolError("invalid generated batch")
        count = schema["properties"]["candidates"]["maxItems"]
        return {
            "candidates": [
                dict(
                    title=f"Point {i}",
                    completion_point=f"{key} round {0 if repeat else round_index} point {i}",
                )
                for i in range(count)
            ]
        }, {"attempts": 1}

    async def judge(job, candidates, *, workers, attempts):
        judged.append((copy.deepcopy(job["group_members"]), candidates))
        if fail_topup_judge and len(judged) > 1:
            raise PositiveRefinementProtocolError("topup Judge response is incomplete")
        values = patterns[min(len(judged) - 1, len(patterns) - 1)]
        return [
            [int(row < values[i % len(values)]) for i in range(len(candidates))]
            for row in range(8)
        ], {"response_task_attempts": 8 if candidates else 0}

    monkeypatch.setattr(worker, "_generate_json", generate)
    monkeypatch.setattr(bounded_retry, "judge_candidates", judge)
    options = {} if mode is None else {"execution_mode": mode}
    result = asyncio.run(
        worker.process_refinement_job(
            job,
            config=config or worker.WorkerConfig(retry_backoff_seconds=0),
            parent_workers=3,
            **options,
        )
    )
    return result, calls, judged


def test_enough_children_do_not_trigger_topup(monkeypatch):
    result, calls, judged = run_job(monkeypatch)
    assert result["audit"]["execution_mode"] == "feedback_batched"
    assert len(calls) == 3
    assert len(judged) == 1
    assert all(len(d["children"]) == 3 for d in result["decompositions"])
    assert all(p["topup_rounds"] == 0 for p in result["audit"]["parents"])


def test_rejected_batch_gets_exactly_one_feedback_round(monkeypatch):
    result, calls, judged = run_job(
        monkeypatch, job=one_parent_job(), patterns=[[0, 8, 1, 7, 0], [2, 6, 7, 8, 0]]
    )
    assert len(calls) == 2
    assert len(judged) == 2
    assert judged[0][0] == judged[1][0]
    assert "Previously rejected" in calls[1][1]
    assert all(
        m["text"] not in prompt
        for _, prompt in calls
        for m in make_job()["group_members"]
    )
    assert [c["pass_count"] for c in result["decompositions"][0]["children"]] == [2, 6]
    assert result["audit"]["parents"][0]["topup_rounds"] == 1
    assert result["audit"]["generator_call_count"] == 2


def test_failed_topup_is_bounded_and_preserves_valid_children(monkeypatch):
    result, calls, judged = run_job(
        monkeypatch, job=one_parent_job(2), patterns=[[4, 0, 1, 7, 8]], fail_topup=True
    )
    assert len(calls) == 4
    assert len(judged) == 1
    assert len(result["decompositions"][0]["children"]) == 1
    audit = result["audit"]["parents"][0]
    assert audit["topup_rounds"] == 1
    assert audit["topup"]["status"] == "generation_failed"
    assert result["audit"]["generator_call_count"] == len(calls)


def test_rejected_topup_stops_without_third_round(monkeypatch):
    result, calls, _ = run_job(
        monkeypatch, job=one_parent_job(), patterns=[[0, 1, 7, 8, 0]]
    )
    assert len(calls) == 2
    assert result["decompositions"] == []


def test_round_limit_and_explicit_single_pass(monkeypatch):
    for config, mode in [
        (worker.WorkerConfig(topup_enabled=False), None),
        (worker.WorkerConfig(topup_round_limit=0), None),
        (worker.WorkerConfig(), "single_pass_batched"),
    ]:
        result, calls, _ = run_job(
            monkeypatch,
            job=one_parent_job(),
            patterns=[[0] * 5],
            config=config,
            mode=mode,
        )
        assert len(calls) == 1
        assert result["decompositions"] == []


def test_cross_round_duplicates_are_not_rescored(monkeypatch):
    result, calls, judged = run_job(
        monkeypatch, job=one_parent_job(), patterns=[[0] * 5], repeat=True
    )
    assert len(calls) == 2
    assert sum(len(batch) for _, batch in judged) == 5
    assert result["decompositions"] == []
    assert len(result["audit"]["parents"][0]["topup"]["duplicate_rejections"]) == 5


def test_disabled_semantics_never_claims_acceptance(monkeypatch):
    job = one_parent_job()
    result, calls, _ = run_job(monkeypatch, job=job)
    meta = result["decompositions"][0]["children"][0]["meta_verification"]
    assert meta["executed"] is False
    assert meta["decision"] == "disabled"
    assert all(meta.get(check) is None for check in META_CHECKS)
    assert result["audit"]["semantic_meta_check_performed"] is False
    assert all(kind == "generate" for kind, _ in calls)
    with tempfile.TemporaryDirectory() as folder:
        registry = PositiveRefinementRegistry(folder)
        assert registry.reserve_job(job)
        assert (
            registry.validate_result(result, created_step=1)["selected_child_count"]
            == 3
        )


def test_semantic_rejections_are_real_and_audited(monkeypatch):
    result, calls, _ = run_job(
        monkeypatch,
        job=one_parent_job(),
        config=worker.WorkerConfig(semantic_meta_enabled=True),
        meta_reject={0, 1, 2},
    )
    assert len(result["decompositions"][0]["children"]) == 2
    assert result["audit"]["semantic_meta_check_performed"] is True
    assert [kind for kind, _ in calls] == ["generate", "meta"]
    assert result["audit"]["semantic_meta_call_count"] == 1
    assert all(
        c["meta_verification"]["executed"]
        for c in result["decompositions"][0]["children"]
    )


def test_seed_reuse_still_runs_semantic_validation(monkeypatch):
    job = one_parent_job()
    job["selected_parents"][0]["seed_candidates"] = [
        dict(
            title=f"Seed {i}",
            completion_point=f"Seed point {i}",
            source_job_id="old-job",
            source_step=1,
            source_pass_count=4,
        )
        for i in range(5)
    ]
    freeze_job(job)
    result, calls, _ = run_job(
        monkeypatch,
        job=job,
        config=worker.WorkerConfig(semantic_meta_enabled=True),
        meta_reject={0},
    )
    assert [kind for kind, _ in calls] == ["meta"]
    assert all(
        c["candidate_source"] == "historical_seed"
        for c in result["decompositions"][0]["children"]
    )
    assert (
        result["audit"]["parents"][0]["joint_meta"]["decisions"][0]["decision"]
        == "reject"
    )


def test_no_filter_retains_constants(monkeypatch):
    monkeypatch.setenv("RUACT_CRITIC_NO_FILTER", "1")
    job = one_parent_job()
    result, calls, _ = run_job(monkeypatch, job=job, patterns=[[0, 8, 1, 7, 0]])
    assert [c["pass_count"] for c in result["decompositions"][0]["children"]] == [
        0,
        8,
        1,
    ]
    assert len(calls) == 1
    with tempfile.TemporaryDirectory() as folder:
        registry = PositiveRefinementRegistry(folder)
        registry.reserve_job(job)
        assert (
            registry.validate_result(result, created_step=1)["selected_child_count"]
            == 3
        )


def test_adapter_preserves_each_parent_budget_and_fallback(monkeypatch):
    import torch
    from ruact.critic.adapter import PositiveRubricRefinementAdapter

    job = make_job()
    result, _, _ = run_job(
        monkeypatch, job=job, patterns=[[2, 3, 4, 5, 6] * 2 + [0] * 5, [0] * 5]
    )
    with tempfile.TemporaryDirectory() as folder:
        registry = PositiveRefinementRegistry(folder)
        registry.reserve_job(job)
        accepted = registry.validate_result(result, created_step=1)[
            "applied_decompositions"
        ]
    assert len(accepted) == 2
    for d in accepted:
        assert d["parent_effective_weight"] + sum(
            c["effective_weight"] for c in d["children"]
        ) == pytest.approx(d["parent_weight"])

    class Batch:
        batch = {"response_mask": torch.ones(8, 2)}
        non_tensor_batch = {}

        def __len__(self):
            return 8

    rewards = torch.tensor([[0.0, 1.0]] * 8)
    extra = dict(
        occurrence_id=[f"r{i}" for i in range(8)],
        prompt_id=["query"] * 8,
        rubric_ids=[["p0", "p1", "p2"] for _ in range(8)],
        rubric_scores=[[1, 1, 1] for _ in range(8)],
        rubric_weights=[[1, 2, 3] for _ in range(8)],
    )
    adapter = PositiveRubricRefinementAdapter.__new__(PositiveRubricRefinementAdapter)
    adapter.config = SimpleNamespace(include_negative_weight=False)
    actual = adapter._apply_decompositions(
        Batch(), rewards, extra, job=job, decompositions=accepted
    )[0]
    for row in range(8):
        coverage = sum(int(row < n) for n in (2, 3, 4)) / 3
        expected = (3 * ((1 - BLEND_LAMBDA) + BLEND_LAMBDA * coverage) + 3) / 6
        assert actual[row, -1].item() == pytest.approx(expected)
    assert (
        adapter._apply_decompositions(
            Batch(), rewards, extra, job=job, decompositions=[]
        )[0]
        is rewards
    )


@pytest.mark.parametrize("change", ["missing", "extra", "index", "boolean", "decision"])
def test_semantic_schema_rejects_malformed_verdicts(change):
    from ruact.critic.feedback_worker import _validate_meta

    value = {
        "results": [
            dict(candidate_index=0, decision="accept", **{c: True for c in META_CHECKS})
        ]
    }
    verdict = value["results"][0]
    if change == "missing":
        verdict.pop(META_CHECKS[0])
    elif change == "extra":
        verdict["unrequested_gate"] = True
    elif change == "index":
        verdict["candidate_index"] = True
    elif change == "boolean":
        verdict[META_CHECKS[0]] = 1
    else:
        verdict["decision"] = "reject"
    with pytest.raises(PositiveRefinementProtocolError):
        _validate_meta(value, [{"candidate_index": 0}])


def test_generator_rejects_non_string_text():
    with pytest.raises(PositiveRefinementProtocolError):
        worker._validate_generated(
            {"candidates": [{"title": 1, "completion_point": "point"}]},
            min_candidate_count=1,
            max_candidate_count=1,
            parent={"description": "parent"},
        )


def test_nonbinary_judge_grid_is_not_coerced():
    with pytest.raises(PositiveRefinementProtocolError):
        worker._attach_judge_scores(
            [{"candidate_index": 0}],
            [[0.5]] * 8,
            response_ids=[str(i) for i in range(8)],
            max_children=1,
        )


def test_registry_rejects_weak_patterns_and_fabricated_meta(monkeypatch):
    job = one_parent_job()
    result, _, _ = run_job(monkeypatch, job=job)
    for mutation in ("weak", "meta"):
        invalid = copy.deepcopy(result)
        child = invalid["decompositions"][0]["children"][0]
        if mutation == "weak":
            child["judge_scores"] = [
                {"response_id": f"r{i}", "score": int(i == 0)} for i in range(8)
            ]
            child["pass_count"] = 1
        else:
            child["meta_verification"].update(
                decision="accept", **{c: True for c in META_CHECKS}
            )
        with tempfile.TemporaryDirectory() as folder:
            registry = PositiveRefinementRegistry(folder)
            registry.reserve_job(job)
            with pytest.raises(PositiveRefinementProtocolError):
                registry.validate_result(invalid, created_step=1)


def test_sidecar_defaults_to_feedback_and_exposes_semantic_switch(monkeypatch):
    from ruact.critic.sidecar import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "critic",
            "--state-dir",
            "state",
            "--semantic-meta-enabled",
            "--topup-round-limit",
            "0",
        ],
    )
    args = parse_args()
    assert args.execution_mode == "feedback_batched"
    assert args.semantic_meta_enabled is True
    assert args.topup_round_limit == 0


def test_audit_counts_both_judge_rounds(monkeypatch):
    result, _, judged = run_job(
        monkeypatch, job=one_parent_job(), patterns=[[0] * 5, [2] * 5]
    )
    assert len(judged) == 2
    assert result["audit"]["fixed_judge"]["response_task_attempts"] == 16
    assert result["audit"]["parents"][0]["generated_candidate_count"] == 10


def test_semantic_topup_covers_new_candidate_indices(monkeypatch):
    result, calls, _ = run_job(
        monkeypatch,
        job=one_parent_job(),
        patterns=[[0] * 5, [2] * 5],
        config=worker.WorkerConfig(semantic_meta_enabled=True),
    )
    assert [kind for kind, _ in calls] == ["generate", "meta", "generate", "meta"]
    assert [
        c["generator_candidate_index"] for c in result["decompositions"][0]["children"]
    ] == [5, 6, 7]
    assert result["audit"]["semantic_meta_call_count"] == 2


def test_registry_requires_explicit_executed_for_enabled_semantics(monkeypatch):
    job = one_parent_job()
    result, _, _ = run_job(
        monkeypatch, job=job, config=worker.WorkerConfig(semantic_meta_enabled=True)
    )
    result["decompositions"][0]["children"][0]["meta_verification"].pop("executed")
    with tempfile.TemporaryDirectory() as folder:
        registry = PositiveRefinementRegistry(folder)
        registry.reserve_job(job)
        with pytest.raises(PositiveRefinementProtocolError):
            registry.validate_result(result, created_step=1)


def test_seed_and_fresh_duplicate_is_validated_only_once(monkeypatch):
    job = one_parent_job()
    job["selected_parents"][0]["seed_candidates"] = [
        dict(
            title="Seed",
            completion_point="p0 round 0 point 0",
            source_job_id="old-job",
            source_step=1,
            source_pass_count=4,
        )
    ]
    freeze_job(job)
    result, calls, judged = run_job(
        monkeypatch, job=job, config=worker.WorkerConfig(semantic_meta_enabled=True)
    )
    assert [kind for kind, _ in calls] == ["generate", "meta"]
    assert len(judged[0][1]) == 4
    assert len(result["audit"]["parents"][0]["duplicate_rejections"]) == 1
    assert (
        result["decompositions"][0]["children"][0]["candidate_source"]
        == "historical_seed"
    )


def test_semantic_rejection_can_trigger_one_feedback_batch(monkeypatch):
    result, calls, _ = run_job(
        monkeypatch,
        job=one_parent_job(),
        config=worker.WorkerConfig(semantic_meta_enabled=True),
        meta_reject=set(range(5)),
    )
    assert [kind for kind, _ in calls] == ["generate", "meta", "generate", "meta"]
    assert "(rejected)" in calls[2][1]
    assert "semantic_meta_rejection" not in calls[2][1]
    assert result["audit"]["parents"][0]["topup_rounds"] == 1
    assert len(result["decompositions"][0]["children"]) == 3


def test_registry_rejects_inconsistent_semantic_execution_audit(monkeypatch):
    job = one_parent_job()
    result, _, _ = run_job(monkeypatch, job=job)
    result["audit"]["semantic_meta_check_performed"] = True
    with tempfile.TemporaryDirectory() as folder:
        registry = PositiveRefinementRegistry(folder)
        registry.reserve_job(job)
        with pytest.raises(PositiveRefinementProtocolError):
            registry.validate_result(result, created_step=1)


def test_successful_topup_preserves_earlier_children_and_parent_cap(monkeypatch):
    result, _, _ = run_job(
        monkeypatch, job=one_parent_job(3), patterns=[[2, 6, 0, 1, 8], [3, 4, 5, 6, 0]]
    )
    children = result["decompositions"][0]["children"]
    assert [c["generator_candidate_index"] for c in children] == [0, 1, 5]
    audit = result["audit"]["parents"][0]
    assert len(audit["candidate_pass_counts"]) == 10
    assert sum(c["retained"] for c in audit["candidate_pass_counts"]) == 3
    assert audit["topup"]["judge_column_map"] == [5, 6, 7, 8, 9]


@pytest.mark.parametrize(
    "error",
    [
        PositiveRefinementProtocolError("invalid Judge grid"),
        bounded_retry.RefinementBudgetExhausted("preparation failed"),
    ],
)
def test_sidecar_persists_terminal_error_without_replaying_job(
    monkeypatch, tmp_path, error
):
    from ruact.critic import sidecar

    job = one_parent_job()
    jobs = tmp_path / f"positive-refinement-jobs-v{SCHEMA_VERSION}"
    jobs.mkdir()
    (jobs / f"{job['job_id']}.json").write_text(json.dumps(job))
    monkeypatch.setattr("sys.argv", ["critic", "--state-dir", str(tmp_path)])
    args = sidecar.parse_args()
    calls = []

    async def fail(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(sidecar, "process_refinement_job", fail)
    result = asyncio.run(
        asyncio.wait_for(
            sidecar._process_available(args, worker.WorkerConfig()), timeout=0.2
        )
    )
    assert result == (1, 1)
    assert len(calls) == 1
    saved = json.loads(
        (
            tmp_path
            / f"positive-refinement-results-v{SCHEMA_VERSION}"
            / f"{job['job_id']}.json"
        ).read_text()
    )
    assert saved["status"] == "terminal_error"
    assert saved["audit"]["terminal_error"]["error_type"] == type(error).__name__
    diagnostic = json.loads(
        (
            tmp_path
            / f"positive-refinement-events-v{SCHEMA_VERSION}"
            / f"{job['job_id']}.terminal.json"
        ).read_text()
    )
    assert diagnostic["will_retry"] is False
    assert asyncio.run(sidecar._process_available(args, worker.WorkerConfig())) == (
        0,
        0,
    )
    assert len(calls) == 1


def test_feedback_prompt_contains_no_outcome_details(monkeypatch):
    _, calls, _ = run_job(
        monkeypatch, job=one_parent_job(), patterns=[[0, 8, 1, 7, 0], [2] * 5]
    )
    prompt = calls[1][1]
    assert "(rejected)" in prompt
    for label in (
        "zero_pass_pattern",
        "full_pass_pattern",
        "weak_imbalanced_pattern",
        "pass_count",
        "pass_rate",
        "semantic_meta_rejection",
        "reason:",
    ):
        assert label not in prompt
    assert all(m["text"] not in prompt for m in one_parent_job()["group_members"])


def test_topup_judge_failure_keeps_initial_children_with_unknown_scores(monkeypatch):
    result, calls, judged = run_job(
        monkeypatch,
        job=one_parent_job(2),
        patterns=[[4, 0, 1, 7, 8]],
        fail_topup_judge=True,
    )
    assert len(calls) == 2
    assert len(judged) == 2
    assert [
        c["generator_candidate_index"] for c in result["decompositions"][0]["children"]
    ] == [0]
    audit = result["audit"]["parents"][0]
    assert audit["topup"]["status"] == "judge_failed"
    assert audit["topup"]["judge_scores_available"] is False
    assert "judge_grid" not in audit["topup"]
    assert len(audit["candidate_pass_counts"]) == 5
    assert result["audit"]["fixed_judge"]["response_task_attempts"] is None
    assert result["audit"]["fixed_judge"]["failed_round_count"] == 1
