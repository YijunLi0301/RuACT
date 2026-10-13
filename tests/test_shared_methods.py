import ast
import asyncio
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from omegaconf import OmegaConf
from rubric_rl import ruscarl
from rubric_rl.simple_reweight import rewards
from rubric_rl.refinement import bounded_retry, single_pass_worker, worker
from test_refinement_singlepass import make_job

ROOT = Path(__file__).resolve().parents[1]


def test_reweight_polarity_and_constant_columns():
    scores = [[0, 1, 1], [1, 1, 1]]
    result, baseline, factors = rewards(scores, [2, -1, 1], 0.5)
    np.testing.assert_allclose(factors, [1.5, 1, 1])
    np.testing.assert_allclose(result, [1 / 5, 4 / 5])
    np.testing.assert_allclose(baseline, [1 / 4, 3 / 4])


def test_scaffold_schedule_and_zero_reveal():
    a = ruscarl._calculate_rubric_ratio(
        0,
        8,
        "step_sigmoid",
        global_steps=1,
        total_training_steps=200,
        step_sigmoid_start_point=0.2,
        step_sigmoid_steepness=125,
    )
    b = ruscarl._calculate_rubric_ratio(
        0,
        8,
        "step_sigmoid",
        global_steps=200,
        total_training_steps=200,
        step_sigmoid_start_point=0.2,
        step_sigmoid_steepness=125,
    )
    assert a > 0.999 and b < 1e-20

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

    truth = {
        "ground_truth": {"rubrics": [{"description": "Required detail", "weight": 1}]}
    }
    original = [1, 2, 3]
    prompt, metrics = ruscarl.scaffold_prompt(
        Tokenizer(),
        original,
        truth,
        sample_index=7,
        group_size=8,
        step=1,
        total_steps=200,
    )
    assert prompt == original and metrics["rubric_count"] == 0
    prompt, metrics = ruscarl.scaffold_prompt(
        Tokenizer(),
        original,
        truth,
        sample_index=0,
        group_size=8,
        step=1,
        total_steps=200,
    )
    assert prompt[-3:] == original and len(prompt) > 3
    assert original == [1, 2, 3]


def test_scaffold_generation_does_not_change_training_prompt():
    source = ast.parse(
        (ROOT / "verl/experimental/agent_loop/single_turn_agent_loop.py").read_text()
    )
    fn = next(
        n
        for n in ast.walk(source)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
    )
    scope = {
        "Any": object,
        "AgentLoopOutput": SimpleNamespace,
        "uuid4": uuid4,
        "simple_timer": lambda *a: nullcontext(),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<agent>", "exec"), scope)
    sent = []

    async def vision(messages):
        return {}

    async def template(*a, **kw):
        return [10, 20]

    async def generate(**kwargs):
        sent.append(kwargs["prompt_ids"])
        return SimpleNamespace(token_ids=[30, 40], log_probs=None, routed_experts=None)

    agent = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"total_training_steps": 200, "ruscarl": {"enabled": True}},
                "actor_rollout_ref": {"rollout": {"n": 8, "max_model_len": 16000}},
            }
        ),
        response_length=10,
        tool_schemas=[],
        process_vision_info=vision,
        apply_chat_template=template,
        tokenizer=SimpleNamespace(encode=lambda text, **kw: list(text.encode())),
        server_manager=SimpleNamespace(generate=generate),
    )
    kwargs = {
        "raw_prompt": [{"role": "user", "content": "Question"}],
        "reward_model": {"rubrics": [{"criterion": "Important", "points": 1}]},
        "rollout_index": 0,
        "training_step": 1,
    }
    out = asyncio.run(scope["run"](agent, {}, **kwargs))
    assert out.prompt_ids == [10, 20]
    assert out.response_ids == [30, 40]
    assert sent[0] != out.prompt_ids and sent[0][-2:] == out.prompt_ids
    kwargs.pop("training_step")
    out = asyncio.run(scope["run"](agent, {}, **kwargs))
    assert sent[-1] == [10, 20]


def test_no_filter_retains_constants_without_changing_scores(monkeypatch):
    pool = [{"candidate_index": i} for i in range(5)]
    grid = [[int(row < count) for count in [0, 1, 4, 7, 8]] for row in range(8)]
    ids = [str(i) for i in range(8)]
    monkeypatch.setenv("RUACT_CRITIC_NO_FILTER", "0")
    filtered, _ = worker._attach_judge_scores(
        pool, grid, response_ids=ids, max_children=5
    )
    assert [x["pass_count"] for x in filtered] == [4]
    monkeypatch.setenv("RUACT_CRITIC_NO_FILTER", "1")
    unfiltered, _ = worker._attach_judge_scores(
        pool, grid, response_ids=ids, max_children=5
    )
    assert [x["pass_count"] for x in unfiltered] == [0, 1, 4, 7, 8]


def test_parent_generation_has_three_total_attempts(monkeypatch):
    calls = []

    async def generate(prompt, schema, *, config):
        calls.append(config.max_attempts)
        raise worker.PositiveRefinementProtocolError("invalid JSON")

    async def prepare(job, parent, config, generator_call):
        return await generator_call("p", {}, config=config)

    async def no_sleep(*a):
        pass

    monkeypatch.setattr(worker, "_generate_json", generate)
    monkeypatch.setattr(bounded_retry.asyncio, "sleep", no_sleep)
    job = make_job()
    with pytest.raises(bounded_retry.RefinementBudgetExhausted):
        asyncio.run(
            bounded_retry.prepare_parent(
                job,
                job["selected_parents"][0],
                worker.WorkerConfig(),
                prepare,
                asyncio.Semaphore(1),
            )
        )
    assert calls == [1, 1, 1]


def test_strict_judge_recovery_preserves_successful_rows(monkeypatch):
    counts = {}
    monkeypatch.setattr(
        bounded_retry.judge_api, "_build_judge_prompt", lambda **kw: kw["final_answer"]
    )
    monkeypatch.setattr(
        bounded_retry.judge_api, "_judge_current_base", lambda *_: ("unused", False)
    )
    monkeypatch.setattr(
        bounded_retry.judge_api, "_judge_breaker_on_success", lambda *_: None
    )

    def request(prompt, *, rubric_count, response_metadata, **kw):
        prompt = prompt.splitlines()[0]
        counts[prompt] = counts.get(prompt, 0) + 1
        response_metadata["finish_reason"] = "stop"
        if prompt == "ANSWER_0" and counts[prompt] == 1:
            return "{}"
        return json.dumps({str(i + 1): "PRESENT" for i in range(rubric_count)})

    async def no_sleep(*a):
        pass

    monkeypatch.setattr(bounded_retry.judge_api, "_request_judge_once", request)
    monkeypatch.setattr(bounded_retry.asyncio, "sleep", no_sleep)
    grid, audit = asyncio.run(
        bounded_retry.judge_candidates(
            make_job(), [{"description": "d"}], workers=8, attempts=3
        )
    )
    assert grid == [[1]] * 8
    assert counts["ANSWER_0"] == 2 and sum(counts.values()) == 9


def test_hydra_profiles_resolve_to_the_shared_runtime():
    from hydra import compose, initialize_config_dir
    from test_unified_submission import load_entry

    entry = load_entry()
    with initialize_config_dir(
        config_dir=str(ROOT / "verl/trainer/config"), version_base=None
    ):
        for domain in ("medicine", "science"):
            for size in ("1.7b", "4b"):
                for method in entry.METHODS:
                    settings = entry.load_settings(domain, size, None)
                    _, _, cmd, _, _ = entry.build(settings, method, preview=True)
                    cfg = compose(config_name="ppo_trainer", overrides=cmd[3:])
                    resolved = OmegaConf.to_container(cfg, resolve=True)
                    assert resolved["trainer"]["total_training_steps"] == 200
                    assert resolved["actor_rollout_ref"]["rollout"]["mode"] == "async"
                    if method == "RuACT":
                        actual = resolved["actor_rollout_ref"]["actor"]["distillation"]
                        assert actual["beta"] * 8 == (0.1 if size == "1.7b" else 0.05)
                        assert actual["token_kl_clip"] == 10
                        assert actual["include_negative_weight_failures"] is True


def test_reversed_critic_selection_reaches_worker(monkeypatch):
    from rubric_rl.refinement.protocol import build_refinement_jobs

    rows = [
        dict(
            group_id="g",
            query_id="q",
            query="Question",
            rubric_query_version=0,
            rollout_index=i,
            original_index=i,
            response_id=f"r{i}",
            response_text="Answer",
            rubric_ids=["a", "b", "c"],
            rubric_titles=["A", "B", "C"],
            rubric_descriptions=["Always absent", "Always present", "Variable"],
            rubric_weights=[1, 1, 1],
            rubric_scores=[0, 1, int(i < 4)],
        )
        for i in range(8)
    ]
    for mode, parent_id, count in [("saturated", "b", 8), ("unreachable", "a", 0)]:
        monkeypatch.setenv("RUACT_CRITIC_SELECTION", mode)
        jobs = build_refinement_jobs(rows, run_id="test", global_step=1)
        assert len(jobs) == 1
        worker.validate_refinement_job(jobs[0])
        assert [
            (p["rubric_id"], p["pass_count"]) for p in jobs[0]["selected_parents"]
        ] == [(parent_id, count)]
