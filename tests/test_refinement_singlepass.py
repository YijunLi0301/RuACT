import asyncio
from rubric_rl.refinement import bounded_retry
import copy
import os
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


from rubric_rl.refinement import worker
from rubric_rl.refinement.protocol import (
    BLEND_LAMBDA,
    REWARD_MODE,
    SCHEMA_VERSION,
    seed_bundle_hash,
    parent_fingerprint,
    ordered_response_set_hash,
    refinement_transaction_id,
    PositiveRefinementProtocolError,
)
from rubric_rl.refinement import transport as judge_worker


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


class SinglePassTests(unittest.IsolatedAsyncioTestCase):
    async def run_job(self, *, patterns=None, job=None, parent_workers=3):
        job = job or make_job()
        self.generator_prompts, self.judge_requests = ([], [])
        patterns = patterns or {
            0: [8, 0, 1, 4, 6],
            1: [2, 3, 4, 5, 6],
            2: [8, 0, 1, 7, 8],
        }

        async def generate(prompt, schema, *, config):
            self.generator_prompts.append(prompt)
            self.assertNotIn("ANSWER_", prompt)
            parent = next((i for i in range(3) if f"Requirement{i}" in prompt))
            count = schema["properties"]["candidates"]["maxItems"]
            return (
                {
                    "candidates": [
                        dict(title=f"P{parent} C{c}", completion_point=f"p{parent}c{c}")
                        for c in range(count)
                    ]
                },
                {"attempts": 1},
            )

        def judge(**kwargs):
            self.judge_requests.append(kwargs)
            row = int(kwargs["final_answer"].split("_")[-1])
            scores = []
            for description in kwargs["descriptions"]:
                parent, child = (int(description[1]), int(description[3]))
                scores.append(int(row < patterns[parent][child]))
            return SimpleNamespace(scores=scores, attempts=1)

        async def judge_candidates(job, candidates, *, workers, attempts):
            return await worker._judge_candidates_forever(
                job,
                candidates,
                workers=workers,
                attempts=attempts,
                descriptions_only=True,
            )

        with (
            patch.object(bounded_retry, "judge_candidates", judge_candidates),
            patch.object(worker, "_generate_json", generate),
            patch.object(judge_worker, "_call_judge", judge),
        ):
            return await worker.process_refinement_job(
                job,
                config=worker.WorkerConfig(),
                parent_workers=parent_workers,
                execution_mode="single_pass_batched",
            )

    async def test_combines_three_parents_into_eight_calls_and_recovers_later_candidates(
        self,
    ):
        result = await self.run_job()
        self.assertEqual(len(self.judge_requests), 8)
        self.assertTrue(
            all((len(r["descriptions"]) == 15 for r in self.judge_requests))
        )
        self.assertEqual(len(self.generator_prompts), 3)
        decompositions = {d["parent_rubric_id"]: d for d in result["decompositions"]}
        self.assertEqual(set(decompositions), {"p0", "p1"})
        self.assertEqual(
            [c["generator_candidate_index"] for c in decompositions["p0"]["children"]],
            [3, 4],
        )
        self.assertEqual(
            [c["generator_candidate_index"] for c in decompositions["p1"]["children"]],
            [0, 1, 2],
        )
        for parent, decomposition in decompositions.items():
            self.assertEqual(decomposition["blend_lambda"], BLEND_LAMBDA)
            for child in decomposition["children"]:
                self.assertTrue(child["description"].startswith(parent))
                self.assertEqual(
                    [r["response_id"] for r in child["judge_scores"]],
                    [f"r{i}" for i in range(8)],
                )
        self.assertEqual(result["audit"]["fixed_judge"]["response_task_attempts"], 8)
        self.assertEqual(result["audit"]["execution_mode"], "single_pass_batched")
        self.assertEqual(
            result["audit"]["parents"][2]["status"], "no_nonconstant_child_pattern"
        )

    async def test_all_rejected_preserves_empty_decompositions_without_regeneration(
        self,
    ):
        result = await self.run_job(
            patterns={i: [0, 8, 1, 7, 0] for i in range(3)}, parent_workers=1
        )
        self.assertEqual(result["decompositions"], [])
        self.assertEqual(len(self.generator_prompts), 3)
        self.assertEqual(len(self.judge_requests), 8)

    async def test_audits_cap_rejections_and_every_judged_candidate(self):
        result = await self.run_job()
        audit = result["audit"]["parents"][1]["candidate_pass_counts"]
        self.assertEqual(len(audit), 5)
        self.assertEqual(sum((c["retained"] for c in audit)), 3)
        self.assertEqual(
            [c["rejection_reason"] for c in audit[-2:]], ["child_cap", "child_cap"]
        )

    async def test_seed_pool_is_scored_once_without_new_generation(self):
        job = make_job()
        for p, parent in enumerate(job["selected_parents"]):
            parent["seed_candidates"] = [
                dict(
                    title=f"Seed{c}",
                    completion_point=f"p{p}c{c}",
                    source_job_id="old-job",
                    source_step=1,
                    source_pass_count=4,
                )
                for c in range(5)
            ]
        job["seed_bundle_hash"] = seed_bundle_hash(job["selected_parents"])
        result = await self.run_job(job=job)
        self.assertEqual(len(self.generator_prompts), 0)
        self.assertEqual(len(self.judge_requests), 8)
        self.assertTrue(
            all(
                (
                    c["candidate_source"] == "historical_seed"
                    for d in result["decompositions"]
                    for c in d["children"]
                )
            )
        )

    async def test_no_filter_constants_survive_registry_validation(self):
        from rubric_rl.refinement.registry import PositiveRefinementRegistry

        job = make_job()
        with patch.dict(os.environ, {"RUACT_CRITIC_NO_FILTER": "1"}):
            result = await self.run_job(
                job=job, patterns={i: [0, 8, 1, 7, 0] for i in range(3)}
            )
            with tempfile.TemporaryDirectory() as folder:
                registry = PositiveRefinementRegistry(folder)
                self.assertTrue(registry.reserve_job(job))
                accepted = registry.validate_result(result, created_step=1)[
                    "applied_decompositions"
                ]
            self.assertEqual(len(accepted), 3)
            for item in accepted:
                self.assertEqual([c["pass_count"] for c in item["children"]], [0, 8, 1])
            self.assertEqual(len(self.generator_prompts), 3)
            self.assertEqual(len(self.judge_requests), 8)

    async def test_registry_accepts_result_and_adapter_preserves_lambda_and_fallback_parent(
        self,
    ):
        import torch
        from rubric_rl.refinement.registry import PositiveRefinementRegistry
        from rubric_rl.refinement.adapter import PositiveRubricRefinementAdapter

        job = make_job()
        result = await self.run_job(job=job)
        with tempfile.TemporaryDirectory() as folder:
            registry = PositiveRefinementRegistry(folder)
            self.assertTrue(registry.reserve_job(job))
            accepted = registry.validate_result(result, created_step=1)[
                "applied_decompositions"
            ]
            for decomposition in accepted:
                self.assertAlmostEqual(
                    decomposition["parent_effective_weight"]
                    + sum((c["effective_weight"] for c in decomposition["children"])),
                    decomposition["parent_weight"],
                )

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
            adapter = PositiveRubricRefinementAdapter.__new__(
                PositiveRubricRefinementAdapter
            )
            adapter.config = SimpleNamespace(include_negative_weight=False)
            actual = adapter._apply_decompositions(
                Batch(), rewards, extra, job=job, decompositions=accepted
            )[0]
            for row in range(8):
                p0 = (int(row < 4) + int(row < 6)) / 2
                p1 = sum((int(row < threshold) for threshold in (2, 3, 4))) / 3
                blend = job["blend_lambda"]
                expected = (
                    (1 - blend) + blend * p0 + 2 * ((1 - blend) + blend * p1) + 3
                ) / 6
                self.assertAlmostEqual(actual[row, -1].item(), expected, places=6)
            unchanged = adapter._apply_decompositions(
                Batch(), rewards, extra, job=job, decompositions=[]
            )[0]
            self.assertIs(unchanged, rewards)

    async def test_rejects_incomplete_merged_grid(self):

        async def incomplete(*args, **kwargs):
            return ([[0] * 15] * 7, {})

        with patch.object(worker, "_judge_candidates_forever", incomplete):
            with self.assertRaisesRegex(
                PositiveRefinementProtocolError, "grid is misaligned"
            ):
                await self.run_job()

    async def test_rejects_duplicate_parent_ids(self):
        job = make_job()
        job["selected_parents"][1]["rubric_id"] = "p0"
        freeze_job(job)
        with self.assertRaisesRegex(
            PositiveRefinementProtocolError, "parent IDs must be unique"
        ):
            await self.run_job(job=job)


if __name__ == "__main__":
    unittest.main()
