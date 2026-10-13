import asyncio, os, unittest
from unittest.mock import patch
from types import SimpleNamespace
from test_refinement_singlepass import make_job
from rubric_rl.refinement import worker, single_pass_worker as sp
from rubric_rl.refinement import transport as dw
from rubric_rl.refinement import bounded_retry


class RefinementConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def run_partial(self, all_empty=False):
        self.calls = []

        async def prepare(job, parent, config, **kw):
            i = int(parent["rubric_id"][1:])
            return (
                [
                    dict(
                        candidate_index=k,
                        title="c",
                        description=f"{i}:{k}",
                        completion_point=f"{i}:{k}",
                        polarity="positive",
                        candidate_source="online_generation",
                        source_candidate_index=k,
                        meta_verification={
                            "necessity": "optional",
                            "semantic_family": f"{i}:{k}",
                        },
                    )
                    for k in range(4)
                ],
                {
                    "generator_call_count": 1,
                    "generator_prompt_character_count": 0,
                    "generator_prompt_utf8_bytes": 0,
                },
            )

        def judge(**kw):
            self.calls.append(kw)
            row = int(kw["final_answer"].split("_")[-1])
            return SimpleNamespace(
                scores=[
                    int(all_empty or int(d.split(":")[0]) == 2 or row < 4)
                    for d in kw["descriptions"]
                ],
                attempts=1,
            )

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
            patch.object(sp, "_prepare_parent", prepare),
            patch.object(dw, "_call_judge", judge),
            patch.dict(os.environ, {"POSITIVE_REFINEMENT_BOUNDED_RETRIES": "0"}),
        ):
            return await sp.process_single_pass_job(
                make_job(), config=worker.WorkerConfig(), parent_workers=3
            )

    async def test_partial_success_survives(self):
        r = await self.run_partial()
        self.assertEqual(len(r["decompositions"]), 2)
        self.assertEqual(len(self.calls), 8)

    async def test_empty_does_not_regenerate(self):
        r = await self.run_partial(True)
        self.assertEqual(r["decompositions"], [])
        self.assertEqual(len(self.calls), 8)

    async def test_generator_bounded_parent_retries(self):
        self.n = 0

        async def prepare(job, parent, config, **kw):
            self.n += 1
            raise worker.PositiveRefinementProtocolError("bad generator output")

        async def sleep(*a):
            pass

        with (
            patch.object(sp, "_prepare_parent", prepare),
            patch.object(sp.asyncio, "sleep", sleep),
            patch.dict(os.environ, {"POSITIVE_REFINEMENT_BOUNDED_RETRIES": "0"}),
        ):
            with self.assertRaises(bounded_retry.RefinementBudgetExhausted):
                await sp.process_single_pass_job(
                    make_job(), config=worker.WorkerConfig(), parent_workers=3
                )
        self.assertEqual(self.n, 9)

    async def test_judge_keeps_good_rows_after_three_failures(self):
        counts = {}

        def judge(**kw):
            answer = kw["final_answer"]
            counts[answer] = counts.get(answer, 0) + 1
            if answer == "ANSWER_0" and counts[answer] < 5:
                raise TimeoutError("temporary")
            return SimpleNamespace(scores=[1], attempts=1)

        with (
            patch.object(dw, "_call_judge", judge),
            patch.object(dw.time, "sleep", lambda *_: None),
            patch.dict(os.environ, {"Refinement_JUDGE_RETRY_UNTIL_SUCCESS": "1"}),
        ):
            rows, a = dw._judge_candidates_forever(
                make_job(),
                [dict(title="c", description="d", polarity="positive")],
                workers=4,
                attempts=3,
            )
        self.assertEqual(len(rows), 8)
        self.assertEqual(counts["ANSWER_0"], 5)
        self.assertTrue(all((v == 1 for k, v in counts.items() if k != "ANSWER_0")))


if __name__ == "__main__":
    unittest.main()
