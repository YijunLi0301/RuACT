import copy
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from rubric_rl import verl_reward as reward
from rubric_rl.reward import aggregate_weighted_reward

path = (
    Path(__file__).resolve().parents[1]
    / "verl/trainer/ppo/distillation_teacher_gate.py"
)
spec = importlib.util.spec_from_file_location("isolated_teacher_gate", path)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


class TeacherScoreReuseTests(unittest.TestCase):
    def setUp(self):
        model = patch.object(reward, "CANONICAL_JUDGE_MODEL", "test-judge")
        model.start()
        self.addCleanup(model.stop)
        self.calls = []
        self.feedback = self.make_feedback()

    def make_feedback(self, text="teacher answer", scores=(1, 1), dynamic=False):
        row = dict(
            judge_valid=True,
            judge_model=reward.CANONICAL_JUDGE_MODEL,
            query="public query",
            candidate_final_answer=text,
            rubric_descriptions=["criterion A", "criterion B"],
            rubric_weights=[2.0, -1.0],
            rubric_scores=list(scores),
            dynamic_rubric_descriptions=["criterion C"] if dynamic else [],
            dynamic_rubric_weights=[1.0] if dynamic else [],
            dynamic_rubric_scores=[1] if dynamic else [],
        )
        for prefix in ("", "dynamic_"):
            descriptions = row[prefix + "rubric_descriptions"]
            row[prefix + "judge_prompt_sha256"] = (
                hashlib.sha256(
                    reward._build_judge_prompt(
                        query=row["query"],
                        final_answer=text,
                        titles=None,
                        descriptions=tuple(descriptions),
                        polarities=None,
                    ).encode()
                ).hexdigest()
                if descriptions
                else ""
            )
        return row

    def call_judge(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(scores=(1,) * len(kwargs["descriptions"]))

    def run_gate(self, feedback, texts=None, anchors=None, teacher_k=1, dynamic=False):
        return gate.filter_anchors_by_teacher_verification(
            anchor_indices=anchors or [0],
            teacher_response_texts=texts or ["teacher answer"],
            teacher_k=teacher_k,
            uids=["q", "q"],
            raw_prompts=[[{"role": "user", "content": "public query"}]] * 2,
            rubric_descriptions=[["criterion A", "criterion B"]] * 2,
            rubric_weights=[[2.0, -1.0]] * 2,
            raw_scores=[0.0, 0.25],
            dynamic_rubric_descriptions=[["criterion C"]] * 2 if dynamic else None,
            dynamic_rubric_weights=[[1.0]] * 2 if dynamic else None,
            candidate_final_answer=reward._candidate_final_answer,
            call_judge=self.call_judge,
            aggregate_reward=aggregate_weighted_reward,
            give_up_exception=reward.JudgeGiveUpError,
            log=lambda _: None,
            teacher_reward_feedback=feedback,
            build_judge_prompt=reward._build_judge_prompt,
            judge_model=reward.CANONICAL_JUDGE_MODEL,
        )

    def test_valid_rewardloop_score_skips_api_and_preserves_negative_weights(self):
        indices, texts, stats, details = self.run_gate([self.feedback])
        self.assertEqual(indices, [0])
        self.assertEqual(details[0]["teacher_total"], 0.5)
        self.assertEqual(self.calls, [])
        self.assertEqual(stats.reused_samples, 1)
        self.assertEqual(stats.rescored_samples, 0)

    def test_same_or_worse_score_rejected_without_api(self):
        self.feedback["rubric_scores"] = [0, 0]
        self.assertEqual(self.run_gate([self.feedback])[0], [])
        self.assertEqual(self.calls, [])

    def test_mismatched_or_invalid_feedback_rescores(self):
        changes = {
            "judge_valid": False,
            "judge_model": "",
            "query": "privileged query",
            "candidate_final_answer": "student answer",
            "rubric_descriptions": ["other"],
            "rubric_weights": [1.0, 1.0],
            "rubric_scores": [1],
            "judge_prompt_sha256": "wrong",
            "dynamic_rubric_descriptions": ["unexpected"],
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                row = copy.deepcopy(self.feedback)
                row[key] = value
                self.calls.clear()
                result = self.run_gate([row])
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(result[2].reused_samples, 0)
                self.assertEqual(result[2].rescored_samples, 1)

    def test_missing_feedback_retains_legacy_scoring(self):
        self.run_gate(None)
        self.assertEqual(len(self.calls), 1)

    def test_mixed_feedback_only_rescores_missing_sample_and_keeps_k_semantics(self):
        result = self.run_gate(
            [self.feedback, None],
            texts=["teacher answer", "second"],
            anchors=[0, 0],
            teacher_k=2,
        )
        self.assertEqual(result[0], [0])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((result[2].reused_samples, result[2].rescored_samples), (1, 1))

    def test_dynamic_channel_requires_its_own_matching_digest(self):
        feedback = self.make_feedback(dynamic=True)
        self.assertEqual(self.run_gate([feedback], dynamic=True)[0], [0])
        self.assertEqual(self.calls, [])
        feedback["dynamic_judge_prompt_sha256"] = "wrong"
        self.run_gate([feedback], dynamic=True)
        self.assertEqual(len(self.calls), 2)

    def test_misaligned_feedback_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_gate([])

    def test_slot_selection_drops_invalid_group_and_padding_but_keeps_k_order(self):
        metadata = {
            "candidate_final_answer": ["a1", "a2", "bad1", "bad2", "c1", "c2", "pad"],
            "judge_valid": [True] * 7,
        }
        selected = gate.select_teacher_reward_feedback(metadata, [0, 1, 4, 5])
        self.assertEqual(
            [row["candidate_final_answer"] for row in selected],
            ["a1", "a2", "c1", "c2"],
        )

    def test_non_integer_cached_scores_fall_back(self):
        for scores in ([True, False], [1.0, 0.0], [float("nan"), 1]):
            with self.subTest(scores=scores):
                row = copy.deepcopy(self.feedback)
                row["rubric_scores"] = scores
                self.calls.clear()
                self.run_gate([row])
                self.assertEqual(len(self.calls), 1)

    def test_real_trainer_method_keeps_output_slots_aligned_after_filtering(self):
        import numpy as np
        import torch

        source = path.with_name("ray_trainer.py").read_text()
        cls = next(
            (
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer"
            )
        )
        method = next(
            (
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_maybe_build_distillation_batch"
            )
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        class Config(dict):
            __getattr__ = dict.__getitem__

        class Batch:
            def __init__(self, n):
                self.batch = {
                    "rm_scores": torch.tensor([[0.0, 0.25]] * n),
                    "response_mask": torch.ones((n, 2)),
                    "token_level_rewards": torch.tensor([[0.0, 0.25]] * n),
                }
                self.batch = type("TensorDictStub", (dict,), {"batch_size": (n,)})(
                    self.batch
                )
                self.non_tensor_batch = {
                    "uid": np.array(["a", "a", "b", "b", "c", "c"][:n]),
                    "raw_prompt": [
                        [{"role": "user", "content": "public query"}] for _ in range(n)
                    ],
                }

            def __getitem__(self, indices):
                return Batch(len(indices))

            def select(self, **kwargs):
                return self

        class StopAfterGate(Exception):
            pass

        captured = []

        def run_real_gate(**kwargs):
            captured.append(gate.filter_anchors_by_teacher_verification(**kwargs))
            raise StopAfterGate

        for k, equal_scores in ((1, False), (2, False), (1, True)):
            with self.subTest(k=k, equal_scores=equal_scores):
                texts = (
                    ["a" + str(i) for i in range(k)]
                    + ["bad"] * k
                    + ["c" + str(i) for i in range(k)]
                )
                scheduled_len = (len(texts) + 3) // 4 * 4
                padded_texts = texts + ["padding"] * (scheduled_len - len(texts))
                rows = [self.make_feedback(text=t) for t in padded_texts]
                if equal_scores:
                    for row in rows:
                        row.update(rubric_weights=[7.0, 3.0], rubric_scores=[1, 0])
                metadata = {}
                for key in rows[0]:
                    metadata[key] = np.empty(len(rows), dtype=object)
                    metadata[key][:] = [row[key] for row in rows]
                output = SimpleNamespace(
                    batch={
                        "responses": torch.arange(scheduled_len).reshape(-1, 1),
                        "response_mask": torch.ones(
                            (scheduled_len, 1), dtype=torch.long
                        ),
                    },
                    non_tensor_batch=metadata,
                )
                rr = Config(
                    teacher_final_only=True,
                    teacher_gate=True,
                    teacher_k=k,
                    max_rubrics=3,
                    teacher_max_response_length=4096,
                )
                actor = Config(
                    policy_loss={"loss_mode": "distillation"}, distillation=rr
                )
                config = SimpleNamespace(
                    actor_rollout_ref=SimpleNamespace(
                        actor=actor,
                        rollout=SimpleNamespace(
                            n=2, agent=SimpleNamespace(num_workers=4)
                        ),
                    )
                )
                judge_module = SimpleNamespace(
                    _candidate_final_answer=reward._candidate_final_answer,
                    _call_judge=self.call_judge,
                    _build_judge_prompt=reward._build_judge_prompt,
                    CANONICAL_JUDGE_MODEL=reward.CANONICAL_JUDGE_MODEL,
                    JudgeGiveUpError=reward.JudgeGiveUpError,
                )
                fake_self = SimpleNamespace(
                    config=config,
                    async_rollout_mode=True,
                    global_steps=6,
                    tokenizer=SimpleNamespace(
                        eos_token_id=None, decode=lambda ids, **kw: padded_texts[ids[0]]
                    ),
                    async_rollout_manager=SimpleNamespace(
                        generate_sequences=lambda _: output
                    ),
                    _load_teacher_gate_modules=lambda: (
                        judge_module,
                        SimpleNamespace(
                            aggregate_weighted_reward=aggregate_weighted_reward
                        ),
                    ),
                )

                def split(text):
                    if text == "bad":
                        raise ValueError("invalid format")
                    return text

                namespace = dict(
                    np=np,
                    torch=torch,
                    select_teacher_reward_feedback=gate.select_teacher_reward_feedback,
                    filter_anchors_by_teacher_verification=run_real_gate,
                    split_thinking_teacher_response=split,
                    build_distillation_plan=lambda **kw: SimpleNamespace(
                        anchor_mask=[True, False, True, False, True, False],
                        privileged_instructions=["hard criterion"] * 6,
                    ),
                    pad_indices_to_multiple=lambda indices, multiple: (
                        indices + [indices[0]] * (scheduled_len - len(indices)),
                        scheduled_len - len(indices),
                    ),
                )
                exec(
                    compile(module, str(path.with_name("ray_trainer.py")), "exec"),
                    namespace,
                )
                batch = Batch(6)
                batch.batch["rm_scores"] = torch.tensor(
                    [[0.0, 0.7 if equal_scores else 0.0]] * 6
                )
                feedback = {
                    "judge_valid": [True] * 6,
                    "rubric_scores": [[1, 0] if equal_scores else [0, 0]] * 6,
                    "rubric_ids": [["a", "b"]] * 6,
                    "rubric_descriptions": [["criterion A", "criterion B"]] * 6,
                    "rubric_weights": [[7.0, 3.0] if equal_scores else [2.0, -1.0]] * 6,
                }
                self.calls.clear()
                with self.assertRaises(StopAfterGate):
                    namespace["_maybe_build_distillation_batch"](
                        fake_self, batch, feedback
                    )
                result = captured[-1]
                self.assertEqual(result[0], [] if equal_scores else [0, 4])
                self.assertEqual(result[1], [] if equal_scores else ["a0", "c0"])
                self.assertEqual(result[2].reused_samples, 2 * k)
                self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
