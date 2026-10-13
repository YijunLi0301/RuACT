import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_entry():
    spec = importlib.util.spec_from_file_location(
        "entry", ROOT / "experiments/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_framework_and_complete_matrix():
    assert [
        p.relative_to(ROOT).as_posix() for p in ROOT.rglob("verl") if p.is_dir()
    ] == ["verl"]
    for domain in ("medicine", "science"):
        for size in ("1.7b", "4b"):
            for name in ("GRPO", "RuACT-Teacher", "RuACT-Critic", "RuACT"):
                assert (ROOT / f"experiments/main/{domain}/{size}/{name}.sh").is_file()
            for name in ("GRPO", "Reweight", "RuscaRL"):
                assert (
                    ROOT / f"experiments/baselines/{domain}/{size}/{name}.sh"
                ).is_file()
    for name in (
        "GRPO-OPSD",
        "DAPO-style",
        "no-gate",
        "no-filter",
        "unreachable-RuACT-Teacher",
        "saturated-RuACT-Critic",
    ):
        assert (ROOT / f"experiments/analysis/medicine/4b/{name}.sh").is_file()


def test_all_profiles_share_framework_and_canonical_critic():
    entry = load_entry()
    for domain in ("medicine", "science"):
        for size in ("1.7b", "4b"):
            settings = entry.load_settings(domain, size, None)
            for method in entry.METHODS:
                base, output, command, sidecar, env = entry.build(
                    settings, method, preview=True
                )
                assert base == ROOT
                assert env["PYTHONPATH"].split(":")[0] == str(ROOT)
                if sidecar:
                    assert env["POSITIVE_REFINEMENT_BOUNDED_RETRIES"] == "1"
                    assert "++trainer.refinement.min_candidate_count=5" in command
                    assert "++trainer.refinement.max_candidate_count=5" in command
                    assert "++trainer.refinement.max_children_per_parent=3" in command
                    assert "++trainer.refinement.seed_candidate_cap=5" in command


def test_analysis_switches_are_explicit():
    entry = load_entry()
    settings = entry.load_settings("medicine", "4b", None)

    def result(method):
        _, _, cmd, _, env = entry.build(settings, method, preview=True)
        return cmd, env

    cmd, _ = result("no-gate")
    assert "++actor_rollout_ref.actor.distillation.teacher_gate=false" in cmd
    cmd, env = result("no-filter")
    assert env["RUACT_CRITIC_NO_FILTER"] == "1"
    cmd, _ = result("GRPO-OPSD")
    assert "++actor_rollout_ref.actor.distillation.all_rubrics=true" in cmd
    assert "++trainer.refinement.enabled=false" in cmd
    cmd, _ = result("DAPO-style")
    assert "++algorithm.dynamic_sampling.enabled=true" in cmd
    assert '++actor_rollout_ref.actor.policy_loss.loss_mode="vanilla"' in cmd
