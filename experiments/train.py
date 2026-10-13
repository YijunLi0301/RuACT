import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "GRPO",
    "RuACT",
    "RuACT-Teacher",
    "RuACT-Critic",
    "Reweight",
    "RuscaRL",
    "GRPO-OPSD",
    "DAPO-style",
    "no-gate",
    "no-filter",
    "unreachable-RuACT-Teacher",
    "saturated-RuACT-Critic",
    "saturated-RuACT-Teacher",
    "unreachable-RuACT-Critic",
)


def _merge(target, values):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def load_settings(domain, size, override_path):
    settings = json.loads((ROOT / "experiments/settings.json").read_text())
    profile = json.loads(
        (ROOT / "experiments/profiles" / (domain + "-" + size + ".json")).read_text()
    )
    _merge(settings, profile)
    if override_path:
        _merge(settings, json.loads(Path(override_path).read_text()))
    settings["domain"] = domain
    settings["size"] = size
    return settings


def build(settings, method, preview=False):
    if method not in METHODS:
        raise ValueError("Unknown method")
    base = ROOT
    use_distillation = method in (
        "RuACT",
        "RuACT-Teacher",
        "GRPO-OPSD",
        "no-gate",
        "unreachable-RuACT-Teacher",
        "saturated-RuACT-Teacher",
    )
    use_refinement = method in (
        "RuACT",
        "RuACT-Critic",
        "no-filter",
        "saturated-RuACT-Critic",
        "unreachable-RuACT-Critic",
    )
    paths = dict(settings["paths"])
    for key in ("model", "train_data", "validation_data", "seed_library"):
        if paths.get(key):
            paths[key] = str(Path(paths[key]).expanduser().resolve())
    output = Path(
        paths["output"]
        or str(
            Path("outputs")
            / settings.get("domain", "medicine")
            / settings.get("size", "4b")
            / method
        )
    ).resolve()
    values = dict(settings["training"])
    values.update(settings.get("method_overrides", {}).get(method, {}))
    for name in ("model", "train_data", "validation_data"):
        if not preview and (not paths[name] or not Path(paths[name]).exists()):
            raise ValueError(f"Configure an existing paths.{name}")
    gpu_ids = settings["gpu_ids"]
    if not preview and (not gpu_ids):
        raise ValueError("Configure gpu_ids explicitly")
    overrides = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": paths["train_data"],
        "data.val_files": paths["validation_data"],
        "data.train_batch_size": values["batch_size"],
        "data.val_batch_size": values["batch_size"],
        "data.seed": values["seed"],
        "data.max_prompt_length": values["prompt_length"],
        "data.max_response_length": values["response_length"],
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": paths["model"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.policy_loss.loss_mode": (
            "distillation" if use_distillation else "vanilla"
        ),
        "actor_rollout_ref.actor.ppo_mini_batch_size": values["mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": values[
            "micro_batch_size"
        ],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": values["kl_coefficient"],
        "actor_rollout_ref.actor.kl_loss_type": "k1",
        "actor_rollout_ref.actor.optim.lr": values["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 12,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": values["rollouts"],
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": values[
            "rollout_memory_fraction"
        ],
        "actor_rollout_ref.rollout.max_model_len": values["model_length"],
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": values[
            "logprob_micro_batch_size"
        ],
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": values[
            "logprob_micro_batch_size"
        ],
        "reward_model.reward_manager": "batch",
        "custom_reward_function.path": str(
            ROOT
            / "scripts/rubric_rl"
            / (
                "reweight_reward.py"
                if method == "Reweight"
                else "rusca_reward.py" if method == "RuscaRL" else "verl_reward.py"
            )
        ),
        "custom_reward_function.name": "compute_score_stream_single",
        "trainer.n_gpus_per_node": len(gpu_ids) or 2,
        "trainer.nnodes": 1,
        "trainer.total_epochs": values["epochs"],
        "trainer.total_training_steps": values["steps"],
        "trainer.save_freq": values["save_every"],
        "trainer.test_freq": values["validate_every"],
        "trainer.val_before_train": False,
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.max_critic_ckpt_to_keep": 2,
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.logger": ["console"],
        "trainer.project_name": "RuACT",
        "trainer.experiment_name": method,
        "trainer.resume_mode": "disable",
        "ray_kwargs.ray_init.address": "local",
        "ray_kwargs.ray_init.include_dashboard": False,
    }
    overrides.update(
        {
            "reward_model.use_reward_loop": True,
            "reward_model.reward_loop_source": "importlib",
            "reward_model.reward_loop_module_path": str(
                base / "scripts/rubric_rl/group_reward_loop.py"
            ),
            "reward_model.reward_loop_class_name": "ImmediateJudgeRewardLoopManager",
            "reward_model.query_group_size": values["rollouts"],
            "reward_model.max_inflight_groups_per_worker": 8,
            "trainer.refinement.enabled": use_refinement,
        }
    )
    if use_distillation:
        d = "actor_rollout_ref.actor.distillation."
        overrides.update(
            {
                d + "beta": values["beta_target"] / values["rollouts"],
                d + "beta_warmup_steps": 50,
                d + "normalization": "valid",
                d + "max_rubrics": 3,
                d + "neutral_rubric_prompt": False,
                d + "teacher_gate": method != "no-gate",
                d + "teacher_k": 1,
                d + "teacher_final_only": False,
                d + "teacher_enable_thinking": False,
                d + "ema_teacher": False,
                d + "loss_temperature": 1.0,
                d + "token_kl_clip": 10.0,
                d + "include_negative_weight_failures": True,
                d + "all_rubrics": method == "GRPO-OPSD",
                d
                + "selection_mode": (
                    "saturated"
                    if method == "saturated-RuACT-Teacher"
                    else "unreachable"
                ),
                d
                + "max_privileged_length": (
                    16384 if method == "GRPO-OPSD" else values["model_length"]
                ),
            }
        )
    if use_refinement:
        overrides.update(
            {
                "trainer.refinement.state_dir": str(output / "refinement"),
                "trainer.refinement.mode": "active",
                "trainer.refinement.blend_lambda": values["refinement_lambda"],
                "trainer.refinement.min_candidate_count": 5,
                "trainer.refinement.max_candidate_count": 5,
                "trainer.refinement.max_children_per_parent": 3,
                "trainer.refinement.seed_library_path": paths.get("seed_library")
                or None,
                "trainer.refinement.seed_candidate_cap": 5,
                "trainer.refinement.max_queries_per_step": 32,
                "trainer.refinement.include_negative_weight": False,
            }
        )
    if method == "Reweight":
        overrides.update(
            {
                "trainer.simple_reweight.enabled": True,
                "trainer.simple_reweight.eta": 0.5,
            }
        )
    if method == "RuscaRL":
        overrides.update(
            {
                "trainer.ruscarl.enabled": True,
                "trainer.ruscarl.start": 0.2,
                "trainer.ruscarl.steepness": 125.0,
                "actor_rollout_ref.rollout.calculate_log_probs": False,
            }
        )
    if method == "DAPO-style":
        overrides.update(
            {
                "algorithm.dynamic_sampling.enabled": True,
                "algorithm.dynamic_sampling.std_epsilon": 1e-08,
                "algorithm.dynamic_sampling.max_rounds": 8,
                "reward_model.launch_reward_fn_async": False,
            }
        )
    if use_refinement and (not preview):
        seed = paths.get("seed_library")
        if seed and (not Path(seed).is_file()):
            raise ValueError("Configured seed library does not exist")
        if not seed and (not settings.get("allow_empty_seed_library", False)):
            raise ValueError(
                "Configure paths.seed_library; empty-seed mode must be explicitly selected"
            )
    overrides.update(settings.get("hydra_overrides", {}))
    command = [sys.executable, "-m", "verl.trainer.main_ppo"] + [
        "++" + k + "=" + json.dumps(v, separators=(",", ":"))
        for k, v in overrides.items()
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(base), str(base / "scripts")])
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    environment["WANDB_MODE"] = "disabled"
    environment["HEALTHBENCH_RUN_ID"] = method
    environment["DISTILLATION_POSITIVE_BLEND_LAMBDA"] = str(values["refinement_lambda"])
    for role in ("judge", "generator"):
        for field, suffix in [
            ("api_base", "API_BASE"),
            ("api_key", "API_KEY"),
            ("model", "MODEL"),
        ]:
            key = "HEALTHBENCH_" + role.upper() + "_" + suffix
            environment[key] = os.environ.get(key) or settings[role].get(field, "")
            if (
                not preview
                and (role == "judge" or use_refinement)
                and (not environment[key])
            ):
                raise ValueError(f"Set {key} or configure {role}.{field}")
    environment["HEALTHBENCH_JUDGE_CANONICAL_API_BASE"] = environment[
        "HEALTHBENCH_JUDGE_API_BASE"
    ]
    environment["POSITIVE_REFINEMENT_BOUNDED_RETRIES"] = "1"
    environment["RUACT_CRITIC_NO_FILTER"] = "1" if method == "no-filter" else "0"
    environment["RUACT_CRITIC_SELECTION"] = (
        "unreachable" if method == "unreachable-RuACT-Critic" else "saturated"
    )
    environment["POSITIVE_REFINEMENT_SKIP_AFTER_EMPTY_ATTEMPTS"] = "3"
    environment["REWEIGHT_JUDGES_CONFIG"] = str(output / "judges.json")
    sidecar = (
        [
            sys.executable,
            str(base / "scripts/rubric_rl/refinement_sidecar.py"),
            "--state-dir",
            str(output / "refinement"),
            "--backend",
            "litellm",
            "--execution-mode",
            "single_pass_batched",
            "--workers",
            "8",
            "--max-jobs",
            "8",
            "--parent-workers",
            "3",
            "--parent-inflight",
            "8",
            "--generator-inflight",
            "4",
            "--judge-inflight",
            "8",
            "--max-attempts",
            "1",
            "--temperature",
            "0.8",
            "--max-tokens",
            "4096",
            "--watch",
        ]
        if use_refinement
        else None
    )
    if sidecar and paths.get("seed_library"):
        sidecar += ["--preverified-library-path", paths["seed_library"]]
    return (base, output, command, sidecar, environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--domain", choices=("medicine", "science"), default="medicine")
    parser.add_argument("--size", choices=("1.7b", "4b"), default="4b")
    parser.add_argument("--settings", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.domain, args.size, args.settings)
    base, output, command, sidecar, environment = build(
        settings, args.method, args.dry_run
    )
    if args.dry_run:
        print(
            json.dumps(
                {"package_root": str(base), "command": command, "sidecar": sidecar},
                indent=2,
            )
        )
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory for a fresh experiment")
    output.mkdir(parents=True, exist_ok=True)
    if args.method == "Reweight":
        (output / "judges.json").write_text(
            json.dumps(
                {
                    "endpoints": [
                        {
                            "id": "judge",
                            "base_url": environment["HEALTHBENCH_JUDGE_API_BASE"],
                            "model": environment["HEALTHBENCH_JUDGE_MODEL"],
                            "api_key_env": "HEALTHBENCH_JUDGE_API_KEY",
                            "enabled": True,
                            "max_concurrency": 32,
                            "weight": 1,
                            "timeout_s": 180,
                            "extra_body": {
                                "enable_thinking": False,
                                "response_format": {"type": "json_object"},
                            },
                        }
                    ]
                }
            )
        )
    worker = None
    trainer = None
    try:
        if sidecar:
            worker = subprocess.Popen(sidecar, cwd=base, env=environment)
            time.sleep(1)
            if worker.poll() is not None:
                raise RuntimeError("Refinement sidecar failed to start")
        trainer = subprocess.Popen(command, cwd=base, env=environment)
        while trainer.poll() is None:
            if worker and worker.poll() is not None:
                raise RuntimeError("Refinement sidecar exited during training")
            time.sleep(1)
        if trainer.returncode:
            raise SystemExit(trainer.returncode)
    finally:
        for process in (trainer, worker):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
