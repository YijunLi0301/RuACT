from collections import defaultdict
import math
import numpy as np
import torch
from verl import DataProto


class DynamicSamplingBuffer:

    def __init__(self, *, target_groups, group_size, std_epsilon=1e-08, max_rounds=8):
        if target_groups < 1 or group_size < 2 or max_rounds < 1:
            raise ValueError("positive target/round count and group_size>=2 required")
        if not math.isfinite(std_epsilon) or std_epsilon < 0:
            raise ValueError("std_epsilon must be finite and nonnegative")
        self.target = int(target_groups)
        self.n = int(group_size)
        self.epsilon = float(std_epsilon)
        self.max_rounds = int(max_rounds)
        self.parts = []
        self.seen_uids = set()
        self.counts = defaultdict(float)
        self.taken = False
        self.extra_keys = None

    def add(self, batch, rewards):
        if self.taken or self.counts["retained_groups"] >= self.target:
            raise RuntimeError(
                "collection already complete; use a new buffer for the next policy"
            )
        if (
            rewards.ndim != 2
            or rewards.shape[0] != len(batch)
            or (not torch.isfinite(rewards).all())
        ):
            raise ValueError("non-finite or misaligned token-level rewards")
        for key in ("uid", "rollout_index", "judge_valid"):
            if key not in batch.non_tensor_batch:
                raise ValueError(f"missing required evidence: {key}")
        if not all(
            (
                isinstance(x, (bool, np.bool_)) and bool(x)
                for x in batch.non_tensor_batch["judge_valid"]
            )
        ):
            raise ValueError(
                "invalid judge result; refusing to convert failure into a filtered group"
            )
        keys = tuple(batch.meta_info.get("reward_extra_keys", ()))
        if not keys or "judge_valid" not in keys:
            raise ValueError("reward metadata schema must include judge_valid")
        if self.extra_keys is not None and set(keys) != set(self.extra_keys):
            raise ValueError("reward metadata schema changed between sampling rounds")
        self.extra_keys = keys
        for key in keys:
            if key not in batch.non_tensor_batch or len(
                batch.non_tensor_batch[key]
            ) != len(batch):
                raise ValueError(f"misaligned reward metadata: {key}")
        groups = defaultdict(list)
        for i, uid in enumerate(batch.non_tensor_batch["uid"]):
            groups[str(uid)].append(i)
        if not groups:
            raise ValueError("empty candidate batch")
        for uid, rows in groups.items():
            if uid in self.seen_uids:
                raise ValueError("reused rollout group UID across sampling rounds")
            if len(rows) != self.n:
                raise ValueError("incomplete rollout group")
            indices = batch.non_tensor_batch["rollout_index"][rows].tolist()
            if sorted(indices) != list(range(self.n)):
                raise ValueError("duplicate or missing rollout_index within group")
        self.seen_uids.update(groups)
        self.counts["rounds"] += 1
        self.counts["sampled_groups"] += len(groups)
        self.counts["sampled_responses"] += len(batch)
        mask = batch.batch["response_mask"]
        response_tokens = int(mask.sum().item())
        all_tokens = int(batch.batch["attention_mask"].sum().item())
        self.counts["response_tokens"] += response_tokens
        self.counts["prompt_tokens"] += all_tokens - response_tokens
        if (
            "judge_called" in batch.non_tensor_batch
            and "judge_attempts" in batch.non_tensor_batch
        ):
            self.counts["judge_cost_observed_responses"] += len(batch)
            self.counts["judge_scored_responses"] += sum(
                (bool(x) for x in batch.non_tensor_batch["judge_called"])
            )
            attempts = np.asarray(batch.non_tensor_batch["judge_attempts"], dtype=float)
            if not np.isfinite(attempts).all() or (attempts < 0).any():
                raise ValueError("invalid judge attempt accounting")
            self.counts["judge_attempts"] += float(attempts.sum())
        totals = rewards.sum(-1).detach().cpu().double().numpy()
        retained_rows = []
        for uid, rows in groups.items():
            std = float(np.std(totals[rows], ddof=0))
            if std <= self.epsilon:
                self.counts["rejected_groups"] += 1
            else:
                self.counts["informative_groups"] += 1
                if self.counts["retained_groups"] < self.target:
                    retained_rows.extend(rows)
                    self.counts["retained_groups"] += 1
                else:
                    self.counts["overflow_groups"] += 1
        if retained_rows:
            selected = batch[np.asarray(retained_rows, dtype=np.int64)]
            selected.batch["_dynamic_sampling_reward"] = rewards[retained_rows].clone()
            selected.meta_info = {
                key: value
                for key, value in batch.meta_info.items()
                if key
                not in {"global_token_num", "images_seqlens", "timing", "metrics"}
            }
            self.parts.append(selected)
        ready = self.counts["retained_groups"] == self.target
        if not ready and self.counts["rounds"] >= self.max_rounds:
            raise RuntimeError(
                f"dynamic sampling round limit exhausted: {dict(self.counts)}"
            )
        return ready

    def take(self):
        if self.taken or self.counts["retained_groups"] != self.target:
            raise RuntimeError(
                "cannot update from an incomplete or consumed collection"
            )
        batch = DataProto.concat(self.parts)
        rewards = batch.batch.pop("_dynamic_sampling_reward")
        extras = {key: batch.non_tensor_batch[key] for key in self.extra_keys}
        assert len(batch) == self.target * self.n
        self.parts.clear()
        self.taken = True
        return (batch, rewards, extras)

    def metrics(self):
        names = (
            "rounds",
            "sampled_groups",
            "sampled_responses",
            "rejected_groups",
            "informative_groups",
            "retained_groups",
            "overflow_groups",
            "response_tokens",
            "prompt_tokens",
            "judge_cost_observed_responses",
            "judge_scored_responses",
            "judge_attempts",
        )
        result = {f"dynamic_sampling/{key}": float(self.counts[key]) for key in names}
        sampled = self.counts["sampled_groups"]
        result["dynamic_sampling/reject_fraction"] = (
            self.counts["rejected_groups"] / sampled if sampled else 0.0
        )
        result["dynamic_sampling/sampling_multiplier"] = sampled / self.target
        return result


def validate_dynamic_sampling_config(config):
    ds = config.algorithm.get("dynamic_sampling", {})
    if not ds.get("enabled", False):
        return False
    actor = config.actor_rollout_ref.actor
    if (
        config.algorithm.adv_estimator != "grpo"
        or actor.policy_loss.loss_mode != "vanilla"
    ):
        raise ValueError(
            "dynamic sampling control requires vanilla GRPO without teacher distillation"
        )
    if config.algorithm.use_kl_in_reward:
        raise ValueError(
            "filter original rubric rewards; keep baseline KL on the loss side"
        )
    if config.reward_model.launch_reward_fn_async:
        raise ValueError(
            "dynamic sampling requires synchronous extraction of streaming rewards"
        )
    if config.trainer.get("refinement", {}).get("enabled", False):
        raise ValueError(
            "Rubric refinement must be disabled for the dynamic sampling control"
        )
    if config.trainer.get("dynamic_rubric", {}).get("enabled", False):
        raise ValueError("dynamic rubric must be disabled")
    if actor.get("self_distillation", {}).get("enabled", False):
        raise ValueError("self distillation must be disabled")
    if config.trainer.get("simple_reweight", {}).get("enabled", False):
        raise ValueError("reweight must be disabled")
    DynamicSamplingBuffer(
        target_groups=int(config.data.train_batch_size),
        group_size=int(config.actor_rollout_ref.rollout.n),
        std_epsilon=float(ds.get("std_epsilon", 1e-08)),
        max_rounds=int(ds.get("max_rounds", 8)),
    )
    return True
