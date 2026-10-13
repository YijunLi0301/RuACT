from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Optional
from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig
from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "DistillationConfig",
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class DistillationConfig(BaseConfig):
    beta: float = 0.05
    beta_warmup_steps: int = 50
    top_k: int = 64
    normalization: str = "valid"
    token_kl_clip: Optional[float] = None
    negative_kl_tolerance: float = 1e-05
    entropy_weighting: bool = False
    entropy_gamma: float = 1.0
    max_rubrics: int = 3
    include_negative_weight_failures: bool = False
    all_rubrics: bool = False
    max_privileged_length: Optional[int] = None
    selection_mode: str = "unreachable"
    neutral_rubric_prompt: bool = False
    teacher_enable_thinking: bool = False
    teacher_final_only: bool = False
    teacher_gate: bool = False
    teacher_k: int = 1
    teacher_max_response_length: int = 1024
    ema_teacher: bool = False
    ema_decay: float = 0.995
    loss_temperature: float = 1.0

    def __post_init__(self):
        coefficient = float(self.beta)
        if not isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("distillation.beta must be finite and non-negative")
        if type(self.beta_warmup_steps) is not int or self.beta_warmup_steps <= 0:
            raise ValueError(
                "distillation.beta_warmup_steps must be a positive integer"
            )
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("distillation.top_k must be a positive integer")
        if self.normalization not in ("valid", "geometric_active"):
            raise ValueError(
                "distillation.normalization must be 'valid' or 'geometric_active'"
            )
        if self.token_kl_clip is not None:
            token_kl_clip = float(self.token_kl_clip)
            if not isfinite(token_kl_clip) or token_kl_clip <= 0.0:
                raise ValueError(
                    "distillation.token_kl_clip must be finite and positive"
                )
        negative_kl_tolerance = float(self.negative_kl_tolerance)
        if not isfinite(negative_kl_tolerance) or negative_kl_tolerance < 0.0:
            raise ValueError(
                "distillation.negative_kl_tolerance must be finite and non-negative"
            )
        if type(self.entropy_weighting) is not bool:
            raise ValueError("distillation.entropy_weighting must be a boolean")
        entropy_gamma = float(self.entropy_gamma)
        if not isfinite(entropy_gamma) or entropy_gamma < 0.0:
            raise ValueError(
                "distillation.entropy_gamma must be finite and non-negative"
            )
        if type(self.max_rubrics) is not int or self.max_rubrics <= 0:
            raise ValueError("distillation.max_rubrics must be a positive integer")
        if type(self.include_negative_weight_failures) is not bool:
            raise ValueError(
                "distillation.include_negative_weight_failures must be a boolean"
            )
        if type(self.neutral_rubric_prompt) is not bool:
            raise ValueError("distillation.neutral_rubric_prompt must be a boolean")
        if (
            type(self.teacher_max_response_length) is not int
            or self.teacher_max_response_length <= 0
        ):
            raise ValueError(
                "distillation.teacher_max_response_length must be a positive integer"
            )
        if type(self.teacher_gate) is not bool:
            raise ValueError("distillation.teacher_gate must be a boolean")
        if type(self.teacher_k) is not int or not 1 <= self.teacher_k <= 4:
            raise ValueError("distillation.teacher_k must be an integer in [1, 4]")
        if self.teacher_k > 1 and (not self.teacher_gate):
            raise ValueError(
                "distillation.teacher_k > 1 requires distillation.teacher_gate=True"
            )
        if type(self.ema_teacher) is not bool:
            raise ValueError("distillation.ema_teacher must be a boolean")
        ema_decay = float(self.ema_decay)
        if not isfinite(ema_decay) or not 0.0 < ema_decay < 1.0:
            raise ValueError("distillation.ema_decay must lie in (0, 1)")
        loss_temperature = float(self.loss_temperature)
        if not isfinite(loss_temperature) or loss_temperature <= 0.0:
            raise ValueError(
                "distillation.loss_temperature must be finite and positive"
            )


@dataclass
class SelfDistillationConfig(BaseConfig):
    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
    )
    solution_template: str = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
    feedback_template: str = (
        "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"self_distillation.alpha must be in [0,1], got {self.alpha}"
            )
        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                f"self_distillation.teacher_regularization must be one of {valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(
                f"self_distillation.is_clip must be positive, got {self.is_clip}"
            )


@dataclass
class RouterReplayConfig(BaseConfig):
    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}"
            )


@dataclass
class PolicyLossConfig(BaseConfig):
    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }
    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(
        default_factory=SelfDistillationConfig
    )
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if (
                self.ppo_micro_batch_size is not None
                and self.ppo_micro_batch_size_per_gpu is not None
            ):
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is supported (the former is deprecated)."
                )
            else:
                assert not (
                    self.ppo_micro_batch_size is None
                    and self.ppo_micro_batch_size_per_gpu is None
                ), "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or 'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )
            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"
        if mbs is None and mbs_per_gpu is None:
            raise ValueError(
                f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
            )
        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.fsdp_config
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = (
                self.ulysses_sequence_parallel_size
            )

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        super().validate(n_gpus, train_batch_size, model_config)
        if (
            self.strategy in {"fsdp", "fsdp2"}
            and self.ulysses_sequence_parallel_size > 1
        ):
            if model_config and (not model_config.get("use_remove_padding", False)):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
