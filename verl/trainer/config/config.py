from dataclasses import dataclass, field
from typing import Any, Optional
from verl.base_config import BaseConfig

__all__ = ["CheckpointConfig", "ProfileConfig", "BaseModelConfig"]


@dataclass
class CheckpointConfig(BaseConfig):
    save_contents: list[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"]
    )
    load_contents: list[str] = field(
        default_factory=lambda: ["model", "optimizer", "extra"]
    )
    async_save: bool = False


@dataclass
class ProfileConfig(BaseConfig):
    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class BaseModelConfig(BaseConfig):
    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False
    lora: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class RewardManagerConfig(BaseConfig):
    source: str = "register"
    name: str = "naive"
    module: Optional[ModuleConfig] = field(default_factory=ModuleConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.source == "register":
            from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY

            assert (
                self.name in REWARD_MANAGER_REGISTRY
            ), f"Reward manager is not registered: self.name={self.name!r} ,REWARD_MANAGER_REGISTRY.keys()={REWARD_MANAGER_REGISTRY.keys()!r}"
        elif self.source == "importlib":
            assert (
                self.module is not None and self.module.path is not None
            ), "When source is importlib, module.path should be set."
