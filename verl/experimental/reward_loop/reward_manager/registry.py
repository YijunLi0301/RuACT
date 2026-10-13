from typing import Callable
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

__all__ = ["register", "get_reward_manager_cls"]
REWARD_LOOP_MANAGER_REGISTRY: dict[str, type[RewardManagerBase]] = {}


def register(name: str) -> Callable[[type[RewardManagerBase]], type[RewardManagerBase]]:

    def decorator(cls: type[RewardManagerBase]) -> type[RewardManagerBase]:
        if (
            name in REWARD_LOOP_MANAGER_REGISTRY
            and REWARD_LOOP_MANAGER_REGISTRY[name] != cls
        ):
            raise ValueError(
                f"reward manager {name} has already been registered: {REWARD_LOOP_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_LOOP_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:
    if name not in REWARD_LOOP_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_LOOP_MANAGER_REGISTRY[name]
