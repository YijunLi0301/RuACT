from abc import ABC, abstractmethod
from torch.distributed.device_mesh import DeviceMesh
from verl import DataProto
from verl.workers.config import HFModelConfig, RewardModelConfig

__all__ = ["BasePPORewardModel"]


class BasePPORewardModel(ABC):
    def __init__(
        self,
        config: RewardModelConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        pass
