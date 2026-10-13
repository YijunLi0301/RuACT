from abc import abstractmethod
from collections.abc import Sized
from omegaconf import DictConfig
from torch.utils.data import Sampler
from verl import DataProto


class AbstractSampler(Sampler[int]):
    @abstractmethod
    def __init__(self, data_source: Sized, data_config: DictConfig):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass
