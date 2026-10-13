from abc import ABC, abstractmethod
from typing import Generator, TypedDict
import torch


class TensorMeta(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class CheckpointEngineRegistry:
    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend} not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):
    @abstractmethod
    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ):
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield from self.weights
        self.weights = None
