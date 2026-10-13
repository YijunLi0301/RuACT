from enum import Enum
from typing import Any, Optional, Union
import numpy as np
import torch


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


NumericType = (int, float, torch.Tensor)
Numeric = int | float | torch.Tensor


class Metric:
    def __init__(
        self,
        aggregation: str | AggregationType,
        value: Optional[Numeric | list[Numeric]] = None,
    ) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = []
        if value is not None:
            self.append(value)

    def append(self, value: Union[Numeric, "Metric"]) -> None:
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach().item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(
                    f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}"
                )
            values = values.values
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        match self.aggregation:
            case AggregationType.MEAN:
                return np.mean(self.values)
            case AggregationType.SUM:
                return np.sum(self.values)
            case AggregationType.MIN:
                return np.min(self.values)
            case AggregationType.MAX:
                return np.max(self.values)

    @classmethod
    def chain(cls, metric_lists: list["Metric"]) -> "Metric":
        if len(metric_lists) == 0:
            return cls(aggregation=AggregationType.MEAN)
        aggregation = metric_lists[0].aggregation
        chained = cls(aggregation=aggregation)
        for ml in metric_lists:
            chained.extend(ml)
        return chained

    @classmethod
    def from_dict(
        cls, data: dict[str, Numeric], aggregation: str | AggregationType
    ) -> dict[str, "Metric"]:
        return {
            key: cls(value=value, aggregation=aggregation)
            for key, value in data.items()
        }

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation)
