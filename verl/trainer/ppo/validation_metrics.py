from numbers import Real
from typing import Any, Mapping
import numpy as np


def _is_scalar_real_numeric(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        return False
    return isinstance(value, (Real, np.number))


def select_scalar_validation_metric_infos(
    infos_dict: Mapping[str, list[Any]],
) -> dict[str, list[Any]]:
    selected = {}
    for name, values in infos_dict.items():
        if name == "pred":
            if len(values) > 0 and all((isinstance(value, str) for value in values)):
                selected[name] = values
        elif len(values) > 0 and all(
            (_is_scalar_real_numeric(value) for value in values)
        ):
            selected[name] = values
    return selected
