from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = (None, None, None, None)


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input
    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import (
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )
    else:
        from flash_attn.bert_padding import (
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )
    _index_first_axis, _pad_input, _rearrange, _unpad_input = (
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )
    return (_index_first_axis, _pad_input, _rearrange, _unpad_input)


def index_first_axis(*args, **kwargs):
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
