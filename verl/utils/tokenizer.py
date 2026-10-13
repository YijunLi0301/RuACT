import types
import warnings

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(
            f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}",
            stacklevel=1,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(
            f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}",
            stacklevel=1,
        )


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    from transformers import AutoTokenizer

    if (
        correct_gemma2
        and isinstance(name_or_path, str)
        and ("gemma-2-2b-it" in name_or_path)
    ):
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.",
            stacklevel=1,
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    from transformers import AutoConfig, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)
        processor.config = config
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                processor.get_rope_index = types.MethodType(
                    Qwen2VLModel.get_rope_index, processor
                )
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                processor.get_rope_index = types.MethodType(
                    Qwen2_5_VLModel.get_rope_index, processor
                )
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                processor.get_rope_index = types.MethodType(
                    Qwen3VLModel.get_rope_index, processor
                )
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                processor.get_rope_index = types.MethodType(
                    Glm4vModel.get_rope_index, processor
                )
            case _:
                raise ValueError(
                    f"Unsupported processor type: {processor.__class__.__name__}"
                )
    except Exception as e:
        processor = None
        warnings.warn(
            f"Failed to create processor: {e}. This may affect multimodal processing",
            stacklevel=1,
        )
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
