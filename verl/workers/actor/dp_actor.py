import logging
import os
from types import SimpleNamespace
from typing import Optional
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_self_distillation_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.attention_utils import (
    index_first_axis,
    pad_input,
    rearrange,
    unpad_input,
)
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.actor.distillation_loss import (
    chunked_logsumexp_over_vocab,
    compose_distillation_loss,
    linear_beta_warmup,
    require_global_step,
)
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(
        self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float
    ) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = (
            student_out.logits if hasattr(student_out, "logits") else student_out[0]
        )
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        self._distillation_ema: Optional[list[torch.Tensor]] = None
        self._distillation_ema_params: Optional[tuple[torch.Tensor, ...]] = None
        self._distillation_swap_cache: Optional[list[torch.Tensor]] = None
        role = "Ref" if actor_optimizer is None else "Actor"
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)
        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")
        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits
        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(
            self.config.fsdp_config.get("dtype", "bfloat16")
        )
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        if self.config.get("calculate_sum_pi_squared", False):
            concentration_from_logits = verl_F.calculate_sum_pi_squared_from_logits
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(concentration_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else concentration_from_logits
            )
            assert not (
                self.use_fused_kernels or self.use_prefix_grouper
            ), f"calculate_sum_pi_squared is not supported with self.use_fused_kernels={self.use_fused_kernels!r} or self.use_prefix_grouper={self.use_prefix_grouper!r} for now."

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "sdpo":
            return
        teacher_regularization = getattr(
            self_distillation_cfg, "teacher_regularization", "ema"
        )
        if teacher_regularization != "ema":
            return
        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError(
                "EMA teacher requires a separate teacher_module in the actor worker."
            )
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(), self.actor_module.parameters()
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(
                    student_data, alpha=update_rate
                )

    def _distillation_ema_enabled(self) -> bool:
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "distillation":
            return False
        distillation_config = getattr(self.config, "distillation", None)
        if distillation_config is None:
            return False
        if not bool(distillation_config.get("ema_teacher", False)):
            return False
        decay = float(distillation_config.get("ema_decay", 0.0))
        return 0.0 < decay < 1.0

    def _distillation_ema_initialize(self) -> None:
        params = tuple(self.actor_module.parameters())
        if not params:
            raise RuntimeError(
                "RuACT-Teacher EMA teacher found no actor parameters to track"
            )
        self._distillation_ema_params = params
        with torch.no_grad():
            ema_buffers = []
            for p in params:
                buffer = torch.empty_like(
                    p.data, dtype=torch.float32, device=p.data.device
                )
                buffer.copy_(p.data)
                ema_buffers.append(buffer)
            for buffer in ema_buffers:
                if not torch.isfinite(buffer).all():
                    raise FloatingPointError(
                        "RuACT-Teacher EMA teacher initialized to non-finite values"
                    )
        self._distillation_ema = ema_buffers

    def _distillation_ema_update(self) -> None:
        if not self._distillation_ema_enabled() or self._distillation_ema is None:
            return
        decay = float(self.config.distillation.get("ema_decay", 0.995))
        alpha = 1.0 - decay
        with torch.no_grad():
            for buffer, p in zip(self._distillation_ema, self._distillation_ema_params):
                buffer.mul_(decay).add_(p.data.to(dtype=torch.float32), alpha=alpha)

    def _distillation_swap_in(self) -> bool:
        if self._distillation_ema is None or not self._distillation_ema_enabled():
            return False
        if self._distillation_swap_cache is None:
            self._distillation_swap_cache = [
                torch.empty_like(p.data) for p in self._distillation_ema_params
            ]
        with torch.no_grad():
            for cache, p, buffer in zip(
                self._distillation_swap_cache,
                self._distillation_ema_params,
                self._distillation_ema,
            ):
                cache.copy_(p.data)
                p.data.copy_(buffer.to(dtype=p.data.dtype))
        return True

    def _distillation_swap_out(self) -> None:
        if self._distillation_swap_cache is None:
            return
        with torch.no_grad():
            for cache, p in zip(
                self._distillation_swap_cache, self._distillation_ema_params
            ):
                p.data.copy_(cache)

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        calculate_sum_pi_squared: Optional[bool] = None,
        module: Optional[nn.Module] = None,
        topk_temperature: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        if calculate_sum_pi_squared is None:
            calculate_sum_pi_squared = self.config.get(
                "calculate_sum_pi_squared", False
            )
        sum_pi_squared_checkpointing = self.config.get(
            "sum_pi_squared_checkpointing", False
        )
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and (not use_topk)
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")
        topk_scale = 1.0
        if (
            use_topk
            and topk_temperature is not None
            and (float(topk_temperature) != float(temperature))
        ):
            topk_scale = float(temperature) / float(topk_temperature)
        model = module or self.actor_module
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and (not self.use_ulysses_sp)
                and (not self.use_fused_kernels)
                and (not self.use_dynamic_bsz)
                and (not return_all_logps)
                and (not use_topk)
            )
            if can_use_pg and "response_mask" in micro_batch and ("uid" in micro_batch):
                from verl.trainer.ppo.prefix_grouper_utils import (
                    forward_micro_batch_with_prefix_grouper,
                )

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get(
                        "entropy_from_logits_with_chunking", False
                    ),
                )
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(
                micro_batch["multi_modal_inputs"]
            )
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)
                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (
                                position_ids.shape[0],
                                1,
                                self.ulysses_sequence_parallel_size,
                            ),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import (
                        process_multi_modal_inputs_for_minicpmo,
                    )

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids,
                        attention_mask,
                        position_ids,
                        cu_seqlens,
                        multi_modal_inputs,
                    )
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config, "vision_config"
                    )
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = (
                            ulysses_pad_and_slice_inputs(
                                input_ids_rmpad,
                                position_ids_rmpad=position_ids_rmpad,
                                sp_size=self.ulysses_sequence_parallel_size,
                            )
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = (
                        torch.log_softmax(logits_rmpad, dim=-1)
                        if compute_all_logps
                        else None
                    )
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if calculate_entropy:
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                        )
                    if use_topk:
                        topk_logits_for_rank = (
                            logits_rmpad * topk_scale
                            if topk_scale != 1.0
                            else logits_rmpad
                        )
                        if topk_indices is None:
                            topk = min(distill_topk, topk_logits_for_rank.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(
                                topk_logits_for_rank, topk, dim=-1
                            )
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = (
                                topk_indices
                            )
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"),
                                indices,
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(
                                topk_logits_for_rank, dim=-1, index=topk_indices_rmpad
                            )
                        logsumexp_rmpad = chunked_logsumexp_over_vocab(
                            topk_logits_for_rank
                        ).unsqueeze(-1)
                        topk_logps_rmpad = topk_logits_rmpad.float() - logsumexp_rmpad
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                if calculate_sum_pi_squared:
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    response_token_mask = attention_mask[:, -response_length:].to(
                        dtype=torch.bool
                    )
                    topk_logps = topk_logps.masked_fill(
                        ~response_token_mask.unsqueeze(-1),
                        torch.finfo(topk_logps.dtype).min,
                    )
                    if return_topk_indices:
                        topk_indices = full_topk_indices[
                            :, -response_length - 1 : -1, :
                        ]
            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        topk_logits_for_rank = (
                            logits * topk_scale if topk_scale != 1.0 else logits
                        )
                        if topk_indices is None:
                            topk = min(distill_topk, topk_logits_for_rank.size(-1))
                            topk_logits, topk_indices = torch.topk(
                                topk_logits_for_rank, topk, dim=-1
                            )
                        else:
                            topk_logits = torch.gather(
                                topk_logits_for_rank, dim=-1, index=topk_indices
                            )
                        logsumexp = chunked_logsumexp_over_vocab(
                            topk_logits_for_rank
                        ).unsqueeze(-1)
                        topk_logps = topk_logits.float() - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                verl_F.entropy_from_logits, logits
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits
                            )
                        )
            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        if not torch.isfinite(grad_norm):
            print(
                f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}"
            )
            self.actor_optimizer.zero_grad()
            return grad_norm
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self, data: DataProto, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        calculate_sum_pi_squared = bool(
            self.config.get("calculate_sum_pi_squared", False)
        )
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")
        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )
        if use_dynamic_bsz:
            max_token_len = (
                data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            )
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len
            )
        else:
            micro_batches = data.split(micro_batch_size)
        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {
                **micro_batch.batch,
                **micro_batch.non_tensor_batch,
                "pad_token_id": pad_token_id,
            }
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    calculate_sum_pi_squared=calculate_sum_pi_squared,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])
        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)
        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        distillation_enabled = loss_mode == "distillation"
        distillation_config = getattr(self.config, "distillation", None)
        distillation_global_step = None
        distillation_beta_effective = None
        distillation_required_keys: set[str] = set()
        if self_distillation_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(
                set(data.batch.keys())
            ), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"
        if distillation_enabled:
            if distillation_config is None:
                raise ValueError(
                    "policy_loss.loss_mode=distillation requires actor.distillation config"
                )
            if self.config.ppo_epochs != 1 or self.config.shuffle:
                raise ValueError(
                    "RuACT-Teacher requires ppo_epochs=1 and shuffle=false"
                )
            if self.config.use_dynamic_bsz:
                raise ValueError(
                    "RuACT-Teacher requires fixed microbatches for query-mean scaling"
                )
            if self.use_fused_kernels:
                raise ValueError(
                    "RuACT-Teacher top-k logits require fused kernels to be disabled"
                )
            distillation_global_step = require_global_step(data.meta_info)
            distillation_beta_effective = linear_beta_warmup(
                beta=distillation_config.beta,
                current_step=distillation_global_step,
                warmup_steps=distillation_config.beta_warmup_steps,
            )
            distillation_required_keys = {
                "privileged_input_ids",
                "privileged_attention_mask",
                "privileged_position_ids",
                "distillation_anchor_mask",
            }
            missing = distillation_required_keys - set(data.batch.keys())
            if missing:
                raise ValueError(f"Missing RuACT-Teacher keys: {sorted(missing)}")
        distillation_topk_temperature = None
        distillation_ema_active = False
        if distillation_enabled:
            loss_temperature = float(distillation_config.get("loss_temperature", 1.0))
            if loss_temperature != 1.0:
                distillation_topk_temperature = loss_temperature
            distillation_ema_active = self._distillation_ema_enabled()
            if distillation_ema_active and self._distillation_ema is None:
                self._distillation_ema_initialize()
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        if distillation_enabled:
            select_keys.extend(list(distillation_required_keys))
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        metrics = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    if len(mini_batch) % self.config.ppo_micro_batch_size_per_gpu != 0:
                        raise ValueError(
                            "PPO mini batch must split into complete micro batches"
                        )
                    self.gradient_accumulation = (
                        len(mini_batch) // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )
                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                        "pad_token_id": pad_token_id,
                    }
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = (
                        self.config.calculate_entropy or entropy_coeff != 0
                    )
                    self_distillation_mask = (
                        model_inputs.get("self_distillation_mask")
                        if self_distillation_enabled
                        else None
                    )
                    distillation_anchor_mask = (
                        model_inputs.get("distillation_anchor_mask")
                        if distillation_enabled
                        else None
                    )
                    if self_distillation_enabled or distillation_enabled:
                        assert (
                            not has_multi_modal_inputs
                        ), "Multi-modal inputs are not supported for distillation"
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = (
                            response_mask.shape[0] / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    teacher_regularization = (
                        self_distillation_cfg.get("teacher_regularization", "ema")
                        if self_distillation_enabled
                        else "same-actor"
                    )
                    if (
                        teacher_regularization == "trust-region"
                        and self.use_fused_kernels
                    ):
                        raise ValueError(
                            "trust-region teacher requires disabling fused kernels to access logits."
                        )
                    return_all_logps = (
                        self_distillation_enabled
                        and self_distillation_cfg.full_logit_distillation
                        and (not self_distillation_cfg.distillation_topk)
                    )
                    distill_topk = (
                        self_distillation_cfg.distillation_topk
                        if self_distillation_enabled
                        and self_distillation_cfg.full_logit_distillation
                        else None
                    )
                    privileged_top_log_probs = None
                    privileged_top_indices = None
                    privileged_teacher_entropy = None
                    if distillation_enabled:
                        privileged_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["privileged_input_ids"],
                            "attention_mask": model_inputs["privileged_attention_mask"],
                            "position_ids": model_inputs["privileged_position_ids"],
                        }
                        ema_swapped = False
                        if distillation_ema_active:
                            ema_swapped = self._distillation_swap_in()
                        try:
                            with torch.no_grad():
                                privileged_outputs = self._forward_micro_batch(
                                    privileged_inputs,
                                    temperature=temperature,
                                    calculate_entropy=bool(
                                        distillation_config.entropy_weighting
                                    ),
                                    distill_topk=distillation_config.top_k,
                                    topk_temperature=distillation_topk_temperature,
                                    module=self.teacher_module or self.actor_module,
                                )
                        finally:
                            if ema_swapped:
                                self._distillation_swap_out()
                        privileged_top_log_probs = privileged_outputs[
                            "topk_logps"
                        ].detach()
                        privileged_top_indices = privileged_outputs[
                            "topk_indices"
                        ].detach()
                        if distillation_config.entropy_weighting:
                            privileged_teacher_entropy = privileged_outputs[
                                "entropys"
                            ].detach()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        calculate_sum_pi_squared=None,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                        topk_indices=(
                            privileged_top_indices if distillation_enabled else None
                        ),
                        topk_temperature=(
                            distillation_topk_temperature
                            if distillation_enabled
                            else None
                        ),
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = (
                        outputs.get("all_logps") if return_all_logps else None
                    )
                    student_topk_logps = (
                        outputs.get("topk_logps") if distill_topk else None
                    )
                    public_top_log_probs = (
                        outputs.get("topk_logps") if distillation_enabled else None
                    )
                    student_topk_indices = (
                        outputs.get("topk_indices") if distill_topk else None
                    )
                    if (
                        hasattr(self.config, "use_rollout_log_probs")
                        and self.config.use_rollout_log_probs
                    ):
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        teacher_model = self.teacher_module or self.actor_module
                        if teacher_regularization == "trust-region" and (
                            self.teacher_module is None
                            or self.teacher_module is self.actor_module
                        ):
                            raise ValueError(
                                "trust-region teacher requires a separate teacher_module in the actor worker."
                            )
                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = (
                            teacher_outputs.get("all_logps")
                            if return_all_logps
                            else None
                        )
                        teacher_topk_logps = (
                            teacher_outputs.get("topk_logps") if distill_topk else None
                        )
                        pg_loss, pg_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                        )
                        pg_metrics["self_distillation/empty_target_batch"] = (
                            self_distillation_mask.sum().item() == 0
                        )
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        policy_loss_fn = get_policy_loss_fn(
                            "vanilla" if distillation_enabled else loss_mode
                        )
                        vanilla_pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        pg_loss = vanilla_pg_loss
                        if distillation_enabled:
                            if (
                                public_top_log_probs is None
                                or privileged_top_log_probs is None
                                or distillation_anchor_mask is None
                            ):
                                raise RuntimeError(
                                    "RuACT-Teacher paired views were not computed"
                                )
                            rr_output = compose_distillation_loss(
                                vanilla_policy_loss=pg_loss,
                                public_top_log_probs=public_top_log_probs,
                                privileged_top_log_probs=privileged_top_log_probs,
                                response_mask=response_mask,
                                anchor_mask=distillation_anchor_mask,
                                beta=distillation_beta_effective,
                                group_size=int(self.config.rollout_n),
                                normalization=distillation_config.get(
                                    "normalization", "valid"
                                ),
                                token_kl_clip=distillation_config.token_kl_clip,
                                negative_kl_tolerance=distillation_config.negative_kl_tolerance,
                                teacher_entropy=privileged_teacher_entropy,
                                entropy_gamma=distillation_config.entropy_gamma,
                            )
                            pg_loss = rr_output.total_loss
                            valid_count = rr_output.valid_token_mask.sum()
                            active_count = rr_output.active_token_mask.sum()
                            clipped_count = rr_output.clipped_token_mask.sum()
                            negative_count = rr_output.negative_token_mask.sum()
                            valid_unclipped_kl = (
                                rr_output.unclipped_token_kl.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            valid_clipped_kl = rr_output.clipped_token_kl.masked_select(
                                rr_output.valid_token_mask
                            )
                            valid_teacher_top_mass = (
                                rr_output.teacher_top_mass.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            valid_student_top_mass = (
                                rr_output.student_top_mass.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            valid_teacher_entropy = (
                                rr_output.teacher_entropy.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            valid_raw_entropy_weight = (
                                rr_output.raw_entropy_weight.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            valid_entropy_weight = (
                                rr_output.entropy_weight.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            valid_weighted_token_kl = (
                                rr_output.weighted_token_kl.masked_select(
                                    rr_output.valid_token_mask
                                )
                            )
                            if valid_entropy_weight.numel():
                                entropy_weight_quantiles = torch.quantile(
                                    valid_entropy_weight.float(),
                                    torch.tensor(
                                        [0.1, 0.5, 0.9],
                                        device=valid_entropy_weight.device,
                                    ),
                                )
                                (
                                    entropy_weight_p10,
                                    entropy_weight_p50,
                                    entropy_weight_p90,
                                ) = entropy_weight_quantiles.unbind()
                                entropy_weight_max = valid_entropy_weight.max()
                            else:
                                entropy_weight_p10 = entropy_weight_p50 = (
                                    entropy_weight_p90
                                ) = entropy_weight_max = torch.zeros(
                                    (), device=rr_output.entropy_weight.device
                                )
                            negative_kl_abs_sum = (
                                -valid_unclipped_kl.clamp_max(0.0)
                            ).sum()
                            positive_kl_sum = valid_unclipped_kl.clamp_min(0.0).sum()
                            weighted_auxiliary_loss = (
                                distillation_beta_effective
                                * int(self.config.rollout_n)
                                * rr_output.auxiliary_loss
                            )
                            vanilla_pg_value = vanilla_pg_loss.detach().item()
                            aux_to_pg_ratio = (
                                weighted_auxiliary_loss.detach().item()
                                / vanilla_pg_value
                                if abs(vanilla_pg_value) > 1e-12
                                else 0.0
                            )
                            aggregate_metrics = {
                                "distillation/valid_token_count": valid_count.detach().item(),
                                "distillation/active_token_count": active_count.detach().item(),
                                "distillation/clipped_token_count": clipped_count.detach().item(),
                                "distillation/negative_token_count": negative_count.detach().item(),
                                "distillation/fkl_unclipped_sum": valid_unclipped_kl.sum()
                                .detach()
                                .item(),
                                "distillation/fkl_clipped_sum": valid_clipped_kl.sum()
                                .detach()
                                .item(),
                                "distillation/teacher_top_mass_sum": valid_teacher_top_mass.sum()
                                .detach()
                                .item(),
                                "distillation/student_top_mass_sum": valid_student_top_mass.sum()
                                .detach()
                                .item(),
                                "distillation/teacher_entropy_sum": valid_teacher_entropy.sum()
                                .detach()
                                .item(),
                                "distillation/entropy_weight_raw_sum": valid_raw_entropy_weight.sum()
                                .detach()
                                .item(),
                                "distillation/entropy_weight_sum": valid_entropy_weight.sum()
                                .detach()
                                .item(),
                                "distillation/entropy_weight_sq_sum": valid_entropy_weight.square()
                                .sum()
                                .detach()
                                .item(),
                                "distillation/entropy_weighted_fkl_sum": valid_weighted_token_kl.sum()
                                .detach()
                                .item(),
                                "distillation/negative_kl_abs_sum": negative_kl_abs_sum.detach().item(),
                                "distillation/positive_kl_sum": positive_kl_sum.detach().item(),
                                "distillation/teacher_top_mass_excess_sum": (
                                    rr_output.teacher_raw_top_mass.masked_select(
                                        rr_output.valid_token_mask
                                    )
                                    - 1.0
                                )
                                .clamp_min(0.0)
                                .sum()
                                .detach()
                                .item(),
                                "distillation/student_top_mass_excess_sum": (
                                    rr_output.student_raw_top_mass.masked_select(
                                        rr_output.valid_token_mask
                                    )
                                    - 1.0
                                )
                                .clamp_min(0.0)
                                .sum()
                                .detach()
                                .item(),
                                "distillation/response_count": rr_output.response_count.detach().item(),
                                "distillation/geometric_denominator_sum": rr_output.geometric_denominator_sum.detach().item(),
                                "distillation/normalization_gain_sum": rr_output.normalization_gain_sum.detach().item(),
                                "distillation/normalization_gain_count": rr_output.normalization_gain_count.detach().item(),
                                "distillation/zero_active_response_count": rr_output.zero_active_response_count.detach().item(),
                            }
                            pg_metrics.update(
                                {
                                    "distillation/beta": float(
                                        distillation_config.beta
                                    ),
                                    "distillation/effective_beta": distillation_beta_effective,
                                    "distillation/beta_warmup_steps": float(
                                        distillation_config.beta_warmup_steps
                                    ),
                                    "distillation/global_step_seen": float(
                                        distillation_global_step
                                    ),
                                    "distillation/is_fkl": 1.0,
                                    "distillation/top_k": float(
                                        distillation_config.top_k
                                    ),
                                    "distillation/has_tail_bucket": 1.0,
                                    "distillation/token_kl_clip": (
                                        float(distillation_config.token_kl_clip)
                                        if distillation_config.token_kl_clip is not None
                                        else 0.0
                                    ),
                                    "distillation/entropy_weighting": float(
                                        distillation_config.entropy_weighting
                                    ),
                                    "distillation/entropy_gamma": float(
                                        distillation_config.entropy_gamma
                                    ),
                                    "distillation/ema_teacher": float(
                                        bool(
                                            distillation_config.get(
                                                "ema_teacher", False
                                            )
                                        )
                                    ),
                                    "distillation/ema_decay": float(
                                        distillation_config.get("ema_decay", 0.0)
                                    ),
                                    "distillation/loss_temperature": float(
                                        distillation_config.get("loss_temperature", 1.0)
                                    ),
                                    "distillation/entropy_weight_p10": entropy_weight_p10.detach().item(),
                                    "distillation/entropy_weight_p50": entropy_weight_p50.detach().item(),
                                    "distillation/entropy_weight_p90": entropy_weight_p90.detach().item(),
                                    "distillation/entropy_weight_max": entropy_weight_max.detach().item(),
                                    "distillation/vanilla_pg_loss": vanilla_pg_loss.detach().item(),
                                    "distillation/auxiliary_loss": rr_output.auxiliary_loss.detach().item(),
                                    "distillation/anchor_mean_loss": rr_output.anchor_mean_loss.detach().item(),
                                    "distillation/weighted_aux_loss": weighted_auxiliary_loss.detach().item(),
                                    "distillation/auxiliary_loss_valid_norm": rr_output.auxiliary_loss_valid_norm.detach().item(),
                                    "distillation/auxiliary_loss_geo_norm": rr_output.auxiliary_loss_geo_norm.detach().item(),
                                    "distillation/aux_term": rr_output.aux_term.detach().item(),
                                    "distillation/aux_to_pg_ratio": aux_to_pg_ratio,
                                    "distillation/geometric_active_norm": float(
                                        distillation_config.get(
                                            "normalization", "valid"
                                        )
                                        == "geometric_active"
                                    ),
                                    "distillation/total_pg_loss": rr_output.total_loss.detach().item(),
                                    "distillation/negative_kl_min": (
                                        valid_unclipped_kl.min().detach().item()
                                        if valid_unclipped_kl.numel()
                                        else 0.0
                                    ),
                                }
                            )
                            pg_metrics.update(aggregate_metrics)
                        micro_batch_metrics.update(pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import (
                            compute_rollout_corr_metrics_from_logprobs,
                        )

                        rollout_corr_metrics = (
                            compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)
                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        micro_batch_metrics["actor/entropy"] = (
                            entropy_agg.detach().item()
                        )
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += (
                            kl_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    metrics["actor/pg_loss"] += (
                        pg_loss.detach().item() * loss_scale_factor
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                grad_norm = self._optimizer_step()
                grad_norm_finite = torch.isfinite(grad_norm).item()
                if grad_norm_finite:
                    did_update = True
                    if distillation_ema_active:
                        self._distillation_ema_update()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
        return metrics
