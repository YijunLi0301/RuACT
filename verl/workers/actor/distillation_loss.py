from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, MutableMapping
import torch

NORMALIZATION_VALID = "valid"
NORMALIZATION_GEOMETRIC_ACTIVE = "geometric_active"
_NORMALIZATIONS = frozenset({NORMALIZATION_VALID, NORMALIZATION_GEOMETRIC_ACTIVE})


@dataclass(frozen=True)
class DistillationLossOutput:
    total_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    anchor_mean_loss: torch.Tensor
    unclipped_token_kl: torch.Tensor
    clipped_token_kl: torch.Tensor
    teacher_top_mass: torch.Tensor
    student_top_mass: torch.Tensor
    teacher_raw_top_mass: torch.Tensor
    student_raw_top_mass: torch.Tensor
    valid_token_mask: torch.Tensor
    active_token_mask: torch.Tensor
    clipped_token_mask: torch.Tensor
    negative_token_mask: torch.Tensor
    teacher_entropy: torch.Tensor
    raw_entropy_weight: torch.Tensor
    entropy_weight: torch.Tensor
    weighted_token_kl: torch.Tensor
    response_count: torch.Tensor
    geometric_denominator_sum: torch.Tensor
    normalization_gain_sum: torch.Tensor
    normalization_gain_count: torch.Tensor
    zero_active_response_count: torch.Tensor
    auxiliary_loss_valid_norm: torch.Tensor
    auxiliary_loss_geo_norm: torch.Tensor
    aux_term: torch.Tensor


def chunked_logsumexp_over_vocab(
    logits: torch.Tensor, *, chunk_size: int = 4096, scale: float = 1.0
) -> torch.Tensor:
    if logits.ndim == 0 or logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if not isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    reduced = None
    for start in range(0, logits.shape[-1], chunk_size):
        chunk = logits[..., start : start + chunk_size].to(dtype=work_dtype)
        if scale != 1.0:
            chunk = chunk * float(scale)
        chunk_reduced = torch.logsumexp(chunk, dim=-1)
        reduced = (
            chunk_reduced
            if reduced is None
            else torch.logaddexp(reduced, chunk_reduced)
        )
    return reduced


def require_global_step(meta_info: Mapping[str, Any]) -> int:
    if "global_steps" not in meta_info:
        raise KeyError("RuACT-Teacher requires data.meta_info['global_steps']")
    current_step = meta_info["global_steps"]
    if type(current_step) is not int or current_step < 0:
        raise ValueError("RuACT-Teacher global_steps must be a non-negative integer")
    return current_step


def linear_beta_warmup(*, beta: float, current_step: int, warmup_steps: int) -> float:
    coefficient = float(beta)
    if not isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if type(current_step) is not int or current_step < 0:
        raise ValueError("current_step must be a non-negative integer")
    if type(warmup_steps) is not int or warmup_steps <= 0:
        raise ValueError("warmup_steps must be a positive integer")
    return coefficient * min(current_step / warmup_steps, 1.0)


def finalize_distillation_metrics(metrics: MutableMapping[str, Any]) -> None:
    valid_count = float(metrics.get("distillation/valid_token_count", 0.0))
    negative_count = float(metrics.get("distillation/negative_token_count", 0.0))
    denominator = max(valid_count, 1.0)
    metrics["distillation/fkl_unclipped_mean"] = (
        float(metrics.get("distillation/fkl_unclipped_sum", 0.0)) / denominator
    )
    metrics["distillation/fkl_clipped_mean"] = (
        float(metrics.get("distillation/fkl_clipped_sum", 0.0)) / denominator
    )
    metrics["distillation/teacher_top_mass"] = (
        float(metrics.get("distillation/teacher_top_mass_sum", 0.0)) / denominator
    )
    metrics["distillation/student_top_mass"] = (
        float(metrics.get("distillation/student_top_mass_sum", 0.0)) / denominator
    )
    metrics["distillation/active_token_fraction"] = (
        float(metrics.get("distillation/active_token_count", 0.0)) / denominator
    )
    metrics["distillation/clipped_token_fraction"] = (
        float(metrics.get("distillation/clipped_token_count", 0.0)) / denominator
    )
    metrics["distillation/negative_token_fraction"] = negative_count / denominator
    metrics["distillation/negative_kl_mean_magnitude"] = float(
        metrics.get("distillation/negative_kl_abs_sum", 0.0)
    ) / max(negative_count, 1.0)
    metrics["distillation/negative_to_positive_kl_ratio"] = float(
        metrics.get("distillation/negative_kl_abs_sum", 0.0)
    ) / max(float(metrics.get("distillation/positive_kl_sum", 0.0)), 1e-30)
    metrics["distillation/teacher_entropy_mean"] = (
        float(metrics.get("distillation/teacher_entropy_sum", 0.0)) / denominator
    )
    metrics["distillation/entropy_weight_raw_mean"] = (
        float(metrics.get("distillation/entropy_weight_raw_sum", 0.0)) / denominator
    )
    weight_sum = float(metrics.get("distillation/entropy_weight_sum", 0.0))
    weight_sq_sum = float(metrics.get("distillation/entropy_weight_sq_sum", 0.0))
    metrics["distillation/entropy_weight_mean"] = weight_sum / denominator
    metrics["distillation/entropy_weight_ess_fraction"] = (
        weight_sum * weight_sum / max(valid_count * weight_sq_sum, 1e-30)
    )
    metrics["distillation/entropy_weighted_fkl_mean"] = (
        float(metrics.get("distillation/entropy_weighted_fkl_sum", 0.0)) / denominator
    )
    response_count = max(float(metrics.get("distillation/response_count", 0.0)), 1.0)
    metrics["distillation/valid_token_count_mean"] = valid_count / response_count
    metrics["distillation/active_token_count_mean"] = (
        float(metrics.get("distillation/active_token_count", 0.0)) / response_count
    )
    metrics["distillation/geometric_denominator_mean"] = (
        float(metrics.get("distillation/geometric_denominator_sum", 0.0))
        / response_count
    )
    metrics["distillation/normalization_gain"] = float(
        metrics.get("distillation/normalization_gain_sum", 0.0)
    ) / max(float(metrics.get("distillation/normalization_gain_count", 0.0)), 1.0)
    metrics["distillation/zero_active_response_fraction"] = (
        float(metrics.get("distillation/zero_active_response_count", 0.0))
        / response_count
    )


def compose_distillation_loss(
    *,
    vanilla_policy_loss: torch.Tensor,
    public_top_log_probs: torch.Tensor,
    privileged_top_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    beta: float,
    group_size: int,
    normalization: str = "valid",
    token_kl_clip: float | None = None,
    negative_kl_tolerance: float = 1e-05,
    teacher_entropy: torch.Tensor | None = None,
    entropy_gamma: float = 1.0,
) -> DistillationLossOutput:
    if public_top_log_probs.ndim != 3:
        raise ValueError(
            "public_top_log_probs must have shape [batch, response, top_k]"
        )
    if privileged_top_log_probs.shape != public_top_log_probs.shape:
        raise ValueError("public and privileged top-k log probabilities must align")
    batch_size, response_length, _ = public_top_log_probs.shape
    if response_mask.shape != (batch_size, response_length):
        raise ValueError("response_mask must align with top-k log probabilities")
    if anchor_mask.shape != (batch_size,):
        raise ValueError("anchor_mask must contain one value per response")
    if teacher_entropy is not None and teacher_entropy.shape != (
        batch_size,
        response_length,
    ):
        raise ValueError("teacher_entropy must align with response tokens")
    coefficient = float(beta)
    if not isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            "normalization must be one of " + ", ".join(sorted(_NORMALIZATIONS))
        )
    if token_kl_clip is not None:
        token_kl_clip = float(token_kl_clip)
        if not isfinite(token_kl_clip) or token_kl_clip <= 0.0:
            raise ValueError("token_kl_clip must be finite and positive")
    negative_kl_tolerance = float(negative_kl_tolerance)
    if not isfinite(negative_kl_tolerance) or negative_kl_tolerance < 0.0:
        raise ValueError("negative_kl_tolerance must be finite and non-negative")
    entropy_gamma = float(entropy_gamma)
    if not isfinite(entropy_gamma) or entropy_gamma < 0.0:
        raise ValueError("entropy_gamma must be finite and non-negative")
    student_log_probs = public_top_log_probs.float()
    teacher_log_probs = privileged_top_log_probs.detach().float()
    student_top_probs = student_log_probs.exp()
    teacher_top_probs = teacher_log_probs.exp()
    probability_epsilon = 1e-07
    student_raw_top_mass = student_top_probs.sum(dim=-1)
    teacher_raw_top_mass = teacher_top_probs.sum(dim=-1)
    student_top_mass = student_raw_top_mass.clamp(
        min=0.0, max=1.0 - probability_epsilon
    )
    teacher_top_mass = teacher_raw_top_mass.clamp(
        min=0.0, max=1.0 - probability_epsilon
    )
    student_tail_prob = (1.0 - student_top_mass).clamp_min(probability_epsilon)
    teacher_tail_prob = (1.0 - teacher_top_mass).clamp_min(probability_epsilon)
    top_contributions = teacher_top_probs * (teacher_log_probs - student_log_probs)
    tail_contribution = teacher_tail_prob * (
        teacher_tail_prob.log() - student_tail_prob.log()
    )
    pointwise_kl = torch.cat(
        [top_contributions, tail_contribution.unsqueeze(-1)], dim=-1
    )
    unclipped_token_kl = pointwise_kl.sum(dim=-1)
    valid_token_mask = response_mask.to(torch.bool) & anchor_mask.to(
        torch.bool
    ).unsqueeze(1)
    valid_float = valid_token_mask.to(pointwise_kl.dtype)
    if teacher_entropy is None:
        teacher_entropy_float = torch.zeros_like(unclipped_token_kl)
        raw_entropy_weight = torch.ones_like(unclipped_token_kl)
    else:
        teacher_entropy_float = teacher_entropy.detach().float()
        valid_entropy = teacher_entropy_float.masked_select(valid_token_mask)
        if valid_entropy.numel() and (not torch.isfinite(valid_entropy).all()):
            raise FloatingPointError(
                "RuACT-Teacher teacher entropy is non-finite on valid tokens"
            )
        if valid_entropy.numel() and valid_entropy.min() < -0.0001:
            raise FloatingPointError(
                "RuACT-Teacher teacher entropy is materially negative"
            )
        raw_entropy_weight = torch.exp(
            -entropy_gamma * teacher_entropy_float.clamp_min(0.0)
        )
    response_token_counts = valid_float.sum(dim=-1)
    raw_weight_mean = (raw_entropy_weight * valid_float).sum(
        dim=-1
    ) / response_token_counts.clamp_min(1.0)
    raw_weight_mean = torch.where(
        response_token_counts > 0, raw_weight_mean, torch.ones_like(raw_weight_mean)
    )
    entropy_weight = raw_entropy_weight / raw_weight_mean.detach().unsqueeze(
        -1
    ).clamp_min(1e-12)
    negative_token_mask = valid_token_mask & (unclipped_token_kl < 0.0)
    if negative_token_mask.any():
        minimum_valid_kl = unclipped_token_kl.masked_select(valid_token_mask).min()
        if minimum_valid_kl < -negative_kl_tolerance:
            raise FloatingPointError(
                f"RuACT-Teacher FKL is materially negative: minimum={minimum_valid_kl.detach().item():.9g}, tolerance={negative_kl_tolerance:.9g}; check top-k log-prob and tail normalization"
            )
    if token_kl_clip is None:
        clipped_token_kl = unclipped_token_kl.clamp_min(0.0)
        clipped_token_mask = torch.zeros_like(valid_token_mask)
    else:
        clipped_token_kl = unclipped_token_kl.clamp(min=0.0, max=token_kl_clip)
        clipped_token_mask = valid_token_mask & (unclipped_token_kl > token_kl_clip)
    weighted_token_kl = entropy_weight * clipped_token_kl
    active_token_mask = valid_token_mask & (clipped_token_kl > 1e-08)
    active_token_counts = active_token_mask.to(weighted_token_kl.dtype).sum(dim=-1)
    token_loss_sum = (weighted_token_kl * valid_float).sum(dim=-1)
    response_loss_valid_norm = token_loss_sum / response_token_counts.clamp_min(1.0)
    geometric_token_counts = torch.sqrt(
        response_token_counts.clamp_min(1.0) * active_token_counts.clamp_min(1.0)
    )
    response_loss_geo_norm = torch.where(
        active_token_counts > 0,
        token_loss_sum / geometric_token_counts,
        torch.zeros_like(token_loss_sum),
    )
    if normalization == NORMALIZATION_GEOMETRIC_ACTIVE:
        response_loss = response_loss_geo_norm
    else:
        response_loss = response_loss_valid_norm
    auxiliary_loss = response_loss.sum() / batch_size
    auxiliary_loss_valid_norm = response_loss_valid_norm.sum() / batch_size
    auxiliary_loss_geo_norm = response_loss_geo_norm.sum() / batch_size
    aux_term = coefficient * group_size * auxiliary_loss
    anchor_rows = anchor_mask.to(torch.bool)
    zero_float = torch.zeros_like(response_token_counts)
    response_count = anchor_rows.sum()
    active_response_mask = anchor_rows & (active_token_counts > 0)
    normalization_gain = torch.sqrt(
        response_token_counts.clamp_min(1.0) / active_token_counts.clamp_min(1.0)
    )
    geometric_denominator_sum = torch.where(
        anchor_rows, geometric_token_counts, zero_float
    ).sum()
    normalization_gain_sum = torch.where(
        active_response_mask, normalization_gain, zero_float
    ).sum()
    normalization_gain_count = active_response_mask.sum()
    zero_active_response_count = (anchor_rows & (active_token_counts <= 0)).sum()
    anchor_count = anchor_rows.sum().clamp_min(1)
    anchor_mean_loss = response_loss.sum() / anchor_count
    total_loss = vanilla_policy_loss + aux_term
    return DistillationLossOutput(
        total_loss=total_loss,
        auxiliary_loss=auxiliary_loss,
        anchor_mean_loss=anchor_mean_loss,
        unclipped_token_kl=unclipped_token_kl,
        clipped_token_kl=clipped_token_kl,
        teacher_top_mass=teacher_top_mass,
        student_top_mass=student_top_mass,
        teacher_raw_top_mass=teacher_raw_top_mass,
        student_raw_top_mass=student_raw_top_mass,
        valid_token_mask=valid_token_mask,
        active_token_mask=active_token_mask,
        clipped_token_mask=clipped_token_mask,
        negative_token_mask=negative_token_mask,
        teacher_entropy=teacher_entropy_float,
        raw_entropy_weight=raw_entropy_weight,
        entropy_weight=entropy_weight,
        weighted_token_kl=weighted_token_kl,
        response_count=response_count,
        geometric_denominator_sum=geometric_denominator_sum,
        normalization_gain_sum=normalization_gain_sum,
        normalization_gain_count=normalization_gain_count,
        zero_active_response_count=zero_active_response_count,
        auxiliary_loss_valid_norm=auxiliary_loss_valid_norm,
        auxiliary_loss_geo_norm=auxiliary_loss_geo_norm,
        aux_term=aux_term,
    )
