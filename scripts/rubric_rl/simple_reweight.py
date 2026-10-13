import numpy as np


def rewards(scores, signed_weights, eta=0.5):
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(signed_weights, dtype=np.float64)
    if s.ndim != 2 or w.shape != (s.shape[1],) or (not np.isin(s, [0, 1]).all()):
        raise ValueError("complete binary group scores required")
    if not np.isfinite(w).all() or not np.any(w) or (not 0 <= eta <= 1):
        raise ValueError("invalid weights or eta")
    good = np.where(w < 0, 1 - s, s)
    p = good.mean(axis=0)
    multiplier = 1 + eta * 4 * p * (1 - p)
    effective = np.abs(w) * multiplier
    return (
        good @ (effective / effective.sum()),
        good @ (np.abs(w) / np.abs(w).sum()),
        multiplier,
    )


def apply(batch, tensor, feedback, eta=0.5, group_size=8):
    import torch

    n = tensor.shape[0]
    groups = {}
    for i, uid in enumerate(batch.non_tensor_batch["uid"]):
        groups.setdefault(str(uid), []).append(i)
    output = np.empty(n)
    static = np.empty(n)
    factors = []
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError("incomplete rollout group")
        first = indices[0]
        schema = (
            list(feedback["rubric_ids"][first]),
            list(feedback["rubric_weights"][first]),
        )
        slots = []
        for i in indices:
            if not feedback["judge_valid"][i] and (
                not feedback["judge_zero_fallback"][i]
            ):
                raise ValueError("invalid judge without explicit zero fallback")
            if feedback["score_representation"][i] != "occurrence":
                raise ValueError("wrong verdict polarity")
            if (
                list(feedback["rubric_ids"][i]),
                list(feedback["rubric_weights"][i]),
            ) != schema:
                raise ValueError("misaligned rubric identities")
            slots.append(int(feedback["rollout_index"][i]))
        if sorted(slots) != list(range(group_size)):
            raise ValueError("duplicate/missing rollout")
        valid_indices = [i for i in indices if feedback["judge_valid"][i]]
        output[indices] = 0.0
        static[indices] = 0.0
        if valid_indices:
            r, b, m = rewards(
                [feedback["rubric_scores"][i] for i in valid_indices], schema[1], eta
            )
            output[valid_indices] = r
            static[valid_indices] = b
        else:
            m = np.ones(len(schema[1]))
        factors.extend(m)
    mask = batch.batch["response_mask"].bool()
    if mask.shape != tensor.shape or not mask.any(dim=1).all():
        raise ValueError("bad response mask")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    ends = positions.masked_fill(~mask, -1).max(dim=1).values
    result = torch.zeros_like(tensor)
    result[torch.arange(n, device=tensor.device), ends] = torch.as_tensor(
        output, device=tensor.device, dtype=tensor.dtype
    )
    feedback["reweight_reward"] = output.tolist()
    feedback["reweight_static_reward"] = static.tolist()
    return (
        result,
        {
            "reweight/eta": eta,
            "reweight/judge_zero_count": sum((not v for v in feedback["judge_valid"])),
            "reweight/multiplier_mean": float(np.mean(factors)),
            "reweight/multiplier_max": float(np.max(factors)),
            "reweight/reward_mean": float(output.mean()),
            "reweight/static_reward_mean": float(static.mean()),
            "reweight/absolute_delta_mean": float(np.abs(output - static).mean()),
        },
    )
