"""token-level（词元级）GRPO（群组相对策略优化）目标函数。"""

from __future__ import annotations

from typing import Any

import torch


def compute_group_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """把同一 case 的 G 个 Reward（奖励分数）标准化为组内 relative advantage（相对优势）。"""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards 必须是一维且至少包含两条 rollout")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Reward 出现 NaN/Inf")
    std = rewards.std(unbiased=False)
    if float(std.detach().cpu()) < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def completion_token_log_probs(
        model: Any,
        *,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        device: Any,
) -> torch.Tensor:
    """
    计算模型对 completion（生成结果）每个 token 的对数概率。

    返回形状为 [completion_tokens]；第一个 completion token 使用 prompt 最后一个位置的
    causal LM（因果语言模型）logits，保证 old/new/reference 三套概率严格对齐。
    """

    if not prompt_token_ids or not completion_token_ids:
        raise ValueError("prompt_token_ids 和 completion_token_ids 都不能为空")
    input_ids = torch.tensor(
        [prompt_token_ids + completion_token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    # Qwen3.5 支持 logits_to_keep（仅保留指定尾部位置的输出分数）。completion 的第一个
    # token 由 prompt 最后一个位置预测，因此保留 completion 长度再加一个位置；随后丢弃
    # 最末尾那个不参与本次目标函数的位置。这样不再为整段 prompt 构造大词表 logits，
    # 但 old/new/reference log probability（旧/新/参考策略对数概率）的数学口径不变。
    logits_to_keep = len(completion_token_ids) + 1
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=logits_to_keep,
    )
    token_logits = outputs.logits[0, :len(completion_token_ids), :].float()
    targets = torch.tensor(completion_token_ids, dtype=torch.long, device=device)
    log_probs = torch.log_softmax(token_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if log_probs.shape != targets.shape:
        raise RuntimeError("completion token log probability 维度未对齐")
    return log_probs


def grpo_token_objective(
        *,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantage: torch.Tensor | float,
        clip_epsilon: float,
        kl_beta: float,
) -> dict[str, torch.Tensor]:
    """计算单条 rollout 的 clipped policy loss（裁剪策略损失）和 token-level KL penalty。"""

    if not (
        new_log_probs.ndim == old_log_probs.ndim == reference_log_probs.ndim == 1
        and new_log_probs.shape == old_log_probs.shape == reference_log_probs.shape
        and new_log_probs.numel() > 0
    ):
        raise ValueError("new/old/reference log probability 必须是一维、非空且形状一致")
    for name, value in (
        ("new_log_probs", new_log_probs),
        ("old_log_probs", old_log_probs),
        ("reference_log_probs", reference_log_probs),
    ):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} 出现 NaN/Inf")

    advantage_tensor = torch.as_tensor(advantage, dtype=new_log_probs.dtype, device=new_log_probs.device)
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor).mean()

    # 使用 GRPO/PPO 常见的非负 token-level KL 近似：exp(ref-new) - (ref-new) - 1。
    log_ratio_to_reference = reference_log_probs.detach() - new_log_probs
    kl = (torch.exp(log_ratio_to_reference) - log_ratio_to_reference - 1.0).mean()
    total_loss = policy_loss + kl_beta * kl
    clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()
    if not torch.isfinite(torch.stack((policy_loss, kl, total_loss))).all():
        raise FloatingPointError("policy loss/KL/total loss 出现 NaN/Inf")
    return {
        "policy_loss": policy_loss,
        "kl": kl,
        "total_loss": total_loss,
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio.mean(),
    }
