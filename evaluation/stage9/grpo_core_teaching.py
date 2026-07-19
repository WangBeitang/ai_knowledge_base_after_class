"""
GRPO 核心训练闭环教学版。

这份代码用于解释阶段 9 如何把阶段 8.5 Reward v1 接入训练。它使用有限 Action 路径
策略代替真实本地大模型：每个 case 从若干候选路径中采样一条，然后由
OfflineRagEnvironment 执行，再用 score_trajectory 得到 reward。

真实训练时要替换 FiniteActionPathPolicy：
- sample_group 对应本地 Planner 模型生成多条 Action JSON；
- log_prob 对应模型对已生成 token 的 log probability 求和；
- ref_log_prob 对应冻结参考模型的 log probability。
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from app.rag.evaluation.case_schema import PlannerEvalCase
from app.rag.evaluation.offline_environment import OfflineRagEnvironment, OfflineTrajectoryResult
from app.rag.evaluation.reward import RewardConfig, TrajectoryReward, score_trajectory
from app.rag.query.contracts import QueryAction


DEFAULT_ACTION_PATHS: list[list[QueryAction]] = [
    [QueryAction.LOCAL_SEARCH, QueryAction.ANSWER],
    [QueryAction.LOCAL_SEARCH, QueryAction.HYDE_SEARCH, QueryAction.ANSWER],
    [QueryAction.WEB_SEARCH, QueryAction.ANSWER],
    [QueryAction.LOCAL_SEARCH, QueryAction.ASK_CLARIFICATION],
    [QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
    [QueryAction.ASK_CLARIFICATION],
    [QueryAction.REFUSE],
]


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """
    一条采样轨迹的训练侧记录。

    path_index 是有限路径策略选中的候选路径编号；actions 是真正交给 Environment 执行的
    Action 序列；old_log_prob 是采样瞬间的策略概率，GRPO loss 要用它计算 ratio。
    """

    case_id: str
    path_index: int
    actions: list[QueryAction]
    old_log_prob: torch.Tensor
    ref_log_prob: torch.Tensor


@dataclass(frozen=True, slots=True)
class ScoredRollout:
    """
    一条已经执行并打分的轨迹。

    reward.total_reward 是训练标量；trajectory 和 reward 明细保留下来，是为了后续生成
    评测报告或排查为什么某个样本 advantage 很高/很低。
    """

    sample: RolloutSample
    trajectory: OfflineTrajectoryResult
    reward: TrajectoryReward
    advantage: torch.Tensor


class FiniteActionPathPolicy(nn.Module):
    """
    教学用有限 Action 路径策略。

    它不是最终要训练的本地大模型，只是用一个可微参数矩阵模拟“模型更偏好哪条路径”。
    logits 的形状是 [case_count, path_count]：
    - 每一行对应一个 case；
    - 每一列对应一条候选 Action path；
    - softmax(logits[row]) 得到该 case 下采样各路径的概率。
    """

    def __init__(
            self,
            *,
            case_ids: Sequence[str],
            action_paths: Sequence[Sequence[QueryAction]] | None = None,
    ) -> None:
        super().__init__()
        if not case_ids:
            raise ValueError("case_ids 不能为空")
        self.case_to_row = {case_id: index for index, case_id in enumerate(case_ids)}
        self.action_paths = [list(path) for path in (action_paths or DEFAULT_ACTION_PATHS)]
        if not self.action_paths:
            raise ValueError("action_paths 不能为空")

        # logits 是唯一可训练参数。真实模型里，这里会换成 Transformer 的全部可训练参数。
        self.logits = nn.Parameter(torch.zeros(len(self.case_to_row), len(self.action_paths)))

    def sample_group(self, case_id: str, *, group_size: int) -> list[RolloutSample]:
        """
        为同一个 case 采样一组轨迹。

        group_size 就是 GRPO 的组大小 G。G 越大，组内相对比较越稳定，但一次训练要跑的
        Environment 轨迹也越多。
        """
        if group_size <= 0:
            raise ValueError("group_size 必须大于 0")

        row = self._case_row(case_id)
        distribution = Categorical(logits=self.logits[row])
        sampled_indices = distribution.sample((group_size,))
        old_log_probs = distribution.log_prob(sampled_indices).detach()

        # 教学版用均匀分布作为 reference policy。真实 GRPO 中，reference 通常是冻结的
        # SFT 模型，用来约束新策略不要偏离太快。
        uniform_ref_log_prob = -math.log(len(self.action_paths))
        ref_log_probs = torch.full_like(old_log_probs, fill_value=uniform_ref_log_prob)

        samples: list[RolloutSample] = []
        for index, old_log_prob, ref_log_prob in zip(sampled_indices, old_log_probs, ref_log_probs, strict=True):
            path_index = int(index.item())
            samples.append(RolloutSample(
                case_id=case_id,
                path_index=path_index,
                actions=list(self.action_paths[path_index]),
                old_log_prob=old_log_prob,
                ref_log_prob=ref_log_prob,
            ))
        return samples

    def log_prob(self, case_id: str, path_indices: torch.Tensor) -> torch.Tensor:
        """
        用当前策略重新计算已采样路径的 log probability。

        path_indices 的形状是 [G]。返回值也是 [G]。GRPO 用 new_log_prob 和
        old_log_prob 的差计算 ratio，从而限制一次更新不能过猛。
        """
        row = self._case_row(case_id)
        distribution = Categorical(logits=self.logits[row])
        return distribution.log_prob(path_indices)

    def _case_row(self, case_id: str) -> int:
        try:
            return self.case_to_row[case_id]
        except KeyError as exc:
            raise KeyError(f"未知 case_id：{case_id}") from exc


def compute_group_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    计算 GRPO 的组内 advantage。

    rewards 形状是 [G]。同一个 case 的 G 条轨迹共享同一个问题，因此可以用组内均值
    做 baseline：高于均值为正，低于均值为负。std 很小时返回 0，避免除以接近 0 的数。
    """
    if rewards.ndim != 1:
        raise ValueError("rewards 必须是一维张量，形状为 [group_size]")
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if float(std.item()) < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def grpo_loss(
        *,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.02,
) -> torch.Tensor:
    """
    计算一组轨迹的 GRPO loss。

    四个输入张量形状都必须是 [G]。old_log_probs 和 ref_log_probs 不参与梯度；
    new_log_probs 来自当前策略，需要反向传播。
    """
    if not (
        new_log_probs.shape
        == old_log_probs.shape
        == ref_log_probs.shape
        == advantages.shape
    ):
        raise ValueError("new/old/ref log_probs 和 advantages 的形状必须一致")

    # ratio 表示新策略相对采样时旧策略，对同一条轨迹概率放大了多少。
    ratio = torch.exp(new_log_probs - old_log_probs.detach())

    # clipping 是 GRPO/PPO 类算法的稳定器：advantage 为正时不允许过度放大，advantage
    # 为负时不允许过度压低，避免一次 batch 把策略推偏。
    unclipped_objective = ratio * advantages
    clipped_ratio = torch.clamp(ratio, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    clipped_objective = clipped_ratio * advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

    # 这是常用的非负近似 KL：exp(ref - new) - (ref - new) - 1。
    # 教学版用被采样路径上的 log_prob 估计；真实大模型训练通常在 token 级别计算。
    log_ratio_to_ref = ref_log_probs.detach() - new_log_probs
    approx_kl = (torch.exp(log_ratio_to_ref) - log_ratio_to_ref - 1.0).mean()

    return policy_loss + kl_beta * approx_kl


def run_one_grpo_step(
        *,
        policy: FiniteActionPathPolicy,
        optimizer: torch.optim.Optimizer,
        env: OfflineRagEnvironment,
        cases: Sequence[PlannerEvalCase],
        group_size: int = 4,
        reward_config: RewardConfig | None = None,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.02,
) -> dict[str, float]:
    """
    执行一次完整 GRPO 更新。

    这一步覆盖：
    1. 对每个 case 采样 group_size 条路径；
    2. Environment 执行路径得到 trajectory；
    3. Reward v1 给 trajectory 打分；
    4. 同 case 组内计算 advantage；
    5. 累加 GRPO loss 并反向传播更新 policy。
    """
    if not cases:
        raise ValueError("cases 不能为空")

    reward_config = reward_config or RewardConfig()
    all_losses: list[torch.Tensor] = []
    all_reward_values: list[float] = []

    policy.train()
    optimizer.zero_grad()

    for case in cases:
        samples = policy.sample_group(case.case_id, group_size=group_size)
        rewards: list[float] = []
        trajectories: list[OfflineTrajectoryResult] = []
        reward_details: list[TrajectoryReward] = []

        for sample in samples:
            trajectory = env.run_action_path(
                case,
                sample.actions,
                run_id=f"grpo_{case.case_id}_{uuid.uuid4().hex[:8]}",
                planner_mode="grpo_teaching",
            )
            reward = score_trajectory(case, trajectory, reward_config)
            rewards.append(reward.total_reward)
            trajectories.append(trajectory)
            reward_details.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=policy.logits.device)
        advantages = compute_group_advantages(rewards_tensor)

        path_indices = torch.tensor(
            [sample.path_index for sample in samples],
            dtype=torch.long,
            device=policy.logits.device,
        )
        old_log_probs = torch.stack([sample.old_log_prob for sample in samples]).to(policy.logits.device)
        ref_log_probs = torch.stack([sample.ref_log_prob for sample in samples]).to(policy.logits.device)
        new_log_probs = policy.log_prob(case.case_id, path_indices)

        loss = grpo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            clip_epsilon=clip_epsilon,
            kl_beta=kl_beta,
        )
        all_losses.append(loss)
        all_reward_values.extend(rewards)

    batch_loss = torch.stack(all_losses).mean()
    batch_loss.backward()
    optimizer.step()

    reward_tensor = torch.tensor(all_reward_values, dtype=torch.float32)
    return {
        "loss": float(batch_loss.detach().cpu().item()),
        "mean_reward": float(reward_tensor.mean().item()),
        "min_reward": float(reward_tensor.min().item()),
        "max_reward": float(reward_tensor.max().item()),
    }