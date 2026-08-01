"""真实 Planner GRPO（群组相对策略优化）训练。"""

from app.rag.training.grpo.config import FormalGrpoConfig, load_grpo_config
from app.rag.training.grpo.objective import compute_group_advantages, grpo_token_objective

__all__ = [
    "FormalGrpoConfig",
    "compute_group_advantages",
    "grpo_token_objective",
    "load_grpo_config",
]
