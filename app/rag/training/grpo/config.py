"""正式 Planner GRPO（群组相对策略优化）训练配置。"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FormalGrpoConfig(BaseModel):
    """
    一轮正式 GRPO（群组相对策略优化）配置。

    该 Schema（结构约束）刻意不提供 max_cases/max_train_samples/max_steps（样本或训练步
    截断）字段；本入口固定读取 75 个 case（训练案例）、每组 4 条 rollout（模型交互轨迹）
    并完成 1 个 epoch（训练轮次），防止正式命令被改回 smoke（冒烟验证）。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    run_name: str = Field(min_length=1, description="正式运行名称，进入 run_id 和全新输出目录。")
    base_model_id: str = Field(min_length=1, description="基础模型身份，必须与 SFT V2 检查点一致。")
    sft_run_id: str = Field(min_length=1, description="初始化 policy/reference 的 SFT V2 运行身份。")
    sft_checkpoint_dir: str = Field(min_length=1, description="SFT V2 checkpoint 根目录。")
    sft_adapter_bytes: int = Field(gt=0, description="SFT V2 adapter 权重字节数，只做身份确认。")

    train_cases: str = Field(min_length=1, description="唯一 GRPO train-only case JSONL 输入。")
    train_case_manifest: str = Field(min_length=1, description="GRPO case manifest（案例清单）。")
    source_sft_train_data: str = Field(min_length=1, description="SFT V2 冻结训练文件，仅核对 SHA256 身份。")
    reward_profile: str = Field(min_length=1, description="冻结 Reward profile（奖励配置）。")
    environment_snapshot: str = Field(min_length=1, description="真实 Provider 执行绑定的环境快照。")
    expected_train_cases_sha256: str
    expected_case_manifest_sha256: str
    expected_source_sft_train_sha256: str
    expected_reward_profile_sha256: str
    expected_environment_snapshot_sha256: str

    output_root: str = Field(min_length=1, description="正式 GRPO 运行父目录；每次自动创建新 run_id。")
    provider_endpoint: str = Field(
        default="http://127.0.0.1:8021",
        description="主业务 Python 中真实 Provider Worker 的本机地址。",
    )
    provider_timeout_seconds: float = Field(default=600.0, gt=0, description="单次真实 Action 最长等待秒数。")

    expected_case_count: int = Field(default=75, description="正式训练必须处理 75 条唯一 case。")
    group_size: int = Field(default=4, description="同一 case 必须采样 4 条 rollout。")
    num_epochs: int = Field(default=1, description="第一轮固定完成 1 个完整 epoch。")
    max_environment_steps: int = Field(default=4, ge=1, le=8, description="单条 rollout 最大 Planner 决策步数。")

    learning_rate: float = Field(default=1e-6, gt=0, description="LoRA（低秩适配）优化学习率。")
    weight_decay: float = Field(default=0.0, ge=0, description="AdamW 权重衰减。")
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1, description="clipped loss（裁剪策略损失）范围。")
    kl_beta: float = Field(default=0.02, ge=0, description="KL penalty（策略偏移惩罚）系数。")
    max_grad_norm: float = Field(default=1.0, gt=0, description="LoRA 梯度范数上限。")
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1, description="学习率 warmup（预热）比例。")
    save_every_optimizer_steps: int = Field(default=25, ge=1, description="周期恢复检查点间隔。")

    max_input_tokens: int = Field(default=3072, ge=128, description="单次 Planner prompt 最大 token 数。")
    max_new_tokens: int = Field(default=128, ge=1, description="单次 PlannerDecision JSON 最大生成 token 数。")
    max_input_chars: int = Field(default=12000, ge=1000, description="Prompt 构造前最大字符数。")
    temperature: float = Field(default=1.0, gt=0, description="四条 rollout 的模型采样温度。")
    torch_dtype: str = Field(default="bfloat16", description="policy/reference 模型加载精度。")
    device: str = Field(default="cuda", description="正式训练设备，必须是 cuda。")
    seed: int = Field(default=20260801, ge=0, description="case 顺序和模型采样随机种子。")
    verify_checkpoint_reload: bool = Field(default=True, description="结束后是否真实重载最终 adapter。")

    @model_validator(mode="after")
    def validate_formal_scope(self) -> "FormalGrpoConfig":
        if self.expected_case_count != 75:
            raise ValueError("正式 GRPO expected_case_count 必须是 75")
        if self.group_size != 4:
            raise ValueError("正式 GRPO group_size 必须是 4")
        if self.num_epochs != 1:
            raise ValueError("第一轮正式 GRPO num_epochs 必须是 1")
        if self.device != "cuda":
            raise ValueError("正式 GRPO device 必须是 cuda")
        if self.max_input_tokens + self.max_new_tokens > 4096:
            raise ValueError("Planner prompt + 输出 token 总数不能超过 Qwen3.5-4B 当前 4096 上下文边界")
        if not self.provider_endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("正式 Provider endpoint 必须是本机地址")
        for field_name in (
            "expected_train_cases_sha256",
            "expected_case_manifest_sha256",
            "expected_source_sft_train_sha256",
            "expected_reward_profile_sha256",
            "expected_environment_snapshot_sha256",
        ):
            if not _SHA256.fullmatch(str(getattr(self, field_name))):
                raise ValueError(f"{field_name} 必须是 SHA256")
        return self


def load_grpo_config(path: str | Path) -> FormalGrpoConfig:
    """读取正式 GRPO config（配置）并执行禁止 smoke 的结构门禁。"""

    return FormalGrpoConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
