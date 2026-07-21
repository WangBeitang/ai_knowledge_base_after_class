"""阶段 8.5 目录所有权契约。

这里集中定义三条流水线的产物位置，避免脚本重新散落 ``candidates/processed/results``
这类按文件格式命名、但无法表达业务阶段的路径。调用脚本可以通过 ``stage85_layout``
传入临时根目录，单元测试和正式运行因此使用完全相同的目录结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STAGE85_ROOT = PROJECT_ROOT / "evaluation/stage8_5"


@dataclass(frozen=True, slots=True)
class Stage85Layout:
    """阶段 8.5 产物目录。

    ``intermediate`` 保存可重放但阶段 9 不直接消费的中间产物；``review`` 保存人工或
    agent 审核证据；``final`` 只保存阶段 9 训练入口直接读取的文件；``reports`` 只面向人。
    """

    root: Path
    public_intermediate: Path
    public_review: Path
    curated_intermediate: Path
    curated_review: Path
    sft_intermediate: Path
    final: Path
    reports: Path


def stage85_layout(root: str | Path = DEFAULT_STAGE85_ROOT) -> Stage85Layout:
    """根据阶段根目录构造稳定布局；函数不创建目录，也不产生文件副作用。"""

    normalized_root = Path(root).resolve()
    artifacts = normalized_root / "artifacts"
    intermediate = artifacts / "intermediate"
    review = artifacts / "review"
    return Stage85Layout(
        root=normalized_root,
        public_intermediate=intermediate / "public_candidate",
        public_review=review / "public_candidate",
        curated_intermediate=intermediate / "curated_gold",
        curated_review=review / "curated_gold",
        sft_intermediate=intermediate / "sft_seed",
        final=artifacts / "final",
        reports=normalized_root / "reports",
    )


__all__ = ["DEFAULT_STAGE85_ROOT", "PROJECT_ROOT", "Stage85Layout", "stage85_layout"]
