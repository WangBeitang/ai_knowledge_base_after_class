"""Planner（规划器）model profile（模型配置档案）加载与校验。"""

from __future__ import annotations

from pathlib import Path

from app.rag.query.model_planner.checkpoint_runtime import ModelProfileSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PROFILE_DIR = PROJECT_ROOT / "configs/planner_model_profiles"


def load_model_profile(path_or_profile_id: str | Path) -> ModelProfileSnapshot:
    """
    读取 model profile（模型配置档案）。

    path_or_profile_id（路径或配置档案身份）可以是完整 JSON 路径，也可以是 `qwen3_5_4b`
    这类 profile_id（配置档案身份）。后者从正式 `configs/planner_model_profiles`
    （规划器模型配置档案目录）解析，避免业务运行时依赖 evaluation/stage9（阶段实验目录）。
    """

    profile_path = resolve_model_profile_path(path_or_profile_id)
    return ModelProfileSnapshot.model_validate_json(profile_path.read_text(encoding="utf-8"))


def resolve_model_profile_path(path_or_profile_id: str | Path) -> Path:
    """
    解析 model profile（模型配置档案）路径。

    显式路径按调用方传入为准；profile_id（配置档案身份）按正式配置目录查找。该函数不加载
    模型权重，只负责定位和校验 JSON（结构化数据）配置文件。
    """

    raw_path = Path(path_or_profile_id)
    if raw_path.suffix == ".json" or raw_path.is_absolute() or len(raw_path.parts) > 1:
        profile_path = raw_path
        if not profile_path.is_absolute():
            profile_path = PROJECT_ROOT / profile_path
        return profile_path
    return DEFAULT_MODEL_PROFILE_DIR / f"{raw_path.name}.json"
