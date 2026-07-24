"""阶段 9 model profile（模型配置档案）加载与校验。"""

from __future__ import annotations

from pathlib import Path

from app.rag.query.model_planner.checkpoint_runtime import ModelProfileSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PROFILE_DIR = PROJECT_ROOT / "evaluation/stage9/configs/model_profiles"


def load_model_profile(path_or_profile_id: str | Path) -> ModelProfileSnapshot:
    """
    读取 model profile（模型配置档案）。

    path_or_profile_id（路径或配置档案身份）可以是完整 JSON 路径，也可以是 `qwen3_5_4b`
    这类 profile_id（配置档案身份）。后者会从阶段 9 默认 profile 目录解析。
    """

    profile_path = _resolve_profile_path(path_or_profile_id)
    return ModelProfileSnapshot.model_validate_json(profile_path.read_text(encoding="utf-8"))


def _resolve_profile_path(path_or_profile_id: str | Path) -> Path:
    raw_path = Path(path_or_profile_id)
    if raw_path.suffix == ".json" or raw_path.is_absolute() or len(raw_path.parts) > 1:
        profile_path = raw_path
        if not profile_path.is_absolute():
            profile_path = PROJECT_ROOT / profile_path
        return profile_path
    return DEFAULT_MODEL_PROFILE_DIR / f"{raw_path.name}.json"
