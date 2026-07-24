"""阶段 9 兼容导出层：model profile（模型配置档案）已迁到正式 Planner runtime。"""

from app.rag.query.model_planner.model_profile import (
    DEFAULT_MODEL_PROFILE_DIR,
    load_model_profile,
    resolve_model_profile_path,
)


__all__ = [
    "DEFAULT_MODEL_PROFILE_DIR",
    "load_model_profile",
    "resolve_model_profile_path",
]
