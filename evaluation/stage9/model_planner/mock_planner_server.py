"""阶段 9 兼容入口：mock PlannerModelServer（模拟规划器模型服务）已迁到 scripts/。"""

from scripts.planner_model_server.mock_planner_server import (
    MOCK_SERVER_VERSION,
    main,
    make_handler,
    run_server,
)


__all__ = [
    "MOCK_SERVER_VERSION",
    "main",
    "make_handler",
    "run_server",
]


if __name__ == "__main__":
    raise SystemExit(main())
