"""正式 Planner GRPO（群组相对策略优化）命令行入口。"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from app.rag.training.grpo.config import load_grpo_config
from app.rag.training.grpo.trainer import run_formal_grpo_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="从本次正式 run 已生成的 recovery checkpoint（恢复检查点）继续。",
    )
    args = parser.parse_args(argv)
    command_text = " ".join(shlex.quote(item) for item in [sys.executable, "-m", __package__ + ".cli", *sys.argv[1:]])
    result = run_formal_grpo_training(
        load_grpo_config(args.config),
        command_text=command_text,
        resume_from=args.resume_from,
    )
    print(f"run_id={result['run_id']}")
    print(f"run_dir={result['run_dir']}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
