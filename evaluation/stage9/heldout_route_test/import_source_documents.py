"""把 9.3.14 的独立来源文档导入生产 Mongo/Milvus 图。

本入口复用 9.3.13 已验证的生产导入实现，只替换来源清单、冻结清单和输出目录。
它会运行 PDF 解析、生产切分、主题识别、embedding（向量化）和 Milvus 索引，
但不会加载或运行 Planner/SFT checkpoint。
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.stage9.balanced_dev.import_source_documents import run_import


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/configs/heldout_route_test_source_manifest_v1.json"
)
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/heldout_route_test/source_import_manifest.json"
)
DEFAULT_IMPORT_ROOT = PROJECT_ROOT / "output/stage9_heldout_route_test"


def main() -> None:
    """导入全部 heldout 来源并打印不含正文的冻结摘要。"""

    result = run_import(
        source_manifest_path=DEFAULT_SOURCE_MANIFEST,
        output_manifest_path=DEFAULT_OUTPUT_MANIFEST,
        import_root=DEFAULT_IMPORT_ROOT,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(DEFAULT_OUTPUT_MANIFEST.relative_to(PROJECT_ROOT)),
                "document_count": len(result.documents),
                "chunk_count": sum(document.chunk_count for document in result.documents),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
