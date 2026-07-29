"""抓取并冻结 balanced dev 的官方 Web 证据。

输出不保存整页 HTML，只记录原始响应 SHA256、抽取文本 SHA256、规范化事实 SHA256
和已在页面中逐条命中的短语。这样既能证明审核时看到的页面版本，又不会把动态网页
整页复制进仓库。运行时 Reward 仍只按规范化 URL 匹配，不声称重新验证了页面 hash。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.query.rrf_service import canonicalize_web_url  # noqa: E402


FREEZE_VERSION = "stage9-balanced-dev-web-evidence-freeze-v1"
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "evaluation/stage9/configs/balanced_dev_web_source_manifest_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation/stage9/artifacts/balanced_dev/web_evidence_manifest.json"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalized_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\s+", "", " ".join(soup.stripped_strings))


def freeze_web_evidence(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    timeout_seconds: float = 30.0,
    freeze_version: str = FREEZE_VERSION,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    captured_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    frozen_sources = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; Stage9EvidenceFreeze/1.0; "
                    "+local-evaluation)"
                )
            }
        )
        for source in source_manifest["sources"]:
            response = session.get(
                source["url"],
                timeout=timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            response_bytes = response.content
            page_text = _normalized_text(response_bytes)
            frozen_facts = []
            for fact in source["facts"]:
                missing = [
                    phrase
                    for phrase in fact["required_phrases"]
                    if re.sub(r"\s+", "", phrase) not in page_text
                ]
                if missing:
                    raise ValueError(
                        f"官方页面缺少冻结短语：source_id={source['source_id']}, "
                        f"fact_id={fact['fact_id']}, missing={missing}"
                    )
                frozen_facts.append(
                    {
                        "fact_id": fact["fact_id"],
                        "statement": fact["statement"],
                        "verified_phrases": fact["required_phrases"],
                    }
                )

            frozen_sources.append(
                {
                    "source_id": source["source_id"],
                    "publisher": source["publisher"],
                    "source_title": source["source_title"],
                    "url": source["url"],
                    "canonical_url": canonicalize_web_url(source["url"]),
                    "resolved_url": response.url,
                    "captured_at": captured_at,
                    "http_status": response.status_code,
                    "response_bytes": len(response_bytes),
                    "response_sha256": _sha256_bytes(response_bytes),
                    "extracted_text_sha256": _sha256_bytes(
                        page_text.encode("utf-8")
                    ),
                    "evidence_content_sha256": _sha256_bytes(
                        _canonical_json(frozen_facts)
                    ),
                    "facts": frozen_facts,
                }
            )

    output = {
        "freeze_version": freeze_version,
        "source_manifest_version": source_manifest["manifest_version"],
        "source_manifest_path": str(
            source_manifest_path.resolve().relative_to(PROJECT_ROOT)
        ),
        "source_manifest_sha256": _sha256_bytes(
            source_manifest_path.read_bytes()
        ),
        "captured_at": captured_at,
        "source_count": len(frozen_sources),
        "sources": frozen_sources,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_web_evidence(
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "source_count": result["source_count"],
                "captured_at": result["captured_at"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
