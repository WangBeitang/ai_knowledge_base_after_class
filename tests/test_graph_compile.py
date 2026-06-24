
def test_import_graph_compiles():
    from app.process.import_.agent.main_graph import kb_import_app

    graph = kb_import_app.get_graph()

    assert {
        "node_entry",
        "node_pdf_to_md",
        "node_md_img",
        "node_document_split",
        "node_subject_name_recognition",
        "node_bge_embedding",
        "node_import_milvus",
    }.issubset(graph.nodes)


def test_query_graph_compiles():
    from app.process.query.agent.main_graph import query_graph_app

    graph = query_graph_app.get_graph()

    assert {
        "node_subject_name_confirm",
        "node_search_embedding",
        "node_search_embedding_hyde",
        "node_web_search_mcp",
        "node_rrf",
        "node_rerank",
        "node_answer_output",
    }.issubset(graph.nodes)

import os
import subprocess
import sys
from pathlib import Path


def test_query_graph_import_does_not_require_mongo():
    project_root = Path(__file__).resolve().parents[1]

    env = os.environ.copy()
    env["MONGO_URL"] = "mongodb://127.0.0.1:1"
    env["MONGO_DB_NAME"] = "pytest_should_not_connect"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.process.query.agent.main_graph; print('ok')",
        ],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
