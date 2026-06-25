from app.process.import_.agent.nodes import node_bge_embedding as import_embedding_node
from app.process.import_.agent.nodes import node_document_split as import_split_node
from app.process.import_.agent.nodes import node_entry as import_entry_node
from app.process.import_.agent.nodes import node_import_milvus as import_milvus_node
from app.process.import_.agent.nodes import node_md_img as import_md_img_node
from app.process.import_.agent.nodes import node_pdf_to_md as import_pdf_node
from app.process.import_.agent.nodes import node_subject_name_recognition as import_subject_node
from app.process.query.agent.nodes import node_answer_output as query_answer_node
from app.process.query.agent.nodes import node_rerank as query_rerank_node
from app.process.query.agent.nodes import node_rrf as query_rrf_node
from app.process.query.agent.nodes import node_subject_name_confirm as query_subject_node


def test_import_entry_returns_only_route_file_fields():
    result = import_entry_node.node_entry(
        {
            "task_id": "task-1",
            "local_file_path": "demo.pdf",
            "local_dir": "",
            "is_md_read_enabled": False,
            "is_pdf_read_enabled": False,
            "md_path": "",
            "pdf_path": "",
            "file_title": "",
        }
    )

    assert set(result) == {
        "is_md_read_enabled",
        "is_pdf_read_enabled",
        "md_path",
        "pdf_path",
        "file_title",
    }


def test_import_processing_nodes_return_partial_state(monkeypatch):
    monkeypatch.setattr(
        import_pdf_node,
        "parse_pdf_to_markdown",
        lambda state: {**state, "md_path": "demo.md", "md_content": "markdown", "extra": "ignored"},
    )
    monkeypatch.setattr(
        import_md_img_node,
        "enrich_markdown_images",
        lambda state: {**state, "md_path": "demo_new.md", "md_content": "new markdown", "extra": "ignored"},
    )
    monkeypatch.setattr(
        import_split_node,
        "split_document",
        lambda state: {**state, "chunks": [{"chunk_id": "c1"}], "extra": "ignored"},
    )
    monkeypatch.setattr(
        import_subject_node,
        "recognize_and_index_subject_name",
        lambda state: {**state, "subject_name": "HAK 180", "chunks": [{"subject_name": "HAK 180"}], "extra": "ignored"},
    )
    monkeypatch.setattr(
        import_embedding_node,
        "generate_chunk_embeddings",
        lambda state: {**state, "chunks": [{"dense_vector": [0.1]}], "extra": "ignored"},
    )
    monkeypatch.setattr(
        import_milvus_node,
        "index_chunks",
        lambda state: {**state, "chunks": [{"chunk_id": 1}], "extra": "ignored"},
    )

    base_state = {"task_id": "task-1"}

    assert set(import_pdf_node.node_pdf_to_md(base_state)) == {"md_path", "md_content"}
    assert set(import_md_img_node.node_md_img(base_state)) == {"md_path", "md_content"}
    assert set(import_split_node.node_document_split(base_state)) == {"chunks"}
    assert set(import_subject_node.node_subject_name_recognition(base_state)) == {"subject_name", "chunks"}
    assert set(import_embedding_node.node_bge_embedding(base_state)) == {"chunks"}
    assert set(import_milvus_node.node_import_milvus(base_state)) == {"chunks"}


def test_query_nodes_return_partial_state(monkeypatch):
    monkeypatch.setattr(
        query_subject_node,
        "confirm_subject_name",
        lambda state: {
            **state,
            "subject_names": ["HAK 180"],
            "rewritten_query": "HAK 180 怎么操作？",
            "history": [],
            "answer": "",
            "extra": "ignored",
        },
    )
    monkeypatch.setattr(
        query_rrf_node,
        "fuse_by_rrf",
        lambda state: {**state, "rrf_chunks": [{"chunk_id": "c1"}], "extra": "ignored"},
    )
    monkeypatch.setattr(
        query_rerank_node,
        "rerank_documents",
        lambda state: {**state, "reranked_docs": [{"title": "doc"}], "extra": "ignored"},
    )
    monkeypatch.setattr(
        query_answer_node,
        "generate_answer",
        lambda state: {**state, "answer": "ok", "image_urls": ["https://example.com/a.png"], "extra": "ignored"},
    )

    base_state = {"session_id": "session-1", "is_stream": False}

    assert set(query_subject_node.node_subject_name_confirm(base_state)) == {
        "subject_names",
        "rewritten_query",
        "history",
        "answer",
    }
    assert set(query_rrf_node.node_rrf(base_state)) == {"rrf_chunks"}
    assert set(query_rerank_node.node_rerank(base_state)) == {"reranked_docs"}
    assert set(query_answer_node.node_answer_output(base_state)) == {"answer", "image_urls"}
