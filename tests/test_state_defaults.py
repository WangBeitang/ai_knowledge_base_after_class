from app.process.import_.agent.state import create_default_state, get_default_state
from app.process.query.agent.state import create_query_default_state, get_query_default_state


def test_import_default_state_fields_are_complete():
    state = get_default_state()

    assert set(state) == {
        "task_id",
        "dataset_id",
        "document_id",
        "index_version",
        "owner_user_id",
        "tenant_id",
        "visibility",
        "local_file_path",
        "local_dir",
        "is_md_read_enabled",
        "is_pdf_read_enabled",
        "pdf_path",
        "md_path",
        "md_content",
        "parse_result_zip_path",
        "parse_result_dir",
        "image_prefix",
        "file_title",
        "subject_id",
        "standard_subject_name",
        "subject_aliases",
        "equipment_model",
        "alarm_code",
        "part_name",
        "sop_type",
        "safety_level",
        "maintenance_stage",
        "chunks",
        "embedding_content",
    }
    assert state["is_md_read_enabled"] is False
    assert state["is_pdf_read_enabled"] is False
    assert state["dataset_id"] == ""
    assert state["document_id"] == ""
    assert state["index_version"] == 0
    assert state["owner_user_id"] == ""
    assert state["tenant_id"] == ""
    assert state["visibility"] == ""
    assert state["parse_result_zip_path"] == ""
    assert state["parse_result_dir"] == ""
    assert state["image_prefix"] == ""
    assert isinstance(state["subject_aliases"], list)
    assert isinstance(state["chunks"], list)
    assert isinstance(state["embedding_content"], list)


def test_import_default_state_supports_overrides_and_deepcopy():
    state = create_default_state(task_id="task-1")
    state["subject_aliases"].append("HAK180")
    state["chunks"].append({"chunk_id": "chunk-1"})

    fresh_state = get_default_state()

    assert state["task_id"] == "task-1"
    assert fresh_state["task_id"] == ""
    assert fresh_state["subject_aliases"] == []
    assert fresh_state["chunks"] == []


def test_query_default_state_fields_are_complete():
    state = get_query_default_state()

    assert set(state) == {
        "session_id",
        "original_query",
        "is_stream",
        "owner_user_id",
        "tenant_id",
        "dataset_ids",
        "rewritten_query",
        "subject_ids",
        "standard_subject_names",
        "history",
        "embedding_chunks",
        "hyde_embedding_chunks",
        "web_search_docs",
        "rrf_chunks",
        "reranked_docs",
        "prompt",
        "answer",
        "image_urls",
    }
    assert state["is_stream"] is False
    assert state["owner_user_id"] == ""
    assert state["tenant_id"] == ""
    assert isinstance(state["dataset_ids"], list)
    assert isinstance(state["subject_ids"], list)
    assert isinstance(state["standard_subject_names"], list)
    assert isinstance(state["history"], list)
    assert isinstance(state["embedding_chunks"], list)
    assert isinstance(state["hyde_embedding_chunks"], list)
    assert isinstance(state["web_search_docs"], list)
    assert isinstance(state["image_urls"], list)


def test_query_default_state_supports_overrides_and_deepcopy():
    state = create_query_default_state(
        session_id="session-1",
        owner_user_id="user_a",
        tenant_id="tenant_default",
        dataset_ids=["dataset_default_equipment_ops"],
    )
    state["subject_ids"].append("subject_hak_180")
    state["standard_subject_names"].append("HAK 180 烫金机")
    state["dataset_ids"].append("dataset_private")

    fresh_state = get_query_default_state()

    assert state["session_id"] == "session-1"
    assert state["owner_user_id"] == "user_a"
    assert fresh_state["session_id"] == ""
    assert fresh_state["dataset_ids"] == []
    assert fresh_state["subject_ids"] == []
    assert fresh_state["standard_subject_names"] == []
