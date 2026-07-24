from app.process.import_.agent.state import create_default_state, get_default_state
from app.process.query.agent.state import copy_query_state, create_query_default_state, get_query_default_state


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
        "query_started_at",
        "rewritten_query",
        "subject_ids",
        "standard_subject_names",
        "subject_resolution_status",
        "subject_candidates",
        "clarification_question",
        "query_identifiers",
        "history",
        "trace_id",
        "planner_step",
        "policy_version",
        "planner_mode",
        "current_planner_decision",
        "planner_action_history",
        "planner_type",
        "planner_runtime_metadata",
        "planner_total_duration_ms",
        "web_search_allowed",
        "safe_guard_triggered",
        "planner_max_steps",
        "current_action_duration_ms",
        "retrieval_observation",
        "retrieval_mode",
        "retrieval_config_version",
        "retrieval_channel_results",
        "embedding_chunks",
        "hyde_embedding_chunks",
        "web_search_docs",
        "rrf_chunks",
        "reranked_docs",
        "citations",
        "terminal_reason_code",
        "answer_runtime_metadata",
        "retrieval_config_snapshot",
        "chunk_status_filter_enabled",
        "disabled_chunk_ids",
        "trace_persistence_enabled",
        "history_persistence_enabled",
        "execution_source",
        "replay_of_trace_id",
        "config_match_status",
        "corpus_match_status",
        "prompt",
        "answer",
        "image_urls",
    }
    assert state["is_stream"] is False
    assert state["owner_user_id"] == ""
    assert state["tenant_id"] == ""
    assert state["query_started_at"] == ""
    assert state["subject_resolution_status"] is None
    assert state["clarification_question"] is None
    assert state["trace_id"] == ""
    assert state["planner_step"] == 0
    assert state["policy_version"] == ""
    assert state["planner_mode"] == ""
    assert state["current_planner_decision"] is None
    assert state["planner_type"] == ""
    assert state["web_search_allowed"] is True
    assert state["safe_guard_triggered"] is False
    assert state["planner_max_steps"] == 6
    assert state["current_action_duration_ms"] == 0
    assert state["retrieval_observation"] is None
    assert state["retrieval_mode"] == "dense_learned_sparse_bm25"
    assert state["retrieval_config_version"] == "retrieval-stage5-final-v1"
    assert state["planner_total_duration_ms"] == 0
    assert state["retrieval_config_snapshot"] == {}
    assert state["chunk_status_filter_enabled"] is False
    assert state["disabled_chunk_ids"] == []
    assert state["trace_persistence_enabled"] is False
    assert state["history_persistence_enabled"] is True
    assert state["execution_source"] == "chat"
    assert state["replay_of_trace_id"] is None
    assert state["config_match_status"] == "unknown"
    assert state["corpus_match_status"] == "unknown"
    assert state["terminal_reason_code"] is None
    assert isinstance(state["dataset_ids"], list)
    assert isinstance(state["subject_ids"], list)
    assert isinstance(state["standard_subject_names"], list)
    assert isinstance(state["subject_candidates"], list)
    assert isinstance(state["query_identifiers"], dict)
    assert isinstance(state["history"], list)
    assert isinstance(state["planner_action_history"], list)
    assert isinstance(state["planner_runtime_metadata"], dict)
    assert isinstance(state["retrieval_channel_results"], dict)
    assert isinstance(state["embedding_chunks"], list)
    assert isinstance(state["hyde_embedding_chunks"], list)
    assert isinstance(state["web_search_docs"], list)
    assert isinstance(state["citations"], list)
    assert isinstance(state["answer_runtime_metadata"], dict)
    assert isinstance(state["disabled_chunk_ids"], list)
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
    state["subject_candidates"].append("HAK 180 Pro 烫金机")
    state["query_identifiers"]["alarm_code"] = ["E021"]
    state["planner_action_history"].append({"step": 1})
    state["planner_runtime_metadata"]["provider"] = "rule"
    state["retrieval_channel_results"]["dense"] = [{"chunk_id": 1}]
    state["disabled_chunk_ids"].append(1001)
    state["citations"].append({"chunk_id": 1})
    state["answer_runtime_metadata"]["model_id"] = "answer-model"

    fresh_state = get_query_default_state()

    assert state["session_id"] == "session-1"
    assert state["owner_user_id"] == "user_a"
    assert fresh_state["session_id"] == ""
    assert fresh_state["dataset_ids"] == []
    assert fresh_state["subject_ids"] == []
    assert fresh_state["standard_subject_names"] == []
    assert fresh_state["subject_candidates"] == []
    assert fresh_state["query_identifiers"] == {}
    assert fresh_state["planner_action_history"] == []
    assert fresh_state["planner_runtime_metadata"] == {}
    assert fresh_state["retrieval_channel_results"] == {}
    assert fresh_state["disabled_chunk_ids"] == []
    assert fresh_state["citations"] == []
    assert fresh_state["answer_runtime_metadata"] == {}


def test_copy_query_state_deepcopies_new_planner_and_observation_containers():
    original = create_query_default_state(
        trace_id="trace-original",
        query_identifiers={"equipment_model": ["HAK 180"]},
        planner_runtime_metadata={"token_usage": {"input": 0}},
        retrieval_channel_results={"dense": [{"chunk_id": 1}]},
        disabled_chunk_ids=[1001],
    )

    copied = copy_query_state(original, trace_id="trace-copy")
    copied["query_identifiers"]["equipment_model"].append("HAK 180 Pro")
    copied["planner_runtime_metadata"]["token_usage"]["input"] = 10
    copied["retrieval_channel_results"]["dense"][0]["chunk_id"] = 2
    copied["disabled_chunk_ids"].append(1002)

    assert original["trace_id"] == "trace-original"
    assert copied["trace_id"] == "trace-copy"
    assert original["query_identifiers"] == {"equipment_model": ["HAK 180"]}
    assert original["planner_runtime_metadata"] == {"token_usage": {"input": 0}}
    assert original["retrieval_channel_results"] == {"dense": [{"chunk_id": 1}]}
    assert original["disabled_chunk_ids"] == [1001]
