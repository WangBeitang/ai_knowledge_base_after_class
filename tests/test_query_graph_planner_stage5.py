import copy

import pytest

from app.process.query.agent import main_graph
from app.process.query.agent.nodes import node_query_planner as planner_node
from app.process.query.agent.nodes import node_rerank as rerank_node
from app.process.query.agent.nodes import node_rrf as rrf_node
from app.process.query.agent.nodes import node_search_embedding as local_node
from app.process.query.agent.nodes import node_search_embedding_hyde as hyde_node
from app.process.query.agent.nodes import node_subject_name_confirm as subject_node
from app.process.query.agent.nodes import node_terminal_response as terminal_node
from app.process.query.agent.nodes import node_web_search_mcp as web_node
from app.process.query.agent.state import create_query_default_state
from app.rag.query import subject_name_confirm_service as subject_service
from app.rag.query.contracts import (
    EvidenceSourceType,
    IdentifierResolutionStatus,
    ObservationStatus,
    PlannerDecision,
    PlannerReasonCode,
    QueryAction,
    RetrievalCandidate,
    RetrievalChannel,
    RetrievalObservation,
    SubjectResolutionStatus,
)


def _candidate(chunk_id: str, *, channel: RetrievalChannel, score: float | None = None) -> dict:
    """创建能穿过 RRF、rerank 和 Observation 强校验的本地候选。"""
    return RetrievalCandidate(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        dataset_id="dataset_default_equipment_ops",
        index_version=1,
        chunk_index=0,
        enabled=True,
        title=f"候选 {chunk_id}",
        source_title="HAK 180 操作手册",
        subject_id="subject_hak_180",
        standard_subject_name="HAK 180 烫金机",
        content="开机前检查急停按钮，并确认防护罩已经关闭。",
        source_type=EvidenceSourceType.LOCAL,
        retrieval_channels=[channel],
        retrieval_rank=1,
        retrieval_score=0.5,
        rerank_score=score,
    ).model_dump(mode="json")


def _confirmed_subject_result(state):
    return {
        **state,
        "rewritten_query": "HAK 180 烫金机如何开机？",
        "subject_ids": ["subject_hak_180"],
        "standard_subject_names": ["HAK 180 烫金机"],
        "subject_resolution_status": SubjectResolutionStatus.CONFIRMED,
        "subject_candidates": [],
        "clarification_question": None,
        "history": [],
    }


def _base_state(**overrides):
    state = create_query_default_state(
        session_id="session-planner-graph",
        original_query="HAK 180 烫金机如何开机？",
        owner_user_id="user-a",
        tenant_id="tenant_default",
        dataset_ids=["dataset_default_equipment_ops"],
        trace_id="trace-planner-graph",
    )
    state.update(overrides)
    return state


def _patch_progress_helpers(monkeypatch):
    for module in (subject_node, local_node, hyde_node, web_node, rrf_node, rerank_node, terminal_node):
        if hasattr(module, "add_running_task"):
            monkeypatch.setattr(module, "add_running_task", lambda *args, **kwargs: None)
        if hasattr(module, "add_done_task"):
            monkeypatch.setattr(module, "add_done_task", lambda *args, **kwargs: None)


def _patch_terminal_delivery(monkeypatch):
    monkeypatch.setattr(terminal_node.answer_service, "try_return_existing_answer", lambda state: True)
    monkeypatch.setattr(terminal_node.answer_service, "save_assistant_message", lambda state: None)


def test_graph_uses_planner_loop_and_skips_hyde_web_when_local_evidence_is_sufficient(monkeypatch):
    calls = []
    local_candidate = _candidate("local-1", channel=RetrievalChannel.ORIGINAL)

    _patch_progress_helpers(monkeypatch)
    monkeypatch.setattr(subject_node, "confirm_subject_name", _confirmed_subject_result)
    monkeypatch.setattr(
        local_node,
        "search_by_embedding",
        lambda state: {
            **state,
            "embedding_chunks": [copy.deepcopy(local_candidate)],
            "retrieval_observation": RetrievalObservation(
                action=QueryAction.LOCAL_SEARCH,
                status=ObservationStatus.SUCCESS,
                channel_counts={"dense_learned_sparse": 1},
                candidate_count=1,
            ),
        },
    )
    monkeypatch.setattr(
        hyde_node,
        "search_by_hyde",
        lambda state: (_ for _ in ()).throw(AssertionError("本地证据充分时不应执行 HyDE")),
    )
    monkeypatch.setattr(
        web_node,
        "search_by_web",
        lambda state: (_ for _ in ()).throw(AssertionError("本地证据充分时不应执行 Web")),
    )

    def fake_rrf(state):
        calls.append("rrf")
        return {**state, "rrf_chunks": copy.deepcopy(state.get("embedding_chunks") or [])}

    def fake_rerank(state):
        calls.append("rerank")
        scored = [{**document, "rerank_score": 0.90} for document in state["rrf_chunks"]]
        return {**state, "reranked_docs": scored}

    monkeypatch.setattr(rrf_node, "fuse_by_rrf", fake_rrf)
    monkeypatch.setattr(rerank_node, "rerank_documents", fake_rerank)
    monkeypatch.setattr(
        terminal_node.answer_service,
        "generate_answer",
        lambda state: {**state, "answer": "请先检查急停按钮。", "image_urls": [], "citations": []},
    )

    result = main_graph.query_graph_app.invoke(_base_state())

    assert result["answer"] == "请先检查急停按钮。"
    assert calls == ["rrf", "rerank"]
    assert [item.decision.action for item in result["planner_action_history"]] == [
        QueryAction.LOCAL_SEARCH,
        QueryAction.ANSWER,
    ]
    assert result["planner_step"] == 2
    assert result["policy_version"] == planner_node.rule_based_planner.policy_version
    assert result["terminal_reason_code"] == PlannerReasonCode.LOCAL_EVIDENCE_SUFFICIENT


def test_graph_runs_hyde_only_after_low_local_observation_then_answers(monkeypatch):
    calls = []
    local_candidate = _candidate("local-low", channel=RetrievalChannel.ORIGINAL)
    hyde_candidate = _candidate("hyde-good", channel=RetrievalChannel.HYDE)

    _patch_progress_helpers(monkeypatch)
    monkeypatch.setattr(subject_node, "confirm_subject_name", _confirmed_subject_result)

    def fake_local(state):
        calls.append("local")
        return {
            **state,
            "embedding_chunks": [copy.deepcopy(local_candidate)],
            "retrieval_observation": RetrievalObservation(
                action=QueryAction.LOCAL_SEARCH,
                status=ObservationStatus.SUCCESS,
                channel_counts={"dense_learned_sparse": 1},
                candidate_count=1,
            ),
        }

    def fake_hyde(state):
        calls.append("hyde")
        return {**state, "hyde_embedding_chunks": [copy.deepcopy(hyde_candidate)]}

    def fake_rrf(state):
        candidates = [
            *copy.deepcopy(state.get("embedding_chunks") or []),
            *copy.deepcopy(state.get("hyde_embedding_chunks") or []),
        ]
        return {**state, "rrf_chunks": candidates}

    def fake_rerank(state):
        has_hyde = bool(state.get("hyde_embedding_chunks"))
        score = 0.91 if has_hyde else 0.40
        scored = [{**document, "rerank_score": score} for document in state["rrf_chunks"]]
        return {**state, "reranked_docs": scored}

    monkeypatch.setattr(local_node, "search_by_embedding", fake_local)
    monkeypatch.setattr(hyde_node, "search_by_hyde", fake_hyde)
    monkeypatch.setattr(
        web_node,
        "search_by_web",
        lambda state: (_ for _ in ()).throw(AssertionError("HyDE 证据充分时不应执行 Web")),
    )
    monkeypatch.setattr(rrf_node, "fuse_by_rrf", fake_rrf)
    monkeypatch.setattr(rerank_node, "rerank_documents", fake_rerank)
    monkeypatch.setattr(
        terminal_node.answer_service,
        "generate_answer",
        lambda state: {**state, "answer": "HyDE 补充后可以回答。", "image_urls": [], "citations": []},
    )

    result = main_graph.query_graph_app.invoke(_base_state())

    assert calls == ["local", "hyde"]
    assert [item.decision.action for item in result["planner_action_history"]] == [
        QueryAction.LOCAL_SEARCH,
        QueryAction.HYDE_SEARCH,
        QueryAction.ANSWER,
    ]
    assert result["terminal_reason_code"] == PlannerReasonCode.HYDE_EVIDENCE_SUFFICIENT


def test_ambiguous_subject_goes_directly_to_clarification_without_retrieval(monkeypatch):
    _patch_progress_helpers(monkeypatch)
    _patch_terminal_delivery(monkeypatch)
    monkeypatch.setattr(
        subject_node,
        "confirm_subject_name",
        lambda state: {
            **state,
            "rewritten_query": state["original_query"],
            "subject_ids": [],
            "standard_subject_names": [],
            "subject_resolution_status": SubjectResolutionStatus.AMBIGUOUS,
            "subject_candidates": ["HAK 180", "HAK 180A"],
            "clarification_question": "请确认是 HAK 180 还是 HAK 180A？",
            "history": [],
        },
    )
    for module, attribute in (
        (local_node, "search_by_embedding"),
        (hyde_node, "search_by_hyde"),
        (web_node, "search_by_web"),
    ):
        monkeypatch.setattr(
            module,
            attribute,
            lambda state: (_ for _ in ()).throw(AssertionError("主体歧义时不能执行检索")),
        )

    result = main_graph.query_graph_app.invoke(_base_state())

    assert result["answer"] == "请确认是 HAK 180 还是 HAK 180A？"
    assert result["citations"] == []
    assert [item.decision.action for item in result["planner_action_history"]] == [
        QueryAction.ASK_CLARIFICATION,
    ]
    assert result["terminal_reason_code"] == PlannerReasonCode.SUBJECT_AMBIGUOUS


def test_expected_external_timeout_becomes_failed_observation_but_programming_error_escapes(monkeypatch):
    _patch_progress_helpers(monkeypatch)
    decision = PlannerDecision(
        action=QueryAction.LOCAL_SEARCH,
        query="HAK 180 如何开机？",
        reason_code=PlannerReasonCode.INITIAL_LOCAL_SEARCH,
    )
    state = _base_state(current_planner_decision=decision)

    monkeypatch.setattr(
        local_node,
        "search_by_embedding",
        lambda state: (_ for _ in ()).throw(TimeoutError("milvus timeout")),
    )
    result = local_node.node_search_embedding(state)

    assert result["embedding_chunks"] == []
    assert result["retrieval_observation"].status == ObservationStatus.FAILED
    assert result["retrieval_observation"].error_code == "LOCAL_SEARCH_TIMEOUTERROR"

    monkeypatch.setattr(
        local_node,
        "search_by_embedding",
        lambda state: (_ for _ in ()).throw(ValueError("代码字段错误")),
    )
    with pytest.raises(ValueError, match="代码字段错误"):
        local_node.node_search_embedding(state)


def test_subject_confirmation_writes_structured_status_and_saves_user_message_for_every_branch(monkeypatch):
    saved_messages = []
    monkeypatch.setattr(subject_service, "params_check", lambda state: (state["original_query"], state["session_id"]))
    monkeypatch.setattr(subject_service, "load_history", lambda session_id: [])
    monkeypatch.setattr(
        subject_service.history_repository,
        "save_message",
        lambda **kwargs: saved_messages.append(copy.deepcopy(kwargs)) or "message-id",
    )

    cases = [
        (
            ["HAK180"],
            ([{"subject_id": "subject_hak_180", "standard_subject_name": "HAK 180 烫金机"}], []),
            SubjectResolutionStatus.CONFIRMED,
        ),
        (["180"], ([], ["HAK 180", "HAK 180A"]), SubjectResolutionStatus.AMBIGUOUS),
        (["ZX900"], ([], []), SubjectResolutionStatus.NOT_FOUND),
        ([], ([], []), SubjectResolutionStatus.NO_MENTION),
    ]

    for index, (mentions, classified, expected_status) in enumerate(cases):
        monkeypatch.setattr(
            subject_service,
            "query_rewrite_and_subject_name_recognition",
            lambda original_query, history_text, mentions=mentions: (original_query, mentions),
        )
        monkeypatch.setattr(subject_service, "search_subject_alias_in_milvus", lambda mentions: {"x": []})
        monkeypatch.setattr(
            subject_service,
            "classify_subject_aliases",
            lambda search_result, classified=classified: classified,
        )
        state = {
            "session_id": f"session-{index}",
            "original_query": f"问题-{index}",
            "is_stream": False,
        }

        result = subject_service.confirm_subject_name(state)

        assert result["subject_resolution_status"] == expected_status
        assert "answer" not in result

    assert [item["text"] for item in saved_messages] == [
        "问题-0",
        "问题-1",
        "问题-2",
        "问题-3",
    ]
    assert saved_messages[0]["standard_subject_names"] == ["HAK 180 烫金机"]


def test_router_only_maps_validated_planner_action():
    state = _base_state(
        current_planner_decision=PlannerDecision(
            action=QueryAction.WEB_SEARCH,
            query="查询最新厂家公告",
            reason_code=PlannerReasonCode.REALTIME_QUERY,
        )
    )

    assert main_graph.route_planner_decision(state) == "node_web_search_mcp"
