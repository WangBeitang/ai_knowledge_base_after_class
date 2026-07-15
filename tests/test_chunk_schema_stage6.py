import pytest
from pydantic import ValidationError

from app.api.schema.chunk_schema import (
    ChunkEnabledFilter,
    ChunkEventListSchema,
    ChunkListItemSchema,
    ChunkManualStatus,
    ChunkStatusChangeRequest,
    ChunkStatusChangeResponse,
    ChunkStatusEventSchema,
    ChunkStatusEventSource,
    ChunkStatusOperation,
    ChunkStatusReasonType,
)


def _event(**overrides):
    data = {
        "event_id": "event-stage6-1",
        "document_id": "doc-stage6",
        "chunk_id": 1001,
        "dataset_id": "dataset_default_equipment_ops",
        "owner_user_id": "user-a",
        "tenant_id": "tenant_default",
        "visibility": "private",
        "index_version": 3,
        "chunk_index": 12,
        "operator_user_id": "user-a",
        "operation": ChunkStatusOperation.DISABLE,
        "previous_enabled": True,
        "enabled": False,
        "reason_type": ChunkStatusReasonType.HEADER_FOOTER,
        "reason_detail": "重复页脚",
        "source": ChunkStatusEventSource.MANUAL,
        "human_confirmed": True,
        "created_at": "2026-07-15T01:00:00+00:00",
    }
    data.update(overrides)
    return data


def test_enabled_filter_converts_to_three_state_bool():
    assert ChunkEnabledFilter.ALL.to_bool() is None
    assert ChunkEnabledFilter.ENABLED.to_bool() is True
    assert ChunkEnabledFilter.DISABLED.to_bool() is False


def test_chunk_status_event_serializes_enum_values_for_api_json():
    event = ChunkStatusEventSchema(**_event())

    serialized = event.model_dump(mode="json")

    assert serialized["operation"] == "disable"
    assert serialized["reason_type"] == "header_footer"
    assert serialized["source"] == "manual"
    assert serialized["human_confirmed"] is True


def test_chunk_status_event_rejects_direction_that_does_not_match_operation():
    with pytest.raises(ValidationError, match="disable 事件必须从 enabled=true"):
        ChunkStatusEventSchema(**_event(previous_enabled=False, enabled=False))

    with pytest.raises(ValidationError, match="enable 事件必须从 enabled=false"):
        ChunkStatusEventSchema(**_event(
            operation=ChunkStatusOperation.ENABLE,
            previous_enabled=True,
            enabled=True,
            reason_type=ChunkStatusReasonType.MANUAL_RESTORE,
        ))


def test_other_reason_requires_detail_in_event_and_request():
    with pytest.raises(ValidationError, match="reason_type=other 时必须填写 reason_detail"):
        ChunkStatusEventSchema(**_event(
            reason_type=ChunkStatusReasonType.OTHER,
            reason_detail="  ",
        ))

    with pytest.raises(ValidationError, match="reason_type=other 时必须填写 reason_detail"):
        ChunkStatusChangeRequest(
            enabled=False,
            expected_index_version=3,
            reason_type=ChunkStatusReasonType.OTHER,
            reason_detail="",
        )


def test_chunk_list_item_allows_milvus_enabled_true_with_manual_disabled_effective_false():
    item = ChunkListItemSchema(
        chunk_id=1001,
        document_id="doc-stage6",
        dataset_id="dataset_default_equipment_ops",
        index_version=3,
        chunk_index=12,
        enabled=True,
        manual_status=ChunkManualStatus.DISABLED,
        effective_enabled=False,
        title="页脚",
        content_preview="第 23 页 / 共 120 页",
        content_length=18,
        latest_event=ChunkStatusEventSchema(**_event()),
    )

    serialized = item.model_dump(mode="json")

    assert serialized["enabled"] is True
    assert serialized["manual_status"] == "disabled"
    assert serialized["effective_enabled"] is False
    assert serialized["content_preview"] == "第 23 页 / 共 120 页"
    assert serialized["latest_event"]["reason_type"] == "header_footer"


def test_chunk_list_item_rejects_inconsistent_effective_enabled():
    with pytest.raises(ValidationError, match="manual_status=disabled"):
        ChunkListItemSchema(
            chunk_id=1001,
            document_id="doc-stage6",
            dataset_id="dataset_default_equipment_ops",
            index_version=3,
            chunk_index=12,
            enabled=True,
            manual_status=ChunkManualStatus.DISABLED,
            effective_enabled=True,
        )

    with pytest.raises(ValidationError, match="Milvus enabled=false"):
        ChunkListItemSchema(
            chunk_id=1001,
            document_id="doc-stage6",
            dataset_id="dataset_default_equipment_ops",
            index_version=3,
            chunk_index=12,
            enabled=False,
            manual_status=ChunkManualStatus.NONE,
            effective_enabled=True,
        )


def test_change_request_and_response_keep_stage6_contract_fields():
    request = ChunkStatusChangeRequest(
        enabled=False,
        expected_index_version=3,
        reason_type=ChunkStatusReasonType.GARBLED_TEXT,
        reason_detail="OCR 乱码",
        trace_id="trace-stage6",
    )
    event = ChunkStatusEventSchema(**_event(reason_type=request.reason_type, reason_detail=request.reason_detail))
    response = ChunkStatusChangeResponse(
        message="chunk 已禁用",
        changed=True,
        document_id="doc-stage6",
        chunk_id=1001,
        index_version=3,
        enabled=True,
        manual_status=ChunkManualStatus.DISABLED,
        effective_enabled=False,
        latest_event=event,
    )

    assert request.model_dump(mode="json") == {
        "enabled": False,
        "expected_index_version": 3,
        "reason_type": "garbled_text",
        "reason_detail": "OCR 乱码",
        "trace_id": "trace-stage6",
    }
    assert response.model_dump(mode="json")["latest_event"]["reason_type"] == "garbled_text"


def test_chunk_event_list_keeps_document_chunk_and_index_identity():
    event_list = ChunkEventListSchema(
        document_id="doc-stage6",
        chunk_id=1001,
        index_version=3,
        items=[ChunkStatusEventSchema(**_event())],
    )

    serialized = event_list.model_dump(mode="json")

    assert serialized["document_id"] == "doc-stage6"
    assert serialized["chunk_id"] == 1001
    assert serialized["index_version"] == 3
    assert serialized["items"][0]["chunk_index"] == 12
