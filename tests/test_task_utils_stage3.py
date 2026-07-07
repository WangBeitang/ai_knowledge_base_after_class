from app.shared.utils import task_utils


def test_registered_import_task_syncs_nodes_and_status(monkeypatch):
    task_id = "task_stage3_registered"
    task_utils.clear_task(task_id)
    node_calls = []
    status_calls = []

    monkeypatch.setattr(
        task_utils,
        "safe_update_task_nodes",
        lambda **kwargs: node_calls.append(kwargs),
    )
    monkeypatch.setattr(
        task_utils,
        "safe_update_task_status",
        lambda task_id, status: status_calls.append((task_id, status)),
    )

    task_utils.register_persistent_task(
        task_id=task_id,
        document_id="doc_1",
        dataset_id="dataset_default_equipment_ops",
    )
    task_utils.add_running_task(task_id, "node_entry")
    task_utils.add_done_task(task_id, "node_entry")
    task_utils.update_task_status(task_id, task_utils.TASK_STATUS_COMPLETED)

    assert node_calls[0] == {
        "task_id": task_id,
        "running_nodes": ["node_entry"],
        "done_nodes": [],
    }
    assert node_calls[1] == {
        "task_id": task_id,
        "running_nodes": [],
        "done_nodes": ["node_entry"],
    }
    assert status_calls == [(task_id, task_utils.TASK_STATUS_COMPLETED)]

    task_utils.clear_task(task_id)


def test_unregistered_query_session_does_not_sync_to_mongo(monkeypatch):
    session_id = "session_stage3_unregistered"
    task_utils.clear_task(session_id)
    node_calls = []
    status_calls = []

    monkeypatch.setattr(
        task_utils,
        "safe_update_task_nodes",
        lambda **kwargs: node_calls.append(kwargs),
    )
    monkeypatch.setattr(
        task_utils,
        "safe_update_task_status",
        lambda task_id, status: status_calls.append((task_id, status)),
    )

    task_utils.add_running_task(session_id, "node_subject_name_confirm")
    task_utils.update_task_status(session_id, task_utils.TASK_STATUS_PROCESSING)

    assert node_calls == []
    assert status_calls == []

    task_utils.clear_task(session_id)
