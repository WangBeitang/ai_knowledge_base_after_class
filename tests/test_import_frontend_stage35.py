from pathlib import Path


IMPORT_HTML = Path("app/resources/http/import.html")


def read_import_html() -> str:
    return IMPORT_HTML.read_text(encoding="utf-8")


def test_import_page_initializes_stable_user_id_and_sends_header():
    html = read_import_html()

    assert "USER_ID_STORAGE_KEY = 'kb_user_id'" in html
    assert "localStorage.getItem(USER_ID_STORAGE_KEY)" in html
    assert "localStorage.setItem(USER_ID_STORAGE_KEY, userId)" in html
    assert "'X-User-Id': currentUserId" in html


def test_import_page_loads_and_renders_document_history():
    html = read_import_html()

    assert "loadDocumentHistory()" in html
    assert "fetch(`${API_BASE}/documents`" in html
    assert "const documents = Array.isArray(data.items) ? data.items : []" in html
    assert "renderDocumentHistory(documents)" in html
    assert "documentRecord.file_name" in html
    assert "documentRecord.created_at" in html
    assert "documentRecord.parse_status" in html
    assert "documentRecord.index_status" in html
    assert "documentRecord.chunk_count" in html
    assert "查看详情" in html


def test_import_page_maps_failed_nodes_to_friendly_messages():
    html = read_import_html()

    assert "FAILED_NODE_MESSAGES" in html
    assert "node_import_milvus: '文档索引入库失败，请稍后重试'" in html
    assert "node_bge_embedding: '文档向量化失败，请稍后重试'" in html
    assert "导入失败，请稍后重试或联系管理员" in html
    assert "failed_node" in html
    assert "error_message" in html


def test_import_page_auto_refreshes_active_history_with_single_timer():
    """历史列表只能有一个定时器，避免多次上传后叠加并发轮询。"""
    html = read_import_html()

    assert "let historyRefreshTimer = null" in html
    assert "clearTimeout(historyRefreshTimer)" in html
    assert "historyRefreshTimer = setTimeout" in html
    assert "['uploaded', 'processing'].includes(item?.status)" in html
    assert "scheduleHistoryRefresh(hasActiveDocument)" in html
    assert "HISTORY_REFRESH_INTERVAL_MS = 2000" in html
    assert "pollStatus(taskId, itemEl);" in html
    assert "loadDocumentHistory();" in html


def test_import_page_distinguishes_restart_failure_from_connection_loss():
    """浏览器断网只是状态未知；只有后端机器码才能确认服务重启中断。"""
    html = read_import_html()

    assert "ERROR_CODE_MESSAGES" in html
    assert (
        "import_service_restarted: "
        "'服务重启导致导入中断，请重新上传或重建索引'"
    ) in html
    assert "ERROR_CODE_MESSAGES[errorCode]" in html
    assert "服务连接中断，任务状态暂时无法确认，正在重试" in html
    assert "if (!res.ok)" in html
    assert "showConnectionUnknownNotice()" in html
