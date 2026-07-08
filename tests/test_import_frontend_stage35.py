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
    assert "renderDocumentHistory(data.items || [])" in html
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
