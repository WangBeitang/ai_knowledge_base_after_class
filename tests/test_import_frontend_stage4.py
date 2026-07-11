from pathlib import Path


IMPORT_HTML = Path("app/resources/http/import.html")


def read_import_html() -> str:
    return IMPORT_HTML.read_text(encoding="utf-8")


def test_document_history_exposes_delete_and_rebuild_actions():
    html = read_import_html()

    assert 'data-action="delete"' in html
    assert 'data-action="rebuild"' in html
    assert 'data-lucide="trash-2"' in html
    assert 'data-lucide="refresh-cw"' in html
    assert "['completed', 'failed'].includes(documentRecord.status)" in html
    assert "button.disabled = isBusy || button.dataset.operable !== 'true'" in html


def test_document_actions_send_owner_header_to_expected_endpoints():
    html = read_import_html()

    assert "const suffix = action === 'rebuild' ? '/rebuild' : ''" in html
    assert "`${API_BASE}/documents/${encodeURIComponent(documentId)}${suffix}`" in html
    assert "method: action === 'rebuild' ? 'POST' : 'DELETE'" in html
    assert "headers: userHeaders()" in html
    assert "'X-User-Id': currentUserId" in html


def test_successful_document_action_refreshes_history():
    html = read_import_html()

    assert "await requestDocumentAction(documentId, action)" in html
    assert "showHistoryNotice(action === 'delete' ? '文档已删除' : '已提交重建索引任务')" in html
    assert "await loadDocumentHistory()" in html


def test_document_action_failure_separates_friendly_and_technical_messages():
    html = read_import_html()

    assert "未找到该文档，或你没有操作权限" in html
    assert "当前文档状态不允许此操作，请刷新后重试" in html
    assert "操作失败，请稍后重试" in html
    assert "messageEl.textContent = documentActionErrorMessage(error)" in html
    assert "detailEl.textContent = `最近操作错误：${error.detail" in html


def test_document_actions_have_delete_confirmation_and_mobile_layout():
    html = read_import_html()

    assert "window.confirm" in html
    assert "删除后该文档将不再参与检索" in html
    assert "@media (max-width: 640px)" in html
    assert ".document-actions" in html
