from pathlib import Path


IMPORT_HTML = Path("app/resources/http/import.html")


def read_import_html() -> str:
    return IMPORT_HTML.read_text(encoding="utf-8")


def test_import_page_exposes_chunk_panel_entry_from_document_history():
    html = read_import_html()

    assert 'data-action="chunks"' in html
    assert 'class="document-action chunk-action"' in html
    assert 'data-lucide="list"' in html
    assert "const canViewChunks = documentRecord.status === 'completed'" in html
    assert "索引完成并生成 chunk 后才能查看" in html
    assert "createChunkPanelHtml(documentRecord)" in html


def test_import_page_loads_chunks_with_owner_header_and_enabled_filter():
    html = read_import_html()

    assert "async function requestDocumentChunks(documentId, enabledFilter)" in html
    assert "/chunks?enabled=${encodeURIComponent(enabledFilter)}&limit=100" in html
    assert "headers: userHeaders()" in html
    assert "const chunks = Array.isArray(data.items) ? data.items : []" in html
    assert "renderChunkList(panelEl, chunks)" in html


def test_import_page_renders_chunk_status_preview_and_manual_override():
    html = read_import_html()

    assert "chunk.content_preview || '无正文预览'" in html
    assert "chunk?.effective_enabled ? '启用' : '禁用'" in html
    assert "manualStatusLabel(chunk.manual_status)" in html
    assert "chunk.latest_event.reason_type" in html
    assert "data-index-version=\"${escapeHtml(chunk.index_version)}\"" in html


def test_import_page_patches_chunk_enabled_with_required_reason_contract():
    html = read_import_html()

    assert "async function requestChunkStatusChange(documentId, chunkId, payload)" in html
    assert "/chunks/${encodeURIComponent(chunkId)}/enabled" in html
    assert "method: 'PATCH'" in html
    assert "'Content-Type': 'application/json'" in html
    assert "expected_index_version: expectedIndexVersion" in html
    assert "reason_type: nextEnabled ? 'manual_restore' : 'other'" in html
    assert "禁用原因不能为空" in html


def test_import_page_displays_backend_chunk_error_detail():
    html = read_import_html()

    assert "throw await apiErrorFromResponse(response, '查询 chunk 列表失败')" in html
    assert "throw await apiErrorFromResponse(response, '修改 chunk 启停状态失败')" in html
    assert "if (error.detail) return error.detail" in html
    assert "chunk 状态已变化，请刷新后重试" in html
    assert "renderChunkMessage(panelEl, chunkErrorMessage(error), 'error')" in html
