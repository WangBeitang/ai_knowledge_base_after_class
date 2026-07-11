from pathlib import Path


CHAT_HTML = Path("app/resources/http/chat.html")


def read_chat_html() -> str:
    return CHAT_HTML.read_text(encoding="utf-8")


def test_chat_page_reuses_import_user_storage_key():
    html = read_chat_html()

    assert "USER_ID_STORAGE_KEY = 'kb_user_id'" in html
    assert "localStorage.getItem(USER_ID_STORAGE_KEY)" in html
    assert "localStorage.setItem(USER_ID_STORAGE_KEY, userId)" in html
    assert "const currentUserId = getOrCreateUserId()" in html


def test_chat_query_request_sends_current_user_header():
    html = read_chat_html()

    assert "fetch(`${API_BASE}/query`" in html
    assert "headers: queryHeaders()" in html
    assert "'X-User-Id': currentUserId" in html

