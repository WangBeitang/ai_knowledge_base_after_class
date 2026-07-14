import zipfile

import pytest

from app.rag.import_ import pdf_parse_service


class FakeResponse:
    def __init__(self, content=b"", json_data=None, status_code=200):
        self.content = content
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data


class FakeUploadSession:
    """模拟上传预签名 URL，只验证代码会完成 PUT，不产生真实网络请求。"""

    trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def put(self, *_args, **_kwargs):
        return FakeResponse(status_code=200)


def configure_failed_mineru_poll(monkeypatch, err_msg):
    """构造“提交和上传成功，但 MinerU 最终解析失败”的完整响应链。"""
    monkeypatch.setattr(pdf_parse_service.infra_config.mineru, "api_key", "test-token")
    monkeypatch.setattr(pdf_parse_service.infra_config.mineru, "base_url", "https://mineru.example")
    monkeypatch.setattr(pdf_parse_service.requests, "Session", FakeUploadSession)
    monkeypatch.setattr(
        pdf_parse_service.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            json_data={
                "code": 0,
                "data": {
                    "file_urls": ["https://upload.example/manual.pdf"],
                    "batch_id": "batch-1",
                },
                "msg": "ok",
            }
        ),
    )
    monkeypatch.setattr(
        pdf_parse_service.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(
            json_data={
                "code": 0,
                "data": {
                    "extract_result": [{
                        "file_name": "manual.pdf",
                        "state": "failed",
                        "err_msg": err_msg,
                    }]
                },
                "msg": "ok",
            }
        ),
    )


def test_download_and_extract_markdown_returns_parse_artifact_paths(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_md = source_dir / "full.md"
    source_md.write_text("# HAK 180\n操作说明", encoding="utf-8")
    zip_source_path = tmp_path / "source.zip"
    with zipfile.ZipFile(zip_source_path, "w") as zip_file:
        zip_file.write(source_md, arcname="full.md")

    monkeypatch.setattr(
        pdf_parse_service.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(zip_source_path.read_bytes()),
    )

    md_path_obj, zip_path_obj, extract_dir_obj = pdf_parse_service.download_and_extract_markdown(
        "https://example.com/result.zip",
        tmp_path,
        "manual",
    )

    assert zip_path_obj == tmp_path / "manual_result.zip"
    assert extract_dir_obj == tmp_path / "manual"
    assert md_path_obj == tmp_path / "manual" / "manual.md"
    assert md_path_obj.read_text(encoding="utf-8") == "# HAK 180\n操作说明"


def test_parse_pdf_to_markdown_writes_parse_artifacts_to_state(monkeypatch, tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    md_path = tmp_path / "manual.md"
    md_path.write_text("# HAK 180\n操作说明", encoding="utf-8")
    zip_path = tmp_path / "manual_result.zip"
    extract_dir = tmp_path / "manual"

    monkeypatch.setattr(
        pdf_parse_service,
        "_upload_pdf_and_poll",
        lambda pdf_path_obj: "https://example.com/result.zip",
    )
    monkeypatch.setattr(
        pdf_parse_service,
        "download_and_extract_markdown",
        lambda zip_url, local_dir_obj, stem: (md_path, zip_path, extract_dir),
    )

    state = {
        "task_id": "task-1",
        "pdf_path": str(pdf_path),
        "local_dir": str(tmp_path),
    }

    result = pdf_parse_service.parse_pdf_to_markdown(state)

    assert result["md_path"] == str(md_path)
    assert result["md_content"] == "# HAK 180\n操作说明"
    assert result["parse_result_zip_path"] == str(zip_path)
    assert result["parse_result_dir"] == str(extract_dir)


def test_failed_mineru_poll_preserves_upstream_error_message(monkeypatch, tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    configure_failed_mineru_poll(monkeypatch, "  文件页数超过限制  ")

    with pytest.raises(RuntimeError, match="MinerU解析失败：文件页数超过限制"):
        pdf_parse_service._upload_pdf_and_poll(pdf_path)


def test_failed_mineru_poll_keeps_generic_message_when_err_msg_is_empty(monkeypatch, tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"fake-pdf")
    configure_failed_mineru_poll(monkeypatch, "   ")

    with pytest.raises(RuntimeError, match="MinerU服务器返回结果异常，解析失败"):
        pdf_parse_service._upload_pdf_and_poll(pdf_path)
