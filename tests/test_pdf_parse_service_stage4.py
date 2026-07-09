import zipfile

from app.rag.import_ import pdf_parse_service


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self.content = content


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
