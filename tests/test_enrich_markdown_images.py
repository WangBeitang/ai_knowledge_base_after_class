from app.rag.import_.enrich_markdown_images import enrich_markdown_images


def test_enrich_markdown_images_skips_missing_images_dir(tmp_path):
    md_path = tmp_path / "manual.md"
    md_path.write_text("# HAK 180\n开机前检查急停按钮。", encoding="utf-8")

    state = {
        "task_id": "task-1",
        "md_path": str(md_path),
        "md_content": "",
    }

    result = enrich_markdown_images(state)

    assert result["md_path"] == str(md_path)
    assert result["md_content"] == "# HAK 180\n开机前检查急停按钮。"


def test_enrich_markdown_images_skips_empty_images_dir(tmp_path):
    md_path = tmp_path / "manual.md"
    md_path.write_text("# HAK 180\n开机前检查急停按钮。", encoding="utf-8")
    (tmp_path / "images").mkdir()

    state = {
        "task_id": "task-1",
        "md_path": str(md_path),
        "md_content": "",
    }

    result = enrich_markdown_images(state)

    assert result["md_path"] == str(md_path)
    assert result["md_content"] == "# HAK 180\n开机前检查急停按钮。"
