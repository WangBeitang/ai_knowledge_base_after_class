from app.rag.import_ import enrich_markdown_images as enrich_module
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


def test_enrich_markdown_images_uses_document_id_image_prefix(monkeypatch, tmp_path):
    md_path = tmp_path / "manual.md"
    md_path.write_text("# HAK 180\n![原图](images/diagram.png)", encoding="utf-8")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "diagram.png"
    image_path.write_bytes(b"fake-image")

    class FakeObject:
        def __init__(self, object_name):
            self.object_name = object_name

    class FakeMinioClient:
        def __init__(self):
            self.list_prefix = ""
            self.deleted_objects = []
            self.uploaded_objects = []

        def list_objects(self, *, bucket_name, prefix, recursive):
            self.list_prefix = prefix
            return [FakeObject("kb-images/doc-1/old.png")]

        def remove_objects(self, *, bucket_name, delete_object_list):
            self.deleted_objects = [item.name for item in delete_object_list]
            return []

        def fput_object(self, *, bucket_name, object_name, file_path, content_type):
            self.uploaded_objects.append(
                {
                    "bucket_name": bucket_name,
                    "object_name": object_name,
                    "file_path": file_path,
                    "content_type": content_type,
                }
            )

    class FakeMinioGateway:
        bucket_name = "enterprise-rag"
        image_dir = "/kb-images"

        def __init__(self):
            self.client_instance = FakeMinioClient()
            self.deleted_prefix = ""

        def client(self):
            return self.client_instance

        def build_image_prefix(self, document_id):
            return f"kb-images/{document_id}"

        def build_image_url(self, image_prefix, object_name):
            return f"http://127.0.0.1:9000/enterprise-rag/{image_prefix}/{object_name}"

        def delete_image_prefix(self, image_prefix):
            self.deleted_prefix = image_prefix
            return 1

    fake_gateway = FakeMinioGateway()
    monkeypatch.setattr(enrich_module, "minio_gateway", fake_gateway)
    monkeypatch.setattr(
        enrich_module,
        "summarize_images",
        lambda images_context, stem: {"diagram.png": "设备结构示意图"},
    )

    state = {
        "task_id": "task-1",
        "document_id": "doc-1",
        "md_path": str(md_path),
        "md_content": "",
    }

    result = enrich_markdown_images(state)

    assert result["image_prefix"] == "kb-images/doc-1"
    assert result["md_path"] == str(tmp_path / "manual_new.md")
    assert "![设备结构示意图](http://127.0.0.1:9000/enterprise-rag/kb-images/doc-1/diagram.png)" in result["md_content"]
    assert fake_gateway.deleted_prefix == "kb-images/doc-1"
    assert fake_gateway.client_instance.uploaded_objects[0]["object_name"] == "kb-images/doc-1/diagram.png"
