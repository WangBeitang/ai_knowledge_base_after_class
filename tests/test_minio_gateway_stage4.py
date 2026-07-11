import pytest

from app.infra.object_storage.minio_gateway import MinioGateway


class FakeObject:
    def __init__(self, object_name):
        self.object_name = object_name


class FakeMinioClient:
    def __init__(self, *, errors=None):
        self.errors = errors or []
        self.list_calls = []
        self.deleted_names = []

    def list_objects(self, *, bucket_name, prefix, recursive):
        self.list_calls.append((bucket_name, prefix, recursive))
        return [FakeObject("kb-images/doc_1/a.png"), FakeObject("kb-images/doc_1/b.png")]

    def remove_objects(self, *, bucket_name, delete_object_list):
        self.deleted_names = [item.name for item in delete_object_list]
        return iter(self.errors)


class FakeMinioGateway(MinioGateway):
    def __init__(self, client):
        self.client_instance = client

    @property
    def bucket_name(self):
        return "enterprise-rag"

    def client(self):
        return self.client_instance


def test_delete_image_prefix_uses_exact_document_prefix():
    client = FakeMinioClient()
    gateway = FakeMinioGateway(client)

    deleted_count = gateway.delete_image_prefix("/kb-images/doc_1/")

    assert deleted_count == 2
    assert client.list_calls == [("enterprise-rag", "kb-images/doc_1/", True)]
    assert client.deleted_names == ["kb-images/doc_1/a.png", "kb-images/doc_1/b.png"]


def test_delete_image_prefix_skips_empty_prefix():
    client = FakeMinioClient()
    gateway = FakeMinioGateway(client)

    assert gateway.delete_image_prefix("") == 0
    assert client.list_calls == []


def test_delete_image_prefix_raises_when_minio_reports_error():
    client = FakeMinioClient(errors=["delete failed"])
    gateway = FakeMinioGateway(client)

    with pytest.raises(RuntimeError, match="删除 MinIO 图片前缀失败"):
        gateway.delete_image_prefix("kb-images/doc_1")
