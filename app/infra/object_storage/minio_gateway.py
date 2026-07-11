from minio.deleteobjects import DeleteObject

from app.infra.config.providers import infra_config
from app.shared.clients.minio_utils import get_minio_client

class MinioGateway:

    @property
    def bucket_name(self):
        return infra_config.minio.bucket_name

    @property
    def image_dir(self):
        return infra_config.minio.minio_img_dir

    def client(self):
        return get_minio_client()

    def build_image_prefix(self, document_id: str) -> str:
        image_dir = self.image_dir.strip("/")
        if image_dir:
            return f"{image_dir}/{document_id}"
        return document_id

    def build_image_url(self, image_prefix: str, object_name: str):
        protocol = "https" if infra_config.minio.minio_secure else "http"
        return f"{protocol}://{infra_config.minio.endpoint}/{self.bucket_name}/{image_prefix.strip('/')}/{object_name}"

    def delete_image_prefix(self, image_prefix: str) -> int:
        """
        删除一个 document 图片前缀下的全部对象，返回成功提交的对象数量。

        空前缀不能访问 bucket，否则可能误删整桶对象；前缀末尾补 `/`，避免
        删除 doc_1 时把名称以 doc_10 开头的对象一并匹配。
        """
        normalized_prefix = str(image_prefix or "").strip("/")
        if not normalized_prefix:
            return 0

        object_prefix = f"{normalized_prefix}/"
        client = self.client()
        objects = list(
            client.list_objects(
                bucket_name=self.bucket_name,
                prefix=object_prefix,
                recursive=True,
            )
        )
        if not objects:
            return 0

        delete_objects = [DeleteObject(item.object_name) for item in objects]
        errors = list(
            client.remove_objects(
                bucket_name=self.bucket_name,
                delete_object_list=delete_objects,
            )
        )
        if errors:
            error_text = "; ".join(str(error) for error in errors)
            raise RuntimeError(
                f"删除 MinIO 图片前缀失败，prefix={object_prefix}, errors={error_text}"
            )
        return len(delete_objects)

minio_gateway = MinioGateway()

if __name__ == "__main__":
    minio_gateway = MinioGateway()
    print(minio_gateway.build_image_url(minio_gateway.build_image_prefix("test"), "test.png"))
    print(minio_gateway.client())
