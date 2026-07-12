from pymilvus import AnnSearchRequest

from app.infra.config.providers import infra_config
from app.shared.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search


class MilvusGateway:

    @property
    def standard_subject_collection(self):
        return infra_config.milvus.standard_subject_collection

    @property
    def subject_alias_collection(self):
        return infra_config.milvus.subject_alias_collection

    @property
    def chunk_collection_name(self):
        return infra_config.milvus.chunks_collection

    @property
    def client(self):
        return get_milvus_client()

# 新引入
    def create_requests(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
            *,
            expr: str = None,
            limit: int = 5,
    ):
        return create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )

    def hybrid_search(
            self,
            *,
            collection_name: str,
            reqs: list[AnnSearchRequest],
            ranker_weights: tuple[float, float] = (0.5, 0.5),
            norm_score: bool = False,
            limit: int = 5,
            output_fields: list[str] | None = None,
            search_params: dict | None = None,
    ):
        return hybrid_search(
            client=self.client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

    def query_entities(
            self,
            *,
            collection_name: str,
            filter_expr: str,
            output_fields: list[str],
            limit: int = 200,
    ) -> list[dict]:
        """
        在指定权限表达式内读取结构化实体，用于生成合法编号候选。

        ``query`` 与向量 ``hybrid_search`` 的用途不同：这里不按语义相似度选文档，而是
        在已经包含 dataset、subject、enabled 和用户可见性限制的范围内，读取真实存在
        的设备型号、报警码等元数据，形成“允许向当前用户展示的编号词典”。调用方不能
        传入删除了权限条件的表达式，否则可能把无权限编号泄露成纠错候选。
        """
        if not str(filter_expr or "").strip():
            raise ValueError("filter_expr 不能为空，禁止无权限范围读取编号候选")
        if not output_fields:
            raise ValueError("output_fields 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")

        return self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=list(output_fields),
            limit=limit,
        )

milvus_gateway = MilvusGateway()
