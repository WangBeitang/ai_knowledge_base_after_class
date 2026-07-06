from app.infra.config.providers import infra_config


class LLMProvider:

    # 获取大语言模型
    def chat(self,model_name:str=None,json_mode:bool=False):
        # 模型相关依赖较重，尤其 embedding/reranker 会间接加载 torch/transformers。
        # 放在方法内部懒加载，可以让图 compile、单元测试和服务启动阶段不提前加载模型栈。
        from app.shared.model.lm_utils import get_llm_client

        return get_llm_client(model_name,json_mode)

    # 获取视觉模型
    def vision_chat(self,model_name:str=None):
        from app.shared.model.lm_utils import get_llm_client

        model_name = model_name or infra_config.llm.lv_model
        return get_llm_client(model=model_name)

    # 获取嵌入模型
    def embedding_mode(self):
        # 只有真正需要向量化时才加载 BGE-M3，避免 import 阶段初始化重依赖。
        from app.shared.model.embedding_utils import get_bge_m3_ef

        return get_bge_m3_ef()

    # 为文本列表生成稠密+稀疏混合向量嵌入
    def embed_documents(self,documents:list[str]):
        # 导入、查询召回、别名索引都会走这里；懒加载不会改变调用方行为。
        from app.shared.model.embedding_utils import generate_embeddings

        return generate_embeddings(documents)

    def reranker_model(self):
        # reranker 只在重排序节点执行时需要，避免查询图导入时加载 FlagEmbedding。
        from app.shared.model.reranker_utils import get_reranker_model

        return get_reranker_model()


llm_provider = LLMProvider()
