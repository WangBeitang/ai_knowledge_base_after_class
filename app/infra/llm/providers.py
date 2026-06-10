from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client, get_bge_m3_ef, generate_embeddings, get_reranker_model


class LLMProvider:

    # 获取大语言模型
    def chat(self,model_name:str=None,json_mode:bool=False):
        return get_llm_client(model_name,json_mode)

    # 获取视觉模型
    def vision_chat(self,model_name:str=None):
        model_name = model_name or infra_config.llm.lv_model
        return get_llm_client(model=model_name)

    # 获取嵌入模型
    def embedding_mode(self):
        return get_bge_m3_ef()

    # 为文本列表生成稠密+稀疏混合向量嵌入
    def embed_documents(self,documents:list[str]):
        return generate_embeddings(documents)

    def reranker_model(self):
        return get_reranker_model()


llm_provider = LLMProvider()