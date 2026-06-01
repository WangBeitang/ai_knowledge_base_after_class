from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client

class LLMProvider:
    # 获取大语言模型
    def chat(self, model_name:str=None, json_mode:bool=False):
        return get_llm_client(model_name, json_mode)

    # 获取视觉模型
    def vision_chat(self, model_name:str=None):
        model_name = model_name or infra_config.vision_model
        return get_llm_client(model_name)