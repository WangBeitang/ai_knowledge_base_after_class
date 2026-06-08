from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    history_repository.save_message(
        session_id=state['session_id'],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get('rewritten_query', ''),
        item_names=state.get("item_names"),
        image_urls=state.get('image_urls'),
    )
    return state