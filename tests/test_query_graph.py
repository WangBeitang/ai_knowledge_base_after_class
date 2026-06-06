
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state

# 执行 动态测试
state = create_query_default_state(
    session_id = "session_9527",
    original_query="中午吃小当家!",
    is_stream=False
)
result_state = query_graph_app.invoke(state)

# 静态测试 获取跳转结果 图结构
query_graph_app.get_graph().print_ascii()