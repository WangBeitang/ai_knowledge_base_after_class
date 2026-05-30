from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger,step_log
from pathlib import Path


@step_log("入口识别服务resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口识别服务：
    1. 校验 local_file_path
    2. 识别文件类型（PDF / Markdown）
    3. 回写 is_pdf_read_enabled / is_md_read_enabled
    4. 回写 pdf_path / md_path / file_title
    """
    # 1.获取local_file_path
    local_file_path = state.get("local_file_path")
    # 2.校验
    if not local_file_path:
        logger.error("local_file_path参数为空，无法继续业务！")
        raise ValueError("local_file_path参数为空，无法继续业务！")
    # 3.识别文件类型
    file_path = Path(local_file_path)
    if file_path.suffix == ".md":
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif file_path.suffix == ".pdf":
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(f"f:传入的文件{local_file_path}类型非法，当前系统仅支持markdown和pdf类型")
        return state

    # 4.获取file_title
    state['file_title'] = file_path.stem
    # 5.返回state
    return state