import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log,logger


@step_log("加载markdown内容")
def load_markdown_content(state):
    """
        思路：
        1.获取三个变量：md_content | md_path | file_title
        2.md_content为空 -> md_path 获取数据 -> 二次判断 -> state[md_content] = md_content
        3.判断file_title为空 -> Path(md_path) -> stem  -> state[file_title] = file_title
        4.md_content内容替换 \n\r -> \n    \r -> \n 清洗数据统一换成符号, 方便2的处理!
        5.return md_content, file_title, md_path_obj
    """

    # 1.获取三个变量：md_content | md_path | file_title
    md_content = state.get("md_content")
    md_path = state.get("md_path")
    file_title = state.get("file_title")

    # 2.md_content为空 -> md_path 获取数据 -> 二次判断 -> state[md_content] = md_content
    if not md_content:
        logger.warning(f"md_content为空，将按md_path读取文件内容：{md_path}")
        if not md_path:
            logger.error("md_content为空,且md_path为空，无法获取md_content，业务无法继续进行！")
            raise ValueError("md_content为空,且无法获取md_content，业务无法继续进行！")
        md_content = Path(md_path).read_text(encoding="utf-8")
        if not md_content:
            logger.error("无法读取文件内容：{md_path}")
            raise ValueError(f"无法读取文件内容：{md_path}")
        # 回写
        state["md_content"] = md_content

    # 3.判断file_title为空 -> Path(md_path) -> stem  -> state[file_title] = file_title
    if not file_title:
        file_title = Path(md_path).stem
        state["file_title"] = file_title
    # 4.md_content内容替换 \n\r -> \n    \r -> \n 清洗数据统一换成符号, 方便2的处理!
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 5.return md_content, file_title, md_path_obj
    return md_content, file_title, Path(md_path)


@step_log("按标题切分文档")
def split_by_titles(md_content, file_title):
    """
        思路：
        1. 定义一个正则
        2. 内容按照 \n 截取,获取所有行 lines
        3. 定义盛放数据的相关容器
        4. 循环处理lines for line in lines :  获取每行数据
        5. 数据清洗 line.strip() 去掉空格
        6. 判断是不是在代码块中 is_code_block
        7. 是不是标题 || 检查是不是在代码块
        8. 最后一个标题进行结算
        9. 整个文档一个标题没有
        10. 返回结果
    """
    # 1. 定义一个正则
    reg = re.compile(r"^\s*#{1,6}\s.+")

    # 2. 内容按照 \n 截取,获取所有行 lines
    lines_list = md_content.split("\n")

    # 3. 定义盛放数据的相关容器
    chunks: list[dict] = []
    current_title = None
    current_lines: list[str] = []
    is_code_block = False
    chunk_size = 0

    # 4. 循环处理lines for line in lines :  获取每行数据
    for line in lines_list:
        # 5. 数据清洗 line.strip() 去掉空格
        line = line.strip()
        if not line:
            continue

        # 6. 判断是不是在代码块中 is_code_block
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_lines.append(line)
            continue

        # 7. 检查是不是在代码块,是不是标题
        if not is_code_block and reg.match(line):
            # 下一个标题行，结算
            if current_title and len(current_lines) > 1:
                chunks.append({
                    "file_title":file_title,
                    "title": current_title,
                    "content": "\n".join(current_lines)
                })
                chunk_size += 1
            current_title = line
            current_lines = [current_title]
        else:
            # 普通行
            current_lines.append(line)
    # 8. 最后一个标题进行结算
    if current_title and len(current_lines) > 1:
        chunks.append({
            "file_title": file_title,
            "title": current_title,
            "content": "\n".join(current_lines)
        })
        chunk_size += 1
    # 9. 整个文档一个标题没有
    if not chunks:
        chunks.append({
            "file_title": file_title,
            "title": "default",
            "content": "\n".join(current_lines)
        })
    # 10. 返回结果
    logger.info(f"{file_title}文档切分完毕，共{chunk_size}个块,切分结果如下：{chunks}")
    return chunks





def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    # 1.读取markdown内容
    md_content, file_title, md_path_obj = load_markdown_content(state)
    # 2.按标题层级做一级粗切
    chunks = split_by_titles(md_content, file_title)
    state["chunks"] = chunks
    return state