import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE, CHUNK_OVERLAP
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


@step_log()
def _split_long_section(chunk, max_size):
    # 1.content格式清理
    title = chunk.get("title")
    content = chunk.get("content")
    body = content[len(title):]

    # 2.定义固定前缀
    prefix = f"{title}\n"
    available_size = max_size - len(prefix)

    # 3.定义递归切割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！"]
    )

    sub_chunks = []

    # 4.递归切割
    for index,chunk_content in enumerate(splitter.split_text(body), start=1):
        sub_chunks.append({
            "file_title": f"{chunk.get('file_title')}",
            "title": f"{chunk.get('title')}_{index}",
            "content": f"{prefix}{chunk_content}",
            "part": index,
            "parent_title": f"{chunk.get('title')}"
        })

    # 5.返回结果
    return sub_chunks


# 子块重编号
@step_log()
def _renumber_chunks(chunks):
    last_patent_title = None
    current_index = 0
    for chunk in chunks:
        parent_title = chunk.get("parent_title")
        if parent_title:
            if parent_title == last_patent_title:
                current_index += 1
            else:
                current_index = 1
                last_patent_title = parent_title
        else:
            parent_title = chunk.get("title")
            chunk["parent_title"] = parent_title
            last_patent_title = None
            current_index = 1

        chunk["part"] = current_index
        chunk["title"] = f"{parent_title}_{current_index}"
    return chunks


def _merge_small_chunks(final_chunks, max_size, min_size):
    # 1.声明合并后的列表结果
    merged_chunks = []

    # 2.记录第一个指针chunk的位置
    start_chunk = None

    # 3.遍历 chunks
    for next_chunk in final_chunks:
        # 第一次
        if start_chunk is None:
            start_chunk = next_chunk
            continue

        # 4.第二次及以后
        is_lt_chunk = len(start_chunk.get("content")) < min_size
        start_parent_title = start_chunk.get("parent_title")
        next_parent_title = next_chunk.get("parent_title")
        is_same_parent_title = start_parent_title and start_parent_title == next_parent_title

        # 5.同一个父标题且长度小于600
        if is_lt_chunk and is_same_parent_title:
            # 6.清理next的标题内容
            next_chunk_to_title = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 2:]
            start_content = start_chunk.get("content")

            # 7.长度校验
            merged_content = start_content + "\n" + next_chunk_to_title
            if len(merged_content) <= max_size:
                start_chunk["content"] = merged_content
                logger.info(f"{start_parent_title}文档合并成功，合并结果如下：{merged_content}")
            else:
                merged_chunks.append(start_chunk)
                start_chunk = next_chunk
                continue
        else:
            merged_chunks.append(start_chunk)
            start_chunk = next_chunk
    # 处理最后一个块
    if start_chunk:
        merged_chunks.append(start_chunk)
    return _renumber_chunks(merged_chunks)

@step_log("对超长文本做二次切分")
def refine_chunks(chunks, file_title, max_size:int = CHUNK_MAX_SIZE, min_size:int = CHUNK_SIZE):
    # 1.判断content有没有超过max_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > max_size:
            final_chunks.extend(_split_long_section(chunk, max_size))
        else:
            final_chunks.append(chunk)

    # 2.判断content有没有小于min_size
    final_merge_chunks = _merge_small_chunks(final_chunks, max_size, min_size)

    # 3.判空赋值
    for chunk in final_merge_chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title")
        if "part" not in chunk:
            chunk["part"] = "1"

    return final_merge_chunks


@step_log("备份chunks")
def backup_chunks(chunks, md_path_obj):
    # 1.获取path对象
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_chunks.json"
    # 2.写入json文件
    json_path_obj.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")


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

    # 3.对超长文本做二次细切
    chunks = refine_chunks(chunks, file_title)

    # 4.备份chunks
    backup_chunks(chunks, md_path_obj)

    # 5.回写state
    state["chunks"] = chunks

    return state