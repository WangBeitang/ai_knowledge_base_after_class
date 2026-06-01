import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log

@step_log("加载Markdown和图片目录")
def load_markdown_and_image_dir(state):
    # 1. 读取 `md_content` 和 `md_path`
    md_path, md_content = state.get("md_path"), state.get("md_content")

    # 2. 校验 `md_path` 是否为空
    if not md_path:
        logger.error("md_path参数为空，请检查！")
        raise ValueError("md_path参数为空，请检查！")


    # 3. 如果 `md_content` 为空，则按 `md_path` 读取文件正文
    md_path_obj = Path(md_path)
    if not md_content:
        logger.info(f"md_content为空，将按md_path:{md_path}读取正文")
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error("md_content解析失败，请检查！")
            raise ValueError("md_content解析失败，请检查！")

    # 4. 拼接图片目录 `images`
    image_path_obj = md_path_obj.parent / "images"
    # 5. 返回正文、Markdown 路径和图片目录路径
    return md_content, md_path, image_path_obj

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

@step_log("扫描Markdown中的图片,获取图片的上下文")
def scan_images(md_content:str,image_path_obj:Path,context_length:int=100):
    images_context = []
    # 1. 从image_path_obj中获取每一个文件
    for image_file_obj in image_path_obj.iterdir():
        image_name = image_file_obj.name
        # 2. 遍历循环 -> 文件判断 -> 是不是图片
        if image_file_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warning(f"{image_file_obj}不是图片文件，请检查！")
            continue

        # 3. 定义这张图片专属的正则规则
        reg = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

        # 4. 使用正则在md_content中进行匹配 search 有 只有一个 或者没有
        match = reg.search(md_content)

        # 5. 没有 -> 该图片没有被md_content引用不用识别上下文!
        if not match:
            logger.warning(f"{image_file_obj}没有被md_content引用，请检查！")
            continue

        # 6. 有 -> 获取start | end 截取上下文
        start, end = match.span()
        pre_context = md_content[max(0, start - context_length):start]
        post_context = md_content[end:min(len(md_content), end + context_length)]

        # 7. 填装数据
        images_context.append((image_name, str(image_file_obj), (pre_context, post_context)))


    # 8. 返回即可
    logger.info(f"图片识别结果：{images_context}")
    return images_context


@step_log("Markdown 图片增强服务")
def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1. 获取操作参数
    md_content, md_path_obj, image_path_obj = load_markdown_and_image_dir(state)
    # 2. 判断image_path_obj是否存在内容
    if not any(image_path_obj.iterdir()):
        logger.warning(f"当前{md_content}没有图片,无需图片处理!正常进入下一个节点!!")
        return state
    # 3. 获取图片的上下文
    images_context: list[tuple[str, str, tuple[str, str]]] = scan_images(md_content, image_path_obj)
    print("图片识别结果：", images_context)
    return  state