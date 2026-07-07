import base64
import mimetypes
import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.infra.llm.providers import llm_provider
from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


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
    return md_content, md_path_obj, image_path_obj

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

@step_log("使用视觉模型获取图片说明")
def summarize_images(images_context, stem):
    """
        思路:
        1.获取视觉模型对象
        2.准备结果容器
        3.遍历图片上下文
        4.提示词加载与封装图片用base64字符串进行传值
        5.与视觉模型进行交互
        6.结果保存
    """
    # 1.获取视觉模型对象
    vision_chat = llm_provider.vision_chat()
    # 2.准备结果容器
    images_summary_dict = {}
    # 3.遍历图片上下文
    for image_name, image_path, (pre_context, post_context) in images_context:
        # 加限制
        apply_api_rate_limit()

        # 4.提示词加载与封装图片用base64字符串进行传值
        image_summary_prompt = load_prompt("image_summary",root_folder=stem, image_content=(pre_context, post_context))
        image_path_obj = Path(image_path)
        image_base64 = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        human_message = HumanMessage(
            content=[
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_base64}"
                    }
                },
                {
                    "type":"text",
                    "text":image_summary_prompt
                }
            ]
        )

        # 5.与视觉模型进行交互
        vision_chains = vision_chat | StrOutputParser()
        result = vision_chains.invoke([human_message])

        # 6.结果保存
        images_summary_dict[image_name] = result

    logger.info(f"图片识别结果：{images_summary_dict}")
    return images_summary_dict


@step_log()
def upload_images_and_replace(images_context, images_summary_dict, md_content, stem):
    """
        思路：
        1.删除原文件在minio中存储的图片信息
        2.循环传递每一张图片到minio服务器
        3.存储每张图片对应的minio地址
        4.循环处理每一张图片，替换md_content内容
    """
    # 1.删除原文件在minio中存储的图片信息
    # 1.1 获取要删除的图片列表
    delete_list = minio_gateway.client().list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}",
        recursive=True
    )
    delete_obj_list = [DeleteObject(delete_obj.object_name) for delete_obj in delete_list]
    # 1.2 删除图片
    errors = minio_gateway.client().remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_obj_list
    )
    for error in errors:
        logger.warning(f"删除图片失败：{error}")
    logger.info(f"已完成文件{stem}的图片删除")

    # 2.循环传递每一张图片到minio服务器
    image_minio_url_dict = {}
    for image_name, image_path, (pre_context, post_context) in images_context:
        try:
            minio_gateway.client().fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=f"{minio_gateway.image_dir}/{stem}/{image_name}",
                file_path=image_path,
                content_type=mimetypes.guess_type(image_name)[0]
            )

            # 3.存储每张图片对应的minio地址
            image_minio_url_dict[image_name] = minio_gateway.build_image_url(stem, image_name)
        except Exception as e:
            logger.error(f"上传图片{image_name}失败：{e}")

    # 4.循环处理每一张图片，替换md_content内容
    for image_name,image_url in image_minio_url_dict.items():
        image_summary = images_summary_dict[image_name]
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = reg.sub(lambda _: f"![{image_summary}]({image_url})", md_content)

    return md_content


@step_log()
def back_up_new_md_content(md_content_new, md_path_obj):
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    new_md_path_obj.write_text(md_content_new, encoding="utf-8")
    return str(new_md_path_obj)


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
    state["md_content"] = md_content
    state["md_path"] = str(md_path_obj)

    # 2. 判断 images 目录是否存在、是否有文件。
    # 纯 Markdown 上传、或 MinerU 解析结果不包含图片时，通常不会生成 images 目录。
    # 这类文档仍然应该继续进入切分和入库流程，不能因为缺少图片目录而中断导入。
    if not image_path_obj.exists():
        logger.warning(f"图片目录不存在：{image_path_obj}，跳过图片增强，正常进入下一个节点")
        return state
    if not any(image_path_obj.iterdir()):
        logger.warning(f"图片目录为空：{image_path_obj}，跳过图片增强，正常进入下一个节点")
        return state

    # 3. 获取图片的上下文
    images_context: list[tuple[str, str, tuple[str, str]]] = scan_images(md_content, image_path_obj)
    if not images_context:
        logger.warning(f"图片目录中没有可处理且被 Markdown 引用的图片：{image_path_obj}，跳过图片增强")
        return state

    # 4. 使用视觉模型获取图片说明
    # 格式 {xx.jpg:描述}
    images_summary_dict = summarize_images(images_context, md_path_obj.stem)

    # 5. 上传图片到MinIO，并替换md_content中的图片地址和描述
    md_content_new = upload_images_and_replace(images_context, images_summary_dict, md_content, md_path_obj.stem)

    # 6. 备份新的md_content_new -> md_path_obj  烫金机.md  烫金机_new.md
    new_md_path_str = back_up_new_md_content(md_content_new, md_path_obj)

    # 7. 更新state md_content md_path
    state['md_content'] = md_content_new
    state['md_path'] = new_md_path_str

    # 8. 返回结果
    return state
