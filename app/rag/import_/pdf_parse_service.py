import shutil
import time
import requests
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_INTERVAL_SECONDS, \
    MINERU_POLL_TIMEOUT_SECONDS
from app.rag.import_.mineru_status import is_success_code, parse_extract_state, MinerUExtractState, is_running_state
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from pathlib import Path
from app.infra.config.providers import infra_config
from app.shared.utils.http_status_utils import is_http_ok, is_retryable_server_error


# pdf文件路径校验
def _validate_pdf_paths(state):
    # 1.路径读取
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    # 2.校验pdf_path
    if not pdf_path:
        logger.error("validate_pdf_paths方法，pdf转md，pdf_path参数为空，无法继续业务！")
        raise ValueError("validate_pdf_paths方法，pdf转md，pdf_path参数为空，无法继续业务！")

    # 3.校验local_dir，为空时，使用默认路径
    if not local_dir:
        logger.warning("validate_pdf_paths方法，pdf转md，local_dir参数为空，使用默认路径")
        local_dir = PROJECT_ROOT / "output"

    # 4.转换为Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 5.pdf文件存在校验
    if not pdf_path_obj.exists():
        logger.error(f"pdf_path:{pdf_path}文件不存在，请检查！")
        raise FileNotFoundError(f"pdf_path:{pdf_path}文件不存在，请检查！")

    # 6.local_dir目录存在校验
    if not local_dir_obj.exists():
        logger.warning(f"local_dir:{local_dir}目录不存在，自行创建！")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj

# MinerU交互：pdf上传和zip_url轮询获取
@step_log("MinerU交互")
def _upload_pdf_and_poll(pdf_path_obj):
    # 1.校验MinerU配置
    if not infra_config.mineru.api_key:
        logger.error("MinerU交互，请求参数api_key为空")
        raise ValueError("MinerU交互，请求参数api_key为空")
    if not infra_config.mineru.base_url:
        logger.error("MinerU交互，请求参数base_url为空")
        raise ValueError("MinerU交互，请求参数base_url为空")

    # 2.申请upload_url和batch_id
    token = infra_config.mineru.api_key
    url = f"{infra_config.mineru.base_url}/file-urls/batch"

    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}", "data_id": str(pdf_path_obj.name)}
        ],
        "model_version": "vlm"
    }

    try:
        response = requests.post(url, json=data, headers=header, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
        # 状态码校验
        if not is_http_ok(response.status_code):
            logger.error(f"MinerU交互，请求失败，请检查！状态码：{response.status_code}")
            raise RuntimeError(f"MinerU交互，请求失败，请检查！状态码：{response.status_code}")

        # 业务状态校验
        response_dict = response.json()
        code = response_dict.get("code")
        if not is_success_code(code):
            logger.error(f"MinerU交互，业务状态码为{code},异常信息：{response_dict.get('msg')}")
            raise RuntimeError(f"MinerU交互，业务状态码为{code},异常信息：{response_dict.get('msg')}")

        # 校验完成，获取upload_url和batch_id
        upload_url = response_dict.get("data",{}).get("file_urls")[0]
        batch_id = response_dict.get("data",{}).get("batch_id")
        logger.info(f"MinerU交互，申请upload_url和batch_id成功，upload_url:{upload_url},batch_id:{batch_id}")
    except Exception as e:
        logger.error(f"MinerU交互，申请upload_url和batch_id请求失败，请检查:token:{token},url:{url}")
        raise e

    # 3.pdf文件上传
    try:
        with requests.Session() as session:
            session.trust_env = False
            put_response = session.put(upload_url, data=pdf_path_obj.read_bytes())
            # 状态码校验
            if not is_http_ok(put_response.status_code):
                logger.error(f"MinerU交互，pdf文件{upload_url}上传失败，请检查！状态码：{put_response.status_code}")
                raise RuntimeError(f"MinerU交互，pdf文件{upload_url}上传失败，请检查！状态码：{put_response.status_code}")
    except Exception as e:
        logger.error(f"MinerU交互，pdf文件上传失败，报错信息：{str(e)}，upload_url:{upload_url},batch_id:{batch_id}")
        raise e

    # 4.获取zip_url

    # 参数准备
    get_zip_url = f"{infra_config.mineru.base_url}/extract-results/batch/{batch_id}"
    timeout = MINERU_POLL_TIMEOUT_SECONDS  # 600
    interval_time = MINERU_POLL_INTERVAL_SECONDS  # 3
    start_time = time.time()

    while True:
        # 4.1 超时校验
        if time.time() - start_time > timeout:
            logger.error(f"MinerU交互，获取zip_url超时，请检查！timeout:{timeout}秒")
            raise RuntimeError(f"MinerU交互，获取zip_url超时，请检查！timeout:{timeout}秒")
        # 4.2 发起请求
        try:
            get_response = requests.get(get_zip_url, headers=header)
        except Exception as e:
            logger.error(f"MinerU交互，获取zip_url请求失败，异常信息：{str(e)}，将会在等待后重试")
            time.sleep(interval_time)
            continue

        # 4.3 状态码校验
        if not is_http_ok(get_response.status_code):
            # 重试500-599的状态码
            if is_retryable_server_error(get_response.status_code):
                logger.warning(f"MinerU交互，获取zip_url请求异常，状态码：{get_response.status_code}，将会在等待后重试")
                time.sleep(interval_time)
                continue
            logger.error(f"MinerU交互，获取zip_url请求失败，请检查！状态码：{get_response.status_code}")
            raise RuntimeError(f"MinerU交互，获取zip_url请求失败，请检查！状态码：{get_response.status_code}")

        # 4.4 业务状态校验
        response_dict = get_response.json()
        if not is_success_code(response_dict.get("code")):
            logger.error(f"MinerU交互，获取zip_url业务状态码为{response_dict.get('code')},异常信息：{response_dict.get('msg')}")
            raise RuntimeError(f"MinerU交互，获取zip_url业务状态码为{response_dict.get('code')},异常信息：{response_dict.get('msg')}")

        # 4.5 获取结果信息
        # 完成:done，waiting-file: 等待文件上传排队提交解析任务中，pending: 排队中，running: 正在解析，failed：解析失败，converting：格式转换中
        result_dict = response_dict.get("data", {}).get("extract_result", [])[0]
        try:
            result_state = parse_extract_state(result_dict.get("state"))
        except ValueError as e:
            logger.error(f"轮询获取zip_url,MinerU服务器返回结果异常，{str(e)}")
            raise RuntimeError(f"轮询获取zip_url,MinerU服务器返回结果异常，{str(e)}") from e

        if result_state == MinerUExtractState.DONE:
            zip_url = result_dict.get("full_zip_url")
            if not zip_url:
                logger.error(f"轮询获取zip_url,MinerU服务器返回结果异常，zip_url为空")
                raise RuntimeError(f"轮询获取zip_url,MinerU服务器返回结果异常，zip_url为空")
            return zip_url
        if result_state == MinerUExtractState.FAILED:
            logger.error(f"轮询获取zip_url,MinerU服务器返回结果异常，解析失败")
            raise RuntimeError(f"轮询获取zip_url,MinerU服务器返回结果异常，解析失败")

        if is_running_state(result_state):
            logger.warning(f"轮询获取zip_url,{pdf_path_obj.name}业务结果信息{result_state},正在解析中......")
            time.sleep(interval_time)
            continue

        logger.error(f"轮询获取zip_url,MinerU服务器返回结果异常，无法处理的解析状态：{result_state}")
        raise RuntimeError(f"轮询获取zip_url,MinerU服务器返回结果异常，无法处理的解析状态：{result_state}")

@step_log()
def download_and_extract_markdown(zip_url: str, local_dir_obj: Path, stem: str):
    # 1.下载MinerU返回的zip结果包
    response = requests.get(zip_url, timeout=MINERU_POLL_TIMEOUT_SECONDS)
    if response.status_code != 200:
        logger.error(f"MinerU文件下载地址{zip_url}下载失败，响应状态码：{response.status_code}")
        raise RuntimeError(f"MinerU文件下载地址{zip_url}下载失败，响应状态码：{response.status_code}")

    # 2.将ZIP保存到输出目录
    # 目标存储位置
    zip_path_obj = local_dir_obj / f"{stem}_result.zip"
    zip_path_obj.write_bytes(response.content)

    # 3.清理旧解压目录并重新解压
    extract_dir_obj = local_dir_obj / f"{stem}"
    if extract_dir_obj.is_dir():
        shutil.rmtree(extract_dir_obj)
    # 创建解压文件夹
    extract_dir_obj.mkdir(parents=True, exist_ok=True)
    # 解压
    shutil.unpack_archive(zip_path_obj, extract_dir_obj)

    # 4.递归查找.md文件
    md_file_obj_list = list(extract_dir_obj.rglob("*.md"))
    if not any(md_file_obj_list):
        logger.error(f"MinerU交互，获取zip_url{zip_url}成功，但未找到.md文件，请检查！")
        raise RuntimeError(f"MinerU交互，获取zip_url{zip_url}成功，但未找到.md文件，请检查！")

    # 5.优先选择与pdf同名的.md文件
    for md_file_obj in md_file_obj_list:
        if md_file_obj.stem == stem:
            return md_file_obj, zip_path_obj, extract_dir_obj

    # 6.没有同名文件，取full.md
    target_md_obj = None
    for md_file_obj in md_file_obj_list:
        if md_file_obj.name.lower() == "full.md":
            target_md_obj = md_file_obj
            break
    # 7.没有full.md，取第一个.md文件
    if not target_md_obj:
        target_md_obj = md_file_obj_list[0]

    # 8.重命名为stem.md
    logger.info(f"MinerU交互，获取zip_url{zip_url}成功，已找到.md文件，将重命名为{stem}.md")
    md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    return md_path_obj, zip_path_obj, extract_dir_obj


@step_log()
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 1.pdf文件的路径校验和完善
    pdf_path_obj, local_dir_obj = _validate_pdf_paths(state)
    # 2.pdf上传和zip-url地址获取
    zip_url = _upload_pdf_and_poll(pdf_path_obj)
    # 3.zip文件下载、解压及md文件重命名
    md_path_obj, zip_path_obj, extract_dir_obj = download_and_extract_markdown(zip_url, local_dir_obj, pdf_path_obj.stem)

    # 4.md_path  md_content 回写
    state["md_path"] = str(md_path_obj)
    state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    state["parse_result_zip_path"] = str(zip_path_obj)
    state["parse_result_dir"] = str(extract_dir_obj)
    return state
