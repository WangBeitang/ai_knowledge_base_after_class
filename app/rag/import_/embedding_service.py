from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger,step_log

@step_log()
def check_chunks(state):
    chunks = state.get("chunks")
    if not chunks:
        logger.error("chunks为空，无法进行向量化！")
        raise ValueError("chunks为空，无法进行向量化！")
    return chunks


@step_log()
def generate_embeddings(chunks,step:int=EMBEDDING_BATCH_SIZE):
    pre_first = chunks[0].copy()

    for i in range(0,len(chunks),step):
        # 当前批次
        chunk_batch = chunks[i:i+step]

        # 组装生成向量的字符串列表
        batch_embedding_list = [
            f"主体名称：{chunk.get('item_name')},内容：{chunk.get('content')}"
            for chunk in chunk_batch
        ]

        # 生成向量
        batch_embedding_dict = llm_provider.embed_documents(batch_embedding_list)
        # 将向量结果补充回chunk
        for j,chunk in enumerate(chunk_batch):
            chunk["dense_vector"] = batch_embedding_dict["dense"][j]
            chunk["sparse_vector"] = batch_embedding_dict["sparse"][j]

    logger.info(f"已完成chunks向量化，原始数据：{pre_first},向量化结果：{chunks[0]}")
    return chunks




@step_log()
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    # 1.参数校验
    chunks = check_chunks(state)

    # 2.生成向量
    chunks = generate_embeddings(chunks)

    # 3.将向量结果补充回chunks
    state["chunks"] = chunks

    return state