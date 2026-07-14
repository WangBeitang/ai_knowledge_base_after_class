import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.shared.model import embedding_utils


class ListLikeArray:
    """提供测试所需的切片和 tolist 接口，模拟 NumPy/CSR 数组。"""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ListLikeArray(self.values[item])
        return self.values[item]

    def tolist(self):
        return list(self.values)


class FakeSparseMatrix:
    def __init__(self, row_count):
        self.indices = ListLikeArray([index for index in range(row_count)])
        self.data = ListLikeArray([0.5 for _ in range(row_count)])
        self.indptr = list(range(row_count + 1))


def fake_embedding_result(row_count):
    return {
        "dense": [ListLikeArray([0.1, 0.2]) for _ in range(row_count)],
        "sparse": FakeSparseMatrix(row_count),
    }


def test_concurrent_get_bge_m3_constructs_model_only_once(monkeypatch):
    """四个线程首次访问时，只允许一个线程真正构造模型。"""
    monkeypatch.setattr(embedding_utils, "_bge_m3_ef", None)
    construction_count = 0
    count_lock = threading.Lock()
    initialized_model = object()

    def build_model(**_kwargs):
        nonlocal construction_count
        with count_lock:
            construction_count += 1
        # 给其它线程进入锁外第一次判空的机会，稳定覆盖并发场景。
        time.sleep(0.05)
        return initialized_model

    monkeypatch.setattr(embedding_utils, "BGEM3EmbeddingFunction", build_model)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: embedding_utils.get_bge_m3_ef(), range(4)))

    assert construction_count == 1
    assert all(result is initialized_model for result in results)


def test_failed_initialization_does_not_cache_partial_model(monkeypatch):
    """第一次构造失败后保持 None，下一次调用必须能够重新加载。"""
    monkeypatch.setattr(embedding_utils, "_bge_m3_ef", None)
    initialized_model = object()
    construction_count = 0

    def build_model(**_kwargs):
        nonlocal construction_count
        construction_count += 1
        if construction_count == 1:
            raise RuntimeError("模拟模型权重加载失败")
        return initialized_model

    monkeypatch.setattr(embedding_utils, "BGEM3EmbeddingFunction", build_model)

    with pytest.raises(RuntimeError, match="模拟模型权重加载失败"):
        embedding_utils.get_bge_m3_ef()

    assert embedding_utils._bge_m3_ef is None
    assert embedding_utils.get_bge_m3_ef() is initialized_model
    assert construction_count == 2


def test_generate_embeddings_serializes_access_to_shared_model(monkeypatch):
    """并发请求可以排队，但同一时刻只能有一个线程操作共享模型。"""
    active_encodes = 0
    max_active_encodes = 0
    metrics_lock = threading.Lock()

    class FakeModel:
        def encode_documents(self, texts):
            nonlocal active_encodes, max_active_encodes
            with metrics_lock:
                active_encodes += 1
                max_active_encodes = max(max_active_encodes, active_encodes)
            try:
                time.sleep(0.03)
                return fake_embedding_result(len(texts))
            finally:
                with metrics_lock:
                    active_encodes -= 1

    monkeypatch.setattr(embedding_utils, "_bge_m3_ef", FakeModel())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda index: embedding_utils.generate_embeddings([f"设备主题 {index}"]),
                range(4),
            )
        )

    assert max_active_encodes == 1
    assert all(result == {"dense": [[0.1, 0.2]], "sparse": [{0: 0.5}]} for result in results)
