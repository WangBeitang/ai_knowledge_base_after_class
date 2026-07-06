from types import SimpleNamespace

from app.infra.vectorstore.milvus_gateway import MilvusGateway
from app.shared.config.milvus_config import MilvusConfig


def test_milvus_config_includes_stage2_subject_collections():
    config = MilvusConfig(
        milvus_url="http://localhost:19530",
        milvus_token="",
        chunks_collection="chunks",
        entity_name_collection="entities",
        subject_name_collection="subjects",
        standard_subject_collection="standard_subjects",
        subject_alias_collection="subject_aliases",
    )

    assert config.standard_subject_collection == "standard_subjects"
    assert config.subject_alias_collection == "subject_aliases"


def test_milvus_gateway_exposes_stage2_collection_names(monkeypatch):
    fake_infra_config = SimpleNamespace(
        milvus=SimpleNamespace(
            subject_name_collection="subjects",
            standard_subject_collection="standard_subjects",
            subject_alias_collection="subject_aliases",
            chunks_collection="chunks",
        )
    )
    monkeypatch.setattr("app.infra.vectorstore.milvus_gateway.infra_config", fake_infra_config)

    gateway = MilvusGateway()

    assert gateway.subject_name_collection == "subjects"
    assert gateway.standard_subject_collection == "standard_subjects"
    assert gateway.subject_alias_collection == "subject_aliases"
    assert gateway.chunk_collection_name == "chunks"
