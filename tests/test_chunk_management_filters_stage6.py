import pytest

from app.rag.query.chunk_retrieval_utils import (
    build_chunk_access_filter,
    build_chunk_management_filter,
)


def _expected_visibility(user_id="user_a", tenant_id="tenant_default"):
    return (
        '(visibility == "public" '
        f'OR (visibility == "shared" AND tenant_id == "{tenant_id}") '
        f'OR owner_user_id == "{user_id}")'
    )


def test_management_filter_all_enabled_does_not_add_enabled_clause():
    expr = build_chunk_management_filter(
        dataset_ids=["dataset_ops"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        document_id="doc_a",
        index_version=3,
        enabled=None,
    )

    assert expr == (
        'dataset_id in ["dataset_ops"] '
        'AND document_id == "doc_a" '
        "AND index_version == 3 "
        f"AND {_expected_visibility()}"
    )
    assert "enabled == true" not in expr
    assert "enabled == false" not in expr


def test_management_filter_can_select_disabled_chunks():
    expr = build_chunk_management_filter(
        dataset_ids=["dataset_ops"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        document_id="doc_a",
        enabled=False,
    )

    assert expr == (
        'dataset_id in ["dataset_ops"] '
        'AND document_id == "doc_a" '
        "AND enabled == false "
        f"AND {_expected_visibility()}"
    )


def test_management_filter_can_select_enabled_chunks_without_changing_query_filter():
    management_expr = build_chunk_management_filter(
        dataset_ids=["dataset_ops"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        enabled=True,
    )
    query_expr = build_chunk_access_filter(
        dataset_ids=["dataset_ops"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
    )

    assert management_expr == (
        'dataset_id in ["dataset_ops"] '
        "AND enabled == true "
        f"AND {_expected_visibility()}"
    )
    assert query_expr == (
        'dataset_id in ["dataset_ops"] '
        "AND enabled == true "
        f"AND {_expected_visibility()}"
    )


def test_management_filter_owner_branch_cannot_bypass_dataset_or_document():
    expr = build_chunk_management_filter(
        dataset_ids=["dataset_ops"],
        owner_user_id="user_a",
        tenant_id="tenant_default",
        document_id="doc_a",
        enabled=None,
    )

    assert expr.startswith('dataset_id in ["dataset_ops"] AND document_id == "doc_a" AND (')
    assert expr.count("AND (") == 1  # document 条件之后必须整体进入可见性 OR 组。
    assert 'OR (visibility == "shared" AND tenant_id == "tenant_default")' in expr
    assert expr.endswith('OR owner_user_id == "user_a")')
    assert 'doc_a" AND visibility == "public" OR owner_user_id' not in expr


def test_management_filter_escapes_dynamic_string_literals():
    expr = build_chunk_management_filter(
        dataset_ids=['data"set\\ops'],
        owner_user_id='user"a',
        tenant_id="tenant\\default",
        document_id='doc\n"a',
    )

    assert 'dataset_id in ["data\\"set\\\\ops"]' in expr
    assert 'document_id == "doc \\"a"' in expr
    assert 'owner_user_id == "user\\"a"' in expr
    assert 'tenant_id == "tenant\\\\default"' in expr


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dataset_ids": []}, "dataset_ids 不能为空，禁止退化为全库检索"),
        ({"dataset_ids": ["", "  "]}, "dataset_ids 不能为空，禁止退化为全库检索"),
        ({"owner_user_id": ""}, "owner_user_id 不能为空"),
        ({"tenant_id": "  "}, "tenant_id 不能为空"),
        ({"document_id": ""}, "document_id 不能为空"),
        ({"document_id": "   "}, "document_id 不能为空"),
        ({"index_version": -1}, "index_version 必须是大于等于 0 的整数"),
        ({"index_version": True}, "index_version 必须是大于等于 0 的整数"),
        ({"enabled": "false"}, "enabled 必须是 bool 或 None"),
    ],
)
def test_management_filter_rejects_invalid_scope_or_filter_values(kwargs, message):
    base_kwargs = {
        "dataset_ids": ["dataset_ops"],
        "owner_user_id": "user_a",
        "tenant_id": "tenant_default",
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_chunk_management_filter(**base_kwargs)


def test_management_filter_rejects_single_dataset_string():
    with pytest.raises(ValueError, match="dataset_ids 必须是字符串列表"):
        build_chunk_management_filter(
            dataset_ids="dataset_ops",
            owner_user_id="user_a",
            tenant_id="tenant_default",
        )
