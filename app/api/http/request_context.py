"""
HTTP 请求上下文的轻量解析工具。

阶段 3.5 的导入接口和阶段 5 的查询接口都使用 ``X-User-Id`` 传递匿名用户的稳定
标识。该标识目前只用于数据归属和后续检索范围过滤，不是登录凭证，也不能替代 JWT、
签名校验或 RBAC。把解析规则集中在这里，可以保证导入和查询对缺失身份的处理一致。
"""

from fastapi import HTTPException, Request


def get_current_user_id(request: Request) -> str:
    """
    读取并校验当前请求的轻量用户标识。

    缺少请求头和只包含空白字符都返回 400。这里不能自动回退到公共
    ``anonymous_user``：如果多个调用方忘记传 header，自动回退会把它们合并成同一个
    用户，既掩盖调用错误，也会给后续私有文档检索带来串数据风险。
    """
    user_id = request.headers.get("X-User-Id", "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少 X-User-Id 请求头")
    return user_id

