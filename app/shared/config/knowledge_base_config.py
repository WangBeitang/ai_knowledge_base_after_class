"""
知识库跨链路默认值。

导入链路负责把 dataset/tenant 归属写进 document 和 chunk，查询链路则使用同一组
默认值构造检索上下文。把这些值放在共享配置层，可以避免导入与查询分别硬编码后逐渐
产生偏差；这里仅定义当前轻量单租户阶段的默认值，不代表已经实现完整权限系统。
"""


DEFAULT_DATASET_ID = "dataset_default_equipment_ops"
DEFAULT_TENANT_ID = "tenant_default"
DEFAULT_VISIBILITY = "private"

