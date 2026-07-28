# 阶段 9 balanced dev 第四轮 clean blind 独立审核报告

## 审核身份与时间

- reviewer_id：`independent-agent-round4-clean`
- reviewer_role：`independent_agent`
- 审核时间：`2026-07-28T11:05:32Z`

## Clean blind review 信息边界声明

本轮执行 clean blind review（干净盲审）。审核只读取：

- `evaluation/stage9/artifacts/balanced_dev/blind_review_bundle_v1/` 内的受控文件；
- `local_source_manifest.json` 明确列出的两份本地 PDF；
- `web_source_manifest.json` 明确列出的五个华为官方 URL。

本轮未读取任何历史审核结论、旧 decisions、旧报告、聊天或任务历史、其他 Agent 输出、Git 历史、项目数据库、MongoDB、Milvus、训练结果或模型输出，也未读取 manifest 未声明的网页、文件或代码。包内污染扫描字段为 `passed`、`historical_decision_count` 为 0；人工关键词检查未发现旧 reviewer、逐 case 历史 verdict 或可直接透露历史审核结果的字段。

## 审核包与完整性校验

- bundle_id：`stage9-balanced-dev-clean-9a23b4617a985b16`
- `bundle_manifest.json` 预期 SHA256：`0a142c4ddd8eca44223974b368dd0a8f85b2f76c8f1ce3f3a9aa5ba9b7827730`
- `bundle_manifest.json` 实测 SHA256：`0a142c4ddd8eca44223974b368dd0a8f85b2f76c8f1ce3f3a9aa5ba9b7827730`
- 受控路径集合：通过，无缺失或额外受控文件
- 受控文件 SHA256 与文件大小：8/8 通过
- case 数量：10，通过
- case_id 唯一性：10/10，通过
- case_fingerprint 唯一性：10/10，通过
- 按 `REVIEW_INSTRUCTIONS.md` 精确 JSON 序列化算法独立复算 fingerprint：10/10 一致
- 两份本地 PDF SHA256：2/2 与 `local_source_manifest.json` 一致

## 审核方法

1. 对每条 case 核对 fingerprint payload、eval contract、顶层证据引用之间的一致性。
2. 对 local case，从 manifest 指定 PDF 中只读提取对应原始段落，核对型号、操作步骤、数值、按键和安全条件。
3. 对 Web case，核对冻结 URL、抓取时间、HTTP 状态、响应/事实 hash 和 fact_id；只读访问五个 manifest 官方 URL，逐项复核预期事实。通用网页抓取器首次对三个 URL 超时或异常后，仅对原 manifest URL 直连重试，未访问页面内其他链接。
4. 按 `route_policy.json` 检查 `expected_behavior`、`acceptable_action_paths` 和 route rationale；HyDE case 同时检查原始目标排名和扩展后目标排名记录。
5. 将所有问题和 variants 与 `leakage_reference.jsonl` 做精确匹配、leakage group 重叠和文本相似性检查，再人工判断问题是否泄漏答案或构造标签。

## 路线分布

| route_bucket | case 数量 | 审核结果 |
|---|---:|---|
| `hyde_fallback` | 3 | 3 approved |
| `web_required` | 5 | 5 approved |
| `ask_clarification` | 2 | 2 approved |
| 合计 | 10 | 10 approved / 0 rejected |

## 逐 case 独立结论

1. `planner-dev-balanced-hyde-p5-internal-jam`：**approved**。P5 PDF 的“纸卡在出纸区域”段落完整支持打开上盖、取出硒鼓粉盒组件和完整拉出卡纸；原始未进 top5、HyDE 后 rank 1，路线成立；问题未泄漏具体步骤。
2. `planner-dev-balanced-hyde-p5-print-serial-page`：**approved**。P5 PDF 同时支持机器码与 S/N 的同义关系及“长按开始键 3 秒—滴声松开—打印信息页查看序列号”；HyDE 将操作 chunk 提升至 rank 1；问题未泄漏按键和时长。
3. `planner-dev-balanced-hyde-b5-scan-quality`：**approved**。B5 PDF 明确“最佳=1200dpi、内容多时建议、扫描较慢”；口语查询原始未进 top5、HyDE 后 rank 1；“清晰优先”是自然需求，不是答案泄漏。
4. `planner-dev-balanced-web-b5-firmware-upgrade-guidance`：**approved**。指定官方页及冻结事实支持通电联网、禁止断电关机、完成后自动重启和显示“01”，且适用产品包含 PixLab B5；指定日期官网表述必须走 Web。
5. `planner-dev-balanced-web-b5-shared-client-systems`：**approved**。指定官方页明确列出 Windows、macOS、Linux、UOS、KOS、中科方德，并要求电脑与打印机处于同一局域网；系统清单有时效性，Web 路线合理。
6. `planner-dev-balanced-web-p5-drum-replacement-guidance`：**approved**。指定官方页适用于擎云 P5，完整支持 CC 与红灯信号、硒鼓粉盒同时更换、关机静置半小时；问题没有泄漏这些答案。
7. `planner-dev-balanced-web-p5-product-os-list`：**approved**。指定产品页支持麒麟 KOS、统信 UOS、中科方德、Win 10 32/64 位及 Win 11 64 位以上；问题要求指定日期产品页列表，Web 路线准确。
8. `planner-dev-balanced-web-p5-official-print-specs`：**approved**。指定规格页支持单面 30 页/分钟、自动双面 14 面/分钟、预装硒鼓 15000 页、预装粉盒 1500 页；四个数值和单位均可直接核验。
9. `planner-dev-balanced-ask-printer-network-reset-model`：**approved**。B5 PDF 只能证明 B5 的网络状态键操作，问题未给具体型号，不能外推至所有华为打印机；直接询问型号是完成按键级回答的必要澄清。
10. `planner-dev-balanced-ask-id-copy-model`：**approved**。B5 PDF 证明 B5 使用 ID 复印键，但问题未给品牌或型号，未知面板不能默认套用 B5；询问型号是最小必要澄清。

## 泄漏检查结果

- 10 条 query/variants 与 `leakage_reference.jsonl` 均无精确文本复用。
- 10 个 `leakage_group_id` 均未出现在参考集合中。
- 未发现问题直接包含预期数值、按键时长、错误信号、耗材条件或完整操作步骤。
- case 9、10 与其他设备的网络重置、身份证复印问题存在正常的任务语义相似，但型号和答案不同，未构成跨 split 答案泄漏。
- 设备型号、产品名、错误码和“截至指定日期”等合法用户条件未被误判为泄漏。

## 汇总结论与 merge 建议

- approved：10
- rejected：0
- merge 建议：**授权这 10 条 case 进入后续 merge**。授权范围仅限本轮 bundle 中 fingerprint 已核验的 10 条 case，不外推到其他数据或版本。

本轮没有修改模型、GPU、SFT、数据标签、审核包、训练配置或其他文档；只新建了任务指定的 decisions JSONL 和本审核报告，未执行 commit 或 push。
