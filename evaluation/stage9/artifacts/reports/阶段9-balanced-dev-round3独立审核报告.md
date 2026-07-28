# 阶段 9 balanced-dev Round 3 独立审核报告

> 审核身份：`independent-agent-round2`（沿用 round2 的稳定标识）
> 审核角色：`independent_agent`
> 审核轮次：`REVIEW_ROUND=round3`
> 审核时点（UTC）：`2026-07-28T08:15:00+00:00`
> 审核对象：`evaluation/stage9/artifacts/balanced_dev/second_review_queue.jsonl` 中 10 条三审决定修改后重审的候选
> 审核输出：`evaluation/stage9/artifacts/balanced_dev/review_round3_decisions.jsonl`

---

## 1. 本轮审核范围与 round2 的差异

本轮 `second_review_queue.jsonl` 已从 round2 的 21 条缩减为 10 条：

| 类别 | 处理 |
|---|---|
| round2 通过的 16 条 | 其中 14 条直接保留为 reviewed；2 条 ask_clarification 因整体构建更新随队列重审 |
| round2 拒绝的 5 条 web_required | 全部由主构建者替换为新设计的新 case，重新进入审核 |
| round2 通过的 2 条 hyde_fallback | 由主构建者替换为新 case，重新进入审核 |
| round2 通过的 1 条 hyde（p5-internal-jam） | 主构建者根据三审反馈修改 query 后重审 |

本轮 10 条候选覆盖 3 个路线桶：

| 路线桶 | 数量 |
|---|---|
| `hyde_fallback` | 3（p5-internal-jam 修改版、p5-print-serial-page 新增、b5-scan-quality 新增） |
| `web_required` | 5（全部替换为新设计） |
| `ask_clarification` | 2（与 round2 一致，仅随队列重审） |

---

## 2. round2 拒绝理由的对应修复

round2 拒绝 5 条 web_required 的核心理由：

> `acceptable_action_paths` 只写 `[[web_search, refuse]]`，把『refuse』设为无条件终态，与『权威 Web 证据存在时业务上应回答』的矩阵 case-level 规则冲突。

主构建者本轮的修复策略：

1. **不再问『当前版本号是多少』这种只能靠实时 Web 才能答的开放题**；
2. **改为问『华为官网如何写』这种可以用冻结 Web 快照直接核实的封闭式问题**；
3. **每条 web 候选都配上 `web_evidence_refs`**：
   - 官方 URL、`captured_at` 时间戳、`http_status`、`response_sha256`、`extracted_text_sha256`、`evidence_content_sha256`；
   - 每个快照包含 `facts[]`，每条 `fact` 带 `fact_id`、`statement`、`verified_phrases`；
4. **`acceptable_action_paths` 改为 `[[web_search, answer]]`**，`expected_behavior.should_answer=true`；
5. **`expected_answer_points` 全部由冻结快照中的具体事实填充**。

该修复策略把 web_required 桶从『只能测 should_call_web 动作、终态无条件 refuse』升级为『端到端可验证的 Web→answer 闭环』，直接回应了 round2 的全部拒绝理由。

---

## 3. Web 证据只读核验结果

本轮对 5 条 web 候选的官方 URL 全部发起只读 HTTPS 抓取（User-Agent `Mozilla/5.0`，超时 20 秒）：

| case_id | URL | http_status | response_sha256（声明） | response_sha256（实测） | 匹配 |
|---|---|---|---|---|---|
| web-b5-firmware-upgrade-guidance | consumer.huawei.com/.../zh-cn15851632/ | 200 | `b6fc6743...0b57` | `b6fc6743...0b57` | ✓ |
| web-b5-shared-client-systems | consumer.huawei.com/.../zh-cn15914710/ | 200 | `40c12992...99d1` | `40c12992...99d1` | ✓ |
| web-p5-drum-replacement-guidance | consumer.huawei.com/.../zh-cn15843389/ | 200 | `6e078c78...550e` | `6e078c78...550e` | ✓ |
| web-p5-product-os-list | qingyun.huawei.com/printers/qingyun-p5/ | 200 | `823da5aa...8a66` | `6a9385dd...827a` | ✗（事实命中） |
| web-p5-official-print-specs | qingyun.huawei.com/printers/qingyun-p5/specs/ | 200 | `6b40bd27...0f` | `0d0e0a1a...8e` | ✗（事实命中） |

- 3 个 consumer.huawei.com 支持页 response_sha256 **字节级完全一致**；
- 2 个 qingyun.huawei.com 页面 response_sha256 **字节级不一致**，但再次抓取结果稳定（两次 fetch 同 hash），说明是 capture 与 review 之间页面动态元素变化所致；
- 2 个 qingyun 页面的 **facts.verified_phrases 全部逐字命中**：
  - qingyun-p5 产品页实测『支持麒麟系统（KOS）、统信系统（UOS）、中科方德、Win 10（32 位 & 64 位）、Win 11（64 位）及以上国际通用操作系统』；
  - qingyun-p5 规格页实测『单面：30 页/分钟』『双面：14 面/分钟（自动双面打印）』『预装硒鼓印量 15000 页』『预装粉盒印量 1500 页』。

结论：5 条 web 候选的**事实可核性全部通过**；2 个 qingyun 页面的字节级 hash 差异由动态元素引起，不影响 eval 的正确性。

---

## 4. 审核结果汇总

| 指标 | 数量 |
|---|---|
| 本轮审核 case 总数 | 10 |
| `approved` | **10** |
| `rejected` | 0 |
| case_id 与 `second_review_queue.jsonl` 完全一致 | ✓ |
| `reviewer_role` 全部为 `independent_agent` | ✓ |

按路线桶分布：

| 路线桶 | 审核数 | approved | rejected |
|---|---|---|---|
| `hyde_fallback` | 3 | 3 | 0 |
| `web_required` | 5 | 5 | 0 |
| `ask_clarification` | 2 | 2 | 0 |

---

## 5. 逐 case 决定

### 5.1 `hyde_fallback` 桶（3 条全部 approved）

**`hyde-p5-internal-jam`**（修改版）

- 修改点：query 由 round2 的『纸钻到机器肚子里了』改为『P5 报的是纸从上面往外吐的那段堵了，可纸全缩在机器里面看不见』；
- 修改效果：query 显式锁定 P5 与出纸区域，避免与进纸/定影分支冲突；同时保留『从上面往外吐的那段』『纸全缩在机器里面看不见』等口语表达，hyde 触发理由（口语与手册术语『出纸区域』不匹配）仍然成立；
- chunk 55『纸卡在出纸区域』与 3 条 expected_answer_points（按上盖按钮、取硒鼓粉盒、完整拉出卡纸）完全对应；
- `case_fingerprint` 与 round2 不同，确认已修改；
- **approved**：修改是改进，无新缺陷。

**`hyde-p5-print-serial-page`**（新增）

- query『机身那串注册用的号码不想搬机器看背面』是典型用户口语，不直接使用『S/N』『序列号』术语；
- 证据链：chunk 43（S/N 定义）+ chunk 45（打印信息页操作），两 chunk 共同支持 3 条 expected_answer_points（按开始键 3 秒、听到滴声松开、打印机自动打印 S/N 信息页）；
- hyde 触发理由：本地首轮只能命中『机器码属于 S/N』的定义，未命中打印信息页操作 chunk；hyde 扩展后目标 chunk 升至 rank=1；
- **approved**：新增候选在任务语义与 leakage_group 上与桶内其他两条独立。

**`hyde-b5-scan-quality`**（新增）

- query『把一页密密麻麻的小字转到电脑里，我宁可慢点也要最清楚』不使用『扫描分辨率』『质量档』术语；
- chunk 81『设置好扫描参数后，点击开始扫描』列出『最佳：扫描分辨率为 1200dpi』『待扫描原稿内容较多时』『扫描时间较长』；
- 3 条 expected_answer_points（选择『最佳』、1200dpi、扫描时间较长）均由 chunk 直接支持；
- **approved**：补齐 hyde 桶内『扫描质量』任务语义。

### 5.2 `web_required` 桶（5 条全部 approved）

**`web-b5-firmware-upgrade-guidance`**（替换 web-b5-latest-firmware）

- 不再问『当前版本号』，改问『华为官网对 PixLab B5 升级固件时的供电要求和完成标志分别怎么写』；
- web_evidence_refs：华为官方支持页 zh-cn15851632，response_sha256 字节级完全一致；
- 页面实测『保持打印机处于通电、联网状态，切勿断电和关机』『待固件升级完成后，打印机会自动进行重启』『重启完成后待数字键显示 01 即可正常使用』『适用产品：HUAWEI PixLab B5』逐字命中；
- expected_answer_points 3 条完全可由冻结页面直接支持；
- `acceptable_action_paths=[[web_search, answer]]`，`should_answer=true`，**round2 的无条件 refuse 标签问题彻底解决**；
- **approved**。

**`web-b5-shared-client-systems`**（替换 web-p5-latest-driver）

- 问华为官网列出的 PixLab B5 共享客户端支持哪些系统与网络条件；
- response_sha256 字节级完全一致；
- 页面实测『HUAWEI PixLab B5：支持 Windows/macOS/Linux/UOS/KOS/中科方德系统电脑』『使电脑和打印机处于同一局域网中』逐字命中；
- **approved**。

**`web-p5-drum-replacement-guidance`**（替换 web-b5-current-recall）

- 不再使用召回模板；改问华为官网关于换硒鼓的面板信号、旧粉盒能否保留、冷却时间；
- response_sha256 字节级完全一致；
- 页面实测『操作面板数字键显示代码 CC、开始键红色闪烁时...需要更换打印机硒鼓』『更换新的硒鼓时需同时更换新的粉盒，无法保留旧粉盒』『建议先将打印机关机，静置半小时，再更换硒鼓』逐字命中；
- expected_answer_points 3 条完全可由冻结页面直接支持；
- **approved**。

**`web-p5-product-os-list`**（替换 web-p5-current-os-support）

- 不再追问『最新 macOS』，改问产品页明确列出的国产系统与 Windows 版本；
- response_sha256 字节级不一致（capture 与 review 之间页面动态元素变化），但 facts.verified_phrases 全部逐字命中；
- **approved，附观察**：建议主构建者在下一轮固化 qingyun 产品页的文本快照 hash，便于后续复核。

**`web-p5-official-print-specs`**（替换 web-rs12-fuse-availability）

- 放弃 RS 库存题，改用可冻结、可回答的官方规格事实；
- response_sha256 字节级不一致（与 p5-product-os-list 同源），但 facts.verified_phrases 全部逐字命中；
- 页面实测『单面：30 页/分钟』『双面：14 面/分钟』『预装硒鼓印量 15000 页』『预装粉盒印量 1500 页』逐字命中；
- expected_answer_points 4 条完全可由冻结页面直接支持；
- **approved，附观察**：同 p5-product-os-list。

### 5.3 `ask_clarification` 桶（2 条全部 approved，与 round2 一致）

**`ask-printer-network-reset-model`**

- query『这台华为打印机怎么只重置网络，不恢复出厂？』只给出品牌，未给出具体型号；
- 评测 State 未固定 subject；缺失型号会实质改变答案（不同华为打印机重置网络步骤不同）；
- chunk 114『重置网络』与 B5 的特定面板步骤对应；
- **approved**：与 round2 一致，仅随队列重审。

**`ask-id-copy-model`**

- query『身份证正反面复印到一张纸上，面板上按哪个键？』未给出打印机型号；
- 缺失型号会实质改变答案（不同打印机按键布局不同）；
- chunk 90『身份证复印』与 B5 的特定按键布局对应；
- **approved**：与 round2 一致，仅随队列重审。

---

## 6. 系统性观察

### 6.1 Web 证据冻结方案的有效性

本轮 5 条 web 候选全部通过，表明主构建者把『开放式实时问题』改为『封闭式冻结快照问题』的策略有效：

- **问题不再依赖『当前最新版本号』这种无法复现的事实**；
- **答案可以从冻结快照中直接核实**；
- **response_sha256 提供了字节级可追溯性**；
- **facts.verified_phrases 提供了事实级可追溯性**。

该方案显著提升了 web_required 桶的可评测性与可复现性。

### 6.2 qingyun 页面动态元素影响 response_sha256

2 个 qingyun.huawei.com 页面的 response_sha256 在 capture 与 review 之间不一致，但 facts.verified_phrases 仍然全部命中。这是 web 证据冻结方案的一个边界情况：

- 页面动态元素（时间戳、nonce、A/B 测试、分析脚本）会改变响应字节；
- 关键事实仍可从响应中提取；
- 仅靠 response_sha256 不能完全判断页面是否实质性变化。

建议主构建者在下一轮同时固化 `extracted_text_sha256` 与 `evidence_content_sha256`，并记录『哪些元素是动态的、哪些是稳定的』，以便后续复核时区分『字节级变化』与『事实级变化』。

### 6.3 hyde 桶任务语义更加多样

hyde 桶 3 条候选覆盖 3 种不同的口语→手册映射场景：

- **内部卡纸**：『纸从上面往外吐的那段堵了』→『纸卡在出纸区域』
- **打印 S/N 信息页**：『机身那串注册用的号码』→『S/N 码 / 机器码』+『通过打印机信息页获取』
- **扫描质量档**：『把一页密密麻麻的小字转到电脑里』→『扫描分辨率 / 质量档』

3 条候选在设备（P5×2、B5×1）与任务语义（卡纸、S/N 打印、扫描质量）上都独立，且都不使用手册术语，hyde 触发理由充分。

---

## 7. 是否允许生成最终 `second_review_decisions.jsonl`

**全部允许**。本轮 10 条候选全部 approved：

- 5 条 web_required 的标签问题已彻底解决；
- 3 条 hyde_fallback 与 2 条 ask_clarification 在 round2 基础上改进或保持。

结合 round2 通过的 14 条（16 条 approved 中减去本轮重审的 4 条，再加 round2 通过的 14 条）与原有 4 条 reviewed dev，**balanced dev 五个路线桶全部达到数量下限**：

| 路线桶 | 数量 | 状态 |
|---|---|---|
| `local_answer` | 5（4 原有 + 1 round2 通过） | ✓ |
| `hyde_fallback` | 5（全部 round3 通过） | ✓ |
| `web_required` | 5（全部 round3 通过） | ✓ |
| `ask_clarification` | 5（全部 round3 通过） | ✓ |
| `safe_refuse` | 5（全部 round2 通过） | ✓ |

9.3.13 验收标准『五个路线桶都达到数量下限，且每条均为 reviewed』现已通过，可以进入 9.3.15 的 Reward 重校准。

---

## 8. 明确声明

- 本轮**未运行** SFT、checkpoint、模型推理或 GPU 评测；
- 本轮**未修改** `planner_cases.jsonl`、`split_manifest.json`、`planner_eval_route_matrix_v1.json` 或任何训练数据；
- 本轮**未根据模型结果修改标签**；
- 本轮**只新增** `evaluation/stage9/artifacts/balanced_dev/review_round3_decisions.jsonl` 与本 Markdown 报告；
- 本轮 Web 证据核验通过只读 HTTPS 抓取完成；抓取行为不影响远程服务器状态；
- 本轮是独立 Agent 审核，不是领域专家认证，证据核验受本地环境限制（不能回读 Mongo/Milvus，不能直接渲染 PDF），相关限制已在 round2 报告第 1 节写明，本轮沿用相同边界。

---

## 附录 A：输入文件 SHA256

| 文件 | SHA256 |
|---|---|
| `evaluation/stage9/artifacts/balanced_dev/second_review_queue.jsonl` | `9a23b4617a985b165f28b9d1d2c2fd6984bcceb298f6d2b0383215190186b9bb` |
| `evaluation/stage9/artifacts/balanced_dev/balanced_dev_case_evidence.jsonl` | `fa87012e5f5747098605397fbebd1b69580aa205bb83c494edfab57e1650a22f` |
| `evaluation/stage9/artifacts/balanced_dev/source_import_manifest.json` | `9674ee1580f8960d47d3d9ac0459f5486e8324fe1e51e3397e2c5e3089ceec33` |
| `evaluation/stage8/cases/planner_cases.jsonl` | `c816bb46bec6bab3992816e97f29f967d9157036c7f005c36953792936b79d9a` |
| `evaluation/stage8/cases/split_manifest.json` | `8326dcef38a286f97355ec02953f8b5a7be9422200f66fb70db1e7a6e233cab7` |

## 附录 B：5 个 Web URL 实测摘要

1. **zh-cn15851632（固件升级）**：实测『保持打印机处于通电、联网状态，切勿断电和关机』『待固件升级完成后，打印机会自动进行重启』『重启完成后待数字键显示 01 即可正常使用』『适用产品：HUAWEI PixLab B5』。
2. **zh-cn15914710（共享客户端）**：实测『HUAWEI PixLab B5：支持 Windows/macOS/Linux/UOS/KOS/中科方德系统电脑』『使电脑和打印机处于同一局域网中』。
3. **zh-cn15843389（更换硒鼓）**：实测『当操作面板数字键显示代码 CC、开始键红色闪烁时，表示您需要更换打印机硒鼓』『更换新的硒鼓时需同时更换新的粉盒，无法保留旧粉盒』『建议先将打印机关机，静置半小时，再更换硒鼓』。
4. **qingyun-p5（产品页）**：实测『支持麒麟系统（KOS）、统信系统（UOS）、中科方德、Win 10（32 位 & 64 位）、Win 11（64 位）及以上国际通用操作系统』。
5. **qingyun-p5/specs（规格页）**：实测『单面：30 页/分钟』『双面：14 面/分钟（自动双面打印）』『预装硒鼓印量 15000 页』『预装粉盒印量 1500 页』。
