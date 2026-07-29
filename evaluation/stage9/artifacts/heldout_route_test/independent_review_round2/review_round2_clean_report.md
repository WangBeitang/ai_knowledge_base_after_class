# Stage 9 heldout route test round2 clean blind review

- bundle_id：`stage9-heldout-route-test-round2-clean-cf984fdfd3d49d7f`
- contamination_status：`clean`
- reviewer_id：`independent-agent-heldout-round2`
- reviewed_at：`2026-07-29T04:44:10Z`
- reviewed：5
- approved：5
- rejected：0

## 审核边界声明

- 已先完整阅读本轮包内 `REVIEW_INSTRUCTIONS.md`。
- 只读取了 `blind_review_bundle_round2/` 内文件，以及 `local_source_manifest.json` 明确指向的 3 份原始本地 PDF。
- 本轮没有 Web case，未访问网页。
- 没有读取 round1 历史审核材料、历史审核决定或其他逐 case 结论。
- 没有读取构建代码、阶段文档、正式 planner case 台账、Git 历史或 Git diff。
- 没有运行 heldout 模型推理、Planner 推理或 heldout test，也没有合并任何审核结果。
- 本报告只说明 case 审核结论，不代表或证明模型泛化能力。

## 文件、字节数与 SHA256 校验

`bundle_manifest.json` 的 8 个 `output_files` 均存在，实际字节数和 SHA256 均与清单一致：

| 文件 | SHA256 校验 | 实际 SHA256 |
|---|---|---|
| `REVIEW_INSTRUCTIONS.md` | 通过 | `27e203262f6d97f951764f2eab15e68fbc3aaf8365bfe860030541abab9d3722` |
| `leakage_reference.jsonl` | 通过 | `5e5b97c8d625f76cd3211264af178b36eabb0e720dbc4cc01d3cc1a77761c90c` |
| `local_evidence_manifest.json` | 通过 | `cdb0482a6b712c91340a1aed374c1a6b1fa4313c733da1462838ed0e97bc9f97` |
| `local_source_manifest.json` | 通过 | `3187929e628fd1d6f33706934cc47aa955812aa2f837a9be2117c8911c1216a8` |
| `review_cases.jsonl` | 通过 | `08bd255eb93d6bae502c0e172caec1aece3a30bfaebc3ea9502921013ed5a185` |
| `route_policy.json` | 通过 | `195ab47d57efddf061cd265379ab5ed6f08c567b6100c2344a57062542409716` |
| `web_evidence_manifest.json` | 通过 | `f03fc82f5f51b0fc2f0418183053a5538552ff780a6cbbacae8f299bee03136e` |
| `web_source_manifest.json` | 通过 | `ed22edbe73e25ea63ee22bb92b30181a28a9c9b142593da89d003da717849fcc` |

清单明确指向的 3 份原始 PDF 也完成 SHA256 身份校验：

| 原始 PDF | SHA256 校验 | 实际 SHA256 |
|---|---|---|
| 华为显示器 B5-341W 用户指南 | 通过 | `b599069adb3003fc70e37e602f36c9d3636a7eecff70b33ddff613e3f3876651` |
| HUAWEI MateBook B3-520 用户手册 | 通过 | `63fdee1c706e09c7bb9f3266cd72ceff73289900a89f12da2a2b8221303cb5c0` |
| 华为平板 C7 用户指南 | 通过 | `29d3023fc8fd6cd9a215cc2df5d880875bc94cc0664a26f87cbdb996fcff9087` |

`bundle_manifest.json` 中的 `input_files` 是来源逻辑路径，不是本轮包内交付文件；为遵守盲审读取边界，没有打开这些外部逻辑路径。

## Fingerprint 校验

按指定的 canonical JSON 序列化方式逐条复算，5 条均与 `case_fingerprint` 一致，case_id 无重复：

| case_id | 复算 fingerprint | 结果 |
|---|---|---|
| `planner-test-heldout-hyde-display-joystick-poweroff` | `cf2f5a7881c8f209714527918586999815becda85be88844016a34ab8e050140` | 通过 |
| `planner-test-heldout-hyde-matebook-upgrade-screen-recovery` | `0a22304896dd5cd60f6257a7f5bbc20e10a6842bf17b46dc5e31121fca49e431` | 通过 |
| `planner-test-heldout-hyde-tablet-screen-reader-off` | `1bb4e98abc594cae737211ddb80eb0178bd879e1e9a541875030f2c949864666` | 通过 |
| `planner-test-heldout-hyde-tablet-recording-transcript` | `89629e64b8b01a3047bd1455fc728add6854b1820eabec8d1515c0b482502622` | 通过 |
| `planner-test-heldout-ask-tablet-multiscreen-root-cause` | `cc85788db23cb8c500a9c5c88b875c6997cdbd110a4cc7a60baf652ee7edc65f` | 通过 |

## 逐 case 决定

### planner-test-heldout-hyde-display-joystick-poweroff — approved

原始 PDF 第 10 页明确说明五向摇杆关机方式为向上长按 3 秒以上，直至指示灯熄灭、显示器关机；设备、控件、时长和结果均与答案点一致。query 已给出 B5-341W 和关机意图，但使用“小疙瘩、彻底黑掉”的口语表达，主体仍唯一。目标 chunk 11 未进入 original top5，经 hypothetical query 后进入第 1 名，HyDE 路线成立。train/dev 未见相同或模板近重复问题，query 未泄漏关键操作。

### planner-test-heldout-hyde-matebook-upgrade-screen-recovery — approved

原始 PDF 第 17 页完整支持升级驱动、替换非官方软件、连接电源后 F10 恢复并先备份 C 盘、仍失败则携设备和购机发票送检的顺序。目标解决方案 chunk 33 未进入 original top5，经 hypothetical query 后进入第 2 名；不是仅从 rank 2 提升到 rank 1。query 明确 B3-520、升级后蓝黑屏和处理意图。train/dev 未见相同或模板近重复问题，也没有答案步骤泄漏。

### planner-test-heldout-hyde-tablet-screen-reader-off — approved

原始 PDF 第 97 页明确支持长按电源键调出关机和重启菜单，再双指长按屏幕 3 秒关闭屏幕朗读；同页行为描述也与“点哪都念、双指操作”的症状一致。目标 chunk 478 未进入 original top5，经 hypothetical query 后进入第 1 名。query 主体和关闭意图明确，但缺少“屏幕朗读”术语，HyDE 路线必要。train/dev 未见重复或答案泄漏。

### planner-test-heldout-hyde-tablet-recording-transcript — approved

原始 PDF 第 70 页完整支持转文本服务入口、华为帐号和免费时长/套餐、选择录音并开始转文本、结果显示位置。目标 chunk 322 未进入 original top5，经 hypothetical query 后进入第 1 名。query 明确 C7、已有会议录音和转成文字的目标，口语表达没有造成对象歧义。train/dev 未见重复，query 未泄漏服务入口和操作步骤。

### planner-test-heldout-ask-tablet-multiscreen-root-cause — approved

原始 PDF 第 40—41 页把“点亮 WLAN 和蓝牙”及“电脑管家 11.1 及以上”都列为连接条件。query 只确认 WLAN，未给出蓝牙状态和电脑管家版本，因此“蓝牙没开”和“版本太低”至少两个候选原因都符合当前已知条件，无法唯一判断。Planner 必须在本地检索后追问这两项状态，而不能直接定因或仅列检查步骤。train/dev 未见相同或模板近重复问题。

## HyDE 检索探针核验

| case_id | target chunk | original top5 | original rank | hypothetical top5 | hypothetical rank 复算 |
|---|---:|---|---|---|---|
| `planner-test-heldout-hyde-display-joystick-poweroff` | 11 | `[19, 26, 14, 20, 27]` | `null`，目标不在 top5 | `[11, 10, 13, 3, 6]` | 1，与声明一致 |
| `planner-test-heldout-hyde-matebook-upgrade-screen-recovery` | 33 | `[24, 32, 18, 55, 0]` | `null`，目标不在 top5 | `[32, 33, 18, 15, 35]` | 2，与声明一致 |
| `planner-test-heldout-hyde-tablet-screen-reader-off` | 478 | `[4, 467, 5, 67, 73]` | `null`，目标不在 top5 | `[478, 477, 101, 99, 100]` | 1，与声明一致 |
| `planner-test-heldout-hyde-tablet-recording-transcript` | 322 | `[123, 321, 118, 301, 122]` | `null`，目标不在 top5 | `[322, 324, 321, 323, 320]` | 1，与声明一致 |

4 条 case 均满足：原始目标证据未进入 top5，`original_target_rank` 为 `null`，目标 chunk 不在 `original_top5_chunk_indices`；hypothetical query 将目标带入 top5，且声明排名与列表位置一致。

## Ask clarification 核验

`planner-test-heldout-ask-tablet-multiscreen-root-cause` 确实缺少唯一判断所需信息。已知只有 WLAN 开启，未知：

1. 平板蓝牙是否开启；
2. 笔记本电脑管家是否为 11.1 及以上版本。

这两个条件均来自原始 PDF，且任一不满足都可能导致无法连接。合理路线为 `local_search -> ask_clarification`。

## Leakage 检查

- 仅使用包内 `leakage_reference.jsonl`。
- 逐条检查 train/dev 的 query、query_variants 和 leakage_group。
- 5 条 case 均未发现相同问题。
- 未发现仅替换型号、数值或近义表达的模板重复。
- 5 个 leakage_group 均未与参考集重叠。
- 4 条 HyDE query 均未直接泄漏完整答案或关键操作步骤。
- ask_clarification query 提出两个候选条件，但没有给出能唯一确定根因的状态，不构成答案泄漏。

## 汇总结论

- reviewed：5
- approved：5
- rejected：0
- 需要修复的 case：无
