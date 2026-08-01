# SFT V2 独立盲审 round3 报告

- 审核身份：`round3` / `reviewer_c`
- 输入盲审包：`/Users/beitang/PycharmProjects/ai_knowledge_base_after_class/evaluation/stage9/artifacts/sft_v2/blind_review_bundle_v1`
- bundle ID：`sft-v2-clean-2d55a91841d2fa87`
- bundle 整体 SHA256：`d068ffd375099ea8fc0b779c409fb23556f49f05f5b66abe504e83fa7b3e89b7`
- 决定完整性：125/125，case_id 唯一
- 决定：approve=38，reject=87，needs_manual_verification=0

## 结论

本轮仅形成独立审核决定，不选出113条正式数据，不合并37条旧数据，不冻结正式集，不训练。主要拒绝原因是追问/拒答路线被人为延长、HyDE 未产生有效提升、网页证据截断或错绑，以及跨路线近义重复。

## 全局检查

- 完全重复：0 条
- 近义/模板重复：15 组，涉及 39 条候选；全局判定仅保留信息更完整且路线自然的代表项
- query 污染：4 条（sft-v2-new-046, sft-v2-new-089, sft-v2-new-091, sft-v2-new-092）
- split query 泄漏：0 条
- split chunk 泄漏：0 条
- evidence binding 失败：10 条（sft-v2-new-002, sft-v2-new-024, sft-v2-new-041, sft-v2-new-043, sft-v2-new-049, sft-v2-new-088, sft-v2-new-089, sft-v2-new-092, sft-v2-new-106, sft-v2-new-111）
- Provider/Observation 绑定失败：0 条

## 17条路线决定分布

| 路线 | 输入 | approve | reject | needs_manual_verification |
|---|---:|---:|---:|---:|
| `ask_clarification` | 7 | 5 | 2 | 0 |
| `local_search -> answer` | 12 | 12 | 0 | 0 |
| `local_search -> ask_clarification` | 9 | 0 | 9 | 0 |
| `local_search -> hyde_search -> answer` | 9 | 0 | 9 | 0 |
| `local_search -> hyde_search -> ask_clarification` | 6 | 0 | 6 | 0 |
| `local_search -> hyde_search -> refuse` | 4 | 0 | 4 | 0 |
| `local_search -> hyde_search -> web_search -> answer` | 12 | 4 | 8 | 0 |
| `local_search -> hyde_search -> web_search -> ask_clarification` | 5 | 0 | 5 | 0 |
| `local_search -> hyde_search -> web_search -> refuse` | 4 | 0 | 4 | 0 |
| `local_search -> refuse` | 5 | 0 | 5 | 0 |
| `local_search -> web_search -> answer` | 15 | 3 | 12 | 0 |
| `local_search -> web_search -> ask_clarification` | 5 | 0 | 5 | 0 |
| `local_search -> web_search -> refuse` | 3 | 0 | 3 | 0 |
| `refuse` | 3 | 3 | 0 | 0 |
| `web_search -> answer` | 16 | 11 | 5 | 0 |
| `web_search -> ask_clarification` | 6 | 0 | 6 | 0 |
| `web_search -> refuse` | 4 | 0 | 4 | 0 |

## 设备家族分布

| 名称 | 输入 | approve | reject | needs_manual_verification |
|---|---:|---:|---:|---:|
| `cnc_systems` | 5 | 2 | 3 | 0 |
| `cnc_turning` | 22 | 8 | 14 | 0 |
| `drive_software` | 5 | 1 | 4 | 0 |
| `hak180_equipment` | 1 | 1 | 0 | 0 |
| `hak180_safety` | 2 | 1 | 1 | 0 |
| `industrial_control_security` | 11 | 1 | 10 | 0 |
| `industrial_drives` | 5 | 1 | 4 | 0 |
| `industrial_firmware` | 7 | 1 | 6 | 0 |
| `industrial_machine_safety` | 17 | 6 | 11 | 0 |
| `lenovo_lj2268_printer` | 3 | 0 | 3 | 0 |
| `lenovo_m7208w_printer` | 3 | 1 | 2 | 0 |
| `lenovo_m7268_printer` | 2 | 1 | 1 | 0 |
| `lenovo_z26_printer` | 2 | 2 | 0 | 0 |
| `lenovo_z35_printer` | 1 | 1 | 0 | 0 |
| `machine_guarding` | 5 | 1 | 4 | 0 |
| `machine_guarding_regulation` | 6 | 1 | 5 | 0 |
| `manufacturing_cybersecurity` | 6 | 2 | 4 | 0 |
| `operational_technology_security` | 5 | 0 | 5 | 0 |
| `panda_pro_printer` | 1 | 1 | 0 | 0 |
| `pantum_p3000_printer` | 2 | 1 | 1 | 0 |
| `pantum_p3030_printer` | 1 | 0 | 1 | 0 |
| `pantum_p3500_printer` | 1 | 0 | 1 | 0 |
| `semiconductor_manufacturing` | 6 | 3 | 3 | 0 |
| `servo_drives` | 4 | 2 | 2 | 0 |
| `z26_generic_printer` | 2 | 0 | 2 | 0 |

## 问题家族分布

| 名称 | 输入 | approve | reject | needs_manual_verification |
|---|---:|---:|---:|---:|
| `cnc_programming_and_operation` | 22 | 8 | 14 | 0 |
| `cnc_system_capability_and_support` | 5 | 2 | 3 | 0 |
| `drive_lifecycle_and_maintenance` | 5 | 1 | 4 | 0 |
| `drive_software_lifecycle` | 5 | 1 | 4 | 0 |
| `firmware_lifecycle_and_security` | 7 | 1 | 6 | 0 |
| `generic_z26_document_operation` | 2 | 0 | 2 | 0 |
| `hot_stamping_equipment_operation` | 1 | 1 | 0 | 0 |
| `hot_stamping_safety_boundary` | 2 | 1 | 1 | 0 |
| `ics_advisory_and_response` | 11 | 1 | 10 | 0 |
| `lj2268_connectivity_and_printing` | 3 | 0 | 3 | 0 |
| `m7208w_document_handling` | 3 | 1 | 2 | 0 |
| `m7268_connectivity_and_operation` | 2 | 1 | 1 | 0 |
| `machine_guarding_compliance` | 6 | 1 | 5 | 0 |
| `machine_guarding_methods` | 5 | 1 | 4 | 0 |
| `manufacturing_cybersecurity_resources` | 6 | 2 | 4 | 0 |
| `ot_security_guidance` | 5 | 0 | 5 | 0 |
| `p3000_setup_and_maintenance` | 2 | 1 | 1 | 0 |
| `p3030_installation_and_safety` | 1 | 0 | 1 | 0 |
| `p3500_network_and_operation` | 1 | 0 | 1 | 0 |
| `panda_pro_features_and_maintenance` | 1 | 1 | 0 | 0 |
| `risk_assessment_and_safeguarding` | 17 | 6 | 11 | 0 |
| `semiconductor_security_profile` | 6 | 3 | 3 | 0 |
| `servo_lifecycle_and_replacement` | 4 | 2 | 2 | 0 |
| `z26_print_scan_operation` | 2 | 2 | 0 | 0 |
| `z35_print_scan_operation` | 1 | 1 | 0 | 0 |

## 来源分布

| 名称 | 输入 | approve | reject | needs_manual_verification |
|---|---:|---:|---:|---:|
| `abb-driveloader-lifecycle-2026` | 5 | 1 | 4 | 0 |
| `abb-legacy-servo-current` | 4 | 2 | 2 | 0 |
| `abb-powertrain-lifecycle-current` | 5 | 1 | 4 | 0 |
| `cisa-ics-advisories-2025-07-10` | 11 | 1 | 10 | 0 |
| `doc_0ec3f4068dfa44bb916a2ca1d68d98e7:467551067585610585:1` | 1 | 1 | 0 | 0 |
| `doc_0ec3f4068dfa44bb916a2ca1d68d98e7:467551067585610636:1` | 1 | 1 | 0 | 0 |
| `doc_1e32d6f9029448c7bd71046a211c8205:467551067585611281:2` | 1 | 1 | 0 | 0 |
| `doc_1e32d6f9029448c7bd71046a211c8205:467551067585611341:2` | 1 | 0 | 1 | 0 |
| `doc_2646f50becbc4c179f48c2ebc4275dd4:467551067585611486:2` | 1 | 1 | 0 | 0 |
| `doc_51d0e1c6b6eb4f9c97fc3a6a58ebfb3c:467551067585609647:1` | 1 | 1 | 0 | 0 |
| `doc_51d0e1c6b6eb4f9c97fc3a6a58ebfb3c:467551067585609652:1` | 1 | 0 | 1 | 0 |
| `doc_6a534dd285ae437ea5becd1d18039909:467551067585609736:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613577:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613580:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613594:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613623:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613631:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613636:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613646:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613659:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613670:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613704:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613737:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613745:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613760:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613776:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613779:1` | 1 | 0 | 1 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613797:1` | 1 | 1 | 0 | 0 |
| `doc_85776854260946abb2eaa0d7c506bb58:467551067585613829:1` | 1 | 0 | 1 | 0 |
| `doc_885b223c4bef450ba0b15752c395a448:467551067585610722:1` | 1 | 1 | 0 | 0 |
| `doc_8bd9c0a26a9a493fbceeba5791114078:467551067585611146:2` | 1 | 0 | 1 | 0 |
| `doc_8e04887560224cfa9332b7ab2247f93c:467551067585610868:1` | 1 | 0 | 1 | 0 |
| `doc_8e04887560224cfa9332b7ab2247f93c:467551067585610887:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585612859:1` | 2 | 1 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585612860:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585612869:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613060:1` | 1 | 1 | 0 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613092:1` | 1 | 1 | 0 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613112:1` | 1 | 1 | 0 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613147:1` | 1 | 1 | 0 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613194:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613265:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613299:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613340:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613372:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613377:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613459:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613502:1` | 1 | 0 | 1 | 0 |
| `doc_98c9f8c5ee7f47808ea511de1416c744:467551067585613562:1` | 1 | 0 | 1 | 0 |
| `doc_9b578874499d4650a2fc46acb271e527:467551067585610012:1` | 1 | 1 | 0 | 0 |
| `doc_9b578874499d4650a2fc46acb271e527:467551067585610016:1` | 1 | 0 | 1 | 0 |
| `doc_9b578874499d4650a2fc46acb271e527:467551067585610162:1` | 1 | 0 | 1 | 0 |
| `doc_cd73a1e7a9374773989382dcbe5898da:467551067585610246:1` | 1 | 1 | 0 | 0 |
| `doc_cd73a1e7a9374773989382dcbe5898da:467551067585610389:1` | 1 | 0 | 1 | 0 |
| `doc_da6e9fbf7cc241eba67db68bbc5e16f7:467551067585609800:1` | 1 | 0 | 1 | 0 |
| `doc_da6e9fbf7cc241eba67db68bbc5e16f7:467551067585609804:1` | 1 | 0 | 1 | 0 |
| `doc_da6e9fbf7cc241eba67db68bbc5e16f7:467551067585609934:1` | 1 | 0 | 1 | 0 |
| `doc_e76c338cd1604701812131e947685ef4:467551067585610964:3` | 1 | 0 | 1 | 0 |
| `nist-ir8546-semiconductor-profile` | 6 | 3 | 3 | 0 |
| `nist-manufacturing-cybersecurity-current` | 6 | 2 | 4 | 0 |
| `nist-sp800-82r3-current` | 5 | 0 | 5 | 0 |
| `osha-1910-212-current` | 6 | 1 | 5 | 0 |
| `osha-machine-guarding-current` | 5 | 1 | 4 | 0 |
| `rockwell-firmware-lifecycle-current` | 7 | 1 | 6 | 0 |
| `siemens-sinumerik-808-current` | 5 | 3 | 2 | 0 |
| `siemens-sinumerik-systems-current` | 5 | 2 | 3 | 0 |

## reject 与 needs_manual_verification

本轮无 `needs_manual_verification`。

| case_id | reason_codes | 直接原因 |
|---|---|---|
| `sft-v2-new-002` | `LOCAL_EVIDENCE_NOT_SUPPORTING_QUERY` | 本地证据是用户自定义按键说明，不能支持告警危险等级判断或该追问条件。 |
| `sft-v2-new-006` | `CLARIFICATION_NOT_REQUIRED` | 证据已经足以给出条件式安全答复：始终双手托底、保持平稳，并在靠近边缘时不要打开出纸盒；无需先追问位置。 |
| `sft-v2-new-023` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-024` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, EVIDENCE_BINDING_FAILED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-025` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-026` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-027` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-028` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-029` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-030` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-031` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失字段已在原始问题中明确出现，追问不是由 local_search 的 Observation 新暴露，路线应直接追问或给条件式答复。 |
| `sft-v2-new-032` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全、权限或法律边界，不应先执行 local_search。 |
| `sft-v2-new-033` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全、权限或法律边界，不应先执行 local_search。 |
| `sft-v2-new-034` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全、权限或法律边界，不应先执行 local_search。 |
| `sft-v2-new-035` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全、权限或法律边界，不应先执行 local_search。 |
| `sft-v2-new-036` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全、权限或法律边界，不应先执行 local_search。 |
| `sft-v2-new-040` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-041` | `EVIDENCE_BINDING_FAILED` | 网页事实只列出生命周期状态字段，没有给出 ACS550 的实际生命周期状态。 |
| `sft-v2-new-043` | `EVIDENCE_BINDING_FAILED` | 网页事实截断在“consistent and”，不足以完整支持生命周期公告的主要目的。 |
| `sft-v2-new-046` | `QUERY_POLLUTION, NEAR_TEMPLATE_DUPLICATE` | 把 NIST 页面勘误提示改造成“是否包含官方变更”的元问题，不是自然制造业任务。 |
| `sft-v2-new-049` | `EVIDENCE_BINDING_FAILED` | 问题询问多项具体产品，但绑定事实仅出现 Siemens TIA Project-Server/TIA Portal，证据覆盖不完整。 |
| `sft-v2-new-053` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失型号、目录号或风险点已在问题中明确出现，web_search 没有暴露新的歧义，应该直接追问。 |
| `sft-v2-new-054` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失型号、目录号或风险点已在问题中明确出现，web_search 没有暴露新的歧义，应该直接追问。 |
| `sft-v2-new-055` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失型号、目录号或风险点已在问题中明确出现，web_search 没有暴露新的歧义，应该直接追问。 |
| `sft-v2-new-056` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失型号、目录号或风险点已在问题中明确出现，web_search 没有暴露新的歧义，应该直接追问。 |
| `sft-v2-new-057` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失型号、目录号或风险点已在问题中明确出现，web_search 没有暴露新的歧义，应该直接追问。 |
| `sft-v2-new-058` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失型号、目录号或风险点已在问题中明确出现，web_search 没有暴露新的歧义，应该直接追问。 |
| `sft-v2-new-059` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求攻击或绕过安全防护，不应先执行 web_search。 |
| `sft-v2-new-060` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求攻击或绕过安全防护，不应先执行 web_search。 |
| `sft-v2-new-061` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求攻击或绕过安全防护，不应先执行 web_search。 |
| `sft-v2-new-062` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求攻击或绕过安全防护，不应先执行 web_search。 |
| `sft-v2-new-063` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-064` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-065` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-066` | `HYDE_NO_EFFECTIVE_IMPROVEMENT` | HyDE 仅使目标 chunk 提升一个名次且目标分数未形成实质改善，不能证明有效改写。 |
| `sft-v2-new-067` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-068` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-069` | `HYDE_NO_EFFECTIVE_IMPROVEMENT` | HyDE 仅使目标 chunk 提升一个名次且目标分数未形成实质改善，不能证明有效改写。 |
| `sft-v2-new-070` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-071` | `HYDE_INITIAL_RETRIEVAL_ALREADY_SUFFICIENT` | 首次 local_search 已召回目标 chunk（多数位于前1至2名），HyDE 没有形成必要的检索升级。 |
| `sft-v2-new-072` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失配置、几何、硬件通道或网络状态已在原始问题中明确，最终追问不是由 Observation 新暴露。 |
| `sft-v2-new-073` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失配置、几何、硬件通道或网络状态已在原始问题中明确，最终追问不是由 Observation 新暴露。 |
| `sft-v2-new-074` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失配置、几何、硬件通道或网络状态已在原始问题中明确，最终追问不是由 Observation 新暴露。 |
| `sft-v2-new-075` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失配置、几何、硬件通道或网络状态已在原始问题中明确，最终追问不是由 Observation 新暴露。 |
| `sft-v2-new-076` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失配置、几何、硬件通道或网络状态已在原始问题中明确，最终追问不是由 Observation 新暴露。 |
| `sft-v2-new-077` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失配置、几何、硬件通道或网络状态已在原始问题中明确，最终追问不是由 Observation 新暴露。 |
| `sft-v2-new-078` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求危险操作或规避法律限制，不应执行 local_search 和 HyDE 后才拒答。 |
| `sft-v2-new-079` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求危险操作或规避法律限制，不应执行 local_search 和 HyDE 后才拒答。 |
| `sft-v2-new-080` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求危险操作或规避法律限制，不应执行 local_search 和 HyDE 后才拒答。 |
| `sft-v2-new-081` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求危险操作或规避法律限制，不应执行 local_search 和 HyDE 后才拒答。 |
| `sft-v2-new-082` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-083` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-084` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-085` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-088` | `EVIDENCE_BINDING_FAILED` | 网页事实截断在“consistent and”，不足以完整回答生命周期模型的目的。 |
| `sft-v2-new-089` | `EVIDENCE_BINDING_FAILED, QUERY_POLLUTION` | 绑定内容只是 OSHA 页面 HTTPS 传输提示，与机器防护业务事实无关。 |
| `sft-v2-new-090` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-091` | `QUERY_POLLUTION, NEAR_TEMPLATE_DUPLICATE` | 把 NIST 页面勘误提示改造成“变更状态”元问题，不是自然 OT 安全任务。 |
| `sft-v2-new-092` | `EVIDENCE_BINDING_FAILED, QUERY_POLLUTION` | 问题询问网络安全教育与劳动力页面更新时间，但绑定 URL 是制造业行业资源页，页面身份与问题对象不一致。 |
| `sft-v2-new-094` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-095` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-096` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-097` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失机器型号、拓扑、固件或产品身份已在问题中明确，local_search/web_search 没有暴露新的歧义。 |
| `sft-v2-new-098` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失机器型号、拓扑、固件或产品身份已在问题中明确，local_search/web_search 没有暴露新的歧义。 |
| `sft-v2-new-099` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED` | 缺失机器型号、拓扑、固件或产品身份已在问题中明确，local_search/web_search 没有暴露新的歧义。 |
| `sft-v2-new-100` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失机器型号、拓扑、固件或产品身份已在问题中明确，local_search/web_search 没有暴露新的歧义。 |
| `sft-v2-new-101` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失机器型号、拓扑、固件或产品身份已在问题中明确，local_search/web_search 没有暴露新的歧义。 |
| `sft-v2-new-102` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求绕过安全框架或官方验证，不应检索后才拒答。 |
| `sft-v2-new-103` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全框架或官方验证，不应检索后才拒答。 |
| `sft-v2-new-104` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY` | 问题已明确要求绕过安全框架或官方验证，不应检索后才拒答。 |
| `sft-v2-new-106` | `EVIDENCE_BINDING_FAILED` | 网页事实只有 OSHA 官方站点安全连接提示，不能支持通用设备防护要求。 |
| `sft-v2-new-107` | `LOCAL_EVIDENCE_ALREADY_SUFFICIENT` | 首次 local_search 已返回直接包含 OSHA 1910.212 防护要求与替代防护方法的本地 chunk，继续 HyDE/web_search 属于不必要升级。 |
| `sft-v2-new-108` | `UNNECESSARY_LOCAL_HYDE_BEFORE_WEB` | 问题已明确指定 NIST/ABB 当前官方信息目标，local_search 与 HyDE 没有增加判定价值，应直接 web_search。 |
| `sft-v2-new-111` | `EVIDENCE_BINDING_FAILED` | 绑定事实未包含“十三项”数量，不能支持“具体有多少个”的回答。 |
| `sft-v2-new-113` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-114` | `NEAR_SEMANTIC_DUPLICATE` | 与同来源、同答案目标的候选构成跨路线近义重复；全局去重后保留信息更完整且路线更自然的代表项。 |
| `sft-v2-new-115` | `UNNECESSARY_LOCAL_HYDE_BEFORE_WEB` | 问题已明确指定 NIST/ABB 当前官方信息目标，local_search 与 HyDE 没有增加判定价值，应直接 web_search。 |
| `sft-v2-new-116` | `MISSING_PRODUCT_IDENTITY_REQUIRES_CLARIFICATION` | 问题未给出伺服品牌或型号，却直接绑定 ABB 生命周期政策并泛化回答；应先追问产品身份。 |
| `sft-v2-new-117` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失版本、危害点、拓扑、设备型号或目录号已在问题中明确，不需要完整三段检索后才追问。 |
| `sft-v2-new-118` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失版本、危害点、拓扑、设备型号或目录号已在问题中明确，不需要完整三段检索后才追问。 |
| `sft-v2-new-119` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失版本、危害点、拓扑、设备型号或目录号已在问题中明确，不需要完整三段检索后才追问。 |
| `sft-v2-new-120` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失版本、危害点、拓扑、设备型号或目录号已在问题中明确，不需要完整三段检索后才追问。 |
| `sft-v2-new-121` | `CLARIFICATION_PREDECLARED_NOT_OBSERVATION_TRIGGERED, NEAR_TEMPLATE_DUPLICATE` | 缺失版本、危害点、拓扑、设备型号或目录号已在问题中明确，不需要完整三段检索后才追问。 |
| `sft-v2-new-122` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求绕过或瘫痪安全系统，应直接拒答，不应执行三段检索。 |
| `sft-v2-new-123` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求绕过或瘫痪安全系统，应直接拒答，不应执行三段检索。 |
| `sft-v2-new-124` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求绕过或瘫痪安全系统，应直接拒答，不应执行三段检索。 |
| `sft-v2-new-125` | `UNSAFE_REQUEST_SHOULD_REFUSE_DIRECTLY, NEAR_TEMPLATE_DUPLICATE` | 问题已明确要求绕过或瘫痪安全系统，应直接拒答，不应执行三段检索。 |

## 指纹与网页边界

- 125条 `blind_case_fingerprint` 已全部独立复算通过。原始 `content_fingerprint` 因盲审字段白名单移除了生命周期字段，无法仅凭盲审包重建；决定仍绑定其原值，并同时记录可复算的盲审指纹。
- 12个官方网页来源中，6个当前响应 SHA256 与冻结值完全一致；4个响应变化但当前官方正文仍支持关键事实；2个 OSHA 页面当前抓取受限，使用 2026-07-31 冻结的官方正文与响应 SHA256。

## 停止边界

- `formal_dataset_frozen=false`
- `training_performed=false`
- 未读取任何其他审核目录、决定、报告或汇总结论。
- 未执行分歧合并、正式集冻结或任务9.3.23。
