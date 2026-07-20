# 阶段 8.5 公开数据候选池 gold 审核报告
## 结论
- 本次按“高置信 gold”标准复核 52 条候选 case。
- 结论：52 条均不能原样进入 gold。
- 原因不是 schema 不合格，而是当前答案要点包含公开来源未直接支撑的维修动作、设备根因或业务推断。
- 这些样本仍可保留为 silver/候选池；其中部分可以重写为只依赖 UCI 来源字段和失效标签的 gold 样本。

## 审核标准
- 每个 expected_answer_points 都必须能被公开来源页面、官方变量说明、profile 标签说明或本地可追溯证据直接支撑。
- 只要答案点包含来源未说明的维修动作、根因、部件失效机理、维护记录或经验性处置，就不能原样进入 gold。
- `approved_cases.jsonl` 只代表阶段 8.5 流程放行，不等于 gold/domain verified。
- `label_source=synthetic` 和 `seed_interpretation_requires_domain_review` 是强风险信号。

## 统计
- 总 case：52
- 原样 gold：0
- 非 gold：52
- 来源支撑等级 partial：26
- 来源支撑等级 unsupported：8
- 来源支撑等级 weak：18
- 后续动作 drop_from_gold_or_keep_as_silver_only：8
- 后续动作 rewrite_to_source_grounded_gold_candidate：44

## 公开来源依据
- uci-metropt3：MetroPT-3 Dataset，https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset。UCI 页面支持：APU compressor 场景、压力/油温/电机电流/进气阀等 15 个信号、Air Leak 故障报告时间段；不直接给出逐条维修动作或多数具体根因。
- uci-ai4i-2020：AI4I 2020 Predictive Maintenance Dataset，https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset。UCI 页面支持：合成预测维护数据、Type/Air temperature/Process temperature/Rotational speed/Torque/Tool wear 等字段，以及 TWF/HDF/PWF/OSF/RNF 五类失效规则；不支持具体维修动作、表面质量、传动链或维护记录推断。
- uci-hydraulic-condition：Condition Monitoring of Hydraulic Systems，https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems。UCI 页面支持：液压试验台、压力/流量/温度/振动/虚拟效率等传感器，以及 cooler、valve、internal pump leakage、accumulator、stable flag 的状态标签；不支持具体维修处置和很多二级故障根因。

## 逐来源判断
### AI4I 2020 Predictive Maintenance Dataset
- `ai4i-2020-heat-dissipation-failure`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `hdf_rule_supported_but_cooling_maintenance_not_supported`。建议：可改成：HDF 条件是 air/process temperature 差值低于 8.6K 且转速低于 1380 rpm。
- `ai4i-2020-low-speed-high-torque`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `speed_and_torque_supported_but_vibration_lubrication_not_supported`。建议：可改成 PWF 规则样本，不加入振动或润滑。
- `ai4i-2020-overstrain-failure`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `osf_rule_supported_but_fixture_material_actions_not_supported`。建议：可改成：OSF 是 tool wear 与 torque 乘积超过 L/M/H 不同阈值。
- `ai4i-2020-post-maintenance-baseline`：2 条 case，结论 `not_gold_as_is`，支撑等级 `unsupported`，原因 `maintenance_records_not_present_in_ai4i_source`。建议：不建议保留；AI4I 来源没有维修记录字段。
- `ai4i-2020-power-failure-high-load`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `pwf_rule_supported_but_drive_maintenance_not_supported`。建议：可改成：PWF 由 torque 和 rotational speed 计算功率，低于 3500W 或高于 9000W 失败。
- `ai4i-2020-product-type-risk`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `product_type_and_osf_thresholds_supported_but_process_window_language_too_broad`。建议：可改成：OSF 对 L/M/H 产品类型使用 11000/12000/13000 minNm 阈值。
- `ai4i-2020-random-failure-review`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `rnf_independence_supported_but_maintenance_record_not_supported`。建议：可改成：RNF 是每个过程 0.1% 随机失败，独立于过程参数，因此不应强行归因到单一传感器。
- `ai4i-2020-temperature-drift`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `temperature_fields_supported_but_drift_sensor_fault_not_supported`。建议：可改成 HDF 规则样本，不加入传感器校准。
- `ai4i-2020-tool-wear-failure`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `twf_rule_supported_but_surface_quality_and_actions_not_supported`。建议：可改成：TWF 是 tool wear 在 200-240 min 随机点发生换刀或失效；回答只引用 Tool wear/TWF。
### Condition Monitoring of Hydraulic Systems
- `hydraulic-condition-accumulator-pressure-low`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `accumulator_pressure_label_supported_but_diaphragm_actions_not_supported`。建议：可改成：profile.txt 第 4 列 accumulator pressure 的 130/115/100/90 bar 状态含义。
- `hydraulic-condition-cooler-efficiency-low`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `cooler_condition_label_supported_but_cleaning_actions_not_supported`。建议：可改成：profile.txt 第 1 列表示 cooler condition，3/20/100 分别代表接近失效、效率降低、完全效率。
- `hydraulic-condition-flow-sensor-drop`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `flow_sensors_supported_but_drop_fault_not_labeled`。建议：可改成字段解释样本：FS1/FS2 是 volume flow 传感器。
- `hydraulic-condition-internal-pump-leakage`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `pump_leakage_label_supported_but_seal_actions_not_supported`。建议：可改成：profile.txt 第 3 列 internal pump leakage 的 0/1/2 分别代表无、弱、严重泄漏。
- `hydraulic-condition-oil-degradation-suspected`：2 条 case，结论 `not_gold_as_is`，支撑等级 `unsupported`，原因 `oil_degradation_not_in_profile_targets`。建议：不建议从该来源构造油液劣化 gold；UCI 页面目标标签不包含油液状态。
- `hydraulic-condition-stable-flag-mismatch`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `stable_flag_supported_but_operational_remediation_not_source_text`。建议：可改成：stable flag 为 0 表示条件稳定，为 1 表示静态条件可能尚未达到；评估时应区分两类周期。
- `hydraulic-condition-valve-switching-delay`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `valve_condition_labels_supported_but_contamination_actions_not_supported`。建议：可改成：profile.txt 第 2 列表示 valve condition，100/90/80/73 对应最优、小延迟、严重延迟、接近失效。
- `hydraulic-condition-vibration-rise`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `vibration_sensor_supported_but_bearing_fault_not_labeled`。建议：可改成字段解释样本：VS1 是 vibration 传感器。
### MetroPT-3 Dataset
- `metropt3-abnormal-current-draw`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `motor_current_signal_supported_but_mechanical_fault_not_supported`。建议：可改成字段解释样本：Motor_current 不同数值大致对应关闭、空载、加载和启动状态。
- `metropt3-air-leak-pressure-recovery`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `source_supports_air_leak_and_signals_but_not_repair_actions`。建议：可改成：MetroPT-3 的 failure report 记录了哪些 Air Leak 时间段？排查 Air Leak 相关异常时可先查看哪些压力/COMP/DV_pressure 信号？
- `metropt3-compressor-over-cycling`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `cycling_diagnosis_exceeds_source_description`。建议：可改成字段解释样本：Reservoirs、COMP、MPG、LPS 如何描述 APU 压力和压缩机状态。
- `metropt3-cooling-fan-ineffective`：2 条 case，结论 `not_gold_as_is`，支撑等级 `unsupported`，原因 `cooling_fan_not_in_source_variables`。建议：不建议从该来源构造 cooling fan 维修 gold；最多保留 Oil_temperature 字段解释。
- `metropt3-maintenance-after-leak`：2 条 case，结论 `not_gold_as_is`，支撑等级 `partial`，原因 `maintenance_timestamps_exist_but_post_maintenance_diagnosis_not_supported`。建议：可改成：MetroPT-3 failure report 中 Air Leak 与 Maintenance 时间如何记录。
- `metropt3-oil-temperature-rise`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `oil_temperature_signal_supported_but_root_causes_not_supported`。建议：可改成字段解释样本：Oil_temperature 是压缩机油温信号，可用于观察温度变化，但不要推断润滑或散热故障。
- `metropt3-pressure-drop-after-stop`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `pressure_drop_after_stop_not_explicitly_supported`。建议：可改成：Reservoirs 表示下游储气罐压力，Air Leak 期间可观察压力相关信号。
- `metropt3-sensor-flatline`：2 条 case，结论 `not_gold_as_is`，支撑等级 `unsupported`，原因 `sensor_flatline_fault_not_in_source_description`。建议：不建议从该来源构造传感器失效 gold；可只做传感器字段含义问答。
- `metropt3-valve-lag-response`：2 条 case，结论 `not_gold_as_is`，支撑等级 `weak`，原因 `valve_lag_failure_mode_not_explicitly_supported`。建议：可改成信号解释样本：COMP、DV electric、TOWERS、MPG 等电信号分别代表什么。

## 建议
- 不要把当前 24 条 `approved_cases.jsonl` 直接当 gold 使用。
- 下一步优先从 AI4I 的 TWF/HDF/PWF/OSF/RNF 规则和 Hydraulic 的 profile 标签重写 10-20 条 source-grounded gold。
- MetroPT 优先做字段解释、Air Leak 时间段和信号含义样本，不做维修动作或复杂根因样本。
- 新 gold 样本应新增 `gold_evidence_text` 或等价字段，保存可读原文依据摘要，不能只写 chunk_id。

## 后续重写结果

- 已由 `build_source_grounded_gold.py` 重写 20 条独立 gold，原 52 条候选和本报告的原样审核结论保持不变。
- AI4I 10 条覆盖 TWF/HDF/PWF/OSF/RNF；Hydraulic 10 条覆盖 cooler/valve/internal pump leakage/accumulator/stable flag。
- `gold_evidence_chunks.jsonl` 保存 10 个证据 chunk、官方 URL、页面定位和原子事实；`gold_case_audit.jsonl` 保存每个答案点到 fact ID 的映射。
- `gold_cases.jsonl` 中不再包含原候选的维修动作、未公开根因或通用安全建议。
- 当前 `second_review_status=pending`；另一个 agent 独立复核通过后再更新二审状态。
- 证据文档仍处于 `gold_evidence_ready_for_import`，导入知识库并生成新环境快照之前不能直接用于阶段 8.5.4 检索评测。
