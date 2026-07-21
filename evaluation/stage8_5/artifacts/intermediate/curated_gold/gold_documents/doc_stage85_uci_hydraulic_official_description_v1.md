# Hydraulic profile cooler labels

来源：Condition Monitoring of Hydraulic Systems
来源链接：https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
来源位置：Attribute Information > profile.txt column 1: Cooler condition / %
许可证：CC BY 4.0

## 已核实事实

- profile.txt 第 1 列表示冷却器状态，数值单位为百分比。
- 冷却器标签 3 表示接近完全失效。
- 冷却器标签 20 表示效率降低。
- 冷却器标签 100 表示完全有效。

# Hydraulic profile valve labels

来源：Condition Monitoring of Hydraulic Systems
来源链接：https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
来源位置：Attribute Information > profile.txt column 2: Valve condition / %
许可证：CC BY 4.0

## 已核实事实

- profile.txt 第 2 列表示阀门状态，数值单位为百分比。
- 阀门标签 100 表示最佳切换行为。
- 阀门标签 90 表示轻微延迟。
- 阀门标签 80 表示严重延迟。
- 阀门标签 73 表示接近完全失效。

# Hydraulic profile internal pump leakage labels

来源：Condition Monitoring of Hydraulic Systems
来源链接：https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
来源位置：Attribute Information > profile.txt column 3: Internal pump leakage
许可证：CC BY 4.0

## 已核实事实

- profile.txt 第 3 列表示泵内部泄漏状态。
- 泵内部泄漏标签 0 表示无泄漏。
- 泵内部泄漏标签 1 表示轻微泄漏。
- 泵内部泄漏标签 2 表示严重泄漏。

# Hydraulic profile accumulator labels

来源：Condition Monitoring of Hydraulic Systems
来源链接：https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
来源位置：Attribute Information > profile.txt column 4: Hydraulic accumulator / bar
许可证：CC BY 4.0

## 已核实事实

- profile.txt 第 4 列表示液压蓄能器压力，单位为 bar。
- 蓄能器标签 130 表示最佳压力。
- 蓄能器标签 115 表示压力略有降低。
- 蓄能器标签 100 表示压力严重降低。
- 蓄能器标签 90 表示接近完全失效。

# Hydraulic profile stable flag labels

来源：Condition Monitoring of Hydraulic Systems
来源链接：https://archive.ics.uci.edu/dataset/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems
来源位置：Attribute Information > profile.txt column 5: stable flag
许可证：CC BY 4.0

## 已核实事实

- profile.txt 第 5 列是 stable flag，用于描述该周期是否达到稳定状态。
- stable flag=0 表示条件稳定。
- stable flag=1 表示可能尚未达到静态条件。
- stable flag 与前四列组件状态标签分列记录。
