# AI4I TWF rule

来源：AI4I 2020 Predictive Maintenance Dataset
来源链接：https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
来源位置：Additional Variable Information > tool wear failure (TWF)
许可证：CC BY 4.0

## 已核实事实

- TWF 的触发时点是在 200 至 240 分钟之间随机选定的刀具磨损时间。
- 到达该时点时，数据生成过程会让刀具被更换或发生失效。
- 数据集中共有 120 个这样的时点，其中 69 次更换刀具、51 次标为失效。
- 更换还是失效是随机分配的。

# AI4I HDF rule

来源：AI4I 2020 Predictive Maintenance Dataset
来源链接：https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
来源位置：Additional Variable Information > heat dissipation failure (HDF)
许可证：CC BY 4.0

## 已核实事实

- HDF 要求工艺温度与空气温度的差值低于 8.6 K。
- HDF 同时要求转速低于 1380 rpm。
- 两个条件必须同时满足才按该规则发生散热失效。
- AI4I 2020 数据集中有 115 个数据点满足 HDF 规则。

# AI4I PWF rule

来源：AI4I 2020 Predictive Maintenance Dataset
来源链接：https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
来源位置：Additional Variable Information > power failure (PWF)
许可证：CC BY 4.0

## 已核实事实

- PWF 使用转矩与以 rad/s 表示的转速相乘，得到过程所需功率。
- 计算功率低于 3500 W 时，过程按 PWF 规则失效。
- 计算功率高于 9000 W 时，过程按 PWF 规则失效。
- AI4I 2020 数据集中有 95 个数据点满足 PWF 规则。

# AI4I OSF rule

来源：AI4I 2020 Predictive Maintenance Dataset
来源链接：https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
来源位置：Additional Variable Information > overstrain failure (OSF)
许可证：CC BY 4.0

## 已核实事实

- OSF 比较刀具磨损时间与转矩的乘积。
- L 产品类型的 OSF 阈值为 11000 minNm。
- M 产品类型的 OSF 阈值为 12000 minNm。
- H 产品类型的 OSF 阈值为 13000 minNm。
- AI4I 2020 数据集中有 98 个数据点满足 OSF 规则。

# AI4I RNF rule

来源：AI4I 2020 Predictive Maintenance Dataset
来源链接：https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
来源位置：Additional Variable Information > random failures (RNF)
许可证：CC BY 4.0

## 已核实事实

- AI4I 2020 中每个过程有 0.1% 的概率发生 RNF。
- RNF 与该过程的工艺参数无关。
- 数据集中实际有 5 个 RNF 数据点。
- 五种失效模式中任意一种为真时，Machine failure 标签都会设为 1。
