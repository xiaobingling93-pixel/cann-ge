# FlowInfo

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

DataFlow的flow信息。

## 函数原型

```
FlowInfo(start_time=0, end_time=0, flow_flags=0, transaction_id=0)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| start_time | int | 开始时间，单位ms。 |
| end_time | int | 结束时间，单位ms。 |
| flow_flags | int | dataflow.FlowFlag，具体请参见[dataflow.FlowFlag](dataflow-FlowFlag.md)。 |
| transaction_id | int | DataFlow数据传输使用的事务ID，非0时有效。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
flow_info = FlowInfo(...)
graph.feed_data(...,flow_info)
```

## 约束说明

无
