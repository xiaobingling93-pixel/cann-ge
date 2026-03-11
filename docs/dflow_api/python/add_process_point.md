# add\_process\_point

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

给FlowNode添加映射的pp，当前一个FlowNode仅能添加一个pp，添加后会默认将FlowNode的输入输出和pp的输入输出按顺序进行映射。

## 函数原型

```
add_process_point(pp)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| pp | Union[GraphProcessPoint, FuncProcessPoint] | FlowNode节点映射的处理点，可以是[GraphProcessPoint](dataflow-GraphProcessPoint.md)或者[FuncProcessPoint](dataflow-FuncProcessPoint.md)。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
flow_node = df.FlowNode(...)
flow_node.add_process_point(pp)
```

## 约束说明

被添加的PP的输入输出个数与FlowNode输入输出一致。
