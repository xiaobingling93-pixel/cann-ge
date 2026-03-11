# add\_invoked\_closure

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

添加FuncProcessPoint调用的GraphProcessPoint或FlowGraphProcessPoint，返回添加好的FuncProcessPoint。

## 函数原型

```
add_invoked_closure(graph_key, graph_pp)
add_invoked_closure(graph_key, flow_graph_pp)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| graph_key | str | 调用的GraphProcessPoint的唯一标识，需要全图唯一。 |
| graph_pp | [GraphProcessPoint](dataflow-GraphProcessPoint.md) | 调用的GraphProcessPoint，具体请参见[dataflow.GraphProcessPoint](dataflow-GraphProcessPoint.md)。 |
| flow_graph_pp | [FlowGraphProcessPoint](dataflow-FlowGraphProcessPoint.md) | 调用的FlowGraphProcessPoint，具体请参见[dataflow.FlowGraphProcessPoint](dataflow-FlowGraphProcessPoint.md)。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
pp1 = df.GraphProcessPoint(...)
pp.add_invoked_closure(graph_key, pp1)
```

## 约束说明

无
