# set\_contains\_n\_mapping\_node

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowGraph是否包含n\_mapping节点。n\_mapping节点指存在以下任意一种情况。

- 一次输入生成非一次（多个，0个，或不定）输出
- 多次输入生成一次输出
- 多次输入生成非一次（多个，0个，或不定）输出

## 函数原型

```
set_contains_n_mapping_node(self, contains_n_mapping_node: bool)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| contains_n_mapping_node | bool | 是否包含n_mapping节点。<br>图中只要有一个n_mapping处理节点，则设置为True，默认为False。 |

## 返回值

无。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
graph.set_contains_n_mapping_node(True)
```

## 约束说明

无
