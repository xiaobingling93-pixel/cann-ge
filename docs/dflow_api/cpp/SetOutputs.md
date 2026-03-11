# SetOutputs

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowGraph的输出节点，并返回该图。

## 函数原型

```
FlowGraph &SetOutputs(const std::vector<FlowOperator> &outputs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outputs | 输出 | 输出的节点集。 |

## 返回值

返回设置了输出节点的FlowGraph图。

## 异常处理

无。

## 约束说明

无。
