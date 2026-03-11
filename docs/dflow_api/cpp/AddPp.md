# AddPp

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

给FlowNode添加映射的ProcessPoint，当前一个FlowNode仅能添加一个ProcessPoint，添加后会默认将FlowNode的输入输出和ProcessPoint的输入输出按顺序进行映射。

## 函数原型

```
FlowNode &AddPp(const ProcessPoint &pp)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pp | 输入 | FlowNode节点映射的ProcessPoint。ProcessPoint可以是[FunctionPp类](FunctionPp类.md)或者[ProcessPoint类](ProcessPoint类.md)。 |

## 返回值

返回一个FlowNode节点。

## 异常处理

无。

## 约束说明

被添加的ProcessPoint的输入/输出个数需要和FlowNode的输入/输出个数保持一致。
