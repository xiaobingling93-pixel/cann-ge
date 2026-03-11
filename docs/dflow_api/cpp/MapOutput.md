# MapOutput

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

给FlowNode映射输出，表示将ProcessPoint的第pp\_output\_index个输出给到FlowNode的第node\_output\_index个输出，返回映射好的FlowNode节点。

可选调用方法，不调用会默认按顺序去映射FlowNode和ProcessPoint的输出。

## 函数原型

```
FlowNode &MapOutput(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| node_output_index | 输入 | FlowNode输出index。 |
| pp | 输入 | FlowNode节点映射的ProcessPoint。 |
| pp_output_index | 输入 | ProcessPoint的输出index。 |

## 返回值

返回映射好输出的FlowNode节点。

## 异常处理

无。

## 约束说明

需要先调用[FlowNode &AddPp](AddPp.md)接口，才可以调用该接口。
