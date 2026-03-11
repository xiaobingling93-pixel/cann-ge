# MapInput

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

给FlowNode映射输入，表示将FlowNode的第node\_input\_index个输入给到ProcessPoint的第pp\_input\_index个输入，并且给ProcessPoint的该输入设置上attrs里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序映射FlowNode和ProcessPoint的输入。

## 函数原型

```
FlowNode &MapInput(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index, const std::vector<DataFlowInputAttr> &attrs = {})
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| node_input_index | 输入 | FlowNode输入index。 |
| pp | 输入 | FlowNode节点映射的ProcessPoint。 |
| pp_input_index | 输入 | ProcessPoint的输入index。 |
| attrs | 输入 | 属性集。属性取值请参考[DataFlowInputAttr结构体](DataFlowInputAttr结构体.md)。 |

## 返回值

返回已经映射输入的FlowNode节点。

## 异常处理

无。

## 约束说明

需要先调用[FlowNode &AddPp](AddPp.md)接口，才可以调用该接口。
