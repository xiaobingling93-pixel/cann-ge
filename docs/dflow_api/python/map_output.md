# map\_output

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

给FlowNode映射输出，表示将pp的第pp\_output\_index个输出给到FlowNode的第node\_output\_index个输出，返回映射好的FlowNode节点。

可选调方法，不调用会默认按顺序去映射FlowNode和pp的输出。

## 函数原型

```
map_output(node_output_index, pp, pp_output_index)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| node_output_index | int | FlowNode输出index。 |
| pp | Union[GraphProcessPoint, FuncProcessPoint] | FlowNode节点映射的pp。可以是[GraphProcessPoint](dataflow-GraphProcessPoint.md)或者[FuncProcessPoint](dataflow-FuncProcessPoint.md)。 |
| pp_output_index | int | pp的输出index。 |

## 返回值

正常场景下返回None。

异常情况如下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
flow_node = df.FlowNode(input_num=2, output_num=2) 
flow_node.add_process_point(pp)
flow_node.map_input(0, pp, 0) 
flow_node.map_input(1, pp, 1) 
flow_node.map_output(0, pp, 1)  
flow_node.map_output(1, pp, 0) 
# 构建连边关系 
flow_node_out = flow_node(data, data1)
```

## 约束说明

无
