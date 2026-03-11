# map\_input

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

给FlowNode映射输入，表示将FlowNode的第node\_input\_index个输入给到ProcessPoint的第pp\_input\_index个输入，并且给ProcessPoint的该输入设置上attr里的所有属性，返回映射好的FlowNode节点。该函数可选，不被调用时会默认按顺序去映射FlowNode和ProcessPoint的输入。

## 函数原型

```
map_input(node_input_index, pp, pp_input_index, input_attrs=[])
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| node_input_index | int | FlowNode节点输入index，小于等于输入个数。 |
| pp | Union[GraphProcessPoint, FuncProcessPoint] | FlowNode节点映射的pp。可以是[GraphProcessPoint](dataflow-GraphProcessPoint.md)或者[FuncProcessPoint](dataflow-FuncProcessPoint.md)。 |
| pp_input_index | int | pp的输入index。 |
| input_attrs | List[Union[TimeBatch, CountBatch]] | 属性集，当前支持[TimeBatch](dataflow-TimeBatch.md)和[CountBatch](dataflow-CountBatch.md)。 |

## 返回值

正常场景下返回None。

异常情况如下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
flow_node = df.FlowNode(input_num=2, output_num=1)
flow_node.add_process_point(pp)
flow_node.map_input(0, pp, 1)
flow_node.map_input(1, pp, 0)
flow_node_out = flow_node(data0, data1)
```

## 约束说明

无
