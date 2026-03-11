# FlowGraph

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

DataFlow的graph，由输入节点FlowData和计算节点FlowNode构成。

## 函数原型

```
FlowGraph(outputs, graph_options={}, name=None, graphpp_builder_async = False)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| outputs | List[FlowOutput] | 需要包含FlowGraph输出节点的所有输出，具体请参见[dataflow.FlowOutput](dataflow-FlowOutput.md)。 |
| graph_options | Dict[str, str] | FlowGraph的编译options，当前支持AddGraph接口传入的配置参数。<br>其中的配置示例请按照Python语言进行适配。 |
| name | str | 图名称，框架会自动保证名称唯一，不设置时会自动生成FlowGraph, FlowGra, FlowGra,...的名称。 |
| graphpp_builder_async | bool | FlowGraph中的GraphProcessPoint的构建器是否异步执行。取值如下：<br><br>  - True：是<br>  - False：否<br><br>默认值：False |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
```

## 约束说明

无
