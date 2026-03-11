# dataflow.FlowGraphProcessPoint

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

FlowGraphProcessPoint构造函数，返回一个FlowGraphProcessPoint对象。

## 函数原型

```
FlowGraphProcessPoint(flow_graph, compile_config_path="", name=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| flow_graph | FlowGraph | dataflow.FlowGraph子图。 |
| compile_config_path | str | 编译FlowGraph时的配置文件路径，缺省不设。 |
| name | str | 处理点名称，框架会自动保证名称唯一，不设置时会自动生成FlowGraphProcessPoint, FlowGraphProcessPoint_1, FlowGraphProcessPoint_2,...的名称。 |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
pp1 = df.FlowGraphProcessPoint(...)
```

## 约束说明

无
