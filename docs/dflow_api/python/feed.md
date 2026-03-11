# feed

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将数据输入到Graph，支持可序列化的任意的输入。

## 函数原型

```
feed(feed_dict, timeout=-1, partial_inputs = False) -> int
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| feed_dict | Dict[FlowData, Any] | key为FlowData节点，value可以是可序列化的任意的输入。<br>输入与npu模型相连接时必须是dataflow的Tensor类型。 |
| timeout | int | 数据输入超时时间，单位：ms，取值范围[0, 2147483647), 取值为-1时表示从不超时。 |
| partial_inputs | bool | 每次调用feed接口时，feed_dict是否支持模型的部分输入，取值如下。<br><br>  - True：feed_dict中可以只包含模型的部分输入。<br>  - False：feed_dict中必须包含模型所有的输入。<br><br>默认为False。 |

## 返回值

正常场景下返回0。

异常场景下返回具体的错误码，并打印错误日志。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
graph.feed(...)
```

## 约束说明

使用非dataflow Tensor类型作为输入时需配合装饰器@pyflow以及装饰器@method进行使用。
