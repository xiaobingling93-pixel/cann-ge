# feed\_data

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将数据输入到Graph。

## 函数原型

```
feed_data(feed_dict, flow_info=None, timeout=-1, partial_inputs = False) -> int
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| feed_dict | Dict[FlowData, Union["numpy.ndarray", Tensor, List]] | key为FlowData节点，value是可转换成numpy.ndarray的任意输入，或者dataflow.tensor。<br>当feed_dict为空时，flow_info必须的flow_flags必须包含DATA_FLOW_FLAG_EOS，此时partial_inputs不起作用。 |
| flow_info | FlowInfo | 按需设置FlowInfo，具体请参见[dataflow.FlowInfo](dataflow-FlowInfo.md)。 |
| timeout | int | 数据输入超时时间，单位：ms，取值范围[0, 2147483647), 取值为-1时表示从不超时。 |
| partial_inputs | bool | 每次调用feed_data接口时，feed_dict是否支持模型的部分输入，取值如下。<br><br>  - True：feed_dict中可以只包含模型的部分输入。<br>  - False：feed_dict中必须包含模型所有的输入。<br><br>默认为False。 |

## 返回值

正常场景下返回0。

异常场景下返回具体的错误码，并打印错误日志。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
graph.feed_data(...)
```

## 约束说明

无
