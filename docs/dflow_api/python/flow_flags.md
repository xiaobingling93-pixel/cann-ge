# flow\_flags

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

以属性方法读取和设置FlowInfo的flow\_flags。

## 函数原型

```
@property
def flow_flags(self)
@flow_flags.setter
def flow_flags(self, new_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| new_value | int | 重新设置的flow_flags的新值。 |

## 返回值

flow\_flags属性。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
flowinfo = FlowInfo(...)
flowinfo.flow_flags = df.FlowFlag.DATA_FLOW_FLAG_EOS | df.FlowFlag.DATA_FLOW_FLAG_SEG
print(flowinfo.flow_flags)
```

## 约束说明

无
