# dataflow.FlowFlag

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowMsg消息头中的flags。

## 函数原型

不涉及

## 参数说明

枚举值如下：

- FlowFlag.DATA\_FLOW\_FLAG\_EOS
- FlowFlag.DATA\_FLOW\_FLAG\_SEG

## 返回值

无

## 调用示例

```
import dataflow as df
flow_info = df.FlowInfo()
flow_info.flow_flags = df.FlowFlag.DATA_FLOW_FLAG_EOS
```

## 约束说明

无
