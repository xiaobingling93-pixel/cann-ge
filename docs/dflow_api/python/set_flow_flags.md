# set\_flow\_flags

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowMsg消息头中的flags。

## 函数原型

```
set_flow_flags(self, flow_flags: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flow_flags | 输入 | 消息头中的flags标志。<br>Flags可以取如下值：<br>dataflow.flow_func.flow_func.FLOW_FLAG_EOS  // 数据流结束标志<br>dataflow.flow_func.flow_func.FLOW_FLAG_SEG  // 非连续数据的分段标志 |

## 返回值

无

## 异常处理

无

## 约束说明

无
