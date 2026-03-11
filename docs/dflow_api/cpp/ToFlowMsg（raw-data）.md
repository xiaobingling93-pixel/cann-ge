# ToFlowMsg（raw data）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据输入的raw data转换成用于承载raw data的FlowMsg。

## 函数原型

```
FlowMsgPtr ToFlowMsg(const RawData &raw_data)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| raw_data | 输入 | RawData对象。 |

## 返回值

转换的FlowMsg指针。

## 异常处理

转换失败则返回NULL。

## 约束说明

无。
