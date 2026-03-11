# AllocEmptyDataMsg（FlowBufferFactory数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

申请空数据的MsgType类型的message。

## 函数原型

```
FlowMsgPtr AllocEmptyDataMsg(MsgType type)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| type | 输入 | 要申请空数据的消息类型。 |

## 返回值

申请的FlowMsg指针。

## 异常处理

申请不到FlowMsg指针则返回NULL。

## 约束说明

无。
