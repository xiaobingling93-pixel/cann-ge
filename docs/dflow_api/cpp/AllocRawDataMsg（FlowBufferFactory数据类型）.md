# AllocRawDataMsg（FlowBufferFactory数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据输入的size申请一块连续内存，用于承载raw data类型的数据。

## 函数原型

```
FlowMsgPtr AllocRawDataMsg(size_t size, uint32_t align = 512U)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| size | 输入 | 申请内存大小。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】。<br>当前为预留参数，不进行参数值校验。 |

## 返回值

申请的FlowMsg指针。

## 异常处理

申请不到FlowMsg指针则返回NULL。

## 约束说明

无。
