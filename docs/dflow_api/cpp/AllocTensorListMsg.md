# AllocTensorListMsg

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据输入的dtype shapes数组分配一块连续内存，用于承载Tensor数组。

## 函数原型

```
virtual std::shared_ptr<FlowMsg> AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes, const std::vector<TensorDataType> &dataTypes, uint32_t align = 512U)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shapes | 输入 | Tensor List对应的shape列表。 |
| dataTypes | 输入 | Tensor List的dataType列表。 |
| align | 输入 | 申请内存地址对齐大小，取值范围 【32、64、128、256、512、1024】。 |

## 返回值

申请的FlowMsg指针。

## 异常处理

申请不到Tensor指针则返回NULL。

## 约束说明

无。
