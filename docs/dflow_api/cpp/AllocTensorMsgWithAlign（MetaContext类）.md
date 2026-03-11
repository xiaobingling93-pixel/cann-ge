# AllocTensorMsgWithAlign（MetaContext类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据shape、data type和对齐大小申请Tensor类型的FlowMsg，与[AllocTensorMsg](AllocTensorMsg（MetaContext类）.md)函数区别是AllocTensorMsg默认申请以64字节对齐，此函数可以指定对齐大小，方便性能调优。

## 函数原型

```
std::shared_ptr<FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape, TensorDataType dataType, uint32_t align)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Tensor的shape。 |
| dataType | 输入 | Tensor的dataType。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】。 |

## 返回值

申请的Tensor指针。

## 异常处理

申请不到Tensor指针则返回NULL。

## 约束说明

无。
