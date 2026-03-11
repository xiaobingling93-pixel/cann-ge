# AllocTensor（FlowBufferFactory数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据shape、data type和对齐大小申请Tensor。

## 函数原型

```
std::shared_ptr<Tensor> AllocTensor(const std::vector<int64_t> &shape, DataType data_type, uint32_t align = 512U)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Tensor的shape。 |
| data_type | 输入 | Tensor的DataType 。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】，当前为预留参数。 |

## 返回值

申请的Tensor指针。

## 异常处理

申请不到Tensor指针则返回NULL。

## 约束说明

无。
