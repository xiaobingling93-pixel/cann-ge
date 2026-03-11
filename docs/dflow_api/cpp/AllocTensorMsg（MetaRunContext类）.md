# AllocTensorMsg（MetaRunContext类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据shape和data type申请Tensor类型的msg。该函数供[Proc](Proc.md)调用。

## 函数原型

```
std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType dataType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| Shape | 输入 | Tensor的Shape。 |
| dataType | 输入 | Tensor的dataType。 |

## 返回值

申请的Tensor指针。

## 异常处理

申请不到Tensor指针则返回NULL。

## 约束说明

无。
