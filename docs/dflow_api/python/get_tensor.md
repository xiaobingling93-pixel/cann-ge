# get\_tensor

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取FlowMsg中的tensor对象。

## 函数原型

```
get_tensor() -> dataflow.Tensor
```

## 参数说明

无

## 返回值

返回dataflow.Tensor类型对象。

## 异常处理

无

## 约束说明

如果FlowMsg中是空，则tensor返回None。
