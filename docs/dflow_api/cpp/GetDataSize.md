# GetDataSize

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取Tensor中的数据大小。

## 函数原型

```
uint64_t *GetDataSize() const
```

## 参数说明

无。

## 返回值

返回Tensor的数据大小。

## 异常处理

无。

## 约束说明

修改返回的TensorDesc信息，不影响Tensor对象中已有的TensorDesc信息。
