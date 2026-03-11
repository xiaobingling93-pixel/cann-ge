# to\_flow\_msg

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将dataflow Tensor转换成FlowMsg。

## 函数原型

```
to_flow_msg(self, tensor)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | 待转换的dataflow Tensor。 |

## 返回值

正常返回FlowMsg的实例。失败返回None。

## 异常处理

无

## 约束说明

无
