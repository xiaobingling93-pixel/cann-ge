# dataflow.Framework

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置原始网络模型的框架类型。

## 函数原型

不涉及

## 参数说明

枚举值如下：

- Framework.TENSORFLOW
- Framework.ONNX
- Framework.MINDSPORE

## 返回值

无

## 调用示例

```
import dataflow as df
framework = df.Framework.TENSORFLOW
pp1 = df.GraphProcessPoint(framework,...)
```

## 约束说明

无
