# dataflow.finalize

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

释放dataflow初始化的资源。

## 函数原型

```
finalize()
```

## 参数说明

无

## 返回值

无

## 调用示例

```
import dataflow as df
# 初始化
df.init(...)
# dataflow处理逻辑
# 释放资源
df.finalize()
```

## 约束说明

无
