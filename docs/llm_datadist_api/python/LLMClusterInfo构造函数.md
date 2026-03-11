# LLMClusterInfo构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

构造LLMClusterInfo，用于[link\_clusters](link_clusters.md)和[unlink\_clusters](unlink_clusters.md)接口的参数类型。

## 函数原型

```
__init__()
```

## 参数说明

无

## 调用示例

```
from llm_datadist import LLMClusterInfo
llm_cluster = LLMClusterInfo()
```

## 返回值

返回LLMClusterInfo的实例。

## 约束说明

无
