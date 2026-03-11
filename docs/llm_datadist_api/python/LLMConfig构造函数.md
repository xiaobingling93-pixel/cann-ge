# LLMConfig构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

构造LLMConfig，调用[init](init.md)接口需要传入一个配置项字典，为了简化配置，可以通过此类来构造该配置项字典。

## 函数原型

```
__init__()
```

## 参数说明

无

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
```

## 返回值

返回LLMConfig的实例。

## 约束说明

无
