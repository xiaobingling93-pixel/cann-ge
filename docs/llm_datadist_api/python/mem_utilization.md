# mem\_utilization

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

配置ge.flowGraphMemMaxSize内存的利用率。默认值0.95。

## 函数原型

```
mem_utilization(mem_utilization)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| mem_utilization | float | 内存利用率。默认值0.95。取值范围0.0~1.0。 |

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
llm_config.mem_utilization = 0.95
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无。
