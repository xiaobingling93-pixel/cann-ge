# LLMDataDist构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

构造LLMDataDist。

## 函数原型

```
__init__(role: LLMRole, cluster_id: int)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| role | [LLMRole](LLMRole.md) | 集群角色。取值如下。<br><br>  - LLMRole.DECODER：增量集群，只能作为Client使用<br>  - LLMRole.PROMPT：全量集群，只能作为Server使用<br>  - LLMRole.MIX：混合部署 |
| cluster_id | int | 集群ID。LLMDataDist标识，在所有参与建链的范围内需要确保唯一。 |

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
```

## 返回值

正常情况下返回LLMDataDist的实例。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
