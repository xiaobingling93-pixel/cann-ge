# LlmDataDist构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

创建LLM-DataDist对象。

## 函数原型

```
LlmDataDist(uint64_t cluster_id, LlmRole role)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cluster_id | 输入 | 集群ID。LlmDataDist标识，在所有参与建链的范围内需要确保唯一。 |
| role | 输入 | 类型是[LlmRole](LlmRole.md)，用于标识当前角色。 |

## 返回值

无

## 异常处理

无

## 约束说明

无
