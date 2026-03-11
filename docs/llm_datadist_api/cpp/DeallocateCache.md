# DeallocateCache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

释放Cache。

## 函数原型

```
Status DeallocateCache(int64_t cache_id)
```

## 参数说明

| 参数名称 | 输入/输出 | 取值说明 |
| --- | --- | --- |
| cache_id | 输入 | Cache的ID。 |

## 调用示例

```
Status ret = llm_datadist.DeallocateCache(cache.cache_id)
```

## 返回值

- LLM\_SUCCESS：成功
- 其他：失败

## 约束说明

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。
