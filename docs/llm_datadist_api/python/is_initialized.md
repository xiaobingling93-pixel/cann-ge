# is\_initialized

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

查询KvCacheManager实例是否已初始化。

## 函数原型

```
is_initialized() -> bool
```

## 参数说明

NA

## 调用示例

```
from llm_datadist import LLMDataDist
llm_datadist = LLMDataDist(LLMRole.Decode, 0)
...
llm_datadist.init(engine_options)
kv_cache_manager = llm_datadist.kv_cache_manager
is_initialized = kv_cache_manager.is_initialized()
```

## 返回值

如果LLMDataDist已初始化，则返回True。

如果LLMDataDist已销毁，则返回False。

## 约束说明

如果该接口返回False，则不应调用[KvCacheManager](KvCacheManager.md)的任何接口。
