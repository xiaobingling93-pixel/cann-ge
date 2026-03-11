# remove\_cache\_key

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

移除CacheKey，仅当LLMRole为PROMPT时可调用。

移除CacheKey后，该Cache将无法再被[pull\_cache](pull_cache.md)拉取。

## 函数原型

```
remove_cache_key(cache_key: CacheKey)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache_key | [CacheKey](CacheKey.md) | 需要被移除的CacheKey。 |

## 调用示例

```
cache_keys = [CacheKey(prompt_cluster_id=0, req_id=1)]
kv_cache_manager.remove_cache_key(cache_keys[0])
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

- 如果CacheKey不存在或已移除，该操作为空操作。
- 本接口不支持并发调用。
