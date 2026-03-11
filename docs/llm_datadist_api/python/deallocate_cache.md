# deallocate\_cache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

释放Cache。

如果该Cache在Allocate时关联了CacheKey，则实际的释放会延后到所有的CacheKey被拉取或执行了[remove\_cache\_key](remove_cache_key.md)。

## 函数原型

```
deallocate_cache(cache: KvCache)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache | [KvCache](KvCache构造函数.md) | 要释放的KV Cache。 |

## 调用示例

```
kv_cache_manager.deallocate_cache(kv_cache)
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

- 如果KvCache不存在或已释放，该操作为空操作。
- 本接口不支持并发调用。
