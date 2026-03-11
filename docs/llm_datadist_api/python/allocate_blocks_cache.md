# allocate\_blocks\_cache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

PagedAttention场景下，分配多个blocks的Cache，Cache分配成功后，可通过[deallocate\_cache](deallocate_cache.md)释放内存。

## 函数原型

```
allocate_blocks_cache(cache_desc: CacheDesc, blocks_cache_key: Optional[BlocksCacheKey] = None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache_desc | [CacheDesc](CacheDesc.md) | Cache的描述。 |
| blocks_cache_key | Optional[[BlocksCacheKey](BlocksCacheKey.md)] | 仅当LLMRole为PROMPT时可设置，用于在Decode拉取KV。 |

## 调用示例

```
from llm_datadist import BlocksCacheKey
num_blocks = 1000
cache_desc = CacheDesc(80, [num_blocks , 128 * 1024], DataType.DT_FLOAT16)
blocks_cache_key = BlocksCacheKey(0, 0)
kv_cache = kv_cache_manager.allocate_blocks_cache(cache_desc, blocks_cache_key)
```

## 返回值

正常情况下返回KvCache。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

本接口不支持并发调用。
