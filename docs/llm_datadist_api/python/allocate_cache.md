# allocate\_cache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

分配Cache，分配成功后，会同时被cache\_id与cache\_keys引用，只有当这些引用都解除后，cache所占用的资源才会实际释放。

cache\_id的引用需通过[deallocate\_cache](deallocate_cache.md)解除，cache\_keys的引用则可以通过以下2种方式解除。

- Decoder调用[pull\_cache](pull_cache.md)接口成功后解除。
- Prompt调用[remove\_cache\_key](remove_cache_key.md)接口时解除。

## 函数原型

```
allocate_cache(cache_desc: CacheDesc, cache_keys: Union[Tuple[CacheKey], List[CacheKey]] = ())
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache_desc | [CacheDesc](CacheDesc.md) | Cache的描述。 |
| cache_keys | Union[Tuple[[CacheKey](CacheKey.md)], List[[CacheKey](CacheKey.md)]] | 仅当LLMRole为PROMPT时可设置，用于在Decode拉取KV。 |

## 调用示例

```
from llm_datadist import *
...
kv_cache_manager = data_dist.kv_cache_manager
cache_desc = CacheDesc(80, [2, 2 * 1024 * 1024], DataType.DT_FLOAT16)
cache_keys = [CacheKey(prompt_cluster_id=0, req_id=1)]
kv_cache = kv_cache_manager.allocate_cache(cache_desc, cache_keys)
```

## 返回值

正常情况下返回KvCache。

参数错误可能抛出TypeError或ValueError。

如果cache\_keys中包含了分配内存时绑定的CacheKey，则抛出[LLMException](LLMException.md)异常。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

- 传入cache\_keys时，如果Cache的batch size\>1，则需要提供相同数量的CacheKey，分别引用一组kv tensor。
- 如果当次推理的batch未占用满，即存在无效batch\_index，则需要插入特殊的CacheKey（将req\_id设置为UINT64\_MAX）占位，如果空闲的batch\_index在末尾，则可以省略。
- 如果cache\_keys存在重复，则最后一个生效。
- 本接口不支持并发调用。
