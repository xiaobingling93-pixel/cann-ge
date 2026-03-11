# pull\_cache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

根据CacheKey，从对应的Prompt节点拉取KV到本地KV Cache，仅当LLMRole为DECODER时可调用。

## 函数原型

```
pull_cache(cache_key: Union[CacheKey, CacheKeyByIdAndIndex], kv_cache: KvCache, batch_index: int = 0, size: int = -1, **kwargs)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache_key | Union[[CacheKey](CacheKey.md), [CacheKeyByIdAndIndex](CacheKeyByIdAndIndex.md)] | 需要被拉取的CacheKey。该CacheKey需要和[allocate_cache](allocate_cache.md)的CacheKey保持一致。<br>通过req_id，prefix_id，model_id拉取则传入CacheKey。<br>通过cache_id，batch_index拉取则传入CacheKeyByIdAndIndex。 |
| kv_cache | [KvCache](KvCache构造函数.md) | 目标KV Cache。 |
| batch_index | int | 表示目标KV Cache的batch index，默认为0。 |
| size | int | 默认为-1。<br>设置为>0的整数，表示要拉取的tensor大小。<br>或设置为-1，表示完整拷贝：本地单个KV的大小减去dst_cache_offset大小。 |
| **kwargs | NA | 这个是Python函数的可扩展参数通用写法，一般通过key=value的方式直接传入参数。<br>可选参数的详细信息请参考[表1](#table37855712016)。 |

**表 1**  \*\*kwargs的可选参数

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| src_layer_range | Optional[range] | 可选参数，用于按层pull kv场景。传输源的layer的范围，step只支持1。不设置时为传输所有layer。需要注意这里是layer的index，而不是tensor的index，即1个layer对应连续N个tensor(K/V)，这里要求分配内存时，必须是KV,...,KV排布，不支持其他场景。N为tensor_num_per_layer的取值，默认为2。 |
| dst_layer_range | Optional[range] | 可选参数，用于按层pull kv场景。传输目标的layer的范围，step只支持1。不设置时为传输所有layer。需要注意这里是layer的index，而不是tensor的index，即1个layer对应连续N个tensor(K/V)，这里要求分配内存时，必须是KV,...,KV排布，不支持其他场景。N为tensor_num_per_layer的取值，默认为2。 |
| src_cache_offset | Optional[int] | 设置>=0的整数。表示从src_cache tensor的offset位置拉取size大小的数据 |
| dst_cache_offset | Optional[int] | 设置>=0的整数。表示将源数据拉取到dst_cache tensor的offset起始位置 |
| tensor_num_per_layer | Optional[int] | 可选参数，表示每层的tensor的数量，默认值是2，取值范围是[1,cache的tensor总数]。当src_layer_range或dst_layer_range取值为非默认值时， tensor_num_per_layer可以保持默认值，也可以输入其他值，输入其他值的时，tensor_num_per_layer的取值还需要被当前cache的tensor总数整除。 |

## 调用示例

```
from llm_datadist import *
...
cache_keys = [CacheKey(1, req_id=1), CacheKey(1, req_id=2)]
kv_cache_manager.pull_cache(cache_keys[0], cache, 0)
# 使能layer_range功能
kv_cache_manager.pull_cache(cache_keys[1], cache, 1, src_layer_range=range(0,2), dst_layer_range=range(2,4))
# 使能offset功能
kv_cache_manager.pull_cache(cache_keys[1], cache, src_cache_offset=0, dst_cache_offset=0)
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

- 使用同一条链路时，不支持该接口和[transfer\_cache\_async](transfer_cache_async.md)接口并发。
- 本接口不支持并发调用。
