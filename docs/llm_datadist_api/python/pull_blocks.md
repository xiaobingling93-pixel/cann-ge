# pull\_blocks

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

PagedAttention场景下，根据BlocksCacheKey，通过block列表的方式从对应的Prompt节点拉取KV到本地KV Cache，仅当LLMRole为DECODER时可调用。

## 函数原型

```
pull_blocks(prompt_cache_key: BlocksCacheKey, decoder_kv_cache: KvCache, prompt_blocks: List[int], decoder_blocks: List[int], **kwargs)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| prompt_cache_key | [BlocksCacheKey](BlocksCacheKey.md) | 需要被拉取的BlocksCacheKey。 |
| decoder_kv_cache | [KvCache](KvCache构造函数.md) | 目标KV Cache。 |
| prompt_blocks | List[int] | Prompt的block index列表。 |
| decoder_blocks | List[int] | Decode的block index列表。 |
| **kwargs | NA | 这个是Python函数的可扩展参数通用写法，一般通过key=value的方式直接传入参数。<br>可选参数的详细信息请参考[表1](#table37855712016)。 |

**表 1**  \*\*kwargs的可选参数

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| src_layer_range | Optional[range] | 可选参数，用于按层pull kv场景。传输源的layer的范围，step只支持1。不设置时为传输所有layer。需要注意这里是layer的index，而不是tensor的index，即1个layer对应连续N个tensor(K/V)，这里要求分配内存时，必须是KV,...,KV排布，不支持其他场景。N为tensor_num_per_layer的取值，默认为2。 |
| dst_layer_range | Optional[range] | 可选参数，用于按层pull kv场景。传输目标的layer的范围，step只支持1。不设置时为传输所有layer。需要注意这里是layer的index，而不是tensor的index，即1个layer对应连续N个tensor(K/V)，这里要求分配内存时，必须是KV,...,KV排布，不支持其他场景。N为tensor_num_per_layer的取值，默认为2。 |
| tensor_num_per_layer | Optional[int] | 可选参数，表示每层的tensor的数量，默认值是2，取值范围是[1,cache的tensor总数]。当src_layer_range或dst_layer_range取值为非默认值时， tensor_num_per_layer可以保持默认值，也可以输入其他值，输入其他值的时，tensor_num_per_layer的取值还需要被当前cache的tensor总数整除。 |

## 调用示例

```
from llm_datadist import *
...
kv_cache_manager.pull_blocks(prompt_cache_key, kv_cache, [0, 1], [2, 3])

# 使能layer_range功能示例
kv_cache_manager.pull_blocks(prompt_cache_key, kv_cache, [0, 1], [2, 3], src_layer_range=range(2), dst_layer_range=range(2))
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

- 使用同一条链路时，此接口和[transfer\_cache\_async](transfer_cache_async.md)接口不支持并发。
- 本接口不能被多线程并发调用。
