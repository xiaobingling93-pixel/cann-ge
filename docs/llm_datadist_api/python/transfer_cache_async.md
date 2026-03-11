# transfer\_cache\_async

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

异步分层传输KV Cache。

## 函数原型

```
transfer_cache_async(src_cache: KvCache,
                     layer_synchronizer: LayerSynchronizer,
                     transfer_configs: Union[List[TransferConfig], Tuple[TransferConfig]],
                     src_block_indices: Optional[Union[List[int], Tuple[int]]] = None,
                     dst_block_indices: Optional[Union[List[int], Tuple[int]]] = None,
                     dst_block_memory_size: Optional[int] = None) -> CacheTask
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| src_cache | [KvCache](KvCache构造函数.md) | 源Cache。 |
| layer_synchronizer | [LayerSynchronizer](LayerSynchronizer.md) | LayerSynchronizer的实现类对象 |
| transfer_configs | Union[List[[TransferConfig](TransferConfig.md)], Tuple[[TransferConfig](TransferConfig.md)]] | 传输配置列表或元组 |
| src_block_indices | Optional[Union[List[int], Tuple[int]]] | 源Cache的block indices，当源Cache为PA场景时设置 |
| dst_block_indices | Optional[Union[List[int], Tuple[int]]] | 目的Cache的block indices，当目的Cache为PA场景时设置 |
| dst_block_memory_size | Optional[int] | 目的Cache每个block占用的内存大小，当目的Cache为PA场景时设置。如果源Cache也为PA场景，则可省略该参数，此时会自动将其设置为源Cache每个block占用的内存大小。<br>该参数设置为0时等同于省略该参数。 |

## 调用示例

```
from llm_datadist import *
...
class LayerSynchronizerImpl(LayerSynchronizer):
    def synchronize_layer(self, layer_index: int, timeout_in_millis: Optional[int]) -> bool:
        # need control time for transfer layer here.
        return True
num_layers = 40
dst_cluster_id = 2
# need register decoder kv addr here.
decoder_addrs = ...
assert(len(decoder_addrs) = 2*num_layers)
transfer_config = TransferConfig(dst_cluster_id, decoder_addrs, range(0, num_layers), 0)
cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(), [transfer_config])
cache_task.synchronize()
cache_task.get_results()
```

## 返回值

正常情况下返回CacheTask。

传入数据类型错误，会抛出TypeError或ValueError异常。

传入数据非法，会抛出[LLMException](LLMException.md)异常。

## 约束说明

- 当前仅支持src\_cache与dst\_cache都为连续cache的场景以及src\_cache与dst\_cache都为PA的场景。
- 使用同一条链路时，此接口和[pull\_cache](pull_cache.md)、[pull\_blocks](pull_blocks.md)接口不支持并发。
- 本接口不支持并发调用。
- 单进程多卡模式下，不支持调用该接口。
