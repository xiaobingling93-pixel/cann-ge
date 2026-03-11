# PullKvCache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

从远端节点拉取Cache到本地Cache，仅当角色为Decoder时可调用。

## 函数原型

```
Status PullKvCache(const CacheIndex &src_cache_index,
                   const Cache &dst_cache,
                   uint32_t batch_index = 0U,
                   int64_t size = -1,
                   const KvCacheExtParam &ext_param = {})
```

## 参数说明

| 参数名称 | 输入/输出 | 取值说明 |
| --- | --- | --- |
| src_cache_index | 输入 | 远端源Cache的索引。 |
| dst_cache | 输入 | 本地目的Cache。 |
| batch_index | 输入 | 本地目的batch的下标。 |
| size | 输入 | 设置为>0的整数，表示要拉取的大小。<br>或设置为-1，表示完整拉取。<br>默认为-1。 |
| ext_param | 输入 | 当前支持ext_param中src_layer_range的sencond与first的差值和dst_layer_range的sencond与first的差值一致。src_layer_range和dst_layer_range的first和second默认值都是-1，表示全部的层。取值范围都是[0, 最大可用层索引]，且first小于等于second。 最大可用层索引值的计算公式如下。<br>(CacheDesc::num_tensors / KvCacheExtParam::tensor_num_per_layer) - 1<br>当前支持tensor_num_per_layer取值范围是[1, 当前cache的tensor总数]，默认值为2。当src_layer_range或dst_layer_range取值为非默认值时， tensor_num_per_layer可以保持默认值，也可以输入其他值，输入其他值的时，tensor_num_per_layer的取值还需要被当前cache的tensor总数整除。 |

## 调用示例

```
CacheIndex cache_index;
cache_index.cluster_id = 0;
cache_index.cache_id = cached_tensors.cache_id;
cache_index.batch_index = 0;
Status ret = llm_datadist.PullKvCache(cache_index, cache)
```

## 返回值

- LLM\_SUCCESS：成功
- LLM\_PARAM\_INVALID：参数错误
- LLM\_NOT\_YET\_LINK：与远端cluster没有建链
- LLM\_TIMEOUT：拉取超时
- LLM\_KV\_CACHE\_NOT\_EXIST：本地或远端KV Cache不存在
- 其他：失败

## 约束说明

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。dst\_cache必须为[AllocateCache](AllocateCache.md)接口申请出的Cache。
