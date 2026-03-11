# PushKvBlocks

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

通过配置block列表的方式，从本地节点推送Cache到远端节点，仅当角色为Prompt时可调用。

## 函数原型

```
Status PushKvBlocks(const Cache &src_cache,
                    const CacheIndex &dst_cache_index,
                    const std::vector<uint64_t> &src_blocks,
                    const std::vector<uint64_t> &dst_blocks,
                    const KvCacheExtParam &ext_param = {});
```

## 参数说明

| 参数名称 | 输入/输出 | 取值说明 |
| --- | --- | --- |
| src_cache | 输入 | 本地源Cache。 |
| dst_cache_index | 输入 | 远端目的Cache的索引。 |
| src_blocks | 输入 | 源Cache的block index列表。 |
| dst_blocks | 输入 | 目的Cache的block index列表。 |
| ext_param | 输入 | 当前支持ext_param中src_layer_range的sencond与first的差值和dst_layer_range的sencond与first的差值一致。src_layer_range和dst_layer_range的first和second默认值都是-1，表示全部的层。取值范围都是[0, 最大可用层索引]，且first小于等于second。 最大可用层索引值的计算公式如下。<br>(CacheDesc::num_tensors / KvCacheExtParam::tensor_num_per_layer) - 1<br>当前支持tensor_num_per_layer取值范围是[1, 当前cache的tensor总数]，默认值为2。当src_layer_range或dst_layer_range取值为非默认值时， tensor_num_per_layer可以保持默认值，也可以输入其他值，输入其他值的时，tensor_num_per_layer的取值还需要被当前cache的tensor总数整除。 |

## 调用示例

```
CacheDesc kv_desc{};
kv_desc.data_type = llm_datadist::DT_INT32;
kv_desc.shape = {4, 16};
kv_desc.num_tensors = 4;
Cache cache{};
llm_datadist.AllocateCache(kv_desc, cache);
CacheIndex dst_cache_key{0, 1};
KvCacheExtParam ext_param{};
ext_param.src_layer_range =  std::pair<int32_t, int32_t>(3, 3);
ext_param.dst_layer_range =  std::pair<int32_t, int32_t>(3, 3);
ext_param.tensor_num_per_layer = 1;
std::vector<uint64_t> prompt_blocks = {0, 1, 2, 3};
std::vector<uint64_t> decoder_blocks = {3, 2, 1, 0};
Status ret = llm_datadist.PushKvBlocks(cache, dst_cache_key, prompt_blocks, decoder_blocks , ext_param);
```

## 返回值

- LLM\_SUCCESS：成功
- LLM\_PARAM\_INVALID：参数错误
- LLM\_NOT\_YET\_LINK：与远端cluster没有建链
- LLM\_TIMEOUT：推送超时
- LLM\_KV\_CACHE\_NOT\_EXIST：本地或远端KV Cache不存在
- 其他：失败

## 约束说明

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。src\_cache必须为[AllocateCache](AllocateCache.md)接口申请出的Cache。
