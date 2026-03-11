# CopyKvCache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

拷贝KV Cache。支持D2D，D2H的拷贝。

当期望PullKvCache和其他使用Cache的操作流水时，可以额外申请一块中转Cache。当其他流程在使用Cache时，可以先将下一次的Cache pull到中转Cache，待其他流程使用完Cache后，拷贝到指定的位置，从而通过pipeline流水将PullKvCache的耗时隐藏，减少总耗时。

公共前缀场景在新请求推理前，可以将公共前缀拷贝到新的内存中与当前请求的KV合并推理。

## 函数原型

```
Status CopyKvCache(const Cache &src_cache,
                   const Cache &dst_cache,
                   uint32_t src_batch_index = 0U,
                   uint32_t dst_batch_index = 0U,
                   uint64_t offset = 0U,
                   int64_t size = -1)
```

## 参数说明

| 参数名称 | 输入/输出 | 取值说明 |
| --- | --- | --- |
| src_cache | 输入 | 源Cache。 |
| dst_cache | 输入 | 目的Cache。 |
| src_batch_index | 输入 | 源Cache的batch的下标。 |
| dst_batch_index | 输入 | 目的Cache的batch的下标。 |
| offset | 输入 | 拷贝偏移，单位为byte。 |
| size | 输入 | 设置为>0的整数，表示要拷贝的大小。<br>或设置为-1，表示完整拷贝。<br>默认为-1。 |

## 调用示例

```
Status ret = llm_datadist.CopyKvCache(src_cache, dst_cache, 0, 0)
```

## 返回值

- LLM\_SUCCESS：成功
- LLM\_PARAM\_INVALID：参数错误
- 其他：失败

## 约束说明

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。只支持Device-\>Device与Device-\>Host的拷贝。
