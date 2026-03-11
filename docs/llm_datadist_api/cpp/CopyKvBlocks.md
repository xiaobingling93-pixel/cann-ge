# CopyKvBlocks

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

PA场景下，通过block列表的方式拷贝KV Cache。支持D2D，D2H，H2D的拷贝。

- D2D场景主要是针对当多个回答需要共用相同block，block没填满时，新增的token需要拷贝到新的block上继续迭代。
- H2D和D2H的拷贝主要用于对应block\_index上Cache内存的换入换出。

## 函数原型

```
Status CopyKvBlocks(const Cache &src_cache,
                    const Cache &dst_cache,
                    const std::vector<uint64_t> &src_blocks,
                    const std::vector<std::vector<uint64_t>> &dst_blocks_list)
```

## 参数说明

| 参数名称 | 输入/输出 | 取值说明 |
| --- | --- | --- |
| src_cache | 输入 | 源Cache。 |
| dst_cache | 输入 | 目的Cache。 |
| src_blocks | 输入 | 源Cache的block index列表。 |
| dst_blocks_list | 输入 | 目标Cache的block index列表的列表，一组src_blocks可以拷贝到多组dst_blocks。 |

## 调用示例

```
Status ret = llm_datadist.CopyKvCache(src_cache, dst_cache, {1,2}, {{1,2},{3,4}})
```

## 返回值

- LLM\_SUCCESS：成功
- LLM\_PARAM\_INVALID：参数错误
- 其他：失败

## 约束说明

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。不支持Host-\>Host的拷贝。
