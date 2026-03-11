# copy\_blocks

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

PagedAttention场景下，拷贝block。

## 函数原型

```
copy_blocks(cache: KvCache, copy_block_info: Dict[int, List[int]])
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache | [KvCache](KvCache构造函数.md) | 目标Cache。 |
| copy_block_info | Dict[int, List[int]] | dict里面内容代表（原始block index，目标block index列表）。 |

## 调用示例

```
from llm_datadist import *
...
kv_cache_manager.copy_blocks(kv_cache, {1: [2,3]})
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

本接口不支持并发，并发会排队等待。
