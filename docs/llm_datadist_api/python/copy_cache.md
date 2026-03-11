# copy\_cache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

拷贝KV Cache。

## 函数原型

```
copy_cache(dst: KvCache, src: KvCache, dst_batch_index: int = 0, src_batch_index: int = 0, offset: int = 0, size: int = -1, req_id: Optional[int] = None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| dst | [KvCache](KvCache构造函数.md) | 目标Cache。 |
| src | [KvCache](KvCache构造函数.md) | 源Cache。 |
| dst_batch_index | int | 目标Cache的batch_index，默认为0。 |
| src_batch_index | int | 源Cache的batch_index，默认为0。 |
| offset | int | 每个tensor的偏移，默认为0。 |
| size | int | 设置为>0的整数，表示要拷贝的大小。<br>或设置为-1，表示完整拷贝。<br>默认为-1。 |
| req_id | Optional[int] | 本次调用关联的req_id，如果设置了该参数则本地调用相关的维测日志中会打印该req_id<br>默认为None |

## 调用示例

```
kv_cache_manager.copy_cache(dst_cache, src_cache, 0, 1, 0, 128)
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

- src/dst的CacheDesc需要匹配。
- 本接口不支持并发，并发会排队等待。
