# AllocateCache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

分配Cache。

## 函数原型

```
Status AllocateCache(const CacheDesc &cache_desc, Cache &cache)
```

## 参数说明

| 参数名称 | 输入/输出 | 取值说明 |
| --- | --- | --- |
| cache_desc | 输入 | Cache的描述。 |
| cache | 输出 | 分配出的Cache，当该接口返回LLM_SUCCESS时有效。 |

## 调用示例

```
CacheDesc kv_desc{};
kv_desc.num_tensors = 80;
kv_desc.data_type = DT_FLOAT16;
kv_desc.shape = {1, 256};
Cache cache;
Status ret = llm_datadist.AllocateCache(kv_desc, cache);
```

## 返回值

- LLM\_SUCCESS：成功
- LLM\_PARAM\_INVALID：参数错误
- LLM\_DEVICE\_OUT\_OF\_MEMORY: Device内存不足
- 其他：失败

## 约束说明

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。

仅支持参数“cache\_desc”中的placement为CachePlacement::kDevice时，该接口生效。Cache的描述请参考[CacheDesc](CacheDesc.md)。
