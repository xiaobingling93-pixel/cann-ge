# get\_cache\_tensors

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

获取cache tensor。

## 函数原型

```
get_cache_tensors(cache: KvCache, tensor_index: int = 0) -> List[Tensor]
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache | [KvCache](KvCache构造函数.md) | 要获取的tensor所在的KvCache。 |
| tensor_index | int | 要获取的tensor在Cache中的index。 |

## 调用示例

```
tensors = kv_cache_manager.get_cache_tensors(kv_cache, 0)
```

## 返回值

正常情况下返回List\[Tensor\]。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明

本接口不支持并发调用。
