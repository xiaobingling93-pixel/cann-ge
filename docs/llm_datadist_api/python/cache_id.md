# cache\_id

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

获取KvCache的id。

## 函数原型

```
@property
cache_id() -> int
```

## 参数说明

无

## 调用示例

```
...
kv_cache = kv_cache_manager.allocate_cache(cache_desc, cache_keys)
print(kv_cache.cache_id)
```

## 返回值

正常情况返回类型为KvCache的id。

## 约束说明

无
