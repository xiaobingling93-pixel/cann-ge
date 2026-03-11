# per\_device\_tensor\_addrs

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

获取KvCache的地址。

## 函数原型

```
@property
per_device_tensor_addrs() -> List[List[int]]
```

## 参数说明

无

## 调用示例

```
...
kv_cache = kv_cache_manager.allocate_cache(cache_desc, cache_keys)
# 两层List结构为单进程多卡设计，单进程单卡时直接获取第一个卡的地址
print(kv_cache.per_device_tensor_addrs[0])
```

## 返回值

正常情况返回类型为KVCache的地址。

## 约束说明

无
