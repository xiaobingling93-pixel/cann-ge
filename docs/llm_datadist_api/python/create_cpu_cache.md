# create\_cpu\_cache

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

创建cpu cache。

## 函数原型

```
create_cpu_cache(cache_desc: CacheDesc, addrs: Union[List[int], List[List[int]]])
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| cache_desc | [CacheDesc](CacheDesc.md) | cache的描述。 |
| addrs | Union[List[int], List[List[int]]] | cpu cache的地址。 |

## 调用示例

```
from llm_datadist import KvCache
cpu_addrs = []
# 单进程多卡模式下，配置多卡的地址。
# cpu_addrs = [[]]
cpu_cache = KvCache.create_cpu_cache(cpu_cache_desc, cpu_addrs) # cpu_addrs来自创建的cpu tensors
```

## 返回值

正常情况返回类型为KvCache的cpu\_cache。

传入数据类型错误情况下会抛出TypeError或ValueError异常。

传入参数为None，会抛出AttributeError异常。

## 约束说明

无
