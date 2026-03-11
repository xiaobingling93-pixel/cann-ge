# swap\_blocks

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

对cpu\_cache和npu\_cache进行换入换出。

对于swap out功能，该接口启用了4个线程执行并行任务；对于swap in功能，该接口启用了1个d2d线程。为了性能稳定，建议进行进程绑核。

swap in功能分为H2D和D2D两个阶段，为了保障性能，该接口申请了4个block大小的buffer用作流水拷贝，所以建议预留出对应的Device内存，防止出现OOM。

## 函数原型

```
swap_blocks(src: KvCache, dst: KvCache, src_to_dst: Dict[int, int])
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| src | [KvCache](KvCache构造函数.md) | 源Cache。 |
| dst | [KvCache](KvCache构造函数.md) | 目标Cache。 |
| src_to_dst | Dict[int, int] | dict里面内容代表（原始block index，目标block index） |

## 调用示例

```
from llm_datadist import KvCache
...
npu_cache = kv_cache_manager.allocate_blocks_cache(npu_cache_desc, npu_cache_key)
cpu_cache = KvCache.create_cpu_cache(cpu_cache_desc, cpu_addrs) # cpu_addrs来自创建的cpu tensors
# swap in
kv_cache_manager.swap_blocks(cpu_cache, npu_cache, {1:2, 3:4})
# swap out
kv_cache_manager.swap_blocks(npu_cache, cpu_cache, {1:2, 3:4})
```

## 返回值

正常情况下无返回值。

传入数据类型错误，源Cache和目标Cache不匹配情况下会抛出TypeError或ValueError异常。

传入参数为None，会抛出AttributeError异常。

## 约束说明

无
