# check\_link\_status

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

可快速检测链路状态是否正常。

## 函数原型

```
check_link_status(remote_cluster_id: int)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| remote_cluster_id | int | 远程集群ID。 |

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole, LLMStatusCode, LLMClusterInfo
...
try:
    data_dist.check_link_status(remote_cluster_id=0)
except LLMException as ex:
    print(f"check_link_status exception:{ex.status_code}")
    raise ex
kv_cache_manager = data_dist.kv_cache_manager
...
kv_cache_manager.pull_cache(prompt_cache_key, local_kv_cache, batch_index=0)
```

## 返回值

正常情况下无返回值。

运行失败会抛出[LLMException](LLMException.md)异常。

参数错误可能抛出TypeError或ValueError。

## 约束说明

只有Client侧可以调用。

调用失败如果异常错误码是不可恢复错误码，需重新建链。

调用失败时，需持续调用该接口直至成功，才能调用[pull\_cache](pull_cache.md)、[pull\_blocks](pull_blocks.md)等接口。

该接口如果和[pull\_cache](pull_cache.md)、[pull\_blocks](pull_blocks.md)等接口并发，可能抛出[LLMException](LLMException.md)异常，错误码为LLM\_LINK\_BUSY。

超时时间由[llm.SyncKvCacheWaitTime](sync_kv_timeout.md)配置项指定。
