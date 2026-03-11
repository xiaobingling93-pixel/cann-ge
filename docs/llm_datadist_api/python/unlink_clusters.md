# unlink\_clusters

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

断链。

## 函数原型

```
unlink_clusters(clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=3000, force=False)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| clusters | Union[List[[LLMClusterInfo](LLMClusterInfo构造函数.md)], Tuple[[LLMClusterInfo](LLMClusterInfo构造函数.md)]] | 集群列表。 |
| timeout | int | 超时时间，单位：ms，默认超时时间3000ms。 |
| force | bool | 是否强制断链，默认False。True表示强制断链。<br><br>  - 强制断链仅强制拆除本端链接，两端都要调用。<br>  - 非强制断链在Decode发起。无故障时两端链路都会拆除。有故障导致断链失败时，需要在Prompt端也发起断链操作。<br>  - 无故障时两端链路都会拆除。<br>  - 有故障导致断链失败时，需要在Prompt端也发起断链操作。 |

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole, LLMStatusCode, LLMClusterInfo
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
cluster = LLMClusterInfo()
cluster.remote_cluster_id = 1
cluster.append_local_ip_info("1.1.1.1", 26000)
cluster.append_remote_ip_info("1.1.1.1", 26000)
ret, rets = llm_datadist.unlink_clusters([cluster], 5000)
if ret != LLMStatusCode.LLM_SUCCESS:
    raise Exception("unlink failed.")
for cluster_i in range(len(rets)):
    unlink_ret = rets[cluster_i]
    if unlink_ret != LLMStatusCode.LLM_SUCCESS:
        print(f"{cluster_i} unlink failed.")
```

## 返回值

正常情况下返回两个值的元组，第一个值是接口的返回值，类型是[LLMStatusCode](LLMStatusCode.md)，第二个是每个集群建链结果的列表，类型是[LLMStatusCode](LLMStatusCode.md)。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
