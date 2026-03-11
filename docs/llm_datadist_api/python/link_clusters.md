# link\_clusters

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

由Client单侧发起建链。由于动态扩缩的部分大部分是Decode侧，因此将P定义为Server端，D定义为Client端，建链过程实现由D向P发起建链的流程。

## 函数原型

```
link_clusters(clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=3000)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| clusters | Union[List[[LLMClusterInfo](LLMClusterInfo构造函数.md)], Tuple[[LLMClusterInfo](LLMClusterInfo构造函数.md)]] | 集群列表。 |
| timeout | int | 超时时间，单位：ms，默认超时时间3000ms。 |

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole, LLMStatusCode, LLMClusterInfo
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
cluster = LLMClusterInfo()
cluster.remote_cluster_id = 1
cluster.append_local_ip_info("1.1.1.1", 26000)
cluster.append_remote_ip_info("1.1.1.1", 26000)
ret, rets = llm_datadist.link_clusters([cluster], 5000)
if ret != LLMStatusCode.LLM_SUCCESS:
    raise Exception("link failed.")
for cluster_i in range(len(rets)):
    link_ret = rets[cluster_i]
    if link_ret != LLMStatusCode.LLM_SUCCESS:
        print(f"{cluster_i} link failed.")
```

## 返回值

正常情况下返回两个值的元组，第一个值是接口的返回值，类型是[LLMStatusCode](LLMStatusCode.md)，第二个是每个集群建链结果的列表，类型是[LLMStatusCode](LLMStatusCode.md)。

参数错误可能抛出TypeError或ValueError。

## 约束说明

- 建链的要求如下。建链数量过多存在内存OOM及KV Cache传输的性能风险。
  - Server侧（P侧）并发建链的数量=16，允许创建的最大通信数量=512。
  - Client侧（D侧）允许创建的最大通信连接数量=512。

- 建议超时时间配置200ms以上。
- 只有Client发起调用
