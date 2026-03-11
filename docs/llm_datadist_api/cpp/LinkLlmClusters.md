# LinkLlmClusters

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

进行Device间建链。

## 函数原型

```
Status LinkLlmClusters(const std::vector<ClusterInfo> &clusters, std::vector<Status> &rets, int32_t timeout_in_millis = 1000)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| clusters | 输入 | 需要建链的cluster信息。类型为[ClusterInfo](ClusterInfo和IpInfo.md)。 |
| rets | 输出 | 每个cluster建链结果。 |
| timeout_in_millis | 输入 | 建链超时时间，单位ms。默认超时1000。 |

## 返回值

- LLM\_SUCCESS：只有所有clusters建链成功，接口才会返回成功。
- 其他：建链失败，需要查看rets每个cluster的建链结果。

## 异常处理

- LLM\_PROCESSING\_LINK：接口有锁保护，一个LLM-DataDist多线程调用建链接口会串行执行，其他线程等待时间超过设置的超时时间会报错退出。
- LLM\_ALREADY\_LINK：增量的cluster已经和全量cluster建立了链接。
- LLM\_LINK\_FAILED：建链失败。
- LLM\_CLUSTER\_NUM\_EXCEED\_LIMIT：建链cluster数量超过上限，当前是16个。

## 约束说明

需要在[Initialize](Initialize.md)接口初始化完成后调用，只支持角色为Decoder时调用。
