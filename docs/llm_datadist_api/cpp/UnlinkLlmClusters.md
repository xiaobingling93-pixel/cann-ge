# UnlinkLlmClusters

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

进行Device间断链。

## 函数原型

```
Status UnlinkLlmClusters(const std::vector<ClusterInfo> &clusters, std::vector<Status> &rets, int32_t timeout_in_millis = 1000, bool force_flag = false)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| clusters | 输入 | 需要断链的cluster信息。类型为[ClusterInfo](ClusterInfo和IpInfo.md)。 |
| rets | 输出 | 每个cluster断链结果。 |
| timeout_in_millis | 输入 | 断链超时时间，单位ms。默认超时1000。 |
| force_flag | 输入 | 是否为强制断链。默认为否。<br>强制断链仅强制拆除本端链接，两端都要调用。<br>非强制断链在Decode发起，无故障时两端链路都会拆除，在链路故障场景会耗时较久，且需要在Prompt端也发起调用。 |

## 返回值

- SUCCESS：只有所有clusters断链成功，接口才会返回成功。
- 其他：执行断链失败，需要查看rets每个cluster的断链结果。

## 异常处理

- LLM\_PROCESSING\_LINK：接口有锁保护，一个LLM-DataDist多线程调用断链接口会串行执行，其他线程等待时间超过设置的超时时间会报错退出。
- LLM\_UNLINK\_FAILED：断链失败。

## 约束说明

需要在[Initialize](Initialize.md)接口初始化完成后调用。
