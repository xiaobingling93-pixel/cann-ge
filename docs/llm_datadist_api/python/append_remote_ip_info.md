# append\_remote\_ip\_info

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

添加远端集群IP信息。

## 函数原型

```
append_remote_ip_info(self, ip: Union[str, int], port: int)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| ip | Union[str, int] | 设置为对端集群Device卡IP的地址。 |
| port | int | 设置为对端集群Device卡的端口。 |

## 调用示例

```
llm_cluster = LLMClusterInfo()
llm_cluster.append_remote_ip_info("1.1.1.1", 10000)
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
