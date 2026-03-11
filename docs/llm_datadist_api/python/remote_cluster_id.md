# remote\_cluster\_id

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

设置对端集群ID。

## 函数原型

```
remote_cluster_id(remote_cluster_id)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| remote_cluster_id | int | 对端集群ID。 |

## 调用示例

```
llm_cluster = LLMClusterInfo()
llm_cluster.remote_cluster_id = 1
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
