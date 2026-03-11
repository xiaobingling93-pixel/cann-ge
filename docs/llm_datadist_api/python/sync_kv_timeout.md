# sync\_kv\_timeout

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

配置一系列接口的超时时间对应底层llm.SyncKvCacheWaitTime配置项。

## 函数原型

```
sync_kv_timeout(sync_kv_timeout)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| sync_kv_timeout | int或str | 同步kv超时时间，单位：ms。不配置默认为1000ms。 |

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
llm_config.sync_kv_timeout = 1000
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
