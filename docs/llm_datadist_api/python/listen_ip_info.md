# listen\_ip\_info

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

PROMPT侧设置集群侦听信息，对应底层llm.listenIpInfo配置项。

## 函数原型

```
listen_ip_info(listen_ip_info)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| listen_ip_info | str | 设置为Device侧的IP地址和端口，支持配置为一个或者多个，配置多个时用英文分号分割。<br><br>  - 单进程单卡场景下，需要配置为一个，例如："192.168.1.1:26000"<br>  - 单进程多卡场景下，需要配置为多个，例如："192.168.1.1:26000;192.168.1.2:26000" |

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
# 单进程单卡设置方法
llm_config.listen_ip_info = "192.168.1.1:26000"
# 单进程多卡设置方法
# llm_config.listen_ip_info = "192.168.1.1:26000;192.168.1.2:26000"
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
