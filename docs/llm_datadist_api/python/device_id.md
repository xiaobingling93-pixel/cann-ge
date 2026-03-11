# device\_id

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

设置当前进程Device ID，对应底层ge.exec.deviceId配置项。

## 函数原型

```
device_id(device_id)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| device_id | Union[int, List[int], Tuple[int]] | 设置当前进程的Device ID。支持配置为一个或者列表，配置为列表时以半角逗号间隔。<br><br>  - 单进程单卡场景下，需要配置为一个，例如：0<br>  - 单进程多卡场景下，需要配置为列表，例如：[0, 1] |

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
# 单进程单卡设置方法
llm_config.device_id = 0
# 单进程多卡设置方法
# llm_config.device_id = [0, 1]
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
