# finalize

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

释放LLMDataDist。

## 函数原型

```
finalize()
```

## 参数说明

无

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole, LLMConfig
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
llm_config = LLMConfig()
llm_config.listen_ip_info = "192.168.1.1:26000"
llm_config.device_id = 0
engine_options = llm_config.generate_options()
llm_datadist.init(engine_options)
llm_datadist.finalize()
```

## 返回值

无

## 约束说明

初始化成功后，系统退出前需要调用[finalize](finalize.md)。

[finalize](finalize.md)不能和其他接口并发调用。
