# init

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

初始化LLMDataDist。通过单边建链，即Client向Server发起建链，Prompt作为Server，Decode作为Client；限制只能从Decode往Prompt拉取KV。

## 函数原型

```
init(options: Dict[str, str])
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| options | Dict[str, str] | 配置项。<br>传入的options可以通过[LLMConfig](LLMConfig.md)来生成。<br>必填字段：device_id和listen_ip_info（由于Prompt作为Server，必填；Decode作为Client无需填充listen_ip_info信息）。 |

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole, LLMConfig
llm_datadist = LLMDataDist(LLMRole.PROMPT, 0)
llm_config = LLMConfig()
llm_config.listen_ip_info = "192.168.1.1:26000"
llm_config.device_id = 0
engine_options = llm_config.generate_options()
llm_datadist.init(engine_options)
```

## 返回值

正常情况下无返回值。

异常情况会抛出[LLMException](LLMException.md)。

参数错误可能抛出TypeError或ValueError。

## 约束说明

初始化成功后，系统退出前需要调用[finalize](finalize.md)。
