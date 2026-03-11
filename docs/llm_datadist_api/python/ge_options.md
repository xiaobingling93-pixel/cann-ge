# ge\_options

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

配置额外的GE配置项。

## 函数原型

```
ge_options(ge_options)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| ge_options | dict[str, str] | 配置GE配置项。<br>其中ge.flowGraphMemMaxSize比较重要，表示所有KV cache占用的最大内存，如果设置的过大，会压缩模型的可用内存，需根据实际情况指定。 |

## 调用示例

```
from llm_datadist import LLMConfig
ge_options = {
    "ge.flowGraphMemMaxSize": "4106127360"
}
llm_config = LLMConfig()
llm_config.ge_options = ge_options
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
