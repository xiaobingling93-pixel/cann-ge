# SetRole

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

设置LLM-DataDist的角色。建议仅在使用PagedAttention的场景使用。

## 函数原型

```
Status SetRole(LlmRole role, const std::map<AscendString, AscendString> &options = {})
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| role | 输入 | 角色类型，类型为[LlmRole](LlmRole.md)。 |
| options | 输入 | 设置角色的参数，当前支持的参数请参见[表1](#table1987921348)。 |

**表 1**  配置项

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| OPTION_LISTEN_IP_INFO | 切换至Pormpt必选 | 设置为Device的IP地址和端口，如"192.168.1.1:26000" |

## 返回值

- LLM\_SUCCESS：设置角色成功
- LLM\_PARAM\_INVALID：参数错误
- LLM\_FEATURE\_NOT\_ENABLED：该特性未使能
- LLM\_EXIST\_LINK：存在残留链路资源
- 其他：失败

## 异常处理

无

## 约束说明

LLM-DataDist初始化时，需要设置OPTION\_ENABLE\_SET\_ROLE为"1"，才可以使用该接口。
