# switch\_role

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

切换当前LLMDataDist的角色，建议仅在使用PagedAttention的场景使用。

## 函数原型

```
switch_role(self, role: LLMRole, switch_options: Optional[Dict[str, str]] = None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| role | [LLMRole](LLMRole.md) | 切换的目标角色。 |
| switch_options | options: Dict[str, str] | 切换角色配置项。<br>可选参数，默认值为None。<br>切换为Prompt时需要设置，其中需包含[listen_ip_info](listen_ip_info.md)配置项。 |

## 调用示例

```
from llm_datadist import LLMDataDist, LLMRole
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
...
switch_options = { 'llm.listenIpInfo': '127.0.0.1:26000' }
llm_datadist.switch_role(LLMRole.PROMPT, switch_options)
```

## 返回值

- 正常情况下无返回值。
- 传入数据类型错误情况下会抛出TypeError或ValueError异常。
- 如果初始化LLMDataDist时LLMConfig未使能[enable\_switch\_role](enable_switch_role.md)，调用该接口则会抛出[LLMException](LLMException.md)，status\_code为LLM\_FEATURE\_NOT\_ENABLED。
- 如果switch\_role时存在残留链路资源，则会抛出[LLMException](LLMException.md)，status\_code为LLM\_EXIST\_LINK。
- 如果switch\_role的目标role与当前role相同，则会抛出[LLMException](LLMException.md)，status\_code为LLM\_PARAM\_INVALID。
- 单进程多卡模式下，不支持调用该接口。

## 约束说明

无
