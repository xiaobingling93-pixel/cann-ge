# enable\_switch\_role

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

配置是否支持角色平滑切换，对应底层llm.EnableSwitchRole配置项。

## 函数原型

```
enable_switch_role(self, enable_switch_role: bool)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| enable_switch_role | bool | 使能角色切换。<br><br>  - True：支持<br>  - False：不支持<br><br>不配置默认为不支持。<br>相关接口：[switch_role](switch_role.md)。 |

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
llm_config.enable_switch_role = True
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
