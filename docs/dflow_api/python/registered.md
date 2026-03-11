# registered

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

判断消息类型ID是否被注册过

## 函数原型

```
registered(msg_type)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| msg_type | int | 注册的类型ID。 |

## 返回值

bool，True表示注册过，False表示没有注册

## 调用示例

```
import dataflow as df
registered = df.msg_type_register.registered(1026)
```

## 约束说明

无
