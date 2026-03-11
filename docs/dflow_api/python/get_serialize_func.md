# get\_serialize\_func

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据消息类型ID获取注册的序列化函数。

## 函数原型

```
get_serialize_func(msg_type)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| msg_type | int | 注册的类型ID。 |

## 返回值

注册的序列化函数，没有注册过返回None。

## 调用示例

```
import dataflow as df
serialize_func = df.msg_type_register.get_serialize_func(1026)
```

## 约束说明

无
