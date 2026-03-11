# get\_msg\_type（dataflow）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据类型定义获取注册的消息类型ID。

## 函数原型

```
get_msg_type(clz)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| clz | 类型定义 | 类型定义，比如int，str，或者自定义的class。 |

## 返回值

注册的消息类型ID，没有注册过返回None。

## 调用示例

```
import dataflow as df
class TestClass():
    def __init__(self, name, val):
        self.name = name
        self.val = val
msg_type = df.msg_type_register.get_msg_type(TestClass)
```

## 约束说明

无
