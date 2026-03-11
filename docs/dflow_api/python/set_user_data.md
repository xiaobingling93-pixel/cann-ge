# set\_user\_data

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置用户信息。

## 函数原型

```
set_user_data(self, user_data: bytearray, offset: int = 0)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| user_data | bytearray | 用户自定义数据。 |
| offset | int | 用户数据的偏移值，需要遵循如下约束。<br>[0, 64), size + offset <= 64 |

## 返回值

正常场景下返回None。

异常情况抛出异常，异常类型dataflow.DfException。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
flow_info = FlowInfo(...)
user_data_str = "UserData123"
user_data_array = bytearray(user_data_str, 'utf-8')
flow_info.set_user_data(user_data_array)
```

## 约束说明

无
