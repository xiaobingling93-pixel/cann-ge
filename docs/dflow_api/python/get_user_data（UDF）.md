# get\_user\_data（UDF）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取用户定义数据。

## 函数原型

```
get_user_data(self, size: int , offset: int = 0) -> bytearray
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| size | 输入 | 用户数据长度。取值范围[0, 64]。 |
| offset | 输入 | 用户数据的偏移值，需要遵循如下约束。<br>[0, 64), size+offset<=64 |

## 返回值

返回用户自定义的数据，类型是bytearray。

## 异常处理

无

## 约束说明

offset不传时默认值=0。返回的类型是bytearray，用户需要根据定义的结构反向解析。

- string类型用byte\_array.decode\("utf-8"\)解析。
- int类型可以用int.from\_bytes\(byte\_array, byteorder='big'\)解析，byteorder根据环境设置。
- float类型可以用struct.unpack\('f', byte\_array\)\[0\]解析。
