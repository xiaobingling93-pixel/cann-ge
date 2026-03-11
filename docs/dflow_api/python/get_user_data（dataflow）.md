# get\_user\_data（dataflow）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取用户信息。

## 函数原型

```
get_user_data(self, size: int = 0, offset: int = 0) -> bytearray
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| size | int | 用户数据长度。取值范围[0, 64]。 |
| offset | int | 用户数据的偏移值，需要遵循如下约束。<br>[0, 64), size+offset<=64 |

## 返回值

正常场景下返回bytearray。

异常情况抛出异常，异常类型dataflow.DfException。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
user_data_str = "UserData123"
result = graph.fetch_data() # 异步取结果
flowinfo = result[1]
fetch_user_data = flowinfo.get_user_data(len(user_data_str))
name = fetch_user_data.decode('utf-8')
print(name)
```

## 约束说明

无
