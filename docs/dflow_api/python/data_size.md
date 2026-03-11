# data\_size

## 产品支持情况


| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取user\_data的长度。

## 函数原型

```
@property
def data_size(self)
```

## 参数说明

无

## 返回值

返回user\_data的长度。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
user_data_str = "UserData123"
result = graph.fetch_data() # 异步取结果
flowinfo = result[1]
user_data_size = flowinfo.data_size
print(user_data_size)
```

## 约束说明

无

