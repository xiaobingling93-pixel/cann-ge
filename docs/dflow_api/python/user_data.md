# user\_data

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取用户信息。

## 函数原型

```
@property
def user_data(self)
```

## 参数说明

无

## 返回值

以属性方式返回user\_data对象。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
user_data_str = "UserData123"
result = graph.fetch_data() # 异步取结果
flowinfo = result[1]
fetch_user_data = flowinfo.user_data[0:len(user_data_str)]
name = fetch_user_data.decode('utf-8')
print(name)
```

## 约束说明

无
