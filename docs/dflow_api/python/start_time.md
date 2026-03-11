# start\_time

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

以属性方式读取和设置FlowInfo的开始时间。

## 函数原型

```
@property
def start_time(self)
@start_time.setter
def start_time(self, new_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| new_value | int | 要设置的FlowInfo开始时间的值。 |

## 返回值

start\_time属性。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
flowinfo = FlowInfo(...)
flowinfo.start_time = 100
print(flowinfo.start_time)
```

## 约束说明

无
