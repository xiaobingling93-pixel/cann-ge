# set\_exception\_catch

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置用户异常捕获功能是否开启。

## 函数原型

```
set_exception_catch(self, enable_exception_catch: bool = False)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enable_exception_catch | bool | 是否启用用户异常捕获功能。取值如下：<br><br>  - True：是，开启异常功能。<br>  - False：否，关闭异常功能。<br><br>默认值：False。 |

## 返回值

无。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
graph.enable_exception_catch(True)
```

## 约束说明

开启异常功能时必须同时使用[set\_inputs\_align\_attrs](set_inputs_align_attrs.md)接口开启数据对齐功能，否则编译报错。
