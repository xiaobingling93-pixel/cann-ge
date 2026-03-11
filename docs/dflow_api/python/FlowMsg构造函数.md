# FlowMsg构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

FlowMsg的构造函数。

## 函数原型

```
__init__(self, flow_msg: flowfunc_wrapper.FlowMsg) -> None
```

## 参数说明

flowfunc\_wrapper模块中定义的FlowMsg。实际执行时由C++代码传入，通过pybind11的绑定关系映射成flowfunc\_wrapper的FlowMsg对象。

## 返回值

返回FlowMsg类型的对象。

## 异常处理

无

## 约束说明

无
