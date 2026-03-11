# get\_msg\_type（UDF）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取FlowMsg的消息类型。

## 函数原型

```
get_msg_type(self)
```

## 参数说明

无

## 返回值

返回FlowMsg的消息类型。

```
import dataflow.flow_func.flow_func as ff
# 消息返回如下两种value：
ff.MSG_TYPE_TENSOR_DATA
ff.MSG_TYPE_RAW_MSG
# 大于等于1024为用户自定义类型
```

## 异常处理

无

## 约束说明

无
