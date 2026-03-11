# GetMsgType（FlowMsg类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取FlowMsg的消息类型。

## 函数原型

```
MsgType GetMsgType() const
```

## 参数说明

无

## 返回值

返回FlowMsg的消息类型。

```
enum class MsgType : uint16_t {
    MSG_TYPE_TENSOR_DATA = 0,          // tensor data msg type
    MSG_TYPE_RAW_MSG = 1,              // raw data msg type
    MSG_TYPE_TENSOR_LIST = 2,          // raw data msg type
    MSG_TYPE_USER_DEFINE_START = 1024  // user define type start
};
```

## 异常处理

无。

## 约束说明

无。
