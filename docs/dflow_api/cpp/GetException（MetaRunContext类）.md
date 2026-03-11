# GetException（MetaRunContext类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

UDF获取异常，如果开启了异常捕获功能，需要在UDF中Proc函数开始位置尝试捕获异常。

## 函数原型

```
bool GetException(int32_t &expCode, uint64_t &userContextId);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| expCode | 输入 | 被捕获异常中用户自定义的异常的错误码。 |
| userContextId | 输入 | 被捕获异常中用户自定义的异常的上下文ID，用于标识该上报的异常，保证其他UDF获取捕获到异常后可以根据该ID值感知到具体产生异常的数据批次。 |

## 返回值

是否存在异常。

- true：存在异常被捕获。
- false：不存在异常需要被捕获。

## 异常处理

无

## 约束说明

如果当前DataFlow graph未通过**SetExceptionCatch**使能异常上报，**GetException**返回值固定为false。
