# RaiseException（MetaContext类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

UDF主动上报异常，该异常可以被同作用域内的其他UDF捕获。

## 函数原型

```
void RaiseException(int32_t expCode, uint64_t userContextId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| expCode | 输入 | 用户自定义的异常的错误码。 |
| userContextId | 输入 | 用户自定义的异常的上下文ID，用于标识该上报的异常，保证其他UDF获取捕获到异常后可以根据该ID值感知到具体产生异常的数据批次。 |

## 返回值

无

## 异常处理

打印错误日志。

## 约束说明

如果当前DataFlow graph未通过**SetExceptionCatch**使能异常上报，UDF中调用该接口会导致进程报错退出。

流式输入（即flow func输入为队列）场景下，不支持UDF主动上报异常。
