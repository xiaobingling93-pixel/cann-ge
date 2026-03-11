# raise\_exception

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

UDF主动上报异常。

## 函数原型

```
raise_exception(self, exception_code: int, user_context_id: int)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| exception_code | 输入 | 用户自定义的异常的错误码。 |
| user_context_id | 输入 | 用户自定义的异常的上下文ID，用于标识该上报的异常，保证其他UDF获取捕捉到异常后可以根据该ID值感知到具体产生异常的数据批次。 |

## 返回值

无

## 异常处理

打印错误日志。

## 约束说明

如果当前dataflow graph未通过set\_exception\_catch接口使能异常上报，调用该接口会导致进程报错退出。

流式输入（flow func入参为队列）场景下，不支持UDF主动上报异常。
