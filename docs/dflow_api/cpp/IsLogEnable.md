# IsLogEnable

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

查询对应级别和类型的日志是否开启。

## 函数原型

```
virtual bool IsLogEnable(FlowFuncLogLevel level) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| level | 输入 | 日志级别取值类型为FlowFuncLogLevel。<br>enum class FLOW_FUNC_VISIBILITY FlowFuncLogLevel {<br>DEBUG = 0,<br>INFO = 1,<br>WARN = 2,<br>ERROR = 3<br>}; |

## 返回值

日志级别是否可记录。

## 异常处理

无。

## 约束说明

无。
