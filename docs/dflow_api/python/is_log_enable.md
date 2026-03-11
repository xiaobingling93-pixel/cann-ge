# is\_log\_enable

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

查询对应级别和类型的日志是否开启。

## 函数原型

```
is_log_enable(self, log_type: fw.FlowFuncLogType, log_level: fw.FlowFuncLogLevel) -> bool
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| log_type | 输入 | 日志类型，取值如下：<br><br>  - DEBUG_LOG<br>  - RUN_LOG |
| log_level | 输入 | 日志级别，取值如下：<br><br>  - DEBUG<br>  - INFO<br>  - WARN<br>  - ERROR |

## 返回值

- True：开启
- False：未开启

## 异常处理

无

## 约束说明

无
