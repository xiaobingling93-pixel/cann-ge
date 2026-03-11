# Proc

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

用户自定义flow func的处理函数。

## 函数原型

```
int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| inputMsgs | 输入 | 算子的输入参数列表。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

如果有不可恢复的异常信息发生，返回ERROR；其他情况则调用SetRetcode设置输出tensor的错误码。如果返回SUCCESS，调度会终止。

## 约束说明

无。
