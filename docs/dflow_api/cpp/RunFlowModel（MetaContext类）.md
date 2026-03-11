# RunFlowModel（MetaContext类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

同步执行指定的模型。该函数供[Proc](Proc.md)调用。

## 函数原型

```
int32_t RunFlowModel(const char *modelKey, const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs,
std::vector<std::shared_ptr<FlowMsg>> &outputMsgs, int32_t timeout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| modelKey | 输入 | 指定的模型key，与AddInvokedClosure中指定的name一致。 |
| inputMsgs | 输入 | 提供给模型的输入。 |
| outputMsgs | 输出 | 模型执行的输出结果。 |
| timeout | 输入 | 等待模型执行超时时间，单位ms，-1表示永不超时。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无。

## 约束说明

无。
