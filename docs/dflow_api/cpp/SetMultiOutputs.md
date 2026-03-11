# SetMultiOutputs

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

批量设置指定index和options的输出，该函数供func函数调用。

## 函数原型

```
int32_t SetMultiOutputs(uint32_t outIdx, const std::vector<std::shared_ptr<FlowMsg>> &outMsgs, const OutOptions &options) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outIdx | 输入 | 输出index，从0开始。 |
| outMsgs | 输入 | 输出信息列表。 |
| options | 输入 | 输出对应的OutOptions。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无。

## 约束说明

无。
