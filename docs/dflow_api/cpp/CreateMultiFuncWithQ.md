# CreateMultiFuncWithQ

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

创建输入类型为[FlowMsgQueue](FlowMsgQueue类.md)的多func处理对象和处理函数，框架内部使用，用户不直接使用。

## 函数原型

创建流式输入（即函数入参为队列）flow func处理对象和处理函数时使用，即flow func输入为flowMsgQueue时使用。

```
int32_t CreateMultiFuncWithQ(std::shared_ptr<MetaMultiFunc> &multiFunc,
std::map<AscendString, PROC_FUNC_WITH_CONTEXT_Q> &procFuncWithQMap) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| multiFunc | 输出 | 多func实例对象 |
| procFuncWithQMap | 输出 | 输入类型为FlowMsgQueue的多flow func的处理函数。 |

## 返回值

- FLOW\_FUNC\_SUCCESS：成功
- 其他表示失败

## 异常处理

无。

## 约束说明

无。
