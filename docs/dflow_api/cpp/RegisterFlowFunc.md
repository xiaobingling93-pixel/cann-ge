# RegisterFlowFunc

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

注册flow func。

不建议直接使用该函数，建议使用[MetaFlowFunc注册函数宏](MetaFlowFunc注册函数宏.md)来注册flow func。

## 函数原型

```
FLOW_FUNC_VISIBILITY bool RegisterFlowFunc(const char *flowFuncName, const FLOW_FUNC_CREATOR_FUNC &func) noexcept
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flowFuncName | 输入 | flow func的名称。不可以设置为NULL，必须以“\0”结尾。 |
| func | 输入 | flow func的创建函数。 |

## 返回值

- true
- false

## 异常处理

无。

## 约束说明

无。
