# GetUserData（FlowMsg数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取用户信息。

## 函数原型

```
Status GetUserData(void *data, size_t size, size_t offset = 0U) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| data | 输入/输出 | 用户数据指针 |
| size | 输入 | 用户数据长度。取值范围 (0, 64]。 |
| offset | 输入 | 用户数据的偏移值，需要遵循如下约束。<br>[0, 64), size + offset <= 64 |

## 返回值

- 0：SUCCESS。
- other：FAILED。

## 异常处理

无。

## 约束说明

无。

该接口的参数size和offset取值需要和[SetUserData](SetUserData（DataFlowInfo数据类型）.md)中的size和offset取值保持一致。
