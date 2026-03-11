# GetVal\(int64\_t &value\)

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取int类型的属性值。

## 函数原型

```
int32_t GetVal(int64_t &value) const = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| Value | 输出 | 获取的int类型属性值。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无。

## 约束说明

无。
