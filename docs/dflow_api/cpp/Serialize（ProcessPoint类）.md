# Serialize（ProcessPoint类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

ProcessPoint的序列化方法。由ProcessPoint的子类去实现该方法的功能。

## 函数原型

```
virtual void Serialize(ge::AscendString &str) const = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| str | 输出 | ProcessPoint序列化的字符串。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
