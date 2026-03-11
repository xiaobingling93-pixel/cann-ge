# GetDataPos

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取输出数据对应权重矩阵中的位置。

## 函数原型

```
const std::vector<std::pair<int32_t, int32_t>> &GetDataPos() const = 0
```

## 参数说明

无。

## 返回值

输出数据对应权重矩阵中的位置， pair中第一个值表示对应的行号，第二个值表示对应的列号。

## 异常处理

无。

## 约束说明

无。
