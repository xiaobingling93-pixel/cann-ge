# BalanceConfig构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

BalanceConfig构造函数。

## 函数原型

```
__init__(self, row_num: int, col_num: int, affinity_policy: AffinityPolicy = AffinityPolicy.NO_AFFINITY) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| row_num | 输入 | 权重矩阵行数 |
| col_num | 输入 | 权重矩阵列数 |
| affinity_policy | 输入 | 亲和策略，默认非亲和 |

## 返回值

返回BalanceConfig类型的对象。

## 异常处理

无

## 约束说明

无
