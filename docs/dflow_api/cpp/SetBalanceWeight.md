# SetBalanceWeight

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置均衡分发权重信息。

## 函数原型

```
void SetBalanceWeight(const BalanceWeight &balanceWeight) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| balanceWeight | 输入 | 均衡分发权重信息。<br>struct BalanceWeight {<br>int32_t rowNum = 0;  // 权重矩阵行数<br>int32_t colNum = 0;  // 权重矩阵列数<br>const int32_t *matrix = nullptr;  // 权重矩阵，有rowNum行、colNum列的数组，如果值为null，表示是值全为1的矩阵<br>}; |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
