# SetDataPos

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置输出数据对应权重矩阵中的位置。

## 函数原型

```
void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &dataPos) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dataPos | 输入 | 输出数据对应的权重矩阵位置。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
