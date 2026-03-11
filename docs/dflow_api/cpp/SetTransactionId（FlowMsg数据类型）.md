# SetTransactionId（FlowMsg数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowMsg消息中的事务ID，事务ID从1开始计数，可用于识别哪一批数据，设置为0时表示不使用自定义的transaction\_id，内部会采用自增的方式自动生成transaction\_id。

## 函数原型

```
virtual void SetTransactionId(uint64_t transaction_id)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| transaction_id | 输入 | 消息的事务ID |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
