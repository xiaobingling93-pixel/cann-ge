# SetTransactionId（DataFlowInfo数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置DataFlow数据传输使用的事务ID。

## 函数原型

```
void SetTransactionId(uint64_t transaction_id)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| transaction_id | 输入 | 数据传输使用的事务ID。<br>设置为0时表示不使用自定义的transaction_id，内部会采用自增的方式自动生成transaction_id。<br>在数据对齐场景下，会使用transaction_id进行路由和对齐。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

- 只有构图接口通过SetContainsNMappingNode设置为true时才生效。
- transaction\_id只能增大不能减小，外部不设置的情况下，transaction\_id从1开始自增。
- transaction\_id达到uint64\_max值后会报错。
- 开启数据对齐时，需要确保每批输入数据的transaction\_id一致，否则可能导致数据不对齐。
- 只有调用SetTransactionId接口传入非0的时候才会使能自定义transaction\_id。
