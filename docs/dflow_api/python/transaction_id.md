# transaction\_id

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

以属性方式读写事务ID。

## 函数原型

```
@property
def transaction_id(self)
@transaction_id.setter
def transaction_id(self, new_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| new_value | int | 设置的事务ID的新值，设置为0时表示不使用自定义的transId，内部会采用自增的方式自动生成transId。 |

## 返回值

transaction\_id的属性。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
flowinfo = FlowInfo(...)
flowinfo.transaction_id = 10
print(flowinfo.transaction_id)
```

## 约束说明

- 只有构图接口通过set\_contains\_n\_mapping\_node设置为true时才生效。
- transaction\_id只能增大不能减小，外部不设置的情况下，transaction\_id从1开始自增。
- transaction\_id达到uint64\_max值后会报错。
- 开启数据对齐时，需要确保每批输入数据的transaction\_id一致，否则可能导致数据不对齐。
- 只有设置transaction\_id非0的时候才会使能自定义transaction\_id。
