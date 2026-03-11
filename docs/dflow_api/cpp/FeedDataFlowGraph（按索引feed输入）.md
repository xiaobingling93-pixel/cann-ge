# FeedDataFlowGraph（按索引feed输入）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将数据按索引输入到Graph图。

## 函数原型

```
Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes, const std::vector<Tensor> &inputs, const DataFlowInfo &info, int32_t timeout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graph_id | 输入 | 要执行图对应的ID。 |
| indexes | 输入 | 输入数据的索引。取值范围是【0~N-1】，N表示输入数据个数。数量需要和inputs入参的数量保持一致。<br>示例：{0,2}，表示对第一个和第三个输入进行feed数据。 |
| inputs | 输入 | 计算图输入Tensor，为Host上分配的内存空间。 |
| info | 输入 | 输入数据流标志（flow flag）。具体请参考[DataFlowInfo数据类型](DataFlowInfo数据类型.md)。 |
| timeout | 输入 | 数据输入超时时间，单位：ms，取值为-1时表示从不超时。 |

## 返回值

| 参数名 | 类型 | 描述 |
| --- | --- | --- |
| - | Status | - SUCCESS：数据输入成功。<br>  - FAILED：数据输入失败。<br>  - 其他错误码请参考[UDF错误码](UDF错误码.md)。 |

## 约束说明

无
