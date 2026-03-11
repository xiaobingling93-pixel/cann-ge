# FetchDataFlowGraph（按索引获取输出数据）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

按索引获取图输出数据。

## 函数原型

```
Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes, std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graph_id | 输入 | 要执行图对应的ID。 |
| indexes | 输入 | 输出数据的索引，不可以重复。取值范围是【0~N-1】，N表示输出数据个数。支持一个或者多个。<br>示例：{0,2}，表示获取第一个和第三个输出数据。 |
| outputs | 输出 | 计算图输出Tensor，用户无需分配内存空间，执行完成后GE会分配内存并赋值。 |
| info | 输出 | 输出数据流标志（flow flag）。具体请参考[DataFlowInfo数据类型](DataFlowInfo数据类型.md)。 |
| timeout | 输入 | 数据提取超时时间，单位：ms，取值为-1时表示从不超时。 |

## 返回值

函数状态结果如下。

| 参数名 | 类型 | 描述 |
| --- | --- | --- |
| - | Status | - SUCCESS：数据获取成功。<br>  - FAILED：数据获取失败。<br>  - 其他错误码请参考[UDF错误码](UDF错误码.md)。 |

## 约束说明

无
