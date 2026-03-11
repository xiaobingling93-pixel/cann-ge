# FetchDataFlowGraph（获取所有输出数据）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取图输出数据。

## 函数原型

```
Status FetchDataFlowGraph(uint32_t graph_id, std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graph_id | 输入 | 要执行图对应的ID。 |
| outputs | 输出 | 计算图输出Tensor，用户无需分配内存空间，执行完成后GE会分配内存并赋值。 |
| info | 输出 | 输出数据流标志（flow flag）。具体请参考[DataFlowInfo数据类型](DataFlowInfo数据类型.md)。 |
| timeout | 输入 | 数据获取超时时间，单位：ms，取值为-1时表示从不超时。 |

## 返回值

函数状态结果

| 参数名 | 类型 | 描述 |
| --- | --- | --- |
| - | Status | - SUCCESS：数据获取成功。<br>  - FAILED：数据获取失败。<br>  - 其他错误码请参考[UDF错误码](UDF错误码.md)。 |

## 约束说明

无
