# FeedDataFlowGraph（feed所有FlowMsg）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将数据输入到Graph图。

## 函数原型

```
Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<FlowMsgPtr> &inputs, int32_t timeout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graph_id | 输入 | 要执行图对应的ID。 |
| inputs | 输入 | 计算图输入FlowMsg的指针，为Host上分配的共享内存。 |
| timeout | 输入 | 数据输入超时时间，单位：ms，取值为-1时表示从不超时。 |

## 返回值

函数状态结果如下。

| 参数名 | 类型 | 描述 |
| --- | --- | --- |
| - | Status | - SUCCESS：数据输入成功。<br>  - FAILED：数据输入失败。<br>  - 其他错误码请参考[UDF错误码](UDF错误码.md)。 |

## 约束说明

无
