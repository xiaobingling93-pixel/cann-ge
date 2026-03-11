# FeedRawData

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将原始数据输入到Graph图。

## 函数原型

```
struct RawData {
  const void *addr;
  size_t len;
};
```

```
Status FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,  const DataFlowInfo &info, int32_t timeout);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graph_id | 输入 | 要执行图对应的ID。 |
| raw_data_list | 输入 | 由输入数据指针和长度组成的数组，可以是1个也可以是多个，如果是多个，框架将自动把多个数据合并成一份数据传递给DataFlow图。 |
| index | 输入 | 对应的DataFlow图的某个输入。 |
| info | 输入 | 输入数据流标志（flow flag）。具体请参考[DataFlowInfo数据类型](DataFlowInfo数据类型.md)。 |
| timeout | 输入 | 数据输入超时时间，单位：ms，取值为-1时表示从不超时。 |

## 返回值

| 参数名 | 类型 | 描述 |
| --- | --- | --- |
| - | Status | - SUCCESS：数据输入成功。<br>  - FAILED：数据输入失败。<br>  - 其他错误码请参考[UDF错误码](UDF错误码.md)。 |

## 约束说明

无
