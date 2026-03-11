# CountBatch

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能介绍

CountBatch功能是指基于UDF为计算处理点将多个数据按batch\_size组成batch。

该功能应用于DataFlow异步场景，具体如下。

- 长时间没有数据输入时，可以通过CountBatch功能设置超时时间，如果没有设置padding，超时后取当前已有数据送计算处理点处理。
- 设置超时时间后，如果数据不满batch\_size时，可以通过CountBatch功能设置padding属性，计算点根据padding设置对数据进行填充到batch\_size后输出。

## 使用方法

用户在DataFlow构图时通过给FlowNode的输入设置属性来添加CountBatch功能。示例如下。

```
CountBatch count_batch = {};
// 按需求设置count_batch中各属性的值
count_batch.timeout = 10;
count_batch.batch_size = 300;
count_batch.padding = true;
count_batch.slide_stride = 5; 
DataFlowInputAttr flow_attr = {DataFlowAttrType::COUNT_BATCH, &count_batch};
std::vector<DataFlowInputAttr> flow_attrs = {flow_attr};
// 通过FlowNode的MapInput设置
FlowNode::MapInput(xx, xx, xx, flow_attrs);
```

| 属性名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| batch_size | int64_t | 组batch大小。 |
| timeout | int64_t | 只有设置了batch_size时，该参数才生效。<br>组batch等待时间，单位（ms），取值范围[0,4294967295)，默认值是0，表示一直等待直到满batch。 |
| padding | bool | 只有设置了batch_size和timeout时，该参数才生效。<br>不足batch时，是否padding。默认值false，表示不padding。 |
| slide_stride | int64_t | 只有设置了batch_size时，该参数才生效。<br>滑窗步长，取值范围[0,batch_size]。<br><br>  - >0且<batch_size时表示启用滑窗方式组batch。<br>  - 不设置，等于0。<br>  - 等于batch_size时按照未设置滑窗步长方式组batch。<br>  - >batch_size报错。 |
| batch_dim | int64_t | 未使能。 |
| flag | int32_t | 未使能。 |
| drop_remainder | bool | 未使能。 |

## 使用注意事项

当前Batch特性无法做负荷分担，因此如果使用2P环境，需要在ge初始化时添加\{"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"\}, \{"ge.exec.logicalDeviceId", "\[0:0\]"\}。
