# dataflow.TimeBatch

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

TimeBatch功能是基于UDF为前提的。

正常模型每次处理一个数据，当需要一次处理一批数据时，就需要将这批数据组成一个batch。最基本的batch方式是将这批N个数据直接拼接，然后shape前加N，而某些场景需要将某段或者某几段时间数据组成一个batch，并且按特定的维度拼接，则可以通过使用TimeBatch功能来组batch。

在ASR\(Automatic Speech Recognition\)自动语音识别场景下，存在按定长时间段组batch或按时间分段（时间不连续）组整批batch两种诉求，可以通过TimeBatch实现。

## 函数原型

```
TimeBatch(time_window=0, batch_dim=0, drop_remainder=False)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| time_window | int64_t | 整型(单位ms)，当值>0时表示按该时间窗来组batch，当值为-1时表示按时间分段来组batch，其它值报错。 |
| batch_dim | int64_t | 只有设置了time_window时，该参数才生效。取值范围[-1,shape维度]。<br><br>  - 默认为-1表示数据输出shape会在第0维添加一个batch维。<br>  - shape维度>batch_dim>=0时表示按某个维度组batch。<br>  - batch_dim>shape维度或者<-1时报错。 |
| drop_remainder | Bool | 只有设置了time_window时，该参数才生效。<br>仅在time_window>0时生效，选择不足time_window时是否丢弃，默认false不丢弃。true则丢弃。举例如下：<br>假如time_window=5ms，输入数据时长为3ms，则：<br><br>  - drop_remainder不配置或者配置为false时，不丢弃输入数据。<br>  - drop_remainder配置为true时如果输入数据未携带EOS或者SEG，会一直等待，不丢弃数据。如果输入数据只携带了SEG，则丢弃数据。如果输入数据携带了EOS标记，则丢弃输入数据，只传递EOS标记。<br>  - 如果输入数据未携带EOS或者SEG，会一直等待，不丢弃数据。<br>  - 如果输入数据只携带了SEG，则丢弃数据。<br>  - 如果输入数据携带了EOS标记，则丢弃输入数据，只传递EOS标记。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
# 按需设置time_batch中的各个属性值
time_batch = df.TimeBatch()
time_batch.time_window = 5
time_batch.batch_dim = 0
# 通过FlowNode的map_input接口使用
df.FlowNode(...).map_input(..., [time_batch])
```

## 约束说明

当前Batch特性无法做负荷分担，因此如果使用2P环境，需要在ge初始化时添加\{"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"\}, \{"ge.exec.logicalDeviceId", "\[0:0\]"\}。其中logicalDeviceId可以是\[0:0\]，也可以是\[0:1\]。logicalDeviceId解释如下。

logical\_device\_cluster\_deploy\_mode为SINGLE时，用于指定模型部署在某个指定的设备上。

配置格式：\[node\_id:device\_id\]

- node\_id：昇腾AI处理器逻辑ID，从0开始，表示资源配置文件中第几个设备。
- device\_id：昇腾AI处理器物理ID。
