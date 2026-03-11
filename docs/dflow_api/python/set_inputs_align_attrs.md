# set\_inputs\_align\_attrs

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowGraph中的输入对齐属性。

## 函数原型

```
set_inputs_align_attrs(self, align_max_cache_num: int, align_timeout: int, dropout_when_not_align: bool = False)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| align_max_cache_num | int | 数据对齐最大缓存数量，默认为0，表示不开启数据对齐功能，取值>0表示开启，最大值为1024。<br>每个缓存表示一组输入。 |
| align_timeout | int | 每组数据对齐等待超时时间，单位ms。<br>-1表示永不超时，配置需要大于0并不超过600*1000ms(10分钟)。 |
| dropout_when_not_align | bool | 超时或超过缓存最大数之后没有对齐的数据是否要丢弃。<br><br>  - True：是<br>  - False：否<br><br>默认为False。 |

## 返回值

无。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
graph.set_inputs_align_attrs(256, 600 * 1000, False)
```

## 约束说明

无
