# SetInputsAlignAttrs

## 产品支持情况


| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowGraph中的输入对齐属性。

## 函数原型

```
FlowGraph &SetInputsAlignAttrs(uint32_t align_max_cache_num, int32_t align_timeout, bool dropout_when_not_align = false)
```

## 参数说明


| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| align_max_cache_num | 输入 | 数据对齐最大缓存数量，默认为0，表示不开启数据对齐功能，取值>0表示开启，最大值为1024。<br>每个缓存表示一组输入。 |
| align_timeout | 输入 | 每组数据对齐等待超时时间，单位ms。<br>-1表示永不超时，配置需要大于0并不超过600 * 1000ms(10分钟)。 |
| dropout_when_not_align | 输入 | 超时或超过缓存最大数之后没有对齐的数据是否要丢弃。<br><br>  - true：是<br>  - false：否<br><br>默认为false。 |

## 返回值

返回设置了对齐属性的FlowGraph图。

## 异常处理

无。

## 约束说明

无。

