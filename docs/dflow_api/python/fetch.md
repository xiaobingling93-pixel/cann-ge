# fetch

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取Graph输出数据。支持可序列化的任意的输出。

## 函数原型

```
fetch(indexes=None, timeout=-1)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| indexes | List[int] | 输出数据的索引，不可以重复。取值范围是【0~N-1】，N表示输出数据个数。支持一个或者多个。默认为None，表示取出所有输出。<br>示例：[0,2]，表示获取第一个和第三个输出数据。 |
| timeout | int | 数据提取超时时间，单位：ms，取值范围[0, 2147483647), 取值为-1时表示从不超时。 |

## 返回值

返回Tuple\[List\["Any"\], int\]。

正常场景下tuple最后一个返回值是0，表示success。

异常场景tuple最后一个返回值代表具体错误码。

## 调用示例

```
import dataflow as df
graph = df.FlowGraph(...)
graph.feed(...)
graph.fetch(...)
```

## 约束说明

使用非dataflow Tensor类型作为输入时需配合装饰器@pyflow以及装饰器@method进行使用。
