# pyflow

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

支持将函数作为pipeline任务在本地或者远端运行。为此，需使用@pyflow装饰函数，以表达需要使用pipeline方式运行此函数。当用户的类或者函数被@pyflow装饰后，会自动添加fnode的构图方法，使用方式参考[调用示例](#section17821439839)。

## 函数原型

```
装饰器@pyflow
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| num_returns | int | 装饰器装饰函数时，用于表示函数的输出个数，不设置该参数时默认函数返回一个返回值。该参数与使用type annotations方式标识函数返回个数与类型的方式选择其一即可。 |
| resources | dict | 用于标识当前func需要的资源信息，支持memory、num_cpus和num_npus。memory单位为M; num_npus表示需要使用npu资源数量，为预留参数，当前仅支持1。例如：{"memory": 100, "num_cpus": 1, "num_npus": 1} |
| stream_input | str | 用于表示当前func的输入为流式输入（即函数入参为队列），当前只支持"Queue"类型，用户可自行从输入队列中取数据。 |
| choice_output | function | 表示当前func为可选输出，只有满足条件的输出才会返回（条件为用户自定义的function）。例如：<br>choice_output=lambda e: e is not None<br>该例子表示只有非None的输出才会返回。 |
| visible_device_enable | bool | 开启后，UDF进程会根据用户配置num_npus资源自动设置ASCEND_RT_VISIBLE_DEVICES，调用get_running_device_id接口获取对应的逻辑ID，当前num_npus仅支持1，因此该场景下get_running_device_id结果为0。 |
| env_hook_func | function | 此钩子函数用于给用户自行扩展在Python UDF初始化之前必要的Python环境准备或import操作。<br>钩子函数仅支持无输入无输出类型。 |

## 返回值

装饰后的类或者函数。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
# current is udf.py
import dataflow as df
@df.pyflow(num_returns=2, resources={"memory": 100, "num_cpus": 1})
def func1(a, b):
    return a + b,a - b

@df.pyflow
def func2(a, b):
    return a + b

@df.pyflow(stream_input='Queue')
def func3(a, b):
    data1 = a.get()
    data2 = a.get()
    data3 = b.get()
    return data1 + data2 + data3

@df.pyflow(choice_output=lambda e: e is not None)
def func4(self, a) -> Tuple[int, int]:
    return None, a  # 根据lambda函数将非空值传给相应输出

# current is graph.py
from UDF import func2
import dataflow as df

# 构图
# 定义输入
data0 = df.FlowData()
data1 = df.FlowData()
# 使用func2自动生成的fnode方法构图
func_out = func2.fnode()(data0, data1)

# 构建FlowGraph
dag = df.FlowGraph([func_out])
```

## 约束说明

环境需安装对应Python版本的cloudpickle包。

流式输入场景下DataFlow框架不支持数据对齐和异常事务处理。
