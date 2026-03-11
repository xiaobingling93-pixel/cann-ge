# method

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

对于复杂场景支持将类作为pipeline任务在本地或者远端运行。为此，需使用@pyflow装饰类，同时使用@method装饰类中的函数，以表达需要使用pipeline方式运行此函数，支持一个类中存在多个被@method修饰的函数，以表达可同时接受输入进行执行，同时被@method修饰的函数均需要参与进行构造FlowGraph图。不直接作为pipeline执行的函数不能使用@method进行装饰，比如内部函数。

## 函数原型

```
装饰器@method
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| num_returns | int | 装饰器装饰函数时，用于表示函数的输出个数，不设置该参数时默认函数返回一个返回值。该参数与使用type annotations方式标识函数返回个数与类型的方式选择其一即可。 |
| stream_input | str | 用于表示当前func的输入为流式输入（即函数入参为队列），当前只支持"Queue"类型，用户可自行从输入队列中取数据。 |
| choice_output | function | 表示当前func为可选输出，只有满足条件的输出才会返回（条件为用户自定义的function）。例如：<br>choice_output=lambda e: e is not None<br>该例子表示只有非None的输出才会返回。 |

## 返回值

正常场景下返回被装饰的函数。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
@df.pyflow
class Foo():
    def __init__(self):
        pass
    # 使用num_returns表达输出个数为2
    @df.method(num_returns=2)
    def func1(a, b):
        return a + b,a - b
    # 使用typing表达输出个数为2
    @df.method()
    def func2(a, b) -> Tuple[int, int]:
        return a + b,a - b
    # 默认返回1个
    @df.method()
    def func3(a, b):
        return a + b

    @df.method(stream_input='Queue')
    def func4(a, b):
        data1 = a.get()
        data2 = a.get()
        data3 = b.get()
        return data1 + data2 + data3

    @df.method(choice_output=lambda e: e is not None)
    def func5(self, a) -> Tuple[int, int]:
        return None, a  # 根据lambda函数将非空值才送到相应输出
```

## 约束说明

环境需安装对应Python版本的cloudpickle包。

被@method修饰的函数必须要参与构图过程，@pyflow修饰的类构图过程自己的输出不能再作为自己的输入，如果函数存在默认值，构图时仍然要求连边。

流式输入场景下DataFlow框架不支持数据对齐和异常事务处理。
