# register

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

注册自定义类型对应的序列化、反序列化、计算size的函数，可结合feed，fetch接口使用，用于feed/fetch任意Python类型。

## 函数原型

```
register(msg_type, clz, serialize_func, deserialize_func, size_func=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| msg_type | int | 注册的类型ID。 |
| clz | 类型定义 | 类型定义，比如int，str，或者自定义的class。 |
| serialize_func | function | 序列化函数，输入是任意的Python对象，输出bytes类型的数据，即对象被序列化后的字节流。 |
| deserialize_func | function | 反序列化函数，输入类型为bytes，表示要反序列化的字节流，输出为被反序列化的对象。可以是任何Python对象类型。 |
| size_func | function | 计算序列化后内存大小的函数，单位字节，预留字段。 |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import cloudpickle
import dataflow as df

class TestClass():
    def __init__(self, name, val):
        self.name = name
        self.val = val

df.msg_type_register.register(1026, TestClass, lambda obj: cloudpickle.dumps(obj), lambda buffer: cloudpickle.loads(buffer))
```

## 约束说明

无
