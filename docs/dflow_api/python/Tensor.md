# Tensor

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

Tensor的构造函数。

## 函数原型

```
Tensor(data, *, tensor_desc: TensorDesc=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| data | Any | tensor的数据，可以是Python的list，也可以是numpy。 |
| tensor_desc | TensorDesc | 表示tensor的描述，当指定时，如果传入的data的dtype与tensor_desc的dtype不一致，会尝试强转到tensor_desc的dtype的类型，如果传入的data的shape与tensor_desc的shape不一致，会报错。 |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
tensor = Tensor([1])
tensor = Tensor(numpy.array([1]))
tensor = Tensor([1], tensor_desc=TensorDesc(dataflow.DT_FLOAT, [1]))
```

## 约束说明

无
