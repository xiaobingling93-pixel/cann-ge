# numpy

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

将Tensor转换到numpy的ndarray。

## 函数原型

```
numpy(copy=False)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| copy | bool | 默认值为False，表示从tensor转换到numpy.ndarray，且数据不做拷贝，如果取值为True，则表示需要对数据进行拷贝。<br>如果tensor是string类型的数据，该参数需要用户设置成True，否则会抛出异常。 |

## 返回值

返回numpy.array。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
tensor = Tensor(1)
array = tensor.numpy()
```

## 约束说明

无
