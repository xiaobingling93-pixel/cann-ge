# dataflow.TensorDesc

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

Tensor的描述函数。

## 函数原型

```
TensorDesc(dtype: DType, shape: Union[List[int], Tuple[int]])
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| dtype | DType | 表示[Tensor](dataflow-Tensor.md)的数据类型的描述。 |
| shape | Union[List[int], Tuple[int]] | 表示Tensor的shape的描述，shape中的每个元素需要>=0的数值。 |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
tensor_desc = TensorDesc(DT_FLOAT, [1,2])
```

## 约束说明

无
