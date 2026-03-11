# TensorDesc构造函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

构造TensorDesc。

## 函数原型

```
__init__(dtype: DataType, shape: Union[List[int], Tuple[int]])
```

## 参数说明

| 参数名 | 数据类型 | 取值说明 |
| --- | --- | --- |
| dtype | [DataType](DataType.md) | 表示Tensor数据类型。 |
| shape | Union[List[int], Tuple[int]] | 表示Tensor的shape的描述。 |

## 返回值

正常情况下返回TensorDesc的实例。

传入数据类型错误情况下会抛出TypeError或ValueError异常。

## 调用示例

```
from llm_datadist import TensorDesc, DataType
tensor_desc = TensorDesc(DataType.DT_FLOAT, [1,2])
```

## 约束说明

无
