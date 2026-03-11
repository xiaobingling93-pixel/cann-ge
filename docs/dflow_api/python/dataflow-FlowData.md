# dataflow.FlowData

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

DataFlow Graph中的数据节点，每个FlowData对应一个输入。

## 函数原型

```
FlowData(data_cls=Tensor, schema:Optional[TensorDesc]=None, name=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| data_cls | class | 当前只支持默认值的Tensor，表示FlowData接收Tensor类型的数据。 |
| schema | Optional[TensorDesc] | 对数据data_cls的描述，由于当前data_cls只支持Tensor，所以schema取值为TensorDesc。 |
| name | str | 节点名称，框架会自动保证名称唯一，不设置时会自动生成FlowData, FlowData_1, FlowData_2,...的名称。 |

## 返回值

正常场景下返回None。

返回raise DfException表示参数类型不正确。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
# 创建不指定数据类型和名称的输入节点，如果用户不指定name，框架将为其分配唯一的节点名称：FlowData FlowData_1
data = df.FlowData()
# 创建接受int32数据类型shape为[1]的输入节点
data = df.FlowData(schema=df.TensorDesc(df.DT_INT32, [1]))
# 创建不指定数据类型，指定名称为data0的输入节点
data = df.FlowData(name="data0")
# 创建数据类型为int32，shape为[1], 名称为data0的输入节点
data = df.FlowData(schema=df.TensorDesc(df.DT_INT32, [1]), name="data0")
```

## 约束说明

无
