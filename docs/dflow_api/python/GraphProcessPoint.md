# GraphProcessPoint

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

GraphProcessPoint构造函数，返回一个GraphProcessPoint对象。

## 函数原型

```
GraphProcessPoint(framework, graph_file, load_params={}, compile_config_path="", name=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| framework | Framework | IR文件的框架类型，详见[dataflow.Framework](dataflow-Framework.md)。 |
| graph_file | str | IR文件路径。 |
| load_params | Dict[str, str] | 配置参数map映射表，key为参数类型，value为参数值，均为String格式，用于描述原始模型解析参数。 |
| compile_config_path | str | 编译graph时的配置文件路径。<br>配置文件实例如下：<br>{"build_option":{},"inputs_tensor_desc":[{"data_type":"DT_UINT32","shape":[3]},{"data_type":"DT_UINT32","shape":[3]}]} |
| name | str | 处理点名称，框架会自动保证名称唯一，不设置时会自动生成GraphProcessPoint, GraphProcessPoint_1, GraphProcessPoint_2,...的名称。 |

**表 1**  GraphProcessPoint的json配置文件

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| build_options | 可选 | 值为map<string, string>, 有需要设置时参考Ascend Graph中的build_options。 |
| inputs_tensor_desc | 可选 | 值为list，Graph的输入节点，list元素为tensor的描述。 |
| inputs_tensor_desc.data_type | 可选 | 字符串类型。<br>取值为Graph中的data_type对应字符串。 |
| inputs_tensor_desc.shape | 可选 | 值为整数类型的列表。<br>取值为Graph中的shape。 |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](DataFlow错误码.md)。

## 调用示例

```
import dataflow as df
pp1 = df.GraphProcessPoint(...)
```

## 约束说明

无
