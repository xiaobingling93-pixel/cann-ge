# SetCompileConfig（GraphPp类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置GraphPp的json配置文件路径和文件名。配置文件用于AscendGraph的描述和编译。

返回设置好的GraphPp。

## 函数原型

```
GraphPp &SetCompileConfig(const char_t *json_file_path)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| json_file_path | 输入 | GraphPp的json配置文件路径和文件名。<br>GraphPp的json配置文件用于AscendGraph的描述和编译。<br>示例如下，参数解释请参考[表1](#zh-cn_topic_0000001411032876_table1179952915232)。<br>{"build_option":{},"inputs_tensor_desc":[{"data_type":"DT_UINT32","shape":[3]},{"data_type":"DT_UINT32","shape":[3]}]} |

**表 1**  GraphPp的json配置文件

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| build_option | 可选 | 值为map<string, string>, 有需要设置时参考Ascend Graph中的build_option。 |
| inputs_tensor_desc | 可选 | 值为list，Ascend Graph的输入节点，list元素为Tensor的描述。 |
| inputs_tensor_desc.data_type | 可选 | 字符串类型。<br>取值为Ascend Graph中的data_type对应字符串。 |
| inputs_tensor_desc.shape | 可选 | 值为整数类型的列表<br>取值为Ascend Graph中的shape。 |

## 返回值

返回设置好的GraphPp。

## 异常处理

无。

## 约束说明

无。
