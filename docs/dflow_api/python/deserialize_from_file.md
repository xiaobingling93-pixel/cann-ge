# deserialize\_from\_file

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

从序列化的pickle文件进行反序列化恢复Python对象。

## 函数原型

```
deserialize_from_file(pkl_file, work_path=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| pkl_file | str | 序列化的pickle文件路径。 |
| work_path | str | 序列化时的工作路径。 |

## 返回值

反序列化恢复的Python对象。

## 调用示例

```
import dataflow as df
obj = df.msg_type_register.deserialize_from_file('file1')
```

## 约束说明

无
