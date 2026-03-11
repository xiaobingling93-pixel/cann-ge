# set\_init\_param

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FuncProcessPoint的初始化参数。

## 函数原型

```
set_init_param(attr_name, attr_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| attr_name | str | 初始化参数名。 |
| attr_value | Union[str,List[str],int,List[int],List[List[int]],float,List[float],bool,List[bool],DType,List[DType]] | 初始化参数值。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
pp0.set_init_param("out_type", df.DT_INT32) # 按UDF实际实现来设置
```

## 约束说明

无
