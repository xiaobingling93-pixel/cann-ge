# set\_attr

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowNode的属性。

## 函数原型

```
set_attr(attr_name, attr_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| attr_name | str | 属性名称<br>当前支持如下三种：<br><br>  - _flow_attr：BOOL类型属性，可以配置为true或false，true表示存在flow属性，false表示不存在flow属性。<br>  -_flow_attr_depth：INT类型属性，指定了队列的深度，范围：大于等于2，若不配置，默认深度为128。<br>  - _flow_attr_enqueue_policy：STRING类型属性，指定的入队的策略，仅支持"FIFO"和"OVERWRITE"，"FIFO"表示队列数据顺序入队，队列满的时候会等待dequeue，"OVERWRITE"表示入队不会等待，数据会循环覆盖，若不配置，默认策略为FIFO。 |
| attr_value | Union[bool, str, int] | 属性值。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```
import dataflow as df
pp = df.FuncProcessPoint(...)
flow_node = df.FlowNode(...)
flow_node.set_attr("_flow_attr",True)
flow_node.set_attr("_flow_attr_depth",5)
flow_node.set_attr("_flow_attr_enqueue_policy","FIFO")
```

## 约束说明

通过\_flow\_attr\_depth和\_flow\_attr\_enqueue\_policy设置depth和policy需要先使能\_flow\_attr，否则不会生效。
