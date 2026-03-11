# set\_balance\_gather

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置节点balance gather属性，具有balance gather属性的UDF可以使用balance options设置负载均衡亲和输出。

## 函数原型

```
set_balance_gather(self) -> None
```

## 参数说明

无

## 返回值

返回None。

## 调用示例

```
import dataflow as df
flow_node = df.FlowNode(...)
flow_node.set_balance_gather()
```

## 约束说明

与[set\_balance\_scatter](set_balance_scatter.md)接口互斥。
