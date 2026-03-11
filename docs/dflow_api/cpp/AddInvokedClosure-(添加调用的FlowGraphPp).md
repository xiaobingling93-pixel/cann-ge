# AddInvokedClosure \(添加调用的FlowGraphPp\)

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

添加FunctionPp调用的FlowGraphPp，返回添加好的FunctionPp。

## 函数原型

```
FunctionPp &AddInvokedClosure(const char_t *name, const FlowGraphPp &graph_pp)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 调用的FlowGraphPp的唯一标识，需要全图唯一。 |
| graph_pp | 输入 | 调用的FlowGraphPp。 |

## 返回值

返回设置好的FunctionPp。

## 异常处理

无。

## 约束说明

无。
