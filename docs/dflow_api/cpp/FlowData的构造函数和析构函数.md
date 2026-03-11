# FlowData的构造函数和析构函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

FlowData构造函数和析构函数，构造函数会返回一个FlowData节点。

## 函数原型

```
FlowData(const char *name, int64_t index)
~FlowData() override
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 输入节点名称，需要全图唯一。 |
| index | 输入 | 输入节点index，全图中输入index需要从0开始累加，表示图中输入的顺序。 |

## 返回值

返回一个FlowData节点。

## 异常处理

无。

## 约束说明

无。
