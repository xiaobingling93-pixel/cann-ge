# GetProcessPointType

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取ProcessPoint的类型。

## 函数原型

```
ProcessPointType GetProcessPointType() const
```

## 参数说明

无

## 返回值

返回一个ProcessPoint的类型。类型取值如下：

```
enum class ProcessPointType {
FUNCTION = 0,
GRAPH = 1,
INNER = 2,
FLOW_GRAPH = 3,
INVALID = 4,
}; 
```

## 异常处理

无。

## 约束说明

无。
