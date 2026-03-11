# SetGraphPpBuilderAsync

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowGraph中的GraphPp的Builder是否异步执行。

## 函数原型

```
void SetGraphPpBuilderAsync(bool graphpp_builder_async)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graphpp_builder_async | 输入 | FlowGraph中的GraphPp的Builder是否异步执行。取值如下：<br><br>  - true：是<br>  - false：否<br><br>默认值：false |

## 返回值

无

## 异常处理

无。

## 约束说明

无。
