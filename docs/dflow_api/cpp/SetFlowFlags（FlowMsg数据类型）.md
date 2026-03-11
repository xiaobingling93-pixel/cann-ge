# SetFlowFlags（FlowMsg数据类型）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置FlowMsg消息头中的flags。

## 函数原型

```
void SetFlowFlags(uint32_t flags)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flags | 输入 | 消息头中的flags标志。<br>flags的枚举值如下：<br>enum class FlowFlag : uint32_t {<br>   FLOW_FLAG_EOS = (1U << 0U),  // 数据流结束标志<br>   FLOW_FLAG_SEG = (1U << 1U)  // 非连续数据的分段标志<br>}; |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
