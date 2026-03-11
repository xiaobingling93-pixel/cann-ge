# GraphPp构造函数和析构函数

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

GraphPp构造函数和析构函数，构造函数会返回一个GraphPp对象。

## 函数原型

```
GraphPp(const char_t *pp_name, const GraphBuilder &builder)
~GraphPp() override
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pp_name | 输入 | GraphPp的名称，需要全图唯一。 |
| builder | 输入 | GE Graph的构建方法：std::function<ge::Graph()><br>GE Graph的构建具体请参考《图模式开发指南》。 |

## 返回值

返回一个GraphPp对象。

## 异常处理

无。

## 约束说明

无。
