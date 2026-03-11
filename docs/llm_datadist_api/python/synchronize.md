# synchronize

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

等待所有层传输完成，并获取整体执行结果。

## 函数原型

```
synchronize(timeout_in_millis: Optional[int] = None) -> LLMStatusCode
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| timeout_in_millis | Optional[int] | 等待超时时间，单位为毫秒，默认为None，表示不超时。 |

## 调用示例

```
ret = cache_task.synchronize()
```

## 返回值

正常情况下返回[LLMStatusCode](LLMStatusCode.md)。

传入数据类型错误情况下会抛出TypeError或ValueError异常。

## 约束说明

无
