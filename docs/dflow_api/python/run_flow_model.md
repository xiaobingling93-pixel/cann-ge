# run\_flow\_model

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

同步执行指定的模型。

## 函数原型

```
run_flow_model(self, model_key:str, input_msgs: List[FlowMsg], timeout: int) -> Tuple[int, FlowMsg]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| model_key | 输入 | 指定的模型key。当使用C++开发DataFlow时，与AddInvokedClosure中指定的name一致。 |
| input_msgs | 输入 | 提供给模型的输入。 |
| timeout | 输入 | 等待模型执行超时时间，单位ms，-1表示永不超时。 |

## 返回值

- 0：SUCCESS
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无

## 约束说明

无
