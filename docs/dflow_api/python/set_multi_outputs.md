# set\_multi\_outputs

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

批量设置指定index的output的tensor。

## 函数原型

```
set_multi_outputs(self, index, outputs: List[Union[FlowMsg, np.ndarray, fw.FlowMsg]], balance_config: BalanceConfig) -> int
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| index | 输入 | 指定输出的index。 |
| output | 输入 | 指定输出的Msg。 |
| balance_config | 输入 | 输出均衡分发相关配置。 |

## 返回值

- 0：SUCCESS
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无

## 约束说明

无
