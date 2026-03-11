# reshape

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

对tensor进行Reshape操作，不改变tensor的内容。

## 函数原型

```
def reshape(self, shape: Union[List[int], Tuple[int]])
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | 要改变的目标shape。<br>要求shape元素个数必须和原来shape的个数一致。 |

## 返回值

- 0：SUCCESS
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无

## 约束说明

如果对输入进行reshape动作，可能会影响其他使用本输入的节点正常执行。
