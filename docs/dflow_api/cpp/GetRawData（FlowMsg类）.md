# GetRawData（FlowMsg类）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取RawData类型的数据对应的数据指针和数据大小。

## 函数原型

```
virtual Status GetRawData(void *&dataPtr, uint64_t &dataSize) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dataPtr | 输出 | RawData的数据指针 |
| dataSize | 输出 | RawData的数据大小 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考错误码。

## 异常处理

非RawData类型的数据FLOW\_FUNC\_FAILED。

## 约束说明

无。
