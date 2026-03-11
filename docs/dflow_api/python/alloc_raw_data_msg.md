# alloc\_raw\_data\_msg

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据输入的size申请一块连续内存，用于承载raw data类型的FlowMsg。

## 函数原型

```
alloc_raw_data_msg(self, size, align:Optional[int] = 64) -> FlowMsg
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| size | 输入 | 申请内存大小，单位字节。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】，默认值为64。 |

## 返回值

正常返回FlowMsg的实例。申请失败返回None。

## 异常处理

无

## 约束说明

无
