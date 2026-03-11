# GetAttr（MetaParams类，获取属性值）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据属性名获取对应的属性值。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。

## 函数原型

```
int32_t GetAttr(const char *attrName, T &value) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| attrName | 输入 | 属性名。 |
| value | 输出 | 属性值。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无。

## 约束说明

无。
