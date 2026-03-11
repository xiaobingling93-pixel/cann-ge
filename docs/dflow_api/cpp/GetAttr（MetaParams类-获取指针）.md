# GetAttr（MetaParams类，获取指针）

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

根据属性名获取AttrValue类型的指针。该函数供[Init（MetaFlowFunc类）](Init（MetaFlowFunc类）.md)调用。

## 函数原型

```
std::shared_ptr<const AttrValue> GetAttr(const char  *attrName) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| attrname | 输入 | 属性名。 |

## 返回值

获取到的AttrValue类型的指针。

## 异常处理

无。

## 约束说明

无。
