# get\_attr\_int\_list

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

获取指定名称的int数组类型属性值。

## 函数原型

```
get_attr_int_list(self, name: str) -> Tuple[int, List[int]]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 属性名。 |

## 返回值

获取返回码及int数组类型的属性值。

- 如果该属性存在，返回的Tuple中第一个元素为FLOW\_FUNC\_SUCCESS，第二个元素为int类型数组的属性值。
- 如果属性不存在，Tuple中仅包含错误码一个元素。

## 异常处理

无

## 约束说明

无
