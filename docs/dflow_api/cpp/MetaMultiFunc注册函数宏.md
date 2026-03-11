# MetaMultiFunc注册函数宏

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

注册MetaMultiFunc的实现类。

## 函数原型

```
FLOW_FUNC_REGISTRAR(clazz)
```

>![](public_sys-resources/icon-note.gif) **说明：**
>该函数的使用示例如下：
>FLOW\_FUNC\_REGISTRAR\(UserFlowFunc\).RegProcFunc\("xxx\_func", &UserFlowFunc::Proc1\).RegProcFunc\("xxx\_func", &UserFlowFunc::Proc2\);

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| clazz | 输入 | MetaMultiFunc类实现的类名。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
