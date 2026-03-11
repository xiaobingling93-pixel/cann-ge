# dataflow.get\_running\_instance\_id

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

UDF执行时获取当前UDF的运行实例ID，该信息来源于data\_flow\_deploy\_info.json中的logic\_device\_list配置。

## 函数原型

```
get_running_instance_id()
```

## 参数说明

无

## 返回值

返回当前UDF的运行实例ID，数据类型为int。

## 调用示例

```
import dataflow as df

@df.pyflow
def func():
    print('running instance_id = ', df.get_running_instance_id())
    return a + b
```

## 约束说明

需配合pyflow装饰器进行使用。
