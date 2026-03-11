# get\_nowait

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

无等待地获取队列中的元素，功能等同于get\(block=False\)。

## 函数原型

```
get_nowait(self)
```

## 参数说明

无

## 返回值

MsgType中所对应类型的数据对象。

## 异常处理

队列为空时会抛出queue.Empty异常。

## 约束说明

无
