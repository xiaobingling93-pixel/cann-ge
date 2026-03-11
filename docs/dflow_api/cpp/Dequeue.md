# Dequeue

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置均衡分发亲和性。

## 函数原型

```
virtual int32_t Dequeue(std::shared_ptr<FlowMsg> &flowMsg, int32_t timeout = -1) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flowMsg | 输出 | 出队的FlowMsg。 |
| timeout | 输入 | 出队超时时间，单位为ms，默认值为-1，表示一直阻塞当前线程，直到队列中有元素可以取出。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理

无。

## 约束说明

无。
