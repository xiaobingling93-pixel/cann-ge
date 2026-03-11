# SetAffinityPolicy

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 函数功能

设置均衡分发亲和性。

## 函数原型

```
void SetAffinityPolicy(AffinityPolicy affinityPolicy) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| affinityPolicy | 输入 | 亲和性。<br>enum class AffinityPolicy : int32_t {<br>NO_AFFINITY = 0,  // 不需要亲和<br>ROW_AFFINITY = 1,  // 按行亲和<br>COL_AFFINITY = 2,  //  按列亲和<br>}; |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
