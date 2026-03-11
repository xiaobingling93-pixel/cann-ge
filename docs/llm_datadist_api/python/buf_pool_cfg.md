# buf\_pool\_cfg

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

用户指定内存档位配置，提高内存申请性能和使用率。

## 函数原型

```
buf_pool_cfg(buf_pool_cfg)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| buf_pool_cfg | string | json数组格式字符串，包含total_size blk_size和max_buf_size三个节点 |

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| total_size | 必选 | 当前档位内存池的大小，单位Byte<br>约束：<br>total_size是2M的倍数，且total_size是blk_size的倍数，最大值不应超过0xFFFFFFFF。 |
| blk_size | 必选 | 当前档位一次可以申请的最小内存值，单位Byte<br>约束：<br>要求满足2^n，且在(0,2M]之间，小于或等于max_buf_size |
| max_buf_size | 必选 | 当前档位一次可以申请的最大内存值，单位Byte<br>约束：小于total_size。如果设置有多个档位，按照档位出现的先后顺序，max_buf_size必须保持严格递增。 |

## 调用示例

```
from llm_datadist import LLMConfig
llm_config = LLMConfig()
llm_config.buf_pool_cfg= '{"buf_cfg": [{"total_size":2097152,"blk_size":256,"max_buf_size":8192},{"total_size": 10485760,"blk_size": 8192,"max_buf_size": 8388608},{"total_size": 69206016,"blk_size": 8192,"max_buf_size": 67108864}]}'
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
