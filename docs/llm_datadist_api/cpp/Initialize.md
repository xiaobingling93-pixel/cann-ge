# Initialize

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 推理系列产品 | √ |
| Atlas A2 训练系列产品 | x |

## 函数功能

初始化LLM-DataDist。

## 函数原型

```
Status Initialize(const std::map<AscendString, AscendString> &options)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| options | 输入 | 初始化参数值。具体请参考[表1](#table8396338161010)。 |

**表 1**  options

| 参数名 | 可选/必选 | 描述 |
| --- | --- | --- |
| OPTION_LISTEN_IP_INFO | Pormpt必选 | 表示Device侧的IP地址和端口。<br>配置示例：如"192.168.1.1:26000"，单进程多卡场景，传入多个时使用英文分号分割。 |
| OPTION_DEVICE_ID | 必选 | 设置当前进程的Device ID，如"0"，单进程多卡场景，传入多个时使用英文分号分割。 |
| OPTION_SYNC_CACHE_WAIT_TIME | 可选 | kv相关操作的超时时间，单位：ms。不配置默认为1000ms。相关接口如下。<br><br>  - [AllocateCache](AllocateCache.md)<br>  - [DeallocateCache](DeallocateCache.md)<br>  - [PullKvCache](PullKvCache.md)<br>  - [PullKvBlocks](PullKvBlocks.md)<br>  - [CopyKvCache](CopyKvCache.md)<br>  - [CopyKvBlocks](CopyKvBlocks.md)<br>  - [PushKvCache](PushKvCache.md)<br>  - [PushKvBlocks](PushKvBlocks.md) |
| OPTION_BUF_POOL_CFG | 可选 | 配置内存池大小与档位信息，提高内存申请性能和使用率。格式为json string，字段含义见[表2](#table75421185315)，使用示例如下所示。<br>{<br>"buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192}],<br>"buf_pool_size": 2147483648<br>} |
| OPTION_ENABLE_SET_ROLE | 可选 | 配置是否支持角色平滑切换。取值如下。<br><br>  - 1：支持。<br>  - 0：不支持，不配置默认为不支持。<br><br>相关接口：[SetRole](SetRole.md)。 |

**表 2**  OPTION\_BUF\_POOL\_CFG配置

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| buf_cfg | 可选 | 内存池档位配置，详见[表3](#table146612414525)。 |
| buf_pool_size | 可选 | 内存池大小，单位为byte。 |

**表 3**  buf\_cfg配置

| 配置项 | 可选/必选 | 描述 |
| --- | --- | --- |
| total_size | 必选 | 当前档位内存池的大小，单位byte。<br> 说明： total_size是2M的倍数，且total_size是blk_size的倍数，最大值不应超过0xFFFFFFFF。 |
| blk_size | 必选 | 当前档位一次可以申请的最小内存值，单位byte。<br> 说明： 要求满足2^n，且在(0,2M]之间，小于或等于max_buf_size。 |
| max_buf_size | 必选 | 当前档位一次可以申请的最大内存值，单位byte。<br> 说明： 小于total_size，max_buf_size必须保持严格递增。 |

## 返回值

- LLM\_SUCCESS：成功
- LLM\_PARAM\_INVALID：参数错误
- 其他：失败

## 异常处理

无

## 约束说明

需要和[Finalize](Finalize.md)配对使用，初始化成功后，任何退出前都需要调用[Finalize](Finalize.md)保证资源释放，否则会出现资源释放顺序不符合预期而导致问题。
