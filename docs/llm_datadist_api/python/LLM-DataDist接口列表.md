# LLM-DataDist接口列表

LLM-DataDist：大模型分布式集群和数据加速组件，提供了集群KV数据管理能力，以支持全量图和增量图分离部署。

- 支持的产品形态如下：
  - Atlas A2 推理系列产品
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品

- 当前仅支持Python3.9与Python3.11。安装方法请参考Python官网[https://www.python.org/](https://www.python.org/)。
- 最大注册50GB的Device内存。注册内存越大，占用的OS内存越多。

LLM-DataDist接口列表如下。

## LLM-DataDist

**表 1**  LLM-DataDist接口

| 接口名称 | 简介 |
| --- | --- |
| [LLMDataDist构造函数](LLMDataDist构造函数.md) | 构造LLMDataDist。 |
| [init](init.md) | 初始化LLMDataDist。 |
| [finalize](finalize.md) | 释放LLMDataDist。 |
| [link_clusters](link_clusters.md) | 建链。 |
| [unlink_clusters](unlink_clusters.md) | 断链。 |
| [check_link_status](check_link_status.md) | 调用此接口可快速检测链路状态是否正常。 |
| [kv_cache_manager](kv_cache_manager.md) | 获取KvCacheManager实例。 |
| [switch_role](switch_role.md) | 切换当前LLMDataDist的角色，建议仅在使用PagedAttention的场景使用。 |

## LLMConfig

**表 2**  LLMConfig接口

| 接口名称 | 简介 |
| --- | --- |
| [LLMConfig构造函数](LLMConfig构造函数.md) | 构造LLMConfig。 |
| [generate_options](generate_options.md) | 生成配置项字典。 |
| [device_id](device_id.md) | 设置当前进程Device ID，对应底层ge.exec.deviceId配置项。 |
| [sync_kv_timeout](sync_kv_timeout.md) | 配置拉取kv等接口超时时间，对应底层llm.SyncKvCacheWaitTime配置项。 |
| [enable_switch_role](enable_switch_role.md) | 配置是否支持角色平滑切换，对应底层llm.EnableSwitchRole配置项。 |
| [ge_options](ge_options.md) | 配置额外的GE配置项。 |
| [listen_ip_info](listen_ip_info.md) | PROMPT侧设置集群侦听信息，对应底层llm.listenIpInfo配置项。 |
| [mem_utilization](mem_utilization.md) | 配置ge.flowGraphMemMaxSize内存的利用率。默认值0.95。 |
| [buf_pool_cfg](buf_pool_cfg.md) | 用户指定内存档位配置，提高内存申请性能和使用率。 |

## KvCacheManager

**表 3**  KvCacheManager接口

| 接口名称 | 简介 |
| --- | --- |
| [KvCacheManager构造函数](KvCacheManager构造函数.md) | 介绍KvCacheManager构造函数。 |
| [is_initialized](is_initialized.md) | 查询KvCacheManager实例是否已初始化。 |
| [allocate_cache](allocate_cache.md) | 分配Cache，Cache分配成功后，会同时被cache_id与cache_keys引用，只有当这些引用都解除后，cache所占用的资源才会实际释放。 |
| [deallocate_cache](deallocate_cache.md) | 释放Cache。 |
| [remove_cache_key](remove_cache_key.md) | 移除CacheKey，仅当LLMRole为PROMPT时可调用。 |
| [pull_cache](pull_cache.md) | 根据CacheKey，从对应的Prompt节点拉取KV到本地KV Cache，仅当LLMRole为DECODER时可调用。 |
| [copy_cache](copy_cache.md) | 拷贝KV。 |
| [get_cache_tensors](get_cache_tensors.md) | 获取cache tensor。 |
| [allocate_blocks_cache](allocate_blocks_cache.md) | PagedAttention场景下，分配多个blocks的Cache。 |
| [pull_blocks](pull_blocks.md) | PagedAttention场景下，根据BlocksCacheKey，通过block列表的方式从对应的Prompt节点拉取KV到本地KV Cache，仅当LLMRole为DECODER时可调用。 |
| [copy_blocks](copy_blocks.md) | PagedAttention场景下，拷贝KV。 |
| [swap_blocks](swap_blocks.md) | 对cpu_cache和npu_cache进行换入换出。 |
| [transfer_cache_async](transfer_cache_async.md) | 异步分层传输KV Cache。 |

## KvCache

**表 4**  KVCache接口

| 接口名称 | 简介 |
| --- | --- |
| [KvCache构造函数](KvCache构造函数.md) | 构造KVCache。 |
| [cache_id](cache_id.md) | 获取KvCache的id。 |
| [cache_desc](cache_desc.md) | 获取KvCache描述。 |
| [per_device_tensor_addrs](per_device_tensor_addrs.md) | 获取KvCache的地址。 |
| [create_cpu_cache](create_cpu_cache.md) | 创建cpu cache。 |

## LLMClusterInfo

**表 5**  LLMClusterInfo接口

| 接口名称 | 简介 |
| --- | --- |
| [LLMClusterInfo构造函数](LLMClusterInfo构造函数.md) | 构造LLMClusterInfo。 |
| [remote_cluster_id](remote_cluster_id.md) | 设置对端集群ID。 |
| [append_local_ip_info](append_local_ip_info.md) | 添加本地集群IP信息。 |
| [append_remote_ip_info](append_remote_ip_info.md) | 添加远端集群IP信息。 |

## CacheTask

**表 6**  CacheTask

| 接口名称 | 简介 |
| --- | --- |
| [CacheTask构造函数](CacheTask构造函数.md) | 构造CacheTask。 |
| [synchronize](synchronize.md) | 等待所有层传输完成，并获取整体执行结果。 |
| [get_results](get_results.md) | 等待所有层传输完成，并获取每个TransferConfig对应执行结果。 |

## 其他

**表 7**  其他

| 接口名称 | 简介 |
| --- | --- |
| [LLMRole](LLMRole.md) | LLMRole的枚举值。 |
| [Placement](Placement.md) | CacheDesc的字段，表示cache所在的设备类型。 |
| [CacheDesc](CacheDesc.md) | 构造CacheDesc。 |
| [CacheKey](CacheKey.md) | 构造CacheKey。 |
| [CacheKeyByIdAndIndex](CacheKeyByIdAndIndex.md) | 构造CacheKeyByIdAndIndex，通常在[pull_cache](pull_cache.md)接口中作为参数类型使用。 |
| [BlocksCacheKey](BlocksCacheKey.md) | PagedAttention场景下，构造BlocksCacheKey。 |
| [LayerSynchronizer](LayerSynchronizer.md) | 等待模型指定层执行完成，用户需要继承LayerSynchronizer并实现该接口。<br>该接口会在执行KvCacheManager.transfer_cache_async时被调用，当该接口返回成功，则开始当前层cache的传输。 |
| [TransferConfig](TransferConfig.md) | 构造TransferConfig。 |
| [TransferWithCacheKeyConfig](TransferWithCacheKeyConfig.md) | 构造TransferWithCacheKeyConfig。 |
| [LLMException](LLMException.md) | 获取异常的错误码。错误码列表详见[LLMStatusCode](LLMStatusCode.md)。 |
| [LLMStatusCode](LLMStatusCode.md) | LLMStatusCode的枚举值。 |
| [DataType](DataType.md) | DataType的枚举类。 |
