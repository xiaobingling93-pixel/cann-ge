# LLM-DataDist接口列表

LLM-DataDist：大模型分布式集群和数据加速组件，提供了集群KV数据管理能力，支持全量图和增量图分离部署。

支持的产品形态如下：

- Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

相关接口存放在："$\{INSTALL\_DIR\}/include/llm\_datadist/llm\_datadist.h"。$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。若安装的Ascend-cann-toolkit软件包，以root安装举例，则安装后文件存储路径为：/usr/local/Ascend/ascend-toolkit/latest。

接口对应的库文件是：libllm\_engine.so。

LLM-DataDist接口列表如下。

**表 1**  LLM-DataDist接口\_V1

| 接口名称 | 简介 |
| --- | --- |
| [LlmDataDist构造函数](LlmDataDist构造函数.md) | 构造LLM-DataDist。 |
| [~LlmDataDist()](LlmDataDist().md) | LLM-DataDist对象析构函数。 |
| [Initialize](Initialize.md) | 初始化LLM-DataDist。 |
| [Finalize](Finalize.md) | 释放LLM-DataDist。 |
| [SetRole](SetRole.md) | 设置当前LLM-DataDist的角色。 |
| [LinkLlmClusters](LinkLlmClusters.md) | 建链。 |
| [UnlinkLlmClusters](UnlinkLlmClusters.md) | 断链。 |
| [PullKvCache](PullKvCache.md) | 以连续内存方式拉取KV Cache。 |
| [PullKvBlocks](PullKvBlocks.md) | 以block列表的方式拉取KV Cache。 |
| [CopyKvCache](CopyKvCache.md) | 以连续内存方式拷贝KV Cache。 |
| [CopyKvBlocks](CopyKvBlocks.md) | 以block列表的方式拷贝KV Cache。 |
| [PushKvCache](PushKvCache.md) | 推送Cache到远端节点，仅当角色为Prompt时可调用。 |
| [PushKvBlocks](PushKvBlocks.md) | 通过block列表的方式，推送Cache到远端节点，仅当角色为Prompt时可调用。 |
| [AllocateCache](AllocateCache.md) | 分配Cache。 |
| [DeallocateCache](DeallocateCache.md) | 释放Cache。 |
