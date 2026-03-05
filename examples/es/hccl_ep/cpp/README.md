# 样例使用指导

## 1、功能描述
本样例演示如何使用HcomAllGather、HcomReduceScatter算子集合通信进行构图，旨在帮助构图开发者快速理解集合通信定义和使用该类型算子进行构图。

## 2、目录结构
```angular2html
cpp/
├── src/
|   ├── CMakeLists.txt                // CMake构建文件
|   ├── es_showcase.h                 // 头文件
|   └── make_ep_graph.cpp             // sample文件
├── rank_table/
|   ├── a2/
|   |   └── rank_table_2p.json        // A2(d802) 2卡rank table配置(v1.0)
|   └── a5/
|       └── rank_table_2p.json        // A5(d806) 2卡rank table配置(v2.0)
├── CMakeLists.txt                    // CMake构建文件
├── main.cpp                          // 程序主入口
├── README.md                         // README文件
├── run_sample.sh                     // 执行脚本
└── utils.h                           // 工具文件
```

## 3、使用方法

### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh
```

### 3.2、编译和执行
#### 1.2.1 生成 es 接口与构建图进行DUMP
只需运行下述命令即可完成清理、生成接口、构图和DUMP图：
```bash
bash run_sample.sh
```
当前 run_sample.sh 的行为是：先自动清理旧的 build，构建 sample并默认执行sample dump 。当看到如下信息，代表执行成功：
```
[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示
```
**1.2.2 输出文件说明**

执行成功后会在当前目录生成以下文件：
- `ge_onnx_*.pbtxt` - 图结构的protobuf文本格式，可用netron查看

#### 1.2.3 构建图并执行

**重要前提：确保您的系统有至少2个可用的NPU设备**

**平台支持说明：**
- A5 平台：`lspci | grep d806` 有输出，脚本自动使用 `rank_table/a5/rank_table_2p.json`
- A2 平台：`lspci | grep d802` 有输出，脚本自动使用 `rank_table/a2/rank_table_2p.json`
- 其他平台：当前版本暂不支持，会在脚本中直接报错退出

除了基本的图构建和dump功能外，本示例还支持在多卡上实际执行EP图。

**使用方法：**
```bash
bash run_sample.sh -t sample_and_run
```

该命令会：
1. 自动生成ES接口
2. 编译sample程序
3. 自动配置 rank table 和环境变量（`RANK_TABLE_FILE`、`RANK_ID`、`DEVICE_ID`）
4. 在2个NPU设备上并行运行图（设备ID从 rank_table 自动读取，每个进程对应一个 rank 和一个设备）
5. 使用HcomAllGather、HcomReduceScatter进行卡间数据同步

**注意事项：**
- 脚本会通过 `lspci` 自动识别硬件并选择对应 rank table（A5 用 `rank_table/a5/rank_table_2p.json`，A2 用 `rank_table/a2/rank_table_2p.json`）
- 如需使用其他设备（如2,3或4,5），请修改对应平台目录下 rank table 文件中的 `device_id`
- `run_sample.sh` 会自动设置所有必需的环境变量，无需手动配置

执行成功后会看到：
```
[Success] sample_and_run 执行成功，pbtxt和data输出dump 已生成在当前目录
```
可通过data文件查看计算结果
### 3.3、日志打印
可执行程序执行过程中如果需要日志打印来辅助定位，可以在bash run_sample.sh之前设置如下环境变量来让日志打印到屏幕
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
```

### 3.4、图编译流程中DUMP图
可执行程序执行过程中，如果需要DUMP图来辅助定位图编译流程，可以在 bash run_sample.sh -t sample_and_run 之前设置如下环境变量来DUMP图到执行路径下
```bash
export DUMP_GE_GRAPH=2
```

## 4、核心概念介绍

### 4.1、构图步骤如下：
- 创建图构建器(用于提供构图所需的上下文、工作空间及构建相关方法)
- 添加起始节点(起始节点指无输入依赖的节点，通常包括图的输入(如 Data 节点)和权重常量(如 Const 节点))
- 添加中间节点(中间节点为具有输入依赖的计算节点，通常由用户构图逻辑生成，并通过已有节点作为输入连接)
- 设置图输出(明确图的输出节点，作为计算结果的终点)

### 4.2、多卡运行关键概念

**环境变量说明：**
运行多卡示例时，脚本会自动设置以下环境变量：
- `RANK_ID`：逻辑进程编号（本示例中为 0 或 1）
- `DEVICE_ID`：物理设备ID（本示例中为 0 或 1）
- `RANK_TABLE_FILE`：rank table 配置文件路径（A5: `rank_table/a5/rank_table_2p.json`；A2: `rank_table/a2/rank_table_2p.json`）
- 关于`RANK_TABLE_FILE`、`RANK_ID`、`DEVICE_ID`的详细介绍可以参考 [以a2为例:rank table配置资源信息](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/hccl/hcclug/hcclug_000067.html)

**GE 初始化配置：**
```cpp
std::map<ge::AscendString, ge::AscendString> config = {
  {"ge.exec.deviceId", device_id},           // 来自环境变量 DEVICE_ID
  {"ge.graphRunMode", "0"},
  {"ge.exec.rankTableFile", rank_table_file}, // 来自环境变量 RANK_TABLE_FILE
  {"ge.exec.rankId", rank_id}                 // 来自环境变量 RANK_ID
};
```

### 4.3、EP图构图
**概念说明：**
EP（Expert Parallel）图是指通过专家并行方式在多卡上运行的图结构。本样例演示了如何使用ES算子构建包含集合通信算子的EP图，实现多卡间的数据同步和并行计算。

**构图 API 特点：**
- 支持多算子组合构图，包括动态量化、矩阵乘法、MoE门控、集合通信等算子
- 使用HcomAllGather和HcomReduceScatter算子实现多卡间数据聚合和分发，需要配置rank table文件
- 支持W8A8量化模式，可在图内部进行FP32到INT8的转换以提升性能

例如 HcomAllGather 算子原型如下所示，ES 构图生成的API是 HcomAllGather（C++）或 EsHcomAllGather（C）
```
  REG_OP(HcomAllGather)
      .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_BFLOAT16, DT_INT64, DT_UINT64,
                            DT_UINT8, DT_UINT16, DT_UINT32, DT_FLOAT64}))
      .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_BFLOAT16, DT_INT64, DT_UINT64,
                            DT_UINT8, DT_UINT16, DT_UINT32, DT_FLOAT64}))
      .REQUIRED_ATTR(rank_size, Int)
      .REQUIRED_ATTR(group, String)
      .ATTR(fusion, Int, 0)
      .ATTR(fusion_id, Int, -1)
      .OP_END_FACTORY_REG(HcomAllGather)
```
其对应的函数原型为：
- 函数名：HcomAllGather（C++）或 EsHcomAllGather（C）
- 参数：共 5 个，依次为 x， rank_size， group， fusion（可选，默认0）， fusion_id（可选，默认-1）
- 返回值：输出 y

**C API中：**
```
EsCTensorHolder *EsHcomAllGather(EsCTensorHolder *x, int64_t rank_size, const char *group, int64_t fusion, int64_t fusion_id);
```
**C++ API：**
```
EsTensorHolder HcomAllGather(const EsTensorHolder &x, int64_t rank_size, const char *group, int64_t fusion=0, int64_t fusion_id=-1);
```
注：C++ API中fusion和fusion_id为可选参数，具有默认值，通常可以省略

例如 HcomReduceScatter 算子原型如下所示，ES 构图生成的API是 HcomReduceScatter（C++）或 EsHcomReduceScatter（C）
```
  REG_OP(HcomReduceScatter)
      .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
      .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
      .REQUIRED_ATTR(reduction, String)
      .ATTR(fusion, Int, 0)
      .ATTR(fusion_id, Int, -1)
      .REQUIRED_ATTR(group, String)
      .REQUIRED_ATTR(rank_size, Int)
      .OP_END_FACTORY_REG(HcomReduceScatter)
```
其对应的函数原型为：
- 函数名：HcomReduceScatter（C++）或 EsHcomReduceScatter（C）
- 参数：共 6 个，依次为 x， reduction， group， rank_size， fusion（可选，默认0）， fusion_id（可选，默认-1）
- 返回值：输出 y

**C API中：**
```
EsCTensorHolder *EsHcomReduceScatter(EsCTensorHolder *x, const char *reduction, const char *group, int64_t rank_size, int64_t fusion, int64_t fusion_id);
```
**C++ API：**
```
EsTensorHolder HcomReduceScatter(const EsTensorHolder &x, const char *reduction, const char *group, int64_t rank_size, int64_t fusion=0, int64_t fusion_id=-1);
```
注：C++ API中fusion和fusion_id为可选参数，具有默认值，通常可以省略
