# 样例使用指导

## 1、功能描述
本样例演示如何使用HcomAllReduce算子集合通信进行构图，旨在帮助构图开发者快速理解集合通信定义和使用该类型算子进行构图。

## 2、目录结构
```angular2html
python/
├── src/
|   └── make_pfa_hcom_graph.py     // sample文件
├── rank_table/
|   ├── a2/
|   |   └── rank_table_2p.json     // A2(d802) 2卡rank table配置(v1.0)
├── CMakeLists.txt                 // 编译脚本
├── README.md                      // README文件
├── run_sample.sh                  // 执行脚本
```

## 3、使用方法

### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh 
```

### 3.2、编译和执行

**重要前提：确保您的系统有至少2个可用的NPU设备**

**平台支持说明：**
- A2 平台：`lspci | grep d802` 有输出，脚本自动使用 `rank_table/a2/rank_table_2p.json`
- A5 平台：`lspci | grep d806` 有输出，脚本会报错退出（不支持此形态）
- 其他平台：当前版本暂不支持，会在脚本中直接报错退出

注：和 C/C++构图对比，Python构图需要额外添加 LD_LIBRARY_PATH 和 PYTHONPATH(参考sample中的配置方式)

**使用方法：**
```bash
bash run_sample.sh -t sample_and_run_python
```

该命令会：
1. 自动生成ES接口
2. 编译sample程序
3. dump图到当前目录
4. 在2个NPU设备上并行运行TP图（设备ID从 rank_table 自动读取）

**注意事项：**
- 脚本会通过 `lspci` 自动识别硬件并选择对应 rank table（当前仅 A2 使用 `rank_table/a2/rank_table_2p.json`；A5 不支持此形态）
- 如需使用其他设备（如2,3或4,5），请修改对应平台目录下 rank table 文件中的 `device_id`

执行成功后会看到：
```
[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示
```

#### 输出文件说明

执行成功后会在当前目录生成以下文件：
- `ge_onnx_*.pbtxt` - 图结构的protobuf文本格式，可用netron查看

### 3.3、日志打印
可执行程序执行过程中如果需要日志打印来辅助定位，可以在bash run_sample.sh -t sample_and_run_python之前设置如下环境变量来让日志打印到屏幕
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
```
### 3.4、图编译流程中DUMP图
可执行程序执行过程中，如果需要DUMP图来辅助定位图编译流程，可以在bash run_sample.sh -t sample_and_run_python 之前设置如下环境变量来DUMP图到执行路径下
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
- `RANK_TABLE_FILE`：rank table 配置文件路径（当前仅 A2: `rank_table/a2/rank_table_2p.json`；A5 不支持此形态）
- 关于`RANK_TABLE_FILE`、`RANK_ID`、`DEVICE_ID`的详细介绍可以参考 [以a2为例:rank table配置资源信息](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/hccl/hcclug/hcclug_000067.html)

**GE 初始化配置：**
```python
config = {
    "ge.exec.deviceId": str(device_id),           # 来自环境变量 DEVICE_ID
    "ge.graphRunMode": "0",                       
    "ge.exec.rankTableFile": rank_table_file,     # 来自环境变量 RANK_TABLE_FILE
    "ge.exec.rankId": rank_id                     # 来自环境变量 RANK_ID
}
```

### 4.3、TP图构图
**概念说明：**
TP（Tensor Parallel）图是指通过张量并行方式在多卡上运行的图结构。本样例演示了如何使用ES算子构建包含集合通信算子的TP图，实现多卡间的数据同步和并行计算。

HcomAllReduce 算子原型如下所示，ES 构图生成的API是`HcomAllReduce()`，支持在 Python 中使用
```
  REG_OP(HcomAllReduce)
      .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
      .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
      .REQUIRED_ATTR(reduction, String)
      .REQUIRED_ATTR(group, String)
      .ATTR(fusion, Int, 1)
      .ATTR(fusion_id, Int, -1) 
      .OP_END_FACTORY_REG(HcomAllReduce)
```
其对应的函数原型为：
- 函数名：HcomAllReduce
- 参数：共 5 个，依次为 x， reduction， group， fusion（可选，默认1）， fusion_id（可选，默认-1）
- 返回值：输出 y

**Python API中：**
```
HcomAllReduce(x: TensorHolder, *, reduction: str, group: str, fusion: int = 1, fusion_id: int = -1) -> TensorHolder
```
注： reduction、group为必选关键字参数，fusion和fusion_id为可选参数，具有默认值
