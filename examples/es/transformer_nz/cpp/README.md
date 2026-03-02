# 样例使用指导

## 1、功能描述
本样例使用NZ数据格式对transformer进行构图，旨在帮助构图开发者快速上手关于es的构图方式
## 2、目录结构
```angular2html
cpp/
├── src/
|   └── CMakeLists.txt              // CMake构建文件
|   └── es_showcase.h               // 头文件
|   └── make_transformer_nz_graph.cpp  // sample文件
├── CMakeLists.txt                  // CMake构建文件
├── main.cpp                        // 程序主入口
├── README.md                       // README文件
├── run_sample.sh                   // 执行脚本
├── utils.h                         // 工具文件
```

## 3、使用方法
### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh 
```
### 3.2、编译和执行

只需运行下述命令即可完成清理、生成接口、构图和DUMP图：
```bash
bash run_sample.sh
```
当前 run_sample.sh 的行为是：先自动清理旧的 build，构建 sample并默认执行sample dump 。当看到如下信息，代表执行成功：
```
[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示
```
#### 输出文件说明

执行成功后会在当前目录生成以下文件：
- `ge_onnx_*.pbtxt` - 图结构的protobuf文本格式，可用netron查看

#### 构建图并执行

除了基本的图构建和dump功能外，esb_sample还支持构建图并实际执行计算。

```bash
bash run_sample.sh -t sample_and_run
```
该命令会：
1. 自动生成ES接口
2. 编译sample程序
3. 生成dump图、运行图并输出计算结果

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

### 4.2、NZ数据格式的说明：
- NZ数据格式是昇腾亲和的数据格式，在昇腾AI处理器上计算效率更高，通常比ND格式性能更好。
- 关于NZ数据格式的详细说明，请参考昇腾官网说明：[数据排布格式](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/graphdevg/atlas_ascendc_10_0099.html)
