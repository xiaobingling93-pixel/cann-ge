# 样例使用指导

## 1、功能描述
本样例使用操作符重载进行构图，旨在帮助构图开发者快速理解操作符重载的定义，因为python接口暂时**未支持异步执行**，所以本样例仅展示同步执行的构图过程。
## 2、目录结构
```angular2html
python/
├── src/
|   └── make_add_graph.py   // sample文件
├── CMakeLists.txt          // 编译脚本
├── README.md               // README文件
├── run_sample.sh           // 执行脚本
```

## 3、使用方法
### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../docs/build.md#2-安装软件包)正确安装`toolkit`和`ops`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh 
```
### 3.2、编译和执行
- 注：和 C/C++构图对比，Python构图需要额外添加 LD_LIBRARY_PATH 和 PYTHONPATH(参考sample中的配置方式)
```bash
bash run_sample.sh -t sample_and_run_python
```
该命令会：
1. 自动生成ES接口
2. 编译sample程序
3. 生成dump图并运行该图

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
可执行程序执行过程中，如果需要DUMP图来辅助定位图编译流程，可以在 bash run_sample.sh -t sample_and_run_python 之前设置如下环境变量来DUMP图到执行路径下
```bash
export DUMP_GE_GRAPH=2 
```

## 4、核心概念介绍

### 4.1、构图步骤如下：
- 创建图构建器(用于提供构图所需的上下文、工作空间及构建相关方法)
- 添加起始节点(起始节点指无输入依赖的节点，通常包括图的输入(如 Data 节点)和权重常量(如 Const 节点))
- 添加中间节点(中间节点为具有输入依赖的计算节点，通常由用户构图逻辑生成，并通过已有节点作为输入连接)
- 设置图输出(明确图的输出节点，作为计算结果的终点)

### 4.2、操作符重载
**概念说明：**
操作符重载是 ES API 提供的语法糖，针对AI算子做语法封装，使构图代码更加简洁和直观

**构图 API 特点：**
- 操作符重载保持与函数调用相同的类型检查和约束