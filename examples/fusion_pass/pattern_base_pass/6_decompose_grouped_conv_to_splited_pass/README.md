# 样例使用指导

## 功能描述

本样例为拆分分组卷积的自定义pass样例，
提供在线推理与atc工具离线编译模型两种方式演示框架如何调用自定义pass完成图优化。
本样例使用eager style api和融合接口实现。

## 目录结构

```
├── src
│   ├──decompose_grouped_conv_to_splited_pass.cpp       // pass实现文件 
├── CMakeLists.txt                                      // 编译脚本
├── data         
|   ├──torch_gen_onnx.py                                // torch脚本用于导出onnx
|   ├──torch_forward.py                                 // torch脚本用于在线推理
|—— gen_es_api
|   |——CMakeLists.txt                                   // 生成eager style api的编译脚本
```

## 环境要求

- 编译器：GCC >= 7.3.x
- 使用python及其依赖库版本：python>=3.9 、pytorch>=2.1
- 已完成[相关环境准备](../../../../docs/build.md)。

## 实现步骤

1. 定义类`DecomposeGroupedConvToSplitedPass`继承`DecomposePass`。
2. 定义构造函数，使用传入的算子类型初始化基类，传入的算子类型会被该pass捕获。
3. 重写父类`DecomposePass`中的2个函数：
   - `MeetRequirements`对模板匹配到的拓扑进行筛选。
   - `Replacement`定义替换部分。其中`InferShape`进行shape推导`，CheckNodeSupportOnAicore`判断ai core是否支持替换节点。
4. 注册`DecomposeGroupedConvToSplitedPass`为自定义decompose pass，构造函数中算子类型传入"Conv2D"，执行阶段为AfterInferShape。

## 程序编译

假设CANN软件包的安装目录为INSTALL_PATH，例如`/home/HwHiAiUser/Ascend/`。

1. 配置环境变量。

   运行软件包中设置环境变量脚本，命令如下：

   ```
   source ${ASCEND_PATH}/set_env.sh
   ```

   `${ASCEND_PATH}`为CANN软件包安装目录下的cann路径。请替换相关软件包的实际安装路径，例如`${INSTALL_PATH}/cann`。

2. 根据实际情况修改**CMakeLists.txt**文件中的如下信息。

   - ASCEND_PATH：可以设置默认的软件包路径，如果通过set_env.sh设置了`$ASCEND_HOME_PATH`，无需修改。

   - PASS_SO_DIR：可以设置自定义融合pass动态库安装目录名，默认为`pass_so_dir`。

   - target_include_directories：需要包含的头文件，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加头文件时，在示例下方直接增加行即可，注意不要删除原有项目。如果网络中有自定义算子，请增加自定义算子的原型定义头文件。

   - target_link_libraries：需要链接的库，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加链接库时，在示例下方直接增加行即可，注意不要删除原有项目。

     > 禁止链接软件包中的其他so，否则后续升级可能会导致兼容性问题。

3. 执行如下命令 生成eager style api

   依次执行:

   ```
   mkdir build && cd build
   cmake ..
   ```
   执行后，在**build**目录下产生的es_all_build/generated_code目录中包含es构图api的头文件及源码。

4. 执行make命令编译自定义pass so，成功编译后通过make install将动态库文件libdecompose_grouped_conv_to_splited_pass.so安装到自定义融合pass目录下。
   可以在make后增加可选参数`-j$(nproc)`用于并行执行构建任务，`$(nproc)`动态获取CPU核心数。
   ```
   make -j$(nproc) decompose_grouped_conv_to_splited_pass
   make install
   ```

## 程序运行

1. 配置环境变量(如已执行，跳过)。

   - 运行软件包中设置环境变量脚本，命令如下：

     ```
     source ${ASCEND_PATH}/set_env.sh
     ```

     `${ASCEND_PATH}`请替换相关软件包的实际安装路径。

2. 使用ATC离线推理。

   - 设置环境变量，dump出编译过程中的模型图：
     ```
     export DUMP_GE_GRAPH=1
     ```
   - 进入data目录执行.py文件导出onnx（文件中使用了torch的onnx导出器，依赖额外的Python包onnx，运行前确保安装。
     此外ATC工具当前最高支持onnx opset_version 18，若当前torch默认导出更高版本，请显式指定，见脚本中注释）：
     ```
     python torch_gen_onnx.py
     ```
   - 执行结束后，在data目录下生成.onnx格式的模型文件，名称为model.onnx。
   - 执行ATC工具命令(关于ATC工具的详细说明，请前往[昇腾文档](https://www.hiascend.com/zh/document)搜索文档“ATC离线模型编译工具”)，`soc_version`请根据实际环境修改：
     ```
     atc --model=./model.onnx --framework=5 --soc_version=xxx --output=./model
     ```
   - 日志中出现如下打印：
     ```
     Define MeetRequirements for DecomposeGroupedConvToSplitedPass
     Define Replacement for DecomposeGroupedConvToSplitedPass
     ```

3. 在线推理
   - 设置环境变量，dump出编译过程中的模型图：
      ```
      export DUMP_GE_GRAPH=1
      ```
   - 进入data目录执行.py文件进行在线推理（在线推理请确保已安装torch_npu插件）：
      ```
      python torch_forward.py
      ```  
   - 日志中出现如下打印：
     ```
      Define MeetRequirements for DecomposeGroupedConvToSplitedPass
      Define Replacement for DecomposeGroupedConvToSplitedPass
     ```

4. 查看运行结果

   - ATC工具命令执行完成后，目录下生成一系列.pbtxt文件。
     对比以下dump图：
      - `ge_onnx_xxxxx_PreRunBegin.pbtxt`执行前dump图
      - `ge_onnx_xxxxx_RunCustomPass_AfterInferShape.pbtxt`执行InferShape后的自定义pass dump图

     可以发现模型已按预期优化，即groups=3的分组卷积被拆分成3个groups=1的卷积。

   - 若未获得预期结果，可设置如下环境变量（如使用atc命令，还需添加参数`--log=debug`）让日志打印到屏幕，来定位原因。
      ```bash
       export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
       export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
      ```
