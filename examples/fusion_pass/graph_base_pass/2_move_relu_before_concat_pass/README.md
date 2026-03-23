# 样例使用指导

## 功能描述

本样例为移动ReLu到Concat前的自定义pass样例，当融合pass场景中需要包含**动态数量输入/出的算子**时，可参考本样例。
样例中提供在线推理与atc工具离线编译模型两种方式演示框架如何调用自定义pass完成图优化，
使用eager style api和融合接口实现。

## 目录结构

```
├── src
│   ├──move_relu_before_concat_pass.cpp                 // pass实现文件 
├── CMakeLists.txt                                      // 编译脚本
├── data         
|   ├──es_gen_air.py                                // 导出air
|   ├──torch_forward.py                                 // torch脚本用于在线推理
|—— gen_es_api
|   |——CMakeLists.txt                                   // 生成eager style api的编译脚本
```

## 环境要求

- 编译器：GCC >= 7.3.x
- 使用python及其依赖库版本：python>=3.9 、pytorch>=2.1
- 已完成[相关环境准备](../../../../docs/build.md)。


## 实现步骤
1. 定义`MoveReluBeforeConcatPass`类继承`FusionBasePass`。
2. 重写基类`FusionBasePass`中的`Run`方法，其中实现自定义pass逻辑。
3. 定义`FindConcatNodesMeetRequirements`遍历图中节点，获取符合条件的Concat节点。
4. 定义`MoveReluBeforeConcat`实现改图，其中：
   - `Replacement`根据concat节点构造替换结构
   - `GetSubgraphBoundary`构造被替换的子图边界boundary
   - 最后调用`SubgraphRewriter`的`Replace`方法实现替换

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

4. 执行make命令编译自定义pass so，成功编译后通过make install将动态库文件libmove_relu_before_concat_pass.so安装到自定义融合pass目录下。
   可以在make后增加可选参数`-j$(nproc)`用于并行执行构建任务，`$(nproc)`动态获取CPU核心数。
   ```
   make -j$(nproc) move_relu_before_concat_pass
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
    - 安装es_all.whl
      ```
      pip install --force-reinstall --upgrade --target ${ASCEND_PATH}/python/site-packages/ 
      ${BUILD_PATH}/es_output/whl/es_all-*****.whl
      ```
      `${BUILD_PATH}`请替换为build目录的实际路径。
    - 设置环境变量，添加es_all.so的路径
      ```
      LD_LIBRARY_PATH="${BUILD_PATH}/es_output/lib64:${LD_LIBRARY_PATH}"
      ```
    - 进入data目录执行.py文件导出air：
      ```
      python es_gen_air.py
      ```
    - 执行结束后，在data目录下生成.air格式的模型文件，名称为graph.air。
    - 执行ATC工具命令(关于ATC工具的详细说明，请前往[昇腾文档](https://www.hiascend.com/zh/document)搜索文档“ATC离线模型编译工具”)，`soc_version`请根据实际环境修改：
      ```
      atc --model=./graph.air --framework=1 --soc_version=xxx --output=./model
      ```
    - 日志中出现如下打印：
      ```
      MoveReluBeforeConcatPass
      Define Replacement for MoveReluBeforeConcatPass
      Replacement of MoveReluBeforeConcatPass succeeded
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
      MoveReluBeforeConcatPass
      Define Replacement for MoveReluBeforeConcatPass
      Replacement of MoveReluBeforeConcatPass succeeded
      ```

4. 查看运行结果

    - ATC工具命令执行完成后，目录下生成一系列.pbtxt文件。
      对比以下dump图：
        - `ge_onnx_xxxxx_PreRunBegin.pbtxt`执行前dump图
        - `ge_onnx_xxxxx_RunCustomPassBeforeInferShape.pbtxt`执行InferShape前的自定义pass dump图

      可以发现模型已按预期优化，即ReLu被移动到到Concat前。

   - 若未获得预期结果，可设置如下环境变量（如使用atc命令，还需添加参数`--log=debug`）让日志打印到屏幕，来定位原因。
     ```bash
      export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
      export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
     ```