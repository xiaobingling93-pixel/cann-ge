# 样例使用指导<a name="ZH-CN_TOPIC_0345664697"></a>

## 功能描述<a name="section5991635456363"></a>

本样例以MatMul+Add融合为GEMM的融合pass为例，介绍capture tensor 功能的使用，
提供在线推理与atc工具离线编译模型两种验证方式，pass使用eager style api和融合接口实现。


## 目录结构<a name="section7668345634665"></a>

```
├── src
│   ├──fuse_matmul_add_pass.cpp  // pass实现文件 
├── CMakeLists.txt               // 编译脚本
├── data         
|   ├──torch_gen_onnx.py         // torch脚本用于导出onnx
|   ├──torch_forward_1.py          // torch脚本用于在线推理，pass成功执行
|   ├──torch_forward_2.py          // torch脚本用于在线推理，pass被拦截
|—— gen_es_api
|   |——CMakeLists.txt            // 生成eager style api的编译脚本
```

## 环境要求<a name="section383335652346"></a>

- 编译器：GCC >= 7.3.x
- 使用python及其依赖库版本：python>=3.9 、pytorch>=2.1
- 已完成[相关环境准备](../../../../docs/build.md)。

## 实现步骤

1. 定义类`FuseMatMulAndAddPass`继承`PatternFusionPass`。
2. 重写基类`PatternFusionPass`中的3个函数：
    - `Patterns`定义匹配模板，用于在整图中获取与该模板相同的拓扑。
      - `pattern->CaptureTensor()`捕获tensor，tensor结构为：`{NodeIo,index}`。
    - `MeetRequirements`对模板匹配到的拓扑进行筛选。
      - `match_result->GetCapturedTensor(kAddCaptureIdx,add_node);`读取捕获的NodeIo，进行检查。
    - `Replacement`定义替换部分。
      - `match_result->GetCapturedTensor(kMatMulCaptureIdx, matmul_node);`读取捕获的NodeIo，提取属性值。
3. 注册`FuseMatMulAndAddPass`为自定义融合pass，执行阶段为BeforeInferShape。

## 程序编译<a name="section6645633456813"></a>

1. 配置环境变量。

   运行软件包中设置环境变量脚本，命令如下：
   
   ```
   source ${ASCEND_PATH}/set_env.sh
   ```
   
   `${ASCEND_PATH}`为CANN软件包安装目录下的cann路径。请替换相关软件包的实际安装路径，例如`${INSTALL_PATH}/cann`。

2. 根据实际情况修改**CMakeLists.txt**文件中的如下信息。

   - ASCEND_PATH：可以设置默认的软件包路径，如果通过set_env.sh设置了`$ASCEND_HOME_PATH`，无需修改。

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

4. 执行make命令编译自定义pass so，成功编译后通过make install将动态库文件libfuse_matmul_add_for_capture_tensor_sample_pass.so安装到自定义融合pass目录下。
   可以在make后增加可选参数`-j$(nproc)`用于并行执行构建任务，`$(nproc)`动态获取CPU核心数。
   ```
   make -j$(nproc) fuse_matmul_add_for_capture_tensor_sample_pass
   make install
    ```

## 程序运行<a name="section4524573456563512"></a>

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
    - 进入data目录执行.py文件导出onnx（文件中使用了torch的onnx导出器，依赖额外的Python包onnx，运行前请确保安装。此外，ATC工具当前最高支持onnx opset_version 18,若当前torch默认导出更高版本，需显示指定，详情见脚本中注释）：
      ```
      python torch_gen_onnx.py
      ```
    - 执行结束后，在data目录下生成.onnx格式的模型文件，名称为model.onnx。
    - 执行ATC工具命令(关于ATC工具的详细说明，请前往[昇腾文档](https://www.hiascend.com/zh/document)搜索文档“ATC离线模型编译工具”)，`soc_version`请根据实际环境修改：
      ```
      atc --model=./model.onnx --framework=5 --soc_version=xxx --output=./model
      ```
    - 日志打印内容：
      ```
      Define pattern for FuseMatMulAndAddPass in capture tensor sample
      Define MeetRequirements for FuseMatMulAndAddPass in capture tensor sample
      Define replacement for FuseMatMulAndAddPass in capture tensor sample
      ```

3. 在线推理
    - 设置环境变量，dump出编译过程中的模型图：
       ```
       export DUMP_GE_GRAPH=1
       ```
    - 进入data目录执行.py文件进行在线推理（在线推理请确保已安装torch_npu插件），执行`torch_forward_1.py`：
       ```
       python torch_forward_1.py
       ```  
   - 对于torch_forward_1.py，日志中出现如下打印：
     ```
     Define pattern for FuseMatMulAndAddPass in capture tensor sample
     Define MeetRequirements for FuseMatMulAndAddPass in capture tensor sample
     Define replacement for FuseMatMulAndAddPass in capture tensor sample
     ```
   - 执行`torch_forward_2.py`：
      ```
      python torch_forward_2.py
      ```
   - 对于torch_forward_2.py，日志中出现如下打印：
      ```
      Define pattern for FuseMatMulAndAddPass in capture tensor sample
      Define MeetRequirements for FuseMatMulAndAddPass in capture tensor sample
      Only support Add inputs are fp32
      ```

4. 查看运行结果

    - 执行完成后，目录下生成一系列.pdtxt文件。
      对比以下dump图：
        - `ge_onnx_xxxxx_PreRunBegin.pdtxt`执行前dump图
        - `ge_onnx_xxxxx_RunCustomPassBeforeInferShape.pdtxt`执行InferShape前的自定义pass dump图

      可以发现模型已按预期优化，即MatMul与Add被GEMM替换。

   - 若未获得预期结果，可设置如下环境变量（如使用atc命令，还需添加参数`--log=debug`）让日志打印到屏幕，来定位原因。
     ```bash
      export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
      export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
     ```


       