# 样例使用指导<a name="ZH-CN_TOPIC_0345664697"></a>

## 功能描述<a name="section5991635456363"></a>

本样例以`MatMul+Add`融合为`GEMM`的融合pass为例，介绍`PatternMatcherConfig`功能的使用，
提供在线推理与atc工具离线编译模型两种验证方式，pass使用eager style api和融合接口实现。


## 目录结构<a name="section7668345634665"></a>

```
├── src
│   ├──fuse_matmul_add_pass.cpp  // pass实现文件 
├── CMakeLists.txt               // 编译脚本
├── data         
|   ├──es_gen_air.py         // 导出air
|   ├──es_forward_1.py        // 使用eager style api构图，调用ge api做在线推理，pass成功执行
|   ├──es_forward_2.py        // 在线推理，EnableConstValueMatch拦截生效，pass被拦截
|   ├──es_forward_3.py        // 在线推理，EnableIrAttrMatch拦截生效， pass被拦截
|—— gen_es_api
|   |——CMakeLists.txt            // 生成eager style api的编译脚本
```

## 环境要求<a name="section383335652346"></a>

- 操作系统及架构：CentOS x86系统、CentOS aarch64系统、Euleros x86系统、Euleros aarch64系统
- 编译器：gcc7及以上
- 芯片：all
- 使用python及其依赖库版本：python>=3.8 、pytorch>=2.1
- [相关环境准备](../../../docs/build.md)

## 实现步骤

1. 定义类`FuseMatMulAndAddPass`继承`PatternFusionPass`。重写构造函数：
    ```
	explicit MatmulAddFusionPass() : atternFusionPass(PatternMatcherConfigBuilder()
                        .EnableConstValueMatch()
                        .EnableIrAttrMatch().Build()){}
    ```
2. 重写基类`PatternFusionPass`中的2个函数：
   - `Patterns`定义匹配模板，用于在整图中获取与该模板相同的拓扑。
   - `Replacement`定义替换部分。
3. 注册`FuseMatMulAndAddPass`为自定义融合pass，执行阶段为BeforeInferShape。

## 程序编译<a name="section6645633456813"></a>

假设CANN软件包的安装目录为INSTALL_PATH, 例如`/home/HwHiAiUser/Ascend/`。

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
   执行后，在**build**目录下产生es_all_build目录，内含es构图api的头文件及源码

5. 执行如下命令编译自定义pass so，并将编译后的动态库文件libfuse_matmul_add_for_matcher_config_sample_pass.so拷贝到自定义融合pass目录下，其中“xxx”为用户自定义目录。
   可以在make后增加可选参数`-j$(nproc)`用于并行执行构建任务，`$(nproc)`动态获取CPU核心数。
   ```
   make -j$(nproc) fuse_matmul_add_for_matcher_config_sample_pass
   make install
   ```

## 程序运行<a name="section4524573456563512"></a>

1. 配置环境变量(如已执行，跳过)。

   - 运行软件包中设置环境变量脚本，命令如下：

     ```
     source ${ASCEND_PATH}/set_env.sh
     ```

     `${ASCEND_PATH}`请替换相关软件包的实际安装路径。
   - 设置环境变量，dump出编译过程中的模型图：
     ```
     export DUMP_GE_GRAPH=1
     ```
   - 安装es_all.whl
      ```
       pip install --force-reinstall --upgrade --target ${ASCEND_PATH}/python/site-packages/ ${BUILD_PATH}/es_output/whl/es_all-*****.whl
     ```
     `${BUILD_PATH}`请替换为build目录的实际路径。
    - 设置环境变量，添加es_all.so的路径
      ```
      export LD_LIBRARY_PATH="${BUILD_PATH}/es_output/lib64:${LD_LIBRARY_PATH}"
      ```
2. 使用ATC离线推理。
   - 进入data目录执行.py文件导出air（文件中使用了 es 的 python 接口来构图）：
     ```
     python es_gen_air.py
     ```
     - 执行结束后，在data目录下生成.air格式的模型文件，名称为graph.air。
     - 执行ATC工具命令(关于ATC工具的详细说明，请前往[昇腾社区](https://www.hiascend.com)查看文档“ATC离线模型编译工具”)，`soc_version`请根据实际环境修改：
       ```
       atc --framework=1 --model=./graph.air --soc_version=xxx --output=./model
       ```
     - 运行成功后，日志中出现如下打印：
        ```
        Define pattern for MatMulAddFusionPass in matcher config sample
        Define replacement for MatMulAddFusionPass in matcher config sample
        ```

3. 在线推理
   - 进入data目录执行.py文件进行在线推理（在线推理请确保已安装torch_npu插件）：
      ```
      python es_forward_1.py/es_forward_2.py/es_forward_3.py
      ```  
   - 对于es_forward_1.py，日志中出现如下打印：
     ```
     Define pattern for MatMulAddFusionPass in matcher config sample
     Define replacement for MatMulAddFusionPass in matcher config sample
     ```
   - 对于es_forward_2.py和es_forward_3.py，日志中出现如下打印：
      ```
      Define pattern for FuseMatMulAndAddPass in matcher config sample
      ```

4. 查看运行结果

   - 执行完成后，目录下生成一系列.pdtxt文件。
     对比以下dump图：
      - `ge_onnx_xxxxx_PreRunBegin.pdtxt`执行前dump图
      - `ge_onnx_xxxxx_RunCustomPassBeforeInferShape.pdtxt`执行InferShape前的自定义pass dump图
     
     可以发现模型已按预期优化，即MatMul与Add被GEMM替换。
   - 若未获得预期结果，可设置如下环境变量让日志打印到屏幕，来定位原因。
     ```bash
      export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
      export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
     ```
   


       