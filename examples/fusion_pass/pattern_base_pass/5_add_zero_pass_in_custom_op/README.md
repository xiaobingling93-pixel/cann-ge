# 样例使用指导

## 功能描述

本样例为自定义算子AddCustom的自定义pass样例，**此用例针对用户可以获取到自定义算子原型的情况**
本例中的pass实现：对于存在一个输入为 0 的AddCustom，进行删除操作。
提供在线推理与atc工具离线编译模型两种方式演示框架如何调用自定义pass完成图优化。
本样例使用eager style api和融合接口实现。

## 目录结构

```
├── src
│   ├──addcustom_zero_pass.cpp   // pass实现文件 
├── CMakeLists.txt               // 编译脚本
├── data         
|   ├──torch_forward.py          // torch脚本用于在线推理
|—— gen_es_api
|   |——CMakeLists.txt            // 生成eager style api的编译脚本
|—— proto                        // 存放自定义算子的算子原型
|   |——add_custom_proto.cc            
|   |——add_custom_proto.h            
```

## 环境要求

- 编译器：GCC >= 7.3.x
- 使用python及其依赖库版本：python>=3.9 、pytorch>=2.1
- 已完成[相关环境准备](../../../../docs/build.md)。

## 准备工作

1. 创建自定义算子工程：编写自定义算子的自定义 pass 的前提是用户已创建自定义算子工程，可参考[自定义算子入图](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00055.html)。在此阶段，用户需要完成：自定义算子实现、自定义算子包编译与部署、自定义算子适配开发。
2. 获取算子原型：编译成功后将自定义算子工程中 build_out/autogen 路径下的自定义算子原型定义复制到当前工程的 proto 目录下。本样例中 proto 目录下已添加 AddCustom 自定义算子的原型，用户可按需替换或增加。

## 程序编译

1. 配置环境变量。

   - 运行软件包中设置环境变量脚本，命令如下：

     ```
     source ${ASCEND_PATH}/set_env.sh
     ```

     `${ASCEND_PATH}`为CANN软件包安装目录下的cann路径。请替换相关软件包的实际安装路径，例如`${INSTALL_PATH}/cann`。

2. 根据实际情况修改**gen_es_api/CMakeLists**文件：
    - 修改自定义算子原型文件路径：add_library(custom_op_proto SHARED .../proto/your_proto_name.cc)

3. 根据实际情况修改**CMakeLists.txt**文件中的如下信息。

  - ASCEND_PATH：可以设置默认的软件包路径，如果通过set_env.sh设置了`$ASCEND_HOME_PATH`，无需修改。

  - PASS_SO_DIR：可以设置自定义融合pass动态库安装目录名，默认为`pass_so_dir`。

  - target_include_directories：需要包含的头文件，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加头文件时，在示例下方直接增加行即可，注意不要删除原有项目。如果网络中有自定义算子，请增加自定义算子的原型定义头文件。

  - target_link_libraries：需要链接的库，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加链接库时，在示例下方直接增加行即可，注意不要删除原有项目。

  >   禁止链接软件包中的其他so，否则后续升级可能会导致兼容性问题。

4. 执行如下命令 生成eager style api

   依次执行:

   ```
   mkdir build && cd build
   cmake ..
   ```
   执行后，在**build**目录下产生的es_all_build/generated_code目录中包含es构图api的头文件及源码。
   
5. 完成pass的编写后，执行如下命令编译自定义pass so，并将编译后的动态库文件libadd_zero_pass.so拷贝到自定义融合pass目录下，其中“xxx”为用户自定义目录。
   可以在make后增加可选参数`-j$(nproc)`用于并行执行构建任务，`$(nproc)`动态获取CPU核心数。
   ```
   make -j$(nproc) add_custom_zero_pass
   make install
   ```

## pass编写
1. 定义类`AddCustomZeroPass`继承`PatternFusionPass`。
2. 重写基类`PatternFusionPass`中的3个函数：
   - `Patterns`定义匹配模板，用于在整图中获取与该模板相同的拓扑。
   - `MeetRequirements`对模板匹配到的拓扑进行筛选。
   - `Replacement`定义替换部分。
3. 注册`AddCustomZeroPass`为自定义融合pass，执行阶段为BeforeInferShape。

## 验证

1. 配置环境变量。

    - 运行软件包中设置环境变量的脚本，命令如下：

      ```
      source ${ASCEND_PATH}/set_env.sh
      ```
      `${ASCEND_PATH}`为CANN软件包安装目录下的cann路径。请替换相关软件包的实际安装路径，例如`${INSTALL_PATH}/cann`。


2. 在线推理
   - 设置环境变量，dump出编译过程中的模型图：
      ```
      export DUMP_GE_GRAPH=1
      ```
   - 进入data目录执行.py文件进行在线推理：
      ```
      python torch_forward.py
      ```  
   - 日志中出现如下打印：
     ```
     Define pattern for AddCustomZeroPass
     Define MeetRequirements for AddCustomZeroPass
     Define replacement for AddCustomZeroPass
     ```

3. 查看运行结果

   - 执行完成后，目录下生成一系列.pdtxt文件。
      对比以下dump图：
     - `ge_onnx_xxxxx_PreRunBegin.pdtxt`执行前dump图
     - `ge_onnx_xxxxx_RunCustomPassBeforeInferShape.pdtxt`执行InferShape前的自定义pass dump图
     
      可以发现模型已按预期优化，即加零节点被删除。

   - 若未获得预期结果，可设置如下环境变量（如使用atc命令，还需添加参数`--log=debug`）让日志打印到屏幕，来定位原因。
     ```bash
      export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
      export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
     ```

