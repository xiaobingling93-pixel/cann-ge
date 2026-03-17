# 样例使用指导<a name="ZH-CN_TOPIC_0345664697"></a>

## 功能描述<a name="section5991635456363"></a>

本样例为MatMul+Add融合为GEMM自定义pass样例，分别使用ATC离线推理和TF在线推理演示框架如何调用自定义pass完成图优化。

## 目录结构<a name="section7668345634665"></a>

```
├── src
│   ├──fuse_matmul_add_pass.cpp  // pass实现文件 
├── CMakeLists.txt               // 编译脚本
├── data         
│   ├──tensorflow_generate.py    // 生成.pb格式的TensorFlow模型用于离线推理
|   ├──tf_forward.py             // TF在线构出原图后进行自定义pass和其他框架内置pass优化，然后执行优化后的图得到结果
```

## 环境要求<a name="section383335652346"></a>

-   操作系统及架构：CentOS x86系统、CentOS aarch64系统、Euleros x86系统、Euleros aarch64系统
-   编译器：g++
-   芯片：all
-   python及依赖的库：python3.7.5、tensorflow1.15.0
-   已完成昇腾AI软件栈在开发环境上的部署


## 程序编译<a name="section6645633456813"></a>

假设CANN软件包的安装目录为INSTALL_PATH，例如`/home/HwHiAiUser/Ascend/`。

1. 配置环境变量。

   运行软件包中设置环境变量脚本，命令如下：

   ```
   source ${ASCEND_PATH}/set_env.sh
   ```

   `${ASCEND_PATH}`为CANN软件包安装目录下的cann路径。请替换相关软件包的实际安装路径，例如`${INSTALL_PATH}/cann`。

2. 根据实际情况修改**CMakeLists.txt**文件中的如下信息。

    - ASCEND_PATH：可以设置默认的软件包路径，如果通过set_env.sh设置了`$ASCEND_HOME_PATH`，无需修改。

    - FUSION_PASS_DIR：可以设置自定义融合pass动态库安装目录名，默认为`fusion_passes`。

    - target_include_directories：需要包含的头文件，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加头文件时，在示例下方直接增加行即可，注意不要删除原有项目。如果网络中有自定义算子，请增加自定义算子的原型定义头文件。

    - target_link_libraries：需要链接的库，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加链接库时，在示例下方直接增加行即可，注意不要删除原有项目。

      > 禁止链接软件包中的其他so，否则后续升级可能会导致兼容性问题。

3. 执行如下命令进行编译，编译结束后，在**build**目录下生成动态库文件**libfuse_matmul_add_pass.so**。

   依次执行:

   ```
   mkdir build && cd build
   cmake .. && make
   ```
4. 成功编译后通过make install将动态库文件libfuse_matmul_add_pass.so安装到自定义融合pass目录下。

   ```
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

   - 在**data**目录执行tensorflow原始模型生成脚本：

     **python tensorflow_generate.py**

     执行结束后，在**data**目录下生成.pb格式的模型文件，名称为**matmul_add.pb**。

   - 执行ATC命令，其中soc_version根据实际模型运行环境填写：

     **atc --model=./matmul_add.pb --framework=3 --soc_version=xxx --output=./matmul_add**

     执行完命令后会在**data**目录下生成**matmul_add.om**模型文件，后续可按照离线推理流程加载执行此模型文件。

   - 检查执行结果：

     - 自定义Pass生效时，对比NPU编译过程中间dump图，发现模型已按照预期被优化，dump图的获取方法请单击[Link](https://hiascend.com/document/redirect/CannCommercialEnvvar)>编译相关>图编译>DUMP_GE_GRAPH获取：
       - 针对8.3.RC1之前的版本，dump图名字为：
         - ge_onnx_xxxxxxxx_RunCustomPassBegin.pbtxt：融合前的图
         - ge_onnx_xxxxxxxx_RunCustomPassEnd.pbtxt：融合后的图
       - 8.3.RC1及后续版本，dump图名字为：
         - ge_onnx_xxxxxxxx_PreRunBegin.pbtxt：融合前的图
         - ge_onnx_xxxxxxxx_RunCustomPassBeforeInfershape.pbtxt：融合后的图

     - 日志中出现如下打印：

       ```
       FuseMatMulAndAddPass begin.
       Find src node: MatMul.
       Find dst node: Add.
       FuseMatMulAndAddPass end.
       ```

3. 使用TF在线推理。

   - 在线推理分别在目标文件夹下存放和不存放自定义pass so，执行如下命令：

     **python tf_forward.py**

     两次运行结果相同，结果展示：

     ```
     ---out---
      [[23. 29.]
      [50. 65.]]
     ```

   - 检查执行结果：

     - 自定义pass生效前后运行结果相同。

     - 自定义Pass生效时，对比NPU编译过程中间dump图，发现模型已按照预期被优化：
       - 针对8.3.RC1之前的版本，dump图名字为：
         - ge_onnx_xxxxxxxx_RunCustomPassBegin.pbtxt：融合前的图
         - ge_onnx_xxxxxxxx_RunCustomPassEnd.pbtxt：融合后的图
       - 8.3.RC1及后续版本，dump图名字为：
         - ge_onnx_xxxxxxxx_PreRunBegin.pbtxt：融合前的图
         - ge_onnx_xxxxxxxx_RunCustomPassBeforeInfershape.pbtxt：融合后的图

     - 日志中出现如下打印：

       ```
       FuseMatMulAndAddPass begin.
       Find src node: MatMul.
       Find dst node: Add.
       FuseMatMulAndAddPass end.
       ```

       