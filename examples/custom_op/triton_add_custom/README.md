# README

# 1. 自定义算子编译工程

## 1.1 前置步骤
1. 安装 CANN 软件，通过[安装指导](../../../docs/build.md#2-安装软件包)正确安装toolkit和ops包, 并正确配置环境变量
2. 安装 Triton-Ascend，进入 [Triton-Ascend 的官网](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/installation_guide.md)，按照指导进行 Triton-Ascend 以及其所需依赖的安装。在您安装完毕后，强烈建议您运行此[链接](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/quick_start.md)对应的 Triton-Ascend 的[入门用例](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/quick_start.md)，以确保您环境安装正确。
3. 安装 TensorFlow 以及对应的框架插件包，可点击如下链接按需进行操作

   1. TensorFlow 1.15：[开源框架 TensorFlow 1.15](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/migration/tfmigr1/tfmigr1_000008.html)，[框架插件包](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/migration/tfmigr1/tfmigr1_000009.html)
   2. TensorFlow 2.6.5：[开源框架 TensorFlow 2.6.5](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/migration/tfmigr2/tfmigr2_000007.html)，[框架插件包](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/migration/tfmigr2/tfmigr2_000008.html)

## 1.2 实现步骤

1. 生成算子 kernel 的 npubin 文件，以 AddCustom 算子为例，其 kernel 的实现见：`custom_op/triton_add_custom/add_custom_kernel/add_custom_kernel.py`，执行该文件以生成 npubin

   ```bash
   python3 add_custom_kernel.py
   ```

   命令执行完成后，若无报错信息，则表明 npubin 已经生成，其生成路径默认在 `~/.triton/cache`​，实际使用过程中，可以通过 `TRITON_CACHE_DIR` 这个环境变量来指定 npubin 的生成位置
2. 提供 TF 的入图交付件，其实现如下，其路径为：`custom_op/triton_add_custom/tensorflow/add_custom_triton_tf.cc`​，可执行同级目录下的 `build.sh` 进行编译

   ```cpp
   bash build_tf.sh
   ```

   执行完成后，会在当前路径下生成：`./outputs/libcustom_ops.so`
3. 实现 GE 的入图交付件，路径为：`custom_op/triton_add_custom/src/custom_op.cpp`

   1. 实现继承类 `class AddCustom : public EagerExecuteOp`
   2. 注册宏 `REG_AUTO_MAPPING_OP(OP_CLASS)` ：注意: OP_CLASS 为类名, 命名空间需注意, 因为宏注册时直接将 OP_CLASS 作为字符串传入作为 Key 值.
   3. 执行 `bash run.sh`
   4. 根据 `run.sh`​ 的执行控制台输出添加环境变量至 `ASCEND_CUSTOM_OPP_PATH`

      ```shell
      # CUSTOM_INSTALL_PATH 应为 ${CMAKE_SOURCE_DIR}/build_out,以当前用例为例，CMAKE_SOURCE_DIR为custom_op/triton_add_custom
      export ASCEND_CUSTOM_OPP_PATH=${CUSTOM_INSTALL_PATH}:$ASCEND_CUSTOM_OPP_PATH
      ```
4. 测试脚本路径：`custom_op/triton_add_custom/script/run_add_custom_tf_1.15.py`，执行如下命令

   ```bash
   cd your_path/custom_op/triton_add_custom/script/
   python3 run_add_custom_tf_1.15.py &> triton_add.log
   grep "The result of tf and ac is" triton_add.log
   ```

   屏幕上若显示：`The result of tf and ac is the same.` 则表示用例执行成功

## 1.3 生成产物

- ​`libcustom_ops.so`​-用于将自定义算子入 TensorFlow 图  
  ​`${CUSTOM_INSTALL_PATH}/tensorflow/output/libcustom_ops.so`
- npu_supported_ops.json - 用于 TensorflowAdapter 加载自定义算子支持  
  ​`${CUSTOM_INSTALL_PATH}/framework/tensorflow/npu_supported_ops.json`
- libcust_opapi.so - 用于 GE 注册自定义算子  
  ​`${CUSTOM_INSTALL_PATH}/libcust_opapi.so`

## 1.4 目录结构

```txt
custom_op
└── triton_add_custom
    ├── add_custom_kernel
    │   └── add_custom_kernel.py      // Triton 算子的实现 kernel
    ├── CMakeLists.txt                // cmake文件
    ├── gen_npu_supported_ops_json.sh // 生成文件脚本
    ├── README.md                     // README
    ├── run.sh                        // GE 交付件编译脚本
    ├── script
    │   └── run_add_custom_tf_1.15.py // TF 1.15的测试脚本
    ├── src
    │   └── custom_op.cpp             // Triton 算子入 GE 交付件
    └── tensorflow
        ├── add_custom_triton_tf.cc   // Triton 算子入 TF 交付件
        └── build_tf.sh               // 编译入 TF 图的 so 的脚本
```
# 2. 特性约束
该特性依赖triton-ascend对应的支持范围，您可查看此[链接](https://gitcode.com/Ascend/triton-ascend#%E7%A1%AC%E4%BB%B6%E6%94%AF%E6%8C%81)来获取您当前产品的支持情况