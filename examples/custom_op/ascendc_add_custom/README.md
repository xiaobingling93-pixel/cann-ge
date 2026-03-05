# README

# torch注册自定义算子直调样例

## 概述

本样例展示了如何使用PyTorch的torch.library机制注册自定义算子，并通过`<<<>>>`内核调用符调用核函数，以简单的Add算子为例，实现两个向量的逐元素相加。

## 算子描述

- 算子功能：

  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：

  ```
  z = x + y
  ```

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddCustom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- 算子实现：

  Ascend C提供的矢量计算接口`Add`的操作元素都为`LocalTensor`，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：`CopyIn`，`Compute`，`CopyOut`。`CopyIn`任务负责将Global Memory上的输入Tensor `xGm`和`yGm`搬运到Local Memory，分别存储在`xLocal`、`yLocal`，`Compute`任务负责对`xLocal`、`yLocal`执行加法操作，计算结果存储在`zLocal`中，`CopyOut`任务负责将输出数据从`zLocal`搬运至Global Memory上的输出Tensor zGm中。

- 自定义算子注册：

  本样例在`add_custom.asc`中定义了一个名为`ascendc_ops`的命名空间，并在其中注册了`ascendc_add`函数。

  PyTorch提供`TORCH_LIBRARY`宏作为自定义算子注册的核心接口，用于创建并初始化自定义算子库，注册后在Python侧可以通过`torch.ops.namespace.op_name`方式进行调用，例如：

  ```c++
  TORCH_LIBRARY(ascendc_ops, m) {
      m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
  }
  ```

  `TORCH_LIBRARY_IMPL`用于将算子逻辑绑定到特定的`DispatchKey`（PyTorch设备调度标识）。针对NPU设备，需要将算子实现注册到`PrivateUse1`和`XLA`这类专属的`DispatchKey`上，例如：

  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```
  
  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, XLA, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```

  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, Meta, m) {
    m.impl("ascendc_add", [](const torch::Tensor& x, const torch::Tensor& y) {
    return at::empty_like(x);
    });
  }
  ```
- GE入图
  1、算子执行类（AddCustom）：继承EagerExecuteOp，实现Execute方法，完成输入输出 Tensor 的获取、内存申请、核函数启动等核心逻辑。
  2、形状推导函数（InferShapeForAdd）：实现 PyTorch 风格的广播语义，推导输出 Tensor 的形状。
  3、数据类型推导函数（InferDataTypeForAdd）：指定输出数据类型与第一个输入保持一致。

  算子注册：通过宏定义完成算子的输入输出类型配置、形状 / 数据类型推导绑定、自动映射注册。

  ```c++
  IMPL_OP(AddCustom).InferShape(InferShapeForAdd).InferDataType(InferDataTypeForAdd);
  ```

- Python测试脚本

  在`add_custom_test.py`调用脚本中，通过`torch.ops.load_library`加载生成的自定义算子库，调用注册的`ascendc_add`函数，并通过对比NPU输出与CPU标准加法结果来验证自定义算子的数值正确性。

## 编译运行

- 安装PyTorch以及Ascend Extension for PyTorch插件(torchair要求python版本≥3.8)

  请参考[pytorch: Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)开源代码仓或[Ascend Extension for PyTorch昇腾社区](https://hiascend.com/document/redirect/Pytorch-index)的安装说明，选取支持的`Python`版本配套发行版，完成`torch`和`torch-npu`的安装。

- 安装前置依赖

  ```bash
  pip3 install expecttest
  ```

- 配置环境变量

  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
   - 默认路径，root用户安装CANN软件包

     ```bash
     source /usr/local/Ascend/cann/set_env.sh
     ```

   - 默认路径，非root用户安装CANN软件包

     ```bash
     source $HOME/Ascend/cann/set_env.sh
     ```

   - 指定路径install_path，安装CANN软件包

     ```bash
     source ${install_path}/cann/set_env.sh
     ```

- 样例执行

  在本样例根ascendc_add_custom目录下执行如下步骤，运行该样例。

  ```bash
  mkdir -p build; cd build
  cmake ..; make -j
  python3 ../script/add_custom_test.py
  ```

  屏幕上若显示：`Ran 1 test in **s` 则表示用例执行成功。

  ```bash
  Ran 1 test in **s
  OK
  ```

## 1.3 生成产物

- ​`libcust_opapi.so`​-用于将自定义算子入Torch图  
  ​`${CUSTOM_INSTALL_PATH}/build/libcust_opapi.so`

## 目录结构介绍

```
custom_op
└── ascendc_add_custom
    ├── add_custom_kernel
    │   └── add_custom.asc            // ascendc算子的实现kernel
    │   └── add_custom_kernel.h       // ascendc算子声明
    │   └── custom_op.cpp             // ascendc算子入GE交付件
    ├── script
    │   └── add_custom_test.py        // ascendc算子的测试脚本
    ├── CMakeLists.txt                // cmake文件
    └── README.md                     // README
```