# ES API 生成说明

本文档说明当前测试框架下 **Eager Style (ES) API** 的生成方式，以及当需要**其它算子的 ES API 构图**时应修改的位置。

---

## 一、ES API 当前是如何生成的

### 1.1 整体流程

ES API 采用 **“算子原型 + generate_es_package.cmake函数”** 的方式生成：

1. **算子原型**：在 `proto/test_ops.h` 中用 `REG_OP(OpName)...OP_END_FACTORY_REG(OpName)` 添加算子原型。
2. **原型库**：将上述文件编译为动态库 `test_ops_proto`，该库在加载时会向 GE 注册其中所有算子。
3. **代码生成**：CMake 在配置/构建时调用 **gen_esb** 工具，为每个算子生成 C / C++  的 ES 构图 API。
4. **产物**：生成 `es_ge_test_ops.h`、`es_ge_test_ops_c.h` 以及每个算子对应的 `es_<OpType>.cpp`、`es_<OpType>.h` 等，并编译成库 `es_ge_test`。

因此：**能参与 ES API 生成的算子，必须先在 `proto/test_ops.h`里用 REG_OP 注册。**

### 1.2 关键组件与路径

| 组件 | 路径/说明 |
|------|------------|
| 算子原型定义 | `tests/framework/eager_style_graph_builder/proto/test_ops.h`（及同目录 `test_ops.cc`） |
| 生成脚本 | `cmake/generate_es_package.cmake`（提供 `add_es_library`） |
| 本目录 CMake | `tests/framework/eager_style_graph_builder/CMakeLists.txt` |
| 生成产物目录 | 构建目录下 `eager_style_graph_builder/build/es_output/`，头文件在 `include/es_ge_test/` |

### 1.3 生成触发方式（CMake）

在 `tests/framework/eager_style_graph_builder/CMakeLists.txt` 中：

- 通过 `include(generate_es_package.cmake)` 引入生成算子的cmake函数。
- 先定义并编译算子原型库 `test_ops_proto`（源文件为 `proto/test_ops.cc`，会包含 `test_ops.h` 中的 REG_OP）。
- 再调用：

```cmake
add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_ge_test
    OPP_PROTO_TARGET test_ops_proto
    OUTPUT_PATH ${ES_OUTPUT_PATH}
)
```

`add_es_library` 会：

1. 根据 `test_ops_proto` 的输出路径得到算子原型所在目录
2. 在构建时设置 `ASCEND_OPP_PATH` 并执行 **gen_esb**。
3. gen_esb 加载该路径下的算子库，生成 C/C++ 代码。
4. 将生成代码编译为 `es_ge_test` 的动态库，并安装到 `ES_OUTPUT_PATH`。

因此：**ES API 的算子集合完全由“被 gen_esb 加载的那份算子原型”决定**，当前就是 `test_ops_proto` 所包含的 REG_OP。

### 1.4 与 gen_esb 的对应关系

- gen_esb 的输入：环境变量 `ASCEND_OPP_PATH` 指向的目录（即由 `test_ops_proto` 推导出的 OPP 路径）。
- gen_esb 会加载该路径下的算子 .so，读取其中通过 REG_OP 注册的算子元信息，然后为每个算子生成：
  - C 接口：`es_<OpType>.cpp` + 声明在 `es_ge_test_ops_c.h`
  - C++ 接口：`es_<OpType>.h`，汇总在 `es_ge_test_ops.h`

---

## 二、需要其它算子的 ES API 构图时该改哪里

若希望用 **更多算子** 的 ES API 构图，只需保证这些算子在 gen_esb 运行时的“算子原型库”里被注册。当前该库就是由本目录下的 `proto/` 构建出的 `test_ops_proto`。

### 2.1 增加新算子

**修改位置：** `tests/framework/eager_style_graph_builder/proto/test_ops.h`（或配合 `proto/test_ops.cc`）

**操作：** 在 `test_ops.h` 中按现有格式增加一行 `REG_OP(NewOpName)...OP_END_FACTORY_REG(NewOpName)`，定义好输入/输出/属性（可参考同文件中的算子原型）。

- 若仅头文件即可完成注册，则只需改 `test_ops.h`。
- 若该算子在其他模块已有 REG_OP 定义，也可在 `test_ops.cc` 里 `#include` 对应头文件，保证链接进 `test_ops_proto.so`。

保存后重新构建，gen_esb 会生成 对应的es算子，并纳入 `es_ge_test_ops.h` / `es_ge_test_ops_c.h`。测试用例中即可：

- 使用 C++：`#include "es_ge_test_ops.h"`，调用 `es::NewOpName(...)` 等。
- 使用 C：`#include "es_ge_test_ops_c.h"`，调用生成的 C 函数。

更多 `add_es_library` 参数与行为见 `cmake/generate_es_package.cmake` 内注释。
