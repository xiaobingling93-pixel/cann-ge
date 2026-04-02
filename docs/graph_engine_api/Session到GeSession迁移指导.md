# Session 到 GeSession 迁移指导

## 概述

本文档旨在指导用户从 `Session` 类迁移到新引入的 `GeSession` 类。`GeSession` 是对原有 `Session` 类的重构和优化，主要变化包括：

- 移除了与 DataFlow 相关的接口
- 简化了编译和加载流程
- 统一了执行接口的参数类型（从 `ge::Tensor` 改为 `gert::Tensor`）
- 优化了接口命名和参数类型

## 库链接变化

| 项目 | Session | GeSession |
|------|---------|-----------|
| 库文件 | `libge_runner.so` | `libge_runner_v2.so` |
| 头文件 | `ge/ge_api.h` | `ge/ge_api_v2.h` |

## 接口对比表

| Session 接口 | GeSession 接口 | 迁移说明                                                                                                |
|--------------|----------------|-----------------------------------------------------------------------------------------------------|
| `Session(options)` | `GeSession(options)` | 构造函数基本一致，但只提供了ABI兼容的 `std::map<AscendString, AscendString>` 类型版本                                    |
| `~Session()` | `~GeSession()` | 析构函数，无变化                                                                                            |
| `AddGraph(uint32_t, const Graph&)` | `AddGraph(uint32_t, const Graph&)` | 接口保持一致                                                                                              |
| `AddGraph(uint32_t, const Graph&, options)` | `AddGraph(uint32_t, const Graph&, options)` | 只提供了ABI兼容的 std::map<AscendString, AscendString> 类型版本          |
| `AddGraphWithCopy` | `AddGraphClone` | 重命名，功能相同                                                                                            |
| `RemoveGraph` | `RemoveGraph` | 无变化                                                                                                 |
| `BuildGraph` | `CompileGraph` | 重命名，功能相同。GeSession 的 CompileGraph 支持 Variable                                                       |
| `CompileGraph` | `CompileGraph` | 合并了 BuildGraph 和 CompileGraph 的功能                                                                   |
| `LoadGraph` | `LoadGraph` | 接口基本一致，但 GeSession 中会自动检查是否需要先 CompileGraph                                                         |
| `RunGraph` | `RunGraph` | **重要变化**：输入输出从 `ge::Tensor` 改为 `gert::Tensor`                                                       |
| `RunGraphWithStreamAsync` | `RunGraphWithStreamAsync` | **重要变化**：输入输出从 `ge::Tensor` 改为 `gert::Tensor`；CompileGraph 和 LoadGraph 可省略                          |
| `ExecuteGraphWithStreamAsync` | `RunGraphWithStreamAsync` | Session 中的 `ExecuteGraphWithStreamAsync`（使用 `gert::Tensor`）合并到 GeSession 的 `RunGraphWithStreamAsync` |
| `RunGraphAsync` | `RunGraphAsync` | **重要变化**：输入输出从 `ge::Tensor` 改为 `gert::Tensor`；回调函数签名从 `RunAsyncCallback` 改为 `RunAsyncCallbackV2`    |
| `RegisterCallBackFunc` | `RegisterCallBackFunc` | 回调函数签名变化，使用 `RunCallback` 类型                                                                        |
| `GetCompiledGraphSummary` | `GetCompiledGraphSummary` | 无变化                                                                                                 |
| `SetGraphConstMemoryBase` | `SetGraphConstMemoryBase` | 无变化                                                                                                 |
| `UpdateGraphFeatureMemoryBase` | `UpdateGraphFeatureMemoryBase` | 无变化                                                                                                 |
| `SetGraphFixedFeatureMemoryBase` | `SetGraphFixedFeatureMemoryBaseWithType` | 接口名称变化，增加了 type 参数                                                                                  |
| `UpdateGraphRefreshableFeatureMemoryBase` | `UpdateGraphRefreshableFeatureMemoryBase` | 无变化                                                                                                 |
| `RegisterExternalAllocator` | `RegisterExternalAllocator` | 无变化                                                                                                 |
| `UnregisterExternalAllocator` | `UnregisterExternalAllocator` | 无变化                                                                                                 |
| `IsGraphNeedRebuild` | `IsGraphNeedRebuild` | 无变化                                                                                                 |
| `GetSessionId` | `GetSessionId` | 无变化                                                                                                 |
| - | `GetCompiledModel` | **新增接口**：获取编译后的模型数据                                                                                 |
| `FeedDataFlowGraph` | - | **已删除**，迁移到 DataFlow 接口                                                                             |
| `FetchDataFlowGraph` | - | **已删除**，迁移到 DataFlow 接口                                                                             |
| `FeedRawData` | - | **已删除**，迁移到 DataFlow 接口                                                                             |
| `GetVariables` | - | **已删除**，无替代接口                                                                                       |
| `ShardGraphsToFile` | - | **已删除**，图分片功能不再提供                                                                                   |
| `ShardGraphs` | - | **已删除**，图分片功能不再提供                                                                                   |
| `SaveGraphsToPb` | - | **已删除**，保存图到pb文件功能不再提供                                                                              |
| `PaRemapped` | - | **已删除**，虚拟内存重映射功能不再提供                                                                               |

## 初始化接口变化

### GEInitialize/GEFinalize

| Session (ge_api.h) | GeSession (ge_api_v2.h) |
|--------------------|-------------------------|
| `GEInitialize(options)` | `GEInitializeV2(options)` |
| `GEFinalize()` | `GEFinalizeV2()` |
| `GEGetErrorMsg()` | `GEGetErrorMsgV3()` |
| `GEGetWarningMsg()` | `GEGetWarningMsgV3()` |

## 最大的变化点：Tensor 类型

### ge::Tensor vs gert::Tensor

| 特性 | ge::Tensor | gert::Tensor |
|------|------------|--------------|
| 命名空间 | `ge` | `gert` |
| 数据结构 | 使用 `std::shared_ptr<TensorImpl>` 管理内部实现 | **POD类型**（Plain Old Data），所有数据内联存储 |
| 内存布局 | 间接访问，通过 impl_ 指针 | 扁平化布局，支持直接memcpy |
| Placement支持 | 通过 TensorDesc 设置 Placement | 支持多种 Placement 类型 |
| 拷贝行为 | 浅拷贝（shared_ptr语义） | 浅拷贝，指针共享 |
| 性能 | 一般 | 高性能 |
| 适用场景 | 图构建阶段 | 运行时执行 |

### 构造 gert::Tensor

```cpp
// 方法1: 使用 gert::Tensor 的基本构造（需要包含 "exe_graph/runtime/tensor.h"）
#include "exe_graph/runtime/tensor.h"

// 创建 host tensor
gert::Tensor tensor;
// 设置 shape、数据类型等
tensor.SetShape(...);
tensor.SetDataType(...);
tensor.SetPlacement(gert::TensorPlacement::kOnHost);

// 方法2: 从已有数据构造
void* data = ...;  // 已有数据指针
size_t size = ...; // 数据大小
gert::Tensor tensor(data, size, gert::TensorPlacement::kOnHost);
```

### Tensor 生命周期说明

#### RunGraph 接口

```cpp
std::vector<gert::Tensor> inputs = ...;
std::vector<gert::Tensor> outputs;
session->RunGraph(graph_id, inputs, outputs);
// inputs 和 outputs 在调用完成后可以安全释放
```

#### RunGraphWithStreamAsync 接口

```cpp
// GeSession 的 RunGraphWithStreamAsync
// 注意：可以不先调用 CompileGraph 和 LoadGraph，会自动处理
std::vector<gert::Tensor> inputs = ...;
std::vector<gert::Tensor> outputs;
session->RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
// inputs 和 outputs 在 stream 同步之前不能释放
// 需要调用 aclrtSynchronizeStream(stream) 或其他同步机制
```

#### RunGraphAsync 接口（**重要**）

```cpp
using RunAsyncCallbackV2 = std::function<void(Status, std::vector<gert::Tensor>&)>;
std::vector<gert::Tensor> inputs = ...;
session->RunGraphAsync(graph_id, inputs,
    [](Status ret, std::vector<gert::Tensor>& outputs) {
        // 处理输出
    });
// ⚠️ 重要：inputs 不能立即释放！
// 必须等到 callback 函数被调用后才能释放 inputs
// 因为模型执行的时候会读取inputs，callback被调用可保证模型执行完成
```

## 重要变化点详解

### 1. CompileGraph/LoadGraph 不再必需

在 GeSession 中，`RunGraph`、`RunGraphAsync` 和 `RunGraphWithStreamAsync` 三个执行接口会自动检查图是否已编译和加载。如果未编译，会先自动编译；如果未加载，会先自动加载。

```cpp
// GeSession 的自动处理机制
GeSession session(options);
session.AddGraph(graph_id, graph);

// 直接执行，无需手动 CompileGraph 和 LoadGraph
session.RunGraph(graph_id, inputs, outputs);  // 自动编译和加载
```

### 2. 执行模式互斥

GeSession 的三种执行模式（`RunGraph`、`RunGraphAsync`、`RunGraphWithStreamAsync`）是互斥的，不能混用。一旦使用了某种执行模式，该图就必须继续使用同一种模式。

```cpp
// 错误示例
session.RunGraph(graph_id, inputs1, outputs1);  // 使用 RunGraph 模式
session.RunGraphAsync(graph_id, inputs2, callback);  // 错误！不能混用
```

## gert::Tensor 构造示例

### 数据结构对比

#### ge::Tensor 的内部结构
```cpp
class Tensor {
private:
  std::shared_ptr<TensorImpl> impl_;  // 使用智能指针管理
};
```
- 使用 shared_ptr 管理 TensorImpl
- 拷贝时共享底层实现
- 数据通过 TensorDesc 描述

#### gert::Tensor 的内部结构
```cpp
class Tensor {
private:
  StorageShape storage_shape_;        // Shape信息
  StorageFormat storage_format_;      // Format信息
  TensorVersion version_;             // 版本
  uint8_t reserved_[3];               // 预留字段
  ge::DataType data_type_;            // 数据类型
  TensorData tensor_data_;            // 数据指针和placement
  uint8_t reserved_field_[40];        // 预留字段
};
```
- 所有字段直接内联在对象中
- 是标准布局类型（`std::is_standard_layout`）

### 构造 gert::Tensor 的方法

#### 方法1: 使用 TensorData 构造（推荐）

```cpp
#include "exe_graph/runtime/tensor.h"
#include "acl_rt.h"

// 构造 Host Tensor
void* host_buf = nullptr;
aclError ret = aclrtMallocHost(&host_buf, data_len);  // 分配Host内存
if (ret != ACL_ERROR_NONE) {
    // 处理错误
}

// 使用 TensorData 构造
gert::TensorData td(host_buf, nullptr, data_len, gert::kOnHost);
gert::Tensor tensor;
tensor.SetData(std::move(td));

// 设置数据类型（如果需要）
// tensor.SetDataType(ge::DT_FLOAT);
```

#### 方法2: 使用构造函数直接创建

```cpp
// 从 shape、format 和 dtype 构造
gert::StorageShape shape = {{batch_size, channels, height, width}, {4}};
gert::StorageFormat format = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
gert::Tensor tensor(shape, format, ge::DT_FLOAT);

// 然后分配内存
void* host_buf = nullptr;
aclrtMallocHost(&host_buf, tensor.GetSize());
gert::TensorData td(host_buf, nullptr, tensor.GetSize(), gert::kOnHost);
tensor.SetData(std::move(td));
```

#### 方法3: 构造 Device Tensor

```cpp
// 分配Device内存
void* dev = nullptr;
aclError ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
if (ret != ACL_ERROR_NONE) {
    // 处理错误
}

// 构造 Device Tensor
gert::TensorData td(dev, nullptr, bytes, gert::kOnDeviceHbm);
gert::Tensor device_tensor;
device_tensor.SetData(std::move(td));
```

### gert::Tensor 常用方法

```cpp
gert::Tensor tensor;

// 获取数据地址
void* addr = tensor.GetAddr();

// 获取数据大小（字节数）
size_t size = tensor.GetSize();

// 获取数据类型
ge::DataType dtype = tensor.GetDataType();

// 获取元素数量
uint32_t elem_count = tensor.GetShapeSize();

// 设置数据
gert::TensorData td(data, nullptr, size, gert::kOnHost);
tensor.SetData(std::move(td));
```

### 完整示例：构造输入数据

```cpp
#include "exe_graph/runtime/tensor.h"
#include "acl_rt.h"

ge::Status PrepareInputTensor(std::vector<gert::Tensor> &inputs, int32_t batch_size) {
    // 定义输入shape
    gert::StorageShape shape = {{batch_size, 3, 224, 224}, {4}};
    gert::StorageFormat format = {ge::FORMAT_ND, ge::FORMAT_ND, {}};

    // 创建Tensor
    gert::Tensor tensor(shape, format, ge::DT_FLOAT);

    // 计算所需内存大小
    const uint32_t elem_count = tensor.GetShapeSize();
    const uint32_t data_size = elem_count * sizeof(float);

    // 分配Host内存
    void *host_buf = nullptr;
    aclError ret = aclrtMallocHost(&host_buf, data_size);
    if (ret != ACL_ERROR_NONE) {
        return ge::FAILED;
    }

    // 填充数据
    float *data_ptr = reinterpret_cast<float*>(host_buf);
    for (uint32_t i = 0; i < elem_count; ++i) {
        data_ptr[i] = 1.0f;  // 填充示例值
    }

    // 设置Tensor数据
    gert::TensorData td(host_buf, nullptr, data_size, gert::kOnHost);
    tensor.SetData(std::move(td));

    inputs.push_back(std::move(tensor));
    return ge::SUCCESS;
}
```

### 释放 gert::Tensor 内存

```cpp
// 释放Host Tensor内存
void FreeHostTensor(gert::Tensor &tensor) {
    if (tensor.GetAddr() != nullptr) {
        aclrtFreeHost(tensor.GetAddr());
    }
}

// 释放Device Tensor内存
void FreeDeviceTensor(gert::Tensor &tensor) {
    if (tensor.GetAddr() != nullptr) {
        aclrtFree(tensor.GetAddr());
    }
}

// 批量释放
void FreeTensorVector(std::vector<gert::Tensor> &tensors, bool is_device) {
    for (auto &t : tensors) {
        if (t.GetAddr() != nullptr) {
            if (is_device) {
                aclrtFree(t.GetAddr());
            } else {
                aclrtFreeHost(t.GetAddr());
            }
        }
    }
}
```

## 编译配置变化

Makefile 或 CMakeLists.txt 需要更新：

```cmake
# 旧配置
target_link_libraries(your_app libge_runner.so)

# 新配置
target_link_libraries(your_app libge_runner_v2.so)
```

## 注意事项总结

1. **头文件变化**：从 `ge/ge_api.h` 改为 `ge/ge_api_v2.h`
2. **库文件变化**：从 `libge_runner.so` 改为 `libge_runner_v2.so`
3. **Tensor 类型变化**：所有 Run 接口的输入输出从 `ge::Tensor` 改为 `gert::Tensor`
4. **异步生命周期**：使用 `RunGraphAsync` 时，inputs 必须保持有效直到 callback 被调用
5. **接口简化**：不需要手动调用 CompileGraph 和 LoadGraph（除非需要显式控制）
6. **DataFlow 分离**：Feed/Fetch 接口已删除，使用 DataFlow 专用接口
7. **执行模式互斥**：三种 Run 模式不能混用
8. **回调函数签名变化**：从 `RunAsyncCallback` 改为 `RunAsyncCallbackV2`
