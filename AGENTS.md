# CLAUDE.md

本文件为 agent 在此代码仓库中工作时提供指导。

## 项目概述

GE (Graph Engine) 是华为 CANN (Compute Architecture for Neural Networks) 的图引擎 - 面向昇腾 AI 处理器的高性能图编译器和执行器。GE 提供图优化、多流并行执行、内存复用优化和模型下沉等能力。

项目支持 PyTorch、TensorFlow、ONNX 和 PB 模型格式，以两种模式运行：
- **在线模式**：GE 作为深度学习框架的后端集成（通过 TorchAir、TFA）
- **离线模式**：使用 atc (Ascend Tensor Compiler) 进行独立模型编译

## 构建命令
### 基础构建
```bash
# 构建所有组件（ge_compiler、ge_executor、dflow）
bash build.sh

# 构建特定组件
bash build.sh --ge_compiler    # 构建编译器包
bash build.sh --ge_executor    # 构建执行器包
bash build.sh --dflow          # 构建 dflow 包
```

### 构建选项
```bash
bash build.sh -j16                  # 16个并行线程（默认：8）
bash build.sh --build_type=Debug    # Debug 构建（默认：Release）
bash build.sh --verbose             # 详细输出
bash build.sh --asan                # 启用 AddressSanitizer
bash build.sh --cov                 # 启用代码覆盖率
bash build.sh --output_path=<PATH>  # 设置自定义输出路径
```

### 第三方依赖
```bash
bash build_third_party.sh
```

## 测试

### GE 单元测试 / 系统测试
**使用技能**: `ge-ut-st`

**适用场景**: 编译和运行 GE 项目的单元测试(UT)和系统测试(ST)。

**指令格式**:
```
/skill ge-ut-st <目标或参数>
```

**支持操作**:
- 编译特定测试目标
- 运行特定测试用例
- 处理测试相关依赖和环境配置

**示例**:
- "编译并运行测试" → 使用 `/skill ge-ut-st`
- "运行某个特定用例" → 使用 `/skill ge-ut-st <测试名称>`

### 清理构建产物
```bash
rm -rf build_ut/ build_st/ output/ build/ build_out/ cov/
```

## 高层架构

### 核心组件

1. **GE 编译器** (`compiler/`)：将 AscendIR 编译为可执行模型（OM 文件）
   - 图级优化（公共子表达式消除、常量折叠、死代码消除）
   - 融合优化（基于 Pattern 和自动融合）
   - 算子在线编译
   - 流规划以实现并行
   - 内存规划与复用
   - 模型序列化

2. **GE 执行器** (`runtime/`)：在昇腾设备上执行模型
   - 模型加载（权重、二进制、执行序列）
   - 模型执行控制（分支、同步）
   - Sink 模式支持设备侧调度

3. **atc** (Ascend Tensor Compiler)：独立的离线编译工具
   - 将 ONNX/PB 模型转换为 AscendIR
   - 无需昇腾设备即可生成 OM 文件
   - 独立于框架运行时

### 关键目录

| 目录 | 用途 |
|------|------|
| `api/` | 公共 API 接口（ACL、ATC、session、Python 绑定） |
| `base/` | 基础组件（图结构、工具、主机 CPU 引擎） |
| `compiler/` | 图编译（分析器、引擎、图编译器、算子编译器） |
| `runtime/` | 运行时执行（C API、算子实现） |
| `dflow/` | 分布式流框架（LLM 数据分发、UDF） |
| `parser/` | 模型格式解析器（ONNX、PB、Caffe、MindSpore） |
| `graph_metadef/` | 图元数据定义和算子注册 |
| `tests/` | 综合测试套件（UT、ST、基准测试） |
| `examples/` | 使用示例和样例 |

### AscendIR (AIR) - 中间表示

- **静态计算图**，使用 DAG 结构
- **核心元素**：Graph（图）、Node（算子）、Tensor（张量）、Attribute（属性）、Data Edge（数据边）、Control Edge（控制边）
- **实现方式**：使用"锚点（Anchor）"系统（DataAnchor 表示数据流，CtrlAnchor 表示控制流）
- **统一入口**：所有输入（框架适配器、atc）在编译前都转换为 AscendIR

### 重要说明：算子仓库

**算子定义不在 GE 仓库中维护。** 算子在独立的仓库中定义（如 ops-math、ops-transformer 等），并在以下场景共享：
- **aclnn**：原生 API 调用
- **GE（图）**：入图编译和执行

这种分离确保：
- CANN 生态中统一的算子语义
- 算子独立演进
- GE 职责清晰（图编译，而非算子定义）

### 执行引擎

多个执行引擎处理不同类型的算子：
- **fe**：融合引擎
- **dvpp**：数字视觉预处理
- **aicpu**：AI CPU 操作
- **ffts**：FFT 操作
- **rts**：运行时服务
- **hcce**：HCCE 引擎
- **host_cpu_engine**：主机侧执行

## 开发规范

### 提交代码/创建PR
**使用技能**: `gitcode-pr`

**适用场景**: 创建 PR、推送代码到远程、使用 PR 模板。支持口语化表达如"提个PR"、"push代码"等。

**指令格式**:
```
/skill gitcode-pr
```

### 读取 issue
**使用技能**: `gitcode-issue`

**适用场景**: 读取 GitCode issue 详情、评论。支持 URL、issue 编号或口语化表达（如"看看issue 123"）。

**指令格式**:
```
/gitcode-issue <issue_url 或 issue_number>
```

### 代码风格
- 遵循 Google 开源代码规范

## 环境要求
- GCC >= 7.3.x
- Python 3 >= 3.9.x（需额外 `pip3 install coverage`）
- CMake >= 3.16.0（推荐 3.20.0）
- 需要安装 CANN Toolkit
- 第三方库：protobuf、grpc、boost、gtest 等

## 语言
使用中文