# 历史原型库协议（History Registry Protocol）

## 1. 定位与职责边界

历史原型库是一套**长期维护的协议与数据产物**，用于**跨版本归档算子 IR 原型信息**，并以稳定的方式对外提供查询能力。

本协议关注：
- 数据是什么（字段语义与约束）
- 数据放在哪（目录与分包）
- 数据怎么生成与发布（当前由 `gen_esb` 生成并随 Ops run 包发布）
- 消费者如何读取（文件系统接口）
- 协议如何演进（向后兼容规则）

本协议不关注：
- 兼容性判定/业务决策逻辑

## 2. 术语

- **IR 原型**：算子在 IR 层的输入/输出/属性/子图等定义信息。
- **历史原型数据（结构化数据）**：将 IR 原型提取并序列化后的数据文件集合。
- **生产者（Producer）**：生成结构化数据的工具。短期内固定为 `gen_esb --extract-history`。
- **消费者（Consumer）**：读取结构化数据的任意工具/流程。当前主要消费者是 `gen_esb`（用于 ES API 生成），未来可扩展到其他工具链。
- **Ops run 包**：Ops 安装后的运行包形态，历史原型库数据随其发布。
- **兼容窗口**：消费者基于版本元信息（例如发布日期）选择需要纳入对比/生成的版本范围（例如“一年窗口”）。

## 3. 目录结构与分包

历史原型库随 Ops run 包发布，按算子分包（如 `math/nn/cv/...`）组织。

目录结构（示例，安装后）：

```
/${CANN_INSTALL_PATH}/cann/opp/history_registry/<pkg>/
├── index.json
└── registry/
    ├── <release_version_1>/
    │   ├── operators.json
    │   └── metadata.json
    ├── <release_version_2>/
    │   ├── operators.json
    │   └── metadata.json
    └── ...
```

说明：
- `<pkg>`：分包名，例如 `math`、`nn`、`cv`。
- `<release_version_x>`：版本目录名，建议与 `metadata.json.release_version` 一致（例如 `8.0.RC1`）。

## 4. 文件与数据格式

本协议包含三个文件类型：
- `index.json`：版本索引（该分包有哪些版本可用）
- `registry/<ver>/metadata.json`：版本元信息（用于选择版本范围、追溯）
- `registry/<ver>/operators.json`：该版本的算子原型数据

### 4.1 index.json（版本索引）

最小字段：
- `version`：索引文件 schema 版本（字符串）
- `releases[]`：版本数组
  - `release_version`：版本号（字符串）
  - `release_date`：发布日期（字符串，建议 `YYYY-MM-DD`）

示例见[附录 A](#附录-ajson-最小示例)。


### 4.2 metadata.json（版本元信息）

建议字段（最小可用 + 可扩展）：
- `release_version`：版本号
- `branch_name`：分支名（可选，但建议提供）

示例见[附录 A](#附录-ajson-最小示例)。


### 4.3 operators.json（算子原型数据）

最小字段：
- `operators[]`
  - `op_type`
  - `inputs[]`：每项至少包含 `name/type/dtype`
  - `outputs[]`：每项至少包含 `name/type/dtype`
  - `attrs[]`：每项至少包含 `name/type/required`，可包含 `default_value`
  - `subgraphs[]`：每项至少包含 `name/type`

约束建议：
- `default_value` 使用字符串承载“JSON 字面量文本”，解析时需结合 `type`；用于跨语言一致性，并避免数值精度/大整数表示等问题。

示例见[附录 A](#附录-ajson-最小示例)。

## 5. 生成与发布（当前固定由 gen_esb 负责）

短期内，为了降低系统复杂度：
- **生产者**：仍复用 `gen_esb`，通过 `--extract-history` 从“当前版本原型 .so”提取 IR 原型并输出结构化数据。

### 5.1 构建 Ops 包的数据流（生成与打包发布）

前置要求：
- 构建环境需要安装 Ops 包（历史原型数据来源于已安装的 Ops run 包）；若无 Ops 包则无法获取历史原型数据。

构建 Ops 包数据流图：

![数据流图](./figures/data_flow_build.svg)

说明：
- 各个版本的历史原型信息存在于 Ops run 包中，按不同的分包（如 `math/nn/cv`）分类存放。

简化后的数据流图：

![简化数据流图](./figures/data_flow_build_simply.svg)

生成模式（示例）：

```bash
gen_esb --extract-history --output-dir <out> --release-version <ver>
```

发布原则：
- 在正式版本发布前生成并归档本版本数据。
- 结构化数据随 Ops run 包打包发布。

## 6. 消费方式（文件系统接口）

消费者按以下方式读取：
- 读取 `<pkg>/index.json` 获取可用版本列表与发布日期
- 选择目标版本集合（例如“一年窗口”）
- 读取 `registry/<ver>/operators.json` 与 `metadata.json` 获取原型与元信息

注意：
- 协议层不绑定任何特定消费策略；消费策略（例如生成哪些重载、如何消歧）由消费者实现。

## 7. 协议演进与向后兼容

原则：
- 优先**新增字段**，避免删除/重命名已发布字段。
- 新增字段需提供默认语义，确保旧消费者可忽略。
- 需要破坏性变更时，应通过 `index.json.version`（或引入显式 `schema_version`）区分，并给出迁移策略。

## 附录 A：JSON 最小示例

### A.1 index.json

```json
{
  "version": "1.0.0",
  "releases": [
    { "release_version": "8.0.RC1", "release_date": "2024-09-30" },
    { "release_version": "8.0.0", "release_date": "2024-12-30" }
  ]
}
```

### A.2 registry/<ver>/metadata.json

```json
{
  "release_version": "8.0.RC1",
  "branch_name": "master"
}
```

### A.3 registry/<ver>/operators.json

```json
{
  "operators": [
    {
      "op_type": "Foo",
      "inputs": [
        { "name": "x", "type": "INPUT", "dtype": "TensorType({DT_FLOAT})" },
        { "name": "xo1", "type": "OPTIONAL_INPUT", "dtype": "TensorType({DT_FLOAT})" }
      ],
      "outputs": [
        { "name": "y", "type": "OUTPUT", "dtype": "TensorType({DT_FLOAT})" }
      ],
      "attrs": [
        { "name": "a", "type": "Int", "required": false, "default_value": "0" },
        { "name": "flag", "type": "Bool", "required": true }
      ],
      "subgraphs": []
    }
  ]
}
```

