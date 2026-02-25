/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_TYPES_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_TYPES_H_

#include <string>
#include <vector>
#include "graph/op_desc.h"

namespace ge {
namespace es {
namespace history {

struct IrInput {
  std::string name;
  IrInputType type;
  std::vector<std::string> dtype;
};

struct IrOutput {
  std::string name;
  IrOutputType type;
  std::vector<std::string> dtype;
};

struct IrAttr {
  std::string name;
  std::string av_type;
  bool required;
  std::string default_value;
};

struct IrSubgraph {
  std::string name;
  SubgraphType type;
};

// IR 算子原型的内存表示，参数向量保留 IR 定义顺序
struct IrOpProto {
  std::string op_type;
  std::vector<IrInput> inputs;
  std::vector<IrOutput> outputs;
  std::vector<IrAttr> attrs;
  std::vector<IrSubgraph> subgraphs;
};

struct VersionMeta {
  std::string release_version;
  std::string release_date;
  std::string branch_name;
};

}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_TYPES_H_
