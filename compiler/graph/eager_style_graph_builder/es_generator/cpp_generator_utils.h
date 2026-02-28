/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_UTILS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_UTILS_H_

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "graph/op_desc.h"
#include "history/overload_planner_types.h"

namespace ge {
namespace es {
namespace cpp_gen {
struct LoweredInputArg {
  const ge::es::history::Param *param = nullptr;
  ge::IrInputType type;
  std::string ir_name;
};

struct LoweredDynamicOutputArg {
  const ge::es::history::Param *param = nullptr;
  std::string ir_name;
};

struct LoweredSubgraphArg {
  const ge::es::history::Param *param = nullptr;
  ge::SubgraphType type;
  std::string ir_name;
};

// LoweredSignature 是面向代码生成的视图，由原始签名(Signature)与 Op 的 IR 顺序联合派生。
// 原始签名(Signature)描述规划后的 C++ API：参数列表、默认值、重载语义、参数角色等。
// LoweredSignature 将这些参数对齐到 C 接口所需的 IR 顺序，
// 并在缺失处补空位（例如 nullptr 占位），便于统一生成调用参数。
struct LoweredSignature {
  std::vector<LoweredInputArg> inputs;
  const ge::es::history::Param *owner_builder = nullptr;
  std::vector<LoweredDynamicOutputArg> dynamic_outputs;
  std::vector<LoweredSubgraphArg> subgraphs;
};

inline void AppendCallArg(std::stringstream &ss, bool &first, const std::string &expr) {
  if (first) {
    first = false;
  } else {
    ss << ", ";
  }
  ss << expr;
}

inline const ge::es::history::Param *FindParamByRoleAndIrName(const ge::es::history::Signature &sig,
                                                              const ge::es::history::ParamRole role,
                                                              const std::string &ir_name) {
  for (const auto &param: sig.params) {
    if (param.role == role && param.ir_name == ir_name) {
      return &param;
    }
  }
  return nullptr;
}

inline const ge::es::history::Param *FindOwnerBuilderParam(const ge::es::history::Signature &sig) {
  for (const auto &param: sig.params) {
    if (param.role == ge::es::history::ParamRole::kOwnerBuilder) {
      return &param;
    }
  }
  return nullptr;
}

inline bool HasTensorLikeParam(const ge::es::history::Signature &sig) {
  for (const auto &param: sig.params) {
    if (param.role == ge::es::history::ParamRole::kInput &&
      param.kind == ge::es::history::ParamCxxKind::kEsTensorLikeRef) {
      return true;
    }
  }
  return false;
}

inline bool UseTensorLikeInSignature(const ge::es::history::Signature &sig, const std::string &ir_name) {
  const auto *param = FindParamByRoleAndIrName(sig, ge::es::history::ParamRole::kInput, ir_name);
  return param != nullptr && param->kind == ge::es::history::ParamCxxKind::kEsTensorLikeRef;
}

/** C 接口 owner_builder 入参判定规则与 C 声明保持一致：
 * 1. 无输入时必有 owner_builder
 * 2. 输入全部 optional 时有 owner_builder
 * 3. 其他场景没有 owner_builder
 */
inline bool NeedOwnerBuilderCArg(const LoweredSignature &lowered) {
  if (lowered.inputs.empty()) {
    return true;
  }
  for (const auto &input : lowered.inputs) {
    if (input.type != kIrInputOptional) {
      return false;
    }
  }
  return true;
}

inline LoweredSignature LowerSignature(const OpDescPtr &op, const ge::es::history::Signature &sig) {
  LoweredSignature lowered;
  for (const auto &in: op->GetIrInputs()) {
    lowered.inputs.push_back({
      FindParamByRoleAndIrName(sig, ge::es::history::ParamRole::kInput, in.first),
      in.second,
      in.first
    });
  }
  lowered.owner_builder = FindOwnerBuilderParam(sig);
  for (const auto &out: op->GetIrOutputs()) {
    if (out.second != kIrOutputDynamic) {
      continue;
    }
    lowered.dynamic_outputs.push_back(
      {
        FindParamByRoleAndIrName(sig, ge::es::history::ParamRole::kDynamicOutputNum, out.first),
        out.first
      });
  }
  for (const auto &subgraph: op->GetOrderedSubgraphIrNames()) {
    lowered.subgraphs.push_back(
      {
        FindParamByRoleAndIrName(sig, ge::es::history::ParamRole::kSubgraph, subgraph.first),
        subgraph.second,
        subgraph.first
      });
  }
  return lowered;
}

inline std::string JoinNames(const std::vector<std::string> &names) {
  std::stringstream ss;
  bool first = true;
  for (const auto &name : names) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << name;
  }
  return ss.str();
}

inline void AppendGeneratorWarning(std::vector<std::string> *warnings,
                                   const std::string &op_type,
                                   const std::string &detail) {
  if (warnings == nullptr) {
    return;
  }
  warnings->emplace_back("cpp_generator: op " + op_type + " " + detail);
}

inline bool ValidateLoweredSignature(const std::string &op_type,
                                     const LoweredSignature &lowered,
                                     std::vector<std::string> *warnings,
                                     std::string &detail) {
  std::vector<std::string> missing_dynamic_outputs;
  std::vector<std::string> missing_subgraphs;
  for (const auto &dyn_out : lowered.dynamic_outputs) {
    if (dyn_out.param == nullptr) {
      missing_dynamic_outputs.emplace_back(dyn_out.ir_name);
    }
  }
  for (const auto &subgraph : lowered.subgraphs) {
    if (subgraph.param == nullptr) {
      missing_subgraphs.emplace_back(subgraph.ir_name);
    }
  }

  if (missing_dynamic_outputs.empty() && missing_subgraphs.empty()) {
    return true;
  }

  std::stringstream ss;
  if (!missing_dynamic_outputs.empty()) {
    ss << "missing dynamic_output_num params for [" << JoinNames(missing_dynamic_outputs) << "]";
  }
  if (!missing_subgraphs.empty()) {
    if (ss.tellp() > 0) {
      ss << ", ";
    }
    ss << "missing subgraph params for [" << JoinNames(missing_subgraphs) << "]";
  }
  detail = ss.str();
  AppendGeneratorWarning(warnings, op_type, detail + ", emit deleted overload");
  return false;
}

inline std::string ParamTypeName(const ge::es::history::Param &param) {
  using ge::es::history::ParamCxxKind;
  static const std::map<ParamCxxKind, const char *> kTypeNameTable = {
    {ParamCxxKind::kEsTensorLikeRef, "const EsTensorLike &"},
    {ParamCxxKind::kTensorHolderRef, "const EsTensorHolder &"},
    {ParamCxxKind::kTensorHoldersVecRef, "const std::vector<EsTensorHolder> &"},
    {ParamCxxKind::kDataType, "ge::DataType"},
    {ParamCxxKind::kTensorUniquePtr, "std::unique_ptr<ge::Tensor>"},
    {ParamCxxKind::kListIntRef, "const std::vector<int64_t> &"},
    {ParamCxxKind::kListFloatRef, "const std::vector<float> &"},
    {ParamCxxKind::kListBoolRef, "const std::vector<uint8_t> &"},
    {ParamCxxKind::kListTypeRef, "const std::vector<ge::DataType> &"},
    {ParamCxxKind::kListListIntRef, "const std::vector<std::vector<int64_t>> &"},
    {ParamCxxKind::kListStringRef, "const std::vector<const char *> &"},
    {ParamCxxKind::kGraphUniquePtr, "std::unique_ptr<ge::Graph>"},
    {ParamCxxKind::kGraphsVec, "std::vector<std::unique_ptr<ge::Graph>>"},
    {ParamCxxKind::kGraphBuilderRef, "const EsGraphBuilder &"},
    {ParamCxxKind::kGraphBuilderPtr, "const EsGraphBuilder *"},
    {ParamCxxKind::kInt64, "int64_t"},
    {ParamCxxKind::kFloat, "float"},
    {ParamCxxKind::kBool, "bool"},
    {ParamCxxKind::kCString, "const char *"},
    {ParamCxxKind::kNullptrT, "std::nullptr_t"}
  };
  const auto iter = kTypeNameTable.find(param.kind);
  return iter == kTypeNameTable.end() ? "std::nullptr_t" : iter->second;
}
} // namespace cpp_gen
} // namespace es
} // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_UTILS_H_
