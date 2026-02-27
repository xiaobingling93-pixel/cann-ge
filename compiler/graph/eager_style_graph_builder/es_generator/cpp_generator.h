/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_H_
#include "utils.h"
#include <sstream>
#include "graph/op_desc.h"
#include "generator_interface.h"
#include "c_generator.h"
#include "graph/utils/default_attr_utils.h"

namespace ge {
namespace es {
class DefaultValueHelper {
 public:
  explicit DefaultValueHelper(const OpDescPtr &op) {
    bool has_required_attr = false;
    for (const auto &attr : GetAllIrAttrsNamesAndTypeInOrder(op)) {
      if (attr.is_required) {
        has_required_attr = true;
        attr_names_to_default_.clear();
      } else {
        attr_names_to_default_[attr.name] = true;
      }
    }
    if (has_required_attr) {
      return;
    }
    for (const auto& ir_out: op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        return ;
      }
    }
    for (const auto &ir_in : op->GetIrInputs()) {
      if (ir_in.second == kIrInputOptional) {
        input_names_to_default_[ir_in.first] = true;
      } else {
        input_names_to_default_.clear();
      }
    }
  }

  bool IsInputHasDefault(const std::string &name) const {
    return input_names_to_default_.count(name) > 0;
  }
  bool IsAttrHasDefault(const std::string &name) const {
    return attr_names_to_default_.count(name) > 0;
  }
  bool IsAnyInputHasDefault() const {
    return !input_names_to_default_.empty();
  }

 private:
  std::unordered_map<std::string, bool> input_names_to_default_;
  std::unordered_map<std::string, bool> attr_names_to_default_;
};

class CppGenerator : public ICodeGenerator {
 public:
  void GenOp(const OpDescPtr &op) override {
    // 只有一个hpp
    std::stringstream h_stream("");
    GenPerOpHHead(op, h_stream);
    GenOutputStructIfNeeded(op, h_stream);
    GenCommentsIfNeeded(op, h_stream, SupportTensorLike(op));
    GenDeclaration(op, h_stream);
    GenFuncBody(op, h_stream);
    GenPerOpHTail(h_stream);
    op_type_to_hss_[op->GetType()] = std::move(h_stream);
  }

  void ResetWhenGenFailed(const OpDescPtr &op) override {
    op_type_to_hss_.erase(op->GetType());
  }

  void GenAggregateHeader() override {
    GenAggregateHHead();
  }

  void GenAggregateIncludes(const std::vector<std::string> &sorted_op_types) override {
    for (const auto &opType : sorted_op_types) {
      hss_ << "#include \"" << PerOpHeaderFileName(opType) << "\"\n";
    }
    GenAggregateHTail();
  }

  std::string GetAggregateFileName() const override {
    return "es_" + module_name_ + "_ops.h";
  }

  std::stringstream &GetAggregateContentStream() override {
    return hss_;
  }

  std::string GetPerOpFileName(const std::string &op_type) const override {
    return PerOpHeaderFileName(op_type);
  }

  const std::unordered_map<std::string, std::stringstream> &GetPerOpContents() const override {
    return op_type_to_hss_;
  }

  void GenPerOpFiles(const std::string &output_dir, const std::vector<std::string> &op_types) override {
    // 生成hpp
    WritePerOpFiles(output_dir, op_types, GetPerOpContents(),
                    [this](const std::string &op_type) { return GetPerOpFileName(op_type); });
  }

  std::string GetGeneratorName() const override {
    return "C++ generator";
  }

  std::string GetCommentStart() const override {
    return "/**";
  }

  std::string GetCommentEnd() const override {
    return " */";
  }

  std::string GetCommentLinePrefix() const override {
    return " * ";
  }

  // 设置模块名，用于生成保护宏
  void SetModuleName(const std::string &module_name) {
    module_name_ = module_name;
  }

  // 设置保护宏前缀
  void SetHGuardPrefix(const std::string &h_guard_prefix) {
    h_guard_prefix_ = h_guard_prefix;
  }

  // 设置历史原型库分包目录
  void SetHistoryRegistry(const std::string &history_registry) {
    history_registry_ = history_registry;
  }

  // 设置当前版本号
  void SetReleaseVersion(const std::string &release_version) {
    release_version_ = release_version;
  }

  static std::string PerOpHeaderFileName(const std::string &op_type) {
    return std::string("es_") + op_type + ".h";
  }

  void GenAggregateHHead() {
    GenCopyright(hss_);
    hss_ << "#ifndef " << MakeGuardFromModule() << std::endl;
    hss_ << "#define " << MakeGuardFromModule() << std::endl;
    hss_ << std::endl;
  }
  void GenAggregateHTail() {
    hss_ << "#endif  // " << MakeGuardFromModule() << std::endl;
  }

 private:
  std::string MakeGuardFromModule() const {
    return ge::es::MakeGuardFromModule(module_name_, h_guard_prefix_, "_OPS_CPP_H");
  }

  std::string MakeGuardFromOp(const std::string &op_type) {
    return ge::es::MakeGuardFromOp(op_type, h_guard_prefix_, "_CPP_H_");
  }
  void GenPerOpHHead(const OpDescPtr &op, std::stringstream &ss) {
    GenCopyright(ss);
    std::string op_type = op->GetType();
    ss << "\n#ifndef " << MakeGuardFromOp(op_type) << "\n";
    ss << "#define " << MakeGuardFromOp(op_type) << "\n";
    ss << "#include <utility>\n";
    ss << "#include \"es_tensor_holder.h\"\n";
    ss << "#include \"es_graph_builder.h\"\n";
    if (SupportTensorLike(op)) {
      ss << "#include \"es_tensor_like.h\"\n";
      ss << "#include \"es_log.h\"\n";
      ss << "#include <iostream>\n";
    }
    ss << "#include \"" << CGenerator::PerOpHeaderFileName(op_type) << "\"\n";
    ss << R"(namespace ge {
namespace es {
)" << std::endl;
  }

  // 支持input类型为TensorLike场景: 1、IrInputs都是可选 2、IrInputs个数大于1且不都是向量
  static bool SupportTensorLike(const OpDescPtr &op) {
    if (IsOpInputsAllOptional(op->GetIrInputs())) {
      return true;
    }

    if (op->GetIrInputs().size() <= 1) {
      return false;
    }
    for (const auto &in : op->GetIrInputs()) {
      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        return true;
      }
    }
    return false;
  }

  static void GenPerOpHTail(std::stringstream &ss) {
    ss << R"(}  // namespace es
}  // namespace ge)"
       << std::endl;
    ss << "#endif\n";
  }
  static std::string CppOutputStructName(const OpDescPtr &op) {
    return op->GetType() + "Output";
  }
  static std::string GenCppOutputStructDef(const OpDescPtr &op) {
    std::stringstream ss;
    ss << "struct " << CppOutputStructName(op) << " {" << std::endl;
    for (const auto &ir_out : op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        ss << "  std::vector<EsTensorHolder> ";
      } else {
        ss << "  EsTensorHolder ";
      }
      ss << OutName(ir_out.first, op) << ";" << std::endl;
    }
    ss << "};";
    return ss.str();
  }
  static std::string GetCppDynamicOutputDef(const OpDescPtr &op) {
    if (op->GetIrOutputs().size() == 1) {
      return "inline std::vector<EsTensorHolder> ";
    } else {
      std::stringstream ss;
      ss << "inline " << CppOutputStructName(op) << " ";
      return ss.str();
    }
  }
  static std::string GetAttrDeclaration(const OpDescPtr &op, const IrAttrInfo &attr) {
    return attr.type_info.cpp_api_type + AttrName(attr.name, op);
  }
  static std::string GetInputDeclaration(const OpDescPtr &op, const DefaultValueHelper &dv) {
    std::stringstream ss;
    const std::string tensor_type_name = SupportTensorLike(op) ? "EsTensorLike" : "EsTensorHolder";
    bool first = true;
    for (const auto &in : op->GetIrInputs()) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        ss << "const " << tensor_type_name << " &" << InName(in.first);
        if (dv.IsInputHasDefault(in.first)) {
          ss << "=nullptr";
        }
      } else {
        ss << "const std::vector<EsTensorHolder> &" << InName(in.first);
      }
    }

    if (first) {  // no inputs
      ss << "const EsGraphBuilder &owner_builder";
    } else if (IsOpInputsAllOptional(op->GetIrInputs())) {
      if (dv.IsAnyInputHasDefault()) {
        ss << ", const EsGraphBuilder *owner_builder = nullptr";
      } else {
        ss << ", const EsGraphBuilder *owner_builder";
      }
    }
    return ss.str();
  }

  static std::vector<std::string> GetDynamicOutputSizeParam(const OpDescPtr &op) {
    std::vector<std::string> dynamic_output_sizes;
    for (const auto &ir_out : op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        dynamic_output_sizes.emplace_back("int64_t " + OutName(ir_out.first, op) + "_num");
      }
    }
    return dynamic_output_sizes;
  }
  static std::string GenSubgraphDeclaration(const OpDescPtr &op) {
    std::stringstream ss;
    for (const auto& ir_subgraph: op->GetOrderedSubgraphIrNames()) {
      ss << ", ";
      if (ir_subgraph.second == kStatic) {
        ss << "std::unique_ptr<ge::Graph> " << SubgraphName(ir_subgraph.first);
      } else {
        ss << "std::vector<std::unique_ptr<ge::Graph>> " << SubgraphName(ir_subgraph.first);
      }
    }
    return ss.str();
  }
  static std::string GenSubgraphPassIn(const OpDescPtr &op) {
    std::stringstream ss;
    for (const auto &ir_subgraph : op->GetOrderedSubgraphIrNames()) {
      ss << ", ";
      if (ir_subgraph.second == kStatic) {
        ss << "static_cast<EsCGraph *>(static_cast<void *>(" << SubgraphName(ir_subgraph.first) << ".release()))";
      } else {
        ss << "GeGraphsToEsCGraphs(std::move(" << SubgraphName(ir_subgraph.first) << ")).data()";
        ss << ", ";
        ss << "esb_" << SubgraphName(ir_subgraph.first);
      }
    }
    return ss.str();
  }

  static std::string GenIrAttrsPassIn(const OpDescPtr &op) {
    std::stringstream ss;
    for (const auto &attr : GetAllIrAttrsNamesAndTypeInOrder(op)) {
      ss << ", ";
      if (attr.type_info.is_list_type) {
        if (strcmp(attr.type_info.av_type, "VT_LIST_BOOL") == 0) {
          ss << "static_cast<const bool *>(static_cast<const void *>(" << AttrName(attr.name, op)
              << ".data())), static_cast<int64_t>(" << AttrName(attr.name, op) << ".size())";
        } else if (strcmp(attr.type_info.av_type, "VT_LIST_LIST_INT") == 0) {
          ss << "ListListTypeToPtrAndCounts<int64_t>(" << AttrName(attr.name, op) << ").first.data(), static_cast<int64_t>("
          << AttrName(attr.name, op) << ".size()), "
          << "ListListTypeToPtrAndCounts<int64_t>(" << AttrName(attr.name, op) << ").second.data()";
        } else if (strcmp(attr.type_info.av_type, "VT_LIST_DATA_TYPE") == 0) {
          ss << "DataTypesToEsCDataTypes(" << AttrName(attr.name, op) << ").data(), static_cast<int64_t>(" << AttrName(attr.name, op) << ".size())";
        } else if (strcmp(attr.type_info.av_type, "VT_LIST_STRING") == 0) {
          ss << "const_cast<const char **>(" << AttrName(attr.name, op) << ".data()), static_cast<int64_t>(" << AttrName(attr.name, op) << ".size())";
        } else {
          ss << AttrName(attr.name, op) << ".data(), static_cast<int64_t>(" << AttrName(attr.name, op) << ".size())";
        }
      } else if (strcmp(attr.type_info.av_type, "VT_DATA_TYPE") == 0) {
        ss << "static_cast<C_DataType>(" << AttrName(attr.name, op) << ")";
      } else if (strcmp(attr.type_info.av_type, "VT_TENSOR") == 0)  {
        ss << "static_cast<EsCTensor *>(static_cast<void *>(" + AttrName(attr.name, op) + ".release()))";
      } else {
        ss << AttrName(attr.name, op);
      }
    }
    return ss.str();
  }

  static std::string GetSubgraphArgDefIfNeed(const OpDescPtr &op) {
    std::stringstream ss;
    for (const auto &ir_subgraph : op->GetOrderedSubgraphIrNames()) {
      if (ir_subgraph.second == kDynamic) {
        ss << "  auto esb_" << SubgraphName(ir_subgraph.first) << "= static_cast<int64_t>("
           << SubgraphName(ir_subgraph.first) << ".size());" << std::endl;
      }
    }
    return ss.str();
  }

  static void GenOutputStructIfNeeded(const OpDescPtr &op, std::stringstream &hss) {
    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
      case OutputType::kOneOutput:
        break;
      case OutputType::kDynamicOutput:
        if (op->GetIrOutputs().size() > 1) {
          hss << GenCppOutputStructDef(op) << std::endl;
        }
        break;
      case OutputType::kMultiOutput:
        hss << GenCppOutputStructDef(op) << std::endl;
        break;
      default:
        throw std::runtime_error("Invalid output type");
    }
  }

  static void GenDeclaration(const OpDescPtr &op, std::stringstream &hss) {
    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
      case OutputType::kOneOutput:
        hss << "inline EsTensorHolder ";
        break;
      case OutputType::kDynamicOutput:
        hss << GetCppDynamicOutputDef(op);
        break;
      case OutputType::kMultiOutput:
        hss << "inline " << CppOutputStructName(op) << " ";
        break;
      default:
        throw std::runtime_error("Invalid output type");
    }
    hss << op->GetTypePtr() << "(";
    const auto dv = DefaultValueHelper(op);
    hss << GetInputDeclaration(op, dv);

    for (const auto& dynamic_output_size: GetDynamicOutputSizeParam(op)) {
      hss << ", " << dynamic_output_size;
    }

    hss << GenSubgraphDeclaration(op);

    for (const auto &attr : GetAllIrAttrsNamesAndTypeInOrder(op)) {
      hss << ", ";
      hss << GetAttrDeclaration(op, attr);
      if (dv.IsAttrHasDefault(attr.name)) {
        hss << "=" << GetDefaultValueString(op, attr.name, attr.type_info.av_type);
      }
    }

    hss << ")";
  }
  static void GenPassInputs(const OpDescPtr &op, std::stringstream &hss) {
    bool first = true;
    for (const auto &in : op->GetIrInputs()) {
      if (first) {
        first = false;
      } else {
        hss << ", ";
      }
      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        if (SupportTensorLike(op)) {
          hss << InName(in.first) << ".ToTensorHolder(owner_graph_builder).GetCTensorHolder()";
        } else {
          hss << InName(in.first) << ".GetCTensorHolder()";
        }
      } else {
        hss << "esb_" + InName(in.first) << ".data(), static_cast<int64_t>(esb_" + InName(in.first) << ".size())";
      }
    }
    if (first) {  // no inputs
      hss << "owner_builder.GetCGraphBuilder()";
    } else if (IsOpInputsAllOptional(op->GetIrInputs())) {  // all optional
      hss << ", owner_builder == nullptr ? nullptr : owner_builder->GetCGraphBuilder()";
    }

    for (const auto& ir_out: op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        hss << ", ";
        hss << OutName(ir_out.first, op) << "_num";
      }
    }

    hss << GenSubgraphPassIn(op);
    hss << GenIrAttrsPassIn(op);
  }

  static bool GenDynamicOutputReturnIfNeeded(const OpDescPtr &op, std::stringstream &hss) {
    std::stringstream dyn_ret_ss;
    dyn_ret_ss << "  return {";
    bool first = true;
    for (const auto &ir_out : op->GetIrOutputs()) {
      if (first) {
        first = false;
      } else {
        dyn_ret_ss << ", ";
      }
      if (ir_out.second == kIrOutputDynamic) { // output is a dynamic output
        hss << "  std::vector<EsTensorHolder> " << OutName(ir_out.first, op) << "_dynamic_outs;" << std::endl;
        hss << "  " << OutName(ir_out.first, op) << "_dynamic_outs" << ".reserve(out." << OutName(ir_out.first, op) <<"_num);" << std::endl;
        hss << "  for (int64_t dyn_idx = 0; dyn_idx < out." << OutName(ir_out.first, op) << "_num; ++dyn_idx) {" << std::endl;
        hss << "    " << OutName(ir_out.first, op) << "_dynamic_outs" << ".emplace_back(out."<< OutName(ir_out.first, op) << "[dyn_idx]);" << std::endl;
        hss << "  }" << std::endl;
        dyn_ret_ss << OutName(ir_out.first, op) << "_dynamic_outs";
      } else { // output is not a dynamic output
        dyn_ret_ss<< "out." << OutName(ir_out.first, op);
      }
    }
    dyn_ret_ss << "};" << std::endl;
    hss << dyn_ret_ss.str();

    return true;
  }

  static void GenFuncBody(const OpDescPtr &op, std::stringstream &hss) {
    hss << " {" << std::endl;
    GenResolveBuilderForTensorLike(op, hss);
    for (const auto &in : op->GetIrInputs()) {
      if (in.second == kIrInputDynamic) {
        hss << "  auto esb_" << InName(in.first) << " = TensorsToEsCTensorHolders(" << InName(in.first) << ");"
            << std::endl;
      }
    }
    hss << GetSubgraphArgDefIfNeed(op);
    hss << "  auto out = " << CGenerator::FuncName(op->GetType()) << "(";
    GenPassInputs(op, hss);
    hss << ");" << std::endl;

    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
      case OutputType::kOneOutput:
        hss << "  return out;" << std::endl;
        break;
      case OutputType::kDynamicOutput:
        GenDynamicOutputReturnIfNeeded(op, hss);
        break;
      case OutputType::kMultiOutput: {
        hss << "  return {";
        bool first = true;
        for (const auto &ir_out : op->GetIrOutputs()) {
          if (first) {
            first = false;
          } else {
            hss << ", ";
          }
          hss << "out." << OutName(ir_out.first, op);
        }
        hss << "};" << std::endl;
        break;
      }
      default:
        throw std::runtime_error("Invalid output type");
    }

    hss << "}" << std::endl;
  }

  static void GenResolveBuilderForTensorLike(const OpDescPtr &op, std::stringstream &hss) {
    if (!SupportTensorLike(op)) {
      return;
    }

    hss << "  auto *owner_graph_builder = ResolveBuilder(";
    bool first = true;
    for (const auto &in : op->GetIrInputs()) {
      if (first) {
        first = false;
      } else {
        hss << ", ";
      }
      hss << InName(in.first);
    }
    if (IsOpInputsAllOptional(op->GetIrInputs())) {
      hss << ", owner_builder";
    }
    hss << ");" << std::endl;
    hss << "  ES_ASSERT_NOTNULL(owner_graph_builder, "
           "\"Failed to resolve owner builder: please ensure at least one input tensor "
           "or an explicit owner_builder is provided when supported.\");" << std::endl;
  }
 private:
  std::stringstream hss_;
  std::unordered_map<std::string, std::stringstream> op_type_to_hss_;
  std::string module_name_{};
  std::string h_guard_prefix_{};
  std::string history_registry_{};
  std::string release_version_{};
};
}  // namespace es
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_H_
