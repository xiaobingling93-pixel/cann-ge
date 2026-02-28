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
#include "cpp_generator_utils.h"
#include "history/attr_type_traits.h"
#include "history/history_registry_interface.h"
#include "history/ir_proto_codec.h"
#include "history/overload_planner.h"

namespace ge {
namespace es {
class CppGenerator : public ICodeGenerator {
 public:
  void GenOp(const OpDescPtr &op) override {
    std::vector<std::string> warnings;
    const auto current = ge::es::history::IrProtoCodec::FromOpDesc(op);
    ge::es::history::HistoryContext history;
    if (!history_registry_.empty()) {
      if (!history_window_error_.empty()) {
        warnings.push_back("op " + op->GetType() + " skip load history : " + history_window_error_);
      } else {
        history = es::history::LoadHistoryChain(history_registry_,
                                                history_window_versions_,
                                                op->GetType(),
                                                warnings);
      }
    }
    const auto plan = ge::es::history::PlanCppOverloads(current, history, warnings);
    GenOpWithPlan(op, plan, warnings);
  }

  void GenOpWithPlan(const OpDescPtr &op, const ge::es::history::OverloadPlan &plan,
                     std::vector<std::string> &warnings) {
    std::stringstream h_stream("");
    const bool has_tensor_like = HasTensorLikeSignature(plan.signatures);
    GenPerOpHHead(op, h_stream, has_tensor_like);
    GenOutputStructIfNeeded(op, h_stream);
    GenCommentsIfNeeded(op, h_stream, has_tensor_like);
    std::stringstream sig_stream("");
    for (const auto &sig : plan.signatures) {
      GenSignature(op, sig, warnings, sig_stream);
    }
    GenWarningsIfNeeded(warnings, h_stream);
    h_stream << sig_stream.str();
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
    // 依赖当前 CreateGenerators 的调用顺序：先 SetHistoryRegistry，再 SetReleaseVersion。
    // 在该顺序下仅此处刷新一次历史窗口即可；若后续调整调用顺序，需要同步调整刷新时机。
    RefreshHistoryWindowVersions();
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
  void RefreshHistoryWindowVersions() {
    history_window_versions_.clear();
    history_window_error_.clear();
    if (history_registry_.empty()) {
      return;
    }
    (void) ge::es::history::LoadHistoryWindowVersions(history_registry_,
                                                      release_version_,
                                                      history_window_versions_,
                                                      history_window_error_);
  }

  std::string MakeGuardFromModule() const {
    return ge::es::MakeGuardFromModule(module_name_, h_guard_prefix_, "_OPS_CPP_H");
  }

  std::string MakeGuardFromOp(const std::string &op_type) {
    return ge::es::MakeGuardFromOp(op_type, h_guard_prefix_, "_CPP_H_");
  }
  void GenPerOpHHead(const OpDescPtr &op, std::stringstream &ss, bool support_tensor_like) {
    GenCopyright(ss);
    std::string op_type = op->GetType();
    ss << "\n#ifndef " << MakeGuardFromOp(op_type) << "\n";
    ss << "#define " << MakeGuardFromOp(op_type) << "\n";
    ss << "#include <utility>\n";
    ss << "#include \"es_tensor_holder.h\"\n";
    ss << "#include \"es_graph_builder.h\"\n";
    if (support_tensor_like) {
      ss << "#include \"es_tensor_like.h\"\n";
      ss << "#include \"es_log.h\"\n";
      ss << "#include <iostream>\n";
    }
    ss << "#include \"" << CGenerator::PerOpHeaderFileName(op_type) << "\"\n";
    ss << R"(namespace ge {
namespace es {
)" << std::endl;
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

  static std::string GenInputPassIn(const cpp_gen::LoweredSignature &lowered,
                                    const ge::es::history::Signature &sig,
                                    bool &first) {
    std::stringstream hss;
    for (const auto &input: lowered.inputs) {
      const auto *input_param = input.param;
      const bool has_input_param = input_param != nullptr;
      const std::string input_name = has_input_param ? input_param->name : InName(input.ir_name);
      if (input.type == kIrInputRequired || input.type == kIrInputOptional) {
        if (!has_input_param) {
          cpp_gen::AppendCallArg(hss, first, "nullptr");
          continue;
        }
        if (cpp_gen::UseTensorLikeInSignature(sig, input.ir_name)) {
          cpp_gen::AppendCallArg(hss, first, input_name + ".ToTensorHolder(owner_graph_builder).GetCTensorHolder()");
        } else {
          cpp_gen::AppendCallArg(hss, first, input_name + ".GetCTensorHolder()");
        }
      } else {
        if (!has_input_param) {
          cpp_gen::AppendCallArg(hss, first, "nullptr");
          cpp_gen::AppendCallArg(hss, first, "0");
        } else {
          cpp_gen::AppendCallArg(hss, first, "esb_" + input_name + ".data()");
          cpp_gen::AppendCallArg(hss, first, "static_cast<int64_t>(esb_" + input_name + ".size())");
        }
      }
    }
    return hss.str();
  }

  static std::string GenOwnerBuilderPassIn(const ge::es::history::Signature &sig,
                                           const cpp_gen::LoweredSignature &lowered,
                                           bool &first) {
    std::stringstream ss;
    if (const auto *owner_param = lowered.owner_builder; owner_param != nullptr) {
      const std::string owner_name = owner_param->name;
      if (owner_param->kind == ge::es::history::ParamCxxKind::kGraphBuilderRef) {
        cpp_gen::AppendCallArg(ss, first, owner_name + ".GetCGraphBuilder()");
      } else {
        cpp_gen::AppendCallArg(ss, first, owner_name + " == nullptr ? nullptr : " + owner_name + "->GetCGraphBuilder()");
      }
      return ss.str();
    }
    if (!cpp_gen::NeedOwnerBuilderCArg(lowered)) {
      return ss.str();
    }
    if (cpp_gen::HasTensorLikeParam(sig)) {
      cpp_gen::AppendCallArg(ss,
                             first,
                             "owner_graph_builder == nullptr ? nullptr : owner_graph_builder->GetCGraphBuilder()");
    } else {
      cpp_gen::AppendCallArg(ss, first, "nullptr");
    }
    return ss.str();
  }

  static std::string GenDynamicOutputPassIn(const cpp_gen::LoweredSignature &lowered) {
    std::stringstream ss;
    for (const auto &dyn_out: lowered.dynamic_outputs) {
      ss << ", ";
      ss << dyn_out.param->name;
    }
    return ss.str();
  }

  static std::string GenSubgraphPassIn(const cpp_gen::LoweredSignature &lowered) {
    std::stringstream ss;
    for (const auto &subgraph: lowered.subgraphs) {
      ss << ", ";
      const std::string subgraph_name = subgraph.param->name;
      if (subgraph.type == kStatic) {
        ss << "static_cast<EsCGraph *>(static_cast<void *>(" << subgraph_name << ".release()))";
      } else {
        ss << "GeGraphsToEsCGraphs(std::move(" << subgraph_name << ")).data()";
        ss << ", ";
        ss << "esb_" << subgraph_name;
      }
    }
    return ss.str();
  }

  static ge::es::history::ParamCxxKind ResolveAttrKind(const IrAttrInfo &attr,
                                                       const ge::es::history::Signature &sig) {
    ge::es::history::ParamCxxKind attr_kind = ge::es::history::ParamCxxKind::kNullptrT;
    const auto *attr_param =
        cpp_gen::FindParamByRoleAndIrName(sig, ge::es::history::ParamRole::kAttr, attr.name);
    if (attr_param != nullptr) {
      return attr_param->kind;
    }
    (void)ge::es::history::AttrTypeTraits::TryGetParamKindByIrTypeInfo(attr.type_info.av_type,
                                                                        attr.type_info.is_list_type,
                                                                        attr_kind);
    return attr_kind;
  }

  static std::string BuildAttrPassExpr(const std::string &attr_name,
                                       ge::es::history::AttrPassStrategy strategy) {
    std::stringstream ss;
    switch (strategy) {
      case ge::es::history::AttrPassStrategy::kListBoolDataAndSize:
        ss << "static_cast<const bool *>(static_cast<const void *>(" << attr_name
           << ".data())), static_cast<int64_t>(" << attr_name << ".size())";
        break;
      case ge::es::history::AttrPassStrategy::kListListIntDataSizeCounts:
        ss << "ListListTypeToPtrAndCounts<int64_t>(" << attr_name << ").first.data(), static_cast<int64_t>("
           << attr_name << ".size()), "
           << "ListListTypeToPtrAndCounts<int64_t>(" << attr_name << ").second.data()";
        break;
      case ge::es::history::AttrPassStrategy::kListTypeDataAndSize:
        ss << "DataTypesToEsCDataTypes(" << attr_name << ").data(), static_cast<int64_t>("
           << attr_name << ".size())";
        break;
      case ge::es::history::AttrPassStrategy::kListStringDataAndSize:
        ss << "const_cast<const char **>(" << attr_name << ".data()), static_cast<int64_t>("
           << attr_name << ".size())";
        break;
      case ge::es::history::AttrPassStrategy::kListDataAndSize:
        ss << attr_name << ".data(), static_cast<int64_t>(" << attr_name << ".size())";
        break;
      case ge::es::history::AttrPassStrategy::kDataTypeCast:
        ss << "static_cast<C_DataType>(" << attr_name << ")";
        break;
      case ge::es::history::AttrPassStrategy::kTensorRelease:
        ss << "static_cast<EsCTensor *>(static_cast<void *>(" + attr_name + ".release()))";
        break;
      case ge::es::history::AttrPassStrategy::kDirect:
      default:
        ss << attr_name;
        break;
    }
    return ss.str();
  }

  static std::string GenIrAttrsPassIn(const OpDescPtr &op, const ge::es::history::Signature &sig) {
    std::stringstream ss;
    for (const auto &attr: GetAllIrAttrsNamesAndTypeInOrder(op)) {
      const auto *attr_param =
          cpp_gen::FindParamByRoleAndIrName(sig, ge::es::history::ParamRole::kAttr, attr.name);
      const std::string attr_name = attr_param == nullptr ? AttrName(attr.name, op) : attr_param->name;
      const auto attr_kind = ResolveAttrKind(attr, sig);
      auto strategy = ge::es::history::AttrTypeTraits::GetAttrPassStrategy(attr_kind);
      if (strategy == ge::es::history::AttrPassStrategy::kDirect && attr.type_info.is_list_type) {
        strategy = ge::es::history::AttrPassStrategy::kListDataAndSize;
      }
      ss << ", ";
      ss << BuildAttrPassExpr(attr_name, strategy);
    }
    return ss.str();
  }

  static void AppendSubgraphArgDefs(const cpp_gen::LoweredSignature &lowered, std::stringstream &hss) {
    for (const auto &subgraph: lowered.subgraphs) {
      if (subgraph.type != kDynamic) {
        continue;
      }
      if (subgraph.param == nullptr) {
        continue;
      }
      const std::string subgraph_name = subgraph.param->name;
      hss << "  auto esb_" << subgraph_name << "= static_cast<int64_t>("
          << subgraph_name << ".size());" << std::endl;
    }
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

  static void GenSignature(const OpDescPtr &op, const ge::es::history::Signature &sig,
                           std::vector<std::string> &warnings, std::stringstream &hss) {
    if (sig.is_deleted) {
      GenGuardSignature(op, sig, hss);
      return;
    }
    const auto lowered = cpp_gen::LowerSignature(op, sig);
    std::string invalid_detail;
    if (!cpp_gen::ValidateLoweredSignature(op->GetType(), lowered, &warnings, invalid_detail)) {
      auto blocked_sig = sig;
      blocked_sig.is_deleted = true;
      blocked_sig.is_deprecated = true;
      blocked_sig.deprecate_msg = "Invalid overload signature blocked: " + invalid_detail;
      GenGuardSignature(op, blocked_sig, hss);
      return;
    }
    GenReturnType(op, hss);
    GenFuncDeclareBySig(op->GetTypePtr(), sig, hss);
    GenFuncBody(op, sig, lowered, hss);
  }

  static void GenPassInputs(const OpDescPtr &op,
                            const ge::es::history::Signature &sig,
                            const cpp_gen::LoweredSignature &lowered,
                            std::stringstream &hss) {
    bool first = true;
    hss << GenInputPassIn(lowered, sig, first);
    hss << GenOwnerBuilderPassIn(sig, lowered, first);
    hss << GenDynamicOutputPassIn(lowered);
    hss << GenSubgraphPassIn(lowered);
    hss << GenIrAttrsPassIn(op, sig);
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

  static void GenWarningsIfNeeded(const std::vector<std::string> &warnings, std::stringstream &hss) {
    if (warnings.empty()) {
      return;
    }
    hss << "/**" << std::endl;
    for (const auto &warning : warnings) {
      hss << " * @warning " << warning << std::endl;
    }
    hss << " */" << std::endl;
  }

  static void GenOutputReturn(const OpDescPtr &op, std::stringstream &hss) {
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
  }
  static void GenDynamicInputs(const cpp_gen::LoweredSignature &lowered, std::stringstream &hss) {
    for (const auto &input : lowered.inputs) {
      if (input.type != kIrInputDynamic) {
        continue;
      }
      if (input.param == nullptr) {
        continue;
      }
      const std::string input_name = input.param->name;
      hss << "  auto esb_" << input_name << " = TensorsToEsCTensorHolders(" << input_name << ");"
          << std::endl;
    }
  }
  static void GenFuncBody(const OpDescPtr &op,
                          const ge::es::history::Signature &sig,
                          const cpp_gen::LoweredSignature &lowered,
                          std::stringstream &hss) {
    hss << " {" << std::endl;
    GenResolveBuilderForTensorLike(sig, hss);
    GenDynamicInputs(lowered, hss);
    AppendSubgraphArgDefs(lowered, hss);
    hss << "  auto out = " << CGenerator::FuncName(op->GetType()) << "(";
    GenPassInputs(op, sig, lowered, hss);
    hss << ");" << std::endl;
    GenOutputReturn(op, hss);
    hss << "}" << std::endl;
  }

  static std::string SignatureParamsToString(const ge::es::history::Signature &sig) {
    std::stringstream ss;
    bool first = true;
    for (const auto &param: sig.params) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      ss << cpp_gen::ParamTypeName(param) << " " << param.name;
      if (param.has_default && !param.default_expr.empty()) {
        ss << param.default_expr;
      }
    }
    return ss.str();
  }

  static void GenReturnType(const OpDescPtr &op, std::stringstream &hss) {
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
  }

  static void GenFuncDeclareBySig(const std::string &func_name,
                                  const ge::es::history::Signature &sig,
                                  std::stringstream &hss) {
    hss << func_name << "(" << SignatureParamsToString(sig) << ")";
  }

  static void GenGuardSignature(const OpDescPtr &op,
                                const ge::es::history::Signature &sig,
                                std::stringstream &hss) {
    if (sig.is_deprecated) {
      if (!sig.deprecate_msg.empty()) {
        hss << "[[deprecated(\"" << sig.deprecate_msg << "\")]] ";
      } else {
        hss << "[[deprecated]] ";
      }
    }
    GenReturnType(op, hss);
    GenFuncDeclareBySig(op->GetTypePtr(), sig, hss);
    hss << " = delete;" << std::endl;
  }

  static void GenResolveBuilderForTensorLike(const ge::es::history::Signature &sig,
                                             std::stringstream &hss) {
    if (!cpp_gen::HasTensorLikeParam(sig)) {
      return;
    }
    hss << "  auto *owner_graph_builder = ResolveBuilder(";
    bool first = true;
    for (const auto &param : sig.params) {
      if (param.role != ge::es::history::ParamRole::kInput) {
        continue;
      }
      if (first) {
        first = false;
      } else {
        hss << ", ";
      }
      hss << param.name;
    }
    if (const auto *owner_builder_ptr = cpp_gen::FindOwnerBuilderParam(sig); owner_builder_ptr != nullptr &&
        owner_builder_ptr->kind == ge::es::history::ParamCxxKind::kGraphBuilderPtr) {
      hss << ", " << owner_builder_ptr->name;
    }
    hss << ");" << std::endl;
    hss << "  ES_ASSERT_NOTNULL(owner_graph_builder, "
           "\"Failed to resolve owner builder: please ensure at least one input tensor "
           "or an explicit owner_builder is provided when supported.\");" << std::endl;
  }
  static bool HasTensorLikeSignature(const std::vector<ge::es::history::Signature> &sigs) {
    for (const auto &sig : sigs) {
      if (cpp_gen::HasTensorLikeParam(sig)) {
        return true;
      }
    }
    return false;
  }
 private:
  std::stringstream hss_;
  std::unordered_map<std::string, std::stringstream> op_type_to_hss_;
  std::string module_name_{};
  std::string h_guard_prefix_{};
  std::string history_registry_{};
  std::string release_version_{};
  // 历史窗口版本是与 op_type 无关的公共上下文：按 release_version 只需计算一次并复用。
  std::vector<ge::es::history::VersionMeta> history_window_versions_;
  std::string history_window_error_;
};
}  // namespace es
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_CPP_GENERATOR_H_
