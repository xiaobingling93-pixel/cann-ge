/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_tiling.h"
#include <nlohmann/json.hpp>
#include "common/checker.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "aicore/converter/autofuse_node_converter.h"
#include "exe_graph/runtime/tiling_context.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/args_format_desc.h"
#include "graph/utils/graph_utils.h"
// compatible tiling need
#include "register/op_tiling_registry.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "graph_builder/bg_compatible_utils.h"
#include "common/omg_util/omg_util.h"
#include "common/ge_inner_attrs.h"
#include "bg_platform.h"
#include "runtime/subscriber/global_tracer.h"
#include "exe_graph/lowering/frame_selector.h"
#include "value_holder_generator.h"
#include "bg_model_desc.h"
#include "graph/utils/op_type_utils.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"
#include "graph/ge_context.h"
#include "common/op_tiling/tiling_dfx.h"
#include "runtime/rt_model.h"
#include "runtime/dev.h"
#include "adump_pub.h"
#include "mmpa/mmpa_api.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
namespace bg {
namespace {
struct TilingAppendDfxInfoInputs {
  const std::vector<ValueHolderPtr> &input_shapes;
  const std::vector<ValueHolderPtr> &output_shapes;
  ValueHolderPtr launch_arg;
  ValueHolderPtr workspace_sizes;
  LoweringGlobalData &global_data;
};
// 若算子注册了老tiling, 没注册新tiling话，走兼容模式
// autotiling自动走新模式
bool NeedTilingCompatible(const ge::NodePtr &node, const gert::OpImplSpaceRegistryV2Ptr &space_registry_ptr) {
  if (space_registry_ptr == nullptr) {
    return true;
  }
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  auto op_funcs = space_registry_ptr->GetOpImpl(node->GetType().c_str());
  if ((op_funcs == nullptr) || (op_funcs->tiling_parse == nullptr) || (op_funcs->tiling == nullptr)) {
    auto &op_func_map = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo();
    auto iter = op_func_map.find(node->GetType());
    if (iter != op_func_map.end()) {
      return true;
    }
  }
  return false;
}

bg::ValueHolderPtr BuildCompileInfo(const ge::NodePtr &node, const std::string &json_key,
                                    const ValueHolderPtr &platform_info,
                                    LoweringGlobalData &global_data) {
  GE_ASSERT_NOTNULL(node);
  const std::string *op_compile_info_json = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), json_key);
  if (op_compile_info_json == nullptr) {
    GELOGE(ge::FAILED, "Op[%s] does not have attr[%s].", node->GetName().c_str(), json_key.c_str());
    return nullptr;
  }
  const auto &json_holder =
      bg::ValueHolder::CreateConst(op_compile_info_json->c_str(), op_compile_info_json->size() + 1, true);
  const auto &node_type = bg::ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  auto opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
  const auto space_registry_addr = global_data.GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(opp_impl_version));
  GE_ASSERT_NOTNULL(space_registry_addr);
  const auto &space_registry = bg::ValueHolder::CreateConst(&space_registry_addr, sizeof(void *), false);
  return bg::ValueHolder::CreateSingleDataOutput("TilingParse",
                                                 {json_holder, platform_info, node_type, space_registry});
}

ge::Status AssembleCompileInfoJson(const ge::OpDesc *const op_desc, std::string &op_compile_info_json) {
  nlohmann::json compile_info_json;
  try {
    compile_info_json = nlohmann::json::parse(op_compile_info_json);
  } catch (nlohmann::json::parse_error &ex) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to set compile_info_value to json of op[%s]. op_compile_info_json:%s",
                      op_desc->GetName().c_str(), op_compile_info_json.c_str());
    GELOGE(ge::FAILED, "Failed to set compile_info_value to json of op[%s]. op_compile_info_json:%s",
           op_desc->GetName().c_str(), op_compile_info_json.c_str());
    return ge::FAILED;
  }
  std::vector<int64_t> atomic_workspace_indices;
  (void)ge::AttrUtils::GetListInt(op_desc, "WorkspaceIndexes", atomic_workspace_indices);
  compile_info_json["_workspace_index_list"] = atomic_workspace_indices;
  op_compile_info_json = compile_info_json.dump();
  return ge::SUCCESS;
}

bg::ValueHolderPtr BuildAtomCompileInfo(const ge::NodePtr &node, const std::string &json_key,
                                        LoweringGlobalData &global_data) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const std::string *json_ptr = ge::AttrUtils::GetStr(op_desc, json_key);
  if (json_ptr == nullptr) {
    GELOGE(ge::FAILED, "Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), json_key.c_str());
    return nullptr;
  }
  std::string op_compile_info_json = *json_ptr;
  if (AssembleCompileInfoJson(op_desc, op_compile_info_json) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to assemble compile info json for op[%s, %s].", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return nullptr;
  }
  const auto &json_holder =
      bg::ValueHolder::CreateConst(op_compile_info_json.c_str(), op_compile_info_json.size() + 1, true);
  std::string node_type;
  if (node->GetType() == "MemSet") {
    node_type= "MemSet";
  } else {
    node_type= "DynamicAtomicAddrClean";
  }
  auto opp_impl_version = op_desc->GetOppImplVersion();
  const auto &node_type_holder = bg::ValueHolder::CreateConst(node_type.c_str(), node_type.size() + 1, true);
  const auto space_registry_addr = global_data.GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(opp_impl_version));
  GE_ASSERT_NOTNULL(space_registry_addr);
  const auto &space_registry = bg::ValueHolder::CreateConst(&space_registry_addr, sizeof(void *), false);
  const auto assembled_platform_info_holders = bg::AppendCoreTypeToPlatform(node, &global_data);
  GE_ASSERT_TRUE(assembled_platform_info_holders.size() == static_cast<size_t>(bg::AssemblePlatformInfoIndex::kNums));
  const auto platform_info =
      assembled_platform_info_holders[static_cast<size_t>(bg::AssemblePlatformInfoIndex::kPlatformInfo)];
  return bg::ValueHolder::CreateSingleDataOutput("TilingParse",
                                                 {json_holder, platform_info, node_type_holder, space_registry});
}

bg::ValueHolderPtr ParseCompileInfo(const ge::NodePtr &node, const ValueHolderPtr &platform_info,
                                    LoweringGlobalData &global_data) {
  return BuildCompileInfo(node, optiling::COMPILE_INFO_JSON, platform_info, global_data);
}
bg::ValueHolderPtr ParseAtomicCompileInfo(const ge::NodePtr &node, LoweringGlobalData &global_data) {
  return BuildAtomCompileInfo(node, optiling::ATOMIC_COMPILE_INFO_JSON, global_data);
}
ge::Status CheckArgsInfoType(const ArgsInfo::ArgsInfoType &args_info_type) {
  return ((args_info_type < ArgsInfo::ArgsInfoType::TYPE_END) && (args_info_type >= ArgsInfo::ArgsInfoType::INPUT))
             ? ge::SUCCESS
             : ge::PARAM_INVALID;
}
ge::Status CheckArgsInfoFormat(const ArgsInfo::ArgsInfoFormat &args_info_fmt) {
  return ((args_info_fmt < ArgsInfo::ArgsInfoFormat::FORMAT_END) &&
          (args_info_fmt >= ArgsInfo::ArgsInfoFormat::DIRECT_ADDR))
             ? ge::SUCCESS
             : ge::PARAM_INVALID;
}

ValueHolderPtr GetOrCreateTilingFunc(const std::string &type_str, ge::OppImplVersion opp_impl_version,
                                     LoweringGlobalData &global_data) {
  auto builder = [&type_str, &opp_impl_version, &global_data]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot(
        [&type_str, &opp_impl_version, &global_data]() -> std::vector<bg::ValueHolderPtr> {
          auto node_type = ValueHolder::CreateConst(type_str.c_str(), type_str.size() + 1, true);
          auto space_registry = bg::HolderOnInit(bg::GetSpaceRegistry(global_data, opp_impl_version));
          return {ValueHolder::CreateSingleDataOutput("FindTilingFunc", {node_type, space_registry})};
        });
  };
  auto tiling_func = global_data.GetOrCreateUniqueValueHolder(
      type_str + "_FindTilingFunc_" + std::to_string(static_cast<int64_t>(opp_impl_version)), builder)[0];
  return tiling_func;
}

bool NeedSymbolTiling(const ge::NodePtr &node) {
  return ge::OpTypeUtils::IsAutofuseNode(node->GetOpDesc());
}

ValueHolderPtr GetTilingFunc(const ge::NodePtr &node, const TilingLowerInput &lower_input) {
  ValueHolderPtr tiling_func;
  if (NeedSymbolTiling(node)) {
    tiling_func = AutofuseNodeConveter::GetAutofuseHandle(
        lower_input.global_data, node, GetAutofuseFuncsOutput::kTilingFunc);
  } else {
    std::string type;
    GE_ASSERT_SUCCESS(ge::GetOriginalType(node, type), "Failed to get original type from %s(%s).",
                      node->GetName().c_str(), node->GetType().c_str());
    ge::OppImplVersion opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
    tiling_func = GetOrCreateTilingFunc(type, opp_impl_version, lower_input.global_data);
  }
  return tiling_func;
}

ge::Status BuildTilingFwkDataInputs(const ge::NodePtr &node, const TilingLowerInput &lower_input,
                                    std::vector<ValueHolderPtr> &tiling_input) {
  const auto tiling_func = GetTilingFunc(node, lower_input);
  GE_ASSERT_NOTNULL(tiling_func);
  tiling_input.emplace_back(
      bg::ValueHolder::CreateSingleDataOutput("PrepareTilingFwkData", {tiling_func, lower_input.launch_arg}));
  return ge::SUCCESS;
}

ge::Status BuildCacheableTilingFwkDataInputs(
    const ge::NodePtr &node, const TilingLowerInput &lower_input,
    const size_t &data_dependency, const std::string &build_tiling_cache_key_func_name,
    std::vector<bg::ValueHolderPtr> &tiling_input) {
  const auto tiling_func = GetTilingFunc(node, lower_input);
  GE_ASSERT_NOTNULL(tiling_func);
  const auto data_dependency_holder = bg::ValueHolder::CreateConst(&data_dependency, sizeof(data_dependency));
  const auto build_tiling_cache_key_func_name_holder = bg::ValueHolder::CreateConst(
      build_tiling_cache_key_func_name.c_str(), build_tiling_cache_key_func_name.size() + 1, true);
  tiling_input.emplace_back(bg::ValueHolder::CreateSingleDataOutput(
      "PrepareCacheableTilingFwkData",
      {tiling_func, lower_input.launch_arg, data_dependency_holder, build_tiling_cache_key_func_name_holder}));
  return ge::SUCCESS;
}

ge::Status BuildTilingDeterministicInput(const ge::NodePtr &node, LoweringGlobalData &global_data,
                                         std::vector<ValueHolderPtr> &tiling_input) {
  auto builder = [&node]() -> std::vector<ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&node]() -> std::vector<ValueHolderPtr> {
      const auto compute_graph = node->GetOwnerComputeGraph();
      GE_ASSERT_NOTNULL(compute_graph);
      const auto root_compute_graph = ge::GraphUtils::FindRootGraph(compute_graph);
      int32_t deterministic = 0;
      (void)ge::AttrUtils::GetInt(root_compute_graph, ge::DETERMINISTIC, deterministic);
      GELOGI("Get DETERMINISTIC: %d", deterministic);
      auto deterministic_holder = bg::HolderOnInit(bg::ValueHolder::CreateConst(&deterministic, sizeof(int32_t)));

      int32_t deterministic_level = 0;
      (void)ge::AttrUtils::GetInt(root_compute_graph, "ge.deterministicLevel", deterministic_level);
      GELOGI("Get DETERMINISTIC LEVEL: %d", deterministic_level);
      auto deterministic_level_holder =
          bg::HolderOnInit(bg::ValueHolder::CreateConst(&deterministic_level, sizeof(int32_t)));
      return {deterministic_holder, deterministic_level_holder};
    });
  };
  auto deterministic_vec = global_data.GetOrCreateUniqueValueHolder("Deterministic", builder);
  GE_ASSERT_TRUE(deterministic_vec.size() == 2UL);
  tiling_input.insert(tiling_input.end(), deterministic_vec.begin(), deterministic_vec.end());
  return ge::SUCCESS;
}

bg::ValueHolderPtr BuildSymbolTilingCacheKey(
    const std::vector<bg::ValueHolderPtr> &input_shapes, const ge::NodePtr &node,
    const TilingLowerInput &lower_inputs) {
  std::vector<bg::ValueHolderPtr> inputs_holders;
  // input_data_size + input_shapes
  auto inputs_with_size = AutofuseNodeConveter::GetSymbolInputsWithSize(
      lower_inputs.global_data, input_shapes, node->GetOwnerComputeGraph()->GetName());
  inputs_holders.insert(inputs_holders.end(), inputs_with_size.begin(), inputs_with_size.end());
  // get_symbol_tiling_cache_key_func_handle
  auto get_symbol_tiling_cache_key_func_handle = AutofuseNodeConveter::GetAutofuseHandle(
      lower_inputs.global_data, node, GetAutofuseFuncsOutput::kGetTilingCacheKey);
  GE_ASSERT_NOTNULL(get_symbol_tiling_cache_key_func_handle);
  inputs_holders.emplace_back(get_symbol_tiling_cache_key_func_handle);
  // all_sym_num_handle, 同一个图的all_sym_num相同，可以用一个节点
  auto all_sym_num_handle = AutofuseNodeConveter::GetAllSymbolNumHolder(lower_inputs.global_data, node);
  GE_ASSERT_NOTNULL(all_sym_num_handle);
  inputs_holders.emplace_back(all_sym_num_handle);
  return bg::ValueHolder::CreateSingleDataOutput("GetSymbolTilingCacheKey", inputs_holders);
}

bg::ValueHolderPtr BuildSymbolTilingParse(const ge::NodePtr &node, const TilingLowerInput &lower_inputs) {
  auto tiling_parse_func_holder = AutofuseNodeConveter::GetAutofuseHandle(
      lower_inputs.global_data, node, GetAutofuseFuncsOutput::kTilingParse);
  GE_ASSERT_NOTNULL(tiling_parse_func_holder);
  return bg::ValueHolder::CreateSingleDataOutput("SymbolTilingParse",
                                                 {lower_inputs.platform_info, tiling_parse_func_holder});
}

std::vector<bg::ValueHolderPtr> BuildCommonSymbolTilingInputs(
    const ge::NodePtr &node, const TilingLowerInput &lower_inputs,
    const std::vector<bg::ValueHolderPtr> &input_shapes) {
  std::vector<bg::ValueHolderPtr> inputs_holders;
  // // input_data_size + input_shapes
  auto inputs_with_size = AutofuseNodeConveter::GetSymbolInputsWithSize(
      lower_inputs.global_data, input_shapes, node->GetOwnerComputeGraph()->GetName());
  inputs_holders.insert(inputs_holders.end(), inputs_with_size.begin(), inputs_with_size.end());
  // tiling parse
  auto tiling_parse_holder = BuildSymbolTilingParse(node, lower_inputs);
  GE_ASSERT_NOTNULL(tiling_parse_holder);
  inputs_holders.emplace_back(tiling_parse_holder);

  return inputs_holders;
}

}  // namespace
ge::Status CheckArgsInfo(const std::vector<ArgsInfo> &args_infos, const ge::NodePtr &compute_node) {
  const auto node_input_num = compute_node->GetInDataNodesAndAnchors().size();
  const auto node_output_num = static_cast<size_t>(compute_node->GetAllOutDataAnchorsSize());

  for (size_t idx = 0U; idx < args_infos.size(); ++idx) {
    GE_ASSERT_SUCCESS(CheckArgsInfoType(args_infos[idx].arg_type));
    GE_ASSERT_SUCCESS(CheckArgsInfoFormat(args_infos[idx].arg_format));
    GE_ASSERT_TRUE(args_infos[idx].start_index >= -1,
                   "[PARAM][INVALID] Args info start index[%d], expect >= -1. Please check args info.",
                   args_infos[idx].start_index);

    if (args_infos[idx].start_index == -1 || args_infos[idx].arg_size == 0U) {
      GE_ASSERT_TRUE((args_infos[idx].start_index == -1 && args_infos[idx].arg_size == 0U),
                     "[PARAM][INVALID]Args info index[%zu], start index doesn't match arg size. "
                     "Start index[%d], expect -1, arg size[%u], expect 0.",
                     idx, args_infos[idx].start_index, args_infos[idx].arg_size);
      continue;
    }
    if (args_infos[idx].start_index == 0xFFFF) {
      continue;
    }
    if (args_infos[idx].arg_format == ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR) {
      GE_ASSERT_TRUE(args_infos[idx].arg_size == 1U,
                     "[PARAM][INVALID] Direct addr arg size[%u], expect 1. Please check args info.",
                     args_infos[idx].arg_size);
    }

    size_t io_idx_upper_bound = 0U;
    GE_ASSERT_TRUE(!ge::AddOverflow(static_cast<size_t>(args_infos[idx].start_index),
                                    static_cast<size_t>(args_infos[idx].arg_size) - 1U, io_idx_upper_bound));
    if (args_infos[idx].arg_type == ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT) {
      GE_ASSERT_TRUE(io_idx_upper_bound < node_input_num,
                     "[PARAM][INVALID] Input index[%zu], expect in range[0, %zu). Please check args info.",
                     io_idx_upper_bound, node_input_num);
      continue;
    }
    if (args_infos[idx].arg_type == ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT) {
      GE_ASSERT_TRUE(io_idx_upper_bound < node_output_num,
                     "[PARAM][INVALID] Output index[%zu], expect in range[0, %zu). Please check args info.",
                     io_idx_upper_bound, node_output_num);
      continue;
    }
  }
  return ge::SUCCESS;
}
void DebugForArgsInfo(const ge::NodePtr &compute_node, const std::vector<ArgsInfo> &args_infos,
                      size_t received_args_info_num) {
  if (GlobalTracer::GetInstance()->GetEnableFlags() == 0U) {
    return;
  }
  auto node_io_num = compute_node->GetInDataNodesAndAnchors().size() + compute_node->GetAllOutDataAnchorsSize();
  std::stringstream ss;
  ss << "\nPrint created args info:\n"
     << "Node name[" << compute_node->GetName() << "], node type[" << compute_node->GetType() << "], node io num["
     << node_io_num << "], received args info num[" << received_args_info_num << "]. Final args info num["
     << args_infos.size() << "].";
  for (size_t idx = 0U; idx < args_infos.size(); ++idx) {
    ss << "\nArgs infos[" << idx << "]: arg type[" << static_cast<int32_t>(args_infos[idx].arg_type) << "], arg format["
       << static_cast<size_t>(args_infos[idx].arg_format) << "], arg start index[" << args_infos[idx].start_index
       << "], arg size[" << args_infos[idx].arg_size << "].";
  }
  GELOGD("%s", ss.str().c_str());
}
ge::Status ParseIoNumFromArgsInfo(const vector<ArgsInfo> &args_infos, size_t &input_args_info_num,
                                  size_t &output_args_info_num, size_t &input_valid_num, size_t &output_valid_num) {
  for (auto &args_info : args_infos) {
    switch (args_info.arg_type) {
      case ArgsInfo::ArgsInfoType::INPUT: {
        if (!args_info.be_folded) {
          ++input_valid_num;
        }
        ++input_args_info_num;
        break;
      }
      case ArgsInfo::ArgsInfoType::OUTPUT: {
        if (!args_info.be_folded) {
          ++output_valid_num;
        }
        ++output_args_info_num;
        break;
      }
      default: {
        GELOGE(ge::PARAM_INVALID, "IoType check failed.");
        return ge::FAILED;
      }
    }
  }
  return ge::SUCCESS;
}

std::vector<ValueHolderPtr> BuildFindCompatibleTilingFunc(const ge::NodePtr &node) {
  const auto &node_type = ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
  auto find_tiling_func_out_num = static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kFindFuncOutputNum);
  return ValueHolder::CreateDataOutput("FindCompatibleTilingFunc", {node_type}, find_tiling_func_out_num);
}

ValueHolderPtr BuildCompatibleTilingParse(const ge::NodePtr &node, const ValueHolderPtr &op,
                                          const std::vector<ValueHolderPtr> &tiling_found_ret) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const std::string *op_compile_info_json = ge::AttrUtils::GetStr(op_desc, optiling::COMPILE_INFO_JSON);
  if (op_compile_info_json == nullptr) {
    GELOGW("Op[%s] does not have attr[%s].", node->GetName().c_str(), optiling::COMPILE_INFO_JSON.c_str());
    return nullptr;
  }
  const auto &compile_info_json_holder =
      bg::ValueHolder::CreateConst(op_compile_info_json->c_str(), op_compile_info_json->size() + 1, true);

  const std::string *op_compile_info_key = ge::AttrUtils::GetStr(op_desc, optiling::COMPILE_INFO_KEY);
  if (op_compile_info_key == nullptr) {
    GELOGW("Op[%s] does not have attr[%s].", node->GetName().c_str(), optiling::COMPILE_INFO_KEY.c_str());
    return nullptr;
  }
  const auto &compile_info_key_holder =
      bg::ValueHolder::CreateConst(op_compile_info_key->c_str(), op_compile_info_key->size() + 1, true);

  const size_t kTilingVersionIndex = static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingVersion);
  const size_t kTilingParseFuncIndex =
      static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingParseFunc);
  return bg::ValueHolder::CreateSingleDataOutput(
      "CompatibleTilingParse", {op, compile_info_json_holder, compile_info_key_holder,
                                tiling_found_ret[kTilingVersionIndex], tiling_found_ret[kTilingParseFuncIndex]});
}

/**
 *    OpBuffer(const)                  NodeType(const)
 *         |                                |
 *         |                      FindCompatibleTilingFunc
 * CreateOpFromBuffer  compile_json       |    |    |
 *    |                    |              |    |    |
 *    |  \                 |              |    |    |
 *    |  CompatibleTilingParse<-----------|----|    |
 *    |           |                       |         |
 *   CompatibleTiling<------------------------------|
 */
std::vector<ValueHolderPtr> BuildCompatibleTilingInputs(const ge::NodePtr &node,
                                                        const std::vector<ValueHolderPtr> &input_shapes,
                                                        const std::vector<ValueHolderPtr> &output_shapes,
                                                        const std::vector<ValueHolderPtr> &tiling_func_ret,
                                                        const ValueHolderPtr &fwk_data) {
  // CreateOp
  auto op_buffer_vec = CompatibleUtils::BuildOpDescBufferConst(node);
  constexpr size_t kOpBufferNum = 2U;
  if (op_buffer_vec.size() != kOpBufferNum) {
    return {};
  }
  const auto &op_holder = ValueHolder::CreateSingleDataOutput("CreateOpFromBuffer", op_buffer_vec);

  // TilingParse
  const auto &compile_info = BuildCompatibleTilingParse(node, op_holder, tiling_func_ret);
  if (compile_info == nullptr) {
    return {};
  }
  /* Tiling, inputs order
   * op buffer, compile info, tiling_version, TilingFwkData(tiling_func, ...), all_inputs, all outputs
   * 此处为了兼容FFTS, 提供了Legacy实现, legacy实现的kTilingFwkData输入实际为tiling_func
   */
  auto build_compatible_tiling_inputs = [&input_shapes, &output_shapes, &op_holder, &compile_info, &fwk_data,
                                         &tiling_func_ret]() {
    const auto kTilingVersionIndex = static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingVersion);
    std::vector<ValueHolderPtr> tiling_inputs;
    tiling_inputs.emplace_back(op_holder);
    tiling_inputs.emplace_back(compile_info);
    tiling_inputs.emplace_back(tiling_func_ret[kTilingVersionIndex]);
    tiling_inputs.emplace_back(fwk_data);
    tiling_inputs.insert(tiling_inputs.cend(), input_shapes.cbegin(), input_shapes.cend());
    tiling_inputs.insert(tiling_inputs.cend(), output_shapes.cbegin(), output_shapes.cend());
    return tiling_inputs;
  };
  return build_compatible_tiling_inputs();
}

std::vector<ValueHolderPtr> BuildTilingCommonInputs(const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &input_shapes,
                                                    const std::vector<ValueHolderPtr> &output_shapes,
                                                    const TilingLowerInput &lower_inputs) {
  std::vector<ValueHolderPtr> tiling_input{input_shapes};
  tiling_input.insert(tiling_input.cend(), output_shapes.cbegin(), output_shapes.cend());
  const auto &compile_info = ParseCompileInfo(node, lower_inputs.platform_info, lower_inputs.global_data);
  tiling_input.emplace_back(compile_info);
  tiling_input.emplace_back(lower_inputs.platform_info);
  return tiling_input;
}

ge::Status GetArgsFormatForTilingData(const ge::NodePtr &node, std::string &args_format_str,
                                      const LoweringGlobalData::NodeCompileResult *compile_result) {
  const auto op_desc = node->GetOpDescBarePtr();
  if (ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_ALIAS_ENGINE_NAME)) {
    // MixL2场景
    const domi::TaskDef &task_def = compile_result->task_defs.at(0U);
    const domi::FftsPlusTaskDef &ffts_plus_task_def = task_def.ffts_plus_task();
    const int32_t mixl2_ctx_no = ffts_plus_task_def.ffts_plus_ctx_size() - 1;
    GELOGD("[Tiling]MixL2 OP[%s] get task no.%d", op_desc->GetNamePtr(), mixl2_ctx_no);
    if (mixl2_ctx_no >= 0) {
      const domi::FftsPlusCtxDef &ctx_def = ffts_plus_task_def.ffts_plus_ctx(mixl2_ctx_no);
      args_format_str = ctx_def.mix_aic_aiv_ctx().args_format();
    }
  } else if (op_desc->GetType() == ge::PARTITIONEDCALL) {
    GELOGW("Node[%s] ffts task not support", node->GetNamePtr());
    return ge::UNSUPPORTED;
  } else {
    const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICore);
    GE_IF_BOOL_EXEC(task_def == nullptr, GELOGW("Node[%s] get task null", node->GetNamePtr());
                    return ge::UNSUPPORTED);
    if (static_cast<ge::ModelTaskType>(task_def->type()) == ge::ModelTaskType::MODEL_TASK_ALL_KERNEL) {
      args_format_str = task_def->kernel_with_handle().context().args_format();
    } else {
      args_format_str = task_def->kernel().context().args_format();
    }
  }
  return ge::SUCCESS;
}

void CheckThirdClassOp(const ge::NodePtr &node, std::vector<int64_t> &out_args_sizes) {
  const size_t kMaxDimNum = 8;
  const size_t kShapeBufferNum = kMaxDimNum + 1U;
  int32_t unknown_shape_type_val = 0;
  const auto op_desc = node->GetOpDescBarePtr();
  (void)ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  // only for third class op
  if (static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val) != ge::DEPEND_SHAPE_RANGE) {
    return;
  }
  const size_t shape_buffer_size = node->GetOpDesc()->GetOutputsSize() * (kShapeBufferNum * sizeof(uint64_t));
  GELOGI("OP [%s] is third class op, shape buf size:%zu.", op_desc->GetNamePtr(), shape_buffer_size);
  if (shape_buffer_size != 0) {
    out_args_sizes.push_back(static_cast<int64_t>(shape_buffer_size));
  }
}

// ffts、mixl2算子 (未切至aicore) 场景不支持 TilingAppendDfxInfo
ge::Status BuildTilingAppendDfxInfo(const ge::NodePtr &node,
                                    std::vector<ValueHolderPtr> &input_vec,
                                    const TilingAppendDfxInfoInputs &dfx_inputs) {
  GELOGD("[Tiling] Start build TilingAppendDfxInfo kernel");
  const auto op_desc = node->GetOpDescBarePtr();

  bool is_mem_check_enable = false;
  (void)ge::AttrUtils::GetBool(op_desc, optiling::kMemoryCheck, is_mem_check_enable);
  bool is_args_exception_enable = Adx::AdumpGetDumpSwitch(Adx::DumpType::ARGS_EXCEPTION);
  if (!is_mem_check_enable && !is_args_exception_enable) {
    return ge::SUCCESS;
  }

  int64_t ori_op_param_size = 0L;
  (void)ge::AttrUtils::GetInt(op_desc, "ori_op_para_size", ori_op_param_size);

  // 1.input_shapes
  input_vec.insert(input_vec.cend(), dfx_inputs.input_shapes.cbegin(), dfx_inputs.input_shapes.cend());

  // 2.output_shapes
  input_vec.insert(input_vec.cend(), dfx_inputs.output_shapes.cbegin(), dfx_inputs.output_shapes.cend());

  // 3.output_sizes
  auto output_shapes_sizes = CalcInputOutputTensorSize(false, false, node, dfx_inputs.output_shapes);
  input_vec.insert(input_vec.cend(), output_shapes_sizes.cbegin(), output_shapes_sizes.cend());

  // 4.launch_arg
  input_vec.emplace_back(dfx_inputs.launch_arg);

  // 5.workspace
  input_vec.emplace_back(dfx_inputs.workspace_sizes);

  // 6.ori_op_param_size
  size_t tiling_ori_param_size = static_cast<size_t>(ori_op_param_size);
  input_vec.emplace_back(bg::ValueHolder::CreateConst(static_cast<void *>(&tiling_ori_param_size),
                         sizeof(size_t), false));

  // 7.is_mem_check_enable
  input_vec.emplace_back(bg::ValueHolder::CreateConst(static_cast<void *>(&is_mem_check_enable),
                         sizeof(bool), false));

  // 8.is_args_exception_enable
  input_vec.emplace_back(bg::ValueHolder::CreateConst(static_cast<void *>(&is_args_exception_enable),
                         sizeof(bool), false));

  // 获取args_size_list 和 args_index_to_io_index
  auto compile_result = dfx_inputs.global_data.FindCompiledResult(node);
  GE_IF_BOOL_EXEC(compile_result == nullptr, GELOGW("Node[%s] get compile result null", node->GetNamePtr());
                  return ge::UNSUPPORTED);
  std::string args_format_str;
  GE_ASSERT_SUCCESS(GetArgsFormatForTilingData(node, args_format_str, compile_result));

  std::vector<int64_t> args_size_list;
  std::vector<optiling::ArgsIndexToIoIndex> args_idx_to_io_idx;
  const auto &op_desc_ptr = node->GetOpDesc();
  if (!args_format_str.empty()) {
    GELOGI("OP [%s] has formatted args_format:[%s].", op_desc->GetNamePtr(), args_format_str.c_str());
    std::vector<ge::ArgDesc> args_descs;
    GE_ASSERT_SUCCESS(ge::ArgsFormatDesc::Parse(op_desc_ptr, args_format_str, args_descs));
    GE_ASSERT_SUCCESS(
      optiling::TilingDfx::GetArgsSizeWithArgsFormat(op_desc_ptr, args_descs, args_size_list, args_idx_to_io_idx));
  } else {
    GELOGI("OP [%s] not has formatted args_format. input_shapes size [%zu], out_shape size [%zu]",
      op_desc->GetNamePtr(),dfx_inputs.input_shapes.size(), dfx_inputs.output_shapes.size());
    GE_ASSERT_SUCCESS(
      optiling::TilingDfx::GetArgsSizeWithoutArgsFormat(
        dfx_inputs.input_shapes.size(), dfx_inputs.output_shapes.size(), args_size_list, args_idx_to_io_idx));
  }
  CheckThirdClassOp(node, args_size_list);

  if (args_size_list.size() == 0U) {
    GELOGI("OP [%s] args size list is empty", op_desc->GetNamePtr());
    input_vec.clear();
    return ge::SUCCESS;
  }

  // 9.args_size_list
  input_vec.emplace_back(CreateContVecHolder(args_size_list));

  // 10.args_index_to_io_index
  input_vec.emplace_back(CreateContVecHolder(args_idx_to_io_idx));
  return ge::SUCCESS;
}

/*                +--> Tiling <---+-----------------+----------------+------------------------+
 *              /        |         \                 \                \                        \
 *  <input-shapes> <output-shapes> TilingParse   PlatformInfo   PrepareTilingFrameworkData   Deterministic
 *                                   /     /    \                  /              \
 *                       space_registry  json    \          FindTilingFunc       (向后可扩展框架内部FwkData输入)
 *                                                \        /         \
 *                                              node_type    space_registry
 */
std::vector<ValueHolderPtr> Tiling(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                   const std::vector<ValueHolderPtr> &output_shapes,
                                   const TilingLowerInput &lower_inputs) {
  constexpr auto tiling_output_num = static_cast<size_t>(kernel::TilingExOutputIndex::kNum);
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    return {static_cast<size_t>(kernel::TilingExOutputIndex::kNum), nullptr};
  }
  // do symbol tiling
  if (NeedSymbolTiling(node)) {
    GELOGD("Node %s type %s only support symbol tiling. Try turns to tiling on symbol.", node->GetNamePtr(),
           node->GetTypePtr());
    auto inputs_holders = BuildCommonSymbolTilingInputs(node, lower_inputs, input_shapes);
    GE_ASSERT_TRUE(!inputs_holders.empty());
    auto get_tiling_cache_key_holder = BuildSymbolTilingCacheKey(input_shapes, node, lower_inputs);
    GE_ASSERT_NOTNULL(get_tiling_cache_key_holder);
    inputs_holders.emplace_back(get_tiling_cache_key_holder);
    GE_ASSERT_SUCCESS(
        BuildCacheableTilingFwkDataInputs(node, lower_inputs, 0U, "BuildSymbolTilingCacheKey", inputs_holders));
    GE_ASSERT_SUCCESS(BuildTilingDeterministicInput(node, lower_inputs.global_data, inputs_holders));
    return bg::ValueHolder::CreateDataOutput("CacheableTiling", inputs_holders, tiling_output_num);
  }
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  // To compatible with old version tiling_fun, build differnt exe graph for tiling
  if (NeedTilingCompatible(node, lower_inputs.global_data.GetSpaceRegistryV2(
                                     static_cast<gert::OppImplVersionTag>(node->GetOpDesc()->GetOppImplVersion())))) {
    GELOGD("Node %s type %s only support compatible tiling. Try turns to tiling on compatible version.",
           node->GetNamePtr(), node->GetTypePtr());
    const auto tiling_func_ret = BuildFindCompatibleTilingFunc(node);
    const auto kTilingFuncIndex = static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingFunc);
    const auto fwk_data = ValueHolder::CreateSingleDataOutput(
        "PrepareTilingFwkData", {tiling_func_ret[kTilingFuncIndex], lower_inputs.launch_arg});
    const auto &inputs = BuildCompatibleTilingInputs(node, input_shapes, output_shapes, tiling_func_ret, fwk_data);
    return bg::ValueHolder::CreateDataOutput("CompatibleTiling", inputs, tiling_output_num);
  }
  std::vector<ValueHolderPtr> tiling_input = BuildTilingCommonInputs(node, input_shapes, output_shapes, lower_inputs);

  std::vector<ValueHolderPtr> tiling_ret;
  size_t data_dependency = 0U;
  if (TilingCacheUtils::IsOpSupportTilingCache(node, lower_inputs.global_data, data_dependency)) {
    GE_ASSERT_SUCCESS(BuildCacheableTilingFwkDataInputs(
        node, lower_inputs, data_dependency, "BuildGeneralTilingCacheKey", tiling_input));
    GE_ASSERT_SUCCESS(BuildTilingDeterministicInput(node, lower_inputs.global_data, tiling_input));
    tiling_ret = ValueHolder::CreateDataOutput("CacheableTiling", tiling_input, tiling_output_num);
  } else {
    GE_ASSERT_SUCCESS(BuildTilingFwkDataInputs(node, lower_inputs, tiling_input));
    GE_ASSERT_SUCCESS(BuildTilingDeterministicInput(node, lower_inputs.global_data, tiling_input));
    tiling_ret = ValueHolder::CreateDataOutput("Tiling", tiling_input, tiling_output_num);
  }

  const auto op_desc = node->GetOpDescBarePtr();
  const auto known_workspace = op_desc->GetWorkspaceBytes();
  if (!known_workspace.empty()) {
    GELOGD("Node: %s has known workspace info.", op_desc->GetNamePtr());
    tiling_ret[TilingContext::kOutputWorkspace] = ValueHolder::CreateSingleDataOutput(
        "TilingAppendWorkspace", {tiling_ret[TilingContext::kOutputWorkspace], CreateContVecHolder(known_workspace)});
  }

  std::vector<ValueHolderPtr> input_vec;
  const auto ret = BuildTilingAppendDfxInfo(
      node, input_vec,
      {input_shapes, output_shapes, tiling_ret[static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)],
       tiling_ret[TilingContext::kOutputWorkspace], lower_inputs.global_data});
  if ((ret == ge::SUCCESS) && (input_vec.size() > 0U)) {
    tiling_ret[TilingContext::kOutputWorkspace] = ValueHolder::CreateSingleDataOutput("TilingAppendDfxInfo", input_vec);
  }
  return tiling_ret;
}

std::vector<ValueHolderPtr> FallibleTiling(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                           const std::vector<ValueHolderPtr> &output_shapes,
                                           const TilingLowerInput &lower_inputs) {
  constexpr auto fallible_output_num = static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum);
  if (node == nullptr || (node->GetOpDescBarePtr() == nullptr)) {
    return {fallible_output_num, nullptr};
  }
  // To compatible with old version tiling_fun, build different exe graph for tiling
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  ge::OppImplVersion opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
  if (NeedTilingCompatible(
          node, lower_inputs.global_data.GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(opp_impl_version)))) {
    GELOGD("Node %s type %s only support compatible tiling. Try turns to tiling on compatible version.",
           node->GetNamePtr(), node->GetTypePtr());
    const auto tiling_func_ret = BuildFindCompatibleTilingFunc(node);
    const auto kTilingFuncIndex = static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingFunc);
    const auto fwk_data = ValueHolder::CreateSingleDataOutput(
        "PrepareTilingFwkData", {tiling_func_ret[kTilingFuncIndex], lower_inputs.launch_arg});
    const auto inputs = BuildCompatibleTilingInputs(node, input_shapes, output_shapes, tiling_func_ret, fwk_data);
    return bg::ValueHolder::CreateDataOutput("FallibleCompatibleTiling", inputs, fallible_output_num);
  }
  std::vector<ValueHolderPtr> tiling_input = BuildTilingCommonInputs(node, input_shapes, output_shapes, lower_inputs);
  std::vector<ValueHolderPtr> tiling_ret;
  size_t data_dependency = 0U;
  if (TilingCacheUtils::IsOpSupportTilingCache(node, lower_inputs.global_data, data_dependency)) {
    GE_ASSERT_SUCCESS(BuildCacheableTilingFwkDataInputs(
        node, lower_inputs, data_dependency, "BuildGeneralTilingCacheKey", tiling_input));
    GE_ASSERT_SUCCESS(BuildTilingDeterministicInput(node, lower_inputs.global_data, tiling_input));
    tiling_ret = ValueHolder::CreateDataOutput("CacheableFallibleTiling", tiling_input, fallible_output_num);
  } else {
    GE_ASSERT_SUCCESS(BuildTilingFwkDataInputs(node, lower_inputs, tiling_input));
    GE_ASSERT_SUCCESS(BuildTilingDeterministicInput(node, lower_inputs.global_data, tiling_input));
    tiling_ret = ValueHolder::CreateDataOutput("FallibleTiling", tiling_input, fallible_output_num);
  }

  std::vector<ValueHolderPtr> input_vec;
  const auto ret = BuildTilingAppendDfxInfo(
      node, input_vec,
      {input_shapes, output_shapes, tiling_ret[static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)],
       tiling_ret[TilingContext::kOutputWorkspace], lower_inputs.global_data});
  if ((ret == ge::SUCCESS) && (input_vec.size() > 0U)) {
    tiling_ret[TilingContext::kOutputWorkspace] = ValueHolder::CreateSingleDataOutput("TilingAppendDfxInfo", input_vec);
  }
  return tiling_ret;
}

std::vector<ValueHolderPtr> TilingForAtomic(const ge::NodePtr &atomic_node, const ValueHolderPtr &workspaces_size,
                                            const std::vector<ValueHolderPtr> &output_clean_sizes,
                                            const ValueHolderPtr &launch_arg, LoweringGlobalData &global_data) {
  const auto &compile_info = ParseAtomicCompileInfo(atomic_node, global_data);
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(workspaces_size);
  for (const auto &output_clean_size : output_clean_sizes) {
    inputs.emplace_back(output_clean_size);
  }
  inputs.emplace_back(compile_info);
  GE_ASSERT_NOTNULL(atomic_node);
  GE_ASSERT_NOTNULL(atomic_node->GetOpDesc());
  ge::OppImplVersion opp_impl_version = atomic_node->GetOpDesc()->GetOppImplVersion();

  ValueHolderPtr tiling_func = nullptr;
  if (atomic_node->GetType() == "MemSet") {
    tiling_func = GetOrCreateTilingFunc("MemSet", opp_impl_version, global_data);
  } else {
    tiling_func = GetOrCreateTilingFunc("DynamicAtomicAddrClean", opp_impl_version, global_data);
  }
  inputs.emplace_back(ValueHolder::CreateSingleDataOutput("PrepareTilingFwkData", {tiling_func, launch_arg}));
  GE_ASSERT_SUCCESS(BuildTilingDeterministicInput(atomic_node, global_data, inputs));
  return ValueHolder::CreateDataOutput("Tiling", inputs, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

// Legacy Tiling lowering function for ffts, which will be deprecated in the future
std::vector<ValueHolderPtr> TilingLegacy(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                         const std::vector<ValueHolderPtr> &output_shapes,
                                         const ValueHolderPtr &platform_info, LoweringGlobalData &global_data) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    return {TilingContext::kOutputNum, nullptr};
  }
  // To compatible with old version tiling_fun, build differnt exe graph for tiling
  if (NeedTilingCompatible(node, global_data.GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(node->GetOpDesc()->GetOppImplVersion())))) {
    GELOGD("Node %s type %s only support compatible tiling. Try turns to tiling on compatible version.",
           node->GetNamePtr(), node->GetTypePtr());
    const auto tiling_func_ret = BuildFindCompatibleTilingFunc(node);
    const auto kTilingFuncIndex = static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingFunc);
    const auto inputs = BuildCompatibleTilingInputs(node, input_shapes, output_shapes, tiling_func_ret,
                                                    tiling_func_ret[kTilingFuncIndex]);
    return bg::ValueHolder::CreateDataOutput("CompatibleTilingLegacy", inputs, TilingContext::kOutputNum);
  }

  std::vector<ValueHolderPtr> tiling_input{input_shapes};
  tiling_input.insert(tiling_input.cend(), output_shapes.cbegin(), output_shapes.cend());
  const auto &compile_info = ParseCompileInfo(node, platform_info, global_data);
  tiling_input.emplace_back(compile_info);
  tiling_input.emplace_back(platform_info);
  std::string type;
  GE_ASSERT_SUCCESS(ge::GetOriginalType(node, type), "Failed to get original type from %s(%s).",
                    node->GetName().c_str(), node->GetType().c_str());
  ge::OppImplVersion opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
  const auto tiling_func = GetOrCreateTilingFunc(type, opp_impl_version, global_data);
  tiling_input.emplace_back(tiling_func);
  auto tiling_ret = ValueHolder::CreateDataOutput("TilingLegacy", tiling_input, TilingContext::kOutputNum);

  const auto op_desc = node->GetOpDescBarePtr();
  const auto known_workspace = op_desc->GetWorkspaceBytes();
  if (!known_workspace.empty()) {
    GELOGD("Node: %s has known workspace info.", op_desc->GetNamePtr());
    tiling_ret[TilingContext::kOutputWorkspace] = ValueHolder::CreateSingleDataOutput(
        "TilingAppendWorkspace", {tiling_ret[TilingContext::kOutputWorkspace], CreateContVecHolder(known_workspace)});
  }
  return tiling_ret;
}

std::vector<ValueHolderPtr> TilingForAtomicLegacy(const ge::NodePtr &atomic_node, const ValueHolderPtr &workspaces_size,
                                                  const std::vector<ValueHolderPtr> &output_clean_sizes,
                                                  LoweringGlobalData &global_data) {
  const auto &compile_info = ParseAtomicCompileInfo(atomic_node, global_data);
  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(workspaces_size);
  for (const auto &output_clean_size : output_clean_sizes) {
    inputs.emplace_back(output_clean_size);
  }
  inputs.emplace_back(compile_info);
  GE_ASSERT_NOTNULL(atomic_node);
  GE_ASSERT_NOTNULL(atomic_node->GetOpDesc());
  ge::OppImplVersion opp_impl_version = atomic_node->GetOpDesc()->GetOppImplVersion();

  ValueHolderPtr tiling_func = nullptr;
  if (atomic_node->GetType() == "MemSet") {
    tiling_func = GetOrCreateTilingFunc("MemSet", opp_impl_version, global_data);
  } else {
    tiling_func = GetOrCreateTilingFunc("DynamicAtomicAddrClean", opp_impl_version, global_data);
  }
  inputs.emplace_back(tiling_func);
  return ValueHolder::CreateDataOutput("TilingLegacy", inputs, TilingContext::kOutputNum);
}
}  // namespace bg
}  // namespace gert
