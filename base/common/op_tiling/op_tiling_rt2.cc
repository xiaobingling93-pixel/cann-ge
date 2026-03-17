/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nlohmann/json.hpp"
#include "op_tiling_rt2.h"

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/sgt_slice_type.h"
#include "common/util.h"
#include "graph/def_types.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ir_definitions_recover.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/math_util.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/lowering/tiling_context_builder.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "register/op_tiling_registry.h"
#include "runtime/mem.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/ge_local_context.h"
#include "mmpa/mmpa_api.h"
#include "graph/op_kernel_bin.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/utils/attr_utils.h"
#include "base/err_msg.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "runtime/dev.h"
#include "register/core_num_utils.h"
#include "acl/acl_rt.h"

namespace optiling {
class TilingSymbolEvalContext;
class SymbolTilingParseContext;
using TilingFunc = ge::graphStatus(*)(TilingSymbolEvalContext *);
using GetTilingDataFunc = size_t(*)();
using TilingParse = ge::graphStatus(*)(SymbolTilingParseContext *);
namespace {
const std::string kCompileInfoJson = "compile_info_json";
constexpr size_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr size_t kMaxWorkspaceCount = 16;
const std::string kMaxTilingSize = "op_para_size";
const std::string kMaxAtomicCleanTilingSize = "atomic_op_para_size";
const std::string kMemSetAttrKey = "tbe_op_atomic_dtypes";
const std::string kDefaultCoreType = "Aicore";
const std::string kMemSetOpType = "MemSet";
const std::string kCompileInfoTmpName = "temp";
const std::string kMatMulSubgraphName = "matmul_subgraph";
const std::string kMatMulNodeType = "MatMulV3";
const std::string kMatMulBatchMatmulNodeType = "BatchMatMulV3";
constexpr int32_t kSocVersionLen = 50;

gert::KernelContextHolder BuildTilingParseContextHolder(const ge::OpDescPtr &op_desc, const char_t * const compile_info,
    const fe::PlatFormInfos &platform_infos, const gert::OpImplKernelRegistry::OpImplFunctions * const funcs) {
  std::vector<std::pair<void *, gert::Chain::Deleter>> tiling_parse_outputs(1, std::make_pair(nullptr, nullptr));
  tiling_parse_outputs[0].first = funcs->compile_info_creator();
  tiling_parse_outputs[0].second = funcs->compile_info_deleter;
  return gert::KernelRunContextBuilder()
      .Inputs({const_cast<ge::char_t *>(compile_info)})
      .Inputs({reinterpret_cast<void *>(const_cast<fe::PlatFormInfos *>(&platform_infos))})
      .Inputs({const_cast<ge::char_t *>(op_desc->GetType().c_str())})
      .Outputs(tiling_parse_outputs)
      .Build(op_desc);
}

ge::graphStatus FindImplFuncs(const char * const op_type, const gert::OpImplKernelRegistry::OpImplFunctions *&funcs,
                              const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  if (space_registry == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Invalid space registry!");
    return ge::GRAPH_FAILED;
  }
  funcs = space_registry->GetOpImpl(op_type);
  if ((funcs == nullptr) || (funcs->tiling == nullptr) || (funcs->tiling_parse == nullptr)) {
    funcs = space_registry->GetOpImpl("DefaultImpl");
    GELOGD("funcs/tiling/tiling_parse is null. op type is %s. Try auto tiling", op_type);
    if ((funcs == nullptr) || (funcs->tiling == nullptr) || (funcs->tiling_parse == nullptr)) {
      GELOGE(ge::GRAPH_FAILED, "auto funcs/tiling/tiling_parse is null. op type is %s.", op_type);
      REPORT_INNER_ERR_MSG("E19999", "auto funcs/tiling/tiling_parse is null. op type is %s.", op_type);
      return ge::GRAPH_FAILED;
    }
  }
  GELOGD("Found rt2 tiling func. op type is %s.", op_type);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvertFromContextToRunInfo(gert::KernelContext *kernel_context, OpRunInfoV2 &run_info) {
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(kernel_context);
  run_info.SetTilingKey(tiling_context->GetTilingKey());
  run_info.SetBlockDim(tiling_context->GetBlockDim());
  run_info.SetAicpuBlockDim(tiling_context->GetAicpuBlockDim());
  run_info.SetClearAtomic(tiling_context->NeedAtomic());
  auto tiling_data = tiling_context->GetRawTilingData();
  run_info.AddTilingData(reinterpret_cast<ge::char_t *>(tiling_data->GetData()), tiling_data->GetDataSize());
  auto ws = kernel_context->GetOutputPointer<gert::ContinuousVector>(gert::TilingContext::kOutputWorkspace);
  GE_ASSERT_NOTNULL(ws);
  std::vector<int64_t> workspaces(reinterpret_cast<const size_t *>(ws->GetData()),
                                  ge::PtrAdd(reinterpret_cast<const size_t *>(ws->GetData()), std::numeric_limits<size_t>::max(), ws->GetSize()));

  run_info.SetWorkspaces(workspaces);
  run_info.SetTilingCond(tiling_context->GetTilingCond());
  run_info.SetScheduleMode(tiling_context->GetScheduleMode());
  run_info.SetLocalMemorySize(tiling_context->GetLocalMemorySize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RtTilingParse(const ge::OpDescPtr &op_desc, const char_t *const compile_info,
                              const fe::PlatFormInfos &platform_infos, ParseContextHolderPtr &context_holder,
                              const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  GE_ASSERT_NOTNULL(op_desc);
  bool is_soft_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync);
  ParseContextHolderPtr default_context_holder;
  context_holder = op_desc->TryGetExtAttr(OP_TILING_PARSE_RESULT, default_context_holder);
  if ((context_holder != nullptr) && (!is_soft_sync)) {
    return ge::GRAPH_SUCCESS;
  }

  const gert::OpImplKernelRegistry::OpImplFunctions *funcs;
  GE_ASSERT_GRAPH_SUCCESS(FindImplFuncs(op_desc->GetType().c_str(), funcs, space_registry));

  auto parse_context_holder = BuildTilingParseContextHolder(op_desc, compile_info, platform_infos, funcs);
  context_holder = ge::MakeShared<gert::KernelContextHolder>(std::move(parse_context_holder));
  GE_ASSERT_NOTNULL(context_holder);
  GE_ASSERT_NOTNULL(context_holder->context_);
  GE_ASSERT_GRAPH_SUCCESS((funcs->tiling_parse)(context_holder->context_), "Op %s tiling parse failed",
                          op_desc->GetType().c_str());
  (void)op_desc->SetExtAttr(OP_TILING_PARSE_RESULT, context_holder);
  return ge::GRAPH_SUCCESS;
}

ge::Status GetCleanIndexes(const ge::NodePtr &node, std::vector<int64_t> &clean_workspace_indexes,
                           std::vector<int64_t> &clean_output_indexes) {
  const auto &op_desc = node->GetOpDesc();
  GELOGI("Begin to do Atomic optiling for op[%s, %s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, clean_output_indexes);
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info;
  atomic_workspace_info = op_desc->TryGetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  if (!atomic_workspace_info.empty()) {
    const std::map<int64_t, int64_t> &workspace_idx = atomic_workspace_info[op_desc->GetName()];
    for (const auto &ws_index : workspace_idx) {
      (void)clean_workspace_indexes.emplace_back(ws_index.first);
    }
  }
  return ge::SUCCESS;
}

ge::Status GetCleanOutputSizes(const ge::NodePtr &origin_node, const std::vector<int64_t> &outputs_indexes,
                               std::vector<int64_t> &output_clean_sizes) {
  const auto origin_op_desc = origin_node->GetOpDesc();
  for (const int64_t idx : outputs_indexes) {
    const ge::ConstGeTensorDescPtr tensor = origin_op_desc->GetOutputDescPtr(static_cast<uint32_t>(idx));
    GE_ASSERT_NOTNULL(tensor, "Get atomic_output_indice failed. op_type: DynamicAtomicAddrClean, op_name:%s",
                      origin_op_desc->GetName().c_str());

    int64_t clean_size = 0;
    GE_ASSERT_GRAPH_SUCCESS(ge::TensorUtils::GetSize(*tensor, clean_size),
                            "Get size of tensor desc failed. op_type: DynamicAtomicAddrClean, op_name:%s",
                            origin_op_desc->GetName().c_str());
    output_clean_sizes.push_back(clean_size);
  }
  return ge::SUCCESS;
}

ge::Status AddDataNodeForAtomic(ge::ComputeGraphPtr &graph, ge::NodePtr &clean_node, size_t output_size) {
  // add data node for workspace
  auto workspace_data_op_desc = std::make_shared<ge::OpDesc>(clean_node->GetName() + "_Data_0", "Data");
  GE_CHECK_NOTNULL(workspace_data_op_desc);
  if (workspace_data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "workspace_data_op_desc add output desc failed");
    return ge::FAILED;
  }
  const auto workspace_data_node = graph->AddNode(workspace_data_op_desc);
  GE_CHECK_NOTNULL(workspace_data_node);
  ge::graphStatus ret = ge::GraphUtils::AddEdge(workspace_data_node->GetOutDataAnchor(0),
                                                clean_node->GetInDataAnchor(0));
  if (ret != ge::SUCCESS) {
    GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", workspace_data_node->GetName().c_str(),
           clean_node->GetName().c_str());
    return ge::FAILED;
  }

  // add data node for output
  for (size_t i = 0U; i < output_size; ++i) {
    auto data_op_desc = std::make_shared<ge::OpDesc>(clean_node->GetName() + "_Data_" + std::to_string(i + 1), "Data");
    GE_CHECK_NOTNULL(data_op_desc);
    if (data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "data_op_desc add output desc failed, i = %zu", i);
      return ge::FAILED;
    }
    auto data_node = graph->AddNode(data_op_desc);
    GE_CHECK_NOTNULL(data_node);
    ret = ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), clean_node->GetInDataAnchor(i + 1));
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", data_node->GetName().c_str(),
             clean_node->GetName().c_str());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

ge::NodePtr BuildAtomicNode(const ge::NodePtr &origin_node, std::vector<int64_t> &output_clean_sizes,
                            ge::ComputeGraphPtr &graph) {
  GELOGD("Generate atomic logic for node %s type %s", origin_node->GetName().c_str(), origin_node->GetType().c_str());
  std::vector<int64_t> workspace_indexes;
  std::vector<int64_t> outputs_indexes;
  GE_ASSERT_SUCCESS(GetCleanIndexes(origin_node, workspace_indexes, outputs_indexes), "Fail to get clean index of %s",
                    origin_node->GetName().c_str());
  GE_ASSERT_SUCCESS(GetCleanOutputSizes(origin_node, outputs_indexes, output_clean_sizes),
                    "Fail to get output clean size of %s", origin_node->GetName().c_str());

  std::shared_ptr<ge::OpDesc> atomic_op_desc;
  if (origin_node->GetOpDesc()->HasAttr(kMemSetAttrKey)) {
    atomic_op_desc = std::make_shared<ge::OpDesc>(origin_node->GetName() + "_MemSet", "MemSet");
  } else {
    atomic_op_desc = std::make_shared<ge::OpDesc>(origin_node->GetName() + "_AtomicClean", "DynamicAtomicAddrClean");
  }
  GE_ASSERT_NOTNULL(atomic_op_desc);

  atomic_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  atomic_op_desc->AppendIrInput("output", ge::kIrInputDynamic);

  GE_ASSERT_SUCCESS(atomic_op_desc->AddInputDesc("workspace", ge::GeTensorDesc()));
  for (size_t i = 0U; i < outputs_indexes.size(); ++i) {
    GE_ASSERT_SUCCESS(atomic_op_desc->AddInputDesc("output" + std::to_string(i + 1U), ge::GeTensorDesc()));
  }
  if (!ge::AttrUtils::SetListInt(atomic_op_desc, "WorkspaceIndexes", workspace_indexes)) {
    return nullptr;
  }
  auto clean_node = graph->AddNode(atomic_op_desc);
  if (clean_node == nullptr) {
    GELOGE(ge::FAILED, "add node failed");
    return nullptr;
  }
  if (AddDataNodeForAtomic(graph, clean_node, outputs_indexes.size()) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "add data node for atomic clean node failed, outputs_indexes size = %zu",
           outputs_indexes.size());
    return nullptr;
  }
  return clean_node;
}

ge::NodePtr BuildMemsetNode(const ge::NodePtr &node, std::vector<int64_t> &workspace_bytes,
                            ge::ComputeGraphPtr &graph) {
  GELOGD("Generate atomic logic for node %s type %s", node->GetName().c_str(), node->GetType().c_str());
  const auto &op_desc = node->GetOpDesc();
  workspace_bytes = op_desc->GetWorkspaceBytes();
  const std::vector<int64_t> workspaces = op_desc->GetWorkspace();

  const auto atomic_op_desc = std::make_shared<ge::OpDesc>(node->GetName(), "MemSet");
  GE_ASSERT_NOTNULL(atomic_op_desc);

  atomic_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  atomic_op_desc->AppendIrInput("output", ge::kIrInputDynamic);

  GE_ASSERT_SUCCESS(atomic_op_desc->AddInputDesc("workspace", ge::GeTensorDesc()));
  for (size_t i = 0U; i < workspace_bytes.size(); ++i) {
    GE_ASSERT_SUCCESS(atomic_op_desc->AddInputDesc("output" + std::to_string(i + 1U), ge::GeTensorDesc()));
  }
  auto clean_node = graph->AddNode(atomic_op_desc);
  GE_ASSERT_NOTNULL(clean_node);
  GE_ASSERT_SUCCESS(AddDataNodeForAtomic(graph, clean_node, workspaces.size()));

  return clean_node;
}

ge::Status AssembleAtomicCompileInfoJson(const ge::OpDescPtr &op_desc, std::string &op_compile_info_json) {
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

ge::graphStatus UpdateNodeShapeBySliceInfo(const ffts::ThreadSliceMapDyPtr &slice_info_ptr,
                                           const ge::OpDescPtr &op_desc,
                                           const uint32_t thread_id, vector<int64_t> &ori_shape, bool &same_shape) {
  if ((thread_id >= slice_info_ptr->input_tensor_slice.size()) ||
      (thread_id >= slice_info_ptr->output_tensor_slice.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Update node shape thread id(%u) err.", thread_id);
    return ge::GRAPH_FAILED;
  }
  ge::GeTensorDescPtr tensor_ptr = nullptr;
  for (auto &index : slice_info_ptr->input_tensor_indexes) {
    tensor_ptr = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape &shape = tensor_ptr->MutableShape();
    auto &tmp_dim = slice_info_ptr->input_tensor_slice[static_cast<size_t>(thread_id)][static_cast<size_t>(index)];
    if (tmp_dim.empty()) {
      return ge::GRAPH_FAILED;
    }
    if (thread_id == 0U) {
      (void)ori_shape.emplace_back(shape.GetDim(0));
      auto &tail_dim = slice_info_ptr->input_tensor_slice[static_cast<size_t>(slice_info_ptr->slice_instance_num - 1U)]
                                                         [static_cast<size_t>(index)];
      if (tail_dim.empty()) {
        return ge::GRAPH_FAILED;
      }
      if (tail_dim[0] != tmp_dim[0]) {
        same_shape = false;
      }
    }
    (void)shape.SetDim(0, tmp_dim[0]);
  }
  for (auto &index : slice_info_ptr->output_tensor_indexes) {
    tensor_ptr = op_desc->MutableOutputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape &shape = tensor_ptr->MutableShape();
    if (thread_id == 0U) {
      (void)ori_shape.emplace_back(shape.GetDim(0));
    }
    auto &tmp_dim = slice_info_ptr->output_tensor_slice[static_cast<size_t>(thread_id)][static_cast<size_t>(index)];
    if (tmp_dim.empty()) {
      return ge::GRAPH_FAILED;
    }
    (void)shape.SetDim(0, tmp_dim[0]);
    GELOGD("Output anchor:%u set dim 0 to %ld", index, tmp_dim[0]);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateNodeShapeBack(const ge::OpDescPtr &op_desc, const ffts::ThreadSliceMapDyPtr &slice_info_ptr,
                                    const vector<int64_t> &ori_shape) {
  if (ori_shape.size() !=
      (slice_info_ptr->input_tensor_indexes.size() + slice_info_ptr->output_tensor_indexes.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Update back node shape size err.");
    return ge::GRAPH_FAILED;
  }
  size_t idx = 0;
  for (auto &index : slice_info_ptr->input_tensor_indexes) {
    ge::GeTensorDescPtr tensor_ptr = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape &shape = tensor_ptr->MutableShape();
    (void)shape.SetDim(0, ori_shape[idx++]);
  }
  for (auto &index : slice_info_ptr->output_tensor_indexes) {
    ge::GeTensorDescPtr tensor_ptr = op_desc->MutableOutputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape &shape = tensor_ptr->MutableShape();
    (void)shape.SetDim(0, ori_shape[idx++]);
  }
  GELOGD("Update node shape back success.");
  return ge::GRAPH_SUCCESS;
}

void UpdateCoreNumByCoreType(const ge::OpDescPtr &op_desc, fe::PlatFormInfos &platform_infos) {
  const std::string * const core_type = ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  // The old OM model does not have cube_vector_core_type attribute, only has sgt_cube_vector attribute,
  // so make it compatible here. By default, all OM models are AIC.
#ifdef __GNUC__
  if (core_type == nullptr) {
    platform_infos.SetCoreNumByCoreType(kDefaultCoreType);
    GELOGI("Set core num [%u] by core type: AIC, node: %s.", platform_infos.GetCoreNum(), op_desc->GetNamePtr());
  } else {
    platform_infos.SetCoreNumByCoreType(*core_type);
    GELOGI("Set core num [%u] by core type: %s, node: %s.", platform_infos.GetCoreNum(), core_type->c_str(),
           op_desc->GetNamePtr());
  }
#endif
}
} // namespace

bool EnableRt2Tiling(const ge::OpDescPtr &op_desc) {
  GE_ASSERT_NOTNULL(op_desc);
  const std::string op_type = op_desc->GetType();
  if (op_desc->GetTilingFuncInfo() != nullptr) {
    GELOGD("RT1 optiling function is found by op type[%s].", op_type.c_str());
    return false;
  }

  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(op_type);
  if (iter != op_func_map.end()) {
    GELOGD("RT1 optiling function is found by op type[%s].", op_type.c_str());
    return false;
  }
  const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
    .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()));
  if ((space_registry != nullptr) && (space_registry->GetOpImpl(op_type.c_str()) != nullptr) &&
      (space_registry->GetOpImpl(op_type.c_str())->tiling != nullptr) &&
      (space_registry->GetOpImpl(op_type.c_str())->tiling_parse != nullptr)) {
    GELOGD("RT2 optiling function is found by op type[%s].", op_type.c_str());
    return true;
  }

  iter = op_func_map.find(OP_TYPE_AUTO_TILING);
  if (iter != op_func_map.end()) {
    GELOGD("RT1 Optiling function of op type[%s] is found by Autotiling.", op_type.c_str());
    return false;
  }
  GELOGD("RT1 Optiling function of op type[%s] not found, go with rt2.", op_type.c_str());
  return true;
}

bool EnableAtomicRt2Tiling(const ge::OpDescPtr &op_desc) {
  const std::string op_type = op_desc->GetType();
  if (op_desc->GetAtomicTilingFuncInfo() != nullptr) {
    GELOGD("RT1 atomic optiling function is found by op type[%s].", op_type.c_str());
    return false;
  }

  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find("DynamicAtomicAddrClean");
  if (iter != op_func_map.end()) {
    GELOGD("RT1 atomic optiling function is found by op type[DynamicAtomicAddrClean].");
    return false;
  }
  GELOGD("RT1 Optiling function of op type[DynamicAtomicAddrClean] not found, go with rt2.");
  return true;
}

static ge::graphStatus AtomicCleanRtParseAndTiling(const ge::NodePtr &origin_node, const ge::Operator &op,
                                                   const fe::PlatFormInfos &platform_infos,
                                                   const std::vector<int64_t> &output_clean_size,
                                                   const OutputsConvertorFun &callback) {
  ParseContextHolderPtr parse_context_holder = nullptr;
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  const std::string origin_op_type = origin_node->GetOpDesc()->GetType();
  std::string json_attr_name = optiling::ATOMIC_COMPILE_INFO_JSON;
  std::string max_size_attr_name = kMaxAtomicCleanTilingSize;
  if (origin_op_type == kMemSetOpType.c_str()) {
    json_attr_name = optiling::COMPILE_INFO_JSON;
    max_size_attr_name = kMaxTilingSize;
  }
  auto op_compile_info_json = ge::AttrUtils::GetStr(op_desc, json_attr_name);
  GE_ASSERT_NOTNULL(op_compile_info_json);
  const auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()));
  GE_ASSERT_GRAPH_SUCCESS(
      RtTilingParse(op_desc, op_compile_info_json->c_str(), platform_infos, parse_context_holder, space_registry));

  // alloc tiling data & workspace
  int64_t max_size = -1;
  if ((!ge::AttrUtils::GetInt(origin_node->GetOpDesc(), max_size_attr_name, max_size)) || (max_size < 0)) {
    GELOGI("No atomic max tiling size attr in opdesc %s.", origin_node->GetName().c_str());
    return ge::GRAPH_SUCCESS;
  }
  const auto aligned_max_size = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  const auto tiling_data = gert::TilingData::CreateCap(aligned_max_size);
  GE_ASSERT_NOTNULL(tiling_data);
  const auto workspace_size = gert::ContinuousVector::Create<size_t>(kMaxWorkspaceCount);
  GE_ASSERT_NOTNULL(workspace_size);

  // get origin node worksapce
  const auto origin_op_desc = origin_node->GetOpDesc();
  const std::vector<int64_t> origin_workspace_bytes = origin_op_desc->GetWorkspaceBytes();
  const auto clean_workspace_size = gert::ContinuousVector::Create<size_t>(origin_workspace_bytes.size());
  GE_ASSERT_NOTNULL(clean_workspace_size);
  const auto clean_workspace = reinterpret_cast<gert::TypedContinuousVector<size_t> *>(clean_workspace_size.get());
  GE_ASSERT_GRAPH_SUCCESS(clean_workspace->SetSize(origin_workspace_bytes.size()));
  GELOGD("Atomic node: %s tiling data size: %ld, workspace size: %zu.", op_desc->GetName().c_str(), aligned_max_size,
         origin_workspace_bytes.size());
  for (size_t i = 0; i < origin_workspace_bytes.size(); ++i) {
    *(clean_workspace->MutableData() + i) = origin_workspace_bytes[i];
  }
  const gert::KernelContextHolder tiling_context_holder =
      gert::AtomicTilingContextBuilder()
        .CompileInfo(*parse_context_holder->context_->GetOutputPointer<void **>(0))
        .CleanWorkspaceSizes(reinterpret_cast<gert::ContinuousVector *>(clean_workspace_size.get()))
        .CleanOutputSizes(output_clean_size)
        .TilingData(tiling_data.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
        .Build(op);
  const gert::OpImplKernelRegistry::OpImplFunctions *funcs;
  GE_ASSERT_GRAPH_SUCCESS(FindImplFuncs(op_desc->GetType().c_str(), funcs, space_registry));
  GE_CHECK_NOTNULL(tiling_context_holder.context_);
  GE_ASSERT_GRAPH_SUCCESS((funcs->tiling)(reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_)));
  GE_ASSERT_GRAPH_SUCCESS(callback(tiling_context_holder.context_));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RtParseAndTiling(const ge::Operator &op, const char_t * const compile_info,
                                 const fe::PlatFormInfos &platform_infos, const OutputsConvertorFun &callback,
                                 const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  ParseContextHolderPtr parse_context_holder = nullptr;
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_GRAPH_SUCCESS(RtTilingParse(op_desc, compile_info, platform_infos, parse_context_holder, space_registry));

  // tiling
  int64_t max_size = -1;
  if (!ge::AttrUtils::GetInt(op_desc, kMaxTilingSize, max_size) || max_size < 0) {
    GELOGI("No max tiling size in opdesc.");
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }

  std::array<char_t, static_cast<size_t>(kSocVersionLen)> soc_version{};
  const char* soc_name = aclrtGetSocName();
  if (soc_name == nullptr) {
    GE_CHK_RT_RET(ACL_ERROR_FAILURE);
  }
  const auto ret = strncpy_s(soc_version.data(), kSocVersionLen, soc_name, static_cast<uint32_t>(kSocVersionLen) - 1);
  if (ret != 0) {
    GE_CHK_RT_RET(ACL_ERROR_FAILURE);
  }

  fe::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::CoreNumUtils::GetGeDefaultPlatformInfo(soc_version.data(), platform_info));

  fe::PlatFormInfos platform_infos_bak = platform_infos;

  // 如果配置了算子级核数，更新到副本PlatformInfos中，后续用副本，防止影响其他算子
  bool is_op_core_num_set = false;
  GE_ASSERT_SUCCESS(ge::CoreNumUtils::UpdatePlatformInfosWithOpDesc(platform_info, op_desc, platform_infos_bak, is_op_core_num_set));

  const auto aligned_max_size = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  const auto tiling_data = gert::TilingData::CreateCap(aligned_max_size);
  GE_ASSERT_NOTNULL(tiling_data);
  const auto workspace_size = gert::ContinuousVector::Create<size_t>(kMaxWorkspaceCount);
  GE_ASSERT_NOTNULL(workspace_size);
  std::string deterministic_str;
  (void)ge::GetThreadLocalContext().GetOption(ge::DETERMINISTIC, deterministic_str);
  const int32_t deterministic = deterministic_str == "1" ? 1 : 0;
  int32_t deterministic_level = 0;
  GE_ASSERT_SUCCESS(GetDeterministicLevel(deterministic_level));

  /*
   * 后续切换OpTilingContextBuilder时，出于兼容性考虑（新GE包+老metadef包），建议deterministic_level的设置通过调用纯C弱符号接口
   * gert_TilingContextBuilder_SetDeterministicLevel实现，调用前可以通过该符号是否为空或者aclsysGetVersionNum > 80500000 (8.5.0)
   * 判断是否是支持该能力的metadef版本。
   */
  auto context_builder = gert::TilingContextBuilder();
  const gert::KernelContextHolder tiling_context_holder =
      context_builder
          .CompileInfo(*parse_context_holder->context_->GetOutputPointer<void **>(0))
          .PlatformInfo(&platform_infos_bak)
          .TilingData(tiling_data.get())
          .Deterministic(deterministic)
          .DeterministicLevel(deterministic_level)
          .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
          .SetSpaceRegistryV2(space_registry, static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .Build(op);

  const gert::OpImplKernelRegistry::OpImplFunctions *funcs;
  GE_ASSERT(op_desc->GetOppImplVersion() < ge::OppImplVersion::kVersionEnd);
  GE_ASSERT_GRAPH_SUCCESS(FindImplFuncs(op_desc->GetType().c_str(), funcs, space_registry));
  GE_CHECK_NOTNULL(tiling_context_holder.context_);
  GE_ASSERT_GRAPH_SUCCESS((funcs->tiling)(reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_)));
  GE_ASSERT_GRAPH_SUCCESS(callback(tiling_context_holder.context_));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NormalAicoreNodeTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       const OutputsConvertorFun &callback) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  const std::string *op_compile_info_json = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON);
  if (op_compile_info_json == nullptr) {
    GE_LOGE("Op[%s] does not have attr[%s].", op_desc->GetName().c_str(), COMPILE_INFO_JSON.c_str());
    return ge::GRAPH_FAILED;
  }
  const auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()));
  GE_ASSERT_NOTNULL(space_registry);
  return RtParseAndTiling(op, op_compile_info_json->c_str(), platform_infos, callback, space_registry);
}

static ge::graphStatus BuildGraphInputShape(const ge::GeTensorDesc &tensor_desc,
                                            std::unique_ptr<uint8_t[]> &shape_holder) {
  gert::StorageShape storage_shape;
  const auto &storage_dims = tensor_desc.GetShape().GetDims();
  for (const auto &dim : storage_dims) {
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
  const auto &origin_dims = tensor_desc.GetOriginShape().GetDims();
  for (const auto &dim : origin_dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
  }
  shape_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::StorageShape));
  GE_ASSERT_NOTNULL(shape_holder);
  auto shape_ptr = ge::PtrToPtr<uint8_t, gert::StorageShape>(shape_holder.get());
  shape_ptr->MutableStorageShape() = storage_shape.GetStorageShape();
  shape_ptr->MutableOriginShape() = storage_shape.GetOriginShape();
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AutofuseNodeCreateInput(const std::vector<ge::NodePtr> &data_input_nodes,
                                               std::vector<void*> &inputs_holder) {
  for (const auto &data_node : data_input_nodes) {
    int64_t index = -1;
    (void)ge::AttrUtils::GetInt(data_node->GetOpDesc(), ge::ATTR_NAME_INDEX, index);
    GE_ASSERT_TRUE((index >= 0) && (index < static_cast<int64_t>(inputs_holder.size())),
                   "Index:%lld of node:%s should in range[0, %zu)", index, data_node->GetName().c_str(),
                   inputs_holder.size());
    std::unique_ptr<uint8_t[]> shape_holder;
    const auto data_op_desc = data_node->GetOpDesc();
    GE_ASSERT_NOTNULL(data_op_desc);
    GE_ASSERT_SUCCESS(BuildGraphInputShape(data_op_desc->GetOutputDesc(0), shape_holder));
    inputs_holder[static_cast<size_t>(index)] = static_cast<void *>(shape_holder.get());
  }

  return ge::GRAPH_SUCCESS;
}

static gert::KernelContextHolder DoTilingParse(
    const fe::PlatFormInfos &platform_infos, const ge::OpDescPtr &op_desc, void *handle) {
  const std::string tiling_parse_func_name("TilingParse");
  TilingParse const tiling_parse_func =
      reinterpret_cast<TilingParse>(mmDlsym(handle, tiling_parse_func_name.c_str()));
  GE_ASSERT_NOTNULL(tiling_parse_func);
  auto autofuse_tiling_parse_context_holder = gert::KernelRunContextBuilder()
      .Inputs({reinterpret_cast<void *>(const_cast<fe::PlatFormInfos *>(&platform_infos))})
      .Outputs({nullptr})
      .Build(op_desc);
  auto autofuse_tiling_parse_context = autofuse_tiling_parse_context_holder.context_;
  GE_ASSERT_NOTNULL(autofuse_tiling_parse_context);
  GE_ASSERT_SUCCESS(tiling_parse_func(reinterpret_cast<SymbolTilingParseContext *>(autofuse_tiling_parse_context)));
  return autofuse_tiling_parse_context_holder;
}

static void *DlopenAutofuseSo(const ge::OpDescPtr &op_desc) {
  auto tiling_so_path = ge::AttrUtils::GetStr(op_desc, "bin_file_path");
  GE_ASSERT_NOTNULL(tiling_so_path);
  std::array<char_t, MMPA_MAX_PATH> real_path{};
  GELOGI("Get autofuse tiling so path: %s", tiling_so_path->c_str());
  GE_ASSERT_TRUE(mmRealPath(tiling_so_path->c_str(), real_path.data(), MMPA_MAX_PATH) == EN_OK);
  return mmDlopen(real_path.data(), static_cast<int32_t>(MMPA_RTLD_NOW));
}

static void *GetTilingData(gert::KernelContextHolder &tiling_parse_holder) {
  auto tiling_parse_context = tiling_parse_holder.GetKernelContext();
  GE_ASSERT_NOTNULL(tiling_parse_context);
  auto tiling_parse_data_av = tiling_parse_context->GetOutput(0U);
  GE_ASSERT_NOTNULL(tiling_parse_data_av);
  return tiling_parse_data_av->GetValue<void *>();
}

static ge::graphStatus AutofuseNodeTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                          const OutputsConvertorFun &callback) {
  const auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  GE_ASSERT_NOTNULL(node);
  const auto graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph);
  const auto root_graph = ge::GraphUtils::FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);
  const auto input_nodes = root_graph->GetInputNodes();
  std::vector<ge::NodePtr> data_input_nodes;
  for (const auto &input_node : input_nodes) {
    GE_ASSERT_NOTNULL(input_node);
    if (input_node->GetType() == ge::DATA) {
      (void)data_input_nodes.emplace_back(input_node);
    }
  }
  // 构造输入, 构造输出
  std::vector<void *> inputs_holder(data_input_nodes.size());
  GE_ASSERT_TRUE(AutofuseNodeCreateInput(data_input_nodes, inputs_holder) == ge::GRAPH_SUCCESS);
  // 打开Autofuse so
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto handle = DlopenAutofuseSo(op_desc);
  GE_ASSERT_NOTNULL(handle);
  GE_MAKE_GUARD(close_handle, [&handle]() {
    (void)mmDlclose(handle);
  });

  auto tiling_parse_context_holder = DoTilingParse(platform_infos, op_desc, handle);
  auto tiling_parse_data = GetTilingData(tiling_parse_context_holder);
  GE_ASSERT_NOTNULL(tiling_parse_data);

  // 构造输出
  std::vector<void *> outputs_holder(static_cast<size_t>(gert::TilingContext::kOutputNum));
  const auto workspace_size = gert::ContinuousVector::Create<size_t>(kMaxWorkspaceCount);
  GE_ASSERT_NOTNULL(workspace_size);
  outputs_holder[static_cast<size_t>(gert::TilingContext::kOutputWorkspace)] =
      static_cast<void *>(workspace_size.get());

  // 获取tiling函数
  const std::string get_tiling_data_size_func_name("GetTilingDataSize");
  GetTilingDataFunc const get_tiling_data_size_func =
      reinterpret_cast<GetTilingDataFunc>(mmDlsym(handle, get_tiling_data_size_func_name.c_str()));
  GE_ASSERT_NOTNULL(get_tiling_data_size_func, "Failed to Dlsym function: %s", get_tiling_data_size_func_name.c_str());
  const auto tiling_data_size = get_tiling_data_size_func();
  auto tiling_data_holder =
      gert::TilingData::CreateCap(ge::RoundUp(static_cast<uint64_t>(tiling_data_size), sizeof(uintptr_t)));
  outputs_holder[static_cast<size_t>(gert::TilingContext::kOutputTilingData)] =
      static_cast<void *>(tiling_data_holder.get());

  const auto input_data_size = data_input_nodes.size();
  const auto autofuse_tiling_context_holder =
      gert::KernelRunContextBuilder()
          .Inputs({reinterpret_cast<void*>(input_data_size)})
          .Inputs(inputs_holder)
          .Inputs({tiling_parse_data})
          .Outputs(outputs_holder)
          .Build(node->GetOpDesc());
  GE_ASSERT_NOTNULL(autofuse_tiling_context_holder.context_);
  const std::string tiling_func_name("TilingFunc");
  TilingFunc const tiling_func = reinterpret_cast<TilingFunc>(mmDlsym(handle, tiling_func_name.c_str()));
  GE_ASSERT_NOTNULL(tiling_func, "Failed to Dlsym function: %s", tiling_func_name.c_str());
  GE_ASSERT_GRAPH_SUCCESS(
      tiling_func(reinterpret_cast<TilingSymbolEvalContext *>(autofuse_tiling_context_holder.context_)));
  GE_ASSERT_GRAPH_SUCCESS(callback(autofuse_tiling_context_holder.context_));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleMatmulTilingCallback(gert::KernelContext *kernel_context, OpRunInfoV2 &run_info,
                                                  size_t &wk_size, uint32_t &aic_num) {
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(kernel_context);
  GE_ASSERT_NOTNULL(tiling_context);
  size_t *workspaces_size = tiling_context->GetWorkspaceSizes(1);
  GE_ASSERT_NOTNULL(workspaces_size);
  wk_size = *(workspaces_size);
  run_info.SetTilingKey(tiling_context->GetTilingKey());
  aic_num = tiling_context->GetBlockDim();
  run_info.SetAicpuBlockDim(tiling_context->GetAicpuBlockDim());
  run_info.SetClearAtomic(tiling_context->NeedAtomic());
  auto tiling_data = tiling_context->GetRawTilingData();
  run_info.AddTilingData(reinterpret_cast<ge::char_t *>(tiling_data->GetData()), tiling_data->GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleAutofuseTilingCallback(gert::KernelContext *kernel_context, OpRunInfoV2 &run_info,
                                                    size_t &wk_size, uint32_t &aiv_num, const ge::OpDescPtr &op_desc) {
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(kernel_context);
  GE_ASSERT_NOTNULL(tiling_context);
  size_t *workspaces_size = tiling_context->GetWorkspaceSizes(1);
  GE_ASSERT_NOTNULL(workspaces_size);
  GELOGI("Get autofuse matmul op(%s) cube wss: %zu, vector wss: %zu, total workspaces_size: %zu",
         op_desc->GetName().c_str(), wk_size, *workspaces_size, wk_size + *(workspaces_size));
  wk_size += *(workspaces_size);
  *(workspaces_size) = wk_size;
  auto ws = kernel_context->GetOutputPointer<gert::ContinuousVector>(gert::TilingContext::kOutputWorkspace);
  GE_ASSERT_NOTNULL(ws);
  std::vector<int64_t> workspaces(
      reinterpret_cast<const size_t *>(ws->GetData()),
      ge::PtrAdd(reinterpret_cast<const size_t *>(ws->GetData()), std::numeric_limits<size_t>::max(), ws->GetSize()));
  run_info.SetWorkspaces(workspaces);
  run_info.SetTilingCond(tiling_context->GetTilingCond());
  run_info.SetScheduleMode(tiling_context->GetScheduleMode());
  run_info.SetLocalMemorySize(tiling_context->GetLocalMemorySize());
  aiv_num = tiling_context->GetBlockDim();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AutofuseNodeWithMatmulTiling(const ge::Operator &op, const fe::PlatFormInfos &ge_platform_infos,
                                             OpRunInfoV2 &run_info, ge::ConstNodePtr node) {
  ge::Operator matmul_op = ge::OpDescUtils::CreateOperatorFromNode(node);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(matmul_op);
  const auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()));
  GE_ASSERT_NOTNULL(space_registry);
  size_t wk_size = 0U;
  uint32_t aic_num = 0U;
  uint32_t aiv_num = 0U;
  const auto callback1 = [&run_info, &wk_size, &aic_num](gert::KernelContext *kernel_context) -> ge::graphStatus {
    return HandleMatmulTilingCallback(kernel_context, run_info, wk_size, aic_num);
  };
  (void)RtParseAndTiling(matmul_op, kCompileInfoTmpName.c_str(), ge_platform_infos, callback1, space_registry);
  const auto callback2 = [&run_info, &wk_size, &aiv_num,
                          &op_desc](gert::KernelContext *kernel_context) -> ge::graphStatus {
    return HandleAutofuseTilingCallback(kernel_context, run_info, wk_size, aiv_num, op_desc);
  };
  (void)AutofuseNodeTiling(op, ge_platform_infos, callback2);
  uint32_t new_block_dim = aic_num * 2 < aiv_num ? (aiv_num + 1) / 2 : aic_num;
  GELOGI("Get autofuse matmul op(%s) tiling key: %llu aic_num: %u, aiv_num: %u, fuse_op_block_dim: %u",
         op_desc->GetName().c_str(), run_info.GetTilingKey(), aic_num, aiv_num, new_block_dim);
  GE_ASSERT_TRUE(new_block_dim > 0U);
  run_info.SetBlockDim(new_block_dim);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicoreRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info) {
  const auto callback = [&run_info](gert::KernelContext *kernel_context) -> ge::graphStatus {
    return ConvertFromContextToRunInfo(kernel_context, run_info);
  };
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (ge::OpTypeUtils::IsAutofuseNode(op_desc)) {
    fe::PlatFormInfos ge_platform_infos;
    fe::OptionalInfos opti_compilation_info;
    GE_ASSERT_GRAPH_SUCCESS(fe::PlatformInfoManager::GeInstance()
                                .GetPlatformInfoWithOutSocVersion(ge_platform_infos, opti_compilation_info));
    ge::ComputeGraphPtr matmul_subgraph = nullptr;
    matmul_subgraph = op_desc->TryGetExtAttr(kMatMulSubgraphName, matmul_subgraph);
    if (matmul_subgraph != nullptr) {
      GELOGI("Get autofuse matmul node: %s", op_desc->GetName().c_str());
      for (auto &node : matmul_subgraph->GetAllNodes()) {
        if ((node->GetType() == kMatMulNodeType.c_str()) || (node->GetType() == kMatMulBatchMatmulNodeType.c_str())) {
          GE_ASSERT_GRAPH_SUCCESS(AutofuseNodeWithMatmulTiling(op, ge_platform_infos, run_info, node));
          return ge::GRAPH_SUCCESS;
        }
      }
      GELOGI("Get autofuse matmul node end: %s", op_desc->GetName().c_str());
    }
    return AutofuseNodeTiling(op, ge_platform_infos, callback);
  }
  return NormalAicoreNodeTiling(op, platform_infos, callback);
}

// for soft sync op
ge::graphStatus SoftSyncOpRtParseAndTiling(const ge::Operator &op, fe::PlatFormInfos &platform_infos,
                                           OpRunInfoV2 &run_info,
                                           const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL(op_desc);
  bool is_soft_sync = false;
  if (!ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync) || !is_soft_sync) {
    return ge::GRAPH_SUCCESS;
  }
  GELOGD("Call tiling for soft sync op: %s, type: %s.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  UpdateCoreNumByCoreType(op_desc, platform_infos);
  const std::string * const op_compile_info_json = ge::AttrUtils::GetStr(op_desc, kCompileInfoJson);
  GE_CHECK_NOTNULL(op_compile_info_json);

  const auto callback = [&run_info](gert::KernelContext *kernel_context)->ge::graphStatus {
    return ConvertFromContextToRunInfo(kernel_context, run_info);
  };

  const auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  GE_CHECK_NOTNULL(node);
  const auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  GE_ASSERT_GRAPH_SUCCESS(ge::RecoverIrDefinitions(graph));

  return RtParseAndTiling(op, op_compile_info_json->c_str(), platform_infos, callback, space_registry);
}

ge::graphStatus AtomicRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                       OpRunInfoV2 &run_info) {
  // build dynamic atomic node
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  if (tmp_graph == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto origin_node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  const auto node = const_cast<ge::Node *>(origin_node.get())->shared_from_this();
  std::vector<int64_t> output_clean_size;
  ge::NodePtr atomic_clean_node = nullptr;
  ge::AscendString origin_op_type;
  GE_ASSERT_GRAPH_SUCCESS(op.GetOpType(origin_op_type));
  std::string attr_name;
  if (origin_op_type == kMemSetOpType.c_str()) {
    // 这里传入的node不是元算子，是memset节点本身
    atomic_clean_node = BuildMemsetNode(node, output_clean_size, tmp_graph);
    attr_name = optiling::COMPILE_INFO_JSON;
  } else {
    // 这里传入的node是元算子
    atomic_clean_node = BuildAtomicNode(node, output_clean_size, tmp_graph);
    attr_name = optiling::ATOMIC_COMPILE_INFO_JSON;
  }
  GE_ASSERT_NOTNULL(atomic_clean_node);
  // parse compile info
  std::string op_compile_info_json;
  const auto atomic_op_desc = atomic_clean_node->GetOpDesc();
  GE_ASSERT_TRUE(
      ge::AttrUtils::GetStr(origin_node->GetOpDesc(), attr_name, op_compile_info_json),
      "Op[%s] does not have attr[%s].", origin_node->GetName().c_str(), attr_name.c_str());
  GE_ASSERT_SUCCESS(AssembleAtomicCompileInfoJson(atomic_op_desc, op_compile_info_json));
  GE_ASSERT_TRUE(ge::AttrUtils::SetStr(atomic_op_desc, attr_name, op_compile_info_json));

  const auto atomic_clean_op = ge::OpDescUtils::CreateOperatorFromNode(atomic_clean_node);
  // parse and tiling
  const auto callback = [&run_info](gert::KernelContext *kernel_context)->ge::graphStatus {
    return ConvertFromContextToRunInfo(kernel_context, run_info);
  };
  return AtomicCleanRtParseAndTiling(node, atomic_clean_op, platform_infos, output_clean_size, callback);
}

ge::graphStatus FftsRtParseAndTiling(const ge::Operator &op, const fe::PlatFormInfos &platform_infos,
                                     std::vector<OpRunInfoV2> &op_run_infos) {
  const auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GELOGD("[OpFftsPlusCalculate]Op_type:%s, op_name:%s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  ffts::ThreadSliceMapDyPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GE_CHECK_NOTNULL(slice_info_ptr);
  if (slice_info_ptr->slice_instance_num != slice_info_ptr->input_tensor_slice.size() ||
      slice_info_ptr->slice_instance_num != slice_info_ptr->output_tensor_slice.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Slice num not equal.");
    return ge::GRAPH_FAILED;
  }
  vector<int64_t> ori_shape; // save original shape
  uint32_t thread_id = 0U;
  op_run_infos.resize(ffts::kSgtTillingNum);
  bool same_shape = true;
  for (size_t i = 0U; i < static_cast<size_t>(ffts::kSgtTillingNum); i++) {
    // update node shape by thread slice info
    if (UpdateNodeShapeBySliceInfo(slice_info_ptr, op_desc, thread_id, ori_shape, same_shape) == ge::GRAPH_FAILED) {
      REPORT_INNER_ERR_MSG("E19999", "Update shape failed.");
      return ge::GRAPH_FAILED;
    }
    // call original interface
    const ge::graphStatus rc = AicoreRtParseAndTiling(op, platform_infos, op_run_infos[i]);
    if (rc != ge::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "OpParaCalculateV2 failed, op_type:%s, op_name:%s", op_desc->GetType().c_str(),
                        op_desc->GetName().c_str());
      return rc;
    }
    if (same_shape) {
      op_run_infos[1] = op_run_infos[0];
      break;
    }
    thread_id = slice_info_ptr->slice_instance_num - 1U;
  }
  // node shape write_back
  (void)UpdateNodeShapeBack(op_desc, slice_info_ptr, ori_shape);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetDeterministicLevel(int32_t &deterministic_level) {
  std::string deterministic_level_str;
  (void)ge::GetThreadLocalContext().GetOption("ge.deterministicLevel", deterministic_level_str);
  deterministic_level_str = deterministic_level_str.empty() ? "0" : deterministic_level_str;
  auto ret = ge::ConvertToInt32(deterministic_level_str, deterministic_level);
  if (ret != ge::SUCCESS || deterministic_level < 0 || deterministic_level > 2) {
    std::string readable_name = ge::GEThreadLocalContext().GetReadableName("ge.deterministicLevel");
    std::string error_msg =
        "Valid values for " + readable_name + " are {0,1,2}.";
    GELOGE(ge::FAILED, "Valid values for %s are {0,1,2}, given value is %s", readable_name.c_str(),
           deterministic_level_str.c_str());
    (void) REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                                     std::vector<const char_t *>(
                                         {
                                           readable_name.c_str(), deterministic_level_str.c_str(),
                                               error_msg.c_str()
                                         }));
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
