/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/graph_manager.h"

#include <pthread.h>
#include <future>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <openssl/sha.h>
#include "common/checker.h"
#include "common/math/math_util.h"
#include "common/memory/mem_type_utils.h"
#include "common/thread_pool.h"
#include "common/dump/dump_manager.h"
#include "common/file_constant_utils.h"
#include "common/model/external_allocator_manager.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "opt_info/ge_opt_info.h"
#include "analyzer/analyzer.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/op/transop_util.h"
#include "graph/ge_context.h"
#include "base/err_mgr.h"
#include "graph/ge_global_options.h"
#include "graph/ir_definitions_recover.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/partition/dynamic_shape_partition.h"
#include "graph/passes/control_flow_and_stream/enter_pass.h"
#include "graph/partition/stage_partitioner.h"
#include "graph/passes/feature/set_ffts_plus_attr_pass.h"
#include "graph/passes/standard_optimize/addn_pass.h"
#include "graph/passes/variable_optimize/assign_remove_pass.h"
#include "graph/passes/memory_conflict/inplace_support_check_pass.h"
#include "graph/passes/memory_conflict/atomic_addr_clean_pass.h"
#include "graph/passes/control_flow_and_stream/attach_stream_label_pass.h"
#include "graph/passes/feature/attached_resource_pass.h"
#include "graph/passes/feature/inner_tensor_move_delete_pass.h"
#include "graph/passes/format_optimize/cast_remove_pass.h"
#include "graph/passes/standard_optimize/common_subexpression_elimination_pass.h"
#include "graph/passes/feature/compile_nodes_pass.h"
#include "graph/passes/control_flow_and_stream/cond_remove_pass.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/passes/feature/constant_clip_pass.h"
#include "graph/passes/standard_optimize/constant_fuse_same_pass.h"
#include "graph/passes/control_flow_and_stream/control_trigger_pass.h"
#include "graph/passes/standard_optimize/ctrl_edge_transfer_pass.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "graph/passes/feature/data_flow_prepare_pass.h"
#include "graph/passes/control_flow_and_stream/flow_ctrl_pass.h"
#include "graph/passes/standard_optimize/fuse_data_nodes_with_common_input_pass.h"
#include "graph/passes/feature/hccl_sequence_adjust_pass.h"
#include "graph/passes/feature/hccl_tailing_optimization_pass.h"
#include "graph/passes/memory_conflict/identity_pass.h"
#include "graph/passes/feature/input_output_connection_identify_pass.h"
#include "graph/passes/feature/iterator_op_pass.h"
#include "graph/passes/feature/link_gen_mask_nodes_pass.h"
#include "graph/passes/control_flow_and_stream/merge_pass.h"
#include "graph/passes/control_flow_and_stream/merge_input_memcpy_pass.h"
#include "graph/passes/control_flow_and_stream/merge_to_stream_merge_pass.h"
#include "graph/passes/shape_optimize/merge_unknown_shape_n_pass.h"
#include "graph/passes/multi_batch/multi_batch_pass.h"
#include "graph/passes/multi_batch/subgraph_multi_dims_pass.h"
#include "graph/passes/control_flow_and_stream/next_iteration_pass.h"
#include "graph/passes/standard_optimize/permute_pass.h"
#include "graph/passes/standard_optimize/prune_pass.h"
#include "graph/passes/variable_optimize/ref_identity_delete_op_pass.h"
#include "graph/passes/standard_optimize/remove_same_const_pass.h"
#include "graph/passes/shape_optimize/reshape_recovery_pass.h"
#include "graph/passes/shape_optimize/reshape_remove_pass.h"
#include "graph/passes/standard_optimize/same_transdata_breadth_fusion_pass.h"
#include "graph/passes/memory_conflict/subgraph_pass.h"
#include "graph/passes/control_flow_and_stream/switch_data_edges_bypass.h"
#include "graph/passes/control_flow_and_stream/switch_dead_branch_elimination.h"
#include "graph/passes/control_flow_and_stream/switch_logic_remove_pass.h"
#include "graph/passes/control_flow_and_stream/switch_to_stream_switch_pass.h"
#include "graph/passes/standard_optimize/start_of_sequence_pass.h"
#include "graph/passes/format_optimize/transop_breadth_fusion_pass.h"
#include "graph/passes/format_optimize/transop_nearby_allreduce_fusion_pass.h"
#include "graph/passes/format_optimize/transop_symmetry_elimination_pass.h"
#include "graph/passes/format_optimize/transop_without_reshape_fusion_pass.h"
#include "graph/passes/format_optimize/transpose_transdata_pass.h"
#include "graph/passes/standard_optimize/useless_control_out_remove_pass.h"
#include "graph/passes/variable_optimize/variable_op_pass.h"
#include "graph/passes/variable_optimize/variable_ref_delete_op_pass.h"
#include "graph/passes/variable_optimize/variable_ref_useless_control_out_delete_pass.h"
#include "graph/passes/standard_optimize/end_of_sequence_add_control_pass.h"
#include "graph/passes/multi_batch/subexpression_migration_pass.h"
#include "graph/passes/multi_batch/subgraph_const_migration_pass.h"
#include "graph/passes/standard_optimize/unused_args_clean_pass.h"
#include "graph/passes/feature/global_step_insert_pass.h"
#include "graph/passes/memory_conflict/memcpy_addr_async_pass.h"
#include "graph/passes/memory_conflict/hccl_continuous_memcpy_pass.h"
#include "graph/passes/standard_optimize/constant_folding/replace_with_empty_const_pass.h"
#include "graph/passes/feature/parallel_group_pass.h"
#include "graph/passes/feature/buffer_pool_memory_pass.h"
#include "graph/passes/memory_optimize/swap_space_pass.h"
#include "graph/passes/feature/recompute_pass.h"
#include "graph/passes/memory_conflict/hccl_memcpy_pass.h"
#include "graph/fusion/pass/fusion_pass_executor.h"
#include "graph/build/label_allocator.h"
#include "graph/build/graph_compile_summary_impl.h"
#include "graph/build/stream/stream_utils.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "graph/passes/pass_manager.h"
#include "api/aclgrph/option_utils.h"
#include "common/util/trace_manager/trace_manager.h"
#include "common/helper/file_saver.h"
#include "common/helper/model_save_helper.h"
#include "common/memory/tensor_trans_utils.h"
#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "framework/executor/ge_executor.h"
#include "graph/utils/math_util.h"
#include "graph/build/model_cache.h"
#include "generator/ge_generator.h"
#include "graph/passes/memory_optimize/concat_notask_pass.h"
#include "graph/passes/format_optimize/dim1_transpose_to_squeeze_pass.h"
#include "graph/optimize/autofuse/autofuse_optimize.h"
#include "graph/passes/standard_optimize/tensor_move_delete_pass.h"
#include "acl/acl_rt.h"

namespace ge {
namespace {
const char *const kSummary = "Summary";
const char *const kSave = "Save";
const char *const kNetOutput = "NetOutput";
const char *const kVariable = "Variable";
const char *const kSend = "Send";
const char *const kRecv = "Recv";
const char *const kCheckPointForGetVar = "CheckPointGraphForGetVar";
const char *const kCheckPointGraph = "checkpoint_graph";
const char *const kVectorEngine = "VectorEngine";
const char *const kAIcoreEngine = "AIcoreEngine";
const char *const kVectorCore = "VectorCore";
const int32_t kDynamicDimsTypeIsGetNext = 0;
const int32_t kDynamicDimsTypeIsData = 1;
const int32_t kBase = 10;
const int32_t kInvalidDeviceId = -1;
const char *const kGetNextName = "IteratorV2";
const uint32_t kInitGraphCount = 1;
const uint32_t kNotAdded = 0;
const uint32_t kStartAdd = 1;
const uint32_t kDoneAdded = 2;
const std::string kIntegerEnableOption = "1";
const std::string kBoolDisableOption = "False";
constexpr uint32_t kDefaultThreadNum = 16U;
constexpr char const *kUbOriginGraphAttrKey = "_original_fusion_graph";
const std::unordered_set<std::string> kDataOpTypes { ge::DATA, ge::AIPPDATA };
const std::string kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";
const char *const kHintInputShape = "ge.inputHintShape";
const char *const kShapeDataName = "ascend_mbatch_shape_data";
const char *const kSubstrOfGetNextNosinkName = "IteratorGetNext";
bool IsTailingOptimization() {
  std::string is_tailing_optimization_option;
  auto ret = ge::GetContext().GetOption(ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION, is_tailing_optimization_option);
  if (ret == ge::GRAPH_SUCCESS) {
    GELOGI("Option ge.exec.isTailingOptimization is %s", is_tailing_optimization_option.c_str());
    // "1" means it's True from frontend option
    return is_tailing_optimization_option == "1";
  }
  GELOGW("OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION not set, use BFSTopologicalSorting by default.");
  return false;
}

ge::graphStatus GetGraphMaxParallelModeNum(int32_t &max_parallel_num) {
  std::string opt = "0";
  (void)ge::GetContext().GetOption(ge::GRAPH_MAX_PARALLEL_MODEL_NUM, opt);
  GE_ASSERT_SUCCESS(ge::ConvertToInt32(opt, max_parallel_num), "option %s, value %s is not int",
                    ge::GetContext().GetReadableName(ge::GRAPH_MAX_PARALLEL_MODEL_NUM).c_str(), opt.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::Status CheckFpCeilingMode() {
  static const std::set<std::string> kValidFpCeilingMode = {"0", "1", "2"};
  std::string mode;
  auto ret = ge::GetContext().GetOption("ge.fpCeilingMode", mode);
  if (ret == ge::GRAPH_SUCCESS) {
    if (kValidFpCeilingMode.count(mode) == 0) {
      const auto readable_name = ge::GetContext().GetReadableName("ge.fpCeilingMode");
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10061", std::vector<const char *>({"value", "parameter", "expected_value"}),
          std::vector<const char *>({mode.c_str(), readable_name.c_str(), "0, 1, or 2"}));
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Get][Option] The fp_ceiling_mode %s is invalid, options are 0, 1, and 2.",
             mode.c_str());
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
    GELOGI("The parameter fp_ceiling_mode is set to %s.", mode.c_str());
    return ge::SUCCESS;
  }
  GELOGW("The parameter fp_ceiling_mode is not set");
  return ge::SUCCESS;
}

ge::Status ModifyAippData(const ge::ComputeGraphPtr &compute_graph) {
  // modify aipp data index, the AIPPDATA node inserted at AippOp::InsertAippToGraph has no attribute name "index"
  uint32_t data_op_index = 0U;
  for (const ge::NodePtr &n : compute_graph->GetDirectNode()) {
    if (kDataOpTypes.count(n->GetType()) == 0U) {
      continue;
    }
    if (n->GetType() == ge::AIPPDATA) {
      (void)ge::AttrUtils::SetInt(n->GetOpDesc(), ge::ATTR_NAME_INDEX, data_op_index);
    }
    ++data_op_index;
  }
  return ge::SUCCESS;
}
ge::Status NormalizeGeTensorOnComputeGraph(const ge::ComputeGraphPtr &compute_graph) {
  for (const auto &node : compute_graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    for (auto &input_desc : op_desc->GetAllInputsDescPtr()) {
      GE_ASSERT_NOTNULL(input_desc);
      ge::TensorAdapter::NormalizeGeTensorDesc(*input_desc);
    }
    for (auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      ge::TensorAdapter::NormalizeGeTensorDesc(*output_desc);
    }
  }
  return ge::SUCCESS;
}

bool IsMemoryAndTypeSupport(const ge::GraphNodePtr &graph_node, const ge::MemoryType type, const void *const memory,
    const size_t size, std::string &reason) {
  const auto graph_id = graph_node->GetGraphId();
  const bool address_null = (memory == nullptr) && (size == 0UL);
  if ((type == ge::MemoryType::MEMORY_TYPE_P2P) && address_null) {
    reason = "[Check][Memory] When the fixed memory type is MEMORY_TYPE_P2P, default behavior cannot be turned off,"
             " graph_id:" + std::to_string(graph_id);
    return false;
  }

  const auto queryed = graph_node->GetFeatureMemoryBase();
  if ((queryed.first != nullptr) && (type == ge::MemoryType::MEMORY_TYPE_DEFAULT)) {
    reason = "[Check][Memory] UpdateGraphFeatureMemoryBase has already been called, and SetGraphFixedFeatureMemoryBase"
             " can not be called or SetGraphFixedFeatureMemoryBaseWithType can be called using the"
             " MEMORY_TYPEDEFAULT parameter, please refer to the guide, graph_id:" + std::to_string(graph_id);
    return false;
  }

  // 开启了动静态图复用，由于fixed feature memory不支持地址刷新，所以必须申请fixed优先内存，因此不能关闭默认行为
  if (address_null && ge::VarManager::IsGeUseExtendSizeMemoryFull()) {
    reason = "option ge.exec.staticMemoryPolicy or env GE_USE_STATIC_MEMORY is 4, the memory reuse for unknown and"
        " known shape graph has been enabled. As fixed feature memory does not support address change, it is necessary"
        " to malloc fixed prior memory, so the default behavior cannot be turned off, graph_id:"
        + std::to_string(graph_id);
    return false;
  }
  return true;
}

Status GetValidGraphNodeForBuild(const GraphManager *const graph_manager, const GraphId &graph_id,
                                 GraphNodePtr &graph_node) {
  Status ret = graph_manager->GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }

  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] graph node is NULL, graphId = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph_id:%u, "
                       "check invalid", graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] graph already running, graph id = %u", graph_node->GetGraphId());
    return GE_GRAPH_ALREADY_RUNNING;
  }
  return SUCCESS;
}

bool IsGelocalOp(const OpDescPtr &op_desc) {
  return op_desc->GetOpKernelLibName() == kGeLocalOpKernelLibName;
}

void PrintCommunicationNodes(const uint32_t graph_id, const std::string &graph_name,
    const std::string &stage, const std::map<std::string, std::vector<std::string>> &group_2_comm_nodes) {
  if (group_2_comm_nodes.empty()) {
    return;
  }
  std::stringstream ss;
  ss << stage << ", record graph[" << graph_name << "], graph_id[" << graph_id <<"] communication nodes, ";
  for (const auto &it : group_2_comm_nodes) {
    ss << "group[" << it.first << "]: ";
    for (const auto &op_name : it.second) {
      ss << op_name << ", ";
    }
  }
  ss << "end.";
  // in case of being truncated out of log limit 1024, set up limit 800
  const size_t max_log_string_len = 800U;
  size_t index = 0U;
  while (index < ss.str().length()) {
    GEEVENT("%s", ss.str().substr(index, max_log_string_len).c_str());
    index += max_log_string_len;
  }
}

Status GetCommunicationNodes(const ComputeGraphPtr &compute_graph, bool is_engine_assigned,
                             std::map<std::string, std::vector<std::string>> &comm_nodes) {
  for (const auto &node : compute_graph->GetAllNodesPtr()) {
    GE_ASSERT_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    // only support hccl engine node currently after engine assigned
    if (is_engine_assigned && (op_desc->GetOpKernelLibName() != ge::kEngineNameHccl)) {
      continue;
    }
    std::string group_attr;
    if (AttrUtils::GetStr(op_desc, "group", group_attr)) {
      comm_nodes[group_attr].emplace_back(op_desc->GetName());
    }
  }
  return SUCCESS;
}

Status VerifyCommNodesOrder(const std::map<std::string, std::vector<std::string>> &base_nodes,
                            const std::map<std::string, std::vector<std::string>> &compare_nodes) {
  for (const auto &base_ele : base_nodes) {
    const auto group = base_ele.first;
    const auto &base_nodes_vec = base_ele.second;
    auto it = compare_nodes.find(group);
    if (it == compare_nodes.end()) {
      continue;
    }
    const auto &compare_nodes_vec = it->second;
    std::map<std::string, size_t> compare_nodes_map;
    // use tmp map record compare nodes name and position
    for (size_t i = 0U; i < compare_nodes_vec.size(); ++i) {
      compare_nodes_map[compare_nodes_vec[i]] = i;
    }
    for (size_t i = 0U; (i < base_nodes_vec.size() && ((i + 1) < base_nodes_vec.size())); ++i) {
      const auto &left_node = base_nodes_vec[i];
      const auto &right_node = base_nodes_vec[i + 1];
      const auto it_left = compare_nodes_map.find(left_node);
      const auto it_right = compare_nodes_map.find(right_node);
      if ((it_left != compare_nodes_map.end()) && (it_right != compare_nodes_map.end())) {
        if (it_left->second > it_right->second) {
          GELOGW("In group %s, communication op %s[%zu] and communication op %s[%zu] is verify unordered, "
                 "please check if ge.topoSortingMode option is set StableRDFS",
                 group.c_str(), left_node.c_str(), it_left->second, right_node.c_str(), it_right->second);
        }
      }
    }
  }
  return SUCCESS;
}

std::vector<int64_t> GetDimsFromGertShape(const gert::Shape &gert_shape) {
  std::vector<int64_t> dims(gert_shape.GetDimNum());
  for (size_t i = 0U; i < gert_shape.GetDimNum(); ++i) {
    dims[i] = gert_shape.GetDim(i);
  }
  return dims;
}

TensorDesc GetTensorDescFromGertTensor(const gert::Tensor &gert_tensor) {
  ge::Shape storage_shape{GetDimsFromGertShape(gert_tensor.GetStorageShape())};
  TensorDesc tensor_desc{std::move(storage_shape), gert_tensor.GetStorageFormat(), gert_tensor.GetDataType()};
  const ge::Shape origin_shape{GetDimsFromGertShape(gert_tensor.GetOriginShape())};

  tensor_desc.SetOriginFormat(gert_tensor.GetOriginFormat());
  tensor_desc.SetOriginShape(origin_shape);
  return tensor_desc;
}

Status SaveRootModel(const GeRootModelPtr &ge_root_model, ModelBufferData &model_buff) {
  GeGenerator::SetModelNameForDump(ge_root_model);
  bool is_unknown_shape = false;
  GE_ASSERT_SUCCESS(ge_root_model->CheckIsUnknownShape(is_unknown_shape),
                    "root model(id:%u) CheckIsUnknownShape failed", ge_root_model->GetModelId());
  GELOGD("begin save root model, cur model is %s", (is_unknown_shape ? "unknown shape model" : "known shape model"));
  GE_CHK_BOOL_EXEC(!ge_root_model->GetSubgraphInstanceNameToModel().empty(),
                   REPORT_INNER_ERR_MSG("E19999", "root model(id:%u) has no sub model.", ge_root_model->GetModelId());
                   return FAILED, "[Get][SubModel] ge root model has no sub model");
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    auto name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
    model_root = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  } else {
    model_root = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  }
  GE_CHECK_NOTNULL(model_root);

  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  GE_CHECK_NOTNULL(model_save_helper);
  model_save_helper->SetSaveMode(false);
  GE_ASSERT_SUCCESS(model_save_helper->SaveToOmRootModel(ge_root_model, ge_root_model->GetModelName(),
    model_buff, is_unknown_shape),
                    "SaveToOmRootModel failed, model id:%u", ge_root_model->GetModelId());
  return SUCCESS;
}

Status OptimizeTensorMove(ge::ComputeGraphPtr &compute_graph) {
  NamesToPass names_to_passes;
  TensorMoveDeletePass tensor_move_delete_pass;

  names_to_passes.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  GE_TRACE_START(names_to_passes);
  auto ret = GEPass(compute_graph).Run(names_to_passes);
  GE_COMPILE_TRACE_TIMESTAMP_END(names_to_passes, "TensorMoveDeletePass");
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][GEPasses] optimize for tensor move failed, ret:%d.", ret);
    return ret;
  }

  return SUCCESS;
}

}  // namespace
GraphManager::~GraphManager() {
  // set threal local omg_contex to defaut, to avoid other use invalid memory
  SetLocalOmgContext(domi::GetContext());
}

Status GraphManager::Initialize(const std::map<std::string, std::string> &options, Executor *executor) {
  if (init_flag_) {
    GELOGW("[Initialize] GraphManager already initialized.");
    return SUCCESS;
  }

  // graph context
  graph_context_ = MakeShared<GraphContext>();
  if (graph_context_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New GraphContext fail");
    GELOGE(MEMALLOC_FAILED, "[New][GraphContext] failed.");
    return MEMALLOC_FAILED;
  }
  graph_rebuild_state_ctrl_ = MakeShared<GraphRebuildStateCtrl>();
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  // parse option parameters
  Status ret = ParseOptions(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Options] failed.");
    return ret;
  }

  ret = CheckFpCeilingMode();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Check][FpCeilingMode] failed.");
    return ret;
  }

  ret = graph_context_->Initialize(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Initialize][GraphContext] failed.");
    return ret;
  }

  executor_ = executor;
  init_flag_ = true;
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);

  thread_run_flag_ = true;
  prerun_thread_v2_ = std::thread(&GraphManager::PreRunThreadV2, this);
  return SUCCESS;
}

void GraphManager::SetExternalGraphRebuildStateCtrl(std::shared_ptr<GraphRebuildStateCtrl> &rebuild_ctrl) {
  if (rebuild_ctrl != nullptr) {
    GELOGI("graph rebuild state ctrl is set from external");
    graph_rebuild_state_ctrl_ = rebuild_ctrl;
  }
}

void GraphManager::UpdateDynamicParams(std::string &input_shape, std::string &dynamic_dims,
                                       int32_t &dynamic_node_type,
                                       const std::map<std::string, std::string> &graph_options) const {
  dynamic_node_type = -1;
  ParseOption(graph_options, INPUT_SHAPE, input_shape);
  ParseOption(graph_options, kDynamicDims, dynamic_dims);
  ParseOption(graph_options, DYNAMIC_NODE_TYPE, dynamic_node_type);
  input_shape = input_shape.empty() ? options_.input_shape : input_shape;
  dynamic_dims = dynamic_dims.empty() ? options_.dynamic_dims : dynamic_dims;
  dynamic_node_type = dynamic_node_type == -1 ? options_.dynamic_node_type : dynamic_node_type;
}

Status GraphManager::UnloadModel(GeRootModelPtr ge_root_model, uint32_t graph_id) {
  GE_CHECK_NOTNULL(executor_);
  return executor_->UnloadGraph(ge_root_model, graph_id);
}

Status GraphManager::Finalize() {
  if (!init_flag_) {
    GELOGW("GraphManager has not been initialized.");
    return SUCCESS;
  }

  StopQueue();
  if (prerun_thread_v2_.joinable()) {
    prerun_thread_v2_.join();
  }
  // check graph whether running or not
  Status unload_model_ret = SUCCESS;
  for (auto iter = graph_map_.cbegin(); iter != graph_map_.cend(); ++iter) {
    GraphNodePtr graph_node = iter->second;
    GE_CHECK_NOTNULL(graph_node);
    GE_CHECK_NOTNULL(graph_node->GetGraph());
    if (graph_node->GetRunFlag()) {
      GELOGW("[GraphManager] finalize failed, graphId=%u.", iter->first);
      unload_model_ret = GE_GRAPH_GRAPH_IS_RUNNING;
      continue;
    }
    // unload model
    auto ge_root_model = graph_node->GetGeRootModel();
    if (CheckModelLoad(ge_root_model, graph_node->GetLoadFlag())) {
      Status ret = UnloadModel(ge_root_model, iter->first);
      if (ret != SUCCESS) {
        unload_model_ret = ret;
        GELOGW("[GraphManager] unload model failed, graph_id=%u.", iter->first);
      }
    }

    // clear analyzer saved info(graph level)
    auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph_node->GetGraph());
    GE_CHECK_NOTNULL(compute_graph);
    auto session_id = compute_graph->GetSessionID();
    auto graph_id = compute_graph->GetGraphID();
    Analyzer::GetInstance()->DestroyGraphJsonObject(session_id, graph_id);

    CompilerStages &stages = GetCompilerStages(graph_id);
    Status res = stages.optimizer.FinalizeSessionInfo(compute_graph);
    if (res != SUCCESS) {
      GELOGE(res, "[Finalize][GraphManager] failed, graph name=%s", compute_graph->GetName().c_str());
      return res;
    }
  }
  graph_map_.clear();
  graph_ids_.clear();
  graph_count_.clear();

  // graph context
  if (graph_context_ != nullptr) {
    Status ret_final = graph_context_->Finalize();
    if (ret_final != SUCCESS) {
      GELOGE(ret_final, "[Finalize][GraphContext] failed!");
      unload_model_ret = ret_final;
    }
  }
  resource_context_mgr_.ClearContext();

  init_flag_ = false;
  return unload_model_ret;
}

Status GraphManager::InitDynamicParams(const ComputeGraphPtr &compute_graph,
                                       const std::map<std::string, std::string> &graph_options) const {
  GE_CHECK_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    GetLocalOmgContext().need_multi_batch = false;
    std::string op_type;
    auto ret = GetOriginalType(node, op_type);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "GetOriginalType from op:%s fail", node->GetName().c_str());
      GELOGE(FAILED, "[Get][OriginalType] from op:%s failed.", node->GetName().c_str());
      return FAILED;
    }
    if (OpTypeUtils::IsDataNode(op_desc->GetType()) || (op_type == kGetNextName)) {
      GELOGI("Maybe need to process multi batch for compute graph %s, op_type:%s.",
             compute_graph->GetName().c_str(), op_desc->GetType().c_str());
      GetLocalOmgContext().need_multi_batch = true;
      break;
    }
  }
  std::string input_shape;
  std::string dynamic_dims;
  int32_t dynamic_node_type = -1;
  UpdateDynamicParams(input_shape, dynamic_dims, dynamic_node_type, graph_options);
  if (!input_shape.empty() && !dynamic_dims.empty()) {
    if (!ge::ParseInputShape(input_shape, GetLocalOmgContext().input_dims,
                             GetLocalOmgContext().user_input_dims, true)) {
      GELOGE(GRAPH_PARAM_INVALID, "[Parse][InputShape] %s failed.", input_shape.c_str());
      return GRAPH_PARAM_INVALID;
    }
    GetLocalOmgContext().dynamic_dims = dynamic_dims;
  }
  if (dynamic_node_type == kDynamicDimsTypeIsGetNext) {
    GetLocalOmgContext().dynamic_node_type = GETNEXT;
  }
  if (dynamic_node_type == kDynamicDimsTypeIsData) {
    GetLocalOmgContext().dynamic_node_type = DATA;
  }
  return SUCCESS;
}

void GraphManager::SetAddGraphCondition(GraphId graph_id, uint32_t cond) {
  std::lock_guard<std::mutex> lock(add_graph_cond_mutex_);
  graph_id_to_add_graph_cond_[graph_id] = cond;
  GELOGD("Graph [id:%u] has been added.", graph_id);
}

uint32_t GraphManager::GetAddGraphCondition(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(add_graph_cond_mutex_);
  std::map<GraphId, uint32_t>::const_iterator it = graph_id_to_add_graph_cond_.find(graph_id);
  if (it != graph_id_to_add_graph_cond_.cend()) {
    return it->second;
  } else {
    GELOGD("Graph [id:%u] has not been added.", graph_id);
    return kNotAdded;
  }
}

void GraphManager::RemoveAddGraphCondition(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(add_graph_cond_mutex_);
  std::map<GraphId, uint32_t>::const_iterator it = graph_id_to_add_graph_cond_.find(graph_id);
  if (it != graph_id_to_add_graph_cond_.cend()) {
    (void)graph_id_to_add_graph_cond_.erase(it);
    GELOGD("Successfully remove add_graph_cond of graph [id:%u].", graph_id);
  } else {
    GELOGD("Graph [id:%u] has not been added, no need to be removed.", graph_id);
  }
}

Status GraphManager::CheckRepeatAdd(uint32_t graph_id, bool &is_added) {
  uint32_t count = 0;
  if (GetGraphCount(graph_id, count) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][GraphCount] failed, graph[id:%u] might have not been added.", graph_id);
    return INTERNAL_ERROR;
  }
  // previous thread owns same graph_id has been in the middle of the AddGraph procession
  if (count > 1 && GetAddGraphCondition(graph_id) == kStartAdd) {
    std::unique_lock<std::mutex> lock(add_graph_mutex_);
    GELOGD("Waitting for build end of previous thread.");
    while (GetAddGraphCondition(graph_id) != kDoneAdded) {
      add_graph_cv_.wait(lock);
    }
    GraphNodePtr graph_node;
    Status ret = GetGraphNode(graph_id, graph_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][GraphNode] failed, graph_id = %u.", graph_id);
      return ret;
    }
    is_added = true;
  }
  return SUCCESS;
}

void GraphManager::SetSessionGraphId(ComputeGraphPtr compute_graph, uint32_t graph_id) const {
  GE_CHECK_NOTNULL_EXEC(compute_graph, return);
  std::string session_graph_id;
  if (!AttrUtils::GetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) || session_graph_id.empty()) {
    session_graph_id = "-1_" + to_string(graph_id);
    if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      GELOGW("Set attribute of compute graph failed.");
    }
    for (auto &subgraph : compute_graph->GetAllSubgraphs()) {
      GE_CHECK_NOTNULL_EXEC(subgraph, return);
      (void)AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    }
    GELOGD("Get graph session_graph_id attr failed, set session id to default value: [0]");
  }
}

Status GraphManager::NotifyWaittingGraph(uint32_t graph_id) {
  uint32_t count = 0;
  if (GetGraphCount(graph_id, count) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][GraphCount] failed, graph[id:%u] might have not been added.", graph_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Add graph finished, graph_id:%u", graph_id);
  if (count > 1) {
    GELOGD("Finish addgraph, graph_id:%u, graph_count:%u, start to notify.", graph_id, count);
    add_graph_cv_.notify_all();
  }
  return SUCCESS;
}

Status GraphManager::CreateGraphNode(uint32_t graph_id, const Graph &graph,
                                     const std::map<std::string, std::string> &options) {
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  GE_IF_BOOL_EXEC(graph_node == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "New GraphNode fail, graph_id:%u", graph_id);
                  GELOGE(FAILED, "[New][GraphNode] fail, graph_id:%u", graph_id);
                  return FAILED);
  std::shared_ptr<Graph> graph_ptr = MakeShared<ge::Graph>(graph);
  GE_IF_BOOL_EXEC(graph_ptr == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "New Graph fail, graph_id:%u", graph_id);
                  GELOGE(FAILED, "[New][Graph] fail, graph_id:%u", graph_id);
                  return FAILED);
  // update option about tuning graph
  ParseOption(options, ir_option::INPUT_FORMAT, options_.input_format);
  ParseOption(options, BUILD_MODE, options_.build_mode);
  ParseOption(options, BUILD_STEP, options_.build_step);
  ParseOption(options, TUNING_PATH, options_.tuning_path);
  ParseOption(options, BUILD_INNER_MODEL, options_.build_inner_model);
  graph_node->SetOptions(options);
  graph_node->SetGraph(graph_ptr);
  graph_node->IncreaseLoadCount();
  AddGraphNode(graph_id, graph_node);
  return SUCCESS;
}

Status GraphManager::SetStagesOptions(uint32_t graph_id) {
  CompilerStages &stages = GetCompilerStages(graph_id);
  auto refreshed_options = options_;
  RefreshOptionByGraph(graph_id, refreshed_options);
  stages.preparer.SetOptions(refreshed_options);
  Status status = stages.optimizer.SetOptions(refreshed_options);
  if (status != SUCCESS) {
    GELOGE(status, "[Set][Options] for Graph optimizer failed, graph id:%u.", graph_id);
    return status;
  }
  stages.builder.SetOptions(refreshed_options);
  return SUCCESS;
}

Status GraphManager::ModifyDataIndex(const Graph &graph, const std::map<std::string,
                                     std::string> &graph_option) const {
  std::vector<OpDescPtr> data_desc;
  std::set<int64_t> indexes;
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  for (auto &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    auto op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (OpTypeUtils::IsDataNode(op->GetType())) {
      int64_t index = std::numeric_limits<int64_t>::max();
      (void) AttrUtils::GetInt(op, ATTR_NAME_INDEX, index);
      (void)indexes.insert(index);
      data_desc.emplace_back(op);
    }
  }
  if (!indexes.empty()) {
    auto first_iter = indexes.begin();
    auto end_iter = indexes.end();
    --end_iter;
    auto data_size = static_cast<int64_t>(data_desc.size());
    // The valid index starts with 0 and increases by 1, and num is equal to data_node.
    if (indexes.size() != data_desc.size() || *first_iter != 0 || *end_iter != data_size - 1) {
      auto iter = graph_option.find(OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE);
      if (iter != graph_option.end() && !iter->second.empty()) {
        // If data inputs shape range is set, user must set valid data index.
        std::string reason = "Data indexes must be consecutive and start from 0 to " + std::to_string(data_size - 1) +
                             " (data node size is " + std::to_string(data_size) +
                             ") when the data shape range is enabled. Current indexes size is " +
                             std::to_string(indexes.size()) + ", first index is " + std::to_string(*first_iter) +
                             ", last index is " + std::to_string(*end_iter);
        REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char *>({"reason"}),
                           std::vector<const char *>({reason.c_str()}));
        GELOGE(GRAPH_PARAM_INVALID, "[COMP][AddGraph]Input data index is invalid when data shape range enabled.");
        return GRAPH_PARAM_INVALID;
      }
      GELOGI("Graph[%s] input data index is invalid, set data index by topo order.", compute_graph->GetName().c_str());
      int64_t index = 0;
      for (auto &op : data_desc) {
        (void) AttrUtils::SetInt(op, ATTR_NAME_INDEX, index++);
      }
    }
  }
  return SUCCESS;
}

void GraphManager::WarningForDeprecatedOptions(const std::map<std::string, std::string> &options) const {
  const auto iter = options.find(OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE);
  if (iter != options.cend()) {
    GELOGW("WARNING: Option %s is deprecated and will be remove in future version, no need set",
        OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE);
  }
}

Status GraphManager::CheckOptionsValid(const ComputeGraphPtr &compute_graph,
                                       const std::map<std::string, std::string> &options) const {
  WarningForDeprecatedOptions(options);
  GE_ASSERT_SUCCESS(CheckOptionValidValues(options, OPTION_FEATURE_BASE_REFRESHABLE, kFeatureMapRefreshOptions));
  GE_ASSERT_SUCCESS(CheckOptionValidValues(options, OPTION_CONST_LIFECYCLE, kConstLifecycleOptions));
  GE_ASSERT_SUCCESS(CheckIoReuseMemIndexesOption(compute_graph, options));
  GE_ASSERT_SUCCESS(CheckOptionValidThreshold(options, OPTION_HOST_SCHEDULING_MAX_THRESHOLD));
  GE_ASSERT_SUCCESS(CheckOptionValidValues(options, TILING_SCHEDULE_OPTIMIZE, kStateOptions));
  GE_ASSERT_GRAPH_SUCCESS(CheckOptimizationOptionValid(options));
  return SUCCESS;
}

Status GraphManager::CheckGraphExisted(const GraphId &graph_id, bool &is_added) {
  IncreaseGraphCount(graph_id);
  // validation for adding graphs of same graph_id in multi-thread secenario
  // 1.previous thread owns same graph_id has finished the AddGraph procession
  if (GetAddGraphCondition(graph_id) == kDoneAdded) {
    GraphNodePtr graph_node;
    if (GetGraphNode(graph_id, graph_node) != SUCCESS) {
      GELOGE(GE_GRAPH_GRAPH_NOT_EXIST, "[Get][GraphNode] failed, Graph does not exist while done adding previously, "
             "graph_id = %u.", graph_id);
      return GE_GRAPH_GRAPH_NOT_EXIST;
    }
    is_added = true;
    GELOGD("Graph %u has been added.", graph_id);
    graph_node->IncreaseLoadCount();
    return SUCCESS;
  }
  // In multi-thread scenario, former thread owns same graph_id has been
  // in the middle of the AddGraph procession while following threads have to wait until
  // done adding graph of the former graph, avoiding repeatively adding same graph.
  is_added = false;
  if (CheckRepeatAdd(graph_id, is_added) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Check][RepeatAdd] for graph[id:%u] failed.", graph_id);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status GraphManager::AddGraph(const GraphId &graph_id, const Graph &graph,
                              const std::map<std::string, std::string> &options,
                              const OmgContext &omg_context) {
  GetThreadLocalContext().SetGraphOption(options);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  GE_ASSERT_SUCCESS(CheckOptionsValid(compute_graph, options));
  bool is_added = false;
  GE_ASSERT_SUCCESS(CheckGraphExisted(graph_id, is_added));
  if (is_added) {
    return SUCCESS;
  }
  // Do add graph
  SetAddGraphCondition(graph_id, kStartAdd);
  if (CheckGraphAdded(graph_id, graph) != SUCCESS) {
    GELOGE(FAILED, "[Check][GraphAdded] failed, graph id:%u.", graph_id);
    return FAILED;
  }
  GE_CHK_STATUS_RET(ModifyDataIndex(graph, options));
  (void)AttrUtils::SetBool(*compute_graph, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, true);
  SetSessionGraphId(compute_graph, graph_id);

  AddLocalOmgContext(graph_id, omg_context);
  // Parse the configuration for the session->addgraph.
  ParseOption(options, OUTPUT_DATATYPE, options_.output_datatype);
  if (!options_.output_datatype.empty()) {
    GetLocalOmgContext().output_type = options_.output_datatype;
  }
  if (InitDynamicParams(compute_graph, options) != SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Init][Params] failed, when online infer is dynamic, graph id:%u.", graph_id);
    return GRAPH_PARAM_INVALID;
  }

  GE_CHK_STATUS_RET_NOLOG(AddGraphForBuild(graph_id, compute_graph, options));

  SetAddGraphCondition(graph_id, kDoneAdded);
  // There are threads waitting for adding same graph
  if (NotifyWaittingGraph(graph_id) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Notify][WaittingGraph] failed, graph id:%u.", graph_id);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status GraphManager::AddGraphForBuild(const GraphId &graph_id,
                                      const ComputeGraphPtr &compute_graph,
                                      const std::map<std::string, std::string> &options,
                                      bool graph_normalized) {
  const auto &graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  if (CreateGraphNode(graph_id, {graph}, options) != SUCCESS) {
    GELOGE(FAILED, "[Create][GraphNode] failed, graph id:%u.", graph_id);
    return FAILED;
  }

  if (SetStagesOptions(graph_id) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Set][StagesOptions] failed, graph id:%u.", graph_id);
    return INTERNAL_ERROR;
  }

  if (graph_normalized) {
    GetCompilerStages(graph_id).preparer.SetGraphNormalized(true);
  }
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  graph_rebuild_state_ctrl_->AddGraph(graph_id, compute_graph);
  return SUCCESS;
}

Status GraphManager::CheckGraphAdded(const GraphId &graph_id, const Graph &graph) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph != nullptr) {
    compute_graph->SetGraphID(graph_id);
    bool graph_has_been_added = false;
    if (AttrUtils::GetBool(*compute_graph, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, graph_has_been_added)
        && graph_has_been_added) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from graph:%u fail.",
                         ATTR_NAME_GRAPH_HAS_BEEN_ADDED.c_str(), graph_id);
      GELOGE(GE_GRAPH_GRAPH_ALREADY_EXIST,  "[Get][Attr] %s from graph:%u fail.",
             ATTR_NAME_GRAPH_HAS_BEEN_ADDED.c_str(), graph_id);
      return GE_GRAPH_GRAPH_ALREADY_EXIST;
    }
  } else {
    REPORT_INNER_ERR_MSG("E19999", "compute_graph from graph:%u is nullptr, check invalid", graph_id);
    GELOGE(FAILED, "[Get][ComputeGraph] failed, compute graph from graph:%u is nullptr", graph_id);
    return FAILED;
  }
  return SUCCESS;
}

Status GraphManager::AddGraphWithCopy(const GraphId &graph_id, const Graph &graph,
                                      const std::map<std::string, std::string> &options,
                                      const OmgContext &omg_context) {
  GetThreadLocalContext().SetGraphOption(options);
  if (HasGraphNode(graph_id)) {
    GELOGE(GE_GRAPH_GRAPH_ALREADY_EXIST, "[Has][GraphNode] graph exists, graph_id = %u", graph_id);
    return GE_GRAPH_GRAPH_ALREADY_EXIST;
  }
  if (CheckGraphAdded(graph_id, graph) != SUCCESS) {
    GELOGE(FAILED, "[Check][GraphAdded] failed, graph_id = %u", graph_id);
    return FAILED;
  }
  IncreaseGraphCount(graph_id);
  // Do add graph
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::vector<NodePtr> input_nodes;
  std::vector<NodePtr> output_nodes;
  auto new_compute_graph = GraphUtils::CloneGraph(compute_graph, "", input_nodes, output_nodes);
  GE_CHECK_NOTNULL(new_compute_graph);
  new_compute_graph->SetGraphID(graph_id);
  SetSessionGraphId(new_compute_graph, graph_id);
  std::shared_ptr<Graph> new_graph_ptr = GraphUtilsEx::CreateGraphPtrFromComputeGraph(new_compute_graph);
  GE_CHECK_NOTNULL(new_graph_ptr);
  if (CreateGraphNode(graph_id, {*new_graph_ptr}, options) != SUCCESS) {
    GELOGE(FAILED, "[Create][GraphNode] failed, graph_id = %u", graph_id);
    return FAILED;
  }

  AddLocalOmgContext(graph_id, omg_context);
  if (!options_.output_datatype.empty()) {
    GetLocalOmgContext().output_type = options_.output_datatype;
  }
  if (InitDynamicParams(new_compute_graph, options) != SUCCESS) {
    GELOGE(GRAPH_PARAM_INVALID, "[Init][Params] failed, when online infer is dynamic, graph_id = %u", graph_id);
    return GRAPH_PARAM_INVALID;
  }

  if (SetStagesOptions(graph_id) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Set][StagesOptions] failed, graph_id = %u", graph_id);
    return INTERNAL_ERROR;
  }
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  graph_rebuild_state_ctrl_->AddGraph(graph_id, new_compute_graph);
  return SUCCESS;
}

Status GraphManager::MergeSubGraph(ComputeGraphPtr &compute_graph, const ge::ComputeGraphPtr &original_compute_graph,
                                   GraphId root_graph_id, EnginePartitioner::Mode mode) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EnginePartitioner &partitioner = GetCompilerStages(root_graph_id).partitioner;
  if (instance_ptr != nullptr && instance_ptr->InitFlag()) {
    GE_ASSERT_SUCCESS(partitioner.MergeAfterSubGraphOptimization(compute_graph, original_compute_graph, mode),
                      "[Merge][SubGraph] merge end and placeholder after subGraph optimization failed.");
    GE_ASSERT_SUCCESS(compute_graph->TopologicalSorting(), "Failed to call TopologicalSorting, graph:%s",
                      compute_graph->GetName().c_str());
  } else {
    auto subgraph_list = partitioner.GetSubGraphMap();
    if (subgraph_list.find(original_compute_graph) != subgraph_list.end() &&
        !subgraph_list[original_compute_graph].empty() && subgraph_list[original_compute_graph][0] != nullptr) {
      compute_graph = subgraph_list[original_compute_graph][0]->GetSubGraph();
    }
  }
  return SUCCESS;
}

Status GraphManager::OptimizeSubGraphWithMultiThreads(ComputeGraphPtr compute_graph,
                                                      Graph2SubGraphInfoList &sub_graph_map, uint64_t session_id) {
  GE_CHECK_NOTNULL(compute_graph);
  // use default 16 multi thread
  uint32_t thread_num = 16;
  ThreadPool executor("ge_optsbgrh", thread_num, false);
  std::vector<std::future<Status>> vector_future;
  const auto &root_subgraph_list = sub_graph_map[compute_graph];
  std::string op_compile_strategy;
  (void)AttrUtils::GetStr(compute_graph, ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
  GELOGD("OptimizeSubGraphWithMultiThreads Process op_compile_strategy:%s", op_compile_strategy.c_str());
  int32_t device_id = kInvalidDeviceId;
  // 离线场景不会SetDevice，所以离线场景GetDevice会报错，可以通过device id是否是-1判断是在线or离线。在线场景需要给子线程SetDevice
  (void)aclrtGetDevice(&device_id);
  for (const auto &subgraph : root_subgraph_list) {
    GE_CHECK_NOTNULL(subgraph);
    if (!op_compile_strategy.empty()) {
      (void) AttrUtils::SetStr(subgraph->GetSubGraph(), ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
    }
    std::future<Status> f = executor.commit(GraphManager::ProcessSubGraphWithMultiThreads, this,
                                            compute_graph->GetGraphID(), subgraph,
                                            compute_graph->GetName(), session_id,
                                            error_message::GetErrMgrContext(),
                                            GetThreadLocalContext(), device_id);
    if (!f.valid()) {
      GELOGE(FAILED, "[Call][Commit] failed, Future is invalid, session_id:%lu", session_id);
      return FAILED;
    }
    vector_future.emplace_back(std::move(f));
  }
  for (auto &function_graph : compute_graph->GetAllSubgraphs()) {
    auto subgraph_list = sub_graph_map[function_graph];
    for (const auto &subgraph : subgraph_list) {
      GE_CHECK_NOTNULL(subgraph);
      if (!op_compile_strategy.empty()) {
        (void) AttrUtils::SetStr(subgraph->GetSubGraph(), ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
      }
      std::future<Status> f = executor.commit(GraphManager::ProcessSubGraphWithMultiThreads, this,
                                              compute_graph->GetGraphID(), subgraph,
                                              compute_graph->GetName(), session_id,
                                              error_message::GetErrMgrContext(),
                                              GetThreadLocalContext(), device_id);
      if (!f.valid()) {
        GELOGE(FAILED, "[Call][Commit] failed, Future is invalid, session_id:%lu", session_id);
        return FAILED;
      }
      vector_future.emplace_back(std::move(f));
    }
  }
  GELOGD("All sub graph num is %zu", vector_future.size());
  for (size_t i = 0; i < vector_future.size(); ++i) {
    Status ret_status = vector_future[i].get();
    if (ret_status != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "subgraph %zu optimize failed", i);
      GELOGE(ret_status, "[Check][Param] subgraph %zu optimize failed", i);
      return ret_status;
    }
  }
  return SUCCESS;
}

Status GraphManager::SetSubgraph(uint64_t session_id, ComputeGraphPtr compute_graph, EnginePartitioner &partitioner) {
  GE_CHECK_NOTNULL(compute_graph);
  auto sub_graph_map = partitioner.GetSubGraphMap();
  Status ret = OptimizeSubGraphWithMultiThreads(compute_graph, sub_graph_map, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][OptimizeSubGraphWithMultiThreads] failed, ret:%d, session_id:%lu", ret, session_id);
    return ret;
  }
  for (const auto &item : sub_graph_map) {
    for (const auto &subgraph_info : item.second) {
      GE_CHECK_NOTNULL(subgraph_info);
      const auto &subgraph = subgraph_info->GetSubGraph();
      GE_CHECK_NOTNULL(subgraph);
      for (const auto &new_graph : subgraph->GetAllSubgraphs()) {
        compute_graph->AddSubGraph(new_graph);
      }
    }
  }
  return SUCCESS;
}

#define GM_RUN_AND_DUMP_PERF(name, func, ...)                                                                    \
  do {                                                                                                           \
    GE_TRACE_RUN_PERF(GraphManager, func, __VA_ARGS__);                                                          \
    GE_DUMP(compute_graph, "PreRunAfter" name);                                                                  \
    GELOGI("Run %s on graph %s(%u) success.", name, compute_graph->GetName().c_str(), graph_node->GetGraphId()); \
  } while (0)

Status GraphManager::PreRunOptimizeOriginalGraph(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                                                 ge::ComputeGraphPtr &compute_graph, uint64_t session_id) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(compute_graph);

  GE_ASSERT_SUCCESS(RunCustomPass(graph_node->GetGraph()),
                    "[Run][CustomPass] in BeforeInferShape stage failed.");
  CompilerStages &stages = GetCompilerStages(graph_node->GetGraphId());
  GM_RUN_AND_DUMP_PERF("InitPreparation", stages.preparer.PrepareInit, graph_node,
                       session_id, graph_rebuild_state_ctrl_.get(), &resource_context_mgr_);
  // summary 处理放到normalize里
  GM_RUN_AND_DUMP_PERF("HandleSummaryOp", stages.optimizer.HandleSummaryOp, compute_graph);
  GM_RUN_AND_DUMP_PERF("NormalizeGraph", stages.preparer.NormalizeGraph, compute_graph, graph_node->GetOptions(),
                       inputs);
  GM_RUN_AND_DUMP_PERF("OptimizeGraphInit", stages.optimizer.OptimizeGraphInit, compute_graph);
  GM_RUN_AND_DUMP_PERF("OptimizeGraphPrepare", stages.optimizer.OptimizeOriginalGraphForQuantize, compute_graph);

  int64_t graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
  (void)AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage);
  if (graph_stage == static_cast<int64_t>(GraphStage::GRAPH_STAGE_FUZZ)) {
    GELOGD("graph_stage:%d.", static_cast<int32_t>(graph_stage));
    return SUCCESS;
  }

  GM_RUN_AND_DUMP_PERF("Prepare", stages.preparer.PrepareDynShape);
  GM_RUN_AND_DUMP_PERF("OptimizeOriginalGraph", stages.optimizer.OptimizeOriginalGraph, compute_graph);

  GM_RUN_AND_DUMP_PERF("PrepareRunningFormatRefiner", stages.preparer.PrepareRunningFormatRefiner);

  GM_RUN_AND_DUMP_PERF("RefineRunningPrecision",
      stages.optimizer.OptimizeOriginalGraphJudgePrecisionInsert, compute_graph);
  AutofuseOptimize autofuser;
  GE_CHK_STATUS_RET(autofuser.Run(compute_graph, inputs), "Failed to auto fuse optimize for graph:%s",
      compute_graph->GetName().c_str());
  GM_RUN_AND_DUMP_PERF("RefineRunningFormat",
      stages.optimizer.OptimizeOriginalGraphJudgeFormatInsert, compute_graph);
  GM_RUN_AND_DUMP_PERF("SubexpressionMigration", SubexpressionMigration, compute_graph);
  GE_RUN(GraphManager, stages.preparer.RecordAIPPInfo, compute_graph);
  if (IsTailingOptimization()) {
    GM_RUN_AND_DUMP_PERF("OptimizeSwitchOp", stages.preparer.SwitchOpOptimize, compute_graph);
  }
  GE_CHK_STATUS_RET(stages.optimizer.IdentifyReference(compute_graph),
                    "[Identify][Reference] failed, graph:%s.", compute_graph->GetName().c_str());
  GM_RUN_AND_DUMP_PERF("Optimize1", OptimizeStage1, compute_graph);
  GM_RUN_AND_DUMP_PERF("OptimizeAfterStage1", stages.optimizer.OptimizeAfterStage1, compute_graph);
  GE_ASSERT_SUCCESS(RunCustomPassAfterOriginGraphOptimize(graph_node->GetGraph()),
                  "[Run][CustomPass] in AfterOriginGraphOptimize stage failed.");
  GM_RUN_AND_DUMP_PERF("InferShape2", GraphUtilsEx::InferShapeInNeed, compute_graph);

  PassManager graph_pass;
  GE_CHK_STATUS_RET(graph_pass.AddPass("PreRun::CtrlEdgeTransferPass", new (std::nothrow) CtrlEdgeTransferPass));
  GE_CHK_STATUS_RET(graph_pass.Run(compute_graph));
  return SUCCESS;
}

Status GraphManager::PreRunOptimizeSubGraph(const GraphNodePtr &graph_node,
                                            ge::ComputeGraphPtr &compute_graph,
                                            uint64_t session_id) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(compute_graph);
  GM_RUN_AND_DUMP_PERF("OptimizeSubgraph", OptimizeSubgraph, graph_node, compute_graph, session_id);

  // Dump graph to tuning path
  if ((GetBuildMode(graph_node) == BUILD_MODE_TUNING) && (GetBuildStep(graph_node) == BUILD_STEP_AFTER_UB_MATCH)) {
    std::string tuning_path;
    (void) GetContext().GetOption(TUNING_PATH, tuning_path);
    GELOGD("Dump path:%s.", tuning_path.c_str());
    GraphUtils::DumpGEGraph(compute_graph, "", true, tuning_path);
  }
  return SUCCESS;
}

Status GraphManager::PreRunAfterOptimizeSubGraph(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                                                 GeRootModelPtr &ge_root_model, uint64_t session_id) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(compute_graph);

  const auto build_step = GetBuildStep(graph_node);
  GELOGD("PreRunAfterOptimizeSubGraph build step: [%s].", build_step.c_str());
  if (build_step != BUILD_STEP_AFTER_BUILD) {
    CompilerStages &stages = GetCompilerStages(graph_node->GetGraphId());
    GM_RUN_AND_DUMP_PERF("OptimizeWholeGraph", stages.optimizer.OptimizeWholeGraph, compute_graph);
    GM_RUN_AND_DUMP_PERF("Optimize2", OptimizeStage2, compute_graph);
    GM_RUN_AND_DUMP_PERF("OptimizeGraphBeforeBuild",
                        GetCompilerStages(graph_node->GetGraphId()).optimizer.OptimizeGraphBeforeBuild,
                        compute_graph);
    GM_RUN_AND_DUMP_PERF("OptimizeTensorMove", OptimizeTensorMove, compute_graph);
    GM_RUN_AND_DUMP_PERF("MemConflictProc", MemConflictProc, compute_graph);
    Status ret = compute_graph->TopologicalSorting();
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "TopologicalSorting fail, graph_id:%u", compute_graph->GetGraphID());
      GELOGE(ret, "[Call][TopologicalSorting] fail, graph_id:%u", compute_graph->GetGraphID());
      return ret;
    }
  }

  if (build_step == BUILD_STEP_BEFORE_BUILD) {
    GE_DUMP(compute_graph, "GraphBeforeBuildForQos");
    GELOGI("Dump graph before build in stpe: %s.", build_step.c_str());
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(UnfoldDynamicShapeGraph(compute_graph));
  GM_RUN_AND_DUMP_PERF("Build", Build, graph_node, compute_graph, ge_root_model, session_id);

  const auto build_mode = GetBuildMode(graph_node);
  const auto tuning_path = GetTuningPath(graph_node);
  std::string recompute_mode;
  (void)GetContext().GetOption(RECOMPUTE, recompute_mode);
  if ((build_mode.empty() || (build_mode == BUILD_MODE_BASELINE)) && recompute_mode.empty() && !tuning_path.empty()) {
    GraphUtils::DumpGEGraph(compute_graph, "", true, tuning_path);
    GELOGD("Success to dump after build graph:%s to tuning path:%s.", compute_graph->GetName().c_str(),
           tuning_path.c_str());
  }
  return SUCCESS;
}

Status GraphManager::UnfoldDynamicShapeGraph(ComputeGraphPtr &compute_graph) const {
  int32_t max_parallel_num = 0;
  GE_ASSERT_SUCCESS(GetGraphMaxParallelModeNum(max_parallel_num));
  // 临时方案：动态shape多实例并行, 则需要在编译阶段做图展开。
  // 原因是：多实例并行时，第一个实例在加载时会改原图，进而导致第二个实例拿到的原图有问题。
  // 正式方案需要实现原图进入rt2加载时只读、图展开默认前移
  if (!(StreamUtils::EnableDynamicShapeMultiStream() || (max_parallel_num > 1))) {
    return SUCCESS;
  }
  bool is_dynamic_shape = false;
  // To be compatible with the old process, do not verify the return value temporarily.
  (void)AttrUtils::GetBool(compute_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if ((!is_dynamic_shape) && (!compute_graph->GetGraphUnknownFlag())) {
    return SUCCESS;
  }
  GE_DUMP(compute_graph, "BeforeUnfoldSubgraphs");

  // known shape input subgraph should also be unfolded
  for (const auto &subgraph : compute_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(subgraph);
    const auto parent_node = subgraph->GetParentNode();
    if ((parent_node == nullptr) || (parent_node->GetType() != PARTITIONEDCALL)) {
      continue;
    }
    if (gert::GraphUnfolder::IsGraphDynamicCompiled(subgraph)) {
      continue;
    }
    const auto &sub_nodes = subgraph->GetDirectNode();
    if (std::all_of(sub_nodes.begin(), sub_nodes.end(), [](const NodePtr &node) {
          return ((node->GetType() == NETOUTPUT) ||
                  (((node->GetType() == CONSTANT) || (node->GetType() == CONSTANTOP) || (node->GetType() == VARIABLE))
                  && (node->GetInNodes().empty())) || IsGelocalOp(node->GetOpDesc()));
        })) {
      subgraph->SetGraphUnknownFlag(true);
    }
  }

  std::shared_ptr<ComputeGraph> merged_graph;
  GE_ASSERT_SUCCESS(gert::GraphUnfolder::UnfoldSubgraphs(compute_graph, merged_graph));
  compute_graph = std::move(merged_graph);
  GE_DUMP(compute_graph, "AfterUnfoldSubgraphs");

  return SUCCESS;
}

Status GraphManager::RunCustomPassAfterOriginGraphOptimize(ConstGraphPtr const_graph) const {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*const_graph);
  GE_CHECK_NOTNULL(compute_graph);
  fusion::FusionPassExecutor fusion_pass_executor;
  GE_TRACE_START(RunCustomPassAfterOriginGraphOptimize);
  GE_ASSERT_SUCCESS(fusion_pass_executor.RunPassesWithLegacyCustom(compute_graph, CustomPassStage::kAfterOriginGraphOptimize),
                    "Run custom pass for graph [%s] in stage [AfterOriginGraphOptimize] failed.", compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(RunCustomPassAfterOriginGraphOptimize, "GraphManager::RunCustomPassAfterOriginGraphOptimize");
  GE_DUMP(compute_graph, "RunCustomPassAfterOriginGraphOptimize");
  return SUCCESS;
}

Status GraphManager::RunCustomPass(ConstGraphPtr const_graph) const {
  auto comp_graph = GraphUtilsEx::GetComputeGraph(*const_graph);
  GE_CHECK_NOTNULL(comp_graph);
  fusion::FusionPassExecutor fusion_pass_executor;
  GE_TRACE_START(RunCustomPass);
  GE_ASSERT_SUCCESS(fusion_pass_executor.RunPassesWithLegacyCustom(comp_graph, CustomPassStage::kBeforeInferShape),
                    "Run custom pass for graph [%s] in stage [BeforeInferShape] failed.", comp_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(RunCustomPass, "GraphManager::RunCustomPass");
  GE_DUMP(comp_graph, "RunCustomPassBeforeInfershape");
  return SUCCESS;
}

Status GraphManager::PreRun(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                            GeRootModelPtr &ge_root_model, uint64_t session_id) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(graph_node->GetGraph());
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph_node->GetGraph());
  GE_CHECK_NOTNULL(compute_graph);
  compute_graph->SetSessionID(session_id);
  GEEVENT("PreRun start: graph node size %zu, session id %lu, graph id %u, graph name %s.",
          compute_graph->GetDirectNodesSize(), session_id, compute_graph->GetGraphID(),
          compute_graph->GetName().c_str());
  const uint32_t graph_id = graph_node->GetGraphId();
  auto analyzer_instance = Analyzer::GetInstance();
  GE_CHK_STATUS_RET(analyzer_instance->BuildJsonObject(session_id, graph_id),
                    "[Build][JsonObject] Failed, session_id:%lu", session_id);

  const auto ret = GeOptInfo::SetOptInfo();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][OptInfo] Set optional information failed.");
    return ret;
  }
  GE_TRACE_START(BuildModel);
  GE_CHECK_NOTNULL(graph_node->GetGraph());
  ComputeGraphPtr root_graph = GraphUtilsEx::GetComputeGraph(*graph_node->GetGraph());
  GE_CHECK_NOTNULL(root_graph);
  graph_node->SetComputeGraph(root_graph);
  GE_CHK_STATUS_RET(BuildModel(graph_node, inputs, root_graph, ge_root_model),
                    "[Build][Model] failed, session_id:%lu, graph_id:%u.", session_id, graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildModel, "ModelBuild");
  ReportTracingRecordDuration(ge::TracingModule::kModelCompile);
  GEEVENT("[GEPERFTRACE] GE PreRun End");
  return SUCCESS;
}

Status GraphManager::BuildModel(const GraphNodePtr &graph_node, const std::vector<GeTensor> &input_tensors,
                                ComputeGraphPtr &root_graph, GeRootModelPtr &ge_root_model) {
  GE_CHECK_NOTNULL(root_graph);
  GE_DUMP(root_graph, "GraphPreRunBegin");
  GELOGI("Build model start, graph id = %d, graph_name = %s", graph_node->GetGraphId(), root_graph->GetName().c_str());

  ModelCache model_cache;
  GE_CHK_STATUS_RET(model_cache.Init(root_graph, graph_rebuild_state_ctrl_.get()),
                    "Failed to init model cache");
  GE_CHK_STATUS_RET(model_cache.TryLoadModelFromCache(root_graph, ge_root_model),
                    "Failed to load model from cache.");
  if (ge_root_model != nullptr) {
    GEEVENT("Load model from cache success.");
    auto const root_graph_cached = ge_root_model->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph_cached);
    const auto graph_cached = GraphUtilsEx::CreateGraphPtrFromComputeGraph(root_graph_cached);
    GE_CHECK_NOTNULL(graph_cached);
    graph_node->SetComputeGraph(root_graph_cached);
    graph_node->SetGraph(graph_cached);
    GE_CHK_STATUS_RET(ResortAndUpdateMultiBatchContext(graph_node), "Resort and update multi batch failed.");
    return SUCCESS;
  } else {
    GE_CHK_STATUS_RET_NOLOG(DoBuildModel(root_graph, input_tensors, ge_root_model));
    GE_CHK_STATUS_RET(model_cache.TryCacheModel(ge_root_model), "Failed to cache model.");
    const auto graph_cached = GraphUtilsEx::CreateGraphPtrFromComputeGraph(root_graph);
    GE_CHECK_NOTNULL(graph_cached);
    graph_node->SetComputeGraph(root_graph);
    graph_node->SetGraph(graph_cached);
    GELOGI("Build model successfully, graph id = %d, graph_name = %s", graph_node->GetGraphId(),
           root_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status GraphManager::SaveOriginCommunicationNodes(const ComputeGraphPtr &compute_graph) const {
  GE_ASSERT_NOTNULL(compute_graph);
  std::map<std::string, std::vector<std::string>> group_2_comm_nodes;
  GE_ASSERT_SUCCESS(GetCommunicationNodes(compute_graph, false, group_2_comm_nodes));
  const uint32_t graph_id = compute_graph->GetGraphID();
  PrintCommunicationNodes(graph_id, compute_graph->GetName(), "Before build", group_2_comm_nodes);
  if (!group_2_comm_nodes.empty()) {
    GraphNodePtr graph_node = nullptr;
    GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph node fail, graph_id is %u", graph_id);
    graph_node->SetCommunicationNodes(group_2_comm_nodes);
  }
  return SUCCESS;
}

Status GraphManager::VerifyCommNodesOrderAfterEngineAssigned(const ComputeGraphPtr &compute_graph) const  {
  GE_ASSERT_NOTNULL(compute_graph);
  std::map<std::string, std::vector<std::string>> group_2_comm_nodes;
  GE_ASSERT_SUCCESS(GetCommunicationNodes(compute_graph, true, group_2_comm_nodes));
  const uint32_t graph_id = compute_graph->GetGraphID();
  PrintCommunicationNodes(graph_id, compute_graph->GetName(), "After engine assigned", group_2_comm_nodes);
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph node fail, graph_id is %u", graph_id);
  GE_ASSERT_SUCCESS(VerifyCommNodesOrder(graph_node->GetCommunicationNodes(), group_2_comm_nodes));
  // need refresh base communication nodes to graph node
  graph_node->SetCommunicationNodes(group_2_comm_nodes);
  return SUCCESS;
}

Status GraphManager::VerifyCommNodesOrderAfterBuild(const ComputeGraphPtr &compute_graph) const  {
  GE_ASSERT_NOTNULL(compute_graph);
  std::map<std::string, std::vector<std::string>> group_2_comm_nodes;
  GE_ASSERT_SUCCESS(GetCommunicationNodes(compute_graph, true, group_2_comm_nodes));
  const uint32_t graph_id = compute_graph->GetGraphID();
  PrintCommunicationNodes(graph_id, compute_graph->GetName(), "After build", group_2_comm_nodes);
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph node fail, graph_id is %u", graph_id);
  GE_ASSERT_SUCCESS(VerifyCommNodesOrder(graph_node->GetCommunicationNodes(), group_2_comm_nodes));
  return SUCCESS;
}

Status GraphManager::DoBuildModel(ComputeGraphPtr &compute_graph,
                                  const std::vector<GeTensor> &input_tensors,
                                  GeRootModelPtr &ge_root_model) {
  GE_ASSERT_SUCCESS(SaveOriginCommunicationNodes(compute_graph));
  GE_CHK_STATUS_RET(OptimizeGraph(input_tensors, compute_graph),
                    "[Optimize][Graph] failed, graph = %s", compute_graph->GetName().c_str());
  auto graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
  (void)AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage);
  if (graph_stage == static_cast<int64_t>(GraphStage::GRAPH_STAGE_FUZZ)) {
    GELOGD("graph_stage:%d.", static_cast<int32_t>(graph_stage));
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(BuildGraph(compute_graph, ge_root_model), "[Build][Graph] failed, graph = %s",
                    compute_graph->GetName().c_str());
  RtContextUtil::GetInstance().DestroyRtContexts(compute_graph->GetSessionID(), compute_graph->GetGraphID());
  GELOGI("[Build][Graph] successfully, graph = %s", compute_graph->GetName().c_str());
  return SUCCESS;
}

Status GraphManager::ResortAndUpdateMultiBatchContext(const GraphNodePtr &graph_node) {
  const auto compute_graph = graph_node->GetComputeGraph();
  GE_CHECK_NOTNULL(compute_graph);
  std::vector<NodePtr> data_nodes;
  for (const NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op_desc = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType()) && (op_desc->GetName() != kShapeDataName) &&
        (op_desc->GetName().find(kSubstrOfGetNextNosinkName) == std::string::npos)) {
      data_nodes.emplace_back(input_node);
      GELOGD("Name of data node is %s.", op_desc->GetName().c_str());
    }
  }
  // the input data name in user_input_dims is not match in graph data name in tfa scenario, so update name first
  // keep same process in multi_batch_clone_pass.cc
  if (GetLocalOmgContext().dynamic_node_type == DATA) {
    GE_CHK_STATUS_RET(multibatch::UpdateNameOfData(compute_graph, data_nodes),
                      "[Update][Name] Of InputShape failed, graph:%s.", compute_graph->GetName().c_str());
  }
  auto &batch_shapes = GetLocalOmgContext().batch_shapes;
  GE_ASSERT_SUCCESS(multibatch::InitDynamicParams(batch_shapes));
  if (batch_shapes.empty()) {
    GELOGD("There is no multi-batch options, no need resort and update dynamic batch options.");
    return SUCCESS;
  }
  // resort dynamic shape
  GE_CHK_STATUS_RET(multibatch::CheckNegativeCountOfOptions(batch_shapes),
                    "[Check][Param] Input_shape and dynamic_dims should set correct params.");
  GE_CHK_STATUS_RET(multibatch::CheckDynamicParams(batch_shapes), "[Check][Params] Invalid multi-batch param");
  GE_ASSERT_SUCCESS(ResortDynamicBatchInput(batch_shapes, data_nodes), "Resort dynamic batch config failed.");
  return GraphManager::SetRunContext(graph_node);
}

Status GraphManager::ResortDynamicBatchInput(const std::vector<std::vector<int64_t>> &batch_shapes,
                                             std::vector<NodePtr> &data_nodes) {
  auto data_name_and_shape = GetLocalOmgContext().user_input_dims;
  std::map<std::string, std::vector<vector<int64_t>>> data_to_dynamic_info;
  GE_CHK_STATUS_RET(multibatch::ParserDataToDynamicInfo(batch_shapes, data_name_and_shape, data_to_dynamic_info),
                    "[Parser][DataToDynamicInfo] failed.");
  if (GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGI("No need to sort dynamic dims when offline infer.");
    GetLocalOmgContext().batch_shapes = batch_shapes;
    return SUCCESS;
  }
  if (data_nodes.empty()) {
    GetLocalOmgContext().batch_shapes = batch_shapes;
    return SUCCESS;
  }
  multibatch::SortDataNodesByName(data_nodes);
  GetLocalOmgContext().data_nodes = data_nodes;
  GE_CHK_STATUS_RET(multibatch::UpdateDataShapeByUserInput(), "Update data shape by user input failed.");

  multibatch::SortDataNodesByIndex(data_nodes);
  GE_CHK_STATUS_RET(UpdateMultiBatchContext(data_nodes, batch_shapes, data_to_dynamic_info),
                    "Update ");
  return SUCCESS;
}

Status GraphManager::UpdateMultiBatchContext(const std::vector<NodePtr> &data_nodes,
    const std::vector<std::vector<int64_t>> &batch_shapes,
    const std::map<std::string, std::vector<vector<int64_t>>> &data_to_dynamic_info) {
  auto data_name_and_shape = GetLocalOmgContext().user_input_dims;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_data_name_and_shape;
  for (size_t i = 0UL; i < data_nodes.size(); i++) {
    int32_t index = -1;
    for (size_t j = 0UL; j < data_name_and_shape.size(); ++j) {
      if (data_name_and_shape[j].first == data_nodes[i]->GetName()) {
        index = static_cast<int32_t>(j);
        break;
      }
    }
    GE_ASSERT_TRUE(index >= 0LL);
    const auto data_shape = data_name_and_shape[static_cast<size_t>(index)].second;
    user_data_name_and_shape.emplace_back(std::make_pair(data_nodes[i]->GetName(), data_shape));
  }
  GetLocalOmgContext().user_input_dims = user_data_name_and_shape;

  // sort dynamic dims with index
  std::vector<std::vector<int64_t>> new_batch_shape;
  new_batch_shape.resize(batch_shapes.size());
  for (const auto &name_and_shape : user_data_name_and_shape) {
    std::string data_name = name_and_shape.first;
    const auto iter = data_to_dynamic_info.find(data_name);
    if (iter != data_to_dynamic_info.end()) {
      const std::vector<std::vector<int64_t>> &dynamic_dim = iter->second;
      for (size_t i = 0UL; i < batch_shapes.size(); i++) {
        GE_ASSERT_TRUE(i < dynamic_dim.size());
        for (size_t j = 0UL; j < dynamic_dim[i].size(); j++) {
          new_batch_shape[i].emplace_back(dynamic_dim[i][j]);
        }
      }
    }
  }
  GetLocalOmgContext().batch_shapes = new_batch_shape;
  GELOGI("Update batch shapes and user input dims success.");
  return SUCCESS;
}

Status GraphManager::InnerRunGraphWithStream(const GraphNodePtr &graph_node, const GraphId &graph_id,
                                             rtStream_t stream, const std::vector<GeTensor> &inputs,
                                             std::vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL(executor_);
  return executor_->RunGraphWithStream(graph_node, graph_id, stream, inputs, outputs);
}

Status GraphManager::SubexpressionMigration(ComputeGraphPtr &compute_graph) const {
  PassManager pass_manager;
  GE_CHK_STATUS_RET(pass_manager.AddPass("SubexpressionMigrationPass", new (std::nothrow) SubexpressionMigrationPass));
  GE_CHK_STATUS_RET(pass_manager.AddPass("UnusedArgsCleanPass", new (std::nothrow) UnusedArgsCleanPass));

  GE_TRACE_START(SubexpressionMigrationPass);
  auto ret = pass_manager.Run(compute_graph);
  GE_COMPILE_TRACE_TIMESTAMP_END(SubexpressionMigrationPass, "GraphManager::SubexpressionMigration");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "[Run][SubexpressionMigrationPass] failed, ret:%u.", ret);
    return ret;
  }

  return SUCCESS;
}

Status GraphManager::StartForRunGraph(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                                      GeRootModelPtr &ge_root_model, uint64_t session_id, const rtStream_t stream) {
  // it will not execute graph prreprocess, optimize, parition, build if the graph has built successful.
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(graph_node->GetGraph());
  Status ret = SUCCESS;
  if (IsGraphNeedBuild(graph_node)) {
    if (graph_node->GetBuildFlag()) {
      REPORT_INNER_ERR_MSG("E19999", "Graph:%u has not build before, can't run directly, "
                         "check invalid", graph_node->GetGraphId());
      GELOGE(PARAM_INVALID,
             "[Get][BuildFlag] The graph %u need to re-build, you should remove it from GE "
             "first, then AddGraph again and rebuild it.",
             graph_node->GetGraphId());
      return PARAM_INVALID;
    }
    // 临时方案使用hint shape填充input，后续重构将input和hint shape解耦开，在符号化解析hint shape
    auto compute_graph_ptr = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
    GE_ASSERT_NOTNULL(compute_graph_ptr);
    std::vector<GeShape> hint_shape;
    GE_ASSERT_SUCCESS(ParseHintInputShape(hint_shape));
    std::vector<GeTensor> ge_tensor_inputs;
    bool is_enable_autofuse = GetAutofuseFlagValue(kAutoFuseEnableOption) == "true";
    // Adapt to offline compile autofuse scenarios
    if (is_enable_autofuse && inputs.empty()) {
      // static scence & dynamic atc scene
      GE_ASSERT_SUCCESS(ConstructInputTensors(compute_graph_ptr, hint_shape, ge_tensor_inputs),
                        "Construct model input tensor desc failed, maybe the input tensor desc is invalid.");
    } else if (is_enable_autofuse && !inputs.empty()){
      // dynamic acl scene
      ge_tensor_inputs = inputs;
      GE_ASSERT_SUCCESS(UpdateInputWithHintShape(hint_shape, ge_tensor_inputs),
                        "Update model input tensor desc failed, maybe the input tensor desc is invalid.");
    } else {
      ge_tensor_inputs = inputs;
    }
    ret = PreRun(graph_node, ge_tensor_inputs, ge_root_model, session_id);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][PreRun] Failed, graph_id:%u, session_id:%lu.", graph_node->GetGraphId(), session_id);
      return ret;
    }
    graph_node->SetBuildFlag(true);
    GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
    graph_rebuild_state_ctrl_->SetGraphBuildEnd(graph_node->GetGraphId());

    auto compute_graph = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
    GE_CHECK_NOTNULL(compute_graph);
    int64_t graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
    (void)AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage);
    if (graph_stage == static_cast<int64_t>(GraphStage::GRAPH_STAGE_FUZZ)) {
      GELOGD("graph_stage:%d.", static_cast<int32_t>(graph_stage));
      return SUCCESS;
    }
    ret = InnerLoadGraph(ge_root_model, graph_node, stream);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Load][Graph] Failed, graph_id:%u.", graph_node->GetGraphId());
      return ret;
    }
    return SUCCESS;
  }
  ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  if (!graph_node->GetLoadFlag()) {
    ret = InnerLoadGraph(ge_root_model, graph_node, stream);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Load][Graph] Failed, graph_id:%u.", graph_node->GetGraphId());
      return ret;
    }
  }
  return SUCCESS;
}

Status GraphManager::TranFrameOp(const GraphNodePtr &graph_node) {
  if (options_.local_fmk_op_flag) {
    GE_CHECK_NOTNULL(graph_node->GetGraph());
    ComputeGraphPtr compute_graph_tmp = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
    GE_ASSERT_NOTNULL(compute_graph_tmp);
    GetCompilerStages(graph_node->GetGraphId()).optimizer.TranFrameOp(compute_graph_tmp);
  }
  return SUCCESS;
}

Status GraphManager::SetFrozenInputAttrs(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) const {
  GE_CHECK_NOTNULL(graph_node);
  const auto &index_to_node_info = graph_node->GetFrozenInputInfo();
  const auto &frozen_indicies = graph_node->GetFrozenInputIndex();
  if (index_to_node_info.empty()) {
    GELOGI("There are no addr and len attr set in frozen index option.");
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(ge_root_model);
  const auto root_graph = ge_root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  for (const auto &node : root_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!OpTypeUtils::IsDataNode(op_desc->GetType())) {
      continue;
    }
    int32_t index = -1;
    if ((!AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, index)) ||
        (frozen_indicies.count(static_cast<uint32_t>(index)) == 0UL)) {
      continue;
    }
    const auto iter = index_to_node_info.find(static_cast<uint32_t>(index));
    GE_ASSERT_TRUE(iter != index_to_node_info.cend(), "Can not find frozen addr and length for input[%d]", index);
    GE_ASSERT_TRUE(AttrUtils::SetBool(op_desc, "frozen_input", true), "Set frozen input attr failed.");
    GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, "addr", iter->second.first), "Set frozen input addr attr failed.");
    GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, "size", iter->second.second), "Set frozen input size attr failed.");
    GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, "placement", ge::Placement::kPlacementDevice),
                   "Set frozen data placement attr failed.");
    const auto &input_desc = op_desc->GetInputDesc(0);
    const auto &input_shape = input_desc.GetShape().GetDims();
    GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, "storage_shape", input_shape), "Set storage_shape attr failed.");
    const auto &org_input_shape = input_desc.GetOriginShape().GetDims();
    for (const int64_t &dim : org_input_shape) {
      GE_ASSERT_TRUE(dim > 0, "All dims in original shape[%s] of frozen data should be greater than 0.",
                     input_desc.GetOriginShape().ToString().c_str());
    }
    GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, "origin_shape", org_input_shape), "Set origin_shape attr failed.");
    const auto &data_type = input_desc.GetDataType();
    GE_ASSERT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", data_type), "Set data type attr failed.");
    GELOGI("Set attrs for frozen input node[%s], addr[%lu], len[%lu], storage shape[%s], storage format[%d],"
           " original shape[%s], original format[%d], datatype[%d].",
        node->GetName().c_str(), iter->second.first, iter->second.second,
        input_desc.GetShape().ToString().c_str(), static_cast<int32_t>(input_desc.GetFormat()),
        input_desc.GetOriginShape().ToString().c_str(), static_cast<int32_t>(input_desc.GetOriginFormat()),
        static_cast<int32_t>(data_type));
  }
  return SUCCESS;
}

Status GraphManager::LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options,
    const rtStream_t stream) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node);
  GeRootModelPtr ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);

  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph_id:%u, check invalid",
      graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] graph already running, graph id = %u", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }
  if (graph_node->GetLoadFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph has been loaded, graph_id = %u", graph_id);
    GELOGE(GE_GRAPH_REPEAT_OPERATION, "[Check][LoadFlag] Graph has been loaded, graph_id = %u", graph_id);
    return GE_GRAPH_REPEAT_OPERATION;
  }
  auto &graph_options = const_cast<std::map<std::string, std::string>&>(graph_node->GetOptions());
  for (const auto &iter : options) {
    GELOGI("Get option key[%s] value[%s].", iter.first.GetString(), iter.second.GetString());
    if (graph_options.find(iter.first.GetString()) == graph_options.end()) {
      (void)graph_options.emplace(iter.first.GetString(), iter.second.GetString());
    }
  }
  GE_MAKE_GUARD(option_recover_guard, ([&options, &graph_options] () {
    for (const auto &iter : options) {
      (void)graph_options.erase(iter.first.GetString());
    }
  }));
  GetThreadLocalContext().SetGraphOption(graph_options);
  GE_ASSERT_SUCCESS(graph_node->ParseFrozenInputIndex(), "Parse frozen input index failed");
  GE_ASSERT_SUCCESS(SetFrozenInputAttrs(ge_root_model, graph_node), "Set frozen input attrs failed");
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator != nullptr) {
    if (logLevel_ <= DLOG_DEBUG) {
      GELOGD("LoadGraph with external allocator = %p", external_allocator.get());
    }
    CompiledGraphSummaryPtr summary = graph_node->GetCompiledGraphSummary();
    if (summary == nullptr) {
      GE_ASSERT_SUCCESS(GetCompiledGraphSummary(graph_id, summary));
      GE_ASSERT_NOTNULL(summary);
    }
  }
  return InnerLoadGraph(ge_root_model, graph_node, stream);
}

Status GraphManager::InnerLoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                                    const rtStream_t stream) const {
  GE_CHECK_NOTNULL(graph_node);
  GELOGI("[LoadGraph] run_graph_flag[%d], graph_id[%u]", options_.run_graph_flag, graph_node->GetGraphId());
  if (!options_.run_graph_flag) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(executor_);

  /*
   * 规避方案：规避aoe流程不实际编译模型，但是会走到加载流程里，后续aoe流程在外面统一判断不走到加载流程
   * 方案详述：直接判断model为空的时候不加载
   * 方案约束：正式编译失败不能走到加载流程
   */
  if (ge_root_model == nullptr) {
    GELOGI("model is null, no need load");
    return SUCCESS;
  }

  Status ret = executor_->LoadGraph(ge_root_model, graph_node, stream);
  if (ret != SUCCESS) {
    return ret;
  }
  graph_node->SetLoadFlag(true);
  return SUCCESS;
}

Status GraphManager::NormalizeInputsOutputs(const ComputeGraphPtr &compute_graph,
                                            const std::vector<GeTensor> &inputs,
                                            const std::vector<GeTensor> &outputs,
                                            std::vector<GeTensor> &normalized_inputs) const {
  if (logLevel_ <= DLOG_DEBUG) {
    GELOGD("Start to normalize input ge tensors of graph %s", compute_graph->GetName().c_str());
  }
  normalized_inputs.reserve(inputs.size());
  for (size_t i = 0U; i < inputs.size(); i++) {
    normalized_inputs.emplace_back(TensorAdapter::NormalizeGeTensor(inputs[i]));
    if (logLevel_ <= DLOG_DEBUG) {
      GELOGD("Normalize graph %s input %lu %s[%s] -> %s[%s] addr %lu size %lu", compute_graph->GetName().c_str(),
             i, ge::TypeUtils::FormatToSerialString(inputs[i].GetTensorDesc().GetFormat()).c_str(),
             inputs[i].GetTensorDesc().GetShape().ToString().c_str(),
             ge::TypeUtils::FormatToSerialString(normalized_inputs.back().MutableTensorDesc().GetFormat()).c_str(),
             normalized_inputs.back().MutableTensorDesc().GetShape().ToString().c_str(),
             PtrToValue(normalized_inputs.back().GetData().GetData()),
             normalized_inputs.back().GetData().GetSize());
    }
  }

  std::vector<GeTensor> normalized_outputs;
  normalized_outputs.reserve(outputs.size());
  for (size_t i = 0U; i < outputs.size(); i++) {
    normalized_outputs.emplace_back(TensorAdapter::NormalizeGeTensor(outputs[i]));
    if (logLevel_ <= DLOG_DEBUG) {
      GELOGD("Normalize graph %s output %lu %s[%s] -> %s[%s] addr %lu size %lu", compute_graph->GetName().c_str(),
             i, ge::TypeUtils::FormatToSerialString(outputs[i].GetTensorDesc().GetFormat()).c_str(),
             outputs[i].GetTensorDesc().GetShape().ToString().c_str(),
             ge::TypeUtils::FormatToSerialString(normalized_outputs.back().MutableTensorDesc().GetFormat()).c_str(),
             normalized_outputs.back().MutableTensorDesc().GetShape().ToString().c_str(),
             PtrToValue(normalized_outputs.back().GetData().GetData()),
             normalized_outputs.back().GetData().GetSize());
    }
  }
  if (compute_graph->GetGraphOutNodesInfo().size() != normalized_outputs.size()) {
    GELOGE(GE_GRAPH_GET_IN_OUT_FAILED, "Run graph with stream async output size mismatch expect %lu but got %lu.",
           compute_graph->GetGraphOutNodesInfo().size(), normalized_outputs.size());
    return GE_GRAPH_GET_IN_OUT_FAILED;
  }

  size_t output_index = 0U;
  for (auto &retval : compute_graph->GetGraphOutNodesInfo()) {
    GE_CHECK_NOTNULL(retval.first);
    GE_CHECK_NOTNULL(retval.first->GetOpDesc());
    GE_ASSERT_SUCCESS(retval.first->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(retval.second),
        normalized_outputs[output_index++].MutableTensorDesc()),
        "Update output desc size failed for op:%s(%s) index:0 ",
        retval.first->GetOpDesc()->GetName().c_str(), retval.first->GetOpDesc()->GetType().c_str());
  }
  return SUCCESS;
}

Status GraphManager::CheckGraphVaildBeforeExecute(const GraphId &graph_id, GraphNodePtr &graph_node) const {
  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph id = %u, "
                       "check invalid.", graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] Run graph with stream async graph already running, "
           "graph id = %u.", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }

  if (IsGraphNeedBuild(graph_node)) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u has not build before, can't execute directly, check invalid",
      graph_node->GetGraphId());
    GELOGE(PARAM_INVALID, " The graph %u need to build", graph_node->GetGraphId());
    return PARAM_INVALID;
  }

  if (!graph_node->GetLoadFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u has not load before, can't execute directly, check invalid",
      graph_node->GetGraphId());
    GELOGE(PARAM_INVALID, "The graph %u need to load before execute.",graph_node->GetGraphId());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status GraphManager::ExecuteGraphWithStreamAsync(const GraphId &graph_id, const rtStream_t stream,
                                                 const std::vector<gert::Tensor> &inputs,
                                                 std::vector<gert::Tensor> &outputs) {
  if (inputs.empty()) {
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("Execute graph with stream async, initialize sub graph has no inputs.");
    }
  }
  
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] failed, Execute graph with stream async, graph does not exist, "
      "graph id = %u.", graph_id);
    return ret;
  }
  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph id = %u, check invalid.", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] Execute graph with stream async, graph node is NULL, "
           "graph id = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  std::lock_guard<std::mutex> lock(graph_node->GetRunMutex());

  Status checkStatus = CheckGraphVaildBeforeExecute(graph_id, graph_node);
  if (checkStatus != SUCCESS) {
    return checkStatus;
  }

  // set graph's run flag
  graph_node->SetRunFlag(true);
  GE_MAKE_GUARD(run_flag_guard, [&graph_node] () { graph_node->SetRunFlag(false); });
  graph_node->SetIsSpecificStream(true);

  GE_CHECK_NOTNULL(executor_);
  return executor_->ExecuteGraphWithStream(graph_node, graph_id, stream, inputs, outputs);
}

Status GraphManager::RunGraphWithStreamAsync(const GraphId &graph_id, const rtStream_t stream, uint64_t session_id,
                                             const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);

  if (inputs.empty()) {
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("Run graph with stream async, initialize sub graph has no inputs.");
    }
  }

  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] failed, Run graph with stream async, graph does not exist, graph id = %u.", graph_id);
    return ret;
  }
  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph id = %u, check invalid.", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] Run graph with stream async, graph node is NULL, "
           "graph id = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  std::lock_guard<std::mutex> lock(graph_node->GetRunMutex());

  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph id = %u, "
                       "check invalid.", graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] Run graph with stream async graph already running, "
           "graph id = %u.", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  UpdateLocalOmgContext(graph_id);

  graph_node->SetIsSpecificStream(true);
  GE_CHECK_NOTNULL(graph_node->GetGraph());
  ComputeGraphPtr compute_graph_tmp = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
  GE_CHECK_NOTNULL(compute_graph_tmp);

  if (options_.local_fmk_op_flag) {
    GetCompilerStages(graph_id).optimizer.TranFrameOp(compute_graph_tmp);
  }

  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator != nullptr) {
    if (logLevel_ <= DLOG_DEBUG) {
      GELOGD("RunGraphWithStreamAsync with external allocator = %p", external_allocator.get());
    }
    
    CompiledGraphSummaryPtr summary = graph_node->GetCompiledGraphSummary();
    if (summary == nullptr) {
      GE_ASSERT_SUCCESS(CompileGraph(graph_id, session_id, {}));
      GE_ASSERT_SUCCESS(GetCompiledGraphSummary(graph_id, summary));
      GE_ASSERT_NOTNULL(summary);
    }
  }
  // set graph's run flag
  graph_node->SetRunFlag(true);
  GE_MAKE_GUARD(run_flag_guard, [&graph_node] () { graph_node->SetRunFlag(false); });
  GeRootModelPtr ge_root_model = nullptr;
  if (IsGraphNeedBuild(graph_node)) {
    std::vector<GeTensor> normalized_inputs;
    GE_ASSERT_SUCCESS(NormalizeInputsOutputs(compute_graph_tmp, inputs, outputs, normalized_inputs));
    ret = StartForRunGraph(graph_node, normalized_inputs, ge_root_model, session_id, stream);
  } else {
    ret = StartForRunGraph(graph_node, inputs, ge_root_model, session_id, stream);
  }
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][StartForRunGraph] failed, session_id:%lu", session_id);
    return ret;
  }
  return InnerRunGraphWithStream(graph_node, graph_id, stream, inputs, outputs);
}

// 会先触发编译，再触发运行
Status GraphManager::RunGraph(const GraphId &graph_id, const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs, uint64_t session_id) {
  // find graph
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node);
  UpdateLocalOmgContext(graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());

  // set graph's run flag
  GE_ASSERT_SUCCESS(TranFrameOp(graph_node));
  GeRootModelPtr ge_root_model = nullptr;
  std::vector<GeTensor> ge_inputs;
  for (auto &item : inputs) {
    ge_inputs.emplace_back(TensorAdapter::AsGeTensor(item));
  }
  auto ret = StartForRunGraph(graph_node, ge_inputs, ge_root_model, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][StartForRunGraph] failed, session_id:%lu", session_id);
    return ret;
  }

  std::vector<gert::Tensor> tensors_view;
  GE_ASSERT_SUCCESS(TensorTransUtils::AsTensorsView(inputs, tensors_view));
  std::vector<gert::Tensor> gert_outputs;
  ret = RunGraph(graph_id, tensors_view, gert_outputs);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][Graph]failed, session_id:%lu graph_id=%u.", session_id, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager RunGraph failed, session_id:%lu graph_id=%u.", session_id, graph_id);
    return ret;
  }
  outputs.clear();
  GE_ASSERT_SUCCESS(TensorTransUtils::GertTensors2Tensors(gert_outputs, outputs));
  GELOGI("[session_id:%lu] run graph success, graph_id=%u.", session_id, graph_id);
  return SUCCESS;
}

// 本接口内不会再编译
Status GraphManager::RunGraph(const GraphId &graph_id, const std::vector<gert::Tensor> &inputs,
                              std::vector<gert::Tensor> &outputs) {
  GELOGI("[RunGraph] start to run graph, graph_id = %u, is_train_graph: %d", graph_id, GetTrainFlag());

  if (inputs.empty()) {
    GELOGI("[RunGraph] initialize sub graph has no inputs");
  }

  // find graph
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);

  std::lock_guard<std::mutex> lock(graph_node->GetRunMutex());

  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph_id:%u, check invalid",
      graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] graph already running, graph id = %u", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }
  GE_CHECK_NOTNULL(graph_node->GetGraph());
  UpdateLocalOmgContext(graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  // set graph's run flag
  graph_node->SetRunFlag(true);
  GE_MAKE_GUARD(run_flag_guard, [&graph_node] () { graph_node->SetRunFlag(false); });

  ComputeGraphPtr compute_graph_tmp = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
  GE_IF_BOOL_EXEC(GetTrainFlag(),
                  GE_IF_BOOL_EXEC(compute_graph_tmp == nullptr,
                                  REPORT_INNER_ERR_MSG("E19999", "compute_graph is nullptr in graph_node, graph_id:%u, "
                                                    "check invalid", graph_id);
                                  GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Get][ComputeGraph] failed, "
                                         "compute_graph_tmp is NULL, graph id = %u.", graph_id);
                                  return GE_GRAPH_GRAPH_NODE_NULL;))
  // excute graph
  GE_CHECK_NOTNULL(executor_);
  auto ret = executor_->RunGraph(graph_node, graph_id, inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  if (GetTrainFlag()) {
    if (compute_graph_tmp->IsSummaryGraph()) {
      ret = SummaryHandle(graph_id, outputs);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Call][SummaryHandle] failed, graph_id:%u", graph_id);
      }
    }

    auto root_model = graph_node->GetGeRootModel();
    GE_CHECK_NOTNULL(root_model);
    GELOGI("Start CheckpointHandle.");
    auto checkPointGraph = root_model->GetRootGraph();
    if (IsCheckpointGraph(checkPointGraph)) {
      ret = CheckpointHandle(graph_id, checkPointGraph, outputs);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Check][PointHandle] failed, graph_id:%u", graph_id);
      }
    }
  }

  GELOGI("[RunGraph] run graph success, graph_id = %u.", graph_id);
  return SUCCESS;
}

Status GraphManager::GenerateInfershapeGraph(GraphId &graph_id) {
  GELOGI("[DumpInfershapeJson] start to DumpInfershapeJson graph, graph_id=%u.", graph_id);
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }

  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph_id:%u, check invalid",
                       graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Get][GraphNode] graph node is NULL, graphId = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  UpdateLocalOmgContext(graph_id);

  ret = GetCompilerStages(graph_id).preparer.GenerateInfershapeGraph(graph_node->GetGraph());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][CompilerStages] failed");
    return ret;
  }

  GELOGI("[DumpInfershapeJson] Dump infershape json success, graph_id=%u.", graph_id);
  return ret;
}

Status GraphManager::BuildGraphForUnregisteredOp(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                                 GeRootModelPtr &ge_root_model, uint64_t session_id) {
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }

  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] graph node is NULL, graphId = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  UpdateLocalOmgContext(graph_id);

  GE_CHECK_NOTNULL(graph_node->GetGraph());
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph_node->GetGraph());
  GE_CHECK_NOTNULL(compute_graph);

  GM_RUN_AND_DUMP_PERF("PrepareInit", GetCompilerStages(graph_id).preparer.PrepareInit,
                       graph_node, session_id);
  GM_RUN_AND_DUMP_PERF("NormalizeGraph", GetCompilerStages(graph_id).preparer.NormalizeGraph, compute_graph,
                       graph_node->GetOptions(), inputs);
  GM_RUN_AND_DUMP_PERF("Prepare", GetCompilerStages(graph_id).preparer.PrepareDynShape);

  for (auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->HasAttr(ATTR_NAME_UNREGST_OPPATH)) {
      std::vector<ge::NodePtr> node_vec = {node};

      auto instance_ptr = ge::GELib::GetInstance();
      if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
        REPORT_INNER_ERR_MSG("E19999", "GELib is not init before, graph_id:%u, check invalid", graph_id);
        GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] GELib is not init before, graph_id:%u", graph_id);
        return GE_CLI_GE_NOT_INITIALIZED;
      }

      OpsKernelInfoStorePtr kernel_info =
          instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(op_desc->GetOpKernelLibName());
      if (kernel_info == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "GetOpsKernelInfoStore fail for op:%s(%s), kernel_lib_name:%s, graph_id:%u, "
                           "check invalid", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                           op_desc->GetOpKernelLibName().c_str(), graph_id);
        GELOGE(FAILED, "[Get][OpsKernelInfoStore] fail for op:%s(%s), kernel_lib_name:%s, graph_id:%u",
               op_desc->GetName().c_str(), op_desc->GetType().c_str(),
               op_desc->GetOpKernelLibName().c_str(), graph_id);
        return FAILED;
      }

      ret = kernel_info->CompileOp(node_vec);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call CompileOp fail for op:%s(%s), kernel_lib_name:%s, graph_id:%u, "
                          "check invalid", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                          op_desc->GetOpKernelLibName().c_str(), graph_id);
        GELOGE(ret, "[Compile][Op] failed, op = %s, graph_id = %u.", op_desc->GetName().c_str(), graph_id);
        return ret;
     }
    }
  }
  GM_RUN_AND_DUMP_PERF("Build", Build, graph_node, compute_graph, ge_root_model, session_id);
  graph_node->SetGeRootModel(ge_root_model);
  return SUCCESS;
}

Status GraphManager::BuildGraph(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                GeRootModelPtr &ge_root_model, uint64_t session_id, bool async) {
  GELOGD("[BuildGraph] start to build graph, graph_id:%u", graph_id);
  if (inputs.empty()) {
    GELOGW("[BuildGraph] BuildGraph warning: empty GeTensor inputs");
  }

  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetValidGraphNodeForBuild(this, graph_id, graph_node);
  if (ret != SUCCESS) {
    return ret;
  }
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  UpdateLocalOmgContext(graph_id);

  graph_node->SetAsync(async);
  // set graph's run flag
  graph_node->SetRunFlag(true);
  GE_MAKE_GUARD(run_flag_guard, [&graph_node] () { graph_node->SetRunFlag(false); });

  std::string refreshable = "0";  // default is not refreshable
  (void)GetContext().GetOption(OPTION_FEATURE_BASE_REFRESHABLE, refreshable);
  graph_node->SetFeatureBaseRefreshable(refreshable.compare("1") == 0);

  ret = StartForRunGraph(graph_node, inputs, ge_root_model, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][StartForRunGraph] failed! graph_id:%u.", graph_id);
    return ret;
  }

  GELOGI("[BuildGraph] build graph success, graph_id=%u, is feature refreshable:%d.",
         graph_id, graph_node->IsFeatureBaseRefreshable());
  return ret;
}

Status GraphManager::BuildGraphWithoutLoad(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                           GeRootModelPtr &ge_root_model, uint64_t session_id, bool async) {
  GELOGD("[BuildGraph] start to build graph, graph_id:%u", graph_id);
  if (inputs.empty()) {
    GELOGW("[BuildGraph] BuildGraph warning: empty GeTensor inputs");
  }

  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetValidGraphNodeForBuild(this, graph_id, graph_node);
  if (ret != SUCCESS) {
    return ret;
  }

  if (!IsGraphNeedBuild(graph_node)) {
    GELOGI("[Check][CompileFlag] Graph has built no need to build, graph_id = %u", graph_id);
    return SUCCESS;
  }
  if (graph_node->GetBuildFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u has not build before, can't run directly, check invalid",
                       graph_node->GetGraphId());
    GELOGE(PARAM_INVALID,
           "[Get][BuildFlag] The graph %u need to re-build, you should remove it from GE "
           "first, then AddGraph again and rebuild it.",
           graph_node->GetGraphId());
    return PARAM_INVALID;
  }

  UpdateLocalOmgContext(graph_id);
  graph_node->SetAsync(async);
  // set graph's run flag
  graph_node->SetRunFlag(true);
  GE_MAKE_GUARD(graph_node_guard, [&graph_node]() {
    graph_node->SetRunFlag(false);
  });

  std::string refreshable = "0";  // default is not refreshable
  (void)GetContext().GetOption(OPTION_FEATURE_BASE_REFRESHABLE, refreshable);
  graph_node->SetFeatureBaseRefreshable(refreshable.compare("1") == 0);
  ret = PreRun(graph_node, inputs, ge_root_model, session_id);
  graph_node->SetBuildFlag(true);
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  graph_rebuild_state_ctrl_->SetGraphBuildEnd(graph_node->GetGraphId());

  GE_CHECK_NOTNULL(graph_node->GetGraph());
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
  GE_CHECK_NOTNULL(compute_graph);
  int64_t graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
  (void)AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage);
  if (graph_stage == static_cast<int64_t>(GraphStage::GRAPH_STAGE_FUZZ)) {
    GELOGD("graph_stage:%d.", static_cast<int32_t>(graph_stage));
    return SUCCESS;
  }

  GELOGI("[BuildGraphWithoutLoad] build graph success, graph_id=%u, is feature refreshable:%d.",
         graph_id, graph_node->IsFeatureBaseRefreshable());
  return ret;
}

///
/// @ingroup ge_graph
/// @brief Save extra attribute to Model
/// @param [in] model: Model attribues will save to.
/// @param [in] type: type of OpDesc.
/// @param [in] attrs: attributes of OpDesc.
/// @param [in] inputs: inputs tensor.
/// @param [in] outputs: outputs tensor.
/// @return: Status
///
Status GraphManager::SaveParams(ge::GeModel &model, const std::string &type, const std::map<std::string,
    GeAttrValue> &attrs, const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs) const {
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetStr(&model, "ATTR_MODEL_OP_TYPE", type), return FAILED,
                   "[Set][Str] model type[%s] fail", type.c_str());

  for (const auto &it : attrs) {
    GE_CHK_BOOL_EXEC(model.SetAttr("ATTR_MODEL_" + it.first, it.second) == GRAPH_SUCCESS, return FAILED,
                     "[Set][Attr] OpDesc attribute[%s] fail", it.first.c_str());
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListTensor(&model, "ATTR_MODEL_TENSOR_INPUTS", inputs), return FAILED,
                   "[Set][InputsTensor] list fail");
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListTensor(&model, "ATTR_MODEL_TENSOR_OUTPUTS", outputs), return FAILED,
                   "[Set][OutputsTensor] list fail");

  return SUCCESS;
}

bool GraphManager::CheckModelLoad(const GeRootModelPtr &ge_root_model, bool load_flag) const {
  return ((ge_root_model != nullptr) && load_flag);
}

Status GraphManager::TryUnloadModel(GraphId graph_id, const GraphNodePtr &graph_node) {
  auto ge_root_model = graph_node->GetGeRootModel();
  if (CheckModelLoad(ge_root_model, graph_node->GetLoadFlag())) {
    // Free const memory before unload model
    auto mem_block = graph_node->GetConstMemBlock();
    if (mem_block != nullptr) {
      mem_block->Free();
      graph_node->SetConstMemBlock(nullptr);
    }
    // Free feature memory before unload model when ge.featureBaseRefreshable disabled
    mem_block = graph_node->GetFeatureMemBlock();
    if (mem_block != nullptr) {
      mem_block->Free();
      graph_node->SetFeatureMemBlock(nullptr);
    }

    Status middle_ret = UnloadModel(ge_root_model, graph_id);
    if (middle_ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "UnloadModel for graph:%u failed, check invalid", graph_id);
      GELOGE(middle_ret, "[Unload][Model] model failed, graph_id=%u.", graph_id);
      return middle_ret;
    }
  }
  return SUCCESS;
}
Status GraphManager::InnerRemoveGraph(const GraphId &graph_id) {
  std::set<GraphId>::const_iterator it = to_be_deleted_graphs_.find(graph_id);
  if (it != to_be_deleted_graphs_.cend()) {
    (void)to_be_deleted_graphs_.erase(it);
  }
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS || graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NOT_EXIST, "[Get][GraphNode] Id %u does not exist.", graph_id);
    return GE_GRAPH_GRAPH_NOT_EXIST;
  }
  if (graph_node->GetRunFlag()) {
    // only put graph into to-be-deleted list when exceptional scenario
    (void)to_be_deleted_graphs_.insert(graph_id);
    GELOGI("[GraphManager] Trying to remove running graph[Id:%u], added into to_be_deleted_graphs_.", graph_id);
    return SUCCESS;
  }

  std::lock_guard<std::mutex> lock(unload_model_mutex_);
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  graph_rebuild_state_ctrl_->RemoveGraph(graph_id);
  RemoveGraphNode(graph_id);
  GE_ASSERT_SUCCESS(TryUnloadModel(graph_id, graph_node));

  RemoveCompilerStages(graph_id);
  RemoveGraphCount(graph_id);
  RemoveAddGraphCondition(graph_id);
  GE_CHK_STATUS_RET(ret, "[Remove][Graph] failed, graph_id=%u.", graph_id);
  GELOGI("[GraphManager] remove graph success, graph_id=%u.", graph_id);
  return SUCCESS;
}

Status GraphManager::RemoveGraph(const GraphId &graph_id) {
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS || graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NOT_EXIST, "[Get][GraphNode] Id %u does not exist.", graph_id);
    return GE_GRAPH_GRAPH_NOT_EXIST;
  }
  GE_ASSERT_NOTNULL(graph_node);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  GE_ASSERT_SUCCESS(InnerRemoveGraph(graph_id));

  if (!graph_node->IsForkedGraph()) {
    graph_ids_to_forked_ids_.erase(graph_id);
    GELOGI("[GraphManager] remove origin graph success, graph_id=%u.", graph_id);
  } else {
    GELOGI("[GraphManager] remove forked graph success, graph_id=%u.", graph_id);
    graph_ids_to_forked_ids_[graph_node->GetOriginGraphId()].erase(graph_id);
  }
  return SUCCESS;
}

Status GraphManager::ParseOptions(const std::map<std::string, std::string> &options) {
  Status ret;

  ParseOption(options, "ge.INPUT_NODES_SET_FP16", options_.input_nodes_set_fp16);
  // parse streams max parallel num
  ret = ParseOption(options, STREAM_MAX_PARALLEL_NUM, options_.stream_max_parallel_num);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID,
           "[Parse][Option] %s value failed, it must be same format as DNN_V100:2,DNN_HCCL:3",
           STREAM_MAX_PARALLEL_NUM.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get stream num
  ret = ParseOption(options, STREAM_NUM, options_.stream_num);
  if ((ret != SUCCESS) || (options_.stream_num == 0)) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.stream_num, its value %d is invalid, "
           "must be not equal zero.", options_.stream_num);
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get perf level, its value please see enum PerfLevel
  ret = ParseOption(options, PERF_LEVEL, options_.perf_level);
  if ((ret != SUCCESS) || IsPerfLevelInvalid(options_.perf_level)) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.perfLevel, its value %d is invalid, "
           "must be enum PerfLevel type.", options_.perf_level);
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get encrypt mode
  ret = ParseOption(options, ENCRYPT_MODE, options_.encrypt_mode);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.encryptMode value invalid.");
                  return GE_GRAPH_OPTIONS_INVALID);

  // get ek file
  ParseOption(options, EK_FILE, options_.ek_file);

  // get cert file
  ParseOption(options, CERT_FILE, options_.cert_file);

  // get hw key file
  ParseOption(options, HW_KEY_FILE, options_.hw_key_file);

  // get private file
  ParseOption(options, PRIVATE_KEY_FILE, options_.private_key_file);

  // get framework type, its value please see enum FrameworkType
  ret = ParseOption(options, FRAMEWORK_TYPE, options_.framework_type);
  if (ret != SUCCESS) {
    // print error log in ParseOption
    return GE_GRAPH_OPTIONS_INVALID;
  }

  // get calibration info file
  ParseOption(options, CALIBRATION_CONF_FILE, options_.calibration_conf_file);

  // get insert op info file
  ParseOption(options, INSERT_OP_FILE, options_.insert_op_file);
  auto iter = options.find(kDynamicImageSize);
  if (iter != options.end()) {  // train
    options_.dynamic_image_size = iter->second;
  } else {                      // atc
    options_.dynamic_image_size = domi::GetContext().dynamic_image_size;
  }

  // get output node name
  ParseOption(options, OUTPUT_NODE_NAME, options_.output_node_name);

  // get function bin path
  ParseOption(options, "ge.func_bin_path", options_.func_bin_path);

  // get core type
  ParseOption(options, CORE_TYPE, options_.core_type);

  // get weight compress flag
  ret = ParseOption(options, COMPRESS_FLAG, options_.compress_flag);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.compressFlag value is invalid, "
                         "must be 0 or 1.");
                  return GE_GRAPH_OPTIONS_INVALID);
  // Set Build model and step
  ParseOption(options, BUILD_MODE, options_.build_mode);
  ParseOption(options, BUILD_STEP, options_.build_step);
  ParseOption(options, TUNING_PATH, options_.tuning_path);

  // ge.graphType.
  options_.run_graph_flag = true;
  ret = ParseOption(options, RUN_FLAG, options_.run_graph_flag);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.runFlag value is invalid, must be 0 or 1.");
                  return GE_GRAPH_OPTIONS_INVALID);

  ret = CheckPrecisionModeParamValid(options);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][PrecisionMode] failed.");
    return GE_GRAPH_OPTIONS_INVALID;
  }
  // ge.graphType
  ret = ParseTrainGraphFlag(options_.train_graph_flag);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][TrainGraphFlag] Key:ge.runFlag value is invalid");
                  return GE_GRAPH_OPTIONS_INVALID);

  // parse FmkOp
  options_.local_fmk_op_flag = false;
  ret = ParseOption(options, LOCAL_FMKOP_FLAG, options_.local_fmk_op_flag);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.localFmkopFlag value is invalid, "
                         "must be 0 or 1.");
                  return GE_GRAPH_OPTIONS_INVALID);
  options_.enable_print_op_pass = true;
  ret = ParseOption(options, ENABLE_PRINT_OP_PASS, options_.enable_print_op_pass);

  options_.is_single_op = false;
  ret = ParseOption(options, SINGLE_OP_FLAG, options_.is_single_op);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.enablePrintOpPass value is invalid, "
                         "must be 0 or 1.");
                  return GE_GRAPH_OPTIONS_INVALID);
  // parse hcom parallel
  options_.hcom_parallel = false;
  ret = ParseOption(options, HCOM_PARALLEL, options_.hcom_parallel);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][Option] Key:ge.hcomParallel value is invalid, "
                         "must be 0 or 1.");
                  return GE_GRAPH_OPTIONS_INVALID);
  // net output node dataType
  ParseOption(options, OUTPUT_DATATYPE, options_.output_datatype);

  // Set save_original_model flag (ge.save_original_model)
  ParseOption(options, SAVE_ORIGINAL_MODEL, options_.save_original_model);
  // Original model file name
  ParseOption(options, ORIGINAL_MODEL_FILE, options_.original_model_file);

  ParseOption(options, INPUT_SHAPE, options_.input_shape);
  ParseOption(options, kDynamicDims, options_.dynamic_dims);
  ParseOption(options, DYNAMIC_NODE_TYPE, options_.dynamic_node_type);
  ParseOption(options, EVENT, options_.event);
  GELOGD("Dynamic dims params: input shape is %s, dynamic dims is %s, dynamic node type is %d",
         options_.input_shape.c_str(), options_.dynamic_dims.c_str(), options_.dynamic_node_type);

  return SUCCESS;
}

// OPTION_GRAPH_RUN_MODE is supposed to be a session-level option, but it used to be set to global-level in the past.
// If can not parse from session, it can parse from global by GetContext().
Status GraphManager::ParseTrainGraphFlag(bool &train_flag) {
  train_flag = false;
  std::string run_mode;
  if (GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == SUCCESS && !run_mode.empty()) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, kBase)) >= TRAIN) {
      train_flag = true;
    }
  }
  domi::GetContext().train_flag = train_flag;
  GELOGI("train flag is %d.", train_flag);
  return SUCCESS;
}

bool GraphManager::IsPerfLevelInvalid(int32_t perf_level) {
  return ((perf_level != static_cast<int32_t>(PerfLevel::GEN_TASK_WITHOUT_L2FUSION)) &&
          (perf_level != static_cast<int32_t>(PerfLevel::GEN_TASK_WITHOUT_FUSION)) &&
          (perf_level != -1));
}

void GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                               std::string &option) {
  auto iter = options.find(key);
  if (iter != options.end()) {
    GELOGD("Set option %s from value:[%s] to value:[%s]", key.c_str(), option.c_str(), iter->second.c_str());
    option = iter->second;
  }
}

Status GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                                 bool &option) {
  auto iter = options.find(key);
  if (iter != options.end()) {
    std::string flag = iter->second;
    if (flag == "0") {
      option = false;
    } else if (flag == "1") {
      option = true;
    } else {
      REPORT_PREDEFINED_ERR_MSG("E10006", std::vector<const char *>({"parameter", "value"}),
                       std::vector<const char *>({key.c_str(), flag.c_str()}));
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] Key:%s, its value %s is invalid, it must be 0 or 1.",
             key.c_str(), flag.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  }
  return SUCCESS;
}

Status GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                                 int32_t &option) {
  const int32_t kDecimal = 10;
  char *ptr = nullptr;
  auto iter = options.find(key);
  if (iter != options.end()) {
    option = static_cast<int32_t>(std::strtol(iter->second.c_str(), &ptr, kDecimal));
    if (ptr != nullptr && *ptr != '\0') {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                       std::vector<const char *>({key.c_str(), iter->second.c_str(), "It is not int32_t type."}));
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] Key:%s, its value %s is invalid, must be int32_t type.",
             key.c_str(), iter->second.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  }
  return SUCCESS;
}

void GraphManager::Trim(std::string &str) {
  if (!str.empty()) {
    auto it = str.find_first_not_of(" ");
    if (it != std::string::npos) {
      (void)str.erase(0, it);
    }
    it = str.find_last_not_of(" ");
    if (it != std::string::npos) {
      (void)str.erase(it + 1);
    }
  }
}

Status GraphManager::ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                                 std::map<std::string, int32_t> &option) {
  auto iter = options.find(key);
  if (iter == options.end()) {
    return SUCCESS;
  }
  GELOGI("Start to parse %s", key.c_str());
  option.clear();
  std::string op_num = iter->second;

  // split std::string by ','
  std::vector<std::string> split;
  std::istringstream f(op_num);
  std::string str_tmp;
  while (getline(f, str_tmp, ',')) {
    split.push_back(str_tmp);
  }

  for (const std::string &engine_parallel : split) {
    // split engine and num by :
    size_t pos = engine_parallel.find(':');
    if (pos == std::string::npos) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                       std::vector<const char *>({key.c_str(), engine_parallel.c_str(),
                       "Engine and num must be connected by \":\"."}));
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] engine and num must be connected by :, "
             "while your input is %s", engine_parallel.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
    std::string engine_name = engine_parallel.substr(0, pos);
    std::string parallel_num = engine_parallel.substr(pos + 1);
    Trim(engine_name);
    Trim(parallel_num);

    Status ret = CheckEngineName(engine_name, key, option);
    if (ret != SUCCESS) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][EngineName] %s failed", engine_name.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }

    int32_t num = 0;
    ret = ParseParallelNum(parallel_num, key, num);
    if (ret != SUCCESS) {
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Parse][ParallelNum] %s failed", parallel_num.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }

    (void)option.insert(std::make_pair(engine_name, num));
  }
  GELOGI("Parse %s successfully", key.c_str());
  return SUCCESS;
}

Status GraphManager::CheckEngineName(const std::string &engine_name, const std::string &key,
                                     const std::map<std::string, int32_t> &option) {
  if (engine_name.empty()) {
    REPORT_PREDEFINED_ERR_MSG("E10004", std::vector<const char *>({"parameter"}),
                       std::vector<const char *>({engine_name.c_str()}));
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] engine name of %s is empty", key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  // judge whether exist in engine list
  GE_CHECK_NOTNULL(GELib::GetInstance());
  if (!GELib::GetInstance()->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
    GELOGW("engine : %s is not registered in %s", engine_name.c_str(), key.c_str());
  }

  auto it_stream_repeat = option.find(engine_name);
  if (it_stream_repeat != option.end()) {
    const auto readable_name = ge::GetContext().GetReadableName(key);
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({readable_name.c_str(), engine_name.c_str(), "Parameter engine_name is repeated."}));
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] engine:%s of %s is repeated", engine_name.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  return SUCCESS;
}

Status GraphManager::ParseParallelNum(const std::string &parallel_num, const std::string &key, int32_t &num) {
  if (parallel_num.empty()) {
    const auto readable_name = ge::GetContext().GetReadableName(key);
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({readable_name.c_str(), "parallel num", "Parameter parallel num is empty."}));
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] parallel num of %s is empty", key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  for (char c : parallel_num) {
    if (!isdigit(c)) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                         std::vector<const char *>({key.c_str(), parallel_num.c_str(),
                         "Parameter parallel num is not a digit."}));
      GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] Option:%s, param parallel num:%s is not digit, check invalid",
             key.c_str(), parallel_num.c_str());
      return GE_GRAPH_OPTIONS_INVALID;
    }
  }

  GE_CHK_STATUS_RET_NOLOG(ConvertToInt32(parallel_num, num));
  if (num < 1) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                       std::vector<const char *>({key.c_str(), parallel_num.c_str(),
                       "Parameter parallel num cannot be smaller than 1."}));
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] parallel num:%s of %s must bigger than 0",
           parallel_num.c_str(), key.c_str());
    return GE_GRAPH_OPTIONS_INVALID;
  }
  return SUCCESS;
}


void GraphManager::AddGraphNode(GraphId graph_id, const GraphNodePtr &graph_node) {
  std::lock_guard<std::mutex> lock(member_mutex_);
  graph_map_.emplace(graph_id, graph_node);
  graph_ids_.emplace_back(graph_id);
}

void GraphManager::RemoveGraphNode(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(member_mutex_);
  (void)graph_map_.erase(graph_id);
  (void)graph_ids_.erase(std::remove(graph_ids_.begin(), graph_ids_.end(), graph_id), graph_ids_.end());
}

bool GraphManager::HasGraphNode(GraphId graph_id) const {
  std::lock_guard<std::mutex> lock(member_mutex_);
  return graph_map_.find(graph_id) != graph_map_.end();
}

Status GraphManager::GetGraphNode(const GraphId &graph_id, GraphNodePtr &out) const {
  std::lock_guard<std::mutex> lock(member_mutex_);
  auto iter = graph_map_.find(graph_id);
  if (iter == graph_map_.end()) {
    out = nullptr;
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NOT_EXIST, "[Check][Param] graph does not exist, graph_id= %u.", graph_id);
    return GE_GRAPH_GRAPH_NOT_EXIST;
  }
  out = iter->second;
  return SUCCESS;
}

Status GraphManager::GetRunGraphMode(uint32_t graph_id, RunGraphMode &mode) const {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get run graph mode failed, graph_id: %u", graph_id);
  mode = graph_node->GetRunGraphMode();
  return SUCCESS;
}

Status GraphManager::SetRunGraphMode(uint32_t graph_id, const RunGraphMode &mode) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "set run graph mode failed, graph_id: %u", graph_id);
  graph_node->SetRunGraphMode(mode);
  return SUCCESS;
}

Status GraphManager::GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer) {
  GELOGI("Start to get the compiled model. graph_id: %u.", graph_id);
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  if (!graph_node->GetBuildFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is not compiled, graph_id:%u", graph_id);
    GELOGE(PARAM_INVALID, "[Check][CompileFlag] Graph is not compiled, graph_id:%u", graph_id);
    return PARAM_INVALID;
  }
  std::string options;
  (void)GetContext().GetOption("ge.exec.variable_acc", options);
  if (options == "True") {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::OPTION_EXEC_VARIABLE_ACC, "True",
        "This interface is incompatible with the configuration where 'option ge.exec.variable_acc' is set to \"True\","
        " Please set the option to \"False\" and try again."}));
    GELOGE(UNSUPPORTED, "This interface is incompatible with the configuration where"
        " 'option ge.exec.variable_acc' is set to \"True\". Please set the option to \"False\" and try again.");
    return UNSUPPORTED;
  }
  const auto ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model, "graph_id:%u", graph_id);
  return SaveRootModel(ge_root_model, model_buffer);
}

Status GraphManager::GetVariable(const std::string &name, Tensor &val) const {
  GeTensorPtr ge_tensor_ptr = TensorAdapter::AsGeTensorPtr(val);
  GE_CHECK_NOTNULL(ge_tensor_ptr);
  GE_CHECK_NOTNULL(GetGraphContext());
  return GetGraphContext()->GetVariableTensor(name, *(ge_tensor_ptr.get()));
}

Status GraphManager::SummaryHandle(const GraphId &graph_id, std::vector<gert::Tensor> &outputs) {
  std::vector<gert::Tensor> without_summary_outputs;
  std::set<int32_t> summary_output_index;
  GELOGI("[GraphManager] SummaryHandle, outputsSize=%zu.", outputs.size());
  const std::map<uint32_t, std::map<std::string, size_t>> &whole_summary_output_indexes =
      GetCompilerStages(graph_id).optimizer.GetSummaryOutputIndexes();
  if (whole_summary_output_indexes.find(graph_id) == whole_summary_output_indexes.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in whole_summary_output_indexes, check invalid", graph_id);
    GELOGE(FAILED, "[Check][Param] Graph:%u does not exist in whole_summary_output_indexes", graph_id);
    return FAILED;
  }
  const std::map<std::string, size_t> &summary_output_indexes = whole_summary_output_indexes.at(graph_id);
  GELOGI("[GraphManager] SummaryHandle, summaryOutputIndexesSize=%zu.", summary_output_indexes.size());
  std::map<std::string, gert::Tensor> summary_results;
  for (auto iter = summary_output_indexes.begin(); iter != summary_output_indexes.end(); ++iter) {
    GELOGI("[GraphManager] SummaryHandle, summaryName=%s, outputIndex=%zu.", iter->first.c_str(), iter->second);
    summary_results.emplace(iter->first, std::move(outputs.at(iter->second)));
    summary_output_index.emplace(iter->second);
  }

  // remove summary data from outputs
  if (!summary_output_index.empty()) {
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (summary_output_index.count(j) == 0) {
        without_summary_outputs.emplace_back(std::move(outputs.at(j)));
      }
    }
    outputs.swap(without_summary_outputs);
    GELOGI("[GraphManager] SummaryHandle, after swap outputsSize=%zu.", outputs.size());
  }

  if (!summary_results.empty()) {
    return PushSummaryData2ME(graph_id, summary_results);
  }

  return SUCCESS;
}

Status GraphManager::CheckpointHandle(const GraphId &graph_id, const ComputeGraphPtr &compute_graph,
                                      const std::vector<gert::Tensor> &outputs) {
  GELOGI("[GraphManager] CheckpointHandle, outputsSize=%zu.", outputs.size());

  GE_CHECK_NOTNULL(compute_graph);
  std::map<std::string, gert::Tensor> save_results;
  NodePtr netoutput = nullptr;
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == kNetOutput) {
      netoutput = node;
      break;
    }
  }
  if (netoutput == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "No netoutput node in graph:%u, check invalid", graph_id);
    GELOGE(FAILED, "[Check][Param] No netoutput node in graph:%u", graph_id);
    return FAILED;
  }
  for (const auto &in : netoutput->GetAllInDataAnchors()) {
    std::string desc_name;
    GE_CHECK_NOTNULL(in);
    auto out_anchor = in->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_anchor, "[Get][PeerOutAnchor] Peer anchor of op:%s(%s), in_index:%d is nullptr, graph_id:%u",
                      netoutput->GetNamePtr(), netoutput->GetTypePtr(), in->GetIdx(), graph_id);
    ge::NodePtr peer_node = out_anchor->GetOwnerNode();
    // find the variable node in graph
    while (peer_node != nullptr && peer_node->GetType() != kVariable) {
      if (peer_node->GetAllInDataAnchorsSize() != 1) {
        REPORT_INNER_ERR_MSG("E19999", "More than one prior nodes of peer_node:%s(%s) in checkpoint Graph:%u, "
                           "check invalid", peer_node->GetName().c_str(), peer_node->GetType().c_str(), graph_id);
        GELOGE(FAILED, "[Check][Param] More than one prior nodes of peer_node:%s(%s) in checkpoint Graph:%u.",
               peer_node->GetName().c_str(), peer_node->GetType().c_str(), graph_id);
        return FAILED;
      }
      auto peer_node_in = peer_node->GetAllInDataAnchors().at(0);
      GE_CHECK_NOTNULL(peer_node_in);
      auto peer_node_out_anchor = peer_node_in->GetPeerOutAnchor();
      if (peer_node_out_anchor != nullptr) {
        peer_node = peer_node_out_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(peer_node);
        if (peer_node->GetType() == kVariable) {
          break;
        }
      }
    }
    GE_ASSERT_NOTNULL(peer_node, "Peer anchor node of op:%s(%s), in_index:%d is nullptr, graph_id:%u, check invalid",
                      netoutput->GetNamePtr(), netoutput->GetTypePtr(), in->GetIdx(), graph_id);

    desc_name = peer_node->GetName();
    GELOGI("[GraphManager] CheckpointHandle, descName=%s.", desc_name.c_str());
    GE_ASSERT_TRUE(in->GetIdx() < static_cast<int32_t>(outputs.size()),
                   "[Check][Param] in index:%d of op:%s(%s) is out of outputs.size:%zu range, graph_id:%u",
                   in->GetIdx(), netoutput->GetName().c_str(), netoutput->GetTypePtr(), outputs.size(), graph_id);
    const auto &out_tensor = outputs.at(in->GetIdx());
    gert::Tensor copy_tensor{out_tensor.GetShape(), out_tensor.GetFormat(),
                             out_tensor.GetPlacement(), out_tensor.GetDataType(),
                             out_tensor.GetTensorData().GetAddr()};
    save_results.emplace(desc_name, std::move(copy_tensor));
  }

  if (!save_results.empty()) {
    return PushSaveData2ME(graph_id, save_results);
  }

  return SUCCESS;
}

Status GraphManager::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback) {
  std::unique_lock<std::shared_mutex> lock(callback_mutex_);
  GELOGI("[GraphManager] RegisterCallBackFunc, key=%s.", key.c_str());
  callback_map2_[key] = callback;
  return SUCCESS;
}

Status GraphManager::PushSummaryData2ME(const GraphId &graph_id,
                                        std::map<std::string, gert::Tensor> &summary_data) {
  std::shared_lock<std::shared_mutex> lock(callback_mutex_);
  GELOGI("[GraphManager] PushSummaryData2ME, dataSize=%zu.", summary_data.size());
  const auto iter = callback_map2_.find(kSummary);
  if (iter != callback_map2_.cend()) {
    std::map<AscendString, gert::Tensor> tmp_summary_data;
    for (auto &data : summary_data) {
      AscendString tmp(data.first.c_str());
      tmp_summary_data.emplace(tmp, std::move(data.second));
    }
    return iter->second(graph_id, tmp_summary_data);
  }
  REPORT_INNER_ERR_MSG("E19999", "No summary callback found, graph_id:%u, check invalid", graph_id);
  GELOGE(FAILED, "[Check][Param] No summary callback found, graph_id:%u", graph_id);
  return FAILED;
}

Status GraphManager::PushSaveData2ME(const GraphId &graph_id, std::map<std::string, gert::Tensor> &save_data) {
  std::shared_lock<std::shared_mutex> lock(callback_mutex_);
  GELOGI("[GraphManager] PushSaveData2ME, dataSize=%zu.", save_data.size());
  const auto iter = callback_map2_.find(kSave);
  if (iter != callback_map2_.cend()) {
    std::map<AscendString, gert::Tensor> tmp_save_data;
    for (auto &data : save_data) {
      AscendString tmp(data.first.c_str());
      tmp_save_data[tmp] = std::move(data.second);
    }
    return iter->second(graph_id, tmp_save_data);
  }
  GELOGW("[Check][Param] No checkpoint callback found, graph_id:%u", graph_id);
  return SUCCESS;
}

bool GraphManager::CheckNetOutputForCheckpointGraph(const NodePtr &node) const {
  GE_RT_FALSE_CHECK_NOTNULL(node);
  size_t in_data_anchor_size = node->GetAllInDataAnchorsSize();
  for (size_t i = 0; i < in_data_anchor_size; ++i) {
    auto in = node->GetInDataAnchor(i);
    if (in == nullptr) {
      return false;
    }
    auto peerin = in->GetPeerOutAnchor();
    GE_RT_FALSE_CHECK_NOTNULL(peerin);
    GE_RT_FALSE_CHECK_NOTNULL(peerin->GetOwnerNode());
    if (peerin->GetOwnerNode()->GetType() != kVariable && (!TransOpUtil::IsTransOp(peerin->GetOwnerNode()))) {
      return false;
    }
  }
  return true;
}

bool GraphManager::CheckVariableForCheckpointGraph(const NodePtr &node) const {
  // this func is for mindspore checkpoint graph
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) || node->GetOpDesc()->HasAttr(kCheckPointForGetVar)) {
    return false;
  }
  auto out = node->GetOutDataAnchor(0);
  if (out == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "anchor index:0 of op:%s(%s) is nullptr, check invalid",
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Get][OutDataAnchor] anchor index:0 of op:%s(%s) is nullptr",
           node->GetName().c_str(), node->GetType().c_str());
    return false;
  }
  auto peer_out = out->GetPeerInDataAnchors();
  for (size_t i = 0; i < peer_out.size(); ++i) {
    GE_RT_FALSE_CHECK_NOTNULL(peer_out.at(i));
    GE_RT_FALSE_CHECK_NOTNULL(peer_out.at(i)->GetOwnerNode());
    if (peer_out.at(i)->GetOwnerNode()->GetType() != kNetOutput &&
        (!TransOpUtil::IsTransOp(peer_out.at(i)->GetOwnerNode()))) {
      return false;
    }
  }
  return true;
}

bool GraphManager::CheckTransOpForCheckpointGraph(const NodePtr &node) const {
  GE_RT_FALSE_CHECK_NOTNULL(node);
  for (const auto &out_node : node->GetOutAllNodes()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_node);
    if ((!TransOpUtil::IsTransOp(out_node)) && (out_node->GetType() != kNetOutput) && (out_node->GetType() != kSend)) {
      return false;
    }
  }

  for (const auto &in_node : node->GetInAllNodes()) {
    GE_RT_FALSE_CHECK_NOTNULL(in_node);
    if ((!TransOpUtil::IsTransOp(in_node)) && (in_node->GetType() != kVariable) && (in_node->GetType() != kRecv)) {
      return false;
    }
  }
  return true;
}

static inline bool CheckConstanOpForCheckpointGraph(const NodePtr &node) { return node->GetOutDataNodes().empty(); }

bool GraphManager::IsCheckpointGraph(ComputeGraphPtr &compute_graph) {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] computeGraph is nullptr.");
    return false;
  }
  for (auto &node : compute_graph->GetAllNodes()) {
    GE_RT_FALSE_CHECK_NOTNULL(node);
    OpDescPtr op = node->GetOpDesc();
    GE_RT_FALSE_CHECK_NOTNULL(op);
    if (op->GetType() == kNetOutput) {
      if (!CheckNetOutputForCheckpointGraph(node)) {
        return false;
      }
    } else if (op->GetType() == kVariable) {
      if (!CheckVariableForCheckpointGraph(node)) {
        return false;
      }
    } else if ((TransOpUtil::IsTransOp(node))) {
      if (!CheckTransOpForCheckpointGraph(node)) {
        return false;
      }
    } else if (op->GetType() == CONSTANTOP) {
      if (!CheckConstanOpForCheckpointGraph(node)) {
        return false;
      }
    } else if (op->GetType() != kSend && op->GetType() != kRecv) {
      GELOGI("this node is not allow in checkpoint sub graph, node_type: %s, node_name: %s.", op->GetType().c_str(),
             op->GetName().c_str());
      return false;
    }
  }
  GELOGI("current graph:[%s] is checkpoint sub graph.", compute_graph->GetName().c_str());
  return true;
}

bool GraphManager::IsBroadCastOpData(const ge::NodePtr &var_node) const {
  GE_RT_FALSE_CHECK_NOTNULL(var_node);
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      ge::NodePtr dst_node = in_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(dst_node);
      if (dst_node->GetType() == HCOMBROADCAST || dst_node->GetType() == HVDCALLBACKBROADCAST) {
        return true;
      }
    }
  }
  return false;
}

void GraphManager::SetAttrForHcomBroadCastOp(ge::ComputeGraphPtr &compute_graph) {
  GE_RT_VOID_CHECK_NOTNULL(compute_graph);
  // add variable attr for hccl broadcast,need to be removed after variable pass online
  for (const ge::NodePtr &node : compute_graph->GetDirectNode()) {
    GE_RT_VOID_CHECK_NOTNULL(node);
    GE_RT_VOID_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetOpDesc()->GetType() != ge::VARIABLE) {
      continue;
    }
    if (IsBroadCastOpData(node)) {
      AdjustBroadCastOpData(node);
    }
    if (IsAssignOpData(node)) {
      AdjustAssignOpData(node);
    }
  }
}

void GraphManager::AdjustBroadCastOpData(const ge::NodePtr &var_node) const {
  GE_RT_VOID_CHECK_NOTNULL(var_node);
  if (!ge::AttrUtils::SetStr(var_node->GetOpDesc(), VAR_ATTR_VAR_IS_BROADCAST, "var_is_restore")) {
    GELOGW("set var_is_restore failed");
  }
}

bool GraphManager::IsAssignOpData(const ge::NodePtr &var_node) {
  GE_RT_FALSE_CHECK_NOTNULL(var_node);
  GELOGD("IsAssignOpData var_node:[%s]", var_node->GetName().c_str());
  std::map<std::string, std::set<int32_t>> assign_ops = {{ASSIGN, {0}}};

  ge::NodePtr assign_node = nullptr;
  if (ConfirmUseOpAndIndexByNode(var_node, assign_ops, assign_node)) {
    return true;
  }

  return false;
}

void GraphManager::AdjustAssignOpData(const ge::NodePtr &var_node) const {
  GE_RT_VOID_CHECK_NOTNULL(var_node);
  if (!ge::AttrUtils::SetStr(var_node->GetOpDesc(), VAR_ATTR_VAR_IS_RESTORE, "var_is_restore")) {
    GELOGW("SetStr var_is_restore failed");
  }
}

bool GraphManager::ConfirmUseOpAndIndexByAnchor(const InDataAnchorPtr &in_anchor,
                                                const std::map<std::string, std::set<int32_t>> &confirm_ops,
                                                NodePtr &use_node) const {
  GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
  ge::NodePtr dst_node = in_anchor->GetOwnerNode();
  GE_RT_FALSE_CHECK_NOTNULL(dst_node);
  ge::OpDescPtr dst_op_desc = dst_node->GetOpDesc();
  GE_RT_FALSE_CHECK_NOTNULL(dst_op_desc);
  const std::string &dst_type = dst_op_desc->GetType();
  int32_t input_index = in_anchor->GetIdx();

  GELOGD("ConfirmUseOpAndIndex, var name:[%s], dst_type:[%s], input index:[%d]", dst_node->GetName().c_str(),
         dst_type.c_str(), input_index);

  if (confirm_ops.count(dst_type) > 0) {
    if (confirm_ops.at(dst_type).count(input_index) > 0) {
      use_node = dst_node;
      return true;
    }
  }
  return false;
}

bool GraphManager::ConfirmUseOpAndIndexByNode(const NodePtr &var_node,
                                              const std::map<std::string, std::set<int32_t>> &confirm_ops,
                                              NodePtr &use_node) const {
  GE_RT_FALSE_CHECK_NOTNULL(var_node);
  for (auto &out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      if (ConfirmUseOpAndIndexByAnchor(in_anchor, confirm_ops, use_node)) {
        return true;
      }
    }
  }
  return false;
}

Status GraphManager::RemoveIsolatedConstInThisGraph(const ge::ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);
  for (ge::NodePtr &n : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(n);
    auto op_desc = n->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    if (op_desc->GetType() == CONSTANT || op_desc->GetType() == CONSTANTOP) {
      // reset const type depend on train_flag
      options_.train_graph_flag ? ge::OpDescUtilsEx::SetType(op_desc, CONSTANTOP)
                                : ge::OpDescUtilsEx::SetType(op_desc, CONSTANT);
      if (n->GetOutAllNodes().empty() && n->GetInAllNodes().empty()) {
        // it is an isolated constant, just remove it
        if (GraphUtils::RemoveJustNode(compute_graph, n) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Remove constant op:%s(%s) failed", n->GetName().c_str(), n->GetType().c_str());
          GELOGE(FAILED, "[Call][RemoveJustNode] remove constant %s failed.", n->GetName().c_str());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphManager::RemoveIsolatedConst(ge::ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHK_STATUS_RET(RemoveIsolatedConstInThisGraph(compute_graph));
  for (auto &sub_graph : compute_graph->GetAllSubgraphs()) {
    GE_CHK_STATUS_RET(RemoveIsolatedConstInThisGraph(sub_graph));
  }
  return SUCCESS;
}

Status GraphManager::OptimizeStage1(ge::ComputeGraphPtr &compute_graph) {
  PassManager after_merge_passes;
  GE_CHK_STATUS_RET(
      after_merge_passes.AddPass("OptimizeStage1_1::MergeInputMemcpyPass", new (std::nothrow) MergeInputMemcpyPass));
  GE_CHK_STATUS_RET(
      after_merge_passes.AddPass("OptimizeStage1_1::SwitchDataEdgesBypass", new (std::nothrow) SwitchDataEdgesBypass));
  GE_CHK_STATUS_RET(
      after_merge_passes.AddPass("OptimizeStage1_1::ConstantFuseSamePass", new (std::nothrow) ConstantFuseSamePass));
  /*
   * Do CSE before FuseDataNodesWithCommonInputPass to resolve the scene in bertlarge as following:
   *            const
   *    /        |        \
   * cast1      cast2     cast3
   *    \         |         /
   *             case
   * the node `const` is the fused const node after ConstantFuseSamePass
   * the nodes `cast1`, `cast2` and 'cast3' will be fused by CSE.
   * in order to eliminate hard code in FuseDataNodesWithCommonInputPass,
   * we do CSE before FuseDataNodesWithCommonInputPass
   * But it is a temp solution, this CSE will be deleted after change pass from graph pass to node pass
   */
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::CommonSubexpressionEliminationPass",
                                               new (std::nothrow) CommonSubexpressionEliminationPass));
  // FuseDataNodesWithCommonInputPass: fuse same data with common input in same graph
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::FuseDataNodesWithCommonInputPass",
                                               new (std::nothrow) FuseDataNodesWithCommonInputPass));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::CommonSubexpressionEliminationPass",
                                               new (std::nothrow) CommonSubexpressionEliminationPass));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::PermutePass", new (std::nothrow) PermutePass));
  /*
   * The SameTransdataBreadthFusionPass should be called before VariableOpPass, because of the scene following:
   *   node3
   *    |
   * transdata1   node2
   *    |         |
   *   cast1  transdata2
   *      \    /
   *        var
   * the node `transdata1` should be moved to the front of the ndoe `cast1`,
   * to ensure that `transdata1` and `transdata2` can be fusion with `var`.
   * But it is a temp solution, because the `SameTransdataBreadthFusionPass`
   * can only move `TransData` but not `Cast` nodes.
   * So if we exchange Cast and TransData, the fusion mechanism will fail.
   */
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::SameTransdataBreadthFusionPass",
                                               new (std::nothrow) SameTransdataBreadthFusionPass));
  std::string var_options;
  if (GetContext().GetOption("ge.exec.variable_acc", var_options) != SUCCESS) {
    GELOGI("get attr ge.exec.variable_acc, no value has been set. set default value.");
  }
  std::string parallel_option;
  (void)GetContext().GetOption(OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE, parallel_option);
  if (parallel_option == kIntegerEnableOption) {
    var_options = kBoolDisableOption;
    GELOGI("get option ge.AllowMultiGraphParallelCompile = \"1\", turn off VariableOpPass");
  }
  if (var_options != kBoolDisableOption) {
    GELOGI("turn on variable accelerator");
    GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::VariableOpPass",
                                                 new (std::nothrow) VariableOpPass(graph_rebuild_state_ctrl_.get())));
  }
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::TransOpWithoutReshapeFusionPass",
                                               new (std::nothrow) TransOpWithoutReshapeFusionPass));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::TransOpBreadthFusionPass",
                                               new (std::nothrow) TransOpBreadthFusionPass));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::DataFlowPreparePass",
                                               new (std::nothrow) DataFlowPreparePass));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage1_1::MergeUnknownShapeNPass",
                                               new (std::nothrow) MergeUnknownShapeNPass));

  GE_TRACE_START(after_merge_passes);
  auto ret = after_merge_passes.Run(compute_graph);
  GE_COMPILE_TRACE_TIMESTAMP_END(after_merge_passes, "GraphManager::OptimizeStage1_1");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "[Run][Passes] when OptimizeStage1_1 failed, ret:%u.", ret);
    return ret;
  }

  GE_DUMP(compute_graph, "OptimizeStage1_1");

  NamesToPass names_to_passes;
  TransOpNearbyAllreduceFusionPass trans_op_nearby_allreduce_fusion_pass;
  ConstantFoldingPass constant_folding_pass;
  ConstantClipPass constant_clip_pass;
  DimensionAdjustPass dimension_adjust_pass;
  EnterPass enter_pass;
  AddNPass addn_pass;
  SwitchDeadBranchElimination switch_dead_branch_elimination;
  SwitchLogicRemovePass switch_logic_remove_pass;
  MergePass merge_pass;
  CastRemovePass cast_remove_pass;
  TransposeTransDataPass transpose_transdata_pass;
  ReshapeRemovePass reshape_remove_pass;
  TransOpSymmetryEliminationPass symmetry_elimination_pass;
  DimensionComputePass dimension_compute_pass;
  UselessControlOutRemovePass useless_control_out_remove_pass;
  ReplaceWithEmptyConstPass replace_with_empty_const_pass;
  Dim1TransposeToSqueezePass dim1transpose_to_squeeze_pass;
  names_to_passes.emplace_back("EnterPass", &enter_pass);
  names_to_passes.emplace_back("AddNPass", &addn_pass);
  names_to_passes.emplace_back("SwitchDeadBranchElimination", &switch_dead_branch_elimination);
  names_to_passes.emplace_back("SwitchLogicRemovePass", &switch_logic_remove_pass);
  names_to_passes.emplace_back("MergePass", &merge_pass);
  names_to_passes.emplace_back("CastRemovePass", &cast_remove_pass);
  names_to_passes.emplace_back("TransposeTransDataPass", &transpose_transdata_pass);
  names_to_passes.emplace_back("ReshapeRemovePass", &reshape_remove_pass);
  names_to_passes.emplace_back("TransOpSymmetryEliminationPass", &symmetry_elimination_pass);
  names_to_passes.emplace_back("TransOpNearbyAllreduceFusionPass", &trans_op_nearby_allreduce_fusion_pass);
  names_to_passes.emplace_back("ReplaceWithEmptyConstPass", &replace_with_empty_const_pass);
  names_to_passes.emplace_back("DimensionComputePass", &dimension_compute_pass);
  names_to_passes.emplace_back("ConstantClipPass", &constant_clip_pass);
  names_to_passes.emplace_back("ConstantFoldingPass", &constant_folding_pass);
  names_to_passes.emplace_back("DimensionAdjustPass", &dimension_adjust_pass);
  names_to_passes.emplace_back("UselessControlOutRemovePass", &useless_control_out_remove_pass);
  names_to_passes.emplace_back("Dim1TransposeToSqueezePass", &dim1transpose_to_squeeze_pass);
  GE_TRACE_START(names_to_passes);
  ret = GEPass(compute_graph).Run(names_to_passes);
  GE_COMPILE_TRACE_TIMESTAMP_END(names_to_passes, "GraphManager::OptimizeStage1_2");
  GE_CHK_STATUS_RET(ret, "[Run][Passes] when OptimizeStage1_2 failed, ret:%u.", ret);

  // Calculate Op/Fe constantfolding cost
  uint64_t op_constant_folding_cost = 0;
  for (auto &it : constant_folding_pass.GetOpConstantFoldingPerfStatistic()) {
    GE_CHK_STATUS_RET(CheckUint64AddOverflow(op_constant_folding_cost, it.second.second));
    op_constant_folding_cost += it.second.second;
    GELOGI("The time cost of %s constant folding is [%lu] micro seconds, calls is %lu.",
           it.first.c_str(), it.second.second, it.second.first);
  }
  GEEVENT("[GEPERFTRACE] The time cost of extern constant folding is [%lu] micro seconds.", op_constant_folding_cost);
  for (auto &it : constant_folding_pass.GetGeConstantFoldingPerfStatistic()) {
    GE_CHK_STATUS_RET(CheckUint64AddOverflow(op_constant_folding_cost, it.second.second));
    op_constant_folding_cost += it.second.second;
    GELOGI("The time cost of %s constant folding is [%lu] micro seconds, calls is %lu.",
           it.first.c_str(), it.second.second, it.second.first);
  }

  GE_DUMP(compute_graph, "OptimizeStage1_2");
  PassManager graph_pass;
  // the prune pass should between SwitchPass and SwitchToStreamSwitchPass
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::SubgraphConstMigrationPass",
                                       new (std::nothrow) SubgraphConstMigrationPass));
  GE_CHK_STATUS_RET(
      graph_pass.AddPass("OptimizeStage1_3::UnusedArgsCleanPass", new (std::nothrow) UnusedArgsCleanPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::PrunePass", new (std::nothrow) PrunePass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::NextIterationPass", new (std::nothrow) NextIterationPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::ControlTriggerPass", new (std::nothrow) ControlTriggerPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::MergeToStreamMergePass",
                                       new (std::nothrow) MergeToStreamMergePass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::SwitchToStreamSwitchPass",
                                       new (std::nothrow) SwitchToStreamSwitchPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::AttachStreamLabelPass",
                                       new (std::nothrow) AttachStreamLabelPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::MultiBatchPass", new (std::nothrow) MultiBatchPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::SubgraphMultiDimsPass",
                                       new (std::nothrow) SubgraphMultiDimsPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::IteratorOpPass", new (std::nothrow) IteratorOpPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::VariableRefUselessControlOutDeletePass",
                                       new (std::nothrow) VariableRefUselessControlOutDeletePass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::ReshapeRecoveryPass",
                                       new (std::nothrow) ReshapeRecoveryPass));
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::RemoveSameConstPass",
                                       new (std::nothrow) RemoveSameConstPass));
  // [临时方案] 对于带_mutable_input属性的算子，需要在其与输入const之前插identity。因此将HcclMemcpyPass在此处再做一次。
  // [正式方案] 读写冲突pass需要放开在无子图场景下的处理。
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::HcclMemcpyPass", new (std::nothrow) HcclMemcpyPass));
  if (IsTailingOptimization()) {
    GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::HcclSequenceAdjustPass",
                                         new (std::nothrow) HcclSequenceAdjustPass));
  }
  if (options_.train_graph_flag) {
    // Priority: The GlobalStepInsertPass should work before graph partitioner.
    // Reason: Make sure that the var "global_step" can be partitioned to known sub graph and allocated memory
    // tmp solution: Skip this pass when single op. Remove this solution when MS remove run_mode_option in single op and
    // graph hybrid train process.
    bool is_single_op = false;
    (void)ge::AttrUtils::GetBool(compute_graph, ATTR_SINGLE_OP_SCENE, is_single_op);
    if (!is_single_op) {
      GE_CHK_STATUS_RET(
          graph_pass.AddPass("OptimizeStage1_3::GlobalStepInsertPass", new (std::nothrow) GlobalStepInsertPass));
    }

    std::string hccl_tailing_optimize;
    if (GetContext().GetOption("ge.exec.hccl_tailing_optimize", hccl_tailing_optimize) == SUCCESS &&
        hccl_tailing_optimize == "1") {
      GELOGI("Add hccl tailing optimize stage");
      GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::HcclTailingOptimizationPass",
                                           new (std::nothrow) HcclTailingOptimizationPass));
    }
  }
  GE_CHK_STATUS_RET(graph_pass.AddPass("OptimizeStage1_3::AttachedResourcePass",
                                       new (std::nothrow) AttachedResourcePass));
  GE_TRACE_START(graph_pass);
  ret = graph_pass.Run(compute_graph);
  GE_COMPILE_TRACE_TIMESTAMP_END(graph_pass, "GraphManager::OptimizeStage1_3");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "[Run][Passes] when OptimizeStage1_3 failed, ret:%u.", ret);
    return ret;
  }

  NamesToPass node_pass;
  GE_TRACE_START(node_pass);
  IdentityPass identity_force_pass(false);  // after SwitchToStreamSwitchPass
  node_pass.emplace_back("IdentityPass", &identity_force_pass);
  ret = GEPass(compute_graph).Run(node_pass);
  GE_COMPILE_TRACE_TIMESTAMP_END(node_pass, "GraphPrepare::node_pass");
  GE_CHK_STATUS_RET(ret, "[Run][Identity] remove pass for preprocess failed, ret:%u.", ret);
  return SUCCESS;
}

Status GraphManager::OptimizeStage2(ge::ComputeGraphPtr &compute_graph) {
  GELOGD("Start optimize after merge sub graph.");

  PassManager after_merge_passes;
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage2::InnerTensorMoveDeletePass",
                                               new (std::nothrow) InnerTensorMoveDeletePass));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage2::AfterMergePasses::LinkGenMaskNodesPass",
                                               new (std::nothrow)
                                                   LinkGenMaskNodesPass(options_.stream_max_parallel_num)));
  GE_CHK_STATUS_RET(after_merge_passes.AddPass("OptimizeStage2::HcclContinuousMemcpyPass",
                                               new (std::nothrow) HcclContinuousMemcpyPass));
  GE_TRACE_START(after_merge_passes);
  auto ret = after_merge_passes.Run(compute_graph);
  GE_COMPILE_TRACE_TIMESTAMP_END(after_merge_passes, "OptimizeStage2::AfterMergePasses");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "[Run][Passes] after merge sub graph failed, ret:%d.", ret);
    return ret;
  }
  SetAttrForHcomBroadCastOp(compute_graph);

  NamesToPass names_to_passes;
  ConstantFoldingPass constant_folding_pass;
  ReshapeRemovePass reshape_remove_pass;
  CondRemovePass condition_remove_pass;
  AssignRemovePass assign_remove_pass;
  InplaceSupportCheckPass inplace_support_check_pass;
  DimensionAdjustPass dimension_adjust_pass;

  names_to_passes.emplace_back("ConstantFoldingPass", &constant_folding_pass);
  names_to_passes.emplace_back("ReshapeRemovePass", &reshape_remove_pass);
  names_to_passes.emplace_back("CondRemovePass", &condition_remove_pass);
  names_to_passes.emplace_back("AssignRemovePass", &assign_remove_pass);
  names_to_passes.emplace_back("DimensionAdjustPass", &dimension_adjust_pass);
  if (GetContext().GetHostExecFlag()) {
    names_to_passes.emplace_back("InplaceSupportCheckPass", &inplace_support_check_pass);
  }
  GE_TRACE_START(names_to_passes);
  ret = GEPass(compute_graph).Run(names_to_passes);
  GE_COMPILE_TRACE_TIMESTAMP_END(names_to_passes, "OptimizeStage2::MergedGraphNameToPasses");
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][GEPasses] optimize for OptimizeAfterMergeSubGraph failed, ret:%d.", ret);
    return ret;
  }

  GE_TRACE_START(RemoveIsolatedConst);
  GE_CHK_STATUS_RET(RemoveIsolatedConst(compute_graph), "Failed to remove isolated const node");
  GE_COMPILE_TRACE_TIMESTAMP_END(RemoveIsolatedConst, "OptimizeStage2::RemoveIsolatedConst");

  PassManager pass_for_control_attr_optimize;
  if (options_.train_graph_flag) {
    GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::ControlAttrOptimize::FlowCtrlPass",
                                                             new (std::nothrow) FlowCtrlPass));
  }

  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::ControlAttrOptimize::MultiBatchPass",
                                                           new (std::nothrow) MultiBatchPass));
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::RefIdentityDeleteOpPass",
                                                           new (std::nothrow) RefIdentityDeleteOpPass));
  // the value of the attr is the original variable name the ref-variable ref from.
  // The attr will be used when allocating memory,
  // the node marked attr will be output to a variable instead of new-allocated memory.
  // Therefore, ComputeGraph should not delete nodes after `VariableRefDeleteOpPass`
  // to prevent unexpected deletion of nodes marked with attr
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::VariableRefDeleteOpPass",
                                                           new (std::nothrow) VariableRefDeleteOpPass));
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::ControlAttrOptimize::CompileNodesPass",
                                                           new (std::nothrow) CompileNodesPass));
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::SwapSpacePass",
                                                           new (std::nothrow) SwapSpacePass));
  GE_CHK_STATUS_RET(
      pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::InputOutputConnectionIdentifyPass",
                                             new (std::nothrow) InputOutputConnectionIdentifyPass));

  std::string recompute_mode;
  const std::string &kAutoRecompute = "auto";
  if ((GetContext().GetOption(RECOMPUTE, recompute_mode) == SUCCESS) && (recompute_mode == kAutoRecompute)) {
    GE_CHK_STATUS_RET(
        pass_for_control_attr_optimize.AddPass("OptimizeStage2::RecomputePass", new (std::nothrow) RecomputePass));
  }
  // When the input node to be cleared is after a `Data` node, the atomic-clean-node should not be inserted.
  // So The ComputeGraph should not delete nodes after `AtomicAddrCleanPass`
  // to prevent unexpected deletion of nodes after a `Data` node
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::AtomicAddrCleanPass",
                                                           new (std::nothrow) AtomicAddrCleanPass));
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::"
                                                           "EndOfSequenceAddControlPass",
                                                           new (std::nothrow) EndOfSequenceAddControlPass));
#ifdef FWK_SUPPORT_TRAINING_TRACE
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::AfterMergePasses::StartOfSequencePass",
                                                           new (std::nothrow) StartOfSequencePass));
#endif
  // 'SubgraphPass' solves memory_assign_conflicts by insert MemcpyAsync node, which depends on multi attrs and
  // graph-structure. Passes after 'SubgraphPass' MUST NOT remove MemcpyAsync/Identity nodes in subgraphs.
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::ControlAttrOptimize::SubgraphPass",
                                                           new (std::nothrow) SubgraphPass));
  // 'AttachStreamLabelPass' modifies attr without changing structure of compute_graph
  // All passes after 'AttachStreamLabelPass' MUST mark stream_label on new nodes by self.
  GE_CHK_STATUS_RET(pass_for_control_attr_optimize.AddPass("OptimizeStage2::ControlAttrOptimize::AttachStreamLabelPass",
                                                           new (std::nothrow) AttachStreamLabelPass(true)));
  GE_TRACE_START(pass_for_control_attr_optimize);
  ret = pass_for_control_attr_optimize.Run(compute_graph);
  GE_COMPILE_TRACE_TIMESTAMP_END(pass_for_control_attr_optimize, "OptimizeStage2::ControlAttrOptimize");
  if (ret != SUCCESS && ret != NOT_CHANGED) {
    GELOGE(ret, "[Run][Passes] when optimize stage 2 failed");
    return ret;
  }

  // Assign functional op labels.
  GE_TRACE_START(AssignFunctionalLabels);
  LabelAllocator label_allocator(compute_graph);
  GE_CHK_STATUS_RET(label_allocator.AssignFunctionalLabels(), "[Assign][Label] failed.");
  GE_COMPILE_TRACE_TIMESTAMP_END(AssignFunctionalLabels, "ModelBuilder::AssignFunctionalLabels");

  // Add memcpy addr asynchronous node.
  GE_TRACE_START(AddMemcpyAddrAsyncNode);
  MemcpyAddrAsyncPass memcpy_addr;
  GE_CHK_STATUS_RET(memcpy_addr.Run(compute_graph), "[Call][Run] Add memcpy_addr_async node failed.");
  GE_COMPILE_TRACE_TIMESTAMP_END(AddMemcpyAddrAsyncNode, "MemcpyAddrAsyncPass::Run");

  // Process offset and dependency for buffer pool memory assigner.
  PassManager pass_manager;
  GE_CHK_STATUS_RET(pass_manager.AddPass("BufferPoolMemoryPass", new (std::nothrow) BufferPoolMemoryPass));
  GE_RUN_PERF(BufferPoolMemoryPass, pass_manager.Run, compute_graph);

  // Handle parallel group .
  GE_TRACE_START(ParallelGroup);
  ParallelGroupPass parallel_group_pass;
  GE_CHK_STATUS_RET(parallel_group_pass.Run(compute_graph), "[Handle][ParallelGroup] failed.");
  GE_COMPILE_TRACE_TIMESTAMP_END(ParallelGroup, "ParallelGroupPass::Run");

  // 图稳定了再做ConcatNotaskPass,该pass放在Stage2的最后边
  PassManager concat_no_task_manager;
  GE_CHK_STATUS_RET(concat_no_task_manager.AddPass("OptimizeStage2::ConcatNotaskPass",
                                                           new (std::nothrow) ConcatNotaskPass));
  GE_RUN_PERF(ConcatNotaskPass, concat_no_task_manager.Run, compute_graph);
  GELOGI("End optimize after merge sub graph.");
  return SUCCESS;
}

Status GraphManager::MemConflictProc(ge::ComputeGraphPtr &compute_graph) {
  GE_TRACE_START(HandleMemoryRWConflict);
  // After while sub graph handle, mark all node rw type
  auto result = GetCompilerStages(compute_graph->GetGraphID()).optimizer.HandleMemoryRWConflict(compute_graph);
  if (result != SUCCESS) {
    GELOGW(
        "Mark node rw type failed. It will take some effect on memory_assign_conflicts handling."
        "Please pay attention to it.");
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(HandleMemoryRWConflict, "HandleMemoryRWConflict");

  GE_DUMP(compute_graph, "BeforeHandleMemoryLayoutConflict");
  GE_TRACE_START(HandleMemLayoutConflict);
  GE_CHK_STATUS_RET(GetCompilerStages(compute_graph->GetGraphID()).optimizer.HandleMemoryLayoutConflict(compute_graph),
                    "[Call][Run] memory layout conflict proc failed.");
  GE_COMPILE_TRACE_TIMESTAMP_END(HandleMemLayoutConflict, "MemLayoutConflictOptimizer::Run");

  PassManager last_passes;
  // SetFftsPlusAttrPass is to-be-deleted
  GE_CHK_STATUS_RET(last_passes.AddPass("OptimizeStage2::SetFftsPlusAttrPass", new (std::nothrow) SetFftsPlusAttrPass));
  GE_RUN_PERF(SetFftsPlusAttrPass, last_passes.Run, compute_graph);
  return SUCCESS;
}

void GraphManager::ChangeConstTypeWhenTraining(const ComputeGraphPtr &compute_graph) const {
  // The constant for train is CONSTANTOP, and is CONSTANT for inference. They will be unified in future.
  GE_RT_VOID_CHECK_NOTNULL(compute_graph);
  if (options_.train_graph_flag) {
    for (NodePtr &n : compute_graph->GetAllNodes()) {
      GE_RT_VOID_CHECK_NOTNULL(n);
      auto op_desc = n->GetOpDesc();
      GE_RT_VOID_CHECK_NOTNULL(op_desc);
      // This can ensure that n is not a null pointer
      if (op_desc->GetType() == CONSTANT) {
        ge::OpDescUtilsEx::SetType(op_desc, CONSTANTOP);
      }
    }
  }
}

Status GraphManager::ProcessSubGraphWithMultiThreads(GraphManager *graph_manager, GraphId root_graph_id,
                                                     const SubGraphInfoPtr &sub_graph_info_ptr,
                                                     const std::string &root_graph_name, uint64_t session_id,
                                                     const struct error_message::ErrorManagerContext &error_context,
                                                     const GEThreadLocalContext &ge_context, int32_t device_id) {
  GE_CHECK_NOTNULL(graph_manager, ", Param of graph_manager is null");
  GE_CHECK_NOTNULL(sub_graph_info_ptr, ", Param of sub_graph_info is null");
  error_message::SetErrMgrContext(error_context);
  GetContext().SetSessionId(session_id);
  GetThreadLocalContext() = ge_context;

  {
    if (device_id != kInvalidDeviceId) {
      GE_CHK_RT_RET(aclrtSetDevice(device_id));
    }
    GE_MAKE_GUARD(reset_device, [device_id]() {
      if (device_id != kInvalidDeviceId) {
        GE_CHK_RT(aclrtResetDevice(device_id));
      }
    });
    graph_manager->UpdateLocalOmgContext(root_graph_id);
    ComputeGraphPtr compute_graph_tmp = sub_graph_info_ptr->GetSubGraph();
    GE_CHECK_NOTNULL(compute_graph_tmp);
    const std::string &engine_name = sub_graph_info_ptr->GetEngineName();
    GELOGD("ProcessSubGraphWithMultiThreads start, graph name is %s, engine_name is %s, thread id is %lu",
           compute_graph_tmp->GetName().c_str(), engine_name.c_str(), pthread_self());
    GE_DUMP(compute_graph_tmp, engine_name + "_OptimizeSubGraphBefore");
    if (!AttrUtils::SetInt(*compute_graph_tmp, ATTR_NAME_ROOT_GRAPH_ID, root_graph_id)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to graph:%u failed", ATTR_NAME_ROOT_GRAPH_ID.c_str(),
                        compute_graph_tmp->GetGraphID());
      GELOGE(FAILED, "[Set][Attr] %s to graph:%u failed", ATTR_NAME_ROOT_GRAPH_ID.c_str(),
             compute_graph_tmp->GetGraphID());
      return FAILED;
    }
    if (!AttrUtils::SetStr(*compute_graph_tmp, ATTR_NAME_ROOT_GRAPH_NAME, root_graph_name)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to graph:%u failed", ATTR_NAME_ROOT_GRAPH_NAME.c_str(),
                        compute_graph_tmp->GetGraphID());
      GELOGE(FAILED, "[Set][Attr] %s to graph:%u failed", ATTR_NAME_ROOT_GRAPH_NAME.c_str(),
             compute_graph_tmp->GetGraphID());
      return FAILED;
    }
    compute_graph_tmp->SetSessionID(session_id);
    GE_TIMESTAMP_START(EngineOptimizeSubGraph);
    Status ret = graph_manager->GetCompilerStages(root_graph_id).optimizer.OptimizeSubGraph(compute_graph_tmp,
                                                                                            engine_name);
    GE_TIMESTAMP_END(EngineOptimizeSubGraph, engine_name.c_str());
    if (ret != SUCCESS) {
      GELOGE(ret, "[Optimize][SubGraph] Failed, engine:%s, graph:%s",
             engine_name.c_str(), compute_graph_tmp->GetName().c_str());
      return ret;
    }
    GELOGD("SubGraph optimize success %s", engine_name.c_str());
    GE_DUMP(compute_graph_tmp, engine_name + "_OptimizeSubGraphAfter");
    sub_graph_info_ptr->SetSubGraph(compute_graph_tmp);
    GELOGD("ProcessSubGraphWithMultiThreads end, graph name is %s, engine_name is %s, thread id is %lu",
           compute_graph_tmp->GetName().c_str(), engine_name.c_str(), pthread_self());
  }

  return SUCCESS;
}

// run graph async on session
Status GraphManager::RunGraphAsync(const GraphId &graph_id, std::vector<gert::Tensor> &&inputs,
    uint64_t session_id, const RunAsyncCallbackV2 &callback) {
  GELOGI("[GraphManager] Start to run graph async, graph_id=%u, inputsSize=%zu.", graph_id, inputs.size());
  GraphNodePtr graph_node = nullptr;
  const auto status = GetGraphNode(graph_id, graph_node);
  if ((status != SUCCESS) || (graph_node == nullptr)) {
    std::vector<gert::Tensor> outputs;
    callback(status, outputs);
    GELOGE(status, "run graph async failed, graph_id=%u", graph_id);
    return status;
  }
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  std::shared_ptr<RunArgs> args;
  GE_MAKE_SHARED(args = std::make_shared<RunArgs>(), return FAILED);
  GE_ASSERT_NOTNULL(args);
  args->graph_node = graph_node;
  args->graph_id = graph_id;
  args->session_id = session_id;
  args->error_context = error_message::GetErrMgrContext();
  args->input_tensor = std::move(inputs);
  args->context = GetThreadLocalContext();
  args->callback = callback;

  const bool ret = prerun_args_v2_q_.Push(args);
  GE_CHK_BOOL_RET_STATUS(ret, FAILED, "[Call][Push] failed, graph_id:%u.", graph_id);
  GELOGI("[GraphManager] run graph async submit success, graph_id=%u.", graph_id);
  return SUCCESS;
}

Status GraphManager::CheckIncreBuildAndPreRun(const std::shared_ptr<RunArgs> &args,
                                              GraphNodePtr &graph_node) {
  GE_CHECK_NOTNULL(graph_node);
  if (!IsGraphNeedBuild(graph_node)) {
    return SUCCESS;
  }
  if (graph_node->GetBuildFlag()) {
    ReturnError(args->callback, PARAM_INVALID,
                "[Check][Param] The graph " + std::to_string(graph_node->GetGraphId()) +
                " need to re-build, you should remove it"
                " from GE first, then AddGraph again and rebuild it.");
    return PARAM_INVALID;
  }

  std::vector<ge::Tensor> inputs_desc;
  for (const auto &gert_tensor : args->input_tensor) {
    const auto tensor_desc = GetTensorDescFromGertTensor(gert_tensor);
    inputs_desc.emplace_back(ge::Tensor(tensor_desc));
  }

  const auto ret = CompileGraph(graph_node->GetGraphId(),  args->session_id, inputs_desc);
  if (ret != SUCCESS) {
    ReturnError(args->callback, ret, "[Call][PreRun] Failed.");
    return ret;
  }
  graph_node->SetBuildFlag(true);
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  graph_rebuild_state_ctrl_->SetGraphBuildEnd(graph_node->GetGraphId());
  return SUCCESS;
}

void GraphManager::PreRunThreadV2() {
  if (prctl(PR_SET_NAME, ("ge_comp_prerunV2")) != 0) {
    GELOGW("Set thread name failed.");
  }

  std::shared_ptr<RunArgs> args;
  while (thread_run_flag_) {
    if (!prerun_args_v2_q_.Pop(args)) {
      continue;
    }
    const auto graph_id = args->graph_id;
    GraphNodePtr graph_node = args->graph_node;

    GELOGI("[PreRunThread] run graph async start, graph_id:%u.", graph_id);
    error_message::SetErrMgrContext(args->error_context);
    GetContext().SetSessionId(args->session_id);

    // more than one graph owns same graph_id
    uint32_t count = 0;
    if (GetGraphCount(graph_id, count) != SUCCESS) {
      ReturnError(args->callback, INTERNAL_ERROR,
                  "[Get][GraphCount] failed, graph_id=" + std::to_string(graph_id));
      GELOGE(INTERNAL_ERROR, "[Get][GraphCount] failed, graph id:%u.", graph_id);
      return;
    }
    // Avoid repeatively prerun for graphs owns same graph_id in online inference concurrency
    if (count > 1 && graph_node->GetBuildFlag()) {
      GELOGD("Avoid repeatively prerun, graph_id:%u.", graph_id);
      // In online inference concurrency senario, graph_node is allowed to be locked for 'count' times
      graph_node->SetSemSize(count);
      graph_node->Lock();
      PushRunArgs(args);
      GELOGI("[PreRunThread] Loop end. Start to run with cached build model.");
      continue;
    }

    GetThreadLocalContext() = args->context;
    UpdateLocalOmgContext(graph_id);
    // Cannot be put ahead of the repeatively prerun judgement
    graph_node->Lock();

    if (graph_node->GetRunFlag()) {
      ReturnError(args->callback, GE_GRAPH_ALREADY_RUNNING,
                  "[Check][Param] graph already running, graph id=" + std::to_string(graph_id));
      graph_node->Unlock();
      return;
    }

    if (graph_node->GetCompiledFlag()) {
      ReturnError(args->callback, GE_GRAPH_UNSUPPORTED,
                  "[Check][Compiled] Incompatible with API CompileGraph, graph_id=" +
                  std::to_string(graph_id));
      graph_node->Unlock();
      return;
    }

    if (graph_node->GetGraph() == nullptr) {
      ReturnError(args->callback, GE_GRAPH_GRAPH_NODE_NULL,
                  "[Get][Graph] from graph_node is nullptr");
      graph_node->Unlock();
      return;
    }
    auto ret = TranFrameOp(graph_node);
    if (ret != SUCCESS) {
      ReturnError(args->callback, GE_GRAPH_GRAPH_NODE_NULL,
                  "TranFrameOp failed.");
      graph_node->Unlock();
      return;
    }

    ret = CheckIncreBuildAndPreRun(args, graph_node);
    if (ret != SUCCESS) {
      if (!ge::Analyzer::GetInstance()->IsEnableNetAnalyzeDebug()) {
        GELOGE(ret, "[Check][IncreBuildAndPreRun] Failed, thread exit..");
        graph_node->Unlock();
        return;
      } else {
        GELOGE(ret, "[Check][IncreBuildAndPreRun] Failed, keep geop continue!");
        graph_node->Unlock();
        continue;
      }
    }
    // set graph's run flag
    graph_node->SetRunFlag(true);

    args->context = GetThreadLocalContext();
    PushRunArgs(args);
    GELOGI("[PreRunThread] Loop end.");
  }
}

void GraphManager::PushRunArgs(const std::shared_ptr<RunArgs> &args) const {
  if (executor_ == nullptr) {
    GELOGW("Just compile model, not support execute.");
    return;
  }

  (void)executor_->PushRunArgs(args);
}
Status GraphManager::SetRunContext(const GraphNodePtr &graph_node) {
  OmeContext ome_context;
  auto &omg_batch_shapes = GetLocalOmgContext().batch_shapes;
  if ((!GetLocalOmgContext().dynamic_dims.empty()) && (!omg_batch_shapes.empty())) {
    std::vector<std::vector<int32_t>> ome_batch_shapes;
    ome_batch_shapes.resize(omg_batch_shapes.size());
    for (size_t i = 0UL; i < omg_batch_shapes.size(); i++) {
      std::for_each(omg_batch_shapes[i].begin(), omg_batch_shapes[i].end(),
          [&ome_batch_shapes, i] (int64_t value) {
            ome_batch_shapes[i].emplace_back(static_cast<int32_t>(value));
          });
    }
    ome_context.dynamic_shape_dims = ome_batch_shapes;
  }
  ome_context.dynamic_node_type = GetLocalOmgContext().dynamic_node_type;
  ome_context.user_input_dims = GetLocalOmgContext().user_input_dims;

  ome_context.data_nodes = GetLocalOmgContext().data_nodes;
  ome_context.getnext_nosink_nodes = GetLocalOmgContext().getnext_nosink_nodes;
  ome_context.is_subgraph_multi_batch = GetLocalOmgContext().is_subgraph_multi_batch;

  GE_CHECK_NOTNULL(graph_node);
  graph_node->SetOmeContext(ome_context);
  return SUCCESS;
}

void GraphManager::StopQueue() {
  thread_run_flag_.store(false);
  prerun_args_v2_q_.Stop();
}
void GraphManager::ReturnError(RunAsyncCallbackV2 callback, Status ret,
  const std::string &log) {
  GELOGE(ret, "%s.", log.c_str());
  std::vector<gert::Tensor> outputs;
  if (callback != nullptr) {
    callback(ret, outputs);
  }
  StopQueue();
}

bool GraphManager::IsGraphNeedRebuild(uint32_t graph_id) {
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id:%u.", graph_id);
    return true;
  }

  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] graph node is NULL, graph_id:%u.", graph_id);
    return true;
  }

  return IsGraphNeedBuild(graph_node);
}

bool GraphManager::IsGraphNeedBuild(const GraphNodePtr &graph_node) const {
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  return !graph_node->GetBuildFlag() || graph_rebuild_state_ctrl_->IsGraphNeedRebuild(graph_node->GetGraphId());
}

bool GraphManager::GetLoadFlag(uint32_t graph_id) const {
  GraphNodePtr graph_node = nullptr;
  (void)GetGraphNode(graph_id, graph_node);
  return graph_node == nullptr ? false : graph_node->GetLoadFlag();
}

bool GraphManager::GetBuildFlag(uint32_t graph_id) const {
  GraphNodePtr graph_node = nullptr;
  (void)GetGraphNode(graph_id, graph_node);
  return graph_node == nullptr ? false : graph_node->GetBuildFlag();
}

Status GraphManager::GetCompiledFlag(uint32_t graph_id, bool &flag) const {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph node failed, graph_id:%u", graph_id);
  GE_ASSERT_NOTNULL(graph_node, nullptr);
  flag = graph_node->GetCompiledFlag();
  return SUCCESS;
}

Status GraphManager::SetCompiledFlag(uint32_t graph_id, bool flag) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node, nullptr);
  graph_node->SetCompiledFlag(flag);
  return SUCCESS;
}

const std::map<std::string, std::string> *GraphManager::GetGraphOptions(uint32_t graph_id) {
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id:%u.", graph_id);
    return nullptr;
  }

  if (!graph_node) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] graph node is NULL, graph_id:%u.", graph_id);
    return nullptr;
  }
  return &(graph_node->GetOptions());
}

void GraphManager::SetOptionsRunGraphFlag(bool run_graph_flag) { options_.run_graph_flag = run_graph_flag; }

Status GraphManager::OptimizeSubgraph(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                                      uint64_t session_id) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(compute_graph);
  // graph partition
  // Stage partition, only for root graph
  GE_DUMP(compute_graph, "BeforeStagePartition");
  GE_TRACE_START(StagePartition);
  TraceOwnerGuard guard("GE", "StagePartitioner", compute_graph->GetName());
  StagePartitioner stage_partitioner(compute_graph);
  auto ret = stage_partitioner.Partition();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][Partition] for Graph:%s by stage Failed", compute_graph->GetName().c_str());
    return ret;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(StagePartition, "OptimizeSubgraph::StagePartition");
  GE_DUMP(compute_graph, "AfterStagePartition");

  // all sub graph list of root graph and sub graph
  GE_TRACE_START(EnginePlacer1);
  TraceOwnerGuard guard2("GE", "DynamicShapePartitioner", compute_graph->GetName());
  auto &engine_placer = GetCompilerStages(graph_node->GetGraphId()).partitioner.GetEnginePlacer();
  engine_placer.SetComputeGraph(compute_graph);
  if (engine_placer.RunAllSubgraphs() != SUCCESS) {
    GELOGE(FAILED, "[Call][Run] Engine placer run failed, graph:%s.", compute_graph->GetName().c_str());
    return FAILED;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(EnginePlacer1, "OptimizeSubgraph::EnginePlacer1");
  GE_DUMP(compute_graph, "AfterEnginePlacer");

  // 提前执行HostcpuEngineUpdatePass，在动态图划分前标记host CPU引擎
  GE_TRACE_START(HostcpuEngineUpdatePass);
  TraceOwnerGuard guard_hostcpu("GE", "HostcpuEngineUpdatePass", compute_graph->GetName());
  GE_ASSERT_SUCCESS(engine_placer.RunHostcpuEngineUpdatePass());
  GE_COMPILE_TRACE_TIMESTAMP_END(HostcpuEngineUpdatePass, "OptimizeSubgraph::HostcpuEngineUpdatePass");
  GE_DUMP(compute_graph, "AfterHostcpuEngineUpdatePass");

  // DynamicShapePartition + EnginePlacer2
  GE_ASSERT_SUCCESS(DoDynamicShapePartition(graph_node, compute_graph));

  // SubgraphPartitionAndOptimization for CompositeEngine and AtomicEngine
  GE_ASSERT_SUCCESS(DoSubgraphPartitionWithMode(graph_node, compute_graph, session_id,
                    EnginePartitioner::Mode::kCompositeEnginePartitioning, "CompositeEngine"));
  GE_ASSERT_SUCCESS(DoSubgraphPartitionWithMode(graph_node, compute_graph, session_id,
                    EnginePartitioner::Mode::kAtomicEnginePartitioning, "AtomicEngine"));

  GE_ASSERT_SUCCESS(VerifyCommNodesOrderAfterEngineAssigned(compute_graph));
  return SUCCESS;
}

Status GraphManager::SubgraphPartitionAndOptimization(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                                                      uint64_t session_id, EnginePartitioner::Mode mode) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(compute_graph);
  bool is_single_op = false;
  (void)ge::AttrUtils::GetBool(compute_graph, ATTR_SINGLE_OP_SCENE, is_single_op);
  const bool is_use_composite_engine = (mode == EnginePartitioner::Mode::kCompositeEnginePartitioning) &&
      (OpsKernelManager::GetInstance().GetCompositeEngines().empty() || is_single_op);
  if (is_use_composite_engine) {
    GELOGI("No composite engine registers or single-op use ffts plus[flag:%d], "
           "ignore subgraph partition and optimization for composite engine", is_single_op);
    return SUCCESS;
  }

  GE_TRACE_START(GraphPartition);
  TraceOwnerGuard guard("GE", "EnginePartitioner", compute_graph->GetName());
  EnginePartitioner &partitioner = GetCompilerStages(graph_node->GetGraphId()).partitioner;
  Status ret = partitioner.Partition(compute_graph, mode);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][Partition] for Graph:%s Failed", compute_graph->GetName().c_str());
    return ret;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphPartition, "OptimizeSubgraph::Partition1");

  auto &optimizer = GetCompilerStages(compute_graph->GetGraphID()).optimizer;
  GE_TRACE_START(SetSubgraphPreProc);
  GE_ASSERT_SUCCESS(optimizer.OptimizeSubgraphPreProc(*compute_graph),
                    "[Set][SubgraphPreProc] failed for graph:%s, session_id:%lu",
                    compute_graph->GetName().c_str(), session_id);
  GE_COMPILE_TRACE_TIMESTAMP_END(SetSubgraphPreProc, "OptimizeSubgraph::SetSubgraphPreProc");
  GE_DUMP(compute_graph, "OptimizeSubgraphPreProc");

  GE_TRACE_START(SetSubgraph);
  TraceOwnerGuard guard1("GE", "SetSubgraph", compute_graph->GetName());
  ret = SetSubgraph(session_id, compute_graph, partitioner);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Subgraph] failed for graph:%s, session_id:%lu", compute_graph->GetName().c_str(), session_id);
    return ret;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(SetSubgraph, "OptimizeSubgraph::SetSubGraph");

  GE_TRACE_START(SetSubgraphPostProc);
  GE_ASSERT_SUCCESS(optimizer.OptimizeSubgraphPostProc(*compute_graph),
                    "[Set][SubgraphPostProc] failed for graph:%s, session_id:%lu",
                    compute_graph->GetName().c_str(), session_id);
  GE_COMPILE_TRACE_TIMESTAMP_END(SetSubgraphPostProc, "OptimizeSubgraph::SetSubgraphPostProc");
  GE_DUMP(compute_graph, "OptimizeSubgraphPostProc");

  if (mode == EnginePartitioner::Mode::kAtomicEnginePartitioning) {
    std::set<std::string> build_steps = {BUILD_STEP_BEFORE_UB_MATCH, BUILD_STEP_AFTER_BUILDER,
                                         BUILD_STEP_AFTER_BUILDER_SUB};
    if ((GetBuildMode(graph_node) == BUILD_MODE_TUNING) && (build_steps.count(GetBuildStep(graph_node)) > 0)) {
      GE_TRACE_START(ConvertGraphToFile);
      std::string tuning_path;
      (void) GetContext().GetOption(TUNING_PATH, tuning_path);
      ret = ConvertGraphToFile(compute_graph, partitioner, tuning_path,
                               (GetBuildStep(graph_node) == BUILD_STEP_AFTER_BUILDER));
      if (ret != SUCCESS) {
        GELOGE(ret, "[Convert][Graph] [%s] to file failed", compute_graph->GetName().c_str());
        return ret;
      }
      GE_COMPILE_TRACE_TIMESTAMP_END(ConvertGraphToFile, "OptimizeSubgraph::ConvertGraphToFile");
      return SUCCESS;
    }
  }
  ComputeGraphPtr merged_compute_graph = nullptr;
  GE_TRACE_START(MergeSubgraph);
  TraceOwnerGuard guard2("GE", "MergeSubGraph", compute_graph->GetName());
  ret = MergeSubGraph(merged_compute_graph, compute_graph, graph_node->GetGraphId(), mode);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Merge][SubGraph] Failed, graph:%s(id:%u)",
           compute_graph->GetName().c_str(), graph_node->GetGraphId());
    return ret;
  }
  GE_CHECK_NOTNULL(merged_compute_graph);
  merged_compute_graph->SetSessionID(session_id);
  merged_compute_graph->SetGraphID(graph_node->GetGraphId());
  merged_compute_graph->SetNeedIteration(compute_graph->GetNeedIteration());
  merged_compute_graph->SetSummaryFlag(compute_graph->IsSummaryGraph());
  for (auto &sub_graph : merged_compute_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    sub_graph->SetSessionID(session_id);
    sub_graph->SetGraphID(graph_node->GetGraphId());
  }

  GE_COMPILE_TRACE_TIMESTAMP_END(MergeSubgraph, "OptimizeSubgraph::MergeSubGraph");
  GE_DUMP(merged_compute_graph, "mergedComputeGraph");
  compute_graph = merged_compute_graph;
  GELOGD("graph [%s] merge finished.", compute_graph->GetName().c_str());
  return SUCCESS;
}

Status GraphManager::ConvertGraphToFile(ComputeGraphPtr &compute_graph,
                                        EnginePartitioner &partitioner,
                                        std::string path,
                                        bool exe_flag) const {
  GE_CHECK_NOTNULL(compute_graph);
  GELOGI("compute_graph [%s] path [%s] Enter ConvertGraphToFile.", compute_graph->GetName().c_str(), path.c_str());
  std::vector<ComputeGraphPtr> non_tuning_subgraphs;
  auto input_node_sub_graph_map = partitioner.graph_2_input_subgraph_;
  auto iter = input_node_sub_graph_map.find(compute_graph);
  if (iter != input_node_sub_graph_map.end()) {
    GE_CHECK_NOTNULL(iter->second);
    non_tuning_subgraphs.push_back(iter->second->GetSubGraph());
  }
  auto sub_graph_map = partitioner.GetSubGraphMap();
  const auto &subgraph_infos = sub_graph_map[compute_graph];
  std::vector<ComputeGraphPtr> tuning_subgraphs;
  for (const auto &sub_graph_info_ptr: subgraph_infos) {
    GE_CHECK_NOTNULL(sub_graph_info_ptr);
    ComputeGraphPtr sub_graph_tmp = sub_graph_info_ptr->GetSubGraph();
    // need to tuning
    if (sub_graph_info_ptr->GetEngineName() == kVectorEngine || sub_graph_info_ptr->GetEngineName() == kAIcoreEngine) {
      tuning_subgraphs.push_back(sub_graph_tmp);
    } else {
      non_tuning_subgraphs.push_back(sub_graph_tmp);
    }
  }
  // for function graphs to tune
  for (auto &function_graph : compute_graph->GetAllSubgraphs()) {
    auto subgraph_list = sub_graph_map[function_graph];
    auto it = input_node_sub_graph_map.find(function_graph);
    if (it != input_node_sub_graph_map.end()) {
      GE_CHECK_NOTNULL(it->second);
      non_tuning_subgraphs.push_back(it->second->GetSubGraph());
    }

    for (const auto &sub_graph_info_ptr : subgraph_list) {
      GE_CHECK_NOTNULL(sub_graph_info_ptr);
      ComputeGraphPtr sub_graph_tmp = sub_graph_info_ptr->GetSubGraph();
      // need to tuning
      if (sub_graph_info_ptr->GetEngineName() == kVectorEngine ||
          sub_graph_info_ptr->GetEngineName() == kAIcoreEngine) {
        tuning_subgraphs.push_back(sub_graph_tmp);
      } else {
        non_tuning_subgraphs.push_back(sub_graph_tmp);
      }
    }
  }
  return TuningUtils::ConvertGraphToFile(tuning_subgraphs, non_tuning_subgraphs, exe_flag, path);
}

Status GraphManager::ComputeHashForConstNodes(const ComputeGraphPtr &compute_graph) {
  ThreadPool thread_pool("ge_hashcnst", kDefaultThreadNum, false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (!NodeUtils::IsConst(*node)) {
      continue;
    }
    const auto &weights = OpDescUtils::MutableWeights(node);
    if (weights.empty() || weights[0]->GetTensorDesc().GetShape().IsEmptyTensor()) {
      GELOGW("Node:%s weight is null or empty tensor", node->GetName().c_str());
      continue;
    }
    const auto &weight = weights[0];
    auto fut = thread_pool.commit([node, weight]() -> Status {
      const auto &data = weight->GetData().GetData();
      const auto size = weight->GetData().GetSize();
      unsigned char sha256[SHA256_DIGEST_LENGTH];
      (void)SHA256(PtrToPtr<uint8_t, unsigned char>(data), size, sha256);
      std::stringstream ss;
      for (const auto &item : sha256) {
        ss << std::hex << static_cast<int32_t>(item);
      }
      GE_ASSERT_TRUE(AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_WEIGHT_SHA256, ss.str()),
                     "Failed to set _value_sha256 for const node:%s", node->GetName().c_str());
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }
  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to set compute hash for const nodes");
  }
  return SUCCESS;
}

Status GraphManager::Build(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                           GeRootModelPtr &ge_root_model, uint64_t session_id) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(compute_graph);

  std::string external_weight = std::to_string(0);
  (void)GetContext().GetOption(EXTERNAL_WEIGHT, external_weight);
  const std::set<std::string> valid_options = {"", "0", "1", "2"};
  GE_ASSERT_TRUE(valid_options.count(external_weight) > 0U, "Invalid option value: %s = %s, only support empty, 0, 1 or 2",
                 EXTERNAL_WEIGHT.c_str(), external_weight.c_str());
  GELOGI("Get option value success, %s = %s", EXTERNAL_WEIGHT.c_str(), external_weight.c_str());
  if (external_weight == kExternalWeightEnabled || external_weight == kExternalWeightCombined) {
    GE_TRACE_START(ComputeHashForConstNodes);
    GE_CHK_STATUS_RET(ComputeHashForConstNodes(compute_graph), "Failed to compute hash for const nodes");
    GE_COMPILE_TRACE_TIMESTAMP_END(ComputeHashForConstNodes, "GraphManager::ComputeHashForConstNodes");
    GE_TRACE_START(ConvertConstToFileConst);
    GE_CHK_STATUS_RET(FileConstantUtils::ConvertConstToFileConst(compute_graph, external_weight == kExternalWeightCombined),
                      "Failed to convert const to fileconstant.");
    GE_COMPILE_TRACE_TIMESTAMP_END(ConvertConstToFileConst, "FileConstantUtils::ConvertConstToFileConst");
  }

  // rt2执行需要确定性计算字段
  std::string deterministic_str;
  (void)ge::GetContext().GetOption(ge::DETERMINISTIC, deterministic_str);
  const int32_t deterministic = (deterministic_str == "1") ? 1 : 0;
  GE_ASSERT_TRUE(AttrUtils::SetInt(compute_graph, ge::DETERMINISTIC, deterministic));
  int32_t deterministic_level = 0;
  GE_ASSERT_SUCCESS(optiling::GetDeterministicLevel(deterministic_level));
  GE_ASSERT_TRUE(AttrUtils::SetInt(compute_graph, "ge.deterministicLevel", deterministic_level));

  // After recover, the node and the IR definition will not be changed
  GE_TRACE_START(RecoverIrDefinitionAndModifyAippData);
  GE_CHK_GRAPH_STATUS_RET(ge::RecoverIrDefinitions(compute_graph, std::vector<std::string>{kUbOriginGraphAttrKey}),
                          "Failed to recover ir definitions");
  GE_CHK_STATUS_RET(ModifyAippData(compute_graph));
  GE_COMPILE_TRACE_TIMESTAMP_END(RecoverIrDefinitionAndModifyAippData, "GraphManager::RecoverIrDefinitionAndModifyAippData");

  auto ret = GetCompilerStages(graph_node->GetGraphId()).builder.Build(compute_graph, ge_root_model, session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][Build] failed, session_id:%lu.", session_id);
    return ret;
  }

  bool is_always_dump = false;
  if (!DumpManager::GetInstance().GetDumpProperties(session_id).GetDumpPath().empty()) {
    is_always_dump = true;
  }

  GraphUtils::DumpGEGraph(compute_graph, "Build", is_always_dump);
  GraphUtils::DumpGEGraphToOnnx(*compute_graph, "Build");
  GraphUtils::DumpGEGraphToReadable(compute_graph, "Build");
  GE_ASSERT_SUCCESS(VerifyCommNodesOrderAfterBuild(compute_graph));
  return SetRunContext(graph_node);
}

Status GraphManager::GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables,
                                        Graph &graph) const {
  ge::ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>(kCheckPointGraph);
  GE_CHECK_NOTNULL(compute_graph);
  OpDescPtr save_desc = MakeShared<ge::OpDesc>(compute_graph->GetName() + "_" + kSave, kSave);
  GE_CHECK_NOTNULL(save_desc);
  uint32_t save_index = 0;
  for (auto iter = all_variables.begin(); iter != all_variables.end(); ++iter) {
    GE_CHK_GRAPH_STATUS_RET(save_desc->AddInputDesc(save_index, iter->second));
    save_index++;
  }
  NodePtr save_node = compute_graph->AddNode(save_desc);

  uint32_t index = 0;
  for (auto iter = all_variables.begin(); iter != all_variables.end(); ++iter) {
    OpDescPtr var_desc = MakeShared<ge::OpDesc>(iter->first, VARIABLE);
    GE_CHECK_NOTNULL(var_desc);
    if (!AttrUtils::SetBool(var_desc, kCheckPointForGetVar, true)) {
      GELOGW("Set check point graph attr failed.");
    }
    GE_CHK_GRAPH_STATUS_RET(var_desc->AddOutputDesc(iter->second));
    NodePtr var_node = compute_graph->AddNode(var_desc);
    GE_ASSERT_NOTNULL(var_node);
    GE_ASSERT_NOTNULL(save_node);
    GE_CHK_STATUS(GraphUtils::AddEdge(var_node->GetOutDataAnchor(0), save_node->GetInDataAnchor(index)),
                  "[Add][Edge][%s->%s] fail.", var_node->GetName().c_str(), save_node->GetName().c_str());
    index++;
  }
  compute_graph->Dump();
  graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  return SUCCESS;
}

Status GraphManager::SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                                   const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values) const {
  std::map<std::string, Tensor> var_results;
  GE_CHK_STATUS_RET(SaveCheckPointResult(graph, outputs, var_results), "[Save][CheckPointResult] failed.");
  if (!var_names.empty()) {
    for (const auto &var_name : var_names) {
      if (var_results.count(var_name) == 0) {
        REPORT_INNER_ERR_MSG("E19999", "Fetch Var:%s result value fail", var_name.c_str());
        GELOGE(FAILED, "[Check][Param] Fetch var[%s] value failed.", var_name.c_str());
        return FAILED;
      } else {
        auto var_tensor = var_results[var_name].GetTensorDesc();
        var_tensor.SetName(var_name.c_str());
        var_results[var_name].SetTensorDesc(var_tensor);
        var_values.emplace_back(var_results[var_name]);
      }
    }
  } else {
    for (auto iter = var_results.begin(); iter != var_results.end(); ++iter) {
      std::string var_name = iter->first;
      auto var_tensor = iter->second.GetTensorDesc();
      var_tensor.SetName(var_name.c_str());
      iter->second.SetTensorDesc(var_tensor);
      var_values.emplace_back(iter->second);
    }
  }
  return SUCCESS;
}

static Status FindVarNodeFromNetoutputIn(NodePtr &peer_node) {
  while (peer_node->GetType() != VARIABLE) {
    if (peer_node->GetAllInDataAnchorsSize() != 1) {
      REPORT_INNER_ERR_MSG("E19999", "peer node:%s(%s) of netoutput has more than 1 input in checkpoint Graph, "
                         "check invalid", peer_node->GetName().c_str(), peer_node->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] node:%s has more than 1 input.", peer_node->GetName().c_str());
      return FAILED;
    }
    auto peer_node_in_anchor = peer_node->GetAllInDataAnchors().at(0);
    GE_CHECK_NOTNULL(peer_node_in_anchor);
    auto peer_node_out_anchor = peer_node_in_anchor->GetPeerOutAnchor();
    if (peer_node_out_anchor != nullptr) {
      peer_node = peer_node_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      if (peer_node->GetType() == VARIABLE) {
        break;
      }
    }
  }
  return SUCCESS;
}

Status GraphManager::SaveCheckPointResult(const Graph &graph, const std::vector<Tensor> &outputs,
                                          std::map<std::string, Tensor> &var_results) const {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  NodePtr netoutput_node = nullptr;
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == NETOUTPUT) {
      netoutput_node = node;
      break;
    }
  }
  GE_CHECK_NOTNULL(netoutput_node);
  for (const auto &in : netoutput_node->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in);
    auto out_anchor = in->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(out_anchor);
    auto peer_node = out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (FindVarNodeFromNetoutputIn(peer_node) != SUCCESS) {
      return FAILED;
    }
    if (peer_node->GetType() != VARIABLE) {
      REPORT_INNER_ERR_MSG("E19999", "peer node:%s(%s) of netoutput is not variable in checkpoint Graph, "
                         "check invalid", peer_node->GetName().c_str(), peer_node->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] peer_node %s is not variable in checkpoint Graph.", peer_node->GetName().c_str());
      return FAILED;
    }
    auto var_name = peer_node->GetName();
    GELOGI("[GraphManager] SaveVariables, varName is %s.", var_name.c_str());
    GE_ASSERT_TRUE(in->GetIdx() < static_cast<int32_t>(outputs.size()),
                   "In index:%d of output node is out of outputs.size:%zu range in checkpoint Graph, "
                   "check invalid",
                   in->GetIdx(), outputs.size());
    var_results.emplace(var_name, outputs.at(in->GetIdx()));
  }
  return SUCCESS;
}

void GraphManager::AddLocalOmgContext(GraphId graph_id, const OmgContext &omg_context) {
  std::lock_guard<std::mutex> lock(member_mutex_);
  omg_contexts_.emplace(graph_id, omg_context);
  SetLocalOmgContext(omg_contexts_[graph_id]);
}

void GraphManager::UpdateLocalOmgContext(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(member_mutex_);
  auto iter = omg_contexts_.find(graph_id);
  if (iter != omg_contexts_.end()) {
    SetLocalOmgContext(iter->second);
  } else {
    GELOGW("OmgContext of graph %u is not found.", graph_id);
  }
}

GraphManager::CompilerStages &GraphManager::GetCompilerStages(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(member_mutex_);
  return compiler_stages_[graph_id];
}

void GraphManager::RemoveCompilerStages(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(member_mutex_);
  (void)compiler_stages_.erase(graph_id);
}

void GraphManager::IncreaseGraphCount(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(graph_count_mutex_);
  std::map<GraphId, uint32_t>::const_iterator it = graph_count_.find(graph_id);
  if (it == graph_count_.cend()) {
    (void)graph_count_.insert({graph_id, kInitGraphCount});
  } else {
    if (CheckUint32AddOverflow(graph_count_[graph_id], 1) == SUCCESS) {
      ++graph_count_[graph_id];
    }
  }
  GELOGD("After increaseGraphCount, graph count of id[%u] is %u.", graph_id, graph_count_[graph_id]);
}

void GraphManager::RemoveGraphCount(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(graph_count_mutex_);
  const auto it = graph_count_.find(graph_id);
  if (it == graph_count_.cend()) {
    GELOGW("Graph of id: %u has not been added, count cannot be decreased", graph_id);
  } else {
    GELOGD("RemoveGraphCount success, graph count of id[%u] is %u", graph_id, graph_count_[graph_id]);
    (void)graph_count_.erase(it);
  }
}

void GraphManager::DecreaseGraphCount(GraphId graph_id) {
  std::lock_guard<std::mutex> lock(graph_count_mutex_);
  auto it = graph_count_.find(graph_id);
  if (it == graph_count_.end()) {
    GELOGW("Graph of id: %u has not been added, count cannot be decreased.", graph_id);
  } else {
    if (CheckUint32SubOverflow(it->second, 1) == SUCCESS) {
      --it->second;
    }
    GELOGD("After DecreaseGraphCount, graph count of id[%u] is %u.", graph_id, graph_count_[graph_id]);
  }
}

Status GraphManager::GetGraphCount(GraphId graph_id, uint32_t &count) {
  std::lock_guard<std::mutex> lock(graph_count_mutex_);
  std::map<GraphId, uint32_t>::const_iterator it = graph_count_.find(graph_id);
  if (it == graph_count_.cend()) {
    GELOGW("Graph [id:%u] has not been added.", graph_id);
    return FAILED;
  }
  count = it->second;
  return SUCCESS;
}

Status GraphManager::OptimizeGraph(const std::vector<GeTensor> &inputs, ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(compute_graph->GetGraphID(), graph_node);
  if (ret != SUCCESS) {
    return ret;
  }
  GE_DUMP(compute_graph, "PreRunBegin");
  const auto build_step = GetBuildStep(graph_node);
  if (build_step == BUILD_STEP_AFTER_BUILD) {
    GELOGD("Skip optimizegraph because build step is [%s].", build_step.c_str());
    return SUCCESS;
  }

  auto session_id = compute_graph->GetSessionID();
  /// 1. BUILD_MODE_TUNING with BUILD_STEP_AFTER_UB_MATCH no need PreRunOptimizeOriginalGraph;
  /// 2. BUILD_MODE_TUNING with BUILD_STEP_AFTER_MERGE no need PreRunOptimizeOriginalGraph.
  /// 3. BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER_SUB no need PreRunOptimizeOriginalGraph.
  bool run_optimize_original_graph = !((GetBuildMode(graph_node) == BUILD_MODE_TUNING) &&
      ((build_step == BUILD_STEP_AFTER_UB_MATCH) ||
       (build_step == BUILD_STEP_AFTER_MERGE) ||
       (build_step == BUILD_STEP_AFTER_BUILDER_SUB)));
  GELOGD("Graph name: %s, id: %u, build mode: [%s], build step: [%s].", compute_graph->GetName().c_str(),
         compute_graph->GetGraphID(), GetBuildMode(graph_node).c_str(), build_step.c_str());

  if (run_optimize_original_graph) {
    ret = PreRunOptimizeOriginalGraph(graph_node, inputs, compute_graph, session_id);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][PreRunOptimizeOriginalGraph] failed for graph:%s, session_id:%lu",
             compute_graph->GetName().c_str(), session_id);
      return ret;
    }
    int64_t graph_stage = static_cast<int64_t>(GraphStage::GRAPH_STAGE_RESERVED);
    (void)AttrUtils::GetInt(compute_graph, kGraphDumpStage, graph_stage);
    if (graph_stage == static_cast<int64_t>(GraphStage::GRAPH_STAGE_FUZZ)) {
      GELOGD("graph_stage:%d.", static_cast<int32_t>(graph_stage));
      return SUCCESS;
    }
  }

  bool run_optimize_subgraph = !((GetBuildMode(graph_node) == BUILD_MODE_TUNING) &&
                               (build_step == BUILD_STEP_AFTER_MERGE));
  if (run_optimize_subgraph) {
    ret = PreRunOptimizeSubGraph(graph_node, compute_graph, session_id);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][PreRunOptimizeSubGraph] failed for graph:%s, session_id:%lu.",
          compute_graph->GetName().c_str(), session_id);
      return ret;
    }
  }
  return SUCCESS;
}

Status GraphManager::BuildGraph(ComputeGraphPtr &compute_graph, GeRootModelPtr &ge_root_model) {
  GE_CHECK_NOTNULL(compute_graph);
  GraphNodePtr graph_node = nullptr;
  Status ret = GetGraphNode(compute_graph->GetGraphID(), graph_node);
  if (ret != SUCCESS) {
    return ret;
  }

  /// 1. BUILD_MODE_TUNING with BUILD_STEP_BEFORE_UB_MATCH no need PreRunAfterOptimizeSubGraph;
  /// 2. BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER no need PreRunAfterOptimizeSubGraph.
  /// 3. BUILD_MODE_TUNING with BUILD_STEP_AFTER_BUILDER_SUB no need PreRunAfterOptimizeSubGraph.
  const auto build_step = GetBuildStep(graph_node);
  bool run_after_optimize_subgraph =
      !((GetBuildMode(graph_node) == BUILD_MODE_TUNING) &&
        ((build_step == BUILD_STEP_BEFORE_UB_MATCH) || (build_step == BUILD_STEP_AFTER_BUILDER) ||
         (build_step == BUILD_STEP_AFTER_BUILDER_SUB)));
  GELOGD("Graph name: %s, id: %u, build mode: [%s], build step: [%s].", compute_graph->GetName().c_str(),
         compute_graph->GetGraphID(), GetBuildMode(graph_node).c_str(), build_step.c_str());
  if (run_after_optimize_subgraph) {
    ret = PreRunAfterOptimizeSubGraph(graph_node, compute_graph, ge_root_model, compute_graph->GetSessionID());
  }
  return ret;
}

std::string GraphManager::GetBuildMode(const GraphNodePtr &graph_node) const {
  const auto it = graph_node->GetOptions().find(BUILD_MODE);
  if (it != graph_node->GetOptions().end()) {
    return it->second;
  }
  return "";
}

std::string GraphManager::GetBuildStep(const GraphNodePtr &graph_node) const {
  const auto it = graph_node->GetOptions().find(BUILD_STEP);
  if (it != graph_node->GetOptions().end()) {
    return it->second;
  }
  return "";
}

std::string GraphManager::GetTuningPath(const GraphNodePtr &graph_node) const {
  const auto it = graph_node->GetOptions().find(TUNING_PATH);
  if (it != graph_node->GetOptions().end()) {
    return it->second;
  }
  return "";
}

void GraphManager::GetExcludeEngines(const GraphNodePtr &graph_node,
                                     GraphManagerOptions &refreshed_options) const {
  // option EXCLUDE_ENGINES
  auto it = graph_node->GetOptions().find(EXCLUDE_ENGINES);
  if (it != graph_node->GetOptions().end()) {
    DNNEngineManager::GetInstance().GetExcludeEngines(it->second, refreshed_options.exclude_engines);
  }

  const std::string &exclude_core_Type = (refreshed_options.core_type == kVectorCore) ? kAIcoreEngine : kVectorEngine;
  refreshed_options.exclude_engines.insert(exclude_core_Type);
}

void GraphManager::RefreshOptionByGraph(uint32_t graph_id, GraphManagerOptions &refreshed_options) {
  GraphNodePtr graph_node;
  if (GetGraphNode(graph_id, graph_node) != SUCCESS) {
    GELOGW("[Get][GraphNode] failed, Graph does not exist while done adding previously, graph_id = %u", graph_id);
    return;
  }
  refreshed_options.build_mode = GetBuildMode(graph_node);
  refreshed_options.build_step = GetBuildStep(graph_node);
  refreshed_options.tuning_path = GetTuningPath(graph_node);
  UpdateDynamicParams(refreshed_options.input_shape,
                      refreshed_options.dynamic_dims,
                      refreshed_options.dynamic_node_type,
                      graph_node->GetOptions());
  GetExcludeEngines(graph_node, refreshed_options);
}

Status GraphManager::GetGraphsMemInfo(std::map<uint32_t, std::vector<uint64_t>> &graphs_mem_info) const {
  graphs_mem_info.clear();
  for (const auto &item : graph_map_) {
    const auto &graph_node = item.second;
    GE_CHECK_NOTNULL(graph_node);
    const auto &ge_root_model = graph_node->GetGeRootModel();
    if (ge_root_model == nullptr) {
      continue;
    }
    if (!CheckModelLoad(ge_root_model, graph_node->GetLoadFlag())) {
      continue;
    }
    uint64_t graph_feature_map_size = 0UL;
    uint64_t graph_weight_size = 0UL;
    for (const auto &name_to_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
      const auto &ge_model = name_to_model.second;
      uint64_t model_mem_size = 0UL;
      uint64_t model_weight_size = 0UL;
      (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, model_mem_size);
      (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, model_weight_size);
      GE_CHK_STATUS_RET(CheckUint64AddOverflow(graph_feature_map_size, model_mem_size),
                        "[OverFlow]Add uint64 overflow! graph_feature_map_size:%lu, model_mem_size:%lu",
                        graph_feature_map_size, model_mem_size);
      graph_feature_map_size += model_mem_size;
      GE_CHK_STATUS_RET(CheckUint64AddOverflow(graph_weight_size, model_weight_size),
                        "[OverFlow]Add uint64 overflow! graph_weight_size:%lu, model_weight_size:%lu",
                        graph_weight_size, model_weight_size);
      graph_weight_size += model_weight_size;
    }
    GELOGD("Graph:%u memory info: feature map size:%lu, weight size:%lu", item.first, graph_feature_map_size,
           graph_weight_size);

    std::vector<uint64_t> graph_mem_info{graph_feature_map_size, graph_weight_size};
    graphs_mem_info.emplace(item.first, graph_mem_info);
  }
  return SUCCESS;
}

Status GraphManager::SetConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  if (graph_node->GetRunFlag()) {
    GELOGE(GE_GRAPH_GRAPH_IS_RUNNING, "[Check][RunFlag] Not allowed to set when running, graph_id:%u.", graph_id);
    return GE_GRAPH_GRAPH_IS_RUNNING;
  }
  if (graph_node->GetLoadFlag()) {
    GELOGE(GE_GRAPH_UNSUPPORTED, "[Check][LoadFlag] Not allowed to set const memory base after run graph, graph_id:%u.",
        graph_id);
    return GE_GRAPH_UNSUPPORTED;
  }
  if (!graph_node->GetBuildFlag()) {
    GELOGE(GE_GRAPH_NOT_BUILT, "[Check][CompiledFlag] Graph needs to be compiled first, graph_id:%u.", graph_id);
    return GE_GRAPH_NOT_BUILT;
  }

  const auto compute_graph = graph_node->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph, "graph_id:%u.", graph_id);
  GE_ASSERT_TRUE(!compute_graph->GetGraphUnknownFlag(), "Not support for dynamic compiled graph.");

  auto const_mem = graph_node->GetConstMemoryBase();
  if (const_mem.first != nullptr) {
    GELOGE(GE_GRAPH_REPEAT_OPERATION, "[Check][Memory] Const memory base has been set, graph_id:%u.", graph_id);
    return GE_GRAPH_REPEAT_OPERATION;
  }

  if (memory == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] invalid null memory ptr , graph_id:%u.", graph_id);
    return PARAM_INVALID;
  }

  size_t required_size = 0UL;
  CompiledGraphSummaryPtr summary = nullptr;
  const auto ret = GetCompiledGraphSummary(graph_id, summary);
  GE_ASSERT((ret == SUCCESS) && (summary != nullptr), "Get compiled graph summary failed, graph_id:%u.", graph_id);
  GE_ASSERT_SUCCESS(summary->GetConstMemorySize(required_size));
  if (size < required_size) {
    GELOGE(PARAM_INVALID, "[Check][Param] Required const memory size is %zu while input size is %zu, graph_id:%u.",
           required_size, size, graph_id);
    return PARAM_INVALID;
  }

  graph_node->SetConstMemoryBase(memory, size);
  graph_node->SetAppRefreshConstMemoryFlag();
  GELOGI("Set graph const memory base success, memory:%p, size:%zu, graph_id:%u", memory, size, graph_id);
  return SUCCESS;
}

Status GraphManager::UpdateFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);
  if (graph_node->GetRunFlag()) {
    GELOGE(GE_GRAPH_GRAPH_IS_RUNNING, "[Check][RunFlag] Not allowed to set when running, graph_id:%u.", graph_id);
    return GE_GRAPH_GRAPH_IS_RUNNING;
  }
  if (!graph_node->GetBuildFlag()) {
    GELOGE(GE_GRAPH_NOT_BUILT, "[Check][CompiledFlag] Graph needs to be compiled first, graph_id:%u.", graph_id);
    return GE_GRAPH_NOT_BUILT;
  }
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  const auto compute_graph = graph_node->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph, "graph_id:%u.", graph_id);
  GE_ASSERT_TRUE(!compute_graph->GetGraphUnknownFlag(), "Not support for dynamic compiled graph.");

  bool has_been_set = false;
  bool user_alloc = false;
  void *fixed_mem = nullptr;
  GE_ASSERT_SUCCESS(CheckFixFeatureMemoryBaseHasBeenSet(graph_node, RT_MEMORY_HBM, has_been_set,
                                                        user_alloc, fixed_mem));
  (void)has_been_set;
  if (user_alloc) {
    GELOGE(GE_GRAPH_UNSUPPORTED,
           "[Check][Memory] after set fixed_feature_memory, the function UpdateGraphFeatureMemoryBase"
           " can not be called, please refer to the guide, graph_id:%u.", graph_id);
    return GE_GRAPH_UNSUPPORTED;
  }

  if (graph_node->IsAppRefreshFeatureMemory() && graph_node->GetRefreshableFeatureMemoryBase().first != nullptr) {
    GELOGE(GE_GRAPH_UNSUPPORTED,
           "[Check][Memory] after call UpdateGraphRefreshableFeatureMemoryBase, the function"
           " UpdateGraphFeatureMemoryBase can not be called, please refer to the guide, graph_id:%u.", graph_id);
    return GE_GRAPH_UNSUPPORTED;
  }
  if (memory == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] invalid null memory ptr , graph_id:%u.", graph_id);
    return PARAM_INVALID;
  }

  size_t required_size = 0UL;
  CompiledGraphSummaryPtr summary = nullptr;
  const auto ret = GetCompiledGraphSummary(graph_id, summary);
  GE_ASSERT((ret == SUCCESS) && (summary != nullptr), "Get compiled graph summary failed, graph_id:%u.", graph_id);
  GE_ASSERT_SUCCESS(summary->GetFeatureMemorySize(required_size));
  if (size < required_size) {
    GELOGE(PARAM_INVALID, "[Check][Param] Required feature memory size is %zu while input size is %zu, graph_id:%u.",
           required_size, size, graph_id);
    return PARAM_INVALID;
  }

  bool is_refreshable = false;
  GE_ASSERT_SUCCESS(summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  const auto queryed = graph_node->GetFeatureMemoryBase();
  if (graph_node->GetLoadFlag()) {
    GE_ASSERT_TRUE(is_refreshable, "Not support for ge.featureBaseRefreshable disabled.");
    GE_ASSERT_SUCCESS(executor_->UpdateFeatureMemoryBase(graph_node, PtrToValue(memory), size), "update mem failed.");
  } else {
    if ((queryed.first != nullptr) && !is_refreshable) {
      GELOGE(GE_GRAPH_REPEAT_OPERATION, "[Check][Memory] Feature memory base has been set, graph_id:%u.", graph_id);
      return GE_GRAPH_REPEAT_OPERATION;
    }
  }
  graph_node->SetFeatureMemoryBase(memory, size);
  graph_node->SetAppRefreshFeatureMemoryFlag();
  GELOGI("Update graph feature memory base success, memory:%p, size:%zu, graph_id:%u", memory, size, graph_id);
  return SUCCESS;
}

/*
 * has_been_set为true，有可能是用户没有设置fixed，GE默认申请的; 想确认是不是用户申请的，用user_alloc
 */
Status GraphManager::CheckFixFeatureMemoryBaseHasBeenSet(const GraphNodePtr graph_node, const rtMemType_t rt_mem_type,
                                                         bool &has_been_set, bool &user_alloc, void *&mem_base) const {
  has_been_set = false;
  user_alloc = false;
  const auto &ge_root_model = graph_node->GetGeRootModel();
  GE_ASSERT_NOTNULL(ge_root_model);

  const auto fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  for (const auto &feature_mem : fixed_feature_mem) {
    if (feature_mem.first == rt_mem_type) {
      has_been_set = true;
      user_alloc = feature_mem.second.user_alloc;
      mem_base = feature_mem.second.addr;
      GELOGI("fixed_feature_memory has been set. %s", feature_mem.second.ToString().c_str());
    }
  }
  return SUCCESS;
}

Status GraphManager::CheckFixedFeatureMemoryBase(const uint32_t graph_id, const MemoryType type,
    const void *const memory, const size_t size, bool &fixed_mem_not_exist) {
  rtMemType_t rt_mem_type;
  if (MemTypeUtils::ExternalMemTypeToRtMemType(type, rt_mem_type) != GRAPH_SUCCESS) {
    GELOGE(PARAM_INVALID,
           "[Check][Param] Translate external memory type:%s to rts memory type failed, graph_id:%u.",
           MemTypeUtils::ToString(type).c_str(), graph_id);
    return PARAM_INVALID;
  }
  /*
   * 用户可以设置地址为nullptr，且size为0，表示不需要GE 框架默认申请fixed feature memory
   * 但是设置size大于0，地址为nullptr的属于无效入参
   */
  if ((size > 0U) && (memory == nullptr)) {
    GELOGE(PARAM_INVALID,
           "[Check][Param] invalid null memory ptr. type:%s, size:%zu, graph_id:%u.",
           MemTypeUtils::ToString(type).c_str(), size, graph_id);
    return PARAM_INVALID;
  }

  CompiledGraphSummaryPtr summary = nullptr;
  const auto ret = GetCompiledGraphSummary(graph_id, summary);
  GE_ASSERT((ret == SUCCESS) && (summary != nullptr), "Get compiled graph summary failed, graph_id:%u.", graph_id);

  FeatureMemoryPtr summary_fixed_mem = nullptr;
  const auto all_feature_memory = summary->GetAllFeatureMemoryTypeSize();
  for (const auto &feature_memory : all_feature_memory) {
    if (feature_memory->IsFixed() && (feature_memory->GetType() == type)) {
      summary_fixed_mem = feature_memory;
    }
  }
  if (summary_fixed_mem == nullptr) {
    fixed_mem_not_exist = true;
    GELOGW("[Check][Param] can not find fix feature memory for type:%s, graph_id:%u.",
           MemTypeUtils::ToString(type).c_str(), graph_id);
    return SUCCESS;
  }
  fixed_mem_not_exist = false;
  if ((size > 0U) && (summary_fixed_mem->GetSize() > size)) {
    GELOGE(PARAM_INVALID,
           "[Check][Param] Required fix feature memory size is %zu while input size is %zu, type: %s, graph_id:%u.",
           summary_fixed_mem->GetSize(), size, MemTypeUtils::ToString(type).c_str(), graph_id);
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status GraphManager::SetFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory,
                                               size_t size) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  if (!graph_node->GetBuildFlag()) {
    GELOGE(GE_GRAPH_NOT_BUILT, "[Check][CompiledFlag] Graph needs to be compiled first, graph_id:%u.", graph_id);
    return GE_GRAPH_NOT_BUILT;
  }
  if (graph_node->GetRunFlag()) {
    GELOGE(GE_GRAPH_GRAPH_IS_RUNNING, "[Check][RunFlag] Not allowed to set when running, graph_id:%u.", graph_id);
    return GE_GRAPH_GRAPH_IS_RUNNING;
  }

  if (graph_node->GetLoadFlag()) {
    GELOGE(GE_GRAPH_UNSUPPORTED, "[Check][LoadFlag] Not allowed to set fix feature memory base after run graph,"
           " graph_id: %u", graph_id);
    return GE_GRAPH_UNSUPPORTED;
  }
  std::string reason;
  if (!IsMemoryAndTypeSupport(graph_node, type, memory, size, reason)) {
    GELOGE(GE_GRAPH_UNSUPPORTED, "%s", reason.c_str());
    REPORT_INNER_ERR_MSG("E19999", "%s", reason.c_str());
    return GE_GRAPH_UNSUPPORTED;
  }

  bool fixed_mem_not_exist = false;
  if (CheckFixedFeatureMemoryBase(graph_id, type, memory, size, fixed_mem_not_exist) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][Param] failed.");
    return PARAM_INVALID;
  }
  if (fixed_mem_not_exist) {
    GELOGW("[Check][Param] can not find fix feature memory for type:%s, graph_id:%u.",
           MemTypeUtils::ToString(type).c_str(), graph_id);
    return SUCCESS;
  }
  rtMemType_t rt_mem_type;
  if (MemTypeUtils::ExternalMemTypeToRtMemType(type, rt_mem_type) != GRAPH_SUCCESS) {
    GELOGE(PARAM_INVALID,
           "[Check][Param] Translate external memory type:%s to rts memory type failed, graph_id:%u.",
           MemTypeUtils::ToString(type).c_str(), graph_id);
    return PARAM_INVALID;
  }
  bool has_been_set = false;
  bool user_alloc = false;
  void *fixed_mem = nullptr;
  GE_ASSERT_SUCCESS(CheckFixFeatureMemoryBaseHasBeenSet(graph_node, rt_mem_type, has_been_set, user_alloc, fixed_mem));
  (void)user_alloc;
  if (has_been_set) {
    GELOGE(GE_GRAPH_REPEAT_OPERATION, "[Check][Memory] fixed feature memory base has been set, type:%s, graph_id:%u.",
           MemTypeUtils::ToString(type).c_str(), graph_id);
    return GE_GRAPH_REPEAT_OPERATION;
  }

  const auto &ge_root_model = graph_node->GetGeRootModel();
  GE_ASSERT_NOTNULL(ge_root_model);

  FixedFeatureMemory fixed_feature_mem{rt_mem_type, const_cast<void *>(memory), size, true, false, false, 0U, nullptr};
  ge_root_model->MutableFixedFeatureMemory().insert({rt_mem_type, fixed_feature_mem});
  GELOGI("Set graph fix feature memory base success, type:%s, memory:%p, size:%zu, graph_id:%u",
         MemTypeUtils::ToString(type).c_str(), memory, size, graph_id);
  return SUCCESS;
}

Status GraphManager::UpdateRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  if (graph_node->GetRunFlag()) {
    GELOGE(GE_GRAPH_GRAPH_IS_RUNNING, "[Check][RunFlag] Not allowed to set when running, graph_id:%u.", graph_id);
    return GE_GRAPH_GRAPH_IS_RUNNING;
  }
  if (!graph_node->GetBuildFlag()) {
    GELOGE(GE_GRAPH_NOT_BUILT, "[Check][CompiledFlag] Graph needs to be compiled first, graph_id:%u.", graph_id);
    return GE_GRAPH_NOT_BUILT;
  }

  const auto compute_graph = graph_node->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph, "graph_id:%u.", graph_id);
  GE_ASSERT_TRUE(!compute_graph->GetGraphUnknownFlag(), "Not support for dynamic compiled graph.");

  if (memory == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param] invalid null memory ptr , graph_id:%u.", graph_id);
    return PARAM_INVALID;
  }
  size_t required_size = 0UL;
  size_t required_fixed_size = 0UL;
  CompiledGraphSummaryPtr summary = nullptr;
  const auto ret = GetCompiledGraphSummary(graph_id, summary);
  GE_ASSERT((ret == SUCCESS) && (summary != nullptr), "Get compiled graph summary failed, graph_id:%u.", graph_id);
  GE_ASSERT_SUCCESS(summary->GetFixedFeatureMemorySize(required_fixed_size));
  bool has_been_set = false;
  bool user_alloc = false;
  void *fixed_mem = nullptr;
  GE_ASSERT_SUCCESS(CheckFixFeatureMemoryBaseHasBeenSet(graph_node, RT_MEMORY_HBM, has_been_set,
                                                        user_alloc, fixed_mem));
  (void)has_been_set;
  // 用户已经设置Fixed Feature内存基地址，但是地址为nullptr，也报错
  if (user_alloc && (fixed_mem == nullptr)) {
    GELOGE(GE_GRAPH_UNSUPPORTED,
           "[Check][Memory] updating refreshable feature memory requires setting fixed feature memory not null,"
           " graph_id:%u.", graph_id);
    return GE_GRAPH_UNSUPPORTED;
  }
  if (graph_node->IsAppRefreshFeatureMemory() && graph_node->GetFeatureMemoryBase().first != nullptr) {
    GELOGE(GE_GRAPH_UNSUPPORTED,
           "[Check][Memory] after call UpdateGraphFeatureMemoryBase, the function"
           " UpdateGraphRefreshableFeatureMemoryBase can not be called, please refer to the guide, graph_id:%u.",
           graph_id);
    return GE_GRAPH_UNSUPPORTED;
  }
  GE_ASSERT_SUCCESS(summary->GetRefreshableFeatureMemorySize(required_size));
  if (size < required_size) {
    GELOGE(PARAM_INVALID,
           "[Check][Param] Required refreshable feature memory size is %zu while input size is %zu, graph_id:%u.",
           required_size, size, graph_id);
    return PARAM_INVALID;
  }

  bool is_refreshable = false;
  GE_ASSERT_SUCCESS(summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  const auto queryed = graph_node->GetRefreshableFeatureMemoryBase();
  if (graph_node->GetLoadFlag()) {
    GE_ASSERT_TRUE(is_refreshable, "Not support for ge.featureBaseRefreshable disabled.");
    // 加载状态下复用UpdateFeatureMemoryBase, update Refreshable Feature Memory
    GE_ASSERT_SUCCESS(executor_->UpdateFeatureMemoryBase(graph_node, PtrToValue(memory), size), "update mem failed.");
  } else {
    if ((queryed.first != nullptr) && !is_refreshable) {
      GELOGE(GE_GRAPH_REPEAT_OPERATION, "[Check][Memory] Refreshable reature memory base has been set, graph_id:%u.",
             graph_id);
      return GE_GRAPH_REPEAT_OPERATION;
    }
  }
  graph_node->SetRefreshableFeatureMemoryBase(memory, size);
  graph_node->SetAppRefreshFeatureMemoryFlag();
  GELOGI("Update graph refreshable feature memory base success, memory:%p, size:%zu, graph_id:%u",
         memory, size, graph_id);
  return SUCCESS;
}

Status GraphManager::RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const {
  GELOGI("Register external allocator success, stream:%p, allocator:%p.",
         stream, allocator.get());
  ExternalAllocatorManager::SetExternalAllocator(stream, allocator);
  return SUCCESS;
}

Status GraphManager::UnregisterExternalAllocator(const void * const stream) const {
  GELOGI("Unregister external allocator success, stream:%p.", stream);
  ExternalAllocatorManager::DeleteExternalAllocator(stream);
  return SUCCESS;
}

Status GraphManager::GetOmeContextByGraphId(const GraphId &graph_id, OmeContext &ome_context) const {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);
  ome_context = graph_node->GetOmeContext();
  return SUCCESS;
}

Status GraphManager::ConstructInputTensors(const ComputeGraphPtr &compute_graph, const std::vector<GeShape> &hint_shape,
                                           std::vector<GeTensor> &inputs, bool support_unknown_shape) const {
  std::vector<GeTensor> inputs_temp;
  for (const auto &node : compute_graph->GetInputNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    const auto &tensor_desc = node->GetOpDesc()->GetOutputDesc(0U);
    if (tensor_desc.IsValid()) {
      GELOGE(FAILED, "The tensor desc of input node %s:%s is invalid", node->GetName().c_str(),
             node->GetType().c_str());
      return FAILED;
    }

    GELOGD("Set input node[%s] tensor desc original shape[%s], shape[%s]", node->GetNamePtr(),
           tensor_desc.GetOriginShape().ToString().c_str(), tensor_desc.GetShape().ToString().c_str());
    GeTensorDesc tensor_desc_tmp(tensor_desc);
    if ((!support_unknown_shape) && tensor_desc.GetShape().IsUnknownShape()) {
      if (!hint_shape.empty()) {
        // 从option中获取输入shape值
        int64_t data_index = -1L;
        GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), "index", data_index), "Get data node %s index failed",
            node->GetName().c_str());
        GE_ASSERT_TRUE(static_cast<size_t>(data_index) < hint_shape.size(),
            "Option ge.inputHintShape is invalid, hint shape num: %zu is less than data node num", hint_shape.size());
        tensor_desc_tmp.SetShape(hint_shape[data_index]);
        tensor_desc_tmp.SetOriginShape(hint_shape[data_index]);
        GELOGI("Set data node %s input[%lld] tensor shape: [%s] from option %s",
            node->GetNamePtr(), data_index, hint_shape[data_index].ToString().c_str(), kHintInputShape);
      } else {
        const std::vector<int64_t> invalid_shape = {-3};
        tensor_desc_tmp.SetShape(GeShape(invalid_shape));
        tensor_desc_tmp.SetOriginShape(GeShape(invalid_shape));
      }
    }
    inputs_temp.emplace_back(tensor_desc_tmp);
  }

  std::vector<GeTensor> outputs_temp;
  for (const auto &output_info : compute_graph->GetGraphOutNodesInfo()) {
    GE_ASSERT_NOTNULL(output_info.first);
    GE_ASSERT_NOTNULL(output_info.first->GetOpDesc());
    outputs_temp.emplace_back(output_info.first->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(output_info.second)));
  }

  GE_ASSERT_SUCCESS(NormalizeInputsOutputs(compute_graph, inputs_temp, outputs_temp, inputs));
  return SUCCESS;
}

Status GraphManager::UpdateInputWithHintShape(const std::vector<GeShape> &hint_shape, std::vector<GeTensor> &inputs) const {
  if (hint_shape.empty()) {
    return SUCCESS;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    GE_ASSERT_TRUE(i < hint_shape.size(), "Option ge.inputHintShape is invalid, hint shape num: %zu is"
      " less than inputs num: %zu", hint_shape.size(), inputs.size());
    const auto &shape = hint_shape[i];
    if (shape.GetDims() == DUMMY_SHAPE) {
      GELOGW("InputHintShape[%u] is dummy shape, not update.", i);
      continue;
    }
    auto tensor_desc = inputs[i].GetTensorDesc();
    GELOGD("Before update input %zu is %s.", i, tensor_desc.GetShape().ToString().c_str());
    tensor_desc.SetShape(hint_shape[i]);
    tensor_desc.SetOriginShape(hint_shape[i]);
    inputs[i].SetTensorDesc(tensor_desc);
    GELOGD("After update input %zu is %s.", i, tensor_desc.GetShape().ToString().c_str());
  }
  return SUCCESS;
}

Status GraphManager::CompileGraph(uint32_t graph_id, uint64_t session_id, const vector<ge::Tensor> &inputs) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, session_id:%lu, graph_id:%u.", session_id, graph_id);
  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph_id:%u, "
                         "check invalid", graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] graph already running, graph id = %u", graph_node->GetGraphId());
    return GE_GRAPH_ALREADY_RUNNING;
  }

  // has compiled before, return
  if (graph_node->GetBuildFlag()) {
    // has compiled before, but now need to re-build, report error
    if (IsGraphNeedBuild(graph_node)) {
      REPORT_INNER_ERR_MSG("E19999", "[Get][BuildFlag] The graph %u need to re-build, you should remove it from GE "
                           "first, then AddGraph again and re-compile it.", graph_node->GetGraphId());
      GELOGE(PARAM_INVALID, "[Get][BuildFlag] The graph %u need to re-build, you should remove it from GE "
            "first, then AddGraph again and re-compile it.", graph_node->GetGraphId());
      return PARAM_INVALID;
    }
    GELOGW("[Check][BuildFlag] Graph is already compiled, session_id:%lu, graph_id:%u, inputs size:%zu.",
           session_id, graph_id, inputs.size());
    return SUCCESS;
  }

  GE_ASSERT_NOTNULL(graph_node->GetGraph());
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*(graph_node->GetGraph()));
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_SUCCESS(TranFrameOp(graph_node));

  UpdateLocalOmgContext(graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  graph_node->SetRunFlag(true);
  GE_MAKE_GUARD(run_flag_guard, [&graph_node] () { graph_node->SetRunFlag(false); });

  std::vector<GeTensor> ge_tensor_inputs;
  if (inputs.empty()) {
    // inputs take precedence over hit_shape in the evaluation.
    std::vector<GeShape> hint_shape;
    GE_ASSERT_SUCCESS(ParseHintInputShape(hint_shape));
    GE_ASSERT_SUCCESS(ConstructInputTensors(compute_graph, hint_shape, ge_tensor_inputs),
                      "Construct model input tensor desc failed, maybe the input tensor desc is invalid.");
  } else {
    for (const auto &input : inputs) {
      ge_tensor_inputs.emplace_back(TensorAdapter::AsGeTensor(input));
    }
  }

  // normalize all ge tensor desc on root graph. to ensure mv storage format to attr on graph
  GE_ASSERT_SUCCESS(NormalizeGeTensorOnComputeGraph(compute_graph), "Normalize ge_tensor_desc on graph failed.");

  GeRootModelPtr ge_root_model = nullptr;
  const auto ret = PreRun(graph_node, ge_tensor_inputs, ge_root_model, session_id);
  GE_ASSERT_SUCCESS(ret, "[Call][PreRun] Failed, graph_id:%u, session_id:%lu.", graph_node->GetGraphId(), session_id);

  graph_node->SetBuildFlag(true);
  GE_ASSERT_NOTNULL(graph_rebuild_state_ctrl_);
  graph_rebuild_state_ctrl_->SetGraphBuildEnd(graph_node->GetGraphId());

  std::string refreshable = "0";  // default is not refreshable
  (void)GetContext().GetOption(OPTION_FEATURE_BASE_REFRESHABLE, refreshable);
  graph_node->SetFeatureBaseRefreshable(refreshable.compare("1") == 0);
  GELOGI("compile graph success, graph_id:%u, session_id:%lu, is feature refreshable:%d.", graph_node->GetGraphId(),
         session_id, graph_node->IsFeatureBaseRefreshable());
  return SUCCESS;
}

bool GraphManager::IsContainVariable(const ComputeGraphPtr &compute_graph) const {
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (OpTypeUtils::IsVariableNode(node->GetType())) {
      GELOGI("graph contain variable op:%s", node->GetName().c_str());
      return true;
    }
  }
  return false;
}

void GraphManager::SaveCompiledMemSize(const GraphNodePtr &graph_node, const CompiledGraphSummaryPtr &summary) const {
  size_t const_size = 0UL;
  size_t feature_mem_size = 0UL;
  size_t refreshable_feature_mem_size = 0UL;
  bool is_refreshable = false;
  summary->GetConstMemorySize(const_size);
  summary->GetFeatureMemorySize(feature_mem_size);
  summary->GetRefreshableFeatureMemorySize(refreshable_feature_mem_size);
  summary->GetFeatureMemoryBaseRefreshable(is_refreshable);

  graph_node->SetFeatureBaseRefreshable(is_refreshable);
  graph_node->SetConstMemoryBase(nullptr, const_size);
  graph_node->SetFeatureMemoryBase(nullptr, feature_mem_size);
  graph_node->SetRefreshableFeatureMemoryBase(nullptr, refreshable_feature_mem_size);
}

Status GraphManager::GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);
  GetThreadLocalContext().SetGraphOption(graph_node->GetOptions());
  if (!graph_node->GetBuildFlag()) {
    GELOGE(GE_GRAPH_NOT_BUILT, "[Check][CompiledFlag] Graph needs to be compiled first, graph_id:%u.", graph_id);
    return GE_GRAPH_NOT_BUILT;
  }
  summary = graph_node->GetCompiledGraphSummary();
  if (summary != nullptr) {
    return SUCCESS;
  }

  const auto &ge_root_model = graph_node->GetGeRootModel();
  GE_ASSERT_NOTNULL(ge_root_model);
  GE_ASSERT(!ge_root_model->GetSubgraphInstanceNameToModel().empty());
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  GE_ASSERT_NOTNULL(ge_model);
  
  summary = CompiledGraphSummary::Builder::Build(ge_model, ge_root_model);
  GE_ASSERT_NOTNULL(summary, "Get compiled graph summary failed, graph_id:%u.", graph_id);
  graph_node->SaveCompiledGraphSummary(summary);
  SaveCompiledMemSize(graph_node, summary);
  return SUCCESS;
}

const std::vector<GraphId> &GraphManager::GetOrderedGraphIds() const {
  return graph_ids_;
}

Status GraphManager::PaRemapped(const GraphId graph_id, const uint64_t va, const uint64_t new_pa,
                                const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) const {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(GetGraphNode(graph_id, graph_node), "Graph:%u does not exist in graph_map, check invalid", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);

  if (!graph_node->GetBuildFlag()) {
    GELOGE(GE_GRAPH_NOT_BUILT, "[Check][CompiledFlag] Graph needs to be compiled first, graph_id:%u.", graph_id);
    return GE_GRAPH_NOT_BUILT;
  }

  const auto compute_graph = graph_node->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph, "graph_id:%u.", graph_id);
  if (compute_graph->GetGraphUnknownFlag()) {
    GELOGW("[Check][PaRemapped] Not support for dynamic compiled graph, graph_id:%u.", graph_id);
    return FAILED;
  }

  GE_CHECK_NOTNULL(executor_);
  return executor_->PaRemapped(graph_node, va, new_pa, len, cross_ranges);
}

Status GraphManager::ForkGraph(uint32_t origin_graph_id, uint32_t forked_graph_id) {
  bool is_fork_exist = false;
  GE_ASSERT_SUCCESS(CheckGraphExisted(forked_graph_id, is_fork_exist));
  if (is_fork_exist) {
    GELOGE(PARAM_INVALID, "Forked Graph %u is already exist", forked_graph_id);
    return PARAM_INVALID;
  }

  GraphNodePtr origin_graph_node = nullptr;
  Status ret = GetGraphNode(origin_graph_id, origin_graph_node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u to fork does not exist in graph_map, check invalid", origin_graph_id);
    GELOGE(ret, "[Get][GraphNode] failed, graph to fork does not exist, graph_id = %u.", origin_graph_id);
    return ret;
  }
  if (!origin_graph_node->GetBuildFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u to fork not compiled, check invalid", origin_graph_id);
    GELOGE(GRAPH_FAILED, "[Get][GraphNode] failed, graph to fork not compiled, graph_id = %u.", origin_graph_id);
    return GRAPH_FAILED;
  }
  auto fork_graph_node = origin_graph_node->Fork(forked_graph_id);
  GE_ASSERT_NOTNULL(fork_graph_node);
  AddGraphNode(forked_graph_id, fork_graph_node);
  GE_ASSERT_TRUE(graph_ids_to_forked_ids_[origin_graph_id].emplace(forked_graph_id).second,
                 "Forked graph id %u already exists", forked_graph_id);
  GELOGI("Fork graph %u success from graph %u", forked_graph_id, origin_graph_id);
  return SUCCESS;
}

Status GraphManager::DoDynamicShapePartition(const GraphNodePtr &graph_node,
                                              const ComputeGraphPtr &compute_graph) {
  // ffts+场景采用优先merge静态子图的策略
  GE_TRACE_START(GraphPartitionDynamicShape);
  bool ffts_flag = OpsKernelManager::GetInstance().GetEnableFftsFlag();
  DynamicShapePartitioner dynamic_shape_partitioner(compute_graph, ffts_flag);
  auto ret = dynamic_shape_partitioner.Partition();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][Partition] for Graph:%s by dynamic shape Failed", compute_graph->GetName().c_str());
    return ret;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphPartitionDynamicShape, "OptimizeSubgraph::GraphPartitionDynamicShape");

  GE_TRACE_START(EnginePlacer2);
  auto &engine_placer = GetCompilerStages(graph_node->GetGraphId()).partitioner.GetEnginePlacer();
  if (engine_placer.ReAssignEngine() != SUCCESS) {
    GELOGE(FAILED, "[Call][Run] Engine placer reassign failed, graph:%s.", compute_graph->GetName().c_str());
    return FAILED;
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(EnginePlacer2, "OptimizeSubgraph::EnginePlacer2");
  GE_DUMP(compute_graph, "AfterDynamicShapePartition");
  return SUCCESS;
}

Status GraphManager::DoSubgraphPartitionWithMode(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                                                  uint64_t session_id, EnginePartitioner::Mode mode,
                                                  const char *mode_name) {
  GE_TRACE_START(SubgraphPartitionAndOptimization_Mode);
  std::string trace_name = std::string("SubgraphPartitionAndOptimization_") + mode_name;
  TraceOwnerGuard guard("GE", trace_name, compute_graph->GetName());
  auto ret = SubgraphPartitionAndOptimization(graph_node, compute_graph, session_id, mode);
  if (ret != SUCCESS) {
    GELOGE(ret, "[SubgraphPartitionAndOptimization][%s] for graph:%s failed", mode_name,
           compute_graph->GetName().c_str());
    return ret;
  }
  std::string timestamp_name = std::string("OptimizeSubgraph::SubgraphPartitionAndOptimization::") + mode_name;
  GE_COMPILE_TRACE_TIMESTAMP_END(SubgraphPartitionAndOptimization_Mode, timestamp_name.c_str());
  std::string dump_name = std::string("MergedComputeGraphAfter") + mode_name + "Partition";
  GE_DUMP(compute_graph, dump_name.c_str());
  return SUCCESS;
}
}  // namespace ge
