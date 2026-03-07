/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/hybrid_model_builder.h"

#include <algorithm>

#include "framework/common/op/ge_op_utils.h"
#include "graph/ge_context.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "common/omg_util/omg_util.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/checker.h"
#include "graph/ir_definitions_recover.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/manager/mem_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/math_util.h"
#include "graph/unfold/graph_unfolder.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/node_executor/node_executor.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"

namespace ge {
namespace hybrid {
namespace {
const uint32_t kHybridSubgraphIndex = 0U;
const int32_t kHybridVarOutputIndex = 0;
const uint64_t kHybridProfilingFpStartLogId = 2U;
const uint64_t kHybridProfilingBpEndLogId = 3U;
const uint64_t kHybridProfilingIterEndLogId = 4U;
const uint8_t kHybridLoopEnterIdx = 0U;
const uint8_t kHybridLoopIterationIdx = 1U;
const uint8_t kHybridLoopMergeSize = 2U;
const uint8_t kHybridStreamSwitchIdx = 1U;
const uint8_t kHybridStreamSwitchNum = 2U;
const int64_t kHybridPlacementHost = 1;
const int64_t kHybridPlacementHostCompileIndependent = 2;
const uint32_t kHybridSubgraphRecursion = 16U;
const std::string kHybridOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
const std::string kHybridProfilingGraph = "ProfilingGraph";
const std::string kHybridProfilingFpNode = "ProfilingFpNode";
const std::string kHybridProfilingBpNode = "ProfilingBpNode";
const std::string kHybridProfilingEndNode = "ProfilingEndNode";
const std::string kHybridProfilingArNode = "ProfilingAllReduceNode";
const std::string kHybridEngineNameRts = "DNN_VM_RTS_OP_STORE";
const std::string kHybridForceInfershape = "_force_infershape_when_running";
const std::string kHybridDataModeDynamic = "dynamic_aipp";

const std::set<std::string> kHybridExecutionDependentTypes{ IF, STATELESSIF, CASE, STREAMSWITCH, STACK };
const std::set<std::string> kHybridMergeInputSkipTypes{ STREAMACTIVE, STREAMSWITCH, CONSTANT, CONSTANTOP };
const std::set<std::string> kHybridStreamActiveTypes{ ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION };
const std::set<std::string> kFftsValidInferDepTypes {CONSTANT, CONSTANTOP, DATA};

Status SetOutputNameAttr(ComputeGraph &graph) {
  std::vector<std::string> output_names;
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    const auto op_type = op_desc->GetType();
    if (op_type == NETOUTPUT) {
      for (InDataAnchor *in_data_anchor : node->GetAllInDataAnchorsPtr()) {
        const OutDataAnchorPtr &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
        const auto &in_node = peer_out_anchor->GetOwnerNodeBarePtr();
        GE_CHECK_NOTNULL(in_node);
        const auto &input_desc = op_desc->GetInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
        auto tensor_name = AttrUtils::GetStr(input_desc, ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME);
        std::string output_name = in_node->GetName();
        if (tensor_name != nullptr) {
          output_name += ":" + std::to_string(peer_out_anchor->GetIdx()) + ":" + *tensor_name;
        }
        output_names.push_back(output_name);
      }
    }
  }
  GE_CHK_BOOL_EXEC(AttrUtils::SetListStr(&graph, ATTR_MODEL_OUT_NODES_NAME, output_names),
      GELOGE(FAILED, "[Invoke][SetListStr] failed, graph:%s name:%s.", graph.GetName().c_str(),
             ATTR_MODEL_OUT_NODES_NAME.c_str());
      REPORT_INNER_ERR_MSG("E19999", "SetListStr failed, graph:%s name:%s.",  graph.GetName().c_str(),
                        ATTR_MODEL_OUT_NODES_NAME.c_str());
      return FAILED);
  return SUCCESS;
}

int64_t CalcTensorSizeInBytes(const GeTensorDesc &tensor_desc) {
  int64_t var_size = 0;
  const auto data_type = tensor_desc.GetDataType();
  if (data_type == DT_STRING) {
    (void)TensorUtils::GetSize(tensor_desc, var_size);
    return var_size;
  }

  if (TensorUtils::GetTensorMemorySizeInBytes(tensor_desc, var_size) != GRAPH_SUCCESS) {
    GELOGW("Failed to calc var data size");
    return -1;
  }

  return var_size;
}

Status CollectDependenciesForFusedGraph(const NodeItem &node_item,
                                        std::vector<std::pair<OpDesc *, std::pair<OpDesc *, int32_t>>> &data_deps) {
  gert::OpImplSpaceRegistryV2Array space_registry_array;
  GE_ASSERT_NOTNULL(node_item.op_desc);
  GE_ASSERT_TRUE(static_cast<size_t>(node_item.op_desc->GetOppImplVersion()) < space_registry_array.size());
  space_registry_array.at(static_cast<size_t>(node_item.op_desc->GetOppImplVersion())) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  for (const auto &node : node_item.fused_subgraph->nodes) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    gert::DataDependentInterpreter ddi(op_desc, space_registry_array);

    for (const auto in_anchor: node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      bool is_data_dependent = false;
      GE_ASSERT_GRAPH_SUCCESS(ddi.IsDataDependent(in_anchor->GetIdx(), is_data_dependent));
      if (!is_data_dependent) {
        continue;
      }
      const auto &src_node = NodeUtils::GetInDataNodeByIndex(*node, in_anchor->GetIdx());
      GE_CHECK_NOTNULL(src_node);
      const auto &src_op_desc = src_node->GetOpDesc();
      GE_CHECK_NOTNULL(src_op_desc);
      if (src_node->GetType() != DATA_TYPE) {
        GELOGE(UNSUPPORTED, "[Check][NodeType][%s(%s)::%s(%s)] Node in fused subgraph can only depend on Data nodes,"
               "but depend on %s actually", node_item.NodeName().c_str(), node_item.NodeType().c_str(),
               node->GetName().c_str(), node->GetType().c_str(),
               src_node->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "[%s(%s)::%s(%s)] Node in fused subgraph can only depend on Data nodes,"
                           "but depend on %s actually.", node_item.NodeName().c_str(), node_item.NodeType().c_str(),
                           node->GetName().c_str(), node->GetType().c_str(),
                           src_node->GetType().c_str());
        return UNSUPPORTED;
      }

      data_deps.emplace_back(src_op_desc.get(), std::make_pair(op_desc.get(), in_anchor->GetIdx()));
      GELOGD("[%s] Dependent added from input of [%s:%d]",
             src_op_desc->GetNamePtr(),
             op_desc->GetNamePtr(),
             in_anchor->GetIdx());
    }
  }

  return SUCCESS;
}
}  // namespace
HybridModelBuilder::HybridModelBuilder(HybridModel &hybrid_model)
    : hybrid_model_(hybrid_model), runtime_param_(hybrid_model.root_runtime_param_) {
  ge_root_model_ = hybrid_model_.ge_root_model_;
}

Status HybridModelBuilder::InitNodeBinMode() {
  std::string node_bin_mode_str;
  if (GetContext().GetOption("RUNTIME_NODE_BIN_MODE", node_bin_mode_str) == GRAPH_SUCCESS) {
    int32_t node_bin_mode = fuzz_compile::kNodeBinModeEnd;
    try {
      node_bin_mode = std::stoi(node_bin_mode_str);
    } catch (std::invalid_argument &) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Failed to analyse the node bin mode, %s", node_bin_mode_str.c_str());
      // report error
      return FAILED;
    } catch (std::out_of_range &) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Failed to analyse the node bin mode, %s", node_bin_mode_str.c_str());
      // report error
      return FAILED;
    }
    if (node_bin_mode >= fuzz_compile::kNodeBinModeEnd) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid node bin mode %s", node_bin_mode_str.c_str());
      // report error
      return FAILED;
    }
    hybrid_model_.SetNodeBinMode(static_cast<fuzz_compile::NodeBinMode>(node_bin_mode));
  } else {
    GELOGD("The RUNTIME_NODE_BIN_MODE not set, run in default mode: one-node-single-bin");
    hybrid_model_.SetNodeBinMode(fuzz_compile::kOneNodeSingleBinMode);
  }
  return SUCCESS;
}

Status HybridModelBuilder::InitOverflowAddr() const {
  const auto &root_graph = ge_root_model_->GetRootGraph();
  int64_t global_workpace_size = 0;
  (void)AttrUtils::GetInt(root_graph, "globalworkspace_size", global_workpace_size);
  if (global_workpace_size > 0) {
    auto *const mem_allocator = NpuMemoryAllocator::GetAllocator(hybrid_model_.device_id_, nullptr);
    GE_CHECK_NOTNULL(mem_allocator);
    hybrid_model_.globalworkspace_overflow_addr_ =
        TensorBuffer::Create(mem_allocator, static_cast<size_t>(global_workpace_size));
    GE_CHECK_NOTNULL(hybrid_model_.globalworkspace_overflow_addr_);
  }
  return SUCCESS;
}

Status HybridModelBuilder::Build() {
  GE_CHK_STATUS_RET(ValidateParams(), "[Invoke][ValidateParams] failed, model_name[%s]", ModelName().c_str());
  hybrid_model_.model_name_ = ge_root_model_->GetModelName();
  GELOGI("[%s] Start to build hybrid model.", ModelName().c_str());
  GE_CHK_STATUS_RET(InitNodeBinMode(), "[Init]][InitNodeBinMode] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitOverflowAddr(), "[Init]][OverflowAddr] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(CopyGraph(), "[Invoke][CopyGraph] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitRuntimeParams(), "[Invoke][InitRuntimeParams] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(RecoverGraphUnknownFlag(),
                    "[Invoke][RecoverGraphUnknownFlag] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(IndexSpecialNodes(), "[Invoke][IndexSpecialNodes] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(IndexTaskDefs(), "[Invoke][IndexTaskDefs] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitWeights(), "[Invoke][InitWeights] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(LoadGraph(), "[Invoke][LoadGraph] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(AssignUninitializedConstantOps(),
                    "[Invoke][AssignUninitializedConstantOps] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(TransAllVarData(), "[Invoke][TransAllVarData] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(CopyVarData(), "[Invoke][CopyVarData] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitModelMem(), "[Invoke][InitModelMem] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitConstantOps(), "[Invoke][InitConstantOps] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitFileConstantOps(), "[Invoke][InitFileConstantOps] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitVariableTensors(), "[Invoke][InitVariableTensors], model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(LoadTasks(), "[Invoke][LoadTasks] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(OptimizeDependenciesForConstantInputs(),
                    "[Invoke][OptimizeDependenciesForConstantInputs] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitAippInfoAndType(), "[Invoke][InitAippInfoAndType] failed, model_name[%s]", ModelName().c_str());
  GELOGI("[%s] Done building hybrid model successfully.", ModelName().c_str());
  return SUCCESS;
}

Status HybridModelBuilder::BuildForSingleOp() {
  GE_CHK_STATUS_RET(ValidateParams(), "[Invoke][ValidateParams] failed, model_name[%s]", ModelName().c_str());
  hybrid_model_.root_graph_ = ge_root_model_->GetRootGraph();
  hybrid_model_.model_name_ = ge_root_model_->GetRootGraph()->GetName();
  GELOGI("[%s] Start to build hybrid model.", ModelName().c_str());
  const auto &name_to_model = ge_root_model_->GetSubgraphInstanceNameToModel();
  const auto it = name_to_model.find(hybrid_model_.root_graph_->GetName());
  if (it == name_to_model.end()) {
    GELOGE(FAILED, "Graph[%s] hybrid Model not found", hybrid_model_.root_graph_->GetName().c_str());
    return FAILED;
  }

  GE_CHK_STATUS_RET(InitNodeBinMode(), "[Init]][InitNodeBinMode] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(RecoverGraphUnknownFlag(),
                    "[Invoke][RecoverGraphUnknownFlag] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(IndexTaskDefs(), "[Invoke][IndexTaskDefs] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(LoadGraph(), "[Invoke][LoadGraph] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(InitWeights(), "[Invoke][InitWeights] failed, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(LoadTasks(), "[Invoke][LoadTasks] failed, model_name[%s]", ModelName().c_str());
  GELOGI("[%s] Done building hybrid model for single op successfully.", ModelName().c_str());
  return SUCCESS;
}

Status HybridModelBuilder::ValidateParams() const {
  GE_CHECK_NOTNULL(ge_root_model_);
  GE_CHECK_NOTNULL(ge_root_model_->GetRootGraph());
  return SUCCESS;
}

Status HybridModelBuilder::CopyGraph() const {
  GELOGD("Copy compute graph begin.");
  const auto &root_graph = ge_root_model_->GetRootGraph();

  const std::string new_graph_name = ge_root_model_->GetRootGraph()->GetName();
  ComputeGraphPtr new_root_graph = MakeShared<ComputeGraph>(new_graph_name);
  GE_CHECK_NOTNULL(new_root_graph);
  const int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  const auto ret = GraphUtils::CopyComputeGraph(root_graph, new_root_graph, node_old_2_new, op_desc_old_2_new, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][ComputeGraph] failed, root_graph:%s.", new_graph_name.c_str());
    return GRAPH_FAILED;
  }
  hybrid_model_.root_graph_ = new_root_graph;

  GELOGD("Copy compute graph [%s] success.", new_graph_name.c_str());
  return SUCCESS;
}

Status HybridModelBuilder::ValidateFftsPlusSubNodeItem(const NodeItem &node_item) const {
  if (!node_item.is_ffts_sub_node_) {
    return SUCCESS;
  }

  if (node_item.shape_inference_type >= DEPEND_SHAPE_RANGE) {
    GELOGE(FAILED, "[%s] shape inference type [%u] is unsupported in fftsplus graph.",
           node_item.node_type.c_str(), static_cast<uint32_t>(node_item.shape_inference_type));
    return FAILED;
  }
  for (const auto &dep_node : node_item.dependents_for_shape_inference) {
    if (kFftsValidInferDepTypes.count(dep_node->GetType()) == 0UL) {
      GELOGE(FAILED, "[%s] shape inference depends node type [%s] is unsupported in fftsplus graph.",
             node_item.node_type.c_str(), dep_node->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
Status HybridModelBuilder::BuildNodeItem(const NodePtr &node, NodeItem &node_item) {
  GE_CHK_STATUS_RET(ParseForceInfershapeNodes(node, node_item),
                    "[Invoke][ParseForceInfershapeNodes]failed, node:[%s(%s)].",
                    node_item.NodeName().c_str(), node_item.NodeType().c_str());
  GE_CHK_STATUS_RET(ParseDependentInputNodes(node_item),
                    "[Invoke][ParseDependentInputNodes]failed, node:[%s(%s)].",
                    node_item.NodeName().c_str(), node_item.NodeType().c_str());
  GE_CHK_STATUS_RET(ValidateFftsPlusSubNodeItem(node_item));

  const int32_t num_outputs = node_item.num_outputs;
  node_item.outputs.resize(static_cast<size_t>(num_outputs));
  for (int32_t i = 0; i < num_outputs; ++i) {
    const auto &out_data_anchor = node->GetOutDataAnchor(i);
    if (out_data_anchor == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Get][OutDataAnchor]out anchor[%d] of node %s(%s) is nullptr",
             i, node->GetName().c_str(), node->GetType().c_str());
      REPORT_INNER_ERR_MSG("E19999", "out anchor[%d] of node %s(%s) is nullptr.",
                        i, node->GetName().c_str(), node->GetType().c_str());
      return INTERNAL_ERROR;
    }
    for (const auto &dst_in_anchor: out_data_anchor->GetPeerInDataAnchors()) {
      const auto &dst_node = dst_in_anchor->GetOwnerNode();
      if (dst_node == nullptr) {
        GELOGW("dst node is nullptr. out anchor = %d", out_data_anchor->GetIdx());
      } else {
        NodeItem *dst_node_item = nullptr;
        GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                          "[GetOrCreate][NodeItem] failed, dst_node:[%s(%s)].",
                          dst_node->GetName().c_str(), dst_node->GetType().c_str());
        int32_t canonical_index;
        const uint32_t in_anchor_index = static_cast<uint32_t>(dst_in_anchor->GetIdx());
        GE_CHK_STATUS_RET(dst_node_item->GetCanonicalInputIndex(in_anchor_index, canonical_index),
                          "[Invoke][GetCanonicalInputIndex] failed, dst_node:[%s(%s)].",
                          dst_node->GetName().c_str(), dst_node->GetType().c_str());

        node_item.outputs[static_cast<size_t>(i)].emplace_back(canonical_index, dst_node_item);
        node_item.SetDataSend(dst_node_item, dst_in_anchor->GetIdx());
      }
    }
  }

  GE_CHK_STATUS_RET_NOLOG(ResolveRefIo(node_item));
  return SUCCESS;
}

Status HybridModelBuilder::ResolveRefIo(NodeItem &node_item) {
  bool is_ref = false;
  auto &op_desc = *node_item.op_desc;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (!is_ref) {
    return SUCCESS;
  }

  const auto &inputs = op_desc.GetAllInputName();
  const auto &outputs = op_desc.GetAllOutputName();
  for (const auto &output : outputs) {
    for (const auto &input : inputs) {
      if (input.first == output.first) {
        int32_t input_idx;
        GE_CHK_STATUS_RET_NOLOG(node_item.GetCanonicalInputIndex(input.second, input_idx));
        const auto output_idx = static_cast<int32_t>(output.second);
        node_item.reuse_inputs[output_idx] = input_idx;
        GELOGD("[%s] Output[%d] reuse input[%d]", node_item.NodeName().c_str(), output_idx, input_idx);
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::GetOrCreateNodeItem(const NodePtr &node, NodeItem *&node_item) const {
  auto &node_items = hybrid_model_.node_items_;
  const std::map<ge::NodePtr, std::unique_ptr<ge::hybrid::NodeItem>>::const_iterator it = node_items.find(node);
  if (it != node_items.cend()) {
    node_item = it->second.get();
    return SUCCESS;
  }

  std::unique_ptr<NodeItem> uniq_node;
  GE_CHK_STATUS_RET(NodeItem::Create(node, uniq_node), "[Invoke][Create] failed, model_name[%s]", ModelName().c_str());
  auto &new_node = *uniq_node;
  GE_CHK_STATUS_RET_NOLOG(NodeExecutorManager::GetInstance().GetExecutor(new_node, new_node.node_executor));
  new_node.node_id = new_node.op_desc->GetId();
  node_item = uniq_node.get();
  node_items[node] = std::move(uniq_node);
  if (new_node.IsFftsSubNode()) {
    return SUCCESS;
  }

  // we do not need L2 Buffer
  const std::string kMarkIsFirstNode = "is_first_node";
  const std::string kMarkIsLastNode = "is_last_node";
  (void)AttrUtils::SetBool(new_node.op_desc, kMarkIsFirstNode, false);
  (void)AttrUtils::SetBool(new_node.op_desc, kMarkIsLastNode, false);

  const auto executor_type = NodeExecutorManager::GetInstance().ResolveExecutorType(new_node);
  new_node.is_profiling_report = (executor_type == NodeExecutorManager::ExecutorType::AICORE) ||
                                 (executor_type == NodeExecutorManager::ExecutorType::AICPU_TF) ||
                                 (executor_type == NodeExecutorManager::ExecutorType::AICPU_CUSTOM);
  std::string data_mode;
  const bool ret = AttrUtils::GetStr(node->GetOpDescBarePtr(), ATTR_DATA_RELATED_AIPP_MODE, data_mode);
  if (ret && (data_mode == kHybridDataModeDynamic)) {
    // Dynamic aipp skip the check for sufficiency of input data.
    (void)new_node.skip_sufficiency_of_input_check_.insert(0);
  }
  return SUCCESS;
}

Status HybridModelBuilder::ParseForceInfershapeNodes(const NodePtr &node, NodeItem &node_item) const {
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // not care result, if no this attr, stand for the op does not need force infershape
  (void)AttrUtils::GetBool(op_desc, kHybridForceInfershape, node_item.is_need_force_infershape);
  GELOGD("Node [%s] need to do infershape, flag is %d",
         op_desc->GetName().c_str(),
         static_cast<int32_t>(node_item.is_need_force_infershape));
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependencies(NodeItem &node_item,
                                             std::set<NodePtr> &dependent_for_shape_inference) {
  if (node_item.fused_subgraph != nullptr) {
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(node_item.op_desc);
  gert::OpImplSpaceRegistryV2Array space_registry_array;
  GE_ASSERT_TRUE(static_cast<size_t>(node_item.op_desc->GetOppImplVersion()) < space_registry_array.size());
  space_registry_array.at(static_cast<size_t>(node_item.op_desc->GetOppImplVersion())) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::DataDependentInterpreter ddi(node_item.op_desc->shared_from_this(), space_registry_array);

  for (const auto in_anchor: node_item.node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    bool is_data_dependent = false;
    GE_ASSERT_GRAPH_SUCCESS(ddi.IsDataDependent(in_anchor->GetIdx(), is_data_dependent));
    if (!is_data_dependent) {
      continue;
    }
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    auto *const src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    if (src_node_item->NodeType() == DATA) {
      const auto &op_desc = src_node_item->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const auto &input_tensor = op_desc->MutableInputDesc(0U);
      GE_CHECK_NOTNULL(input_tensor);
      int64_t input_type = 0;
      if (AttrUtils::GetInt(input_tensor, ATTR_NAME_PLACEMENT, input_type) &&
          ((input_type == kHybridPlacementHost) || (input_type == kHybridPlacementHostCompileIndependent))) {
        GELOGD("Skip d2h memcpy, get hostmem from node %s, input_type = %ld.",
               src_node_item->NodeName().c_str(), input_type);
        continue;
      }
    }
    (void) src_node_item->to_const_output_id_list.emplace(peer_out_anchor->GetIdx());
    (void) dependent_for_shape_inference.emplace(src_node);
    host_input_value_dependencies_[&node_item].emplace_back(peer_out_anchor->GetIdx(), src_node_item);
    GELOGD("[%s][%s] infer shape dependent value of output of [%s:%d]",
           node_item.NodeName().c_str(),
           node_item.node_type.c_str(),
           src_node_item->NodeName().c_str(),
           peer_out_anchor->GetIdx());
  }
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentInputNodes(NodeItem &node_item) {
  std::set<NodePtr> dependent_for_execution;
  GE_CHK_STATUS_RET_NOLOG(ParseDependentInData(node_item, dependent_for_execution));

  const auto &ge_node = node_item.node;
  if (node_item.node_type == NETOUTPUT) {
    for (const auto &src_node : ge_node->GetInControlNodes()) {
      auto *const src_node_item = MutableNodeItem(src_node);
      if ((src_node_item != nullptr) && src_node_item->IsHcclOp()) {
        GELOGD("[%s](%s) Add input control dependent node [%s](%s)",
               ge_node->GetName().c_str(), ge_node->GetType().c_str(),
               src_node->GetName().c_str(), src_node->GetType().c_str());
        (void)dependent_for_execution.emplace(src_node);
      }
    }
  }

  // cond or branch need to be prepared before the execution of IF or CASE
  // data flow source also need input to be prepared
  if (kHybridExecutionDependentTypes.count(node_item.node_type) > 0U) {
    const auto &src_node = NodeUtils::GetInDataNodeByIndex(*ge_node, 0); // cond input
    GE_CHECK_NOTNULL(src_node);
    auto *const src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    (void)dependent_for_execution.emplace(src_node);
    GELOGD("[%s] Dependent added from %s for control op's cond/branch or data flow source",
           node_item.NodeName().c_str(), src_node_item->NodeName().c_str());
  }

  std::set<NodePtr> dependent_for_shape_inference;
  GE_CHK_STATUS_RET(ParseDependencies(node_item, dependent_for_shape_inference));
  GE_CHK_STATUS_RET(ParseDependentForFusedSubgraph(node_item, dependent_for_shape_inference));

  for (const auto &dep_node : dependent_for_shape_inference) {
    auto *const src_node_item = MutableNodeItem(dep_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->has_observer = true;
    node_item.dependents_for_shape_inference.emplace_back(dep_node);
  }

  for (const auto &dep_node : dependent_for_execution) {
    auto *const src_node_item = MutableNodeItem(dep_node);
    GE_CHECK_NOTNULL(src_node_item);
    src_node_item->has_observer = true;
    node_item.dependents_for_execution.emplace_back(dep_node);
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentInData(const NodeItem &node_item,
                                                std::set<NodePtr> &dependent_for_execution) const {
  // The input tensors become valid after computation is done for parent nodes of type DEPEND_COMPUTE.
  // Wait for these parent nodes before execution.
  const auto &ge_node = node_item.node;
  for (const auto &in_anchor : ge_node->GetAllInDataAnchorsPtr()) {
    const auto &peer_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      GELOGD("[%s] input [%d] does not have peer anchor", node_item.NodeName().c_str(), in_anchor->GetIdx());
      continue;
    }
    auto src_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(src_node, src_node_item),
                      "[GetOrCreate][NodeItem] failed for [%s(%s)]",
                      src_node->GetName().c_str(), src_node->GetType().c_str());

    if ((src_node_item->shape_inference_type == DEPEND_COMPUTE) || IsFftsKernelNode(*node_item.GetOpDesc()) ||
        node_item.IsHcclOp() || src_node_item->IsHcclOp()) {
      GELOGD("[%s](%s) Add input data dependent node [%s](%s), shape inference type = %d", ge_node->GetName().c_str(),
             ge_node->GetType().c_str(), src_node->GetName().c_str(), src_node->GetType().c_str(),
             static_cast<int32_t>(src_node_item->shape_inference_type));
      src_node_item->has_observer = true;
      (void)dependent_for_execution.emplace(src_node);
    }

    if (src_node_item->shape_inference_type == DEPEND_SHAPE_RANGE) {
      GELOGD("[%s] Add input shape dependent node [%s] due to inference type = DEPEND_SHAPE_RANGE",
             node_item.NodeName().c_str(), src_node_item->NodeName().c_str());
      src_node_item->has_observer = true;
      (void)dependent_for_execution.emplace(src_node);
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentForFusedSubgraph(const NodeItem &node_item,
                                                          std::set<NodePtr> &dependencies) const {
  if (node_item.fused_subgraph == nullptr) {
    return SUCCESS;
  }

  std::vector<std::pair<OpDesc *, std::pair<OpDesc *, int32_t>>> data_deps;
  GE_CHK_STATUS_RET_NOLOG(CollectDependenciesForFusedGraph(node_item, data_deps));
  auto &subgraph_deps = node_item.fused_subgraph->data_dependencies;
  for (const auto &dep : data_deps) {
    const auto &data_op_desc = dep.first;
    int32_t parent_index = 0;
    if (!AttrUtils::GetInt(*data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(INTERNAL_ERROR, "[Get][Attr] failed, node:[%s(%s)] attr:[%s]", data_op_desc->GetName().c_str(),
             data_op_desc->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_INNER_ERR_MSG("E19999", "invoke GetInt failed, node:[%s(%s)]  attr:[%s]",
                        data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str(),
                        ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return INTERNAL_ERROR;
    }

    const auto &in_anchor = node_item.node->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(src_node, src_node_item));
    GELOGD("[%s::%s] Find the external source nodes that the nodes in the fused subgraph depend on, src_node = %s",
           node_item.NodeName().c_str(),
           dep.second.first->GetName().c_str(),
           src_node_item->NodeName().c_str());
    (void)src_node_item->to_const_output_id_list.emplace(peer_out_anchor->GetIdx());
    (void)dependencies.emplace(src_node);
    GELOGD("[%s][%s] infer shape dependent value from output of [%s:%d]",
           node_item.NodeName().c_str(),
           node_item.NodeType().c_str(),
           src_node_item->NodeName().c_str(),
           peer_out_anchor->GetIdx());
    subgraph_deps.emplace_back(std::make_pair(src_node_item->op_desc->GetId(), peer_out_anchor->GetIdx()), dep.second);
  }

  return SUCCESS;
}

Status HybridModelBuilder::UpdateAnchorStatus(const NodePtr &node) {
  if (NodeUtils::SetAllAnchorStatus(node) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Invoke][SetAllAnchorStatus] failed for node:[%s(%s)].",
           node->GetName().c_str(), node->GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "SetAllAnchorStatus failed for node:[%s(%s)].",
                      node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    const auto &peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Invoke][SetStatus] failed to set ANCHOR_SUSPEND, node:[%s(%s)].",
               node->GetName().c_str(), node->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "SetStatus failed to set ANCHOR_SUSPEND, node:[%s(%s)].",
                          node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    } else if (peer_anchor->GetOwnerNode()->GetType() == CONSTANT) {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_CONST) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Invoke][SetStatus] failed to set ANCHOR_CONST, node:[%s(%s)].",
               node->GetName().c_str(), node->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "SetStatus failed to set ANCHOR_CONST, node:[%s(%s)].",
                          node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    } else {
      if (AnchorUtils::SetStatus(anchor, ANCHOR_DATA) != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Invoke][SetStatus] failed to set ANCHOR_DATA, node:[%s(%s)].",
               node->GetName().c_str(), node->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "SetStatus failed to set ANCHOR_DATA, node:[%s(%s)].",
                          node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::DoUnlinkDataAnchors(const OutDataAnchorPtr &out_data_anchor,
                                               const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(out_data_anchor->Unlink(in_data_anchor),
                          "[Invoke][Unlink] failed to unlink %s(%s):%d from %s(%s):%d",
                          out_data_anchor->GetOwnerNode()->GetName().c_str(),
                          out_data_anchor->GetOwnerNode()->GetType().c_str(), out_data_anchor->GetIdx(),
                          in_data_anchor->GetOwnerNode()->GetName().c_str(),
                          in_data_anchor->GetOwnerNode()->GetType().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in unlinking %s:%d from %s:%d",
         out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(),
         in_data_anchor->GetOwnerNode()->GetName().c_str(),
         in_data_anchor->GetIdx());
  return SUCCESS;
}

Status HybridModelBuilder::DoLinkDataAnchors(const OutDataAnchorPtr &out_data_anchor,
                                             const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(out_data_anchor->LinkTo(in_data_anchor),
                          "[Invoke][LinkTo]Failed to link %s(%s):%d to %s(%s):%d",
                          out_data_anchor->GetOwnerNode()->GetName().c_str(),
                          out_data_anchor->GetOwnerNode()->GetType().c_str(), out_data_anchor->GetIdx(),
                          in_data_anchor->GetOwnerNode()->GetName().c_str(),
                          in_data_anchor->GetOwnerNode()->GetType().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in linking %s:%d to %s:%d",
         out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(),
         in_data_anchor->GetOwnerNode()->GetName().c_str(),
         in_data_anchor->GetIdx());
  return SUCCESS;
}

Status HybridModelBuilder::MergeInputNodes(ComputeGraph &compute_graph) {
  const auto &wrapped_node = compute_graph.GetParentNode();
  std::set<NodePtr> root_nodes;
  for (const auto &node : compute_graph.GetDirectNode()) {
    GE_CHK_STATUS_RET_NOLOG(MergeInputInData(node, wrapped_node, root_nodes));
  }

  // transfer in control edges to all root nodes
  for (const auto &root_node : root_nodes) {
    const auto &in_nodes = root_node->GetInAllNodes();
    const std::set<NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (const auto &in_control_node : wrapped_node->GetInControlNodes()) {
      if ((in_node_set.count(in_control_node) == 0U) &&
          (kHybridMergeInputSkipTypes.count(root_node->GetType()) == 0U)) {
        GELOGD("[%s] Restore control edge to [%s]", in_control_node->GetName().c_str(), root_node->GetName().c_str());
        GE_CHECK_NOTNULL(in_control_node->GetOutControlAnchor());
        (void)in_control_node->GetOutControlAnchor()->LinkTo(root_node->GetInControlAnchor());
      }
    }
  }

  wrapped_node->GetInControlAnchor()->UnlinkAll();
  return SUCCESS;
}

Status HybridModelBuilder::MergeInputInData(const NodePtr &node, const NodePtr &wrapped_node,
                                            std::set<NodePtr> &root_nodes) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() != DATA_TYPE) {
    if (node->GetInAllNodes().empty()) {
      (void)root_nodes.emplace(node);
    }
    return SUCCESS;
  }

  const auto &data_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(data_op_desc);

  int32_t parent_index = 0;
  if (!AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED, "[Invoke][GetInt] failed, node:[%s(%s)] attr:[%s]",
           data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        REPORT_INNER_ERR_MSG("E19999", "GetInt failed, node:[%s(%s)] attr:[%s]",
                          data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str(),
                          ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return FAILED;
  }

  const auto &wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
  GE_CHECK_NOTNULL(wrapped_node_in_anchor);
  const auto &src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  if ((src_out_anchor == nullptr) || (src_out_anchor->GetOwnerNode() == nullptr)) {
    return SUCCESS;
  }
  wrapped_node_in_anchor->UnlinkAll();

  // link src to outputs of DataNode
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      const auto &dst_node = peer_in_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dst_node);
      const auto &in_nodes = dst_node->GetInAllNodes();
      if (std::all_of(in_nodes.begin(), in_nodes.end(), [](const NodePtr &n) { return n->GetType() == DATA; })) {
        (void)root_nodes.emplace(dst_node);
      }
      GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(out_data_anchor, peer_in_data_anchor));
      GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, peer_in_data_anchor));
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::MergeNetOutputNode(ComputeGraph &compute_graph) {
  const auto &parent_node = compute_graph.GetParentNode();
  const NodePtr &net_output_node = compute_graph.FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("Graph has no netoutput no need to merge");
    return SUCCESS;
  }
  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  const auto all_in_nodes = net_output_node->GetInAllNodes();
  const auto all_out_nodes = parent_node->GetOutAllNodes();
  net_output_node->GetInControlAnchor()->UnlinkAll();
  parent_node->GetOutControlAnchor()->UnlinkAll();

  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    GE_CHK_STATUS_RET_NOLOG(MergeNetOutputInData(parent_node, net_output_desc, in_data_anchor));
  }

  // transfer out control edges
  const std::set<NodePtr> in_node_set(all_in_nodes.begin(), all_in_nodes.end());
  const std::set<NodePtr> out_node_set(all_out_nodes.begin(), all_out_nodes.end());
  for (const auto &src_node : in_node_set) {
    GELOGD("[%s] process in node.", src_node->GetName().c_str());
    const auto &out_nodes = src_node->GetOutAllNodes();
    const std::set<NodePtr> node_set(out_nodes.begin(), out_nodes.end());
    for (auto &dst_node : out_node_set) {
      if (node_set.count(dst_node) == 0U) {
        (void)src_node->GetOutControlAnchor()->LinkTo(dst_node->GetInControlAnchor());
        GELOGD("[%s] Restore control edge to [%s]", src_node->GetName().c_str(), dst_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::MergeNetOutputInData(const NodePtr &parent_node, const OpDescPtr &net_output_desc,
                                                const InDataAnchorPtr &in_data_anchor) {
  const auto &src_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_out_anchor);
  GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNode());
  GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(src_out_anchor, in_data_anchor));

  const auto anchor_index = in_data_anchor->GetIdx();
  const auto &input_desc = net_output_desc->MutableInputDesc(static_cast<uint32_t>(anchor_index));
  if (input_desc == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Invoke][MutableInputDesc][%s(%s)] Failed to get input desc[%d]",
           net_output_desc->GetName().c_str(), net_output_desc->GetType().c_str(), anchor_index);
        REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Failed to get input desc[%d].",
                          net_output_desc->GetName().c_str(), net_output_desc->GetType().c_str(), anchor_index);
    return INTERNAL_ERROR;
  }

  int32_t parent_index = 0;
  if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGW("SubGraph: %s NetOutput input tensor %d, attr %s not found.",
           net_output_desc->GetName().c_str(), anchor_index, ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return SUCCESS;
  }

  const OutDataAnchorPtr &parent_out_anchor = parent_node->GetOutDataAnchor(parent_index);
  GE_CHECK_NOTNULL(parent_out_anchor);
  for (InDataAnchorPtr &dst_in_anchor : parent_out_anchor->GetPeerInDataAnchors()) {
    if (dst_in_anchor == nullptr) {
      continue;
    }

    GE_CHECK_NOTNULL(dst_in_anchor->GetOwnerNode());
    GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(parent_out_anchor, dst_in_anchor));
    GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, dst_in_anchor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::UnfoldSubgraphs(const ComputeGraphPtr &root_graph, ComputeGraphPtr &merged_graph) {
  merged_graph = MakeShared<ComputeGraph>("MergedGraph");
  GE_CHECK_NOTNULL(merged_graph);
  merged_graph->SetGraphUnknownFlag(root_graph->GetGraphUnknownFlag());
  merged_graph->SetGraphID(root_graph->GetGraphID());
  for (const auto &node : root_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto &op_type = node->GetType();
    if ((op_type != PARTITIONEDCALL) || (IsFftsGraphNode(*op_desc))) {
      (void)merged_graph->AddNode(node);
      GELOGD("[%s] Node added to merged graph.", op_desc->GetName().c_str());
      continue;
    }

    const auto &subgraph = NodeUtils::GetSubgraph(*node, kHybridSubgraphIndex);
    GE_CHECK_NOTNULL(subgraph);
    if (!subgraph->GetGraphUnknownFlag()) {
      (void)merged_graph->AddNode(node);
      GELOGD("[%s] Known shape partitioned call added to merged graph.", op_desc->GetName().c_str());
      continue;
    }
    if (op_desc->HasAttr(ATTR_STAGE_LEVEL)) {
      int64_t stage_level = std::numeric_limits<int64_t>::max();
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level)) {
        for (const auto &stage_node : subgraph->GetAllNodes()) {
          GELOGD("Set ATTR_STAGE_LEVEL on node %s, stage_level=%ld", stage_node->GetName().c_str(), stage_level);
          (void)AttrUtils::SetInt(stage_node->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level);
        }
      }
    }
    GE_CHK_STATUS_RET_NOLOG(UnfoldSubgraph(root_graph, merged_graph, *subgraph));
  }

  // invoke before adding subgraphs. in case modify node id in known-shaped subgraphs.
  GE_CHK_GRAPH_STATUS_RET(merged_graph->TopologicalSorting(),
                          "[Invoke][TopologicalSorting]Failed to invoke TopologicalSorting on merged graph.");
  GE_DUMP(merged_graph, "hybrid_merged_graph_BeforeStageSort");
  merged_graph->TopologicalSorting([](const NodePtr &a, const NodePtr &b) -> bool {
    int64_t a_level = std::numeric_limits<int64_t>::max();
    (void)AttrUtils::GetInt(a->GetOpDesc(), ATTR_STAGE_LEVEL, a_level);
    int64_t b_level = std::numeric_limits<int64_t>::max();
    (void)AttrUtils::GetInt(b->GetOpDesc(), ATTR_STAGE_LEVEL, b_level);
    return a_level < b_level;
  });

  for (auto &remained_subgraph : root_graph->GetAllSubgraphs()) {
    GELOGD("Adding subgraph [%s] to merged-graph.", remained_subgraph->GetName().c_str());
    GE_CHK_GRAPH_STATUS_RET(merged_graph->AddSubgraph(remained_subgraph),
                            "[Invoke][AddSubgraph]Failed to add subgraph [%s]",
                            remained_subgraph->GetName().c_str());
    remained_subgraph->SetParentGraph(merged_graph);
  }

  return SUCCESS;
}

Status HybridModelBuilder::UnfoldSubgraph(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &parent_graph,
                                          ComputeGraph &sub_graph, const uint32_t depth) {
  if (depth >= kHybridSubgraphRecursion) {
    GELOGE(FAILED, "[Invoke][Unfold]There are too much recursion:%u > max:%u", depth, kHybridSubgraphRecursion);
    REPORT_INNER_ERR_MSG("E19999", "[Unfold]There are too much recursion:%u > max:%u", depth, kHybridSubgraphRecursion);
    return FAILED;
  }
  const auto &parent_node = sub_graph.GetParentNodeBarePtr();
  GE_CHECK_NOTNULL(parent_node);

  GE_CHK_STATUS_RET(MergeInputNodes(sub_graph),
                    "[Invoke][MergeInputNodes][%s] Failed to merge data nodes for subgraph",
                    sub_graph.GetName().c_str());
  GE_CHK_STATUS_RET(MergeNetOutputNode(sub_graph),
                    "[Invoke][MergeNetOutputNode][%s] Failed to merge net output nodes for subgraph",
                    sub_graph.GetName().c_str());
  GELOGD("[%s] Done merging subgraph inputs and outputs successfully", sub_graph.GetName().c_str());

  for (const auto &sub_node : sub_graph.GetDirectNode()) {
    const auto sub_op_type = sub_node->GetType();
    if ((sub_op_type == DATA_TYPE) || (sub_op_type == NETOUTPUT)) {
      continue;
    }
    const auto &op_desc = sub_node->GetOpDesc();
    if ((sub_op_type == PARTITIONEDCALL) && (!IsFftsGraphNode(*op_desc))) {
      const auto &sub_sub_graph = NodeUtils::GetSubgraph(*sub_node, kHybridSubgraphIndex);
      GE_CHECK_NOTNULL(sub_sub_graph);
      if (sub_sub_graph->GetGraphUnknownFlag()) {
        GE_CHK_STATUS_RET(UnfoldSubgraph(root_graph, parent_graph, *sub_sub_graph, depth + 1U),
                          "[Invoke][UnfoldSubgraph][%s] Failed to merge subgraph",
                          sub_sub_graph->GetName().c_str());
        continue;
      }
    }

    if (!op_desc->GetSubgraphInstanceNames().empty()) {
      for (size_t i = 0U; i < op_desc->GetSubgraphInstanceNames().size(); ++i) {
        const auto &sub_sub_graph = NodeUtils::GetSubgraph(*sub_node, static_cast<uint32_t>(i));
        GE_CHECK_NOTNULL(sub_sub_graph);
        sub_sub_graph->SetParentGraph(parent_graph);
      }
    }
    (void)parent_graph->AddNode(sub_node);
    GELOGD("[%s:%s] added to parent graph: [%s].",
           sub_graph.GetName().c_str(), sub_node->GetName().c_str(), parent_graph->GetName().c_str());
    (void)sub_node->SetOwnerComputeGraph(parent_graph);
  }

  GELOGD("[%s] Done merging subgraph. remove it from root graph", sub_graph.GetName().c_str());
  auto anchors = parent_node->GetAllInDataAnchorsPtr();
  std::for_each(anchors.begin(), anchors.end(), [](InDataAnchor *anchor) {
      anchor->UnlinkAll();
  });
  root_graph->RemoveSubgraph(sub_graph.GetName());
  return SUCCESS;
}

Status HybridModelBuilder::BuildOutputMapping(GraphItem &graph_item,
                                              const NodeItem &node_item,
                                              const bool is_root_graph) const {
  const auto output_size = node_item.num_inputs;
  graph_item.output_edges_.resize(static_cast<size_t>(output_size));

  for (const auto &in_data_anchor : node_item.node->GetAllInDataAnchorsPtr()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    const auto &src_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);

    auto *const src_node_item = GetNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    const auto output_idx = in_data_anchor->GetIdx();
    GELOGI("Output[%d], node = %s, output_index = %d, output_offset = %d",
           output_idx, src_node_item->NodeName().c_str(),
           peer_out_anchor->GetIdx(), src_node_item->output_start + peer_out_anchor->GetIdx());

    GE_CHECK_LE(output_idx, (output_size - 1));
    graph_item.output_edges_[static_cast<size_t>(output_idx)] = {src_node_item, peer_out_anchor->GetIdx()};
  }

  if (!is_root_graph) {
    for (int32_t i = 0; i < output_size; ++i) {
      int32_t p_index = i;
      // Net output of Subgraph of while do not have parent index
      const auto &input_desc = node_item.op_desc->GetInputDesc(static_cast<uint32_t>(i));
      if (AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, p_index)) {
        GELOGD("[%s] Parent index not set for input[%d].", node_item.NodeName().c_str(), i);
      }

      graph_item.output_index_mapping_.emplace_back(p_index);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::CheckForObserver(const ComputeGraph &graph) const {
  for (const auto &node : graph.GetDirectNode()) {
    NodeItem *node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node, node_item));
    if (node_item->has_observer) {
      hybrid_model_.has_observer_ = true;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::SetStageCache(const ComputeGraph &graph, const GraphItem &stage_graph) const {
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    NodeItem *node_item = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node, node_item));
    GE_CHECK_NOTNULL(node_item);
    // Event sync is not added in the pipeline scenario will lead to aic error.
    // Now, we add callback by default, and then do optimization after the problem is solved.
    if (stage_graph.NumGroups() > 1UL) {
      node_item->has_observer = true;
    }
    for (int32_t i = 0; i < node_item->num_outputs; ++i) {
      const auto &output_nodes = node_item->outputs[static_cast<size_t>(i)];
      for (const auto &output_node : output_nodes) {
        auto &dst_node_item = output_node.second;
        GE_CHK_STATUS_RET_NOLOG(stage_graph.GetStageCache().CreatePropagator(*node_item, i, *dst_node_item,
                                                                             output_node.first));
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::LoadGraph() {
  auto root_graph = hybrid_model_.root_graph_;
  if (!GetContext().GetHostExecFlag()) {
    hybrid_model_.orig_root_graph_ = root_graph;
    if (gert::GraphUnfolder::IsGraphNeedUnfold(root_graph)) {
      std::shared_ptr<ComputeGraph> merged_graph;
      GELOGI("Before merging subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu",
             root_graph->GetDirectNodesSize(), root_graph->GetAllNodesSize());
      GE_CHK_GRAPH_STATUS_RET(UnfoldSubgraphs(root_graph, merged_graph),
                              "[Invoke][UnfoldSubgraphs]Fail to unfold subgraphs, model_name[%s]", ModelName().c_str());
      root_graph = std::move(merged_graph);
      GELOGI("After merging subgraphs DirectNodesSize = %zu, GetAllNodesSize = %zu",
             root_graph->GetDirectNodesSize(), root_graph->GetAllNodesSize());
    }
  }

  hybrid_model_.root_graph_ = root_graph;
  (void)RecoverShapeConsistency(*hybrid_model_.root_graph_);
  GE_CHK_STATUS_RET(RelinkNextIteration(), "[Relink][NextIteration] failed for model_name[%s]", ModelName().c_str());
  // Reset node id by topological order across all subgraphs
  int64_t op_index = 0;
  for (const auto &node : root_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto &parent_graph = node->GetOwnerComputeGraph();
    // No need to update nodes in known subgraph
    if ((parent_graph != nullptr) && (!parent_graph->GetGraphUnknownFlag())) {
      continue;
    }

    // No need to update nodes in ffts plus subgraph
    const auto &functional_node = parent_graph->GetParentNodeBarePtr();
    if ((functional_node != nullptr) && IsFftsGraphNode(*functional_node->GetOpDescBarePtr())) {
      continue;
    }

    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetId(op_index++);
  }
  GE_DUMP(root_graph, "hybrid_merged_graph");
  GE_CHK_STATUS_RET(LoadDynamicSubgraph(root_graph, true),
                    "[Invoke][LoadDynamicSubgraph]Failed to load root graph, model_name[%s]", ModelName().c_str());
  GE_CHK_GRAPH_STATUS_RET(ge::RecoverIrDefinitions(root_graph), "Failed to recover ir definitions");
  GE_CHK_STATUS_RET(CheckForObserver(*root_graph),
                    "[Invoke][CheckForObserver]Failed to load root graph, model_name[%s]", ModelName().c_str());
  GELOGD("Done loading root graph successfully.");
  GE_CHK_STATUS_RET(hybrid_model_.root_graph_item_->GroupNodes(),
                    "[Invoke][GroupNodes]Failed to group nodes for root graph, model_name[%s]", ModelName().c_str());
  GE_CHK_STATUS_RET(SetStageCache(*root_graph, *hybrid_model_.root_graph_item_),
                    "[Set[StageCache]Failed to for root graph, model_name[%s]", ModelName().c_str());

  for (const auto &sub_graph : root_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(sub_graph);
    GELOGD("Start to load subgraph [%s]", sub_graph->GetName().c_str());
    const auto &parent_node = sub_graph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    auto *const parent_node_item = MutableNodeItem(parent_node);
    // parent node is in another known subgraph
    if (parent_node_item == nullptr) {
      GELOGD("[%s] Subgraph is in another known shaped subgraph, skip it.", sub_graph->GetName().c_str());
      continue;
    }

    if (sub_graph->GetGraphUnknownFlag()) {
      GE_CHK_STATUS_RET(LoadDynamicSubgraph(sub_graph, false),
                        "[Invoke][LoadDynamicSubgraph]Failed to load subgraph: [%s]",
                        sub_graph->GetName().c_str());
      GE_CHK_STATUS_RET(SetStageCache(*sub_graph, *hybrid_model_.subgraph_items_[sub_graph->GetName()]),
                        "[Set[StageCache]Failed to for sub graph, model_name_:%s.", sub_graph->GetName().c_str());
    } else {
      // if parent is function control op. need add a virtual partitioned call
      if (parent_node_item->IsControlFlowV2Op()) {
        GE_CHK_STATUS_RET(LoadKnownShapedSubgraph(*sub_graph, *parent_node_item),
                          "[Invoke][LoadKnownShapedSubgraph]Failed to load function control op subgraph [%s]",
                          sub_graph->GetName().c_str());
      }
    }
  }
  for (const auto &it : hybrid_model_.known_shape_sub_models_) {
    auto *const node_item = MutableNodeItem(it.first);
    GE_CHECK_NOTNULL(node_item);
    const auto compute_graph = it.second->GetGraph();
    GE_CHECK_NOTNULL(compute_graph, "[Get][Name] of subgraph failed");
    const auto &subgraph = hybrid_model_.GetRootGraph()->GetSubgraph(compute_graph->GetName());
    GE_CHECK_NOTNULL(subgraph);
    GE_CHK_STATUS_RET(IdentifyVariableOutputs(*node_item, subgraph),
                      "[Invoke][IdentifyVariableOutputs] [%s(%s)] Failed to identify ref outputs.",
                      node_item->NodeName().c_str(), node_item->NodeType().c_str());
  }
  GE_CHK_STATUS_RET(ParseDependentByParallelGroup(),
                    "[Invoke][ParseDependentByParallelGroup]Failed to establish dependencies for hccl ops, "
                    "model_name[%s]", ModelName().c_str());
  GELOGI("Done loading all subgraphs successfully.");
  return SUCCESS;
}

const NodeItem *HybridModelBuilder::GetNodeItem(const NodePtr &node) const {
  return hybrid_model_.GetNodeItem(node);
}

NodeItem *HybridModelBuilder::MutableNodeItem(const NodePtr &node) const {
  return hybrid_model_.MutableNodeItem(node);
}

Status HybridModelBuilder::VarNodeToTensor(const NodePtr &var_node, std::unique_ptr<TensorValue> &tensor) const {
  const std::string var_name = var_node->GetName();
  const auto &tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0U);
  GE_CHECK_NOTNULL(tensor_desc);
  uint8_t *var_logic = nullptr;
  GE_CHK_STATUS_RET(var_manager_->GetVarAddr(var_name, *tensor_desc, var_logic),
                    "[Invoke][GetVarAddr]Failed to get var addr. var_name = %s, session_id = %ld",
                    var_name.c_str(),
                    hybrid_model_.GetSessionId());

  rtMemType_t memory_type = RT_MEMORY_HBM;
  uint32_t mem_type = 0U;
  if (AttrUtils::GetInt(var_node->GetOpDesc(), ATTR_OUTPUT_MEMORY_TYPE, mem_type) && (mem_type == 1U)) {
    memory_type = RT_MEMORY_RDMA_HBM;
  }
  uint8_t *const dev_mem = var_manager_->GetVarMemoryAddr(var_logic, memory_type, hybrid_model_.device_id_);
  if (dev_mem == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Invoke][GetVarMemoryAddr]Failed to copy var %s(%s) from device,"
           "cant not get var addr from logic addr %p",
           var_node->GetName().c_str(), var_node->GetType().c_str(), var_logic);
    REPORT_INNER_ERR_MSG("E19999", "GetVarMemoryAddr failed, Failed to copy var %s(%s) from device,"
                      "cant not get var addr from logic addr %p",
                      var_node->GetName().c_str(), var_node->GetType().c_str(), var_logic);
    return INTERNAL_ERROR;
  }

  const int64_t var_size = CalcTensorSizeInBytes(*tensor_desc);
  GE_CHECK_GE(var_size, 0);
  tensor = MakeUnique<TensorValue>(dev_mem, static_cast<size_t>(var_size));
  GE_CHECK_NOTNULL(tensor);
  GELOGI("Get var memory addr %p for node %s, size = %ld, mem_type=%u", dev_mem, var_name.c_str(), var_size, mem_type);
  return SUCCESS;
}

Status HybridModelBuilder::CopyConstantData(const NodePtr &node, const GeTensor &tensor,
                                            const std::unique_ptr<TensorValue> &var_tensor) const {
  const auto output_size = var_tensor->GetSize();
  const auto output_addr = var_tensor->MutableData();

  GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] mem_size[%zu] datasize[%zu]",
         runtime_param_.graph_id, node->GetName().c_str(), 0, output_addr, output_size, tensor.GetData().size());
  GE_CHK_RT_RET(rtMemcpy(output_addr, output_size, tensor.GetData().data(), tensor.GetData().size(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status HybridModelBuilder::AssignUninitializedConstantOps() const {
  if (GetContext().GetHostExecFlag()) {
    GELOGI("no need to assign when exec on host.");
    return SUCCESS;
  }
  for (const auto &it : constant_op_nodes_) {
    const std::string &var_name = it.first;
    const NodePtr &var_node = it.second;
    const auto &tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(tensor_desc);
    if (!var_manager_->IsVarExist(var_name, *tensor_desc)) {
      // allocate constant
      GELOGD("[%s] Constant not allocated during graph building. now allocate it.", var_name.c_str());
      const auto &op_desc = var_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      GE_CHK_STATUS_RET(var_manager_->AssignVarMem(var_name, op_desc, *tensor_desc, RT_MEMORY_HBM));
      GE_CHK_STATUS_RET(var_manager_->SetAllocatedGraphId(var_name, runtime_param_.graph_id));
    }
  }

  for (const auto &it : hybrid_model_.device_variable_nodes_) {
    const std::string &var_name = it.first;
    const NodePtr &var_node = it.second;
    const auto &tensor_desc = var_node->GetOpDesc()->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(tensor_desc);
    if (!var_manager_->IsVarExist(var_name, *tensor_desc)) {
      // allocate constant
      GELOGD("[%s] Constant not allocated during graph building. now allocate it.", var_name.c_str());
      const auto &op_desc = var_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      GE_CHK_STATUS_RET(var_manager_->AssignVarMem(var_name, op_desc, *tensor_desc, RT_MEMORY_HBM));
      GE_CHK_STATUS_RET(VarMemAssignUtil::AssignData2Fp32Var(var_node, runtime_param_.session_id));
      GE_CHK_STATUS_RET(var_manager_->SetAllocatedGraphId(var_name, runtime_param_.graph_id));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitConstantOps() const {
  const bool is_host_exec = GetContext().GetHostExecFlag();
  for (const auto &it : constant_op_nodes_) {
    const std::string &var_name = it.first;
    const NodePtr &var_node = it.second;
    const auto &op_desc = var_node->GetOpDesc();
    if (var_node->GetType() != CONSTANTOP) {
      continue;
    }
    const auto &v_weights = ModelUtils::GetWeights(op_desc);
    if (v_weights.empty()) {
      GELOGE(INTERNAL_ERROR, "[Check][Size][%s(%s)] Constant op has no weight",
             var_node->GetName().c_str(), var_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
    const auto &ge_tensor = v_weights[0U];
    GE_CHECK_NOTNULL(ge_tensor);

    std::unique_ptr<TensorValue> var_tensor;
    if (is_host_exec) {
      // Address for eigen kernel should be aligned with 16 bytes
      // Tensors return by api GetWeights share data with proto, whose addr is not confirmed to be aligned
      GeTensor aligned_tensor = ge_tensor->Clone();
      GELOGD("Init tensor with host constant %s size = %zu", var_name.c_str(), aligned_tensor.MutableData().GetSize());
      if (aligned_tensor.GetData().size() != 0U) {
        auto &mem_instance = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM);
        GE_CHECK_NOTNULL(mem_instance.Malloc(aligned_tensor.GetAlignedPtr(), aligned_tensor.GetData().size()));
        var_tensor = MakeUnique<TensorValue>(aligned_tensor.MutableData().data(), aligned_tensor.GetData().size());
      } else {
        var_tensor = MakeUnique<TensorValue>(nullptr, 0U);
      }
      GE_CHECK_NOTNULL(var_tensor);
    } else {
      GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, var_tensor));
      GELOGD("Init const op tensor. name = %s, size = %ld", var_name.c_str(), var_tensor->GetSize());
      var_tensor->SetName("ConstOp_" + var_name);
      if (ge_tensor->GetData().size() != 0U) {
        GE_CHK_STATUS_RET_NOLOG(CopyConstantData(var_node, *ge_tensor, var_tensor));
      } else {
        GELOGI("[%s] Const op has no weight data.", op_desc->GetName().c_str());
      }
    }

    (void)hybrid_model_.variable_tensors_.emplace(var_name, std::move(var_tensor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitFileConstantInHost(const int64_t output_size, const OpDescPtr &op_desc,
                                                  std::unique_ptr<TensorValue> &var_tensor) const {
  if (output_size > 0) {
    size_t malloc_size = static_cast<size_t>(output_size);
    void *const mem_addr = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Malloc(malloc_size);
    GE_CHK_BOOL_RET_STATUS((mem_addr != nullptr), MEMALLOC_FAILED, "Failed to malloc for node[%s]",
                           op_desc->GetName().c_str());
    std::string file_path;
    size_t offset = 0U;
    size_t length = 0U;
    GE_CHK_STATUS_RET(FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map_, file_path, offset, length),
                      "Failed to get file path.");
    const size_t file_length = (length == 0U ? static_cast<size_t>(output_size) : length);
    GE_CHK_STATUS_RET(FileConstantUtils::CopyOneWeightFromFile(mem_addr, file_path, offset, file_length, malloc_size),
                      "Failed to copy data.");
    var_tensor = MakeUnique<TensorValue>(mem_addr, file_length);
  } else {
    var_tensor = MakeUnique<TensorValue>(nullptr, 0U);
  }
  return SUCCESS;
}

Status HybridModelBuilder::InitFileConstantInDevice(const int64_t output_size, const OpDescPtr &op_desc,
                                                    const TensorValue &tensor_value) const {
  if (tensor_value.GetSize() > 0U) {
    std::string file_path;
    size_t offset = 0U;
    size_t length = 0U;
    GE_CHK_STATUS_RET(FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map_, file_path, offset, length),
                      "Failed to get file path.");
    size_t left_size = tensor_value.GetSize();
    const size_t file_length = (length == 0U ? static_cast<size_t>(output_size) : length);
    return FileConstantUtils::CopyOneWeightFromFile(tensor_value.GetData(), file_path, offset, file_length, left_size);
  } else {
    GELOGI("File constant [%s] has no weight data.", op_desc->GetName().c_str());
    return SUCCESS;
  }
}

Status HybridModelBuilder::InitFileConstantOps() {
  GE_CHK_STATUS_RET(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map_), "Failed to get file path.");
  const bool is_host_exec = GetContext().GetHostExecFlag();
  for (const auto &it : constant_op_nodes_) {
    const std::string &var_name = it.first;
    const NodePtr &var_node = it.second;
    const auto &op_desc = var_node->GetOpDesc();
    if (var_node->GetType() != FILECONSTANT) {
      continue;
    }

    const ConstGeTensorDescPtr tensor_desc = op_desc->GetOutputDescPtr(0U);
    GE_CHECK_NOTNULL(tensor_desc);
    int64_t weight_size = 0;
    GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weight_size),
                      "Failed to get file constant size.");

    if (!hybrid_model_.GetFileConstantWeightDir().empty()) {
      GE_CHK_STATUS_RET(FileConstantUtils::SetExternalPath(op_desc, hybrid_model_.GetFileConstantWeightDir()),
                        "Failed to set external path.");
    }
    std::unique_ptr<TensorValue> var_tensor;
    if (is_host_exec) {
      GE_CHK_STATUS_RET(InitFileConstantInHost(weight_size, op_desc, var_tensor),
                        "Failed to init file constant in host.");
      GE_CHECK_NOTNULL(var_tensor);
    } else {
      GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, var_tensor));
      GELOGD("Init const op tensor. name = %s, size = %ld", var_name.c_str(), var_tensor->GetSize());
      var_tensor->SetName("FileConstOp_" + var_name);
      const auto var_instance = VarManager::Instance(runtime_param_.session_id);
      GE_ASSERT_NOTNULL(var_instance);
      if (!var_instance->IsVarReady(var_node->GetName(), *tensor_desc, hybrid_model_.device_id_)) {
        GELOGI("process the file constant op:%s", var_node->GetName().c_str());
        GE_CHK_STATUS_RET(InitFileConstantInDevice(weight_size, op_desc, *var_tensor),
                          "Failed to init file constant in device.");
        var_instance->SetVarIsReady(var_node->GetName(), *tensor_desc, hybrid_model_.device_id_);
      }
    }
    (void)hybrid_model_.variable_tensors_.emplace(var_name, std::move(var_tensor));
  }
  return SUCCESS;
}

void *HybridModelBuilder::GetOrCreateVarMem(const std::string &var_name,
                                            const OpDescPtr &var_desc,
                                            const rtMemType_t memory_type) const {
  const GeTensorDesc &output_tensor = var_desc->GetOutputDesc(0U);
  if (var_manager_->IsVarExist(var_name, output_tensor)) {
    GELOGD("Skip initialized %s %s with out[0]", var_desc->GetType().c_str(), var_desc->GetName().c_str());
  } else {
    GELOGI("Assign %s %s logical memory with out[0]", var_desc->GetType().c_str(), var_name.c_str());
    GE_ASSERT_SUCCESS(var_manager_->AssignVarMem(var_name, var_desc, output_tensor, memory_type),
                      "[Assign][VarMem] for %s failed.", var_name.c_str());
  }
  uint8_t *var_addr = nullptr;
  GE_ASSERT_SUCCESS(var_manager_->GetVarAddr(var_name, output_tensor, var_addr),
                    "[Get][VarAddr] failed, var name[%s]", var_name.c_str());
  return static_cast<void *>(var_addr);
}

Status HybridModelBuilder::InitVariableTensors() const {
  for (auto &it : hybrid_model_.device_variable_nodes_) {
    std::string var_name = it.first;
    NodePtr &var_node = it.second;
    std::unique_ptr<TensorValue> tensor;
    GE_CHK_STATUS_RET_NOLOG(VarNodeToTensor(var_node, tensor));
    GELOGD("Init variable tensor. name = %s, size = %ld, addr = %p",
           var_name.c_str(),
           tensor->GetSize(),
           tensor->GetData());
    tensor->SetName("Var_" + var_name);
    (void)hybrid_model_.variable_tensors_.emplace(var_name, std::move(tensor));
  }

  for (const auto &it : hybrid_model_.host_variable_nodes_) {
    const auto &op_desc = it.second->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const GeTensorDesc &output_tensor = op_desc->GetOutputDesc(0U);
    int64_t tensor_size = 0;
    if (TensorUtils::CalcTensorMemSize(output_tensor.GetShape(), output_tensor.GetFormat(),
                                       output_tensor.GetDataType(), tensor_size) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "CalcTensorMemSize failed, node:%s(%s)",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Calculate][TensorMemSize] failed, node:%s(%s)",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return INTERNAL_ERROR;
    }

    // Host variable will be assigned to allocated shared memory first.
    const auto &var_name = it.first;
    SharedMemInfo mem_info;
    void *mem_addr = nullptr;
    auto &mem_instance = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM);
    if (HostMemManager::Instance().QueryVarMemInfo(var_name, mem_info)) {
      mem_addr = mem_instance.Malloc(mem_info.host_aligned_ptr, static_cast<size_t>(tensor_size));
    } else {
      mem_addr = GetOrCreateVarMem(var_name, op_desc, RT_MEMORY_HOST);
    }

    if (mem_addr == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "[Malloc][HostMem] for variable [%s(%s)] failed.",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(MEMALLOC_FAILED, "[Malloc][HostMem] for variable [%s(%s)] failed.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return MEMALLOC_FAILED;
    }
    GELOGD("Host variable [%s] malloc success, size=%ld, addr=%p.", it.first.c_str(), tensor_size, mem_addr);

    std::unique_ptr<TensorValue> tensor = MakeUnique<TensorValue>(mem_addr, static_cast<size_t>(tensor_size));
    GE_CHECK_NOTNULL(tensor);
    (void)hybrid_model_.variable_tensors_.emplace(it.first, std::move(tensor));
  }

  return SUCCESS;
}

Status HybridModelBuilder::InitWeights() const {
  // For constant in root graph
  const bool is_host_exec = GetContext().GetHostExecFlag();
  for (const auto &subgraph_model : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    const auto weight_data = subgraph_model.second->GetWeightData();
    const size_t weight_size = subgraph_model.second->GetWeightSize();
    if (weight_size == 0U) {
      GELOGD("weight is empty. subgraph_name = %s", subgraph_model.first.c_str());
      continue;
    }

    uint32_t device_id = is_host_exec ? static_cast<uint32_t>(-1) : hybrid_model_.device_id_;
    auto *const mem_allocator = NpuMemoryAllocator::GetAllocator(device_id, nullptr);
    GE_CHECK_NOTNULL(mem_allocator);
    auto sub_weight_buffer = TensorBuffer::Create(mem_allocator, weight_size);
    GE_CHECK_NOTNULL(sub_weight_buffer);
    const auto weight_base = PtrToValue(sub_weight_buffer->GetData());

    if (is_host_exec) {
      GE_CHK_STATUS_RET(GeMemcpy(PtrToPtr<void, uint8_t>(sub_weight_buffer->GetData()), sub_weight_buffer->GetSize(),
                                 weight_data, weight_size),
                        "Copy weight data failed.");
    } else {
      GE_CHK_RT_RET(rtMemcpy(sub_weight_buffer->GetData(),
                             sub_weight_buffer->GetSize(),
                             weight_data,
                             weight_size,
                             RT_MEMCPY_HOST_TO_DEVICE));
    }

    GELOGI("Init weight mem successfully, weight base %p, weight size = %zu",
           sub_weight_buffer->GetData(), sub_weight_buffer->GetSize());
    auto subgraph = subgraph_model.second->GetGraph();
    if (subgraph != ge_root_model_->GetRootGraph()) {
      subgraph = hybrid_model_.root_graph_->GetSubgraph(subgraph_model.first);
    } else {
      subgraph = hybrid_model_.root_graph_;
    }
    GE_CHECK_NOTNULL(subgraph);
    (void)hybrid_model_.weight_buffer_map_.emplace(subgraph->GetName(), std::move(sub_weight_buffer));
    for (const auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() != CONSTANT) {
        continue;
      }

      const auto &op_desc = node->GetOpDesc();
      const auto &v_weights = ModelUtils::GetWeights(op_desc);
      if (v_weights.empty()) {
        GELOGE(INTERNAL_ERROR, "[Invoke][GetWeights][%s(%s)] Constant has no value",
               node->GetName().c_str(), node->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Constant has no value.",
                          node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
      const GeTensor *const ge_tensor = v_weights[0U].get();
      GE_CHECK_NOTNULL(ge_tensor);
      const GeTensorDesc &tensor_desc = ge_tensor->GetTensorDesc();
      int64_t tensor_size = 0;
      const auto output_desc = op_desc->MutableOutputDesc(0U);
      GE_CHECK_NOTNULL(output_desc);
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetSize(*output_desc, tensor_size),
                              "[Invoke][GetSize][%s(%s)] Failed to get output tensor size",
                              node->GetName().c_str(), node->GetType().c_str());
      int64_t data_offset = 0;
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetDataOffset(tensor_desc, data_offset),
                              "[Invoke][GetDataOffset][%s(%s)] Failed to get data offset",
                              node->GetName().c_str(), node->GetType().c_str());
      GELOGD("[%s] Start to init Constant node [%s], size = %ld, offset = %ld",
             ModelName().c_str(), node->GetName().c_str(), tensor_size, data_offset);

      auto tensor_data = TensorBuffer::Create(ValueToPtr(weight_base + static_cast<size_t>(data_offset)),
                                              static_cast<size_t>(tensor_size));
      GE_CHECK_NOTNULL(tensor_data);
      std::unique_ptr<TensorValue> constant_tensor = MakeUnique<TensorValue>(std::move(tensor_data));
      GE_CHECK_NOTNULL(constant_tensor);
      constant_tensor->SetName("Constant_" + op_desc->GetName());
      (void)hybrid_model_.constant_tensors_.emplace(node, std::move(constant_tensor));
      GELOGD("[%s] Constant node [%s] added, size = %ld", ModelName().c_str(), node->GetName().c_str(), tensor_size);
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::LoadTask(NodeItem &node_item) const {
  const auto &node_ptr = node_item.node;
  GELOGD("[%s] Start to build kernel task", node_ptr->GetName().c_str());
  const auto load_ret = node_item.node_executor->LoadTask(hybrid_model_,
                                                          node_ptr,
                                                          node_item.kernel_task);
  if ((load_ret != UNSUPPORTED) && (load_ret != SUCCESS)) {
    GELOGE(load_ret, "[Invoke][LoadTask][%s(%s)] Failed to load task",
           node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Failed to load task",
                      node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
    return load_ret;
  }

  GELOGD("[%s] Done loading task successfully.", node_ptr->GetName().c_str());
  return SUCCESS;
}

Status HybridModelBuilder::LoadTasks() const {
  GE_CHK_STATUS_RET(CheckAicpuOpList(), "[Check][AicpuOpList] failed.");
  std::map<int64_t, std::map<std::string, NodeItem *>> ordered_partitioned_calls;
  for (const auto &it : hybrid_model_.node_items_) {
    auto &node_item = it.second;
    if (node_item->node_type == NETOUTPUT) {
      continue;
    }
    if (((node_item->node_type == PARTITIONEDCALL) && (!IsFftsGraphNode(*node_item->op_desc)))
        || (node_item->IsFftsSubNode())) {
      ordered_partitioned_calls[node_item->node_id][node_item->node_name] = node_item.get();
      continue;
    }
    GE_CHK_STATUS_RET_NOLOG(LoadTask(*node_item));
  }

  // HCCL operators need to be loaded in the same order across different processes
  for (const auto &it : ordered_partitioned_calls) {
    for (const auto &it2 : it.second) {
      GE_CHK_STATUS_RET_NOLOG(LoadTask(*it2.second));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadGeModel(ComputeGraph &sub_graph, const GeModelPtr &ge_model) {
  const auto &parent_node = sub_graph.GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  GE_CHECK_NOTNULL(ge_model->GetModelTaskDefPtr());
  const auto op_type = parent_node->GetType();
  if (IsControlFlowV2Op(op_type)) {
    GELOGD("Set ge_model for control op subgraph: [%s], task_size = %d",
           sub_graph.GetName().c_str(),
           ge_model->GetModelTaskDefPtr()->task_size());
    (void)subgraph_models_.emplace(sub_graph.GetName(), ge_model);
  } else {
    GELOGD("Set ge_model for subgraph: [%s], task_size = %d",
           sub_graph.GetName().c_str(),
           ge_model->GetModelTaskDefPtr()->task_size());
    (void)hybrid_model_.known_shape_sub_models_.emplace(parent_node, ge_model);
  }

  GE_CHK_STATUS_RET_NOLOG(InitHcclExecutorOnDemand(ge_model));
  return SUCCESS;
}

Status HybridModelBuilder::InitHcclExecutorOnDemand(const GeModelPtr &ge_model) {
  if (NodeExecutorManager::GetInstance().IsExecutorInitialized(NodeExecutorManager::ExecutorType::HCCL)) {
    return SUCCESS;
  }

  // HCCL tasks in known-shaped subgraph which resides in a dynamic root graph
  // still depends on the initialization of the HcclExecutor
  const auto &tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int32_t i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    if (task_type == ModelTaskType::MODEL_TASK_HCCL) {
      const NodeExecutor *unused = nullptr;
      return NodeExecutorManager::GetInstance().GetOrCreateExecutor(NodeExecutorManager::ExecutorType::HCCL, unused);
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::IndexTaskDefs(const ComputeGraphPtr &sub_graph, const GeModelPtr &ge_model) {
  // index task defs
  GELOGD("To index tasks for subgraph: %s", sub_graph->GetName().c_str());
  std::unordered_map<int64_t, NodePtr> node_map;
  for (const auto &node : sub_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    const auto node_id = node->GetOpDesc()->GetId();
    GELOGD("op_index = %ld, node_name = %s", node_id, node->GetName().c_str());
    (void)node_map.emplace(node_id, node);
    LoadSgtKernelBinToOpDesc(*node->GetOpDesc(), sub_graph, ge_model); // for FFTS Plus.
  }

  // The sub model has been verified during task loading, and the root model returned success without any tasks
  if (ge_model->GetModelTaskDefPtr() == nullptr) {
    return SUCCESS;
  }
  const auto &tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int32_t i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    GELOGI("Task id = %d, task type = %d", i, task_def.type());
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    uint32_t op_index = std::numeric_limits<uint32_t>::max();
    if (task_type == ModelTaskType::MODEL_TASK_KERNEL) {
      op_index = task_def.kernel().context().op_index();
    } else if (task_type == ModelTaskType::MODEL_TASK_KERNEL_EX) {
      op_index = task_def.kernel_ex().op_index();
    } else if (task_type == ModelTaskType::MODEL_TASK_HCCL) {
      op_index = task_def.kernel_hccl().op_index();
    } else if (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) {
      op_index = task_def.kernel_with_handle().context().op_index();
    } else if (task_type == ModelTaskType::MODEL_TASK_FFTS_PLUS) {
      op_index = task_def.ffts_plus_task().op_index();
    } else {
      GELOGD("Skip task type: %d", static_cast<int32_t>(task_type));
      continue;
    }
    GELOGD("op_index = %u, task_type = %d", op_index, task_type);

    const auto iter = node_map.find(static_cast<int64_t>(op_index));
    if (iter == node_map.end()) {
      GELOGE(INTERNAL_ERROR, "[Find][Node]Failed to get node by op_index = %u", op_index);
      REPORT_INNER_ERR_MSG("E19999", "Failed to get node by op_index = %u.", op_index);
      return INTERNAL_ERROR;
    }

    const auto &node = iter->second;
    LoadTbeKernelBinToOpDesc(task_type, ge_model, node); // for offline
    GELOGD("Task loaded for node: %s, task type = %d, op_index = %u", node->GetName().c_str(), task_type, op_index);
    hybrid_model_.task_defs_[node].emplace_back(task_def);
  }

  return SUCCESS;
}

Status HybridModelBuilder::IndexTaskDefs() {
  const auto &root_graph = hybrid_model_.root_graph_;
  const auto &root_graph_name = root_graph->GetName();
  if (SetOutputNameAttr(*root_graph) != SUCCESS) {
    GELOGW("Set output name attr failed.");
  }

  for (const auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    const auto &name = it.first;
    const auto &ge_model = it.second;
    GE_CHECK_NOTNULL(ge_model);

    auto sub_graph = root_graph->GetSubgraph(name);
    if (name != root_graph_name) {
      if (sub_graph == nullptr) {
        continue;
      }

      const bool is_unknown_shape = sub_graph->GetGraphUnknownFlag();
      if (!is_unknown_shape) {
        GE_CHK_STATUS_RET_NOLOG(LoadGeModel(*sub_graph, ge_model));
        continue;
      }
    } else {
      sub_graph = root_graph;
    }

    GE_CHK_STATUS_RET_NOLOG(IndexTaskDefs(sub_graph, ge_model));
  }

  return SUCCESS;
}

Status HybridModelBuilder::IndexSpecialNodes() {
  GELOGD("Start to index special nodes");
  const auto &root_graph = hybrid_model_.root_graph_;
  for (const auto &node : root_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    const auto op_type = node->GetType();
    GELOGD("node name = %s, node type = %s", node->GetName().c_str(), node->GetType().c_str());
    if (op_type == VARIABLE) {
      auto placement = AttrUtils::GetStr(node->GetOpDesc(), ATTR_VARIABLE_PLACEMENT);
      if ((placement != nullptr) && (*placement == "host")) {
        (void)hybrid_model_.host_variable_nodes_.emplace(node->GetName(), node);
      } else {
        (void)hybrid_model_.device_variable_nodes_.emplace(node->GetName(), node);
      }
    } else if ((op_type == CONSTANTOP) || (op_type == FILECONSTANT)) {
      (void)constant_op_nodes_.emplace(node->GetName(), node);
    } else if (op_type == STREAMMERGE) {
      (void)stream_merge_op_nodes_.emplace(node->GetName(), node);
    } else if ((op_type == NEXTITERATION) || (op_type == REFNEXTITERATION)) {
      (void)next_iteration_op_nodes_.emplace(node->GetName(), node);
    } else if ((op_type == DATA) && (node->GetOwnerComputeGraph() != root_graph)) {
      NodePtr src_node;
      int32_t peer_out_index = -1;
      GE_CHK_STATUS_RET_NOLOG(GetPeerNodeAcrossSubGraphs(node, src_node, peer_out_index));
      GELOGD("Got peer node for data node %s, peer node = %s(%s)",
             node->GetName().c_str(), src_node->GetName().c_str(), src_node->GetType().c_str());

      const bool is_ref_input = (src_node->GetType() == CONSTANTOP) || (src_node->GetType() == VARIABLE);
      if (is_ref_input) {
        for (auto &dst_node_and_in_anchor : node->GetOutDataNodesAndAnchors()) {
          const auto &dst_node = dst_node_and_in_anchor.first;
          const auto &in_anchor = dst_node_and_in_anchor.second;
          node_ref_inputs_[dst_node].emplace_back(std::make_pair(in_anchor->GetIdx(), src_node));
        }
      }
    } else {
      // Nothing for other types.
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::GetPeerNodeAcrossSubGraphs(const NodePtr &data_node,
                                                      NodePtr &peer_node,
                                                      int32_t &peer_out_index) {
  const auto &sub_graph = data_node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(sub_graph);
  GELOGD("To get peer node of %s::%s", sub_graph->GetName().c_str(), data_node->GetName().c_str());
  const auto &wrapped_node = data_node->GetOwnerComputeGraph()->GetParentNodeBarePtr();
  if (wrapped_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "parentnode of node:[%s(%s)] in root graph is nullptr.",
                       data_node->GetName().c_str(), data_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][GetParentNode] for node [%s:%s] in root graph is nullptr.",
           data_node->GetName().c_str(), data_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  const auto &data_op_desc = data_node->GetOpDesc();
  int32_t parent_index = 0;
  if (!AttrUtils::GetInt(data_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    REPORT_INNER_ERR_MSG("E19999", "get attr [%s] on node:[%s(%s)] failed.",
                      ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                      data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] [%s] on node:[%s(%s)] failed",
           ATTR_NAME_PARENT_NODE_INDEX.c_str(), data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto &wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
  GE_CHECK_NOTNULL(wrapped_node_in_anchor);
  const auto &src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  if ((src_out_anchor == nullptr) || (src_out_anchor->GetOwnerNode() == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Parent node of node:%s(%s) do not have peer anchor.",
                       data_node->GetName().c_str(), data_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Parent node of node:%s(%s) do not have peer anchor.",
           data_node->GetName().c_str(), data_node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto &src_wrapped_node_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_wrapped_node_out_anchor);
  const auto &src_wrapped_node = src_wrapped_node_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(src_wrapped_node);

  // connected to root-graph's DATA
  const auto src_node_type = src_wrapped_node->GetType();
  if (src_node_type != PARTITIONEDCALL) {
    peer_node = src_wrapped_node;
    peer_out_index = kHybridVarOutputIndex;
    GELOGD("[%s] Node is connected to root graph's node: %s",
           data_node->GetName().c_str(),
           peer_node->GetName().c_str());
    return SUCCESS;
  }

  const auto &src_graph = NodeUtils::GetSubgraph(*src_wrapped_node, kHybridSubgraphIndex);
  GE_CHECK_NOTNULL(src_graph);
  const auto &src_net_output_node = src_graph->FindFirstNodeMatchType(NETOUTPUT);
  if (src_net_output_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to find NetOutput in subgraph: %s", src_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][FindFirstNodeMatchType]Failed to find NetOutput in subgraph: %s",
           src_graph->GetName().c_str());
    return INTERNAL_ERROR;
  }
  const auto &net_output_desc = src_net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  const auto out_index = static_cast<uint32_t>(src_wrapped_node_out_anchor->GetIdx());
  GELOGD("src graph = %s, src parent output index = %u", src_graph->GetName().c_str(), out_index);

  // link src to outputs of DataNode
  const size_t input_size = net_output_desc->GetAllInputsSize();
  GE_CHECK_LE(input_size, std::numeric_limits<uint32_t>::max());
  for (uint32_t i = 0U; i < static_cast<uint32_t>(input_size); ++i) {
    uint32_t p_index = 0U;
    if (!AttrUtils::GetInt(net_output_desc->GetInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, p_index)) {
      GELOGW("SubGraph: %s input tensor %u attr %s not found.",
             src_graph->GetName().c_str(), i, ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    GELOGD("NetOutput's input[%u], parent_node_index = %u", i, p_index);
    if (p_index == out_index) {
      const auto &in_anchor = src_net_output_node->GetInDataAnchor(static_cast<int32_t>(i));
      GE_CHECK_NOTNULL(in_anchor);
      const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      peer_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      peer_out_index = peer_out_anchor->GetIdx();
      GELOGD("Found peer node of Data node: %s::%s is %s::%s",
             sub_graph->GetName().c_str(),
             data_node->GetName().c_str(),
             src_graph->GetName().c_str(),
             peer_node->GetName().c_str());
      return SUCCESS;
    }
  }

  GELOGE(FAILED, "[Get][PeerNode]Failed to find peer node for %s::%s(%s)", sub_graph->GetName().c_str(),
         data_node->GetName().c_str(), data_node->GetType().c_str());
  REPORT_INNER_ERR_MSG("E19999", "Failed to find peer node for %s::%s(%s).",
                     sub_graph->GetName().c_str(), data_node->GetName().c_str(), data_node->GetType().c_str());
  return FAILED;
}
Status HybridModelBuilder::InitRuntimeParams() {
  int64_t value = 0;
  bool ret = false;
  if (ge_root_model_->GetSubgraphInstanceNameToModel().empty()) {
    GELOGE(INTERNAL_ERROR, "[Get][SubModel]Root model has no sub model, model_name[%s]", ModelName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Root model has no sub model, model_name[%s]", ModelName().c_str());
    return INTERNAL_ERROR;
  }

  // session id and var size is same for every model
  const auto &first_model = ge_root_model_->GetSubgraphInstanceNameToModel().begin()->second;
  GE_CHECK_NOTNULL(first_model);
  ret = AttrUtils::GetInt(first_model, MODEL_ATTR_SESSION_ID, value);
  runtime_param_.session_id = ret ? static_cast<uint64_t>(value) : 0U;
  ret = AttrUtils::GetInt(first_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, value);
  runtime_param_.logic_var_base = ret ? static_cast<uint64_t>(value) : 0U;
  runtime_param_.graph_id = hybrid_model_.root_graph_->GetGraphID();
  value = 0;
  for (const auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    (void)AttrUtils::GetInt(it.second, ATTR_MODEL_VAR_SIZE, value);
    if (value > 0) {
      runtime_param_.var_size = static_cast<uint64_t>(value);
      break;
    }
  }

  GELOGI("InitRuntimeParams(), session_id:%lu, var_size:%lu. graph_id = %u",
         runtime_param_.session_id, runtime_param_.var_size, runtime_param_.graph_id);

  var_manager_ = VarManager::Instance(runtime_param_.session_id);
  GE_CHECK_NOTNULL(var_manager_);
  if (!var_manager_->HasMemoryManager()) {
    var_manager_->SetMemManager(&MemManager::Instance());
  }
  return SUCCESS;
}

Status HybridModelBuilder::IdentifyVariableOutputs(NodeItem &node_item, const ComputeGraphPtr &subgraph) {
  GELOGD("Start to parse outputs of node: %s", node_item.NodeName().c_str());
  const auto &net_output_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("[%s] Subgraph do not got net output", subgraph->GetName().c_str());
    return SUCCESS;
  }
  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  // constants connected to net output
  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    const auto &src_node = GetPeerNode(in_data_anchor);
    GE_CHECK_NOTNULL(src_node);
    const auto src_op_type = src_node->GetType();
    if ((src_op_type == CONSTANTOP) || (src_op_type == CONSTANT)) {
      (void)known_subgraph_constant_output_refs_[&node_item].emplace(in_data_anchor->GetIdx(), src_node);
    }
  }

  // Data nodes marked with REF_VAR_SRC_VAR_NAME
  // Using variable tensor as data's output
  for (const auto &node : subgraph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetType() != DATA) {
      continue;
    }

    std::string ref_var_name;
    const std::string* ref_var_name_str =  AttrUtils::GetStr(node->GetOpDescBarePtr(), REF_VAR_SRC_VAR_NAME);
    if (ref_var_name_str != nullptr) {
      ref_var_name = *ref_var_name_str;
    }
    if (ref_var_name.empty()) {
      continue;
    }
    GELOGD("Data node ref to variable: %s", ref_var_name.c_str());
    NodePtr src_node;
    auto var_node = hybrid_model_.GetVariableNode(ref_var_name);
    GE_CHECK_NOTNULL(var_node);
    GELOGD("Found var node [%s] by ref_var_name [%s]", var_node->GetName().c_str(), ref_var_name.c_str());
    int32_t peer_output_index = -1;
    GE_CHK_STATUS_RET_NOLOG(GetPeerNodeAcrossSubGraphs(node, src_node, peer_output_index));
    auto *const src_node_item = MutableNodeItem(src_node);
    GE_CHECK_NOTNULL(src_node_item);
    (void)src_node_item->ref_outputs.emplace(peer_output_index, var_node);
  }

  return SUCCESS;
}

NodePtr HybridModelBuilder::GetPeerNode(const InDataAnchorPtr &in_data_anchor) {
  const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    return peer_out_anchor->GetOwnerNode();
  }

  return nullptr;
}

Status HybridModelBuilder::InitModelMem() {
  auto *const mem_allocator = NpuMemoryAllocator::GetAllocator(hybrid_model_.device_id_, nullptr);
  GE_CHECK_NOTNULL(mem_allocator);
  hybrid_model_.global_step_ = TensorBuffer::Create(mem_allocator, sizeof(int64_t));
  GE_CHECK_NOTNULL(hybrid_model_.global_step_);

  GELOGI("[Init] Variable max size attr do not set, var mem auto malloc.");
  runtime_param_.var_base = 0;
  const auto page_size = VarManager::IsVariableUse1gHugePage() ? kDrv1GPageSize : kDrvPageSize;
  auto allocator =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(runtime_param_.session_id,
                                                                                       hybrid_model_.device_id_,
                                                                                       RT_MEMORY_HBM, page_size);
  (void) var_manager_->InitExpandableMemoryAllocator(allocator);
  return SUCCESS;
}

Status HybridModelBuilder::TransAllVarData() const {
  GELOGI("TransAllVarData start: session_id:%lu, graph_id: %u.", runtime_param_.session_id, runtime_param_.graph_id);
  std::vector<NodePtr> variable_node_list;
  for (const auto &it : hybrid_model_.device_variable_nodes_) {
    variable_node_list.emplace_back(it.second);
    GELOGD("[%s] added for trans var data", it.first.c_str());
  }

  GE_CHK_STATUS_RET(TransVarDataUtils::TransAllVarData(variable_node_list, runtime_param_.session_id,
                                                       runtime_param_.graph_id, hybrid_model_.device_id_),
                    "[Invoke][TransAllVarData] failed.");

  GELOGI("TransAllVarData success.");
  return SUCCESS;
}

Status HybridModelBuilder::CopyVarData() const {
  std::vector<NodePtr> variable_node_list;
  for (const auto &it : hybrid_model_.device_variable_nodes_) {
    GELOGD("[%s] added for trans var data", it.first.c_str());
    variable_node_list.emplace_back(it.second);
  }

  GE_CHK_STATUS_RET(TransVarDataUtils::CopyVarData(hybrid_model_.root_graph_, variable_node_list,
                                                   runtime_param_.session_id,
                                                   hybrid_model_.device_id_),
                    "[Invoke][CopyVarData] failed.");
  GELOGI("CopyVarData success.");
  return SUCCESS;
}

Status HybridModelBuilder::LoadKnownShapedSubgraph(const ComputeGraph &graph, const NodeItem &parent_node_item) {
  GELOGD("Start to load known shaped subgraph [%s]", graph.GetName().c_str());
  auto graph_item = MakeUnique<GraphItem>();
  GE_CHECK_NOTNULL(graph_item);
  graph_item->is_dynamic_ = false;
  const auto subgraph_name = graph.GetName();
  const auto wrapper_op_desc = MakeShared<OpDesc>(subgraph_name + "_partitioned_call", PARTITIONEDCALL);
  GE_CHECK_NOTNULL(wrapper_op_desc);

  for (const auto &node : graph.GetDirectNode()) {
    GE_CHK_STATUS_RET_NOLOG(LoadKnownNodeItem(*graph_item, node, wrapper_op_desc));
  }

  const auto temp_graph = MakeShared<ComputeGraph>("temp");
  GE_CHECK_NOTNULL(temp_graph);
  const auto wrapper_node = temp_graph->AddNode(wrapper_op_desc);
  GE_CHECK_NOTNULL(wrapper_node);
  wrapper_op_desc->SetId(parent_node_item.node_id);
  const GeModelPtr ge_model = subgraph_models_[subgraph_name];
  GE_CHECK_NOTNULL(ge_model);
  (void)hybrid_model_.known_shape_sub_models_.emplace(wrapper_node, ge_model);

  NodeItem *node_item = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(wrapper_node, node_item));
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->outputs.resize(static_cast<size_t>(node_item->num_outputs));
  graph_item->node_items_.emplace_back(node_item);
  graph_item->output_node_ = node_item;
  graph_item->total_inputs_ = node_item->num_inputs;
  graph_item->total_outputs_ = node_item->num_outputs;

  GELOGD("NodeItem create for known shape subgraph [%s], NodeItem = %s",
         graph.GetName().c_str(),
         node_item->DebugString().c_str());

  graph_item->SetName(graph.GetName());
  GELOGD("Done loading known shape subgraph: [%s]", graph_item->GetName().c_str());
  (void)hybrid_model_.subgraph_items_.emplace(graph.GetName(), std::move(graph_item));
  return SUCCESS;
}

Status HybridModelBuilder::LoadKnownNodeItem(GraphItem &graph_item, const NodePtr &node,
                                             const OpDescPtr &wrapper_op_desc) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const auto &op_desc = node->GetOpDesc();
  const auto &op_type = node->GetType();

  if (op_type == DATA) {
    int32_t data_index = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
      GELOGE(FAILED, "[Invoke][GetInt] [%s(%s)] Failed to get attr [%s]",
             node->GetName().c_str(), node->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return FAILED;
    }

    (void)wrapper_op_desc->AddInputDesc(op_desc->GetInputDesc(0U));
    graph_item.input_index_mapping_.emplace_back(data_index);
  }

  if (op_type == NETOUTPUT) {
    int32_t output_index = 0;
    for (const auto &output_desc : op_desc->GetAllInputsDescPtr()) {
      int32_t data_index = output_index++;
      if (!AttrUtils::GetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
        GELOGW("[%s] Failed to get attr [%s]", node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      }

      GE_CHK_GRAPH_STATUS_RET(wrapper_op_desc->AddOutputDesc(*output_desc),
                              "[Invoke][AddOutputDesc][%s] Failed to add output desc. output index = %d",
                              node->GetName().c_str(), output_index);
      graph_item.output_index_mapping_.emplace_back(data_index);
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::RecoverGraphUnknownFlag() const {
  const bool is_host_exec = GetContext().GetHostExecFlag();
  auto graphs = hybrid_model_.root_graph_->GetAllSubgraphs();
  graphs.push_back(hybrid_model_.root_graph_);
  for (auto &graph : graphs) {
    GE_CHECK_NOTNULL(graph);
    if (is_host_exec) {
      graph->SetGraphUnknownFlag(true);
    } else {
      for (const auto &node : graph->GetDirectNode()) {
        bool is_unknown_shape = false;
        (void)AttrUtils::GetBool(node->GetOpDesc(), kHybridOwnerGraphIsUnknown, is_unknown_shape);
        graph->SetGraphUnknownFlag(is_unknown_shape);
        break;
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::GenerateFpProfilingTask(const OpDescPtr &op_desc,
                                                   std::vector<domi::TaskDef> &task_def_list) {
  GELOGD("The first FP operator is %s", op_desc->GetName().c_str());

  domi::TaskDef fp_task_def;
  fp_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  fp_task_def.set_stream_id(static_cast<uint32_t>(op_desc->GetStreamId()));
  domi::LogTimeStampDef *const fp_log_def = fp_task_def.mutable_log_timestamp();
  if (fp_log_def != nullptr) {
    fp_log_def->set_logid(kHybridProfilingFpStartLogId);
    fp_log_def->set_notify(false);
  }
  task_def_list.emplace_back(fp_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::GenerateArProfilingTask(const OpDescPtr &op_desc, const int64_t profiling_log_id,
                                                   std::vector<domi::TaskDef> &task_def_list) {
  domi::TaskDef ar_task_def;
  ar_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  ar_task_def.set_stream_id(static_cast<uint32_t>(op_desc->GetStreamId()));
  domi::LogTimeStampDef *const ar_log_def = ar_task_def.mutable_log_timestamp();
  if (ar_log_def != nullptr) {
    ar_log_def->set_logid(static_cast<uint64_t>(profiling_log_id));
    ar_log_def->set_notify(false);
  }
  task_def_list.emplace_back(ar_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::GenerateBpProfilingTask(const OpDescPtr &op_desc,
                                                   std::vector<domi::TaskDef> &task_def_list) {
  domi::TaskDef bp_task_def;
  bp_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  bp_task_def.set_stream_id(static_cast<uint32_t>(op_desc->GetStreamId()));
  domi::LogTimeStampDef *const bp_log_def = bp_task_def.mutable_log_timestamp();
  GE_CHECK_NOTNULL(bp_log_def);
  bp_log_def->set_logid(kHybridProfilingBpEndLogId);
  bp_log_def->set_notify(false);
  task_def_list.emplace_back(bp_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::GenerateEndProfilingTask(const OpDescPtr &op_desc,
                                                    std::vector<domi::TaskDef> &task_def_list) {
  domi::TaskDef end_task_def;
  end_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  end_task_def.set_stream_id(static_cast<uint32_t>(op_desc->GetStreamId()));
  domi::LogTimeStampDef *const end_log_def = end_task_def.mutable_log_timestamp();
  GE_CHECK_NOTNULL(end_log_def);
  end_log_def->set_logid(kHybridProfilingIterEndLogId);
  end_log_def->set_notify(true);
  task_def_list.emplace_back(end_task_def);

  return SUCCESS;
}

Status HybridModelBuilder::CreateProfilingNodeBefore(GraphItem &graph_item, const NodePtr &node,
                                                     uint32_t &prev_num) const {
  GE_CHECK_NOTNULL(node);
  const OpDescPtr &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto &compute_graph = MakeShared<ComputeGraph>(kHybridProfilingGraph);
  GE_CHECK_NOTNULL(compute_graph);

  NodePtr node_ptr = nullptr;
  std::map<NodePtr, std::vector<domi::TaskDef>> node_task_map;
  // create fp node
  bool is_insert_fp_profiling_task = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_FP_PROFILILNG_TASK, is_insert_fp_profiling_task);
  if (is_insert_fp_profiling_task) {
    std::vector<domi::TaskDef> task_def_list;
    (void)GenerateFpProfilingTask(op_desc, task_def_list);
    const auto fp_desc = MakeShared<OpDesc>(kHybridProfilingFpNode, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(fp_desc);
    fp_desc->SetOpKernelLibName(kHybridEngineNameRts);
    node_ptr = compute_graph->AddNode(fp_desc);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create fp profiling node success before.");
  }
  // creat all reduce start node
  bool is_insert_bp_profiling_task = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  const bool is_all_reduce = ((op_desc->GetType() == HCOMALLREDUCE) || (op_desc->GetType() == HVDCALLBACKALLREDUCE));
  if (is_all_reduce && is_insert_bp_profiling_task) {
    std::vector<domi::TaskDef> task_def_list;
    int64_t profiling_log_id = 0;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, profiling_log_id);
    GELOGD("All reduce node profiling task log id: %ld before", profiling_log_id);
    (void)GenerateArProfilingTask(op_desc, profiling_log_id, task_def_list);
    const std::string op_name = std::string(kHybridProfilingArNode) + std::to_string(profiling_log_id);
    const auto ar_desc_start = MakeShared<OpDesc>(op_name, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(ar_desc_start);
    ar_desc_start->SetOpKernelLibName(kHybridEngineNameRts);
    node_ptr = compute_graph->AddNode(ar_desc_start);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create all reduce start profiling node success before.");
  }

  if (!node_task_map.empty()) {
    for (const auto &node_task : node_task_map) {
      const NodePtr &profiling_node = node_task.first;
      const std::vector<domi::TaskDef> &task_def_lists = node_task.second;
      for (const auto &task_def : task_def_lists) {
        hybrid_model_.task_defs_[profiling_node].emplace_back(task_def);
      }
      if (op_desc->HasAttr(ATTR_STAGE_LEVEL)) {
        int64_t stage_level = std::numeric_limits<int64_t>::max();
        (void)AttrUtils::GetInt(op_desc, ATTR_STAGE_LEVEL, stage_level);
        GE_CHECK_NOTNULL(node_ptr);
        (void)AttrUtils::SetInt(node_ptr->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level);
      }
      NodeItem *node_item = nullptr;
      GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(profiling_node, node_item));
      GE_CHECK_NOTNULL(node_item);
      node_item->input_start = 0;
      node_item->output_start = 0;
      graph_item.node_items_.emplace_back(node_item);
      ++prev_num;
    }
  } else {
    GELOGD("No need to create profiling node before.");
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateProfilingNodeAfter(GraphItem &graph_item, const NodePtr &node,
                                                    uint32_t &post_num) const {
  GE_CHECK_NOTNULL(node);
  const OpDescPtr &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto &compute_graph = MakeShared<ComputeGraph>(kHybridProfilingGraph);
  GE_CHECK_NOTNULL(compute_graph);

  NodePtr node_ptr = nullptr;
  std::map<NodePtr, std::vector<domi::TaskDef>> node_task_map;
  // Create all reduce end node
  bool is_insert_bp_profiling_task = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_insert_bp_profiling_task);
  const bool is_all_reduce = ((op_desc->GetType() == HCOMALLREDUCE) || (op_desc->GetType() == HVDCALLBACKALLREDUCE));
  if (is_all_reduce && is_insert_bp_profiling_task) {
    std::vector<domi::TaskDef> task_def_list;
    int64_t profiling_log_id = 0;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, profiling_log_id);
    GELOGD("All reduce node profiling task log id: %ld after", profiling_log_id);
    (void)GenerateArProfilingTask(op_desc, profiling_log_id + 1, task_def_list);
    const std::string op_name = std::string(kHybridProfilingArNode) + std::to_string(profiling_log_id + 1);
    const auto ar_desc_end = MakeShared<OpDesc>(op_name, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(ar_desc_end);
    ar_desc_end->SetOpKernelLibName(kHybridEngineNameRts);
    node_ptr = compute_graph->AddNode(ar_desc_end);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create all reduce end profiling node success after.");
  }
  // create bp node
  if ((!is_all_reduce) && is_insert_bp_profiling_task) {
    std::vector<domi::TaskDef> task_def_list;
    (void)GenerateBpProfilingTask(op_desc, task_def_list);
    const auto bp_op_desc = MakeShared<OpDesc>(kHybridProfilingBpNode, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(bp_op_desc);
    bp_op_desc->SetOpKernelLibName(kHybridEngineNameRts);
    node_ptr = compute_graph->AddNode(bp_op_desc);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create bp profiling node success after.");
  }
  // create end node
  bool is_insert_end_profiling_task = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_INSERT_END_PROFILILNG_TASK, is_insert_end_profiling_task);
  if (is_insert_end_profiling_task) {
    std::vector<domi::TaskDef> task_def_list;
    (void)GenerateEndProfilingTask(op_desc, task_def_list);
    const auto end_desc = MakeShared<OpDesc>(kHybridProfilingEndNode, PROFILINGTRAININGTRACE);
    GE_CHECK_NOTNULL(end_desc);
    end_desc->SetOpKernelLibName(kHybridEngineNameRts);
    node_ptr = compute_graph->AddNode(end_desc);
    GE_CHECK_NOTNULL(node_ptr);
    node_task_map[node_ptr] = task_def_list;
    GELOGD("Create end profiling node success after.");
  }

  if (!node_task_map.empty()) {
    for (const auto &node_task : node_task_map) {
      const NodePtr &profiling_node = node_task.first;
      const std::vector<domi::TaskDef> &task_def_lists = node_task.second;
      for (const auto &task_def : task_def_lists) {
        hybrid_model_.task_defs_[profiling_node].emplace_back(task_def);
      }
      if (op_desc->HasAttr(ATTR_STAGE_LEVEL)) {
        int64_t stage_level = std::numeric_limits<int64_t>::max();
        (void)AttrUtils::GetInt(op_desc, ATTR_STAGE_LEVEL, stage_level);
        (void)AttrUtils::SetInt(profiling_node->GetOpDesc(), ATTR_STAGE_LEVEL, stage_level);
      }
      NodeItem *node_item = nullptr;
      GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(profiling_node, node_item));
      GE_CHECK_NOTNULL(node_item);
      node_item->input_start = 0;
      node_item->output_start = 0;
      graph_item.node_items_.emplace_back(node_item);
      ++post_num;
    }
  } else {
    GELOGD("No need to create profiling node after.");
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadDynamicSubgraph(const ComputeGraphPtr &graph, const bool is_root_graph) {
  GELOGD("Start to load subgraph [%s]", graph->GetName().c_str());
  // for known partitioned call, load all nodes
  auto graph_item = MakeUnique<GraphItem>();
  GE_CHECK_NOTNULL(graph_item);

  graph_item->is_dynamic_ = true;
  graph_item->is_root_graph_ = is_root_graph;
  graph_item->node_items_.reserve(graph->GetDirectNodesSize());

  const auto &functional_node = graph->GetParentNodeBarePtr();
  if ((functional_node != nullptr) && IsFftsGraphNode(*functional_node->GetOpDescBarePtr())) {
    graph_item->is_ffts_graph_ = true;
  }

  std::vector<NodeItem *> input_nodes;
  std::map<size_t, std::pair<uint32_t, uint32_t>> profiling_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET_NOLOG(LoadDynamicNodeItem(*graph_item, node, input_nodes, profiling_nodes));
  }

  GE_CHK_STATUS_RET_NOLOG(BuildInputMapping(*graph_item, input_nodes, is_root_graph));
  GE_CHK_STATUS_RET_NOLOG(BuildProfilingControl(*graph_item, profiling_nodes));
  if (is_root_graph) {
    graph_item->SetName("Root-Graph");
    GELOGD("Done loading dynamic subgraph: [%s]", graph_item->GetName().c_str());
    hybrid_model_.root_graph_item_ = std::move(graph_item);
  } else {
    graph_item->SetName(graph->GetName());
    GELOGD("Done loading dynamic subgraph: [%s]", graph_item->GetName().c_str());
    (void)hybrid_model_.subgraph_items_.emplace(graph->GetName(), std::move(graph_item));
  }

  return SUCCESS;
}

Status HybridModelBuilder::LoadDynamicNodeItem(GraphItem &graph_item, const NodePtr &node,
                                               std::vector<NodeItem *> &input_nodes,
                                               std::map<size_t, std::pair<uint32_t, uint32_t>> &profiling_nodes) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const auto &op_type = node->GetType();

  NodeItem *node_item = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node, node_item));
  GE_CHK_STATUS_RET_NOLOG(BuildNodeItem(node, *node_item));
  GE_CHK_STATUS_RET_NOLOG(UpdateAnchorStatus(node)); // needed by FE generate task

  GE_CHK_STATUS_RET_NOLOG(BuildFrameGroupIndex(*node_item));
  GE_CHK_STATUS_RET_NOLOG(BuildControlFlowGroup(graph_item, node, *node_item));
  if (node->GetInAllNodes().empty()) {
    graph_item.root_items_.emplace_back(node_item);
    GELOGD("[%s] add to root node list", node->GetName().c_str());
  }

  node_item->input_start = graph_item.total_inputs_;
  node_item->output_start = graph_item.total_outputs_;
  graph_item.total_inputs_ += node_item->num_inputs;
  graph_item.total_outputs_ += node_item->num_outputs;

  if ((op_type == DATA_TYPE) || (op_type == AIPP_DATA_TYPE)) {
    input_nodes.emplace_back(node_item);
  }
  if (op_type == NETOUTPUT) {
    graph_item.output_node_ = node_item;
    GE_CHK_STATUS_RET_NOLOG(BuildOutputMapping(graph_item, *node_item, graph_item.is_root_graph_));
  }

  uint32_t prev_num = 0U;
  uint32_t post_num = 0U;
  GE_CHK_STATUS_RET_NOLOG(CreateProfilingNodeBefore(graph_item, node, prev_num));
  const size_t node_index = graph_item.node_items_.size();
  graph_item.node_items_.emplace_back(node_item);
  GE_CHK_STATUS_RET_NOLOG(CreateProfilingNodeAfter(graph_item, node, post_num));
  if ((prev_num > 0U) || (post_num > 0U)) {
    profiling_nodes[node_index] = { prev_num, post_num };
  }
  // parse var outputs
  GE_CHK_STATUS_RET_NOLOG(ParseVarOutputs(*node_item));
  GELOGD("NodeItem created: %s", node_item->DebugString().c_str());

  // load fused graph as graph_item on hybrid model when fuzz compile mode
  if ((node_item->fused_subgraph != nullptr) &&
      (hybrid_model_.node_bin_mode_ == fuzz_compile::kOneNodeMultipleBinsMode)) {
    const auto &origin_fused_graph = node_item->fused_subgraph->graph;
    GE_CHK_STATUS_RET(LoadDynamicSubgraph(origin_fused_graph, false),
                      "[Invoke][LoadDynamicSubgraph]Failed to load origin fused_subgraph: [%s] of node: [%s]",
                      origin_fused_graph->GetName().c_str(), node->GetName().c_str());
    GELOGI("Load origin fused graph [%s] of node [%s] successfully.", origin_fused_graph->GetName().c_str(),
           node->GetName().c_str());
  }
  return SUCCESS;
}

Status HybridModelBuilder::ParseVarOutputs(NodeItem &node_item) const {
  for (int32_t i = 0; i < node_item.num_outputs; ++i) {
    const auto &output_tensor_desc = node_item.op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    auto var_name = AttrUtils::GetStr(output_tensor_desc, ASSIGN_VAR_NAME);
    if (var_name == nullptr) {
        var_name = AttrUtils::GetStr(output_tensor_desc, REF_VAR_SRC_VAR_NAME);
    }
    if (var_name != nullptr) {
      const auto var_node = hybrid_model_.GetVariableNode(*var_name);
      GE_CHECK_NOTNULL(var_node);
      node_item.ref_outputs[i] = var_node;
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::BuildInputMapping(GraphItem &graph_item,
                                             const std::vector<NodeItem *> &input_nodes,
                                             const bool is_root_graph) {
  uint32_t data_op_index = 0U;
  for (const auto &node_item : input_nodes) {
    const auto &node = node_item->node;
    uint32_t data_index = data_op_index;
    if (is_root_graph) {
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, data_index)) {
        GELOGI("ge_train: get new index %u, old %u", data_index, data_op_index);
      }
      data_op_index++;
    } else {
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, data_index)) {
        GELOGE(FAILED, "[Invoke][GetInt] [%s(%s)] Failed to get attr [%s]",
               node->GetName().c_str(), node->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        REPORT_INNER_ERR_MSG("E19999", "call GetInt failed, [%s(%s)] Failed to get attr [%s]",
                          node->GetName().c_str(), node->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        return FAILED;
      }
    }

    if (graph_item.input_nodes_.size() <= static_cast<size_t>(data_index)) {
      graph_item.input_nodes_.resize(static_cast<size_t>(data_index + 1U));
    }

    graph_item.input_nodes_[static_cast<size_t>(data_index)] = node_item;
  }

  return SUCCESS;
}

Status HybridModelBuilder::CheckAicpuOpList() const {
  std::vector<std::string> aicpu_optype_list;
  std::vector<std::string> aicpu_tf_optype_list;
  std::set<std::string> aicpu_optype_set;
  std::set<std::string> aicpu_tf_optype_set;
  for (const auto &it : ge_root_model_->GetSubgraphInstanceNameToModel()) {
    const auto &ge_model = it.second;
    GE_CHECK_NOTNULL(ge_model);
    if (AttrUtils::GetListStr(*ge_model, "needCheckCpu", aicpu_optype_list)) {
      aicpu_optype_set.insert(aicpu_optype_list.cbegin(), aicpu_optype_list.cend());
    }

    if (AttrUtils::GetListStr(*ge_model, "needCheckTf", aicpu_tf_optype_list)) {
      aicpu_tf_optype_set.insert(aicpu_tf_optype_list.cbegin(), aicpu_tf_optype_list.cend());
    }
  }
  // reset list with set
  aicpu_optype_list.assign(aicpu_optype_set.begin(), aicpu_optype_set.end());
  aicpu_tf_optype_list.assign(aicpu_tf_optype_set.begin(), aicpu_tf_optype_set.end());
  GE_CHK_STATUS_RET(ModelManager::GetInstance().LaunchKernelCheckAicpuOp(aicpu_optype_list, aicpu_tf_optype_list),
                    "[Launch][KernelCheckAicpuOp] failed.");
  return SUCCESS;
}

Status HybridModelBuilder::CollectParallelGroups(NodeItem &node_item) {
  const auto &node = node_item.node;
  const auto executor_type = NodeExecutorManager::GetInstance().ResolveExecutorType(node_item);
  if (executor_type == NodeExecutorManager::ExecutorType::HCCL) {
    auto parallel_group = AttrUtils::GetStr(node->GetOpDescBarePtr(), ATTR_NAME_PARALLEL_GROUP);
    if (parallel_group != nullptr) {
      GELOGD("[%s] Got parallel group = [%s]", node_item.NodeName().c_str(), (*parallel_group).c_str());
      (void)parallel_group_to_nodes_[*parallel_group].emplace(&node_item);
      const std::set<std::string> group{*parallel_group};
      (void)node_to_parallel_groups_[&node_item].emplace(*parallel_group);
    }
  }

  if (executor_type == NodeExecutorManager::ExecutorType::COMPILED_SUBGRAPH) {
    std::set<std::string> parallel_groups;
    GELOGD("[%s] To collect parallel group for known-shaped subgraph", node_item.NodeName().c_str());
    for (const auto &subgraph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
      GELOGD("[%s] Start to get parallel group from subgraph: %s",
             node_item.NodeName().c_str(), subgraph_name.c_str());
      const auto &subgraph = hybrid_model_.root_graph_->GetSubgraph(subgraph_name);
      GE_CHECK_NOTNULL(subgraph);
      for (const auto &sub_node : subgraph->GetAllNodes()) {
        std::string parallel_group;
        if (AttrUtils::GetStr(sub_node->GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, parallel_group)) {
          GELOGD("[%s:%s] Got parallel group = %s",
                 subgraph_name.c_str(), sub_node->GetName().c_str(), parallel_group.c_str());
          (void)parallel_groups.emplace(parallel_group);
        }
      }
    }

    if (!parallel_groups.empty()) {
      for (const auto &parallel_group : parallel_groups) {
        (void)parallel_group_to_nodes_[parallel_group].emplace(&node_item);
        GELOGD("[%s] has parallel group: %s", node_item.NodeName().c_str(), parallel_group.c_str());
      }
      (void)node_to_parallel_groups_.emplace(&node_item, std::move(parallel_groups));
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::ParseDependentByParallelGroup() {
  for (const auto &it : hybrid_model_.node_items_) {
    GE_CHK_STATUS_RET_NOLOG(CollectParallelGroups(*it.second));
  }
  for (const auto &it : node_to_parallel_groups_) {
    auto *const node_item = it.first;
    const auto dst_executor_type = NodeExecutorManager::GetInstance().ResolveExecutorType(*node_item);
    for (const auto &parallel_group : it.second) {
      NodeItem *nearest_dep_node = nullptr;
      int64_t max_id = -1;
      for (auto &dep_node : parallel_group_to_nodes_[parallel_group]) {
        if ((dep_node->node_id < node_item->node_id) && (dep_node->node_id > max_id)) {
          nearest_dep_node = dep_node;
          max_id = dep_node->node_id;
        }
      }

      if (nearest_dep_node != nullptr) {
        GELOGD("[%s] Nearest node = [%s]", node_item->NodeName().c_str(), nearest_dep_node->NodeName().c_str());
        const auto src_engine_type = NodeExecutorManager::GetInstance().ResolveExecutorType(*nearest_dep_node);
        if (src_engine_type == dst_executor_type) {
          GELOGD("No need to add dependency for nodes with same executor type");
          continue;
        }
        auto &deps = node_item->dependents_for_execution;
        if (std::find(deps.begin(), deps.end(), nearest_dep_node->node) != deps.end()) {
          GELOGD("%s->%s Already has dependency, skip it",
                 nearest_dep_node->node->GetName().c_str(),
                 node_item->NodeName().c_str());
          continue;
        }
        nearest_dep_node->has_observer = true;
        deps.emplace_back(nearest_dep_node->node);
        GELOGD("Add dependency for nodes with the same parallel group[%s], src = [%s], dst = [%s]",
               parallel_group.c_str(),
               nearest_dep_node->NodeName().c_str(),
               node_item->NodeName().c_str());
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::OptimizeDependenciesForConstantInputs() {
  std::map<NodePtr, std::set<uint32_t>> converted;
  for (const auto &it : host_input_value_dependencies_) {
    auto *const node_item = it.first;
    std::map<NodeItem *, int32_t> ref_counts;
    bool changed = false;
    for (const auto &output_idx_and_node : it.second) {
      const auto output_idx = output_idx_and_node.first;
      auto *const src_node_item = output_idx_and_node.second;
      ++ref_counts[src_node_item];
      NodePtr constant_node;
      if ((src_node_item->node_type == CONSTANT) || (src_node_item->node_type == CONSTANTOP)) {
        constant_node = src_node_item->node;
        GELOGD("src node [%s] is a constant", src_node_item->NodeName().c_str());
      } else {
        const auto iter = known_subgraph_constant_output_refs_.find(src_node_item);
        if (iter != known_subgraph_constant_output_refs_.end()) {
          constant_node = iter->second[output_idx];
          if (constant_node != nullptr) {
            GELOGD("Output[%u] of subgraph [%s] is a constant", output_idx, src_node_item->NodeName().c_str());
          }
        }
      }
      if (constant_node == nullptr) {
        GELOGD("Output[%u] of [%s] is not a constant", output_idx, src_node_item->NodeName().c_str());
        continue;
      }
      if (converted[constant_node].count(output_idx) == 0U) {
        GE_CHK_STATUS_RET(Convert2HostTensor(constant_node, src_node_item->node_id, output_idx),
                          "[Convert][HostTensor] [%s(%s)] Failed to convert constant to host tensor",
                          constant_node->GetName().c_str(), constant_node->GetType().c_str());
        (void)converted[constant_node].emplace(output_idx);
      }
      (void)src_node_item->to_const_output_id_list.erase(static_cast<int32_t>(output_idx));
      --ref_counts[src_node_item];
      changed = true;
    }
    if (changed) {
      std::vector<NodePtr> depends_to_keep;
      for (const auto &ref_count_it : ref_counts) {
        if (ref_count_it.second == 0) {
          GELOGD("[%s] no longer depends on [%s] for shape inference",
                 node_item->NodeName().c_str(),
                 ref_count_it.first->NodeName().c_str());
        } else {
          depends_to_keep.emplace_back(ref_count_it.first->node);
        }
      }
      node_item->dependents_for_shape_inference.swap(depends_to_keep);
    }
  }

  return SUCCESS;
}
Status HybridModelBuilder::Convert2HostTensor(const NodePtr &node, const int64_t node_id, const uint32_t output_idx) {
  const auto tensor_value = hybrid_model_.GetTensor(node);
  GE_CHECK_NOTNULL(tensor_value);
  const auto &tensor_desc = node->GetOpDesc()->MutableOutputDesc(0U);
  GE_CHECK_NOTNULL(tensor_desc);
  GeTensorPtr ge_tensor = MakeShared<GeTensor>(*tensor_desc);
  GE_CHECK_NOTNULL(ge_tensor);
  int64_t tensor_size = -1;
  GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size),
                          "[Get][TensorSize] In Bytes for [%s(%s)] failed",
                          node->GetName().c_str(), node->GetType().c_str());
  if (tensor_size > 0) {
    const auto copy_size = static_cast<size_t>(tensor_size);
    GE_CHECK_GE(tensor_value->GetSize(), copy_size);
    std::vector<uint8_t> buffer(copy_size);
    GE_CHK_RT_RET(rtMemcpy(buffer.data(),
                           copy_size,
                           tensor_value->GetData(),
                           copy_size,
                           RT_MEMCPY_DEVICE_TO_HOST));
    (void)ge_tensor->SetData(std::move(buffer));
    GELOGD("[%s] Copy constant tensor to host successfully, size = %zu", node->GetName().c_str(), copy_size);
  }

  hybrid_model_.host_tensors_[node_id].emplace_back(output_idx, std::move(ge_tensor));
  return SUCCESS;
}

Status HybridModelBuilder::RelinkNextIteration() const {
  for (const auto &item : stream_merge_op_nodes_) {
    const auto &merge = item.second;
    auto node_name = AttrUtils::GetStr(merge->GetOpDesc(), ATTR_NAME_NEXT_ITERATION);
    if (node_name == nullptr) {
      GELOGD("[%s] no attribute[%s], not in while loop", merge->GetName().c_str(), ATTR_NAME_NEXT_ITERATION.c_str());
      continue;
    }

    const auto it = next_iteration_op_nodes_.find(*node_name);
    if (it == next_iteration_op_nodes_.end()) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] [%s(%s)] expect NextIteration[%s] not found",
             merge->GetName().c_str(), merge->GetType().c_str(), (*node_name).c_str());
      return INTERNAL_ERROR;
    }

    const auto &iteration = it->second;
    if (GraphUtils::AddEdge(iteration->GetOutDataAnchor(0), merge->GetInDataAnchor(1)) != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[%s(%s)] -> [%s(%s)] Add edge failed",
             (*node_name).c_str(), iteration->GetType().c_str(), merge->GetName().c_str(), merge->GetType().c_str());
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::BuildProfilingControl(GraphItem &graph_item,
                                                 const std::map<size_t, std::pair<uint32_t, uint32_t>> &nodes) const {
  const auto node_size = graph_item.node_items_.size();
  for (const auto &item : nodes) {
    const auto node_index = item.first;
    GE_CHK_BOOL_RET_STATUS(node_index < node_size, FAILED,
                           "[Check][Param] node index:%zu invalid, node size:%zu", node_index, node_size);
    const auto &node_item = graph_item.node_items_[node_index];
    if (item.second.first > 0U) {
      const auto prev_num = item.second.first;
      if (node_index == prev_num) {
        // Profiling Before root node.
        for (uint32_t i = 1U; i <= prev_num; ++i) {
          GE_CHK_BOOL_RET_STATUS((node_index - i) < node_size, FAILED,
                                 "[Check][Param] node index:%zu invalid, node size:%zu", node_index, node_size);
          const auto &curr_item = graph_item.node_items_[node_index - i];
          (void)graph_item.root_items_.emplace(graph_item.root_items_.begin(), curr_item);
        }
      } else {
        GE_CHK_BOOL_RET_STATUS(((node_index - prev_num) - 1U) < node_size, FAILED,
                               "[Check][Param] prev index:%u invalid, node index:%zu, node size:%zu",
                               prev_num, node_index, node_size);
        const auto &prev_item = graph_item.node_items_[(node_index - prev_num) - 1U];
        for (uint32_t i = 1U; i <= prev_num; ++i) {
          GE_CHK_BOOL_RET_STATUS((node_index - i) < node_size, FAILED,
                                 "[Check][Param] node index:%zu invalid, node size:%zu", node_index, node_size);
          const auto &curr_item = graph_item.node_items_[node_index - i];
          prev_item->SetCtrlSend(curr_item, std::numeric_limits<uint32_t>::max());
          curr_item->SetCtrlSend(node_item, std::numeric_limits<uint32_t>::max());
        }
      }
    }

    if (item.second.second > 0U) {
      const auto post_num = item.second.second;
      if (node_size == ((node_index + post_num) + 1U)) {
        // Profiling After last node.
        for (uint32_t i = 1U; i <= post_num; ++i) {
          GE_CHK_BOOL_RET_STATUS((node_index + i) < node_size, FAILED,
                                 "[Check][Param] node index:%zu invalid, node size:%zu", node_index, node_size);
          const auto &curr_item = graph_item.node_items_[node_index + i];
          node_item->SetCtrlSend(curr_item, std::numeric_limits<uint32_t>::max());
        }
      } else {
        GE_CHK_BOOL_RET_STATUS(((node_index + post_num) + 1U) < node_size, FAILED,
                               "[Check][Param] post num:%u invalid, node index:%zu, node size:%zu",
                               post_num, node_index, node_size);
        std::set<const NodeItem *> post_items;
        post_items.insert(node_item->data_send_.cbegin(), node_item->data_send_.cend());
        post_items.insert(node_item->ctrl_send_.cbegin(), node_item->ctrl_send_.cend());
        for (uint32_t i = 1U; i <= post_num; ++i) {
          GE_CHK_BOOL_RET_STATUS((node_index + i) < node_size, FAILED,
                                 "[Check][Param] node index:%zu invalid, node size:%zu", node_index, node_size);
          const auto &curr_item = graph_item.node_items_[node_index + i];
          node_item->SetCtrlSend(curr_item, std::numeric_limits<uint32_t>::max());
          GELOGI("Set control from %s to %s.", node_item->NodeName().c_str(), curr_item->NodeName().c_str());
          for (const auto &node_send : post_items) {
            NodeItem *post_item = nullptr;
            GE_CHK_STATUS_RET_NOLOG(GetOrCreateNodeItem(node_send->node, post_item));
            curr_item->SetCtrlSend(post_item, std::numeric_limits<uint32_t>::max());
            GELOGI("Set control from %s to %s.", curr_item->NodeName().c_str(), post_item->NodeName().c_str());
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status HybridModelBuilder::BuildFrameGroupIndex(NodeItem &node_item) {
  if (node_item.is_root_node_) {
    GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
           node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
    return SUCCESS;
  }

  int64_t ctrl_flow_group = -1;
  if (node_item.IsEnterOp() && AttrUtils::GetInt(node_item.op_desc, ATTR_NAME_CONTROL_FLOW_GROUP, ctrl_flow_group)) {
    return BuildFrameGroupIndexForEnter(node_item, ctrl_flow_group);
  }

  for (const auto &src_node : node_item.node->GetInAllNodes()) {
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(src_node, src_node_item),
                      "[GetOrCreate][NodeItem] for [%s(%s)] failed",
                      src_node->GetName().c_str(), src_node->GetType().c_str());
    if (src_node_item->is_root_node_) {
      continue;
    }

    if (src_node_item->IsExitOp()) {
      const std::map<int64_t, int64_t>::const_iterator it = parent_frame_group_.find(src_node_item->frame_index_);
      node_item.frame_index_ = (it != parent_frame_group_.cend()) ? it->second : -1;
    } else {
      node_item.frame_index_ = src_node_item->frame_index_;
    }

    const std::map<int64_t, int64_t>::const_iterator it = parent_frame_group_.find(node_item.frame_index_);
    node_item.parent_frame_ = (it != parent_frame_group_.cend()) ? it->second : -1;
    GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
           node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
    return SUCCESS;
  }

  GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
         node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
  return SUCCESS;
}

Status HybridModelBuilder::BuildFrameGroupIndexForEnter(NodeItem &node_item, const int64_t ctrl_flow_group) {
  node_item.frame_index_ = ctrl_flow_group;
  for (const auto &src_node : node_item.node->GetInAllNodes()) {
    NodeItem *src_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(src_node, src_node_item),
                      "[GetOrCreate][NodeItem] for [%s(%s)] failed",
                      src_node->GetName().c_str(), src_node->GetType().c_str());
    if (!src_node_item->is_root_node_) {
      if (src_node_item->IsExitOp()) {
        const std::map<int64_t, int64_t>::const_iterator iter = parent_frame_group_.find(src_node_item->frame_index_);
        parent_frame_group_[node_item.frame_index_] = (iter != parent_frame_group_.cend()) ? iter->second : -1;
        GELOGD("[%s] frame index: %ld, [%s] is Exit, parent frame index: %ld, real parent frame index: %ld.",
               node_item.node_name.c_str(), node_item.frame_index_, src_node_item->node_name.c_str(),
               src_node_item->frame_index_, parent_frame_group_[node_item.frame_index_]);
      } else {
        GELOGD("[%s] frame index: %ld, [%s] parent frame index: %ld", node_item.node_name.c_str(),
               node_item.frame_index_, src_node_item->node_name.c_str(), src_node_item->frame_index_);
        parent_frame_group_[node_item.frame_index_] = src_node_item->frame_index_;
      }
      break;
    }
  }

  const std::map<int64_t, int64_t>::const_iterator it = parent_frame_group_.find(node_item.frame_index_);
  node_item.parent_frame_ = (it != parent_frame_group_.cend()) ? it->second : -1;
  GELOGD("[%s] control flow frame group: %ld, parent frame: %ld",
         node_item.node_name.c_str(), node_item.frame_index_, node_item.parent_frame_);
  return SUCCESS;
}

Status HybridModelBuilder::BuildControlFlowGroup(GraphItem &graph_item, const NodePtr &node,
                                                 NodeItem &node_item) const {
  GELOGD("Build control flow for node %s", node->GetName().c_str());
  using GroupBuilder = std::function<Status(const HybridModelBuilder *, const NodePtr &, NodeItem &)>;
  static const std::map<std::string, GroupBuilder> control_flow {
    { STREAMACTIVE, &HybridModelBuilder::CreateStreamActiveGroup },
    { STREAMSWITCH, &HybridModelBuilder::CreateStreamSwitchGroup },
    { NEXTITERATION, &HybridModelBuilder::CreateNextIterationGroup },
    { REFNEXTITERATION, &HybridModelBuilder::CreateNextIterationGroup },
    { SWITCH, &HybridModelBuilder::CreateSwitchGroup },
    { REFSWITCH, &HybridModelBuilder::CreateSwitchGroup },
    { LABELSET, &HybridModelBuilder::CreateNotImplement },
    { LABELGOTOEX, &HybridModelBuilder::CreateNotImplement },
    { LABELSWITCHBYINDEX, &HybridModelBuilder::CreateNotImplement }
  };

  Status ret = SUCCESS;
  const auto it = control_flow.find(node_item.node_type);
  if (it == control_flow.end()) {
    ret = CreateNormalNodeGroup(node, node_item);
  } else {
    graph_item.has_ctrl_flow_op_ = true;
    ret = it->second(this, node, node_item);
  }
  GELOGD("Node: %s, control by: %zu, control for: %zu, switch group: %zu", node->GetName().c_str(),
         node_item.ctrl_recv_.size(), node_item.ctrl_send_.size(), node_item.switch_groups_.size());
  return ret;
}

Status HybridModelBuilder::CreateNormalNodeGroup(const NodePtr &node, NodeItem &node_item) const {
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    if ((dst_node->GetType() == STREAMACTIVE) && (kHybridStreamActiveTypes.count(node->GetType()) == 0U)) {
      GELOGI("[%s] ignore control to [%s]", node->GetName().c_str(), dst_node->GetName().c_str());
      continue;
    }

    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                      "[GetOrCreate][NodeItem] for [%s(%s)] failed",
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    node_item.SetCtrlSend(dst_node_item, std::numeric_limits<uint32_t>::max());
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateMergeEnterGroup(const NodePtr &node, NodeItem &node_item) const {
  // Enter --> StreamActive --> StreamMerge
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    if (dst_node->GetType() != STREAMMERGE) {
      GELOGI("[%s] Skip Not StreamMerge node [%s]", node->GetName().c_str(), dst_node->GetName().c_str());
      continue;
    }
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                      "[GetOrCreate][NodeItem] [%s(%s)] failed",
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    // Set Enter Control to StreamMerge as Group 0.
    dst_node_item->switch_groups_.resize(kHybridLoopMergeSize);
    dst_node_item->SetMergeCtrl(&node_item, kHybridLoopEnterIdx);
    node_item.SetDataSend(dst_node_item, 0);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateMergeIterationGroup(const NodePtr &node, NodeItem &node_item) const {
  // NextIteration --> StreamActive {-->} StreamMerge
  for (const auto &src_node : node->GetInControlNodes()) {
    GE_CHECK_NOTNULL(src_node);
    if (kNextIterationOpTypes.count(src_node->GetType()) == 0U) {
      GELOGI("[%s] Skip Not NextIteration node [%s]", node->GetName().c_str(), src_node->GetName().c_str());
      continue;
    }
    auto node_name = AttrUtils::GetStr(src_node->GetOpDescBarePtr(), ATTR_NAME_NEXT_ITERATION);
    if (node_name == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Get][Attr] [%s] on input node [%s(%s)] failed",
             ATTR_NAME_NEXT_ITERATION.c_str(), src_node->GetName().c_str(), src_node->GetType().c_str());
      return INTERNAL_ERROR;
    }

    const auto it = stream_merge_op_nodes_.find(*node_name);
    if (it == stream_merge_op_nodes_.end()) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] [%s(%s)] expect StreamMerge[%s] not found",
             node->GetName().c_str(), node->GetType().c_str(), (*node_name).c_str());
      return INTERNAL_ERROR;
    }

    const auto &dst_node = it->second;
    GE_CHECK_NOTNULL(dst_node);
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                      "[GetOrCreate][NodeItem] for [%s(%s)] failed",
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    // Set NextIteration Control to StreamMerge as Group 1.
    dst_node_item->SetMergeCtrl(&node_item, kHybridLoopIterationIdx);
    node_item.SetDataSend(dst_node_item, 1);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateStreamActiveGroup(const NodePtr &node, NodeItem &node_item) const {
  if (node_item.node_type != STREAMACTIVE) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Called by %s is invalid", node_item.node_type.c_str());
    return INTERNAL_ERROR;
  }

  const auto ctrl_nodes = node->GetInControlNodes();
  if (ctrl_nodes.empty()) {
    GELOGW("Skip no in control node: %s", node->GetName().c_str());
    return SUCCESS;
  }

  const auto IsEnterNode = [](const NodePtr &n) {
    return kEnterOpTypes.count(n->GetType()) > 0U;
  };
  const auto IsIterationNode = [](const NodePtr &n) {
    return kNextIterationOpTypes.count(n->GetType()) > 0U;
  };

  if (std::any_of(ctrl_nodes.begin(), ctrl_nodes.end(), IsEnterNode)) {
    // Enter --> StreamActive --> StreamMerge
    node_item.is_enter_active_ = true;
    return CreateMergeEnterGroup(node, node_item);
  }
  if (std::any_of(ctrl_nodes.begin(), ctrl_nodes.end(), IsIterationNode)) {
    // NextIteration --> StreamActive {-->} StreamMerge
    return CreateMergeIterationGroup(node, node_item);
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateStreamSwitchGroup(const NodePtr &node, NodeItem &node_item) const {
  if (node_item.node_type != STREAMSWITCH) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Called by %s is invalid", node_item.node_type.c_str());
    return INTERNAL_ERROR;
  }

  // Consider as two groups, group[0] set empty for false, group[1] for true.
  node_item.switch_groups_.resize(kHybridStreamSwitchNum);
  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                      "[GetOrCreate][NodeItem] for [%s(%s)] failed",
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    node_item.SetCtrlSend(dst_node_item, kHybridStreamSwitchIdx);
  }
  return SUCCESS;
}

Status HybridModelBuilder::CreateNextIterationGroup(const NodePtr &node, NodeItem &node_item) const {
  if ((node_item.node_type != NEXTITERATION) && (node_item.node_type != REFNEXTITERATION)) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Called by %s is invalid", node_item.node_type.c_str());
    return INTERNAL_ERROR;
  }

  return CreateNormalNodeGroup(node, node_item);
}

Status HybridModelBuilder::CreateSwitchGroup(const NodePtr &node, NodeItem &node_item) const {
  if ((node_item.node_type != SWITCH) && (node_item.node_type != REFSWITCH)) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] Called by %s is invalid", node_item.node_type.c_str());
    return INTERNAL_ERROR;
  }

  for (const auto &dst_node : node->GetOutControlNodes()) {
    GE_CHECK_NOTNULL(dst_node);
    NodeItem *dst_node_item = nullptr;
    GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                      "[GetOrCreate][NodeItem] [%s(%s)] failed",
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    node_item.SetCtrlSend(dst_node_item, std::numeric_limits<uint32_t>::max());
  }

  // Group switch flow by out put data.
  node_item.switch_groups_.resize(static_cast<size_t>(SWITCH_OUTPUT_NUM));
  for (uint32_t i = 0U; i < SWITCH_OUTPUT_NUM; ++i) {
    for (const auto &dst_node : node->GetOutDataNodes()) {
      GE_CHECK_NOTNULL(dst_node);
      NodeItem *dst_node_item = nullptr;
      GE_CHK_STATUS_RET(GetOrCreateNodeItem(dst_node, dst_node_item),
                        "[GetOrCreate][NodeItem] [%s(%s)] failed",
                        dst_node->GetName().c_str(), dst_node->GetType().c_str());
      node_item.SetCtrlSend(dst_node_item, i); // take switch data as ctrl.
    }
  }

  return SUCCESS;
}

Status HybridModelBuilder::CreateNotImplement(const NodePtr &node, NodeItem &node_item) const {
  GELOGE(UNSUPPORTED, "[Check][Param] [%s:%s] Not implemented.", node->GetName().c_str(), node_item.node_type.c_str());
  return UNSUPPORTED;
}

Status HybridModelBuilder::InitAippInfoAndType() {
  return hybrid_model_.InitAippInfoAndType();
}

Status HybridModelBuilder::RecoverShapeConsistency(const ComputeGraph &root_graph) {
  for (const auto &node : root_graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHECK_LE(op_desc->GetOutputsSize(), static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const auto num_outputs = static_cast<int32_t>(op_desc->GetOutputsSize());
    for (int32_t output_index = 0; output_index < num_outputs; ++output_index) {
      const auto &tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(output_index));
      GE_CHECK_NOTNULL(tensor_desc);
      if (tensor_desc->MutableShape().IsUnknownShape()) {
        continue;
      }

      const auto &output_anchor = node->GetOutDataAnchor(output_index);
      GE_CHECK_NOTNULL(output_anchor);
      for (const auto &in_data_anchor : output_anchor->GetPeerInDataAnchorsPtr()) {
        const auto &peer_node = in_data_anchor->GetOwnerNodeBarePtr();
        GE_CHECK_NOTNULL(peer_node);
        const auto &peer_op_desc = peer_node->GetOpDescBarePtr();
        GE_CHECK_NOTNULL(peer_op_desc);
        const auto &peer_tensor_desc = peer_op_desc->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
        GE_CHECK_NOTNULL(peer_tensor_desc);
        if (peer_tensor_desc->GetShape().IsUnknownShape()) {
          GELOGW("Inconsistency detected, src_tensor = %s:%d, dst_tensor = %s:%d",
                 op_desc->GetName().c_str(),
                 output_index,
                 peer_op_desc->GetName().c_str(),
                 in_data_anchor->GetIdx());
          GELOGI("Update peer input shape from [%s] to [%s], origin shape from [%s] to [%s]",
                 peer_tensor_desc->GetShape().ToString().c_str(),
                 tensor_desc->GetShape().ToString().c_str(),
                 peer_tensor_desc->GetOriginShape().ToString().c_str(),
                 tensor_desc->GetOriginShape().ToString().c_str());
          peer_tensor_desc->SetShape(tensor_desc->GetShape());
          (void)peer_tensor_desc->SetShapeRange({});
          peer_tensor_desc->SetOriginShape(tensor_desc->GetOriginShape());
          (void)peer_tensor_desc->SetOriginShapeRange({});
        }
      }
    }
  }

  return SUCCESS;
}
void HybridModelBuilder::LoadTbeKernelBinToOpDesc(const ModelTaskType task_type, const GeModelPtr &ge_model,
                                                  const NodePtr &node) {
  if ((task_type == ModelTaskType::MODEL_TASK_KERNEL) || (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) ||
      (task_type == ModelTaskType::MODEL_TASK_FFTS_PLUS)) {
    ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(node->GetOpDesc());
  }
}

void HybridModelBuilder::LoadSgtKernelBinToOpDesc(const OpDesc &op_desc, const ComputeGraphPtr &sub_graph,
                                                  const GeModelPtr &ge_model) {
  if ((op_desc.GetType() == PARTITIONEDCALL) && IsFftsGraphNode(op_desc)) {
    GELOGI("Load Kernel for FFTS-Plus node: %s", op_desc.GetName().c_str());
    const auto sgt_graph = sub_graph->GetSubgraph(op_desc.GetSubgraphInstanceName(0U));
    GE_CHECK_NOTNULL_JUST_RETURN(sgt_graph);
    for (const auto &sgt_node : sgt_graph->GetAllNodes()) {
      const auto &sgt_op_desc = sgt_node->GetOpDesc();
      GE_CHECK_NOTNULL_JUST_RETURN(sgt_op_desc);
      GELOGI("Load Kernel for FFTS-Plus graph node: %s", sgt_op_desc->GetName().c_str());
      ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(sgt_op_desc);
    }
  }
}
}  // namespace hybrid
}  // namespace ge
