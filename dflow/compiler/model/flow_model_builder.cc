/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/model/flow_model_builder.h"
#include "common/debug/log.h"
#include "dflow/base/model/model_relation.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "dflow/compiler/pne/npu/npu_process_node_engine.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "dflow/compiler/data_flow_graph/data_flow_attr_utils.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_auto_deployer.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_model_relation_builder.h"
#include "dflow/compiler/model/flow_model_cache.h"
#include "api/aclgrph/option_utils.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "common/thread_pool/thread_pool.h"

// need ge add interface
#include "graph/passes/pass_manager.h"
#include "graph/passes/standard_optimize/save_pass.h"
#include "graph/passes/feature/net_output_pass.h"
#include "graph/passes/control_flow_and_stream/data_pass.h"

namespace {
constexpr const char *ATTR_NAME_DATA_FLOW_DEVICE_MEM_CFG = "_dflow_logic_device_memory_config";
constexpr const char *kAttrNameInvokedByBuiltIn = "_dflow_invoked_by_built_in";
constexpr const char *kAttrNameInvokedModelFusionInputs = "_invoked_model_fusion_inputs";
constexpr const char *ATTR_NAME_DATA_FLOW_SUB_DATA_FLOW_DEPLOY_INFOS = "_sub_data_flow_deploy_infos";
constexpr const char *kDeployInfoFilePrefix = "deploy_info_file;";
constexpr const char *ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE = "_dflow_data_flow_scope";
std::string GetInputStr(const std::map<int32_t, std::string> &inputs_info) {
  std::stringstream ss;
  for (const auto &info : inputs_info) {
    ss << "[index:" << info.first << " " << "name:" << info.second << "] ";
  }
  return ss.str();
}
}  // namespace

namespace ge {
Status FlowModelBuilder::CheckCacheGraphIoNodesWithGraphAdded(const ComputeGraphPtr &cached_graph,
                                                              const ComputeGraphPtr &added_graph) {
  GE_CHECK_NOTNULL(cached_graph);
  GE_CHECK_NOTNULL(added_graph);
  bool is_data_flow_graph = false;
  (void)AttrUtils::GetBool(added_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, is_data_flow_graph);
  if (!is_data_flow_graph) {
    bool cache_is_data_flow_graph = false;
    (void)AttrUtils::GetBool(cached_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, cache_is_data_flow_graph);
    if (!cache_is_data_flow_graph) {
      GELOGI("Skip to compare input output number result of added graph is not dataflow graph");
      return SUCCESS;
    } else {
      GELOGE(FAILED, "Added graph must be dataflow graph while cached graph is dataflow graph.");
      return FAILED;
    }
  }
  GE_CHK_STATUS_RET(ModifyDataIndex(added_graph), "[ModifyDataIndex] failed, graph_name = %s",
                    added_graph->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(cached_graph->GetOutputNodes().size() == added_graph->GetOutputNodes().size(), FAILED,
      "Cache output num[%zu] is not equal to add output num[%zu].", cached_graph->GetOutputNodes().size(),
      added_graph->GetOutputNodes().size());
  std::map<int32_t, std::string> cached_input_info;
  std::map<int32_t, std::string> added_input_info;
  for (const auto &node : cached_graph->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      int32_t index = -1;
      (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index);
      cached_input_info[index] = node->GetName();
    }
  }
  for (const auto &node : added_graph->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      int32_t index = -1;
      (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index);
      added_input_info[index] = node->GetName();
    }
  }
  GE_CHK_BOOL_RET_STATUS(cached_input_info.size() == added_input_info.size(), FAILED, "Cache input info %s and added"
      " input info %s are mismatch.", GetInputStr(cached_input_info).c_str(), GetInputStr(added_input_info).c_str());
  for (const auto &cached_input : cached_input_info) {
    const auto iter = added_input_info.find(cached_input.first);
    if ((iter == added_input_info.cend()) || (iter->second != cached_input.second)) {
      GELOGE(FAILED, "Cache input info %s and added input info %s are mismatch.",
          GetInputStr(cached_input_info).c_str(), GetInputStr(added_input_info).c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FlowModelBuilder::BuildModel(Graph &graph, const std::vector<GeTensor> &input_tensors,
                                    const std::map<std::string, std::string> &options, FlowModelPtr &flow_model) const {
  ComputeGraphPtr root_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(root_graph);
  GE_DUMP(root_graph, "FlowGraphPreRunBegin");
  GELOGI("Build model start, graph id = %d, graph_name = %s", root_graph->GetGraphID(), root_graph->GetName().c_str());

  FlowModelCache flow_model_cache;
  GE_CHK_STATUS_RET(flow_model_cache.Init(root_graph), "Failed to init flow model cache");
  GE_CHK_STATUS_RET(flow_model_cache.TryLoadFlowModelFromCache(root_graph, flow_model),
                    "Failed to load flow model from cache.");
  if (flow_model != nullptr) {
    GEEVENT("Load flow model from cache success.");
    const auto compute_graph_cached = flow_model->GetRootGraph();
    GE_CHK_STATUS_RET(CheckCacheGraphIoNodesWithGraphAdded(compute_graph_cached, root_graph),
                     "Input nodes or outputs nodes in cached graph is not same as graph added.");
    GE_CHECK_NOTNULL(compute_graph_cached);
    graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph_cached);
    return SUCCESS;
  }

  flow_model = MakeShared<FlowModel>(root_graph);
  GE_CHECK_NOTNULL(flow_model);
  flow_model->SetModelName(root_graph->GetName());
  CacheParam cache_param = {flow_model_cache.EnableCache(), flow_model_cache.ManualCheck(),
                            flow_model_cache.DebugMode()};
  const Status ret = BuildModel(root_graph, input_tensors, options, flow_model, cache_param);
  GE_CHK_STATUS_RET(ret, "Build model failed.");
  flow_model->SetRootGraph(root_graph);
  GE_CHK_STATUS_RET(flow_model_cache.TryCacheFlowModel(flow_model), "Failed to cache flow model.");
  GELOGI("Build model successfully, graph id = %d, graph_name = %s",
         root_graph->GetGraphID(),
         root_graph->GetName().c_str());
  GE_DUMP(root_graph, "AfterBuildFlowModel");
  return SUCCESS;
}

Status FlowModelBuilder::BuildModel(ComputeGraphPtr &root_graph, const std::vector<GeTensor> &input_tensors,
                                    const std::map<std::string, std::string> &options, const FlowModelPtr &flow_model,
                                    const CacheParam &cache_param) const {
  Status ret = SUCCESS;
  bool is_data_flow_graph = false;
  (void)AttrUtils::GetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, is_data_flow_graph);
  if (is_data_flow_graph) {
    std::string deploy_info_path;
    if (GetContext().GetOption(OPTION_DATAFLOW_DEPLOY_INFO_PATH, deploy_info_path) != GRAPH_SUCCESS) {
      GELOGD("data flow deploy info path option[%s] does not exist.", OPTION_DATAFLOW_DEPLOY_INFO_PATH);
    } else {
      deploy_info_path = kDeployInfoFilePrefix + deploy_info_path;
    }
    DataFlowGraphParam df_param = {"", deploy_info_path, 0};
    GE_CHK_STATUS_RET_NOLOG(BuildDataFlowGraph(root_graph, options, flow_model, cache_param, df_param));
  } else {
    ret = BuildHeterogeneousModel(root_graph, input_tensors, options, flow_model);
  }
  GE_CHK_STATUS_RET_NOLOG(ModelRelationFlattener::Flatten(flow_model));
  return ret;
}

Status FlowModelBuilder::GetEschedPriority(const ComputeGraphPtr &graph, const std::string &attr_name,
                                           std::map<std::string, int32_t> &esched_priority) {
  if (AttrUtils::HasAttr(graph, attr_name)) {
    int64_t priority = 0;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(graph, attr_name, priority), FAILED,
                           "Failed to get attr[%s] from graph[%s].", attr_name.c_str(), graph->GetName().c_str());
    GELOGD("[ModelEschedPriority]: graph name[%s], attr name[%s], value[%ld]", graph->GetName().c_str(),
           attr_name.c_str(), priority);
    esched_priority[attr_name] = static_cast<int32_t>(priority);
  }
  return SUCCESS;
}

Status FlowModelBuilder::GetModelEschedPriority(const PneModelPtr &pne_model,
                                                std::map<std::string, int32_t> &esched_priority) {
  const auto &graph = pne_model->GetRootGraph();
  GE_CHK_STATUS_RET(GetEschedPriority(graph, ATTR_NAME_ESCHED_PROCESS_PRIORITY, esched_priority),
                    "Failed to get [%s] for graph name[%s].", ATTR_NAME_ESCHED_PROCESS_PRIORITY.c_str(),
                    graph->GetName().c_str());
  GE_CHK_STATUS_RET(GetEschedPriority(graph, ATTR_NAME_ESCHED_EVENT_PRIORITY, esched_priority),
                    "Failed to get [%s] for graph name[%s].", ATTR_NAME_ESCHED_EVENT_PRIORITY.c_str(),
                    graph->GetName().c_str());
  return SUCCESS;
}

Status FlowModelBuilder::BuildModelEschedPriority(const FlowModelPtr &flow_model) {
  const auto &submodels = flow_model->GetSubmodels();
  std::map<std::string, std::map<std::string, int32_t>> models_esched_priority;
  for (const auto &submodel : submodels) {
    std::map<std::string, int32_t> esched_priority;
    GE_CHK_STATUS_RET(GetModelEschedPriority(submodel.second, esched_priority),
                      "Failed to get model esched priority for model[%s].", submodel.first.c_str());
    if (esched_priority.empty()) {
      continue;
    }
    if (submodel.second->GetSubmodels().empty()) {
      models_esched_priority[submodel.second->GetModelName()] = esched_priority;
      continue;
    }
    for (const auto &sub_submodel : submodel.second->GetSubmodels()) {
      models_esched_priority[sub_submodel.second->GetModelName()] = esched_priority;
    }
  }
  flow_model->SetModelsEschedPriority(models_esched_priority);
  return SUCCESS;
}

Status FlowModelBuilder::MergeInvokedModel(const FlowModelPtr &flow_model, const std::string &invoke_key,
                                           const FlowModelPtr &invoked_flow_model, bool invoked_by_built_in) {
  auto model_relation = flow_model->GetModelRelation();
  GE_CHECK_NOTNULL(model_relation);
  auto invoked_model_relation = invoked_flow_model->GetModelRelation();
  GE_CHECK_NOTNULL(invoked_model_relation);
  model_relation->submodel_endpoint_infos.insert(invoked_model_relation->submodel_endpoint_infos.begin(),
                                                 invoked_model_relation->submodel_endpoint_infos.end());
  model_relation->endpoints.insert(model_relation->endpoints.cend(), invoked_model_relation->endpoints.begin(),
                                   invoked_model_relation->endpoints.end());
  model_relation->invoked_model_queue_infos.insert(invoked_model_relation->invoked_model_queue_infos.begin(),
                                                 invoked_model_relation->invoked_model_queue_infos.end());
  ModelRelation::InvokedModelQueueInfo queue_info{
      invoked_model_relation->root_model_endpoint_info.input_endpoint_names,
      invoked_model_relation->root_model_endpoint_info.output_endpoint_names};
  model_relation->invoked_model_queue_infos.emplace(invoke_key, std::move(queue_info));
  auto &invoked_submodels = invoked_flow_model->GetSubmodels();
  for (auto &invoked_submodel_iter : invoked_submodels) {
    auto &invoked_submodel = invoked_submodel_iter.second;
    GE_CHECK_NOTNULL(invoked_submodel);
    (void)AttrUtils::SetBool(invoked_submodel->GetRootGraph(), kAttrNameInvokedByBuiltIn, invoked_by_built_in);
    GE_CHK_STATUS_RET(flow_model->AddSubModel(invoked_submodel, invoked_submodel->GetModelType()),
                      "Failed to add invoked sub model[%s] to root model", invoked_submodel->GetModelName().c_str());
  }
  return SUCCESS;
}

Status FlowModelBuilder::GetInputDataTensorDescs(const ComputeGraph &graph,
                                                 std::vector<GeTensorDesc> &input_tensor_descs) {
  std::map<int32_t, NodePtr> ordered_data_node_map;
  for (const auto &node : graph.GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      int32_t index = -1;
      GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_INDEX, index), FAILED,
                             "Failed to get attr[%s] for node[%s]", ge::ATTR_NAME_INDEX.c_str(),
                             node->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(ordered_data_node_map.emplace(index, node).second, FAILED,
                             "Duplicated data index %d on graph %s", index, graph.GetName().c_str());
    }
  }

  input_tensor_descs.reserve(ordered_data_node_map.size());
  for (const auto &it : ordered_data_node_map) {
    const auto op_desc = it.second->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const GeTensorDesc &data_in_desc = op_desc->GetInputDesc(0);
    input_tensor_descs.emplace_back(data_in_desc.GetShape(), data_in_desc.GetFormat(), data_in_desc.GetDataType());
  }
  return SUCCESS;
}

Status FlowModelBuilder::UpdateTensorDescByOption(std::vector<GeTensorDesc> &input_tensor_descs,
                                                  const std::map<std::string, std::string> &options) {
  auto mode_iter = options.find(OPTION_EXEC_DYNAMIC_EXECUTE_MODE);
  bool enable_dynamic_execute_mode = (mode_iter != options.end()) && (mode_iter->second == "dynamic_execute");
  if (!enable_dynamic_execute_mode) {
    GELOGD("no need update by shape range, as can not find %s option in graph options or option value is empty.",
           OPTION_EXEC_DYNAMIC_EXECUTE_MODE);
    return SUCCESS;
  }
  auto iter = options.find(OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE);
  bool enable_input_shape_range = (iter != options.end()) && (!iter->second.empty());
  if (!enable_input_shape_range) {
    GELOGD("no need update by shape range, as can not find %s option in graph options or option value is empty.",
           OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE);
    return SUCCESS;
  }
  std::vector<std::vector<std::pair<int64_t, int64_t>>> range_vec;
  if (ParseInputShapeRange(iter->second, range_vec) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parse][ShapeRange] Parse dynamic input shape range failed.");
    return PARAM_INVALID;
  }
  if (range_vec.size() != input_tensor_descs.size()) {
    GELOGE(PARAM_INVALID, "[Check][Param] Dynamic input shape range size is %zu, inputs size is %zu. Not match.",
           range_vec.size(), input_tensor_descs.size());
    return PARAM_INVALID;
  }

  for (size_t input_index = 0UL; input_index < range_vec.size(); ++input_index) {
    const auto &input_ranges = range_vec[input_index];
    const auto &tensor_shape = input_tensor_descs[input_index].GetShape();
    std::vector<int64_t> shape;
    // if input is unknown dim num, use input range to generate input.
    if (tensor_shape.IsUnknownDimNum()) {
      for (const auto &range : input_ranges) {
        shape.emplace_back(range.first);
      }
    } else {
      if (tensor_shape.GetDimNum() != input_ranges.size()) {
        GELOGE(PARAM_INVALID, "[Check][Param] input[%zu] dim num=%zu != input range num = %zu.", input_index,
               tensor_shape.GetDimNum(), input_ranges.size());
        return PARAM_INVALID;
      }
      for (size_t dim_idx = 0UL; dim_idx < input_ranges.size(); ++dim_idx) {
        int64_t dim = tensor_shape.GetDim(dim_idx);
        // if unknown dim, use range instead.
        dim = (dim == UNKNOWN_DIM) ? input_ranges[dim_idx].first : dim;
        shape.emplace_back(dim);
      }
    }
    input_tensor_descs[input_index].SetShape(GeShape(shape));
  }
  return SUCCESS;
}

Status FlowModelBuilder::MakeInputTensors(const ComputeGraphPtr &graph,
                                          const std::map<std::string, std::string> &options,
                                          std::vector<GeTensor> &input_tensors) {
  std::vector<GeTensorDesc> input_tensor_descs;
  GE_CHK_STATUS_RET(GetInputDataTensorDescs(*graph, input_tensor_descs), "Failed to get input tensor descs, graph[%s]",
                    graph->GetName().c_str());
  if (input_tensor_descs.empty()) {
    GELOGD("graph no input data, no need make input tensor, graph[%s].", graph->GetName().c_str());
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(UpdateTensorDescByOption(input_tensor_descs, options),
                    "Failed to get update tensor desc, graph[%s]", graph->GetName().c_str());

  for (const auto &input_tensor_desc : input_tensor_descs) {
    input_tensors.emplace_back(input_tensor_desc);
  }
  return SUCCESS;
}

Status FlowModelBuilder::BuildFlowSubgraph(ComputeGraphPtr &graph, const std::vector<GeTensor> &input_tensors,
                                           const std::map<std::string, std::string> &options,
                                           const FlowModelPtr &flow_model) const {
  GELOGD("Begin to build flow subgraph[%s].", graph->GetName().c_str());
  // use subgraph options
  GetThreadLocalContext().SetGraphOption(options);
  return BuildGraph(graph, input_tensors, options, true, flow_model);
}

Status FlowModelBuilder::BuildGraph(ComputeGraphPtr &graph, const vector<GeTensor> &input_tensors,
                                    const map<std::string, std::string> &options, bool is_sub_graph,
                                    const FlowModelPtr &flow_model) const {
  GE_CHK_STATUS_RET(ProcessNetOutput(graph), "Failed to process net out put");
  GE_CHK_STATUS_RET(DoBuildGraph(graph, options, input_tensors, is_sub_graph, flow_model),
                    "Failed to build graph, graph[%s].", graph->GetName().c_str());
  GE_CHK_STATUS_RET(FlowModelHelper::EnsureWithModelRelation(flow_model),
                    "Graph[%s] ensure with model relation failed.", graph->GetName().c_str());
  GELOGD("Graph[%s] was build success.", graph->GetName().c_str());
  return SUCCESS;
}

Status FlowModelBuilder::BuildFlowSubgraph(ComputeGraphPtr graph, const std::map<std::string, std::string> &options,
                                           const FlowModelPtr &flow_model) const {
  GELOGD("prepare to build flow subgraph[%s].", graph->GetName().c_str());
  // generate input_tensors from node.
  std::vector<GeTensor> input_tensors;
  GE_CHK_STATUS_RET(MakeInputTensors(graph, options, input_tensors), "Failed to make input tensors, graph[%s].",
                    graph->GetName().c_str());
  return BuildFlowSubgraph(graph, input_tensors, options, flow_model);
}

Status FlowModelBuilder::CheckAndSetUdfInvokeKeys(const std::shared_ptr<PneModel> &pne_model,
                                                  const std::shared_ptr<ModelRelation> &model_relation) {
  if (pne_model->GetModelType() != PNE_ID_UDF) {
    return SUCCESS;
  }
  const auto &graph = pne_model->GetRootGraph();
  GE_CHECK_NOTNULL(graph);
  if (!AttrUtils::HasAttr(graph, dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS)) {
    // no attribute means no need invoke other model.
    return SUCCESS;
  }
  std::vector<std::string> invoke_keys;
  if (!AttrUtils::GetListStr(graph, dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS, invoke_keys) || invoke_keys.empty()) {
    GELOGW("Graph[%s] has [%s] attr but get failed or empty, model[%s].", graph->GetName().c_str(),
           dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS, pne_model->GetModelName().c_str());
    return SUCCESS;
  }
  auto invoke_model_queue_iter = model_relation->submodel_endpoint_infos.find(pne_model->GetModelName());
  if (invoke_model_queue_iter == model_relation->submodel_endpoint_infos.end()) {
    GELOGE(FAILED, "no invoke model queue info found in model relation, model[%s].", pne_model->GetModelName().c_str());
    return FAILED;
  }
  invoke_model_queue_iter->second.invoke_model_keys = invoke_keys;
  GELOGD("udf invoke key num=%zu, model[%s].", invoke_keys.size(), pne_model->GetModelName().c_str());
  return SUCCESS;
}

Status FlowModelBuilder::SetUdfInvokeKeysRecursively(const std::shared_ptr<PneModel> &pne_model,
                                                     const std::shared_ptr<ModelRelation> &model_relation,
                                                     int32_t depth) {
  GE_CHECK_NOTNULL(pne_model);
  GE_CHECK_NOTNULL(model_relation);
  const auto &submodels = pne_model->GetSubmodels();

  // leaf model
  if (submodels.empty()) {
    return CheckAndSetUdfInvokeKeys(pne_model, model_relation);
  }
  constexpr int32_t kMaxDepth = 16;
  if (depth >= kMaxDepth) {
    GELOGE(UNSUPPORTED, "Depth limit(%d) reached", kMaxDepth);
    return UNSUPPORTED;
  }
  for (const auto &submodel : submodels) {
    GE_CHK_STATUS_RET_NOLOG(SetUdfInvokeKeysRecursively(submodel.second, pne_model->GetModelRelation(), depth + 1));
  }
  return SUCCESS;
}

Status FlowModelBuilder::MergeDataFlowLoadedModel(const DataFlowGraph &data_flow_graph,
                                                  const FlowModelPtr &flow_model) {
  size_t loaded_model_num = data_flow_graph.GetAllLoadedModels().size();
  size_t invoked_model_num = 0;
  for (const auto &loaded_model_pair : data_flow_graph.GetAllLoadedModels()) {
    const auto &graph_name = loaded_model_pair.first;
    const auto &sub_flow_model = loaded_model_pair.second;
    // if model is invoked, the invoked key will be not empty.
    const auto &invoked_key = data_flow_graph.GetInvokedGraphKey(graph_name);
    if (invoked_key.empty()) {
      GE_CHK_STATUS_RET(flow_model->AddSubModel(sub_flow_model), "Failed to add loaded sub flow model to root model:%s",
                        graph_name.c_str());
    } else {
      auto invoked_by_built_in = data_flow_graph.InvokedByBuiltIn(invoked_key);
      GELOGI("sub flow model[%s] is invoked, invoke key=%s, invoked by built-in = %d",
             graph_name.c_str(), invoked_key.c_str(), static_cast<int32_t>(invoked_by_built_in));
      GE_CHK_STATUS_RET(MergeInvokedModel(flow_model, invoked_key, sub_flow_model, invoked_by_built_in),
                        "Failed to MergeInvokedModel, loaded graph_name[%s], invoked_key[%s].", graph_name.c_str(),
                        invoked_key.c_str());
      ++invoked_model_num;
    }
  }
  if (loaded_model_num > 0) {
    GEEVENT("Merge data flow loaded model end, loaded_model_num=%zu(invoked_model_num=%zu).", loaded_model_num,
            invoked_model_num);
  }
  return SUCCESS;
}

Status FlowModelBuilder::PostProcessSubFlowModel(const DataFlowGraph &data_flow_graph, const FlowModelPtr &flow_model,
                                                 const ComputeGraphPtr &subgraph, const FlowModelPtr &sub_flow_model) {
  const std::string graph_name = subgraph->GetName();
  GE_CHECK_NOTNULL(sub_flow_model);
  std::string pne_id = PNE_ID_NPU;
  (void)AttrUtils::GetStr(subgraph, ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
  if (pne_id == PNE_ID_UDF) {
    const auto &invoke_keys = data_flow_graph.GetInvokeKeys(graph_name);
    if (!invoke_keys.empty()) {
      GE_CHK_STATUS_RET(SetUdfInvokeKeysRecursively(sub_flow_model, sub_flow_model->GetModelRelation(), 0),
                        "Failed to set udf invoke key, graph[%s].", graph_name.c_str());
      std::string invoked_model_attrs;
      data_flow_graph.GetInvokedModelFusionAttrs(invoke_keys, invoked_model_attrs);
      if (!invoked_model_attrs.empty()) {
        const auto graph = sub_flow_model->GetRootGraph();
        GE_CHECK_NOTNULL(graph);
        (void)AttrUtils::SetStr(graph, kAttrNameInvokedModelFusionInputs, invoked_model_attrs);
        GELOGI("Set fusion attr size[%zu] for graph [%s] success.", invoked_model_attrs.size(), graph_name.c_str());
      }
    }
    GE_CHK_STATUS_RET(flow_model->AddSubModel(sub_flow_model, pne_id), "Failed to add sub flow model[%s].",
                      graph_name.c_str());
  } else {
    // if model is invoked, the invoked key will be not empty.
    const auto &invoked_key = data_flow_graph.GetInvokedGraphKey(graph_name);
    if (invoked_key.empty()) {
      sub_flow_model->SetModelName(graph_name);
      GE_CHK_STATUS_RET(flow_model->AddSubModel(sub_flow_model), "Failed to add sub flow model to root model:%s",
                        graph_name.c_str());
    } else {
      auto invoked_by_built_in = data_flow_graph.InvokedByBuiltIn(invoked_key);
      GE_CHK_STATUS_RET(MergeInvokedModel(flow_model, invoked_key, sub_flow_model, invoked_by_built_in),
                        "Failed to MergeInvokedModel, graph_name[%s], invoked_key[%s].", graph_name.c_str(),
                        invoked_key.c_str());
    }
  }
  return SUCCESS;
}

Status FlowModelBuilder::PostOfDataFlowSubGraphsBuild(const DataFlowGraph &data_flow_graph,
                                                      std::vector<std::future<Status>> &vector_future,
                                                      const std::vector<FlowModelPtr> &sub_flow_models,
                                                      const FlowModelPtr &flow_model) {
  Status result = SUCCESS;
  size_t i = 0U;
  for (const auto &graph_pair : data_flow_graph.GetAllSubgraphs()) {
    const auto &graph = graph_pair.second;
    auto ret = vector_future[i].get();
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to build dataflow graph[%s].", graph->GetName().c_str());
      result = ret;
    } else {
      const auto &sub_flow_model = sub_flow_models[i];
      ret = PostProcessSubFlowModel(data_flow_graph, flow_model, graph, sub_flow_model);
      if (ret != SUCCESS) {
        result = ret;
      }
    }
    ++i;
  }
  return result;
}

Status FlowModelBuilder::FindInvokesAndGetSubDataFlowDeployInfos(
    const DataFlowGraph &data_flow_graph, std::map<std::string, DataFlowGraphParam> &deploy_infos) {
  for (const auto &graph_pair : data_flow_graph.GetAllSubgraphs()) {
    const auto &graph = graph_pair.second;
    if (!data_flow_graph.IsInvokedGraph(graph->GetName())) {
      continue;
    }

    bool is_data_flow_graph = false;
    (void)AttrUtils::GetBool(graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, is_data_flow_graph);
    if (!is_data_flow_graph) {
      (void)AttrUtils::SetBool(graph, ATTR_NAME_DATA_FLOW_UDF_INVOKED_NN, true);
      continue;
    }

    std::string subgraph_infos;
    (void)AttrUtils::GetStr(graph, ATTR_NAME_DATA_FLOW_SUB_DATA_FLOW_DEPLOY_INFOS, subgraph_infos);

    const auto &parent_pp_name = data_flow_graph.IsRootDataFlow() ? graph->GetName() + "/"
                                 : data_flow_graph.GetDataFlowScope() + graph->GetName() + "/";
    // key: graph name, value: df scope to deploy info
    deploy_infos[graph->GetName()] = {parent_pp_name, subgraph_infos, data_flow_graph.GetDataFlowDepth() + 1};
  }
  return SUCCESS;
}

Status FlowModelBuilder::CheckInvokedDataFlowDepth(uint32_t depth) {
  constexpr uint32_t kMaxDepth = 4U;
  if (depth > kMaxDepth) {
    GELOGE(FAILED, "Dataflow graph depth is over 4.");
    return FAILED;
  }
  return SUCCESS;
}

Status FlowModelBuilder::BuildDataFlowSubGraphs(const DataFlowGraph &data_flow_graph,
                                                const std::map<std::string, std::string> &options,
                                                const FlowModelPtr &flow_model, const CacheParam &cache_param) const {
  std::vector<std::future<Status>> vector_future;
  std::vector<FlowModelPtr> sub_flow_models(data_flow_graph.GetAllSubgraphs().size(), nullptr);
  const std::string cache_dir = FlowModelCache::GetCacheDirFromContext();
  const std::string graph_key = FlowModelCache::GetGraphKeyFromContext();
  size_t i = 0U;
  std::map<std::string, DataFlowGraphParam> invoked_deploy_infos;
  FindInvokesAndGetSubDataFlowDeployInfos(data_flow_graph, invoked_deploy_infos);
  // use default 16 multi thread
  uint32_t thread_num = data_flow_graph.IsRootDataFlow() ? 16U : 1U;
  ThreadPool thread_pool("ge_hetc_bld", thread_num);
  for (const auto &graph_pair : data_flow_graph.GetAllSubgraphs()) {
    auto &sub_flow_model = sub_flow_models[i++];
    auto func = [this, &data_flow_graph, &graph_pair, &options, &cache_dir, &graph_key, &invoked_deploy_infos,
                 &sub_flow_model, &cache_param]() -> Status {
      auto graph = graph_pair.second;
      const std::string graph_name = graph->GetName();
      std::string pne_id = PNE_ID_NPU;
      (void)AttrUtils::GetStr(graph, ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
      (void)AttrUtils::SetStr(graph, ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE, data_flow_graph.GetDataFlowScope());
      GE_TRACE_START(BuildDataFlowSubGraph);
      FlowModelCache sub_flow_model_cache;
      GE_CHK_STATUS_RET(sub_flow_model_cache.InitSubmodelCache(graph, cache_dir, graph_key),
                        "Failed to init subgraphs flow model cache, graph[%s].", graph_name.c_str());
      GE_CHK_STATUS_RET(sub_flow_model_cache.TryLoadFlowModelFromCache(graph, sub_flow_model),
                        "Failed to load flow model from cache, graph[%s].", graph_name.c_str());
      if (sub_flow_model != nullptr) {
        GEEVENT("Load flow model from cache successfully, graph[%s], pne[%s].", graph_name.c_str(), pne_id.c_str());
        GE_CHK_STATUS_RET(UpdateDeployInfo(graph, sub_flow_model), "Failed to update deploy info for graph[%s].",
                          graph_name.c_str());
      } else {
        sub_flow_model = MakeShared<FlowModel>(graph);
        GE_CHECK_NOTNULL(sub_flow_model);
        if (pne_id == PNE_ID_UDF) {
          GE_CHK_STATUS_RET(this->DoBuildGraph(graph, options, {}, true, sub_flow_model),
                            "Failed to build graph, graph[%s].", graph_name.c_str());
          sub_flow_model->SetModelName(graph_name);
          GE_CHK_STATUS_RET(FlowModelHelper::EnsureWithModelRelation(sub_flow_model),
                            "Ensure with model relation failed, model name=%s.", graph_name.c_str());
        } else {
          const auto iter_ret = invoked_deploy_infos.find(graph_name);
          if (iter_ret != invoked_deploy_infos.end()) {
            // is invoked flow graph
            GE_CHK_STATUS_RET(CheckInvokedDataFlowDepth(iter_ret->second.df_depth),
                              "Failed to build data flow graph %s", graph_name.c_str());
            GE_CHK_STATUS_RET(BuildDataFlowGraph(graph, options, sub_flow_model, cache_param, iter_ret->second),
                              "Failed to build data flow graph %s", graph_name.c_str());
          } else {
            const auto &subgraph_options = data_flow_graph.GetGraphBuildOptions(graph_name);
            GE_CHK_STATUS_RET_NOLOG(this->BuildFlowSubgraph(graph, subgraph_options, sub_flow_model));
          }
        }

        GE_CHK_STATUS_RET(ModelRelationFlattener::Flatten(sub_flow_model), "Failed to flatten flow model[%s].",
                          graph_name.c_str());
        GE_CHK_STATUS_RET(sub_flow_model_cache.TryCacheFlowModel(sub_flow_model), "Failed to cache flow model[%s].",
                          graph_name.c_str());
      }
      std::string trace_log = "building data flow subgraph[" + graph_name + "], pne=" + pne_id;
      GE_COMPILE_TRACE_TIMESTAMP_END(BuildDataFlowSubGraph, trace_log.c_str());
      return SUCCESS;
    };
    std::future<Status> f = thread_pool.commit(func);
    GE_CHK_BOOL_RET_STATUS(f.valid(), FAILED, "Failed to build graph[%s].", graph_pair.second->GetName().c_str());
    vector_future.emplace_back(std::move(f));
  }
  GEEVENT("Submit dataflow graph[%s] all subgraph build task end, task num=%zu, subgraph num=%zu",
          data_flow_graph.GetName().c_str(), vector_future.size(), data_flow_graph.GetAllSubgraphs().size());
  GE_CHK_STATUS_RET(PostOfDataFlowSubGraphsBuild(data_flow_graph, vector_future, sub_flow_models, flow_model),
                    "Failed to build dataflow graph[%s].", data_flow_graph.GetName().c_str());
  return BuildModelEschedPriority(flow_model);
}

Status FlowModelBuilder::BuildDataFlowGraph(const ComputeGraphPtr &root_graph,
                                            const std::map<std::string, std::string> &options,
                                            const FlowModelPtr &flow_model, const CacheParam &cache_param,
                                            const DataFlowGraphParam &df_param) const {
  GE_TRACE_START(BuildDataFlowGraph);
  DataFlowGraph data_flow_graph(root_graph, df_param.df_scope, cache_param.enable_cache,
                                cache_param.manual_check, df_param.df_depth);
  GE_CHK_STATUS_RET_NOLOG(data_flow_graph.Initialize());
  GE_CHK_STATUS_RET(DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(data_flow_graph, df_param.deploy_info),
                    "Auto deploy data flow graph[%s] failed.", data_flow_graph.GetName().c_str());
  GE_CHK_STATUS_RET(DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(data_flow_graph),
                    "Update data flow graph[%s]'s deploy info failed.", data_flow_graph.GetName().c_str());

  // Normalize graph
  GE_CHK_STATUS_RET(ModifyDataIndex(root_graph), "[ModifyDataIndex] failed, graph_name = %s",
                    root_graph->GetName().c_str());
  GE_CHK_STATUS_RET(ProcessNetOutput(root_graph), "[ProcessNetOutput] failed, graph_name = %s",
                    root_graph->GetName().c_str());

  GE_CHK_STATUS_RET(DataFlowAttrUtils::SupplementFlowAttr(root_graph),
                    "Failed to supplement flow attr for graph[%s].", root_graph->GetName().c_str());
  std::unique_ptr<ModelRelation> model_relation;
  GE_CHK_STATUS_RET(DataFlowGraphModelRelationBuilder().BuildFromDataFlowGraph(data_flow_graph, model_relation),
                    "Failed to build ModelRelation from root graph: %s", root_graph->GetName().c_str());
  flow_model->SetModelRelation(std::shared_ptr<ModelRelation>(model_relation.release()));
  GE_CHK_STATUS_RET(MergeDataFlowLoadedModel(data_flow_graph, flow_model),
                    "Failed to merge data flow loaded models, graph: %s", root_graph->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(!data_flow_graph.GetAllSubgraphs().empty(), FAILED,
                         "The subgraphs is empty, please check your graph.");
  GE_CHK_STATUS_RET(BuildDataFlowSubGraphs(data_flow_graph, options, flow_model, cache_param),
                    "Failed to build data flow graph[%s].", root_graph->GetName().c_str());
  const auto &logic_dev_id_to_mem_cfg = root_graph->TryGetExtAttr(ATTR_NAME_DATA_FLOW_DEVICE_MEM_CFG,
      std::map<std::string, std::pair<uint32_t, uint32_t>>());
  flow_model->SetLogicDeviceToMemCfg(logic_dev_id_to_mem_cfg);
  // graph options may be changed by subgraph option, need reset root graph options
  GetThreadLocalContext().SetGraphOption(options);
  GE_CHK_STATUS_RET(RemoveDataFlowSubgraphs(flow_model, cache_param),
                    "Remove all subgraphs from dataflow root graph failed.");
  std::string trace_log = "building data flow graph[" + root_graph->GetName() + "]";
  GE_COMPILE_TRACE_TIMESTAMP_END(BuildDataFlowGraph, trace_log.c_str());
  return SUCCESS;
}

Status FlowModelBuilder::RemoveDataFlowSubgraphs(const FlowModelPtr &flow_model, const CacheParam &cache_param) {
  // subgraph cache without whole graph remove subgraphs
  if (cache_param.enable_cache && cache_param.debug_mode) {
    const auto compute_graph = flow_model->GetRootGraph();
    GE_CHECK_NOTNULL(compute_graph);
    auto subgraphs = compute_graph->GetAllSubgraphs();
    for (const auto &graph : subgraphs) {
      compute_graph->RemoveSubgraph(graph->GetName());
    }
    GEEVENT("Remove all subgraphs from root graph while subgraph cache is enable.");
  }
  return SUCCESS;
}

Status FlowModelBuilder::BuildHeterogeneousModel(ComputeGraphPtr &root_graph,
                                                 const std::vector<GeTensor> &input_tensors,
                                                 const std::map<std::string, std::string> &options,
                                                 const FlowModelPtr &flow_model) const {
  GE_CHK_STATUS_RET(BuildGraph(root_graph, input_tensors, options, false, flow_model), "Failed to build graph[%s].",
                    root_graph->GetName().c_str());
  return BuildModelEschedPriority(flow_model);
}

Status FlowModelBuilder::GetOrAssignDefaultEngine(const ComputeGraphPtr &compute_graph,
                                                  std::string &process_node_engine_id) {
  (void) ge::AttrUtils::GetStr(compute_graph, ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, process_node_engine_id);
  if (!process_node_engine_id.empty()) {
    if (GetContext().GetHostExecFlag()) {
      GE_CHK_BOOL_RET_STATUS(process_node_engine_id == PNE_ID_CPU, PARAM_INVALID, "option[%s] is HOST, but attr[%s] ",
                             GE_OPTION_EXEC_PLACEMENT, ATTR_NAME_PROCESS_NODE_ENGINE_ID.c_str());
    }
    static const std::set<std::string> kSupportedEngines = {PNE_ID_CPU, PNE_ID_NPU, PNE_ID_UDF};
    GE_CHK_BOOL_RET_STATUS(
        kSupportedEngines.find(process_node_engine_id) != kSupportedEngines.cend(), PARAM_INVALID,
        "unsupported process node, engine=%s, support list=%s", process_node_engine_id.c_str(),
        ToString(std::vector<std::string>(kSupportedEngines.cbegin(), kSupportedEngines.cend())).c_str());
  } else {
    process_node_engine_id = GetContext().GetHostExecFlag() ? PNE_ID_CPU : PNE_ID_NPU;
    (void) ge::AttrUtils::SetStr(compute_graph, ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, process_node_engine_id);
  }
  return SUCCESS;
}

Status FlowModelBuilder::InitProcessNodeEngines(const std::map<std::string, std::string> &options,
                                                const std::shared_ptr<ProcessNodeEngineImpl> &pneImpl) {
  auto &engines = ProcessNodeEngineManager::GetInstance().GetEngines();
  if (engines.find(PNE_ID_NPU) == engines.cend()) {
    GELOGW("[Initialize][NPUProcessNodeEngine] is not registered.");
    auto creator = []() -> ::ge::ProcessNodeEngine * {
      return new(std::nothrow) NPUProcessNodeEngine();
    };
    ProcessNodeEngineRegisterar pne_register __attribute__((unused))(PNE_ID_NPU, creator);
  }

  for (auto &process_node_engine_pair : engines) {
    GE_CHECK_NOTNULL(process_node_engine_pair.second);
    auto engine_id = process_node_engine_pair.first;
    // every graph manger has one ProcessNodeEngine instance
    auto pne = ProcessNodeEngineManager::GetInstance().CloneEngine(engine_id);
    if (pne != nullptr) {
      process_node_engines_[engine_id] = pne;
      GE_CHK_STATUS_RET(pne->Initialize(options),
                        "[Initialize][ProcessNodeEngine] %s failed.",
                        pne->GetEngineName().c_str());
      // special ProcessNodeEngine process
      if ((pne->GetEngineName() == PNE_ID_NPU) || (pne->GetEngineName() == PNE_ID_CPU)) {
        pne->SetImpl(pneImpl);
      }
    }
  }
  return SUCCESS;
}

void FlowModelBuilder::Finalize() {
  for (auto &pne_pair : process_node_engines_) {
    if (pne_pair.second != nullptr) {
      Status ret = pne_pair.second->Finalize();
      if (ret != SUCCESS) {
        GELOGE(ret, "[Finalize] %s process node engine failed!", pne_pair.first.c_str());
      }
    }
  }
  process_node_engines_.clear();
}

Status FlowModelBuilder::DoBuildGraph(ComputeGraphPtr &compute_graph,
                                      const std::map<std::string, std::string> &options,
                                      const std::vector<GeTensor> &input_tensors,
                                      bool is_sub_graph,
                                      const FlowModelPtr &flow_model) const {
  std::string pne_id;
  GE_CHK_STATUS_RET(GetOrAssignDefaultEngine(compute_graph, pne_id), "assign default engine failed.");
  ProcessNodeEnginePtr process_node_engine;
  GE_CHK_STATUS_RET_NOLOG(GetEngine(pne_id, process_node_engine));
  bool user_set_host_flag = GetContext().GetHostExecFlag();
  // set host placement flag if user not set
  if (!user_set_host_flag) {
    UpdateThreadLocalOptions(pne_id);
  }

  ScopeGuard clear_host_guard([user_set_host_flag, &pne_id] {
    // clear host placement flag if user not set
    if (!user_set_host_flag) {
      ClearThreadLocalOptions(pne_id);
    }
  });

  // dflow 添加子图固定从2000000000开始
  static std::atomic<uint32_t> inner_graph_id_gen_{2000000000};
  if (is_sub_graph) {
    compute_graph->SetGraphID(inner_graph_id_gen_++);
    GELOGD("reassign inner graph = %s, graph id=%u", compute_graph->GetName().c_str(), compute_graph->GetGraphID());
  }

  PneModelPtr pne_model = nullptr;
  Status build_ret =
      process_node_engine->BuildGraph(compute_graph->GetGraphID(), compute_graph, options, input_tensors, pne_model);
  GE_CHK_STATUS_RET(build_ret, "[Build][PneModel] failed, graph=%s, engine=%s", compute_graph->GetName().c_str(),
                    process_node_engine->GetEngineName().c_str());
  GELOGI("[Build][PneModel] successfully, graph=%s, engine=%s", compute_graph->GetName().c_str(),
         process_node_engine->GetEngineName().c_str());
  if (pne_model != nullptr) {
    GE_CHK_STATUS_RET(flow_model->AddSubModel(pne_model, pne_id),
                      "[Add][Submodel] failed, graph = %s",
                      compute_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status FlowModelBuilder::GetEngine(const std::string &pne_id, ProcessNodeEnginePtr &engine) const {
  const auto find_ret = process_node_engines_.find(pne_id);
  GE_CHK_BOOL_RET_STATUS(find_ret != process_node_engines_.cend(), GE_CLI_GE_NOT_INITIALIZED,
                         "[Run][GetEngine] failed find process node engine for pne_id: [%s].", pne_id.c_str());
  engine = find_ret->second;
  GE_CHECK_NOTNULL(engine, "process node engine is null, pne_id=%s.", pne_id.c_str());
  return SUCCESS;
}

void FlowModelBuilder::UpdateThreadLocalOptions(const std::string &pne_id) {
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  if (pne_id == PNE_ID_CPU) {
    graph_options[GE_OPTION_EXEC_PLACEMENT] = "HOST";
  } else {
    graph_options.erase(GE_OPTION_EXEC_PLACEMENT);
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

void FlowModelBuilder::ClearThreadLocalOptions(const std::string &pne_id) {
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  if (pne_id == PNE_ID_CPU) {
    graph_options.erase(GE_OPTION_EXEC_PLACEMENT);
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

Status FlowModelBuilder::UpdateDeployInfo(const ComputeGraphPtr &graph, const FlowModelPtr &flow_model) {
  std::string logic_device_id;
  std::string redundant_logic_device_id;
  (void)AttrUtils::GetStr(graph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id);
  (void)AttrUtils::GetStr(graph, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID, redundant_logic_device_id);
  for (const auto &sub_model : flow_model->GetSubmodels()) {
    GE_CHECK_NOTNULL(sub_model.second);
    std::string old_logic_device_id = sub_model.second->GetLogicDeviceId();
    if (old_logic_device_id != logic_device_id) {
        sub_model.second->SetLogicDeviceId(logic_device_id);
        GELOGD("Update logic device id from[%s] to [%s] for model[%s].", old_logic_device_id.c_str(),
               logic_device_id.c_str(), sub_model.first.c_str());
    }
    std::string old_redundant_logic_device_id = sub_model.second->GetRedundantLogicDeviceId();
    if (old_redundant_logic_device_id != redundant_logic_device_id) {
        sub_model.second->SetRedundantLogicDeviceId(redundant_logic_device_id);
        GELOGD("Update redundant logic device id from[%s] to [%s] for model[%s].",
               old_redundant_logic_device_id.c_str(),
               redundant_logic_device_id.c_str(), sub_model.first.c_str());
    }
  }
  return SUCCESS;
}

Status FlowModelBuilder::ProcessNetOutput(const ComputeGraphPtr &compute_graph) {
  PassManager graph_passes;
  GE_CHK_STATUS_RET(graph_passes.AddPass("ProcessNetOutput::SavePass", new (std::nothrow) SavePass),
                    "add SavePass failed");
  GE_CHK_STATUS_RET(graph_passes.AddPass("ProcessNetOutput::NetOutputPass", new (std::nothrow) NetOutputPass),
                    "add NetOutputPass failed");
  GE_CHK_STATUS_RET(graph_passes.AddPass("ProcessNetOutput::DataPass", new (std::nothrow) DataPass),
                    "add DataPass failed");  // Add NetOutput first.

  auto ret = graph_passes.Run(compute_graph);
  if ((ret != SUCCESS) && (ret != NOT_CHANGED)) {
    GELOGE(ret, "[Run][GraphPasses] process net output pass failed, ret:%u.", ret);
    return ret;
  }
  return SUCCESS;
}

Status FlowModelBuilder::ModifyDataIndex(const ComputeGraphPtr &compute_graph) {
  std::vector<OpDescPtr> data_desc;
  std::set<int64_t> indexes;
  GE_CHECK_NOTNULL(compute_graph);
  for (auto &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    auto op_desc = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      int64_t index = std::numeric_limits<int64_t>::max();
      (void)AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index);
      (void)indexes.insert(index);
      data_desc.emplace_back(op_desc);
    }
  }
  if (!indexes.empty()) {
    auto first_iter = indexes.begin();
    auto end_iter = indexes.end();
    --end_iter;
    auto data_num = static_cast<int64_t>(data_desc.size());
    // The valid index starts with 0 and increases by 1, and num is equal to data_node.
    if (indexes.size() != data_desc.size() || *first_iter != 0 || *end_iter != data_num - 1) {
      GELOGI("Graph[%s] input data index is invalid, set data index by topo order.", compute_graph->GetName().c_str());
      int64_t index = 0;
      for (auto &op : data_desc) {
        (void)AttrUtils::SetInt(op, ATTR_NAME_INDEX, index++);
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
