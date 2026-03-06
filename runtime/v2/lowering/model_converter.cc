/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_converter.h"
#include <cinttypes>
#include "graph_converter.h"
#include "graph/ir_definitions_recover.h"
#include "graph/op_desc.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/unfold/graph_unfolder.h"
#include "common/helper/model_parser_base.h"
#include "common/checker.h"
#include "ge/ge_feature_memory.h"
#include "common/memory/mem_type_utils.h"
#include "framework/common/helper/model_helper.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "static_compiled_graph_converter.h"
#include "common/ge_inner_attrs.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/host_resource_center/host_resource_center.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
// Number of stream resources used in dynamic graphs.
struct StreamResource {
  int64_t total_stream_num = 1;
  int64_t reusable_stream_num = 1;
  int64_t reusable_event_num = 0;
  int64_t reusable_notify_num = 0;
  int64_t attached_stream_num = 0;
};

ge::graphStatus GetNonRootModelResourceNum(const std::map<std::string, ge::GeModelPtr> &ge_models,
                                           const std::string &root_graph_name, int64_t &static_stream_num,
                                           int64_t &static_event_num, int64_t &static_notify_num) {
  for (const auto &it : ge_models) {
    const auto &name = it.first;
    const auto &ge_model = it.second;
    GE_ASSERT_NOTNULL(ge_model);
    if (name != root_graph_name) {
      int64_t model_stream_num = 0;
      (void)ge::AttrUtils::GetInt(ge_model, ge::ATTR_MODEL_STREAM_NUM, model_stream_num);
      static_stream_num += model_stream_num;
      int64_t model_event_num = 0;
      (void)ge::AttrUtils::GetInt(ge_model, ge::ATTR_MODEL_EVENT_NUM, model_event_num);
      static_event_num += model_event_num;
      int64_t model_notify_num = 0;
      (void)ge::AttrUtils::GetInt(ge_model, ge::ATTR_MODEL_NOTIFY_NUM, model_notify_num);
      static_notify_num += model_notify_num;
      GELOGI("Static sub model %s, stream_num %" PRId64 ", event_num %" PRId64 ", notify_num %" PRId64 ".",
             name.c_str(), model_stream_num, model_event_num, model_notify_num);
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus LoadSgtKernelBinToOpDesc(const ge::NodePtr &node, const ge::ComputeGraphPtr &graph,
                                         const ge::GeModelPtr &ge_model, const ge::ModelTaskType task_type) {
  if (task_type != ge::ModelTaskType::MODEL_TASK_FFTS_PLUS) {
    return ge::GRAPH_SUCCESS;
  }
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (op_desc->GetType() == ge::PARTITIONEDCALL) {
    GELOGD("Load Kernel for FFTS-Plus node: %s", node->GetNamePtr());
    const auto &sgt_graph = graph->GetSubgraph(op_desc->GetSubgraphInstanceName(0U));
    GE_CHECK_NOTNULL(sgt_graph);
    for (const auto sgt_node : sgt_graph->GetAllNodesPtr()) {
      const auto &sgt_op_desc = sgt_node->GetOpDesc();
      GE_CHECK_NOTNULL(sgt_op_desc);
      GELOGD("Load Kernel for FFTS-Plus graph node: %s", sgt_op_desc->GetNamePtr());
      ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(sgt_op_desc);
    }
  } else {
    GELOGD("Load Kernel for mix l2 node: %s", node->GetNamePtr());
    ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(node->GetOpDesc());
  }
  return ge::GRAPH_SUCCESS;
}
namespace {
void LoadTbeKernelBinToOpDesc(const ge::ModelTaskType task_type, const ge::GeModelPtr &ge_model,
                              const ge::NodePtr &node) {
  if ((task_type == ge::ModelTaskType::MODEL_TASK_KERNEL) ||
      (task_type == ge::ModelTaskType::MODEL_TASK_ALL_KERNEL)) {
    ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(node->GetOpDesc());
  }
}

void LoadCustAicpuKernelBinToOpdesc(const ge::GeModelPtr &ge_model, const ge::NodePtr &node) {
  ge_model->GetCustAICPUKernelStore().LoadCustAICPUKernelBinToOpDesc(node->GetOpDesc());
}

ge::graphStatus ReadInModelTaskDefs(const ge::ComputeGraphPtr &graph, const ge::GeModelPtr &model,
                                    std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> &nodes_to_task_defs) {
  // index task defs
  GELOGD("To index tasks for subgraph: %s", graph->GetName().c_str());
  std::unordered_map<int64_t, ge::NodePtr> node_map;
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_CHECK_NOTNULL(op_desc);
    node_map[op_desc->GetId()] = node;
  }

  // The sub model has been verified during task loading, and the root model returned success without any tasks
  if (model->GetModelTaskDefPtr() == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  const auto &tasks = model->GetModelTaskDefPtr()->task();
  for (int32_t i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    GELOGI("Task id = %d, task type = %d", i, task_def.type());
    const auto task_type = static_cast<ge::ModelTaskType>(task_def.type());
    // todo 这里挺挫的，下一步归一或去掉TaskDef
    uint32_t op_index = std::numeric_limits<uint32_t>::max();
    if (task_type == ge::ModelTaskType::MODEL_TASK_KERNEL) {
      op_index = task_def.kernel().context().op_index();
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_KERNEL_EX) {
      op_index = task_def.kernel_ex().op_index();
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_HCCL) {
      op_index = task_def.kernel_hccl().op_index();
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_ALL_KERNEL) {
      op_index = task_def.kernel_with_handle().context().op_index();
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_FFTS_PLUS) {
      op_index = task_def.ffts_plus_task().op_index();
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_DVPP) {
      op_index = task_def.dvpp_task().op_index();
    } else if (task_type == ge::ModelTaskType::MODEL_TASK_DSA) {
      op_index = task_def.dsa_task().op_index();
    } else {
      GELOGD("Skip task type: %d", static_cast<int32_t>(task_type));
      continue;
    }
    GELOGD("op_index = %u, task_type = %d", op_index, task_type);

    const auto iter = node_map.find(static_cast<int64_t>(op_index));
    if (iter == node_map.cend()) {
      GELOGE(ge::INTERNAL_ERROR, "[Find][Node]Failed to get node by op_index = %u", op_index);
      return ge::INTERNAL_ERROR;
    }

    const auto &node = iter->second;
    // todo for offline
    LoadTbeKernelBinToOpDesc(task_type, model, node);  // for offline
    if (LoadSgtKernelBinToOpDesc(node, graph, model, task_type) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Find][Node]Failed to load node[%s] kernel bin.", node->GetName().c_str());
      return ge::INTERNAL_ERROR;
    }
    LoadCustAicpuKernelBinToOpdesc(model, node);
    GELOGD("Task loaded for node: %s, task type = %d, op_index = %u", node->GetNamePtr(), task_type, op_index);
    nodes_to_task_defs[node].emplace_back(task_def);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadInCompileResults(const ge::ComputeGraphPtr &root_graph, const ge::GeRootModelPtr &root_model,
                                     std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> &nodes_to_task_defs,
                                     std::unordered_map<std::string, ge::GeModelPtr> &graph_to_static_models) {
  const auto &root_graph_name = root_graph->GetName();

  for (const auto &it : root_model->GetSubgraphInstanceNameToModel()) {
    const auto &name = it.first;
    const auto &ge_model = it.second;
    GE_CHECK_NOTNULL(ge_model);

    ge::ComputeGraphPtr sub_graph;
    if (name == root_graph_name) {
      sub_graph = root_graph;
    } else {
      sub_graph = root_graph->GetSubgraph(name);
    }
    if (sub_graph == nullptr) {
      continue;
    }

    // 对于const节点在子图中直连netoutput的场景，netoutput节点对placement没有要求，导致获取的是host的lowering result
    // 如果partitionedcall连接的对端节点是device placement，会导致额外的h2d拷贝。
    // 临时方案：考虑到const大多存在于input图，input图虽然是静态子图，因为没有task，并不会走davincimodel的流程，而是走的
    // lowering parititionedcall流程，此处修改让input图直接在merged graph中展开，规避const连接到netoutput节点的情况。
    // 临时方案只能解决input图，无法解决控制节点的动态子图中存在const直连netoutput的场景。
    // 正式方案需要在lowering时按需构造需要placement的lowering result。
    if (IsGraphStaticCompiled(sub_graph) && IsStaticCompiledGraphHasTaskToLaunch(ge_model.get())) {
      sub_graph->SetGraphUnknownFlag(false);
      GELOGI("Read-in static compiled graph %s", sub_graph->GetName().c_str());
      graph_to_static_models[sub_graph->GetName()] = ge_model;
      continue;
    }

    sub_graph->SetGraphUnknownFlag(true);
    GELOGI("Read-in dynamic compiled graph %s", sub_graph->GetName().c_str());
    auto ret = ReadInModelTaskDefs(sub_graph, ge_model, nodes_to_task_defs);
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
  }
  root_model->SetNodesToTaskDef(nodes_to_task_defs);
  root_model->SetGraphToStaticModels(graph_to_static_models);
  return ge::GRAPH_SUCCESS;
}

ge::ComputeGraphPtr FlattenComputeGraph(const ge::ComputeGraphPtr &graph) {
  ge::ComputeGraphPtr flatten_graph;
  GraphUnfolder::UnfoldSubgraphs(graph, flatten_graph);
  return flatten_graph;
}

LoweringGlobalData BuildGlobalData(const ge::ComputeGraphPtr &graph,
                                   std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> nodes_to_task_defs,
                                   std::unordered_map<std::string, ge::GeModelPtr> graph_to_static_models,
                                   ge::HostResourceCenter *const host_resource_center) {
  LoweringGlobalData global_data;
  for (const auto &node : graph->GetAllNodes()) {
    auto task_defs_iter = nodes_to_task_defs.find(node);
    if (task_defs_iter != nodes_to_task_defs.end()) {
      global_data.AddCompiledResult(node, {std::move(task_defs_iter->second)});
    }
  }
  for (auto &graph_to_static_model : graph_to_static_models) {
    global_data.AddStaticCompiledGraphModel(graph_to_static_model.first, &(*graph_to_static_model.second));
  }
  global_data.SetHostResourceCenter(host_resource_center);
  return global_data;
}

ge::graphStatus InitConstWeights(const ge::GeRootModelPtr &root_model, int64_t &graph_flatten_offset) {
  auto root_graph = root_model->GetRootGraph();
  const auto &root_graph_name = root_graph->GetName();
  // For constant in root graph
  const ge::Tensor::DeleteFunc kDoNothing = [](uint8_t *data) {(void)data;};
  for (const auto &subgraph_model : root_model->GetSubgraphInstanceNameToModel()) {
    GE_ASSERT_NOTNULL(subgraph_model.second, "Compiled model of %s is nullptr", subgraph_model.first.c_str());
    const auto &name = subgraph_model.first;
    const auto sub_model_weight_size  = static_cast<int64_t>(subgraph_model.second->GetWeightSize());
    GELOGD("set FLATTEN_OFFSET to model[%s], {%ld, %ld}", subgraph_model.second->GetName().c_str(),
           graph_flatten_offset, sub_model_weight_size);
    subgraph_model.second->SetAttr(
        ge::ATTR_NAME_GRAPH_FLATTEN_OFFSET,
        ge::GeAttrValue::CreateFrom<std::vector<int64_t>>({graph_flatten_offset, sub_model_weight_size}));
    if (sub_model_weight_size == 0) {
      GELOGD("weight is empty. subgraph_name = %s", subgraph_model.first.c_str());
      continue;
    }
    ge::ComputeGraphPtr sub_graph;
    if (name == root_graph_name) {
      sub_graph = root_graph;
    } else {
      sub_graph = root_graph->GetSubgraph(name);
    }
    GE_CHECK_NOTNULL(sub_graph);
    for (const auto &node : sub_graph->GetNodes(sub_graph->GetGraphUnknownFlag())) {
      if (node->GetType() != ge::CONSTANT) {
        continue;
      }
      const auto op_desc = node->GetOpDescBarePtr();
      GE_CHECK_NOTNULL(op_desc);
      ge::GeTensorPtr weight;
      if (!ge::AttrUtils::MutableTensor(op_desc, "value", weight)) {
        GELOGE(ge::INTERNAL_ERROR, "weight is empty. node = %s", node->GetName().c_str());
      }
      GE_CHECK_NOTNULL(weight);
      ge::GeTensorDesc &tensor_desc = weight->MutableTensorDesc();
      int64_t tensor_size = 0;
      GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0U));
      GE_CHK_GRAPH_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->MutableOutputDesc(0U), tensor_size),
                              "[Invoke][GetSize][%s(%s)] Failed to get output tensor size", node->GetNamePtr(),
                              node->GetTypePtr());
      int64_t data_offset = 0;
      GE_CHK_GRAPH_STATUS_RET(ge::TensorUtils::GetDataOffset(tensor_desc, data_offset),
                              "[Invoke][GetDataOffset][%s(%s)] Failed to get data offset", node->GetNamePtr(),
                              node->GetTypePtr());
      GELOGD("[%s] Start to init Const node [%s], size = %" PRId64 ", offset = %" PRId64 ", graph_offset = %" PRId64 "",
             root_graph_name.c_str(), node->GetNamePtr(), tensor_size, data_offset, graph_flatten_offset);
      const auto flatten_off = graph_flatten_offset + data_offset;
      const auto weight_size = static_cast<int64_t>(ge::TensorUtils::GetWeightSize(tensor_desc));
      tensor_desc.SetAttr(ge::ATTR_NAME_GRAPH_FLATTEN_OFFSET,
                          ge::GeAttrValue::CreateFrom<std::vector<int64_t>>({flatten_off, weight_size}));
      GELOGI("set offset to node[%s], offset[%ld], size[%ld]", node->GetNamePtr(), flatten_off, weight_size);
      GE_CHECK_NOTNULL(subgraph_model.second->GetWeightData() + static_cast<size_t>(data_offset));
      weight->SetData(subgraph_model.second->GetWeightData() + static_cast<size_t>(data_offset),
                      tensor_size, kDoNothing); // use zero copy to reduce host mem
    }
    graph_flatten_offset += sub_model_weight_size;
  }
  GELOGD("total weight data size[%ld]", graph_flatten_offset);
  root_model->SetWeightSize(graph_flatten_offset);
  return ge::SUCCESS;
}

// 获取模型中静态子图的最大workspace size
void GetRequiredStaicModelWsSize(std::unordered_map<std::string, ge::GeModelPtr> &graph_to_static_models,
                                 int64_t &require_size) {
  require_size = 0;
  for (const auto &model_info : graph_to_static_models) {
    int64_t total_hbm_mem_size = 0;
    const auto ge_model = model_info.second;
    const std::vector<ge::MemInfo> mem_infos = ge::ModelUtils::GetAllMemoryTypeSize(ge_model);
    for (const auto &mem_info : mem_infos) {
      if ((mem_info.memory_size > 0) && (mem_info.memory_type == RT_MEMORY_HBM)) {
        total_hbm_mem_size += mem_info.memory_size;
      }
    }
    require_size = total_hbm_mem_size > require_size ? total_hbm_mem_size : require_size;
  }
}

void CleanMultiStreamAttrs(ge::OpDesc *const op_desc) {
  op_desc->SetStreamId(0);
  if (ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_SEND_EVENT_IDS)) {
    ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_SEND_EVENT_IDS, {});
  }
  if (ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_RECV_EVENT_IDS)) {
    ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {});
  }
}

// Old om may have nodes whose stream_id is -1
ge::graphStatus RefreshStreamIdOfSingleStreamGraph(const ge::ComputeGraphPtr &root_graph) {
  for (const auto node : root_graph->GetDirectNodePtr()) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_CHECK_NOTNULL(op_desc);
    CleanMultiStreamAttrs(op_desc);
  }
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    if (!subgraph->GetGraphUnknownFlag()) {
      continue;
    }
    for (const auto node : subgraph->GetDirectNodePtr()) {
      const auto op_desc = node->GetOpDescBarePtr();
      GE_CHECK_NOTNULL(op_desc);
      CleanMultiStreamAttrs(op_desc);
    }
  }
  GELOGI("Finish to refresh nodes' stream_id in unknown graph to 0.");

  return ge::SUCCESS;
}

ge::graphStatus GetReusableStreamResourceNum(const ge::GeRootModelPtr &root_model, StreamResource &stream_resource) {
  auto root_graph = root_model->GetRootGraph();
  const auto &ge_models = root_model->GetSubgraphInstanceNameToModel();

  int64_t static_stream_num = 0;
  int64_t static_event_num = 0;
  int64_t static_notify_num = 0;
  GE_ASSERT_SUCCESS(GetNonRootModelResourceNum(ge_models, root_graph->GetName(), static_stream_num, static_event_num,
                                               static_notify_num));

  const auto iter_root = ge_models.find(root_graph->GetName());
  if (iter_root == ge_models.end()) {
    // 当前动态shape图展开没有默认开启，为了兼容attach流的多流场景引入的临时修改，待动态shape图展开改为默认开启后删除该代码
    int64_t model_stream_num{1};
    (void)ge::AttrUtils::GetInt(root_graph, ge::ATTR_MODEL_STREAM_NUM, model_stream_num);
    (void)ge::AttrUtils::GetInt(root_graph, ge::ATTR_MODEL_EVENT_NUM, stream_resource.reusable_event_num);
    (void)ge::AttrUtils::GetInt(root_graph, ge::ATTR_MODEL_NOTIFY_NUM, stream_resource.reusable_notify_num);
    (void)ge::AttrUtils::GetInt(root_graph, "_attached_stream_num", stream_resource.attached_stream_num);
    stream_resource.total_stream_num = model_stream_num + static_stream_num;
    GE_ASSERT_TRUE(model_stream_num > stream_resource.attached_stream_num);
    stream_resource.reusable_stream_num = model_stream_num - stream_resource.attached_stream_num;
    GELOGI("Root graph total stream_num %" PRId64 ", reusable stream num %" PRId64 ", attached stream num %" PRId64
           ", event_num %" PRId64 ", notify_num %" PRId64 ".",
           stream_resource.total_stream_num, stream_resource.reusable_stream_num, stream_resource.attached_stream_num,
           stream_resource.reusable_event_num, stream_resource.reusable_notify_num);

    return ge::GRAPH_SUCCESS;
  }

  int64_t total_event_num = 0;
  int64_t total_notify_num = 0;
  (void)ge::AttrUtils::GetInt(iter_root->second, ge::ATTR_MODEL_STREAM_NUM, stream_resource.total_stream_num);
  (void)ge::AttrUtils::GetInt(iter_root->second, ge::ATTR_MODEL_EVENT_NUM, total_event_num);
  (void)ge::AttrUtils::GetInt(iter_root->second, ge::ATTR_MODEL_NOTIFY_NUM, total_notify_num);
  (void)ge::AttrUtils::GetInt(iter_root->second, "_attached_stream_num", stream_resource.attached_stream_num);
  GELOGI("Root model %s, total_stream_num %" PRId64 ", attached_stream_num %" PRId64 ", event_num %" PRId64
         ", notify_num %" PRId64 ".",
         iter_root->first.c_str(), stream_resource.total_stream_num, stream_resource.attached_stream_num,
         total_event_num, total_notify_num);

  // default set stream_num at least one
  stream_resource.reusable_stream_num = 1;
  stream_resource.reusable_event_num = 0;
  stream_resource.reusable_notify_num = 0;
  // todo stream num = 1, some ut static streaam num will more than 1
  if ((stream_resource.total_stream_num != 0) && (stream_resource.total_stream_num != 1)) {
    GE_ASSERT_TRUE(static_stream_num >= 0);
    GE_ASSERT_TRUE(static_event_num >= 0);
    GE_ASSERT_TRUE(static_notify_num >= 0);
    GE_ASSERT_TRUE(stream_resource.attached_stream_num >= 0);
    int64_t occupied_stream_num = static_stream_num + stream_resource.attached_stream_num;
    GE_ASSERT_TRUE((stream_resource.total_stream_num > occupied_stream_num),
                   "Total stream num %" PRId64 " is insufficient, static stream nums is %" PRId64
                   ", attached stream nums is %" PRId64 ".",
                   stream_resource.total_stream_num, static_stream_num, stream_resource.attached_stream_num);
    GE_ASSERT_TRUE((total_event_num >= static_event_num),
                   "Total event num %" PRId64 " is less than static event nums is %" PRId64 ".", total_event_num,
                   static_event_num);
    GE_ASSERT_TRUE((total_notify_num >= static_notify_num),
                   "Total notify num %" PRId64 " is less than static notify nums is %" PRId64 ".", total_notify_num,
                   static_notify_num);
    stream_resource.reusable_stream_num = stream_resource.total_stream_num - occupied_stream_num;
    stream_resource.reusable_event_num = total_event_num - static_event_num;
    stream_resource.reusable_notify_num = total_notify_num - static_notify_num;
    GELOGI("Root graph total stream_num %" PRId64 ", reusable stream num %" PRId64 ", attached stream num %" PRId64
           ", event_num %" PRId64 ", notify_num %" PRId64 ".",
           stream_resource.total_stream_num, stream_resource.reusable_stream_num, stream_resource.attached_stream_num,
           stream_resource.reusable_event_num, stream_resource.reusable_notify_num);
  }
  return ge::GRAPH_SUCCESS;
}

bool NeedRollBackToSingleStream(int64_t total_stream_num, int64_t reusable_stream_num,
                                StreamAllocator *const stream_allocator, EventAllocator *const event_allocator,
                                NotifyAllocator *const notify_allocator) {
  if ((stream_allocator == nullptr) || (event_allocator == nullptr) || (notify_allocator == nullptr)) {
    GELOGD("Stream allocator or event allocator is null. Its come from acl. No need rollback.");
    return false;
  }
  uint32_t free_stream_num = 0U;
  auto ret = rtGetAvailStreamNum(RT_NORMAL_STREAM, &free_stream_num);
  if (ret != RT_ERROR_NONE) {
    GELOGW("Fail to get available stream num on device. Better to roll back to single stream.");
    return true;
  }
  if (static_cast<int64_t>(free_stream_num) < total_stream_num) {
    GEEVENT("Model total required %" PRId64 " streams, including reusable stream_num %" PRId64
            ", but current available stream num is %u. Need rollback to single stream",
            total_stream_num, reusable_stream_num, free_stream_num);
    return true;
  }
  return false;
}

ge::graphStatus ReserveReusableStreamResource(const ModelDesc &model_desc,
                                              const StreamAllocator *const stream_allocator,
                                              const EventAllocator *const event_allocator,
                                              const NotifyAllocator *const notify_allocator) {
  size_t stream_num = model_desc.GetReusableStreamNum() + model_desc.GetAttachedStreamNum();
  if (stream_num == 1U) {
    GELOGD("Model is single stream, no need acquire reusable stream");
    return ge::GRAPH_SUCCESS;
  }
  if ((stream_allocator == nullptr) || (event_allocator == nullptr) || (notify_allocator == nullptr)) {
    GELOGD("No external stream allocator during load, model will use inner stream allocator when executing.");
    return ge::GRAPH_SUCCESS;
  }

  auto streams = stream_allocator->AcquireStreams(stream_num);
  GE_ASSERT_NOTNULL(streams, "Failed to reserve streams, num %zu", stream_num);
  auto events = event_allocator->AcquireEvents(model_desc.GetReusableEventNum());
  GE_ASSERT_NOTNULL(events, "Failed to reserve events, num %zu", model_desc.GetReusableEventNum());

  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  auto notifies = notify_allocator->AcquireNotifies(device_id, model_desc.GetReusableNotifyNum());
  GE_ASSERT_NOTNULL(notifies, "Failed to reserve notifies, num %zu", model_desc.GetReusableNotifyNum());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CollectAndReserveStreamResource(const ge::GeRootModelPtr &root_model,
                                                StreamAllocator *const stream_allocator,
                                                EventAllocator *const event_allocator,
                                                NotifyAllocator *const notify_allocator,
                                                ModelDescHolder &model_desc_holder) {
  StreamResource resource;
  GE_ASSERT_SUCCESS(GetReusableStreamResourceNum(root_model, resource));
  const bool need_rollback = NeedRollBackToSingleStream(resource.total_stream_num, resource.reusable_stream_num,
                                                        stream_allocator, event_allocator, notify_allocator);
  int64_t used_stream_num = resource.reusable_stream_num + resource.attached_stream_num;
  if ((used_stream_num > 1) && need_rollback) {
    GE_ASSERT_SUCCESS(RefreshStreamIdOfSingleStreamGraph(root_model->GetRootGraph()));
    resource.reusable_stream_num = 1;
    resource.reusable_event_num = 0;
    resource.reusable_notify_num = 0;
    resource.attached_stream_num = 0;
  } else if (resource.reusable_stream_num == 1) {
    GE_ASSERT_SUCCESS(RefreshStreamIdOfSingleStreamGraph(root_model->GetRootGraph()));
  }

  GEEVENT("Model %s require reusable stream num is %" PRId64 ", attached stream num is %" PRId64 ", event num is %" PRId64
          ", notify num is %" PRId64 ".",
          root_model->GetModelName().c_str(), resource.reusable_stream_num, resource.attached_stream_num,
          resource.reusable_event_num, resource.reusable_notify_num);
  model_desc_holder.MutableModelDesc().SetReusableStreamNum(static_cast<size_t>(resource.reusable_stream_num));
  model_desc_holder.MutableModelDesc().SetReusableEventNum(static_cast<size_t>(resource.reusable_event_num));
  model_desc_holder.MutableModelDesc().SetReusableNotifyNum(static_cast<size_t>(resource.reusable_notify_num));
  model_desc_holder.MutableModelDesc().SetAttachedStreamNum(static_cast<size_t>(resource.attached_stream_num));
  GE_ASSERT_SUCCESS(ReserveReusableStreamResource(model_desc_holder.GetModelDesc(), stream_allocator, event_allocator,
                                                  notify_allocator));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetFixedFeatureMemory(const ge::GeRootModelPtr &root_model, LoweringGlobalData &global_data) {
  // fix feature map, hbm and p2p
  std::vector<ge::FeatureMemoryPtr> all_feature_memory;
  size_t hbm_fixed_feature_mem;
  GE_ASSERT_SUCCESS(root_model->GetSummaryFeatureMemory(all_feature_memory, hbm_fixed_feature_mem));
  (void) hbm_fixed_feature_mem;

  // 只设置长度，如果用户设置了地址，会覆盖
  for (const auto &summary_feature_mem : all_feature_memory) {
    if (summary_feature_mem->IsFixed()) {
      rtMemType_t rt_mem_type;
      GE_ASSERT_SUCCESS(ge::MemTypeUtils::ExternalMemTypeToRtMemType(summary_feature_mem->GetType(), rt_mem_type),
                        "external type: %s", ge::MemTypeUtils::ToString(summary_feature_mem->GetType()).c_str());
      global_data.SetFixedFeatureMemoryBase(rt_mem_type, nullptr, summary_feature_mem->GetSize());
      GELOGI("fixed_feature_memory type:%s, size:%zu",
             ge::MemTypeUtils::ToString(rt_mem_type).c_str(), summary_feature_mem->GetSize());
    }
  }

  const auto fixed_feature_mem = root_model->GetFixedFeatureMemory();
  for (const auto fixed_iter : fixed_feature_mem) {
    global_data.SetFixedFeatureMemoryBase(fixed_iter.first, fixed_iter.second.addr,
                                          fixed_iter.second.size);
    GELOGI("Set fixed_feature_memory base to global data. %s", fixed_iter.second.ToString().c_str());
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::ExecuteGraphPtr ModelConverter::ConvertGeModelToExecuteGraph(const ge::GeRootModelPtr &root_model,
                                                                 const Args &args) {
  if ((root_model == nullptr) || (root_model->GetRootGraph() == nullptr)) {
    return nullptr;
  }

  GE_ASSERT_SUCCESS(CreateModelDesc(root_model, args.stream_allocator, args.event_allocator, args.notify_allocator));

  auto root_graph = root_model->GetRootGraph();
  // Compile results of dynamic compiled graph
  std::unordered_map<ge::NodePtr, std::vector<domi::TaskDef>> nodes_to_task_defs;
  // Compile results of static compiled graph
  std::unordered_map<std::string, ge::GeModelPtr> graph_to_static_models;
  int64_t require_weight_size = 0;
  ge::ComputeGraphPtr flatten_graph = root_model->GetFlattenGraph();
  if (flatten_graph == nullptr) {
    GE_ASSERT_GRAPH_SUCCESS(ReadInCompileResults(root_graph, root_model, nodes_to_task_defs, graph_to_static_models));
    InitConstWeights(root_model, require_weight_size);
    if (GraphUnfolder::IsGraphNeedUnfold(root_graph)) {
      flatten_graph = FlattenComputeGraph(root_graph);
    } else {
      flatten_graph = root_graph;
    }
    GE_ASSERT_NOTNULL(flatten_graph);
    GE_ASSERT_GRAPH_SUCCESS(ge::RecoverIrDefinitions(flatten_graph), "Failed to recover ir definitions");
    root_model->SetFlattenGraph(flatten_graph);
  } else {
    nodes_to_task_defs = root_model->GetNodesToTaskDef();
    graph_to_static_models = root_model->GetGraphToStaticModels();
    require_weight_size = root_model->GetWeightSize();
  }

  int64_t require_static_model_ws_size = 0;
  GetRequiredStaicModelWsSize(graph_to_static_models, require_static_model_ws_size);
  LoweringGlobalData global_data =
      BuildGlobalData(flatten_graph, std::move(nodes_to_task_defs), std::move(graph_to_static_models),
                      root_model->GetHostResourceCenterPtr().get());
  global_data.SetModelWeightSize(static_cast<size_t>(require_weight_size));
  auto registries = GetModelDescHolder().GetSpaceRegistries();
  GE_ASSERT_NOTNULL(registries);
  global_data.SetSpaceRegistriesV2(*registries);
  global_data.SetLoweringOption(args.option);
  global_data.SetStaicModelWsSize(require_static_model_ws_size);
  GE_ASSERT_SUCCESS(SetFixedFeatureMemory(root_model, global_data));
  if (args.file_constant_mems != nullptr) {
    global_data.SetFileConstantMem(*args.file_constant_mems);
  }
  auto graph = GraphConverter()
                   .SetModelDescHolder(&model_desc_holder_)
                   .ConvertComputeGraphToExecuteGraph(flatten_graph, args.option, global_data);
  GE_ASSERT_NOTNULL(graph, "Failed lowering compute graph %s", flatten_graph->GetName().c_str());
  ge::DumpGraph(graph.get(), "ExecuteGraphAfterSplit");
  return graph;
}

ge::ExecuteGraphPtr LoadExecuteGraphFromModelFile(const ge::char_t *const model_path, ge::graphStatus &error_code) {
  ge::ModelParserBase base;
  ge::ModelData model_data;
  error_code = base.LoadFromFile(model_path, -1, model_data);
  if (error_code != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Failed to load model data form model path");
    return nullptr;
  }

  ge::ModelHelper model_helper;
  error_code = model_helper.LoadRootModel(model_data);
  if (error_code != ge::GRAPH_SUCCESS) {
    delete[] static_cast<char *>(model_data.model_data);
    model_data.model_data = nullptr;
    GELOGE(ge::FAILED, "Failed to load root model from model data");
    return nullptr;
  }

  // todo refact code
  delete[] static_cast<char *>(model_data.model_data);
  model_data.model_data = nullptr;

  auto graph = ModelConverter().ConvertGeModelToExecuteGraph(model_helper.GetGeRootModel());
  if (graph == nullptr) {
    error_code = ge::GRAPH_FAILED;
  }
  return graph;
}

ge::graphStatus ModelConverter::CreateModelDesc(const ge::GeRootModelPtr &root_model,
                                                StreamAllocator *const stream_allocator,
                                                EventAllocator *const event_allocator,
                                                NotifyAllocator *const notify_allocator) {
  // calc stream num and reserve stream and event
  GE_ASSERT_GRAPH_SUCCESS(CollectAndReserveStreamResource(root_model, stream_allocator, event_allocator,
                                                          notify_allocator, model_desc_holder_));
  std::shared_ptr<gert::OpImplSpaceRegistryV2Array> space_registries{nullptr};
  GE_ASSERT_SUCCESS(ge::ModelUtils::GetSpaceRegistries(root_model, space_registries));
  model_desc_holder_.SetSpaceRegistries(space_registries);
  model_desc_holder_.SetFileConstantWeightDir(root_model->GetFileConstantWeightDir());
  // 模型编译后序列化时，root_model可能没有GeModel,编译期造了一个空的GeModel给到了root_model，因此此处遍历一把去取模型上的属性(与子图无关属性）
  for (const auto &ge_model : root_model->GetSubgraphInstanceNameToModel()) {
    std::vector<std::string> out_node_name;
    if (ge::AttrUtils::GetListStr(ge_model.second, ge::ATTR_MODEL_OUT_NODES_NAME, out_node_name)) {
      GELOGD("Get model out node names success, size = %zu", out_node_name.size());
      model_desc_holder_.SetOutputNodeName(out_node_name);
      // 从ge_model里面找到了就返回
      break;
    }
  }
  return ge::SUCCESS;
}
}  // namespace gert
