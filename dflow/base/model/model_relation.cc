/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_relation.h"
#include "common/plugin/ge_make_unique_util.h"
#include "endpoint.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr int32_t kSubgraphIndex = 0;
constexpr uint32_t kDefaultQueueDepth = 128U;
constexpr int32_t kDataOutputAnchorIndex = 0;
constexpr int32_t kKernelInsideTransferType = 1;
const std::string kAttrIsolatedData = "_isolate_data_after_prune";
}  // namespace

Status ModelRelationBuilder::BuildFromRootGraph(const ComputeGraph &root_graph,
                                                std::unique_ptr<ModelRelation> &model_relation) {
  model_relation = MakeUnique<ModelRelation>();
  GE_CHECK_NOTNULL(model_relation);
  GE_CHK_STATUS_RET_NOLOG(DoBuild(root_graph));
  *model_relation = std::move(model_relation_);
  return SUCCESS;
}

Status ModelRelationBuilder::CreateQueueForDataNode(const Node &node, const std::string &prefix,
                                                    std::string &queue_name, const bool inner_node_flag) {
  queue_name = prefix + ":" + node.GetName();
  if (inner_node_flag) {
    GELOGD("Node:%s is inner data node, no need add to model relation, queue name is %s.",
           node.GetName().c_str(), queue_name.c_str());
    return SUCCESS;
  }
  bool is_dummy = false;
  (void)AttrUtils::GetBool(node.GetOpDesc(), kAttrIsolatedData, is_dummy);
  GELOGD("queue name is %s, is dummy %d.", queue_name.c_str(), static_cast<int32_t>(is_dummy));

  GE_CHK_STATUS_RET_NOLOG(
      CreateQueueDef(node.GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(kDataOutputAnchorIndex)),
                     queue_name, node, is_dummy));
  int64_t data_index = -1;
  (void) AttrUtils::GetInt(node.GetOpDesc(), ATTR_NAME_INDEX, data_index);
  if ((data_index < 0) || (data_index >= INT32_MAX)) {
    GELOGE(PARAM_INVALID, "[%s] Data index out of range, data index = %ld",
           node.GetName().c_str(), data_index);
    return PARAM_INVALID;
  }
  if (static_cast<size_t>(data_index) >= model_relation_.root_model_endpoint_info.input_endpoint_names.size()) {
    model_relation_.root_model_endpoint_info.input_endpoint_names.resize(static_cast<uint64_t>(data_index + 1));
  }
  model_relation_.root_model_endpoint_info.input_endpoint_names[static_cast<uint64_t>(data_index)] = queue_name;
  GELOGD("Get data node[%s] as input %ld", node.GetName().c_str(), data_index);
  return SUCCESS;
}

Status ModelRelationBuilder::BuildForSingleModel(const ComputeGraph &root_graph, ModelRelation &model_relation) {
  for (const auto &node : root_graph.GetDirectNode()) {
    const auto &op_type = node->GetType();
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if ((op_type == DATA) || OpTypeUtils::IsInputRefData(node->GetOpDesc())) {
      std::string unused;
      GE_CHK_STATUS_RET(CreateQueueForDataNode(*node, root_graph.GetName(), unused),
                        "Failed to create queue for data: %s", node->GetName().c_str());
    } else if (op_type == NETOUTPUT) {
      const size_t num_outputs = node->GetOpDesc()->GetAllInputsSize();
      for (size_t i = 0U; i < num_outputs; ++i) {
        const std::string queue_name = root_graph.GetName() + ":output:" + std::to_string(i);
        GE_CHK_STATUS_RET_NOLOG(
            CreateQueueDef(node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(i)), queue_name, *node));
        model_relation_.root_model_endpoint_info.output_endpoint_names.emplace_back(queue_name);
      }
    } else {
      // do nothing
    }
  }
  model_relation_.root_model_endpoint_info.model_name = root_graph.GetName();
  model_relation_.submodel_endpoint_infos[root_graph.GetName()] = model_relation_.root_model_endpoint_info;
  model_relation = std::move(model_relation_);
  return SUCCESS;
}

Status ModelRelationBuilder::CheckNetOutputNode(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(out_data_anchor);
    const auto peer_node = out_data_anchor->GetOwnerNodeBarePtr();
    GE_CHECK_NOTNULL(peer_node);
    if (peer_node->GetType() != PARTITIONEDCALL) {
      GELOGE(INTERNAL_ERROR, "Peer node of NetOutput is not a PartitionedCall, type = %s",
             peer_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuildForData(const NodePtr &node,
                                            std::map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                                            const ComputeGraph &root_graph) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GELOGD("Begin to build relation for data node: %s.", node->GetName().c_str());
  const bool inner_node_flag = CheckInnerNode(node);
  std::string queue_name;
  GE_CHK_STATUS_RET(CreateQueueForDataNode(*node, root_graph.GetName(), queue_name, inner_node_flag),
                    "Failed to create queue for data: %s", node->GetName().c_str());
  const auto &out_data_anchor = node->GetOutDataAnchor(kDataOutputAnchorIndex);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto &peer_node = in_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (peer_node->GetType() != PARTITIONEDCALL) {
      GELOGE(INTERNAL_ERROR, "Peer node of Data is not a PartitionedCall, type = %s", peer_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
    (void)paired_inputs[peer_node].emplace(in_data_anchor->GetIdx(), queue_name);
    if (!inner_node_flag) {
      const auto &op_desc = peer_node->GetOpDesc();
      ModelRelation::ModelEndpointInfo *dst_model_queues = nullptr;
      GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelEndpointInfo(*op_desc, dst_model_queues));
      const size_t in_anchor_idx = static_cast<size_t>(in_data_anchor->GetIdx());
      if (in_anchor_idx >= dst_model_queues->input_endpoint_names.size()) {
        dst_model_queues->input_endpoint_names.resize(in_anchor_idx + 1UL);
      }
      dst_model_queues->input_endpoint_names[in_anchor_idx] = queue_name;
    }
  }
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuildForPartitionedCall(const NodePtr &node,
                                                       std::map<NodePtr, std::map<int32_t,
                                                       std::string>> &paired_inputs) {
  // check all input are valid
  std::vector<std::string> unused;
  GE_CHK_STATUS_RET_NOLOG(GetInputQueueNames(node, paired_inputs, unused));
  // create queue for submodel outputs, and set input to peer submodel
  ModelRelation::ModelEndpointInfo *model_queues = nullptr;
  GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelEndpointInfo(*node->GetOpDesc(), model_queues));
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    const size_t output_idx = static_cast<size_t>(out_data_anchor->GetIdx());
    const std::string queue_name = node->GetName() + ":" + std::to_string(output_idx);
    const bool is_dummy = out_data_anchor->GetPeerInDataAnchors().empty();
    GELOGD("queue_name is %s, is_dummy[%d]", queue_name.c_str(), static_cast<int32_t>(is_dummy));
    bool all_output_inner_nodes_flag = !out_data_anchor->GetPeerInDataAnchors().empty();
    GELOGD("out_data_anchor->GetPeerInDataAnchors() size is %zu.", out_data_anchor->GetPeerInDataAnchors().size());
    for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(in_data_anchor);
      const auto &dequeue_node = in_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dequeue_node);
      GE_CHECK_NOTNULL(dequeue_node->GetOpDesc());
      const bool inner_node_flag = CheckInnerNode(dequeue_node);
      all_output_inner_nodes_flag = !inner_node_flag ? false : all_output_inner_nodes_flag;
      GELOGD("Dequeue node:%s, inner_node_flag:%d, all_output_inner_nodes_flag:%d.",
             dequeue_node->GetName().c_str(), static_cast<int32_t>(inner_node_flag),
             static_cast<int32_t>(all_output_inner_nodes_flag));
      if ((dequeue_node->GetType() == PARTITIONEDCALL) && (!inner_node_flag)) {
        ModelRelation::ModelEndpointInfo *dst_model_queues = nullptr;
        GE_CHK_STATUS_RET_NOLOG(GetOrCreateModelEndpointInfo(*dequeue_node->GetOpDesc(), dst_model_queues));
        const size_t input_idx = static_cast<size_t>(in_data_anchor->GetIdx());
        if (input_idx >= dst_model_queues->input_endpoint_names.size()) {
          dst_model_queues->input_endpoint_names.resize(input_idx + 1UL);
        }
        dst_model_queues->input_endpoint_names[input_idx] = queue_name;
        GELOGD("Save input queue_name:%s for node:%s, index:%d.",
               queue_name.c_str(), dequeue_node->GetName().c_str(), input_idx);
      }
      (void)paired_inputs[dequeue_node].emplace(in_data_anchor->GetIdx(), queue_name);
    }
    if (!all_output_inner_nodes_flag) {
      GE_CHK_STATUS_RET(CreateQueueDef(node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(output_idx)), queue_name,
                                       *node, is_dummy),
                        "Create queue in model relation failed.");
      if (output_idx >= model_queues->output_endpoint_names.size()) {
        model_queues->output_endpoint_names.resize(output_idx + 1UL);
      }
      model_queues->output_endpoint_names[output_idx] = queue_name;
      GELOGD("Save output queue_name:%s for node:%s, index:%zu.",
             queue_name.c_str(), node->GetName().c_str(), output_idx);
    }
  }
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuildForNetOutput(const NodePtr &node,
                                                 const std::map<NodePtr, std::map<int32_t, std::string>>
                                                 &paired_inputs) {
  GE_CHECK_NOTNULL(node);
  GELOGD("Begin to build model relation for netoutput node:%s.", node->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(CheckNetOutputNode(node));
  std::vector<std::string> unused;
  std::vector<std::string> &input_endpoint_names = CheckInnerNode(node) ? unused :
                                                model_relation_.root_model_endpoint_info.output_endpoint_names;
  GE_CHK_STATUS_RET_NOLOG(GetInputQueueNames(node, paired_inputs, input_endpoint_names));
  return SUCCESS;
}

Status ModelRelationBuilder::DoBuild(const ComputeGraph &root_graph) {
  // key1 : graph name = model instance name, key2 : endpoint name  val : endpoints
  const auto &all_endpoints_by_graph =
      root_graph.TryGetExtAttr<std::map<std::string, std::map<std::string, std::vector<Endpoint>>>>(
          ATTR_NAME_MODEL_EVENTS, {});
  model_relation_.root_model_endpoint_info.model_name = root_graph.GetName();
  std::map<NodePtr, std::map<int32_t, std::string>> paired_inputs;
  for (const auto &node : root_graph.GetDirectNode()) {
    GELOGD("root_graph:%s, node:%s", root_graph.GetName().c_str(), node->GetName().c_str());
    const auto &op_type = node->GetType();
    if (OpTypeUtils::IsDataNode(op_type)) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForData(node, paired_inputs, root_graph));
    } else if (op_type == PARTITIONEDCALL) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForPartitionedCall(node, paired_inputs));
    } else if (op_type == NETOUTPUT) {
      GE_CHK_STATUS_RET_NOLOG(DoBuildForNetOutput(node, paired_inputs));
    } else {
      GELOGW("Unexpected node in root graph, name = %s, type = %s",
             node->GetName().c_str(),
             op_type.c_str());
    }
  }
  return SUCCESS;
}

bool ModelRelationBuilder::GetFlowAttr(const AttrHolder *obj, const std::string &queue_name, int64_t &depth,
                                       std::string &enqueue_policy) {
  if (obj == nullptr) {
    return false;
  }
  if (AttrUtils::HasAttr(obj, ATTR_NAME_FLOW_ATTR)) {
    if (AttrUtils::GetInt(obj, ATTR_NAME_FLOW_ATTR_DEPTH, depth)) {
      GELOGD("[%s] Got queue depth = [%ld] from flow attr", queue_name.c_str(), depth);
    }
    if (AttrUtils::GetStr(obj, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, enqueue_policy)) {
      GELOGD("[%s] Got enqueue_policy = [%s] from flow attr", queue_name.c_str(), enqueue_policy.c_str());
    }
    return true;
  }
  return false;
}

void ModelRelationBuilder::GetFlowAttr(const std::string &queue_name, const GeTensorDesc &tensor_desc,
                                       const Node &node, int64_t &depth, std::string &enqueue_policy) {
  if (GetFlowAttr(&tensor_desc, queue_name, depth, enqueue_policy)) {
    GELOGD("[%s] Got flow attr from tensor desc flow attr", queue_name.c_str());
    return;
  }

  if (GetFlowAttr(node.GetOpDesc().get(), queue_name, depth, enqueue_policy)) {
    GELOGD("[%s] Got flow attr from op desc flow attr", queue_name.c_str());
    return;
  }

  const auto graph = node.GetOwnerComputeGraph();
  if (GetFlowAttr(graph.get(), queue_name, depth, enqueue_policy)) {
    GELOGD("[%s] Got flow attr from graph flow attr", queue_name.c_str(), enqueue_policy.c_str());
    return;
  }

  GELOGD("[%s] Can not get flow attr from tensor, node[%s] and graph[%s].", queue_name.c_str(), node.GetNamePtr(),
         (graph == nullptr) ? "NULL" : graph->GetName().c_str());
}

Status ModelRelationBuilder::CreateQueueDef(const GeTensorDesc &tensor_desc, const std::string &queue_name,
                                            const Node &node, bool is_dummy) {
  const std::map<std::string, Endpoint>::iterator &it = endpoints_.find(queue_name);
  if (it != endpoints_.end()) {
    GELOGE(PARAM_INVALID, "Duplicate queue name: %s", queue_name.c_str());
    return PARAM_INVALID;
  }

  int64_t depth = static_cast<int64_t>(kDefaultQueueDepth);
  std::string enqueue_policy = "FIFO";

  GetFlowAttr(queue_name, tensor_desc, node, depth, enqueue_policy);
  const EndpointType endpoint_type = is_dummy ? EndpointType::kDummyQueue : EndpointType::kQueue;
  Endpoint queue_def(queue_name, endpoint_type);
  (void)QueueNodeUtils(queue_def).SetDepth(depth).SetEnqueuePolicy(enqueue_policy).
    SetNodeAction(kQueueActionDefault);

  GE_CHK_BOOL_RET_STATUS(endpoints_.emplace(queue_name, queue_def).second,
                         PARAM_INVALID,
                         "Duplicate queue name: %s",
                         queue_name.c_str());
  model_relation_.endpoints.emplace_back(std::move(queue_def));
  return SUCCESS;
}

ModelRelation::ModelEndpointInfo *ModelRelationBuilder::GetOrCreateModelEndpointInfo(const std::string &model_name) {
  ModelRelation::ModelEndpointInfo *model_endpoint_info = nullptr;
  const auto &it = model_relation_.submodel_endpoint_infos.find(model_name);
  if (it != model_relation_.submodel_endpoint_infos.cend()) {
    model_endpoint_info = &it->second;
  }
  auto &ret = model_relation_.submodel_endpoint_infos[model_name];
  ret.model_name = model_name;
  model_endpoint_info = &ret;
  GELOGI("Create model endpoint, model name = %s.", model_name.c_str());
  return model_endpoint_info;
}

Status ModelRelationBuilder::GetOrCreateModelEndpointInfo(const OpDesc &op_desc,
                                                          ModelRelation::ModelEndpointInfo *&model_endpoint_info) {
  const auto &subgraph_names = op_desc.GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGE(PARAM_INVALID, "PartitionedCall [%s] does not have subgraph.", op_desc.GetName().c_str());
    return PARAM_INVALID;
  }

  const auto &model_name = subgraph_names[static_cast<uint64_t>(kSubgraphIndex)];
  model_endpoint_info = GetOrCreateModelEndpointInfo(model_name);
  return SUCCESS;
}

Status ModelRelationBuilder::GetInputQueueNames(const NodePtr &node,
                                                const map<NodePtr, std::map<int32_t, std::string>> &paired_inputs,
                                                std::vector<std::string> &input_queue_names) {
  GE_CHECK_NOTNULL(node);
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_LE(op_desc->GetInputsSize(), static_cast<uint64_t>(INT32_MAX));
  const int32_t input_size = static_cast<int32_t>(op_desc->GetInputsSize());
  if (input_size == 0) {
    GELOGD("Node [%s] does not have input.", op_desc->GetName().c_str());
    return SUCCESS;
  }

  const auto &it = paired_inputs.find(node);
  if (it == paired_inputs.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Node [%s] was not paired", op_desc->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "Node [%s] was not paired", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (int32_t i = 0; i < input_size; ++i) {
    const auto name_it = it->second.find(i);
    if (name_it == it->second.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Input[%d] of node [%s] was not paired", i, op_desc->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "Input[%d] of node [%s] was not paired", i, op_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }

    input_queue_names.emplace_back(name_it->second);
  }
  return SUCCESS;
}

bool ModelRelationBuilder::CheckInnerNode(const NodePtr &node) const {
  int32_t data_transfer_type = -1;
  (void)AttrUtils::GetInt(node->GetOpDesc(), "_data_transfer_type", data_transfer_type);
  return (data_transfer_type == kKernelInsideTransferType);
}

const Endpoint *ModelRelationReader::GetEndpoint(const std::string &queue_name) const {
  const auto &it = endpoints_.find(queue_name);
  if (it == endpoints_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "queue name not found. name = %s", queue_name.c_str());
    GELOGE(PARAM_INVALID, "queue name not found. name = %s", queue_name.c_str());
    return nullptr;
  }
  return it->second;
}

void ModelRelationReader::LogDebugString(const ModelRelation &model_relation) {
  GELOGD("endpoints.size: %zu.", model_relation.endpoints.size());
  GELOGD("root_model_endpoint_info.model_name: %s.",
         model_relation.root_model_endpoint_info.model_name.c_str());
  GELOGD("root_model_endpoint_info.input_endpoint_names.size: %zu.",
         model_relation.root_model_endpoint_info.input_endpoint_names.size());
  GELOGD("root_model_endpoint_info.output_endpoint_names.size: %zu.",
         model_relation.root_model_endpoint_info.output_endpoint_names.size());
}

Status ModelRelationReader::Initialize() {
  for (const auto &endpoint : model_relation_.endpoints) {
    (void)endpoints_.emplace(endpoint.GetName(), &endpoint);
  }
  GE_CHK_STATUS_RET_NOLOG(BatchGetEndpoints(model_relation_.root_model_endpoint_info.input_endpoint_names,
                                            input_endpoints_));
  GE_CHK_STATUS_RET_NOLOG(BatchGetEndpoints(model_relation_.root_model_endpoint_info.output_endpoint_names,
                                            output_endpoints_));
  return SUCCESS;
}

Status ModelRelationReader::BatchGetEndpoints(const vector<std::string> &endpoint_names,
                                              vector<const Endpoint *> &endpoints) const {
  for (const auto &endpoint_name : endpoint_names) {
    auto endpoint = GetEndpoint(endpoint_name);
    GE_CHECK_NOTNULL(endpoint);
    endpoints.emplace_back(endpoint);
  }
  return SUCCESS;
}

const ModelRelation::InvokedModelQueueInfo *ModelRelationReader::GetInvokedModelQueueInfo(
    const std::string &invoke_key) const {
  const auto find_ret = model_relation_.invoked_model_queue_infos.find(invoke_key);
  if (find_ret == model_relation_.invoked_model_queue_infos.cend()) {
    GELOGE(PARAM_INVALID, "Failed to find invoke model queue, invoke key=%s", invoke_key.c_str());
    return nullptr;
  }
  return &(find_ret->second);
}

ModelRelationReader::ModelRelationReader(const ModelRelation &model_relation) : model_relation_(model_relation) {
}

const ModelRelation::ModelEndpointInfo *ModelRelationReader::GetSubmodelQueueInfo(const string &model_name) const {
  const auto &it = model_relation_.submodel_endpoint_infos.find(model_name);
  if (it == model_relation_.submodel_endpoint_infos.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to get submodel queue info, name = %s", model_name.c_str());
    GELOGE(PARAM_INVALID, "Failed to get submodel queue info, name = %s", model_name.c_str());
    return nullptr;
  }
  return &it->second;
}
}  // namespace ge
