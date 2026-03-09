/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_converter.h"
#include <queue>
#include "register/node_converter_registry.h"
#include "common/checker.h"
#include "common/ge_inner_attrs.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "runtime/model_v2_executor.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/frame_selector.h"
#include "common/types.h"
#include "common/omg_util/omg_util.h"
#include "pass/offline_optimizer.h"
#include "static_compiled_graph_converter.h"
#include "placement/placed_lowering_result.h"
#include "core/builder/node_types.h"
#include "exe_graph_serializer.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/runtime/gert_const_types.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "node_priority_calculator.h"
#include "graph_builder/multi_stream/bg_event.h"
#include "graph/fast_graph/edge.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "lowering_utils.h"


namespace gert {
namespace {
constexpr char const *kUbOriginGraphAttrKey = "_original_fusion_graph";
constexpr const ge::char_t *kGlobalDataSplitRtStreams = "SplitRtStreams";
constexpr const ge::char_t *kGlobalDataRtNotifies = "ExecuteArgRtNotifies";

HyperStatus CollectLowerResultOfInDataNodes(const ge::NodePtr &node, int32_t inputs_placement, LowerInput &lower_input,
                                            OrderInputs &order_inputs) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  DataDependentInterpreter ddi(node->GetOpDesc(), lower_input.global_data->GetSpaceRegistriesV2());
  for (const auto &peer_node_and_anchor : node->GetInDataNodesAndAnchors()) {
    const auto &peer_node = peer_node_and_anchor.first;
    auto out_index = peer_node_and_anchor.second->GetIdx();

    auto *lower_result = peer_node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    if (lower_result == nullptr) {
      return HyperStatus::ErrorStatus(
          static_cast<const char *>(
              "Failed to construct LowerInput for node %s, because can not find the lower result on input node name:%s,"
              " type:%s, engine:%s."), node->GetNamePtr(), peer_node->GetNamePtr(), peer_node->GetTypePtr(),
              peer_node->GetOpDescBarePtr()->GetOpKernelLibName().c_str());
    }

    bool is_data_dependent = false;
    auto ret = ddi.IsDataDependent(static_cast<int32_t>(lower_input.input_shapes.size()), is_data_dependent);
    if (ret != ge::SUCCESS) {
      return HyperStatus::ErrorStatus(
          static_cast<const char *>("Failed to get data dependent flag for node %s, src node %s, input index %zu"),
          node->GetName().c_str(), peer_node->GetName().c_str(), lower_input.input_shapes.size());
    }
    bool is_tiling_dependent = false;
    if (!is_data_dependent) {
      auto tiling_ret = ddi.IsTilingInputDataDependent(static_cast<int32_t>(lower_input.input_shapes.size()),
          is_tiling_dependent);
      if (tiling_ret != ge::SUCCESS) {
        return HyperStatus::ErrorStatus(
            static_cast<const char *>("Failed to get tiling dependent flag for node %s, src node %s, input index %zu"),
            node->GetName().c_str(), peer_node->GetName().c_str(), lower_input.input_shapes.size());
      }
    }
    GELOGD("Node: %s, type: %s input: %zu data/tiling depend flag: %d/%d, peer node: %s, type: %s.", node->GetNamePtr(),
           node->GetTypePtr(), lower_input.input_shapes.size(), is_data_dependent, is_tiling_dependent,
           peer_node->GetNamePtr(), peer_node->GetTypePtr());
    is_data_dependent = (is_data_dependent || is_tiling_dependent);

    auto holder = bg::ValueHolder::SetScopedCurrentComputeNode(peer_node);
    const auto &result =
        lower_result->GetOutputResult(*lower_input.global_data, out_index,
                                      {inputs_placement, node->GetOpDescBarePtr()->GetStreamId()}, is_data_dependent);
    GE_ASSERT_NOTNULL(result);
    lower_input.input_shapes.emplace_back(result->shape);
    lower_input.input_addrs.emplace_back(result->address);
    for (const auto &ordered_input : result->order_holders) {
      if (order_inputs.ordered_inputs_set.insert(ordered_input).second) {
        order_inputs.ordered_inputs_list.emplace_back(ordered_input);
      }
    }
  }
  return HyperStatus::Success();
}
HyperStatus CollectOrderHoldersOfInControlNodes(const ge::NodePtr &node, OrderInputs &order_inputs) {
  const auto &in_control_anchor = node->GetInControlAnchor();
  if (in_control_anchor == nullptr) {
    return HyperStatus::ErrorStatus(
        static_cast<const char *>(
            "Failed to get control anchor from node %s[%s], because in control anchor is nullptr"),
        node->GetName().c_str(), node->GetType().c_str());
  }
  for (const auto peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchorsPtr()) {
    const auto peer_control_node = peer_out_control_anchor->GetOwnerNodeBarePtr();
    if (peer_control_node == nullptr) {
      return HyperStatus::ErrorStatus(
          static_cast<const char *>(
              "Failed to get control peer control node from node %s[%s], because in control node is nullptr"),
          node->GetName().c_str(), node->GetType().c_str());
    }
    const auto *const_lower_result =
        peer_control_node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    if (const_lower_result == nullptr) {
      return HyperStatus::ErrorStatus(
          static_cast<const char *>(
              "Failed to construct LowerInput for node %s, because can not find the lower result on input node %s."),
          node->GetName().c_str(), peer_control_node->GetName().c_str());
    }
    auto result = const_lower_result->GetResult();
    if (result == nullptr) {
      return HyperStatus::ErrorStatus(static_cast<const char *>("Failed to find lower result for node %s, src node %s"),
                                      node->GetName().c_str(), peer_control_node->GetName().c_str());
    }
    for (const auto &ordered_input : result->order_holders) {
      if (order_inputs.ordered_inputs_set.insert(ordered_input).second) {
        order_inputs.ordered_inputs_list.emplace_back(ordered_input);
      }
    }
  }
  return HyperStatus::Success();
}

const NodeConverterRegistry::ConverterRegisterData *GetNodeConvertData(const ge::NodePtr &node) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  std::string lowering_func;
  if (ge::AttrUtils::GetStr(op_desc, "_ge_attr_lowering_func", lowering_func)) {
    auto data = NodeConverterRegistry::GetInstance().FindRegisterData(lowering_func);
    if (data == nullptr) {
      return nullptr;
    }
    return data;
  }
  std::string type;
  if (ge::GetOriginalType(node, type) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get original type from %s(%s).", node->GetName().c_str(), node->GetType().c_str());
    return nullptr;
  }
  auto data = NodeConverterRegistry::GetInstance().FindRegisterData(type);
  if (data != nullptr) {
    return data;
  }

  data = NodeConverterRegistry::GetInstance().FindRegisterData(op_desc->GetOpKernelLibName());
  if (data != nullptr) {
    return data;
  }
  GELOGE(ge::FAILED, "Failed to find the converter for node %s type %s", node->GetName().c_str(),
         node->GetType().c_str());
  return nullptr;
}

HyperStatus AddDependencyForOrderedHolders(const std::vector<bg::ValueHolderPtr> &ordered_inputs,
                                           const LowerResult &node_lower_result) {
  if ((ordered_inputs.empty()) || (node_lower_result.order_holders.empty())) {
    return HyperStatus::Success();
  }

  for (const auto &src : ordered_inputs) {
    for (const auto &dst : node_lower_result.order_holders) {
      HyperStatus ret;
      bool is_equal = bg::ValueHolderUtils::IsNodeEqual(src, dst);
      if (is_equal && (strcmp(bg::ValueHolderUtils::GetNodeTypeBarePtr(src),
                              GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0)) {
        const auto src_on_init = HolderOnInit(src);
        const auto dst_on_init = HolderOnInit(dst);
        ret = bg::ValueHolder::AddDependency(src_on_init, dst_on_init);
      } else {
        ret = bg::ValueHolder::AddDependency(src, dst);
      }
      if (!ret.IsSuccess()) {
        GELOGW("add dependency for order holders not success, reason:%s", ret.GetErrorMessage());
      }
    }
  }

  return HyperStatus::Success();
}
/*
 * after lowering node call this func to lowering send event of this node
 * todo 1.order holder need to support pre and post
 *      2. value holder change to devmem valueholder
 */
HyperStatus LoweringEventSync(const ge::NodePtr &node, LowerResult &lower_result, LoweringGlobalData &global_data) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t logic_stream_id = op_desc->GetStreamId();

  // lowering waitEvents
  std::vector<int64_t> recive_ids;
  bg::ValueHolderPtr wait_event_holder = nullptr;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_RECV_EVENT_IDS, recive_ids);
  if (!recive_ids.empty()) {
    wait_event_holder = bg::WaitEvents(logic_stream_id, recive_ids, global_data);
    if (wait_event_holder == nullptr) {
      return HyperStatus::ErrorStatus("Failed to lowering wait event of node %s.", node->GetNamePtr());
    }
  }

  // lowering sendEvents
  std::vector<int64_t> send_ids;
  bg::ValueHolderPtr send_event_holder = nullptr;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_SEND_EVENT_IDS, send_ids);
  if (!send_ids.empty()) {
    send_event_holder = bg::SendEvents(logic_stream_id, send_ids, global_data);
    if (send_event_holder == nullptr) {
      return HyperStatus::ErrorStatus("Failed to lowering send event of node %s.", node->GetNamePtr());
    }
  }

  // add dependency from origin order holder to new order holder
  for (const auto &holder : lower_result.order_holders) {
    if (wait_event_holder != nullptr) {
      auto ret = bg::ValueHolder::AddDependency(wait_event_holder, holder);
      if (!ret.IsSuccess()) {
        GELOGW("add dependency for order holders from %s to %s not success, reason:%s",
               bg::ValueHolderUtils::GetNodeNameBarePtr(wait_event_holder),
               bg::ValueHolderUtils::GetNodeNameBarePtr(holder), ret.GetErrorMessage());
        // pre node may lowering inside init graph, return init node output as order holder
      }
    }
    if (send_event_holder != nullptr) {
      auto ret = bg::ValueHolder::AddDependency(holder, send_event_holder);
      if (!ret.IsSuccess()) {
        GELOGW("add dependency for order holders from %s to %s not success, reason:%s",
               bg::ValueHolderUtils::GetNodeNameBarePtr(holder),
               bg::ValueHolderUtils::GetNodeNameBarePtr(send_event_holder), ret.GetErrorMessage());
        // pre node may lowering inside init graph, return init node output as order holder
      }
    }
  }
  // make send/wait event as new order holder, replace all other holder
  if (wait_event_holder != nullptr) {
    lower_result.order_holders.emplace_back(wait_event_holder);
  }
  if (send_event_holder != nullptr) {
    lower_result.order_holders.emplace_back(send_event_holder);
  }
  return HyperStatus::Success();
}

HyperStatus LoweringAccessMemCrossStream(const ge::NodePtr &node, LowerInput &inputs) {
  if (IsTypeNetOutput(node->GetTypePtr())) {
    const auto owner_graph = node->GetOwnerComputeGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    if (!owner_graph->GetGraphUnknownFlag()) {
      // return if netoutput owner graph is static sub graph
      return HyperStatus::Success();
    }
    if ((owner_graph->GetParentNode() != nullptr) && IsWhileType(owner_graph->GetParentNode()->GetTypePtr())) {
      // return if netoutput owner graph is while body, access mem cross in while converter
      return HyperStatus::Success();
    }
  }
  return bg::LoweringAccessMemCrossStream(node, inputs.input_addrs);
}

HyperStatus LoweringNode(const ge::NodePtr &node, LowerInput &input,
                         const std::vector<bg::ValueHolderPtr> &ordered_inputs,
                         const NodeConverterRegistry::NodeConverter &func) {
  bg::ValueHolder::SetCurrentComputeNode(node);
  auto ret = LoweringAccessMemCrossStream(node, input);
  if (!ret.IsSuccess()) {
    return HyperStatus::ErrorStatus("Failed to lowering access_mem_cross_stream for node %s.", node->GetNamePtr());
  }

  LowerInputInfo lower_input_info;
  lower_input_info.input_shapes = input.input_shapes;
  lower_input_info.input_addrs = input.input_addrs;
  (void)node->GetOpDescBarePtr()->SetExtAttr(kLoweringInputInfo, lower_input_info);

  auto lowering_result = func(node, input);
  if (!lowering_result.result.IsSuccess()) {
    return lowering_result.result;
  }

  ret = LoweringEventSync(node, lowering_result, *input.global_data);
  if (!ret.IsSuccess()) {
    return HyperStatus::ErrorStatus("Failed to lowering event sync for node %s.", node->GetNamePtr());
  }

  if (lowering_result.order_holders.empty()) {
    lowering_result.order_holders = ordered_inputs;
  } else {
    ret = AddDependencyForOrderedHolders(ordered_inputs, lowering_result);
    if (!ret.IsSuccess()) {
      return ret;
    }
  }

  if (!node->GetOpDescBarePtr()->SetExtAttr(kLoweringResult, PlacedLoweringResult(node, std::move(lowering_result)))) {
    return HyperStatus::ErrorStatus("Failed to add lowering result to node %s", node->GetName().c_str());
  }
  return HyperStatus::Success();
}

ge::graphStatus AddContainerNode(SubExeGraphType sub_exe_graph_type) {
  auto node_type = GetSubExeGraphTypeStr(sub_exe_graph_type);
  GE_ASSERT_NOTNULL(node_type);

  auto node_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(node_type, {});
  GE_ASSERT_NOTNULL(node_holder);
  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame(node_holder, node_type));

  return ge::GRAPH_SUCCESS;
}

// 在根图上创建init node，输出个数为const data的输出个数
// 同时把根图上const data的输出放到global data里
ge::graphStatus AddInitContainerNode(LoweringGlobalData &global_data) {
  auto node_type = GetSubExeGraphTypeStr(kInitExeGraph);
  GE_ASSERT_NOTNULL(node_type);
  size_t const_data_num = static_cast<size_t>(ConstDataType::kTypeEnd);
  auto node_holder = bg::ValueHolder::CreateDataOutput(node_type, {}, const_data_num);
  GE_ASSERT_TRUE(node_holder.size() == const_data_num);

  for (size_t i = 0U; i < const_data_num; ++i) {
    const auto &const_data_name = GetConstDataTypeStr(static_cast<ConstDataType>(i));
    GE_ASSERT_TRUE(!const_data_name.empty());
    global_data.SetUniqueValueHolder(const_data_name, node_holder[i]);
  }
  GE_ASSERT_TRUE(!node_holder.empty());
  GE_ASSERT_NOTNULL(node_holder[0U]);
  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame(node_holder[0U], node_type));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateExeGraph(const std::vector<ge::FastNode *> &main_graph_nodes) {
  // check main graph can not hold ConstData
  for (const auto node : main_graph_nodes) {
    GE_ASSERT_NOTNULL(node);
    if (IsConstFeedType(node->GetTypePtr())) {
      GELOGE(ge::INTERNAL_ERROR,
             "Main graph can not hold ConstData %s. Because ConstData only can held by init graph. Please check "
             "lowering logic.",
             node->GetName().c_str());
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool IsNetOutput(const ge::NodePtr &node) {
  return node->GetType() == "NetOutput";
}

ge::FastNode *AddNoOpNodeToExeGraph(const ge::ExecuteGraphPtr &exe_graph, const std::string &prefix) {
  auto op_desc = ge::MakeShared<ge::OpDesc>(prefix, ge::NOOP);
  GE_ASSERT_NOTNULL(op_desc);
  GELOGI("Add Noop %s to exe_graph %s.", op_desc->GetNamePtr(), exe_graph->GetName().c_str());
  return exe_graph->AddNode(op_desc);
}

ge::graphStatus ExpandPartitionedCallToParentGraph(const ge::ExecuteGraphPtr &exe_graph, ge::FastNode *partitioned_call,
                                                   ge::FastNode *&pre_handle, ge::FastNode *&post_handle) {
  // create a control noop for each layer
  pre_handle = AddNoOpNodeToExeGraph(exe_graph, partitioned_call->GetName() + "_pre_noop");
  GE_ASSERT_NOTNULL(pre_handle);
  post_handle = AddNoOpNodeToExeGraph(exe_graph, partitioned_call->GetName() + "_post_noop");
  GE_ASSERT_NOTNULL(post_handle);

  auto pcall_subgraph = ge::FastNodeUtils::GetSubgraphFromNode(partitioned_call, 0U);
  GE_ASSERT_NOTNULL(pcall_subgraph);

  // connect pnode input into node in subgraph
  std::vector<ge::FastNode *> input_nodes;
  ge::FastNode *inner_netoutput = nullptr;
  for (const auto node : pcall_subgraph->GetDirectNode()) {
    if (IsInnerDataType(node->GetTypePtr())) {
      input_nodes.emplace_back(node);
    }
    if (IsTypeInnerNetOutput(node->GetTypePtr())) {
      inner_netoutput = node;
    }
  }
  GE_ASSERT_NOTNULL(inner_netoutput);

  for (const auto inner_data : input_nodes) {
    int32_t parent_node_index = -1;
    GE_ASSERT_TRUE(ge::AttrUtils::GetInt(inner_data->GetOpDescBarePtr(), "index", parent_node_index),
                   "Failed to find index attr of inner data %s.", inner_data->GetNamePtr());
    const auto in_data_edge = partitioned_call->GetInDataEdgeByIndex(parent_node_index);
    GE_ASSERT_NOTNULL(in_data_edge);
    const auto src_node = in_data_edge->src;
    GE_ASSERT_NOTNULL(src_node);
    const auto src_output = in_data_edge->src_output;
    GE_ASSERT_GRAPH_SUCCESS(exe_graph->RemoveEdge(in_data_edge), "Remove in data edge %s:%d failed.",
                            partitioned_call->GetNamePtr(), parent_node_index);
    for (const auto &out_data_edges : inner_data->GetAllOutDataEdgesRef()) {
      for (const auto out_data_edge : out_data_edges) {
        if (out_data_edge == nullptr) {
          continue;
        }
        const auto dst_node = out_data_edge->dst;
        GE_ASSERT_NOTNULL(dst_node);
        const auto dst_input = out_data_edge->dst_input;
        GE_ASSERT_GRAPH_SUCCESS(pcall_subgraph->RemoveEdge(out_data_edge));
        GE_ASSERT_NOTNULL(exe_graph->AddNode(dst_node));
        GE_ASSERT_GRAPH_SUCCESS(dst_node->GetExtendInfo()->SetOwnerGraph(exe_graph.get(), dst_node));
        GE_ASSERT_NOTNULL(exe_graph->AddEdge(src_node, src_output, dst_node, dst_input),
                          "Add edge %s:%d->%s:%d failed.", src_node->GetNamePtr(), src_output, dst_node->GetNamePtr(),
                          dst_input);
        if (!dst_node->IsDirectlyControlledByNode(pre_handle)) {
          GE_ASSERT_NOTNULL(exe_graph->AddEdge(pre_handle, ge::kControlEdgeIndex, dst_node, ge::kControlEdgeIndex),
                            "Add control edge %s->%s failed.", pre_handle->GetNamePtr(), dst_node->GetNamePtr());
        }
      }
    }
    GE_ASSERT_GRAPH_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(pcall_subgraph, inner_data));
  }

  for (const auto in_control_edge : inner_netoutput->GetAllInControlEdgesRef()) {
    if (in_control_edge != nullptr) {
      const auto src_control_node = in_control_edge->src;
      GE_ASSERT_NOTNULL(src_control_node);
      GE_ASSERT_GRAPH_SUCCESS(pcall_subgraph->RemoveEdge(in_control_edge));
      GE_ASSERT_NOTNULL(exe_graph->AddNode(src_control_node));
      GE_ASSERT_GRAPH_SUCCESS(src_control_node->GetExtendInfo()->SetOwnerGraph(exe_graph.get(), src_control_node));
      GE_ASSERT_NOTNULL(
          exe_graph->AddEdge(src_control_node, ge::kControlEdgeIndex, post_handle, ge::kControlEdgeIndex));
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(pcall_subgraph, inner_netoutput));

  for (const auto node : pcall_subgraph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(exe_graph->AddNode(node));
    GE_ASSERT_GRAPH_SUCCESS(node->GetExtendInfo()->SetOwnerGraph(exe_graph.get(), node));
  }
  // PartitionedCall 子图展开，需要同时更新子图中边的 owner，确保子图节点和边归属于同一张图
  for (const auto node : exe_graph->GetDirectNode()) {
    for (const auto edge : node->GetAllInDataEdgesRef()) {
      if ((edge != nullptr) && !exe_graph->CheckEdgeIsInGraph(edge)) {
        GE_ASSERT_GRAPH_SUCCESS(exe_graph->MoveEdgeToGraph(edge));
      }
    }
    for (const auto edge : node->GetAllInControlEdgesRef()) {
      if ((edge != nullptr) && !exe_graph->CheckEdgeIsInGraph(edge)) {
        GE_ASSERT_GRAPH_SUCCESS(exe_graph->MoveEdgeToGraph(edge));
      }
    }
  }

  GE_ASSERT_GRAPH_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(exe_graph.get(), partitioned_call));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddDependencyFromOutputToLastExeNode(const std::vector<bg::ValueHolderPtr> &last_exe_nodes,
                                                     const ge::ExecuteGraphPtr &exe_graph) {
  const auto net_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output);
  for (const auto &last_exe_node : last_exe_nodes) {
    GE_ASSERT_NOTNULL(last_exe_node);
    const auto fast_node = last_exe_node->GetFastNode();
    GE_ASSERT_NOTNULL(fast_node);
    for (const auto body_output : net_output->GetAllInNodes()) {
      GE_ASSERT_NOTNULL(body_output);
      GE_ASSERT_NOTNULL(exe_graph->AddEdge(body_output, ge::kControlEdgeIndex, fast_node, ge::kControlEdgeIndex));
    }
    GE_ASSERT_NOTNULL(exe_graph->AddEdge(fast_node, ge::kControlEdgeIndex, net_output, ge::kControlEdgeIndex));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpandLastSyncExeNodesToMainGraph(const ge::ExecuteGraphPtr &exe_graph) {
  auto stage_ids_to_last_pcall =
      exe_graph->GetExtAttr<std::vector<bg::ValueHolderPtr>>(bg::kStageIdsToLastPartitionedCall);
  if (stage_ids_to_last_pcall == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  GE_ASSERT_TRUE(stage_ids_to_last_pcall->size() == static_cast<size_t>(bg::OnMainRootLastExecStage::kStageSize));
  auto net_output = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output);

  for (size_t i = 0U; i < stage_ids_to_last_pcall->size(); ++i) {
    auto last_sync_pcall = stage_ids_to_last_pcall->at(i);
    if (last_sync_pcall == nullptr) {
      GELOGD("No last sync stage partition_call, stage id is %u.", i);
      continue;
    }
    const auto pcall_node = last_sync_pcall->GetFastNode();
    GE_ASSERT_NOTNULL(pcall_node);
    ge::FastNode *pre_noop = nullptr;
    ge::FastNode *post_noop = nullptr;
    GE_ASSERT_GRAPH_SUCCESS(ExpandPartitionedCallToParentGraph(exe_graph, pcall_node, pre_noop, post_noop));
    GE_ASSERT_NOTNULL(pre_noop);
    GE_ASSERT_NOTNULL(post_noop);

    // link control from all input_node of netoutput to pre noop
    for (const auto &body_output : net_output->GetAllInNodes()) {
      GE_ASSERT_NOTNULL(body_output);
      GE_ASSERT_NOTNULL(exe_graph->AddEdge(body_output, ge::kControlEdgeIndex, pre_noop, ge::kControlEdgeIndex));
    }
    // link control from post noop to netoutput
    GE_ASSERT_NOTNULL(exe_graph->AddEdge(post_noop, ge::kControlEdgeIndex, net_output, ge::kControlEdgeIndex));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpandFirstExeNodesToMainGraph(const ge::ExecuteGraphPtr &exe_graph) {
  // expand FirstEventSync partitioned call
  auto stage_ids_to_first_pcall =
      exe_graph->GetExtAttr<std::vector<bg::ValueHolderPtr>>(bg::kStageIdsToFirstPartitionedCall);
  if (stage_ids_to_first_pcall == nullptr) {
    return ge::GRAPH_SUCCESS;
  }

  // first event sync should execute before all kernel relies on l2 allocators, to avoid cause loop in graph
  auto split_rt_streams = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), kGlobalDataSplitRtStreams);
  std::vector<ge::FastNode *> all_select_l2_allocators;
  for (const auto &out_data_edges : split_rt_streams->GetAllOutDataEdgesRef()) {
    for (const auto edge : out_data_edges) {
      if ((edge != nullptr) && (edge->dst != nullptr) && (edge->dst->GetType() == "SelectL2Allocator")) {
        all_select_l2_allocators.emplace_back(edge->dst);
      }
    }
  }
  std::vector<ge::FastNode *> out_nodes_of_l2_allocator;
  for (const auto &select_l2_allocator : all_select_l2_allocators) {
    for (const auto &out_data_edges : select_l2_allocator->GetAllOutDataEdgesRef()) {
      for (const auto edge : out_data_edges) {
        if ((edge != nullptr) && (edge->dst != nullptr) && edge->dst->GetType() != ge::PARTITIONEDCALL) {
          out_nodes_of_l2_allocator.emplace_back(edge->dst);
        }
      }
    }
  }

  GE_ASSERT_TRUE(stage_ids_to_first_pcall->size() == static_cast<size_t>(bg::OnMainRootFirstExecStage::kStageSize));
  auto first_sync_pcall =
      stage_ids_to_first_pcall->at(static_cast<size_t>(bg::OnMainRootFirstExecStage::kFirstEventSyncStage));
  GE_ASSERT_NOTNULL(first_sync_pcall);
  auto pcall_node = first_sync_pcall->GetFastNode();
  GE_ASSERT_NOTNULL(pcall_node);
  ge::FastNode *pre_noop = nullptr;
  ge::FastNode *post_noop = nullptr;
  GE_ASSERT_GRAPH_SUCCESS(ExpandPartitionedCallToParentGraph(exe_graph, pcall_node, pre_noop, post_noop));
  GE_ASSERT_NOTNULL(pre_noop);
  GE_ASSERT_NOTNULL(post_noop);

  for (const auto out_node : out_nodes_of_l2_allocator) {
    GE_ASSERT_NOTNULL(exe_graph->AddEdge(post_noop, ge::kControlEdgeIndex, out_node, ge::kControlEdgeIndex));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LowerConstDataNode(std::vector<bg::ValueHolderPtr> &const_data_outputs) {
  size_t const_data_num = static_cast<size_t>(ConstDataType::kTypeEnd);
  for (size_t i = 0U; i < const_data_num; ++i) {
    auto const_data_holder = bg::ValueHolder::CreateConstData(static_cast<int64_t>(i));
    GE_ASSERT_NOTNULL(const_data_holder);
    const_data_outputs.emplace_back(const_data_holder);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoweringExecuteArgFeeds(LoweringGlobalData &global_data, ExecuteGraphType exe_graph_type,
                                        int64_t stream_num) {
  // init graph Feed(-1) means external stream
  // main graph Feed(-1) means external stream
  // main graph Feed(-3) means all streams
  if (exe_graph_type == ExecuteGraphType::kInit) {
    stream_num = 1;
  }
  auto rt_streams = global_data.LoweringAndSplitRtStreams(stream_num);
  GE_ASSERT_EQ(rt_streams.size(), static_cast<size_t>(stream_num));

  global_data.SetExternalAllocator(
      bg::ValueHolder::CreateFeed(static_cast<int64_t>(ExecuteArgIndex::kExternalAllocator)), exe_graph_type);
  auto rt_events = bg::ValueHolder::CreateFeed(static_cast<int64_t>(ExecuteArgIndex::kRtEvents));

  if (exe_graph_type == ExecuteGraphType::kMain) {
    global_data.SetUniqueValueHolder(bg::kGlobalDataRtEvents, rt_events);
  }

  const auto notify_holder = bg::ValueHolder::CreateFeed(static_cast<int64_t>(ExecuteArgIndex::kNotifies));
  GE_ASSERT_NOTNULL(notify_holder);
  if (exe_graph_type == ExecuteGraphType::kMain) {
    global_data.SetUniqueValueHolder(kGlobalDataRtNotifies, notify_holder);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateInitNode(LoweringGlobalData &global_data) {
  // create init container node
  GE_ASSERT_SUCCESS(AddInitContainerNode(global_data));

  // lower const data
  std::vector<bg::ValueHolderPtr> const_data_outputs;
  GE_ASSERT_SUCCESS(LowerConstDataNode(const_data_outputs));

  // lower stream & allocator.
  // init graph stream num is 1
  LoweringExecuteArgFeeds(global_data, ExecuteGraphType::kInit, 1);
  GE_ASSERT_NOTNULL(bg::ValueHolder::PopGraphFrame(const_data_outputs, {}));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateDeInitNode() {
  GE_ASSERT_SUCCESS(AddContainerNode(kDeInitExeGraph));
  GE_ASSERT_NOTNULL(bg::ValueHolder::PopGraphFrame());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoweringRtNotifies(const ModelDescHolder &model_desc_holder, LoweringGlobalData &global_data) {
  const int64_t notify_num = model_desc_holder.GetModelDesc().GetReusableNotifyNum();
  if (notify_num > 0) {
    const auto notify_num_holder = bg::ValueHolder::CreateConst(&notify_num, sizeof(notify_num));
    GE_ASSERT_NOTNULL(notify_num_holder);

    const auto notify_holder = global_data.GetUniqueValueHolder(kGlobalDataRtNotifies);
    GE_ASSERT_NOTNULL(notify_holder);
    const auto notifies =
        bg::ValueHolder::CreateDataOutput("CreateNotifies", {notify_holder, notify_num_holder}, notify_num);
    GE_ASSERT_EQ(notifies.size(), static_cast<size_t>(notify_num));
    global_data.SetRtNotifies(notifies);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoweringStreamResources(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data,
                                        ModelDescHolder &model_desc_holder) {
  // prepare stream num in init for l2 allocator
  int64_t stream_num = model_desc_holder.GetModelDesc().GetReusableStreamNum();
  auto init_out = bg::FrameSelector::OnInitRoot([&stream_num, &global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });

  std::vector<std::vector<bg::EventInfo>> stage_2_events(static_cast<size_t>(bg::SyncEventStage::kStageEnd));
  (void)bg::CollectAndCreateGertEvents(graph, model_desc_holder.GetModelDesc(), global_data, stage_2_events);
  int64_t event_num = model_desc_holder.GetModelDesc().GetReusableEventNum();
  auto &first_sync_events = stage_2_events[static_cast<size_t>(bg::SyncEventStage::kFirstSyncStage)];
  auto &last_sync_events = stage_2_events[static_cast<size_t>(bg::SyncEventStage::kLastSyncStage)];
  auto &last_resource_clean_events = stage_2_events[static_cast<size_t>(bg::SyncEventStage::kLastResourceCleanStage)];
  bg::LoweringFirstSyncEvents(first_sync_events, event_num, global_data);
  bg::LoweringLastSyncEvents(last_sync_events, event_num + first_sync_events.size(), global_data);
  bg::LoweringLastResourceCleanEvents(last_resource_clean_events,
                                      event_num + first_sync_events.size() + last_sync_events.size(), global_data);
  model_desc_holder.MutableModelDesc().SetReusableEventNum(event_num + first_sync_events.size() +
                                                           last_sync_events.size() + last_resource_clean_events.size());

  GE_ASSERT_GRAPH_SUCCESS(LoweringRtNotifies(model_desc_holder, global_data));
  return ge::GRAPH_SUCCESS;
}

ge::Status GetUbOriginalGraphNodeIds(const ge::ComputeGraphPtr &graph,
                                     std::map<std::string, int64_t> &ub_graph_node_2_parent_node_id) {
  for (const auto node : graph->GetAllNodesPtr()) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    ge::ComputeGraphPtr ub_graph = nullptr;
    (void)ge::AttrUtils::GetGraph(op_desc, kUbOriginGraphAttrKey, ub_graph);
    if (ub_graph == nullptr) {
      continue;
    }
    const int64_t node_id = op_desc->GetId();
    for (const auto direct_node : ub_graph->GetDirectNodePtr()) {
      ub_graph_node_2_parent_node_id.emplace(direct_node->GetName(), node_id);
    }
  }
  return ge::SUCCESS;
}

bool CheckUbGraphNodeAndSetIds(const std::map<std::string, int64_t> &ub_graph_node_2_parent_node_id,
                               const ge::Node *const target_node) {
  const auto iter = ub_graph_node_2_parent_node_id.find(target_node->GetName());
  if (iter == ub_graph_node_2_parent_node_id.end()) {
    return false;
  }
  const auto op_desc = target_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetId(iter->second);
  return true;
}

ge::graphStatus CheckMainFrameComputeNode(const ge::ComputeGraphPtr &graph, const bg::GraphFrame &frame) {
  std::map<std::string, int64_t> ub_graph_node_2_parent_node_id;
  GE_ASSERT_SUCCESS(GetUbOriginalGraphNodeIds(graph, ub_graph_node_2_parent_node_id));
  const auto &all_nodes = graph->GetAllNodesPtr();
  for (const auto &compute_node : frame.GetIndexesToNode()) {
    // main frame只能有编译图里的计算节点、通过公共接口创建的计算节点和ub graph的计算节点
    if ((!LoweringUtils::IsEngineTaskNode(compute_node)) &&
        (!CheckUbGraphNodeAndSetIds(ub_graph_node_2_parent_node_id, compute_node.get())) &&
        (std::find(all_nodes.cbegin(), all_nodes.cend(), compute_node.get()) == all_nodes.cend())) {
      GELOGE(ge::GRAPH_PARAM_INVALID, "Compute node [%s] from main frame is invalid", compute_node->GetName().c_str());
      return ge::GRAPH_PARAM_INVALID;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateMainNode(const ge::ComputeGraphPtr &graph, ModelDescHolder &model_desc_holder,
                               LoweringGlobalData &global_data, ge::ExecuteGraphPtr &main_graph) {
  AddContainerNode(kMainExeGraph);
  int64_t split_stream_num =
      model_desc_holder.GetModelDesc().GetReusableStreamNum() + model_desc_holder.GetModelDesc().GetAttachedStreamNum();
  GE_ASSERT_GRAPH_SUCCESS(LoweringExecuteArgFeeds(global_data, ExecuteGraphType::kMain, split_stream_num));
  GE_ASSERT_GRAPH_SUCCESS(LoweringStreamResources(graph, global_data, model_desc_holder));

  auto graph_result = LoweringComputeGraph(graph, global_data);
  GE_ASSERT_NOTNULL(graph_result);
  GE_ASSERT_HYPER_SUCCESS(graph_result->result);
  auto targets = graph_result->order_holders;
  auto &outputs = graph_result->out_shapes;
  auto last_exe_nodes = bg::ValueHolder::GetLastExecNodes();
  auto frame = bg::ValueHolder::PopGraphFrame(outputs, targets, ge::NETOUTPUT);
  GE_ASSERT_NOTNULL(frame);
  main_graph = frame->GetExecuteGraph();

  // todo ExpandPartitionCall is temp solution
  // Final solution: to support dynamic partitioned call execution.
  GE_ASSERT_SUCCESS(AddDependencyFromOutputToLastExeNode(last_exe_nodes, main_graph));
  // 为了在DUMP图时识别单算子场景
  bool is_single_op = ge::GraphUtils::IsSingleOpScene(graph);
  if (is_single_op) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetBool(main_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op));
  }

  ge::DumpGraph(frame->GetExecuteGraph().get(), "Before_MultiStream_LoweringFirstLastEventSync");
  GE_ASSERT_SUCCESS(ExpandFirstExeNodesToMainGraph(main_graph));
  GE_ASSERT_SUCCESS(ExpandLastSyncExeNodesToMainGraph(main_graph));
  GE_ASSERT_SUCCESS(CheckMainFrameComputeNode(graph, *frame.get()));
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus GraphConverter::AppendGraphLevelData(const bg::GraphFrame &frame,
                                                     const ge::ComputeGraphPtr &compute_graph,
                                                     ge::ExecuteGraph *const execute_graph,
                                                     const std::vector<ge::FastNode *> &root_graph_nodes) const {
  if (!frame.IsRootFrame()) {
    GELOGE(ge::PARAM_INVALID, "Failed to append graph level data, current exe_graph is not the root graph");
    return ge::GRAPH_FAILED;
  }
  GE_ASSERT_SUCCESS(ExeGraphSerializer(frame)
                        .SetComputeGraph(compute_graph)
                        .SetExecuteGraph(execute_graph)
                        .SetModelDescHolder(model_desc_holder_)
                        .SaveSerialization(root_graph_nodes));
  return ge::GRAPH_SUCCESS;
}

HyperStatus ConstructInputs(const ge::NodePtr &node, int32_t inputs_placement, LowerInput &lower_input,
                            OrderInputs &order_inputs) {
  auto ret = CollectLowerResultOfInDataNodes(node, inputs_placement, lower_input, order_inputs);
  if (!ret.IsSuccess()) {
    return ret;
  }
  ret = CollectOrderHoldersOfInControlNodes(node, order_inputs);
  if (!ret.IsSuccess()) {
    return ret;
  }
  return HyperStatus::Success();
}

const LowerResult *LoweringComputeGraph(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data) {
  GE_ASSERT_NOTNULL(graph);
  if (IsNeedLoweringAsStaticCompiledGraph(graph, global_data)) {
    return LoweringStaticCompiledComputeGraph(graph, global_data);
  }
  const LowerResult *graph_result = nullptr;
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    auto lowering_reg_data = GetNodeConvertData(node);
    GE_ASSERT_TRUE((lowering_reg_data != nullptr) && (lowering_reg_data->converter != nullptr),
                   "Failed to get lowering func for node name[%s], node type[%s], engine[%s]",
                   node->GetNamePtr(), node->GetTypePtr(), node->GetOpDescBarePtr()->GetOpKernelLibName().c_str());

    LowerInput inputs{{}, {}, &global_data};
    OrderInputs order_inputs;
    auto ret = ConstructInputs(node, lowering_reg_data->require_placement, inputs, order_inputs);
    if (!ret.IsSuccess()) {
      GELOGE(ge::FAILED, "Failed to construct inputs for node %s, reason %s", node->GetName().c_str(),
             ret.GetErrorMessage());
      return nullptr;
    }
    ret = LoweringNode(node, inputs, order_inputs.ordered_inputs_list, lowering_reg_data->converter);
    if (!ret.IsSuccess()) {
      GELOGE(ge::FAILED, "Failed to lowering node %s, reason %s", node->GetName().c_str(), ret.GetErrorMessage());
      return nullptr;
    }
    if (IsNetOutput(node)) {
      const auto op_desc = node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      graph_result = op_desc->GetExtAttr<PlacedLoweringResult>(kLoweringResult)->GetResult();
    }
  }
  return graph_result;
}

const LowerResult *LoweringNode(const ge::NodePtr &node, LowerInput &inputs,
                                const std::vector<bg::ValueHolderPtr> &ordered_inputs) {
  GE_ASSERT_NOTNULL(node);
  auto lowering_reg_data = GetNodeConvertData(node);
  GE_ASSERT_NOTNULL(lowering_reg_data);
  GE_ASSERT_NOTNULL(lowering_reg_data->converter);
  GE_ASSERT_HYPER_SUCCESS(LoweringNode(node, inputs, ordered_inputs, lowering_reg_data->converter));
  return node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult)->GetResult();
}

const LowerResult *ConvertComputeSubgraphToExecuteGraph(const ge::ComputeGraphPtr &graph,
                                                        LoweringGlobalData &global_data, int32_t start_index,
                                                        const std::vector<int32_t> &parent_inputs_placement,
                                                        const std::vector<int32_t> &parent_outputs_placement) {
  static_cast<void>(parent_outputs_placement);
  for (const auto node : graph->GetDirectNodePtr()) {
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const auto op_type = node->GetTypePtr();
    if (IsFeedType(op_type) && !op_desc->HasAttr("_ge_attr_lowering_func")) {
      (void)ge::AttrUtils::SetStr(op_desc, "_ge_attr_lowering_func", ge::kSubgraphInput);
      (void)ge::AttrUtils::SetInt(op_desc, "_inner_data_start_index", start_index);
      if (!parent_inputs_placement.empty()) {
        int32_t parent_input_index = -1;
        GE_ASSERT(ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_input_index),
                  "Failed get attr '%s' from compute data node %s", ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                  node->GetNamePtr());
        GE_ASSERT(parent_input_index >= 0);
        GE_ASSERT(static_cast<size_t>(parent_input_index) < parent_inputs_placement.size());
        (void)ge::AttrUtils::SetInt(op_desc, "_placement", parent_inputs_placement[parent_input_index]);
      }
      continue;
    }
    if (IsOutputType(op_type) && !op_desc->HasAttr("_ge_attr_lowering_func")) {
      (void)ge::AttrUtils::SetStr(op_desc, "_ge_attr_lowering_func", ge::kSubgraphOutput);
      continue;
    }
  }

  return LoweringComputeGraph(graph, global_data);
}
ge::ExecuteGraphPtr GraphConverter::ConvertComputeGraphToExecuteGraph(const ge::ComputeGraphPtr &graph,
                                                                      const LoweringOption &optimize_option,
                                                                      LoweringGlobalData &global_data) const {
  GE_MAKE_GUARD(clear_graph_frame, []() {
    bg::ValueHolder::ClearGraphFrameResource();
  });
  GE_TIMESTAMP_START(ConvertComputeGraphToExecuteGraphAll);
  GE_ASSERT_NOTNULL(graph);
  GE_DUMP(graph, "ComputeGraphBeforeLowering");
  GE_ASSERT_NOTNULL(bg::ValueHolder::PushGraphFrame());

  GE_ASSERT_SUCCESS(CreateInitNode(global_data));
  GE_ASSERT_SUCCESS(CreateDeInitNode());
  GE_TIMESTAMP_START(CreateMainNode);
  ge::ExecuteGraphPtr main_graph;
  GE_ASSERT_NOTNULL(model_desc_holder_);
  GE_ASSERT_SUCCESS(CreateMainNode(graph, *model_desc_holder_, global_data, main_graph));
  GE_TIMESTAMP_EVENT_END(CreateMainNode, "ConvertComputeGraphToExecuteGraph::CreateMainNode");

  auto root_frame = bg::ValueHolder::PopGraphFrame();
  GE_ASSERT_NOTNULL(root_frame);
  GE_ASSERT_NOTNULL(root_frame->GetExecuteGraph());
  auto exe_graph = root_frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(exe_graph);
  // 为了在DUMP图时识别单算子场景
  bool is_single_op = ge::GraphUtils::IsSingleOpScene(graph);
  if (is_single_op) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetBool(exe_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op));
  }
  ge::DumpGraph(exe_graph.get(), "ExeGraphBeforeOptimize");

  GE_TIMESTAMP_START(RunAllPass);
  GE_ASSERT_SUCCESS(bg::OfflineOptimizer(optimize_option, global_data).Run(exe_graph.get()));
  GE_TIMESTAMP_EVENT_END(RunAllPass, "ConvertComputeGraphToExecuteGraph::RunAllPass");

  GE_TIMESTAMP_START(TopologicalSorting);
  GE_ASSERT_SUCCESS(exe_graph->TopologicalSorting());
  GE_TIMESTAMP_EVENT_END(TopologicalSorting, "ConvertComputeGraphToExecuteGraph::TopologicalSorting");

  const auto root_graph_nodes = exe_graph->GetAllNodes();
  const auto main_graph_nodes = main_graph->GetAllNodes();

  GE_TIMESTAMP_START(CalculatePriority);
  GE_ASSERT_SUCCESS(
      bg::NodePriorityCalculator(*root_frame).CalcNodeExecutionPriorities(main_graph_nodes, root_graph_nodes.size()));
  GE_TIMESTAMP_EVENT_END(CalculatePriority, "ConvertComputeGraphToExecuteGraph::CalculatePriority");

  GE_TIMESTAMP_START(AppendGraphLevelData);
  GE_ASSERT_SUCCESS(AppendGraphLevelData(*root_frame, graph, exe_graph.get(), root_graph_nodes));
  GE_TIMESTAMP_EVENT_END(AppendGraphLevelData, "ConvertComputeGraphToExecuteGraph::AppendGraphLevelData");

  GE_ASSERT_SUCCESS(ValidateExeGraph(main_graph_nodes));
  // todo 这个extattr会被删除，因为compute graph无法序列化，删除后通过其他方式传给subscriber
  exe_graph->SetExtAttr(kComputeGraph, graph);
  GE_TIMESTAMP_EVENT_END(ConvertComputeGraphToExecuteGraphAll, "ConvertComputeGraphToExecuteGraph::All");
  return exe_graph;
}
}  // namespace gert
