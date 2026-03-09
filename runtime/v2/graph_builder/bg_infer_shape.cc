/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_infer_shape.h"
#include "bg_infer_shape_range.h"
#include "securec.h"
#include "common/checker.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "exe_graph/lowering/value_holder.h"
#include "storage_format.h"
#include "graph/ir_definitions_recover.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/inference_rule.h"
#include "bg_compatible_utils.h"
#include "framework/common/types.h"
#include "register/node_converter_registry.h"
#include "exe_graph/lowering/frame_selector.h"
#include "bg_model_desc.h"
#include "engine/node_converter_utils.h"
#include "aicore/converter/autofuse_node_converter.h"

namespace gert {
namespace bg {
namespace {
constexpr char const *kRetValType  = "_RetVal";
struct LowerIOShapes {
  std::vector<ValueHolderPtr> input_shapes;
  std::vector<ValueHolderPtr> output_shapes;
};

struct UBGraphOutNodesInfos {
  ge::NodePtr node;
  int32_t node_out_idx;
  int32_t parent_node_idx;
};
bool IsInferShapeRegistered(const std::string &type, const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  if (space_registry == nullptr) {
    return false;
  }
  auto op_funcs = space_registry->GetOpImpl(type.c_str());
  if ((op_funcs == nullptr) || (op_funcs->infer_shape == nullptr)) {
    return false;
  }
  return true;
}

bool NeedSymbolInferShape(const ge::NodePtr &node) {
  return ge::OpTypeUtils::IsAutofuseNode(node->GetOpDesc());
}

std::vector<ValueHolderPtr> BuildCompatibleInferShapeGraph(const ge::NodePtr &node,
                                                           const std::vector<ValueHolderPtr> &input_shapes,
                                                           LoweringGlobalData &global_data) {
  std::string type;
  GE_ASSERT_SUCCESS(ge::GetOriginalType(node, type), "Failed to get original type from %s(%s).",
                    node->GetName().c_str(), node->GetType().c_str());

  // 调用处已经校验node\op_desc不为空
  auto builder = [&type]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = ValueHolder::CreateConst(type.c_str(), type.size() + 1, true);
      return {ValueHolder::CreateSingleDataOutput("FindCompatibleInferShapeFunc", {node_type})};
    });
  };
  auto infer_func = global_data.GetOrCreateUniqueValueHolder(
      type + "_FindCompatibleInferShapeFunc_", builder)[0];

  auto op_buffer_vec = CompatibleUtils::BuildOpDescBufferConst(node);
  auto op = ValueHolder::CreateSingleDataOutput("CreateOpFromBuffer", op_buffer_vec);

  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(op);
  inputs.emplace_back(infer_func);
  inputs.insert(inputs.cend(), input_shapes.cbegin(), input_shapes.cend());

  return ValueHolder::CreateDataOutput("CompatibleInferShape", inputs, node->GetAllOutDataAnchorsSize());
}

/*
 *        InferShape
 *        /        \
 * all-shapes      FindInferShapeFunc
 *                    /         \
 *               node-type   space_registry
 */
std::vector<ValueHolderPtr> BuildInferShapeGraph(const ge::NodePtr &node,
                                                 const std::vector<ValueHolderPtr> &input_shapes,
                                                 LoweringGlobalData &global_data) {
  std::string type;
  if (ge::GetOriginalType(node, type) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get original type from %s(%s).", node->GetName().c_str(), node->GetType().c_str());
    return {};
  }
  // 调用处已经校验node\op_desc不为空
  ge::OppImplVersion opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
  auto builder = [&type, &opp_impl_version, &global_data]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = ValueHolder::CreateConst(type.c_str(), type.size() + 1, true);
      auto space_registry = bg::HolderOnInit(bg::GetSpaceRegistry(global_data, opp_impl_version));
      return {ValueHolder::CreateSingleDataOutput("FindInferShapeFunc", {node_type, space_registry})};
    });
  };
  auto infer_func = global_data.GetOrCreateUniqueValueHolder(
    type + "_FindInferShapeFunc_" + to_string(static_cast<int32_t>(opp_impl_version)), builder)[0];
  auto inputs = input_shapes;
  inputs.emplace_back(infer_func);
  return ValueHolder::CreateDataOutput("InferShape", inputs, node->GetAllOutDataAnchorsSize());
}

/*
 *       SymbolInferShape
 *        /        \
 *   sym_shapes   DlsymFunctionFromHandles
 *                    /            \
 *               Dlopenso     symbol_infer_func_name(const)
 *                  /
 *             bin_file_path(const)
 */
std::vector<ValueHolderPtr> BuildSymbolInferShapeGraph(const ge::NodePtr &node,
                                                       const std::vector<ValueHolderPtr> &input_shapes,
                                                       LoweringGlobalData &global_data) {
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  // infershape连边
  std::vector<bg::ValueHolderPtr> inputs_holders;
  // input shapes with size
  auto input_shapes_with_size = AutofuseNodeConveter::GetSymbolInputsWithSize(
      global_data, input_shapes, node->GetOwnerComputeGraph()->GetName());
  inputs_holders.insert(inputs_holders.end(), input_shapes_with_size.begin(), input_shapes_with_size.end());
  // 3. all symbol number
  auto all_sym_num_holder = AutofuseNodeConveter::GetAllSymbolNumHolder(global_data, node);
  GE_ASSERT_NOTNULL(all_sym_num_holder);
  inputs_holders.emplace_back(all_sym_num_holder);
  // 4. DfxInputSymbolInfo function
  auto dfx_func = AutofuseNodeConveter::GetAutofuseHandle(
      global_data, node, GetAutofuseFuncsOutput::kDfxInputSymbolInfo);
  GE_ASSERT_NOTNULL(dfx_func);
  inputs_holders.emplace_back(dfx_func);;
  // last: infershape function
  auto get_infer_shape_func = AutofuseNodeConveter::GetAutofuseHandle(
      global_data, node, GetAutofuseFuncsOutput::kInferShape);
  GE_ASSERT_NOTNULL(get_infer_shape_func);
  inputs_holders.emplace_back(get_infer_shape_func);

  auto output_num = op_desc->GetOutputsSize();
  auto infer_merge_key = ge::AttrUtils::GetStr(op_desc, "_symbol_infer_shape_cache_key");
  if (infer_merge_key != nullptr && !infer_merge_key->empty()) {
    GELOGD("[%s][%s] get infer merge key[%s] success, merge infer node.", node->GetNamePtr(), node->GetTypePtr(),
           infer_merge_key->c_str());
    auto builder = [&inputs_holders, &output_num]() -> std::vector<ValueHolderPtr> {
      return bg::ValueHolder::CreateDataOutput("InferShape", inputs_holders, output_num);
    };
    return global_data.GetOrCreateUniqueValueHolder(*infer_merge_key, builder);
  } else {
    return bg::ValueHolder::CreateDataOutput("InferShape", inputs_holders, output_num);
  }
}

/*
 *     InferShapeByRule
 *           |
 *     LoadShapeRuleFromBinary/LoadShapeRuleFromJson
 *           |             \                 \
 *     Const(rule binary) Const(rule size)  Const(rule json)
 */
std::vector<ValueHolderPtr> BuildInferShapeGraphByRule(const std::string &rule, const ge::Buffer &compiled_rule,
                                                       const std::vector<ValueHolderPtr> &input_shapes,
                                                       const size_t num_outputs, LoweringGlobalData &global_data) {
  auto builder = [&rule, &compiled_rule]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&rule, &compiled_rule]() -> std::vector<bg::ValueHolderPtr> {
      auto rule_json_holder = ValueHolder::CreateConst(rule.c_str(), rule.size() + 1, true);
      if (compiled_rule.size() > 0U) {
        auto rule_binary_holder = ValueHolder::CreateConst(compiled_rule.data(), compiled_rule.size(), false);
        const size_t rule_binary_size = compiled_rule.size();
        auto rule_binary_size_holder = ValueHolder::CreateConst(&rule_binary_size, sizeof(size_t), false);
        return {ValueHolder::CreateSingleDataOutput("LoadShapeRuleFromBinary",
                                                    {rule_binary_holder, rule_binary_size_holder, rule_json_holder})};
      }
      return {ValueHolder::CreateSingleDataOutput("LoadShapeRuleFromJson", {rule_json_holder})};
    });
  };

  const auto holders = global_data.GetOrCreateUniqueValueHolder(rule, builder);
  GE_ASSERT_EQ(holders.size(), 1U);
  std::vector<bg::ValueHolderPtr> inputs_holders = input_shapes;
  inputs_holders.emplace_back(holders.back());

  return bg::ValueHolder::CreateDataOutput("InferShapeByRule", inputs_holders, num_outputs);
}
} // namespace

// 在convert中打开函数放到global_data中
std::vector<ValueHolderPtr> InferStorageShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                              LoweringGlobalData &global_data) {
  if (node == nullptr) {
    return {};
  }
  std::string type;
  if (ge::GetOriginalType(node, type) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get original type from %s(%s).", node->GetName().c_str(), node->GetType().c_str());
    return {};
  }
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  if (NeedSymbolInferShape(node)) {
    return BuildSymbolInferShapeGraph(node, input_shapes, global_data);
  }
  auto real_inputs_size = node->GetInDataNodesAndAnchors().size();
  GE_ASSERT_EQ(input_shapes.size(), real_inputs_size);
  if (IsInferShapeRegistered(type, global_data.GetSpaceRegistryV2(
                                       static_cast<gert::OppImplVersionTag>(node->GetOpDesc()->GetOppImplVersion())))) {
    return BuildInferShapeGraph(node, input_shapes, global_data);
  }
  const std::string infer_rule = ge::InferenceRule::GetInferenceRule(node->GetOpDesc());
  if (!infer_rule.empty()) {
    ge::Buffer compiled_rule;
    ge::AttrUtils::GetBytes(node->GetOpDesc(), ge::COMPILED_INFERENCE_RULE_BINARY, compiled_rule);
    GELOGD("Node %s type %s infer shape by rule: %s.", node->GetName().c_str(), type.c_str(), infer_rule.c_str());
    return BuildInferShapeGraphByRule(infer_rule, compiled_rule, input_shapes, node->GetOpDesc()->GetOutputsSize(),
                                      global_data);
  }
  // To compatible with old version infer_fun, build differnt exe graph for infershape
  GELOGW("Node %s type %s not support v2 infershape. Turns to v1 infershape.", node->GetName().c_str(), type.c_str());
  return BuildCompatibleInferShapeGraph(node, input_shapes, global_data);
}

HyperStatus LowerInnerData(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                           std::map<int64_t, LowerIOShapes> &node_2_shapes) {
  bg::ValueHolder::AddRelevantInputNode(node);
  auto index = GetParentNodeInputIndex(node);
  if ((index < 0) || (index >= static_cast<int64_t>(input_shapes.size()))) {
    GELOGE(ge::FAILED, "Node %s parent node index is %ld, not valid.", node->GetName().c_str(), index);
    return HyperStatus::ErrorStatus("Parent node index is invalid.");
  }
  node_2_shapes[node->GetOpDescBarePtr()->GetId()].output_shapes.emplace_back(input_shapes[index]);
  return HyperStatus::Success();
}

std::vector<ValueHolderPtr> LowerInnerOutNodes(
    const std::vector<UBGraphOutNodesInfos> &ub_graph_out_nodes_info,
    std::map<int64_t, LowerIOShapes> &node_2_shapes) {
  std::vector<ValueHolderPtr> output_shapes(ub_graph_out_nodes_info.size());
  for (const auto &out_node_info : ub_graph_out_nodes_info) {
    auto index = out_node_info.parent_node_idx;
    if ((index < 0) || (index >= static_cast<int64_t>(ub_graph_out_nodes_info.size()))) {
      GELOGE(ge::FAILED, "Index[%ld] is invalid.", index);
      return {};
    }
    const int64_t peer_node_id = out_node_info.node->GetOpDescBarePtr()->GetId();
    auto peer_node_out_shapes = node_2_shapes[peer_node_id].output_shapes;
    output_shapes[index] = peer_node_out_shapes.at(out_node_info.node_out_idx);
  }
  return output_shapes;
}

std::vector<ValueHolderPtr> LowerInnerRetVal(const std::vector<ge::NodePtr> &nodes,
                                             std::map<int64_t, LowerIOShapes> &node_2_shapes) {
  std::vector<ValueHolderPtr> output_shapes(nodes.size());
  for (const auto &ret_val : nodes) {
    auto index = GetParentNodeInputIndex(ret_val);
    if ((index < 0) || (index >= static_cast<int64_t>(nodes.size()))) {
      GELOGE(ge::FAILED, "Index[%ld] is invalid.", index);
      return {};
    }
    const auto &ret_val_inputs_and_anchors = ret_val->GetInDataNodesAndAnchors();
    if (ret_val_inputs_and_anchors.size() != 1U) {
      GELOGE(ge::FAILED, "Retval %s inputs size %zu is invalid.", ret_val->GetName().c_str(),
             ret_val_inputs_and_anchors.size());
      return {};
    }
    auto peer_node_2_out_index = ret_val_inputs_and_anchors.at(0);
    const int64_t peer_node_id = peer_node_2_out_index.first->GetOpDescBarePtr()->GetId();
    auto peer_node_out_shapes = node_2_shapes[peer_node_id].output_shapes;
    output_shapes[index] = peer_node_out_shapes.at(peer_node_2_out_index.second->GetIdx());
  }
  return output_shapes;
}

HyperStatus GetNodeInputShapes(const ge::NodePtr &node, std::map<int64_t, LowerIOShapes> &node_2_shapes) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t node_id = op_desc->GetId();
  for (const auto &peer_node_and_anchor : node->GetInDataNodesAndAnchors()) {
    const auto &peer_node = peer_node_and_anchor.first;
    const auto out_index = peer_node_and_anchor.second->GetIdx();
    const auto &peer_output_shapes = node_2_shapes[peer_node->GetOpDescBarePtr()->GetId()].output_shapes;
    if (out_index < 0 || static_cast<size_t>(out_index) >= peer_output_shapes.size()) {
      GELOGE(ge::FAILED, "Node %s output shapes size is %zu, try invalid index %d.", peer_node->GetName().c_str(),
             peer_output_shapes.size(), out_index);
      return HyperStatus::ErrorStatus("Index %d out of range of peer out shape %zu.", out_index,
                                      peer_output_shapes.size());
    }
    node_2_shapes[node_id].input_shapes.emplace_back(peer_output_shapes.at(out_index));
  }
  return HyperStatus::Success();
}

ge::Status LoweringNormalNode(const ge::NodePtr &node, std::map<int64_t, LowerIOShapes> &node_2_shapes,
                              LoweringGlobalData &global_data) {
  bg::ValueHolder::SetCurrentComputeNode(node);
  if (!GetNodeInputShapes(node, node_2_shapes).IsSuccess()) {
    GELOGE(ge::FAILED, "get node input shape failed, node:%s", node->GetName().c_str());
    return ge::FAILED;
  }
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t node_id = op_desc->GetId();
  const auto &output_shapes = InferStorageShape(node, node_2_shapes[node_id].input_shapes, global_data);
  node_2_shapes[node_id].output_shapes = output_shapes;
  return ge::SUCCESS;
}

std::vector<ValueHolderPtr> InferUbGraphShape(const ge::ComputeGraphPtr &compute_graph,
                                              const std::vector<ValueHolderPtr> &input_shapes,
                                              LoweringGlobalData &global_data) {
  if ((compute_graph->TopologicalSorting() != ge::GRAPH_SUCCESS) ||
      (ge::RecoverIrDefinitions(compute_graph) != ge::GRAPH_SUCCESS)) {
    return {};
  }
  std::map<int64_t, LowerIOShapes> node_2_shapes;
  std::vector<ge::NodePtr> ret_val_nodes;
  std::vector<UBGraphOutNodesInfos> ub_graph_out_nodes_info;
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto type = ge::NodeUtils::GetNodeType(node);
    if (type == ge::DATA) {
      LowerInnerData(node, input_shapes, node_2_shapes);
      continue;
    }
    if (type == kRetValType) {
      ret_val_nodes.emplace_back(node);
      continue;
    }
    if (type == ge::NETOUTPUT) {
      size_t i = 0U;
      for (const auto &out_node_and_anchor : node->GetInDataNodesAndAnchors()) {
        int32_t parent_node_index = -1;
        (void)ge::AttrUtils::GetInt(node->GetOpDesc()->GetInputDesc(i), ge::ATTR_NAME_PARENT_NODE_INDEX,
                                    parent_node_index);
        ub_graph_out_nodes_info.emplace_back(
            UBGraphOutNodesInfos{out_node_and_anchor.first, out_node_and_anchor.second->GetIdx(), parent_node_index});
        i++;
      }
      continue;
    }
    // lowering normal node
    auto ret = LoweringNormalNode(node, node_2_shapes, global_data);
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "lowering normal node failed, node:%s(%s)", node->GetName().c_str(), type.c_str());
      return {};
    }
  }
  if (!ret_val_nodes.empty()) {
    GE_ASSERT_TRUE(ub_graph_out_nodes_info.empty(), "retval nodes size is %zu, should not has NetOutput node",
                   ret_val_nodes.size());
    return LowerInnerRetVal(ret_val_nodes, node_2_shapes);
  }
  GE_ASSERT_TRUE(ret_val_nodes.empty(), "NetOutput exist, should not has retval node");
  return LowerInnerOutNodes(ub_graph_out_nodes_info, node_2_shapes);
}

bool IsOutputUnkownShape(const ge::OpDescPtr &op_desc) {
  for (auto &out_tensor : op_desc->GetAllOutputsDescPtr()) {
    if (out_tensor != nullptr && out_tensor->GetShape().IsUnknownShape()) {
      return true;
    }
  }
  return false;
}

bool IsUnkownShape(const ge::OpDescPtr &op_desc) {
  if (IsOutputUnkownShape(op_desc)) {
    return true;
  }
  for (auto &in_tensor : op_desc->GetAllInputsDescPtr()) {
    if (in_tensor != nullptr && in_tensor->GetShape().IsUnknownShape()) {
      return true;
    }
  }
  return false;
}
}  // namespace bg
}  // namespace gert
