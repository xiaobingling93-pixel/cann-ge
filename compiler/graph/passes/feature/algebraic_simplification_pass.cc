/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "algebraic_simplification_pass.h"

#include "checker.h"
#include "common/types.h"
#include "debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/passes/standard_optimize/prune_pass.h"
#include "graph_utils.h"
#include "graph_utils_ex.h"
#include "node_adapter.h"
#include "node_utils.h"
#include "op_desc_utils.h"
#include "operator_factory.h"
#include "register/custom_pass_context_impl.h"
#include "util/mem_utils.h"

namespace ge {
namespace {
constexpr auto *kOpTypeDiv = "Div";
constexpr auto kPatternDataConst = "_Data_Const";

uint16_t ToFloat16(const int32_t value) {
  constexpr uint16_t kFp16One = 15360;
  return value == 0 ? 0 : kFp16One;
}

uint16_t ToBf16(const int32_t value) {
  constexpr uint16_t kBf16One = 16256;
  return value == 0 ? 0 : kBf16One;
}

class UselessBinaryOpRemovePass : public fusion::PatternFusionPass {
 public:
  UselessBinaryOpRemovePass() = default;
  ~UselessBinaryOpRemovePass() override = default;

 protected:
  std::vector<fusion::PatternUniqPtr> Patterns() override;
  fusion::GraphUniqPtr Replacement(const std::unique_ptr<fusion::MatchResult> &match_result) override;
  // 判断符合pattern结构的拓扑中，Const是否为0
  bool MeetRequirements(const std::unique_ptr<fusion::MatchResult> &match_result) override;

 private:
  static NodePtr AddNode(const ComputeGraphPtr &compute_graph, const std::string &op_name, const std::string &op_type);
  static Status AddPattern(const std::string &op_type, const std::string &x1_type, const std::string &x2_type,
                           std::vector<fusion::PatternUniqPtr> &patterns);
  static Status AddPattern(const std::string &op_type, bool lhs_can_be_const,
                           std::vector<fusion::PatternUniqPtr> &patterns);
  template <typename T>
  static bool IsAll(const GeTensor &tensor, T value);
  static bool CanRemove(const GeTensor &tensor, const std::string &op_type);
  static std::string GetPatternName(const fusion::MatchResult &match_result);
  static NodePtr GetTargetNode(const fusion::MatchResult &match_result);
  static NodePtr AddBroadcastNode(const ComputeGraphPtr &graph, const NodePtr &node, int32_t data_index);
};

std::vector<fusion::PatternUniqPtr> UselessBinaryOpRemovePass::Patterns() {
  std::vector<fusion::PatternUniqPtr> patterns;
  GE_ASSERT_SUCCESS(AddPattern(ADD, true, patterns));
  GE_ASSERT_SUCCESS(AddPattern(MUL, true, patterns));
  GE_ASSERT_SUCCESS(AddPattern(SUB, false, patterns));
  GE_ASSERT_SUCCESS(AddPattern(kOpTypeDiv, false, patterns));
  return patterns;
}

NodePtr UselessBinaryOpRemovePass::AddBroadcastNode(const ComputeGraphPtr &graph, const NodePtr &node, int32_t data_index) {
  const auto &shape = node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  const auto shape_tensor = MakeShared<GeTensor>();
  GE_ASSERT_NOTNULL(shape_tensor);
  auto &tensor_desc = shape_tensor->MutableTensorDesc();
  const auto shape_tensor_shape = GeShape(std::vector<int64_t>{static_cast<int64_t>(shape.size())});
  tensor_desc.Update(shape_tensor_shape, FORMAT_ND, DT_INT64);
  tensor_desc.SetOriginShape(shape_tensor_shape);
  shape_tensor->SetData(PtrToPtr<int64_t, uint8_t>(shape.data()), shape.size() * sizeof(int64_t));
  const auto shape_op_desc = OpDescUtils::CreateConstOpZeroCopy(shape_tensor);
  GE_ASSERT_NOTNULL(shape_op_desc);
  const auto shape_node = graph->AddNode(shape_op_desc);
  GE_ASSERT_NOTNULL(shape_node);
  const auto &brc_node_name = node->GetName() + "_broadcast_to_by_UselessBinaryOpRemovePass";
  const auto broadcast_to_op = ge::OperatorFactory::CreateOperator(brc_node_name.c_str(), BROADCAST_TO);
  broadcast_to_op.BreakConnect();
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(broadcast_to_op);
  GE_ASSERT_NOTNULL(op_desc);

  const auto &[src_node, out_anchor] = NodeUtils::GetInDataNodeAndAnchorByIndex(*node, data_index);
  GE_ASSERT_NOTNULL(src_node);
  const auto &src_op_desc = src_node->GetOpDesc();
  GE_ASSERT_NOTNULL(src_op_desc);
  constexpr uint32_t kIndexData = 0U;
  constexpr uint32_t kIndexShape = 1U;
  GE_ASSERT_GRAPH_SUCCESS(op_desc->UpdateInputDesc(kIndexData, src_op_desc->GetOutputDesc(out_anchor->GetIdx())));
  GE_ASSERT_GRAPH_SUCCESS(op_desc->UpdateInputDesc(kIndexShape, tensor_desc));
  GE_ASSERT_GRAPH_SUCCESS(op_desc->UpdateOutputDesc(0, node->GetOpDesc()->GetOutputDesc(0)));
  auto broadcast_to_node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(broadcast_to_node);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), broadcast_to_node->GetInDataAnchor(1)));
  return broadcast_to_node;
}

fusion::GraphUniqPtr UselessBinaryOpRemovePass::Replacement(const std::unique_ptr<fusion::MatchResult> &match_result) {
  const std::string pattern_name = GetPatternName(*match_result);
  const auto target_node = GetTargetNode(*match_result);
  const auto &op_desc = target_node->GetOpDesc();
  OpDescPtr const_op_desc;
  const int32_t data_index = pattern_name.find(kPatternDataConst) != std::string::npos ? 0 : 1;
  const auto compute_graph = ComGraphMakeShared<ComputeGraph>("replacement");
  GE_ASSERT_NOTNULL(compute_graph);
  const auto x1_node = AddNode(compute_graph, "x1", DATA);
  GE_ASSERT_NOTNULL(x1_node);
  GE_ASSERT_TRUE(AttrUtils::SetInt(x1_node->GetOpDesc(), ATTR_NAME_INDEX, data_index));
  const auto x2_node = AddNode(compute_graph, "x2", DATA);
  GE_ASSERT_NOTNULL(x2_node);
  GE_ASSERT_TRUE(AttrUtils::SetInt(x2_node->GetOpDesc(), ATTR_NAME_INDEX, (1 - data_index)));
  const auto net_output_desc = ComGraphMakeShared<OpDesc>("output", NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_desc);
  const auto &data_shape = op_desc->GetInputDesc(data_index).GetShape().GetDims();
  const auto &output_shape = op_desc->GetOutputDesc(0).GetShape().GetDims();
  auto parent_node = x1_node;
  if (data_shape != output_shape) { // need broadcast
    GELOGD("node: %s need broadcast, in_shape = %s, out_shape = %s", target_node->GetNamePtr(), ToString(data_shape).c_str(), ToString(output_shape).c_str());
    parent_node = AddBroadcastNode(compute_graph, target_node, data_index);
    GE_ASSERT_NOTNULL(parent_node);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), parent_node->GetInDataAnchor(0)));
  }
  GE_ASSERT_GRAPH_SUCCESS(net_output_desc->AddInputDesc(target_node->GetOpDesc()->GetOutputDesc(0)));
  const auto net_output_node = compute_graph->AddNode(net_output_desc);
  GE_ASSERT_NOTNULL(net_output_node);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(parent_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0)));
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto ret = ComGraphMakeUnique<Graph>(graph);
  GE_ASSERT_NOTNULL(ret);
  return ret;
}

bool UselessBinaryOpRemovePass::MeetRequirements(const std::unique_ptr<fusion::MatchResult> &match_result) {
  const auto &pattern_name = GetPatternName(*match_result);
  const auto target_node = GetTargetNode(*match_result);
  GE_WARN_ASSERT(target_node != nullptr);
  const int32_t const_index = pattern_name.find(kPatternDataConst) != std::string::npos ? 1 : 0;
  const auto const_node = NodeUtils::GetInDataNodeByIndex(*target_node, const_index);
  if (!NodeUtils::IsConst(*const_node)) {
    return false;
  }
  GELOGD("match pattern name: %s", pattern_name.c_str());
  ConstGeTensorPtr tensor;
  GE_WARN_ASSERT(AttrUtils::GetTensor(const_node->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor));
  const auto can_remove = CanRemove(*tensor, target_node->GetType());
  GELOGI("node:%s[%s] can remove = %d", target_node->GetNamePtr(), target_node->GetTypePtr(),
         static_cast<int32_t>(can_remove));
  return can_remove;
}

NodePtr UselessBinaryOpRemovePass::AddNode(const ComputeGraphPtr &compute_graph, const std::string &op_name,
                                           const std::string &op_type) {
  const auto op = OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
  op.BreakConnect();
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc, "Failed to create op, type = %s", op_type.c_str());
  const auto node = compute_graph->AddNode(op_desc);
  return node;
}
Status UselessBinaryOpRemovePass::AddPattern(const std::string &op_type, const std::string &x1_type,
                                             const std::string &x2_type,
                                             std::vector<fusion::PatternUniqPtr> &patterns) {
  std::string pattern_name = "fusion_patteran_" + op_type + "_" + x1_type + "_" + x2_type;
  const auto compute_graph = ComGraphMakeShared<ComputeGraph>(pattern_name);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto node = AddNode(compute_graph, op_type, op_type);
  GE_ASSERT_NOTNULL(node);
  const auto x1_node = AddNode(compute_graph, "x1", DATA);
  GE_ASSERT_NOTNULL(x1_node);
  GE_ASSERT_TRUE(AttrUtils::SetInt(x1_node->GetOpDesc(), ATTR_NAME_INDEX, 0));
  const auto x2_node = AddNode(compute_graph, "x2", DATA);
  GE_ASSERT_NOTNULL(x2_node);
  GE_ASSERT_TRUE(AttrUtils::SetInt(x2_node->GetOpDesc(), ATTR_NAME_INDEX, 1));
  const auto net_output_desc = ComGraphMakeShared<OpDesc>("output", NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_desc);
  GE_ASSERT_GRAPH_SUCCESS(net_output_desc->AddInputDesc(node->GetOpDesc()->GetOutputDesc(0)));
  NodePtr constant_node;
  const auto net_output_node = compute_graph->AddNode(net_output_desc);
  GE_ASSERT_NOTNULL(net_output_node);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), node->GetInDataAnchor(0)));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(x2_node->GetOutDataAnchor(0), node->GetInDataAnchor(1)));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0)));
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto pattern = ComGraphMakeUnique<fusion::Pattern>(std::move(graph));
  GE_ASSERT_NOTNULL(pattern);
  patterns.emplace_back(std::move(pattern));
  return SUCCESS;
}

Status UselessBinaryOpRemovePass::AddPattern(const std::string &op_type, bool lhs_can_be_const,
                                             std::vector<fusion::PatternUniqPtr> &patterns) {
  if (lhs_can_be_const) {
    GE_ASSERT_SUCCESS(AddPattern(op_type, CONSTANT, DATA, patterns));
  }
  GE_ASSERT_SUCCESS(AddPattern(op_type, DATA, CONSTANT, patterns));
  return SUCCESS;
}

template <typename T>
bool UselessBinaryOpRemovePass::IsAll(const GeTensor &tensor, T value) {
  const auto tensor_size = tensor.GetData().GetSize();
  const auto tensor_data = tensor.GetData().GetData();
  for (size_t i = 0UL; i < tensor_size; i += sizeof(value)) {
    // 浮点类型也使用完全匹配
    if (memcmp(tensor_data + i, &value, sizeof(value)) != EOK) {
      return false;
    }
  }
  return true;
}

bool UselessBinaryOpRemovePass::CanRemove(const GeTensor &tensor, const std::string &op_type) {
  int32_t value = -1;
  if (op_type == MUL || op_type == kOpTypeDiv) {
    value = 1;
  } else if (op_type == ADD || op_type == SUB) {
    value = 0;
  } else {
    // do nothing
  }
  GE_WARN_ASSERT((value == 0) || (value == 1));
  switch (tensor.GetTensorDesc().GetDataType()) {
    case DT_INT8:
    case DT_UINT8:
      return IsAll(tensor, static_cast<uint8_t>(value));
    case DT_INT16:
    case DT_UINT16:
      return IsAll(tensor, static_cast<uint16_t>(value));
    case DT_INT32:
    case DT_UINT32:
      return IsAll(tensor, static_cast<uint32_t>(value));
    case DT_INT64:
    case DT_UINT64:
      return IsAll(tensor, static_cast<uint64_t>(value));
    case DT_FLOAT16:
      return IsAll(tensor, ToFloat16(value));
    case DT_BF16:
      return IsAll(tensor, ToBf16(value));
    case DT_FLOAT:
      return IsAll(tensor, static_cast<float>(value));
    case DT_DOUBLE:
      return IsAll(tensor, static_cast<double>(value));
    default:
      GELOGI("dtype %s is not supported",
             TypeUtils::DataTypeToSerialString(tensor.GetTensorDesc().GetDataType()).c_str());
      return false;
  }
}

std::string UselessBinaryOpRemovePass::GetPatternName(const fusion::MatchResult &match_result) {
  AscendString graph_name;
  GE_ASSERT_GRAPH_SUCCESS(match_result.GetPatternGraph().GetName(graph_name));
  return graph_name.GetString();
}

NodePtr UselessBinaryOpRemovePass::GetTargetNode(const fusion::MatchResult &match_result) {
  std::set<std::string> kTargetOpTypes{MUL, ADD, SUB, kOpTypeDiv};
  const std::vector<GNode> &matched_nodes = match_result.GetMatchedNodes();
  for (const auto &node : matched_nodes) {
    AscendString type;
    GE_ASSERT_GRAPH_SUCCESS(node.GetType(type));
    if (kTargetOpTypes.find(type.GetString()) != kTargetOpTypes.end()) {
      return NodeAdapter::GNode2Node(node);
    }
  }
  return nullptr;
}
}  // namespace
graphStatus AlgebraicSimplificationPass::Run(const ComputeGraphPtr &compute_graph) {
  GELOGD("Run UselessBinaryOpRemovePass start");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto graph_ptr = std::shared_ptr<Graph>(&graph, [](Graph *) -> void {});
  UselessBinaryOpRemovePass used_binary_op_remove_pass;
  CustomPassContext context;
  const auto ret = used_binary_op_remove_pass.Run(graph_ptr, context);
  GE_WARN_ASSERT(ret == SUCCESS || ret == NOT_CHANGED, "Failed to run UselessBinaryOpRemovePass");
  if (ret == SUCCESS) {
    (void) PrunePass().Run(compute_graph);
  }
  GELOGD("Run UselessBinaryOpRemovePass end");
  return GRAPH_SUCCESS;
}
}  // namespace ge