/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_optimize.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/passes/standard_optimize/common_subexpression_elimination_pass.h"
#include "graph/passes/pass_manager.h"
#include "common/checker.h"
#include "graph/operator_factory.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "api/aclgrph/option_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_pre_processor.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_post_processor.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "graph/passes/feature/auto_fuse_pass.h"
#include "mmpa/mmpa_api.h"
#include "common/compile_profiling/ge_trace_wrapper.h"

namespace ge {
namespace {
constexpr char const *kCastInsertBeforeAutoFuse = "_is_insert_before_autofuse";
constexpr char const *kAutoFuseEnableOption = "--enable_autofuse";

Status InsertCastForDataTypeUnconsistantNode(const ge::ComputeGraphPtr &compute_graph) {
  size_t insert_cast_op_count = 0UL;
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
      GE_ASSERT_NOTNULL(out_anchor);
      auto out_data_type = op_desc->GetOutputDesc(out_anchor->GetIdx()).GetDataType();
      for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_NOTNULL(peer_in_anchor);
        const auto peer_node = peer_in_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(peer_node);
        const auto &peer_op_desc = peer_node->GetOpDesc();
        GE_ASSERT_NOTNULL(peer_op_desc);
        auto peer_in_data_type = peer_op_desc->GetInputDesc(peer_in_anchor->GetIdx()).GetDataType();
        if (out_data_type != peer_in_data_type) {
          auto cast_op_name = "insert_before_autofuse_cast_op_" + std::to_string(insert_cast_op_count);
          insert_cast_op_count++;
          auto cast_operator = OperatorFactory::CreateOperator(cast_op_name.c_str(), CAST);
          auto cast_op = OpDescUtils::GetOpDescFromOperator(cast_operator);
          GE_ASSERT_NOTNULL(cast_op);
          GE_ASSERT_TRUE(AttrUtils::SetBool(cast_op, kCastInsertBeforeAutoFuse, true));
          GE_ASSERT_TRUE(AttrUtils::SetInt(cast_op, CAST_ATTR_DST_TYPE, static_cast<int64_t>(peer_in_data_type)));
          GE_ASSERT_SUCCESS(cast_op->UpdateInputDesc(0, op_desc->GetOutputDesc(out_anchor->GetIdx())));
          GE_ASSERT_SUCCESS(cast_op->UpdateOutputDesc(0, peer_op_desc->GetInputDesc(peer_in_anchor->GetIdx())));
          GE_ASSERT_TRUE(GraphUtils::InsertNodeAfter(out_anchor, {peer_in_anchor}, cast_op, 0, 0) != nullptr);
          GELOGI("Insert node:%s(%s) between node:%s(%s) output:%d and node:%s(%s) input:%d before autofuse.",
              cast_op->GetName().c_str(), cast_op->GetType().c_str(), op_desc->GetName().c_str(),
              op_desc->GetType().c_str(), out_anchor->GetIdx(), peer_op_desc->GetName().c_str(),
              peer_op_desc->GetType().c_str(), peer_in_anchor->GetIdx());
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

OpDescPtr CreateTensorShape(const GeTensorDesc &data_tensor) {
  GeTensorPtr tensor = MakeShared<GeTensor>();
  GE_ASSERT_NOTNULL(tensor, "New GeTensor failed");

  tensor->MutableTensorDesc().SetDataType(DT_INT64);
  tensor->MutableTensorDesc().SetFormat(FORMAT_ND);
  const auto& dst_ge_shape = data_tensor.GetOriginShape();
  auto dim_cnt = static_cast<int64_t>(dst_ge_shape.GetDimNum());
  if (dim_cnt == 0) {  // if the dim_cnt is 0, the tensor is a scalar
    tensor->MutableTensorDesc().SetShape(GeShape({0}));
  } else {
    tensor->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({dim_cnt})));
    auto dst_shape = MakeUnique<int64_t[]>(dim_cnt);
    GE_ASSERT_NOTNULL(dst_shape, "Malloc buffer failed, size:%zu", dim_cnt);
    for (int64_t i = 0; i < dim_cnt; ++i) {
      dst_shape[i] = dst_ge_shape.GetDim(static_cast<size_t>(i));
    }
    GE_ASSERT_GRAPH_SUCCESS(
        tensor->SetData(reinterpret_cast<const uint8_t *>(dst_shape.get()), dim_cnt * sizeof(int64_t)),
        "Set data to tensor failed");
  }
  tensor->MutableTensorDesc().SetOriginShape(tensor->MutableTensorDesc().GetShape());
  GELOGD("Create shape input dim [%s]", dst_ge_shape.ToString().c_str());
  return OpDescUtils::CreateConstOp(tensor);
}

Status AbnormalReshapeRecovery(const ge::ComputeGraphPtr &compute_graph) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() != RESHAPE || node->GetAllInDataAnchorsSize() == 2U) {
      continue;
    }
    GELOGI("Start recover abnormal reshape node %s[%s]", node->GetNamePtr(), node->GetTypePtr());
    auto output_desc = node->GetOpDesc()->GetOutputDescPtr(0);
    GE_ASSERT_NOTNULL(output_desc);
    GE_ASSERT_TRUE(!output_desc->GetShape().IsUnknownShape(),
                   "reshape output shape is unknown shape while current reshape node %s[%s] is abnormal",
                   node->GetNamePtr(), node->GetTypePtr());
    auto shape_op_desc = CreateTensorShape(*output_desc);
    GE_ASSERT_NOTNULL(shape_op_desc, "[Create][TensorShape] Failed to add shape for reshape");
    auto owner_compute_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(owner_compute_graph);
    auto shape_node = owner_compute_graph->InsertNodeBefore(node, shape_op_desc);
    GE_ASSERT_NOTNULL(shape_node, "Add node:%s(%s) to graph:%s failed", shape_op_desc->GetNamePtr(),
                      shape_op_desc->GetTypePtr(), owner_compute_graph->GetName().c_str());
    GE_ASSERT_SUCCESS(node->AddLinkFrom(shape_node));
    GELOGD("[Reshape][Const] In %s Add edge between op:%s(%s)(out_index:0) and op:%s(%s)(in_index:1) success.",
           owner_compute_graph->GetName().c_str(), shape_node->GetNamePtr(), shape_node->GetTypePtr(),
           node->GetNamePtr(), RESHAPE);
  }
  return ge::GRAPH_SUCCESS;
}

Status GraphOptimizerBeforeAutofuse(const ge::ComputeGraphPtr &compute_graph) {
  GE_ASSERT_NOTNULL(compute_graph);
  // 1. 在FE的融合阶段，一些融合pass会产生非标的reshape算子，reshape算子只有一个输入一个输出，输出是静态，恢复一下该非标reshape
  GE_ASSERT_SUCCESS(AbnormalReshapeRecovery(compute_graph));

  GEPass ge_passes(compute_graph);
  NamesToPass names_to_passes;
  ConstantFoldingPass constant_folding_pass;
  names_to_passes.emplace_back("BeforeAutofuse::ConstantFoldingPass", &constant_folding_pass);
  GE_ASSERT_SUCCESS(ge_passes.Run(names_to_passes));

  PassManager graph_pass;
  GE_ASSERT_SUCCESS(graph_pass.AddPass("BeforeAutofusePass::CommonSubexpressionEliminationPass",
      new (std::nothrow) CommonSubexpressionEliminationPass));
  GE_ASSERT_SUCCESS(graph_pass.Run(compute_graph));
  return GRAPH_SUCCESS;
}

Status DeleteCastForDataTypeUnconsistantNode(const ge::ComputeGraphPtr &compute_graph) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (AttrUtils::HasAttr(node->GetOpDesc(), kCastInsertBeforeAutoFuse)) {
      // 控制子图场景
      GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(node, {0}),
        "Isolate node[%s][%s] failed.",node->GetTypePtr(), node->GetNamePtr());
      const auto owner_graph = node->GetOwnerComputeGraph();
      GE_ASSERT_NOTNULL(owner_graph);
      GE_ASSERT_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(owner_graph, node),
        "Remove node[%s][%s] failed.",node->GetTypePtr(), node->GetNamePtr());
      GELOGI("Delete node:[%][%s] after autofuse", node->GetName().c_str(), node->GetType().c_str());
    }
  }
  return GRAPH_SUCCESS;
}
bool IsEnableAutofuse() {
  // 新的自动融合配置选项
  if (GetAutofuseFlagValue(kAutoFuseEnableOption) == "true") {
    return true;
  }
  GELOGI("Option --enable_autofuse=true is not set in env AUTOFUSE_FLAGS, skip autofuse.");
  return false;
}
}

Status AutofuseOptimize::PreProcess(const ge::ComputeGraphPtr &compute_graph) const {
  // 临时方案：由于当前fe做了精度选择和格式选择分离后，当type不连续时，fe在精度选择接口中插入Cast调整type的逻辑改动工作量较大，
  // 此处GE先在自动融合前帮忙插入Cast算子，并在自动融合后，删除剩余的在自动融合前阶段插入的Cast算子，
  // fe重构时需要在fe实现此逻辑
  GE_ASSERT_SUCCESS(InsertCastForDataTypeUnconsistantNode(compute_graph));
  // 此处插入Cast，并做常量折叠时，可能存在以下问题，当前通过增加pass处理
  // 1. 当输出分支较多时，插入Cast个数较多，增加CommonSubexpressionEliminationPass消减Cast个数
  // 2. 由于此处提前做了常量折叠，需要对非标Reshape做标准化，增加ReshapeRemovePass和ReshapeRecoveryPass
  GE_ASSERT_SUCCESS(GraphOptimizerBeforeAutofuse(compute_graph));
  return GRAPH_SUCCESS;
}

Status AutofuseOptimize::PostProcess(const ge::ComputeGraphPtr &compute_graph) const {
  GE_ASSERT_SUCCESS(SymbolicInfoPostProcessor::Run(compute_graph));
  // 做完自动融合之后，需要删除剩余的在InsertCastForDataTypeUnconsistantNode接口插入的Cast算子
  GE_ASSERT_SUCCESS(DeleteCastForDataTypeUnconsistantNode(compute_graph));
  return GRAPH_SUCCESS;
}

Status AutofuseOptimize::Run(const ge::ComputeGraphPtr &compute_graph, const std::vector<GeTensor> &inputs) const {
  if (!IsEnableAutofuse()) {
    GELOGI("Auto fuse env is disable, skip it.");
    return GRAPH_SUCCESS;
  }

  bool is_single_op_scene = false;
  ge::AttrUtils::GetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene);
  if (is_single_op_scene) {
    GELOGI("Skip auto fuse for single op scene.");
    return GRAPH_SUCCESS;
  }

  GE_ASSERT_SUCCESS(PreProcess(compute_graph));
  PassManager graph_pass_for_autofuse;
  GE_TRACE_START(Symbolize);
  GE_ASSERT_GRAPH_SUCCESS(SymbolicShapeSymbolizer::Symbolize(compute_graph, inputs), "Symbolize graph input failed, graph %s",
                          compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(Symbolize, ("SymbolicShapeInference::Symbolize::" + compute_graph->GetName()).c_str());
  GE_ASSERT_SUCCESS(SymbolicInfoPreProcessor::Run(compute_graph, inputs));
  GE_CHK_STATUS_RET(graph_pass_for_autofuse.AddPass("PreRun::AutoFusePass", new (std::nothrow) AutoFusePass()));
  GE_CHK_STATUS_RET(graph_pass_for_autofuse.Run(compute_graph));

  GE_ASSERT_SUCCESS(PostProcess(compute_graph));
  GE_DUMP(compute_graph, "After_AutoFusePass");
  return GRAPH_SUCCESS;
}
}