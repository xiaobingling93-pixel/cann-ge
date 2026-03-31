/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/auto_fuse_pass.h"

#include "op_desc_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "common/checker.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/utils/node_utils.h"
#include "autofuse_frame/autofuse_frames.h"
#include "common/ge_types.h"
#include "mmpa/mmpa_api.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/passes/base_pass.h"
#include "algebraic_simplification_pass.h"
#include "graph/passes/standard_optimize/prune_pass.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/optimize/autofuse/autofuse/pattern_fusion/pattern_fusion.h"

namespace ge {
namespace {
// 标记ascbc node的引擎为AiCoreEngine
Status MarkEngineAttrForAutofuseNode(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (OpTypeUtils::IsAutofuseNode(op_desc)) {
      op_desc->SetOpEngineName(kEngineNameAiCore);
      op_desc->SetOpKernelLibName(kEngineNameAiCore);
      // AscBacendOp复用Tbe流程，设置ImplyType为TVM
      AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<int32_t>(domi::ImplyType::TVM));
      (void)ge::AttrUtils::SetStr(op_desc, ge::kAttrLowingFunc, "kAutoFuseLoweringFunc");
    }
  }
  return SUCCESS;
}
class AutofuseCounter : public Counter {
 public:
  AutofuseCounter() = default;
  ~AutofuseCounter() override = default;
  int64_t NextId() override {
    return unique_id_.fetch_add(1);
  }
 private:
  std::atomic<int64_t> unique_id_{0L};
};
}  // namespace

Status AutoFusePass::Run(ComputeGraphPtr graph) {
  GE_DUMP(graph, "AutoFuser_BeforeAutoFuse");
  GE_TRACE_START(PreProcess);
  GE_ASSERT_SUCCESS(PreProcess(graph));
  GE_COMPILE_TRACE_TIMESTAMP_END(PreProcess, ("AutoFusePass::PreProcess::" + graph->GetName()).c_str());
  SymbolicShapeInference symbolic_shape_inference;
  GE_TRACE_START(Infer);
  GE_ASSERT_SUCCESS(symbolic_shape_inference.Infer(graph));
  GE_COMPILE_TRACE_TIMESTAMP_END(Infer, ("SymbolicShapeInference::Infer::" + graph->GetName()).c_str());
  GE_DUMP(graph, "AutoFuser_AfterPreprocess");

  auto root_graph = ge::GraphUtils::FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);
  // 设置context
  ShapeEnvGuarder guarder(root_graph->GetAttrsGroup<ShapeEnvAttr>());

  auto autofuse_counter = MakeUnique<AutofuseCounter>();
  GE_ASSERT_NOTNULL(autofuse_counter);
  GE_ASSERT_SUCCESS(LoweringAndCanFuseWithCounter(graph, autofuse_counter.get()));

  GE_ASSERT_SUCCESS(MarkEngineAttrForAutofuseNode(graph));
  // todo: inner attrs 生命周期在自动融合结束时结束，外部只能看到公开的属性
  GE_DUMP(graph, "AutoFuser_AfterAllFuse");

  return SUCCESS;
}

Status AutoFusePass::PreProcess(const ComputeGraphPtr &graph) {
  // 代数简化
  GE_ASSERT_SUCCESS(AlgebraicSimplificationPass::Run(graph));
  // 在符号化推导之前执行的 PatternFusion Pass，不需要符号化信息
  GraphPasses graph_passes;
  graph_passes.prune_graph_func = [](const ComputeGraphPtr &graph) -> Status { return PrunePass().Run(graph); };
  graph_passes.constant_folding_func = [](NodePtr &node) -> Status { return ConstantFoldingPass().Run(node); };
  GE_ASSERT_SUCCESS(ge::PatternFusion::RunEarlyPasses(graph, graph_passes));
  return SUCCESS;
}

REG_PASS_OPTION("AutoFusePass").LEVELS(OoLevel::kO3);
}  // namespace ge
 