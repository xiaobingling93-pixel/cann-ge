/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "autofuser.h"
#include "ge_context.h"
#include "backend/backend_spec.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "common/platform_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_reg.h"
#include "pattern_fusion/pattern_fusion.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "post_process/asc_backend_post_processor.h"
#include "utils/autofuse_utils.h"
#include "utils/auto_fuse_config.h"
#include "autofuse_frame/autofuse_frames.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "platform/platform_info.h"

namespace ge {
using namespace autofuse;
namespace {
ge::Status InitFuseGraphOpTensor(const ge::ComputeGraphPtr &graph) {
  gert::SymbolShape symbol_shape({ge::Symbol(1)});
  for (const auto &ascbc_node : graph->GetAllNodesPtr()) {
    GE_ASSERT_NOTNULL(ascbc_node);
    const auto op_desc = ascbc_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
      const auto input_desc = op_desc->MutableInputDesc(i);
      GE_ASSERT_NOTNULL(input_desc);
      auto input_attr = input_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(input_attr);
      input_attr->symbolic_tensor.MutableOriginSymbolShape() = symbol_shape;
      input_desc->SetDataType(ge::DT_INT8);
    }

    for (size_t i = 0UL; i < op_desc->GetOutputsSize(); ++i) {
      const auto output_desc = op_desc->MutableOutputDesc(i);
      GE_ASSERT_NOTNULL(output_desc);
      auto output_attr = output_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(output_attr);
      output_attr->symbolic_tensor.MutableOriginSymbolShape() = symbol_shape;
      output_desc->SetDataType(ge::DT_INT8);
    }
  }
  return ge::SUCCESS;
}

const static std::unordered_set<std::string> kUnsupportedV1CtrlOps{
    "StreamActive", "StreamSwitch", "StreamMerge",   "Exit",
    "Enter",        "RefEnter",     "NextIteration", "RefNextIteration"};

static bool HasUnsupportedControlOp(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    if (kUnsupportedV1CtrlOps.count(node->GetType()) > 0UL) {
      GELOGD("Cannot apply can_fuse to graph [%s] with unsupported node [%s].", graph->GetName().c_str(),
             node->GetTypePtr());
      return true;
    }
  }
  return false;
}
}  // namespace

REG_OP(AscBackend)
    .DYNAMIC_INPUT(inputs, TensorType::ALL())
    .DYNAMIC_OUTPUT(outputs, TensorType::ALL())
    .OP_END_FACTORY_REG(AscBackend);
REG_OP(AscBackendNoKernelOp)
    .DYNAMIC_INPUT(inputs, TensorType::ALL())
    .DYNAMIC_OUTPUT(outputs, TensorType::ALL())
    .OP_END_FACTORY_REG(AscBackendNoKernelOp);
REG_OP(FusedAscBackend)
    .DYNAMIC_INPUT(inputs, TensorType::ALL())
    .DYNAMIC_OUTPUT(outputs, TensorType::ALL())
    .OP_END_FACTORY_REG(FusedAscBackend);
REG_OP(AscGraph)
    .DYNAMIC_INPUT(inputs, TensorType::ALL())
    .DYNAMIC_OUTPUT(outputs, TensorType::ALL())
    .ATTR(ascgraph, String, "")
    .OP_END_FACTORY_REG(AscGraph);

Autofuser::Autofuser(AutofuserOptions &options, CounterPtr counter) {
  options_ = options;
  counter_ = counter;
}

ge::Status UpdateAutoFuseConfigByChipType() {
  // 先初始化platform信息
  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);
  GELOGD("Get soc_version [%s] from context.", soc_version.c_str());
  fe::PlatFormInfos plat_form_infos;
  fe::OptionalInfos optional_infos;
  if (fe::PlatformInfoManager::GeInstance().GetPlatformInfos(soc_version, plat_form_infos, optional_infos) == 0U) {
    std::string npu_arch;
    GE_ASSERT_TRUE(plat_form_infos.GetPlatformRes("version", "NpuArch", npu_arch));
    ge::PlatformContext::GetInstance().SetPlatform(npu_arch);
  }

  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (backend_spec != nullptr) {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_input_nums_after_fuse =
        backend_spec->max_input_nums_after_fuse;
    GELOGI("update autofuse config: max_input_nums_after_fuse to %u by chip type",
           AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_input_nums_after_fuse);
  }
  return ge::SUCCESS;
}

ge::Status Autofuser::Fuse(const ge::ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    // 对整个根图进行融合，子图场景直接返回Success
    return ge::SUCCESS;
  }

  GE_ASSERT_SUCCESS(UpdateAutoFuseConfigByChipType());
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = options_.fwk_type;
  GELOGI("Framework type:%d", AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type);
  if (options_.fwk_type == AutoFuseFwkType::kTorch) {
    return FuseLite(graph);
  }

  ge::PatternFusion patter_fusion;
  GE_ASSERT_SUCCESS(patter_fusion.RunAllPatternFusion(graph));
  GE_DUMP(graph, "AutoFuser_AfterPatternFusion");

  GE_ASSERT_GRAPH_SUCCESS(graph->TopologicalSorting());
  ge::AscIrLowerer lowerer(counter_);
  auto start_lowering = std::chrono::high_resolution_clock::now();
  GE_ASSERT_SUCCESS(lowerer.Lowering(graph));
  auto end_lowering = std::chrono::high_resolution_clock::now();
  GEEVENT("[AUTOFUSE_PERFTRACE] The time cost of [Lowering] is [%lu] micro seconds.",
          std::chrono::duration_cast<std::chrono::microseconds>(end_lowering - start_lowering).count());
  GE_DUMP(graph, "AutoFuser_AfterLowering");

  AutofuseUtils::ClearUniqueNumber();
  // fuse前做反推，保证反推后的ascgraph跟torch一致
  GE_ASSERT_SUCCESS(asc_adapt::GeFallback(graph));
  GE_ASSERT_SUCCESS(asc_adapt::SaveReduceOriginalAxisToFuseAttr(graph));
  bool disable_can_fuse = HasUnsupportedControlOp(graph);
  if (!disable_can_fuse) {
    ge::FusionStrategySolver fusion_strategy_solver(counter_);
    GE_ASSERT_SUCCESS(fusion_strategy_solver.Fuse(graph));
    GE_DUMP(graph, "AutoFuser_AfterFusionStrategySolve");
  } else {
    GELOGI("Skip can_fuse for graph %s as it contains v1 control types op", graph->GetName().c_str());
  }

  auto start_lifting = std::chrono::high_resolution_clock::now();
  GE_ASSERT_SUCCESS(lowerer.Lifting(graph));
  auto end_lifting = std::chrono::high_resolution_clock::now();
  GEEVENT("[AUTOFUSE_PERFTRACE] The time cost of [Lifting] is [%lu] micro seconds.",
          std::chrono::duration_cast<std::chrono::microseconds>(end_lifting - start_lifting).count());
  GE_DUMP(graph, "AutoFuser_AfterLifting");

  AscBackendPostProcessor post_processor;
  GE_ASSERT_SUCCESS(post_processor.Do(graph));
  GE_DUMP(graph, "AutoFuser_AscBackendPostProcess");
  return ge::SUCCESS;
}

ge::Status Autofuser::FuseLite(const ge::ComputeGraphPtr &graph) const {
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 1U;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_proximity = std::numeric_limits<int64_t>::max();
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size =
      std::numeric_limits<uint64_t>::max();
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_input_nums_after_fuse =
      std::numeric_limits<uint32_t>::max();
  // torch fusegraph 没有输入输出信息，后端融合需要使用，此处补填
  GE_ASSERT_SUCCESS(InitFuseGraphOpTensor(graph));
  AutofuseUtils::ClearUniqueNumber();
  GE_ASSERT_SUCCESS(asc_adapt::SaveReduceOriginalAxisToFuseAttr(graph));
  ge::FusionStrategySolver fusion_strategy_solver;
  GE_ASSERT_SUCCESS(fusion_strategy_solver.Fuse(graph));
  GE_DUMP(graph, "AutoFuser_AfterFusionStrategySolve");

  AscBackendPostProcessor post_processor;
  GE_ASSERT_SUCCESS(post_processor.Do(graph));
  GE_DUMP(graph, "AutoFuser_AscBackendPostProcess");
  return ge::SUCCESS;
}

extern "C" {
ge::Status LoweringAndCanFuse(const ge::ComputeGraphPtr &graph) {
  AutofuserOptions options;
  Autofuser autofuser(options);
  return autofuser.Fuse(graph);
}

ge::Status LoweringAndCanFuseWithCounter(const ge::ComputeGraphPtr &graph, CounterPtr counter) {
  AutofuserOptions options;
  Autofuser autofuser(options, counter);
  return autofuser.Fuse(graph);
}
}

}  // namespace ge
