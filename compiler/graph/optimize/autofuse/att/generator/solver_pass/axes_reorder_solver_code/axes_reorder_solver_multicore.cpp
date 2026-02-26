/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axes_reorder_solver_code_common.h"

namespace att {

// ============================================================================
// Section 4: Multicore Tiling
// ============================================================================

namespace {
inline std::string GenFindOptimalMcVal() {
  return R"(
    while (!(last_boundary == boundary && last_val == var->value)) {
      last_boundary = boundary;
      last_val = var->value;
      var->value = CeilDiv((boundary + var->value) / 2, var->align);
      bool not_satisfied_mc_cons = !SatisfyMCCons();
      if (block_loop_auto_tune) {
        cur_obj = GetPerf();
        var->value += 1;
        next_obj = GetPerf();
        var->value -= 1;
        not_satisfied_mc_cons = not_satisfied_mc_cons || cur_obj > pre_obj || cur_obj > next_obj;
      }
      if (not_satisfied_mc_cons) {
        boundary = var->value;
        var->value = last_val;
      } else {
        if (block_loop_auto_tune) {
          pre_obj = cur_obj;
        }
      }
    }
)";
}
}

std::string GenMutiCoreTilingCore() {
  std::string codes = R"(
bool AxesReorderSolver::MulticoreTilingCore(bool block_loop_auto_tune) {
  const int32_t num_vars = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (int32_t i = num_vars - 1; i >= 0; --i) {
    auto &var = vars[i];
    int64_t boundary = var->align;
    auto init_val = var->value;
    int64_t last_boundary = -1;
    int64_t last_val = -1;
    double pre_obj = -1.0;
    double cur_obj = -1.0;
    double next_obj = -1.0;
    if (block_loop_auto_tune) {
      pre_obj = GetPerf();
    })";
  std::string find_optimal_mc_val = GenFindOptimalMcVal();
  codes += find_optimal_mc_val;

  // 生成 OptimizeMCVariable 函数调用
  std::string optimize_mc_variable = GenOptimizeMCVariable();
  codes += optimize_mc_variable;

  codes.append("  }\n"
                 "  if (!SatisfyCons(ConstraintType::MC_MIXED)) {\n"
                 "    OP_LOGW(OP_NAME, \"Multicore Tiling Calculation failed in the final check, input: %s\", input_.DebugString().c_str());\n"
                 "    return false;\n"
                 "  }\n"
                 "  return true;\n"
                 "}\n");
  return codes;
}

std::string GenOptimizeMCVariable() {
  return R"(
    if (!SatisfyMCCons()) {
      var->value = init_val;
    }
    while (!SatisfyCons(var, ConstraintType::MC_MIXED) && var->value != init_val) {
      var->value += var->align;
    }
  )";
}

std::string GenMulticoreTiling() {
  std::string codes = R"(
bool AxesReorderSolver::MulticoreTiling(bool block_loop_auto_tune, bool enable_workload_balance) {
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyMCCons()) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the first check, input: %s.",
            input_.DebugString().c_str());
    return false;
  }
  if (!MulticoreTilingCore(block_loop_auto_tune)) {
    OP_LOGW(OP_NAME, "Multicore tiling core calculation failed in the core check, block_loop_auto_tune: %d, input: %s.",
            block_loop_auto_tune, input_.DebugString().c_str());
    return false;
  }
)";
  return codes + GenWorkLoadBalance() + "  return true;\n}\n";
}

} // namespace att
