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
// Section 2: Class Definition
// ============================================================================

std::string GenDestructor() {
  std::string codes;
  codes += "  ~AxesReorderSolver() {}\n";
  return codes;
}

std::string GenAxesReorderSolverDefaultPrivateFuncDefine() {
  return R"(
private:
  inline bool FindNextUpperBlockDim(const uint32_t block_dim, uint32_t &next_lower_block_dim) const;
  inline bool FindNextLowerBlockDim(const uint32_t block_dim, uint32_t &next_upper_block_dim) const;
  inline void SaveInputTilingVars(TilingVariable *tiling_vars, TilingVariable *pure_mc_vars,
                                  TilingVariable *local_buffer_vars) const;
  inline void RestoreInputTilingVars(const TilingVariable *tiling_vars, const TilingVariable *pure_mc_vars,
                                     const TilingVariable *local_buffer_vars) const;
  inline bool WorkloadBalance();
  inline bool IsSatisfyCons(TilingVariable *var, int64_t value, ConstraintType cons_type, double ratio = 1.0);
  inline bool BinaryFindUpperBoundSatisfiedUBLimit(TilingVariable *var, int64_t lower_bound,
                                                   int64_t &result);
  inline bool BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable *var, uint32_t idx, int64_t lower_bound,
                                                           int64_t &result);
  inline std::pair<int64_t, int64_t> BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable *var, uint32_t idx,
                                                                          int64_t lower_bound);
  // Multi-variable versions of binary search functions
  bool BinaryFindUpperBoundSatisfiedUBLimit(TilingVariable **vars, uint32_t *idxs, size_t count, int64_t lower_bound,
                                            int64_t &result);
  bool BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable **vars, uint32_t *idxs, size_t count,
                                                    int64_t lower_bound, int64_t &result);
  std::pair<int64_t, int64_t> BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable **vars, uint32_t *idxs, size_t count,
                                                                   int64_t lower_bound);
  // Multi-variable version with per-variable upper bounds
  // Returns: {value_for_vars[0], value_for_vars[1], ...}
  // Each value is the binary search result (left) clipped to that variable's upper bound
  // When variables have different upper bounds, this enables non-symmetric solutions
  // Example: if upper_bounds={32, 128} and left=64, returns {32, 64}
  //          where vars[0] (a-axis) gets 32 (smaller), vars[1] (b-axis) gets 64 (larger)
  std::pair<int64_t, int64_t> BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable **vars, uint32_t *idxs,
                                                                    size_t count, int64_t lower_bound,
                                                                    const int64_t *upper_bounds);
  bool TuneNotailVar(TilingVariable *var);
  bool InitLocalBufferVars();
  bool InitMulticoreVars();
  bool ProcessMCAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num);
  bool ProcessSingleAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num);
  bool MulticoreTiling(bool block_loop_auto_tune, bool enable_workload_balance=false);
  bool MulticoreTilingCore(bool block_loop_auto_tune);
)";
}

// Helper function: Generate basic equal-order function declarations
std::string GenEqualOrderBasicFuncDeclarations(bool enable_equal_order_tiling) {
  std::string codes;
  std::string equal_order_param = enable_equal_order_tiling ? ", const bool enable_equal_order" : "";
  codes += "  inline bool GetTiling(const bool is_tuning, const bool block_loop_auto_tune, const bool "
           "enable_workload_balance" + equal_order_param + ");\n";
  codes += "  inline bool GetMaxBlockDimTiling(const uint32_t block_dim" + equal_order_param + ");\n";
  codes += "  inline bool AutoTuning(const bool is_trade_off" + equal_order_param + ");\n";
  codes += "  inline void FindBetterSolutionByUpperBlockDim(double next_upper_perf, uint32_t next_upper_block_dim" +
           equal_order_param + ");\n";
  codes += "  inline void FindBetterSolutionByLowerBlockDim(double next_lower_perf, uint32_t next_lower_block_dim" +
           equal_order_param + ");\n";
  return codes;
}

// Helper function: Generate local buffer tiling function declarations
std::string GenLocalBufTilingFuncDeclarations(bool enable_equal_order_tiling) {
  std::string codes;
  std::string equal_order_param = enable_equal_order_tiling ? ", const bool enable_equal_order" : "";
  std::string equal_order_tiling_param = enable_equal_order_tiling ? "const bool enable_equal_order" : "";
  codes += "  bool NaiveLocalBufTiling(" + equal_order_tiling_param + ");\n";
  codes += "  bool BinaryLocalBufTiling(" + equal_order_tiling_param + ");\n";
  codes += "  bool BinaryLocalBufTilingCore(const std::vector<bool> &solved_axes);\n";
  codes += "  void ApplyPromptAlign(TilingVariable *var);\n";
  codes += "  bool LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune" + equal_order_param + ");\n";
  return codes;
}

// Helper function: Generate utility function declarations
std::string GenUtilityFuncDeclarations() {
  std::string codes;
  codes += "  bool DecreaseUntilSatisfied(TilingVariable *var, ConstraintType cons_type, "
           "    std::function<bool(TilingVariable *)> tune_func);\n";
  codes += "  bool ShrinkBoundaryUntilSatisfied(TilingVariable *var, int64_t boundary, ConstraintType cons_type);\n";
  return codes;
}

// Helper function: Generate equal priority axes function declarations
std::string GenEqualPriorityAxesFuncDeclarations(bool enable_equal_order_tiling) {
  if (!enable_equal_order_tiling) {
    return "";
  }
  std::string codes;
  codes += "  // Equal priority axes support functions\n";
  codes += "  bool IdentifyEqualPriorityAxes(const uint32_t axes_num, uint32_t *axis_idx, int64_t &min_upper_bound, "
           "    int64_t &max_aligned);\n";
  codes += "  bool BinarySearchEqualPriorityAxes(uint32_t axis_num, uint32_t *axis_idx, int64_t lower_bound, "
           "    int64_t upper_bound);\n";
  codes += "  bool IterativeSolveEqualPriorityAxes(const uint32_t axis_num, uint32_t *axis_idx, int64_t lower_bound, "
           "    int64_t upper_bound);\n";
  codes += "  bool BinarySearchWithAlignment(TilingVariable **vars, uint32_t var_num, int64_t lower_bound, "
           "    int64_t upper_bound, int64_t align, const char *log_prefix);\n";
  codes += "  bool TuneNoTail(TilingVariable **vars, uint32_t id);\n";
  return codes;
}

// Helper function: Generate dual threshold function declarations
std::string GenDualThresholdFuncDeclarations(bool enable_equal_order_tiling) {
  if (!enable_equal_order_tiling) {
    return "";
  }
  std::string codes;
  codes += "  // NaiveLocalBufTiling dual threshold support functions\n";
  codes += "  bool SolveEqualPriorityAxesWithDualThreshold(uint32_t *axis_idx, int64_t low_bound, "
           "    int64_t upper_bound, std::vector<bool> &solved_axes);\n";
  codes += "  bool BinarySearchEqualPriorityAxesWithDualThreshold(TilingVariable **vars, int64_t lower_bound, "
           "    int64_t upper_bound, int64_t &upper_ub_a, int64_t &upper_ub_b);\n";
  codes += "  bool ProcessNonMCAxes(TilingVariable **vars, uint32_t *axis_idx, std::vector<bool> &solved_axes);\n";
  codes += "  bool ProcessSingleMCAxis(TilingVariable **vars, bool first_is_mc, uint32_t *axis_idx, "
           "    std::vector<bool> &solved_axes);\n";
  codes += "  bool ProcessDualMCAxes(TilingVariable **vars, uint32_t *axis_idx, std::vector<bool> &solved_axes);\n";
  codes += "  bool FinalizeEqualPriorityAxes(uint32_t *axis_idx, std::vector<bool> &solved_axes) const;\n";
  return codes;
}

// Helper function: Generate DualAxesInfo struct and three-phase framework declarations
std::string GenThreePhaseFrameworkDeclarations(bool enable_equal_order_tiling) {
  if (!enable_equal_order_tiling) {
    return "";
  }
  std::string codes;
  codes += "  // ProcessDualMCAxes helper functions\n";
  codes += "  struct DualAxesInfo {\n";
  codes += "    TilingVariable *var_a{nullptr};\n";
  codes += "    TilingVariable *var_b{nullptr};\n";
  codes += "    uint32_t idx_a{0U};\n";
  codes += "    uint32_t idx_b{1U};\n";
  codes += "    int64_t upper_bound_a{0L};\n";
  codes += "    int64_t upper_bound_b{0L};\n";
  codes += "    int64_t low_bound_both{0L};\n";
  codes += "    int64_t upper_bound_both{0L};\n";
  codes += "  };\n";
  codes += "  // Three-phase algorithm framework\n";
  codes += "  DualAxesInfo InitializeDualAxesInfo(TilingVariable **vars, uint32_t *axis_idx) const;\n";
  codes += "  bool ExecuteStep1_FindUBLimit(const DualAxesInfo &info, int64_t &low_bound_both_ub, \n"
           "    int64_t &low_bound_b_ub, int64_t &range_a_high, int64_t &range_b_high);\n";
  codes += "  bool ExecuteStep2_FindUBThreshold(const DualAxesInfo &info, int64_t range_a_low, \n"
           "    int64_t range_a_high, int64_t &range_a_low_out, int64_t &range_b_low_out);\n";
  codes += "  bool ExecuteStep3_FindCoreNumTileSize(const DualAxesInfo &info, int64_t range_a_low, \n"
           "    int64_t range_a_high, int64_t range_b_high, TilingVariable **vars_final);\n";
  return codes;
}

// Helper function: Generate protected section declarations
std::string GenProtectedFuncDeclarations() {
  return R"(protected:
  bool SatisfyCons(ConstraintType cons_type);
  bool SatisfyCons(TilingVariable *var, ConstraintType cons_type, double ratio = 1.0);
  bool SatisfyMCCons();
)";
}

std::string GenAxesReorderSolverPrivateFuncDefine(bool enable_equal_order_tiling) {
  std::string codes = GenAxesReorderSolverDefaultPrivateFuncDefine();
  codes += GenEqualOrderBasicFuncDeclarations(enable_equal_order_tiling);
  codes += GenLocalBufTilingFuncDeclarations(enable_equal_order_tiling);
  codes += GenUtilityFuncDeclarations();
  codes += GenEqualPriorityAxesFuncDeclarations(enable_equal_order_tiling);
  codes += GenDualThresholdFuncDeclarations(enable_equal_order_tiling);
  codes += GenThreePhaseFrameworkDeclarations(enable_equal_order_tiling);
  codes += GenProtectedFuncDeclarations();
  return codes;
}

std::string GenAxesReorderSolver(bool enable_equal_order_tiling) {
  std::string codes;
  codes += "class AxesReorderSolver {\n";
  codes += "public:\n";
  codes += "  explicit AxesReorderSolver(const AxesReorderSolverInput &input) : input_(input) {}\n";
  codes += GenDestructor();
  std::string run_decl =
      "  bool Run(const bool is_trade_off, const bool block_loop_auto_tune, const bool enable_auto_tune, const bool "
      "enable_equal_order = false);\n";
  run_decl += "  bool IsEmptyTensor() const { return is_empty_tensor_; }\n";
  std::string function_define = R"(protected:
  virtual bool CalUsedCoreNum(double &used_core_num) = 0;
  virtual bool CalRealUsedCoreNum(int64_t &used_corenum) = 0;
  virtual bool SatisfyThresholdUBSize() = 0;
  virtual double GetPerf() = 0;
  virtual bool SatisfyUBSizeCacheLine(uint32_t idx) = 0;
  AxesReorderSolverInput input_;
  bool is_empty_tensor_{false};
)";
  function_define = run_decl + function_define;
  codes += function_define;
  codes += GenAxesReorderSolverPrivateFuncDefine(enable_equal_order_tiling);
  codes += "};\n\n";
  return codes;
}

} // namespace att
