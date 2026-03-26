/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_H_
#define COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_H_
#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>
#include <algorithm>
#include "common/checker.h"
#include "ge_common/ge_api_types.h"
#include "ascgen_log.h"

namespace att {
using char_t = char;
// AUTOFUSE_FLAGS
constexpr char_t kExperimentalAutofusionEnablePGO[] = "autofuse_enable_pgo";
// AUTOFUSE_DFX_FLAGS
constexpr char_t kExperimentalAutofusionAttTilingAlgorithm[] = "autofuse_att_algorithm";
constexpr char_t kExperimentalAutofusionAttEnableSmallShapeStrategy[] = "att_enable_small_shape_strategy";//bool
constexpr char_t kExperimentalAutofusionAttUbThreshold[] = "att_ub_threshold";
constexpr char_t kExperimentalAutofusionAttCorenumThreshold[] = "att_corenum_threshold";
constexpr char_t kExperimentalAutofusionAttEnableMulticoreUBTradeoff[] = "att_enable_multicore_ub_tradeoff";//bool
constexpr char_t kExperimentalAutofusionAttProfiling[] = "att_profiling";//bool
constexpr char_t kExperimentalAutofusionAttSolutionAccuracyLevel[] = "att_accuracy_level";
constexpr char_t kExperimentalAutofusionEnableTilingCache[] = "autofuse_enable_tiling_cache";//bool
constexpr char_t kExperimentalAutofusionEnablePgoOptAlgo[] = "autofuse_pgo_algo";
constexpr char_t kExperimentalAutofusionEnablePgoStepMax[] = "autofuse_pgo_step_max";
// 用于强制模板选择，不对外开放
constexpr char_t kExperimentalAutofusionAttScheduleResult[] = "force_schedule_result";
constexpr char_t kExperimentalAutofusionAttTilingCase[] = "force_tiling_case";
constexpr char_t kExperimentalAutofusionAttForceOpName[] = "force_template_op_name";

template <typename T>
class AutoFuseConfigValue {
 public:
  AutoFuseConfigValue(const T &&t, const std::vector<T> &&valid_range) : default_val_(t), valid_range_(valid_range) {}
  Status SetVal(const T &t) {
    bool valid = false;
    GE_ASSERT_SUCCESS(IsValid(t, valid));
    if (valid) {
      val_ = t;
    } else {
      val_ = default_val_;
    }
    return ge::SUCCESS;
  }
  T GetVal() const {
    return val_;
  }
  Status IsValid(const T &input, bool &is_valid) {
    if constexpr (std::is_integral_v<T>) {
      constexpr int32_t kRangeSize = 2;
      GE_ASSERT_TRUE(valid_range_.size() == kRangeSize,
                     "Error: valid_range must have exactly two elements for int type.");
      int min_val = std::min(valid_range_[0], valid_range_[1]);
      int max_val = std::max(valid_range_[0], valid_range_[1]);
      is_valid = (input >= min_val) && (input <= max_val);
    } else if constexpr (std::is_same_v<T, std::string>) {
      if (valid_range_.empty()) { // 未设置范围表示无范围约束
        is_valid = true;
        return ge::SUCCESS;
      }
      is_valid = std::find(valid_range_.begin(), valid_range_.end(), input) != valid_range_.end();
    }
    return ge::SUCCESS;
  }
 private:
  T val_;
  T default_val_;
  std::vector<T> valid_range_;
};

class AutoFuseConfigBase {
 public:
  AutoFuseConfigBase() = default;
  virtual ~AutoFuseConfigBase() = default;
  virtual Status Init() {
    return ge::GRAPH_SUCCESS;
  };
};

class FusionStrategySolverConfig : AutoFuseConfigBase {
 public:
  FusionStrategySolverConfig() = default;
  ~FusionStrategySolverConfig() override = default;
  uint32_t max_fuse_rounds = 10U;  // 尝试融合的最大次数
  int64_t max_proximity = 64;      // 融合节点里原始节点的最小排序和最大排序差值，较大可能会导致内存峰值增加
  size_t max_fusion_size = 64U;    // 融合节点里原始节点最大个数
};

class LoweringStrategyConfig : AutoFuseConfigBase {
 public:
  LoweringStrategyConfig() = default;
  ~LoweringStrategyConfig() override = default;
  uint64_t max_fused_loop_ops{64};       // loop融合循环节点的最大loop ops数
  int64_t max_fused_loop_loads{4};      // loop融合循环节点的最大load数
  int64_t max_k_for_vectorize_mm{256};  // 在n=1时，k小于等于该值，则触发将mm转换为mul+reduce的vector计算
};

class AttStrategyConfig : AutoFuseConfigBase {
 public:
  AttStrategyConfig() = default;
  ~AttStrategyConfig() override = default;
  Status SetEnvVal(std::unordered_map<std::string, std::string> &merged_configs);
  Status Init() override;
  Status Reset();
  std::string tiling_algorithm{};  // ATT tiling选择算法(范围：AxesReorder|HighPerf|Golden)
  int64_t max_iter_num{1000};       // tiling求解最大迭代次数(过低可能会导致求不出解，过高可能求解时间过长，范围：1-2000)
  int64_t solution_accuracy_level{1L};  // 求解的精度，级别越高表示求解精度越高(Kernel性能越好)，当前范围仅有0-1，默认为1
  std::string force_tiling_case; // 强制选择的tiling case
  int64_t force_schedule_result{-1L}; // 强制选择的schedule result
  std::string force_template_op_name; // 指定强制模板选择的op名
  std::string enable_small_shape_strategy{"false"};  // 是否开启小shape快速求解策略(false:不开启，true:开启)
  int64_t ub_threshold{20};  // ub利用率阈值，百分比，范围0-100，如果超过阈值，考虑多核和ub平衡
  int64_t corenum_threshold{40};  // 核数利用率阈值，百分比，范围0-100，如果超过少于阈值，ub停止增加，平衡多核占用
  std::string enable_multicore_ub_tradeoff{"false"}; // 是否开启多核ub权衡(false:不开启，true:开启)
  std::string att_profiling{"false"}; // 是否开启att profiling(false:不开启，true:开启)
  std::string enable_tiling_cache{"true"}; // 是否开启tiling缓存(false:不开启，true:开启)
  // 环境变量是否设置，设置了为true，否则为false
  bool set_env_tiling_algorithm{false};
  bool set_env_solution_accuracy_level{false};
  bool set_env_ub_threshold{false};
  bool set_env_corenum_threshold{false};
  bool set_env_enable_small_shape_strategy{false};
  bool set_env_enable_multicore_ub_tradeoff{false};
  bool set_env_att_profiling{false};
  bool set_env_enable_tiling_cache{false};
  // 用于强制模板选择，不对外开放
  bool set_force_tiling_case{false};
  bool set_force_schedule_result{false};
  bool set_force_template_op_name{false};

 private:
  bool initialized_ = false;
};

class PgoStrategyConfig : AutoFuseConfigBase {
 public:
  PgoStrategyConfig() = default;
  ~PgoStrategyConfig() override = default;
  Status SetEnvVal(std::unordered_map<std::string, std::string> &merged_configs);
  Status Init() override;

  std::string enable_autofuse_pgo{"false"}; // 是否开启pgo(false:不开启，true:开启)
  std::string autofuse_pgo_algo_select{"core_select"}; // pgo 调优算法(core_select:控核，pruning:剪枝)
  int64_t autofuse_pgo_algo_step_max{16};
  bool set_env_enable_autofuse_pgo{false};
  bool set_env_autofuse_pgo_algo_select{false};
  bool set_env_autofuse_pgo_algo_step_max{false};
  bool is_first_init{true};
};

class AutoFuseConfig {
 public:
  static const AutoFuseConfig &Config();
  static AutoFuseConfig &MutableConfig();

  AutoFuseConfig(const AutoFuseConfig &) = delete;
  AutoFuseConfig &operator=(const AutoFuseConfig &) = delete;
  AutoFuseConfig(AutoFuseConfig &&) = delete;
  AutoFuseConfig &operator=(AutoFuseConfig &&) = delete;
  static const LoweringStrategyConfig &GetLoweringConfig() {
    return Config().lowering_strategy_config_;
  }
  static LoweringStrategyConfig &MutableLoweringConfig() {
    return MutableConfig().lowering_strategy_config_;
  }
  static const FusionStrategySolverConfig &GetFusionStrategySolverConfig() {
    return Config().fusion_strategy_solver_;
  }
  static FusionStrategySolverConfig &MutableFusionStrategySolver() {
    return MutableConfig().fusion_strategy_solver_;
  }
  static const AttStrategyConfig &GetAttStrategyConfig() {
    return Config().att_strategy_config_;
  }
  static AttStrategyConfig &MutableAttStrategyConfig() {
    return MutableConfig().att_strategy_config_;
  }
  static const PgoStrategyConfig &GetPgoStrategyConfig() {
    return Config().pgo_strategy_config_;
  }
  static PgoStrategyConfig &MutablePgoStrategyConfig() {
    return MutableConfig().pgo_strategy_config_;
  }

 private:
  AutoFuseConfig() = default;
  static AutoFuseConfig &Instance();
  LoweringStrategyConfig lowering_strategy_config_;
  FusionStrategySolverConfig fusion_strategy_solver_;
  AttStrategyConfig att_strategy_config_;
  PgoStrategyConfig pgo_strategy_config_;
};

}  // namespace att

#endif  // COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_H_
