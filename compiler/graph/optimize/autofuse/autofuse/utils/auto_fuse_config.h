/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTO_FUSE_CONFIG_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTO_FUSE_CONFIG_H_
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include "base/base_types.h"
#include "common/checker.h"
#include "graph/types.h"
#include "mmpa/mmpa_api.h"

#include "common/autofuse_base_type.h"

namespace ge {
namespace autofuse {
constexpr int AUTOFUSE_FLAGS_ENV_VALUES = 2;
constexpr int AUTOFUSE_DFX_FLAGS_ENV_VALUES = 2;
inline bool ReadBoolEnv(const char *env_name, const bool default_value) {
  const char *env_value = std::getenv(env_name);
  if (env_value == nullptr) {
    return default_value;
  }
  return std::string(env_value) == "1";
}

inline std::vector<std::string> Split(const std::string &s, char delimeter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(s);
  while (std::getline(token_stream, token, delimeter)) {
    tokens.push_back(token);
  }
  return tokens;
}

inline void ReadAutoFuseEnv(std::unordered_map<std::string, std::string> &all_flags) {
  const char_t *autofuse_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_AUTOFUSE_FLAGS, autofuse_env);
  if ((autofuse_env == nullptr) || (strlen(autofuse_env) == 0U)) {
    return;
  }
  auto flag_parts = Split(autofuse_env, ';');
  for (const auto &part : flag_parts) {
    auto kv = Split(part, '=');
    if (kv.size() == AUTOFUSE_FLAGS_ENV_VALUES) {
      all_flags[kv[0]] = kv[1];
    }
  }
  return;
}

inline void ReadAutoFuseDfxEnv(std::unordered_map<std::string, std::string> &all_flags) {
  const char_t *autofuse_dfx_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_AUTOFUSE_DFX_FLAGS, autofuse_dfx_env);
  if ((autofuse_dfx_env == nullptr) || (strlen(autofuse_dfx_env) == 0U)) {
    return;
  }
  auto flag_parts = Split(autofuse_dfx_env, ';');
  for (const auto &part : flag_parts) {
    auto kv = Split(part, '=');
    if (kv.size() == AUTOFUSE_DFX_FLAGS_ENV_VALUES) {
      all_flags[kv[0]] = kv[1];
    }
  }
  return;
}

inline std::string ReadStringEnv(const char *env_name, const std::string default_value) {
  const char *env_value = std::getenv(env_name);
  if (env_value == nullptr) {
    return default_value;
  }
  return std::string(env_value);
}

inline std::unordered_set<std::string> ReadImprovePrecisionBlacklist(std::string &input) {
  std::unordered_set<std::string> tokens;
  // 移除末尾的标点符号（如果存在）
  if (!input.empty() && std::ispunct(input.back())) {
    input.pop_back();
  }

  // 使用 stringstream 分割字符串
  size_t start = 0U;
  size_t end = input.find(',');
  while (end != std::string::npos) {
    std::string token = input.substr(start, end - start);
    tokens.insert(token);  // 直接插入，无空格处理
    start = end + 1U;
    end = input.find(',', start);
  }
  // 添加最后一个token
  tokens.insert(input.substr(start));
  return tokens;
}

inline std::unordered_set<std::string> ReadImprovePrecisionBlacklist(const char *env_name) {
  std::string input = ReadStringEnv(env_name, "");
  std::unordered_set<std::string> tokens;
  
  // 移除末尾的标点符号（如果存在）
  if (!input.empty() && std::ispunct(input.back())) {
    input.pop_back();
  }
  
  // 使用 stringstream 分割字符串
  size_t start = 0;
  size_t end = input.find(',');
  while (end != std::string::npos) {
    std::string token = input.substr(start, end - start);
    tokens.insert(token);  // 直接插入，无空格处理
    start = end + 1U;
    end = input.find(',', start);
  }
  // 添加最后一个token
  tokens.insert(input.substr(start));
  return tokens;
}

inline void Trim(std::string &str) {
  str.erase(0, str.find_first_not_of(" \t\r\n"));
  str.erase(str.find_last_not_of(" \t\r\n") + 1);
}

inline void ParseSkipNodeNamesConfig(const std::string &config_path, 
                                    std::unordered_set<std::string> &skip_node_types,
                                    std::unordered_set<std::string> &skip_node_names) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    GELOGW("Failed to open skip node names config file: %s", config_path.c_str());
    return;
  }
  
  std::string line;
  enum class ParseSection {
    NONE,
    BY_NODE_TYPE,
    BY_NODE_NAME
  };
  ParseSection current_section = ParseSection::NONE;
  
  while (std::getline(file, line)) {
    Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    
    if (line[0] == '[' && line.back() == ']') {
      std::string section_name = line.substr(1, line.length() - 2);
      Trim(section_name);
      if (section_name == "ByNodeType") {
        current_section = ParseSection::BY_NODE_TYPE;
      } else if (section_name == "ByNodeName") {
        current_section = ParseSection::BY_NODE_NAME;
      } else {
        current_section = ParseSection::NONE;
      }
      continue;
    }
    
    if (current_section == ParseSection::BY_NODE_TYPE) {
      skip_node_types.insert(line);
      GELOGD("Add skip node type: %s", line.c_str());
    } else if (current_section == ParseSection::BY_NODE_NAME) {
      skip_node_names.insert(line);
      GELOGD("Add skip node name: %s", line.c_str());
    }
  }
  
  file.close();
  GELOGI("Loaded skip node config from %s, skip types: %zu, skip names: %zu", 
         config_path.c_str(), skip_node_types.size(), skip_node_names.size());
}

class AutoFuseConfig {
 public:
  struct FusionStrategySolverConfig {
    uint32_t max_fuse_rounds = 10U;  // 尝试融合的最大次数
    int64_t max_proximity =
        std::numeric_limits<int64_t>::max();  // 融合节点里原始节点的最小排序和最大排序差值，较大可能会导致内存峰值增加
    uint64_t max_fusion_size = 64U;           // 融合节点里原始节点最大个数
    uint32_t max_input_nums_after_fuse = 8U;  // 融合后节点的最大输入个数
    uint32_t max_op_name_len = 140U;          // ascbackend和fusedascbackend节点名最大长度
    int64_t max_output_memory_size_after_fusion = 13LL * 1024 * 1024 * 1024;  // 节点融合后输出内存最大值，暂时设置为13G
    AutoFuseFwkType fwk_type = AutoFuseFwkType::kGe;
    std::unordered_set<std::string> improve_precision_blacklist;
    size_t max_reduce_can_fuse_elementwise_nums = 3U;  // Reduce可以向后融合elementwise节点的最大个数
  };

  struct LoweringStrategyConfig {
    uint64_t max_fused_loop_ops{64U};   // loop融合循环节点的最大loop ops数
    size_t max_buffer_readers{4U};    // kernel box最大允许的读取node数量，超过该数量则会终止融合
    size_t max_k_for_vectorize_mm{32U};  // 在n=1时，k小于等于该值，则触发将mm转换为mul+reduce的vector计算
    size_t recomputation_threshold{1U};  // 单输出节点重计算阈值，节点输出output个数大于该值将realize
    bool experimental_lowering_reduce{false};
    bool experimental_lowering_concat{false};
    bool experimental_lowering_split{false};
    bool experimental_lowering_slice{false};
    bool experimental_lowering_transpose{false};
    bool experimental_lowering_gather{false};
    bool experimental_lowering_matmul{false};
    bool experimental_disable_lifting{false};
    std::unordered_set<std::string> skip_node_types;     // 需要跳过lowering的节点类型
    std::unordered_set<std::string> skip_node_names;     // 需要跳过lowering的节点名称
  };

 public:
  static const AutoFuseConfig &Config();
  static AutoFuseConfig &MutableConfig();

  AutoFuseConfig(const AutoFuseConfig &) = delete;
  AutoFuseConfig &operator=(const AutoFuseConfig &) = delete;
  AutoFuseConfig(AutoFuseConfig &&) = delete;
  AutoFuseConfig &operator=(AutoFuseConfig &&) = delete;

  [[nodiscard]] const static LoweringStrategyConfig &LoweringConfig() {
    return Config().lowering_strategy_config_;
  }

  static LoweringStrategyConfig &MutableLoweringConfig() {
    return MutableConfig().lowering_strategy_config_;
  }

  [[nodiscard]] const FusionStrategySolverConfig &GetFusionStrategySolver() const {
    return fusion_strategy_solver_;
  }

  FusionStrategySolverConfig &GetMutableFusionStrategySolver() {
    return fusion_strategy_solver_;
  }

 private:
  AutoFuseConfig() {
    (void)UpdateAutoFuseConfigByEnv();
    (void)UpdateAutoFuseDfxConfigByEnv();
  }
  static AutoFuseConfig &Instance();

  void UpdateImprovePrecisionBlacklist(std::unordered_map<std::string, std::string> &all_flags) {
    auto precision_blacklist = all_flags.find("--autofuse_enhance_precision_blacklist");
    std::unordered_set<std::string> improve_precision_blacklist;
    if (precision_blacklist != all_flags.end()) {
      improve_precision_blacklist = ReadImprovePrecisionBlacklist(precision_blacklist->second);
    }
    this->fusion_strategy_solver_.improve_precision_blacklist.insert(improve_precision_blacklist.begin(),
                                                                     improve_precision_blacklist.end());
  }

  void LoweringEnableConfigUpdate(bool &enableSwitch, const std::string &nodeName,
                                  const std::unordered_map<std::string, std::string> &all_flags) const {
    auto enable_lowering_node_types = all_flags.find("--autofuse_enable_pass");
    if (enable_lowering_node_types != all_flags.end()) {
      std::vector<std::string> autofuse_lowering_node_types = Split(enable_lowering_node_types->second, ',');
      enableSwitch = std::find(autofuse_lowering_node_types.begin(), autofuse_lowering_node_types.end(), nodeName) !=
                     autofuse_lowering_node_types.end();
      auto disable_lowering_node_types = all_flags.find("--autofuse_disable_pass");
      if (disable_lowering_node_types != all_flags.end()) {
        autofuse_lowering_node_types = Split(disable_lowering_node_types->second, ',');
        bool disableSwitch = std::find(autofuse_lowering_node_types.begin(), autofuse_lowering_node_types.end(),
                                       nodeName) != autofuse_lowering_node_types.end();
        if (disableSwitch && enableSwitch) {
          GELOGW("%s for --autofuse_disable_pass and --autofuse_enable_pass can not be set", nodeName.c_str());
          enableSwitch = false;
        }
      }
    }
  }

  void LoweringRecomputationThresholdConfigUpdate(size_t &recomputation_threshold,
    const std::unordered_map<std::string, std::string> &all_flags) const {
    constexpr int64_t recomputation_max = 255L;
    auto recomputation_threshold_flag = all_flags.find("--recomputation_threshold");
    if (recomputation_threshold_flag != all_flags.end()) {
      std::stringstream ss(recomputation_threshold_flag->second);
      int64_t value;
      ss >> value;
      if (ss.fail() || !ss.eof() || value > recomputation_max || value < 0) {
        GELOGW("Recomputation threshold value is out of range");
        recomputation_threshold = 1U;
        return;
      }
      recomputation_threshold = static_cast<size_t>(value);
      return;
    }
    recomputation_threshold = 1U;
  }

  void UpdateMaxFusionSizeConfig(const std::unordered_map<std::string, std::string> &all_flags) {
    auto max_fusion_size_flag = all_flags.find("--max_fusion_size");
    if (max_fusion_size_flag != all_flags.end()) {
      const std::string &input = max_fusion_size_flag->second;
      std::string numeric_part;

      for (char c : input) {
        if (std::isdigit(c)) {
          numeric_part += c;
        } else {
          break;
        }
      }

      if (numeric_part.empty()) {
        GELOGW("Invalid max_fusion_size value: %s. Valid range: 0-18446744073709551615", input.c_str());
        return;
      }

      if (numeric_part.length() < input.length()) {
        char next_char = input[numeric_part.length()];
        if (next_char != ';' && next_char != ',') {
          GELOGW("Invalid max_fusion_size value: %s. Valid range: 0-18446744073709551615", input.c_str());
          return;
        }
      }

      std::stringstream ss(numeric_part);
      uint64_t value;
      ss >> value;
      if (ss.fail()) {
        GELOGW("Invalid max_fusion_size value: %s. Valid range: 0-18446744073709551615", input.c_str());
        return;
      }
      std::string remaining;
      ss >> remaining;
      if (!remaining.empty()) {
        GELOGW("Invalid max_fusion_size value: %s. Valid range: 0-18446744073709551615", input.c_str());
        return;
      }
      this->fusion_strategy_solver_.max_fusion_size = value;
      this->lowering_strategy_config_.max_fused_loop_ops = value;
    }
  }

  void UpdateAutoFuseConfigByEnv() {
    std::unordered_map<std::string, std::string> all_flags;
    (void)ReadAutoFuseEnv(all_flags);
    bool enable_lowering_concat = false;
    bool enable_lowering_reduce = false;
    bool enable_lowering_slice = false;
    bool enable_lowering_split = false;
    bool enable_lowering_transpose = false;
    bool enable_lowering_gather = false;
    bool enable_lowering_matmul = false;
    size_t recomputation_threshold = 1U;

    LoweringEnableConfigUpdate(enable_lowering_concat, "concat", all_flags);
    LoweringEnableConfigUpdate(enable_lowering_reduce, "reduce", all_flags);
    LoweringEnableConfigUpdate(enable_lowering_slice, "slice", all_flags);
    LoweringEnableConfigUpdate(enable_lowering_split, "split", all_flags);
    LoweringEnableConfigUpdate(enable_lowering_transpose, "transpose", all_flags);
    LoweringEnableConfigUpdate(enable_lowering_gather, "gather", all_flags);
    LoweringEnableConfigUpdate(enable_lowering_matmul, "matmul", all_flags);
    LoweringRecomputationThresholdConfigUpdate(recomputation_threshold, all_flags);
    // remove old env soon
    this->lowering_strategy_config_.experimental_lowering_concat = enable_lowering_concat;
    this->lowering_strategy_config_.experimental_lowering_split = enable_lowering_split;
    this->lowering_strategy_config_.experimental_lowering_reduce = enable_lowering_reduce;
    this->lowering_strategy_config_.experimental_lowering_slice = enable_lowering_slice;
    this->lowering_strategy_config_.experimental_lowering_transpose = enable_lowering_transpose;
    this->lowering_strategy_config_.experimental_lowering_gather = enable_lowering_gather;
    this->lowering_strategy_config_.experimental_lowering_matmul = enable_lowering_matmul;
    this->lowering_strategy_config_.recomputation_threshold = recomputation_threshold;
    UpdateImprovePrecisionBlacklist(all_flags);
    UpdateMaxFusionSizeConfig(all_flags);
    return;
  }

  void UpdateAutoFuseDfxConfigByEnv() {
    std::unordered_map<std::string, std::string> all_flags;
    (void)ReadAutoFuseDfxEnv(all_flags);
    bool disable_lifting =
        all_flags.find("--disable_lifting") != all_flags.end() && all_flags["--disable_lifting"] == "true";
    this->lowering_strategy_config_.experimental_disable_lifting = disable_lifting;
    
    auto skip_node_names_cfg = all_flags.find("--skip_node_names_cfg");
    if (skip_node_names_cfg != all_flags.end()) {
      ParseSkipNodeNamesConfig(skip_node_names_cfg->second, 
                               this->lowering_strategy_config_.skip_node_types,
                               this->lowering_strategy_config_.skip_node_names);
    }
    return;
  }
  LoweringStrategyConfig lowering_strategy_config_;
  FusionStrategySolverConfig fusion_strategy_solver_;
};
} // namespace autofuse
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_UTILS_AUTO_FUSE_CONFIG_H_
