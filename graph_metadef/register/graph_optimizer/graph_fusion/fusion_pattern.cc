/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"

namespace fe {
const uint32_t kFuzzyOutIndex = 0xFFFFFFFF;
constexpr size_t MAX_LOG_LENGTH = 900;
#define FE_PATTERN_ERROR_RETURN_IF(condition, ...) \
  do {                                             \
    if (condition) {                               \
      SetError();                                  \
      GELOGW(__VA_ARGS__);                         \
      return *this;                                \
    }                                              \
  } while (0)

#define FE_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                         \
    try {                                      \
      exec_expr0;                              \
    } catch (...) {                            \
      GELOGW("Make shared failed");            \
      exec_expr1;                              \
    }                                          \
  } while (0)

FusionPattern::FusionPattern(const std::string name) : name_(name), output_(nullptr) {}

FusionPattern::~FusionPattern() {
  for (const auto &ops: ops_) {
    ops->inputs.clear();
    ops->outputs.clear();
  }
  ops_.clear();
  op_map_.clear();
}

/**
 * @ingroup fe
 * @brief set pattern name
 */
FusionPattern &FusionPattern::SetName(const std::string &name) {
  name_ = name;
  return *this;
}

/**
 * @ingroup fe
 * @brief add Op description with unknown number of args
 */
FusionPattern &FusionPattern::AddOpDesc(const std::string &id, const std::initializer_list<std::string> &types,
                                        const bool allow_dumpable, const bool check_unique) {
  return AddOpDesc(id, std::vector<std::string>(types), allow_dumpable, check_unique);
}

/**
 * @ingroup fe
 * @brief add Op description with vector
 */
FusionPattern &FusionPattern::AddOpDesc(const std::string &id, const std::vector<std::string> &types,
                                        const bool allow_dumpable, const bool check_unique) {
  FE_PATTERN_ERROR_RETURN_IF(id.empty(), "ID cannot be empty.");

  FE_PATTERN_ERROR_RETURN_IF(GetOpDesc(id) != nullptr, "ID already exists. (id:%s)", id.c_str());

  std::shared_ptr<OpDesc> op;
  FE_MAKE_SHARED(op = std::make_shared<OpDesc>(), return *this);
  FE_PATTERN_ERROR_RETURN_IF(op == nullptr, "new an object failed.");

  op->id = id;
  op->types = types;
  op->repeatable = false;
  op->is_output = false;
  op->is_output_fullmatch = true;
  op->output_size = 0UL;
  op->allow_dumpable = allow_dumpable;
  op->check_unique = check_unique;

  ops_.push_back(op);
  op_map_[id] = op;

  return *this;
}

/**
 * @ingroup fe
 * @brief set input Ops with unknown number of args
 */
FusionPattern &FusionPattern::SetInputs(const std::string &id, const std::initializer_list<std::string> &input_ids) {
  return SetInputs(id, std::vector<std::string>(input_ids));
}

/**
 * @ingroup fe
 * @brief set input Ops with vector
 */
FusionPattern &FusionPattern::SetInputs(const std::string &id, const std::vector<std::string> &input_ids) {
  FE_PATTERN_ERROR_RETURN_IF(id.empty(), "Id cannot be empty.");
  const std::shared_ptr<FusionPattern::OpDesc> op_desc = GetOpDesc(id);
  FE_PATTERN_ERROR_RETURN_IF(op_desc == nullptr, "Id does not exist. (id:%s)", id.c_str());

  op_desc->inputs.clear();

  for (const std::string &input_id : input_ids) {
    const std::shared_ptr<FusionPattern::OpDesc> input_op_desc = GetOpDesc(input_id);
    FE_PATTERN_ERROR_RETURN_IF(input_op_desc == nullptr, "Id does not exist. (id:%s)", input_id.c_str());
    op_desc->inputs.push_back(input_op_desc);
  }

  return *this;
}

/**
 * @ingroup fe
 * @brief set output Ops with vector
 */
FusionPattern &FusionPattern::SetOutputs(const std::string &id, const FusionPattern::OutputMapVecStr &output_map,
                                         bool is_fullmatched) {
  if (id.empty()) {
    GELOGW("Id cannot be empty.");
    return *this;
  }
  const std::shared_ptr<FusionPattern::OpDesc> op_desc = GetOpDesc(id);
  FE_PATTERN_ERROR_RETURN_IF(op_desc == nullptr, "Id does not exist. (id:%s)", id.c_str());
  op_desc->outputs.clear();
  for (auto &iter : output_map) {
    for (const std::string &output_id : iter.second) {
      const std::shared_ptr<FusionPattern::OpDesc> output_op_desc = GetOpDesc(output_id);
      FE_PATTERN_ERROR_RETURN_IF(output_op_desc == nullptr, "Id does not exist. (id:%s)", output_id.c_str());
      if (op_desc->outputs.find(iter.first) == op_desc->outputs.end()) {
        op_desc->outputs[iter.first] = {};
      }
      op_desc->outputs[iter.first].emplace_back(output_op_desc);
      FE_PATTERN_ERROR_RETURN_IF(op_desc->output_size == std::numeric_limits<size_t>::max(),
        "op_desc->output_size has wrapped around.");
      ++op_desc->output_size;
    }
  }
  op_desc->is_output_fullmatch = is_fullmatched;
  return *this;
}
/**
 * @ingroup fe
 * @brief set output Ops with vector
 */
FusionPattern &FusionPattern::SetOutputs(const std::string &id, const FusionPattern::OutputMapStr &output_map,
                                         bool is_fullmatched) {
  if (id.empty()) {
    GELOGW("Id cannot be empty.");
    return *this;
  }
  const std::shared_ptr<FusionPattern::OpDesc> op_desc = GetOpDesc(id);
  FE_PATTERN_ERROR_RETURN_IF(op_desc == nullptr, "Id does not exist. (id:%s)", id.c_str());

  op_desc->outputs.clear();
  for (auto &iter : output_map) {
    const std::string output_id(iter.second);
    const std::shared_ptr<FusionPattern::OpDesc> output_op_desc = GetOpDesc(output_id);
    FE_PATTERN_ERROR_RETURN_IF(output_op_desc == nullptr, "Id does not exist. (id:%s)", output_id.c_str());
    op_desc->outputs[iter.first].emplace_back(output_op_desc);
    FE_PATTERN_ERROR_RETURN_IF(op_desc->output_size == std::numeric_limits<size_t>::max(),
      "op_desc->output_size has wrapped around.");
    ++op_desc->output_size;
  }
  op_desc->is_output_fullmatch = is_fullmatched;
  return *this;
}

/**
 * @ingroup fe
 * @brief set output Op
 */
FusionPattern &FusionPattern::SetOutput(const std::string &id) {
  FE_PATTERN_ERROR_RETURN_IF(id.empty(), "Id cannot be empty.");
  const std::shared_ptr<FusionPattern::OpDesc> op_desc = GetOpDesc(id);
  FE_PATTERN_ERROR_RETURN_IF(op_desc == nullptr, "Id does not exist. (id:%s)", id.c_str());

  op_desc->is_output = true;

  return *this;
}

/**
 * @ingroup fe
 * @brief build pattern and check if error exists
 */
bool FusionPattern::Build() {
  if (has_error_) {
    return false;
  }

  // check whether output node already exists
  for (const std::shared_ptr<OpDesc> op : ops_) {
    if (op->is_output) {
      if (output_ != nullptr) {
        SetError();
        GELOGW("[FusionPattern][Build] Multiple outputs are not supported, (id:%s)", op->id.c_str());
        break;
      }
      output_ = op;
    }
  }

  if (output_ == nullptr) {
    SetError();
    GELOGW("[FusionPattern][Build] Output must be set to a value.");
  }

  return !has_error_;
}

/**
 * @ingroup fe
 * @brief get pattern name
 */
const std::string &FusionPattern::GetName() const { return name_; }
/**
 * @ingroup fe
 * @brief get the OpDesc of input Ops (const)
 */

const std::vector<std::shared_ptr<FusionPattern::OpDesc>> *FusionPattern::GetInputs(
    const std::shared_ptr<FusionPattern::OpDesc> op_desc) {
  if (op_desc == nullptr) {
    return nullptr;
  }
  return &(op_desc->inputs);
}

const FusionPattern::OutputMapDesc &FusionPattern::GetOutputs(const OpDescPtr op_desc) {
  return op_desc->outputs;
}

size_t FusionPattern::GetOutputSize(const OpDescPtr op_desc) {
  return op_desc->output_size;
}

/**
 * @ingroup fe
 * @brief get the OpDesc of output Op
 */
const std::shared_ptr<FusionPattern::OpDesc> FusionPattern::GetOutput() const { return output_; }

/**
 * @ingroup fe
 * @brief print pattern
 */
void FusionPattern::Dump() const {
  std::ostringstream oss;
  oss << std::endl << "Pattern (" << name_ << "):" << std::endl;
  for (const auto &op : ops_) {
    oss << "  " << op->id << ": {";
    for (const std::string &type : op->types) {
      oss << type << ", ";
    }
    oss << "} {";
    for (const auto &input : op->inputs) {
      oss << input->id << ", ";
    }
    oss << "}";

    if (op->is_output) {
      oss << " [output]";
    }

    oss << std::endl;
  }
  size_t len = oss.str().length();
  size_t startIndex = 0;
  size_t recursive_times = 0;
  constexpr int32_t kMaxTurnCount = 10;
  do {
      recursive_times++;
      const int32_t endIndex = static_cast<int32_t>(std::min(startIndex + MAX_LOG_LENGTH, len));
      std::string subStr = oss.str().substr(startIndex, static_cast<size_t>(endIndex - startIndex));
      GELOGD("%s", subStr.c_str());
      startIndex = static_cast<size_t>(endIndex);
  } while (startIndex < len && static_cast<int32_t>(recursive_times) < kMaxTurnCount);
}

/**
 * @ingroup fe
 * @brief get OpDesc based on ID, return nullptr if failed
 */
std::shared_ptr<FusionPattern::OpDesc> FusionPattern::GetOpDesc(const std::string &id) const {
  const auto it = op_map_.find(id);
  if (it != op_map_.end()) {
    return it->second;
  }
  return nullptr;
}

const std::vector<std::shared_ptr<FusionPattern::OpDesc>> &FusionPattern::GetOpDescs() const { return ops_; }
/**
 * @ingroup fe
 * @brief record error
 */
void FusionPattern::SetError() { has_error_ = true; }
}
