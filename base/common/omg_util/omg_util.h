/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_OMG_UTIL_H_
#define GE_GRAPH_COMMON_OMG_UTIL_H_

#include <string>
#include <vector>

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/node.h"

namespace ge {
static constexpr int64_t kBufferPoolMemAlignSize = 512;
static constexpr uint32_t kBufferPoolNodeOutIndex = 0U;
static constexpr uint32_t kEventReuseThreshold = 65500U;

/// @brief get the Original Type of FrameworkOp
/// @param [in] node
/// @param [out] type
/// @return Status
Status GetOriginalType(const ge::NodePtr &node, std::string &type);

/// @brief set op stream_label
/// @param [in] node
/// @param [in] label
/// @return Status
Status SetStreamLabel(const ge::NodePtr &node, const std::string &label);

/// @brief set op cycle_event flag
/// @param [in] node
/// @return Status
Status SetCycleEvent(const ge::NodePtr &node);

/// @brief set op active_label_list
/// @param [in] node
/// @param [in] label
/// @return Status
Status SetActiveLabelList(const ge::NodePtr &node, const std::vector<std::string> &active_label_list);

/// @brief set op branch_label
/// @param [in] node
/// @param [in] branch_label
/// @return Status
Status SetSwitchBranchNodeLabel(const ge::NodePtr &node, const std::string &branch_label);

/// @brief set op true_branch flag
/// @param [in] node
/// @param [in] value
/// @return Status
Status SetSwitchTrueBranchFlag(const ge::NodePtr &node, const bool value);

/// @brief set op original name
/// @param [in] node
/// @param [in] orig_name
/// @return Status
Status SetOriginalNodeName(const ge::NodePtr &node, const std::string &orig_name);

/// @brief set op cyclic_dependence flag
/// @param [in] node
/// @return Status
Status SetCyclicDependenceFlag(const ge::NodePtr &node);

/// @brief set op next_iteration name
/// @param [in] Merge Node
/// @param [in] NextIteration Node
/// @return Status
Status SetNextIteration(const NodePtr &node, const NodePtr &next_node);

/// @brief Align the memory
/// @param [in/out] memory size
/// @param [in] alinment
/// @return void
void AlignMemSize(int64_t &mem_size, const int64_t align_size);

/// @brief Get memory size from tensor desc
/// @param [in] node
/// @param [out] memory size
/// @return Status
Status GetMemorySize(const NodePtr &node, int64_t &output_size);

/// @brief Set Op _control_flow_group flag
/// @param [in] node
/// @param [in] group, condition group index of node.
/// @return
void SetControlFlowGroup(const NodePtr &node, const int64_t group);
}  // namespace ge

#endif  // GE_GRAPH_COMMON_OMG_UTIL_H_
