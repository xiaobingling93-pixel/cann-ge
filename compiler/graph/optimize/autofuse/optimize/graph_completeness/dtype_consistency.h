/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef OPTIMIZE_GRAPH_COMPLETENESS_DTYPE_CONSISTENCY_H
#define OPTIMIZE_GRAPH_COMPLETENESS_DTYPE_CONSISTENCY_H

#include <vector>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace optimize {
// Actual dtype requirements of nodes
struct NodeDtypeRequirement {
  ge::AscNodePtr node;
  std::vector<ge::DataType> input_dtypes;
  std::vector<ge::DataType> output_dtypes;
};

class DtypeConsistency {
 public:
  // Ensure dtype consistency for the graph: insert necessary Cast nodes and remove redundant ones
  static ge::Status EnsureDtypeConsistency(ge::AscGraph &graph);

 private:
  // Collect dtype requirements for all nodes
  static ge::Status CollectDtypeRequirements(ge::AscGraph &graph, std::vector<NodeDtypeRequirement> &requirements);

  // Process output dtype: directly modify the dtype of node's output tensor
  static ge::Status ProcessOutputDtype(const NodeDtypeRequirement &req);

  // Process input dtype: insert Cast when dtype does not match
  static ge::Status ProcessInputDtype(ge::AscGraph &graph, const NodeDtypeRequirement &req);

  // Check if cast conversion is supported
  static ge::Status CheckCastSupported(ge::DataType src_dtype, ge::DataType dst_dtype, const ge::AscNodePtr &node,
                                       size_t input_idx);

  // Try to merge with upstream cast, return whether merge succeeded
  static bool TryMergeWithUpstreamCast(ge::AscGraph &graph, const ge::AscNodePtr &upstream_cast,
                                       const ge::AscNodePtr &downstream_node, size_t input_idx,
                                       ge::DataType target_dtype);

  // Merge cast when there's only one downstream consumer
  static bool MergeCastWithSingleConsumer(const ge::AscNodePtr &upstream_cast, const ge::AscNodePtr &downstream_node,
                                          size_t input_idx, ge::DataType target_dtype);

  // Merge cast when there are multiple downstream consumers
  static bool MergeCastWithMultipleConsumers(ge::AscGraph &graph, const ge::AscNodePtr &upstream_cast,
                                             const ge::AscNodePtr &downstream_node, size_t input_idx,
                                             ge::DataType target_dtype);

  // Insert a new cast node
  static ge::Status InsertCastNode(ge::AscGraph &graph, const ge::AscNodePtr &src_node, const ge::AscNodePtr &dst_node,
                                   size_t input_idx, ge::DataType target_dtype);

  static ge::Status ApplyDtypeConversions(ge::AscGraph &graph, const std::vector<NodeDtypeRequirement> &requirements);

  // Remove redundant Cast operators
  static ge::Status CancelRedundantCast(ge::AscGraph &graph);

  // Merge multiple identical dtype Cast nodes from the same upstream into one
  static ge::Status DoCastCSE(ge::AscGraph &graph);

  // Remove Cast(A->A) redundancy
  static ge::Status CancelIdentityCast(ge::AscGraph &graph);
};

}  // namespace optimize
#endif  // OPTIMIZE_GRAPH_COMPLETENESS_DTYPE_CONSISTENCY_H
