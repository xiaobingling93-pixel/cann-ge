/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_TRANS_VAR_DATA_UTILS_H_
#define GE_GRAPH_MANAGER_TRANS_VAR_DATA_UTILS_H_

#include <string>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/context.h"

namespace ge {
class TransVarDataUtils {
 public:
  static Status TransAllVarData(const std::vector<NodePtr> &variable_nodes, const uint64_t session_id,
                                const uint32_t graph_id, const uint32_t device_id = kDefaultDeviceId);

  static Status CopyVarData(const ComputeGraphPtr &compute_graph, const std::vector<NodePtr> &variable_nodes,
                            const uint64_t session_id, const uint32_t device_id);
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_TRANS_VAR_DATA_UTILS_H_
