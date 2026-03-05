/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_UTILS_H
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_UTILS_H

#include "common/om2/codegen/om2_codegen_types.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"

namespace ge {
class Om2ModelUtils {
 public:
  static Status GenInputAddrCode(TaskDistributionContext &context, std::vector<AddrGenInfo> &input_addr_nodes);

  static Status GenOutputAddrCode(TaskDistributionContext &context, std::vector<AddrGenInfo> &output_addr_nodes,
                                  const bool has_optional_addr);

  static Status GenWorkspaceAddrsCode(TaskDistributionContext &context,
                                      std::vector<AddrGenInfo> &workspace_addr_nodes);

 private:
  static uint64_t GetWorkspaceMemTypeByPriority(const bool is_p2p_memory, const bool is_l1_memory,
                                                const bool is_ub_memory, const bool session_scope_memory);

  static bool ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                               const int64_t size);
  static Status GetValidatedTensorMemType(const GeTensorDescPtr &tensor_desc, const std::vector<int64_t> &mem_types,
                                          size_t index, uint64_t &memory_type);

  static Status GetOrCreateInputVarName(TaskDistributionContext &context, size_t input_idx,
                                        size_t non_const_idx, const std::vector<int64_t> &input_offsets,
                                        std::string &input_ptr_name, std::vector<AstNode *> &input_nodes);
};
} // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_UTILS_H
