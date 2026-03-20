/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TYPES_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TYPES_H_

#include "common/om2/codegen/ast/ast_nodes.h"
#include "ast/ast_context.h"
#include "graph/op_desc.h"
#include "proto/task.pb.h"

namespace ge {
constexpr int64_t kInvalidOpIndex = -1;
constexpr int64_t kInvalidOpId = -1;
constexpr int32_t kInvalidAnchorIndex = -1;

struct OpInputEdges {
  std::vector<int64_t> input_op_ids;
  std::vector<int32_t> input_anchor_indices;
  std::vector<std::string> output_var_names;
};

// Runtime parameters extracted from model
struct Om2RuntimeParam {
  uint64_t mem_size = 0U;
  uint64_t weight_size = 0U;
  uint32_t stream_num = 0U;
  uint32_t notify_num = 0U;
  uint32_t event_num = 0U;
  uint32_t label_num = 0U;
  uint32_t kernel_bin_num = 0U;
};

struct ArgsInfo {
  std::vector<uint64_t> args_sizes;
  std::vector<uint64_t> args_offset;
  std::multimap<uint64_t, uint64_t> io_addr_offset_map; // mapping between compiled offset addr to host arg offset addr
  uint64_t host_args_len = 0U;
};

struct TaskDistributionContext {
  AstContext &ast_ctx;
  std::vector<AstNode *> &nodes;
  const OpDescPtr op_desc;
  const domi::TaskDef &task_def;
  int64_t op_index;
  std::unordered_map<std::string, uint32_t> &func_handle_indices;
  std::unordered_map<int64_t, OpInputEdges> &op_id_to_input_edges;
  std::unordered_map<int64_t, std::string> &weight_offset_to_varname;
  Om2RuntimeParam runtime_param;
  uint64_t aicpu_task_index;
  ArgsInfo &args_info;
  uint64_t &args_table_index;
  std::set<int64_t> model_io_offsets;
};

struct TaskDistributionImplContext {
  AstContext &ast_ctx;
  std::vector<AstNode *> &nodes;
};

enum class Om2MemoryAppType : int32_t {
  kMemoryTypeFix,  // const and var and fix fm
  kMemoryTypeFeatureMap,
  kMemoryTypeModelIo,
  kEnd
};

struct AddrGenInfo {
  std::vector<AstNode *> nodes;
  std::string var_name;
  Om2MemoryAppType mem_type;
  int64_t compile_state_io_addr_offset;
};
} // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TYPES_H_
