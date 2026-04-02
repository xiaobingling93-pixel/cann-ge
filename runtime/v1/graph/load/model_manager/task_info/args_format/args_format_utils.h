/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_ARGS_FORMAT_UTILS_H
#define AIR_CXX_ARGS_FORMAT_UTILS_H

#include "graph/op_desc.h"
#include "exe_graph/lowering/device_tiling_context_builder.h"
#include "graph/load/model_manager/davinci_model.h"
namespace ge {
constexpr char_t const *kTilingContextAddrs = "_tiling_context_addr";
constexpr char_t const *kTilingSinkTaskInfo = "_tiling_sink_task_info";
struct TilingSinkTaskInfo {
  void *ffts_task_handle{nullptr};
  aclrtStream stream{nullptr};
  uint32_t task_id{0U};
};

struct TilingContextAddr {
  uint64_t op_type_addr{0UL};
  uint64_t tiling_context_addr{0UL};
  uint64_t tiling_key_addr{0UL};
  uint64_t tiling_data_addr{0UL};
  uint64_t block_dim_addr{0UL};
};

class ArgsFormatUtils {
 public:
  static Status GetHcomHiddenInputs(const OpDescPtr &op_desc, const DavinciModel &davinci_model,
                                    std::vector<void *> &hidden_addrs,
                                    const HiddenInputsType hi_type = HiddenInputsType::HCOM);
  static Status GetTileFwkHiddenInputs(const OpDescPtr &op_desc, const DavinciModel &davinci_model,
                                       std::vector<void *> &hidden_addrs,
                                       const HiddenInputsType hi_type = HiddenInputsType::TILEFWK);
  static Status SinkTilingContext(const NodePtr &node, DavinciModel &davinci_model,
                                  std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor,
                                  void *platform_infos_addr, const bool is_args_exception_enable,
                                  const uint64_t atomic_index);
};
}  // namespace ge
#endif  // AIR_CXX_ARGS_FORMAT_UTILS_H
