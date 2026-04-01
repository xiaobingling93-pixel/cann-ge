/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/task_info.h"
#include "common/checker.h"

namespace ge {
namespace {
constexpr std::array<const char *, static_cast<size_t>(ArgsPlacement::kEnd) + 1U> g_placement_str = {
    "hbm",  // kArgsPlacementHbm
    "ts",   // kArgsPlacementTs
    "sqe",  // kArgsPlacementSqe
    "host_svm",  // kArgsPlacementHostSvm
    "unknown"};
}
Status TaskInfo::SetStream(const uint32_t stream_id, const std::vector<aclrtStream> &stream_list) {
  if (stream_list.size() == 1U) {
    stream_ = stream_list[0U];
  } else if (stream_list.size() > stream_id) {
    stream_ = stream_list[static_cast<size_t>(stream_id)];
  } else {
    REPORT_INNER_ERR_MSG("E19999", "index:%u >= stream_list.size(): %zu, check invalid", stream_id, stream_list.size());
    GELOGE(FAILED, "[Check][Param] index:%u >= stream_list.size():%zu.", stream_id, stream_list.size());
    return FAILED;
  }

  return SUCCESS;
}

void TaskInfo::SetTaskTag(const char_t *const op_name) {
  GE_CHK_RT(rtSetTaskTag(op_name));
}
const char_t *GetArgsPlacementStr(ArgsPlacement placement) {
  auto i = static_cast<size_t>(placement);
  if (i > static_cast<size_t>(ArgsPlacement::kEnd)) {
    i = static_cast<size_t>(ArgsPlacement::kEnd);
  }
  return g_placement_str[i];
}
}  // namespace ge
