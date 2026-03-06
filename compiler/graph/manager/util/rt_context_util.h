/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_
#define GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_

#include <vector>
#include <map>
#include <mutex>

#include "runtime/context.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class RtContextUtil {
 public:
  static RtContextUtil &GetInstance();

  Status SetRtContext(const uint64_t session_id, const uint32_t graph_id, const int32_t device_id,
                      const rtCtxMode_t mode, rtContext_t rt_context) const;
  void AddRtContext(uint64_t session_id, rtContext_t context);
  void AddRtContext(uint64_t session_id, uint32_t graph_id, rtContext_t context);
  void DestroyRtContexts(uint64_t session_id);
  void DestroyRtContexts(uint64_t session_id, uint32_t graph_id);
  void DestroyAllRtContexts();

  RtContextUtil &operator=(const RtContextUtil &) = delete;
  RtContextUtil(const RtContextUtil &RtContextUtil) = delete;

 private:
  RtContextUtil() = default;
  ~RtContextUtil() {}

  void DestroyRtContexts(uint64_t session_id, int64_t graph_id, std::vector<rtContext_t> &contexts) const;

  std::map<uint64_t, std::map<int64_t, std::vector<rtContext_t>>> rt_contexts_;
  std::mutex ctx_mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_UTIL_RT_CONTEXT_UTIL_H_

