/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_OPDEBUG_REGISTER_H_
#define GE_COMMON_DUMP_OPDEBUG_REGISTER_H_

#include <map>
#include "framework/common/debug/log.h"
#include "common/dump/data_dumper.h"

namespace ge {
class OpDebugTask {
 public:
  OpDebugTask() = default;
  ~OpDebugTask();

 private:
  uint32_t debug_stream_id_ = 0U;
  uint32_t debug_task_id_ = 0U;
  void *op_debug_addr_ = nullptr;
  friend class OpdebugRegister;
};

class OpdebugRegister {
 public:
  OpdebugRegister() = default;
  ~OpdebugRegister() = default;

  Status RegisterDebugForModel(rtModel_t const model_handle, const uint32_t op_debug_mode, DataDumper &data_dumper);
  void UnregisterDebugForModel(rtModel_t const model_handle);

  Status RegisterDebugForStream(aclrtStream const stream, const uint32_t op_debug_mode, DataDumper &data_dumper);
  void UnregisterDebugForStream(aclrtStream const stream);

 private:
  Status MallocMemForOpdebug();
  static Status CreateOpDebugTaskByStream(aclrtStream const stream, const uint32_t op_debug_mode);
  Status MallocP2PDebugMem(const void * const op_debug_addr);

  void *op_debug_addr_ = nullptr;
  void *p2p_debug_addr_ = nullptr;
  static std::mutex mu_;
  static std::map<aclrtStream, std::unique_ptr<OpDebugTask>> op_debug_tasks_;
  static std::map<aclrtStream, uint32_t> stream_ref_count_;
};
}  // namespace ge
#endif  // GE_COMMON_DUMP_OPDEBUG_REGISTER_H_
