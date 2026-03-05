/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_CPU_ID_RESOURCE_MANAGER_H
#define AIR_CXX_CPU_ID_RESOURCE_MANAGER_H

#include <vector>
#include <mutex>
#include "framework/common/ge_types.h"

namespace ge {
constexpr uint32_t kMaxResourceId = 1024U;
constexpr uint32_t kMaxStreamId = 3072U;
class CpuIdResourceManager {
 public:
  static CpuIdResourceManager &GetInstance();

  Status Allocate(uint32_t &id);
  Status DeAllocate(std::vector<uint32_t> &ids);
  Status GenerateAicpuStreamId(uint32_t &id);
  Status FreeAicpuStreamId(const std::vector<uint32_t> &ids);

 private:
  std::vector<bool> resources_ = std::vector<bool>(kMaxResourceId, false);
  std::vector<bool> streams_ = std::vector<bool>(kMaxStreamId, false);
  std::mutex mutex_;
};

class AicpuModelIdResourceManager : public CpuIdResourceManager {
 public:
  static AicpuModelIdResourceManager &GetInstance() {
    static AicpuModelIdResourceManager instance;
    return instance;
  }
  Status GenerateAicpuModelId(uint32_t &id);
};

class NotifyIdResourceManager : public CpuIdResourceManager {
 public:
  static NotifyIdResourceManager &GetInstance() {
    static NotifyIdResourceManager instance;
    return instance;
  }

  Status GenerateNotifyId(uint32_t &id);
};
} // namespace ge
#endif  // AIR_CXX_CPU_ID_RESOURCE_MANAGER_H
