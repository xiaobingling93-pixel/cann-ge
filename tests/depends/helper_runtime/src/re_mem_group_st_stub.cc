/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include <string>
#include "runtime/rt_mem_queue.h"

RTS_API rtError_t rtMemGrpCreate(const char *name, const rtMemGrpConfig_t *cfg) {
  const std::string group_name(name);
  const std::string qs_name("DM_QS_GROUP");
  if (group_name.find(qs_name) == group_name.npos) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtMemGrpCacheAlloc(const char *name,
                                     int32_t devId,
                                     const rtMemGrpCacheAllocPara *para) {
  return 0;
}

RTS_API rtError_t rtMemGrpAddProc(const char *name, int32_t pid, const rtMemGrpShareAttr_t *attr) {
  const std::string group_name(name);
  const std::string qs_name("DM_QS_GROUP");
  if (group_name.find(qs_name) == group_name.npos) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtMemGrpAttach(const char *name, int32_t timeout) {
  if (timeout < 1000) {
    return 1;
  }
  return 0;
}


RTS_API rtError_t rtQueryDevPid(rtBindHostpidInfo_t *info, int32_t *devPid) {
  if ((info != nullptr) && (info->chipId == 0xff)) {
    return 1;
  }
  *devPid = 100;
  return 0;
}


RTS_API rtError_t rtMemQueueInitFlowGw(int32_t devId, const rtInitFlowGwInfo_t *const initInfo) {
  const std::string group_name(initInfo->groupName);
  const std::string qs_name("DM_QS_GROUP");
  if (group_name.find(qs_name) == group_name.npos) {
    return 1;
  }
  return 0;
}
