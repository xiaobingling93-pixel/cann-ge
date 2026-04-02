/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_UTILS_H
#define AIR_CXX_RUNTIME_V2_UTILS_H

#include "ge/ge_api_types.h"
#include "acl/acl_rt.h"
#include "core/executor/executor_base_def.h"

namespace gert {
ge::Status DoRtStreamSyncWithTimeout(aclrtStream stream);

ge::Status GetKernelStream(const Node *node, aclrtStream &stream);

bool IsInputPlacementOnDeviceHbm();

bool IsEnableRmLaunchFreeEdge();
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_UTILS_H
