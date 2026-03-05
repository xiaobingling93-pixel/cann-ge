/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_CODEGEN_UTILS_H
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_CODEGEN_UTILS_H
#include <unordered_map>
#include <string>
#include "common/opskernel/ops_kernel_info_types.h"
#include "ge_common/ge_api_types.h"
#include "graph/op_desc.h"
#include "framework/common/taskdown_common.h"
#include "fwk_adpt_struct.h"

namespace ge {
const std::map<int32_t, int32_t> kTopicTypeToRtsFlagMap {
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY), 0},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST), RT_KERNEL_DEVICE_FIRST},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY), RT_KERNEL_HOST_ONLY},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST), RT_KERNEL_HOST_FIRST}
};

class Om2CodegenUtils {
 public:
  static std::string GetKernelNameWithExtension(const std::string &kernel_name);
  static std::string GetOpName(const OpDescPtr &op_desc);
  static Status GetMagic(const OpDescPtr &op_desc, std::string &magic);
  static bool IsSupportedTask(ModelTaskType model_task_type);
  static bool IsAllKernel(const ModelTaskType task_type);
  static bool IsAICoreKernel(const ge::ccKernelType kernel_type);
  static bool RequireBinaryKernel(const ModelTaskType kernel_type);
  static bool RequireArgsTable(ModelTaskType model_task_type);
  static bool IsSuppoprtAddrRefreshable(const uint64_t mem_type);
  static bool IsUnsupportedNodeType(const std::string &type);
  static bool IsNeedAtomicCleanTask(const OpDescPtr &op_desc);
  static bool IsSeparatelyCleanTask(const OpDescPtr &op_desc, const std::string &kernel_name);
  static bool OpNeedPrint(const OpDescPtr &op_desc);
  static bool IsSoftSyncOp(const OpDescPtr &op_desc);
  static bool IsBlockingAicpuOp(const OpDescPtr &op_desc);
  static int32_t TopicTypeToRtsFlag(const int32_t topic_type);
};
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_CODEGEN_UTILS_H
