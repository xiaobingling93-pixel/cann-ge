/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>
#include "offline_build_config_parse.h"
#include <securec.h>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>

#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/memory/memory_api.h"
#include "graph/ge_local_context.h"
#include "framework/common/ge_types.h"  // ge对外options
#include "hccl/hcom.h"
#include "hcom_op_utils.h"
#include "ops_kernel_builder_base.h"
#include "op_hcom_comm.h"
#include "mmpa/mmpa_api.h"

using namespace std;

namespace hccl {
static std::mutex g_taskNumCalModeMutex;

const std::unordered_map<std::string, DevType> SOC_VER_CONVERT{
    {"Ascend310P1", DevType::DEV_TYPE_310P3},
    {"Ascend310P3", DevType::DEV_TYPE_310P3},
    {"Ascend310P5", DevType::DEV_TYPE_310P3},
    {"Ascend310P7", DevType::DEV_TYPE_310P3},
    {"Ascend310B1", DevType::DEV_TYPE_310P3},  // 临时映射，规避当前Ascend310B1
                                               // torch_npu未与hccl的so解耦；计划20250630完成解耦，解耦后删除
    {"Ascend910", DevType::DEV_TYPE_910},
    {"Ascend910A", DevType::DEV_TYPE_910},
    {"Ascend910B", DevType::DEV_TYPE_910},
    {"Ascend910ProA", DevType::DEV_TYPE_910},
    {"Ascend910ProB", DevType::DEV_TYPE_910},
    {"Ascend910PremiumA", DevType::DEV_TYPE_910},
    {"Ascend910B1", DevType::DEV_TYPE_910B},
    {"Ascend910B2", DevType::DEV_TYPE_910B},
    {"Ascend910B2C", DevType::DEV_TYPE_910B},
    {"Ascend910B3", DevType::DEV_TYPE_910B},
    {"Ascend910B4", DevType::DEV_TYPE_910B},
    {"Ascend910B4-1", DevType::DEV_TYPE_910B},
    {"Ascend910_9391", DevType::DEV_TYPE_910_93},
    {"Ascend910_9381", DevType::DEV_TYPE_910_93},
    {"Ascend910_9392",
     DevType::DEV_TYPE_910_93},  // Ascend910_9392、Ascend910_9382为预留类型，当前版本暂不支持，待跟随后续版本节奏交付
    {"Ascend910_9382", DevType::DEV_TYPE_910_93},
    {"Ascend910_9372", DevType::DEV_TYPE_910_93},
    {"Ascend910_9362", DevType::DEV_TYPE_910_93},
    {"nosoc", DevType::DEV_TYPE_NOSOC}};

bool IsOfflineCompilation() {
  std::string offlineString;
  if (ge::GetThreadLocalContext().GetOption("ge.offline_hccl_compile", offlineString) == ge::GRAPH_SUCCESS) {
    return true;
  }
  return false;
}

HcclResult GetOffDeviceTypeWithoutDev(DevType &devType) {
  // 离线编译第一阶段获取devType从SOC_VERSION里获取
  std::string socVersion;
  if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("[offline][compilation] get soc version failed.");
    return HCCL_E_NOT_FOUND;
  }

  DevType tempDevType = DevType::DEV_TYPE_COUNT;
  if (socVersion == "Ascend310B1") {
    HCCL_WARNING("[GetOffDeviceTypeWithoutDev] Ascend310B1 not support! please check usage");
  }
  if (socVersion.find("Ascend950") != std::string::npos) {
#ifdef MACRO_DEV_TYPE_NEW
    tempDevType = DevType::DEV_TYPE_950;
#else
    tempDevType = DevType::DEV_TYPE_910_95;
#endif
    return HCCL_SUCCESS;
  }
  auto iter = SOC_VER_CONVERT.find(socVersion);
  if (iter == SOC_VER_CONVERT.end()) {
    HCCL_ERROR("[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get illegal chipver, chip_ver[%s].",
               HCCL_ERROR_CODE(HCCL_E_RUNTIME), socVersion.c_str());
    return HCCL_E_RUNTIME;
  }
  tempDevType = iter->second;

  if (tempDevType != DevType::DEV_TYPE_910 && tempDevType != DevType::DEV_TYPE_910B &&
      tempDevType != DevType::DEV_TYPE_310P1 && tempDevType != DevType::DEV_TYPE_310P3 &&
#ifdef MACRO_DEV_TYPE_NEW
      tempDevType != DevType::DEV_TYPE_910_93 && tempDevType != DevType::DEV_TYPE_950) {
#else
      tempDevType != DevType::DEV_TYPE_910_93 && tempDevType != DevType::DEV_TYPE_910_95) {
#endif
    HCCL_ERROR("[offline][compilation] cur dev type[%u] is not support.", tempDevType);
    return HCCL_E_RUNTIME;
  }
  devType = tempDevType;
  HCCL_DEBUG("[offline] Get devtype[%u]....", devType);
  return HCCL_SUCCESS;
}

HcclResult GetDeterministic(u8 &deterministic) {
  deterministic = DETERMINISTIC_DISABLE;  // 默认为不支持
  DevType devType;
  CHK_RET(GetOffDeviceTypeWithoutDev(devType));

  char *mmSysGetEnvValue = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
  std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
  if (hcclDeterministicEnv != "EmptyString") {
    // 环境变量优先
    std::transform(hcclDeterministicEnv.begin(), hcclDeterministicEnv.end(), hcclDeterministicEnv.begin(), ::toupper);
    if (hcclDeterministicEnv == "FALSE") {
      deterministic = DETERMINISTIC_DISABLE;
    } else if (hcclDeterministicEnv == "TRUE") {
      deterministic = DETERMINISTIC_ENABLE;
    } else if (hcclDeterministicEnv == "STRICT") {
      CHK_PRT_RET(devType != DevType::DEV_TYPE_910B && devType != DevType::DEV_TYPE_910_93,
                  HCCL_ERROR("ParserHcclDeterministic: "
                             "reduce order preservation is not supported for devType[%d]",
                             devType),
                  HCCL_E_NOT_SUPPORT);
      deterministic = DETERMINISTIC_STRICT;
    } else {
      HCCL_ERROR("[GetDeterministic] HCCL_DETERMINISTIC is set to [%s], which is incorrect. Please check",
                 hcclDeterministicEnv.c_str());
      return HCCL_E_PARA;
    }
  } else {
    // 未配环境变量，检查ge option
    std::string geOption;
    if (ge::GetThreadLocalContext().GetOption(ge::DETERMINISTIC, geOption) == ge::GRAPH_SUCCESS) {
      if (geOption == "1") {
        deterministic = DETERMINISTIC_ENABLE;
      } else if (geOption == "2") {
        CHK_PRT_RET(devType != DevType::DEV_TYPE_910B && devType != DevType::DEV_TYPE_910_93,
                    HCCL_ERROR("ParserHcclDeterministic: "
                               "reduce order preservation is not supported for devType[%d]",
                               devType),
                    HCCL_E_NOT_SUPPORT);
        deterministic = DETERMINISTIC_STRICT;
      }
    }
  }

  return HCCL_SUCCESS;
}
}  // namespace hccl
