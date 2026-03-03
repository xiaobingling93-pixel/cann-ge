/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_test_util.h"
#include "mmpa/mmpa_api.h"
#include "runtime/base.h"
#include "aprof_pub.h"

rtError_t rtRegDeviceStateCallback(const char *regName, rtDeviceStateCallback callback) {
  return 0;
}

int32_t MsprofInit(uint32_t dataType, void *data, uint32_t dataLen) {
  const char * const kEnvRecordPath = "MS_PROF_INIT_FAIL";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }

  return 0;
}

int32_t MsprofFinalize() {
  const char * const kEnvRecordPath = "MS_PROF_FINALIZE_FAIL";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return 0;
}

int32_t MsprofUnsetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId){
  return 0;
}

int32_t MsprofSetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId) {
  return 0;
}

int32_t MsprofSetConfig(uint32_t configType, const char *config, size_t configLength) {
  return 0;
}

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi *api) {
  return ge::ProfilingTestUtil::Instance().RunProfFunc(0, ge::InfoType::kApi, const_cast<MsprofApi *>(api), sizeof(api));
 }

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) {
   return ge::ProfilingTestUtil::Instance().RunProfFunc(0, ge::InfoType::kInfo, const_cast<void *>(data), length);
 }
int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) {
  return ge::ProfilingTestUtil::Instance().RunProfFunc(0, ge::InfoType::kCompactInfo, const_cast<void *>(data), length);
 }
int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) {
  return 0;
}

int32_t MsprofReportEvent(uint32_t agingFlag, const MsprofEvent *event) {
  return ge::ProfilingTestUtil::Instance().RunProfFunc(0, ge::InfoType::kEvent, const_cast<MsprofEvent *>(event), 1);
}

uint64_t MsprofSysCycleTime() {
  return 1;
}
int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName) {
  ge::ProfilingTestUtil::Instance().RegisterType(level, typeId, typeName);
  return 0;
}

uint64_t MsprofGetHashId(const char *hashInfo, size_t length) {
  if (ge::ProfilingTestUtil::Instance().hash_func_ != nullptr) {
    return ge::ProfilingTestUtil::Instance().hash_func_(hashInfo, length);
  }
  std::string name(hashInfo, length);
  std::hash<std::string> hs;
  ++ge::ProfilingTestUtil::Instance().GetHashCount();
  return hs(name);
}

int32_t MsprofStart(uint32_t dataType, const void *data, uint32_t dataLen) {
  const char * const kEnvRecordPath = "MS_PROF_FOR_HOST_FAIL";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return 0;
}

int32_t MsprofStop(uint32_t dataType, const void *data, uint32_t dataLen) {
  const char * const kEnvRecordPath = "MS_PROF_FOR_HOST_FAIL";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return 0;
}
