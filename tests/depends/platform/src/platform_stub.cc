/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_set>
#include <cstdint>
#include "platform_info.h"

namespace {
constexpr const char *kInvlaidSocVersion = "invalid_version";
constexpr const char *kInvlaidInitSocVersion = "invalid_init_version";
}

fe::PlatformInfoManager& fe::PlatformInfoManager::Instance() {
  static fe::PlatformInfoManager pf;
  return pf;
}

fe::PlatformInfoManager& fe::PlatformInfoManager::GeInstance() {
  static fe::PlatformInfoManager pf;
  return pf;
}

uint32_t fe::PlatformInfoManager::InitializePlatformInfo() {
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetPlatformInstanceByDevice(const uint32_t &device_id,
                                                              PlatFormInfos &platform_infos) {
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetPlatformInfo(const std::string SoCVersion,
                                                  PlatformInfo &platform_info,
                                                  OptionalInfo &opti_compilation_info) {
  static std::unordered_set<std::string> socSet{"Ascend910B1", "Ascend910B2",
                                                "Ascend910B3", "Ascend910B4"};
  const bool jit_compile = (socSet.find(SoCVersion) != socSet.cend()) ? false : true;
  platform_info.soc_info.ai_core_cnt = 32;
  platform_info.soc_info.vector_core_cnt = 32;
  platform_info.software_spec.jit_compile_default_value = jit_compile;
  platform_info.software_spec.jit_compile_mode = AUTO;
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetPlatformInfoWithOutSocVersion(fe::PlatformInfo&, fe::OptionalInfo&) {
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetPlatformInfoWithOutSocVersion(fe::PlatFormInfos&, fe::OptionalInfos&) {
  return 0U;
}

fe::PlatformInfoManager::PlatformInfoManager() {}
fe::PlatformInfoManager::~PlatformInfoManager() {}

uint32_t fe::PlatformInfoManager::GetPlatformInfos(const std::string SoCVersion, fe::PlatFormInfos&, fe::OptionalInfos&) {
  if (SoCVersion == kInvlaidSocVersion) {
    return 1U;
  }

  if (SoCVersion == "test_instance_constant_soc_version" && !runtime_init_flag_) {
    // test for GeInstance or Instance
    return 1U;
  }
  return 0U;
}

uint32_t fe::PlatformInfoManager::UpdatePlatformInfos(fe::PlatFormInfos&) {
  return 0U;
}

uint32_t fe::PlatformInfoManager::UpdatePlatformInfos(const std::string &soc_version, fe::PlatFormInfos&) {
  return 0U;
}

uint32_t fe::PlatformInfoManager::UpdateRuntimePlatformInfosByDevice(
    const uint32_t &device_id, fe::PlatFormInfos &platform_infos) {
  return 0U;
}

uint32_t fe::PlatformInfoManager::InitRuntimePlatformInfos(const string &soc_version) {
  if (soc_version == kInvlaidInitSocVersion) {
    return 1U;
  }

  if (soc_version == "test_instance_constant_soc_version") {
    // test for GeInstance or Instance
    runtime_init_flag_ = true;;
  }
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetRuntimePlatformInfosByDevice(
    const uint32_t &device_id, fe::PlatFormInfos &platform_infos, bool need_deep_copy) {
  return 0U;
}


void fe::PlatformInfoManager::SetOptionalCompilationInfo(fe::OptionalInfos&) {}

uint32_t fe::PlatFormInfos::GetCoreNum() const {
  return 8U;
}

void fe::PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type) {
  core_num_ = 15U;
  return;
}

void fe::PlatFormInfos::SetCoreNum(const uint32_t &core_num) {
  return;
}

std::string fe::PlatFormInfos::SaveToBuffer() {
  return "";
}

bool fe::PlatFormInfos::GetPlatformRes(std::string const&, std::string const& key, std::string &val) {
  if (key == "Short_SoC_version") {
    val = "Ascend910B";
  }
  return true;
}

bool fe::PlatFormInfos::GetPlatformRes(std::string const&, std::map<std::string, std::string, std::less<std::string>,
                                       std::allocator<std::pair<std::string const, std::string> > >&) {
  return true;
}

void fe::PlatFormInfos::SetPlatformRes(std::string const&, std::map<std::string, std::string, std::less<std::string>,
                                       std::allocator<std::pair<std::string const, std::string> > >&) {
}

bool fe::PlatFormInfos::GetPlatformResWithLock(std::string const &label,
                                               std::map<std::string, std::string, std::less<std::string>,
                                               std::allocator<std::pair<std::string const, std::string> > > &res) {
  if (label == "SoCInfo") {
    res = {{"ai_core_cnt", "24"}, {"vector_core_cnt", "24"}, {"cube_core_cnt", "24"}};
  } else if (label == "DtypeMKN") {
    res = {{"DT_UINT8", "16,32,16"},
        {"DT_INT8", "16,32,16"},
        {"DT_INT4", "16,64,16"},
        {"DT_INT2", "16,128,16"},
        {"DT_UINT2", "16,128,16"},
        {"DT_UINT1", "16,256,16"}};
  }
  return true;
}

bool fe::PlatFormInfos::GetPlatformResWithLock(const string &label, const string &key, string &val) {
  if (label == "DtypeMKN" && key == "Default") {
    val = "16,16,16";
  } else if (label == "version" && key == "Short_SoC_Version") {
    val = "Ascend910B";
  } else if (label == "version" && key == "NpuArch") {
    val = "2201";
  } else {
    val = "0";
  }
  return true;
}

void fe::PlatFormInfos::SetPlatformResWithLock(std::string const&,
                                               std::map<std::string, std::string, std::less<std::string>,
                                               std::allocator<std::pair<std::string const, std::string> > >&) {
}

uint32_t fe::PlatFormInfos::GetCoreNumWithLock() const {
  return 8U;
}

void fe::OptionalInfos::SetSocVersion(std::string) {
}

void fe::OptionalInfos::SetAICoreNum(unsigned int) {}

bool fe::PlatFormInfos::Init() {
  return true;
}

bool fe::PlatFormInfos::LoadFromBuffer(const char *buf_ptr, const size_t buf_len) {
  return true;
}

uint32_t fe::PlatFormInfos::GetCoreNumByType(const std::string &core_type) {
  return 8U;
}