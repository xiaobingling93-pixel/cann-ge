/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/memory/memory_api.h"

#include <memory>

#include "common/math/math_util.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/def_types.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/rdma_pool_allocator.h"
#include "graph/utils/type_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "graph/utils/tensor_utils.h"
#include "common/checker.h"

namespace ge {
Status InitRdmaPool(size_t size, rtMemType_t mem_type) {
  GELOGD("InitRdmaPool in");
  return MemManager::Instance().RdmaPoolInstance(mem_type).InitMemory(size);
}

Status RdmaRemoteRegister(const std::vector<HostVarInfo> &var_info, rtMemType_t mem_type) {
  GELOGD("Start to register rdma memory with host var size %zu", var_info.size());
  uint64_t device_base = 0U;
  uint64_t device_size = 0U;
  GE_CHK_STATUS_RET(MemManager::Instance().RdmaPoolInstance(mem_type).GetBaseAddr(device_base, device_size));
  const size_t table_len = var_info.size() + 1U;
  const std::unique_ptr<MemRegisterAddr[]> reg_addrs = MakeUnique<MemRegisterAddr[]>(table_len);
  GE_CHECK_NOTNULL(reg_addrs);
  for (size_t i = 0U; i < var_info.size(); ++i) {
    reg_addrs[i] = {var_info[i].base_addr, var_info[i].var_size};
  }
  reg_addrs[table_len - 1U] = {device_base, device_size};

  const std::string file_name = "libhccl.so";
  std::string path = GetModelPath();
  (void)path.append(file_name);
  const std::string canonical_path = RealPath(path.c_str());
  if (canonical_path.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "canonical_path:%s is empty, check invalid", canonical_path.c_str());
    GELOGE(FAILED, "[Call][RealPath] Failed to get realpath of %s", path.c_str());
    return FAILED;
  }
  GELOGI("FileName:%s, Path:%s.", file_name.c_str(), canonical_path.c_str());
  const auto handle = mmDlopen(canonical_path.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                                                                            static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  GE_CHECK_NOTNULL(handle);
  GE_MAKE_GUARD(not_used_var, [&handle]() {
    if (mmDlclose(handle) != 0) {
      GELOGW("Failed to close handle %s", mmDlerror());
    }
  });

  const auto hcom_remote_mem_register =
      reinterpret_cast<HcclResult(*)(const MemRegisterAddr *, uint32_t)>(mmDlsym(handle, "HcomRegRemoteAccessMem"));
  if (hcom_remote_mem_register == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Symbol HcomRegRemoteAccessMem can't find in %s, check invalid",
                      canonical_path.c_str());
    GELOGE(FAILED, "[Check][Param] Symbol HcomRegRemoteAccessMem can't find in %s", canonical_path.c_str());
    return FAILED;
  }

  const HcclResult hccl_ret = hcom_remote_mem_register(reg_addrs.get(), static_cast<uint32_t>(table_len));
  if (hccl_ret != HCCL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call hcom_remote_mem_register failed, ret:%d,", hccl_ret);
    GELOGE(HCCL_E_INTERNAL, "[Call][HcomRemoteMemRegister] Rdma mem register failed, ret:0x%X", hccl_ret);
    return HCCL_E_INTERNAL;
  }
  return SUCCESS;
}

Status MallocSharedMemory(const TensorInfo &tensor_info, uint64_t &dev_addr, uint64_t &memory_size) {
  GELOGD("MallocSharedMemory in");
  ge::GeTensorDesc tensor_desc;
  int64_t calculate_size = 0L;
  tensor_desc.SetDataType(tensor_info.data_type);
  tensor_desc.SetShape(ge::GeShape(tensor_info.dims));
  tensor_desc.SetFormat(FORMAT_ND);
  GE_CHK_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(tensor_desc, calculate_size),
                    "[Calculate][SharedMemory] failed, op name:%s", tensor_info.var_name.c_str());
  memory_size = static_cast<uint64_t>(calculate_size);
  GELOGI("[Calculate][SharedMemory] size is:%" PRIu64 " for op name:%s", memory_size, tensor_info.var_name.c_str());
  SharedMemInfo mem_info(tensor_info.var_name, memory_size);
  const Status ret = HostMemManager::Instance().MallocHostSharedMemory(mem_info);
  if (ret != SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Malloc][SharedMemory] failed, op name [%s]", tensor_info.var_name.c_str());
    return GRAPH_FAILED;
  }
  dev_addr = PtrToValue(mem_info.device_address);
  GELOGD("MallocSharedMemory Succeeded");
  return SUCCESS;
}

Status GetVarBaseAddrAndSize(const std::string &var_name, uint64_t &base_addr, uint64_t &var_size) {
  GELOGD("GetVarBaseAddrAndSize in, var name:[%s]", var_name.c_str());
  SharedMemInfo mem_info;
  if (!HostMemManager::Instance().QueryVarMemInfo(var_name, mem_info)) {
    GELOGE(FAILED, "Get addr and size failed, name:[%s]", var_name.c_str());
    return FAILED;
  }
  base_addr = PtrToValue(mem_info.host_aligned_ptr->Get());
  var_size = mem_info.mem_size;
  return SUCCESS;
}

Status GetVarBaseAddrAndSize(const char_t *var_name, uint64_t &base_addr, uint64_t &var_size) {
  std::string var_name_str;
  GE_ASSERT_NOTNULL(var_name);
  var_name_str = std::string(var_name);
  return GetVarBaseAddrAndSize(var_name_str, base_addr, var_size);
}

}  // namespace ge

#ifdef __cplusplus
extern "C" {
#endif

ge::Status GeApiWrapper_InitRdmaPool(size_t size, rtMemType_t mem_type) {
  return ge::InitRdmaPool(size, mem_type);
}

ge::Status GeApiWrapper_RdmaRemoteRegister(const std::vector<std::pair<uint64_t, uint64_t>> &var_info,
                                           rtMemType_t mem_type) {
  std::vector<ge::HostVarInfo> ge_var_info;
  for (const auto& info : var_info) {
    ge::HostVarInfo host_var_info;
    host_var_info.base_addr = info.first;
    host_var_info.var_size = info.second;
    ge_var_info.push_back(host_var_info);
  }
  return ge::RdmaRemoteRegister(ge_var_info, mem_type);
}

ge::Status GeApiWrapper_GetVarBaseAddrAndSize(const char *var_name, uint64_t &base_addr, uint64_t &var_size) {
  return ge::GetVarBaseAddrAndSize(var_name, base_addr, var_size);
}

ge::Status GeApiWrapper_MallocSharedMemory(const std::string &var_name, const std::vector<int64_t> &dims,
                                           ge::DataType data_type, uint64_t &dev_addr, uint64_t &memory_size) {
  ge::TensorInfo tensor_info;
  tensor_info.var_name = var_name;
  tensor_info.dims = dims;
  tensor_info.data_type = data_type;
  return ge::MallocSharedMemory(tensor_info, dev_addr, memory_size);
}

#ifdef __cplusplus
}
#endif
