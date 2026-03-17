/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_resource_manager.h"
#include "framework/common/debug/ge_log.h"
#include "exe_graph/runtime/kernel_context.h"
#include "runtime/kernel.h"
#include "common/checker.h"
#include "common/debug/log.h"
#include "framework/common/scope_guard.h"
#include "runtime/mem.h"
#include "runtime/context.h"
#include "aicpu_engine_struct.h"
#include "register/kernel_registry.h"
#include "register/host_cpu_context.h"
#include "mmpa/mmpa_api.h"
#include "graph/load/model_manager/model_manager.h"

namespace gert {
namespace {
void FreeHbmMem(void *p) {
  if (p != nullptr) {
    (void) rtFree(p);
  }
}

const std::string kHostCpuLibRelativePathOld = "/op_impl/built-in/host_cpu/libconstant_folding_ops.so";
const std::string kHostCpuLibRelativePath = "/built-in/op_impl/host_cpu/libconstant_folding_ops.so";

ge::graphStatus GetRealPath(std::string &path) {
  const std::string real_path = ge::RealPath(path.c_str());
  GE_ASSERT_TRUE(!real_path.empty());
  path = real_path;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetLibPath(std::string &lib_path) {
  GELOGI("Start to get host cpu lib path.");
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  GE_ASSERT_TRUE(path_env != nullptr);

  lib_path = std::string(path_env);
  GE_ASSERT_TRUE(!lib_path.empty());

  lib_path += kHostCpuLibRelativePath;
  if (GetRealPath(lib_path) != ge::GRAPH_SUCCESS) {
    lib_path = std::string(path_env) + kHostCpuLibRelativePathOld;
    if (GetRealPath(lib_path) != ge::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "GetLibPath failed, lib_path = %s", lib_path.c_str());
      GELOGE(ge::INTERNAL_ERROR, "[Invoke][GetLibPath] failed. path = %s", lib_path.c_str());
      return ge::INTERNAL_ERROR;
    }
  }

  GELOGI("Get host cpu so path from env: %s", lib_path.c_str());
  return ge::GRAPH_SUCCESS;
}
} // namespace

AicpuResourceManager &AicpuResourceManager::GetInstance() {
  static AicpuResourceManager aicpu_resource_manager;
  return aicpu_resource_manager;
}

AicpuResourceManager::~AicpuResourceManager() {
  if (so_handle_ != nullptr) {
    (void)mmDlclose(so_handle_);
    so_handle_ = nullptr;
  }
}

ge::graphStatus AicpuResourceManager::LoadConstantFoldingLib() {
  const std::lock_guard<std::mutex> lk(mutex_);
  if (run_cpu_kernel_ != nullptr) {
    GELOGD("Constant folding lib has been loaded.");
    return ge::GRAPH_SUCCESS;
  }
  std::string lib_path;
  GE_ASSERT_GRAPH_SUCCESS(GetLibPath(lib_path));

  GELOGI("To invoke dlopen on lib: %s", lib_path.c_str());
  const auto open_flag = static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL);

  so_handle_ = mmDlopen(lib_path.c_str(), static_cast<int32_t>(open_flag));
  if (so_handle_ == nullptr) {
    const ge::char_t *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    REPORT_INNER_ERR_MSG("E19999", "mmDlopen failed, path = %s, error = %s", lib_path.c_str(), error);
    GELOGE(ge::INTERNAL_ERROR, "[Invoke][DlOpen] failed. path = %s, error = %s", lib_path.c_str(), error);
    return ge::INTERNAL_ERROR;
  }

  const auto initialize = reinterpret_cast<ge::Status (*)(const ge::HostCpuContext &)>(mmDlsym(so_handle_,
                                                                                               "Initialize"));
  if (initialize != nullptr) {
    GELOGI("Invoke function Initialize in lib: %s", lib_path.c_str());
    if (initialize(ge::HostCpuContext()) != ge::SUCCESS) {
      GELOGW("Failed to invoke function Initialize in lib: %s", lib_path.c_str());
    }
  }
  run_cpu_kernel_ = reinterpret_cast<uint32_t (*)(void *)>(mmDlsym(so_handle_, "RunHostCpuKernel"));
  GE_ASSERT_NOTNULL(run_cpu_kernel_);

  aicpu_host_find_func_ = reinterpret_cast<AicpuHostProcFunc (*)(std::string)>(mmDlsym(so_handle_,
                          "AicpuHostFindFunc"));
  GE_ASSERT_NOTNULL(aicpu_host_find_func_);

  GELOGI("Lib: %s has been opened", lib_path.c_str());
  return ge::GRAPH_SUCCESS;
}

std::function<uint32_t(void *)> AicpuResourceManager::GetRunCpuKernel() const {
  return run_cpu_kernel_;
}

std::function<AicpuHostProcFunc(std::string)> AicpuResourceManager::GetAicpuHostFindFunc() const {
  return aicpu_host_find_func_;
}

ge::graphStatus AicpuResourceManager::CheckOrCreateHandle(const std::string &op_name, const rtStream_t stream,
                                                          const GertTensorData *handle_data) {
  if (handles_.find(op_name) == handles_.end()) {
    GE_ASSERT_RT_OK(rtStreamSynchronize(stream));
    uint64_t handle = 0;
    GE_ASSERT_RT_OK(rtMemcpy(&handle, sizeof(uint64_t), handle_data->GetAddr(), sizeof(uint64_t),
                             RT_MEMCPY_DEVICE_TO_HOST));
    handles_[op_name] = handle;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuResourceManager::PushTensor(const std::string &op_name, const rtStream_t stream,
                                                 const GertTensorData *tensor_data, const GertTensorData *handle_data) {
  const std::lock_guard<std::mutex> lk(mutex_);
  GE_ASSERT_SUCCESS(CheckOrCreateHandle(op_name, stream, handle_data));
  const auto handle = handles_[op_name];
  auto &tensors = tensors_[handle];
  tensors.push_back(GertTensorData());
  tensors.back().ShareFrom(*tensor_data);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuResourceManager::PopTensor(const std::string &op_name, const rtStream_t stream,
                                                const GertTensorData *handle_data) {
  const std::lock_guard<std::mutex> lk(mutex_);
  GE_ASSERT_SUCCESS(CheckOrCreateHandle(op_name, stream, handle_data));
  const auto handle = handles_[op_name];
  auto &tensors = tensors_[handle];
  if (!tensors.empty()) {
    tensors.pop_back();
  }
  return ge::GRAPH_SUCCESS;
}

void AicpuResourceManager::ClearTensors() {
  const std::lock_guard<std::mutex> lk(mutex_);
  for (auto &iter : tensors_) {
    iter.second.clear();
  }
}

ge::graphStatus AicpuResourceManager::HasLoadedCustAicpuSo(const std::string &so_name, bool &loaded) {
  // get current context
  rtContext_t rt_current_ctx = nullptr;
  GE_CHK_RT_RET(aclrtGetCurrentContext(&rt_current_ctx));
 
  // use current context as resource key
  const std::lock_guard<std::mutex> lk(cust_aicpu_so_mutex_);
 
  const uintptr_t resource_id =
      static_cast<uintptr_t>(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rt_current_ctx)));
  if (cust_aicpu_context_so_.find(resource_id) == cust_aicpu_context_so_.end()) {
    cust_aicpu_context_so_[resource_id] = so_name;
    loaded = false;
    GELOGI("New added aicpu so name %s, resource id %lu.", so_name.c_str(), resource_id);
    return ge::GRAPH_SUCCESS;
  }
  GELOGI("Had added so name %s, resource id %lu has been loaded.", so_name.c_str(), resource_id);
  loaded = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EnsureCreateTfSession(KernelContext *context) {
  auto session_id = context->GetInputPointer<uint64_t>(0UL);
  GE_ASSERT_NOTNULL(session_id);
  return ge::ModelManager::GetInstance().CreateAicpuSession(*session_id);
}
REGISTER_KERNEL(EnsureCreateTfSession).RunFunc(EnsureCreateTfSession);

ge::graphStatus CreateStepId(KernelContext *context) {
  auto step_id = context->GetOutputPointer<void *>(0U);
  auto iteration = context->GetOutputPointer<int64_t>(1U);
  GE_ASSERT_NOTNULL(step_id);
  GE_ASSERT_NOTNULL(*step_id);
  GE_ASSERT_NOTNULL(iteration);
  *iteration = 0;

  GE_ASSERT_RT_OK(rtMemcpy(*step_id, sizeof(int64_t), iteration, sizeof(int64_t), RT_MEMCPY_HOST_TO_DEVICE));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOutputForStepId(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto chain = context->GetOutput(0U);
  GE_CHECK_NOTNULL(chain);

  void *step_id = nullptr;
  GE_ASSERT_RT_OK(rtMalloc(&step_id, sizeof(int64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  chain->Set(step_id, FreeHbmMem);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateStepId).RunFunc(CreateStepId).OutputsCreator(CreateOutputForStepId);

ge::graphStatus IncreaseStepId(KernelContext *context) {
  auto step_id = context->GetInputValue<void *>(0U);
  auto iteration = context->MutableInputPointer<int64_t>(1U);
  auto stream = context->GetInputValue<rtStream_t>(2U); // 2 stream idx
  GE_CHECK_NOTNULL(step_id);
  GE_CHECK_NOTNULL(iteration);

  *iteration += 1;
  GE_ASSERT_RT_OK(rtMemcpyAsync(step_id, sizeof(int64_t), iteration, sizeof(int64_t), RT_MEMCPY_HOST_TO_DEVICE_EX,
                                stream));
  AicpuResourceManager::GetInstance().ClearTensors();
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(IncreaseStepId).RunFunc(IncreaseStepId);
}  // namespace gert
