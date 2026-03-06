/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/single_op_manager.h"

#include <mutex>
#include <string>

#include "graph/manager/mem_manager.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "runtime/stream.h"

namespace ge {
Status SingleOpManager::GetOpFromModel(const std::string &model_name, const ModelData &model_data,
                                       void *const stream, SingleOp **const single_op, const uint64_t model_id) {
  GELOGI("GetOpFromModel in. model name = %s, model id = %" PRIu64, model_name.c_str(), model_id);
  if (single_op == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Param:single_op] is null.");
    REPORT_INNER_ERR_MSG("E19999", "input param single_op is nullptr, check invalid");
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  uintptr_t resource_id = 0U;
  GE_CHK_STATUS_RET(GetResourceId(stream, resource_id));
  StreamResource *const res = GetResource(resource_id, stream);
  if (res == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Get][Resource] failed.");
    REPORT_INNER_ERR_MSG("E19999", "GetOpFromModel fail because GetResource return nullptr.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  SingleOp *const op = res->GetOperator(model_id);
  if (op != nullptr) {
    GELOGD("Got operator from stream cache");
    *single_op = op;
    return SUCCESS;
  }

  return res->BuildOperator(model_data, single_op, model_id);
}

Status SingleOpManager::ReleaseResource(const void *const stream) {
  const uintptr_t resource_id = static_cast<uintptr_t>(PtrToValue(stream));
  GELOGI("ReleaseResource in. resource id = 0x%" PRIx64, static_cast<uint64_t>(resource_id));
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto it = stream_resources_.find(resource_id);
  if (it == stream_resources_.end()) {
    MemManager::Instance().CachingInstance(RT_MEMORY_HBM).TryFreeBlocks();
    hybrid::NpuMemoryAllocator::FreeCachedMem();
    return SUCCESS;
  }
  (void)stream_resources_.erase(it);
  MemManager::Instance().CachingInstance(RT_MEMORY_HBM).TryFreeBlocks();
  hybrid::NpuMemoryAllocator::FreeCachedMem();
  return SUCCESS;
}

StreamResource *SingleOpManager::GetResource(const uintptr_t resource_id, rtStream_t const stream) {
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto it = stream_resources_.find(resource_id);
  StreamResource *res = nullptr;
  if (it == stream_resources_.end()) {
    auto guarded_res = MakeUnique<StreamResource>(resource_id);
    if (guarded_res != nullptr) {
      if (guarded_res->Init() != SUCCESS) {
        GELOGE(FAILED, "[Malloc][Memory]Failed to malloc device buffer.");
        return nullptr;
      }
      res = guarded_res.get();
      res->SetStream(stream);
      res->SetAllocator(res->GetInternalAllocator());
      (void)stream_resources_.emplace(resource_id, std::move(guarded_res));
    }
  } else {
    res = it->second.get();
  }
  return res;
}

Status SingleOpManager::GetDynamicOpFromModel(const std::string &model_name, const ModelData &model_data,
                                              void *const stream, DynamicSingleOp **const single_op,
                                              const uint64_t model_id) {
  GELOGI("GetOpFromModel in, model name = %s, model id = %" PRIu64, model_name.c_str(), model_id);
  RegisterTilingFunc();

  GE_CHECK_NOTNULL(single_op);
  uintptr_t resource_id = 0U;
  GE_CHK_STATUS_RET(GetResourceId(stream, resource_id));
  StreamResource *const res = GetResource(resource_id, stream);
  if (res == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Get][Resource] failed.");
    REPORT_INNER_ERR_MSG("E19999", "GetDynamicOpFromModel fail because GetResource return nullptr.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  DynamicSingleOp *const op = res->GetDynamicOperator(model_id);
  if (op != nullptr) {
    GELOGD("Got operator from stream cache");
    *single_op = op;
    return SUCCESS;
  }

  return res->BuildDynamicOperator(model_data, single_op, model_id);
}

Status SingleOpManager::DeleteSingleOp(const uint64_t op_id) {
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  for (const auto &it : stream_resources_) {
    GE_CHECK_NOTNULL(it.second);
    GE_CHK_STATUS_RET(it.second->DeleteOperator(op_id));
  }
  return SUCCESS;
}

Status SingleOpManager::DeleteDynamicSingleOp(const uint64_t op_id) {
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  for (const auto &it : stream_resources_) {
    GE_CHECK_NOTNULL(it.second);
    GE_CHK_STATUS_RET(it.second->DeleteDynamicOperator(op_id));
  }
  return SUCCESS;
}

void SingleOpManager::RegisterTilingFunc() {
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  if (tiling_func_registered_) {
    return;
  }

  op_tiling_manager_.LoadSo();
  tiling_func_registered_ = true;
}

Status SingleOpManager::GetResourceId(rtStream_t const stream, uintptr_t &resource_id) {
  // runtime uses NULL to denote a default stream for each device
  if (stream == nullptr) {
    // get current context default stream
    rtStream_t curCtxDefaultStream = nullptr;
    GE_CHK_RT_RET(rtCtxGetCurrentDefaultStream(&curCtxDefaultStream));
    GELOGI("use current context default stream as resource index");
    resource_id = static_cast<uintptr_t>(PtrToValue(curCtxDefaultStream));
  } else {
    GELOGI("use stream as resource key instead when create stream");
    resource_id = static_cast<uintptr_t>(PtrToValue(stream));
  }

  return SUCCESS;
}

Status SingleOpManager::SetAllocator(rtStream_t const stream, Allocator *const allocator) {
  uintptr_t resource_id = 0U;
  GE_ASSERT_SUCCESS(GetResourceId(stream, resource_id));
  auto res = GetResource(resource_id, stream);
  GE_ASSERT_NOTNULL(res);
  res->SetAllocator(allocator);
  return SUCCESS;
}
}  // namespace ge
