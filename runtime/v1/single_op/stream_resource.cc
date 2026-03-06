/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/stream_resource.h"

#include "framework/common/debug/log.h"
#include "runtime/rt.h"
#include "single_op/single_op_model.h"
#include "framework/runtime/device_memory_recorder.h"

namespace ge {
namespace {
// limit available device mem size  1M
constexpr int32_t kThreadNumDefault = 8;
}  // namespace

InternalAllocator::~InternalAllocator() {
  for (const auto &mem : memory_list_) {
    if (mem != nullptr) {
      mem->Free();
    }
  }
  gert::DeviceMemoryRecorder::ClearReserveMemory();
}

MemBlock *InternalAllocator::Malloc(size_t size) {
  if (size == 0U) {
    GELOGD("Mem size == 0");
    return nullptr;
  }

  if ((size <= max_memory_size_) && (!memory_list_.empty())) {
    GELOGD("reuse last memory");
    return memory_list_.back().get();
  }

  if (!memory_list_.empty()) {
    uint8_t *const current_buffer = reinterpret_cast<uint8_t *>(memory_list_.back()->GetAddr());
    memory_list_.pop_back();
    if (rtStreamSynchronize(stream_) != RT_ERROR_NONE) {
      GELOGW("Failed to invoke rtStreamSynchronize");
    }
    (void)rtFree(current_buffer);
  }

  uint8_t *buffer = nullptr;
  auto ret = rtMalloc(PtrToPtr<uint8_t *, void *>(&buffer), size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[RtMalloc][Memory] failed, size = %zu, ret = %d", size, ret);
    REPORT_INNER_ERR_MSG("E19999", "rtMalloc failed, size = %zu, ret = %d.", size, ret);
    return nullptr;
  }
  ret = rtMemset(buffer, size, 0U, size);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[RtMemset][Memory] failed, ret = %d", ret);
    REPORT_INNER_ERR_MSG("E19999", "rtMemset failed, ret = %d.", ret);
    const auto rt_ret = rtFree(buffer);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[RtFree][Memory] failed"));
    return nullptr;
  }

  GELOGD("Malloc new memory succeeded. size = %zu", size);
  max_memory_size_ = size;
  auto mem_block = MakeUnique<MemBlock>(*this, buffer, size);
  GE_ASSERT_NOTNULL(mem_block);
  memory_list_.emplace_back(std::move(mem_block));
  return memory_list_.back().get();
}

StreamResource::StreamResource(const uintptr_t resource_id) : resource_id_(resource_id) {
}

StreamResource::~StreamResource() noexcept {
  for (const auto weight : weight_list_) {
    if (weight != nullptr) {
      const auto rt_ret = rtFree(weight);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Free][Rt] failed."));
    }
  }

  if (device_buffer_ != nullptr) {
    const auto rt_ret = rtFree(device_buffer_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Free][Rt] failed."));
  }

  if (callback_manager_ != nullptr) {
    (void)callback_manager_->Destroy();
  }
  FreeExMem();
}

Status StreamResource::Init() {
  const auto rt_ret = rtMalloc(&device_buffer_, kFuzzDeviceBufferSize, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Malloc][Rt] failed."));
  return SUCCESS;
}

SingleOp *StreamResource::GetOperator(const uint64_t key) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = op_map_.find(key);
  if (it == op_map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

Status StreamResource::DeleteOperator(const uint64_t key) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = op_map_.find(key);
  if (it != op_map_.end()) {
    // need to stream sync before erase
    GELOGI("static op %" PRIu64 " need to be deleted, start to sync stream %p", key, stream_);
    GE_CHK_RT_RET(rtStreamSynchronize(stream_));
    (void)op_map_.erase(it);
    GELOGI("static op %" PRIu64 " delete success", key);
  }
  return SUCCESS;
}

Status StreamResource::DeleteDynamicOperator(const uint64_t key) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = dynamic_op_map_.find(key);
  if (it != dynamic_op_map_.end()) {
    // need to stream sync before erase
    GELOGI("dynamic op %" PRIu64 " need to be deleted, start to sync stream %p", key, stream_);
    GE_CHK_RT_RET(rtStreamSynchronize(stream_));
    (void)dynamic_op_map_.erase(it);
    GELOGI("dynamic op %" PRIu64 " delete success", key);
  }
  return SUCCESS;
}

DynamicSingleOp *StreamResource::GetDynamicOperator(const uint64_t key) {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = dynamic_op_map_.find(key);
  if (it == dynamic_op_map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

rtStream_t StreamResource::GetStream() const {
  return stream_;
}

void StreamResource::SetStream(const rtStream_t stream) {
  stream_ = stream;
}

uint8_t *StreamResource::DoMallocMemory(const std::string &purpose, const size_t size, ge::MemBlock *&block) const {
  const auto mem = allocator_->Malloc(size);
  GE_ASSERT_NOTNULL(mem);
  GE_PRINT_DYNAMIC_MEMORY(AllocatorMalloc, purpose.c_str(), size);
  block = mem;
  return reinterpret_cast<uint8_t *>(block->GetAddr());
}

Status StreamResource::InitOverflowMemory() {
  const auto ret = rtCtxGetOverflowAddr(&overflow_addr_);
  GE_CHK_RT_RET(ret);
  return SUCCESS;
}

uint8_t *StreamResource::MallocMemory(const std::string &purpose, const size_t size, const bool holding_lock) {
  GELOGD("To Malloc memory, size = %zu", size);
  ge::MemBlock *block = nullptr;
  return MallocMemory(purpose, size, holding_lock, block);
}

uint8_t *StreamResource::MallocMemory(const std::string &purpose, const size_t size, const bool holding_lock,
                                      ge::MemBlock *&block) {
  GELOGD("To Malloc memory, size = %zu", size);
  if (holding_lock) {
    return DoMallocMemory(purpose, size, block);
  } else {
    const std::lock_guard<std::mutex> lk(stream_mu_);
    return DoMallocMemory(purpose, size, block);
  }
}

uint8_t *StreamResource::MallocWeight(const std::string &purpose, const size_t size) {
  GELOGD("To Malloc weight, size = %zu", size);
  uint8_t *buffer = nullptr;
  const auto ret = rtMalloc(PtrToPtr<uint8_t *, void *>(&buffer), size, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[RtMalloc][Memory] failed, size = %zu, ret = %d", size, ret);
    REPORT_INNER_ERR_MSG("E19999", "rtMalloc failed, size = %zu, ret = %d.", size, ret);
    return nullptr;
  }

  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, purpose.c_str(), size);
  weight_list_.emplace_back(buffer);
  return buffer;
}

Status StreamResource::BuildDynamicOperator(const ModelData &model_data,
                                            DynamicSingleOp **const single_op,
                                            const uint64_t model_id) {
  const std::string &model_name = std::to_string(model_id);
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = dynamic_op_map_.find(model_id);
  if (it != dynamic_op_map_.end()) {
    *single_op = it->second.get();
    return SUCCESS;
  }

  SingleOpModel model(model_name, model_data.model_data, static_cast<uint32_t>(model_data.model_len));
  const auto ret = model.Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][SingleOpModel] failed. model = %s, ret = %u", model_name.c_str(), ret);
    REPORT_INNER_ERR_MSG("E19999", "SingleOpModel init failed, model = %s, ret = %u", model_name.c_str(), ret);
    return ret;
  }

  auto new_op = MakeUnique<DynamicSingleOp>(&tensor_pool_, resource_id_, &stream_mu_, stream_);
  GE_CHECK_NOTNULL(new_op);
  GE_CHECK_NOTNULL(new_op->impl_);

  GELOGI("To build operator: %s", model_name.c_str());
  GE_CHK_STATUS_RET(model.BuildDynamicOp(*this, *new_op->impl_), "[Build][DynamicOp]failed. Op:%s", model_name.c_str());
  *single_op = new_op.get();
  dynamic_op_map_[model_id] = std::move(new_op);
  return SUCCESS;
}

Status StreamResource::BuildOperator(const ModelData &model_data, SingleOp **const single_op, const uint64_t model_id) {
  const std::string &model_name = std::to_string(model_id);
  const std::lock_guard<std::mutex> lk(mu_);
  const auto it = op_map_.find(model_id);
  if (it != op_map_.end()) {
    *single_op = it->second.get();
    return SUCCESS;
  }

  SingleOpModel model(model_name, model_data.model_data, static_cast<uint32_t>(model_data.model_len));
  const auto ret = model.Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][SingleOpModel] failed. model = %s, ret = %u", model_name.c_str(), ret);
    REPORT_INNER_ERR_MSG("E19999", "SingleOpModel init failed, model = %s, ret = %u", model_name.c_str(), ret);
    return ret;
  }

  auto new_op = MakeUnique<SingleOp>(this, &stream_mu_, stream_);
  GE_CHECK_NOTNULL(new_op);
  GE_CHECK_NOTNULL(new_op->impl_);

  GELOGI("To build operator: %s", model_name.c_str());
  GE_CHK_STATUS_RET(model.BuildOp(*this, *new_op->impl_), "[Build][SingleOp] failed. Op:%s", model_name.c_str());

  *single_op = new_op.get();
  op_map_[model_id] = std::move(new_op);
  return SUCCESS;
}

Status StreamResource::GetThreadPool(ThreadPool **const thread_pool) {
  GE_CHECK_NOTNULL(thread_pool);
  if (thread_pool_ == nullptr) {
    thread_pool_ = MakeUnique<ThreadPool>("ge_prepare", kThreadNumDefault, false);
    GE_CHECK_NOTNULL(thread_pool_);
  }
  *thread_pool = thread_pool_.get();
  return SUCCESS;
}

Status StreamResource::GetCallbackManager(hybrid::CallbackManager **const callback_manager) {
  GE_CHECK_NOTNULL(callback_manager);
  if (callback_manager_ == nullptr) {
    callback_manager_ = MakeUnique<hybrid::RtCallbackManager>();
    GE_CHECK_NOTNULL(callback_manager_);
    GE_CHK_STATUS_RET_NOLOG(callback_manager_->Init());
  }
  *callback_manager = callback_manager_.get();
  return SUCCESS;
}

Status StreamResource::MallocExMem(const uint32_t device_id, RuntimeParam &runtime_param) {
  GE_CHK_STATUS_RET(ModelUtils::MallocExMem(device_id, runtime_param), "MallocExMem failed.");
  device_2_meminfos_[device_id].push_back(runtime_param.memory_infos);
  return SUCCESS;
}

void StreamResource::FreeExMem() {
  RuntimeParam runtime_param;
  for (auto &device_id_2_meminfos : device_2_meminfos_) {
    for (auto &meminfo : device_id_2_meminfos.second) {
      runtime_param.memory_infos = meminfo;
      ModelUtils::FreeExMem(device_id_2_meminfos.first, runtime_param);
    }
  }
  device_2_meminfos_.clear();
}
}  // namespace ge
