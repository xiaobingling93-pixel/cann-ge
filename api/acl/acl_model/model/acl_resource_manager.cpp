/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_resource_manager.h"
#include "framework/runtime/gert_api.h"
#include "framework/common/profiling_definitions.h"
#include "framework/memory/allocator_desc.h"
#include "framework/executor/ge_executor.h"
#include "mmpa/mmpa_api.h"
#include "common/log_inner.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "ge/ge_allocator.h"
#include "acl/acl_rt.h"

namespace {
std::atomic<std::uint64_t> atomicModelId(0UL);
}
namespace acl {
namespace {
constexpr int32_t OM_FILE_SUFFIX_LEN = 3;
constexpr int32_t OM_DIR_MAX_DEPTH = 3;
constexpr int32_t DECIMAL = 10;
const std::string ACL_MAX_OPQUEUE_NUM = "max_opqueue_num";
}

AclResourceManager::AclResourceManager() {
    GetRuntimeV2Env();
}

void AclResourceManager::AddBundleSubmodelId(const uint32_t bundleId, uint32_t modelId)
{
  const std::lock_guard<std::mutex> locker(mutex_);
  bundleInfos_[bundleId].loadedSubModelIdSet.insert(modelId);
  (void)bundleInnerIds_.insert(modelId);
}

aclError AclResourceManager::SetBundleInfo(const uint32_t bundleId, const BundleModelInfo &bundleInfos)
{
  const std::lock_guard<std::mutex> locker(mutex_);
  bundleInfos_[bundleId] = bundleInfos;
  for (const auto &modelId : bundleInfos.loadedSubModelIdSet) {
    (void)bundleInnerIds_.insert(modelId);
  }
  return ACL_SUCCESS;
}

void AclResourceManager::DeleteBundleSubmodelId(const uint32_t bundleId, uint32_t modelId)
{
  const std::lock_guard<std::mutex> locker(mutex_);
  bundleInfos_[bundleId].loadedSubModelIdSet.erase(modelId);
  (void)bundleInnerIds_.erase(modelId);
}

aclError AclResourceManager::GetBundleInfo(const uint32_t bundleId, BundleModelInfo &bundleInfos)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    const auto it = bundleInfos_.find(bundleId);
    if (it == bundleInfos_.end()) {
        ACL_LOG_ERROR("This model %u is not bundle model, can not get bundle info.", bundleId);
        return ACL_ERROR_INVALID_BUNDLE_MODEL_ID;
    }
    bundleInfos = it->second;
    return ACL_SUCCESS;
}

bool AclResourceManager::IsBundleInnerId(const uint32_t modelId)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    return (bundleInnerIds_.count(modelId) > 0U);
}

void AclResourceManager::DeleteBundleInfo(const uint32_t bundleId)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    const auto it = bundleInfos_.find(bundleId);
    if (it != bundleInfos_.end()) {
        for (const auto &id : it->second.loadedSubModelIdSet) {
            (void)bundleInnerIds_.erase(id);
        }
        (void)bundleInfos_.erase(it);
    }
}

void AclResourceManager::AddExecutor(uint32_t &modelId, std::unique_ptr<gert::ModelV2Executor> &&executor,
                                     const std::shared_ptr<gert::RtSession> &rtSession)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    ++modelIdGenerator_;
    modelId = modelIdGenerator_.load();
    executorMap_[modelId] = std::move(executor);
    rtSessionMap_[modelId] = rtSession;
}

void AclResourceManager::AddOm2Executor(uint32_t &modelId, std::unique_ptr<gert::Om2ModelExecutor> &&executor,
                                        const std::shared_ptr<gert::RtSession> &rtSession) {
    const std::lock_guard<std::mutex> locker(mutex_);
    ++modelIdGenerator_;
    modelId = modelIdGenerator_.load();
    om2ExecutorMap_[modelId] = std::move(executor);
    rtSessionMap_[modelId] = rtSession;
}

aclError AclResourceManager::DeleteExecutor(const uint32_t modelId)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    const auto iter = executorMap_.find(modelId);
    if (iter == executorMap_.end()) {
        ACL_LOG_ERROR("model is not loaded, modelId is %u", modelId);
        return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
    }
    (void)executorMap_.erase(iter);

    const auto it = rtSessionMap_.find(modelId);
    if (it == rtSessionMap_.end()) {
        ACL_LOG_ERROR("model is not loaded, modelId is %u", modelId);
        return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
    }
    (void)rtSessionMap_.erase(it);
    return ACL_SUCCESS;
}

aclError AclResourceManager::DeleteOm2Executor(const uint32_t modelId)
{
  const std::lock_guard<std::mutex> locker(mutex_);
  const auto iter = om2ExecutorMap_.find(modelId);
  if (iter == om2ExecutorMap_.end()) {
    ACL_LOG_ERROR("model is not loaded, modelId is %u", modelId);
    return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  }
  (void)om2ExecutorMap_.erase(iter);

  const auto it = rtSessionMap_.find(modelId);
  if (it == rtSessionMap_.end()) {
    ACL_LOG_ERROR("model is not loaded, modelId is %u", modelId);
    return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  }
  (void)rtSessionMap_.erase(it);
  return ACL_SUCCESS;
}

void AclResourceManager::GetRuntimeV2Env()
{
    const char_t *enableRuntimeV2Flag = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ENABLE_RUNTIME_V2, enableRuntimeV2Flag);
    if (enableRuntimeV2Flag != nullptr) {
        if (enableRuntimeV2Flag[0] == '0') { // 0 both model and singleOp disable
            enableRuntimeV2ForModel_ = false;
            enableRuntimeV2ForSingleOp_ = false;
        } else if (enableRuntimeV2Flag[0] == '2') { // 2: model enable, singleOp disable
            enableRuntimeV2ForModel_ = true;
            enableRuntimeV2ForSingleOp_ = false;
        } else {
            enableRuntimeV2ForModel_ = true;
            enableRuntimeV2ForSingleOp_ = true;
        }
    }
    ACL_LOG_EVENT("runtime v2 flag : model flag = %d, singleOp flag = %d",
                  static_cast<int32_t>(enableRuntimeV2ForModel_),
                  static_cast<int32_t>(enableRuntimeV2ForSingleOp_));
}

std::shared_ptr<gert::ModelV2Executor> AclResourceManager::GetExecutor(const uint32_t modelId)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    const auto iter = executorMap_.find(modelId);
    if (iter == executorMap_.end()) {
        return nullptr;
    }
    return iter->second;
}

std::shared_ptr<gert::Om2ModelExecutor> AclResourceManager::GetOm2Executor(const uint32_t modelId)
{
  const std::lock_guard<std::mutex> locker(mutex_);
  const auto iter = om2ExecutorMap_.find(modelId);
  if (iter == om2ExecutorMap_.end()) {
    return nullptr;
  }
  return iter->second;
}

std::shared_ptr<gert::RtSession> AclResourceManager::CreateRtSession()
{
    const std::lock_guard<std::mutex> locker(mutex_);
    ++sessionIdGenerator_;
    auto sessionId = sessionIdGenerator_.load();
    return std::make_shared<gert::RtSession>(sessionId);
}

std::shared_ptr<gert::RtSession> AclResourceManager::GetRtSession(const uint32_t rtSessionId)
{
    const std::lock_guard<std::mutex> locker(mutex_);
    const auto iter = rtSessionMap_.find(rtSessionId);
    if (iter == rtSessionMap_.cend()) {
        return nullptr;
    }
    return iter->second;
}

void *AclResourceManager::GetKeyByStreamOrDefaultStream(const aclrtStream stream)
{
    if (stream != nullptr) {
        return stream;
    }
    // get current context default stream
    aclrtStream curCtxDefaultStream = nullptr;
    const aclError aclErr = aclrtCtxGetCurrentDefaultStream(&curCtxDefaultStream);
    if (aclErr != ACL_ERROR_NONE) {
        ACL_LOG_CALL_ERROR("get current default stream failed, ret:%d", static_cast<int32_t>(aclErr));
        return nullptr;
    }
    return curCtxDefaultStream;
}

std::shared_ptr<gert::Allocators> AclResourceManager::GetAllocators(const aclrtStream stream, bool createDefaultAllocator)
{
    void *cacheKey = stream;
    if (cacheKey == nullptr) {
        cacheKey = GetKeyByStreamOrDefaultStream(stream);
    }

    if (cacheKey == nullptr) {
        return nullptr;
    }

    // If the user does not initially register an external allocator, then create a default allocator
    // and use the default allocator throughout the entire lifecycle of the stream (even if the user
    // later registers an external allocator for the stream).
    const std::unique_lock<std::recursive_mutex> lk(streamAllocatorMutex_);
    // Try using exist default allocator.
    const auto iter = streamDefaultAllocator_.find(cacheKey);
    if (iter != streamDefaultAllocator_.end()) {
        return iter->second;
    }
    // Try using exist or create external allocator.
    std::shared_ptr<gert::Allocators> externalAllocators = UpdateExternalAllocators(stream);
    if (externalAllocators != nullptr) {
        return externalAllocators;
    }
    // For model inference (dynamic shape), the default Allocator is not created to improve performance.
    if (!createDefaultAllocator) {
        ACL_LOG_INFO("The default deviceAllocator is not created.");
        return nullptr;
    }
    return CreateDefaultAllocators(cacheKey);
}

std::shared_ptr<gert::Allocators> AclResourceManager::CreateAllocators(std::shared_ptr<ge::Allocator> &deviceAllocator)
{
    std::shared_ptr<ge::Allocator> hostAllocator(gert::AllocatorFactory::Create(gert::kOnHost).release());
    if ((hostAllocator == nullptr)) {
        ACL_LOG_ERROR("hostAllocator is nullptr");
        return nullptr;
    }
    // create a new alloctors
    std::shared_ptr<gert::Allocators> allocators(new (std::nothrow) gert::Allocators());
    ACL_REQUIRES_NOT_NULL_RET_NULL(allocators);
    // only support allocator with placement kOnDeviceHbm, kOnHost, kFollowing
    for (size_t i = 0U; i < static_cast<size_t>(gert::AllocatorUsage::kEnd); ++i) {
        (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kOnDeviceHbm), i, deviceAllocator);
        (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kOnHost), i, hostAllocator);
        (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kFollowing), i, hostAllocator);
    }
    return allocators;
}

std::shared_ptr<gert::Allocators> AclResourceManager::CreateDefaultAllocators(const void * const cacheKey)
{
    std::shared_ptr<ge::Allocator> deviceAllocator(
        gert::AllocatorFactory::Create(gert::kOnDeviceHbm).release());
    if (deviceAllocator == nullptr) {
        return nullptr;
    }
    std::shared_ptr<gert::Allocators> allocators = CreateAllocators(deviceAllocator);
    ACL_REQUIRES_NOT_NULL_RET_NULL(allocators);
    (void)streamDefaultAllocator_.insert({cacheKey, allocators});
    return allocators;
}

std::shared_ptr<gert::Allocators> AclResourceManager::CreateExternalAllocators(const void * const cacheKey, ExternalAllocatorDesc &allocatorDesc)
{
    ge::AllocatorDesc geAllocatorDesc;
    geAllocatorDesc.obj = allocatorDesc.obj;
    geAllocatorDesc.alloc_func = allocatorDesc.allocFunc;
    geAllocatorDesc.free_func = allocatorDesc.freeFunc;
    geAllocatorDesc.alloc_advise_func = allocatorDesc.allocAdviseFunc;
    geAllocatorDesc.get_addr_from_block_func = allocatorDesc.getAddrFromBlockFunc;
    std::shared_ptr<ge::Allocator> deviceAllocator(gert::CreateExternalAllocator(&geAllocatorDesc).release());
    if (deviceAllocator == nullptr) {
        return nullptr;
    }
    std::shared_ptr<gert::Allocators> allocators = CreateAllocators(deviceAllocator);
    ACL_REQUIRES_NOT_NULL_RET_NULL(allocators);
    (void)streamExternalAllocator_.insert({cacheKey, std::make_pair(allocatorDesc, allocators)});
    return allocators;
}

void AclResourceManager::CleanAllocators(const void * const cacheKey)
{
    if (cacheKey == nullptr) {
        return;
    }
    const std::unique_lock<std::recursive_mutex> lk(streamAllocatorMutex_);
    (void)streamDefaultAllocator_.erase(cacheKey);
    (void)streamExternalAllocator_.erase(cacheKey);
}

std::shared_ptr<gert::Allocators> AclResourceManager::UpdateExternalAllocators(aclrtStream stream)
{
    void *cacheKey = stream;
    aclrtAllocatorDesc new_desc = nullptr;
    aclrtAllocator allocator = nullptr;
    aclrtAllocatorAllocFunc allocFunc = nullptr;
    aclrtAllocatorFreeFunc freeFunc = nullptr;
    aclrtAllocatorAllocAdviseFunc allocAdviseFunc = nullptr;
    aclrtAllocatorGetAddrFromBlockFunc getAddrFromBlockFunc = nullptr;
    bool new_desc_exist = aclrtAllocatorGetByStream(stream, &new_desc, &allocator, &allocFunc,
                            &freeFunc, &allocAdviseFunc, &getAddrFromBlockFunc) == ACL_SUCCESS;
    const auto iter_old_desc = streamExternalAllocator_.find(cacheKey);
    bool old_desc_exist = iter_old_desc != streamExternalAllocator_.end();
    // "old_desc_exist" indicates whether the external allocator stored by streamExternalAllocator_ exists (referred
    // to as the old allocator desc). "new_desc_exist" indicates whether the user has newly registered the desc
    // (referred to as the new "allocator desc"). Based on the old allocator description and whether the user has
    // newly registered the allocator description, there are four possible scenarios.
    // Scenario 1: If the user has neither registered an external allocator before nor currently, simply return nullptr.
    if (!old_desc_exist && !new_desc_exist) {
        return nullptr;
    }
    // Situation 2: The user has newly registered an allocator desc and has never registered before. We will create an
    // external allocator for the user.
    if (!old_desc_exist && new_desc_exist) {
        ExternalAllocatorDesc allocatorDesc = ExternalAllocatorDesc(allocator, allocFunc, freeFunc, allocAdviseFunc, getAddrFromBlockFunc);
        return CreateExternalAllocators(cacheKey, allocatorDesc);
    }
    // Situation 3: The user previously registered for "allocator desc", but in their most recent query, they have
    // unregistered "allocator desc". We delete the previously created external allocator for the user.
    if (old_desc_exist && !new_desc_exist) {
        (void)streamExternalAllocator_.erase(cacheKey);
        return nullptr;
    }
    // Situation 4: The user has previously registered with the "allocator desc" and the latest query has confirmed
    // the user's registration. We compare the latest allocator description retrieved with the previously registered
    // ones. If they are equal, it indicates that the user has not registered a new allocator description.In this case,
    // we can still use the previously saved allocator. If they are not equal, it indicates that the user has registered
    // a new allocator description to replace the old one. We need to delete the old allocator and create a new one for
    // the user.
    if (old_desc_exist && new_desc_exist) {
        ExternalAllocatorDesc allocatorDesc = ExternalAllocatorDesc(allocator, allocFunc, freeFunc, allocAdviseFunc, getAddrFromBlockFunc);
        if (allocatorDesc == iter_old_desc->second.first) {
            return iter_old_desc->second.second;
        }
        (void)streamExternalAllocator_.erase(cacheKey);
        return CreateExternalAllocators(cacheKey, allocatorDesc);
    }
    return nullptr;
}

AclResourceManager::~AclResourceManager()
{
    // note: op_model的executor的释放依赖allocator，所以先释放executor，再释放allocator
    streamDefaultAllocator_.clear();
    streamExternalAllocator_.clear();
}

void AclResourceManager::HandleReleaseSourceByDevice(int32_t deviceId, aclrtDeviceState state, void *args) const
{
    (void)args;
    ACL_LOG_INFO("start to execute HandleReleaseSourceByDevice, devId:%d.", deviceId);
    if (state != ACL_RT_DEVICE_STATE_RESET_PRE) {
        ACL_LOG_INFO("it's not reset pre device callback, currently do nothing.");
        return;
    }
    (void)ge::GeExecutor::ReleaseResource();
    ACL_LOG_INFO("successfully execute HandleReleaseSourceByDevice, devId:%d.", deviceId);
}

void AclResourceManager::HandleReleaseSourceByStream(aclrtStream stream, aclrtStreamState state, void *args)
{
    (void)args;
    ACL_LOG_INFO("start to execute HandleReleaseSourceByStream.");
    if (state != ACL_RT_STREAM_STATE_DESTROY_PRE) {
        ACL_LOG_INFO("it's not destroy stream callback, currently do nothing.");
        return;
    }
    (void)CleanAllocators(stream);
    ACL_LOG_INFO("successfully execute HandleReleaseSourceByStream.");
}

AclResourceManager &AclResourceManager::GetInstance()
{
  static AclResourceManager instance;
  return instance;
}
}

