/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACL_RESOURCE_MANAGER_H
#define ASCEND_ACL_RESOURCE_MANAGER_H

#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include "acl/acl_base.h"
#include "model_desc_internal.h"
#include "graph/ge_attr_value.h"
#include "mmpa/mmpa_api.h"
#include "framework/runtime/mem_allocator.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/stream_executor.h"
#include "ge/ge_allocator.h"
#include "acl/acl_rt_allocator.h"
#include "acl/acl_rt.h"

namespace acl {
class ExternalAllocatorDesc {
public:
    ExternalAllocatorDesc(): obj(nullptr), allocFunc(nullptr), freeFunc(nullptr), allocAdviseFunc(nullptr), getAddrFromBlockFunc(nullptr) {}
    ExternalAllocatorDesc(aclrtAllocator allocator,
                     aclrtAllocatorAllocFunc allocFunction,
                     aclrtAllocatorFreeFunc freeFunction,
                     aclrtAllocatorAllocAdviseFunc allocAdviseFunction,
                     aclrtAllocatorGetAddrFromBlockFunc getAddrFromBlockFunction) :
                     obj(allocator),
                     allocFunc(allocFunction),
                     freeFunc(freeFunction),
                     allocAdviseFunc(allocAdviseFunction),
                     getAddrFromBlockFunc(getAddrFromBlockFunction) {}
    ~ExternalAllocatorDesc() {}
    bool operator==(const ExternalAllocatorDesc &allocatorDesc) const
    {
        return obj == allocatorDesc.obj &&
                allocFunc == allocatorDesc.allocFunc &&
                freeFunc == allocatorDesc.freeFunc &&
                allocAdviseFunc == allocatorDesc.allocAdviseFunc &&
                getAddrFromBlockFunc == allocatorDesc.getAddrFromBlockFunc;
    }
    aclrtAllocator obj;
    aclrtAllocatorAllocFunc allocFunc;
    aclrtAllocatorFreeFunc freeFunc;
    aclrtAllocatorAllocAdviseFunc allocAdviseFunc;
    aclrtAllocatorGetAddrFromBlockFunc getAddrFromBlockFunc;
};

struct BundleModelInfo {
  bool isInit = false; // aclmdlBundleInit scene means true
  std::shared_ptr<gert::RtSession> rtSession;
  size_t varSize = 0U;
  std::string fromFilePath;
  std::shared_ptr<const uint8_t> bundleModelData;
  size_t bundleModelSize = 0U;
  std::vector<BundleSubModelInfo> subModelInfos;
  std::vector<uint32_t> loadedSubModelId; // aclmdlBundleGetModelId use this when aclmdlBundleLoadFromxx is called
  std::set<uint32_t> loadedSubModelIdSet;
};

class ACL_FUNC_VISIBILITY AclResourceManager {
public:
    ~AclResourceManager();

    static AclResourceManager &GetInstance()
    {
        static AclResourceManager instance;
        return instance;
    }

    // executor
    bool IsRuntimeV2Enable(bool isModel) const
    {
        return isModel ? enableRuntimeV2ForModel_ : enableRuntimeV2ForSingleOp_;
    }

    void AddExecutor(uint32_t &modelId, std::unique_ptr<gert::ModelV2Executor> &&executor,
                     const std::shared_ptr<gert::RtSession> &rtSession);
    std::shared_ptr<gert::ModelV2Executor> GetExecutor(const uint32_t modelId);
    aclError DeleteExecutor(const uint32_t modelId);

    std::shared_ptr<gert::RtSession> CreateRtSession();
    std::shared_ptr<gert::RtSession> GetRtSession(const uint32_t rtSessionId);
    // allocator
    std::shared_ptr<gert::Allocators> GetAllocators(const aclrtStream stream, bool createDefaultAllocator = true);
    void CleanAllocators(const void * const cacheKey);
    std::shared_ptr<gert::Allocators> CreateExternalAllocators(const void * const cacheKey,
                                                               ExternalAllocatorDesc &allocatorDesc);
    std::shared_ptr<gert::Allocators> CreateDefaultAllocators(const void * const cacheKey);
    std::shared_ptr<gert::Allocators> UpdateExternalAllocators(aclrtStream stream);
    std::shared_ptr<gert::Allocators> CreateAllocators(std::shared_ptr<ge::Allocator> &deviceAllocator);

    static void *GetKeyByStreamOrDefaultStream(const aclrtStream stream);

    std::shared_ptr<ge::Allocator> GetDeviceAllocator(const aclrtStream stream, bool createDefaultAllocator = true);

    void AddBundleSubmodelId(const uint32_t bundleId, uint32_t modelId);

    void DeleteBundleSubmodelId(const uint32_t bundleId, uint32_t modelId);

    aclError SetBundleInfo(const uint32_t bundleId, const BundleModelInfo &bundleInfos);

    aclError GetBundleInfo(const uint32_t bundleId, BundleModelInfo &bundleInfos);

    void DeleteBundleInfo(const uint32_t bundleId);

    bool IsBundleInnerId(const uint32_t modelId);

    void HandleReleaseSourceByStream(aclrtStream stream, aclrtStreamState state, void *args);
    void HandleReleaseSourceByDevice(int32_t deviceId, aclrtDeviceState state, void *args) const;
private:
    AclResourceManager();
    void GetRuntimeV2Env();
private:
    // executor
    // model id 0 is invalid value
    std::unordered_map<uint32_t, std::shared_ptr<gert::ModelV2Executor>> executorMap_{{0U, nullptr}};
    std::atomic_uint32_t modelIdGenerator_ {std::numeric_limits<uint32_t>::max() / 2U};
    std::atomic_uint64_t sessionIdGenerator_ {std::numeric_limits<uint64_t>::max() / 2U};
    bool enableRuntimeV2ForModel_ = true;
    bool enableRuntimeV2ForSingleOp_ = true;
    std::mutex mutex_;
    std::map<uint32_t, std::shared_ptr<gert::RtSession>> rtSessionMap_;

    // allocator
    // note: op_model的executor的释放依赖allocator
    std::map<const void *, std::shared_ptr<gert::Allocators>> streamDefaultAllocator_; // cacheKey + default_allocator_handle
    std::map<const void *, std::pair<ExternalAllocatorDesc, std::shared_ptr<gert::Allocators>>> streamExternalAllocator_;
    std::recursive_mutex streamAllocatorMutex_;

    std::unordered_map<uint32_t, BundleModelInfo> bundleInfos_;
    std::unordered_set<uint32_t> bundleInnerIds_;
};
}

#endif // ASCEND_ACL_RESOURCE_MANAGER_H
