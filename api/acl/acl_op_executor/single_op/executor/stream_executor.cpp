/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_executor.h"
#include "common/common_inner.h"
#include "single_op/compile/op_kernel_selector.h"
#include "op_task.h"
#include "securec.h"
#include "utils/math_utils.h"
#include "utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"

namespace acl {
namespace {
constexpr const size_t MAX_WORKSPACES = 16U;
constexpr size_t DATA_MEMORY_ALIGN_SIZE = 32UL;

aclError GetAlignedAndPaddingSize(const size_t size, const bool isPadding, size_t &alignedSize)
{
  const size_t padding_size = static_cast<size_t>(ge::TensorUtils::GetPaddingSize());
  const size_t appendSize = isPadding ? (DATA_MEMORY_ALIGN_SIZE + padding_size)
                                      : DATA_MEMORY_ALIGN_SIZE;
  if ((size + appendSize) < size) {
    ACL_LOG_INNER_ERROR("[Check][Size]size too large: %zu", size);
    return ACL_ERROR_INVALID_PARAM;
  }

  alignedSize = (size + appendSize - 1UL) / DATA_MEMORY_ALIGN_SIZE * DATA_MEMORY_ALIGN_SIZE;
  return ACL_SUCCESS;
}
}

std::recursive_mutex Executors::mu;
std::map<uintptr_t, std::unique_ptr<StreamExecutor>> Executors::executors;

StreamExecutor::StreamExecutor(ResourceManager *const resourceMgr, const aclrtStream aclStream)
    : resMgr_(resourceMgr), stream_(aclStream)
{
}

aclError StreamExecutor::ExecuteAsync(const AclOp &aclOpDesc,
                                      const aclDataBuffer *const *const inputs,
                                      aclDataBuffer *const *const outputs)
{
    std::shared_ptr<OpKernelDesc> desc;
    const auto ret = OpKernelSelector::GetInstance().GetOpKernelDesc(aclOpDesc, desc);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][OpKernelDesc]Op with the given shape was not compiled. aclOp = %s",
            aclOpDesc.DebugString().c_str());
        return ret;
    }

    ACL_REQUIRES_NOT_NULL(desc);
    ACL_REQUIRES_OK(ExecuteAsync(*desc, aclOpDesc.numInputs, inputs, aclOpDesc.numOutputs, outputs));
    return ACL_SUCCESS;
}

aclError StreamExecutor::ExecuteAsync(OpKernelDesc &kernelDesc,
                                      const int32_t numInputs,
                                      const aclDataBuffer *const *const inputs,
                                      const int32_t numOutputs,
                                      aclDataBuffer *const *const outputs)
{
    ACL_LOG_DEBUG("Start to execute op by dynamic kernel");
    kernelDesc.timestamp = attr_utils::GetCurrentTimestamp();
    TbeOpTask task(kernelDesc.stubFunc, kernelDesc.blockDim);
    ACL_REQUIRES_OK(InitTbeTask(kernelDesc, numInputs, numOutputs, task));
    ACL_REQUIRES_OK(task.ExecuteAsync(numInputs, inputs, numOutputs, outputs, stream_));
    return ACL_SUCCESS;
}

aclError StreamExecutor::InitTbeTask(const OpKernelDesc &desc, const int32_t numInputs,
    const int32_t numOutputs, TbeOpTask &task)
{
    // create new op task
    const size_t numWorkSpaces = desc.workspaceSizes.size();
    if (numWorkSpaces > MAX_WORKSPACES) {
        ACL_LOG_INNER_ERROR("[Check][numWorkSpaces]numWorkSpaces invalid, "
            "numWorkSpace[%zu] is larger than MAX_WORKSPACES[%zu]",
            numWorkSpaces, MAX_WORKSPACES);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::vector<uintptr_t> workspaces;
    ACL_REQUIRES_OK(AllocateWorkspaces(desc.workspaceSizes, workspaces));

    // assemble args
    const std::string &tilingDesc = desc.extendArgs;
    int32_t sum = 0;
    ACL_CHECK_ASSIGN_INT32_ADD(numInputs, numOutputs, sum);
    size_t numArgs = 0U;
    ACL_CHECK_ASSIGN_SIZET_ADD(static_cast<size_t>(sum), numWorkSpaces, numArgs);
    size_t numSize = 0U;
    ACL_CHECK_ASSIGN_SIZET_MULTI(numArgs, sizeof(void *), numSize);
    size_t argSize = 0U;
    const size_t descSize = tilingDesc.size();
    ACL_CHECK_ASSIGN_SIZET_ADD(numSize, descSize, argSize);

    auto args = std::unique_ptr<uint8_t[]>(new(std::nothrow)uint8_t[argSize]);
    ACL_CHECK_MALLOC_RESULT(args);
    auto *const argBase = args.get();

    // set workspace addresses
    auto *workspaceBase = reinterpret_cast<uintptr_t *>(argBase) + numInputs + numOutputs;
    for (const uintptr_t wsAddr : workspaces) {
        *workspaceBase = wsAddr;
        workspaceBase++;
    }

    // set tiling
    if (!tilingDesc.empty()) {
        void *const tilingStart = argBase + (numArgs * sizeof(void *));
        ACL_LOG_DEBUG("tiling desc size = %zu", tilingDesc.size());
        if (memcpy_s(tilingStart, tilingDesc.size(), tilingDesc.data(), tilingDesc.size()) != EOK) {
            ACL_LOG_INNER_ERROR("[Check][Memcpy]Invoking memcpy_s failed");
            return ACL_ERROR_FAILURE;
        }
    }

    task.SetArgs(std::move(args), static_cast<uint32_t>(argSize));
    return ACL_SUCCESS;
}

aclError StreamExecutor::AllocateWorkspaces(const std::vector<size_t> &workspaceSizes,
    std::vector<uintptr_t> &workspaces)
{
    const auto numWs = workspaceSizes.size();
    if (numWs > MAX_WORKSPACES) {
        ACL_LOG_INNER_ERROR("[Check][numWs]numWs invalid, numWs[%zu] is larger than MAX_WORKSPACES[%zu]",
            numWs, MAX_WORKSPACES);
        return ACL_ERROR_INVALID_PARAM;
    }

    size_t totalSize = 0U;
    std::vector<uintptr_t> offsets;
    uintptr_t offset = 0U;
    for (const size_t wsSize : workspaceSizes) {
        offsets.emplace_back(offset);
        size_t alignedSize = 0U;
        ACL_REQUIRES_OK(GetAlignedAndPaddingSize(wsSize, true, alignedSize));
        totalSize += alignedSize;
        ACL_CHECK_ASSIGN_SIZET_ADD(totalSize, alignedSize, totalSize);
        offset += alignedSize;
        ACL_CHECK_ASSIGN_SIZET_ADD(offset, alignedSize, offset);
    }

    const std::lock_guard<std::mutex> lk(mu_);
    void *wsMemory = nullptr;
    ACL_REQUIRES_OK(resMgr_->GetMemory(&wsMemory, totalSize));

    const auto wsBase = reinterpret_cast<uintptr_t>(wsMemory);
    for (const uintptr_t wsOffset : offsets) {
        workspaces.emplace_back(wsBase + wsOffset);
    }

    return ACL_SUCCESS;
}

StreamExecutor::~StreamExecutor()
{
    ACL_LOG_INFO("StreamExecutor::~StreamExecutor IN");
}


StreamExecutor *Executors::GetOrCreate(const aclrtContext context, const aclrtStream stream)
{
    auto key = reinterpret_cast<uintptr_t>(stream);
    if (stream == nullptr)
    {
        // get current context default stream
        aclrtStream curCtxDefaultStream = nullptr;
        const aclError aclErr = aclrtCtxGetCurrentDefaultStream(&curCtxDefaultStream);
        if (aclErr != ACL_ERROR_NONE) {
            ACL_LOG_CALL_ERROR("get current default stream failed, ret:%d", static_cast<int32_t>(aclErr));
            return nullptr;
        }
        key = reinterpret_cast<uintptr_t>(curCtxDefaultStream);
        ACL_LOG_INFO("use current context default stream as resource key.");
    }
    const std::lock_guard<std::recursive_mutex> lk(mu);
    const auto it = executors.find(key);
    if (it != executors.end()) {
        return it->second.get();
    }

    auto *resMgr = new(std::nothrow) ResourceManager(context);
    if (resMgr == nullptr) {
        return nullptr;
    }

    auto *const executor = new(std::nothrow) StreamExecutor(resMgr, stream);
    if (executor == nullptr) {
        ACL_DELETE_AND_SET_NULL(resMgr);
        return nullptr;
    }

    (void)executors.emplace(key, std::unique_ptr<StreamExecutor>(executor));
    return executor;
}

void Executors::RemoveExecutor(const aclrtStream stream)
{
    auto key = reinterpret_cast<uintptr_t>(stream);
    if (key != 0U) {
        ACL_LOG_INFO("To remove executor by stream = %lu", key);
    } else {
        // get current context default stream
        aclrtStream curCtxDefaultStream = nullptr;
        const aclError aclErr = aclrtCtxGetCurrentDefaultStream(&curCtxDefaultStream);
        if (aclErr != ACL_ERROR_NONE) {
            ACL_LOG_CALL_ERROR("get current default stream failed, ret:%d", static_cast<int32_t>(aclErr));
            return;
        }
        key = reinterpret_cast<uintptr_t>(curCtxDefaultStream);
        ACL_LOG_INFO("To remove executor by current context default stream = %lu", key);
    }
    const std::lock_guard<std::recursive_mutex> lk(mu);
    (void)executors.erase(key);
}
} // namespace acl
