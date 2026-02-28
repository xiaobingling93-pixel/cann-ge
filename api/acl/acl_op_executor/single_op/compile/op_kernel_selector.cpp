/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_kernel_selector.h"
#include <map>
#include <memory>
#include "common/log_inner.h"
#include "op_kernel_registry.h"
#include "utils/attr_utils.h"

namespace {
    constexpr uint64_t DEFAULT_MAX_OP_NUM_FOR_HANDLE = UINT64_MAX;
}

namespace acl {
OpKernelSelector::OpKernelSelector()
{
    kernelDescMap_.SetMaxOpNum(DEFAULT_MAX_OP_NUM_FOR_HANDLE);
}

bool OpKernelSelector::Register(const std::string &opType, aclopCompileFunc func)
{
    const std::lock_guard<std::mutex> lk(mu_);
    const auto iter = selectors_.emplace(opType, func);
    return iter.second;
}

void OpKernelSelector::Unregister(const std::string &opType)
{
    const std::lock_guard<std::mutex> lk(mu_);
    (void)selectors_.erase(opType);
}

aclopCompileFunc OpKernelSelector::GetSelectFunc(const std::string &opType)
{
    const std::lock_guard<std::mutex> lk(mu_);
    aclopCompileFunc func = nullptr;
    std::map<std::string, aclopCompileFunc>::const_iterator iter = selectors_.find(opType);
    if (iter != selectors_.cend()) {
        func = iter->second;
    }

    return func;
}

aclError OpKernelSelector::InsertAclop2KernelDesc(const AclOp &op, const std::shared_ptr<OpKernelDesc> &desc) const
{
    ACL_LOG_DEBUG("start InsertAclop2KernelDesc");
    ACL_REQUIRES_NOT_NULL(desc);
    desc->opType = op.opType;

    for (int32_t i = 0; i < op.numInputs; ++i) {
        ACL_REQUIRES_NOT_NULL(op.inputDesc[i]);
        desc->inputDescArr.emplace_back(*(op.inputDesc[i]));
    }
    ACL_LOG_DEBUG("Insert inputDescArr success!");
    for (int32_t i = 0; i < op.numOutputs; ++i) {
        ACL_REQUIRES_NOT_NULL(op.outputDesc[i]);
        desc->outputDescArr.emplace_back(*(op.outputDesc[i]));
    }
    ACL_LOG_DEBUG("Insert outputDescArr success!");

    // if aclOp.opAttr is nullptr, desc->opAttr is a empty attr object
    if (op.opAttr != nullptr) {
        for (const auto &attrVal : op.opAttr->Attrs()) {
            (void)desc->opAttr.EmplaceAttr(attrVal.first, attrVal.second);
        }
    }
    ACL_LOG_DEBUG("Insert attr success!");
    return ACL_SUCCESS;
}

aclError OpKernelSelector::SelectOpKernel(const AclOp &op)
{
    const auto func = GetSelectFunc(op.opType);
    if (func == nullptr) {
        ACL_LOG_WARN("Op not found, opType = %s", op.opType.c_str());
        return ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED;
    }

    const auto desc = std::shared_ptr<OpKernelDesc>(new (std::nothrow)OpKernelDesc);
    ACL_CHECK_MALLOC_RESULT(desc);
    ACL_REQUIRES_OK(InsertAclop2KernelDesc(op, desc));
    ACL_LOG_DEBUG("To invoke select func, opType = %s", op.opType.c_str());
    const auto ret = func(op.numInputs, op.inputDesc, op.numOutputs, op.outputDesc, op.opAttr, desc.get());
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Call][Compile]Failed to call op compile, errorCode = %d", ret);
        return ret;
    }

    ACL_LOG_DEBUG("selecting kernel succeeded. kernelId = %s", desc->kernelId.c_str());
    desc->stubFunc = OpKernelRegistry::GetInstance().GetStubFunc(op.opType, desc->kernelId);
    if (desc->stubFunc == nullptr) {
        ACL_LOG_INNER_ERROR("Stub function not registered. kernelId = %s", desc->kernelId.c_str());
        return ACL_ERROR_KERNEL_NOT_FOUND;
    }
    desc->timestamp = attr_utils::GetCurrentTimestamp();
    std::shared_ptr<OpKernelDesc> agingDesc = nullptr;
    bool isRegistered = false;
    (void)kernelDescMap_.Insert(op, desc, agingDesc, isRegistered);
    if (agingDesc != nullptr) {
        ACL_LOG_DEBUG("find aging op %s", agingDesc->opType.c_str());
    }
    return ACL_SUCCESS;
}

aclError OpKernelSelector::GetOpKernelDesc(const AclOp &op, std::shared_ptr<OpKernelDesc> &desc)
{
    return kernelDescMap_.Get(op, desc);
}

bool OpKernelSelector::HasSelectFunc(const std::string &opType) const
{
    return selectors_.count(opType) != 0U;
}

void OpKernelSelector::SetMaxOpNum(const uint64_t maxOpNum)
{
    kernelDescMap_.SetMaxOpNum(maxOpNum);
}
} // namespace acl