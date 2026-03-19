/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/log_inner.h"

namespace acl {
namespace file_utils {
class MmDirEntGuard {
public:
    MmDirEntGuard(mmDirent2 **const dirEnts, const int32_t valCount) : dirEntries_(dirEnts), count_(valCount)
    {
    }

    ~MmDirEntGuard()
    {
        mmScandirFree2(dirEntries_, count_);
    }

private:
    mmDirent2 **dirEntries_;
    int32_t count_;
};

static int32_t RegularFileFilterFn(const mmDirent2 *const entry)
{
    // some xfs file system may return DT_UNKNOWN
    const bool isFile = (static_cast<int32_t>(entry->d_type) == MM_DT_REG) ||
        (static_cast<int32_t>(entry->d_type) == DT_UNKNOWN);
    const bool ret = (static_cast<int32_t>(entry->d_type) == MM_DT_DIR) || isFile;
    return static_cast<int32_t>(ret);
}

aclError ListFiles(const std::string &dirName, FileNameFilterFn filter, std::vector<std::string> &names,
    const int32_t maxDepth)
{
    if (maxDepth <= 0) {
        return ACL_SUCCESS;
    }

    mmDirent2 **dirEntries = nullptr;
    const auto ret = mmScandir2(dirName.c_str(), &dirEntries, &RegularFileFilterFn, nullptr);
    if (ret < 0) {
        ACL_LOG_INNER_ERROR("[Scan][Dir]scan dir failed. path = %s, ret = %d", dirName.c_str(), ret);
        return ACL_ERROR_READ_MODEL_FAILURE;
    }
    const std::string tmpStr("/");
    const MmDirEntGuard guard(dirEntries, ret);
    for (int32_t i = 0; i < ret; ++i) {
        mmDirent2 *const dirEnt = dirEntries[i];
        ACL_REQUIRES_NOT_NULL(dirEnt);
        // some xfs file system may return DT_UNKNOWN
        const bool isFile = (static_cast<int32_t>(dirEnt->d_type) == MM_DT_REG) ||
            (static_cast<int32_t>(dirEnt->d_type) == DT_UNKNOWN);
        const std::string name = std::string(dirEnt->d_name);
        const std::string fullName = dirName + tmpStr + name;
        if ((static_cast<int32_t>(dirEnt->d_type) == MM_DT_DIR) && (name != ".") && (name != "..")) {
            ACL_REQUIRES_OK(ListFiles(fullName, filter, names, maxDepth - 1));
        } else if (isFile) {
            if ((filter == nullptr) || filter(name)) {
                names.emplace_back(fullName);
            }
        } else {
            continue;
        }
    }

    return ACL_SUCCESS;
}
} // namespace file_utils
} // namespace acl

