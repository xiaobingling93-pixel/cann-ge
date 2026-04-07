/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_DEPENDS_MMPA_SRC_MMAP_STUB_H_
#define AIR_TEST_ENGINES_NNENG_DEPENDS_MMPA_SRC_MMAP_STUB_H_

#include <cstdint>
#include <memory>
#include "mmpa/mmpa_api.h"

namespace fe {
std::string GetCurpath();

class MmpaStubApi {
 public:
  virtual ~MmpaStubApi() = default;

  virtual void *DlOpen(const char *file_name, int32_t mode) {
    return dlopen(file_name, mode);
  }

  virtual void *DlSym(void *handle, const char *func_name) {
    return dlsym(handle, func_name);
  }

  virtual INT32 GetEnv(const CHAR *name, CHAR *value, UINT32 len)
  {
    const char *env = getenv(name);
    if (env == nullptr) {
      return EN_ERROR;
    }

    strncpy(value, env, len);
    return EN_OK;
  }

  virtual int32_t DlClose(void *handle) {
    return dlclose(handle);
  }

  virtual int32_t DlAddr(VOID *addr, mmDlInfo *info) {
    return dladdr(addr, (Dl_info*)info);
  }

  virtual int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) {
    INT32 ret = EN_OK;
    char *ptr = realpath(path, realPath);
    if (ptr == nullptr) {
      ret = EN_ERROR;
    }
    return ret;
  };

  virtual INT32 WaitPid(mmProcess pid, INT32 *status, INT32 options) {
    if ((options != MMPA_ZERO) && (options != M_WAIT_NOHANG) && (options != M_WAIT_UNTRACED)) {
    return EN_INVALID_PARAM;
    }

    INT32 ret = waitpid(pid, status, options);
    if (ret == EN_ERROR) {
      return EN_ERROR;
    }
    if ((ret > MMPA_ZERO) && (ret == pid)) {
      if (status != NULL) {
        if (WIFEXITED(*status)) {
          *status = WEXITSTATUS(*status);
        }
        if(WIFSIGNALED(*status)) {
          *status = WTERMSIG(*status);
        }
      }
      return EN_ERR;
    }
    return EN_OK;
  }
};

class MmpaStub {
 public:
  static MmpaStub& GetInstance() {
    static MmpaStub instance;
    return instance;
  }

  void SetImpl(const std::shared_ptr<MmpaStubApi> &impl) {
    impl_ = impl;
  }

  std::weak_ptr<MmpaStubApi> GetImplWeakPtr() {
    return impl_;
  }

  MmpaStubApi* GetImpl() {
    return impl_.get();
  }

  void Reset() {
    impl_ = std::make_shared<MmpaStubApi>();
  }

 private:
  MmpaStub(): impl_(std::make_shared<MmpaStubApi>()) {
  }

  std::shared_ptr<MmpaStubApi> impl_;
};

}  // namespace fe

#endif  // AIR_TEST_ENGINES_NNENG_DEPENDS_MMPA_SRC_MMAP_STUB_H_
