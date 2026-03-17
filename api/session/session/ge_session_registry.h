/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_SESSION_GE_SESSION_REGISTRY_H_
#define GE_SESSION_GE_SESSION_REGISTRY_H_

#include <map>
#include <mutex>
#include <functional>

namespace ge {

class GeSessionRegistry {
 public:
  using FinalizeFunc = std::function<void()>;
  using ImplPtr = void*;

  static GeSessionRegistry& Instance() {
    static GeSessionRegistry instance;
    return instance;
  }

  // 注册 GeSession::Impl 及其 Finalize 函数
  void Register(ImplPtr impl, FinalizeFunc finalize_func) {
    if (impl == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    impls_.insert({impl, finalize_func});
  }

  // 注销 GeSession::Impl（析构时调用）
  void Unregister(ImplPtr impl) {
    if (impl == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    impls_.erase(impl);
  }

  // GEFinalizeV2 中调用：遍历所有 Impl，执行 Finalize 并清理
  void FinalizeAllSessions() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : impls_) {
      if (pair.second) {
        pair.second();
      }
    }
    impls_.clear();
  }

 private:
  GeSessionRegistry() = default;
  ~GeSessionRegistry() = default;
  GeSessionRegistry(const GeSessionRegistry&) = delete;
  GeSessionRegistry& operator=(const GeSessionRegistry&) = delete;

  std::map<ImplPtr, FinalizeFunc> impls_;
  std::mutex mutex_;
};

}  // namespace ge

#endif  // GE_SESSION_GE_SESSION_REGISTRY_H_
