/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_SINGLE_OP_MANAGER_H_
#define GE_SINGLE_OP_SINGLE_OP_MANAGER_H_

#include <mutex>
#include <unordered_map>
#include <string>
#include "common/plugin/op_tiling_manager.h"
#include "single_op/single_op_model.h"
#include "single_op/stream_resource.h"

namespace ge {
class SingleOpManager {
 public:
  ~SingleOpManager() = default;

  static SingleOpManager &GetInstance() {
    static SingleOpManager instance;
    return instance;
  }

  Status GetOpFromModel(const std::string &model_name, const ModelData &model_data,
                        void *const stream, SingleOp **const single_op, const uint64_t model_id);

  Status GetDynamicOpFromModel(const std::string &model_name, const ModelData &model_data,
                               void *const stream, DynamicSingleOp **const single_op, const uint64_t model_id);

  Status DeleteSingleOp(const uint64_t op_id);

  Status DeleteDynamicSingleOp(const uint64_t op_id);

  StreamResource *GetResource(const uintptr_t resource_id, aclrtStream const stream);

  Status ReleaseResource(const void *const stream);

  void RegisterTilingFunc();

  Status SetAllocator(aclrtStream const stream, Allocator *const allocator);

 private:
  static Status GetResourceId(aclrtStream const stream, uintptr_t &resource_id);
  std::recursive_mutex mutex_;
  bool tiling_func_registered_ = false;
  std::unordered_map<uintptr_t, std::unique_ptr<StreamResource>> stream_resources_;
  OpTilingManager op_tiling_manager_;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_SINGLE_OP_MANAGER_H_
