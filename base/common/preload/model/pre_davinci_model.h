/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PRELOAD_PRE_DAVINCI_MODEL_H_
#define GE_COMMON_PRELOAD_PRE_DAVINCI_MODEL_H_
#include "common/preload/model/pre_model_types.h"
#include "common/preload/model/pre_model_utils.h"
#include "framework/common/util.h"
#include "ge/ge_api_types.h"
#include "proto/task.pb.h"
#include "common/model/ge_model.h"

namespace ge {
using GetOpIndexFunc = std::function<int64_t(const domi::TaskDef &)>;
using TypeToEngineNameToGetOpIndexFunc = std::map<uint32_t, std::pair<std::string, GetOpIndexFunc>>;
class PreDavinciModel {
 public:
  PreDavinciModel() = default;
  virtual ~PreDavinciModel() = default;
  void Assign(const GeModelPtr &ge_model);
  virtual Status Init();
  virtual Status DoPartitionProcess();
  Status DoTaskSink(const EngineType engine_type);
  virtual Status InitNodes(const ComputeGraphPtr &compute_graph);
  void InitKernelOffset();
  void InitRuntimeParams();
  void DoReset() const;

 private:
  Status GetEngineNameAndOpDesc(const EngineType engine_type, const domi::TaskDef &task_def,
                                std::string &engine_name, OpDescPtr &op_desc) const;
  Status GetEngineNameAndOpDescByType(const uint32_t type,
                                           const TypeToEngineNameToGetOpIndexFunc &type_to_engine_name_to_get_op_index_func,
                                           const domi::TaskDef &task_def,
                                           std::string &engine_name,
                                           OpDescPtr &op_desc) const;
  std::string PrintTaskDef(const domi::TaskDef &task_def) const;

 protected:
  // get Op
  OpDescPtr GetOpByIndex(const uint32_t op_index) const;
  GeModelPtr ge_model_;
  std::map<int64_t, OpDescPtr> op_list_;
  uint32_t model_id_{0U};
  uint32_t huge_stream_size_{0U};
  uint32_t task_num_{0U};
  PreRuntimeParam runtime_param_;
  std::unordered_map<std::string, uint32_t> names_to_bin_offset_;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_DAVINCI_MODEL_H_