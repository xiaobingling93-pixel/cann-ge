/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_EAGER_TASK_INFO_H
#define CANN_GRAPH_ENGINE_EAGER_TASK_INFO_H

#include "graph/args_format_desc.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "framework/omg/parser/parser_types.h"
#include "framework/common/types.h"
#include "graph/load/model_manager/sink_only_allocator.h"
#include "register/op_tiling_registry.h"

namespace ge {
class CustomTaskInfo : public TaskInfo {
 public:
  CustomTaskInfo() = default;
  ~CustomTaskInfo() override {
    davinci_model_ = nullptr;
  }

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args = {},
              const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status Release() override;

  uint32_t GetTaskID() const override {
    return task_id_;
  }

  uint32_t GetStreamId() const override {
    return stream_id_;
  }

  void PostProcess(const domi::TaskDef &task_def) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  void UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs);

  Status ConstructCustomKernelContextInputsOutputs(const ge::OpDescPtr &op_desc,
                                                   std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                   std::vector<std::unique_ptr<uint8_t[]>> &outputs) const;
  DavinciModel *davinci_model_{};
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  OpDescPtr op_desc_;

  ArgsPlacement args_placement_{ArgsPlacement::kArgsPlacementHbm};
  std::vector<uint64_t> input_data_addrs_;
  std::vector<uint64_t> output_data_addrs_;
  std::vector<uint64_t> workspace_addrs_;
  std::vector<uint64_t> input_mem_types_;
  std::vector<uint64_t> output_mem_types_;
  std::vector<uint64_t> workspace_mem_types_;
  gert::KernelContextHolder eager_context_holder_{};
  std::shared_ptr<gert::memory::SinkOnlyAllocator> sink_only_allocator_;
};
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_EAGER_TASK_INFO_H
