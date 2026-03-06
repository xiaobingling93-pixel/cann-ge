/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_LABEL_SWITCH_BY_INDEX_TASK_INFO_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_LABEL_SWITCH_BY_INDEX_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"

namespace ge {
class LabelSwitchByIndexTaskInfo : public TaskInfo {
 public:
  LabelSwitchByIndexTaskInfo() = default;

  ~LabelSwitchByIndexTaskInfo() override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  Status InitIndexValue(const DavinciModel &davinci_model, const IowAddrs &iow_addrs);

  void *index_value_{nullptr};    // switch index input.
  uint32_t branch_max_{0U};       // max branch count.
  void *args_{nullptr};           // label info memory.
  uint32_t args_size_{0U};        // label info length.
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_LABEL_SWITCH_BY_INDEX_TASK_INFO_H_
