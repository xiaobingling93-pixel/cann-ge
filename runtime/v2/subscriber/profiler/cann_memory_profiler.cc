/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_memory_profiler.h"
#include <functional>
#include <map>
#include "kernel/memory/device/device_allocator.h"
#include "graph/def_types.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "framework/runtime/device_memory_recorder.h"
#include "core/builder/node_types.h"
#include "framework/runtime/exe_graph_executor.h"
#include "framework/runtime/model_v2_executor.h"
#include "kernel/common_kernel_impl/memory_copy.h"

namespace gert {
CannMemoryProfiler::CannMemoryProfiler(const std::shared_ptr<const SubscriberExtendInfo> &extend_info)
    : extend_info_(extend_info) {
  if ((extend_info_ != nullptr) && (extend_info_->executor != nullptr) && IsEnabled(ProfilingType::kMemory)) {
    Init();
  }
}

ge::graphStatus CannMemoryProfiler::DoProf(const Node *node, const int32_t subgraph_type) {
  if (DeviceMemoryRecorder::IsRecorderEmpty()) {
    return ge::SUCCESS;
  }
  auto memory_info_data = reinterpret_cast<MsprofMemoryInfo *>(task_memory_info_.data);
  GE_ASSERT_TRUE(subgraph_type < static_cast<int32_t>(prof_extend_info_vec_.size()));
  auto &prof_extend_info = prof_extend_info_vec_[subgraph_type];
  GE_ASSERT_TRUE(node->node_id < prof_extend_info.size());
  memory_info_data->nodeId = prof_extend_info[node->node_id].node_name_idx;
  while (!DeviceMemoryRecorder::IsRecorderEmpty()) {
    const MemoryRecorder &record_memory_info = DeviceMemoryRecorder::GetRecorder();
    task_memory_info_.timeStamp = record_memory_info.time_stamp;
    memory_info_data->size = record_memory_info.size;
    memory_info_data->addr = record_memory_info.addr;
    memory_info_data->totalAllocateMemory = record_memory_info.total_allocate_memory;
    memory_info_data->totalReserveMemory = record_memory_info.total_reserve_memory;
    GELOGI("[CannMemoryProfiler][DoProf] Report memory info: node_id: %llu, "
        "addr: %llu, size: %lld, total allocate size: %llu, "
        "total reserve size: %lld, time stamp: %llu",
        memory_info_data->nodeId, memory_info_data->addr,
        memory_info_data->size,
        memory_info_data->totalAllocateMemory,
        memory_info_data->totalReserveMemory,
        task_memory_info_.timeStamp);
    GE_ASSERT_MSPROF_OK(
        MsprofReportAdditionalInfo(true, &task_memory_info_,
                                   static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  }
  return ge::SUCCESS;
}

void CannMemoryProfiler::Init() {
  if (is_device_prof_inited_) {
    return;
  }
  for (int32_t i = static_cast<int32_t>(kInitExeGraph);
      i < static_cast<int32_t>(kSubExeGraphTypeEnd); i++) {
    const auto execution_data = static_cast<const ExecutionData *>(
        extend_info_->executor->GetExeGraphExecutor(
            static_cast<SubExeGraphType>(i))->GetExecutionData());
    if (execution_data == nullptr) {
      GELOGW("[Cann Profiling] Execution data is empty, do not init profiler.");
      return;
    }
    std::vector<ProfExtendInfo> prof_extend_info;
    GELOGI("[Cann Profiling] Init for cann memory profiling for subgraph[%ld].", i);
    (void)InitNameAndTypeWithHash(*execution_data, prof_extend_info);
    prof_extend_info_vec_.push_back(prof_extend_info);
  }
  // init data
  task_memory_info_.threadId = static_cast<uint32_t>(mmGetTid());
  task_memory_info_.type = MSPROF_REPORT_NODE_TASK_MEMORY_TYPE;
  task_memory_info_.level = MSPROF_REPORT_NODE_LEVEL;
  task_memory_info_.dataLen = static_cast<uint32_t>(sizeof(MsprofMemoryInfo));
  auto memory_info_data = reinterpret_cast<MsprofMemoryInfo *>(task_memory_info_.data);
  int32_t device_id = 0;
  (void)rtGetDevice(&device_id);
  memory_info_data->deviceId = static_cast<uint32_t>(device_id);
  memory_info_data->deviceType = 0U;

  is_device_prof_inited_ = true;
}

void CannMemoryProfiler::OnExecuteEvent(int32_t sub_exe_graph_type,
                                        CannMemoryProfiler *profiler,
                                        ExecutorEvent event, const void *node,
                                        KernelStatus result) {
  (void)result;
  if (profiler == nullptr) {
    return;
  }

  if (event == kExecuteEnd) {
    (void)profiler->DoProf(static_cast<const Node *>(node), sub_exe_graph_type);
    return;
  }

  if (event == kModelStart) {
    profiler->Init();
    return;
  }
}
}
