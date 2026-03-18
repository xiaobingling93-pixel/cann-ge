/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_SCHEDULE_H
#define AIR_CXX_RUNTIME_V2_TASK_SCHEDULE_H

#include <memory>
#include <vector>
#include <array>
#include "task_schedule_data.h"
#include "core/executor/multi_thread_topological/executor/schedule/worker/task_worker_group.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "ge/ge_api_types.h"
#include "graph/ge_error_codes.h"
#include "acl/acl_rt.h"

namespace gert {
class TaskScheduler {
 public:
  using ScheduleData = TaskScheduleData;

 public:
  explicit TaskScheduler(TaskProducer &producer);
  ~TaskScheduler();

  ge::Status AddWorker(TaskWorker &worker, ExecTaskType type);
  ge::Status LaunchWorkers();
  ge::Status WakeupWorkers();
  ge::Status StopWorkers();
  ge::Status SleepWorkers();

  ge::graphStatus Prepare(const ScheduleData &data);
  KernelStatus Schedule();
  KernelStatus Schedule(int sub_graph_type, ExecutorSubscriber *es);

  // only be used when scheduler is stopped!
  void Dump() const;

  // could be used when scheduler is running!
  void DumpBrief() const;

 public:
  size_t GetScheduledTaskCount() const {
    return total_submitted_count_;
  }

  size_t GetCompletedTaskCount() const {
    return total_completed_count_;
  }

  bool ShouldScheduleMore() const {
    return total_completed_count_ < total_submitted_count_;
  }

 private:
  ge::Status StartUp();
  ge::Status EndUp();
  void RecycleTaskWhenExecuteFailed();
  void SetExecuteStreamForWorkers();
  aclrtStream GetExecuteMainStream() const;
  bool ExecuteTasks(TaskWorkerId *curr_worker_group_ids);
  ge::Status RecycleTasks();
  void GetAllThreadId(std::vector<uint32_t> &all_thread_id);

 private:
  void DumpScheduler() const;
  void DumpProducer() const;
  void DumpWorkersBrief() const;
  void DumpWorkersDetail() const;

 private:
  size_t total_submitted_count_{0U};
  size_t total_completed_count_{0U};

 private:
  bool has_launched_{false};
  size_t schedule_limit_{0U};
  bool force_quit_{false};
  const ExecutionData *execution_data_{nullptr};

 private:
  std::unique_ptr<TaskProducer> task_producer_;
  std::vector<TaskWorkerGroup> worker_groups_;
  std::array<ExecTaskType, static_cast<size_t>(ExecTaskType::MAX)> worker_group_index_;
  std::vector<uint32_t> all_thread_id_;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_SCHEDULE_H