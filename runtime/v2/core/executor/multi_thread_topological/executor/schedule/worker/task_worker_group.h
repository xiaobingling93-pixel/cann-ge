/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_WORKER_GROUP_H
#define AIR_CXX_RUNTIME_V2_TASK_WORKER_GROUP_H

#include <vector>
#include "task_worker.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "runtime/subscriber/executor_subscriber_c.h"

namespace gert {
using TaskWorkerId = size_t;

class TaskWorkerGroup {
 public:
  explicit TaskWorkerGroup(ExecTaskType type) : type_(type) {}

  ~TaskWorkerGroup();

  void Add(TaskWorker &worker);

  size_t GetWorkerNum() const {
    return worker_size_;
  }

  bool Start();
  void WaitDoneAndStop(TaskPackage &completedTasks);
  void WakeupWorkers();
  void SleepWorkers();
  void SetExecuteStream(aclrtStream stream);
  void SetSubscriber(int sub_graph_type, ExecutorSubscriber *es);

  void GetAllThreadId(std::vector<uint32_t> &all_thread_id) const;

  bool IsFinish() const {
    return completed_count_ >= submitted_count_;
  }

  bool ExecuteTask(ExecTask &task, TaskWorkerId workerId);
  size_t FetchResult(TaskPackage &result);

  ExecTaskType GetType() const {
    return type_;
  }

  size_t GetSubmittedTaskCount() const {
    return submitted_count_;
  }

  size_t GetCompletedTaskCount() const {
    return completed_count_;
  }

  // only be used when worker group is stopped!
  void Dump() const;

  // could be used when worker group is running!
  void DumpTitle() const;

 private:
  void DumpWorkers() const;

 private:
  size_t submitted_count_{0U};
  size_t completed_count_{0U};
  size_t submit_failed_count_{0U};

 private:
  bool has_launched_{false};

 private:
  ExecTaskType type_;
  std::vector<TaskWorker *> workers_;
  size_t worker_size_{0U};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_WORKER_GROUP_H
