/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_worker_group.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
TaskWorkerGroup::~TaskWorkerGroup() {
  for (auto &worker : workers_) {
    delete worker;
  }
}

void TaskWorkerGroup::Add(TaskWorker &worker) {
  workers_.push_back(&worker);
  worker_size_++;
}

bool TaskWorkerGroup::Start() {
  if (has_launched_) {
    return true;
  }

  for (auto &worker : workers_) {
    if (!worker->IsRunning()) {
      if (worker->Start()) {
        has_launched_ = true;
      } else {
        GELOGE(ge::GRAPH_FAILED, "Launch task worker failed in group type %s!", ExecTaskType_ToString(type_));
      }
    }
  }
  return has_launched_;
}

void TaskWorkerGroup::WaitDoneAndStop(TaskPackage &completedTasks) {
  if (!has_launched_) {
    return;
  }

  for (auto &worker : workers_) {
    worker->WaitDone(completedTasks);
    completed_count_ += completedTasks.size();
  }
  has_launched_ = false;
}

void TaskWorkerGroup::WakeupWorkers() {
  if (!has_launched_) {
    return;
  }
  for (auto &worker : workers_) {
    worker->WakeupThreads();
  }
}

void TaskWorkerGroup::SleepWorkers() {
  if (!has_launched_) {
    return;
  }
  for (auto &worker : workers_) {
    worker->SleepThreads();
  }
}

void TaskWorkerGroup::SetExecuteStream(aclrtStream stream) {
  if (!has_launched_) {
    return;
  }
  for (auto &worker : workers_) {
    worker->SetExecuteStream(stream);
  }
}

void TaskWorkerGroup::GetAllThreadId(std::vector<uint32_t> &all_thread_id) const {
  for (const auto &worker : workers_) {
    worker->GetAllThreadId(all_thread_id);
  }
}

void TaskWorkerGroup::SetSubscriber(int sub_graph_type, ExecutorSubscriber *es) {
  for (auto &worker : workers_) {
    worker->SetSubscriber(sub_graph_type, es);
  }
}

bool TaskWorkerGroup::ExecuteTask(ExecTask &task, TaskWorkerId workerId) {
  auto &worker = workers_[workerId];
  if (!worker->Submit(task)) {
    submit_failed_count_++;
    return false;
  }
  submitted_count_++;
  return true;
}

size_t TaskWorkerGroup::FetchResult(TaskPackage &result) {
  if (IsFinish()) {
    return 0;
  }

  size_t fetchedCount = 0;
  for (auto &worker : workers_) {
    fetchedCount += worker->Fetch(result);
  }
  completed_count_ += fetchedCount;
  return fetchedCount;
}

void TaskWorkerGroup::DumpTitle() const {
  if (workers_.empty()) {
    return;
  }

  GEEVENT("|-- Worker Group [type : %s, size : %ld, running : %s]", ExecTaskType_ToString(type_), worker_size_,
          has_launched_ ? "true" : "false");

  GEEVENT("    |-- submitted count = %ld, completed count = %ld, submit task failed = %ld", submitted_count_,
          completed_count_, submit_failed_count_);
}

void TaskWorkerGroup::DumpWorkers() const {
  for (auto &worker : workers_) {
    worker->Dump();
  }
}

void TaskWorkerGroup::Dump() const {
  if (workers_.empty()) {
    return;
  }
  DumpTitle();
  DumpWorkers();
}
}  // namespace gert
