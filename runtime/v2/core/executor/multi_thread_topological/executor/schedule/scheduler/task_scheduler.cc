/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_scheduler.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "common/checker.h"
#include "core/executor_error_code.h"
#include "core/utils/executor_utils.h"
#include "acl/acl_rt.h"
#include "securectype.h"
#include "runtime/subscriber/global_profiler.h"

namespace gert {
TaskScheduler::TaskScheduler(TaskProducer &producer) : task_producer_(&producer) {
  worker_group_index_.fill(ExecTaskType::NORMAL);

  for (size_t i = 0; i < static_cast<size_t>(ExecTaskType::MAX); i++) {
    worker_groups_.emplace_back(ExecTaskType(i));
  }
}

TaskScheduler::~TaskScheduler() {
  (void)StopWorkers();
}

ge::Status TaskScheduler::AddWorker(TaskWorker &worker, ExecTaskType type) {
  GE_ASSERT_TRUE(type < ExecTaskType::MAX);

  worker_groups_[static_cast<size_t>(type)].Add(worker);
  worker_group_index_[static_cast<size_t>(type)] = type;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::LaunchWorkers() {
  if (has_launched_) {
    return ge::SUCCESS;
  }

  for (auto &workerGroup : worker_groups_) {
    if (workerGroup.Start()) {
      has_launched_ = true;
    }
  }
  return has_launched_ ? ge::SUCCESS : ge::FAILED;
}

ge::Status TaskScheduler::StopWorkers() {
  for (size_t i = 0; i < static_cast<size_t>(ExecTaskType::MAX); i++) {
    TaskPackage completed_tasks;
    for (auto &worker_group : worker_groups_) {
      worker_group.WakeupWorkers();
      worker_group.WaitDoneAndStop(completed_tasks);
    }
    if (completed_tasks.size() > 0) {
      total_completed_count_ += completed_tasks.size();
      GE_ASSERT_SUCCESS(task_producer_->Recycle(completed_tasks));
    }
  }
  has_launched_ = false;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::WakeupWorkers() {
  for (auto &worker_group : worker_groups_) {
    worker_group.WakeupWorkers();
  }
  return ge::SUCCESS;
}

ge::Status TaskScheduler::SleepWorkers() {
  for (auto &worker_group : worker_groups_) {
    worker_group.SleepWorkers();
  }
  return ge::SUCCESS;
}

bool TaskScheduler::ExecuteTasks(TaskWorkerId *curr_worker_group_ids) {
  TaskPackage unprocessed_tasks = task_producer_->Produce();
  if (unprocessed_tasks.size() > 0) {
    while (auto task = unprocessed_tasks.pop_front()) {
      task->SetForceQuit(&force_quit_);
      auto exec_worker_group_id = static_cast<size_t>(worker_group_index_[static_cast<size_t>(task->GetType())]);
      TaskWorkerGroup &worker_group = worker_groups_[exec_worker_group_id];

      TaskWorkerId worker_id_max = worker_group.GetWorkerNum();
      if (worker_id_max != 0) {
        size_t execWorkerId = curr_worker_group_ids[exec_worker_group_id]++ % worker_id_max;
        if (worker_group.ExecuteTask(*task, execWorkerId)) {
          total_submitted_count_++;
        } else {
          unprocessed_tasks.push_front(*task);
        }
      }
    }
  } else {
    if (!ShouldScheduleMore()) {
      return false;
    }
  }
  return true;
}

ge::Status TaskScheduler::RecycleTasks() {
  for (TaskPackage completed_tasks; true;) {
    for (auto &worker_group : worker_groups_) {
      worker_group.FetchResult(completed_tasks);
    }
    if (completed_tasks.size() > 0) {
      total_completed_count_ += completed_tasks.size();
      return task_producer_->Recycle(completed_tasks);
    }
  }
}

ge::graphStatus TaskScheduler::Prepare(const ScheduleData &data) {
  GE_ASSERT_TRUE(data.execution_data != nullptr);
  GE_ASSERT_TRUE(data.schedule_limit > 0);

  execution_data_ = static_cast<const ExecutionData *>(data.execution_data);
  GE_ASSERT_SUCCESS(task_producer_->Prepare(data.execution_data));
  GE_ASSERT_SUCCESS(LaunchWorkers());

  schedule_limit_ = data.schedule_limit;
  return ge::GRAPH_SUCCESS;
}

aclrtStream TaskScheduler::GetExecuteMainStream() const {
  if (execution_data_ == nullptr) {
    return nullptr;
  }
  if (execution_data_->base_ed.input_num < static_cast<size_t>(ExecuteArgIndex::kNum)) {
    GELOGW("Invalid input num %zu, less than execute arg count %zu.", execution_data_->base_ed.input_num,
           static_cast<size_t>(ExecuteArgIndex::kNum));
    return nullptr;
  }
  const auto stream_idx = CalcArgIndex(execution_data_->base_ed.input_num, ExecuteArgIndex::kStream);
  if (stream_idx >= execution_data_->base_ed.input_num) {
    GELOGW("Invalid stream arg index %zu from input num %zu.", stream_idx, execution_data_->base_ed.input_num);
    return nullptr;
  }
  auto stream_chain = reinterpret_cast<Chain *>(execution_data_->base_ed.input_values[stream_idx]);
  if (stream_chain == nullptr) {
    GELOGW("Stream chain is nullptr at input index %zu.", stream_idx);
    return nullptr;
  }
  auto rt_streams = stream_chain->GetValue<ContinuousVector *>();
  if (rt_streams == nullptr) {
    GELOGW("Stream vector is nullptr at input index %zu.", stream_idx);
    return nullptr;
  }
  if (rt_streams->GetSize() == 0U) {
    GELOGW("Stream vector is empty at input index %zu.", stream_idx);
    return nullptr;
  }
  return *(reinterpret_cast<aclrtStream *>(rt_streams->MutableData()) + 0U);
}

void TaskScheduler::SetExecuteStreamForWorkers() {
  const auto stream = GetExecuteMainStream();
  int32_t stream_id = -1;
  if (stream != nullptr) {
    (void)aclrtStreamGetId(stream, &stream_id);
  }
  GELOGI("Scheduler dispatch stream %p (id=%d) to %zu worker groups.", stream, stream_id, worker_groups_.size());
  for (auto &worker_group : worker_groups_) {
    worker_group.SetExecuteStream(stream);
  }
}

void TaskScheduler::RecycleTaskWhenExecuteFailed() {
  force_quit_ = true;
  while (ShouldScheduleMore()) {
    for (size_t i = 0; i < static_cast<size_t>(ExecTaskType::MAX); i++) {
      TaskPackage completed_tasks;
      for (auto &worker_group : worker_groups_) {
        worker_group.FetchResult(completed_tasks);
      }
      total_completed_count_ += completed_tasks.size();
    }
  }
}

KernelStatus TaskScheduler::Schedule() {
  GE_ASSERT_SUCCESS(StartUp());
  SetExecuteStreamForWorkers();

  TaskWorkerId exec_worker_group_ids[static_cast<size_t>(ExecTaskType::MAX)] = {0};

  WakeupWorkers();
  while (true) {
    if (!ExecuteTasks(exec_worker_group_ids)) {
      GE_ASSERT_SUCCESS(EndUp());
      SleepWorkers();
      return kStatusSuccess;
    }
    auto ret = RecycleTasks();
    if (ret != ge::SUCCESS) {
      RecycleTaskWhenExecuteFailed();
      return ret;
    }
  }
}

KernelStatus TaskScheduler::Schedule(int sub_graph_type, ExecutorSubscriber *es) {
  GE_ASSERT_NOTNULL(es);
  GE_ASSERT_NOTNULL(es->callback);
  GE_ASSERT_SUCCESS(StartUp());
  SetExecuteStreamForWorkers();
  for (auto &workerGroup : worker_groups_) {
    workerGroup.SetSubscriber(sub_graph_type, es);
  }

  TaskWorkerId exec_worker_group_ids[static_cast<size_t>(ExecTaskType::MAX)] = {0};

  es->callback(sub_graph_type, es->arg, kModelStart, nullptr, kStatusSuccess);

  WakeupWorkers();
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kTaskTime)) {
    if (all_thread_id_.empty()) {
      GetAllThreadId(all_thread_id_);
    }
    for (auto thread_id : all_thread_id_) {
      MsprofEvent model_execute_info{};
      GlobalProfilingWrapper::GetInstance()->ReportDefaultEventForRt2MultiThread(GeProfInfoType::kModelExecute,
                                                                                 thread_id, model_execute_info);
    }
  }
  while (true) {
    if (!ExecuteTasks(exec_worker_group_ids)) {
      GE_ASSERT_SUCCESS(EndUp());
      SleepWorkers();
      es->callback(sub_graph_type, es->arg, kModelEnd, nullptr, kStatusSuccess);
      if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
        for (auto thread_id : all_thread_id_) {
          MsprofEvent model_execute_info{};
          GlobalProfilingWrapper::GetInstance()->ReportDefaultEventForRt2MultiThread(GeProfInfoType::kModelExecute,
                                                                                     thread_id, model_execute_info);
        }
      }
      return kStatusSuccess;
    }
    auto ret = RecycleTasks();
    if (ret != ge::SUCCESS) {
      RecycleTaskWhenExecuteFailed();
      return ret;
    }
  }
}

void TaskScheduler::GetAllThreadId(std::vector<uint32_t> &all_thread_id) {
  for (const auto &worker_group : worker_groups_) {
    worker_group.GetAllThreadId(all_thread_id);
  }
}

ge::Status TaskScheduler::StartUp() {
  GE_ASSERT_TRUE(has_launched_);
  GE_ASSERT_TRUE(schedule_limit_ != 0);
  GE_ASSERT_SUCCESS(task_producer_->StartUp());
  total_completed_count_ = 0U;
  total_submitted_count_ = 0U;
  force_quit_ = false;
  return ge::SUCCESS;
}

ge::Status TaskScheduler::EndUp() {
  GE_ASSERT_SUCCESS(task_producer_->EndUp());
  return ge::SUCCESS;
}

void TaskScheduler::DumpScheduler() const {
  GEEVENT("|-- Task Scheduler [%s]", has_launched_ ? "running" : "stopped");
  GEEVENT("    |-- scheduled count = %ld, completed count = %ld", total_submitted_count_, total_completed_count_);
}

void TaskScheduler::DumpProducer() const {
  task_producer_->Dump();
}

void TaskScheduler::DumpWorkersBrief() const {
  for (auto &worker_group : worker_groups_) {
    worker_group.DumpTitle();
  }
}

void TaskScheduler::DumpWorkersDetail() const {
  for (auto &worker_group : worker_groups_) {
    worker_group.Dump();
  }
}

void TaskScheduler::DumpBrief() const {
  DumpScheduler();
  DumpProducer();
  DumpWorkersBrief();
}

void TaskScheduler::Dump() const {
  DumpScheduler();
  DumpProducer();
  DumpWorkersDetail();
}
}  // namespace gert