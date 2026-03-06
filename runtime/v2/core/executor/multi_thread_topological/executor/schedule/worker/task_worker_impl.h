/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_WORKER_IMPL_H
#define AIR_CXX_RUNTIME_V2_TASK_WORKER_IMPL_H

#include <thread>
#include <vector>
#include <condition_variable>
#include <algorithm>
#include "task_worker.h"
#include "task_thread.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_worker_config.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "runtime/context.h"
#include "runtime/rt.h"
#include "checker.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"
#include "graph/ge_local_context.h"
#include "mmpa/mmpa_api.h"

namespace gert {
template <typename QUEUE>
class TaskWorkerImpl : public TaskWorker {
 public:
  explicit TaskWorkerImpl(const TaskWorkerConfig &cfg)
      : name_(std::string(ExecTaskType_ToString(cfg.bind_task_type)) + "_" + cfg.name),
        is_running_(false),
        thread_mode_(cfg.thread_mode),
        threads_(cfg.thread_count, nullptr),
        pending_task_queue_(cfg.pending_queue_size_log2),
        completed_task_queue_(cfg.completed_queue_size_log2) {}

  ~TaskWorkerImpl() override{
    Stop();
    DestroyThreads();
  }

 private:
  bool Start() override {
    if (is_running_.load(std::memory_order_relaxed)) {
      return true;
    }

    is_running_.store(true, std::memory_order_release);

    return StartThreads();
  }

  void Stop() override {
    if (!is_running_.load(std::memory_order_relaxed)) {
      return;
    }

    is_running_.store(false, std::memory_order_release);

    StopThreads();
  }

  bool IsRunning() const override {
    return is_running_.load(std::memory_order_relaxed);
  }

  bool Submit(ExecTask &task) override {
    if (!pending_task_queue_.Push(&task)) {
      submit_failed_count_++;
      GELOGD("Pending task queue is full, queue size: %u", GetPendingSize());
      return false;
    }
    task_submitted_count_++;
    return true;
  }

  void GetAllThreadId(std::vector<uint32_t> &all_thread_id) override {
    std::unique_lock<std::mutex> lk(all_thread_id_mtx_);
    all_thread_id.insert(all_thread_id.end(), all_thread_id_.begin(), all_thread_id_.end());
  }

  size_t Submit(TaskPackage &task_package) override {
    size_t submitted_count = 0;
    while (auto task = task_package.pop_front()) {
      if (!pending_task_queue_.Push(task)) {
        task_package.push_front(*task);
        submit_failed_count_++;
        GELOGD("Pending task queue is full, queue size: %u", GetPendingSize());
        break;
      }
      submitted_count++;
    }
    task_submitted_count_ += submitted_count;
    return submitted_count;
  }

  size_t Fetch(TaskPackage &task_package) override {
    size_t fetched_count = 0;
    ExecTask *task = nullptr;
    while (completed_task_queue_.Pop(task)) {
      if (!task) {
        break;
      }
      task_package.push_back(*task);
      fetched_count++;
    }
    task_fetched_count_ += fetched_count;
    return fetched_count;
  }

  void WaitDone(TaskPackage &task_package) override {
    while (!pending_task_queue_.IsEmpty()) {
      Fetch(task_package);
    }
    Stop();                         // Stop first for saving the last running task!
    Fetch(task_package);             // Fetching remain completed tasks!
    FetchLeakedTasks(task_package);  // Fetching the task which put in completed queue failed!
  }

  void WakeupThreads() override {
    Notify();
  }

  void SleepThreads() override {
    is_sleep_.store(true, std::memory_order_release);
  }

  void SetSubscriber(int sub_graph_type, ExecutorSubscriber *es) override {
    sub_graph_type_ = sub_graph_type;
    es_ = es;
  }

  size_t GetPendingSize() const override {
    return pending_task_queue_.GetSize();
  }

  size_t GetCompletedSize() const override {
    return completed_task_queue_.GetSize();
  }

  void Dump() const override {
    GEEVENT("    |-- Worker [%s : running(%s)]", name_.c_str(),
            is_running_.load(std::memory_order_relaxed) ? "true" : "false");

    GEEVENT("        |-- pendding queue size = %ld, completed queue size = %ld", GetPendingSize(), GetCompletedSize());

    GEEVENT("        |-- submitted = %ld, fetched = %ld, submit failed = %ld", task_submitted_count_,
            task_fetched_count_, submit_failed_count_);

    DumpThreads();
  }

 private:
  void FetchLeakedTasks(TaskPackage &task_package) {
    for (auto &thread : threads_) {
      auto task = thread->FetchLeakedTask();
      if (task != nullptr) {
        task_package.push_back(*task);
      }
    }
  }

 private:
  void GetCurrentCtx(rtContext_t &ctx) const {
    auto ret = rtCtxGetCurrent(&ctx);
    if ((ret == RT_ERROR_NONE) && (ctx != nullptr)) {
      return;
    }
    GELOGW("Failed to get current context, ret %d", ret);
  }

  void SaveCurrentThreadId() {
    std::unique_lock<std::mutex> lk(all_thread_id_mtx_);
    all_thread_id_.template emplace_back(mmGetTid());
  }

  bool StartThreads() {
    size_t index = 0;
    rtContext_t ctx = nullptr;
    GetCurrentCtx(ctx);
    GE_ASSERT_NOTNULL(ctx);
    const error_message::ErrorManagerContext &error_context = error_message::GetErrMgrContext();
    const auto ge_context = ge::GetThreadLocalContext();
    for (auto &thread : threads_) {
      std::string threadName = name_ + std::string("_t") + std::to_string(index++);
      thread = new (std::nothrow) TaskThread(threadName, thread_mode_);
      if (!thread) {
        break;
      }
      if (!thread->Start([this, &thread, ctx, ge_context, error_context]() {
            auto rtErr = rtCtxSetCurrent(ctx);
            if (rtErr != RT_ERROR_NONE) {
              GELOGW("Failed to set current context, ret %d", rtErr);
              REPORT_INNER_ERR_MSG("E19999", "Set context failed, ret %d", rtErr);
            }
            ge::GetThreadLocalContext() = ge_context;
            error_message::SetErrMgrContext(error_context);
            SaveCurrentThreadId();
            Wait();
            ExecuteTasks(*thread);
          })) {
        break;
      }
    }
    if (index < threads_.size()) {
      Stop();
      return false;
    }
    return true;
  }

  void StopThreads() {
    for (auto &thread : threads_) {
      if (thread) {
        thread->Stop();
      }
    }
  }

  void DumpThreads() const {
    for (auto &thread : threads_) {
      if (thread) {
        thread->Dump();
      }
    }
  }

  void DestroyThreads() {
    for (auto &thread : threads_) {
      delete thread;
      thread = nullptr;
    }
  }

  void ExecuteTasks(TaskThread &thread) {
    while (is_running_.load(std::memory_order_acquire)) {
      ExecTask *task = nullptr;
      Wait();
      if (!pending_task_queue_.Pop(task)) {
        thread.OnTaskPopFailed();
        thread.Await();
        continue;
      }
      if (!task) {
        thread.OnTaskExecFailed();
        continue;
      } else {
        if (es_ != nullptr) {
          task->Execute(sub_graph_type_, es_);
        } else {
          task->Execute();
        }
        thread.OnTaskExecuted();
      }
      while (!completed_task_queue_.Push(task)) {
        GELOGD("Completed task queue is full, queue size: %u", GetCompletedSize());
        thread.OnTaskPushFailed();
        thread.Await();
        if (!is_running_.load(std::memory_order_acquire)) {
          // Stop task worker when pushing completed task failed,
          // saving this task to thread context.
          // Should fetch this task through WaitDone;
          thread.OnTaskLeaked(task);
          break;
        }
      }
    }
  }

  void Wait() {
    if (is_sleep_.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lk(lk_);
      cv_.wait(lk, [this] { return !is_sleep_.load(std::memory_order_relaxed); });
    }
  }

  void Notify() {
    if (is_sleep_.load(std::memory_order_relaxed)) {
      is_sleep_.store(false, std::memory_order_release);
      std::unique_lock<std::mutex> lk(lk_);
      cv_.notify_all();
    }
  }

 private:
  size_t task_submitted_count_{0U};
  size_t task_fetched_count_{0U};
  size_t submit_failed_count_{0U};

 private:
  std::string name_;
  std::atomic<bool> is_running_;

  TaskThreadMode thread_mode_;
  std::vector<TaskThread *> threads_;

 private:
  QUEUE pending_task_queue_;
  QUEUE completed_task_queue_;
  std::mutex lk_;
  std::condition_variable cv_;
  std::atomic<bool> is_sleep_{true};
  int sub_graph_type_{1};
  ExecutorSubscriber *es_{nullptr};
  std::mutex all_thread_id_mtx_;
  std::vector<uint32_t> all_thread_id_;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_WORKER_IMPL_H