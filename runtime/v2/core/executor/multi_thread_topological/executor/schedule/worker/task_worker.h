/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_WORKER_H
#define AIR_CXX_RUNTIME_V2_TASK_WORKER_H

#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "runtime/subscriber/executor_subscriber_c.h"
#include "runtime/exe_graph_executor.h"
#include "acl/acl_rt.h"

namespace gert {
class TaskWorker {
 public:
  virtual bool Start() = 0;
  virtual void Stop() = 0;
  virtual bool IsRunning() const = 0;

  virtual bool Submit(ExecTask &task) = 0;
  virtual size_t Submit(TaskPackage &task_package) = 0;
  virtual size_t Fetch(TaskPackage &task_package) = 0;
  virtual void WaitDone(TaskPackage &task_package) = 0;
  virtual void WakeupThreads() = 0;
  virtual void SleepThreads() = 0;
  virtual void SetExecuteStream(aclrtStream stream) = 0;
  virtual void SetSubscriber(int sub_graph_type, ExecutorSubscriber *es) = 0;
  virtual void GetAllThreadId(std::vector<uint32_t> &all_thread_id) = 0;

  virtual size_t GetPendingSize() const = 0;
  virtual size_t GetCompletedSize() const = 0;
  virtual void Dump() const = 0;

  virtual ~TaskWorker() = default;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_WORKER_H
