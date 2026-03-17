/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INC_4704E2C9273544F7B39C4485DFA055B0_H
#define INC_4704E2C9273544F7B39C4485DFA055B0_H

#include "core/execution_data.h"
#include "core/priority_queue.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/task_package.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"
#include "common/fake_node_helper.h"
#include "faker/continuous_vector_builder.h"
#include "faker/kernel_run_context_facker.h"
#include "acl/acl_rt.h"
#include "runtime/model_v2_executor.h"
#include <array>
#include <mutex>
#include <map>

namespace gert {

constexpr size_t MAX_NODE_SIZE = 100;

struct KernelSpy {
  static KernelSpy &GetInstance();
  static UINT32 KernelStub(KernelRunContext *context);
  static UINT32 KernelStubFailed(KernelRunContext *context);
  static UINT32 KernelStubEndOfSequence(KernelRunContext *context);
  static void Clear();

  static std::mutex mutex;
  static std::vector<NodeIdentity> nodes;
};

#define KERNEL_RUN_EXPECT(...)                                                         \
  ASSERT_EQ(std::vector<NodeIdentity>({__VA_ARGS__}), KernelSpy::GetInstance().nodes); \
  KernelSpy::GetInstance().Clear();

struct FakeWatcher : Watcher {
  NodeIdentity node_ids[10];
};

struct KernelAttr {
  const char *op_name;
  const char *kernel_type;
};

struct FakeExecutionData {
  FakeExecutionData(uint64_t size, bool set_priority = false) : node_size(size) {
    Reset();
    if (set_priority) {
      SetPriorityById();
    }
    executionData.base_ed.node_num = node_size;
    executionData.base_ed.nodes = kernel_ptrs.data();
    executionData.node_watchers = watchers_ptrs.data();
    executionData.start_nodes = start_nodes.data();
    executionData.node_indegrees = node_indegrees.data();
    executionData.node_indegrees_backup = node_indegrees_backup.data();
  }

  FakeExecutionData &FuncFailed(NodeIdentity id, UINT32 status) {
    (void) status;
    kernel_nodes[id].func = KernelSpy::KernelStubFailed;
    return *this;
  }

  FakeExecutionData &FuncEndOfSequence(NodeIdentity id, UINT32 status) {
    (void) status;
    kernel_nodes[id].func = KernelSpy::KernelStubEndOfSequence;
    return *this;
  }

  FakeExecutionData &KernelAttr(const std::map<NodeIdentity, KernelAttr> &kernel_attrs) {
    for (auto &iter : kernel_attrs) {
      NodeHolder fake_node = FakeNodeHelper::FakeNode(iter.second.op_name, iter.second.kernel_type, iter.first);
      context_holders[iter.first] = std::move(fake_node.context);
      kernel_nodes[iter.first].context.kernel_extend_info = std::move(fake_node.node.context.kernel_extend_info);
      kernel_nodes[iter.first].context.compute_node_info = std::move(fake_node.node.context.compute_node_info);
    }
    return *this;
  }

  FakeExecutionData &Chain(const std::vector<NodeIdentity> &node_ids) {
    if (node_ids.size() <= 1) {
      return *this;
    }
    for (size_t i = 0; i < node_ids.size() - 1; i++) {
      Edge(node_ids[i], node_ids[i + 1]);
    }
    return *this;
  }

  FakeExecutionData &Edge(NodeIdentity src, NodeIdentity dst) {
    watchers[src].node_ids[watchers[src].watch_num] = dst;
    watchers[src].watch_num++;
    node_indegrees[dst]++;
    node_indegrees_backup[dst]++;
    return *this;
  }

  FakeExecutionData &StartNodes(const std::vector<NodeIdentity> &node_ids) {
    size_t index = 0;
    for (auto id : node_ids) {
      start_nodes[index++] = &kernel_nodes[id];
    }
    executionData.start_num = node_ids.size();
    return *this;
  }

  FakeExecutionData &ExecuteStream(aclrtStream stream) {
    execute_stream_holder = ContinuousVectorBuilder::Create<aclrtStream>({stream});
    execute_stream_value.Set(execute_stream_holder.get(), nullptr);
    input_values.fill(nullptr);
    input_values[0] = reinterpret_cast<AsyncAnyValue *>(&execute_stream_value);
    executionData.base_ed.input_num = static_cast<size_t>(ExecuteArgIndex::kNum);
    executionData.base_ed.input_values = input_values.data();
    return *this;
  }

  ExecutionData *Data() {
    return &executionData;
  }

 private:
  void SetPriorityById() {
    for (size_t i = 0; i < MAX_NODE_SIZE; i++) {
      reinterpret_cast<PriorityQueueElementHead *>(kernel_ptrs[i])->priority = static_cast<long long>(i);
    }
  }

  void Reset() {
    for (size_t i = 0; i < MAX_NODE_SIZE; i++) {
      InitNode(kernel_nodes[i], i);
      watchers[i].watch_num = 0;
      node_indegrees[i] = 0;
      node_indegrees_backup[i] = 0;
      kernel_ptrs[i] = &kernel_nodes[i];
      watchers_ptrs[i] = &watchers[i];
      reinterpret_cast<PriorityQueueElementHead *>(kernel_ptrs[i])->priority = std::numeric_limits<int64_t>::max();
    }
  }

  void InitNode(::Node &node, UINT32 id) {
    node.node_id = id;
    node.context.input_size = id;  // stub for test
    node.func = KernelSpy::KernelStub;
    node.context.compute_node_info = nullptr;
    node.context.kernel_extend_info = nullptr;
  }

  ExecutionData executionData{};
  std::array<KernelRunContextHolder, MAX_NODE_SIZE> context_holders;
  std::array<::Node, MAX_NODE_SIZE> kernel_nodes;
  std::array<::Node *, MAX_NODE_SIZE> kernel_ptrs;
  std::array<::Node *, MAX_NODE_SIZE> start_nodes;
  std::array<FakeWatcher, MAX_NODE_SIZE> watchers;
  std::array<::Watcher *, MAX_NODE_SIZE> watchers_ptrs;
  std::array<int64_t, MAX_NODE_SIZE> node_indegrees;
  std::array<int64_t, MAX_NODE_SIZE> node_indegrees_backup;
  std::unique_ptr<uint8_t[]> execute_stream_holder;
  ::gert::Chain execute_stream_value{};
  std::array<AsyncAnyValue *, static_cast<size_t>(ExecuteArgIndex::kNum)> input_values{};
  size_t node_size;
};

void TaskRun(TaskPackage &package);

#define BASE(producer) (static_cast<TaskProducer &>(producer))

}  // namespace gert

#endif
