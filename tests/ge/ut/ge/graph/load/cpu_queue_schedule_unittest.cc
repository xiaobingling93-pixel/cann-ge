/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/cpu_queue_schedule.h"
#include "graph/def_types.h"
#include "runtime_stub.h"
#include "stub/gert_runtime_stub.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
extern std::string g_runtime_stub_mock;

namespace ge {
class UtestCpuQueueSchedule : public testing::Test {
 protected:
  void SetUp() {
    g_runtime_stub_mock = "";
  }

  void TearDown() {
    g_runtime_stub_mock = "";
  }
};

// test Init_CpuTaskZeroCopy_succ
TEST_F(UtestCpuQueueSchedule, CpuTaskZeroCopy_Init) {
  CpuTaskZeroCopy cpu_task_zero_copy(nullptr);
  std::vector<uintptr_t> mbuf_list;
  map<uint32_t, ZeroCopyOffset> outside_addrs;
  ZeroCopyOffset addr_mapping;
  addr_mapping.addr_count_ = 1;
  std::vector<uintptr_t> addr_offset{0x11110000U};
  uintptr_t addr = 0x12340000U;
  std::map<uintptr_t, std::vector<uintptr_t>> outside_addr{{addr, addr_offset}};
  addr_mapping.outside_addrs_.emplace_back(outside_addr);
  mbuf_list.emplace_back(addr);
  uint32_t index = 0;
  outside_addrs[index] = addr_mapping;
  std::vector<bool> no_tiling_list = {false};
  ZeroCpyArgs args = {.cpy_type = ZeroCpyType::kAllStatic, .has_tensor_desc = false, .need_distribute = true};
  args.fusion_offsets.resize(mbuf_list.size());
  EXPECT_EQ(cpu_task_zero_copy.Init(mbuf_list, outside_addrs, no_tiling_list, args), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskZeroCopy_InitAddrs_fail) {
  std::vector<uintptr_t> mbuf_list;
  std::map<uint32_t, ZeroCopyOffset> outside_addrs;
  std::vector<bool> is_no_tiling_list;
  ZeroCpyArgs cpy_args;

  ZeroCopyOffset member;
  outside_addrs[0] = member;

  CpuTaskZeroCopy cpu_task_zero_copy(nullptr);
  EXPECT_EQ(cpu_task_zero_copy.InitAddrs(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args), PARAM_INVALID);

  std::map<uintptr_t, std::vector<uintptr_t>> m;
  m[0] = std::vector<uintptr_t>({0});
  outside_addrs[0].outside_addrs_.push_back(m);
  cpy_args.cpy_type = ZeroCpyType::kAllDynamic;
  is_no_tiling_list.push_back(true);

  //EXPECT_EQ(cpu_task_zero_copy.InitAddrs(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args), SUCCESS); //??
}

TEST_F(UtestCpuQueueSchedule, CpuTaskInfo_Init_args_valid) {
  CpuTaskZeroCopy cpu_task_zero_copy(nullptr);
  CpuTaskActiveEntry cpu_task_active_entry(nullptr);
  CpuTaskModelDequeue cpu_task_model_dequeue(nullptr);
  CpuTaskModelRepeat cpu_task_model_repeat(nullptr);
  CpuTaskWaitEndGraph cpu_task_wait_end_graph(nullptr);
  CpuTaskModelEnqueue cpu_task_model_enqueue(nullptr);
  CpuTaskProcessOutput cpu_task_post_dynamic_output(nullptr, ProcessStage::kPostDynamic);
  CpuTaskProcessOutput cpu_task_post_static_output(nullptr, ProcessStage::kPostStatic);
  CpuTaskProcessOutput cpu_task_prepare_output(nullptr, ProcessStage::kPrepare);
  CpuTaskMarkStep cpu_task_mark_step(nullptr);
  CpuTaskProcessInputsMemCopy cpu_task_process_input_mem_cp(nullptr);
  EXPECT_EQ(cpu_task_zero_copy.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_active_entry.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_dequeue.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_repeat.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_wait_end_graph.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_enqueue.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_prepare_output.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_post_dynamic_output.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_post_static_output.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_mark_step.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_process_input_mem_cp.Distribute(), FAILED);

  rtStream_t stream = (rtStream_t)99;
  cpu_task_zero_copy.stream_ = stream;
  cpu_task_active_entry.stream_ = stream;
  cpu_task_model_dequeue.stream_ = stream;
  cpu_task_model_repeat.stream_ = stream;
  cpu_task_wait_end_graph.stream_ = stream;
  cpu_task_model_enqueue.stream_ = stream;
  cpu_task_prepare_output.stream_ = stream;
  cpu_task_post_dynamic_output.stream_ = stream;
  cpu_task_post_static_output.stream_ = stream;
  cpu_task_mark_step.stream_ = stream;

  cpu_task_zero_copy.args_ = ValueToPtr(99);
  cpu_task_zero_copy.args_size_ = 1;

  cpu_task_active_entry.args_ = ValueToPtr(99);
  cpu_task_active_entry.args_size_ = 1;

  cpu_task_model_dequeue.args_ = ValueToPtr(99);
  cpu_task_model_dequeue.args_size_ = 1;

  cpu_task_model_repeat.args_ = ValueToPtr(99);
  cpu_task_model_repeat.args_size_ = 1;

  cpu_task_wait_end_graph.args_ = ValueToPtr(99);
  cpu_task_wait_end_graph.args_size_ = 1;

  cpu_task_model_enqueue.args_ = ValueToPtr(99);
  cpu_task_model_enqueue.args_size_ = 1;

  cpu_task_prepare_output.args_ = ValueToPtr(99);
  cpu_task_prepare_output.args_size_ = 1;

  cpu_task_post_dynamic_output.args_ = ValueToPtr(99);
  cpu_task_post_dynamic_output.args_size_ = 1;

  cpu_task_post_static_output.args_ = ValueToPtr(99);
  cpu_task_post_static_output.args_size_ = 1;

  cpu_task_mark_step.args_ = ValueToPtr(99);
  cpu_task_mark_step.args_size_ = 1;

  cpu_task_process_input_mem_cp.args_ = ValueToPtr(99);
  cpu_task_process_input_mem_cp.args_size_ = 1;

  g_runtime_stub_mock = "rtCpuKernelLaunchWithFlag";
  EXPECT_EQ(cpu_task_zero_copy.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_dequeue.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_repeat.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_wait_end_graph.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_model_enqueue.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_prepare_output.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_post_dynamic_output.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_post_static_output.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_mark_step.Distribute(), FAILED);
  EXPECT_EQ(cpu_task_process_input_mem_cp.Distribute(), FAILED);

  g_runtime_stub_mock = "rtStreamActive";
  EXPECT_EQ(cpu_task_active_entry.Distribute(), FAILED);

  cpu_task_zero_copy.args_ = nullptr;
  cpu_task_active_entry.args_ = nullptr;
  cpu_task_model_dequeue.args_ = nullptr;
  cpu_task_model_repeat.args_ = nullptr;
  cpu_task_wait_end_graph.args_ = nullptr;
  cpu_task_model_enqueue.args_ = nullptr;
  cpu_task_prepare_output.args_ = nullptr;
  cpu_task_post_dynamic_output.args_ = nullptr;
  cpu_task_post_static_output.args_ = nullptr;
  cpu_task_mark_step.args_ = nullptr;
  cpu_task_process_input_mem_cp.args_ = nullptr;
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelDequeue_Init_failed) {
  class MockAclRuntime : public ge::AclRuntimeStub {
   public:
    aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) override {
      return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) override {
      return ACL_ERROR_RT_INTERNAL_ERROR;
    }
  };
  auto mock_acl_runtime = std::make_shared<MockAclRuntime>();
  rtStream_t stream = nullptr;
  CpuTaskModelDequeue cpu_task_model_dequeue(stream);
  uint32_t queue_id = 0U;
  uintptr_t in_mbuf = 0U;

  domi::TaskDef task_def;
  DavinciModel *davinci_model = nullptr;
  EXPECT_EQ(cpu_task_model_dequeue.Init(task_def, davinci_model), SUCCESS);

  cpu_task_model_dequeue.args_size_ = 1;
  auto ret = cpu_task_model_dequeue.Init(queue_id, in_mbuf);
  EXPECT_EQ(ret, FAILED);

  ge::AclRuntimeStub::SetInstance(mock_acl_runtime);

  cpu_task_model_dequeue.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  g_runtime_stub_mock = "aclrtMalloc";
  EXPECT_NE(cpu_task_model_dequeue.Init(queue_id, in_mbuf), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  g_runtime_stub_mock = "aclrtMemcpy";
  EXPECT_NE(cpu_task_model_dequeue.Init(queue_id, in_mbuf), SUCCESS);
  ge::AclRuntimeStub::Reset();
}

TEST_F(UtestCpuQueueSchedule, CpuTaskZeroCopy_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskZeroCopy cpu_task_zero_copy(stream);
  std::vector<uintptr_t> mbuf_list;
  std::map<uint32_t, ZeroCopyOffset> outside_addrs;
  std::vector<bool> is_no_tiling_list;
  ZeroCpyArgs cpy_args;

  EXPECT_EQ(cpu_task_zero_copy.Init(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args), SUCCESS);

  cpu_task_zero_copy.args_size_ = 1;
  auto ret = cpu_task_zero_copy.Init(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args);
  EXPECT_EQ(ret, FAILED);

  cpu_task_zero_copy.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_EQ(cpu_task_zero_copy.Init(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args), SUCCESS);  //??

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_EQ(cpu_task_zero_copy.Init(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args), SUCCESS);  //??
}

TEST_F(UtestCpuQueueSchedule, CpuTaskProcessOutput_Init_failed) {
  rtStream_t stream = nullptr;
  ProcessStage stage = ProcessStage::kPrepare;
  CpuTaskProcessOutput cpu_task_process_output(stream, stage);
  uintptr_t addr = 0U;
  uint32_t size = 0U;
  uintptr_t in_mbuf = 0U;
  uintptr_t out_mbuf = 0U;

  EXPECT_EQ(cpu_task_process_output.Init(addr, size, in_mbuf, out_mbuf), SUCCESS);

  cpu_task_process_output.args_size_ = 1;
  auto ret = cpu_task_process_output.Init(addr, size, in_mbuf, out_mbuf);
  EXPECT_EQ(ret, FAILED);

  cpu_task_process_output.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_NE(cpu_task_process_output.Init(addr, size, in_mbuf, out_mbuf), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_NE(cpu_task_process_output.Init(addr, size, in_mbuf, out_mbuf), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskProcessOutput_Init_success) {
  rtStream_t stream = nullptr;
  ProcessStage stage = ProcessStage::kPrepare;
  CpuTaskProcessOutput cpu_task_process_output(stream, stage);
  uintptr_t addr = 0U;
  uint32_t size = 0U;
  uintptr_t in_mbuf = 0U;
  uintptr_t out_mbuf = 0U;
  InputOutputDescInfo tensor_desc;
  tensor_desc.data_type = 0;
  ShapeDescription shape;
  shape.dims = {1, 2};
  tensor_desc.shape_info = shape;
  EXPECT_EQ(cpu_task_process_output.Init(addr, size, in_mbuf, out_mbuf, &tensor_desc), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelEnqueue_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskModelEnqueue cpu_task_model_enqueue(stream);
  uint32_t queue_id = 0;
  uintptr_t out_mbuf = 0U;

  EXPECT_EQ(cpu_task_model_enqueue.Init(queue_id, out_mbuf), SUCCESS);

  cpu_task_model_enqueue.args_size_ = 1;
  auto ret = cpu_task_model_enqueue.Init(queue_id, out_mbuf);
  EXPECT_EQ(ret, FAILED);

  cpu_task_model_enqueue.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_NE(cpu_task_model_enqueue.Init(queue_id, out_mbuf), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_NE(cpu_task_model_enqueue.Init(queue_id, out_mbuf), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskMarkStep_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskMarkStep cpu_task_mark_step(stream);
  GroupInfo group_info;
  group_info.group_total_count = 0U;
  group_info.group_index = 0U;
  group_info.group_policy = 0U;
  std::string dump_step = "0|2-4|8";
  uint64_t step_id = 0UL;

  EXPECT_EQ(cpu_task_mark_step.Init(group_info, dump_step, step_id, true), SUCCESS);
  EXPECT_NE(cpu_task_mark_step.Distribute(), SUCCESS);

  cpu_task_mark_step.args_size_ = 1;
  auto ret = cpu_task_mark_step.Init(group_info, dump_step, step_id, true);
  EXPECT_EQ(ret, FAILED);

  cpu_task_mark_step.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_NE(cpu_task_mark_step.Init(group_info, dump_step, step_id, true), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_NE(cpu_task_mark_step.Init(group_info, dump_step, step_id, true), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskMarkStep_Init_Succ) {
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  CpuTaskMarkStep cpu_task_mark_step(stream);
  GroupInfo group_info;
  group_info.group_total_count = 0U;
  group_info.group_index = 0U;
  group_info.group_policy = 0U;
  std::string dump_step = "0|2-4|8";
  void* malloc_mem = nullptr;
  (void)rtMalloc(&malloc_mem, sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  uintptr_t step_id = static_cast<uintptr_t>(PtrToValue(malloc_mem));

  EXPECT_EQ(cpu_task_mark_step.Init(group_info, dump_step, step_id, false), SUCCESS);
  EXPECT_EQ(cpu_task_mark_step.Distribute(), SUCCESS);
  rtStreamDestroy(stream);
  rtFree(ValueToPtr(static_cast<uint64_t>(step_id)));
  step_id = 0U;
}

TEST_F(UtestCpuQueueSchedule, CpuTaskActiveEntry_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskActiveEntry cpu_task_active_entry(stream);
  rtStream_t stream2 = nullptr;

  EXPECT_NE(cpu_task_active_entry.Init(stream2), SUCCESS);   //??

  cpu_task_active_entry.args_size_ = 1;
  auto ret = cpu_task_active_entry.Init(stream2);
  EXPECT_EQ(ret, FAILED);

  cpu_task_active_entry.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_NE(cpu_task_active_entry.Init(stream2), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_NE(cpu_task_active_entry.Init(stream2), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskWaitEndGraph_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskWaitEndGraph cpu_task_wait_end_graph(stream);
  uint32_t model_id = 0U;

  EXPECT_EQ(cpu_task_wait_end_graph.Init(model_id), SUCCESS);

  cpu_task_wait_end_graph.args_size_ = 1;
  auto ret = cpu_task_wait_end_graph.Init(model_id);
  EXPECT_EQ(ret, FAILED);

  cpu_task_wait_end_graph.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_NE(cpu_task_wait_end_graph.Init(model_id), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_NE(cpu_task_wait_end_graph.Init(model_id), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelReportStatus_Init) {
  rtStream_t stream = nullptr;
  CpuTaskModelReportStatus cpu_task_model_report_status(stream);
  const uint32_t model_uuid = 0U;
  const QueueAttrs status_output_queue = {0, 0, 0, 0};
  const QueueAttrs input_queue = {0, 0, 0, 0};
  std::vector<QueueAttrs> input_queues;
  input_queues.emplace_back(input_queue);

  EXPECT_EQ(cpu_task_model_report_status.Init(model_uuid,
    status_output_queue, input_queues), SUCCESS);

  cpu_task_model_report_status.args_size_ = 1;
  EXPECT_EQ(cpu_task_model_report_status.Init(model_uuid,
    status_output_queue, input_queues), FAILED);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelReportStatus_Distribute) {
  rtStream_t stream = (rtStream_t)99;
  CpuTaskModelReportStatus cpu_task_model_report_status(stream);
  cpu_task_model_report_status.args_size_ = 1;
  cpu_task_model_report_status.args_ = (void *)1;

  EXPECT_EQ(cpu_task_model_report_status.Distribute(), SUCCESS);
  cpu_task_model_report_status.args_ = nullptr;
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelRepeat_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskModelRepeat cpu_task_model_repeat(stream);
  uint32_t model_id = 0U;

  EXPECT_EQ(cpu_task_model_repeat.Init(model_id), SUCCESS);

  cpu_task_model_repeat.args_size_ = 1;
  auto ret = cpu_task_model_repeat.Init(model_id);
  EXPECT_EQ(ret, FAILED);

  cpu_task_model_repeat.args_size_ = 0;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_NE(cpu_task_model_repeat.Init(model_id), SUCCESS);

  g_runtime_stub_mock = "rtMemcpy";
  EXPECT_NE(cpu_task_model_repeat.Init(model_id), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelBatchDequeue) {
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  CpuTaskModelBatchDequeue dequeue_task(stream);

  domi::TaskDef task_def;
  DavinciModel *davinci_model = nullptr;
  EXPECT_EQ(dequeue_task.Init(task_def, davinci_model), SUCCESS);

  std::vector<uint32_t> queue_ids{1, 2};
  std::vector<uint32_t> align_offsets{0, 1};
  std::vector<uintptr_t> in_mbufs;
  auto ret = dequeue_task.Init(1, queue_ids, align_offsets, in_mbufs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(in_mbufs.size(), 2);
  EXPECT_EQ(dequeue_task.Distribute(), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestCpuQueueSchedule, CpuTaskModelGatherDequeue) {
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  CpuTaskModelGatherDequeue dequeue_task(stream);

  domi::TaskDef task_def;
  DavinciModel *davinci_model = nullptr;
  EXPECT_EQ(dequeue_task.Init(task_def, davinci_model), SUCCESS);
  QueueAttrs queue_0 = {.queue_id = 100, .device_type = NPU, .device_id = 0, .logic_id = 0};
  QueueAttrs queue_1 = {.queue_id = 101, .device_type = CPU, .device_id = 0, .logic_id = 0};
  QueueAttrs queue_2 = {.queue_id = 102, .device_type = NPU, .device_id = 1, .logic_id = 0};
  std::vector<QueueAttrs> queues = {queue_0, queue_1, queue_2};
  InputAlignAttrs align_offsets = {.align_max_cache_num = 4, .align_timeout = 200, .drop_when_not_align = true};
  std::vector<uintptr_t> in_mbufs;
  auto ret = dequeue_task.Init(queues, align_offsets, in_mbufs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(in_mbufs.size(), 3);
  EXPECT_NE(dequeue_task.args_, nullptr);
  uint32_t queue_num = queues.size();
  EXPECT_EQ(dequeue_task.args_size_, sizeof(GatherDequeueKernelArgs) +
      queue_num *(sizeof(uint32_t) * 3 + sizeof(uint64_t)* 2));
  auto *ptr = reinterpret_cast<GatherDequeueKernelArgs *>(dequeue_task.args_);
  // check data in memory is expected
  // num timeout cache_num drop_out 
  // queue_list_ptr mbuf_list_addr_ptr device_id_ptr device_type_ptrs
  // queue_id_list input_addrs_list device_id_list device_type_list input_mbuff(prepare for aicpu to set real addr)
  EXPECT_EQ(ptr->input_nums, queue_num);
  EXPECT_EQ(ptr->inputs_align_timeout, align_offsets.align_timeout);
  EXPECT_EQ(ptr->inputs_align_max_cache_num, align_offsets.align_max_cache_num);
  EXPECT_EQ(ptr->inputs_align_drop_out, static_cast<int32_t>(align_offsets.drop_when_not_align));

  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_ids_addr)[0], 100);
  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_ids_addr)[1], 101);
  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_ids_addr)[2], 102);

  uint64_t input_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr->queue_device_type_addr)) +
      sizeof(uint32_t) * queue_num;
  EXPECT_EQ(reinterpret_cast<uint64_t *>(ptr->mbuf_addrs_addr)[0], input_addr);
  EXPECT_EQ(reinterpret_cast<uint64_t *>(ptr->mbuf_addrs_addr)[1], input_addr + sizeof(uint64_t));
  EXPECT_EQ(reinterpret_cast<uint64_t *>(ptr->mbuf_addrs_addr)[2], input_addr + sizeof(uint64_t) * 2);
  EXPECT_EQ(reinterpret_cast<uint64_t *>(ptr->mbuf_addrs_addr)[0], in_mbufs[0]);
  EXPECT_EQ(reinterpret_cast<uint64_t *>(ptr->mbuf_addrs_addr)[1], in_mbufs[1]);
  EXPECT_EQ(reinterpret_cast<uint64_t *>(ptr->mbuf_addrs_addr)[2], in_mbufs[2]);

  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_device_ids_addr)[0], 0);
  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_device_ids_addr)[1], 0);
  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_device_ids_addr)[2], 1);

  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_device_type_addr)[0], 0);
  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_device_type_addr)[1], 1);
  EXPECT_EQ(reinterpret_cast<uint32_t *>(ptr->queue_device_type_addr)[2], 0);

  EXPECT_EQ(dequeue_task.Distribute(), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestCpuQueueSchedule, CpuInputMemCpy_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskProcessInputsMemCopy cpu_task_input_cp(stream);
  std::vector<uintptr_t> mbuf_list;
  std::vector<uintptr_t> data_addr_list;
  std::vector<uint64_t> length_list;
  std::vector<int32_t> input_fusion_offset_list;

  EXPECT_EQ(cpu_task_input_cp.Init(mbuf_list, data_addr_list, length_list, input_fusion_offset_list), SUCCESS);
  EXPECT_EQ(cpu_task_input_cp.Distribute(), FAILED);

  mbuf_list.emplace_back(0x1);
  mbuf_list.emplace_back(0x1);
  data_addr_list.emplace_back(0x3);
  length_list.emplace_back(0x4);
  input_fusion_offset_list.emplace_back(0);
  EXPECT_EQ(cpu_task_input_cp.Init(mbuf_list, data_addr_list, length_list, input_fusion_offset_list), FAILED);
}

TEST_F(UtestCpuQueueSchedule, CpuInputMemCpy_Init_Succ) {
  int32_t stream_addr = 0x123;
  rtStream_t stream = static_cast<void *>(&stream_addr);
  CpuTaskProcessInputsMemCopy cpu_task_input_cp(stream);
  std::vector<uintptr_t> mbuf_list;
  std::vector<uintptr_t> data_addr_list;
  std::vector<uint64_t> length_list;
  std::vector<int32_t> input_fusion_offset_list;
  mbuf_list.emplace_back(0x1);
  data_addr_list.emplace_back(0x2);
  length_list.emplace_back(12);
  input_fusion_offset_list.emplace_back(0);
  EXPECT_EQ(cpu_task_input_cp.Init(mbuf_list, data_addr_list, length_list, input_fusion_offset_list), SUCCESS);
  EXPECT_EQ(cpu_task_input_cp.Distribute(), SUCCESS);
}

TEST_F(UtestCpuQueueSchedule, CpuInputShapeCheck_Init_failed) {
  rtStream_t stream = nullptr;
  CpuTaskProcessInputsShapeCheck cpu_task_input_shape_chk(stream);
  cpu_task_input_shape_chk.args_size_ = 1;
  std::vector<uintptr_t> mbuf_list;
  std::vector<int32_t> input_fusion_offset_list;
  EXPECT_EQ(cpu_task_input_shape_chk.Init(mbuf_list, input_fusion_offset_list), FAILED);

  cpu_task_input_shape_chk.args_size_ = 0;
  EXPECT_EQ(cpu_task_input_shape_chk.Init(mbuf_list, input_fusion_offset_list), SUCCESS);
  EXPECT_EQ(cpu_task_input_shape_chk.Distribute(), FAILED);

  mbuf_list.emplace_back(0x1);
  mbuf_list.emplace_back(0x1);
  input_fusion_offset_list.emplace_back(0);
  EXPECT_EQ(cpu_task_input_shape_chk.Init(mbuf_list, input_fusion_offset_list), FAILED);
}

TEST_F(UtestCpuQueueSchedule, CpuInputShapeCheck_Init_Succ) {
  int32_t stream_addr = 0x123;
  rtStream_t stream = static_cast<void *>(&stream_addr);
  CpuTaskProcessInputsShapeCheck cpu_task_input_shape_chk(stream);
  std::vector<uintptr_t> mbuf_list;
  std::vector<int32_t> input_fusion_offset_list;
  mbuf_list.emplace_back(0x1);
  input_fusion_offset_list.emplace_back(0);

  EXPECT_EQ(cpu_task_input_shape_chk.Init(mbuf_list, input_fusion_offset_list), SUCCESS);
  EXPECT_EQ(cpu_task_input_shape_chk.Distribute(), SUCCESS);
}
}  // namespace ge
