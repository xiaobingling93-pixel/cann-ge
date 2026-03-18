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
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_scheduler_config.h"
#include "fake_execution_data.h"
#include "core/executor_error_code.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/global_variables/diagnose_switch.h"
#include "runtime/subscriber/executor_subscribers_scheduler.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "subscriber/tracer/executor_tracer.h"
#include "stub/acl_runtime_stub_impl.h"
#include "stub/runtime_stub_impl.h"

using namespace gert;

namespace {
struct RuntimeStubGuard {
  RuntimeStubGuard()
      : runtime_stub(std::make_shared<RuntimeStubImpl>()),
        acl_runtime_stub(std::make_shared<AclRuntimeStubImpl>()) {
    ge::RuntimeStub::SetInstance(runtime_stub);
    ge::AclRuntimeStub::SetInstance(acl_runtime_stub);
  }
  ~RuntimeStubGuard() {
    ge::RuntimeStub::Reset();
    ge::AclRuntimeStub::Reset();
  }

  std::shared_ptr<RuntimeStubImpl> runtime_stub;
  std::shared_ptr<AclRuntimeStubImpl> acl_runtime_stub;
};
}  // namespace

class TaskSchedulerUnitTest : public testing::Test {
  void SetUp() override {
    KernelSpy::GetInstance().Clear();
  }
};

TEST_F(TaskSchedulerUnitTest, should_schedule_single_task_in_single_worker) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);

  FakeExecutionData executionData(2);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  scheduler->Dump();

  ASSERT_EQ(1, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(1, scheduler->GetCompletedTaskCount());
  KERNEL_RUN_EXPECT(3, 7, 5, 8, 6);
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_without_execute_args_and_skip_stream_binding) {
  RuntimeStubGuard runtime_stub_guard;
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::SINGLE;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::LOW_LOAD, 1);

  FakeExecutionData execution_data(10);
  execution_data.Chain({3, 7, 6}).StartNodes({3});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{execution_data.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  delete scheduler;

  EXPECT_TRUE(runtime_stub_guard.acl_runtime_stub->GetUseStreamResRecords().empty());
  EXPECT_TRUE(runtime_stub_guard.acl_runtime_stub->GetNotUseStreamResRecords().empty());
}

TEST_F(TaskSchedulerUnitTest, should_rebind_execute_stream_between_schedules_and_unbind_on_stop) {
  RuntimeStubGuard runtime_stub_guard;
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;
  cfg.worker_cfgs.resize(1);

  FakeExecutionData first_execution_data(10);
  first_execution_data.Chain({3, 7, 6}).StartNodes({3}).ExecuteStream(reinterpret_cast<aclrtStream>(0x11));

  FakeExecutionData second_execution_data(10);
  second_execution_data.Chain({5, 8, 6}).StartNodes({5}).ExecuteStream(reinterpret_cast<aclrtStream>(0x22));

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_NE(scheduler, nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{first_execution_data.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11)}));
  EXPECT_TRUE(runtime_stub_guard.acl_runtime_stub->GetNotUseStreamResRecords().empty());

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{second_execution_data.Data()}));
  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11), reinterpret_cast<aclrtStream>(0x22)}));
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetNotUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11)}));

  delete scheduler;
  EXPECT_EQ(runtime_stub_guard.acl_runtime_stub->GetNotUseStreamResRecords(),
            std::vector<aclrtStream>({reinterpret_cast<aclrtStream>(0x11), reinterpret_cast<aclrtStream>(0x22)}));
}

TEST_F(TaskSchedulerUnitTest, should_schedule_chain_task_in_single_worker) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;
  cfg.worker_cfgs.resize(1);

  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(3, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(3, scheduler->GetCompletedTaskCount());
  KERNEL_RUN_EXPECT(3, 7, 5, 8, 6);
  delete scheduler;
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
}

TEST_F(TaskSchedulerUnitTest, schedule_kernel_task_end_of_squence) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::KERNEL;
  cfg.producer_cfg.thread_num = 3;
  cfg.worker_cfgs.resize(1);

  FakeExecutionData executionData(10);
  executionData.Chain({0,1}).StartNodes({0});
  executionData.FuncEndOfSequence(1, ge::END_OF_SEQUENCE);

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  ASSERT_EQ(ge::END_OF_SEQUENCE, scheduler->Schedule());
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_chain_task_in_multiple_workers) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;
  cfg.worker_cfgs.resize(5);

  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(3, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(3, scheduler->GetCompletedTaskCount());
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_large_chain_task_in_multiple_workers) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::CHAIN;

  cfg.AddWorkers(2, ExecTaskType::NORMAL, TaskThreadMode::URGENT, 1);

  FakeExecutionData executionData(20);
  executionData.Chain({1, 2, 3, 4, 8, 11, 12}).Chain({1, 5, 6, 7, 8}).Chain({7, 9, 10, 12}).StartNodes({1});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(3, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(3, scheduler->GetCompletedTaskCount());
  KERNEL_RUN_EXPECT(1, 2, 3, 4, 5, 6, 7, 9, 10, 8, 11, 12);

  scheduler->Dump();
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_op_task_in_multiple_thread_workers) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::OP;
  cfg.worker_cfgs.resize(1);
  cfg.worker_cfgs[0].thread_count = 2;

  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "CalcSize"}},
                   {5, {"transdata", "AllocMemHbm"}},
                   {6, {"transdata", "Tiling"}},
                   {7, {"transdata", "Launch"}},
                   {8, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3})
      .Chain({1, 4, 3})
      .Chain({5, 6, 7})
      .Chain({1, 5})
      .Chain({3, 7})
      .Chain({7, 8})
      .StartNodes({1});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData{executionData.Data()}));

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(5, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(5, scheduler->GetCompletedTaskCount());

  scheduler->Dump();
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, should_schedule_op_task_in_multiple_thread_workers_by_type) {
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::OP;
  cfg.worker_cfgs.resize(2);
  cfg.worker_cfgs[0].thread_count = 1;
  cfg.worker_cfgs[0].bind_task_type = ExecTaskType::MEMORY;
  cfg.worker_cfgs[1].thread_count = 2;

  FakeExecutionData executionData(10);
  executionData
      .KernelAttr({{1, {"conv2d", "AllocMemHbm"}},
                   {2, {"conv2d", "Tiling"}},
                   {3, {"conv2d", "Launch"}},
                   {4, {"conv2d", "CalcSize"}},
                   {5, {"transdata", "AllocMemHbm"}},
                   {6, {"transdata", "Tiling"}},
                   {7, {"transdata", "Launch"}},
                   {8, {"Netoutput", "Output"}}})
      .Chain({1, 2, 3})
      .Chain({1, 4, 3})
      .Chain({5, 6, 7})
      .Chain({1, 5})
      .Chain({3, 7})
      .Chain({7, 8})
      .StartNodes({1});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data()));
  scheduler->DumpBrief();

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule());

  ASSERT_EQ(5, scheduler->GetScheduledTaskCount());
  ASSERT_EQ(5, scheduler->GetCompletedTaskCount());

  scheduler->Dump();
  delete scheduler;
}

TEST_F(TaskSchedulerUnitTest, sub_thread_profiling_report_success) {
  ge::diagnoseSwitch::EnableProfiling({ProfilingType::kTaskTime});
  size_t report_event_count = 0U;
  auto default_check_func = [&](uint32_t moduleId, uint32_t type, void *data,
                                uint32_t len) -> int32_t {
    if (type == ge::InfoType::kEvent) {
      ++report_event_count;
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(default_check_func);
  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerType::KERNEL;
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::URGENT, 2);

  FakeExecutionData executionData(10);
  executionData.Chain({3, 7, 6}).Chain({5, 8, 6}).StartNodes({3, 5});

  auto scheduler = TaskSchedulerFactory::GetInstance().Create(cfg);
  ASSERT_TRUE(scheduler != nullptr);

  ASSERT_EQ(ge::GRAPH_SUCCESS, scheduler->Prepare(TaskScheduler::ScheduleData(executionData.Data())));

  sleep(1);
  ExecutorSubscribersScheduler ess;
  const auto kEnableFunc = []() -> bool { return true; };
  ess.AddBuiltIn<ExecutorTracer>(BuiltInSubscriberType::kTracer, 1UL, nullptr, kMainExeGraph, kEnableFunc);

  ASSERT_EQ(kStatusSuccess, scheduler->Schedule(kMainExeGraph, &ess.GetSubscriber(kMainExeGraph)));

  EXPECT_EQ(report_event_count, 4);
  delete scheduler;
  ge::ProfilingTestUtil::Instance().Clear();
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0);
}