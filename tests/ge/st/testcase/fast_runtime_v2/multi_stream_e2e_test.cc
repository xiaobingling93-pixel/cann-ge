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
#include <iostream>
#include "faker/fake_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "runtime/model_v2_executor.h"
#include "common/bg_test.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/dynamic_rnn_impl.h"
#include "op_impl/data_flow_op_impl.h"
#include "lowering/model_converter.h"
#include "securec.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_context.h"
#include "check/executor_statistician.h"
#include "common/helper/model_parser_base.h"
#include "common/executor_tracer_on.h"
#include "depends/checker/memory_profiling_log_matcher.h"
#include "common/topo_checker.h"
#include "depends/checker/mem_trace_checker.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace gert {
namespace {
std::unique_ptr<gert::Allocators> CreateDefaultAllocators() {
  std::shared_ptr<ge::Allocator> device_allocator(AllocatorFactory::Create(kOnDeviceHbm).release());
  std::shared_ptr<ge::Allocator> host_allocator(AllocatorFactory::Create(kOnHost).release());
  if ((device_allocator == nullptr) || (host_allocator == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "device_allocator is nullptr or host_allocator is nullptr");
  }
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  for (size_t i = 0U; i < static_cast<size_t>(kTensorPlacementEnd); ++i) {
    for (size_t j = 0U; j < static_cast<size_t>(AllocatorUsage::kEnd); ++j) {
      if (i == static_cast<size_t>(kOnDeviceHbm)) {
        allocators->SetAllocator(static_cast<TensorPlacement>(i), j, device_allocator);
      } else if (i == static_cast<size_t>(kOnHost) || i == static_cast<size_t>(kFollowing)) {
        allocators->SetAllocator(static_cast<TensorPlacement>(i), j, host_allocator);
      } else {
        GELOGE(ge::PARAM_INVALID, "Unsupported placement %zu to set allocator", i);
      }
    }
  }
  return allocators;
}

class AicpuTfLaunchStub : public RuntimeStub {
 public:
  rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                          const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                          rtStream_t stream, uint32_t flags) override {
    EXPECT_EQ(argsInfo->kernelOffsetInfoNum, 3);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].addrOffset, 80);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[0].dataOffset, 112);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].addrOffset, 88);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[1].dataOffset, 126);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[2].addrOffset, 104);
    EXPECT_EQ(argsInfo->kernelOffsetInfoPtr[2].dataOffset, 150);
    return RT_ERROR_NONE;
  }
};

ge::graphStatus InferShapeStub(InferShapeContext *context) { return SUCCESS;}
IMPL_OP(Conv2d).InferShape(InferShapeStub);
IMPL_OP(Relu).InferShape(InferShapeStub);
} // namespace
class GraphExecutorMultiStreamSystemTest : public bg::BgTest {
 protected:
  void SetUp() override {
    setenv("ENABLE_TILING_CACHE", "0", 1);
    bg::BgTest::SetUp();
    rtSetDevice(0);
    std::string opp_path = "./";
    std::string opp_version = "version.info";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    system(("touch " + opp_version).c_str());
    system(("echo 'Version=3.20.T100.0.B356' > " + opp_version).c_str());
    memory::RtsCachingMemAllocator::GetAllocator(0, RT_MEMORYINFO_HBM)->Recycle();
    memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
  }

  void TearDown() override {
    Test::TearDown();
    system("rm -f ./version.info");
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("ENABLE_TILING_CACHE");
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
  }
};

const std::string TransDataStubName = "TransDataStubBin";
const std::string TransData13StubName = "TransData17StubBin";
const std::string DynamicAtomicStubName = "DynamicAtomicBin";
const std::string DynamicRnnv3StubName = "DynamicRNNV3StubBin";
const std::string AddStubName = "AddStubBin";
const std::string AssignStubName = "AssignStubBin";
const std::string CastStubName = "CastStubBin";
const std::string ReluStubName = "ReluStubBin";
const std::string MulStubName = "MulStubBin";
const std::string ReduceSumStubName = "ReduceSumStubBin";

/*
 *  data1  data2
 *    \   /
 *     add1(streamid=0)(send_id_list:[0])
 *      |
 *     relu (streamid=1)(send_id_list:[1],recive_id_list:[0])
 *      |
 *    netoutput(streamid=0)(recive_id_list:[1]))
 *
 * 1. 用例描述：测试两个级联节点分别在两条流上的执行
 * 2. 预置条件：
 *   （1）外部创建主Stream
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) kernel trace流相关的节点执行序正确
     （2）add节点的stream为主流，relu节点的stream为辅流
     （3）通过日志校验add的输出内存生命周期：申请、流转、本地回收、借用回收、出生地回收
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case01_TwoStream_AccessMemCrossStream_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_AccessMemCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    auto ess = StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    // check stream in launch arg
    auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(AddStubName);
    ASSERT_NE(add_launch_args, nullptr);
    ASSERT_EQ(add_launch_args->GetStream(), stream);  // main stream 0
    auto relu_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(ReluStubName);
    ASSERT_NE(relu_launch_args, nullptr);
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    ASSERT_EQ(relu_launch_args->GetStream(), all_rt_streams[1]);  // stream 1

    // check kernel trace
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "WaitEvents"), 1);

    // check task on stream 0
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream0 = {
        TaskTypeOnStream::rtKernelLaunchWithFlagV2, TaskTypeOnStream::rtEventRecord,
        TaskTypeOnStream::rtStreamWaitEvent, TaskTypeOnStream::rtMemcpyAsync, TaskTypeOnStream::rtStreamWaitEvent};
    auto task_on_stream0 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(stream);
    EXPECT_EQ(task_on_stream0.size(), expect_task_infos_on_stream0.size());
    for (size_t i = 0; i < task_on_stream0.size(); ++i) {
      EXPECT_EQ(task_on_stream0[i], expect_task_infos_on_stream0[i]);
    }

    // check task on stream 1
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {TaskTypeOnStream::rtStreamWaitEvent,
                                                                  TaskTypeOnStream::rtKernelLaunchWithFlagV2,
                                                                  TaskTypeOnStream::rtEventRecord, TaskTypeOnStream::rtEventRecord};
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }

    // todo check error log in allocator, if has mem leak or not
    // check add output addr life cycle
    EXPECT_TRUE(MemoryTraceChecker(runtime_stub.GetSlogStub(), add_launch_args->GetLaunchAddresses()[2])
        .AppendExpectEvent(kAllocRe, 0) // (1) alloc in stream 0
        .AppendExpectEventWithSrc(kWander, 0, 1) // (2) wander from stream 0 to stream 1
        .AppendExpectEvent(kFreeRe, 0)  // (3) free on stream 0, trigger LocalRecycle
        .AppendExpectEvent(kLocalRecycleRe, 0)
        .AppendExpectEvent(kSendEventWithMem, 0)  // (4) send memblock which local recyled from stream 0, wait at stream 1
        .AppendExpectEvent(kWaitEventWithMem, 1)
        .AppendExpectEvent(kFreeRe, 1) // (5) free at stream 1, trigger BorrowRecycle
        .AppendExpectEvent(kBorrowRecycleRe, 1)
        .AppendExpectEvent(kSendEventWithMem, 1)  // (6) send memblock which need return to birth from stream 1, wait at stream 0
        .AppendExpectEvent(kWaitEventWithMem, 0)
        .AppendExpectEvent(kBirthRecycleRe, 0) // (7) trigger BirthRecycle at stream 0
        .AsYouWish());
    model_executor.reset(nullptr);
    rtStreamDestroy(stream);
    auto stream_addr = "0x" + std::to_string(reinterpret_cast<uint64_t>(stream));
    EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kPoolShrink, {{1, stream_addr}}) >= 0);
  }
  runtime_stub.Clear();
}

/*
 *      data1
 *       |
 *      cast(stream:0)(send[0])
 *      /    \
 * transdata    relu (stream:1)(send:[1],recive:[0])
 * (stream:0)  /
 *      \     /
 *    netoutput(stream:0)(recive_id_list:[1]))
 *
 * 1. 用例描述：测试某个单输出多引用内存，消费者在当前流和跨流上.同时测试atomic clean下发在辅流
 * 2. 预置条件：
 *   （1）外部创建主Stream
 *    (2) fake relu算子携带atomic clean task
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) kernel trace流相关的节点执行序正确.
     （2）cast/transdata节点的stream为主流，relu节点的stream为辅流
      (3) cast和transdata之间无跨流访问节点
     （4）relu的atomic clean task也下发在辅流
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case02_TwoStream_ConsumersInAndCrossStream_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphConsumersInAndCrossStream(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Cast", AiCoreTaskDefFaker(CastStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName).AtomicStubNum(DynamicAtomicStubName))
      .AddTaskDef("TransData", AiCoreTaskDefFaker(TransDataStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_ConsumersInAndCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    auto ess = StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({2048}, 2);
    auto inputs = FakeTensors({2048}, 1);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();

    // check stream in launch arg
    auto cast_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(CastStubName);
    ASSERT_NE(cast_launch_args, nullptr);
    ASSERT_EQ(cast_launch_args->GetStream(), stream);

    auto transdata_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(TransDataStubName);
    ASSERT_NE(transdata_launch_args, nullptr);
    ASSERT_EQ(transdata_launch_args->GetStream(), stream);

    auto relu_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(ReluStubName);
    ASSERT_NE(relu_launch_args, nullptr);
    ASSERT_EQ(relu_launch_args->GetStream(), all_rt_streams[1]);

    // check kernel trace
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Cast", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("TransData", "AccessMemCrossStream"), 0);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "WaitEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NetOutput", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NetOutput", "WaitEvents"), 1);

    // check task on stream 0
    std::vector<TaskTypeOnStream> expect_task_infos_on_main = {
        TaskTypeOnStream::rtKernelLaunchWithFlagV2, TaskTypeOnStream::rtEventRecord,
        TaskTypeOnStream::rtKernelLaunchWithFlagV2, TaskTypeOnStream::rtStreamWaitEvent,
        TaskTypeOnStream::rtMemcpyAsync, TaskTypeOnStream::rtStreamWaitEvent};
    auto task_on_stream = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(stream);
    EXPECT_EQ(task_on_stream.size(), expect_task_infos_on_main.size());
    for (size_t i = 0U; i < task_on_stream.size(); ++i) {
      EXPECT_EQ(task_on_stream[i], expect_task_infos_on_main[i]);
    }
    // check task on stream 1
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {
        TaskTypeOnStream::rtStreamWaitEvent, TaskTypeOnStream::rtKernelLaunchWithFlagV2,
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // atomic clean
        TaskTypeOnStream::rtEventRecord, TaskTypeOnStream::rtEventRecord};
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }

    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}
/*
 *     data1(stream:0)
 *       |
 *     shape
 * (stream:0)(send:1)   data2(stream:0)(send:0)
 *             \    /
 *              add (stream:1)(send:[2],recive:[0,1])
 *               |
 *            netoutput(stream:0)(recive:[2])
 *
 * 1. 用例描述：测试host内存跨流访问，框架产生的H2D拷贝，dst stream要与目标辅流算子一致
 * 2. 预置条件：
 *   （1）外部创建主Stream
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) kernel trace流相关的节点执行序正确.
     （2）shape stream为主流，add节点的stream为辅流
      (3) shape和add之间有拷贝和跨流访问节点

待修改：
1. copyH2D和send event的控制关系
2. placedLoweringResult的src和dst流
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case03_TwoStream_HostMemAccessCrossStream_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamWithHostMemAccessCrossStream(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_HostMemAccessCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    auto ess = StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({4}, 1);
    auto input0 = FakeTensors({3, 4, 5, 6}, 1);
    auto input1 = FakeTensors({4}, 1);
    std::vector<Tensor *> inputs = {input0.GetTensorList()[0], input1.GetTensorList()[0]};

    ASSERT_EQ(
        model_executor->Execute({i3.value}, inputs.data(), inputs.size(), outputs.GetTensorList(), outputs.size()),
        ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    // check stream in launch arg
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    auto add_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(AddStubName);
    ASSERT_NE(add_launch_args, nullptr);
    ASSERT_EQ(add_launch_args->GetStream(), all_rt_streams[1]);  // stream 1

    // check kernel trace
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Data", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Shape", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "CopyFlowLaunch"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NetOutput", "WaitEvents"), 1);

    // check task on stream 0
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream0 = {
        TaskTypeOnStream::rtEventRecord,      // data1 send
        TaskTypeOnStream::rtEventRecord,      // shape send
        TaskTypeOnStream::rtStreamWaitEvent,  // netoutput wait
        TaskTypeOnStream::rtMemcpyAsync, TaskTypeOnStream::rtStreamWaitEvent};     // model copy?
    auto task_on_stream0 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(stream);
    EXPECT_EQ(task_on_stream0.size(), expect_task_infos_on_stream0.size());
    for (size_t i = 0; i < task_on_stream0.size(); ++i) {
      EXPECT_EQ(task_on_stream0[i], expect_task_infos_on_stream0[i]);
    }

    // check task on stream 1
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {
        //  TaskTypeOnStream::rtMemcpyAsync,             // copy h2d optimized to copy flow launch
        TaskTypeOnStream::rtStreamWaitEvent,         // wait event 0
        TaskTypeOnStream::rtStreamWaitEvent,         // wait event 1
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // add launch
        TaskTypeOnStream::rtEventRecord, TaskTypeOnStream::rtEventRecord};            // add send
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }

    // check shape output goes to launch args
    EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoNum, 1);
    EXPECT_EQ(add_launch_args->GetArgsEx()->hostInputInfoPtr[0].addrOffset, 0); //?
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}

/*
 * refdata1   data1
 *(stream:0)  (stream:0)
 *      \     /
 *      assign(stream:0)(send[0])
 *      /   \
 * transdata  relu (stream:1)(send:[1],recive:[0])
 * (stream:0)  /
 *      \     /
 *    netoutput(stream:0)(recive_id_list:[1]))
 *
 * 1. 用例描述：测试引用类内存的跨流访问，assign类算子输出引用输入，其输出跨流访问
 * 2. 预置条件：
 *   （1）外部创建主Stream
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) kernel trace流相关的节点执行序正确.
     （2）assign输入内存来自用户输入，在主流上引用+1，在流1上跨流访问，最后在流0上
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case04_TwoStream_AccessRefMemCrossStream_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphAccessRefMemCrossStream(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Assign", AiCoreTaskDefFaker(AssignStubName))
      .AddTaskDef("TransData", AiCoreTaskDefFaker(TransDataStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_AccessRefMemCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    auto ess = StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({3, 4, 5, 6}, 2);
    auto inputs = FakeTensors({3, 4, 5, 6}, 2);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    // check stream in launch arg
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    auto assign_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(AssignStubName);
    ASSERT_NE(assign_launch_args, nullptr);
    ASSERT_EQ(assign_launch_args->GetStream(), all_rt_streams[0]);  // stream 0

    auto relu_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(ReluStubName);
    ASSERT_NE(relu_launch_args, nullptr);
    ASSERT_EQ(relu_launch_args->GetStream(), all_rt_streams[1]);  // stream 1

    // check kernel trace
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Assign", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "WaitEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NetOutput", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("NetOutput", "WaitEvents"), 1);

    // check task on stream 0
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream0 = {
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // assign launch
        TaskTypeOnStream::rtEventRecord,             // assign send
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // transdata launch
        TaskTypeOnStream::rtStreamWaitEvent,         // netoutput wait
        TaskTypeOnStream::rtMemcpyAsync, TaskTypeOnStream::rtStreamWaitEvent};            // model copy?
    auto task_on_stream0 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(stream);
    EXPECT_EQ(task_on_stream0.size(), expect_task_infos_on_stream0.size());
    for (size_t i = 0; i < task_on_stream0.size(); ++i) {
      EXPECT_EQ(task_on_stream0[i], expect_task_infos_on_stream0[i]);
    }

    // check task on stream 1
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {
        TaskTypeOnStream::rtStreamWaitEvent,         // relu wait
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // relu launch
        TaskTypeOnStream::rtEventRecord,  TaskTypeOnStream::rtEventRecord           // relu send
    };
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }

    // check refdata1 output addr life cycle
    const auto refdata_addr = assign_launch_args->GetLaunchAddresses()[0];
    EXPECT_TRUE(MemoryTraceChecker(runtime_stub.GetSlogStub(), refdata_addr)
                    .AppendExpectEvent(kFreeRe, 0)  // (1) free in stream 0
                    .AppendExpectEventWithSrc(kWander, 0, 1) // (2) wander from stream 0 to stream1. relu asscess mem from assign. trigger free on stream0
                    .AppendExpectEvent(kFreeRe, 0) // (3) free on stream1
                    .AppendExpectEvent(kFreeRe, 1)
                    .AppendExpectEvent(kLocalRecycleRe, 1)
                    .AppendExpectEvent(kSendEventWithMem, 1)
                    .AppendExpectEvent(kWaitEventWithMem, 0)
                    .AsYouWish());
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}

/*
 * refdata1    data1
 *(stream:0)  (stream:0)
 *(send:[0])  (send:[1])
 *      \     /
 *      assign(stream:1)(send:[2], recive:[0,1])
 *        |
 *    netoutput(stream:0)(recive_id_list:[2]))
 *
 *
 * 1. 用例描述：测试引用类内存的跨流引用，assign类算子输出引用输入，其输入为跨流访问的内存
 * 2. 预置条件：
 *   （1）外部创建主Stream
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
      (3)校验图结构
     （4）todo 构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) assign输入内存来自用户输入，输出引用了refdata，也引用了输入，最终使用了输入的value holder
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case05_TwoStream_RefMemAccessCrossStream_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphRefMemCrossStream(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Assign", AiCoreTaskDefFaker(AssignStubName))
      .AddTaskDef("TransData", AiCoreTaskDefFaker(TransDataStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_RefMemAccessCrossStream");

  // check the topo order is correct
  auto assign_launch_node =
    ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "LaunchKernelWithFlag");
  ASSERT_EQ(assign_launch_node.size(), 1U);
  EXPECT_EQ(FastNodeTopoChecker(assign_launch_node.at(0))
                .StrictConnectFrom({{"SplitRtStreams",1},
                                    {"InnerData",0},
                                    {"CacheableTiling", 1},
                                    {"AllocBatchHbm", 0},
                                    {"InnerData", 0},
                                    {"InnerData", 0},
                                    {"InnerData", 0},
                                    {"CacheableTiling", 6},
                                    {"InnerData", 0},
                                    {"CacheableTiling", 9},
                                    {"CacheableTiling", 7},
                                    {"AccessMemCrossStream", 0}, // input 0
                                    {"AccessMemCrossStream", 0},  // input 1
                                    {"AccessMemCrossStream", 0}, // output 0 ref input 0
                                    {"SplitDataTensor", 0},
                                    {"SplitDataTensor", 0},
                                    {"InferShape", 0},
                                    {"WaitEvents", -1},
                                    {"SendEvents", -1},
                                    {"SendEvents", -1}}),
            "success");
}

/*
     ┌──────────────────────────────────────────────┐
     │                                              │
     │   ┌──────┐                                   │
     │   │data_i├───┐                               │
     │   └──────┘   │ ┌───┐     ┌───────────────┐   │
     │              │►│add├────►│sub_1_netoutput│   │
     |              | |s1 |     |               |   |
     │   ┌───────┐  │ └───┘     └───────────────┘   │
     │   │const_1├──┘                               │
     │   └───────┘                                  │
     │                                              │
     └─────────────────────────┬────────────────────┘
                               │
 ┌───────┐     ┌────┐     ┌────▼───┐
 │ data_1├────►│relu├────►│known_op├───┐
 |s0     |     | s1 |     |  s0    |   |
 └───────┘     └────┘     └────────┘   │  ┌──────────────┐
                                       ├─►│root_netoutput|
                                       │  |   s0         |
 ┌───────┐               ┌──────────┐  │  └──────────────┘
 │ data_2├──────────────►│   relu1  ├──┘
 |  s0   |               |    s0    |
 └───────┘               └────-─────┘
 *
 * 1. 用例描述：测试带静态子图的多流场景
 * 2. 预置条件：
 *   （1）外部创建主Stream
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) 静态子图节点在主流0上
     （2）与前序节点存在event等待
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case06_TwoStream_WithStaticSubGraph_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphDynamicAndStaticGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("TransData", AiCoreTaskDefFaker(TransDataStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_DynamicAndStaticGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    auto ess = StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    RtSession rt_session;
    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({3, 4, 5, 6}, 2);
    auto inputs = FakeTensors({3, 4, 5, 6}, 1);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    // check stream in launch arg
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    ASSERT_EQ(all_rt_streams.size(), stream_num);
    auto relu_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(ReluStubName);
    ASSERT_NE(relu_launch_args, nullptr);
    ASSERT_EQ(relu_launch_args->GetStream(), all_rt_streams[1]);  // stream 1. 0~1 dynamic stream, 2~4 static stream

    auto transdata_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(TransDataStubName);
    ASSERT_NE(transdata_launch_args, nullptr);
    ASSERT_EQ(transdata_launch_args->GetStream(), all_rt_streams[0]);  // stream 0

    // check kernel trace
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "WaitEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Relu", "SendEvents"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "AccessMemCrossStream"), 1);
    EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("PartitionedCall", "WaitEvents"), 1);

    // check task on stream 0
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream0 = {
        TaskTypeOnStream::rtEventRecord,             // data1 send
        TaskTypeOnStream::rtStreamWaitEvent,         // partitioned call wait
        TaskTypeOnStream::rtMemcpyAsync,             // davinci model input copy
        TaskTypeOnStream::rtModelExecute,            // partitioned execute
        TaskTypeOnStream::rtMemcpyAsync,             // davinci model output copy
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // transdata launch
        TaskTypeOnStream::rtMemcpyAsync,
        TaskTypeOnStream::rtStreamWaitEvent,
        TaskTypeOnStream::rtStreamWaitEvent,
        TaskTypeOnStream::rtStreamWaitEvent,         // branch to main stream
    };  // model copy?
    auto task_on_stream0 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(stream);
    EXPECT_EQ(task_on_stream0.size(), expect_task_infos_on_stream0.size());
    for (size_t i = 0; i < task_on_stream0.size(); ++i) {
      EXPECT_EQ(task_on_stream0[i], expect_task_infos_on_stream0[i]);
    }

    // check task on stream 1
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {
        TaskTypeOnStream::rtStreamWaitEvent,         // relu wait
        TaskTypeOnStream::rtKernelLaunchWithFlagV2,  // relu launch
        TaskTypeOnStream::rtEventRecord,             // relu send'
        TaskTypeOnStream::rtEventRecord
    };
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }
    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}

/*

 ┌───────---┐               ┌──────────┐   ┌──────────────┐
 │ fileconst├──────────────►│   relu1  ├-->│root_netoutput|
 |  s0      |               |    s1    |   |   s0         |
 └───────---┘               └────-─────┘   └──────────────┘
 *
 * 1. 用例描述：测试带FileConstant的多流场景
 * 2. 预置条件：
 *   （1）外部创建主Stream
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) graph conveter success
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case07_TwoStream_WithFileConstant_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphFileConstantGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_WithFileConstant");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    (void)StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({stream},{&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({3, 4, 5, 6}, 1);
    auto inputs = FakeTensors({3, 4, 5, 6}, 0);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}
/*
 *  data1(s0)  data2(streamid=1)(send_id_list:[0])
 *         \   /
 *         add1(streamid=0)(recive_id_list:[0])
 *          |
 *       netoutput(streamid=0)
 * 1. 用例描述：测试带首部流同步的多流场景
 * 2. 预置条件：
 *   （1）外部创建主Stream
           （2）使用data模拟GetNext，使其编译时分流到辅流
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) 有从主流到辅流的同步事件
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case07_TwoStream_WithFirstEventSync_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphWithFirstEventSyncGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_AccessMemCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    (void)StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    // check task on stream 0
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream0 = {
        TaskTypeOnStream::rtEventRecord, TaskTypeOnStream::rtStreamWaitEvent,
        TaskTypeOnStream::rtKernelLaunchWithFlagV2, TaskTypeOnStream::rtStreamWaitEvent};
    auto task_on_stream0 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(stream);
    EXPECT_EQ(task_on_stream0.size(), expect_task_infos_on_stream0.size());
    for (size_t i = 0; i < task_on_stream0.size(); ++i) {
      EXPECT_EQ(task_on_stream0[i], expect_task_infos_on_stream0[i]);
    }

    // check task on stream 1
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {TaskTypeOnStream::rtStreamWaitEvent,
                                                                  TaskTypeOnStream::rtEventRecord, TaskTypeOnStream::rtEventRecord};
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }
    model_executor.reset(nullptr);
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}

/*
 *        data1(s0)                            data2(s0)(send_id_list:[1])
 *               \                                  |
 *             relu(streamid=0)(send_id_list:[0])   |
 *             /          \                         |
 *   netoutput(streamid=0)  add(s1)(recive_id_list:[0,1])
 * 1. 用例描述：测试带首部流同步的多流场景
 * 2. 预置条件：
 *   （1）外部创建主Stream
           （2）使用data模拟GetNext，使其编译时分流到辅流
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) 有从主流到辅流的同步事件
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case08_TwoStream_WithLastEventSync_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamGraphWithLastEventSyncGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker add_task_def;
  auto ge_root_model = builder.AddTaskDef("Relu", AiCoreTaskDefFaker(AddStubName)).AddTaskDef("Add", add_task_def)
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_WithLastEventSync");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
/*  auto aicpu_launch_stub = std::make_shared<AicpuTfLaunchStub>();
  RuntimeStub::Install(aicpu_launch_stub.get());*/
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    (void)StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    EXPECT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    // check task on stream 0
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    EXPECT_EQ(stream, all_rt_streams[0]);
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream0 = {TaskTypeOnStream::rtKernelLaunchWithFlagV2,
                                                                  TaskTypeOnStream::rtEventRecord,
                                                                  TaskTypeOnStream::rtEventRecord,
                                                                  TaskTypeOnStream::rtMemcpyAsync,
                                                                  TaskTypeOnStream::rtStreamWaitEvent,
                                                                  TaskTypeOnStream::rtStreamWaitEvent};
    auto task_on_stream0 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[0]);
    EXPECT_EQ(task_on_stream0.size(), expect_task_infos_on_stream0.size());
    for (size_t i = 0; i < task_on_stream0.size(); ++i) {
      EXPECT_EQ(task_on_stream0[i], expect_task_infos_on_stream0[i]);
    }

    // check task on stream 1
    std::vector<TaskTypeOnStream> expect_task_infos_on_stream1 = {TaskTypeOnStream::rtStreamWaitEvent, // wait input
                                                                  TaskTypeOnStream::rtStreamWaitEvent, // wait input
                                                                  TaskTypeOnStream::rtEventRecord,// last send
                                                                  TaskTypeOnStream::rtEventRecord};
    auto task_on_stream1 = runtime_stub.GetRtsRuntimeStub().GetAllTaskOnStream(all_rt_streams[1]);
    EXPECT_EQ(task_on_stream1.size(), expect_task_infos_on_stream1.size());
    for (size_t i = 0; i < task_on_stream1.size(); ++i) {
      EXPECT_EQ(task_on_stream1[i], expect_task_infos_on_stream1[i]);
    }
    model_executor.reset(nullptr);
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}

/*
 *  data1  data2
 *    \   /
 *     add1(streamid=0)(send_id_list:[0])
 *      |
 *     relu (streamid=1)(send_id_list:[1],recive_id_list:[0])
 *      |
 *    netoutput(streamid=0)(recive_id_list:[1]))
 *
 * 1. 用例描述：外置allocator场景，测试两个级联节点分别在两条流上的执行，每个step执行结束后L2将空闲内存归还给L1
 * 2. 预置条件：
 *   （1）外部创建主Stream
 *   （2）外部创建allocator
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) 每个step执行结束后，l1 allocator的内存占用大小与初始大小一样
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case09_TwoStream_ExternalAllocator_RecycleToL1_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_AccessMemCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  auto external_allocator = CreateDefaultAllocators();
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto inputs = FakeTensors({2048}, 2);
    auto outputs = FakeTensors({2048}, 1);
    auto l1_device_allocator = reinterpret_cast<memory::CachingMemAllocator *>(
        external_allocator->GetAllocator(kOnDeviceHbm, static_cast<size_t>(AllocatorUsage::kAllocNodeWorkspace)));
    auto origin_occupied_size = l1_device_allocator->GetScalableAllocator()->GetOccupiedMemSize();

    ASSERT_EQ(model_executor->Execute({i3.value, external_allocator.get()}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    EXPECT_TRUE(l1_device_allocator->GetScalableAllocator()->GetOccupiedMemSize() == origin_occupied_size);

    ASSERT_EQ(model_executor->Execute({i3.value, external_allocator.get()}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    EXPECT_TRUE(l1_device_allocator->GetScalableAllocator()->GetOccupiedMemSize() == origin_occupied_size);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    model_executor.reset(nullptr);
    rtStreamDestroy(stream);
  }
  runtime_stub.Clear();
}

/*
 *  data1  data2
 *    \   /
 *     add1(streamid=0)(send_id_list:[0])
 *      |
 *     relu (streamid=1)(send_id_list:[1],recive_id_list:[0])
 *      |
 *    netoutput(streamid=0)(recive_id_list:[1]))
 *
 * 1. 用例描述：外置allocator场景，测试两个级联节点分别在两条流上的执行，且外部没有分配输出内存，执行器析构后输出tensor能够正常释放
 * 2. 预置条件：
 *   （1）外部创建主Stream
 *   （2）外部创建allocator
 * 3. 测试步骤：
     （1）通过计算图构造RootModel
     （2）model转 exe graph
     （3）构造ModelV2Executor，构造输入输出Tensor，执行
 * 4. 预期结果
     （1) 执行器析构后输出Tensor能够正常访问和释放
 */
TEST_F(GraphExecutorMultiStreamSystemTest, Case10_TwoStream_ExternalAllocator_InnerAllocOutMem_ok) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName))
      .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_AccessMemCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  auto external_allocator = CreateDefaultAllocators();
  Tensor output_tensor;
  std::vector<Tensor *> outputs = {&output_tensor};
  auto inputs = FakeTensors({2048}, 2);
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    ASSERT_EQ(model_executor->Execute({i3.value, external_allocator.get()}, inputs.GetTensorList(), inputs.size(),
                                      outputs.data(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    model_executor.reset(nullptr);
    rtStreamDestroy(stream);
  }
  ASSERT_NE(output_tensor.GetAddr(), nullptr);
  ASSERT_EQ(output_tensor.MutableTensorData().Free(), ge::GRAPH_SUCCESS);
  runtime_stub.Clear();
}

// MultiStreamL2Allocator构造函数覆盖
TEST_F(GraphExecutorMultiStreamSystemTest, MultiStreamL2Allocator_Constructor_Cover) {
  auto allocator = new memory::MultiStreamL2Allocator(nullptr);
  ASSERT_NE(allocator, nullptr);
  delete allocator;
}

TEST_F(GraphExecutorMultiStreamSystemTest, Case11_rtMalloc_Failed) {
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker(AddStubName))
                           .AddTaskDef("Relu", AiCoreTaskDefFaker(ReluStubName))
                           .SetRootModelStreamNum(stream_num)
                           .SetRootModelEventNum(event_num)
                           .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "MultiStreamST_AccessMemCrossStream");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  class MockRuntime : public RuntimeStubImpl {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      return -1;
    }
  };

  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::Install(mock_runtime.get());
  {
    auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);

    ASSERT_NE(model_executor, nullptr);
    ExecutorTracerOn executor_tracer_on;  // 开启trace
    (void)StartExecutorStatistician(model_executor);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreateWithFlags(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), 0), RT_ERROR_NONE);
    auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));


    RtSession rt_session;
    EXPECT_EQ(model_executor->Load({i3.value}, {&rt_session}), ge::GRAPH_SUCCESS);

    auto outputs = FakeTensors({2048}, 1);
    auto inputs = FakeTensors({2048}, 2);

    ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                      outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  }
}
// todo multi stream profiling case
}  // namespace gert
