/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/model_v2_executor.h"
#include "lowering/graph_converter.h"
#include "lowering/model_converter.h"
#include <gtest/gtest.h>

#include "common/share_graph.h"
#include "stub/gert_runtime_stub.h"
#include "faker/global_data_faker.h"
#include "faker/fake_value.h"
#include "graph/utils/graph_dump_utils.h"

#include "op_impl/dynamicatomicaddrclean/dynamic_atomic_addr_clean_impl.h"
#include "op_impl/transdata/trans_data_positive_source_tc_1010.h"
#include "op_impl/dynamic_rnn_impl.h"
namespace gert {
namespace {
const char *const TransDataStubName = "TransDataStubBin";
const char *const TransData13StubName = "TransData17StubBin";
const char *const DynamicAtomicStubName = "DynamicAtomicBin";
const char *const DynamicRnnv3StubName = "DynamicRNNV3StubBin";
const char *const AddStubName = "AddStubBin";
const char *const MulStubName = "MulStubBin";
}  // namespace
class NetworkLstmpST : public testing::Test {
 public:
  ge::ExecuteGraphPtr GenerateLstmpExeGraph() {
    auto graph = ShareGraph::LstmpGraph();
    graph->TopologicalSorting();
    GE_DUMP(graph, "LstmpST_ComputeGraph");
    GeModelBuilder builder(graph);
    ge_root_model_ =
        builder
            .AddTaskDef("TransData",
                        AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName).WithHandle())
            .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName))
            .BuildGeRootModel();

    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model_);
    GE_ASSERT_NOTNULL(exe_graph);
    ge::DumpGraph(exe_graph.get(), "LstmpST_ExecuteGraph1");
    return exe_graph;
  }
  ge::ExecuteGraphPtr GenerateLstmpExeGraph2() {
    auto graph = ShareGraph::LstmpGraph();
    graph->TopologicalSorting();
    GE_DUMP(graph, "LstmpST_ComputeGraph");
    GeModelBuilder builder(graph);
    ge_root_model2_ =
        builder.AddTaskDef("transdata_13", AiCoreTaskDefFaker(TransData13StubName).AtomicStubNum(DynamicAtomicStubName))
            .AddTaskDef("TransData", AiCoreTaskDefFaker(TransDataStubName).AtomicStubNum(DynamicAtomicStubName))
            .AddTaskDef("DynamicRNNV3", AiCoreTaskDefFaker(DynamicRnnv3StubName).AtomicStubNum(DynamicAtomicStubName))
            .BuildGeRootModel();

    auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model2_);
    GE_ASSERT_NOTNULL(exe_graph);
    ge::DumpGraph(exe_graph.get(), "LstmpST_ExecuteGraph2");
    return exe_graph;
  }

 protected:
  //GeModel must not be destructed, it's weight data will be use in Init Graph when sink weight.
  ge::GeRootModelPtr ge_root_model_;
  ge::GeRootModelPtr ge_root_model2_;
};
TEST_F(NetworkLstmpST, Lstmp_LoadUseDefaultStream) {
  auto exe_graph = GenerateLstmpExeGraph();
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model_);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  auto &records = stub.GetAclRuntimeStub().GetRtMemcpyRecords();
  ASSERT_EQ(records.size(), 3);
  for (const auto &rec : records) {
    ASSERT_EQ(rec.copy_type, ge::MemoryCopyType::kRtMemcpyAsync);
    ASSERT_EQ(rec.stream, nullptr);
  }
}
TEST_F(NetworkLstmpST, Lstmp_LoadUseParameterStream) {
  auto exe_graph = GenerateLstmpExeGraph();
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model_);
  ASSERT_NE(model_executor, nullptr);
  ModelExecuteArg arg = {(void *)1024};
  EXPECT_EQ(model_executor->Load(arg), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  auto &records = stub.GetAclRuntimeStub().GetRtMemcpyRecords();
  ASSERT_EQ(records.size(), 3);
  for (const auto &rec : records) {
    ASSERT_EQ(rec.copy_type, ge::MemoryCopyType::kRtMemcpyAsync);
    ASSERT_EQ(rec.stream, (void *)1024);
  }
}
TEST_F(NetworkLstmpST, Lstmp_LaunchArgCorrect) {
  auto exe_graph = GenerateLstmpExeGraph2();
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model2_);
  ASSERT_NE(model_executor, nullptr);

  GertRuntimeStub runtime_stub;

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({2048}, 3);
  auto inputs = FakeTensors({2048}, 3);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  // CheckRtsLaunchParas
  auto dynmic_automic_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(DynamicAtomicStubName);// 不需要修改么？
  ASSERT_NE(dynmic_automic_launch_args, nullptr);
  ASSERT_EQ(dynmic_automic_launch_args->GetStream(), stream);
  ASSERT_EQ(dynmic_automic_launch_args->GetArgsEx()->argsSize, 88);
  ASSERT_EQ(dynmic_automic_launch_args->GetArgsEx()->tilingAddrOffset, 8);
  ASSERT_EQ(dynmic_automic_launch_args->GetArgsEx()->tilingDataOffset, 32);
  auto automic_tiling = dynmic_automic_launch_args->GetArgsTilingData<DynamicAtomicAddrCleanParam>();
  ASSERT_EQ(automic_tiling->need_core_num_input_scalar, 8);

  auto transdata_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(TransDataStubName);
  ASSERT_NE(transdata_launch_args, nullptr);
  ASSERT_EQ(transdata_launch_args->GetStream(), stream);
  ASSERT_EQ(transdata_launch_args->GetArgsEx()->argsSize, 352);
  ASSERT_EQ(transdata_launch_args->GetArgsEx()->tilingAddrOffset, 24);
  ASSERT_EQ(transdata_launch_args->GetArgsEx()->tilingDataOffset, 32);
  auto transdata_tiling = transdata_launch_args->GetArgsTilingData<kernel::transdata::TransDataMode1010Param>();
  ASSERT_EQ(transdata_tiling->tiling_mode, 1010);

  auto transdata13_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(TransData13StubName);
  ASSERT_NE(transdata13_launch_args, nullptr);
  ASSERT_EQ(transdata13_launch_args->GetStream(), stream);
  ASSERT_EQ(transdata13_launch_args->GetArgsEx()->argsSize, 408);
  ASSERT_EQ(transdata13_launch_args->GetArgsEx()->tilingAddrOffset, 24);
  ASSERT_EQ(transdata13_launch_args->GetArgsEx()->tilingDataOffset, 32);
  auto transdata13_tiling = transdata_launch_args->GetArgsTilingData<kernel::transdata::TransDataMode1010Param>();
  ASSERT_EQ(transdata13_tiling->tiling_mode, 1010);

  auto dynamic_launch_args = runtime_stub.GetRtsRuntimeStub().PopLaunchArgsByStubName(DynamicRnnv3StubName);
  ASSERT_NE(dynamic_launch_args, nullptr);
  EXPECT_EQ(dynamic_launch_args->GetStream(), stream);
  EXPECT_EQ(dynamic_launch_args->GetArgsEx()->argsSize, 268);
  EXPECT_EQ(dynamic_launch_args->GetArgsEx()->tilingAddrOffset, 240);
  EXPECT_EQ(dynamic_launch_args->GetArgsEx()->tilingDataOffset, 248);

  auto dynamic_tiling = dynamic_launch_args->GetArgsTilingData<DynamicRnnV3Param>();
  ASSERT_EQ(dynamic_tiling->sequenceLength, 8);
  runtime_stub.Clear();
  rtStreamDestroy(stream);
}
TEST_F(NetworkLstmpST, Lstmp_LaunchTilingArgsMatches) {
  auto exe_graph = GenerateLstmpExeGraph();
  ASSERT_NE(exe_graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model_);
  ASSERT_NE(model_executor, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2048}, 3);
  auto inputs = FakeTensors({2048}, 3);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(stub.CheckLaunchWhenStubTiling());
  stub.Clear();
  rtStreamDestroy(stream);
}
}  // namespace gert