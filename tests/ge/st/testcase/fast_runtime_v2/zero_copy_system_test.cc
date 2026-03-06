/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/gert_api.h"
#include <gtest/gtest.h>

#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/fake_value.h"
#include "faker/aicpu_taskdef_faker.h"
#include "stub/gert_runtime_stub.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/attr_utils.h"
#include "lowering/model_converter.h"
namespace gert {
class ZeroCopyST : public testing::Test {};
/**
 * 用例描述：用户提供输出，零拷贝使能
 *
 * 预置条件：
 * 1. fake Add算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 构造输入Tensor，shape为[1,2,3,4]，输出Tensor的shape为[1024]，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址未改变
 * 3. 输出Tensor shape为[1,2,3,4]
 * 4. 通过launch参数检查，Launch的输出地址为传入的输出Tensor地址
 */
TEST_F(ZeroCopyST, ZeroCopy_Enabled_WhenOuptutGiven) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args.size(), 1U);
  ASSERT_EQ(launch_args.begin()->second.size(), 1U);
  auto addresses = (*(launch_args.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_EQ(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args1 = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args1.size(), 1U);
  ASSERT_EQ(launch_args1.begin()->second.size(), 1U);
  addresses = (*(launch_args1.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_EQ(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：用户提供输出，零拷贝使能
 *
 * 预置条件：
 * 1. fake Add算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造带If算子的计算图
 * 2. lowering、加载计算图
 * 3. 构造输入Tensor，shape为[8,3,224,224]，输出Tensor的shape为[8,3,224,224]，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址未改变
 * 3. 输出Tensor shape为[8,3,224,224]
 * 4. 通过launch参数检查，Launch的输出地址为传入的输出Tensor地址
 */
TEST_F(ZeroCopyST, ZeroCopy_If) {
  auto graph = ShareGraph::IfGraph5();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Cast", AiCoreTaskDefFaker("CastStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args.size(), 1U);
  ASSERT_EQ(launch_args.begin()->second.size(), 1U);
  auto addresses = (*(launch_args.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_EQ(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args1 = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args1.size(), 1U);
  ASSERT_EQ(launch_args1.begin()->second.size(), 1U);
  addresses = (*(launch_args1.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_EQ(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：用户提供输出，零拷贝使能
 *
 * 预置条件：
 * 1. fake Add算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 构造输入Tensor，shape为[1,2,3,4]，输出Tensor的shape为[1024]，输出地址为host地址, 执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址未改变，但是存在数据拷贝
 * 3. 输出Tensor shape为[1,2,3,4]
 * 4. 通过launch参数检查，Launch的输出地址不是传入的输出Tensor地址
 */
TEST_F(ZeroCopyST, ZeroCopy_Disable_WhenOuptutGivenButPlacementNotMatch) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnHost); // 外部给的地址placement与算子需要的placement不一样
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args.size(), 1U);
  ASSERT_EQ(launch_args.begin()->second.size(), 1U);
  auto addresses = (*(launch_args.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_NE(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_FALSE(runtime_stub.GetAclRuntimeStub().GetRtMemcpyRecords().empty());

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args1 = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args1.size(), 1U);
  ASSERT_EQ(launch_args1.begin()->second.size(), 1U);
  addresses = (*(launch_args1.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_NE(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_FALSE(runtime_stub.GetAclRuntimeStub().GetRtMemcpyRecords().empty());

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：用户未提供输出，模型中申请的内存传出
 *
 * 预置条件：
 * 1. fake Add算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. lowering、加载计算图
 * 3. 构造输入Tensor，shape为[8 * 3 * 224 * 224]，输出Tensor中不申请内存，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址不为空
 * 3. 输出Tensor shape为[8 * 3 * 224 * 224]
 */
TEST_F(ZeroCopyST, ZeroCopy_Disable_WhenOuptutNotGiven) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().DontAllocData().Build();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_NE(output.GetTensor()->GetAddr(), nullptr);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs().size(), 1U);
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size(), 0U);

  runtime_stub.Clear();
  output.GetTensor()->MutableTensorData().SetAddr(nullptr, nullptr);
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_NE(output.GetTensor()->GetAddr(), nullptr);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs().size(), 1U);
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size(), 0U);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
/**
 * 用例描述：用户提供输出，但是模型输出不需要申请内存，模型输出拷贝到用户提供的输出地址
 *
 * 预置条件：
 * 1. fake Reshape算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Reshape的计算图
 * 2. lowering、加载计算图
 * 3. 构造输入Tensor，shape为[1,2,3,4]，输出Tensor中申请内存，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址不为空
 * 3. 输出Tensor shape为[1,2,3,4]
 * 4. 观察rts的memcpy接口，有拷贝到输出的动作
 */
//TEST_F(ZeroCopyST, ZeroCopy_CopyOut_LastNodeDoesNotAllocMemory) {}
/**
 * 用例描述：输入节点通过reshape后连接到输出，用户提供的输入、输出地址相同，零拷贝生效，仅刷新输出shape
 *
 * 预置条件：
 * 1. fake Reshape算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Reshape的计算图
 * 2. lowering、加载计算图
 * 3. 构造Tensor shape为[1,2,3,4]，同时作为输入和输出，执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址不为空
 * 3. 输出Tensor shape为[4,3,2,1]
 * 4. 观察rts的memcpy接口，未发生调用
 */
//TEST_F(ZeroCopyST, ZeroCopy_OnlyUpdateShape_InToOut) {}
/**
 * 用例描述：always-zero-copy功能使能后，网络功能正常
 *
 * 预置条件：
 * 1. fake Add算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. 打开always_zero_copy选项，加载计算图
 * 3. 执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址未改变
 * 3. 输出Tensor shape正确
 * 4. 通过launch参数检查，Launch的输出地址为传入的输出Tensor地址
 */
TEST_F(ZeroCopyST, AlwaysZeroCopy_Enabled_WhenOuptutGiven) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();
  LoweringOption option{.trust_shape_on_out_tensor = false, .always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().Shape({8, 3, 224, 224}).Build();
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args.size(), 1U);
  ASSERT_EQ(launch_args.begin()->second.size(), 1U);
  auto addresses = (*(launch_args.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_EQ(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  auto &launch_args1 = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args1.size(), 1U);
  ASSERT_EQ(launch_args1.begin()->second.size(), 1U);
  addresses = (*(launch_args1.begin()->second.begin()))->GetLaunchAddresses();
  EXPECT_EQ(addresses[2], out_addr);  // 第0和第1个launch地址分别对应两个add输入
  ASSERT_TRUE(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().empty());

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：always-zero-copy功能使能后，4类aicpu算子网络功能正常
 *
 * 预置条件：
 * 1. fake Add算子的lowering、tiling等整套实现
 *
 * 测试步骤：
 * 1. 构造tf引起的单算子Add的计算图
 * 2. 打开always_zero_copy选项，加载计算图
 * 3. 执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址未改变
 * 3. 输出Tensor shape正确
 * 4. 通过launch参数检查，Launch的输出地址为传入的输出Tensor地址
 */
TEST_F(ZeroCopyST, AlwaysZeroCopy_Enabled_Unknown_Aicpu_WhenOuptutGiven) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op = graph->FindNode("add1")->GetOpDesc();
  add_op->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  ge::AttrUtils::SetInt(add_op, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  ge::AttrUtils::SetBool(add_op, "SmallShapeHostcpu", false);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).BuildGeRootModel();
  LoweringOption option{.trust_shape_on_out_tensor = false, .always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().Shape({8, 3, 224, 224}).Build();
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：always-zero-copy功能使能后，对无法生效零拷贝的网络，功能正常
 *
 * 预置条件：
 * 1. fake Reshape算子的整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Reshape的计算图
 * 2. 打开always_zero_copy选项，加载计算图
 * 3. 执行
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor地址未改变
 * 3. 输出Tensor shape正确
 */
TEST_F(ZeroCopyST, AlwaysZeroCopy_Success_WhenAllocNodeNotFound) {
  auto graph = ShareGraph::BuildReshapeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  LoweringOption option{.trust_shape_on_out_tensor = false, .always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EReshapeGraph");

  GertRuntimeStub runtime_stub;

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker()
                    .StorageShape({8, 1, 224, 224, 16})
                    .OriginShape({8, 3, 224, 224})
                    .DataType(ge::DT_INT16)
                    .OriginFormat(ge::FORMAT_NCHW)
                    .StorageFormat(ge::FORMAT_NC1HWC0)
                    .Build();
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto input = TensorFaker().Shape({1, 2, 3, 4}).Build();
  std::vector<Tensor *> inputs = {input.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({2, 12}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({2, 12}));
  EXPECT_EQ(output.GetTensor()->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output.GetTensor()->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output.GetTensor()->GetStorageFormat(), ge::FORMAT_ND);
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size(), 0U);

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.data(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_TRUE(out_addr == output.GetTensor()->GetAddr());
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({2, 12}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({2, 12}));
  EXPECT_EQ(output.GetTensor()->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output.GetTensor()->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output.GetTensor()->GetStorageFormat(), ge::FORMAT_ND);
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords().size(), 0U);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：always-zero-copy功能使能后，如果未传入输出Tensor的地址，执行失败
 *
 * 预置条件：
 * 1. fake Add算子的整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. 打开always_zero_copy选项，加载计算图
 * 3. 输出Tensor地址为空，执行
 *
 * 预期结果：
 * 1. 执行失败
 * 2. 校验错误码为ge::PARAM_INVALID
 */
TEST_F(ZeroCopyST, AlwaysZeroCopy_Failed_WhenOuptutNotGiven) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = false, .always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().DontAllocData().Build();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::PARAM_INVALID);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::PARAM_INVALID);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
/**
 * 用例描述：always-zero-copy功能使能后，如果输出Tensor的长度不足，执行失败
 *
 * 预置条件：
 * 1. fake Add算子的整套实现
 *
 * 测试步骤：
 * 1. 构造单算子Add的计算图
 * 2. 打开always_zero_copy选项，加载计算图
 * 3. 构造输入Tensor的shape为[1,2,3,4]，输出Tensor的shape为[1,2,3,3]，开始执行
 *
 * 预期结果：
 * 1. 执行失败
 */
// todo 待校验规则明确后放开
#if 0
TEST_F(ZeroCopyST, AlwaysZeroCopy_Failed_WhenOuptutSizeNotEnough) {
  auto graph =
      ShareGraph::BuildSingleNodeGraph("Add", {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
                                       {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
                                       {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(
    ge_root_model, {.trust_shape_on_out_tensor = false, .always_zero_copy = true});
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().Shape({1, 2, 3, 1}).Build();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({1, 2, 3, 4}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
#endif
}  // namespace gert
