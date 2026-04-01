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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/hybrid_model_rt_v1_executor.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_manager.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/ge_context.h"
#include "common/share_graph.h"
#include "register/node_converter_registry.h"
#include "op_impl/less_important_op_impl.h"
#include "faker/ge_model_builder.h"
#include "stub/gert_runtime_stub.h"
#include "faker/aicore_taskdef_faker.h" 
#include "register/op_impl_registry.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "common/env_path.h"

#include <graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h>

using namespace ge;
using namespace hybrid;
using namespace gert;
using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;
using namespace gert;
namespace {
class Listener : public ModelListener {
 public:
  Listener(std::function<void()> done) : done_(done) {}
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code,
                       std::vector<gert::Tensor> &outputs) override {
    done_();
    return SUCCESS;
  }
  std::function<void()> done_;
};

static ge::OpDescPtr CreateOpDesc(const std::string &name, const std::string &type) {
  OpDescPtr op_desc = std::make_shared<ge::OpDesc>(name, type);
  ge::GeTensorDesc ge_tensor_desc;
  op_desc->AddInputDesc("input", ge_tensor_desc);
  op_desc->AddOutputDesc("output", ge_tensor_desc);

  return op_desc;
}
std::vector<gert::Tensor> InputData2GertTensors(const InputData &input_data) {
  std::vector<gert::Tensor> inputs;
  for (size_t i = 0; i < input_data.blobs.size(); ++i) {
    gert::Tensor tensor;
    for (size_t j = 0U; j < input_data.shapes[i].size(); ++j) {
      tensor.MutableStorageShape().SetDim(0, input_data.shapes[i][j]);
    }
    tensor.MutableTensorData().SetAddr(input_data.blobs[i].data, nullptr);
    tensor.MutableTensorData().SetSize(input_data.blobs[i].length);
    tensor.MutableStorageShape().SetDimNum(input_data.shapes[i].size());
    inputs.emplace_back(std::move(tensor));
  }
  return inputs;
}
}  // namespace
class UtestHybridModelAsyncExecutor : public testing::Test {
 protected:
  void SetUp() {
    unsetenv("ENABLE_RUNTIME_V2");
  }

  void TearDown() {
  }
};

TEST_F(UtestHybridModelAsyncExecutor, Test_execute_by_runGraph_with_rtv1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  HybridModelAsyncExecutor executor(&hybrid_model);
  executor.SetDeviceId(0U);
  executor.SetModelId(1U);
  ASSERT_EQ(executor.Init(), SUCCESS);
  HybridModelRtV1Executor *executor_rt_v1 = reinterpret_cast<HybridModelRtV1Executor *>(executor.executor_.get());
  auto &context = executor_rt_v1->context_;
  HybridModelExecutor::ExecuteArgs args;
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  dynamic_cast<RtCallbackManager *>(context.callback_manager)->callback_queue_.Push(eof_entry);
  std::vector<GeTensor> inputs;
  const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
  inputs.push_back(ge_tensor);
  std::vector<GeTensor> outputs;
  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs);
  std::vector<gert::Tensor> gert_outputs;
  ASSERT_EQ(executor.Execute(gert_inputs, gert_outputs), SUCCESS);

  HybridModelAsyncExecutor executor2(&hybrid_model);
  executor2.SetDeviceId(0U);
  executor2.SetModelId(1U);
  ASSERT_EQ(executor2.Init(), SUCCESS);
  // ExecuteWithStreamAsync not support rt1 with dynamic model
  rtStream_t stream = (void*)0x01;
  ASSERT_NE(executor2.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, Test_execute_by_executeGraph_with_rtv1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  HybridModelAsyncExecutor executor(&hybrid_model);
  executor.SetDeviceId(0U);
  executor.SetModelId(1U);
  ASSERT_EQ(executor.Init(), SUCCESS);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(1);
  inputs.resize(2);
  // ExecuteWithStreamAsync not support rt1 with dynamic model
  rtStream_t stream = (void*)0x01;
  ASSERT_NE(executor.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, Test_execute_by_runGraph_with_rtv1_with_gert_tensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  HybridModelAsyncExecutor executor(&hybrid_model);
  executor.SetDeviceId(0U);
  executor.SetModelId(1U);
  ASSERT_EQ(executor.Init(), SUCCESS);
  HybridModelRtV1Executor *executor_rt_v1 = reinterpret_cast<HybridModelRtV1Executor *>(executor.executor_.get());
  auto &context = executor_rt_v1->context_;
  HybridModelExecutor::ExecuteArgs args;
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  dynamic_cast<RtCallbackManager *>(context.callback_manager)->callback_queue_.Push(eof_entry);
  std::vector<GeTensor> inputs;
  const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
  inputs.push_back(ge_tensor);
  std::vector<GeTensor> outputs;
  std::vector<gert::Tensor> gert_inputs_pro;
  TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs_pro);
  std::vector<gert::Tensor> gert_outputs_pro;
  ASSERT_EQ(executor.Execute(gert_inputs_pro, gert_outputs_pro), SUCCESS);

  HybridModelAsyncExecutor executor2(&hybrid_model);
  executor2.SetDeviceId(0U);
  executor2.SetModelId(1U);
  ASSERT_EQ(executor2.Init(), SUCCESS);
  // ExecuteWithStreamAsync not support rt1 with dynamic model
  rtStream_t stream = (void*)0x01;
  ASSERT_NE(executor2.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
  std::vector<gert::Tensor> gert_inputs;
  gert_inputs.resize(1);
  gert_inputs[0] = {{{-1, 16, 16, 3}, {-1, 16, 16, 3}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) tensor_data.data()};
  std::vector<gert::Tensor> gert_outputs;
  ASSERT_EQ(executor.Execute(gert_inputs, gert_outputs), SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, Test_execute_by_runGraph_with_rtv2) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  rtStream_t stream = (void*)0x01;
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);

  std::vector<GeTensor> inputs;
  const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
  ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
  inputs.push_back(ge_tensor);
  std::vector<GeTensor> outputs;
  std::vector<gert::Tensor> gert_inputs_pro;
  TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs_pro);
  std::vector<gert::Tensor> gert_outputs_pro;
  ASSERT_EQ(executor.Execute(gert_inputs_pro, gert_outputs_pro), SUCCESS);
  EXPECT_EQ(executor.Init(stream), SUCCESS);
  ASSERT_EQ(executor.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestHybridModelAsyncExecutor, Test_multiStream_execute_by_runGraph_with_rtv2) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  EXPECT_EQ(stream_num, 2);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Add" || node->GetType() == "Relu") {
      MockLessImportantNodeKernel(node);
    }
  }
  graph->TopologicalSorting();

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {
    GeModelBuilder builder(graph);
    auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
        .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubBin"))
        .SetRootModelStreamNum(stream_num)
        .SetRootModelEventNum(event_num)
        .BuildGeRootModel();

    HybridModel hybrid_model(ge_root_model);
    hybrid_model.root_graph_item_.reset(new GraphItem);
    hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
    EXPECT_EQ(hybrid_model.Init(), SUCCESS);
    EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

    HybridModelAsyncExecutor executor(&hybrid_model);
    rtStream_t stream = (void *)0x01;
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    EXPECT_NE(executor.executor_, nullptr);
    ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetAllRtStreams().size(), 1);  // require 1 sub stream when load

    std::vector<GeTensor> inputs;
    const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
    GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
    tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
    auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
    ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
    inputs.push_back(ge_tensor);
    inputs.push_back(ge_tensor);
    std::vector<GeTensor> outputs;
    std::vector<gert::Tensor> gert_inputs_pro;
    TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs_pro);
    std::vector<gert::Tensor> gert_outputs_pro;
    ASSERT_EQ(executor.Execute(gert_inputs_pro, gert_outputs_pro), SUCCESS);
    auto all_rt_streams = runtime_stub.GetRtsRuntimeStub().GetAllRtStreams();
    ASSERT_EQ(all_rt_streams.size(), stream_num - 1); // // total require 1 sub stream when executing
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    ASSERT_EQ(executor.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
  }
  unsetenv("ENABLE_RUNTIME_V2");
  runtime_stub.Clear();
}

TEST_F(UtestHybridModelAsyncExecutor, Test_multiStream_execute_by_runGraph_with_rtv2_rollback_singleStream) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  setenv("MOCK_AVAIL_STREAM_NUM", "1", 0); // only has 1 stream
  int64_t stream_num = 1;
  int64_t event_num = 0;
  auto graph = ShareGraph::MultiStreamTwoNodeGraph(stream_num, event_num);
  EXPECT_TRUE(stream_num > 1);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Add" || node->GetType() == "Relu") {
      MockLessImportantNodeKernel(node);
    }
  }

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin"))
      .AddTaskDef("Relu", AiCoreTaskDefFaker("ReluStubBin"))
      .SetRootModelStreamNum(stream_num)
      .SetRootModelEventNum(event_num)
      .BuildGeRootModel();

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();
  {

    HybridModel hybrid_model(ge_root_model);
    hybrid_model.root_graph_item_.reset(new GraphItem);
    hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
    EXPECT_EQ(hybrid_model.Init(), SUCCESS);
    EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

    HybridModelAsyncExecutor executor(&hybrid_model);
    rtStream_t stream = (void *)0x01;
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    EXPECT_NE(executor.executor_, nullptr);
    ASSERT_EQ(runtime_stub.GetAclRuntimeStub().GetAllRtStreams().size(), 0);  // require 0 sub stream when load

    std::vector<GeTensor> inputs;
    const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
    GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
    tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
    auto ge_tensor = GeTensor(*tensor_desc, tensor_data.data(), sizeof(uint8_t) * tensor_data.size());
    ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);
    inputs.push_back(ge_tensor);
    inputs.push_back(ge_tensor);
    std::vector<GeTensor> outputs;
    std::vector<gert::Tensor> gert_inputs_pro;
    TensorTransUtils::GeTensors2GertTensors(inputs, gert_inputs_pro);
    std::vector<gert::Tensor> gert_outputs_pro;
    ASSERT_EQ(executor.Execute(gert_inputs_pro, gert_outputs_pro), SUCCESS);
    EXPECT_EQ(executor.Init(stream), SUCCESS);
    auto all_rt_streams = runtime_stub.GetAclRuntimeStub().GetAllRtStreams();
    ASSERT_EQ(all_rt_streams.size(), 0); // execute on 1 streams, use external stream, no need create streams
    ASSERT_EQ(executor.ExecuteWithStreamAsync(inputs, outputs, stream), SUCCESS);
  }
  unsetenv("ENABLE_RUNTIME_V2");
  unsetenv("MOCK_AVAIL_STREAM_NUM");
  runtime_stub.Clear();
}

TEST_F(UtestHybridModelAsyncExecutor, Test_execute_by_loadModelWithQueue_with_rtv1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  HybridModelAsyncExecutor executor(&hybrid_model);
  executor.SetDeviceId(0U);
  executor.SetModelId(1U);
  ASSERT_EQ(executor.Init(), SUCCESS);
  HybridModelRtV1Executor *executor_rt_v1 = reinterpret_cast<HybridModelRtV1Executor *>(executor.executor_.get());
  auto &context = executor_rt_v1->context_;
  HybridModelExecutor::ExecuteArgs args;
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  dynamic_cast<RtCallbackManager *>(context.callback_manager)->callback_queue_.Push(eof_entry);
  context.dump_properties.is_train_op_debug_ = true;
  std::vector<DataBuffer> inputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> output_desc;

  const std::vector<uint8_t> tensor_data{1, 212, 32, 32};
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  DataBuffer buffer;
  buffer.data = const_cast<void *>(reinterpret_cast<const void *>(tensor_data.data()));
  buffer.length = sizeof(uint8_t) * tensor_data.size();
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  input_desc.push_back(*tensor_desc);
  inputs.emplace_back(buffer);
  ASSERT_EQ(executor.Execute(inputs, input_desc, outputs, output_desc), SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, Test_execute_by_loadModelWithQueue_with_rtv2) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  GertRuntimeStub runtime_stub;
  HybridModelAsyncExecutor executor(&hybrid_model);
  EXPECT_EQ(executor.Init(), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);
  std::vector<DataBuffer> inputs;
  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> outputs;
  std::vector<GeTensorDesc> output_desc;

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  DataBuffer buffer(data_buf.get(), 3072, false, static_cast<uint32_t>(Placement::kPlacementDevice));
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  input_desc.push_back(*tensor_desc);
  inputs.emplace_back(buffer);
  runtime_stub.Clear();
  ASSERT_EQ(executor.Execute(inputs, input_desc, outputs, output_desc), SUCCESS);
  ASSERT_TRUE(runtime_stub.GetAclRuntimeStub().GetRtMemcpyRecords().empty());
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestHybridModelAsyncExecutor, Test_online_execute_with_blockingOp) {
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  HybridModelAsyncExecutor executor(&hybrid_model);

  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  ASSERT_EQ(executor.Init(), SUCCESS);
  bool reached = false;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>([&reached]() { reached = true; });
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  InputData input_data;
  input_data.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  input_data.shapes.push_back({1, 16, 16, 3});

  auto data = std::make_shared<RunArgs>();
  data->input_tensor = std::move(InputData2GertTensors(input_data));
  ASSERT_EQ(executor.Stop(), SUCCESS);
  ASSERT_EQ(executor.Init(), SUCCESS);
  ASSERT_EQ(executor.Start(listener), SUCCESS);
  ASSERT_EQ(executor.EnqueueData(data), SUCCESS);
  size_t kMaxWaitSeconds = 5U;
  for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
    if (reached) {
      break;
    }
    sleep(1);
  }
  ASSERT_EQ(executor.Stop(), SUCCESS);
  ASSERT_EQ(executor.Init(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestHybridModelAsyncExecutor, Test_online_execute_rtv1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  HybridModelAsyncExecutor executor(&hybrid_model);
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  bool reached = false;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>([&reached]() { reached = true; });
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  InputData input_data;
  input_data.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  input_data.shapes.push_back({1, 16, 16, 3});
  auto data = std::make_shared<RunArgs>();
  data->input_tensor = std::move(InputData2GertTensors(input_data));
  domi::GetContext().is_online_model = true;
  ASSERT_EQ(executor.Init(), SUCCESS);
  EXPECT_NE(executor.GeContext(), nullptr);
  ASSERT_EQ(executor.Start(listener), SUCCESS);
  ASSERT_EQ(executor.EnqueueData(data), SUCCESS);
  size_t kMaxWaitSeconds = 5U;
  for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
    if (reached) {
      break;
    }
    sleep(1);
  }
  ASSERT_EQ(executor.Stop(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestHybridModelAsyncExecutor, Test_online_execute_rtv2) {
  setenv("ENABLE_RUNTIME_V2", "1", 0);
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  HybridModelAsyncExecutor executor(&hybrid_model);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(executor.Init(), SUCCESS);
  EXPECT_NE(executor.executor_, nullptr);

  bool reached = false;
  std::shared_ptr<ModelListener> listener = std::make_shared<Listener>([&reached]() { reached = true; });

  executor.Start(listener);

  InputData inputs;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  inputs.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  inputs.shapes.push_back({1, 16, 16, 3});
  auto wrapper = std::make_shared<RunArgs>();
  wrapper->input_tensor = std::move(InputData2GertTensors(inputs));
  executor.EnqueueData(wrapper);

  size_t kMaxWaitSeconds = 5U;
  for (size_t seconds_wait = 0U; seconds_wait < kMaxWaitSeconds; seconds_wait++) {
    if (reached) {
      break;
    }
    sleep(1);
  }
  executor.Stop();
  unsetenv("ENABLE_RUNTIME_V2");
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestHybridModelAsyncExecutor, Test_MaxGraphParallelModelNum_rt2_use_new_stream) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  domi::GetContext().is_online_model = true;

  HybridModelAsyncExecutor executor1(&hybrid_model);
  EXPECT_EQ(executor1.Init(), SUCCESS);
  EXPECT_NE(executor1.executor_, nullptr);
  EXPECT_NE(executor1.stream_, nullptr);
  EXPECT_FALSE(executor1.owner_stream_);

  std::map<std::string, std::string> options;
  options["ge.graphMaxParallelModelNum"] = "8";
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModelAsyncExecutor executor2(&hybrid_model);
  EXPECT_EQ(executor2.Init(), SUCCESS);
  EXPECT_NE(executor2.executor_, nullptr);
  EXPECT_NE(executor2.stream_, nullptr);

  EXPECT_NE(executor1.stream_, executor2.stream_);
  EXPECT_TRUE(executor2.owner_stream_);
  domi::GetContext().is_online_model = false;
  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridModelAsyncExecutor, Test_AbnormalMaxGraphParallelModelNum_failed) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }
  auto netoutput = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  netoutput->GetOpDesc()->SetSrcName({"foo"});
  netoutput->GetOpDesc()->SetSrcIndex({0});
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);
  EXPECT_TRUE(hybrid_model.execute_by_rt_v2_);

  domi::GetContext().is_online_model = true;

  std::map<std::string, std::string> options;
  options["ge.graphMaxParallelModelNum"] = "aa";
  ge::GetThreadLocalContext().SetGraphOption(options);

  HybridModelAsyncExecutor executor1(&hybrid_model);
  EXPECT_NE(executor1.Init(), SUCCESS);
  
  domi::GetContext().is_online_model = false;
  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestHybridModelAsyncExecutor, TestExecutor_Ok_TwoModelUseDifferentStream) {
  auto graph = ShareGraph::SimpleFooGraph();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == "Foo") {
      MockLessImportantNodeKernel(node);
    }
  }

  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  HybridModel hybrid_model(ge_root_model);
  EXPECT_EQ(hybrid_model.Init(), SUCCESS);

  domi::GetContext().is_online_model = true;

  HybridModelAsyncExecutor executor(&hybrid_model);
  executor.SetModelId(0);
  EXPECT_EQ(executor.Init(), SUCCESS);

  HybridModelAsyncExecutor executor_1(&hybrid_model);
  executor_1.SetModelId(1);
  EXPECT_EQ(executor_1.Init(), SUCCESS);
  // 为model_0和model_1分别创建默认流
  EXPECT_TRUE(HybridModelAsyncExecutor::default_stream_by_dev_.count(std::make_pair(0, 0)) > 0);
  EXPECT_TRUE(HybridModelAsyncExecutor::default_stream_by_dev_.count(std::make_pair(0, 1)) > 0);

  domi::GetContext().is_online_model = false;
}
}  // namespace ge
