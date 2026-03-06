/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/mem_conflict_share_graph.h"
#include "common/memory/tensor_trans_utils.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/hybrid_model_rt_v1_executor.h"

#include "hybrid/model/hybrid_model_builder.h"
#include "graph/utils/graph_utils.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "depends/runtime/src/runtime_stub.h"
#include "stub/gert_runtime_stub.h"

using namespace std;
using namespace testing;

namespace ge {
namespace {
size_t AlignSize(size_t data_size) {
  return (data_size + 32 - 1) / 32 * 32;
}

std::unique_ptr<uint8_t []> CreateBuffer(size_t size) {
  std::unique_ptr<uint8_t []> buffer(new uint8_t[size]);
  int32_t *data = reinterpret_cast<int32_t *>(buffer.get());
  for (int32_t i = 0; i < static_cast<int32_t>(size / sizeof(int32_t)); i++) {
    data[i] = i;
  }
  return buffer;
}

bool CheckData(const void *buffer, size_t size) {
  const int32_t *data = reinterpret_cast<const int32_t *>(buffer);
  for (int32_t i = 0; i < static_cast<int32_t>(size / sizeof(int32_t)); i++) {
    if (data[i] != i) {
      std::cerr << "expect value: " << i << " actual value: " << data[i] << std::endl;
      return false;
    }
  }
  return true;
}

gert::Tensor CreateGertTensor(const std::initializer_list<int64_t> &dims, Format format, DataType type,
  void *buffer, size_t size) {
  auto tensor = gert::Tensor({dims, dims},
                 {format, format, {}},
                 gert::kOnDeviceHbm, type, buffer);
  tensor.SetSize(size);
  return tensor;
}
}
using namespace hybrid;
class MockMalloc : public RuntimeStub {
 public:
  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
    malloc_flag += 1;
    *dev_ptr = new uint8_t[size];
    memset_s(*dev_ptr, size, 0, size);

    return RT_ERROR_NONE;
  }

  rtError_t rtFree(void *dev_ptr) override {
    malloc_flag -= 1;
    delete[](uint8_t *) dev_ptr;
    return RT_ERROR_NONE;
  }

  int64_t malloc_flag = 0;
};

class MockAclMalloc : public AclRuntimeStub {
 public:
  aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) override {
    malloc_flag += 1;
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);

    return ACL_ERROR_NONE;
  }

  aclError aclrtFree(void *devPtr) override {
    malloc_flag -= 1;
    delete[](uint8_t *) devPtr;
    return ACL_ERROR_NONE;
  }

  int64_t malloc_flag = 0;
};

class UtestHybridRt1Executor : public testing::Test {
 protected:
  void SetUp() {
    unsetenv("ENABLE_RUNTIME_V2");
  }

  void TearDown() { }
};

TEST_F(UtestHybridRt1Executor, Test_execute_for_singleop) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeTensorDesc());
  op_desc->AddInputDesc(*tensor_desc);
  op_desc->AddOutputDesc(*tensor_desc);

  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.has_observer_ = true;

  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);
  ASSERT_EQ(executor.Init(), SUCCESS);
  HybridModelExecutor::ExecuteArgs args;
  args.input_desc.push_back(tensor_desc);
  TensorValue tensor;
  args.inputs.push_back(tensor);
  ASSERT_EQ(executor.ExecuteForSingleOp(args), SUCCESS);
  ASSERT_EQ(executor.Cleanup(), SUCCESS);
}

TEST_F(UtestHybridRt1Executor, Test_execute_for_dump_op_debug) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeTensorDesc());
  op_desc->AddInputDesc(*tensor_desc);
  op_desc->AddOutputDesc(*tensor_desc);

  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.has_observer_ = true;

  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);
  executor.context_.dump_properties.is_train_op_debug_ = true;
  ASSERT_EQ(executor.DumpOpDebug(), SUCCESS);
  executor.Stop();
  executor.context_.dump_properties.is_infer_op_debug_ = true;
  ASSERT_EQ(executor.DumpOpDebug(), SUCCESS);
  executor.Stop();
}

TEST_F(UtestHybridRt1Executor, Test_prepare_data_fail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  const auto data1 = CreateNode(*graph, "data1", "Data", 1, 1);
  const auto data2 = CreateNode(*graph, "data2", "Data", 1, 1);
  const auto add1 = CreateNode(*graph, "add", "Add", 2, 1);
  const auto output = CreateNode(*graph, "net_output", "Netoutput", 1, 1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), output->GetInDataAnchor(0));
  std::unique_ptr<NodeItem> data2_item;
  ASSERT_EQ(NodeItem::Create(data2, data2_item), SUCCESS);
  ASSERT_NE(data2_item, nullptr);
  data2_item->has_observer = true;
  std::unique_ptr<NodeItem> add_item;
  ASSERT_EQ(NodeItem::Create(add1, add_item), SUCCESS);
  ASSERT_NE(add_item, nullptr);
  data2_item->input_start = 0;
  data2_item->output_start = 0;
  add_item->input_start = 0;
  add_item->output_start = 0;

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ASSERT_NE(ge_root_model, nullptr);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.root_graph_item_->input_nodes_.emplace_back(data2_item.get());
  hybrid_model.has_observer_ = true;

  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);
  EXPECT_EQ(executor.Init(), SUCCESS);
  InputData input_data;
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  input_data.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  input_data.shapes.push_back({1, 16, 16, 3});
  HybridModelExecutor::ExecuteArgs args;
  EXPECT_EQ(executor.PrepareExecuteArgs(input_data, args), PARAM_INVALID);
}

TEST_F(UtestHybridRt1Executor, Test_prepare_dynamic_input_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  op_desc->AddInputDesc(*tensor_desc);
  op_desc->AddOutputDesc(*tensor_desc);

  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.has_observer_ = true;
  hybrid_model.SetNodeBinMode(fuzz_compile::kOneNodeSingleBinMode);

  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  HybridModelExecutor::ExecuteArgs args;
  args.input_desc.resize(1);
  const GeShape shape({1, 16, 16, 3});
  unique_ptr<uint8_t[]> data_buf(new (std::nothrow) uint8_t[3072]);
  tensor_desc->SetShapeRange({{1, 10}, {16, 16}, {16, 16}, {3, 3}});
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_INT32);
  executor.index_to_tensor_desc_[0] = tensor_desc;
  int64_t tensor_size = 0L;
  EXPECT_EQ(executor.PrepareDynamicInput(args, 0, shape,
            DataBuffer(data_buf.get(), 3072, false), tensor_size), SUCCESS);
  const GeShape shape1({11, 16, 16, 3});
  EXPECT_EQ(executor.PrepareDynamicInput(args, 0, shape1,
            DataBuffer(data_buf.get(), 3072, false), tensor_size), PARAM_INVALID);
}

TEST_F(UtestHybridRt1Executor, Test_copy_input_success) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  op_desc->AddInputDesc(*tensor_desc);
  op_desc->AddOutputDesc(*tensor_desc);

  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);
  hybrid_model.has_observer_ = true;
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  unique_ptr<uint8_t[]> data_buf_data(new (std::nothrow) uint8_t[3072]);
  DataBuffer data_buff(data_buf_data.get(), 3072, false, 1);
  HybridModelExecutor::ExecuteArgs args;
  args.input_desc.resize(1);
  EXPECT_EQ(executor.CopyDataToExecutArgs(3072, args, 0, data_buff), SUCCESS);
  DataBuffer data_buff1(data_buf_data.get(), 3072, false, 0);
  EXPECT_EQ(executor.CopyDataToExecutArgs(3072, args, 0, data_buff1), SUCCESS);
}

TEST_F(UtestHybridRt1Executor, handle_result_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  TensorValue input_tensor;
  HybridModelExecutor::ExecuteArgs args;
  args.inputs.emplace_back(input_tensor);
  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2, 2, 2, 2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);

  OutputData output_data;
  std::vector<ge::Tensor> outputs;
  auto ret = executor.HandleResult(1000, 1, args, &output_data, nullptr);
  ASSERT_EQ(ret, INTERNAL_ERROR);
  args.ctrl_args.is_eos = true;
  ret = executor.HandleResult(SUCCESS, 1, args, &output_data, nullptr);
  ASSERT_EQ(ret, END_OF_SEQUENCE);
  args.ctrl_args.is_eos = false;
  ret = executor.HandleResult(SUCCESS, 1, args, &output_data, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestHybridRt1Executor, ExecuteWithGertTensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  std::vector<gert::Tensor> inputs(2U);
  std::vector<gert::Tensor> outputs;

  const std::initializer_list<int64_t> dims = {1, 16, 16, 3};
  GeShape shape(dims);
  Format format(ge::FORMAT_ND);
  DataType data_type(ge::DT_INT32);
  int64_t data_size;
  ASSERT_EQ(TensorUtils::CalcTensorMemSize(shape, format, data_type, data_size), SUCCESS);
  const auto aligned_size = AlignSize(data_size);

  unique_ptr<uint8_t[]> data_buf1 = CreateBuffer(aligned_size);
  inputs[0] = CreateGertTensor(dims, format, data_type, data_buf1.get(), aligned_size);
  unique_ptr<uint8_t[]> data_buf2 = CreateBuffer(aligned_size);
  inputs[1] = CreateGertTensor(dims, format, data_type, data_buf2.get(), aligned_size);

  HybridModelExecutor::CtrlArgs ctrl_args;
  auto ret = executor.Execute(inputs, outputs, ctrl_args);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestHybridRt1Executor, CopyOutputs_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  TensorValue input_tensor;
  HybridModelExecutor::ExecuteArgs args;
  args.inputs.emplace_back(input_tensor);
  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2, 2, 2, 2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);

  OutputData output_data;
  std::vector<ge::Tensor> outputs;
  auto ret = executor.CopyOutputs(args, &output_data, outputs);
  ASSERT_EQ(ret, SUCCESS);
}

/*
 * 背景：
 * 当用户调用RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)时，
 * 最终执行器产生的输出是device上的gert::Tensor inner_outputs, 会调用CopyOutputs拷贝到host上，该host上的gert::Tensor就是要返回给
 * 用户的。
 *
 * 步骤：
 * 1. inner_outputs 模拟执行器申请的device上的gert::Tensor，通过CopyOutputs获得 user_outputs.
 * 2. 释放inner_outputs，校验user_outputs数据
 * 3. 测试user_outputs可以调用ShareFrom
 */
TEST_F(UtestHybridRt1Executor, CopyOutputsGertTensor_Success_CheckValueAndTestShareFrom) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  std::vector<gert::Tensor> inner_outputs(2);
  std::vector<gert::Tensor> user_outputs;

  const std::initializer_list<int64_t> dims = {1, 16, 16, 3};
  GeShape shape(dims);
  Format format(ge::FORMAT_ND);
  DataType data_type(ge::DT_INT32);
  int64_t data_size;
  ASSERT_EQ(TensorUtils::CalcTensorMemSize(shape, format, data_type, data_size), SUCCESS);
  const auto aligned_size = AlignSize(data_size);

  unique_ptr<uint8_t[]> data_buf1 = CreateBuffer(aligned_size);
  inner_outputs[0] = CreateGertTensor(dims, format, data_type, data_buf1.get(), aligned_size);

  unique_ptr<uint8_t[]> data_buf2 = CreateBuffer(aligned_size);
  inner_outputs[1] = CreateGertTensor(dims, format, data_type, data_buf2.get(), aligned_size);

  auto ret = executor.CopyOutputs(inner_outputs, user_outputs);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(user_outputs.size(), inner_outputs.size());
  for (size_t i = 0; i < inner_outputs.size(); i++) {
    EXPECT_EQ(user_outputs[i].GetSize(), data_size);
    EXPECT_EQ(inner_outputs[i].GetShape(), user_outputs[i].GetShape());
    EXPECT_EQ(inner_outputs[i].GetDataType(), user_outputs[i].GetDataType());
    EXPECT_EQ(inner_outputs[i].GetStorageFormat(), user_outputs[i].GetStorageFormat());
    EXPECT_EQ(user_outputs[i].GetPlacement(), gert::TensorPlacement::kOnHost);
  }

  // inner_outputs释放， user_outputs还可以正常访问
  inner_outputs.clear();
  data_buf1.reset();
  data_buf2.reset();
  for (const auto &tensor : user_outputs) {
    ASSERT_TRUE(CheckData(tensor.GetAddr(), tensor.GetSize()));
  }

  // 测试share_from, user_outputs释放，还可以访问user_output_share
  gert::Tensor user_output_share(user_outputs[0].GetShape(), user_outputs[0].GetFormat(), user_outputs[0].GetDataType());
  user_output_share.MutableTensorData().ShareFrom(user_outputs[0].GetTensorData());
  user_outputs.clear();

  ASSERT_TRUE(CheckData(user_output_share.GetAddr(), user_output_share.GetSize()));
}

TEST_F(UtestHybridRt1Executor, CopyOutputsGertTensor_Success_DeviceMemory) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  // 构造输出需要device内存的option
  std::map<std::string, std::string> options(
    {{OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "lazy_recompile"},
    {OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, "1"}});
  OptionSetter temp_option(options);

  std::vector<gert::Tensor> inner_outputs(2);
  std::vector<gert::Tensor> user_outputs;

  const std::initializer_list<int64_t> dims = {1, 16, 16, 3};
  GeShape shape(dims);
  Format format(ge::FORMAT_ND);
  DataType data_type(ge::DT_INT32);
  int64_t data_size;
  ASSERT_EQ(TensorUtils::CalcTensorMemSize(shape, format, data_type, data_size), SUCCESS);
  const auto aligned_size = AlignSize(data_size);

  unique_ptr<uint8_t[]> data_buf1 = CreateBuffer(aligned_size);
  inner_outputs[0] = CreateGertTensor(dims, format, data_type, data_buf1.get(), aligned_size);

  unique_ptr<uint8_t[]> data_buf2 = CreateBuffer(aligned_size);
  inner_outputs[1] = CreateGertTensor(dims, format, data_type, data_buf2.get(), aligned_size);

  auto ret = executor.CopyOutputs(inner_outputs, user_outputs);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(user_outputs.size(), inner_outputs.size());
  for (size_t i = 0; i < inner_outputs.size(); i++) {
    EXPECT_EQ(user_outputs[i].GetSize(), data_size);
    EXPECT_EQ(inner_outputs[i].GetShape(), user_outputs[i].GetShape());
    EXPECT_EQ(inner_outputs[i].GetDataType(), user_outputs[i].GetDataType());
    EXPECT_EQ(inner_outputs[i].GetStorageFormat(), user_outputs[i].GetStorageFormat());
    EXPECT_EQ(user_outputs[i].GetPlacement(), gert::TensorPlacement::kOnDeviceHbm); // 校验placement为hbm
    EXPECT_EQ(user_outputs[i].GetAddr(), inner_outputs[i].GetAddr()); // 校验地址一致
  }
}

TEST_F(UtestHybridRt1Executor, CopyOutputsGertTensor_Success_StringDataTypeCheckSize) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  std::vector<gert::Tensor> inner_outputs(1);
  std::vector<gert::Tensor> user_outputs;

  const std::initializer_list<int64_t> dims = {1, 16, 16, 3};
  GeShape shape(dims);
  Format format(ge::FORMAT_ND);
  DataType data_type(DT_STRING);
  int64_t data_size;
  ASSERT_EQ(TensorUtils::CalcTensorMemSize(shape, format, data_type, data_size), SUCCESS);
  const auto aligned_size = AlignSize(data_size);

  unique_ptr<uint8_t[]> data_buf1 = CreateBuffer(aligned_size);
  inner_outputs[0] = CreateGertTensor(dims, format, data_type, data_buf1.get(), aligned_size);

  auto ret = executor.CopyOutputs(inner_outputs, user_outputs);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(user_outputs.size(), inner_outputs.size());
  for (size_t i = 0; i < inner_outputs.size(); i++) {
    EXPECT_EQ(user_outputs[i].GetSize(), aligned_size); // String类型的并没有重新计算size，而是使用执行器输出tensor的size
    EXPECT_EQ(inner_outputs[i].GetShape(), user_outputs[i].GetShape());
    EXPECT_EQ(inner_outputs[i].GetDataType(), user_outputs[i].GetDataType());
    EXPECT_EQ(inner_outputs[i].GetStorageFormat(), user_outputs[i].GetStorageFormat());
    EXPECT_EQ(user_outputs[i].GetPlacement(), gert::TensorPlacement::kOnHost);
  }
}

TEST_F(UtestHybridRt1Executor, CopyOutputsGertTensor_Success_ZeroSize) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  std::vector<gert::Tensor> inner_outputs(1);
  std::vector<gert::Tensor> user_outputs;


  const std::initializer_list<int64_t> dims = {0};
  GeShape shape(dims);
  Format format(ge::FORMAT_ND);
  DataType data_type(DT_INT32);
  int64_t data_size;
  ASSERT_EQ(TensorUtils::CalcTensorMemSize(shape, format, data_type, data_size), SUCCESS);
  const auto aligned_size = AlignSize(data_size);

  unique_ptr<uint8_t[]> data_buf1 = CreateBuffer(aligned_size);
  inner_outputs[0] = CreateGertTensor(dims, format, data_type, data_buf1.get(), aligned_size);

  auto ret = executor.CopyOutputs(inner_outputs, user_outputs);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(user_outputs.size(), inner_outputs.size());
  for (size_t i = 0; i < inner_outputs.size(); i++) {
    EXPECT_EQ(user_outputs[i].GetSize(), 0); // size 为0
    EXPECT_EQ(inner_outputs[i].GetShape(), user_outputs[i].GetShape());
    EXPECT_EQ(inner_outputs[i].GetDataType(), user_outputs[i].GetDataType());
    EXPECT_EQ(inner_outputs[i].GetStorageFormat(), user_outputs[i].GetStorageFormat());
    EXPECT_EQ(user_outputs[i].GetPlacement(), gert::TensorPlacement::kOnHost);
  }
}

TEST_F(UtestHybridRt1Executor, CopyOutputsGertTensor_Failed_SizeTooSmall) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  std::vector<gert::Tensor> inner_outputs(1);
  std::vector<gert::Tensor> user_outputs;

  const std::initializer_list<int64_t> dims = {1, 16, 16, 3};
  GeShape shape(dims);
  Format format(ge::FORMAT_ND);
  DataType data_type(DT_INT32);
  int64_t data_size;
  ASSERT_EQ(TensorUtils::CalcTensorMemSize(shape, format, data_type, data_size), SUCCESS);
  const auto aligned_size = AlignSize(data_size);

  unique_ptr<uint8_t[]> data_buf1 = CreateBuffer(aligned_size);
  inner_outputs[0] = CreateGertTensor(dims, format, data_type, data_buf1.get(), aligned_size);
  inner_outputs[0].SetSize(data_size - 1);
  auto ret = executor.CopyOutputs(inner_outputs, user_outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestHybridRt1Executor, BuildDeviceTensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  GeTensorDesc ge_tensor_desc;
  int64_t output_size = 100;
  std::vector<ge::Tensor> outputs;
  (void)executor.BuildDeviceTensor(tensor, ge_tensor_desc, output_size, outputs);
  auto size = tensor.GetSize();
  ASSERT_EQ(size, 100);
}

TEST_F(UtestHybridRt1Executor, TestMemLeak) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);

  auto malloc_mock = std::make_shared<MockMalloc>();
  RuntimeStub::SetInstance(malloc_mock);
  auto malloc_acl_mock = std::make_shared<MockAclMalloc>();
  AclRuntimeStub::SetInstance(malloc_acl_mock);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  GeTensorDesc ge_tensor_desc;
  std::vector<ge::Tensor> outputs;

  (void)executor.BuildDeviceTensor(tensor, ge_tensor_desc, 100, outputs);

  EXPECT_EQ(malloc_acl_mock->malloc_flag > 0, true);
  outputs.clear();
  EXPECT_EQ(malloc_acl_mock->malloc_flag, 0);
}
TEST_F(UtestHybridRt1Executor, test_ExecuteWithStreamAsync) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelRtV1Executor executor(&hybrid_model, 0, nullptr);
  
  std::vector<gert::Tensor> gert_input;
  std::vector<gert::Tensor> gert_output;
  rtStream_t stream = nullptr;
  EXPECT_EQ(executor.ExecuteWithStreamAsync(gert_input, gert_output, stream), ge::FAILED);
}
} // namespace ge
