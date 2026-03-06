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
#include <memory>
#include <fstream>

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_local_context.h"
#include "depends/profiler/src/profiling_auto_checker.h"

#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/profiling/profiling_manager.h"
#include "common/dump/dump_manager.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/load/model_manager/task_info/ge/profiler_trace_task_info.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_task_info.h"
#include "graph/load/model_manager/task_info/fe/kernel_task_info.h"
#include "framework/common/runtime_tensor_desc.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "faker/space_registry_faker.h"
#include "runtime/subscriber/global_dumper.h"
#include "common/global_variables/diagnose_switch.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "common/share_graph.h"
#include "common/mem_conflict_share_graph.h"
#include "depends/runtime/src/runtime_stub.h"
#include "rt_error_codes.h"
#include "graph/load/model_manager/model_args_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "hcom/hcom_topo_info.h"
#include "runtime_stub.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "ge_runtime_stub/include/common/env_path.h"
#include "graph/debug/ge_attr_define.h"

using namespace std;

extern std::string g_runtime_stub_mock;

namespace ge {
namespace {
ModelParam default_parm;
class MockRtExecute : public ge::RuntimeStub {
 public:
  MOCK_METHOD4(rtModelExecuteSync, rtError_t(rtModel_t model, rtStream_t stream, uint32_t flag, int32_t timeout));
};

void TestNnExecute() {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  {
    OpDescPtr op_desc = CreateOpDesc("sk", "SuperKernel");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 4
    AttrUtils::SetGraph(op_desc, "_sk_sub_graph", graph);
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);

  const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char_t record_path[MMPA_MAX_PATH] = "mock_fail";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
  EXPECT_NE(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  unsetenv(kEnvRecordPath);

  input_data.blobs[0].length = 128;  // 128 not enough.
  EXPECT_NE(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

void TestNnExecuteWithGertTensor() {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;

  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  unique_ptr<uint8_t[]> data_buf_input(new (std::nothrow) uint8_t[512]);
  unique_ptr<uint8_t[]> data_buf_output(new (std::nothrow) uint8_t[512]);
  input_tensor.resize(1);
  output_tensor.resize(1);
  input_tensor[0] = {{{1, 4, 4, 8}, {1, 4, 4, 8}},                // shape
                             {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                             gert::kOnDeviceHbm,                                // placement
                             ge::DT_FLOAT,                              // data type
                             (void *) data_buf_input.get()};
  output_tensor[0] = {{{1, 4, 4, 8}, {1, 4, 4, 8}},                // shape
                             {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                             gert::kOnDeviceHbm,                                // placement
                             ge::DT_FLOAT,                              // data type
                             (void *) data_buf_output.get()};


  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  EXPECT_EQ(model.NnExecute(stream, true, input_tensor, output_tensor), SUCCESS);
}

void TestNnExecuteWithHostPlsModelIo() {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
}

void TestNnExecuteWithHostPlsModelIo_ZeroCopyReuse() {
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);

  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2580);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_NE(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

GeModelPtr ConstructGeModel(const size_t mem_size) {
  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_size);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetSrcName( { "data" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);
  }
  return ge_model;
}

void BuildDavinciModel(DavinciModel &model){
  model.SetKnownNode(true);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 2);
  AttrUtils::SetListInt(ge_model, ATTR_MODEL_NOTIFY_TYPES, {0, 1});

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,2}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_input);    // op_index = 0

  OpDescPtr op_kernel = CreateOpDesc("square", "Square");
  op_kernel->AddInputDesc(tensor);
  op_kernel->AddOutputDesc(tensor);
  op_kernel->SetInputOffset({1024});
  op_kernel->SetOutputOffset({1024});
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_kernel->GetName(), std::move(kernel_bin));
  EXPECT_TRUE(op_kernel->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
  EXPECT_TRUE(AttrUtils::SetStr(op_kernel, op_kernel->GetName() + "_kernelname", op_kernel->GetName()));
  AttrUtils::SetStr(op_kernel, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_kernel, ATTR_NAME_KERNEL_BIN_ID, "te_square_123");
  NodePtr node_kernel = graph->AddNode(op_kernel);  // op_index = 1

  OpDescPtr op_memcpy = CreateOpDesc("memcpy", MEMCPYASYNC);
  op_memcpy->AddInputDesc(tensor);
  op_memcpy->AddOutputDesc(tensor);
  op_memcpy->SetInputOffset({1024});
  op_memcpy->SetOutputOffset({5120});
  NodePtr node_memcpy = graph->AddNode(op_memcpy);  // op_index = 2

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({5120});
  op_output->SetSrcName( { "memcpy" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node_output = graph->AddNode(op_output);  // op_index = 3


  domi::TaskDef *task_def1 = model_task_def->add_task();
  task_def1->set_stream_id(0);
  task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def1->mutable_kernel();
  kernel_def->set_stub_func("stub_func");
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2U);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  domi::TaskDef *task_def2 = model_task_def->add_task();
  task_def2->set_stream_id(0);
  task_def2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  domi::MemcpyAsyncDef *memcpy_async = task_def2->mutable_memcpy_async();
  memcpy_async->set_src(1024);
  memcpy_async->set_dst(5120);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(2);

  model.Assign(ge_model);
}

void BuildDavinciModelWithMultiTasks(DavinciModel &model){
  model.SetKnownNode(true);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 2);
  AttrUtils::SetListInt(ge_model, ATTR_MODEL_NOTIFY_TYPES, {0, 1});

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,2}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_input);    // op_index = 0

  OpDescPtr op_kernel = CreateOpDesc("square", "Square");
  op_kernel->AddInputDesc(tensor);
  op_kernel->AddOutputDesc(tensor);
  op_kernel->SetInputOffset({1024});
  op_kernel->SetOutputOffset({1024});
  NodePtr node_kernel = graph->AddNode(op_kernel);  // op_index = 1

  OpDescPtr op_memcpy = CreateOpDesc("memcpy", MEMCPYASYNC);
  op_memcpy->AddInputDesc(tensor);
  op_memcpy->AddOutputDesc(tensor);
  op_memcpy->SetInputOffset({1024});
  op_memcpy->SetOutputOffset({5120});
  AttrUtils::SetInt(op_memcpy, ATTR_NAME_NODE_SQE_NUM, 66);
  NodePtr node_memcpy = graph->AddNode(op_memcpy);  // op_index = 2

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({5120});
  op_output->SetSrcName( { "memcpy" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node_output = graph->AddNode(op_output);  // op_index = 3


  domi::TaskDef *task_def1 = model_task_def->add_task();
  task_def1->set_stream_id(0);
  task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  domi::MemcpyAsyncDef *memcpy_async = task_def1->mutable_memcpy_async();
  memcpy_async->set_src(1024);
  memcpy_async->set_dst(5120);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(2);

  domi::TaskDef *task_def2 = model_task_def->add_task();
  task_def2->set_stream_id(0);
  task_def2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  domi::MemcpyAsyncDef *memcpy_async2 = task_def2->mutable_memcpy_async();
  memcpy_async2->set_src(1024);
  memcpy_async2->set_dst(5120);
  memcpy_async2->set_dst_max(512);
  memcpy_async2->set_count(1);
  memcpy_async2->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async2->set_op_index(2);

  model.Assign(ge_model);
}

void BuildDavinciModelWithFftsTask(DavinciModel &model, bool io_reuse_flag){
  model.SetKnownNode(true);
  model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120000);
  if (io_reuse_flag) {
    AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 5120000);
  }

  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task_def->set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def->mutable_ffts_plus_task();

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  OpDescPtr op_ffts = CreateOpDesc("ffts_node", PARTITIONEDCALL);
  op_ffts->AddInputDesc(tensor);
  op_ffts->AddOutputDesc(tensor);
  op_ffts->SetInputOffset({1024});
  op_ffts->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_ffts);    // op_index = 0

  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({8});
    op_desc->SetOutputOffset({8});
    NodePtr node = graph->AddNode(op_desc);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({8});
    op_desc->SetSrcName( { "data" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);
  }

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(512);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  aicaivdef->add_kernel_name("aivtest");
  aicaivdef->add_src_slot(1);
  model.Assign(ge_model);
}

void InitModel() {
  DavinciModel model(0, nullptr);
  BuildDavinciModel(model);
  ProfilingProperties::Instance().is_load_profiling_ = true;
  diagnoseSwitch::EnableDeviceProfiling();
  model.Init();
}
}
class OpsKernelInfoStoreStub : public OpsKernelInfoStore {
 public:
  OpsKernelInfoStoreStub() = default;
  Status Initialize(const std::map<std::string, std::string> &options) override { return SUCCESS; }
  Status Finalize() override { return SUCCESS; }
  void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const override {}
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const override { return true; }
  Status LoadTask(GETaskInfo &task) override {
    HcclDumpInfo dump_info = {0U, 0U, 0U, (void *)0x01, 1U, (void *)0x02, 1U};
    GETaskKernelHcclInfo kernel_hccl_info;
    task.kernelHcclInfo.emplace_back(kernel_hccl_info);
    task.kernelHcclInfo[0].hccl_dump_info.emplace_back(dump_info);
    return SUCCESS;
  }
  Status UnloadTask(GETaskInfo &task) { return SUCCESS; }
};

namespace {
class DModelListener : public ModelListener {
 public:
  DModelListener(){};
  ~DModelListener() = default;
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result, std::vector<gert::Tensor> &outputs) {
    complete_flag_ = true;
    return 0;
  }
  bool complete_flag_{false};
};
shared_ptr<ModelListener> g_local_call_back(new DModelListener());
}

class UtestDavinciModel : public testing::Test {
 protected:
  void SetUp() {
    VarManager::Instance(0)->Init(0, 0, 0, 0);
    const std::map<string, string> options{ {GRAPH_MEMORY_MAX_SIZE, "1048576"}, {VARIABLE_MEMORY_MAX_SIZE, "1048576"} };
    VarManager::Instance(0)->SetMemoryMallocSize(options, 10UL * 1024UL * 1024UL);
    MemManager::Instance().Initialize({ RT_MEMORY_HBM, RT_MEMORY_P2P_DDR });
    g_runtime_stub_mock.clear();
    RuntimeStub::GetInstance()->input_mem_copy_batch_count_ = 0;
    RTS_STUB_SETUP();
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_case_name =  test_info->test_case_name();
    test_work_dir = EnvPath().GetOrCreateCaseTmpPath(test_case_name);

  }

  void TearDown() {
    g_runtime_stub_mock.clear();
    VarManager::Instance(0)->FreeVarMemory();
    MemManager::Instance().Finalize();
    ProfilingTestUtil::Instance().Clear();
    RuntimeStub::GetInstance()->input_mem_copy_batch_count_ = 0;
    RTS_STUB_TEARDOWN();
    EnvPath().RemoveRfCaseTmpPath(test_case_name);
  }
protected:
  std::string test_case_name;
  std::string test_work_dir;
};

TEST_F(UtestDavinciModel, davinci_init_with_sub_memory_info) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN)->NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024, 1U});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
  (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  model.runtime_param_.fixed_mem_base = 10U;
  EXPECT_EQ(model.InitRuntimeParams(), SUCCESS);
  EXPECT_EQ(model.runtime_param_.fm_memory_infos.size(), 2U);
  EXPECT_EQ(model.runtime_param_.fm_memory_infos[0U].logic_memory_base, 0);
  EXPECT_EQ(model.runtime_param_.fm_memory_infos[0U].memory_size, 1024);
  EXPECT_EQ(model.runtime_param_.fm_memory_infos[1U].logic_memory_base, 1024);
  EXPECT_EQ(model.runtime_param_.fm_memory_infos[1U].memory_size, 1024);
  EXPECT_EQ(model.runtime_param_.fixed_fm_memory_infos[0U].logic_memory_base, 2048);
  EXPECT_EQ(model.runtime_param_.fixed_fm_memory_infos[0U].memory_size, 1024);
}

TEST_F(UtestDavinciModel, davinci_init_success) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  uint32_t mem_offset = 0U;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN));   // ccKernelType::TE
    CHAIN(NODE("_var_0", VARIABLE)->NODE("allreduce", HCOMALLREDUCE)->EDGE(0, 0)->NODE("relu", RELU)); // HCCL
    CHAIN(NODE("_arg_1", CONSTANTOP)->EDGE(0, 1)->NODE("add_n")->EDGE(0, 1)->NODE("relu")-> // ccKernelType::CUSTOMIZED
          NODE("square", SQUARE)->EDGE(0, 0)->      // ccKernelType::AI_CPU
          NODE("reshape", RESHAPE)->EDGE(0, 0)->    // ccKernelType::CUST_AI_CPU
          NODE("deque", FRAMEWORKOP)->EDGE(0, 0)->  // KERNEL_EX
          NODE("memcpy", MEMCPYASYNC)->EDGE(0, 0)-> // MEMCPY_ASYNC
          NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  SetKnownOpKernel(graph, mem_offset);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, kMemoryHostSVMFeatureMapLogicBase));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  {
    const auto &node = graph->FindNode("_arg_0");
    const auto &op_desc = node->GetOpDesc();
    AttrUtils::SetInt(op_desc, "globalworkspace_type", 0);
    AttrUtils::SetInt(op_desc, "globalworkspace_size", 1);
    AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  }

  {
    const auto &node = graph->FindNode("_var_0");
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetOutputOffset(
        {static_cast<int64_t>(1024 + VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase())});
    AttrUtils::SetBool(op_desc, VAR_ATTR_VAR_IS_BROADCAST, true);
    VarManager::Instance(graph->GetSessionID())->SetAllocatedGraphId(op_desc->GetName(), graph->GetGraphID());
  }

  {
    const auto &node = graph->FindNode("_arg_1");
    const auto &op_desc = node->GetOpDesc();

    std::vector<uint8_t> weights_value(64, 'A');
    GeTensorDesc data_desc = node->GetOpDesc()->GetOutputDesc(0);
    GeTensorPtr weight_value = MakeShared<GeTensor>(data_desc, weights_value.data(), weights_value.size());
    EXPECT_TRUE(AttrUtils::SetTensor(node->GetOpDesc(), ATTR_NAME_WEIGHTS, weight_value));
  }

  const std::shared_ptr<OpsKernelInfoStore> ops_kernel_store = MakeShared<OpsKernelInfoStoreStub>();
  {
    const auto &node = graph->FindNode("allreduce");
    const auto &op_desc = node->GetOpDesc();
    AttrUtils::SetInt(op_desc, "globalworkspace_type", 0);
    AttrUtils::SetInt(op_desc, "globalworkspace_size", 1);
    op_desc->SetExtAttr("OpsKernelInfoStorePtr", ops_kernel_store.get());

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
    task_def.set_stream_id(op_desc->GetStreamId());
    task_def.set_private_def("hccl_task"); // for GetPrivateDefByTaskDef

    auto &hccl_def = *task_def.mutable_kernel_hccl();
    hccl_def.set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();
    (void)AttrUtils::SetBool(op_desc, public_attr::OP_EXEC_NEVER_TIMEOUT, true);

    int32_t run_mode = static_cast<uint32_t>(domi::ImplyType::TVM);
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode));
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_add_n_123");
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def.set_stream_id(op_desc->GetStreamId());

    auto &kernel_def = *task_def.mutable_kernel();
    kernel_def.set_stub_func("stub_func");
    kernel_def.set_args_size(64);
    string args(64, '1');
    kernel_def.set_args(args.data(), 64);

    auto &context = *kernel_def.mutable_context();
    context.set_op_index(op_desc->GetId());
    context.set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    uint16_t args_offset[9] = {0};
    context.set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = graph->FindNode("relu");
    const auto &op_desc = node->GetOpDesc();
    const char task[] = "opattr";
    AttrUtils::SetBytes(op_desc, ATTR_NAME_OPATTR, Buffer::CopyFrom((uint8_t *)task, sizeof(task)));

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def.set_stream_id(op_desc->GetStreamId());

    auto &kernel_def = *task_def.mutable_kernel();
    kernel_def.set_stub_func("stub_func");
    kernel_def.set_args_size(64);
    string args(64, '1');
    kernel_def.set_args(args.data(), 64);

    auto &context = *kernel_def.mutable_context();
    context.set_op_index(op_desc->GetId());
    context.set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU));
    uint16_t args_offset[9] = {0};
    context.set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = graph->FindNode("square");
    const auto &op_desc = node->GetOpDesc();

    int32_t run_mode = static_cast<uint32_t>(domi::ImplyType::AI_CPU);
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode));
    EXPECT_TRUE(AttrUtils::SetBool(op_desc, ATTR_NO_TASK_AND_DUMP_NEEDED, true));    // for IsNoTaskAndDumpNeeded
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"ok"}));

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def.set_stream_id(op_desc->GetStreamId());

    auto &kernel_def = *task_def.mutable_kernel();
    kernel_def.set_stub_func("stub_func");
    kernel_def.set_args_size(64);
    string args(64, '1');
    kernel_def.set_args(args.data(), 64);

    auto &context = *kernel_def.mutable_context();
    context.set_op_index(op_desc->GetId());
    context.set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU));
    uint16_t args_offset[9] = {0};
    context.set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = graph->FindNode("reshape");
    const auto &op_desc = node->GetOpDesc();

    std::vector<char> kernel_bin(128, '0');
    const auto aicpu_kernel = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, aicpu_kernel);

    domi::TaskDef &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def.set_stream_id(op_desc->GetStreamId());

    std::vector<char> args_info(64U, '0');
    domi::KernelDef &kernel_def = *task_def.mutable_kernel();
    kernel_def.set_args_size(args_info.size());
    kernel_def.set_args(args_info.data(), args_info.size());
    kernel_def.set_so_name("libfeatures.so");
    kernel_def.set_kernel_name("features");

    domi::KernelContext &context = *kernel_def.mutable_context();
    context.set_kernel_type(static_cast<uint32_t>(ccKernelType::CUST_AI_CPU));
    context.set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("deque");
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetWorkspace({1308});   // offset
    op_desc->SetWorkspaceBytes({120U});  // length

    domi::TaskDef &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX));
    task_def.set_stream_id(op_desc->GetStreamId());

    std::vector<uint8_t> task_info(120U, 0U);
    domi::KernelExDef &kernel_def = *task_def.mutable_kernel_ex();
    kernel_def.set_task_info(task_info.data(), task_info.size());
    kernel_def.set_task_info_size(task_info.size());
    kernel_def.set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("memcpy");
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetOpKernelLibName(kEngineNameRts);

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    task_def.set_stream_id(op_desc->GetStreamId());

    auto &memcpy_async = *task_def.mutable_memcpy_async();
    memcpy_async.set_src(op_desc->GetInputOffset()[0]);
    memcpy_async.set_dst(op_desc->GetOutputOffset()[0]);
    memcpy_async.set_dst_max(512);
    memcpy_async.set_count(1);
    memcpy_async.set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async.set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("Node_Output");
    const auto &op_desc = node->GetOpDesc();

    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
  }

  {
    gert::GertRuntimeStub runtime_stub;
    dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 1);

    // Scene 1: Normal flow.
    DavinciModel model(0, nullptr);
    domi::GetContext().is_online_model = true;
    model.Assign(ge_model);
    model.SetAiCpuCustFlag(true);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_EQ(model.stream_to_task_index_list_.size(), 1); // 单条流
    EXPECT_EQ(model.stream_to_task_index_list_[1].size(), 7); // 单条流，hccl所在流上有7个task
    model.main_follow_stream_mapping_[1].push_back(ge::ValueToPtr((2))); // hccl从流
    EXPECT_EQ(model.RecoverModel(), SUCCESS); // 单条流，hccl所在流上有7个task
    runtime_stub.Clear();
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 1);
  }

  {
    gert::GertRuntimeStub runtime_stub;
    dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 1);

    DavinciModel model(0, nullptr);
    domi::GetContext().is_online_model = true;
    model.Assign(ge_model);
    model.SetAiCpuCustFlag(true);
    EXPECT_EQ(model.Init(), SUCCESS);

    class TaskIdChanger : public TaskInfo {
    public:
      TaskIdChanger(TaskInfoPtr original_task) : original_task_(original_task) {
        is_support_redistribute_ = true;
        original_task_id_ = original_task->GetTaskID();
        new_task_id_ = original_task_id_ + 1000;
      }

      uint32_t GetTaskID() const override {
        if (!distributed_) {
          return original_task_id_;
        } else {
          return new_task_id_;
        }
      }

      Status Distribute() override {
        auto status = original_task_->Distribute();
        if (status == SUCCESS) {
          distributed_ = true;
          GELOGI("Task distributed, task_id will change from %u to %u", original_task_id_, new_task_id_);
        }
        return status;
      }

      Status Init(const domi::TaskDef &task_def, DavinciModel *davinci_model,
                  const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                  const IowAddrs &iow_addrs) override {
        return original_task_->Init(task_def, davinci_model, args, persistent_workspace, iow_addrs);
      }

      uint32_t GetStreamId() const override { return original_task_->GetStreamId(); }
      bool IsSupportReDistribute() override { return true; }

    private:
      TaskInfoPtr original_task_;
      uint32_t original_task_id_;
      uint32_t new_task_id_;
      mutable bool distributed_ = false;
    };

    for (size_t i = 0; i < model.task_list_.size() && i < 3; ++i) {
      auto original_task = model.task_list_[i];
      if (original_task) {
        auto wrapper = std::make_shared<TaskIdChanger>(original_task);
        model.task_list_[i] = wrapper;
        GELOGI("Wrapped task %zu: will change task_id from %u to %u",
              i, wrapper->GetTaskID(), wrapper->GetTaskID() + 1000);
      }
    }

    EXPECT_EQ(model.stream_to_task_index_list_.size(), 1);
    model.main_follow_stream_mapping_[1].push_back(ge::ValueToPtr((2)));

    EXPECT_EQ(model.RecoverModel(), SUCCESS);

    runtime_stub.Clear();
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 1);
  }

  {
    // Scene 1: Normal flow.
    DavinciModel model(0, nullptr);

    model.Assign(ge_model);
    model.SetAiCpuCustFlag(true);
    EXPECT_EQ(model.Init(), SUCCESS);
    const auto model_bind_streams = RuntimeStub::GetInstance()->model_bind_streams_;
    EXPECT_EQ(model_bind_streams.size(), model.stream_to_first_task_id_.size());
    ASSERT_TRUE(model_bind_streams.size() > 0U);
    auto pre_stream = model_bind_streams[0];
    int32_t pre_stream_id = -1;
    for (size_t i = 0U; i < model.stream_list_.size(); i++) {
      if (model.stream_list_[i] == pre_stream) {
        pre_stream_id = static_cast<int32_t>(i);
      }
    }
    ASSERT_TRUE(model.stream_to_first_task_id_.find(pre_stream_id) != model.stream_to_first_task_id_.end());
    auto pre_stream_first_task_id = model.stream_to_first_task_id_[pre_stream_id];
    for (size_t i = 1U; i < model_bind_streams.size(); i++) {
      auto cur_stream = model_bind_streams[i];
      int32_t cur_stream_id = -1;
      for (size_t j = 0U; j < model.stream_list_.size(); j++) {
        if (model.stream_list_[j] == cur_stream) {
          cur_stream_id = static_cast<int32_t>(j);
        }
      }
      ASSERT_TRUE(model.stream_to_first_task_id_.find(cur_stream_id) != model.stream_to_first_task_id_.end());
      auto cur_stream_first_task_id = model.stream_to_first_task_id_[cur_stream_id];
      ASSERT_TRUE(pre_stream_first_task_id < cur_stream_first_task_id);
      pre_stream_first_task_id = cur_stream_first_task_id;
    }
    EXPECT_EQ(model.GetAiCpuCustFlag(), true);
    model.SetAiCpuCustFlag(false);

    EXPECT_EQ(model.input_addrs_list_.size(), 1);
    EXPECT_EQ(model.output_addrs_list_.size(), 1);
    EXPECT_EQ(model.task_list_.size(), model_def->task_size());

    OutputData output_data;
    std::vector<gert::Tensor> outputs;
    EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
    EXPECT_EQ(output_data.blobs.size(), 1);
    EXPECT_EQ(outputs.size(), 1);
    ProfilingProperties::Instance().is_load_profiling_ = false;
  }
  const auto &bind_model_streams = RuntimeStub::GetInstance()->model_bind_streams_;
  for (const auto stream : RuntimeStub::GetInstance()->model_unbind_streams_) {
    auto iter = std::find(bind_model_streams.begin(), bind_model_streams.end(), stream);
    EXPECT_NE(iter, bind_model_streams.end());
  }

  EXPECT_EQ(setenv(kEnvGeuseStaticMemory.c_str(), "1", true), 0);
  {
    // Scene 2: Special env.
    DavinciModel model(0, nullptr);
    model.data_dumper_.dump_properties_.is_train_op_debug_ = true;

    model.Assign(ge_model);
    EXPECT_EQ(model.Init(), SUCCESS);

    EXPECT_EQ(model.input_addrs_list_.size(), 1);
    EXPECT_EQ(model.output_addrs_list_.size(), 1);
    EXPECT_EQ(model.task_list_.size(), model_def->task_size());

    OutputData output_data;
    std::vector<gert::Tensor> outputs;
    EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
    EXPECT_EQ(output_data.blobs.size(), 1);
    EXPECT_EQ(outputs.size(), 1);
    ProfilingProperties::Instance().is_load_profiling_ = false;
  }
  EXPECT_EQ(unsetenv(kEnvGeuseStaticMemory.c_str()), 0);

  {
    // Scene 3: MDC Queue
    DavinciModel model(0, nullptr);
    QueueAttrs inputQueue = {1001U, 0, 0, 0U};
    QueueAttrs outputQueue = {1004U, 0, 0, 0U};
    std::vector<QueueAttrs> input_queue_attrs;
    input_queue_attrs.emplace_back(inputQueue);
    std::vector<QueueAttrs> output_queue_attrs;
    output_queue_attrs.emplace_back(outputQueue);
    model.SetQueIds(input_queue_attrs, output_queue_attrs);

    model.Assign(ge_model);
    model.input_no_tiling_flag_ = { false };
    model.output_no_tiling_flag_ = { false };
    EXPECT_EQ(model.Init(), SUCCESS);

    EXPECT_EQ(model.input_addrs_list_.size(), 1);
    EXPECT_EQ(model.output_addrs_list_.size(), 1);
    EXPECT_EQ(model.task_list_.size(), model_def->task_size());
    // AddHeadStream: +0
    // BindInputQueue: +1
    // CpuTaskModelZeroCopy: +1
    // BindOutputQueue: +0
    // CpuTaskModelZeroCopy: +0
    // CpuActiveStream: +1
    // CpuWaitEndGraph: +1
    // BindEnqueue: +1
    // CpuPostProcess: +1
    // CpuModelRepeat: +1
    EXPECT_EQ(model.cpu_task_list_.size(), 7);
  }

  {
    // Scene 4: MDC Queue
    DavinciModel model(0, nullptr);
    QueueAttrs inputQueue = {1001U, 0, 0, 0U};
    QueueAttrs outputQueue = {1004U, 0, 0, 0U};
    std::vector<QueueAttrs> input_queue_attrs;
    input_queue_attrs.emplace_back(inputQueue);
    std::vector<QueueAttrs> output_queue_attrs;
    output_queue_attrs.emplace_back(outputQueue);
    model.SetQueIds(input_queue_attrs, output_queue_attrs);

    model.Assign(ge_model);
    model.input_no_tiling_flag_ = { true };
    model.output_no_tiling_flag_ = { true };
    model.has_no_tiling_input_ = true;
    model.has_no_tiling_output_ = true;
    EXPECT_EQ(model.Init(), SUCCESS);

    EXPECT_EQ(model.input_addrs_list_.size(), 1);
    EXPECT_EQ(model.output_addrs_list_.size(), 1);
    EXPECT_EQ(model.task_list_.size(), model_def->task_size());
    EXPECT_EQ(model.cpu_task_list_.size(), 7);
  }

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
    // Scene 5: extend size statis memory
    DavinciModel model1(0, nullptr);
    model1.data_dumper_.dump_properties_.is_train_op_debug_ = true;

    model1.Assign(ge_model);
    EXPECT_EQ(model1.Init(), SUCCESS);

    DavinciModel model2(0, nullptr);
    model2.data_dumper_.dump_properties_.is_train_op_debug_ = true;

    model2.Assign(ge_model);
    EXPECT_EQ(model2.Init(), SUCCESS);

    // test1, model2 reuse model1 feature map memory
    EXPECT_EQ(model1.mem_base_, model2.mem_base_);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20240));
    DavinciModel model3(0, nullptr);
    model3.data_dumper_.dump_properties_.is_train_op_debug_ = true;

    // test2, auto  extend feature map memory
    model3.Assign(ge_model);
    EXPECT_EQ(model3.Init(), SUCCESS);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
    ProfilingProperties::Instance().is_load_profiling_ = false;
  }

  {
    // Scene 6: dump on wather mode
    DavinciModel model(0, nullptr);
    DumpProperties dump_properties;
    dump_properties.SetDumpMode("output");
    dump_properties.AddPropertyValue(DUMP_LAYER_OP_MODEL, {"deque","allreduce"});
    dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"square", "allreduce"});
    model.SetDumpProperties(dump_properties);
    model.Assign(ge_model);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_FALSE(model.OpNeedDump("square"));

    EXPECT_TRUE(model.OpNeedSetDumpFlagOnWatcherModel("deque"));
    EXPECT_TRUE(model.OpNeedSetDumpFlagOnWatcherModel("square"));
    EXPECT_TRUE(model.OpNeedSetDumpFlagOnWatcherModel("allreduce"));

    EXPECT_FALSE(model.OpNeedDumpOnWatcherModel("deque"));
    EXPECT_TRUE(model.OpNeedDumpOnWatcherModel("square"));
    EXPECT_TRUE(model.OpNeedDumpOnWatcherModel("allreduce"));

    EXPECT_TRUE(model.OpNoNeedDumpOnWatcherModel("deque"));
    EXPECT_FALSE(model.OpNoNeedDumpOnWatcherModel("square"));
    EXPECT_FALSE(model.OpNoNeedDumpOnWatcherModel("allreduce"));
    EXPECT_TRUE(model.IsDumpWatcherModelEnable());
  }

  {
    // Scene 7: dump on wather mode for hccl layer
    DavinciModel model(0, nullptr);
    DumpProperties dump_properties;
    dump_properties.SetDumpMode("output");
    dump_properties.AddPropertyValue(DUMP_LAYER_OP_MODEL, {"allreduce"});
    dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"square"});
    model.SetDumpProperties(dump_properties);
    model.Assign(ge_model);
    EXPECT_EQ(model.Init(), SUCCESS);

    EXPECT_TRUE(model.OpNeedSetDumpFlagOnWatcherModel("allreduce"));
    EXPECT_TRUE(model.OpNeedSetDumpFlagOnWatcherModel("square"));

    EXPECT_FALSE(model.OpNeedDumpOnWatcherModel("allreduce"));
    EXPECT_TRUE(model.OpNeedDumpOnWatcherModel("square"));

    EXPECT_TRUE(model.OpNoNeedDumpOnWatcherModel("allreduce"));
    EXPECT_FALSE(model.OpNoNeedDumpOnWatcherModel("square"));
    EXPECT_TRUE(model.IsDumpWatcherModelEnable());
  }

  {
    // Scene 8: dump on op_ranges
    DavinciModel model(0, nullptr);
    DumpProperties dump_properties;
    dump_properties.SetDumpMode("output");
    model.om_name_ = "test";
    std::vector<std::pair<std::string, std::string>> op_ranges = {{"square", "deque"},};
    dump_properties.SetOpDumpRange("test", op_ranges);

    model.SetDumpProperties(dump_properties);
    model.Assign(ge_model);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_TRUE(model.OpNeedDump("square"));
    EXPECT_TRUE(model.OpNeedDump("deque"));
  }

  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, init_data_op) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  uint8_t mem_base_addr = 0;
  MemInfo one_fm_mem_info;
  one_fm_mem_info.memory_type = RT_MEMORY_HBM;
  one_fm_mem_info.logic_memory_base = 0x80000000;
  one_fm_mem_info.memory_size = 512;
  one_fm_mem_info.memory_base = &mem_base_addr;
  one_fm_mem_info.is_fixed_addr_prior = 1;
  model.runtime_param_.fixed_fm_memory_infos.push_back(one_fm_mem_info);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_input);

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({1024});
  op_output->SetSrcName( { "data" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node_output = graph->AddNode(op_output);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_op_subgraph) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_input);

  uint32_t data_op_index = 0;
  map<uint32_t, OpDescPtr> data_by_index;
  set<uint64_t> input_outside_addrs;
  EXPECT_EQ(model.InitDataOp(nullptr, node, data_op_index, data_by_index, input_outside_addrs), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 0);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(data_op_index == 0);
  EXPECT_TRUE(data_by_index.empty());
}

TEST_F(UtestDavinciModel, init_data_op_subgraph_with_refdata) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  ComputeGraphPtr root_graph = MakeShared<ComputeGraph>("main");
  OpDescPtr op_partition = CreateOpDesc("partitionedcall", PARTITIONEDCALL);
  op_partition->AddInputDesc(tensor);
  op_partition->AddOutputDesc(tensor);
  op_partition->SetInputOffset({1024});
  op_partition->SetOutputOffset({1024});
  NodePtr partitioned_node = root_graph->AddNode(op_partition);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_input);

  OpDescPtr op_refdata = CreateOpDesc("refdata", REFDATA);
  op_refdata->AddInputDesc(tensor);
  op_refdata->AddOutputDesc(tensor);
  op_refdata->SetInputOffset({1024});
  op_refdata->SetOutputOffset({1024});
  NodePtr refdata_node = graph->AddNode(op_refdata);
  graph->SetParentGraph(root_graph);
  graph->SetParentNode(partitioned_node);
  GraphUtils::AddEdge(node->GetOutControlAnchor(), refdata_node->GetInControlAnchor());

  uint32_t data_op_index = 0;
  map<uint32_t, OpDescPtr> data_by_index;
  set<uint64_t> input_outside_addrs;
  EXPECT_EQ(model.InitDataOp(graph, refdata_node, data_op_index, data_by_index, input_outside_addrs), SUCCESS);

  EXPECT_TRUE(data_by_index.empty());
  EXPECT_TRUE(input_outside_addrs.empty());
}

TEST_F(UtestDavinciModel, init_netoutput_op_subgraph) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({1024});
  op_output->SetSrcName( { "data" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node = graph->AddNode(op_output);

  std::vector<OpDescPtr> output_op_list;
  set<uint64_t> output_outside_addrs;
  EXPECT_EQ(model.InitNetOutput(nullptr, node, output_op_list, output_outside_addrs), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 0);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(output_op_list.empty());
}

TEST_F(UtestDavinciModel, init_multi_task_model) {
  DavinciModel model(0, nullptr);
  BuildDavinciModelWithMultiTasks(model);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_EQ(model.stream_task_num_.size(), 1);
  EXPECT_EQ(model.stream_task_num_[0], 66);
}

TEST_F(UtestDavinciModel, init_unknown) {
  DavinciModel model(0, nullptr);
  BuildDavinciModel(model);
  ProfilingProperties::Instance().is_load_profiling_ = true;
  EXPECT_EQ(model.Init(), SUCCESS);
  ProfilingProperties::Instance().is_load_profiling_ = false;

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_EQ(model.task_list_.size(), 2);

  EXPECT_EQ(model.task_list_[0]->UpdateArgs(), SUCCESS);
  EXPECT_EQ(model.task_list_[1]->UpdateArgs(), SUCCESS);

  std::vector<string> out_shape_info;
  model.GetOutputShapeInfo(out_shape_info);

  std::vector<InputOutputDescInfo> input_descs;
  std::vector<InputOutputDescInfo> output_descs;
  EXPECT_EQ(model.GetInputOutputDescInfo(input_descs, output_descs), SUCCESS);

  int32_t virtual_addr = 0;
  const std::vector<void *> inputs = { &virtual_addr };
  const std::vector<void *> outputs = { &virtual_addr  };
  EXPECT_EQ(model.UpdateKnownNodeArgs(VPtrToValue(inputs), VPtrToValue(outputs)), SUCCESS);
}

TEST_F(UtestDavinciModel, init_outputs_pls_more_outputs) {
  DavinciModel model(0, nullptr);
  BuildDavinciModel(model);
  ProfilingProperties::Instance().is_load_profiling_ = true;
  EXPECT_EQ(model.Init(), SUCCESS);
  ProfilingProperties::Instance().is_load_profiling_ = false;

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_EQ(model.task_list_.size(), 2);

  EXPECT_EQ(model.task_list_[0]->UpdateArgs(), SUCCESS);
  EXPECT_EQ(model.task_list_[1]->UpdateArgs(), SUCCESS);

  std::vector<string> out_shape_info;
  model.GetOutputShapeInfo(out_shape_info);

  std::vector<InputOutputDescInfo> input_descs;
  std::vector<InputOutputDescInfo> output_descs;
  EXPECT_EQ(model.GetInputOutputDescInfo(input_descs, output_descs), SUCCESS);

  int32_t virtual_addr = 0;
  const std::vector<void *> inputs = { &virtual_addr };
  const std::vector<void *> outputs = { &virtual_addr, &virtual_addr};
  EXPECT_EQ(model.UpdateKnownNodeArgs(VPtrToValue(inputs), VPtrToValue(outputs)), SUCCESS);
}

TEST_F(UtestDavinciModel, InitDavinciModel_ReportExpectedProfiling_WithProfilingOn) {
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(InitModel, 1, 2, 2, 1);
}

TEST_F(UtestDavinciModel, InitDavinciModelWithAddrFixedOpt) {
  DavinciModel model(0, nullptr);
  BuildDavinciModel(model);
  ge::GetThreadLocalContext().SetGraphOption({{"ge.exec.static_model_addr_fixed", "1"}});
  EXPECT_EQ(model.Init(), SUCCESS);
  ge::GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestDavinciModel, Init_variable_op) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.runtime_param_.mem_size = 51200;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr var1 = CreateOpDesc("var1", VARIABLE);
  var1->AddInputDesc(tensor);
  var1->AddOutputDesc(tensor);
  var1->SetInputOffset({1024 + kMemoryVarLogicBase});
  var1->SetOutputOffset({1024 + kMemoryVarLogicBase});
  AttrUtils::SetBool(var1, VAR_ATTR_VAR_IS_BROADCAST, true);
  graph->AddNode(var1);

  OpDescPtr var2 = CreateOpDesc(NODE_NAME_GLOBAL_STEP, VARIABLE);
  var2->AddInputDesc(tensor);
  var2->AddOutputDesc(tensor);
  var2->SetInputOffset({1024 + kMemoryVarLogicBase});
  var2->SetOutputOffset({1024 + kMemoryVarLogicBase});
  graph->AddNode(var2);

  EXPECT_EQ(model.InitNodes(graph), SUCCESS);

  //EXPECT_EQ(model.UpdateStepInfo(), SUCCESS);

  OutputData output_data;
  EXPECT_FALSE(model.has_output_node_);
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.CopyOutputData(output_data, output_tensor), SUCCESS);
  model.is_async_mode_ = true;
  EXPECT_EQ(model.UpdateStepInfoWithStream(), SUCCESS);

  std::vector<gert::Tensor> outputs;
  model.AssembleListenerOutput(nullptr, 1, outputs);
  free(reinterpret_cast<void *>(model.runtime_param_.mem_base));
  model.runtime_param_.mem_base = 0;
}

TEST_F(UtestDavinciModel, Init_hcom_op) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  ge::GeTensorDesc tensor_desc(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op_allreduce ", HVDCALLBACKALLREDUCE);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->SetOpKernelLibName(ge::kEngineNameHccl);
  graph->AddNode(op_desc);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  model.Assign(ge_model);

  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  EXPECT_EQ(model.HasHcclTask(), true);
}

TEST_F(UtestDavinciModel, init_constant_op) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.runtime_param_.mem_size = 51200;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc1 = CreateOpDesc("FileConstant1", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc1, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc1, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc1->AddOutputDesc(tensor_desc);
  op_desc1->SetOutputOffset({0});
  graph->AddNode(op_desc1);

  OpDescPtr op_desc2 = CreateOpDesc("FileConstant2", FILECONSTANT);
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc2, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc2, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc2, "shape", shape));
  op_desc2->AddOutputDesc(tensor_desc);
  op_desc2->SetOutputOffset({0});
  graph->AddNode(op_desc2); // test ExternalWeightManager::CheckAndSetWeightLoaded

  std::unique_ptr<float[]> float_buf(new float[16]);
  std::string file_name = "tmp_weight_file.bin";
  std::ofstream out1(file_name, std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write((char *)float_buf.get(), 16 * sizeof(float));
  out1.close();
  model.file_id_and_path_map_.insert(std::pair<std::string, std::string>("file", "tmp_weight_file.bin"));
  EXPECT_EQ(model.InitNodes(graph), SUCCESS);
  free(reinterpret_cast<void *>(model.runtime_param_.mem_base));
  model.runtime_param_.mem_base = 0;
  (void)remove("tmp_weight_test_copy_one_weight.bin");
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Op_OK) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./";
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  std::unique_ptr<float[]> float_buf(new float[512 / sizeof(float)]);
  std::string file_name = "tmp_weight_file.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 512);
  out1.close();

  model.file_id_and_path_map_.insert(std::pair<std::string, std::string>("file", "tmp_weight_file.bin"));
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void*>(model.weights_mem_base_));
  (void)remove("tmp_weight_file.bin");
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Success_UserDeviceMem) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./weight/";
  const size_t mem_size = 64U;
  uint8_t user_mem[mem_size];
  FileConstantMem file_conststant_mem{"weight_2132345.bin", user_mem, mem_size};

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_PATH, "./weight_2132345.bin");
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  std::unique_ptr<float[]> float_buf(new float[512 / sizeof(float)]);
  std::string file_name = "weight_2132345.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 512);
  out1.close();

  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  model.FreeFileConstantMem();
  (void)remove("weight_2132345.bin");

  model.SetFileConstantUserDeviceMem({file_conststant_mem});
  status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  auto iter = model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0));
  ASSERT_NE(iter, model.runtime_param_.fileconstant_addr_mapping.end());

  EXPECT_EQ(iter->second, reinterpret_cast<uintptr_t>(user_mem));
  free(reinterpret_cast<void*>(model.weights_mem_base_));
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Success_UserDeviceMem_HandleIndividualWeights) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./weight/";
  const size_t mem_size = 64U;
  uint8_t user_mem_1[mem_size];
  uint8_t user_mem_2[mem_size];
  FileConstantMem file_conststant_mem1{"weight_21323451.bin", user_mem_1, mem_size};
  FileConstantMem file_conststant_mem2{"weight_21323452.bin", user_mem_2, mem_size};

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_PATH, "./weight_21323451.bin");
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  OpDescPtr op_desc2 = CreateOpDesc("FileConstant", FILECONSTANT);
  AttrUtils::SetStr(op_desc2, ATTR_NAME_FILE_PATH, "./weight_21323452.bin");

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc2, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc2, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc2, "shape", shape));
  GeTensorDesc tensor_desc2(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc2, 128);
  op_desc2->AddOutputDesc(tensor_desc2);
  op_desc2->SetOutputOffset({1});
  graph->AddNode(op_desc2);

  std::unique_ptr<float[]> float_buf(new float[512 / sizeof(float)]);
  std::string file_name = "weight_21323451.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 512);
  out1.close();
  file_name = "weight_21323452.bin";
  std::ofstream out2(file_name, std::ios::binary);
  EXPECT_TRUE(out2.is_open());
  out2.write((char *)float_buf.get(), 512);
  out2.close();

  (void)remove("weight_21323451.bin");
  (void)remove("weight_21323452.bin");

  model.SetFileConstantUserDeviceMem({file_conststant_mem1, file_conststant_mem2});
  auto status  = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  auto iter = model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0));
  ASSERT_NE(iter, model.runtime_param_.fileconstant_addr_mapping.end());
  auto iter1 = model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(1));
  ASSERT_NE(iter1, model.runtime_param_.fileconstant_addr_mapping.end());

  EXPECT_EQ(iter->second, reinterpret_cast<uintptr_t>(user_mem_1));
  EXPECT_EQ(iter1->second, reinterpret_cast<uintptr_t>(user_mem_2));
  free(reinterpret_cast<void*>(model.weights_mem_base_));
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Failed_UserDeviceMemSizeInvalid) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./weight/";
  const size_t mem_size = 64U;
  uint8_t user_mem[mem_size];
  FileConstantMem file_conststant_mem{"weight_2132345.bin", user_mem, mem_size - 1U};
  model.SetFileConstantUserDeviceMem({file_conststant_mem});

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_PATH, "/home/weight_2132345.bin");
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_NE(status, SUCCESS);
  free(reinterpret_cast<void*>(model.weights_mem_base_));
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Success_WeightCombined) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./";

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant0", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "weight_combined_2132345.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 768));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 768);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});

  OpDescPtr op_desc1 = CreateOpDesc("FileConstant1", FILECONSTANT);

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc1, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_OFFSET, 1024));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_LOCATION, "weight_combined_2132345.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_LENGTH, 1024));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc1, "shape", shape));
  GeTensorDesc tensor_desc1(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc1, 1024);
  op_desc1->AddOutputDesc(tensor_desc1);
  op_desc1->SetOutputOffset({1});
  graph->AddNode(op_desc);
  graph->AddNode(op_desc1);

  std::unique_ptr<float[]> float_buf(new float[2048 / sizeof(float)]);
  std::string file_name = "weight_combined_2132345.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 2048);
  out1.close();

  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(1)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void*>(model.weights_mem_base_));
  (void)remove("weight_combined_2132345.bin");
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Success_IndividualWeights) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 2;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  model.file_constant_weight_dir_ = "./";

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant0", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 0));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "weight_combined_1.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 768));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 768);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});

  OpDescPtr op_desc1 = CreateOpDesc("FileConstant1", FILECONSTANT);

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc1, "dtype", DT_INT32));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_OFFSET, 1024));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc1, ATTR_NAME_LOCATION, "weight_combined_2.bin"));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc1, ATTR_NAME_LENGTH, 1024));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc1, "shape", shape));
  GeTensorDesc tensor_desc1(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc1, 1024);
  op_desc1->AddOutputDesc(tensor_desc1);
  op_desc1->SetOutputOffset({1});
  graph->AddNode(op_desc);
  graph->AddNode(op_desc1);

  std::unique_ptr<float[]> float_buf(new float[1024 / sizeof(float)]);
  std::string file_name = "weight_combined_1.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 1024);
  out1.close();

  file_name = "weight_combined_2.bin";
  std::ofstream out2(file_name, std::ios::binary);
  EXPECT_TRUE(out2.is_open());
  out2.write((char *)float_buf.get(), 1024);
  out2.close();

  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, SUCCESS);
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(0)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  ASSERT_NE(model.runtime_param_.fileconstant_addr_mapping.find(static_cast<int64_t>(1)),
            model.runtime_param_.fileconstant_addr_mapping.end());
  free(reinterpret_cast<void*>(model.weights_mem_base_));
  (void)remove("weight_combined_1.bin");
  (void)remove("weight_combined_2.bin");
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Op_Memory_Allocation_Failed) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 3;
  model.weights_mem_base_ = 0;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({1});
  graph->AddNode(op_desc);

  std::unique_ptr<float[]> float_buf(new float[512 / sizeof(float)]);
  std::string file_name = "tmp_weight_file.bin";
  std::ofstream out1(file_name, std::ios::binary);
  EXPECT_TRUE(out1.is_open());
  out1.write((char *)float_buf.get(), 512);
  out1.close();

  model.file_id_and_path_map_.insert(std::pair<std::string, std::string>("file", "tmp_weight_file.bin"));
  g_runtime_stub_mock = "rtMalloc";
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, ACL_ERROR_GE_MEMORY_ALLOCATION);
}

TEST_F(UtestDavinciModel, Preprocess_Fileconstant_Op_Param_Failed) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.session_id_ = 4;
  model.runtime_param_.mem_size = 51200;
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0, 1});
  graph->AddNode(op_desc);
  auto status = model.PreProcessFileConstants(graph, default_parm);
  EXPECT_EQ(status, PARAM_INVALID);
  free(reinterpret_cast<void*>(model.weights_mem_base_));
}

TEST_F(UtestDavinciModel, output_no_tiling_data) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.runtime_param_.mem_size = 51200;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr where = MakeShared<OpDesc>("where", "Where");
  std::vector<int64_t> shape = {-1, 2};
  GeTensorDesc tensor(GeShape(shape), FORMAT_ND, DT_INT64);
  const std::vector<std::pair<int64_t, int64_t>> range = {{1, 10}, {2, 2}};
  tensor.SetShapeRange(range);
  where->AddOutputDesc(tensor);
  where->SetOutputOffset({1024});
  AttrUtils::SetBool(tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  AttrUtils::SetInt(tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 2048);
  graph->AddNode(where);

  OpDescPtr output = MakeShared<OpDesc>("output", "NetOutput");
  output->SetInputOffset({1024});
  const std::vector<string> src_name = {"where"};
  output->SetSrcName(src_name);
  const std::vector<int64_t> src_index = {0};
  output->SetSrcIndex(src_index);
  output->AddInputDesc(tensor);
  graph->AddNode(output);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  RuntimeTensorDesc *addr = reinterpret_cast<RuntimeTensorDesc *>(model.output_data_info_[0].GetBasicAddr());
  addr->shape[0] = 2;
  addr->shape[1] = 5;
  addr->shape[2] = 2;
  model.is_async_mode_ = true;
  std::vector<gert::Tensor> outputs;
  model.AssembleListenerOutput(nullptr, 1, outputs);

  model.is_online_infer_dynamic_ = true;

  model.has_output_node_ = false;
  model.ReturnSequenceResult(nullptr, 1, false);

  model.has_output_node_ = true;
  model.ReturnSequenceResult(nullptr, 1, true);

  free(reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base));
  model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestDavinciModel, copy_input_data_no_tiling) {
  DavinciModel model(0, g_local_call_back);
  model.ge_model_ = MakeShared<GeModel>();
  model.runtime_param_.mem_size = 51200;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(malloc(model.runtime_param_.mem_size));
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  OpDescPtr data = CreateOpDesc("data", DATA);
  GeTensorDesc tensor_desc(GeShape({-1,-1,224,224}), FORMAT_NCHW, DT_FLOAT);
  data->SetOutputOffset({2048});
  AttrUtils::SetBool(tensor_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
  TensorUtils::SetSize(tensor_desc, 10240);
  data->AddOutputDesc(tensor_desc);
  NodePtr data_node = graph->AddNode(data);

  uint32_t data_op_index = 0;
  set<uint64_t> input_outside_addrs;
  map<uint32_t, OpDescPtr> data_by_index;
  EXPECT_EQ(model.InitDataOp(graph, data_node, data_op_index, data_by_index, input_outside_addrs), SUCCESS);

  InputData input_data;
  std::vector<int64_t> shape = {4,3,224,224};
  input_data.shapes.push_back(shape);
  size_t size = 5160;
  void *data_addr = (void *)malloc(size);
  DataBuffer buffer(data_addr, size, false);
  input_data.blobs.push_back(buffer);
  EXPECT_EQ(model.CopyInputData(input_data), SUCCESS);
  free(reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base));
  free(data_addr);
  model.runtime_param_.mem_base = 0U;

  input_data.blobs.pop_back();
  ZeroCopyOffset zero_copy_offset_input0;
  std::vector<std::pair<int64_t, uint64_t>> zero_data_info_input0;
  zero_data_info_input0.emplace_back(1,0x1111);
  zero_copy_offset_input0.data_info_ = zero_data_info_input0;
  model.input_data_info_[0] = zero_copy_offset_input0;
  EXPECT_NE(model.CopyInputData(input_data), SUCCESS);
}

TEST_F(UtestDavinciModel, InitRealSizeAndShapeInfo_succ1) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  OpDescPtr op_output = CreateOpDesc("output_ascend_mbatch_batch_1", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({1024});
  NodePtr node_output = graph->AddNode(op_output);
  EXPECT_EQ(model.InitRealSizeAndShapeInfo(graph, node_output), SUCCESS);
}

TEST_F(UtestDavinciModel, InitRealSizeAndShapeInfo_succ2) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_graph");

  OpDescPtr data1 = CreateOpDesc("data1", DATA);
  GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
  data1->AddInputDesc(shape_desc);
  data1->AddOutputDesc(shape_desc);
  NodePtr data1_node = graph->AddNode(data1);

  OpDescPtr case_node = CreateOpDesc("case1", CASE);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  case_node->AddInputDesc(tensor);
  case_node->AddOutputDesc(tensor);
  NodePtr case1_node = graph->AddNode(case_node);

  OpDescPtr output = CreateOpDesc("output1", NETOUTPUT);
  output->AddInputDesc(tensor);
  output->SetSrcName( { "case1" } );
  output->SetSrcIndex( { 0 } );
  NodePtr output_node = graph->AddNode(output);

  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), case1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(case1_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  (void)AttrUtils::SetBool(case_node, ATTR_INSERT_BY_MBATCH, true);

  model.is_getnext_sink_dynamic_ = false;
  model.run_context_.dynamic_shape_dims = {{1}, {2}, {3}, {4}};
  auto ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  // GetGearAndRealOutShapeInfo without ATTR_NAME_DYNAMIC_OUTPUT_DIMS
  EXPECT_EQ(ret, SUCCESS);
  std::vector<string> dynamic_output_dims = {"0,0,1,1,0,2,2,0,4,3,0,8"};
  (void)AttrUtils::SetListStr(output_node->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_dims);
  ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(model.InitCase(case_node), SUCCESS);
}

TEST_F(UtestDavinciModel, InitRealSizeAndShapeInfo_succ3) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("test_graph");

  OpDescPtr data1 = CreateOpDesc("data1", DATA);
  GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
  data1->AddInputDesc(shape_desc);
  data1->AddOutputDesc(shape_desc);
  NodePtr data1_node = graph->AddNode(data1);

  OpDescPtr shape_node = CreateOpDesc("ascend_mbatch_get_dynamic_dims_node", GETDYNAMICDIMS);
  GeTensorDesc in_tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_tensor(GeShape({4,3}), FORMAT_NCHW, DT_FLOAT);
  shape_node->AddInputDesc(in_tensor);
  shape_node->AddOutputDesc(out_tensor);
  NodePtr get_dynamic_dims_node = graph->AddNode(shape_node);

  OpDescPtr output = CreateOpDesc("output1", NETOUTPUT);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  output->AddInputDesc(tensor);
  output->SetSrcName( { "data1", "ascend_mbatch_get_dynamic_dims_node" } );
  output->SetSrcIndex( { 0, 1 } );
  NodePtr output_node = graph->AddNode(output);
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(get_dynamic_dims_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(1));

  (void)AttrUtils::SetStr(output_node->GetOpDesc(), ATTR_ALL_GEARS_INFO, "1,3;;4,3;,3");

  model.is_getnext_sink_dynamic_ = true;
  model.is_online_infer_dynamic_ = false;
  auto ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  EXPECT_EQ(ret, SUCCESS);
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 4;
  ret = model.InitRealSizeAndShapeInfo(graph, output_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestDavinciModel, init_data_aipp_info) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);

  NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_dynamic));
  aipp_attr.SetAttr("related_input_rank", GeAttrValue::CreateFrom<int64_t>(0));
  aipp_attr.SetAttr("max_src_image_size", GeAttrValue::CreateFrom<int64_t>(2048));
  aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<int64_t>(1));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr));

  AippConfigInfo aipp_info;
  EXPECT_EQ(model.GetAippInfo(0, aipp_info), ACL_ERROR_GE_AIPP_NOT_EXIST);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetAippInfo(0, aipp_info), SUCCESS);
  EXPECT_EQ(aipp_info.aipp_mode, domi::AippOpParams_AippMode_dynamic);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_static) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);

  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "static_aipp");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp_index, 0xFFFFFFFFu);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_dynamic) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0
  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
  AttrUtils::SetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_releated) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);   // op_index 0
    AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
    AttrUtils::SetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp");
  }
  {
    OpDescPtr op_desc = CreateOpDesc("releated_aipp", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);   // op_index 1
  }

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITH_DYNAMIC_AIPP);
  EXPECT_EQ(aipp_index, 1);

  EXPECT_EQ(model.input_addrs_list_.size(), 2);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_dynamic_conf) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0
  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_conf");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), SUCCESS);
  EXPECT_EQ(aipp_type, DYNAMIC_AIPP_NODE);
  EXPECT_EQ(aipp_index, 0xFFFFFFFFU);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_hcom_nodes) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();  // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("hcom", HCOMALLREDUCE);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  op_desc->SetStreamId(0);

  std::vector<NamedAttrs> attached_stream_infos;
  NamedAttrs attrs0;
  NamedAttrs attrs1;
  attached_stream_infos.push_back(attrs0);
  attached_stream_infos.push_back(attrs1);
  AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
  op_desc->SetAttachedStreamIds({-1, 1});
  NodePtr node = graph->AddNode(op_desc);  // op_index 0

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_EQ(model.hcom_attach_streams_.size(), 1U);
  EXPECT_EQ(model.hcom_streams_.size(), 1U);
  EXPECT_EQ(model.hcom_attach_streams_.insert(1).second, false);
  EXPECT_EQ(model.hcom_streams_.insert(0).second, false);
}

TEST_F(UtestDavinciModel, init_data_aipp_dynamic_invalid) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0
  AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_invalid");

  InputAippType aipp_type;
  size_t aipp_index = 0;
  EXPECT_EQ(model.GetAippType(0, aipp_type, aipp_index), PARAM_INVALID);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), ACL_ERROR_GE_AIPP_MODE_INVALID);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_input_info_empty) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  std::vector<string> inputs = {};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = {};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  OriginInputInfo orig_input_info;
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), SUCCESS);
  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_input_info_normal) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  std::vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  OriginInputInfo orig_input_info;
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), SUCCESS);
  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_input_info_invalid) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  std::vector<string> inputs = { "NCHW:DT_FLOAT:TensorName" };     // Invalid
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  OriginInputInfo orig_input_info;
  EXPECT_EQ(model.GetOrigInputInfo(0, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), ACL_ERROR_GE_AIPP_MODE_INVALID);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetOrigInputInfo(3, orig_input_info), ACL_ERROR_GE_AIPP_NOT_EXIST);
  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

TEST_F(UtestDavinciModel, init_data_aipp_input_dims_normal) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();   // for CustAICPUKernelStore::GetCustAICPUKernelStore()
  model.runtime_param_.mem_base = 0x08000000;
  model.runtime_param_.mem_size = 5120000;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);   // op_index 0

  std::vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);

  std::vector<InputOutputDims> input_dims;
  std::vector<InputOutputDims> output_dims;
  EXPECT_EQ(model.GetAllAippInputOutputDims(0, input_dims, output_dims), ACL_ERROR_GE_AIPP_NOT_EXIST);

  model.ge_model_->SetGraph(graph);
  std::vector<NodePtr> variable_nodes;
  EXPECT_EQ(model.InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  EXPECT_EQ(model.GetAllAippInputOutputDims(0, input_dims, output_dims), SUCCESS);
  EXPECT_EQ(input_dims.size(), 1);
  EXPECT_EQ(output_dims.size(), 1);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_TRUE(model.op_list_.empty());
}

// test label_set_task Init
TEST_F(UtestDavinciModel, label_task_success) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor, 64);

  {
    OpDescPtr op_desc = CreateOpDesc("label_switch", LABELSWITCHBYINDEX);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
    EXPECT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, {0, 1}));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX));
    domi::LabelSwitchByIndexDef *label_task_def = task_def1->mutable_label_switch_by_index();
    label_task_def->set_op_index(op_desc->GetId());
    label_task_def->set_label_max(2);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_then", LABELSET);
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 1));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    domi::LabelSetDef *label_task_def = task_def1->mutable_label_set();
    label_task_def->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_goto", LABELGOTOEX);
    NodePtr node = graph->AddNode(op_desc);      // op_index = 2
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 2));

    domi::TaskDef *task_def2 = model_task_def->add_task();
    task_def2->set_stream_id(0);
    task_def2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_LABEL_GOTO));
    domi::LabelGotoExDef *label_task_def = task_def2->mutable_label_goto_ex();
    label_task_def->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_else", LABELSET);
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 0));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    domi::LabelSetDef *label_task_def = task_def1->mutable_label_set();
    label_task_def->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_leave", LABELSET);
    NodePtr node = graph->AddNode(op_desc);  // op_index = 4
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 2));

    domi::TaskDef *task_def1 = model_task_def->add_task();
    task_def1->set_stream_id(0);
    task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    domi::LabelSetDef *label_task_def = task_def1->mutable_label_set();
    label_task_def->set_op_index(op_desc->GetId());
  }

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 3));
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_EQ(model.input_addrs_list_.size(), 0);
  EXPECT_EQ(model.output_addrs_list_.size(), 0);
  EXPECT_EQ(model.task_list_.size(), 5);
}

TEST_F(UtestDavinciModel, LoadWithQueue_fail_with_diff_args) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  QueueAttrs inputQueue2 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.input_queue_attrs_.emplace_back(inputQueue2);
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  ModelQueueParam queue_param;
  queue_param.need_check_inputs = true;
  queue_param.need_model_config = true;
  queue_param.mark_dump_step = true;
  queue_param.io_with_tensor_desc = true;
  queue_param.copy_inputs_for_non_zero_copy = true;
  model.SetModelQueueParam(queue_param);
  EXPECT_EQ(model.LoadWithQueue(), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID); // input queue size mismatch
  EXPECT_EQ(model.input_data_info_.size(), 0);
  ZeroCopyOffset zero_copy_offset;
  model.input_data_info_[0] = zero_copy_offset;
  model.input_data_info_[1] = zero_copy_offset;
  QueueAttrs outputQueue1 = {0, 0, 0, 0U};
  QueueAttrs outputQueue2 = {1, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(outputQueue1);
  model.output_queue_attrs_.emplace_back(outputQueue2);
  model.output_data_info_[0] = zero_copy_offset;
  EXPECT_EQ(model.LoadWithQueue(), ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID); // output queue size mismatch
  EXPECT_EQ(model.output_data_info_.size(), 1);
  model.output_data_info_[1] = zero_copy_offset;
  EXPECT_EQ(model.LoadWithQueue(), INTERNAL_ERROR); // AddHeadStream
  EXPECT_EQ(model.active_stream_list_.size(), 0);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
}

TEST_F(UtestDavinciModel, LoadWithQueue_WithDummyQ) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  ZeroCopyOffset zero_copy_offset;
  QueueAttrs outputQueue1 = {UINT32_MAX, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(outputQueue1);
  model.output_data_info_[0] = zero_copy_offset;
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  ModelQueueParam queue_param;
  queue_param.need_check_inputs = true;
  queue_param.need_model_config = true;
  queue_param.mark_dump_step = true;
  queue_param.io_with_tensor_desc = true;
  queue_param.copy_inputs_for_non_zero_copy = true;
  model.SetModelQueueParam(queue_param);
  EXPECT_EQ(model.LoadWithQueue(), SUCCESS); // AddHeadStream
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, LoadWithQueue_ReportStatus_succ) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  ZeroCopyOffset zero_copy_offset;
  QueueAttrs outputQueue1 = {UINT32_MAX, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(outputQueue1);
  model.output_data_info_[0] = zero_copy_offset;
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  model.need_report_status_ = true;
  ModelQueueParam queue_param;
  queue_param.need_check_inputs = true;
  queue_param.need_model_config = true;
  queue_param.mark_dump_step = true;
  queue_param.io_with_tensor_desc = true;
  queue_param.copy_inputs_for_non_zero_copy = true;
  model.SetModelQueueParam(queue_param);
  EXPECT_EQ(model.LoadWithQueue(), SUCCESS);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, LoadWithQueue_ReportStatus_fail) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  ZeroCopyOffset zero_copy_offset;
  QueueAttrs outputQueue1 = {UINT32_MAX, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(outputQueue1);
  model.output_data_info_[0] = zero_copy_offset;
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  model.need_report_status_ = true;
  g_runtime_stub_mock = "rtMalloc";
  ModelQueueParam queue_param;
  queue_param.need_check_inputs = true;
  queue_param.need_model_config = true;
  queue_param.mark_dump_step = true;
  queue_param.io_with_tensor_desc = true;
  queue_param.copy_inputs_for_non_zero_copy = true;
  model.SetModelQueueParam(queue_param);
  EXPECT_NE(model.LoadWithQueue(), SUCCESS);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, LoadWithQueue_ControlInput) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  EXPECT_EQ(model.LoadWithQueue(), SUCCESS);
  EXPECT_TRUE(model.use_control_input_queue_);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, LoadWithQueue_ControlOutput) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  QueueAttrs outputQueue1 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.output_queue_attrs_.emplace_back(outputQueue1);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  EXPECT_EQ(model.LoadWithQueue(), SUCCESS);
  EXPECT_TRUE(model.use_control_output_queue_);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, LoadWithQueue_ControlOutputWithDummyQ) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  QueueAttrs outputQueue1 = {UINT32_MAX, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(outputQueue1);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  EXPECT_EQ(model.LoadWithQueue(), SUCCESS);
  EXPECT_TRUE(model.use_control_output_queue_);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, LoadWithQueue_HWQ_ReportStatus_succ) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  model.is_hw_q_ = true;
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  QueueAttrs outputQueue1 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.output_queue_attrs_.emplace_back(outputQueue1);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  model.need_report_status_ = true;
  ZeroCopyOffset zero_copy_offset;
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data;
  virtual_addr_out_data[0x1111].emplace_back(0x1111111);
  zero_copy_offset.outside_addrs_.emplace_back(virtual_addr_out_data);
  model.input_data_info_[0] = zero_copy_offset;
  model.output_data_info_[0] = zero_copy_offset;
  ModelQueueParam queue_param;
  queue_param.need_check_inputs = true;
  queue_param.need_model_config = true;
  queue_param.mark_dump_step = true;
  queue_param.io_with_tensor_desc = true;
  queue_param.copy_inputs_for_non_zero_copy = true;
  model.SetModelQueueParam(queue_param);
  EXPECT_EQ(model.LoadWithHardwareQueue(), SUCCESS);
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, CpuModelDequeue_WithInputAlignAttrs) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs input_queue_0 = {0, 0, 0, 0U};
  QueueAttrs input_queue_1 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(input_queue_0);
  model.input_queue_attrs_.emplace_back(input_queue_1);
  QueueAttrs output_queue_1 = {UINT32_MAX, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(output_queue_1);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  model.model_queue_param_.input_align_attrs = {.align_max_cache_num = 4,
                                                .align_timeout = 200,
                                                .drop_when_not_align = true};

  EXPECT_EQ(model.CpuModelDequeue(), SUCCESS);
  EXPECT_EQ(model.input_mbuf_list_.size(), 2);
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, Helper_CopyInput_succ) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  QueueAttrs inputQueue2 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.input_queue_attrs_.emplace_back(inputQueue2);
  model.input_fusion_offsets_.resize(model.input_queue_attrs_.size());
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  EXPECT_EQ(model.input_data_info_.size(), 0);
  ZeroCopyOffset zero_copy_offset_input0;
  std::vector<std::pair<int64_t, uint64_t>> zero_data_info_input0;
  zero_data_info_input0.emplace_back(1,0x1111);
  zero_copy_offset_input0.data_info_ = zero_data_info_input0;
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data;
  virtual_addr_out_data[0x1111].emplace_back(0x1111111);
  zero_copy_offset_input0.outside_addrs_.emplace_back(virtual_addr_out_data);
  model.input_data_info_[0] = zero_copy_offset_input0;

  ZeroCopyOffset zero_copy_offset_input1;
  std::vector<std::pair<int64_t, uint64_t>> zero_data_info_input1;
  zero_data_info_input1.emplace_back(1,0x2222);
  zero_copy_offset_input1.data_info_ = zero_data_info_input1;
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data1;
  virtual_addr_out_data1[0x2222].emplace_back(0x2111111);
  zero_copy_offset_input1.outside_addrs_.emplace_back(virtual_addr_out_data1);
  model.input_data_info_[1] = zero_copy_offset_input1;
  model.input_mbuf_list_.emplace_back(0x888);
  model.input_mbuf_list_.emplace_back(0x999);
  model.copy_only_addrs_.Insert(0x2222);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  EXPECT_EQ(model.CpuInputCopyProcess(), SUCCESS); // AddHeadStream
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, Helper_CopyInput_with_desc) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  EXPECT_EQ(model.input_data_info_.size(), 0);

  RuntimeTensorDesc runtime_desc;
  runtime_desc.data_addr = 0x11111;
  ZeroCopyOffset zero_copy_offset_input0;
  // 有tensordesc的inputdata实际存储的时desc数据
  std::vector<std::pair<int64_t, uint64_t>> zero_data_info_input0;
  zero_data_info_input0.emplace_back(10, PtrToValue(&runtime_desc));
  zero_copy_offset_input0.data_info_ = zero_data_info_input0;
  // 有tensordesc时候outside_addrs_ 维度为2 分别时desc的outside 和data的outside
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_desc;
  virtual_addr_out_desc[PtrToValue(&runtime_desc)].emplace_back(0x10101010);
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data;
  virtual_addr_out_data[0x11111].emplace_back(0x1111111);
  zero_copy_offset_input0.outside_addrs_.emplace_back(virtual_addr_out_desc);
  zero_copy_offset_input0.outside_addrs_.emplace_back(virtual_addr_out_data);
  model.input_data_info_[0] = zero_copy_offset_input0;

  model.input_mbuf_list_.emplace_back(0x888);
  model.copy_only_addrs_.Insert(0x10101010);
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};
  EXPECT_EQ(model.CpuInputCopyProcess(), SUCCESS); // AddHeadStream
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, Helper_CopyInput_Fail) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  QueueAttrs inputQueue2 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.input_queue_attrs_.emplace_back(inputQueue2);
  model.input_fusion_offsets_.resize(model.input_queue_attrs_.size());
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};

  EXPECT_EQ(model.input_data_info_.size(), 0);
  EXPECT_EQ(model.CpuInputCopyProcess(), INTERNAL_ERROR);

  ZeroCopyOffset zero_copy_offset;
  model.input_data_info_[0] = zero_copy_offset;
  model.input_data_info_[1] = zero_copy_offset;
  EXPECT_EQ(model.CpuInputCopyProcess(), INTERNAL_ERROR);


  ZeroCopyOffset zero_copy_offset_input0;
  std::vector<std::pair<int64_t, uint64_t>> zero_data_info_input0;
  zero_data_info_input0.emplace_back(1,0x1111);
  zero_copy_offset_input0.data_info_ = zero_data_info_input0;
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data0;
  virtual_addr_out_data0[0x1111].emplace_back(0x1111111);
  zero_copy_offset_input0.outside_addrs_.emplace_back(virtual_addr_out_data0);
  model.input_data_info_[0] = zero_copy_offset_input0;
  model.copy_only_addrs_.Insert(0x2222);

  ZeroCopyOffset zero_copy_offset_input1;
  std::vector<std::pair<int64_t, uint64_t>> zero_data_info_input1;
  zero_data_info_input1.emplace_back(1,0x2222);
  zero_copy_offset_input1.data_info_ = zero_data_info_input1;
  std::map<uintptr_t, std::vector<uintptr_t>> virtual_addr_out_data1;
  virtual_addr_out_data1[0x2222].emplace_back(0x1111111);
  zero_copy_offset_input1.outside_addrs_.emplace_back(virtual_addr_out_data1);
  model.input_data_info_[1] = zero_copy_offset_input1;
  EXPECT_EQ(model.CpuInputCopyProcess(), INTERNAL_ERROR);

  model.input_mbuf_list_.emplace_back(0x888);
  model.input_mbuf_list_.emplace_back(0x999);

  EXPECT_EQ(model.CpuInputCopyProcess(), SUCCESS);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

TEST_F(UtestDavinciModel, Helper_InputShapeValidate) {
  DavinciModel model(0, nullptr);
  ModelQueueParam queue_param;
  queue_param.need_check_inputs = true;
  queue_param.need_model_config = true;
  queue_param.mark_dump_step = true;
  queue_param.io_with_tensor_desc = true;
  queue_param.copy_inputs_for_non_zero_copy = true;
  model.SetModelQueueParam(queue_param);
  EXPECT_EQ(model.CpuStaticInputShapeValidate(), SUCCESS);
  model.aicpu_resources_.static_model_shape_config_result_ = true;
  InputOutputDescInfo input_info = {};
  input_info.shape_info.dims = {-1,-1};
  model.origin_input_descs_.emplace_back(input_info);
  EXPECT_EQ(model.CpuStaticInputShapeValidate(), SUCCESS);

  model.origin_input_descs_.clear();
  input_info.shape_info.dims = {1,1};
  model.origin_input_descs_.emplace_back(input_info);
  model.ge_model_ = MakeShared<GeModel>();
  QueueAttrs inputQueue1 = {0, 0, 0, 0U};
  QueueAttrs inputQueue2 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.input_queue_attrs_.emplace_back(inputQueue2);
  model.input_fusion_offsets_.resize(model.input_queue_attrs_.size());
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  rtStream_t active_stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  rtStreamCreate(&active_stream, 0);
  model.active_stream_list_ = {active_stream};

  EXPECT_EQ(model.input_data_info_.size(), 0);
  EXPECT_EQ(model.CpuStaticInputShapeValidate(), INTERNAL_ERROR);

  ZeroCopyOffset zero_copy_offset;
  model.input_data_info_[0] = zero_copy_offset;
  model.input_data_info_[1] = zero_copy_offset;
  EXPECT_EQ(model.CpuStaticInputShapeValidate(), INTERNAL_ERROR);

  model.input_mbuf_list_.emplace_back(0x888);
  model.input_mbuf_list_.emplace_back(0x999);

  EXPECT_EQ(model.CpuStaticInputShapeValidate(), SUCCESS);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  rtStreamDestroy(active_stream);
}

class ClassTest {
public:
    virtual ~ClassTest() {}

    virtual int func0() {
        return 0;
    }
    virtual int func1(int a) {
        return a;
    }
    virtual int func2(int a, int b) {
        return a + b;
    }
    virtual int func3(int a, int b) const {
        return a - b;
    }
};

class MockTest : public ClassTest {
public:
    MOCK_METHOD0(func0, int());
    MOCK_METHOD1(func1, int(int a));
    MOCK_METHOD2(func2, int(int a, int b));

    MOCK_CONST_METHOD2(func3, int(int a, int b));
};

TEST_F(UtestDavinciModel, simple_test_gmock) {
    MockTest mock_stub;

    ON_CALL(mock_stub, func0()).WillByDefault(testing::Return(250));
    EXPECT_EQ(mock_stub.func0(), 250);
    EXPECT_EQ(mock_stub.func0(), 250);
    EXPECT_EQ(mock_stub.func0(), 250);

    EXPECT_CALL(mock_stub, func1(testing::_)).Times(2).WillOnce(testing::Return(1024)).WillOnce(testing::Return(250));
    EXPECT_EQ(mock_stub.func1(1), 1024);
    EXPECT_EQ(mock_stub.func1(1), 250);

    EXPECT_CALL(mock_stub, func2(testing::_, 5)).Times(3).WillRepeatedly(testing::Return(1023));
    EXPECT_EQ(mock_stub.func2(1, 5), 1023);
    EXPECT_EQ(mock_stub.func2(2, 5), 1023);
    EXPECT_EQ(mock_stub.func2(3, 5), 1023);
}

TEST_F(UtestDavinciModel, NnExecute) {
 TestNnExecute();
}

TEST_F(UtestDavinciModel, NnExecuteWithHostPlsModelIo) {
 TestNnExecuteWithHostPlsModelIo();
}

TEST_F(UtestDavinciModel, NnExecuteWithHostPlsModelIo_ZeroCopyReuse) {
 TestNnExecuteWithHostPlsModelIo_ZeroCopyReuse();
}

TEST_F(UtestDavinciModel, NnExecute_ReportProfiling_ProfilingOn) {
  ge::diagnoseSwitch::DisableProfiling();
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime});
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestNnExecute, 6, 1, 4, 0);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(UtestDavinciModel, NnExecuteWithGertTensor_ReportProfiling_ProfilingOn) {
  ge::diagnoseSwitch::DisableProfiling();
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime});
  EXPECT_DefaultProfilingTestWithExpectedCallTimes(TestNnExecuteWithGertTensor, 4, 1, 4, 0);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(UtestDavinciModel, test_clear_dfx_cache) {
  ge::diagnoseSwitch::DisableProfiling();
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kTaskTime});
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  {
    DavinciModel model(0, nullptr);
    BuildDavinciModel(model);
    model.SetClearDfxCacheFlagAfterInit(true);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_TRUE(model.node_basic_infos_.empty());
    EXPECT_FALSE(model.exception_dumper_.op_desc_info_.empty());
  }
  {
    DavinciModel model(0, nullptr);
    BuildDavinciModel(model);
    model.SetClearDfxCacheFlagAfterInit(false);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_FALSE(model.node_basic_infos_.empty());
    EXPECT_FALSE(model.exception_dumper_.op_desc_info_.empty());
  }
  ge::diagnoseSwitch::DisableProfiling();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  {
    DavinciModel model(0, nullptr);
    BuildDavinciModel(model);
    model.SetClearDfxCacheFlagAfterInit(true);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_TRUE(model.node_basic_infos_.empty());
    EXPECT_TRUE(model.exception_dumper_.op_desc_info_.empty());
  }
  {
    DavinciModel model(0, nullptr);
    BuildDavinciModel(model);
    model.SetClearDfxCacheFlagAfterInit(false);
    EXPECT_EQ(model.Init(), SUCCESS);
    EXPECT_FALSE(model.node_basic_infos_.empty());
    EXPECT_FALSE(model.exception_dumper_.op_desc_info_.empty());
  }
}

TEST_F(UtestDavinciModel, NnExecute_multi_batch) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    std::string shapes = "2,2,-1";
    (void)AttrUtils::SetStr(op_desc, ATTR_ALL_GEARS_INFO, shapes);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("batch", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);
  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;
  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), INTERNAL_ERROR);

  InputData input_data_empty;
  OutputData output_data_empty;
  for (auto &item : outputs) {
    GeTensor ge_tensor;
    TensorTransUtils::GertTensor2GeTensor(item, ge_tensor);
    output_tensor.emplace_back(std::move(ge_tensor));
  }
  input_tensor = output_tensor;
  EXPECT_EQ(model.NnExecute(stream, false, input_data_empty, output_data_empty,
            input_tensor, output_tensor), INTERNAL_ERROR);
}

TEST_F(UtestDavinciModel, NnExecute_multi_batch_with_gerttensor) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    std::string shapes = "2,2,-1";
    (void)AttrUtils::SetStr(op_desc, ATTR_ALL_GEARS_INFO, shapes);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("batch", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);
  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;
  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  input_tensor.resize(2);
  unique_ptr<uint8_t[]> data_buf_1(new (std::nothrow) uint8_t[512]);
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_1.get()};
  unique_ptr<uint8_t[]> data_buf_2(new (std::nothrow) uint8_t[512]);
  input_tensor[1] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_2.get()};
  unique_ptr<uint8_t[]> data_buf_3(new (std::nothrow) uint8_t[512]);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_3.get()};
  EXPECT_EQ(model.NnExecute(stream, true, input_tensor, output_tensor), INTERNAL_ERROR);

  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  output_data.blobs[0].placement = 0; // dev mem
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  MemAllocationSlice mem_allocation_slice;
  mem_allocation_slice.id = 0;
  mem_allocation_slice.offset = 0;
  mem_allocation_slice.data_size = 64;
  model.output_indexes_to_copy_info_.clear();
  model.input_indexes_to_copy_info_.clear();
  model.output_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  model.zero_copy_input_indexes_.clear();
  model.input_index_to_allocation_ids_.clear();
  model.zero_copy_input_indexes_.emplace_back(0);
  model.input_index_to_allocation_ids_.emplace_back(0);

  model.zero_copy_output_indexes_.clear();
  model.output_index_to_allocation_ids_.clear();
  model.zero_copy_output_indexes_.emplace_back(0);
  model.output_index_to_allocation_ids_.emplace_back(0);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = 64;
  mem_allocation0.tensor_size = 64;
  mem_allocation0.logical_addr = 100;

  MemAllocation mem_allocation1 = {};
  mem_allocation0.data_size = 64;
  mem_allocation0.tensor_size = 64;
  mem_allocation0.logical_addr = 200;

  model.logical_mem_allocations_.clear();
  model.logical_mem_allocations_.emplace_back(mem_allocation0);
  model.logical_mem_allocations_.emplace_back(mem_allocation1);

  EXPECT_EQ(model.UpdateAllNodeArgs(input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  model.input_index_to_allocation_ids_.clear();
  model.input_index_to_allocation_ids_.emplace_back(0);
  model.output_index_to_allocation_ids_.clear();
  model.output_index_to_allocation_ids_.emplace_back(0);
  EXPECT_EQ(model.UpdateAllNodeArgs(input_data, output_data,
            input_tensor, output_tensor), SUCCESS);

  InputData input_data_kHost;
  OutputData output_data_kHost;
  DataBuffer input_datas;
  DataBuffer input_datas_1;
  DataBuffer output_datas;
  unique_ptr<uint8_t[]> data_buf_11(new (std::nothrow) uint8_t[512]);
  unique_ptr<uint8_t[]> data_buf_12(new (std::nothrow) uint8_t[512]);
  unique_ptr<uint8_t[]> data_buf_13(new (std::nothrow) uint8_t[512]);
  input_datas.length = 512;
  input_datas.data = data_buf_11.get();

  input_datas_1.length = 512;
  input_datas_1.data = data_buf_13.get();

  output_datas.length = 512;
  output_datas.data = data_buf_12.get();

  input_data_kHost.blobs.emplace_back(input_datas_1);
  output_data_kHost.blobs.emplace_back(output_datas);
  output_data_kHost.blobs[0].placement = 0; // host mem
  input_data_kHost.blobs[0].placement = 0; // host mem
  model.output_indexes_to_copy_info_.clear();
  model.input_indexes_to_copy_info_.clear();
  model.output_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  EXPECT_EQ(model.UpdateAllNodeArgs(input_data_kHost, output_data_kHost,
            input_tensor, output_tensor), SUCCESS);
  std::vector<GeTensor> input_ge_tensor;
  std::vector<GeTensor> output_ge_tensor;
  model.refreshable_input_index_and_allocation_ids_.emplace_back(std::make_pair(0,0));
  model.refreshable_output_index_and_allocation_ids_.emplace_back(std::make_pair(0,1));
  // EXPECT_EQ(model.UpdateAllNodeArgsByAddrRefreshOp(input_data_kHost, output_data_kHost,
  //           input_ge_tensor, output_ge_tensor), 1343225857);
  input_data_kHost.blobs.emplace_back(input_datas);
  EXPECT_NE(model.UpdateAllNodeArgs(input_data_kHost, output_data_kHost,
            input_tensor, output_tensor), SUCCESS);
}

TEST_F(UtestDavinciModel, UpdateAllNodeArgs_with_frozen_input) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    std::string shapes = "2,2,-1";
    (void)AttrUtils::SetStr(op_desc, ATTR_ALL_GEARS_INFO, shapes);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("batch", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }
  dlog_setlevel(0,0,0);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);
  InputData input_data;
  OutputData output_data;

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  input_tensor.resize(1);
  unique_ptr<uint8_t[]> data_buf_1(new (std::nothrow) uint8_t[512]);
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_1.get()};
  unique_ptr<uint8_t[]> data_buf_3(new (std::nothrow) uint8_t[512]);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_3.get()};

  MemAllocationSlice mem_allocation_slice;
  mem_allocation_slice.id = 0;
  mem_allocation_slice.offset = 0;
  mem_allocation_slice.data_size = 64;
  model.output_indexes_to_copy_info_.clear();
  model.input_indexes_to_copy_info_.clear();
  model.output_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  model.zero_copy_input_indexes_.clear();
  model.input_index_to_allocation_ids_.clear();
  model.zero_copy_input_indexes_.emplace_back(0);
  model.input_index_to_allocation_ids_.emplace_back(0);

  unordered_set<uint32_t> frozen_input_indexes;
  frozen_input_indexes.insert(0);
  model.refreshable_input_index_and_allocation_ids_.emplace_back(std::make_pair(0,0));
  EXPECT_EQ(model.zero_copy_input_indexes_no_frozen_.size(), 0);
  EXPECT_EQ(model.refreshable_input_index_no_frozen_and_allocation_ids_.size(), 0);


  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = 512;
  mem_allocation0.tensor_size = 128;
  mem_allocation0.logical_addr = PtrToValue(output_tensor[0].GetAddr());

  model.logical_mem_allocations_.clear();
  model.logical_mem_allocations_.emplace_back(mem_allocation0);
  uint32_t ret_up = 0;
  auto id_to_up = model.args_manager_.GetId2Policy();

  model.refreshable_input_index_and_allocation_ids_.clear();
  model.refreshable_input_index_and_allocation_ids_.emplace_back(std::make_pair(0,1));
  model.input_index_to_active_mem_base_addrs_[0] = PtrToValue(input_tensor[0].GetAddr());
  model.output_index_to_active_mem_base_addrs_[0] = PtrToValue(output_tensor[0].GetAddr());
  EXPECT_EQ(model.ConstructZeroCopyIoActiveBaseAddrs(model.refreshable_input_index_and_allocation_ids_,
                                           {}, input_tensor, true, ret_up, id_to_up), SUCCESS);
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_1.get()};
  model.allocation_ids_to_active_base_addr_[1] = 0x1;
  EXPECT_EQ(model.ConstructZeroCopyIoActiveBaseAddrs(model.refreshable_input_index_and_allocation_ids_,
                                           {}, input_tensor, true, ret_up, id_to_up), SUCCESS);
  EXPECT_EQ(model.ConstructActiveMemBaseAddrsForKnownNode(ret_up, model.input_index_to_active_mem_base_addrs_,
                                              model.output_index_to_active_mem_base_addrs_), SUCCESS);

  model.input_index_to_allocation_ids_.clear();
  model.input_index_to_allocation_ids_.emplace_back(0);
  model.output_index_to_allocation_ids_.clear();
  model.output_index_to_allocation_ids_.emplace_back(0);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], PtrToValue(input_tensor[0].GetAddr()));

  model.input_index_to_active_mem_base_addrs_[0] = 0x1234;
  EXPECT_EQ(model.ConstructActiveMemBaseAddrsForKnownNode(ret_up, model.input_index_to_active_mem_base_addrs_,
                                              model.output_index_to_active_mem_base_addrs_), SUCCESS);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], 0x1234);

  model.input_index_to_active_mem_base_addrs_[0] = 0x4321;
  model.is_first_time_model_execute_ = true;
  EXPECT_EQ(model.ConstructActiveMemBaseAddrsForKnownNode(ret_up, model.input_index_to_active_mem_base_addrs_,
                                              model.output_index_to_active_mem_base_addrs_), SUCCESS);
  // 只刷新了fm，不刷新input，frozen生效
  // 先规避，后续更新用例
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[0], 0);

  model.input_index_to_active_mem_base_addrs_[0] = 0x1234;
  EXPECT_EQ(model.ConstructActiveMemBaseAddrsForKnownNode(ret_up, model.input_index_to_active_mem_base_addrs_,
                                              model.output_index_to_active_mem_base_addrs_), SUCCESS);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], 0x1234);
  model.allocation_ids_to_active_base_addr_[0] = 0x1234;
  EXPECT_EQ(model.ConstructActiveMemBaseAddrsForKnownNode(ret_up, model.input_index_to_active_mem_base_addrs_,
                                              model.output_index_to_active_mem_base_addrs_), SUCCESS);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[0], 0);
}

TEST_F(UtestDavinciModel, UpdateAllNodeArgs_with_GeTensor) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    std::string shapes = "2,2,-1";
    (void)AttrUtils::SetStr(op_desc, ATTR_ALL_GEARS_INFO, shapes);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("batch", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }
  dlog_setlevel(0,0,0);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;
  GeTensorDesc tensor_output(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  std::vector<GeTensor> ge_outputs;
  GeTensor ge_output_tensor;
  ge_output_tensor.SetTensorDesc(tensor_output);
  ge_outputs.emplace_back(ge_output_tensor);

  GeTensorDesc tensor_input(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  std::vector<GeTensor> ge_inputs;
  GeTensor ge_input_tensor;
  ge_input_tensor.SetTensorDesc(tensor_input);
  ge_inputs.emplace_back(ge_input_tensor);

  GeTensorDesc tensor_input1(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  GeTensor ge_input_tensor1;
  ge_input_tensor1.SetTensorDesc(tensor_input1);
  ge_inputs.emplace_back(ge_input_tensor1);

  model.allocation_ids_to_active_base_addr_[0] = 0x1234;
  model.allocation_ids_to_active_base_addr_[1] = 0x1234;
  model.refreshable_input_index_and_allocation_ids_.emplace_back(std::make_pair(1,1));
  EXPECT_EQ(model.UpdateAllNodeArgs({}, {},
            ge_inputs, ge_outputs), SUCCESS);

  InputData input_data;
  OutputData output_data;
  input_data.blobs.emplace_back(DataBuffer());
  input_data.blobs.emplace_back(DataBuffer());
  input_data.blobs[0].placement = kPlacementHost;
  input_data.blobs[1].placement = kPlacementHost;
  model.allocation_ids_to_active_base_addr_[1] = 0x1234;
  model.refreshable_input_index_and_allocation_ids_.emplace_back(std::make_pair(1,1));
  EXPECT_EQ(model.UpdateAllNodeArgs(input_data, output_data,
            ge_inputs, ge_outputs), SUCCESS);
  dlog_setlevel(0,3,0);
}

TEST_F(UtestDavinciModel, test_GetGeTensorBlobs) {
  GeModelPtr ge_model = ConstructGeModel(2560);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;

  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<GeTensor> output_tensor = {};
  for (auto &item : outputs) {
    GeTensor ge_tensor;
    TensorTransUtils::GertTensor2GeTensor(item, ge_tensor);
    output_tensor.emplace_back(std::move(ge_tensor));
  }
  InputData result_data;
  model.GetGeTensorBlobs(result_data, output_tensor);
  EXPECT_EQ(result_data.blobs.size(), 1);
}

TEST_F(UtestDavinciModel, InitAddrRefreshKernelBin_Test) {
  TBEHandleStore::GetInstance().bin_key_to_handle_.clear();
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  dlog_setlevel(0, 0, 0);
  std::string soc_version = "Ascend910B1";
  std::string lib_path = PathUtils::Join({test_work_dir, "runtime/lib64"});
  std::string lib_path_cann = PathUtils::Join({test_work_dir, "cann-8.5.0/lib64"});
  std::string lib_path_null = "";
  std::string lib_path_error = "runtime/lib64/UpdateModelParam";
  // set extend size static memory option
  std::map<std::string, std::string> graph_options;
  graph_options.emplace(make_pair(ge::SOC_VERSION, soc_version));


  EXPECT_EQ(model.InitAddrRefreshKernelBin(), SUCCESS);

  char *path = (char *)lib_path.c_str();
  char *path_null = (char *)lib_path_null.c_str();
  char *path_error = (char *)lib_path_error.c_str();
  char *path_cann = (char *)lib_path_cann.c_str();

  setenv("LD_LIBRARY_PATH", path_null, 1);
  EXPECT_EQ(model.InitAddrRefreshKernelBin(), SUCCESS);
  GetThreadLocalContext().SetGraphOption(graph_options);
  setenv("LD_LIBRARY_PATH", path_error, 1);
  EXPECT_EQ(model.InitAddrRefreshKernelBin(), SUCCESS);
  setenv("LD_LIBRARY_PATH", path_cann, 1);
  std::string save_path_cann = lib_path_cann + "/UpdateModelParam_dav_2201.o";
    DEF_GRAPH(g2) {
      CHAIN(NODE("cons2", "Const")->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file_cann = ToGeGraph(g2);
  graph_to_file_cann.SaveToFile(save_path_cann.c_str());
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_EQ(model.InitAddrRefreshKernelBin(), SUCCESS);

  std::string bin_handle_key = "UpdateModelParam_AicoreKernel";
  auto kernel_handles_manager = model.GetKernelHandlesManager(KernelHandleType::kAicore);
  EXPECT_NE(kernel_handles_manager->FindKernel(bin_handle_key), nullptr);
  kernel_handles_manager->ClearKernel();

  setenv("LD_LIBRARY_PATH", path, 1);
  std::string save_path = lib_path + "/UpdateModelParam_dav_2201.o";
  DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file = ToGeGraph(g1);
  graph_to_file.SaveToFile(save_path.c_str());
  EXPECT_EQ(model.InitAddrRefreshKernelBin(), SUCCESS);

  EXPECT_NE(kernel_handles_manager->FindKernel(bin_handle_key), nullptr);
  kernel_handles_manager->ClearKernel();
  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestDavinciModel, InitAddrRefreshKernelBin_Test_OverflowDump_Enabled) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  dlog_setlevel(0, 0, 0);
  std::string soc_version = "Ascend910B1";
  std::string lib_path = PathUtils::Join({test_work_dir, "runtime/lib64"});
  // set extend size static memory option
  std::map<std::string, std::string> graph_options;
  graph_options.emplace(make_pair(ge::SOC_VERSION, soc_version));

  char *path = (char *)lib_path.c_str();
  GetThreadLocalContext().SetGraphOption(graph_options);
  setenv("LD_LIBRARY_PATH", path, 1);
  std::string save_path = lib_path + "/UpdateModelParam_dav_2201.o";
  DEF_GRAPH(g1) {
      CHAIN(NODE("cons1", "Const")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    };
  ge::Graph graph_to_file = ToGeGraph(g1);
  graph_to_file.SaveToFile(save_path.c_str());

  gert::GlobalDumper::GetInstance()->SetEnableFlags(
  gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kOverflowDump}));
  EXPECT_EQ(model.Init(), SUCCESS);
  //EXPECT_EQ(model.InitAddrRefreshKernelBin(), SUCCESS);
  EXPECT_EQ(model.args_manager_.func_handle_, nullptr);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestDavinciModel, test_GetGeTensorBlobsWithGertTensor) {
  GeModelPtr ge_model = ConstructGeModel(2560);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;

  OutputData output_data;

  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  std::vector<uint8_t> output_data_1(96, 0xFF);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) output_data_1.data()};

  InputData result_data;
  model.GetGeTensorBlobs(result_data, output_tensor);
  EXPECT_EQ(result_data.blobs.size(), 1);

  std::vector<int64_t> output_tensor_dims;
  output_tensor_dims = model.GetTensorDims(output_tensor[0].GetStorageShape());
  EXPECT_EQ(output_tensor_dims[1], 4);
}

TEST_F(UtestDavinciModel, update_output_getensor_success) {
  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  std::vector<GeTensor> ge_outputs;
  GeTensor ge_tensor;
  ge_tensor.SetTensorDesc(tensor);
  ge_outputs.emplace_back(ge_tensor);

  DavinciModel model(0, nullptr);
  model.UpdateOutputTensorShape(ge_outputs);
  EXPECT_EQ(ge_outputs[0].GetTensorDesc().GetShape().GetDim(0), 1);

  GeTensorDesc tensor_desc(GeShape({8,2,2,2}), FORMAT_NCHW, DT_FLOAT);
  GeTensor ge_tensor1;
  ge_tensor1.SetTensorDesc(tensor_desc);
  ge_outputs.emplace_back(ge_tensor1);

  model.is_online_infer_dynamic_ = true;
  map<vector<int32_t>, vector<int64_t>> batch_2_dims;
  model.cur_dynamic_dims_ = {2,4,4,8};
  batch_2_dims[model.cur_dynamic_dims_] = {2,4,4,8};
  model.merge_nodes_gear_and_real_out_shape_info_[0] = batch_2_dims;

  // output 1 is notiling
  model.output_no_tiling_flag_ = {false, true};
  model.output_shape_info_ = {GeShape({1,4,4,8}), GeShape({4,2,2,2})};
  model.UpdateOutputTensorShape(ge_outputs);
  EXPECT_EQ(ge_outputs[0].GetTensorDesc().GetShape().GetDim(0), 2);
  EXPECT_EQ(ge_outputs[1].GetTensorDesc().GetShape().GetDim(0), 4);
}

TEST_F(UtestDavinciModel, update_output_gerttensor_success) {
  std::vector<gert::Tensor> ge_outputs;
  std::vector<uint8_t> output_data_1(96, 0xFF);
  ge_outputs.resize(1);
  ge_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                             {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                             gert::kOnDeviceHbm,                                // placement
                             ge::DT_FLOAT,                              // data type
                             (void *) output_data_1.data()};

  DavinciModel model(0, nullptr);
  model.UpdateOutputTensorShape(ge_outputs);
  gert::Shape shape = ge_outputs[0].GetStorageShape();
  EXPECT_EQ(shape.GetDim(0), 1);
  std::vector<gert::Tensor> ge_outputs_2;
  std::vector<uint8_t> output_data_2(96, 0xFF);
  ge_outputs_2.resize(2);
  ge_outputs_2[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                             {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                             gert::kOnDeviceHbm,                                // placement
                             ge::DT_FLOAT,                              // data type
                             (void *) output_data_1.data()};
  ge_outputs_2[1] = {{{8,2,2,2}, {8,2,2,2}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) output_data_2.data()};

  model.is_online_infer_dynamic_ = true;
  map<vector<int32_t>, vector<int64_t>> batch_2_dims;
  model.cur_dynamic_dims_ = {2,4,4,8};
  batch_2_dims[model.cur_dynamic_dims_] = {2,4,4,8};
  model.merge_nodes_gear_and_real_out_shape_info_[0] = batch_2_dims;

  // output 1 is notiling
  model.output_no_tiling_flag_ = {false, true};
  model.output_shape_info_ = {GeShape({1,4,4,8}), GeShape({4,2,2,2})};
  model.UpdateOutputTensorShape(ge_outputs_2);
  EXPECT_EQ(ge_outputs_2[0].GetStorageShape().GetDim(0), 2);
  EXPECT_EQ(ge_outputs_2[1].GetStorageShape().GetDim(0), 4);
}

TEST_F(UtestDavinciModel, update_io_addr_success) {
  DavinciModel model(0, nullptr);
  uint32_t task_id = 1;
  uint32_t stream_id = 2;
  model.fixed_mem_base_  = 0x22;
  model.mem_base_ = reinterpret_cast<uintptr_t>(&task_id);
  OpDescInfo op_desc_info;
  op_desc_info.op_name = "Save";
  op_desc_info.op_type = "Save";
  op_desc_info.id.task_id = 1;
  op_desc_info.id.stream_id = 2;
  op_desc_info.input_format = {FORMAT_NCHW};
  op_desc_info.input_shape = {{1}};
  op_desc_info.input_data_type = {DT_FLOAT};
  op_desc_info.input_addrs = {nullptr};
  op_desc_info.input_size = {2};
  op_desc_info.output_format = {FORMAT_NCHW};
  op_desc_info.output_shape = {{1}};
  op_desc_info.output_data_type = {DT_FLOAT};
  op_desc_info.output_addrs = {nullptr};
  op_desc_info.output_size = {2};
  model.exception_dumper_.op_desc_info_ = {op_desc_info};
  std::vector<void *> io_addr = {nullptr, nullptr};
  model.UpdateOpIOAddrs(task_id, stream_id, VPtrToValue(io_addr));
  EXPECT_EQ(op_desc_info.input_addrs.size(), 1U);
  EXPECT_EQ(op_desc_info.input_addrs[0U], nullptr);
  EXPECT_EQ(op_desc_info.output_addrs.size(), 1U);
  EXPECT_EQ(op_desc_info.output_addrs[0U], nullptr);
  model.mem_base_ = 0;
}
TEST_F(UtestDavinciModel, get_total_memsize_exclude_zero_copy) {
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 1024;
  model.runtime_param_.zero_copy_size = 2048;
  int64_t total_useful_size = 0;
  EXPECT_EQ(model.GetTotalMemSizeExcludeZeroCopy(total_useful_size), FAILED);
  EXPECT_TRUE(total_useful_size == 0);
  model.runtime_param_.zero_copy_size = 512;
  EXPECT_EQ(model.GetTotalMemSizeExcludeZeroCopy(total_useful_size), SUCCESS);
  EXPECT_EQ(total_useful_size, 512);
}

TEST_F(UtestDavinciModel, init_tbe_handle_offline) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  OpDescPtr op_desc = CreateOpDesc("reduce_mean", REDUCEMEAN);
  AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<uint32_t>(domi::ImplyType::TVM));

  // atomic
  AttrUtils::SetStr(op_desc, kAtomicPrefix + TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  (void)ge::AttrUtils::SetBool(op_desc, "need_gentask_atomic", true);
  std::string atomic_kernel_name = "_atomic_kernel_1";
  AttrUtils::SetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel_name);
  std::vector<char> buffer(512, 'c');
  TBEKernelPtr atomic_kernel_ptr = MakeShared<OpKernelBin>(atomic_kernel_name, std::move(buffer));
  model.ge_model_->GetTBEKernelStore().AddTBEKernel(atomic_kernel_ptr);
  // static kernel
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::string reduce_kernel_name = "_kernel_x124_reduce_mean_1";
  TBEKernelPtr tbe_kernel_ptr = MakeShared<OpKernelBin>(reduce_kernel_name, std::move(buffer));
  model.ge_model_->GetTBEKernelStore().AddTBEKernel(tbe_kernel_ptr);
  AttrUtils::SetStr(op_desc, "_kernelname", reduce_kernel_name);
  EXPECT_EQ(model.InitTbeHandle(op_desc), SUCCESS);

  // dynamic kernel
  OpDescPtr op_desc2 = CreateOpDesc("relu", RELU);
  AttrUtils::SetBool(op_desc2, "_kernel_list_first_name", true);
  AttrUtils::SetInt(op_desc2, ATTR_NAME_IMPLY_TYPE, static_cast<uint32_t>(domi::ImplyType::TVM));

  AttrUtils::SetStr(op_desc2, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::string relu_kernel_name = "_kernel_x124_relu_1";
  TBEKernelPtr dynamic_kernel_ptr = MakeShared<OpKernelBin>(relu_kernel_name, std::move(buffer));
  model.ge_model_->GetTBEKernelStore().AddTBEKernel(dynamic_kernel_ptr);
  AttrUtils::SetStr(op_desc2, "_kernelname", reduce_kernel_name);

  EXPECT_EQ(model.InitTbeHandle(op_desc2), SUCCESS);
}

TEST_F(UtestDavinciModel, run_with_task) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, g_local_call_back);
  model.SetId(1);
  model.isGraphLevelSat_ = true;
  (void)model.copy_host_input_indexes_.insert(0U);

  auto args = MakeShared<RunArgs>();
  model.data_inputer_.Push(args);
  model.data_inputer_.Push(args);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);
  model.has_output_node_ = true;
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  std::vector<int64_t> input_offset;
  input_offset.emplace_back(0);
  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->SetInputOffset(input_offset);
  model.InitOutputTensorInfo(op_desc);
  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, RunWithTask_GertTensor) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, g_local_call_back);
  model.SetId(1);
  model.isGraphLevelSat_ = true;
  (void)model.copy_host_input_indexes_.insert(0U);

  std::vector<gert::Tensor> inputs(1);
  auto args = std::make_shared<RunArgs>();
  args->input_tensor = std::move(inputs);
  model.data_inputer_.Push(args);
  model.data_inputer_.Push(args);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(RT_MODEL_TASK_PROFILER_TRACE);
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);
  model.has_output_node_ = true;
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  std::vector<int64_t> input_offset;
  input_offset.emplace_back(0);
  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->SetInputOffset(input_offset);
  model.InitOutputTensorInfo(op_desc);
  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, run_with_task_UpdateForExecute_failed) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, nullptr);
  model.SetId(1);
  model.isGraphLevelSat_ = true;

  auto data = MakeShared<RunArgs>();
  model.data_inputer_.Push(data);
  model.data_inputer_.Push(data);
  shared_ptr<DModelListener> listener = make_shared<DModelListener>();
  model.listener_ = listener;
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);
  model.has_output_node_ = true;
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  std::vector<int64_t> input_offset;
  input_offset.emplace_back(0);
  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->SetInputOffset(input_offset);
  model.InitOutputTensorInfo(op_desc);

  model.args_manager_.update_policies_to_model_data_[3] = nullptr;
  model.args_manager_.has_args_ =true;
  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_TRUE(listener->complete_flag_);
}

TEST_F(UtestDavinciModel, run_with_task_MallocPhysicalMemory_fail) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  model.isGraphLevelSat_ = true;
  model.support_extend_memory_full_ = true;
  auto data = MakeShared<RunArgs>();
  model.data_inputer_.Push(data);
  shared_ptr<DModelListener> listener(new DModelListener());
  model.listener_ = listener;
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);
  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;

  // 构造MallocPhysicalMemory失败
  model.is_first_execute_ = false;
  model.active_memorys_.push_back(std::make_pair(reinterpret_cast<uint8_t *>(110),20));
  auto mem_allocator =
      SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(model.session_id_, model.GetDeviceId());
  mem_allocator->expandable_memory_allocator_.active_memory_allocator_.virtual_memory_addr_base_ =
    reinterpret_cast<uint8_t *>(100);
  mem_allocator->expandable_memory_allocator_.active_memory_allocator_.virtual_memory_size_ = 50;
  mem_allocator->expandable_memory_allocator_.active_memory_allocator_.vapa_check_failed_ = true;
  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  EXPECT_TRUE(listener->complete_flag_);
  mem_allocator->expandable_memory_allocator_.active_memory_allocator_.virtual_memory_addr_base_ =
    reinterpret_cast<uint8_t *>(0);
  mem_allocator->expandable_memory_allocator_.active_memory_allocator_.virtual_memory_size_ = 0;
  mem_allocator->expandable_memory_allocator_.active_memory_allocator_.vapa_check_failed_ = false;
}

TEST_F(UtestDavinciModel, run_with_task_handle_input_data_fail) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  model.isGraphLevelSat_ = true;

  auto data = MakeShared<RunArgs>();
  model.data_inputer_.Push(data);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);
  model.is_online_infer_dynamic_ = true;
  model.is_getnext_sink_dynamic_ = false;
  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
}

TEST_F(UtestDavinciModel, NoNeedToCopyInputOutputWithGertTensor) {
  OutputData data;
  dlog_setlevel(0,0,0);
  data.blobs.emplace_back(DataBuffer());
  DavinciModel davinci_model(0, nullptr);
  MemAllocationSlice mem_allocation_slice;
  mem_allocation_slice.id = 0;
  mem_allocation_slice.offset = 0;
  mem_allocation_slice.data_size = 0;
  davinci_model.output_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  davinci_model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  davinci_model.output_no_tiling_flag_ = {false};
  davinci_model.output_shape_info_ = {GeShape({1,4,4,8})};
  davinci_model.has_output_node_ = true;
  std::vector<gert::Tensor> tensor;
  tensor.resize(1);
  tensor[0] = {{{1, 16, 256}, {1, 16, 256}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT16,                              // data type
                            (void *) 0xabc};
  EXPECT_EQ(davinci_model.CopyOutputData(tensor), SUCCESS);
  std::vector<DataBuffer> input_data {DataBuffer()};
  EXPECT_EQ(davinci_model.CopyInputForNoZeroCopy(input_data, davinci_model.input_indexes_to_copy_info_, tensor), SUCCESS);
  davinci_model.is_async_mode_ = true;
  EXPECT_EQ(davinci_model.CopyOutputData(tensor), SUCCESS);
  davinci_model.has_output_node_ = false;
  EXPECT_EQ(davinci_model.CopyOutputData(tensor), SUCCESS);
  davinci_model.host_pls_output_indexes_to_copy_info_[0] = {0, 0U, 16*256};
  davinci_model.has_output_node_ = true;
  EXPECT_NE(davinci_model.CopyOutputData(tensor), SUCCESS);

  uint8_t args1[70] = {123};
  std::vector<DataBuffer> input_data2 {DataBuffer(&args1, 70, false)};
  davinci_model.input_indexes_to_copy_info_.clear();
  mem_allocation_slice.data_size = 64;
  uint8_t buf[100];
  davinci_model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  //davinci_model.allocation_ids_to_active_base_addr_.emplace_back((uint64_t)buf);

  // uint64_t* active_base_addr = (uint64_t*)malloc(100);

  std::vector<uint64_t> active_base_addr_vec;
  for (size_t i = 0; i < davinci_model.logical_mem_allocations_.size(); i++) {
    active_base_addr_vec.emplace_back(davinci_model.allocation_ids_to_active_base_addr_[i]);
  }
  active_base_addr_vec.emplace_back((uint64_t)buf);
  davinci_model.allocation_ids_to_active_base_addr_ =
    reinterpret_cast<uint64_t*>(static_cast<void*>(active_base_addr_vec.data()));
  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = 32U;
  mem_allocation0.tensor_size = 2;
  mem_allocation0.logical_addr = 0x23;
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  EXPECT_EQ(davinci_model.CopyInputForNoZeroCopy(input_data2, davinci_model.input_indexes_to_copy_info_, tensor), SUCCESS);
  dlog_setlevel(0,3,0);
}

TEST_F(UtestDavinciModel, Test_CheckRtStreamSynchronize) {
  DavinciModel davinci_model(0, nullptr);
  rtError_t rt_ret = ACL_ERROR_RT_STREAM_SYNC_TIMEOUT;
  EXPECT_EQ(davinci_model.CheckRtStreamSynchronize(rt_ret), FAILED);
  rt_ret = RT_ERROR_NONE;
  EXPECT_EQ(davinci_model.CheckRtStreamSynchronize(rt_ret), SUCCESS);
  rt_ret = ACL_ERROR_RT_COPY_DATA;
  EXPECT_EQ(davinci_model.CheckRtStreamSynchronize(rt_ret), FAILED);
}

TEST_F(UtestDavinciModel, run_with_task_model_execute_fail) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  model.isGraphLevelSat_ = true;

  auto data = MakeShared<RunArgs>();
  model.data_inputer_.Push(data);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);
  mmSetEnv("CONSTANT_FOLDING_PASS_8", "mock_fail", 1);
  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  EXPECT_EQ(model.ModelRunStop(), SUCCESS);
  unsetenv("CONSTANT_FOLDING_PASS_8");
}

TEST_F(UtestDavinciModel, reinit_platform) {
  DavinciModel model(0, nullptr);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  model.ge_model_ = ge_model;
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor, 64);
  OpDescPtr op_ = CreateOpDesc("ffts_plus_fun", "ssss");
  NodePtr node = graph->AddNode(op_);  // op_index = 0

  // model.is_platform_infos_launched_ = false;
  void *platform_infos_addr{nullptr};
  model.LaunchPlatformInfos(platform_infos_addr, node);
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr, node), SUCCESS);
  // EXPECT_EQ(model.is_platform_infos_launched_, true);
  void *platform_infos_addr1{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr1, node), SUCCESS);
  // EXPECT_EQ(model.is_platform_infos_launched_, true);
  EXPECT_EQ(platform_infos_addr, platform_infos_addr1);
}

TEST_F(UtestDavinciModel, init_with_core_num_setting) {
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  DavinciModel model(0, nullptr);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.ge_model_ = ge_model;
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor, 64);

  OpDescPtr op_ = CreateOpDesc("ffts_plus_fun", "ssss");
  (void)ge::AttrUtils::SetStr(op_, "_op_aicore_num", "5");
  (void)ge::AttrUtils::SetStr(op_, "_op_vectorcore_num", "10");
  NodePtr node = graph->AddNode(op_);  // op_index = 0
  void *platform_infos_addr_5_10_1{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr_5_10_1, node), SUCCESS);
  void *platform_infos_addr_null_5_10_2{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr_null_5_10_2, node), SUCCESS);

  OpDescPtr op2_ = CreateOpDesc("ffts_plus_fun", "ssss");
  (void)ge::AttrUtils::SetStr(op2_, "_op_aicore_num", "6");
  (void)ge::AttrUtils::SetStr(op2_, "_op_vectorcore_num", "12");
  NodePtr node2 = graph->AddNode(op2_);
  void *platform_infos_addr_6_12_1{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr_6_12_1, node2), SUCCESS);
  OpDescPtr op3_ = CreateOpDesc("ffts_plus_fun", "ssss");
  (void)ge::AttrUtils::SetStr(op3_, "_op_aicore_num", "6");
  (void)ge::AttrUtils::SetStr(op3_, "_op_vectorcore_num", "12");
  NodePtr node3 = graph->AddNode(op3_);
  void *platform_infos_addr_6_12_2{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr_6_12_2, node3), SUCCESS);

  OpDescPtr op4_ = CreateOpDesc("ffts_plus_fun", "ssss");
  NodePtr node4 = graph->AddNode(op4_);
  void *platform_infos_global_1{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_global_1, node4), SUCCESS);
  OpDescPtr op5_ = CreateOpDesc("ffts_plus_fun", "ssss");
  NodePtr node5 = graph->AddNode(op5_);
  void *platform_infos_global_2{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_global_2, node5), SUCCESS);

  OpDescPtr op6_ = CreateOpDesc("ffts_plus_fun", "ssss");
  (void)ge::AttrUtils::SetStr(op6_, "_op_vectorcore_num", "12");
  NodePtr node6 = graph->AddNode(op6_);
  void *platform_infos_addr_null_12_1{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr_null_12_1, node6), SUCCESS);
  OpDescPtr op7_ = CreateOpDesc("ffts_plus_fun", "ssss");
  (void)ge::AttrUtils::SetStr(op7_, "_op_vectorcore_num", "12");
  NodePtr node7 = graph->AddNode(op7_);
  void *platform_infos_addr_null_12_2{nullptr};
  EXPECT_EQ(model.LaunchPlatformInfos(platform_infos_addr_null_12_2, node7), SUCCESS);

  EXPECT_EQ(platform_infos_addr_5_10_1, platform_infos_addr_null_5_10_2);
  EXPECT_EQ(platform_infos_addr_6_12_1, platform_infos_addr_6_12_2);
  EXPECT_EQ(platform_infos_global_1, platform_infos_global_2);
  EXPECT_EQ(platform_infos_addr_null_12_1, platform_infos_addr_null_12_2);

  EXPECT_NE(platform_infos_addr_5_10_1, platform_infos_addr_6_12_1);
  EXPECT_NE(platform_infos_addr_5_10_1, platform_infos_global_1);
  EXPECT_NE(platform_infos_addr_5_10_1, platform_infos_addr_null_12_1);
  EXPECT_NE(platform_infos_addr_6_12_1, platform_infos_global_1);
  EXPECT_NE(platform_infos_addr_6_12_1, platform_infos_addr_null_12_1);
  EXPECT_NE(platform_infos_global_1, platform_infos_addr_null_12_1);

  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

// test label_set_task Init
TEST_F(UtestDavinciModel, test_task_distribute_ffts_plus) {
  DavinciModel model(0, nullptr);
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("output");
  dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"ffts_plus_fun"});
  model.SetDumpProperties(dump_properties);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  model.ge_model_ = ge_model;
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor, 64);
  OpDescPtr op_ = CreateOpDesc("ffts_plus_fun", "ssss");
  NodePtr node = graph->AddNode(op_);  // op_index = 0
  model.op_list_[op_->GetId()] = op_;
  domi::TaskDef *task_def1 = model_task_def->add_task();
  task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def1->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(op_->GetId());
  model.task_list_.resize(1);
  model.task_list_[0] = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task_def1->type()));
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task = std::dynamic_pointer_cast<FftsPlusTaskInfo>(model.task_list_[0]);
  ffts_plus_task->davinci_model_ = &model;
  ffts_plus_task->op_desc_ = op_;
  ffts_plus_task->ffts_flus_args_helper_ = MakeUnique<FftsPlusArgsHelper>(model.GetRuntimeParam());
  {
    OpDescPtr op_desc = CreateOpDesc("label_then", LABELSET);
    //model.op_list_[op_desc->GetId()] = op_desc;
    NodePtr node1 = graph->AddNode(op_desc);  // op_index = 1
    domi::FftsPlusCtxDef* ctx_task_def = ffts_plus_task_def->add_ffts_plus_ctx();
    //ctx_task_def->set_op_index(op_desc->GetId());
    ctx_task_def->set_op_index(6);
    ctx_task_def->set_context_type(RT_CTX_TYPE_AICORE);
    ctx_task_def->set_context_id(0);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_else", LABELSET);
    NodePtr node1 = graph->AddNode(op_desc);  // op_index = 1
    domi::FftsPlusCtxDef* ctx_task_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx_task_def->set_op_index(op_desc->GetId());
    ctx_task_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
    ctx_task_def->set_context_id(1);
    model.op_list_[op_desc->GetId()] = op_desc;
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_leave", LABELSET);
    NodePtr node1 = graph->AddNode(op_desc);  // op_index = 1
    domi::FftsPlusCtxDef* ctx_task_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx_task_def->set_op_index(op_desc->GetId());
    ctx_task_def->set_context_type(RT_CTX_TYPE_LABEL);
    ctx_task_def->set_context_id(2);
    model.op_list_[op_desc->GetId()] = op_desc;
  }
  {
    OpDescPtr op_desc = CreateOpDesc("label_leat", LABELSET);
    NodePtr node1 = graph->AddNode(op_desc);  // op_index = 1
    domi::FftsPlusCtxDef* ctx_task_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx_task_def->set_op_index(op_desc->GetId());
    ctx_task_def->set_context_type(RT_CTX_TYPE_AICPU);
    ctx_task_def->set_context_id(3);
    model.op_list_[op_desc->GetId()] = op_desc;
  }

  {
    OpDescPtr op_desc = CreateOpDesc("label_leave", LABELSET);
    NodePtr node1 = graph->AddNode(op_desc);  // op_index = 1
    domi::FftsPlusCtxDef* ctx_task_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx_task_def->set_op_index(op_desc->GetId());
    ctx_task_def->set_context_type(RT_CTX_TYPE_AT_START);
    ctx_task_def->set_context_id(4);
    model.op_list_[op_desc->GetId()] = op_desc;
  }
  model.mdl_prof_.enable_flag = 1;
  model.mdl_prof_.task_type_to_distribute_time[0] = 100;
  model.mdl_prof_.task_type_to_distribute_num[0] = 1;
  EXPECT_EQ(model.DistributeTask(*model_task_def), SUCCESS);
  model.PrintfModelProfOfModelLoad();
}

TEST_F(UtestDavinciModel, TestAllocateResources) {
  DavinciModel model(0, nullptr);
  uint8_t weight[1024] {};
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_base = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_size = 1024;

  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_UINT32);
  TensorUtils::SetDataOffset(tensor_desc, 512);
  TensorUtils::SetSize(tensor_desc, sizeof(uint32_t));
  OpDescPtr op_desc = CreateOpDesc("Enqueue", "Enqueue");
  op_desc->AddInputDesc(tensor_desc);
  auto graph = MakeShared<ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  // no resource to create
  ASSERT_EQ(model.AllocateResource(*node), SUCCESS);

  // wrong attribute type
  AttrUtils::SetStr(op_desc, "_resource_list", "wrong-attr-type");
  ASSERT_EQ(model.AllocateResource(*node), INTERNAL_ERROR);

  // add queue resource
  std::vector<NamedAttrs> resources(1);
  NamedAttrs &queue_resource = resources.back();
  op_desc->DelAttr("_resource_list");
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);

  // no resource type
  ASSERT_EQ(model.AllocateResource(*node), PARAM_INVALID);

  // unsupported resource type
  AttrUtils::SetStr(queue_resource, "resource_type", "RES_SUPPORTED");
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);
  ASSERT_EQ(model.AllocateResource(*node), UNSUPPORTED);

  // missing queue name
  AttrUtils::SetStr(queue_resource, "resource_type", "RES_QUEUE");
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);
  ASSERT_EQ(model.AllocateResource(*node), PARAM_INVALID);

  // missing queue id idx
  AttrUtils::SetStr(queue_resource, "queue_name", "some_queue");
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);
  ASSERT_EQ(model.AllocateResource(*node), PARAM_INVALID);

  // no source node
  AttrUtils::SetInt(queue_resource, "queue_id_idx", 0);
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);
  ASSERT_EQ(model.AllocateResource(*node), PARAM_INVALID);

  // source is not a const
  OpDescPtr src_op_desc = CreateOpDesc("SrcNode", "NotConst");
  src_op_desc->AddOutputDesc(tensor_desc);
  auto src_node = graph->AddNode(src_op_desc);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ASSERT_EQ(model.AllocateResource(*node), PARAM_INVALID);

  // SUCCESS
  op_desc->SetIsInputConst(std::vector<bool>{true});
  ge::OpDescUtilsEx::SetType(src_op_desc, CONSTANT);
  ASSERT_EQ(model.AllocateResource(*node), SUCCESS);

  ASSERT_EQ(model.aicpu_resources_.aicpu_queues_.count("some_queue"), 1);
  uint32_t queue_id = model.aicpu_resources_.aicpu_queues_["some_queue"];
  uint32_t op_queue_id = *reinterpret_cast<uint32_t*>(weight + 512);
  ASSERT_EQ(queue_id, op_queue_id);

  // another node with same queue_name
  TensorUtils::SetDataOffset(*op_desc->MutableInputDesc(0), 256);
  ASSERT_EQ(model.AllocateResource(*node), SUCCESS);
  ASSERT_EQ(model.aicpu_resources_.aicpu_queues_.count("some_queue"), 1);
  op_queue_id = *reinterpret_cast<uint32_t*>(weight + 256);
  ASSERT_EQ(queue_id, op_queue_id);

  model.aicpu_resources_.ReleaseResources();
  ASSERT_TRUE(model.aicpu_resources_.aicpu_queues_.empty());
}

TEST_F(UtestDavinciModel, TestAllocateChannelResources) {
  DavinciModel model(0, nullptr);
  uint8_t weight[1024] {};
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_base = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_size = 1024;

  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_UINT32);
  TensorUtils::SetDataOffset(tensor_desc, 512);
  TensorUtils::SetSize(tensor_desc, sizeof(uint32_t));
  OpDescPtr op_desc = CreateOpDesc("Enqueue", "Enqueue");
  op_desc->AddInputDesc(tensor_desc);
  auto graph = MakeShared<ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  // add channel resource
  std::vector<NamedAttrs> resources(1);
  NamedAttrs &channel_resource = resources.back();
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);

  AttrUtils::SetStr(channel_resource, "resource_type", "RES_CHANNEL");
  AttrUtils::SetListNamedAttrs(op_desc, "_resource_list", resources);

  ASSERT_NE(model.AllocateResource(*node), SUCCESS);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  ASSERT_EQ(model.AllocateResource(*node), SUCCESS);
  ASSERT_EQ(model.aicpu_resources_.aicpu_channels_.size(), 1);

  // repeat execution
  rtStream_t stream2 = nullptr;
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  model.stream_list_ = { stream, stream2 };
  ASSERT_EQ(model.AllocateResource(*node), SUCCESS);
  ASSERT_EQ(model.aicpu_resources_.aicpu_channels_.size(), 1);

  model.aicpu_resources_.ReleaseResources();
  ASSERT_TRUE(model.aicpu_resources_.aicpu_channels_.empty());
}

TEST_F(UtestDavinciModel, TestAicpuModelConfig) {
  DavinciModel model(0, nullptr);
  uint8_t weight[1024] {};
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_base = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_size = 1024;
  model.SetNeedModelConfig(true);
  ASSERT_EQ(model.SetModelConfig(), SUCCESS);
}

TEST_F(UtestDavinciModel, TestAicpuModelShapeConfig) {
  DavinciModel model(0, nullptr);
  uint8_t weight[1024] {};
  model.weights_mem_base_ = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_base = reinterpret_cast<uintptr_t>(weight);
  model.runtime_param_.weight_size = 1024;
  auto old_origin_input_descs = model.origin_input_descs_;
  InputOutputDescInfo input0;
  input0.data_type = 0;
  input0.shape_info.dims = {1, 4, 4, 8};
  InputOutputDescInfo input1;
  input1.data_type = 0;
  input1.shape_info.dims = {2, 5, 6};
  model.origin_input_descs_ = {input0, input1};
  model.model_queue_param_.need_check_inputs = true;
  model.SetStaticModelShapeConfig();
  AiCpuResources::AiCpuModelShapeConfig config = {};
  config.model_id = model.model_id_;
  config.runtime_model_id = model.runtime_model_id_;
  ASSERT_EQ(model.aicpu_resources_.SetStaticModelShapeConfig(config, model.origin_input_descs_), SUCCESS);
}

TEST_F(UtestDavinciModel, save_profile_task_info) {
  dlog_setlevel(0,0,0);
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICORE));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  ffts_plus_task_info->prof_api_.begin_time = 1;
  ffts_plus_task_info->prof_api_.end_time = 2;
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;
  uint32_t op_impl_mode = 0x40;
  (void)AttrUtils::SetInt(op_desc, "_op_impl_mode_enum", op_impl_mode);
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  model.SaveProfilingInfoByContext(ffts_plus_task_def->ffts_plus_ctx(0), op_desc, ffts_plus_task_info->prof_api_, true);
  EXPECT_EQ(model.node_basic_infos_[0].node_basic_info.data.nodeBasicInfo.opFlag, 0x1);
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA));
  op_desc->SetStreamId(0);
  op_desc->SetAttachedStreamId(1);
  ge::AttrUtils::SetInt(op_desc, "_logic_stream_id", 0);
  ge::AttrUtils::SetInt(op_desc, "_logic_attached_stream_id", 1);
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  op_desc->SetStreamId(1);
  op_desc->SetAttachedStreamId(0);
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX));
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  model.split_logic_stream_2_origin_logic_stream_ = {{3, 1}, {4, 2}};
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  OpDescPtr attached_stream_op_desc = CreateOpDesc("attached_stream", PARTITIONEDCALL);
  attached_stream_op_desc->SetStreamId(0);
  ge::AttrUtils::SetListInt(attached_stream_op_desc, "_logic_attached_stream_ids", {1, 2});
  task->set_stream_id(3);
  model.SaveProfilingTaskDescInfo(attached_stream_op_desc, *task_info, *task);
  task->set_stream_id(4);
  model.SaveProfilingTaskDescInfo(attached_stream_op_desc, *task_info, *task);
  const auto &logic_stream_ids_to_physic_stream_ids = model.logic_stream_ids_to_physic_stream_ids_;
  EXPECT_TRUE(logic_stream_ids_to_physic_stream_ids.find(1) !=  logic_stream_ids_to_physic_stream_ids.end());
  EXPECT_TRUE(logic_stream_ids_to_physic_stream_ids.find(2) !=  logic_stream_ids_to_physic_stream_ids.end());
  EXPECT_EQ(model.task_desc_info_.size(), 6);
  EXPECT_FALSE(model.context_id_infos_.empty());
  EXPECT_FALSE(model.node_basic_infos_.empty());
  for (auto &ele : model.context_id_infos_) {
    reinterpret_cast<MsprofContextIdInfo *>(ele.context_id_info.data)->opName = 0UL;
  }
  for (auto &ele :model.prof_launch_apis_) {
    ele.api.itemId = 0UL;
  }
  EXPECT_EQ(model.ReportProfilingData(), SUCCESS);

  domi::TaskDef task_def;
  auto ifa_op_desc = MakeShared<OpDesc>("ifa", "IFA");
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def.mutable_kernel()->set_block_dim(7);
  auto &kernel_def = *task_def.mutable_kernel();
  auto &context = *kernel_def.mutable_context();
  context.set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  auto block_dim = model.GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, ifa_op_desc);
  EXPECT_EQ(block_dim, 7);
  (void)ge::AttrUtils::SetBool(ifa_op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
  block_dim = model.GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, ifa_op_desc);
  EXPECT_EQ(block_dim, 0xFFFFFFFF);
  context.set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU));
  block_dim = model.GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, ifa_op_desc);
  EXPECT_EQ(block_dim, 7);

  task_def.mutable_kernel()->set_block_dim(0);
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.mutable_kernel_with_handle()->set_block_dim(8);
  auto & all_kernel_def = *task_def.mutable_kernel_with_handle();
  auto & all_kernel_context = *all_kernel_def.mutable_context();
  all_kernel_context.set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  (void)ge::AttrUtils::SetBool(ifa_op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, false);
  block_dim = model.GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, ifa_op_desc);
  EXPECT_EQ(block_dim, 8);
  (void)ge::AttrUtils::SetBool(ifa_op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
  block_dim = model.GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, ifa_op_desc);
  EXPECT_EQ(block_dim, 0xFFFFFFFF);
  all_kernel_context.set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU));
  block_dim = model.GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, ifa_op_desc);
  EXPECT_EQ(block_dim, 8);
}

TEST_F(UtestDavinciModel, save_profile_task_info_vector) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<KernelTaskInfo> vector_task_info = MakeShared<KernelTaskInfo>();
  vector_task_info->task_id_ = 0;
  vector_task_info->prof_api_.begin_time = 1;
  vector_task_info->prof_api_.end_time = 2;
  model.task_list_.push_back(vector_task_info);
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;
  uint32_t op_impl_mode = 0x40;
  (void)AttrUtils::SetInt(op_desc, "_op_impl_mode_enum", op_impl_mode);

  domi::TaskDef *task1 = model_task_def.add_task();
  task1->mutable_kernel()->set_block_dim(7);
  task1->_impl_.stream_id_ = 0;
  task1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_VECTOR_KERNEL));
  model.SaveProfilingTaskDescInfo(op_desc, *vector_task_info, *task1);

  std::shared_ptr<KernelTaskInfo> vector_task_info2 = MakeShared<KernelTaskInfo>();
  vector_task_info2->task_id_ = 0;
  vector_task_info2->prof_api_.begin_time = 1;
  vector_task_info2->prof_api_.end_time = 2;
  model.task_list_.push_back(vector_task_info2);
  domi::TaskDef *task2 = model_task_def.add_task();
  task2->_impl_.stream_id_ = 0;
  task2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL));
  task2->mutable_kernel_with_handle()->set_block_dim(15);
  model.SaveProfilingTaskDescInfo(op_desc, *vector_task_info2, *task2);

  EXPECT_EQ(model.task_desc_info_.size(), 2UL);
  EXPECT_EQ(model.node_basic_infos_.size(), 2UL);
  EXPECT_EQ(model.node_basic_infos_[0].node_basic_info.data.nodeBasicInfo.blockDim, 7);
  EXPECT_EQ(model.node_basic_infos_[1].node_basic_info.data.nodeBasicInfo.blockDim, 15);
  EXPECT_FALSE(model.node_basic_infos_.empty());

  EXPECT_EQ(model.ReportProfilingData(), SUCCESS);
}

TEST_F(UtestDavinciModel, save_profile_task_info_mix) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<KernelTaskInfo> vector_task_info = MakeShared<KernelTaskInfo>();
  vector_task_info->task_id_ = 0;
  vector_task_info->prof_api_.begin_time = 1;
  vector_task_info->prof_api_.end_time = 2;
  model.task_list_.push_back(vector_task_info);
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;
  uint32_t op_impl_mode = 0x40;
  uint32_t task_ratio = 2;
  (void)AttrUtils::SetInt(op_desc, "_op_impl_mode_enum", op_impl_mode);
  (void)AttrUtils::SetBool(op_desc, "_is_fftsplus_task", true);
  (void)AttrUtils::SetInt(op_desc, "_task_ratio", task_ratio);

  // task 0, core type MIX_AIC
  domi::TaskDef *task1 = model_task_def.add_task();
  task1->mutable_kernel()->set_block_dim(7);
  task1->_impl_.stream_id_ = 0;
  task1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_VECTOR_KERNEL));
  model.SaveProfilingTaskDescInfo(op_desc, *vector_task_info, *task1);

  // task 1, change core type to MIX_AIV by set MIX_IS_AIV to true,
  (void)AttrUtils::SetBool(op_desc, "_mix_is_aiv", true);
  std::shared_ptr<KernelTaskInfo> vector_task_info2 = MakeShared<KernelTaskInfo>();
  vector_task_info2->task_id_ = 0;
  vector_task_info2->prof_api_.begin_time = 1;
  vector_task_info2->prof_api_.end_time = 2;
  model.task_list_.push_back(vector_task_info2);
  domi::TaskDef *task2 = model_task_def.add_task();
  task2->_impl_.stream_id_ = 0;
  task2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL));
  task2->mutable_kernel_with_handle()->set_block_dim(15);
  model.SaveProfilingTaskDescInfo(op_desc, *vector_task_info2, *task2);

  EXPECT_EQ(model.task_desc_info_.size(), 2UL);
  EXPECT_EQ(model.node_basic_infos_.size(), 2UL);
  EXPECT_EQ(model.node_basic_infos_[0].node_basic_info.data.nodeBasicInfo.blockDim, 7 | (task_ratio << 16U));
  EXPECT_EQ(model.node_basic_infos_[1].node_basic_info.data.nodeBasicInfo.blockDim, 15 | (task_ratio << 16U));
  EXPECT_EQ(model.node_basic_infos_[0].node_basic_info.data.nodeBasicInfo.taskType, MSPROF_GE_TASK_TYPE_MIX_AIC);
  EXPECT_EQ(model.node_basic_infos_[1].node_basic_info.data.nodeBasicInfo.taskType, MSPROF_GE_TASK_TYPE_MIX_AIV);
  EXPECT_FALSE(model.node_basic_infos_.empty());

  EXPECT_EQ(model.context_id_infos_.size(), 2UL);
  const auto &context_id_info =
      reinterpret_cast<MsprofContextIdInfo *>(model.context_id_infos_[0].context_id_info.data);
  EXPECT_EQ(context_id_info->ctxIdNum, 1);
  EXPECT_EQ(context_id_info->ctxIds[0], 0);  // only 1 context id, and it's 0.
  EXPECT_EQ(model.ReportProfilingData(), SUCCESS);
}

TEST_F(UtestDavinciModel, save_aicpu_profile_task_info) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;
  InitAicpuCtxCtx(op_desc, aicpudef);

  aicpudef->set_non_tail_block_dim(8);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);

  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV");
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  model.SaveProfilingInfoByContext(ffts_plus_task_def->ffts_plus_ctx(0), op_desc, ffts_plus_task_info->prof_api_, true);

  for(size_t i = 0U; i < model.task_desc_info_.size(); i++ ){
    TaskDescInfo ctx_desc_info = model.task_desc_info_[i];
    if (ctx_desc_info.task_type == kTaskTypeAicpu){
        EXPECT_EQ(ctx_desc_info.block_dim, 8);
    }
  }
}

TEST_F(UtestDavinciModel, save_atomic_profile_task_info) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *atmoicctx = ffts_plus_task_def->add_ffts_plus_ctx();
  atmoicctx->set_op_index(0);
  atmoicctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  atmoicctx->set_op_type(domi::FftsPlusCtxDef_OpType_ATOMIC);
  atmoicctx->set_uniq_ctx_name("MemSet");
  domi::FftsPlusAicAivCtxDef *atomicdef = atmoicctx->mutable_aic_aiv_ctx();
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;
  InitAicAivCtx(atomicdef, true);

  atomicdef->set_non_tail_block_dim(8);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);

  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV");
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  model.SaveProfilingInfoByContext(ffts_plus_task_def->ffts_plus_ctx(0), op_desc, ffts_plus_task_info->prof_api_, true);

  for(size_t i = 0U; i < model.task_desc_info_.size(); i++ ){
    TaskDescInfo ctx_desc_info = model.task_desc_info_[i];
    if (ctx_desc_info.task_type == kTaskTypeAicpu){
        EXPECT_EQ(ctx_desc_info.block_dim, 8);
    }
  }
}

TEST_F(UtestDavinciModel, save_cmo_profile_task_info) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *atmoicctx = ffts_plus_task_def->add_ffts_plus_ctx();
  atmoicctx->set_op_index(0);
  atmoicctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);

  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  model.SaveProfilingInfoByContext(ffts_plus_task_def->ffts_plus_ctx(0), op_desc, ffts_plus_task_info->prof_api_, true);

  EXPECT_EQ(model.node_basic_infos_.size(), 1);
}

TEST_F(UtestDavinciModel, save_mix_profile_task_info) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef *mixaicaivdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixaicaivdef);

  mixaicaivdef->set_non_tail_block_ratio_n(8);
  mixaicaivdef->set_non_tail_block_dim(8);

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);
  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  model.op_list_[0] = op_desc;
  model.SaveProfilingTaskDescInfo(op_desc, *task_info, *task);
  model.SaveProfilingInfoByContext(ffts_plus_task_def->ffts_plus_ctx(0), op_desc, ffts_plus_task_info->prof_api_, true);

  for(size_t i = 0U; i < model.task_desc_info_.size(); i++ ){
    TaskDescInfo ctx_desc_info = model.task_desc_info_[i];
    if (ctx_desc_info.task_type == kTaskTypeMixAic){
        EXPECT_EQ(ctx_desc_info.block_dim, 0x80008);
    }
  }
  auto block_dim = model.GetBlockDim(*mixaicaivctx);
  EXPECT_EQ(block_dim, 0x80008);
  (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
  block_dim = model.GetBlockDim(*mixaicaivctx);
  EXPECT_EQ(block_dim, 0xFFFFFFFF);
}

TEST_F(UtestDavinciModel, SaveProfileInfo_FftsPlusTask) {
  DavinciModel model(0, nullptr);
  model.SetId(1);

  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"1", "2"}));
  model.op_list_[0] = op_desc;

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_addr_size(2);
  ffts_plus_task_def->set_op_index(0);
  domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx_def->set_op_index(0);
  ctx_def->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  ctx_def->set_uniq_ctx_name("test");
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  ffts_plus_task_info->prof_api_.begin_time = 1;
  ffts_plus_task_info->prof_api_.end_time = 2;
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);
  model.SaveProfilingInfoByContext(ffts_plus_task_def->ffts_plus_ctx(0), op_desc, ffts_plus_task_info->prof_api_, true);
  EXPECT_EQ(model.node_basic_infos_.size(), 1);
  EXPECT_EQ(model.node_basic_infos_[0].op_name, "test");
}

TEST_F(UtestDavinciModel, init_model_profile_ffts_plus) {
  DavinciModel model(0, nullptr);
  model.SetId(1);

  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"1", "2"}));
  model.op_list_[op_desc->GetId()] = op_desc;

  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  task->_impl_.stream_id_ = 0;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task->mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(op_desc->GetId());
  ffts_plus_task_def->set_addr_size(2);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  std::shared_ptr<FftsPlusTaskInfo> ffts_plus_task_info = MakeShared<FftsPlusTaskInfo>();
  ffts_plus_task_info->task_id_ = 0;
  FusionOpInfo fusion_op_info;
  fusion_op_info.op_index = op_desc->GetId();
  fusion_op_info.stream_id = 0;
  fusion_op_info.original_op_names.push_back("conv");
  fusion_op_info.original_op_names.push_back("add");
  ffts_plus_task_info->fusion_op_info_.emplace_back(fusion_op_info);
  TaskInfoPtr task_info = ffts_plus_task_info;
  model.task_list_.push_back(task_info);

  EXPECT_EQ(model.InitFusionProfiling(fusion_op_info), SUCCESS);
}

TEST_F(UtestDavinciModel, parse_inputs_dims_data) {
  DavinciModel model(0, nullptr);

  OmeContext context;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  const auto data1 = CreateNode(*compute_graph, DATA, "data1", 1, 1);
  const auto next1 = CreateNode(*compute_graph, GETNEXT, "data1", 1, 1);

  std::vector<std::vector<int64_t>> tensor_input_dims;
  std::vector<vector<int64_t>> user_real_input_dims;

  model.SetRunContext(context);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), SUCCESS);  // dynamic_node_type is empty, just return

  context.dynamic_node_type = DATA;
  model.SetRunContext(context);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), SUCCESS);  // ParseInputsDimsForData

  context.getnext_nosink_nodes.emplace_back(next1);
  model.SetRunContext(context);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), SUCCESS);  // ParseInputsDimsForGetNexNosinkAndData
}

TEST_F(UtestDavinciModel, parse_inputs_dims_getnext) {
  DavinciModel model(0, nullptr);

  OmeContext context;
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  const auto data1 = CreateNode(*compute_graph, DATA, "data1", 1, 1);
  const auto next1 = CreateNode(*compute_graph, GETNEXT, "data1", 1, 1);

  std::vector<std::vector<int64_t>> tensor_input_dims;
  std::vector<vector<int64_t>> user_real_input_dims;

  tensor_input_dims.emplace_back(std::vector<int64_t>{});
  context.dynamic_node_type = GETNEXT;
  model.SetRunContext(context);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), SUCCESS);  // just getnext_sink

  context.getnext_nosink_nodes.emplace_back(next1);
  model.SetRunContext(context);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), SUCCESS);  // ParseInputsDimsForData

  context.data_nodes.emplace_back(data1);
  model.SetRunContext(context);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), PARAM_INVALID);  // ParseInputsDimsForGetNexNosinkAndData
  AttrUtils::SetInt(next1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  EXPECT_EQ(model.ParseInputsDims(tensor_input_dims, user_real_input_dims), SUCCESS);  // ParseInputsDimsForGetNexNosinkAndData
}

TEST_F(UtestDavinciModel, parse_queue_data) {
  DavinciModel model(0, nullptr);
  uint8_t mem[1024] {};
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
  model.runtime_param_.mem_size = sizeof(mem);
  std::set<uint64_t> input_outside_addrs;
  ASSERT_EQ(model.InitQueueDataNodes({}, 0, input_outside_addrs), SUCCESS);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  const auto queue_data_1 = CreateNode(*compute_graph, QUEUE_DATA, "queue_data_1", 0, 1);
  const auto queue_data_2 = CreateNode(*compute_graph, QUEUE_DATA, "queue_data_2", 0, 1);
  // multiple QueueData
  ASSERT_EQ(model.InitQueueDataNodes({queue_data_1, queue_data_2}, 0, input_outside_addrs), UNSUPPORTED);
  // not in LoadModelWithQ
  ASSERT_EQ(model.InitQueueDataNodes({queue_data_1}, 0, input_outside_addrs), UNSUPPORTED);
  QueueAttrs outputQueue1 = {1, 0, 0, 0U};
  model.output_queue_attrs_.emplace_back(outputQueue1);
  // missing attribute: queue_name
  ASSERT_EQ(model.InitQueueDataNodes({queue_data_1}, 0, input_outside_addrs), PARAM_INVALID);
  // success
  AttrUtils::SetStr(queue_data_1->GetOpDesc(), "queue_name", "some_name");
  ASSERT_EQ(model.InitQueueDataNodes({queue_data_1}, 0, input_outside_addrs), SUCCESS);
}

static NodePtr CreateNodeV2(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  std::vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  std::vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

static NodePtr CreateNodeV3(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num,
                            int32_t index) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape({4,2}), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  std::vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  std::vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

TEST_F(UtestDavinciModel, TestIsInputOfNetoutputCanZeroCopy) {
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);

  {
    uint8_t mem[1024] {};
    std::vector<OpDescPtr> output_op_list;
    std::set<uint64_t> output_outside_addrs;
    std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
    NodePtr data1 = CreateNodeV2(*graph, "data1", DATA, 0, 1);
    NodePtr data2 = CreateNodeV2(*graph, "data2", DATA, 0, 1);
    NodePtr netoutput = CreateNodeV2(*graph, "netoutput", NETOUTPUT, 2, 0);
    data1->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
    data2->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
    netoutput->AddLinkFrom(data1);
    netoutput->AddLinkFrom(data2);

    DavinciModel model(0, nullptr);
    output_op_list.clear();
    output_outside_addrs.clear();
    model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
    model.runtime_param_.mem_size = sizeof(mem);

    uint32_t data_op_index = 0U;
    std::map<uint32_t, OpDescPtr> data_by_index;
    EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitDataOp(graph, data2, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitNetOutput(graph, netoutput, output_op_list, output_outside_addrs), SUCCESS);
    EXPECT_TRUE(model.copy_only_addrs_.copy_only_addrs.empty()); // All outputs can zero-copy
  }

  {
    uint8_t mem[1024] {};
    std::vector<OpDescPtr> output_op_list;
    std::set<uint64_t> output_outside_addrs;
    std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
    NodePtr data1 = CreateNodeV2(*graph, "data1", DATA, 0, 1);
    NodePtr data2 = CreateNodeV2(*graph, "data2", DATA, 0, 1);
    NodePtr netoutput = CreateNodeV2(*graph, "netoutput", NETOUTPUT, 2, 0);
    netoutput->AddLinkFrom(data1);
    netoutput->AddLinkFrom(data2);

    data1->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));

    DavinciModel model(0, nullptr);
    output_op_list.clear();
    output_outside_addrs.clear();
    model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
    model.runtime_param_.mem_size = sizeof(mem);

    uint32_t data_op_index = 0U;
    std::map<uint32_t, OpDescPtr> data_by_index;
    EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitDataOp(graph, data2, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitNetOutput(graph, netoutput, output_op_list, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.copy_only_addrs_.copy_only_addrs.size(), 2); // one input + one output
  }

  {
    uint8_t mem[1024] {};
    std::vector<OpDescPtr> output_op_list;
    std::set<uint64_t> output_outside_addrs;
    std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
    NodePtr data1 = CreateNodeV2(*graph, "data1", DATA, 0, 1);
    NodePtr data2 = CreateNodeV2(*graph, "data2", DATA, 0, 1);
    NodePtr netoutput = CreateNodeV2(*graph, "netoutput", NETOUTPUT, 2, 0);
    netoutput->AddLinkFrom(data1);
    netoutput->AddLinkFrom(data2);

    DavinciModel model(0, nullptr);
    output_op_list.clear();
    output_outside_addrs.clear();
    model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
    model.runtime_param_.mem_size = sizeof(mem);

    uint32_t data_op_index = 0U;
    std::map<uint32_t, OpDescPtr> data_by_index;
    EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitDataOp(graph, data2, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitNetOutput(graph, netoutput, output_op_list, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.copy_only_addrs_.copy_only_addrs.size(), 4);  // two input + two output
  }

  {
    uint8_t mem[1024] {};
    std::vector<OpDescPtr> output_op_list;
    std::set<uint64_t> output_outside_addrs;
    std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
    ge::NodePtr data1 = CreateNodeV2(*graph, "data1", DATA, 0, 1);
    ge::NodePtr netoutput = CreateNodeV2(*graph, "netoutput", NETOUTPUT, 1, 0);
    netoutput->AddLinkFrom(data1);

    DavinciModel model(0, nullptr);
    output_op_list.clear();
    output_outside_addrs.clear();
    model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
    model.runtime_param_.mem_size = sizeof(mem);
    (void)AttrUtils::SetListInt(netoutput->GetOpDesc(), "_op_max_size", {100000});

    uint32_t data_op_index = 0U;
    std::map<uint32_t, OpDescPtr> data_by_index;
    EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
  }

  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, TestRefdataAsInputOfNetoutputShouldZeroCopy) {
  uint8_t mem[1024]{};
  std::vector<OpDescPtr> output_op_list;
  std::set<uint64_t> output_outside_addrs;
  std::set<uint64_t> input_outside_addrs;
  std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
  NodePtr data1 = CreateNodeV3(*graph, "data1", REFDATA, 0, 1, 0);
  NodePtr data2 = CreateNodeV3(*graph, "data2", DATA, 0, 1, 1);
  NodePtr netoutput = CreateNodeV3(*graph, "netoutput", NETOUTPUT, 2, 0, 0);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
  netoutput->AddLinkFrom(data1);
  netoutput->AddLinkFrom(data2);

  DavinciModel model(0, nullptr);
  output_op_list.clear();
  output_outside_addrs.clear();
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
  model.runtime_param_.mem_size = sizeof(mem);

  uint32_t data_op_index = 0U;
  std::map<uint32_t, OpDescPtr> data_by_index;
  EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, input_outside_addrs), SUCCESS);
  EXPECT_EQ(model.InitDataOp(graph, data2, data_op_index, data_by_index, input_outside_addrs), SUCCESS);
  EXPECT_EQ(model.InitNetOutput(graph, netoutput, output_op_list, output_outside_addrs), SUCCESS);
  // only data1 support zero copy, data2 disable zero cpy
  EXPECT_EQ(model.copy_only_addrs_.copy_only_addrs.size(), 1);
}

TEST_F(UtestDavinciModel, TestDataAsInputOfNetoutputShouldZeroCopy) {
  uint8_t mem[1024]{};
  std::vector<OpDescPtr> output_op_list;
  std::set<uint64_t> output_outside_addrs;
  std::set<uint64_t> input_outside_addrs;
  std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
  NodePtr data1 = CreateNodeV3(*graph, "data1", DATA, 0, 1, 0);
  NodePtr data2 = CreateNodeV3(*graph, "data2", DATA, 0, 1, 1);
  NodePtr netoutput = CreateNodeV3(*graph, "netoutput", NETOUTPUT, 2, 0, 0);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
  netoutput->AddLinkFrom(data1);
  netoutput->AddLinkFrom(data2);

  DavinciModel model(0, nullptr);
  output_op_list.clear();
  output_outside_addrs.clear();
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
  model.runtime_param_.mem_size = sizeof(mem);

  uint32_t data_op_index = 0U;
  std::map<uint32_t, OpDescPtr> data_by_index;
  EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
  EXPECT_EQ(model.InitDataOp(graph, data2, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
  EXPECT_EQ(model.InitNetOutput(graph, netoutput, output_op_list, output_outside_addrs), SUCCESS);
  // only data1 support zero copy, data2 disable zero cpy
  EXPECT_EQ(model.copy_only_addrs_.copy_only_addrs.size(), 2);
}

TEST_F(UtestDavinciModel, copy_input_data_null_tensor) {
  DavinciModel model(0, nullptr);
  InputData input_data;
  std::vector<int64_t> shape = {0};
  input_data.shapes.push_back(shape);
  DataBuffer buffer(nullptr, 0, false);
  input_data.blobs.push_back(buffer);
  model.input_data_info_.emplace(0, ZeroCopyOffset());
  EXPECT_EQ(model.CopyInputData(input_data), SUCCESS);
}

TEST_F(UtestDavinciModel, prof_fusion_op_info_test) {
  OpDescPtr op_desc = CreateOpDesc("fusion_op_1", "Enqueue");
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_UINT32);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  const std::vector<int64_t> workspace_bytes = {1,2,3};
  op_desc->SetWorkspaceBytes(workspace_bytes);
  const std::vector<uint8_t> weights_value(64, 'A');
  GeTensorPtr weight_value = MakeShared<GeTensor>(op_desc->GetOutputDesc(0), weights_value.data(), weights_value.size());
  AttrUtils::SetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight_value);

  uint64_t info_num = 0;
  std::vector<MsprofAdditionalInfo> infos{};
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len)->int32_t{
    if (type != ge::InfoType::kInfo) {
      return 0;
    }
    ++info_num;
    auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
    infos.emplace_back(*info);
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(check_func);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_[0] = op_desc;
  FusionOpInfo fusion_op_info{};
  fusion_op_info.original_op_names = {"op_1", "op_2", "op_3", "op_4", "op_5", "op_6","op_7","op_8","op_9"};
  fusion_op_info.op_index = 0;
  fusion_op_info.op_name = "fusion_op_1";
  davinci_model.InitFusionProfiling(fusion_op_info);
  davinci_model.ReportFusionOpInfo();
  EXPECT_EQ(info_num, 2);
  EXPECT_EQ(infos[0].dataLen, 116);
  EXPECT_EQ(infos[1].dataLen, 60);
}

TEST_F(UtestDavinciModel, ProfFusionOpInfo_Ok_UpdateHashOnExecute) {
  OpDescPtr op_desc = CreateOpDesc("fusion_op_1", "Enqueue");
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_UINT32);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  const std::vector<int64_t> workspace_bytes = {1,2,3};
  op_desc->SetWorkspaceBytes(workspace_bytes);
  const std::vector<uint8_t> weights_value(64, 'A');
  GeTensorPtr weight_value = MakeShared<GeTensor>(op_desc->GetOutputDesc(0), weights_value.data(), weights_value.size());
  AttrUtils::SetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight_value);

  uint64_t info_num = 0;
  std::vector<MsprofAdditionalInfo> infos{};
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len)->int32_t{
    if (type != ge::InfoType::kInfo) {
      return 0;
    }
    ++info_num;
    auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
    infos.emplace_back(*info);
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(check_func);
  DavinciModel davinci_model(0, nullptr);
  davinci_model.op_list_[0] = op_desc;
  FusionOpInfo fusion_op_info{};
  fusion_op_info.original_op_names = {"op_1", "op_2", "op_3", "op_4", "op_5", "op_6","op_7","op_8","op_9"};
  fusion_op_info.op_index = 0;
  fusion_op_info.op_name = "fusion_op_1";
  davinci_model.InitFusionProfiling(fusion_op_info);
  for (auto &ele : davinci_model.profile_list_) {
    for (auto &prof : ele.prof_fusion_data_lst) {
      for (size_t i = 0UL; i < 8; ++i) {
        reinterpret_cast<ProfFusionOpInfo *>(prof.data)->opName=0;
        reinterpret_cast<ProfFusionOpInfo *>(prof.data)->fusionOpId[i] = 0;
      }
    }
  }
  davinci_model.ReportFusionOpInfo();
  std::hash<std::string> hs;
  for (auto &ele : davinci_model.profile_list_) {
    for (size_t i = 0UL; i < ele.prof_fusion_data_lst.size(); ++i) {
      auto prof_fusion_info = reinterpret_cast<ProfFusionOpInfo *>(ele.prof_fusion_data_lst[i].data);
      for (size_t j = 0UL; j < prof_fusion_info->fusionOpNum; ++j) {
        EXPECT_EQ(prof_fusion_info->fusionOpId[j],
                  hs(fusion_op_info.original_op_names[i * MSPROF_GE_FUSION_OP_NUM + j]));
      }
    }
  }
  EXPECT_EQ(info_num, 2);
  EXPECT_EQ(infos[0].dataLen, 116);
  EXPECT_EQ(infos[1].dataLen, 60);
}

TEST_F(UtestDavinciModel, GetSomeInfo) {
  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_input = CreateOpDesc("data", DATA);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);
  op_input->SetInputOffset({1024});
  op_input->SetOutputOffset({1024});
  NodePtr node_input = graph->AddNode(op_input);    // op_index = 0

  OpDescPtr op_output = CreateOpDesc("output", NETOUTPUT);
  op_output->AddInputDesc(tensor);
  op_output->SetInputOffset({5120});
  op_output->SetSrcName( { "memcpy" } );
  op_output->SetSrcIndex( { 0 } );
  NodePtr node_output = graph->AddNode(op_output);  // op_index = 1

  // get input output
  std::vector<InputOutputDescInfo> input_descs;
  std::vector<InputOutputDescInfo> output_descs;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;
   bool by_dims = false;
  auto ret = model.GetInputOutputDescInfo(input_descs, output_descs, output_formats, output_formats, by_dims);
  EXPECT_EQ(ret, SUCCESS);

  // get dynamic batch
  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type;
  ret = model.GetDynamicBatchInfo(batch_info, dynamic_type);
  EXPECT_EQ(ret, SUCCESS);

  // get Combined Dynamic Dims
  model.GetCombinedDynamicDims(batch_info);
  EXPECT_EQ(batch_info.size(), 0);

  // get User Designate Shape Order
  std::vector<std::string> user_input_shape_order;
  model.GetUserDesignateShapeOrder(user_input_shape_order);
  EXPECT_EQ(user_input_shape_order.size(), 0);

  // get Flowctrl Index
  uint32_t op_index = 0;
  auto ret2 = model.GetFlowctrlIndex(op_index);
  EXPECT_EQ(ret2, 0);

  // get CurDynamic Dims
  std::vector<std::vector<int64_t>> tensor_input_dims;
  std::vector<int32_t> cur_dynamic_dims;
  auto ret3 = model.GetCurDynamicDims(tensor_input_dims, cur_dynamic_dims);
  EXPECT_EQ(ret3, INTERNAL_ERROR);
}

TEST_F(UtestDavinciModel, SetSomething) {
  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120000);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  // Set EndGraphId
  uint32_t task_id = 1;
  uint32_t stream_id = 2;
  model.SetEndGraphId(task_id, stream_id);

  // Set DynamicSize
  std::vector<uint64_t> batch_num;
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH);
  model.SetDynamicSize(batch_num, dynamic_type);
  batch_num.emplace_back(1);
  model.SetDynamicSize(batch_num, dynamic_type);
  EXPECT_EQ(model.dynamic_type_, dynamic_type);
}

TEST_F(UtestDavinciModel, InitSomething) {
  DavinciModel model(0, nullptr);
  OpDescPtr op_desc = CreateOpDesc("data", DATA);

  // Init StreamActive
  //op_desc->SetAttr(ATTR_NAME_SWITCH_BRANCH_NODE_LABEL, 0);
  auto ret = model.InitStreamActive(op_desc);
  EXPECT_EQ(ret, SUCCESS);

  // Init StreamSwitch
  ret = model.InitStreamSwitch(op_desc);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestDavinciModel, DavinciModel_HeadFile) {
  DavinciModel model(0, nullptr);
  std::map<int64_t, std::vector<rtStream_t>> hccl_flow;
  hccl_flow = model.GetHcclFolowStream();
  EXPECT_EQ(hccl_flow.size(), 0);

  string om_name = "om_abc";
  model.SetOmName(om_name);
  EXPECT_EQ(model.om_name_, om_name);
}

// for coverage all below test
TEST_F(UtestDavinciModel, IsInputOfNetoutputCanZeroCopy_fail) {
  DavinciModel *model = new DavinciModel(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  OpDescPtr op_input = CreateOpDesc("data", DATA);
  NodePtr node_input = graph->AddNode(op_input);

  model->DisableZeroCopyInReuseMemoryMode(node_input, 10, nullptr);

  EXPECT_EQ(model->copy_only_addrs_.copy_only_addrs.size(), 1);

  g_runtime_stub_mock = "";
  delete model;
}

TEST_F(UtestDavinciModel, Assign_fail) {
  DavinciModel *model = new DavinciModel(0, nullptr);

  model->Assign(nullptr);
  EXPECT_EQ(model->ge_model_, nullptr);

  void *addr1 = new uint8_t[1];
  void *addr2 = new uint8_t[1];
  model->stream_2_event_[addr1] = addr2;
  g_runtime_stub_mock = "rtEventDestroy";
  delete model;
  delete[] (uint8_t *)addr1;
}

TEST_F(UtestDavinciModel, InitWeightMem_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  uintptr_t mem_ptr = 0;
  uintptr_t weight_ptr = 0;
  size_t weight_size = 10;

  model.is_weight_mem_has_inited_ = true;
  EXPECT_EQ(model.InitWeightMem(mem_ptr, weight_ptr, weight_size), FAILED);

  model.is_weight_mem_has_inited_ = false;
  ge_model->weights_buffer_ = Buffer(20, 0);
  EXPECT_NE(model.InitWeightMem(mem_ptr, weight_ptr, weight_size), FAILED);   //??

  model.FreeWeightsMem();
  model.is_weight_mem_has_inited_ = false;
  weight_size = 30;
  g_runtime_stub_mock = "rtMalloc";
  EXPECT_EQ(model.InitWeightMem(mem_ptr, weight_ptr, weight_size), ACL_ERROR_GE_MEMORY_ALLOCATION);

  model.is_weight_mem_has_inited_ = false;
  weight_ptr = 10000;
  weight_size = 5;
  EXPECT_EQ(model.InitWeightMem(mem_ptr, weight_ptr, weight_size), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestDavinciModel, InitFeatureMapAndP2PMem_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  uintptr_t mem_ptr = 10;
  size_t mem_size = 10;

  model.is_feature_map_mem_has_inited_ = false;
  std::map<std::string, std::string> param;
  param[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGlobalOption(param);

  MemInfo mem_info;
  model.runtime_param_.fm_memory_infos.push_back(mem_info);
  EXPECT_EQ(model.InitFeatureMapAndP2PMem(mem_ptr, mem_size), SUCCESS);  // fm_memory_infos is ok

  mem_ptr = 0;
  model.runtime_param_.mem_size = 20;
  model.runtime_param_.zero_copy_size = 10;
  g_runtime_stub_mock = "rtMalloc";

  EXPECT_NE(model.InitFeatureMapAndP2PMem(mem_ptr, mem_size), SUCCESS); // call more than once
  param[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGlobalOption(param);
}

TEST_F(UtestDavinciModel, InitFeatureMapAndP2PMem_fixed1) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  uintptr_t mem_ptr = 10;
  size_t mem_size = 10;

  model.is_feature_map_mem_has_inited_ = false;
  model.runtime_param_.mem_size = 64;
  model.runtime_param_.zero_copy_size = 0;

  uint8_t mem_base_addr = 0;
  MemInfo mem_info;
  mem_info.memory_type = RT_MEMORY_HBM;
  mem_info.logic_memory_base = 0x8800;
  mem_info.memory_size = 32;
  mem_info.memory_base = &mem_base_addr;
  mem_info.is_fixed_addr_prior = 0;
  model.runtime_param_.fm_memory_infos.push_back(mem_info);
  MemInfo one_fm_mem_info;
  one_fm_mem_info.memory_type = RT_MEMORY_HBM;
  one_fm_mem_info.logic_memory_base = 0x8000;
  one_fm_mem_info.memory_size = 16;
  one_fm_mem_info.memory_base = &mem_base_addr;
  one_fm_mem_info.is_fixed_addr_prior = 1;
  model.runtime_param_.fixed_fm_memory_infos.push_back(one_fm_mem_info);
  EXPECT_EQ(model.InitFixedFeatureMap(0x100000, 32), SUCCESS);
  EXPECT_NE(model.InitFeatureMapAndP2PMem(mem_ptr, mem_size), SUCCESS);
  model.runtime_param_.fixed_mem_base = 10U;
  model.is_feature_map_mem_has_inited_ = false;
  EXPECT_EQ(model.InitFeatureMapAndP2PMem(0U, 0U), SUCCESS);
}

TEST_F(UtestDavinciModel, InitFeatureMapAndP2PMem_fixed2) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  uint8_t mem_base_addr = 0;
  MemInfo one_fm_mem_info;
  one_fm_mem_info.memory_type = RT_MEMORY_HBM;
  one_fm_mem_info.logic_memory_base = 0x8000;
  one_fm_mem_info.memory_size = 16;
  one_fm_mem_info.memory_base = &mem_base_addr;
  one_fm_mem_info.is_fixed_addr_prior = 1;
  model.runtime_param_.fixed_fm_memory_infos.push_back(one_fm_mem_info);
  EXPECT_EQ(model.InitFixedFeatureMap(0x100000, 32), SUCCESS);
  model.runtime_param_.fixed_mem_base = 10U;
  model.runtime_param_.mem_size = 16;
  model.runtime_param_.zero_copy_size = 0;
  model.runtime_param_.fm_memory_infos.clear();
  model.is_feature_map_mem_has_inited_ = false;
  EXPECT_EQ(model.InitFeatureMapAndP2PMem(0x9000, 32U), SUCCESS);
}

TEST_F(UtestDavinciModel, MallocFeatureMapMem_failed) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  uint8_t mem_base_addr = 0;
  MemInfo one_fm_mem_info;
  one_fm_mem_info.memory_type = RT_MEMORY_HBM;
  one_fm_mem_info.logic_memory_base = 0x8000;
  one_fm_mem_info.memory_size = 1024;
  one_fm_mem_info.memory_base = &mem_base_addr;
  one_fm_mem_info.is_fixed_addr_prior = 1;
  model.runtime_param_.fm_memory_infos.push_back(one_fm_mem_info);
  dlog_setlevel(0, 0, 0);
  model.runtime_param_.fixed_mem_base = 10U;
  EXPECT_EQ(model.MallocFeatureMapMem(16), nullptr);
  dlog_setlevel(0, 3, 0);

  MemInfo one_fm_mem_info_1;
  one_fm_mem_info_1.memory_type = RT_MEMORY_HBM;
  one_fm_mem_info_1.logic_memory_base = 0x8000;
  one_fm_mem_info_1.memory_size = 16;
  one_fm_mem_info_1.memory_base = &mem_base_addr;
  one_fm_mem_info_1.is_fixed_addr_prior = 1;
  model.runtime_param_.zero_copy_size = 1024;
  model.runtime_param_.fm_memory_infos.clear();
  model.runtime_param_.fm_memory_infos.push_back(one_fm_mem_info_1);
  dlog_setlevel(0, 0, 0);
  model.runtime_param_.fixed_mem_base = 10U;
  model.runtime_param_.mem_size = 16;
  EXPECT_EQ(model.MallocFeatureMapMem(16), nullptr);
  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestDavinciModel, test_CpuModelPrepareOutput) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  EXPECT_EQ(model.CpuModelPrepareOutput(0, 0, 0), FAILED);

  void *t = new uint8_t[1];
  model.input_mbuf_list_ = std::vector<uintptr_t>({ PtrToValue(t) });
  model.output_mbuf_list_ = std::vector<uintptr_t>({ PtrToValue(t) });

  EXPECT_EQ(model.CpuModelPrepareOutput(0, 0, 0), SUCCESS);
  delete[] (uint8_t *)t;
}

TEST_F(UtestDavinciModel, test_CpuModelDequeue) {
  DavinciModel model(0, nullptr);
  EXPECT_EQ(model.CpuModelDequeue(), SUCCESS);  // empty
  QueueAttrs inputQueue1 = {1, 0, 0, 0U};
  QueueAttrs inputQueue2 = {2, 0, 0, 0U};
  QueueAttrs inputQueue3 = {3, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  model.input_queue_attrs_.emplace_back(inputQueue2);
  model.input_queue_attrs_.emplace_back(inputQueue3);
  EXPECT_EQ(model.CpuModelDequeue(), SUCCESS);  // independent dequeue

  DavinciModel model2(0, nullptr);
  model2.input_queue_attrs_.emplace_back(inputQueue1);
  AttrUtils::SetInt(model2.align_attrs_[0], ATTR_NAME_INPUTS_ALIGN_OFFSET, 0);
  AttrUtils::SetInt(model2.align_attrs_[0], ATTR_NAME_INPUTS_ALIGN_INTERVAL, 2);
  EXPECT_EQ(model2.CpuModelDequeue(), SUCCESS);  // batch dequeue
}

TEST_F(UtestDavinciModel, test_CreateOutput) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);

  GeTensorDesc tensor(GeShape({16, 16, 16, 16}), FORMAT_FRACTAL_Z, DT_FLOAT);
  AttrUtils::SetInt(tensor, ATTR_NAME_SPECIAL_OUTPUT_SIZE, 32);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);

  InputOutputDescInfo output;
  uint32_t format_result;
  model.CreateOutput(0, op_desc, output, format_result);
  EXPECT_EQ(format_result, FORMAT_HWCN);
}

TEST_F(UtestDavinciModel, test_ReportModelExtInfo) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  domi::GetContext().is_online_model = true;

  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len)->int32_t{
    if (type != InfoType::kInfo) {
      return 0;
    }
    auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
    EXPECT_EQ(info->dataLen, 16);
    EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[8]), model.runtime_param_.graph_id);
    std::hash<std::string> hs;
    EXPECT_EQ(*reinterpret_cast<uint64_t *>(&info->data[0]), hs(model.name_));
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  model.ReportModelExtInfo(mmGetTid(), 0);
}

TEST_F(UtestDavinciModel, test_InitConstant) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("cosnt", CONSTANT);
  EXPECT_EQ(model.InitConstant(op_desc), PARAM_INVALID);

  const std::vector<uint8_t> weights_value(64, 'A');
  GeTensorPtr weight = MakeShared<GeTensor>(GeTensorDesc(GeShape({16}), FORMAT_NCHW, DT_STRING), weights_value.data(), weights_value.size());
  AttrUtils::SetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight);

  GeTensorDesc tensor(GeShape({16}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  model.runtime_param_.mem_size = 2000;
  model.runtime_param_.mem_base = PtrToValue(malloc(2000));

  EXPECT_EQ(model.InitConstant(op_desc), PARAM_INVALID);
  free(reinterpret_cast<void*>(model.runtime_param_.mem_base));
}

TEST_F(UtestDavinciModel, test_InitStreamActive) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("var", VARIABLE);
  std::vector<uint32_t> active_stream_list = {1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);

  EXPECT_EQ(model.InitStreamActive(op_desc), SUCCESS);
}

TEST_F(UtestDavinciModel, test_InitStreamSwitch) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("var", VARIABLE);
  std::vector<uint32_t> active_stream_list = {1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);

  EXPECT_EQ(model.InitStreamSwitch(op_desc), SUCCESS);
}

TEST_F(UtestDavinciModel, test_InitCase) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("var", VARIABLE);
  AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 2);

  EXPECT_EQ(model.InitCase(op_desc), FAILED);
  for (uint32_t i = 0U; i < 2; ++i) {
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    std::vector<int64_t> batch_shape = {1, 2};
    AttrUtils::SetListInt(op_desc, attr_name, batch_shape);
    const std::string attr_combined_batch = ATTR_NAME_COMBINED_BATCH + "_" + std::to_string(i);
    AttrUtils::SetListInt(op_desc, attr_combined_batch, batch_shape);
  }
  EXPECT_EQ(model.InitCase(op_desc), SUCCESS);
}

TEST_F(UtestDavinciModel, TransAllVarData_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  std::vector<NodePtr> variable_nodes { nullptr };

  g_runtime_stub_mock = "rtCtxGetCurrent";
  EXPECT_EQ(model.TransAllVarData(graph, variable_nodes), FAILED); // rtCtxGetCurrent failed.
}

TEST_F(UtestDavinciModel, SetDataDumperArgs_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  g_runtime_stub_mock = "rtGetDevice";
  model.SetDataDumperArgs(graph, std::map<std::string, OpDescPtr>());

  EXPECT_EQ(model.data_dumper_.device_id_, 0);
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
}

TEST_F(UtestDavinciModel, InitL1DataDumperArgs_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  model.data_dumper_.dump_properties_.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", std::set<std::string>());
  g_runtime_stub_mock = "rtDumpAddrSet";
  EXPECT_EQ(model.InitL1DataDumperArgs(), FAILED);
}

TEST_F(UtestDavinciModel, GetEventIdForBlockingAicpuOp_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  rtStream_t stream = (rtStream_t)0;
  uint32_t event_id = 0;

  model.stream_2_event_[stream] = (rtEvent_t)2;
  g_runtime_stub_mock = "rtGetEventID";
  EXPECT_NE(model.GetEventIdForBlockingAicpuOp(op_desc, stream, event_id), SUCCESS);

  model.stream_2_event_.clear();
  EXPECT_NE(model.GetEventIdForBlockingAicpuOp(op_desc, stream, event_id), SUCCESS);

  g_runtime_stub_mock = "rtEventCreateWithFlag";
  EXPECT_EQ(model.GetEventIdForBlockingAicpuOp(op_desc, stream, event_id), SUCCESS); //??
}

TEST_F(UtestDavinciModel, UpdateOpInputValue_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);

  EXPECT_EQ(model.UpdateOpInputValue(op_desc, 10, 10), PARAM_INVALID);
}

TEST_F(UtestDavinciModel, GetCurDynamicDims_fail) {
  DavinciModel model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  model.Assign(ge_model);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  OpDescPtr op_input = CreateOpDesc("data", DATA);
  NodePtr node_input = graph->AddNode(op_input);

  AttrUtils::SetInt(op_input, ATTR_NAME_INDEX, 10);

  model.run_context_.data_nodes.push_back(node_input);
  model.run_context_.getnext_nosink_nodes.push_back(node_input);

  model.run_context_.dynamic_node_type = DATA;

  std::vector<std::vector<int64_t>> tensor_input_dims;
  std::vector<int32_t> cur_dynamic_dims;
  EXPECT_EQ(model.GetCurDynamicDims(tensor_input_dims, cur_dynamic_dims), INTERNAL_ERROR);

  model.run_context_.dynamic_node_type = "Add";
  model.run_context_.data_nodes.clear();
  model.run_context_.getnext_nosink_nodes.clear();
  model.run_context_.user_input_dims.push_back(std::make_pair(std::string("data"), std::vector<int64_t>()));
  EXPECT_EQ(model.GetCurDynamicDims(tensor_input_dims, cur_dynamic_dims), INTERNAL_ERROR);

  model.run_context_.dynamic_node_type = DATA;
  model.run_context_.user_input_dims.push_back(std::make_pair(std::string("data"), std::vector<int64_t>({1, 2})));

  AttrUtils::SetInt(op_input, ATTR_NAME_INDEX, 0);
  GeTensorDesc tensor(GeShape(std::vector<int64_t>({1, 2})), FORMAT_NCHW, DT_FLOAT);
  op_input->AddInputDesc(tensor);
  op_input->AddOutputDesc(tensor);

  tensor_input_dims.push_back(std::vector<int64_t>({1, 2}));
  model.run_context_.dynamic_shape_dims.push_back(std::vector<int32_t>({1, 2}));
  EXPECT_NE(model.GetCurDynamicDims(tensor_input_dims, cur_dynamic_dims), SUCCESS);  //>>

  tensor_input_dims[0].push_back(3);
  EXPECT_EQ(model.GetCurDynamicDims(tensor_input_dims, cur_dynamic_dims), INTERNAL_ERROR);
}

TEST_F(UtestDavinciModel, TestInsertMultiBatchShpaeData) {
  DavinciModel model(0, nullptr);
  model.cur_dynamic_dims_ = {1,1,1,1};
  std::vector<DataBuffer> blobs;
  model.runtime_param_.mem_size = 100;
  model.mem_base_size_ = 40;
  model.is_getnext_sink_dynamic_ = false;
  model.is_online_infer_dynamic_ = true;
  model.run_context_.dynamic_shape_dims = {{1,1,1,1}};
  model.CreateMultiBatchDataBuffer(blobs);
  EXPECT_EQ(blobs.size(), 1);
  EXPECT_EQ(blobs[0].length, 16);
  EXPECT_EQ(blobs[0].data, model.cur_dynamic_dims_.data());
}

TEST_F(UtestDavinciModel, CheckModelNoInputAndOutput) {
  DavinciModel model(0, nullptr);
  auto flag = model.CheckModelNoInputAndOutput();
  EXPECT_EQ(flag, true);
  QueueAttrs inputQueue1 = {1, 0, 0, 0U};
  model.input_queue_attrs_.emplace_back(inputQueue1);
  flag = model.CheckModelNoInputAndOutput();
  EXPECT_EQ(flag, false);
}

TEST_F(UtestDavinciModel, TestCheckUserAndModelSize) {
  DavinciModel model(0, nullptr);
  int64_t size = 0;
  int64_t op_size = size + 65;
  model.is_dynamic_ = false;
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(model.CheckUserAndModelSize(size, op_size, "output"), false);

  size = std::numeric_limits<int64_t>::max() - 63;
  EXPECT_EQ(model.CheckUserAndModelSize(size, op_size, "output"), true);

  op_size = size;
  EXPECT_EQ(model.CheckUserAndModelSize(size, op_size, "output"), true);

  model.is_dynamic_ = true;
  EXPECT_EQ(model.CheckUserAndModelSize(size, op_size, "output"), true);
  model.is_dynamic_ = false;

  model.is_dynamic_aipp_ = true;
  EXPECT_EQ(model.CheckUserAndModelSize(size, op_size, "output"), true);

  EXPECT_EQ(model.CheckUserAndModelSize(size, op_size, "output"), true);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, InitOutputTensorInfoTest) {
  DavinciModel model(0, nullptr);
  model.is_getnext_sink_dynamic_ =  true;
  auto invalid_op = MakeShared<OpDesc>();
  Status ret = model.InitOutputTensorInfo(invalid_op);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestDavinciModel, InitSpaceRegistry_SUCCESS_SpaceRegistryExist) {
  GeRootModelPtr root_model = std::make_shared<GeRootModel>();

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(gert::SpaceRegistryFaker().BuildRegistryArray());
  Status ret = model.InitSpaceRegistry(root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestDavinciModel, InitSpaceRegistry_SUCCESS_HasSoBin) {
  dlog_setlevel(0, 0, 0);
  GeRootModelPtr root_model = std::make_shared<GeRootModel>();
  std::vector<OpSoBinPtr> kernels;
  std::string so_name("libopmaster.so");
  std::string vendor_name("/opp/MDC/");
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, vendor_name, std::move(so_bin), so_name.length());
  kernels.emplace_back(op_so_bin);
  root_model->op_so_store_.kernels_ = std::move(kernels);

  DavinciModel model(0, nullptr);
  model.SetSpaceRegistries(nullptr);
  Status ret = model.InitSpaceRegistry(root_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestDavinciModel, GetRealOutputSizeOfCaseTest) {
  ComputeGraphPtr sub_graph = MakeShared<ComputeGraph>("sub_graph");
  OpDescPtr sub_netoutput = CreateOpDesc("sub_netoutput", "NetOutput");
  NodePtr sub_netoutput_node = sub_graph->AddNode(sub_netoutput);
  std::string invalid_batch_label_1 = "Batch_a";
  AttrUtils::SetStr(sub_netoutput_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, invalid_batch_label_1);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  graph->AddSubGraph(sub_graph);
  OpDescPtr op_case = CreateOpDesc("case", "Case");
  NodePtr node_case = graph->AddNode(op_case);
  node_case->GetOpDesc()->AddSubgraphName("sub_graph");
  node_case->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");

  DavinciModel model(0, nullptr);
  auto invalid_op = MakeShared<OpDesc>();
  Status ret = model.GetRealOutputSizeOfCase(graph, 0, node_case);
  EXPECT_NE(ret, SUCCESS);

  std::string invalid_batch_label_2 = "Batch_" + std::to_string(INT64_MAX);
  AttrUtils::SetStr(sub_netoutput_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, invalid_batch_label_2);
  ret = model.GetRealOutputSizeOfCase(graph, 0, node_case);
  EXPECT_NE(ret, SUCCESS);
}
TEST_F(UtestDavinciModel, run_with_task_fail) {
  DavinciModel model(0, nullptr);
  model.SetId(1);

  auto data = MakeShared<RunArgs>();
  model.data_inputer_.Push(data);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  model.task_list_.push_back(task_info);

  const char_t * const kEnvRecordPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);

  EXPECT_EQ(model.ModelRunStart(), SUCCESS);
  sleep(5);
  model.ModelRunStop();
  EXPECT_EQ(unsetenv(kEnvRecordPath), SUCCESS);
}

TEST_F(UtestDavinciModel, malloc_and_free_dynamic_memory) {
  DavinciModel model(0, nullptr);
  const size_t mem_size = 1024U;
  auto workspaces = model.MallocDynamicMemory(mem_size);
  EXPECT_NE(workspaces, nullptr);
  model.FreeDynamicWorkspaceMemory();
}

TEST_F(UtestDavinciModel, GetMemEventIdAddr) {
  DavinciModel model(0, nullptr);
  auto p1 = model.GetMemEventIdAddr(1);
  EXPECT_NE(p1, nullptr);
  auto p2 = model.GetMemEventIdAddr(1);
  EXPECT_EQ(p1, p2);
  EXPECT_EQ(model.mem_event_id_mem_map_.size(), 1);
  auto p3 = model.GetMemEventIdAddr(2);
  EXPECT_NE(p1, p3);
  EXPECT_EQ(model.mem_event_id_mem_map_.size(), 2);
}

TEST_F(UtestDavinciModel, NnExecute_update_step) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  options["ge.exec.enableDump"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);


  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);

  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  uint64_t step_value = *(PtrToPtr<void, uint64_t>(ValueToPtr(model.GetGlobalStep())));
  EXPECT_EQ(step_value, 0U);

  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  step_value = *(PtrToPtr<void, uint64_t>(ValueToPtr(model.GetGlobalStep())));
  EXPECT_EQ(step_value, 1U);
}

TEST_F(UtestDavinciModel, NnExecute_set_timeout_when_execute_without_stream) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  options["ge.exec.enableDump"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);

  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  // 模拟ACL在执行前设置超时时间的方法
  ge::GetContext().SetStreamSyncTimeout(15000);
  auto sync_exec = [](rtModel_t model, rtStream_t stream, uint32_t flag, int32_t timeout) -> rtError_t {
    // 模拟RTS接口时获取超时时间的方法
    EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), 15000);
    return RT_ERROR_NONE;
  };
  auto rts_stub = std::make_shared<MockRtExecute>();
  ge::RuntimeStub::SetInstance(rts_stub);
  EXPECT_CALL(*rts_stub, rtModelExecuteSync).WillRepeatedly(sync_exec);
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
                            input_tensor, output_tensor), SUCCESS);
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, NnExecute_set_timeout_when_execute_with_stream) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  options["ge.exec.enableDump"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);

  rtStream_t stream = (void*)0x01;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);

  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  // 模拟ACL在执行前设置超时时间的方法
  ge::GetContext().SetStreamSyncTimeout(15000);
  auto sync_exec = [](rtModel_t model, rtStream_t stream, uint32_t flag, int32_t timeout) -> rtError_t {
    // 模拟RTS接口时获取超时时间的方法
    EXPECT_EQ(ge::GetContext().StreamSyncTimeout(), 15000);
    return RT_ERROR_NONE;
  };
  auto rts_stub = std::make_shared<MockRtExecute>();
  ge::RuntimeStub::SetInstance(rts_stub);
  EXPECT_CALL(*rts_stub, rtModelExecuteSync).WillRepeatedly(sync_exec);
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
                            input_tensor, output_tensor), SUCCESS);
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, NnExecute_update_step_with_gert_tensor) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  options["ge.exec.enableDump"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);


  rtStream_t stream = nullptr;

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);

  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  input_tensor.resize(1);
  unique_ptr<uint8_t[]> data_buf_1(new (std::nothrow) uint8_t[512]);
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_1.get()};
  unique_ptr<uint8_t[]> data_buf_3(new (std::nothrow) uint8_t[512]);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_3.get()};
  EXPECT_EQ(model.NnExecute(stream, true, input_tensor, output_tensor), SUCCESS);
  uint64_t step_value = *(PtrToPtr<void, uint64_t>(ValueToPtr(model.GetGlobalStep())));
  EXPECT_EQ(step_value, 0U);
}

TEST_F(UtestDavinciModel, NnExecute_triggerZeroCopyMemoryInsufficientError) {
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  ProfilingProperties::Instance().is_load_profiling_ = true;
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2580);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);

  rtStream_t stream = nullptr;
  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);
  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  input_tensor.resize(1);
  unique_ptr<uint8_t[]> data_buf_1(new (std::nothrow) uint8_t[512]);
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},
                            gert::kOnDeviceHbm,
                            ge::DT_INT32,
                            (void *) data_buf_1.get()};
  unique_ptr<uint8_t[]> data_buf_3(new (std::nothrow) uint8_t[512]);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},
                            gert::kOnHost,
                            ge::DT_INT32,
                            (void *) data_buf_3.get()};
  EXPECT_NE(model.NnExecute(stream, true, input_tensor, output_tensor), SUCCESS);
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, NnExecute_with_gert_tensor_sync) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  options["ge.exec.enableDump"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);


  rtStream_t stream = nullptr;

  ProfilingManager::Instance().device_id_.emplace_back(0);
  model.task_list_.resize(1);

  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  input_tensor.resize(1);
  unique_ptr<uint8_t[]> data_buf_1(new (std::nothrow) uint8_t[512]);
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                     {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                     gert::kOnDeviceHbm,                                // placement
                     ge::DT_INT32,                              // data type
                     (void *) data_buf_1.get()};
  unique_ptr<uint8_t[]> data_buf_3(new (std::nothrow) uint8_t[512]);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                      {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                      gert::kOnDeviceHbm,                                // placement
                      ge::DT_INT32,                              // data type
                      (void *) data_buf_3.get()};
  // 模拟ACL在执行前设置超时时间的方法，同步方式调用该接口
  ge::GetContext().SetStreamSyncTimeout(15000);
  EXPECT_EQ(model.NnExecute(stream, false, input_tensor, output_tensor), GRAPH_SUCCESS);
}

TEST_F(UtestDavinciModel, NnExecute_LaunchKfcEvent_Ok) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group0", (void*)1), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group1", (void*)2), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group2", (void*)3), GRAPH_SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy1", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group1", "group2", "group3"});

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_EQ(model.hccl_group_ordered_event_list_.size(), 3);
  EXPECT_EQ(model.hccl_group_ordered_stream_list_.size(), 3);
  EXPECT_TRUE(std::find(model.hccl_group_ordered_stream_list_.begin(), model.hccl_group_ordered_stream_list_.end(), (void*)1) != model.hccl_group_ordered_stream_list_.end());
  EXPECT_TRUE(std::find(model.hccl_group_ordered_stream_list_.begin(), model.hccl_group_ordered_stream_list_.end(), (void*)2) != model.hccl_group_ordered_stream_list_.end());
  EXPECT_TRUE(std::find(model.hccl_group_ordered_stream_list_.begin(), model.hccl_group_ordered_stream_list_.end(), (void*)3) != model.hccl_group_ordered_stream_list_.end());

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<GeTensor> input_tensor;
  GeTensor ge_tensor_i;
  GeTensorDesc tensor_desc_i(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_i.SetTensorDesc(tensor_desc_i);
  std::vector<int32_t> input_data_1(1 * 4 * 4 * 8, 0);
  ge_tensor_i.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i);

  std::vector<GeTensor> output_tensor;
  GeTensor ge_tensor_o;
  GeTensorDesc tensor_desc_o(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_o.SetTensorDesc(tensor_desc_i);
  std::vector<int32_t> input_data_2(1 * 4 * 4 * 8, 0);
  ge_tensor_o.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  output_tensor.push_back(ge_tensor_o);

  EXPECT_EQ(model.NnExecute(stream, true, input_buffer, output_buffer, input_tensor, output_tensor), SUCCESS);
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group0");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group1");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group2");
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, LaunchEventForKfcStreamFailed) {
  DavinciModel model(0, nullptr);
  rtEvent_t event_1;
  rtEvent_t event_2;
  model.hccl_group_ordered_event_list_.push_back(event_1);
  model.hccl_group_ordered_event_list_.push_back(event_2);
  model.hccl_group_ordered_stream_list_.push_back((void*)1);
  model.hccl_group_ordered_stream_list_.push_back((void*)2);
  model.is_async_mode_ = true;

  RTS_STUB_RETURN_VALUE(rtEventRecord, rtError_t, 0x78000001);
  EXPECT_NE(model.LaunchEventForHcclGroupOrderedStream((void*)1), SUCCESS);

  RTS_STUB_RETURN_VALUE(rtEventRecord, rtError_t, RT_ERROR_NONE);
  RTS_STUB_RETURN_VALUE(rtStreamWaitEventWithTimeout, rtError_t, 0x78000001);
  EXPECT_NE(model.LaunchEventForHcclGroupOrderedStream((void*)1), SUCCESS);

  model.hccl_group_ordered_event_list_.clear();
  model.hccl_group_ordered_stream_list_.clear();
}

TEST_F(UtestDavinciModel, NnExecute_FmMemoryRefreshOk) {
  GeModelPtr ge_model = ConstructGeModel(2560);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  model.SetFeatureBaseRefreshable(true);
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);

  uint8_t *bufer = nullptr;
  model.UpdateHbmFmMemBases({bufer});
  EXPECT_EQ(model.mem_base_, PtrToValue(nullptr));
  vector<uint8_t *> buffer2;
  buffer2.clear();
  EXPECT_EQ(model.UpdateHbmFmMemBases(buffer2), SUCCESS);
}

TEST_F(UtestDavinciModel, NnExecute_FmMemoryRefresh_CopyOutput) {
  GeModelPtr ge_model = ConstructGeModel(0);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);

  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

//  model.task_list_.resize(1);
  model.SetFeatureBaseRefreshable(true);
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
}

TEST_F(UtestDavinciModel, NnExecute_FmMemoryRefresh_CopyOutputWithEmptyDataBuffer) {
  GeModelPtr ge_model = ConstructGeModel(0);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);

  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  OutputData empty_output_data;

//  model.task_list_.resize(1);
  model.SetFeatureBaseRefreshable(true);
  vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  vector<GeTensor> input_tensor;
  vector<GeTensor> output_tensor;
  for (auto &item : outputs) {
    GeTensor ge_tensor;
    TensorTransUtils::GertTensor2GeTensor(item, ge_tensor);
    output_tensor.emplace_back(std::move(ge_tensor));
  }
  input_tensor = output_tensor;
  EXPECT_EQ(model.NnExecute(stream, false, input_data, empty_output_data,
            input_tensor, output_tensor), SUCCESS);
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
}

TEST_F(UtestDavinciModel, NnExecute_FmMemoryRefresh_CopyOutputWithGertTensor) {
  GeModelPtr ge_model = ConstructGeModel(0);
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);

  setenv("GE_DAVINCI_MODEL_PROFILING", "1", true);
  EXPECT_EQ(model.Init(), SUCCESS);

  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  OutputData empty_output_data;

//  model.task_list_.resize(1);
  model.SetFeatureBaseRefreshable(true);
  vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  vector<gert::Tensor> input_tensor;
  vector<gert::Tensor> output_tensor;
  input_tensor.resize(1);
  output_tensor.resize(1);
  std::vector<uint8_t> output_data_1(96, 0xFF);
  output_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) output_data_1.data()};
  input_tensor[0] = {{{1,4,4,8}, {1,4,4,8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) output_data_1.data()};
  EXPECT_EQ(model.NnExecute(stream, true, input_tensor, output_tensor), SUCCESS);
  unsetenv("GE_DAVINCI_MODEL_PROFILING");
}

TEST_F(UtestDavinciModel, test_copy_model_data) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  map<string, string> options;
  options[SOC_VERSION] = "Ascend910";
  options["ge.exec.enableDump"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ProfilingProperties::Instance().is_load_profiling_ = true;

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  DumpProperties dp;
  dp.InitByOptions();
  model.SetDumpProperties(dp);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT); // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  model.Assign(ge_model);
  EXPECT_EQ(model.GetGlobalStep(), 0U);
  EXPECT_EQ(model.Init(), SUCCESS);
  EXPECT_NE(model.GetGlobalStep(), 0U);
  model.is_getnext_sink_dynamic_ = false;
  model.is_online_infer_dynamic_ = true;

  std::vector<gert::Tensor> input_tensor;
  std::vector<gert::Tensor> output_tensor;
  output_tensor.resize(1);
  input_tensor.resize(1);
  unique_ptr<uint8_t[]> data_buf_1(new (std::nothrow) uint8_t[512]);
  input_tensor[0] = {{{1,2}, {1,2}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_1.get()};
  // unique_ptr<uint8_t[]> data_buf_2(new (std::nothrow) uint8_t[512]);
  // input_tensor[1] = {{{1,4,4,8}, {1,4,4,8}},                // shape
  //                           {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
  //                           gert::kOnDeviceHbm,                                // placement
  //                           ge::DT_INT32,                              // data type
  //                           (void *) data_buf_2.get()};
  unique_ptr<uint8_t[]> data_buf_3(new (std::nothrow) uint8_t[512]);
  output_tensor[0] = {{{1,2}, {1,2}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) data_buf_3.get()};
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  InputData input_data;
  OutputData output_data;

  std::vector<std::vector<int64_t>> tensor_input_dims;
  model.run_context_.dynamic_node_type = DATA;
  model.run_context_.user_input_dims.push_back(std::make_pair(std::string("data"), std::vector<int64_t>({-1, -2})));
  tensor_input_dims.push_back(std::vector<int64_t>({1, 2}));
  model.run_context_.dynamic_shape_dims.push_back(std::vector<int32_t>({1, 2}));

  EXPECT_NE(model.CopyModelData(input_tensor, output_tensor), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, NnExecute_ffts_zcpy) {
  DavinciModel model(0, nullptr);
  BuildDavinciModelWithFftsTask(model, true);
  EXPECT_EQ(model.Init(), SUCCESS);
  uint8_t data[512];
  ZeroCopyOffset zero_copy_offset;
  zero_copy_offset.data_size_ = 512;
  zero_copy_offset.basic_addr_ = &data;
  model.input_data_info_[0] = zero_copy_offset;
  model.output_data_info_[0] = zero_copy_offset;
  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  output_data.blobs[0].placement = 1; // dev mem
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);

  // dev_mem
  model.SetFeatureBaseRefreshable(false);
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  // 后续不再刷新这个表
  EXPECT_EQ(model.input_index_to_active_mem_base_addrs_[0], 0);
  EXPECT_NE(model.output_index_to_active_mem_base_addrs_[0], PtrToValue(output_data.blobs[0].data));
}

TEST_F(UtestDavinciModel, GenOutputMemAllocations_test) {
  DavinciModel model(0, nullptr);
  BuildDavinciModelWithFftsTask(model, false);
  EXPECT_EQ(model.Init(), SUCCESS);

  model.is_getnext_sink_dynamic_ = true;

  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  GeTensorDesc tensor1(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  TensorUtils::SetSize(tensor1, 512);

  OpDescPtr op_desc = CreateOpDesc("data", REFDATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor1);
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor1);
  op_desc->SetInputOffset({0, 8});
  op_desc->SetOutputOffset({0 , 8});

  std::vector<OpDescPtr> output_op_list;
  output_op_list.emplace_back(op_desc);
  EXPECT_EQ(model.GenOutputMemAllocations(output_op_list), SUCCESS);

  model.is_getnext_sink_dynamic_ = false;
  EXPECT_EQ(model.GenOutputMemAllocations(output_op_list), SUCCESS);
}

TEST_F(UtestDavinciModel, GenSliceMemAllocationsZeroCopy_test) {
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);

  {
    uint8_t mem[1024] {};
    std::vector<OpDescPtr> output_op_list;
    std::set<uint64_t> output_outside_addrs;
    std::shared_ptr<ComputeGraph> graph(new (std::nothrow) ComputeGraph("graph"));
    NodePtr data1 = CreateNodeV2(*graph, "data1", DATA, 0, 1);
    NodePtr data2 = CreateNodeV2(*graph, "data2", DATA, 0, 1);
    NodePtr netoutput = CreateNodeV2(*graph, "netoutput", NETOUTPUT, 2, 0);
    data1->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
    data2->GetOpDesc()->MutableOutputDesc(0)->SetAttr(ATTR_IS_ZERO_COPY_BLOCK, GeAttrValue::CreateFrom<bool>(true));
    netoutput->AddLinkFrom(data1);
    netoutput->AddLinkFrom(data2);

    DavinciModel model(0, nullptr);
    output_op_list.clear();
    output_outside_addrs.clear();
    model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(mem);
    model.runtime_param_.mem_size = sizeof(mem);

    uint32_t data_op_index = 0U;
    std::map<uint32_t, OpDescPtr> data_by_index;
    EXPECT_EQ(model.InitDataOp(graph, data1, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitDataOp(graph, data2, data_op_index, data_by_index, output_outside_addrs), SUCCESS);
    EXPECT_EQ(model.InitNetOutput(graph, netoutput, output_op_list, output_outside_addrs), SUCCESS);
    EXPECT_TRUE(model.copy_only_addrs_.copy_only_addrs.empty()); // All outputs can zero-copy

    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);
    model.Assign(ge_model);
    model.is_getnext_sink_dynamic_ = true;
    model.mem_base_ = model.runtime_param_.mem_base;
    EXPECT_EQ(model.GenSliceOutputMemAllocations(output_op_list), SUCCESS); // All outputs can zero-copy
    model.is_getnext_sink_dynamic_ = false;
    EXPECT_EQ(model.GenSliceOutputMemAllocations(output_op_list), SUCCESS); // All outputs can zero-copy
  }

  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, NnExecute_ffts_no_zcpy) {
  DavinciModel model(0, nullptr);
  BuildDavinciModelWithFftsTask(model, false);
  EXPECT_EQ(model.Init(), SUCCESS);
  uint8_t data[512];
  ZeroCopyOffset zero_copy_offset;
  zero_copy_offset.data_size_ = 512;
  zero_copy_offset.basic_addr_ = &data;
  model.input_data_info_[0] = zero_copy_offset;
  model.output_data_info_[0] = zero_copy_offset;
  rtStream_t stream = nullptr;
  InputData input_data;
  OutputData output_data;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(output_data, outputs), SUCCESS);
  EXPECT_EQ(output_data.blobs.size(), 1);
  EXPECT_EQ(output_data.blobs[0].length, 512);
  EXPECT_EQ(outputs.size(), 1);
  output_data.blobs[0].placement = 0; // host mem
  input_data.blobs = output_data.blobs;
  EXPECT_EQ(input_data.blobs.size(), 1);
  model.mem_base_ = reinterpret_cast<uintptr_t>(malloc(1024));
  model.runtime_param_.fm_memory_infos[0].memory_base = PtrToPtr<void, uint8_t>(ValueToPtr(model.mem_base_));

  // host_mem disable mem_reuse
  model.is_async_mode_ = true;
  std::vector<GeTensor> input_tensor = {};
  std::vector<GeTensor> output_tensor = {};
  EXPECT_EQ(model.NnExecute(stream, false, input_data, output_data,
            input_tensor, output_tensor), SUCCESS);
  EXPECT_NE(model.input_index_to_active_mem_base_addrs_[0], PtrToValue(input_data.blobs[0].data));
  EXPECT_NE(model.output_index_to_active_mem_base_addrs_[0], PtrToValue(output_data.blobs[0].data));
  free(ValueToPtr(model.mem_base_));
}

TEST_F(UtestDavinciModel, FreeInnerFeatureMapMem_fail) {
  DavinciModel davinci_model(0, nullptr);
  const char_t *const kEnvRecordPath = "TIMEOUT";
  char_t record_path[MMPA_MAX_PATH] = "timeout";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
  void *mem = new (std::nothrow)  int[10];
  davinci_model.mem_base_ = reinterpret_cast<uintptr_t>(mem);
  davinci_model.is_inner_mem_base_ = true;
  davinci_model.FreeInnerFeatureMapMem();
  EXPECT_EQ(davinci_model.mem_base_, 0);
  unsetenv(kEnvRecordPath);

  g_runtime_stub_mock = "rtStreamSynchronizeWithTimeout";
  void *mem1 = new (std::nothrow)  int[10];
  davinci_model.mem_base_ = reinterpret_cast<uintptr_t>(mem1);
  davinci_model.mem_base_ = reinterpret_cast<uintptr_t>(mem1);
  davinci_model.is_inner_mem_base_ = true;
  davinci_model.FreeInnerFeatureMapMem();
  EXPECT_NE(davinci_model.mem_base_, 0);
}

TEST_F(UtestDavinciModel, NoNeedToCopyInputOutput) {
  OutputData data;
  data.blobs.emplace_back(DataBuffer());
  DavinciModel davinci_model(0, nullptr);
  MemAllocationSlice mem_allocation_slice;
  mem_allocation_slice.id = 0;
  mem_allocation_slice.offset = 0;
  mem_allocation_slice.data_size = 0;
  davinci_model.output_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  davinci_model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  davinci_model.output_no_tiling_flag_ = {false};
  davinci_model.output_shape_info_ = {GeShape({1,4,4,8})};
  davinci_model.has_output_node_ = true;
  std::vector<GeTensor> tensor = {};
  EXPECT_EQ(davinci_model.CopyOutputData(data, tensor), SUCCESS);
  std::vector<DataBuffer> input_data {DataBuffer()};
  EXPECT_EQ(davinci_model.CopyInputForNoZeroCopy(input_data, davinci_model.input_indexes_to_copy_info_, tensor), SUCCESS);
  davinci_model.is_async_mode_ = true;
  EXPECT_EQ(davinci_model.CopyOutputData(data, tensor), SUCCESS);

  uint8_t args1[70] = {123};
  std::vector<DataBuffer> input_data2 {DataBuffer(&args1, 70, false)};
  davinci_model.input_indexes_to_copy_info_.clear();
  mem_allocation_slice.data_size = 64;
  uint8_t buf[100];
  davinci_model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  std::vector<uint64_t> active_base_addr_vec;
  for (size_t i = 0; i < davinci_model.logical_mem_allocations_.size(); i++) {
    active_base_addr_vec.emplace_back(davinci_model.allocation_ids_to_active_base_addr_[i]);
  }
  active_base_addr_vec.emplace_back((uint64_t)buf);
  davinci_model.allocation_ids_to_active_base_addr_ =
    reinterpret_cast<uint64_t*>(static_cast<void*>(active_base_addr_vec.data()));

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = 32U;
  mem_allocation0.tensor_size = 32;
  mem_allocation0.logical_addr = 0x23;
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  //davinci_model.allocation_ids_to_active_base_addr_.emplace_back((uint64_t)buf);
  EXPECT_EQ(davinci_model.CopyInputForNoZeroCopy(input_data2, davinci_model.input_indexes_to_copy_info_, tensor), SUCCESS);
}

TEST_F(UtestDavinciModel, NoNeedToCopyInputOutputWithemptyData) {
  OutputData data;
  DavinciModel davinci_model(0, nullptr);
  MemAllocationSlice mem_allocation_slice;
  mem_allocation_slice.id = 0;
  mem_allocation_slice.offset = 0;
  mem_allocation_slice.data_size = 0;
  davinci_model.output_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  davinci_model.input_indexes_to_copy_info_.emplace(0, mem_allocation_slice);
  davinci_model.output_no_tiling_flag_ = {false};
  davinci_model.output_shape_info_ = {GeShape({1,4,4,8})};
  davinci_model.has_output_node_ = true;
  std::vector<GeTensor> tensor = {};
  GeTensorDesc td(GeShape({1,4,4,8}), FORMAT_ND, DT_FLOAT);
  GeTensor geTensor(td);
  tensor.emplace_back(geTensor);

  EXPECT_EQ(davinci_model.CopyOutputData(data, tensor), SUCCESS);
  std::vector<DataBuffer> input_data;
  EXPECT_EQ(davinci_model.CopyInputForNoZeroCopy(input_data, davinci_model.input_indexes_to_copy_info_, tensor), SUCCESS);
  davinci_model.is_async_mode_ = true;
  EXPECT_EQ(davinci_model.CopyOutputData(data, tensor), SUCCESS);
}

TEST_F(UtestDavinciModel, InitModelInputsMergeCopyHostMem_input_fusion_size_zero) {
  ge::GetThreadLocalContext().SetGraphOption({{OPTION_EXEC_INPUT_FUSION_SIZE, "0"}});
  DavinciModel davinci_model(0, nullptr);
  davinci_model.InitModelInputsMergeCopyHostMem();
  EXPECT_EQ(davinci_model.input_merge_copy_mem_base_, nullptr);  // no merge copy
  ge::GetThreadLocalContext().SetGraphOption({});
}

// model with 3 input
// input 0: size: input_fusion_size - 1, merge copy
// input 1: size: input_fusion_size, merge copy
// input 2: size: input_fusion_size + 1, non-merge copy
// device mem layout: input1-input0-input2
TEST_F(UtestDavinciModel, InputMergeCopy) {
  const uint64_t input_fusion_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test
  std::map<std::string, std::string> options_map;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);  // set input fusion size

  const uint64_t input0_size = input_fusion_size - 1U;  // smaller than input_fusion_size
  const uint64_t input1_size = input_fusion_size;       // equal to input_fusion_size
  const uint64_t input2_size = input_fusion_size + 1U;  // bigger than input_fusion_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  // init mem locations
  //  | mem_allocation1 | mem_allocation0 | mem_allocation2 |
  //  |                 |                 | <-- start_logic_addr + mem_allocation0.data_size
  //  |                 | <-- start_logic_addr
  //  | <-- start_logic_addr - mem_allocation1.data_size
  // merge copy: input 0, input 1
  // non mergr copy: input 2
  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  EXPECT_NE(davinci_model.input_merge_copy_mem_base_, nullptr);
  EXPECT_EQ(davinci_model.input_merge_copy_mem_size_, total_size);
  EXPECT_EQ(davinci_model.fisrt_input_index_of_merge_copy_, 1U);
  EXPECT_EQ(davinci_model.input_index_to_merge_copy_offset_.size(), 2U);
  EXPECT_EQ(davinci_model.input_index_to_merge_copy_offset_.find(2),
            davinci_model.input_index_to_merge_copy_offset_.end());  // input 2 is non-merge-copy

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  // test fail path
  input_data.blobs.emplace_back(DataBuffer());                                           // zero
  input_data.blobs.emplace_back(DataBuffer(input_buffer, max_input_size + 32U, false));  // input length is bigger
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));           // valid input
  EXPECT_EQ(davinci_model.HandleInputData(input_data), PARAM_INVALID);

  // re-init input
  input_data.blobs.clear();

  // test success path
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));      // merge copy, h2h2d ok
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U));  // non-merge-copy, not h2d, d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));      // non-merge-copy, h2d
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetGraphOption({});  // restore option
}

TEST_F(UtestDavinciModel, GenInputMemAllocations_InValid) {
  DavinciModel davinci_model(0, nullptr);
  std::map<uint32_t, OpDescPtr> index_to_data;
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  index_to_data.emplace(0, op_desc);
  EXPECT_EQ(davinci_model.GenInputMemAllocations(index_to_data), PARAM_INVALID);
}
namespace {
void BuildGraphRefdataToMemcpy(ComputeGraphPtr &graph, shared_ptr<domi::ModelTaskDef> &model_task_def) {
  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);  // sizeof(float) * 1 * 4 * 4 * 8 = 512
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", REFDATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName({"memcpy"});
    op_desc->SetSrcIndex({0});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }
}
}  // namespace

TEST_F(UtestDavinciModel, CheckRefDataDisableZeroCopy_ReportError) {
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  BuildGraphRefdataToMemcpy(graph, model_task_def);

  model.Assign(ge_model);
  EXPECT_NE(model.Init(), SUCCESS);
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, CheckRefDataWithHostInputIndex_ReportError) {
  std::map<std::string, std::string> graph_options;
  graph_options["ge.exec.hostInputIndexes"] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 560);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  BuildGraphRefdataToMemcpy(graph, model_task_def);

  const auto &node = graph->FindNode("data");
  const auto &op_desc = node->GetOpDesc();
  op_desc->SetOutputOffset({2510});

  model.Assign(ge_model);
  EXPECT_NE(model.Init(), SUCCESS);
  graph_options["ge.exec.hostInputIndexes"] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, test_GenFrozenInputIndex_no_frozen) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  ProfilingProperties::Instance().is_load_profiling_ = true;
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  BuildGraphRefdataToMemcpy(graph, model_task_def);
  auto memcpy_node = graph->FindFirstNodeMatchType(MEMCPYASYNC);
  memcpy_node->GetOpDesc()->SetInputOffset({});

  std::unordered_set<uint32_t> frozen_input_indexes_;
  frozen_input_indexes_.insert(1);
  model.SetNoFrozenInputIndexes(frozen_input_indexes_);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  EXPECT_EQ(model.refreshable_input_index_and_allocation_ids_.size(), 1);
  EXPECT_EQ(model.zero_copy_input_indexes_no_frozen_.size(), 1);
  EXPECT_EQ(model.refreshable_input_index_no_frozen_and_allocation_ids_.size(), 1);
}

TEST_F(UtestDavinciModel, ChangeMemCpyTaskToAddr_failed_memcpy_inputoffset_empty) {
  RTS_STUB_RETURN_VALUE(rtMemcpy, rtError_t, RT_ERROR_NONE);
  dlog_setlevel(0,0,0);
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ProfilingProperties::Instance().is_load_profiling_ = true;
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  BuildGraphRefdataToMemcpy(graph, model_task_def);
  auto memcpy_node = graph->FindFirstNodeMatchType(MEMCPYASYNC);
  memcpy_node->GetOpDesc()->SetInputOffset({});

  dlog_setlevel(0,0,0);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // check memcpy task not changed to addr task
  const domi::TaskDef &task_def1 = model_task_def->task(0);
  EXPECT_EQ(static_cast<ModelTaskType>(task_def1.type()), ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  const domi::MemcpyAsyncDef &memcpy_async1 = task_def1.memcpy_async();
  EXPECT_EQ(static_cast<tagRtMemcpyKind>(memcpy_async1.kind()), RT_MEMCPY_DEVICE_TO_DEVICE);
  dlog_setlevel(0,3,0);
}

TEST_F(UtestDavinciModel, ChangeMemCpyTaskToAddr_failed_memcpy_inputoffset_not_right) {
  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");

  ProfilingProperties::Instance().is_load_profiling_ = true;
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2560);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  BuildGraphRefdataToMemcpy(graph, model_task_def);
  auto memcpy_node = graph->FindFirstNodeMatchType(MEMCPYASYNC);
  memcpy_node->GetOpDesc()->SetInputOffset({512});

  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // check memcpy task not changed to addr task
  const domi::TaskDef &task_def1 = model_task_def->task(0);
  EXPECT_EQ(static_cast<ModelTaskType>(task_def1.type()), ModelTaskType::MODEL_TASK_MEMCPY_ASYNC);
  const domi::MemcpyAsyncDef &memcpy_async1 = task_def1.memcpy_async();
  EXPECT_EQ(static_cast<tagRtMemcpyKind>(memcpy_async1.kind()), RT_MEMCPY_DEVICE_TO_DEVICE);
}

TEST_F(UtestDavinciModel, SinkTimeProf_DoNothing_WhenProfilingOff) {
  DavinciModel model(0, nullptr);
  ge::diagnoseSwitch::DisableProfiling();
  uint64_t records = 0UL;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    ++records;
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(check_func);
  model.SinkTimeProfile(0, 0);
  EXPECT_EQ(records, 0);
}

TEST_F(UtestDavinciModel, RecordExceptionDumperInfo_Failed_MockRtMemcpyFailed) {
  ExtraOpInfo extra_op_info{};
  auto arg_holder = std::unique_ptr<uint8_t[]>(new uint8_t[24]());
  TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
  extra_op_info.args = reinterpret_cast<uintptr_t>(arg_holder.get());
  extra_op_info.args_size = 24;
  DavinciModel model(0, nullptr);
  setenv("CONSTANT_FOLDING_PASS", "mock_fail", 1);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  OpDescPtr op_desc = CreateOpDesc("fakeOP", "FakeOp");
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({512});
  op_desc->SetOutputOffset({512});
  model.InitExceptionDumpInfo(op_desc, reinterpret_cast<uintptr_t>(arg_holder.get()), 24, {}, extra_op_info);
  EXPECT_EQ(extra_op_info.args_before_execute, "");
  EXPECT_EQ(extra_op_info.tiling_key, 0);
  EXPECT_EQ(extra_op_info.tiling_data, "");
  unsetenv("CONSTANT_FOLDING_PASS");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
}

TEST_F(UtestDavinciModel, static_graph_memory_extend) {
  auto graph = gert::ShareGraph::SimpleVariableAssignGraph();
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 404480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
  (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
  // extend size statis memory
  DavinciModel model1(0, nullptr);
  model1.Assign(ge_model);
  EXPECT_EQ(model1.Init(), SUCCESS);
  EXPECT_EQ(model1.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model1.GetSessionId(), 0)->active_memorys_.size(), 1);

  DavinciModel model2(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 7168));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 3072});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 5120, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 6144, 1024});
  (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  model2.Assign(ge_model);
  EXPECT_EQ(model2.Init(), SUCCESS);

  // test1, model2 reuse model1 feature map memory, extend 2048
  EXPECT_EQ(model2.GetFmMemoryInfos().size(), 3);
  auto allocator = SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model2.GetSessionId(), 0);
  EXPECT_EQ(allocator->active_memorys_.size(), 3);
  auto max_addr = std::max(allocator->active_memorys_[0].active_addr + allocator->active_memorys_[0].total_size,
                           allocator->active_memorys_[1].active_addr + allocator->active_memorys_[1].total_size);
  max_addr = std::max(max_addr, allocator->active_memorys_[2].active_addr + allocator->active_memorys_[2].total_size);
  EXPECT_EQ(model2.GetFmMemoryInfos()[1].memory_base, model1.GetFmMemoryInfos()[0].memory_base);
  EXPECT_EQ(model2.GetRuntimeParam().GetMemAddr(1024), model2.GetFmMemoryInfos()[0].memory_base + 1024);
  EXPECT_EQ(model2.GetRuntimeParam().GetMemAddr(2048), model2.GetFmMemoryInfos()[1].memory_base);
  EXPECT_EQ(model2.GetRuntimeParam().GetMemAddr(5120), model2.GetFmMemoryInfos()[2].memory_base);
  EXPECT_EQ(model2.GetRuntimeParam().GetMemAddr(5120 + 512), model2.GetFmMemoryInfos()[2].memory_base + 512);
  EXPECT_EQ(model2.GetRuntimeParam().GetMemAddr(6144), max_addr);

  DavinciModel model3(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 512});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2560, 0});
  (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  // test2, reuse feature map memory
  model3.Assign(ge_model);
  EXPECT_EQ(model3.Init(), SUCCESS);
  EXPECT_EQ(model3.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model3.GetSessionId(), 0)->active_memorys_.size(), 3);
  EXPECT_EQ(model3.GetFmMemoryInfos()[0].memory_base, model1.GetFmMemoryInfos()[0].memory_base);

  DavinciModel model4(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2098688));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2098176});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2098176, 512});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2098688, 0});
  (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  // test3, no reuse feature map memory
  model4.Assign(ge_model);
  EXPECT_EQ(model4.Init(), SUCCESS);
  EXPECT_EQ(model4.GetFmMemoryInfos().size(), 2);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model4.GetSessionId(), 0)->active_memorys_.size(), 4);
  }

  graph_options[STATIC_MEMORY_POLICY] = "";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, static_graph_memory_extend_fail) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
    // extend size statis memory
    DavinciModel model1(0, nullptr);
    model1.Assign(ge_model);
    EXPECT_EQ(model1.Init(), SUCCESS);
    EXPECT_EQ(model1.GetFmMemoryInfos().size(), 1);
    EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
        .GetMemAllocator(model1.GetSessionId(), 0)->active_memorys_.size(), 1);

    DavinciModel model2(0, nullptr);
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2103296));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
    sub_memory_infos.clear();
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 3072});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 5120, 1024 * 1024 * 2});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 5120 + 1024 * 1024 * 2, 1024});
    (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

    g_runtime_stub_mock = "rtMalloc";
    model2.Assign(ge_model);
    EXPECT_EQ(model2.Init(), ACL_ERROR_GE_MEMORY_ALLOCATION);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, static_graph_memory_no_extend) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
  // no extend static memory
  DavinciModel model1(0, nullptr);
  model1.Assign(ge_model);
  EXPECT_EQ(model1.Init(), SUCCESS);
  EXPECT_EQ(model1.GetFmMemoryInfos().size(), 1); // error
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model1.GetSessionId(), 0)->active_memorys_.size(), 2);

  DavinciModel model2(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 3072));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 0});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  model2.Assign(ge_model);
  EXPECT_EQ(model2.Init(), SUCCESS);

  // test1, reuse model1 feature map memory
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model2.GetSessionId(), 0)->active_memorys_.size(), 2);
  EXPECT_EQ(model2.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(model2.GetFmMemoryInfos()[0].memory_base, model1.GetFmMemoryInfos()[0].memory_base);

  DavinciModel model3(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 512});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2560, 0});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  // test2, reuse model1 feature map memory
  model3.Assign(ge_model);
  EXPECT_EQ(model3.Init(), SUCCESS);
  EXPECT_EQ(model3.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model3.GetSessionId(), 0)->active_memorys_.size(), 2);
  EXPECT_EQ(model3.GetFmMemoryInfos()[0].memory_base, model1.GetFmMemoryInfos()[0].memory_base);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, no_split_info_om) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
  // no extend size statis memory
  DavinciModel model1(0, nullptr);
  model1.Assign(ge_model);
  EXPECT_EQ(model1.Init(), SUCCESS);
  EXPECT_EQ(model1.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model1.GetSessionId(), 0)->active_memorys_.size(), 2);

  DavinciModel model2(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 7168));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));

  model2.Assign(ge_model);
  EXPECT_EQ(model2.Init(), SUCCESS);

  // test1, model2 malloc feature map memory
  EXPECT_EQ(model2.GetFmMemoryInfos().size(), 1); // fm
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model2.GetSessionId(), 0)->active_memorys_.size(), 3);
  EXPECT_NE(model2.GetFmMemoryInfos()[0].memory_base, model1.GetFmMemoryInfos()[0].memory_base);

  DavinciModel model3(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));

  // test2, reuse model2 feature map memory
  model3.Assign(ge_model);
  EXPECT_EQ(model3.Init(), SUCCESS);
  EXPECT_EQ(model3.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model3.GetSessionId(), 0)->active_memorys_.size(), 3);
  EXPECT_EQ(model3.GetFmMemoryInfos()[0].memory_base, model2.GetFmMemoryInfos()[0].memory_base);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, split_info_om_no_extend) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);

  // no extend size statis memory
  DavinciModel model1(0, nullptr);
  model1.Assign(ge_model);
  EXPECT_EQ(model1.Init(), SUCCESS);
  EXPECT_EQ(model1.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model1.GetSessionId(), 0)->active_memorys_.size(), 0);

  DavinciModel model2(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 7168));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 3072});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 5120, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 6144, 1024});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  model2.Assign(ge_model);
  EXPECT_EQ(model2.Init(), SUCCESS);

  // test1, model2 malloc feature map memory
  EXPECT_EQ(model2.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model2.GetSessionId(), 0)->active_memorys_.size(), 0);
  EXPECT_NE(model2.GetFmMemoryInfos()[0].memory_base, model1.GetFmMemoryInfos()[0].memory_base);

  DavinciModel model3(0, nullptr);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2560));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
  sub_memory_infos.clear();
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 2048});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 512});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2560, 0});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);

  // test2, malloc feature map memory
  model3.Assign(ge_model);
  EXPECT_EQ(model3.Init(), SUCCESS);
  EXPECT_EQ(model3.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(model3.GetSessionId(), 0)->active_memorys_.size(), 0);
  EXPECT_NE(model3.GetFmMemoryInfos()[0].memory_base, model1.GetFmMemoryInfos()[0].memory_base);
}

TEST_F(UtestDavinciModel, static_graph_memory_outer_alloc) {
  auto graph = gert::ShareGraph::SimpleVariableAssignGraph();
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 404480));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 4096));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 4096));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 4096});
  (void)AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  {
  // extend size statis memory
  DavinciModel model1(0, nullptr);
  model1.Assign(ge_model);
  ModelParam param;
  void *mem_base = nullptr;
  rtMalloc(&mem_base, 4096U, RT_MEMORY_HBM, 45U);
  EXPECT_TRUE(mem_base!=nullptr);
  param.mem_base = reinterpret_cast<uintptr_t>(mem_base);
  EXPECT_EQ(model1.Init(param), SUCCESS);
  EXPECT_EQ(model1.GetFmMemoryInfos().size(), 1);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
    .GetMemAllocator(model1.GetSessionId(), 0)->active_memorys_.size(), 0);

  rtFree(mem_base);
  }
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, static_graph_memory_fixed_sub_memory_infos) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
  std::vector<std::vector<int64_t>> sub_memory_infos;
  sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024, 1});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024, 1});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
  sub_memory_infos.push_back({RT_MEMORY_HBM, 4096, 1024});
  (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  // extend size statis memory
  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  ModelParam param{};
  // fixed Feature Memory addr set
  std::vector<uint8_t> fixmem(2048, 0);
  param.fixed_mem_base = (uintptr_t)(fixmem.data());
  param.fixed_mem_size = 2048;
  EXPECT_EQ(model.Init(param), SUCCESS);
  EXPECT_EQ(model.GetFmMemoryInfos().size(), 2);
  EXPECT_EQ(model.runtime_param_.fixed_fm_memory_infos.size(), 2);
  std::vector<uint8_t> mem(2048, 0);
  size_t used_size = 0U;
  EXPECT_EQ(model.UpdateHbmFmMemBases((uintptr_t)(mem.data()), 2048, used_size), SUCCESS);
}

TEST_F(UtestDavinciModel, multi_davinci_model_load_unload) {
  StaticMemoryPolicy4Guarder guarder;
  gert::GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, 1, 0); // enable pa va check
  {
    // graph 1
    ComputeGraphPtr graph1 = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model1 = MakeShared<GeModel>();
    ge_model1->SetGraph(graph1);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model1, ATTR_MODEL_MEMORY_SIZE, 5120));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model1, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 5120});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 5120, 1024});
    (void) AttrUtils::SetListListInt(ge_model1, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
    const auto model_def = MakeShared<domi::ModelTaskDef>();
    ge_model1->SetModelTaskDef(model_def);

    // extend size statis memory
    DavinciModel model1(0, nullptr);
    model1.Assign(ge_model1);
    ModelParam param1{};
    EXPECT_EQ(model1.Init(param1), SUCCESS);

    // graph 2
    ComputeGraphPtr graph2 = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model2 = MakeShared<GeModel>();
    ge_model2->SetGraph(graph2);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model2, ATTR_MODEL_MEMORY_SIZE, 512));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model2, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
    std::vector<std::vector<int64_t>> sub_memory_infos2;
    sub_memory_infos2.push_back({RT_MEMORY_HBM, 0, 512});
    sub_memory_infos2.push_back({RT_MEMORY_HBM, 512, 512});
    (void) AttrUtils::SetListListInt(ge_model2, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos2);
    const auto model_def2 = MakeShared<domi::ModelTaskDef>();
    ge_model2->SetModelTaskDef(model_def2);

    // extend size statis memory
    DavinciModel model2(0, nullptr);
    model2.Assign(ge_model2);
    ModelParam param2{};
    runtime_stub.GetSlogStub().Clear();
    EXPECT_EQ(model2.Init(param2), SUCCESS);
    auto find = runtime_stub.GetSlogStub().FindLog(3, "virtual and physical page mapping check failed");
    EXPECT_TRUE(find == -1);

    // graph3
    ComputeGraphPtr graph3 = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model3 = MakeShared<GeModel>();
    ge_model3->SetGraph(graph3);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model3, ATTR_MODEL_MEMORY_SIZE, 6656));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model3, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0));
    std::vector<std::vector<int64_t>> sub_memory_infos3;
    sub_memory_infos3.push_back({RT_MEMORY_HBM, 0, 512});
    sub_memory_infos3.push_back({RT_MEMORY_HBM, 512, 512});
    sub_memory_infos3.push_back({RT_MEMORY_HBM, 1024, 5120});
    sub_memory_infos3.push_back({RT_MEMORY_HBM, 6144, 512});
    sub_memory_infos3.push_back({RT_MEMORY_HBM, 512, 512});
    (void) AttrUtils::SetListListInt(ge_model3, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos3);
    const auto model_def3 = MakeShared<domi::ModelTaskDef>();
    ge_model3->SetModelTaskDef(model_def3);

    // extend size statis memory
    DavinciModel model3(0, nullptr);
    model3.Assign(ge_model3);
    ModelParam param3{};
    runtime_stub.GetSlogStub().Clear();
    EXPECT_EQ(model3.Init(param3), SUCCESS);
    find = runtime_stub.GetSlogStub().FindLog(3, "virtual and physical page mapping check failed");
    EXPECT_TRUE(find == -1);
  }
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(UtestDavinciModel, expandable_memory_allocator) {
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 1);
  ASSERT_NE(mem_allocator1, nullptr);
  auto mem_allocator2 =
     SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 2);
  ASSERT_NE(mem_allocator2, nullptr);
  auto ptr = mem_allocator1->MallocMemory("test", 1024UL);
  ASSERT_NE(ptr, nullptr);
  ptr = mem_allocator2->MallocMemory("test", 2048UL);
  ASSERT_NE(ptr, nullptr);
  ptr = mem_allocator1->MallocMemory("test", 4096UL);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(mem_allocator1->used_memory_size_, 4096);
  ASSERT_EQ(mem_allocator2->used_memory_size_, 2048);
  EXPECT_EQ(mem_allocator2->FreeMemory(), ge::SUCCESS);
  EXPECT_EQ(mem_allocator1->FreeMemory(), ge::SUCCESS);
  ASSERT_EQ(mem_allocator1->used_memory_size_, 4096);
  EXPECT_EQ(mem_allocator1->FreeMemory(), ge::SUCCESS);
  ASSERT_EQ(mem_allocator1->used_memory_size_, 0);
  ASSERT_EQ(mem_allocator2->used_memory_size_, 0);
  ASSERT_EQ(mem_allocator1->active_memory_allocator_.ActiveMemorySize(), 0);
  ASSERT_EQ(mem_allocator2->active_memory_allocator_.ActiveMemorySize(), 0);
}

TEST_F(UtestDavinciModel, ExpandableActiveMemoryAllocator_Use1GSuccess_CheckPgType) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      *handle = (rtDrvMemHandle) new uint8_t[8];
      pg_type_local = prop->pg_type;
      return 0;
    }
    uint32_t pg_type_local = 0U;
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 1, RT_MEMORY_HBM, kDrv1GPageSize);
  ASSERT_NE(mem_allocator1, nullptr);
  auto ptr = mem_allocator1->MallocMemory("test", 4096UL);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->pg_type_local, 2U);
  EXPECT_EQ(mem_allocator1->FreeMemory(), ge::SUCCESS);

  auto mem_allocator2 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 2);
  ASSERT_NE(mem_allocator2, nullptr);
  ptr = mem_allocator2->MallocMemory("test", 1024UL, true);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->pg_type_local, 1U);

  EXPECT_EQ(mem_allocator2->FreeMemory(), ge::SUCCESS);
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, ExpandableActiveMemoryAllocator_Use1GMallocSuccessThenFailed) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      pg_type_local = prop->pg_type;
      ++call_count;
      if (call_count == 2 && prop->pg_type == 2) {
        return -1;
      }
      *handle = (rtDrvMemHandle) new uint8_t[8];
      return 0;
    }
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) {
      *free = 64UL * 1024U * 1024U * 1024U;
      *total = 64UL * 1024U * 1024U * 1024U;
      return 0;
    }
    rtError_t rtGetRtCapability(rtFeatureType_t featureType, int32_t featureInfo, int64_t *value) {
      *value = 1;
      return 0;
    }
    uint32_t call_count = 0U;
    uint32_t pg_type_local = 0U;
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 1, RT_MEMORY_HBM, kDrv1GPageSize);
  ASSERT_NE(mem_allocator1, nullptr);
  auto ptr = mem_allocator1->MallocMemory("test", 1 *  1024U * 1024U * 1024U, true);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->pg_type_local, 2U);
  EXPECT_EQ(mock_runtime->call_count, 1U);

  // 预期申请1G 大页失败后，转为申请2M大页，最后申请成功。
  auto ptr2 = mem_allocator1->MallocMemory("test", 1 *  1024U * 1024U * 1024U, true);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(mock_runtime->pg_type_local, 1U);
  EXPECT_EQ(mock_runtime->call_count, 3U);

  EXPECT_EQ(mem_allocator1->FreeMemory(), ge::SUCCESS);
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, ExpandableActiveMemoryAllocator_Use1GMallocFailed_NotSupport) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      pg_type_local = prop->pg_type;
      ++call_count;
      size_local = size;
      if (prop->pg_type == 2) {
        return -1;
      }
      *handle = (rtDrvMemHandle) new uint8_t[8];
      return 0;
    }
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) {
      *free = 64UL * 1024U * 1024U * 1024U;
      *total = 64UL * 1024U * 1024U * 1024U;
      return 0;
    }
    rtError_t rtGetRtCapability(rtFeatureType_t featureType, int32_t featureInfo, int64_t *value) {
      *value = 0;
      return 0;
    }
    uint32_t call_count = 0U;
    uint32_t pg_type_local = 0U;
    uint32_t size_local = 0;
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 1, RT_MEMORY_HBM, kDrv1GPageSize);
  ASSERT_NE(mem_allocator1, nullptr);
  auto ptr = mem_allocator1->MallocMemory("test", 20480, true);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->pg_type_local, 1U);
  EXPECT_EQ(mock_runtime->call_count, 2U);
  EXPECT_EQ(mock_runtime->size_local, kDrv1GPageSize);
  EXPECT_EQ(mem_allocator1->FreeMemory(), ge::SUCCESS);
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, not_support_expandable_memory_allocator) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtReserveMemAddress(void** devPtr, size_t size, size_t alignment, void *devAddr, uint64_t flags) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 1);
  ASSERT_NE(mem_allocator1, nullptr);
  auto mem_allocator2 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(1, 1);
  ASSERT_NE(mem_allocator2, nullptr);
  EXPECT_EQ(mem_allocator1, mem_allocator2);
  auto ptr = mem_allocator1->MallocMemory("test", 1024UL);
  ASSERT_EQ(ptr, nullptr);

  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(1, 7)->device_id_, 7);
  EXPECT_EQ(SessionMemAllocator<ActiveMemoryAllocator>::Instance()
      .GetMemAllocator(1, 7)->memory_type_, RT_MEMORY_HBM);
  mem_allocator1->FreeMemory();
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, expandable_memory_allocator_fail) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMapMem(void* devPtr, size_t size, size_t offset, rtDrvMemHandle handle, uint64_t flags) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(2, 1);
  ASSERT_NE(mem_allocator1, nullptr);
  dlog_setlevel(0,0,0);
  auto ptr = mem_allocator1->MallocMemory("test", 60UL * 1024UL);
  dlog_setlevel(0,3,0);
  EXPECT_EQ(ptr, nullptr);
  ptr = mem_allocator1->MallocMemory("test", 60UL * 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(ptr, nullptr);
  ge::RuntimeStub::Reset();
  mem_allocator1->FreeMemory();
}

TEST_F(UtestDavinciModel, MallocDynamicMemorySuccess) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
      call_count++;
      *dev_ptr = new uint8_t[size];
      memset_s(*dev_ptr, size, 0, size);
      return RT_ERROR_NONE;
    }

    rtError_t rtFree(void *dev_ptr) override {
      --call_count;
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }
    uint32_t call_count = 0U;
    uint32_t size_local = 0;
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);

  DavinciModel model(0, nullptr);
  for (size_t i = 0U; i < 2 * 1024 * 1024 / 512; ++i) {
    auto ptr = model.MallocDynamicMemory(1U);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(mock_runtime->call_count, 1);
  }
  auto ptr = model.MallocDynamicMemory(1U);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mock_runtime->call_count, 2);

  model.FreeDynamicWorkspaceMemory();
  EXPECT_EQ(mock_runtime->call_count, 0);
  ge::RuntimeStub::Reset();
}

TEST_F(UtestDavinciModel, test_restore_variable) {
  DavinciModel model(0, nullptr);
  model.session_id_ = 8765U;
  auto var_op = std::make_shared<OpDesc>("var1", VARIABLE);
  ASSERT_NE(var_op, nullptr);
  GeTensorDesc desc;
  ge::TensorUtils::SetSize(desc, 64);
  var_op->AddOutputDesc(desc);
  var_op->SetOutputOffset({137438959572U});
  auto node = NodeUtils::CreatNodeWithoutGraph(var_op);
  ASSERT_NE(node, nullptr);
  std::vector<NodePtr> variable_nodes{node};
  model.runtime_param_.logic_var_base = 137438959572U;
  model.runtime_param_.var_size = 1024U;
  EXPECT_EQ(model.RestoreDeviceVarMem(variable_nodes, default_parm), SUCCESS);
  VarManager::Instance(model.session_id_)->Destory();
}

TEST_F(UtestDavinciModel, davinci_dump_by_original_name) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  uint32_t mem_offset = 0U;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN));   // ccKernelType::TE
    CHAIN(NODE("_var_0", VARIABLE)->NODE("allreduce", HCOMALLREDUCE)->EDGE(0, 0)->NODE("relu", RELU)); // HCCL
    CHAIN(NODE("_arg_1", CONSTANTOP)->EDGE(0, 1)->NODE("add_n")->EDGE(0, 1)->NODE("relu")-> // ccKernelType::CUSTOMIZED
          NODE("square", SQUARE)->EDGE(0, 0)->      // ccKernelType::AI_CPU
          NODE("reshape", RESHAPE)->EDGE(0, 0)->    // ccKernelType::CUST_AI_CPU
          NODE("deque", FRAMEWORKOP)->EDGE(0, 0)->  // KERNEL_EX
          NODE("memcpy", MEMCPYASYNC)->EDGE(0, 0)-> // MEMCPY_ASYNC
          NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  SetKnownOpKernel(graph, mem_offset);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 128));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, kMemoryHostSVMFeatureMapLogicBase));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  {
    const auto &node = graph->FindNode("_arg_0");
    const auto &op_desc = node->GetOpDesc();
    AttrUtils::SetInt(op_desc, "globalworkspace_type", 0);
    AttrUtils::SetInt(op_desc, "globalworkspace_size", 1);
    AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  }

  {
    const auto &node = graph->FindNode("_var_0");
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetInputOffset({1024 + kMemoryVarLogicBase});
    AttrUtils::SetBool(op_desc, VAR_ATTR_VAR_IS_BROADCAST, true);
    VarManager::Instance(graph->GetSessionID())->SetAllocatedGraphId(op_desc->GetName(), graph->GetGraphID());
  }

  {
    const auto &node = graph->FindNode("_arg_1");
    const auto &op_desc = node->GetOpDesc();

    std::vector<uint8_t> weights_value(64, 'A');
    GeTensorDesc data_desc = node->GetOpDesc()->GetOutputDesc(0);
    GeTensorPtr weight_value = MakeShared<GeTensor>(data_desc, weights_value.data(), weights_value.size());
    EXPECT_TRUE(AttrUtils::SetTensor(node->GetOpDesc(), ATTR_NAME_WEIGHTS, weight_value));
  }

  const std::shared_ptr<OpsKernelInfoStore> ops_kernel_store = MakeShared<OpsKernelInfoStoreStub>();
  {
    const auto &node = graph->FindNode("allreduce");
    const auto &op_desc = node->GetOpDesc();
    AttrUtils::SetInt(op_desc, "globalworkspace_type", 0);
    AttrUtils::SetInt(op_desc, "globalworkspace_size", 1);
    op_desc->SetExtAttr("OpsKernelInfoStorePtr", ops_kernel_store.get());

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
    task_def.set_stream_id(op_desc->GetStreamId());
    task_def.set_private_def("hccl_task"); // for GetPrivateDefByTaskDef

    auto &hccl_def = *task_def.mutable_kernel_hccl();
    hccl_def.set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();
    DavinciModel model(0, nullptr);

    int32_t run_mode = static_cast<uint32_t>(domi::ImplyType::TVM);
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode));
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    std::vector<std::string> original_addn = {"addn_ori1", "addn_ori2"};
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_addn));

    ge::DumpProperties dump_properties;
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
    dump_properties.AddPropertyValue("LAYER_OP_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"addn_ori1", "addn_ori2"});
    dump_properties.SetDumpMode("all");
    //ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
    model.SetDumpProperties(dump_properties);
    EXPECT_TRUE(model.OpNeedDump(op_desc));
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);

    auto &task_def = *model_def->add_task();
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def.set_stream_id(op_desc->GetStreamId());

    auto &kernel_def = *task_def.mutable_kernel();
    kernel_def.set_stub_func("stub_func");
    kernel_def.set_args_size(64);
    string args(64, '1');
    kernel_def.set_args(args.data(), 64);

    auto &context = *kernel_def.mutable_context();
    context.set_op_index(op_desc->GetId());
    context.set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    uint16_t args_offset[9] = {0};
    context.set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }
}

TEST_F(UtestDavinciModel, run_with_task_no_data) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  auto mem_allocator1 =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(0, 0);
  {
    ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(graph);

    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 5120));
    EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 1024));
    std::vector<std::vector<int64_t>> sub_memory_infos;
    sub_memory_infos.push_back({RT_MEMORY_HBM, 0, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 1024, 1024, 1});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 2048, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 3072, 1024});
    sub_memory_infos.push_back({RT_MEMORY_HBM, 4096, 1024});
    (void) AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
    const auto model_def = MakeShared<domi::ModelTaskDef>();
    ge_model->SetModelTaskDef(model_def);

    DavinciModel model(0, g_local_call_back);
    model.Assign(ge_model);
    model.SetId(1);
    model.isGraphLevelSat_ = true;

    domi::ModelTaskDef model_task_def;
    domi::TaskDef *task = model_task_def.add_task();
    task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
    task->_impl_.stream_id_ = 0;
    rtStream_t stream = nullptr;
    model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model.stream_list_ = { stream };
    TaskInfoPtr task_info = MakeShared<ProfilerTraceTaskInfo>();
    model.task_list_.push_back(task_info);
    model.has_output_node_ = true;
    OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
    std::vector<int64_t> input_offset;
    input_offset.emplace_back(0);
    GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset(input_offset);
    model.InitOutputTensorInfo(op_desc);
    model.Init();
    EXPECT_EQ(model.ModelRunStart(), SUCCESS);
    sleep(1);
    EXPECT_EQ(model.ModelRunStop(), SUCCESS);
    graph_options[STATIC_MEMORY_POLICY] = "";
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDavinciModel, test_device_id) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  auto mem_allocator =
      SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(1, 2);
  ASSERT_NE(mem_allocator, nullptr);
  LogicalMemorys logical_memorys;
  logical_memorys.emplace_back(0, 10240);
  std::vector<std::pair<uint8_t *, size_t>> mem_size;
  (void)mem_allocator->MallocMemory("", logical_memorys, mem_size, 0);
  EXPECT_EQ(mem_allocator->expandable_memory_allocator_.device_id_, 2);
}

TEST_F(UtestDavinciModel, NnExecute_HostInput_GeTensor_OK) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  std::map<std::string, std::string> options_map;
  options_map["ge.exec.hostInputIndexes"] = "0;1;2";
  GetThreadLocalContext().SetGraphOption(options_map);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 16384);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({3000}); // 不支持零拷贝, 不生成io段
    op_desc->SetOutputOffset({4000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({7000}); // 支持零拷贝
    op_desc->SetOutputOffset({8000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data2", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({9000}); // 支持零拷贝
    op_desc->SetOutputOffset({10000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 获取model args manager 地址
  auto &model_update_data = model.args_manager_.update_policies_to_model_data_[1];
  EXPECT_NE(model_update_data, nullptr);
  EXPECT_EQ(model_update_data->h2d_copy_datas.size(), 1);
  uint64_t host_input_device_addr = model_update_data->h2d_copy_datas[0].device_addr;
  void *host_input_host_addr = model_update_data->h2d_copy_datas[0].host_addr;
  uint64_t host_input_len = model_update_data->h2d_copy_datas[0].len;
  EXPECT_EQ(host_input_host_addr != nullptr, true);
  EXPECT_GE(host_input_len, 1024);

  // io段的active地址已经更新为model args mnanager的物理地址
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], host_input_device_addr);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[2], host_input_device_addr + 512);

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<GeTensor> input_tensor;
  GeTensor ge_tensor_i_0;
  GeTensorDesc tensor_desc_i_0(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_0.SetPlacement(kPlacementHost);
  ge_tensor_i_0.SetTensorDesc(tensor_desc_i_0);
  std::vector<int32_t> input_data_0(1 * 4 * 4 * 8, 0);
  ge_tensor_i_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_0);

  GeTensor ge_tensor_i_1;
  GeTensorDesc tensor_desc_i_1(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_1.SetTensorDesc(tensor_desc_i_1);
  std::vector<int32_t> input_data_1(1 * 4 * 4 * 8, 1);
  ge_tensor_i_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_1);

  GeTensor ge_tensor_i_2;
  GeTensorDesc tensor_desc_i_2(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_2.SetTensorDesc(tensor_desc_i_2);
  std::vector<int32_t> input_data_2(1 * 4 * 4 * 8, 2);
  ge_tensor_i_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_2);

  std::vector<GeTensor> output_tensor;
  GeTensor ge_tensor_o;
  GeTensorDesc tensor_desc_o(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_o.SetTensorDesc(tensor_desc_o);
  std::vector<int32_t> output_data(1 * 4 * 4 * 8, 0);
  ge_tensor_o.SetData(reinterpret_cast<uint8_t *>(output_data.data()), output_data.size() * sizeof(int32_t));
  output_tensor.push_back(ge_tensor_o);

  EXPECT_EQ(model.NnExecute(stream, true, input_buffer, output_buffer, input_tensor, output_tensor), SUCCESS);

  // 判断model args manger host地址是否被刷新
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    uint64_t dst_address = PtrToValue(args.dst_address);
    void *input1_addr = const_cast<void*>(args.src_address);
    void *input2_addr = (void*)(static_cast<uint8_t *>(const_cast<void *>(args.src_address)) + 512);
    if (dst_address == host_input_device_addr) {
      EXPECT_EQ(std::memcmp(input1_addr, (void*)input_data_1.data(), 512), 0);
      EXPECT_EQ(std::memcmp(input2_addr, (void*)input_data_2.data(), 512), 0);
    }
  }

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
  mmSetEnv(kEnvRecordPath, "", 1);
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, NnExecute_HostInput_GeTensorHugeData_OK) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.Clear();
  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  const char * const kEnvRecordPath1 = "MOCK_MEMCPY_HUGE";
  char record_path1[MMPA_MAX_PATH] = {"1"};
  mmSetEnv(kEnvRecordPath1, &record_path1[0U], MMPA_MAX_PATH);

  std::map<std::string, std::string> options_map;
  options_map["ge.exec.hostInputIndexes"] = "0;1;2";
  GetThreadLocalContext().SetGraphOption(options_map);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2150419968);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2150415872);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({3000}); // 不支持零拷贝, 不生成io段
    op_desc->SetOutputOffset({4000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({7000}); // 支持零拷贝
    op_desc->SetOutputOffset({8000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    GeTensorDesc tensor_1(GeShape({4200000,4,4,8}), FORMAT_NCHW, DT_FLOAT);
    OpDescPtr op_desc = CreateOpDesc("data2", DATA);
    op_desc->AddInputDesc(tensor_1);
    op_desc->AddOutputDesc(tensor_1);
    op_desc->SetInputOffset({9000}); // 支持零拷贝
    op_desc->SetOutputOffset({10000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 获取model args manager 地址
  auto &model_update_data = model.args_manager_.update_policies_to_model_data_[1];
  EXPECT_NE(model_update_data, nullptr);
  EXPECT_EQ(model_update_data->h2d_copy_datas.size(), 1);
  uint64_t host_input_device_addr = model_update_data->h2d_copy_datas[0].device_addr;
  void *host_input_host_addr = model_update_data->h2d_copy_datas[0].host_addr;
  uint64_t host_input_len = model_update_data->h2d_copy_datas[0].len;
  EXPECT_EQ(host_input_host_addr != nullptr, true);
  EXPECT_GE(host_input_len, 1024);

  // io段的active地址已经更新为model args mnanager的物理地址
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], host_input_device_addr);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[2], host_input_device_addr + 512);

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<GeTensor> input_tensor;
  GeTensor ge_tensor_i_0;
  GeTensorDesc tensor_desc_i_0(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_0.SetPlacement(kPlacementHost);
  ge_tensor_i_0.SetTensorDesc(tensor_desc_i_0);
  std::vector<int32_t> input_data_0(1 * 4 * 4 * 8, 0);
  ge_tensor_i_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_0);

  GeTensor ge_tensor_i_1;
  GeTensorDesc tensor_desc_i_1(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_1.SetTensorDesc(tensor_desc_i_1);
  std::vector<int32_t> input_data_1(1 * 4 * 4 * 8, 1);
  ge_tensor_i_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_1);

  GeTensor ge_tensor_i_2;
  GeTensorDesc tensor_desc_i_2(GeShape({4200000,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_2.SetTensorDesc(tensor_desc_i_2);
  std::vector<int32_t> input_data_2(4200000 * 4 * 4 * 8, 2);
  ge_tensor_i_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_2);

  std::vector<GeTensor> output_tensor;
  GeTensor ge_tensor_o;
  GeTensorDesc tensor_desc_o(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_o.SetTensorDesc(tensor_desc_o);
  std::vector<int32_t> output_data(1 * 4 * 4 * 8, 0);
  ge_tensor_o.SetData(reinterpret_cast<uint8_t *>(output_data.data()), output_data.size() * sizeof(int32_t));
  output_tensor.push_back(ge_tensor_o);

  EXPECT_EQ(model.NnExecute(stream, true, input_buffer, output_buffer, input_tensor, output_tensor), SUCCESS);

  // 判断model args manger host地址是否被刷新
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    uint64_t dst_address = PtrToValue(args.dst_address);
    void *input1_addr = const_cast<void*>(args.src_address);
    void *input2_addr = (void*)(static_cast<uint8_t *>(const_cast<void *>(args.src_address)) + 512);
    if (dst_address == host_input_device_addr) {
      EXPECT_EQ(std::memcmp(input1_addr, (void*)input_data_1.data(), 512), 0);
      EXPECT_EQ(std::memcmp(input2_addr, (void*)input_data_2.data(), 2150400000), 0);
    }
  }

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
  unsetenv(kEnvRecordPath1);
  mmSetEnv(kEnvRecordPath, "", 1);
  runtime_stub.Clear();
}

TEST_F(UtestDavinciModel, NnExecute_HostInput_GertTensor_OK) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  std::map<std::string, std::string> options_map;
  options_map["ge.exec.hostInputIndexes"] = "0;1;2";
  GetThreadLocalContext().SetGraphOption(options_map);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 16384);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({3000}); // 不支持零拷贝, 不生成io段
    op_desc->SetOutputOffset({4000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({7000}); // 支持零拷贝
    op_desc->SetOutputOffset({8000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data2", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({9000}); // 支持零拷贝
    op_desc->SetOutputOffset({10000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 获取model args manager 地址
  auto &model_update_data = model.args_manager_.update_policies_to_model_data_[1];
  EXPECT_NE(model_update_data, nullptr);
  EXPECT_EQ(model_update_data->h2d_copy_datas.size(), 1);
  uint64_t host_input_device_addr = model_update_data->h2d_copy_datas[0].device_addr;
  void *host_input_host_addr = model_update_data->h2d_copy_datas[0].host_addr;
  uint64_t host_input_len = model_update_data->h2d_copy_datas[0].len;
  EXPECT_EQ(host_input_host_addr != nullptr, true);
  EXPECT_GE(host_input_len, 1024);

  // io段的active地址已经更新为model args mnanager的物理地址
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], host_input_device_addr);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[2], host_input_device_addr + 512);

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(3);
  gert_outputs.resize(1);

  std::vector<int32_t> input_data_0(1 * 4 * 4 * 8, 0);
  gert_inputs[0] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) input_data_0.data()};

  std::vector<int32_t> input_data_1(1 * 4 * 4 * 8, 1);
  gert_inputs[1] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 4 * 4 * 8, 2);
  gert_inputs[2] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},               // shape
                            {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) input_data_2.data()};

  std::vector<int32_t> output_data(1 * 4 * 4 * 8, 0);
  gert_outputs[0] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},                // shape
                        {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                        gert::kOnDeviceHbm,                                // placement
                        ge::DT_FLOAT,                              // data type
                        nullptr};

  EXPECT_EQ(model.NnExecute(stream, true, gert_inputs, gert_outputs), SUCCESS);

  // 判断model args manger host地址是否被刷新
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    uint64_t dst_address = PtrToValue(args.dst_address);
    void *input1_addr = const_cast<void*>(args.src_address);
    void *input2_addr = (void*)(static_cast<uint8_t *>(const_cast<void *>(args.src_address)) + 512);
    if (dst_address == host_input_device_addr) {
      EXPECT_EQ(std::memcmp(input1_addr, (void*)input_data_1.data(), 512), 0);
      EXPECT_EQ(std::memcmp(input2_addr, (void*)input_data_2.data(), 512), 0);
    }
  }

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
  mmSetEnv(kEnvRecordPath, "", 1);
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, NnExecute_HostInput_GertTensorHugeData_OK) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  const char *const kEnvRecordPath1 = "MOCK_MEMCPY_HUGE";
  char record_path1[MMPA_MAX_PATH] = {"1"};
  mmSetEnv(kEnvRecordPath1, &record_path1[0U], MMPA_MAX_PATH);


  std::map<std::string, std::string> options_map;
  options_map["ge.exec.hostInputIndexes"] = "0;1;2";
  GetThreadLocalContext().SetGraphOption(options_map);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2150419968);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 2150415872);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({3000}); // 不支持零拷贝, 不生成io段
    op_desc->SetOutputOffset({4000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({7000}); // 支持零拷贝
    op_desc->SetOutputOffset({8000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    GeTensorDesc tensor_1(GeShape({4200000,4,4,8}), FORMAT_NCHW, DT_FLOAT);
    OpDescPtr op_desc = CreateOpDesc("data2", DATA);
    op_desc->AddInputDesc(tensor_1);
    op_desc->AddOutputDesc(tensor_1);
    op_desc->SetInputOffset({9000}); // 支持零拷贝
    op_desc->SetOutputOffset({10000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 获取model args manager 地址
  auto &model_update_data = model.args_manager_.update_policies_to_model_data_[1];
  EXPECT_NE(model_update_data, nullptr);
  EXPECT_EQ(model_update_data->h2d_copy_datas.size(), 1);
  uint64_t host_input_device_addr = model_update_data->h2d_copy_datas[0].device_addr;
  void *host_input_host_addr = model_update_data->h2d_copy_datas[0].host_addr;
  uint64_t host_input_len = model_update_data->h2d_copy_datas[0].len;
  EXPECT_EQ(host_input_host_addr != nullptr, true);
  EXPECT_GE(host_input_len, 1024);

  // io段的active地址已经更新为model args mnanager的物理地址
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[1], host_input_device_addr);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[2], host_input_device_addr + 512);

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(3);
  gert_outputs.resize(1);

  std::vector<int32_t> input_data_0(1 * 4 * 4 * 8, 0);
  gert_inputs[0] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) input_data_0.data()};

  std::vector<int32_t> input_data_1(1 * 4 * 4 * 8, 1);
  gert_inputs[1] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(4200000 * 4 * 4 * 8, 2);
  gert_inputs[2] = {{{4200000, 4, 4, 8}, {4200000, 4, 4, 8}},               // shape
                            {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                            gert::kOnHost,                                // placement
                            ge::DT_FLOAT,                              // data type
                            (void *) input_data_2.data()};

  std::vector<int32_t> output_data(1 * 4 * 4 * 8, 0);
  gert_outputs[0] = {{{1, 4 ,4, 8}, {1, 4, 4, 8}},                // shape
                        {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                        gert::kOnDeviceHbm,                                // placement
                        ge::DT_FLOAT,                              // data type
                        nullptr};

  EXPECT_EQ(model.NnExecute(stream, true, gert_inputs, gert_outputs), SUCCESS);

  // 判断model args manger host地址是否被刷新
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    uint64_t dst_address = PtrToValue(args.dst_address);
    void *input1_addr = const_cast<void*>(args.src_address);
    void *input2_addr = (void*)(static_cast<uint8_t *>(const_cast<void *>(args.src_address)) + 512);
    if (dst_address == host_input_device_addr) {
      EXPECT_EQ(std::memcmp(input1_addr, (void*)input_data_1.data(), 512), 0);
      EXPECT_EQ(std::memcmp(input2_addr, (void*)input_data_2.data(), 2150400000), 0);
    }
  }

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
  mmSetEnv(kEnvRecordPath, "", 1);
  unsetenv(kEnvRecordPath1);
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, NnExecute_HostInput_GeTensor_kernelbin_OK) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.Clear();

  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  std::map<std::string, std::string> options_map;
  options_map["ge.exec.hostInputIndexes"] = "1;2";
  GetThreadLocalContext().SetGraphOption(options_map);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 16384);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({5000}); // 支持零拷贝
    op_desc->SetOutputOffset({6000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({7000}); // 支持零拷贝
    op_desc->SetOutputOffset({8000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({9000}); // 支持零拷贝
    op_desc->SetOutputOffset({10000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  model.args_manager_.SetFuncHandle((void*)100);
  EXPECT_EQ(model.Init(), SUCCESS);


  // 获取model args manager 地址
  uint64_t host_input_device_addr = PtrToValue(model.args_manager_.activate_mem_base_device_addrs_dev_) + 64; // 64：原来的active membase长度

  // io段的active地址已经更新为model args mnanager的物理地址
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[2], host_input_device_addr);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[3], host_input_device_addr + 512);

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<GeTensor> input_tensor;
  GeTensor ge_tensor_i_0;
  GeTensorDesc tensor_desc_i_0(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_i_0.SetTensorDesc(tensor_desc_i_0);
  std::vector<int32_t> input_data_0(1 * 4 * 4 * 8, 0);
  ge_tensor_i_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_0);

  GeTensor ge_tensor_i_1;
  GeTensorDesc tensor_desc_i_1(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_1.SetTensorDesc(tensor_desc_i_1);
  std::vector<int32_t> input_data_1(1 * 4 * 4 * 8, 0);
  ge_tensor_i_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_1);

  GeTensor ge_tensor_i_2;
  GeTensorDesc tensor_desc_i_2(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_2.SetTensorDesc(tensor_desc_i_2);
  std::vector<int32_t> input_data_2(1 * 4 * 4 * 8, 0);
  ge_tensor_i_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_2);

  std::vector<GeTensor> output_tensor;
  GeTensor ge_tensor_o;
  GeTensorDesc tensor_desc_o(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_o.SetTensorDesc(tensor_desc_o);
  std::vector<int32_t> output_data(1 * 4 * 4 * 8, 0);
  ge_tensor_o.SetData(reinterpret_cast<uint8_t *>(output_data.data()), output_data.size() * sizeof(int32_t));
  output_tensor.push_back(ge_tensor_o);

  EXPECT_EQ(model.NnExecute(stream, true, input_buffer, output_buffer, input_tensor, output_tensor), SUCCESS);

  // active mem base 是否跟新
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpyRecords()) {
    uint64_t dst_address = PtrToValue(args.dst_address);
    void *input1_addr = const_cast<void*>(args.src_address);
    void *input2_addr = (void*)(static_cast<uint8_t *>(const_cast<void *>(args.src_address)) + 512);
    if (dst_address == host_input_device_addr) {
      EXPECT_EQ(std::memcmp(input1_addr, (void*)input_data_1.data(), 512), 0);
      EXPECT_EQ(std::memcmp(input2_addr, (void*)input_data_2.data(), 512), 0);
    }
  }

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
  mmSetEnv(kEnvRecordPath, "", 1);
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}


TEST_F(UtestDavinciModel, NnExecute_HostInput_GeTensor_kernelbin_pciebar_OK) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);

  const char * const kEnvRecordPath = "RT_MEMORY_HBM";
  char record_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &record_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&record_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvRecordPath, fail_collect_path.c_str(), 1);

  std::map<std::string, std::string> options_map;
  options_map["ge.exec.hostInputIndexes"] = "1;2";
  GetThreadLocalContext().SetGraphOption(options_map);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 20480);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 16384);

  shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data0", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({5000}); // 支持零拷贝
    op_desc->SetOutputOffset({6000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    GeTensorDesc tensor(GeShape({1,4,4,1}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 128);
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({8000}); // 支持零拷贝
    op_desc->SetOutputOffset({9000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    GeTensorDesc tensor(GeShape({1,4,4,1}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 128);
    OpDescPtr op_desc = CreateOpDesc("data1", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({10000}); // 支持零拷贝
    op_desc->SetOutputOffset({11000});
    NodePtr node = graph->AddNode(op_desc);
  }
  {
    OpDescPtr op_desc = CreateOpDesc("memcpy0", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({512});
    op_desc->SetOutputOffset({512});
    AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
    NodePtr node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(1024);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }
  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({2048});
    op_desc->SetSrcName( { "memcpy0" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  DavinciModel model(0, nullptr);
  model.Assign(ge_model);
  model.args_manager_.SetFuncHandle((void*)100);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 获取model args manager 地址
  auto &model_update_data = model.args_manager_.update_policies_to_model_data_[1];
  EXPECT_NE(model_update_data, nullptr);
  EXPECT_EQ(model_update_data->h2d_copy_datas.size(), 1);
  uint64_t host_input_device_addr = model_update_data->h2d_copy_datas[0].device_addr;
  void *host_input_host_addr = model_update_data->h2d_copy_datas[0].host_addr;
  uint64_t host_input_len = model_update_data->h2d_copy_datas[0].len;
  EXPECT_EQ(host_input_host_addr != nullptr, true);
  EXPECT_GE(host_input_len, 192);

  // io段的active地址已经更新为model args mnanager的物理地址
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[2], host_input_device_addr);
  EXPECT_EQ(model.allocation_ids_to_active_base_addr_[3], host_input_device_addr + 64);

  // 检测launch args
  EXPECT_EQ(model.args_manager_.addr_update_op_args_.argsSize, 272); // sizeof(KernelLaunchOpArgs) 80, active_mem_base_addr_len_align32b 64  host_input_size_ 192
  // tiling检查待补充

  // 判断model index和offset是否正确
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetRtMemcpySyncRecords()) {
    uint64_t dst_address = PtrToValue(args.dst_address);
    void *src_address = const_cast<void*>(args.src_address);
    std::cout << "src_address " <<src_address << "dst_address" << dst_address << endl;
    if (dst_address == PtrToValue(model.args_manager_.model_args_device_offset_)) {
      uint64_t *offset_ptr = (uint64_t *)dst_address;
      for (size_t i = 8 ; i < 16; i ++) {
        EXPECT_EQ(*(offset_ptr + i), 0);
      }
    } else if (dst_address == PtrToValue(model.args_manager_.model_args_device_index_)) {
      uint32_t *index_ptr = (uint32_t *)dst_address;
      for (size_t i = 8 ; i < 16; i ++) {
        EXPECT_EQ(*(index_ptr + i * 2), i * 2 * sizeof(uint32_t));
        EXPECT_EQ(*(index_ptr + i * 2 + 1), (i * 2 + 1) * sizeof(uint32_t));
      }
    }
  }

  rtStream_t stream = nullptr;
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = 0;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = 0;

  std::vector<GeTensor> input_tensor;
  GeTensor ge_tensor_i_0;
  GeTensorDesc tensor_desc_i_0(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_i_0.SetTensorDesc(tensor_desc_i_0);
  std::vector<int32_t> input_data_0(1 * 4 * 4 * 8, 0);
  ge_tensor_i_0.SetData(reinterpret_cast<uint8_t *>(input_data_0.data()), input_data_0.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_0);

  GeTensor ge_tensor_i_1;
  GeTensorDesc tensor_desc_i_1(GeShape({1,4,4,1}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_1.SetTensorDesc(tensor_desc_i_1);
  std::vector<int32_t> input_data_1(1 * 4 * 4 * 1, 1);
  ge_tensor_i_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_1);

  GeTensor ge_tensor_i_2;
  GeTensorDesc tensor_desc_i_2(GeShape({1,4,4,1}), FORMAT_NCHW, DT_FLOAT);
  tensor_desc_i_1.SetPlacement(kPlacementHost);
  ge_tensor_i_2.SetTensorDesc(tensor_desc_i_2);
  std::vector<int32_t> input_data_2(1 * 4 * 4 * 1, 2);
  ge_tensor_i_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  input_tensor.push_back(ge_tensor_i_2);

  std::vector<GeTensor> output_tensor;
  GeTensor ge_tensor_o;
  GeTensorDesc tensor_desc_o(GeShape({1,4,4,8}), FORMAT_NCHW, DT_FLOAT);
  ge_tensor_o.SetTensorDesc(tensor_desc_o);
  std::vector<int32_t> output_data(1 * 4 * 4 * 8, 0);
  ge_tensor_o.SetData(reinterpret_cast<uint8_t *>(output_data.data()), output_data.size() * sizeof(int32_t));
  output_tensor.push_back(ge_tensor_o);

  EXPECT_EQ(model.NnExecute(stream, true, input_buffer, output_buffer, input_tensor, output_tensor), SUCCESS);

  // active membase是否被更新
  void *input1_addr =   static_cast<uint8_t *>(model.args_manager_.launched_args_unique_ptr_.get()) + sizeof(ge::ModelArgsManager::KernelLaunchOpArgs) + 64;
  void *input2_addr = static_cast<void *>(static_cast<uint8_t *>(input1_addr) + 64);
  EXPECT_EQ(std::memcmp(input1_addr, (void*)input_data_1.data(), 64), 0);
  EXPECT_EQ(std::memcmp(input2_addr, (void*)input_data_2.data(), 64), 0);

  options_map.clear();
  GetThreadLocalContext().SetGraphOption(options_map);
  mmSetEnv(kEnvRecordPath, "", 1);
  runtime_stub.Clear();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DWithMergeH2DEnabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);
  const uint64_t input_fusion_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test
  std::map<std::string, std::string> options_map;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);
  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);  // set input fusion size

  const uint64_t input0_size = input_fusion_size - 1U;  // smaller than input_fusion_size
  const uint64_t input1_size = input_fusion_size;       // equal to input_fusion_size
  const uint64_t input2_size = input_fusion_size + 1U;  // bigger than input_fusion_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  // const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;
  const auto total_size = mem_allocation0.data_size + mem_allocation1.data_size + mem_allocation2.data_size;


  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  // EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  // test success path
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));      // merge copy, h2h2d ok
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U));  // non-merge-copy, not h2d, d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));      // non-merge-copy, h2d, batch h2d
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetGraphOption({});  // restore option
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DWithMergeH2DEnabled2) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);
  const uint64_t input_fusion_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test
  std::map<std::string, std::string> options_map;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);
  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);  // set input fusion size

  const uint64_t input0_size = input_fusion_size - 1U;  // smaller than input_fusion_size
  const uint64_t input1_size = input_fusion_size + 2U;       // bigger than input_fusion_size
  const uint64_t input2_size = input_fusion_size + 1U;  // bigger than input_fusion_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  // const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;
  const auto total_size = mem_allocation0.data_size + mem_allocation1.data_size + mem_allocation2.data_size;


  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  // EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  // test success path
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));      // merge copy, h2h2d ok
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false));  // non-merge-copy, batch h2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));      // non-merge-copy, h2d, batch h2d
  // no need merge h2d copy, fusion input num: 1
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 3);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetGraphOption({});  // restore option
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DWithMergeH2DDisabled) {
  dlog_setlevel(0, 0, 0);
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);
  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;  // smaller than input_size
  const uint64_t input1_size = input_size;       // equal to input_size
  const uint64_t input2_size = input_size + 1U;  // bigger than input_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  // EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U));  // d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 2);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DWithDeviceIdIs1) {
  dlog_setlevel(0, 0, 0);
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, RT_ERROR_NONE);
  RuntimeStub::GetInstance()->cur_device_id = 1;
  SetMockRtGetDeviceWay(1);

  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;  // smaller than input_size
  const uint64_t input1_size = input_size;       // equal to input_size
  const uint64_t input2_size = input_size + 1U;  // bigger than input_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  // EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U));  // d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 2);
  EXPECT_EQ(RuntimeStub::GetInstance()->batch_memcpy_device_id, 1);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
  RuntimeStub::GetInstance()->cur_device_id = 0;
  RuntimeStub::GetInstance()->batch_memcpy_device_id = 0;

  SetMockRtGetDeviceWay(0);
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DButNotSupportedWithMergeH2DEnabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  const uint64_t input_fusion_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test
  std::map<std::string, std::string> options_map;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);
  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);  // set input fusion size

  const uint64_t input0_size = input_fusion_size - 1U;  // smaller than input_fusion_size
  const uint64_t input1_size = input_fusion_size;       // equal to input_fusion_size
  const uint64_t input2_size = input_fusion_size + 1U;  // bigger than input_fusion_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  // EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  // test success path
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));      // merge copy, h2h2d ok
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U));  // non-merge-copy, not h2d, d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));      // non-merge-copy, h2d, batch h2d
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetGraphOption({});  // restore option
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DButNotSupportedWithMergeH2DDisabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;  // smaller than input_size
  const uint64_t input1_size = input_size;       // equal to input_size
  const uint64_t input2_size = input_size + 1U;  // bigger than input_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U)); // d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
}


TEST_F(UtestDavinciModel, InputBatchCopyH2DFailedWithMergeH2DEnabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_PARAM_INVALID);
  const uint64_t input_fusion_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test
  std::map<std::string, std::string> options_map;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);
  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);  // set input fusion size

  const uint64_t input0_size = input_fusion_size - 1U;  // smaller than input_fusion_size
  const uint64_t input1_size = input_fusion_size + 1U;       // equal to input_fusion_size
  const uint64_t input2_size = input_fusion_size + 1U;  // bigger than input_fusion_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  // test success path
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));      // merge copy, h2h2d ok
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false));  // non-merge-copy, not h2d, d2d
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));      // non-merge-copy, h2d, batch h2d
  // switch of input_batch_cpy is open and only one item exists, call rtMemcpy
  EXPECT_EQ(davinci_model.HandleInputData(input_data), RT_FAILED);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetGraphOption({});  // restore option
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DFailedWithMergeH2DDisabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_PARAM_INVALID);
  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;  // smaller than input_size
  const uint64_t input1_size = input_size;       // equal to input_size
  const uint64_t input2_size = input_size + 1U;  // bigger than input_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};
  // EXPECT_EQ(davinci_model.HandleInputData(input_data), FAILED);

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false, 1U));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_EQ(davinci_model.HandleInputData(input_data), RT_FAILED);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DEnbledButOneInputWithMergeH2DEnabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_PARAM_INVALID);
  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 25600U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;
  const uint64_t max_input_size = input0_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size;

  ZeroCopyOffset zero_copy_offset0 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  // zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false, 1));  // d2d
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyH2DEnbledButOneInputWithMergeH2DDisabled) {
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_PARAM_INVALID);
  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;
  const uint64_t max_input_size = input0_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size;

  ZeroCopyOffset zero_copy_offset0 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  // zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_EQ(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, InputBatchCopyFallbackFailedWithMergeH2DDisabled) {
  RTS_STUB_RETURN_VALUE(rtMemcpy, rtError_t, -1);
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  const uint64_t input_size = 25600U;
  const uint64_t start_logic_addr = 30902000U;  // random value for test

  std::map<std::string, std::string> options_map;
  const uint64_t input_fusion_size = 0U;
  options_map[OPTION_EXEC_INPUT_FUSION_SIZE] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  const string input_batch_cpy_str = "1";
  std::map<std::string, std::string> options_map_session;
  options_map_session[configure_option::INPUT_BATCH_CPY] = input_batch_cpy_str;
  ge::GetThreadLocalContext().SetSessionOption(options_map_session);

  const uint64_t input0_size = input_size - 1U;  // smaller than input_size
  const uint64_t input1_size = input_size;       // equal to input_size
  const uint64_t input2_size = input_size + 1U;  // bigger than input_size
  const uint64_t max_input_size = input2_size;

  // test init merge copy
  DavinciModel davinci_model(0, nullptr);
  davinci_model.zero_copy_input_indexes_.emplace_back(0);
  davinci_model.zero_copy_input_indexes_.emplace_back(1);
  davinci_model.zero_copy_input_indexes_.emplace_back(2);

  davinci_model.input_index_to_allocation_ids_.emplace_back(0);
  davinci_model.input_index_to_allocation_ids_.emplace_back(1);
  davinci_model.input_index_to_allocation_ids_.emplace_back(2);

  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = input0_size + 32U;
  mem_allocation0.tensor_size = input0_size;
  mem_allocation0.logical_addr = start_logic_addr;

  MemAllocation mem_allocation1;
  mem_allocation1.data_size = input1_size + 32U;
  mem_allocation1.tensor_size = input1_size;
  mem_allocation1.logical_addr = start_logic_addr - mem_allocation1.data_size;

  MemAllocation mem_allocation2;
  mem_allocation2.data_size = input2_size + 32U;
  mem_allocation2.tensor_size = input2_size;
  mem_allocation2.logical_addr = start_logic_addr + mem_allocation0.data_size;

  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation1);
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation2);

  const auto total_size = mem_allocation0.logical_addr + mem_allocation0.data_size - mem_allocation1.logical_addr;

  ZeroCopyOffset zero_copy_offset0 = {};
  ZeroCopyOffset zero_copy_offset1 = {};
  ZeroCopyOffset zero_copy_offset2 = {};

  zero_copy_offset0.data_size_ = input0_size + 32U;
  zero_copy_offset1.data_size_ = input1_size + 32U;
  zero_copy_offset2.data_size_ = input2_size + 32U;

  void *device_buffer = malloc(total_size);  // malloc
  zero_copy_offset1.basic_addr_ = device_buffer;

  davinci_model.input_data_info_[0] = zero_copy_offset0;
  davinci_model.input_data_info_[1] = zero_copy_offset1;
  davinci_model.input_data_info_[2] = zero_copy_offset2;

  // merge copy init success
  davinci_model.InitModelInputsMergeCopyHostMem();
  davinci_model.InitBatchMemcpyH2d();

  // test copy input data
  davinci_model.global_step_addr_ = 0U;  // to skip HandleInputData() -> UpdateStepInfo()
  davinci_model.is_first_execute_ = false;
  davinci_model.is_online_infer_dynamic_ = false;
  InputData input_data = {};

  void *input_buffer = malloc(max_input_size);  // malloc, shared by all inputs
  EXPECT_NE(input_buffer, nullptr);

  input_data.blobs.emplace_back(DataBuffer(input_buffer, input0_size, false));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input1_size, false));
  input_data.blobs.emplace_back(DataBuffer(input_buffer, input2_size, false));
  // all CopyInputData -> rtsMemcpyBatch
  EXPECT_NE(davinci_model.HandleInputData(input_data), SUCCESS);
  EXPECT_EQ(RuntimeStub::GetInstance()->input_mem_copy_batch_count_, 0);

  free(input_buffer);
  free(device_buffer);
  ge::GetThreadLocalContext().SetSessionOption({});
}

TEST_F(UtestDavinciModel, testConstructActiveMemBaseAddrs) {
  DavinciModel davinci_model(0, nullptr);
  MemAllocation mem_allocation0 = {};
  mem_allocation0.data_size = 25600U + 32U; // random value for test
  mem_allocation0.tensor_size = 25600U; // random value for test
  mem_allocation0.logical_addr = 30902000U; // random value for test
  mem_allocation0.type = MemAllocation::Type::INPUT;
  davinci_model.logical_mem_allocations_.emplace_back(mem_allocation0);
  davinci_model.allocation_ids_to_active_base_addr_ = new uint64_t[1];
  davinci_model.allocation_ids_to_active_base_addr_[0] = 123456; // random value for test
  davinci_model.ConstructActiveMemBaseAddrs();
  EXPECT_EQ(davinci_model.allocation_ids_to_active_base_addr_[0], mem_allocation0.logical_addr);
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  davinci_model.ConstructActiveMemBaseAddrs();
  EXPECT_EQ(davinci_model.allocation_ids_to_active_base_addr_[0], 0U);
  davinci_model.logical_mem_allocations_[0].type = MemAllocation::Type::OUTPUT;
  davinci_model.ConstructActiveMemBaseAddrs();
  EXPECT_EQ(davinci_model.allocation_ids_to_active_base_addr_[0], 0U);
  delete[] davinci_model.allocation_ids_to_active_base_addr_;
  davinci_model.allocation_ids_to_active_base_addr_ = nullptr;
}

/**
 * 场景: CheckIoReuseAddrs - 空配置时直接返回成功
 *
 * 说明：
 * - io_same_addr_pairs_ 为空
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 SUCCESS
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_EmptyPairs_Success) {
  DavinciModel model(0, nullptr);

  std::vector<DataBuffer> input_blobs = {{reinterpret_cast<void *>(0x1000), 512, false}};
  std::vector<DataBuffer> output_blobs = {{reinterpret_cast<void *>(0x2000), 512, false}};

  std::vector<GeTensor> empty_ge_inputs;
  std::vector<GeTensor> empty_ge_outputs;
  // 显式传入空 vector，编译器才能匹配到 (Blobs, Blobs, GeTensor, GeTensor) 的重载
  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_ge_inputs, empty_ge_outputs), SUCCESS);

  std::vector<gert::Tensor> empty_gert_inputs;
  std::vector<gert::Tensor> empty_gert_outputs;
  // 显式传入空 vector，匹配 (Blobs, Blobs, gert::Tensor, gert::Tensor) 的重载
  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_gert_inputs, empty_gert_outputs), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 地址相同时校验通过
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}，表示 input[0] 应复用给 output[0]
 * - input[0] 和 output[0] 地址相同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 SUCCESS
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_SameAddress_Success) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0"); // 0复用0
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  void *shared_addr = reinterpret_cast<void *>(0x1000);
  std::vector<DataBuffer> input_blobs = {{shared_addr, 512, false}};
  std::vector<DataBuffer> output_blobs = {{shared_addr, 512, false}}; // 地址相同

  std::vector<GeTensor> empty_tensors;

  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 地址不同时校验失败
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}
 * - input[0] 和 output[0] 地址不同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 FAILED
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_DifferentAddress_Fail) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0"); // 0复用0
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  std::vector<DataBuffer> input_blobs = {{reinterpret_cast<void *>(0x1000), 512, false}};
  std::vector<DataBuffer> output_blobs = {{reinterpret_cast<void *>(0x2000), 512, false}}; // 地址不同
  std::vector<GeTensor> empty_tensors;

  EXPECT_NE(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 输出索引越界
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 5}，但 output_blobs 只有 2 个元素
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 FAILED
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_OutputIndexOutOfRange_Fail) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,5"); // 配置输出索引为 5
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  void *shared_addr = reinterpret_cast<void *>(0x1000);
  std::vector<DataBuffer> input_blobs = {{shared_addr, 512, false}};
  // 只有2个输出，索引5越界
  std::vector<DataBuffer> output_blobs = {{shared_addr, 512, false}, {shared_addr, 512, false}};
  std::vector<GeTensor> empty_tensors;

  EXPECT_NE(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 输入索引越界
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {5, 0}，但 input_blobs 只有 2 个元素
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 FAILED
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_InputIndexOutOfRange_Fail) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "5,0"); // 配置输入索引为 5
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  void *shared_addr = reinterpret_cast<void *>(0x1000);
  // 只有2个输入，索引5越界
  std::vector<DataBuffer> input_blobs = {{shared_addr, 512, false}, {shared_addr, 512, false}};
  std::vector<DataBuffer> output_blobs = {{shared_addr, 512, false}};
  std::vector<GeTensor> empty_tensors;

  EXPECT_NE(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 多对索引全部匹配
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}, {1, 1}
 * - 所有配对的输入输出地址都相同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 SUCCESS
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_MultiplePairs_AllMatch_Success) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|1,1");
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  void *addr0 = reinterpret_cast<void *>(0x1000);
  void *addr1 = reinterpret_cast<void *>(0x2000);
  std::vector<DataBuffer> input_blobs = {{addr0, 512, false}, {addr1, 512, false}};
  std::vector<DataBuffer> output_blobs = {{addr0, 512, false}, {addr1, 512, false}};
  std::vector<GeTensor> empty_tensors;

  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 多对索引部分不匹配
 *
 * 说明：
 * - io_same_addr_pairs_ 包含 {0, 0}, {1, 1}
 * - output[1] 地址与 input[1] 不同
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 FAILED
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_MultiplePairs_PartialMismatch_Fail) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0|1,1");
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  void *addr0 = reinterpret_cast<void *>(0x1000);
  void *addr1 = reinterpret_cast<void *>(0x2000);
  void *addr2 = reinterpret_cast<void *>(0x3000);
  std::vector<DataBuffer> input_blobs = {{addr0, 512, false}, {addr1, 512, false}};
  std::vector<DataBuffer> output_blobs = {{addr0, 512, false}, {addr2, 512, false}}; // 1号位不匹配
  std::vector<GeTensor> empty_tensors;

  EXPECT_NE(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - 空输入/输出 blobs
 *
 * 说明：
 * - input_blobs 或 output_blobs 为空
 *
 * 预期：
 * - CheckIoReuseAddrs 返回 SUCCESS（跳过校验）
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_EmptyBlobs_Success) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  std::vector<DataBuffer> empty_blobs;
  std::vector<DataBuffer> output_blobs = {{reinterpret_cast<void *>(0x1000), 512, false}};
  std::vector<GeTensor> empty_tensors;

  // 输入为空，无法进行地址比对，应直接返回成功
  EXPECT_EQ(model.CheckIoReuseAddrs(empty_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

TEST_F(UtestDavinciModel, CheckIoReuseAddrs_NoConfig_SkipCheck_Success) {
  DavinciModel model(0, nullptr);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  // 关键点：不设置复用属性
  // AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "");
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  void *addr0 = reinterpret_cast<void *>(0x1000);
  void *addr1 = reinterpret_cast<void *>(0x2000);

  std::vector<DataBuffer> input_blobs = {{addr0, 512, false}};
  std::vector<DataBuffer> output_blobs = {{addr1, 512, false}}; // 地址不同
  std::vector<GeTensor> empty_tensors;

  // 因为没配置复用规则，所以即便地址不同，也应该返回 SUCCESS
  EXPECT_EQ(model.CheckIoReuseAddrs(input_blobs, output_blobs, empty_tensors, empty_tensors), SUCCESS);
}

/**
 * - input_tensors 和 output_tensors 非空
 * - 放入同一个 GeTensor 对象（假设 GeTensor 拷贝是浅拷贝或引用，或者地址未变）
 * 预期：SUCCESS
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_GeTensor_SameAddress_Success) {
  DavinciModel model(0, nullptr);

  // 初始化配置
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 构造一个 GeTensor 并分配数据
  GeTensor tensor;
  std::vector<uint8_t> data(100, 1);
  tensor.SetData(data); // 内部会分配内存地址

  // 构造输入输出 Tensor 列表
  // 注意：这里假设 vector push_back 后的 GeTensor 仍然指向同一块 Data 内存
  std::vector<GeTensor> input_tensors = {tensor};
  std::vector<GeTensor> output_tensors = {tensor};
  std::vector<DataBuffer> empty_blobs;

  EXPECT_EQ(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);
}

/**
 * - input[0] 和 output[0] 是两个独立分配内存的 GeTensor
 * 预期：FAILED
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_GeTensor_DiffAddress_Fail) {
  DavinciModel model(0, nullptr);

  // 初始化配置
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 构造两个独立的 GeTensor
  GeTensor input_tensor;
  std::vector<uint8_t> data_in(100, 1);
  input_tensor.SetData(data_in);

  GeTensor output_tensor;
  std::vector<uint8_t> data_out(100, 2);
  output_tensor.SetData(data_out); // 分配了新的地址

  std::vector<GeTensor> input_tensors = {input_tensor};
  std::vector<GeTensor> output_tensors = {output_tensor};
  std::vector<DataBuffer> empty_blobs;

  EXPECT_NE(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - gert::Tensor 地址相同
 *
 * 修复点：
 * - gert::Tensor 不可拷贝，直接通过 vector(size) 构造，避免 push_back 或拷贝初始化
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_GertTensor_SameAddress_Success) {
  DavinciModel model(0, nullptr);

  // 1. 初始化模型配置
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 2. 构造数据 (直接在 vector 内部操作，避免拷贝)
  void *shared_addr = reinterpret_cast<void *>(0x120000);

  // 创建包含 1 个元素的 vector
  std::vector<gert::Tensor> input_tensors(1);
  input_tensors[0].MutableTensorData().SetAddr(shared_addr, nullptr);

  std::vector<gert::Tensor> output_tensors(1);
  output_tensors[0].MutableTensorData().SetAddr(shared_addr, nullptr); // 地址一致

  std::vector<DataBuffer> empty_blobs;

  // 3. 执行校验
  EXPECT_EQ(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);
}

/**
 * 场景: CheckIoReuseAddrs - gert::Tensor 地址不同
 *
 * 修复点：
 * - gert::Tensor 不可拷贝，直接通过 vector(size) 构造
 */
TEST_F(UtestDavinciModel, CheckIoReuseAddrs_GertTensor_DiffAddress_Fail) {
  DavinciModel model(0, nullptr);

  // 1. 初始化模型配置
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(MakeShared<ComputeGraph>("test"));
  AttrUtils::SetStr(ge_model, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, "0,0");
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  model.Assign(ge_model);
  EXPECT_EQ(model.Init(), SUCCESS);

  // 2. 构造数据 (直接在 vector 内部操作)
  std::vector<gert::Tensor> input_tensors(1);
  input_tensors[0].MutableTensorData().SetAddr(reinterpret_cast<void *>(0x120000), nullptr);

  std::vector<gert::Tensor> output_tensors(1);
  output_tensors[0].MutableTensorData().SetAddr(reinterpret_cast<void *>(0x999999), nullptr); // 地址不同

  std::vector<DataBuffer> empty_blobs;

  // 3. 执行校验
  EXPECT_NE(model.CheckIoReuseAddrs(empty_blobs, empty_blobs, input_tensors, output_tensors), SUCCESS);
}

}  // namespace ge
