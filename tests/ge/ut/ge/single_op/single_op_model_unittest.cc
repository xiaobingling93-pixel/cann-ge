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
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "runtime/rt.h"
#include "single_op/single_op_model.h"
#include "aicpu_task_struct.h"
#include "single_op/task/tbe_task_builder.h"
#include "single_op/task/rts_kernel_task_builder.h"
#include "single_op/task/op_task.h"
#include "framework/common/helper/model_helper.h"
#include "single_op/stream_resource.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "framework/generator/ge_generator.h"
#include "common/utils/executor_utils.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "macro_utils/dt_public_unscope.h"
#include "framework/ge_runtime_stub/include/faker/fake_allocator.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;
using namespace ge;

namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));
}  // namespace

class UtestSingleOpModel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

//rt api stub
rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId) {
  return RT_ERROR_NONE;
}

Status check_input_dataops_index(SingleOpModel &model) {
  const auto nodes = model.root_graph_->GetDirectNode();
  for (const auto &node : nodes) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto op_type = op_desc->GetType();
    if (ge::OpTypeUtils::IsDataNode(op_type)) {
      size_t index = 0;
      AttrUtils::GetInt(op_desc, "index", index);
      if (model.data_ops_[index] != op_desc) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

TEST_F(UtestSingleOpModel, test_init_model) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  ASSERT_NE(model.InitModel(), SUCCESS);

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetOutputOffset(std::vector<int64_t>{4});
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = compute_graph->AddNode(op_desc);

  GeTensorDesc tensor(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32);

  ge::OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", DATA_TYPE);
  data_op_desc->AddOutputDesc(tensor);
  AttrUtils::SetInt(tensor, ATTR_NAME_PLACEMENT, 1);
  data_op_desc->AddInputDesc(tensor);
  TensorUtils::SetSize(*data_op_desc->MutableOutputDesc(0), 16);
  data_op_desc->SetOutputOffset(std::vector<int64_t>{4});
  ge::NodePtr data_node = compute_graph->AddNode(data_op_desc);

  ge::OpDescPtr const_op_desc = std::make_shared<OpDesc>("const", CONSTANT);
  ge::NodePtr const_node = compute_graph->AddNode(const_op_desc);
  const_op_desc->AddOutputDesc(tensor);
  const_op_desc->AddInputDesc(tensor);
  const_op_desc->SetOutputOffset(std::vector<int64_t>{4});
  TensorUtils::SetSize(*const_op_desc->MutableOutputDesc(0), 16);

  ge::OpDescPtr netoutput_op_desc = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
  netoutput_op_desc->AddOutputDesc(tensor);
  netoutput_op_desc->AddInputDesc(tensor);
  netoutput_op_desc->SetOutputOffset(std::vector<int64_t>{4});
  TensorUtils::SetSize(*netoutput_op_desc->MutableOutputDesc(0), 16);
  ge::NodePtr netoutput_node = compute_graph->AddNode(netoutput_op_desc);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  std::shared_ptr<domi::ModelTaskDef> tasks = std::make_shared<domi::ModelTaskDef>();
  domi::TaskDef *task = tasks->add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto context = task->mutable_kernel()->mutable_context();
  ge_model->task_ = tasks;

  ModelHelper model_helper;
  model_helper.model_ = ge_model;
  model.model_helper_ = model_helper;
  model.root_ge_model_ = ge_model;
  model.root_graph_ = compute_graph;
  ASSERT_NE(model.InitModel(), SUCCESS);
  ASSERT_EQ(model.LoadRootGraph(), SUCCESS);
  ASSERT_EQ(check_input_dataops_index(model), SUCCESS);
  ASSERT_EQ(model.ParseInputsAndOutputs(), SUCCESS);

  StreamResource *stream_resource = new StreamResource(1);
  std::mutex *stream_mutex = nullptr;
  rtStream_t stream = nullptr;
  SingleOpImpl single_op(stream_resource, stream_mutex, stream);
  model.input_offset_list_ = std::vector<ptrdiff_t>({0, 0});
  model.output_offset_list_ = std::vector<ptrdiff_t>({0});

  model.input_sizes_ = std::vector<size_t>({16, 16});
  model.output_sizes_ = std::vector<size_t>({16});

  ASSERT_EQ(model.SetInputsAndOutputs(single_op), SUCCESS);

  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  ASSERT_NE(model.BuildTaskList(*stream_resource, single_op), SUCCESS);

  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU));
  ASSERT_NE(model.BuildTaskList(*stream_resource, single_op), SUCCESS);

  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::CUST_AI_CPU));
  ASSERT_NE(model.BuildTaskList(*stream_resource, single_op), SUCCESS);

  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::INVALID));
  ASSERT_NE(model.BuildTaskList(*stream_resource, single_op), SUCCESS);

  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX));
  ASSERT_EQ(model.BuildTaskList(*stream_resource, single_op), SUCCESS);

  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  ASSERT_NE(model.BuildTaskList(*stream_resource, single_op), SUCCESS);

  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_EVENT_WAIT));
  ASSERT_EQ(model.BuildTaskList(*stream_resource, single_op), SUCCESS);
  delete stream_resource;
}

TEST_F(UtestSingleOpModel, test_InitModelMem) {
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ge::NodePtr node = compute_graph->AddNode(op_desc);
  ge_model->SetGraph(compute_graph);
  uint8_t *data = new uint8_t[5];
  size_t length = 5;
  Buffer buffer = BufferUtils::CreateCopyFrom(data, length);

  ge_model->SetWeight(buffer);

  std::string model_data_str = "model";
  SingleOpModel model(model_data_str, model_data_str.c_str(), model_data_str.size());

  ModelHelper model_helper;
  StreamResource resource(1);
  model_helper.model_ = ge_model;
  model.model_helper_ = model_helper;
  model.has_weight_ = true;
  model.root_ge_model_ = ge_model;
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 5);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 5);
  AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 100U);

  AttrUtils::SetInt(compute_graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(compute_graph, "globalworkspace_size", 0);
  ASSERT_EQ(model.InitModelMem(resource), SUCCESS);

  AttrUtils::SetInt(compute_graph, "globalworkspace_size", 1);
  EXPECT_EQ(model.InitModelMem(resource), SUCCESS);
  ASSERT_EQ(model.model_params_.runtime_param.mem_size, 10U);
  ASSERT_EQ(model.model_params_.runtime_param.zero_copy_size, 5);
  ASSERT_EQ(model.model_params_.runtime_param.weight_size, 5U);
  ASSERT_EQ(model.model_params_.runtime_param.host_svm_size, 100U);
  delete[] data;
}

TEST_F(UtestSingleOpModel, test_BuildOp) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetOutputOffset(std::vector<int64_t>{4});
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = compute_graph->AddNode(op_desc);

  std::vector<std::string> depend_names = {"__input0"};
  (void)AttrUtils::SetListStr(op_desc, "_op_infer_depends", depend_names);
  (void)AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, true);

  GeTensorDesc tensor_desc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32);

  AttrUtils::SetInt(tensor_desc, ATTR_NAME_PLACEMENT, 1);

  ge::OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", DATA_TYPE);
  data_op_desc->AddOutputDesc(tensor_desc);
  AttrUtils::SetInt(tensor_desc, ATTR_NAME_PLACEMENT, 1);
  data_op_desc->AddInputDesc(tensor_desc);
  TensorUtils::SetSize(*data_op_desc->MutableOutputDesc(0), 16);
  data_op_desc->SetOutputOffset(std::vector<int64_t>{4});
  ge::NodePtr data_node = compute_graph->AddNode(data_op_desc);


  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  ge::OpDescPtr const_op_desc = std::make_shared<OpDesc>("const", CONSTANT);
  ge::NodePtr const_node = compute_graph->AddNode(const_op_desc);
  const_op_desc->AddOutputDesc(tensor_desc);
  const_op_desc->AddInputDesc(tensor_desc);
  const_op_desc->SetOutputOffset(std::vector<int64_t>{4});
  TensorUtils::SetSize(*const_op_desc->MutableOutputDesc(0), 16);
  GeTensor const_tensor(tensor_desc);
  std::vector<int64_t> tensor_data(1, 16);
  const_tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data.data()), tensor_data.size() * sizeof(int64_t));
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", const_tensor);

  ge::OpDescPtr netoutput_op_desc = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
  netoutput_op_desc->AddOutputDesc(tensor_desc);
  netoutput_op_desc->AddInputDesc(tensor_desc);
  netoutput_op_desc->SetOutputOffset(std::vector<int64_t>{4});
  TensorUtils::SetSize(*netoutput_op_desc->MutableOutputDesc(0), 16);
  ge::NodePtr netoutput_node = compute_graph->AddNode(netoutput_op_desc);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  std::shared_ptr<domi::ModelTaskDef> tasks = std::make_shared<domi::ModelTaskDef>();
  domi::TaskDef *task = tasks->add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  ge_model->task_ = tasks;

  ModelHelper model_helper;
  model_helper.model_ = ge_model;

  GeRootModelPtr root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  model_helper.root_model_ = root_model;
  model.model_helper_ = model_helper;
  ASSERT_NE(model.InitModel(), SUCCESS);

  StreamResource *stream_resource = new StreamResource(1);
  std::mutex *stream_mutex = nullptr;
  rtStream_t stream = nullptr;
  SingleOpImpl single_op(stream_resource, stream_mutex, stream);
  model.root_ge_model_ = ge_model;
  ASSERT_NE(model.BuildOp(*stream_resource, single_op), SUCCESS);
  delete stream_resource;
}

TEST_F(UtestSingleOpModel, test_build_soft_sync_op) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{2, 3, 4};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Mul", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = compute_graph->AddNode(op_desc);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelBinRegistry::GetInstance().AddKernel("model/_tvmbin",
      std::unique_ptr<KernelHolder>(new KernelHolder((const char_t*)stub_func, kernel_bin)));
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  std::shared_ptr<domi::ModelTaskDef> tasks = std::make_shared<domi::ModelTaskDef>();
  domi::TaskDef *atomic_task = tasks->add_task();
  atomic_task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto atomic_kernel = atomic_task->mutable_kernel();
  std::vector<uint8_t> args_info(24, 0);
  atomic_kernel->set_args(args_info.data(), args_info.size());
  atomic_kernel->set_args_size(args_info.size());
  auto atomic_context = atomic_kernel->mutable_context();
  atomic_context->set_op_index(0);
  atomic_context->set_kernel_type(2);    // ccKernelType::TE
  domi::TaskDef *tbe_task = tasks->add_task();
  tbe_task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  auto tbe_context = tbe_task->mutable_kernel_with_handle()->mutable_context();
  tbe_context->set_op_index(0);
  tbe_context->set_kernel_type(2);    // ccKernelType::TE
  ge_model->task_ = tasks;
  model.op_list_[0] = node;
  model.root_graph_ = compute_graph;

  ModelHelper model_helper;
  model_helper.model_ = ge_model;

  GeRootModelPtr root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  model_helper.root_model_ = root_model;
  model.model_helper_ = model_helper;

  StreamResource *stream_resource = new StreamResource(1);
  std::mutex *stream_mutex = nullptr;
  rtStream_t stream = nullptr;
  SingleOpImpl single_op(stream_resource, stream_mutex, stream);
  model.root_ge_model_ = ge_model;
  EXPECT_EQ(model.BuildOp(*stream_resource, single_op), SUCCESS);
  delete stream_resource;
}

TEST_F(UtestSingleOpModel, test_build_soft_sync_mix_op) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<int32_t> output_indices{2, 3, 4};
  AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Mul", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = compute_graph->AddNode(op_desc);
  std::shared_ptr<ge::OpKernelBin> kernel_bin = std::make_shared<ge::OpKernelBin>("bin_name", std::vector<char>());
  void *stub_func = ValueToPtr(1234U);
  KernelBinRegistry::GetInstance().AddKernel("model/_tvmbin",
                                             std::unique_ptr<KernelHolder>(new KernelHolder((const char_t*)stub_func, kernel_bin)));
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  std::shared_ptr<domi::ModelTaskDef> tasks = std::make_shared<domi::ModelTaskDef>();
  domi::TaskDef *atomic_task = tasks->add_task();
  atomic_task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto atomic_kernel = atomic_task->mutable_kernel();
  std::vector<uint8_t> args_info(24, 0);
  atomic_kernel->set_args(args_info.data(), args_info.size());
  atomic_kernel->set_args_size(args_info.size());
  auto atomic_context = atomic_kernel->mutable_context();
  atomic_context->set_op_index(0);
  atomic_context->set_kernel_type(2);    // ccKernelType::TE
  domi::TaskDef *tbe_task = tasks->add_task();
  tbe_task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  auto tbe_context = tbe_task->mutable_kernel_with_handle()->mutable_context();
  tbe_context->set_op_index(0);
  tbe_context->set_kernel_type(2);    // ccKernelType::TE
  tbe_context->set_args_format("{ffts_addr}");
  std::vector<uint8_t> tbe_args_info(24, 0);
  tbe_task->mutable_kernel_with_handle()->set_args(tbe_args_info.data(), tbe_args_info.size());
  tbe_task->mutable_kernel_with_handle()->set_args_size(tbe_args_info.size());
  ge_model->task_ = tasks;
  model.op_list_[0] = node;
  model.root_graph_ = compute_graph;

  ModelHelper model_helper;
  model_helper.model_ = ge_model;

  GeRootModelPtr root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  model_helper.root_model_ = root_model;
  model.model_helper_ = model_helper;

  StreamResource *stream_resource = new StreamResource(1);
  std::mutex *stream_mutex = nullptr;
  rtStream_t stream = nullptr;
  SingleOpImpl single_op(stream_resource, stream_mutex, stream);
  model.root_ge_model_ = ge_model;
  EXPECT_EQ(model.BuildOp(*stream_resource, single_op), SUCCESS);
  delete stream_resource;
}

void ParseOpModelParamsMock(ModelHelper &model_helper, SingleOpModelParam &param) {}

TEST_F(UtestSingleOpModel, test_parse_input_node) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  auto op_desc = make_shared<OpDesc>("Data", "Data");

  ASSERT_NE(model.ParseInputNode(op_desc), SUCCESS);

  vector<int64_t> shape{1, 2, 3, 4};
  vector<int64_t> offsets{16};
  GeShape ge_shape(shape);
  GeTensorDesc desc(ge_shape);
  op_desc->AddOutputDesc(desc);
  op_desc->SetOutputOffset(offsets);
  ASSERT_EQ(model.ParseInputNode(op_desc), SUCCESS);

  op_desc->AddOutputDesc(desc);
  offsets.push_back(32);
  op_desc->SetOutputOffset(offsets);
  ASSERT_NE(model.ParseInputNode(op_desc), SUCCESS);
}

TEST_F(UtestSingleOpModel, test_parse_output_node) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  auto op_desc = make_shared<OpDesc>("NetOutput", "NetOutput");

  vector<int64_t> shape{1, 2, 3, 4};
  vector<int64_t> offsets{16};

  GeShape ge_shape(shape);
  GeTensorDesc desc(ge_shape);
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset(offsets);
  op_desc->AddOutputDesc(desc);
  op_desc->SetOutputOffset(offsets);

  ASSERT_NO_THROW(model.ParseOutputNode(op_desc));
  ASSERT_NO_THROW(model.ParseOutputNode(op_desc));
}

TEST_F(UtestSingleOpModel, test_build_dynamic_op) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  model.netoutput_op_ = make_shared<OpDesc>("NetOutput", "NetOutput");
  model.model_helper_.model_ = ge::MakeShared<ge::GeModel>();

  // make graph
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto compute_graph = builder.GetGraph();

  model.model_helper_.model_->SetGraph(compute_graph);
  model.op_list_[0] = transdata;

  auto op_desc = transdata->GetOpDesc();
  const vector<string> depend_names = { "Data" };
  op_desc->SetOpInferDepends(depend_names);
  (void)AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, true);

  // set task_def
  auto model_task_def = make_shared<domi::ModelTaskDef>();
  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def->mutable_kernel();
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_kernel_type(2);    // ccKernelType::TE
  model.model_helper_.model_->SetModelTaskDef(model_task_def);

  std::mutex stream_mu_;
  DynamicSingleOpImpl dynamic_single_op(nullptr, 0, &stream_mu_, nullptr);

  GeRootModelPtr root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  dynamic_single_op.hybrid_model_ = std::unique_ptr<hybrid::HybridModel>(new hybrid::HybridModel(root_model));

  StreamResource res((uintptr_t)1);
  model.root_ge_model_ = model.model_helper_.model_;
  model.root_graph_ = compute_graph;
  model.BuildDynamicOp(res, dynamic_single_op);

  op_desc->impl_->input_name_idx_["Data"] = 0;
  model.BuildDynamicOp(res, dynamic_single_op);

  auto tensor = std::make_shared<GeTensor>();
  auto data_desc = data->GetOpDesc();
  auto tensor_desc = data_desc->MutableInputDesc(0);
  AttrUtils::SetTensor(tensor_desc, "_value", tensor);
  model.BuildDynamicOp(res, dynamic_single_op);
}

TEST_F(UtestSingleOpModel, test_host_mem) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());

  // make graph
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  model.op_with_hostmem_[0] = data;

  std::mutex stream_mu_;
  DynamicSingleOpImpl single_op(nullptr, 0, &stream_mu_, nullptr);
  model.SetHostMemTensorAndNode(single_op);
  ASSERT_EQ(single_op.hostmem_node_id_map_[0U], data->GetOpDesc()->GetId());
}

TEST_F(UtestSingleOpModel, BuildTaskList) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("memcpy", MEMCPYASYNC);
    GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(0);
    memcpy_async->set_dst(0);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(0);
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  MemcpyAsyncTask mem_task;
  for (auto &task: single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, build_dynamic_task01) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX));

  domi::TaskDef *task_def2 = model_task_def->add_task();
  task_def2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def2->mutable_kernel();
  std::vector<uint8_t> args_info(150U, 0U);
  kernel_def->set_args(args_info.data(), args_info.size());
  kernel_def->set_args_size(args_info.size());
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_kernel_type(6);    // ccKernelType::AI_CPU

  domi::TaskDef *task_def3 = model_task_def->add_task();
  task_def3->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  auto kernel_def3 = task_def3->mutable_kernel_with_handle();
  auto context3 = kernel_def3->mutable_context();
  context3->set_kernel_type(2);    // ccKernelType::TE

  domi::TaskDef *task_def4 = model_task_def->add_task();
  task_def4->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def4 = task_def4->mutable_kernel();
  std::vector<uint8_t> args_info2(150U, 0U);
  kernel_def4->set_args(args_info2.data(), args_info2.size());
  kernel_def4->set_args_size(args_info2.size());
  domi::KernelContext *context4 = kernel_def4->mutable_context();
  context4->set_kernel_type(2);    // ccKernelType::TE

  domi::TaskDef *task_def5 = model_task_def->add_task();
  task_def5->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def5 = task_def5->mutable_kernel();
  domi::KernelContext *context5 = kernel_def5->mutable_context();
  context5->set_op_index(0);
  context5->set_kernel_type(2);    // ccKernelType::TE

  string model_data_str = "dynamic_model";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl single_op(nullptr, 0, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  auto op_desc = std::make_shared<ge::OpDesc>("add", "Add");
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  NodePtr node = graph->AddNode(op_desc);
  model.op_list_[0] = node;
  StreamResource *res = new (std::nothrow) StreamResource(1);

  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.ParseTasks(), SUCCESS);
  model.node_tasks_[node] = {*task_def4 };
  op_desc->SetOpKernelLibName(kEngineNameAiCore);
  AttrUtils::SetInt(op_desc, "op_para_size", 64);
  AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, true);
  ASSERT_EQ(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);

  model.node_tasks_[node] = { *task_def3, *task_def4 };
  ASSERT_NE(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);

  model.node_tasks_[node] = { *task_def };
  op_desc->SetOpKernelLibName(kEngineNameAiCpuTf);
  ASSERT_EQ(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);

  model.node_tasks_[node] = { *task_def2 };
  op_desc->SetOpKernelLibName(kEngineNameAiCpu);
  ASSERT_EQ(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);

  AtomicAddrCleanOpTask *atomic_task = nullptr;
  ASSERT_NE(model.BuildAtomicTask(*task_def5, &atomic_task, *res), SUCCESS);
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, build_dynamic_task02) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 6;
  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def->mutable_kernel();
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);

  char ext_mem[sizeof(ge::hybrid::AicpuExtInfo) + sizeof(int32_t)]{};
  ge::hybrid::AicpuExtInfo &aicpu_ext_info = *(ge::hybrid::AicpuExtInfo *)(ext_mem);
  aicpu_ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info.infoLen = sizeof(int32_t);
  int32_t type = ge::DEPEND_COMPUTE;
  memcpy_s(aicpu_ext_info.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));

  kernel_def->set_kernel_ext_info(ext_mem, sizeof(ge::hybrid::AicpuExtInfo) + sizeof(int32_t));
  kernel_def->set_kernel_ext_info_size(sizeof(ge::hybrid::AicpuExtInfo) + sizeof(int32_t));
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_kernel_type(6);    // ccKernelType::AI_CPU

  string model_data_str = "dynamic_model";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl single_op(nullptr, 0, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  auto op_desc = std::make_shared<ge::OpDesc>("add", "Add");
  AttrUtils::SetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, ge::DEPEND_COMPUTE);
  NodePtr node = graph->AddNode(op_desc);
  model.op_list_[0] = node;
  StreamResource *res = new (std::nothrow) StreamResource(1);

  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.ParseTasks(), SUCCESS);
  model.node_tasks_[node] = { *task_def, *task_def };
  op_desc->SetOpKernelLibName(kEngineNameAiCpu);
  ASSERT_NE(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);
  model.node_tasks_[node] = { *task_def};
  ASSERT_NE(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);
  context->set_kernel_type(7);
  model.node_tasks_[node] = {*task_def};
  ASSERT_NE(model.BuildTaskListForDynamicOp(*res, single_op), SUCCESS);
  delete res;
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOpModel, build_memcpoy_task) {
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 6;
  domi::KernelDef kernel_def;
  kernel_def.set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def.set_args_size(args.head.length);
  AiCpuCCTask aicpu_task;
  ASSERT_EQ(aicpu_task.SetMemCopyTask(kernel_def), INTERNAL_ERROR);
  kernel_def.set_args_size(0);
  ASSERT_EQ(aicpu_task.SetMemCopyTask(kernel_def), FAILED);
  const char* args2 = "123";
  kernel_def.set_args(reinterpret_cast<const char *>(&args2), 3);
  kernel_def.set_args_size(3);
  ASSERT_EQ(aicpu_task.SetMemCopyTask(kernel_def), FAILED);
}

TEST_F(UtestSingleOpModel, build_mixl2_task) {
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("MATMUL", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  domi::TaskDef task;
  task.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(op_desc->GetId());
  OpTask * mixl2_task = nullptr;
  StreamResource resource(1);
  ASSERT_NE(model.BuildMixL2KernelTask(task, mixl2_task, resource), SUCCESS);
  model.op_list_[op_desc->GetId()] = node;
  ASSERT_NE(model.BuildMixL2KernelTask(task, mixl2_task, resource), SUCCESS);
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "MIX_AIC");
  ASSERT_EQ(model.BuildMixL2KernelTask(task, mixl2_task, resource), SUCCESS);
  delete mixl2_task;
}


TEST_F(UtestSingleOpModel, load_atomic_workspace) {

  string model_data_str = "dynamic_model";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());

  OpDescPtr op_desc = std::make_shared<OpDesc>("workspace", DATA);
  GeAttrValue::NAMED_ATTRS workspaces;

  GeAttrValue::NamedAttrs workspaces_attrs;
  vector<int> dimTypeList;
  dimTypeList.push_back(1);
  dimTypeList.push_back(2);
  dimTypeList.push_back(3);
  AttrUtils::SetListInt(workspaces_attrs,op_desc->GetName(), dimTypeList);
  AttrUtils::SetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces_attrs);

  map<string, map<int64_t, int64_t>> workspace_info;
  map<int64_t, int64_t> workspace_info_pair;
  workspace_info_pair.insert(std::make_pair(1, 1));
  workspace_info.insert(std::make_pair("1", workspace_info_pair));
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO,workspace_info);
  EXPECT_EQ(ExecutorUtils::LoadAtomicWorkspace(op_desc), ge::SUCCESS);
}

TEST_F(UtestSingleOpModel, BuildClearFloatTask) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("clear_float", NPUCLEARFLOATSTATUS);
    GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_CLEAR_FLOAT_STATUS));
    domi::NpuClearFloatStatusDef *clear_float = task_def->mutable_npu_clear_float_status();
    clear_float->set_mode(0);
    clear_float->set_op_index(0);
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  MemcpyAsyncTask mem_task;
  ASSERT_TRUE(!single_op.tasks_.empty());
  for (auto &task: single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, BuildGetFloatTask) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("get_float", NPUGETFLOATSTATUS);
    GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    vector<bool> is_input_const = {true};
    op_desc->SetIsInputConst(is_input_const);
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_GET_FLOAT_STATUS));
    domi::NpuGetFloatStatusDef *get_float = task_def->mutable_npu_get_float_status();
    get_float->set_mode(0);
    get_float->set_output_addr(0);
    get_float->set_output_size(8);
    get_float->set_op_index(0);
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  MemcpyAsyncTask mem_task;
  ASSERT_TRUE(!single_op.tasks_.empty());
  for (auto &task: single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, BuildClearFloatDugStatusTask) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("clear_float", NPUCLEARFLOATDEBUGSTATUS);
    GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_CLEAR_DEBUG_FLOAT_STATUS));
    domi::NpuClearFloatDebugStatusDef *clear_float = task_def->mutable_npu_clear_float_debug_status();
    clear_float->set_mode(0);
    clear_float->set_op_index(0);
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  MemcpyAsyncTask mem_task;
  ASSERT_TRUE(!single_op.tasks_.empty());
  for (auto &task : single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, BuildGetFloatDebugStatusTask) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("get_float", NPUGETFLOATDEBUGSTATUS);
    GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({0});
    vector<bool> is_input_const = {true};
    op_desc->SetIsInputConst(is_input_const);
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NPU_GET_DEBUG_FLOAT_STATUS));
    domi::NpuGetFloatDebugStatusDef *get_float = task_def->mutable_npu_get_float_debug_status();
    get_float->set_mode(0);
    get_float->set_output_addr(0);
    get_float->set_output_size(8);
    get_float->set_op_index(0);
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  MemcpyAsyncTask mem_task;
  ASSERT_TRUE(!single_op.tasks_.empty());
  for (auto &task : single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, BuildDsaTask_no_state) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("dsa", "DSA");
    GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0,0,0,0,0});
    op_desc->SetOutputOffset({0});
    op_desc->SetWorkspace({0});
    op_desc->SetWorkspaceBytes({0});
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA));
    domi::DSATaskDef *dsa_task = task_def->mutable_dsa_task();
    dsa_task->set_op_index(0);
    dsa_task->set_start(1);
    dsa_task->set_sqe_type(1);
    dsa_task->set_distribution_type(1);
    dsa_task->set_data_type(1);
    dsa_task->set_alg_type(1);
    dsa_task->set_input_vld(1);
    dsa_task->set_input_value_addr_flag(1);
    dsa_task->set_input1_value_or_ptr(0);
    dsa_task->set_seed_value_or_ptr(0);
    dsa_task->set_random_count_value_or_ptr(0);
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  ASSERT_TRUE(!single_op.tasks_.empty());
  for (auto &task: single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, BuildDsaTask_has_state) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("single_op");
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  NodePtr node = nullptr;
  {
    auto op_desc = std::make_shared<ge::OpDesc>("dsa", "DSA");
    GeTensorDesc tensor(GeShape({8}), FORMAT_NCHW, DT_FLOAT);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({0,0,0,0});
    op_desc->SetOutputOffset({0});
    op_desc->SetWorkspace({0,0});
    op_desc->SetWorkspaceBytes({0,0});
    node = graph->AddNode(op_desc);

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA));
    domi::DSATaskDef *dsa_task = task_def->mutable_dsa_task();
    dsa_task->set_op_index(0);
    dsa_task->set_start(1);
    dsa_task->set_sqe_type(1);
    dsa_task->set_distribution_type(1);
    dsa_task->set_data_type(1);
    dsa_task->set_alg_type(1);
    dsa_task->set_input_vld(1);
    dsa_task->set_input_value_addr_flag(1);
    dsa_task->set_input1_value_or_ptr(1);
    dsa_task->set_seed_value_or_ptr(1);
    dsa_task->set_random_count_value_or_ptr(1);
    domi::DSATaskArgsDef *dsa_task_args = dsa_task->mutable_args();
    dsa_task_args->set_seed_value_or_addr("5");
    dsa_task_args->set_random_count_value_or_addr("6");
    dsa_task_args->set_input1_value_or_addr("1");
    dsa_task_args->set_input2_value_or_addr("2");
  }

  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);
  model.model_helper_.model_ = ge_model;
  model.op_list_.emplace(0, node);
  model.root_ge_model_ = ge_model;
  ASSERT_EQ(model.BuildTaskList(*res, single_op), SUCCESS);
  ASSERT_TRUE(!single_op.tasks_.empty());
  for (auto &task: single_op.tasks_) {
    ASSERT_EQ(task->LaunchKernel(single_op.stream_), SUCCESS);
  }
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOpModel, parse_op_model_params) {
  GeRootModelPtr root_model = std::make_shared<GeRootModel>();
  std::vector<OpSoBinPtr> kernels;
  std::string so_name("libopmaster.so");
  std::string vendor_name("MDC/opp/");
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new (std::nothrow) char[so_name.length()]);
  (void) memcpy_s(so_bin.get(), so_name.length(), so_name.data(), so_name.length());
  OpSoBinPtr op_so_bin = std::make_shared<OpSoBin>(so_name, vendor_name, std::move(so_bin), so_name.length());
  kernels.emplace_back(op_so_bin);
  root_model->op_so_store_.kernels_ = std::move(kernels);

  ModelHelper model_helper;
  model_helper.root_model_ = root_model;
  string model_data_str = "123456789";
  SingleOpModel model("model", model_data_str.c_str(), model_data_str.size());
  model.model_helper_ = model_helper;
  EXPECT_EQ(model.ParseOpModelParams(), ge::SUCCESS);
}
