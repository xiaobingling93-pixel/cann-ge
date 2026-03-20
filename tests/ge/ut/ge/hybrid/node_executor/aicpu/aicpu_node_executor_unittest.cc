/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/runtime_inference_context.h"
#include "aicpu_task_struct.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;

namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));
}  // namespace

namespace ge {
using namespace hybrid;

class UtestAicpuNodeExecutor : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }
  void TearDown() {
    RTS_STUB_TEARDOWN();
    AclRuntimeStub::SetErrorResultApiName("");
  }
};

static NodePtr CreateNode(ComputeGraphPtr graph, const string &name, const string &type,
                          int in_num, int out_num, const bool host_mem_input = false) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  if (host_mem_input) {
    AttrUtils::SetInt(tensor, ATTR_NAME_PLACEMENT, 2);
  }
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);
  return graph->AddNode(op_desc);
}

TEST_F(UtestAicpuNodeExecutor, aicpu_tf_node_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "frameworkop", FRAMEWORK_OP_TYPE, 2, 2);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_COMPUTE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 4;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 4; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1 = 512;
  TensorValue out_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  // task
  domi::TaskDef task_def;
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_kernel_ext_info_size(12);

  char ext_mem[sizeof(AicpuExtInfo) + sizeof(int32_t)]{};
  AicpuExtInfo &aicpu_ext_info = *(AicpuExtInfo *)(ext_mem);
  aicpu_ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info.infoLen = sizeof(int32_t);
  int32_t type = node_item->shape_inference_type;
  memcpy_s(aicpu_ext_info.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));
  std::string ext_info(ext_mem, sizeof(AicpuExtInfo) + sizeof(int32_t));

  std::string *mutable_ext_info = kernel_ex_def->mutable_kernel_ext_info();
  (*mutable_ext_info) = ext_info;

  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def, task_def});

  AicpuTfNodeTask aicpu_tf_node_task(node_item, task_def);

  auto *const mem_allocator = NpuMemoryAllocator::GetAllocator(0, nullptr);
  ASSERT_NE(mem_allocator, nullptr);
  hybrid_model.global_step_ = TensorBuffer::Create(mem_allocator, sizeof(int64_t));
  ASSERT_NE(hybrid_model.global_step_, nullptr);

  ASSERT_EQ(aicpu_tf_node_task.Init(hybrid_model), SUCCESS);
  ASSERT_EQ(aicpu_tf_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 4;

  domi::TaskDef task_def2;
  task_def2.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDef *kernel_def = task_def2.mutable_kernel();
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);

  char ext_mem2[sizeof(AicpuExtInfo) + sizeof(int32_t)]{};
  AicpuExtInfo &aicpu_ext_info2 = *(AicpuExtInfo *)(ext_mem2);
  aicpu_ext_info2.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info2.infoLen = sizeof(int32_t);
  memcpy_s(aicpu_ext_info2.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));
  kernel_def->set_kernel_ext_info(ext_mem2, sizeof(AicpuExtInfo) + sizeof(int32_t));
  kernel_def->set_kernel_ext_info_size(sizeof(AicpuExtInfo) + sizeof(int32_t));
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def2, task_def2});

  AicpuNodeTask aicpu_node_task(node_item, task_def2);
  ASSERT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
  ASSERT_EQ(aicpu_node_task.UpdateIoAddr(*node_state->GetTaskContext()), SUCCESS);
  ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);
  node_item->is_dynamic = false;
  ASSERT_EQ(aicpu_node_task.UpdateIoAddr(*node_state->GetTaskContext()), SUCCESS);
  //kernel_ex_def->set_allocated_kernel_ext_info(nullptr);
}

// tf_aicpu_node and aicpu_node
TEST_F(UtestAicpuNodeExecutor, aicpu_node_task_with_host_mem_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(graph, "frameworkop", FRAMEWORK_OP_TYPE, 2, 2, true);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_COMPUTE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 2; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1 = 512;
  TensorValue out_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  // task
  domi::TaskDef task_def;
  domi::KernelExDef *kernel_ex_def = task_def.mutable_kernel_ex();
  kernel_ex_def->set_kernel_ext_info_size(12);

  char ext_mem[sizeof(AicpuExtInfo) + sizeof(int32_t)]{};
  AicpuExtInfo &aicpu_ext_info = *(AicpuExtInfo *)(ext_mem);
  aicpu_ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info.infoLen = sizeof(int32_t);
  int32_t type = node_item->shape_inference_type;
  memcpy_s(aicpu_ext_info.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));
  std::string ext_info(ext_mem, sizeof(AicpuExtInfo) + sizeof(int32_t));

  std::string *mutable_ext_info = kernel_ex_def->mutable_kernel_ext_info();
  (*mutable_ext_info) = ext_info;

  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def, task_def});

  AicpuTfNodeTask aicpu_tf_node_task(node_item, task_def);
  aicpu_tf_node_task.need_host_mem_opt_ = true;
  EXPECT_EQ(aicpu_tf_node_task.Init(hybrid_model), SUCCESS);
  EXPECT_NE(aicpu_tf_node_task.host_mem_input_data_offset_, 0);
  EXPECT_EQ(aicpu_tf_node_task.UpdateIoAddr(*node_state->GetTaskContext()), SUCCESS);
  EXPECT_EQ(aicpu_tf_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);

  // check host mem input data address
  uint64_t *host_mem_input_addr = PtrToPtr<void, uint64_t>(aicpu_tf_node_task.input_output_addr_->GetData());
  uint64_t host_mem_input_data_addr = PtrToValue(aicpu_tf_node_task.input_output_addr_->GetData()) +
                                 aicpu_tf_node_task.host_mem_input_data_offset_;
  EXPECT_EQ(*host_mem_input_addr, host_mem_input_data_addr);


  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 4;

  domi::TaskDef task_def2;
  task_def2.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDef *kernel_def = task_def2.mutable_kernel();
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);

  char ext_mem2[sizeof(AicpuExtInfo) + sizeof(int32_t)]{};
  AicpuExtInfo &aicpu_ext_info2 = *(AicpuExtInfo *)(ext_mem2);
  aicpu_ext_info2.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info2.infoLen = sizeof(int32_t);
  memcpy_s(aicpu_ext_info2.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));
  kernel_def->set_kernel_ext_info(ext_mem2, sizeof(AicpuExtInfo) + sizeof(int32_t));
  kernel_def->set_kernel_ext_info_size(sizeof(AicpuExtInfo) + sizeof(int32_t));
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def2, task_def2});

  AicpuNodeTask aicpu_node_task(node_item, task_def2);
  aicpu_node_task.need_host_mem_opt_ = true;
  EXPECT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
  EXPECT_NE(aicpu_node_task.host_mem_input_data_offset_, 0);
  EXPECT_EQ(aicpu_node_task.UpdateIoAddr(*node_state->GetTaskContext()), SUCCESS);

  // check args
  EXPECT_EQ(aicpu_node_task.args_ex_.args, aicpu_node_task.args_.get());
  ASSERT_EQ(aicpu_node_task.args_ex_.hostInputInfoNum, 2);
  EXPECT_NE(aicpu_node_task.args_ex_.hostInputInfoPtr, nullptr);
  if (aicpu_node_task.args_ex_.hostInputInfoPtr != nullptr) {
    EXPECT_EQ(aicpu_node_task.args_ex_.hostInputInfoPtr[0].addrOffset, sizeof(aicpu::AicpuParamHead));
    EXPECT_EQ(aicpu_node_task.args_ex_.hostInputInfoPtr[1].addrOffset, sizeof(aicpu::AicpuParamHead) + sizeof(void *));

    // check host mem input data address
    uint64_t *addr1 = PtrToPtr<void, uint64_t>(ValueToPtr(PtrToValue(aicpu_node_task.args_ex_.args) +
                                                         aicpu_node_task.args_ex_.hostInputInfoPtr[0].addrOffset));
    uint64_t *addr2 = PtrToPtr<void, uint64_t>(ValueToPtr(PtrToValue(aicpu_node_task.args_ex_.args) +
                                               aicpu_node_task.args_ex_.hostInputInfoPtr[1].addrOffset));
    EXPECT_EQ(*addr1, (*addr2) - sizeof(void *));
  }

  EXPECT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);
  node_item->is_dynamic = false;
  EXPECT_EQ(aicpu_node_task.UpdateIoAddr(*node_state->GetTaskContext()), SUCCESS);
  //kernel_ex_def->set_allocated_kernel_ext_info(nullptr);
}

TEST_F(UtestAicpuNodeExecutor, aicpu_memcopy_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  NodePtr node = CreateNode(*graph, "frameworkop", FRAMEWORK_OP_TYPE, 4, 2);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 6;

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);
  node_item->num_outputs = 0;
  AicpuNodeTask aicpu_node_task(node_item, task_def);
  ASSERT_EQ(aicpu_node_task.SetMemCopyTask(task_def), SUCCESS);
  node_item->num_outputs = 1;
  AicpuNodeTask aicpu_node_task2(node_item, task_def);
  ASSERT_EQ(aicpu_node_task2.SetMemCopyTask(task_def), INTERNAL_ERROR);
  kernel_def->set_args_size(0);
  ASSERT_EQ(aicpu_node_task2.SetMemCopyTask(task_def), FAILED);
  const char* args2 = "123";
  kernel_def->set_args(reinterpret_cast<const char *>(&args2), 3);
  kernel_def->set_args_size(3);
  ASSERT_EQ(aicpu_node_task2.SetMemCopyTask(task_def), FAILED);
}

TEST_F(UtestAicpuNodeExecutor, aicpu_copy_data_to_hbm) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "frameworkop", FRAMEWORK_OP_TYPE, 2, 2);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_COMPUTE;
  node_item->num_outputs = 2;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 4;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  graph_context.model = &hybrid_model;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  ASSERT_EQ(graph_context.callback_manager->Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 4; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1 = 512;
  TensorValue out_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  // task
  domi::TaskDef task_def;
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 4;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);

  char ext_mem[sizeof(AicpuExtInfo) + sizeof(int32_t)]{};
  AicpuExtInfo &aicpu_ext_info = *(AicpuExtInfo *)(ext_mem);
  aicpu_ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info.infoLen = sizeof(int32_t);
  int32_t type = node_item->shape_inference_type;
  memcpy_s(aicpu_ext_info.infoMsg, sizeof(int32_t), &type, sizeof(int32_t));
  kernel_def->set_kernel_ext_info(ext_mem, sizeof(AicpuExtInfo) + sizeof(int32_t));
  kernel_def->set_kernel_ext_info_size(sizeof(AicpuExtInfo) + sizeof(int32_t));
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def, task_def});

  AicpuNodeTask aicpu_node_task(node_item, task_def);
  std::vector<std::unique_ptr<TensorBuffer>> out_shape_hbm;
  ASSERT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
  for (int i = 0; i < node_item->num_outputs; i++) {
    auto &summary = aicpu_node_task.output_summary_host_[i];
    summary.shape_data_ptr = 0;
    summary.shape_data_size = 1;
    summary.raw_data_ptr = 0;
    summary.raw_data_size = 1;
  }
  for (int i = 0; i < node_item->num_outputs; i++) {
    std::unique_ptr<TensorBuffer> shape_buffer;
    aicpu_node_task.AllocTensorBuffer(1, shape_buffer);
    out_shape_hbm.emplace_back(std::move(shape_buffer));
  }
  ASSERT_EQ(aicpu_node_task.CopyDataToHbm(*node_state->GetTaskContext(), out_shape_hbm), SUCCESS);
  auto &task_context = *node_state->GetTaskContext();
  task_context.outputs_start_[0].ref_buffer_ = nullptr;
  task_context.outputs_start_[0].buffer_ = nullptr;
  ASSERT_EQ(aicpu_node_task.AllocOutputBuffer(task_context, 0, 1), SUCCESS);

  task_context.outputs_start_[0].ref_buffer_ = &value_1;
  ASSERT_EQ(aicpu_node_task.AllocOutputBuffer(task_context, 0, 0), SUCCESS);
  ASSERT_EQ(aicpu_node_task.AllocOutputBuffer(task_context, 0, 100), GRAPH_PARAM_INVALID);
  ASSERT_EQ(graph_context.callback_manager->Destroy(), SUCCESS);
}

TEST_F(UtestAicpuNodeExecutor, aicpu_blocking_node_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "deque", FRAMEWORK_OP_TYPE, 1, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_execution_context;
  SubgraphContext subgraph_context(&graph_item, &graph_execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_execution_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_execution_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 512;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);

  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);
  domi::TaskDef task_def;

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 2;

  kernel_def.set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def.set_args_size(args.head.length);
  domi::KernelDef *kernel_def_tmp = task_def.mutable_kernel();
  *kernel_def_tmp = kernel_def;
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def});
  AicpuNodeTask aicpu_node_task(node_item, task_def);
  ASSERT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
  ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);

  node_item->shape_inference_type = DEPEND_COMPUTE;
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  kernel_ex_def.set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_ex_def.set_args_size(args.head.length);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def, task_def});

  AicpuTfNodeTask aicpu_tf_node_task(node_item, task_def);
  ASSERT_EQ(aicpu_tf_node_task.Init(hybrid_model), SUCCESS);
  ASSERT_EQ(aicpu_tf_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);
}

TEST_F(UtestAicpuNodeExecutor, aicpu_blocking_node_task_fail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "deque", FRAMEWORK_OP_TYPE, 1, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->num_outputs =1;
  node_item->num_inputs = 1;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_execution_context;
  graph_execution_context.model = &hybrid_model;
  SubgraphContext subgraph_context(&graph_item, &graph_execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_execution_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_execution_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 512;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);

  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo*>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo*>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);
  domi::TaskDef task_def;

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 2;

  kernel_def.set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def.set_args_size(args.head.length);
  domi::KernelDef *kernel_def_tmp = task_def.mutable_kernel();
  *kernel_def_tmp = kernel_def;
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def});

  {
    AicpuNodeTask aicpu_node_task(node_item, task_def);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_node_task.Init(hybrid_model), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_node_task.Init(hybrid_model), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
    ASSERT_EQ(aicpu_node_task.Init(hybrid_model), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), FAILED);

    ASSERT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtStreamWaitEventWithTimeout, rtError_t, 0x78000001);
    RTS_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), FAILED);
  }

  {
    AicpuNodeTask aicpu_node_task(node_item, task_def);
    ASSERT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
    AclRuntimeStub::SetErrorResultApiName("aclrtResetEvent");
    ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), FAILED);
    AclRuntimeStub::SetErrorResultApiName("");

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
    ASSERT_EQ(aicpu_node_task.Init(hybrid_model), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
    ASSERT_EQ(aicpu_node_task.LaunchTask(*node_state->GetTaskContext()), SUCCESS);
  }

  node_item->shape_inference_type = DEPEND_COMPUTE;
  domi::KernelExDef kernel_ex_def;
  kernel_ex_def.set_kernel_ext_info(buf, len);
  kernel_ex_def.set_kernel_ext_info_size(len);
  kernel_ex_def.set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_ex_def.set_args_size(args.head.length);
  domi::KernelExDef *kernel_ex_def_tmp = task_def.mutable_kernel_ex();
  *kernel_ex_def_tmp = kernel_ex_def;
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def, task_def});

  {
    auto aicpu_tf_node_task = std::make_shared<AicpuTfNodeTask>(node_item, task_def);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_tf_node_task->Init(hybrid_model), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_tf_node_task->Init(hybrid_model), FAILED);

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_SUPPORT + 1);
    ASSERT_EQ(aicpu_tf_node_task->Init(hybrid_model), FAILED);

    ASSERT_EQ(aicpu_tf_node_task->Init(hybrid_model), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtStreamWaitEventWithTimeout, rtError_t, 0x78000001);
    RTS_STUB_RETURN_VALUE(rtStreamWaitEvent, rtError_t, 0x78000001);
    ASSERT_EQ(aicpu_tf_node_task->LaunchTask(*node_state->GetTaskContext()), FAILED);
  }

  {
    auto aicpu_tf_node_task = std::make_shared<AicpuTfNodeTask>(node_item, task_def);

    ASSERT_EQ(aicpu_tf_node_task->Init(hybrid_model), SUCCESS);
    AclRuntimeStub::SetErrorResultApiName("aclrtResetEvent");
    ASSERT_EQ(aicpu_tf_node_task->LaunchTask(*node_state->GetTaskContext()), FAILED);
    AclRuntimeStub::SetErrorResultApiName("");

    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
    EXPECT_EQ(aicpu_tf_node_task->Init(hybrid_model), SUCCESS);
    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
    EXPECT_EQ(aicpu_tf_node_task->LaunchTask(*node_state->GetTaskContext()), SUCCESS);
    std::vector<std::unique_ptr<TensorBuffer>> out_shape_hbm;
    for (int i = 0; i < node_item->num_outputs; i++) {
      std::unique_ptr<TensorBuffer> shape_buffer;
      aicpu_tf_node_task->AllocTensorBuffer(1, shape_buffer);
      aicpu_tf_node_task->output_summary_[i] = std::move(shape_buffer);
      auto &summary = aicpu_tf_node_task->output_summary_host_[i];
      summary.shape_data_ptr = 0;
      summary.shape_data_size = 16;
      summary.raw_data_ptr = 0;
      summary.raw_data_size = 8;
    }
    for (int i = 0; i < node_item->num_outputs; i++) {
      std::unique_ptr<TensorBuffer> shape_buffer;
      aicpu_tf_node_task->AllocTensorBuffer(1, shape_buffer);
      out_shape_hbm.emplace_back(std::move(shape_buffer));
    }
    AicpuShapeAndType input;
    aicpu_tf_node_task->aicpu_ext_handle_.input_shape_and_type_.push_back(&input);
    aicpu_tf_node_task->aicpu_ext_handle_.output_shape_and_type_.push_back(&input);

    EXPECT_EQ(aicpu_tf_node_task->UpdateShapeAndDataByResultSummary(*node_state->GetTaskContext()), SUCCESS);
    EXPECT_EQ(aicpu_tf_node_task->UpdateShapeByHbmBuffer(*node_state->GetTaskContext(), out_shape_hbm), SUCCESS);

    EXPECT_EQ(aicpu_tf_node_task->CopyDataToHbm(*node_state->GetTaskContext(), out_shape_hbm), SUCCESS);

    EXPECT_EQ(aicpu_tf_node_task->UpdateIoAddr(*node_state->GetTaskContext()), SUCCESS);

    EXPECT_EQ(aicpu_tf_node_task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
    EXPECT_EQ(aicpu_tf_node_task->UpdateOutputShapeFromExtInfo(*node_state->GetTaskContext()), SUCCESS);
    EXPECT_EQ(aicpu_tf_node_task->ReadResultSummaryAndPrepareMemory(*node_state->GetTaskContext(), out_shape_hbm),
              SUCCESS);
    ASSERT_EQ(graph_execution_context.callback_manager->Init(), SUCCESS);
    EXPECT_EQ(aicpu_tf_node_task->TaskCallback(*node_state->GetTaskContext()), SUCCESS);
    std::function<void()> callback = []() {};
    EXPECT_EQ(aicpu_tf_node_task->ExecuteAsync(*node_state->GetTaskContext(), callback), SUCCESS);
    ASSERT_EQ(graph_execution_context.callback_manager->Destroy(), SUCCESS);
  }
}

TEST_F(UtestAicpuNodeExecutor, check_index) {
  AicpuExtInfoHandler handle("NpOp", 1, 1, DEPEND_IN_SHAPE);
  AicpuShapeAndType shape_and_type;
  handle.input_shape_and_type_.push_back(&shape_and_type);
  handle.output_shape_and_type_.push_back(&shape_and_type);
  GeShape shape;
  DataType dtype;
  EXPECT_EQ(handle.GetOutputShapeAndType(-1, shape, dtype), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(handle.Parse(""), ACL_ERROR_GE_PARAM_INVALID);

  char ext_mem[sizeof(AicpuExtInfo) + sizeof(AicpuShapeAndType)]{};
  AicpuExtInfo &aicpu_ext_info = *(AicpuExtInfo *)(ext_mem);
  aicpu_ext_info.infoLen = 1;
  EXPECT_EQ(handle.ParseExtAsyncWait(aicpu_ext_info), ACL_ERROR_GE_PARAM_INVALID);
  EXPECT_EQ(handle.ParseExtWorkSpaceInfo(aicpu_ext_info), ACL_ERROR_GE_PARAM_INVALID);
  EXPECT_EQ(handle.ParseExtInputShape(aicpu_ext_info), ACL_ERROR_GE_PARAM_INVALID);
  EXPECT_EQ(handle.ParseExtOutputShape(aicpu_ext_info), ACL_ERROR_GE_PARAM_INVALID);
  EXPECT_EQ(handle.ParseExtSessionInfo(aicpu_ext_info), ACL_ERROR_GE_PARAM_INVALID);

  aicpu_ext_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  aicpu_ext_info.infoLen = sizeof(AicpuShapeAndType);
  AicpuShapeAndType type;
  memcpy_s(aicpu_ext_info.infoMsg, sizeof(AicpuShapeAndType), &type, sizeof(AicpuShapeAndType));
  EXPECT_EQ(handle.ParseExtInputShape(aicpu_ext_info), SUCCESS);
  EXPECT_EQ(handle.ParseExtOutputShape(aicpu_ext_info), SUCCESS);
  EXPECT_EQ(handle.ParseExtBitMap(aicpu_ext_info), PARAM_INVALID);
  GeTensorDesc tensor_desc(GeShape({1,-1}), FORMAT_ND);
  handle.unknown_type_ = DEPEND_SHAPE_RANGE;
  tensor_desc.SetShapeRange({{1, 10}});
  EXPECT_EQ(handle.UpdateOutputShapeAndType(0, tensor_desc), SUCCESS);
  EXPECT_EQ(handle.UpdateOutputShapeAndType(10, tensor_desc), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(handle.UpdateInputShapeAndType(0, tensor_desc), SUCCESS);
  EXPECT_EQ(handle.UpdateInputShapeAndType(10, tensor_desc), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(handle.UpdateShapeAndType(GeShape({1,1,1,1,1,1,1,1,1,}), DT_FLOAT, &type),
            ACL_ERROR_GE_PARAM_INVALID);

  char ext_mem2[sizeof(AicpuExtInfo) + sizeof(AicpuSessionInfo)]{};
  AicpuExtInfo &aicpu_ext_sess_info = *(AicpuExtInfo *)(ext_mem2);
  aicpu_ext_sess_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
  aicpu_ext_sess_info.infoLen = sizeof(AicpuSessionInfo);
  AicpuSessionInfo session_info;
  memcpy_s(aicpu_ext_sess_info.infoMsg, sizeof(AicpuSessionInfo), &session_info, sizeof(AicpuSessionInfo));
  EXPECT_EQ(handle.ParseExtSessionInfo(aicpu_ext_sess_info), SUCCESS);

  char ext_mem3[sizeof(AicpuExtInfo) + sizeof(AsyncWaitInfo)]{};
  AicpuExtInfo &aicpu_ext_wait_info = *(AicpuExtInfo *)(ext_mem3);
  aicpu_ext_wait_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  aicpu_ext_wait_info.infoLen = sizeof(AsyncWaitInfo);
  AsyncWaitInfo wait_info;
  memcpy_s(aicpu_ext_wait_info.infoMsg, sizeof(AsyncWaitInfo), &wait_info, sizeof(AsyncWaitInfo));
  EXPECT_EQ(handle.UpdateEventId(10), FAILED);
  EXPECT_EQ(handle.ParseExtAsyncWait(aicpu_ext_wait_info), SUCCESS);
  EXPECT_EQ(handle.UpdateEventId(10), SUCCESS);
  EXPECT_EQ(handle.async_wait_->waitId, 10);

  char ext_mem4[sizeof(AicpuExtInfo) + sizeof(int64_t)]{};
  AicpuExtInfo &aicpu_ext_bitmap_info = *(AicpuExtInfo *)(ext_mem4);;
  uint64_t bit_map = 1;
  aicpu_ext_bitmap_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP;
  aicpu_ext_bitmap_info.infoLen = sizeof(uint64_t);
  memcpy_s(aicpu_ext_bitmap_info.infoMsg, sizeof(int64_t), &bit_map, sizeof(uint64_t));
  handle.ParseExtBitMap(aicpu_ext_bitmap_info);
  EXPECT_EQ(handle.UpdateExecuteMode(false), SUCCESS);
  EXPECT_EQ(handle.UpdateExecuteMode(true), SUCCESS);

  handle.UpdateSessionInfoId(10);


  EXPECT_EQ(handle.session_info_->sessionId, 10);
  handle.UpdateSessionInfo(1, 2, false);
  EXPECT_EQ(handle.session_info_->sessFlag, false);
  EXPECT_EQ(handle.session_info_->sessionId, 1);
  EXPECT_EQ(handle.session_info_->kernelId, 2);

  char ext_mem5[sizeof(AicpuExtInfo) + sizeof(WorkSpaceInfo)]{};
  AicpuExtInfo &aicpu_ext_space_info = *(AicpuExtInfo *)(ext_mem5);
  aicpu_ext_space_info.infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_WORKSPACE_INFO;
  aicpu_ext_space_info.infoLen = sizeof(WorkSpaceInfo);
  WorkSpaceInfo space_info;
  memcpy_s(aicpu_ext_space_info.infoMsg, sizeof(WorkSpaceInfo), &space_info, sizeof(WorkSpaceInfo));
  EXPECT_EQ(handle.ParseExtWorkSpaceInfo(aicpu_ext_space_info), SUCCESS);
  EXPECT_EQ(handle.UpdateWorkSpaceInfo(0U, 0U), SUCCESS);
}

TEST_F(UtestAicpuNodeExecutor, UpdateBlockDimInfo) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  NodePtr node = CreateNode(graph, "deque", FRAMEWORK_OP_TYPE, 1, 1);
  const auto &op_desc = node->GetOpDesc();
  const auto &desc = op_desc->MutableInputDesc(0);
  std::vector<int64_t> shape = {2, 1, 1, 1, 1, 1, 2, 1, 1};
  GeShape ge_shape(shape);
  desc->SetShape(ge_shape);
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  ge::AttrUtils::SetInt(op_desc, ATTR_NAME_BLOCKDIM_INDEX, -1);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  domi::TaskDef task_def;
  AicpuNodeTask aicpu_node_task(node_item, task_def);
  (void)aicpu_node_task.UpdateBlockDimInfo(-1);
  EXPECT_EQ(aicpu_node_task.block_num_, 1);
}

TEST_F(UtestAicpuNodeExecutor, test_executor_load_and_prepare_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "deque", FRAMEWORK_OP_TYPE, 1, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->num_outputs = 1;
  node_item->num_inputs = 1;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_execution_context;
  SubgraphContext subgraph_context(&graph_item, &graph_execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_execution_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_execution_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_0 = 512;
  TensorValue in_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetInput(*node_item, 0, in_tensor0);

  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  int len = sizeof(hybrid::AicpuExtInfo) + sizeof(hybrid::AsyncWaitInfo);
  vector<char> aicpu_ext_info(len, 0);
  char *buf = aicpu_ext_info.data();
  int offset = 0;
  hybrid::AicpuExtInfo *ext_info = reinterpret_cast<hybrid::AicpuExtInfo *>(buf + offset);
  ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  ext_info->infoLen = sizeof(hybrid::AsyncWaitInfo);
  offset += sizeof(hybrid::AicpuExtInfo);
  hybrid::AsyncWaitInfo *async_wait_info = reinterpret_cast<hybrid::AsyncWaitInfo *>(buf + offset);
  async_wait_info->waitType = 0;
  async_wait_info->waitId = 0;
  async_wait_info->timeOut = 0;
  async_wait_info->reserved = 0;

  domi::KernelDef kernel_def;
  kernel_def.set_kernel_ext_info(buf, len);
  kernel_def.set_kernel_ext_info_size(len);
  domi::TaskDef task_def;

  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  args.head.ioAddrNum = 2;

  kernel_def.set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def.set_args_size(args.head.length);
  domi::KernelDef *kernel_def_tmp = task_def.mutable_kernel();
  *kernel_def_tmp = kernel_def;
  hybrid_model.task_defs_[node] = std::vector<domi::TaskDef>({task_def});
  hybrid_model.node_items_[node] = std::move(new_node);
  std::shared_ptr<NodeTask> aicpu_node_task = std::make_shared<AicpuNodeTask>(node_item, task_def);
  AiCpuNodeExecutor executor;

  executor.PrepareTask(*aicpu_node_task, *node_state->GetTaskContext());
  executor.LoadTask(hybrid_model, node, aicpu_node_task);
}

TEST_F(UtestAicpuNodeExecutor, check_overflow_test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "deque", FRAMEWORK_OP_TYPE, 1, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_IS_BLOCKING_OP, true);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->num_outputs =1;
  node_item->num_inputs = 1;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_execution_context;
  graph_execution_context.model = &hybrid_model;
  SubgraphContext subgraph_context(&graph_item, &graph_execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_execution_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_execution_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  domi::TaskDef task_def;
  AicpuNodeTask aicpu_node_task(node_item, task_def);
  auto task_context = *node_state->GetTaskContext();
  task_context.execution_context_->dump_properties.is_train_op_debug_ = true;
  ASSERT_EQ(aicpu_node_task.CheckOverflow(task_context), SUCCESS);

  const char_t *const kAicpuEnvOverFlowPath = "SYNCSTREAM_OVERFLOW_RET";
  char_t aicpu_over_flow_path[MMPA_MAX_PATH] = "aicpu";
  mmSetEnv(kAicpuEnvOverFlowPath, &aicpu_over_flow_path[0U], MMPA_MAX_PATH);
  ASSERT_EQ(aicpu_node_task.CheckOverflow(task_context), SUCCESS);
  unsetenv(kAicpuEnvOverFlowPath);

  const char_t *const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char_t record_path[MMPA_MAX_PATH] = "mock_fail";
  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
  ASSERT_NE(aicpu_node_task.CheckOverflow(task_context), SUCCESS);
  unsetenv(kEnvRecordPath);

  AicpuTfNodeTask aicpu_tf_node_task(node_item, task_def);
  ASSERT_EQ(aicpu_tf_node_task.CheckOverflow(task_context), SUCCESS);

  mmSetEnv(kEnvRecordPath, &record_path[0U], MMPA_MAX_PATH);
  ASSERT_NE(aicpu_tf_node_task.CheckOverflow(task_context), SUCCESS);
  unsetenv(kEnvRecordPath);

  const char_t *const kEnvPath = "END_OF_SEQUENCE";
  char_t env_path[MMPA_MAX_PATH] = "end";
  mmSetEnv(kEnvPath, &env_path[0U], MMPA_MAX_PATH);
  ASSERT_NE(aicpu_tf_node_task.CheckOverflow(task_context), SUCCESS);
  unsetenv(kEnvPath);

  const char_t *const kEnvOverFlowPath = "ACL_ERROR_RT_OVER_FLOW";
  char_t over_flow_path[MMPA_MAX_PATH] = "over_flow";
  mmSetEnv(kEnvOverFlowPath, &over_flow_path[0U], MMPA_MAX_PATH);
  ASSERT_EQ(aicpu_tf_node_task.CheckOverflow(task_context), SUCCESS);
  unsetenv(kEnvOverFlowPath);
}
}  // namespace ge

