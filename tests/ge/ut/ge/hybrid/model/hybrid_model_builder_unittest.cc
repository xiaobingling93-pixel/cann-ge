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
#include <memory>
#include <fstream>
#include "macro_utils/dt_public_scope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"
#include "graph/manager/host_mem_manager.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "common/omg_util/omg_util.h"
#include "hybrid/executor/worker/shape_inference_engine.h"
#include "graph/runtime_inference_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "hybrid/model/node_item.h"
#include "graph/ge_context.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestHybridModelBuilder : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    std::map<std::string, string> options;
    options["ge.exec.placement"] = "DEVICE";
    GetThreadLocalContext().SetGraphOption(options);
  }
};

static NodePtr CreateConstantNode(const ComputeGraphPtr &graph, const string &name, size_t size) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, CONSTANTOP);
  op_desc->AddOutputDesc(GeTensorDesc());
  GeTensorPtr value = std::make_shared<GeTensor>(GeTensorDesc(), size);
  (void)AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, value);

  return graph->AddNode(op_desc);
}

static NodePtr CreateFileConstantNode(const ComputeGraphPtr &graph, const string &name, size_t size) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, FILECONSTANT);
  op_desc->AddOutputDesc(GeTensorDesc());
  GeTensorPtr value = std::make_shared<GeTensor>(GeTensorDesc(), 64);
  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, name);
  std::vector<int64_t> shape = {2,2,2,2};
  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  return graph->AddNode(op_desc);
}

TEST_F(UtestHybridModelBuilder, normal_hybrid_model_build) {
/*******************************************************************************
 *    Enter1(another loop)
 *       |
 *      Exit         Identify
 *        \         /       \.        Enter3
 *         \       /         \.        |
 *          Switch           Add <-- Constant
 *         /     |            |
 * Active /      |            |
 *       /       |            |
 *  LoopCond     |            |
 *      \        |            |
 *       \       |            |
 *        \      |            |
 *       Less    |            |
 *          \    |       NextIteration
 *           \   |            |
 *            \  |            |   Active
 *            Merge <---------|
 *              |
 *              |   Active
 *              |
 *            Enter
 ******************************************************************************/
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);

  auto data1 = CreateNode(*graph, "data", DATA, 1, 1);
  auto enter1 = CreateNode(*graph, "enter", ENTER, 1, 1);
  auto enter2 = CreateNode(*graph, "enter2", ENTER, 1, 1);
  auto enter3 = CreateNode(*graph, "enter3", ENTER, 0, 1);
  auto merge1 = CreateNode(*graph, "merge", STREAMMERGE, 2, 2);
  auto constant = CreateNode(*graph, "constant1", CONSTANT, 0, 0);
  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);
  less1->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  auto loop1 = CreateNode(*graph, "loopcond", LOOPCOND, 1, 1);
  auto switch_t = CreateNode(*graph, "switch_t", STREAMSWITCH, 2, 0);
  auto switch_f = CreateNode(*graph, "switch_f", STREAMSWITCH, 2, 0);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 2, 1);
  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  add1->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  auto next1 = CreateNode(*graph, "next", NEXTITERATION, 1, 1);
  auto exit1 = CreateNode(*graph, "exit", EXIT, 1, 1);
  auto value0 = CreateNode(*graph, "const1", CONSTANT, 0, 1);
  auto value1 = CreateNode(*graph, "const2", CONSTANT, 0, 1);
  auto active1 = CreateNode(*graph, "active1", STREAMACTIVE, 0, 0);
  auto active2 = CreateNode(*graph, "active2", STREAMACTIVE, 0, 0);
  auto active3 = CreateNode(*graph, "active3", STREAMACTIVE, 0, 0);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(enter3->GetOutControlAnchor(), constant->GetInControlAnchor());
  GraphUtils::AddEdge(constant->GetOutControlAnchor(), add1->GetInControlAnchor());
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), enter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(enter1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), loop1->GetInDataAnchor(0));

  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(1));
  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(value0->GetOutDataAnchor(0), switch_f->GetInDataAnchor(1));

  GraphUtils::AddEdge(switch_f->GetOutControlAnchor(), exit1->GetInControlAnchor());
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), exit1->GetInDataAnchor(0));

  GraphUtils::AddEdge(switch_t->GetOutControlAnchor(), ident1->GetInControlAnchor());
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), ident1->GetInDataAnchor(0));

  GraphUtils::AddEdge(ident1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), next1->GetInDataAnchor(0));

  GraphUtils::AddEdge(enter1->GetOutControlAnchor(), active1->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(next1->GetOutControlAnchor(), active3->GetInControlAnchor());
  SetNextIteration(merge1, next1);  // for relink NextIteration --> StreamMerge

  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_t->GetInControlAnchor());  // Test for not merge.

  GraphUtils::AddEdge(loop1->GetOutControlAnchor(), active2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_f->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_t->GetInControlAnchor());

  GraphUtils::AddEdge(exit1->GetOutDataAnchor(0), enter2->GetInDataAnchor(0));
  GraphUtils::AddEdge(enter2->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  AttrUtils::SetBool(enter1->GetOpDesc(), ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(output1->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_INSERT_FP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(add1->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);

  SetControlFlowGroup(enter1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(active1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(merge1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(loop1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(active2, switch_t->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_t, switch_t->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_f, switch_t->GetOpDesc()->GetId());
  SetControlFlowGroup(next1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(active3, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(exit1, loop1->GetOpDesc()->GetId());
  SetControlFlowGroup(enter2, enter2->GetOpDesc()->GetId());

  // Build -> IndexSpecialNodes --> stream_merge_op_nodes_
  // Build -> LoadGraph -> RelinkNextIteration
  // Build -> LoadGraph -> LoadDynamicSubgraph --> BuildNodeItem --> NodeItem::SetDataSend
  // Build -> LoadGraph -> LoadDynamicSubgraph --> BuildControlFlowGroup --> NodeItem::SetCtrlSend
  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("AIcoreEngine", NodeExecutorManager::ExecutorType::AICORE);
  engine_mapping.emplace("DNN_VM_GE_LOCAL_OP_STORE", NodeExecutorManager::ExecutorType::GE_LOCAL);
  engine_mapping.emplace("aicpu_tf_kernel", NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping.emplace("aicpu_ascend_kernel", NodeExecutorManager::ExecutorType::AICPU_TF);
  engine_mapping.emplace("ops_kernel_info_hccl", NodeExecutorManager::ExecutorType::HCCL);
  engine_mapping.emplace("DNN_VM_RTS_OP_STORE", NodeExecutorManager::ExecutorType::RTS);
  engine_mapping.emplace("DNN_VM_HOST_CPU_OP_STORE", NodeExecutorManager::ExecutorType::HOST_CPU);

  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::AICORE, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::AICPU_TF, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::HCCL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::RTS, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::HOST_CPU, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  const auto control_group_index = loop1->GetOpDesc()->GetId();
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ge::ExecutionRuntimeUtils::global_in_heterogeneous_executor_ = false;
  ge::ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  ASSERT_EQ(hybrid_model_builder.Build(), SUCCESS);

  const auto TestFrameGroup = [&hybrid_model](const NodePtr &n, int64_t index) {
    const auto it = hybrid_model.node_items_.find(n);
    ASSERT_NE(hybrid_model.node_items_.end(), it);
    ASSERT_EQ(it->second->frame_index_, index);
    ASSERT_EQ(it->second->parent_frame_, -1);
  };
  auto root_graph = hybrid_model.root_graph_;
  auto enter1_node = root_graph->FindNode("enter");
  auto active1_node = root_graph->FindNode("active1");
  auto active2_node = root_graph->FindNode("active2");
  auto active3_node = root_graph->FindNode("active3");
  auto output1_node = root_graph->FindNode("net_output");
  TestFrameGroup(enter1_node, control_group_index);
  TestFrameGroup(active1_node, control_group_index);
  TestFrameGroup(active2_node, control_group_index);
  TestFrameGroup(active3_node, control_group_index);
  TestFrameGroup(output1_node, 2);
  ASSERT_EQ(hybrid_model.GetVariableNode("nothing"), nullptr);
  ASSERT_EQ(hybrid_model.GetGeModel(enter1_node), nullptr);
  ComputeGraphPtr sub_graph = nullptr;
  ASSERT_EQ(hybrid_model.GetSubgraphItem(sub_graph), nullptr);
  ASSERT_EQ(hybrid_model.GetSubgraphItem("nothing"), nullptr);

  vector<vector<int64_t>> batch_info{{1, 2}};
  int32_t dynamic_type = 2;
  ASSERT_EQ(hybrid_model.GetDynamicBatchInfo(batch_info, dynamic_type), SUCCESS);
  ASSERT_EQ(batch_info.size(), 0);
  vector<string> user_input_shape_order{"data1, data2"};
  hybrid_model.GetUserDesignateShapeOrder(user_input_shape_order);
  ASSERT_EQ(user_input_shape_order.size(), 0);
  std::vector<std::string> dynamic_output_shape_info{"test"};
  hybrid_model.GetModelAttr(dynamic_output_shape_info);
  ASSERT_EQ(dynamic_output_shape_info.size(), 0);

  vector<InputOutputDescInfo> inputs;
  vector<InputOutputDescInfo> outputs;
  vector<uint32_t> input_formats;
  vector<uint32_t> output_formats;
  hybrid_model.root_graph_item_->GetOutputNode()->op_desc->SetSrcIndex({0});
  hybrid_model.root_graph_item_->GetOutputNode()->op_desc->SetSrcName({"test"});
  ASSERT_EQ(hybrid_model.GetInputOutputDescInfo(inputs, outputs, input_formats, output_formats), SUCCESS);
  engine_mapping.clear();
  task_executor.clear();
}

TEST_F(UtestHybridModelBuilder, create_called_invalid) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto node = CreateNode(*graph, "node", PARTITIONEDCALL, 1, 1);
  NodeItem node_item(node);

  auto switch_node = CreateNode(*graph, "switch_node", SWITCH, 1, 1);
  NodeItem switch_node_item(switch_node);

  ASSERT_EQ(hybrid_model_builder.CreateStreamActiveGroup(node, node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateStreamSwitchGroup(node, node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateNextIterationGroup(node, node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateSwitchGroup(node, node_item), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.CreateSwitchGroup(switch_node, switch_node_item), SUCCESS);

  ASSERT_EQ(hybrid_model_builder.CreateNotImplement(node, node_item), UNSUPPORTED);
}

TEST_F(UtestHybridModelBuilder, init_constant_op_host_) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto const_1 = CreateConstantNode(graph, "const_1", 0);
  hybrid_model_builder.constant_op_nodes_.emplace(const_1->GetName(), const_1);
  auto const_2 = CreateConstantNode(graph, "const_2", 10);
  hybrid_model_builder.constant_op_nodes_.emplace(const_2->GetName(), const_2);

  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(options);

  EXPECT_EQ(hybrid_model_builder.InitConstantOps(), SUCCESS);
  auto tensor_data1 = "test_hybrid";
  GeTensor tensor(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_STRING), (uint8_t *)tensor_data1, 8);
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 120);
  auto tensor_value = std::unique_ptr<TensorValue>(new TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release())));
  EXPECT_EQ(hybrid_model_builder.CopyConstantData(const_1, tensor, tensor_value), SUCCESS);
  EXPECT_EQ(hybrid_model_builder.hybrid_model_.variable_tensors_.size(), 2);
}

TEST_F(UtestHybridModelBuilder, init_FileConstant_op_host) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto const_1 = CreateFileConstantNode(graph, "file_const_1", 0);
  auto const_2 = CreateFileConstantNode(graph, "file_const_2", 10);
  hybrid_model_builder.constant_op_nodes_.emplace(const_1->GetName(), const_1);
  hybrid_model_builder.constant_op_nodes_.emplace(const_2->GetName(), const_2);

  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(options);

  std::unique_ptr<float[]> float_buf(new float[16]);
  std::string file_name = "./test_copy_one_weight.bin";
  std::ofstream out1("./test_copy_one_weight.bin", std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write((char *)float_buf.get(), 16 * sizeof(float));
  out1.close();
  hybrid_model_builder.file_id_and_path_map_.insert(
      std::pair<std::string, std::string>("file_const_1", "./test_copy_one_weight.bin"));
  hybrid_model_builder.file_id_and_path_map_.insert(
      std::pair<std::string, std::string>("file_const_2", "./test_copy_one_weight.bin"));
  EXPECT_EQ(hybrid_model_builder.InitFileConstantOps(), SUCCESS);
  (void)remove("test_copy_one_weight.bin");
}

TEST_F(UtestHybridModelBuilder, init_FileConstant_op_dev_) {
  MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
  VarManager::Instance(0)->Init(0, 0, 0, 0);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  hybrid_model_builder.var_manager_ = ge::VarManager::Instance(0);
  hybrid_model_builder.var_manager_->SetMemManager(&MemManager::Instance());

  auto const_1 = CreateFileConstantNode(graph, "file_const_1", 0);
  hybrid_model_builder.constant_op_nodes_.emplace(const_1->GetName(), const_1);
  auto const_2 = CreateFileConstantNode(graph, "file_const_2", 10);
  hybrid_model_builder.constant_op_nodes_.emplace(const_2->GetName(), const_2);

  VarManager::Instance(0)->AssignVarMem("file_const_1", nullptr, const_1->GetOpDesc()->GetOutputDesc(0), RT_MEMORY_HBM);
  VarManager::Instance(0)->AssignVarMem("file_const_2", nullptr, const_2->GetOpDesc()->GetOutputDesc(0), RT_MEMORY_HBM);
  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST1";
  GetThreadLocalContext().SetGraphOption(options);
  std::unique_ptr<float[]> float_buf(new float[16]);
  std::string file_name = "test_copy_one_weight.bin";
  std::ofstream out1("test_copy_one_weight.bin", std::ios::binary);
  if (!out1.is_open()) {
    return;
  }
  out1.write((char *)float_buf.get(), 16 * sizeof(float));
  out1.close();
  hybrid_model_builder.file_id_and_path_map_.insert(
      std::pair<std::string, std::string>("file_const_1", "test_copy_one_weight.bin"));
  hybrid_model_builder.file_id_and_path_map_.insert(
      std::pair<std::string, std::string>("file_const_2", "test_copy_one_weight.bin"));
  EXPECT_EQ(hybrid_model_builder.InitFileConstantOps(), SUCCESS);
  (void)remove("test_copy_one_weight.bin");
}

TEST_F(UtestHybridModelBuilder, init_host_var_with_host_mem) {
  MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  OpDescPtr op_desc = std::make_shared<OpDesc>("host_params", VARIABLE);
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NHWC, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 512);
  op_desc->AddOutputDesc(tensor_desc);
  auto host_var = graph->AddNode(op_desc);

  hybrid_model.host_variable_nodes_.emplace("host_params", host_var);
  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(options);
  hybrid_model_builder.var_manager_  = VarManager::Instance(0);
  ASSERT_NE(hybrid_model.GetVariableNode("host_params"), nullptr);
  EXPECT_EQ(hybrid_model_builder.InitVariableTensors(), SUCCESS);
  EXPECT_EQ(hybrid_model_builder.hybrid_model_.variable_tensors_.size(), 1);
}

TEST_F(UtestHybridModelBuilder, init_host_var_with_host_shared_mem) {
  MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  OpDescPtr op_desc = std::make_shared<OpDesc>("host_params", VARIABLE);
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NHWC, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 512);
  op_desc->AddOutputDesc(tensor_desc);
  auto host_var = graph->AddNode(op_desc);

  hybrid_model.host_variable_nodes_.emplace("host_params", host_var);
  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(options);

  SharedMemInfo info;
  uint8_t tmp(0);
  info.device_address = &tmp;
  std::shared_ptr<AlignedPtr> aligned_ptr = std::make_shared<AlignedPtr>(512, 16);
  info.host_aligned_ptr = aligned_ptr;
  info.fd = 0;
  info.mem_size = 100;
  info.op_name = "host_params";
  HostMemManager::Instance().var_memory_base_map_["host_params"] = info;

  hybrid_model_builder.var_manager_  = VarManager::Instance(0);
  EXPECT_EQ(hybrid_model_builder.InitVariableTensors(), SUCCESS);
  EXPECT_EQ(hybrid_model_builder.hybrid_model_.variable_tensors_.size(), 1);
  HostMemManager::Instance().var_memory_base_map_.clear();
}

TEST_F(UtestHybridModelBuilder, InitModelMemOnHost) {
  MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  OpDescPtr op_desc = std::make_shared<OpDesc>("host_params", VARIABLE);
  GeTensorDesc tensor_desc(GeShape(), FORMAT_NHWC, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 512);
  op_desc->AddOutputDesc(tensor_desc);
  auto host_var = graph->AddNode(op_desc);

  hybrid_model.host_variable_nodes_.emplace("host_params", host_var);
  std::map<std::string, string> options;
  options["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetGraphOption(options);

  hybrid_model_builder.var_manager_  = VarManager::Instance(0);
  EXPECT_EQ(hybrid_model_builder.InitModelMem(), SUCCESS);
  HostMemManager::Instance().var_memory_base_map_.clear();
}

TEST_F(UtestHybridModelBuilder, TestInitHcclExecutorOnDemand) {
  NodeExecutorManager::GetInstance().builders_.erase(NodeExecutorManager::ExecutorType::HCCL);
  // build aicore task
  domi::ModelTaskDef model_task_def;
  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  // No hccl task
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), SUCCESS);

  // get executor failed due to no builder
  task_def = model_task_def_ptr->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), INTERNAL_ERROR);

  // get executor success
  REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HCCL, NodeExecutor);
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), SUCCESS);

  // repeat get, do not access builder
  NodeExecutorManager::GetInstance().builders_.erase(NodeExecutorManager::ExecutorType::HCCL);
  ASSERT_EQ(HybridModelBuilder::InitHcclExecutorOnDemand(ge_model), SUCCESS);
}

TEST_F(UtestHybridModelBuilder, copy_graph_success) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
HybridModel hybrid_model(ge_root_model);
HybridModelBuilder hybrid_model_builder(hybrid_model);

Status st = hybrid_model_builder.CopyGraph();
EXPECT_EQ(st, SUCCESS);
}

TEST_F(UtestHybridModelBuilder, init_data_aipp_info_and_type) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("GeLocal", NodeExecutorManager::ExecutorType::GE_LOCAL);
  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_dynamic));
  aipp_attr.SetAttr("related_input_rank", GeAttrValue::CreateFrom<int64_t>(0));
  aipp_attr.SetAttr("max_src_image_size", GeAttrValue::CreateFrom<int64_t>(2048));
  aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<int64_t>(1));

  {
    OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
    GeTensorDesc tensor_desc(GeShape(),FORMAT_NHWC,DT_FLOAT);
    TensorUtils::SetSize(tensor_desc, 512);
    op_desc->AddInputDesc(tensor_desc);
    op_desc->AddOutputDesc(tensor_desc);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    op_desc->SetOpKernelLibName("GeLocal");
    auto data_node = graph->AddNode(op_desc);
    AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
    AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr);
    AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
    AttrUtils::SetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp_data");
  }

  {
    OpDescPtr op_desc = std::make_shared<OpDesc>("releated_aipp_data", AIPPDATA);
    GeTensorDesc tensor_desc(GeShape(),FORMAT_NHWC,DT_FLOAT);
    TensorUtils::SetSize(tensor_desc, 512);
    op_desc->AddInputDesc(tensor_desc);
    op_desc->AddOutputDesc(tensor_desc);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    op_desc->SetOpKernelLibName("GeLocal");
    auto aipp_data_node = graph->AddNode(op_desc);
    AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 1);
    AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr);
    AttrUtils::SetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_conf");
    AttrUtils::SetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "data");
  }

  AippConfigInfo aipp_info;
  InputAippType aipp_type;
  size_t aipp_index = 0;
  Status ret;
  // Has not set
  ret = hybrid_model_builder.hybrid_model_.GetAippInfo(0, aipp_info);
  EXPECT_EQ(ret, ACL_ERROR_GE_AIPP_NOT_EXIST);
  ret = hybrid_model_builder.hybrid_model_.GetAippType(0, aipp_type, aipp_index);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITHOUT_AIPP);
  EXPECT_EQ(aipp_index, 0xFFFFFFFF);

  // Set aipp infos and types when Build
  ret = hybrid_model_builder.Build();
  ASSERT_EQ(ret, SUCCESS);

  // Has been set
  ret = hybrid_model_builder.hybrid_model_.GetAippInfo(0, aipp_info);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(aipp_info.aipp_mode, domi::AippOpParams_AippMode_dynamic);
  ret = hybrid_model_builder.hybrid_model_.GetAippType(0, aipp_type, aipp_index);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITH_DYNAMIC_AIPP);
  EXPECT_EQ(aipp_index, 1);
}

TEST_F(UtestHybridModelBuilder, init_data_aipp_info_and_type_static_mode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("GeLocal", NodeExecutorManager::ExecutorType::GE_LOCAL);
  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_static_));
  aipp_attr.SetAttr("related_input_rank", GeAttrValue::CreateFrom<int64_t>(0));
  aipp_attr.SetAttr("max_src_image_size", GeAttrValue::CreateFrom<int64_t>(2048));
  aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<int64_t>(1));

  OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDesc data_tensor_desc(GeShape(),FORMAT_NHWC,DT_FLOAT);
  TensorUtils::SetSize(data_tensor_desc, 512);
  data_op_desc->AddInputDesc(data_tensor_desc);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  data_op_desc->SetInputOffset({1024});
  data_op_desc->SetOutputOffset({1024});
  data_op_desc->SetOpKernelLibName("GeLocal");
  auto data_node = graph->AddNode(data_op_desc);

  // Static mode
  AttrUtils::SetNamedAttrs(data_op_desc, ATTR_NAME_AIPP, aipp_attr);
  AttrUtils::SetStr(data_op_desc, ATTR_DATA_RELATED_AIPP_MODE, "static_aipp");

  HybridModelBuilder hybrid_model_builder(hybrid_model);
  auto ret = hybrid_model_builder.Build();
  ASSERT_EQ(ret, SUCCESS);

  AippConfigInfo aipp_info;
  InputAippType aipp_type;
  size_t aipp_index = 0;
  ret = hybrid_model_builder.hybrid_model_.GetAippInfo(0, aipp_info);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(aipp_info.aipp_mode, domi::AippOpParams_AippMode_static_);
  ret = hybrid_model_builder.hybrid_model_.GetAippType(0, aipp_type, aipp_index);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(aipp_type, DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp_index, 0xFFFFFFFF);
}

TEST_F(UtestHybridModelBuilder, init_data_aipp_info_and_type_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("GeLocal", NodeExecutorManager::ExecutorType::GE_LOCAL);
  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_dynamic));
  aipp_attr.SetAttr("related_input_rank", GeAttrValue::CreateFrom<int64_t>(0));
  aipp_attr.SetAttr("max_src_image_size", GeAttrValue::CreateFrom<int64_t>(2048));
  aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<int64_t>(1));

  OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDesc data_tensor_desc(GeShape(),FORMAT_NHWC,DT_FLOAT);
  TensorUtils::SetSize(data_tensor_desc, 512);
  data_op_desc->AddInputDesc(data_tensor_desc);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  data_op_desc->SetInputOffset({1024});
  data_op_desc->SetOutputOffset({1024});
  data_op_desc->SetOpKernelLibName("GeLocal");
  auto data_node = graph->AddNode(data_op_desc);

  // Both ATTR_NAME_AIPP and ATTR_DATA_RELATED_AIPP_MODE attributes are needed, only has ATTR_NAME_AIPP.
  {
    AttrUtils::SetNamedAttrs(data_op_desc, ATTR_NAME_AIPP, aipp_attr);
    HybridModelBuilder hybrid_model_builder(hybrid_model);
    auto ret = hybrid_model_builder.Build();
    EXPECT_EQ(ret, INTERNAL_ERROR);
  }

  // Dynamic mode does not has ATTR_DATA_AIPP_DATA_NAME_MAP
  {
    AttrUtils::SetNamedAttrs(data_op_desc, ATTR_NAME_AIPP, aipp_attr);
    AttrUtils::SetStr(data_op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");

    HybridModelBuilder hybrid_model_builder(hybrid_model);
    auto ret = hybrid_model_builder.Build();
    EXPECT_EQ(ret, INTERNAL_ERROR);
  }

  // Invalid mode
  {
    AttrUtils::SetNamedAttrs(data_op_desc, ATTR_NAME_AIPP, aipp_attr);
    AttrUtils::SetStr(data_op_desc, ATTR_DATA_RELATED_AIPP_MODE, "invalid_mode");
    AttrUtils::SetStr(data_op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp_data");

    HybridModelBuilder hybrid_model_builder(hybrid_model);
    auto ret = hybrid_model_builder.Build();
    EXPECT_EQ(ret, INTERNAL_ERROR);
  }

  // Can not find releated AippData node
  {
    AttrUtils::SetNamedAttrs(data_op_desc, ATTR_NAME_AIPP, aipp_attr);
    AttrUtils::SetStr(data_op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
    AttrUtils::SetStr(data_op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, "releated_aipp_data");

    HybridModelBuilder hybrid_model_builder(hybrid_model);
    auto ret = hybrid_model_builder.Build();
    EXPECT_EQ(ret, INTERNAL_ERROR);
  }
}

TEST_F(UtestHybridModelBuilder, build_node_with_fused_graph_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
  engine_mapping.emplace("DNN_VM_RTS_OP_STORE", NodeExecutorManager::ExecutorType::RTS);
  engine_mapping.emplace("DNN_VM_GE_LOCAL_OP_STORE", NodeExecutorManager::ExecutorType::GE_LOCAL);
  auto &task_executor = NodeExecutorManager::GetInstance().executors_;
  task_executor.emplace(NodeExecutorManager::ExecutorType::RTS, std::unique_ptr<NodeExecutor>(new NodeExecutor()));
  task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new NodeExecutor()));

  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  auto data1 = CreateNode(*sub_graph, "data", DATA, 1, 1);
  auto abs1 = CreateNode(*sub_graph, "abs", ABSVAL, 1, 1);
  auto output1 = CreateNode(*sub_graph, "output1", "_RetVal", 1, 0);
  abs1->GetOpDesc()->SetOpInferDepends({"__input0"});
  
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  AttrUtils::SetInt(output1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), abs1->GetInDataAnchor(0));
  GraphUtils::AddEdge(abs1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  auto root_data = CreateNode(*graph, "root_data", DATA, 1, 1);
  auto fused_node = CreateNode(*graph, "fused_node", ABSVAL, 1, 1);
  auto root_out = CreateNode(*graph, "root_out", "_RetVal", 1, 0);
  AttrUtils::SetGraph(fused_node->GetOpDesc(), "_original_fusion_graph", sub_graph);
  AttrUtils::SetBool(fused_node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  AttrUtils::SetBool(fused_node->GetOpDesc(), ATTR_NAME_INSERT_BP_PROFILILNG_TASK, true);
  AttrUtils::SetBool(fused_node->GetOpDesc(), ATTR_NAME_INSERT_END_PROFILILNG_TASK, true);
  GraphUtils::AddEdge(root_data->GetOutDataAnchor(0), fused_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(fused_node->GetOutDataAnchor(0), root_out->GetInDataAnchor(0));

  HybridModelBuilder hybrid_model_builder(hybrid_model);
  hybrid_model_builder.hybrid_model_.device_id_ = 2048;
  auto ret = hybrid_model_builder.Build();
  EXPECT_EQ(ret, SUCCESS);

  NodeItem *fused_node_item = nullptr;
  auto &node_items = hybrid_model_builder.hybrid_model_.node_items_;
  for (const auto &item : node_items) {
    if (item.first->GetName() == fused_node->GetName()) {
      fused_node_item = item.second.get();
      break;
    }
  }
  ASSERT_NE(fused_node_item, nullptr);
  // check fused_graph retval replaced by netoutput
  ASSERT_NE(fused_node_item->fused_subgraph->output_mapping.size(), 0);
  NodePtr netoutput = nullptr;
  for (const auto &node : fused_node_item->fused_subgraph->graph->GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      netoutput = node;
    }
  }
  ASSERT_NE(netoutput, nullptr);
  // check parent node index on netoutput
  for (const auto &input_desc : netoutput->GetOpDesc()->GetAllInputsDesc()) {
    ASSERT_TRUE(AttrUtils::HasAttr(input_desc, ATTR_NAME_PARENT_NODE_INDEX));
  }

  GraphExecutionContext execution_context;
  ShapeInferenceEngine shape_infer_engine(&execution_context, false);
  // not create RuntimeInferenceContext
  ret = shape_infer_engine.SetDependingTensor(*(fused_node_item->fused_subgraph));
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // not SetTensor
  ret = shape_infer_engine.SetDependingTensor(*(fused_node_item->fused_subgraph));
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // normal
  auto data_dep = fused_node_item->fused_subgraph->data_dependencies;
  auto src_node_id = data_dep.at(0).first.first;
  auto src_output_idx = data_dep.at(0).first.second;
  GeTensorPtr tensor = make_shared<GeTensor>();
  auto status = execution_context.runtime_context_.SetTensor(src_node_id, src_output_idx, tensor);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  ret = shape_infer_engine.InferShapeForSubgraph(*fused_node_item, *(fused_node_item->fused_subgraph));
  EXPECT_EQ(ret, SUCCESS);
  auto dst_op_desc = data_dep.at(0).second.first;
  auto dst_input_idx = data_dep.at(0).second.second;
  auto dst_tensor_desc = dst_op_desc->MutableInputDesc(static_cast<uint32_t>(dst_input_idx));
  bool has_val = AttrUtils::HasAttr(dst_tensor_desc, ATTR_NAME_VALUE);
  EXPECT_EQ(has_val, true);
  EXPECT_EQ(hybrid_model_builder.LoadKnownNodeItem(*hybrid_model.root_graph_item_.get(), fused_node, fused_node->GetOpDesc()),
            SUCCESS);
  std::vector<domi::TaskDef> task_def_list;
  EXPECT_EQ(hybrid_model_builder.GenerateArProfilingTask(fused_node->GetOpDesc(), 1, task_def_list), SUCCESS);
  EXPECT_EQ(hybrid_model_builder.LoadKnownShapedSubgraph(*graph, *fused_node_item), FAILED);
}

TEST_F(UtestHybridModelBuilder, TestRecoverShapeInconsistency) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto data = CreateNode(*graph, "data", DATA, 1, 1);
  auto foo = CreateNode(*graph, "Foo", "Foo", 1, 0);
  auto tensor_desc = foo->GetOpDesc()->MutableInputDesc(0);
  auto shape = tensor_desc->GetShape();
  tensor_desc->SetShape(GeShape({-2}));
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), foo->GetInDataAnchor(0));
  ASSERT_EQ(HybridModelBuilder::RecoverShapeConsistency(*graph), SUCCESS);
  ASSERT_EQ(tensor_desc->GetShape(), shape);
}

TEST_F(UtestHybridModelBuilder, AssignData2Fp32Var) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  auto node = CreateNode(*graph, "node", DATA, 0, 0);
  NodeItem node_item(node);

  AttrUtils::SetStr(node->GetOpDesc(), VAR_ATTR_SRC_VAR_NAME, std::string("src_name"));
  VarMemAssignUtil::AssignData2Fp32Var(node, -1);
}

TEST_F(UtestHybridModelBuilder, PrintDynamicType_dynamic_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto data = CreateNode(*graph, "data", DATA, 1, 1);
  data->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({-1}), FORMAT_ND, DT_FLOAT));
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.Init(false);
}

TEST_F(UtestHybridModelBuilder, PrintDynamicType_dynamic_progress) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto data = CreateNode(*graph, "add", ADD, 1, 1);
  data->GetOpDesc()->AddInputDesc(GeTensorDesc(GeShape({-1}), FORMAT_ND, DT_FLOAT));
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.Init(false);
}

TEST_F(UtestHybridModelBuilder, test_hybrid_get_method) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  GeTensorDesc desc;
  auto output_desc = std::make_shared<GeTensorDesc>(GeShape({1, -1, 3, 4}), FORMAT_FRACTAL_Z);
  output_desc->SetShapeRange({{1,10},{2,10},{3,10},{4,10}});
  uint32_t format_res = 0;
  InputOutputDescInfo output_desc_info;
  hybrid_model.CreateOutput(output_desc, output_desc_info, format_res);
  ASSERT_EQ(format_res, static_cast<uint32_t>(FORMAT_HWCN));
  auto node_ptr = std::make_shared<Node>();
  ASSERT_EQ(hybrid_model.GetConstant(nullptr), nullptr);
  ASSERT_EQ(hybrid_model.GetConstant(node_ptr), nullptr);
  ASSERT_EQ(hybrid_model.GetTensor(node_ptr), nullptr);
  string attr_val;
  ASSERT_EQ(hybrid_model.GetOpAttr("test", "test_attr", attr_val), SUCCESS);
  hybrid_model.op_name_to_attrs_["test"] = {};
  ASSERT_EQ(hybrid_model.GetOpAttr("test", "test_attr", attr_val), SUCCESS);
  hybrid_model.op_name_to_attrs_["test"] = {{"test_attr", {"1", "2", "3"}}};
  ASSERT_EQ(hybrid_model.GetOpAttr("test", "test_attr", attr_val), SUCCESS);
  ASSERT_EQ(attr_val, "[1]1[1]2[1]3");

  std::vector<InputOutputDescInfo> input_descs;
  std::vector<InputOutputDescInfo> output_descs;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;
  hybrid_model.root_graph_item_.reset(new GraphItem());
  ASSERT_NE(hybrid_model.GetInputOutputDescInfo(input_descs, output_descs, input_formats, output_formats), SUCCESS);
}

TEST_F(UtestHybridModelBuilder, test_hybrid_check_host_mem_input_optimization) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);

  auto node = CreateNode(*graph, "node", DATA, 0, 0);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  hybrid_model.node_items_[node] = std::move(node_item);

  std::vector<NodePtr> node_with_hostmem;
  // node with hostmem is empty
  EXPECT_FALSE(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem));

  // kernel task is null
  node_with_hostmem.emplace_back(node);
  EXPECT_FALSE(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem));

  // args is not extended for host mem input
  domi::TaskDef task_def;
  auto aicpu_node = MakeShared<AicpuNodeTask>(hybrid_model.node_items_[node].get(), task_def);
  hybrid_model.node_items_[node]->kernel_task = aicpu_node;
  EXPECT_FALSE(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem));

  // return true
  aicpu_node->host_mem_input_data_offset_ = 1U;
  EXPECT_TRUE(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem));
  hybrid_model.SetNeedHostMemOpt(node_with_hostmem, true);
  EXPECT_TRUE(aicpu_node->need_host_mem_opt_);
}

TEST_F(UtestHybridModelBuilder, normal_hybrid_model_build_on_host) {
  DEF_GRAPH(graph1) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeType2Op")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto var1 = OP_CFG(VARIABLEV2)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("fused_op1", fake_type2_op1)
              ->NODE("fused_op2", fake_type2_op2)
              ->NODE("fused_op3", fake_type2_op3)
              ->NODE("Node_Output", net_output));
    CHAIN(NODE("_var", var1)->NODE("Node_Output", net_output));
  };
  auto root_graph = ToComputeGraph(graph1);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  auto var1 = root_graph->FindFirstNodeMatchType(VARIABLEV2);
  ASSERT_TRUE(var1 != nullptr);
  std::cout << "var1->GetOpDesc()->MutableOutputDesc(0)->GetShape().ToString() = "
            << var1->GetOpDesc()->MutableOutputDesc(0)->GetShape().ToString() << std::endl;
  std::vector<uint8_t> i(16, 0);
  DataBuffer weight_buf(i.data(), i.size());
  ge_sub_model->SetWeightDataBuf(weight_buf);
  ge_sub_model->SetGraph(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);

  const auto graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> options;
  options["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ASSERT_TRUE(ge::GetContext().GetHostExecFlag());

  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_ = root_graph;
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.InitWeights(), SUCCESS);
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::hybrid::NpuMemoryAllocator::Finalize();
}

}  // namespace ge
