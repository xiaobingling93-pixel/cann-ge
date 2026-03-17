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
#include <nlohmann/json.hpp>
#include "fe_llt_utils.h"
#include "common/fe_op_info_common.h"
#include "common/scope_allocator.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "ops_kernel_builder/aicore_ops_kernel_builder.h"
#include "graph/utils/tensor_utils.h"
#include "ops_kernel_builder/task_builder/superkernel_task_builder.h"
#include "ops_kernel_builder/task_builder/superkernel_args_format_utils.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "register/op_ext_gentask_registry.h"
#define private public
#include "common/platform_utils.h"
#undef private

ge::Status StubFunc(const ge::Node &node, std::vector<std::vector<domi::TaskDef>> &subTasks,
                    const std::vector<ge::Node *> &sub_nodes, std::vector<domi::TaskDef> &tasks) {
  domi::TaskDef sk_task = {};
  tasks.emplace_back(sk_task);
  return fe::SUCCESS;
}

ge::Status StubFuncFailed(const ge::Node &node, std::vector<std::vector<domi::TaskDef>> &subTasks,
                    const std::vector<ge::Node *> &sub_nodes, std::vector<domi::TaskDef> &tasks) {
  return fe::FAILED;
}

using namespace ge;
namespace fe {
class SuperkernelTaskBuilderUT : public testing::Test {
protected:
  static void SetUpTestCase() {
    OpStoreAdapterManager::Instance(AI_CORE_NAME).Finalize();
    map<string, string> options;
    OpStoreAdapterManager::Instance(AI_CORE_NAME).Initialize(options);
    fe::PlatformInfoManager::Instance().InitializePlatformInfo();
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
    cout << "SuperkernelTaskBuilderUT SetUp" << endl;
  }

  static void TearDownTestCase() {
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
    cout << "SuperkernelTaskBuilderUT TearDown" << endl;
  }

  /**
   *    Const \
   * Data -> Conv2D -> Relu -> Add -> SoftmaxV2 -> StridedSliceD
   *          Data -> Sigmoid -/
   *
   */
  ge::ComputeGraphPtr CreateGraphWithType(const int case_type) {
    vector<int64_t> dims = {3, 4, 5, 6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    ge::TensorUtils::SetSize(tensor_desc, 1472);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "PlaceHolder");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    OpDescPtr softmax_op = std::make_shared<OpDesc>("softmax", "SoftmaxV2");
    OpDescPtr add_op = std::make_shared<OpDesc>("add", "Add");
    OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");

    data1_op->AddOutputDesc(tensor_desc);
    data2_op->AddOutputDesc(tensor_desc);
    const_op->AddOutputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddOutputDesc(tensor_desc);
    relu_op->AddInputDesc(tensor_desc);
    relu_op->AddOutputDesc(tensor_desc);
    sigmoid_op->AddInputDesc(tensor_desc);
    sigmoid_op->AddOutputDesc(tensor_desc);
    add_op->AddInputDesc(tensor_desc);
    add_op->AddInputDesc(tensor_desc);
    add_op->AddOutputDesc(tensor_desc);
    softmax_op->AddInputDesc(tensor_desc);
    softmax_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(sigmoid_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(add_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(softmax_op, "_fe_imply_type", 6);

    AttrUtils::SetStr(conv_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(relu_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(sigmoid_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(add_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(softmax_op, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

    AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});
    AttrUtils::SetBool(conv_op, ATTR_NAME_IS_FIRST_NODE, true);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr const_node = graph->AddNode(const_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
    NodePtr add_node = graph->AddNode(add_op);
    NodePtr softmax_node = graph->AddNode(softmax_op);
    ge::AttrUtils::SetStr(data1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(const_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(conv_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(data2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(sigmoid_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(add_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(softmax_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    int64_t scope_id_1 = ScopeAllocator::Instance().AllocateSkpScopeId();
    switch (case_type) {
      case 1:
        ScopeAllocator::SetSkpScopeAttr(conv_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(relu_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(sigmoid_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(add_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(softmax_op, scope_id_1);
        AttrUtils::SetBool(relu_op, ge::ATTR_NAME_NOTASK, true);
        AttrUtils::SetBool(sigmoid_op, ge::ATTR_NAME_NOTASK, true);
        AttrUtils::SetBool(add_op, ge::ATTR_NAME_NOTASK, true);
        AttrUtils::SetBool(softmax_op, ge::ATTR_NAME_NOTASK, true);
        break;
      case 2:
        ScopeAllocator::SetSkpScopeAttr(conv_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(relu_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(sigmoid_op, scope_id_1);
        ScopeAllocator::SetSkpScopeAttr(softmax_op, scope_id_1);
        AttrUtils::SetBool(relu_op, ge::ATTR_NAME_NOTASK, true);
        AttrUtils::SetBool(sigmoid_op, ge::ATTR_NAME_NOTASK, true);
        AttrUtils::SetBool(softmax_op, ge::ATTR_NAME_NOTASK, true);
        break;
    }

    return graph;
  }
};

TEST_F(SuperkernelTaskBuilderUT, superkernel_plus_case1) {
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(1);

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  FillGraphNodeParaType(graph_ptr);
  ge::ComputeGraphPtr sub_graph_ptr = CreateGraphWithType(1);
  FillGraphNodeParaType(sub_graph_ptr);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_ptr);
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      Status ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 2);
}

TEST_F(SuperkernelTaskBuilderUT, superkernel_plus_case2) {
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);

  shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  FillGraphNodeParaType(graph_ptr);
  ge::ComputeGraphPtr sub_graph_ptr = CreateGraphWithType(2);
  FillGraphNodeParaType(sub_graph_ptr);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_ptr);
    if (IsTbeOp(node->GetOpDesc()) && !IsNoTaskOp(node)) {
      Status ret = ops_kernel_builder->GenerateTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 4);
}

TEST_F(SuperkernelTaskBuilderUT, superkernel_plus_case3) {
  auto options_bk = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.buildMode"] = "tuning";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);
  shared_ptr<SuperkernelTaskBuilder> super_kernel_builder = make_shared<SuperkernelTaskBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  FillGraphNodeParaType(graph_ptr);
  ge::ComputeGraphPtr sub_graph_ptr = CreateGraphWithType(2);
  FillGraphNodeParaType(sub_graph_ptr);
  for (const ge::NodePtr &node : sub_graph_ptr->GetDirectNode()) {
    if (node->GetType() == "PlaceHolder" || node->GetType() == "Const") {
      continue;
    }
    ge::AttrUtils::SetInt(node->GetOpDesc(), "_op_dfx_buffer_size", 10086);
    if (node->GetType() == "SoftmaxV2") {
      std::vector<int64_t> workspace = {10086};
      node->GetOpDesc()->SetWorkspaceBytes(workspace);
      node->GetOpDesc()->SetWorkspace(workspace);
    }
  }

  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_ptr);
      Status ret = super_kernel_builder->GenerateSuperKernelTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
      break;
    }
  }
  EXPECT_EQ(tasks.size(), 1);

  OpExtGenTaskRegistry::GetInstance().RegisterSKFunc("Conv2D", StubFuncFailed);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      Status ret = super_kernel_builder->GenerateSuperKernelTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 2);

  OpExtGenTaskRegistry::GetInstance().RegisterSKFunc("Conv2D", StubFunc);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      Status ret = super_kernel_builder->GenerateSuperKernelTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 3);
  ge::GetThreadLocalContext().SetGraphOption(options_bk);
  vector<int64_t> dims = {3, 4, 5, 6};
  ge::GeShape shape(dims);
  ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  ge::TensorUtils::SetSize(tensor_desc, 1472);
  OpDescPtr sk_op = std::make_shared<OpDesc>("sk", "SuperKernel");
  ComputeGraphPtr graph_tmp = std::make_shared<ComputeGraph>("sk_graph");
  NodePtr sk_node = graph_tmp->AddNode(sk_op);
  tasks.clear();
  std::shared_ptr<AICoreOpsKernelBuilder> ops_kernel_builder = make_shared<AICoreOpsKernelBuilder>();
  ops_kernel_builder->GenerateTask(*sk_node, context, tasks);
}

TEST_F(SuperkernelTaskBuilderUT, superkernel_plus_reuse_binary_not_tiling_sink_failed) {
  auto options_bk = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.buildMode"] = "tuning";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);
  shared_ptr<SuperkernelTaskBuilder> super_kernel_builder = make_shared<SuperkernelTaskBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  FillGraphNodeParaType(graph_ptr);
  ge::ComputeGraphPtr sub_graph_ptr = CreateGraphWithType(2);
  FillGraphNodeParaType(sub_graph_ptr);
  for (const ge::NodePtr &node : sub_graph_ptr->GetDirectNode()) {
    if (node->GetType() == "PlaceHolder" || node->GetType() == "Const") {
      continue;
    }
    ge::AttrUtils::SetInt(node->GetOpDesc(), "_op_dfx_buffer_size", 10086);
    ge::AttrUtils::SetBool(node->GetOpDesc(), "super_kernel_reuse_binary", true);
    if (node->GetType() == "SoftmaxV2") {
      std::vector<int64_t> workspace = {10086};
      node->GetOpDesc()->SetWorkspaceBytes(workspace);
      node->GetOpDesc()->SetWorkspace(workspace);
    }
  }
  OpExtGenTaskRegistry::GetInstance().RegisterSKFunc("Conv2D", StubFunc);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_ptr);
      Status ret = super_kernel_builder->GenerateSuperKernelTask(*node, context, tasks);
      EXPECT_EQ(ret, FAILED);
    }
  }
  ge::GetThreadLocalContext().SetGraphOption(options_bk);
}
TEST_F(SuperkernelTaskBuilderUT, superkernel_plus_case4) {
  auto options_bk = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.buildMode"] = "tuning";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);
  shared_ptr<SuperkernelTaskBuilder> super_kernel_builder = make_shared<SuperkernelTaskBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  FillGraphNodeParaType(graph_ptr);
  ge::ComputeGraphPtr sub_graph_ptr = CreateGraphWithType(2);
  FillGraphNodeParaType(sub_graph_ptr);
  for (const ge::NodePtr &node : sub_graph_ptr->GetDirectNode()) {
    if (node->GetType() == "PlaceHolder" || node->GetType() == "Const") {
      continue;
    }
    ge::AttrUtils::SetInt(node->GetOpDesc(), "_op_dfx_buffer_size", 1);
    if (node->GetType() == "SoftmaxV2") {
      std::vector<int64_t> workspace = {1};
      node->GetOpDesc()->SetWorkspaceBytes(workspace);
      node->GetOpDesc()->SetWorkspace(workspace);
    }
  }
  OpExtGenTaskRegistry::GetInstance().RegisterSKFunc("Conv2D", StubFunc);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    ge::AttrUtils::SetInt(node->GetOpDesc(), "_op_dfx_buffer_size", 1);
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_ptr);
      Status ret = super_kernel_builder->GenerateSuperKernelTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 1);
  ge::GetThreadLocalContext().SetGraphOption(options_bk);
}
TEST_F(SuperkernelTaskBuilderUT, superkernel_plus_case5) {
  auto options_bk = ge::GetThreadLocalContext().GetAllGraphOptions();
  std::map<std::string, std::string> option_tmp;
  option_tmp["ge.buildMode"] = "tuning";
  ge::GetThreadLocalContext().SetGraphOption(option_tmp);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);
  shared_ptr<SuperkernelTaskBuilder> super_kernel_builder = make_shared<SuperkernelTaskBuilder>();
  uint8_t base = 128;
  ge::RunContext context;
  context.dataMemBase = &base;
  context.weightMemBase = &base;
  std::vector<domi::TaskDef> tasks;
  FillGraphNodeParaType(graph_ptr);
  ge::ComputeGraphPtr sub_graph_ptr = CreateGraphWithType(2);
  FillGraphNodeParaType(sub_graph_ptr);
  for (const ge::NodePtr &node : sub_graph_ptr->GetDirectNode()) {
    if (node->GetType() == "PlaceHolder" || node->GetType() == "Const") {
      continue;
    }
    if (node->GetType() == "SoftmaxV2") {
      std::vector<int64_t> workspace = {1};
      node->GetOpDesc()->SetWorkspaceBytes(workspace);
      node->GetOpDesc()->SetWorkspace(workspace);
    }
  }
  OpExtGenTaskRegistry::GetInstance().RegisterSKFunc("Conv2D", StubFunc);
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      node->GetOpDesc()->SetExtAttr("_sk_sub_graph", sub_graph_ptr);
      Status ret = super_kernel_builder->GenerateSuperKernelTask(*node, context, tasks);
      EXPECT_EQ(ret, SUCCESS);
    }
  }
  EXPECT_EQ(tasks.size(), 1);
  ge::GetThreadLocalContext().SetGraphOption(options_bk);
}

TEST_F(SuperkernelTaskBuilderUT, get_arg_format_v2_null) {
    domi::TaskDef task_def = {};
    std::string arg_format = "";
    ge::Status status = fe::GetArgFormatV2(task_def, arg_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, get_arg_format_v2) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_KERNEL);
    std::string args_format;

    ge::Status status = fe::GetArgFormatV2(task_def, args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, get_arg_format_v2_all_kernel) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_ALL_KERNEL);
    std::string args_format;

    ge::Status status = fe::GetArgFormatV2(task_def, args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, is_aicpu_task_def_fail) {
    domi::TaskDef task_def = {};
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    auto kernel_context = kernel_def->mutable_context();
    ge::Status status = fe::IsAICpuTaskDef(task_def, kernel_context);
    EXPECT_EQ(status, false);
}

TEST_F(SuperkernelTaskBuilderUT, get_unique_graph_id_for_node_fail) {
    ge::OpDescPtr super_kernel_op_desc = std::make_shared<ge::OpDesc>("A", "A");
    std::string str_id = fe::GetUniqueGraphIdForNode(super_kernel_op_desc);
    EXPECT_NE(str_id, " ");
}

TEST_F(SuperkernelTaskBuilderUT, kernel_launch_fail_1) {
    std::string stub_func = "test_kernel";
    domi::TaskDef task_def = {};
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    kernel_def = nullptr;
    
    void *args = (void *)malloc(16);
    bool status = fe::KernelLaunch(stub_func, 8, args, 16, nullptr, task_def);
    EXPECT_EQ(status, true);
}

TEST_F(SuperkernelTaskBuilderUT, kernel_launch_fail_2) {
    std::string stub_func = "test_kernel";
    domi::TaskDef task_def = {};
    task_def.set_type(RT_MODEL_TASK_KERNEL);
    task_def.set_stream_id(1);
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    kernel_def->set_kernel_name("A"); 
    
    void *args = (void *)malloc(16);
    bool status = fe::KernelLaunch(stub_func, 8, args, 16, nullptr, task_def);
    EXPECT_EQ(status, true);
}

TEST_F(SuperkernelTaskBuilderUT, check_dfx) {
    uint32_t args_size_workspace = 0;
    std::string super_kernel_args_format;
    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    uint32_t buffer_size = 32;

    ge::AttrUtils::SetInt(node->GetOpDesc(), "_op_dfx_buffer_size", buffer_size);

    ge::Status status1 = fe::GetWorkspacePattern(*node, super_kernel_args_format, 0);
    EXPECT_EQ(status1, ge::SUCCESS);

    ge::Status status = fe::GetWorkspacePattern(*node, super_kernel_args_format, 64);
    EXPECT_EQ(status, ge::SUCCESS);

    fe::CheckDFXOpen(args_size_workspace, *node, super_kernel_args_format, 0);
    EXPECT_EQ(args_size_workspace, 8);
}

TEST_F(SuperkernelTaskBuilderUT, get_spk_workspace) {
    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    std::vector<int64_t> super_ws = {32, 32};
    std::vector<int64_t> super_ws_bytes = {256, 256};
    node->GetOpDesc()->SetWorkspace(super_ws);
    node->GetOpDesc()->SetWorkspaceBytes(super_ws_bytes);

    int64_t ws = fe::GetSuperKernelWorkspace(*node);
    EXPECT_EQ(ws, 256);

}

TEST_F(SuperkernelTaskBuilderUT, get_spk_workspace_null) {
    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    std::vector<int64_t> super_ws = {32, 32};
    super_ws.clear();
    std::vector<int64_t> super_ws_bytes = {256, 256};
    node->GetOpDesc()->SetWorkspace(super_ws);
    node->GetOpDesc()->SetWorkspaceBytes(super_ws_bytes);

    int64_t ws = fe::GetSuperKernelWorkspace(*node);
    EXPECT_EQ(ws, 256);

}

TEST_F(SuperkernelTaskBuilderUT, set_taskdef_value) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_KERNEL);
    std::string args_format;
    ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("A", "A");

    ge::Status status = fe::FillTaskDefAfterGenTask(op_desc, task_def, args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, get_arg_format) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_PREPROCESS_KERNEL);
    auto kernel_def = task_def.mutable_kernel();
    kernel_def->set_block_dim(24);
    auto kernel_context = kernel_def->mutable_context();
    kernel_context->set_args_count(1);
    kernel_context->set_args_format("{ws0}");
    kernel_context->set_kernel_type(6); // KernelType对应为:ge::ccKernelType::AI_CPU

    std::string args_format;
    ge::OpDescPtr super_kernel_op_desc = std::make_shared<ge::OpDesc>("A", "A");

    std::string super_kernel_args_format;
    uint32_t args_size_total = 8;

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(task_def);
    std::vector<std::vector<domi::TaskDef>> sub_tasks;
    sub_tasks.emplace_back(tasks);

    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    const std::string subOpType = "SubKernel";

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    nodeOpDescPtr->SetId(0U);

    ge::OpDescPtr subNodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, subOpType);
    ge::Node * subNode = node.get();
    subNodeOpDescPtr->SetId(0U);
    std::vector<ge::Node *> sub_nodes;
    sub_nodes.push_back(subNode);


    ge::Status status = fe::GetArgFormat(sub_nodes, args_size_total, sub_tasks,
                     super_kernel_op_desc, tasks, node, super_kernel_args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, get_arg_format_2) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_PREPROCESS_KERNEL);
    auto kernel_def = task_def.mutable_kernel();
    kernel_def->set_block_dim(24);
    auto kernel_context = kernel_def->mutable_context();
    kernel_context->set_args_count(1);
    kernel_context->set_args_format("{ws0}");

    std::string args_format;
    ge::OpDescPtr super_kernel_op_desc = std::make_shared<ge::OpDesc>("A", "A");

    std::string super_kernel_args_format;
    uint32_t args_size_total = 8;

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(task_def);
    std::vector<std::vector<domi::TaskDef>> sub_tasks;
    sub_tasks.emplace_back(tasks);

    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    const std::string subOpType = "SubKernel";

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    nodeOpDescPtr->SetId(0U);

    ge::OpDescPtr subNodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, subOpType);
    ge::Node * subNode = node.get();
    subNodeOpDescPtr->SetId(0U);
    std::vector<ge::Node *> sub_nodes;
    sub_nodes.push_back(subNode);

    ge::Status status = fe::GetArgFormat(sub_nodes, args_size_total, sub_tasks,
                     super_kernel_op_desc, tasks, node, super_kernel_args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, get_arg_format_4) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_KERNEL);
    auto kernel_def = task_def.mutable_kernel();
    kernel_def->set_block_dim(24);
    auto kernel_context = kernel_def->mutable_context();
    kernel_context->set_args_count(1);
    kernel_context->set_args_format("{ws0}");

    std::string args_format;
    ge::OpDescPtr super_kernel_op_desc = std::make_shared<ge::OpDesc>("A", "A");

    std::string super_kernel_args_format;
    uint32_t args_size_total = 8;

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(task_def);
    std::vector<std::vector<domi::TaskDef>> sub_tasks;
    sub_tasks.emplace_back(tasks);

    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    const std::string subOpType = "SubKernel";

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    nodeOpDescPtr->SetId(0U);

    ge::OpDescPtr subNodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, subOpType);
    ge::Node * subNode = node.get();
    subNodeOpDescPtr->SetId(0U);
    std::vector<ge::Node *> sub_nodes;
    sub_nodes.push_back(subNode);


    ge::Status status = fe::GetArgFormat(sub_nodes, args_size_total, sub_tasks,
                     super_kernel_op_desc, tasks, node, super_kernel_args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}


TEST_F(SuperkernelTaskBuilderUT, get_arg_format_3) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_ALL_KERNEL);
    auto kernel_def = task_def.mutable_kernel();
    kernel_def->set_block_dim(24);
    auto kernel_context = kernel_def->mutable_context();
    kernel_context->set_args_count(1);
    kernel_context->set_args_format("{ws0}");

    std::string args_format;
    ge::OpDescPtr super_kernel_op_desc = std::make_shared<ge::OpDesc>("A", "A");

    std::string super_kernel_args_format;
    uint32_t args_size_total = 8;

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(task_def);
    std::vector<std::vector<domi::TaskDef>> sub_tasks;
    sub_tasks.emplace_back(tasks);

    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    const std::string subOpType = "SubKernel";

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    nodeOpDescPtr->SetId(0U);

    ge::OpDescPtr subNodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, subOpType);
    ge::Node * subNode = node.get();
    subNodeOpDescPtr->SetId(0U);
    std::vector<ge::Node *> sub_nodes;
    sub_nodes.push_back(subNode);


    ge::Status status = fe::GetArgFormat(sub_nodes, args_size_total, sub_tasks,
                     super_kernel_op_desc, tasks, node, super_kernel_args_format);
    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, set_arg_format_value) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_KERNEL);
    auto kernel_def = task_def.mutable_kernel();
    kernel_def->set_block_dim(24);
    auto kernel_context = kernel_def->mutable_context();
    kernel_context->set_args_count(1);
    kernel_context->set_args_format("{ws0}");

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(task_def);
    std::vector<std::vector<domi::TaskDef>> sub_tasks;
    sub_tasks.emplace_back(tasks);

    uint32_t args_size_total = 512;

    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    const std::string subOpType = "SubKernel";

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);
    nodeOpDescPtr->SetId(0U);

    ge::OpDescPtr subNodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, subOpType);
    ge::Node * subNode = node.get();
    subNodeOpDescPtr->SetId(0U);
    std::vector<ge::Node *> sub_nodes;
    sub_nodes.push_back(subNode);

    uint32_t args_size_workspace = 8;

    void *all_args_buff_total = (void *)malloc(args_size_total);

    ge::Status status = fe::SetArgFormatValue(args_size_workspace, sub_tasks,
                         sub_nodes, all_args_buff_total, args_size_total);

    EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(SuperkernelTaskBuilderUT, set_arg_format_value_1) {
    domi::TaskDef task_def{};
    task_def.set_type(RT_MODEL_TASK_KERNEL);
    auto kernel_def = task_def.mutable_kernel();
    kernel_def->set_block_dim(24);
    auto kernel_context = kernel_def->mutable_context();
    kernel_context->set_args_count(1);
    kernel_context->set_args_format("{ws0}");

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(task_def);
    std::vector<std::vector<domi::TaskDef>> sub_tasks;
    sub_tasks.emplace_back(tasks);

    uint32_t args_size_total = 512;

    const std::string graphName = "testSuperkernelGentaskProtoGraph";
    const std::string opDescName = "testSuperkernelGentaskProtoOpDesc";
    const std::string opType = "SuperKernel";
    const std::string subOpType = "SubKernel";

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(graphName);
    ge::OpDescPtr nodeOpDescPtr = std::make_shared<ge::OpDesc>(opDescName, opType);
    nodeOpDescPtr->SetId(0U);
    std::vector<uint32_t> sk_send_event_ids = {1};
    std::vector<uint32_t> sk_rcv_event_ids = {1};
    (void)ge::AttrUtils::SetListInt(nodeOpDescPtr, "_sk_send_event_ids", sk_send_event_ids);
    (void)ge::AttrUtils::SetListInt(nodeOpDescPtr, "_sk_rcv_event_ids", sk_rcv_event_ids);
    ge::NodePtr node = graph->AddNode(nodeOpDescPtr);

    ge::Node * subNode = node.get();
    std::vector<ge::Node *> sub_nodes;
    sub_nodes.push_back(subNode);

    uint32_t args_size_workspace = 8;

    void *all_args_buff_total = (void *)malloc(args_size_total);

    ge::Status status = fe::SetArgFormatValue(args_size_workspace, sub_tasks,
                         sub_nodes, all_args_buff_total, args_size_total);

    EXPECT_EQ(status, ge::SUCCESS);
}
}
