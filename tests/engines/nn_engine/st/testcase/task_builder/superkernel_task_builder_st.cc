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

using namespace ge;
namespace fe {
class SuperkernelTaskBuilderST : public testing::Test {
protected:
  static void SetUpTestCase() {
    OpStoreAdapterManager::Instance(AI_CORE_NAME).Finalize();
    map<string, string> options;
    OpStoreAdapterManager::Instance(AI_CORE_NAME).Initialize(options);
    cout << "SuperkernelTaskBuilderST SetUp" << endl;
    fe::PlatformInfoManager::Instance().InitializePlatformInfo();
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
  }

  static void TearDownTestCase() {
    PlatformUtils::Instance().short_soc_version_ = "Ascend910B";
    cout << "SuperkernelTaskBuilderST TearDown" << endl;
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

TEST_F(SuperkernelTaskBuilderST, superkernel_plus_case1) {
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

TEST_F(SuperkernelTaskBuilderST, superkernel_plus_case2) {
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

TEST_F(SuperkernelTaskBuilderST, superkernel_plus_case3) {
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
  context.dataMemSize = 20480;
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

TEST_F(SuperkernelTaskBuilderST, superkernel_plus_reuse_binary_not_tiling_sink_failed) {
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
}
