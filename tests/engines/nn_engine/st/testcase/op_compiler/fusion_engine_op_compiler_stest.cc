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
#define protected public
#define private public
#include "stdlib.h"
#include "graph_optimizer/op_compiler/op_compiler.h"
#include "graph_optimizer/op_compiler/op_compiler_normal.h"
#include "graph_optimizer/op_compiler/op_compiler_optune.h"
#include "graph_optimizer/op_compiler/op_format_tune.h"
#include "graph_optimizer/json_parser/tbe_json_parse.h"
#include "graph/utils/graph_utils.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "fusion_manager/fusion_manager.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/sgt_slice_type.h"
#include "common/graph_comm.h"
#include "common/scope_allocator.h"
#include "mmpa/src/mmpa_stub.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/fe_type_utils.h"
#include "graph/tuning_utils.h"
#include "graph/ge_local_context.h"
#undef private
#undef protected

using namespace testing;
using namespace fe;
using namespace ge;
using namespace te;

namespace {
std::map<uint64_t, te::TbeOpInfo> te_task_map;
bool SelectTbeOpFormatStub(const te::TbeOpInfo &tbe_op_info, std::string &format_str) {
  return true;
}
bool CheckTbeSupportedStub(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  return true;
}

bool PreBuildTbeOpStub(te::TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId) {
  std::string op_type;
  opinfo.GetOpType(op_type);
  string op_pattern = "ElemWise";
  if (op_type == "Conv2D") {
    op_pattern = "Convolution";
  }
  opinfo.SetPattern(op_pattern);
  te_task_map.emplace(taskId, opinfo);
  return true;
}
te::LX_QUERY_STATUS GetOpInfoStub(const te::TbeOpInfo &tbeOpInfo, std::string &result) {
  result = "qwer";
  return te::LX_QUERY_SUCC;
}
bool GetOpUniqueKeyStub(const te::TbeOpInfo& opinfo, std::vector<std::string> &opUniqueKeys) {
  std::string opUniqueKey;
  opinfo.GetOpType(opUniqueKey);
  opUniqueKeys.push_back(opUniqueKey);
  return true;
}
bool WaitAllFinishedStubFailed(uint64_t graphId, vector<te::FinComTask> &tasks) {
  return false;
}
bool WaitAllFinishedStub(uint64_t graphId, vector<te::FinComTask> &tasks) {
  std::string json_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/te_conv2d_compress.json";
  for (auto &item : te_task_map) {
    te::FinComTask task;
    task.taskId = item.first;
    std::string name;
    std::string op_type;
    item.second.GetName(name);
    item.second.GetOpType(op_type);
    task.status = fe::SUCCESS;
    task.teNodeOpDesc = std::make_shared<ge::OpDesc>(name, op_type);
    ge::AttrUtils::SetStr(task.teNodeOpDesc, "json_file_path", json_path);
    ge::AttrUtils::SetStr(task.teNodeOpDesc, COMPILE_INFO_JSON, "compile_info_json");
    ge::AttrUtils::SetStr(task.teNodeOpDesc, COMPILE_INFO_KEY, "compile_info_key");
    tasks.push_back(task);
  }
  te_task_map.clear();
  return true;
}
bool TeGeneralizeStub(const te::TbeOpInfo &tbeOpInfo, const te::TE_GENERALIZE_TYPE &generalizeType,
                      const ge::NodePtr &nodePtr) {
  auto op_desc = nodePtr->GetOpDesc();
  vector<int64_t> dims = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> new_shape_range = {{1, 16}, {2, 16}, {3, 16}, {4, 16}};
  GeShape shape(dims);
  for (auto input_desc : op_desc->GetAllInputsDescPtr()) {
    input_desc->SetOriginShape(shape);
    input_desc->SetOriginShapeRange(new_shape_range);
  }
  for (auto output_desc : op_desc->GetAllOutputsDescPtr()) {
    output_desc->SetOriginShape(shape);
    output_desc->SetOriginShapeRange(new_shape_range);
  }
  return true;
}
bool TeGeneralizeStub2(const te::TbeOpInfo &tbeOpInfo, const te::TE_GENERALIZE_TYPE &generalizeType,
                      const ge::NodePtr &nodePtr) {
  return false;
}
}

class STEST_fusion_engine_op_compiler : public testing::Test
{
protected:
  static void SetUpTestCase() {
    Configuration::Instance(AI_CORE_NAME).InitLibPath();
  }
  void SetUp()
  {
    tbe_adapter_ptr_ = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_adapter_ptr_->SelectTbeOpFormat = SelectTbeOpFormatStub;
    tbe_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedStub;
    tbe_adapter_ptr_->PreBuildTbeOp = PreBuildTbeOpStub;
    tbe_adapter_ptr_->GetOpInfo = GetOpInfoStub;
    tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedStub;
    tbe_adapter_ptr_->GetOpUniqueKeyFunc = GetOpUniqueKeyStub;
    tbe_adapter_ptr_->InitializeInnerHelp();
    ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(AI_CORE_NAME);

    FEOpsStoreInfo tbe_custom {
            2,
            "tbe-custom",
            EN_IMPL_CUSTOM_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
            true,
            true,
            true};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();

    std::map<std::string, std::string> options;
    ops_kernel_info_store_ptr_->Initialize(options);

    FusionRuleManagerPtr fusion_rule_mgr_ptr = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    FusionPriorityMgrPtr fusion_priority_mgr = std::make_shared<FusionPriorityManager>(AI_CORE_NAME, fusion_rule_mgr_ptr);
    lx_fusion_optimizer_ = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store_ptr_);
    lx_fusion_optimizer_->Initialize();

    graph_ = CreateTestGraph();
    graph_cce_ = CreateCceGraph();
    graph_mix_ = CreateMixGraph();
  }

  void TearDown()
  {
    te_task_map.clear();
  }

  static NodePtr CreateCceNode(string name, GeTensorDescPtr tensor_desc_ptr, ComputeGraphPtr graph)
  {
    OpDescPtr other_desc_ptr = std::make_shared<OpDesc>(name, "otherNode");
    //set OpDesc
    auto local_tensor_desc = tensor_desc_ptr->Clone();
    // add two input desc
    for (int i = 0; i < 2; ++i) {
      AttrUtils::SetStr(local_tensor_desc, "name", name + "In" + std::to_string(i));
      other_desc_ptr->AddInputDesc(local_tensor_desc);
    }
    // add two output desc
    for (int i = 0; i < 2; ++i) {
      AttrUtils::SetStr(local_tensor_desc, "name", name + "Out" + std::to_string(i));
      other_desc_ptr->AddOutputDesc(local_tensor_desc);
    }
    // add node from other_desc_ptr to graph
    // set attr
    AttrUtils::SetInt(other_desc_ptr, "T", DT_FLOAT);
    AttrUtils::SetInt(other_desc_ptr, "_fe_imply_type", EN_IMPL_HW_GENERAL_CCE);

    NodePtr node_other = graph->AddNode(other_desc_ptr);

    return node_other;
  }

  static NodePtr CreateOtherNode(string name, GeTensorDescPtr tensor_desc_ptr, ComputeGraphPtr graph)
  {
    OpDescPtr other_desc_ptr = std::make_shared<OpDesc>(name, "otherNode");
    //set OpDesc
    auto local_tensor_desc = tensor_desc_ptr->Clone();
    // add two input desc
    for (int i = 0; i < 2; ++i) {
      AttrUtils::SetStr(local_tensor_desc, "name", name + "In" + std::to_string(i));
      other_desc_ptr->AddInputDesc(local_tensor_desc);
    }
    // add two output desc
    for (int i = 0; i < 2; ++i) {
      AttrUtils::SetStr(local_tensor_desc, "name", name + "Out" + std::to_string(i));
      other_desc_ptr->AddOutputDesc(local_tensor_desc);
    }
    // add node from other_desc_ptr to graph
    // set attr
    AttrUtils::SetInt(other_desc_ptr, "T", DT_FLOAT);
    AttrUtils::SetInt(other_desc_ptr, "_fe_imply_type", EN_IMPL_CUSTOM_TBE);

    NodePtr node_other = graph->AddNode(other_desc_ptr);

    return node_other;
  }

  static ComputeGraphPtr CreateCceGraph()
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    // new a output GeTensorDesc
    GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
    general_ge_tensor_desc->SetFormat(FORMAT_NCHW);
    general_ge_tensor_desc->SetDataType(DT_FLOAT);

    int total_node_num = 4;
    vector<NodePtr> nodes;
    for (int i = 0; i < total_node_num; ++i) {
      nodes.push_back(CreateCceNode("test/other" + std::to_string(i), general_ge_tensor_desc, graph));
    }
    /* add link of anchors */
    std::vector<OutDataAnchorPtr> srcs;
    std::vector<InDataAnchorPtr> dsts;
    for (int i = 0; i < total_node_num - 1; ++i) {
      srcs.push_back(nodes[i]->GetOutDataAnchor(0));
      dsts.push_back(nodes[i + 1]->GetInDataAnchor(0));
      srcs.push_back(nodes[i]->GetOutDataAnchor(1));
      dsts.push_back(nodes[i + 1]->GetInDataAnchor(1));
    }

    // add edges
    for (int i = 0; i < srcs.size(); ++i)
    {
      GraphUtils::AddEdge(srcs[i], dsts[i]);
    }

    return graph;
  }

  static ComputeGraphPtr CreateMixGraph()
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    // new a output GeTensorDesc
    GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
    general_ge_tensor_desc->SetFormat(FORMAT_NCHW);
    general_ge_tensor_desc->SetDataType(DT_FLOAT);

    int total_node_num = 4;
    vector<NodePtr> nodes;
    for (int i = 0; i < 2; ++i) {
      nodes.push_back(CreateOtherNode("test/other" + std::to_string(i), general_ge_tensor_desc, graph));
    }
    for (int i = 2; i < total_node_num; ++i) {
      nodes.push_back(CreateCceNode("test/other" + std::to_string(i), general_ge_tensor_desc, graph));
    }
    /* add link of anchors */
    std::vector<OutDataAnchorPtr> srcs;
    std::vector<InDataAnchorPtr> dsts;
    for (int i = 0; i < total_node_num - 1; ++i) {
      srcs.push_back(nodes[i]->GetOutDataAnchor(0));
      dsts.push_back(nodes[i + 1]->GetInDataAnchor(0));
      srcs.push_back(nodes[i]->GetOutDataAnchor(1));
      dsts.push_back(nodes[i + 1]->GetInDataAnchor(1));
    }

    // add edges
    for (int i = 0; i < srcs.size(); ++i)
    {
      GraphUtils::AddEdge(srcs[i], dsts[i]);
    }

    return graph;
  }

  static ComputeGraphPtr CreateTestGraph()
  {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    // new a output GeTensorDesc
    GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
    general_ge_tensor_desc->SetFormat(FORMAT_NCHW);
    general_ge_tensor_desc->SetDataType(DT_FLOAT);

    int total_node_num = 4;
    vector<NodePtr> nodes;
    for (int i = 0; i < total_node_num; ++i) {
      nodes.push_back(CreateOtherNode("test/other" + std::to_string(i), general_ge_tensor_desc, graph));
    }
    /* add link of anchors */
    std::vector<OutDataAnchorPtr> srcs;
    std::vector<InDataAnchorPtr> dsts;
    for (int i = 0; i < total_node_num - 1; ++i) {
      srcs.push_back(nodes[i]->GetOutDataAnchor(0));
      dsts.push_back(nodes[i + 1]->GetInDataAnchor(0));
      srcs.push_back(nodes[i]->GetOutDataAnchor(1));
      dsts.push_back(nodes[i + 1]->GetInDataAnchor(1));
    }

    // add edges
    for (int i = 0; i < srcs.size(); ++i)
    {
      GraphUtils::AddEdge(srcs[i], dsts[i]);
    }

    return graph;
  }

  static ComputeGraphPtr BuildTestGraph(const int32_t &strategy) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", "Conv2D");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "RelU");
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "RelU");

    int64_t scope_id_1 = ScopeAllocator::Instance().AllocateScopeId();
    int64_t scope_id_2 = ScopeAllocator::Instance().AllocateScopeId();
    int64_t scope_id_3 = ScopeAllocator::Instance().AllocateScopeId();
    switch (strategy) {
      case 1:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_3);
        break;
      case 2:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);
        break;
      case 3:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);
        ScopeAllocator::SetL1ScopeAttr(conv1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu1, scope_id_3);
        break;
      case 4:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_3);
        break;
      case 5:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_1);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_1);
        break;
      default:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_3);
    }

    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);

    // add descriptor
    vector<int64_t> dim = {4, 4, 1, 4};
    GeShape shape(dim);
    GeTensorDesc tenosr_desc(shape);

    conv1->AddInputDesc(tenosr_desc);
    conv1->AddInputDesc(tenosr_desc);
    conv1->AddOutputDesc(tenosr_desc);

    conv2->AddInputDesc(tenosr_desc);
    conv2->AddInputDesc(tenosr_desc);
    conv2->AddOutputDesc(tenosr_desc);

    relu1->AddInputDesc(tenosr_desc);
    relu1->AddOutputDesc(tenosr_desc);
    relu2->AddInputDesc(tenosr_desc);
    relu2->AddOutputDesc(tenosr_desc);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);

    ge::AttrUtils::SetStr(conv1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(conv2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));

    return graph;
  }

  static ComputeGraphPtr BuildSomeGraph(const bool& is_dynamic, const int32_t &type) {
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", "Conv2D");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");

    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
    AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
    AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
    AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

    fe::ToOpStructPtr l1_info_ptr = std::make_shared<fe::ToOpStruct_t>();
    std::vector<int64_t> op_input_l1 = {1,2,3,4};
    vector<uint32_t> dynamic_input_start_idx = {0};
    vector<uint32_t> dynamic_input_end_idx = {1};
    switch (type) {
      case 1:
        l1_info_ptr->op_l1_fusion_type = {1};
        l1_info_ptr->op_l1_space = 2;
        l1_info_ptr->op_l1_workspace_flag = 3;
        l1_info_ptr->op_l1_workspace_size = 4;
        l1_info_ptr->slice_input_shape = {{5}};
        l1_info_ptr->slice_input_offset = {{6}};
        l1_info_ptr->slice_output_shape = {{7}};
        l1_info_ptr->slice_output_offset = {{8}};
        l1_info_ptr->total_shape = {9};
        l1_info_ptr->split_index = 0;
        conv1->SetExtAttr(ge::ATTR_NAME_L1_FUSION_EXTEND_PTR, l1_info_ptr);
        conv2->SetExtAttr(fe::ATTR_NAME_L2_FUSION_EXTEND_PTR, l1_info_ptr);
        break;
      case 2:
        ge::AttrUtils::SetListInt(relu1, ge::ATTR_NAME_OP_INPUT_L1_FLAG, op_input_l1);
        ge::AttrUtils::SetListInt(relu1, ge::ATTR_NAME_OP_INPUT_L1_ADDR, op_input_l1);
        ge::AttrUtils::SetListInt(relu1, ge::ATTR_NAME_OP_INPUT_L1_VALID_SIZE, op_input_l1);
        ge::AttrUtils::SetBool(relu2, NEED_RE_PRECOMPILE, true);
        break;
      case 3:
        ge::AttrUtils::SetStr(relu1, ge::ATTR_NAME_UNREGST_OPPATH, "unregst_oppath");
        ge::AttrUtils::SetStr(relu2, ge::ATTR_NAME_UNREGST_OPPATH, "unregst_oppath");
        break;
      case 4:
        ge::AttrUtils::SetStr(conv1, ge::ATTR_NAME_UNREGST_OPPATH, "unregst_oppath");
        ge::AttrUtils::SetStr(conv2, ge::ATTR_NAME_UNREGST_OPPATH, "unregst_oppath");
        ge::AttrUtils::SetListInt(conv1, ge::ATTR_NAME_DYNAMIC_INPUT_START, dynamic_input_start_idx);
        ge::AttrUtils::SetListInt(conv1, ge::ATTR_NAME_DYNAMIC_INPUT_END, dynamic_input_end_idx);
        break;
    }

    // add descriptor
    vector<int64_t> dim = {4, 4, 1, 4};
    if (is_dynamic) {
      dim[1] = -1;
    }
    GeShape shape(dim);
    GeTensorDesc tenosr_desc(shape);

    conv1->AddInputDesc(tenosr_desc);
    conv1->AddInputDesc(tenosr_desc);
    conv1->AddOutputDesc(tenosr_desc);

    conv2->AddInputDesc(tenosr_desc);
    conv2->AddInputDesc(tenosr_desc);
    conv2->AddOutputDesc(tenosr_desc);

    relu1->AddInputDesc(tenosr_desc);
    relu1->AddOutputDesc(tenosr_desc);
    relu2->AddInputDesc(tenosr_desc);
    relu2->AddOutputDesc(tenosr_desc);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);

    ge::AttrUtils::SetStr(conv1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(conv2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));

    return graph;
  }

  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  LxFusionOptimizerPtr lx_fusion_optimizer_;
  TbeOpStoreAdapterPtr tbe_adapter_ptr_;
  ComputeGraphPtr graph_;
  ComputeGraphPtr graph_cce_;
  ComputeGraphPtr graph_mix_;
};

TEST_F(STEST_fusion_engine_op_compiler, save_fusion_node_found)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  auto node = graph_->GetDirectNode().at(0);
  AttrUtils::SetInt(node->GetOpDesc(), "fusion_scope", 1);

  int64_t scope_id = 1;

  ScopeNodeIdMap fusion_node_map;
  std::vector<ge::Node*> fusion_nodes;
  fusion_node_map.emplace(std::make_pair(1, fusion_nodes));

  Status status = op_compiler_ptr->AddNodeToFusionMap(*node, scope_id, fusion_node_map);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(STEST_fusion_engine_op_compiler, save_fusion_node_not_found)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  auto node = graph_->GetDirectNode().at(0);
  AttrUtils::SetInt(node->GetOpDesc(), "fusion_scope", 1);

  int64_t scope_id = 1;

  ScopeNodeIdMap fusion_node_map;
  std::vector<ge::Node*> fusion_nodes;
  fusion_node_map.emplace(std::make_pair(2, fusion_nodes));

  Status status = op_compiler_ptr->AddNodeToFusionMap(*node, scope_id, fusion_node_map);

  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(STEST_fusion_engine_op_compiler, case_get_scope_node_map_suc)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  std::vector<ge::NodePtr> scope_nodes;
  for (auto &node : graph_->GetDirectNode()) {
  ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, true);
  scope_nodes.push_back(node);
  }
  ScopeNodeIdMap fusion_node_map;
  ge::AttrUtils::SetStr(graph_, ge::ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id_0");
  Status ret = op_compiler_ptr->GetScopeNodeMap(*graph_, scope_nodes, fusion_node_map);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, case_check_compile_condition_fail)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  auto node = graph_->GetDirectNode().at(0);
  ge::AttrUtils::SetBool(node->GetOpDesc(), kOpShapeOrRangeUnsupport, true);

  bool res = op_compiler_ptr->CheckCompileCondition(node);

  EXPECT_EQ(false, res);
}

TEST_F(STEST_fusion_engine_op_compiler, case_run_compile_process_no_need_compile)
{
  auto op_compiler_ptr = std::make_shared<OpCompilerNormal>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

  // add descriptor
  vector<int64_t> dims = {288, 32, 16, 16};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_Z);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc1);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NHWC);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_HWCN);
  out_desc2.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc2);
  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  OpImplType op_impl_type = EN_IMPL_CUSTOM_TBE;
  for(FEOpsStoreInfo &fe_op_store_info: Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_) {
    if (op_impl_type == fe_op_store_info.op_impl_type) {
      fe_op_store_info.need_compile = false;
    }
  }

  Status ret = op_compiler_ptr->RunCompileProcess(*graph);
  for(FEOpsStoreInfo &fe_op_store_info: Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_) {
    if (op_impl_type == fe_op_store_info.op_impl_type) {
      fe_op_store_info.need_compile = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, case_run_compile_process_failed1)
{
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedStubFailed;
  auto op_compiler_ptr = std::make_shared<OpCompilerBaseline>("baseline compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

  PlatformUtils::Instance().soc_version_ = "Ascend910B1";

  // add descriptor
  vector<int64_t> dims = {288, 32, 16, 16};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_Z);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc1);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NHWC);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_HWCN);
  out_desc2.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc2);
  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  ge::AttrUtils::SetStr(bn_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(relu_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>("test", nullptr);

  Status ret = op_compiler_ptr->Initialize();
  EXPECT_EQ(fe::SUCCESS, ret);

  // init pass mgr ptr
  fe_graph_optimizer_ptr->fusion_pass_mgr_ptr_ = std::make_shared<FusionPassManager>();
  ret = fe_graph_optimizer_ptr->fusion_pass_mgr_ptr_->Initialize(AI_CORE_NAME);
  EXPECT_EQ(fe::SUCCESS, ret);

  ret = op_compiler_ptr->RunCompileProcess(*graph);
  EXPECT_EQ(fe::FAILED, ret);

  ret = op_compiler_ptr->RunCompileProcess(*graph);
  EXPECT_EQ(fe::FAILED, ret);

  ret = op_compiler_ptr->Initialize();
  EXPECT_EQ(fe::SUCCESS, ret);
  FusionInfo info1{graph->GetGraphID(), std::to_string(graph->GetSessionID()), "TbeConvStridedwriteFusionPass", 8};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info1);
  FusionInfo info2{graph->GetGraphID(), std::to_string(graph->GetSessionID()), "MatmulConfusiontransposeUbFusion", 2};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info2);

  ret = op_compiler_ptr->RunCompileProcess(*graph);
  EXPECT_EQ(fe::FAILED, ret);
  std::string session_and_graph_id = std::to_string(graph->GetSessionID()) + "_" + std::to_string(graph->GetGraphID());
  auto iter = FusionStatisticRecorder::Instance().buffer_fusion_info_map_.find(session_and_graph_id);
  if (iter != FusionStatisticRecorder::Instance().buffer_fusion_info_map_.end()) {
    for (auto iter1 = iter->second.begin(); iter1 != iter->second.end(); iter1++) {
      if (iter1->second.GetPassName() == "TbeConvStridedwriteFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 8);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 0);
      }
      if (iter1->second.GetPassName() == "MatmulConfusiontransposeUbFusion") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 2);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 0);
      }
    }
  }

  ret = op_compiler_ptr->RunCompileProcess(*graph);
  EXPECT_EQ(fe::FAILED, ret);
  iter = FusionStatisticRecorder::Instance().buffer_fusion_info_map_.find(session_and_graph_id);
  if (iter != FusionStatisticRecorder::Instance().buffer_fusion_info_map_.end()) {
    for (auto iter1 = iter->second.begin(); iter1 != iter->second.end(); iter1++) {
      if (iter1->second.GetPassName() == "TbeConvStridedwriteFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 13);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 10);
      }
      if (iter1->second.GetPassName() == "MatmulConfusiontransposeUbFusion") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), -5);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 3);
      }
    }
  }
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedStub;
}

TEST_F(STEST_fusion_engine_op_compiler, has_compile_strategy_suc)
{
  auto op_compiler_ptr = std::make_shared<OpCompilerNormal>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
  ge::AttrUtils::SetStr(bn_op, "_op_compile_strategy", "NO_TUNE");
  ge::AttrUtils::SetStr(relu_op, "_op_compile_strategy", "NO_TUNE");
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  std::vector<ge::NodePtr> nodes_be_compiled;
  nodes_be_compiled.push_back(bn_node);
  nodes_be_compiled.push_back(relu_node);
  Status ret = op_compiler_ptr->HasCompileStrategy(nodes_be_compiled);
  EXPECT_EQ(true, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, has_compile_strategy_failed)
{
  auto op_compiler_ptr = std::make_shared<OpCompilerNormal>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  std::vector<ge::NodePtr> nodes_be_compiled;
  nodes_be_compiled.push_back(bn_node);
  nodes_be_compiled.push_back(relu_node);
  Status ret = op_compiler_ptr->HasCompileStrategy(nodes_be_compiled);
  EXPECT_EQ(false, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, case_run_compile_process_failed2)
{
  auto op_compiler_ptr = std::make_shared<OpCompilerBaseline>("baseline compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

  // add descriptor
  vector<int64_t> dims = {288, 32, 16, 16};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_Z);
  in_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddInputDesc("x", in_desc1);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NHWC);
  out_desc1.SetDataType(DT_FLOAT16);
  bn_op->AddOutputDesc("y", out_desc1);
  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NCHW);
  in_desc2.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc2);
  GeTensorDesc out_desc2(shape);
  out_desc2.SetFormat(FORMAT_HWCN);
  out_desc2.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc2);
  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_CUSTOM_TBE));
  (void)ge::AttrUtils::SetBool(bn_op, NEED_RE_PRECOMPILE, true);
  (void)ge::AttrUtils::SetBool(relu_op, NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetInt(bn_op, L1_SCOPE_ID_ATTR, 123);
  ge::AttrUtils::SetInt(relu_op, L1_SCOPE_ID_ATTR, 123);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();

  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_);
  Status ret = op_compiler_ptr->Initialize();
  EXPECT_EQ(fe::SUCCESS, ret);

  // init pass mgr ptr
  fe_graph_optimizer_ptr->fusion_pass_mgr_ptr_ = std::make_shared<FusionPassManager>();
  ret = fe_graph_optimizer_ptr->fusion_pass_mgr_ptr_->Initialize(AI_CORE_NAME);
  EXPECT_EQ(fe::SUCCESS, ret);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  LxFusionOptimizeResult opt_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  ret = op_compiler_ptr->ReCompileOpAfterLxFusion(*graph, compile_info, opt_ret);
  EXPECT_EQ(fe::FAILED, ret);
  vector<ge::NodePtr> nodes_be_re_compiled;
  vector<ge::NodePtr> all_nodes;
  ret = op_compiler_ptr->GetFusionScope(*graph, compile_info.fusion_nodes_map, nodes_be_re_compiled, all_nodes);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_thread_op_no_thread_scope_id_suc)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  bool sgt_flag = false;
  Status ret = op_compiler_ptr->PreCompileThreadOp(*graph_, sgt_flag);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_thread_op_has_thread_scope_id_suc)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  for (auto &node : graph_->GetDirectNode()) {
    AttrUtils::SetInt(node->GetOpDesc(), kThreadScopeId, 1);
    ffts::ThreadSliceMap subgraphInfo;
    vector<vector<vector<ffts::DimRange>>> inputTensorSlice;
    vector<vector<vector<int64_t>>> oriInputTensorSlice;
    for (size_t i = 0; i < 2; i++) {
      vector<int64_t> vec1 = {0, 288, 0, 32, 0, 16, 0, 16};
      vector<ffts::DimRange> vdr1;
      for (size_t j = 0; j < vec1.size() - 1;) {
        ffts::DimRange dr;
        dr.lower = vec1[j];
        dr.higher = vec1[j + 1];
        vdr1.push_back(dr);
        j = j + 2;
      }
      vector<vector<ffts::DimRange>> threadSlice;
      threadSlice.push_back(vdr1);
      threadSlice.push_back(vdr1);

      vector<int64_t> vec2 = {0, 3, 0, 3, 0, 512, 0, 512};
      vector<int64_t> vdr2;
      for (size_t j = 0; j < vec2.size() - 1;) {
        ffts::DimRange dr;
        dr.lower = vec2[j];
        dr.higher = vec2[j + 1];
        vdr2.push_back(dr.higher - dr.lower);
        j = j + 2;
      }
      vector<vector<int64_t>> oriThreadSlice;
      oriThreadSlice.push_back(vdr2);
      oriThreadSlice.push_back(vdr2);
      inputTensorSlice.push_back(threadSlice);
      oriInputTensorSlice.push_back(oriThreadSlice);
    }

    vector<vector<vector<ffts::DimRange>>> outputTensorSlice;
    vector<vector<vector<int64_t>>> oriOutputTensorSlice;
    for (size_t i = 0; i < 2; i++) {
      vector<int64_t> vec3 = {0, 1, 0, 32, 0, 14, 0, 14, 0, 16};
      vector<ffts::DimRange> vdr3;
      for (size_t j = 0; j < vec3.size() - 1;) {
        ffts::DimRange dr;
        dr.lower = vec3[j];
        dr.higher = vec3[j + 1];
        vdr3.push_back(dr);
        j = j + 2;
      }
      vector<vector<ffts::DimRange>> threadSlice;
      threadSlice.push_back(vdr3);
      threadSlice.push_back(vdr3);

      vector<int64_t> vec4 = {0, 1, 0, 32, 0, 14, 0, 14, 0, 16};
      vector<int64_t> vdr4;
      for (size_t j = 0; j < vec4.size() - 1;) {
        ffts::DimRange dr;
        dr.lower = vec4[j];
        dr.higher = vec4[j + 1];
        vdr4.push_back(dr.higher - dr.lower);
        j = j + 2;
      }
      vector<vector<int64_t>> oriThreadSlice;
      oriThreadSlice.push_back(vdr4);
      oriThreadSlice.push_back(vdr4);
      outputTensorSlice.push_back(threadSlice);
      oriOutputTensorSlice.push_back(oriThreadSlice);
    }

    subgraphInfo.input_tensor_slice = inputTensorSlice;
    subgraphInfo.ori_input_tensor_shape = oriInputTensorSlice;
    subgraphInfo.output_tensor_slice = outputTensorSlice;
    subgraphInfo.ori_output_tensor_shape = oriOutputTensorSlice;
    ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(subgraphInfo);
    node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
  }
  OpImplType op_impl_type = EN_IMPL_HW_TBE;
  for(FEOpsStoreInfo &fe_op_store_info: Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_) {
    if (op_impl_type == fe_op_store_info.op_impl_type) {
     fe_op_store_info.need_pre_compile = false;
    }
  }
  for (auto &node : graph_->GetDirectNode()) {
    ge::GeTensorDescPtr tensor_ptr = node->GetOpDesc()->MutableInputDesc(0);
    if (tensor_ptr == nullptr) {
      continue;
    }
    ge::GeShape& shape = tensor_ptr->MutableShape();
    vector<int64_t> dim1 = {-1, 1};
    GeShape shape1(dim1);
    shape = shape1;
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
    break;
  }
  bool sgt_flag = false;
  Status status = op_compiler_ptr->PreCompileThreadOp(*graph_, sgt_flag);
  for(FEOpsStoreInfo &fe_op_store_info: Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_) {
    if (op_impl_type == fe_op_store_info.op_impl_type) {
      fe_op_store_info.need_pre_compile = true;
    }
  }
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(STEST_fusion_engine_op_compiler, case_parse_tvm_json_to_set_attr_fail)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  string json_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_with_so.json";
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_INT8);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  op_desc->AddInputDesc("input", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  OpDescPtr op_desc_ptr = node->GetOpDesc();
  Status status = op_compiler_ptr->ParseTvmJsonToSetAttr(node, op_desc_ptr, json_file_path);
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(STEST_fusion_engine_op_compiler, parse_json_and_update_op_fail1)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_INT8);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  op_desc->AddInputDesc("input", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  OpDescPtr op_desc_ptr = node->GetOpDesc();
  CompileResultInfo com_ret_info(GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.json");
  std::vector<CompileResultInfo> compile_results = {com_ret_info};
  Status status = op_compiler_ptr->ParseJsonAndUpdateOp(node, op_desc_ptr, compile_results);
  EXPECT_EQ(fe::SUCCESS, status);
  
  std::map<std::string, std::string> geGraphOptions = {{"ge.buildMode", "tuning"}};
  ge::GetThreadLocalContext().SetGraphOption(geGraphOptions);
  status = op_compiler_ptr->ParseJsonAndUpdateOp(node, op_desc_ptr, compile_results);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(STEST_fusion_engine_op_compiler, parse_json_and_update_op_fail2)
{
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_INT8);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  op_desc->AddInputDesc("input", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  OpDescPtr op_desc_ptr = node->GetOpDesc();

  ffts::ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_scope_id = 1;
  ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);;
  op_desc_ptr->SetExtAttr("_sgt_json_info", tsmp_ptr);
  CompileResultInfo com_ret_info(GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.json");
  std::vector<CompileResultInfo> compile_results = {com_ret_info};
  Status status = op_compiler_ptr->ParseJsonAndUpdateOp(node, op_desc_ptr, compile_results);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_op_success) {
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->Initialize();
  EXPECT_EQ(fe::SUCCESS, status);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto op_type = "ProposalD";
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("test1", op_type);
  OpDescPtr op_desc2 = std::make_shared<OpDesc>("test2", op_type);
  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  op_desc1->AddInputDesc("cls_prob", out_desc);
  op_desc1->AddInputDesc("bbox_delta", out_desc);
  op_desc1->AddInputDesc("im_info", out_desc);
  op_desc1->AddInputDesc("rpn_bbox", out_desc);
  op_desc1->AddOutputDesc("rois", out_desc);
  op_desc1->AddOutputDesc("actual_rois_num", out_desc);

  op_desc2->AddInputDesc("cls_prob", out_desc);
  op_desc2->AddInputDesc("bbox_delta", out_desc);
  op_desc2->AddInputDesc("im_info", out_desc);
  op_desc2->AddInputDesc("rpn_bbox", out_desc);
  op_desc2->AddOutputDesc("rois", out_desc);
  auto proposal_node1 = graph->AddNode(op_desc1);
  auto proposal_node2 = graph->AddNode(op_desc2);

  OpKernelInfoPtr op_kernel_info = std::shared_ptr<OpKernelInfo>(new (std::nothrow) OpKernelInfo(op_type));
  InputOrOutputInfoPtr output_info_ptr = std::make_shared<InputOrOutputInfo>("rois");
  output_info_ptr->op_param_type_ = OpParamType::REQUIRED;
  op_kernel_info->output_infos_.push_back(output_info_ptr);
  InputOrOutputInfoPtr output_info_ptr1 = std::make_shared<InputOrOutputInfo>("actual_rois_num");
  output_info_ptr1->op_param_type_ = OpParamType::OPTIONAL;
  op_kernel_info->output_infos_.push_back(output_info_ptr1);
  Status res = op_compiler_ptr->SetMemoryTypeForOutput(proposal_node1, op_kernel_info);
  EXPECT_EQ(fe::SUCCESS, res);
  res = op_compiler_ptr->SetMemoryTypeForOutput(proposal_node2, op_kernel_info);
  EXPECT_EQ(fe::SUCCESS, res);

  for (const auto &node : graph->GetDirectNode()) {
    for (int i = 0; i != node->GetOpDesc()->GetAllOutputsDesc().size(); ++i) {
      bool res = IsMemoryEmpty(op_desc1->GetOutputDesc(i));
      if (i == 1) {
        EXPECT_EQ(true, res);
      } else {
        EXPECT_EQ(false, res);
      }
    }
    for (const auto &tensor_desc : node->GetOpDesc()->GetAllInputsDesc()) {
      EXPECT_EQ(false, IsMemoryEmpty(tensor_desc));
    }
  }
}

TEST_F(STEST_fusion_engine_op_compiler, ParseJsonAndCompressOp_success_1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_0 = graph->AddNode(op_desc_0);

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> nodes_be_compiled;
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status ret = op_compiler_ptr->ParseJsonAndCompressOp(*graph, compile_ret_map, nodes_be_compiled);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, ParseJsonAndCompressOp_success_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  ge::AttrUtils::SetInt(op_desc_0, SCOPE_ID_ATTR, -4);
  NodePtr node_0 = graph->AddNode(op_desc_0);

  map<string, string> options;
  options.emplace(ge::BUILD_STEP, ge::BUILD_STEP_AFTER_MERGE);
  ge::GetThreadLocalContext().SetGraphOption(options);
  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> nodes_be_compiled;
  nodes_be_compiled.push_back(node_0);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status ret = op_compiler_ptr->ParseJsonAndCompressOp(*graph, compile_ret_map, nodes_be_compiled);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, ParseJsonAndCompressOp_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddOutputDesc(out_desc);
  ge::AttrUtils::SetInt(op_desc_0, SCOPE_ID_ATTR, 4);
  NodePtr node_0 = graph->AddNode(op_desc_0);

  map<string, string> options;
  options.emplace(ge::BUILD_STEP, ge::BUILD_STEP_AFTER_MERGE);
  ge::GetThreadLocalContext().SetGraphOption(options);
  std::vector<ge::NodePtr> nodes_be_compiled;
  CompileResultInfo com_ret_info(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/op_compiler/json/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.json");
  std::vector<CompileResultInfo> compile_results = {com_ret_info};
  CompileResultMap compile_ret_map;
  compile_ret_map.emplace(4, compile_results);
  nodes_be_compiled.push_back(node_0);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status ret = op_compiler_ptr->ParseJsonAndCompressOp(*graph, compile_ret_map, nodes_be_compiled);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_fusion_engine_op_compiler, setcompressweightattr_03)
{
  vector<int64_t> dim1 = {1, 64, 56, 56};
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_INT8);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  vector<int64_t> dim2 = {256, 64, 1, 1};
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2);
  tensor_desc2.SetOriginDataType(DT_INT8);
  tensor_desc2.SetDataType(DT_INT8);
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetOriginShape(shape2);

  vector<int64_t> dim3 = {256};
  GeShape shape3(dim3);
  GeTensorDesc tensor_desc3(shape3);
  tensor_desc3.SetOriginDataType(DT_INT8);
  tensor_desc3.SetDataType(DT_INT8);
  tensor_desc3.SetFormat(FORMAT_NCHW);
  tensor_desc3.SetOriginFormat(FORMAT_NCHW);
  tensor_desc3.SetOriginShape(shape3);

  vector<int64_t> dim4 = {1, 256, 56, 56};
  GeShape shape4(dim4);
  GeTensorDesc tensor_desc4(shape4);
  tensor_desc4.SetOriginDataType(DT_INT8);
  tensor_desc4.SetDataType(DT_INT8);
  tensor_desc4.SetFormat(FORMAT_NCHW);
  tensor_desc4.SetOriginFormat(FORMAT_NCHW);
  tensor_desc4.SetOriginShape(shape4);

  OpDescPtr op_desc = std::make_shared<OpDesc>("conv2d2", "Conv2D");
  op_desc->AddInputDesc("input", tensor_desc1);
  op_desc->AddInputDesc("filter", tensor_desc2);
  op_desc->AddInputDesc("bias", tensor_desc3);
  op_desc->AddOutputDesc("out", tensor_desc4);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  PlatformUtils::Instance().soc_version_ = "Ascend910B1";

  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->SetCompressWeightAttr(node);
  EXPECT_EQ(fe::SUCCESS, status);
  bool has_fe_weight_attr = ge::AttrUtils::HasAttr(op_desc, ATTR_NAME_FE_WEIGHT_COMPRESS);
  EXPECT_EQ(has_fe_weight_attr, false);
}

TEST_F(STEST_fusion_engine_op_compiler, setcompressweightattr_04)
{
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_INT8);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("fc", "FullyConnection");
  op_desc->AddInputDesc("input", tensor_desc1);
  op_desc->AddInputDesc("filter", tensor_desc1);
  op_desc->AddInputDesc("bias", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->SetCompressWeightAttr(node);
  EXPECT_EQ(fe::SUCCESS, status);
  bool has_fe_weight_attr = ge::AttrUtils::HasAttr(op_desc, ATTR_NAME_FE_WEIGHT_COMPRESS);
  EXPECT_EQ(has_fe_weight_attr, true);
  bool fe_weight_compress = false;
  ge::AttrUtils::GetBool(op_desc, ATTR_NAME_FE_WEIGHT_COMPRESS, fe_weight_compress);
  EXPECT_EQ(fe_weight_compress, true);
}

TEST_F(STEST_fusion_engine_op_compiler, setcompressweightattr_05)
{
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_INT8);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  op_desc->AddInputDesc("input", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->SetCompressWeightAttr(node);
  EXPECT_EQ(fe::SUCCESS, status);
  bool has_fe_weight_attr = ge::AttrUtils::HasAttr(op_desc, ATTR_NAME_FE_WEIGHT_COMPRESS);
  EXPECT_EQ(has_fe_weight_attr, false);
}

TEST_F(STEST_fusion_engine_op_compiler, setcompressweightattr_06)
{
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_UNDEFINED);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("MatMulV2", "MatMulV2");
  op_desc->AddInputDesc("x", tensor_desc1);
  op_desc->AddInputDesc("y", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->SetCompressWeightAttr(node);
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(STEST_fusion_engine_op_compiler, setcompressweightattr_07)
{
  vector<int64_t> dim(4, 2);
  GeShape shape1(dim);
  GeTensorDesc tensor_desc1(shape1);
  tensor_desc1.SetOriginDataType(DT_MAX);
  tensor_desc1.SetDataType(DT_INT8);
  tensor_desc1.SetFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginShape(shape1);

  OpDescPtr op_desc = std::make_shared<OpDesc>("MatMulV2", "MatMulV2");
  op_desc->AddInputDesc("x", tensor_desc1);
  op_desc->AddInputDesc("y", tensor_desc1);
  op_desc->AddOutputDesc("out", tensor_desc1);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->SetCompressWeightAttr(node);
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(STEST_fusion_engine_op_compiler, getfusionscope_1)
{
  ComputeGraphPtr graph = BuildTestGraph(1);
  ScopeNodeIdMap fusion_nodes_map;
  vector<ge::NodePtr> nodes_be_compiled;
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->GetFusionScope(*graph, buff_fus_rollback_nodes, fusion_nodes_map, nodes_be_compiled);
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto iter = fusion_nodes_map.begin(); iter != fusion_nodes_map.end(); iter++) {
    if (iter->first == 3) {
      EXPECT_EQ(iter->second.size(), 4);
    }
  }
}

TEST_F(STEST_fusion_engine_op_compiler, getfusionscope_2)
{
  ComputeGraphPtr graph = BuildTestGraph(2);
  ScopeNodeIdMap fusion_nodes_map;
  vector<ge::NodePtr> nodes_be_compiled;
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->GetFusionScope(*graph, buff_fus_rollback_nodes, fusion_nodes_map, nodes_be_compiled);
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto iter = fusion_nodes_map.begin(); iter != fusion_nodes_map.end(); iter++) {
    if (iter->first == 1 || iter->first == 2) {
      EXPECT_EQ(iter->second.size(), 2);
    }
  }
}

TEST_F(STEST_fusion_engine_op_compiler, getfusionscope_3)
{
  ComputeGraphPtr graph = BuildTestGraph(3);
  ScopeNodeIdMap fusion_nodes_map;
  vector<ge::NodePtr> nodes_be_compiled;
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->GetFusionScope(*graph, buff_fus_rollback_nodes, fusion_nodes_map, nodes_be_compiled);
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto iter = fusion_nodes_map.begin(); iter != fusion_nodes_map.end(); iter++) {
    if (iter->first == 2 || iter->first == 3) {
      EXPECT_EQ(iter->second.size(), 2);
    }
  }
}

TEST_F(STEST_fusion_engine_op_compiler, getfusionscope_4)
{
  ComputeGraphPtr graph = BuildTestGraph(4);
  ScopeNodeIdMap fusion_nodes_map;
  vector<ge::NodePtr> nodes_be_compiled;
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->GetFusionScope(*graph, buff_fus_rollback_nodes, fusion_nodes_map, nodes_be_compiled);
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto iter = fusion_nodes_map.begin(); iter != fusion_nodes_map.end(); iter++) {
    if (iter->first == 1 || iter->first == 3) {
      EXPECT_EQ(iter->second.size(), 2);
    }
  }
}

TEST_F(STEST_fusion_engine_op_compiler, getfusionscope_5)
{
  ComputeGraphPtr graph = BuildTestGraph(5);
  ScopeNodeIdMap fusion_nodes_map;
  vector<ge::NodePtr> nodes_be_compiled;
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status status = op_compiler_ptr->GetFusionScope(*graph, buff_fus_rollback_nodes, fusion_nodes_map, nodes_be_compiled);
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(STEST_fusion_engine_op_compiler, change_buffer_optimize_1)
{
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  std::map<std::string, std::string> options;
  options.emplace("ge.bufferOptimize", "l1_optimize");
  std::map<std::string, std::string> new_options;
  tbe_adapter_ptr->ChangeBufferOptimize(options, new_options);
  EXPECT_EQ(new_options.size(), 0);
  options["ge.bufferOptimize"] = "l2_optimize";
  tbe_adapter_ptr->ChangeBufferOptimize(options, new_options);
  EXPECT_EQ(new_options.size(), 0);
}

TEST_F(STEST_fusion_engine_op_compiler, init_tbe_func)
{
  std::string tbe_so_path = GetCodeDir() + "/build/tests/engines/nn_engine/depends/te_fusion/libte_fusion_stub.so";
  std::string real_path = GetRealPath(tbe_so_path);
  PluginManagerPtr plugin_manager_ptr = std::make_shared<PluginManager>(tbe_so_path);
  plugin_manager_ptr->OpenPlugin(real_path);
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->StopCompileOpInTuningAndAfterUBMatchMode();
  Status ret = tbe_adapter_ptr->InitTbeFunctions(plugin_manager_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
  plugin_manager_ptr->CloseHandle();
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_1)
{
  tbe_adapter_ptr_->support_parallel_compile = false;
  ComputeGraphPtr graph = BuildSomeGraph(false, 0);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_2)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  ComputeGraphPtr graph = BuildSomeGraph(false, 0);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_3)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  ComputeGraphPtr graph = BuildSomeGraph(true, 0);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_4)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  ComputeGraphPtr graph = BuildSomeGraph(false, 1);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_5)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  ComputeGraphPtr graph = BuildSomeGraph(false, 2);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_6)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  ComputeGraphPtr graph = BuildSomeGraph(false, 3);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_7)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  ComputeGraphPtr graph = BuildSomeGraph(true, 4);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_8)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  tbe_adapter_ptr_->TeGeneralize = TeGeneralizeStub;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv3", "PadV3");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = graph->AddNode(dy_op);
  (void)ge::AttrUtils::SetBool(dy_op, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
  (void)ge::AttrUtils::SetBool(dy_op, kStaticToDynamicSoftSyncOp, true);
  (void)ge::AttrUtils::SetBool(dy_op, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(dy_op, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> exp_orishape_range = {{1, 16}, {2, 16}, {3, 16}, {4, 16}};
  std::vector<std::pair<int64_t, int64_t>> exp_shape_range = {{1, 16}, {1, 1}, {3, 16}, {4, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> orishape_range;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)dy_op->GetInputDescPtr(0)->GetOriginShapeRange(orishape_range);
  (void)dy_op->GetInputDescPtr(0)->GetShapeRange(shape_range);
  auto input_shape = dy_op->GetInputDesc(0).GetOriginShape().GetDims();
  EXPECT_EQ(input_shape, dims);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_9)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  tbe_adapter_ptr_->TeGeneralize = TeGeneralizeStub;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv4", "PadV4");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = graph->AddNode(dy_op);
  (void)ge::AttrUtils::SetBool(dy_op, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
  (void)ge::AttrUtils::SetBool(dy_op, kStaticToDynamicSoftSyncOp, true);
  (void)ge::AttrUtils::SetBool(dy_op, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(dy_op, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> exp_orishape_range = {{1, 16}, {2, 16}, {3, 16}, {4, 16}};
  std::vector<std::pair<int64_t, int64_t>> exp_shape_range = {{1, 16}, {1, 1}, {3, 16}, {4, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> orishape_range;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)dy_op->GetInputDescPtr(0)->GetOriginShapeRange(orishape_range);
  (void)dy_op->GetInputDescPtr(0)->GetShapeRange(shape_range);
  auto input_shape = dy_op->GetInputDesc(0).GetOriginShape().GetDims();
  EXPECT_EQ(input_shape, dims);
}

TEST_F(STEST_fusion_engine_op_compiler, pre_compile_case_10)
{
  tbe_adapter_ptr_->support_parallel_compile = true;
  tbe_adapter_ptr_->TeGeneralize = TeGeneralizeStub2;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr dy_op = std::make_shared<OpDesc>("padv4", "PadV4");
  vector<int64_t> dims = {1, 2, 3, 4};
  vector<int64_t> dims2 = {1, 1, 3, 4, 16};
  GeShape shape(dims);
  GeShape shape2(dims2);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NC1HWC0);
  in_desc1.SetOriginFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  in_desc1.SetOriginDataType(DT_FLOAT);
  in_desc1.SetShape(shape2);
  in_desc1.SetOriginShape(shape);
  dy_op->AddInputDesc("x", in_desc1);
  dy_op->AddOutputDesc("Y", in_desc1);
  NodePtr dy_node = graph->AddNode(dy_op);
  (void)ge::AttrUtils::SetBool(dy_op, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, true);
  (void)ge::AttrUtils::SetBool(dy_op, kStaticToDynamicSoftSyncOp, true);
  (void)ge::AttrUtils::SetBool(dy_op, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(dy_op, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->PreCompileOp(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
}

namespace {
void CheckOpCompileAttr(const ge::OpDescPtr &op_desc, const string &impl_switch, bool isDynamic = false) {
  bool dynamic_compile_static = false;
  bool has_dynamic_attr = ge::AttrUtils::GetBool(op_desc, kAttrDynamicCompileStatic, dynamic_compile_static);
  if (has_dynamic_attr) {
    EXPECT_EQ(dynamic_compile_static, isDynamic);
  }

  std::string op_impl_switch_str;
  bool has_op_impl_attr = ge::AttrUtils::GetStr(op_desc, kAttrOpImplSwitch, op_impl_switch_str);
  EXPECT_EQ(has_op_impl_attr, !impl_switch.empty());
  EXPECT_EQ(op_impl_switch_str, impl_switch);
}
}

TEST_F(STEST_fusion_engine_op_compiler, update_compile_params_1)
{
  map<string, string> options;
  options.emplace(ge::BUILD_MODE, ge::BUILD_MODE_TUNING);
  options.emplace(ge::BUILD_STEP, ge::BUILD_STEP_AFTER_BUILDER);
  ge::GetThreadLocalContext().SetGraphOption(options);
  
  // OpimplySwitch: dsl,tik; DynamicCompileStatic: tune
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("op1", "ScatterNdUpdate");
  op_desc1->AddInputDesc("var", tensor_desc);
  op_desc1->AddInputDesc("x", tensor_desc);
  op_desc1->AddInputDesc("y", tensor_desc);
  op_desc1->AddOutputDesc("var", tensor_desc);
  (void)ge::AttrUtils::SetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(op_desc1, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  // OpImplSwitch: dsl
  OpDescPtr op_desc2 = std::make_shared<OpDesc>("op2", "DynamicCompileStatic");
  op_desc2->AddInputDesc("x", tensor_desc);
  op_desc2->AddOutputDesc("y", tensor_desc);
  (void)ge::AttrUtils::SetBool(op_desc2, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(op_desc2, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  // OpimplySwitch: dsl,tik; DynamicCompileStatic: tune
  OpDescPtr op_desc3 = std::make_shared<OpDesc>("op3", "ScatterNdUpdate");
  op_desc3->AddInputDesc("var", tensor_desc);
  op_desc3->AddInputDesc("x", tensor_desc);
  op_desc3->AddInputDesc("y", tensor_desc);
  op_desc3->AddOutputDesc("var", tensor_desc);
  (void)ge::AttrUtils::SetInt(op_desc3, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  // IsUnKnownShapeOp
  OpDescPtr op_desc4 = std::make_shared<OpDesc>("op4", "Relu");
  vector<int64_t> dims_dy = {1, 2, -3, 4};
  GeShape shape_dy(dims_dy);
  GeTensorDesc tensor_desc_dy(shape_dy, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc_dy.SetOriginFormat(FORMAT_NCHW);
  tensor_desc_dy.SetOriginDataType(DT_FLOAT);
  tensor_desc_dy.SetOriginShape(shape_dy);
  op_desc4->AddInputDesc("x", tensor_desc_dy);
  op_desc4->AddOutputDesc("y", tensor_desc_dy);
  (void)ge::AttrUtils::SetBool(op_desc4, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(op_desc4, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc1);
  NodePtr node2 = graph->AddNode(op_desc2);
  NodePtr node3 = graph->AddNode(op_desc3);
  NodePtr node4 = graph->AddNode(op_desc4);

  auto op_compiler_ptr = std::make_shared<OpCompilerOpTune>("op tune compiler", AI_CORE_NAME, lx_fusion_optimizer_, nullptr);
  Status ret = op_compiler_ptr->UpdateCompileParams(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  CheckOpCompileAttr(op_desc1, "dsl, tik", true);
  CheckOpCompileAttr(op_desc2, "dsl", false);
  CheckOpCompileAttr(op_desc3, "dsl, tik", false);
  CheckOpCompileAttr(op_desc4, "", false);
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_);
  EXPECT_EQ(fe_graph_optimizer_ptr->CheckExportFusionResCond(*graph), false);
}

TEST_F(STEST_fusion_engine_op_compiler, update_compile_params_2)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);
  
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("op1", "ScatterNdUpdate");
  op_desc1->AddInputDesc("var", tensor_desc);
  op_desc1->AddInputDesc("x", tensor_desc);
  op_desc1->AddInputDesc("y", tensor_desc);
  op_desc1->AddOutputDesc("var", tensor_desc);
  (void)ge::AttrUtils::SetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetBool(op_desc1, kAttrDynamicCompileStatic, false);
  (void)ge::AttrUtils::SetStr(op_desc1, kAttrOpImplSwitch, "dsl");
  (void)ge::AttrUtils::SetInt(op_desc1, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc1);
  auto op_compiler_ptr = std::make_shared<OpCompilerNormal>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  Status ret = op_compiler_ptr->UpdateCompileParams(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool is_dynamic = false;
  bool has_attr = ge::AttrUtils::GetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic);
  EXPECT_EQ(has_attr, true);
  EXPECT_EQ(is_dynamic, false);
  string op_impl;
  has_attr = ge::AttrUtils::GetStr(op_desc1, kAttrOpImplSwitchValue, op_impl);
  EXPECT_EQ(has_attr, true);
  EXPECT_EQ(op_impl, "dsl");
}

TEST_F(STEST_fusion_engine_op_compiler, update_compile_params_3)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);
  
  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("op1", "ScatterNdUpdate");
  op_desc1->AddInputDesc("var", tensor_desc);
  op_desc1->AddInputDesc("x", tensor_desc);
  op_desc1->AddInputDesc("y", tensor_desc);
  op_desc1->AddOutputDesc("var", tensor_desc);
  (void)ge::AttrUtils::SetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(op_desc1, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc1);

  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status ret = op_compiler_ptr->UpdateCompileParams(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool is_dynamic = false;
  bool has_attr = ge::AttrUtils::GetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic);
  EXPECT_EQ(has_attr, true);
  EXPECT_EQ(is_dynamic, false);
  string op_impl;
  has_attr = ge::AttrUtils::GetStr(op_desc1, kAttrOpImplSwitchValue, op_impl);
  EXPECT_EQ(has_attr, true);
  EXPECT_EQ(op_impl, "dsl");
}

TEST_F(STEST_fusion_engine_op_compiler, update_compile_params_4)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("op1", "ScatterNdUpdate");
  op_desc1->AddInputDesc("var", tensor_desc);
  op_desc1->AddInputDesc("x", tensor_desc);
  op_desc1->AddInputDesc("y", tensor_desc);
  op_desc1->AddOutputDesc("var", tensor_desc);
  (void)ge::AttrUtils::SetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);
  (void)ge::AttrUtils::SetInt(op_desc1, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc1);
  setenv("MIN_COMPILE_RESOURCE_USAGE_CTRL", "ub_fusion,op_compile", 1);
  Configuration::Instance(AI_CORE_NAME).InitParamFromEnv();
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, nullptr);
  Status ret = op_compiler_ptr->UpdateCompileParams(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  bool is_dynamic = false;
  bool has_attr = ge::AttrUtils::GetBool(op_desc1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic);
  EXPECT_EQ(has_attr, true);
  EXPECT_EQ(is_dynamic, true);
  string op_impl;
  has_attr = ge::AttrUtils::GetStr(op_desc1, kAttrOpImplSwitchValue, op_impl);
  EXPECT_EQ(has_attr, false);
  EXPECT_EQ(op_impl, "");
}

TEST_F(STEST_fusion_engine_op_compiler, get_aoe_type_attr_from_rootgraph)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

   vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_HWCN);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("softmax", "SoftMax");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  vector<uint16_t> data_vec(16, 10);
  GeTensorPtr weight_ptr =
          std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), 16 * sizeof(uint16_t));
  ge::OpDescPtr const_op = ge::OpDescUtils::CreateConstOp(weight_ptr);
  NodePtr const_node = graph->AddNode(const_op);
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  std::string unsupport_reason;
  ops_kernel_info_store_ptr_->CheckSupported(node, unsupport_reason);

  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  ge::AttrUtils::SetStr(root_graph, "aoe_type", "op_format");
  graph->SetParentGraph(root_graph);

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  std::string aoe_type;
  auto ret = op_compiler_format_tune_ptr->SetTuneFormatReq(*graph, ops_kernel_info_store_ptr_);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, check_is_need_tuneformat_01)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

  vector<int64_t> dims = {1, 2, 3, 4};
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("op1", "ScatterNdUpdate");
  op_desc1->AddInputDesc("x", tensor_desc);
  op_desc1->AddOutputDesc("y", tensor_desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc1);

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  InputOrOutputInfoPtr input0_info_ptr = std::make_shared<InputOrOutputInfo>("input0");
  InputOrOutputInfoPtr output0_info_ptr = std::make_shared<InputOrOutputInfo>("output0");
  input0_info_ptr->op_tune_format_switch_ = true;
  output0_info_ptr->op_tune_format_switch_ = true;
  OpKernelInfoPtr op_kernel_info = std::shared_ptr<OpKernelInfo>(new (std::nothrow) OpKernelInfo("ScatterNdUpdate"));
  op_kernel_info->input_infos_.push_back(input0_info_ptr);
  op_kernel_info->output_infos_.push_back(output0_info_ptr);
  vector<int64_t> input_tuneformat_index_vec;
  vector<int64_t> output_tuneformat_index_vec;
  auto ret = op_compiler_format_tune_ptr->IsNeedTuneFormat(node, op_kernel_info,
                                                           input_tuneformat_index_vec, output_tuneformat_index_vec);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(input_tuneformat_index_vec.size(), 1);
  EXPECT_EQ(output_tuneformat_index_vec.size(), 1);
}

TEST_F(STEST_fusion_engine_op_compiler, check_is_need_tuneformat_02)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

  vector<int64_t> dims = {1, -1, 3, 4};
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT16);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  OpDescPtr op_desc1 = std::make_shared<OpDesc>("op1", "ScatterNdUpdate");
  op_desc1->AddInputDesc("x", tensor_desc);
  op_desc1->AddOutputDesc("y", tensor_desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc1);
  vector<int64_t> input_tuneformat_index_vec;
  vector<int64_t> output_tuneformat_index_vec;
  OpKernelInfoPtr op_kernel_info = std::shared_ptr<OpKernelInfo>(new (std::nothrow) OpKernelInfo("ScatterNdUpdate"));
  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  auto ret = op_compiler_format_tune_ptr->IsNeedTuneFormat(node, op_kernel_info,
                                                           input_tuneformat_index_vec, output_tuneformat_index_vec);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_fusion_engine_op_compiler, get_support_format_map_01)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

  vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginFormat(FORMAT_HWCN);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("softmax", "SoftMax");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  vector<uint16_t> data_vec(16, 10);
  GeTensorPtr weight_ptr =
          std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), 16 * sizeof(uint16_t));
  ge::OpDescPtr const_op = ge::OpDescUtils::CreateConstOp(weight_ptr);
  NodePtr const_node = graph->AddNode(const_op);
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  auto op_desc_ptr = node->GetOpDesc();
  std::string unsupport_reason;
  ops_kernel_info_store_ptr_->CheckSupported(node, unsupport_reason);
  int64_t op_impl_type = static_cast<int64_t>(EN_RESERVED);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, op_impl_type);
  OpImplType impl_type = static_cast<OpImplType>(op_impl_type);
  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  InputOrOutputInfoPtr input0_info_ptr = std::make_shared<InputOrOutputInfo>("input0");
  InputOrOutputInfoPtr output0_info_ptr = std::make_shared<InputOrOutputInfo>("output0");
  input0_info_ptr->unique_name_ = "input0.x";
  output0_info_ptr->unique_name_ = "output0.y";
  OpKernelInfoPtr op_kernel_info_ptr =
          OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(impl_type, op_desc_ptr->GetType());
  vector<int64_t> input_tuneformat_index_vec = {0};
  vector<int64_t> output_tuneformat_index_vec = {0};
  auto ret = op_compiler_format_tune_ptr->GetFormatSolutionSpace(node, op_kernel_info_ptr, ops_kernel_info_store_ptr_,
                                                                 input_tuneformat_index_vec, output_tuneformat_index_vec);
  vector<int32_t> matched_format_index_vec_check = {0, 3, 5};
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(op_compiler_format_tune_ptr->matched_format_index_vec, matched_format_index_vec_check);
  std::vector<int64_t> tuneformat_req_vec;
  std::vector<int64_t> tuneformat_req_vec_check = {2, 29, 4};
  ge::AttrUtils::GetListInt(op_desc_ptr->GetInputDescPtr(0), AOE_TUNEFORMAT_REQ, tuneformat_req_vec);
  EXPECT_EQ(tuneformat_req_vec, tuneformat_req_vec_check);
  ge::AttrUtils::GetListInt(op_desc_ptr->GetOutputDescPtr(0), AOE_TUNEFORMAT_REQ, tuneformat_req_vec);
  EXPECT_EQ(tuneformat_req_vec, tuneformat_req_vec_check);
}

TEST_F(STEST_fusion_engine_op_compiler, get_support_format_map_02)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

  vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginFormat(FORMAT_ND);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("softmax", "SoftMax");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  vector<uint16_t> data_vec(16, 10);
  GeTensorPtr weight_ptr =
          std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), 16 * sizeof(uint16_t));
  ge::OpDescPtr const_op = ge::OpDescUtils::CreateConstOp(weight_ptr);
  NodePtr const_node = graph->AddNode(const_op);
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  auto op_desc_ptr = node->GetOpDesc();
  std::string unsupport_reason;
  ops_kernel_info_store_ptr_->CheckSupported(node, unsupport_reason);
  int64_t op_impl_type = static_cast<int64_t>(EN_RESERVED);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, op_impl_type);
  OpImplType impl_type = static_cast<OpImplType>(op_impl_type);
  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  InputOrOutputInfoPtr input0_info_ptr = std::make_shared<InputOrOutputInfo>("input0");
  InputOrOutputInfoPtr output0_info_ptr = std::make_shared<InputOrOutputInfo>("output0");
  input0_info_ptr->unique_name_ = "input0.x";
  output0_info_ptr->unique_name_ = "output0.y";
  OpKernelInfoPtr op_kernel_info_ptr =
          OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(impl_type, op_desc_ptr->GetType());
  vector<int64_t> input_tuneformat_index_vec = {0};
  vector<int64_t> output_tuneformat_index_vec;
  auto ret = op_compiler_format_tune_ptr->GetFormatSolutionSpace(node, op_kernel_info_ptr, ops_kernel_info_store_ptr_,
                                                                 input_tuneformat_index_vec, output_tuneformat_index_vec);
  vector<int32_t> matched_format_index_vec_check = {0, 3};
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(op_compiler_format_tune_ptr->matched_format_index_vec, matched_format_index_vec_check);
  std::vector<int64_t> tuneformat_req_vec;
  std::vector<int64_t> tuneformat_req_vec_check = {2, 29};
  ge::AttrUtils::GetListInt(op_desc_ptr->GetInputDescPtr(0), AOE_TUNEFORMAT_REQ, tuneformat_req_vec);
  EXPECT_EQ(tuneformat_req_vec, tuneformat_req_vec_check);
  ge::AttrUtils::GetListInt(op_desc_ptr->GetOutputDescPtr(0), AOE_TUNEFORMAT_REQ, tuneformat_req_vec);
  EXPECT_EQ(tuneformat_req_vec, tuneformat_req_vec_check);
}

TEST_F(STEST_fusion_engine_op_compiler, get_support_format_map_03)
{
  map<string, string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);

  vector<int64_t> dims(4, 2);
  GeShape shape(dims);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginFormat(FORMAT_NHWC);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);

  OpDescPtr op_desc = std::make_shared<OpDesc>("softmax", "SoftMax");
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc);
  vector<uint16_t> data_vec(16, 10);
  GeTensorPtr weight_ptr =
          std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), 16 * sizeof(uint16_t));
  ge::OpDescPtr const_op = ge::OpDescUtils::CreateConstOp(weight_ptr);
  NodePtr const_node = graph->AddNode(const_op);
  GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  auto op_desc_ptr = node->GetOpDesc();
  std::string unsupport_reason;
  ops_kernel_info_store_ptr_->CheckSupported(node, unsupport_reason);
  int64_t op_impl_type = static_cast<int64_t>(EN_RESERVED);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, op_impl_type);
  OpImplType impl_type = static_cast<OpImplType>(op_impl_type);
  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  InputOrOutputInfoPtr input0_info_ptr = std::make_shared<InputOrOutputInfo>("input0");
  InputOrOutputInfoPtr output0_info_ptr = std::make_shared<InputOrOutputInfo>("output0");
  input0_info_ptr->unique_name_ = "input0.x";
  output0_info_ptr->unique_name_ = "output0.y";
  OpKernelInfoPtr op_kernel_info_ptr =
          OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(impl_type, op_desc_ptr->GetType());
  vector<int64_t> input_tuneformat_index_vec = {0};
  vector<int64_t> output_tuneformat_index_vec;
  auto ret = op_compiler_format_tune_ptr->GetFormatSolutionSpace(node, op_kernel_info_ptr, ops_kernel_info_store_ptr_,
                                                                 input_tuneformat_index_vec, output_tuneformat_index_vec);
  vector<int32_t> matched_format_index_vec_check = {0, 2, 3, 4};
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(op_compiler_format_tune_ptr->matched_format_index_vec, matched_format_index_vec_check);
  std::vector<int64_t> tuneformat_req_vec;
  std::vector<int64_t> tuneformat_req_vec_check = {2, 1, 29, 3};
  ge::AttrUtils::GetListInt(op_desc_ptr->GetInputDescPtr(0), AOE_TUNEFORMAT_REQ, tuneformat_req_vec);
  EXPECT_EQ(tuneformat_req_vec, tuneformat_req_vec_check);
  ge::AttrUtils::GetListInt(op_desc_ptr->GetOutputDescPtr(0), AOE_TUNEFORMAT_REQ, tuneformat_req_vec);
  EXPECT_EQ(tuneformat_req_vec, tuneformat_req_vec_check);
}

// can not get attr tuneformat
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_node_attr_001) {
  /*
   *
   *Graph will be like:
   *             a(NCHW)
   *                |
   *                |
   *                |
   *             aa(NCHW)
   *                |
   *                |
   *                |
   *             b(NCHW)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr a_desc = std::make_shared<OpDesc>("a", "B");
  OpDescPtr aa_desc = std::make_shared<OpDesc>("aa", "AA");
  OpDescPtr b_desc = std::make_shared<OpDesc>("b", "B");
  ge::GeShape shape({3,4});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  a_desc->AddOutputDesc(tensor_desc);
  b_desc->AddInputDesc(tensor_desc);
  aa_desc->AddInputDesc(tensor_desc);
  aa_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(a_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(b_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(aa_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  NodePtr a_node = graph->AddNode(a_desc);
  NodePtr aa_node = graph->AddNode(aa_desc);
  NodePtr b_node = graph->AddNode(b_desc);
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), aa_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(aa_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByNodeAttr(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  const std::vector<std::string> nodes_type_vec = {"B", "AA", "B"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
}

// get attr tuneformat invalid
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_node_attr_002) {
  /*
   *
   *Graph will be like:
   *             B(NCHW)
   *                |
   *                |
   *                |       "tuneformat":50
   *             AA(NCHW)
   *                |       "tuneformat":50
   *                |
   *                |
   *             B(NCHW)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr a_desc = std::make_shared<OpDesc>("a", "B");
  OpDescPtr aa_desc = std::make_shared<OpDesc>("aa", "AA");
  OpDescPtr b_desc = std::make_shared<OpDesc>("b", "B");
  ge::GeShape shape({3,4});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  a_desc->AddOutputDesc(tensor_desc);
  b_desc->AddInputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(tensor_desc, "tuneformat", 50);
  aa_desc->AddInputDesc(tensor_desc);
  aa_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(a_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(b_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(aa_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  NodePtr a_node = graph->AddNode(a_desc);
  NodePtr aa_node = graph->AddNode(aa_desc);
  NodePtr b_node = graph->AddNode(b_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), aa_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(aa_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByNodeAttr(*graph);
  EXPECT_EQ(ret, fe::FAILED);
  const std::vector<std::string> nodes_type_vec = {"B", "AA", "B"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
}

// get attr tuneformat equal to format
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_node_attr_003) {
  /*
   *
   *Graph will be like:
   *             B(NCHW)
   *                |
   *                |
   *                |       "tuneformat":0
   *             AA(NCHW)
   *                |       "tuneformat":0
   *                |
   *                |
   *             B(NCHW)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr a_desc = std::make_shared<OpDesc>("a", "B");
  OpDescPtr aa_desc = std::make_shared<OpDesc>("aa", "AA");
  OpDescPtr b_desc = std::make_shared<OpDesc>("b", "B");
  ge::GeShape shape({3,4});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  a_desc->AddOutputDesc(tensor_desc);
  b_desc->AddInputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(tensor_desc, "tuneformat", 0);
  aa_desc->AddInputDesc(tensor_desc);
  aa_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(a_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(b_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(aa_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  NodePtr a_node = graph->AddNode(a_desc);
  NodePtr aa_node = graph->AddNode(aa_desc);
  NodePtr b_node = graph->AddNode(b_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), aa_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(aa_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByNodeAttr(*graph);
  EXPECT_EQ(ret, fe::SUCCESS);
  const std::vector<std::string> nodes_type_vec = {"B", "AA", "B"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
}

// ND->TransData->NZ
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_node_attr_004) {
  /*
   *
   *Graph will be like:
   *          PlaceHolder(ND)                           PlaceHolder(ND) 
   *                |                                          |
   *                |                                          |
   *                |                                      TransData
   *                |                                          |
   *                |      "tuneformat":29                     |
   *              AA(ND)                        ->          AA(NZ)
   *                |      "tuneformat":29                     |
   *                |                                          |
   *                |                                      TransData
   *                |                                          |
   *                |                                          |
   *              End(ND)                                   End(ND)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr placeholder_desc = std::make_shared<OpDesc>("PlaceHolder", "PlaceHolder");
  OpDescPtr aa_desc = std::make_shared<OpDesc>("aa", "AA");
  OpDescPtr end_desc = std::make_shared<OpDesc>("End", "End");
  ge::GeShape shape({1,2,3,4});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  placeholder_desc->AddOutputDesc(tensor_desc);
  end_desc->AddInputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(tensor_desc, "tuneformat", 29);
  aa_desc->AddInputDesc(tensor_desc);
  aa_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(placeholder_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(end_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(aa_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  NodePtr placeholder_node = graph->AddNode(placeholder_desc);
  NodePtr aa_node = graph->AddNode(aa_desc);
  NodePtr end_node = graph->AddNode(end_desc);

  GraphUtils::AddEdge(placeholder_node->GetOutDataAnchor(0), aa_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(aa_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByNodeAttr(*graph);
  const std::vector<std::string> nodes_type_vec = {"PlaceHolder", "TransData", "AA", "TransData", "End"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
  EXPECT_EQ(ret, fe::SUCCESS);
}

// checksupported fail, return to original graph
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_node_attr_005) {
  /*
   *
   *Graph will be like:
   *          PlaceHolder(NZ)                           PlaceHolder(NZ)
   *                |                                          |
   *                |                                          |
   *                |                                          |
   *                |                                          |
   *                |      "tuneformat":3                      |
   *              AA(NZ)                        ->          AA(NZ)
   *                |      "tuneformat":3                      |
   *                |                                          |
   *                |                                          |
   *                |                                          |
   *                |                                          |
   *              End(NZ)                                   End(NZ)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr placeholder_desc = std::make_shared<OpDesc>("PlaceHolder", "PlaceHolder");
  OpDescPtr aa_desc = std::make_shared<OpDesc>("aa", "AA");
  OpDescPtr end_desc = std::make_shared<OpDesc>("End", "End");
  ge::GeShape shape({1,2,3,4,5,6});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  placeholder_desc->AddOutputDesc(tensor_desc);
  end_desc->AddInputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(tensor_desc, "tuneformat", 3);
  aa_desc->AddInputDesc(tensor_desc);
  aa_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(placeholder_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(end_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(aa_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  NodePtr placeholder_node = graph->AddNode(placeholder_desc);
  NodePtr aa_node = graph->AddNode(aa_desc);
  NodePtr end_node = graph->AddNode(end_desc);

  GraphUtils::AddEdge(placeholder_node->GetOutDataAnchor(0), aa_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(aa_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByNodeAttr(*graph);
  const std::vector<std::string> nodes_type_vec = {"PlaceHolder", "TransData", "AA", "End"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
  EXPECT_EQ(ret, fe::FAILED);
}

//cann kb result incomplete
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_cann_kb_result_001) {
  /*
   *
   *Graph will be like:
   *             B(NCHW)
   *                |
   *                |
   *                |
   *             BB(NCHW)
   *                |
   *                |
   *                |
   *             B(NCHW)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr a_desc = std::make_shared<OpDesc>("a", "B");
  OpDescPtr bb_desc = std::make_shared<OpDesc>("bb", "BB");
  OpDescPtr b_desc = std::make_shared<OpDesc>("b", "B");
  ge::GeShape shape({3,4});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  a_desc->AddOutputDesc(tensor_desc);
  b_desc->AddInputDesc(tensor_desc);
  bb_desc->AddInputDesc(tensor_desc);
  bb_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(a_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(b_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(bb_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  
  NodePtr a_node = graph->AddNode(a_desc);
  NodePtr bb_node = graph->AddNode(bb_desc);
  NodePtr b_node = graph->AddNode(b_desc);
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), bb_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(bb_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr =
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  bool need_re_precompile = false;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByCannKbResult(*graph, need_re_precompile);
  const std::vector<std::string> nodes_type_vec = {"B", "BB", "B"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
  EXPECT_EQ(ret, fe::FAILED);
}

//NCHW->TransposeD->NHWC
TEST_F(STEST_fusion_engine_op_compiler, update_tuneformat_by_cann_kb_result_002) {
  /*
   *
   *Graph will be like:
   *             B(NCHW)                                         B(NCHW)
   *                |                                               |
   *                |                                               |
   *                |                                           TransposeD
   *                |                                               |
   *                |       "tuneformat":1                          |
   *             AA(NCHW)                        ->             AA(NHWC)
   *                |       "tuneformat":1                          |
   *                |                                               |
   *                |                                           TransposeD
   *                |                                               |
   *                |                                               |
   *              B(NCHW)                                        B(NCHW)
   */
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr a_desc = std::make_shared<OpDesc>("a", "B");
  OpDescPtr aa_desc = std::make_shared<OpDesc>("aa", "AA");
  OpDescPtr b_desc = std::make_shared<OpDesc>("b", "B");
  ge::GeShape shape({3,4});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  a_desc->AddOutputDesc(tensor_desc);
  b_desc->AddInputDesc(tensor_desc);
  aa_desc->AddInputDesc(tensor_desc);
  aa_desc->AddOutputDesc(tensor_desc);
  (void)ge::AttrUtils::SetInt(a_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(b_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);
  (void)ge::AttrUtils::SetInt(aa_desc, FE_IMPLY_TYPE, fe::EN_IMPL_CUSTOM_TBE);

  NodePtr a_node = graph->AddNode(a_desc);
  NodePtr aa_node = graph->AddNode(aa_desc);
  NodePtr b_node = graph->AddNode(b_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), aa_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(aa_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));

  auto op_compiler_format_tune_ptr = 
      std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME);
  op_compiler_format_tune_ptr->engine_name_ = fe::AI_CORE_NAME;
  bool need_re_precompile = false;
  Status ret = op_compiler_format_tune_ptr->UpdateTuneFormatByCannKbResult(*graph, need_re_precompile);
  const std::vector<std::string> nodes_type_vec = {"Const", "Const", "B", "Transpose", "AA", "Transpose", "B"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), nodes_type_vec.at(index));
    index++;
  }
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_fusion_engine_op_compiler, op_compiler_coverage) {
  auto op_compiler_ptr = std::make_shared<OpCompiler>("normal compiler", AI_CORE_NAME, lx_fusion_optimizer_);
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  ge::GeShape shape_weight(dim_weight);
  ge::GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  ge::GeShape shape_weight1(dim_weight1);
  ge::GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.emplace_back(Node1);
  buff_fus_compile_failed_nodes.emplace_back(Node2);
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  EXPECT_EQ(op_compiler_ptr->ReCompileL1FusionFailedNodes(*graph, compile_info), fe::SUCCESS);
}
