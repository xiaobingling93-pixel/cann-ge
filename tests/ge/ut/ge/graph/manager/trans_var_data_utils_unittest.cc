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
#include <memory>

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/trans_var_data_utils.h"
#include <framework/common/debug/log.h>
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "runtime/dev.h"
#include "graph/manager/mem_manager.h"
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/build/memory/var_mem_assign_util.h"
namespace ge {
class UtestTransVarDataTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  NodePtr AddNode(const std::string &name, const std::string &type,
                  std::initializer_list<std::string> input_names,
                  std::initializer_list<std::string> output_names,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace ut

NodePtr UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt);

ComputeGraphPtr BuildGraphTransVarData() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto addn = builder.AddNode("addn", "AddN", 2, 1);

  builder.AddDataEdge(data, 0, addn, 0);
  builder.AddDataEdge(const1, 0, addn, 1);
  return builder.GetGraph();
}

TEST_F(UtestTransVarDataTest, CopyVarData) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph");
  auto node_0 = UtAddNode(graph, "data1", VARIABLE, 1, 1);
  auto node_1 = UtAddNode(graph, "data2", VARIABLE, 1, 1);
  node_0->GetInDataAnchor(0)->LinkFrom(node_1->GetOutDataAnchor(0));
  std::vector<NodePtr> variable_nodes { node_0, node_1 };

  auto device_id = GetContext().DeviceId();
  uint64_t session_id = 1;
  EXPECT_EQ(TransVarDataUtils::CopyVarData(graph, variable_nodes, session_id, device_id), SUCCESS);

  ComputeGraphPtr graph2 = nullptr;
  EXPECT_EQ(TransVarDataUtils::CopyVarData(graph2, variable_nodes, session_id, device_id), FAILED);
}

TEST_F(UtestTransVarDataTest, TransAllVarData) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph");
  auto node_0 = UtAddNode(graph, "data1", VARIABLE, 1, 1);
  auto node_1 = UtAddNode(graph, "data2", VARIABLE, 1, 1);
  node_0->GetInDataAnchor(0)->LinkFrom(node_1->GetOutDataAnchor(0));
  std::vector<NodePtr> variable_nodes { node_0, node_1 };
  uint64_t session_id = 1;
  uint32_t graph_id = 1;
  uint32_t device_id = 1;
  EXPECT_EQ(TransVarDataUtils::TransAllVarData(variable_nodes, session_id, graph_id, device_id), INTERNAL_ERROR);
}


TEST_F(UtestTransVarDataTest, TransAllVarData_WithSkipNode) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph");

  auto node_0 = UtAddNode(graph, "data1", VARIABLE, 1, 1);
  auto node_1 = UtAddNode(graph, "data2", VARIABLE, 1, 1);
  node_0->GetInDataAnchor(0)->LinkFrom(node_1->GetOutDataAnchor(0));
  std::vector<NodePtr> variable_nodes { node_0, node_1 };

  uint64_t session_id = 1;
  uint32_t graph_id = 1;
  uint32_t device_id = 1;

  GeTensorDesc input_desc;

  std::vector<int64_t> shape = {1, 1, 224, 224};
  input_desc.SetShape(GeShape(shape));
  input_desc.SetFormat(FORMAT_NCHW);
  input_desc.SetDataType(DT_FLOAT);
  input_desc.SetOriginFormat(FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape(shape));
  input_desc.SetOriginDataType(DT_FLOAT);
  GeTensorDesc output_desc = input_desc;

  TransNodeInfo trans_node_info = {.node_type = TRANSDATA, .input = input_desc, .output = output_desc};
  TransNodeInfo transpose_node_info = {.node_type = TRANSPOSED, .input = input_desc, .output = output_desc};
  TransNodeInfo cast_node_info = {.node_type = CAST, .input = input_desc, .output = output_desc};
  TransNodeInfo reshape_node_info = {.node_type = RESHAPE, .input = input_desc, .output = output_desc};
  TransNodeInfo squeeze_node_info = {.node_type = SQUEEZEV2, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  fusion_road.emplace_back(transpose_node_info);
  fusion_road.emplace_back(cast_node_info);
  fusion_road.emplace_back(reshape_node_info);
  fusion_road.emplace_back(squeeze_node_info);
  VarManager::Instance(session_id)->Init(0, session_id, device_id, 1);
  VarManager::Instance(session_id)->SetTransRoad(node_0->GetName(), fusion_road);
  VarManager::Instance(session_id)->SetAllocatedGraphId(node_0->GetName(), 0);
  VarManager::Instance(session_id)->SetChangedGraphId(node_0->GetName(), graph_id);
  graph->SetSessionID(session_id);
  VarManager::Instance(session_id)->SetMemManager(&ge::MemManager::Instance());
  VarMemAssignUtil::AssignStaticMemory2Node(graph);
  EXPECT_EQ(TransVarDataUtils::TransAllVarData(variable_nodes, session_id, graph_id, device_id), INTERNAL_ERROR);
  VarManager::Instance(session_id)->FreeVarMemory();
}

TEST_F(UtestTransVarDataTest, CopyVarData_failed)
{
  ComputeGraphPtr graph = BuildGraphTransVarData();
  OpDescPtr op_desc = make_shared<OpDesc>("Variable", "Variable");
  GeTensorDesc dims_tensor_desc(GeShape({1,1,1,1}), FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dims_tensor_desc_in(GeShape({1,1,1,1}), FORMAT_NCHW, DT_FLOAT16);
  op_desc->AddInputDesc(dims_tensor_desc_in);
  op_desc->AddOutputDesc(dims_tensor_desc);

  NodePtr src_node = graph->AddNode(op_desc);
  std::vector<NodePtr> variable_nodes { src_node };
  (void)AttrUtils::SetStr(src_node->GetOpDesc(), "_copy_from_var_node", "addn");
  (void)AttrUtils::SetBool(src_node->GetOpDesc(), "_copy_value", false);

  auto device_id = GetContext().DeviceId();
  uint64_t session_id = 1;
  EXPECT_EQ(TransVarDataUtils::CopyVarData(graph, variable_nodes, session_id, device_id), FAILED);
}

TEST_F(UtestTransVarDataTest, TransAllVarData_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph");
  auto node_0 = UtAddNode(graph, "data1", VARIABLE, 1, 1);
  auto node_1 = UtAddNode(graph, "data2", VARIABLE, 1, 1);
  node_0->GetInDataAnchor(0)->LinkFrom(node_1->GetOutDataAnchor(0));
  std::vector<NodePtr> variable_nodes;
  variable_nodes.push_back(node_0);
  variable_nodes.push_back(node_1);
  uint64_t session_id = 1;
  uint32_t graph_id = 1;
  uint32_t device_id = 1;

  const char_t * const kEnvValue = "SET_TRANS_VAR_DATA";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  EXPECT_EQ(TransVarDataUtils::TransAllVarData(variable_nodes, session_id, graph_id, device_id), FAILED);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

} // namespace ge
