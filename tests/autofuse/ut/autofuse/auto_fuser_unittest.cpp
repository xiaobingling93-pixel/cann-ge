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
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "autofuser.h"
#include "pattern_fusion/pad_slice_optimize_pass.h"
#include "operator_factory.h"
#include "utils/autofuse_utils.h"
#include "autofuse_frame/autofuse_frames.h"
#include "graph_utils.h"
#include "op_creator_register.h"

namespace ge {
class AutofuserTest : public testing::Test {
 public:
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
  }
  std::unique_ptr<es::Graph> es_graph_;

  static ComputeGraphPtr CreateGraphEleAndEle() {
    auto es_graph_ = std::make_unique<es::Graph>("graph");
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto relu = es::Relu(abs);
    relu.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(relu);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(exp, 1);
    auto graph = es_graph_->Build();
    return GraphUtilsEx::GetComputeGraph(*graph);
  }

  static ComputeGraphPtr CreateGraphEleAndRed() {
    auto es_graph_ = std::make_unique<es::Graph>("graph");
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto relu = es::Relu(abs);
    relu.SetSymbolShape({"s0", "s1", "s2"});
    auto sum = es::ReduceSumD(relu, {1}, true);
    sum.SetSymbolShape({"s0", "1", "s2"});
    auto abs1 = es::Abs(sum);
    abs1.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(exp, 1);
    auto graph = es_graph_->Build();
    return GraphUtilsEx::GetComputeGraph(*graph);
  }
};

class GraphBuilder {
 public:
  GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ComputeGraph>(name);
    node_type_ = node_type;
  }

  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    tensor_desc->SetShape(GeShape(std::move(shape)));
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    tensor_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscBackend");
    for (int i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  NodePtr AddNodeByIr(const std::string &op_name, const std::string &op_type) {
    auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
    if (op.IsEmpty()) {
      return nullptr;
    }
    OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    return graph_->AddNode(op_desc);
  }

  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
    GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
  }

  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

  static void AddSubgraph(const ComputeGraphPtr &graph, const string &call_name, const ComputeGraphPtr &subgraph) {
    const auto &call_node = graph->FindNode(call_name);
    if (call_node == nullptr) {
      return;
    }
    call_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    size_t index = call_node->GetOpDesc()->GetSubgraphInstanceNames().size();
    call_node->GetOpDesc()->AddSubgraphName(subgraph->GetName());
    call_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph->GetName());
    subgraph->SetParentNode(call_node);
    subgraph->SetParentGraph(graph);
    GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph);
  }

 private:
  ComputeGraphPtr graph_;
  std::string node_type_;
};

ComputeGraphPtr BuildGraphWithSubGraph() {
  auto root_builder = GraphBuilder("root");
  const auto &data0 = root_builder.AddNode("data0", "Data", 1, 1);
  const auto &case0 = root_builder.AddNode("case0", "Case", 1, 1);
  const auto &relu0 = root_builder.AddNode("relu0", "Relu", 1, 1);
  const auto &relu1 = root_builder.AddNode("relu1", "Relu", 1, 1);
  const auto &netoutput = root_builder.AddNode("netoutput", "NetOutput", 1, 1);
  const auto &root_graph = root_builder.GetGraph();
  root_builder.AddDataEdge(data0, 0, case0, 0);
  root_builder.AddDataEdge(case0, 0, relu0, 0);
  root_builder.AddDataEdge(relu0, 0, relu1, 0);
  root_builder.AddDataEdge(relu1, 0, netoutput, 0);

  auto sub_builder1 = GraphBuilder("sub1");
  const auto &data1 = sub_builder1.AddNode("data1", "Data", 0, 1);
  const auto &sub_graph1 = sub_builder1.GetGraph();
  root_graph->AddSubGraph(sub_graph1);
  sub_graph1->SetParentNode(case0);
  sub_graph1->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch1");
  case0->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");

  auto sub_builder2 = GraphBuilder("sub2");
  const auto &data2 = sub_builder2.AddNode("data2", "Data", 0, 1);
  const auto &sub_graph2 = sub_builder2.GetGraph();
  root_graph->AddSubGraph(sub_graph2);
  sub_graph2->SetParentNode(case0);
  sub_graph2->SetParentGraph(root_graph);
  case0->GetOpDesc()->AddSubgraphName("branch2");
  case0->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  root_graph->TopologicalSorting();
  return root_graph;
}

TEST_F(AutofuserTest, CreateOperatorOk) {
  auto cg = CreateGraphEleAndEle();

  // 校验AscBackend算子原型创建正常
  auto node_name = "fused_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber());
  auto asc_op = OperatorFactory::CreateOperator(node_name.c_str(), "AscBackend");
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", 2);
  asc_op.DynamicOutputRegister("outputs", 3);
  auto asc_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  ASSERT_NE(asc_desc, nullptr);

  AutofuseUtils::AddOperatorPrototypeAttrs(asc_desc);
  std::vector<std::string> input_name_list;
  std::vector<int64_t> input_type_list;
  std::vector<std::string> output_name_list;
  std::vector<int64_t> output_type_list;

  AttrUtils::GetListStr(asc_desc, "_input_name_list", input_name_list);
  AttrUtils::GetListInt(asc_desc, "_input_para_type_list", input_type_list);
  AttrUtils::GetListStr(asc_desc, "_output_name_list", output_name_list);
  AttrUtils::GetListInt(asc_desc, "_output_para_type_list", output_type_list);

  // 校验给算子原型添加属性正常
  ASSERT_EQ(input_name_list.size(), 2);
  ASSERT_EQ(input_type_list.size(), 2);
  ASSERT_EQ(output_name_list.size(), 3);
  ASSERT_EQ(output_name_list.size(), 3);

  // 校验AscGraph算子原型创建正常
  node_name = "fused_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber());
  asc_op = OperatorFactory::CreateOperator(node_name.c_str(), "AscGraph");
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", 2);
  asc_op.DynamicOutputRegister("outputs", 1);
  asc_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  ASSERT_NE(asc_desc, nullptr);
  auto new_node = cg->AddNode(asc_desc);
  ASSERT_NE(new_node, nullptr);

  // 校验FusedAscBackend算子原型创建正常
  node_name = "fused_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber());
  asc_op = OperatorFactory::CreateOperator(node_name.c_str(), "FusedAscBackend");
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", 2);
  asc_op.DynamicOutputRegister("outputs", 1);
  asc_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  ASSERT_NE(asc_desc, nullptr);
  new_node = cg->AddNode(asc_desc);
  ASSERT_NE(new_node, nullptr);
}

TEST_F(AutofuserTest, EleAndEleAutofuse) {
  auto cg = CreateGraphEleAndEle();
  AutofuserOptions options;
  Autofuser autofuser(options);
  ASSERT_EQ(autofuser.Fuse(cg), SUCCESS);
}

TEST_F(AutofuserTest, EleAndReduceAutofuse) {
  auto cg = CreateGraphEleAndRed();
  AutofuserOptions options;
  Autofuser autofuser(options);
  ASSERT_EQ(autofuser.Fuse(cg), SUCCESS);
}

TEST_F(AutofuserTest, EleAndReduceAutofuse1) {
  auto cg = CreateGraphEleAndRed();
  ASSERT_EQ(LoweringAndCanFuse(cg), SUCCESS);
}

TEST_F(AutofuserTest, EleAndReduceAutofuse2) {
  auto cg = CreateGraphEleAndRed();
  ASSERT_EQ(LoweringAndCanFuseWithCounter(cg, nullptr), SUCCESS);
}

TEST_F(AutofuserTest, SubGraphOk) {
  auto cg = BuildGraphWithSubGraph();
  AutofuserOptions options;
  Autofuser autofuser(options);
  ASSERT_EQ(autofuser.Fuse(cg), SUCCESS);
  for (const auto &subgraph : cg->GetAllSubgraphs()) {
    ASSERT_EQ(autofuser.Fuse(subgraph), SUCCESS);
  }
}

TEST_F(AutofuserTest, CastCastRemoveOk) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1"});
    auto cast1 = es::Cast(data, DT_FLOAT);
    cast1.SetSymbolShape({"s0", "s1"});
    auto cast2 = es::Cast(cast1, DT_FLOAT);
    cast2.SetSymbolShape({"s0", "s1"});
    auto abs = es::Abs(cast2);
    abs.SetSymbolShape({"s0", "s1"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cast1 = cg->FindNode("Cast_0");
  ASSERT_NE(cast1, nullptr);
  auto cast_desc1 = cast1->GetOpDesc()->MutableOutputDesc(0);
  cast_desc1->SetDataType(DT_FLOAT);
  cast_desc1->SetOriginDataType(DT_FLOAT);

  auto cast2 = cg->FindNode("Cast_1");
  ASSERT_NE(cast2, nullptr);
  auto cast_desc2 = cast2->GetOpDesc()->MutableOutputDesc(0);
  cast_desc2->SetDataType(DT_FLOAT);
  cast_desc2->SetOriginDataType(DT_FLOAT);

  auto abs3 = cg->FindNode("Abs_2");
  ASSERT_NE(abs3, nullptr);
  auto abs_desc1 = abs3->GetOpDesc()->MutableOutputDesc(0);
  abs_desc1->SetDataType(DT_FLOAT);
  abs_desc1->SetOriginDataType(DT_FLOAT);
  auto abs_desc2 = abs3->GetOpDesc()->MutableInputDesc(0);
  abs_desc2->SetDataType(DT_FLOAT);
  abs_desc2->SetOriginDataType(DT_FLOAT);

  auto abs4 = cg->FindNode("Abs_3");
  ASSERT_NE(abs4, nullptr);
  auto abs1_desc1 = abs4->GetOpDesc()->MutableOutputDesc(0);
  abs1_desc1->SetDataType(DT_FLOAT);
  abs1_desc1->SetOriginDataType(DT_FLOAT);
  auto abs1_desc2 = abs4->GetOpDesc()->MutableInputDesc(0);
  abs1_desc2->SetDataType(DT_FLOAT);
  abs1_desc2->SetOriginDataType(DT_FLOAT);
  AutofuserOptions options;
  Autofuser autofuser(options);
  ASSERT_EQ(autofuser.Fuse(cg), SUCCESS);
  auto after_autofuse_cast1 = cg->FindNode("Cast_0");
  ASSERT_EQ(after_autofuse_cast1, nullptr);
}
}  // namespace ge
