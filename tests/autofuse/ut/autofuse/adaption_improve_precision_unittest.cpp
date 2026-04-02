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
#include "graph/utils/graph_utils_ex.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "ascir_ops.h"
#include "utils/autofuse_utils.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "post_process/scheduler_adapter/adaption_fallback_scalar.h"
#include "post_process/asc_backend_post_processor.h"
#include "utils/autofuse_attrs.h"
#include "ascgen_log.h"
#include "attribute_group/attr_group_shape_env.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"

namespace ge {
using namespace autofuse;
class AdaptionImprovePrecisionTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::string work_path;
};

namespace {
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

  struct AscGraphComponents {
    Symbol ONE;
    Expression A, B, C, D, E;
    Axis a, b, c, d, e;
  };

  AscGraphComponents CreateAscGraphSymbolsAndAxes(ge::AscGraph &graph) {
    AscGraphComponents components;
    components.ONE = Symbol(1);
    components.A = graph.CreateSizeVar("A");
    components.B = graph.CreateSizeVar("B");
    components.C = graph.CreateSizeVar("C");
    components.D = graph.CreateSizeVar("D");
    components.E = graph.CreateSizeVar("E");

    components.a = graph.CreateAxis("A", components.A);
    components.b = graph.CreateAxis("B", components.B);
    components.c = graph.CreateAxis("C", components.C);
    components.d = graph.CreateAxis("D", components.D);
    components.e = graph.CreateAxis("E", components.E);

    return components;
  }

    std::shared_ptr<ge::ascir_op::Data> CreateDataOp(const std::string &name, ge::AscGraph &graph,
                                      const AscGraphComponents &components, bool set_loop_axis = true) {
    auto data = std::make_shared<ge::ascir_op::Data>(name.c_str(), graph);
    data->attr.sched.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    if (set_loop_axis) {
      data->attr.sched.loop_axis = components.c.id;
    }
    data->y.dtype = DT_FLOAT16;
    *data->y.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    *data->y.repeats = {components.A, components.B, components.C, components.D, components.E};
    *data->y.strides = {components.B * components.C * components.D * components.E,
                       components.C * components.D * components.E, components.D * components.E, components.E,
                       components.ONE};
    return data;
  }

  std::shared_ptr<ge::ascir_op::Load> CreateLoadOp(const std::string &name, const ge::AscOpOutput &data,
                                    const AscGraphComponents &components) {
    auto load = std::make_shared<ge::ascir_op::Load>(name.c_str());
    load->x = data;
    load->attr.sched.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    load->y.dtype = DT_FLOAT16;
    *load->y.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    *load->y.repeats = {components.A, components.B, components.C, components.D, components.E};
    *load->y.strides = {components.B * components.C * components.D * components.E,
                       components.C * components.D * components.E, components.D * components.E,
                       components.E, components.ONE};
    return load;
  }

  std::shared_ptr<ge::ascir_op::Add> CreateAddOp(const std::string &name, const ge::AscOpOutput &x1,
                                   const ge::AscOpOutput &x2,
                                   const AscGraphComponents &components) {
    auto add = std::make_shared<ge::ascir_op::Add>(name.c_str());
    add->x1 = x1;
    add->x2 = x2;
    add->attr.sched.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    add->y.dtype = DT_FLOAT16;
    AscOutputAttrDataType output_data_type(add.get(), 0);
    output_data_type = ge::DT_FLOAT16;
    *add->y.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    *add->y.repeats = {components.A, components.B, components.C, components.D, components.E};
    *add->y.strides = {components.B * components.C * components.D * components.E,
                       components.C * components.D * components.E, components.D * components.E, components.E,
                       components.ONE};
    return add;
  }

  std::shared_ptr<ge::ascir_op::Store> CreateStoreOp(const std::string &name, const ge::AscOpOutput &input,
                                      const AscGraphComponents &components) {
    auto store = std::make_shared<ge::ascir_op::Store>(name.c_str());
    store->x = input;
    store->attr.sched.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    store->attr.sched.loop_axis = components.c.id;
    store->y.dtype = DT_FLOAT16;
    *store->y.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    *store->y.repeats = {components.A, components.B, components.C, components.D, components.E};
    *store->y.strides = {components.B * components.C * components.D * components.E,
                         components.C * components.D * components.E, components.D * components.E,
                         components.E, components.ONE};
    return store;
  }

  std::shared_ptr<ge::ascir_op::Output> CreateOutputOp(const std::string &name, const ge::AscOpOutput &input,
                                         const AscGraphComponents &components) {
    auto output = std::make_shared<ge::ascir_op::Output>(name.c_str());
    output->x = input;
    output->attr.sched.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    output->attr.sched.loop_axis = components.c.id;
    output->y.dtype = DT_FLOAT16;
    *output->y.axis = {components.a.id, components.b.id, components.c.id, components.d.id, components.e.id};
    *output->y.repeats = {components.A, components.B, components.C, components.D, components.E};
    *output->y.strides = {components.B * components.C * components.D * components.E,
                          components.C * components.D * components.E, components.D * components.E,
                          components.E, components.ONE};
    return output;
  }

  std::shared_ptr<AscGraph> CreatAddAscGraph(ge::AscGraph &graph) {
    auto components = CreateAscGraphSymbolsAndAxes(graph);

    auto x1 = CreateDataOp("x1_1", graph, components);
    auto x1Local = CreateLoadOp("x1Local_2", x1->y, components);
    auto x2 = CreateDataOp("x2_3", graph, components);
    auto x2Local = CreateLoadOp("x2Local_4", x2->y, components);
    auto add = CreateAddOp("add_4", x1Local->y, x2Local->y, components);
    auto x_out = CreateStoreOp("x_out_5", add->y, components);
    auto x_output = CreateOutputOp("x_output", x_out->y, components);

    auto x_out_node = graph.FindNode("x_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  std::shared_ptr<AscGraph> CreatAddAscGraphWithUnsupportedDtype(ge::AscGraph &graph) {
    auto components = CreateAscGraphSymbolsAndAxes(graph);

    auto x1 = CreateDataOp("x1_1", graph, components);
    auto x1Local = CreateLoadOp("x1Local_2", x1->y, components);
    auto x2 = CreateDataOp("x2_3", graph, components);
    auto x2Local = CreateLoadOp("x2Local_4", x2->y, components);
    auto add = CreateAddOp("add_4", x1Local->y, x2Local->y, components);

    auto x_out = CreateStoreOp("x_out_5", add->y, components);
    auto x_output = CreateOutputOp("x_output", x_out->y, components);

    auto add_node = graph.FindNode("add_4");
    if (add_node != nullptr) {
      GeTensorDescPtr output_tensor_desc;
      asc_adapt::GetOutputTensorDesc(add_node, output_tensor_desc);
      output_tensor_desc->SetDataType(DT_INT8);
    }

    auto x_out_node = graph.FindNode("x_output");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  /**
   *      netoutput1
   *         |
   *       shapeNo1
   *        |
   *      addnYes1
   *     /    \.
   *   /       \.
   * const1   const2
   */
  ComputeGraphPtr BuildGraph1(const std::string node_type = "") {
    auto builder = GraphBuilder("test", node_type);
    auto const1 = builder.AddNode("const1", "CONSTANT", 0, 1);
    auto const2 = builder.AddNode("const2", "CONSTANT", 0, 1);
    auto addn1 = builder.AddNode("addn1", "AddNYes", 2, 1);
    auto shape1 = builder.AddNode("shape1", "ShapeNo", 1, 1);
    auto netoutput1 = builder.AddNode("netoutput", "NETOUTPUT", 1, 0);

    builder.AddDataEdge(const1, 0, addn1, 0);
    builder.AddDataEdge(const2, 0, addn1, 1);
    builder.AddDataEdge(addn1, 0, shape1, 0);
    builder.AddDataEdge(shape1, 0, netoutput1, 0);

    return builder.GetGraph();
  }
}

// 测试all黑名单：所有节点都通过数据类型检查，不升精度
TEST_F(AdaptionImprovePrecisionTest, ImprovePrecision_BlackListAll_Pass) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackListAllPass");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  
  // 设置all黑名单
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("all");
  
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 验证：当黑名单为all且所有节点都通过数据类型检查时，不应该插入Cast节点
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  // 所有节点都通过数据类型检查，不应该插入Cast节点
  EXPECT_EQ(cast_cnt, 0U);
  
  // 清理黑名单
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();
}

// 测试all黑名单：有节点必须升精度（数据类型检查失败），忽略all开关
TEST_F(AdaptionImprovePrecisionTest, ImprovePrecision_BlackListAll_Fail) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackListAllFail");
  attr1->SetAscGraph(CreatAddAscGraphWithUnsupportedDtype(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  
  // 设置all黑名单
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("all");
  
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 验证：当黑名单为all且有节点数据类型检查失败时，应该插入Cast节点升精度
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  // 有节点数据类型检查失败，应该插入Cast节点升精度
  EXPECT_GT(cast_cnt, 0U);
  
  // 清理黑名单
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();
}

// 测试混合场景：all黑名单与特定算子黑名单共存
TEST_F(AdaptionImprovePrecisionTest, ImprovePrecision_BlackListAll_WithSpecific) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);

  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto op_desc1 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  std::vector<std::pair<std::string, DataType>> names = {{"const1", DT_FLOAT16},
                                                        {"const2", DT_FLOAT16},
                                                        {"shape1", DT_FLOAT16},
                                                        {"netoutput", DT_FLOAT}};
  for (auto name : names) {
    auto node = compute_graph->FindNode(name.first);
    ASSERT_NE(node, nullptr);
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    op_desc->SetType("NotAscBc");
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    ASSERT_NE(attr, nullptr);
  }

  ge::AscGraph add_graph1("CreatAscGraphWithImprovePrecisionBlackListMixed");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(asc_adapt::GeFallback(compute_graph), SUCCESS);
  
  // 设置all黑名单和特定算子黑名单
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("all");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Add");
  
  auto shape_env_attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(compute_graph), SUCCESS);

  // 验证：all优先级最高，应该对所有节点检查数据类型
  size_t cast_cnt = 0U;
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    GeTensorDescPtr output_tensor_desc;
    ASSERT_EQ(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc), SUCCESS);
    if (node->GetType() == "Cast") {
      cast_cnt++;
    }
  }
  // 所有节点都通过数据类型检查，不应该插入Cast节点
  EXPECT_EQ(cast_cnt, 0U);
  
  // 清理黑名单
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();
}
}  // namespace ge
