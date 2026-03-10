
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <chrono>
#include <gtest/gtest.h>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "lowering/lowerings.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "utils/autofuse_attrs.h"
#include "util/mem_utils.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "platform_context.h"

#include "expression/testcase/source_stub.h"
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "op_creator_register.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
namespace {
REGISTER_LOWERING(DynamicQuantStub) {
  (void)loop::Store(node->GetOutDataAnchor(0), loop::Abs(loop::Load(node->GetInDataAnchor(0))));
  (void)loop::Store(node->GetOutDataAnchor(1), loop::Exp(loop::Load(node->GetInDataAnchor(0))));
  return GRAPH_SUCCESS;
}

std::string ReadableComputeGraph(const ComputeGraphPtr &graph, bool only_can_reached = true) {
  std::stringstream ss;
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  ss << "ComputeGraph(" << graph->GetName() << ")" << std::endl;
  std::set<NodePtr> can_reached;
  std::stack<NodePtr> stack;
  auto sink = graph->FindFirstNodeMatchType(NETOUTPUT);
  can_reached.insert(sink);
  if (sink != nullptr) {
    stack.push(sink);
    while (!stack.empty()) {
      auto current = stack.top();
      stack.pop();
      for (auto &in_node : current->GetInAllNodes()) {
        if (can_reached.insert(in_node).second) {
          stack.push(in_node);
        }
      }
    }
  }
  std::vector<std::string> unused_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (only_can_reached && can_reached.find(node) == can_reached.end()) {
      unused_nodes.emplace_back(node->GetName());
      continue;
    }
    std::vector<std::string> input_names;
    std::vector<std::string> control_names;
    for (auto &anchor : node->GetAllInDataAnchors()) {
      auto peer = anchor->GetPeerOutAnchor();
      if (peer == nullptr) {
        continue;
      }
      input_names.emplace_back(anchor_name[peer]);
    }
    for (auto &in_control : node->GetInControlNodes()) {
      control_names.emplace_back(in_control->GetName());
    }
    std::vector<std::string> output_names;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
    }
    if (output_names.size() > 1U) {
      ss << loop::StrJoin(output_names) << " = ";
    } else if (!output_names.empty()) {
      ss << output_names[0] << " = ";
    }
    if (control_names.empty()) {
      ss << "ge." << node->GetType() << "(" << node->GetName() << ", " << loop::StrJoin(input_names) << ")"
         << std::endl;
    } else {
      ss << "ge." << node->GetType() << "(" << node->GetName() << ", " << loop::StrJoin(input_names) << ", "
         << loop::StrJoin(control_names) << ")" << std::endl;
    }
  }
  ss << "ununsed nodes: " << loop::StrJoin(unused_nodes) << std::endl;
  return ss.str();
}
}  // namespace

class LoopAscIrLowerPrunerUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
    setenv("ENABLE_LOWER_MATMUL", "true", 1);
  }
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopAscIrLowerPrunerUT, GraphRecoverAsSmallFuseAfterAscIrLowerLifting) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
ununsed nodes: []
)");
}

template <typename T>
es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  return result;
}

TEST_F(LoopAscIrLowerPrunerUT, SkipLiftingIfAscIrOnlyContainTranspose) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(128, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(512, MakeShared<GraphInputShapeSourceStub>(0, 2));
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  SetCurShapeEnvContext(nullptr);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Const(FileConstant_0, [])
tmp2 = ge.AscBackend(autofuse_pointwise_0_Transpose, [tmp0], [FileConstant_0])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopAscIrLowerPrunerUT, noSkipLiftingIfAscIrContainTransposeWithTailAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{0, 2, 1});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(128, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(512, MakeShared<GraphInputShapeSourceStub>(0, 2));
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  SetCurShapeEnvContext(nullptr);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Const(FileConstant_0, [])
tmp3 = ge.NetOutput(NetOutput, [])
tmp4 = ge.Transpose(Transpose_1, [tmp0, tmp1])
ununsed nodes: []
)");
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopAscIrLowerPrunerUT, SkipLoweringIfGraphAlreadyFused) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  const auto kFusePass = [](const ComputeGraphPtr &cg) {
    ge::AscIrLowerer lowerer;
    ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
    ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

    EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs_Exp, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
ununsed nodes: []
)");
  };

  kFusePass(cg);
  kFusePass(cg);
}

TEST_F(LoopAscIrLowerPrunerUT, GraphFusedAndPrunedAfterAscIrLowerLifting) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs_Exp, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, GraphWithMultiOutNodePrunedAfterAscIrLowerLifting) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::DynamicQuant(data0);  // Don't care op type, Just for multi output
    n.y.SetSymbolShape({"s0", "s1", "s2"});
    n.scale.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(n.y, 0);
    es_graph_->SetOutput(n.scale, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("DynamicQuant")->GetOpDesc()->SetType("DynamicQuantStub");
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  cg->FindFirstNodeMatchType("DynamicQuantStub")->GetOpDesc()->SetType("DynamicQuant");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
[tmp1, tmp2] = ge.DynamicQuant(DynamicQuant_0, [tmp0])
[tmp3, tmp4] = ge.NetOutput(NetOutput, [tmp1, tmp2])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, GraphWithMultiOutNodePrunedAfterAscIrLowerLiftingNoRealLifting) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::DynamicQuant(data0);  // Don't care op type, Just for multi output
    n.y.SetSymbolShape({"s0", "s1", "s2"});
    n.scale.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(n.y);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(n.scale, 0);
    es_graph_->SetOutput(abs, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("DynamicQuant")->GetOpDesc()->SetType("DynamicQuantStub");
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  cg->FindFirstNodeMatchType("DynamicQuantStub")->GetOpDesc()->SetType("DynamicQuant");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_DynamicQuantStub, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_DynamicQuantStub_Abs, [tmp0])
[tmp3, tmp4] = ge.NetOutput(NetOutput, [tmp1, tmp2])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, NoExtraDataOutputAfterCanFuseLiftingEnd) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::ReduceSumD(abs1, {2}, true);
    n.SetSymbolShape({"s0", "s1", "1"});
    auto abs2 = es::Abs(n);
    abs2.SetSymbolShape({"s0", "s1", "1"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.ReduceSumD(ReduceSumD_1, [tmp1])
tmp3 = ge.AscBackend(autofuse_reduce_0_Abs_ReduceSumD, [tmp0])
tmp4 = ge.Abs(Abs_2, [])
tmp5 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp3])
tmp6 = ge.NetOutput(NetOutput, [tmp5])
ununsed nodes: []
)");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_reduce_0_Abs_ReduceSumD, [tmp0])
tmp2 = ge.Abs(Abs_2, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, LiftingEndAfterCanFuseFuseAscBackendTwice) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::ReduceSumD(abs1, {2}, true);
    n.SetSymbolShape({"s0", "s1", "1"});
    auto abs2 = es::Abs(n);
    abs2.SetSymbolShape({"s0", "s1", "1"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.ReduceSumD(ReduceSumD_1, [tmp1])
tmp3 = ge.AscBackend(autofuse_reduce_0_Abs_ReduceSumD, [tmp0])
tmp4 = ge.Abs(Abs_2, [])
tmp5 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp3])
tmp6 = ge.NetOutput(NetOutput, [tmp5])
ununsed nodes: []
)");
  auto asc_node1 = cg->FindNode("autofuse_reduce_0_Abs_ReduceSumD");
  auto stub_desc = AttrUtils::CloneOpDesc(asc_node1->GetOpDesc());
  auto stub_fused_asc = cg->InsertNode(asc_node1, stub_desc);
  stub_fused_asc->GetOpDesc()->SetType("FusedAscBackend");
  stub_fused_asc->GetOpDesc()->SetName("autofuse_reduce_0_Abs_ReduceSumD_fused");
  ASSERT_EQ(GraphUtils::ReplaceNodeAnchors(stub_fused_asc, asc_node1, {0}, {0}), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::RemoveNodeWithoutRelink(cg, asc_node1), GRAPH_SUCCESS);
  NodeUtils::UnlinkAll(*asc_node1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.FusedAscBackend(autofuse_reduce_0_Abs_ReduceSumD_fused, [tmp0])
tmp2 = ge.Abs(Abs_2, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, NoExtraDataOutputAfterCanFuseLiftingBoth) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::ReduceSumD(data0, {2}, true);
    n.SetSymbolShape({"s0", "s1", "1"});
    auto abs2 = es::Abs(n);
    abs2.SetSymbolShape({"s0", "s1", "1"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.ReduceSumD(ReduceSumD_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_reduce_0_ReduceSumD, [tmp0])
tmp3 = ge.Abs(Abs_1, [])
tmp4 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp2])
tmp5 = ge.NetOutput(NetOutput, [tmp4])
ununsed nodes: []
)");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.ReduceSumD(ReduceSumD_0, [tmp0])
tmp2 = ge.Abs(Abs_1, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, NoExtraDataOutputAfterCanFuseLiftingBegin) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::ReduceSumD(data0, {2}, true);
    n.SetSymbolShape({"s0", "s1", "1"});
    auto abs1 = es::Abs(n);
    abs1.SetSymbolShape({"s0", "s1", "1"});
    auto abs2 = es::Abs(abs1);
    abs2.SetSymbolShape({"s0", "s1", "1"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.ReduceSumD(ReduceSumD_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_reduce_0_ReduceSumD, [tmp0])
tmp3 = ge.Abs(Abs_1, [])
tmp4 = ge.Abs(Abs_2, [tmp3])
tmp5 = ge.AscBackend(autofuse_pointwise_1_Abs_Abs, [tmp2])
tmp6 = ge.NetOutput(NetOutput, [tmp5])
ununsed nodes: []
)");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.ReduceSumD(ReduceSumD_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_Abs_Abs, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, DfxAfterAscIrLowerLifting) {
  [this]() {
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
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (auto &node : graph->GetAllNodes()) {
    NodePtr ori_node = NodeAdapter::GNode2Node(node);
    std::string type = ori_node->GetType();
    std::vector<string> origin_op_types = {ori_node->GetType()};
    std::vector<string> origin_op_names = {ori_node->GetName()};
    ge::AttrUtils::SetListStr(ori_node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_op_types);
    ge::AttrUtils::SetListStr(ori_node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, origin_op_names);
    if (type == "Exp") {
      ge::AttrUtils::SetBool(ori_node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, false);
    }
  }

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoopAscIrLowerPrunerUT, KeepDataOutputAsConcreteEdgeStillInUse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0"});
    auto is_nan = es::IsNan(add);
    is_nan.SetSymbolShape({"s0"});
    auto cast = es::Cast(is_nan, DT_FLOAT16);
    cast.SetSymbolShape({"s0"});
    auto exp = es::Exp(cast);
    exp.SetSymbolShape({"s0"});
    auto logical_and = es::LogicalAnd(is_nan, data2);
    logical_and.SetSymbolShape({"s0"});
    auto cast2 = es::Cast(logical_and, DT_UNDEFINED);
    cast2.SetSymbolShape({"s0"});
    es_graph_->SetOutput(exp, 0);
    es_graph_->SetOutput(cast2, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(data2, [])
tmp3 = ge.Add(Add_0, [tmp0, tmp1])
tmp4 = ge.IsNan(IsNan_1, [tmp3])
tmp5 = ge.AscBackend(autofuse_pointwise_0_Add_IsNan, [tmp0, tmp1])
tmp6 = ge.Cast(Cast_2, [])
tmp7 = ge.Exp(Exp_3, [tmp6])
tmp8 = ge.AscBackend(autofuse_pointwise_1_Cast_Exp, [tmp5])
tmp9 = ge.LogicalAnd(LogicalAnd_4, [tmp2])
tmp10 = ge.AscBackend(autofuse_pointwise_2_LogicalAnd, [tmp5, tmp2])
tmp11 = ge.Cast(Cast_5, [tmp10])
[tmp12, tmp13] = ge.NetOutput(NetOutput, [tmp8, tmp11])
ununsed nodes: []
)");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(data2, [])
tmp3 = ge.AscBackend(autofuse_pointwise_0_Add_IsNan, [tmp0, tmp1])
tmp4 = ge.AscBackend(autofuse_pointwise_1_Cast_Exp, [tmp3])
tmp5 = ge.LogicalAnd(LogicalAnd_4, [tmp3, tmp2])
tmp6 = ge.Cast(Cast_5, [tmp5])
[tmp7, tmp8] = ge.NetOutput(NetOutput, [tmp4, tmp6])
ununsed nodes: []
)");
}

TEST_F(LoopAscIrLowerPrunerUT, SimpleConcatIgnoreLowering) {
  int concatN = 33;
  [this, concatN]() {
    vector<es::Tensor> datas;
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"1", "s1", "s2"});
    int i = 1;
    for (; i < concatN; i++) {
      string str = "data" + to_string(i);
      auto datatmp = es_graph_->CreateInput(i, str.c_str(), nullptr);
      datatmp.SetSymbolShape({"1", "s1", "s2"});
      datas.push_back(datatmp);
    }

    auto abs = es::Abs(data);
    abs.SetSymbolShape({"1", "s1", "s2"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"1", "s1", "s2"});
    datas.push_back(abs1);
    auto concat = es::ConcatD(datas, 0, concatN);
    concat.SetSymbolShape({"64", "s1", "s2"});
    auto abs2 = es::Abs(concat);
    abs2.SetSymbolShape({"64", "s1", "s2"});
    auto abs3 = es::Abs(abs2);
    abs3.SetSymbolShape({"64", "s1", "s2"});
    es_graph_->SetOutput(abs3, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  auto readable = ReadableComputeGraph(cg, false);
  EXPECT_TRUE(readable.find("ge.ConcatD") != std::string::npos);
}

TEST_F(LoopAscIrLowerPrunerUT, SimpleConcatIgnoreLifting) {
  int concatN = 64;
  [this, concatN]() {
    vector<es::Tensor> datas;
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"1", "16", "32"});
    int i = 1;
    for (; i < concatN; i++) {
      string str = "data" + to_string(i);
      auto datatmp = es_graph_->CreateInput(i, str.c_str(), nullptr);
      datatmp.SetSymbolShape({"1", "16", "32"});
      datas.push_back(datatmp);
    }

    auto abs = es::Abs(data);
    abs.SetSymbolShape({"1", "16", "32"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"1", "16", "32"});
    datas.push_back(abs1);
    auto concat = es::ConcatD(datas, 0, concatN);
    concat.SetSymbolShape({"64", "16", "32"});
    auto abs2 = es::Abs(concat);
    abs2.SetSymbolShape({"64", "16", "32"});
    auto abs3 = es::Abs(abs2);
    abs3.SetSymbolShape({"64", "16", "32"});
    es_graph_->SetOutput(abs3, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  string readable_lifting = "ComputeGraph(graph)\n";
  int j = 0;
  for (; j < concatN; j++) {
    readable_lifting += "tmp" + to_string(j) + " = ge.Data(data" + to_string(j) + ", [])\n";
  }
  readable_lifting += "tmp" + to_string(j++) + " = ge.AscBackend(autofuse_pointwise_0_Abs_Abs, [tmp0])\n";
  string concatStr = "tmp" + to_string(j++) + " = ge.AscBackend(autofuse_concat_1_ConcatD, [tmp1";
  for (int k = 2; k <= concatN; k++) {
    concatStr += ", tmp" + to_string(k);
  }
  concatStr += "])\n";
  readable_lifting += concatStr;
  readable_lifting += "tmp" + to_string(j++) + " = ge.AscBackend(autofuse_pointwise_2_Abs_Abs, [tmp" + to_string(j - 1) + "])\n";
  readable_lifting += "tmp" + to_string(j++) + " = ge.NetOutput(NetOutput, [tmp" + to_string(j - 1) + "])\n";
  readable_lifting += "ununsed nodes: []\n";
  EXPECT_EQ(ReadableComputeGraph(cg, false), readable_lifting);
}

TEST_F(LoopAscIrLowerPrunerUT, SimpleConcatIgnoreLifting1) {
  int concatN = 32;
  [this, concatN]() {
    vector<es::Tensor> datas;
    int i = 0;
    for (; i < concatN; i++) {
      string str = "data" + to_string(i);
      auto datatmp = es_graph_->CreateInput(i, str.c_str(), nullptr);
      datatmp.SetSymbolShape({"1", "16", "32"});
      datas.push_back(datatmp);
      datas.push_back(datatmp);
    }

    auto concat = es::ConcatD(datas, 0, concatN);
    concat.SetSymbolShape({"64", "16", "32"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  string readable_lifting = "ComputeGraph(graph)\n";
  int j = 0;
  for (; j < concatN; j++) {
    readable_lifting += "tmp" + to_string(j) + " = ge.Data(data" + to_string(j) + ", [])\n";
  }
  string concatStr = "tmp" + to_string(j++) + " = ge.AscBackend(autofuse_concat_0_ConcatD, [tmp0";
  for (int k = 1; k < concatN; k++) {
    concatStr += ", tmp" + to_string(k);
  }
  concatStr += "])\n";
  readable_lifting += concatStr;
  readable_lifting += "tmp" + to_string(j++) + " = ge.NetOutput(NetOutput, [tmp" + to_string(j - 1) + "])\n";
  readable_lifting += "ununsed nodes: []\n";
  EXPECT_EQ(ReadableComputeGraph(cg, false), readable_lifting);
}

TEST_F(LoopAscIrLowerPrunerUT, SimpleReshapeConst) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "1", "s1", "1", "1"});
    auto shape = es::FileConstant(*es_graph_, {2}, ge::DT_INT32);
    GeTensorDesc desc(GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
    vector<int32_t> vec = {1, 3, 4};
    GeTensorPtr tensor =
        std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(vec.data()), sizeof(int32_t) * vec.size());
    AttrUtils::SetTensor(shape.GetEsbTensor()->GetProducer()->GetOpDesc(), "shape", tensor);
    shape.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoopAscIrLowerPrunerUT, ReshapeUnsqueezeTwoViewNode) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "1", "s1", "1", "1"});
    auto shape = es::FileConstant(*es_graph_, {2}, ge::DT_INT32);
    GeTensorDesc desc(GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
    vector<int32_t> vec = {1, 3, 4};
    GeTensorPtr tensor =
        std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(vec.data()), sizeof(int32_t) * vec.size());
    AttrUtils::SetTensor(shape.GetEsbTensor()->GetProducer()->GetOpDesc(), "shape", tensor);
    shape.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});

    auto unsqueeze = es::Unsqueeze(reshape, {2});
    unsqueeze.SetSymbolShape({"s0", "s1", "1"});
    es_graph_->SetOutput(unsqueeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoopAscIrLowerPrunerUT, ReshapeMultiOutAnchorFallBackLowering) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "1", "s1", "1", "1"});
    auto shape = es::FileConstant(*es_graph_, {2}, ge::DT_INT32);
    GeTensorDesc desc(GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
    vector<int32_t> vec = {1, 3, 4};
    GeTensorPtr tensor =
        std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(vec.data()), sizeof(int32_t) * vec.size());
    AttrUtils::SetTensor(shape.GetEsbTensor()->GetProducer()->GetOpDesc(), "shape", tensor);
    shape.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});
    auto unsqueeze = es::Unsqueeze(reshape, {2});
    unsqueeze.SetSymbolShape({"s0", "s1", "1"});

    auto abs = es::Abs(reshape);
    abs.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(unsqueeze, 0);
    es_graph_->SetOutput(abs, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoopAscIrLowerPrunerUT, SimpleUnsqueezeSqueeze) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1"});
    auto unsqueeze = es::Unsqueeze(x, {2});
    unsqueeze.SetSymbolShape({"s0", "s1", "1"});
    auto squeeze = es::Squeeze(unsqueeze, {2});
    squeeze.SetSymbolShape({"s0", "s1"});
    auto abs = es::Abs(unsqueeze);
    abs.SetSymbolShape({"s0", "s1", "1"});
    es_graph_->SetOutput(squeeze, 0);
    es_graph_->SetOutput(abs, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  GE_DUMP(cg, "lowering0");
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  GE_DUMP(cg, "lowering1");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  GE_DUMP(cg, "lowering2");
}

TEST_F(LoopAscIrLowerPrunerUT, SimpleSkipLifting) {
  AutoFuseConfig::MutableLoweringConfig().experimental_disable_lifting = true;
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(x);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(x, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
  AutoFuseConfig::MutableLoweringConfig().experimental_disable_lifting = false;
}


class TestCounter : public Counter {
public:
  TestCounter() = default;
  virtual ~TestCounter() = default;
  virtual int64_t NextId() { return id_++;};
private:
  int64_t id_ = 0;
};

TEST_F(LoopAscIrLowerPrunerUT, SimpleLoweringName) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(x);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto sum = es::ReduceSumD(abs1, {-1}, true);
    sum.SetSymbolShape({"s0", "s1", "1"});
    auto abs2 = es::Abs(sum);
    abs2.SetSymbolShape({"s0", "s1", "1"});
    auto squeeze = es::Squeeze(abs2, {-1});
    squeeze.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  TestCounter counter;
  ge::AscIrLowerer lowerer(&counter);
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(x, [])
tmp1 = ge.AscBackend(autofuse_reduce_0_Abs_Abs_ReduceSumD, [tmp0])
tmp2 = ge.Abs(Abs_3, [tmp1])
tmp3 = ge.Squeeze(Squeeze_4, [tmp2])
tmp4 = ge.NetOutput(NetOutput, [tmp3])
ununsed nodes: []
)");
}
}  // namespace ge
