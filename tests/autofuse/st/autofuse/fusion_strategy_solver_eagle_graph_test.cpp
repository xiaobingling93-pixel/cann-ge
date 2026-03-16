
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
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "graph/utils/graph_utils_ex.h"
#include "op_creator_register.h"
#include "attribute_group/attr_group_shape_env.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/lowerings.h"
#include "utils/autofuse_attrs.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"

using namespace std;
using namespace testing;

namespace ge {
namespace {
}  // namespace

class FusionStrategySolverST : public testing::Test {
 public:
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    dlog_setlevel(0, 3, 0);
    RegisterAllOpCreator();
  }
  void TearDown() override {}
  std::unique_ptr<es::Graph> es_graph_;
};

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

TEST_F(FusionStrategySolverST, ReduceAndConcatCanNotfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto concat = es::ConcatD({abs, data0}, 0);
    concat.SetSymbolShape({"s0 * 2", "s1", "s2"});
    auto sum = es::ReduceSumD(concat, {1}, true);
    sum.SetSymbolShape({"s0 * 2", "1", "s2"});
    auto abs1 = es::Abs(sum);
    abs1.SetSymbolShape({"s0 * 2", "1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(exp, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  ASSERT_EQ(cg->GetAllNodesSize(), 5U);
}

TEST_F(FusionStrategySolverST, ConcatAndReduceCanNotfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto concat = es::ConcatD({abs, data0}, 0);
    concat.SetSymbolShape({"s0 * 2", "s1", "s2"});
    auto sum = es::ReduceSumD({abs}, {1}, true);
    sum.SetSymbolShape({"s0", "1", "s2"});
    auto abs1 = es::Abs(sum);
    abs1.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(exp, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  GraphUtils::AddEdge(cg->FindNode("ConcatD_2")->GetOutControlAnchor(),
                      cg->FindNode("ReduceSumD_3")->GetInControlAnchor());
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetName() << std::endl;
  }
  ASSERT_EQ(cg->GetAllNodesSize(), 5U);
}

/*      data
 *        |
 *       abs
 *      /   \
 *  concat-->abs
 *     \     /
 *      netout
 */
TEST_F(FusionStrategySolverST, AbsAndConcatCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto concat = es::ConcatD({data0, abs}, 0);
    concat.SetSymbolShape({"s0 * 2", "s1", "s2"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(concat, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  GraphUtils::AddEdge(cg->FindNode("ConcatD_1")->GetOutControlAnchor(), cg->FindNode("Abs_2")->GetInControlAnchor());
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  ASSERT_EQ(cg->GetAllNodesSize(), 4U);
}

TEST_F(FusionStrategySolverST, AbsAndSliceNotCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    const std::vector<int64_t> begin = {0, 0, 1};
    const std::vector<int64_t> end = {10, 10, 10};
    const std::vector<int64_t> strides = {1, 1, 1};
    auto slice = es::StridedSliceD(abs, begin, end, strides);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    auto abs1 = es::Abs(slice);
    abs1.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  ASSERT_EQ(cg->GetAllNodesSize(), 4U);
}

TEST_F(FusionStrategySolverST, SliceEndDimIsOneNotCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    const std::vector<int64_t> begin = {0, 0, 1};
    const std::vector<int64_t> end = {10, 10, 2};
    const std::vector<int64_t> strides = {1, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, strides);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    auto abs = es::Abs(slice);
    abs.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  ASSERT_EQ(cg->GetAllNodesSize(), 3U);
}
TEST_F(FusionStrategySolverST, ZerolikeHorizonFuseWithTransposeNotCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"165", "22", "34"});
    auto zero_like = es::ZerosLike(data0);
    zero_like.SetSymbolShape({"165", "22", "34"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 2, 0});
    auto trans = es::Transpose(zero_like, perms);
    trans.SetSymbolShape({"22", "34", "165"});
    es_graph_->SetOutput(trans, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  ASSERT_EQ(cg->GetAllNodesSize(), 4U);
}
}  // namespace ge
