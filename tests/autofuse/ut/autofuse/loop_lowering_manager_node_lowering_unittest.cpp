
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <utility>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"

#include "graph/utils/graph_utils_ex.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/auto_fuse_config.h"
#include "ascgen_log.h"

#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include <gtest/gtest.h>

using namespace std;
using namespace testing;

namespace ge {

class LoopNodeLoweringUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("Hi Lowering graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {}
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopNodeLoweringUT, In2Out1Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, MatMulLowering1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"4", "4", "4"});
    data1.SetSymbolShape({"4", "4", "4"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"4", "4", "4"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_matmul = true;
  ASSERT_EQ(LoweringManager::Lowering(mm), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsCube());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreMatMul(\"MatMul_0:0\", [tmp0, tmp1], transpose_x1=0, transpose_x2=0, offset_x=0, "
            "enable_hf32=1, has_bias=0, has_offset_w=0)\n");
  ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_matmul = false;
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase1Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d0, 1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase2Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"1", "1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1, 1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase3Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "1"});
    data1.SetSymbolShape({"1", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d0, 1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Broadcast(tmp2, \"[1, d1]->[d0, d1]\")\n"
            "tmp4 = ops.Add(tmp1, tmp3)\n"
            "tmp5 = ops.Store(\"Add_0:0\", tmp4)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase4Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2Out1WithBroadcastCase5Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"1"});
    data1.SetSymbolShape({"s0", "s1"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1]->[d0, d1]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In1Out1Lowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp = cg->FindNode("Exp_0");
  ASSERT_NE(exp, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(exp), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(exp->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, PointwiseFusion) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(exp);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto exp = cg->FindNode("Exp_0");
  ASSERT_NE(exp, nullptr);
  auto abs = cg->FindNode("Abs_1");
  ASSERT_NE(abs, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_exp = ge::loop::GetKernelBox(exp->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_exp.IsExternKernel());
  EXPECT_EQ(kernel_exp.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n");

  auto kernel_abs = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs.IsExternKernel());
  EXPECT_EQ(kernel_abs.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n"
            "tmp3 = ops.Abs(tmp2)\n"
            "tmp4 = ops.Store(\"Abs_1:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In1PointwiseFuseion2useOf1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(exp);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(exp);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(abs2, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto exp = cg->FindNode("Exp_0");
  ASSERT_NE(exp, nullptr);
  auto abs1 = cg->FindNode("Abs_1");
  ASSERT_NE(abs1, nullptr);
  auto abs2 = cg->FindNode("Abs_2");
  ASSERT_NE(abs2, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_abs1 = ge::loop::GetKernelBox(abs1->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs1.IsExternKernel());
  EXPECT_EQ(kernel_abs1.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n"
            "tmp3 = ops.Abs(tmp2)\n"
            "tmp4 = ops.Store(\"Abs_1:0\", tmp3)\n");
  auto kernel_abs2 = ge::loop::GetKernelBox(abs2->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs2.IsExternKernel());
  EXPECT_EQ(kernel_abs2.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Exp(tmp0)\n"
            "tmp2 = ops.Store(\"Exp_0:0\", tmp1)\n"
            "tmp3 = ops.Abs(tmp2)\n"
            "tmp4 = ops.Store(\"Abs_2:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, In2PointwiseFuseion2useOf1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto div = es::Div(add, data2);
    div.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(div, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto div = cg->FindNode("Div_1");
  ASSERT_NE(div, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_div = ge::loop::GetKernelBox(div->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_div.IsExternKernel());
  EXPECT_EQ(kernel_div.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Div(tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Div_1:0\", tmp5)\n");
}

TEST_F(LoopNodeLoweringUT, In1And2PointwiseAnd2use1ComplexLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto div = es::Div(add, data2);
    div.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(div);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(div);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
    es_graph_->SetOutput(abs, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto exp = cg->FindNode("Exp_2");
  ASSERT_NE(exp, nullptr);
  auto abs = cg->FindNode("Abs_3");
  ASSERT_NE(abs, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel_exp = ge::loop::GetKernelBox(exp->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_exp.IsExternKernel());
  EXPECT_EQ(kernel_exp.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Div(tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Div_1:0\", tmp5)\n"
            "tmp7 = ops.Exp(tmp6)\n"
            "tmp8 = ops.Store(\"Exp_2:0\", tmp7)\n");

  auto kernel_abs = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_TRUE(!kernel_abs.IsExternKernel());
  EXPECT_EQ(kernel_abs.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"Add_0:0\", tmp2)\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Div(tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Div_1:0\", tmp5)\n"
            "tmp7 = ops.Abs(tmp6)\n"
            "tmp8 = ops.Store(\"Abs_3:0\", tmp7)\n");
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

TEST_F(LoopNodeLoweringUT, LoweringReduceSumInt32) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSum_1:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumNegAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{-2});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSum_1:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumNegAxisOverRank) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{-4});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    LoweringManager::Lowering(node);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumInt64) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"ReduceSum_1:0\", ops.Sum(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatInt32) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreConcat(\"Concat_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatInt64) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreConcat(\"Concat_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatNegAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.StoreConcat(\"Concat_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringConcatNegAxisOverRank) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3"});
    auto concat_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-3});
    auto reduce = es::Concat(concat_dim, {data0, data1}, 2);
    reduce.SetSymbolShape({"s0", "5"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("Concat_1");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumUnsupportedDtype) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_FLOAT, {1}, std::vector<float>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "ReduceSum") {
      ASSERT_NE(LoweringManager::Lowering(node), GRAPH_SUCCESS);
    } else {
      ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
    }
  }
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumSupportedMultiAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {2}, std::vector<int32_t>{1, 2});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
}

TEST_F(LoopNodeLoweringUT, LoweringReduceSumUnsupportedAllAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s1", "s2", "s3"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 2, 3});
    auto reduce = es::ReduceSum(data0, axis, true);
    reduce.SetSymbolShape({"1", "1", "1", "1"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, ComplexBiasaddNHWCLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s2"});
    auto add = es::BiasAdd(data0, data1, "NHWC");
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("BiasAdd_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Broadcast(tmp1, \"[d2]->[d0, d1, d2]\")\n"
            "tmp3 = ops.Add(tmp0, tmp2)\n"
            "tmp4 = ops.Store(\"BiasAdd_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, ComplexBiasaddFailLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s2"});
    auto add = es::BiasAdd(data0, data1, "NHW");
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("BiasAdd_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_FAILED);
}

TEST_F(LoopNodeLoweringUT, ComplexZerosLikeCaseLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s2"});
    auto add = es::BiasAdd(data0, data1, "NHWC");
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto zeros_like = es::ZerosLike(add);
    zeros_like.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(zeros_like, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("ZerosLike_1");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Scalar(\"DT_FLOAT(0)\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"ZerosLike_1:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringZeroslikeBool) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"28", "28"});
    auto zeroslike = es::ZerosLike(data0);
    zeroslike.SetSymbolShape({"28", "28"});
    es_graph_->SetOutput(zeroslike, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto data0 = cg->FindNode("data0");
  auto data_desc1 = data0->GetOpDesc()->MutableInputDesc(0);
  data_desc1->SetDataType(ge::DT_BOOL);
  data_desc1->SetOriginDataType(ge::DT_BOOL);
  auto data_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  data_desc->SetDataType(ge::DT_BOOL);
  data_desc->SetOriginDataType(ge::DT_BOOL);
  auto tmp = cg->FindNode("ZerosLike_0");
  auto tmp_desc1 = tmp->GetOpDesc()->MutableInputDesc(0);
  tmp_desc1->SetDataType(ge::DT_BOOL);
  tmp_desc1->SetOriginDataType(ge::DT_BOOL);
  auto tmp_desc = tmp->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(ge::DT_BOOL);
  tmp_desc->SetOriginDataType(ge::DT_BOOL);

  ASSERT_EQ(LoweringManager::Lowering(tmp), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),"tmp0 = ops.Scalar(\"DT_BOOL(0)\")\n"
"tmp1 = ops.Broadcast(tmp0, \"[]->[d0, d1]\")\n"
"tmp2 = ops.Store(\"ZerosLike_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, ComplexBiasaddNCHWLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto bias_add = es::BiasAdd(abs, data1, "NCHW");
    bias_add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(bias_add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto bias_add = cg->FindNode("BiasAdd_1");
  ASSERT_NE(bias_add, nullptr);
  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  ASSERT_EQ(LoweringManager::Lowering(bias_add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(bias_add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_0:0\", tmp1)\n"
            "tmp3 = ops.Load(\"data1:0\")\n"
            "tmp4 = ops.Broadcast(tmp3, \"[d0]->[d0, d1, d2]\")\n"
            "tmp5 = ops.Add(tmp2, tmp4)\n"
            "tmp6 = ops.Store(\"BiasAdd_1:0\", tmp5)\n");
}

TEST_F(LoopNodeLoweringUT, SelectAndSigmoidLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    data2.SetSymbolShape({"s1", "s2"});
    auto select = es::Select(data0, data1, data2);
    select.SetSymbolShape({"s0", "s1", "s2"});
    auto sigmoid = es::Sigmoid(select);
    sigmoid.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(sigmoid, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto sigmoid = cg->FindNode("Sigmoid_1");
  ASSERT_NE(sigmoid, nullptr);
  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }

  ASSERT_EQ(LoweringManager::Lowering(sigmoid), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(sigmoid->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Load(\"data2:0\")\n"
            "tmp3 = ops.Broadcast(tmp2, \"[d1, d2]->[d0, d1, d2]\")\n"
            "tmp4 = ops.Where(tmp0, tmp1, tmp3)\n"
            "tmp5 = ops.Store(\"Select_0:0\", tmp4)\n"
            "tmp6 = ops.Sigmoid(tmp5)\n"
            "tmp7 = ops.Store(\"Sigmoid_1:0\", tmp6)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringAddN) {
  [this]() {
    std::vector<es::Tensor> datas = {};
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1"});
    datas.emplace_back(data0);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s1", "s0"});
    datas.emplace_back(data1);
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s2", "1", "1"});
    datas.emplace_back(data2);
    auto addN = es::AddN(datas, datas.size());
    addN.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(addN, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto addN = cg->FindNode("AddN_0");
  ASSERT_NE(addN, nullptr);
  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel = ge::loop::GetKernelBox(addN->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsRealizedPersistent());
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Broadcast(tmp2, \"[1, d1, d2]->[d0, d1, d2]\")\n"
            "tmp4 = ops.Add(tmp1, tmp3)\n"
            "tmp5 = ops.Load(\"data2:0\")\n"
            "tmp6 = ops.Broadcast(tmp5, \"[d0, 1, 1]->[d0, d1, d2]\")\n"
            "tmp7 = ops.Add(tmp4, tmp6)\n"
            "tmp8 = ops.Store(\"AddN_0:0\", tmp7)\n");
}

TEST_F(LoopNodeLoweringUT, TileDTensorLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s1", "s2"});
    auto tile = es::TileD(data0, {5, 1, 1});
    tile.SetSymbolShape({"5", "s1", "s2"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("TileD_0");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1, d1, d2]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"TileD_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, TileDScalarLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s2"});
    auto tile = es::TileD(data0, {5, 1, 1});
    tile.SetSymbolShape({"5", "1", "s2"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("TileD_0");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d2]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"TileD_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, TileDLoweringFailed) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s2"});
    auto tile = es::TileD(data0, {5, 1, 2});
    tile.SetSymbolShape({"5", "1", "s3"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("TileD_0");
  ASSERT_NE(tile, nullptr);

  ASSERT_NE(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, TileTensorLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s1", "s2"});
    auto multiplies = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{5, 1, 1});
    auto tile = es::Tile(data0, multiplies);
    tile.SetSymbolShape({"5", "s1", "s2"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("Tile_1");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[1, d1, d2]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"Tile_1:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, TileScalarLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s2"});
    auto multiplies = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{5, 1, 1});
    auto tile = es::Tile(data0, multiplies);
    tile.SetSymbolShape({"5", "1", "s2"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("Tile_1");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d2]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"Tile_1:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, TileLoweringFailed) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s2"});
    auto multiplies = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{5, 1, 2});
    auto tile = es::Tile(data0, multiplies);
    tile.SetSymbolShape({"5", "1", "s3"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("Tile_1");
  ASSERT_NE(tile, nullptr);

  ASSERT_NE(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, TileTensorLoweringConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"2", "s1", "s2"});
    auto multiplies = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{2, 1, 1});
    auto tile = es::Tile(data0, multiplies);
    tile.SetSymbolShape({"4", "s1", "s2"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("Tile_1");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(tile), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.StoreConcat(\"Tile_1:0\", [tmp0, tmp1], concat_dim=0)\n");
}

TEST_F(LoopNodeLoweringUT, SquareLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto square = es::Square(data0);
    square.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(square, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto square = cg->FindNode("Square_0");
  ASSERT_NE(square, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(square), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(square->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.Mul(tmp1, tmp1)\n"
            "tmp3 = ops.Store(\"Square_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, SquaredDifferenceLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto sd = es::SquaredDifference(data0, data1);
    sd.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(sd, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto sd = cg->FindNode("SquaredDifference_0");
  ASSERT_NE(sd, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(sd), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(sd->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Load(\"data1:0\")\n"
            "tmp4 = ops.Sub(tmp1, tmp3)\n"
            "tmp5 = ops.Sub(tmp1, tmp3)\n"
            "tmp6 = ops.Mul(tmp5, tmp5)\n"
            "tmp7 = ops.Store(\"SquaredDifference_0:0\", tmp6)\n");
}

TEST_F(LoopNodeLoweringUT, SquaredDifferenceWithBroadcastLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "1"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto sd = es::SquaredDifference(data0, data1);
    sd.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(sd, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto sd = cg->FindNode("SquaredDifference_0");
  ASSERT_NE(sd, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(sd), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(sd->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.Broadcast(tmp1, \"[d0, d1, 1]->[d0, d1, d2]\")\n"
            "tmp3 = ops.Broadcast(tmp1, \"[d0, d1, 1]->[d0, d1, d2]\")\n"
            "tmp4 = ops.Load(\"data1:0\")\n"
            "tmp5 = ops.Load(\"data1:0\")\n"
            "tmp6 = ops.Sub(tmp3, tmp5)\n"
            "tmp7 = ops.Sub(tmp3, tmp5)\n"
            "tmp8 = ops.Mul(tmp7, tmp7)\n"
            "tmp9 = ops.Store(\"SquaredDifference_0:0\", tmp8)\n");
}

TEST_F(LoopNodeLoweringUT, CastLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto cast = es::Cast(data0, DT_INT32);
    cast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(cast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cast = cg->FindNode("Cast_0");
  ASSERT_NE(cast, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(cast), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(cast->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Cast(tmp0, DT_INT32)\n"
            "tmp2 = ops.Store(\"Cast_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, CastLoweringBool) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto cast = es::Cast(data0, DT_BOOL);
    cast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(cast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cast = cg->FindNode("Cast_0");
  ASSERT_NE(cast, nullptr);

  ASSERT_NE(LoweringManager::Lowering(cast), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, SquareSumV1Lowering) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<int64_t> axis = {0, -2};
    auto squaresumv1 = es::SquareSumV1(x, axis, true);

    squaresumv1.SetSymbolShape({"1", "1", "s2"});
    es_graph_->SetOutput(squaresumv1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squaresumv1 = cg->FindNode("SquareSumV1_0");
  ASSERT_NE(squaresumv1, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(squaresumv1), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(squaresumv1->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Load(\"x:0\")\n"
            "tmp2 = ops.Mul(tmp1, tmp1)\n"
            "tmp3 = ops.StoreReduction(\"SquareSumV1_0:0\", ops.Sum(tmp2, \"[d0, d1, d2]->[1, 1, d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, SquareSumV1LoweringAllOne) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"1", "1", "s0", "s1", "s2"});
    std::vector<int64_t> axis = {0, 1};
    auto squaresumv1 = es::SquareSumV1(x, axis, true);

    squaresumv1.SetSymbolShape({"1", "1", "s2"});
    es_graph_->SetOutput(squaresumv1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squaresumv1 = cg->FindNode("SquareSumV1_0");
  ASSERT_NE(squaresumv1, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(squaresumv1), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(squaresumv1->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Load(\"x:0\")\n"
            "tmp2 = ops.Mul(tmp1, tmp1)\n"
            "tmp3 = ops.Squeeze(tmp2, 0)\n"
            "tmp4 = ops.Squeeze(tmp3, 0)\n"
            "tmp5 = ops.Store(\"SquareSumV1_0:0\", tmp4)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringBiasAddGrad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto add = es::BiasAddGrad(data0, "NHWC");
    add.SetSymbolShape({"s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("BiasAddGrad_0");
  ASSERT_NE(add, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(add), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.StoreReduction(\"BiasAddGrad_0:0\", ops.Sum(tmp0, \"[d0, d1]->[d1]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringBroadcastTo) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"3"});
    auto broadcastto = es::BroadcastTo(x, shape);
    broadcastto.SetSymbolShape({"s2", "s0", "s1"});
    es_graph_->SetOutput(broadcastto, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto broadcastto = cg->FindNode("BroadcastTo_0");
  ASSERT_NE(broadcastto, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(broadcastto->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1, d2]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"BroadcastTo_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringLayerNormBetaGammaBackpropV2Neg) {
  [this]() {
    auto dy = es_graph_->CreateInput(0, "dy", nullptr);
    dy.SetSymbolShape({"s0", "s1", "s2"});
    auto res_for_gamma = es_graph_->CreateInput(1, "res_for_gamma", nullptr);
    res_for_gamma.SetSymbolShape({"s0", "s1", "s2"});
    vector<int64_t> shape_gamma = {-1, 0, -1};
    auto tmp = es::LayerNormBetaGammaBackpropV2(dy, res_for_gamma, shape_gamma);
    tmp.pd_gamma.SetSymbolShape({"s1"});
    tmp.pd_beta.SetSymbolShape({"s1"});
    es_graph_->SetOutput(tmp.pd_gamma, 0);
    es_graph_->SetOutput(tmp.pd_beta, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tmp = cg->FindNode("LayerNormBetaGammaBackpropV2_0");
  ASSERT_NE(tmp, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"dy:0\")\n"
            "tmp1 = ops.Load(\"res_for_gamma:0\")\n"
            "tmp2 = ops.Mul(tmp0, tmp1)\n"
            "tmp3 = ops.StoreReduction(\"LayerNormBetaGammaBackpropV2_0:0\", ops.Sum(tmp2, \"[d0, d1, d2]->[d1]\"))\n");

  auto kernel1 = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(1));
  ASSERT_FALSE(kernel1.IsExternKernel());
  EXPECT_EQ(kernel1.Readable(),
            "tmp0 = ops.Load(\"dy:0\")\n"
            "tmp1 = ops.StoreReduction(\"LayerNormBetaGammaBackpropV2_0:1\", ops.Sum(tmp0, \"[d0, d1, d2]->[d1]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringLayerNormBetaGammaBackpropV2) {
  [this]() {
    auto dy = es_graph_->CreateInput(0, "dy", nullptr);
    dy.SetSymbolShape({"s0", "s1", "s2"});
    auto res_for_gamma = es_graph_->CreateInput(1, "res_for_gamma", nullptr);
    res_for_gamma.SetSymbolShape({"s0", "s1", "s2"});
    vector<int64_t> shape_gamma = {2, 2};
    auto tmp = es::LayerNormBetaGammaBackpropV2(dy, res_for_gamma, shape_gamma);
    tmp.pd_gamma.SetSymbolShape({"s2"});
    tmp.pd_beta.SetSymbolShape({"s2"});
    es_graph_->SetOutput(tmp.pd_gamma, 0);
    es_graph_->SetOutput(tmp.pd_beta, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tmp = cg->FindNode("LayerNormBetaGammaBackpropV2_0");
  ASSERT_NE(tmp, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"dy:0\")\n"
            "tmp1 = ops.Load(\"res_for_gamma:0\")\n"
            "tmp2 = ops.Mul(tmp0, tmp1)\n"
            "tmp3 = ops.StoreReduction(\"LayerNormBetaGammaBackpropV2_0:0\", ops.Sum(tmp2, \"[d0, d1, d2]->[d2]\"))\n");

  auto kernel1 = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(1));
  ASSERT_FALSE(kernel1.IsExternKernel());
  EXPECT_EQ(kernel1.Readable(),
            "tmp0 = ops.Load(\"dy:0\")\n"
            "tmp1 = ops.StoreReduction(\"LayerNormBetaGammaBackpropV2_0:1\", ops.Sum(tmp0, \"[d0, d1, d2]->[d2]\"))\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReluGrad) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"gradients:0\")\n"
"tmp1 = ops.Load(\"features:0\")\n"
"tmp2 = ops.Scalar(\"DT_FLOAT(0.00000000000000000000000000000000000001175494350822)\")\n"
"tmp3 = ops.Scalar(\"DT_FLOAT(274877906944.000)\")\n"
"tmp4 = ops.Scalar(\"DT_FLOAT(17592186044416.000)\")\n"
"tmp5 = ops.Scalar(\"DT_FLOAT(17592186044416.000)\")\n"
"tmp6 = ops.Broadcast(tmp2, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
"tmp8 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp9 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp10 = ops.Scalar(\"DT_FLOAT(0)\")\n"
"tmp11 = ops.Broadcast(tmp10, \"[]->[d0, d1, d2]\")\n"
"tmp12 = ops.Minimum(tmp1, tmp6)\n"
"tmp13 = ops.Maximum(tmp12, tmp11)\n"
"tmp14 = ops.Mul(tmp13, tmp7)\n"
"tmp15 = ops.Mul(tmp14, tmp9)\n"
"tmp16 = ops.Mul(tmp15, tmp9)\n"
"tmp17 = ops.Mul(tmp0, tmp16)\n"
"tmp18 = ops.Store(\"ReluGrad_0:0\", tmp17)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringReluGradINT32) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  auto nodeptr1 = cg->FindNode("gradients");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT32);
  tmp_desc->SetOriginDataType(DT_INT32);
  auto nodeptr2 = cg->FindNode("features");
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_INT32);
  tmp_desc2->SetOriginDataType(DT_INT32);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"gradients:0\")\n"
"tmp1 = ops.Load(\"features:0\")\n"
"tmp2 = ops.Scalar(\"DT_INT32(1)\")\n"
"tmp3 = ops.Scalar(\"DT_INT32(1)\")\n"
"tmp4 = ops.Scalar(\"DT_INT32(1)\")\n"
"tmp5 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp6 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp8 = ops.Scalar(\"DT_INT32(0)\")\n"
"tmp9 = ops.Broadcast(tmp8, \"[]->[d0, d1, d2]\")\n"
"tmp10 = ops.Minimum(tmp1, tmp5)\n"
"tmp11 = ops.Maximum(tmp10, tmp9)\n"
"tmp12 = ops.Mul(tmp11, tmp6)\n"
"tmp13 = ops.Mul(tmp12, tmp7)\n"
"tmp14 = ops.Mul(tmp0, tmp13)\n"
"tmp15 = ops.Store(\"ReluGrad_0:0\", tmp14)\n");
}

//TEST_F(LoopNodeLoweringUT, LoweringReluGradINT64) {
//  [this]() {
//    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
//    gradients.SetSymbolShape({"s0", "s1", "s2"});
//    auto features = es_graph_->CreateInput(1, "features", nullptr);
//    features.SetSymbolShape({"s0", "s1", "s2"});
//    auto tensor = es::ReluGrad(gradients, features);
//    tensor.SetSymbolShape({"s0", "s1", "s2"});
//    es_graph_->SetOutput(tensor, 0);
//  }();
//
//  auto graph = es_graph_->Build();
//  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
//  auto node = cg->FindNode("ReluGrad_0");
//  ASSERT_NE(node, nullptr);
//
//  auto nodeptr1 = cg->FindNode("gradients");
//  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
//  tmp_desc->SetDataType(DT_INT64);
//  tmp_desc->SetOriginDataType(DT_INT64);
//  auto nodeptr2 = cg->FindNode("features");
//  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
//  tmp_desc2->SetDataType(DT_INT64);
//  tmp_desc2->SetOriginDataType(DT_INT64);
//
//  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
//  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
//  ASSERT_FALSE(kernel.IsExternKernel());
//  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"gradients:0\")\n"
//"tmp1 = ops.Load(\"features:0\")\n"
//"tmp2 = ops.Scalar(\"DT_INT64(0)\")\n"
//"tmp3 = ops.Scalar(\"DT_INT64(0)\")\n"
//"tmp4 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
//"tmp5 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
//"tmp6 = ops.Le(tmp1, tmp5)\n"
//"tmp7 = ops.Where(tmp6, tmp5, tmp0)\n"
//"tmp8 = ops.Store(\"ReluGrad_0:0\", tmp7)\n");
//}

TEST_F(LoopNodeLoweringUT, LoweringReluGradUINT8) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  auto nodeptr1 = cg->FindNode("gradients");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_UINT8);
  tmp_desc->SetOriginDataType(DT_UINT8);
  auto nodeptr2 = cg->FindNode("features");
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_UINT8);
  tmp_desc2->SetOriginDataType(DT_UINT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringReluGradBF16) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  auto nodeptr1 = cg->FindNode("gradients");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_BF16);
  tmp_desc->SetOriginDataType(DT_BF16);
  auto nodeptr2 = cg->FindNode("features");
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_BF16);
  tmp_desc2->SetOriginDataType(DT_BF16);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringRsqrtGrad) {
  [this]() {
    auto y = es_graph_->CreateInput(0, "y", nullptr);
    y.SetSymbolShape({"s0", "s1", "s2"});
    auto dy = es_graph_->CreateInput(1, "dy", nullptr);
    dy.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::RsqrtGrad(y, dy);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("RsqrtGrad_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"y:0\")\n"
            "tmp1 = ops.Load(\"y:0\")\n"
            "tmp2 = ops.Load(\"y:0\")\n"
            "tmp3 = ops.Load(\"dy:0\")\n"
            "tmp4 = ops.Scalar(\"DT_FLOAT(-0.5)\")\n"
            "tmp5 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
            "tmp6 = ops.Mul(tmp2, tmp2)\n"
            "tmp7 = ops.Mul(tmp2, tmp6)\n"
            "tmp8 = ops.Mul(tmp3, tmp7)\n"
            "tmp9 = ops.Mul(tmp5, tmp8)\n"
            "tmp10 = ops.Store(\"RsqrtGrad_0:0\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringMuls) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::Muls(x, 1.5);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("Muls_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Scalar(\"DT_FLOAT(1.5000000)\")\n"
            "tmp2 = ops.Broadcast(tmp1, \"[]->[d0, d1, d2]\")\n"
            "tmp3 = ops.Mul(tmp0, tmp2)\n"
            "tmp4 = ops.Store(\"Muls_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringAxpy) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::Axpy(x, x, 0.8);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("Axpy_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Load(\"x:0\")\n"
            "tmp2 = ops.Axpy(tmp0, tmp1, 0.8)\n"
            "tmp3 = ops.Store(\"Axpy_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringSigmoidGrad) {
  [this]() {
    auto y = es_graph_->CreateInput(0, "y", nullptr);
    y.SetSymbolShape({"s0", "s1", "s2"});
    auto dy = es_graph_->CreateInput(1, "dy", nullptr);
    dy.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::SigmoidGrad(y, dy);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("SigmoidGrad_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"y:0\")\n"
            "tmp1 = ops.Scalar(\"DT_FLOAT(1)\")\n"
            "tmp2 = ops.Sub(tmp1, tmp0)\n"
            "tmp3 = ops.Load(\"y:0\")\n"
            "tmp4 = ops.Mul(tmp3, tmp2)\n"
            "tmp5 = ops.Load(\"dy:0\")\n"
            "tmp6 = ops.Mul(tmp4, tmp5)\n"
            "tmp7 = ops.Store(\"SigmoidGrad_0:0\", tmp6)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringFill) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto tmp = es::Fill(data0, data1);
    tmp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tmp = cg->FindNode("Fill_0");
  ASSERT_NE(tmp, nullptr);
  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsRealizedPersistent());
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data1:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d1, d2]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Store(\"Fill_0:0\", tmp1)\n");
}
TEST_F(LoopNodeLoweringUT, LoweringAddV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto tmp = es::AddV2(data0, data1);
    tmp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tmp = cg->FindNode("AddV2_0");
  ASSERT_NE(tmp, nullptr);
  for (auto &node : cg->GetAllNodes()) {
    ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);
  }
  auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Add(tmp0, tmp1)\n"
            "tmp3 = ops.Store(\"AddV2_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringUnsqueeze) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto unsqueeze = es::Unsqueeze(data0, {0, 1});
    unsqueeze.SetSymbolShape({"1", "1", "s0", "2"});
    es_graph_->SetOutput(unsqueeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto unsqueeze = cg->FindFirstNodeMatchType("Unsqueeze");
  ASSERT_NE(unsqueeze, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(unsqueeze->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 0)\n"
            "tmp2 = ops.Unsqueeze(tmp1, 1)\n"
            "tmp3 = ops.Store(\"Unsqueeze_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringSqueeze) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "2", "1", "1"});
    auto squeeze = es::Squeeze(data0, {1, 3, -1});
    squeeze.SetSymbolShape({"s0", "2"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeeze = cg->FindFirstNodeMatchType("Squeeze");
  ASSERT_NE(squeeze, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(squeeze->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 1)\n"
            "tmp2 = ops.Squeeze(tmp1, 2)\n"
            "tmp3 = ops.Squeeze(tmp2, 2)\n"
            "tmp4 = ops.Store(\"Squeeze_0:0\", tmp3)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringSqueezeLowerAxisIsNull) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s1"});
    auto squeezed = es::Squeeze(data0, {});
    squeezed.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(squeezed, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeezed = cg->FindNode("Squeeze_0");
  ASSERT_NE(squeezed, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(squeezed->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 1)\n"
            "tmp2 = ops.Store(\"Squeeze_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringDivNoNan) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "6", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3", "0"});
    auto divnonan = es::DivNoNan(data0, data1);
    divnonan.SetSymbolShape({"1", "2", "0"});
    es_graph_->SetOutput(divnonan, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto divnonan = cg->FindFirstNodeMatchType("DivNoNan");
  ASSERT_NE(divnonan, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(divnonan->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Scalar(\"DT_FLOAT(0)\")\n"
            "tmp4 = ops.Scalar(\"DT_FLOAT(0)\")\n"
            "tmp5 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
            "tmp6 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
            "tmp7 = ops.Eq(tmp2, tmp6)\n"
            "tmp8 = ops.Div(tmp0, tmp2)\n"
            "tmp9 = ops.Where(tmp7, tmp6, tmp8)\n"
            "tmp10 = ops.Store(\"DivNoNan_0:0\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringLeakyRelu) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "6", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3", "0"});
    auto leakyrelu = es::LeakyRelu(data0);
    leakyrelu.SetSymbolShape({"1", "2", "0"});
    es_graph_->SetOutput(leakyrelu, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto leakyrelu = cg->FindFirstNodeMatchType("LeakyRelu");
  ASSERT_NE(leakyrelu, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(leakyrelu->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.LeakyRelu(tmp0, 0)\n"
            "tmp2 = ops.Store(\"LeakyRelu_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringLeakyReluGrad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto leakyrelu = es::LeakyReluGrad(data0, data1,0.0);
    leakyrelu.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(leakyrelu, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto leakyrelugrad = cg->FindFirstNodeMatchType("LeakyReluGrad");
  ASSERT_NE(leakyrelugrad, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(leakyrelugrad->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"data0:0\")\n"
"tmp1 = ops.Load(\"data0:0\")\n"
"tmp2 = ops.Load(\"data1:0\")\n"
"tmp3 = ops.Scalar(\"DT_FLOAT(0.0)\")\n"
"tmp4 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
"tmp5 = ops.Scalar(\"DT_FLOAT(0.0000000)\")\n"
"tmp6 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Mul(tmp1, tmp6)\n"
"tmp8 = ops.Gt(tmp2, tmp4)\n"
"tmp9 = ops.Where(tmp8, tmp1, tmp7)\n"
"tmp10 = ops.Store(\"LeakyReluGrad_0:0\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, LogLowering) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto log = es::Log(x, 2.0, 2.0, 1.0);

    log.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(log, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto log = cg->FindNode("Log_0");
  ASSERT_NE(log, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(log), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(log->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"x:0\")\n"
"tmp1 = ops.Scalar(\"DT_FLOAT(2.0000000)\")\n"
"tmp2 = ops.Broadcast(tmp1, \"[]->[d0, d1, d2]\")\n"
"tmp3 = ops.Mul(tmp0, tmp2)\n"
"tmp4 = ops.Scalar(\"DT_FLOAT(1.0000000)\")\n"
"tmp5 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp6 = ops.Add(tmp3, tmp5)\n"
"tmp7 = ops.Ln(tmp6)\n"
"tmp8 = ops.Scalar(\"DT_FLOAT(1.44269502162933349609)\")\n"
"tmp9 = ops.Broadcast(tmp8, \"[]->[d0, d1, d2]\")\n"
"tmp10 = ops.Mul(tmp7, tmp9)\n"
"tmp11 = ops.Store(\"Log_0:0\", tmp10)\n");
}

TEST_F(LoopNodeLoweringUT, LogLoweringDefault) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto log = es::Log(x);

    log.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(log, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto log = cg->FindNode("Log_0");
  ASSERT_NE(log, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(log), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(log->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"x:0\")\n"
"tmp1 = ops.Ln(tmp0)\n"
"tmp2 = ops.Store(\"Log_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringPackLastDim_Fuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto pack = es::Pack({data0, data0}, 2, 2);
    pack.SetSymbolShape({"s0", "2", "2"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetNamePtr() << std::endl;
  }
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(pack->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringPackLastDim_NoFuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto pack =
        es::Pack({data0, data0, data0, data0, data0, data0, data0, data0,
                  data0, data0, data0, data0, data0, data0, data0, data0,
                  data0}, 2, 17);
    data0.SetSymbolShape({"s0", "2"});
    pack.SetSymbolShape({"s0", "2", "2"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetNamePtr() << std::endl;
  }
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(pack->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, LoweringPackNonLastDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto pack = es::Pack({data0, data0}, 1, 2);
    pack.SetSymbolShape({"s0", "2", "2"});
    es_graph_->SetOutput(pack, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  for (const auto &node : cg->GetAllNodes()) {
    std::cout << node->GetNamePtr() << std::endl;
  }
  auto pack = cg->FindNode("Pack_0");
  ASSERT_NE(pack, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(pack->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.StoreConcat(\"Pack_0:0\", [tmp0, tmp1], concat_dim=1)\n");
}


TEST_F(LoopNodeLoweringUT, ReshapeLowering1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"5"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "1", "s1", "s2", "1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(reshape->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 1)\n"
            "tmp2 = ops.Unsqueeze(tmp1, 4)\n"
            "tmp3 = ops.Store(\"Reshape_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, ReshapeLowering2) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "1", "s1", "1"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(reshape->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 1)\n"
            "tmp2 = ops.Squeeze(tmp1, 2)\n"
            "tmp3 = ops.Store(\"Reshape_0:0\", tmp2)\n");
}

TEST_F(LoopNodeLoweringUT, ReshapeLowering3) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(3), Symbol(2), Symbol(3)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"3"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(6), Symbol(3)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(reshape->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
//   EXPECT_EQ(kernel.Readable(),
//   "tmp0 = ops.Load(\"x:0\")\n"
// "tmp1 = ops.Reshape(tmp0, [2, 3, 2, 3] -> [2, 6, 3])\n"
// "tmp2 = ops.Store(\"Reshape_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, ReshapeLowering4) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(6), Symbol(3)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(3), Symbol(2), Symbol(3)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(reshape->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
//   EXPECT_EQ(kernel.Readable(),
//   "tmp0 = ops.Load(\"x:0\")\n"
// "tmp1 = ops.Reshape(tmp0, [2, 6, 3] -> [2, 3, 2, 3])\n"
// "tmp2 = ops.Store(\"Reshape_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, ReshapeLowering5) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(128), Symbol(16)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"3"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(128), Symbol(1), Symbol(16)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(reshape->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
    EXPECT_EQ(kernel.Readable(),
"tmp0 = ops.Load(\"x:0\")\n"
"tmp1 = ops.Unsqueeze(tmp0, 1)\n"
"tmp2 = ops.Store(\"Reshape_0:0\", tmp1)\n");
}

TEST_F(LoopNodeLoweringUT, ReshapeLoweringFailure1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"3"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1", "s3"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_NE(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, ReshapeLoweringFailure2) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_NE(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, ReshapeLoweringFailure3) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_NE(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, ReshapeLoweringFailure4) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(x, shape, 0, 1);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reshape = cg->FindNode("Reshape_0");
  ASSERT_NE(reshape, nullptr);

  ASSERT_NE(LoweringManager::Lowering(reshape), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, LoweringExpandDims) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{2});
    auto expanddims = es::ExpandDims(data0, axis);
    expanddims.SetSymbolShape({"s0", "2", "1"});
    auto axis1 = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{2});
    auto expanddims1 = es::ExpandDims(data0, axis1);
    expanddims1.SetSymbolShape({"s0", "2", "1"});
    auto axis2 = CreateConst(*es_graph_, ge::DT_INT16, {1}, std::vector<int16_t>{2});
    auto expanddims2 = es::ExpandDims(data0, axis2);
    expanddims2.SetSymbolShape({"s0", "2", "1"});
    es_graph_->SetOutput(expanddims, 0);
    es_graph_->SetOutput(expanddims1, 1);
    es_graph_->SetOutput(expanddims2, 2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto expanddims = cg->FindNode("ExpandDims_1");
  ASSERT_NE(expanddims, nullptr);
  auto expanddims1 = cg->FindNode("ExpandDims_3");
  ASSERT_NE(expanddims1, nullptr);
  auto expanddims2 = cg->FindNode("ExpandDims_5");
  ASSERT_NE(expanddims2, nullptr);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(expanddims->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 2)\n"
            "tmp2 = ops.Store(\"ExpandDims_1:0\", tmp1)\n");

  auto kernel1 = ge::loop::GetKernelBox(expanddims1->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel1.IsExternKernel());
  EXPECT_EQ(kernel1.Readable(),
          "tmp0 = ops.Load(\"data0:0\")\n"
          "tmp1 = ops.Unsqueeze(tmp0, 2)\n"
          "tmp2 = ops.Store(\"ExpandDims_3:0\", tmp1)\n");

  auto kernel2 = ge::loop::GetKernelBox(expanddims2->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel2.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, SelectLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1"});
    auto select = es::Select(data0, data1, data2);
    select.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(select, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto select = cg->FindNode("Select_0");
  ASSERT_NE(select, nullptr);
  auto nodeptr1 = cg->FindNode("data0");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_UINT8);
  tmp_desc->SetOriginDataType(DT_UINT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(select->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 1)\n"
            "tmp2 = ops.Broadcast(tmp1, \"[d0, 1]->[d0, d1]\")\n"
            "tmp3 = ops.Load(\"data1:0\")\n"
            "tmp4 = ops.Load(\"data2:0\")\n"
            "tmp5 = ops.Where(tmp2, tmp3, tmp4)\n"
            "tmp6 = ops.Store(\"Select_0:0\", tmp5)\n");
}

TEST_F(LoopNodeLoweringUT, ApplyAdagradDLowering) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto accum = es_graph_->CreateInput(1, "accum", nullptr);
    accum.SetSymbolShape({"s0", "s1", "s2"});
    auto lr = es_graph_->CreateInput(2, "lr", nullptr);
    lr.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdagradD(var, accum, lr, grad);
    rst.accum.SetSymbolShape({"s0", "s1", "s2"});
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.accum, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdagradD_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
  EXPECT_EQ(asc_kernel.Readable(),"tmp0 = ops.Load(\"accum:0\")\n"
"tmp1 = ops.Load(\"grad:0\")\n"
"tmp2 = ops.Load(\"grad:0\")\n"
"tmp3 = ops.Load(\"grad:0\")\n"
"tmp4 = ops.Mul(tmp3, tmp3)\n"
"tmp5 = ops.Add(tmp0, tmp4)\n"
"tmp6 = ops.Load(\"var:0\")\n"
"tmp7 = ops.Load(\"lr:0\")\n"
"tmp8 = ops.Broadcast(tmp7, \"[]->[d0, d1, d2]\")\n"
"tmp9 = ops.Sqrt(tmp5)\n"
"tmp10 = ops.Mul(tmp8, tmp3)\n"
"tmp11 = ops.Div(tmp10, tmp9)\n"
"tmp12 = ops.Sub(tmp6, tmp11)\n"
"tmp13 = ops.Store(\"ApplyAdagradD_0:0\", tmp12)\n");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  ASSERT_FALSE(asc_kernel1.IsExternKernel());
  EXPECT_EQ(asc_kernel1.Readable(), "tmp0 = ops.Load(\"accum:0\")\n"
"tmp1 = ops.Load(\"grad:0\")\n"
"tmp2 = ops.Load(\"grad:0\")\n"
"tmp3 = ops.Mul(tmp2, tmp2)\n"
"tmp4 = ops.Add(tmp0, tmp3)\n"
"tmp5 = ops.Store(\"ApplyAdagradD_0:1\", tmp4)\n");
}

TEST_F(LoopNodeLoweringUT, ApplyAdagradDLowering1) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto accum = es_graph_->CreateInput(1, "accum", nullptr);
    accum.SetSymbolShape({"s0", "s1", "s2"});
    auto lr = es_graph_->CreateInput(2, "lr", nullptr);
    lr.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdagradD(var, accum, lr, grad, false);
    rst.accum.SetSymbolShape({"s0", "s1", "s2"});
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.accum, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdagradD_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
  EXPECT_EQ(asc_kernel.Readable(), "tmp0 = ops.Load(\"accum:0\")\n"
"tmp1 = ops.Load(\"grad:0\")\n"
"tmp2 = ops.Load(\"var:0\")\n"
"tmp3 = ops.Load(\"lr:0\")\n"
"tmp4 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
"tmp5 = ops.Sqrt(tmp0)\n"
"tmp6 = ops.Mul(tmp4, tmp1)\n"
"tmp7 = ops.Div(tmp6, tmp5)\n"
"tmp8 = ops.Sub(tmp2, tmp7)\n"
"tmp9 = ops.Store(\"ApplyAdagradD_0:0\", tmp8)\n");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  ASSERT_FALSE(asc_kernel1.IsExternKernel());
  EXPECT_EQ(asc_kernel1.Readable(), "tmp0 = ops.Load(\"accum:0\")\n"
"tmp1 = ops.Store(\"ApplyAdagradD_0:1\", tmp0)\n");
}

TEST_F(LoopNodeLoweringUT, ApplyAdamDLowering) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto m = es_graph_->CreateInput(1, "m", nullptr);
    m.SetSymbolShape({"s0", "s1", "s2"});
    auto v = es_graph_->CreateInput(2, "v", nullptr);
    v.SetSymbolShape({"s0", "s1", "s2"});
    auto beta1_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.08});
    beta1_power.SetSymbolShape({});
    auto beta2_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.25});
    beta2_power.SetSymbolShape({});
    auto lr = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.13});
    lr.SetSymbolShape({});
    auto beta1 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.77});
    beta1.SetSymbolShape({});
    auto beta2 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.05});
    beta2.SetSymbolShape({});
    auto epsilon = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.53});
    epsilon.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad);
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    rst.m.SetSymbolShape({"s0", "s1", "s2"});
    rst.v.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.m, 1);
    es_graph_->SetOutput(rst.v, 2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdamD_6");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
  EXPECT_EQ(asc_kernel.Readable(), "tmp0 = ops.Load(\"var:0\")\n"
"tmp1 = ops.Load(\"m:0\")\n"
"tmp2 = ops.Load(\"m:0\")\n"
"tmp3 = ops.Load(\"v:0\")\n"
"tmp4 = ops.Load(\"v:0\")\n"
"tmp5 = ops.Scalar(\"DT_FLOAT(0.230000)\")\n"
"tmp6 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Scalar(\"DT_FLOAT(0.950000)\")\n"
"tmp8 = ops.Broadcast(tmp7, \"[]->[d0, d1, d2]\")\n"
"tmp9 = ops.Scalar(\"DT_FLOAT(5.29999971389770507812e-01)\")\n"
"tmp10 = ops.Broadcast(tmp9, \"[]->[d0, d1, d2]\")\n"
"tmp11 = ops.Load(\"grad:0\")\n"
"tmp12 = ops.Load(\"grad:0\")\n"
"tmp13 = ops.Load(\"grad:0\")\n"
"tmp14 = ops.Sub(tmp13, tmp2)\n"
"tmp15 = ops.Mul(tmp6, tmp14)\n"
"tmp16 = ops.Add(tmp2, tmp15)\n"
"tmp17 = ops.Scalar(\"DT_FLOAT(0.122373)\")\n"
"tmp18 = ops.Broadcast(tmp17, \"[]->[d0, d1, d2]\")\n"
"tmp19 = ops.Mul(tmp13, tmp13)\n"
"tmp20 = ops.Sub(tmp19, tmp4)\n"
"tmp21 = ops.Mul(tmp8, tmp20)\n"
"tmp22 = ops.Add(tmp4, tmp21)\n"
"tmp23 = ops.Sqrt(tmp22)\n"
"tmp24 = ops.Add(tmp10, tmp23)\n"
"tmp25 = ops.Div(tmp16, tmp24)\n"
"tmp26 = ops.Mul(tmp18, tmp25)\n"
"tmp27 = ops.Sub(tmp0, tmp26)\n"
"tmp28 = ops.Store(\"ApplyAdamD_6:0\", tmp27)\n");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  ASSERT_FALSE(asc_kernel1.IsExternKernel());
  EXPECT_EQ(asc_kernel1.Readable(), "tmp0 = ops.Load(\"m:0\")\n"
"tmp1 = ops.Load(\"m:0\")\n"
"tmp2 = ops.Scalar(\"DT_FLOAT(0.230000)\")\n"
"tmp3 = ops.Broadcast(tmp2, \"[]->[d0, d1, d2]\")\n"
"tmp4 = ops.Load(\"grad:0\")\n"
"tmp5 = ops.Sub(tmp4, tmp1)\n"
"tmp6 = ops.Mul(tmp3, tmp5)\n"
"tmp7 = ops.Add(tmp1, tmp6)\n"
"tmp8 = ops.Store(\"ApplyAdamD_6:1\", tmp7)\n");

  loop::KernelBox asc_kernel2 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(2));
  ASSERT_FALSE(asc_kernel2.IsExternKernel());
  EXPECT_EQ(asc_kernel2.Readable(), "tmp0 = ops.Load(\"v:0\")\n"
"tmp1 = ops.Load(\"v:0\")\n"
"tmp2 = ops.Scalar(\"DT_FLOAT(0.950000)\")\n"
"tmp3 = ops.Broadcast(tmp2, \"[]->[d0, d1, d2]\")\n"
"tmp4 = ops.Load(\"grad:0\")\n"
"tmp5 = ops.Load(\"grad:0\")\n"
"tmp6 = ops.Mul(tmp5, tmp5)\n"
"tmp7 = ops.Sub(tmp6, tmp1)\n"
"tmp8 = ops.Mul(tmp3, tmp7)\n"
"tmp9 = ops.Add(tmp1, tmp8)\n"
"tmp10 = ops.Store(\"ApplyAdamD_6:2\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, ApplyAdamDLowering1) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto m = es_graph_->CreateInput(1, "m", nullptr);
    m.SetSymbolShape({"s0", "s1", "s2"});
    auto v = es_graph_->CreateInput(2, "v", nullptr);
    v.SetSymbolShape({"s0", "s1", "s2"});
    auto beta1_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.08});
    beta1_power.SetSymbolShape({});
    auto beta2_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.25});
    beta2_power.SetSymbolShape({});
    auto lr = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.13});
    lr.SetSymbolShape({});
    auto beta1 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.77});
    beta1.SetSymbolShape({});
    auto beta2 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.05});
    beta2.SetSymbolShape({});
    auto epsilon = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.53});
    epsilon.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, false, true);
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    rst.m.SetSymbolShape({"s0", "s1", "s2"});
    rst.v.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.m, 1);
    es_graph_->SetOutput(rst.v, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdamD_6");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
  EXPECT_EQ(asc_kernel.Readable(), "tmp0 = ops.Load(\"var:0\")\n"
"tmp1 = ops.Load(\"m:0\")\n"
"tmp2 = ops.Load(\"m:0\")\n"
"tmp3 = ops.Load(\"v:0\")\n"
"tmp4 = ops.Load(\"v:0\")\n"
"tmp5 = ops.Scalar(\"DT_FLOAT(7.69999980926513671875e-01)\")\n"
"tmp6 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Scalar(\"DT_FLOAT(0.230000)\")\n"
"tmp8 = ops.Scalar(\"DT_FLOAT(0.230000)\")\n"
"tmp9 = ops.Broadcast(tmp8, \"[]->[d0, d1, d2]\")\n"
"tmp10 = ops.Broadcast(tmp8, \"[]->[d0, d1, d2]\")\n"
"tmp11 = ops.Scalar(\"DT_FLOAT(0.950000)\")\n"
"tmp12 = ops.Broadcast(tmp11, \"[]->[d0, d1, d2]\")\n"
"tmp13 = ops.Scalar(\"DT_FLOAT(5.29999971389770507812e-01)\")\n"
"tmp14 = ops.Broadcast(tmp13, \"[]->[d0, d1, d2]\")\n"
"tmp15 = ops.Load(\"grad:0\")\n"
"tmp16 = ops.Load(\"grad:0\")\n"
"tmp17 = ops.Load(\"grad:0\")\n"
"tmp18 = ops.Load(\"grad:0\")\n"
"tmp19 = ops.Sub(tmp18, tmp2)\n"
"tmp20 = ops.Mul(tmp10, tmp19)\n"
"tmp21 = ops.Add(tmp2, tmp20)\n"
"tmp22 = ops.Scalar(\"DT_FLOAT(0.122373)\")\n"
"tmp23 = ops.Broadcast(tmp22, \"[]->[d0, d1, d2]\")\n"
"tmp24 = ops.Mul(tmp18, tmp18)\n"
"tmp25 = ops.Sub(tmp24, tmp4)\n"
"tmp26 = ops.Mul(tmp12, tmp25)\n"
"tmp27 = ops.Add(tmp4, tmp26)\n"
"tmp28 = ops.Sqrt(tmp27)\n"
"tmp29 = ops.Add(tmp14, tmp28)\n"
"tmp30 = ops.Mul(tmp10, tmp18)\n"
"tmp31 = ops.Mul(tmp21, tmp6)\n"
"tmp32 = ops.Add(tmp31, tmp30)\n"
"tmp33 = ops.Div(tmp32, tmp29)\n"
"tmp34 = ops.Mul(tmp23, tmp33)\n"
"tmp35 = ops.Sub(tmp0, tmp34)\n"
"tmp36 = ops.Store(\"ApplyAdamD_6:0\", tmp35)\n");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  ASSERT_FALSE(asc_kernel1.IsExternKernel());
  EXPECT_EQ(asc_kernel1.Readable(), "tmp0 = ops.Load(\"m:0\")\n"
"tmp1 = ops.Load(\"m:0\")\n"
"tmp2 = ops.Scalar(\"DT_FLOAT(0.230000)\")\n"
"tmp3 = ops.Broadcast(tmp2, \"[]->[d0, d1, d2]\")\n"
"tmp4 = ops.Load(\"grad:0\")\n"
"tmp5 = ops.Sub(tmp4, tmp1)\n"
"tmp6 = ops.Mul(tmp3, tmp5)\n"
"tmp7 = ops.Add(tmp1, tmp6)\n"
"tmp8 = ops.Store(\"ApplyAdamD_6:1\", tmp7)\n");

  loop::KernelBox asc_kernel2 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(2));
  ASSERT_FALSE(asc_kernel2.IsExternKernel());
  EXPECT_EQ(asc_kernel2.Readable(), "tmp0 = ops.Load(\"v:0\")\n"
"tmp1 = ops.Load(\"v:0\")\n"
"tmp2 = ops.Scalar(\"DT_FLOAT(0.950000)\")\n"
"tmp3 = ops.Broadcast(tmp2, \"[]->[d0, d1, d2]\")\n"
"tmp4 = ops.Load(\"grad:0\")\n"
"tmp5 = ops.Load(\"grad:0\")\n"
"tmp6 = ops.Mul(tmp5, tmp5)\n"
"tmp7 = ops.Sub(tmp6, tmp1)\n"
"tmp8 = ops.Mul(tmp3, tmp7)\n"
"tmp9 = ops.Add(tmp1, tmp8)\n"
"tmp10 = ops.Store(\"ApplyAdamD_6:2\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, ApplyAdamDLowering2_beta1_power_eq_1) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto m = es_graph_->CreateInput(1, "m", nullptr);
    m.SetSymbolShape({"s0", "s1", "s2"});
    auto v = es_graph_->CreateInput(2, "v", nullptr);
    v.SetSymbolShape({"s0", "s1", "s2"});
    auto beta1_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{1});
    beta1_power.SetSymbolShape({});
    auto beta2_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.25});
    beta2_power.SetSymbolShape({});
    auto lr = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.13});
    lr.SetSymbolShape({});
    auto beta1 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.77});
    beta1.SetSymbolShape({});
    auto beta2 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.05});
    beta2.SetSymbolShape({});
    auto epsilon = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.53});
    epsilon.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, false, true);
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    rst.m.SetSymbolShape({"s0", "s1", "s2"});
    rst.v.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.m, 1);
    es_graph_->SetOutput(rst.v, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdamD_6");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsExternKernel());
}

TEST_F(LoopNodeLoweringUT, ApplyGradientDescentLowering) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto alpha = es_graph_->CreateInput(1, "alpha", nullptr);
    alpha.SetSymbolShape({});
    auto delta = es_graph_->CreateInput(2, "delta", nullptr);
    delta.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyGradientDescent(var, alpha, delta);
    rst.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyGradientDescent_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
  EXPECT_EQ(asc_kernel.Readable(), "tmp0 = ops.Load(\"var:0\")\n"
"tmp1 = ops.Load(\"alpha:0\")\n"
"tmp2 = ops.Broadcast(tmp1, \"[]->[d0, d1, d2]\")\n"
"tmp3 = ops.Load(\"delta:0\")\n"
"tmp4 = ops.Mul(tmp3, tmp2)\n"
"tmp5 = ops.Sub(tmp0, tmp4)\n"
"tmp6 = ops.Store(\"ApplyGradientDescent_0:0\", tmp5)\n");
}

class TestCounter : public Counter {
public:
  TestCounter() = default;
  virtual ~TestCounter() = default;
  virtual int64_t NextId() { return id_++;};
private:
  int64_t id_ = 0;
};

TEST_F(LoopNodeLoweringUT, SimpleLoweringName) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(x);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto sum = es::ReduceSumD(abs1, {-1}, true);
    es_graph_->SetOutput(sum, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  CounterPtr counter = new TestCounter;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, kAscBackendFuseConfig, counter), GRAPH_SUCCESS);
  delete counter;
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringByPass) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 0, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_NE(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringInt32) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 0, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("x");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT32);
  input0_desc->SetOriginDataType(DT_INT32);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_NE(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
}

TEST_F(LoopNodeLoweringUT, TransposeLowering1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"16", "32", "64"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 0, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"32", "16", "64"});
    es_graph_->SetOutput(transpose, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1, d2]->[d1, d0, d2]\")\n"
            "tmp2 = ops.Store(\"Transpose_1:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLowering2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"16", "64", "1"});
    data1.SetSymbolShape({"16", "64", "32"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"16", "64", "32"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"16", "32", "64"});
    es_graph_->SetOutput(transpose, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_2");
  ASSERT_NE(transpose, nullptr);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto add = cg->FindNode("Add_0");
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Type(), ge::loop::FuseType::kPointwise);
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Broadcast(tmp0, \"[d0, d1, 1]->[d0, d1, d2]\")\n"
            "tmp2 = ops.Load(\"data1:0\")\n"
            "tmp3 = ops.Add(tmp1, tmp2)\n"
            "tmp4 = ops.Store(\"Add_0:0\", tmp3)\n");
  kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Add_0:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1, d2]->[d0, d2, d1]\")\n"
            "tmp2 = ops.Store(\"Transpose_2:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLowering3) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"16", "32", "64"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"16", "32", "64"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(abs0, perms);
    transpose.SetSymbolShape({"16", "64", "32"});
    auto abs1 = es::Abs(transpose);
    abs1.SetSymbolShape({"16", "64", "32"});
    es_graph_->SetOutput(abs1, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto abs0 = cg->FindNode("Abs_0");
  auto transpose = cg->FindNode("Transpose_2");
  auto abs = cg->FindNode("Abs_3");
  ASSERT_NE(abs0, nullptr);
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(abs, nullptr);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(abs0->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_0:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Abs_0:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1, d2]->[d0, d2, d1]\")\n"
            "tmp2 = ops.Store(\"Transpose_2:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Transpose_2:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_3:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, InvalidTransposeLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"16", "32", "1"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"16", "32", "1"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(abs0, perms);
    transpose.SetSymbolShape({"16", "1", "32"});
    auto abs1 = es::Abs(transpose);
    abs1.SetSymbolShape({"16", "1", "32"});
    es_graph_->SetOutput(abs1, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto abs0 = cg->FindNode("Abs_0");
  auto transpose = cg->FindNode("Transpose_2");
  auto abs = cg->FindNode("Abs_3");
  ASSERT_NE(abs0, nullptr);
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(abs, nullptr);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(abs0->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_0:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Abs_0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 2)\n"
            "tmp2 = ops.Store(\"Transpose_2:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Transpose_2:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_3:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, InvalidTransposeLowering1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "32", "64", "128"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"1", "32", "64", "128"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{1, 2, 0, 3});
    auto transpose = es::Transpose(abs0, perms);
    transpose.SetSymbolShape({"32", "64", "1", "128"});
    auto abs1 = es::Abs(transpose);
    abs1.SetSymbolShape({"32", "64", "1", "128"});
    es_graph_->SetOutput(abs1, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto abs0 = cg->FindNode("Abs_0");
  auto transpose = cg->FindNode("Transpose_2");
  auto abs = cg->FindNode("Abs_3");
  ASSERT_NE(abs0, nullptr);
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(abs, nullptr);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(abs0->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_0:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Abs_0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 0)\n"
            "tmp2 = ops.Store(\"Transpose_2:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Transpose_2:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_3:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, InvalidTransposeLowering2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "1", "1"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"32", "1", "1"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose = es::Transpose(abs0, perms);
    transpose.SetSymbolShape({"1", "1", "32"});
    auto abs1 = es::Abs(transpose);
    abs1.SetSymbolShape({"1", "1", "32"});
    es_graph_->SetOutput(abs1, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto abs0 = cg->FindNode("Abs_0");
  auto transpose = cg->FindNode("Transpose_2");
  auto abs = cg->FindNode("Abs_3");
  ASSERT_NE(abs0, nullptr);
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(abs, nullptr);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(abs0->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_0:0\", tmp1)\n");

  kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Abs_0:0\")\n"
            "tmp1 = ops.Squeeze(tmp0, 2)\n"
            "tmp2 = ops.Squeeze(tmp1, 1)\n"
            "tmp3 = ops.Store(\"Transpose_2:0\", tmp2)\n");

  kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"Transpose_2:0\")\n"
            "tmp1 = ops.Abs(tmp0)\n"
            "tmp2 = ops.Store(\"Abs_3:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringPerm10) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"16", "32"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {2}, std::vector<int32_t>{1, 0});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"32", "16"});
    es_graph_->SetOutput(transpose, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1]->[d1, d0]\")\n"
            "tmp2 = ops.Store(\"Transpose_1:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringPerm0132) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"16", "32", "64", "128"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {4}, std::vector<int32_t>{0, 1, 3, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"16", "32", "128", "64"});
    es_graph_->SetOutput(transpose, 0);
  }();
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(transpose->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.Permute(tmp0, \"[d0, d1, d2, d3]->[d0, d1, d3, d2]\")\n"
            "tmp2 = ops.Store(\"Transpose_1:0\", tmp1)\n");
  ge::autofuse::AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
}

TEST_F(LoopNodeLoweringUT, TransposeLoweringDtypeNotSupport) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "data0", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT32, {4}, std::vector<int32_t>{0, 1, 3, 2});
    auto transpose = es::Transpose(x, perms);
    transpose.SetSymbolShape({"s0", "s1", "s3", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  setenv("ENABLE_LOWER_TRANSPOSE", "true", 1);
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT4);
  input0_desc->SetOriginDataType(DT_INT4);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ASSERT_EQ(LoweringManager::Lowering(transpose), GRAPH_FAILED);
  unsetenv("ENABLE_LOWER_TRANSPOSE");
}

TEST_F(LoopNodeLoweringUT, FusedMulAddNLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1"});
    auto fused_mul_addn = es::FusedMulAddN(data0, data1, data2);
    fused_mul_addn.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(fused_mul_addn, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto fused_mul_addn = cg->FindNode("FusedMulAddN_0");
  ASSERT_NE(fused_mul_addn, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(fused_mul_addn->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data1:0\")\n"
            "tmp2 = ops.Load(\"data2:0\")\n"
            "tmp3 = ops.Mul(tmp0, tmp2)\n"
            "tmp4 = ops.Add(tmp3, tmp1)\n"
            "tmp5 = ops.Store(\"FusedMulAddN_0:0\", tmp4)\n");
}

TEST_F(LoopNodeLoweringUT, LoweringDTypeUnsupported) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT64);
  input0_desc->SetOriginDataType(DT_INT64);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto abs = cg->FindNode("Abs_0");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_EQ(asc_kernel.IsExternKernel(), true);
}

TEST_F(LoopNodeLoweringUT, LoweringLoadSupportAbsUnsupported) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_INT16);
  input0_desc->SetOriginDataType(DT_INT16);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto abs = cg->FindNode("Abs_0");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  ASSERT_EQ(asc_kernel.IsExternKernel(), true);
}

TEST_F(LoopNodeLoweringUT, LoweringTileToSingleConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto multiples = es_graph_->CreateVector({1, 2});
    multiples.SetSymbolShape({"2"});
    auto tile = es::Tile(data0, multiples);
    tile.SetSymbolShape({"s0", "4"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tile = cg->FindNode("Tile_1");
  ASSERT_NE(tile, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(tile->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.StoreConcat(\"Tile_1:0\", [tmp0, tmp1], concat_dim=1)\n");
}

TEST_F(LoopNodeLoweringUT, L2LossLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto l2_loss = es::L2Loss(data0);
    l2_loss.SetSymbolShape({});
    es_graph_->SetOutput(l2_loss, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto l2_loss = cg->FindNode("L2Loss_0");
  ASSERT_NE(l2_loss, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(l2_loss->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"data0:0\")\n"
            "tmp1 = ops.Load(\"data0:0\")\n"
            "tmp2 = ops.Mul(tmp1, tmp1)\n"
            "tmp3 = ops.Scalar(\"DT_FLOAT(0.5)\")\n"
            "tmp4 = ops.Broadcast(tmp3, \"[]->[d0, d1]\")\n"
            "tmp5 = ops.Mul(tmp2, tmp4)\n"
            "tmp6 = ops.StoreReduction(\"L2Loss_0:0\", ops.Sum(tmp5, \"[d0, d1]->[]\"))\n");
}

TEST_F(LoopNodeLoweringUT, BNInferenceLowering) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto mean = es_graph_->CreateInput(1, "mean", nullptr);
    mean.SetSymbolShape({"s3"});
    auto variance = es_graph_->CreateInput(2, "variance", nullptr);
    variance.SetSymbolShape({"s3"});
    auto bninferenced = es::BNInferenceD(x, mean, variance);
    bninferenced.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(bninferenced, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("BNInferenceD_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"x:0\")\n"
"tmp1 = ops.Load(\"mean:0\")\n"
"tmp2 = ops.Broadcast(tmp1, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp3 = ops.Load(\"variance:0\")\n"
"tmp4 = ops.Broadcast(tmp3, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp5 = ops.Add(tmp0, tmp2)\n"
"tmp6 = ops.Mul(tmp4, tmp5)\n"
"tmp7 = ops.Store(\"BNInferenceD_0:0\", tmp6)\n");
}

TEST_F(LoopNodeLoweringUT, BNInferenceLowering1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto mean = es_graph_->CreateInput(1, "mean", nullptr);
    mean.SetSymbolShape({"s3"});
    auto variance = es_graph_->CreateInput(2, "variance", nullptr);
    variance.SetSymbolShape({"s3"});
    auto scale = es_graph_->CreateInput(3, "scale", nullptr);
    scale.SetSymbolShape({"s3"});
    auto bninferenced = es::BNInferenceD(x, mean, variance, scale);
    bninferenced.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(bninferenced, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("BNInferenceD_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"x:0\")\n"
"tmp1 = ops.Load(\"mean:0\")\n"
"tmp2 = ops.Broadcast(tmp1, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp3 = ops.Load(\"variance:0\")\n"
"tmp4 = ops.Broadcast(tmp3, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp5 = ops.Load(\"scale:0\")\n"
"tmp6 = ops.Broadcast(tmp5, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp7 = ops.Add(tmp0, tmp2)\n"
"tmp8 = ops.Mul(tmp4, tmp7)\n"
"tmp9 = ops.Mul(tmp8, tmp6)\n"
"tmp10 = ops.Store(\"BNInferenceD_0:0\", tmp9)\n");
}

TEST_F(LoopNodeLoweringUT, BNInferenceLowering2) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto mean = es_graph_->CreateInput(1, "mean", nullptr);
    mean.SetSymbolShape({"s3"});
    auto variance = es_graph_->CreateInput(2, "variance", nullptr);
    variance.SetSymbolShape({"s3"});
    auto scale = es_graph_->CreateInput(3, "scale", nullptr);
    scale.SetSymbolShape({"s3"});
    auto b = es_graph_->CreateInput(4, "b", nullptr);
    b.SetSymbolShape({"s3"});
    auto bninferenced = es::BNInferenceD(x, mean, variance, scale, b);
    bninferenced.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(bninferenced, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("BNInferenceD_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"x:0\")\n"
"tmp1 = ops.Load(\"mean:0\")\n"
"tmp2 = ops.Broadcast(tmp1, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp3 = ops.Load(\"variance:0\")\n"
"tmp4 = ops.Broadcast(tmp3, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp5 = ops.Load(\"scale:0\")\n"
"tmp6 = ops.Broadcast(tmp5, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp7 = ops.Load(\"b:0\")\n"
"tmp8 = ops.Broadcast(tmp7, \"[d3]->[d0, d1, d2, d3]\")\n"
"tmp9 = ops.Add(tmp0, tmp2)\n"
"tmp10 = ops.Mul(tmp4, tmp9)\n"
"tmp11 = ops.Mul(tmp10, tmp6)\n"
"tmp12 = ops.Add(tmp11, tmp8)\n"
"tmp13 = ops.Store(\"BNInferenceD_0:0\", tmp12)\n");
}

TEST_F(LoopNodeLoweringUT, SimpleClipByValueConstScalar) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"28", "28"});
    auto min = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{1.0});
    min.SetSymbolShape({});
    auto max = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{2.0});
    max.SetSymbolShape({});
    auto expanddims = es::ClipByValue(data0, min, max);
    expanddims.SetSymbolShape({"28", "28"});
    es_graph_->SetOutput(expanddims, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ClipByValue_2");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());

EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Scalar(\"DT_FLOAT(1.00000000000000000000e+00)\")\n"
            "tmp1 = ops.Unsqueeze(tmp0, 0)\n"
            "tmp2 = ops.Unsqueeze(tmp1, 1)\n"
            "tmp3 = ops.Scalar(\"DT_FLOAT(2.00000000000000000000e+00)\")\n"
            "tmp4 = ops.Unsqueeze(tmp3, 0)\n"
            "tmp5 = ops.Unsqueeze(tmp4, 1)\n"
            "tmp6 = ops.Broadcast(tmp2, \"[1, 1]->[d0, d1]\")\n"
            "tmp7 = ops.Broadcast(tmp5, \"[1, 1]->[d0, d1]\")\n"
            "tmp8 = ops.Load(\"data0:0\")\n"
            "tmp9 = ops.Minimum(tmp8, tmp7)\n"
            "tmp10 = ops.Maximum(tmp9, tmp6)\n"
            "tmp11 = ops.Store(\"ClipByValue_2:0\", tmp10)\n");
}

TEST_F(LoopNodeLoweringUT, ClipByValueWithTensorInput) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"2", "28"});
    auto min = es_graph_->CreateInput(1, "data1", nullptr);
    min.SetSymbolShape({"28"});
    auto max = es_graph_->CreateInput(2, "data2", nullptr);
    max.SetSymbolShape({"28"});
    auto expanddims = es::ClipByValue(data0, min, max);
    expanddims.SetSymbolShape({"2", "28"});
    es_graph_->SetOutput(expanddims, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ClipByValue_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());

  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"data1:0\")\n"
              "tmp1 = ops.Unsqueeze(tmp0, 0)\n"
              "tmp2 = ops.Load(\"data2:0\")\n"
              "tmp3 = ops.Unsqueeze(tmp2, 0)\n"
              "tmp4 = ops.Broadcast(tmp1, \"[1, d1]->[d0, d1]\")\n"
              "tmp5 = ops.Broadcast(tmp3, \"[1, d1]->[d0, d1]\")\n"
              "tmp6 = ops.Load(\"data0:0\")\n"
              "tmp7 = ops.Minimum(tmp6, tmp5)\n"
              "tmp8 = ops.Maximum(tmp7, tmp4)\n"
              "tmp9 = ops.Store(\"ClipByValue_0:0\", tmp8)\n");
}

#define TEST_F_LOWER_INST1(BACKENDOP, OP)                           \
  TEST_F(LoopNodeLoweringUT, Lowering##OP) {                        \
    [this]() {                                                      \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);     \
      data0.SetSymbolShape({"s0", "s1", "s2"});                     \
      auto tmp = es::OP(data0);                                     \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                       \
      es_graph_->SetOutput(tmp, 0);                                 \
    }();                                                            \
                                                                    \
    auto graph = es_graph_->Build();                                \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                \
    auto tmp = cg->FindNode(#OP "_0");                              \
    ASSERT_NE(tmp, nullptr);                                        \
    for (auto &node : cg->GetAllNodes()) {                          \
      ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);    \
    }                                                               \
    auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0)); \
    ASSERT_TRUE(kernel.IsRealizedPersistent());                     \
    ASSERT_FALSE(kernel.IsExternKernel());                          \
    EXPECT_EQ(kernel.Readable(),                                    \
              "tmp0 = ops.Load(\"data0:0\")\n"                      \
              "tmp1 = ops." #BACKENDOP                              \
              "(tmp0)\n"                                            \
              "tmp2 = ops.Store(\"" #OP "_0:0\", tmp1)\n");         \
  }

#define TEST_F_LOWER_INST2(BACKENDOP, OP)                           \
  TEST_F(LoopNodeLoweringUT, Lowering##OP) {                        \
    [this]() {                                                      \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);     \
      data0.SetSymbolShape({"s0", "s1", "s2"});                     \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);     \
      data1.SetSymbolShape({"s0", "s1", "s2"});                     \
      auto tmp = es::OP(data0, data1);                              \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                       \
      es_graph_->SetOutput(tmp, 0);                                 \
    }();                                                            \
                                                                    \
    auto graph = es_graph_->Build();                                \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                \
    auto tmp = cg->FindNode(#OP "_0");                              \
    ASSERT_NE(tmp, nullptr);                                        \
    for (auto &node : cg->GetAllNodes()) {                          \
      ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);    \
    }                                                               \
    auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0)); \
    ASSERT_TRUE(kernel.IsRealizedPersistent());                     \
    ASSERT_FALSE(kernel.IsExternKernel());                          \
    EXPECT_EQ(kernel.Readable(),                                    \
              "tmp0 = ops.Load(\"data0:0\")\n"                      \
              "tmp1 = ops.Load(\"data1:0\")\n"                      \
              "tmp2 = ops." #BACKENDOP                              \
              "(tmp0, tmp1)\n"                                      \
              "tmp3 = ops.Store(\"" #OP "_0:0\", tmp2)\n");         \
  }

#define TEST_F_LOWER_INST3(BACKENDOP, OP)                           \
  TEST_F(LoopNodeLoweringUT, Lowering##OP) {                        \
    [this]() {                                                      \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);     \
      data0.SetSymbolShape({"s0", "s1", "s2"});                     \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);     \
      data1.SetSymbolShape({"s0", "s1", "s2"});                     \
      auto data2 = es_graph_->CreateInput(2, "data2", nullptr);     \
      data2.SetSymbolShape({"s0", "s1", "s2"});                     \
      auto tmp = es::OP(data0, data1, data2);                       \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                       \
      es_graph_->SetOutput(tmp, 0);                                 \
    }();                                                            \
                                                                    \
    auto graph = es_graph_->Build();                                \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                \
    auto tmp = cg->FindNode(#OP "_0");                              \
    ASSERT_NE(tmp, nullptr);                                        \
    for (auto &node : cg->GetAllNodes()) {                          \
      ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);    \
    }                                                               \
    auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0)); \
    ASSERT_TRUE(kernel.IsRealizedPersistent());                     \
    ASSERT_FALSE(kernel.IsExternKernel());                          \
    EXPECT_EQ(kernel.Readable(),                                    \
              "tmp0 = ops.Load(\"data0:0\")\n"                      \
              "tmp1 = ops.Load(\"data1:0\")\n"                      \
              "tmp2 = ops.Load(\"data2:0\")\n"                      \
              "tmp3 = ops." #BACKENDOP                              \
              "(tmp0, tmp1, tmp2)\n"                                \
              "tmp4 = ops.Store(\"" #OP "_0:0\", tmp3)\n");         \
  }

#define TEST_F_LOWER_REDUCE(BACKENDOP, OP)                                                                           \
  TEST_F(LoopNodeLoweringUT, Lowering##OP) {                                                                         \
    [this]() {                                                                                                       \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                                      \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                                      \
      auto reduce = es::OP(data0, {1}, false);                                                                       \
      reduce.SetSymbolShape({"s0", "s2"});                                                                           \
      es_graph_->SetOutput(reduce, 0);                                                                               \
    }();                                                                                                             \
                                                                                                                     \
    auto graph = es_graph_->Build();                                                                                 \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                                 \
    auto reduce = cg->FindNode(#OP "_0");                                                                            \
    ASSERT_NE(reduce, nullptr);                                                                                      \
    for (auto &node : cg->GetAllNodes()) {                                                                           \
      ASSERT_EQ(LoweringManager::Lowering(node), GRAPH_SUCCESS);                                                     \
    }                                                                                                                \
    auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));                                               \
    ASSERT_FALSE(kernel.IsExternKernel());                                                                           \
    EXPECT_EQ(kernel.Readable(),                                                                                     \
              "tmp0 = ops.Load(\"data0:0\")\n"                                                                       \
              "tmp1 = ops.StoreReduction(\"" #OP "_0:0\", ops." #BACKENDOP "(tmp0, \"[d0, d1, d2]->[d0, d2]\"))\n"); \
  }

TEST_F_LOWER_INST1(Abs, Abs)
TEST_F_LOWER_INST1(Erf, Erf)
TEST_F_LOWER_INST1(Exp, Exp)
TEST_F_LOWER_INST1(IsFinite, IsFinite)
TEST_F_LOWER_INST1(IsNan, IsNan)
TEST_F_LOWER_INST1(LogicalNot, LogicalNot)
TEST_F_LOWER_INST1(Neg, Neg)
TEST_F_LOWER_INST1(Reciprocal, Reciprocal)
TEST_F_LOWER_INST1(Relu, Relu)
TEST_F_LOWER_INST1(Rsqrt, Rsqrt)
TEST_F_LOWER_INST1(Sigmoid, Sigmoid)
TEST_F_LOWER_INST1(Sign, Sign)
TEST_F_LOWER_INST1(Sqrt, Sqrt)
TEST_F_LOWER_INST1(Tanh, Tanh)
TEST_F_LOWER_INST1(Gelu, Gelu)

TEST_F_LOWER_INST2(Add, Add)
TEST_F_LOWER_INST2(Div, Div)
TEST_F_LOWER_INST2(Eq, Equal)
TEST_F_LOWER_INST2(Ge, GreaterEqual)
TEST_F_LOWER_INST2(Gt, Greater)
TEST_F_LOWER_INST2(Le, LessEqual)
TEST_F_LOWER_INST2(Lt, Less)
TEST_F_LOWER_INST2(LogicalAnd, LogicalAnd)
TEST_F_LOWER_INST2(LogicalOr, LogicalOr)
TEST_F_LOWER_INST2(Maximum, Maximum)
TEST_F_LOWER_INST2(Minimum, Minimum)
TEST_F_LOWER_INST2(Mul, Mul)
TEST_F_LOWER_INST2(Ne, NotEqual)
TEST_F_LOWER_INST2(Pow, Pow)
TEST_F_LOWER_INST2(Sub, Sub)
TEST_F_LOWER_INST2(TrueDiv, RealDiv)
TEST_F_LOWER_INST2(FloorDiv, FloorDiv)

TEST_F_LOWER_REDUCE(Sum, ReduceSumD)
TEST_F_LOWER_REDUCE(Max, ReduceMaxD)
TEST_F_LOWER_REDUCE(Mean, ReduceMeanD)
TEST_F_LOWER_REDUCE(Min, ReduceMinD)
TEST_F_LOWER_REDUCE(Prod, ReduceProdD)
}  // namespace ge
