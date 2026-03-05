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
#include "pattern_fusion/pad_slice_optimize_pass.h"
#include "operator_factory.h"
#include "utils/autofuse_utils.h"
#include "graph_utils.h"
#include "op_creator_register.h"

namespace ge {
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

class PadSliceOptimizePassTest : public testing::Test {
 public:
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
  }
  std::unique_ptr<es::Graph> es_graph_;

  static ComputeGraphPtr CreateGraphPadSlice() {
    auto es_graph_ = std::make_unique<es::Graph>("graph");
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr); 
    data0.SetSymbolShape({"s0", "50", "s2"}); 
    data0.SetShape({100, 50, 20}); 
    auto paddings = CreateConst(*es_graph_, ge::DT_INT64, {3, 2}, std::vector<std::vector<int64_t>>{{0, 0}, {0, 50}, {0, 0}}); 
    paddings.SetShape({3, 2}); 
    auto pad = es::Pad(data0, paddings); 
    pad.SetShape({100, 100, 20}); 
    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0}); 
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{100, 50, 20}); 
    auto slice = es::Slice(pad, offset, size); 
    slice.SetSymbolShape({"s0", "100", "s2"}); 
    slice.SetShape({100, 100, 20}); 
    es_graph_->SetOutput(slice, 0);
    auto graph = es_graph_->Build();
    return GraphUtilsEx::GetComputeGraph(*graph);
  }
};

void SetShapeAndType(ComputeGraphPtr &cg) {
  auto pad1 = cg->FindNode("Pad_1");
  auto pad_desc1 = pad1->GetOpDesc()->MutableOutputDesc(0);
  pad_desc1->SetDataType(DT_INT64);
  pad_desc1->SetOriginDataType(DT_INT64);
  auto pad_input_desc = pad1->GetOpDesc()->MutableInputDesc(0);
  pad_input_desc->SetShape(GeShape(std::vector<int64_t>{100, 50, 20}));
  pad_input_desc->SetDataType(DT_INT64);
  pad_input_desc->SetOriginDataType(DT_INT64);

  auto pad_input_desc1 = pad1->GetOpDesc()->MutableInputDesc(1);
  pad_input_desc1->SetShape(GeShape(std::vector<int64_t>{3, 2}));
  pad_input_desc1->SetDataType(DT_INT64);
  pad_input_desc1->SetOriginDataType(DT_INT64);

  auto slice1 = cg->FindNode("Slice_4");
  auto slice_desc1 = slice1->GetOpDesc()->MutableOutputDesc(0);
  slice_desc1->SetDataType(DT_INT64);
  slice_desc1->SetOriginDataType(DT_INT64);
  return;
}

TEST_F(PadSliceOptimizePassTest, PadSlicePattern1) {
  auto cg = CreateGraphPadSlice();
  (void)SetShapeAndType(cg);
  PadSliceOptimizePass padSliceOptimizePass;
  bool changed = false;
  EXPECT_EQ(padSliceOptimizePass.Run(cg, changed), ge::GRAPH_SUCCESS);
}

TEST_F(PadSliceOptimizePassTest, PadSliceNotPattern) { 
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "50", "s2"});
    data0.SetShape({100, 50, 20});
    auto paddings = CreateConst(*es_graph_, ge::DT_INT32, {3, 2}, std::vector<std::vector<int64_t>>{{0, 0}, {0, 50}, {0, 0}});
    paddings.SetShape({3, 2});
    auto pad = es::Pad(data0, paddings);
    pad.SetShape({100, 100, 20});
    auto abs = es::Abs(pad);
    abs.SetSymbolShape({"s0", "100", "s2"});
    abs.SetShape({100, 100, 20});
    es_graph_->SetOutput(abs, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto pad1 = cg->FindNode("Pad_1");
  ASSERT_NE(pad1, nullptr);
  auto pad_desc1 = pad1->GetOpDesc()->MutableOutputDesc(0);
  pad_desc1->SetDataType(DT_FLOAT);
  pad_desc1->SetOriginDataType(DT_FLOAT);
  auto pad_input_desc = pad1->GetOpDesc()->MutableInputDesc(0);
  pad_input_desc->SetShape(GeShape(std::vector<int64_t>{100, 50, 20}));

  PadSliceOptimizePass padSliceOptimizePass;
  bool changed = false;
  EXPECT_EQ(padSliceOptimizePass.Run(cg, changed), ge::GRAPH_SUCCESS);
}
 
TEST_F(PadSliceOptimizePassTest, PadSlicePatternPostProcess) { 
  auto cg = CreateGraphPadSlice();
  (void)SetShapeAndType(cg);
  auto pad1 = cg->FindNode("Pad_1");
  ASSERT_NE(pad1, nullptr);

  PadSliceOptimizePass padSliceOptimizePass;
  EXPECT_EQ(padSliceOptimizePass.PostProcess(cg, pad1), ge::GRAPH_SUCCESS);
}
}  // namespace ge