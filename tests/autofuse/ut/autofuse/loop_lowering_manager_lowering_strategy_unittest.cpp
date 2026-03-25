
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

#include "graph/debug/ge_op_types.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"

#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_ir_lowerer.h"
#include "lowering/lowerings.h"
#include "lowering/liftings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"

#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"

using namespace std;
using namespace testing;
namespace ge {
using namespace autofuse;
const static bool _ = []() {
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_reduce = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_slice = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_gather = true;
  return true;
}();

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

namespace {
REGISTER_LOWERING(ZeroLikeStub) {
  auto src = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src);
  auto desc = src->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto dtype = desc->GetDataType();

  auto zero = loop::Scalar("0", dtype);
  (void)loop::Store(node->GetOutDataAnchor(0),
                    loop::Broadcast(zero, std::vector<loop::BroadcastOp::DimKind>(
                                              node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDimNum(),
                                              loop::BroadcastOp::DimKind::NORMAL)));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(DynamicQuantStub) {
  (void)loop::Store(node->GetOutDataAnchor(0), loop::Abs(loop::Load(node->GetInDataAnchor(0))));
  (void)loop::Store(node->GetOutDataAnchor(1), loop::Exp(loop::Load(node->GetInDataAnchor(0))));
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(AbsRepeat5) {
  auto v = loop::Load(node->GetInDataAnchor(0));
  for (size_t i = 0U; i < 5U; i++) {
    v = loop::Abs(v);
  }
  (void)loop::Store(node->GetOutDataAnchor(0), v);
  return GRAPH_SUCCESS;
}

std::string ReadableComputeGraph(const ComputeGraphPtr &graph) {
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
  for (const auto &node : graph->GetAllNodes()) {
    if (can_reached.find(node) == can_reached.end()) {
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
  return ss.str();
}

}  // namespace

class LoopGraphLoweringStrategyUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
    RegisterOpCreatorV2("ZeroLikeStub", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
    RegisterOpCreatorV2("AbsRepeat5", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  }
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopGraphLoweringStrategyUT, SimpleIn2Out1Lowering) {
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

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  EXPECT_TRUE(kernel.IsRealized());

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
}

TEST_F(LoopGraphLoweringStrategyUT, RealizeAsTooManyOps) {
  size_t num_ops = 50U;
  [this, num_ops]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<es::Tensor> ops{data0};
    for (size_t i = 0; i < num_ops; i++) {
      auto exp = es::Abs(ops.back());
      exp.SetSymbolShape({"s0", "s1", "s2"});
      ops.push_back(exp);
    }
    es_graph_->SetOutput(ops.back(), 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  std::vector<ge::NodePtr> exps;
  for (size_t i = 0; i < num_ops; i++) {
    auto exp = cg->FindNode("Abs_" + std::to_string(i));
    ASSERT_NE(exp, nullptr);
    exps.push_back(exp);
  }

  LoweringConfig config;
  config.max_loop_ops = 5U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  for (size_t i = 0; i < num_ops; i++) {
    if (i + 1 == num_ops) {
      EXPECT_TRUE(loop::GetKernelBox(exps[i]->GetOutDataAnchor(0)).IsRealized());
    } else {
      EXPECT_EQ(loop::GetKernelBox(exps[i]->GetOutDataAnchor(0)).IsRealized(), ((i + 1) % config.max_loop_ops) == 0);
    }
  }

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
}

TEST_F(LoopGraphLoweringStrategyUT, RealizeAsTooManyLoads) {
  size_t num_ops = 50U;
  [this, num_ops]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<es::Tensor> ops{data0};
    for (size_t i = 0; i < num_ops; i++) {
      auto data = es_graph_->CreateInput(i + 1, ("data" + std::to_string(i + 1)).c_str(), nullptr);
      data.SetSymbolShape({"s0", "s1", "s2"});
      auto add = es::Add(ops.back(), data);
      add.SetSymbolShape({"s0", "s1", "s2"});
      ops.push_back(add);
    }
    es_graph_->SetOutput(ops.back(), 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  std::vector<ge::NodePtr> adds;
  for (size_t i = 0; i < num_ops; i++) {
    auto add = cg->FindNode("Add_" + std::to_string(i));
    ASSERT_NE(add, nullptr);
    adds.push_back(add);
  }

  LoweringConfig config;
  config.max_loop_loads = 5U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  for (size_t i = 0; i < num_ops; i++) {
    if (i + 1 == num_ops) {
      EXPECT_TRUE(loop::GetKernelBox(adds[i]->GetOutDataAnchor(0)).IsRealized());
    } else {
      EXPECT_EQ(loop::GetKernelBox(adds[i]->GetOutDataAnchor(0)).IsRealized(), (i % 4) == 3);
    }
  }

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
}

TEST_F(LoopGraphLoweringStrategyUT, RealizeAsReduction) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    auto abs = es::Abs(reduce);
    abs.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_TRUE(kernel.IsRealized());
}

TEST_F(LoopGraphLoweringStrategyUT, RealizeAsReductionFailure) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = es_graph_->CreateInput(1, "axes", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    auto abs = es::Abs(reduce);
    abs.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto reduce = cg->FindNode("ReduceSum_0");
  ASSERT_NE(reduce, nullptr);
  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, SkipReductionAsDisabled) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(data0, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});
    auto abs = es::Abs(reduce);
    abs.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto reduce = cg->FindNode("ReduceSum_1");
  ASSERT_NE(reduce, nullptr);

  auto origin = AutoFuseConfig::LoweringConfig().experimental_lowering_reduce;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().experimental_lowering_reduce = origin; });
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_reduce = false;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(reduce->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, SkipSliceOutputNoSymbolic) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 20};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, SkipConcatAsDisabled) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"2", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"3", "s1", "s2"});
    auto concat = es::ConcatD({data0, data1}, 0);
    concat.SetSymbolShape({"5", "s1", "s2"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("ConcatD_0");
  ASSERT_NE(concat, nullptr);

  auto origin = AutoFuseConfig::LoweringConfig().experimental_lowering_concat;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = origin; });
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = false;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(concat->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, SkipSliceAsDisabled) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 20};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"30", "60", "19"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);

  auto origin = AutoFuseConfig::LoweringConfig().experimental_lowering_slice;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().experimental_lowering_slice = origin; });
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_slice = false;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, SkipGatherAsDisabled) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(gather, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto gather = cg->FindNode("GatherV2_1");
  ASSERT_NE(gather, nullptr);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  auto origin = AutoFuseConfig::LoweringConfig().experimental_lowering_gather;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().experimental_lowering_gather = origin; });
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_gather = false;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(gather->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, SkippedFuseToAscBackendAsTooFewAscIrNodes) {
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

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 2U;  // fused kernel box has only 1
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  auto asc_node_name = loop::BufferName(add->GetOutDataAnchor(0)) + "_asc";
  auto asc_node = cg->FindNode(asc_node_name);
  ASSERT_EQ(asc_node, nullptr);
}

TEST_F(LoopGraphLoweringStrategyUT, FuseToAscBackendAsConfigured) {
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

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  ASSERT_EQ(asc_node->GetInDataAnchor(0)->GetPeerOutAnchor(), cg->FindNode("data0")->GetOutDataAnchor(0));
  ASSERT_EQ(asc_node->GetInDataAnchor(1)->GetPeerOutAnchor(), cg->FindNode("data1")->GetOutDataAnchor(0));
}

TEST_F(LoopGraphLoweringStrategyUT, OriginComputeGraphIsCorrect) {
  size_t num_ops = 2U;
  [this, num_ops]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<es::Tensor> ops{data0};
    for (size_t i = 0; i < num_ops; i++) {
      auto exp = es::Abs(ops.back());
      exp.SetSymbolShape({"s0", "s1", "s2"});
      ops.push_back(exp);
    }
    es_graph_->SetOutput(ops.back(), 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  std::vector<ge::NodePtr> exps;
  for (size_t i = 0; i < num_ops; i++) {
    auto exp = cg->FindNode("Abs_" + std::to_string(i));
    ASSERT_NE(exp, nullptr);
    exps.push_back(exp);
  }

  LoweringConfig config;
  config.max_loop_ops = 1U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  AscBackendFuseConfig asc_config;
  asc_config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, asc_config), GRAPH_SUCCESS);

  auto exp1_asc = cg->FindNode("autofuse_pointwise_0_Abs");
  ASSERT_NE(exp1_asc, nullptr);
  auto exp2_asc = cg->FindNode("autofuse_pointwise_1_Abs");
  ASSERT_NE(exp2_asc, nullptr);

  auto fuse1_attrs = exp1_asc->GetOpDesc()->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fuse1_attrs, nullptr);
  ASSERT_EQ(LoweringManager::GetFusedOriginComputeGraph(*fuse1_attrs, exp1_asc), GRAPH_SUCCESS);


  auto fuse2_attrs = exp2_asc->GetOpDesc()->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fuse2_attrs, nullptr);
  ASSERT_EQ(LoweringManager::GetFusedOriginComputeGraph(*fuse2_attrs, exp2_asc), GRAPH_SUCCESS);

}


TEST_F(LoopGraphLoweringStrategyUT, DifferentScope) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"s0", "s1"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1"});
    auto cast = es::Cast(abs1, DT_UINT8);
    cast.SetSymbolShape({"s0", "s1"});
    auto logicalnot = es::LogicalNot(cast);
    logicalnot.SetSymbolShape({"s0", "s1"});
    auto logicalnot1 = es::LogicalNot(logicalnot);
    logicalnot1.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(logicalnot1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto cast = cg->FindNode("Abs_0");
  ASSERT_NE(cast, nullptr);
  (void)ge::AttrUtils::SetStr(cast->GetOpDesc(), "_op_vectorcore_num", "20");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
}

TEST_F(LoopGraphLoweringStrategyUT, SkipFuseAsDifferentStreamLabel) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Abs(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    auto exp2 = es::Abs(exp1);
    exp2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp1 = cg->FindNode("Abs_0");
  auto exp2 = cg->FindNode("Abs_1");
  ASSERT_NE(exp1, nullptr);
  ASSERT_NE(exp2, nullptr);

  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_label", "stream1");
  ge::AttrUtils::SetStr(exp2->GetOpDesc(), "_user_stream_label", "stream2");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  ASSERT_NE(cg->FindNode("autofuse_pointwise_0_Abs"), nullptr);
  ASSERT_NE(cg->FindNode("autofuse_pointwise_1_Abs"), nullptr);
}

TEST_F(LoopGraphLoweringStrategyUT, FuseAsSameStreamLabel) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Abs(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    auto exp2 = es::Abs(exp1);
    exp2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp1 = cg->FindNode("Abs_0");
  auto exp2 = cg->FindNode("Abs_1");
  ASSERT_NE(exp1, nullptr);
  ASSERT_NE(exp2, nullptr);

  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_label", "stream1");
  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_priority", "2");
  ge::AttrUtils::SetStr(exp2->GetOpDesc(), "_user_stream_label", "stream1");
  ge::AttrUtils::SetStr(exp2->GetOpDesc(), "_user_stream_priority", "2");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  ASSERT_EQ(cg->FindNode("autofuse_pointwise_0_Abs"), nullptr);
  auto exp2_asc = cg->FindNode("autofuse_pointwise_0_Abs_Abs");
  ASSERT_NE(exp2_asc, nullptr);
  std::string stream_label;
  std::string stream_priority;
  ASSERT_TRUE(AttrUtils::GetStr(exp2_asc->GetOpDesc(), "_user_stream_label", stream_label));
  ASSERT_TRUE(AttrUtils::GetStr(exp2_asc->GetOpDesc(), "_user_stream_priority", stream_priority));
  EXPECT_EQ(stream_label, "stream1");
  EXPECT_EQ(stream_priority, "2");
}

TEST_F(LoopGraphLoweringStrategyUT, FuseAsOnesideStreamLabelCase1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Abs(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    auto exp2 = es::Abs(exp1);
    exp2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp1 = cg->FindNode("Abs_0");
  auto exp2 = cg->FindNode("Abs_1");
  ASSERT_NE(exp1, nullptr);
  ASSERT_NE(exp2, nullptr);

  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_label", "stream1");
  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_priority", "2");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  ASSERT_EQ(cg->FindNode("autofuse_pointwise_0_Abs"), nullptr);
  auto exp2_asc = cg->FindNode("autofuse_pointwise_0_Abs_Abs");
  ASSERT_NE(exp2_asc, nullptr);
  std::string stream_label;
  std::string stream_priority;
  ASSERT_TRUE(AttrUtils::GetStr(exp2_asc->GetOpDesc(), "_user_stream_label", stream_label));
  ASSERT_TRUE(AttrUtils::GetStr(exp2_asc->GetOpDesc(), "_user_stream_priority", stream_priority));
  EXPECT_EQ(stream_label, "stream1");
  EXPECT_EQ(stream_priority, "2");
}

TEST_F(LoopGraphLoweringStrategyUT, FuseAsOnesideStreamLabelCase2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Abs(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    auto exp2 = es::Abs(exp1);
    exp2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp1 = cg->FindNode("Abs_0");
  auto exp2 = cg->FindNode("Abs_1");
  ASSERT_NE(exp1, nullptr);
  ASSERT_NE(exp2, nullptr);

  ge::AttrUtils::SetStr(exp2->GetOpDesc(), "_user_stream_label", "stream1");
  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_priority", "2");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  ASSERT_EQ(cg->FindNode("autofuse_pointwise_0_Abs"), nullptr);
  auto exp2_asc = cg->FindNode("autofuse_pointwise_0_Abs_Abs");
  ASSERT_NE(exp2_asc, nullptr);
  std::string stream_label;
  std::string stream_priority;
  ASSERT_TRUE(AttrUtils::GetStr(exp2_asc->GetOpDesc(), "_user_stream_label", stream_label));
  ASSERT_TRUE(AttrUtils::GetStr(exp2_asc->GetOpDesc(), "_user_stream_priority", stream_priority));
  EXPECT_EQ(stream_label, "stream1");
  EXPECT_EQ(stream_priority, "2");
}

TEST_F(LoopGraphLoweringStrategyUT, FuseCaseStreamLabelLabel1ThenNULLThenLabel2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Abs(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    auto exp2 = es::Abs(exp1);
    exp2.SetSymbolShape({"s0", "s1", "s2"});
    auto exp3 = es::Abs(exp2);
    exp3.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp3, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto exp1 = cg->FindNode("Abs_0");
  auto exp2 = cg->FindNode("Abs_1");
  auto exp3 = cg->FindNode("Abs_2");
  ASSERT_NE(exp1, nullptr);
  ASSERT_NE(exp2, nullptr);
  ASSERT_NE(exp3, nullptr);

  ge::AttrUtils::SetStr(exp1->GetOpDesc(), "_user_stream_label", "stream1");
  ge::AttrUtils::SetStr(exp3->GetOpDesc(), "_user_stream_label", "stream2");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  ASSERT_NE(cg->FindNode("autofuse_pointwise_0_Abs_Abs"), nullptr);
  ASSERT_NE(cg->FindNode("autofuse_pointwise_1_Abs"), nullptr);
}

TEST_F(LoopGraphLoweringStrategyUT, NodeWithControlIfLoweringNotUseInputData) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto zero_like = es::ZerosLike(data0);
    zero_like.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(zero_like, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("ZerosLike")->GetOpDesc()->SetType("ZeroLikeStub");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_ZeroLikeStub, [], [data0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, NodeInputCanRealizeEvenLoweringNotUseInputData) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto zero_like = es::ZerosLike(abs);
    zero_like.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(zero_like, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("ZerosLike")->GetOpDesc()->SetType("ZeroLikeStub");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub, [], [autofuse_pointwise_0_Abs])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LiftingOneNodeAsOnlyOneCanfusePointwise) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, NodeLiftSucceedIfLoweringNotUseInputData) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(data0);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    auto zero_like = es::ZerosLike(exp);
    zero_like.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(zero_like, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("ZerosLike")->GetOpDesc()->SetType("ZeroLikeStub");
  AscIrLowerer asc_ir_lower;
  ASSERT_EQ(asc_ir_lower.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Exp, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub, [], [autofuse_pointwise_0_Exp])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
  ASSERT_EQ(asc_ir_lower.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Exp(Exp_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub, [], [Exp_0])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LiftingOneNodeAsOnlyOneCanfuseReduction) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(data0, {0});
    reduce.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(reduce, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_reduce_0_ReduceSumD, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.ReduceSumD(ReduceSumD_0, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LiftingOneNodeAsOutputDontUseInputData) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto zero_like = es::ZerosLike(abs);
    zero_like.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(zero_like, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("ZerosLike")->GetOpDesc()->SetType("ZeroLikeStub");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub, [], [autofuse_pointwise_0_Abs])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  cg->FindFirstNodeMatchType("ZeroLikeStub")->GetOpDesc()->SetType("ZerosLike");
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub, [], [Abs_0])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LiftingOneNodeAsOutputDontUseInputDataButFused) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto zero_like = es::ZerosLike(abs);
    zero_like.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Abs(zero_like);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("ZerosLike")->GetOpDesc()->SetType("ZeroLikeStub");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub_Abs, [], [autofuse_pointwise_0_Abs])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_ZeroLikeStub_Abs, [], [Abs_0])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LiftingCombinedOneNodeAscBackendOp) {
  size_t num_ops = 3U;
  [this, num_ops]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<es::Tensor> ops{data0};
    for (size_t i = 0; i < num_ops; i++) {
      auto abs = es::Abs(ops.back());
      abs.SetSymbolShape({"s0", "s1", "s2"});
      ops.push_back(abs);
    }
    es_graph_->SetOutput(ops.back(), 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  LoweringConfig config;
  config.max_loop_ops = 1U;  // trigger one node asc backend op
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp1])
tmp3 = ge.AscBackend(autofuse_pointwise_2_Abs, [tmp2])
tmp4 = ge.NetOutput(NetOutput, [tmp3])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.Abs(Abs_1, [tmp1])
tmp3 = ge.Abs(Abs_2, [tmp2])
tmp4 = ge.NetOutput(NetOutput, [tmp3])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, DynamicOutputLiftingAsAllNodesNeedLifting) {
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
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_DynamicQuantStub, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_DynamicQuantStub, [tmp0])
[tmp3, tmp4] = ge.NetOutput(NetOutput, [tmp1, tmp2])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  cg->FindFirstNodeMatchType("DynamicQuantStub")->GetOpDesc()->SetType("DynamicQuant");
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
[tmp1, tmp2] = ge.DynamicQuant(DynamicQuant_0, [tmp0])
[tmp3, tmp4] = ge.NetOutput(NetOutput, [tmp1, tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, DynamicOutputSkipLiftingAsOneNodesFused) {
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
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_DynamicQuantStub, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_DynamicQuantStub_Abs, [tmp0])
[tmp3, tmp4] = ge.NetOutput(NetOutput, [tmp1, tmp2])
)");
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
  cg->FindFirstNodeMatchType("DynamicQuantStub")->GetOpDesc()->SetType("DynamicQuant");
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_DynamicQuantStub, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_DynamicQuantStub_Abs, [tmp0])
[tmp3, tmp4] = ge.NetOutput(NetOutput, [tmp1, tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, TriggerRealizeAsControlEdges) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  GraphUtils::AddEdge(cg->FindNode("Abs_0")->GetOutControlAnchor(), cg->FindNode("Abs_1")->GetInControlAnchor());
  LoweringConfig config;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp1], [autofuse_pointwise_0_Abs])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, SkipRealizeAsControlEdgesFromCFCase1) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});

    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(abs1, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});

    auto abs2 = es::Abs(reduce);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cons1 = cg->FindNode("FileConstant_1");
  ASSERT_NE(cons1, nullptr);
  (void) AttrUtils::SetBool(cons1->GetOpDesc(), "_is_from_constant_folding", true);

  GraphUtils::AddEdge(cg->FindNode("Abs_0")->GetOutControlAnchor(), cg->FindNode("FileConstant_1")->GetInControlAnchor());
  GraphUtils::AddEdge(cg->FindNode("FileConstant_1")->GetOutControlAnchor(), cg->FindNode("Abs_3")->GetInControlAnchor());
  LoweringConfig config;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
}

TEST_F(LoopGraphLoweringStrategyUT, SkipRealizeAsControlEdgesFromCFCase2) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});

    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(abs1, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});

    auto axis1 = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce1 = es::ReduceSum(data, axis1, true);
    reduce1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(reduce, 0);
    es_graph_->SetOutput(reduce1, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cons1 = cg->FindNode("FileConstant_1");
  ASSERT_NE(cons1, nullptr);
  auto reduce2 = cg->FindNode("ReduceSum_4");
  ASSERT_NE(reduce2, nullptr);
  (void) AttrUtils::SetBool(cons1->GetOpDesc(), "_is_from_constant_folding", true);

  GraphUtils::AddEdge(reduce2->GetOutControlAnchor(), cg->FindNode("FileConstant_1")->GetInControlAnchor());
  LoweringConfig config;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LiftingManager::LiftingGraph(cg), GRAPH_SUCCESS);
}

TEST_F(LoopGraphLoweringStrategyUT, TriggerRealizeAsHeavyOps) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    abs = es::Abs(exp);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  LoweringConfig config;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs_Exp, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, TriggerRealizeAsTooManyReaders) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(abs2, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  LoweringConfig config;
  config.max_buffer_readers = 1U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp1])
tmp3 = ge.AscBackend(autofuse_pointwise_2_Abs, [tmp1])
[tmp4, tmp5] = ge.NetOutput(NetOutput, [tmp2, tmp3])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, NotTriggerRealizeAsUnreachNumReaders) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(abs2, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  LoweringConfig config;
  config.max_buffer_readers = 3U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_Abs, [tmp1])
tmp3 = ge.AscBackend(autofuse_pointwise_2_Abs, [tmp1])
[tmp4, tmp5] = ge.NetOutput(NetOutput, [tmp2, tmp3])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, TriggerRealizeInputsAndLoweringAndSucceed) {
  [this]() {
    std::vector<es::Tensor> datas;
    constexpr int32_t num_loads = 5;
    for (int i = 0; i < num_loads; i++) {
      datas.push_back(es_graph_->CreateInput(i, ("data" + std::to_string(i)).c_str(), nullptr));
      datas.back().SetSymbolShape({"s0", "s1", "s2"});
    }
    auto addn1 = es::AddN({datas[0], datas[1], datas[2]}, 3);  // 3 inputs
    addn1.SetSymbolShape({"s0", "s1", "s2"});
    auto addn2 = es::AddN({addn1, datas[3], datas[4]}, 3);  // 5 inputs after lowering fuse
    addn2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(addn2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  LoweringConfig config;
  config.max_loop_loads = 4U;  // trigger realize inputs after lowering addn2
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(data2, [])
tmp3 = ge.Data(data3, [])
tmp4 = ge.Data(data4, [])
tmp5 = ge.AscBackend(autofuse_pointwise_0_AddN, [tmp0, tmp1, tmp2])
tmp6 = ge.AscBackend(autofuse_pointwise_1_AddN, [tmp5, tmp3, tmp4])
tmp7 = ge.NetOutput(NetOutput, [tmp6])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, TriggerRealizeInputsAsTooManyOpsAndLoweringAndSucceed) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindNode("Abs_0")->GetOpDesc()->SetType("AbsRepeat5");
  cg->FindNode("Abs_1")->GetOpDesc()->SetType("AbsRepeat5");
  LoweringConfig config;
  config.max_loop_ops = 6U;  // trigger realize inputs after lowering abs1
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_AbsRepeat5, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_1_AbsRepeat5, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, TriggerRealizeInputsAsTooManyOpsAndLoweringAndFallback) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindNode("Abs_1")->GetOpDesc()->SetType("AbsRepeat5");
  LoweringConfig config;
  config.max_loop_ops = 4U;  // trigger realize inputs after lowering abs1
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  cg->FindNode("Abs_1")->GetOpDesc()->SetType("Abs");
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.Abs(Abs_1, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, FallbackLoweringAsInSuperKernelScope1) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AttrUtils::SetStr(cg->FindNode("Abs_1")->GetOpDesc(), "_super_kernel_scope", "scope");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.Abs(Abs_1, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, FallbackLoweringAsInSuperKernelScope2) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AttrUtils::SetStr(cg->FindNode("Abs_0")->GetOpDesc(), "_super_kernel_scope", "scope");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, FallbackLoweringAsInDiableAutofuseScope) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  AttrUtils::SetBool(cg->FindNode("Abs_1")->GetOpDesc(), "_disable_autofuse_scope", true);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp2 = ge.Abs(Abs_1, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, DropLoweringResultAsUnsupportedDtype) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Cast(data, DT_UNDEFINED);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Cast(abs1, DT_FLOAT);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Cast(Cast_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_0_Cast, [tmp1])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, KeepLoweringResultAsSupportedDtype) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(abs1);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs_Abs, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, DropLowerResultForScalarGraph) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, DropLowerResultForScalarGraphShape1) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"1"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"1"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, InputOutputDescCopiedFromOriginBuffer) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto abs = cg->FindNode("Abs_0");
  ASSERT_NE(abs, nullptr);
  auto buffer_desc = abs->GetOpDesc()->MutableOutputDesc(0);
  buffer_desc->SetDataType(DT_FLOAT16);
  buffer_desc->SetOriginDataType(DT_FLOAT);
  auto data = cg->FindNode("data0");
  ASSERT_NE(data, nullptr);
  auto input0_desc = data->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_FLOAT16);
  input0_desc->SetOriginDataType(DT_FLOAT);
  LoweringConfig config;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  auto asc = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc, nullptr);
  auto asc_input0_desc = asc->GetOpDesc()->MutableInputDesc(0);
  ASSERT_EQ(asc_input0_desc->GetDataType(), input0_desc->GetDataType());
  ASSERT_EQ(asc_input0_desc->GetOriginDataType(), input0_desc->GetOriginDataType());
  auto asc_desc = asc->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_EQ(buffer_desc->GetDataType(), asc_desc->GetDataType());
  ASSERT_EQ(buffer_desc->GetOriginDataType(), asc_desc->GetOriginDataType());
  std::vector<Expression> buffer_shape;
  std::vector<Expression> asc_shape;
  loop::GetBufferShape(abs->GetOutDataAnchor(0), buffer_shape);
  loop::GetBufferShape(asc->GetOutDataAnchor(0), asc_shape);
  ASSERT_EQ(buffer_shape.size(), asc_shape.size());
  for (size_t i = 0; i < buffer_shape.size(); i++) {
    ASSERT_EQ(buffer_shape[i], asc_shape[i]);
  }
}

TEST_F(LoopGraphLoweringStrategyUT, LoweringEmptyTensor1) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"0", "s0", "s1"});
    auto sum = es::ReduceSumD(data, {0}, false);
    sum.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(sum, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  auto sum = cg->FindNode("ReduceSumD_0");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(sum->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsExternKernel());
}

TEST_F(LoopGraphLoweringStrategyUT, LoweringEmptyTensor2) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"0", "s0", "s1"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"0", "s0", "s1"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"0", "s0", "s1"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  GE_DUMP(cg, "lowering1");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  GE_DUMP(cg, "lowering2");
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackendNoKernelOp(autofuse_pointwise_0_Abs_Abs, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LoweringCheckDType) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1"});
    auto abs = es::Abs(data);
    abs.SetSymbolShape({"s0", "s1"});
    auto abs1 = es::Abs(abs);
    abs1.SetSymbolShape({"s0", "s1"});
    auto cast = es::Cast(abs1, DT_UINT8);
    cast.SetSymbolShape({"s0", "s1"});
    auto logicalnot = es::LogicalNot(cast);
    logicalnot.SetSymbolShape({"s0", "s1"});
    auto logicalnot1 = es::LogicalNot(logicalnot);
    logicalnot1.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(logicalnot1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto cast = cg->FindNode("Cast_2");
  ASSERT_NE(cast, nullptr);
  auto cast_desc = cast->GetOpDesc()->MutableOutputDesc(0);
  cast_desc->SetDataType(DT_UINT8);
  cast_desc->SetOriginDataType(DT_UINT8);
  auto logicalnot = cg->FindNode("LogicalNot_3");
  ASSERT_NE(logicalnot, nullptr);
  auto logicalnot_desc = logicalnot->GetOpDesc()->MutableOutputDesc(0);
  logicalnot_desc->SetDataType(DT_UINT8);
  logicalnot_desc->SetOriginDataType(DT_UINT8);
  auto logicalnot_desc1 = logicalnot->GetOpDesc()->MutableInputDesc(0);
  logicalnot_desc1->SetDataType(DT_UINT8);
  logicalnot_desc1->SetOriginDataType(DT_UINT8);
  auto logicalnot1 = cg->FindNode("LogicalNot_4");
  ASSERT_NE(logicalnot1, nullptr);
  auto logicalnot1_desc = logicalnot1->GetOpDesc()->MutableOutputDesc(0);
  logicalnot1_desc->SetDataType(DT_UINT8);
  logicalnot1_desc->SetOriginDataType(DT_UINT8);
  auto logicalnot1_desc1 = logicalnot1->GetOpDesc()->MutableInputDesc(0);
  logicalnot1_desc1->SetDataType(DT_UINT8);
  logicalnot1_desc1->SetOriginDataType(DT_UINT8);
  GE_DUMP(cg, "lowering1");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  GE_DUMP(cg, "lowering2");
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_pointwise_0_Abs_Abs, [tmp0])
tmp2 = ge.Cast(Cast_2, [tmp1])
tmp3 = ge.AscBackend(autofuse_pointwise_1_LogicalNot_LogicalNot, [tmp2])
tmp4 = ge.NetOutput(NetOutput, [tmp3])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, LoweringReduceSumFP16) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(data0);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    auto n = es::ReduceSumD(abs2, {2}, true);
    n.SetSymbolShape({"s0", "s1", "1"});

    es_graph_->SetOutput(n, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto data = cg->FindNode("data0");
  auto data_desc = data->GetOpDesc()->MutableOutputDesc(0);
  data_desc->SetDataType(DT_FLOAT16);
  data_desc->SetOriginDataType(DT_FLOAT16);
  auto abs = cg->FindNode("Abs_0");
  auto abs1_desc = abs->GetOpDesc()->MutableInputDesc(0);
  abs1_desc->SetDataType(DT_FLOAT16);
  abs1_desc->SetOriginDataType(DT_FLOAT16);
  auto abs_desc = abs->GetOpDesc()->MutableOutputDesc(0);
  abs_desc->SetDataType(DT_FLOAT16);
  abs_desc->SetOriginDataType(DT_FLOAT16);
  auto reduce = cg->FindNode("ReduceSumD_1");
  auto reduce_desc = reduce->GetOpDesc()->MutableInputDesc(0);
  reduce_desc->SetDataType(DT_FLOAT16);
  reduce_desc->SetOriginDataType(DT_FLOAT16);
  auto reduce1_desc = reduce->GetOpDesc()->MutableOutputDesc(0);
  reduce1_desc->SetDataType(DT_FLOAT16);
  reduce1_desc->SetOriginDataType(DT_FLOAT16);
  GE_DUMP(cg, "lowering");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.AscBackend(autofuse_reduce_0_Abs_ReduceSumD, [tmp0])
tmp2 = ge.NetOutput(NetOutput, [tmp1])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, MatMul) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto offset_w = es_graph_->CreateInput(2, "offset_w", nullptr);
    data0.SetSymbolShape({"4", "4"});
    data1.SetSymbolShape({"4", "4"});
    offset_w.SetSymbolShape({"4", "4"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"4", "4"});
    auto mm = es::MatMulV2(abs, data1, nullptr, offset_w);
    mm.SetSymbolShape({"4", "4"});
    auto abs1 = es::Abs(mm);
    abs1.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mm = cg->FindNode("MatMulV2_1");
  ASSERT_NE(mm, nullptr);
  auto tmp_desc = mm->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0 = cg->FindNode("data0");
  auto tmp_desc0 = data0->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1 = cg->FindNode("data1");
  auto tmp_desc1 = data1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3 = cg->FindNode("offset_w");
  auto tmp_desc3 = data3->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);
  auto abs = cg->FindNode("Abs_0");
  auto tmp_desc4 = abs->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_FLOAT16);
  tmp_desc4->SetOriginDataType(DT_FLOAT16);

  LoweringConfig config;
  config.max_buffer_readers = 3U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(offset_w, [])
tmp3 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp4 = ge.AscBackend(autofuse_cube_1_MatMulV2, [tmp3, tmp1, tmp2])
tmp5 = ge.AscBackend(autofuse_pointwise_2_Abs, [tmp4])
tmp6 = ge.NetOutput(NetOutput, [tmp5])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, BatchMatMul) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto offset_w = es_graph_->CreateInput(2, "offset_w", nullptr);
    data0.SetSymbolShape({"4", "4"});
    data1.SetSymbolShape({"4", "4"});
    offset_w.SetSymbolShape({"4", "4"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"4", "4"});
    auto mm = es::BatchMatMulV2(abs, data1, nullptr, offset_w);
    mm.SetSymbolShape({"4", "4"});
    auto abs1 = es::Abs(mm);
    abs1.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mm = cg->FindNode("BatchMatMulV2_1");
  ASSERT_NE(mm, nullptr);
  auto tmp_desc = mm->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0 = cg->FindNode("data0");
  auto tmp_desc0 = data0->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1 = cg->FindNode("data1");
  auto tmp_desc1 = data1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3 = cg->FindNode("offset_w");
  auto tmp_desc3 = data3->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);
  auto abs = cg->FindNode("Abs_0");
  auto tmp_desc4 = abs->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_FLOAT16);
  tmp_desc4->SetOriginDataType(DT_FLOAT16);

  LoweringConfig config;
  config.max_buffer_readers = 3U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(offset_w, [])
tmp3 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp4 = ge.AscBackend(autofuse_cube_1_BatchMatMulV2, [tmp3, tmp1, tmp2])
tmp5 = ge.AscBackend(autofuse_pointwise_2_Abs, [tmp4])
tmp6 = ge.NetOutput(NetOutput, [tmp5])
)");
}

TEST_F(LoopGraphLoweringStrategyUT, SkipFuseAsInSkipCfg) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto relu2 = es::Relu(abs1);
    relu2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(relu2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto abs1 = cg->FindNode("Abs_0");
  auto relu2 = cg->FindNode("Relu_1");
  ASSERT_NE(abs1, nullptr);
  ASSERT_NE(relu2, nullptr);

  AutoFuseConfig::MutableLoweringConfig().skip_node_types = {"Abs"};
  AutoFuseConfig::MutableLoweringConfig().skip_node_names = {"Relu_1"};
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  AscBackendFuseConfig config;
  config.min_ascend_ir_nodes = 1U;
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg, config), GRAPH_SUCCESS);

  // 融合后的计算图应该没有AscBackend节点
  size_t asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      asc_backend_node_count++;
    }
  }
  EXPECT_EQ(asc_backend_node_count, 0);
  AutoFuseConfig::MutableLoweringConfig().skip_node_types = {};
  AutoFuseConfig::MutableLoweringConfig().skip_node_names = {};
}
}  // namespace ge
