
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
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"

#include "pattern_fusion/pattern_fusion.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/asc_backend_post_processor.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "ascgen_log.h"

#include "common/util/mem_utils.h"
#include "expression/testcase/source_stub.h"
#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "base/att_const_values.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
namespace {
struct ScopedEnv {
  explicit ScopedEnv(const char* k, const char* v) : key_(k) {
    old_ = std::getenv(k);
    setenv(k, v, 1);
  }
  ~ScopedEnv() {
    if (old_) setenv(key_, old_, 1);
    else      unsetenv(key_);
  }
private:
  const char* key_;
  const char* old_;
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

uint8_t AscSubgraphNodeCount(const NodePtr & AscNode , const string &node_type) {
  const auto attr = AscNode->GetOpDesc()->GetAttrsGroup<ge::AutoFuseAttrs>();
  uint8_t count = 0;
  for (const auto &node : attr->GetAscGraph()->GetAllNodes()) {
    if (node->GetType() == node_type) {
      count++;
    }
  }
  return count;
}
}  // namespace

class LoweringAndCanfuseUT : public testing::Test {
  public:
  protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_split = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_slice = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_reduce = true;
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;

  void BuildReluCastReshapeMultiRefConcatGraph() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"512", "32"});

    auto data4 = es_graph_->CreateInput(1, "data4", nullptr);
    data4.SetSymbolShape({"512", "1", "32"});
    auto data5 = es_graph_->CreateInput(2, "data5", nullptr);
    data5.SetSymbolShape({"512", "1", "32"});
    auto data6 = es_graph_->CreateInput(3, "data6", nullptr);
    data6.SetSymbolShape({"512", "1", "32"});

    auto relu = es::Relu(data0);
    relu.SetSymbolShape({"512", "32"});
    auto cast1 = es::Cast(relu, ge::DT_FLOAT16);
    cast1.SetSymbolShape({"512", "32"});

    auto expand_axis2 = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{2});
    expand_axis2.SetSymbolShape({"1"});
    auto expand2 = es::ExpandDims(cast1, expand_axis2);
    expand2.SetSymbolShape({"512", "32", "1"});

    auto cast3 = es::Cast(expand2, ge::DT_FLOAT);
    cast3.SetSymbolShape({"512", "32", "1"});

    auto expand_axis1 = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    expand_axis1.SetSymbolShape({"1"});
    auto expand1 = es::ExpandDims(cast1, expand_axis1);
    expand1.SetSymbolShape({"512", "1", "32"});

    auto cast2 = es::Cast(expand1, ge::DT_FLOAT);
    cast2.SetSymbolShape({"512", "1", "32"});
    auto concat = es::ConcatD({cast2, data4, data5, data6}, 1);
    concat.SetSymbolShape({"512", "4", "32"});

    es_graph_->SetOutput(concat, 0);
    es_graph_->SetOutput(cast3, 1);
  }

  void PrintComputeGraphNodes(const ComputeGraphPtr &cg) {
    for (const auto &node : cg->GetAllNodes()) {
      std::cout << "Node: " << node->GetName() << ", Type: " << node->GetType() << std::endl;
    }
  }

  void PrintAscBackendNodesInfo(const ComputeGraphPtr &cg) {
    for (const auto &node : cg->GetDirectNode()) {
      if (node->GetType() == "AscBackend") {
        auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(node);
        ASSERT_NE(autofuse_attr, nullptr);

        bool is_concat = autofuse_attr->HasFuseType(loop::FuseType::kConcat);
        std::cout << "=== AscBackend: " << node->GetName() << ", is_concat: " << is_concat << " ===" << std::endl;

        const auto attr = node->GetOpDesc()->GetAttrsGroup<ge::AutoFuseAttrs>();
        ASSERT_NE(attr, nullptr);
        ASSERT_NE(attr->GetAscGraph(), nullptr);

        for (const auto &asc_node : attr->GetAscGraph()->GetAllNodes()) {
          asc_adapt::TensorInfo tensor_desc;
          ASSERT_EQ(asc_adapt::GetTensorInfo(asc_node, tensor_desc), SUCCESS);
          std::cout << "  AscNode: " << asc_node->GetName() << ", Type: " << asc_node->GetType()
                    << ", Repeats: " << AutofuseUtils::VectorToStr(tensor_desc.repeats) << std::endl;
        }
      }
    }
  }

  void VerifyAscNodeNoSizeOneAxis(const NodePtr &asc_node, bool is_concat) {
    asc_adapt::TensorInfo tensor_desc;
    ASSERT_EQ(asc_adapt::GetTensorInfo(asc_node, tensor_desc), SUCCESS);
    std::cout << "  AscNode: " << asc_node->GetName() << ", Type: " << asc_node->GetType()
              << ", Repeats: " << AutofuseUtils::VectorToStr(tensor_desc.repeats) << std::endl;
    if (!is_concat) {
      for (size_t i = 0; i < tensor_desc.repeats.size(); ++i) {
        EXPECT_NE(tensor_desc.repeats[i], 1) << "Found size=1 axis in " << asc_node->GetName();
      }
    }
  }

  void VerifyAscBackendNode(const NodePtr &node) {
    auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(node);
    ASSERT_NE(autofuse_attr, nullptr);

    bool is_concat = autofuse_attr->HasFuseType(loop::FuseType::kConcat);
    std::cout << "=== AscBackend: " << node->GetName() << ", is_concat: " << is_concat << " ===" << std::endl;

    const auto attr = node->GetOpDesc()->GetAttrsGroup<ge::AutoFuseAttrs>();
    ASSERT_NE(attr, nullptr);
    ASSERT_NE(attr->GetAscGraph(), nullptr);

    for (const auto &asc_node : attr->GetAscGraph()->GetAllNodes()) {
      VerifyAscNodeNoSizeOneAxis(asc_node, is_concat);
    }
  }

  void VerifyFusedAscBackendNodes(const ComputeGraphPtr &cg) {
    for (const auto &node : cg->GetDirectNode()) {
      if (node->GetType() != "FusedAscBackend") {
        continue;
      }
      std::cout << "=== FusedAscBackend: " << node->GetName() << " ===" << std::endl;
      const auto attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
      auto ge_or_fused_asc_backend_graph = attr->GetFuseComputeGraph();
      for (const auto &node : ge_or_fused_asc_backend_graph->GetAllNodes()) {
        if (node->GetType() == "AscBackend") {
          VerifyAscBackendNode(node);
        }
      }
    }
  }
};

TEST_F(LoweringAndCanfuseUT, EleAndEleLoweringCanfuse) {
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
  }

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoweringAndCanfuseUT, EleAndReduceLoweringCanfuse) {
  [this]() {
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
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
}

TEST_F(LoweringAndCanfuseUT, EleAndTransposeLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_2");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, TransposeAndMulLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data1.SetSymbolShape({"s1", "s0", "s2"});
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto add = es::Add(data1, data2);
    add.SetSymbolShape({"s1", "s0", "s2"});
    auto tan = es::Tanh(add);
    tan.SetSymbolShape({"s1", "s0", "s2"});
    auto mul = es::Mul(transpose, tan);
    mul.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(mul, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(86, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(1536, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, TransposeAndTransposeLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms0 = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose0 = es::Transpose(data0, perms0);
    transpose0.SetSymbolShape({"s2", "s1", "s0"});
    auto perms1 = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose1 = es::Transpose(transpose0, perms1);
    transpose1.SetSymbolShape({"s1", "s2", "s0"});
    es_graph_->SetOutput(transpose1, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto nodes = cg->GetAllNodes();
  for (auto node : nodes) {
    string temp = node->GetName();
    cout << temp;
  }

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 融合后的计算图应该没有AscBackend节点
  size_t asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      asc_backend_node_count++;
    }
  }
  EXPECT_EQ(asc_backend_node_count, 0);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, A3BroadCastAndTransposeLoweringNoCanfuse1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s0", "s2"});
    auto abs0 = es::Abs(data1);
    abs0.SetSymbolShape({"1", "s0", "s2"});
    auto add = es::Add(abs0, transpose);
    add.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 验证融合后的计算图
  size_t asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kTranspose), 0);
      asc_backend_node_count++;
    }
  }
  EXPECT_EQ(asc_backend_node_count, 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, TransposeHorizonFuseWithElementFail) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    auto mul = es::Mul(data0, data1);
    mul.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(mul);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(transpose, 0);
    es_graph_->SetOutput(abs1, 1);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kTranspose), 0);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, ZerolikeHorizonFuseWithTransposeFail) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto zero_like = es::ZerosLike(data0);
    zero_like.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 2, 0});
    auto trans = es::Transpose(zero_like, perms);
    trans.SetSymbolShape({"s1", "s2", "s0"});
    es_graph_->SetOutput(trans, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kTranspose), 0);
    }
  }
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, EleAndTransposeLoweringCanfusePerm120) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 2, 0});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s2", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, EleAndTransposeLoweringCanfusePerm201) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 0, 1});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s2", "s0", "s1"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, InvalidTransposeLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "1"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"s0", "s1", "1"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(abs0, perms);
    transpose.SetSymbolShape({"s0", "1", "s1"});
    auto abs1 = es::Abs(transpose);
    abs1.SetSymbolShape({"s0", "1", "s1"});
    es_graph_->SetOutput(abs1, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}


TEST_F(LoweringAndCanfuseUT, MutiRefEleAndTransposeNotFuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(abs0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 0, 1});
    auto transpose = es::Transpose(abs1, perms);
    transpose.SetSymbolShape({"s2", "s0", "s1"});
    es_graph_->SetOutput(abs1, 0);
    es_graph_->SetOutput(transpose, 1);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 添加判断AscBackend节点对应的子图是否有transpose节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kTranspose), 0);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, EleAndTransposeLoweringCanfusePerm210) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 0, 1});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, EleAndTransposeLoweringCanfusePerm021) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s0", "s2", "s1"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, EleAndTransposeLoweringCanfusePerm102) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto transpose = cg->FindNode("Transpose_1");
  ASSERT_NE(transpose, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, CubeAndReshapeLoweringCanfuse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"s0", "s2"});
    auto reshape = es::Reshape(matmul, data2);
    reshape.SetSymbolShape({"1", "s0", "s2"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto reshape_1 = cg->FindNode("Reshape_1");
  ASSERT_NE(reshape_1, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, CubeAndReshapeLoweringCanfuseCanNotFuseInThisChip) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto reshape = es::Reshape(matmul, data2);
    reshape.SetSymbolShape({"1", "4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto reshape_1 = cg->FindNode("Reshape_1");
  ASSERT_NE(reshape_1, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, AbsAndSliceAndConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "6", "64"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"32", "32", "16", "64"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"32", "32", "11", "64"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"32", "32", "20", "64"});

    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"32", "32", "6", "64"});
    auto abs1 = es::Abs(data1);
    abs1.SetSymbolShape({"32", "32", "16", "64"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"32", "32", "11", "64"});
    auto abs3 = es::Abs(data3);
    abs3.SetSymbolShape({"32", "32", "20", "64"});

    const std::vector<int64_t> begin = {0, 0, 0, 0};
    const std::vector<int64_t> end = {32, 32, 10, 64};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    auto slice = es::StridedSliceD(abs2, begin, end, strides);
    slice.SetSymbolShape({"32", "32", "10", "64"});
    auto concat = es::ConcatD({abs0, abs1, slice, abs3}, 2);
    concat.SetSymbolShape({"32", "32", "52", "64"});

    auto abs5 = es::Abs(concat);
    abs5.SetSymbolShape({"32", "32", "52", "64"});
    es_graph_->SetOutput(abs5, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto concat = cg->FindNode("ConcatD_5");
  ASSERT_NE(concat, nullptr);
  BackendUtils::DumpAscGraph(concat);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, AbsAndConcatAndSlice) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "6", "64"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"32", "32", "16", "64"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"32", "32", "11", "64"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"32", "32", "20", "64"});

    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"32", "32", "6", "64"});
    auto abs1 = es::Abs(data1);
    abs1.SetSymbolShape({"32", "32", "16", "64"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"32", "32", "11", "64"});
    auto abs3 = es::Abs(data3);
    abs3.SetSymbolShape({"32", "32", "20", "64"});

    const std::vector<int64_t> begin = {0, 0, 0, 0};
    const std::vector<int64_t> end = {32, 32, 10, 64};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    auto concat = es::ConcatD({abs0, abs1, abs2, abs3}, 2);
    concat.SetSymbolShape({"32", "32", "53", "64"});
    auto slice = es::SliceD(concat, begin, end);
    slice.SetSymbolShape({"32", "32", "10", "64"});
    es_graph_->SetOutput(slice, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, ControlEdgeProcess1) {
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
  GraphUtils::AddEdge(cg->FindNode("Abs_0")->GetOutControlAnchor(), cg->FindNode("Abs_3")->GetInControlAnchor());
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
}

TEST_F(LoweringAndCanfuseUT, ControlEdgeProcess2) {
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
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
}

TEST_F(LoweringAndCanfuseUT, ControlEdgeProcess3) {
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
  (void) AttrUtils::SetBool(cons1->GetOpDesc(), "_is_from_constant_folding", false);

  GraphUtils::AddEdge(cg->FindNode("Abs_0")->GetOutControlAnchor(), cg->FindNode("FileConstant_1")->GetInControlAnchor());
  GraphUtils::AddEdge(cg->FindNode("FileConstant_1")->GetOutControlAnchor(), cg->FindNode("Abs_3")->GetInControlAnchor());
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
}

TEST_F(LoweringAndCanfuseUT, ControlEdgeProcess4) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(data);  // 3 inputs
    abs1.SetSymbolShape({"s0", "s1", "s2"});

    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto reduce = es::ReduceSum(abs1, axis, false);
    reduce.SetSymbolShape({"s0", "s2"});

    auto abs2 = es::Abs(abs1);  // 5 inputs after lowering fuse
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cons1 = cg->FindNode("FileConstant_1");
  ASSERT_NE(cons1, nullptr);

  GraphUtils::AddEdge(cg->FindNode("Abs_0")->GetOutControlAnchor(), cg->FindNode("FileConstant_1")->GetInControlAnchor());
  GraphUtils::AddEdge(cg->FindNode("FileConstant_1")->GetOutControlAnchor(), cg->FindNode("ReduceSum_2")->GetInControlAnchor());
  GraphUtils::AddEdge(cg->FindNode("ReduceSum_2")->GetOutControlAnchor(), cg->FindNode("Abs_3")->GetInControlAnchor());
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
}

REG_OP(SplitV)
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                          DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_BOOL, DT_STRING}))
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                                   DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                                   DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
                                   DT_BF16, DT_BOOL, DT_STRING}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

TEST_F(LoweringAndCanfuseUT, SplitVLoweringCanfuseStatic) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto size_splits = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{32, 32, 32});
    size_splits.SetSymbolShape({"3"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    split_dim.SetSymbolShape({"1"});
    auto split_outputs = es::SplitV(data0,size_splits,split_dim,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "32", "16"});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, TestFusedAscBackendBranchOfFunctionPreNodeInputIsSimplestLoad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"64", "96", "16"});
    auto abs1 = es::Abs(abs0);
    abs1.SetSymbolShape({"64", "96", "16"});
    auto abs2 = es::Abs(abs0);
    abs2.SetSymbolShape({"64", "96", "16"});
    // auto data1 = es_graph_->CreateInput(0, "data1", nullptr);
    // data1.SetSymbolShape({"64", "96", "16"});
    auto concat = es::ConcatD({abs1}, 2, 1);
    concat.SetSymbolShape({"64", "96", "32"});
    es_graph_->SetOutput(concat, 0);
    es_graph_->SetOutput(abs2, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(LoweringAndCanfuseUT, SliceAscBackendContainsViewCannotFuse) {
  [this]() {
    auto data1 = es_graph_->CreateInput(0, "data1", nullptr);
    data1.SetSymbolShape({"128", "300"});
    auto data2 = es_graph_->CreateInput(1, "data2", nullptr);
    data2.SetSymbolShape({"1", "1", "32"});
    auto data3 = es_graph_->CreateInput(2, "data3", nullptr);
    data3.SetSymbolShape({"128", "64", "32"});
    auto data4 = es_graph_->CreateInput(3, "data4", nullptr);
    data4.SetSymbolShape({"128", "64", "32"});
    auto slice = es::StridedSliceD(data1, {0,0}, {128,64}, {1,1});
    slice.SetSymbolShape({"128", "64"});
    auto expand_axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    expand_axis.SetSymbolShape({"1"});
    auto expand = es::ExpandDims(slice, expand_axis);
    expand.SetSymbolShape({"128", "64", "1"});
    auto mul = es::Mul(data2, data3);
    mul.SetSymbolShape({"128", "64", "32"});
    auto scalar = CreateConst(*es_graph_, ge::DT_FLOAT, {1}, std::vector<float>{1.23});
    scalar.SetSymbolShape({"1"});
    auto add = es::Add(mul,scalar);
    add.SetSymbolShape({"128", "64", "32"});
    auto sqrt = es::Sqrt(add);
    sqrt.SetSymbolShape({"128", "64", "32"});
    auto div = es::Div(expand, sqrt);
    div.SetSymbolShape({"128", "64", "32"});
    auto sub = es::Sub(div, data4);
    sub.SetSymbolShape({"128", "64", "32"});
    auto prod = es::ReduceProdD(sub,{1});
    prod.SetSymbolShape({"128", "1", "32"});
    es_graph_->SetOutput(prod, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}
TEST_F(LoweringAndCanfuseUT, SliceAndElemAndConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "16", "64"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"32", "32", "16", "64"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"32", "32", "21", "64"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"32", "32", "20", "64"});

    const std::vector<int64_t> begin0 = {0, 0, 0, 0};
    const std::vector<int64_t> end0 = {32, 32, 6, 64};
    // const std::vector<int64_t> strides = {1, 1, 1, 1};

    auto slice0 = es::SliceD(data0, begin0, end0);
    slice0.SetSymbolShape({"32", "32", "6", "64"});

    const std::vector<int64_t> begin1 = {0, 0, 0, 0};
    const std::vector<int64_t> end1 = {32, 32, 11, 64};
    auto slice1 = es::SliceD(data2, begin1, end1);
    slice1.SetSymbolShape({"32", "32", "11", "64"});

    auto abs0 = es::Abs(slice0);
    abs0.SetSymbolShape({"32", "32", "6", "64"});
    auto abs1 = es::Abs(data1);
    abs1.SetSymbolShape({"32", "32", "16", "64"});
    auto abs2 = es::Abs(slice1);
    abs2.SetSymbolShape({"32", "32", "11", "64"});
    auto abs3 = es::Abs(data3);
    abs3.SetSymbolShape({"32", "32", "20", "64"});

    auto concat = es::ConcatD({abs0, abs1, abs2, abs3}, 2);
    concat.SetSymbolShape({"32", "32", "53", "64"});

    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
}

void VerifyBroadcastNode(const NodePtr &broadcast_node, const Symbol &s0, const Symbol &s1, const Symbol &s2) {
  NodePtr broadcast_input_node;
  ASSERT_EQ(asc_adapt::GetPeerOutNode(broadcast_node, broadcast_input_node, 0), SUCCESS);
  ASSERT_EQ(broadcast_input_node->GetType(), "Broadcast");

  GeTensorDescPtr broadcast_tensor_desc;
  ASSERT_EQ(asc_adapt::GetOutputTensorDesc(broadcast_node, broadcast_tensor_desc), SUCCESS);
  auto broadcast_attr = broadcast_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  ASSERT_NE(broadcast_attr, nullptr);
  ASSERT_EQ(broadcast_attr->repeats.size(), 3);
  ASSERT_EQ(broadcast_attr->repeats[0], s0);
  ASSERT_EQ(broadcast_attr->repeats[1], s1);
  ASSERT_EQ(broadcast_attr->repeats[2], s2);
}

void VerifyReduceNode(const NodePtr &reduce_node, const Symbol &ONE, const Symbol &s1, const Symbol &s2) {
  NodePtr reduce_input_node;
  ASSERT_EQ(asc_adapt::GetPeerOutNode(reduce_node, reduce_input_node, 0), SUCCESS);
  ASSERT_EQ(reduce_input_node->GetType(), "Broadcast");

  GeTensorDescPtr reduce_tensor_desc;
  ASSERT_EQ(asc_adapt::GetOutputTensorDesc(reduce_node, reduce_tensor_desc), SUCCESS);
  auto reduce_attr = reduce_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  ASSERT_NE(reduce_attr, nullptr);
  ASSERT_EQ(reduce_attr->repeats.size(), 3);
  ASSERT_EQ(reduce_attr->repeats[0], ONE);
  ASSERT_EQ(reduce_attr->repeats[1], s1);
  ASSERT_EQ(reduce_attr->repeats[2], s2);
}

void VerifyAscBackendNode(const NodePtr &node, size_t &broadcast_cnt, bool &found_reduce, 
                          NodePtr &reduce_node, NodePtr &broadcast_node) {
  cout << kAscBackendType << "    " << node->GetName() << endl;
  const auto &op_desc = node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  const auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
  EXPECT_NE(attr, nullptr);
  EXPECT_NE(attr->GetAscGraph(), nullptr);
  for (auto asc_node : AscGraphUtils::GetComputeGraph(*(attr->GetAscGraph()))->GetDirectNode()) {
    asc_adapt::TensorInfo tensor_desc;
    ASSERT_EQ(asc_adapt::GetTensorInfo(asc_node, tensor_desc), SUCCESS);
    cout << "ascgraph node: " << asc_node->GetName() << AutofuseUtils::VectorToStr(tensor_desc.repeats).c_str() << endl;
    if (asc_node->GetType() == "Sum") {
      found_reduce = true;
      reduce_node = asc_node;
    }
    if (asc_node->GetType() == "Broadcast") {
      broadcast_cnt++;
      broadcast_node = asc_node;
    }
  }
}

void VerifyBroadcastAndReduceNodes(const ComputeGraphPtr &cg, const Symbol &s0, const Symbol &s1, const Symbol &s2,
                                     const Symbol &ONE) {
  size_t broadcast_cnt = 0;
  bool found_reduce = false;
  NodePtr reduce_node = nullptr;
  NodePtr broadcast_node = nullptr;

  for (const auto &node : cg->GetDirectNode()) {
    cout << node->GetName() << endl;
    if (node->GetType() == kAscBackendType) {
      VerifyAscBackendNode(node, broadcast_cnt, found_reduce, reduce_node, broadcast_node);

      ASSERT_TRUE(found_reduce);
      ASSERT_EQ(broadcast_cnt, 3);
      ASSERT_NE(broadcast_node, nullptr);

      VerifyBroadcastNode(broadcast_node, s0, s1, s2);
      VerifyReduceNode(reduce_node, ONE, s1, s2);
    }
  }
}

void RunCommonProcessing(const ComputeGraphPtr &cg) {
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::SaveReduceOriginalAxisToFuseAttr(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
}

// reduce不支持后融合带broadcast的elewise,不经过canfuse融合更新Fuseattr的reduce前的轴信息验证
TEST_F(LoweringAndCanfuseUT, BroadcastWithReduceInSameAxis) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "1", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto broadcast = es::BroadcastTo(data0, data1);
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(broadcast, {0}, true);
    reduce.SetSymbolShape({"1", "s1", "s2"});
    auto add = es::Add(reduce, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  RunCommonProcessing(cg);
  SetCurShapeEnvContext(nullptr);
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);

  VerifyBroadcastAndReduceNodes(cg, s0, s1, s2, ONE);
}

// reduce后融合纯elewise（不带broadcast）在canfuse做融合后的Fuseattr是否有reduce前的轴信息保存验证
TEST_F(LoweringAndCanfuseUT, BroadcastWithReduceInSameAxis2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "1", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "s1", "s2"});
    auto broadcast = es::BroadcastTo(data0, data1);
    broadcast.SetSymbolShape({"s0", "s1", "s2"});
    auto reduce = es::ReduceSumD(broadcast, {0}, true);
    reduce.SetSymbolShape({"1", "s1", "s2"});
    auto add = es::Add(reduce, data2);
    add.SetSymbolShape({"1", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(32, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(64, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  for (const auto &node : cg->GetDirectNode()) {
    cout << "lowering:" << node->GetName() << endl;
  }
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::SaveReduceOriginalAxisToFuseAttr(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);

  VerifyBroadcastAndReduceNodes(cg, s0, s1, s2, ONE);
}

TEST_F(LoweringAndCanfuseUT, ReluCastReshapeMultiRefConcat) {
  BuildReluCastReshapeMultiRefConcatGraph();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  std::cout << "=== Before Lowering ===" << std::endl;
  PrintComputeGraphNodes(cg);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  PrintAscBackendNodesInfo(cg);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  std::cout << "=== After Post Process ===" << std::endl;
  PrintComputeGraphNodes(cg);
  VerifyFusedAscBackendNodes(cg);

  SetCurShapeEnvContext(nullptr);
}
}  // namespace ge
