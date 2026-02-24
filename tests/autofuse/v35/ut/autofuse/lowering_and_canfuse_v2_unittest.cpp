
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
#include "platform_context.h"
#include "base/att_const_values.h"
#include "depends/runtime/src/runtime_stub.h"

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
}  // namespace

class UTestLoweringAndCanfuseV2 : public testing::Test {
 public:
 protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_split = true;
    AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_matmul = true;
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

std::string GetAscTensorLoop(const OutDataAnchorPtr &anchor) {
  auto attr = anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(anchor->GetIdx())->GetAttrsGroup<AscTensorAttr>();
  if (attr == nullptr || (attr->axis.empty() && attr->repeats.empty() && attr->strides.empty())) {
    return "";
  }
  std::stringstream ss;
  const static auto kExpressionStr = [](const Expression &e) { return std::string(e.Str().get()); };
  ss << "axis = " << loop::StrJoin(attr->axis, [](const int64_t &e) { return std::to_string(e); });
  ss << ", repeats = " << loop::StrJoin(attr->repeats, kExpressionStr);
  ss << ", strides = " << loop::StrJoin(attr->strides, kExpressionStr);
  return ss.str();
}

std::string ReadableAscGraph(const AscGraph &asc_graph, bool trip_scope = true) {
  std::stringstream ss;
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  ss << "AscGraph(" << asc_graph.GetName() << ", axis="
     << loop::StrJoin(asc_graph.GetAllAxis(),
                      [](const AxisPtr &axis) { return std::to_string(axis->id) + ":" + axis->size.Str().get(); })
     << ")" << std::endl;
  for (const auto &node : asc_graph.GetAllNodes()) {
    std::vector<std::string> input_names;
    for (auto &anchor : node->GetAllInDataAnchors()) {
      auto peer = anchor->GetPeerOutAnchor();
      if (peer == nullptr) {
        continue;
      }
      input_names.emplace_back(anchor_name[peer]);
    }
    std::vector<std::string> output_names;
    std::map<std::string, std::string> output_loop;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
      auto loop = GetAscTensorLoop(anchor);
      if (!loop.empty()) {
        output_loop[output_names.back()] = loop;
      }
    }
    if (output_names.size() > 1U) {
      ss << loop::StrJoin(output_names) << " = ";
    } else if (!output_names.empty()) {
      ss << output_names[0] << " = ";
    }
    std::string name = node->GetName();
    if (trip_scope) {
      auto pos = name.find_last_of('/');
      if (pos != std::string::npos) {
        name = name.substr(pos + 1);
      }
    }
    ss << "ascir." << node->GetType() << "(" << name << ", " << loop::StrJoin(input_names) << ")" << std::endl;
    for (auto &loop : output_loop) {
      ss << loop.first << ".attr = {" << loop.second << "}" << std::endl;
    }
  }
  return ss.str();
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

TEST_F(UTestLoweringAndCanfuseV2, A5TransposeAndElementwiseCanFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3", "s4"});

    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {5}, std::vector<int64_t>{2, 1, 0, 4, 3});
    auto transpose = es::Transpose(data0, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0", "s4", "s3"});

    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s2", "s1", "s0", "s4", "s3"});

    auto add = es::Add(transpose, data1);
    add.SetSymbolShape({"s2", "s1", "s0", "s4", "s3"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(5, MakeShared<GraphInputShapeSourceStub>(0, 3));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<GraphInputShapeSourceStub>(0, 4));

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  // 验证融合后的计算图
  size_t asc_backend_node_count = 0;
  size_t non_asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      asc_backend_node_count++;
    } else if (node->GetType() != "Data" && node->GetType() != "NetOutput") {
      non_asc_backend_node_count++;
    }
  }
  EXPECT_EQ(asc_backend_node_count, 1);
  EXPECT_EQ(non_asc_backend_node_count, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, A5TransposeAndTransposeLoweringCanfuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
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
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  auto shape_env_attr = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  // 验证融合后的计算图
  size_t asc_backend_node_count = 0;
  size_t non_asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      asc_backend_node_count++;
    } else if (node->GetType() != "Data" && node->GetType() != "NetOutput") {
      non_asc_backend_node_count++;
    }
  }
  // 支持后开放
  // EXPECT_EQ(asc_backend_node_count, 1);
  // EXPECT_EQ(non_asc_backend_node_count, 2);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, A5ElementwiseAndTransposeCanFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
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
  size_t non_asc_backend_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == "AscBackend") {
      asc_backend_node_count++;
    } else if (node->GetType() != "Data" && node->GetType() != "NetOutput") {
      non_asc_backend_node_count++;
    }
  }
  EXPECT_EQ(asc_backend_node_count, 1);
  EXPECT_EQ(non_asc_backend_node_count, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, A5BroadCastAndTransposeLoweringCanfuse2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "1", "s2", "s3"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{0,1,3,2});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"s0", "s1", "s3", "s2"});
    es_graph_->SetOutput(transpose, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(8, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 3));
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
  std::vector<NodePtr> asc_backend_nodes;
  // 添加判断AscBackend节点对应的子图是否有transpose和broadcast节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 0);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(UTestLoweringAndCanfuseV2, A5BroadCastAndTransposeLoweringCanfuse3) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "1", "s3", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{0,1,3,2});
    auto transpose = es::Transpose(data1, perms);
    transpose.SetSymbolShape({"s0", "1", "s2", "s3"});
    auto add = es::Add(data0, transpose);
    add.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(8, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(16, MakeShared<GraphInputShapeSourceStub>(0, 3));
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
  std::vector<NodePtr> asc_backend_nodes;
  // 添加判断AscBackend节点对应的子图是否有transpose和broadcast节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kTranspose), 1);
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kBroadcast), 1);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, A5BroadCastAndTransposeLoweringCanfuse4) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s1", "s2", "s0"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{3,1,2,0});
    auto transpose = es::Transpose(data1, perms);
    transpose.SetSymbolShape({"s0", "s1", "s2", "1"});
    auto add = es::Add(data0, transpose);
    add.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(215, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(41, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(77, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto s3 = shape_env.CreateSymbol(55, MakeShared<GraphInputShapeSourceStub>(0, 3));
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
  std::vector<NodePtr> asc_backend_nodes;
  // 添加判断AscBackend节点对应的子图是否有transpose和broadcast节点的逻辑
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kTranspose), 1);
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kBroadcast), 1);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, A5MutiRefEleAndTransposeNotFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
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
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndMulLoweringCanfuseV2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s1", "s2"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "s0", "s2"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"s0", "s2"});
    auto mul = es::Mul(matmul, data2);
    mul.SetSymbolShape({"1", "s0", "s2"});
    es_graph_->SetOutput(mul, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mul_1 = cg->FindNode("Mul_1");
  ASSERT_NE(mul_1, nullptr);

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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndReshapeLoweringCanfuseV2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
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
  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto MatMulV3_1_after = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_1_after, nullptr);

  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndAddLoweringCanfuseV2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto reshape = es::Add(matmul, data2);
    reshape.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto reshape_1 = cg->FindNode("Add_1");
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, AbsAndCubeLoweringCanfuseV2CanNotFuseForward) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4", "4"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(abs, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto reshape = es::Add(matmul, data2);
    reshape.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto Abs_0 = cg->FindNode("Abs_0");
  ASSERT_NE(Abs_0, nullptr);
  auto MatMulV3_1 = cg->FindNode("MatMulV3_1");
  ASSERT_NE(MatMulV3_1, nullptr);
  auto Add_2 = cg->FindNode("Add_2");
  ASSERT_NE(Add_2, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  CubeFixpipPass cube_fixpip_pass;
  EXPECT_EQ(cube_fixpip_pass.Run(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto Abs_0_after = cg->FindNode("Abs_0");
  ASSERT_NE(Abs_0_after, nullptr);
  auto MatMulV3_1_after = cg->FindNode("MatMulV3_1");
  ASSERT_EQ(MatMulV3_1_after, nullptr);
  auto Add_2_after = cg->FindNode("Add_2");
  ASSERT_EQ(Add_2_after, nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndBroadcastLoweringCanfuseV2CanFuseBroadcast) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4"});
    auto broadcast = es::BroadcastTo(data2, data1);
    broadcast.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto reshape = es::Add(matmul, broadcast);
    reshape.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto Broadcast_0 = cg->FindNode("BroadcastTo_0");
  ASSERT_NE(Broadcast_0, nullptr);
  auto MatMulV3_1 = cg->FindNode("MatMulV3_1");
  ASSERT_NE(MatMulV3_1, nullptr);
  auto Add_2 = cg->FindNode("Add_2");
  ASSERT_NE(Add_2, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  CubeFixpipPass cube_fixpip_pass;
  EXPECT_EQ(cube_fixpip_pass.Run(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }
  auto Broadcast_0_tmp = cg->FindNode("Broadcast_0");
  ASSERT_EQ(Broadcast_0_tmp, nullptr);
  auto MatMulV3_1_tmp = cg->FindNode("MatMulV3_1");
  ASSERT_EQ(MatMulV3_1_tmp, nullptr);
  auto autofuse_pointwise_1_BroadcastTo_Add = cg->FindNode("autofuse_pointwise_1_BroadcastTo_Add");
  ASSERT_EQ(autofuse_pointwise_1_BroadcastTo_Add, nullptr);
  auto Add_2_tmp = cg->FindNode("Add_2");
  ASSERT_EQ(Add_2_tmp, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndBroadcastLoweringCanfuseV2CanNotFuseBatchBroadcast) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4", "4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4"});
    auto broadcast = es::BroadcastTo(data2, data1);
    broadcast.SetSymbolShape({"4", "4", "4", "4"});
    auto matmul = es::BatchMatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4", "4", "4"});
    auto reshape = es::Add(matmul, broadcast);
    reshape.SetSymbolShape({"4", "4", "4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto Broadcast_0 = cg->FindNode("BroadcastTo_0");
  ASSERT_NE(Broadcast_0, nullptr);
  auto MatMulV3_1 = cg->FindNode("BatchMatMulV3_1");
  ASSERT_NE(MatMulV3_1, nullptr);
  auto Add_2 = cg->FindNode("Add_2");
  ASSERT_NE(Add_2, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  CubeFixpipPass cube_fixpip_pass;
  EXPECT_EQ(cube_fixpip_pass.Run(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }
  auto Broadcast_0_tmp = cg->FindNode("Broadcast_0");
  ASSERT_EQ(Broadcast_0_tmp, nullptr);
  auto MatMulV3_1_tmp = cg->FindNode("BatchMatMulV3_1");
  ASSERT_NE(MatMulV3_1_tmp, nullptr);
  auto autofuse_pointwise_1_BroadcastTo_Add = cg->FindNode("autofuse_pointwise_1_BroadcastTo_Add");
  ASSERT_NE(autofuse_pointwise_1_BroadcastTo_Add, nullptr);
  auto Add_2_tmp = cg->FindNode("Add_2");
  ASSERT_EQ(Add_2_tmp, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

// 特殊场景vector的轴信息为m 1 n，对于matmul的m n两根轴来说看起来是batch 轴broadcast，实际可以做无效轴删除然后变成非batch轴broadcast的可融合场景
TEST_F(UTestLoweringAndCanfuseV2, CubeAndBroadcastLoweringCanfuseV2CanFuseBatchBroadcast2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "1", "4"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"4", "1", "4"});
    auto broadcast = es::BroadcastTo(data2, data3);
    broadcast.SetSymbolShape({"4", "1", "4"});
    auto reshape1 = es::Reshape(matmul, data3);
    reshape1.SetSymbolShape({"4", "1", "4"});
    auto Add = es::Add(reshape1, broadcast);
    Add.SetSymbolShape({"4", "1", "4"});
    auto reshape2 = es::Reshape(Add, data1);
    reshape2.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(reshape2, 0);
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
  CubeFixpipPass cube_fixpip_pass;
  EXPECT_EQ(cube_fixpip_pass.Run(cg), SUCCESS);

  auto Broadcast_1 = cg->FindNode("Broadcast_1");
  ASSERT_EQ(Broadcast_1, nullptr);
  auto MatMulV3_0 = cg->FindNode("MatMulV3_0");
  ASSERT_EQ(MatMulV3_0, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndBroadcastLoweringCanfuseV2CanNotFuseDirectBroadcast) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4", "4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto broadcast = es::BroadcastTo(matmul, data2);
    broadcast.SetSymbolShape({"4", "4", "4"});
    auto reshape = es::Add(broadcast, data2);
    reshape.SetSymbolShape({"4", "4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto MatMulV3_1 = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_1, nullptr);
  auto Broadcast_0 = cg->FindNode("BroadcastTo_1");
  ASSERT_NE(Broadcast_0, nullptr);
  auto Add_2 = cg->FindNode("Add_2");
  ASSERT_NE(Add_2, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  CubeFixpipPass cube_fixpip_pass;
  EXPECT_EQ(cube_fixpip_pass.Run(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }
  auto Broadcast_0_tmp = cg->FindNode("BroadcastTo_1");
  ASSERT_EQ(Broadcast_0_tmp, nullptr);
  auto MatMulV3_1_tmp = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_1_tmp, nullptr);
  auto autofuse_pointwise_1_BroadcastTo_Add = cg->FindNode("autofuse_pointwise_1_BroadcastTo_Add");
  ASSERT_NE(autofuse_pointwise_1_BroadcastTo_Add, nullptr);
  auto Add_2_tmp = cg->FindNode("Add_2");
  ASSERT_EQ(Add_2_tmp, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndScalarLoweringCanfuseV2CanFuseScalar) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4", "4"});
    auto zeroslike = es::ZerosLike(data2);
    zeroslike.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto reshape = es::Add(matmul, zeroslike);
    reshape.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(reshape, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto ZerosLike_0 = cg->FindNode("ZerosLike_0");
  ASSERT_NE(ZerosLike_0, nullptr);
  auto MatMulV3_1 = cg->FindNode("MatMulV3_1");
  ASSERT_NE(MatMulV3_1, nullptr);
  auto Add_2 = cg->FindNode("Add_2");
  ASSERT_NE(Add_2, nullptr);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto ZerosLike_0_tmp = cg->FindNode("ZerosLike_0");
  ASSERT_EQ(ZerosLike_0_tmp, nullptr);
  auto MatMulV3_1_tmp = cg->FindNode("MatMulV3_1");
  ASSERT_EQ(MatMulV3_1_tmp, nullptr);
  auto autofuse_pointwise_1_ZerosLike_Add = cg->FindNode("autofuse_pointwise_1_ZerosLike_Add");
  ASSERT_EQ(autofuse_pointwise_1_ZerosLike_Add, nullptr);
  auto Add_2_tmp = cg->FindNode("Add_2");
  ASSERT_EQ(Add_2_tmp, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndReduceLoweringCanfuseV2CanNotFuseReduce) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "16"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "16"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4", "16"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "16"});
    auto add = es::Add(matmul, data2);
    add.SetSymbolShape({"4", "16"});
    auto sum = es::ReduceSumD(add, {1}, true);
    sum.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(sum, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto MatMulV3_0 = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_0, nullptr);
  auto Add_1 = cg->FindNode("Add_1");
  ASSERT_NE(Add_1, nullptr);
  auto ReduceSumD_2 = cg->FindNode("ReduceSumD_2");
  ASSERT_NE(ReduceSumD_2, nullptr);


  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto MatMulV3_0_tmp = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_0_tmp, nullptr);
  auto autofuse_reduce_1_Add_ReduceSumD = cg->FindNode("autofuse_reduce_1_Add_ReduceSumD");
  ASSERT_NE(autofuse_reduce_1_Add_ReduceSumD, nullptr);
  auto Add_1_tmp = cg->FindNode("Add_1");
  ASSERT_EQ(Add_1_tmp, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndElementwiseLoweringCanfuseV2CanNotFuseVertical) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto add = es::Add(data1, data2);
    add.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(matmul, 0);
    es_graph_->SetOutput(add, 1);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto MatMulV3_0 = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_0, nullptr);
  auto Add_1 = cg->FindNode("Add_1");
  ASSERT_NE(Add_1, nullptr);


  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto MatMulV3_0_tmp = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_0_tmp, nullptr);
  auto Add_1_tmp = cg->FindNode("Add_1");
  ASSERT_NE(Add_1_tmp, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndReduceLoweringCanfuseReluAbs) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto add = es::Relu(matmul);
    add.SetSymbolShape({"4", "4"});
    auto sum = es::Abs(add);
    sum.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(sum, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto MatMulV3_0 = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_0, nullptr);
  auto Add_1 = cg->FindNode("Relu_1");
  ASSERT_NE(Add_1, nullptr);
  auto ReduceSumD_2 = cg->FindNode("Abs_2");
  ASSERT_NE(ReduceSumD_2, nullptr);


  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto MatMulV3_0_tmp = cg->FindNode("MatMulV3_0");
  ASSERT_EQ(MatMulV3_0_tmp, nullptr);
  auto autofuse_reduce_1_Add_ReduceSumD = cg->FindNode("autofuse_pointwise_1_Relu_Abs");
  ASSERT_EQ(autofuse_reduce_1_Add_ReduceSumD, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, CubeAndReduceLoweringCanfuseOnlyRelu) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4"});
    auto matmul = es::MatMulV3(data0, data1);
    matmul.SetSymbolShape({"4", "4"});
    auto add = es::Relu(matmul);
    add.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(add, 0);
  }();
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto MatMulV3_0 = cg->FindNode("MatMulV3_0");
  ASSERT_NE(MatMulV3_0, nullptr);
  auto Add_1 = cg->FindNode("Relu_1");
  ASSERT_NE(Add_1, nullptr);


  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  for (auto &node : cg->GetDirectNode()) {
    std::cout << "node name is " << node->GetNamePtr() << std::endl;
  }

  auto MatMulV3_0_tmp = cg->FindNode("MatMulV3_0");
  ASSERT_EQ(MatMulV3_0_tmp, nullptr);
  auto autofuse_reduce_1_Add_ReduceSumD = cg->FindNode("autofuse_pointwise_1_Relu");
  ASSERT_EQ(autofuse_reduce_1_Add_ReduceSumD, nullptr);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, EleAndSplitLoweringCanfuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitAndSplitLoweringCanfuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split_outputs1 = es::SplitD(data0,1,3);
    for (auto output: split_outputs1) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  auto split1 = cg->FindNode("SplitD_1");
  ASSERT_NE(split1, nullptr);

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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, EleAndSplitLoweringCanfuseStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto split_outputs = es::SplitD(data0,1,3);
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

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, ReluAndSplitHorizontalLoweringCanfuseStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "32", "16"});
      es_graph_->SetOutput(esb_out,index++);
    }
    auto relu = es::Relu(data0);
    relu.SetSymbolShape({"64", "96", "16"});
    es_graph_->SetOutput(relu.GetEsbTensor(),index++);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);


  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitAndEleLoweringCanfuseStaticNoLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    split_outputs[0].SetSymbolShape({"64", "32", "16"});
    es_graph_->SetOutput(split_outputs[0],0);

    split_outputs[1].SetSymbolShape({"64", "32", "16"});
    auto abs0 = es::Abs(split_outputs[1]);
    abs0.SetSymbolShape({"64", "32", "16"});
    es_graph_->SetOutput(abs0,1);

    split_outputs[2].SetSymbolShape({"64", "32", "16"});
    auto abs1 = es::Abs(split_outputs[2]);
    abs1.SetSymbolShape({"64", "32", "16"});
    es_graph_->SetOutput(abs1,2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

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
  RuntimeStub::Reset();
}

int DumpAllAscGraphs(const ComputeGraphPtr &cg, std::string s) {
  for (auto node: cg->GetAllNodes()) {
    GELOGD("node: %s(%s), AscGraph: %s", node->GetName().c_str(), node->GetType().c_str(), s.c_str());
    BackendUtils::DumpAscGraph(node);
  }
  return 0;
}

TEST_F(UTestLoweringAndCanfuseV2, SingleSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
    auto split_outputs = es::SplitD(data0,1,3);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      esb_out->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
      es_graph_->SetOutput(esb_out,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SingleGiantSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "4096", "20"});
    auto split_outputs = es::SplitD(data0,1,1024);
    int index = 0 ;
    for (auto output: split_outputs) {
      output.SetSymbolShape({"32", "4", "20"});

      es_graph_->SetOutput(output,index++);
    }
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitSingleOutputDoubleQuotesLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_outputs = es::SplitD(data0,1,1);
    int index = 0 ;
    for (auto output: split_outputs) {
      output.SetSymbolShape({"32", "64", "20"});
    }
    auto abs0 = es::Abs(split_outputs[0]);
    auto abs1 = es::Abs(split_outputs[0]);
    abs0.SetSymbolShape({"32", "64", "20"});
    abs1.SetSymbolShape({"32", "64", "20"});
    es_graph_->SetOutput(abs0,index++);
    es_graph_->SetOutput(abs1,index++);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitSingleOutputDoubleQuotesLoweringCanfuseLiftingNotFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_outputs = es::SplitD(data0,1,1);
    int index = 0 ;
    for (auto output: split_outputs) {
      output.SetSymbolShape({"32", "64", "20"});
    }
    auto reduce_axis0 = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{0});
    auto reduce0 = es::ReduceAll(split_outputs[0],reduce_axis0);
    reduce0.SetSymbolShape({"64", "20"});
    auto reduce1 = es::ReduceAll(split_outputs[0],reduce_axis0);
    reduce1.SetSymbolShape({"64", "20"});
    es_graph_->SetOutput(reduce0,index++);
    es_graph_->SetOutput(reduce1,index++);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto split = cg->FindNode("SplitD_0");
  ASSERT_NE(split, nullptr);

  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
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
TEST_F(UTestLoweringAndCanfuseV2, SplitVLoweringCanfuseStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
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
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitVLiftingErrorStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"64", "96", "16"});
    auto size_splits = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{32, 32, 32});
    size_splits.SetSymbolShape({"3"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    split_dim.SetSymbolShape({"1"});
    auto split_outputs = es::SplitV(abs,size_splits,split_dim,3);
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
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SingleSplitVLiftingErrorStatic) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"64", "96", "16"});
    auto abs0 = es::Abs(data0);
    abs0.SetSymbolShape({"64", "96", "16"});
    auto abs1 = es::Abs(abs0);
    abs1.SetSymbolShape({"64", "96", "16"});
    auto size_splits = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{96});
    size_splits.SetSymbolShape({"1"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    split_dim.SetSymbolShape({"1"});
    auto split_outputs = es::SplitV(abs1,size_splits,split_dim,1);
    int index = 0 ;
    for (auto output: split_outputs) {
      auto esb_out = output.GetEsbTensor();
      // 上边这种写法产生的不是ConstExpr
      // esb_out->SetSymbolShape({Symbol("64"), Symbol("32"), Symbol("16")});
      output.SetSymbolShape({"64", "96", "16"});
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, FlattenSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,2);
    int index = 0 ;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "20"});
      auto esb_out = output.GetEsbTensor();
    }
    auto split1_outputs = es::Split(split_dim,split0_outputs[0],4);
    for (auto output: split1_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split2_outputs = es::Split(split_dim,split0_outputs[1],4);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, FlattenSplitVLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto size_splits0 = CreateConst(*es_graph_, ge::DT_INT64, {2}, std::vector<int64_t>{32,32});
    auto split0_outputs = es::SplitV(data0,size_splits0,split_dim,2);
    int index = 0 ;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "20"});
      auto esb_out = output.GetEsbTensor();
    }
    auto size_splits1 = CreateConst(*es_graph_, ge::DT_INT64, {4}, std::vector<int64_t>{8,8,8,8});
    auto split1_outputs = es::SplitV(split0_outputs[0],size_splits1,split_dim,4);
    for (auto output: split1_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split2_outputs = es::SplitV(split0_outputs[1],size_splits1,split_dim,4);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, FlattenSplitDLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "64", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::SplitD(data0,1,2);
    int index = 0 ;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "20"});
      auto esb_out = output.GetEsbTensor();
    }
    auto split1_outputs = es::SplitD(split0_outputs[0],1,4);
    for (auto output: split1_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
      es_graph_->SetOutput(esb_out,index++);
    }
    auto split2_outputs = es::SplitD(split0_outputs[1],1,4);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"32", "8", "20"});
      auto esb_out = output.GetEsbTensor();
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
  dlog_setlevel(0, 4, 1);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  dlog_setlevel(0, 0, 1);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, HierarchicalFlattenSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"512", "64", "20"});
    data0.GetEsbTensor()->SetSymbolShape({Symbol("512"), Symbol("64"), Symbol("20")});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{0});
    auto split0_outputs = es::Split(split_dim,data0,2);
    split0_outputs[0].SetSymbolShape({"256", "64", "20"});
    split0_outputs[0].GetEsbTensor()->SetSymbolShape({Symbol("256"), Symbol("64"), Symbol("20")});
    split0_outputs[1].SetSymbolShape({"256", "64", "20"});
    auto split1_outputs = es::Split(split_dim,split0_outputs[0],2);
    split1_outputs[0].SetSymbolShape({"128", "64", "20"});
    split1_outputs[1].SetSymbolShape({"128", "64", "20"});
    auto split2_outputs = es::Split(split_dim,split1_outputs[0],2);
    for (auto output: split2_outputs) {
      output.SetSymbolShape({"64", "64", "20"});
    }
    auto split3_outputs = es::Split(split_dim,split2_outputs[0],2);
    split3_outputs[0].SetSymbolShape({"32", "64", "20"});
    split3_outputs[1].SetSymbolShape({"32", "64", "20"});
    auto split4_outputs = es::Split(split_dim,split2_outputs[1],2);
    split4_outputs[0].SetSymbolShape({"32", "64", "20"});
    split4_outputs[1].SetSymbolShape({"32", "64", "20"});
    es_graph_->SetOutput(split3_outputs[0],0);
    es_graph_->SetOutput(split4_outputs[0],1);
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, GiantSplitLoweringCanfuseLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "4096", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,32);
    size_t index = 0U;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "128", "20"});
      auto split2_outputs = es::Split(split_dim,output,32);
      for (auto output: split2_outputs) {
        output.SetSymbolShape({"64", "4", "20"});
        es_graph_->SetOutput(output,index++);
      }
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, GiantSplitLoweringCanfuseNoLifting) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "512", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,4);
    size_t index = 0U;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "128", "20"});
      auto split2_outputs = es::Split(split_dim,output,32);
      size_t count = 0;
      for (auto output: split2_outputs) {
        output.SetSymbolShape({"64", "4", "20"});
        if (count == 0U) {
          auto abs = es::Abs(output);
          abs.SetSymbolShape({"64", "4", "20"});
          es_graph_->SetOutput(abs,index++);
        } else {
          es_graph_->SetOutput(output,index++);
        }
        count++;
      }
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, GiantSplitLoweringCanfuseNoLowering) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "4096", "20"});
    auto split_dim = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{1});
    auto split0_outputs = es::Split(split_dim,data0,32);
    size_t index = 0U;
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "128", "20"});
      auto split2_outputs = es::Split(split_dim,output,32);
      size_t count = 0;
      for (auto output: split2_outputs) {
        output.SetSymbolShape({"64", "4", "20"});
        if (count == 0U) {
          auto abs = es::Abs(output);
          abs.SetSymbolShape({"64", "4", "20"});
          es_graph_->SetOutput(abs,index++);
        } else {
          es_graph_->SetOutput(output,index++);
        }
        count++;
      }
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitAndAddNoBroadcast) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"20", "20", "20"});
    auto split_outputs = es::SplitD(data0,1,2);
    split_outputs[0].SetSymbolShape({"1","20","20"});
    split_outputs[1].SetSymbolShape({"19","20","20"});
    es_graph_->SetOutput(split_outputs[1],1);
    auto add0=es::Add(data0,split_outputs[0]);
    add0.SetSymbolShape({"20","20","20"});
    es_graph_->SetOutput(add0,0);
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
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitAndConcatAndAbsSplitPartialFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "96", "64"});
    auto split0_outputs = es::SplitD(data0,2U,3);
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "32", "64"});
    }
    auto abs0 = es::Abs(split0_outputs[2]);
    abs0.SetSymbolShape({"32", "32", "32", "64"});
    auto reduce_axis0 = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{3});
    auto reduce0 = es::ReduceAny(split0_outputs[0],reduce_axis0);
    reduce0.SetSymbolShape({"32", "32", "32"});
    auto reduce_axis1 = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{3});
    auto reduce1 = es::ReduceAny(split0_outputs[1],reduce_axis1);
    reduce1.SetSymbolShape({"32", "32", "32"});
    es_graph_->SetOutput(reduce0, 0);
    es_graph_->SetOutput(reduce1, 1);
    es_graph_->SetOutput(abs0, 2);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto prev_node_num = cg->GetDirectNode().size();
  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  auto post_node_num = cg->GetDirectNode().size();
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  EXPECT_EQ(prev_node_num - 1, post_node_num);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

TEST_F(UTestLoweringAndCanfuseV2, SplitAndReduceAndAbsSplitLowFuseRatio) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"32", "32", "96", "64"});
    auto split0_outputs = es::SplitD(data0,2U,6);
    for (auto output: split0_outputs) {
      output.SetSymbolShape({"32", "32", "32", "64"});
    }
    auto abs0 = es::Abs(split0_outputs[0]);
    abs0.SetSymbolShape({"32", "32", "16", "64"});
    for (int32_t i = 0; i < 5U; i++) {
      auto reduce_axis = CreateConst(*es_graph_,DT_INT64,{1},std::vector<int64_t>{3});
      auto reduce = es::ReduceAny(split0_outputs[i],reduce_axis);
      reduce.SetSymbolShape({"32", "32", "16"});
      es_graph_->SetOutput(reduce, i);
    }
    es_graph_->SetOutput(abs0, 5);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto prev_node_num = cg->GetDirectNode().size();
  ge::PatternFusion patter_fusion;
  ASSERT_EQ(patter_fusion.RunAllPatternFusion(cg),GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  auto post_node_num = cg->GetDirectNode().size();
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  EXPECT_EQ(prev_node_num, post_node_num);
  SetCurShapeEnvContext(nullptr);
  RuntimeStub::Reset();
}

// Gather + Elewis, Elewis包含Broadcast信息，Gather未与Broadcast直连，可以融合
TEST_F(UTestLoweringAndCanfuseV2, GatherAddBroadcastFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"130", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "200", "200"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"128", "200", "1", "1"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"128", "200", "200", "1"});
    auto add = es::Add(gather, data2);
    add.SetSymbolShape({"128", "200", "200", "1"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kGather), 1);
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kBroadcast), 1);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather + Elewis, Elewis包含Broadcast信息，Gather与Broadcast直连，不可融合
TEST_F(UTestLoweringAndCanfuseV2, GatherAddBroadcastFuseInvalid) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"130", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "200", "200"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"128", "200", "200", "200"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"128", "200", "200", "1"});
    auto add = es::Add(gather, data2);
    add.SetSymbolShape({"128", "200", "200", "200"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);

  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 2);

  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 0);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather + Reduce, Reduce包含Broadcast信息，Gather未与Broadcast直连，可以融合
TEST_F(UTestLoweringAndCanfuseV2, GatherReduceBroadcastFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"130", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "200", "200"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"128", "200", "1", "1"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"128", "200", "200", "1"});
    auto add = es::Add(gather, data2);
    add.SetSymbolShape({"128", "200", "200", "1"});
    auto sum = es::ReduceSumD(add, {1}, true);
    sum.SetSymbolShape({"128", "1", "200", "1"});

    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kSum), 1);
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kBroadcast), 1);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather + Reduce, Reduce包含Broadcast信息，Gather与Broadcast直连，不可以融合
TEST_F(UTestLoweringAndCanfuseV2, GatherReduceBroadcastFuseInvalid) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"130", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "200", "200"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"128", "200", "200", "100"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"128", "200", "200", "1"});
    auto add = es::Add(gather, data2);
    add.SetSymbolShape({"128", "200", "200", "100"});
    auto sum = es::ReduceSumD(add, {1}, true);
    sum.SetSymbolShape({"128", "1", "200", "100"});

    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kSum), 1);
      ASSERT_EQ(AscSubgraphNodeCount(node, att::kBroadcast), 1);
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather + Elewis + Squeeze, 不可以融合
TEST_F(UTestLoweringAndCanfuseV2, GatherAddSqueezeFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"130", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "200", "200"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"128", "200", "200"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"128", "200", "200", "1"});
    auto squeeze = es::Squeeze(gather, {3});
    squeeze.SetSymbolShape({"128", "200", "200"});
    auto add = es::Add(squeeze, data2);
    add.SetSymbolShape({"128", "200", "200"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 0);

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather + Broc + Concat, 不可以融合
TEST_F(UTestLoweringAndCanfuseV2, GatherBrocConcatFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"10", "6"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"128", "1", "6"});
    auto multiples = es_graph_->CreateVector({128, 1, 1});
    multiples.SetSymbolShape({"3"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"1", "1", "6"});
    auto tile = es::Tile(gather, multiples);
    tile.SetSymbolShape({"128", "1", "6"});
    auto concat = es::ConcatD({tile, data2}, 1);
    concat.SetSymbolShape({"128", "2", "6"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  auto gather_nodeptr = cg->FindNode("GatherV2_2");
  ASSERT_NE(gather_nodeptr, nullptr);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);
  gather_nodeptr = cg->FindNode("GatherV2_2");
  ASSERT_NE(gather_nodeptr, nullptr);
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType || node->GetType() == kFusedAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1); // 存在一个tile + concat的融合算子

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather + Concat, Concat与Gather直连分支上不包含Broc信息，可以融合
TEST_F(UTestLoweringAndCanfuseV2, GatherBrocConcatFuse2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"10", "6"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"1", "1", "6"});
    auto multiples = es_graph_->CreateVector({128, 1, 1});
    multiples.SetSymbolShape({"3"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"128", "1", "6"});
    auto tile = es::Tile(data2, multiples);
    tile.SetSymbolShape({"128", "1", "6"});
    auto concat = es::ConcatD({gather, tile}, 1);
    concat.SetSymbolShape({"128", "2", "6"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  auto gather_nodeptr = cg->FindNode("GatherV2_2");
  ASSERT_NE(gather_nodeptr, nullptr);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(cg), SUCCESS);

  gather_nodeptr = cg->FindNode("GatherV2_2");
  ASSERT_EQ(gather_nodeptr, nullptr);
  size_t asc_node_count = 0;
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kAscBackendType || node->GetType() == kFusedAscBackendType) {
      asc_node_count++;
    }
  }
  ASSERT_EQ(asc_node_count, 1); // 存在一个tile + concat + gather的融合算子

  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

// Gather/Gather + Concat 辅助函数
// 构建 Gather+Gather+Tile+Concat 计算图
static void BuildGatherGatherConcatGraph(es::Graph &es_graph) {
  auto data0 = es_graph.CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"10", "6"});
  auto data1 = es_graph.CreateInput(1, "data1", nullptr);
  data1.SetSymbolShape({"128", "1"});
  auto data2 = es_graph.CreateInput(2, "data2", nullptr);
  data2.SetSymbolShape({"1", "1", "6"});
  auto multiples = es_graph.CreateVector({128, 1, 1});
  multiples.SetSymbolShape({"3"});
  auto axis = CreateConst(es_graph, ge::DT_INT64, {1}, std::vector<int64_t>{0});
  axis.SetSymbolShape({});
  auto gather1 = es::GatherV2(data0, data1, axis);
  gather1.SetSymbolShape({"128", "1", "6"});
  auto gather2 = es::GatherV2(data0, data1, axis);
  gather2.SetSymbolShape({"128", "1", "6"});
  auto tile = es::Tile(data2, multiples);
  tile.SetSymbolShape({"128", "1", "6"});
  auto concat = es::ConcatD({gather1, gather2, tile}, 1);
  concat.SetSymbolShape({"128", "3", "6"});
  es_graph.SetOutput(concat, 0);
}

// 验证 GatherGatherConcat 融合后的结果
static void VerifyGatherGatherConcatFuseResult(const ComputeGraphPtr &cg) {
  for (const auto &node : cg->GetDirectNode()) {
    if (node->GetType() == kFusedAscBackendType) {
      const auto attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
      auto autofuse_gather_0_GatherV2 = attr->GetFuseComputeGraph()->FindNode("autofuse_gather_0_GatherV2");
      ASSERT_NE(autofuse_gather_0_GatherV2, nullptr);
      auto autofuse_gather_1_GatherV2 = attr->GetFuseComputeGraph()->FindNode("autofuse_gather_1_GatherV2");
      ASSERT_NE(autofuse_gather_1_GatherV2, nullptr);
    }
  }
}

// 执行完整的 Lowering/Fusion/Lifting/Post-process 流程
static Status ProcessGraphWithFullPipeline(ComputeGraphPtr &cg) {
  ge::AscIrLowerer lowerer;
  GE_ASSERT_SUCCESS(lowerer.Lowering(cg));
  GE_ASSERT_SUCCESS(asc_adapt::GeFallback(cg));
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  GE_ASSERT_SUCCESS(fusion_strategy_solver.Fuse(cg));
  GE_ASSERT_SUCCESS(lowerer.Lifting(cg));
  AscBackendPostProcessor post_processor;
  return post_processor.Do(cg);
}

// Gather/Gather + Concat, 可以融合,但是FusedAscBackend内部的Gather不能融合
TEST_F(UTestLoweringAndCanfuseV2, GatherGatherConcatFuse) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);

  // 构建测试图
  BuildGatherGatherConcatGraph(*es_graph_);

  // 获取计算图并设置数据类型
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);

  // 执行完整的 lowering/fusion/lifting/post-process 流程
  EXPECT_EQ(ProcessGraphWithFullPipeline(cg), SUCCESS);

  // 验证融合结果
  VerifyGatherGatherConcatFuseResult(cg);

  // 清理环境
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}
}  // namespace ge