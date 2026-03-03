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
#include <nlohmann/json.hpp>

#include "runtime/model_v2_executor.h"
#include "lowering/model_converter.h"
#include "runtime/v2/graph_builder/bg_infer_shape.h"

// stub and faker
#include "graph_utils_ex.h"
#include "register/node_converter_registry.h"
#include "graph/utils/inference_rule.h"
#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/magic_ops.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "graph/operator_reg.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/checker.h"
#include "framework/common/ge_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "runtime/mem.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/optimize/symbolic/expect_node_info_check_test.h"

using namespace ge;
using Json = nlohmann::json;

REG_OP(ShapeRuleOp)
    .DYNAMIC_INPUT(x, TensorType({NumberType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({NumberType(), DT_VARIANT}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ShapeRuleOp);

namespace gert {
LowerResult LoweringShapeRuleOp(const ge::NodePtr &node, const LowerInput &lower_input) {
  const auto output_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *lower_input.global_data);
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  output_addrs.resize(output_shapes.size(), lower_input.input_addrs[0]);
  return {HyperStatus::Success(), {}, output_shapes, output_addrs};
}

REGISTER_NODE_CONVERTER("ShapeRuleOp", LoweringShapeRuleOp);

namespace {
ComputeGraphPtr ShapeRuleOpGraph(const std::string &rule, const bool &with_binary = true) {
  auto rule_json = Json::parse(rule);
  const size_t num_inputs = rule_json["shape"]["inputs"].size();
  const size_t num_outputs = rule_json["shape"]["outputs"].size();
  DEF_GRAPH(g1) {
    for (size_t i = 0; i < num_inputs; i++) {
      CHAIN(NODE("data" + std::to_string(i), "Data")->EDGE(0, i)->NODE("rule_op", "ShapeRuleOp"));
    }
    for (size_t i = 0; i < num_outputs; i++) {
      CHAIN(NODE("rule_op", "ShapeRuleOp")->EDGE(i, 0)->NODE("shape" + std::to_string(i), "Shape"));
    }
    for (size_t i = 0; i < num_outputs; i++) {
      CHAIN(NODE("shape" + std::to_string(i), "Shape")->EDGE(0, i)->NODE("NetOutput", "NetOutput"));
    }
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  for (size_t i = 0; i < num_inputs; i++) {
    const auto data = graph->FindNode("data" + std::to_string(i));
    AttrUtils::SetInt(data->GetOpDesc(), "index", i);
    data->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
    data->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  }

  const auto rule_op = graph->FindNode("rule_op");
  std::vector<uint8_t> binary;
  ShapeInferenceRule::CompileJsonString(rule, binary);
  AttrUtils::SetStr(rule_op->GetOpDesc(), "_inference_rule", rule);
  if (with_binary) {
    AttrUtils::SetBytes(rule_op->GetOpDesc(), "_inference_rule_binary", Buffer::CopyFrom(binary.data(), binary.size()));
  }

  std::vector<std::string> src_names;
  std::vector<int64_t> src_indexes;
  for (size_t i = 0; i < num_outputs; i++) {
    src_names.push_back("rule_op");
    src_indexes.push_back(i);
  }

  const auto net_output = graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName(src_names);
  net_output->GetOpDesc()->SetSrcIndex(src_indexes);
  net_output->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  net_output->GetOpDesc()->SetOpEngineName(kEngineNameGeLocal);
  SetGraphOutShapeRange(graph);
  return graph;
}

std::string ValueEqual(const gert::Tensor *shape_tensor, std::vector<int64_t> dims) {
  const size_t shape_size = shape_tensor->GetShapeSize();
  if (shape_size != dims.size()) {
    return "shape size not equal, expect " + std::to_string(dims.size()) + ", got " + std::to_string(shape_size);
  }
  auto *value = shape_tensor->GetData<int32_t>();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (value[i] != dims[i]) {
      return "value[" + std::to_string(i) + "] not equal, expect " + std::to_string(*(dims.begin() + i)) + ", got " +
             std::to_string(value[i]);
    }
  }

  return "";
}

class RuleMaker {
 public:
  RuleMaker() {
    json["shape"]["inputs"] = Json::array();
    json["shape"]["outputs"] = Json::array();
    json["dtype"] = Json::array();
  }

  RuleMaker &Input(const Json::array_t &input, std::initializer_list<int64_t> dims) {
    json["shape"]["inputs"].push_back(input);
    inputs.emplace_back(FakeTensors(dims, 1));
    input_ptrs.push_back(&inputs.back().at(0));
    return *this;
  }

  RuleMaker &Output(const Json::array_t &output, std::vector<int64_t> expected) {
    json["shape"]["outputs"].push_back(output);
    const auto holder = std::make_shared<std::vector<int32_t>>();
    holder->resize(expected.size());
    output_holders.emplace_back(holder);
    outputs.emplace_back(FakeTensors({static_cast<int64_t>(expected.size())}, 1, output_holders.back()->data(), kOnHost,
                                     FORMAT_ND, DT_INT32));
    output_ptrs.push_back(&outputs.back().at(0));
    expected_outputs.emplace_back(expected);
    return *this;
  }

  std::string CheckEqual() const {
    for (size_t i = 0; i < outputs.size(); i++) {
      std::string error_msg = ValueEqual(output_ptrs[i], expected_outputs[i]);
      if (!error_msg.empty()) {
        return "output[" + std::to_string(i) + "] not equal: " + error_msg;
      }
    }
    return "";
  }

  std::string Str() const {
    return json.dump();
  }

  Json json;

  std::vector<FakeTensors> inputs;
  std::vector<FakeTensors> outputs;
  std::vector<std::shared_ptr<std::vector<int32_t>>> output_holders;

  std::vector<Tensor *> input_ptrs;
  std::vector<Tensor *> output_ptrs;

  std::vector<std::vector<int64_t>> expected_outputs;
};
}  // namespace

class ShapeRuleOpST : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ShapeRuleOpST, ComplexRuleVertical) {
  RuleMaker rule_maker;
  int64_t s0 = 8;
  int64_t s1 = 24;
  const std::string rule = rule_maker.Input({"s0", "s1"}, {8, 24})
                               .Output({"s1+s0"}, {s1 + s0})
                               .Output({"s1-s0"}, {s1 - s0})
                               .Output({"s1*s0"}, {s1 * s0})
                               .Output({"Div(s1,s0)"}, {s1 / s0})
                               .Output({"Floor(Div(s1,3))"}, {s1 / 3})
                               .Output({"Ceil(Div(s1,3))"}, {(s1 + 2) / 3})
                               .Output({"Pow(s0,2)"}, {s0 * s0})
                               .Output({"Mod(s1,7)"}, {s1 % 7})
                               .Str();
  auto cg = ShapeRuleOpGraph(rule);

  GeModelBuilder builder(cg);
  auto ge_root_model = builder.BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i1.value}, rule_maker.input_ptrs.data(), rule_maker.input_ptrs.size(),
                                    rule_maker.output_ptrs.data(), rule_maker.output_ptrs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(rule_maker.CheckEqual(), "");

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(ShapeRuleOpST, ComplexRuleHorizontal) {
  RuleMaker rule_maker;
  int64_t s0 = 8;
  int64_t s1 = 24;
  const std::string rule = rule_maker.Input({"s0", "s1"}, {8, 24})
                               .Output({"s1+s0", "s1-s0", "s1*s0", "Div(s1,s0)", "Floor(Div(s1,3))", "Ceil(Div(s1,3))",
                                        "Pow(s0,2)", "Mod(s1,7)"},
                                       {s1 + s0, s1 - s0, s1 * s0, s1 / s0, s1 / 3, (s1 + 2) / 3, s0 * s0, s1 % 7})
                               .Str();
  auto cg = ShapeRuleOpGraph(rule);

  GeModelBuilder builder(cg);
  auto ge_root_model = builder.BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i1.value}, rule_maker.input_ptrs.data(), rule_maker.input_ptrs.size(),
                                    rule_maker.output_ptrs.data(), rule_maker.output_ptrs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(rule_maker.CheckEqual(), "");

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(ShapeRuleOpST, ComplexRuleVerticalJit) {
  RuleMaker rule_maker;
  int64_t s0 = 8;
  int64_t s1 = 24;
  const std::string rule = rule_maker.Input({"s0", "s1"}, {8, 24})
                               .Output({"s1+s0"}, {s1 + s0})
                               .Output({"s1-s0"}, {s1 - s0})
                               .Output({"s1*s0"}, {s1 * s0})
                               .Output({"Div(s1,s0)"}, {s1 / s0})
                               .Output({"Floor(Div(s1,3))"}, {s1 / 3})
                               .Output({"Ceil(Div(s1,3))"}, {(s1 + 2) / 3})
                               .Output({"Pow(s0,2)"}, {s0 * s0})
                               .Output({"Mod(s1,7)"}, {s1 % 7})
                               .Str();
  auto cg = ShapeRuleOpGraph(rule, false);

  GeModelBuilder builder(cg);
  auto ge_root_model = builder.BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i1.value}, rule_maker.input_ptrs.data(), rule_maker.input_ptrs.size(),
                                    rule_maker.output_ptrs.data(), rule_maker.output_ptrs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(rule_maker.CheckEqual(), "");

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(ShapeRuleOpST, ComplexRuleHorizontalJit) {
  RuleMaker rule_maker;
  int64_t s0 = 8;
  int64_t s1 = 24;
  const std::string rule = rule_maker.Input({"s0", "s1"}, {8, 24})
                               .Output({"s1+s0", "s1-s0", "s1*s0", "Div(s1,s0)", "Floor(Div(s1,3))", "Ceil(Div(s1,3))",
                                        "Pow(s0,2)", "Mod(s1,7)"},
                                       {s1 + s0, s1 - s0, s1 * s0, s1 / s0, s1 / 3, (s1 + 2) / 3, s0 * s0, s1 % 7})
                               .Str();
  auto cg = ShapeRuleOpGraph(rule, false);

  GeModelBuilder builder(cg);
  auto ge_root_model = builder.BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i1 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i1.value}, rule_maker.input_ptrs.data(), rule_maker.input_ptrs.size(),
                                    rule_maker.output_ptrs.data(), rule_maker.output_ptrs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(rule_maker.CheckEqual(), "");

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}  // namespace gert