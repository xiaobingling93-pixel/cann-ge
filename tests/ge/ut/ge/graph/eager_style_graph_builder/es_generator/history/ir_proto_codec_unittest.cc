/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history/ir_proto_codec.h"

#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "default_attr_utils.h"
#include "graph/operator_reg.h"
#include "op_desc_utils.h"

using namespace ge;
using namespace ge::es::history;
namespace {
nlohmann::json PhonyAllOpJson() {
  return {
      {"op_type", "phony_all"},
      {"inputs", nlohmann::json::array({
                     {{"name", "x"}, {"type", "INPUT"}},
                     {{"name", "xo"}, {"type", "OPTIONAL_INPUT"}},
                     {{"name", "dx"}, {"type", "DYNAMIC_INPUT"}},
                 })},
      {"outputs", nlohmann::json::array({
                      {{"name", "y"}, {"type", "OUTPUT"}},
                      {{"name", "dy"}, {"type", "DYNAMIC_OUTPUT"}},
                 })},
      {"subgraphs", nlohmann::json::array({
                 {{"name", "g"}, {"type", "STATIC"}},
                 {{"name", "dg"}, {"type", "DYNAMIC"}},
                 })},
      {"attrs", nlohmann::json::array({
                    {{"name", "ri"}, {"type", "Int"}, {"required", true}},
                    {{"name", "i"}, {"type", "Int"}, {"required", false}, {"default_value", "7"}},
                    {{"name", "f"}, {"type", "Float"}, {"required", false}, {"default_value", "1.0"}},
                    {{"name", "s"}, {"type", "String"}, {"required", false}, {"default_value", "\"a\""}},
                    {{"name", "b"}, {"type", "Bool"}, {"required", false}, {"default_value", "true"}},
                    {{"name", "dt"}, {"type", "Type"}, {"required", false}, {"default_value", "\"DT_INT64\""}},
                    {{"name", "li"}, {"type", "ListInt"}, {"required", false}, {"default_value", "[1,-2]"}},
                    {{"name", "lf"}, {"type", "ListFloat"}, {"required", false}, {"default_value", "[1.0,9.99999993922529e-09]"}},
                    {{"name", "lb"}, {"type", "ListBool"}, {"required", false}, {"default_value", "[true,false]"}},
                    {{"name", "ldt"}, {"type", "ListType"}, {"required", false}, {"default_value", "[\"DT_FLOAT\",\"DT_DOUBLE\"]"}},
                    {{"name", "lli"}, {"type", "ListListInt"}, {"required", false}, {"default_value", "[[],[2,3],[]]"}},
                    {{"name", "t"}, {"type", "Tensor"}, {"required", false}, {"default_value", "\"Tensor()\""}},
                    {{"name", "ls"}, {"type", "ListString"}, {"required", false}, {"default_value", "[\"a\",\"b\"]"}},
                    {{"name", "rlb"}, {"type", "ListBool"}, {"required", true}},
                })},
  };
}

const IrAttr *FindAttr(const std::vector<IrAttr> &attrs, const std::string &name) {
  for (const auto &attr : attrs) {
    if (attr.name == name) {
      return &attr;
    }
  }
  return nullptr;
}

void ExpectPhonyAllInputs(const IrOpProto &proto) {
  ASSERT_EQ(proto.inputs.size(), 3U);
  EXPECT_EQ(proto.inputs[0].name, "x");
  EXPECT_EQ(proto.inputs[0].type, kIrInputRequired);
  EXPECT_EQ(proto.inputs[1].name, "xo");
  EXPECT_EQ(proto.inputs[1].type, kIrInputOptional);
  EXPECT_EQ(proto.inputs[2].name, "dx");
  EXPECT_EQ(proto.inputs[2].type, kIrInputDynamic);
}

void ExpectPhonyAllOutputs(const IrOpProto &proto) {
  ASSERT_EQ(proto.outputs.size(), 2U);
  EXPECT_EQ(proto.outputs[0].name, "y");
  EXPECT_EQ(proto.outputs[0].type, kIrOutputRequired);
  EXPECT_EQ(proto.outputs[1].name, "dy");
  EXPECT_EQ(proto.outputs[1].type, kIrOutputDynamic);
}

void ExpectPhonyAllSubgraphs(const IrOpProto &proto) {
  ASSERT_EQ(proto.subgraphs.size(), 2U);
  EXPECT_EQ(proto.subgraphs[0].name, "g");
  EXPECT_EQ(proto.subgraphs[0].type, kStatic);
  EXPECT_EQ(proto.subgraphs[1].name, "dg");
  EXPECT_EQ(proto.subgraphs[1].type, kDynamic);
}

void ExpectPhonyAllRequiredAttrs(const IrOpProto &proto) {
  EXPECT_EQ(proto.attrs.front().name, "ri");
  const auto *attr_ri = FindAttr(proto.attrs, "ri");
  ASSERT_NE(attr_ri, nullptr);
  EXPECT_EQ(attr_ri->av_type, "Int");
  EXPECT_TRUE(attr_ri->required);
  EXPECT_EQ(attr_ri->default_value, "");

  EXPECT_EQ(proto.attrs.back().name, "rlb");
  const auto *attr_rlb = FindAttr(proto.attrs, "rlb");
  ASSERT_NE(attr_rlb, nullptr);
  EXPECT_EQ(attr_rlb->av_type, "ListBool");
  EXPECT_TRUE(attr_rlb->required);
  EXPECT_EQ(attr_rlb->default_value, "");
}

void ExpectPhonyAllScalarOptionalAttrs(const IrOpProto &proto) {
  const auto *attr_i = FindAttr(proto.attrs, "i");
  ASSERT_NE(attr_i, nullptr);
  EXPECT_EQ(attr_i->av_type, "Int");
  EXPECT_FALSE(attr_i->required);
  EXPECT_EQ(attr_i->default_value, "7");
  EXPECT_EQ(nlohmann::json::parse(attr_i->default_value).get<int64_t>(), 7);

  const auto *attr_f = FindAttr(proto.attrs, "f");
  ASSERT_NE(attr_f, nullptr);
  EXPECT_EQ(attr_f->av_type, "Float");
  EXPECT_FALSE(attr_f->required);
  EXPECT_EQ(attr_f->default_value, "1.0");
  EXPECT_FLOAT_EQ(nlohmann::json::parse(attr_f->default_value).get<float>(), 1.0);

  const auto *attr_s = FindAttr(proto.attrs, "s");
  ASSERT_NE(attr_s, nullptr);
  EXPECT_EQ(attr_s->av_type, "String");
  EXPECT_FALSE(attr_s->required);
  EXPECT_EQ(attr_s->default_value, "\"a\"");
  EXPECT_EQ(nlohmann::json::parse(attr_s->default_value).get<std::string>(), "a");

  const auto *attr_b = FindAttr(proto.attrs, "b");
  ASSERT_NE(attr_b, nullptr);
  EXPECT_EQ(attr_b->av_type, "Bool");
  EXPECT_FALSE(attr_b->required);
  EXPECT_EQ(attr_b->default_value, "true");
  EXPECT_EQ(nlohmann::json::parse(attr_b->default_value).get<bool>(), true);

  const auto *attr_dt = FindAttr(proto.attrs, "dt");
  ASSERT_NE(attr_dt, nullptr);
  EXPECT_EQ(attr_dt->av_type, "Type");
  EXPECT_FALSE(attr_dt->required);
  EXPECT_EQ(attr_dt->default_value, "\"DT_INT64\"");
  EXPECT_EQ(nlohmann::json::parse(attr_dt->default_value).get<ge::DataType>(), DT_INT64);
}

void ExpectPhonyAllListOptionalAttrs(const IrOpProto &proto) {
  const auto *attr_li = FindAttr(proto.attrs, "li");
  ASSERT_NE(attr_li, nullptr);
  EXPECT_EQ(attr_li->av_type, "ListInt");
  EXPECT_FALSE(attr_li->required);
  EXPECT_EQ(attr_li->default_value, "[1,-2]");
  EXPECT_EQ(nlohmann::json::parse(attr_li->default_value).get<std::vector<int64_t>>(),
            std::vector<int64_t>({1, -2}));

  const auto *attr_lf = FindAttr(proto.attrs, "lf");
  ASSERT_NE(attr_lf, nullptr);
  EXPECT_EQ(attr_lf->av_type, "ListFloat");
  EXPECT_FALSE(attr_lf->required);
  EXPECT_EQ(attr_lf->default_value, "[1.0,9.99999993922529e-09]");
  const auto lf_parsed = nlohmann::json::parse(attr_lf->default_value).get<std::vector<float>>();
  const std::vector<float> lf_expected{1.0f, 1e-08f};
  ASSERT_EQ(lf_parsed.size(), lf_expected.size());
  for (size_t i = 0; i < lf_parsed.size(); ++i) {
    EXPECT_FLOAT_EQ(lf_parsed[i], lf_expected[i]);
  }

  const auto *attr_lb = FindAttr(proto.attrs, "lb");
  ASSERT_NE(attr_lb, nullptr);
  EXPECT_EQ(attr_lb->av_type, "ListBool");
  EXPECT_FALSE(attr_lb->required);
  EXPECT_EQ(attr_lb->default_value, "[true,false]");
  EXPECT_EQ(nlohmann::json::parse(attr_lb->default_value).get<std::vector<bool>>(),
            std::vector<bool>({true, false}));

  const auto *attr_ldt = FindAttr(proto.attrs, "ldt");
  ASSERT_NE(attr_ldt, nullptr);
  EXPECT_EQ(attr_ldt->av_type, "ListType");
  EXPECT_FALSE(attr_ldt->required);
  EXPECT_EQ(attr_ldt->default_value, "[\"DT_FLOAT\",\"DT_DOUBLE\"]");
  EXPECT_EQ(nlohmann::json::parse(attr_ldt->default_value).get<std::vector<ge::DataType>>(),
            std::vector<ge::DataType>({DT_FLOAT, DT_DOUBLE}));

  const auto *attr_tli = FindAttr(proto.attrs, "lli");
  ASSERT_NE(attr_tli, nullptr);
  EXPECT_EQ(attr_tli->av_type, "ListListInt");
  EXPECT_FALSE(attr_tli->required);
  EXPECT_EQ(attr_tli->default_value, "[[],[2,3],[]]");
  EXPECT_EQ(nlohmann::json::parse(attr_tli->default_value).get<std::vector<std::vector<int64_t>>>(),
            std::vector<std::vector<int64_t>>({{}, {2, 3}, {}}));

  const auto *attr_ls = FindAttr(proto.attrs, "ls");
  ASSERT_NE(attr_ls, nullptr);
  EXPECT_EQ(attr_ls->av_type, "ListString");
  EXPECT_FALSE(attr_ls->required);
  EXPECT_EQ(attr_ls->default_value, "[\"a\",\"b\"]");
  EXPECT_EQ(nlohmann::json::parse(attr_ls->default_value).get<std::vector<std::string>>(),
            std::vector<std::string>({"a", "b"}));
}

void ExpectPhonyAllTensorAttr(const IrOpProto &proto) {
  const auto *attr_t = FindAttr(proto.attrs, "t");
  ASSERT_NE(attr_t, nullptr);
  EXPECT_EQ(attr_t->av_type, "Tensor");
  EXPECT_FALSE(attr_t->required);
  EXPECT_EQ(attr_t->default_value, "\"Tensor()\"");
  auto tensor = std::make_shared<ge::GeTensor>(ge::GeTensor());
  EXPECT_EQ(nlohmann::json::parse(attr_t->default_value).get<ge::ConstGeTensorPtr>()->GetTensorDesc(), tensor.get()->GetTensorDesc());
}

void ExpectPhonyAllAttrs(const IrOpProto &proto) {
  ExpectPhonyAllRequiredAttrs(proto);
  ExpectPhonyAllScalarOptionalAttrs(proto);
  ExpectPhonyAllListOptionalAttrs(proto);
  ExpectPhonyAllTensorAttr(proto);
}

template <typename F>
void ExpectRuntimeErrorWithMessage(F &&fn, const std::string &expected) {
  try {
    fn();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_EQ(std::string(e.what()), expected);
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}
}  // namespace

REG_OP(phony_all)
  .INPUT(x, TensorType::ALL())
  .OPTIONAL_INPUT(xo, TensorType::ALL())
  .DYNAMIC_INPUT(dx, TensorType::ALL())
  .GRAPH(g)
  .DYNAMIC_GRAPH(dg)
  .REQUIRED_ATTR(ri, Int)
  .ATTR(i, Int, 7)
  .ATTR(f, Float, 1.0)
  .ATTR(s, String, "a")
  .ATTR(b, Bool, true)
  .ATTR(dt, Type, DT_INT64)
  .ATTR(li, ListInt, {1, -2})
  .ATTR(lf, ListFloat, {1.0, 1e-08f})
  .ATTR(lb, ListBool, {true, false})
  .ATTR(ldt, ListType, {DT_FLOAT, DT_DOUBLE})
  .ATTR(lli, ListListInt, {{}, {2, 3}, {}})
  .ATTR(t, Tensor, Tensor())
  .ATTR(ls, ListString, {"a", "b"})
  .REQUIRED_ATTR(rlb, ListBool)
  .OUTPUT(y, TensorType::NumberType())
  .DYNAMIC_OUTPUT(dy, TensorType::All())
  .OP_END_FACTORY_REG(phony_all);

class IrProtoCodecUT : public ::testing::Test {};

TEST(IrProtoCodecUT, ExtractIrProtoFromOpDesc) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op::phony_all("phony_all"));
  const auto proto = IrProtoCodec::FromOpDesc(op_desc);
  EXPECT_EQ(proto.op_type, "phony_all");
  ExpectPhonyAllInputs(proto);
  ExpectPhonyAllOutputs(proto);
  ExpectPhonyAllAttrs(proto);
  ExpectPhonyAllSubgraphs(proto);
}

TEST(IrProtoCodecUT, ExtractIrProtoFromJson) {
  const nlohmann::json op_json = PhonyAllOpJson();
  const auto proto = IrProtoCodec::FromJson(op_json);
  EXPECT_EQ(proto.op_type, "phony_all");
  ExpectPhonyAllInputs(proto);
  ExpectPhonyAllOutputs(proto);
  ExpectPhonyAllAttrs(proto);
  ExpectPhonyAllSubgraphs(proto);
}

TEST(IrProtoCodecUT, RoundTripJson) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op::phony_all("phony_all"));
  const auto proto = IrProtoCodec::FromOpDesc(op_desc);
  const auto json_obj = IrProtoCodec::ToJson(proto);

  const std::string file_path = "./operators_" + std::to_string(getpid()) + ".json";
  {
    std::ofstream out(file_path, std::ios::out | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out << json_obj.dump(2);
  }
  std::ifstream in(file_path);
  ASSERT_TRUE(in.is_open());
  const auto read_json = nlohmann::json::parse(in);
  const auto parsed = IrProtoCodec::FromJson(read_json);

  EXPECT_EQ(parsed.op_type, proto.op_type);
  EXPECT_EQ(parsed.inputs.size(), proto.inputs.size());
  EXPECT_EQ(parsed.outputs.size(), proto.outputs.size());
  EXPECT_EQ(parsed.attrs.size(), proto.attrs.size());
  EXPECT_EQ(parsed.subgraphs.size(), proto.subgraphs.size());

  for (size_t i = 0U; i < proto.inputs.size(); ++i) {
    EXPECT_EQ(parsed.inputs[i].name, proto.inputs[i].name);
    EXPECT_EQ(parsed.inputs[i].type, proto.inputs[i].type);
    EXPECT_EQ(parsed.inputs[i].dtype, proto.inputs[i].dtype);
  }
  for (size_t i = 0U; i < proto.outputs.size(); ++i) {
    EXPECT_EQ(parsed.outputs[i].name, proto.outputs[i].name);
    EXPECT_EQ(parsed.outputs[i].type, proto.outputs[i].type);
    EXPECT_EQ(parsed.outputs[i].dtype, proto.outputs[i].dtype);
  }
  for (size_t i = 0U; i < proto.attrs.size(); ++i) {
    EXPECT_EQ(parsed.attrs[i].name, proto.attrs[i].name);
    EXPECT_EQ(parsed.attrs[i].av_type, proto.attrs[i].av_type);
    EXPECT_EQ(parsed.attrs[i].required, proto.attrs[i].required);
    EXPECT_EQ(parsed.attrs[i].default_value, proto.attrs[i].default_value);
  }
  for (size_t i = 0U; i < proto.subgraphs.size(); ++i) {
    EXPECT_EQ(parsed.subgraphs[i].name, proto.subgraphs[i].name);
    EXPECT_EQ(parsed.subgraphs[i].type, proto.subgraphs[i].type);
  }
  std::remove(file_path.c_str());
}

TEST(IrProtoCodecUT, FromJsonThrowsOnMissingOpType) {
  auto op_json = PhonyAllOpJson();
  op_json.erase("op_type");
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); }, "op_type is required and must be a string");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnEmptyOpType) {
  auto op_json = PhonyAllOpJson();
  op_json["op_type"] = "";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); }, "op_type cannot be empty");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnInputsNotArray) {
  auto op_json = PhonyAllOpJson();
  op_json["inputs"] = "x";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); }, "inputs field is not an array");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnInputMissingName) {
  auto op_json = PhonyAllOpJson();
  op_json["inputs"] = nlohmann::json::array({{{"type", "INPUT"}}});
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); },
                                "inputs[0].name is required and must be a string");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnInvalidInputType) {
  auto op_json = PhonyAllOpJson();
  op_json["inputs"][0]["type"] = "BAD_INPUT";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); },
                                "inputs[0].type invalid: BAD_INPUT");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnOutputsNotArray) {
  auto op_json = PhonyAllOpJson();
  op_json["outputs"] = "y";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); }, "outputs field is not an array");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnSubgraphsNotArray) {
  auto op_json = PhonyAllOpJson();
  op_json["subgraphs"] = "g";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); }, "subgraphs field is not an array");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnAttrsNotArray) {
  auto op_json = PhonyAllOpJson();
  op_json["attrs"] = "attrs";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); }, "attrs field is not an array");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnAttrRequiredNotBool) {
  auto op_json = PhonyAllOpJson();
  op_json["attrs"][0]["required"] = "true";
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); },
                                "attrs[0].required is required and must be a boolean");
}

TEST(IrProtoCodecUT, FromJsonThrowsOnAttrDefaultValueNotString) {
  auto op_json = PhonyAllOpJson();
  op_json["attrs"][2]["default_value"] = 7;
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::FromJson(op_json); },
                                "attrs[2].default_value is not a string");
}

TEST(IrProtoCodecUT, ToJsonThrowsOnInvalidIrInputType) {
  IrOpProto proto;
  proto.op_type = "BadOp";
  proto.inputs = {{"x", static_cast<IrInputType>(5), {}}};
  ExpectRuntimeErrorWithMessage([&]() { IrProtoCodec::ToJson(proto); }, "invalid input ir type: 5");
}
