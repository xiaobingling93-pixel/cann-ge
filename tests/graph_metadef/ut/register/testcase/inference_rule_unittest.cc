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
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>
#include <string>
#include <unordered_set>
#include <cctype>
#include <regex>
#include <stack>
#include <fcntl.h>

#include "op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_reg.h"
#include "framework/common/debug/ge_log.h"
#include "register/shape_inference.h"
#include "registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/tensor.h"
#include "faker/space_registry_faker.h"

using Json = nlohmann::json;
using namespace gert;

namespace ge {
REG_OP(RuleInferOp)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(RuleInferOp);
}  // namespace ge

namespace {
class CtxMaker {
 public:
  CtxMaker() {
    json["shape"]["inputs"] = Json::array();
    json["shape"]["outputs"] = Json::array();
    json["dtype"] = Json::array();
  }

  CtxMaker &Input(const Json::array_t &input, const std::initializer_list<int64_t> runtime_input) {
    json["shape"]["inputs"].push_back(input);
    compile_inputs.emplace_back(NewShape());
    runtime_inputs.emplace_back(NewShape(runtime_input));
    auto &compile_input = compile_inputs.back()->MutableOriginShape();
    compile_input.SetDimNum(runtime_input.size());
    for (size_t i = 0; i < runtime_input.size(); ++i) {
      const auto &dim = input[i];
      if (dim.is_string()) {
        compile_input.SetDim(i, -1);
      } else if (dim.is_number_integer()) {
        const int64_t dim_value = dim.get<int64_t>();
        compile_input.SetDim(i, dim_value);
      } else {
        compile_input.SetDim(i, -3);
      }
    }
    return *this;
  }

  CtxMaker &ValueInput(const Json::array_t &input, const std::initializer_list<int64_t> runtime_input,
                       ge::DataType dtype) {
    json["shape"]["inputs"].push_back(input);
    compile_inputs.emplace_back(NewTensor(runtime_input, dtype));
    runtime_inputs.emplace_back(NewTensor(runtime_input, dtype));
    return *this;
  }

  CtxMaker &NullInput() {
    json["shape"]["inputs"].push_back(nullptr);
    compile_inputs.emplace_back(nullptr);
    runtime_inputs.emplace_back(nullptr);
    return *this;
  }

  CtxMaker &Output(const Json::array_t &output) {
    json["shape"]["outputs"].push_back(output);
    compile_outputs.emplace_back(NewShape());
    runtime_outputs.emplace_back(NewShape());
    return *this;
  }

  CtxMaker &Dtypes(const Json::array_t &dtypes) {
    json["dtype"] = dtypes;
    output_dtypes.resize(dtypes.size(), ge::DataType::DT_UNDEFINED);
    for (auto &output_dtype : output_dtypes) {
      ctx_dtypes.emplace_back(&output_dtype);
    }
    return *this;
  }

  std::string Str() const {
    return json.dump();
  }

  void Build(bool with_rule = true) {
    const auto rule_op = std::make_shared<ge::op::RuleInferOp>("op");
    rule_op->create_dynamic_input_x(compile_inputs.size());
    rule_op->create_dynamic_output_y(compile_outputs.size());
    for (size_t i = 0; i < compile_inputs.size(); ++i) {
      if (compile_inputs[i] == nullptr) {
        rule_op->UpdateDynamicInputDesc("x", i, ge::TensorDesc());
        continue;
      }
      auto &storage_shape = compile_inputs[i]->MutableOriginShape();
      std::vector<int64_t> dims;
      dims.reserve(storage_shape.GetDimNum());
      for (size_t j = 0; j < storage_shape.GetDimNum(); ++j) {
        dims.push_back(storage_shape.GetDim(j));
      }
      rule_op->UpdateDynamicInputDesc("x", i, ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, ge::DT_FLOAT16));
    }
    desc = ge::OpDescUtils::GetOpDescFromOperator(*rule_op);
    if (with_rule) {
      ge::AttrUtils::SetStr(desc, "_inference_rule", Str());
    }
    op = rule_op;

    std::vector<void *> inputs;
    std::vector<void *> outputs;
    inputs.reserve(compile_inputs.size());
    for (auto &input : compile_inputs) {
      inputs.emplace_back(input);
    }
    outputs.reserve(compile_outputs.size());
    for (auto &output : compile_outputs) {
      outputs.emplace_back(output);
    }

    std::vector<void *> rt_inputs;
    std::vector<void *> rt_outputs;
    rt_inputs.reserve(runtime_inputs.size());
    for (auto &input : runtime_inputs) {
      rt_inputs.emplace_back(input);
    }
    rt_outputs.reserve(runtime_outputs.size());
    for (auto &output : runtime_outputs) {
      rt_outputs.emplace_back(output);
    }
  }

  ge::OpDescPtr OpDesc() const {
    return desc;
  }

  ge::Operator &Operator() const {
    return *op;
  }

  StorageShape *NewShape() {
    holders.emplace_back(std::make_shared<StorageShape>());
    return holders.back().get();
  }

  StorageShape *NewTensor(const std::initializer_list<int64_t> &runtime_input, ge::DataType dtype) {
    values.emplace_back(std::shared_ptr<void>(malloc(sizeof(int64_t) * runtime_input.size()), std::free));
    auto shape = StorageShape({static_cast<long>(runtime_input.size())}, {static_cast<long>(runtime_input.size())});
    tensor_holders.emplace_back(std::make_shared<gert::Tensor>(shape, StorageFormat(), kOnHost, dtype, values.back().get()));
    if (dtype == ge::DT_INT32) {
      const auto data = tensor_holders.back()->GetData<int32_t>();
      size_t i = 0;
      for (const auto dim : runtime_input) {
        data[i++] = static_cast<int32_t>(dim);
      }
    } else if (dtype == ge::DT_INT64) {
      const auto data = tensor_holders.back()->GetData<int64_t>();
      size_t i = 0;
      for (const auto dim : runtime_input) {
        data[i++] = dim;
      }
    } else if (dtype == ge::DT_UINT32) {
      const auto data = tensor_holders.back()->GetData<uint32_t>();
      size_t i = 0;
      for (const auto dim : runtime_input) {
        data[i++] = static_cast<uint32_t>(dim);
      }
    }
    return reinterpret_cast<StorageShape *>(tensor_holders.back().get());
  }

  StorageShape *NewShape(const std::initializer_list<int64_t> &runtime_input) {
    holders.emplace_back(std::make_shared<StorageShape>(runtime_input, runtime_input));
    return holders.back().get();
  }

  Json json;
  std::vector<StorageShape *> compile_inputs;
  std::vector<StorageShape *> runtime_inputs;
  std::vector<StorageShape *> compile_outputs;
  std::vector<StorageShape *> runtime_outputs;

  std::vector<std::shared_ptr<StorageShape>> holders;

  std::vector<std::shared_ptr<void>> values;
  std::vector<std::shared_ptr<gert::Tensor>> tensor_holders;

  std::vector<void *> ctx_dtypes;
  std::vector<ge::DataType> output_dtypes;

  std::shared_ptr<ge::Operator> op = nullptr;
  ge::OpDescPtr desc = nullptr;
};
}  // namespace

class InferenceRuleUtest : public testing::Test {
 protected:
  void SetUp() override {
    // construct op impl registry
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("RuleInferOp");
  }

  void TearDown() override {}

  static std::string ShapeEqual(Shape *shape, std::initializer_list<int64_t> dims) {
    std::stringstream ss;
    if (shape == nullptr) {
      return "shape == nullptr";
    }
    if (shape->GetDimNum() != dims.size()) {
      ss << "dim num not equal, expect " << dims.size() << ", got " << shape->GetDimNum();
      return ss.str();
    }
    for (size_t i = 0; i < dims.size(); ++i) {
      if (shape->GetDim(i) != *(dims.begin() + i)) {
        ss << "dim[" << i << "] not equal, expect " << *(dims.begin() + i) << ", got " << shape->GetDim(i);
        return ss.str();
      }
    }
    return "";
  }

  static std::string ShapeEqual(const ge::GeShape &shape, std::initializer_list<int64_t> dims) {
    std::stringstream ss;
    if (shape.GetDimNum() != dims.size()) {
      ss << "dim num not equal, expect " << dims.size() << ", got " << shape.GetDimNum();
      return ss.str();
    }
    for (size_t i = 0; i < dims.size(); ++i) {
      if (shape.GetDim(i) != *(dims.begin() + i)) {
        ss << "dim[" << i << "] not equal, expect " << *(dims.begin() + i) << ", got " << shape.GetDim(i);
        return ss.str();
      }
    }
    return "";
  }
};

TEST_F(InferenceRuleUtest, CalledByInferShapeOnCompile) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();

  const auto desc = ctx_maker.OpDesc();
  ASSERT_EQ(InferShapeOnCompile(ctx_maker.Operator(), desc), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(desc->GetOutputDesc(0).GetShape(), {-1}), "");
}

TEST_F(InferenceRuleUtest, CalledByInferShapeOnCompileNoRule) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build(false);

  const auto desc = ctx_maker.OpDesc();
  ASSERT_NE(InferShapeOnCompile(ctx_maker.Operator(), desc), ge::GRAPH_SUCCESS);
}

TEST_F(InferenceRuleUtest, CalledByInferShapeOnCompileInvalidRule) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0+s4"}).Build();

  const auto desc = ctx_maker.OpDesc();
  ASSERT_NE(InferShapeOnCompile(ctx_maker.Operator(), desc), ge::GRAPH_SUCCESS);
}

TEST_F(InferenceRuleUtest, CalledByInferDtypeOnCompile) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Dtypes({ge::DT_FLOAT16}).Build();

  const auto desc = ctx_maker.OpDesc();
  ASSERT_EQ(InferDataTypeOnCompile(desc), ge::GRAPH_SUCCESS);
  ASSERT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
}

TEST_F(InferenceRuleUtest, CalledByInferDtypeOnCompileNoRule) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Dtypes({ge::DT_FLOAT16}).Build(false);

  const auto desc = ctx_maker.OpDesc();
  ASSERT_NE(InferDataTypeOnCompile(desc), ge::GRAPH_SUCCESS);
}

TEST_F(InferenceRuleUtest, CalledByInferDtypeOnCompileInvalidRule) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Dtypes({ge::DT_UNDEFINED}).Build();

  const auto desc = ctx_maker.OpDesc();
  ASSERT_NE(InferDataTypeOnCompile(desc), ge::GRAPH_SUCCESS);
}

TEST_F(InferenceRuleUtest, NeverFailedForInferShapeRange) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();

  const auto desc = ctx_maker.OpDesc();
  const auto tensor_desc = desc->MutableInputDesc(0);
  ASSERT_TRUE(tensor_desc->GetShape().IsUnknownShape());

  std::vector<std::vector<int64_t>> shape_range;
  shape_range.push_back({1, 32});
  ge::AttrUtils::SetListListInt(tensor_desc, "shape_range", shape_range);

  std::vector<std::pair<int64_t, int64_t>> got_shape_range;
  tensor_desc->GetShapeRange(got_shape_range);
  ASSERT_TRUE(!got_shape_range.empty());

  ASSERT_EQ(InferShapeRangeOnCompile(ctx_maker.Operator(), desc), ge::GRAPH_SUCCESS);
}