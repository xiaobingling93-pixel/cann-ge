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
#include "utils/inference_rule.h"
#include "graph_metadef/depends/faker/kernel_run_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
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
  CtxMaker() : compile_holder(), runtime_holder(), dtypes_holder() {
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

    compile_holder = InferShapeContextFaker()
                         .IrInputNum(inputs.size())
                         .NodeIoNum(inputs.size(), outputs.size())
                         .InputShapes(inputs)
                         .OutputShapes(outputs)
                         .Build();

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

    runtime_holder = InferShapeContextFaker()
                         .IrInputNum(rt_inputs.size())
                         .NodeIoNum(rt_inputs.size(), rt_outputs.size())
                         .InputShapes(rt_inputs)
                         .OutputShapes(rt_outputs)
                         .Build();

    dtypes_holder = InferDataTypeContextFaker()
                        .IrInputNum(rt_inputs.size())
                        .NodeIoNum(rt_inputs.size(), rt_outputs.size())
                        .OutputDataTypes(ctx_dtypes)
                        .Build();
  }

  InferShapeContext *CompileCtx() {
    return compile_holder.GetContext<InferShapeContext>();
  }

  InferShapeContext *RuntimeCtx() {
    return runtime_holder.GetContext<InferShapeContext>();
  }

  InferDataTypeContext *DtypeCtx() {
    return dtypes_holder.GetContext<InferDataTypeContext>();
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
    tensor_holders.emplace_back(std::make_shared<Tensor>(shape, StorageFormat(), kOnHost, dtype, values.back().get()));
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
  FakeKernelContextHolder compile_holder;
  FakeKernelContextHolder runtime_holder;
  FakeKernelContextHolder dtypes_holder;

  std::vector<std::shared_ptr<void>> values;
  std::vector<std::shared_ptr<Tensor>> tensor_holders;

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

TEST_F(InferenceRuleUtest, BasicDimSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32}), "");
}

TEST_F(InferenceRuleUtest, MultiDimSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0", "s1"}, {32, 64}).Output({"s1", "s0"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1, -1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {64, 32}), "");
}

TEST_F(InferenceRuleUtest, DimSymbolWithFunctionVertical) {
  CtxMaker ctx_maker;
  int64_t s0 = 32;
  int64_t s1 = 64;
  // "+", "-", "*", "Div", "Floor", "Ceil", "Pow", "Mod"
  ctx_maker.Input({"s0", "s1"}, {s0, s1})
      .Output({"s1+s0"})
      .Output({"s1-s0"})
      .Output({"s1*s0"})
      .Output({"Div(s1,s0)"})
      .Output({"Floor(Div(s1,3))"})
      .Output({"Ceil(Div(s1,3))"})
      .Output({"Pow(s0,2)"})
      .Output({"Mod(s1,7)"})
      .Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(1), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(2), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(3), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(4), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(5), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(6), {-1}), "");
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(7), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {s1 + s0}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(1), {s1 - s0}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(2), {s1 * s0}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(3), {s1 / s0}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(4), {s1 / 3}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(5), {(s1 + 2) / 3}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(6), {s0 * s0}), "");
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(7), {s1 % 7}), "");
}

TEST_F(InferenceRuleUtest, DimSymbolWithFunctionHorizontal) {
  CtxMaker ctx_maker;
  int64_t s0 = 32;
  int64_t s1 = 64;
  // "+", "-", "*", "Div", "Floor", "Ceil", "Pow", "Mod"
  ctx_maker.Input({"s0", "s1"}, {s0, s1})
      .Output(
          {"s1+s0", "s1-s0", "s1*s0", "Div(s1,s0)", "Floor(Div(s1,3))", "Ceil(Div(s1,3))", "Pow(s0,2)", "Mod(s1,7)"})
      .Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1, -1, -1, -1, -1, -1, -1, -1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0),
                       {s1 + s0, s1 - s0, s1 * s0, s1 / s0, s1 / 3, (s1 + 2) / 3, s0 * s0, s1 % 7}),
            "");
}

TEST_F(InferenceRuleUtest, StaticDimSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Output({"128", "32+24"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {128, 56}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {128, 56}), "");
}

TEST_F(InferenceRuleUtest, NullDimSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0", nullptr, "s1"}, {32, 20, 24}).Output({"s0+s1"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {56}), "");
}

TEST_F(InferenceRuleUtest, RepeatDimSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0", "s0"}, {32, 32}).Input({"s1"}, {24}).Input({"s1"}, {24}).Output({"s0+s1"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {56}), "");
}

TEST_F(InferenceRuleUtest, SymbolMixStrAndIntAndNull) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0", 128, "s1", nullptr, "s3", "24"}, {4, 128, 8, 0, 16, 24})
      .Output({"s1", "128", 32, "128+32"})
      .Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1, 128, 32, 160}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {8, 128, 32, 160}), "");
}

TEST_F(InferenceRuleUtest, SymbolWithNullInput) {
  CtxMaker ctx_maker;
  ctx_maker.NullInput().Input({"s0"}, {32}).Output({"s0"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32}), "");
}

TEST_F(InferenceRuleUtest, ValueSymbolBasic) {
  CtxMaker ctx_maker;
  ctx_maker.ValueInput({"v0"}, {32}, ge::DT_INT32).Output({"v0+3"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {35}), "");
}

TEST_F(InferenceRuleUtest, ValueSymbolMultiDtype) {
  CtxMaker ctx_maker;
  ctx_maker.ValueInput({"v0"}, {32}, ge::DT_INT32)
      .ValueInput({"v1"}, {24}, ge::DT_UINT32)
      .ValueInput({"v2"}, {8}, ge::DT_INT64)
      .Output({"v0+v1+v2"})
      .Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32 + 24 + 8}), "");
}

TEST_F(InferenceRuleUtest, MultiValueSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.ValueInput({"v0", "v2", "v1"}, {32, 2, 6}, ge::DT_INT32).Output({"v0+v1+v2"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32 + 2 + 6}), "");
}

TEST_F(InferenceRuleUtest, ValueSymbolMixNull) {
  CtxMaker ctx_maker;
  ctx_maker.ValueInput({"v0", nullptr, "v1"}, {32, 2, 6}, ge::DT_INT32).Output({"v0+v1"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32 + 6}), "");
}

TEST_F(InferenceRuleUtest, ValueSymbolMixDimSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0", "s1"}, {3, 4})
      .ValueInput({"v0", nullptr, "v1"}, {32, 2, 6}, ge::DT_INT32)
      .Output({"v0+s0", "v1+s1"})
      .Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1, -1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32 + 3, 6 + 4}), "");
}

TEST_F(InferenceRuleUtest, CompileAndLoadSucceed) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();

  std::vector<uint8_t> binary;
  ASSERT_EQ(ge::ShapeInferenceRule::CompileJsonString(ctx_maker.Str(), binary), ge::GRAPH_SUCCESS);
  const auto handle = ge::ShapeInferenceRule::FromCompiledBinary(binary);
  ASSERT_EQ(handle.Error(), "");

  const auto compile_ctx = ctx_maker.CompileCtx();
  ASSERT_EQ(handle.InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(compile_ctx->GetOutputShape(0), {-1}), "");

  const auto runtime_ctx = ctx_maker.RuntimeCtx();
  ASSERT_EQ(handle.InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ShapeEqual(runtime_ctx->GetOutputShape(0), {32}), "");
}

TEST_F(InferenceRuleUtest, OutputWithUndefinedSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s1"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "Error parsing output tensors: Symbol 's1' used in output but not defined in inputs");
}

TEST_F(InferenceRuleUtest, InputIsNotRawSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"t0"}, {32}).Output({"t1"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Error parsing input symbols: Invalid input[0].size(0): t0, symbol dimension must start with 's' or 'v' "
            "and follow with a number");
}

TEST_F(InferenceRuleUtest, InputIsNotSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0+2"}, {32}).Output({"s0+2"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Error parsing input symbols: Invalid input[0].size(0): s0+2, symbol dimension must start with 's' or 'v' "
            "and follow with a number");
}

TEST_F(InferenceRuleUtest, NoShapeFiled) {
  const auto handle = ge::ShapeInferenceRule::FromJsonString("{}");
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "Missing 'shape' field in rule json.");
}

TEST_F(InferenceRuleUtest, InputsFormatError) {
  {
    Json json;
    json["shape"]["inputs"] = 3;
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Invalid 'shape.inputs' field: 3 field must be an array or null.");
  }

  {
    Json json;
    json["shape"]["inputs"] = {3};
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Invalid 'shape.inputs' field: [3] element must be an array of dimension expressions.");
  }

  {
    Json json;
    json["shape"]["inputs"] = {{2.5}};
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(),
              "Invalid 'shape.inputs' field: [[2.5]] dimension expression must be a string or integer.");
  }
}

TEST_F(InferenceRuleUtest, OutputsFormatError) {
  {
    Json json;
    json["shape"]["outputs"] = 3;
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Invalid 'shape.outputs' field: 3 field must be an array or null.");
  }

  {
    Json json;
    json["shape"]["outputs"] = {3};
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Invalid 'shape.outputs' field: [3] element must be an array of dimension expressions.");
  }

  {
    Json json;
    json["shape"]["outputs"] = {{2.5}};
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(),
              "Invalid 'shape.outputs' field: [[2.5]] dimension expression must be a string or integer.");
  }

  {
    Json json;
    json["shape"]["outputs"] = {{nullptr}};
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Error parsing output tensors: Invalid output[0].size(0): empty dimension");
  }

  {
    Json json;
    json["shape"]["outputs"] = {{""}};
    const auto handle = ge::ShapeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Error parsing output tensors: Invalid output[0].size(0): empty dimension");
  }
}

TEST_F(InferenceRuleUtest, UnsupportedFunction) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"Abc(s0)"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Error parsing output tensors: Invalid dim expr 'Abc(s0)': Invalid function: Abc, supported [Div, Floor, "
            "Ceil, Pow, Mod]");
}

TEST_F(InferenceRuleUtest, UnsupportedOperator) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0 / 3"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Error parsing output tensors: Invalid dim expr 's0 / 3': Expression contains invalid characters");
}

TEST_F(InferenceRuleUtest, IllegalExpression_UnmatchedRightParenthesis) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0)"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "Error parsing output tensors: Invalid dim expr 's0)': Unmatched ')'");
}

TEST_F(InferenceRuleUtest, IllegalExpression_UnmatchedLeftParenthesis) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"(s0"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "Error parsing output tensors: Invalid dim expr '(s0': Unmatched '('");
}

TEST_F(InferenceRuleUtest, IllegalExpression_InvalidSymbol) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"2s0)"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Error parsing output tensors: Invalid dim expr '2s0)': Invalid identifier: '2s0', expected start with 's' "
            "or 'v' and follow with a number");
}

TEST_F(InferenceRuleUtest, IllegalExpression_SyntaxError) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0 ++ 2"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Failed to compile C++ code to shared object:\nextern \"C\" {bool infer_shape(Ctx *ctx) {\n    "
            "GET_SYMBOL_DIM(s0, 0, 0);\n\n    SET_OUTPUT_RANK(0, 1);\n    SET_OUTPUT_DIM(0, 0, static_cast<int64_t>(s0 "
            "++ 2));\n\n    return true;\n}\nbool infer_shape_on_compile(Ctx *ctx) {\n    SET_OUTPUT_RANK(0, 1);\n    "
            "SET_OUTPUT_DIM(0, 0, -1);\n\n    return true;\n}}\nError: syntax error");
}

TEST_F(InferenceRuleUtest, BasicDtypeInfer) {
  CtxMaker ctx_maker;
  ctx_maker.Output({128}).Dtypes({ge::DataType::DT_BF16}).Build();

  const auto handle = ge::DtypeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  const auto dtype_ctx = ctx_maker.DtypeCtx();

  ASSERT_EQ(handle->InferDtype(dtype_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(dtype_ctx->GetOutputDataType(0), ge::DataType::DT_BF16);
}

TEST_F(InferenceRuleUtest, InvalidDtype1) {
  CtxMaker ctx_maker;
  ctx_maker.Output({128}).Dtypes({ge::DataType::DT_UNDEFINED}).Build();

  const auto handle = ge::DtypeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Element 28 in 'dtype' field is out of range [0,43(DT_MAX)) and cannot be 28(DT_UNDEFINED).");
}

TEST_F(InferenceRuleUtest, InvalidDtype2) {
  CtxMaker ctx_maker;
  ctx_maker.Output({128}).Dtypes({ge::DataType::DT_MAX}).Build();

  const auto handle = ge::DtypeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Element 43 in 'dtype' field is out of range [0,43(DT_MAX)) and cannot be 28(DT_UNDEFINED).");
}

TEST_F(InferenceRuleUtest, InvalidDtype3) {
  CtxMaker ctx_maker;
  ctx_maker.Output({128}).Dtypes({-1}).Build();

  const auto handle = ge::DtypeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Element -1 in 'dtype' field is out of range [0,43(DT_MAX)) and cannot be 28(DT_UNDEFINED).");
}

TEST_F(InferenceRuleUtest, DtypesFormatError) {
  {
    Json json;
    const auto handle = ge::DtypeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Missing 'dtype' field in rule json.");
  }

  {
    Json json;
    json["dtype"] = 3;
    const auto handle = ge::DtypeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Field 'dtype' must be an array.");
  }

  {
    Json json;
    json["dtype"] = {nullptr};
    const auto handle = ge::DtypeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Element in 'dtype' field must not be null.");
  }

  {
    Json json;
    json["dtype"] = {2.5};
    const auto handle = ge::DtypeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Element in 'dtype' field must be an integer.");
  }

  {
    Json json;
    json["dtype"] = nullptr;
    const auto handle = ge::DtypeInferenceRule::FromJsonString(json.dump());
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(handle->Error(), "Filed 'dtype' must not be null.");
  }
}

TEST_F(InferenceRuleUtest, JsonFormatError) {
  Json json;
  const auto handle = ge::DtypeInferenceRule::FromJsonString("{");
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(),
            "Error parsing json: [json.exception.parse_error.101] parse error at line 1, column 2: syntax error while "
            "parsing object key - unexpected end of input; expected string literal");
}

TEST_F(InferenceRuleUtest, CalledByInvalidDimCtx) {
  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  {
    CtxMaker ctx_bug;
    ctx_bug.Build();

    const auto compile_ctx = ctx_bug.CompileCtx();
    ASSERT_NE(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);

    const auto runtime_ctx = ctx_bug.RuntimeCtx();
    ASSERT_NE(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  }

  {
    CtxMaker ctx_bug;
    ctx_bug.Input({"s0"}, {32}).Build();

    const auto compile_ctx = ctx_bug.CompileCtx();
    ASSERT_NE(handle->InferOnCompile(compile_ctx), ge::GRAPH_SUCCESS);

    const auto runtime_ctx = ctx_bug.RuntimeCtx();
    ASSERT_NE(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  }
}

TEST_F(InferenceRuleUtest, CalledByInvalidValueCtx) {
  CtxMaker ctx_maker;
  ctx_maker.ValueInput({"v0", "v1"}, {32, 24}, ge::DT_INT32).Output({"v1"}).Build();

  const auto handle = ge::ShapeInferenceRule::FromJsonString(ctx_maker.Str());
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(handle->Error(), "");

  {
    CtxMaker ctx_bug;
    ctx_bug.Build();
    const auto runtime_ctx = ctx_bug.RuntimeCtx();
    ASSERT_NE(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  }

  {
    CtxMaker ctx_bug;
    ctx_bug.ValueInput({"v0"}, {32}, ge::DT_INT32).Output({"v0"}).Build();
    const auto runtime_ctx = ctx_bug.RuntimeCtx();
    ASSERT_NE(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  }

  {
    CtxMaker ctx_bug;
    ctx_bug.ValueInput({"v0, v1"}, {32, 24}, ge::DT_INT16).Output({"v1"}).Build();
    const auto runtime_ctx = ctx_bug.RuntimeCtx();
    ASSERT_NE(handle->InferOnRuntime(runtime_ctx), ge::GRAPH_SUCCESS);
  }
}

TEST_F(InferenceRuleUtest, CompileInvalidJsonStrOrCode) {
  std::vector<uint8_t> binary;
  ASSERT_NE(ge::ShapeInferenceRule::CompileJsonString("{", binary), ge::GRAPH_SUCCESS);

  CtxMaker ctx_maker;
  ctx_maker.Input({"s0"}, {32}).Output({"s0 ++ 2"}).Build();
  ASSERT_NE(ge::ShapeInferenceRule::CompileJsonString(ctx_maker.Str(), binary), ge::GRAPH_SUCCESS);
}

TEST_F(InferenceRuleUtest, CallInvalidRule) {
  {
    const auto rule = ge::ShapeInferenceRule::FromJsonString("{");

    CtxMaker ctx_maker;
    ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();
    ASSERT_NE(rule->InferOnCompile(ctx_maker.CompileCtx()), ge::GRAPH_SUCCESS);
    ASSERT_NE(rule->InferOnRuntime(ctx_maker.RuntimeCtx()), ge::GRAPH_SUCCESS);
  }

  {
    const auto rule = ge::DtypeInferenceRule::FromJsonString("{");

    CtxMaker ctx_maker;
    ctx_maker.Input({"s0"}, {32}).Output({"s0"}).Build();
    ASSERT_NE(rule->InferDtype(ctx_maker.DtypeCtx()), ge::GRAPH_SUCCESS);
  }
}

TEST_F(InferenceRuleUtest, JustForCoverage) {
  auto handle = ge::ShapeInferenceRule::FromCompiledBinary({});
  ASSERT_NE(handle.Error(), "");

  ASSERT_TRUE(ge::ShapeInferenceRule::GetInferenceRule(nullptr).empty());
}
