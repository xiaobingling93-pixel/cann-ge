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
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include <utility>
#include <iostream>
#include <common/plugin/ge_make_unique_util.h>
#include <symengine/symengine_rcp.h>
#include "eager_style_graph_builder/compliant_op_desc_builder.h"
#include "graph/utils/graph_utils_ex.h"
#include "eager_style_graph_builder/esb_graph.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "faker/space_registry_faker.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "exe_graph/runtime/symbolic_shape.h"
#include "exe_graph/runtime/symbolic_tensor.h"
#include "register/op_impl_registry.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include <common_error_codes.h>

namespace ge {
namespace {
template <typename T>
std::vector<void *> GetVoidPtr(const std::vector<std::unique_ptr<T>> &outputs_holders) {
  std::vector<void *> outputs;
  outputs.reserve(outputs_holders.size());
  for (const auto &output_holder : outputs_holders) {
    outputs.push_back(output_holder.get());
  }
  return outputs;
}

std::pair<gert::OpImplKernelRegistry::InferSymbolShapeKernelFunc, uint64_t> GetInferFunc(const std::string &op_type) {
  const auto &functions = gert::OpImplInferSymbolShapeRegistry::GetInstance().GetOpImpl(op_type.c_str());
  return std::make_pair(functions->infer_symbol_shape, functions->inputs_dependency);
}

class InferSymbolShapeContextTestBuilder {
public:
  InferSymbolShapeContextTestBuilder() = default;

  InferSymbolShapeContextTestBuilder(std::string op_type, std::string op_name)
    : op_name_(std::move(op_name)),
      op_type_(std::move(op_type)) {
  }

  ~InferSymbolShapeContextTestBuilder() {
    Destroy();
  }

  // shape是指输入的symbol shape，symbol_value是指输入的symbolic value
  // 只有data dependent时传symbol_value才有意义
  InferSymbolShapeContextTestBuilder &AppendInputSymbolTensor(
      const gert::SymbolShape &shape, bool is_data_dependent = false,
      const std::vector<ge::Expression> *symbol_value = nullptr) {
    auto input_holder = ge::ComGraphMakeUnique<gert::SymbolTensor>();
    if (input_holder == nullptr) {
      GELOGE(GRAPH_FAILED, "Failed to malloc input holder.");
      return *this;
    }
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
      input_holder->MutableOriginSymbolShape().AppendDim(shape.GetDim(i));
    }
    if (is_data_dependent) {
      if (symbol_value == nullptr) {
        GELOGE(GRAPH_FAILED, "symbol_value is nullptr while is_data_dependent is true!.");
      } else {
        auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression>>(*symbol_value);
        if (symbolic_value_unique != nullptr) {
          input_holder->SetSymbolicValue(std::move(symbolic_value_unique));
        }
      }
    }
    input_holders_.emplace_back(std::move(input_holder));
    return *this;
  }

  InferSymbolShapeContextTestBuilder &AppendInputSymbolTensor(const gert::SymbolTensor &tensor) {
    return this->AppendInputSymbolTensor(tensor.GetOriginSymbolShape(), true, tensor.GetSymbolicValue());
  }

  InferSymbolShapeContextTestBuilder &OutputNum(size_t num) {
    output_holders_.reserve(num);
    for (size_t i = 0U; i < num; ++i) {
      auto output_holder = ge::ComGraphMakeUnique<gert::SymbolShape>();
      output_holders_.emplace_back(std::move(output_holder));
    }
    return *this;
  }

  OpDescPtr GetOrCreateOpDescPtr() {
    if (op_desc_ == nullptr) {
      op_desc_ = ComGraphMakeShared<OpDesc>(op_name_, op_type_);
    }
    return op_desc_;
  }

  gert::InferSymbolShapeContext *Build() {
    if (op_desc_ == nullptr) {
      op_desc_ = ComGraphMakeShared<OpDesc>(op_name_, op_type_);
    }
    kernel_context_holder_ = gert::KernelRunContextBuilder()
                             .Inputs(GetVoidPtr<gert::SymbolTensor>(input_holders_))
                             .Outputs(GetVoidPtr<gert::SymbolShape>(output_holders_))
                             .Build(op_desc_);
    auto infer_context = reinterpret_cast<gert::InferSymbolShapeContext *>(kernel_context_holder_.context_);
    return infer_context;
  }

  void Destroy() {
    input_holders_.clear();
    output_holders_.clear();
    op_desc_ = nullptr;
  }

private:
  std::string op_name_;
  std::string op_type_;
  OpDescPtr op_desc_;
  std::vector<std::unique_ptr<gert::SymbolTensor>> input_holders_;
  std::vector<std::unique_ptr<gert::SymbolShape>> output_holders_;
  gert::KernelContextHolder kernel_context_holder_;
};

#define EXPECT_RUN_CONCATV2D_TEST(input_shapes, attr_value, output_shapes, exp_status)                      \
  {                                                                                                      \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                        \
    auto op_descPtr = builder.GetOrCreateOpDescPtr();                                                    \
    op_descPtr->AppendIrAttrName(attr_name);                                                             \
    AttrUtils::SetInt(op_descPtr, attr_name, attr_value);                                                \
    op_descPtr->AppendIrAttrName("N");                                                                   \
    AttrUtils::SetInt(op_descPtr, "N", static_cast<int64_t>(input_shapes.size()));                       \
    op_descPtr->AddDynamicInputDesc("x", input_shapes.size(), true);                                     \
    for (auto &shape : input_shapes) {                                                                   \
      builder.AppendInputSymbolTensor(shape);                                                            \
    }                                                                                                    \
                                                                                                         \
    auto infer_context = builder.OutputNum(output_shapes.size()).Build();                                \
    auto func = GetInferFunc(infer_context->GetNodeType());                                              \
    ASSERT_TRUE(func.first != nullptr);                                                                  \
    ASSERT_EQ(func.first(infer_context), exp_status);                                                    \
    if (exp_status == ge::GRAPH_SUCCESS) {                                                               \
      for (size_t i = 0; i < output_shapes.size(); i++) {                                                \
        ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDimNum(), output_shapes.at(i).GetDimNum()); \
        for (size_t j = 0; j < infer_context->GetOutputSymbolShape(i)->GetDimNum(); j++) {               \
          ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDim(j), output_shapes.at(i).GetDim(j));   \
        }                                                                                                \
      }                                                                                                  \
    }                                                                                                    \
  }

void ConcatV2DTest(const std::string &op_type) {
  using namespace gert;
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto s3 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 4));

  auto k1 = Symbol(1);
  auto k3 = Symbol(3);

  auto op_name = "concat";
  auto attr_name = "concat_dim";
  vector<SymbolShape> input_shapes;
  vector<SymbolShape> output_shapes;
  // 场景1：多轴，多输入，只有拼接轴dim不同，拼接轴非首尾 且 正 -> 正常
  input_shapes = {SymbolShape({s0, s2, s1}), SymbolShape({s0, s3, s1}), SymbolShape({s0, s4, s1})};
  output_shapes = {SymbolShape({s0, s2 + s3 + s4, s1})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 1, output_shapes, GRAPH_SUCCESS);

  // 场景2：多轴，多输入，只有拼接轴dim不同，拼接轴为尾 且 负 -> 正常
  input_shapes = {SymbolShape({s0, s1, s2}), SymbolShape({s0, s1, s3}), SymbolShape({s0, s1, s4})};
  output_shapes = {SymbolShape({s0, s1, s2 + s3 + s4})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -1, output_shapes, GRAPH_SUCCESS);

  // 场景3：多轴，多输入，只有拼接轴dim不同，拼接轴为首 且 负 -> 正常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -3, output_shapes, GRAPH_SUCCESS);

  // 场景4：多轴，多输入，只有拼接轴dim不同，拼接轴为正 但 越界 -> 异常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 5, output_shapes, PARAM_INVALID);

  // 场景5：多轴，多输入，只有拼接轴dim不同，拼接轴为负 但 越界 -> 异常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -5, output_shapes, PARAM_INVALID);

  // 场景6：多轴，多输入，只有拼接轴dim不同，拼接轴为负 但 越界 -> 异常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -5, output_shapes, PARAM_INVALID);

  // 场景7：多轴，多输入，多个轴dim不同，拼接轴正常 -> 异常
  input_shapes = {SymbolShape({s2, s1, s2}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 0, output_shapes, PARAM_INVALID);

  // 场景8：多轴，单输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 0, output_shapes, GRAPH_SUCCESS);

  // 场景9：多轴，单输入，拼接轴异常 -> 正常
  input_shapes = {SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -10, output_shapes, GRAPH_SUCCESS);

  // 场景10：单轴，多输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape({s2}), SymbolShape({s3}), SymbolShape({s4})};
  output_shapes = {SymbolShape({s2 + s3 + s4})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 0, output_shapes, GRAPH_SUCCESS);

  // 场景11：单轴，多输入，拼接轴异常 -> 异常
  input_shapes = {SymbolShape({s2}), SymbolShape({s3}), SymbolShape({s4})};
  output_shapes = {SymbolShape({s2 + s3 + s4})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -2, output_shapes, PARAM_INVALID);

  // 场景12：标量，多输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape(), SymbolShape(), SymbolShape()};
  output_shapes = {SymbolShape({k3})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 0, output_shapes, GRAPH_SUCCESS);

  // 场景13：标量，多输入，拼接轴异常 -> 异常
  input_shapes = {SymbolShape(), SymbolShape(), SymbolShape()};
  output_shapes = {SymbolShape({k3})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -3, output_shapes, PARAM_INVALID);

  // 场景14：标量，单输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape()};
  output_shapes = {SymbolShape()};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 0, output_shapes, GRAPH_SUCCESS);

  // 场景15：标量，单输入，拼接轴异常 -> 正常
  input_shapes = {SymbolShape()};
  output_shapes = {SymbolShape()};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -3, output_shapes, GRAPH_SUCCESS);
}

#define RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, exp_status)                            \
  {                                                                                                      \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                        \
    auto op_descPtr = builder.GetOrCreateOpDescPtr();                                                    \
    op_descPtr->AppendIrAttrName("N");                                                                   \
    AttrUtils::SetInt(op_descPtr, "N", static_cast<int64_t>(input_shapes.size()));                       \
    if (op_type.find("V2") != std::string::npos) {                                                       \
      op_descPtr->AddDynamicInputDesc("x", input_shapes.size(), true);                                   \
      op_descPtr->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));                            \
      for (auto &shape : input_shapes) {                                                                 \
        builder.AppendInputSymbolTensor(shape);                                                          \
      }                                                                                                  \
      builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &dim_value);                            \
    } else {                                                                                             \
      op_descPtr->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));                            \
      op_descPtr->AddDynamicInputDesc("x", input_shapes.size(), true);                                   \
      builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &dim_value);                            \
      for (auto &shape : input_shapes) {                                                                 \
        builder.AppendInputSymbolTensor(shape);                                                          \
      }                                                                                                  \
    }                                                                                                    \
    auto infer_context = builder.OutputNum(output_shapes.size()).Build();                                \
    auto func = GetInferFunc(infer_context->GetNodeType());                                              \
    ASSERT_TRUE(func.first != nullptr);                                                                  \
    ASSERT_EQ(func.first(infer_context), exp_status);                                                    \
    if (exp_status == ge::GRAPH_SUCCESS) {                                                               \
      for (size_t i = 0; i < output_shapes.size(); i++) {                                                \
        ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDimNum(), output_shapes.at(i).GetDimNum()); \
        for (size_t j = 0; j < infer_context->GetOutputSymbolShape(i)->GetDimNum(); j++) {               \
          ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDim(j), output_shapes.at(i).GetDim(j));   \
        }                                                                                                \
      }                                                                                                  \
    }                                                                                                    \
  }

void ConcatV2Test(const std::string &op_type) {
  using namespace gert;
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  Symbol s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  Symbol s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  Symbol s3 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 3));
  Symbol s4 = shape_env.CreateSymbol(8, MakeShared<InputShapeSource>(0, 4));

  auto k1 = Symbol(1);
  auto k3 = Symbol(3);

  auto op_name = "concat";
  vector<SymbolShape> input_shapes;
  vector<SymbolShape> output_shapes;
  std::vector<ge::Expression> dim_value = {Symbol(1)};
  // 场景1：多轴，多输入，只有拼接轴dim不同，拼接轴非首尾 且 正 -> 正常
  input_shapes = {SymbolShape({s0, s2, s1}), SymbolShape({s0, s3, s1}), SymbolShape({s0, s4, s1})};
  output_shapes = {SymbolShape({s0, s2 + s3 + s4, s1})};
  dim_value = {Symbol(1)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景2：多轴，多输入，只有拼接轴dim不同，拼接轴为尾 且 负 -> 正常
  input_shapes = {SymbolShape({s0, s1, s2}), SymbolShape({s0, s1, s3}), SymbolShape({s0, s1, s4})};
  output_shapes = {SymbolShape({s0, s1, s2 + s3 + s4})};
  dim_value = {Symbol(-1)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景3：多轴，多输入，只有拼接轴dim不同，拼接轴为首 且 负 -> 正常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  dim_value = {Symbol(-3)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景4：多轴，多输入，只有拼接轴dim不同，拼接轴为正 但 越界 -> 异常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  dim_value = {Symbol(5)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  // 场景5：多轴，多输入，只有拼接轴dim不同，拼接轴为负 但 越界 -> 异常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  dim_value = {Symbol(-5)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  // 场景6：多轴，多输入，只有拼接轴dim不同，拼接轴为负 但 越界 -> 异常
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  // 场景7：多轴，多输入，多个轴dim不同，拼接轴正常 -> 异常
  input_shapes = {SymbolShape({s2, s1, s2}), SymbolShape({s3, s1, s0}), SymbolShape({s4, s1, s0})};
  output_shapes = {SymbolShape({s2 + s3 + s4, s1, s0})};
  dim_value = {Symbol(0)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  // 场景8：多轴，单输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  dim_value = {Symbol(0)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景9：多轴，单输入，拼接轴异常 -> 正常
  input_shapes = {SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  dim_value = {Symbol(-10)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景10：单轴，多输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape({s2}), SymbolShape({s3}), SymbolShape({s4})};
  output_shapes = {SymbolShape({s2 + s3 + s4})};
  dim_value = {Symbol(0)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景11：单轴，多输入，拼接轴异常 -> 异常
  input_shapes = {SymbolShape({s2}), SymbolShape({s3}), SymbolShape({s4})};
  output_shapes = {SymbolShape({s2 + s3 + s4})};
  dim_value = {Symbol(-2)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  // 场景12：标量，多输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape(), SymbolShape(), SymbolShape()};
  output_shapes = {SymbolShape({k3})};
  dim_value = {Symbol(0)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景13：标量，多输入，拼接轴异常 -> 异常
  input_shapes = {SymbolShape(), SymbolShape(), SymbolShape()};
  output_shapes = {SymbolShape({k3})};
  dim_value = {Symbol(-3)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  // 场景15：标量，单输入，拼接轴异常 -> 正常
  input_shapes = {SymbolShape()};
  output_shapes = {SymbolShape()};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景14：标量，单输入，拼接轴正常 -> 正常
  input_shapes = {SymbolShape()};
  output_shapes = {SymbolShape()};
  dim_value = {Symbol(0)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, GRAPH_SUCCESS);

  // 场景15：标量，单输入，拼接轴异常 -> 异常
  input_shapes = {SymbolShape()};
  output_shapes = {SymbolShape()};
  dim_value = {Symbol(1.1)};
  RUN_CONCATV2_TEST(input_shapes, dim_value, output_shapes, PARAM_INVALID);

  input_shapes = {SymbolShape()};
  output_shapes = {SymbolShape()};
  {
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);
    auto op_descPtr = builder.GetOrCreateOpDescPtr();
    op_descPtr->AppendIrAttrName("N");
    AttrUtils::SetInt(op_descPtr, "N", static_cast<int64_t>(input_shapes.size()));
    op_descPtr->AddDynamicInputDesc("x", input_shapes.size(), true);
    op_descPtr->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
    for (auto &shape : input_shapes) {
      builder.AppendInputSymbolTensor(shape);
    }
    builder.AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr);
    auto infer_context = builder.OutputNum(output_shapes.size()).Build();
    auto func = GetInferFunc(infer_context->GetNodeType());
    ASSERT_TRUE(func.first != nullptr);
    ASSERT_EQ(func.first(infer_context), UNSUPPORTED);
  }
}
} // namespace

class SymbolicShapeInferFuncUT : public testing::Test {
 public:
 protected:
  static void SetUpTestSuite() {
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  }
  static void TearDownTestSuite() {
  }
  void SetUp() override {}
  void TearDown() override {}
};
#define RUN_BROADCAST_TEST(input_shape0, input_shape1, expect_output_shape)                                       \
  {                                                                                                               \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                                 \
    auto infer_context =                                                                                          \
        builder.AppendInputSymbolTensor(input_shape0).AppendInputSymbolTensor(input_shape1).OutputNum(1).Build(); \
    ASSERT_TRUE(func.first != nullptr);                                                                           \
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);                                                      \
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_output_shape.GetDims());                  \
  }

#define RUN_ClipByValue_TEST(input_shape0, input_shape1, input_shape2, expect_output_shape, exp_status)           \
{                                                                                                                 \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                                 \
    auto infer_context =                                                                                          \
    builder.AppendInputSymbolTensor(input_shape0).AppendInputSymbolTensor(input_shape1)                           \
    .AppendInputSymbolTensor(input_shape2).OutputNum(1).Build();                                                  \
    ASSERT_TRUE(func.first != nullptr);                                                                           \
    ASSERT_EQ(func.first(infer_context), exp_status);                                                             \
    if(exp_status == ge::GRAPH_SUCCESS){                                                                          \
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_output_shape.GetDims());                  \
    }                                                                                                             \
}

#define EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, exp_status)             \
  {                                                                                                      \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                        \
    auto op_descPtr = builder.GetOrCreateOpDescPtr();                                                    \
    for (auto &shape : input_shapes) {                                                                   \
      op_descPtr->AddInputDesc(GeTensorDesc());                                                          \
      (void)shape;                                                                                       \
    }                                                                                                    \
    for (auto &tensor : data_depends) {                                                                  \
      op_descPtr->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));                            \
      (void)tensor;                                                                                      \
    }                                                                                                    \
                                                                                                         \
    for (auto &shape : input_shapes) {                                                                   \
      builder.AppendInputSymbolTensor(shape);                                                            \
    }                                                                                                    \
    for (auto &tensor : data_depends) {                                                                  \
      builder.AppendInputSymbolTensor(tensor);                                                           \
    }                                                                                                    \
                                                                                                         \
    auto infer_context = builder.OutputNum(output_shapes.size()).Build();                                \
    auto func = GetInferFunc(infer_context->GetNodeType());                                              \
    ASSERT_TRUE(func.first != nullptr);                                                                  \
    ASSERT_EQ(func.first(infer_context), exp_status);                                                    \
    if (exp_status == ge::GRAPH_SUCCESS) {                                                               \
      for (size_t i = 0; i < output_shapes.size(); i++) {                                                \
        ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDimNum(), output_shapes.at(i).GetDimNum()); \
        for (size_t j = 0; j < infer_context->GetOutputSymbolShape(i)->GetDimNum(); j++) {               \
          ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDim(j), output_shapes.at(i).GetDim(j));   \
        }                                                                                                \
      }                                                                                                  \
    }                                                                                                    \
  }

#define RUN_REDUCE_TEST(input_shape, symbol_axes, keep_dims, expect_output_shape, status)          \
  {                                                                                                \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                  \
    auto op_descPtr = builder.GetOrCreateOpDescPtr();                                              \
    op_descPtr->AppendIrAttrName("keep_dims");                                                     \
    AttrUtils::SetBool(op_descPtr, "keep_dims", keep_dims);                                        \
    op_descPtr->AddInputDesc(GeTensorDesc());                                                      \
    op_descPtr->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));                        \
    auto infer_context = builder.AppendInputSymbolTensor(input_shape)                              \
                             .AppendInputSymbolTensor(gert::SymbolShape({}), true, symbol_axes)    \
                             .OutputNum(1)                                                         \
                             .Build();                                                             \
    ASSERT_TRUE(func.first != nullptr);                                                            \
    ASSERT_EQ(func.first(infer_context), status);                                                  \
    if (status == ge::GRAPH_SUCCESS) {                                                             \
      ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_output_shape.GetDims()); \
    }                                                                                              \
  }

#define RUN_REDUCE_D_TEST(input_shape, axes, keep_dims, expect_output_shape, status)               \
  {                                                                                                \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                  \
    auto op_descPtr = builder.GetOrCreateOpDescPtr();                                              \
    op_descPtr->AppendIrAttrName("axes");                                                          \
    AttrUtils::SetListInt(op_descPtr, "axes", axes);                                               \
    op_descPtr->AppendIrAttrName("keep_dims");                                                     \
    AttrUtils::SetBool(op_descPtr, "keep_dims", keep_dims);                                        \
    auto infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();        \
    ASSERT_TRUE(func.first != nullptr);                                                            \
    ASSERT_EQ(func.first(infer_context), status);                                                  \
    if (status == ge::GRAPH_SUCCESS) {                                                             \
      ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_output_shape.GetDims()); \
    }                                                                                              \
  }

#define EXPECT_RUN_UNPACK_TEST(input_shapes, attr_value, output_shapes, exp_status)                      \
  {                                                                                                      \
    InferSymbolShapeContextTestBuilder builder(op_type, op_name);                                        \
    auto op_descPtr = builder.GetOrCreateOpDescPtr();                                                    \
    op_descPtr->AppendIrAttrName("num");                                                                 \
    AttrUtils::SetInt(op_descPtr, "num", static_cast<int64_t>(output_shapes.size()));                    \
    op_descPtr->AppendIrAttrName(attr_name);                                                             \
    AttrUtils::SetInt(op_descPtr, attr_name, attr_value);                                                \
    op_descPtr->AddDynamicInputDesc("x", input_shapes.size(), true);                                     \
    for (auto &shape : input_shapes) {                                                                   \
      builder.AppendInputSymbolTensor(shape);                                                            \
    }                                                                                                    \
    auto infer_context = builder.OutputNum(output_shapes.size()).Build();                                \
    auto func = GetInferFunc(infer_context->GetNodeType());                                              \
    ASSERT_TRUE(func.first != nullptr);                                                                  \
    ASSERT_EQ(func.first(infer_context), exp_status);                                                    \
    if (exp_status == ge::GRAPH_SUCCESS) {                                                               \
      for (size_t i = 0; i < output_shapes.size(); i++) {                                                \
        ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDimNum(), output_shapes.at(i).GetDimNum()); \
        for (size_t j = 0; j < infer_context->GetOutputSymbolShape(i)->GetDimNum(); j++) {               \
          ASSERT_EQ(infer_context->GetOutputSymbolShape(i)->GetDim(j), output_shapes.at(i).GetDim(j));   \
        }                                                                                                \
      }                                                                                                  \
    }                                                                                                    \
  }

static void TestForSoftmax(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);

  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto s3 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 4));
  auto input_shape0 = gert::SymbolShape({s0, s1, s2, s3});
  auto input_shape1 = gert::SymbolShape({s1, s2, s3, s4});
  InferSymbolShapeContextTestBuilder builder(op_type, op_name);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0)
                              .AppendInputSymbolTensor(input_shape1)
                              .OutputNum(2).Build();
  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), input_shape1.GetDims());

  const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 4);
  const std::set<std::string> expect_guard = {"ExpectLt(s3, s4)","ExpectLt(s2, s3)",
                                              "ExpectLt(s1, s2)","ExpectLt(s0, s1)"};
  for (auto &iter : guard_infos) {
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxCrossEntropyWithLogits) {
  TestForSoftmax("softmaxCrossEntropyWithLogits", "SoftmaxCrossEntropyWithLogits");
}

static void TestForSoftmaxbig(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto s3 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 4));
  auto input_shape0 = gert::SymbolShape({s1, s2, s3, s4});
  auto input_shape1 = gert::SymbolShape({s0, s1, s2, s3});
  InferSymbolShapeContextTestBuilder builder(op_type, op_name);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0)
                              .AppendInputSymbolTensor(input_shape1)
                              .OutputNum(2).Build();
  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), input_shape0.GetDims());

  const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 4);
  const std::set<std::string> expect_guard = {"ExpectLe(s3, s4)","ExpectLe(s2, s3)",
                                              "ExpectLe(s1, s2)","ExpectLe(s0, s1)"};
  for (auto &iter : guard_infos) {
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxCrossEntropyWithLogitsbig) {
  TestForSoftmaxbig("softmaxCrossEntropyWithLogits", "SoftmaxCrossEntropyWithLogits");
}

static void TestForSoftmax2(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto input_shape0 = gert::SymbolShape({s0, s1});
  auto input_shape1 = gert::SymbolShape({s1, s2});
  auto input_shape2 = gert::SymbolShape({s1});
  InferSymbolShapeContextTestBuilder builder(op_type, op_name);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0)
                              .AppendInputSymbolTensor(input_shape1)
                              .OutputNum(2).Build();
  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), input_shape1.GetDims());

  const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 2);
  EXPECT_EQ(std::string(guard_infos[0].expr.Serialize().get()), "ExpectLt(s0, s1)");
  EXPECT_EQ(std::string(guard_infos[1].expr.Serialize().get()), "ExpectLt(s1, s2)");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxCrossEntropyWithLogits2) {
  TestForSoftmax2("softmaxCrossEntropyWithLogits", "SoftmaxCrossEntropyWithLogits");
}

static void TestForSoftmax2big(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto input_shape0 = gert::SymbolShape({s1, s2});
  auto input_shape1 = gert::SymbolShape({s0, s1});
  auto input_shape2 = gert::SymbolShape({s1});
  InferSymbolShapeContextTestBuilder builder(op_type, op_name);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0)
                              .AppendInputSymbolTensor(input_shape1)
                              .OutputNum(2).Build();
  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), input_shape0.GetDims());

  const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 2);
  EXPECT_EQ(std::string(guard_infos[0].expr.Serialize().get()), "ExpectLe(s0, s1)");
  EXPECT_EQ(std::string(guard_infos[1].expr.Serialize().get()), "ExpectLe(s1, s2)");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxCrossEntropyWithLogits2big) {
  TestForSoftmax2big("softmaxCrossEntropyWithLogits", "SoftmaxCrossEntropyWithLogits");
}

static void TestForElementWise(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto input_shape0 = gert::SymbolShape({s0, s1});
  InferSymbolShapeContextTestBuilder builder(op_type, op_name);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0).OutputNum(1).Build();
  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForRelu) {
  TestForElementWise("relu1", "Relu");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForHcomAllReduce) {
  TestForElementWise("hcomAllReduce", "HcomAllReduce");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSigmoidGrad) {
  TestForElementWise("sigmoidGrad", "SigmoidGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForIsNan) {
  TestForElementWise("isnan", "IsNan");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSign) {
  TestForElementWise("sign", "Sign");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForExp) {
  TestForElementWise("exp", "Exp");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForZerosLike) {
  TestForElementWise("zerosLike", "ZerosLike");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSquare) {
  TestForElementWise("square", "Square");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForErf) {
  TestForElementWise("erf", "Erf");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReciprocal) {
  TestForElementWise("reciprocal", "Reciprocal");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLeakyRelu) {
  TestForElementWise("leakyRelu", "LeakyRelu");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLeakyElu) {
  TestForElementWise("Elu", "Elu");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForRsqrtGrad) {
  TestForElementWise("rsqrtgrad", "RsqrtGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForMuls) {
  TestForElementWise("muls", "Muls");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForTanh) {
  TestForElementWise("tanh", "Tanh");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForIsFinite) {
  TestForElementWise("isFinite", "IsFinite");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForStopGradient) {
  TestForElementWise("stopGradient", "StopGradient");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLog) {
  TestForElementWise("log", "Log");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLog1p) {
  TestForElementWise("log1p", "Log1p");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxGradExt) {
  TestForElementWise("softmaxGradExt", "SoftmaxGradExt");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForGelu) {
  TestForElementWise("gelu", "Gelu");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForFusedMulAddN) {
  TestForElementWise("fusedMulAddN", "FusedMulAddN");
}

static void TestForBroadCast(const std::string &op_name, const std::string &op_type) {
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 3));
    auto kOne = ge::Symbol(1);
    auto kTwo = ge::Symbol(2);

    auto func = GetInferFunc(op_type);
    ASSERT_TRUE(func.first != nullptr);
    // Test cases
    RUN_BROADCAST_TEST(gert::SymbolShape({s0, s0}), gert::SymbolShape({s0, s0}),

                       gert::SymbolShape({s0, s0}));  // {s0, s0} {s0, s0} -> {s0, s0}

    RUN_BROADCAST_TEST(gert::SymbolShape({kOne, s0}), gert::SymbolShape({s0, kOne}),
                       gert::SymbolShape({s0, s0}));  // broadcast {1, s0} {s0, 1} -> {s0, s0}

    RUN_BROADCAST_TEST(gert::SymbolShape({kOne, s0}), gert::SymbolShape({s0, kOne, s0}),
                       gert::SymbolShape({s0, kOne, s0}));  // broadcast {1, s0} {s0, 1, s0} -> {s0, 1, s0}

    RUN_BROADCAST_TEST(gert::SymbolShape({s2, s1, s0}), gert::SymbolShape({s0}),
                       gert::SymbolShape({s2, s1, s0}));  // broadcast {s2, s1, s0} {s0} -> {s2, s1, s0}

    RUN_BROADCAST_TEST(gert::SymbolShape({s3, s2, kOne, s0}), gert::SymbolShape({kOne, s0}),
                       gert::SymbolShape({s3, s2, kOne, s0}));  // broadcast {s2, s1, s0} {s0} -> {s2, s1, s0}

    RUN_BROADCAST_TEST(gert::SymbolShape({s3, s2, kOne, s0}), gert::SymbolShape(),
                       gert::SymbolShape({s3, s2, kOne, s0}));  // broadcast {s3, s2, 1, s0} {} -> {s3, s2, 1, s0}

    // broadcast {10, 1, 9} {10, 7, 1} -> {10, 7, 9}
    RUN_BROADCAST_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(1), ge::Symbol(9)}),
                       gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(1)}),
                       gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}));
  }
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);

    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));

    auto kOne = ge::Symbol(1);
    auto kThree = ge::Symbol(3);

    auto func = GetInferFunc(op_type);
    ASSERT_TRUE(func.first != nullptr);

    RUN_BROADCAST_TEST(gert::SymbolShape({s0, s1, s2,kThree}),
                       gert::SymbolShape({s3,s4,kOne,s5}),
                       gert::SymbolShape({s0, s4, s2,kThree}));

    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 8);
    const std::set<std::string> expect_guard = {"ExpectEq(1, s1)", "ExpectEq(1, s3)", "ExpectEq(3, s5)",
                                                "ExpectNe(1, s2)", "ExpectNe(1, s4)", "ExpectNe(s0, s3)",
                                                "ExpectNe(s1, s4)", "ExpectNe(1, s0)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForMul) {
  TestForBroadCast("mul", "Mul");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAdd) {
  TestForBroadCast("add", "Add");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAddV2) {
  TestForBroadCast("addV2", "AddV2");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSub) {
  TestForBroadCast("sub", "Sub");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForRealDiv) {
  TestForBroadCast("realdiv", "RealDiv");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForCast) {
  TestForElementWise("cast", "Cast");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForEqual) {
  TestForBroadCast("equal", "Equal");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForGreater) {
  TestForBroadCast("greater", "Greater");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForNotEqual) {
  TestForBroadCast("notequal", "NotEqual");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForMaximum) {
  TestForBroadCast("maximum", "Maximum");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForMinimum) {
  TestForBroadCast("minimum", "Minimum");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLogicalAnd) {
  TestForBroadCast("logicaland", "LogicalAnd");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLogicalOr) {
  TestForBroadCast("logicalor", "LogicalOr");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForDiv) {
  TestForBroadCast("div", "Div");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLessEqual) {
  TestForBroadCast("lessequal", "LessEqual");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForGreaterEqual) {
  TestForBroadCast("greaterEqual", "GreaterEqual");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForDivNoNan) {
  TestForBroadCast("divNoNan", "DivNoNan");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLeakyReluGrad) {
  TestForBroadCast("leakyReluGrad", "LeakyReluGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReluGrad) {
  TestForBroadCast("relugrad", "ReluGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForBitwiseAnd) {
  TestForBroadCast("bitewiseand", "BitwiseAnd");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForFloorDiv) {
  TestForBroadCast("floorDiv", "FloorDiv");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForFloorMod) {
  TestForBroadCast("floorMod", "FloorMod");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSqrt) {
  TestForElementWise("sqrt", "Sqrt");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftplus) {
  TestForElementWise("softplus", "Softplus");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForEluGrad) {
  TestForBroadCast("eluGrad", "EluGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxGrad) {
  TestForBroadCast("softmaxGrad", "SoftmaxGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForTanhGrad) {
  TestForBroadCast("tanhGrad", "TanhGrad");
}

static void TestForReduce(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto s3 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  auto kZero = ge::Symbol(0);
  auto kOne = ge::Symbol(1);
  auto kTwo = ge::Symbol(2);
  auto kNegOne = ge::Symbol(-1);
  auto kBig = ge::Symbol(999);

  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);

  // Test cases
  // keep_dims=true {s0, s0} axes={0} -> {1, s0}
  std::vector<ge::Expression> symbol_axes = {kZero};
  RUN_REDUCE_TEST(gert::SymbolShape({s0, s0}), &symbol_axes, true, gert::SymbolShape({kOne, s0}), ge::GRAPH_SUCCESS);

  // keep_dims=true {s3, s2, s1, s0} axes={1,2} -> {s3, 1, 1, s0}
  symbol_axes = {kOne, kTwo};
  RUN_REDUCE_TEST(gert::SymbolShape({s3, s2, s1, s0}), &symbol_axes, true, gert::SymbolShape({s3, kOne, kOne, s0}),
                  ge::GRAPH_SUCCESS);

  // keep_dims=true {10, 7, 9} axes={-1} -> {10, 7, 1}
  symbol_axes = {kNegOne};
  RUN_REDUCE_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), &symbol_axes, true,
                  gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(1)}), ge::GRAPH_SUCCESS);

  // keep_dims=false {s3, s2, s1, s0} axes={1,2} -> {s3, s0}
  symbol_axes = {kOne, kTwo};
  RUN_REDUCE_TEST(gert::SymbolShape({s3, s2, s1, s0}), &symbol_axes, false, gert::SymbolShape({s3, s0}),
                  ge::GRAPH_SUCCESS);

  // keep_dims=false {10, 7, 9} axes={1} -> {10, 9}
  symbol_axes = {kOne};
  RUN_REDUCE_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), &symbol_axes, false,
                  gert::SymbolShape({ge::Symbol(10), ge::Symbol(9)}), ge::GRAPH_SUCCESS);

  // keep_dims=false {10, 7, 9} axes={-1} -> {10, 7}
  symbol_axes = {kNegOne};
  RUN_REDUCE_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), &symbol_axes, false,
                  gert::SymbolShape({ge::Symbol(10), ge::Symbol(7)}), ge::GRAPH_SUCCESS);

  // keep_dims=false  {10, 7, 9} axes={} -> {10, 7, 9}
  symbol_axes = {};
  RUN_REDUCE_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), &symbol_axes, false,
                  gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), ge::GRAPH_SUCCESS);

  // axes is symbol, check return UNSUPPORTED
  symbol_axes = {s2};
  RUN_REDUCE_TEST(gert::SymbolShape({s3, s2, s1, s0}), &symbol_axes, true, gert::SymbolShape({s3, kOne, kOne, s0}),
                  UNSUPPORTED);

  // axes not in dim num, check return PARAM_INVALID
  symbol_axes = {kBig};
  RUN_REDUCE_TEST(gert::SymbolShape({s3, s2, s1, s0}), &symbol_axes, true, gert::SymbolShape({s3, kOne, kOne, s0}),
                  PARAM_INVALID);

  // keep_dims=false  {10, 7, 9} axes=nullptr -check return UNSUPPORTED
  RUN_REDUCE_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), nullptr, false,
                  gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceSum) {
  TestForReduce("reduceSum", "ReduceSum");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceMax) {
  TestForReduce("reduceMax", "ReduceMax");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceMin) {
  TestForReduce("reduceMin", "ReduceMin");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceProd) {
  TestForReduce("reduceProd", "ReduceProd");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceMean) {
  TestForReduce("reduceMean", "ReduceMean");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceAll) {
  TestForReduce("reduceAll", "ReduceAll");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceAny) {
  TestForReduce("reduceAny", "ReduceAny");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForNeg) {
  TestForElementWise("neg", "Neg");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForNPow) {
  TestForBroadCast("pow", "Pow");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLogicalNot) {
  TestForElementWise("logicalnot", "LogicalNot");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForIdentity) {
  TestForElementWise("Identity", "Identity");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSoftmaxV2) {
  TestForElementWise("SoftmaxV2", "SoftmaxV2");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForApplyGradientDescent) {
  TestForElementWise("ApplyGradientDescent", "ApplyGradientDescent");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAssignAdd) {
  TestForElementWise("AssignAdd", "AssignAdd");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSplit) {
  auto func = GetInferFunc("Split");
  ASSERT_TRUE(func.first != nullptr);

  // input_shape是null
  InferSymbolShapeContextTestBuilder builder("Split", "split");
  auto infer_context = builder.Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  // 不存在attr
  builder.Destroy();

  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

  std::vector<ge::Expression> symbol_value = {Symbol(1)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // num_split不是正数
  builder.Destroy();
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", -1);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // split_num不合法
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  symbol_value = {Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // split_dim nullptr
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), false, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  // split_dim value为空
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  symbol_value = {};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  // split_dim value为空
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  symbol_value = {s0};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  // 正常split_dim = 1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  symbol_value = {Symbol(1)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto output_shape0 = infer_context->GetOutputSymbolShape(0);
  auto output_shape1 = infer_context->GetOutputSymbolShape(1);
  auto in_shape = infer_context->GetInputSymbolShape(1);
  ASSERT_EQ(in_shape->GetDimNum(), output_shape0->GetDimNum());
  ASSERT_EQ(in_shape->GetDimNum(), output_shape1->GetDimNum());

  // dim 1
  ASSERT_EQ(in_shape->GetDim(0), output_shape0->GetDim(0));
  ASSERT_EQ(in_shape->GetDim(0), output_shape1->GetDim(0));
  // dim 2
  auto dim2_expect = in_shape->GetDim(1) / ge::Symbol(2);
  ASSERT_EQ(dim2_expect, output_shape0->GetDim(1));
  ASSERT_EQ(dim2_expect, output_shape1->GetDim(1));

  // 正常split_dim = -1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  symbol_value = {Symbol(-2)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  output_shape0 = infer_context->GetOutputSymbolShape(0);
  output_shape1 = infer_context->GetOutputSymbolShape(1);
  in_shape = infer_context->GetInputSymbolShape(1);
  ASSERT_EQ(in_shape->GetDimNum(), output_shape0->GetDimNum());
  ASSERT_EQ(in_shape->GetDimNum(), output_shape1->GetDimNum());

  // dim 1
  auto dim1_expect = in_shape->GetDim(0) / ge::Symbol(2);
  ASSERT_EQ(dim1_expect, output_shape0->GetDim(0));
  ASSERT_EQ(dim1_expect, output_shape1->GetDim(0));
  // dim 2
  ASSERT_EQ(in_shape->GetDim(1), output_shape0->GetDim(1));
  ASSERT_EQ(in_shape->GetDim(1), output_shape1->GetDim(1));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSplitV) {
  // input0: data
  // input1: size_splits
  // input2: split_dim
  // attr0: num_split
  auto func = GetInferFunc("SplitV");
  ASSERT_TRUE(func.first != nullptr);
  InferSymbolShapeContextTestBuilder builder("SplitV", "splitv");
  {
    // 1. num_split校验失败
    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", -1);
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

    std::vector<ge::Expression> size_splits_symbol_value = {Symbol(1), Symbol(-1), Symbol(2)};
    std::vector<ge::Expression> split_dim_symbol_value = {Symbol(0)};

    auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                                .AppendInputSymbolTensor(gert::SymbolShape(), true, &size_splits_symbol_value)
                                .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
                                .OutputNum(3)
                                .Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

    // test no symbol value case 1
    builder.Destroy();
    op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 2);
    infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
        .AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr)
        .AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr)
        .OutputNum(3)
        .Build();
    ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);

    // test no symbol value case 2
    builder.Destroy();
    op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 2);
    infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
        .AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr)
        .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
        .OutputNum(3)
        .Build();
    ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);

    // test no symbol value case 3
    builder.Destroy();
    op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 3);
    size_splits_symbol_value = {s0, Symbol(-1), Symbol(2)};
    split_dim_symbol_value = {Symbol(1)};
    infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
        .AppendInputSymbolTensor(gert::SymbolShape(), true, &size_splits_symbol_value)
        .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
        .OutputNum(3)
        .Build();
    ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);

    // 2. split_dim 校验失败
    builder.Destroy();
    op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 3);
    size_splits_symbol_value = {Symbol(1), Symbol(-1), Symbol(2)};
    split_dim_symbol_value = {Symbol(3)};
    infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &size_splits_symbol_value)
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
                           .OutputNum(3)
                           .Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

    // 3. dynamic_value_num校验失败,有多个-1
    builder.Destroy();
    op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 3);
    size_splits_symbol_value = {Symbol(1), Symbol(-1), Symbol(-1)};
    split_dim_symbol_value = {Symbol(1)};
    infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &size_splits_symbol_value)
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
                           .OutputNum(3)
                           .Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  }
  {
    // 4. 正常情况
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 3);
    std::vector<ge::Expression> size_splits_symbol_value = {Symbol(1), Symbol(-1), Symbol(2)};
    std::vector<ge::Expression> split_dim_symbol_value = {Symbol(1)};
    auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &size_splits_symbol_value)
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
                           .OutputNum(3)
                           .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto output_shape0 = infer_context->GetOutputSymbolShape(0);
    auto output_shape1 = infer_context->GetOutputSymbolShape(1);
    auto output_shape2 = infer_context->GetOutputSymbolShape(2);
    auto in_shape = infer_context->GetInputSymbolShape(0);
    ASSERT_EQ(in_shape->GetDimNum(), output_shape0->GetDimNum());
    ASSERT_EQ(in_shape->GetDimNum(), output_shape1->GetDimNum());
    ASSERT_EQ(in_shape->GetDimNum(), output_shape2->GetDimNum());

    // dim 1
    ASSERT_EQ(in_shape->GetDim(0), output_shape0->GetDim(0));
    ASSERT_EQ(in_shape->GetDim(0), output_shape1->GetDim(0));
    ASSERT_EQ(in_shape->GetDim(0), output_shape2->GetDim(0));
    // dim 2
    ASSERT_EQ(Symbol(1), output_shape0->GetDim(1));
    ASSERT_EQ(in_shape->GetDim(1) - Symbol(3), output_shape1->GetDim(1));
    ASSERT_EQ(Symbol(2), output_shape2->GetDim(1));

    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 1);
    const std::set<std::string> assert_guard = {
      "ExpectLe(3, s1)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // 5. 异常情况：size_splits之和大于dim
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("num_split");
    AttrUtils::SetInt(op_desc, "num_split", 3);
    std::vector<ge::Expression> size_splits_symbol_value = {Symbol(3), Symbol(-1), Symbol(2)};
    std::vector<ge::Expression> split_dim_symbol_value = {Symbol(1)};
    auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &size_splits_symbol_value)
                           .AppendInputSymbolTensor(gert::SymbolShape(), true, &split_dim_symbol_value)
                           .OutputNum(3)
                           .Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForFill) {
  auto func = GetInferFunc("Fill");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("Fill", "fill");

  // 正常场景：入参dims值为[3, 3, 4, 5], 期望fill输出的shape:{3, 3, 4, 5}
  auto s3 = ge::Symbol(3);
  auto s4 = ge::Symbol(4);
  auto s5 = ge::Symbol(5);
  std::vector<ge::Expression> dims = {s3, s3, s4, s5};
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, &dims).OutputNum(1).Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);

  auto output_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_EQ(output_shape->GetDims(), gert::SymbolShape({s3, s3, s4, s5}).GetDims());

  builder.Destroy();
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr).OutputNum(1).Build();

  ASSERT_EQ(func.first(infer_context), UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSlice) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(8, MakeShared<InputShapeSource>(0, 0));
  Symbol s1 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 1));
  auto unkown = Symbol(-1);
  auto unkown2 = Symbol(-2);
  auto k0 = Symbol(0);
  auto k1 = Symbol(1);
  auto k2 = Symbol(2);
  auto k3 = Symbol(3);
  auto k36 = Symbol(36);
  auto k37 = Symbol(37);

  auto op_type = "Slice";
  auto op_name = "slice";
  // 场景1：size没有-1 -> 结果为 size
  vector input_shapes = {gert::SymbolShape({s0, s1})};
  vector data_depends = {gert::SymbolTensor({k2}, {k0, k0}), gert::SymbolTensor({k2}, {k1, k2})};
  vector output_shapes = {gert::SymbolShape({k1, k2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景2：size有-1 -> 结果为 input - offset
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({k2}, {k1, k2}), gert::SymbolTensor({k2}, {k1, unkown})};
  output_shapes = {gert::SymbolShape({k1, s1 - k2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景3：size全是-1 -> 结果为 input - offset
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({k2}, {k1, k2}), gert::SymbolTensor({k2}, {unkown, unkown})};
  output_shapes = {gert::SymbolShape({s0 - k1, s1 - k2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景4：offset有-1 -> 参数无效，不支持
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({k2}, {k1, unkown}), gert::SymbolTensor({k2}, {unkown, unkown})};
  output_shapes = {gert::SymbolShape({s0 - k1, s1 - k2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, PARAM_INVALID);

  // 场景5：size.dim != input.dim -> 参数无效，不支持
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({k2}, {k0, k0}), gert::SymbolTensor({}, {})};
  output_shapes = {gert::SymbolShape({unkown, unkown})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, PARAM_INVALID);

  // 场景6：offset.dim != input.dim-> 参数无效，不支持
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({}, {}), gert::SymbolTensor({k2}, {k36, k37})};
  output_shapes = {gert::SymbolShape({unkown, unkown})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, PARAM_INVALID);

  // test no symbol value case 1
  input_shapes = {gert::SymbolShape({s0, s1})};
  auto tensorx = gert::SymbolTensor();
  data_depends = {gert::SymbolTensor(), gert::SymbolTensor()};
  output_shapes = {gert::SymbolShape({unkown, unkown})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, UNSUPPORTED);

  // test no symbol value case 2
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({k2}, {k36, k37}), gert::SymbolTensor()};
  output_shapes = {gert::SymbolShape({unkown, unkown})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForRange) {
  using namespace gert;
  auto op_type = "Range";
  auto op_name = "range";
  vector<SymbolShape> input_shapes;
  vector<SymbolTensor> data_depends;
  vector<SymbolShape> output_shapes;
  auto k1 = Symbol(1);
  // 场景1：都非空，范围整数，步长整数，start < end -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}),  // start
                  SymbolTensor({k1}, {Symbol(10)}), // end
                  SymbolTensor({k1}, {Symbol(1)})}; // 步长
  output_shapes = {SymbolShape({Symbol(9)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景2：都非空，范围整数，步长小数，start < end -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}),    // start
                  SymbolTensor({k1}, {Symbol(2)}),    // end
                  SymbolTensor({k1}, {Symbol(0.4)})}; // 步长
  output_shapes = {SymbolShape({Symbol(3)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景3：都非空，范围整数，步长小数，start = end -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}),    // start
                  SymbolTensor({k1}, {Symbol(1)}),    // end
                  SymbolTensor({k1}, {Symbol(0.4)})}; // 步长
  output_shapes = {SymbolShape({Symbol(0)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景4：都非空，范围整数，步长小数，start > end，且步长为正数 -> 异常
  data_depends = {SymbolTensor({}, {Symbol(1)}),    // start
                  SymbolTensor({}, {Symbol(0)}),    // end
                  SymbolTensor({}, {Symbol(0.4)})}; // 步长
  output_shapes = {SymbolShape({Symbol(0)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, PARAM_INVALID);
  // 场景5：都非空，范围整数，步长整数，start < end，且start为负数 -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(-1)}), // start
                  SymbolTensor({k1}, {Symbol(10)}), // end
                  SymbolTensor({k1}, {Symbol(1)})}; // 步长
  output_shapes = {SymbolShape({Symbol(11)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景6：都非空，范围整数，步长小数，start < end，且start为负数 -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(-1)}),   // start
                  SymbolTensor({k1}, {Symbol(10)}),   // end
                  SymbolTensor({k1}, {Symbol(0.4)})}; // 步长
  output_shapes = {SymbolShape({Symbol(28)})};

  // 场景7：都非空，范围整数，步长整数，start < end，且都为负数 -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(-10)}), // start
                  SymbolTensor({k1}, {Symbol(-2)}),  // end
                  SymbolTensor({k1}, {Symbol(2)})};  // 步长
  output_shapes = {SymbolShape({Symbol(4)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景8：都非空，范围整数，步长小数，start < end，且都为负数 -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(-10)}),  // start
                  SymbolTensor({k1}, {Symbol(-2)}),   // end
                  SymbolTensor({k1}, {Symbol(0.7)})}; // 步长
  output_shapes = {SymbolShape({Symbol(12)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景9：都非空，范围小数，步长整数，start < end，且都为负数 -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(-10.7)}), // start
                  SymbolTensor({k1}, {Symbol(-2.8)}),  // end
                  SymbolTensor({k1}, {Symbol(2)})};    // 步长
  output_shapes = {SymbolShape({Symbol(4)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景10：都非空，范围小数，步长小数，start < end，且都为负数 -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(-10.7)}), // start
                  SymbolTensor({k1}, {Symbol(-2.8)}),  // end
                  SymbolTensor({k1}, {Symbol(0.3)})};  // 步长
  output_shapes = {SymbolShape({Symbol(27)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景11：步长为空(默认值为1) -> 正常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}), // start
                  SymbolTensor({k1}, {Symbol(3)}), // end
                  SymbolTensor()};                 // 步长
  output_shapes = {SymbolShape({Symbol(2)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 场景12：end为空(默认值为start) -> 异常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}),  // start
                  SymbolTensor(),                   // end
                  SymbolTensor({k1}, {Symbol(3)})}; // 步长
  output_shapes = {SymbolShape({Symbol(0)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, UNSUPPORTED);

  // 场景13：start为空(默认值为0) -> 异常
  data_depends = {SymbolTensor(),                     // start
                  SymbolTensor({k1}, {Symbol(1)}),    // end
                  SymbolTensor({k1}, {Symbol(0.3)})}; // 步长
  output_shapes = {SymbolShape({Symbol(4)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, UNSUPPORTED);

  // 场景14：都为空 -> 异常
  data_depends = {SymbolTensor(),  // start
                  SymbolTensor(),  // end
                  SymbolTensor()}; // 步长
  output_shapes = {SymbolShape({Symbol(0)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, UNSUPPORTED);

  {
    // 场景15：都非空，范围符号，步长整数，start < end -> 正常
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    data_depends = {SymbolTensor({k1}, {s0}), // start
                    SymbolTensor({k1}, {s1}), // end
                    SymbolTensor({k1}, {Symbol(1)})};   // 步长
    output_shapes = {SymbolShape({sym::Ceiling(s1 - s0)})};
    EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 1);
    const std::set<std::string> expect_guard = {"ExpectLt(s0, s1)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 0);
  }

  // 场景16：都非空，范围整数，步长为0 -> 异常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}),  // start
                  SymbolTensor({k1}, {Symbol(3)}),  // end
                  SymbolTensor({k1}, {Symbol(0)})}; // 步长
  output_shapes = {SymbolShape()};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, PARAM_INVALID);

  // 场景17：缺少输入 -> 异常
  data_depends = {SymbolTensor({k1}, {Symbol(1)}),  // start
                  SymbolTensor({k1}, {Symbol(3)})}; // end
  output_shapes = {SymbolShape({Symbol(2)})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  {
    // 场景18：有且仅有第一个输入 -> 正常
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
    data_depends = {SymbolTensor({}, {s0}), SymbolTensor(), SymbolTensor()};
    output_shapes = {SymbolShape({sym::Ceiling(s0)})};
    EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 1);
    const std::set<std::string> expect_guard = {"ExpectLt(0, s0)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 0);
  }
  {
    // 场景19：start>end, delta为负数  -> 异常
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(10, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(-1, MakeShared<InputShapeSource>(0, 2));
    data_depends = {SymbolTensor({}, {s0}), SymbolTensor({}, {s1}), SymbolTensor({}, {s2})};
    output_shapes = {SymbolShape({sym::Ceiling((s1 - s0) / s2)})};
    EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 2);
    const std::set<std::string> expect_guard = {"ExpectLt(s1, s0)", "ExpectLe(s1, s0)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 2);
    const std::set<std::string> assert_guard = {"ExpectNe(0, s2)", "ExpectLt(s2, 0)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConcatV2D) {
  ConcatV2DTest("ConcatV2D");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConcatD) {
  ConcatV2DTest("ConcatD");
}

static void TestForReduceD(const std::string &op_name, const std::string &op_type) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  auto s3 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 3));

  auto kOne = ge::Symbol(1);

  auto func = GetInferFunc(op_type);
  ASSERT_TRUE(func.first != nullptr);

  // Test cases
  // keep_dims=true {s0, s1, s2} axes={0, 1} -> {1, 1, s2}
  RUN_REDUCE_D_TEST(gert::SymbolShape({s0, s1, s2}), std::vector<int64_t>({0, 1}), true,
                    gert::SymbolShape({kOne, kOne, s2}), GRAPH_SUCCESS);

  // keep_dims=true {10, 7, 9} axes={-1} -> {10, 7, 1}
  RUN_REDUCE_D_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), std::vector<int64_t>({-1}), true,
                    gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(1)}), ge::GRAPH_SUCCESS);

  // keep_dims=false {s3, s2, s1, s0} axes={1,2} -> {s3, s0}
  RUN_REDUCE_D_TEST(gert::SymbolShape({s3, s2, s1, s0}), std::vector<int64_t>({1, 2}), false,
                    gert::SymbolShape({s3, s0}), ge::GRAPH_SUCCESS);

  // keep_dims=false {10, 7, 9} axes={1} -> {10, 9}
  RUN_REDUCE_D_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), std::vector<int64_t>({1}), false,
                    gert::SymbolShape({ge::Symbol(10), ge::Symbol(9)}), ge::GRAPH_SUCCESS);

  // keep_dims=false {10, 7, 9} axes={-1} -> {10, 7}
  RUN_REDUCE_D_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), std::vector<int64_t>({-1}),
                    false, gert::SymbolShape({ge::Symbol(10), ge::Symbol(7)}), ge::GRAPH_SUCCESS);

  // keep_dims=false  {10, 7, 9} axes={} -> {10, 7, 9}
  RUN_REDUCE_D_TEST(gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), std::vector<int64_t>({}), false,
                    gert::SymbolShape({ge::Symbol(10), ge::Symbol(7), ge::Symbol(9)}), ge::GRAPH_SUCCESS);

  // axes not in dim num, check return PARAM_INVALID
  RUN_REDUCE_D_TEST(gert::SymbolShape({s3, s2, s1, s0}), std::vector<int64_t>({999}), true,
                    gert::SymbolShape({s3, kOne, kOne, s0}), PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceSumD) {
  TestForReduceD("reduceSumD", "ReduceSumD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceMaxD) {
  TestForReduceD("reduceMaxD", "ReduceMaxD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceMinD) {
  TestForReduceD("reduceMinD", "ReduceMinD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceMeanD) {
  TestForReduceD("reduceMeanD", "ReduceMeanD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceProdD) {
  TestForReduceD("reduceProdD", "ReduceProdD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceAllD) {
  TestForReduceD("reduceAllD", "ReduceAllD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReduceAnyD) {
  TestForReduceD("reduceAnyD", "ReduceAnyD");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSelect) {
  auto func = GetInferFunc("Select");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

  auto condition = gert::SymbolShape({s0, s1});
  auto input_shape0 = gert::SymbolShape({s0, s1});
  auto input_shape1 = gert::SymbolShape({s0, s1});
  InferSymbolShapeContextTestBuilder builder("Select", "select");
  {
    // 全相等
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 5));
    auto condition = gert::SymbolShape({s0, s1});
    auto input_shape0 = gert::SymbolShape({s2, s3});
    auto input_shape1 = gert::SymbolShape({s4, s5});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                                .AppendInputSymbolTensor(input_shape0)
                                .AppendInputSymbolTensor(input_shape1)
                                .OutputNum(1)
                                .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(guard_infos.size(), 4);
    const std::set<std::string> expect_guard = {"ExpectEq(s1, s5)", "ExpectEq(s0, s4)",
                                                "ExpectEq(s3, s5)", "ExpectEq(s2, s4)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
  }
  {
    // condition维度数量为1且等于输入的第0维度
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 5));
    auto condition = gert::SymbolShape({s0});
    auto input_shape0 = gert::SymbolShape({s2, s3});
    auto input_shape1 = gert::SymbolShape({s4, s5});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                                .AppendInputSymbolTensor(input_shape0)
                                .AppendInputSymbolTensor(input_shape1)
                                .OutputNum(1)
                                .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(guard_infos.size(), 3);
    const std::set<std::string> expect_guard = {"ExpectEq(s0, s4)", "ExpectEq(s3, s5)", "ExpectEq(s2, s4)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
  }
  {
    // condition维度数量为0
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 5));
    auto condition = gert::SymbolShape({});
    auto input_shape0 = gert::SymbolShape({s2, s3});
    auto input_shape1 = gert::SymbolShape({s4, s5});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                                .AppendInputSymbolTensor(input_shape0)
                                .AppendInputSymbolTensor(input_shape1)
                                .OutputNum(1)
                                .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(guard_infos.size(), 2);
    const std::set<std::string> expect_guard = {"ExpectEq(s3, s5)", "ExpectEq(s2, s4)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
  }
  {
    // condition维度数量不为0或1且维度数量不等于后面的输入
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 5));
    auto s6 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 6));
    auto s7 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 7));
    auto condition = gert::SymbolShape({s0, s1});
    auto input_shape0 = gert::SymbolShape({s2, s3, s4});
    auto input_shape1 = gert::SymbolShape({s5, s6, s7});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                                .AppendInputSymbolTensor(input_shape0)
                                .AppendInputSymbolTensor(input_shape1)
                                .OutputNum(1)
                                .Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(guard_infos.size(), 3);
    const std::set<std::string> expect_guard = {"ExpectEq(s2, s5)", "ExpectEq(s3, s6)", "ExpectEq(s4, s7)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForTile) {
  auto func = GetInferFunc("Tile");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  InferSymbolShapeContextTestBuilder builder("Tile", "tile");

  // 场景1: 输入shape和重复次数大小相等
  // 输入shape{s0, s1}, 重复次数[2, 3]
  // 期望输出shape{s0*2, s1*3}
  auto input_shape = gert::SymbolShape({s0, s1});
  std::vector<ge::Expression> multiples = {Symbol(2), Symbol(3)};
  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                              .AppendInputSymbolTensor(gert::SymbolShape(), true, &multiples)
                              .OutputNum(1)
                              .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s0 * ge::Symbol(2), s1 * ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 场景2：输入shape大于重复次数大小
  // 输入shape{s0, s1, s2}, 重复次数[2, 3]
  // 期望输出shape{s0*1, s1*2, s2*3}
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1, s2});
  multiples = {Symbol(2), Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &multiples)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape = gert::SymbolShape(
      {s0 * ge::Symbol(1), s1 * ge::Symbol(2), s2 * ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 场景2：输入shape小于重复次数大小
  // 输入shape{s0, s1}, 重复次数[2, 3, 4]
  // 期望输出shape{1*2, s0*3, s1*4}
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1});
  multiples = {Symbol(2), Symbol(3), Symbol(4)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &multiples)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape = gert::SymbolShape(
      {ge::Symbol(1) * ge::Symbol(2), s0 * ge::Symbol(3), s1 * ge::Symbol(4)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // test no symbol value
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1, s2});
  multiples = {Symbol(2), Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
      .AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr)
      .OutputNum(1)
      .Build();
  ASSERT_EQ(func.first(infer_context), UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForTileD) {
  auto func = GetInferFunc("TileD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("TileD", "tile");

  // 场景1: 输入shape和重复次数大小相等
  // 输入shape{s0, s1}, 重复次数[2, 3]
  // 期望输出shape{s0*2, s1*3}
  auto input_shape = gert::SymbolShape({s0, s1});
  std::vector<int64_t> multiples = {2, 3};
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("multiples");
  AttrUtils::SetListInt(op_descPtr, "multiples", multiples);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s0 * ge::Symbol(2), s1 * ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 场景2：输入shape大于重复次数大小
  // 输入shape{s0, s1, s2}, 重复次数[2, 3]
  // 期望输出shape{s0*1, s1*2, s2*3}
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1, s2});
  multiples = {2, 3};
  op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("multiples");
  AttrUtils::SetListInt(op_descPtr, "multiples", multiples);
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape = gert::SymbolShape(
      {s0 * ge::Symbol(1), s1 * ge::Symbol(2), s2 * ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 场景3：输入shape小于重复次数大小
  // 输入shape{s0, s1}, 重复次数[2, 3, 4]
  // 期望输出shape{1*2, s0*3, s1*4}
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1});
  multiples = {2, 3, 4};
  op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("multiples");
  AttrUtils::SetListInt(op_descPtr, "multiples", multiples);
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape = gert::SymbolShape(
      {ge::Symbol(1) * ge::Symbol(2), s0 * ge::Symbol(3), s1 * ge::Symbol(4)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForTranpose) {
  // input_0 data
  // input_1 perm
  // attr0 inserted_by_fe
  auto func = GetInferFunc("Transpose");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  InferSymbolShapeContextTestBuilder builder("Transpose", "transpose");
  // 1. 异常场景1：perm的维度跟输入维度不同
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("_inserted_by_fe");
  AttrUtils::SetInt(op_desc, "_inserted_by_fe", 0);
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  std::vector<ge::Expression> symbol_value = {Symbol(1), Symbol(0), Symbol(2)};
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                              .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                              .OutputNum(1)
                              .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 2. 正常场景1：inserted_by_fe == 0
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("_inserted_by_fe");
  AttrUtils::SetInt(op_desc, "_inserted_by_fe", 0);
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  symbol_value = {Symbol(2), Symbol(0), Symbol(1)};
  infer_context =
      builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
             .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
             .OutputNum(1)
             .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto in_shape = infer_context->GetInputSymbolShape(0);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), in_shape->GetDimNum());
  // dim1
  ASSERT_EQ(out_shape->GetDim(0), in_shape->GetDim(2));
  // dim2
  ASSERT_EQ(out_shape->GetDim(1), in_shape->GetDim(0));
  // dim3
  ASSERT_EQ(out_shape->GetDim(2), in_shape->GetDim(1));
  // 3. 正常场景2：inserted_by_fe != 0
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("_inserted_by_fe");
  AttrUtils::SetInt(op_desc, "_inserted_by_fe", -1);
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));
  symbol_value = {Symbol(1), Symbol(0), Symbol(2)};
  infer_context =
      builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
             .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
             .OutputNum(1)
             .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  in_shape = infer_context->GetInputSymbolShape(0);
  out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), in_shape->GetDimNum());
  // dim1
  ASSERT_EQ(out_shape->GetDim(0), in_shape->GetDim(0));
  // dim2
  ASSERT_EQ(out_shape->GetDim(1), in_shape->GetDim(1));
  // dim3
  ASSERT_EQ(out_shape->GetDim(2), in_shape->GetDim(2));

  // test no symbol value
  builder.Destroy();
  infer_context =
      builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
          .AppendInputSymbolTensor(gert::SymbolShape(), true, nullptr)
          .OutputNum(1)
          .Build();
  ASSERT_EQ(func.first(infer_context), UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForTranposeD) {
  // input_0 data
  // attr0 perm
  // attr1 perm_num
  auto func = GetInferFunc("TransposeD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("TransposeD", "transposeD");
  // 1. 异常场景1：perm的维度跟输入维度不同
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("perm");
  std::vector<int64_t> perm_list0 = {1, 2, 0};
  AttrUtils::SetListInt(op_desc, "perm", perm_list0);
  op_desc->AddInputDesc(GeTensorDesc());
  auto infer_context =
      builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1})).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 2. 正常场景1：perm=[1, 2, 0]
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("perm");
  std::vector<int64_t> perm_list1 = {1, 2, 0};
  AttrUtils::SetListInt(op_desc, "perm", perm_list1);
  op_desc->AddInputDesc(GeTensorDesc());
  infer_context =
      builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
             .OutputNum(1)
             .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto in_shape = infer_context->GetInputSymbolShape(0);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), in_shape->GetDimNum());
  // dim1
  ASSERT_EQ(out_shape->GetDim(0), in_shape->GetDim(1));
  // dim2
  ASSERT_EQ(out_shape->GetDim(1), in_shape->GetDim(2));
  // dim3
  ASSERT_EQ(out_shape->GetDim(2), in_shape->GetDim(0));
  builder.Destroy();
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForReshape) {
  auto func = GetInferFunc("Reshape");
  ASSERT_TRUE(func.first != nullptr);
  InferSymbolShapeContextTestBuilder builder("Reshape", "reshape");
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 4));
  auto s4 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 5));
  // 1. 异常场景：shape为0的下标大于输入数据的维度
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));

  std::vector<ge::Expression> symbol_value = {Symbol(1), Symbol(2), Symbol(0)};
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                              .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                              .OutputNum(1)
                              .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 2. 异常场景：shape里的常量符号不为int32或者int64
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));

  symbol_value = {Symbol(1.1), Symbol(2), Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  // 3. 异常场景：shape里有多个-1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));

  symbol_value = {Symbol(1), Symbol(-1), Symbol(-1)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 4. 异常场景：shape的长度为0
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));

  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, {})
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  auto out_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_EQ(out_shape->GetDimNum(), 0);
  // 5. 正常场景：含有-1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  symbol_value = {Symbol(1), Symbol(-1), Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), 3);
  ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
  auto expect_dim = (s0 * s1 * s2) / (Symbol(1) * Symbol(3));
  ASSERT_EQ(out_shape->GetDim(1), expect_dim);
  ASSERT_EQ(out_shape->GetDim(2), Symbol(3));
  // 6. 正常场景：不含-1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));

  symbol_value = {Symbol(1), Symbol(0), Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s3}))
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), 3);
  ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
  ASSERT_EQ(out_shape->GetDim(1), infer_context->GetInputSymbolShape(0)->GetDim(1));
  ASSERT_EQ(out_shape->GetDim(2), Symbol(3));

  // 7. 正常场景，包含符号
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));

  symbol_value = {Symbol(1), s4, Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s3}))
                         .AppendInputSymbolTensor(gert::SymbolShape(), true, &symbol_value)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), 3);
  ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
  ASSERT_EQ(out_shape->GetDim(1), s4);
  ASSERT_EQ(out_shape->GetDim(2), Symbol(3));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSqueeze) {
  const auto func = GetInferFunc("Squeeze");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("Squeeze", "squeeze");

  // 正常场景1：不传入axis值，压缩所有维度值等于1的维度
  // input_shape={s0, 1, s1, 1}  axis={}
  // 期望out_shape={s0, s1}
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  auto input_shape = gert::SymbolShape({s0, ge::Symbol(1), s1, ge::Symbol(1)});
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", {});
  auto infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s0, s1});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 正常场景2：传入axis且所要压缩的维度值等于1
  // input_shape={s0, 1, s1, 1}  axis={1，3}
  // 期望out_shape={s0, s1}
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", {1, 3});
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape = gert::SymbolShape({s0, s1});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 异常场景1：传入axis, 要压缩的维度值不等于1
  // input_shape={s0, 1, s1, 2}  axis={3}
  // 推导失败
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, ge::Symbol(1), s1, ge::Symbol(2)});
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", {3});
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForUnSqueeze) {
  const auto func = GetInferFunc("Unsqueeze");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("UnSqueeze", "unSqueeze");
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  // 正常场景1：不传入axis，输出shape等于输入shape
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", {});

  auto infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s0, s1, s2});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 正常场景2：传入axis, 在对应维度上扩展新的维度1
  // input_shape={s0, s1, s2}, axis={1，3}
  // 期望out_shape={s0, 1, s1, 1， s2}
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", {1, 3});

  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape =
      gert::SymbolShape({s0, ge::Symbol(1), s1, ge::Symbol(1), s2});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 异常场景1：传入axis，axis的值超过输入的shape dim num + axis 的size
  // input_shape={s0, s1, s2}, axis={5}
  // 推导失败
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", {5});

  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), PARAM_INVALID);
}

static void EXPECT_GatherV2CommonTest(int64_t axis, int64_t batch_dims, graphStatus status = GRAPH_SUCCESS,
                                      const gert::SymbolShape &indices_shape = gert::SymbolShape(),
                                      const std::vector<ge::Expression> &indices_symbol_value = {}, DataType dt = DT_INT32, bool add_axis = true) {
  auto func = GetInferFunc("GatherV2");
  ASSERT_TRUE(func.first != nullptr);
  InferSymbolShapeContextTestBuilder builder("GatherV2", "gatherv2");

  auto op_desc = builder.GetOrCreateOpDescPtr();
  // set batch_dims
  op_desc->AppendIrAttrName("batch_dims");
  AttrUtils::SetInt(op_desc, "batch_dims", batch_dims);

  // input0: x
  // input1: indices
  // input2: axis
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, dt));
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  // x shape: {s0, s1, s2}
  // indices shape: 函数入参输入, 可能是标量，向量，张量
  // axis: 常量，函数输入
  std::vector<ge::Expression> axis_symbolic_value = {Symbol(axis)};
  auto axis_symbolic_value_ptr = &axis_symbolic_value;
  if (!add_axis) {
    axis_symbolic_value_ptr = nullptr;
  }

  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2}))
                              .AppendInputSymbolTensor(indices_shape, false, &indices_symbol_value)
                              .AppendInputSymbolTensor(gert::SymbolShape(), true, axis_symbolic_value_ptr)
                              .OutputNum(1)
                              .Build();
  ASSERT_EQ(func.first(infer_context), status);
  if (status != GRAPH_SUCCESS) {
    return;
  }
  auto input_x = infer_context->GetInputSymbolShape(0);
  auto input_indices = infer_context->GetInputSymbolShape(1);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  auto out_dim_num = static_cast<int64_t>(out_shape->GetDimNum());
  auto x_dim_num = static_cast<int64_t>(input_x->GetDimNum());
  auto indices_dim_num = static_cast<int64_t>(input_indices->GetDimNum());

  axis = axis < 0 ? axis + x_dim_num : axis;
  batch_dims = batch_dims < 0 ? batch_dims + indices_dim_num : batch_dims;
  // indices是标量
  if (indices_dim_num == 0) {
    ASSERT_EQ(out_dim_num, x_dim_num - 1);
    for (int64_t i = 0; i < out_dim_num; ++i) {
      if (i == axis) {
        continue;
      }
      ASSERT_EQ(out_shape->GetDim(i), input_x->GetDim(i));
    }
  }
  // indices是向量
  else if (indices_dim_num == 1) {
    ASSERT_EQ(out_dim_num, x_dim_num);
    for (int64_t i = 0; i < out_dim_num; ++i) {
      auto expect = input_x->GetDim(i);
      if (i == axis) {
        expect = input_indices->GetDim(0);
      }
      ASSERT_EQ(out_shape->GetDim(i), expect);
    }
  }
  // indices是张量
  else {
    int64_t expect_dim_num = (x_dim_num - 1) + (indices_dim_num - batch_dims);
    ASSERT_EQ(out_dim_num, expect_dim_num);
    int64_t i = 0;
    for (; i < axis; i++) {
      ASSERT_EQ(out_shape->GetDim(i), input_x->GetDim(i));
    }
    int64_t start_i = i;
    for (; i < axis + (indices_dim_num - batch_dims); i++) {
      ASSERT_EQ(out_shape->GetDim(i), input_indices->GetDim(batch_dims + (i - start_i)));
    }
    start_i = i;
    for (; i < expect_dim_num; i++) {
      ASSERT_EQ(out_shape->GetDim(i), input_x->GetDim(axis + 1 + (i - start_i)));
    }
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForGathverV2) {
  // 异常情况
  // axis超过x的范围
  EXPECT_GatherV2CommonTest(4, 0, PARAM_INVALID);
  // batch_dims超过indices的范围
  EXPECT_GatherV2CommonTest(1, 1, PARAM_INVALID);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));

  // batch_dims大于axis
  auto indices_shape = gert::SymbolShape({s0, s1});
  EXPECT_GatherV2CommonTest(0, 1, PARAM_INVALID, indices_shape);
  // batch_dims大于x的维度
  indices_shape = gert::SymbolShape({s0, s1, s2, s3});
  EXPECT_GatherV2CommonTest(0, 3, PARAM_INVALID, indices_shape);

  // 正常情况
  // indices为标量
  EXPECT_GatherV2CommonTest(1, 0, GRAPH_SUCCESS, gert::SymbolShape(), {Symbol(1)});
  // indices为标量,且Dtype为int64
  EXPECT_GatherV2CommonTest(1, 0, GRAPH_SUCCESS, gert::SymbolShape(), {Symbol(1)}, DT_INT64);
  // indices为向量
  EXPECT_GatherV2CommonTest(1, 0, GRAPH_SUCCESS, gert::SymbolShape({s0}));
  // indices为张量
  EXPECT_GatherV2CommonTest(1, 1, GRAPH_SUCCESS, gert::SymbolShape({s0, s1, s2}));
  // axis、 batch_dims为负
  EXPECT_GatherV2CommonTest(-1, -1, GRAPH_SUCCESS, gert::SymbolShape({s0, s1, s2}));

  EXPECT_GatherV2CommonTest(0, 0, UNSUPPORTED, gert::SymbolShape({s0, s1, s2}), {}, DT_INT32, false);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForPack) {
  using namespace gert;
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  Symbol s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
  Symbol s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
  Symbol s3 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  auto k1 = Symbol(1);
  auto k3 = Symbol(3);

  auto op_type = "Pack";
  auto op_name = "pack";
  auto attr_name = "axis";
  vector<SymbolShape> input_shapes;
  vector<SymbolShape> output_shapes;

  // 正常场景1: axis = 1 ,n=3, input shapes dims=3
  input_shapes = {SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1})};
  output_shapes = {SymbolShape({s0, k3, s2, s1})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 1, output_shapes, GRAPH_SUCCESS);

  // 正常场景2：axis = -1 ,n=3, input shapes dims=3
  input_shapes = {SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1})};
  output_shapes = {SymbolShape({s0, s2, s1, k3})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -1, output_shapes, GRAPH_SUCCESS);

  // 正常场景3：axis = 1 ,n=1, input shapes dims=1
  input_shapes = {SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, k1, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 1, output_shapes, GRAPH_SUCCESS);

  // 正常场景4：axis = 0 ,n=2, input shapes dims=0 标量场景
  input_shapes = {SymbolShape({}), SymbolShape({}), SymbolShape({})};
  output_shapes = {SymbolShape({k3})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 0, output_shapes, GRAPH_SUCCESS);

  // 异常场景1：输入axis超过dims范围  axis = 5 ,n=3, input shapes dims=3
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 5, output_shapes, PARAM_INVALID);

  // 异常场景2：输入axis超过dims范围负数越界  axis = -5 ,n=3, input shapes dims=3
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, -5, output_shapes, PARAM_INVALID);

  // 异常场景3：输入shape之间dims不相同  axis = 1 ,n=3, input shapes dims=3
  input_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s2, s3, s0}), SymbolShape({s2, s1, s0})};
  output_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_CONCATV2D_TEST(input_shapes, 1, output_shapes, PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForUnPack) {
  using namespace gert;
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));

  auto k1 = Symbol(1);
  auto k3 = Symbol(3);

  auto op_type = "Unpack";
  auto op_name = "unpack";
  auto attr_name = "axis";
  vector<SymbolShape> input_shapes;
  vector<SymbolShape> output_shapes;

  // 正常场景1: axis = 1 ,n=3, input shapes dims=4
  output_shapes = {SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1})};
  input_shapes = {SymbolShape({s0, k3, s2, s1})};
  EXPECT_RUN_UNPACK_TEST(input_shapes, 1, output_shapes, GRAPH_SUCCESS);

  // 正常场景2：axis = -1 ,n=3, input shapes dims=4
  output_shapes = {SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1}), SymbolShape({s0, s2, s1})};
  input_shapes = {SymbolShape({s0, s2, s1, k3})};
  EXPECT_RUN_UNPACK_TEST(input_shapes, -1, output_shapes, GRAPH_SUCCESS);

  // 正常场景3：axis = 1 ,n=1, input shapes dims=4
  output_shapes = {SymbolShape({s2, s1, s0})};
  input_shapes = {SymbolShape({s2, k1, s1, s0})};
  EXPECT_RUN_UNPACK_TEST(input_shapes, 1, output_shapes, GRAPH_SUCCESS);

  // 正常场景4：axis = 0 ,n=2, input shapes dims=1 拆分成标量场景
  output_shapes = {SymbolShape({}), SymbolShape({}), SymbolShape({})};
  input_shapes = {SymbolShape({k3})};
  EXPECT_RUN_UNPACK_TEST(input_shapes, 0, output_shapes, GRAPH_SUCCESS);

  // 异常场景1：输入axis超过dims范围  axis = 5 ,n=3, input shapes dims=3
  output_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0})};
  input_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_UNPACK_TEST(input_shapes, 5, output_shapes, PARAM_INVALID);

  // 异常场景2：输入axis超过dims范围负数越界  axis = -5 ,n=3, input shapes dims=3
  output_shapes = {SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0}), SymbolShape({s2, s1, s0})};
  input_shapes = {SymbolShape({s2, s1, s0})};
  EXPECT_RUN_UNPACK_TEST(input_shapes, -5, output_shapes, PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForExpandDims) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));

  auto kNeg1 = Symbol(-1);
  auto k0 = Symbol(0);
  auto k1 = Symbol(1);
  auto k2 = Symbol(2);
  auto k3 = Symbol(3);
  auto kNeg4 = Symbol(-4);

  auto op_type = "ExpandDims";
  auto op_name = "expandDims";

  // 正常场景1: axis = 0 ,input shape={3,2},output shape={1,3,2}
  vector input_shapes = {gert::SymbolShape({k3, k2})};
  vector data_depends = {gert::SymbolTensor({k1}, {k0})};
  vector output_shapes = {gert::SymbolShape({k1, k3, k2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS); // 设置范围满足，复用已有SLICE_TEST宏定义

  // 正常场景2: axis = -1 ,input shape={s0,s1,s2},output shape={s0,s1,s2,1}
  input_shapes = {gert::SymbolShape({s0, s1, s2})};
  data_depends = {gert::SymbolTensor({k1}, {kNeg1})};
  output_shapes = {gert::SymbolShape({s0, s1, s2, k1})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 正常场景3: 边界场景axis = -4 ,input shape={s0,s1,s2},output shape={1，s0,s1,s2}
  input_shapes = {gert::SymbolShape({s0, s1, s2})};
  data_depends = {gert::SymbolTensor({k1}, {kNeg4})};
  output_shapes = {gert::SymbolShape({k1, s0, s1, s2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_SUCCESS);

  // 异常场景1: axis数量超过一个 axis = {1,2} ,input shape={s0,s1,s2},output shape={s0,s1,s2,1}
  input_shapes = {gert::SymbolShape({s0, s1, s2})};
  data_depends = {gert::SymbolTensor({k2}, {k0, k1})};
  output_shapes = {gert::SymbolShape({s0, s1, s2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_FAILED);

  // 异常场景2: axis值超过范围 axis = {3} ,input shape={s0,s1,s2},output shape={s0,s1,s2,1}
  input_shapes = {gert::SymbolShape({s0, s1})};
  data_depends = {gert::SymbolTensor({k1}, {k3})};
  output_shapes = {gert::SymbolShape({s0, s1})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, GRAPH_FAILED);

  // test no symbol value
  input_shapes = {gert::SymbolShape({k3, k2})};
  data_depends = {gert::SymbolTensor({k1})};
  output_shapes = {gert::SymbolShape({k1, k3, k2})};
  EXPECT_EXPECT_RUN_COMMON_TEST(input_shapes, output_shapes, data_depends, UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForPad) {
  const auto func = GetInferFunc("Pad");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("Pad", "pad");
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  // 正常场景1： paddings 的shape是[n,2] n-input 的 dim number
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  auto paddings_shape = gert::SymbolShape({ge::Symbol(3), ge::Symbol(2)});
  auto paddings = std::vector<Expression>{Symbol(1), Symbol(2), Symbol(2), Symbol(1), Symbol(3), Symbol(3)};
  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                              .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                              .OutputNum(1)
                              .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);

  auto expect_shape =
      gert::SymbolShape({s0 + ge::Symbol(1) + ge::Symbol(2), s1 + ge::Symbol(2) + ge::Symbol(1),
                         s2 + ge::Symbol(3) + ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 异常场景1：paddings为空
  builder.Destroy();
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);

  // 异常场景2：paddings的shape的维度一不等于input的维度数
  builder.Destroy();
  paddings_shape = gert::SymbolShape({ge::Symbol(2), ge::Symbol(2)});
  paddings = std::vector<Expression>{Symbol(1), Symbol(2), Symbol(2), Symbol(1)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 异常场景3：paddings的shape的维度二不等于2
  builder.Destroy();
  paddings_shape = gert::SymbolShape({ge::Symbol(3), ge::Symbol(1)});
  paddings = std::vector<Expression>{Symbol(1), Symbol(2), Symbol(2)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // test no symbol value
  builder.Destroy();
  paddings_shape = gert::SymbolShape({ge::Symbol(3), ge::Symbol(1)});
  infer_context = builder.AppendInputSymbolTensor(input_shape)
      .AppendInputSymbolTensor(paddings_shape, true, nullptr)
      .OutputNum(1)
      .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
}

static void BuildStridedSliceInferContext(InferSymbolShapeContextTestBuilder &builder,
                                          const std::initializer_list<Expression> &x_values,
                                          const std::vector<Expression> &begin_values,
                                          const std::vector<Expression> &end_values,
                                          const std::vector<Expression> &strides_values, int64_t start_mask = 0,
                                          int64_t end_mask = 0, int64_t ellipsis_mask = 0, int64_t new_axis_mask = 0,
                                          int64_t shrink_axis_maks = 0) {
  auto op_desc = builder.GetOrCreateOpDescPtr();
  // 设置各种mask
  op_desc->AppendIrAttrName("start_mask");
  op_desc->AppendIrAttrName("end_mask");
  op_desc->AppendIrAttrName("ellipsis_mask");
  op_desc->AppendIrAttrName("new_axis_mask");
  op_desc->AppendIrAttrName("shrink_axis_mask");
  AttrUtils::SetInt(op_desc, "start_mask", start_mask);
  AttrUtils::SetInt(op_desc, "end_mask", end_mask);
  AttrUtils::SetInt(op_desc, "ellipsis_mask", ellipsis_mask);
  AttrUtils::SetInt(op_desc, "new_axis_mask", new_axis_mask);
  AttrUtils::SetInt(op_desc, "shrink_axis_mask", shrink_axis_maks);

  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));

  builder
      .AppendInputSymbolTensor(gert::SymbolShape(x_values))                                               // input_x
      .AppendInputSymbolTensor(gert::SymbolShape({Symbol(begin_values.size())}), true, &begin_values)     // begin
      .AppendInputSymbolTensor(gert::SymbolShape({Symbol(end_values.size())}), true, &end_values)         // end
      .AppendInputSymbolTensor(gert::SymbolShape({Symbol(strides_values.size())}), true, &strides_values) // strides
      .OutputNum(1);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForStridedSlice_1) {
  auto func = GetInferFunc("StridedSlice");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 5));

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  auto c4 = Symbol(4);
  auto c5 = Symbol(5);
  auto c6 = Symbol(6);
  auto c7 = Symbol(7);
  InferSymbolShapeContextTestBuilder builder("StridedSlice", "stridedslice");
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 5));
    // 稠密模式：begin end strides跟input长度相同
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1});
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);

    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1 - c1);
    ASSERT_EQ(out_shape->GetDim(2), s2 - c2);
    ASSERT_EQ(out_shape->GetDim(3), c0);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {"ExpectLe(s0, 5)", "ExpectLe(s1, 5)", "ExpectLe(s2, 5)",
                                                "ExpectLe(s3, 5)", "ExpectLe(s3, 3)", "ExpectLt(0, s0)",
                                                "ExpectLt(1, s1)", "ExpectLt(2, s2)", "ExpectLt(4, s4)",
                                                "ExpectLt(5, s4)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_guard_infos.size(), 3);
    const std::set<std::string> assert_guard = {"ExpectLe(2, s2)",
                                                "ExpectLe(1, s1)",
                                                "ExpectLe(0, s0)"};
    for (auto &iter : assert_guard_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto cb1 = Symbol(128);
    auto cb2 = Symbol(512);
    // end需要截取,需要除以strides,然后向上取整测试用例
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {c3, c1, c7, c4, c6}, {c0, c0, c0, c0, c0}, {c1, c1, c6, c2, c7},
                                  {c1, c1, c2, c3, c4}, 0, 0, 0, 0, 0);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(1), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(2), Symbol(3));
    ASSERT_EQ(out_shape->GetDim(3), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(4), Symbol(2));
    // 当input长度小于begin，但是有new_axis,begin_mask,end_mask,strides不为1
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {cb1, cb1}, {c0, c0, c0, c0}, {c0, c0, c0, c0}, {c7, c7, c7, c7}, 3, 3, 0,
                                  12, 0);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 4);
    ASSERT_EQ(out_shape->GetDim(0), Symbol(19));
    ASSERT_EQ(out_shape->GetDim(1), Symbol(19));
    ASSERT_EQ(out_shape->GetDim(2), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(3), Symbol(1));
    // 当input长度小于begin，但是有new_axis,begin_mask,end_mask,strides不为1
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {cb1, cb1}, {c0, c0, c0, c0}, {c0, c0, c0, c0}, {c3, c3, c3, c3}, 3, 3, 0,
                                  12, 0);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 4);
    ASSERT_EQ(out_shape->GetDim(0), Symbol(43));
    ASSERT_EQ(out_shape->GetDim(1), Symbol(43));
    ASSERT_EQ(out_shape->GetDim(2), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(3), Symbol(1));
    // 当input长度小于begin，但是有new_axis,begin_mask,end_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {cb1, cb1}, {c0, c0, c0, c0}, {c0, c0, c0, c0}, {c1, c1, c1, c1}, 3, 3, 0,
                                  12, 0);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);

    ASSERT_EQ(out_shape->GetDimNum(), 4);
    ASSERT_EQ(out_shape->GetDim(0), Symbol(128));
    ASSERT_EQ(out_shape->GetDim(1), Symbol(128));
    ASSERT_EQ(out_shape->GetDim(2), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(3), Symbol(1));
    // 当input长度小于begin，但是有new_axis,begin_mask,end_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {cb1, cb1}, {c0, c0, c0, c0}, {c0, c0, c0, c0}, {c1, c1, c1, c1}, 3, 3, 0,
                                  0b111001, 0);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);

    ASSERT_EQ(out_shape->GetDimNum(), 4);
    ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(1), Symbol(128));
    ASSERT_EQ(out_shape->GetDim(2), Symbol(0));
    ASSERT_EQ(out_shape->GetDim(3), Symbol(1));
    // 当input长度小于begin，但是有new_axis
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {cb1, cb2}, {c0, c0, c0}, {c0, c0, c0}, {c1, c1, c1}, 5, 5, 0, 2, 0);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);

    ASSERT_EQ(out_shape->GetDimNum(), 3);
    ASSERT_EQ(out_shape->GetDim(0), Symbol(128));
    ASSERT_EQ(out_shape->GetDim(1), Symbol(1));
    ASSERT_EQ(out_shape->GetDim(2), Symbol(512));
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 0);
    const std::vector<SymbolCheckInfo> assert_guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_guard_infos.size(), 0);
  }
  {
    // 稀疏模式：begin end strides小于input的长度
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 5));
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3}, {c5, c5, c5, c5}, {c1, c1, c1, c1});
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1 - c1);
    ASSERT_EQ(out_shape->GetDim(2), s2 - c2);
    ASSERT_EQ(out_shape->GetDim(3), c0);
    ASSERT_EQ(out_shape->GetDim(4), s4);
    // 稀疏模式下ellipsis_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2}, {c5, c5, c5}, {c1, c1, c1}, 0, 0,
                                  static_cast<int64_t>(0b0010));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), s3);
    ASSERT_EQ(out_shape->GetDim(4), c3);
    // begin_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1}, static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), c0);
    ASSERT_EQ(out_shape->GetDim(4), c5);
    // end_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1}, 0, static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1 - c1);
    ASSERT_EQ(out_shape->GetDim(2), s2 - c2);
    ASSERT_EQ(out_shape->GetDim(3), c0);
    ASSERT_EQ(out_shape->GetDim(4), s4 - c4);
    // ellipsis_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1}, 0, 0, static_cast<int64_t>(0b00100));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1 - c1);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), c0);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    // new_axis_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1}, 0, 0, 0, static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 8);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), c1);
    ASSERT_EQ(out_shape->GetDim(3), c0);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    ASSERT_EQ(out_shape->GetDim(5), s2);
    ASSERT_EQ(out_shape->GetDim(6), s3);
    ASSERT_EQ(out_shape->GetDim(7), s4);
    // shrink_axis_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1}, 0, 0, 0, 0, static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 2);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c0);
    // ellipsis_mask和new_axis_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3, c4}, {c5, c5, c5, c5, c5},
                                  {c1, c1, c1, c1, c1}, 0, 0, static_cast<int64_t>(0b00100),
                                  static_cast<int64_t>(0b01110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 7);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), s1);
    ASSERT_EQ(out_shape->GetDim(3), s2);
    ASSERT_EQ(out_shape->GetDim(4), s3);
    ASSERT_EQ(out_shape->GetDim(5), c1);
    ASSERT_EQ(out_shape->GetDim(6), c1);
    // ellipsis_mask和new_axis_mask
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3}, {c5, c5, c5, c5}, {c1, c1, c1, c1},
                                  0, 0, static_cast<int64_t>(0b0100), static_cast<int64_t>(0b1110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 7);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), s1);
    ASSERT_EQ(out_shape->GetDim(3), s2);
    ASSERT_EQ(out_shape->GetDim(4), s3);
    ASSERT_EQ(out_shape->GetDim(5), s4);
    ASSERT_EQ(out_shape->GetDim(6), c1);
    // 5个mask混合使用
    builder.Destroy();
    BuildStridedSliceInferContext(builder, {s0, s1, s2, s3, s4}, {c0, c1, c2, c3}, {c5, c5, c5, c5}, {c1, c1, c1, c1},
                                  static_cast<int64_t>(0b0010), static_cast<int64_t>(0b0010),
                                  static_cast<int64_t>(0b0100), static_cast<int64_t>(0b1101),
                                  static_cast<int64_t>(0b0111));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 6);
    ASSERT_EQ(out_shape->GetDim(0), c1);
    ASSERT_EQ(out_shape->GetDim(1), s1);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), s3);
    ASSERT_EQ(out_shape->GetDim(4), s4);
    ASSERT_EQ(out_shape->GetDim(5), c1);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 20);
    const std::set<std::string> expect_guard = {"ExpectLe(0, s4)", "ExpectLe(0, s3)", "ExpectLe(0, s2)",
                                                "ExpectLe(0, s1)", "ExpectLe(s0, 5)", "ExpectLe(s1, 3)",
                                                "ExpectLe(s1, 5)", "ExpectLe(s2, 5)", "ExpectLe(s3, 3)",
                                                "ExpectLe(s3, 5)", "ExpectLt(0, s0)", "ExpectLt(0, s2)",
                                                "ExpectLt(0, s3)", "ExpectLt(0, s4)", "ExpectLt(1, s0)",
                                                "ExpectLt(1, s1)", "ExpectLt(2, s2)", "ExpectLt(2, s4)",
                                                "ExpectLt(4, s4)", "ExpectLt(5, s4)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_guard_infos.size(), 8);
    const std::set<std::string> assert_guard = {"ExpectLe(4, s4)",
                                                "ExpectLe(2, s2)",
                                                "ExpectLe(1, s1)",
                                                "ExpectLe(0, s0)",
                                                "ExpectLe(0, s1)",
                                                "ExpectLe(0, s2)",
                                                "ExpectLe(0, s3)",
                                                "ExpectLe(0, s4)"};
    for (auto &iter : assert_guard_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
}
TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForStridedSlicewithBeginIsNull) {
  auto func = GetInferFunc("StridedSlice");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("StridedSlice", "stridedslice");
  auto op_desc = builder.GetOrCreateOpDescPtr();
  // 设置各种mask
  op_desc->AppendIrAttrName("start_mask");
  op_desc->AppendIrAttrName("end_mask");
  op_desc->AppendIrAttrName("ellipsis_mask");
  op_desc->AppendIrAttrName("new_axis_mask");
  op_desc->AppendIrAttrName("shrink_axis_mask");
  AttrUtils::SetInt(op_desc, "start_mask", 0);
  AttrUtils::SetInt(op_desc, "end_mask", 0);
  AttrUtils::SetInt(op_desc, "ellipsis_mask", 0);
  AttrUtils::SetInt(op_desc, "new_axis_mask", 0);
  AttrUtils::SetInt(op_desc, "shrink_axis_mask", 0);
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));

  auto c1 = Symbol(1);
  const std::vector<Expression> &end_values{c1};
  const std::vector<Expression> &strides_values{c1};
  builder
      .AppendInputSymbolTensor(gert::SymbolShape({c1}))                                               // input_x
      .AppendInputSymbolTensor(gert::SymbolShape({Symbol(0)}), false)     // begin
      .AppendInputSymbolTensor(gert::SymbolShape({Symbol(1)}), true, &end_values)         // end
      .AppendInputSymbolTensor(gert::SymbolShape({Symbol(1)}), true, &strides_values) // strides
      .OutputNum(1);
  auto infer_context = builder.Build();
  ASSERT_EQ(func.first(infer_context), UNSUPPORTED);
}
void EXPECT_BatchMatMulV2TestCommon(const gert::SymbolShape &x1, const gert::SymbolShape &x2, bool adj_x1, bool adj_x2,
                                    const gert::SymbolShape &expect_out, graphStatus expect_status) {
  auto func = GetInferFunc("BatchMatMulV2");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("BatchMatMulV2", "batchmatmulv2");

  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("adj_x1");
  op_desc->AppendIrAttrName("adj_x2");
  AttrUtils::SetBool(op_desc, "adj_x1", adj_x1);
  AttrUtils::SetBool(op_desc, "adj_x2", adj_x2);

  auto infer_context = builder.AppendInputSymbolTensor(x1).AppendInputSymbolTensor(x2).OutputNum(1).Build();
  ASSERT_TRUE(func.first(infer_context) == expect_status);
  if (expect_status != GRAPH_SUCCESS) {
    return;
  }
  auto out_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_TRUE(*out_shape == expect_out);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForBatchMatMulV2) {
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 5));

    auto c1 = Symbol(1);
    // x1, x2仅有1维
    EXPECT_BatchMatMulV2TestCommon(gert::SymbolShape({s0}), gert::SymbolShape({s0}), false, false,
                                   gert::SymbolShape({c1, c1}), GRAPH_SUCCESS);

    // 两个张量维度不同
    EXPECT_BatchMatMulV2TestCommon(gert::SymbolShape({s0, s1, s2, s3, s4}), gert::SymbolShape({s2, s4, s3}), false,
                                   false, gert::SymbolShape({s0, s1, s2, s3, s3}), GRAPH_SUCCESS);

    EXPECT_BatchMatMulV2TestCommon(gert::SymbolShape({s2, s4, s3}), gert::SymbolShape({s0, s1, s2, s3, s4}), false,
                                   false, gert::SymbolShape({s0, s1, s2, s4, s4}), GRAPH_SUCCESS);

    EXPECT_BatchMatMulV2TestCommon(gert::SymbolShape({c1, s4, s3}), gert::SymbolShape({s0, s1, s2, s3, s4}), false,
                                   false, gert::SymbolShape({s0, s1, s2, s4, s4}), GRAPH_SUCCESS);

    // adj为true
    EXPECT_BatchMatMulV2TestCommon(gert::SymbolShape({s0, s1, s2, s3, s4}), gert::SymbolShape({s2, s4, s3}), true, true,
                                   gert::SymbolShape({s0, s1, s2, s4, s4}), GRAPH_SUCCESS);
  }
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 5));
    auto kThree = ge::Symbol(3);

    EXPECT_BatchMatMulV2TestCommon(gert::SymbolShape({s0, kThree, s1}), gert::SymbolShape({s2, s3, s4}), false, false,
                                   gert::SymbolShape({s2, kThree, s4}), GRAPH_SUCCESS);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 3);
    const std::set<std::string> expect_guard = {"ExpectEq(1, s0)", "ExpectNe(1, s2)", "ExpectNe(s0, s2)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 1);
    const std::set<std::string> assert_guard = {"ExpectEq(s1, s3)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeTestCommon) {
  int64_t value = 0;
  gert::SymbolTensor tensor;
  tensor.MutableSymbolicValue()->push_back(ge::Symbol(1));
  EXPECT_EQ(ge::SymbolicInferUtil::GetConstInt(&tensor, DT_INT64, value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ge::SymbolicInferUtil::GetConstInt(&tensor, DT_INT16, value), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForMatmulV2) {
  auto func = GetInferFunc("MatMulV2");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("MatMulV2", "matMulV2");

  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("transpose_x1");
  op_desc->AppendIrAttrName("transpose_x2");
  AttrUtils::SetBool(op_desc, "transpose_x1", true);
  AttrUtils::SetBool(op_desc, "transpose_x2", true);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));

  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                              .AppendInputSymbolTensor(gert::SymbolShape({s3, s0}))
                              .OutputNum(1)
                              .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);

  auto output_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_EQ(output_shape->GetDimNum(), 2);
  ASSERT_EQ(output_shape->GetDim(0), s1);
  ASSERT_EQ(output_shape->GetDim(1), s3);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConcatV2) {
  ConcatV2Test("ConcatV2");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConcat) {
  ConcatV2Test("Concat");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForPadD) {
  const auto func = GetInferFunc("PadD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("PadD", "padd");
  // 正常场景1： paddings 的shape是[n,2] n-input 的 dim number
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  std::vector<vector<int64_t>> paddings = {{1, 2}, {2, 1}, {3, 3}};
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("paddings");
  AttrUtils::SetListListInt(op_descPtr, "paddings", paddings);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                              .OutputNum(1)
                              .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape =
      gert::SymbolShape({s0 + ge::Symbol(1) + ge::Symbol(2), s1 + ge::Symbol(2) + ge::Symbol(1),
                         s2 + ge::Symbol(3) + ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());
  //
  // 异常场景1：paddings为空
  builder.Destroy();
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 异常场景2：paddings的shape的维度一不等于input的维度数
  builder.Destroy();
  paddings = {{1, 2}, {2, 1}};
  op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("paddings");
  AttrUtils::SetListListInt(op_descPtr, "paddings", paddings);
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 异常场景3：paddings的shape的维度二不等于2
  builder.Destroy();
  paddings = {{1}, {2}, {3}};
  op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("paddings");
  AttrUtils::SetListListInt(op_descPtr, "paddings", paddings);
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSplitD) {
  //INPUT
  //ATTR split_dim
  //ATTR num_split
  auto func = GetInferFunc("SplitD");
  ASSERT_TRUE(func.first != nullptr);
  // 1. input_shape是null
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));

  InferSymbolShapeContextTestBuilder builder("SplitD", "splitd");
  auto infer_context = builder.Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
  // 2. 不存在attr
  builder.Destroy();
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 3. num_split不是正数
  builder.Destroy();
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 1);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", -1);
  //std::vector<ge::Expression> symbol_value = {Symbol(1)};
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 4. split_num不合法
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 3);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 5. 正常split_dim = 1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 1);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto output_shape0 = infer_context->GetOutputSymbolShape(0);
  auto output_shape1 = infer_context->GetOutputSymbolShape(1);
  auto in_shape = infer_context->GetInputSymbolShape(0);
  ASSERT_EQ(in_shape->GetDimNum(), output_shape0->GetDimNum());
  ASSERT_EQ(in_shape->GetDimNum(), output_shape1->GetDimNum());

  // dim 1
  ASSERT_EQ(in_shape->GetDim(0), output_shape0->GetDim(0));
  ASSERT_EQ(in_shape->GetDim(0), output_shape1->GetDim(0));
  // dim 2
  auto dim2_expect = in_shape->GetDim(1) / ge::Symbol(2);
  ASSERT_EQ(dim2_expect, output_shape0->GetDim(1));
  ASSERT_EQ(dim2_expect, output_shape1->GetDim(1));

  // 6. 正常split_dim = -2
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", -2);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 2);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(2)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  output_shape0 = infer_context->GetOutputSymbolShape(0);
  output_shape1 = infer_context->GetOutputSymbolShape(1);
  in_shape = infer_context->GetInputSymbolShape(0);
  ASSERT_EQ(in_shape->GetDimNum(), output_shape0->GetDimNum());
  ASSERT_EQ(in_shape->GetDimNum(), output_shape1->GetDimNum());

  // dim 1
  auto dim1_expect = in_shape->GetDim(0) / ge::Symbol(2);
  ASSERT_EQ(dim1_expect, output_shape0->GetDim(0));
  ASSERT_EQ(dim1_expect, output_shape1->GetDim(0));
  // dim 2
  ASSERT_EQ(in_shape->GetDim(1), output_shape0->GetDim(1));
  ASSERT_EQ(in_shape->GetDim(1), output_shape1->GetDim(1));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSplitVD) {
  // input: data
  // attr: size_splits
  // attr: split_dim
  // attr: num_split
  auto func = GetInferFunc("SplitVD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));

  InferSymbolShapeContextTestBuilder builder("SplitVD", "splitvd");
  // 1. num_split校验失败
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("size_splits");
  AttrUtils::SetListInt(op_desc, "size_splits", {1, -1, 2});
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 0);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", -1);
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                              .OutputNum(3)
                              .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 2. split_dim 校验失败
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("size_splits");
  AttrUtils::SetListInt(op_desc, "size_splits", {1, -1, 2});
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 3);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 3);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(3)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 3. dynamic_value_num校验失败
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("size_splits");
  AttrUtils::SetListInt(op_desc, "size_splits", {1, -1, -1});
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 1);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 3);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(3)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 4. 正常情况
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("size_splits");
  AttrUtils::SetListInt(op_desc, "size_splits", {1, -1, 2});
  op_desc->AppendIrAttrName("split_dim");
  AttrUtils::SetInt(op_desc, "split_dim", 1);
  op_desc->AppendIrAttrName("num_split");
  AttrUtils::SetInt(op_desc, "num_split", 3);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                         .OutputNum(3)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto output_shape0 = infer_context->GetOutputSymbolShape(0);
  auto output_shape1 = infer_context->GetOutputSymbolShape(1);
  auto output_shape2 = infer_context->GetOutputSymbolShape(2);
  auto in_shape = infer_context->GetInputSymbolShape(0);
  ASSERT_EQ(in_shape->GetDimNum(), output_shape0->GetDimNum());
  ASSERT_EQ(in_shape->GetDimNum(), output_shape1->GetDimNum());
  ASSERT_EQ(in_shape->GetDimNum(), output_shape2->GetDimNum());

  // dim 1
  ASSERT_EQ(in_shape->GetDim(0), output_shape0->GetDim(0));
  ASSERT_EQ(in_shape->GetDim(0), output_shape1->GetDim(0));
  ASSERT_EQ(in_shape->GetDim(0), output_shape2->GetDim(0));
  // dim 2
  ASSERT_EQ(Symbol(1), output_shape0->GetDim(1));
  ASSERT_EQ(in_shape->GetDim(1) - Symbol(3), output_shape1->GetDim(1));
  ASSERT_EQ(Symbol(2), output_shape2->GetDim(1));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLayerNorm) {
  const auto func = GetInferFunc("LayerNorm");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("LayerNorm", "layernorm");
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("begin_norm_axis");
  AttrUtils::SetInt(op_descPtr, "begin_norm_axis", -1);
  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                              .OutputNum(3)
                              .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape1 = gert::SymbolShape({s0, s1, s2});
  auto expect_shape2 = gert::SymbolShape({s0, s1, Symbol(1)});
  auto expect_shape3 = gert::SymbolShape({s0, s1, Symbol(1)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), expect_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(2)->GetDims(), expect_shape3.GetDims());
  // 异常场景：axis,不在范围内
  builder.Destroy();
  op_descPtr = builder.GetOrCreateOpDescPtr();
  op_descPtr->AppendIrAttrName("begin_norm_axis");
  AttrUtils::SetInt(op_descPtr, "begin_norm_axis", -5);
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                         .OutputNum(1)
                         .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
}
static void BuildStridedSliceDInferContext(InferSymbolShapeContextTestBuilder &builder,
                                           const std::initializer_list<Expression> &x_values,
                                           const std::vector<int64_t> &begin_values,
                                           const std::vector<int64_t> &end_values,
                                           const std::vector<int64_t> &strides_values, int64_t start_mask = 0,
                                           int64_t end_mask = 0, int64_t ellipsis_mask = 0, int64_t new_axis_mask = 0,
                                           int64_t shrink_axis_maks = 0) {
  auto op_desc = builder.GetOrCreateOpDescPtr();
  // 设置begin end stride
  op_desc->AppendIrAttrName("begin");
  op_desc->AppendIrAttrName("end");
  op_desc->AppendIrAttrName("strides");
  AttrUtils::SetListInt(op_desc, "begin", begin_values);
  AttrUtils::SetListInt(op_desc, "end", end_values);
  AttrUtils::SetListInt(op_desc, "strides", strides_values);
  // 设置各种mask
  op_desc->AppendIrAttrName("start_mask");
  op_desc->AppendIrAttrName("end_mask");
  op_desc->AppendIrAttrName("ellipsis_mask");
  op_desc->AppendIrAttrName("new_axis_mask");
  op_desc->AppendIrAttrName("shrink_axis_mask");
  AttrUtils::SetInt(op_desc, "start_mask", start_mask);
  AttrUtils::SetInt(op_desc, "end_mask", end_mask);
  AttrUtils::SetInt(op_desc, "ellipsis_mask", ellipsis_mask);
  AttrUtils::SetInt(op_desc, "new_axis_mask", new_axis_mask);
  AttrUtils::SetInt(op_desc, "shrink_axis_mask", shrink_axis_maks);

  op_desc->AddInputDesc(GeTensorDesc());

  builder
      .AppendInputSymbolTensor(gert::SymbolShape(x_values)) // input_x
      .OutputNum(1);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForStridedSliceD) {
  auto func = GetInferFunc("StridedSliceD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(8, MakeShared<InputShapeSource>(0, 4));
  auto s4 = shape_env.CreateSymbol(9, MakeShared<InputShapeSource>(0, 5));

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  auto c4 = Symbol(4);
  auto c5 = Symbol(5);
  auto c6 = Symbol(6);
  InferSymbolShapeContextTestBuilder builder("StridedSliceD", "stridedsliceD");
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(8, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(9, MakeShared<InputShapeSource>(0, 5));
    // 稠密模式：begin end strides跟input长度相同
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1});
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c4);
    ASSERT_EQ(out_shape->GetDim(2), c3);
    ASSERT_EQ(out_shape->GetDim(3), c2);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    // new_axis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c3, c4, c5, c6}, {2, 1, 2, 3}, {2, 4, 5, 5}, {1, 1, 1, 1}, 0, 0, 0, 0);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 4);
    ASSERT_EQ(out_shape->GetDim(0), c0);
    ASSERT_EQ(out_shape->GetDim(1), c3);
    ASSERT_EQ(out_shape->GetDim(2), c3);
    ASSERT_EQ(out_shape->GetDim(3), c2);
    // new_axis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c1, c2, c3, c4, c5}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 0,
                                   0, 0, static_cast<int64_t>(0b11010));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 8);
    ASSERT_EQ(out_shape->GetDim(0), c1);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), c0);
    ASSERT_EQ(out_shape->GetDim(3), c1);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    ASSERT_EQ(out_shape->GetDim(5), c3);
    ASSERT_EQ(out_shape->GetDim(6), c4);
    ASSERT_EQ(out_shape->GetDim(7), c5);
    // 稠密模式：begin end strides跟input长度相同
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c3, c2, c3}, {1, 0, 0}, {2, 2, 2}, {1, 1, 1});
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 3);
    ASSERT_EQ(out_shape->GetDim(0), c1);
    ASSERT_EQ(out_shape->GetDim(1), c2);
    ASSERT_EQ(out_shape->GetDim(2), c2);
    // 稀疏模式：begin end strides小于input的长度
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c1, c2, c3, c4, c5}, {0, 1, 2, 3}, {5, 5, 5, 5}, {1, 1, 1, 1});
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), c1);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), c1);
    ASSERT_EQ(out_shape->GetDim(3), c1);
    ASSERT_EQ(out_shape->GetDim(4), c5);
    // begin_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c1, c2, c3, c4, c5}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1},
                                   static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), c1);
    ASSERT_EQ(out_shape->GetDim(1), c2);
    ASSERT_EQ(out_shape->GetDim(2), c3);
    ASSERT_EQ(out_shape->GetDim(3), c1);
    ASSERT_EQ(out_shape->GetDim(4), c5);
    // end_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 0,
                                   static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1 - c1);
    ASSERT_EQ(out_shape->GetDim(2), s2 - c2);
    ASSERT_EQ(out_shape->GetDim(3), c2);
    ASSERT_EQ(out_shape->GetDim(4), s4 - c4);
    // ellipsis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 0,
                                   0, static_cast<int64_t>(0b00100));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c4);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), c2);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    // new_axis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 0,
                                   0, 0, static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 8);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), c1);
    ASSERT_EQ(out_shape->GetDim(3), c2);
    ASSERT_EQ(out_shape->GetDim(4), c1);
    ASSERT_EQ(out_shape->GetDim(5), s2);
    ASSERT_EQ(out_shape->GetDim(6), s3);
    ASSERT_EQ(out_shape->GetDim(7), s4);
    // shrink_axis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 0,
                                   0, 0, 0, static_cast<int64_t>(0b10110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 2);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c2);
    // 稀疏模式下ellipsis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2}, {5, 5, 5}, {1, 1, 1}, 0, 0,
                                   static_cast<int64_t>(0b0010));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 5);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s1);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), s3);
    ASSERT_EQ(out_shape->GetDim(4), c3);
    // ellipsis_mask和new_axis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3, 4}, {5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 0,
                                   0, static_cast<int64_t>(0b00100), static_cast<int64_t>(0b01110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 7);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), s1);
    ASSERT_EQ(out_shape->GetDim(3), s2);
    ASSERT_EQ(out_shape->GetDim(4), s3);
    ASSERT_EQ(out_shape->GetDim(5), c1);
    ASSERT_EQ(out_shape->GetDim(6), c1);
    // ellipsis_mask和new_axis_mask
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3}, {5, 5, 5, 5}, {1, 1, 1, 1}, 0, 0,
                                   static_cast<int64_t>(0b0100), static_cast<int64_t>(0b1110));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 7);
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), s1);
    ASSERT_EQ(out_shape->GetDim(3), s2);
    ASSERT_EQ(out_shape->GetDim(4), s3);
    ASSERT_EQ(out_shape->GetDim(5), s4);
    ASSERT_EQ(out_shape->GetDim(6), c1);
    // 5个mask混合使用
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {s0, s1, s2, s3, s4}, {0, 1, 2, 3}, {5, 5, 5, 5}, {1, 1, 1, 1},
                                   static_cast<int64_t>(0b0010), static_cast<int64_t>(0b0010),
                                   static_cast<int64_t>(0b0100), static_cast<int64_t>(0b1101),
                                   static_cast<int64_t>(0b0111));
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 6);
    ASSERT_EQ(out_shape->GetDim(0), c1);
    ASSERT_EQ(out_shape->GetDim(1), s1);
    ASSERT_EQ(out_shape->GetDim(2), s2);
    ASSERT_EQ(out_shape->GetDim(3), s3);
    ASSERT_EQ(out_shape->GetDim(4), s4);
    ASSERT_EQ(out_shape->GetDim(5), c1);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 20);
    const std::set<std::string> expect_guard = {"ExpectLe(0, s4)", "ExpectLe(0, s3)", "ExpectLe(0, s2)",
                                                "ExpectLe(0, s1)", "ExpectLe(s0, 5)", "ExpectLt(0, s0)",
                                                "ExpectLt(0, s2)", "ExpectLt(0, s3)", "ExpectLt(0, s4)",
                                                "ExpectLt(1, s0)", "ExpectLt(1, s1)", "ExpectLt(2, s2)",
                                                "ExpectLt(2, s4)", "ExpectLt(3, s1)", "ExpectLt(3, s3)",
                                                "ExpectLt(4, s4)", "ExpectLt(5, s1)", "ExpectLt(5, s2)",
                                                "ExpectLt(5, s3)", "ExpectLt(5, s4)"};
    for (auto &iter : guard_infos) {
      EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_guard_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_guard_infos.size(), 8);
    const std::set<std::string> assert_guard = {"ExpectLe(4, s4)",
                                                "ExpectLe(2, s2)",
                                                "ExpectLe(1, s1)",
                                                "ExpectLe(0, s0)",
                                                "ExpectLe(0, s1)",
                                                "ExpectLe(0, s2)",
                                                "ExpectLe(0, s3)",
                                                "ExpectLe(0, s4)"};
    for (auto &iter : assert_guard_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    // 基础负数场景
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c3, c4, c5}, {0, 0, 0}, {-1, -1, -1}, {1, 1, 1});
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 3);
    ASSERT_EQ(out_shape->GetDim(0), c2);
    ASSERT_EQ(out_shape->GetDim(1), c3);
    ASSERT_EQ(out_shape->GetDim(2), c4);
    // 负步长
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c3, c2, c3}, {2, 0, 0}, {0, 2, 3}, {-1, 1, 1});
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 3);
    ASSERT_EQ(out_shape->GetDim(0), c2);
    ASSERT_EQ(out_shape->GetDim(1), c2);
    ASSERT_EQ(out_shape->GetDim(2), c3);
    // 负步长和负索引
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c3, c2, c3}, {-1, -1, 0}, {0, 0, 3}, {-1, -1, 1});
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 3);
    ASSERT_EQ(out_shape->GetDim(0), c2);
    ASSERT_EQ(out_shape->GetDim(1), c1);
    ASSERT_EQ(out_shape->GetDim(2), c3);
    // 维度反向
    builder.Destroy();
    BuildStridedSliceDInferContext(builder, {c3, c2, c3}, {0, 1, 0}, {3, -1, 3}, {1, -1, 1});
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), 3);
    ASSERT_EQ(out_shape->GetDim(0), c3);
    ASSERT_EQ(out_shape->GetDim(1), c0);
    ASSERT_EQ(out_shape->GetDim(2), c3);
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForClipByValue) {
  using namespace gert;
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
  auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 5));
  auto s5 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 6));

  auto kOne = ge::Symbol(1);
  auto kTwo = ge::Symbol(2);
  auto k3 = ge::Symbol(3);
  auto k4 = ge::Symbol(4);
  auto k5 = ge::Symbol(5);
  auto op_type = "ClipByValue";
  auto op_name = "clipByValue";
  vector<SymbolShape> input_shapes;
  vector<SymbolShape> output_shapes;

  auto func = GetInferFunc("ClipByValue");
  ASSERT_TRUE(func.first != nullptr);
  // min和max为标量
  RUN_ClipByValue_TEST(gert::SymbolShape({s3, s4}), gert::SymbolShape({}), gert::SymbolShape({}),
                       gert::SymbolShape({s3, s4}), ge::GRAPH_SUCCESS);
  //max为标量
  RUN_ClipByValue_TEST(gert::SymbolShape({s3, s4}), gert::SymbolShape({}), gert::SymbolShape({}),
                       gert::SymbolShape({s3, s4}), ge::GRAPH_SUCCESS);
  //max广播至{s2,s3}
  RUN_ClipByValue_TEST(gert::SymbolShape({s2, s3}), gert::SymbolShape({}), gert::SymbolShape({s3}),
                       gert::SymbolShape({s2, s3}), ge::GRAPH_SUCCESS);
  //广播至{s2,s3}
  RUN_ClipByValue_TEST(gert::SymbolShape({s2, s3}), gert::SymbolShape({s2,kOne}), gert::SymbolShape({s3}),
                       gert::SymbolShape({s2, s3}), ge::GRAPH_SUCCESS);
  //全标量
  RUN_ClipByValue_TEST(gert::SymbolShape({}), gert::SymbolShape({}), gert::SymbolShape({}),
                       gert::SymbolShape({}), ge::GRAPH_SUCCESS);
  //拓展维度
  RUN_ClipByValue_TEST(gert::SymbolShape({k5,kOne,k3}), gert::SymbolShape({k5,k3}), gert::SymbolShape({kOne,k3}),
                       gert::SymbolShape({k5,k5,k3}), ge::GRAPH_SUCCESS);
  //不同维度广播至同一目标形状
  RUN_ClipByValue_TEST(gert::SymbolShape({s2,s3,s4}), gert::SymbolShape({s2,kOne,kOne}),
                       gert::SymbolShape({kOne,s3,s4}),
                       gert::SymbolShape({s2,s3,s4}), ge::GRAPH_SUCCESS);
  //异常用例
  RUN_ClipByValue_TEST(gert::SymbolShape({k3,k4}), gert::SymbolShape({k5,k5}), gert::SymbolShape({}),
                       gert::SymbolShape({k3,k4}), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAddN_1) {
  auto func = GetInferFunc("AddN");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("AddN", "addN");

  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", 3);
  op_desc->AddDynamicInputDesc("x", 3, true);
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({Symbol(5), Symbol(2), Symbol(3), Symbol(4)}))
                              .AppendInputSymbolTensor(gert::SymbolShape({Symbol(1), Symbol(2), Symbol(1), Symbol(1)}))
                              .AppendInputSymbolTensor(gert::SymbolShape({Symbol(2), Symbol(3), Symbol(1)}))
                              .OutputNum(1)
                              .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto output_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_EQ(output_shape->GetDimNum(), 4);
  ASSERT_EQ(output_shape->GetDim(0), Symbol(5));
  ASSERT_EQ(output_shape->GetDim(1), Symbol(2));
  ASSERT_EQ(output_shape->GetDim(2), Symbol(3));
  ASSERT_EQ(output_shape->GetDim(3), Symbol(4));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAddN_2) {
  auto func = GetInferFunc("AddN");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("AddN", "addN");

  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("n1");
  AttrUtils::SetInt(op_desc, "n1", 3);
  op_desc->AddDynamicInputDesc("x", 3, true);
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({Symbol(2), Symbol(3)}))
                              .AppendInputSymbolTensor(gert::SymbolShape({Symbol(2), Symbol(3)}))
                              .AppendInputSymbolTensor(gert::SymbolShape({Symbol(2), Symbol(3)}))
                              .OutputNum(1)
                              .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);

  auto output_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_EQ(output_shape->GetDimNum(), 2);
  ASSERT_EQ(output_shape->GetDim(0), Symbol(2));
  ASSERT_EQ(output_shape->GetDim(1), Symbol(3));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForBiasAddGrad) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 2));
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  InferSymbolShapeContextTestBuilder builder("BiasAddGrad", "BiasAddGrad");
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("data_format");
  AttrUtils::SetStr(op_desc, "data_format", "NHWC");
  auto infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  auto func = GetInferFunc("BiasAddGrad");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({s2}).GetDims());

  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("data_format");
  AttrUtils::SetStr(op_desc, "data_format", "TEST");
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_NE(func.first(infer_context), ge::GRAPH_SUCCESS);

  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("data_format");
  AttrUtils::SetStr(op_desc, "data_format", "NCHW");
  infer_context = builder.AppendInputSymbolTensor(input_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({s0}).GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForBroadcastTo) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto input_shape = gert::SymbolShape({s1});
  InferSymbolShapeContextTestBuilder builder("BroadcastTo", "BroadcastTo");
  std::vector<ge::Expression> symbol_value = {s0, s1};
  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                              .AppendInputSymbolTensor({s0}, true, &symbol_value)
                              .OutputNum(1).Build();
  auto func = GetInferFunc("BroadcastTo");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), symbol_value);

  // 异常场景1 shape小于input_shape
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1});
  symbol_value = {s1};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor({Symbol(1)}, true, &symbol_value)
                      .OutputNum(1)
                      .Build();
  ASSERT_NE(func.first(infer_context), ge::GRAPH_SUCCESS);

  // 异常场景2 input_shape无法广播到shape
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1});
  symbol_value = {s1, s0};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor({s0}, true, &symbol_value)
                      .OutputNum(1)
                      .Build();
  ASSERT_NE(func.first(infer_context), ge::GRAPH_SUCCESS);

  // shape_value中包含-1
  builder.Destroy();
  input_shape = gert::SymbolShape({s0, s1});
  symbol_value = {Symbol(-1), Symbol(-1), s1};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor({Symbol(2)}, true, &symbol_value)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto out_shape = infer_context->GetOutputSymbolShape(0);
  ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
  ASSERT_EQ(out_shape->GetDim(1), s0);
  ASSERT_EQ(out_shape->GetDim(2), s1);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConfusionSoftmaxGrad) {
  TestForBroadCast("ConfusionSoftmaxGrad", "ConfusionSoftmaxGrad");
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLayerNormBetaGammaBackpropV2) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 2));
  auto kone = ge::Symbol(1);
  auto ktwo = ge::Symbol(2);
  auto dy_shape = gert::SymbolShape({s0, s1, s2});
  auto res_for_gamma_shape = gert::SymbolShape({s0, s1, s2});
  InferSymbolShapeContextTestBuilder builder("LayerNormBetaGammaBackpropV2", "LayerNormBetaGammaBackpropV2");

  vector<int64_t> gamma_shape = {-1, 0, -1};
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("gamma_shape");
  AttrUtils::SetListInt(op_desc, "gamma_shape", gamma_shape);
  auto infer_context = builder.AppendInputSymbolTensor(dy_shape)
                              .AppendInputSymbolTensor(res_for_gamma_shape)
                              .OutputNum(2).Build();

  auto func = GetInferFunc("LayerNormBetaGammaBackpropV2");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({kone, s1, kone}).GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), gert::SymbolShape({kone, s1, kone}).GetDims());

  builder.Destroy();
  gamma_shape = {2};
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("gamma_shape");
  AttrUtils::SetListInt(op_desc, "gamma_shape", gamma_shape);
  infer_context = builder.AppendInputSymbolTensor(dy_shape)
                         .AppendInputSymbolTensor(res_for_gamma_shape)
                         .OutputNum(2).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({ktwo}).GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), gert::SymbolShape({ktwo}).GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLayerNormV3) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 2));
  auto kone = ge::Symbol(1);
  auto x_shape = gert::SymbolShape({s0, s1, s2});
  auto gamma_shape = gert::SymbolShape({s0, s1, s2});
  auto beta_shape = gert::SymbolShape({s0, s1, s2});
  InferSymbolShapeContextTestBuilder builder("LayerNormV3", "LayerNormV3");

  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("begin_norm_axis");
  AttrUtils::SetInt(op_desc, "begin_norm_axis", 2);
  auto infer_context = builder.AppendInputSymbolTensor(x_shape)
                              .AppendInputSymbolTensor(gamma_shape)
                              .AppendInputSymbolTensor(beta_shape)
                              .OutputNum(3).Build();

  auto func = GetInferFunc("LayerNormV3");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), x_shape.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), gert::SymbolShape({s0, s1, kone}).GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(2)->GetDims(), gert::SymbolShape({s0, s1, kone}).GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLayerNormXBackpropV3) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto input_shape0 = gert::SymbolShape({s0, s1});
  auto input_shape1 = gert::SymbolShape({s0, s1});
  auto input_shape2 = gert::SymbolShape({s0, s1});
  auto input_shape3 = gert::SymbolShape({s0, s1});
  auto input_shape4 = gert::SymbolShape({s0, s1});
  InferSymbolShapeContextTestBuilder builder("LayerNormXBackpropV3", "LayerNormXBackpropV3");
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0)
                              .AppendInputSymbolTensor(input_shape1)
                              .AppendInputSymbolTensor(input_shape2)
                              .AppendInputSymbolTensor(input_shape3)
                              .AppendInputSymbolTensor(input_shape4)
                              .OutputNum(2).Build();
  auto func = GetInferFunc("LayerNormXBackpropV3");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), input_shape1.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForLayerNormXBackpropV2) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto input_shape0 = gert::SymbolShape({s0, s1});
  auto input_shape1 = gert::SymbolShape({s0, s1});
  auto input_shape2 = gert::SymbolShape({s0, s1});
  auto input_shape3 = gert::SymbolShape({s0, s1});
  auto input_shape4 = gert::SymbolShape({s0, s1});
  InferSymbolShapeContextTestBuilder builder("LayerNormXBackpropV2", "LayerNormXBackpropV2");
  auto infer_context = builder.AppendInputSymbolTensor(input_shape0)
                              .AppendInputSymbolTensor(input_shape1)
                              .AppendInputSymbolTensor(input_shape2)
                              .AppendInputSymbolTensor(input_shape3)
                              .AppendInputSymbolTensor(input_shape4)
                              .OutputNum(2).Build();
  auto func = GetInferFunc("LayerNormXBackpropV2");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), input_shape1.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConv2DBackpropInputD) {
  auto func = GetInferFunc("Conv2DBackpropInputD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 4));
  InferSymbolShapeContextTestBuilder builder("Conv2DBackpropInputD", "conv2DBackpropInputD");
  // 1. 异常场景1：Attr input原始形状输入维度不为4
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("input_size");
  std::vector<int64_t> input_list0 = {1, 2, 3};
  AttrUtils::SetListInt(op_desc, "input_size", input_list0);
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                           .AppendInputSymbolTensor(gert::SymbolShape({s2}))
                           .OutputNum(1)
                           .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 2. 正常场景1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("input_size");
  std::vector<int64_t> input_list1 = {1, 2, 3, 4};
  AttrUtils::SetListInt(op_desc, "input_size", input_list1);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2, s3}))
                      .AppendInputSymbolTensor(gert::SymbolShape({s2}))
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
  // dim1
  ASSERT_EQ(out_shape->GetDim(0), Symbol(input_list1[0]));
  // dim2
  ASSERT_EQ(out_shape->GetDim(1), Symbol(input_list1[1]));
  // dim3
  ASSERT_EQ(out_shape->GetDim(2), Symbol(input_list1[2]));
  // dim4
  ASSERT_EQ(out_shape->GetDim(3), Symbol(input_list1[3]));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConv2DBackpropFilterD) {
  auto func = GetInferFunc("Conv2DBackpropFilterD");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 4));
  InferSymbolShapeContextTestBuilder builder("Conv2DBackpropFilterD", "conv2DBackpropFilterD");
  // 1. 异常场景1：Attr input原始形状输入维度不为4
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("input_size");
  std::vector<int64_t> input_list0 = {1, 2, 3};
  AttrUtils::SetListInt(op_desc, "input_size", input_list0);
  auto infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1}))
                           .AppendInputSymbolTensor(gert::SymbolShape({s2}))
                           .OutputNum(1)
                           .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  // 2. 正常场景1
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("input_size");
  std::vector<int64_t> input_list1 = {1, 2, 3, 4};
  AttrUtils::SetListInt(op_desc, "input_size", input_list1);
  infer_context = builder.AppendInputSymbolTensor(gert::SymbolShape({s0, s1, s2, s3}))
                      .AppendInputSymbolTensor(gert::SymbolShape({s2}))
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
  // dim1
  ASSERT_EQ(out_shape->GetDim(0), Symbol(input_list1[0]));
  // dim2
  ASSERT_EQ(out_shape->GetDim(1), Symbol(input_list1[1]));
  // dim3
  ASSERT_EQ(out_shape->GetDim(2), Symbol(input_list1[2]));
  // dim4
  ASSERT_EQ(out_shape->GetDim(3), Symbol(input_list1[3]));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSquareSumV1) {
  auto func = GetInferFunc("SquareSumV1");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 2));
  auto x_shape = gert::SymbolShape({s0, s1, s2});
  InferSymbolShapeContextTestBuilder builder("SquareSumV1", "SquareSumV1");

  // 1. 异常场景1：Attr axis参数非法
  auto op_desc = builder.GetOrCreateOpDescPtr();
  std::vector<int64_t> axis0 = {0, 3};
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", axis0);
  op_desc->AppendIrAttrName("keep_dims");
  AttrUtils::SetBool(op_desc, "keep_dims", true);
  auto infer_context = builder.AppendInputSymbolTensor(x_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 2. 正常场景1 keep_dims true
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  std::vector<int64_t> axis1 = {0, 1};
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", axis1);
  op_desc->AppendIrAttrName("keep_dims");
  AttrUtils::SetBool(op_desc, "keep_dims", true);
  infer_context = builder.AppendInputSymbolTensor(x_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), x_shape.GetDimNum());
  // dim0
  ASSERT_EQ(out_shape->GetDim(0), Symbol(1));
  // dim1
  ASSERT_EQ(out_shape->GetDim(1), Symbol(1));
  // dim2
  ASSERT_EQ(out_shape->GetDim(2), x_shape.GetDim(2));

  // 3. 正常场景2 keep_dims false
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  std::vector<int64_t> axis2 = {0, 1};
  op_desc->AppendIrAttrName("axis");
  AttrUtils::SetListInt(op_desc, "axis", axis2);
  op_desc->AppendIrAttrName("keep_dims");
  AttrUtils::SetBool(op_desc, "keep_dims", false);
  infer_context = builder.AppendInputSymbolTensor(x_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), 1);
  // dim0
  ASSERT_EQ(out_shape->GetDim(0), x_shape.GetDim(2));
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForRepeat) {
  auto func = GetInferFunc("Repeat");
  ASSERT_TRUE(func.first != nullptr);
  InferSymbolShapeContextTestBuilder builder("Repeat", "Repeat");

  // UNSUPPORTED场景1 输入为scalar
  auto data_shape0 = gert::SymbolShape({});
  gert::SymbolShape data_shape1({Symbol(1)});
  std::vector<ge::Expression> data_shape1_value = {Symbol(8)};
  auto infer_context = builder.AppendInputSymbolTensor(data_shape0)
                              .AppendInputSymbolTensor(data_shape1, true, &data_shape1_value)
                              .OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);

  // UNSUPPORTED场景2 repeat_num输入empty
  data_shape0 = gert::SymbolShape({Symbol(1), Symbol(2)});
  infer_context = builder.AppendInputSymbolTensor(data_shape0)
                              .AppendInputSymbolTensor(data_shape1, true, nullptr)
                              .OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
}

static void BuildConv2DInferContext(InferSymbolShapeContextTestBuilder &builder,
                                    const std::initializer_list<Expression> &x_values,
                                    const std::initializer_list<Expression> &w_values,
                                    const std::vector<int64_t> &strides_values, const std::vector<int64_t> &pads_values,
                                    const std::vector<int64_t> &dilations_values, int64_t groups = 1,
                                    Format x_format = FORMAT_NCHW, Format w_format = FORMAT_NCHW) {
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AppendIrAttrName("strides");
  op_desc->AppendIrAttrName("pads");
  op_desc->AppendIrAttrName("dilations");
  op_desc->AppendIrAttrName("groups");
  AttrUtils::SetListInt(op_desc, "strides", strides_values);
  AttrUtils::SetListInt(op_desc, "pads", pads_values);
  AttrUtils::SetListInt(op_desc, "dilations", dilations_values);
  AttrUtils::SetInt(op_desc, "groups", groups);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), x_format, DT_FLOAT));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), w_format, DT_FLOAT));
  builder
      .AppendInputSymbolTensor(gert::SymbolShape(x_values))  // input_x
      .AppendInputSymbolTensor(gert::SymbolShape(w_values))  // w
      .OutputNum(1);
  op_desc->MutableInputDesc(0)->SetOriginFormat(x_format);
  op_desc->MutableInputDesc(1)->SetOriginFormat(w_format);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForConv2D) {
  auto func = GetInferFunc("Conv2D");
  ASSERT_TRUE(func.first != nullptr);
  InferSymbolShapeContextTestBuilder builder("Conv2D", "conv2D");
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(16, MakeShared<InputShapeSource>(0, 8));
    // 异常场景：输入x格式异常
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0,s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_ND, FORMAT_NHWC);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), PARAM_INVALID);
    // 异常场景：输入filter格式异常
    builder.Destroy();
    BuildConv2DInferContext(builder, {s0,s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_ND);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    // 异常场景：in_channels(>0) should be divisible by kernel_channels when groups = 1
    builder.Destroy();
    auto s8 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 4));
    BuildConv2DInferContext(builder, {s0,s1, s2, s8}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
     ASSERT_EQ(guard_infos.size(), 2);
     const std::set<std::string> expect_guard = {
       "ExpectNe(0, Mod(s8, s6))", "ExpectNe(0, s6)"};
     for (auto &iter : guard_infos) {
       EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 2);
    const std::set<std::string> assert_guard = {
      "ExpectLt(0, s4)", "ExpectLt(0, s5)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    // 异常场景：out_channels should be divisible by groups.
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(15, MakeShared<InputShapeSource>(0, 7));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0,s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 2,
                            FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 0);
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 2);
    const std::set<std::string> assert_guard = {
      "ExpectLt(0, s4)", "ExpectLt(0, s5)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    // 异常场景：strides list should be 4D
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(16, MakeShared<InputShapeSource>(0, 7));
    std::vector<int64_t> input_list0 = {1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0,s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 2,
                            FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    // 异常场景：Conv2D attr strides should be positive
    builder.Destroy();
    input_list0 = {0, 0, 0, 0};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 2,
                            FORMAT_NHWC, FORMAT_HWCN);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    // 异常场景：dilations list should be 4D
    builder.Destroy();
    input_list0 = {1, 1, 1, 1};
    input_list2 = {1, 1, 1};
    BuildConv2DInferContext(builder, {s0,s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 2,
                            FORMAT_NHWC, FORMAT_HWCN);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    // 异常场景：Conv2D attr dilations should be positive
    builder.Destroy();
    input_list2 = {0, 0, 0, 0};
    BuildConv2DInferContext(builder, {s0,s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 2,
                            FORMAT_NHWC, FORMAT_HWCN);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 0);
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 3);
    const std::set<std::string> assert_guard = {
      "ExpectLt(0, s4)", "ExpectLt(0, s5)","ExpectEq(0, Mod(s7, 2))"};
    for (auto &iter : assert_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(assert_guard.find(guard_str), assert_guard.end());
    }
  }
  {
    // zerotensor
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(0, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(16, MakeShared<InputShapeSource>(0, 7));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), sym::Floor(s1-(s4-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(s2-(s5-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 7);
    const std::set<std::string> expect_guard = {
        "ExpectNe(s2, s5)", "ExpectEq(0, s0)",
        "ExpectNe(0, s3)", "ExpectNe(s1, s4)",
        "ExpectNe((s3 / (s6)), 0)",
        "ExpectEq(0, Mod(s3, s6))", "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 6);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                                "ExpectLt(0, s5)",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe(0, Floor((s1 - (-1 + s4))))",
                                                "ExpectLe(0, s7)",
                                                "ExpectLe(0, s0)",
                                                "ExpectLe(0, Floor((s2 - (-1 + s5))))"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // 标准卷积
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(28, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(16, MakeShared<InputShapeSource>(0, 7));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                        FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), sym::Floor(s1-(s4-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(s2-(s5-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), s7);
    // NCHW
    builder.Destroy();
    BuildConv2DInferContext(builder, {s0, s3, s1, s2}, {s7, s6, s4, s5}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NCHW, FORMAT_NCHW);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s7);
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(s1-(s4-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), sym::Floor(s2-(s5-Symbol(1))));
    // NCHW NHWC
    builder.Destroy();
    BuildConv2DInferContext(builder, {s0, s3, s1, s2}, {s7, s4, s5, s6}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NCHW, FORMAT_NHWC);
    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), s7);
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(s1-(s4-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), sym::Floor(s2-(s5-Symbol(1))));
    // 分组卷积
    builder.Destroy();
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 2,
                            FORMAT_NHWC, FORMAT_HWCN);
    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("data_format");
    op_desc->AppendIrAttrName("offset_x");
    op_desc->AppendIrAttrName("padding");
    AttrUtils::SetStr(op_desc, "data_format", "NHWC");
    AttrUtils::SetInt(op_desc, "offset_x", 0);
    AttrUtils::SetStr(op_desc, "padding", "");

    infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    out_shape = infer_context->GetOutputSymbolShape(0);

    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), sym::Floor(s1-(s4-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(s2-(s5-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {"ExpectNe(0, s7)", "ExpectNe(0, s1)",
                                                "ExpectNe(s2, s5)",
                                                "ExpectNe(0, s0)", "ExpectNe(0, s3)",
                                                "ExpectNe(s1, s4)",
                                                "ExpectNe(0, s2)", "ExpectNe((s3 / (s6)), 0)",
                                                "ExpectEq(0, Mod(s3, s6))",
                                                "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 6);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                                "ExpectLt(0, s5)",
                                                "ExpectEq(0, Mod(s7, 2))",
                                                "ExpectLe(s4, s1)",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe(s5, s2)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }

  {
    // 带填充的卷积
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(32, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(32, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(8, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {2, 2, 2, 2};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                        FORMAT_NHWC, FORMAT_HWCN);
    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("data_format");
    op_desc->AppendIrAttrName("offset_x");
    op_desc->AppendIrAttrName("padding");
    AttrUtils::SetStr(op_desc, "data_format", "NHWC");
    AttrUtils::SetInt(op_desc, "offset_x", 0);
    AttrUtils::SetStr(op_desc, "padding", "EXPLICIT");

    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);

    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), sym::Floor(Symbol(4)+s1-(s4-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(Symbol(4)+s2-(s5-Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {"ExpectNe(0, s7)", "ExpectNe(0, s1)",
                                                "ExpectNe((4 + s2), s5)",
                                                "ExpectNe(0, s0)", "ExpectNe(0, s3)",
                                                "ExpectNe((4 + s1), s4)",
                                                "ExpectNe(0, s2)",
                                                "ExpectNe((s3 / (s6)), 0)",
                                                "ExpectEq(0, Mod(s3, s6))",
                                                "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 5);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                                "ExpectLt(0, s5)",
                                                "ExpectLe(s4, (4 + s1))",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe(s5, (4 + s2))"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // 膨胀卷积
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 2, 2, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                        FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), sym::Floor(s1 - (s4 - Symbol(1)) * Symbol(2)));
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(s2 - (s5 - Symbol(1)) * Symbol(2)));
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {"ExpectNe(0, s7)", "ExpectNe(0, s1)",
                                                "ExpectNe((1 + s2), (2 * s5))",
                                                "ExpectNe(0, s0)", "ExpectNe(0, s3)",
                                                "ExpectNe((1 + s1), (2 * s4))",
                                                "ExpectNe(0, s2)", "ExpectNe((s3 / (s6)), 0)",
                                                "ExpectEq(0, Mod(s3, s6))",
                                                "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 5);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                                "ExpectLt(0, s5)",
                                                "ExpectLe((2 * s4), (1 + s1))",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe((2 * s5), (1 + s2))"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // 异常场景,输入尺寸过小
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {0, 0, 0, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {"ExpectNe(0, s7)", "ExpectNe(0, s1)",
                                                "ExpectNe(s2, s5)",
                                                "ExpectNe(0, s0)", "ExpectNe(0, s3)",
                                                "ExpectNe(s1, s4)",
                                                "ExpectNe(0, s2)", "ExpectNe((s3 / (s6)), 0)",
                                                "ExpectEq(0, Mod(s3, s6))",
                                                "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 3);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)", "ExpectLt(0, s5)",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // 非对称填充
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(1, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 1, 1, 1};
    std::vector<int64_t> input_list1 = {1, 0, 1, 0};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);
    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    ASSERT_EQ(out_shape->GetDim(0), s0);
    ASSERT_EQ(out_shape->GetDim(1), sym::Floor(Symbol(1) + s1 - (s4 - Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(2), sym::Floor(Symbol(1) + s2 - (s5 - Symbol(1))));
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {"ExpectNe(0, s7)", "ExpectNe(0, s1)",
                                                "ExpectNe((1 + s2), s5)",
                                                "ExpectNe(0, s0)", "ExpectNe(0, s3)",
                                                "ExpectNe((1 + s1), s4)",
                                                "ExpectNe(0, s2)",
                                                "ExpectNe((s3 / (s6)), 0)",
                                                "ExpectEq(0, Mod(s3, s6))",
                                                "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 5);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)", "ExpectLt(0, s5)",
                                                "ExpectLe(s4, (1 + s1))",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe(s5, (1 + s2))"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // padding same
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(48, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(112, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(112, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(64, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 2, 2, 1};
    std::vector<int64_t> input_list1 = {-1, -1, -1, -1};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);

    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("data_format");
    op_desc->AppendIrAttrName("offset_x");
    op_desc->AppendIrAttrName("padding");
    AttrUtils::SetStr(op_desc, "data_format", "NHWC");
    AttrUtils::SetInt(op_desc, "offset_x", 0);
    AttrUtils::SetStr(op_desc, "padding", "SAME");

    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    // 48
    ASSERT_EQ(out_shape->GetDim(0), s0);
    // 56
    ASSERT_EQ(out_shape->GetDim(1).Simplify(), sym::Floor((Symbol(2) * sym::Floor((s4 - Symbol(2)) / Symbol(2))
      - Symbol(1) + sym::Mod(s4 - Symbol(2), Symbol(2)) + s1 - (s4 - Symbol(1))) / Symbol(2)) + Symbol(1));
    // 56
    ASSERT_EQ(out_shape->GetDim(2).Simplify(), sym::Floor((Symbol(2) * sym::Floor((s5 - Symbol(2)) / Symbol(2))
      - Symbol(1) + sym::Mod(s5 - Symbol(2), Symbol(2)) + s2 - (s5 - Symbol(1))) / Symbol(2)) + Symbol(1));
    // 64
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 13);
    const std::set<std::string> expect_guard = {"ExpectNe(0, s7)",
                                                "ExpectNe(0, s1)",
                                                "ExpectNe(((Mod((-2 + s4), 2) * Rational(1 , 2)) + (Rational(1 , 2) * s1) + Floor(((-2 + s4) * Rational(1 , 2)))), (Rational(1 , 2) * s4))",
                                                "ExpectNe(0, s0)",
                                                "ExpectNe(0, s3)",
                                                "ExpectNe(((Mod((-2 + s5), 2) * Rational(1 , 2)) + (Rational(1 , 2) * s2) + Floor(((-2 + s5) * Rational(1 , 2)))), (Rational(1 , 2) * s5))",
                                                "ExpectNe(0, s2)",
                                                "ExpectNe((s3 / (s6)), 0)",
                                                "ExpectEq(0, Mod(s3, s6))",
                                                "ExpectLt(2, s5)",
                                                "ExpectLe(Mod(s1, 2), 0)",
                                                "ExpectLt(2, s4)",
                                                "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 9);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                                "ExpectLe(0, Floor(((-2 + s4) * Rational(1 , 2))))",
                                                "ExpectLt(0, s5)",
                                                "ExpectLe(0, Floor(((-2 + s5) * Rational(1 , 2))))",
                                                "ExpectLe(0, (Floor(((-2 + s4) * Rational(1 , 2))) + Mod((-2 + s4), 2)))",
                                                "ExpectLe(0, (Floor(((-2 + s5) * Rational(1 , 2))) + Mod((-2 + s5), 2)))",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe(0, ((2 * Floor(((-2 + s4) * Rational(1 , 2)))) + -1 + Mod((-2 + s4), 2) + s1 - (-1 + s4)))",
                                                "ExpectLe(s4, ((2 * Floor(((-2 + s4) * Rational(1 , 2)))) + Mod((-2 + s4), 2) + s1))",
                                                "ExpectLe(s5, ((2 * Floor(((-2 + s5) * Rational(1 , 2)))) + Mod((-2 + s5), 2) + s2))"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // padding valid
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(48, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(112, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(112, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(64, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 2, 2, 1};
    std::vector<int64_t> input_list1 = {-1, -1, -1, -1};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);

    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("data_format");
    op_desc->AppendIrAttrName("offset_x");
    op_desc->AppendIrAttrName("padding");
    AttrUtils::SetStr(op_desc, "data_format", "NHWC");
    AttrUtils::SetInt(op_desc, "offset_x", 0);
    AttrUtils::SetStr(op_desc, "padding", "VALID");

    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    auto out_shape = infer_context->GetOutputSymbolShape(0);
    ASSERT_EQ(out_shape->GetDimNum(), input_list1.size());
    // 48
    ASSERT_EQ(out_shape->GetDim(0), s0);
    // 56
    ASSERT_EQ(out_shape->GetDim(1).Simplify(), sym::Floor((s1 - Symbol(1) - (s4 - Symbol(1))) / Symbol(2)) + Symbol(1));
    // 56
    ASSERT_EQ(out_shape->GetDim(2).Simplify(), sym::Floor((s2 - Symbol(1) - (s5 - Symbol(1))) / Symbol(2)) + Symbol(1));
    // 64
    ASSERT_EQ(out_shape->GetDim(3), s7);
    const std::vector<SymbolCheckInfo> guard_infos = shape_env.GetAllSymbolCheckInfos();
    ASSERT_EQ(guard_infos.size(), 10);
    const std::set<std::string> expect_guard = {
        "ExpectNe(0, s7)", "ExpectNe(0, s1)",
        "ExpectNe((Rational(1 , 2) * s2), (Rational(1 , 2) * s5))",
        "ExpectNe(0, s0)", "ExpectNe(0, s3)",
        "ExpectNe((Rational(1 , 2) * s1), (Rational(1 , 2) * s4))",
        "ExpectNe(0, s2)", "ExpectNe((s3 / (s6)), 0)", "ExpectEq(0, Mod(s3, s6))",
        "ExpectNe(0, s6)"};
    for (auto &iter : guard_infos) {
      const std::string guard_str = std::string(iter.expr.Serialize().get());
      std::cout << "guard info: " << guard_str << std::endl;
      EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
    }
    const std::vector<SymbolCheckInfo> assert_infos = shape_env.GetAllSymbolAssertInfos();
    ASSERT_EQ(assert_infos.size(), 5);
    const std::set<std::string> assert_guard = {"ExpectLt(0, s4)", "ExpectLt(0, s5)",
                                                "ExpectLe(s4, s1)",
                                                "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                                "ExpectLe(s5, s2)"};
    for (auto &iter : assert_infos) {
      EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
    }
  }
  {
    // 异常场景：padding无效字符串
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    builder.Destroy();
    auto s0 = shape_env.CreateSymbol(48, MakeShared<InputShapeSource>(0, 1));
    auto s1 = shape_env.CreateSymbol(112, MakeShared<InputShapeSource>(0, 2));
    auto s2 = shape_env.CreateSymbol(112, MakeShared<InputShapeSource>(0, 3));
    auto s3 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 4));
    auto s4 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 5));
    auto s5 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 6));
    auto s6 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 7));
    auto s7 = shape_env.CreateSymbol(64, MakeShared<InputShapeSource>(0, 8));
    std::vector<int64_t> input_list0 = {1, 2, 2, 1};
    std::vector<int64_t> input_list1 = {-1, -1, -1, -1};
    std::vector<int64_t> input_list2 = {1, 1, 1, 1};
    BuildConv2DInferContext(builder, {s0, s1, s2, s3}, {s4, s5, s6, s7}, input_list0, input_list1, input_list2, 1,
                            FORMAT_NHWC, FORMAT_HWCN);

    auto op_desc = builder.GetOrCreateOpDescPtr();
    op_desc->AppendIrAttrName("data_format");
    op_desc->AppendIrAttrName("offset_x");
    op_desc->AppendIrAttrName("padding");
    AttrUtils::SetStr(op_desc, "data_format", "NHWC");
    AttrUtils::SetInt(op_desc, "offset_x", 0);
    AttrUtils::SetStr(op_desc, "padding", "AABB");

    auto infer_context = builder.Build();
    ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAddLayerNorm) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("AddLayerNorm", "addlayernorm");
  auto x1_shape = gert::SymbolShape({s0, s1, s2});
  auto x2_shape = gert::SymbolShape({s0, s1, s2});
  auto gamma_shape = gert::SymbolShape({s0, s1, s2});
  auto beta_shape = gert::SymbolShape({s0, s1, s2});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(x1_shape)
                              .AppendInputSymbolTensor(x2_shape)
                              .AppendInputSymbolTensor(gamma_shape)
                              .AppendInputSymbolTensor(beta_shape)
                              .OutputNum(4)
                              .Build();

  const auto func = GetInferFunc("AddLayerNorm");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape1 = gert::SymbolShape({s0, s1, s2});
  auto expect_shape2 = gert::SymbolShape({s0, s1, Symbol(1)});
  auto expect_shape3 = gert::SymbolShape({s0, s1, Symbol(1)});
  auto expect_shape4 = gert::SymbolShape({s0, s1, s2});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), expect_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(2)->GetDims(), expect_shape3.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(3)->GetDims(), expect_shape4.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForApplyAdagradD) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("ApplyAdagradD", "ApplyAdagradD");
  auto var_shape = gert::SymbolShape({s0, s1, s2});
  auto accum_shape = gert::SymbolShape({s0, s1, s2});
  auto lr_shape = gert::SymbolShape({s0, s1, s2});
  auto grad_shape = gert::SymbolShape({s0, s1, s2});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(var_shape)
                              .AppendInputSymbolTensor(accum_shape)
                              .AppendInputSymbolTensor(lr_shape)
                              .AppendInputSymbolTensor(grad_shape)
                              .OutputNum(2)
                              .Build();

  const auto func = GetInferFunc("ApplyAdagradD");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape1 = gert::SymbolShape({s0, s1, s2});
  auto expect_shape2 = gert::SymbolShape({s0, s1, s2});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), expect_shape2.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForApplyAdamD) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("ApplyAdamD", "ApplyAdamD");
  auto var_shape = gert::SymbolShape({s0, s1, s2});
  auto m_shape = gert::SymbolShape({s0, s1, s2});
  auto v_shape = gert::SymbolShape({s0, s1, s2});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(var_shape)
                              .AppendInputSymbolTensor(m_shape)
                              .AppendInputSymbolTensor(v_shape)
                              .OutputNum(3)
                              .Build();

  const auto func = GetInferFunc("ApplyAdamD");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s0, s1, s2});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), expect_shape.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(2)->GetDims(), expect_shape.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForAssign) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("Assign", "Assign");
  auto ref_shape = gert::SymbolShape({s0, s1, s2});
  auto value_shape = gert::SymbolShape({s0, s1, s2});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(ref_shape)
                              .AppendInputSymbolTensor(value_shape)
                              .OutputNum(1)
                              .Build();

  const auto func = GetInferFunc("Assign");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape1 = gert::SymbolShape({s0, s1, s2});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape1.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForBNTrainingUpdate) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  InferSymbolShapeContextTestBuilder builder("BNTrainingUpdate", "BNTrainingUpdate");
  auto x_shape = gert::SymbolShape({s0, s1, s2});
  auto scale_shape = gert::SymbolShape({s0, s1});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(x_shape)
                              .AppendInputSymbolTensor(x_shape)
                              .AppendInputSymbolTensor(x_shape)
                              .AppendInputSymbolTensor(scale_shape)
                              .AppendInputSymbolTensor(scale_shape)
                              .AppendInputSymbolTensor(scale_shape)
                              .AppendInputSymbolTensor(scale_shape)
                              .OutputNum(5)
                              .Build();

  const auto func = GetInferFunc("BNTrainingUpdate");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape1 = gert::SymbolShape({s0, s1, s2});
  auto expect_shape2 = gert::SymbolShape({s0, s1});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape1.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), expect_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(2)->GetDims(), expect_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(3)->GetDims(), expect_shape2.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(4)->GetDims(), expect_shape2.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForL2Loss) {
  auto func = GetInferFunc("L2Loss");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto x_shape = gert::SymbolShape({s0, s1});
  InferSymbolShapeContextTestBuilder builder("L2Loss", "L2Loss");

  auto infer_context = builder.AppendInputSymbolTensor(x_shape).OutputNum(1).Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto out_shape = infer_context->GetOutputSymbolShape(0);

  ASSERT_EQ(out_shape->GetDimNum(), 0);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSelectV2) {
  auto func = GetInferFunc("SelectV2");
  ASSERT_TRUE(func.first != nullptr);
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

  auto condition = gert::SymbolShape({s0, s1});
  auto input_shape0 = gert::SymbolShape({s0, s1});
  auto input_shape1 = gert::SymbolShape({s0, s1});
  InferSymbolShapeContextTestBuilder builder("SelectV2", "selectv2");
  {
    // 全相等
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 5));
    auto condition = gert::SymbolShape({s0, s1});
    auto input_shape0 = gert::SymbolShape({s2, s3});
    auto input_shape1 = gert::SymbolShape({s4, s5});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                             .AppendInputSymbolTensor(input_shape0)
                             .AppendInputSymbolTensor(input_shape1)
                             .OutputNum(1)
                             .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
  }
  {
    // condition维度数量为1且等于输入的第0维度
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto condition = gert::SymbolShape({s0});
    auto input_shape0 = gert::SymbolShape({s0, s1, s2});
    auto input_shape1 = gert::SymbolShape({s1, s2});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                             .AppendInputSymbolTensor(input_shape0)
                             .AppendInputSymbolTensor(input_shape1)
                             .OutputNum(1)
                             .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
  }
  {
    // condition维度数量为0
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto condition = gert::SymbolShape({});
    auto input_shape0 = gert::SymbolShape({s0, s1});
    auto input_shape1 = gert::SymbolShape({s2, s3});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                             .AppendInputSymbolTensor(input_shape0)
                             .AppendInputSymbolTensor(input_shape1)
                             .OutputNum(1)
                             .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
  }
  {
    // condition维度数量不为0或1且维度数量不等于后面的输入
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 5));
    auto s6 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 6));
    auto s7 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 7));
    auto condition = gert::SymbolShape({s0, s1});
    auto input_shape0 = gert::SymbolShape({s2, s3, s4});
    auto input_shape1 = gert::SymbolShape({s5, s6, s7});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                             .AppendInputSymbolTensor(input_shape0)
                             .AppendInputSymbolTensor(input_shape1)
                             .OutputNum(1)
                             .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), input_shape0.GetDims());
  }
  {
    // condition维度数量大于后面的输入可以进行广播
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 5));
    auto s6 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 6));
    auto s7 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 7));
    auto condition = gert::SymbolShape({s0, s1, s2, s3});
    auto input_shape0 = gert::SymbolShape({s4, s5});
    auto input_shape1 = gert::SymbolShape({s6, s7});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                             .AppendInputSymbolTensor(input_shape0)
                             .AppendInputSymbolTensor(input_shape1)
                             .OutputNum(1)
                             .Build();
    ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({s0, s1, s4, s5}).GetDims());
  }
  {
    // condition和输入的维度不同且无法广播
    builder.Destroy();
    ShapeEnvAttr shape_env;
    ShapeEnvGuarder guarder(&shape_env);
    auto s0 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto s1 = shape_env.CreateSymbol(3, MakeShared<InputShapeSource>(0, 1));
    auto s2 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 2));
    auto s3 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
    auto s4 = shape_env.CreateSymbol(6, MakeShared<InputShapeSource>(0, 4));
    auto s5 = shape_env.CreateSymbol(7, MakeShared<InputShapeSource>(0, 5));
    auto s6 = shape_env.CreateSymbol(8, MakeShared<InputShapeSource>(0, 6));
    auto s7 = shape_env.CreateSymbol(9, MakeShared<InputShapeSource>(0, 7));
    auto condition = gert::SymbolShape({s0, s1, s2, s3});
    auto input_shape0 = gert::SymbolShape({s4, s5});
    auto input_shape1 = gert::SymbolShape({s6, s7});
    auto infer_context = builder.AppendInputSymbolTensor(condition)
                             .AppendInputSymbolTensor(input_shape0)
                             .AppendInputSymbolTensor(input_shape1)
                             .OutputNum(1)
                             .Build();
    ASSERT_NE(func.first(infer_context), ge::GRAPH_SUCCESS);
  }
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForBNTrainingUpdateGrad) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));

  InferSymbolShapeContextTestBuilder builder("BNTrainingUpdateGrad", "BNTrainingUpdateGrad");
  auto x_shape = gert::SymbolShape({s0, s1, s2, s3});
  auto batch_variance = gert::SymbolShape({s3});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(x_shape)
                           .AppendInputSymbolTensor(x_shape)
                           .AppendInputSymbolTensor(batch_variance)
                           .AppendInputSymbolTensor(batch_variance)
                           .OutputNum(2)
                           .Build();

  const auto func = GetInferFunc("BNTrainingUpdateGrad");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s3});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), expect_shape.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForDiagPartD) {
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto s3 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 4));

  InferSymbolShapeContextTestBuilder builder("DiagPartD", "diagpartd");
  auto x_shape = gert::SymbolShape({s0, s1, s2, s3});
  auto assist = gert::SymbolShape({s0, s1, s2, s3});
  auto op_descPtr = builder.GetOrCreateOpDescPtr();
  auto infer_context = builder.AppendInputSymbolTensor(x_shape)
                           .AppendInputSymbolTensor(assist)
                           .OutputNum(1)
                           .Build();

  const auto func = GetInferFunc("DiagPartD");
  ASSERT_TRUE(func.first != nullptr);
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  auto expect_shape = gert::SymbolShape({s0, s1});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForPadV3) {
  const auto func = GetInferFunc("PadV3");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("PadV3", "padv3");
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));
  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  // 正常场景1： paddings 的shape是[n,2] n-input 的 dim number
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  auto paddings_shape = gert::SymbolShape({ge::Symbol(3), ge::Symbol(2)});
  auto paddings = std::vector<Expression>{Symbol(1), Symbol(2), Symbol(2), Symbol(1), Symbol(3), Symbol(3)};
  string mode = "reflect";
  bool paddings_contiguous = true;
  op_desc->AppendIrAttrName("mode");
  AttrUtils::SetStr(op_desc, "mode", mode);
  op_desc->AppendIrAttrName("paddings_contiguous");
  AttrUtils::SetBool(op_desc, "paddings_contiguous", paddings_contiguous);

  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                           .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                           .OutputNum(1)
                           .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);

  auto expect_shape =
      gert::SymbolShape({s0 + ge::Symbol(1) + ge::Symbol(2), s1 + ge::Symbol(2) + ge::Symbol(1),
                         s2 + ge::Symbol(3) + ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 正常场景2：paddings_contiguous为false
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT64));
  mode = "edge";
  paddings_contiguous = false;
  op_desc->AppendIrAttrName("mode");
  AttrUtils::SetStr(op_desc, "mode", mode);
  op_desc->AppendIrAttrName("paddings_contiguous");
  AttrUtils::SetBool(op_desc, "paddings_contiguous", paddings_contiguous);
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  expect_shape =
      gert::SymbolShape({s0 + ge::Symbol(1) + ge::Symbol(1), s1 + ge::Symbol(2) + ge::Symbol(3),
                         s2 + ge::Symbol(2) + ge::Symbol(3)});
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), expect_shape.GetDims());

  // 异常场景1：the paddings type is not int32 or int64
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT8));
  op_desc->AppendIrAttrName("mode");
  AttrUtils::SetStr(op_desc, "mode", mode);
  op_desc->AppendIrAttrName("paddings_contiguous");
  AttrUtils::SetBool(op_desc, "paddings_contiguous", paddings_contiguous);
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 异常场景2：the paddings num is not twice of the input rank
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  op_desc->AppendIrAttrName("mode");
  AttrUtils::SetStr(op_desc, "mode", mode);
  op_desc->AppendIrAttrName("paddings_contiguous");
  AttrUtils::SetBool(op_desc, "paddings_contiguous", paddings_contiguous);
  paddings_shape = gert::SymbolShape({ge::Symbol(2), ge::Symbol(2)});
  paddings = std::vector<Expression>{Symbol(1), Symbol(2), Symbol(2), Symbol(1)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor(paddings_shape, true, &paddings)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 异常场景3：the paddings are null
  builder.Destroy();
  op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  paddings = {};
  op_desc->AppendIrAttrName("mode");
  AttrUtils::SetStr(op_desc, "mode", mode);
  op_desc->AppendIrAttrName("paddings_contiguous");
  AttrUtils::SetBool(op_desc, "paddings_contiguous", paddings_contiguous);
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::UNSUPPORTED);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForUnsortedSegmentMin) {
  const auto func = GetInferFunc("UnsortedSegmentMin");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("UnsortedSegmentMin", "unsortedsegmentmin");
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));
  auto s2 = shape_env.CreateSymbol(5, MakeShared<InputShapeSource>(0, 3));

  auto op_desc = builder.GetOrCreateOpDescPtr();
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc());
  op_desc->AddInputDesc(GeTensorDesc(GeShape(), FORMAT_ND, DT_INT32));
  auto input_shape = gert::SymbolShape({s0, s1, s2});
  auto segment_ids = gert::SymbolShape({s0});
  auto segment_num = gert::SymbolShape({Symbol(1)});
  auto segment_val = std::vector<Expression>{Symbol(5)};

  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                           .AppendInputSymbolTensor(segment_ids)
                           .AppendInputSymbolTensor(segment_num, true, &segment_val)
                           .OutputNum(1)
                           .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({Symbol(5), s1, s2}).GetDims());

  // 异常场景1：segment_num 维度size不为1
  builder.Destroy();
  segment_num = gert::SymbolShape({s0, s1});
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor(segment_ids)
                      .AppendInputSymbolTensor(segment_num)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);

  // 异常场景2：segment_num 维度size为1, value size不为1
  builder.Destroy();
  segment_num = gert::SymbolShape({Symbol(2)});
  segment_val = std::vector<Expression>{Symbol(2), Symbol(3)};
  infer_context = builder.AppendInputSymbolTensor(input_shape)
                      .AppendInputSymbolTensor(segment_ids)
                      .AppendInputSymbolTensor(segment_num, true, &segment_val)
                      .OutputNum(1)
                      .Build();
  ASSERT_EQ(func.first(infer_context), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferFuncUT, InferSymbolicShapeForSparseSoftmaxCrossEntropyWithLogits) {
  const auto func = GetInferFunc("SparseSoftmaxCrossEntropyWithLogits");
  ASSERT_TRUE(func.first != nullptr);

  InferSymbolShapeContextTestBuilder builder("SparseSoftmaxCrossEntropyWithLogits", "sparsesoftmaxcrossentropywithlogits");
  ShapeEnvAttr shape_env;
  ShapeEnvGuarder guarder(&shape_env);
  auto s0 = shape_env.CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s1 = shape_env.CreateSymbol(2, MakeShared<InputShapeSource>(0, 2));

  auto input_shape = gert::SymbolShape({s0, s1});
  auto label = gert::SymbolShape({s0});

  auto infer_context = builder.AppendInputSymbolTensor(input_shape)
                           .AppendInputSymbolTensor(label)
                           .OutputNum(2)
                           .Build();

  ASSERT_EQ(func.first(infer_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(infer_context->GetOutputSymbolShape(0)->GetDims(), gert::SymbolShape({s0}).GetDims());
  ASSERT_EQ(infer_context->GetOutputSymbolShape(1)->GetDims(), gert::SymbolShape({s0, s1}).GetDims());
}
} // namespace ge
