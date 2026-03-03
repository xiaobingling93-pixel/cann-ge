/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <gtest/gtest.h>
#include <dlfcn.h>
#include "graph/utils/graph_utils_ex.h"
#include "es_ge_test_ops_c.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/passes/feature/auto_fuse_pass.h"
#include "framework/common/types.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "common/env_path.h"
#include "common/topo_checker.h"
#include "utils/autofuse_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/omg_util.h"
#include "mmpa/mmpa_api.h"
#include "graph/node.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h"
#include "graph/operator_reg.h"

#include <graph/manager/graph_manager.h>
#include <graph/optimize/symbolic/shape_env_guarder.h>
#include "graph/optimize/autofuse/autofuse_optimize.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {

class RuntimeMock : public RuntimeStub {
 public:
  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)  {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "fake"); // fake
    return RT_ERROR_NONE;
  }
};


class SymbolizeValueST : public testing::Test {
 public:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
    gert::LoadDefaultSpaceRegistry();
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, ori_opp_path_env_);
    MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, ori_ld_path_env_);
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, (ascend_install_path + "/opp").c_str(), 1, ret_);
    MM_SYS_SET_ENV(MM_ENV_LD_LIBRARY_PATH, (ascend_install_path + "/runtime/lib64").c_str(), 1, ret_);
    mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
    ori_global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    ori_graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    ori_session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});
    std::map<std::string, std::string> options;
    GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
  }
  void TearDown() override {
    RuntimeStub::Reset();
    unsetenv("AUTOFUSE_FLAGS");
    if (ori_ld_path_env_ != nullptr) {
      MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, ori_opp_path_env_, 1, ret_);
    } else {
      MM_SYS_UNSET_ENV(MM_ENV_ASCEND_OPP_PATH, ret_);
    }
    if (ori_ld_path_env_ != nullptr) {
      MM_SYS_SET_ENV(MM_ENV_LD_LIBRARY_PATH, ori_ld_path_env_, 1, ret_);
    } else {
      MM_SYS_UNSET_ENV(MM_ENV_LD_LIBRARY_PATH, ret_);
    }
    ge::GetThreadLocalContext().SetGlobalOption(ori_global_options_);
    ge::GetThreadLocalContext().SetGraphOption(ori_graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(ori_session_options_);
    gert::UnLoadDefaultSpaceRegistry();
  }

 private:
  int32_t ret_{EN_ERROR};
  const char_t *ori_opp_path_env_{nullptr};
  const char_t *ori_ld_path_env_{nullptr};
  const char_t *enable_auto_fuse_{"1"};
  std::map<std::string, std::string> ori_global_options_;
  std::map<std::string, std::string> ori_graph_options_;
  std::map<std::string, std::string> ori_session_options_;
};

/*
 *
 *                           data0
 *                             |
 *                            abs
 *                            |
 *             -----------------------------------
 *      data1  |   data2   |   data3  |   data4  |
 *        |   /      |    /      |   /      |   /
 *        | /        |  /        |  /       |  /
 *     repeat1     repeat2      repeat3    repeat4
 *           |      |                 |    |
 *            \     /                  \  /
 *             mul1                    add1
 *               |---------    ----------|
 *                        |    |
 *                       Netoutput
 */
namespace {
REG_OP(Repeat)
    .INPUT(x, TensorType::ALL())
    .INPUT(repeat_times, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Repeat)

        IMPL_OP(Repeat)
    .InputsDataDependency({1});  // repeat归属自定义二类算子，符号化推导需要获取
graphStatus TestRepeatInferSymbolShapeFunc(gert::InferSymbolShapeContext *context) {
  auto input0 = context->GetInputSymbolShape(0);
  GE_ASSERT_NOTNULL(input0);
  auto input1 = context->GetInputSymbolTensor(1);
  GE_ASSERT_NOTNULL(input1);
  auto symbol_value = input1->GetSymbolicValue();
  if (symbol_value == nullptr) {
    GELOGW("Infer Symbol shape failed, symbol_value is nullptr!");
    return UNSUPPORTED;
  }
  auto output = context->GetOutputSymbolShape(0);
  *output = *input0;
  Expression expr(Symbol(0));
  for (const auto &sym : *symbol_value) {
    expr = expr + sym;
  }
  GE_ASSERT_TRUE(!output->GetDims().empty());
  output->MutableDim(0) = expr;
  return ge::SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Repeat).InferSymbolShape(TestRepeatInferSymbolShapeFunc);
}  // namespace
TEST_F(SymbolizeValueST, test_symbolize_value_and_repeat_infer) {
  // dlog_setlevel(0, 0, 0);
  auto data0 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .TensorDesc(FORMAT_ND, DT_INT32, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .TensorDesc(FORMAT_ND, DT_INT64, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data2");
  auto data3 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 3)
                   .TensorDesc(FORMAT_ND, DT_UINT32, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data3");
  auto data4 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 4)
                   .TensorDesc(FORMAT_ND, DT_UINT64, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data4");

  auto repeat1 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat1");
  auto repeat2 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_INT64, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat2");
  auto repeat3 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_UINT32, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat3");
  auto repeat4 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_UINT64, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat4");
  auto abs =
      OP_CFG("Abs").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(1).OutCnt(1).OutNames({"y"}).Build("abs");
  auto add =
      OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("add");
  auto mul =
      OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("mul");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat2));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat3));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat4));

    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(repeat2));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(repeat3));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(repeat4));

    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(repeat2)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(repeat3)->EDGE(0, 0)->NODE(mul));
    CHAIN(NODE(repeat4)->EDGE(0, 1)->NODE(mul));

    CHAIN(NODE(add)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(mul)->EDGE(0, 1)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({16, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  GeTensor tensor1(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_INT32));
  vector<int32_t> data_int32(16, 1);
  tensor1.SetData(reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));

  GeTensor tensor2(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_INT64));
  vector<int64_t> data_int64(16, 1);
  tensor2.SetData(reinterpret_cast<uint8_t *>(data_int64.data()), data_int64.size() * sizeof(int64_t));

  GeTensor tensor3(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_UINT32));
  vector<uint32_t> data_uint32(16, 2);
  tensor3.SetData(reinterpret_cast<uint8_t *>(data_uint32.data()), data_uint32.size() * sizeof(uint32_t));

  GeTensor tensor4(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_UINT64));
  vector<uint64_t> data_uint64(16, 2);
  tensor4.SetData(reinterpret_cast<uint8_t *>(data_uint64.data()), data_uint64.size() * sizeof(uint64_t));

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, {tensor0, tensor1, tensor2, tensor3, tensor4}), ge::GRAPH_SUCCESS);

  auto shape_env = graph->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);  // 此处是为了用例最后校验hint，需要保证有shape_env

  auto repeat1_node = graph->FindNode("repeat1");
  ASSERT_NE(repeat1_node, nullptr);
  auto op_desc1 = repeat1_node->GetOpDesc();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = op_desc1->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape().GetDim(0);

  int64_t hint = -1;
  EXPECT_EQ(symbol_expr1.GetHint(hint), true);
  EXPECT_EQ(hint, 16);

  auto mul_node = graph->FindNode("mul");
  ASSERT_NE(mul_node, nullptr);
  auto opdesc3 = mul_node->GetOpDesc();
  ASSERT_NE(opdesc3, nullptr);
  auto attr3 = opdesc3->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr3, nullptr);
  auto symbol_expr3 = attr3->symbolic_tensor.GetOriginSymbolShape().GetDim(0);
  EXPECT_EQ(symbol_expr3.GetHint(hint), true);
  EXPECT_EQ(hint, 16 * 2);
}
}  // namespace ge