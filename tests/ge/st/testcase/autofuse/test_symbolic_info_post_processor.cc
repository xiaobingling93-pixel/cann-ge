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
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
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
#include "common/omg_util/omg_util.h"
#include "mmpa/mmpa_api.h"
#include "graph/node.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_post_processor.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h"
#include "graph/operator_reg.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"

namespace ge {
namespace {
graphStatus InferShape4FooTestGuard(gert::InferSymbolShapeContext *context) {
  auto shape0 = context->GetInputSymbolShape(0);
  GE_ASSERT_NOTNULL(shape0);
  auto shape1 = context->GetInputSymbolShape(1);
  GE_ASSERT_NOTNULL(shape1);
  GE_ASSERT_TRUE(shape0->GetDimNum() == shape1->GetDimNum());
  for (size_t i = 0U; i < shape0->GetDimNum(); ++i) {
    if (EXPECT_SYMBOL_EQ(shape0->GetDim(i), shape1->GetDim(i))) {
      context->GetOutputSymbolShape(0)->AppendDim(shape0->GetDim(i));
    } else if (EXPECT_SYMBOL_EQ(shape0->GetDim(i), Symbol(1))) {
      context->GetOutputSymbolShape(0)->AppendDim(shape1->GetDim(i));
    } else if (EXPECT_SYMBOL_EQ(shape1->GetDim(i), Symbol(1))) {
      context->GetOutputSymbolShape(0)->AppendDim(shape0->GetDim(i));
    } else {
      return GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FooTestGuard).InferSymbolShape(InferShape4FooTestGuard);

int memfd_create(const char *name, unsigned int flags) {
  return syscall(__NR_memfd_create, name, flags);
}

void CheckGuardCodegen(ComputeGraphPtr &compute_graph) {
  // 检查生成的代码
  std::string code;
  AttrUtils::GetStr(compute_graph, "_guard_check_func", code);
  std::string buffer;
  AttrUtils::GetStr(compute_graph, "_guard_check_so_data", buffer);

  int so_fd = memfd_create("libdemo.so", 0);
  write(so_fd, buffer.c_str(), buffer.size());

  char so_path[128];
  // 通过/proc访问文件描述符对应的"文件"
  snprintf_s(so_path, sizeof(so_path), sizeof(so_path), "/proc/self/fd/%d", so_fd);

  void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr);
  // 传入合法参数，校验返回值成功
  void* func = dlsym(handle, "GuardCheckFunc");
  gert::Tensor tensor0 = {{{1000, 1000, 1, 1000}, {3, 1000, 1, 1000}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{1000, 1, 1000, 1000}, {3, 1, 1000, 1000}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor *> inputs;
  inputs.emplace_back(&tensor0);
  inputs.emplace_back(&tensor1);
  char_t reason[1024];
  bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  EXPECT_EQ(ret, true);
  // 传入非法参数，校验返回值失败
  inputs[1]->MutableOriginShape().SetDim(0, 10);
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  EXPECT_EQ(ret, false);
  std::string reason1(reason);
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectEq(s0, s4)"), std::string::npos);
  close(so_fd);
  dlclose(handle);
}
}
class SymbolicInfoPostProcessorST : public testing::Test {
 public:
  void SetUp() override {
    gert::LoadDefaultSpaceRegistry();
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env_, MMPA_MAX_PATH);
    (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env_, MMPA_MAX_PATH);
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
    graph_ = EsCreateGraphBuilder("Hello");
    env = getenv("LD_PRELOAD");
    unsetenv("LD_PRELOAD");
  }

  void TearDown() override {
    gert::UnLoadDefaultSpaceRegistry();
    EsDestroyGraphBuilder(graph_);
    graph_ = nullptr;
    unsetenv("LD_LIBRARY_PATH");
    mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env_, 1);
    mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env_, 1);
    if (env != nullptr) {
      setenv("LD_PRELOAD", env, 1);
    }
  }
 protected:
  EsCGraphBuilder *graph_{nullptr};
 private:
  char old_opp_path_env_[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env_[MMPA_MAX_PATH] = {'\0'};
  const char* env;
};
REG_OP(FooTestGuard)
    .INPUT(x1, TensorType::NumberType())
        .INPUT(x2, TensorType::NumberType())
        .OUTPUT(y, TensorType::NumberType())
        .OP_END_FACTORY_REG(FooTestGuard);

TEST_F(SymbolicInfoPostProcessorST, run_test) {
  auto data1 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 0)
      .InCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .Build("data1");
  auto data2 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 1)
      .InCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .Build("data2");
  auto foo1 = OP_CFG("FooTestGuard")
      .InCnt(2)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .Build("foo1");

  DEF_GRAPH(g_test_guard) {
                            CHAIN(NODE(data1)->NODE(foo1)->NODE("NetOutput", "NetOutput"));
                            CHAIN(NODE(data2)->NODE(foo1)->NODE("NetOutput", "NetOutput"));
                          };
  auto compute_graph = ToComputeGraph(g_test_guard);
  GeShape shape1({3, 4, 1, 7});
  GeTensor ge_tensor1;
  ge_tensor1.MutableTensorDesc().MutableOriginShape() = shape1;
  ge_tensor1.MutableTensorDesc().MutableShape() = shape1;

  GeShape shape2({3, 1, 5, 7});
  GeTensor ge_tensor2;
  ge_tensor2.MutableTensorDesc().MutableOriginShape() = shape2;
  ge_tensor2.MutableTensorDesc().MutableShape() = shape2;
  SymbolicShapeSymbolizer::Symbolize(compute_graph, {ge_tensor1, ge_tensor2});

  AutoFusePass pass;
  EXPECT_EQ(pass.Run(compute_graph), ge::SUCCESS);

  ASSERT_EQ(SymbolicInfoPostProcessor::Run(compute_graph), SUCCESS);

  CheckGuardCodegen(compute_graph);

  auto op_desc = compute_graph->FindNode("foo1")->GetOpDesc();
  std::string infer_key;
  (void)AttrUtils::GetStr(op_desc, "_symbol_infer_shape_merge_key", infer_key);
  ASSERT_EQ(infer_key, "[s4_s1_s6_s7][]");

  size_t all_sym_num;
  (void)AttrUtils::GetInt(compute_graph, "_all_symbol_num", all_sym_num);
  ASSERT_EQ(all_sym_num, 8);
}
}