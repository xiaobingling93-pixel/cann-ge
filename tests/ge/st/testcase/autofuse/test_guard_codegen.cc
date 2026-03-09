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
#include "attribute_group/attr_group_shape_env.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h"
#include "graph/operator_reg.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "depends/mmpa/src/mmpa_stub.h"

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
}
class GuardCodeGenST : public testing::Test {
 public:
  void SetUp() override {
    env = getenv("LD_PRELOAD");
    unsetenv("LD_PRELOAD");
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env_, MMPA_MAX_PATH);
    (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env_, MMPA_MAX_PATH);
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
    graph_ = EsCreateGraphBuilder("Hello");
  }

  void TearDown() override {
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
  const char *env;
};
int memfd_create(const char *name, unsigned int flags) {
  return syscall(__NR_memfd_create, name, flags);
}
REG_OP(FooTestGuard)
    .INPUT(x1, TensorType::NumberType())
    .INPUT(x2, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(FooTestGuard);

TEST_F(GuardCodeGenST, GenGuardCodeAndSimpleTest1) {
  dlog_setlevel(0, 1, 0);
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

  GuardCodegen codegen;
  ASSERT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), SUCCESS);
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
  dlog_setlevel(0, 3, 0);
}

TEST_F(GuardCodeGenST, GenGuardCodeAndSimpleTest) {
  GuardCodegen codegen;
  std::unique_ptr<Graph> graph(reinterpret_cast<Graph*>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  // 产生guard
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  // 检查生成的代码
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

  gert::Tensor tensor0 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
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
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectEq(s0, s3)"), std::string::npos);
  close(so_fd);
  dlclose(handle);
}

TEST_F(GuardCodeGenST, GenGuardCodeWithInvalidIncludePath) {
  GuardCodegen codegen;
  std::unique_ptr<Graph> graph(reinterpret_cast<Graph*>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);

  class MockMmpaRealPath : public ge::MmpaStubApiGe {
  public:
    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      // 1. 检查输入参数合法性
      if (path == nullptr || realPath == nullptr || realPathLen <= 0) {
        return EN_INVALID_PARAM; // 假设 EN_INVALID_PARAM 是预定义错误码
      }

      // 2. 计算 path 的长度（不含终止符）
      size_t path_len = strlen(path);

      // 3. 检查目标缓冲区是否足够容纳路径（含终止符）
      if (static_cast<size_t>(realPathLen) < path_len + 1) {
        return EN_ERROR; // 缓冲区不足
      }

      // 4. 直接拷贝 path 到 realPath
      strncpy(realPath, path, realPathLen);
      realPath[realPathLen - 1] = '\0'; // 确保终止符

      return EN_OK;
    };
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
  // 根据guard生成代码
  setenv("ASCEND_OPP_PATH", ";", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  setenv("ASCEND_OPP_PATH", "rm -rf ./;", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  setenv("ASCEND_OPP_PATH", "$", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  // 切换回RealPath实际实现，就不会有..，所以路径是合法的
  MmpaStub::GetInstance().SetImpl(std::make_shared<MmpaStubApiGe>());
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  unsetenv("ASCEND_OPP_PATH");
}
}