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
#include "graph/utils/graph_utils.h"
#include "es_ge_test_ops_c.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "common/env_path.h"
#include "common/topo_checker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/attr_utils.h"
#include "common/share_graph.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/symbolizer/guard_dfx_context.h"

namespace ge {
class GuardCodeGenUT : public testing::Test {
public:
  void SetUp() override {
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
    graph_ = EsCreateGraphBuilder("Hello");
  }

  void TearDown() override {
    EsDestroyGraphBuilder(graph_);
    graph_ = nullptr;
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("LD_LIBRARY_PATH");
  }

  EsCGraphBuilder *graph_{nullptr};
};
int memfd_create(const char *name, unsigned int flags) {
  return syscall(__NR_memfd_create, name, flags);
}
TEST_F(GuardCodeGenUT, GenGuardCodeAndSimpleTest) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  auto symbol3 = attr->CreateSymbol(7, MakeShared<InputShapeSource>(1, 0));
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 1));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 2));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 3));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_NE(symbol1, symbol2);
  EXPECT_SYMBOL_LE(sym::Max(symbol0, symbol5), sym::Min(symbol6, Symbol(8)));
  EXPECT_SYMBOL_LT(sym::Pow(symbol1, sym::Rational(1, 2)), symbol3);
  EXPECT_SYMBOL_GE(sym::Pow(symbol4 + symbol5, Symbol(2)), sym::Ceiling(symbol3));
  EXPECT_SYMBOL_GT(sym::Abs(symbol0 - symbol6), sym::Log(Symbol(1)));

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

  gert::Tensor tensor0 = {{{3, 4, 5}, {3, 4, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{7, 3, 4, 5}, {7, 3, 4, 5}},                      // shape
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
  inputs[0]->MutableOriginShape().SetDim(1, 81);
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  EXPECT_EQ(ret, false);
  std::string reason1(reason);
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectLt(Sqrt(s1), s3)"), std::string::npos);
  close(so_fd);
  dlclose(handle);
}

TEST_F(GuardCodeGenUT, GenGuardCodeWithValue) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(12, MakeShared<InputValueSumSource>(0, DT_INT32));
  auto symbol1 = attr->CreateSymbol(15, MakeShared<InputValueSumSource>(1, DT_INT32));
  auto symbol2 = attr->CreateSymbol(18, MakeShared<InputValueSumSource>(2, DT_INT32));

  auto symbol3 = attr->CreateSymbol(7, MakeShared<InputShapeSource>(3, 0));
  auto symbol4 = attr->CreateSymbol(12, MakeShared<InputShapeSource>(3, 1));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(3, 2));
  auto symbol6 = attr->CreateSymbol(20, MakeShared<InputShapeSource>(3, 3));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_NE(symbol1, symbol2);
  EXPECT_SYMBOL_LE(sym::Min(symbol6, Symbol(8)), sym::Max(symbol0, symbol1));
  EXPECT_SYMBOL_LT(sym::Pow(symbol1, sym::Rational(1, 2)), symbol3);
  EXPECT_SYMBOL_GE(sym::Pow(symbol4 + symbol5, Symbol(2)), sym::Ceiling(symbol3));
  EXPECT_SYMBOL_GT(sym::Abs(symbol0 - symbol6), sym::Log(Symbol(1)));

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
  std::vector<int32_t> data_value0{3, 4, 5};
  std::vector<int32_t> data_value1{4, 5, 6};
  std::vector<int32_t> data_value2{5, 6, 7};

  gert::Tensor tensor0 = {{{3}, {3}},                          // shape
                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                          gert::kOnHost,                       // placement
                          ge::DT_INT32,                        // data type
                          reinterpret_cast<void *>(data_value0.data()), nullptr};
  gert::Tensor tensor1 = {{{3}, {3}},                          // shape
                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                          gert::kOnHost,                       // placement
                          ge::DT_INT32,                        // data type
                          reinterpret_cast<void *>(data_value1.data()), nullptr};
  gert::Tensor tensor2 = {{{3}, {3}},                          // shape
                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                          gert::kOnHost,                       // placement
                          ge::DT_INT32,                        // data type
                          reinterpret_cast<void *>(data_value2.data()), nullptr};
  gert::Tensor tensor3 = {{{7, 12, 4, 20}, {7, 12, 4, 20}},                // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0, nullptr};
  std::vector<gert::Tensor*> inputs;
  inputs.emplace_back(&tensor0);
  inputs.emplace_back(&tensor1);
  inputs.emplace_back(&tensor2);
  inputs.emplace_back(&tensor3);
  char_t reason[1024];
  bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 4, reason, 1024);
  EXPECT_EQ(ret, true);
  // 传入非法参数，校验返回值失败
  std::vector<int32_t> data_value_invalid{12, 4, 5};
  inputs[0]->SetData(gert::TensorData(data_value_invalid.data(), nullptr, 12, gert::kOnHost));
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 4, reason, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  EXPECT_EQ(ret, false);
  std::string reason1(reason);
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectEq(s0, s4)"), std::string::npos);
  close(so_fd);
  dlclose(handle);
}
TEST_F(GuardCodeGenUT, GenGuardCodeFloat) {
  // dlog_setlevel(0, 1, 0);
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  EXPECT_SYMBOL_EQ(Symbol(28), sym::Floor(Symbol(84) / Symbol(3)) * sym::Rational(1, 3) * symbol0);

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
  std::vector<int32_t> data_value0{3, 4, 5};

  gert::Tensor tensor0 = {{{3}, {3}},                          // shape
                          {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                          gert::kOnHost,                       // placement
                          ge::DT_INT32,                        // data type
                          reinterpret_cast<void *>(data_value0.data()), nullptr};

  std::vector<gert::Tensor*> inputs;
  inputs.emplace_back(&tensor0);
  char_t reason[1024];
  bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 4, reason, 1024);
  EXPECT_EQ(ret, true);
  close(so_fd);
  dlclose(handle);
}

TEST_F(GuardCodeGenUT, GenGuardCodeWithInvalidIncludePath) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
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
  setenv("ASCEND_OPP_PATH", "bash;", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  setenv("ASCEND_OPP_PATH", "\\", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  setenv("ASCEND_OPP_PATH", "/usr/local/../sal/da/", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  // 默认会带有../include，此时由于RealPath打桩了，不会实际调用realpath，路径中包含.. ，所以也是非法的
  setenv("ASCEND_OPP_PATH", "/usr/local/./sal/da/", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  setenv("ASCEND_OPP_PATH", "/usr/local//sal/da/", 1);
  EXPECT_NE(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  // 切换回RealPath实际实现，就不会有..，所以路径是合法的
  MmpaStub::GetInstance().SetImpl(std::make_shared<MmpaStubApiGe>());
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  unsetenv("ASCEND_OPP_PATH");
}

TEST_F(GuardCodeGenUT, Guard_Miss_Has_Dfx_Info_When_Set_Guard_Context) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  GuardDfxContext dfx_context("Test Dfx For UT");
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  auto symbol3 = attr->CreateSymbol(7, MakeShared<InputShapeSource>(1, 0));
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 1));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 2));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 3));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);

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

  gert::Tensor tensor0 = {{{2, 4, 5}, {2, 4, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{7, 3, 4, 5}, {7, 3, 4, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor *> inputs;
  inputs.emplace_back(&tensor0);
  inputs.emplace_back(&tensor1);
  char_t reason[1024];
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  
  EXPECT_EQ(ret, false);
  std::string reason1(reason);
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectEq(s0, s4)"), std::string::npos);
  EXPECT_NE(reason1.find("Context Info: Test Dfx For UT missed"), std::string::npos);
  close(so_fd);
  dlclose(handle);
}

TEST_F(GuardCodeGenUT, Guard_Miss_Has_No_Dfx_Info_When_Clear_Guard_Context) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);

  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  auto symbol3 = attr->CreateSymbol(7, MakeShared<InputShapeSource>(1, 0));
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 1));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 2));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 3));
  {
    GuardDfxContext dfx_context("Test Dfx For UT");
    EXPECT_SYMBOL_EQ(symbol0, symbol4);
  }
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
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

  gert::Tensor tensor0 = {{{3, 4, 5}, {3, 4, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{7, 3, 3, 5}, {7, 3, 3, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor *> inputs;
  inputs.emplace_back(&tensor0);
  inputs.emplace_back(&tensor1);
  char_t reason[1024];
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  
  EXPECT_EQ(ret, false);
  std::string reason1(reason);
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectEq(s1, s5)"), std::string::npos);
  EXPECT_EQ(reason1.find("Context Info: Test Dfx For UT missed"), std::string::npos);

  inputs[0]->MutableOriginShape().SetDim(0, 81); // 构造第一个guard失败，仍然有dfx
  char_t reason0[1024];
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason0, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  EXPECT_EQ(ret, false);
  std::string reason2(reason0);
  EXPECT_NE(reason2.find("Check Symbol Check Expression: ExpectEq(s0, s4)"), std::string::npos);
  EXPECT_NE(reason2.find("Context Info: Test Dfx For UT missed"), std::string::npos);

  close(so_fd);
  dlclose(handle);
}

TEST_F(GuardCodeGenUT, Guard_Miss_Has_New_Dfx_Info_When_Set_Guard_Context_Twice) {
  GuardCodegen codegen;
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  auto attr = compute_graph->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  GuardDfxContext dfx_context("Test Dfx For UT");
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  auto symbol3 = attr->CreateSymbol(7, MakeShared<InputShapeSource>(1, 0));
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 1));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 2));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 3));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);

  GuardDfxContext dfx_context1("Test Dfx For UT Second");
  EXPECT_SYMBOL_EQ(symbol1, symbol5);

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

  gert::Tensor tensor0 = {{{2, 4, 5}, {2, 4, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{7, 3, 4, 5}, {7, 3, 4, 5}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor *> inputs;
  inputs.emplace_back(&tensor0);
  inputs.emplace_back(&tensor1);
  char_t reason[1024];
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  
  EXPECT_EQ(ret, false);
  std::string reason1(reason);
  EXPECT_NE(reason1.find("Check Symbol Check Expression: ExpectEq(s0, s4)"), std::string::npos);
  EXPECT_NE(reason1.find("Context Info: Test Dfx For UT missed"), std::string::npos);

  inputs[0]->MutableOriginShape().SetDim(0, 3);
  inputs[0]->MutableOriginShape().SetDim(1, 3); // 构造第二个guard miss
  char_t reason0[1024];
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason0, 1024);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  EXPECT_EQ(ret, false);
  std::string reason2(reason0);
  EXPECT_NE(reason2.find("Check Symbol Check Expression: ExpectEq(s1, s5)"), std::string::npos);
  EXPECT_NE(reason2.find("Context Info: Test Dfx For UT Second missed"), std::string::npos);

  close(so_fd);
  dlclose(handle);
}
}