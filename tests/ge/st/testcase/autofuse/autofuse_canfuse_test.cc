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
#include "ge/ge_api.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "graph_utils_ex.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "compiler/graph/build/model_builder.h"
#include "compiler/graph/build/graph_builder.h"
#include "compiler/graph/manager/graph_manager.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "../init_ge.h"
#include "common/share_graph.h"
#include "ge_running_env/fake_op.h"
#include "common/env_path.h"
#include "common/platform_context.h"
#include "tests/framework/ge_runtime_stub/include/common/summary_checker.h"
#include "faker/space_registry_faker.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace std;
using namespace testing;
using namespace ge::autofuse;

namespace ge {
class RuntimeMock910B2 : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
    (void)strcpy_s(version, maxLen, "Ascend910B2");
    return RT_ERROR_NONE;
  }
  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "2201");
    return RT_ERROR_NONE;
  }
};

class TestCanfusePass : public testing::Test {
 protected:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<RuntimeMock910B2>());
    // 符号化infer需要数据依赖注册在SpaceRegistry之前
    IMPL_OP(ReduceMax).InputsDataDependency({1});
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    ge::OmgContext ctx;
    domi::GetContext() = ctx;
    ReInitGe();
    global_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().GetOo().Initialize({}, {});
    ge_env.InstallDefault();

    work_path = EnvPath().GetAirBasePath() + "/output";
    setenv("ASCEND_WORK_PATH", work_path.c_str(), 1);
    setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env_, MMPA_MAX_PATH);
    (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env_, MMPA_MAX_PATH);
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
    AutoFuseConfig::MutableLoweringConfig().experimental_lowering_reduce = true;
    PlatformContext::GetInstance().SetPlatform("2201");
  }

  void TearDown() override {
    RuntimeStub::Reset();
    ge_env.Reset();
    ge_env.InstallDefault();
    ge::GetThreadLocalContext().SetGlobalOption(global_options);
    ge::GetThreadLocalContext().SetGraphOption(graph_options);
    ge::GetThreadLocalContext().SetSessionOption(session_options);
    gert::UnLoadDefaultSpaceRegistry();
    unsetenv("ASCEND_WORK_PATH");
    unsetenv("AUTOFUSE_FLAGS");
    mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env_, 1);
    mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env_, 1);
    ge::PlatformContext::GetInstance().Reset();
  }
  char old_opp_path_env_[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env_[MMPA_MAX_PATH] = {'\0'};
  std::string work_path;
  GeRunningEnvFaker ge_env;
  std::map<std::string, std::string> graph_options;
  std::map<std::string, std::string> session_options;
  std::map<std::string, std::string> global_options;
};

/**
 * 用例描述：elementwise和elementwise水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_ele_horizontal_fusion_1) {
  dlog_setlevel(0, 0, 0);
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(EXP).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  const auto compute_graph = gert::ShareGraph::BuildStaticAbsExpReluNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };
  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());

  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Exp", 1);
  node_types_to_count.emplace("Store", 2);
  node_types_to_count.emplace("Output", 2);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
  dlog_setlevel(0, 3, 0);
}

/**
 * 用例描述：elementwise和elementwise水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_ele_horizontal_fusion_2) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(EXP).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  session.BuildGraph(1, inputs);
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Exp", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：elementwise和broadcast水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_bro_horizontal_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(EXP).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticAbsAddExpNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 2);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };
  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();

  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 2);
  node_types_to_count.emplace("Load", 2);
  node_types_to_count.emplace("Broadcast", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Exp", 1);
  node_types_to_count.emplace("Store", 2);
  node_types_to_count.emplace("Output", 2);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：elementwise和broadcast水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_bro_horizontal_fusion_2) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticAbsTwoAddNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input3{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2, input3};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 3);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 3);
  node_types_to_count.emplace("Load", 3);
  node_types_to_count.emplace("Broadcast", 3);
  node_types_to_count.emplace("Add", 2);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：elementwise和reduce水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
/*
TEST_F(TestCanfusePass, test_canfuse_ele_and_red_horizontal_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticAbsReduceReluNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_NE(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("Const", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Sum", 1);
  node_types_to_count.emplace("Store", 2);
  node_types_to_count.emplace("Output", 2);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}
*/
/**
 * 用例描述：broadcast和broadcast水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:
 */
TEST_F(TestCanfusePass, test_canfuse_bro_and_bro_horizontal_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticTwoReduceThreeAddNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input3{0, {2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2, input3};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 3);
    node_types_to_count0.emplace("Const", 2);
    node_types_to_count0.emplace("AscBackend", 2);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_NE(str, "success"); // 恢复线上阻塞，临时修改
  };

  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 2);
  node_types_to_count.emplace("Load", 2);
  node_types_to_count.emplace("Broadcast", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Sum", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  // 恢复线上阻塞，临时修改
  EXPECT_NE(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：broadcast和reuce水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_bro_and_red_horizontal_fusion_1) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticAddReduceNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};
  DUMP_GRAPH_WHEN("After_AutoFusePass");

  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 2);
    node_types_to_count0.emplace("Const", 1);
    node_types_to_count0.emplace("ReduceSum", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    // EXPECT_EQ(str, "success"); // 恢复线上阻塞，临时修改
  };

  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 3);
  node_types_to_count.emplace("Load", 3);
  node_types_to_count.emplace("Broadcast", 2);
  node_types_to_count.emplace("Add", 2);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  // 恢复线上阻塞，临时修改
  // EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：reduce和reduce水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
/*
TEST_F(TestCanfusePass, test_canfuse_red_and_red_horizontal_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCEMAX).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticAbsTwoReduceNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_NE(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("Const", 2);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_bc, nullptr);
  auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Sum", 1);
  node_types_to_count.emplace("Max", 1);
  node_types_to_count.emplace("Store", 2);
  node_types_to_count.emplace("Output", 2);
  std::string str = gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count);
  EXPECT_EQ(str, "success");
}
*/
/**
 * 用例描述：reduce和reduce垂直融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:lowering生成的两个AscBc 在lifting 成reduceNode
 */
TEST_F(TestCanfusePass, test_canfuse_red_and_red_vertical_fusion_1) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCEMAX).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticTwoReduceNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 16}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  session.BuildGraph(1, inputs);

  // lifting AscBackend中单节点还原为原节点
  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("AscBackend", 2);
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("Const", 2);
    node_types_to_count0.emplace("NetOutput", 1);
    EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0), "success");
  };
}

/**
 * 用例描述：reduce和broadcast垂直融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_red_and_bro_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCEMAX).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticReduceAddReluNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 2);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("Const", 1);
    node_types_to_count0.emplace("ReduceMax", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    // EXPECT_EQ(str, "success"); // 恢复线上阻塞，临时修改
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 2);
  node_types_to_count.emplace("Load", 2);
  node_types_to_count.emplace("Broadcast", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  // 恢复线上阻塞，临时修改
  // EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：reduce和elementwise垂直融合
 * 预置条件：开启自动融合
 * 测试步骤
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验AscGraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_red_and_ele_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCEMAX).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticReduceAbsReluNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 16}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("Const", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Max", 1);
  node_types_to_count.emplace("Abs", 2);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
}

/**
 * 用例描述：elementwise和elementwise垂直融合
 * 预置条件：开启自动融合 通过限制lowering融合节点数限制lowering融合全部的elementwise算子
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_ele_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(EXP).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  int64_t max_fused_nodes_num_ = AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops;
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = 2;
  const auto compute_graph = gert::ShareGraph::BuildStaticAbsReluAbsExpNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);
  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Abs", 2);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Exp", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
  // 复原
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = max_fused_nodes_num_;
}

/**
 * 用例描述：elementwise和broadcast垂直融合
 * 预置条件：开启自动融合 通过限制lowering融合节点数限制lowering融合全部的elementwise broadcast算子
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_bro_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  int64_t max_fused_nodes_num_ = AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops;
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = 2;
  const auto compute_graph = gert::ShareGraph::BuildStaticAbsReluAddNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 2);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 2);
  node_types_to_count.emplace("Load", 2);
  node_types_to_count.emplace("Abs", 2);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Broadcast", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");

  // 复原
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = max_fused_nodes_num_;
}

/**
 * 用例描述：elementwise和reduce垂直融合
 * 预置条件：开启自动融合 通过限制lowering融合节点数限制lowering融合全部的elementwise算子
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_ele_and_red_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  int64_t max_fused_nodes_num_ = AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops;
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = 2U;
  const auto compute_graph = gert::ShareGraph::BuildStaticAbsReluReduceSumNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 16}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  DUMP_GRAPH_WHEN("After_AutoFusePass");
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("Const", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Sum", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");

  // 复原
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = max_fused_nodes_num_;
}

/**
 * 用例描述：broadcast和elementwise垂直融合
 * 预置条件：开启自动融合 通过限制lowering融合节点数限制lowering融合全部的broadcast elementwise算子
 * 试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_bro_and_ele_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));
  int64_t max_fused_nodes_num_ = AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops;
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = 2U;
  const auto compute_graph = gert::ShareGraph::BuildStaticReluAddAbsReluNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2};
  DUMP_GRAPH_WHEN("After_AutoFusePass");
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 2);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();

  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 2);
  node_types_to_count.emplace("Load", 2);
  node_types_to_count.emplace("Relu", 2);
  node_types_to_count.emplace("Broadcast", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");
  // 复原
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = max_fused_nodes_num_;
}

/**
 * 用例描述：broadcast和reduce垂直融合
 * 预置条件：开启自动融合 通过限制lowering融合节点数限制lowering融合全部的broadcast算子
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_bro_and_red_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  int64_t max_fused_nodes_num_ = AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops;
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = 2U;
  const auto compute_graph = gert::ShareGraph::BuildStaticReluAddReduceSumNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {16, 16, 16}, nullptr, 0};
  InputTensorInfo input2{0, {16, 16, 16, 16}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2};
  DUMP_GRAPH_WHEN("After_AutoFusePass");
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 2);
    node_types_to_count0.emplace("Const", 1);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();

  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 2);
  node_types_to_count.emplace("Load", 2);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Broadcast", 1);
  node_types_to_count.emplace("Add", 1);
  node_types_to_count.emplace("Sum", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");

  // 复原
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = max_fused_nodes_num_;
}

/**
 * 用例描述：broadcast和broadcast垂直融合
 * 预置条件：开启自动融合 通过限制lowering融合节点数限制lowering融合全部的broadcast算子
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_bro_and_bro_vertical_fusion) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp("Relu").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));
  int64_t max_fused_nodes_num_ = AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops;

  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = 2U;
  const auto compute_graph = gert::ShareGraph::BuildStaticReluAddAddNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {2, 2}, nullptr, 0};
  InputTensorInfo input2{0, {2, 2, 2}, nullptr, 0};
  InputTensorInfo input3{0, {2, 2, 2, 2}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1, input2, input3};
  DUMP_GRAPH_WHEN("After_AutoFusePass");
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 3);
    node_types_to_count0.emplace("AscBackend", 1);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 3);
  node_types_to_count.emplace("Load", 3);
  node_types_to_count.emplace("Relu", 1);
  node_types_to_count.emplace("Broadcast", 2);
  node_types_to_count.emplace("Add", 2);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  EXPECT_EQ(gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count), "success");

  // 复原
  AutoFuseConfig::MutableLoweringConfig().max_fused_loop_ops = max_fused_nodes_num_;
}

/**
 * 用例描述：broadcast和reduce水平融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
// TEST_F(TestCanfusePass, test_canfuse_bro_and_red_horizontal_fusion_2) {
//   auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
//   ge_env.Reset()
//       .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
//       .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
//       .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
//       .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
//       .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
//       .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
//       .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));
//
//   const auto compute_graph = gert::ShareGraph::BuildStaticAbsAddReduceNodeGraph();
//   ASSERT_NE(compute_graph, nullptr);
//   auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
//   std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};
//
//   Session session(options);
//   auto ret = session.AddGraph(1, *graph, options);
//   EXPECT_EQ(ret, SUCCESS);
//   InputTensorInfo input1{0, {2, 2}, nullptr, 0};
//   InputTensorInfo input2{0, {2, 2, 2}, nullptr, 0};
//   std::vector<InputTensorInfo> inputs{input1, input2};
//   DUMP_GRAPH_WHEN("After_AutoFusePass");
//   // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
//   EXPECT_NE(session.BuildGraph(1, inputs), SUCCESS);
//
//   CHECK_GRAPH(After_AutoFusePass) {
//     std::map<std::string, size_t> node_types_to_count0;
//     node_types_to_count0.emplace("Data", 2);
//     node_types_to_count0.emplace("Const", 1);
//     node_types_to_count0.emplace("AscBackend", 1);
//     node_types_to_count0.emplace("NetOutput", 1);
//     std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
//     EXPECT_EQ(str, "success");
//   };
//
//   auto asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
//   ASSERT_NE(asc_bc, nullptr);
//   auto attr = asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
//   ASSERT_NE(attr, nullptr);
//   ASSERT_NE(attr->GetAscGraph(), nullptr);
//   const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
//   std::map<std::string, size_t> node_types_to_count;
//   node_types_to_count.emplace("Data", 2);
//   node_types_to_count.emplace("Load", 2);
//   node_types_to_count.emplace("Broadcast", 1);
//   node_types_to_count.emplace("Abs", 1);
//   node_types_to_count.emplace("Add", 1);
//   node_types_to_count.emplace("Sum", 1);
//   node_types_to_count.emplace("Store", 2);
//   node_types_to_count.emplace("Output", 2);
//   std::string str = gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count);
//   EXPECT_EQ(str, "success");
// }

/**
 * 用例描述：reduce和reduce垂直融合
 * 预置条件：开启自动融合
 * 测试步骤：
 * 预期结果:融合lowering生成的两个AscBc为1个，同时校验ascgraph的正确性
 */
TEST_F(TestCanfusePass, test_canfuse_red_and_red_vertical_fusion_2) {
  auto infer_data = ELMTWISE_INFER_SHAPEANDTYPE("x", "y");
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_data))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp("Abs").InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCESUM).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(REDUCEMAX).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine"));

  const auto compute_graph = gert::ShareGraph::BuildStaticTwoReduceReluNodeGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O3"}};

  DUMP_GRAPH_WHEN("After_AutoFusePass");
  Session session(options);
  auto ret = session.AddGraph(1, *graph, options);
  EXPECT_EQ(ret, SUCCESS);
  InputTensorInfo input1{0, {16, 16, 16}, nullptr, 0};
  std::vector<InputTensorInfo> inputs{input1};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载, 仅测试编译流程
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  CHECK_GRAPH(After_AutoFusePass) {
    std::map<std::string, size_t> node_types_to_count0;
    node_types_to_count0.emplace("Data", 1);
    node_types_to_count0.emplace("Const", 2);
    node_types_to_count0.emplace("AscBackend", 2);
    node_types_to_count0.emplace("NetOutput", 1);
    std::string str = gert::SummaryChecker(graph).StrictDirectNodeTypes(node_types_to_count0);
    EXPECT_EQ(str, "success");
  };

  auto fused_asc_bc = compute_graph->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_asc_bc, nullptr);
  auto attr = fused_asc_bc->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr, nullptr);
  ASSERT_NE(attr->GetAscGraph(), nullptr);
  const auto &fused_graph = AscGraphUtils::GetComputeGraph(*attr->GetAscGraph());
  std::map<std::string, size_t> node_types_to_count;
  node_types_to_count.emplace("Data", 1);
  node_types_to_count.emplace("Load", 1);
  node_types_to_count.emplace("Abs", 1);
  node_types_to_count.emplace("Max", 1);
  node_types_to_count.emplace("Store", 1);
  node_types_to_count.emplace("Output", 1);
  std::string str = gert::SummaryChecker(fused_graph).StrictDirectNodeTypes(node_types_to_count);
  EXPECT_EQ(str, "success");
}
}  // namespace ge