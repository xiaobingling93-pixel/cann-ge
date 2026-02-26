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
#include "ge_graph_dsl/graph_dsl.h"
#include "es_ge_test_ops.h"
#include "graph/utils/graph_utils_ex.h"
#include "jit_execution/user_graphs_manager.h"
#include "stub/gert_runtime_stub.h"
#include <vector>
#include "jit_share_graph.h"
#include "common/model/external_allocator_manager.h"
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"
#include "ge_running_env/dir_env.h"
#include "faker/space_registry_faker.h"
#include "common_setup.h"
#include "ge/ge_api.h"
#include "api/aclgrph/option_utils.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/execute/model_executor.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"
#include "common/platform_context.h"
using namespace testing;

namespace ge {
bool EnableSliceSchedule() { // 桩函数
  return true;
}
class RuntimeMock : public RuntimeStub {
public:
  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "fake"); // 用例不应该走自动融合
    return RT_ERROR_NONE;
  }
};

class UserGraphsManagerlUT : public testing::Test {
 protected:
  void SetUp() override {
    CommonSetupUtil::CommonSetup();
    gert_stub_.GetKernelStub().StubTiling();
    RuntimeStub::Install(nullptr); // gert的rts stub不能在多线程环境下工作，因此使用默认rts stub
    RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend310"}};
    GetThreadLocalContext().SetGlobalOption(options);
    ge::PlatformContext::GetInstance().SetPlatform("fake");
  }
  void TearDown() override {
    unsetenv("AUTOFUSE_FLAGS");
    CommonSetupUtil::CommonTearDown();
    gert_stub_.Clear();
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
  gert::GertRuntimeStub gert_stub_;
};

TEST_F(UserGraphsManagerlUT, AddGraph_RemoveGraph_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  EXPECT_EQ(user_graph_manager.BuildGraph(user_graph_id, {}, 0), SUCCESS);
  EXPECT_FALSE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}
TEST_F(UserGraphsManagerlUT, AddGraph_Twice_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  EXPECT_EQ(user_graph_manager.BuildGraph(user_graph_id, {}, 0), SUCCESS);
  EXPECT_FALSE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, RemoveGraph_NotExist) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  EXPECT_EQ(user_graph_manager.RemoveGraph(0), FAILED);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, GetSetCompiledFlag_Failed) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  bool flag = false;
  EXPECT_NE(user_graph_manager.GetCompiledFlag(0, flag), SUCCESS);
  EXPECT_NE(user_graph_manager.SetCompiledFlag(0, flag), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, RunGraphAsync_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<gert::Tensor> inputs(1);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    ASSERT_EQ(outputs.size(), 1);
    auto out_shape = outputs[0].GetStorageShape();
    auto out_dims = TensorTransUtils::GetDimsFromGertShape(out_shape);
    EXPECT_EQ(out_dims, shape_dim);
  };
  EXPECT_EQ(user_graph_manager.RunGraphAsync(user_graph_id, std::move(inputs), 0, callback), SUCCESS);
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, IsGraphNeedRebuild_False) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<gert::Tensor> inputs(1);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto out_shape = outputs[0].GetStorageShape();
    auto out_dims = TensorTransUtils::GetDimsFromGertShape(out_shape);
    EXPECT_EQ(out_dims, shape_dim);
    return SUCCESS;
  };
  EXPECT_EQ(user_graph_manager.RunGraphAsync(user_graph_id, std::move(inputs), 0, callback), SUCCESS);
  // graph is built, no need build
  EXPECT_FALSE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));

  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);
  
  // graph is not exist, need rebuild
  EXPECT_TRUE(user_graph_manager.IsGraphNeedRebuild(user_graph_id));
  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphsManagerlUT, ExecuteGraphWithStreamAsync_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

#if 0
TEST_F(UserGraphsManagerlUT, return_compile_load_skip_summary_not_null_execute_success_when_input_dynamic_graph_not_partition) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_skip_summary_not_null_execute_success_when_input_dynamic_graph_contain_partition) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<int64_t> input_data_2{1, 2, 3, 4, 0, 0, 0, 0};
  gert_inputs[1] = {{{4}, {4}},                                  // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT64,                                // data type
                    (void *) input_data_2.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}
#endif

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_execute_success_when_input_static_graph_not_partition) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, {}, {}, nullptr};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_not_null_execute_success_when_input_static_graph_contain_partition) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode({1, 2, 3, 4}, {4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());

  const std::map<std::string, std::string> options;
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<int64_t> input_data_2{1, 2, 3, 4, 0, 0, 0, 0};
  gert_inputs[1] = {{{4}, {4}},                                  // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT64,                                // data type
                    (void *) input_data_2.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_outputs.clear();
  gert_outputs.resize(1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, nullptr, gert_inputs, gert_outputs, 0), SUCCESS); // hint guard no compile
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_load_fail_when_input_static_graph_not_partition_not_compile) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_EQ(summary, nullptr);

  std::map<AscendString, AscendString> load_options;
  EXPECT_NE(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_load_succ_when_input_dynamic_graph_partition_not_compile) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_EQ(summary, nullptr);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, nullptr), SUCCESS);

  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_not_null_execute_success_when_input_static_graph_contain_partition_extern_stream) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::OneReshapeNode({1, 2, 3, 4}, {4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());

  const std::map<std::string, std::string> options;
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), false);
  std::vector<ge::Shape> output_shape;
  EXPECT_NE(summary->GetOutputShapes(output_shape), ge::SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, new_stream), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<int64_t> input_data_2{1, 2, 3, 4, 0, 0, 0, 0};
  gert_inputs[1] = {{{4}, {4}},                                  // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT64,                                // data type
                    (void *) input_data_2.data()};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_outputs.clear();
  gert_outputs.resize(1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs, 0), SUCCESS); // hint guard no compile
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_execute_success_when_input_static_graph_not_partition_extern_stream) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, new_stream), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, {}, {}, nullptr};
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_load_summary_execute_success_when_input_static_graph_not_partition_extern_stream_external_output) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  auto relu1 = compute_graph->FindNode("Relu_1");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes{{relu1, 0}};
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  std::map<AscendString, AscendString> load_options;
  EXPECT_EQ(user_graph_manager.LoadGraph(user_graph_id, load_options, new_stream), SUCCESS);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  std::vector<uint8_t> output_data_1(96, 0xFF);
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                     {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                     gert::kOnDeviceHbm,                                // placement
                     ge::DT_INT32,                              // data type
                     (void *) output_data_1.data()};
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs, 0), SUCCESS);
  EXPECT_EQ(gert_outputs.size(), 1);
  EXPECT_EQ(gert_outputs[0].GetOriginShape(), gert::Shape({1, 2, 3, 4}));

  gert_inputs.clear();
  gert_outputs.clear();
  graph_manager.UnregisterExternalAllocator(new_stream);
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, return_compile_summary_execute_success_when_input_static_graph_not_partition_extern_stream) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  UserGraphsManager user_graph_manager(graph_manager);
  rtStream_t new_stream;
  (void)rtStreamCreate(&new_stream, 0);

  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes({1, 2, 3, 4});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  const std::map<std::string, std::string> options;
  EXPECT_EQ(user_graph_manager.AddGraph(user_graph_id, *graph, options), SUCCESS);
  
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(user_graph_manager.CompileGraph(user_graph_id, 0, {}), SUCCESS);
  
  CompiledGraphSummaryPtr summary;
  EXPECT_EQ(user_graph_manager.GetCompiledGraphSummary(user_graph_id, summary), SUCCESS);
  EXPECT_NE(summary, nullptr);
  // static shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{1, 2, 3, 4};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);

  // prepare run task
  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(1);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *) input_data_1.data()};
  gert_outputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, {}, {}, nullptr};
  EXPECT_NE(user_graph_manager.ExecuteGraphWithStreamAsync(user_graph_id, new_stream, gert_inputs, gert_outputs, 0), SUCCESS); // 未load报错
 
  gert_inputs.clear();
  gert_outputs.clear();
  EXPECT_EQ(user_graph_manager.RemoveGraph(user_graph_id), SUCCESS);

  EXPECT_EQ(user_graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  rtStreamDestroy(new_stream);
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
}

TEST_F(UserGraphsManagerlUT, set_memory_skip_by_slice_scheduler_enable) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1); // 开启自动融合
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  dlog_setlevel(GE_MODULE_NAME, 0, 1);
  EXPECT_EQ(UNSUPPORTED, session.SetGraphConstMemoryBase(graph_id, nullptr, 0));
  EXPECT_EQ(UNSUPPORTED, session.UpdateGraphFeatureMemoryBase(graph_id, nullptr, 0));
  EXPECT_EQ(UNSUPPORTED, session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0));
  EXPECT_EQ(UNSUPPORTED, session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0));

  std::vector<std::string> expect_log_list = {
    "SetGraphConstMemoryBase unsupport slice scheduler currently",
    "UpdateGraphFeatureMemoryBase unsupport slice scheduler currently",
    "SetGraphFixedFeatureMemoryBaseWithType unsupport slice scheduler currently",
    "UpdateGraphRefreshableFeatureMemoryBase unsupport slice scheduler currently"
  };
  for (auto &it : expect_log_list) {
    EXPECT_NE(gert_stub_.GetSlogStub().FindLog(-1, it.c_str()), -1);
  }
  dlog_setlevel(GE_MODULE_NAME, 3, 1);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
}
}  // namespace ge