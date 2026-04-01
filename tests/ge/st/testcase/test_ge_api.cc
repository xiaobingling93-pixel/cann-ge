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

#include "macro_utils/dt_public_scope.h"
#include "common/plugin/ge_make_unique_util.h"
#include "proto/ge_ir.pb.h"
#include "ge/ge_api.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "session/session_manager.h"
#include "init_ge.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "runtime/base.h"
#include "utils/taskdef_builder.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "common/args_checker.h"
#include "init_ge.h"
#include "utils/mock_ops_kernel_builder.h"
#include "register/register_custom_pass.h"
#include "common/global_variables/diagnose_switch.h"
#include "array_ops.h"
#include "common/env_path.h"
#include "common/share_graph.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "operator_reg.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"

using namespace gert;
namespace ge {
namespace {
static FakeOpsKernelInfoStore g_fake_hccl_ops_kernel_info_store;
bool test_callback_called = false;

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *       netoutput
 */
ge::Graph BuildDynamicAddGraph() {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int64_t> shape{-1, -1, 3, 4};
  auto data_1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("data_1");
  auto data_2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("data_2");
  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("add_1");

  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(add_1)->EDGE(0, 0)->NODE(netoutput));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(add_1));
    ADD_OUTPUT(add_1, 0);
  };

  auto graph = ToGeGraph(g1);
  return graph;
}
}
class GeApiTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(GeApiTest, ge_init_run_mode_train) {
  std::map<std::string, std::string> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}
TEST_F(GeApiTest, ge_init_run_mode_online_infer) {
  std::map<std::string, std::string> options;
  options[OPTION_EXEC_DEVICE_ID] = "0";
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(GeApiTest, ge_init_option_autoTune) {
  std::map<std::string, std::string> options;
  options["ge.autoTuneMode"] = "on";
  EXPECT_NE(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(GeApiTest, profiling_option_fail) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_MODE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  EXPECT_NE(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(GeApiTest, tiling_sink_option_invalid) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>(ge::TILING_SCHEDULE_OPTIMIZE, "invalid"));
  EXPECT_NE(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(GeApiTest, ge_init_option_invalid) {
  std::map<std::string, std::string> options;
  options["ge.optionInvalid"] = "invalid";
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  GEFinalize();
}

TEST_F(GeApiTest, ge_init_with_core_num) {
  EXPECT_EQ(GEFinalize(), SUCCESS);
  std::map<std::string, std::string> options;
  options[AICORE_NUM] = "100|100";
  EXPECT_NE(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(GeApiTest, ge_session_auto_tune_invalid) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>("ge.autoTuneMode", "invalid"));
  Session session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_NE(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiTest, ge_session_session_id_invalid) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  options.insert({"ge.session_device_id", "1"});
  Session session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiTest, parallel_api_deleted) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  options.insert({"ge.session_device_id", "1"});
  Session session(options);
  EXPECT_EQ(session.ShardGraphs(), FAILED);
  EXPECT_EQ(session.ShardGraphsToFile(nullptr), FAILED);
  EXPECT_EQ(session.SaveGraphsToPb(nullptr), FAILED);
}


TEST_F(GeApiTest, ge_session_session_id_invalid_02) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  options.insert({"ge.session_device_id", "abcdefg"});
  Session session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiTest, ge_session_op_precision_mode_invalid_02) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  options.insert({"ge.exec.op_precision_mode", "abcdefg"});
  Session session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_NE(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiTest, ge_session_test) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options;
  Session session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);

  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph, ascend_options), FAILED);

  ascend_options["ge.autoTuneMode"] = "RA";
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), FAILED);

  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(session.BuildGraph(graph_id, tensors), FAILED);

  vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), SUCCESS); // Push to queue.

  vector<string> var_inputs;
  EXPECT_EQ(session.GetVariables(var_inputs, outputs), FAILED);

  vector<AscendString> var_names;
  EXPECT_EQ(session.GetVariables(var_names, outputs), FAILED);

  std::string key;
  pCallBackFunc ge_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key, ge_callback), SUCCESS);

  session::pCallBackFunc session_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), session_callback), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, ge_session_test_fail) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  options.insert(pair<std::string, std::string>("ge.optionInvalid", "invalid"));
  Session session1(options);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};
  Session session2(ascend_options);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, AddGraph_test_fail) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  uint32_t graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> option;
  Session session(options);
  options.insert(pair<std::string, std::string>("ge.optionInvalid", "invalid"));
  (void)session.AddGraph(graph_id, graph, option);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};
  (void)session.AddGraph(graph_id, graph, ascend_options);
  (void)session.AddGraphWithCopy(graph_id, graph, ascend_options);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, run_graph_with_device_tensor) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<ge::Tensor> inputs;
  std::vector<float> input_data(1 * 224 * 224, 0);
  for (int i = 0; i < 2; ++i) {
    TensorDesc desc(Shape({1, 1, 224, 224}));
    desc.SetPlacement(Placement::kPlacementDevice);
    inputs.emplace_back(desc, (uint8_t *)input_data.data(), input_data.size() * sizeof(float));
  }
  std::vector<ge::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, run_graph_with_checkpoint) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto variable_1 = OP_CFG(VARIABLE)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("variable_1");

  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
  DEF_GRAPH(g1) {
    CHAIN(NODE(variable_1)->EDGE(0, 0)->NODE(netoutput));
    ADD_OUTPUT(variable_1, 0);
  };

  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, ge_session_info_test) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  uint64_t session_id = 0;
  {
    Session session(options);
    session_id = session.sessionId_;
    EXPECT_EQ(session_id, session.GetSessionId());
  }
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, CheckOptionsValueInvalid_test) {
  std::map<AscendString, AscendString> options = {
    {AscendString("ge.key"), AscendString("")}};
  Status ret = ge::GEInitialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

using namespace gert;
void ConstructInputOutputTensor(std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                                size_t output_num = 1U) {
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  TensorDesc desc_1(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_1{desc_1};
  input_tensor_1.SetData(reinterpret_cast<uint8_t *>(input_data_1.data()), input_data_1.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_1);

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  TensorDesc desc_2(Shape({1, 2, 3, 4}));
  ge::Tensor input_tensor_2{desc_2};
  input_tensor_2.SetData(reinterpret_cast<uint8_t *>(input_data_2.data()), input_data_2.size() * sizeof(int32_t));
  inputs.emplace_back(input_tensor_2);

  for (size_t i = 0; i < output_num; ++i) {
    std::vector<uint8_t> output_data_1(96, 0xFF);
    TensorDesc output_desc_1(Shape({1, 2, 3, 4}));
    ge::Tensor output_tensor_1{output_desc_1};
    output_tensor_1.SetData(output_data_1.data(), output_data_1.size());
    outputs.emplace_back(output_tensor_1);
  }
  return;
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *       netoutput
 */
ge::Graph BuildAddGraph() {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  std::vector<int64_t> shape{1, 2, 3, 4};
  auto data_1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("data_1");
  auto data_2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(1)
      .OutCnt(1)
      .Build("data_2");
  auto add_1 = OP_CFG(ADD)
      .TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_1_fake_id")
      .Build("add_1");

  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(add_1)->EDGE(0, 0)->NODE(netoutput));
    CHAIN(NODE(data_2)->EDGE(0, 1)->NODE(add_1));
    ADD_OUTPUT(add_1, 0);
  };

  auto graph = ToGeGraph(g1);
  return graph;
}

void MockGenerateTask() {
auto aicore_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
  if (node.GetType() == CONSTANT) {
    return SUCCESS;
  }

  auto op_desc = node.GetOpDesc();
  op_desc->SetOpKernelLibName("AiCoreLib");
  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
  const char kernel_bin[] = "kernel_bin";
  vector<char> buffer(kernel_bin, kernel_bin + strlen(kernel_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>("test", std::move(buffer));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
  size_t arg_size = 100;
  std::vector<uint8_t> args(arg_size, 0);
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto kernel_info = task_def.mutable_kernel();
  kernel_info->set_args(args.data(), args.size());
  kernel_info->set_args_size(arg_size);
  kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_info->set_kernel_name(node.GetName());
  kernel_info->set_block_dim(1);
  uint16_t args_offset[2] = {0};
  kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
  kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

  tasks.emplace_back(task_def);
  return SUCCESS;
  };

  auto rts_func = [](const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("RTSLib", rts_func);
}

void RunSession(bool compile, int threadId) {
  // 多线程用例, slog stub在主线程已配置，避免进程结束有先后造成的内存泄漏
  gert::GertRuntimeStub runtime_stub(false);
  std::unique_ptr<ArgsChecker> args_checker;

  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  {
    Session session(options);

    auto graph = BuildAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    if (compile) {
      auto ret = session.CompileGraph(graph_id);
      ASSERT_EQ(ret, SUCCESS);

      const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
      EXPECT_NE(summary, nullptr);
      size_t weight_size, feature_size;
      EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
      EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
      bool is_refreshable = false;
      EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
      EXPECT_EQ(is_refreshable, true);

      std::vector<uint8_t> feature_mem(feature_size, 0);
      EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

      std::vector<ge::Tensor> inputs;
      std::vector<ge::Tensor> outputs;
      ConstructInputOutputTensor(inputs, outputs);
      ge::diagnoseSwitch::DisableDumper();
      runtime_stub.Clear();
      EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

      std::vector<gert::Tensor> gert_inputs;
      std::vector<gert::Tensor> gert_outputs;
      gert_inputs.resize(2);
      gert_outputs.resize(1);
      std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
      gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) input_data_1.data()};

      std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
      gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                                {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                                gert::kOnDeviceHbm,                                // placement
                                ge::DT_INT32,                              // data type
                                (void *) input_data_2.data()};

      std::vector<uint8_t> output_data_1(96, 0xFF);
      gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              gert::kOnDeviceHbm,                                // placement
                              ge::DT_INT32,                              // data type
                              (void *) output_data_1.data()};
      ge::diagnoseSwitch::DisableDumper();
      runtime_stub.Clear();
      EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));
      CHECK_GRAPH(PreRunAfterBuild) {
        args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
      };

      EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
      EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
      EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
      EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
    } else {
      std::vector<ge::Tensor> inputs;
      std::vector<ge::Tensor> outputs;
      ConstructInputOutputTensor(inputs, outputs);
      EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
    }
  }
  runtime_stub.Clear();
}

void ThreadRunSession(int threadId) {
  for (int i = 0; i < 10; i++) {
    RunSession(false, threadId);
    RunSession(true, threadId);
  }
}

TEST_F(GeApiTest, session_run_graph_with_stream_async_parallel) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  MockGenerateTask();
  DUMP_GRAPH_WHEN("PreRunAfterBuild");

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
  ge::SlogStub::SetInstance(std::make_shared<SlogStubImpl>());

  std::thread t1(ThreadRunSession, 1);
  std::thread t2(ThreadRunSession, 2);
  t1.join();
  t2.join();

  ge::SlogStub::SetInstance(nullptr);
  mmSetEnv(kEnvValue, "", 1);
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  ReInitGe();
}

TEST_F(GeApiTest, ExecuteGraphWithStreamAsync_Ok_StaticGraphEnableGeProfiling) {
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  MockGenerateTask();
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
  ge::SlogStub::SetInstance(std::make_shared<SlogStubImpl>());

  setenv("GE_PROFILING_TO_STD_OUT", "1", 1); // Reset for it`s set in main.
  RunSession(true, 0);
  unsetenv("GE_PROFILING_TO_STD_OUT");
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  std::string ge_profiling_path = current_path;
  ge_profiling_path += "/ge_profiling_" + std::to_string(mmGetPid()) + ".txt";
  // todo 疑似没有落盘,受到其它用例的影响,待用例作者修复,不影响CI
  EXPECT_EQ(mmAccess(ge_profiling_path.c_str()), EN_OK);

  ge::SlogStub::SetInstance(nullptr);
  mmSetEnv(kEnvValue, "", 1);
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  ReInitGe();
}

TEST_F(GeApiTest, CheckOptionsKeyInvalid_test) {
  GEFinalize();
  std::map<AscendString, AscendString> options = {
    {AscendString(""), AscendString("Placeholder:0;Placeholder_1:1")}};
  Status ret = ge::GEInitialize(options);
  EXPECT_NE(ret, SUCCESS);
  ge::GEGetErrorMsgV2();
  ge::GEGetWarningMsgV2();
}

TEST_F(GeApiTest, RunGraphAsync) {
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  MockGenerateTask();
  std::map<std::string, std::string> str_options;
  EXPECT_EQ(GEInitialize(str_options), SUCCESS);
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLogOnFial = "Run graph async failed";
  std::map<AscendString, AscendString> options;
  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  std::vector<ge::Tensor> inputs;

  // invalid graph id
  // RunGraphAsync submit failed
  test_callback_called = false;
  auto callback = [](Status status, std::vector<ge::Tensor> &outputs) {
    EXPECT_NE(status, SUCCESS);
    test_callback_called = true;
  };

  // get graph_node fail
  EXPECT_NE(session_ptr->RunGraphAsync(10000, inputs, callback), SUCCESS);
  sleep(1);  // wait callback
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  // after RunGraphAsync run failed before, RunGraphAsync submit success
  EXPECT_EQ(session_ptr->RunGraphAsync(graph_id, inputs, callback), SUCCESS);
  sleep(1);  // wait callback
  EXPECT_EQ(test_callback_called, true);
  delete session_ptr;
  ReInitGe();
}

TEST_F(GeApiTest, RunGraphAsync_Success_Twice) {
  GeTensorDesc desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  std::vector<float32_t> value(1, 1);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(desc, (uint8_t *)value.data(), sizeof(float32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 1, 1});
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 1, 1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("const_1", const1)->EDGE(0, 1)->NODE("add_1", add1));
  };
  auto graph = ToGeGraph(g1);

  map<AscendString, AscendString> options;
  options["ge.graphMaxParallelModelNum"] = "8";
  Session session(options);
  auto mem = ModelManager::MallocWeightsMem("0_1_g1_1", 0, 1536);
  EXPECT_NE(mem, nullptr);

  session.AddGraph(1, graph, options);
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<float32_t> input_data(1, 0);
  TensorDesc desc1(Shape({1, 1, 1, 1}));
  desc1.SetPlacement(Placement::kPlacementDevice);
  inputs.emplace_back(desc1, (uint8_t *)input_data.data(), input_data.size() * sizeof(float32_t));

  test_callback_called = false;
  size_t sleep_times = 0U;
  const size_t sleep_times_max = 5U;
  auto callback = [](Status status, std::vector<ge::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    test_callback_called = true;
  };

  EXPECT_EQ(session.RunGraphAsync(1, inputs, callback), SUCCESS);
  while (!test_callback_called) {
    sleep(1);  // wait callback
    if (++sleep_times > sleep_times_max) {
      break;
    }
  }
  EXPECT_TRUE(test_callback_called);
  test_callback_called = false;
  sleep_times = 0U;

  EXPECT_EQ(session.RunGraphAsync(1, inputs, callback), SUCCESS);
  while (!test_callback_called) {
    sleep(1);  // wait callback
    if (++sleep_times > sleep_times_max) {
      break;
    }
  }

  EXPECT_TRUE(test_callback_called);
  ModelManager::FreeWeightsMem("0_1_g1_1", 0, mem);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiTest, RunGraphAsync_RunCustomPass_Success) {
  // 定义自定义 Pass 函数
  auto MyCustomPass = [](ge::GraphPtr &graph, CustomPassContext &context) -> Status {
    if (graph->GetName() == "test") {
      context.SetErrorMessage("graph name is invalid");
      return FAILED;
    }
    return SUCCESS;
  };
  REGISTER_CUSTOM_PASS("TestCustomPass").CustomPassFn(MyCustomPass);
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLog = "Run custom pass [TestCustomPass] success";
  std::map<AscendString, AscendString> options;
  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  runtime_stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(session_ptr->RunGraph(graph_id, inputs, outputs), FAILED);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog) >= 0);
  runtime_stub.GetSlogStub().Clear();
  delete session_ptr;
}

TEST_F(GeApiTest, AddGraph_for_max_load_option) {
  std::map<AscendString, AscendString> options;
  options.emplace("ge.graphMaxParallelModelNum", "10");
  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);
  delete session_ptr;
}

TEST_F(GeApiTest, AddGraph_for_max_load_option2) {
  std::map<AscendString, AscendString> options;
  options.emplace("ge.graphMaxParallelModelNum", "-1");
  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);
  delete session_ptr;
}

TEST_F(GeApiTest, test_Construct_session_fail_log) {
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLogOnFial = "Construct session failed";  // key log for session construct failed for tool analyze

  GEFinalize();
  std::map<std::string, std::string> options;
  Session sess1(options);  // ge not initialized
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  std::map<AscendString, AscendString> ascend_options;
  Session sess2(ascend_options);  // ge not initialized
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  ReInitGe();
  ascend_options[AscendString()] = "";  // option key is empty
  Session sess3(ascend_options);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  options["ge.exec.precision_mode"] = "invalid";  // invalid option value
  Session sess4(options);
  runtime_stub.GetSlogStub().Clear();

  std::map<AscendString, AscendString> ascend_options1;
  ascend_options1[AscendString("ge.exec.precision_mode")] = "invalid";  // invalid option value
  Session sess5(ascend_options1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.Clear();
}

TEST_F(GeApiTest, tes_AddGraph_fail_log) {
  const char_t *kKeyLogOnFial = "Add graph failed";  // key log for AddGraph failed for tool analyze
  GertRuntimeStub runtime_stub;
  ReInitGe();

  // add graph test
  std::map<std::string, std::string> options;
  Session sess(options);  // contruct session successfully
  std::map<AscendString, AscendString> graph_options;

  // empty graph, get graph name failed
  Graph g1;
  EXPECT_NE(sess.AddGraph(1, g1, graph_options), SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  // empty graph
  Graph g2("g2");
  EXPECT_NE(sess.AddGraph(1, g2, graph_options), SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  // graph option key is empty
  graph_options[AscendString()] = "";
  EXPECT_NE(sess.AddGraph(1, g2, graph_options), SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, "") >= 0);

  runtime_stub.Clear();
}

TEST_F(GeApiTest, test_ExecuteGraphWithStreamAsync) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<std::string, std::string> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  std::map<AscendString, AscendString> options;
  {
  Session session(options_init);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  // no compiled failed
  EXPECT_NE(session.LoadGraph(graph_id, options, nullptr), SUCCESS);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // Loadgraph with invalid option
  std::map<AscendString, AscendString> options_invalid;
  options_invalid.emplace("ge.exec.frozenInputIndexes", "2a");
  ret = session.LoadGraph(graph_id, options_invalid, nullptr);
  EXPECT_NE(ret, SUCCESS);

  options_invalid.clear();
  options_invalid.emplace("ge.exec.frozenInputIndexes", "0,b,8");
  ret = session.LoadGraph(graph_id, options_invalid, nullptr);
  EXPECT_NE(ret, SUCCESS);

  options_invalid.clear();
  options_invalid.emplace("ge.exec.frozenInputIndexes", "0,99999999999999999999999999999999999999999999999999999999999999999999999,8");
  ret = session.LoadGraph(graph_id, options_invalid, nullptr);
  EXPECT_NE(ret, SUCCESS);

  options.emplace("ge.exec.frozenInputIndexes", "0,1111,10");
  ret = session.LoadGraph(graph_id, options, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};

  std::vector<uint8_t> output_data_1(96, 0xFF);
  gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                                // placement
                          ge::DT_INT32,                              // data type
                          (void *) output_data_1.data()};
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));

  rtStream_t stream = (void *)0x123;
  EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
  runtime_stub.Clear();
}
}

TEST_F(GeApiTest, test_ExecuteGraphWithStreamAsync_with_hint_option) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  // key log for RunGraphWithStreamAsync failed for tool analyze
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<std::string, std::string> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  {
    Session session(options_init);

    auto graph = gert::ShareGraph::OnlyDataGraph({-1, -1}, {-1, -1});
    uint32_t graph_id = 1;
    std::map<AscendString, AscendString> options;
    options.emplace(std::make_pair("ge.inputHintShape", "0:[4, 2];1:[4, 2]"));
    session.AddGraph(graph_id, graph, options);

    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);

    ret = session.LoadGraph(graph_id, options, nullptr);
    EXPECT_EQ(ret, SUCCESS);

    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    ConstructInputOutputTensor(inputs, outputs);
    ge::diagnoseSwitch::DisableDumper();

    std::vector<gert::Tensor> gert_inputs;
    std::vector<gert::Tensor> gert_outputs;
    gert_inputs.resize(2);
    gert_outputs.resize(2);
    std::vector<int32_t> input_data_1(2 * 4, 666);
    gert_inputs[0] = {{{4, 2}, {4, 2}},                // shape
                      {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                      gert::kOnDeviceHbm,                                // placement
                      ge::DT_INT32,                              // data type
                      (void *) input_data_1.data()};

    std::vector<int32_t> input_data_2(2 * 4, 666);
    gert_inputs[1] = {{{4, 2}, {4, 2}},                // shape
                      {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                      gert::kOnDeviceHbm,                                // placement
                      ge::DT_INT32,                              // data type
                      (void *) input_data_2.data()};

    std::vector<uint8_t> output_data_1(96, 0xFF);
    gert_outputs[0] = {{{4, 2}, {4, 2}},                // shape
                       {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                       gert::kOnDeviceHbm,                                // placement
                       ge::DT_INT32,                              // data type
                       (void *) output_data_1.data()};
    std::vector<uint8_t> output_data_2(96, 0xFF);
    gert_outputs[1] = {{{4, 2}, {4, 2}},                // shape
                       {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                       gert::kOnDeviceHbm,                                // placement
                       ge::DT_INT32,                              // data type
                       (void *) output_data_2.data()};
    ge::diagnoseSwitch::DisableDumper();
    runtime_stub.Clear();
    EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));

    rtStream_t stream = (void *)0x123;
    EXPECT_EQ(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
    runtime_stub.Clear();
  }
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(GeApiTest, test_GeSessionExecuteGraphWithStreamAsync) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<std::string, std::string> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  std::map<AscendString, AscendString> options;
  {
  Session session(options_init);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  ret = GeSessionLoadGraph(session, graph_id, options, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<uint8_t> feature_mem(feature_size, 0);
  EXPECT_EQ(SUCCESS, session.UpdateGraphFeatureMemoryBase(graph_id, feature_mem.data(), feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};

  std::vector<uint8_t> output_data_1(96, 0xFF);
  gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                                // placement
                          ge::DT_INT32,                              // data type
                          (void *) output_data_1.data()};
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, GeSessionExecuteGraphWithStreamAsync(session, graph_id, nullptr, gert_inputs, gert_outputs));
}
}

TEST_F(GeApiTest, test_ExecuteGraphWithStreamAsync_without_compile) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<std::string, std::string> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  std::map<AscendString, AscendString> options;
  {
  Session session(options_init);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  auto ret = session.LoadGraph(graph_id, options, nullptr);
  EXPECT_NE(ret, SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs);
  ge::diagnoseSwitch::DisableDumper();

  std::vector<gert::Tensor> gert_inputs;
  std::vector<gert::Tensor> gert_outputs;
  gert_inputs.resize(2);
  gert_outputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 666);
  gert_inputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_1.data()};

  std::vector<int32_t> input_data_2(1 * 2 * 3 * 4, 666);
  gert_inputs[1] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            gert::kOnDeviceHbm,                                // placement
                            ge::DT_INT32,                              // data type
                            (void *) input_data_2.data()};

  std::vector<uint8_t> output_data_1(96, 0xFF);
  gert_outputs[0] = {{{1,2,3,4}, {1,2,3,4}},                // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                                // placement
                          ge::DT_INT32,                              // data type
                          (void *) output_data_1.data()};
  ge::diagnoseSwitch::DisableDumper();
  runtime_stub.Clear();
  EXPECT_NE(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));
  runtime_stub.Clear();
}
}

TEST_F(GeApiTest, session_execute_graph_with_graph_not_compile) {
  gert::GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(1);
  outputs.resize(1);
  EXPECT_NE(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  runtime_stub.Clear();
}

TEST_F(GeApiTest, session_execute_graph_with_graph_not_load) {
  gert::GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(1);
  outputs.resize(1);
  EXPECT_NE(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  runtime_stub.Clear();
}

TEST_F(GeApiTest, session_execute_invalid) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  Session session(options);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;

  session.AddGraph(graph_id, graph);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(7);
  outputs.resize(6);

  std::vector<ge::Tensor> ge_inputs;
  std::vector<ge::Tensor> ge_outputs;

  // incorrect input
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, ge_inputs, ge_outputs));
  EXPECT_NE(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // incorrect outputs
  Tensor te;
  std::vector<ge::Tensor> invalid_ge_outputs = {te, te, te};
  auto graph2 = BuildAddGraph();
  EXPECT_EQ(session.AddGraph(6 , graph2), SUCCESS);
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(6, nullptr, ge_inputs, invalid_ge_outputs));

  // 空tensor输入
  std::vector<gert::Tensor> empty_inputs;
  std::vector<gert::Tensor> empty_outputs;
  EXPECT_NE(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, empty_inputs, empty_outputs));

  // 构造与图model io size不同的tensor
  inputs.resize(5);
  outputs.resize(5);
  EXPECT_NE(SUCCESS, session.ExecuteGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  inputs.resize(7);
  outputs.resize(6);
  Session session1(options);
  const auto compute_graph1 = MakeShared<ComputeGraph>("test_graph");
  // empty graph
  session1.AddGraph(3, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph1));
  EXPECT_NE(SUCCESS, session1.ExecuteGraphWithStreamAsync(4, nullptr, inputs, outputs));
}

TEST_F(GeApiTest, test_RunGraphWithStreamAsync_fail_log) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  const char_t *kKeyLogOnFial = "Run graph with stream async failed";
  GertRuntimeStub runtime_stub;
  ReInitGe();

  std::map<std::string, std::string> options;
  Session sess(options);  // contruct session successfully

  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  Status ret = sess.RunGraphWithStreamAsync(10000, nullptr, inputs, outputs);  // invalid graph id
  EXPECT_NE(ret, SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.Clear();
}

TEST_F(GeApiTest, test_RemoveGraph_fail_log) {
  const char_t *kKeyLogOnFial = "Remove graph failed";  // key log for RemoveGraph failed for tool analyze
  GertRuntimeStub runtime_stub;
  ReInitGe();

  std::map<std::string, std::string> options;
  Session sess(options);  // contruct session successfully

  Status ret = sess.RemoveGraph(10000);  // invalid graph id
  EXPECT_NE(ret, SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.Clear();
}

TEST_F(GeApiTest, test_RunGraph_fail_log) {
  const char_t *kKeyLogOnFial = "Run graph failed";  // key log for RemoveGraph failed for tool analyze
  GertRuntimeStub runtime_stub;
  ReInitGe();

  std::map<std::string, std::string> options;
  Session sess(options);  // contruct session successfully

  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  Status ret = sess.RunGraph(10000, inputs, outputs);  // invalid graph id
  EXPECT_NE(ret, SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.Clear();
}

extern Graph BuildHcomGraph1();
Status GenerateTaskForHcomAllReduce(const Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) {
  std::cout << "======node.GetType():" << node.GetType()  << std::endl;
  if (node.GetType() != "HcomAllReduce") {
      std::cout << "*****return***"<< std::endl;
      return SUCCESS;
  }

  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
  task_def.set_stream_id(0);
  const auto op_desc = node.GetOpDesc();
  OpsKernelInfoStore *ptr = &g_fake_hccl_ops_kernel_info_store;
  op_desc->SetExtAttr("OpsKernelInfoStorePtr", ptr);
  (void)ge::AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "sum");
  int32_t root_id = 0;
  (void)ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
  auto &kernel_hccl_def = *task_def.mutable_kernel_hccl();
  kernel_hccl_def.set_op_index(op_desc->GetId());
  kernel_hccl_def.set_hccl_type("HcomAllReduce");
  tasks.emplace_back(task_def);
  return SUCCESS;
}

uint32_t GetModelIdByGraphId(uint32_t graph_id, Session &session) {
  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  ge::SessionPtr inner_session = session_manager->GetSession(session.sessionId_);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);

  return ge_root_model->GetModelId();
}

TEST_F(GeApiTest, pa_remapped_test_001) {
  GertRuntimeStub runtime_stub;
  //MockIoMemReuse  mock hccl task;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph1();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size, fix_feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  EXPECT_EQ(SUCCESS, summary->GetFixedFeatureMemorySize(fix_feature_size));
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);
  std::vector<uint8_t> fixed_feature_mem(fix_feature_size, 0);
  EXPECT_EQ(SUCCESS, session.SetGraphFixedFeatureMemoryBase(graph_id, fixed_feature_mem.data(), fix_feature_size));

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  const auto model_id = GetModelIdByGraphId(graph_id, session);
  const auto davinci_model = ModelManager::GetInstance().GetModel(model_id);
  const auto allocator = davinci_model->GetLogicalMemAllocation();

  for (size_t i = 0; i < allocator.size() - 1UL; i++) {
    if (allocator[i].type == ge::MemAllocation::Type::FEATURE_MAP ||
        allocator[i].type == ge::MemAllocation::Type::FIXED_FEATURE_MAP) {
      ret = session.PaRemapped(allocator[i].logical_addr, 0UL, allocator[i].data_size);
      EXPECT_EQ(ret, SUCCESS);
    } else {
      ret = session.PaRemapped(allocator[i].logical_addr, 0UL, allocator[i].data_size - 8UL);
      EXPECT_EQ(ret, PARAM_INVALID);
    }
    std::cout << "allocator " << i << " : " << allocator[i].ToString() << std::endl;
  }

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(GeApiTest, pa_remapped_test_002) {
  GertRuntimeStub runtime_stub;
  //MockIoMemReuse  mock hccl task;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "0";
  Session session(options);

  auto graph = BuildHcomGraph1();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(graph_id);
  EXPECT_NE(summary, nullptr);
  size_t weight_size, feature_size;
  EXPECT_EQ(SUCCESS, summary->GetConstMemorySize(weight_size));
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemorySize(feature_size));
  bool is_refreshable = true;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, false);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  const auto model_id = GetModelIdByGraphId(graph_id, session);
  const auto davinci_model = ModelManager::GetInstance().GetModel(model_id);
  const auto allocator = davinci_model->GetLogicalMemAllocation();

  for (size_t i = 0; i < allocator.size() - 1UL; i++) {
    if (allocator[i].type == ge::MemAllocation::Type::FEATURE_MAP ||
        allocator[i].type == ge::MemAllocation::Type::FIXED_FEATURE_MAP) {
      ret = session.PaRemapped(allocator[i].logical_addr, 0UL, allocator[i].data_size);
      EXPECT_EQ(ret, FAILED);
    }
    std::cout << "allocator " << i << " : " << allocator[i].ToString() << std::endl;
  }

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(GeApiTest, pa_remapped_test_003) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph1 = BuildHcomGraph1();
  auto graph2 = BuildHcomGraph1();
  uint32_t first_graph_id = 1U;
  uint32_t second_graph_id = 2U;
  std::map<AscendString, AscendString> options2;
  options2.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  session.AddGraph(first_graph_id, graph1, options2);
  session.AddGraph(second_graph_id, graph2, options2);
  auto ret = session.CompileGraph(first_graph_id);
  ret |= session.CompileGraph(second_graph_id);
  EXPECT_EQ(ret, SUCCESS);

  // get graph summary
  const CompiledGraphSummaryPtr summary = session.GetCompiledGraphSummary(first_graph_id);
  EXPECT_NE(summary, nullptr);
  bool is_refreshable = false;
  EXPECT_EQ(SUCCESS, summary->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);
  const CompiledGraphSummaryPtr summary2 = session.GetCompiledGraphSummary(second_graph_id);
  EXPECT_NE(summary2, nullptr);
  EXPECT_EQ(SUCCESS, summary2->GetFeatureMemoryBaseRefreshable(is_refreshable));
  EXPECT_EQ(is_refreshable, true);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ConstructInputOutputTensor(inputs, outputs, 2);
  TensorDesc output_desc(Shape({1, 2, 3, 4}));
  runtime_stub.Clear();
  EXPECT_EQ(session.PaRemapped(0x1234UL, 0UL, 100UL), PARAM_INVALID);
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(first_graph_id, nullptr, inputs, outputs));
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(second_graph_id, nullptr, inputs, outputs));

  const auto first_model_id = GetModelIdByGraphId(first_graph_id, session);
  const auto davinci_model1 = ModelManager::GetInstance().GetModel(first_model_id);
  const auto allocator1 = davinci_model1->GetLogicalMemAllocation();

  for (size_t i = 0; i < allocator1.size() - 1UL; i++) {
    if (allocator1[i].type == ge::MemAllocation::Type::FEATURE_MAP ||
        allocator1[i].type == ge::MemAllocation::Type::FIXED_FEATURE_MAP) {
      ret = session.PaRemapped(allocator1[i].logical_addr, 0UL, allocator1[i].data_size);
      EXPECT_EQ(ret, SUCCESS);
    } else {
      ret = session.PaRemapped(allocator1[i].logical_addr, 0UL, allocator1[i].data_size - 8UL);
      EXPECT_EQ(ret, PARAM_INVALID);
    }
    std::cout << "allocator1 " << i << " : " << allocator1[i].ToString() << std::endl;
  }

  const auto second_model_id = GetModelIdByGraphId(second_graph_id, session);
  const auto davinci_model2 = ModelManager::GetInstance().GetModel(second_model_id);
  const auto allocator2 = davinci_model2->GetLogicalMemAllocation();

  for (size_t i = 0; i < allocator2.size() - 1UL; i++) {
    if (allocator2[i].type == ge::MemAllocation::Type::FEATURE_MAP ||
        allocator2[i].type == ge::MemAllocation::Type::FIXED_FEATURE_MAP) {
      ret = session.PaRemapped(allocator2[i].logical_addr, 0UL, allocator2[i].data_size);
      EXPECT_EQ(ret, SUCCESS);
    } else {
      ret = session.PaRemapped(allocator2[i].logical_addr, 0UL, allocator2[i].data_size - 8UL);
      EXPECT_EQ(ret, PARAM_INVALID);
    }
    std::cout << "allocator2 " << i << " : " << allocator2[i].ToString() << std::endl;
  }

  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(GeApiTest, pa_remapped_test_004) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph1();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.PaRemapped(0UL, 0UL, 128UL);
  EXPECT_NE(ret, SUCCESS);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(GeApiTest, pa_remapped_test_005) {
  GertRuntimeStub runtime_stub;
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHcomAllReduce);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  Session session(options);

  auto graph = BuildHcomGraph1();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->SetGraphUnknownFlag(true);
  ret = session.PaRemapped(0UL, 0UL, 128UL);
  EXPECT_NE(ret, SUCCESS);
  OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  runtime_stub.Clear();
}

TEST_F(GeApiTest, RunGraphAsync_RunCustomPass_AfterInferShape_Success) {
  auto MyCustomPass = [](ge::GraphPtr &graph, CustomPassContext &context) -> Status {
    for (auto n : graph->GetDirectNode()) {
      AscendString type;
      n.GetType(type);
      if (type != ADD) {
        continue;
      }
      auto id_op = op::Identity("identity");
      auto id_node = graph->AddNodeByOp(id_op);
      graph->AddDataEdge(n, 0, id_node, 0);

      TensorDesc td;
      n.GetOutputDesc(0U, td);
      id_op.UpdateInputDesc(0U, td);
      id_op.InferShapeAndType();
    }
    return SUCCESS;
  };
  REGISTER_CUSTOM_PASS("TestCustomPassAfterInferShape")
      .CustomPassFn(MyCustomPass)
      .Stage(CustomPassStage::kAfterInferShape);
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLog = "Run custom pass [TestCustomPassAfterInferShape] success";
  const char_t *kKeyStageLog = "Starting custom pass [TestCustomPassAfterInferShape] in stage [AfterInferShape]";

  REGISTER_CUSTOM_PASS("TestCustomPassBeforeInferShape")
      .CustomPassFn([](ge::GraphPtr &graph, CustomPassContext &context) -> Status { return SUCCESS; });
  const char_t *kKeyLog1 = "Run custom pass [TestCustomPassBeforeInferShape] success";
  const char_t *kKeyStageLog1 = "Starting custom pass [TestCustomPassBeforeInferShape] in stage [BeforeInferShape]";

  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
  };
  auto graph = ToGeGraph(g1);

  std::map<AscendString, AscendString> options;
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  runtime_stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), FAILED);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog1) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog1) >= 0);
  EXPECT_EQ(graph.GetAllNodes().size(), 4UL);
  runtime_stub.GetSlogStub().Clear();
}

/* 用例描述: 动态图带变量，在线执行，不提前申请输出内存
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. Session编译，加载，执行，卸载，析构
*
* 预期结果：
* 1. 不申请输出内存，执行成功
*/
TEST_F(GeApiTest, DynamicMode_RunGraphWithStreamAsync_NotAllocOutputs) {
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    Session session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);

    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);

    ret = session.LoadGraph(graph_id, {}, stream);
    EXPECT_EQ(ret, SUCCESS);
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    ConstructInputOutputTensor(inputs, outputs, 0);
    inputs[0].SetPlacement(ge::Placement::kPlacementDevice);
    inputs[1].SetPlacement(ge::Placement::kPlacementDevice);
    outputs.clear();
    ret = session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
    rtStreamDestroy(stream);
    ret = session.RemoveGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ReInitGe();
}

TEST_F(GeApiTest, RunGraphAsync_RunCustomPass_AfterBuiltInFusion_Success) {
  auto MyCustomPass = [](ge::GraphPtr &graph, CustomPassContext &context) -> Status {
    for (auto n : graph->GetDirectNode()) {
      AscendString type;
      n.GetType(type);
      if (type != ADD) {
        continue;
      }
      auto id_op = op::Identity("identity");
      auto id_node = graph->AddNodeByOp(id_op);
      graph->AddDataEdge(n, 0, id_node, 0);

      TensorDesc td;
      n.GetOutputDesc(0U, td);
      id_op.UpdateInputDesc(0U, td);
      id_op.InferShapeAndType();
    }
    return SUCCESS;
  };
  REGISTER_CUSTOM_PASS("TestCustomPassAfterBuiltinFusionPass")
      .CustomPassFn(MyCustomPass)
      .Stage(CustomPassStage::kAfterBuiltinFusionPass);
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLog = "Run custom pass [TestCustomPassAfterBuiltinFusionPass] success";
  const char_t *kKeyStageLog = "Starting custom pass [TestCustomPassAfterBuiltinFusionPass] in stage [AfterBuiltinFusionPass]";

  REGISTER_CUSTOM_PASS("TestCustomPassBeforeInferShape")
      .CustomPassFn([](ge::GraphPtr &graph, CustomPassContext &context) -> Status { return SUCCESS; });
  const char_t *kKeyLog1 = "Run custom pass [TestCustomPassBeforeInferShape] success";
  const char_t *kKeyStageLog1 = "Starting custom pass [TestCustomPassBeforeInferShape] in stage [BeforeInferShape]";

  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
  };
  auto graph = ToGeGraph(g1);

  std::map<AscendString, AscendString> options;
  Session session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);

  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  runtime_stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), FAILED);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog1) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog1) >= 0);
  EXPECT_EQ(graph.GetAllNodes().size(), 4UL);
  runtime_stub.GetSlogStub().Clear();
}

TEST_F(GeApiTest, RunGraph_GraphMaxParallelModelNum_Success) {
  GeTensorDesc desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  std::vector<float32_t> value(1, 1);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(desc, (uint8_t *)value.data(), sizeof(float32_t));
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1);
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 1, 1});
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 1, 1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("const_1", const1)->EDGE(0, 1)->NODE("add_1", add1));
  };
  auto graph = ToGeGraph(g1);

  map<AscendString, AscendString> options;
  options["ge.graphMaxParallelModelNum"] = "8";
  Session session(options);
  auto mem = ModelManager::MallocWeightsMem("0_1_g1_1", 0, 1536);
  EXPECT_NE(mem, nullptr);

  session.AddGraph(1, graph, options);
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  std::vector<float32_t> input_data(1, 0);
  TensorDesc desc1(Shape({1, 1, 1, 1}));
  desc1.SetPlacement(Placement::kPlacementDevice);
  inputs.emplace_back(desc1, (uint8_t *)input_data.data(), input_data.size() * sizeof(float32_t));

  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  ModelManager::FreeWeightsMem("0_1_g1_1", 0, mem);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ReInitGe();
}


#define EXPECT_STR_EQ(x, y) EXPECT_EQ(std::string(x.GetString()), std::string(y))

REG_OP(QueryIrTestOp1)
  .INPUT(required_x1, TensorType::ALL())
  .OPTIONAL_INPUT(optional_x2, TensorType::ALL())
  .DYNAMIC_INPUT(dynamic_x3, TensorType::ALL())
  .OUTPUT(required_y1, TensorType::ALL())
  .DYNAMIC_OUTPUT(dynamic_y1, TensorType::ALL())
  .OP_END_FACTORY_REG(QueryIrTestOp1)

TEST_F(GeApiTest, QueryIrInputOutput) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp1", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 3U);
  EXPECT_EQ(outputs.size(), 2U);
  EXPECT_EQ(inputs[0].first, "required_x1");
  EXPECT_EQ(inputs[0].second, "required");
  EXPECT_EQ(inputs[1].first, "optional_x2");
  EXPECT_EQ(inputs[1].second, "optional");
  EXPECT_EQ(inputs[2].first, "dynamic_x3");
  EXPECT_EQ(inputs[2].second, "dynamic");
  EXPECT_EQ(outputs[0].first, "required_y1");
  EXPECT_EQ(outputs[0].second, "required");
  EXPECT_EQ(outputs[1].first, "dynamic_y1");
  EXPECT_EQ(outputs[1].second, "dynamic");
  EXPECT_EQ(attrs.size(), 0U);
}

REG_OP(QueryIrTestOp2)
  .DYNAMIC_INPUT(dynamic_x3, TensorType::ALL())
  .INPUT(required_x1, TensorType::ALL())
  .OPTIONAL_INPUT(optional_x2, TensorType::ALL())
  .DYNAMIC_OUTPUT(dynamic_y1, TensorType::ALL())
  .OUTPUT(required_y1, TensorType::ALL())
  .OP_END_FACTORY_REG(QueryIrTestOp2)

TEST_F(GeApiTest, QueryIrInputOutputKeepOrder) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp2", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 3U);
  EXPECT_EQ(outputs.size(), 2U);
  EXPECT_EQ(inputs[0].first, "dynamic_x3");
  EXPECT_EQ(inputs[0].second, "dynamic");
  EXPECT_EQ(inputs[1].first, "required_x1");
  EXPECT_EQ(inputs[1].second, "required");
  EXPECT_EQ(inputs[2].first, "optional_x2");
  EXPECT_EQ(inputs[2].second, "optional");
  EXPECT_EQ(outputs[0].first, "dynamic_y1");
  EXPECT_EQ(outputs[0].second, "dynamic");
  EXPECT_EQ(outputs[1].first, "required_y1");
  EXPECT_EQ(outputs[1].second, "required");
  EXPECT_EQ(attrs.size(), 0U);
}

REG_OP(QueryIrTestOp3)
  .REQUIRED_ATTR(attr1, Int)
  .REQUIRED_ATTR(attr2, Float)
  .REQUIRED_ATTR(attr3, String)
  .REQUIRED_ATTR(attr4, Bool)
  .REQUIRED_ATTR(attr5, Tensor)
  .REQUIRED_ATTR(attr6, Type)
  .REQUIRED_ATTR(attr7, NamedAttrs)
  .REQUIRED_ATTR(attr8, ListInt)
  .REQUIRED_ATTR(attr9, ListFloat)
  .REQUIRED_ATTR(attr10, ListString)
  .REQUIRED_ATTR(attr11, ListBool)
  .REQUIRED_ATTR(attr12, ListTensor)
  .REQUIRED_ATTR(attr13, Bytes)
  .REQUIRED_ATTR(attr14, ListListInt)
  .REQUIRED_ATTR(attr15, ListNamedAttrs)
  .OP_END_FACTORY_REG(QueryIrTestOp3)

TEST_F(GeApiTest, QueryIrAttr) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp3", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 0U);
  EXPECT_EQ(outputs.size(), 0U);
  ASSERT_EQ(attrs.size(), 15U);
  EXPECT_STR_EQ(attrs[0].first, "attr1");
  EXPECT_STR_EQ(attrs[0].second, "VT_INT");
  EXPECT_STR_EQ(attrs[1].first, "attr2");
  EXPECT_STR_EQ(attrs[1].second, "VT_FLOAT");
  EXPECT_STR_EQ(attrs[2].first, "attr3");
  EXPECT_STR_EQ(attrs[2].second, "VT_STRING");
  EXPECT_STR_EQ(attrs[3].first, "attr4");
  EXPECT_STR_EQ(attrs[3].second, "VT_BOOL");
  EXPECT_STR_EQ(attrs[4].first, "attr5");
  EXPECT_STR_EQ(attrs[4].second, "VT_TENSOR");
  EXPECT_STR_EQ(attrs[5].first, "attr6");
  EXPECT_STR_EQ(attrs[5].second, "VT_DATA_TYPE");
  EXPECT_STR_EQ(attrs[6].first, "attr7");
  EXPECT_STR_EQ(attrs[6].second, "VT_NAMED_ATTRS");
  EXPECT_STR_EQ(attrs[7].first, "attr8");
  EXPECT_STR_EQ(attrs[7].second, "VT_LIST_INT");
  EXPECT_STR_EQ(attrs[8].first, "attr9");
  EXPECT_STR_EQ(attrs[8].second, "VT_LIST_FLOAT");
  EXPECT_STR_EQ(attrs[9].first, "attr10");
  EXPECT_STR_EQ(attrs[9].second, "VT_LIST_STRING");
  EXPECT_STR_EQ(attrs[10].first, "attr11");
  EXPECT_STR_EQ(attrs[10].second, "VT_LIST_BOOL");
  EXPECT_STR_EQ(attrs[11].first, "attr12");
  EXPECT_STR_EQ(attrs[11].second, "VT_LIST_TENSOR");
  EXPECT_STR_EQ(attrs[12].first, "attr13");
  EXPECT_STR_EQ(attrs[12].second, "VT_BYTES");
  EXPECT_STR_EQ(attrs[13].first, "attr14");
  EXPECT_STR_EQ(attrs[13].second, "VT_LIST_LIST_INT");
  EXPECT_STR_EQ(attrs[14].first, "attr15");
  EXPECT_STR_EQ(attrs[14].second, "VT_LIST_NAMED_ATTRS");
}

REG_OP(QueryIrTestOp4)
  .ATTR(attr1, Int, 3)
  .ATTR(attr2, Float, 2.0)
  .REQUIRED_ATTR(attr3, String)
  .ATTR(attr4, Bool, false)
  .REQUIRED_ATTR(attr5, Tensor)
  .ATTR(attr6, Type, DT_BF16)
  .REQUIRED_ATTR(attr7, NamedAttrs)
  .REQUIRED_ATTR(attr8, ListInt)
  .REQUIRED_ATTR(attr9, ListFloat)
  .REQUIRED_ATTR(attr10, ListString)
  .REQUIRED_ATTR(attr11, ListBool)
  .REQUIRED_ATTR(attr12, ListTensor)
  .REQUIRED_ATTR(attr13, Bytes)
  .REQUIRED_ATTR(attr14, ListListInt)
  .REQUIRED_ATTR(attr15, ListNamedAttrs)
  .OP_END_FACTORY_REG(QueryIrTestOp4)

TEST_F(GeApiTest, QueryIrAttrKeepOrder) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp4", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 0U);
  EXPECT_EQ(outputs.size(), 0U);
  ASSERT_EQ(attrs.size(), 15U);
  EXPECT_STR_EQ(attrs[0].first, "attr1");
  EXPECT_STR_EQ(attrs[0].second, "VT_INT");
  EXPECT_STR_EQ(attrs[1].first, "attr2");
  EXPECT_STR_EQ(attrs[1].second, "VT_FLOAT");
  EXPECT_STR_EQ(attrs[2].first, "attr3");
  EXPECT_STR_EQ(attrs[2].second, "VT_STRING");
  EXPECT_STR_EQ(attrs[3].first, "attr4");
  EXPECT_STR_EQ(attrs[3].second, "VT_BOOL");
  EXPECT_STR_EQ(attrs[4].first, "attr5");
  EXPECT_STR_EQ(attrs[4].second, "VT_TENSOR");
  EXPECT_STR_EQ(attrs[5].first, "attr6");
  EXPECT_STR_EQ(attrs[5].second, "VT_DATA_TYPE");
  EXPECT_STR_EQ(attrs[6].first, "attr7");
  EXPECT_STR_EQ(attrs[6].second, "VT_NAMED_ATTRS");
  EXPECT_STR_EQ(attrs[7].first, "attr8");
  EXPECT_STR_EQ(attrs[7].second, "VT_LIST_INT");
  EXPECT_STR_EQ(attrs[8].first, "attr9");
  EXPECT_STR_EQ(attrs[8].second, "VT_LIST_FLOAT");
  EXPECT_STR_EQ(attrs[9].first, "attr10");
  EXPECT_STR_EQ(attrs[9].second, "VT_LIST_STRING");
  EXPECT_STR_EQ(attrs[10].first, "attr11");
  EXPECT_STR_EQ(attrs[10].second, "VT_LIST_BOOL");
  EXPECT_STR_EQ(attrs[11].first, "attr12");
  EXPECT_STR_EQ(attrs[11].second, "VT_LIST_TENSOR");
  EXPECT_STR_EQ(attrs[12].first, "attr13");
  EXPECT_STR_EQ(attrs[12].second, "VT_BYTES");
  EXPECT_STR_EQ(attrs[13].first, "attr14");
  EXPECT_STR_EQ(attrs[13].second, "VT_LIST_LIST_INT");
  EXPECT_STR_EQ(attrs[14].first, "attr15");
  EXPECT_STR_EQ(attrs[14].second, "VT_LIST_NAMED_ATTRS");
}

TEST_F(GeApiTest, QueryUnregisteredIr) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_NE(GetRegisteredIrDef("QueryIrTestOpNotRegistered", inputs, outputs, attrs), SUCCESS);
}
}  // namespace ge
