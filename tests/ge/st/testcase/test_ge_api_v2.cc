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
#include <iostream>
#include <regex>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mmpa/mmpa_api.h"
#include "graph/utils/file_utils.h"
#include "ge_running_env/path_utils.h"
#include "common/path_utils.h"
#include "macro_utils/dt_public_scope.h"
#include "common/plugin/ge_make_unique_util.h"
#include "proto/ge_ir.pb.h"
#include "ge/ge_api_v2.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "session/session_manager.h"
#include "session/ge_session_impl.h"
#include "init_ge.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"
#include "dflow/base/model/flow_model_om_saver.h"
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
#include "common/mem_conflict_share_graph.h"
#include "common/share_graph.h"
#include "framework/executor/ge_executor.h"
#include "utils/model_data_builder.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "runtime/gert_api.h"

namespace ge {
namespace {
static FakeOpsKernelInfoStore g_fake_hccl_ops_kernel_info_store;
bool test_callback_called = false;

bool CheckWeightFile(const std::string &external_weight_dir) {
  if (mmAccess(external_weight_dir.c_str()) != EN_OK || !ge::IsDir(external_weight_dir.c_str())) {
    return false;
  }

  // 使用正则表达式匹配文件模式
  std::regex pattern("weight_.*");
  DIR *dir = opendir(external_weight_dir.c_str());
  if (dir == nullptr) {
    return false;
  }

  struct dirent *entry;
  bool found = false;
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    std::string entry_path = external_weight_dir + "/" + entry->d_name;
    if (ge::IsFile(entry_path.c_str())) {
      std::string filename = entry->d_name;
      if (std::regex_match(filename, pattern)) {
        found = true;
        break;
      }
    }
  }
  closedir(dir);
  return found;
}

int RemoveAll(const std::string &path) {
  return ge::PathUtils::RemoveDirectories(path);
}
}
class GeApiV2Test : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(GeApiV2Test, ge_init_run_mode_train) {
  std::string long_value(900, 'a');  // 创建 900 字符的长字符串
  std::map<AscendString, AscendString> options = {
    {AscendString("ge.test.long_option"), AscendString(long_value.c_str())}
  };
  options[OPTION_GRAPH_RUN_MODE] = "1";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}
TEST_F(GeApiV2Test, ge_init_run_mode_online_infer) {
  std::map<AscendString, AscendString> options;
  options[OPTION_EXEC_DEVICE_ID] = "0";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(GeApiV2Test, ge_init_option_autoTune) {
  std::map<AscendString, AscendString> options;
  options["ge.autoTuneMode"] = "on";
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(GeApiV2Test, profiling_option_fail) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_MODE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(GeApiV2Test, tiling_sink_option_invalid) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>(ge::TILING_SCHEDULE_OPTIMIZE.c_str(), "invalid"));
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(GeApiV2Test, ge_init_option_invalid) {
  std::map<AscendString, AscendString> options;
  options["ge.optionInvalid"] = "invalid";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GEFinalizeV2();
}

TEST_F(GeApiV2Test, api_not_init_failed) {
  std::map<AscendString, AscendString> options;
  GeSession session(options);
  Graph graph;
  EXPECT_EQ(session.AddGraphClone(1, graph, options), FAILED);
  EXPECT_EQ(session.RemoveGraph(1), FAILED);
  std::vector<gert::Tensor> inputs, outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphWithStreamAsync(1, nullptr, inputs, outputs), FAILED);
  RunCallback callback;
  EXPECT_EQ(session.RegisterCallBackFunc(nullptr, callback), FAILED);
  EXPECT_EQ(session.LoadGraph(1, options, nullptr), FAILED);
  RunAsyncCallbackV2 callback2;
  EXPECT_EQ(session.RunGraphAsync(1, inputs, callback2), FAILED);
}

TEST_F(GeApiV2Test, ge_init_with_core_num) {
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  std::map<AscendString, AscendString> options;
  options[AICORE_NUM.c_str()] = "100|100";
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(GeApiV2Test, ge_session_auto_tune_invalid) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>("ge.autoTuneMode", "invalid"));
  GeSession session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_NE(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiV2Test, ge_session_session_id_invalid) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  options.insert({"ge.session_device_id", "1"});
  GeSession session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiV2Test, ge_session_session_id_invalid_02) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  options.insert({"ge.session_device_id", "abcdefg"});
  GeSession session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiV2Test, ge_session_op_precision_mode_invalid_02) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  options.insert({"ge.exec.op_precision_mode", "abcdefg"});
  GeSession session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_NE(session.AddGraph(graph_id, graph), SUCCESS);
}

TEST_F(GeApiV2Test, ge_session_test) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options;
  GeSession session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);

  EXPECT_EQ(session.AddGraphClone(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraphClone(graph_id, graph, ascend_options), FAILED);

  ascend_options["ge.autoTuneMode"] = "RA";
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), FAILED);

  vector<gert::Tensor> inputs;
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);

  vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), SUCCESS); // Push to queue.
  // 不能混用Run接口
  EXPECT_NE(session.RunGraph(1, inputs, outputs), SUCCESS);
  EXPECT_NE(session.RunGraphWithStreamAsync(1, nullptr, inputs, outputs), SUCCESS);
  std::string key;
  RunCallback ge_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), ge_callback), SUCCESS);

  RunCallback session_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), session_callback), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, ge_session_test_fail) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  options.insert(pair<AscendString, AscendString>("ge.optionInvalid", "invalid"));
  GeSession session1(options);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};
  GeSession session2(ascend_options);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, AddGraph_test_fail) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  uint32_t graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> option;
  GeSession session(options);
  options.insert(pair<AscendString, AscendString>("ge.optionInvalid", "invalid"));
  (void)session.AddGraph(graph_id, graph, option);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};
  (void)session.AddGraph(graph_id, graph, ascend_options);
  (void)session.AddGraphClone(graph_id, graph, ascend_options);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, run_graph_with_device_tensor) {
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
  GeSession session(options);
  session.AddGraph(1, graph, options);

  std::vector<gert::Tensor> inputs;
  inputs.resize(2U);
  ge::TensorCheckUtils::ConstructGertTensor(inputs[0], {1, 1, 224, 224}, DT_FLOAT);
  ge::TensorCheckUtils::ConstructGertTensor(inputs[1], {1, 1, 224, 224}, DT_FLOAT);
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, run_graph_with_checkpoint) {
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
  GeSession session(options);
  session.AddGraph(1, graph, options);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, ge_session_info_test) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, CheckOptionsValueInvalid_test) {
  std::map<AscendString, AscendString> options = {
    {AscendString("ge.key"), AscendString("")}};
  Status ret = ge::GEInitializeV2(options);
  EXPECT_EQ(ret, SUCCESS);
}

using namespace gert;
void ConstructInputOutputTensor(std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
                                size_t output_num = 1U) {
  inputs.resize(2U);
  ge::TensorCheckUtils::ConstructGertTensor(inputs[0], {1, 2, 3, 4});
  ge::TensorCheckUtils::ConstructGertTensor(inputs[1], {1, 2, 3, 4});

  outputs.resize(output_num);
  for (size_t i = 0; i < output_num; ++i) {
    ge::TensorCheckUtils::ConstructGertTensor(outputs[0], {1, 2, 3, 4});
  }
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

/**
 *      data  const
 *        \   /
 *         add
 *          |
 *       netoutput
 */
ge::Graph BuildConstGraph() {
  std::vector<int64_t> shape{1, 2, 3, 4};
  GeTensorDesc desc(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_INT32);
  std::vector<int32_t> value(24, 1);
  GeTensorPtr data_tensor1 = make_shared<GeTensor>(desc, (uint8_t *)value.data(), sizeof(int32_t) * 24);
  auto const1 = OP_CFG(CONSTANT).Weight(data_tensor1).TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("const1");
  std::vector<int32_t> value2(24, 1);
  GeTensorPtr data_tensor2 = make_shared<GeTensor>(desc, (uint8_t *)value2.data(), sizeof(int32_t));
  auto const2 = OP_CFG(CONSTANT).Weight(data_tensor2).TensorDesc(FORMAT_NCHW, DT_INT32, shape)
      .InCnt(2)
      .OutCnt(1)
      .Build("const2");

  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};

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
    CHAIN(NODE(const1)->EDGE(0, 0)->NODE(add_1)->EDGE(0, 0)->NODE(netoutput));
    CHAIN(NODE(const2)->EDGE(0, 1)->NODE(add_1));
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
    GeSession session(options);

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

      std::vector<gert::Tensor> inputs;
      std::vector<gert::Tensor> outputs;
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
      EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));
      CHECK_GRAPH(PreRunAfterBuild) {
        args_checker = std::make_unique<ArgsChecker>(graph, graph_id, session.GetSessionId(), runtime_stub);
      };

      EXPECT_EQ(SUCCESS, args_checker->SetFmAddr((uint64_t)feature_mem.data(), feature_size));
      EXPECT_EQ(SUCCESS, args_checker->SetModelInputAddr({0, 1}, inputs));
      EXPECT_EQ(SUCCESS, args_checker->SetModelOutputAddr({0}, outputs));
      EXPECT_EQ(SUCCESS, args_checker->TaskIoAddressesAreCorrect());
    } else {
      std::vector<gert::Tensor> inputs;
      std::vector<gert::Tensor> outputs;
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

TEST_F(GeApiV2Test, session_run_graph_with_stream_async_parallel) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
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

TEST_F(GeApiV2Test, CheckOptionsKeyInvalid_test) {
  GEFinalizeV2();
  std::map<AscendString, AscendString> options = {
    {AscendString(""), AscendString("Placeholder:0;Placeholder_1:1")}};
  Status ret = ge::GEInitializeV2(options);
  EXPECT_NE(ret, SUCCESS);
  ge::GEGetErrorMsgV2();
  ge::GEGetWarningMsgV2();
}

TEST_F(GeApiV2Test, RunGraphAsync) {
  ModelManager::GetInstance().cust_aicpu_so_.clear();
  MockGenerateTask();
  std::map<AscendString, AscendString> str_options;
  EXPECT_EQ(GEInitializeV2(str_options), SUCCESS);
  ReInitGe();
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLogOnFial = "Run graph async failed";
  std::map<AscendString, AscendString> options;
  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  std::vector<gert::Tensor> inputs;

  // invalid graph id
  // RunGraphAsync submit failed
  test_callback_called = false;
  auto callback = [](Status status, std::vector<gert::Tensor> &outputs) {
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
}

TEST_F(GeApiV2Test, RunGraphAsync_RunCustomPass_Success) {
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
  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  runtime_stub.GetSlogStub().SetLevelDebug();
  EXPECT_NE(session_ptr->RunGraph(graph_id, inputs, outputs), SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog) >= 0);
  runtime_stub.GetSlogStub().Clear();
  delete session_ptr;
}

TEST_F(GeApiV2Test, AddGraph_for_max_load_option) {
  std::map<AscendString, AscendString> options;
  options.emplace("ge.graphMaxParallelModelNum", "10");
  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);
  delete session_ptr;
}

TEST_F(GeApiV2Test, AddGraph_for_max_load_option2) {
  std::map<AscendString, AscendString> options;
  options.emplace("ge.graphMaxParallelModelNum", "-1");
  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);
  delete session_ptr;
}

TEST_F(GeApiV2Test, test_Construct_session_fail_log) {
  GertRuntimeStub runtime_stub;
  const char_t *kKeyLogOnFial = "Construct session failed";  // key log for session construct failed for tool analyze

  GEFinalizeV2();
  std::map<AscendString, AscendString> options;
  GeSession sess1(options);  // ge not initialized
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  std::map<AscendString, AscendString> ascend_options;
  GeSession sess2(ascend_options);  // ge not initialized
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  ReInitGe();
  ascend_options[AscendString()] = "";  // option key is empty
  GeSession sess3(ascend_options);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.GetSlogStub().Clear();

  options["ge.exec.precision_mode"] = "invalid";  // invalid option value
  GeSession sess4(options);
  runtime_stub.GetSlogStub().Clear();

  std::map<AscendString, AscendString> ascend_options1;
  ascend_options1[AscendString("ge.exec.precision_mode")] = "invalid";  // invalid option value
  GeSession sess5(ascend_options1);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.Clear();
}

TEST_F(GeApiV2Test, tes_AddGraph_fail_log) {
  const char_t *kKeyLogOnFial = "Add graph failed";  // key log for AddGraph failed for tool analyze
  GertRuntimeStub runtime_stub;
  ReInitGe();

  // add graph test
  std::map<AscendString, AscendString> options;
  GeSession sess(options);  // contruct session successfully
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

TEST_F(GeApiV2Test, test_ExecuteGraphWithStreamAsync) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<AscendString, AscendString> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  std::map<AscendString, AscendString> options;
  {
    GeSession session(options_init);

    auto graph = BuildAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);

    auto ret = session.CompileGraph(graph_id);
    ASSERT_EQ(ret, SUCCESS);

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

    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
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
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));

    rtStream_t stream = (void *)0x123;
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
    // 不能混用Run接口
    EXPECT_NE(session.RunGraph(1, inputs, outputs), SUCCESS);
    EXPECT_NE(session.RunGraphAsync(1, inputs, nullptr), SUCCESS);
    runtime_stub.Clear();
  }
}

TEST_F(GeApiV2Test, test_ExecuteGraphWithStreamAsync_with_hint_option) {
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
  std::map<AscendString, AscendString> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  {
    GeSession session(options_init);

    auto graph = gert::ShareGraph::OnlyDataGraph({-1, -1}, {-1, -1});
    uint32_t graph_id = 1;
    std::map<AscendString, AscendString> options;
    options.emplace(std::make_pair("ge.inputHintShape", "0:[4, 2];1:[4, 2]"));
    session.AddGraph(graph_id, graph, options);

    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);

    ret = session.LoadGraph(graph_id, options, nullptr);
    EXPECT_EQ(ret, SUCCESS);

    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
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
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));

    rtStream_t stream = (void *)0x123;
    EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, stream, gert_inputs, gert_outputs));
    runtime_stub.Clear();
  }
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(GeApiV2Test, RunGraphWithStreamAsync_Success_WithoutCompileGraph) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  gert::GertRuntimeStub runtime_stub;
  std::unique_ptr<ArgsChecker> args_checker;
  std::map<AscendString, AscendString> options_init;
  options_init.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options_init.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options_init.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  std::map<AscendString, AscendString> options;
  {
  GeSession session(options_init);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  auto ret = session.LoadGraph(graph_id, options, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
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
  EXPECT_EQ(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, gert_inputs, gert_outputs));
  runtime_stub.Clear();
}
}

TEST_F(GeApiV2Test, RunGraphWithStreamAsync_Success_WithoutCompileAndLoadGraph) {
  gert::GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  GeSession session(options);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(1);
  outputs.resize(1);
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  runtime_stub.Clear();
}

TEST_F(GeApiV2Test, RunGraphWithStreamAsync_Success_WithoutLoadGraph) {
  gert::GertRuntimeStub runtime_stub;
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  GeSession session(options);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(1);
  outputs.resize(1);
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  runtime_stub.Clear();
}

TEST_F(GeApiV2Test, session_execute_invalid) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");

  GeSession session(options);

  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;

  session.AddGraph(graph_id, graph);

  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(7);
  outputs.resize(6);

  std::vector<gert::Tensor> ge_inputs;
  std::vector<gert::Tensor> ge_outputs;

  // incorrect input
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, ge_inputs, ge_outputs));
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));

  // incorrect outputs
  std::vector<gert::Tensor> invalid_ge_outputs;
  invalid_ge_outputs.resize(3);
  auto graph2 = BuildAddGraph();
  EXPECT_EQ(session.AddGraph(6 , graph2), SUCCESS);
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(6, nullptr, ge_inputs, invalid_ge_outputs));

  // 空tensor输入
  std::vector<gert::Tensor> empty_inputs;
  std::vector<gert::Tensor> empty_outputs;
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, empty_inputs, empty_outputs));

  // 构造与图model io size不同的tensor
  inputs.resize(5);
  outputs.resize(5);
  EXPECT_NE(SUCCESS, session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs));
  inputs.resize(7);
  outputs.resize(6);
  GeSession session1(options);
  const auto compute_graph1 = MakeShared<ComputeGraph>("test_graph");
  // empty graph
  session1.AddGraph(3, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph1));
  EXPECT_NE(SUCCESS, session1.RunGraphWithStreamAsync(4, nullptr, inputs, outputs));
}

TEST_F(GeApiV2Test, GetCompileGraphModel_Success) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  GeSession session(options);
  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  ModelBufferData model_buff;
  EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
}

TEST_F(GeApiV2Test, GetCompileGraphModel_Success_LoadDirectly) {
  ModelBufferData model_buff;
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    GeSession session(options);
    auto graph = BuildAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
  }

  ge::GeExecutor ge_executor;
  uint32_t id = 0U;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  modelData.weight_path = "";
  ge::ModelLoadArg loadArgs{};
  auto ret = ge_executor.LoadModelFromDataWithArgs(id, modelData, loadArgs);
  EXPECT_EQ(ret, SUCCESS);
  ge_executor.UnloadModel(id);
  ge_executor.Finalize();
  ReInitGe();
}

TEST_F(GeApiV2Test, GetCompileGraphModel_Success_SaveToOmThenLoadFromFile) {
  ModelBufferData model_buff;
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    GeSession session(options);
    auto graph = BuildAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
  }

  std::string model_file = "./test_model";
  auto ret = aclgrphSaveModel(model_file, model_buff);
  ASSERT_EQ(ret, SUCCESS);

  ge::GeExecutor ge_executor;
  uint32_t id = 0U;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  modelData.weight_path = "";
  ge::ModelLoadArg loadArgs{};
  ModelData model_data;
  ret = ge_executor.LoadDataFromFile(model_file + ".om", model_data);
  ASSERT_EQ(ret, SUCCESS);

  ret = ge_executor.LoadModelFromDataWithArgs(id, model_data, loadArgs);
  ASSERT_EQ(ret, SUCCESS);
  ge_executor.UnloadModel(id);
  delete[] static_cast<char_t *>(model_data.model_data);
  ge_executor.Finalize();
  ReInitGe();
}

TEST_F(GeApiV2Test, GetCompileGraphModel_Failed_OptionInvalid) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  options.emplace("ge.exec.variable_acc", "True");
  GeSession session(options);
  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  ModelBufferData model_buff;
  EXPECT_NE(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
}

TEST_F(GeApiV2Test, GetCompileGraphModel_Failed_NotCompiled) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  GeSession session(options);
  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  ModelBufferData model_buff;
  EXPECT_NE(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
}

TEST_F(GeApiV2Test, GetCompileGraphModel_Failed_GraphIdInvalid) {
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  GeSession session(options);
  auto graph = BuildAddGraph();
  uint32_t graph_id = 1;
  session.AddGraph(graph_id, graph);
  auto ret = session.CompileGraph(graph_id);
  EXPECT_EQ(ret, SUCCESS);
  ModelBufferData model_buff;
  EXPECT_NE(session.GetCompiledModel(10010, model_buff), SUCCESS);
}

TEST_F(GeApiV2Test, test_RunGraphWithStreamAsync_fail_log) {
  // key log for RunGraphWithStreamAsync failed for tool analyze
  const char_t *kKeyLogOnFial = "Run graph with stream async failed";
  ReInitGe();

  std::map<AscendString, AscendString> options;
  GeSession sess(options);  // contruct session successfully

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  GertRuntimeStub runtime_stub;
  Status ret = sess.RunGraphWithStreamAsync(10000, nullptr, inputs, outputs);  // invalid graph id
  EXPECT_NE(ret, SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_ERROR, kKeyLogOnFial) >= 0);
  runtime_stub.Clear();
}

TEST_F(GeApiV2Test, test_RemoveGraph_fail_log) {
  const char_t *kKeyLogOnFial = "Remove graph failed";  // key log for RemoveGraph failed for tool analyze
  GertRuntimeStub runtime_stub;
  ReInitGe();

  std::map<AscendString, AscendString> options;
  GeSession sess(options);  // contruct session successfully

  Status ret = sess.RemoveGraph(10000);  // invalid graph id
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

uint32_t GetModelIdByGraphId(uint32_t graph_id, GeSession &session) {
  ge::SessionManager *session_manager = GetSessionManager();
  EXPECT_NE(session_manager, nullptr);
  auto session_id = session.GetSessionId();
  ge::SessionPtr inner_session = session_manager->GetSession(session_id);
  EXPECT_NE(inner_session, nullptr);
  const ge::GraphManager &graph_manager = inner_session->getGraphManagerObj(); // 当前无函数可以获取graph manager
  GraphNodePtr graph_node = nullptr;
  (void)graph_manager.GetGraphNode(graph_id, graph_node);
  EXPECT_NE(graph_node, nullptr);
  const auto &ge_root_model = graph_node->GetGeRootModel();
  EXPECT_NE(ge_root_model, nullptr);
  return ge_root_model->GetModelId();
}

TEST_F(GeApiV2Test, RunGraphAsync_RunCustomPass_AfterInferShape_Success) {
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
  GeSession session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  runtime_stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), FAILED);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog1) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog1) >= 0);
  EXPECT_EQ(graph.GetAllNodes().size(), 4UL);
  runtime_stub.GetSlogStub().Clear();
}

TEST_F(GeApiV2Test, RunGraphAsync_RunCustomPass_AfterBuiltInFusion_Success) {
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
  GeSession session(options);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  runtime_stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), FAILED);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyLog1) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, kKeyStageLog1) >= 0);
  EXPECT_EQ(graph.GetAllNodes().size(), 4UL);
  runtime_stub.GetSlogStub().Clear();
}

TEST_F(GeApiV2Test, RunGraph_GraphMaxParallelModelNum_Success) {
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
  GeSession session(options);
  auto mem = ModelManager::MallocWeightsMem("0_1_g1_1", 0, 1536);
  EXPECT_NE(mem, nullptr);

  session.AddGraph(1, graph, options);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  inputs.resize(1);
  ge::TensorCheckUtils::ConstructGertTensor(inputs[0], {1, 1, 1, 1}, DT_FLOAT);

  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  EXPECT_NE(session.RunGraphAsync(1, inputs, nullptr), SUCCESS);
  EXPECT_NE(session.RunGraphWithStreamAsync(1, nullptr, inputs, outputs), SUCCESS);
  ModelManager::FreeWeightsMem("0_1_g1_1", 0, mem);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ReInitGe();
}

TEST_F(GeApiV2Test, RunGraph_Failed_NotAdded) {
  map<AscendString, AscendString> options;
  GeSession session(options);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_NE(session.RunGraph(10086, inputs, outputs), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ReInitGe();
}

/* 用例描述: 动态图带变量，在线执行，不提前申请输出内存
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. GeSession编译，加载，执行，卸载，析构
*
* 预期结果：
* 1. 不申请输出内存，执行成功
*/
TEST_F(GeApiV2Test, DynamicMode_RunGraphWithStreamAsync_NotAllocOutputs) {
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);

    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);

    ret = session.LoadGraph(graph_id, {}, stream);
    EXPECT_EQ(ret, SUCCESS);
    std::vector<gert::Tensor> inputs(2);
    std::vector<gert::Tensor> outputs;
    TensorCheckUtils::ConstructGertTensor(inputs[0]);
    TensorCheckUtils::ConstructGertTensor(inputs[1]);
    ret = session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
    rtStreamDestroy(stream);

    ret = session.RemoveGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ReInitGe();
}

/* 用例描述: 带流异步执行，忽略编译和加载步骤，执行成功
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. GeSession AddGraph, RunGraphWithStreamAsync
*
* 预期结果：
* 1. 执行成功
*/
TEST_F(GeApiV2Test, RunGraphWithStreamAsync_WithoutCompileAndLoad) {
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);

    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);
    std::vector<gert::Tensor> inputs(2);
    std::vector<gert::Tensor> outputs;
    TensorCheckUtils::ConstructGertTensor(inputs[0]);
    TensorCheckUtils::ConstructGertTensor(inputs[1]);
    auto ret = session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
    rtStreamDestroy(stream);

    ret = session.RemoveGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ReInitGe();
}

/* 用例描述: 带流异步执行，省略加载步骤，执行成功
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. GeSession AddGraph, CompileGraph, RunGraphWithStreamAsync
*
* 预期结果：
* 1. 执行成功
*/
TEST_F(GeApiV2Test, RunGraphWithStreamAsync_WithoutCompile) {
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);

    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);

    auto ret = session.LoadGraph(graph_id, {}, stream);
    EXPECT_EQ(ret, SUCCESS);

    std::vector<gert::Tensor> inputs(2);
    std::vector<gert::Tensor> outputs;
    TensorCheckUtils::ConstructGertTensor(inputs[0]);
    TensorCheckUtils::ConstructGertTensor(inputs[1]);
    ret = session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
    rtStreamDestroy(stream);

    ret = session.RemoveGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ReInitGe();
}

/* 用例描述: 带流异步执行，省略编译步骤，执行成功
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. GeSession AddGraph, CompileGraph, RunGraphWithStreamAsync
*
* 预期结果：
* 1. 执行成功
*/
TEST_F(GeApiV2Test, RunGraphWithStreamAsync_WithoutLoad) {
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);
    std::vector<gert::Tensor> inputs(2);
    std::vector<gert::Tensor> outputs;
    TensorCheckUtils::ConstructGertTensor(inputs[0]);
    TensorCheckUtils::ConstructGertTensor(inputs[1]);
    ret = session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
    rtStreamDestroy(stream);

    ret = session.RemoveGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ReInitGe();
}

/* 用例描述: 开启外置权重，测试在线图编译获得离线模型，然后做离线模型加载执行。
* 预置条件：
* 1. 构造带有权重图
* 2. 设置保存外置权重的目录，用例退出时要删除该目录
*
* 测试步骤：
* 1. 开启外置权重，并设置OPTION_EXTERNAL_WEIGHT_DIR option指定外置权重路径，避免session析构后临时权重文件被删除
* 2. CompileGraph进行图编译，GetCompiledModel获取序列化的离线模型
* 3. 创建GeExecutor对象，调用LoadModelFromDataWithArgs和ExecModel
*
* 预期结果：
* 1. 图编译成功，获取离线模型成功。
* 2. 指定外置权重路径下存在weight_开头的文件
* 3. 离线模型加载成功，执行成功
*/
TEST_F(GeApiV2Test, GetCompileGraphModel_Success_LoadAndExec) {
  ModelBufferData model_buff;
  std::string external_weight_dir = "./user_weight_dir/";
  GE_MAKE_GUARD(remove_dir, [&external_weight_dir] () {
    RemoveAll(external_weight_dir);
  });
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::EXTERNAL_WEIGHT.c_str(), "1");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(ge::OPTION_EXTERNAL_WEIGHT_DIR, external_weight_dir.c_str());
    GeSession session(options);
    auto graph = BuildConstGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
  }
  ASSERT_TRUE(CheckWeightFile(external_weight_dir));
  ge::GeExecutor ge_executor;
  uint32_t id = 0U;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  modelData.weight_path = external_weight_dir;
  ge::ModelLoadArg loadArgs{};
  auto ret = ge_executor.LoadModelFromDataWithArgs(id, modelData, loadArgs);
  EXPECT_EQ(ret, SUCCESS);

  RunModelData run_input_data;
  std::vector<uint8_t> out_0(512, 0);
  RunModelData run_output_data;
  run_output_data.blobs.emplace_back(DataBuffer{out_0.data(), out_0.size(), false, 0});

  EXPECT_EQ(ge_executor.ExecModel(id, nullptr, run_input_data, run_output_data, true), SUCCESS);
  ge_executor.UnloadModel(id);
  ge_executor.Finalize();
  ReInitGe();
}

/* 用例描述: 动态图带变量，测试在线图编译获得离线模型，然后做离线模型加载执行。
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. CompileGraph进行图编译，GetCompiledModel获取序列化的离线模型
* 2. 创建ModelV2Executor对象，调用LoadExecutorFromModelData和Execute, 传入RtSession
*
* 预期结果：
* 1. 图编译成功，获取离线模型成功。
* 2. 离线模型加载成功，执行成功
* 3. 模型卸载，检查变量内存释放
*/
TEST_F(GeApiV2Test, GetCompileGraphModel_Success_Dynamic) {
  ModelBufferData model_buff;
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
  }
  std::unique_ptr<gert::ModelV2Executor> executor;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  ge::graphStatus ret;
  uint64_t session_id = 20251201;
  gert::RtSession session(session_id);
  LoadExecutorArgs load_executor_args{&session};
  // 如果不设置RtSession，ModelV2ExecutorBuilder::RestoreDeviceVarMem就不会工作，执行的时候图中有variable就会获取var manager失败
  executor = gert::LoadExecutorFromModelData(modelData, load_executor_args, ret);
  ASSERT_NE(executor, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  gert::ModelExecuteArg exe_arg;
  gert::ModelLoadArg load_arg(&session, {});
  // 图中有global_step变量，在variable.cc中获取rt_session->GetVarManager()，因此一定要传入RtSession，
  // ModelV2Executor::InitRtVarManager会设置var manager
  ret = executor->Load(exe_arg, load_arg);
  ASSERT_EQ(ret, SUCCESS);

  ModelExecuteArg execute_arg;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_NE(stream, nullptr);
  execute_arg.stream = stream;

  std::vector<gert::Tensor *> inputs(2);
  std::vector<gert::Tensor> inputs_holders(2);
  TensorCheckUtils::ConstructGertTensor(inputs_holders[0]);
  TensorCheckUtils::ConstructGertTensor(inputs_holders[1]);
  inputs[0] = &inputs_holders[0];
  inputs[1] = &inputs_holders[1];

  std::vector<gert::Tensor> outputs_holders(1);
  std::vector<gert::Tensor *> outputs(1);
  TensorCheckUtils::ConstructGertTensor(outputs_holders[0]);
  outputs[0] = &outputs_holders[0];

  ret = executor->Execute(execute_arg, &inputs[0], inputs.size(), &outputs[0], 1);
  EXPECT_EQ(ret, SUCCESS);

  // 离线场景，模型卸载不会触发变量内存的释放,需要调用session的DestroyResources函数释放
  EXPECT_TRUE(VarManager::Instance(session_id)->IsVarExist("ge_global_step"));
  EXPECT_EQ(executor->UnLoad(), SUCCESS);
  EXPECT_TRUE(VarManager::Instance(session_id)->IsVarExist("ge_global_step"));
  session.DestroyResources();
  EXPECT_FALSE(VarManager::Instance(session_id)->IsVarExist("ge_global_step"));
  rtStreamDestroy(stream);
  ReInitGe();
}

/* 用例描述: 动态图带变量，在线执行，测试模型卸载不会释放变量内存，session析构会释放变量内存。
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. GeSession编译，加载，执行，卸载，析构
*
* 预期结果：
* 1. 编译/加载/执行成功
* 2. 模型卸载，校验变量内存未释放
* 3. GeSession析构，校验变量内存释放
*/
TEST_F(GeApiV2Test, GeSessionUnloadDynamicModel_NotReleaseVarMemory) {
  uint64_t session_id;
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个ge_global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);

    std::vector<gert::Tensor> inputs(2);
    TensorCheckUtils::ConstructGertTensor(inputs[0]);
    TensorCheckUtils::ConstructGertTensor(inputs[1]);

    std::vector<gert::Tensor> outputs(1);
    TensorCheckUtils::ConstructGertTensor(outputs[0]);

    rtStream_t stream = nullptr;
    rtStreamCreate(&stream, 0);

    ret = session.LoadGraph(graph_id, {}, stream);
    EXPECT_EQ(ret, SUCCESS);

    ret = session.RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
    rtStreamDestroy(stream);

    session_id = session.GetSessionId();
    EXPECT_TRUE(VarManager::Instance(session_id)->IsVarExist("ge_global_step"));
    ret = session.RemoveGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);

    // 在线场景，模型卸载不释放变量
    EXPECT_TRUE(VarManager::Instance(session_id)->IsVarExist("ge_global_step"));
  }
  // 在线场景，GeSession析构，触发变量内存释放
  EXPECT_FALSE(VarManager::Instance(session_id)->IsVarExist("ge_global_step"));
  ReInitGe();
}

/* 用例描述: 动态图带变量，测试在线图编译获得离线模型，然后做离线模型加载执行。加载不传入RTSesssion
* 预置条件：
* 1. 构造动态shape图
*
* 测试步骤：
* 1. CompileGraph进行图编译，GetCompiledModel获取序列化的离线模型
* 2. 创建ModelV2Executor对象，调用LoadExecutorFromModelData和Execute, 不传入RtSession
*
* 预期结果：
* 1. 图编译成功，获取离线模型成功。
* 2. 离线模型加载成功，执行成功
*/
TEST_F(GeApiV2Test, GetCompileGraphModel_Success_Dynamic_LoadWithoutRtSession) {
  ModelBufferData model_buff;
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个global_step的变量
    GeSession session(options);
    auto graph = BuildDynamicAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
  }
  std::unique_ptr<gert::ModelV2Executor> executor;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  ge::graphStatus ret;
  RtSession session;
  LoadExecutorArgs load_executor_args{&session, {}}; // 这里是与上面用例的区别
  // ModelV2ExecutorBuilder::RestoreDeviceVarMem发现有变量，且没有设置rt_session，就创建一个有效的session_id
  executor = gert::LoadExecutorFromModelData(modelData, load_executor_args, ret);
  ASSERT_NE(executor, nullptr);
  EXPECT_EQ(ret, SUCCESS);

  gert::ModelExecuteArg exe_arg;
  gert::ModelLoadArg load_arg(&session, {}); // 这里是与上面用例的区别
  // ModelV2Executor::InitRtVarManager使用内部有效的session id
  ret = executor->Load(exe_arg, load_arg);
  ASSERT_EQ(ret, SUCCESS);

  ModelExecuteArg execute_arg;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_NE(stream, nullptr);
  execute_arg.stream = stream;

  std::vector<gert::Tensor *> inputs(2);
  std::vector<gert::Tensor> inputs_holders(2);
  TensorCheckUtils::ConstructGertTensor(inputs_holders[0]);
  TensorCheckUtils::ConstructGertTensor(inputs_holders[1]);
  inputs[0] = &inputs_holders[0];
  inputs[1] = &inputs_holders[1];

  std::vector<gert::Tensor> outputs_holders(1);
  std::vector<gert::Tensor *> outputs(1);
  TensorCheckUtils::ConstructGertTensor(outputs_holders[0]);
  outputs[0] = &outputs_holders[0];

  ret = executor->Execute(execute_arg, &inputs[0], inputs.size(), &outputs[0], 1);
  EXPECT_EQ(ret, SUCCESS);
  rtStreamDestroy(stream);
  ReInitGe();
}

/* 用例描述: 静态图带变量，测试在线图编译获得离线模型，然后做离线模型加载执行。
* 预置条件：
* 1. 构造静态shape图
*
* 测试步骤：
* 1. CompileGraph进行图编译，GetCompiledModel获取序列化的离线模型
* 2. 创建GeExecutor对象，调用LoadModelFromDataWithArgs和ExecModel
*
* 预期结果：
* 1. 图编译成功，获取离线模型成功。
* 2. 离线模型加载成功，执行成功
*/
TEST_F(GeApiV2Test, GetCompileGraphModel_Success_StaticHasVariable) {
  ModelBufferData model_buff;
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1"); //会添加一个global_step的变量
    GeSession session(options);
    auto graph = BuildAddGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
  }
  ge::GeExecutor ge_executor;
  uint32_t id = 0U;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  ge::ModelLoadArg loadArgs{};
  auto ret = ge_executor.LoadModelFromDataWithArgs(id, modelData, loadArgs);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<uint8_t> in_0(512, 0);
  RunModelData run_input_data;
  run_input_data.blobs.emplace_back(DataBuffer{in_0.data(), in_0.size(), false, 0});
  run_input_data.blobs.emplace_back(DataBuffer{in_0.data(), in_0.size(), false, 0});

  std::vector<uint8_t> out_0(512, 0);
  RunModelData run_output_data;
  run_output_data.blobs.emplace_back(DataBuffer{out_0.data(), out_0.size(), false, 0});

  EXPECT_EQ(ge_executor.ExecModel(id, nullptr, run_input_data, run_output_data, true), SUCCESS);
  ge_executor.UnloadModel(id);
  ge_executor.Finalize();
  ReInitGe();
}

Status CopyWeights(const std::vector<ExternalWeightDescPtr> &external_weight_paths,
  std::vector<FileConstantMem> &file_constant_mems) {
  for (const auto &external_weight_path : external_weight_paths) {
    if (external_weight_path == nullptr) {
      std::cerr << "external_weight_path is nullptr" << std::endl;
      return FAILED;
    }
    const auto file_path = external_weight_path->GetLocation();

    // 打开文件
    std::ifstream file(file_path.GetString(), std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "无法打开文件进行读取: " << file_path.GetString() << std::endl;
      return FAILED;
    }
    // 移动到文件开始
    file.seekg(0, std::ios::beg);
    // 移动到文件末尾
    file.seekg(0, std::ios::end);
    const auto file_size = static_cast<size_t>(file.tellg());
    // 移动到文件开始
    file.seekg(0, std::ios::beg);
    // 读取文件内容到主机内存
    std::vector<char> file_data(file_size);
    file.read(file_data.data(), file_size);

    if (!file) {
      std::cerr << "文件读取失败: " << file_path.GetString() << std::endl;
      return FAILED;
    }

    file.close();

    std::cout << "文件读取成功: " << file_path.GetString() << ", size: " << file_size << " 字节" << std::endl;

    void *dev_ptr = nullptr;
    const auto alloc_size = (file_size + 32 - 1) / 32 * 32;
    if (rtMalloc(&dev_ptr, alloc_size, RT_MEMORY_HBM, GE_MODULE_NAME) != 0) {
      std::cerr << "无法分配设备内存" << std::endl;
      return FAILED;
    }
    std::string file_constant_file_name;
    std::string file_dir;
    SplitFilePath(file_path.GetString(), file_dir, file_constant_file_name);
    (void)file_dir;
    FileConstantMem file_constant_mem{ file_constant_file_name, dev_ptr, file_size};
    file_constant_mems.emplace_back(file_constant_mem);
    // 拷贝到设备内存
    if (rtMemcpy(dev_ptr, alloc_size, file_data.data(), file_size, RT_MEMCPY_HOST_TO_DEVICE)) {
      std::cerr << "数据拷贝到设备内存失败" << std::endl;
      return FAILED;
    }
  }
  return SUCCESS;
}
/* 用例描述: 开启外置权重，测试在线图编译获得离线模型，指定外置权重device地址，然后做离线模型加载执行。
* 预置条件：
* 1. 构造带有权重图
* 2. 设置保存外置权重的目录，用例退出时要删除该目录
*
* 测试步骤：
* 1. 开启外置权重，并设置OPTION_EXTERNAL_WEIGHT_DIR option指定外置权重路径，避免session析构后临时权重文件被删除
* 2. CompileGraph进行图编译，GetCompiledModel获取序列化的离线模型
* 3. GetExternalWeightPaths 获取外置权重文件路径，申请device内存
* 3. 创建GeExecutor对象，设置外置权重device地址，调用LoadModelFromDataWithArgs和ExecModel
*
* 预期结果：
* 1. 图编译成功，获取离线模型成功。
* 2. 指定外置权重路径下存在weight_开头的文件
* 3. 离线模型加载成功，执行成功
*/
TEST_F(GeApiV2Test, GetCompileGraphModel_UserSetExternalWeightAddress_Success_LoadAndExec) {
  ModelBufferData model_buff;
  std::vector<ExternalWeightDescPtr> external_weight_paths;
  std::string external_weight_dir = "./user_weight_dir/";
  GE_MAKE_GUARD(remove_dir, [&external_weight_dir] () {
    RemoveAll(external_weight_dir);
  });
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::EXTERNAL_WEIGHT.c_str(), "1");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(ge::OPTION_EXTERNAL_WEIGHT_DIR, external_weight_dir.c_str());
    GeSession session(options);
    auto graph = BuildConstGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(session.GetCompiledModel(graph_id, model_buff), SUCCESS);
    CompiledGraphSummaryPtr summary_ptr = session.GetCompiledGraphSummary(graph_id);
    ASSERT_NE(summary_ptr, nullptr);
    EXPECT_EQ(summary_ptr->GetExternalWeightPaths(external_weight_paths), SUCCESS);
    EXPECT_FALSE(external_weight_paths.empty());
  }
  ASSERT_TRUE(CheckWeightFile(external_weight_dir));
  ge::GeExecutor ge_executor;
  uint32_t id = 0U;
  ge::ModelData modelData;
  modelData.model_data = reinterpret_cast<void *>(model_buff.data.get());
  modelData.model_len = static_cast<uint64_t>(model_buff.length);
  modelData.priority = 1;
  modelData.om_path = "";
  modelData.weight_path = external_weight_dir;
  ge::ModelLoadArg load_args{};
  GE_MAKE_GUARD(free_device_mem, [&load_args] () {
    for (const auto &external_weight_path : load_args.file_constant_mems) {
      rtFree(const_cast<void *>(external_weight_path.device_mem));
    }
  });
  ASSERT_EQ(CopyWeights(external_weight_paths, load_args.file_constant_mems), 0);
  GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  auto ret = ge_executor.LoadModelFromDataWithArgs(id, modelData, load_args);
  EXPECT_EQ(ret, SUCCESS);
  auto check_ret = stub.GetSlogStub().FindLog(DLOG_INFO, " found user device memory ");
  EXPECT_TRUE(check_ret != -1);

  RunModelData run_input_data;
  std::vector<uint8_t> out_0(512, 0);
  RunModelData run_output_data;
  run_output_data.blobs.emplace_back(DataBuffer{out_0.data(), out_0.size(), false, 0});
  EXPECT_EQ(ge_executor.ExecModel(id, nullptr, run_input_data, run_output_data, true), SUCCESS);
  ge_executor.UnloadModel(id);
  ge_executor.Finalize();
  ReInitGe();
}

/* 用例描述: 开启外置权重和编译缓存，并指定外置权重路径，测试编译缓存与指定路径叠加功能
* 预置条件：
* 1. 构造带有权重图
* 2. 设置保存外置权重的目录，用例退出时要删除该目录
*
* 测试步骤：
* 1. 开启外置权重，并设置OPTION_EXTERNAL_WEIGHT_DIR option指定外置权重路径，避免session析构后临时权重文件被删除
* 2. 开启编译缓存功能，CompileGraph进行图编译
* 3. 新启动一个session，也开启编译缓存功能，CompileGraph进行图编译
*
* 预期结果：
* 1. 图编译成功
* 2. 新session图编译成功，能正常找到外置权重文件
*/
TEST_F(GeApiV2Test, ExternalWeightDirAndModelCache_Success) {
  std::string model_cache_dir = "./build_cache_dir";
  std::string graph_key = "test_graph_001";
  std::string external_weight_dir = "./user_weight_dir/";
  ASSERT_TRUE(ge::CreateDir(model_cache_dir) == 0);
  GE_MAKE_GUARD(remove_model_cache_dir, [&model_cache_dir] () {
    RemoveAll(model_cache_dir);
  });
  GE_MAKE_GUARD(remove_dir, [&external_weight_dir] () {
    RemoveAll(external_weight_dir);
  });
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::EXTERNAL_WEIGHT.c_str(), "1");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(ge::OPTION_EXTERNAL_WEIGHT_DIR, external_weight_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_COMPILER_CACHE_DIR, model_cache_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_KEY, graph_key.c_str());
    GeSession session(options);
    auto graph = BuildConstGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ASSERT_TRUE(CheckWeightFile(external_weight_dir));
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::EXTERNAL_WEIGHT.c_str(), "1");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(ge::OPTION_EXTERNAL_WEIGHT_DIR, external_weight_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_COMPILER_CACHE_DIR, model_cache_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_KEY, graph_key.c_str());
    GeSession session(options);
    auto graph = BuildConstGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
    ret = session.RunGraph(graph_id, inputs, outputs);
    EXPECT_EQ(ret, SUCCESS);
  }
  ReInitGe();
}

/* 用例描述: 异常用例。开启外置权重和编译缓存，第一个session指定外置权重路径编译成功，第二个session未指定外置权重路径，编译失败
* 预置条件：
* 1. 构造带有权重图
* 2. 设置保存外置权重的目录，用例退出时要删除该目录
*
* 测试步骤：
* 1. 开启外置权重，并设置OPTION_EXTERNAL_WEIGHT_DIR option指定外置权重路径，避免session析构后临时权重文件被删除
* 2. 开启编译缓存功能，CompileGraph进行图编译
* 3. 新启动一个session，也开启编译缓存功能，CompileGraph进行图编译
*
* 预期结果：
* 1. 第一个session图编译成功
* 2. 第二个session图编译成功，但是由于在model_cache_dir/weight找不到外置权重，执行报错
*/
TEST_F(GeApiV2Test, ExternalWeightDirAndModelCache_Failed_SecondSessionNotSetExternalWeightDir) {
  std::string model_cache_dir = "./build_cache_dir";
  std::string graph_key = "test_graph_001";
  std::string external_weight_dir = "./user_weight_dir/";
  ASSERT_TRUE(ge::CreateDir(model_cache_dir) == 0);
  GE_MAKE_GUARD(remove_model_cache_dir, [&model_cache_dir] () {
    RemoveAll(model_cache_dir);
  });
  GE_MAKE_GUARD(remove_dir, [&external_weight_dir] () {
    RemoveAll(external_weight_dir);
  });
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::EXTERNAL_WEIGHT.c_str(), "1");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(ge::OPTION_EXTERNAL_WEIGHT_DIR, external_weight_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_COMPILER_CACHE_DIR, model_cache_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_KEY, graph_key.c_str());
    GeSession session(options);
    auto graph = BuildConstGraph();
    uint32_t graph_id = 1;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
  }
  ASSERT_TRUE(CheckWeightFile(external_weight_dir));
  {
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::EXTERNAL_WEIGHT.c_str(), "1");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(ge::OPTION_GRAPH_COMPILER_CACHE_DIR, model_cache_dir.c_str());
    options.emplace(ge::OPTION_GRAPH_KEY, graph_key.c_str());
    GeSession session(options);
    auto graph = BuildConstGraph();
    uint32_t graph_id = 20;
    session.AddGraph(graph_id, graph);
    auto ret = session.CompileGraph(graph_id);
    EXPECT_EQ(ret, SUCCESS);
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
    GertRuntimeStub stub;
    ret = session.RunGraph(graph_id, inputs, outputs);
    EXPECT_NE(ret, SUCCESS);
    std::string exp_err_log = "Failed to copy data to file constant";
    auto log_check = stub.GetSlogStub().FindLog(DLOG_ERROR, exp_err_log.c_str());
    EXPECT_NE(log_check, -1);
  }
  ReInitGe();
}

TEST(FeatureQueryTest, QuerySupportedIr) {
  bool existingFeature = IsIrRepSupport("_inference_rule");
  EXPECT_TRUE(existingFeature);
}

TEST(FeatureQueryTest, QueryUnsupportedIr) {
  bool futureNewFeature = IsIrRepSupport("future_new_feature_rule");
  EXPECT_FALSE(futureNewFeature);

  bool noFeature = IsIrRepSupport("");
  EXPECT_FALSE(noFeature);

  bool similarErrorFeature = IsIrRepSupport("Inference_Rule");
  EXPECT_FALSE(similarErrorFeature);
}
}  // namespace ge
