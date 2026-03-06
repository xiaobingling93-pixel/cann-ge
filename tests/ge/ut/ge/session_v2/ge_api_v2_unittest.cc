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

#include <vector>
#include <string>
#include <map>
#include "common/share_graph.h"
#include "ge/ge_api_error_codes.h"
#include "stub/gert_runtime_stub.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/plugin/ge_make_unique_util.h"
#include "proto/ge_ir.pb.h"
#include "ge/ge_api_v2.h"
#include "session/session_manager.h"
#include "session/ge_session_impl.h"
#include "session/session_utils.h"
#include "framework/memory/memory_api.h"
#include "graph/utils/graph_utils_ex.h"
#include "register/ops_kernel_builder_registry.h"
#include "register/node_converter_registry.h"
#include "register/optimization_option_registry.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/load/graph_loader.h"
#include "ge/ge_api_types.h"
#include "graph/operator_factory_impl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/label/label_maker.h"
#include "api/gelib/gelib.h"
#include "graph/build/stream/stream_utils.h"
#include "register/register_custom_pass.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "common/env_path.h"
#include "ge_running_env/fake_engine.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "nlohmann/json.hpp"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"
using namespace std;

namespace ge {
using Json = nlohmann::json;
namespace {
constexpr size_t kMaxSleepTimes = 15U;
class FakeLabelMaker : public LabelMaker {
 public:
  FakeLabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : LabelMaker(graph, owner) {}

  ~FakeLabelMaker() override {}

  virtual Status Run(uint32_t &label_index) { return ge::GRAPH_SUCCESS; }
};

Status InitializeHeterogeneousRuntime(const std::map<AscendString, AscendString> &options) {
  return SUCCESS;
}
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return dlsym(handle, func_name);
  }
};


int32_t g_so_addr = 0;
class MockMmpa1 : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libmodel_deployer.so") == file_name) {
      return (void *) &g_so_addr;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return dlsym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    return 0;
  }
};

class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    return nullptr;
  }
  void Free(MemBlock *block) override {
  }
};

class GeFakeOpsKernelBuilder : public OpsKernelBuilder {
 public:
  GeFakeOpsKernelBuilder() = default;

 private:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  };
  Status Finalize() override {
    return SUCCESS;
  };
  Status CalcOpRunningParam(Node &node) override {
    return SUCCESS;
  };
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    return SUCCESS;
  };
};

ge::graphStatus StubInferShape(ge::Operator &op) {
  auto x_input_desc = op.GetInputDesc(0);
  auto x_shape = x_input_desc.GetShape().GetDims();
  auto x_type = x_input_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  (void)x_input_desc.GetShapeRange(x_shape_range);
  TensorDesc op_output_desc = op.GetOutputDesc(0);
  op_output_desc.SetShape(ge::Shape(x_shape));
  op_output_desc.SetOriginShape(ge::Shape(x_shape));
  op_output_desc.SetDataType(x_type);
  if (!x_shape_range.empty()) {
    op_output_desc.SetShapeRange(x_shape_range);
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  return op_desc->UpdateOutputDesc(0, TensorAdapter::TensorDesc2GeTensorDesc(op_output_desc));
}

ge::graphStatus GetShapeInferShape(ge::Operator &op) {
  std::cout << "Enter infershape getshape" << std::endl;
  std::vector<std::string> tiling_inline_engine;
  tiling_inline_engine.push_back("AIcoreEngine");
  vector<std::string> export_shape_engine;
  export_shape_engine.push_back("AIcoreEngine");
  op.SetAttr("_op_tiling_inline_engine", tiling_inline_engine);
  op.SetAttr("_op_export_shape_engine", export_shape_engine);
  return ge::GRAPH_SUCCESS;
}

void InitEngines(std::shared_ptr<GELib> &ge_lib, GeRunningEnvFaker &ge_env) {
  //  SchedulerConf conf;
  SchedulerConf scheduler_conf;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->name = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->id = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->skip_assign_stream = true;

  scheduler_conf.cal_engines["AIcoreEngine"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["AIcoreEngine"]->name = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->id = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->independent = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->attach = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_AICPU"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->name = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->id = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->name = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->id = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->skip_assign_stream = true;

  ge_lib->DNNEngineManagerObj().schedulers_["multi_batch"] = scheduler_conf;

  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));
}
}  // namespace

REGISTER_LABEL_MAKER(PARTITIONEDCALL, FakeLabelMaker);
REGISTER_LABEL_MAKER(CASE, FakeLabelMaker);

class UtestGeApiV2 : public testing::Test {
 protected:
  void SetUp() override {
    OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Add", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});
    GetThreadLocalContext().SetGlobalOption({});
    GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().SetGraphOption({});
    GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
  }

  void TearDown() override {
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
    RuntimeStub::Reset();
  }

  void CreateSharedLibrary(const std::string &path) {
    std::ofstream ofs(path + ".cpp");
    ofs << R"(
      #include <iostream>
      extern "C" void hello() {
        std::cout << "Hello, world!" << std::endl;
      }
    )";
    ofs.close();
    std::string cmd = "g++ -shared -fPIC -o " + path + ".so " + path + ".cpp";
    system(cmd.c_str());
    std::remove((path + ".cpp").c_str());
  }
};

TEST_F(UtestGeApiV2, run_graph_with_stream) {
  vector<gert::Tensor> inputs;
  vector<gert::Tensor> outputs;
  std::map<AscendString, AscendString> options;
  GeSession session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiV2, run_graph_with_stream_not_find_graph_node) {
  GEInitializeV2({});
  vector<gert::Tensor> inputs;
  vector<gert::Tensor> outputs;
  std::map<AscendString, AscendString> options;
  GeSession session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  GEFinalizeV2();
}

TEST_F(UtestGeApiV2, build_graph_success) {
  vector<Tensor> inputs;
  std::map<AscendString, AscendString> options;
  GeSession session(options);
  auto ret = session.CompileGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiV2, ge_initialize_modify_mixlist) {
  GEFinalizeV2();
  std::map<AscendString, AscendString> options = {
    {"ge.exec.modify_mixlist", "/mixlist.json"}
  };
  Json option_name_map;
  option_name_map.emplace("ge.enableSmallChannel", "enable_small_channel");
  options.insert(pair<AscendString, AscendString>("ge.optionNameMap", AscendString(option_name_map.dump().c_str())));
  auto ret = GEInitializeV2(options);
  ASSERT_NE(ret, SUCCESS);
  GEFinalizeV2();
}

TEST_F(UtestGeApiV2, ge_initialize_fail) {
  std::map<AscendString, AscendString> options = {
    {"ge.optionInvalid", "Invalid"}
  };

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);
  auto ret = GEInitializeV2(options);
  ASSERT_EQ(ret, SUCCESS);
  auto find_log = runtime_stub.GetSlogStub().FindWarnLogEndsWith("unsupported option(ge.optionInvalid) by global level, Please check!");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  GEFinalizeV2();
}

TEST_F(UtestGeApiV2, execute_graph_with_stream) {
  vector<gert::Tensor> inputs;
  vector<gert::Tensor> outputs;
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.graphExecTimeout", "600000"));
  GeSession session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiV2, ge_not_initialized) {
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  vector<gert::Tensor> gert_inputs;
  vector<gert::Tensor> gert_outputs;
  std::map<AscendString, AscendString> options;
  std::map<AscendString, AscendString> ascend_options;
  GeSession session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, gert_inputs,gert_outputs);
  ASSERT_NE(ret, SUCCESS);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph, {}), FAILED);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), FAILED);

  EXPECT_EQ(session.AddGraphClone(graph_id, graph, {}), FAILED);
  EXPECT_EQ(session.AddGraphClone(graph_id, graph, ascend_options), FAILED);

  vector<gert::Tensor> inputs;
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  vector<gert::Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), FAILED);

  RunCallback session_callback = nullptr;
  EXPECT_EQ(session.RegisterCallBackFunc("1", session_callback), FAILED);

  EXPECT_FALSE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), FAILED);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

bool test_callback_called = false;

TEST_F(UtestGeApiV2, AddGraph_for_max_load_option) {
  std::map<AscendString, AscendString> options;
  options.emplace("ge.graphMaxParallelModelNum", "10");
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), {}), SUCCESS);
  delete session_ptr;
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ge::GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestGeApiV2, AddGraph_for_max_load_option2) {
  std::map<AscendString, AscendString> options;
  options.emplace("ge.graphMaxParallelModelNum", "-1");
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), {}), SUCCESS);
  delete session_ptr;
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ge::GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestGeApiV2, ge_session_ascend_string) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  GeSession session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session.AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), {}), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);

  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, ge_session_test) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.graphExecTimeout", "600000"));
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  GeSession session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph, {}), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);

  EXPECT_EQ(session.AddGraphClone(graph_id, graph, {}), FAILED);
  EXPECT_EQ(session.AddGraphClone(graph_id, graph, ascend_options), FAILED);

  vector<gert::Tensor> inputs;
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  vector<gert::Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_NE(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), SUCCESS); // Push to queue.

  RunCallback session_callback = nullptr;
  EXPECT_EQ(session.RegisterCallBackFunc("1", session_callback), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, ge_session_test1) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.graphExecTimeout", "-1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_GRAPH_RUN_MODE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_DEVICE_ID, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_JOB_ID, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_IS_USEHCOM, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_IS_USEHVD, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_DEPLOY_MODE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_POD_NAME, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_RANK_ID, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_RANK_TABLE_FILE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_SESSION_ID, "1"));
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  GeSession session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(session.AddGraph(graph_id, graph, {}), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);

  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  vector<gert::Tensor> inputs(1);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};

  std::vector<uint8_t> data2({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
                {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
                gert::kOnDeviceHbm,              // placement
                ge::DT_FLOAT,                    // data type
                (void *) &data2[0]};
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, ge_session_test_fail) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  options.insert(pair<AscendString, AscendString>("ge.optionInvalid", "invalid"));
  GeSession session1(options);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);
  GeSession session2(ascend_options);
  auto find_log = runtime_stub.GetSlogStub().FindWarnLogEndsWith("unsupported option(ge.optionInvalid) by session level, Please check!");
  EXPECT_TRUE(find_log >= -1);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  GEFinalizeV2();
}

TEST_F(UtestGeApiV2, OptionCheck_Failed_MultiGraphCompileAndVarialbeAcc) {
  GEFinalizeV2();
  std::map<AscendString, AscendString> options {
      {AscendString(ge::OPTION_EXEC_VARIABLE_ACC), AscendString("True")},
      {AscendString(ge::OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE), AscendString("1")},
  };
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
}

TEST_F(UtestGeApiV2, OptionCheck_Failed_MultiGraphCompileAndVarialbeAcc2) {
  std::map<AscendString, AscendString> options {
      {AscendString(ge::OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE), AscendString("1")},
  };
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  std::map<AscendString, AscendString> options2 {
      {AscendString(ge::OPTION_EXEC_VARIABLE_ACC), AscendString("True")}
  };
  GeSession session(options2);
  EXPECT_NE(session.AddGraph(1, Graph(), {}), SUCCESS);
}

TEST_F(UtestGeApiV2, AddGraph_test) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<AscendString, AscendString>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_GRAPH_RUN_MODE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_DEVICE_ID, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_JOB_ID, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_IS_USEHCOM, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_IS_USEHVD, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_DEPLOY_MODE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_POD_NAME, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_RANK_ID, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_RANK_TABLE_FILE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_SESSION_ID, "1"));
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  uint32_t graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> option;
  GeSession session(options);
  Status ret = session.AddGraph(graph_id, graph, option);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, AddGraph_test_fail) {
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

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);
  (void)session.AddGraph(graph_id, graph, ascend_options);
  (void)session.AddGraphClone(graph_id, graph, ascend_options);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  auto find_log = runtime_stub.GetSlogStub().FindWarnLogEndsWith("unsupported option(ge.optionInvalid) by graph level, Please check!");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(UtestGeApiV2, CheckOptionsValid_Invalid_JobId_test) {
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>(OPTION_EXEC_JOB_ID, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
  Status ret = ge::GEInitializeV2(options);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, CheckOptionsInvalid_test) {
  std::map<AscendString, AscendString> options = {
    {AscendString(""), AscendString("Placeholder:0;Placeholder_1:1")}};
  Status ret = ge::GEInitializeV2(options);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, GEInitialize_test) {
  std::map<AscendString, AscendString> options = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  Status ret = ge::GEInitializeV2(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);

  std::map<AscendString, AscendString> options1 = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString(nullptr)}};
  ret = ge::GEInitializeV2(options1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);

  std::map<AscendString, AscendString> options2 = {
    {AscendString("ge.autoTuneMode"), AscendString("RA")}};
  ret = ge::GEInitializeV2(options2);
  EXPECT_NE(ret, SUCCESS);

  ge::GEGetErrorMsgV3();
  ge::GEGetWarningMsgV3();
  GeSession session(options);
  GeSession session1(options1);
  GeSession session2(options2);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, GEInitialize_load_custom_pass_failed) {
  std::map<AscendString, AscendString> options;
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv("ASCEND_OPP_PATH", path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes";
  system(("mkdir -p " + custom_path).c_str());
  system(("touch " + custom_path + "/concat_pass.so").c_str());
  Status ret = ge::GEInitializeV2(options);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  system(("rm -rf " + path).c_str());
}

TEST_F(UtestGeApiV2, GEInitialize_load_custom_pass_success) {
  std::map<AscendString, AscendString> options;
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv("ASCEND_OPP_PATH", path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes/add";
  system(("mkdir -p " + custom_path).c_str());

  CreateSharedLibrary(custom_path);
  Status ret = ge::GEInitializeV2(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  system(("rm -rf " + path).c_str());
}

TEST_F(UtestGeApiV2, ge_session_info_test) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  {
    GeSession session(options);
  }
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  vector<Tensor> inputs;
  auto ret = session.CompileGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, GetCompileGraphSummary_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    GeSession session(options);
    EXPECT_EQ(session.GetCompiledGraphSummary(graph_id), nullptr); // not init
  }

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_EQ(session.GetCompiledGraphSummary(graph_id), nullptr); // not add graph
}

TEST_F(UtestGeApiV2, SetGraphConstMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    GeSession session(options);
    EXPECT_NE(session.SetGraphConstMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_NE(session.SetGraphConstMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApiV2, UpdateGraphFeatureMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    GeSession session(options);
    EXPECT_NE(session.UpdateGraphFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_NE(session.UpdateGraphFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApiV2, SetGraphFixedFeatureMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  {
    GeSession session(options);
    EXPECT_NE(session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_NE(session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApiV2, SetGraphFixedFeatureMemoryBaseWithType_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_NE(session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApiV2, UpdateGraphRefreshableFeatureMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    GeSession session(options);
    EXPECT_NE(session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_NE(session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApiV2, CompileGraph_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    GeSession session(options);
    EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS); // not add graph

  Graph graph = gert::ShareGraph::BuildSwitchMergeGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph, {}), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, CompileGraph_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({-1, -1, -1, -1}, {-1, -1, -1, -1});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[4, 1, 3, 2];1:[4, 2, 1, 2]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 8);
  ShapeEnvGuarder guard(shape_env_attr);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[0].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[1].first, Symbol(1)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[2].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[3].first, Symbol(4)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[4].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[5].first, Symbol(3)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[6].first, Symbol(1)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[7].first, Symbol(4)), true);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "s2");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "s3");

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s6");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s7");
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApiV2, CompileGraph_unknown_rank_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({-2}, {-1, -1, -1, -1});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[3, 2];1:[4, 2, 3, 2]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 7);
  ShapeEnvGuarder guard(shape_env_attr);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[0].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[1].first, Symbol(3)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[2].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[3].first, Symbol(4)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[4].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[5].first, Symbol(3)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[6].first, Symbol(2)), true);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s2");

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s6");

  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApiV2, CompileGraph_scalar_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({}, {-1, -1});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[];1:[4, 2]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 2);
  ShapeEnvGuarder guard(shape_env_attr);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[0].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[1].first, Symbol(4)), true);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 0);

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApiV2, CompileGraph_static_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({2, 3}, {2, 3});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[2, 3];1:[2, 3]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id, {}), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 0);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "3");

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApiV2, RegisterExternalAllocator_test) {
  std::map<AscendString, AscendString> options;
  uint32_t stream = 1;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  EXPECT_EQ(session.RegisterExternalAllocator(&stream, external_allocator), SUCCESS);
  EXPECT_EQ(session.UnregisterExternalAllocator(&stream), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, LoadGraph_with_graph_id) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::map<AscendString, AscendString> options_init;
  GeSession session(options_init);
  EXPECT_EQ(session.LoadGraph(graph_id, options, nullptr), FAILED);
  EXPECT_EQ(GEInitializeV2(options_init), SUCCESS);

  GeSession session1(options_init);

  options.insert(pair<AscendString, AscendString>("ge.exec.frozenInputIndexes", "1,2"));
  EXPECT_NE(session1.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, Test_LoadGraphApi) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::map<AscendString, AscendString> options_init;
  GeSession session(options_init);
  EXPECT_EQ(session.LoadGraph(graph_id, options, nullptr), FAILED);
  EXPECT_EQ(GEInitializeV2(options_init), SUCCESS);

  GeSession session1(options_init);

  options.insert(pair<AscendString, AscendString>("ge.exec.frozenInputIndexes", "1,2"));
  EXPECT_NE(session1.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, run_graph_with_stream_async) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<gert::Tensor> inputs(2);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};
  std::vector<uint8_t> data1({1, 2, 3, 4});
  inputs[1] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data1[0]};

  std::vector<uint8_t> data3({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data3[0]};
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id, {}), SUCCESS);
  ASSERT_NE(session.GetCompiledGraphSummary(graph_id), nullptr);

  EXPECT_EQ(session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, (void *)0x3558, 4000000), SUCCESS);

  auto ret = session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, load_graph_not_compile) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<gert::Tensor> inputs(2);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};
  std::vector<uint8_t> data1({1, 2, 3, 4});
  inputs[1] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data1[0]};

  std::vector<uint8_t> data3({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data3[0]};
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));
  session.LoadGraph(graph_id, {}, nullptr);
  EXPECT_FALSE(session.IsGraphNeedRebuild(graph_id));
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, run_graph_with_stream_with_dynamic) {
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
  OperatorFactoryImpl::RegisterInferShapeFunc("Add", [](Operator &op) {return GRAPH_SUCCESS;});
  OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});

  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);

  vector<gert::Tensor> inputs(1);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};

  std::vector<uint8_t> data2({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data2[0]};

  std::map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options[ge::OO_LEVEL] = "O3";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id, {}), SUCCESS);
  ASSERT_NE(session.GetCompiledGraphSummary(graph_id), nullptr);
  // dynamic shape graph
  EXPECT_EQ(session.GetCompiledGraphSummary(graph_id)->IsStatic(), false);

  rtStream_t stream = (void*)0x01;
  auto ret = session.RunGraphWithStreamAsync(1, stream, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, run_graph_with_stream_with_multi_batch) {
  vector<gert::Tensor> inputs(1);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};

  std::vector<uint8_t> data2({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
                {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
                gert::kOnDeviceHbm,              // placement
                ge::DT_FLOAT,                    // data type
                (void *) &data2[0]};

  std::map<AscendString, AscendString> options;
  options["ge.inputShape"] = "data1:-1,-1,-1;data2:-1,-1,-1";
  options["ge.dynamicDims"] = "1,1,1,1,1,1;3,3,3,3,3,3;5,5,5,5,5,5";
  options["ge.dynamicNodeType"] = "1";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);

  GraphId graph_id = 4;
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id, {}), SUCCESS);
  auto summary = session.GetCompiledGraphSummary(graph_id);
  ASSERT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{5, 5, 5};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, CompileGraph_Success_MultiGraphParallelCompile) {
  vector<gert::Tensor> inputs(1);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};

  std::vector<uint8_t> data2({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
                {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
                gert::kOnDeviceHbm,              // placement
                ge::DT_FLOAT,                    // data type
                (void *) &data2[0]};

  std::map<AscendString, AscendString> options;
  options[OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE] = "1";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  GeSession session(options);
  size_t thread_num = 15U;
  std::map<uint32_t, Graph> graph_id_2_maps;
  GraphId graph_id = 11010;
  for (size_t i = 0U; i < thread_num; i++) {
    ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
    auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
    EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
    graph_id_2_maps[graph_id] = graph;
    ++graph_id;
  }

  std::vector<std::thread> threads;
  for (auto &id_graph : graph_id_2_maps) {
    threads.emplace_back(std::thread([&session, &id_graph]() {
      EXPECT_EQ(session.CompileGraph(id_graph.first, {}), SUCCESS);
    }));
  }
  for (auto &thread : threads) {
    thread.join();
  }
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, GetCompiledGraph_Success) {
  vector<gert::Tensor> inputs(1);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};

  std::vector<uint8_t> data2({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
                {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
                gert::kOnDeviceHbm,              // placement
                ge::DT_FLOAT,                    // data type
                (void *) &data2[0]};

  std::map<AscendString, AscendString> options;
  options["ge.inputShape"] = "data1:-1,-1,-1;data2:-1,-1,-1";
  options["ge.dynamicDims"] = "1,1,1,1,1,1;3,3,3,3,3,3;5,5,5,5,5,5";
  options["ge.dynamicNodeType"] = "1";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);

  GraphId graph_id = 4;
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id, {}), SUCCESS);
  ModelBufferData model_buffer;
  EXPECT_EQ(session.GetCompiledModel(graph_id, model_buffer), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, GetCompiledGraph_Failed_InvalidOption) {
  vector<gert::Tensor> inputs(1);
  vector<gert::Tensor> outputs(1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  inputs[0] = {{{3, 3, 3}, {3, 3, 3}},          // shape
               {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
               gert::kOnDeviceHbm,              // placement
               ge::DT_FLOAT,                    // data type
               (void *) &data[0]};

  std::vector<uint8_t> data2({1, 2, 3, 4});
  outputs[0] = {{{1, 3, 3}, {1, 3, 3}},          // shape
                {FORMAT_NCHW, FORMAT_NCHW, {}},  // format
                gert::kOnDeviceHbm,              // placement
                ge::DT_FLOAT,                    // data type
                (void *) &data2[0]};

  std::map<AscendString, AscendString> options;
  options["ge.inputShape"] = "data1:-1,-1,-1;data2:-1,-1,-1";
  options["ge.dynamicDims"] = "1,1,1,1,1,1;3,3,3,3,3,3;5,5,5,5,5,5";
  options["ge.dynamicNodeType"] = "1";
  options["ge.exec.variable_acc"] = "True";
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);

  GraphId graph_id = 4;
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id, {}), SUCCESS);
  ModelBufferData model_buffer;
  EXPECT_NE(session.GetCompiledModel(graph_id, model_buffer), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, GetCompiledGraph_Failed_NotCompiled) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  GraphId graph_id = 4;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  ModelBufferData model_buffer;
  EXPECT_NE(session.GetCompiledModel(graph_id, model_buffer), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, GetCompiledGraph_Failed_GraphIdInvalid) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);
  GeSession session(options);
  ModelBufferData model_buffer;
  EXPECT_NE(session.GetCompiledModel(10010, model_buffer), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, profiling_option_fail) {
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  std::map<AscendString, AscendString> options;
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_MODE, "1"));
  options.insert(pair<AscendString, AscendString>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, CheckOptionsValid_featureBaseRefreshable) {
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "2");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  EXPECT_NE(GEInitializeV2(options), SUCCESS);
}

TEST_F(UtestGeApiV2, Construct_session) {
  GEFinalizeV2();
  std::map<AscendString, AscendString> options;
  GeSession sess1(options);  // ge not initialized

  std::map<AscendString, AscendString> ascend_options;
  GeSession sess2(ascend_options);  // ge not initialized

  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  ascend_options[AscendString()] = "";  // option key is empty
  GeSession sess3(ascend_options);

  options["ge.exec.precision_mode"] = "invalid";  // invalid option value
  GeSession sess4(options);

  std::map<AscendString, AscendString> ascend_options1;
  ascend_options1[AscendString("ge.exec.precision_mode")] = "invalid";  // invalid option value
  GeSession sess5(ascend_options1);

  // add graph test
  std::map<AscendString, AscendString> options1;
  GeSession sess6(options1);  // contruct session successfully
  Graph g("hello");
  std::map<AscendString, AscendString> graph_options;
  graph_options[AscendString()] = "";  // graph option key is empty
  EXPECT_EQ(sess6.AddGraph(1, g, graph_options), FAILED);
}

TEST_F(UtestGeApiV2, ge_session_oo_init) {
  std::map<AscendString, AscendString> global_options;
  global_options.emplace(OO_LEVEL, "O3");
  global_options.emplace(OO_CONSTANT_FOLDING, "false");
  EXPECT_EQ(GEInitializeV2(global_options), SUCCESS);

  std::map<AscendString, AscendString> session_options;
  session_options.emplace(OO_LEVEL, "O3");
  session_options.emplace(OO_CONSTANT_FOLDING, "false");
  GeSession session(session_options);

  GraphId graph_id = 1;
  ComputeGraphPtr compute_graph = gert::ShareGraph::AicoreGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(OO_LEVEL, "O1");
  graph_options.emplace(OO_CONSTANT_FOLDING, "true");
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  vector<gert::Tensor> inputs;
  vector<gert::Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  std::string opt_value;
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApiV2, GEInitialize_oo_init_param_invalid) {
  std::map<AscendString, AscendString> global_options;
  global_options.emplace(OO_LEVEL, "O4");
  global_options.emplace(OO_CONSTANT_FOLDING, "false");
  EXPECT_NE(GEInitializeV2(global_options), SUCCESS);

  global_options[OO_LEVEL] = "O1";
  global_options[OO_CONSTANT_FOLDING] = "False";
  EXPECT_NE(GEInitializeV2(global_options), SUCCESS);

  global_options[OO_LEVEL] = "O1";
  global_options[OO_CONSTANT_FOLDING] = "0";
  EXPECT_NE(GEInitializeV2(global_options), SUCCESS);

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApiV2, Session_oo_init_param_invalid) {
  std::map<AscendString, AscendString> global_options;
  EXPECT_EQ(GEInitializeV2(global_options), SUCCESS);

  std::map<AscendString, AscendString> session_options;
  GeSession session(session_options);


  GraphId graph_id = 1;
  ComputeGraphPtr compute_graph = gert::ShareGraph::AicoreGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<AscendString, AscendString> graph_options;
  graph_options[OO_LEVEL] = "O4";
  graph_options[OO_CONSTANT_FOLDING] = "false";
  EXPECT_NE(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  graph_options[OO_LEVEL] = "O1";
  graph_options[OO_CONSTANT_FOLDING] = "False";
  EXPECT_NE(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  graph_options[OO_LEVEL] = "O1";
  graph_options[OO_CONSTANT_FOLDING] = "0";
  EXPECT_NE(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  GEFinalizeV2();
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApiV2, RunGraph_Success_CallBack) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;

  size_t call_count = 0U;
  auto call_back = [&call_count](uint32_t graph_id, const std::map<AscendString, gert::Tensor>& params_list) {
    (void)graph_id;
    (void)params_list;
    call_count++;
    return SUCCESS;
  };
  session.RegisterCallBackFunc("Save", call_back);
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  EXPECT_EQ(call_count, 1U);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraphAsync_Success_WithCompileAndLoad) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(1), SUCCESS);
  EXPECT_EQ(session.LoadGraph(1, {}, nullptr), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;

  size_t call_count = 0U;
  auto call_back = [&call_count](uint32_t graph_id, std::vector<gert::Tensor> &outputs) {
    (void)graph_id;
    (void)outputs;
    call_count++;
    return SUCCESS;
  };
  EXPECT_EQ(session.RunGraphAsync(1, inputs, call_back), SUCCESS);
  size_t sleep_times = 0U;
  while (call_count == 0U) {
    sleep(1);
    if (++sleep_times > kMaxSleepTimes) {
      break;
    }
  }
  EXPECT_EQ(call_count, 1U);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraphAsync_Success_OnlyAdd) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  const auto session_ptr = new GeSession(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), {}), SUCCESS);

  std::vector<gert::Tensor> inputs;

  // invalid graph id
  // RunGraphAsync submit failed
  test_callback_called = false;
  auto callback = [](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_NE(status, SUCCESS);
    test_callback_called = true;
  };

  // get graph_node fail
  EXPECT_NE(session_ptr->RunGraphAsync(10, inputs, callback), SUCCESS);

  // after RunGraphAsync run failed before, RunGraphAsync submit success
  EXPECT_EQ(session_ptr->RunGraphAsync(graph_id, inputs, callback), SUCCESS);
  sleep(1);  // wait callback
  EXPECT_EQ(test_callback_called, true);
  delete session_ptr;
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}

TEST_F(UtestGeApiV2, RunGraphAsync_Success_WithOnlyCompile) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(1), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;

  size_t call_count = 0U;
  auto call_back = [&call_count](uint32_t graph_id, std::vector<gert::Tensor> &outputs) {
    (void)graph_id;
    (void)outputs;
    call_count++;
    return SUCCESS;
  };
  EXPECT_EQ(session.RunGraphAsync(1, inputs, call_back), SUCCESS);
  size_t sleep_times = 0U;
  while (call_count == 0U) {
    sleep(1);
    if (++sleep_times > kMaxSleepTimes) {
      break;
    }
  }
  EXPECT_EQ(call_count, 1U);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraphAsync_Success_WithOnlyLoad) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);
  // EXPECT_EQ(session.CompileGraph(1), SUCCESS);
  EXPECT_EQ(session.LoadGraph(1, {}, nullptr), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;

  size_t call_count = 0U;
  auto call_back = [&call_count](uint32_t graph_id, std::vector<gert::Tensor> &outputs) {
    (void)graph_id;
    (void)outputs;
    call_count++;
    return SUCCESS;
  };
  EXPECT_EQ(session.RunGraphAsync(1, inputs, call_back), SUCCESS);
  size_t sleep_times = 0U;
  while (call_count == 0U) {
    sleep(1);
    if (++sleep_times > kMaxSleepTimes) {
      break;
    }
  }
  EXPECT_EQ(call_count, 1U);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraph_Success_WithCompileAndLoadGraph) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(1), SUCCESS);
  EXPECT_EQ(session.LoadGraph(1, {}, nullptr), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraph_Success_OnlyAdd) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraph_Success_WithOnlyLoadGraph) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);
  EXPECT_EQ(session.LoadGraph(1, {}, nullptr), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraph_Success_WithOnlyCompileGraph) {
  std::map<AscendString, AscendString> options;
  options[ge::OPTION_HOST_ENV_OS] = "linux";
  options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  ASSERT_NE(instance_ptr, nullptr);
  GeRunningEnvFaker ge_env;
  InitEngines(instance_ptr, ge_env);

  ge_env.InstallDefault();
  DEF_GRAPH(g1) {
    CHAIN(NODE("Save", VARIABLE)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  GeSession session(options);
  EXPECT_EQ(session.AddGraph(1, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(1), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(session.RunGraph(1, inputs, outputs), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST_F(UtestGeApiV2, RunGraph_Failed_InvalidGraphId) {
  std::map<AscendString, AscendString> options;
  auto init_status = ge::GEInitializeV2(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << std::endl;
  }

  GeSession session(options);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_NE(session.RunGraph(1, inputs, outputs), SUCCESS);
}

TEST_F(UtestGeApiV2, Session_export_compile_stat_valid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  std::map<AscendString, AscendString> global_options;
  std::string opt_value;
  global_options[OPTION_EXPORT_COMPILE_STAT] = "0";
  GEFinalizeV2();
  EXPECT_EQ(GEInitializeV2(global_options), SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "0");

  std::map<AscendString, AscendString> session_options;
  session_options[OPTION_EXPORT_COMPILE_STAT] = "1";
  GeSession session(session_options);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "1");

  GraphId graph_id = 1;
  ComputeGraphPtr compute_graph = gert::ShareGraph::AicoreGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXPORT_COMPILE_STAT] = "2";
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "2");

  vector<gert::Tensor> inputs;
  vector<gert::Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "2");
  std::string oo_value;
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OPTION_EXPORT_COMPILE_STAT, oo_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(oo_value, "2");

  GEFinalizeV2();
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApiV2, Session_export_compile_stat_invalid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  std::map<AscendString, AscendString> global_options;
  std::string opt_value;
  global_options[OPTION_EXPORT_COMPILE_STAT] = "3";
  EXPECT_NE(GEInitializeV2(global_options), SUCCESS);
  EXPECT_NE(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

namespace {
  class AbnormalRtsStub : public RuntimeStub {
  public:
    rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) override {
      return 1; // failed
    }
  };
} // namespace
  /**
   * 若session创建失败，确保session manager没有残留的未成功创建的session
   */
TEST_F(UtestGeApiV2, CreateSessionFailed) {
  auto rts_stub = std::make_shared<AbnormalRtsStub>();
  RuntimeStub::Install(rts_stub.get());

  GEFinalizeV2();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitializeV2(options), SUCCESS);

  GeSession sess1(options); // rt create context failed
  Graph tmp_graph;
  EXPECT_NE(sess1.AddGraph(1, tmp_graph, {}), SUCCESS);
  EXPECT_EQ(SessionUtils::NumSessions(), 0);

  GeSession sess2(options);  // rt create context failed
  EXPECT_NE(sess1.AddGraph(2, tmp_graph, {}), SUCCESS);
  EXPECT_EQ(SessionUtils::NumSessions(), 0);

  RuntimeStub::UnInstall(rts_stub.get());
}

#define EXPECT_STR_EQ(x, y) EXPECT_EQ(std::string(x.GetString()), std::string(y))

REG_OP(QueryIrTestOp1)
  .INPUT(required_x1, TensorType::ALL())
  .OPTIONAL_INPUT(optional_x2, TensorType::ALL())
  .DYNAMIC_INPUT(dynamic_x3, TensorType::ALL())
  .OUTPUT(required_y1, TensorType::ALL())
  .DYNAMIC_OUTPUT(dynamic_y1, TensorType::ALL())
  .OP_END_FACTORY_REG(QueryIrTestOp1)

TEST_F(UtestGeApiV2, QueryIrInputOutput) {
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

TEST_F(UtestGeApiV2, QueryIrInputOutputKeepOrder) {
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

TEST_F(UtestGeApiV2, QueryIrAttr) {
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

TEST_F(UtestGeApiV2, QueryIrAttrKeepOrder) {
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

TEST_F(UtestGeApiV2, QueryUnregisteredIr) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_NE(GetRegisteredIrDef("QueryIrTestOpNotRegistered", inputs, outputs, attrs), SUCCESS);
}

TEST_F(UtestGeApiV2, QuerySameVersionIr) {
  EXPECT_TRUE(::IsIrRepSupport(INFERENCE_RULE));
  EXPECT_FALSE(::IsIrRepSupport("future_new_feature_rule"));
  EXPECT_FALSE(::IsIrRepSupport(""));
}

TEST_F(UtestGeApiV2, GEInitialize_long_option_value) {
  // 测试当 option value 长度超过 800 字符时，能够正常打印
  std::string long_value(900, 'a');  // 创建 900 字符的长字符串
  std::map<AscendString, AscendString> options = {
    {AscendString("ge.test.long_option"), AscendString(long_value.c_str())}
  };
  Status ret = ge::GEInitializeV2(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);

  // 测试 value 长度不超过 800 字符的情况
  std::string short_value(100, 'b');
  std::map<AscendString, AscendString> options2 = {
    {AscendString("ge.test.short_option"), AscendString(short_value.c_str())}
  };
  ret = ge::GEInitializeV2(options2);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
}
}  // namespace ge
