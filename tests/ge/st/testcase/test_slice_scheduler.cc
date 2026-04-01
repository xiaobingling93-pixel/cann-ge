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
#include <gmock/gmock.h>
#include <condition_variable>
#include <mutex>
#include <future>
#include "graph/ge_context.h"
#include "graph/load/model_manager/model_manager.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "register/op_tiling_registry.h"
#include "api/gelib/gelib.h"

#include "graph/operator_reg.h"
#include "graph/ge_attr_value.h"
#include "ge_graph_dsl/assert/filter_scope_guard.h"

#include "ge/ge_api.h"
#include "graph/utils/tensor_adapter.h"
#include "init_ge.h"
#include "utils/graph_factory.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"

#include "api/atc/main_impl.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "mmpa/mmpa_api.h"
#include "faker/global_data_faker.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "register/register_custom_pass.h"
#include "stub/gert_runtime_stub.h"
#include "common/env_path.h"
#include "ge/st/testcase/common_setup.h"

namespace ge {
namespace {
class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD7(rtKernelLaunchWithFlagV2,
               int32_t(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                       rtStream_t stream, uint32_t flag, const rtTaskCfgInfo_t *cfgInfo));
};

bool GuardCheckHitFunc(gert::Tensor **tensors, size_t num_tensors, char_t *reason, size_t reason_size)
{
  return true;
}
class MockMmpaSliceGuardHit : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string(file_name).find("/proc/self/fd/") != std::string::npos) {
      return (void *) 0x8888;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "GuardCheckFunc") {
      return (void *)&GuardCheckHitFunc;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {

    if (handle == (void *) 0x8888) {
      return 0;
    }
    return dlclose(handle);
  }

  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    strncpy(realPath, path, realPathLen);
    return 0;
  }
};

void InitGeLib() {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

/*
Status GenerateTaskForTaskWithHandle(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTaskWithHandle());
  return SUCCESS;
}
*/

template <typename Func, typename... Args>
using invoke_result = typename std::result_of<Func(Args...)>::type;

template <typename Func, typename... Args>
std::map<int, invoke_result<Func, Args...>> ConcurrentProcess(Func&& func, int thread_nums, Args&&... args) {
  typedef invoke_result<Func, Args...> ResultType;
  std::map<int, std::future<ResultType>> futureMap;

  for (int i = 0; i < thread_nums; ++i) {
    futureMap.insert(std::make_pair(i,std::async(std::launch::async, std::bind(func, std::forward<Args>(args)...))));
  }
  std::map<int, ResultType> resultMap;
  for (auto& entry : futureMap) {
    ResultType result = entry.second.get();
    resultMap.insert(std::make_pair(entry.first, result));
  }
  return resultMap;
}

Status EXPECT_AddGraph(std::mutex &mutex, Session &session, GraphId graph_id, Graph &recover_ir_graph, std::map<AscendString, AscendString> &graph_options) {
  std::lock_guard<std::mutex> lock(mutex);
  GELOGT(TRACE_RUNNING, "Thread %d AddGraph start session addr is %d", std::this_thread::get_id(), &session);
  Status ret = session.AddGraph(graph_id, recover_ir_graph, graph_options);
  EXPECT_EQ(ret, SUCCESS);
  GELOGT(TRACE_RUNNING, "Thread %d AddGraph end", std::this_thread::get_id());
  return ret;
}
std::vector<Tensor> EXPECT_RunGraphAsync_withStatus(std::mutex &mutex, Session &session, GraphId graph_id, const std::vector<Tensor> &inputs, Status expectStatus) {
  std::lock_guard<std::mutex> lock(mutex);
  GELOGT(TRACE_RUNNING, "Thread %d RunGraphAsync start session addr is %d", std::this_thread::get_id(), &session);
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  Status ret = SUCCESS;
  std::vector<Tensor> outputs;
  RunAsyncCallback callback = [&](Status status, std::vector<ge::Tensor> &output_tensors) {
    std::unique_lock<std::mutex> lk(mu);
    ret = status;
    outputs = output_tensors;
    done = true;
    cv.notify_all();
  };

  ret = session.RunGraphAsync(graph_id, inputs, callback);
  if (ret == SUCCESS) {
    std::unique_lock<std::mutex> lk(mu);
    if (!cv.wait_for(lk, std::chrono::seconds(15), [&]() { return done; })) {
      ret = FAILED;
    }
  }
  EXPECT_EQ(ret, expectStatus);
  GELOGT(TRACE_RUNNING, "Thread %d RunGraphAsync end", std::this_thread::get_id());
  return outputs;
}

std::vector<Tensor> EXPECT_RunGraphAsync(std::mutex &mutex, Session &session, GraphId graph_id, const std::vector<Tensor> &inputs) {
  return EXPECT_RunGraphAsync_withStatus(mutex, session, graph_id, inputs, SUCCESS);
}
}  // namespace

class SliceSchedulerTest : public testing::Test {
public:
    char old_opp_path_env_[MMPA_MAX_PATH] = {'\0'};
    char old_ld_path_env_[MMPA_MAX_PATH] = {'\0'};
protected:
void SetUp() {
  InitGeLib();
  CommonSetupUtil::CommonSetup();
  char runtime2_env[MMPA_MAX_PATH] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));

  // 开启自动融合、编译guard so 所需的环境变量
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env_, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env_, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1); // 开启自动融合


  /*    MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);*/

    runtime_stub_.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub_.GetSlogStub().Clear();
    runtime_stub_.GetKernelStub().StubTiling();
    RuntimeStub::Install(nullptr); // gert的rts stub不能在多线程环境下工作，因此使用默认rts stub
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaSliceGuardHit>());
}

void TearDown() {
    //GEFinalize();
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
    unsetenv("AUTOFUSE_FLAGS");

    runtime_stub_.GetSlogStub().Clear();
    MockRuntime::Reset();
    MmpaStub::GetInstance().Reset();
}
  gert::GertRuntimeStub runtime_stub_;
};
namespace {
Tensor CreateTensor(const TensorDesc &tensor_desc) {
    int64_t tensor_size = -1;
    TensorUtils::GetTensorSizeInBytes(TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), tensor_size);
    std::vector<uint8_t> tensor_buffer(tensor_size);
    Tensor tensor(tensor_desc);
    tensor.SetData(std::move(tensor_buffer));
    return tensor;
  }

Tensor CreateTensor(const std::vector<int64_t> &dims, Format format = FORMAT_ND, DataType data_type = DT_FLOAT) {
    auto tensor_desc = TensorDesc(Shape(dims), format, data_type);
    tensor_desc.SetOriginShape(Shape(dims));
    tensor_desc.SetOriginFormat(format);
    return CreateTensor(tensor_desc);
}

Status RunGraphAsync(Session &session, uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    Status ret = SUCCESS;
    RunAsyncCallback callback = [&](Status status, std::vector<ge::Tensor> &output_tensors) {
        std::unique_lock<std::mutex> lk(mu);
        ret = status;
        outputs = std::move(output_tensors);
        done = true;
        std::cout<< "callback" <<std::endl;
        lk.unlock();
        cv.notify_one();
    };

    auto run_ret = session.RunGraphAsync(graph_id, inputs, callback);
    if (run_ret != SUCCESS) {
      return run_ret;
    }
    std::cout<< "run success" <<std::endl;

    std::unique_lock<std::mutex> lk(mu);
    if (!cv.wait_for
         (lk, std::chrono::seconds(15), [&]() { return done; })) {
        std::cout<< "finish wait1" <<std::endl;
        return SUCCESS; 
    }
    std::cout<< "finish wait" <<std::endl;
    return ret;
}

bool CheckLogExist(gert::GertRuntimeStub &runtime_stub, const std::vector<std::string> &expect_log_list) {
  for(auto &log : expect_log_list) {
    std::cout << log << std::endl;
    EXPECT_NE(runtime_stub.GetSlogStub().FindLog(-1, log.c_str()), -1);
  }
  return true;
}
bool CheckLogNotExist(gert::GertRuntimeStub &runtime_stub, const std::vector<std::string> &unexpect_log_list) {
  for(auto &log : unexpect_log_list) {
    std::cout << log << std::endl;
    EXPECT_EQ(runtime_stub.GetSlogStub().FindLog(-1, log.c_str()), -1);
  }
  return true;
}
}  // namespace

/*TEST_F(SliceSchedulerTest, TestSliceScheduler_ForDynamicGraph_EnableDynamicBatch) {
    auto compute_graph = GraphFactory::BuildDynamicInputGraph();

    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";  
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow";  
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    gert::GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub.GetSlogStub().Clear();

    Session session(options);
    GraphId graph_id = 2;
    std::map<AscendString, AscendString> graph_options;
    EXPECT_EQ(session.AddGraph(graph_id, compute_graph, graph_options), SUCCESS);

    std::vector<Tensor> inputs;
    inputs.emplace_back(CreateTensor({16}));  
    inputs.emplace_back(CreateTensor({16}));

    EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

    std::vector<Tensor> outputs;
    EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
    const std::string log = "Start to commit user graph execution task";
    EXPECT_EQ(runtime_stub.GetSlogStub().FindLog(DLOG_DEBUG, log.c_str()), -1);

    // session.RemoveGraph(graph_id);
    runtime_stub.GetSlogStub().Clear();
}*/
    
/*TEST_F(SliceSchedulerTest, TestSliceScheduler_ForStaticGraph) {
    MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    auto recover_ir_graph = GraphFactory::BuildStaticInputGraph();
    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";  
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; 
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    std::map<AscendString, AscendString> graph_options;
    gert::GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub.GetSlogStub().Clear();
    {
      Session session(options);
      GraphId graph_id = 3;
      EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);
      //EXPECT_EQ(GetDump(recover_ir_graph, "StaticInputGraph"), SUCCESS);

      std::vector<Tensor> inputs;
      inputs.emplace_back(CreateTensor({1, 1, 24, 24}));
      inputs.emplace_back(CreateTensor({1, 1, 24, 24}));
      std::vector<Tensor> outputs;
      EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

      EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
      // no slice here
      const std::string log = "GetRemainingNodes completed. infer size:4, uninfer size:0";
      EXPECT_NE(runtime_stub.GetSlogStub().FindLog(-1, log.c_str()), -1);

      session.RemoveGraph(graph_id);
    }
    runtime_stub.GetSlogStub().Clear();
}*/
/*

┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────────┐  (0,0)   ┌─────────────┐
│ _arg_0 │ ───────> │  add   │ ───────> │ unique │ ───────> │   mul   │ ───────> │ Node_Output │
└────────┘          └────────┘          └────────┘          └─────────┘          └─────────────┘
                      ∧                                       ∧
                      │ (0,1)                                 │ (0,1)
                      │                                       │
                    ┌────────┐                              ┌─────────┐
                    │ _arg_1 │                              │ const_0 │
                    └────────┘                              └─────────┘

                                                                                       */
TEST_F(SliceSchedulerTest, TestSliceScheduler_ForDynamicGraph) {
    auto recover_ir_graph = GraphFactory::BuildDynamicInputGraph();
    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; 
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    
    std::map<AscendString, AscendString> graph_options;
    runtime_stub_.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub_.GetSlogStub().Clear();
    {
      setenv("DUMP_GE_GRAPH", "2", 1);
      Session session(options);
      GraphId graph_id = 4;
      EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);

      std::vector<Tensor> inputs;
      inputs.emplace_back(CreateTensor({16}));
      inputs.emplace_back(CreateTensor({16}));
      std::vector<Tensor> outputs;
      EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

      EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);

      std::vector<std::string> expect_log_list = {
          "GetRemainingNodes completed. infer size:5, uninfer size:2",
          "Start to compile GEP[0] for EP[0]",
          "ExecuteGraphWithStreamAsync GEP[ins_id:0] of EP[0] USER_GRAPH[4]",
          "GetRemainingNodes completed. infer size:4, uninfer size:0",
          "Start to compile GEP[1] for EP[1]",
          "ExecuteGraphWithStreamAsync GEP[ins_id:1] of EP[1] USER_GRAPH[4]"
      };
      CheckLogExist(runtime_stub_, expect_log_list);
      runtime_stub_.GetSlogStub().Clear();

      // tests guard cache hit
      EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
      std::vector<std::string> expect_log_list1 = {
          "ExecuteGraphWithStreamAsync GEP[ins_id:0] of EP[0] USER_GRAPH[4]",
          "ExecuteGraphWithStreamAsync GEP[ins_id:1] of EP[1] USER_GRAPH[4]"
      };
      std::vector<std::string> unexpect_log_list1 = {
          "Start to compile GEP",
          "GetRemainingNodes completed"
      };
      CheckLogExist(runtime_stub_, expect_log_list1);
      CheckLogNotExist(runtime_stub_, unexpect_log_list1);
      runtime_stub_.GetSlogStub().Clear();

      // todo tests guard cache miss
      runtime_stub_.GetSlogStub().SetLevel(DLOG_ERROR);
      session.RemoveGraph(graph_id);
    }
    runtime_stub_.GetSlogStub().Clear();
}

TEST_F(SliceSchedulerTest, TestSliceScheduler_ForDynamicGraph_MultiInstance) {
  auto ir_graph = GraphFactory::BuildDynamicInputGraph();

  std::map<AscendString, AscendString> session_options;
  session_options[OPTION_GRAPH_RUN_MODE] = "1";
  session_options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
  session_options[OPTION_EXEC_DUMP_PATH] = "./";
  session_options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow";
  session_options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  session_options[JIT_COMPILE.c_str()] = "1";

  runtime_stub_.GetSlogStub().SetLevel(DLOG_DEBUG);
  runtime_stub_.GetSlogStub().Clear();
  {
    std::map<AscendString, AscendString> graph_options;
    Session session(session_options);
    GraphId graph_id = 9;
    std::vector<Tensor> inputs;
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    // add graph
    std::mutex mtx;
    std::map<int, Status> addResultMap = ConcurrentProcess(EXPECT_AddGraph, 3, std::ref(mtx), std::ref(session),
                                                           graph_id, std::ref(ir_graph), std::ref(graph_options));
    // build graph
    EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
    // run graph
    std::map<int, std::vector<Tensor>> resultMap =
        ConcurrentProcess(EXPECT_RunGraphAsync, 3, std::ref(mtx), std::ref(session), graph_id, inputs);
    std::vector<std::string> expect_log_list = {
        "ExecuteGraphWithStreamAsync GEP[ins_id:0] of EP[0] USER_GRAPH[9]",
        "ExecuteGraphWithStreamAsync GEP[ins_id:1] of EP[1] USER_GRAPH[9]"
    };
    for(auto &log : expect_log_list) {
      std::cout << log << std::endl;
      EXPECT_EQ(runtime_stub_.GetSlogStub().CountLog(-1, log.c_str()), 3);
    }
    session.RemoveGraph(graph_id);
  }
  runtime_stub_.GetSlogStub().Clear();
}

/*// 构图中const有问题，无法验证
TEST_F(SliceSchedulerTest, TestSliceScheduler_ForDynamicGraph_WithMultiType2OP) {
    *//*MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);*//*
    auto recover_ir_graph = GraphFactory::BuildEnhancedUniqueGraph();
    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; 
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    std::map<AscendString, AscendString> graph_options;
    runtime_stub_.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub_.GetSlogStub().Clear();

    Session session(options);
    GraphId graph_id = 5;
    EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);
    EXPECT_EQ(GetDump(recover_ir_graph, "SuperDeepNetGraph"), SUCCESS);

    std::vector<Tensor> inputs;
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    std::vector<Tensor> outputs;
    EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

    EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
    
    EXPECT_EQ(LogCheck(runtime_stub_, true, true, true), SUCCESS);
    // session.RemoveGraph(graph_id);
    runtime_stub_.GetSlogStub().Clear();
}

TEST_F(SliceSchedulerTest, TestSliceScheduler_ForDynamicGraph_WithVarAndConst) {
    MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
    auto graph_1 = GraphFactory::BuildDynamicInputGraphWithVarAndConst();
    auto graph_2 = GraphFactory::BuildDynamicInputGraphWithVarAndConst2();
    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1"; 
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; 
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    std::map<AscendString, AscendString> graph_options;
    gert::GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub.GetSlogStub().Clear();

    Session session(options);
    GraphId graph_id_1 = 71;
    GraphId graph_id_2 = 72;
    std::vector<Tensor> inputs;
    inputs.emplace_back(CreateTensor({16}));

    std::vector<Tensor> outputs_1;
    EXPECT_EQ(session.AddGraph(graph_id_1, graph_1, graph_options), SUCCESS);
    EXPECT_EQ(GetDump(graph_1, "DynamicInputGraphWithVarAndConst"), SUCCESS);
    EXPECT_EQ(session.BuildGraph(graph_id_1, inputs), SUCCESS);
    EXPECT_EQ(RunGraphAsync(session, graph_id_1, inputs, outputs_1), SUCCESS);
    EXPECT_EQ(LogCheck(runtime_stub, true, true, true), SUCCESS);    
    runtime_stub.GetSlogStub().Clear();

    std::vector<Tensor> outputs_2;
    EXPECT_EQ(session.AddGraph(graph_id_2, graph_2, graph_options), SUCCESS);
    EXPECT_EQ(GetDump(graph_2, "DynamicInputGraphWithVarAndConst1"), SUCCESS);
    EXPECT_EQ(session.BuildGraph(graph_id_2, inputs), SUCCESS);
    EXPECT_EQ(RunGraphAsync(session, graph_id_2, inputs, outputs_2), SUCCESS);
    EXPECT_EQ(LogCheck(runtime_stub, true, true, true), SUCCESS);    

    //session.RemoveGraph(graph_id_1);

    std::vector<Tensor> outputs_3;
    EXPECT_NE(RunGraphAsync(session, graph_id_1, inputs, outputs_3), SUCCESS);

    //session.RemoveGraph(graph_id_2);
    runtime_stub.GetSlogStub().Clear();
}
TEST_F(SliceSchedulerTest, TestSliceScheduler_ForDynamicGraph_WithException) {
    // Mock 相关生成任务
    MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
  
    std::map<AscendString, AscendString> options;
    options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
    options.emplace(ge::OPTION_CONST_LIFECYCLE, "graph");
    options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
    options.emplace(JIT_COMPILE.c_str(), "1");
    gert::GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub.GetSlogStub().Clear();

    Session session1(options);
    auto graph = GraphFactory::BuildDynamicInputGraph();
    uint32_t graph_id = 11;
    std::vector<ge::Tensor> ge_inputs;
    std::vector<ge::Tensor> ge_outputs;
    EXPECT_EQ(session1.AddGraph(graph_id, graph), SUCCESS);
    dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
    
    //incorrect build
    std::vector<Tensor> build_inputs;
    build_inputs.emplace_back(CreateTensor({16}));
    build_inputs.emplace_back(CreateTensor({16}));
    //空graph
    EXPECT_NE(SUCCESS, session1.BuildGraph(999, build_inputs));
    //session1.RemoveGraph(graph_id);

    //incorrect run
    Session session2(options);
    auto graph2 = GraphFactory::BuildDynamicInputGraph();
    EXPECT_EQ(session2.AddGraph(graph_id , graph2), SUCCESS);
    EXPECT_EQ(SUCCESS, session2.BuildGraph(graph_id, build_inputs));
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
    // incorrect input
    EXPECT_NE(SUCCESS, RunGraphAsync(session2, graph_id, ge_inputs, ge_outputs));
  
    // incorrect outputs
    Tensor te;
    std::vector<ge::Tensor> invalid_ge_outputs = {te, te, te};
    EXPECT_NE(SUCCESS, RunGraphAsync(session2, 6, ge_inputs, invalid_ge_outputs));
  
    // 空tensor输入
    std::vector<ge::Tensor> empty_inputs;
    std::vector<ge::Tensor> empty_outputs;
    EXPECT_NE(SUCCESS, RunGraphAsync(session2, graph_id, empty_inputs, empty_outputs));
  
    // 构造与图model io size不同的tensor
    ge_inputs.resize(5);
    ge_outputs.resize(5);
    // EXPECT_NE(SUCCESS, RunGraphAsync(session2, graph_id, ge_inputs, ge_outputs));
    ge_inputs.resize(7);
    ge_outputs.resize(6);
    Session session3(options);
    const auto compute_graph1 = MakeShared<ComputeGraph>("test_graph");
    
    // empty graph
    session3.AddGraph(3, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph1));
    EXPECT_NE(SUCCESS, RunGraphAsync(session3, 4, ge_inputs, ge_outputs));
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
    //session2.RemoveGraph(graph_id);
}

TEST_F(SliceSchedulerTest, TestSliceScheduler_ForBreathGraph) {
    MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
    auto recover_ir_graph = GraphFactory::BuildDynamicAndBoarderInputGraph();
    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";  // train
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; 
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    std::map<AscendString, AscendString> graph_options;
    gert::GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub.GetSlogStub().Clear();

    Session session(options);
    GraphId graph_id = 12;
    EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);
    EXPECT_EQ(GetDump(recover_ir_graph, "DynamicAndBoarderInputGraph"), SUCCESS);

    std::vector<Tensor> inputs;
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    inputs.emplace_back(CreateTensor({16}));
    std::vector<Tensor> outputs;

    EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

    EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
    EXPECT_EQ(LogCheck(runtime_stub, true, true, true), SUCCESS);

    // session.RemoveGraph(graph_id);
    runtime_stub.GetSlogStub().Clear();
}*/

/*
 * 构图不对，reshape infershape失败
TEST_F(SliceSchedulerTest, TestSliceScheduler_ForOutput) {
    MockForGenerateTask("AIcoreEngine", GenerateTaskForTaskWithHandle);
    MockForGenerateTask("aicpu_ascend_kernel", GenerateTaskForAicpuDependRange);
    auto recover_ir_graph = GraphFactory::BuildOutPutGraph();
    std::map<AscendString, AscendString> options;
    options[OPTION_GRAPH_RUN_MODE] = "1";  // train
    options[OPTION_EXEC_ENABLE_DUMP_DEBUG] = "1";
    options[OPTION_EXEC_DUMP_PATH] = "./";
    options[OPTION_EXEC_DUMP_DEBUG_MODE] = "aicore_overflow"; 
    options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
    options[JIT_COMPILE.c_str()] = "1";
    std::map<AscendString, AscendString> graph_options;
    gert::GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub.GetSlogStub().Clear();


    Session session(options);
    GraphId graph_id = 13;
    EXPECT_EQ(session.AddGraph(graph_id, recover_ir_graph, graph_options), SUCCESS);
    EXPECT_EQ(GetDump(recover_ir_graph, "BuildOutPutGraph"), SUCCESS);

    std::vector<Tensor> inputs;
    inputs.emplace_back(CreateTensor({3, 4}));
    std::vector<Tensor> outputs;

    EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
    EXPECT_EQ(LogCheck(runtime_stub, true, true, true), SUCCESS);

    EXPECT_EQ(RunGraphAsync(session, graph_id, inputs, outputs), SUCCESS);
    for(auto &output : outputs) {
        EXPECT_EQ(PrintShape(output.GetTensorDesc().GetShape()), "(3,4)");
    }
    
    // session.RemoveGraph(graph_id);
    runtime_stub.GetSlogStub().Clear();
}
*/


} // namespace ge
