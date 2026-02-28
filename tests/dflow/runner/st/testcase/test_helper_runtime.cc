/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <unistd.h>
#include <cerrno>
#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "generator/ge_generator.h"
#include "ge_running_env/tensor_utils.h"

#include "common/share_graph.h"
#include "common/profiling/profiling_properties.h"
#include "framework/common/debug/ge_log.h"
#include "ge/ge_api_error_codes.h"
#include "runtime/rt_mem_queue.h"

#include "macro_utils/dt_public_scope.h"
#include "executor/cpu_sched_event_dispatcher.h"
#include "common/config/json_parser.h"
#include "common/config/config_parser.h"
#include "common/file_constant_utils.h"
#include "hybrid/node_executor/node_executor.h"
#include "common/mem_grp/memory_group_manager.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/subprocess/subprocess_manager.h"
#include "deploy/flowrm/flowgw_client.h"
#include "deploy/flowrm/network_manager.h"
#include "deploy/model_send/flow_model_sender.h"
#include "deploy/model_recv/flow_model_receiver.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/deployer/deployer_proxy.h"
#include "deploy/deployer/heterogeneous_model_deployer.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/utils/tensor_utils.h"
#include "deploy/deployer/master_model_deployer.h"
#include "deploy/abnormal_status_handler/abnormal_status_handler.h"
#include "daemon/deployer_daemon_client.h"
#include "deploy/execfwk/builtin_executor_client.h"
#include "deploy/execfwk/udf_proxy_client.h"
#include "deploy/deployer/deploy_context.h"
#include "deploy/execfwk/executor_manager.h"
#include "deploy/deployer/deployer_var_manager.h"
#include "executor/event_handler.h"
#include "executor/engine_daemon.h"
#include "hybrid/hybrid_davinci_model.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "dflow/executor/inner_process_msg_forwarding.h"
#include "dflow/executor/heterogeneous_model_executor.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/config/configurations.h"
#include "common/dump/dump_manager.h"
#include "daemon/daemon_service.h"
#include "macro_utils/dt_public_unscope.h"

#include "hccl/hccl_types.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/ge_local_context.h"
#include "proto/deployer.pb.h"
#include "dflow/base/model/model_relation.h"
#include "common/utils/heterogeneous_profiler.h"
#include "common/subprocess/subprocess_manager.h"
#include "framework/common/runtime_tensor_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/helper_runtime/src/caas_dataflow_auth_stub.h"
#include "ge/ge_api.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ge_context.h"
#include "deploy/heterogeneous_execution_runtime.h"
#include "framework/common/ge_types.h"
#include "deploy/rpc/deployer_server.h"
#include "depends/runtime/src/runtime_stub.h"
#include "init_ge.h"

#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "flow_graph/data_flow.h"
#include "proto/dflow.pb.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "deploy/flowrm/tsd_client.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"
#include "common/env_path.h"
#include "common/util/sanitizer_options.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "dflow/compiler/pne/cpu/cpu_process_node_engine.h"
#include "dflow/base/model/flow_model_om_loader.h"

using namespace std;
using namespace ::testing;
using namespace ge;
using namespace gert;

namespace {
  int32_t AicpuLoadModelWithQStub(void *ptr) {
    (void) ptr;
    return 0;
  }

  int32_t AICPUModelDestroyStub(uint32_t modelId) {
    (void) modelId;
    return 0;
  }

  int32_t InitCpuSchedulerStub(const CpuSchedInitParam *const initParam) {
    (void) initParam;
    return 0;
  }

  int32_t AicpuLoadModelStub(void *ptr) {
    (void) ptr;
    return 0;
  }

  int32_t AICPUModelStopStub(const ReDeployConfig *const reDeployConfig) {
    (void) reDeployConfig;
    return 0;
  }

  int32_t AICPUModelClearInputAndRestartStub(const ReDeployConfig *const reDeployConfig) {
    (void) reDeployConfig;
    return 0;
  }

  int32_t AICPUModelCheckKernelSupportedStub(const CheckKernelSupportedConfig * const cfgPtr) {
    *(reinterpret_cast<int32_t *>(reinterpret_cast<uintptr_t>(cfgPtr->checkResultAddr))) = 0;
    return 0;
  }

  int32_t AICPUModelProcessDataExceptionStub(const DataFlowExceptionNotify *const notify) {
    (void)notify;
    return 0;
  }

  int32_t DestroyHccl() {
    return 0;
  }

  uint32_t TsdGetProcStatusExited(const uint32_t device_id, ProcStatusParam *status, uint32_t num) {
    for (uint32_t i = 0; i < num; ++i) {
      status[i].curStat = SUB_PROCESS_STATUS_EXITED;
    }
    return 0U;
  }
  uint32_t TsdGetProcStatusFailed(const uint32_t device_id, ProcStatusParam *status, uint32_t num) {
    return 100U;
  }
}

vector<int8_t> placeholder(224U * 224U * sizeof(int64_t) * 10);
bool enqueue_dequeue_error_flag = false;
bool is_auto_malloc_test = false;
rtError_t rtMemQueueEnQueueBuff(int32_t devId, uint32_t qid, rtMemQueueBuff_t *inBuf, int32_t timeout) {
  if (!enqueue_dequeue_error_flag) {
    return 0;
  }
  return 207014;
}
namespace ge {
void *mock_handle = nullptr;
void *mock_method = nullptr;
namespace {
constexpr int32_t kLoadSyncEventModel = 1;

static std::vector<uint8_t> g_buffer(1024 * 1024);

class MockRuntime : public RuntimeStub {
 public:
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) {
    *mbuf = placeholder.data();
    return 0;
  }

  rtError_t rtMbufFree(rtMbufPtr_t mbuf) {
    return 0;
  }

  rtError_t rtMemQueueDeQueueBuff(int32_t device, uint32_t qid, rtMemQueueBuff_t *outBuf, int32_t timeout) {
    RuntimeTensorDesc mbuf_tensor_desc;
    mbuf_tensor_desc.shape[0] = 4;
    mbuf_tensor_desc.shape[1] = 1;
    mbuf_tensor_desc.shape[2] = 1;
    mbuf_tensor_desc.shape[3] = 224;
    mbuf_tensor_desc.shape[4] = 224;
    mbuf_tensor_desc.dtype = static_cast<int64_t>(DT_INT64);
    mbuf_tensor_desc.data_addr = static_cast<int64_t>(reinterpret_cast<intptr_t>(outBuf->buffInfo->addr));
    if (memcpy_s(outBuf->buffInfo->addr, sizeof(RuntimeTensorDesc), &mbuf_tensor_desc, sizeof(RuntimeTensorDesc)) !=
        EOK) {
      printf("Failed to copy mbuf data, dst size:%zu, src size:%zu\n", outBuf->buffInfo->len, sizeof(RuntimeTensorDesc));
      return -1;
    }
    return 0;
  }

  rtError_t rtMemQueuePeek(int32_t device, uint32_t qid, size_t *bufLen, int32_t timeout) {
    *bufLen = sizeof(RuntimeTensorDesc) + 224U * 224U;
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) {
    *databuf = mbuf;
    return 0;
  }

  rtError_t rtMemQueueEnQueue(int32_t devId, uint32_t qid, void *mbuf) {
    if (!enqueue_dequeue_error_flag) {
      return 0;
    }
    return 207014;
  }

  rtError_t rtMemQueueDeQueue(int32_t devId, uint32_t qid, void **mbuf) {
    if (!enqueue_dequeue_error_flag) {
      *mbuf = placeholder.data();
      return 0;
    }
    return 207013;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) {
    if (!is_auto_malloc_test) {
      *size = placeholder.size();
    } else {
      *size = 4;
    }

    return 0;
  }

  rtError_t rtMbufGetPrivInfo(rtMbufPtr_t mbuf, void **priv, uint64_t *size) {
    static char priv_fake[1024] = {};
    *priv = priv_fake;
    *size = 512;
    return 0;
  }

  rtError_t rtEschedWaitEvent(int32_t device_id,
                              uint32_t group_id,
                              uint32_t thread_id,
                              int32_t timeout,
                              rtEschedEventSummary_t *event) override {
    event->subeventId = 0;
    return RT_ERROR_NONE;
  }

  rtError_t rtBuffAlloc(uint64_t size, void **buff) override {
    *buff = &g_buffer[0];
    return RT_ERROR_NONE;
  }
};

class ModelHandleMock2 : public ExecutorContext::ModelHandle {
 public:
  explicit ModelHandleMock2() : ModelHandle() {}

  MOCK_METHOD1(ClearModel, Status(const int32_t));
  MOCK_METHOD2(ExceptionNotify, Status(uint32_t, uint64_t));
  MOCK_METHOD2(GetModelRuntimeIdOrHandle, Status(std::vector<uint32_t> &,
    std::vector<ExecutorContext::ModelHandle *> &));
};

class RuntimeMock : public RuntimeStub {
 public:
  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
    return 10;
  }
};
}

class GrpcServer {
  class MockDeployerMessageClient : public DeployerMessageClient {
   public:
    MockDeployerMessageClient(int32_t device_id) : DeployerMessageClient(device_id, true) {}

    Status WaitForProcessInitialized() override {
      return SUCCESS;
    }
  };

  class MockDeployerDaemonClient : public DeployerDaemonClient {
   public:
    explicit MockDeployerDaemonClient(int64_t client_id) : DeployerDaemonClient(client_id) {
      std::string ctx_name = std::string("client_") + std::to_string(client_id);
      context_.SetName(ctx_name);
      context_.SetDeployerPid(client_id);
      context_.Initialize();
    }

    std::shared_ptr<DeployerMessageClient> CreateMessageClient() override {
      return MakeShared<MockDeployerMessageClient>(0);
    }

    Status ProcessDeployRequest(const deployer::DeployerRequest &request, deployer::DeployerResponse &response) override {
      GE_CHK_STATUS_RET_NOLOG(DeployerServiceImpl::GetInstance().Process(context_, request, response));
      return SUCCESS;
    }

    Status ProcessHeartbeatRequest(const deployer::DeployerRequest &request,
                                   deployer::DeployerResponse &response) override {
      GE_CHK_STATUS_RET_NOLOG(DeployerServiceImpl::GetInstance().Process(context_, request, response));
      return SUCCESS;
    }

   private:
    DeployContext context_;
  };

  class MockDaemonClientManager : public DaemonClientManager {
   public:
    std::unique_ptr<DeployerDaemonClient> CreateClient(int64_t client_id) override {
      return MakeUnique<MockDeployerDaemonClient>(client_id);
    }
  };

  class MockDaemonService : public DaemonService {
   public:
    Status InitClientManager() override {
      client_manager_ = MakeUnique<MockDaemonClientManager>();
      GE_CHECK_NOTNULL(client_manager_);
      GE_CHK_STATUS_RET(client_manager_->Initialize(), "Failed to initialize ClientManager");
      return SUCCESS;
    }
  };

  class MockDeployerDaemonService : public DeployerDaemonService {
   public:
    Status Initialize() override {
      daemon_service_ = MakeUnique<MockDaemonService>();
      GE_CHECK_NOTNULL(daemon_service_);
      return daemon_service_->Initialize();
    }
  };

 public:
  Status Run() {
    grpc_server_.SetServiceProvider(std::unique_ptr<MockDeployerDaemonService>(new MockDeployerDaemonService()));
    return grpc_server_.Run();
  }

  void Finalize() {
    grpc_server_.Finalize();
  }

 private:
  ge::DeployerServer grpc_server_;
};

class MockDynamicModelExecutor : public DynamicModelExecutor {
 public:
  MockDynamicModelExecutor(bool is_host, int32_t load_mode) : DynamicModelExecutor(is_host), load_mode_(load_mode) {}
  void SetIsHost(const bool is_host) { is_host_ = is_host; }
 protected:
  Status DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph) override {
    if (load_mode_ == kLoadSyncEventModel) {
      return SUCCESS;
    }
    (void) DynamicModelExecutor::DoLoadModel(model_data, root_graph);
    model_id_ = 2;
    aicpu_model_id_ = 1021;
    auto hybrid_model = std::make_shared<hybrid::HybridDavinciModel>();
    ModelManager::GetInstance().InsertModel(model_id_, hybrid_model);
    return SUCCESS;
  }

  Status DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs) override {
    (void) DynamicModelExecutor::DoExecuteModel(inputs, outputs);
    output_tensor_descs_.resize(1);
    std::vector<int64_t> dims{1, 8};
    output_tensor_descs_[0].SetShape(GeShape(dims));
    output_tensor_descs_[0].SetOriginShape(GeShape(dims));
    return SUCCESS;
  }
 private:
  int32_t load_mode_{};
};

class MockProxyDynamicModelExecutor : public ProxyDynamicModelExecutor {
 public:
  explicit MockProxyDynamicModelExecutor() : ProxyDynamicModelExecutor() {};
  Status DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs) override {
    (void) DynamicModelExecutor::DoExecuteModel(inputs, outputs);
    DataBuffer &data_buffer = outputs[0];
    if (data_buffer.data == nullptr) {
      data_buffer.data = output_buffer_;
    }
    data_buffer.length = 8;
    auto output_values = reinterpret_cast<int32_t *>(data_buffer.data);
    output_values[0] = 222;
    output_values[1] = 666;
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape({2}));
    output_tensor_descs_ = {tensor_desc};
    return SUCCESS;
  }

  void UnloadModel() {  
    (void) DynamicModelExecutor::UnloadModel();
    ModelManager::GetInstance().DeleteModel(model_id_);
  }
 private:
  void Dispatcher() override {
    return;
  }
 private:
  uint8_t output_buffer_[8];
};

class MockModelHandle : public ExecutorContext::ModelHandle {
 public:
  MockModelHandle(int32_t load_mode) : ModelHandle(), load_mode_(load_mode) {}
 protected:
  unique_ptr<DynamicModelExecutor> CreateDynamicModelExecutor(bool is_host) override {
    (void) ModelHandle::CreateDynamicModelExecutor(is_host);
    return MakeUnique<MockDynamicModelExecutor>(is_host, load_mode_);
  }

  unique_ptr<ProxyDynamicModelExecutor> CreateProxyDynamicModelExecutor() override {
    return MakeUnique<MockProxyDynamicModelExecutor>();
  }
 private:
  int32_t load_mode_{};
};

class MockExecutorContext : public ExecutorContext {
 public:
  MockExecutorContext() : ExecutorContext() {}
  MockExecutorContext(int32_t load_mode) : ExecutorContext(), load_mode_(load_mode) {}
 protected:
  ExecutorContext::ModelHandle *GetOrCreateModelHandle(uint32_t root_model_id, uint32_t model_id) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto &submodels = model_handles_[root_model_id];
    const auto &it = submodels.find(model_id);
    if (it != submodels.cend()) {
      return it->second.get();
    }
    mock_handle = MakeUnique<MockModelHandle>(load_mode_);
    if (!model_handles_[root_model_id].emplace(model_id, std::move(mock_handle)).second) {
      return nullptr;
    }
    return model_handles_[root_model_id][model_id].get();
  }

  unique_ptr<ModelHandle> mock_handle;
  int32_t load_mode_{};
};

class MockExecutorMessageClient : public ExecutorMessageClient {
 public:
  MockExecutorMessageClient() : ExecutorMessageClient(0) {
    rsp_msg_queue_id_ = 1;
    get_stat_func_ = [this]() -> Status { return ge::SUCCESS; };
    auto base_dir = GetHostDirByEnv() + "/runtime/deploy_res/";
    event_handler_.SetBaseDir(base_dir);
    (void)event_handler_.Initialize();
    event_handler_.context_ = MakeUnique<MockExecutorContext>();
    event_handler_.context_->SetBaseDir(base_dir);
  }
  Status SendRequest(const deployer::ExecutorRequest &request, deployer::ExecutorResponse &resp, int64_t timeout) override {
    event_handler_.HandleEvent(const_cast<deployer::ExecutorRequest &>(request), resp);
    WaitResponse(resp, -1);
    return SUCCESS;
  }

  Status WaitResponse(deployer::ExecutorResponse &response, int64_t timeout) override {
    ExecutorMessageClient::WaitResponse(response, 1);
    response.set_error_code(SUCCESS);
    return SUCCESS;
  }

  std::string GetHostDirByEnv() {
    char_t res_path[MMPA_MAX_PATH]{};
    (void)mmGetEnv("HOME", res_path, sizeof(res_path));
    return RealPath(res_path);
  }

 private:
  mutable EventHandler event_handler_;
};

class MockPneExecutorClient : public BuiltinExecutorClient {
 public:
  explicit MockPneExecutorClient(int32_t device_id) : BuiltinExecutorClient(device_id) {}

 protected:
  Status ForAndInit(int32_t device_id, std::unique_ptr<ExecutorMessageClient> &executor_process) override {
    executor_process = MakeUnique<MockExecutorMessageClient>();
    return SUCCESS;
  }
};

class MockHostCpuExecutorClient : public BuiltinExecutorClient {
 public:
  explicit MockHostCpuExecutorClient(int32_t device_id) : BuiltinExecutorClient(device_id) {}

 protected:
  Status ForAndInit(int32_t device_id, std::unique_ptr<ExecutorMessageClient> &executor_process) override {
    executor_process = MakeUnique<MockExecutorMessageClient>();
    std::map<std::string, std::string> options {
        {"ge.exec.placement", "HOST"},
    };
    ge::GetThreadLocalContext().SetGlobalOption(options);
    return SUCCESS;
  }
};

class MockUdfExecutorClient : public UdfExecutorClient {
 public:
  explicit MockUdfExecutorClient(int32_t device_id) : UdfExecutorClient(device_id) {}

 protected:
  Status LoadProcess(
      const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
      const std::string &msg_file_path,
      const std::string &group_name) {
    return SUCCESS;
  }
};

class MockUdfProxyClient : public UdfProxyClient {
 public:
  explicit MockUdfProxyClient(int32_t device_id) : UdfProxyClient(device_id) {}

  Status LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) override {
    static pid_t pid = 100000;
    std::unique_lock<std::mutex> guard(mutex_);
    model_id_to_pids_[load_model_desc.root_model_id()].emplace_back(pid++);
    // for cover
    const auto message_client = MakeShared<ExecutorMessageClient>(0);
    uint32_t req_id = UINT32_MAX;
    uint32_t rsp_id = UINT32_MAX;
    message_client->CreateMessageQueue("name_suffix_stub", req_id, rsp_id, true);
    return SUCCESS;
  }
};

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    return mock_handle;
  }
  void *DlSym(void *handle, const char *func_name) override {
    return mock_method;
  }

  int32_t DlClose(void *handle) override {
    return 0;
  }

  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    return 0;
  }
};

class MockMmpa2 : public MockMmpa {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (std::string(file_name) == "libaicpu_scheduler.so") {
      return (void *) 0x12345678;
    }
    return MockMmpa::DlOpen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "AicpuLoadModelWithQ") {
      return (void *) &AicpuLoadModelWithQStub;
    } else if (std::string(func_name) == "AICPUModelDestroy") {
      return (void *) &AICPUModelDestroyStub;
    } else if (std::string(func_name) == "InitCpuScheduler") {
      return (void *) &InitCpuSchedulerStub;
    } else if (std::string(func_name) == "AicpuLoadModel") {
      return (void *) &AicpuLoadModelStub;
    } else if (std::string(func_name) == "AICPUModelStop") {
      return (void *) &AICPUModelStopStub;
    } else if (std::string(func_name) == "AICPUModelClearInputAndRestart") {
      return (void *) &AICPUModelClearInputAndRestartStub;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *) &TsdCapabilityGet;
    } else if (std::string(func_name) == "CheckKernelSupported") {
      return (void *) &AICPUModelCheckKernelSupportedStub;
    } else if (std::string(func_name) == "AICPUModelProcessDataException") {
      return (void *)&AICPUModelProcessDataExceptionStub;
    }

    return MockMmpa::DlSym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 0;
    }
    return MockMmpa::DlClose(handle);
  }
};

class MockRuntimeForClient: public RuntimeStub {
 public:
  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    return 0;
  }

  rtError_t rtMemQueueEnQueue(int32_t dev_id, uint32_t qid, void *mem_buf) override {
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    *databuf = data_;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    *size = 0;
    return 0;
  }

  rtError_t rtMbufFree(rtMbufPtr_t mbuf) override {
    // 由MockRuntimeNoLeaks统一释放
    return RT_ERROR_NONE;
  }
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) override {
    // 此处打桩记录所有申请的Mbuf,此UT不会Dequeue和Free而造成泄漏,因此在MockRuntime析构时统一释放
    RuntimeStub::rtMbufAlloc(mbuf, size);
    std::lock_guard<std::mutex> lk(mu_);
    mem_bufs_.emplace_back(*mbuf);
    return 0;
  }

  ~MockRuntimeForClient() {
    for (auto &mbuf : mem_bufs_) {
      RuntimeStub::rtMbufFree(mbuf);
    }
    mem_bufs_.clear();
  }

 private:
  std::mutex mu_;
  uint8_t data_[1024] = {};
  std::vector<void *> mem_bufs_;
};

class MockMmpaUdfClient : public ge::MmpaStubApiGe {
 public:
  int32_t WaitPid(mmProcess pid, INT32 *status, INT32 options) override {
    std::cout << "mock wait pid stub begin\n";
    auto ret = waitpid(pid, status, options);
    if (ret != 0) {
      // always wait success
      *status = 0;
      return ret;
    }
    return 0;
  }

  void *DlSym(void *handle, const char *func_name) override {
    std::cout << "func name:" << func_name << " begin to stub\n";
    if (std::string(func_name) == "TsdGetProcListStatus") {
      return (void *) &TsdGetProcListStatus;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *) &TsdCapabilityGet;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *) 0xFFFFFFFF;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    std::cout << "dlopen stub file name = " << fileName << std::endl;
    if (std::string(fileName) == "libtsdclient.so") {
      return (void *) 0xFFFFFFFF;
    }
    return dlopen(fileName, mode);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0xFFFFFFFF) {
      return 0;
    }
    return dlclose(handle);
  }
};

class ModelDeployerMock : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model,
                     DeployResult &deploy_result) override {
    deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    deploy_result.output_queue_attrs = {{4, 0, 0}};
    deploy_result.dev_abnormal_callback = []() -> Status { return SUCCESS; };

    uint32_t input0_qid = 1;
    ExecutionRuntime::GetInstance()->GetExchangeService().CreateQueue(0, "input0", 2, 0, input0_qid);
    uint32_t input1_qid = 2;
    ExecutionRuntime::GetInstance()->GetExchangeService().CreateQueue(0, "input1", 2, 0, input1_qid);
    uint32_t input2_qid = 3;
    ExecutionRuntime::GetInstance()->GetExchangeService().CreateQueue(0, "input2", 2, 0, input2_qid);
    auto model_relation = std::make_shared<ModelRelation>();
    const auto &root_model = flow_model->GetSubmodels().begin()->second;
    GE_CHK_STATUS_RET_NOLOG(ModelRelationBuilder().BuildForSingleModel(*root_model->GetRootGraph(), *model_relation));
    flow_model->SetModelRelation(model_relation);
    return SUCCESS;
  }

  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
  }
};

class MockRemoteDeployer : public RemoteDeployer {
 public:
  explicit MockRemoteDeployer(const NodeConfig &node_config) : RemoteDeployer(node_config) {}
  MOCK_METHOD2(Process, Status(deployer::DeployerRequest & , deployer::DeployerResponse & ));
};

class ModelDeployerMock2 : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model,
                     DeployResult &deploy_result) override {
    deploy_result.input_queue_attrs = {};
    deploy_result.output_queue_attrs = {{1, 0, 0}};
    deploy_result.dev_abnormal_callback = []() -> Status { return SUCCESS; };

    NodeConfig npu_node_1;
    // npu_node_1.device_id = 0;

    auto &deployer_proxy = DeployerProxy::GetInstance();
    deployer_proxy.deployers_.emplace_back(MakeUnique<LocalDeployer>());
    deployer_proxy.deployers_.emplace_back(MakeUnique<MockRemoteDeployer>(npu_node_1));

    auto &resources = ResourceManager::GetInstance();
    DeviceInfo cpu_device(0, CPU, 0);
    DeviceInfo npu_device(1, NPU, 0);
    resources.device_info_list_.push_back(cpu_device);
    resources.device_info_list_.push_back(npu_device);
    resources.device_info_map_[0][0][CPU] = &cpu_device;
    resources.device_info_map_[1][0][NPU] = &npu_device;

    auto &remote_device =
        reinterpret_cast<MockRemoteDeployer &>(*deployer_proxy.deployers_[1]);
    EXPECT_CALL(remote_device, Process).WillRepeatedly(Return(SUCCESS));

    EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 1, 0), SUCCESS);
    auto op_desc = make_shared<OpDesc>("var_name", FILECONSTANT);
    GeShape shape({16, 16});
    GeTensorDesc tensor_desc(shape, FORMAT_ND, DT_INT16);
    op_desc->AddOutputDesc(tensor_desc);
    std::vector<int16_t> tensor(16 * 16);
    auto size = 16 * 16 * 2;
    EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("var_name", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);
    std::string buffer(reinterpret_cast<char *>(tensor.data()), size);
    std::stringstream ss(buffer);

    std::map<std::string, std::string> options;
    options["ge.exec.value_bins"] =
        R"({"value_bins":[{"value_bin_id":"vector_search_buchet_value_bin", "value_bin_file":"hello.bin"}]})";
    ge::GetThreadLocalContext().SetGraphOption(options);
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "vector_search_buchet_value_bin"));

    // uint64_t handle = 0U;
    // std::string rank_table_json_str = "test";
    // int32_t rank_id = 0;
    // DeployContext::LocalContext().CreateHcomHandle(0, rank_table_json_str, rank_id, handle);

    std::map<int32_t, std::set<int32_t>> sub_device_ids{{1, {0, 1}}};
    std::map<int32_t, std::set<uint64_t>> sessions{{1, {0}}};
    std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> node_need_transfer_memory;
    std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>> sess_map;
    std::map<OpDescPtr, std::set<int32_t>> op_desc_map;
    op_desc_map[op_desc].emplace(0);
    op_desc_map[op_desc].emplace(1);

    sess_map.insert({0, op_desc_map});
    node_need_transfer_memory.insert({1, sess_map});
    (void)system("echo > hello.bin");
    (void)system("rm -f hello.bin");
    return SUCCESS;
  }


  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
  }
 public:
  MasterModelDeployer master_deployer_;
};

class HeterogeneousModelExecutorMock : public HeterogeneousModelExecutor {
 public:
  HeterogeneousModelExecutorMock(const FlowModelPtr &flow_model, const DeployResult &deploy_result)
      : HeterogeneousModelExecutor(flow_model, deploy_result) {}

  ~HeterogeneousModelExecutorMock() = default;
};

class ExecutionRuntimeHeterogeneousMock : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return 0;
  }
  Status Finalize() override {
    return 0;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 public:
  HeterogeneousExchangeService exchange_service_;
  ModelDeployerMock model_deployer_;
};

class ExecutionRuntimeHeterogeneousMock2 : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return 0;
  }
  Status Finalize() override {
    return 0;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 public:
  HeterogeneousExchangeService exchange_service_;
  ModelDeployerMock2 model_deployer_;
};

class ExchangeServiceMock : public ExchangeService {
 public:
  Status CreateQueue(int32_t device_id,
                     const string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status DestroyQueue(int32_t device_id, uint32_t queue_id) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, const void *data, size_t size,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override {
    return 0;
  }
  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) override {
    return 0;
  }
  Status DequeueTensor(const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                       ControlInfo &control_info) override {
    return dequeue_tonsor_result;
  }
  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }
  MOCK_METHOD4(DequeueMbuf, Status(int32_t, uint32_t, rtMbufPtr_t *, int32_t));
  MOCK_METHOD5(Dequeue, Status(int32_t, uint32_t, void *, size_t, ControlInfo &));
  MOCK_METHOD5(Enqueue, Status(const int32_t, const uint32_t, const size_t,
      const ExchangeService::FillFunc &, const ExchangeService::ControlInfo &));
  uint32_t dequeue_tonsor_result = FAILED;
};

class ExecutionRuntimeHeterogeneousMock3 : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return 0;
  }
  Status Finalize() override {
    return 0;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 public:
  ExchangeServiceMock exchange_service_;
  ModelDeployerMock model_deployer_;
};

class ExecutionRuntimeHeterogeneousMock4 : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    GE_CHK_STATUS_RET_NOLOG(Configurations::GetInstance().InitInformation());
    GE_CHK_STATUS_RET_NOLOG(SubprocessManager::GetInstance().Initialize());
    GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufInit());
    (void) MemoryGroupManager::GetInstance().Initialize(Configurations::GetInstance().GetLocalNode());
    GE_CHK_STATUS_RET(HeterogeneousExchangeService::GetInstance().Initialize(0), "Failed to init model deployer");
    GE_CHK_STATUS_RET(model_deployer_.Initialize(options), "Failed to init model deployer");
    GE_CHK_STATUS_RET(NumaConfigManager::InitNumaConfig(), "Failed to init numa config");
    return 0;
  }
  Status Finalize() override {
    GE_CHK_STATUS(model_deployer_.Finalize());
    GE_CHK_STATUS(HeterogeneousExchangeService::GetInstance().Finalize());
    GE_CHK_STATUS(NetworkManager::GetInstance().Finalize());
    SubprocessManager::GetInstance().Finalize();
    Configurations::GetInstance().Finalize();
    return 0;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 public:
  ExchangeServiceMock exchange_service_;
  MasterModelDeployer model_deployer_;
};

class MockDeployerVarManager : public DeployerVarManager {};

int32_t MockHcomDestroy() {
  return 0;
}

Status MockInitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<ExecutionRuntimeHeterogeneousMock>());
  return SUCCESS;
}

Status MockInitializeHeterogeneousRuntime2(const std::map<std::string, std::string> &options) {
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<ExecutionRuntimeHeterogeneousMock2>());
  return SUCCESS;
}

class STEST_helper_runtime : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string cmd = R"(
mkdir -p ./temp_udf_st/build/_test/Ascend/release
cd ./temp_udf_st/build/_test/Ascend/release
touch func_pp0_release.om
touch func_pp0_release.so
echo "Hello" > func_pp0_release.om
echo "test1_release" > func_pp0_release.so
tar -cvf func_pp0_release.tar.gz func_pp0_release.om func_pp0_release.so
rm -rf func_pp0_release.om func_pp0_release.so
cd -
mkdir -p ./temp_udf_st/build/_test/X86/release
cd ./temp_udf_st/build/_test/X86/release
touch func_pp1_release.om
touch func_pp1_release.so
echo "Hello" > func_pp1_release.om
echo "test1_release" > func_pp1_release.so
tar -cvf func_pp1_release.tar.gz func_pp1_release.om func_pp1_release.so
rm -rf func_pp1_release.om func_pp1_release.so
cd -
cp ./temp_udf_st/build/_test/Ascend/release/func_pp0_release.tar.gz ./temp_udf_st/build/_test/X86/release/
cp ./temp_udf_st/build/_test/X86/release/func_pp1_release.tar.gz ./temp_udf_st/build/_test/Ascend/release/
)";
    (void)system(cmd.c_str());
  }

  static void TearDownTestSuite() {
    (void)system("rm -rf ./temp_udf_st");
  }

  void SetUp() {
    st_dir_path = PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/dflow/runner/st/"});
    hybrid::NodeExecutorManager::GetInstance().
        engine_mapping_.emplace("AiCoreLib", hybrid::NodeExecutorManager::ExecutorType::AICORE);
    hybrid::NodeExecutorManager::GetInstance().
        engine_mapping_.emplace("AicpuLib", hybrid::NodeExecutorManager::ExecutorType::AICPU_CUSTOM);
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

    PneExecutorClientCreatorRegistrar<MockPneExecutorClient> npu_registrar(PNE_ID_NPU);
    PneExecutorClientCreatorRegistrar<MockHostCpuExecutorClient> cpu_registrar(PNE_ID_CPU);
    // default config
    auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
    setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  }
  void TearDown() {
    MemoryGroupManager::GetInstance().Finalize();
    ExecutionRuntime::FinalizeExecutionRuntime();
    mock_handle = nullptr;
    mock_method = nullptr;
    Configurations::GetInstance().information_ = DeployerConfig{};
    ResourceManager::GetInstance().Finalize();
    DeployerProxy::GetInstance().Finalize();
    DeployContext::LocalContext().has_hcom_rank_table_.clear();
    DeployContext::LocalContext().submodel_devices_.clear();
    DeployContext::LocalContext().submodel_routes_.clear();
    DeployContext::LocalContext().var_managers_.clear();
    DeployContext::LocalContext().flow_model_receiver_.receiving_files_.clear();
    DeployContext::LocalContext().flow_model_receiver_.deploy_states_.clear();
    DeployContext::LocalContext().flow_model_receiver_.deploy_states_.clear();
    SubprocessManager::GetInstance().Finalize();
    MmpaStub::GetInstance().Reset();
    RuntimeStub::Reset();
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  static void CreateCompilerJson(const std::string &npu_compile_config_file) {
    nlohmann::json npu_compiler_json = {
        {"compiler",
         {
             {
                 {"resource_type", "Ascend"},
                 {"toolchain", "/usr/bin/g++"},
             },
         }},
    };
    std::ofstream json_file(npu_compile_config_file);
    json_file << npu_compiler_json << std::endl;
  }

  ComputeGraphPtr BuildDynamicRootGraph(const std::vector<int64_t> &shape, bool add_align_attr = false, 
                                                bool with_attr = true, bool with_file_constant = false) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    ComputeGraphPtr root_graph = nullptr;
    auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 0);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, shape);
    if (with_file_constant) {
      (void) system("echo 1 > hello.bin");
      auto neg = OP_CFG(FILECONSTANT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape)
                          .Attr(ATTR_NAME_INDEX, 0)
                          .Attr(ATTR_NAME_LOCATION, "hello.bin")
                          .Attr(ATTR_NAME_OFFSET, 0)
                          .Attr(ATTR_NAME_LENGTH, 2);
      DEF_GRAPH(graph) {
        CHAIN(NODE("Node_data_1", data)->EDGE(0, 0)->NODE("Neg", neg)->NODE("Node_output_1", netoutput));
      };
      root_graph = ToComputeGraph(graph);
    } else {
      auto neg = OP_CFG(NEG).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 1);
      DEF_GRAPH(graph) {
        CHAIN(NODE("Node_data_1", data)->EDGE(0, 0)->NODE("Neg", neg)->NODE("Node_output_1", netoutput));
      };
      root_graph = ToComputeGraph(graph);
    }
    auto output_node = root_graph->FindNode("Node_output_1");
    output_node->GetOpDesc()->SetSrcIndex({1});
    output_node->GetOpDesc()->SetSrcName({"neg"});
    if (add_align_attr) {
      auto data_node = root_graph->FindNode("Node_data_1");
      NamedAttrs align_attr;
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 10U);
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 5U);
      AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);
    }
    auto netoutput_node = root_graph->FindNode("Node_output_1");
    netoutput_node->GetOpDesc()->SetSrcIndex({0});
    netoutput_node->GetOpDesc()->SetSrcName({"neg"});
    const auto &tensor_desc = netoutput_node->GetOpDesc()->MutableInputDesc(0U);
    if (with_attr) {
      AttrUtils::SetInt(*tensor_desc, "_graph_output_max_size", 8);
    }
    root_graph->TopologicalSorting();
    return root_graph;
  }

 protected:
  std::string st_dir_path;
};
namespace {
  PneModelPtr BuildPneModel(ComputeGraphPtr root_graph) {
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
    auto ge_model = MakeShared<ge::GeModel>();
    auto model_task_def = MakeShared<domi::ModelTaskDef>();
    model_task_def->set_version("test_v100_r001");
    ge_model->SetModelTaskDef(model_task_def);
    ge_model->SetName(root_graph->GetName());
    ge_model->SetGraph(root_graph);
    ge_root_model->SetModelName(root_graph->GetName());	
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);
    bool is_unknown_shape = false;
    auto ret = ge_root_model->CheckIsUnknownShape(is_unknown_shape);
    EXPECT_EQ(ret, SUCCESS);
    ModelBufferData model_buffer_data{};
    const auto model_save_helper =
        ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
    EXPECT_NE(model_save_helper, nullptr);
    model_save_helper->SetSaveMode(false);
    ret = model_save_helper->SaveToOmRootModel(ge_root_model, "NoUse", model_buffer_data, is_unknown_shape);
    EXPECT_EQ(ret, SUCCESS);
    ModelData model_data{};
    model_data.model_data = model_buffer_data.data.get();
    model_data.model_len = model_buffer_data.length;
    PneModelPtr pne_model = FlowModelHelper::ToPneModel(model_data, root_graph);
    return pne_model;
  }

  ComputeGraphPtr BuildTwoInputDynamicRootGraph(const std::vector<int64_t> &shape, bool add_align_attr = false) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto data0 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 0);
    auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 1);
    auto neg = OP_CFG(NEG).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, shape).Attr(ATTR_NAME_INDEX, 2);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, shape);
    DEF_GRAPH(graph) {
      CHAIN(NODE("Node_data_1", data0)->EDGE(0, 0)->NODE("Neg", neg)->NODE("Node_output_1", netoutput));
      CHAIN(NODE("Node_data_2", data1)->EDGE(0, 1)->NODE("Neg", neg));
    };
    auto root_graph = ToComputeGraph(graph);
    auto output_node = root_graph->FindNode("Node_output_1");
    output_node->GetOpDesc()->SetSrcIndex({2});
    output_node->GetOpDesc()->SetSrcName({"neg"});
    if (add_align_attr) {
      auto data_node = root_graph->FindNode("Node_data_1");
      NamedAttrs align_attr;
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_INTERVAL, 10U);
      AttrUtils::SetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, 5U);
      AttrUtils::SetNamedAttrs(data_node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr);
    }
    auto netoutput_node = root_graph->FindNode("Node_output_1");
    const auto &tensor_desc = netoutput_node->GetOpDesc()->MutableInputDesc(0U);
    AttrUtils::SetInt(*tensor_desc, "_graph_output_max_size", 8);
    root_graph->TopologicalSorting();
    return root_graph;
  }
}

static void StartServer(ge::GrpcServer &grpc_server) {
  auto res = grpc_server.Run();
  if (res != ge::SUCCESS) {
    std::cout << "run failed" << std::endl;
    return;
  }
}

class MockRuntimeForSharedContent : public RuntimeStub {
 public:
  rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output) {
    return 1;
  }

  rtError_t rtEschedWaitEvent(int32_t device_id,
                              uint32_t group_id,
                              uint32_t thread_id,
                              int32_t timeout,
                              rtEschedEventSummary_t *event) override {
    event->subeventId = 0;
    return RT_ERROR_NONE;
  }

  rtError_t rtBuffAlloc(uint64_t size, void **buff) override {
    *buff = &g_buffer[0];
    return RT_ERROR_NONE;
  }

  rtError_t rtMemQueueDeQueueBuff(int32_t device, uint32_t qid, rtMemQueueBuff_t *outBuf, int32_t timeout) {
    return 0;
  }

  rtError_t rtMemQueuePeek(int32_t device, uint32_t qid, size_t *bufLen, int32_t timeout) {
    *bufLen = sizeof(RuntimeTensorDesc) + 2U;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) {
    *size = 2UL;
    return 0;
  }
};

TEST_F(STEST_helper_runtime, helper_host_dir_invalid) {
  class MockMmpaRealPath : public ge::MmpaStubApiGe {
   public:
    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      memcpy_s(realPath, realPathLen, "../../", strlen("../../"));
      return 0;
    }
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
  std::string work_dir;
  EXPECT_NE(Configurations::GetInstance().GetWorkingDir(work_dir), SUCCESS);
}

TEST_F(STEST_helper_runtime, test_var_manager_serial_deserial) {
  const map<string, string> options{};
  Status ret = VarManager::Instance(1)->Init(static_cast<uint32_t>(SessionVersion::MINI_VERSION), 1, 0, 0x5a5a);
  ret = VarManager::Instance(1)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  size_t graph_mem_max_size = VarManager::Instance(1)->graph_mem_max_size_;
  size_t var_mem_max_size = VarManager::Instance(1)->var_mem_max_size_;
  size_t var_mem_logic_base = VarManager::Instance(1)->var_mem_logic_base_;
  size_t use_max_mem_size = VarManager::Instance(1)->use_max_mem_size_;
  std::vector<int64_t> s = {1, 2, 3, 4};
  GeShape shape(s);
  GeTensorDesc tensor_desc(shape);
  TensorUtils::SetSize(tensor_desc, shape.GetShapeSize());
  std::string str = "global_step";
  ret = VarManager::Instance(1)->AssignVarMem(str, nullptr, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(ret, SUCCESS);
  TransNodeInfo trans_node_info;
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(1)->SetTransRoad(str, fusion_road);

  VarBroadCastInfo broadcast_info;
  broadcast_info.var_name = "test";
  VarManager::Instance(1)->SaveBroadCastInfo(0, broadcast_info);

  deployer::VarManagerInfo info;
  ret = VarManager::Instance(1)->VarManagerToSerial(1, info);
  EXPECT_EQ(ret, SUCCESS);
  auto session_id = info.session_id();
  EXPECT_EQ(session_id, 1);

  ret = VarManager::Instance(1)->VarManagerToDeserial(1, info);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(VarManager::Instance(1)->graph_mem_max_size_, 872415232);
  EXPECT_EQ(VarManager::Instance(1)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(1)->version_, SessionVersion::MINI_VERSION);
  EXPECT_EQ(VarManager::Instance(1)->device_id_, 0);
  EXPECT_EQ(VarManager::Instance(1)->job_id_, 0x5a5a);
  EXPECT_TRUE(VarManager::Instance(1)->graph_mem_max_size_ == graph_mem_max_size);
  EXPECT_TRUE(VarManager::Instance(1)->var_mem_max_size_ == var_mem_max_size);
  EXPECT_TRUE(VarManager::Instance(1)->var_mem_logic_base_ == var_mem_logic_base);
  EXPECT_TRUE(VarManager::Instance(1)->use_max_mem_size_ == use_max_mem_size);
  EXPECT_EQ(VarManager::Instance(1)->var_resource_->session_id_, 1);

  EXPECT_EQ(VarManager::Instance(1)->var_resource_->var_offset_map_.size(), 1);
  EXPECT_EQ(VarManager::Instance(1)->var_resource_->var_addr_mgr_map_.size(), 1);
  EXPECT_EQ(VarManager::Instance(1)->var_resource_->cur_var_tensor_desc_map_.size(), 1);

  EXPECT_EQ(VarManager::Instance(1)->var_resource_->IsVarExist(str, tensor_desc), true);
  EXPECT_EQ(VarManager::Instance(1)->mem_resource_map_.size(), 1);
  auto resource_src = VarManager::Instance(1)->mem_resource_map_[RT_MEMORY_HBM];
  auto resource = VarManager::Instance(1)->mem_resource_map_[RT_MEMORY_HBM];
  EXPECT_EQ(resource->var_mem_size_, 1536);
  EXPECT_EQ(resource->var_mem_size_, resource_src->var_mem_size_);

  ret = VarManager::Instance(1)->AssignVarMem("Hello_variable", nullptr, tensor_desc, RT_MEMORY_HBM);
  EXPECT_EQ(ret, SUCCESS);

  OpDescPtr file_const1 = std::make_shared<OpDesc>("file_const1", FILECONSTANT);
  file_const1->AddOutputDesc(tensor_desc);
  int64_t offset = 0U;
  auto length = static_cast<int64_t>(shape.GetShapeSize());
  std::string file_path1 = "tmp_weight/12345/weight.bin";
  FileConstantUtils::SetFileConstantPath(file_const1, file_path1, offset, length);
  EXPECT_EQ(VarManager::Instance(1)->AssignVarMem("file_const1", file_const1, tensor_desc, RT_MEMORY_HBM), SUCCESS);
  OpDescPtr file_const2 = std::make_shared<OpDesc>("file_const2", FILECONSTANT);
  file_const2->AddOutputDesc(tensor_desc);
  FileConstantUtils::SetFileConstantPath(file_const2, file_path1, offset, length);
  EXPECT_EQ(VarManager::Instance(1)->AssignVarMem("file_const2", file_const2, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  FlowModelSender flow_model_sender;
  EXPECT_EQ(flow_model_sender.DeployDevCfg(0, DeviceDebugConfig::ConfigType::kConfigTypeEnd), SUCCESS);
}

TEST_F(STEST_helper_runtime, GetVarMemBase) {
  DeployContext deploy_context;
  const int32_t device_id = 0;
  deployer::VarManagerInfo varinfo;
  varinfo.set_device_id(device_id);
  varinfo.set_session_id(1);
  varinfo.set_var_mem_max_size(128);
  MockDeployerVarManager var_manager;
  ASSERT_EQ(var_manager.Initialize(varinfo), SUCCESS);
  var_manager.Finalize();
}

TEST_F(STEST_helper_runtime, GetMallocVarMem) {
  DeployContext deploy_context;
  const int32_t device_id = 0;
  void *dev_addr = nullptr;
  MockDeployerVarManager var_manager;
  auto ret = var_manager.MallocVarMem(1, device_id, &dev_addr);
  EXPECT_EQ(ret, SUCCESS);
  var_manager.Finalize();
}

TEST_F(STEST_helper_runtime, ExecutorProcessFinalize) {
  auto executor_process = MakeUnique<MockExecutorMessageClient>();
  executor_process->pid_ = 1;
  Status ret = executor_process->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_helper_runtime, TestDeployModel) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &MockInitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  options["ge.exec.graphExecTimeout"] = "600000";
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<Tensor> input_tensors;
  std::vector<TensorDesc> input_desc_list;
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> data(224 * 224 * sizeof(float));
    Tensor tensor(tensor_desc, data);
    input_tensors.emplace_back(tensor);
    input_desc_list.emplace_back(tensor_desc);
  }

  std::vector<Tensor> output_tensors;
  ret = session.RunGraph(1, input_tensors, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);
}

TEST_F(STEST_helper_runtime, TestDeployModelWithFileConstant) {
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &MockInitializeHeterogeneousRuntime2;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> shape = {2, 2, 2, 2};
  auto file_const_op = OP_CFG(FILECONSTANT).Attr("shape", shape).Attr("dtype", DT_FLOAT).Attr("file_id",
                                                                                              "vector_search_bucker_value_bin");

  int64_t dims_size = 1;
  vector<int64_t> data_vec = {2, 2, 2, 2};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<float> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *) data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(float));
  std::cout << "davinci_model_execute_with_file_constant" << data_value_vec.size() << std::endl;
  auto const_op = OP_CFG(CONSTANT).Weight(data_tensor);
  auto add = OP_CFG(ADD);
  auto output = OP_CFG(NETOUTPUT);
  DEF_GRAPH(g1) {
    CHAIN(NODE("file_constant_1", file_const_op)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("const_op", const_op)->EDGE(0, 1)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE(NODE_NAME_NET_OUTPUT, output));
  };

  {
    size_t file_const_size = 64;
    float *float_buf = (float *) malloc(file_const_size);
    if (float_buf == nullptr) {
      return;
    }
    std::ofstream out1("test_copy_one_weight.bin", std::ios::binary);
    if (!out1.is_open()) {
      free(float_buf);
      return;
    }
    out1.write((char *) float_buf, file_const_size);
    out1.close();
    free(float_buf);
  }

  auto graph = ToGeGraph(g1);
  AscendString a = "ge.exec.value_bins";
  AscendString b =
      "{\"value_bins\":[{\"value_bin_id\":\"vector_search_bucker_value_bin\", "
      "\"value_bin_file\":\"./test_copy_one_weight.bin\"}]}";
  map<AscendString, AscendString> options{{a, b}};
  options["ge.exec.graphExecTimeout"] = "600000";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  std::vector<Tensor> input_tensors;
  ret = session.BuildGraph(1, input_tensors);
  ASSERT_EQ(ret, SUCCESS);
  (void) remove("test_copy_one_weight.bin");
}

TEST_F(STEST_helper_runtime, TestDeployModelNoTiling) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &MockInitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto add_2 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor0_range;
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0.SetShapeRange(tensor0_range);
    TensorUtils::SetSize(tensor0, 501760);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor1_range;
    tensor1_range.push_back(std::make_pair(1, 1));
    tensor1_range.push_back(std::make_pair(1, 10));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1.SetShapeRange(tensor1_range);
    TensorUtils::SetSize(tensor1, 501760);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
        vector<std::string> tiling_inline;
        vector<std::string> export_shape;
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        tiling_inline.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
        export_shape.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
      } else {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName({"add"});
        op_desc->SetSrcIndex({0});
      }
    }
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph, nullptr);
  SetUnknownOpKernelForNoTiling(compute_graph->GetDirectNode());

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  options["ge.exec.graphExecTimeout"] = "600000";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<Tensor> input_tensors;
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> data(224 * 224 * sizeof(float));
    Tensor tensor(tensor_desc, data);
    input_tensors.emplace_back(tensor);
  }

  std::vector<Tensor> output_tensors;
  ret = session.RunGraph(1, input_tensors, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);
}

namespace {
class CallbackManager {
 public:
  static CallbackManager &GetInstance() {
    static CallbackManager callbackManager;
    return callbackManager;
  }

  void Register(const char *moduleName, rtTaskFailCallback callback) {
    const std::string name = moduleName;
    callbacks_.emplace(name, callback);
  }

  void Call(const char *moduleName, rtExceptionInfo *excpt_info) {
    const std::string name = moduleName;
    auto iter = callbacks_.find(name);
    rtTaskFailCallback callback = iter->second;
    callback(excpt_info);
  }

  std::map<std::string, rtTaskFailCallback> callbacks_;
};

class MockRuntime2 : public MockRuntimeForClient {
 public:
  rtError_t rtRegTaskFailCallbackByModule(const char *moduleName,
                                          rtTaskFailCallback callback) override {
    CallbackManager::GetInstance().Register(moduleName, callback);
    return RT_ERROR_NONE;
  }
};

class MockRuntime3 : public RuntimeStub {
 public:
  rtError_t rtEschedWaitEvent(int32_t device_id,
                              uint32_t group_id,
                              uint32_t thread_id,
                              int32_t timeout,
                              rtEschedEventSummary_t *event) override {
    event->subeventId = 0;
    return RT_ERROR_NONE;
  }
};

class MockRuntimeUDF : public MockRuntimeForClient {
 public:
  MockRuntimeUDF() {
    response_.set_error_code(0);
    response_.set_error_message("success !!");
  }
  rtError_t rtMemQueueDeQueueBuff(int32_t device, uint32_t qid, rtMemQueueBuff_t *outBuf, int32_t timeout) {
    if (memcpy_s(outBuf->buffInfo->addr, response_.ByteSizeLong(), &response_, response_.ByteSizeLong()) !=
        EOK) {
      printf("Failed to copy mbuf data \n");
      return -1;
    }
    return 0;
  }

  rtError_t rtMemQueueDeQueue(int32_t devId, uint32_t qid, void **mbuf) {
    *mbuf = &response_;
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) {
    *databuf = mbuf;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) {
    *size = response_.ByteSizeLong();
    return 0;
  }
  rtError_t rtMbufFree(rtMbufPtr_t mbuf) {
    return 0;
  }
  deployer::ExecutorResponse response_;
};

int32_t AicpuLoadModelWithQStub(void *ptr) {
  (void) ptr;
  return 0;
}

int32_t AicpuLoadModel(void *ptr) {
  (void) ptr;
  return 0;
}

int32_t AICPUModelDestroyStub(uint32_t modelId) {
  (void) modelId;
  return 0;
}

int32_t StopCPUSchedulerStub(const uint32_t deviceId, const pid_t hostPid) {
  (void) deviceId;
  (void) hostPid;
  return 0;
}

int32_t InitCpuSchedulerStub(const CpuSchedInitParam *const initParam) {
  (void) initParam;
  return 0;
}


class MockMmpaForHeterogeneousRuntime : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) {
    std::cout << "dlopen stub file name = " << file_name << std::endl;
    if (std::string(file_name) == "libmodel_deployer.so" ||
        std::string(file_name) == "libaicpu_scheduler.so" ||
        std::string(file_name) == "libhost_aicpu_scheduler.so" ||
        std::string(file_name) == "libtsdclient.so" ||
        std::string(file_name) == "libdataflow_auth.so") {
      return (void *) 0x8888;
    } else if (std::string(file_name).find("libhcom_graph_adaptor.so") != std::string::npos ||
        std::string(file_name).find("libhccl.so") != std::string::npos) {
      return mock_handle;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    } else if (std::string(func_name) == "AicpuLoadModelWithQ") {
      return (void *) &AicpuLoadModelWithQStub;
    } else if (std::string(func_name) == "AICPUModelDestroy") {
      return (void *) &AICPUModelDestroyStub;
    } else if (std::string(func_name) == "InitCpuScheduler") {
      return (void *) &InitCpuSchedulerStub;
    } else if (std::string(func_name) == "AicpuLoadModel") {
      return (void *) &AicpuLoadModel;
    } else if (std::string(func_name) == "StopCPUScheduler") {
      return (void *) &StopCPUSchedulerStub;
    } else if (std::string(func_name) == "TsdFileLoad") {
      return (void *) &TsdFileLoad;
    } else if (std::string(func_name) == "TsdFileUnLoad") {
      return (void *) &TsdFileUnLoad;
    } else if (std::string(func_name) == "TsdGetProcListStatus") {
      return tsd_get_proc_status_func_;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *) &TsdCapabilityGet;
    } else if (std::string(func_name) == "TsdInitFlowGw") {
      return (void *) &TsdInitFlowGw;
    } else if (std::string(func_name) == "NewSignResult") {
      return (void *) &NewSignResult;
    } else if (std::string(func_name) == "DeleteSignResult") {
      return (void *) &DeleteSignResult;
    } else if (std::string(func_name) == "GetSignLength") {
      return (void *) &GetSignLength;
    } else if (std::string(func_name) == "GetSignData") {
      return (void *) &GetSignData;
    } else if (std::string(func_name) == "DataFlowAuthMasterInit") {
      return (void *) &DataFlowAuthMasterInit;
    } else if (std::string(func_name) == "DataFlowAuthSign") {
      return (void *) &DataFlowAuthSign;
    } else if (std::string(func_name) == "DataFlowAuthVerify") {
      return (void *) &DataFlowAuthVerify;
    } else if (std::string(func_name) == "CheckKernelSupported") {
      return (void *) &AICPUModelCheckKernelSupportedStub;
    } else if (std::string(func_name) == "HcomDestroy") {
      return (void *) &DestroyHccl;
    } else if (std::string(func_name) == "AICPUModelProcessDataException") {
      return (void *) &AICPUModelProcessDataExceptionStub;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 0;
    }
    return dlclose(handle);
  }

  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    char tmp_path[MMPA_MAX_PATH];
    auto ptr = realpath(path, tmp_path);
    if (ptr == nullptr) {
      (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    } else {
      (void)strncpy_s(realPath, realPathLen, ptr, strlen(ptr));
    }
    return EN_OK;
  }

  INT32 StatGet(const CHAR *path, mmStat_t *buffer) override {
    buffer->st_mode = S_IFREG;
    return EN_OK;
  }

  INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
    auto fd = open(path_name, flags, mode);
    if (fd != -1) {
      return fd;
    }
    return 0;
  }
  void *tsd_get_proc_status_func_ = (void *)&TsdGetProcListStatus;
};

Graph BuildGraph() {
  DEF_GRAPH(graph_def) {
    auto arg_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto var = OP_CFG(VARIABLE)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto neg_1 = OP_CFG(NEG)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    CHAIN(NODE("arg_0", arg_0)
              ->NODE("neg_1", neg_1)
              ->NODE("Node_Output", net_output));

    CHAIN(NODE("var", var)
              ->NODE("Node_Output", net_output));
  };

  auto graph = ToGeGraph(graph_def);
  return graph;
}

Graph BuildHostCpuDynamicGraph() {
  auto shape = std::vector<int64_t>{-1};
  auto shape1 = std::vector<int64_t>{16};
  DEF_GRAPH(graph_def) {
     auto arg_0 = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_INDEX, 0)
         .TensorDesc(FORMAT_ND, DT_INT32, shape);

     auto neg_1 = OP_CFG(SQRT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, shape);

     auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, shape1);

     CHAIN(NODE("arg_0", arg_0)
               ->NODE("neg_1", neg_1)
               ->NODE("Node_Output", net_output));
   };

  auto graph = ToGeGraph(graph_def);
  return graph;
}

Graph BuildUdfGraph(const std::string &name, const std::string &pp0_config_file, const std::string &pp1_config_file) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);
  auto node1 = dflow::FlowNode("node1", 1, 1).SetInput(0, node0);

  // function pp
  auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig(pp0_config_file.c_str());
  node0.AddPp(pp0);
  dflow::TimeBatch time_batch = {0};
  time_batch.time_window = -1;
  dflow::DataFlowInputAttr input0_attr = {dflow::DataFlowAttrType::TIME_BATCH, &time_batch};
  node0.MapInput(0, pp0, 0, {input0_attr});

  auto pp1 = dflow::FunctionPp("func_pp1").SetCompileConfig(pp1_config_file.c_str());
  node1.AddPp(pp1);
  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node1};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Graph BuildUdfGraph(const std::string &name, const std::string &udf_config_file) {
  return BuildUdfGraph(name, udf_config_file, udf_config_file);
}

/*
 * netoutput
 *     |
 *    add
 *    /  \
 * data1  data2
 */
Graph BuildSimpleGraph() {
  std::vector<int64_t> shape = {1};
  DEF_GRAPH(simple_graph) {
    auto data1 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("data1");
    auto data2 = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_INT32, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("data2");
    auto add = OP_CFG(ADD)
        .TensorDesc(FORMAT_ND, DT_INT32, shape)
        .Build("add");
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));
  };
  auto graph = ToGeGraph(simple_graph);
  return graph;
}
}  // namespace

TEST_F(STEST_helper_runtime, TestProcessSharedConstant) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForSharedContent>());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  std::map<std::string, std::string> options;
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);

  DeployContext context;
  deployer::DeployerResponse response;

  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(0);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  init_process_resource_request.set_device_id(1);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_id(1);
  std::vector<int32_t> res_ids_1 = {1};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_1.begin(), res_ids_1.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  deployer::MultiVarManagerRequest info;
  info.add_device_ids(0);
  auto var_manager_info = info.mutable_multi_var_manager_info()->add_var_manager_info();
  var_manager_info->set_session_id(1);
  var_manager_info->set_use_max_mem_size(128);
  var_manager_info->set_var_mem_logic_base(0);
  context.ProcessMultiVarManager(info);

  deployer::MultiVarManagerRequest info2;
  info2.add_device_ids(1);
  auto var_manager_info2 = info2.mutable_multi_var_manager_info()->add_var_manager_info();
  var_manager_info2->set_session_id(1);
  var_manager_info2->set_use_max_mem_size(128);
  var_manager_info2->set_var_mem_logic_base(0);
  context.ProcessMultiVarManager(info2);

  deployer::SharedContentDescRequest shared_info;
  auto shared_content_desc = shared_info.mutable_shared_content_desc();
  shared_content_desc->set_session_id(1);
  shared_content_desc->set_node_name("var_node");
  shared_content_desc->set_total_length(2);
  shared_content_desc->set_current_offset(0);
  shared_info.add_device_ids(0);
  shared_info.add_device_ids(1);
  auto remote_plan = shared_info.mutable_flow_route();
  {
    auto send_endpoint = remote_plan->add_endpoints();
    send_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    send_endpoint->set_name("transfer_file_send");
    send_endpoint->set_device_id(0);
    auto queue_desc_dev0 = send_endpoint->mutable_queue_desc();
    queue_desc_dev0->set_name("transfer_file_receive");
    queue_desc_dev0->set_depth(16);
    queue_desc_dev0->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto queue_endpoint = remote_plan->add_endpoints();
    queue_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint->set_name("transfer_file_receive");
    queue_endpoint->set_device_id(0);
    auto queue_desc = queue_endpoint->mutable_queue_desc();
    queue_desc->set_name("transfer_file_receive");
    queue_desc->set_depth(16);
    queue_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto queue_endpoint_send = remote_plan->add_endpoints();
    queue_endpoint_send->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint_send->set_name("transfer_file_send");
    queue_endpoint_send->set_device_id(0);
    auto queue_desc_send = queue_endpoint_send->mutable_queue_desc();
    queue_desc_send->set_name("transfer_file_send");
    queue_desc_send->set_depth(16);
    queue_desc_send->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto queue_endpoint_send1 = remote_plan->add_endpoints();
    queue_endpoint_send1->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint_send1->set_name("transfer_file_send1");
    queue_endpoint_send1->set_device_id(1);
    auto queue_desc_send1 = queue_endpoint_send1->mutable_queue_desc();
    queue_desc_send1->set_name("transfer_file_send1");
    queue_desc_send1->set_depth(16);
    queue_desc_send1->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto binding = remote_plan->add_bindings();
    binding->set_src_index(0);
    binding->set_dst_index(1);
  }
  auto ret = context.ProcessSharedContent(shared_info, response);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(context.tansfer_routes_.size(), 1);
  context.Finalize();
  ExecutionRuntime::FinalizeExecutionRuntime();
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
  TsdClient::GetInstance().Finalize();
  RuntimeStub::GetInstance()->Reset();
}

TEST_F(STEST_helper_runtime, TestUpdateDeployPlan) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kUpdateDeployPlan);
  auto pre_deploy_req = request.mutable_update_deploy_plan_request();
  pre_deploy_req->set_root_model_id(1);

  auto submodel_desc = pre_deploy_req->add_submodel_descs();
  submodel_desc->set_engine_name(PNE_ID_NPU);

  auto submodel_desc2 = pre_deploy_req->add_submodel_descs();
  submodel_desc2->set_engine_name(PNE_ID_NPU);
  submodel_desc2->set_rank_id("0");

  DeployContext context;
  deployer::DeployerResponse response;
  DeployerServiceImpl::UpdateDeployPlanProcess(context, request, response);
  ASSERT_EQ(response.error_code(), ge::SUCCESS);
  DeployState *deploy_state = nullptr;
  context.flow_model_receiver_.GetDeployState(1, deploy_state);
  ASSERT_NE(deploy_state, nullptr);
  ASSERT_EQ(deploy_state->local_submodel_descs_.size(), 1);
  auto it = deploy_state->local_submodel_descs_.begin();
  ASSERT_EQ(it->first.rank_id, "0");
}

TEST_F(STEST_helper_runtime, UpdateProfInfo_Succ) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  request.set_type(deployer::kUpdateProfilingInfo);
  auto prof_info = request.mutable_prof_info();
  prof_info->set_model_id(2);
  prof_info->set_is_prof_start(1);
  prof_info->set_prof_data("test");

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey executor_key = {};
  executor_key.engine_name = PNE_ID_NPU;
  context.submodel_devices_[2].emplace(executor_key);
  DeployerServiceImpl::UpdateProfilingInfoProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
}

TEST_F(STEST_helper_runtime, TestDeployUdfModelOnServer) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeUDF>());
  CreateCompilerJson("./npu_udf_compile.json");

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 2. BuildGraph

  constexpr const char *udf_config_file = "./udf_config.json";
  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";

    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"compiler", "./npu_udf_compile.json"},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildUdfGraph("udf_model", udf_config_file);
  Session session(options);
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  // 4. Cleanup
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  remove(udf_config_file);
  system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
  system("rm -fr ./npu_udf_compile.json");
}

TEST_F(STEST_helper_runtime, TestDeployHeavyLoadUdfModelOnServer) {
  constexpr const char *deploy_info_path = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(deploy_info_path);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0","node1"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeUDF>());

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 2. BuildGraph

  constexpr const char *udf_config_file = "./udf_config.json";
  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";

    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"heavy_load", true},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildUdfGraph("udf_model", udf_config_file);
  Session session(options);

  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", deploy_info_path}};
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  // 4. Cleanup
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  remove(udf_config_file);
  remove(deploy_info_path);
    system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
}

TEST_F(STEST_helper_runtime, TestDeployHeavyLoadUdfModelOnServerWithHostFlowgw) {
  constexpr const char *deploy_info_path = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(deploy_info_path);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0","node1"],
            "logic_device_list":"0:0:0:0,0:0:1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server_4dev.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 2. BuildGraph

  constexpr const char *udf_config_file = "./udf_config.json";
  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";

    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"heavy_load", true},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildUdfGraph("udf_model", udf_config_file);
  Session session(options);

  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", deploy_info_path}};
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  auto host = const_cast<DeviceInfo *>(ResourceManager::GetInstance().device_info_map_[0][0][CPU]);
  host->SetSupportFlowgw(true);
  host->SetSupportHcom(false);

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  // 4. Cleanup
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  remove(udf_config_file);
  remove(deploy_info_path);
  system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
}

TEST_F(STEST_helper_runtime, TestDeployHeavyLoadUdfModelOnDiffServerWithHostFlowgw) {
  constexpr const char *deploy_info_path = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(deploy_info_path);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:1:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/device/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);

  // 0. start server
  EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
  ge::GrpcServer grpc_server;
  std::thread server_thread = std::thread([&]() {
    StartServer(grpc_server);
  });
  std::this_thread::sleep_for(std::chrono::seconds(1));

  Configurations::GetInstance().Finalize();

  real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);

  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 2. BuildGraph

  constexpr const char *udf_config_file = "./udf_config.json";
  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";

    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"heavy_load", true},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildUdfGraph("udf_model", udf_config_file);
  Session session(options);

  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", deploy_info_path}};
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  auto host = const_cast<DeviceInfo *>(ResourceManager::GetInstance().device_info_map_[0][0][CPU]);
  host->SetSupportFlowgw(true);
  host->SetSupportHcom(false);

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  // 4. Cleanup
  // 4. Cleanup
  grpc_server.Finalize();
  if (server_thread.joinable()) {
    server_thread.join();
  }
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  remove(udf_config_file);
  remove(deploy_info_path);
  system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
}

TEST_F(STEST_helper_runtime, TestDeployUdfModelsOnServerWithHostFlowgw) {
  CreateCompilerJson("./npu_udf_compile.json");
  constexpr const char *deploy_info_path = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(deploy_info_path);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0,0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server_4dev.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 2. BuildGraph

  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
  }
  constexpr const char *pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"compiler", "./npu_udf_compile.json"},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  constexpr const char *pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"heavy_load", true},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildUdfGraph("udf_model", pp0_config_file, pp1_config_file);
  Session session(options);

  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", deploy_info_path}};
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  //  TODO : mock rts
  auto host = const_cast<DeviceInfo *>(ResourceManager::GetInstance().device_info_map_[0][0][CPU]);
  host->SetSupportFlowgw(true);
  host->SetSupportHcom(false);

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  // 4. Cleanup
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  remove(pp0_config_file);
  remove(pp1_config_file);
  remove(deploy_info_path);
    system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
  system("rm -fr ./npu_udf_compile.json");
}

TEST_F(STEST_helper_runtime, TestDeployNpuUdfModelsOnServerWithHostFlowgw) {
  CreateCompilerJson("./npu_udf_compile.json");
  constexpr const char *deploy_info_path = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(deploy_info_path);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0,0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:2:0,0:0:3:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server_4dev.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 2. BuildGraph

  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
  }
  constexpr const char *pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"compiler", "./npu_udf_compile.json"},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  constexpr const char *pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"compiler", "./npu_udf_compile.json"},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildUdfGraph("udf_model", pp0_config_file, pp1_config_file);
  Session session(options);

  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", deploy_info_path}};
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  //  TODO : mock rts
  auto host = const_cast<DeviceInfo *>(ResourceManager::GetInstance().device_info_map_[0][0][CPU]);
  host->SetSupportFlowgw(true);
  host->SetSupportHcom(false);

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  // 4. Cleanup
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  remove(pp0_config_file);
  remove(pp1_config_file);
  remove(deploy_info_path);
    system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
  system("rm -fr ./npu_udf_compile.json");
}

TEST_F(STEST_helper_runtime, TestDeployUdfModelWriteTarSuccessByMultiTimes) {
  class MockMmpaWriteByMultiTimes : public MockMmpaForHeterogeneousRuntime {
   public:
    mmSsize_t Write(INT32 fd, VOID *mm_buf, UINT32 mm_count) override {
      if (is_first_time_) {
        is_first_time_ = false;
        return write(fd, mm_buf, 1);
      } else {
        return write(fd, mm_buf, mm_count);
      }
    }
    INT32 WaitPid(mmProcess pid, INT32 *status, INT32 options) override {
      return waitpid(pid, status, options);
    }

   private:
    bool is_first_time_ = true;
  };

  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaWriteByMultiTimes>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  // 2. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();
  Session session0(options);
  // 3. BuildAndExecuteGraph

  constexpr const char *udf_config_file = "./udf_config.json";
  {
    std::string cmd = "mkdir -p ./temp_udf_st; cd temp_udf_st; echo aaaa > libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_udf_st/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";

    nlohmann::json udf_cfg_json = {{"workspace", "./temp_udf_st"},
                                   {"target_bin", "libtest.so"},
                                   {"input_num", 1},
                                   {"output_num", 1},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(udf_config_file);
    json_file << udf_cfg_json << std::endl;
  }

  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  PneExecutorClientCreatorRegistrar<MockUdfProxyClient> udf_proxy_registrar(PNE_ID_UDF, true);
  PneExecutorClientCreatorRegistrar<MockUdfExecutorClient> udf_registrar(PNE_ID_UDF);
  auto graph = BuildUdfGraph("udf_model", udf_config_file);
  Session session(options);
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  ExecutionRuntime::FinalizeExecutionRuntime();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();

  remove(udf_config_file);
    system("rm -fr `ls ./temp_udf_st/* | grep -v build`");
}

TEST_F(STEST_helper_runtime, TestHeterogeneousInitInvalid) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  GEFinalize();
  std::map<AscendString, AscendString> options;

  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host_invalid";
  // has no local node
  std::string config_file = real_path + std::string("/numa_config_without_local_node.json");
  setenv("RESOURCE_CONFIG_PATH", config_file.c_str(), 1);
  EXPECT_EQ(ge::GEInitialize(options), ACL_ERROR_GE_PARAM_INVALID);
  unsetenv("RESOURCE_CONFIG_PATH");
  GEFinalize();
  // invalid cluster
  config_file = real_path + std::string("/numa_config_invalid_cluster.json");
  setenv("RESOURCE_CONFIG_PATH", config_file.c_str(), 1);
  EXPECT_EQ(ge::GEInitialize(options), ACL_ERROR_GE_PARAM_INVALID);
  unsetenv("RESOURCE_CONFIG_PATH");
  GEFinalize();
  // invalid resource type
  config_file = real_path + std::string("/numa_config_invalid_resource.json");
  setenv("RESOURCE_CONFIG_PATH", config_file.c_str(), 1);
  EXPECT_EQ(ge::GEInitialize(options), ACL_ERROR_GE_PARAM_INVALID);
  unsetenv("RESOURCE_CONFIG_PATH");

  ge::NumaConfig numa_config;
  ASSERT_EQ(ge::ConfigParser::InitNumaConfig(config_file, numa_config), ACL_ERROR_GE_PARAM_INVALID);

  // invalid node_def
  config_file = real_path + std::string("/numa_config_invalid_node_def.json");
  ASSERT_EQ(ge::ConfigParser::InitNumaConfig(config_file, numa_config), ACL_ERROR_GE_PARAM_INVALID);

  // invalid item_def
  config_file = real_path + std::string("/numa_config_invalid_item_def.json");
  ASSERT_EQ(ge::ConfigParser::InitNumaConfig(config_file, numa_config), ACL_ERROR_GE_PARAM_INVALID);

  // invalid num
  config_file = real_path + std::string("/numa_config_invalid_num.json");
  ASSERT_EQ(ge::ConfigParser::InitNumaConfig(config_file, numa_config), ACL_ERROR_GE_PARAM_INVALID);
  GEFinalize();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, TestDeployHeterogeneousModelFusionInput) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();

  // 3. BuildAndExecuteGraph
  auto graph = BuildSimpleGraph();
  Session session(options);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_ENABLE_FUSION] = "true";
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({1}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor1(tensor_desc);
  uint8_t buffer1[1 * 4];
  tensor1.SetData(buffer1, sizeof(buffer1));
  Tensor tensor2(tensor_desc);
  uint8_t buffer2[1 * 4];
  tensor2.SetData(buffer2, sizeof(buffer2));

  std::vector<Tensor> inputs{tensor1, tensor2};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

  std::vector<Tensor> output_tensors;
  auto ret = session.RunGraph(graph_id, inputs, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);

  mock_handle = nullptr;
  GEFinalize();

  ExecutionRuntime::FinalizeExecutionRuntime();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, TestDeployWithCompileRes) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  GEFinalize();
  // 1. start server
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/device/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);

  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  flow_model->AddSubModel(BuildPneModel(graph));
  std::shared_ptr<ModelCompileResource> compile_res;
  flow_model->SetCompileResource(compile_res);

  auto &resources = ResourceManager::GetInstance();
  DeviceInfo cpu_device(0, CPU, 0);
  DeviceInfo npu_device0(1, NPU, 0);
  npu_device0.node_mesh_index_ = {0, 0, 0, 0};
  DeviceInfo npu_device1(1, NPU, 1);
  npu_device1.node_mesh_index_ = {0, 0, 0, 1};
  resources.device_info_list_.push_back(cpu_device);
  resources.device_info_list_.push_back(npu_device0);
  resources.device_info_list_.push_back(npu_device1);
  resources.device_info_map_[0][0][CPU] = &cpu_device;
  resources.device_info_map_[1][0][NPU] = &npu_device0;
  resources.device_info_map_[1][1][NPU] = &npu_device1;
  resources.compile_resource_.host_resource_type = "stub_host";
  resources.compile_resource_.logic_dev_id_to_res_type["0:0:0:0"] = "stub_dev";
  resources.compile_resource_.logic_dev_id_to_res_type["0:0:0:1"] = "stub_dev";

  DeployResult deploy_result;
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);
  // host resource type error
  const auto compile_res1 = MakeShared<ModelCompileResource>();
  compile_res1->host_resource_type = "ERROR";
  flow_model->SetCompileResource(compile_res1);
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);

  // host resource type is correct, can not found device
  const auto compile_res2 = MakeShared<ModelCompileResource>();
  compile_res2->host_resource_type = ResourceManager::GetInstance().compile_resource_.host_resource_type;
  compile_res2->logic_dev_id_to_res_type["0:1:0"] = "ERROR";
  flow_model->SetCompileResource(compile_res2);
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);

  // host resource type is correct, device is less than current resource ,but type is incorrect
  const auto compile_res3 = MakeShared<ModelCompileResource>();
  compile_res3->host_resource_type = ResourceManager::GetInstance().compile_resource_.host_resource_type;
  compile_res3->logic_dev_id_to_res_type["0:0:0"] = "ERROR";
  flow_model->SetCompileResource(compile_res3);
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);

  // host resource type is correct, device is less than current resource , and type is correct
  const auto compile_res4 = MakeShared<ModelCompileResource>();
  compile_res4->host_resource_type = ResourceManager::GetInstance().compile_resource_.host_resource_type;
  for (auto dev_to_type : ResourceManager::GetInstance().compile_resource_.logic_dev_id_to_res_type) {
    compile_res4->logic_dev_id_to_res_type[dev_to_type.first] = dev_to_type.second;
  }
  flow_model->SetCompileResource(compile_res4);
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);

  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(STEST_helper_runtime, TestDeployWithFlow) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  GEFinalize();
  // 1. start server
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/device/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
  setenv("NPU_COLLECT_PATH_EXE", "/var/log/npu/dump/", 0);

  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  flow_model->AddSubModel(BuildPneModel(graph));

  auto &resources = ResourceManager::GetInstance();
  DeviceInfo cpu_device(0, CPU, 0);
  DeviceInfo npu_device0(1, NPU, 0);
  DeviceInfo npu_device1(1, NPU, 1);
  resources.device_info_list_.push_back(cpu_device);
  resources.device_info_list_.push_back(npu_device0);
  resources.device_info_list_.push_back(npu_device1);
  resources.device_info_map_[0][0][CPU] = &cpu_device;
  resources.device_info_map_[1][0][NPU] = &npu_device0;
  resources.device_info_map_[1][1][NPU] = &npu_device1;

  DeployResult deploy_result;
  // can not transfer submodel
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);
  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("NPU_COLLECT_PATH_EXE");
}

TEST_F(STEST_helper_runtime, TestDeployHostCpuDynamicModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());

  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();

  // 1. Init master
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  ge::ProcessNodeEngineRegisterar cpu_engine_register __attribute__((unused)) (
      PNE_ID_CPU, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::CPUProcessNodeEngine(); });
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();

  CpuSchedEventDispatcher::GetInstance().models_.clear();
  CpuSchedEventDispatcher::GetInstance().Initialize(0, false);

  EXPECT_EQ(VarManager::Instance(0)->Init(0U, 1, 0, 0x5a5a), SUCCESS);
  // 3. BuildAndExecuteGrap
  auto graph = BuildHostCpuDynamicGraph();
  const map<AscendString, AscendString> session_options{{"ge.exec.placement", "HOST"}, {"ge.outputMaxSize", "64"}};
  Session session(session_options);

  std::map<AscendString, AscendString> graph_options;
  graph_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";
  graph_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = "[1~20]";
  DumpProperties dump_properties;
  dump_properties.enable_dump_ = "1";
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);

  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{tensor};
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);

  rtEschedEventSummary_t event_info{};
  CpuSchedEventDispatcher::GetInstance().OnInputsReady(event_info);
  AICPUSubEventInfo subevent_info{};
  subevent_info.modelId = 1023 - 0;
  event_info.msgLen = sizeof(subevent_info);
  event_info.msg = (char *)&subevent_info;
  CpuSchedEventDispatcher::GetInstance().OnInputsReady(event_info);
  subevent_info.modelId = 1023 - 1;
  CpuSchedEventDispatcher::GetInstance().OnInputsReady(event_info);

  auto executor = CpuSchedEventDispatcher::GetInstance().models_[1021];
  EXPECT_TRUE(executor != nullptr);
  rtMbufPtr_t input_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 16;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 16;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  rtMbufAlloc(&input_mbuf, sizeof(input_runtime_tensor_desc) + 8);
  void *input_buffer = nullptr;
  rtMbufGetBuffAddr(input_mbuf, &input_buffer);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  executor->input_mbuf_addresses_[0] = input_mbuf;

  subevent_info.modelId = 1023 - 2;
  CpuSchedEventDispatcher::GetInstance().OnInputsReady(event_info);

  GEFinalize();
  CpuSchedEventDispatcher::GetInstance().Finalize();
  rtMbufFree(input_mbuf);
}

void DeleteLines(const char* real_path, const std::vector<int>& lineNumbers) {
  std::ifstream inputFile(real_path);
  if (!inputFile.is_open()) {
    std::cerr << "Failed to open input file: " << real_path << std::endl;
    return;
  }
  std::vector<std::string> fileContent;
  std::string line;
  while (std::getline(inputFile, line)) {
    fileContent.push_back(line);
  }
  inputFile.close();
  std::stringstream ss;
  ss << real_path << "_bk";
  std::ofstream outputFile(ss.str());
  if (!outputFile.is_open()) {
    std::cerr << "Failed to open output file: " << ss.str() << std::endl;
    return;
  }
  for (const auto& line : fileContent) {
    outputFile << line << std::endl;
  }
  outputFile.close();

  std::vector<std::string> newContent;
  int lineNumber = 1;
  for (const auto& line : fileContent) {
    if (std::find(lineNumbers.begin(), lineNumbers.end(), lineNumber) == lineNumbers.end()) {
      newContent.push_back(line);
    }
    lineNumber++;
  }
  std::ofstream inputFile2(real_path, std::ios::trunc);
  if (!inputFile2.is_open()) {
    std::cerr << "Failed to open input file: " << real_path << std::endl;
    return;
  }
  for (const auto& line : newContent) {
    inputFile2 << line << std::endl;
  }
  inputFile2.close();
}

void CreateRedeployFile(const char* real_path) {
  std::string parent_path = real_path;
  parent_path = parent_path.substr(0, parent_path.find_last_of("/\\") + 1);
  std::ofstream redeploy_file(parent_path + "redeploy");
  if (redeploy_file.is_open()) {
    std::cout << "redeploy file " << parent_path << " created!" << std::endl;
    redeploy_file.close();
  }
  else {
    std::cerr << "Error creating redeploy file!" << std::endl;
  }
}

void RestoreFile(const char* path) {
  std::string str_path(path);
  size_t pos = str_path.find_last_of("/\\");
  if (pos == std::string::npos) {
    std::cerr << "Invalid path: " << path << std::endl;
    return;
  }
  std::string dir_path = str_path.substr(0, pos);
  std::string file_name = str_path.substr(pos + 1);
  std::string bk_file_name = file_name + "_bk";
  std::string bk_file_path = dir_path + "/" + bk_file_name;
  std::string new_file_path = dir_path + "/" + file_name;

  std::remove(new_file_path.c_str());
  std::rename(bk_file_path.c_str(), new_file_path.c_str());
}

TEST_F(STEST_helper_runtime, TestDeployHeterogeneousModelMaintenanceCfg) {
  const std::string kEnableFlag = "1";
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  // init log/dump/profiling cfg
  std::map<std::string, std::string> kLogEnvs =
      {{"ASCEND_GLOBAL_LOG_LEVEL", "1"}, {"ASCEND_GLOBAL_EVENT_ENABLE", "1"},
       {"ASCEND_HOST_LOG_FILE_NUM", "1"}};

  setenv("ASCEND_GLOBAL_LOG_LEVEL", kLogEnvs["ASCEND_GLOBAL_LOG_LEVEL"].c_str(), 1);
  setenv("ASCEND_GLOBAL_EVENT_ENABLE", kLogEnvs["ASCEND_GLOBAL_EVENT_ENABLE"].c_str(), 1);
  setenv("ASCEND_HOST_LOG_FILE_NUM", kLogEnvs["ASCEND_HOST_LOG_FILE_NUM"].c_str(), 1);
  DumpProperties dump_properties;
  dump_properties.enable_dump_ = kEnableFlag;
  DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
  ProfilingProperties::Instance().UpdateDeviceIdCommandParams("mode=all");

  // 1. start server
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/device/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
  ge::GrpcServer grpc_server;
  std::thread server_thread = std::thread([&]() {
    StartServer(grpc_server);
  });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();

  // 2. Init master
  real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_2server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();

  // 3. BuildAndExecuteGraph
  auto graph = BuildGraph();
  Session session(options);
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{tensor};
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  GEFinalize();

  // 4. Cleanup
  grpc_server.Finalize();
  if (server_thread.joinable()) {
    server_thread.join();
  }
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(STEST_helper_runtime, TestDeployServerModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime3>());

  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();

  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  GEFinalize();
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  GeRunningEnvFaker ge_env;
  ge_env.InstallDefault();

  // 3. BuildAndExecuteGraph
  auto graph = BuildGraph();
  Session session(options);
  uint32_t graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  Shape shape(std::vector<int64_t>({16}));
  TensorDesc tensor_desc(shape, FORMAT_ND, DT_INT32);
  Tensor tensor(tensor_desc);
  uint8_t buffer[16 * 4];
  tensor.SetData(buffer, sizeof(buffer));

  std::vector<Tensor> inputs{tensor};
  mock_handle = (void *) 0x12345678;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  mock_handle = nullptr;
  GEFinalize();

  ExecutionRuntime::FinalizeExecutionRuntime();
  MmpaStub::GetInstance().Reset();
  RuntimeStub::Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(STEST_helper_runtime, TestGetNodeStat) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  std::unique_ptr<Deployer> deployer = nullptr;
  deployer.reset((Deployer*)(new LocalDeployer()));
  DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer));
  EXPECT_EQ(DeployerProxy::GetInstance().GetNodeStat(), ge::SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, TestGetDeviceAbnormalCode) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime2>());
  DeviceAbnormalStatusHandler::Instance().Initialize();
  rtExceptionInfo excpt_info{};
  excpt_info.deviceid = 1;
  excpt_info.retcode = ACL_ERROR_RT_DEVICE_OOM;
  CallbackManager::GetInstance().Call("deployer_dev_abnormal", &excpt_info);

  std::unique_ptr<Deployer> deployer = nullptr;
  deployer.reset((Deployer*)(new LocalDeployer()));
  DeployerProxy::GetInstance().deployers_.emplace_back(std::move(deployer));
  DeployerProxy::GetInstance().GetNodeStat();
  EXPECT_EQ(DeployerProxy::GetInstance().GetDeviceAbnormalCode(), ACL_ERROR_RT_DEVICE_OOM);
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, ProcessHeartBeat01) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  auto bult_exe_client = new BuiltinExecutorClient(0);
  bult_exe_client->sub_proc_stat_ = ProcStatus::STOPPED;
  bult_exe_client->heartbeat_listening_ = true;
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  exe_client.reset((PneExecutorClient*)bult_exe_client);
  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 0, 0, "NPU", "", 666};
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);
  deployer::DeployerRequest req;
  deployer::DeployerResponse res;
  printf("ProcessHeartBeat01 start Process 1, key.process=%d\n", key.process_id);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, bool> submodel_instance_name;
  deployer::SubmodelDesc submodel_desc1;
  submodel_desc1.set_model_name("model_1");
  submodel_desc1.set_model_instance_name("model_666_1");
  deployer::SubmodelDesc submodel_desc2;
  submodel_desc2.set_model_name("model_2");
  submodel_desc2.set_model_instance_name("model_666_2");
  deployer::SubmodelDesc submodel_desc3;
  submodel_desc3.set_model_name("model_1");
  submodel_desc3.set_model_instance_name("model_777_1");

  context.local_rootmodel_to_submodel_descs_[1][key].push_back(submodel_desc1);
  submodel_instance_name.emplace("model_666_1", false);
  context.local_rootmodel_to_submodel_descs_[1][key].push_back(submodel_desc2);
  submodel_instance_name.emplace("model_666_2", false);
  ExecutorManager::ExecutorKey key2 = {1, 0, 1, "NPU", "", 777};
  context.local_rootmodel_to_submodel_descs_[1][key2].push_back(submodel_desc3);
  submodel_instance_name.emplace("model_777_1", false);
  context.ProcessHeartbeat(req, res);
  EXPECT_EQ(res.error_code(), FAILED);
  for (const auto &submodel_instances : context.abnormal_submodel_instances_name_) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second) {
      printf("local abnormal submodel_instance_name is %s, i=%u\n", submodel_instance.first.c_str(), i);
      auto iter = submodel_instance_name.find(submodel_instance.first);
      EXPECT_NE(iter, submodel_instance_name.end());
      i++;
    }
  }
  const auto abnormal_submodel_instance_name = res.heartbeat_response().abnormal_submodel_instance_name();
  for (const auto &submodel_instances : abnormal_submodel_instance_name) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second.submodel_instance_name()) {
      printf("remote abnormal submodel_instance_name is %s, i=%u\n", submodel_instance.first.c_str(), i);
      auto iter = submodel_instance_name.find(submodel_instance.first);
      EXPECT_NE(iter, submodel_instance_name.end());
      i++;
    }
  }
  res.set_error_code(FAILED);
  dlog_setlevel(0, 3, 0);
  printf("ProcessHeartBeat01 end Process 1\n");

  auto bult_exe_client2 = new BuiltinExecutorClient(1);
  bult_exe_client2->sub_proc_stat_ = ProcStatus::EXITED;
  bult_exe_client2->heartbeat_listening_ = true;
  std::unique_ptr<PneExecutorClient> exe_client2 = nullptr;
  exe_client2.reset((PneExecutorClient*)bult_exe_client2);
  context.executor_manager_.executor_clients_[key] = std::move(exe_client2);
  // context.dgw_client_.dgw_status_map_[0] = ProcStatus::EXITED;
  printf("ProcessHeartBeat01 start Process 2\n");
  dlog_setlevel(0, 0, 0);
  context.ProcessHeartbeat(req, res);
  dlog_setlevel(0, 3, 0);
  printf("ProcessHeartBeat01 end Process 2\n");

  res.set_error_code(FAILED);
  printf("ProcessHeartBeat01 start Process 3\n");
  dlog_setlevel(0, 0, 0);
  context.ProcessHeartbeat(req, res);
  dlog_setlevel(0, 3, 0);
  printf("ProcessHeartBeat01 end Process 3\n");
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, ProcessHeartBeat03) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  auto bult_exe_client = new BuiltinExecutorClient(0);
  bult_exe_client->sub_proc_stat_ = ProcStatus::EXITED;
  bult_exe_client->heartbeat_listening_ = true;
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  exe_client.reset((PneExecutorClient*)bult_exe_client);
  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 0, 0, "NPU", "", 666};
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);
  // context.dgw_client_.dgw_status_map_[0] = ProcStatus::STOPPED;
  // context.dgw_client_.dgw_status_map_[1] = ProcStatus::NORMAL;
  deployer::DeployerRequest req;
  deployer::DeployerResponse res;
  printf("ProcessHeartBeat03 start Process 1, key.process=%d\n", key.process_id);
  dlog_setlevel(0, 0, 0);
  std::map<std::string, bool> submodel_instance_name;
  deployer::SubmodelDesc submodel_desc1;
  submodel_desc1.set_model_name("model_1");
  submodel_desc1.set_model_instance_name("model_666_1");
  deployer::SubmodelDesc submodel_desc2;
  submodel_desc2.set_model_name("model_2");
  submodel_desc2.set_model_instance_name("model_666_2");
  deployer::SubmodelDesc submodel_desc3;
  submodel_desc3.set_model_name("model_1");
  submodel_desc3.set_model_instance_name("model_777_1");

  context.local_rootmodel_to_submodel_descs_[1][key].push_back(submodel_desc1);
  submodel_instance_name.emplace("model_666_1", false);
  context.local_rootmodel_to_submodel_descs_[1][key].push_back(submodel_desc2);
  submodel_instance_name.emplace("model_666_2", false);
  ExecutorManager::ExecutorKey key2 = {1, 0, 1, "NPU", "", 777};
  context.local_rootmodel_to_submodel_descs_[1][key2].push_back(submodel_desc3);
  submodel_instance_name.emplace("model_777_1", false);
  context.ProcessHeartbeat(req, res);
  EXPECT_EQ(res.error_code(), FAILED);
  for (const auto &submodel_instances : context.abnormal_submodel_instances_name_) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second) {
      printf("local abnormal submodel_instance_name is %s, i=%u\n", submodel_instance.first.c_str(), i);
      auto iter = submodel_instance_name.find(submodel_instance.first);
      EXPECT_NE(iter, submodel_instance_name.end());
      i++;
    }
  }
  const auto abnormal_submodel_instance_name = res.heartbeat_response().abnormal_submodel_instance_name();
  for (const auto &submodel_instances : abnormal_submodel_instance_name) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second.submodel_instance_name()) {
      printf("remote abnormal submodel_instance_name is %s, i=%u\n", submodel_instance.first.c_str(), i);
      auto iter = submodel_instance_name.find(submodel_instance.first);
      EXPECT_NE(iter, submodel_instance_name.end());
      i++;
    }
  }
  dlog_setlevel(0, 3, 0);
  printf("ProcessHeartBeat03 end Process 1\n");
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, ProcessBindHostSucc) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  auto bult_exe_client = std::unique_ptr<BuiltinExecutorClient>(new BuiltinExecutorClient(0, true));
  bult_exe_client->pid_ = 1;
  PneExecutorClient::ClientContext context;
  context.device_type = CPU;
  bult_exe_client->SetContext(context);
  deployer::ExecutorRequest_ModelQueuesAttrs model_queue_attrs;
  auto *const input_queue_def = model_queue_attrs.mutable_input_queues_attrs()->Add();
  input_queue_def->set_queue_id(0);
  input_queue_def->set_device_type(NPU);
  input_queue_def->set_device_id(1);
  ASSERT_EQ(bult_exe_client->GrantQueuesForProcess(100, NPU, model_queue_attrs), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, AddAbnormalSubmodelInstance) {
  DeployContext context;
  deployer::DeployerResponse response;
  std::map<uint32_t, std::vector<std::string>> model_instance_name;
  model_instance_name[1].push_back("model1");
  model_instance_name[1].push_back("model1");
  context.AddAbnormalSubmodelInstance(response, model_instance_name);
  const auto abnormal_submodel_instance_name = response.heartbeat_response().abnormal_submodel_instance_name();
  for (const auto &submodel_instances : abnormal_submodel_instance_name) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second.submodel_instance_name()) {
      printf("add abnormal submodel instance abnormal submodel_instance_name is %s, i=%u\n",
          submodel_instance.first.c_str(), i);
      EXPECT_EQ(submodel_instance.first, "model1");
      i++;
    }
  }
  context.Finalize();
}

TEST_F(STEST_helper_runtime, UpdateModelInsNameByPid) {
  UdfExecutorClient client(0);
  client.pid_to_model_id_[11] = 1;
  client.pid_to_model_id_[21] = 2;
  client.pid_to_model_instances_name_[11].push_back("model_1");
  client.pid_to_model_instances_name_[11].push_back("model_2");
  client.pid_to_model_instances_name_[21].push_back("model_3");
  client.UpdateModelInsNameByPid(11);
  EXPECT_EQ(client.abnormal_model_instances_name_.size(), 1);
  EXPECT_EQ(client.abnormal_model_instances_name_[1][0], "model_1");
  EXPECT_EQ(client.abnormal_model_instances_name_[1][1], "model_2");
}

TEST_F(STEST_helper_runtime, GetAbnormalModelInsName) {
  UdfExecutorClient client(0);
  client.pid_to_model_id_[11] = 1;
  client.pid_to_model_instances_name_[11].push_back("model");
  client.abnormal_model_instances_name_[1].push_back("model_1");
  client.abnormal_model_instances_name_[1].push_back("model_2");
  client.abnormal_model_instances_name_[2].push_back("model2_0");
  std::map<uint32_t, std::vector<std::string>> abnormal_model_instances_name;
  abnormal_model_instances_name[1].push_back("model_0");
  client.GetAbnormalModelInsName(abnormal_model_instances_name);
  EXPECT_EQ(abnormal_model_instances_name[1][0], "model_0");
  EXPECT_EQ(abnormal_model_instances_name[1][1], "model_1");
  EXPECT_EQ(abnormal_model_instances_name[1][2], "model_2");
  EXPECT_EQ(abnormal_model_instances_name[2][0], "model2_0");
}

TEST_F(STEST_helper_runtime, TestProcManager) {
  class MockWaitPid : public MockMmpa {
   private:
    std::function<int32_t(mmProcess, INT32 *, INT32)> m_waitPid = nullptr;
   public:
    void SetMockFunc(std::function<int32_t(mmProcess, INT32 *, INT32)> mock_func) {
      m_waitPid = mock_func;
    }

    int32_t WaitPid(mmProcess pid, INT32 *status, INT32 options) override {
      if (m_waitPid != nullptr) {
        return m_waitPid(pid, status, options);
      } else {
        MockMmpa::WaitPid(pid, status, options);
      }
      return 0;
    }

    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      (void)strncpy_s(realPath, realPathLen, path, strlen(path));
      return 0;
    }
  };
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  auto mock = std::make_shared<MockWaitPid>();
  mock->SetMockFunc([](mmProcess pid, INT32 *status, INT32 options) -> int32_t {
      *status = 0x7f;
      return 3;
  });
  MmpaStub::GetInstance().SetImpl(mock);
  ProcStatus status_result = ProcStatus::NORMAL;
  std::function<void(const ProcStatus &)> func = [&](const ProcStatus &status) {
    status_result = status;
  };
  pid_t pid = getpid();
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  EXPECT_EQ(SubprocessManager::GetInstance().Initialize(), SUCCESS);
  // 10ms, Wait for RunThread.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  SubprocessManager::GetInstance().run_flag_.store(false);
  if (SubprocessManager::GetInstance().watch_sub_proc_thread_.joinable()) {
    SubprocessManager::GetInstance().watch_sub_proc_thread_.join();
  }
  EXPECT_EQ(status_result, ProcStatus::STOPPED);
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);

  mock->SetMockFunc([](mmProcess pid, INT32 *status, INT32 options) -> int32_t {
    *status = 0xd;
    return 3;
  });
  MmpaStub::GetInstance().SetImpl(mock);
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  EXPECT_EQ(SubprocessManager::GetInstance().Initialize(), SUCCESS);
  // 10ms, Wait for RunThread.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);

  mock->SetMockFunc([](mmProcess pid, INT32 *status, INT32 options) -> int32_t {
    return -1;
  });
  MmpaStub::GetInstance().SetImpl(mock);
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  EXPECT_EQ(SubprocessManager::GetInstance().Initialize(), SUCCESS);
  usleep(10);  // 0.01ms, Wait for RunThread.
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
  MmpaStub::GetInstance().Reset();
}

TEST_F(STEST_helper_runtime, TestHeterogeneousRegCallback) {
  mock_handle = (void *) 0xffffffff;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  auto mock_runtime = std::make_shared<MockRuntime2>();
  RuntimeStub::SetInstance(mock_runtime);
  EngineDaemon engine_daemon;
  auto device_id = std::to_string(0);
  auto queue_id = std::to_string(0);
  auto event_group_id = std::to_string(1);
  const std::string process_name = "npu_executor";
  const std::string without_model_executor = std::to_string(false);
  const char_t *argv[] = {
      process_name.c_str(),
      "BufferGroupName",
      queue_id.c_str(),
      device_id.c_str(),
      event_group_id.c_str(),
      without_model_executor.c_str(),
  };
  EXPECT_EQ(engine_daemon.InitializeWithArgs(6, (char **)argv), SUCCESS);
  rtExceptionInfo exceptionInfo;
  exceptionInfo.retcode = ACL_ERROR_RT_SOCKET_CLOSE;
  CallbackManager::GetInstance().Call("NpuExe", &exceptionInfo);
  exceptionInfo.retcode = 507018U;
  CallbackManager::GetInstance().Call("NpuExe", &exceptionInfo);
  exceptionInfo.retcode = 555U;
  CallbackManager::GetInstance().Call("NpuExe", &exceptionInfo);
  engine_daemon.rt_context_ = (rtContext_t)1;
  engine_daemon.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(STEST_helper_runtime, TestEngineDaemon) {
  class MockDaemonRuntime : public MockRuntimeForClient {
   public:
    struct MbufStub {
      MbufStub() {
        buffer.resize(1, 0);
        head.resize(512, 0);
        // set finalize msg type
        ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
            static_cast<uint8_t *>(&head[0]) + 512 - sizeof(ExchangeService::MsgInfo));
        msg_info->msg_type = 2;
      }

      std::vector<uint8_t> head;
      std::vector<uint8_t> buffer;
    };

    rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
      *mbuf = &mbuf_stub_;
      return 0;
    }

   private:
    MbufStub mbuf_stub_;
  };
  mock_handle = (void *) 0xffffffff;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  auto mock_runtime = std::make_shared<MockDaemonRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  EngineDaemon engine_daemon;
  auto device_id = std::to_string(0);
  auto queue_id = std::to_string(0);
  auto event_group_id = std::to_string(1);
  const std::string process_name = "npu_executor";
  const char_t *argv[] = {
      process_name.c_str(),
      "BufferGroupName",
      queue_id.c_str(),
      device_id.c_str(),
      event_group_id.c_str(),
      "--base_dir=./",
      "--device_id=0",
      "--msg_queue_device_id=0",
  };
  EXPECT_EQ(engine_daemon.InitializeWithArgs(8, (char **)argv), SUCCESS);
  std::thread loop_thread = std::thread([&]() { engine_daemon.LoopEvents(); });
  sleep(1);
  engine_daemon.SignalHandler(9);
  if (loop_thread.joinable()) {
    loop_thread.join();
  }
  engine_daemon.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(STEST_helper_runtime, TestInitProfilingFromOption) {
  EngineDaemon engine_daemon;
  std::map<std::string, std::string> options;
  options["PROFILING_DEVICE_CONFIG_DATA"] = "test1";
  options["PROFILING_IS_EXECUTE_ON"] = "1";

  const char_t * const kEnvValue = "MS_PROF_INIT_FAIL";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
  auto ret = engine_daemon.InitProfilingFromOption(options);
  EXPECT_NE(ret, SUCCESS);
  unsetenv(kEnvValue);

  // init profiling with ge option
  EXPECT_EQ(engine_daemon.InitProfilingFromOption(options), SUCCESS);
}

TEST_F(STEST_helper_runtime, TestEnqueueAndDequeueFail) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  enqueue_dequeue_error_flag = true;
  HeterogeneousExchangeService exchange_service;
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  uint8_t buf[128];
  ExchangeService::ControlInfo enqueue_control_info = {};
  enqueue_control_info.timeout = 2000;
  exchange_service.Enqueue(0, queue_id, buf, sizeof(buf), enqueue_control_info);

  std::vector<ExchangeService::BuffInfo> buffs;
  exchange_service.AddClientQueue(queue_id);
  EXPECT_EQ(exchange_service.Enqueue(0, queue_id, buffs, enqueue_control_info), ACL_ERROR_RT_QUEUE_FULL);

  ExchangeService::ControlInfo control_info;
  control_info.timeout = 2000;
  exchange_service.Dequeue(0, queue_id, buf, sizeof(buf), control_info);
  enqueue_dequeue_error_flag = false;
  exchange_service.Finalize();
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, TestHeterogeneousProfiler) {
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  HeterogeneousExchangeService exchange_service;
  exchange_service.Initialize(0);
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  uint8_t buf[128];
  ExchangeService::ControlInfo enqueue_control_info = {};
  enqueue_control_info.timeout = 2000;
  exchange_service.Enqueue(0, queue_id, buf, sizeof(buf), enqueue_control_info);
  ExchangeService::ControlInfo control_info;
  control_info.timeout = 2000;
  exchange_service.Dequeue(0, queue_id, buf, sizeof(buf), control_info);
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestDeployWithFlowOnServerWithSharedContent) {
  VarManager::Instance(0)->Destory();
  ge::GetContext().SetSessionId(0);
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForSharedContent>());
  std::map<std::string, std::string> options;
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};

  ge::GrpcServer grpc_server;
  std::thread server_thread = std::thread([&]() {
    StartServer(grpc_server);
  });
  std::this_thread::sleep_for(std::chrono::seconds(1));

  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  auto graph = ShareGraph::AicoreGraph();
  auto op_desc_ptr = MakeShared<OpDesc>();
  op_desc_ptr->SetName("test_file_const");
  op_desc_ptr->SetType(FILECONSTANT);
  GeShape shape({1, 2});
  GeTensorDesc tensor_desc(shape, FORMAT_ND);
  TensorUtils::SetSize(tensor_desc, 2);
  op_desc_ptr->AddOutputDesc(tensor_desc);
  (void)system("echo 1 > hello.bin");
  AttrUtils::SetStr(op_desc_ptr, ATTR_NAME_LOCATION, "hello.bin");
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_LENGTH, 2);
  graph->AddNode(op_desc_ptr);
  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(true);
  EXPECT_EQ(VarManager::Instance(0)->AssignVarMem("test_file_const", nullptr, tensor_desc, RT_MEMORY_HBM), SUCCESS);

  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0:0");
  flow_model->AddSubModel(graph_model);

  DeployResult deploy_result;
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), SUCCESS);

  grpc_server.Finalize();
  if (server_thread.joinable()) {
    server_thread.join();
  }
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestDeployWithFlowOnServer) {
  class MockRuntimeForServer : public MockRuntime {
   public:
    rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output)
    {
      return 1;
    }
  };
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForServer>());
  std::map<std::string, std::string> options;
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);
  ge::GetThreadLocalContext().SetSessionOption({{"ge.flowGraphMemMaxSize","123456789"}});

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  AttrUtils::SetInt(graph, "_inputs_align_max_cache_num", 100);
  AttrUtils::SetInt(graph, "_inputs_align_timeout", 30 * 1000);
  AttrUtils::SetBool(graph, "_inputs_align_dropout", true);
  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0:0");
  flow_model->AddSubModel(graph_model);

  DeployResult deploy_result;
  MasterModelDeployer model_deployer;
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutorMock(flow_model, deploy_result));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  std::vector<ge::Tensor> input_tensors;
  input_tensors.resize(3);

  std::vector<ge::GeTensor> input_ge_tensors(3);
  std::vector<ge::GeTensor> output_ge_tensors;
  auto ret = executor->Execute(input_ge_tensors, output_ge_tensors);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(model_deployer.Finalize(), SUCCESS);
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
  ge::GetThreadLocalContext().SetSessionOption({{}});
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestEnqueueAndDequeueMbufTensorSuccess) {
  HeterogeneousExchangeService exchange_service;
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);

  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffer_size, m_buf, control_info), SUCCESS);
  ASSERT_EQ(exchange_service.DequeueMbufTensor(0, queue_id, aligned_ptr, buffer_size, control_info), SUCCESS);
  rtMbufFree(m_buf);
  ExecutionRuntime::FinalizeExecutionRuntime();
  exchange_service.Finalize();
}

TEST_F(STEST_helper_runtime, TestEnqueueAndDequeueSuccess) {
  HeterogeneousExchangeService exchange_service;
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  uint32_t client_queue_id = 1;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128 * 1024 * 1024;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufPtr_t dev_m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);
  void *data = nullptr;
  rtMbufGetBuffAddr(m_buf, &data);

  RuntimeTensorDesc runtime_tensor_desc;
  const std::vector<ExchangeService::BuffInfo> buffs{
      {.addr = &runtime_tensor_desc, .len = sizeof(runtime_tensor_desc)},
      {.addr = ValueToPtr(PtrToValue(data)), .len = buffer_size},
      {.addr = nullptr, .len = 0}};
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffs, control_info), SUCCESS);
  // 避免直接使用私有成员，此处queue_id不应复用
  exchange_service.AddClientQueue(client_queue_id);
  ASSERT_EQ(exchange_service.Enqueue(0, client_queue_id, buffs, control_info), SUCCESS);
  rtMbufFree(m_buf);
  // rtQueue中的Mbuf需要释放
  ASSERT_EQ(exchange_service.DequeueMbuf(0, queue_id, &dev_m_buf, control_info.timeout), SUCCESS);
  rtMbufFree(dev_m_buf);
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  GE_MAKE_GUARD(recover, []() { MmpaStub::GetInstance().Reset(); });
  g_runtime_stub_mock = "rtCtxGetCurrent";
  GE_MAKE_GUARD(mock, []() { g_runtime_stub_mock = ""; });
  GeTensor output_tensor;
  ASSERT_EQ(exchange_service.DequeueTensor(0, client_queue_id, output_tensor, control_info), SUCCESS);

  ExecutionRuntime::FinalizeExecutionRuntime();
  exchange_service.Finalize();
}

TEST_F(STEST_helper_runtime, TestModelIoProfiling) {
  HeterogeneousExchangeService exchange_service;
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);

  bool old_value = ProfilingProperties::Instance().ProfilingTrainingTraceOn();
  ProfilingProperties::Instance().SetTrainingTrace(true);
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffer_size, m_buf, control_info), SUCCESS);
  ASSERT_EQ(exchange_service.DequeueMbufTensor(0, queue_id, aligned_ptr, buffer_size, control_info), SUCCESS);
  rtMbufFree(m_buf);
  ExecutionRuntime::FinalizeExecutionRuntime();
  exchange_service.Finalize();
  ProfilingProperties::Instance().SetTrainingTrace(old_value);
}

void SetSubGraph(OpDesc &op_desc, const std::string &name) {
  auto subgraph = std::make_shared<ComputeGraph>(name);
  op_desc.AddSubgraphName(name);
  op_desc.SetSubgraphInstanceName(0, name);
}

TEST_F(STEST_helper_runtime, TestBuiltinExecutorClient_host) {
  SubprocessManager::GetInstance().executable_paths_[PNE_ID_NPU] = "npu_executor";
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForClient>());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaUdfClient>());

  BuiltinExecutorClient client(0);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.GetSubProcStat(), ProcStatus::NORMAL);
  EXPECT_EQ(client.Finalize(), SUCCESS);
}

TEST_F(STEST_helper_runtime, TestUDFClient_host) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaUdfClient>());
  UdfExecutorClient client(0);
  client.model_id_to_pids_[0].emplace_back(1111111111111);
  client.model_id_to_pids_[1].emplace_back(2222222222222);
  EXPECT_EQ(client.Initialize(), SUCCESS);
  EXPECT_EQ(client.Finalize(), SUCCESS);
  EXPECT_EQ(client.model_id_to_pids_.size(), 0U);
}

TEST_F(STEST_helper_runtime, CheckDevPidStatusStartByFork) {
  UdfExecutorClient client(0);
  EXPECT_EQ(client.CheckDevPidStatus(0, 0), SUCCESS);
  client.sub_proc_stat_flag_[100] = ProcStatus::NORMAL;
  EXPECT_EQ(client.CheckDevPidStatus(0, 100), SUCCESS);
}

TEST_F(STEST_helper_runtime, TestRemoteMemGroupSize) {
  ge::MemoryGroupManager group_manager;
  auto ret = group_manager.SetRemoteGroupCacheConfig("99999999999999999999");
  ASSERT_NE(ret, SUCCESS);

  std::string remote_group_cache_config = std::to_string(15 * 1024 * 1024 * 1024UL) + ":," +
                                          std::to_string(2 * 1024 * 1024 * 1024UL) + ":" +
                                          std::to_string(2 * 1024 * 1024);
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_NE(ret, SUCCESS);

  remote_group_cache_config = std::to_string(15 * 1024 * 1024 * 1024UL) + "," +
                              std::to_string(2 * 1024 * 1024 * 1024UL) + ":" + std::to_string(2 * 1024 * 1024);
  ret = group_manager.SetRemoteGroupCacheConfig(remote_group_cache_config);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(group_manager.remote_group_cache_alloc_size_, 15 * 1024 * 1024 + 2 * 1024 * 1024);
  EXPECT_EQ(group_manager.remote_group_cache_pool_list_.size(), 2);
}

TEST_F(STEST_helper_runtime, TransModelDataToComputeGraph) {
  auto root_graph = BuildDynamicRootGraph({-1}, false);
  root_graph->SetName("graph");
  root_graph->SetSessionID(0);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), ge::SUCCESS);
  auto ge_model = MakeShared<ge::GeModel>();
  auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->set_version("test_v100_r001");
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName("graph");
  ge_model->SetGraph(root_graph);
  ge_root_model->SetModelName("graph");	
  ge_root_model->SetSubgraphInstanceNameToModel("graph", ge_model);	
  bool is_unknown_shape = false;
  EXPECT_EQ(ge_root_model->CheckIsUnknownShape(is_unknown_shape), ge::SUCCESS);
  ModelBufferData model_buffer_data{};
  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  model_save_helper->SetSaveMode(false);
  EXPECT_EQ(model_save_helper->SaveToOmRootModel(ge_root_model, "graph", model_buffer_data, is_unknown_shape), ge::SUCCESS);
  ModelData model_data{};
  model_data.model_data = model_buffer_data.data.get();
	model_data.model_len = model_buffer_data.length;

  ComputeGraphPtr test_compute_graph;
  EXPECT_EQ(FlowModelOmLoader::TransModelDataToComputeGraph(model_data, test_compute_graph), ge::SUCCESS);
  EXPECT_NE(test_compute_graph, nullptr);
  EXPECT_EQ(test_compute_graph->GetName(), "graph");
}

TEST_F(STEST_helper_runtime, TestProxyDynamicModel_Execute_Success) {
  mock_handle = (void *) 0x12345678;
  mock_method = (void *) &MockHcomDestroy;

  auto root_graph = BuildDynamicRootGraph({-1}, true, true, true);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->rank_id_, "0");;
  EXPECT_EQ(handler.context_->rank_table_, "rank_table_test");
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  MockProxyDynamicModelExecutor *executor =
      reinterpret_cast<MockProxyDynamicModelExecutor *>(handle->dynamic_model_executor_.get());
  EXPECT_EQ(executor->ExceptionNotify(0, 100), SUCCESS);
  executor->OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  EXPECT_EQ(executor->data_ret_code_, 0);
  {
    // 此处释放SetRequest产生的mbuf，用例中未释放
    rtMbufPtr_t m_buf;
    (void)HeterogeneousExchangeService::GetInstance().DequeueMbuf(0, 3, &m_buf, 3000);
    (void)rtMbufFree(m_buf);
  }
  executor->Finalize();
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestProxyDynamicModel_Execute_Without_Max_attr_Success) {
  mock_handle = (void *) 0x12345678;
  mock_method = (void *) &MockHcomDestroy;

  auto root_graph = BuildDynamicRootGraph({-1}, true, false, true);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->rank_id_, "0");;
  EXPECT_EQ(handler.context_->rank_table_, "rank_table_test");
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  MockProxyDynamicModelExecutor *executor =
      reinterpret_cast<MockProxyDynamicModelExecutor *>(handle->dynamic_model_executor_.get());
  executor->OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  EXPECT_EQ(executor->data_ret_code_, 0);
  {
    // 此处释放SetRequest产生的mbuf，用例中未释放
    rtMbufPtr_t m_buf;
    (void)HeterogeneousExchangeService::GetInstance().DequeueMbuf(0, 3, &m_buf, 3000);
    (void)rtMbufFree(m_buf);
  }
  executor->Finalize();
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestProxyDynamicModel_with_retcode) {
  mock_handle = (void *) 0x12345678;
  mock_method = (void *) &MockHcomDestroy;

  auto root_graph = BuildDynamicRootGraph({-1}, true);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 0;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 0;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));

  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  (void)rtMbufGetPrivInfo(req_msg_mbuf, &head_buf, &head_size);
  if ((head_buf != nullptr) && (head_size >= sizeof(ExchangeService::MsgInfo))) {
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    msg_info->ret_code = 9999;
    msg_info->data_flag = 0;
  }
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  MockProxyDynamicModelExecutor *executor =
      reinterpret_cast<MockProxyDynamicModelExecutor *>(handle->dynamic_model_executor_.get());
  executor->OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  EXPECT_TRUE(executor->is_need_execute_model_);
  {
    // 此处释放SetRequest产生的mbuf，用例中未释放
    rtMbufPtr_t m_buf;
    (void)HeterogeneousExchangeService::GetInstance().DequeueMbuf(0, 3, &m_buf, 3000);
    rtMbufFree(m_buf);
  }
  executor->Finalize();
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestProxyDynamicModel_null_data_flag) {
  mock_handle = (void *) 0x12345678;
  mock_method = (void *) &MockHcomDestroy;

  auto root_graph = BuildDynamicRootGraph({-1}, true);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 0;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 0;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc));
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));

  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  (void)rtMbufGetPrivInfo(req_msg_mbuf, &head_buf, &head_size);
  if ((head_buf != nullptr) && (head_size >= sizeof(ExchangeService::MsgInfo))) {
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    msg_info->ret_code = 0;
    msg_info->data_flag = 1;
  }
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  MockProxyDynamicModelExecutor *executor =
      reinterpret_cast<MockProxyDynamicModelExecutor *>(handle->dynamic_model_executor_.get());
  executor->OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  EXPECT_TRUE(executor->is_need_execute_model_);
  handle->UnloadModel();
  {
    // 此处释放SetRequest产生的mbuf，用例中未释放
    rtMbufPtr_t m_buf;
    (void)HeterogeneousExchangeService::GetInstance().DequeueMbuf(0, 3, &m_buf, 3000);
    rtMbufFree(m_buf);
  }
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestDynamicModel_WithInputAlign_Execute_Success) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  map<std::string, std::string> new_option;
  new_option["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetSessionOption(new_option);
  mock_handle = (void *) 0x12345678;

  auto root_graph = BuildTwoInputDynamicRootGraph({-1}, false);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *input_queues1 = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues1->set_queue_id(1);
  input_queues1->set_device_type(CPU);
  input_queues1->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(false);
  model_request.set_enable_exception_catch(true);
  auto input_align_attrs = model_request.mutable_input_align_attrs();
  input_align_attrs->set_align_max_cache_num(4);
  input_align_attrs->set_align_timeout(200);
  input_align_attrs->set_drop_when_not_align(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->rank_id_, "0");;
  EXPECT_EQ(handler.context_->rank_table_, "rank_table_test");
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 2);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 2);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
  handler.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(STEST_helper_runtime, TestProxyDynamicModel_WithInputAlign_Execute_Success) {
  mock_handle = (void *) 0x12345678;
  mock_method = (void *) &MockHcomDestroy;

  auto root_graph = BuildTwoInputDynamicRootGraph({-1}, false);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *input_queues1 = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues1->set_queue_id(1);
  input_queues1->set_device_type(CPU);
  input_queues1->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  auto input_align_attrs = model_request.mutable_input_align_attrs();
  input_align_attrs->set_align_max_cache_num(4);
  input_align_attrs->set_align_timeout(200);
  input_align_attrs->set_drop_when_not_align(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->rank_id_, "0");;
  EXPECT_EQ(handler.context_->rank_table_, "rank_table_test");
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 2);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 2);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  uint64_t output_data = 0UL;
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc) + sizeof(uint64_t) + sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  memcpy(reinterpret_cast<void * >(reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc)),
         &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  uint64_t output_addr = reinterpret_cast<uintptr_t>(&output_data);
  void *output_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(input_buffer) + sizeof(RuntimeTensorDesc) * 2);
  memcpy(output_buffer, &output_addr, sizeof(uint64_t));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  MockProxyDynamicModelExecutor *executor =
      reinterpret_cast<MockProxyDynamicModelExecutor *>(handle->dynamic_model_executor_.get());
  executor->OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  EXPECT_EQ(executor->data_ret_code_, 0);
  {
    // 此处释放SetRequest产生的mbuf，用例中未释放
    rtMbufPtr_t m_buf;
    (void)HeterogeneousExchangeService::GetInstance().DequeueMbuf(0, 3, &m_buf, 3000);
    (void)rtMbufFree(m_buf);
  }
  executor->Finalize();
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestProxyDynamicModelWithDummy_Execute_Success) {
  mock_handle = (void *) 0x12345678;
  mock_method = (void *) &MockHcomDestroy;

  auto root_graph = BuildDynamicRootGraph({-1}, true);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(CPU);
  input_queues->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(UINT32_MAX);
  output_queues->set_device_type(CPU);
  output_queues->set_device_id(0);
  model_request.set_is_dynamic_sched(true);
  model_request.set_need_report_status(true);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(CPU);
  status_output_queues->set_device_id(0);
  auto *status_input_queues = model_request.mutable_status_queues()->add_input_queues_attrs();
  status_input_queues->set_queue_id(3);
  status_input_queues->set_device_type(CPU);
  status_input_queues->set_device_id(0);

  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->rank_id_, "0");;
  EXPECT_EQ(handler.context_->rank_table_, "rank_table_test");
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
  // prepare req_msg_mbuf
  uint64_t input_data = 0UL;
  rtMbufPtr_t req_msg_mbuf = nullptr;
  RuntimeTensorDesc input_runtime_tensor_desc{};
  input_runtime_tensor_desc.shape[0] = 1;
  input_runtime_tensor_desc.shape[1] = 2;
  input_runtime_tensor_desc.original_shape[0] = 1;
  input_runtime_tensor_desc.original_shape[1] = 2;
  input_runtime_tensor_desc.dtype = DT_FLOAT;
  input_runtime_tensor_desc.format = FORMAT_ND;
  input_runtime_tensor_desc.data_addr = reinterpret_cast<uint64_t>(&input_data);
  const size_t req_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_mbuf_size), RT_ERROR_NONE);
  void *input_buffer = nullptr;
  EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
  memcpy(input_buffer, &input_runtime_tensor_desc, sizeof(input_runtime_tensor_desc));
  // prepare resp_msg_mbuf
  rtMbufPtr_t resp_msg_mbuf = nullptr;
  const size_t resp_mbuf_size = sizeof(RuntimeTensorDesc);
  EXPECT_EQ(rtMbufAlloc(&resp_msg_mbuf, resp_mbuf_size), RT_ERROR_NONE);
  MockProxyDynamicModelExecutor *executor =
      reinterpret_cast<MockProxyDynamicModelExecutor *>(handle->dynamic_model_executor_.get());
  executor->OnInputsReady(req_msg_mbuf, resp_msg_mbuf);
  EXPECT_EQ(executor->data_ret_code_, 0);
  {
    // 此处释放SetRequest产生的mbuf，用例中未释放
    rtMbufPtr_t m_buf;
    (void)HeterogeneousExchangeService::GetInstance().DequeueMbuf(0, 3, &m_buf, 3000);
    (void)rtMbufFree(m_buf);
  }
  executor->Finalize();
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestEventHandlerClearModel) {
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  uint32_t rootModelId = 0U;
  uint32_t modelId = 0U;
  ModelHandleMock2 *modelHandleMockPtr = new ModelHandleMock2();
  handler.context_->model_handles_[rootModelId].emplace(modelId, modelHandleMockPtr);
  auto &modelHandle = *reinterpret_cast<ModelHandleMock2 *>(
    handler.context_->model_handles_[rootModelId][modelId].get());

  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;

  auto clear_model_request = request.mutable_clear_model_message();
  clear_model_request->set_model_id(rootModelId);
  clear_model_request->set_clear_msg_type(0);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(SUCCESS));
  auto mock_get_clear_model_handle =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    dynamic_model_handles.emplace_back(&modelHandle);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  clear_model_request->set_clear_msg_type(2);
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  auto mock_get_clear_model_handle2 =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    dynamic_model_handles.emplace_back(&modelHandle);
    dynamic_model_handles.emplace_back(&modelHandle);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle2));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  clear_model_request->set_clear_msg_type(2);
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle2));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle2));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  auto runtime_stub = std::make_shared<RuntimeMock>();
  RuntimeStub::SetInstance(runtime_stub);

  auto mock_get_clear_model_handle3 =
    [](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    davinci_model_runtime_ids.emplace_back(0U);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle3));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  auto mock_get_clear_model_handle4 =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    davinci_model_runtime_ids.emplace_back(0U);
    dynamic_model_handles.emplace_back(&modelHandle);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle4));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  RuntimeStub::Reset();

  auto model_handle = MakeShared<ExecutorContext::ModelHandle>();
  std::vector<uint32_t> davinci_model_runtime_ids;
  std::vector<ExecutorContext::ModelHandle *> dynamic_model_handles;
  EXPECT_EQ(model_handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids,
    dynamic_model_handles), FAILED);

  model_handle->dynamic_model_executor_ = model_handle->CreateProxyDynamicModelExecutor();
  EXPECT_NE(model_handle->dynamic_model_executor_.get(), nullptr);
  model_handle->is_dynamic_proxy_controlled_ = true;
  EXPECT_EQ(model_handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids,
    dynamic_model_handles), SUCCESS);
  EXPECT_EQ(dynamic_model_handles.size(), 1U);
  EXPECT_EQ(davinci_model_runtime_ids.size(), 1U);

  model_handle->dynamic_model_executor_.reset(nullptr);
  model_handle->is_dynamic_proxy_controlled_ = false;
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 0U;
  model_handle->inner_model_id_ = davinci_model_id;
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  dynamic_model_handles.clear();
  davinci_model_runtime_ids.clear();
  EXPECT_EQ(model_handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids,
    dynamic_model_handles), SUCCESS);
  EXPECT_EQ(dynamic_model_handles.size(), 0U);
  EXPECT_EQ(davinci_model_runtime_ids.size(), 1U);

  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestExceptionNotify) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa2>());
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  uint32_t root_model_id = 0U;
  uint32_t model_id = 0U;
  uint32_t invoke_model_id = 1U;
  ModelHandleMock2 *model_handle_mock_ptr = new ModelHandleMock2();
  ModelHandleMock2 *invoke_model_handle_mock_ptr = new ModelHandleMock2();
  handler.context_->model_handles_[root_model_id].emplace(model_id, model_handle_mock_ptr);
  handler.context_->model_handles_[root_model_id].emplace(invoke_model_id, invoke_model_handle_mock_ptr);
  auto &model_handle = *reinterpret_cast<ModelHandleMock2 *>(
      handler.context_->model_handles_[root_model_id][model_id].get());
  auto &invoke_model_handle = *reinterpret_cast<ModelHandleMock2 *>(
      handler.context_->model_handles_[root_model_id][invoke_model_id].get());
  invoke_model_handle.is_invoked_nn_ = true;

  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  request.set_type(deployer::kExecutorExceptionNotify);
  auto exception_notify_request = request.mutable_exception_notify_request();
  // not exist root model id.
  exception_notify_request->set_root_model_id(999);
  auto exception_notify = exception_notify_request->mutable_exception_notify();
  exception_notify->set_trans_id(100);
  exception_notify->set_type(0);
  exception_notify->set_scope("");
  exception_notify->set_user_context_id(111);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  exception_notify_request->set_root_model_id(root_model_id);
  EXPECT_CALL(model_handle, ExceptionNotify).WillRepeatedly(testing::Return(SUCCESS));
  auto mock_get_model_handle2 =
      [&model_handle](std::vector<uint32_t> &davinci_model_runtime_ids,
                      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
        dynamic_model_handles.emplace_back(&model_handle);
        dynamic_model_handles.emplace_back(&model_handle);
        return SUCCESS;
      };
  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_model_handle2));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Return(FAILED));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_model_handle2));
  EXPECT_CALL(model_handle, ExceptionNotify).WillRepeatedly(testing::Return(FAILED));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  auto runtime_stub = std::make_shared<MockRuntimeForClient>();
  RuntimeStub::SetInstance(runtime_stub);
  auto mock_get_clear_model_handle3 =
      [&model_handle](std::vector<uint32_t> &davinci_model_runtime_ids,
                      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
        davinci_model_runtime_ids.emplace_back(0U);
        return SUCCESS;
      };
  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_clear_model_handle3));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  auto mock_get_clear_model_handle4 =
      [&model_handle](std::vector<uint32_t> &davinci_model_runtime_ids,
                      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
        davinci_model_runtime_ids.emplace_back(0U);
        dynamic_model_handles.emplace_back(&model_handle);
        return SUCCESS;
      };
  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_clear_model_handle4));
  EXPECT_CALL(model_handle, ExceptionNotify).WillRepeatedly(testing::Return(SUCCESS));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  auto update_prof_request = request.mutable_update_prof_message();
  update_prof_request->set_is_prof_start(1);
  update_prof_request->set_prof_data("test");

  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  RuntimeStub::Reset();
  handler.Finalize();
}

TEST_F(STEST_helper_runtime, TestDynamicModelClearModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa2>());
  auto dynamic_model_executor = std::make_shared<DynamicModelExecutor>(false);
  dynamic_model_executor->aicpu_handle_ = (void *) 0x12345678;
  std::thread thread_run([&dynamic_model_executor]() {
    dynamic_model_executor->Run();
  });
  EXPECT_EQ(dynamic_model_executor->ClearModel(1), SUCCESS);
  EXPECT_EQ(dynamic_model_executor->ClearModel(2), SUCCESS);
  const DynamicModelExecutor::ModelExecuteParam eof_param {.callback = nullptr, .req_mbuf = nullptr, .resp_mbuf = nullptr};
  dynamic_model_executor->task_queue_.Push(eof_param);
  thread_run.join();
  dynamic_model_executor->aicpu_handle_ = nullptr;
  MmpaStub::GetInstance().Reset();
}

TEST_F(STEST_helper_runtime, TestCreateFakeAicpuModelAndStreamSuccess) {
  DynamicModelExecutor dynamic_model_executor(true);
  dynamic_model_executor.model_id_ = 1U;
  ASSERT_EQ(dynamic_model_executor.CreateFakeAicpuModelAndStream(), SUCCESS);
  // test npu
  dynamic_model_executor.is_host_ = false;
  ASSERT_EQ(dynamic_model_executor.CreateFakeAicpuModelAndStream(), SUCCESS);
  dynamic_model_executor.Finalize();
}

TEST_F(STEST_helper_runtime, TestDynamicSchedDeployWithFlowOnServer) {
  class MockRuntimeForServer : public MockRuntime {
   public:
    rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output)
    {
      return 1;
    }
  };
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForServer>());
  std::map<std::string, std::string> options;
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, 0);
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);
  ge::GetThreadLocalContext().SetSessionOption({{"ge.flowGraphMemMaxSize","123456789"}});

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0:0");
  flow_model->AddSubModel(graph_model);

  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  DeployResult deploy_result;
  MasterModelDeployer model_deployer;
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutorMock(flow_model, deploy_result));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  std::vector<ge::Tensor> input_tensors;
  input_tensors.resize(3);

  std::vector<ge::GeTensor> input_ge_tensors(3);
  std::vector<ge::GeTensor> output_ge_tensors;
  auto ret = executor->Execute(input_ge_tensors, output_ge_tensors);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(model_deployer.Finalize(), SUCCESS);
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
  ge::GetThreadLocalContext().SetSessionOption({{}});
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestRedeployWithMulModelInstanceOnServer) {
  class MockRuntimeForServer : public MockRuntime {
   public:
    rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output)
    {
      return 1;
    }
  };
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server_2dev.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForServer>());
  std::map<std::string, std::string> options;
  auto heterogeneous_runtime = ge::MakeShared<ge::ExecutionRuntimeHeterogeneousMock4>();
  EXPECT_NE(heterogeneous_runtime, nullptr);
  EXPECT_EQ(heterogeneous_runtime->Initialize(options), SUCCESS);
  ge::ExecutionRuntime::SetExecutionRuntime(heterogeneous_runtime);
  auto runtime_execution = (ExecutionRuntimeHeterogeneousMock4 *)ExecutionRuntime::GetInstance();
  auto &exchange_service = (ExchangeServiceMock &) runtime_execution->GetExchangeService();
  exchange_service.dequeue_tonsor_result = ACL_ERROR_RT_QUEUE_EMPTY;
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, Enqueue).WillRepeatedly(Return(SUCCESS));

  ge::GetThreadLocalContext().SetSessionOption({{"ge.flowGraphMemMaxSize", "123456789"}});

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});

  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0:0,0:0:1:0");
  flow_model->AddSubModel(graph_model);

  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  DeployResult deploy_result;
  auto *const execution_runtime = ExecutionRuntime::GetInstance();
  auto &model_deployer = execution_runtime->GetModelDeployer();
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutorMock(flow_model, deploy_result));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  std::vector<ge::Tensor> input_tensors;
  input_tensors.resize(3);

  std::vector<ge::GeTensor> input_ge_tensors(3);
  std::vector<ge::GeTensor> output_ge_tensors;
  (void)executor->Execute(input_ge_tensors, output_ge_tensors);
  dlog_setlevel(0, 0, 0);
  // 更改RESOURCE_CONFIG_PATH
  std::vector<int> lineNumbers = {17, 18, 19, 20, 21};
  DeleteLines(real_path.c_str(), lineNumbers);
  CreateRedeployFile(real_path.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  dlog_setlevel(0, 3, 0);
  exchange_service.dequeue_tonsor_result = FAILED;
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
  ge::GetThreadLocalContext().SetSessionOption({{}});
  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  RestoreFile(real_path.c_str());
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestRedeployWithProcessAbnormal) {
  class MockRuntimeForServer : public MockRuntime {
   public:
    rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output)
    {
      return 1;
    }
  };
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server_2dev.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForServer>());
  std::map<std::string, std::string> options;
  auto heterogeneous_runtime = ge::MakeShared<ge::ExecutionRuntimeHeterogeneousMock4>();
  EXPECT_NE(heterogeneous_runtime, nullptr);
  EXPECT_EQ(heterogeneous_runtime->Initialize(options), SUCCESS);
  ge::ExecutionRuntime::SetExecutionRuntime(heterogeneous_runtime);
  auto runtime_execution = (ExecutionRuntimeHeterogeneousMock4 *)ExecutionRuntime::GetInstance();
  auto &exchange_service = (ExchangeServiceMock &) runtime_execution->GetExchangeService();
  exchange_service.dequeue_tonsor_result = ACL_ERROR_RT_QUEUE_EMPTY;
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, Enqueue).WillRepeatedly(Return(SUCCESS));

  ge::GetThreadLocalContext().SetSessionOption({{"ge.flowGraphMemMaxSize","123456789"}});

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0:0,0:0:1:0");
  flow_model->AddSubModel(graph_model);

  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  DeployResult deploy_result;
  auto *const execution_runtime = ExecutionRuntime::GetInstance();
  auto &model_deployer = execution_runtime->GetModelDeployer();
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutorMock(flow_model, deploy_result));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  std::vector<ge::Tensor> input_tensors;
  input_tensors.resize(3);

  std::vector<ge::GeTensor> input_ge_tensors(3);
  std::vector<ge::GeTensor> output_ge_tensors;
  (void)executor->Execute(input_ge_tensors, output_ge_tensors);

  dlog_setlevel(0, 0, 0);
  // 触发进程异常
  auto &deploy_context = DeployContext::LocalContext();
  {
    std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
    deploy_context.AddAbnormalSubmodelInstanceName(1, "model1");
    deploy_context.AddAbnormalSubmodelInstanceName(2, "model2");
    deploy_context.AddAbnormalSubmodelInstanceName(3, "model3");
    NodeConfig node_config;
    deploy_context.AddAbnormalNodeConfig(node_config);;
    DeployPlan::DeviceInfo device_info(1, 0, 0);
    deploy_context.AddAbnormalDeviceInfo(device_info);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  dlog_setlevel(0, 3, 0);
  exchange_service.dequeue_tonsor_result = FAILED;
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
  ge::GetThreadLocalContext().SetSessionOption({{}});
  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestRedeployWithOneModelInstanceOnServer) {
  class MockRuntimeForServer : public MockRuntime {
   public:
    rtError_t rtMemGrpQuery(rtMemGrpQueryInput_t * const input, rtMemGrpQueryOutput_t *output)
    {
      return 1;
    }
  };
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server_2dev.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForServer>());
  std::map<std::string, std::string> options;
  auto heterogeneous_runtime = ge::MakeShared<ge::ExecutionRuntimeHeterogeneousMock4>();
  EXPECT_NE(heterogeneous_runtime, nullptr);
  EXPECT_EQ(heterogeneous_runtime->Initialize(options), SUCCESS);
  ge::ExecutionRuntime::SetExecutionRuntime(heterogeneous_runtime);
  auto runtime_execution = (ExecutionRuntimeHeterogeneousMock4 *)ExecutionRuntime::GetInstance();
  auto &exchange_service = (ExchangeServiceMock &) runtime_execution->GetExchangeService();
  exchange_service.dequeue_tonsor_result = ACL_ERROR_RT_QUEUE_EMPTY;
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, Dequeue).WillRepeatedly(Return(ACL_ERROR_RT_QUEUE_EMPTY));
  EXPECT_CALL(exchange_service, Enqueue).WillRepeatedly(Return(SUCCESS));

  ge::GetThreadLocalContext().SetSessionOption({{"ge.flowGraphMemMaxSize","123456789"}});

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:1");
  flow_model->AddSubModel(graph_model);

  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);
  DeployResult deploy_result;
  auto *const execution_runtime = ExecutionRuntime::GetInstance();
  auto &model_deployer = execution_runtime->GetModelDeployer();
  ASSERT_EQ(model_deployer.DeployModel(flow_model, deploy_result), SUCCESS);
  ASSERT_EQ(model_deployer.UpdateProfilingInfo(true), SUCCESS);
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutorMock(flow_model, deploy_result));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  std::vector<ge::Tensor> input_tensors;
  input_tensors.resize(3);

  std::vector<ge::GeTensor> input_ge_tensors(3);
  std::vector<ge::GeTensor> output_ge_tensors;
  (void)executor->Execute(input_ge_tensors, output_ge_tensors);

  // 更改RESOURCE_CONFIG_PATH
  std::vector<int> lineNumbers = {17, 18, 19, 20, 21};
  DeleteLines(real_path.c_str(), lineNumbers);
  CreateRedeployFile(real_path.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  exchange_service.dequeue_tonsor_result = FAILED;
  ExecutionRuntime::FinalizeExecutionRuntime();
  TsdClient::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
  ge::GetThreadLocalContext().SetSessionOption({{}});
  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  RestoreFile(real_path.c_str());
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

uint32_t g_sched_cnt_ = 0;
bool g_is_dynamic_sched_ = false;
Status DynamicSchedDequeueMbufStub(int32_t device_id, uint32_t queue_id,
                                   rtMbufPtr_t *m_buf, int32_t timeout) {
  if (!g_is_dynamic_sched_) {
    return SUCCESS;
  }
  if (queue_id == 102) {
    domi::FlowgwRequest flowgw_request;
    flowgw_request.set_input_index(102);
    auto queue_infos = flowgw_request.add_queue_infos();
    domi::QueueAttrs *queue_attrs = queue_infos->mutable_queue_attrs();
    queue_attrs->set_queue_id(1);
    queue_attrs->set_logic_id(1);
    queue_infos->set_model_uuid(1);
    queue_infos->set_logic_group_id(1);
    // use queue id as trans id
    queue_infos->set_trans_id(queue_id);
    queue_infos->set_route_label(0);
    queue_infos->set_choose_logic_id(0);
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    uint64_t input_buffer_size = 0;
    auto req_msg_mbuf_size = flowgw_request.ByteSizeLong();
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffSize(req_msg_mbuf, &input_buffer_size), RT_ERROR_NONE);
    EXPECT_EQ(flowgw_request.SerializeToArray(input_buffer, static_cast<int32_t>(req_msg_mbuf_size)), true);
    *m_buf = req_msg_mbuf;
    return SUCCESS;
  } else if (g_sched_cnt_ > 0 && queue_id == 105) {
    domi::SubmodelStatus submodel_status;
    auto queue_status = submodel_status.add_queue_statuses();
    domi::QueueAttrs *queue_attrs = queue_status->mutable_queue_attrs();
    queue_attrs->set_queue_id(0);
    queue_attrs->set_logic_id(0);
    queue_status->set_input_consume_num(0);
    queue_status->set_queue_depth(0);
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    uint64_t input_buffer_size = 0;
    auto req_msg_mbuf_size = submodel_status.ByteSizeLong();
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, req_msg_mbuf_size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffSize(req_msg_mbuf, &input_buffer_size), RT_ERROR_NONE);
    EXPECT_EQ(submodel_status.SerializeToArray(input_buffer, static_cast<int32_t>(req_msg_mbuf_size)), true);
    *m_buf = req_msg_mbuf;
    return SUCCESS;
  }
  return FAILED;
}

bool g_dynamic_sched_by_cache_ = false;
Status DynamicSchedEnqueueStub(int32_t device_id, uint32_t queue_id, size_t size, const ExchangeService::FillFunc &fill_func,
                   const ExchangeService::ControlInfo &control_info) {
  DT_ALLOW_LEAKS_GUARD(DynamicSchedEnqueueStub);
  if (queue_id == 101) {
    rtMbufPtr_t req_msg_mbuf = nullptr;
    void *input_buffer = nullptr;
    EXPECT_EQ(rtMbufAlloc(&req_msg_mbuf, size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufSetDataLen(req_msg_mbuf, size), RT_ERROR_NONE);
    EXPECT_EQ(rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer), RT_ERROR_NONE);
    EXPECT_EQ(fill_func(input_buffer, size), SUCCESS);
    google::protobuf::io::ArrayInputStream stream(input_buffer, static_cast<int32_t>(size));
    domi::FlowgwResponse flowgw_response;
    EXPECT_EQ(flowgw_response.ParseFromZeroCopyStream(&stream), true);
    {
      const auto &queue_infos = flowgw_response.queue_infos(0);
      if (g_dynamic_sched_by_cache_) {
        EXPECT_EQ(queue_infos.choose_logic_id(), 1); // cache记录为transid=102，index=1；
      } else {
        EXPECT_EQ(queue_infos.choose_logic_id(), 2); // 遍历选路索引从小到大，当优先级一样时，选择最大的index
      }
    }
    g_sched_cnt_++;
  }
  return SUCCESS;
}

TEST_F(STEST_helper_runtime, TestDynamicSchedFindGroupIndexBySched) {
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  std::map<std::string, std::string> options;
  auto heterogeneous_runtime = ge::MakeShared<ge::ExecutionRuntimeHeterogeneousMock3>();
  EXPECT_NE(heterogeneous_runtime, nullptr);
  EXPECT_EQ(heterogeneous_runtime->Initialize(options), SUCCESS);
  ge::ExecutionRuntime::SetExecutionRuntime(heterogeneous_runtime);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0");
  flow_model->AddSubModel(graph_model);

  DeployResult deploy_result;
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutor(flow_model, deploy_result));
  executor->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr1;
  DeployQueueAttr queue_attr2;
  DeployQueueAttr queue_attr3;
  executor->status_input_queue_attrs_.push_back(queue_attr1);
  executor->status_input_queue_attrs_[0].queue_id = 105;
  executor->sched_input_queue_attrs_.push_back(queue_attr2);
  executor->sched_input_queue_attrs_[0].queue_id = 101;
  executor->sched_input_queue_attrs_[0].global_logic_id = 1;
  executor->sched_output_queue_attrs_.push_back(queue_attr3);
  executor->sched_output_queue_attrs_[0].queue_id = 102;
  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {1, {route1, route2, route3}};
  executor->model_index_info_ = {{1, {{1,
      {index_info, {{1, group_info}}}
  }}}};
  executor->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    executor->cached_trans_ids_[i] = {0};
  }
  executor->routelabel_cache_info_ = {
      {{1, 0}, {{9, std::make_pair(10, "")}}}
  };
  HeterogeneousModelExecutor::QueueStatus status;
  status.queue_depth = 0;
  status.device_id = 0;
  status.device_type = 0;
  executor->queue_status_info_[1].first = status;
  executor->queue_status_info_[1].second = 0;
  auto runtime_execution = (ExecutionRuntimeHeterogeneousMock3 *)ExecutionRuntime::GetInstance();
  auto &exchange_service = (ExchangeServiceMock &) runtime_execution->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_CALL(exchange_service, Enqueue).WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  g_sched_cnt_ = 0;
  g_is_dynamic_sched_ = true;
  g_dynamic_sched_by_cache_ = false;

  ASSERT_EQ(executor->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  ASSERT_EQ(executor->ModelRunStop(), SUCCESS);
  ASSERT_EQ(executor->ModelRunStop(), SUCCESS);  // not started
  EXPECT_EQ(executor->cached_trans_ids_.size(), 1024);
  EXPECT_FALSE(executor->routelabel_cache_info_.empty());
  EXPECT_EQ(executor->routelabel_cache_info_.find({1, 0}), executor->routelabel_cache_info_.cend());
  executor->is_dynamic_sched_ = false;
  executor->status_input_queue_attrs_.clear();
  executor->sched_input_queue_attrs_.clear();
  executor->sched_output_queue_attrs_.clear();
  executor->model_index_info_.clear();
  executor->datagw_request_bindings_.clear();
  executor->cached_trans_ids_.clear();
  g_sched_cnt_ = 0;
  g_is_dynamic_sched_ = false;
  g_dynamic_sched_by_cache_ = false;
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(STEST_helper_runtime, TestDynamicSchedFindGroupIndexByCache) {
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  std::map<std::string, std::string> options;
  auto heterogeneous_runtime = ge::MakeShared<ge::ExecutionRuntimeHeterogeneousMock3>();
  EXPECT_NE(heterogeneous_runtime, nullptr);
  EXPECT_EQ(heterogeneous_runtime->Initialize(options), SUCCESS);
  ge::ExecutionRuntime::SetExecutionRuntime(heterogeneous_runtime);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0");
  flow_model->AddSubModel(graph_model);

  DeployResult deploy_result;
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutor(flow_model, deploy_result));
  executor->is_dynamic_sched_ = true;
  DeployQueueAttr queue_attr1;
  DeployQueueAttr queue_attr2;
  DeployQueueAttr queue_attr3;
  executor->status_input_queue_attrs_.push_back(queue_attr1);
  executor->status_input_queue_attrs_[0].queue_id = 105;
  executor->sched_input_queue_attrs_.push_back(queue_attr2);
  executor->sched_input_queue_attrs_[0].queue_id = 101;
  executor->sched_input_queue_attrs_[0].global_logic_id = 1;
  executor->sched_output_queue_attrs_.push_back(queue_attr3);
  executor->sched_output_queue_attrs_[0].queue_id = 102;
  DeployPlan::DeviceInfo device_info =
      DeployPlan::DeviceInfo(static_cast<int32_t>(CPU), 0, 0);
  DeployPlan::ExtendedIndexInfo index_info;
  index_info.device_info = device_info;
  index_info.submodel_instance_name = "model1";
  index_info.is_normal = true;
  DeployPlan::DynamicGroupRouteInfo route1 = {0, 0, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route2 = {1, 1, index_info, false};
  DeployPlan::DynamicGroupRouteInfo route3 = {2, 2, index_info, false};
  DeployPlan::DstGroupInfo group_info {3, {route1, route2, route3}};
  executor->model_index_info_ = {{1, {{1,
      {index_info, {{1, group_info}}}
  }}}};
  executor->datagw_request_bindings_ = {{1, 102}};
  for (size_t i = 1; i <= 1025; ++i) {
    executor->cached_trans_ids_[i] = {0};
  }
  executor->routelabel_cache_info_ = {
    {{102, 0}, {{3, std::make_pair(1, "")}}},
    {{1, 0}, {{9, std::make_pair(10, "")}}}
  };
  auto runtime_execution = (ExecutionRuntimeHeterogeneousMock3 *)ExecutionRuntime::GetInstance();
  auto &exchange_service = (ExchangeServiceMock &) runtime_execution->GetExchangeService();
  EXPECT_CALL(exchange_service, DequeueMbuf).WillRepeatedly(testing::Invoke(DynamicSchedDequeueMbufStub));
  EXPECT_CALL(exchange_service, Enqueue).WillRepeatedly(testing::Invoke(DynamicSchedEnqueueStub));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  g_sched_cnt_ = 0;
  g_is_dynamic_sched_ = true;
  g_dynamic_sched_by_cache_ = true;

  ASSERT_EQ(executor->ModelRunStart(), SUCCESS);

  // wait sched
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_EQ(executor->ModelRunStop(), SUCCESS);
  ASSERT_EQ(executor->ModelRunStop(), SUCCESS);  // not started
  executor->is_dynamic_sched_ = false;
  executor->status_input_queue_attrs_.clear();
  executor->sched_input_queue_attrs_.clear();
  executor->sched_output_queue_attrs_.clear();
  executor->model_index_info_.clear();
  executor->datagw_request_bindings_.clear();
  executor->cached_trans_ids_.clear();
  executor->routelabel_cache_info_.clear();
  g_sched_cnt_ = 0;
  g_is_dynamic_sched_ = false;
  g_dynamic_sched_by_cache_ = false;
}

TEST_F(STEST_helper_runtime, TestHostCpuEngineModel_Execute_Success) {
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  map<std::string, std::string> new_option;
  new_option["ge.exec.placement"] = "HOST";
  GetThreadLocalContext().SetSessionOption(new_option);
  auto root_graph = BuildDynamicRootGraph({-1}, true);
  // init request
  deployer::ExecutorRequest request;
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  batch_load_model_messgae->set_rank_table("rank_table_test");
  batch_load_model_messgae->set_rank_id(std::to_string(0));
  deployer::ExecutorRequest_LoadModelRequest model_request;
  auto *input_queues = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queues->set_queue_id(0);
  input_queues->set_device_type(NPU);
  input_queues->set_device_id(0);
  auto *output_queues = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queues->set_queue_id(2);
  output_queues->set_device_type(NPU);
  output_queues->set_device_id(0);
  auto *status_output_queues = model_request.mutable_status_queues()->add_output_queues_attrs();
  status_output_queues->set_queue_id(3);
  status_output_queues->set_device_type(NPU);
  status_output_queues->set_device_id(0);
  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(false);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  handler.context_->LocalContext().AddLocalModel(0, 0, BuildPneModel(root_graph));
  ASSERT_FALSE(handler.context_.get() == nullptr);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->rank_id_, "0");;
  EXPECT_EQ(handler.context_->rank_table_, "rank_table_test");
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
}

TEST_F(STEST_helper_runtime, TestDynamicSchedDeployWithFlow) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  GEFinalize();

  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config_1server.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);
  flow_model->AddSubModel(BuildPneModel(graph));

  (void)AttrUtils::SetBool(flow_model->GetRootGraph(), "dynamic_schedule_enable", true);

  auto &resources = ResourceManager::GetInstance();
  DeviceInfo cpu_device(0, CPU, 0);
  DeviceInfo npu_device0(1, NPU, 0);
  DeviceInfo npu_device1(1, NPU, 1);
  resources.device_info_list_.push_back(cpu_device);
  resources.device_info_list_.push_back(npu_device0);
  resources.device_info_list_.push_back(npu_device1);
  resources.device_info_map_[0][0][CPU] = &cpu_device;
  resources.device_info_map_[1][0][NPU] = &npu_device0;
  resources.device_info_map_[1][1][NPU] = &npu_device1;

  DeployResult deploy_result;
  // can not transfer submodel
  ASSERT_EQ(MasterModelDeployer().DeployModel(flow_model, deploy_result), FAILED);
  DeployerProxy::GetInstance().deployers_.clear();
  ResourceManager::GetInstance().Finalize();
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("NPU_COLLECT_PATH_EXE");
}

TEST_F(STEST_helper_runtime, UpdateAbnormalInstanceInTrimmingModel) {
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  std::map<std::string, std::string> options;
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  auto data2 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
  auto data3 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2);
  DEF_GRAPH(g1, "g1") {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto output_node = graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_2"});
  auto flow_model = MakeShared<FlowModel>(graph);;
  auto graph_model = BuildPneModel(graph);
  graph_model->SetLogicDeviceId("0:0:0");
  flow_model->AddSubModel(graph_model);

  DeployResult deploy_result;
  std::unordered_set<std::string> trimming_names = {"test_model", "test_model2"};
  deploy_result.model_trimming_edges_model_instances.emplace_back(trimming_names);
  auto executor = unique_ptr<HeterogeneousModelExecutor>(new HeterogeneousModelExecutorMock(flow_model, deploy_result));
  ASSERT_EQ(executor->Initialize(), SUCCESS);
  RootModelId2SubmodelName abnormal_submodel_instances_name;
  abnormal_submodel_instances_name[1]["test_model"] = true;
  executor->UpdateAbnormalInstanceList(abnormal_submodel_instances_name);
  EXPECT_EQ(executor->abnormal_submodel_instances_name_.size(), 1);
  EXPECT_EQ(executor->abnormal_submodel_instances_name_[1].size(), 2);
  ExecutionRuntime::FinalizeExecutionRuntime();
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(STEST_helper_runtime, TestClearModelExceptionData_flowgw) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForSharedContent>());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
  setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  std::map<std::string, std::string> options;
  EXPECT_EQ(InitializeHeterogeneousRuntime(options), SUCCESS);

  DeployContext context;
  deployer::DeployerResponse response;

  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(0);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  init_process_resource_request.set_device_id(1);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_id(1);
  std::vector<int32_t> res_ids_1 = {1};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_1.begin(), res_ids_1.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  context.Finalize();
  ExecutionRuntime::FinalizeExecutionRuntime();
  unsetenv("RESOURCE_CONFIG_PATH");
  unsetenv("GE_PROFILING_TO_STD_OUT");
  TsdClient::GetInstance().Finalize();
  RuntimeStub::GetInstance()->Reset();
}

TEST_F(STEST_helper_runtime, TestPrepareMsgMbuf_Success) {
  MockProxyDynamicModelExecutor executor;
  executor.dispatcher_running_flag_ = true;
  void *req_msg_mbuf = nullptr;
  void *resp_msg_mbuf = nullptr;
  EXPECT_EQ(executor.PrepareMsgMbuf(req_msg_mbuf, resp_msg_mbuf), SUCCESS);
  EXPECT_NE(req_msg_mbuf, nullptr);
  EXPECT_NE(resp_msg_mbuf, nullptr);
  (void)rtMbufFree(req_msg_mbuf);
  (void)rtMbufFree(resp_msg_mbuf);
}

class TestPeekFailedMockRuntime : public RuntimeStub {
public:
  rtError_t rtMemQueuePeek(int32_t device, uint32_t qid, size_t *bufLen, int32_t timeout) override {
    *bufLen = 0;
    return ACL_ERROR_RT_PARAM_INVALID;
  }
};

TEST_F(STEST_helper_runtime, TestPrepareMsgMbuf_Failed) {
  auto mock_runtime = std::make_shared<TestPeekFailedMockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  EXPECT_EQ(RuntimeStub::GetInstance(), mock_runtime.get());
  MockProxyDynamicModelExecutor executor;
  executor.dispatcher_running_flag_ = true;
  void *req_msg_mbuf = nullptr;
  void *resp_msg_mbuf = nullptr;
  EXPECT_NE(executor.PrepareMsgMbuf(req_msg_mbuf, resp_msg_mbuf), SUCCESS);
  EXPECT_EQ(req_msg_mbuf, nullptr);
  EXPECT_EQ(resp_msg_mbuf, nullptr);
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, TestUdfSendClearMsg) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeUDF>());
  UdfExecutorClient client(0);
  class MockExecutorMessageClientUDF : public ExecutorMessageClient {
   public:
    MockExecutorMessageClientUDF(int32_t device_id) : ExecutorMessageClient(device_id) {}
    Status SendRequest(const deployer::ExecutorRequest &request, deployer::ExecutorResponse &resp,
                       int64_t timeout) override {
      return SUCCESS;
    }
  };
  MockExecutorMessageClientUDF handle_mock(0);
  handle_mock.req_msg_queue_id_ = 100;
  handle_mock.rsp_msg_queue_id_ = 101;
  const auto get_stat_func_ = []() -> Status { return FAILED; };
  EXPECT_EQ(handle_mock.Initialize(101, get_stat_func_, false), SUCCESS);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  client.model_id_to_pids_[0].emplace_back(102);
  client.npu_device_id_related_pids_[2].insert(100);
  client.npu_device_id_related_pids_[3].insert(102);
  client.npu_device_id_related_pids_[5].insert(101);
  std::set<int32_t> device_ids = {2, 3};
  client.pid_to_message_client_[101] = MakeUnique<MockExecutorMessageClientUDF>(0);
  const auto ret = client.ClearModelRunningData(0, 1, device_ids);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handle_mock.Finalize(), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, TestUdfSendExceptionNotify) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntimeUDF>());
  UdfExecutorClient client(0);
  class MockExecutorMessageClientUDF : public ExecutorMessageClient {
   public:
    MockExecutorMessageClientUDF(int32_t device_id) : ExecutorMessageClient(device_id) {}
    Status SendRequest(const deployer::ExecutorRequest &request, deployer::ExecutorResponse &resp,
                       int64_t timeout) override {
      if (request.has_exception_notify_request()) {
        auto trans_id = request.exception_notify_request().exception_notify().trans_id();
        if (trans_id == 999) {
          return FAILED;
        }
      }
      return SUCCESS;
    }
  };
  MockExecutorMessageClientUDF handle_mock(0);
  handle_mock.req_msg_queue_id_ = 100;
  handle_mock.rsp_msg_queue_id_ = 101;
  const auto get_stat_func_ = []() -> Status { return FAILED; };
  EXPECT_EQ(handle_mock.Initialize(101, get_stat_func_, false), SUCCESS);
  client.model_id_to_pids_[0].emplace_back(100);
  client.model_id_to_pids_[0].emplace_back(101);
  client.model_id_to_pids_[1].emplace_back(200);
  client.pid_to_message_client_[100] = MakeUnique<MockExecutorMessageClientUDF>(0);
  client.pid_to_message_client_[101] = MakeUnique<MockExecutorMessageClientUDF>(0);
  deployer::DataFlowExceptionNotifyRequest req_body;
  req_body.set_root_model_id(100);
  req_body.mutable_exception_notify()->set_trans_id(1);
  auto ret = client.DataFlowExceptionNotify(req_body);
  EXPECT_NE(ret, SUCCESS);
  req_body.set_root_model_id(0);
  ret = client.DataFlowExceptionNotify(req_body);
  EXPECT_EQ(ret, SUCCESS);
  req_body.set_root_model_id(999);
  ret = client.DataFlowExceptionNotify(req_body);
  EXPECT_NE(ret, SUCCESS);
  // model id exist but message client not exist
  req_body.set_root_model_id(1);
  ret = client.DataFlowExceptionNotify(req_body);
  EXPECT_NE(ret, SUCCESS);
  req_body.set_root_model_id(0);
  req_body.mutable_exception_notify()->set_trans_id(999);
  ret = client.DataFlowExceptionNotify(req_body);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(handle_mock.Finalize(), SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(STEST_helper_runtime, GrantAndGetUdfAicpuPid) {
  MockUdfExecutorClient udf_executor_client(0);
  pid_t aicpu_pid = 0;
  auto ret = udf_executor_client.GrantAndGetUdfAicpuPid(1, 999, aicpu_pid);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_helper_runtime, NotifyUdfContinue) {
  const auto message_client = MakeShared<ExecutorMessageClient>(0);
  MockUdfExecutorClient udf_executor_client(0);
  auto ret = udf_executor_client.NotifyUdfContinue(message_client, 1, 2);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_helper_runtime, proxy_udf_proc_status) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  MockUdfProxyClient udf_proxy_client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  udf_proxy_client.LoadModel(load_model_desc);
  ProcStatus status = udf_proxy_client.GetSubProcStat();
  EXPECT_EQ(status, ProcStatus::NORMAL);
  TsdClient::GetInstance().Finalize();
}

TEST_F(STEST_helper_runtime, TestLoadModelWithInvoke) {
  DeployContext context;
  deployer::UpdateDeployPlanRequest req;
  req.set_device_id(0);
  req.set_root_model_id(1);
  auto options = req.mutable_options();
  options->mutable_global_options()->insert({"TestGlobalOption", "TestGlobalOptionValue"});
  options->mutable_session_options()->insert({"TestSessionOption", "TestSessionOptionValue"});
  options->mutable_graph_options()->insert({"TestGraphOption", "TestGraphOptionValue"});
  auto submodel_desc = req.add_submodel_descs();
  submodel_desc->set_model_path("model.om");
  submodel_desc->set_is_dynamic(false);
  submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
  auto proto_invoked_model_queues = submodel_desc->mutable_invoked_model_queues();
  deployer::ModelQueueIndices proto_model_queue_indices;
  proto_model_queue_indices.add_input_queue_indices(0);
  proto_model_queue_indices.add_output_queue_indices(1);
  (*proto_invoked_model_queues)["invoke_stub"] = std::move(proto_model_queue_indices);
  context.flow_model_receiver_.UpdateDeployPlan(req);

  deployer::AddFlowRoutePlanRequest req2;
  req2.set_root_model_id(1);
  auto exchange_plan = req2.mutable_flow_route_plan();
  deployer::EndpointDesc endpoint_desc;
  endpoint_desc.set_name("test1");
  endpoint_desc.set_type(1);  // queue
  *exchange_plan->add_endpoints() = endpoint_desc;
  *exchange_plan->add_endpoints() = endpoint_desc;
  context.flow_model_receiver_.AddFlowRoutePlan(req2);

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
}

TEST_F(STEST_helper_runtime, proxy_udf_proc_status_exit) {
  auto mock_mmpa = std::make_shared<MockMmpaForHeterogeneousRuntime>();
  mock_mmpa->tsd_get_proc_status_func_ = (void *)TsdGetProcStatusExited;
  MmpaStub::GetInstance().SetImpl(mock_mmpa);
  MockUdfProxyClient udf_proxy_client(0);
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  udf_proxy_client.LoadModel(load_model_desc);
  ProcStatus status = udf_proxy_client.GetSubProcStat();
  EXPECT_EQ(status, ProcStatus::EXITED);
  TsdClient::GetInstance().Finalize();
}

TEST_F(STEST_helper_runtime, proxy_udf_proc_status_error) {
  auto mock_mmpa = std::make_shared<MockMmpaForHeterogeneousRuntime>();
  mock_mmpa->tsd_get_proc_status_func_ = (void *)TsdGetProcStatusFailed;
  MmpaStub::GetInstance().SetImpl(mock_mmpa);
  MockUdfProxyClient udf_proxy_client(0);
  udf_proxy_client.model_id_to_pids_[0].emplace_back(111111111);
  udf_proxy_client.model_id_to_pids_[1].emplace_back(222222222);
  udf_proxy_client.pid_to_model_instances_name_[111111111] = {"model_111111111"};
  udf_proxy_client.pid_to_model_instances_name_[222222222] = {"model_222222222"};
  udf_proxy_client.pid_to_model_id_[111111111] = 0;
  udf_proxy_client.pid_to_model_id_[222222222] = 1;
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  udf_proxy_client.LoadModel(load_model_desc);
  ProcStatus status = udf_proxy_client.GetSubProcStat();
  EXPECT_EQ(status, ProcStatus::EXITED);
  EXPECT_EQ(udf_proxy_client.abnormal_model_instances_name_.size(), 0);
  status = udf_proxy_client.GetSubProcStat();
  EXPECT_EQ(udf_proxy_client.abnormal_model_instances_name_.size(), 0);
  status = udf_proxy_client.GetSubProcStat();
  EXPECT_EQ(udf_proxy_client.abnormal_model_instances_name_.size(), 2);
  EXPECT_EQ(status, ProcStatus::EXITED);
  TsdClient::GetInstance().Finalize();
}

TEST_F(STEST_helper_runtime, destroy_invalid_handle) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  auto ret = flowgw_client.DestroyHcomHandle();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_helper_runtime, destroy_valid_handle) {
  FlowGwClient flowgw_client(0, 0, {0}, false);
  flowgw_client.SetHcomInfo("test", 0);
  auto ret = flowgw_client.CreateHcomHandle();
  EXPECT_EQ(ret, SUCCESS);
  ret = flowgw_client.DestroyHcomHandle();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STEST_helper_runtime, run_exception) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  FlowGwClient flowgw_client(0, 0, {0}, false);
  EXPECT_EQ(flowgw_client.GetSubProcStat(), ProcStatus::INVALID);
  EXPECT_EQ(flowgw_client.Initialize(), SUCCESS);
  flowgw_client.SetExceptionFlag();
  const std::set<uint32_t> model_ids;
  EXPECT_EQ(flowgw_client.ClearFlowgwModelData(model_ids, 1),
      SUCCESS);
  flowgw_client.Finalize();
}

TEST_F(STEST_helper_runtime, TestDeployerDaemonCLient_ProcessMessage) {
  class MockDeployerMessageClient : public DeployerMessageClient {
   public:
    MockDeployerMessageClient(int32_t device_id) : DeployerMessageClient(device_id, true) {}

    Status WaitResponseWithMessageId(deployer::DeployerResponse &response, uint64_t message_id,
                                     int64_t timeout) override {
      return SUCCESS;
    }
  };
  class MockDeployerDaemonClient : public DeployerDaemonClient {
   public:
    explicit MockDeployerDaemonClient(int64_t client_id) : DeployerDaemonClient(client_id) {}

    std::shared_ptr<DeployerMessageClient> CreateMessageClient() override {
      return MakeShared<MockDeployerMessageClient>(0);
    }
  };

  RuntimeStub::SetInstance(std::make_shared<MockRuntimeForClient>());
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForHeterogeneousRuntime>());
  MemoryGroupManager::GetInstance().SetQsMemGroupName("DM_QS_GROUP_0");
  SubprocessManager::GetInstance().executable_paths_["deployer_daemon"] = "deployer_daemon";
  SubprocessManager::GetInstance().excpt_handle_callbacks_.clear();
  MockDeployerDaemonClient client(0);
  std::map<std::string, std::string> deployer_envs = {};
  ASSERT_EQ(client.Initialize(deployer_envs), SUCCESS);
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  ASSERT_EQ(client.ProcessDeployRequest(request, response), SUCCESS);
  ASSERT_EQ(client.ProcessHeartbeatRequest(request, response), SUCCESS);
  client.sub_deployer_proc_stat_ = ProcStatus::EXITED;
  ASSERT_NE(client.ProcessHeartbeatRequest(request, response), SUCCESS);
  client.sub_deployer_proc_stat_ = ProcStatus::STOPPED;
  ASSERT_NE(client.ProcessHeartbeatRequest(request, response), SUCCESS);
  ASSERT_EQ(client.Finalize(), SUCCESS);

  HeterogeneousExchangeService::GetInstance().Finalize();
  RuntimeStub::Reset();
  MmpaStub::GetInstance().Reset();
}
}  // namespace ge
