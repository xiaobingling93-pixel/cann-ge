/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow_api.h"
#include <atomic>
#include <malloc.h>
#include "dflow_session_manager.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "rt_error_codes.h"
#include "common/option_supportion_checker/option_supportion_checker.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "acl/acl.h"
namespace ge {
namespace dflow {
namespace {
constexpr uint32_t kExternalErrorCodeMaxValue = 9999999U; // user define error code max value
constexpr uint64_t INVALID_SESSION_ID = 0xFFFFFFFFFFFFFFFFULL;
std::atomic<bool> acl_initialized{false};

void ConvertAscendStringMap(const std::map<ge::AscendString, ge::AscendString> &options,
                            std::map<std::string, std::string> &str_options) {
  for (auto &option_item : options) {
    if (option_item.first.GetLength() == 0) {
      GELOGE(ge::FAILED, "Construct session failed, option key is empty.");
      REPORT_INNER_ERR_MSG("E19999", "Construct session failed, option key is empty.");
      return;
    }
    const std::string &key = option_item.first.GetString();
    const std::string &val = option_item.second.GetString();
    str_options[key] = val;
  }
}

std::atomic_bool g_dflow_ge_initialized{false};
std::mutex g_dflow_ge_release_mutex;  // GEFinalize and ~DFlowSession use
std::shared_ptr<DFlowSessionManager> g_dflow_session_manager;
}

Status DFlowInitialize(const std::map<AscendString, AscendString> &options) {
  if (g_dflow_ge_initialized) {
    GELOGW("DFlowInitialize is called more than once");
    return SUCCESS;
  }
  if (!acl_initialized) {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
      GELOGE(FAILED, "ACL init failed.");
      return FAILED;
    } else {
      GELOGI("ACL init success.");
      acl_initialized.store(true);
    }
  }
  // todo call GEInitialize in new so
  GE_TIMESTAMP_START(DflowInitializeAll);
  GELOGI("sessionManager initial.");
  GE_TIMESTAMP_START(DflowSessionManagerInitialize);
  g_dflow_session_manager = ge::MakeShared<dflow::DFlowSessionManager>();
  if (g_dflow_session_manager == nullptr) {
    GELOGE(GE_CLI_INIT_FAILED, "[Init][Create]SessionManager failed");
    return FAILED;
  }
  g_dflow_session_manager->Initialize();
  GE_TIMESTAMP_EVENT_END(DflowSessionManagerInitialize, "InnerInitialize::DflowSessionManagerInitialize");

  GE_CHK_STATUS_RET(ge::DFlowInitializeInner(options), "Failed to call dflow initialize inner");
  g_dflow_ge_initialized = true;
  GELOGT(TRACE_STOP, "DFlowInitialize finished");
  GE_TIMESTAMP_EVENT_END(DflowInitializeAll, "DflowInitialize::All");
  return SUCCESS;
}

// DFlow finalize, releasing all resources
Status DFlowFinalize() {
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kGEFinalize);
  // check init status
  if (!g_dflow_ge_initialized) {
    GELOGW("[FINAL]DFlowFinalize is called before DFlowInitialize");
    return SUCCESS;
  }
  if (acl_initialized) {
    aclFinalize();
    acl_initialized.store(false);
  }
  std::lock_guard<std::mutex> lock(g_dflow_ge_release_mutex);
  GELOGT(TRACE_INIT, "DFlowFinalize start.");

  GELOGI("DflowSessionManager finalization.");
  if (g_dflow_session_manager != nullptr) {
    g_dflow_session_manager->Finalize();  // always success.
  }
  (void)malloc_trim(0);
  g_dflow_ge_initialized = false;
  ge::DFlowFinalizeInner();
  // todo GEFinalize
  GELOGT(TRACE_STOP, "DFlowFinalize finished");
  return SUCCESS;
}

namespace {
void ConstructSession(const std::map<std::string, std::string> &options, SessionPtr &session_impl) {
  GELOGT(TRACE_INIT, "DFlowSession Constructor start");
  // check init status
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Construct session failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Construct session failed because lack GEInitialize call before.");
    return;
  }
  // call Initialize
  if (ge::GEAPICheckSupportedSessionOptions(options) != SUCCESS) {
    GELOGW("[Check][Param] Check supported options failed.");
  }
  uint64_t tmp_session_id = 0UL;
  session_impl = g_dflow_session_manager->CreateSession(options, tmp_session_id);
  // failed guarder, should call GE_DISMISS_GUARD if success
  GE_DISMISSABLE_GUARD(create_failed, ([tmp_session_id]() {g_dflow_session_manager->DestroySession(tmp_session_id);}));
  if (session_impl == nullptr) {
    GELOGE(FAILED, "Construct session failed.");
    REPORT_INNER_ERR_MSG("E19999", "Construct session failed.");
    return;
  }
  GE_DISMISS_GUARD(create_failed);
  GELOGT(TRACE_STOP, "DFlowSession construct finished, session id is %lu", tmp_session_id);
}
} // namespace

DFlowSession::DFlowSession(const std::map<AscendString, AscendString> &options) {
  std::map<std::string, std::string> str_options;
  ConvertAscendStringMap(options, str_options);
  ConstructSession(str_options, dflow_session_impl_);
}

DFlowSession::~DFlowSession() {
  if (dflow_session_impl_ == nullptr) {
    return;
  }
  GELOGT(TRACE_INIT, "Start to destroy session.");
  // 0.check init status
  if (!g_dflow_ge_initialized) {
    GELOGW("GE is not yet initialized or is finalized.");
    return;
  }
  Status ret = FAILED;
  std::lock_guard<std::mutex> lock(g_dflow_ge_release_mutex);
  try {
    const uint64_t session_id = dflow_session_impl_->GetSessionId();
    // call DestroySession
    GELOGT(TRACE_RUNNING, "DFlowSession id is %lu", session_id);
    ret = g_dflow_session_manager->DestroySession(session_id);
  } catch (std::exception &e) {
    (void)e;
    GELOGE(GE_CLI_SESS_DESTROY_FAILED, "[Destructor][DFlowSession]Failed: an exception occurred");
    REPORT_INNER_ERR_MSG("E19999", "Failed to destroy session: an exception occurred");
  }

  // check return status, return, update session id if success
  if (ret != SUCCESS) {
    GELOGE(ret, "[Destructor][DFlowSession]Failed, error code:%u.", ret);
    REPORT_INNER_ERR_MSG("E19999", "Destroy session failed, error code:%u.", ret);
  }

  GELOGT(TRACE_STOP, "DFlowSession has been successfully destroyed");
}

Status DFlowSession::AddGraph(uint32_t graph_id, const FlowGraph &graph, const std::map<AscendString, AscendString> &options) {
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][DFlowSession]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }
  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();
  const std::string graph_name = graph.GetName();
  GE_ASSERT_TRUE((!graph_name.empty()), "Add graph failed, get graph name failed.");
  GELOGT(TRACE_INIT, "Start to add graph in DFlowSession. graph_id: %u, graph_name: %s, session_id: %lu.", graph_id,
         graph_name.c_str(), session_id);

  std::map<std::string, std::string> str_options;
  ConvertAscendStringMap(options, str_options);
  if (ge::GEAPICheckSupportedGraphOptions(str_options) != SUCCESS) {
    GELOGW("[Check][Param] Check supported options failed.");
  }
  GELOGD("Adding graph to session");
  const Status ret = dflow_session_impl_->AddGraph(graph_id, graph, str_options);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Add graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         session_id, graph_id);

  GELOGI("AddGraph finished in DFlowSession, graph_id: %u, session_id: %lu.", graph_id, session_id);
  return SUCCESS;
}

Status DFlowSession::RemoveGraph(uint32_t graph_id) {
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][DFlowSession]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }
  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kRemoveGraph);
  GELOGT(TRACE_INIT, "DFlowSession RemoveGraph start, graph_id: %u", graph_id);

  // call RemoveGraph
  const Status ret = dflow_session_impl_->RemoveGraph(graph_id);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Remove graph failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, session_id, graph_id);

  GELOGT(TRACE_STOP, "DFlowSession RemoveGraph finished, graph_id: %u, session_id:%lu", graph_id, session_id);
  return ret;
}

Status DFlowSession::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][DFlowSession]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kBuildGraph);
  GELOGT(TRACE_INIT, "start to build graph, session_id: %lu, graph_id: %u, input size %zu",
         session_id, graph_id, inputs.size());

  const Status ret = dflow_session_impl_->BuildGraph(graph_id, inputs);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Build graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         session_id, graph_id);
  GELOGD("BuildGraph finished in DFlowSession, graph_id: %u", graph_id);
  return SUCCESS;
}

uint64_t DFlowSession::GetSessionId() const {
  if (dflow_session_impl_ != nullptr)  {
    return dflow_session_impl_->GetSessionId();
  }
  return INVALID_SESSION_ID;
}

Status DFlowSession::FeedDataFlowGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, const DataFlowInfo &info,
                                       int32_t timeout) {
  return FeedDataFlowGraph(graph_id, {}, inputs, info, timeout);
}

Status DFlowSession::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                  const std::vector<Tensor> &inputs, const DataFlowInfo &info, int32_t timeout) {
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Feed][Data]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Feed data failed because lack GEInitialize call before.");
    return FAILED;
  }

  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();

  GELOGI("Feed data flow graph, graph_id: %u, timeout: %d ms", graph_id, timeout);
  const Status ret = dflow_session_impl_->FeedDataFlowGraph(graph_id, indexes, inputs, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(ret, "[Feed][Data]Failed, error code:%u, session_id:%lu, graph_id:%u.", ret, session_id, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Feed data flow graph failed , error code:%u, session_id:%lu, graph_id:%u", ret,
                      session_id, graph_id);
    return (ret > kExternalErrorCodeMaxValue) ? FAILED : ret;
  }
  return ret;
}

Status DFlowSession::FeedDataFlowGraph(uint32_t graph_id, const std::vector<FlowMsgPtr> &inputs, int32_t timeout) {
  return FeedDataFlowGraph(graph_id, {}, inputs, timeout);
}

Status DFlowSession::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                  const std::vector<FlowMsgPtr> &inputs, int32_t timeout) {
  GE_CHK_BOOL_RET_STATUS(g_dflow_ge_initialized, FAILED,
                         "[Feed][FlowMsg]Failed because lack GEInitialize call before.");

  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();

  GELOGI("Feed flow msg, graph_id: %u, timeout: %d ms", graph_id, timeout);
  const Status ret = dflow_session_impl_->FeedDataFlowGraph(graph_id, indexes, inputs, timeout);
  const auto status = ret > kExternalErrorCodeMaxValue ? FAILED : ret;
  GE_CHK_BOOL_RET_STATUS((ret == SUCCESS || ret == ACL_ERROR_GE_REDEPLOYING || ret == ACL_ERROR_GE_SUBHEALTHY),
                         status, "[Feed][FlowMsg]Failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, session_id, graph_id);
  return ret;
}

Status DFlowSession::FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, uint32_t index,
                            const DataFlowInfo &info, int32_t timeout) {
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Feed][RawData]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Feed raw data failed because lack GEInitialize call before.");
    return FAILED;
  }
  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();

  GELOGI("Feed raw data to data flow graph, graph_id: %u, timeout: %d ms", graph_id, timeout);
  const Status ret = dflow_session_impl_->FeedRawData(graph_id, raw_data_list, index, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(ret, "[Feed][Data]Failed, error code:%u, session_id:%lu, graph_id:%u.", ret, session_id, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Feed data flow graph failed , error code:%u, session_id:%lu, graph_id:%u", ret,
                      session_id, graph_id);
    return (ret > kExternalErrorCodeMaxValue) ? FAILED : ret;
  }
  return ret;
}

Status DFlowSession::FetchDataFlowGraph(uint32_t graph_id, std::vector<Tensor> &outputs, DataFlowInfo &info,
                                   int32_t timeout) {
  return FetchDataFlowGraph(graph_id, {}, outputs, info, timeout);
}

Status DFlowSession::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                   std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout) {
  if (!g_dflow_ge_initialized) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Fetch][Data]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Fetch data failed because lack GEInitialize call before.");
    return FAILED;
  }

  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();

  GELOGI("Fetch data flow graph, graph_id: %u, timeout: %d ms", graph_id, timeout);
  Status ret = dflow_session_impl_->FetchDataFlowGraph(graph_id, indexes, outputs, info, timeout);
  const bool need_convert_error_code = ((ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) && timeout != 0);
  ret = need_convert_error_code ? ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT : ret;
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(ret, "[Fetch][Data]Failed, error code:%u, session_id:%lu, graph_id:%u.", ret, session_id, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Fetch data flow graph failed , error code:%u, session_id:%lu, graph_id:%u", ret,
                      session_id, graph_id);
    return (ret > kExternalErrorCodeMaxValue) ? FAILED : ret;
  }
  return ret;
}

Status DFlowSession::FetchDataFlowGraph(uint32_t graph_id, std::vector<FlowMsgPtr> &outputs, int32_t timeout) {
  return FetchDataFlowGraph(graph_id, {}, outputs, timeout);
}

Status DFlowSession::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                   std::vector<FlowMsgPtr> &outputs, int32_t timeout) {
  GE_CHK_BOOL_RET_STATUS(g_dflow_ge_initialized, FAILED,
                         "[Fetch][FlowMsg]Failed because lack GEInitialize call before.");
  GE_CHECK_NOTNULL(dflow_session_impl_);
  const auto &session_id = dflow_session_impl_->GetSessionId();

  GELOGI("Fetch flow msg, graph_id: %u, timeout: %d ms", graph_id, timeout);
  Status ret = dflow_session_impl_->FetchDataFlowGraph(graph_id, indexes, outputs, timeout);
  const bool need_convert_error_code = ((ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) && timeout != 0);
  ret = need_convert_error_code ? ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT : ret;
  const auto status = ret > kExternalErrorCodeMaxValue ? FAILED : ret;
  GE_CHK_BOOL_RET_STATUS((ret == SUCCESS || ret == ACL_ERROR_GE_REDEPLOYING || ret == ACL_ERROR_GE_SUBHEALTHY),
                         status, "[Fetch][FlowMsg]Failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, session_id, graph_id);
  return ret;
}
} // namespace dflow
} // namespace ge