/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "session/session_manager.h"
#include <memory>
#include <utility>
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/manager/session_id_manager.h"
#include "common/memory/tensor_trans_utils.h"

namespace ge {
Status SessionManager::Initialize() {
  if (init_flag_) {
    GELOGW("Session Manager has been initialized.");
    return SUCCESS;
  }
  init_flag_ = true;
  return SUCCESS;
}

Status SessionManager::Finalize() {
  if (!init_flag_) {
    GELOGW("Session Manager has not been initialized.");
    return SUCCESS;
  }
  const std::unique_lock<std::shared_mutex> lock(mutex_);
  for (auto iter = session_manager_map_.cbegin(); iter != session_manager_map_.cend(); ++iter) {
    (void)iter->second->Finalize();
  }
  session_manager_map_.clear();
  init_flag_ = false;
  return SUCCESS;
}

Status SessionManager::SetRtContext(SessionId session_id, rtContext_t rt_context) const {
  GELOGI("set rt_context RT_CTX_NORMAL_MODE, device id:%u.", GetContext().DeviceId());
  GE_CHK_STATUS_RET(rtCtxCreate(&rt_context, RT_CTX_NORMAL_MODE, static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddRtContext(session_id, rt_context);
  return SUCCESS;
}

Status SessionManager::CreateSession(const std::map<std::string, std::string> &options, SessionId &session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Create][Session]fail for Session manager is not initialized.");
    REPORT_INNER_ERR_MSG("E19999", "CreateSession fail for Session manager is not initialized.");
    return GE_SESSION_MANAGER_NOT_INIT;
  }

  for (const auto &item : options) {
    GELOGI("GE option: %s, value: [%s].", item.first.c_str(), item.second.c_str());
  }

  SessionId next_session_id = 0;

  const auto nextSessionIdRet = GetNextSessionId(next_session_id);
  if (nextSessionIdRet != SUCCESS) {
    return nextSessionIdRet;
  }

  SessionPtr session_ptr = MakeShared<InnerSession>(next_session_id, options);
  if (session_ptr == nullptr) {
    return MEMALLOC_FAILED;
  }

  Status ret = session_ptr->Initialize();
  if (ret != SUCCESS) {
    return ret;
  }

  {
    const std::unique_lock<std::shared_mutex> lock(mutex_);
    (void)session_manager_map_.emplace(std::pair<SessionId, SessionPtr>(next_session_id, session_ptr));
  }

  session_id = next_session_id;

  // create a context
  ret = SetRtContext(session_id, rtContext_t());

  return ret;
}

Status SessionManager::DestroySession(SessionId session_id) {
  if (!init_flag_) {
    GELOGW("[Destroy][Session]Session manager is not initialized, session_id:%lu.", session_id);
    return SUCCESS;
  }

  SessionPtr session_ptr = nullptr;
  {
    const std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto it = session_manager_map_.find(session_id);
    if (it == session_manager_map_.end()) {
      return GE_SESSION_NOT_EXIST;
    }
    session_ptr = it->second;
  }

  // Unified destruct rt_context
  RtContextUtil::GetInstance().DestroyRtContexts(session_id);

  const auto ret = session_ptr->Finalize();
  if (ret != SUCCESS) {
    return ret;
  }

  {
    const std::unique_lock<std::shared_mutex> lock(mutex_);
    (void)session_manager_map_.erase(session_id);
  }

  return ret;
}

SessionPtr SessionManager::GetSession(SessionId session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Get][Session]fail for Session manager is not initialized, session_id:%lu.",
           session_id);
    REPORT_INNER_ERR_MSG("E19999", "GetSession fail for Session manager is not initialized, session_id:%lu.", session_id);
    return nullptr;
  }
  const std::shared_lock<std::shared_mutex> lock(mutex_);
  const auto it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    GELOGE(GE_SESSION_NOT_EXIST, "[Find][InnerSession] fail for %lu does not exist", session_id);
    REPORT_INNER_ERR_MSG("E19999", "GetSession fail for InnerSession does not exist, session_id:%lu.", session_id);
    return nullptr;
  }
  return it->second;
}

Status SessionManager::GetNextSessionId(SessionId &next_session_id) const {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Get][NextSessionId]fail for Session manager is not initialized.");
    REPORT_INNER_ERR_MSG("E19999", "GetNextSessionId fail for Session manager is not initialized.");
    return GE_SESSION_MANAGER_NOT_INIT;
  }

  next_session_id = SessionIdManager::GetNextSessionId();
  return SUCCESS;
}
Status SessionManager::GetVariables(SessionId session_id, const std::vector<std::string> &var_names,
                                    std::vector<Tensor> &var_values) {
  // step 0: get session
  const SessionPtr &inner_session = GetSession(session_id);
  if (inner_session == nullptr) {
    GELOGE(FAILED, "[Get][Session] failed, session_id:%lu.", session_id);
    return FAILED;
  }

  // step 1: get all variable
  std::map<std::string, GeTensorDesc> all_variables;
  Status ret = inner_session->GetAllVariables(all_variables);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][AllVariables]failed.");
    return FAILED;
  }

  // srep 2: create check point graph
  Graph graph = Graph("checkpoint");
  ret = inner_session->GenCheckPointGraph(all_variables, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[GenCheck][PointGraph] failed.");
    return FAILED;
  }

  // step 3: run check point graph
  const uint32_t graph_id = GetCurrentSecondTimestap();
  ret = inner_session->AddGraph(graph_id, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Add][Graph] failed.");
    return FAILED;
  }

  const std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  ret = inner_session->RunGraph(graph_id, inputs, outputs);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Run][Graph] failed.");
    return FAILED;
  }

  // step 4: save variables
  ret = inner_session->SaveVariables(graph, var_names, outputs, var_values);
  GELOGD("[SessionManager] outputs size is [%zu], var values size is [%zu].", outputs.size(), var_values.size());

  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][Variables] failed.");
    return FAILED;
  }

  // step 5: remove graph
  ret = inner_session->RemoveGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Remove][Graph] failed.");
    return FAILED;
  }
  return ret;
}

size_t SessionManager::NumSessions() {
  const std::shared_lock<std::shared_mutex> lock(mutex_);
  return session_manager_map_.size();
}
}  // namespace ge
