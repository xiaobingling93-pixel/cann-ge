/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_graphs_manager.h"
#include "common/checker.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/memory/tensor_trans_utils.h"
#include "api/aclgrph/option_utils.h"

namespace ge {
Status UserGraphsManager::AddGraph(uint32_t user_graph_id, const Graph &graph,
  const std::map<std::string, std::string> &options) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.AddGraph(user_graph_id, graph, options, domi::GetContext());
  }
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  SetLocalOmgContext(domi::GetContext());
  GetThreadLocalContext().SetGraphOption(options);
  std::lock_guard<std::mutex> locker(user_graph_ctrl_mutex_);
  auto iter = ids_to_user_graph_ctrl_.find(user_graph_id);
  if (iter == ids_to_user_graph_ctrl_.end()) {
    auto user_graph_ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context_, graph_manager_);
    GE_ASSERT_NOTNULL(user_graph_ctrl);
    GE_ASSERT_SUCCESS(user_graph_ctrl->AddGraphInstance());
    ids_to_user_graph_ctrl_[user_graph_id] = std::move(user_graph_ctrl);
  } else {
    GE_ASSERT_SUCCESS(iter->second->AddGraphInstance());
  }
  return SUCCESS;
}

Status UserGraphsManager::BuildGraph(uint32_t user_graph_id, const std::vector<GeTensor> &inputs,
  uint64_t session_id) const {
  if (!EnableSliceSchedule()) {
    GeRootModelPtr ge_root_model;
    return graph_manager_.BuildGraph(user_graph_id, inputs, ge_root_model, session_id, true);
  }
  (void)user_graph_id;
  (void)inputs;
  return SUCCESS;
}

Status UserGraphsManager::RunGraphAsync(uint32_t user_graph_id, std::vector<gert::Tensor> &&inputs,
    uint64_t session_id, const RunAsyncCallbackV2 &callback) {

  if (!EnableSliceSchedule()) {
    return graph_manager_.RunGraphAsync(user_graph_id, std::move(inputs), session_id, callback);
  }
  UserGraphControl *user_graph_control = nullptr;
  {
    std::lock_guard<std::mutex> locker(user_graph_ctrl_mutex_);
    auto iter = ids_to_user_graph_ctrl_.find(user_graph_id);
    GE_ASSERT_TRUE(iter != ids_to_user_graph_ctrl_.end());
    user_graph_control = iter->second.get();
  }
  GE_ASSERT_NOTNULL(user_graph_control, "Failed to find user graph ctrl of graph[%u], session[]", user_graph_id);

  auto exe_task = MakeUnique<UserGraphExecution>(user_graph_id, std::move(inputs), callback, session_id);
  GE_ASSERT_NOTNULL(exe_task);
  user_graph_control->RunGraphAsync(exe_task);
  return SUCCESS;
}

UserGraphControl* UserGraphsManager::GetUserGraphControl(uint32_t user_graph_id) {
  std::lock_guard<std::mutex> locker(user_graph_ctrl_mutex_);
  UserGraphControl *user_graph_control = nullptr;
  auto iter = ids_to_user_graph_ctrl_.find(user_graph_id);
  GE_ASSERT_TRUE(iter != ids_to_user_graph_ctrl_.end(), "Failed to find user graph ctrl of graph[%u]", user_graph_id);
  user_graph_control = iter->second.get();
  return user_graph_control;
}

Status UserGraphsManager::CompileGraph(uint32_t user_graph_id, uint64_t session_id, const vector<ge::Tensor> &inputs) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.CompileGraph(user_graph_id, session_id, inputs);
  }
  UserGraphControl *user_graph_control = GetUserGraphControl(user_graph_id);
  GE_ASSERT_NOTNULL(user_graph_control, "Failed to find user graph ctrl of graph[%u], session[]", user_graph_id);
  GE_ASSERT_SUCCESS(user_graph_control->CompileGraph(session_id));
  return SUCCESS;
}

Status UserGraphsManager::GetCompiledGraphSummary(uint32_t user_graph_id, CompiledGraphSummaryPtr &summary) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.GetCompiledGraphSummary(user_graph_id, summary);
  }
  UserGraphControl *user_graph_control = GetUserGraphControl(user_graph_id);
  GE_ASSERT_NOTNULL(user_graph_control, "Failed to find user graph ctrl of graph[%u], session[]", user_graph_id);
  summary = user_graph_control->GetCompiledGraphSummary();
  return SUCCESS;
}

Status UserGraphsManager::LoadGraph(const uint32_t user_graph_id, const std::map<AscendString, AscendString> &options,
                                    void *stream) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.LoadGraph(user_graph_id, options, stream);
  }
  UserGraphControl *user_graph_control = GetUserGraphControl(user_graph_id);
  GE_ASSERT_NOTNULL(user_graph_control, "Failed to find user graph ctrl of graph[%u], session[]", user_graph_id);
  GE_ASSERT_SUCCESS(user_graph_control->LoadGraph(options, stream));
  return SUCCESS;
}

Status UserGraphsManager::ExecuteGraphWithStreamAsync(uint32_t user_graph_id, void *stream, 
                                                      const std::vector<gert::Tensor> &inputs,
                                                      std::vector<gert::Tensor> &outputs, uint64_t session_id) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.ExecuteGraphWithStreamAsync(user_graph_id, stream, inputs, outputs);
  }
  UserGraphControl *user_graph_control = GetUserGraphControl(user_graph_id);
  GE_ASSERT_NOTNULL(user_graph_control, "Failed to find user graph ctrl of graph[%u], session[]", user_graph_id);
  auto exe_task = MakeUnique<UserGraphExecution>(user_graph_id, inputs, nullptr, session_id);
  GE_ASSERT_NOTNULL(exe_task);
  exe_task->stream = stream;
  exe_task->session_id = session_id;
  exe_task->rt_outputs = &outputs;
  exe_task->load_options = user_graph_control->GetLoadOptions();
  GE_ASSERT_SUCCESS(user_graph_control->ExecuteGraphWithStreamAsync(std::move(exe_task)));
  return SUCCESS;
}

Status UserGraphsManager::Finalize() {
  ids_to_user_graph_ctrl_.clear();
  return SUCCESS;
}

Status UserGraphsManager::RemoveGraph(uint32_t user_graph_id) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.RemoveGraph(user_graph_id);
  }
  std::lock_guard<std::mutex> locker(user_graph_ctrl_mutex_);
  auto iter = ids_to_user_graph_ctrl_.find(user_graph_id);
  if (iter == ids_to_user_graph_ctrl_.end()) {
    GELOGE(PARAM_INVALID, "Failed to remove graph %u which does not exist.", user_graph_id);
    return FAILED;
  }
  GE_ASSERT_SUCCESS(iter->second->Finalize());
  (void)ids_to_user_graph_ctrl_.erase(user_graph_id);
  GELOGI("Remove graph %u success.", user_graph_id);
  return SUCCESS;
}

bool UserGraphsManager::IsGraphNeedRebuild(uint32_t user_graph_id) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.IsGraphNeedRebuild(user_graph_id);
  }
  std::lock_guard<std::mutex> locker(user_graph_ctrl_mutex_);
  auto iter = ids_to_user_graph_ctrl_.find(user_graph_id);
  if (iter == ids_to_user_graph_ctrl_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist, check rebuild invalid", user_graph_id);
    GELOGE(PARAM_INVALID, "Graph %u need rebuild when does not exist.", user_graph_id);
    return true;
  }
  return iter->second->IsUserGraphNeedRebuild();
}

Status UserGraphsManager::GetCompiledFlag(uint32_t user_graph_id, bool &flag) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.GetCompiledFlag(user_graph_id, flag);
  }
  const UserGraphControl *user_graph_control = GetUserGraphControl(user_graph_id);
  GE_ASSERT_NOTNULL(user_graph_control);
  flag = user_graph_control->GetCompiledFlag();
  return SUCCESS;
}

Status UserGraphsManager::SetCompiledFlag(uint32_t user_graph_id, bool flag) {
  if (!EnableSliceSchedule()) {
    return graph_manager_.SetCompiledFlag(user_graph_id, flag);
  }
  UserGraphControl *user_graph_control = GetUserGraphControl(user_graph_id);
  GE_ASSERT_NOTNULL(user_graph_control);
  user_graph_control->SetCompiledFlag(flag);
  return SUCCESS;
}

Status UserGraphsManager::GetOmeContextByGraphId(const GraphId &graph_id, OmeContext &ome_context) const {
  GE_ASSERT_SUCCESS(graph_manager_.GetOmeContextByGraphId(graph_id, ome_context));
  return SUCCESS;
}

}  // namespace ge