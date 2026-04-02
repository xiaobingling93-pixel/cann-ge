/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "graph/bin_cache/op_binary_cache.h"
#include "rt_error_codes.h"
#include "framework/common/taskdown_common.h"
#include "graph/ge_context.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "single_op/task/build_task_utils.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"

namespace ge {
namespace hybrid {
namespace {
const std::string COMPILE_INFO_JSON = "compile_info_json";
const std::string COMPILE_INFO_KEY = "compile_info_key";
const std::string ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
const std::string ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";
constexpr int32_t kDefaultTimeOut = -1;

Status BuildAicoreNodeTask(const NodePtr &node, const std::vector<domi::TaskDef> *const task_defs,
                           const HybridModel &model, AiCoreNodeTask &node_task) {
  AiCoreTaskBuilder builder(node, *task_defs, model, node_task);
  GE_CHK_STATUS_RET(builder.BuildTask(), "[Invoke][BuildTask][%s(%s)] Failed to build op tasks.",
                    node->GetName().c_str(), node->GetType().c_str());

  GE_CHK_STATUS_RET(builder.LoadAicpuTask());
  // set fused_task
  if (model.GetNodeBinMode() == fuzz_compile::kOneNodeMultipleBinsMode) {
    GE_CHK_STATUS_RET(builder.LoadFusedTask());
  }
  return SUCCESS;
}
}  // namespace
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, AiCoreNodeExecutor);
AiCoreNodeTask::AiCoreNodeTask() : NodeTask() {
}
AiCoreNodeTask::AiCoreNodeTask(std::vector<std::unique_ptr<AiCoreOpTask>> &&tasks)
    : NodeTask(), tasks_(std::move(tasks)) {
}

Status AiCoreNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGI("AiCoreNodeExecutor(%s) LoadTask Start.", node->GetName().c_str());
  const auto *const task_defs = model.GetTaskDefs(node);
  std::unique_ptr<AiCoreNodeTask> node_task = MakeUnique<AiCoreNodeTask>();
  GE_CHECK_NOTNULL(node_task);
  if ((task_defs == nullptr) || task_defs->empty()) {
    const auto node_item = model.GetNodeItem(node);
    GE_CHECK_NOTNULL(node_item);
    if (node_item->IsNoOp()) {
      task = MakeShared<NoOpTask>();
      return SUCCESS;
    } else {
      if (model.GetNodeBinMode() != fuzz_compile::kOneNodeMultipleBinsMode) {
        GELOGE(FAILED, "[Get][Task_defs]Task_defs is empty for node (%s(%s)), check invalid", node->GetName().c_str(),
               node->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "Task_defs is empty for node (%s(%s)), check invalid", node->GetName().c_str(),
                          node->GetType().c_str());
        return FAILED;
      }
      const std::vector<domi::TaskDef> empty_task_defs;
      GE_CHK_STATUS_RET(BuildAicoreNodeTask(node, &empty_task_defs, model, *node_task));
      task = std::move(node_task);
      GELOGI("AiCoreNodeExecutor(%s) LoadTask without kernel End.", node->GetName().c_str());
      return SUCCESS;
    }
  }
  GE_CHK_STATUS_RET(BuildAicoreNodeTask(node, task_defs, model, *node_task));
  task = std::move(node_task);
  GELOGI("AiCoreNodeExecutor(%s) LoadTask with kernel End.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreNodeTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] Start");
  if (context.GetNodeItem().IsNoOp()) {
    GELOGD("[%s] Skipping execution for op with empty outputs", context.GetNodeName());
    const auto ret = context.TryExecuteCallback(done_callback);
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] End");
    return ret;
  }

  if (aicpu_exec_) {
    GELOGI("Node[%s] can't Partially supported task executing tiling, switch to aicpu execution.",
           context.GetNodeName());
    return aicpu_task_->ExecuteAsync(context, done_callback);
  }

  if (origin_fused_graph_exec_) {
    GELOGI("The task of Aicore Node[%s] will switch to origin fused graph execution.",
           context.GetNodeName());
    origin_fused_graph_exec_ = false;
    return fused_task_->ExecuteAsync(context, done_callback);
  }
  GELOGI("[%s] ExecuteAsync Start.", context.GetNodeName());
  for (auto it = tasks_.begin(); it != tasks_.end(); ++it) {
    // AtomicAddrClean has 2 tasks
    if ((tasks_.size() == 2U) && (it == tasks_.begin()) && (!(*(tasks_.rbegin()))->GetClearAtomic())) {
      // add for misra 6-6-3
    } else {
      RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] Start");
      GE_CHK_STATUS_RET_NOLOG((*it)->LaunchKernel(context.GetStream()));
      GE_CHK_STATUS_RET_NOLOG(CheckOverflow(context));
      GE_CHECK_NOTNULL(context.GetExecutionContext()->model);
      GELOGD("[DEBUG_TASK_INFO : Executor Task] %s/%s %s",
             context.GetExecutionContext()->model->GetModelName().c_str(),
             (*it)->GetName().empty() ? (*it)->GetLogName().c_str() : (*it)->GetName().c_str(),
             BuildTaskUtils::GetTaskInfo(context).c_str());
      // save profiling data
      GE_CHK_STATUS_RET(context.SaveProfilingTaskDescInfo(kTaskTypeAicore, (*it)->GetBlockDim(), (*it)->GetOpType()),
                        "[Save][Profiling] failed for node[%s]!", context.GetNodeName());
      (*it)->FillExtraOpInfo(context.MutableExtraOpInfo());
      RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeLaunchKernel] End");
    }
  }
  auto callback = done_callback;
  // only last task need update outputs shape
  const auto task = tasks_.back().get();
  if (task->GetUnknownShapeOpType() == DEPEND_SHAPE_RANGE) {
    callback = [task, done_callback, &context]() {
      Status callback_ret = SUCCESS;
      GELOGD("Node[%s] need update outputs shape.", context.GetNodeName());
      callback_ret = task->UpdateOutputsShape(context);
      if (done_callback != nullptr) {
        context.SetStatus(callback_ret);
        done_callback();
      }
    };
  }
  if (callback != nullptr) {
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeRegisterCallback] Start");
    GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(callback));
    RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeRegisterCallback] End");
  }

  GELOGD("[%s] ExecuteAsync End.", context.GetNodeName());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCoreNodeTaskExecuteAsync] End");
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateArgs(TaskContext &context) {
  GELOGI("[%s] AiCoreNodeTask UpdateArgs Start.", context.GetNodeName());
  if (aicpu_exec_) {
    GE_CHK_STATUS_RET_NOLOG(aicpu_task_->UpdateArgs(context));
  } else {
    for (auto it = tasks_.rbegin(); it != tasks_.rend(); ++it) {
      GE_CHK_STATUS_RET_NOLOG((*it)->UpdateArgs(context));
      // AtomicAddrClean has 2 tasks
      if ((tasks_.size() == 2U) && (it == tasks_.rbegin()) && (!(*it)->GetClearAtomic())) {
        break;
      }
    }
  }

  GELOGI("[%s] AiCoreNodeTask UpdateArgs End.", context.GetNodeName());
  return SUCCESS;
}

Status AiCoreNodeTask::SelectBin(TaskContext &task_context, const GraphExecutionContext *const ctx) {
  if (bin_selector_ == nullptr) {
    GELOGI("There is no bin_selector on aicore_node_task.");
    return SUCCESS;
  }

  std::vector<domi::TaskDef> task_defs;
  const auto cci = bin_selector_->SelectBin(task_context.GetNodeItem().node, ctx->ge_context, task_defs);
  if (cci == nullptr) {
    GELOGD("Can not find any support cache_item. Try turn to other executor.");
    if (aicpu_task_ != nullptr) {
      GELOGI("Node %s will switch to aicpu execution.", task_context.GetNodeName());
      aicpu_exec_ = true;
      return SUCCESS;
    } else if (fused_task_ != nullptr) {
      GELOGI("Node %s will switch to origin fused graph execution.", task_context.GetNodeName());
      origin_fused_graph_exec_ = true;
      return SUCCESS;
    } else {
      GELOGE(FAILED, "Node %s can not find cacheitem and any supported executor.",
             task_context.GetNodeName());
      return FAILED;
    }
  }
  if (cci->GetCacheItemId() == UINT64_MAX) {
    return SUCCESS;
  }
  GELOGD("Found cacheitem, id is: %lu, last_bin_id is: %lu.", cci->GetCacheItemId(), last_bin_);
  if (cci->GetCacheItemId() != last_bin_) {
    const auto ret = UpdateOpTasks(task_context, *ctx->model, *cci, task_defs);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to update op task on node %s.", task_context.GetNodeName());
      return FAILED;
    }
    last_bin_ = cci->GetCacheItemId();
  }
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateOpTasks(const TaskContext &task_context, const HybridModel &model,
                                     const NodeCompileCacheItem &cci, const std::vector<domi::TaskDef> &task_defs) {
  const auto &cur_node = task_context.GetNodeItem().node;
  if (tasks_.empty()) {
    GELOGD("Node %s is load without kernel, now load task on select bin.", cur_node->GetName().c_str());
    // build task on task_def
    GE_CHK_STATUS_RET(BuildAicoreNodeTask(cur_node, &task_defs, model, *this),
                      "Failed to load task when select bin of node %s.", cur_node->GetName().c_str());
    return SUCCESS;
  }

  for (size_t i = 0U; i < tasks_.size(); ++i) {
    const auto ret = tasks_[i]->UpdateBinHandle(task_context, cci);
    if (ret != SUCCESS) {
      GELOGD("[%s] Task update bin handle failed.", tasks_[i]->GetName().c_str());
      return ret;
    }
  }

  GELOGD("Node %s op_tasks is updated successfully by cci:%lu.", cur_node->GetName().c_str(), cci.GetCacheItemId());
  return SUCCESS;
}

Status AiCoreNodeTask::UpdateTilingData(TaskContext &context) {
  GELOGD("[%s] PrepareWithShape started.", context.GetNodeName());
  for (auto it = tasks_.rbegin(); it != tasks_.rend(); ++it) {
    const auto ret = (*it)->PrepareWithShape(context);
    if (ret != SUCCESS) {
      if (aicpu_task_ != nullptr) {
        aicpu_exec_ = true;
        return SUCCESS;
      } else {
        REPORT_INNER_ERR_MSG("EZ9999", "[Update][Tilingdata] Node[%s](%s) tiling failed!", context.GetNodeName(),
                             context.GetNodeItem().NodeType().c_str());
        GELOGE(ret, "[Update][Tilingdata] Node[%s](%s) tiling failed!", context.GetNodeName(),
               context.GetNodeItem().NodeType().c_str());
        return ret;
      }
    }

    // AtomicAddrClean has 2 tasks
    if ((tasks_.size() == 2U) && (it == tasks_.rbegin()) && (!(*it)->GetClearAtomic())) {
      break;
    }
  }
  GELOGD("[%s] Done PrepareWithShape successfully.", context.GetNodeName());
  return SUCCESS;
}

bool AiCoreNodeTask::IsSupportDynamicShape() {
  for (size_t i = 0U; i < tasks_.size(); ++i) {
    if (!tasks_[i]->IsDynamicShapeSupported()) {
      GELOGD("[%s] Task does not support dynamic shape.", tasks_[i]->GetName().c_str());
      return false;
    }
  }

  return true;
}

bool AiCoreNodeTask::IsSupportHostMemInputOpt() const {
  return true;
}

bool AiCoreNodeTask::IsArgsExtendedForHostMemInput() const {
  for (const auto &task : tasks_) {
    if ((task != nullptr) && (task->IsArgsExtendedForHostMemInput())) {
      return true;
    }
  }
  return false;
}

void AiCoreNodeTask::SetNeedHostMemOpt(const bool need_host_mem_opt) {
  for (auto &task : tasks_) {
    if (task != nullptr) {
      task->SetNeedHostMemOpt(need_host_mem_opt);
    }
  }
}

Status AiCoreNodeTask::CheckOverflow(TaskContext &context) const {
  const DumpProperties &dump_properties = context.GetDumpProperties();
  if (dump_properties.IsOpDebugOpen()) {
    GELOGD("Op %s is doing overflow check in hybrid engine", context.GetNodeName());
    std::string stream_synchronize_timeout;
    (void)ge::GetContext().GetOption(OPTION_EXEC_STREAM_SYNC_TIMEOUT, stream_synchronize_timeout);
    auto timeout = (!stream_synchronize_timeout.empty())
                       ? static_cast<int32_t>(std::strtol(stream_synchronize_timeout.c_str(), nullptr, 10))
                       : kDefaultTimeOut;
    const auto rt_ret = rtStreamSynchronizeWithTimeout(context.GetStream(), timeout);
    if ((rt_ret == ACL_ERROR_RT_AICORE_OVER_FLOW) || (rt_ret == ACL_ERROR_RT_AIVEC_OVER_FLOW)) {
      context.SetOverFlow(true);
      (void)rtsGetThreadLastTaskId(context.MutableTaskId());
      (void)aclrtStreamGetId(context.GetStream(), reinterpret_cast<int32_t*>(context.MutableStreamId()));
      GELOGW("Dynamic shape op %s is over flow", context.GetNodeName());
      return SUCCESS;
    } else if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      GELOGE(rt_ret, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, ret:%d.", rt_ret);
      return FAILED;
    } else if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Invoke][RtStreamSynchronize] failed, ret:%d.", rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronize failed, ret:%d.", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    } else {
      // add for misra rule 6-4-2
    }
    return SUCCESS;
  }
  GELOGD("Opdebug is not open in hybrid engine");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
