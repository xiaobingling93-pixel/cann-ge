/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/subgraph_executor.h"
#include "graph/ge_context.h"
#include "common/profiling/profiling_properties.h"
#include "common/profiling/profiling_manager.h"
#include "hybrid/executor/worker/execution_engine.h"
#include "hybrid/node_executor/node_executor.h"
#include "common/profiling_definitions.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"

namespace ge {
namespace hybrid {
namespace {
constexpr uint32_t kDefaultThreadNums = 6U;
constexpr uint32_t kReadyQueueSize = 16U;
}

SubgraphExecutor::SubgraphExecutor(const GraphItem *const graph_item, GraphExecutionContext *const context,
                                   const bool force_infer_shape, ThreadPool *pre_run_pool)
    : SubgraphContext(graph_item, context), ShapeInferenceEngine(context, force_infer_shape),
      graph_item_(graph_item),
      context_(context),
      pre_run_pool_(pre_run_pool),
      own_thread_pool_(false),
      ready_queue_(kReadyQueueSize) {
  ShapeInferenceEngine::Config(this);
  shape_inference_engine_  = this;
  subgraph_context_ = this;
}

SubgraphExecutor::~SubgraphExecutor() {
  if (own_thread_pool_ && (pre_run_pool_ != nullptr)) {
    delete pre_run_pool_;
    pre_run_pool_ = nullptr;
  }
  GELOGD("[%s] SubgraphExecutor destroyed.", graph_item_ != nullptr ? graph_item_->GetName().c_str() : "");
}

Status SubgraphExecutor::Init() {
  if (pre_run_pool_ == nullptr) {
    pre_run_pool_ = new (std::nothrow) ThreadPool("ge_prepare", kDefaultThreadNums, false);
    GE_CHECK_NOTNULL(pre_run_pool_);
    own_thread_pool_ = true;
  }
  GE_CHK_STATUS_RET(subgraph_context_->Init(),
                    "[Init][SubgraphContext][%s] Failed to init subgraph context.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::InitInputsForKnownShape(const std::vector<TensorValue> &inputs) {
  auto &input_index_mapping = graph_item_->GetInputIndexMapping();
  for (size_t i = 0U; i < input_index_mapping.size(); ++i) {
    auto &parent_input_index = input_index_mapping[i];
    if (static_cast<size_t>(parent_input_index) >= inputs.size()) {
      GELOGE(INTERNAL_ERROR, "[Check][Size][%s] Number of inputs [%zu] is not sufficient for subgraph"
             "which needs at lease [%d] inputs", graph_item_->GetName().c_str(), inputs.size(),
             parent_input_index + 1);
      REPORT_INNER_ERR_MSG("E19999", "[%s] Number of inputs [%zu] is not sufficient for subgraph"
                         "which needs at lease [%d] inputs",
                         graph_item_->GetName().c_str(), inputs.size(), parent_input_index + 1);
      return INTERNAL_ERROR;
    }

    auto &input_tensor = inputs[static_cast<size_t>(parent_input_index)];
    (void)subgraph_context_->SetInput(static_cast<int32_t>(i), input_tensor);
    GELOGD("[%s] Set input tensor[%zu] with inputs with index = %d, tensor = %s",
           graph_item_->GetName().c_str(),
           i,
           parent_input_index,
           input_tensor.DebugString().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(const std::vector<TensorValue> &inputs,
                                      const std::vector<ConstGeTensorDescPtr> &input_desc,
                                      const std::vector<TensorValue> &outputs) {
  GELOGD("[%s] is dynamic = %s", graph_item_->GetName().c_str(), graph_item_->IsDynamic() ? "true" : "false");
  GE_CHK_STATUS_RET(InitInputs(inputs, input_desc), "[Invoke][Init]failed for [%s].", graph_item_->GetName().c_str());
  if (!outputs.empty()) {
    GE_CHK_STATUS_RET(EnableOutputZeroCopy(outputs),
                      "[Invoke][EnableOutputZeroCopy] Failed by user provided outputs.");
  }

  GE_CHK_STATUS_RET_NOLOG(PreRunSubgraph());
  if (!graph_item_->IsDynamic()) {
    return ExecuteAsyncForKnownShape(inputs);
  }

  HYBRID_CHK_STATUS_RET(ScheduleTasks(), "[Call][ScheduleTasks] [%s] Failed to execute tasks.",
                        graph_item_->GetName().c_str());
  GELOGD("[%s] Done executing subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(const std::vector<TensorValue> &inputs,
                                      const std::vector<ConstGeTensorDescPtr> &input_desc) {
  return ExecuteAsync(inputs, input_desc, {});
}

Status SubgraphExecutor::ExecuteAsyncForKnownShape(const std::vector<TensorValue> &inputs) {
  (void)inputs;
  GELOGD("[%s] subgraph is not dynamic.", graph_item_->GetName().c_str());
  if (graph_item_->GetAllNodes().size() != 1U) {
    REPORT_INNER_ERR_MSG("E19999", "[%s] Invalid known shape subgraph. node size = %zu", graph_item_->GetName().c_str(),
                         graph_item_->GetAllNodes().size());
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] Invalid known shape subgraph. node size = %zu",
           graph_item_->GetName().c_str(), graph_item_->GetAllNodes().size());
    return INTERNAL_ERROR;
  }

  const auto node_item = graph_item_->GetAllNodes()[0U];
  GE_CHECK_NOTNULL(node_item);
  auto node_state = subgraph_context_->GetNodeState(node_item);
  GE_CHECK_NOTNULL(node_state);
  node_state->SetKernelTask(node_item->kernel_task);

  std::function<void()> callback;
  GE_CHK_STATUS_RET_NOLOG(InitCallback(node_state, callback));
  HYBRID_CHK_STATUS_RET(ExecutionEngine::ExecuteAsync(*node_state, node_state->GetTaskContext(), *context_, callback),
                        "[Call][ExecuteAsync] [%s] Failed to execute node [%s(%s)] for known subgraph.",
                        graph_item_->GetName().c_str(), node_state->GetName().c_str(), node_state->GetType().c_str());

  GELOGD("[%s] Done execute non-dynamic subgraph successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::ExecuteAsync(const TaskContext &task_context) {
  std::vector<TensorValue> inputs;
  std::vector<ConstGeTensorDescPtr> input_desc;
  for (int32_t i = 0; i < task_context.NumInputs(); ++i) {
    const auto tensor = task_context.GetInput(i);
    GE_CHECK_NOTNULL(tensor);
    inputs.emplace_back(*tensor);
    input_desc.emplace_back(task_context.GetInputDesc(i));
  }

  GE_CHK_STATUS_RET(ExecuteAsync(inputs, input_desc), "[Invoke][ExecuteAsync] failed for [%s].",
                    graph_item_->GetName().c_str());

  GE_CHK_STATUS_RET(SetOutputsToParentNode(task_context),
                    "[Invoke][SetOutputsToParentNode][%s] Failed to set output shapes to parent node.",
                    graph_item_->GetName().c_str());
  return SUCCESS;
}

BlockingQueue<const NodeItem *> &SubgraphExecutor::GetPrepareQueue(const int32_t group) {
  const std::lock_guard<std::mutex> lk(mu_);
  return prepare_queues_[group];
}

Status SubgraphExecutor::NodeEnqueue(NodeState *const node_state) {
  if (!ready_queue_.Push(node_state)) {
    if (context_->is_eos_) {
      GELOGD("Got end of sequence");
      return SUCCESS;
    }
    GELOGE(INTERNAL_ERROR, "[Check][State][%s] Error occurs while launching tasks. quit from preparing nodes.",
           graph_item_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s] Error occurs while launching tasks. quit from preparing nodes.",
                         graph_item_->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("[%s] Push node [%s] to queue.", graph_item_->GetName().c_str(), node_state->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::PrepareNode(const NodeItem &node_item) {
  GELOGD("[%s] Start to prepare node [%s].", graph_item_->GetName().c_str(), node_item.NodeName().c_str());
  // for while op
  if (IsForceInferShape() && (!node_item.is_dynamic)) {
    GELOGD("[%s] Force infer shape is set, updating node to dynamic.", node_item.NodeName().c_str());
    auto &mutable_node_item = const_cast<NodeItem &>(node_item);
    mutable_node_item.SetToDynamic();
  }

  const auto p_node_state = subgraph_context_->GetNodeState(&node_item);
  GE_CHECK_NOTNULL(p_node_state);
  if (p_node_state->MaySkipSchedule()) {
    return SUCCESS;
  }
  if (node_item.node_type == NETOUTPUT) {
    GE_CHK_STATUS_RET_NOLOG(NodeEnqueue(p_node_state));
    return AfterPrepared(p_node_state);
  }

  p_node_state->SetKernelTask(node_item.kernel_task);
  const auto &task = p_node_state->GetKernelTask();
  if (task == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Get][KernelTask] failed for[%s(%s)], NodeTask is null.", p_node_state->GetName().c_str(),
           p_node_state->GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "GetKernelTask failed for %s(%s), nodetask is null.", p_node_state->GetName().c_str(),
                      p_node_state->GetType().c_str());
    return INTERNAL_ERROR;
  }

  // only do shape inference and compilation for nodes with dynamic shapes or ffts sub node.
  if (node_item.is_dynamic || node_item.is_ffts_sub_node_) {
    GE_CHK_STATUS_RET_NOLOG(shape_inference_engine_->InferShape(*p_node_state));
    if (task->IsNeedTilling()) {
      PROFILING_START(p_node_state->GetProfilingIndex(), profiling::kCommitTilingTask);
      const auto prepare_func = [this, p_node_state](const error_message::ErrorManagerContext &error_context) -> Status {
        error_message::SetErrMgrContext(error_context);
        GetContext().SetSessionId(context_->session_id);
        GetContext().SetContextId(context_->context_id);
        GE_CHK_STATUS_RET_NOLOG(PrepareForExecution(context_, *p_node_state));
        return AfterPrepared(p_node_state);
      };
      auto prepare_future = pre_run_pool_->commit(prepare_func, error_message::GetErrMgrContext());
      PROFILING_END(p_node_state->GetProfilingIndex(), profiling::kCommitTilingTask);

      p_node_state->SetPrepareFuture(std::move(prepare_future));
      return NodeEnqueue(p_node_state);
    }
  } else {
    GELOGD("[%s] Skipping shape inference and compilation for node with static shape.",
           node_item.NodeName().c_str());
  }
  GE_CHK_STATUS_RET_NOLOG(NodeEnqueue(p_node_state));
  return AfterPrepared(p_node_state);
}

Status SubgraphExecutor::ReportTimeoutInfo(const NodeState *const node_state) const {
  std::stringstream ss;
  if (node_state != nullptr) {
    const auto &node_item = node_state->GetNodeItem();
    ss << "Node: ";
    ss << "id = " << node_item.node_id;
    ss << ", name = [" << node_item.NodeName();
    ss << "], type = " << node_item.NodeType();
    ss << ", num_iteration = " << node_state->GetIterationCount();
    ss << ", num_inputs = " << node_item.num_inputs;
    ss << ", num_data_inputs = " << node_item.data_recv_.size();
    ss << ", num_scheduled_data = " << node_state->GetDataScheduledNum();
    if (node_state->GetDataScheduledNum() < node_item.data_recv_.size()) {
      ss << ", missed_data_inputs = " << node_item.data_recv_.size() - node_state->GetDataScheduledNum();
    }
    ss << ", num_ctrl_inputs = " << node_item.ctrl_recv_.size();
    ss << ", num_scheduled_ctrl = " << node_state->GetCtrlScheduledNum();
    if (node_state->GetCtrlScheduledNum() < node_item.ctrl_recv_.size()) {
      ss << ", missed_ctrl_inputs = " << node_item.ctrl_recv_.size() - node_state->GetCtrlScheduledNum();
    }
    ss << ", num_enter_data_inputs = " << node_item.enter_data_.size();
    ss << ", num_enter_ctrl_inputs = " << node_item.enter_ctrl_.size();
    ss << ", num_root_data_inputs = " << node_item.root_data_.size();
    ss << ", num_root_ctrl_inputs = " << node_item.root_ctrl_.size();
  }

  REPORT_INNER_ERR_MSG("E19999", "[Call][Pop]Await timeout, detail: %s.", ss.str().c_str());
  GELOGE(FAILED, "[Call][Pop]Await timeout, detail: %s.", ss.str().c_str());
  return FAILED;
}

Status SubgraphExecutor::PrepareNodes(const int32_t group) {
  GE_CHECK_NOTNULL(pre_run_pool_);
  const size_t node_size = graph_item_->GetNodeSize(group);
  GELOGD("[%s] Start to prepare nodes. group = %d, size = %zu", graph_item_->GetName().c_str(), group, node_size);
  if (!graph_item_->HasCtrlFlowOp()) {
    for (const auto &node_item : graph_item_->GetAllNodes(group)) {
      RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] Start");
      GE_CHK_STATUS_RET(PrepareNode(*node_item), "[Prepare][Node] [%s(%s)] failed to prepare task.",
                        node_item->NodeName().c_str(), node_item->NodeType().c_str());
      RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] End");
    }

    GELOGD("[%s] Done preparing nodes successfully.", graph_item_->GetName().c_str());
    return SUCCESS;
  }

  // Initialize the ready queue
  size_t node_count = 0U;
  bool node_complete = false;
  for (const auto &node_item : graph_item_->GetRootNodes(group)) {
    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] Start");
    GE_CHK_STATUS_RET(PrepareNode(*node_item), "[Prepare][Node] [%s(%s)] failed to prepare task.",
                      node_item->NodeName().c_str(), node_item->NodeType().c_str());
    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] End");
    node_complete = node_item->NodeType() == NETOUTPUT;
    node_count++;
  }

  GELOGD("[%s] Done preparing root nodes.", graph_item_->GetName().c_str());
  NodeState *tmp_state = nullptr;
  BlockingQueue<const NodeItem *> &prepare_queue = GetPrepareQueue(group);
  while (((group != -1) && (node_count < node_size)) || ((group == -1) && (!node_complete))) {
    const NodeItem *node_item = nullptr;
    bool is_stuck = false;
    if (!prepare_queue.Pop(node_item, is_stuck)) {
      if (is_stuck) {
        return ReportTimeoutInfo(tmp_state);
      }
      if (context_->is_eos_) {
        GELOGD("[%s] Got end of sequence.", graph_item_->GetName().c_str());
        GELOGD("[%s] Done preparing nodes successfully.", graph_item_->GetName().c_str());
        return SUCCESS;
      }
      GELOGW("[Pop][Node] failed, graph:[%s] Context status: %u.",
             graph_item_->GetName().c_str(), context_->GetStatus());
      return SUCCESS;
    }

    if (node_item == nullptr) {
      GELOGD("[%s] Got EOF from queue.", graph_item_->GetName().c_str());
      break;
    }

    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] Start");
    GE_CHK_STATUS_RET(PrepareNode(*node_item),
                      "[Prepare][Node] [%s(%s)] failed to prepare task.",
                      node_item->NodeName().c_str(), node_item->NodeType().c_str());
    RECORD_EXECUTION_EVENT(context_, node_item->NodeName().c_str(), "[PrepareNode] End");
    node_complete = node_item->NodeType() == NETOUTPUT;
    node_count++;
    tmp_state = subgraph_context_->GetNodeState(node_item);
  }

  GELOGD("[%s] Done preparing nodes successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::NodeScheduled(NodeState *const node_state) {
  GELOGD("Graph[%s] After [%s] scheduled, data size: %zu, ctrl size: %zu, switch index: %d, merge index: %d",
         graph_item_->GetName().c_str(), node_state->GetName().c_str(),
         node_state->GetNodeItem().data_send_.size(), node_state->GetNodeItem().ctrl_send_.size(),
         node_state->GetSwitchIndex(), node_state->GetMergeIndex());

  GE_CHECK_NOTNULL(pre_run_pool_);
  const auto prepare_func = [this, node_state](const struct error_message::ErrorManagerContext &error_context) -> Status {
    error_message::SetErrMgrContext(error_context);
    RECORD_CALLBACK_EVENT(context_, node_state->GetName().c_str(), "[NodeScheduled] Start");
    const std::function<void(const NodeItem *)> callback = [this, &node_state](const NodeItem *const node_item) {
      const auto &node_name = node_item->node_name;
      const auto group = subgraph_context_->ScheduleGroup(node_item);
      GELOGI("After [%s] scheduled, [%s] is ready for prepare.", node_state->GetName().c_str(), node_name.c_str());
      BlockingQueue<const NodeItem *> &prepare_queue = GetPrepareQueue(group);
      if (!prepare_queue.Push(node_item)) {
        if (!context_->is_eos_) {
          GELOGE(INTERNAL_ERROR, "[Check][State][%s] error occurs when push to queue.", graph_item_->GetName().c_str());
          REPORT_INNER_ERR_MSG("E19999", "[%s] error occurs when push to queue.", graph_item_->GetName().c_str());
        }
      }
    };

    GE_CHK_STATUS_RET_NOLOG(node_state->NodeScheduled(callback));
    RECORD_CALLBACK_EVENT(context_, node_state->GetName().c_str(), "[NodeScheduled] End");
    return SUCCESS;
  };
  auto future = pre_run_pool_->commit(prepare_func, error_message::GetErrMgrContext());

  node_state->SetScheduleFuture(std::move(future));

  if (context_->is_eos_) {
    GELOGD("[%s] Got end of sequence", graph_item_->GetName().c_str());
  }

  return SUCCESS;
}

Status SubgraphExecutor::AfterPrepared(NodeState *const node_state) {
  if (!graph_item_->HasCtrlFlowOp()) {
    return SUCCESS;
  }
  if (node_state->IsShapeDependence()) {
    return SUCCESS;
  }

  // Not control flow node, propagate state.
  return NodeScheduled(node_state);
}

void SubgraphExecutor::AfterExecuted(NodeState *const node_state) {
  PROFILING_SCOPE_CONST(node_state->GetProfilingIndex(), profiling::kAfterExecuted);
  if (!node_state->IsShapeDependence()) {
    return;
  }

  // For control flow node, propagate state.
  const auto error = NodeScheduled(node_state);
  if (error != SUCCESS) {
    const auto task_context = node_state->GetTaskContext();
    task_context->OnError(error);
  }
}

void SubgraphExecutor::OnNodeDone(NodeState *const node_state) {
  PROFILING_START(node_state->GetProfilingIndex(), profiling::kOnNodeDoneCallback);
  const auto task_context = node_state->GetTaskContext();
  NodeDoneCallback cb(context_, task_context);
  const auto error = cb.OnNodeDone();
  PROFILING_END(node_state->GetProfilingIndex(), profiling::kOnNodeDoneCallback);
  if (error != SUCCESS) {
    task_context->OnError(error);
  }

  if (node_state->IsShapeDependence() && graph_item_->HasCtrlFlowOp()) {
    AfterExecuted(node_state);
  }
}

bool SubgraphExecutor::HasOnNodeDoneCallback(const NodeState &node_state) const {
  const auto task_context = node_state.GetTaskContext();
  if (task_context == nullptr) {
    GELOGW("The task context is null for node %s", node_state.GetName().c_str());
    return false;
  }
  if (task_context->NeedCallback() || (node_state.IsShapeDependence() && graph_item_->HasCtrlFlowOp())) {
    GELOGD("Node %s has OnNodeDoneCallBack func", node_state.GetName().c_str());
    return true;
  }
  return false;
}

Status SubgraphExecutor::InitCallback(NodeState *const node_state, std::function<void()> &callback,
                                      const std::shared_ptr<ScopeGuard> tensor_guard) {
  const auto task_context = node_state->GetTaskContext();
  GE_CHECK_NOTNULL(task_context);
  if (task_context->NeedCallback()) {
    callback = [this, node_state, tensor_guard]() {
      (void)tensor_guard;
      OnNodeDone(node_state);
    };
  } else if (node_state->IsShapeDependence() && graph_item_->HasCtrlFlowOp()) {
    callback = [this, node_state, tensor_guard]() {
      (void)tensor_guard;
      AfterExecuted(node_state);
    };
  } else {
    // add for misra rule 6-4-2
  }

  return SUCCESS;
}

Status SubgraphExecutor::PrepareForExecution(const GraphExecutionContext *const ctx, NodeState &node_state) const {
  const auto &task = node_state.GetKernelTask(); // checked not null outside
  GE_CHK_RT_RET(rtCtxSetCurrent(ctx->rt_context));
  auto &node_item = node_state.GetNodeItem();
  if (node_item.IsNoOp()) {
    GELOGD("[%s] Skipping tiling and selectbin for op with empty outputs.", node_state.GetName().c_str());
    return SUCCESS;
  }
  TaskContext &task_context = *node_state.GetTaskContext();
  PROFILING_START(node_state.GetProfilingIndex(), profiling::kSelectBin);
  GE_CHK_STATUS_RET_NOLOG(task->SelectBin(task_context, ctx));
  PROFILING_END(node_state.GetProfilingIndex(), profiling::kSelectBin);

  PROFILING_START(node_state.GetProfilingIndex(), profiling::kTiling);
  GE_CHK_STATUS_RET_NOLOG(task->UpdateTilingData(*node_state.GetTaskContext()));
  PROFILING_END(node_state.GetProfilingIndex(), profiling::kTiling);
  return SUCCESS;
}

Status SubgraphExecutor::LaunchTasks() {
  while (true) {
    NodeState *node_state = nullptr;
    if (!ready_queue_.Pop(node_state)) {
      GELOGE(INTERNAL_ERROR, "[Invoke][Pop] failed for [%s].", graph_item_->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "invoke pop failed for %s.", graph_item_->GetName().c_str());
      return INTERNAL_ERROR;
    }

    if (node_state == nullptr) {
      GELOGD("[%s] Got EOF from queue.", graph_item_->GetName().c_str());
      return SUCCESS;
    }

    if (node_state->GetType() == NETOUTPUT) {
      // Wait for all inputs become valid
      // after PrepareNodes returned. all output tensors and shapes are valid
      GE_CHK_STATUS_RET_NOLOG(node_state->AwaitDependShapes(*context_));
      GE_CHK_STATUS_RET_NOLOG(node_state->AwaitInputTensors(*context_));
      GELOGD("[%s] Done executing node successfully.", node_state->GetName().c_str());
      continue;
    }
    PROFILING_START(node_state->GetProfilingIndex(), profiling::kWaitForPrepareDone);
    GE_CHK_STATUS_RET_NOLOG(node_state->WaitForPrepareDone());
    PROFILING_END(node_state->GetProfilingIndex(), profiling::kWaitForPrepareDone);

    GELOGD("[%s] Start to execute.", node_state->GetName().c_str());
    const auto shared_task_context = node_state->GetTaskContext();
    GE_CHECK_NOTNULL(shared_task_context);
    shared_task_context->SetForceInferShape(IsForceInferShape());

    std::function<void()> tensor_callback = [shared_task_context, node_state, this]() {
      if (graph_item_->IsFftsPlusGraph()) {
        // ffts no need release all mem after launch
      } else if (HasOnNodeDoneCallback(*node_state)) {
        // release output must be after OnNodeDone and PropagateOutputs
        shared_task_context->ReleaseAllOutput();
      } else {
        shared_task_context->ReleaseAllMem();
      }
    };
    const auto tensor_guard = MakeShared<ScopeGuard>(tensor_callback);
    GE_CHECK_NOTNULL(tensor_guard);
    std::function<void()> callback;
    GE_CHK_STATUS_RET_NOLOG(InitCallback(node_state, callback, tensor_guard));
    HYBRID_CHK_STATUS_RET(ExecutionEngine::ExecuteAsync(*node_state, shared_task_context, *context_, callback),
                          "[Invoke][ExecuteAsync] failed for [%s(%s)].", node_state->GetName().c_str(),
                          node_state->GetType().c_str());
    GELOGD("[%s] Done executing node successfully.", node_state->GetName().c_str());
  }
}

Status SubgraphExecutor::ScheduleTasks(const int32_t group) {
  GELOGD("[%s] Start to schedule prepare workers.", graph_item_->GetName().c_str());
  subgraph_context_->SetGroup(group);
  GE_CHECK_NOTNULL(pre_run_pool_);
  PROFILING_START(-1, profiling::kCommitInferShapeTask);
  const auto prepare_func = [this, group](const error_message::ErrorManagerContext &error_context) -> Status {
    error_message::SetErrMgrContext(error_context);
    GetContext().SetSessionId(context_->session_id);
    GetContext().SetContextId(context_->context_id);
    const auto ret = PrepareNodes(group);
    ready_queue_.Push(nullptr);
    return ret;
  };
  auto prepare_future = pre_run_pool_->commit(prepare_func, error_message::GetErrMgrContext());
  PROFILING_END(-1, profiling::kCommitInferShapeTask);

  GELOGD("[%s] Start to execute subgraph.", graph_item_->GetName().c_str());
  const auto ret = LaunchTasks();
  if (ret != SUCCESS) {
    subgraph_context_->OnError(ret);
    context_->SetErrorCode(ret);
    ready_queue_.Stop();
    for (auto &item : prepare_queues_) {
      item.second.Stop();
    }
    prepare_future.wait();
    return ret;
  }

  GE_CHK_STATUS_RET(prepare_future.get(), "[Invoke][get] [%s] Error occurred in task preparation.",
                    graph_item_->GetName().c_str());

  GELOGD("[%s] Done launching all tasks successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::GetContextOutputs(std::vector<TensorValue> &outputs) {
  return subgraph_context_->GetOutputs(outputs);
}

Status SubgraphExecutor::GetOutputs(std::vector<TensorValue> &outputs, std::vector<ConstGeTensorDescPtr> &output_desc) {
  GE_CHK_STATUS_RET(GetContextOutputs(outputs), "[Invoke][GetOutputs] failed for [%s].",
                    graph_item_->GetName().c_str());

  // copy output data from op to designated position
  GE_CHK_STATUS_RET(graph_item_->GetOutputDescList(output_desc),
                    "[Invoke][GetOutputDescList][%s] Failed to get output tensor desc.",
                    graph_item_->GetName().c_str());
  if (outputs.size() != output_desc.size()) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]Number of outputs(%zu) mismatch number of output_desc(%zu).",
           outputs.size(), output_desc.size());
    REPORT_INNER_ERR_MSG("E19999", "Number of outputs(%zu) mismatch number of output_desc(%zu).",
                       outputs.size(), output_desc.size());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

void SubgraphExecutor::Reset() {
  PROFILING_SCOPE(-1, profiling::kResetSubgraphExecutor);
  subgraph_context_->Reset();
  ready_queue_.Clear();
  for (auto &item : prepare_queues_) {
    item.second.Clear();
  }
}

Status SubgraphExecutor::Synchronize() const {
  PROFILING_SCOPE(-1, profiling::kRtStreamSync);
  GELOGD("[%s] Synchronize start.", graph_item_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(context_->Synchronize(context_->stream));
  GELOGD("[%s] Done synchronizing successfully.", graph_item_->GetName().c_str());
  return SUCCESS;
}

Status SubgraphExecutor::SetOutputsToParentNode(const TaskContext &task_context) const {
  // get output tensors and tensor desc list
  std::vector<TensorValue> outputs;
  std::vector<ConstGeTensorDescPtr> output_desc_list;
  GE_CHK_STATUS_RET(subgraph_context_->GetOutputs(outputs), "[Invoke][GetOutputs][%s] Failed to get output tensors.",
                    graph_item_->GetName().c_str());
  GE_CHK_STATUS_RET(graph_item_->GetOutputDescList(output_desc_list),
                    "[Invoke][GetOutputDescList][%s] Failed to get output tensor desc.",
                    graph_item_->GetName().c_str());

  if (outputs.size() != output_desc_list.size()) {
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] num of output tensors = %zu, num of output tensor desc = %zu not equal",
           graph_item_->GetName().c_str(), outputs.size(), output_desc_list.size());
    REPORT_INNER_ERR_MSG("E19999", "%s num of output tensors = %zu, num of output tensor desc = %zu not equal",
                         graph_item_->GetName().c_str(), outputs.size(), output_desc_list.size());
    return INTERNAL_ERROR;
  }

  // mapping to parent task context
  for (size_t i = 0U; i < outputs.size(); ++i) {
    const int32_t parent_output_index = graph_item_->GetParentOutputIndex(i);
    GE_CHECK_GE(parent_output_index, 0);
    // update tensor
    GELOGD("[%s] Updating output[%zu] to parent output[%d]",
           graph_item_->GetName().c_str(),
           i,
           parent_output_index);

    GELOGD("[%s] Updating output tensor, index = %d, tensor = %s",
           graph_item_->GetName().c_str(),
           parent_output_index,
           outputs[i].DebugString().c_str());
    GE_CHK_STATUS_RET(task_context.SetOutput(parent_output_index, outputs[i]));

    // updating shapes. dynamic format/dtype is not supported.
    // It should be noted that even the subgraph is of known shape, it is also necessary to update parent output desc,
    // for instance, IfOp may have two known-shaped subgraphs of different output shapes
    const auto &output_desc = output_desc_list[i];
    const auto parent_output_desc = task_context.MutableOutputDesc(parent_output_index);
    GE_CHECK_NOTNULL(parent_output_desc);
    GELOGD("[%s] Updating output shape[%d] from [%s] to [%s]",
           graph_item_->GetName().c_str(),
           parent_output_index,
           parent_output_desc->MutableShape().ToString().c_str(),
           output_desc->GetShape().ToString().c_str());
    parent_output_desc->SetShape(output_desc->GetShape());

    GELOGD("[%s] Updating output original shape[%d] from [%s] to [%s]",
           graph_item_->GetName().c_str(),
           parent_output_index,
           parent_output_desc->GetOriginShape().ToString().c_str(),
           output_desc->GetOriginShape().ToString().c_str());
    parent_output_desc->SetOriginShape(output_desc->GetOriginShape());
    int64_t out_size = 0;
    (void)TensorUtils::GetSize(*output_desc, out_size);
    (void)TensorUtils::SetSize(*parent_output_desc, out_size);
  }

  return SUCCESS;
}

Status SubgraphExecutor::EnableOutputZeroCopy(const std::vector<TensorValue> &outputs) {
  GELOGD("To enable zero copy, output number = %zu", outputs.size());
  const auto &output_edges = graph_item_->GetOutputEdges();
  // Op -> MetOutput, set the output tensor of Op that output to the NetOutput node
  if (outputs.size() != output_edges.size()) {
    GELOGE(PARAM_INVALID, "[Check][Size]Output number mismatches, expect = %zu, but given = %zu",
           output_edges.size(), outputs.size());
    REPORT_INNER_ERR_MSG("E19999", "Output number mismatches, expect = %zu, but given = %zu", output_edges.size(),
                         outputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0U; i < outputs.size(); ++i) {
    auto &output_tensor = outputs[i];
    if (output_tensor.GetData() == nullptr) {
      continue;
    }
    auto &output_node = output_edges[i].first;
    GE_CHECK_NOTNULL(output_node);
    const int32_t output_idx = output_edges[i].second;
    GELOGD("[%s] Set output tensor[%zu] to [%s]'s output[%d], tensor = %s",
           graph_item_->GetName().c_str(),
           i,
           output_node->NodeName().c_str(),
           output_idx,
           output_tensor.DebugString().c_str());
    const auto node_state = subgraph_context_->GetNodeState(output_node);
    GE_CHECK_NOTNULL(node_state);
    node_state->SetUserAllocated(true);

    GE_CHK_STATUS_RET(subgraph_context_->SetOutput(*output_node, output_idx, output_tensor),
                      "[Invoke][SetOutput][%s] Failed to set input tensor[%zu]",
                      graph_item_->GetName().c_str(), i);
  }

  GELOGD("Done enabling zero copy for outputs successfully.");
  return SUCCESS;
}

Status SubgraphExecutor::PartialExecuteAsync(const int32_t task_group) {
  return ScheduleTasks(task_group);
}

Status SubgraphExecutor::InitInputs(const std::vector<TensorValue> &inputs,
                                    const std::vector<ConstGeTensorDescPtr> &input_desc) {
  PROFILING_SCOPE(-1, profiling::kUpdateShape);
  if (graph_item_->IsDynamic()) {
    GE_CHK_STATUS_RET(shape_inference_engine_->InitInferShapes(graph_item_, inputs, input_desc),
                      "[Call][InitInputsForUnknownShape][%s] Failed to set inputs.",
                      graph_item_->GetName().c_str());
  } else {
    GE_CHK_STATUS_RET(InitInputsForKnownShape(inputs),
                      "[Invoke][InitInputsForKnownShape][%s] Failed to init executor for known shape subgraph.",
                      graph_item_->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
