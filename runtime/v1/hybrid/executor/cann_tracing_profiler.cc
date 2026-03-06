/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_tracing_profiler.h"
#include <unordered_map>
#include "runtime/model_v2_executor.h"
#include "core/builder/node_types.h"
#include "graph/def_types.h"
#include "graph/debug/ge_attr_define.h"
#include "lowering/placement/placed_lowering_result.h"
#include "lowering/pass_changed_kernels_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "subscriber/subscriber_utils.h"
#include "engine/aicore/launch_kernel/ai_core_launch_kernel.h"

namespace gert {
namespace {
constexpr uint16_t kFpBeginLogId = 2U;
constexpr uint16_t kBpEndLogId = 3U;

size_t CalcArgIndex(size_t total_num, ExecuteArgIndex arg_index) {
  size_t tensor_num = total_num - static_cast<size_t>(ExecuteArgIndex::kNum);
  return tensor_num + static_cast<size_t>(static_cast<int64_t>(arg_index) * -1 - 1);
}
}

void CannTracingProfiler::Init() {
  const auto execution_data =
      static_cast<const ExecutionData *>(extend_info_.executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  if (execution_data == nullptr) {
    GELOGW("Execution data is empty, do not init tracing profiler.");
    return;
  }
  GELOGD("Training trace init, execute node num = %zu", execution_data->base_ed.node_num);
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto node = execution_data->base_ed.nodes[i];
    auto kernel_context = reinterpret_cast<KernelContext *>(&node->context);
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
    if (kernel_extend_info == nullptr) {
      GELOGW("Kernel extend info is nullptr.");
      return;
    }
    if (ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_extend_info->GetKernelType())) {
      const auto compute_node_info = static_cast<const ComputeNodeInfo *>(kernel_context->GetComputeNodeExtend());
      if (compute_node_info == nullptr) {
        continue;
      }
      const auto node_name = compute_node_info->GetNodeName();
      const auto iter = extend_info_.node_names_to_attrs.find(node_name);
      if (iter != extend_info_.node_names_to_attrs.cend()) {
        node_ids_to_attrs_[node->node_id] = iter->second;
      }
    }
  }
}

CannTracingProfiler::CannTracingProfiler(SubscriberExtendInfo extend_info)
    : BaseExecutorProfiler(), extend_info_(std::move(extend_info)) {
  if ((extend_info_.executor != nullptr)) {
    Init();
  }
}

ge::Status CannTracingProfiler::ReportTraceInfo(uint16_t tag_id, const Node *node) {
  if (rt_streams_ == nullptr) {
    const auto execution_data = static_cast<const ExecutionData *>(
        extend_info_.executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
    GE_ASSERT_NOTNULL(execution_data);
    const auto stream_idx = CalcArgIndex(execution_data->base_ed.input_num, ExecuteArgIndex::kStream);
    rt_streams_ =
        reinterpret_cast<Chain *>(execution_data->base_ed.input_values[stream_idx])->GetValue<ContinuousVector *>();
  }
  rtStream_t cur_stream = extend_info_.stream;
  int64_t logic_stream_id = 0;
  if (node_ids_to_attrs_.find(node->node_id) != node_ids_to_attrs_.end()) {
    auto trace_info = node_ids_to_attrs_[node->node_id];
    logic_stream_id = trace_info.logic_stream_id;
    GE_ASSERT_NOTNULL(rt_streams_);
    cur_stream =
        *(reinterpret_cast<rtStream_t *>(rt_streams_->MutableData()) + static_cast<size_t>(logic_stream_id));
  }
  GE_ASSERT_RT_OK(rtProfilerTraceEx(iteration_num_, static_cast<uint64_t>(extend_info_.model_id),
      tag_id, cur_stream));
  GELOGD(
      "Profiling Step Info TraceTask execute async success, index_id = %llu, model_id = %u, tag_id = %u, "
      "logic_stream_id= %lld",
      iteration_num_, extend_info_.model_id, tag_id, logic_stream_id);
  return ge::SUCCESS;
}

ge::Status CannTracingProfiler::ReportStartTraceInfo(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  if (node_ids_to_attrs_.find(node->node_id) != node_ids_to_attrs_.end()) {
    auto trace_info = node_ids_to_attrs_[node->node_id];
    if (trace_info.is_fp) {
      GE_ASSERT_SUCCESS(ReportTraceInfo(kFpBeginLogId, node));
    }

    if (trace_info.start_log_id > 0LL) { // all reduce and get next
      GE_ASSERT_SUCCESS(ReportTraceInfo(static_cast<uint16_t>(trace_info.start_log_id), node));
    }
  }

  return ge::SUCCESS;
}

ge::Status CannTracingProfiler::ReportEndTraceInfo(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  if (node_ids_to_attrs_.find(node->node_id) != node_ids_to_attrs_.end()) {
    auto trace_info = node_ids_to_attrs_[node->node_id];
    // 编译时的逻辑是只要是all reduce都会打上bp的标签
    if (trace_info.is_bp) {
      if (trace_info.start_log_id > 0LL) { // all reduce end
        GE_ASSERT_SUCCESS(ReportTraceInfo(static_cast<uint16_t>(trace_info.start_log_id + 1LL), node));
      } else { // bp
        GE_ASSERT_SUCCESS(ReportTraceInfo(kBpEndLogId, node));
      }
    } else if (trace_info.start_log_id > 0LL) { // get next end
      GE_ASSERT_SUCCESS(ReportTraceInfo(static_cast<uint16_t>(trace_info.start_log_id + 1LL), node));
    }
  }

  return ge::SUCCESS;
}

void CannTracingProfiler::OnExecuteEvent(int32_t sub_exe_graph_type, CannTracingProfiler *profiler, ExecutorEvent event,
                                         const void *node, KernelStatus result) {
  (void)result;
  (void)sub_exe_graph_type;
  if (profiler == nullptr) {
    GELOGW("Cann tracing profiler is nullptr.");
    return;
  }

  if (event == kExecuteStart) {
    (void)profiler->ReportStartTraceInfo(static_cast<const Node *>(node));
  }

  if (event == kExecuteEnd) {
    (void)profiler->ReportEndTraceInfo(static_cast<const Node *>(node));
  }

  if (event == kModelEnd) {
    profiler->IncreaseIterationNum();
  }
}
}
