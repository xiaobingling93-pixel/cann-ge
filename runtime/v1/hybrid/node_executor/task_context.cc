/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/task_context.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/log.h"
#include "graph/utils/tensor_utils.h"
#include "graph/types.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_executor.h"
#include "common/profiling/profiling_manager.h"
#include "common/dump/dump_manager.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"

namespace ge {
namespace hybrid {
TaskContext::TaskContext(GraphExecutionContext *const execution_context,
                         NodeState *const node_state,
                         SubgraphContext *const subgraph_context)
    : node_state_(node_state),
      node_item_(&node_state->GetNodeItem()),
      execution_context_(execution_context),
      subgraph_context_(subgraph_context),
      inputs_start_(subgraph_context->all_inputs_.data() + node_state->GetNodeItem().input_start),
      outputs_start_(subgraph_context->all_outputs_.data() + node_state->GetNodeItem().output_start),
      skip_sufficiency_of_input_check_(node_state->GetNodeItem().skip_sufficiency_of_input_check_) {}

TaskContext::~TaskContext() {
  GELOGD("[%s] TaskContext destroyed.", node_item_->NodeName().c_str());
  Reset();
}

void TaskContext::ReleaseWorkspace() {
  GELOGD("[%s] Start ReleaseWorkspace.", node_item_->NodeName().c_str());
  for (const auto ws_addr : workspaces_) {
    execution_context_->allocator->Deallocate(ws_addr);
  }
  workspaces_.clear();
}

void TaskContext::ReleaseAllMem() {
  ReleaseWorkspace();
  for (int32_t i = 0; i < NumInputs(); ++i) {
    ReleaseInput(i);
  }
  for (int32_t i = 0; i < NumOutputs(); ++i) {
    ReleaseOutput(i);
  }
}

std::unique_ptr<TaskContext> TaskContext::Create(NodeState *const node_state, SubgraphContext *const subgraph_context) {
  const NodeItem &node_item = node_state->GetNodeItem();
  GELOGI("[%s] To create task context, input start = %d, num_inputs = %d, output start = %d, num_outputs = %d.",
         node_item.NodeName().c_str(),
         node_item.input_start,
         node_item.num_inputs,
         node_item.output_start,
         node_item.num_outputs);
  if ((node_item.input_start < 0) || (node_item.output_start < 0)) {
    REPORT_INNER_ERR_MSG("E19999", "NodeItem:%s(%s) not property initialized."
                       "input_start:%d or output_start:%d less than 0",
                       node_item.NodeName().c_str(), node_item.NodeType().c_str(),
                       node_item.input_start, node_item.output_start);
    GELOGE(INTERNAL_ERROR,
           "[Check][Param]NodeItem:%s(%s) not property initialized. input_start = %d, output_start = %d",
           node_item.NodeName().c_str(), node_item.NodeType().c_str(),
           node_item.input_start, node_item.output_start);
    return nullptr;
  }
  GE_ASSERT_NOTNULL(subgraph_context->execution_context_);
  auto task_ctx = MakeUnique<TaskContext>(subgraph_context->execution_context_, node_state, subgraph_context);
  if (task_ctx == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Create TaskContext failed for [%s(%s)].",
                      node_item.NodeName().c_str(), node_item.NodeType().c_str());
    GELOGE(MEMALLOC_FAILED, "[Create][TaskContext] failed for [%s(%s)].",
           node_item.NodeName().c_str(), node_item.NodeType().c_str());
    return nullptr;
  }
  return task_ctx;
}

void TaskContext::Reset() {
  task_id_ = 0U;
  stream_id_ = 0U;
  status_ = SUCCESS;
  force_infer_shape_ = false;
  is_over_flow_ = false;
  ClearProfilingTaskDescInfo();
  ExceptionDumper::Reset(extra_op_info_);
  for (int32_t i = 0; i < NumOutputs(); ++i) {
    const auto output_tensor = MutableOutput(i);
    if (output_tensor != nullptr) {
      output_tensor->Destroy();
    }
  }
  ReleaseWorkspace();
}

int32_t TaskContext::NumInputs() const {
  return node_item_->num_inputs;
}

int32_t TaskContext::NumOutputs() const {
  return node_item_->num_outputs;
}

TensorValue *TaskContext::MutableInput(const int32_t idx) const {
  if ((idx < 0) || (idx >= node_item_->num_inputs)) {
    REPORT_INNER_ERR_MSG("E19999", "Index out of range, check invalid. index = %d, num_inputs = %d, node:%s(%s)",
                       idx, node_item_->num_inputs,
                       node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Index out of range. index = %d, num_inputs = %d, node:%s(%s)",
           idx, node_item_->num_inputs,
           node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    return nullptr;
  }
  return inputs_start_ + idx;
}

const TensorValue *TaskContext::GetOutput(const int32_t idx) const {
  if ((idx < 0) || (idx >= node_item_->num_outputs)) {
    REPORT_INNER_ERR_MSG("E19999", "Index out of range, check invalid. index = %d, num_outputs = %d, node:%s(%s)",
                       idx, node_item_->num_outputs,
                       node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Index out of range. index = %d, num_outputs = %d, node:%s(%s)",
           idx, node_item_->num_outputs,
           node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    return nullptr;
  }

  return outputs_start_ + idx;
}

TensorValue *TaskContext::MutableOutput(const int32_t idx) const {
  if ((idx < 0) || (idx >= node_item_->num_outputs)) {
    REPORT_INNER_ERR_MSG("E19999", "Index out of range, check invalid. index = %d, num_outputs = %d, node:%s(%s)",
                       idx, node_item_->num_outputs,
                       node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Index out of range. index = %d, num_outputs = %d, node:%s(%s)",
           idx, node_item_->num_outputs,
           node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    return nullptr;
  }
  return outputs_start_ + idx;
}

int32_t TaskContext::NumWorkspaces() const {
  return static_cast<int32_t>(workspaces_.size());
}

void *TaskContext::MutableWorkspace(const int32_t idx) const {
  if ((idx < 0) || (static_cast<size_t>(idx) >= workspaces_.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Index:%d out of range, check invalid. number:%zu of workspaces_, node:%s(%s)",
                       idx, workspaces_.size(), node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Index:%d out of range. number:%zu of workspaces_, node:%s(%s)",
           idx, workspaces_.size(), node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    return nullptr;
  }

  return workspaces_[static_cast<size_t>(idx)];
}

const TensorValue *TaskContext::GetInput(const int32_t idx) const {
  if ((idx < 0) || (idx >= node_item_->num_inputs)) {
    REPORT_INNER_ERR_MSG("E19999", "Index:%d out of range, check invalid. num_inputs:%d node:%s(%s)",
                       idx, node_item_->num_inputs, node_item_->NodeName().c_str(),
                       node_item_->NodeType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Index:%d out of range. num_inputs:%d node:%s(%s)",
           idx, node_item_->num_inputs, node_item_->NodeName().c_str(), node_item_->NodeType().c_str());
    return nullptr;
  }

  return inputs_start_ + idx;
}

Status TaskContext::AllocateWorkspaces() {
  const auto workspace_sizes = node_item_->node->GetOpDesc()->GetWorkspaceBytes();
  for (const auto work_size : workspace_sizes) {
    void *workspace = execution_context_->allocator->Allocate(static_cast<size_t>(work_size));
    if (workspace == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "node:%s(%s) Allocate workspace failed, size: %" PRId64 "",
                        node_item_->NodeName().c_str(), node_item_->NodeType().c_str(), work_size);
      GELOGE(ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED, "[Allocate][workspace] failed for node:%s(%s), size: %ld",
             node_item_->NodeName().c_str(), node_item_->NodeType().c_str(), work_size);
      return ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED;
    }

    workspaces_.emplace_back(workspace);
  }
  return SUCCESS;
}

Status TaskContext::RegisterCallback(const std::function<void()> &callback_fun) const {
  if (callback_fun == nullptr) {
    GELOGW("[%s] Callback is NULL", GetNodeName());
    return SUCCESS;
  }
  const auto ret = execution_context_->callback_manager->RegisterCallbackFunc(GetStream(), callback_fun);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "RegisterCallback failed for [%s]", GetNodeName());
    GELOGE(ret, "[Register][Callback] failed for [%s]", GetNodeName());
    (void)execution_context_->callback_manager->Destroy();
    return ret;
  }

  return SUCCESS;
}

std::string TaskContext::TensorDesc2String(const GeTensorDesc &desc) {
  std::stringstream ss;
  ss << "[TensorDesc] ";
  ss << "DataType = " << desc.GetDataType();
  ss << ", Format = " << desc.GetFormat();
  ss << ", Shape = [";
  for (const auto dim : desc.GetShape().GetDims()) {
    ss << dim << ", ";
  }
  ss << "]";

  return ss.str();
}

Status TaskContext::AllocateTensor(const GeTensorDesc &tensor_desc_in, TensorValue &tensor_out,
                                   const AllocationAttr * const attr) const {
  int64_t size_out = 0;
  if (ge::TensorUtils::GetSize(tensor_desc_in, size_out) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get TensorSize failed, tensor:%s", tensor_desc_in.GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][TensorSize] failed, tensor:%s", tensor_desc_in.GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (size_out == 0) {
    GELOGW("size from tensor_desc == 0");
  }

  auto buffer_tensor = TensorBuffer::Create(execution_context_->allocator, static_cast<size_t>(size_out), attr);
  if (buffer_tensor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "tensor:%s Allocate tensor failed, size: %" PRId64 "",
                      tensor_desc_in.GetName().c_str(), size_out);
    GELOGE(ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED, "[Allocate][workspace] failed for tensor:%s, size: %ld",
           tensor_desc_in.GetName().c_str(), size_out);
    return ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED;
  }
  tensor_out = TensorValue(shared_ptr<TensorBuffer>(buffer_tensor.release()));
  return SUCCESS;
}

bool TaskContext::HasAllocated(const int32_t idx) const {
  if (outputs_start_[idx].GetData() != nullptr) {
    GELOGD("[%s]'s [%d] output has been allocated", GetNodeName(), idx);
    return true;
  }
  for (const auto &pair : node_item_->reuse_outputs) {
    if ((pair.second == idx) && (outputs_start_[pair.first].GetData() != nullptr)) {
      GELOGD("[%s] reuse output [%d] with output [%d], but [%d] has allocated, so reuse it.", GetNodeName(), pair.first,
             idx, pair.first);
      outputs_start_[idx] = outputs_start_[pair.first];
      return true;
    }
  }
  return false;
}

Status TaskContext::AllocateOutput(const int32_t idx,
                                   const GeTensorDesc &tensor_desc_in,
                                   TensorValue **const tensor_out,
                                   const AllocationAttr * const attr) const {
  GELOGI("To allocate output for node: %s. index = %d, tensor desc = %s",
         node_item_->NodeName().c_str(),
         idx,
         TensorDesc2String(tensor_desc_in).c_str());

  if ((idx < 0) || (idx >= node_item_->num_outputs)) {
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) output index out of range check invalid. num_output = %d, index = %d",
                       node_item_->NodeName().c_str(), node_item_->NodeType().c_str(),
                       node_item_->num_outputs, idx);
    GELOGE(PARAM_INVALID, "[Check][Param] %s(%s) output index out of range. num_output = %d, index = %d",
           node_item_->NodeName().c_str(), node_item_->NodeType().c_str(),
           node_item_->num_outputs, idx);
    return PARAM_INVALID;
  }

  auto &task_tensor_out = outputs_start_[idx];
  if (HasAllocated(idx)) {
    int64_t expected_size = 0;
    GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc_in.GetShape(), tensor_desc_in.GetFormat(),
                                                     tensor_desc_in.GetDataType(), expected_size));
    const auto allocated_size = static_cast<int64_t>(task_tensor_out.GetSize());
    if (expected_size > allocated_size) {
      GELOGE(GRAPH_PARAM_INVALID,
             "[Check][Size] %s(%s) index[%d] mem size out of range! Expected size: %ld, but given input size: %ld.",
             node_item_->NodeName().c_str(), node_item_->NodeType().c_str(), idx, expected_size, allocated_size);

      std::string reason = "The memory " + std::to_string(expected_size) + " required by the output " + std::to_string(idx) +
                           " of the node " + node_item_->NodeName().c_str() + "(" +
                           node_item_->NodeType().c_str() + ") is greater than the allocated memory " +
                           std::to_string(allocated_size);

      REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                std::vector<const char_t *>({reason.c_str()}));
      return GRAPH_PARAM_INVALID;
    }

    GELOGI("already allocated as net output");
    return SUCCESS;
  }

  if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(tensor_desc_in)) {
    task_tensor_out = TensorValue();
    return SUCCESS;
  }

  const auto &it = node_item_->ref_outputs.find(idx);
  if (it != node_item_->ref_outputs.end()) {
    auto &ref_node = it->second;
    GELOGD("source node of %s:%d = %s, op_type = %s",
           node_item_->NodeName().c_str(),
           idx,
           ref_node->GetName().c_str(),
           ref_node->GetType().c_str());

    const TensorValue * const ref_tensor = execution_context_->model->GetTensor(ref_node);
    GE_CHECK_NOTNULL(ref_tensor);
    task_tensor_out = *ref_tensor;
  } else {
    const auto &reuse_output_it = node_item_->reuse_outputs.find(idx);
    if (reuse_output_it != node_item_->reuse_outputs.end()) {
      GELOGD("[%s] reuse output [%d] with output [%d]", GetNodeName(), idx, reuse_output_it->second);
      outputs_start_[idx] = outputs_start_[reuse_output_it->second];
    } else {
      const auto reuse_input = node_item_->reuse_inputs.find(idx);
      if (reuse_input != node_item_->reuse_inputs.end()) {
        GELOGD("[%s] Output[%d] is referenced to input[%d]", GetNodeName(), idx, reuse_input->second);
        task_tensor_out = inputs_start_[reuse_input->second];
      } else {
        GE_CHK_STATUS_RET_NOLOG(AllocateTensor(tensor_desc_in, task_tensor_out, attr));
        GELOGD("Allocating output successfully. node: %s. index = %d, size = %zu",
               node_item_->NodeName().c_str(), idx, task_tensor_out.GetSize());
      }
    }
  }

  if (execution_context_->trace_enabled) {
    task_tensor_out.SetName(node_item_->NodeName() + "_out_" + std::to_string(idx));
  }

  if (tensor_out != nullptr) {
    *tensor_out = outputs_start_ + idx;
  }

  return SUCCESS;
}

Status TaskContext::AllocateOutputs(AllocationAttr *const attr) const {
  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    const auto &output_desc = node_item_->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    if (attr == nullptr) {
      const auto tmp_attr = AllocationAttr(0, nullptr, node_item_->output_mem_types_[static_cast<size_t>(i)]);
      GE_CHK_STATUS_RET_NOLOG(AllocateOutput(i, *output_desc, nullptr, &tmp_attr));
    } else {
      attr->SetMemType(node_item_->output_mem_types_[static_cast<size_t>(i)]);
      GE_CHK_STATUS_RET_NOLOG(AllocateOutput(i, *output_desc, nullptr, attr));
    }
  }

  return SUCCESS;
}

const NodeItem &TaskContext::GetNodeItem() const {
  return *node_item_;
}

Status TaskContext::SetOutput(const int32_t index, const TensorValue &tensor_in) const {
  if ((index < 0) || (index >= node_item_->num_outputs)) {
    REPORT_INNER_ERR_MSG("E19999", "%s(%s) output index out of range check invalid. num_output = %d, index = %d",
                       node_item_->NodeName().c_str(), node_item_->NodeType().c_str(),
                       node_item_->num_outputs, index);
    GELOGE(PARAM_INVALID, "[Check][Param]%s(%s) output index out of range. num_output = %d, index = %d",
           node_item_->NodeName().c_str(), node_item_->NodeType().c_str(),
           node_item_->num_outputs, index);
    return PARAM_INVALID;
  }

  GELOGD("Set %s:%d with tensor: %s",
         node_item_->NodeName().c_str(),
         index,
         tensor_in.DebugString().c_str());
  outputs_start_[index] = tensor_in;
  return SUCCESS;
}

rtStream_t TaskContext::GetStream() const {
  return execution_context_->stream;
}

void TaskContext::SetStatus(const Status stat) {
  status_ = stat;
  if (stat != SUCCESS) {
    execution_context_->SetErrorCode(stat);
  }
}

uint32_t TaskContext::GetTaskId() const {
  return task_id_;
}

uint32_t TaskContext::GetStreamId() const {
  return stream_id_;
}

void TaskContext::SetOverFlow(const bool over_flow) {
  is_over_flow_ = over_flow;
}

bool TaskContext::IsOverFlow() const {
  return is_over_flow_;
}

Status TaskContext::AllocateWorkspace(const size_t alloc_size, void *&alloc_buffer, void *const ori_addr) {
  if (ori_addr == nullptr) {
    alloc_buffer = execution_context_->allocator->Allocate(alloc_size, nullptr);
  } else {
    const AllocationAttr attr(ori_addr);
    alloc_buffer = execution_context_->allocator->Allocate(alloc_size, &attr);
  }

  if (alloc_buffer == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Allocate Workspace failed, size = %zu", alloc_size);
    GELOGE(MEMALLOC_FAILED, "[Allocate][Workspace] failed, size = %zu", alloc_size);
    return MEMALLOC_FAILED;
  }

  GELOGD("[%s] Allocating workspace of size = %zu successfully", node_item_->NodeName().c_str(), alloc_size);
  workspaces_.emplace_back(alloc_buffer);
  return SUCCESS;
}

Status TaskContext::PropagateOutputs() const {
  // propagate outputs
  const auto &guard = node_item_->MutexGuard("PropagateOutputs");
  for (int32_t i = 0; i < NumOutputs(); ++i) {
    const auto &tensor = MutableOutput(i);
    GE_CHECK_NOTNULL(tensor);
    if (tensor->GetData() == nullptr) {
      GELOGD("[%s] Node output[%d] is null.", node_item_->NodeName().c_str(), i);
    }
    const auto &output_nodes = node_item_->outputs[static_cast<size_t>(i)];
    for (const auto &dst_input_index_and_node : output_nodes) {
      const auto dst_input_idx = dst_input_index_and_node.first;
      const auto dst_node_item = dst_input_index_and_node.second;
      const auto input_offset = dst_node_item->input_start + dst_input_idx;
      GELOGD("Propagate output of node %s, output index = %d, dst node = %s, "
             "dst_input_index = %d, dst_input_offset = %d.",
             node_item_->NodeName().c_str(),
             i,
             dst_node_item->NodeName().c_str(),
             dst_input_idx,
             input_offset);

      if (subgraph_context_->all_inputs_.size() <= static_cast<size_t>(input_offset)) {
        REPORT_INNER_ERR_MSG("E19999",
                           "[%s(%s)] input index out of range check invalid. index = %d, total input num = %zu",
                           GetNodeName(), dst_node_item->NodeType().c_str(),
                           input_offset, subgraph_context_->all_inputs_.size());
        GELOGE(INTERNAL_ERROR, "[Check][Size][%s(%s)] input index out of range. index = %d, total input num = %zu",
               GetNodeName(), dst_node_item->NodeType().c_str(), input_offset, subgraph_context_->all_inputs_.size());
        return INTERNAL_ERROR;
      }

      subgraph_context_->all_inputs_[static_cast<size_t>(input_offset)] = *tensor;
      if (execution_context_->trace_enabled) {
        subgraph_context_->all_inputs_[static_cast<size_t>(input_offset)].SetName(
            dst_node_item->NodeName() + "_in_" + std::to_string(dst_input_idx));
      }
    }
  }
  (void)guard;
  return SUCCESS;
}

const char_t *TaskContext::GetNodeName() const {
  return node_item_->NodeName().c_str();
}

void TaskContext::ReleaseInput(const int32_t index) {
  const auto input_tensor = MutableInput(index);
  if (input_tensor != nullptr) {
    node_state_->SavePersistTensor(index, *input_tensor);
    input_tensor->Destroy();
    GELOGD("[%s] Tensor of input[%d] released", GetNodeName(), index);
  }
}

void TaskContext::ReleaseOutput(const int32_t index) {
  const auto output_tensor = MutableOutput(index);
  if (output_tensor != nullptr) {
    output_tensor->Destroy();
    GELOGD("[%s] Tensor of output[%d] released", GetNodeName(), index);
  }
}

void TaskContext::ReleaseAllOutput() {
  for (int32_t i = 0; i < NumOutputs(); ++i) {
    ReleaseOutput(i);
  }
}

ConstGeTensorDescPtr TaskContext::GetOutputDesc(const int32_t index) const {
  return node_item_->MutableOutputDesc(index);
}

ConstGeTensorDescPtr TaskContext::GetInputDesc(const int32_t index) const {
  return node_item_->MutableInputDesc(index);
}

GeTensorDescPtr TaskContext::MutableInputDesc(const int32_t index) const {
  return node_item_->MutableInputDesc(index);
}

GeTensorDescPtr TaskContext::MutableOutputDesc(const int32_t index) const {
  return node_item_->MutableOutputDesc(index);
}

bool TaskContext::IsForceInferShape() const {
  return force_infer_shape_;
}

void TaskContext::SetForceInferShape(const bool force_infer_shape) {
  force_infer_shape_ = force_infer_shape;
}

void TaskContext::NodeDone() {
  subgraph_context_->NodeDone(node_item_->node);
}

void TaskContext::OnError(const Status error) const {
  subgraph_context_->OnError(error);
  execution_context_->SetErrorCode(error);
}

bool TaskContext::IsTraceEnabled() const {
  return execution_context_->trace_enabled;
}

TensorValue *TaskContext::GetVariable(const std::string &name) const {
  return execution_context_->model->GetVariable(name);
}

bool TaskContext::IsDumpEnabled() const {
  const DumpProperties &dump_properties = GetDumpProperties();
  return dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen() ||
         DumpManager::GetInstance().IsDumpExceptionOpen();
}

Status TaskContext::TryExecuteCallback(const function<void()> &callback_fun) const {
  if (!callback_fun) {
    return SUCCESS;
  }

  if (node_item_->has_observer) {
    return RegisterCallback(callback_fun);
  }

  callback_fun();
  return SUCCESS;
}

const DumpProperties &TaskContext::GetDumpProperties() const {
  return execution_context_->dump_properties;
}

bool TaskContext::NeedCallback() const {
  return (node_item_->has_observer || IsDumpEnabled());
}

Status TaskContext::Synchronize() const {
  return execution_context_->Synchronize(GetStream());
}

Status TaskContext::SaveProfilingTaskDescInfo(const std::string &task_type, const uint32_t block_dim,
                                              const std::string &op_type) {
  if (DumpManager::GetInstance().IsDumpExceptionOpen() || ProfilingManager::Instance().ProfilingModelLoadOn() ||
      ProfilingProperties::Instance().ProfilingSubscribeOn()) {
    GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
    GE_CHK_RT_RET(aclrtStreamGetId(GetStream(), reinterpret_cast<int32_t*>(&stream_id_)));
    GELOGD("Get Node[%s] task id: %u, stream id: %u.", GetNodeName(), task_id_, stream_id_);
  }
  if (ProfilingManager::Instance().ProfilingModelLoadOn() || ProfilingManager::Instance().ProfilingSubscribeOn()) {
    const NodeItem &node_item = GetNodeItem();
    const auto op_desc = node_item.GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const GraphExecutionContext * const graph_context = GetExecutionContext();
    GE_CHECK_NOTNULL(graph_context);
    const HybridModel *const model = graph_context->model;
    GE_CHECK_NOTNULL(model);

    const std::string dynamic_model_name = model->GetModelName();
    TaskDescInfo op_task_desc_info;
    op_task_desc_info.model_name = dynamic_model_name;
    op_task_desc_info.op_name = op_desc->GetName();
    op_task_desc_info.op_type = op_type;
    op_task_desc_info.block_dim = block_dim;
    op_task_desc_info.task_type = task_type;
    op_task_desc_info.task_id = task_id_;
    op_task_desc_info.stream_id = stream_id_;
    op_task_desc_info.shape_type = "dynamic";
    op_task_desc_info.cur_iter_num = (model->IsSingleOp()) ? ProfilingManager::Instance().GetStepInfoIndex() :
                                     (execution_context_->iteration + 1);
    task_desc_info.emplace_back(op_task_desc_info);
  }
  GELOGD("save profling task desc info size is %zu", task_desc_info.size());
  return SUCCESS;
}

NodeState *TaskContext::GetNodeState() const {
  return node_state_;
}

Status TaskContext::GetInputDesc(const int32_t index, GeTensorDesc &tensor_desc) const {
  return node_item_->GetInputDesc(index, tensor_desc);
}

Status TaskContext::UpdateInputDesc(const int32_t index, const GeTensorDesc &tensor_desc) const {
  return node_item_->UpdateInputDesc(index, tensor_desc);
}
}  // namespace hybrid
}  // namespace ge
