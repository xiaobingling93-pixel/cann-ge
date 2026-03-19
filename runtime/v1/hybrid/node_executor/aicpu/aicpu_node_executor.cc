/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"
#include "framework/common/taskdown_common.h"
#include "formats/formats.h"
#include "aicpu_task_struct.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/utils/node_utils.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/model/hybrid_model.h"
#include "runtime/rt.h"
#include "rt_error_codes.h"
#include "graph/def_types.h"
#include "common/utils/executor_utils.h"
#include "graph/ge_context.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"

namespace ge {
namespace hybrid {
namespace {
const char_t *const kFunctionOp = "FunctionOp";
const char_t *const kExceptionAbort = "_exception_abort_flag";
// mem need release
const char_t *const kAicpuOpAllshape = "_AllShape";
const int32_t kBlockdimAxisDefaultIndex = -1;
const int32_t kDefaultBase = 10;
Status GetExtendedIoLenOfTfNodeForHostMemInput(const size_t io_len, size_t &host_mem_offset, size_t &extend_len) {
  // input address of tf kernel cpu must be aligned to 64B
  GE_CHK_STATUS_RET(CheckUint32AddOverflow(io_len, (kAlignBytes64 - 1U)),
                    "Padding size is beyond the UINT32_MAX.");
  host_mem_offset = ((io_len + kAlignBytes64 - 1U) / kAlignBytes64) * kAlignBytes64;
  extend_len = (host_mem_offset - io_len) + kMaxHostMemInputLen;
  return SUCCESS;
}
}
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_TF, AiCpuNodeExecutor);
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_CUSTOM, AiCpuNodeExecutor);

AicpuNodeTaskBase::~AicpuNodeTaskBase() {
  if (rt_event_ != nullptr) {
    (void)aclrtDestroyEvent(rt_event_);
  }
}

Status AicpuNodeTaskBase::AllocTensorBuffer(const size_t size, std::unique_ptr<TensorBuffer> &tensor_buffer,
                                            NpuMemoryAllocator *const allocator) const {
  GE_CHECK_NOTNULL(allocator);
  if (deploy_type_flag_ == RT_KERNEL_HOST_ONLY) {
    const auto tmp_attr = AllocationAttr(0, nullptr, HOST_SVM);
    tensor_buffer = TensorBuffer::Create(allocator, size, &tmp_attr);
  } else {
    tensor_buffer = TensorBuffer::Create(allocator, size);
  }
  GE_CHECK_NOTNULL(tensor_buffer);
  GELOGD("Malloc %s memory, addr is %p for %d", deploy_type_flag_ == RT_KERNEL_HOST_ONLY ? "host" : "device",
         tensor_buffer->GetData(), deploy_type_flag_);
  return SUCCESS;
}

Status AicpuNodeTaskBase::InitExtInfo(const std::string &kernel_ext_info, const uint64_t session_id) {
  if (kernel_ext_info.empty()) {
    if (node_item_->is_dynamic) {
      // dynamic node must have ext info
      REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] parse ext info failed as ext info is empty.",
                         node_name_.c_str(), node_type_.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param:kernel_ext_info] Node[%s(%s)] parse ext info failed as ext info is empty.",
             node_name_.c_str(), node_type_.c_str());
      return PARAM_INVALID;
    } else {
      // if no ext info no need copy to device.
      GELOGD("Node[%s] kernel_ext_info is empty, no need copy to device, is_dynamic=%s.",
             node_name_.c_str(), node_item_->is_dynamic ? "true" : "false");
      return SUCCESS;
    }
  }

  GE_CHK_STATUS_RET(aicpu_ext_handle_.Parse(kernel_ext_info),
                    "[Invoke][Parse]Node[%s(%s)] parse kernel ext info failed, kernel_ext_info_size = %zu.",
                    node_name_.c_str(), node_type_.c_str(), kernel_ext_info.size());
  deploy_type_flag_ = aicpu_ext_handle_.GetDeployTypeFlag();
  memcpy_kind_ = aicpu_ext_handle_.GetMemcpyKind();
  if (deploy_type_flag_ == RT_KERNEL_HOST_ONLY) {
    callback_memcpy_kind_ = RT_MEMCPY_HOST_TO_HOST;
  }
  qos_level_flag_ = aicpu_ext_handle_.GeQosLevelFlag();
  GELOGD("To update aicpu_task ext_info session_info session_id to %lu with deploy type %d, qos level flag %u",
         session_id, deploy_type_flag_, qos_level_flag_);
  GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateSessionInfoId(session_id),
                    "[Update][SessionInfoSessionId] failed, node:%s(%s), session_id:%lu.",
                    node_name_.c_str(), node_type_.c_str(), session_id);

  if (is_blocking_aicpu_op_) {
    if (UpdateEventIdForBlockingAicpuOp() != SUCCESS) {
      GELOGE(FAILED, "[Call][UpdateEventIdForBlockingAicpuOp] failed, node:%s(%s)",
             node_name_.c_str(), node_type_.c_str());
      return FAILED;
    }
  }

  // copy task args buf
  GE_CHK_STATUS_RET(AllocTensorBuffer(aicpu_ext_handle_.GetExtInfoLen(), ext_info_addr_dev_),
                    "[Invoke][AllocTensorBuffer] Node[%s(%s)] alloc kernel_ext_info buf failed, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), aicpu_ext_handle_.GetExtInfoLen());

  // copy default ext info
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_->GetData(), ext_info_addr_dev_->GetSize(), aicpu_ext_handle_.GetExtInfo(),
                         aicpu_ext_handle_.GetExtInfoLen(), memcpy_kind_));

  (void)ge::GetContext().GetOption(OPTION_EXEC_STREAM_SYNC_TIMEOUT, stream_sync_timeout_);
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateOutputShapeFromExtInfo(TaskContext &context) {
  if (node_item_->num_outputs == 0) {
    GELOGD("Task [%s] output_num is 0, no need update output shape.", node_name_.c_str());
    return SUCCESS;
  }
  // copy to host buf
  GE_CHK_RT_RET(rtMemcpy(aicpu_ext_handle_.GetExtInfo(), aicpu_ext_handle_.GetExtInfoLen(),
                         ext_info_addr_dev_->GetData(), ext_info_addr_dev_->GetSize(), callback_memcpy_kind_));
  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    GeShape shape;
    // not support update data type now, just for param
    DataType data_type;
    (void)aicpu_ext_handle_.GetOutputShapeAndType(static_cast<uint32_t>(i), shape, data_type);
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(context, shape, i),
                      "[Invoke][UpdateShapeToOutputDesc]Update node %s(%s) [%d]th output shape[datatype:%d] failed.",
                      node_name_.c_str(), node_type_.c_str(), i, data_type);
  }
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateShapeToOutputDesc(const TaskContext &context,
                                                  const GeShape &shape_new,
                                                  const int32_t output_index) const {
  const auto output_desc = context.MutableOutputDesc(output_index);
  GE_CHECK_NOTNULL(output_desc);
  const auto shape_old = output_desc->GetShape();
  GELOGD("Update node[%s] out[%d] shape from %s to %s.", node_name_.c_str(), output_index,
         shape_old.ToString().c_str(), shape_new.ToString().c_str());

  const auto origin_shape_old = output_desc->GetOriginShape();
  const auto origin_format = output_desc->GetOriginFormat();
  const auto format = output_desc->GetFormat();
  if (origin_format == format) {
    return context.GetNodeState()->UpdateOutputShapes(output_index, shape_new, shape_new);
  }

  // if format is not same need convert shape
  std::vector<int64_t> origin_dims_new;
  const auto trans_ret = formats::TransTensorShape(format, shape_new.GetDims(),
                                                   output_desc->GetDataType(), origin_format, origin_dims_new);
  GE_CHK_STATUS_RET(trans_ret,
                    "[Trans][Shape] failed for Node[%s(%s)] out[%d] originFormat[%d] is not same as format[%d], "
                    "shape=%s.", node_name_.c_str(), node_type_.c_str(), output_index,
                    origin_format, format, shape_new.ToString().c_str());
  const auto origin_shape_new = GeShape(origin_dims_new);
  GE_CHK_STATUS_RET(context.GetNodeState()->UpdateOutputShapes(output_index, shape_new, origin_shape_new),
                    "[Update][OutputShapes] failed for Node[%s(%s)], index = %d",
                    node_name_.c_str(), node_type_.c_str(), output_index);
  GELOGD("Node[%s] out[%d] originFormat[%d] is not same as format[%d], need update from %s ro %s.",
         node_name_.c_str(), output_index, origin_format, format,
         origin_shape_old.ToString().c_str(), origin_shape_new.ToString().c_str());
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateExtInfo() {
  GELOGI("Node[%s] update ext info begin, unknown_type=%d.", node_name_.c_str(), unknown_type_);
  if ((node_item_->num_inputs == 0) && (node_item_->num_outputs == 0)) {
    GELOGD("Node[%s] has no input and output, no need update ext info.", node_name_.c_str());
    return SUCCESS;
  }

  for (auto i = 0; i < node_item_->num_inputs; ++i) {
    const auto input_desc = node_item_->MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateInputShapeAndType(static_cast<uint32_t>(i), *input_desc),
                      "[Update][InputShapeAndType] failed for Node[%s(%s)] input[%d].",
                      node_name_.c_str(), node_type_.c_str(), i);
  }

  if ((unknown_type_ != DEPEND_COMPUTE) || (!node_item_->is_dynamic)) {
    for (auto j = 0; j < node_item_->num_outputs; ++j) {
      const auto output_desc = node_item_->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);

      GE_CHK_STATUS_RET(aicpu_ext_handle_.UpdateOutputShapeAndType(static_cast<uint32_t>(j), *output_desc),
                        "[Update][OutputShapeAndType] failed for Node[%s(%s)] output[%d].",
                        node_name_.c_str(), node_type_.c_str(), j);
    }
  }
  // copy input and output shapes to device
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_->GetData(), ext_info_addr_dev_->GetSize(), aicpu_ext_handle_.GetExtInfo(),
                         aicpu_ext_handle_.GetExtInfoLen(), memcpy_kind_));

  GELOGD("Node[%s] update ext info end.", node_name_.c_str());
  return SUCCESS;
}

static int64_t CeilDivisor(const int64_t x, const int64_t base) {
  GE_CHECK_SIZE(base);
  int64_t ret = x / base;
  if ((x % base) != 0) {
    ret++;
  }
  return ret;
}

Status AicpuNodeTaskBase::UpdateBlockDimInfo(const int32_t block_dim_index) {
  GELOGD("Node[%s] UpdateBlockDimInfo block_dim_index[%d].", node_name_.c_str(), block_dim_index);
  const auto input_desc = node_item_->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  const auto &input_shape = input_desc->GetShape();
  const auto input_dims = input_shape.GetDims();

  int64_t total = 1;
  if (block_dim_index == kBlockdimAxisDefaultIndex) {
    total = input_shape.GetShapeSize();
  } else {
    total = input_shape.GetDim(static_cast<size_t>(block_dim_index));
  }
  uint32_t ai_cpu_cnt = 1U;
  const auto ret = rtGetAiCpuCount(&ai_cpu_cnt);
  GELOGD("Node[%s] ai_cpu_cnt[%u]!", node_name_.c_str(), ai_cpu_cnt);
  if (ret != 0) {
    GELOGD("Node[%s] get AiCpuCount failed!", node_name_.c_str());
  }
  const int64_t max_shard_num = static_cast<int64_t>(ai_cpu_cnt) * 2;
  const int64_t per_unit_size = total / std::min(std::max(int64_t{1}, static_cast<int64_t>(ai_cpu_cnt)),
                                                 total);
  int64_t block_size = std::max(int64_t{1}, std::min(total, per_unit_size));
  int64_t shard_num = CeilDivisor(total, block_size);
  shard_num = std::min(max_shard_num, shard_num);
  if (shard_num == 0) {
    GELOGE(FAILED, "[UpdateBlockDimInfo]shard num is %ld", shard_num);
    return FAILED;
  }
  block_size = CeilDivisor(total, shard_num);
  block_num_ = static_cast<uint32_t>(CeilDivisor(total, block_size));
  GELOGD("GetBlockDimInfo[%s] total[%ld] blocksize[%ld] blockdim[%u].", node_name_.c_str(), total, block_size,
         block_num_);
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateArgs(TaskContext &context) {
  GELOGD("Node[%s] update args begin. is_dynamic=%s, unknown_type=%d",
         node_name_.c_str(), node_item_->is_dynamic ? "true" : "false", unknown_type_);
  if ((node_item_->num_inputs == 0) && (node_item_->num_outputs == 0)) {
    GELOGD("Node[%s] has no input and output, no need update args.", node_name_.c_str());
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(UpdateIoAddr(context), "[Update][IoAddr] failed for Node[%s].", node_name_.c_str());
  bool all_shape = false;
  const OpDescPtr op_desc = node_item_->GetOpDesc();
  (void)AttrUtils::GetBool(op_desc, kAicpuOpAllshape, all_shape);
  if (node_item_->is_dynamic || all_shape) {
    // dynamic node and all_shape kernel need update ext info.
    GE_CHK_STATUS_RET(UpdateExtInfo(), "[Update][ExtInfo] failed for Node[%s(%s)].",
                      node_name_.c_str(), node_type_.c_str());
  }

  if (node_item_->is_dynamic && is_support_block_) {
    // dynamic node and supportBlockdim need update blockdim.
    GE_CHK_STATUS_RET(UpdateBlockDimInfo(axis_index_),
                      "[Update][BlockDiminfo] failed for Node[%s(%s)] blocknum[%u].",
                      node_name_.c_str(), node_type_.c_str(), block_num_);
  }

  GELOGD("Node[%s] update args end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuNodeTaskBase::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[AicpuNodeTaskBaseExecuteAsync] Start");
  GELOGD("Node[%s] execute async start. unknown_type=%d.", node_name_.c_str(), unknown_type_);

  HYBRID_CHK_STATUS_RET(LaunchTask(context), "[Launch][Task] failed for [%s(%s)].",
                        node_name_.c_str(), node_type_.c_str());

  // save profiling data
  GE_CHK_STATUS_RET(context.SaveProfilingTaskDescInfo(kTaskTypeAicpu, 0U, node_type_),
                    "[Save][Profiling] failed for node[%s]!", context.GetNodeName());
  const auto callback = [this, done_callback, &context]() {
    GELOGD("Node[%s] callback start.", node_name_.c_str());
    RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[TaskCallback] Start");
    const Status callback_ret = TaskCallback(context);
    if (callback_ret != SUCCESS) {
      context.OnError(callback_ret);
    }
    RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[TaskCallback] End");

    GELOGD("Node[%s] task callBack ret = %u.", node_name_.c_str(), callback_ret);
    if (done_callback != nullptr) {
      context.SetStatus(callback_ret);
      done_callback();
    }

    GELOGD("Node[%s] callback end.", node_name_.c_str());
  };

  GE_CHK_STATUS_RET_NOLOG(context.RegisterCallback(callback));

  GELOGD("Node[%s] execute async end.", node_name_.c_str());
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(),
                         "[AicpuNodeTaskBaseExecuteAsync] End");
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateEventIdForBlockingAicpuOp() {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED,
           "[Call][CheckDeviceSupportBlockingAicpuOpProcess] for node:%s(%s) failed",
           node_name_.c_str(), node_type_.c_str());
    return FAILED;
  }
  if (!is_support) {
    GELOGD("Device not support blocking aicpu op process");
    return SUCCESS;
  }
  uint32_t event_id = 0U;
  auto rt_ret = aclrtCreateEventWithFlag(
    &rt_event_,ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS | ACL_EVENT_TIME_LINE);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtCreateEventWithFlag failed for node:%s(%s), ret:%d",
                      node_name_.c_str(), node_type_.c_str(), rt_ret);
    GELOGE(RT_FAILED, "[Call][aclrtCreateEventWithFlag] failed for node:%s(%s), ret:%d",
           node_name_.c_str(), node_type_.c_str(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = aclrtGetEventId(rt_event_, &event_id);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtGetEventId failed for node:%s(%s), ret:%d",
                      node_name_.c_str(), node_type_.c_str(), rt_ret);
    GELOGE(RT_FAILED, "[Call][aclrtGetEventId] failed for node:%s(%s), ret:%d",
           node_name_.c_str(), node_type_.c_str(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (aicpu_ext_handle_.UpdateEventId(event_id) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Update event id failed for node:%s(%s).",
                      node_name_.c_str(), node_type_.c_str());
    GELOGE(FAILED, "[Update][EventId] Update event id failed for node:%s(%s)",
           node_name_.c_str(), node_type_.c_str());
    return FAILED;
  }
  GELOGI("Update event_id=%u success", event_id);
  return SUCCESS;
}

Status AicpuNodeTaskBase::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const {
  int32_t device_id = 0;
  auto rt_ret = aclrtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtGetDevice failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][aclrtGetDevice] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  int32_t value = 0;
  rt_ret = rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtGetDeviceCapability failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtGetDeviceCapability] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if ((value != RT_AICPU_BLOCKING_OP_NOT_SUPPORT) && (value != RT_AICPU_BLOCKING_OP_SUPPORT)) {
    REPORT_INNER_ERR_MSG("E19999", "Value should be %d or %d but %d",
                       RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    GELOGE(FAILED, "[Check][Value] Value should be %d or %d but %d",
           RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    return FAILED;
  }
  is_support = (value == RT_AICPU_BLOCKING_OP_SUPPORT);
  return SUCCESS;
}

Status AicpuNodeTaskBase::DistributeWaitTaskForAicpuBlockingOp(rtStream_t stream) const {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED,
           "[Call][CheckDeviceSupportBlockingAicpuOpProcess] failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("Device not support blocking aicpu op process.");
    return SUCCESS;
  }
  GELOGD("Distribute queue task begin");
  if (rt_event_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "rt_event_ is nullptr");
    GELOGE(FAILED, "[Check][Param] rt_event_ is nullptr");
    return FAILED;
  }
  SetTaskTag();
  auto rt_ret = rtStreamWaitEvent(stream, rt_event_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtStreamWaitEvent failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtStreamWaitEvent] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  SetTaskTag();
  rt_ret = aclrtResetEvent(rt_event_, stream);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtResetEvent failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][aclrtResetEvent] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status AicpuNodeTaskBase::CheckOverflow(TaskContext &context) const {
  bool is_debug_open = context.GetDumpProperties().IsOpDebugOpen();
  GELOGD("Op %s is debug open: %s", context.GetNodeName(), is_debug_open ? "true" : "false");
  if (is_debug_open) {
    const auto rt_ret = rtStreamSynchronize(context.GetStream());
    // AICPU is responsible for dump itself. This code is only reserved code for future solution.
    if (rt_ret == ACL_ERROR_RT_OVER_FLOW) {
      context.SetOverFlow(true);
      (void)rtsGetThreadLastTaskId(context.MutableTaskId());
      (void)rtsStreamGetId(context.GetStream(), reinterpret_cast<int32_t*>(context.MutableStreamId()));
      GELOGW("TaskBase Dynamic shape op %s is over flow", context.GetNodeName());
      return SUCCESS;
    }
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Invoke][RtStreamSynchronize] failed TaskBase, ret:%d.", rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronize failed, ret:%d.", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::InitForDependComputeTask() {
  if ((unknown_type_ != DEPEND_COMPUTE) || (!node_item_->is_dynamic) ||
      (node_item_->num_outputs == 0)) {
    GELOGD("Node[%s] type[%s] unknown_type is %d, output num is %d.",
           node_name_.c_str(), node_item_->node_type.c_str(), unknown_type_, node_item_->num_outputs);
    return SUCCESS;
  }

  output_summary_.resize(static_cast<size_t>(node_item_->num_outputs));
  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(aicpu::FWKAdapter::ResultSummary),
                                        output_summary_[static_cast<size_t>(i)]),
                      "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy result summary info, size = %zu.",
                      node_name_.c_str(), node_type_.c_str(), sizeof(aicpu::FWKAdapter::ResultSummary));
  }
  output_summary_host_.resize(static_cast<size_t>(node_item_->num_outputs));

  // init for mem copy task
  // copy task need copy output_data and output_shape, max len is 2 * output_num
  const size_t copy_input_buf_len = static_cast<size_t>(node_item_->num_outputs) * 2U * sizeof(uint64_t);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_release_flag_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input release_flag, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_data_size_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input data_size, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_src_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input src, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_dst_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input dst, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);

  // copy task args buf
  GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(STR_FWK_OP_KERNEL), copy_task_args_buf_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task args, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), sizeof(STR_FWK_OP_KERNEL));

  std::vector<uint64_t> copy_io_addr;
  copy_io_addr.emplace_back(PtrToValue(copy_input_release_flag_dev_->GetData()));
  copy_io_addr.emplace_back(PtrToValue(copy_input_data_size_dev_->GetData()));
  copy_io_addr.emplace_back(PtrToValue(copy_input_src_dev_->GetData()));
  copy_io_addr.emplace_back(PtrToValue(copy_input_dst_dev_->GetData()));

  // mem copy op has 4 inputs and 0 output.
  const auto copy_io_addr_size = sizeof(uint64_t) * copy_io_addr.size();

  // can alloc in init, it can reuse
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_io_addr_size, copy_ioaddr_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task ioaddr, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_io_addr_size);
  GE_CHK_RT_RET(rtMemcpy(copy_ioaddr_dev_->GetData(), copy_io_addr_size, &copy_io_addr[0U], copy_io_addr_size,
                         memcpy_kind_));
  return SUCCESS;
}

static bool CanSyncStream(const OpDescPtr op_desc, const std::string node_type) {
  if (node_type.find(GETNEXT) != std::string::npos) {
    GELOGD("Op type[%s].", node_type.c_str());
    return true;
  }
  if ((node_type == kFunctionOp) &&
      (AttrUtils::HasAttr(op_desc, kExceptionAbort))) {
    bool is_exception_abort = false;
    (void)AttrUtils::GetBool(op_desc, kExceptionAbort, is_exception_abort);
    if (is_exception_abort) {
      return true;
    }
  }
  return false;
}

Status AicpuTfNodeTask::InitTopicTypAndExtInfo(const HybridModel &model) {
  GE_CHK_BOOL_RET_STATUS(task_def_.has_kernel_ex(), FAILED,
                         "[Check][TaskDef] Node[%s(%s)] is tf node"
                         "but task def does not has kernel ex.",
                         node_name_.c_str(), node_type_.c_str());
  auto &kernel_ex_def = task_def_.kernel_ex();
  auto &kernel_ext_info = kernel_ex_def.kernel_ext_info();
  const auto kernel_ext_info_size = kernel_ex_def.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "[Check][Size]Node[%s(%s)] task def kernel_ext_info.size = %zu,"
                         "but kernel_ext_info_size = %u.",
                         node_name_.c_str(), node_type_.c_str(), kernel_ext_info.size(), kernel_ext_info_size);

  const uint64_t ext_session_id = model.GetSessionId();
  // init extinfo
  GE_CHK_STATUS_RET(InitExtInfo(kernel_ext_info, ext_session_id), "[Init][ExtInfo] failed for Node[%s(%s)].",
                    node_name_.c_str(), node_type_.c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::AssembleWorkSpaceAddr(const domi::KernelExDef &kernel_ex_def) {
  const auto kernel_workspace_size = kernel_ex_def.task_info().size();
  GE_CHK_STATUS_RET(AllocTensorBuffer(kernel_workspace_size, kernel_workspace_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy kernel workspace, size = %zu.",
                    node_name_.c_str(), node_type_.c_str(), kernel_workspace_size);
  GE_CHK_RT_RET(rtMemcpy(kernel_workspace_->GetData(), kernel_workspace_size, kernel_ex_def.task_info().data(),
                         kernel_workspace_size, memcpy_kind_));
  GELOGI("op %s use %s mem %p for workspace with flag %d", node_name_.c_str(),
         deploy_type_flag_ == RT_KERNEL_HOST_ONLY ? "host" : "device", kernel_workspace_->GetData(), deploy_type_flag_);
  return SUCCESS;
}

Status AicpuTfNodeTask::AssembleKernelBuffer(STR_FWK_OP_KERNEL *fwk_op_kernel) {
  GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(STR_FWK_OP_KERNEL), kernel_buf_),
                    "[Alloc][TensorBuffer] for Node[%s(%s)] to copy kernel_buf, size=%zu.", node_name_.c_str(),
                    node_type_.c_str(), sizeof(STR_FWK_OP_KERNEL));
  GE_CHK_RT_RET(rtMemcpy(kernel_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL), fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                         memcpy_kind_));
  GELOGI("op %s use %s mem %p for kernel buffer with flag %d", node_name_.c_str(),
         deploy_type_flag_ == RT_KERNEL_HOST_ONLY ? "host" : "device", kernel_buf_->GetData(), deploy_type_flag_);
  return SUCCESS;
}

Status AicpuTfNodeTask::Init(const HybridModel &model) {
  GELOGI("Node[%s] init start.", node_name_.c_str());
  // init block info
  const OpDescPtr op_desc = node_item_->GetOpDesc();
  InitBlockAicpuOp(op_desc);
  GE_CHK_STATUS_RET(InitTopicTypAndExtInfo(model), "[Int][TopicTypAndExtInfo] for Node[%s(%s)] failed.",
                    node_name_.c_str(), node_type_.c_str());
  auto &kernel_ex_def = task_def_.kernel_ex();
  GE_CHK_STATUS_RET(AssembleWorkSpaceAddr(kernel_ex_def), "[Assemble][WorkSpaceAddr] for Node[%s(%s)] failed.",
                    node_name_.c_str(), node_type_.c_str());
  const int32_t input_output_num = node_item_->num_inputs + node_item_->num_outputs;
  auto input_output_size = static_cast<size_t>(input_output_num) * sizeof(uint64_t);
  if (ExecutorUtils::HasHostMemInput(node_item_->GetOpDesc())) {
    size_t extend_len;
    // input address of tf kernel cpu must be aligned to 64B
    GE_CHK_STATUS_RET(GetExtendedIoLenOfTfNodeForHostMemInput(input_output_size,
                                                              host_mem_input_data_offset_, extend_len),
                      "extend io length failed");
    input_output_size += extend_len;
    GELOGD("Node[%s(%s)] has host memory input, io addr host is extended %zu, length = %zu,"
           "host_mem_input_data_offset = %zu.", node_name_.c_str(), node_type_.c_str(), extend_len,
           input_output_size, host_mem_input_data_offset_);
  }
  // alloc input output addr buf, allow alloc size 0
  GE_CHK_STATUS_RET(AllocTensorBuffer(input_output_size, input_output_addr_),
                    "[Alloc][TensorBuffer] for Node[%s(%s)] to copy io addr, size = %zu.",
                    node_name_.c_str(), node_type_.c_str(), input_output_size);

  GE_CHK_STATUS_RET(InitForDependComputeTask(), "[Init][DependComputeTask] failed for Node[%s(%s)].",
                    node_name_.c_str(), node_type_.c_str());

  // build fwk_op_kernel.
  GE_CHECK_GE(kernel_ex_def.args().size(), kernel_ex_def.args_size());
  GE_IF_BOOL_EXEC(kernel_ex_def.args_size() > sizeof(STR_FWK_OP_KERNEL),
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] sizeof STR_FWK_OP_KERNEL is: %zu, but args_size is: %u",
                                     node_name_.c_str(), node_type_.c_str(),
                                     sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args_size());
                  GELOGE(FAILED, "[Check][Size] Node[%s(%s)] sizeof STR_FWK_OP_KERNEL is: %zu, but args_size is: %u",
                         node_name_.c_str(), node_type_.c_str(), sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args_size());
                  return FAILED);
  STR_FWK_OP_KERNEL fwk_op_kernel = {};
  const errno_t sec_ret = memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                                   kernel_ex_def.args().data(), static_cast<size_t>(kernel_ex_def.args_size()));
  GE_CHK_BOOL_RET_STATUS(sec_ret == EOK, INTERNAL_ERROR,
                         "[Update][FwkOpKernel] failed for Node[%s(%s)], ret:%d.",
                         node_name_.c_str(), node_type_.c_str(), sec_ret);

  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = PtrToValue(kernel_workspace_->GetData());
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = PtrToValue(input_output_addr_->GetData());

  if (ext_info_addr_dev_ != nullptr) {
    // set ext info addr and ext info num
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = PtrToValue(ext_info_addr_dev_->GetData());
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = ext_info_addr_dev_->GetSize();
  }

  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = GetStepIdAddr(model);
  fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = AicpuExtInfoHandler::GenerateKernelId();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = model.GetSessionId();

  const auto session_id = fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID;
  GE_CHK_STATUS_RET(EnsureSessionCreated(session_id),
                    "[Invoke][EnsureSessionCreated] Node[%s(%s)] create session id %lu failed.",
                    node_name_.c_str(), node_type_.c_str(), session_id);
  // Assemble kernel_buf_
  GE_CHK_STATUS_RET(AssembleKernelBuffer(&fwk_op_kernel), "[Assemble][ernelBuffer] failed for node[%s(%s)] .",
                    node_name_.c_str(), node_type_.c_str());
  const auto node_type = NodeUtils::GetNodeType(node_item_->node);
  if (ge::hybrid::CanSyncStream(op_desc, node_type)) {
    GELOGD("[%s] set need sync to true, node type = %s", node_name_.c_str(), node_type.c_str());
    need_sync_ = true;
  }
  const auto task_defs = model.GetTaskDefs(node_item_->node);
  GE_CHECK_NOTNULL(task_defs);
  if ((unknown_type_ == DEPEND_COMPUTE) && node_item_->is_dynamic) {
    GE_CHK_STATUS_RET_NOLOG(SetMemCopyTask((*task_defs).back()));
  }
  GELOGI("Node[%s] init end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::SetMemCopyTask(const domi::TaskDef &task_def) {
  if (node_item_->num_outputs == 0) {
    GELOGD("Node[%s] type[%s] has no output, no need set mem_copy task.",
           node_name_.c_str(), node_item_->node_type.c_str());
    return SUCCESS;
  }

  GELOGD("Start to set memcpy task for node[%s].", node_name_.c_str());
  const domi::KernelExDef &kernel_def = task_def.kernel_ex();
  GE_CHECK_GE(kernel_def.args().size(), kernel_def.args_size());
  if (kernel_def.args_size() > sizeof(STR_FWK_OP_KERNEL)) {
    GELOGE(PARAM_INVALID, "[Check][Size]sizeof STR_FWK_OP_KERNEL is:%lu, but args_size:%u is bigger",
           sizeof(STR_FWK_OP_KERNEL), kernel_def.args_size());
    REPORT_INNER_ERR_MSG("E19999", "sizeof STR_FWK_OP_KERNEL is:%" PRIu64 ", but args_size:%u is bigger.",
                       static_cast<uint64_t>(sizeof(STR_FWK_OP_KERNEL)), kernel_def.args_size());
    return PARAM_INVALID;
  }
  STR_FWK_OP_KERNEL aicpu_task = {};
  const auto sec_ret = memcpy_s(&aicpu_task, sizeof(STR_FWK_OP_KERNEL),
                                kernel_def.args().data(), static_cast<size_t>(kernel_def.args_size()));
  if (sec_ret != EOK) {
    GELOGE(FAILED, "[Update][aicpu_task] failed, ret: %d", sec_ret);
    REPORT_INNER_ERR_MSG("E19999", "update aicpu_task failed, ret: %d.", sec_ret);
    return FAILED;
  }

  GE_CHECK_GE(kernel_def.task_info().size(), kernel_def.task_info_size());
  GE_CHK_STATUS_RET(AllocTensorBuffer(static_cast<size_t>(kernel_def.task_info_size()), copy_workspace_buf_),
                    "[Alloc][TensorBuffer] for Node[%s(%s)] to copy task workspace buf, size=%u.", node_name_.c_str(),
                    node_type_.c_str(), kernel_def.task_info_size());

  GE_CHK_RT_RET(rtMemcpy(copy_workspace_buf_->GetData(), static_cast<uint64_t>(kernel_def.task_info_size()),
                         kernel_def.task_info().data(), static_cast<uint64_t>(kernel_def.task_info_size()),
                         memcpy_kind_));
  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = PtrToValue(copy_ioaddr_dev_->GetData());
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = PtrToValue(copy_workspace_buf_->GetData());
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0U;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoLen = 0U;
  GE_CHK_RT_RET(rtMemcpy(copy_task_args_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL), &aicpu_task,
                         sizeof(STR_FWK_OP_KERNEL), memcpy_kind_));
  GELOGD("Set memcpy task for node[%s] successfully.", node_name_.c_str());
  return SUCCESS;
}

uint64_t AicpuTfNodeTask::GetStepIdAddr(const HybridModel &model) {
  // get step_id_addr
  const auto var_tensor = model.GetGlobalStep();
  uint64_t step_id_addr = 0U;
  if (var_tensor != nullptr) {
    step_id_addr = PtrToValue(var_tensor);
  }
  return step_id_addr;
}

Status AicpuTfNodeTask::EnsureSessionCreated(const uint64_t session_id) {
  GE_CHK_STATUS_RET(ModelManager::GetInstance().CreateAicpuSession(session_id),
                    "[Create][AicpuSession] failed, session_id:%lu", session_id);
  return SUCCESS;
}

Status AicpuNodeTaskBase::AllocOutputBuffer(const TaskContext &context, const int32_t idx,
                                            const uint64_t data_size) const {
  const auto &tensor = context.GetOutput(idx);
  GE_CHECK_NOTNULL(tensor);
  if (tensor->GetData() != nullptr) {
    const auto allocated_size = static_cast<uint64_t>(tensor->GetSize());
    if (data_size > allocated_size) {
      GELOGE(GRAPH_PARAM_INVALID,
             "[Check][Size] %s(%s) index[%d] mem size out of range! Expected size: %lu, but given input size: %lu.",
             node_name_.c_str(), node_type_.c_str(), idx, data_size, allocated_size);

      std::string reason = "The memory " + std::to_string(data_size) + " required by the output " + std::to_string(idx) +
                           " of the node " + node_name_ + "(" + node_type_ + ") is greater than the allocated memory " +
                           std::to_string(allocated_size);
      REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                std::vector<const char_t *>({reason.c_str()}));
      return GRAPH_PARAM_INVALID;
    }
    return SUCCESS;
  }

  std::unique_ptr<TensorBuffer> tensor_buffer;
  const auto allocator = NpuMemoryAllocator::GetAllocator(context.GetStream());
  GE_CHK_STATUS_RET(AllocTensorBuffer(data_size, tensor_buffer, allocator),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] out[%d] to copy tensor buffer, data_size:%lu",
                    node_name_.c_str(), node_type_.c_str(), idx, data_size);
  GE_CHK_STATUS_RET(context.SetOutput(idx, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer.release()))),
                    "[Set][Output] failed for Node[%s(%s)], output:%d.",
                    node_name_.c_str(), node_type_.c_str(), idx);
  return SUCCESS;
}

Status AicpuNodeTaskBase::ReadResultSummaryAndPrepareMemory(const TaskContext &context,
                                                            std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    auto &result_summary = output_summary_host_[static_cast<size_t>(i)];
    GE_CHK_RT_RET(rtMemcpy(&result_summary, sizeof(aicpu::FWKAdapter::ResultSummary),
                           output_summary_[static_cast<size_t>(i)]->GetData(),
                           output_summary_[static_cast<size_t>(i)]->GetSize(), callback_memcpy_kind_));
    const auto raw_data_size = result_summary.raw_data_size;
    GE_CHK_STATUS_RET(AllocOutputBuffer(context, i, raw_data_size),
                      "[Alloc][TensorBuffer] failed for Node[%s(%s)] index[%d] output buffer, raw_data_size:%lu",
                      node_name_.c_str(), node_type_.c_str(), i, raw_data_size);

    const auto shape_data_size = result_summary.shape_data_size;
    std::unique_ptr<TensorBuffer> shape_buffer;
    const auto allocator = NpuMemoryAllocator::GetAllocator();
    GE_CHK_STATUS_RET(AllocTensorBuffer(shape_data_size, shape_buffer, allocator),
                      "[Alloc][TensorBuffer] failed for Node[%s(%s)] out[%d] to copy shape buffer, shape_data_size:%lu",
                      node_name_.c_str(), node_type_.c_str(), i, shape_data_size);
    out_shape_hbm.emplace_back(std::move(shape_buffer));
  }
  return SUCCESS;
}

Status AicpuNodeTask::CopyDataToHbm(TaskContext &context,
                                    const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == static_cast<std::size_t>(node_item_->num_outputs),
                         INTERNAL_ERROR,
                         "[Check][Size] Node[%s(%s)] has %d outputs but out shape is %zu not equal.",
                         node_name_.c_str(), node_type_.c_str(), node_item_->num_outputs, out_shape_hbm.size());

  GE_CHK_STATUS_RET_NOLOG(PrepareCopyInputs(context, out_shape_hbm));

  rtArgsEx_t args_ex = {};
  args_ex.args = memcpy_args_.get();
  args_ex.argsSize = memcpy_args_size_;
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[LaunchCopy] Start");
  const auto rt_ret = rtCpuKernelLaunchWithFlag(PtrToPtr<const char, const void>(memcpy_so_name_.c_str()),
                                                PtrToPtr<const char, const void>(memcpy_kernel_name_.c_str()),
                                                1U,  // default core dim is 1
                                                &args_ex,
                                                nullptr, context.GetStream(), RT_KERNEL_DEFAULT);
  GE_CHK_RT_RET(rt_ret);

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[LaunchCopy] End");

  HYBRID_CHK_STATUS_RET(rtStreamSynchronizeWithTimeout(
      context.GetStream(), static_cast<int32_t>(std::strtol(stream_sync_timeout_.c_str(), nullptr, kDefaultBase))));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[SynchronizeCopy] End");

  // save profiling data
  GE_CHK_STATUS_RET(context.SaveProfilingTaskDescInfo(kTaskTypeAicpu, 0U, node_type_),
                    "[Save][Profiling] failed for node[%s]!", context.GetNodeName());

  return SUCCESS;
}

Status AicpuTfNodeTask::CopyDataToHbm(TaskContext &context,
                                      const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == static_cast<std::size_t>(node_item_->num_outputs),
                         INTERNAL_ERROR,
                         "[Check][Size] Node[%s(%s)] has %d outputs but out shape is %zu not equal.",
                         node_name_.c_str(), node_type_.c_str(), node_item_->num_outputs, out_shape_hbm.size());

  GE_CHK_STATUS_RET_NOLOG(PrepareCopyInputs(context, out_shape_hbm));

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[LaunchCopy] Start");
  GE_CHK_RT_RET(rtKernelLaunchFwk(node_name_.c_str(), copy_task_args_buf_->GetData(), sizeof(STR_FWK_OP_KERNEL),
                                  RT_KERNEL_DEFAULT, context.GetStream()));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[LaunchCopy] End");

  HYBRID_CHK_STATUS_RET(rtStreamSynchronizeWithTimeout(
      context.GetStream(), static_cast<int32_t>(std::strtol(stream_sync_timeout_.c_str(), nullptr, kDefaultBase))));
  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[SynchronizeCopy] End");

  // save profiling data
  GE_CHK_STATUS_RET(context.SaveProfilingTaskDescInfo(kTaskTypeAicpu, 0U, node_type_),
                    "[Save][Profiling] failed for node[%s]!", context.GetNodeName());

  return SUCCESS;
}

Status AicpuNodeTaskBase::PrepareCopyInputs(const TaskContext &context,
                                            const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  std::vector<uint64_t> copy_input_release_flag;
  std::vector<uint64_t> copy_input_data_size;
  std::vector<uint64_t> copy_input_src;
  std::vector<uint64_t> copy_input_dst;

  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    const auto &summary = output_summary_host_[static_cast<size_t>(i)];
    GELOGD("Node[%s] out[%d] summary, shape data=0x%lx, shape data size=%lu, raw data=0x%lx, raw data size=%lu.",
           node_name_.c_str(), i,
           summary.shape_data_ptr, summary.shape_data_size,
           summary.raw_data_ptr, summary.raw_data_size);
    const auto output = context.GetOutput(i);
    GE_CHECK_NOTNULL(output);
    copy_input_release_flag.emplace_back(kReleaseFlag);
    copy_input_data_size.emplace_back(summary.raw_data_size);
    copy_input_src.emplace_back(summary.raw_data_ptr);
    copy_input_dst.emplace_back(PtrToValue(output->GetData()));

    const auto &shape_buffer = out_shape_hbm[static_cast<size_t>(i)];
    GE_CHECK_NOTNULL(shape_buffer);
    copy_input_release_flag.emplace_back(kReleaseFlag);
    copy_input_data_size.emplace_back(summary.shape_data_size);
    copy_input_src.emplace_back(summary.shape_data_ptr);
    copy_input_dst.emplace_back(PtrToValue(shape_buffer->GetData()));
  }

  // copy task need copy all output_data and output_shape, len is 2 * output_num
  const size_t copy_input_buf_len = static_cast<size_t>(node_item_->num_outputs) * 2U * sizeof(uint64_t);
  GE_CHK_RT_RET(rtMemcpy(copy_input_release_flag_dev_->GetData(), copy_input_release_flag_dev_->GetSize(),
                         &copy_input_release_flag[0U], copy_input_buf_len, memcpy_kind_));
  GE_CHK_RT_RET(rtMemcpy(copy_input_data_size_dev_->GetData(), copy_input_data_size_dev_->GetSize(),
                         &copy_input_data_size[0U], copy_input_buf_len, memcpy_kind_));
  GE_CHK_RT_RET(rtMemcpy(copy_input_src_dev_->GetData(), copy_input_src_dev_->GetSize(), &copy_input_src[0U],
                         copy_input_buf_len, memcpy_kind_));
  GE_CHK_RT_RET(rtMemcpy(copy_input_dst_dev_->GetData(), copy_input_dst_dev_->GetSize(), &copy_input_dst[0U],
                         copy_input_buf_len, memcpy_kind_));

  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateShapeByHbmBuffer(const TaskContext &context,
                                                 const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) {
  GE_CHK_BOOL_RET_STATUS(out_shape_hbm.size() == static_cast<std::size_t>(node_item_->num_outputs),
                         INTERNAL_ERROR,
                         "[Check][Param] Node[%s(%s)] has %d outputs but out shape is %zu",
                         node_name_.c_str(), node_type_.c_str(), node_item_->num_outputs, out_shape_hbm.size());
  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    const auto &result_summary = output_summary_host_[static_cast<size_t>(i)];
    std::vector<int64_t> shape_dims;
    if (result_summary.shape_data_size > 0U) {
      const auto &shape_hbm = out_shape_hbm[static_cast<size_t>(i)];
      GE_CHK_BOOL_RET_STATUS(((result_summary.shape_data_size % sizeof(int64_t)) == 0U), INTERNAL_ERROR,
                             "[Check][Size]Node[%s(%s)] [%d]th output shape data size is %" PRIu64 " "
                             "is not divided by int64_t.",
                             node_name_.c_str(), node_type_.c_str(), i, result_summary.shape_data_size);
      const size_t dim_num = static_cast<size_t>(result_summary.shape_data_size) / sizeof(int64_t);
      GELOGD("Node[%s] [%d]th output dim num=%zu.", node_name_.c_str(), i, dim_num);
      const std::unique_ptr<int64_t[]> shape_addr = MakeUnique<int64_t[]>(dim_num);
      GE_CHECK_NOTNULL(shape_addr);
      GE_CHK_RT_RET(rtMemcpy(shape_addr.get(), result_summary.shape_data_size, shape_hbm->GetData(),
                             shape_hbm->GetSize(), callback_memcpy_kind_));
      for (size_t dim_idx = 0U; dim_idx < dim_num; ++dim_idx) {
        shape_dims.emplace_back(shape_addr[dim_idx]);
        GELOGD("Node[%s] [%d]th output dim[%zu]=%ld.", node_name_.c_str(), i, dim_idx, shape_addr[dim_idx]);
      }
    }
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(context, GeShape(shape_dims), i),
                      "[Invoke][UpdateShapeToOutputDesc] Node[%s(%s)] update [%d]th output shape failed.",
                      node_name_.c_str(), node_type_.c_str(), i);
  }
  return SUCCESS;
}

Status AicpuNodeTaskBase::UpdateShapeAndDataByResultSummary(TaskContext &context) {
  GELOGD("Node[%s] update shape and data by result summary begin.", node_name_.c_str());

  std::vector<std::unique_ptr<TensorBuffer>> out_shape_hbm;
  GE_CHK_STATUS_RET(ReadResultSummaryAndPrepareMemory(context, out_shape_hbm),
                    "[Invoke][ReadResultSummaryAndPrepareMemory] failed for Node[%s(%s)].",
                    node_name_.c_str(), node_type_.c_str());

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(),
                        "[ReadResultSummaryAndPrepareMemory] End");

  GE_CHK_STATUS_RET(CopyDataToHbm(context, out_shape_hbm),
                    "[Invoke][CopyDataToHbm] failed for Node[%s(%s)] copy data to output.",
                    node_name_.c_str(), node_type_.c_str());

  RECORD_CALLBACK_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[CopyDataToHbm] End");

  GE_CHK_STATUS_RET(UpdateShapeByHbmBuffer(context, out_shape_hbm),
                    "[Update][ShapeByHbmBuffer] failed for Node[%s(%s)].",
                    node_name_.c_str(), node_type_.c_str());

  GELOGD("Node[%s] update shape and data by result summary end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateHostMemInputArgs(const TaskContext &context, void *const args,
                                               const size_t args_size) {
  if (!need_host_mem_opt_) {
    return SUCCESS;
  }
  std::vector<rtHostInputInfo_t> host_inputs;
  GE_CHK_RT_RET(ExecutorUtils::UpdateHostMemInputArgs(context, args, args_size,
                                                      host_mem_input_data_offset_, host_inputs, true));

  for (auto &host_input : host_inputs) {
    const size_t index = host_input.addrOffset / sizeof(uintptr_t);
    uint64_t *const host_mem_input_index = PtrAdd(PtrToPtr<void, uint64_t>(args), args_size / sizeof(uint64_t), index);
    GE_ASSERT_NOTNULL(host_mem_input_index);
    *host_mem_input_index = PtrToValue(input_output_addr_->GetData()) + host_input.dataOffset;
  }
  return SUCCESS;
}

Status AicpuTfNodeTask::UpdateIoAddr(TaskContext &context) {
  const size_t io_num = static_cast<size_t>(node_item_->num_inputs + node_item_->num_outputs);
  GE_CHECK_LE(io_num * sizeof(uint64_t), input_output_addr_->GetSize());

  const auto args = MakeUnique<uint8_t[]>(input_output_addr_->GetSize());
  GE_CHECK_NOTNULL(args);
  uint64_t *io_addrs = PtrToPtr<uint8_t, uint64_t>(args.get());
  for (int32_t i = 0; i < node_item_->num_inputs; ++i) {
    const auto inputData = context.GetInput(i);
    GE_CHECK_NOTNULL(inputData);
    GELOGD("Node[%s] input[%d] addr = %p, size = %zu, mem_type = %d", node_name_.c_str(), i,
           inputData->GetData(), inputData->GetSize(), inputData->GetMemType());
    *io_addrs = PtrToValue(inputData->GetData());
    io_addrs++;
  }

  // known shape or not depend compute
  if ((!node_item_->is_dynamic) || (unknown_type_ != DEPEND_COMPUTE)) {
    // unknown type 4 do this in call back.
    GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
    for (auto j = 0; j < node_item_->num_outputs; ++j) {
      const auto outputData = context.GetOutput(j);
      GE_CHECK_NOTNULL(outputData);

      GELOGD("Node[%s] output[%d] addr = %p, size = %zu",
             node_name_.c_str(), j, outputData->GetData(), outputData->GetSize());
      *io_addrs = PtrToValue(outputData->GetData());
      io_addrs++;
    }
  } else {
    // unknown type 4 use result summary update ioaddr.
    GELOGD("Node[%s] is depend compute node, use result summary as out addr.", node_name_.c_str());
    GE_CHK_BOOL_RET_STATUS(output_summary_.size() == static_cast<std::size_t>(node_item_->num_outputs),
                           INTERNAL_ERROR,
                           "[Check][Size]Node[%s(%s)] has %d output but %zu output summary not equal.",
                           node_name_.c_str(), node_type_.c_str(), node_item_->num_outputs, output_summary_.size());

    for (int32_t j = 0; j < node_item_->num_outputs; ++j) {
      *io_addrs = PtrToValue(output_summary_[static_cast<size_t>(j)]->GetData());
      io_addrs++;
    }
  }

  if (need_host_mem_opt_) {
    GE_CHK_STATUS_RET(UpdateHostMemInputArgs(context, args.get(), input_output_addr_->GetSize()),
                      "[Update][HostMemInputArgs] failed for Node[%s(%s)].",
                      node_name_.c_str(), node_type_.c_str());
  }

  // if has input and output, need copy to ioaddr
  if (io_num > 0U) {
    // copy input and output
    GE_CHK_RT_RET(rtMemcpy(input_output_addr_->GetData(), input_output_addr_->GetSize(),
                           PtrToPtr<uint8_t, void>(args.get()), input_output_addr_->GetSize(), memcpy_kind_));
  }

  return SUCCESS;
}

Status AicpuTfNodeTask::LaunchTask(TaskContext &context) {
  GELOGD("Node[%s] launch task start, unknown_type=%d.", node_name_.c_str(), unknown_type_);
  uint32_t flag = RT_KERNEL_DEFAULT;
  flag = flag | static_cast<uint32_t>(deploy_type_flag_) | qos_level_flag_;
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[AicpuTfNodertKernelLaunchEx] Start");
  SetTaskTag();
  GE_CHK_RT_RET(rtKernelLaunchFwk(node_name_.c_str(), kernel_buf_->GetData(),
                                  static_cast<uint32_t>(kernel_buf_->GetSize()), flag, context.GetStream()));
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), node_name_.c_str(), "[AicpuTfNodertKernelLaunchEx] End");
  GELOGD("Node[%s] launch end.", node_name_.c_str());
  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp(context.GetStream()) != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] for node:%s(%s) failed",
             node_name_.c_str(), node_type_.c_str());
      return FAILED;
    }
  }
  GE_CHK_STATUS_RET_NOLOG(CheckOverflow(context));
  if (need_sync_) {
    GELOGD("[%s] Task needs sync", node_name_.c_str());
    GE_CHK_STATUS_RET_NOLOG(context.Synchronize());
  }
  return SUCCESS;
}

Status AicpuNodeTaskBase::TaskCallback(TaskContext &context) {
  GELOGD("Node[%s] task callback start. is_dynamic=%s, unknown_type=%d.",
         node_name_.c_str(), node_item_->is_dynamic ? "true" : "false", unknown_type_);
  Status callback_ret = SUCCESS;
  if (node_item_->is_dynamic) {
    // check need update shape, call update shape.
    if (unknown_type_ == DEPEND_SHAPE_RANGE) {
      // check result
      callback_ret = UpdateOutputShapeFromExtInfo(context);
    }
    if (unknown_type_ == DEPEND_COMPUTE) {
      callback_ret = UpdateShapeAndDataByResultSummary(context);
    }
  }
  GELOGD("Node[%s] task callback end.", node_name_.c_str());
  return callback_ret;
}

void AicpuNodeTaskBase::SetTaskTag() const {
  const rtError_t rt_set_tag = rtSetTaskTag(op_name_.c_str());
  if (rt_set_tag != RT_ERROR_NONE) {
    GELOGW("[Call][rtSetTaskTag] failed, ret:0x%X", rt_set_tag);
  }
}

void AicpuNodeTaskBase::InitBlockAicpuOp(const OpDescPtr& op_desc) {
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_SUPPORT_BLOCKDIM_FLAG, is_support_block_);
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_BLOCKDIM_INDEX, axis_index_);
  GELOGD("Node[%s] is_support_block_[%d] axis_index_[%d]", node_name_.c_str(), static_cast<int32_t>(is_support_block_),
         axis_index_);
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op_);
  GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc->GetName().c_str(),
         static_cast<int32_t>(is_blocking_aicpu_op_));
}

Status AicpuNodeTask::SetMemCopyTask(const domi::TaskDef &task_def) {
  if (node_item_->num_outputs == 0) {
    GELOGD("Node[%s] type[%s] has no output, no need set mem_copy task.",
           node_name_.c_str(), node_item_->node_type.c_str());
    return SUCCESS;
  }

  GELOGD("Start to set memcpy task for node[%s].", node_name_.c_str());
  const domi::KernelDef &kernel_def = task_def.kernel();
  auto &memcpy_args = kernel_def.args();
  memcpy_args_size_ = kernel_def.args_size();
  memcpy_so_name_ = kernel_def.so_name();
  memcpy_kernel_name_ = kernel_def.kernel_name();
  if (memcpy_args.size() != memcpy_args_size_) {
    REPORT_INNER_ERR_MSG("E19999", "MemCopy task def args.size = %zu, but args_size = %u not equal.",
                       memcpy_args.size(), memcpy_args_size_);
    GELOGE(FAILED, "[Check][Size] MemCopy task def args.size = %zu, but args_size = %u not equal.",
           memcpy_args.size(), memcpy_args_size_);
    return FAILED;
  }

  if (memcpy_args_size_ < sizeof(aicpu::AicpuParamHead)) {
    REPORT_INNER_ERR_MSG("E19999", "Task def args_size = %u is less than aicpu param head len = %zu.",
                       memcpy_args_size_, sizeof(aicpu::AicpuParamHead));
    GELOGE(FAILED, "[Check][Size] Task def args_size = %u is less than aicpu param head len = %zu.",
           memcpy_args_size_, sizeof(aicpu::AicpuParamHead));
    return FAILED;
  }

  memcpy_args_ = MakeUnique<uint8_t[]>(static_cast<size_t>(memcpy_args_size_));
  if (memcpy_args_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "new memory failed for Node[%s(%s)], task_size[%u].",
                       node_name_.c_str(), node_type_.c_str(), memcpy_args_size_);
    GELOGE(FAILED, "[Malloc][Memory] failed for Node[%s(%s)], task_size[%u].",
           node_name_.c_str(), node_type_.c_str(), memcpy_args_size_);
    return FAILED;
  }

  const errno_t sec_ret = memcpy_s(memcpy_args_.get(), static_cast<size_t>(memcpy_args_size_), memcpy_args.c_str(),
                                   memcpy_args.size());
  if (sec_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "memcpy_s argc_ failed for Node[%s(%s)], ret: %d",
                       node_name_.c_str(), node_type_.c_str(), static_cast<int32_t>(sec_ret));
    GELOGE(INTERNAL_ERROR,
           "[Update][Args] failed for Node[%s(%s)], ret: %d",
           node_name_.c_str(), node_type_.c_str(), static_cast<int32_t>(sec_ret));
    return FAILED;
  }

  aicpu::AicpuParamHead *const memcpy_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(memcpy_args_.get());
  const uint32_t memcpy_io_num = memcpy_param_head->ioAddrNum;
  // if has input and output, need copy to ioaddr
  const errno_t cpy_ret = memcpy_s(&memcpy_args_[sizeof(aicpu::AicpuParamHead)],
                                   static_cast<size_t>(memcpy_args_size_) - sizeof(aicpu::AicpuParamHead),
                                   &copy_io_addr_[0U], sizeof(uint64_t) * memcpy_io_num);
  if (cpy_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] memcpy io addr to AicpuParamHead failed,"
                       "ret = %d, args_size = %u, io nums = %u.",
                       node_name_.c_str(), node_type_.c_str(), cpy_ret, memcpy_args_size_, memcpy_io_num);
    GELOGE(INTERNAL_ERROR, "[Update][IoAddr] Node[%s(%s)] memcpy io addr to AicpuParamHead failed,"
           "ret = %d, args_size = %u, io nums = %u.",
           node_name_.c_str(), node_type_.c_str(), cpy_ret, memcpy_args_size_, memcpy_io_num);
    return INTERNAL_ERROR;
  }
  GELOGD("Set memcpy task for node[MemCopy] successfully.");
  return SUCCESS;
}

Status AicpuNodeTask::InitForDependComputeTask() {
  if ((unknown_type_ != DEPEND_COMPUTE) || (!node_item_->is_dynamic) ||
      (node_item_->num_outputs == 0)) {
    GELOGD("Node[%s] type[%s] unknown_type is %d, output num is %d.",
           node_name_.c_str(), node_item_->node_type.c_str(), unknown_type_, node_item_->num_outputs);
    return SUCCESS;
  }

  output_summary_.resize(static_cast<size_t>(node_item_->num_outputs));
  for (int32_t i = 0; i < node_item_->num_outputs; ++i) {
    GE_CHK_STATUS_RET(AllocTensorBuffer(sizeof(aicpu::FWKAdapter::ResultSummary),
                                        output_summary_[static_cast<size_t>(i)]),
                      "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy result summary info, size = %zu.",
                      node_name_.c_str(), node_type_.c_str(), sizeof(aicpu::FWKAdapter::ResultSummary));
  }
  output_summary_host_.resize(static_cast<size_t>(node_item_->num_outputs));

  // init for mem copy task
  // copy task need copy output_data and output_shape, max len is 2 * output_num
  const size_t copy_input_buf_len = static_cast<size_t>(node_item_->num_outputs) * 2U * sizeof(uint64_t);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_release_flag_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input release_flag, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_data_size_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input data_size, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_src_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input src, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);
  GE_CHK_STATUS_RET(AllocTensorBuffer(copy_input_buf_len, copy_input_dst_dev_),
                    "[Alloc][TensorBuffer] failed for Node[%s(%s)] to copy task input dst, size = %zu",
                    node_name_.c_str(), node_type_.c_str(), copy_input_buf_len);

  copy_io_addr_.emplace_back(PtrToValue(copy_input_release_flag_dev_->GetData()));
  copy_io_addr_.emplace_back(PtrToValue(copy_input_data_size_dev_->GetData()));
  copy_io_addr_.emplace_back(PtrToValue(copy_input_src_dev_->GetData()));
  copy_io_addr_.emplace_back(PtrToValue(copy_input_dst_dev_->GetData()));
  return SUCCESS;
}

Status AicpuNodeTask::InitTopicTypAndExtInfo(const HybridModel &model) {
  GE_CHK_BOOL_RET_STATUS(task_def_.has_kernel(), FAILED, "[Call][HasKernel] Node[%s(%s)] task def does not has kernel.",
                         node_name_.c_str(), node_type_.c_str());
  auto &kernel_def = task_def_.kernel();
  const auto &kernel_ext_info = kernel_def.kernel_ext_info();
  const auto kernel_ext_info_size = kernel_def.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "[Check][Size] Node[%s(%s)] task def kernel_ext_info.size = %zu, "
                         "but kernel_ext_info_size = %u",
                         node_name_.c_str(), node_type_.c_str(), kernel_ext_info.size(), kernel_ext_info_size);
  const uint64_t ext_session_id = model.GetSessionId();
  GE_CHK_STATUS_RET(InitExtInfo(kernel_ext_info, ext_session_id), "[Init][ExtInfo] failed for Node[%s(%s)].",
                    node_name_.c_str(), node_type_.c_str());
  return SUCCESS;
}

Status AicpuNodeTask::Init(const HybridModel &model) {
  GELOGD("Node[%s] init start.", node_name_.c_str());
  // init block info
  const OpDescPtr op_desc = node_item_->GetOpDesc();
  InitBlockAicpuOp(op_desc);
  GE_CHK_STATUS_RET(InitTopicTypAndExtInfo(model), "[Int][TopicTypAndExtInfo] for Node[%s(%s)] failed.",
                    node_name_.c_str(), node_type_.c_str());
  auto &kernel_def = task_def_.kernel();

  auto &args = kernel_def.args();
  args_size_ = kernel_def.args_size();

  const std::string &so_name = kernel_def.so_name();
  const auto &context = kernel_def.context();
  const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    bool loaded = false;
    GE_CHK_STATUS_RET(ModelManager::GetInstance().LoadCustAicpuSo(op_desc, so_name, loaded),
                      "[Load][CustAicpuSo] failed, op:%s(%s), so:%s.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), so_name.c_str());
    if (!loaded) {
      GE_CHK_STATUS_RET(ModelManager::GetInstance().LaunchCustAicpuSo(),
                        "[Launch][CustAicpuSo] failed, node:%s(%s).", node_name_.c_str(), node_type_.c_str());
    }
  }

  GE_IF_BOOL_EXEC(args.size() != args_size_,
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] task def args.size = %zu, but args_size = %u not equal.",
                                     node_name_.c_str(), node_type_.c_str(), args.size(), args_size_);
                  GELOGE(FAILED, "[Check][Size] Node[%s(%s)] task def args.size = %zu, but args_size = %u not equal.",
                         node_name_.c_str(), node_type_.c_str(), args.size(), args_size_);
                  return FAILED);

  GE_IF_BOOL_EXEC(args_size_ < sizeof(aicpu::AicpuParamHead),
                  REPORT_INNER_ERR_MSG("E19999",
                                     "Node[%s(%s)] task def args_size = %u is less than aicpu param head len = %zu.",
                                     node_name_.c_str(), node_type_.c_str(), args_size_, sizeof(aicpu::AicpuParamHead));
                  GELOGE(FAILED,
                         "[Check][Size] Node[%s(%s)] task def args_size = %u is less than aicpu param head len = %zu.",
                         node_name_.c_str(), node_type_.c_str(), args_size_, sizeof(aicpu::AicpuParamHead));
                  return FAILED);

  if (ExecutorUtils::HasHostMemInput(op_desc)) {
    host_mem_input_data_offset_ = static_cast<size_t>(args_size_);
    args_size_ += static_cast<uint32_t>(kMaxHostMemInputLen);
    GELOGD("Node[%s(%s)] has host memory input, args size is extended %zu, args_size_ = %u,"
        " host_mem_input_data_offset = %zu.", node_name_.c_str(), node_type_.c_str(), kMaxHostMemInputLen, args_size_,
        host_mem_input_data_offset_);
  }

  args_ = MakeUnique<uint8_t[]>(static_cast<size_t>(args_size_));
  GE_IF_BOOL_EXEC(args_ == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "new memory failed for Node[%s(%s)], args_size_ = %u.",
                                     node_name_.c_str(), node_type_.c_str(), args_size_);
                  GELOGE(FAILED, "[Malloc][Memory] failed for Node[%s(%s)], args_size_ = %u.",
                         node_name_.c_str(), node_type_.c_str(), args_size_);
                  return FAILED);
  args_ex_.args = args_.get();
  args_ex_.argsSize = args_size_;
  const errno_t sec_ret = memcpy_s(args_.get(), static_cast<size_t>(args_size_), args.c_str(), args.size());
  GE_IF_BOOL_EXEC(sec_ret != EOK,
                  REPORT_INNER_ERR_MSG("E19999", "memcpy_s argc_ failed for Node[%s(%s)], ret: %d",
                                     node_name_.c_str(), node_type_.c_str(), sec_ret);
                  GELOGE(INTERNAL_ERROR, "[Update][Args] failed for Node[%s(%s)], ret: %d",
                         node_name_.c_str(), node_type_.c_str(), sec_ret);
                  return FAILED);

  aicpu::AicpuParamHead *const aicpu_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(args_.get());
  const auto io_num = node_item_->num_inputs + node_item_->num_outputs;

  // check AicpuParamHead ioAddrNum is right.
  GE_IF_BOOL_EXEC((aicpu_param_head->ioAddrNum != static_cast<uint32_t>(io_num)),
                  REPORT_INNER_ERR_MSG("E19999",
                                     "Node[%s(%s)] param head ioAddrNum = %u, but node has %d inputs and %d outputs.",
                                     node_name_.c_str(), node_type_.c_str(), aicpu_param_head->ioAddrNum,
                                     node_item_->num_inputs, node_item_->num_outputs);
                  GELOGE(PARAM_INVALID, "[Check][IoAddrNum] Node[%s(%s)] param head ioAddrNum = %u, "
                         "but node has %d inputs and %d outputs.",
                         node_name_.c_str(), node_type_.c_str(), aicpu_param_head->ioAddrNum,
                         node_item_->num_inputs, node_item_->num_outputs);
                  return PARAM_INVALID;);

  const auto mini_len = sizeof(aicpu::AicpuParamHead) + (static_cast<size_t>(io_num) * sizeof(uint64_t));
  // check args len must over mini len.
  GE_CHK_BOOL_RET_STATUS((mini_len <= aicpu_param_head->length), PARAM_INVALID,
                         "[Check][DataLen] Node[%s(%s)] param head length = %u, but min len need %zu.",
                         node_name_.c_str(), node_type_.c_str(), aicpu_param_head->length, mini_len);
  GE_CHK_STATUS_RET(InitForDependComputeTask(), "[Init][DependComputeTask] failed for Node[%s(%s)].",
                    node_name_.c_str(), node_type_.c_str());
  if (ext_info_addr_dev_ == nullptr) {
    aicpu_param_head->extInfoLength = 0U;
    aicpu_param_head->extInfoAddr = 0U;
  } else {
    aicpu_param_head->extInfoLength = ext_info_addr_dev_->GetSize();
    aicpu_param_head->extInfoAddr = PtrToValue(ext_info_addr_dev_->GetData());
  }

  const auto task_defs = model.GetTaskDefs(node_item_->node);
  GE_CHECK_NOTNULL(task_defs);
  if ((unknown_type_ == DEPEND_COMPUTE) && node_item_->is_dynamic) {
    GE_CHK_STATUS_RET_NOLOG(SetMemCopyTask((*task_defs).back()));
  }
  GELOGD("Node[%s] init end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuNodeTask::UpdateHostMemInputArgs(const TaskContext &context) {
  args_ex_.hostInputInfoPtr = nullptr;
  args_ex_.hostInputInfoNum = 0U;
  if (!need_host_mem_opt_) {
    return SUCCESS;
  }
  vector<rtHostInputInfo_t> host_inputs;
  GE_CHECK_LE(sizeof(aicpu::AicpuParamHead), host_mem_input_data_offset_);
  GE_CHECK_LE(host_mem_input_data_offset_, args_size_);
  GE_CHK_RT_RET(ExecutorUtils::UpdateHostMemInputArgs(context,
                                                      &args_[sizeof(aicpu::AicpuParamHead)],
                                                      args_size_ - sizeof(aicpu::AicpuParamHead),
                                                      host_mem_input_data_offset_ - sizeof(aicpu::AicpuParamHead),
                                                      host_inputs));

  host_inputs_info_ = MakeUnique<rtHostInputInfo_t[]>(host_inputs.size());
  GE_CHECK_NOTNULL(host_inputs_info_);
  size_t idx = 0U;
  for (auto &host_input : host_inputs) {
    host_input.dataOffset = host_input.dataOffset + static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead));
    host_input.addrOffset = host_input.addrOffset + static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead));
    host_inputs_info_[idx++] = host_input;
  }
  args_ex_.hostInputInfoPtr = host_inputs_info_.get();
  args_ex_.hostInputInfoNum = host_inputs.size();
  return SUCCESS;
}

Status AicpuNodeTask::UpdateIoAddr(TaskContext &context) {
  uint64_t *io_addrs = PtrToPtr<uint8_t, uint64_t>(&args_[sizeof(aicpu::AicpuParamHead)]);
  GE_CHECK_LE((((static_cast<size_t>(node_item_->num_inputs + node_item_->num_outputs)) * sizeof(uint64_t)) +
              sizeof(aicpu::AicpuParamHead)), args_size_);
  for (int32_t i = 0; i < node_item_->num_inputs; ++i) {
    const auto inputData = context.GetInput(i);
    GE_CHECK_NOTNULL(inputData);

    GELOGD("Node[%s] input[%d] = %p, size = %zu", node_name_.c_str(), i, inputData->GetData(), inputData->GetSize());
    *io_addrs = PtrToValue(inputData->GetData());
    io_addrs++;
  }

  // known shape or not depend compute
  if ((!node_item_->is_dynamic) || (unknown_type_ != DEPEND_COMPUTE)) {
    // unknown type 4 do this in call back.
    GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
    for (int32_t j = 0; j < node_item_->num_outputs; ++j) {
      const auto outputData = context.GetOutput(j);
      GE_CHECK_NOTNULL(outputData);
      GELOGD("Node[%s] output[%d] addr = %p, size = %zu",
             node_name_.c_str(), j, outputData->GetData(), outputData->GetSize());
      *io_addrs = PtrToValue(outputData->GetData());
      io_addrs++;
    }
  } else {
    // unknown type 4 use result summary update ioaddr.
    GELOGD("Node[%s] is depend compute node, use result summary as out addr.", node_name_.c_str());
    GE_CHK_BOOL_RET_STATUS(output_summary_.size() == static_cast<std::size_t>(node_item_->num_outputs),
                           INTERNAL_ERROR,
                           "[Check][Size] Node[%s(%s)] has %d output but %zu output summary not equal.",
                           node_name_.c_str(), node_type_.c_str(), node_item_->num_outputs, output_summary_.size());

    for (int32_t j = 0; j < node_item_->num_outputs; ++j) {
      *io_addrs = PtrToValue(output_summary_[static_cast<size_t>(j)]->GetData());
      io_addrs++;
    }
  }

  if (need_host_mem_opt_) {
    GE_CHK_STATUS_RET(UpdateHostMemInputArgs(context),
                      "[Update][HostMemInputArgs] failed for Node[%s(%s)].",
                      node_name_.c_str(), node_type_.c_str());
  }
  return SUCCESS;
}

Status AicpuNodeTask::CheckOverflow(TaskContext &context) const {
  const DumpProperties &dump_properties = context.GetDumpProperties();
  if (dump_properties.IsOpDebugOpen()) {
    GELOGD("Op %s is doing overflow check in hybrid engine", context.GetNodeName());
    const auto rt_ret = rtStreamSynchronizeWithTimeout(
        context.GetStream(), static_cast<int32_t>(std::strtol(stream_sync_timeout_.c_str(), nullptr, kDefaultBase)));
    // AICPU is responsible for dump itself. This code is only reserved code for future solution.
    if (rt_ret == ACL_ERROR_RT_OVER_FLOW) {
      context.SetOverFlow(true);
      (void)rtsGetThreadLastTaskId(context.MutableTaskId());
      (void)rtsStreamGetId(context.GetStream(), reinterpret_cast<int32_t*>(context.MutableStreamId()));
      GELOGW("Dynamic shape op %s is over flow", context.GetNodeName());
      return SUCCESS;
    }
    if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      GELOGE(rt_ret, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, ret:%d.", rt_ret);
      return FAILED;
    }
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Invoke][RtStreamSynchronize] failed, ret:%d.", rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronize failed, ret:%d.", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    return SUCCESS;
  }
  GELOGD("Opdebug is not open in hybrid engine");
  return SUCCESS;
}

Status AicpuNodeTask::LaunchTask(TaskContext &context) {
  GELOGD("Node[%s] launch task start. unknown_type=%d.", node_name_.c_str(), unknown_type_);
  const auto &so_name = task_def_.kernel().so_name();
  const auto &kernel_name = task_def_.kernel().kernel_name();
  const auto &kernel_context = task_def_.kernel().context();
  const auto kernel_type = static_cast<ccKernelType>(kernel_context.kernel_type());
  uint32_t flag = RT_KERNEL_DEFAULT;
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    flag |= static_cast<uint32_t>(RT_KERNEL_CUSTOM_AICPU);
  }
  flag = flag | static_cast<uint32_t>(deploy_type_flag_) | qos_level_flag_;
  const rtKernelLaunchNames_t launch_name = {so_name.c_str(), kernel_name.c_str(), node_name_.c_str()};
  SetTaskTag();
  // default core dim is 1
  const auto rt_ret = rtAicpuKernelLaunchWithFlag(&launch_name, block_num_, &args_ex_, nullptr,
                                                  context.GetStream(), flag);
  GE_CHK_RT_RET(rt_ret);
  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp(context.GetStream()) != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] failed for node:%s(%s)",
             node_name_.c_str(), node_type_.c_str());
      return FAILED;
    }
  }
  GE_CHK_STATUS_RET_NOLOG(CheckOverflow(context));
  GELOGD("Node[%s] launch task end.", node_name_.c_str());
  return SUCCESS;
}

Status AiCpuNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  // malloc HBM memory at Init, here just update them
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCpuNodeExecutorPrepareTask] Start");
  const Status status = task.UpdateArgs(context);
  RECORD_EXECUTION_EVENT(context.GetExecutionContext(), context.GetNodeName(), "[AiCpuNodeExecutorPrepareTask] End");
  return status;
}

Status AiCpuNodeExecutor::LoadTask(const HybridModel &model,
                                   const NodePtr &node,
                                   std::shared_ptr<NodeTask> &task) const {
  GE_CHECK_NOTNULL(node);
  GELOGD("Node[%s] load task start.", node->GetName().c_str());
  auto node_item = model.GetNodeItem(node);
  GE_CHECK_NOTNULL(node_item);
  const auto task_defs = model.GetTaskDefs(node);
  GE_CHECK_NOTNULL(task_defs);
  if ((node_item->shape_inference_type == DEPEND_COMPUTE) && node_item->is_dynamic) {
    // when the operator is the fourth type, and corresponding node is unknown, then 2 tasks are required.
    GE_CHK_BOOL_RET_STATUS((*task_defs).size() == 2UL, PARAM_INVALID,
                           "[Check][Size]Node[%s(%s)] task_def num[%zu] != 2",
                           node->GetName().c_str(), node->GetType().c_str(), (*task_defs).size());
  } else {
    GE_CHK_BOOL_RET_STATUS((*task_defs).size() == 1UL, PARAM_INVALID,
                           "[Check][Size]Node[%s(%s)] task_def num[%zu] != 1",
                           node->GetName().c_str(), node->GetType().c_str(), (*task_defs).size());
  }
  const auto &task_def = (*task_defs)[0UL];
  std::shared_ptr<AicpuNodeTaskBase> aicpu_task;
  if (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_KERNEL_EX) {
    GELOGI("Node[%s] task type=%u is AicpuTfNodeTask.", node->GetName().c_str(), task_def.type());
    aicpu_task = MakeShared<AicpuTfNodeTask>(node_item, task_def);
  } else if (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    GELOGI("Node[%s] task type=%u is AicpuNodeTask.", node->GetName().c_str(), task_def.type());
    aicpu_task = MakeShared<AicpuNodeTask>(node_item, task_def);
  } else {
    GELOGE(UNSUPPORTED, "[Check][Type] Node[%s(%s)] task type = %u is not supported by aicpu node executor,"
           "ModelTaskType::MODEL_TASK_KERNEL_EX or ModelTaskType::MODEL_TASK_KERNEL is supported.",
           node->GetName().c_str(), node->GetType().c_str(), task_def.type());
    REPORT_INNER_ERR_MSG("E19999", "Node[%s(%s)] task type = %u is not supported by aicpu node executor,"
                       "ModelTaskType::MODEL_TASK_KERNEL_EX or ModelTaskType::MODEL_TASK_KERNEL is supported.",
                       node->GetName().c_str(), node->GetType().c_str(), task_def.type());
    return UNSUPPORTED;
  }

  GE_CHK_BOOL_RET_STATUS(aicpu_task != nullptr, MEMALLOC_FAILED,
                         "[Check][State]Load task for node %s(%s) failed.",
                         node->GetName().c_str(), node->GetType().c_str());

  GE_CHK_STATUS_RET(aicpu_task->Init(model),
                    "[Init][AicpuNodeTaskBase] failed for Node[%s(%s)].",
                    node->GetName().c_str(), node->GetType().c_str());

  task = std::move(aicpu_task);
  GELOGD("Node[%s] load task end.", node->GetName().c_str());
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
