/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "stream_merge_op.h"

#include "op_factory.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"
#include "../acl_rt_memcpy_kind.h"
constexpr uint64_t MAX_MEMCPY_SIZE_OF_D2D = 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4G

using namespace ge;
namespace cce {
namespace runtime {
StreamMergeOp::StreamMergeOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status StreamMergeOp::Init() {
  RTS_LOGI("StreamMergeOp Init, node:%s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();

  return Op::Init();
}

Status StreamMergeOp::Run(vector<TaskDef> &tasks) {
  if (v_output_data_addr_.empty()) {
    RTS_REPORT_CALL_ERROR("Stream merge op run failed, thers is no output data address!");
    return INTERNAL_ERROR;
  }

  int64_t maxInputSize = 0;
  for (const auto &inputDesc : op_desc_->GetAllInputsDescPtr()) {
    int64_t currentInputSize = 0;
    if (TensorUtils::GetTensorSizeInBytes(*inputDesc, currentInputSize) != GRAPH_SUCCESS) {
      RTS_LOGI("Ignore tensor size for node %s.", op_desc_->GetName().c_str());
      continue;
    }
    maxInputSize = std::max(maxInputSize, currentInputSize);
  }

  RTS_LOGI("StreamMergeOp run, outSize:%" PRId64 ", copy size: %" PRId64, v_output_size_[0], maxInputSize);
  if (maxInputSize == 0) {
    RTS_LOGI("empty tensor don't emerge taskdef.");
    return SUCCESS;
  }

  const uint64_t sqSize = MAX_MEMCPY_SIZE_OF_D2D;  // 4G
  uint64_t realSize = maxInputSize;
  uint64_t remainSize = maxInputSize;
  uint64_t doneSize = 0U;
  const uint32_t streamId = op_desc_->GetStreamId();
  const uint32_t opIndex = static_cast<uint32_t>(op_desc_->GetId());
  while (remainSize > 0U) {
    const uint64_t doingSize = (remainSize >= sqSize) ? sqSize : remainSize;
    realSize = doingSize;
    domi::TaskDef taskDef = {};
    taskDef.set_type(ACL_RT_MODEL_TASK_MEMCPY_ASYNC);
    taskDef.set_stream_id(streamId);
    domi::MemcpyAsyncDef *memcpyDef = taskDef.mutable_memcpy_async();
    memcpyDef->set_op_index(opIndex);
    memcpyDef->set_dst(reinterpret_cast<uintptr_t>(v_output_data_addr_[0]) + doneSize);
    memcpyDef->set_dst_max(doingSize);
    memcpyDef->set_src(reinterpret_cast<uintptr_t>(v_input_data_addr_[0]) + doneSize);
    memcpyDef->set_count(doingSize);
    memcpyDef->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    tasks.push_back(taskDef);
    doneSize += realSize;
    remainSize -= realSize;
  }

  const Status retCode = UpdateOutDescFromInDesc(v_input_data_addr_[0], v_output_data_addr_[0], tasks);
  if (retCode != SUCCESS) {
    RTS_LOGE("Update output desc failed, ret=%d.", retCode);
  }

  return retCode;
}

}  // namespace runtime
}  // namespace cce
