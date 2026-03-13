/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "memcpy_addr_async_op.h"

#include "common/util.h"
#include "op_factory.h"
#include "graph/utils/tensor_utils.h"
#include "graph/args_format_desc.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"
#include "../acl_rt_memcpy_kind.h"
constexpr uint64_t MAX_MEMCPY_SIZE_OF_D2D = 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4G
#define RT_ERROR_INVALID_VALUE 0x07110001
using namespace ge;
namespace cce {
namespace runtime {
MemcpyAddrAsyncOp::MemcpyAddrAsyncOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status MemcpyAddrAsyncOp::Init() {
  RTS_LOGI("MemcpyAddrAsyncOp Init, node:%s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();
  return Op::Init();
}

Status MemcpyAddrAsyncOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("MemcpyAddrAsync Op Init start, node:%s.", name_.c_str());
  if (v_output_data_addr_.size() != v_input_data_addr_.size() || v_output_data_addr_.size() != v_output_size_.size() ||
      v_input_data_addr_.size() != v_input_size_.size()) {
    RTS_REPORT_CALL_ERROR(
        "Memcpy address async op run failed, input num(%zu) input data addr num(%zu) output num(%zu)"
        " output addr num(%zu) invalid, node:%s.",
        v_input_size_.size(), v_input_data_addr_.size(), v_output_size_.size(), v_output_data_addr_.size(),
        name_.c_str());
    return INTERNAL_ERROR;
  }

  for (size_t index = 0; index < v_input_data_addr_.size(); index++) {
    int64_t inputSize = 0;
    auto inputDesc = op_desc_->GetInputDesc(index);
    if (TensorUtils::GetTensorSizeInBytes(inputDesc, inputSize) != GRAPH_SUCCESS) {
      RTS_LOGI("Ignore tensor size for node %s", op_desc_->GetName().c_str());
      continue;
    }

    const uint64_t sqSize = MAX_MEMCPY_SIZE_OF_D2D;  // 4G
    uint64_t realSize = inputSize;
    uint64_t remainSize = inputSize;
    uint64_t doneSize = 0U;
    while (remainSize > 0U) {
      const uint64_t doingSize = (remainSize >= sqSize) ? sqSize : remainSize;
      realSize = doingSize;
      domi::TaskDef taskDef = {};
      taskDef.set_type(ACL_RT_MODEL_TASK_MEMCPY_ADDR_ASYNC);
      taskDef.set_stream_id(op_desc_->GetStreamId());
      domi::MemcpyAsyncDef *memcpyDef = taskDef.mutable_memcpy_async();
      memcpyDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
      memcpyDef->set_dst(reinterpret_cast<uintptr_t>(v_output_data_addr_[index]) + doneSize);
      memcpyDef->set_dst_max(doingSize);
      memcpyDef->set_src(reinterpret_cast<uintptr_t>(v_input_data_addr_[index]) + doneSize);
      memcpyDef->set_count(doingSize);
      memcpyDef->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
      const rtError_t ret = AddArgsFormatDescInfo(memcpyDef, doingSize);
      if (ret != RT_ERROR_NONE) {
        return ret;
      }
      tasks.push_back(taskDef);
      doneSize += realSize;
      remainSize -= realSize;
    }
    RTS_LOGI("MemcpyAddrAsync Op Init Loop, index=%u, stream_id=%u", index, op_desc_->GetStreamId());
  }

  return SUCCESS;
}

rtError_t MemcpyAddrAsyncOp::AddArgsFormatDescInfo(domi::MemcpyAsyncDef *const memcpyAsyncDef, const uint64_t count) {
  const int32_t socVersionLen = 50;
  char_t version[socVersionLen] = {0};
  auto ret = GetSocVersion(version, socVersionLen);
  if (ret != SUCCESS) {
    RTS_LOGE("GetSocVersion failed, ret=%#x", ret);
    return RT_ERROR_INVALID_VALUE;
  }
  RTS_LOGI("Soc version is [%s]", version);
  std::string res;
  ArgsFormatDesc desc;
  if ((strncmp(version, "Ascend950", strlen("Ascend950")) == 0)) {
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.Append(AddrType::INPUT_INSTANCE, 0);
    desc.Append(AddrType::OUTPUT_INSTANCE, 0);
    desc.AppendCustomValue(count, ArgsFormatWidth::BIT32);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT32);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    res = desc.ToString();
  } else {
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.Append(AddrType::INPUT_INSTANCE, 0);
    desc.Append(AddrType::OUTPUT_INSTANCE, 0);
    res = desc.ToString();
  }
  memcpyAsyncDef->set_args_format(res);

  return RT_ERROR_NONE;
}
// function of op {ENTER, LOOPCOND, NEXTITERATION, EXIT} is similar: pass through data without change
}  // namespace runtime
}  // namespace cce
