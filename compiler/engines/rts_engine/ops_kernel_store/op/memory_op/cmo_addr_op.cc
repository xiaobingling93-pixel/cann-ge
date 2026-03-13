/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmo_addr_op.h"

#include "op_factory.h"
#include "graph/utils/tensor_utils.h"
#include "graph/args_format_desc.h"
#include "common/util.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
CmoAddrOp::CmoAddrOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status CmoAddrOp::Init() {
  RTS_LOGI("CmoAddrOp Init, node:%s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();
  return Op::Init();
}

Status CmoAddrOp::Run(vector<TaskDef> &tasks) {
  // Get rtCmoAddrTaskLaunch input of Op
  for (size_t index = 0; index < v_input_data_addr_.size(); index++) {
    int32_t cmoOpCode = RT_CMO_RESERVED;

    ge::AttrUtils::GetInt(op_desc_, "type", cmoOpCode);
    RTS_LOGD("CmoAddrOp run, input data addr num(%zu), cmoOpCode:%u, cmoAddrInfo=%p", v_input_data_addr_.size(),
             cmoOpCode, v_input_data_addr_[index]);
    TaskDef taskDef = {};
    taskDef.set_type(ACL_RT_MODEL_TASK_CMO_ADDR);
    taskDef.set_stream_id(op_desc_->GetStreamId());

    const uint32_t opIndex = static_cast<uint32_t>(op_desc_->GetId());
    domi::CmoAddrTaskDef *cmoAddrDef = taskDef.mutable_cmo_addr_task();
    cmoAddrDef->set_op_index(opIndex);
    cmoAddrDef->set_src(reinterpret_cast<uintptr_t>(v_input_data_addr_[index]));
    cmoAddrDef->set_length_inner(0);
    cmoAddrDef->set_num_outer(0);
    cmoAddrDef->set_num_inner(0);
    cmoAddrDef->set_stride_outer(0);
    cmoAddrDef->set_stride_inner(0);
    cmoAddrDef->set_cmo_op_code(cmoOpCode);
    Status ret = AddArgsFormatDescInfo(cmoAddrDef);
    if (ret != SUCCESS) {
      return ret;
    }
    tasks.push_back(taskDef);
  }

  return SUCCESS;
}

Status CmoAddrOp::AddArgsFormatDescInfo(domi::CmoAddrTaskDef *const cmoAddrDef) {
  uint32_t len_inner = 0U;
  Status ret = CalculateLenInner(len_inner);
  if (ret != SUCCESS) {
    RTS_REPORT_INNER_ERROR("CalculateLenInner failed, ret=%#x", ret);
    return ret;
  }

  const int32_t socVersionLen = 50;
  char_t version[socVersionLen] = {0};
  ret = GetSocVersion(version, socVersionLen);
  if (ret != SUCCESS) {
    RTS_REPORT_INNER_ERROR("GetSocVersion failed, ret=%#x", ret);
    return ret;
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
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.AppendCustomValue(len_inner, ArgsFormatWidth::BIT32);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT32);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    res = desc.ToString();
  } else {
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT32);
    desc.AppendCustomValue(len_inner, ArgsFormatWidth::BIT32);
    desc.Append(AddrType::INPUT_INSTANCE, 0);
    desc.AppendPlaceholder(ArgsFormatWidth::BIT64);
    res = desc.ToString();
  }
  cmoAddrDef->set_args_format(res);

  return SUCCESS;
}

Status CmoAddrOp::CalculateLenInner(uint32_t &len_inner) {
  const GeTensorDesc &tensor_desc = op_desc_->GetInputDesc(0U);
  int64_t num_cnt = tensor_desc.GetShape().IsScalar() ? 1 : tensor_desc.GetShape().GetShapeSize();
  int64_t shape_len = GetSizeInBytes(num_cnt, tensor_desc.GetDataType());
  if (shape_len <= 0) {
    RTS_REPORT_INNER_ERROR("invalid shape_len %" PRId64 " should be greater than 0.", shape_len);
    return FAILED;
  }
  int64_t offset{0};
  (void)AttrUtils::GetInt(op_desc_, kAttrAddrOffset, offset);
  RTS_LOGD("[%s] got offset [%" PRId64 "], size:[%" PRId64 "]", op_desc_->GetNamePtr(), offset, shape_len);

  if ((offset < 0) || (offset >= shape_len)) {
    RTS_REPORT_INNER_ERROR("The offset %" PRId64 " should be within the range of [0, %" PRId64 ").", offset, shape_len);
    return FAILED;
  }

  shape_len -= offset;

  uint32_t max_size{0U};
  (void)AttrUtils::GetInt(op_desc_, kAttrMaxSize, max_size);
  if (max_size == 0) {
    max_size = kMaxPrefetchLen;
  }

  len_inner = std::min(static_cast<uint32_t>(shape_len), max_size);
  return SUCCESS;
}
// function of op {ENTER, LOOPCOND, NEXTITERATION, EXIT} is similar: pass through data without change
}  // namespace runtime
}  // namespace cce