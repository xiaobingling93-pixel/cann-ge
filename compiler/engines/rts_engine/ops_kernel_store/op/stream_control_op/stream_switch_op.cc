/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "stream_switch_op.h"
#include "graph/debug/ge_attr_define.h"
#include "op_factory.h"
#include "common/util/log.h"
#include "../acl_rt_compare_data_type.h"
#include "../acl_rt_condition.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
static const uint32_t STREAM_SWITCH_INPUT_NUM = 2;
static const uint32_t TRUE_BRANCH_STREAM_NUM = 1;

StreamSwitchOp::StreamSwitchOp(const Node &node, RunContext &runContext)
    : Op(node, runContext), cond_(ACL_RT_GREATER), trueStreamIndex_(0), data_type_(ACL_RT_SWITCH_INT64) {}

Status StreamSwitchOp::Init() {
  RTS_LOGI("StreamSwitchOp init, node: %s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init StreamSwitchOp failed, retCode=%#x", ret);
    return ret;
  }

  int64_t cond = 0;
  bool getAttrSucc = AttrUtils::GetInt(op_desc_, ATTR_NAME_STREAM_SWITCH_COND, cond);
  if (!getAttrSucc) {
    RTS_LOGE("StreamSwitchOp get attr ITERATORS_PER_LOOP fail");
    return FAILED;
  }
  cond_ = static_cast<aclrtCondition>(cond);
  if (input_num_ != STREAM_SWITCH_INPUT_NUM || v_input_data_addr_.size() != STREAM_SWITCH_INPUT_NUM) {
    RTS_LOGE(
        "StreamSwitchOp input_num should be %u,"
        " actually input num is %zu, input addr size is %zu.",
        STREAM_SWITCH_INPUT_NUM, input_num_, v_input_data_addr_.size());
    return FAILED;
  }
  vector<uint32_t> active_stream_list;
  getAttrSucc = AttrUtils::GetListInt(op_desc_, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
  if (!getAttrSucc) {
    RTS_LOGE("StreamSwitchOp[node:%s] get attr ACTIVE_STREAM_LIST fail.", name_.c_str());
    return FAILED;
  }
  if (active_stream_list.size() != TRUE_BRANCH_STREAM_NUM) {
    RTS_LOGE("Stream num of switch true branch must be %u[node:%s].", TRUE_BRANCH_STREAM_NUM, name_.c_str());
    return FAILED;
  }
  trueStreamIndex_ = static_cast<uint32_t>(active_stream_list.front());
  if (op_desc_->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE)) {
    int64_t dataType = 0;
    getAttrSucc = AttrUtils::GetInt(op_desc_, ATTR_NAME_SWITCH_DATA_TYPE, dataType);
    if (!getAttrSucc) {
      RTS_LOGE("StreamSwitchOp[node:%s] get attr SWITCH_DATA_TYPE fail.", name_.c_str());
      return FAILED;
    }
    data_type_ = static_cast<aclrtCompareDataType>(dataType);
  }
  RTS_LOGI("StreamSwitchOp init, cond: %d.", cond_);
  return SUCCESS;
}

Status StreamSwitchOp::Run(vector<TaskDef> &tasks) {
  if (v_input_data_addr_.size() != STREAM_SWITCH_INPUT_NUM) {
    RTS_LOGE("invalid v_input_data_addr_ size(%zu), v_input_data_addr_ size must be %u.", v_input_data_addr_.size(),
             STREAM_SWITCH_INPUT_NUM);
    return INTERNAL_ERROR;
  }
  RTS_LOGI("StreamSwitchOp run, name: %s, cond: %d, datatype: %d.", name_.c_str(), cond_, data_type_);

  const uint32_t streamId = op_desc_->GetStreamId();
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_STREAM_SWITCH);
  taskDef.set_stream_id(streamId);
  domi::StreamSwitchDef *streamSwitchDef = taskDef.mutable_stream_switch();
  streamSwitchDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  streamSwitchDef->set_true_stream_id(trueStreamIndex_);
  streamSwitchDef->set_value(0);
  tasks.push_back(taskDef);
  RTS_LOGI("end StreamSwitchOp streamId:%u.", streamId);
  return SUCCESS;
}

Status StreamSwitchOp::UpdateTaskDef(vector<TaskDef> &tasks) {
  const uint32_t streamId = op_desc_->GetStreamId();
  vector<uint32_t> activeStreamList;
  bool getAttrSucc = AttrUtils::GetListInt(op_desc_, ATTR_NAME_ACTIVE_STREAM_LIST, activeStreamList);
  if (!getAttrSucc) {
    RTS_LOGE("StreamSwitchOp[node:%s] get attr ACTIVE_STREAM_LIST fail.", name_.c_str());
    return FAILED;
  }

  if (activeStreamList.size() != TRUE_BRANCH_STREAM_NUM) {
    RTS_LOGE("Stream num of switch true branch must be %u[node:%s].", TRUE_BRANCH_STREAM_NUM, name_.c_str());
    return FAILED;
  }
  const uint32_t streamIndex = activeStreamList.front();

  for (auto &taskDef : tasks) {
    taskDef.set_stream_id(streamId);
    domi::StreamSwitchDef *streamSwitchDef = taskDef.mutable_stream_switch();
    streamSwitchDef->set_true_stream_id(streamIndex);
  }

  RTS_LOGI("update StreamSwitchOp taskDefSize:%zu, stream_id:%u,stream_index:%u.", tasks.size(), streamId, streamIndex);
  return SUCCESS;
}

Status StreamSwitchOp::GenerateCtxDef(const Node &node) {
  constexpr uint32_t readProtectConfig = 2;
  std::shared_ptr<domi::FftsPlusCtxDef> fftsCondContext = nullptr;
  try {
    fftsCondContext = std::make_shared<domi::FftsPlusCtxDef>();
  } catch (...) {
    RTS_REPORT_INNER_ERROR("Failed to create a context information.");
    return FAILED;
  }

  if (node.GetOpDesc() == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op failed, param can not be NULL.");
    return FAILED;
  }

  fftsCondContext->set_op_index(node.GetOpDesc()->GetId());
  fftsCondContext->set_context_type(RT_CTX_TYPE_COND_SWITCH);
  domi::FftsPlusCondSwitchCtxDef *condSwitchCtx = fftsCondContext->mutable_cond_switch_ctx();

  condSwitchCtx->set_atm(0);
  condSwitchCtx->set_thread_id(0);
  condSwitchCtx->set_thread_dim(1);

  if (data_type_ == ACL_RT_SWITCH_INT32) {
    condSwitchCtx->set_ar_size(2);  // data type only support int32 int32:arsize=2, int64:arsize=3
  } else {
    RTS_LOGE("not support data type, data_type=%u", data_type_);
    return INTERNAL_ERROR;
  }

  condSwitchCtx->set_snoop(0);
  condSwitchCtx->set_ar_cache(0);
  condSwitchCtx->set_ar_prot(readProtectConfig);
  condSwitchCtx->set_va(1);
  condSwitchCtx->set_load_addr0_base(reinterpret_cast<uintptr_t>(v_input_data_addr_[0]));
  condSwitchCtx->set_ld0_en(1);
  condSwitchCtx->set_load_addr0_offset(0);
  condSwitchCtx->set_load_addr1_base(reinterpret_cast<uintptr_t>(v_input_data_addr_[1]));
  condSwitchCtx->set_ld1_en(1);
  condSwitchCtx->set_load_addr1_offset(0);
  condSwitchCtx->set_cmp_value_1(0);
  condSwitchCtx->set_cmp_value_2(0);

  node.GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", fftsCondContext);
  RTS_LOGI("Generate FFTSPlus context for StreamSwitchOp success, node name %s", name_.c_str());
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
