/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "stream_switchN_op.h"
#include "runtime/rt.h"

#include "graph/debug/ge_attr_define.h"
#include "op_factory.h"
#include "common/util/log.h"
#include "../acl_rt_compare_data_type.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

constexpr uint64_t MAX_UINT64_NUM = 0xFFFFFFFFFFFFFFFFULL;

using namespace ge;
namespace cce {
namespace runtime {
static const uint32_t STREAM_SWITCH_N_MIN_NUM = 1;

StreamSwitchNOp::StreamSwitchNOp(const Node &node, RunContext &runContext)
    : Op(node, runContext), element_size_(0), size_(0), data_type_(ACL_RT_SWITCH_INT64) {}

Status StreamSwitchNOp::Init() {
  RTS_LOGI("StreamSwitchNOp init, node:%s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init StreamSwitchNOp failed, retCode=%#x.", ret);
    return ret;
  }

  uint32_t elementSize = 0;
  bool getAttrSucc = AttrUtils::GetInt(op_desc_, ATTR_NAME_BATCH_NUM, elementSize);
  if (!getAttrSucc) {
    RTS_REPORT_CALL_ERROR("StreamSwitchNOp get attr ATTR_NAME_BATCH_NUM fail.");
    return FAILED;
  }
  RTS_LOGI("StreamSwitchNOp init get elementSize from op_desc_ elementSize = %u", elementSize);
  element_size_ = elementSize;
  if (op_desc_->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE)) {
    int64_t dataType = 0;
    getAttrSucc = AttrUtils::GetInt(op_desc_, ATTR_NAME_SWITCH_DATA_TYPE, dataType);
    if (!getAttrSucc) {
      RTS_REPORT_CALL_ERROR("StreamSwitchNOp[node:%s] get attr SWITCH_DATA_TYPE failed.", name_.c_str());
      return FAILED;
    }
    data_type_ = static_cast<aclrtCompareDataType>(dataType);
  }

  for (size_t i = 0; i < element_size_; i++) {
    if (data_type_ == ACL_RT_SWITCH_INT32) {
      std::vector<int32_t> valueList;
      std::string attrName = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
      getAttrSucc = AttrUtils::GetListInt(op_desc_, attrName, valueList);
      if (!getAttrSucc) {
        RTS_REPORT_CALL_ERROR("StreamSwitchNOp[node:%s] get attr ATTR_NAME_PRED_VALUE index=%zu failed.", name_.c_str(),
                              i);
        return FAILED;
      }
      size_ = valueList.size();
      for (size_t j = 0; j < size_; j++) {
        target_value_.push_back(valueList[j]);
      }
    } else {
      vector<int64_t> value64List;
      std::string attrName = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
      getAttrSucc = AttrUtils::GetListInt(op_desc_, attrName, value64List);
      if (!getAttrSucc) {
        RTS_REPORT_CALL_ERROR("StreamSwitchNOp[node:%s] get attr ATTR_NAME_PRED_VALUE index=%zu failed.", name_.c_str(),
                              i);
        return FAILED;
      }
      size_ = value64List.size();
      for (size_t j = 0; j < size_; j++) {
        target_value64_.push_back(value64List[j]);
      }
    }
  }

  if (!AttrUtils::GetListInt(op_desc_, ATTR_NAME_ACTIVE_STREAM_LIST, activeStreamList_)) {
    RTS_REPORT_CALL_ERROR("StreamSwitchNOp get attr ACTIVE_STREAM_LIST fail. node:%s.", name_.c_str());
    return FAILED;
  }
  if (activeStreamList_.size() < element_size_) {
    RTS_REPORT_CALL_ERROR("StreamSwitchNOp[node:%s] activeStreamSize:%zu, elementSize:%u failed.", name_.c_str(),
                          activeStreamList_.size(), element_size_);
    return FAILED;
  }
  RTS_LOGI("StreamSwitchNOp get attribute[%s] list size=%zu", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
           activeStreamList_.size());
  return SUCCESS;
}

Status StreamSwitchNOp::Run(vector<TaskDef> &tasks) {
  if (v_input_data_addr_.size() < STREAM_SWITCH_N_MIN_NUM) {
    RTS_REPORT_CALL_ERROR("v_input_data_addr_ invalid, valid address size range is [%u, %#" PRIx64 "].",
                          STREAM_SWITCH_N_MIN_NUM, MAX_UINT64_NUM);
    return INTERNAL_ERROR;
  }
  RTS_LOGI("StreamSwitchNOp run start, node: %s.", name_.c_str());
  domi::TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_STREAM_SWITCH_N);
  taskDef.set_stream_id(op_desc_->GetStreamId());
  domi::StreamSwitchNDef *streamSwitchNDef = taskDef.mutable_stream_switch_n();
  streamSwitchNDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  streamSwitchNDef->set_size(size_);
  streamSwitchNDef->set_element_size(element_size_);
  streamSwitchNDef->set_data_type(data_type_);

  for (size_t i = 0; i < size_ * element_size_; i++) {
    if (data_type_ == ACL_RT_SWITCH_INT32) {
      streamSwitchNDef->add_target_value((reinterpret_cast<int32_t *>(&target_value_[0]))[i]);
    } else {
      streamSwitchNDef->add_target_value((reinterpret_cast<int64_t *>(&target_value64_[0]))[i]);
    }
  }

  for (size_t i = 0; i < element_size_; i++) {
    streamSwitchNDef->add_true_stream_id(activeStreamList_[i]);
  }
  tasks.push_back(taskDef);
  RTS_LOGI("end StreamSwitchNOp stream_id:%u.", op_desc_->GetStreamId());
  return SUCCESS;
}

Status StreamSwitchNOp::UpdateTaskDef(vector<TaskDef> &tasks) {
  const uint32_t streamId = op_desc_->GetStreamId();
  vector<uint32_t> activeStreamList;
  bool getAttrSucc = AttrUtils::GetListInt(op_desc_, ATTR_NAME_ACTIVE_STREAM_LIST, activeStreamList);
  if (!getAttrSucc) {
    RTS_LOGE("StreamSwitchOp[node:%s] UpdateTaskDef get attr ACTIVE_STREAM_LIST fail.", name_.c_str());
    return FAILED;
  }
  uint32_t elementSize = 0U;
  getAttrSucc = AttrUtils::GetInt(op_desc_, ATTR_NAME_BATCH_NUM, elementSize);
  if (!getAttrSucc) {
    RTS_REPORT_CALL_ERROR("StreamSwitchNOp UpdateTaskDef get attr ATTR_NAME_BATCH_NUM fail.");
    return FAILED;
  }
  if (activeStreamList.size() < elementSize) {
    RTS_LOGE("StreamSwitchNOp activeStreamSize:%zu, elementSize:%u failed.", activeStreamList.size(), elementSize);
    return FAILED;
  }
  for (auto &taskDef : tasks) {
    taskDef.set_stream_id(streamId);
    domi::StreamSwitchDef *streamSwitchDef = taskDef.mutable_stream_switch();
    for (size_t i = 0; i < elementSize; i++) {
      streamSwitchDef->set_true_stream_id(activeStreamList[i]);
    }
  }
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
