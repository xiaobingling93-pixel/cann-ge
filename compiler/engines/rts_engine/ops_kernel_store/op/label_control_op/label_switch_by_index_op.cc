/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "label_switch_by_index_op.h"

#include "op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
static const uint32_t LABEL_SWITCH_INPUT_NUM = 1;

LabelSwitchByIndexOp::LabelSwitchByIndexOp(const Node &node, RunContext &runContext)
    : Op(node, runContext), branch_max_(0), data_type_(DT_INT64) {}

Status LabelSwitchByIndexOp::Init() {
  RTS_LOGI("LabelSwitchByIndexOp Init, node: %s.", name_.c_str());
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init failed, node: %s, retCode:%#x", name_.c_str(), ret);
    return ret;
  }

  if (input_num_ != LABEL_SWITCH_INPUT_NUM || v_input_data_addr_.size() != LABEL_SWITCH_INPUT_NUM) {
    RTS_REPORT_CALL_ERROR(
        "Label switch by index op init failed, input_num should be %u, actually input num is %zu, input addr size is "
        "%zu.",
        LABEL_SWITCH_INPUT_NUM, input_num_, v_input_data_addr_.size());
    return FAILED;
  }

  data_type_ = op_desc_->GetInputDescPtr(0)->GetDataType();

  std::vector<uint32_t> label_idx_list;
  if (!AttrUtils::GetListInt(op_desc_, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list)) {
    RTS_REPORT_CALL_ERROR("LabelSwitchByIndexOp: node: %s get %s failed.", name_.c_str(),
                          ATTR_NAME_LABEL_SWITCH_LIST.c_str());
    return FAILED;
  }

  if (label_idx_list.empty()) {
    RTS_REPORT_CALL_ERROR("Label switch by index op init failed, node: %s label list is empty.", name_.c_str());
    return FAILED;
  }
  branch_max_ = label_idx_list.size();

  return SUCCESS;
}

Status LabelSwitchByIndexOp::Run(vector<TaskDef> &tasks) {
  const uint32_t streamId = op_desc_->GetStreamId();
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX);
  taskDef.set_stream_id(streamId);
  domi::LabelSwitchByIndexDef *labelSwitchDef = taskDef.mutable_label_switch_by_index();
  labelSwitchDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  labelSwitchDef->set_label_max(branch_max_);
  tasks.push_back(taskDef);
  RTS_LOGI("end LabelSwitchByIndexOp streamId:%u.", streamId);
  return SUCCESS;
}

Status LabelSwitchByIndexOp::GenerateCtxDef(const Node &node) {
  if (node.GetOpDesc() == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op failed, param can not be NULL.");
    return FAILED;
  }

  constexpr uint32_t readProtectConfig = 2U;
  std::shared_ptr<domi::FftsPlusCtxDef> fftsSwitchContext = nullptr;
  try {
    fftsSwitchContext = std::make_shared<domi::FftsPlusCtxDef>();
  } catch (...) {
    RTS_REPORT_INNER_ERROR("Failed to create a context information.");
    return FAILED;
  }

  fftsSwitchContext->set_op_index(node.GetOpDesc()->GetId());
  fftsSwitchContext->set_context_type(RT_CTX_TYPE_CASE_SWITCH);
  domi::FftsPlusCaseSwitchCtxDef *caseSwitchCtx = fftsSwitchContext->mutable_case_switch_ctx();

  caseSwitchCtx->set_atm(0);
  caseSwitchCtx->set_thread_id(0);
  caseSwitchCtx->set_thread_dim(1);

  if (data_type_ == DT_INT32) {
    caseSwitchCtx->set_ar_size(2);  // data type only support int32.int32:arsize=2
  } else if (data_type_ == DT_INT64) {
    caseSwitchCtx->set_ar_size(3);  // int64:arsize=3
  } else {
    RTS_REPORT_CALL_ERROR("current data type %u, is not supported.", data_type_);
    return FAILED;
  }
  caseSwitchCtx->set_snoop(0);
  caseSwitchCtx->set_ar_cache(0);
  caseSwitchCtx->set_ar_prot(readProtectConfig);
  caseSwitchCtx->set_va(1);
  caseSwitchCtx->set_load_addr0_base(reinterpret_cast<uintptr_t>(v_input_data_addr_[0]));
  caseSwitchCtx->set_ld0_en(1);
  caseSwitchCtx->set_load_addr0_offset(0);
  caseSwitchCtx->set_load_addr1_base(0);
  caseSwitchCtx->set_ld1_en(0);
  caseSwitchCtx->set_load_addr1_offset(0);

  node.GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", fftsSwitchContext);
  RTS_LOGI("Generate FFTSPlus context for LabelSwitchByIndexOp success, node name %s", name_.c_str());
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
