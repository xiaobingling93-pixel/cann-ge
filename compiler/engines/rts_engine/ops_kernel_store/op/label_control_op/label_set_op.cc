/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "label_set_op.h"
#include "op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
LabelSetOp::LabelSetOp(const Node &node, RunContext &runContext) : Op(node, runContext), labelIndex_(0) {}

Status LabelSetOp::Init() {
  RTS_LOGI("LabelSetOp Init, node: %s.", name_.c_str());
  input_num_ = 0;
  output_num_ = 0;

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init failed, node: %s, retCode: %#x", name_.c_str(), ret);
    return ret;
  }

  uint32_t labelIndex = 0U;
  if (!AttrUtils::GetInt(op_desc_, ATTR_NAME_LABEL_SWITCH_INDEX, labelIndex)) {
    RTS_REPORT_CALL_ERROR("LabelSetOp: %s attr [%s] not exist!", name_.c_str(), ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return FAILED;
  }
  labelIndex_ = labelIndex;

  return SUCCESS;
}

Status LabelSetOp::Run(vector<TaskDef> &tasks) {
  const uint32_t streamId = op_desc_->GetStreamId();
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_LABEL_SET);
  taskDef.set_stream_id(streamId);
  domi::LabelSetDef *labelSetDef = taskDef.mutable_label_set();
  labelSetDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  labelSetDef->set_label_id(labelIndex_);

  tasks.push_back(taskDef);
  RTS_LOGI("end LabelSetOp streamId:%u, labelIndex:%u.", streamId, labelIndex_);
  return SUCCESS;
}

Status LabelSetOp::GenerateCtxDef(const Node &node) {
  if (node.GetOpDesc() == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op failed, param can not be NULL.");
    return FAILED;
  }

  std::shared_ptr<domi::FftsPlusCtxDef> fftsLabelContext = nullptr;
  try {
    fftsLabelContext = std::make_shared<domi::FftsPlusCtxDef>();
  } catch (...) {
    RTS_REPORT_INNER_ERROR("Failed to create a context information.");
    return FAILED;
  }

  fftsLabelContext->set_op_index(node.GetOpDesc()->GetId());
  fftsLabelContext->set_context_type(RT_CTX_TYPE_LABEL);
  domi::FftsPlusLabelCtxDef *caseLabelCtx = fftsLabelContext->mutable_label_ctx();
  caseLabelCtx->set_pred_cnt(0);
  node.GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", fftsLabelContext);

  RTS_LOGI("Generate FFTSPlus context for LabelSetOp success, node name %s", name_.c_str());
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
