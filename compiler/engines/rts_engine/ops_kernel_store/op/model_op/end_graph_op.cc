/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "end_graph_op.h"
#include "op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/compute_graph.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {

EndGraphOp::EndGraphOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status EndGraphOp::Init() {
  RTS_LOGD("EndGraphOp Init, node:%s.", name_.c_str());
  input_num_ = 0;
  output_num_ = 0;

  auto op_desc = node_.GetOpDesc();
  ComputeGraphPtr owner_graph = node_.GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    RTS_REPORT_CALL_ERROR("Owner graph is nullptr");
    return FAILED;
  }

  const auto parent = owner_graph->GetParentGraph();
  if (parent != nullptr) {
    bool is_dsp_split_parent = false;
    (void)AttrUtils::GetBool(parent, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dsp_split_parent);
    if (!owner_graph->GetGraphUnknownFlag() && (is_dsp_split_parent || parent->GetGraphUnknownFlag())) {
      need_gen_endgraph_task_ = true;
    } else {
      need_gen_endgraph_task_ = false;
      RTS_LOGI("Skip subgraph NetOutput node: %s.", op_desc->GetName().c_str());
    }
  }

  RTS_LOGI("endGraph start need_gen_endgraph_task_:%d.", need_gen_endgraph_task_);
  return SUCCESS;
}

Status EndGraphOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("endGraph start need_gen_endgraph_task_:%d.", need_gen_endgraph_task_);
  if (need_gen_endgraph_task_) {
    const uint32_t streamId = op_desc_->GetStreamId();
    TaskDef taskDef = {};
    taskDef.set_type(ACL_RT_MODEL_TASK_MODEL_END_GRAPH);
    taskDef.set_stream_id(streamId);
    tasks.push_back(taskDef);
    RTS_LOGI("end endGraph streamId:%u.", streamId);
  }

  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
