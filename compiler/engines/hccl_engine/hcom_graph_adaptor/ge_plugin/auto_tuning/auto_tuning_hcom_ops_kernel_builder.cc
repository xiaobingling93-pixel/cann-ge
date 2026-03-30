/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_tuning_hcom_ops_kernel_builder.h"
#include <functional>
#include <securec.h>
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/ops_kernel_builder_registry.h"
#include "hcom_op_utils.h"

namespace hccl {
REGISTER_OPS_KERNEL_BUILDER(AUTOTUNE_HCCL_OPS_LIB_NAME, hccl::AutoTuningHcomOpsKernelBuilder);

AutoTuningHcomOpsKernelBuilder::AutoTuningHcomOpsKernelBuilder() {}

AutoTuningHcomOpsKernelBuilder::~AutoTuningHcomOpsKernelBuilder() {}

HcclResult AutoTuningHcomOpsKernelBuilder::GetSupportedOP(std::vector<std::string> &hcclSupportOp) const {
  hcclSupportOp.assign(AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.begin(), AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.end());
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelBuilder::GetOriginalGraphShapeTypeFromDesc(const ge::OpDescPtr &op, u32 &shapeType) {
  if (ge::AttrUtils::HasAttr(op, ORIGINAL_GRAPH_SHAPE_TYPE)) {
    if (ge::AttrUtils::GetInt(op, ORIGINAL_GRAPH_SHAPE_TYPE, shapeType) == false) {
      HCCL_ERROR(
          "[Get][OriginalGraphShapeType]errNo[0x%016llx]: get shapeType failed. get \"shapeType\" from"
          "opDesc failed",
          HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  } else {
    shapeType = (u32)ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;
  }
  HCCL_INFO("get shapeType [%u] success.", shapeType);
  return HCCL_SUCCESS;
}

// 返回运行参数，包括workspace 、stream数量以及atomic标志位
ge::Status AutoTuningHcomOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  HCCL_INFO("calculate hccl running parameters start.");
  CHK_PRT_RET(
      !node.GetOpDesc(),
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetOpDesc failed. null ptr.", HCOM_ERROR_CODE(HCCL_E_PTR)),
      ge::INTERNAL_ERROR);

  bool unknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(node, unknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Calc][OpRunningParam]node[%s] get node unknown status failed", node.GetName().c_str()),
              ge::INTERNAL_ERROR);
  if (unknownShapeNode) {
    HCCL_INFO("node:%s is unknown shape, does not need to Calc Op Running Param", node.GetName().c_str());
    return HCCL_SUCCESS;
  }

  // 获取需回传的信息
  u64 streamNum = 0;
  u64 opMemSize = 0;
  std::string sCollectiveType = node.GetOpDesc()->GetType();
  HcclResult ret = CheckSupportedOP(sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op type[%s] is not supported.", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);
  // 获取并设定stream 数量
  if (ge::AttrUtils::SetInt(node.GetOpDesc(), "used_stream_num", streamNum) == false) {
    HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op[%s]: set stream number[%llu] to OpDesc failed.",
               HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), streamNum);
    return ge::INTERNAL_ERROR;
  }

  std::vector<int64_t> workspaceBytes;
  workspaceBytes.push_back(opMemSize);
  node.GetOpDesc()->SetWorkspaceBytes(workspaceBytes);

  // 设置内存属性
  ret = SetOpMemAttr(node, node.GetOpDesc()->GetType(), opMemSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set node[%s] mem attr failed.", HCOM_ERROR_CODE(ret),
                         node.GetName().c_str()),
              ge::INTERNAL_ERROR);

  // 设置output size 大小
  ret = SetOpOutputMemSize(node, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set op[%s] output size failed.", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);

  HCCL_INFO("calcute hccl running parameters completed. stream num:[%llu], workspace size:[%llu]bytes", streamNum,
            opMemSize);
  return ge::SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelBuilder::GetRankSizeFromDesc(const ge::OpDescPtr &op, uint32_t &rankSize,
                                                               const std::string &sCollectiveType) {
  rankSize = 0;
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
    if (ge::AttrUtils::HasAttr(op, "rank_size")) {
      if (ge::AttrUtils::GetInt(op, "rank_size", rankSize) == false) {
        HCCL_ERROR(
            "[Get][RankSize]errNo[0x%016llx] op[%s]: get rank size failed. get \"rank_size\" from"
            "opDesc failed",
            HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
        return HCCL_E_PARA;
      }
    } else {
      HCCL_ERROR("[Get][RankSize]errNo[0x%016llx] op[%s]: get rank size failed. no \"rank_size\" in opDesc",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }
  HCCL_INFO("get dest rank[%u] success.", rankSize);
  return HCCL_SUCCESS;
}

ge::Status AutoTuningHcomOpsKernelBuilder::GenerateTask(const ge::Node &node, [[maybe_unused]] ge::RunContext &runContext,
                                                        std::vector<domi::TaskDef> &taskDefList) {
  bool unknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(node, unknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Generate][Task]node[%s] get node unknown status failed", node.GetName().c_str()),
              HCCL_E_PARA);
  if (unknownShapeNode) {
    HCCL_INFO("op:%s is unknown shape, does not need to generate Task.", node.GetName().c_str());
    return HCCL_SUCCESS;
  }

  AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf = {0, 0, 0, HCCL_DATA_TYPE_INT8};
  domi::TaskDef taskDef;
  HCCL_INFO("GenerateTask start.");
  CHK_PRT_RET(!node.GetOpDesc(),
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] opDesc is null.", HCOM_ERROR_CODE(HCCL_E_PTR)),
              ge::INTERNAL_ERROR);
  taskDef.clear_kernel_hccl();
  domi::KernelHcclDef *kernelDefHccl = taskDef.mutable_kernel_hccl();
  CHK_PRT_RET((kernelDefHccl == nullptr),
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] kernelDefHccl is null.", HCOM_ERROR_CODE(HCCL_E_PTR)),
              ge::INTERNAL_ERROR);
  taskDef.set_type(RT_MODEL_TASK_HCCL);
  taskDef.set_stream_id(node.GetOpDesc()->GetStreamId());
  std::string sCollectiveType = node.GetOpDesc()->GetType();
  HcclResult ret = CheckSupportedOP(sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] op type[%s] is not supported.", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);
  kernelDefHccl->set_hccl_type(sCollectiveType);
  kernelDefHccl->set_op_index(node.GetOpDesc()->GetId());
  HCCL_INFO("GenerateTask: hccl op id[%d].", node.GetOpDesc()->GetId());

  // 获取 hcom 必需的参数
  u32 rankSize = 0;
  ret = GetRankSizeFromDesc(node.GetOpDesc(), rankSize, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get dest_rank failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);
  privateDefBuf.rankSize = rankSize;
  u32 shapeType = 0;
  ret = GetOriginalGraphShapeTypeFromDesc(node.GetOpDesc(), shapeType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get shapeType failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  privateDefBuf.originalGraphShapeType = shapeType;

  privateDefBuf.outputBytes = 0;
  u32 outputSize = node.GetOpDesc()->GetOutputsSize();
  if (outputSize != 0) {
    int64_t outputBytes = 0;
    CHK_PRT_RET((ge::TensorUtils::GetSize(node.GetOpDesc()->GetOutputDesc(0), outputBytes) != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Get][Size] from TensorDesc failed, op:%s, output index:%u.",
                           node.GetOpDesc()->GetName().c_str(), 0),
                ge::INTERNAL_ERROR);
    privateDefBuf.outputBytes = outputBytes;
  }
  ret = HcomOpUtils::ConversionOpDataType(node.GetOpDesc(), sCollectiveType, privateDefBuf.dataType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] conversion op data type failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  // 设定 privateDefBuf 到 protubuf 的 private_def
  taskDef.set_private_def(&privateDefBuf, sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF));
  taskDefList.push_back(taskDef);
  HCCL_INFO("GenerateTask success.");
  return ge::SUCCESS;
}
}  // namespace hccl