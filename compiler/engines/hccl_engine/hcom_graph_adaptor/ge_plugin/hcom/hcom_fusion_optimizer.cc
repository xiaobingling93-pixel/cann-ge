/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_fusion_optimizer.h"
#include <nlohmann/json.hpp>
#include "hcom_alltoallvc_fusion.h"
#include "hcom_allgather_fusion.h"
#include "hcom_reducescatter_fusion.h"
#include "hcom_ops_kernel_info_store.h"
#include "hcom_op_utils.h"
#include "hccl/hcom.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "framework/memory/memory_api.h"
#include "ge/ge_api_types.h"            // ge对内options
#include "framework/common/ge_types.h"  // ge对外options
#include "common/util/trace_manager/trace_manager.h"
#include "offline_build_config_parse.h"
#include "graph/utils/graph_utils.h"

using namespace std;

namespace hccl {
HcomFusionOptimizer::HcomFusionOptimizer() {}

HcomFusionOptimizer::~HcomFusionOptimizer() {}

ge::Status HcomFusionOptimizer::Initialize([[maybe_unused]] const std::map<std::string, std::string> &options,
                                           [[maybe_unused]] ge::OptimizeUtility *const optimizeUtility) {
  return ge::SUCCESS;
}

ge::Status HcomFusionOptimizer::OptimizeGraphPrepare([[maybe_unused]] ge::ComputeGraph &graph) {
  return ge::SUCCESS;
}

ge::Status HcomFusionOptimizer::Finalize() {
  return ge::SUCCESS;
}

ge::Status HcomFusionOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph) {
  HcclResult ret = HcomOptimizeOriginalGraph(graph);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: Original Optimize failed. ret[%d]", graph.GetName().c_str(), ret),
      ge::INTERNAL_ERROR);

  return ge::SUCCESS;
}

HcclResult HcomFusionOptimizer::HcomOptimizeOriginalGraph(ge::ComputeGraph &graph) {
  HcclResult ret = FuseHcomAlltoAllVCNode(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: fuse HcomAlltoAllVC node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);
  ret = FuseHcomAllgatherNode(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: fuse HcomAllGather node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);
  ret = FuseHcomReduceScatterNode(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: fuse HcomReduceScatter node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);
  ret = HcomOptimizeSetAttr(graph);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: set attr node failed. ret[%d]", graph.GetName().c_str(), ret),
      HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult HcomFusionOptimizer::HcomOptimizeSetAttr(ge::ComputeGraph &graph) {
  // 通过通信算子输出个数区分图融合与非融合场景。为了减少插入splitv和concat算子，在融合场景给算子打上连续内存输入输出标记
  for (auto nodePtr : graph.GetAllNodes()) {
    bool bRet = false;
    auto opDescPtr = nodePtr->GetOpDesc();
    if ((opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLGATHER && opDescPtr->GetInputsSize() != 1) ||
        (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCESCATTER && opDescPtr->GetInputsSize() != 1)) {
      bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
      CHK_PRT_RET(
          !bRet,
          HCCL_ERROR("[Set][OpAttrs]node[%s] set continuous input attr to OpDesc failed", opDescPtr->GetName().c_str()),
          HCCL_E_INTERNAL);
      bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_CONTINUOUS_OUTPUT, true);
      CHK_PRT_RET(!bRet,
                  HCCL_ERROR("[Set][OpAttrs]node[%s] set continuous output attr to OpDesc failed.",
                             opDescPtr->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }
    // 对于reducescatter算子，由于打上连续内存输入输出属性在复用内存场景padding位置内存需清零，否则会参与reduce计算导致内存溢出
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCESCATTER && opDescPtr->GetInputsSize() != 1) {
      vector<int64_t> atomicInputIndex(1, -1);  // 回传vector的值为-1，作为标志位
      if (!ge::AttrUtils::SetListInt(opDescPtr, "atomic_input_index", atomicInputIndex)) {
        HCCL_ERROR("[Set][OpAtomicInputIndex]errNo[0x%016llx]: set op[%s] atomic index failed.",
                   HCOM_ERROR_CODE(HCCL_E_PARA), opDescPtr->GetType().c_str());
        return HCCL_E_PARA;
      }
    }
  }
  return HCCL_SUCCESS;
}

ge::Status HcomFusionOptimizer::OptimizeFusedGraph([[maybe_unused]] ge::ComputeGraph &graph) {
  return ge::SUCCESS;
}

ge::Status HcomFusionOptimizer::OptimizeWholeGraph([[maybe_unused]] ge::ComputeGraph &graph) {
  return ge::SUCCESS;
}

ge::Status HcomFusionOptimizer::GetAttributes(ge::GraphOptimizerAttribute &attrs) const {
  attrs.engineName = HCCL_OPS_ENGIN;
  attrs.scope = ge::UNIT;
  HCCL_DEBUG("hccl graph optimizer get attr success. engine[%s] scope[%d]", attrs.engineName.c_str(), attrs.scope);
  return ge::SUCCESS;
}

HcclResult HcomFusionOptimizer::FuseHcomAlltoAllVCNode(ge::ComputeGraph &graph) {
  HcomAlltoAllVCFusion fusionHcomAlltoAllVCOp;
  HCCL_INFO("graph[%s] start fusion HcomAlltoAllVC node.", graph.GetName().c_str());
  HcclResult ret = fusionHcomAlltoAllVCOp.Run(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Fuse][HcomAlltoAllVCNode]graph[%s]: fuse HcomAlltoAllVC node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    HcomAlltoAllVCFusion fusionSubGraphOp;
    ret = fusionSubGraphOp.Run(*subgraph[index]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Fuse][HcomAlltoAllVCNode]fuse HcomAlltoAllVC op failed in subgraph[%s]. ret[%d]",
                           (*subgraph[index]).GetName().c_str(), ret),
                ret);
  }

  HCCL_INFO("graph[%s] with[%d] subgraphs: end fusion HcomAlltoAllVC node.", graph.GetName().c_str(), subgraph.size());
  return HCCL_SUCCESS;
}

HcclResult HcomFusionOptimizer::FuseHcomAllgatherNode(ge::ComputeGraph &graph) {
  HcomAllGatherFusion fusionHcomAllGatherOp;
  HCCL_INFO("graph[%s] start fusion HcomAllGather node.", graph.GetName().c_str());
  HcclResult ret = fusionHcomAllGatherOp.Run(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Fuse][HcomAlltGatherNode]graph[%s]: fuse HcomAllGather node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    HcomAllGatherFusion fusionSubGraphOp;
    ret = fusionSubGraphOp.Run(*subgraph[index]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Fuse][HcomAlltGatherNode]fuse HcomAllGather op failed in subgraph[%s]. ret[%d]",
                           (*subgraph[index]).GetName().c_str(), ret),
                ret);
  }

  HCCL_INFO("graph[%s] with[%d] subgraphs: end fusion HcomAllGather node.", graph.GetName().c_str(), subgraph.size());
  return HCCL_SUCCESS;
}

HcclResult HcomFusionOptimizer::FuseHcomReduceScatterNode(ge::ComputeGraph &graph) {
  HcomReduceScatterFusion fusionHcomReduceScatterOp;
  HCCL_INFO("graph[%s] start fusion HcomReduceScatter node.", graph.GetName().c_str());
  HcclResult ret = fusionHcomReduceScatterOp.Run(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Fuse][HcomReduceScatternode]graph[%s]: fuse HcomReduceScatter node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    HcomReduceScatterFusion fusionSubGraphOp;
    ret = fusionSubGraphOp.Run(*subgraph[index]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Fuse][HcomReduceScatternode]fuse HcomReduceScatter op failed in subgraph[%s]. ret[%d]",
                           (*subgraph[index]).GetName().c_str(), ret),
                ret);
  }

  HCCL_INFO("graph[%s] with[%d] subgraphs: end fusion HcomReduceScatter node.", graph.GetName().c_str(),
            subgraph.size());
  return HCCL_SUCCESS;
}
}  // namespace hccl
