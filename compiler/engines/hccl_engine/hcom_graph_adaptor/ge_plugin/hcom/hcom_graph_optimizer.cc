/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <nlohmann/json.hpp>
#include "hcom_graph_optimizer.h"
#include "hcom_all_reduce_fusion.h"
#include "hcom_broadcast_fusion.h"
#include "hcom_reduce_fusion.h"
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
#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"
#include "acl/acl_rt.h"
#include "hcom_acl_adapter.h"

using namespace std;

namespace hccl {
const std::string NO_CALCULATION = "_NO_CALCULATION";
std::mutex g_setTaskNumCalModeLock;
HcomGraphOptimizer::HcomGraphOptimizer()
    : fusionTensorSizeLimit_(0), hcomMultiMode_(0), optionFeatureBaseRefreshable_(0) {}

HcomGraphOptimizer::~HcomGraphOptimizer() {}

ge::Status HcomGraphOptimizer::Initialize(const std::map<std::string, std::string> &options,
                                          ge::OptimizeUtility *const optimizeUtility) {
  HCCL_INFO("init hccl graph optimizer.");

  HcclResult ret = HcomGraphOptimizeInitialize(options, optimizeUtility);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][HcomGraphOptimizer] hcom init failed"), ge::INTERNAL_ERROR);

  return ge::SUCCESS;
}

HcclResult HcomGraphOptimizer::HcomGraphOptimizeInitialize(const map<std::string, std::string> &options,
                                                           ge::OptimizeUtility *const optimizeUtility) {
  HCCL_DEBUG("[Init][HcomGraphOptimizer] optimizeUtility[%p]", optimizeUtility);
  auto iter = options.find(ge::FUSION_TENSOR_SIZE);
  if (iter != options.end()) {
    u64 value = 0;
    HcclResult ret = SalStrToULonglong(iter->second, HCCL_BASE_DECIMAL, value);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][HcomGraphOptimizer]FUSION_TENSOR_SIZE[%s] is not a valid interger", iter->second.c_str()),
        HCCL_E_PARA);
    fusionTensorSizeLimit_ = value;
    HCCL_INFO("Initialize: FUSION_TENSOR_SIZE[%llu]Byte is setted.", fusionTensorSizeLimit_);
  } else {
    fusionTensorSizeLimit_ = 524288000;  // 默认融合tensor大小限制 524288000 = 500 * 1024 * 1024 = 500MB
    HCCL_INFO("Initialize: FUSION_TENSOR_SIZE is unsetted, default[%llu]Byte.", fusionTensorSizeLimit_);
  }

  auto iterMultiMode = options.find(ge::HCOM_MULTI_MODE);
  if (iterMultiMode != options.end()) {
    HcclResult ret = SalStrToInt(iterMultiMode->second, HCCL_BASE_DECIMAL, hcomMultiMode_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][HcomGraphOptimizer]HCOM_MULTI_MODE[%s] is not a valid interger",
                           iterMultiMode->second.c_str()),
                HCCL_E_PARA);
    if ((hcomMultiMode_ < 0) || (hcomMultiMode_ > 1)) {
      HCCL_ERROR("[Init][HcomGraphOptimizer]Initialize: HCOM_MULTI_MODE[%d] is invaild.", hcomMultiMode_);
      return HCCL_E_PARA;
    }
    HCCL_INFO("Initialize: HCOM_MULTI_MODE is %d.", hcomMultiMode_);
  } else {
    hcomMultiMode_ = 0;
    HCCL_INFO("Initialize: HCOM_MULTI_MODE is unsetted, default[%d].", hcomMultiMode_);
  }

  return HCCL_SUCCESS;
}

ge::Status HcomGraphOptimizer::OptimizeGraphPrepare(ge::ComputeGraph &graph) {
  HCCL_INFO("start hccl graph optimizer prepare.");

  GroupParaLabel groupLabels;
  std::string group;
  HcclResult ret;
  int label = 0;
  ge::TraceManager::GetInstance().SetTraceOwner("HCCL", "OptimizeGraphPrepare", graph.GetName());
  for (auto nodePtr : graph.GetAllNodes()) {
    if (!nodePtr) {
      HCCL_WARNING("OptimizeGraphPrepare: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("OptimizeGraphPrepare: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    // 集合通信的算子，设定format_Agnostic属性, 均设置成格式不敏感
    if (CheckSupportedOP(opDescPtr->GetType()) == HCCL_SUCCESS) {
      // 设置其他相关属性
      ret = SetHcomOpAttrs(opDescPtr);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Optimize][Graph]node[%s] set hcom op attrs failed, op type[%s]",
                             opDescPtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                  ge::INTERNAL_ERROR);
      // 若节点没有group
      bool bRet = ge::AttrUtils::HasAttr(opDescPtr, "group");
      if (!bRet) {
        group = "hccl_world_group";
      } else {
        bRet = ge::AttrUtils::GetStr(opDescPtr, "group", group);
        CHK_PRT_RET(!bRet,
                    HCCL_ERROR("[Optimize][Graph]node[%s] get attr \"group\" failed. ", nodePtr->GetName().c_str()),
                    ge::INTERNAL_ERROR);
        CHK_PRT_RET(group.empty(),
                    HCCL_ERROR("[Optimize][Graph]node[%s] get group is empty.", nodePtr->GetName().c_str()),
                    ge::INTERNAL_ERROR);
      }
      std::string setLabel = "hcom_op_";
      auto iterNodeLabel = groupLabels.find(group);
      if (iterNodeLabel == groupLabels.end()) {
        label++;
        setLabel += std::to_string(label);
        groupLabels.insert(std::make_pair(group, setLabel));
      } else {
        setLabel = iterNodeLabel->second;
      }
      ret = SetHcomOpParallelLabel(*nodePtr, setLabel);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Optimize][Graph]node[%s] set group para label attr failed, op type[%s]",
                             opDescPtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                  ge::INTERNAL_ERROR);
    }
  }
  ge::TraceManager::GetInstance().ClearTraceOwner();
  ge::GraphUtils::DumpGEGraph(graph.shared_from_this(), "HcclAfterOptimizeGraphPrepare");
  ge::GraphUtils::DumpGEGraphToOnnx(graph, "HcclAfterOptimizeGraphPrepare");
  HCCL_INFO("end hccl graph optimizer prepare.");
  return ge::SUCCESS;
}

ge::Status HcomGraphOptimizer::Finalize() {
  HCCL_INFO("finalize hccl graph optimizer.");
  return ge::SUCCESS;
}

HcclResult HcomGraphOptimizer::UpdateFusionTensorSizeLimit(bool unknownShape, u64 &fusionTensorSize) {
  // unkonwshape图，tensorsize不能大于200MB
  if (unknownShape) {
    u64 maxSize = 0;
    CHK_RET(GetCCLBufferAvailableSize(maxSize));
    if (static_cast<u64>(fusionTensorSizeLimit_) > maxSize) {
      fusionTensorSize = maxSize;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::HcomOptimizeOriginalGraph(ge::ComputeGraph &graph, bool &uknownShapeGraph) {
  HcclResult ret;
  // 更新下fusionTensorSize，使用实时更新获取的最大数值,在静态图下不用更新，动态图更新
  u64 fusionTensorSize = fusionTensorSizeLimit_;
  ret = UpdateFusionTensorSizeLimit(uknownShapeGraph, fusionTensorSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: UpdateFusionTensorSizeLimit graph"
                         "failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);

  ret = FuseHcomAllReduceNode(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: fuse HcomBroadcast node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);

  ret = FuseHcomBroadcastNode(graph, fusionTensorSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: fuse HcomReduce node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);

  ret = FuseHcomReduceNode(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: fuse HcomReduce node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              HCCL_E_PARA);

  return HCCL_SUCCESS;
}

ge::Status HcomGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph) {
  HcclResult ret;
  bool uknownShapeGraph = false;

  ge::GraphUtils::DumpGEGraph(graph.shared_from_this(), "HcclBeforeOptimizeOriginalGraph");
  ge::GraphUtils::DumpGEGraphToOnnx(graph, "HcclBeforeOptimizeOriginalGraph");
  ge::TraceManager::GetInstance().SetTraceOwner("HCCL", "OptimizeOriginalGraph", graph.GetName());
  ret = OriginalGraphShapeTypeCfg(graph, uknownShapeGraph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: OriginalGraphShapeTypeCfg failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ge::INTERNAL_ERROR);
  ret = SetUnknownShapeAttr(graph, uknownShapeGraph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: SetUnknownShapeAttr failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ge::INTERNAL_ERROR);

  ret = HcomOptimizeOriginalGraph(graph, uknownShapeGraph);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Optimize][OriginalGraph]graph[%s]: Original Optimize failed. ret[%d]", graph.GetName().c_str(), ret),
      ge::INTERNAL_ERROR);
  ge::TraceManager::GetInstance().ClearTraceOwner();
  ge::GraphUtils::DumpGEGraph(graph.shared_from_this(), "HcclAfterOptimizeOriginalGraph");
  ge::GraphUtils::DumpGEGraphToOnnx(graph, "HcclAfterOptimizeOriginalGraph");
  HCCL_INFO("graph[%s] end fusion HcomReduce Op.", graph.GetName().c_str());
  return ge::SUCCESS;
}

ge::Status HcomGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph &graph) {
  (void)graph;
  return ge::SUCCESS;
}

ge::Status HcomGraphOptimizer::OptimizeSubgraphPreProc(ge::ComputeGraph &graph) {
  bool uknownShapeGraph = false;

  HcclResult ret = OriginalGraphShapeTypeCfg(graph, uknownShapeGraph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][OptimizeSubgraphPreProc]graph[%s]: OriginalGraphShapeTypeCfg failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ge::INTERNAL_ERROR);

  ge::TraceManager::GetInstance().SetTraceOwner("HCCL", "OptimizeSubgraphPreProc", graph.GetName());
  for (auto nodePtr : graph.GetAllNodes()) {
    if (!nodePtr) {
      HCCL_WARNING("null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    if (CheckSupportedOP(opDescPtr->GetType()) != HCCL_SUCCESS) {
      continue;
    }
    ret = CalcOpRunningParam(*nodePtr, uknownShapeGraph);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Optimize][FusedGraph]errNo[0x%016llx] Calc Op Running Params failed.", HCOM_ERROR_CODE(ret)),
        ge::INTERNAL_ERROR);
  }
  ge::TraceManager::GetInstance().ClearTraceOwner();
  return ge::SUCCESS;
}

ge::Status HcomGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph &graph) {
  HCCL_DEBUG("[Optimize][OptimizeFusedGraph] graph[%s]", graph.GetName().c_str());
  return ge::SUCCESS;
}

ge::Status HcomGraphOptimizer::GetAttributes(ge::GraphOptimizerAttribute &attrs) const {
  attrs.engineName = HCCL_OPS_ENGIN;
  attrs.scope = ge::UNIT;
  HCCL_DEBUG("hccl graph optimizer get attr success. engine[%s] scope[%d]", attrs.engineName.c_str(), attrs.scope);
  return ge::SUCCESS;
}

HcclResult HcomGraphOptimizer::FuseHcomAllReduceNode(ge::ComputeGraph &graph) {
  HcomAllReduceFusion fusionHcomAllReduceOp;
  HCCL_INFO("graph[%s] start fusion HcomAllReduce node.", graph.GetName().c_str());
  HcclResult ret = fusionHcomAllReduceOp.Run(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Fuse][HcomAllReduceNode]graph[%s]: fuse HcomAllReduce node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    HcomAllReduceFusion fusionSubGraphOp;
    ret = fusionSubGraphOp.Run(*subgraph[index]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Fuse][HcomAllReduceNode]fuse HcomAllReduce op failed in subgraph[%s]. ret[%d]",
                           (*subgraph[index]).GetName().c_str(), ret),
                ret);
  }

  HCCL_INFO("graph[%s] with[%d] subgraphs: end fusion HcomAllReduce node.", graph.GetName().c_str(), subgraph.size());
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::FuseHcomBroadcastNode(ge::ComputeGraph &graph, u64 fusionTensorSize) {
  HcomBroadcastFusion fusionHcomBroadcastOp;
  HCCL_INFO("graph[%s] start fusion HcomBroadcast node.", graph.GetName().c_str());
  HcclResult ret = fusionHcomBroadcastOp.Run(graph, fusionTensorSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Fuse][HcomBroadcastNode]graph[%s]: fuse HcomBroadcast node failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    HcomBroadcastFusion fusionSubGraphOp;
    CHK_RET(fusionSubGraphOp.Run(*subgraph[index], fusionTensorSize));
  }

  HCCL_INFO("graph[%s] with [%d] subgraphs: end fusion HcomBroadcast node.", graph.GetName().c_str(), subgraph.size());
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetHcomOpAttrs(ge::OpDescPtr &opDescPtr) {
  // 连续内存属性从图编译移至图优化
  bool bRet = false;
  if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE || opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_BROADCAST ||
      opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCE) {
    // Note: 需要融合的算子需设定输入/输出连续属性
    bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
    CHK_PRT_RET(
        !bRet,
        HCCL_ERROR("[Set][OpAttrs]node[%s] set continuous input attr to OpDesc failed", opDescPtr->GetName().c_str()),
        HCCL_E_INTERNAL);
    bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_CONTINUOUS_OUTPUT, true);
    CHK_PRT_RET(
        !bRet,
        HCCL_ERROR("[Set][OpAttrs]node[%s] set continuous output attr to OpDesc failed.", opDescPtr->GetName().c_str()),
        HCCL_E_INTERNAL);
  }

  if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE ||
      opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCESCATTER || opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCE) {
    bRet = ge::AttrUtils::SetBool(opDescPtr, "_input_mutable", true);
    HCCL_DEBUG("node[%s] op type [%s] input mutable attr is set", opDescPtr->GetName().c_str(),
               opDescPtr->GetType().c_str());
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpAttrs]node[%s] SetBool _input_mutable failed, op type[%s]",
                           opDescPtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                HCCL_E_INTERNAL);
  }

  if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE || opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCE ||
      opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
    // 设置溢出检测属性
    int globalWorkSpaceSize = 1;
    int globalWorkSpaceType = static_cast<int>(GlobalWorkSpaceType::OVERFLOW_DETECT_MODE);
    bRet = ge::AttrUtils::SetInt(opDescPtr, "globalworkspace_size", globalWorkSpaceSize);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpAttrs]node[%s] set globalworkspace_size failed", opDescPtr->GetName().c_str()),
                HCCL_E_INTERNAL);
    bRet = ge::AttrUtils::SetInt(opDescPtr, "globalworkspace_type", globalWorkSpaceType);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpAttrs]node[%s] set globalworkspace_type failed", opDescPtr->GetName().c_str()),
                HCCL_E_INTERNAL);
  }

  HcclResult ret = SetHcomOpFormat(opDescPtr);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Set][HcomOpAttrs]node[%s] set format failed, op type[%s]", opDescPtr->GetName().c_str(),
                         opDescPtr->GetType().c_str()),
              HCCL_E_INTERNAL);

  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::FuseHcomReduceNode(ge::ComputeGraph &graph) {
  HcomReduceFusion fusionHcomReduceOp;
  HCCL_INFO("graph[%s] start fusion HcomReduce node.", graph.GetName().c_str());
  HcclResult ret = fusionHcomReduceOp.Run(graph);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Fuse][HcomReduceNode]graph[%s]: fuse HcomReduce node failed. ret[%d]", graph.GetName().c_str(), ret),
      ret);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    HcomReduceFusion fusionSubGraphOp;
    ret = fusionSubGraphOp.Run(*subgraph[index]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Fuse][HcomReduceNode]fuse HcomReduce op failed in subgraph[%s]. ret[%d]",
                           (*subgraph[index]).GetName().c_str(), ret),
                ret);
  }

  HCCL_INFO("graph[%s] with [%d] subgraphs: end fusion HcomReduce node.", graph.GetName().c_str(), subgraph.size());
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::CheckSupportedOP(const std::string &sCollectiveType) const {
  std::vector<std::string>::const_iterator it =
      std::find(HCOM_SUPPORTED_OP_TYPE.begin(), HCOM_SUPPORTED_OP_TYPE.end(), sCollectiveType);
  return (it != HCOM_SUPPORTED_OP_TYPE.end()) ? HCCL_SUCCESS : HCCL_E_PARA;
}

bool HcomGraphOptimizer::IsSubgraphMultiBatch(ge::ComputeGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!node) {
      HCCL_WARNING("null node exists.");
      continue;
    }
    auto opDescPtr = node->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("desc of node[%s] is null.", node->GetName().c_str());
      continue;
    }
    // 如果有ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE,则为判断自动分档
    if (ge::AttrUtils::HasAttr(opDescPtr, ge::ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE)) {
      HCCL_INFO("graph[%s] node [%s] has attr _subgraph_multi_dims_input_shape", graph.GetName().c_str(),
                node->GetName().c_str());
      return true;
    }
  }
  return false;
}

HcclResult HcomGraphOptimizer::OriginalGraphShapeTypeCfg(ge::ComputeGraph &graph, bool &uknownShapeGraph) {
  // 遍历原图所有算子，如果是自动分档，则默认整图动态shape
  if (IsSubgraphMultiBatch(graph)) {
    uknownShapeGraph = false;
    return HCCL_SUCCESS;
  }

  /* 遍历原图所有算子 */
  for (auto nodePtr : graph.GetAllNodes()) {
    if (!nodePtr) {
      HCCL_WARNING("null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    bool unknownShapeNode = false;
    /* 判断算子是不是unknownShapeNode，有一个算子是unknown，原图就是unknown */
    CHK_PRT_RET(
        (ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, unknownShapeNode) != ge::GRAPH_SUCCESS),
        HCCL_ERROR("[Original][GraphShapeTypeCfg]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
        HCCL_E_PARA);
    if (unknownShapeNode) {
      uknownShapeGraph = true;
      break;
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetUnknownShapeAttr(ge::ComputeGraph &graph, bool uknownShapeGraph) {
  std::string iterRefreShable = "0";
  ge::graphStatus status = ge::GetContext().GetOption(ge::OPTION_FEATURE_BASE_REFRESHABLE, iterRefreShable);

  bool statusFlag = (status == ge::GRAPH_SUCCESS) && (iterRefreShable.compare("1") == 0);
  if (statusFlag) {
    HcclResult ret = SalStrToInt(iterRefreShable, HCCL_BASE_DECIMAL, optionFeatureBaseRefreshable_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][HcomGraphOptimizer]OPTION_FEATURE_BASE_REFRESHABLE[%s] is not a valid interger",
                           iterRefreShable.c_str()),
                HCCL_E_PARA);
    if ((optionFeatureBaseRefreshable_ < 0) || (optionFeatureBaseRefreshable_ > 1)) {
      HCCL_ERROR("[Init][HcomGraphOptimizer]Initialize: OPTION_FEATURE_BASE_REFRESHABLE[%d] is invaild.",
                 optionFeatureBaseRefreshable_);
      return HCCL_E_PARA;
    }
    HCCL_INFO("Initialize: OPTION_FEATURE_BASE_REFRESHABLE is %d.", optionFeatureBaseRefreshable_);
  } else {
    optionFeatureBaseRefreshable_ = 0;
    HCCL_INFO("Initialize: OPTION_FEATURE_BASE_REFRESHABLE is unsetted, default[%d].", optionFeatureBaseRefreshable_);
  }

  if (!uknownShapeGraph && optionFeatureBaseRefreshable_ == 0) {
    HCCL_DEBUG("graph[%s] is known shape", graph.GetName().c_str());
    return HCCL_SUCCESS;
  }
  /* 遍历原图所有算子 */
  for (auto nodePtr : graph.GetAllNodes()) {
    if (!nodePtr) {
      HCCL_WARNING("null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    if (CheckSupportedOP(opDescPtr->GetType()) != HCCL_SUCCESS) {
      continue;
    }

    bool bRet = false;
    bRet = ge::AttrUtils::SetInt(opDescPtr, ORIGINAL_GRAPH_SHAPE_TYPE, ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][UnknownShapeAttr]graph[%s]: node [%s] SetInt unknown shape failed, op type[%s]",
                           graph.GetName().c_str(), nodePtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                HCCL_E_PARA);

    bool unknownShapeNode = false;
    /* 判断算子是不是unknownShapeNode */
    CHK_PRT_RET(
        (ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, unknownShapeNode) != ge::GRAPH_SUCCESS),
        HCCL_ERROR("[Set][UnknownShapeAttr]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
        HCCL_E_PARA);

    if (unknownShapeNode) {
      continue;
    }

    // alltoallv、alltoallvc、alltoall暂不支持走动态shape下沉
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLTOALLV || opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLTOALL ||
        opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLGATHERV ||
        opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
      bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
      HCCL_DEBUG("graph[%s]: node [%s] op type [%s] unknown shape value is set", graph.GetName().c_str(),
                 nodePtr->GetName().c_str(), opDescPtr->GetType().c_str());
      CHK_PRT_RET(!bRet,
                  HCCL_ERROR("[Set][UnknownShapeAttr]graph[%s]: node [%s] SetBool unknown shape failed, op type[%s]",
                             graph.GetName().c_str(), nodePtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                  HCCL_E_PARA);
      continue;
    }

    /* 对动态shap下的集合通信known算子内存进行检查，如果大于ccl
     * buf，就把算子修改为unkown。GE会按照单算子模式调用 */
    uint64_t memSize = 0;
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLGATHER || opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_RECEIVE) {
      CHK_RET(HcomOpUtils::GetAllOutputsTensorMemSize(opDescPtr, memSize));
    } else {
      CHK_RET(HcomOpUtils::GetAllInputsTensorMemSize(opDescPtr, memSize));
    }

    HcomOpParam hcomOpParam;
    HcomResResponse hcomResResponse;
    std::string sCollectiveType;
    std::string sGroup;
    std::string socVersion;
    std::vector<int64_t> sendCountMatrix;
    std::vector<int64_t> sendCounts;
    std::vector<int64_t> sendDispls;
    std::vector<int64_t> recvCounts;
    std::vector<int64_t> recvDispls;
    std::vector<u32> curRanks;
    std::string rankTableStr;
    std::string rankTableM;

    CHK_RET(SetHcomOpParam(*nodePtr, &hcomOpParam, sCollectiveType, sGroup, socVersion, sendCountMatrix, sendCounts,
                           sendDispls, recvCounts, recvDispls, curRanks, rankTableStr, rankTableM));

    CHK_RET(HcomCalcOpOnline(&hcomOpParam, &hcomResResponse));
    u64 maxSize = 0;
    CHK_RET(GetCCLBufferAvailableSize(maxSize));
    u64 opMemSize = hcomResResponse.opMemSize;

    // 适配1,2包，RTS不支持二级地址偏移拷贝
    if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS) {
      HCCL_ERROR("[offline][compilation] get soc version failed.");
      return HCCL_E_INTERNAL;
    }

    bool supportSecAddrCopyWithOffset = HcomGetSecAddrCopyFlag(socVersion.c_str());
    HCCL_INFO("[SetUnknownShapeAttr] supportSecAddrCopyWithOffset %d", supportSecAddrCopyWithOffset);
    if (static_cast<u64>(memSize) > maxSize) {
      if (supportSecAddrCopyWithOffset) {
        HCCL_DEBUG("Supports level-2 address offset copy.");
      } else {
        bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
        HCCL_DEBUG("graph[%s]: node [%s] op type [%s] unknown shape value is set", graph.GetName().c_str(),
                   nodePtr->GetName().c_str(), opDescPtr->GetType().c_str());
        CHK_PRT_RET(!bRet,
                    HCCL_ERROR("[Set][UnknownShapeAttr]graph[%s]: node [%s] SetBool unknown shape failed, op type[%s]",
                               graph.GetName().c_str(), nodePtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                    HCCL_E_PARA);
      }
    }

    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLTOALLVC && opMemSize > maxSize) {
      bRet = ge::AttrUtils::SetBool(opDescPtr, ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
      HCCL_DEBUG("graph[%s]: node [%s] op type [%s] unknown shape value is set", graph.GetName().c_str(),
                 nodePtr->GetName().c_str(), opDescPtr->GetType().c_str());
      CHK_PRT_RET(!bRet,
                  HCCL_ERROR("[Set][UnknownShapeAttr]graph[%s]: node [%s] SetBool unknown shape failed, op type[%s]",
                             graph.GetName().c_str(), nodePtr->GetName().c_str(), opDescPtr->GetType().c_str()),
                  HCCL_E_PARA);
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetHcomOpFormat(ge::OpDescPtr &opDescPtr) {
  CHK_SMART_PTR_NULL(opDescPtr);
  bool bRet = false;
  if (hcomMultiMode_ == 0) {
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE ||
        opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_BROADCAST) {
      bRet = ge::AttrUtils::SetInt(opDescPtr, "_format_agnostic", HCCL_FORMAT_PAIRED_INPUT_OUTPUT);
      HCCL_DEBUG("op type[%s] format agnostic value is set", opDescPtr->GetType().c_str());
      CHK_PRT_RET(
          !bRet, HCCL_ERROR("[Set][OpFormat]SetBool format_Agnostic failed, op type[%s]", opDescPtr->GetType().c_str()),
          HCCL_E_PARA);
    }
  } else {
    // npu为了加速计算在不同的rank上可能将算子的逻辑format转换为不同的物理format进行运算，
    // 同一tensor在不同format时，内存布局、length均可能不同。
    // 为避免在不同rank上算子的物理format不一致，导致集合通信结果异常，需要将所有节点上的通信算子format设为同一类型。
    // 此处先行将所有的通信算子的format固定设置为NHWC，后续计划引入节点间format协商机制。
    size_t inputSize = opDescPtr->GetAllInputsSize();
    for (size_t i = 0; i < inputSize; ++i) {
      auto inTensorDescPtr = opDescPtr->MutableInputDesc(i);
      inTensorDescPtr->SetFormat(ge::FORMAT_NHWC);
      HCCL_DEBUG("input[%zu / %zu] has been setted foramt with NHWC.", i, inputSize);
    }
    size_t outputSize = opDescPtr->GetOutputsSize();
    for (size_t i = 0; i < outputSize; ++i) {
      auto outTensorDescPtr = opDescPtr->MutableOutputDesc(i);
      outTensorDescPtr->SetFormat(ge::FORMAT_NHWC);
      HCCL_DEBUG("output[%zu / %zu] has been setted foramt with NHWC.", i, outputSize);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetHcomOpParallelLabel(ge::Node &node, std::string groupLabel) {
  auto opDesc = node.GetOpDesc();
  if (ge::AttrUtils::HasAttr(opDesc, ge::ATTR_NAME_PARALLEL_GROUP)) {
    // 如果框架已指定算子的并行分组方式，则通信库不再重新指定。
    string currentLabel;
    if (ge::AttrUtils::GetStr(opDesc, ge::ATTR_NAME_PARALLEL_GROUP, currentLabel)) {
      HCCL_INFO("[Set][HcomOpParallelLabel] attr \"_parallel_group\" (%s) has existed.", currentLabel.c_str());
    } else {
      HCCL_INFO("[Set][HcomOpParallelLabel] attr \"_parallel_group\" has existed.");
    }
    return HCCL_SUCCESS;
  }

  ge::graphStatus geRet = ge::NodeUtils::SetNodeParallelGroup(node, groupLabel.c_str());
  CHK_PRT_RET(geRet != ge::GRAPH_SUCCESS,
              HCCL_ERROR("[Set][OpParallelLabel]errNo[0x%016llx] node[%s] op[%s] set para label[%s] failed",
                         HCCL_ERROR_CODE(HCCL_E_INTERNAL), opDesc->GetName().c_str(), opDesc->GetType().c_str(),
                         groupLabel.c_str()),
              HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::HcomGetAccuracyCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                            HcclDataType dataType, u64 &count, u32 rankSize) {
  u32 dataTypeSize = 0;
  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  CHK_PRT_RET(dataTypeSize == 0,
              HCCL_ERROR("[Get][CountFromOpDesc]dataType size is zero."),
              HCCL_E_PARA);
  
  // Receive 算子不支持获取count
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    HCCL_WARNING("[%s][Get][Count] op[%s] get count failed. receive op not support get count.",
              __func__, sCollectiveType.c_str());
    return HCCL_SUCCESS;
  }

  // broadcast等搬运算子在图优化阶段没有input，只有output，无法通过getsize方式获取到数据量，需要用memoutput的方式获取
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    CHK_RET(GetMemOutPutForCountCalc(op, sCollectiveType, dataTypeSize, count));
    return HCCL_SUCCESS;
  }
  // 其他算子通用处理
  CHK_RET(HcomOpUtils::CalcCommonCount(op, sCollectiveType, dataTypeSize, rankSize, count));
  return HCCL_SUCCESS;
}

// broadcast等搬运算子在图优化阶段没有input，只有output，无法通过getsize方式获取到数据量，需要用memoutput的方式获取
HcclResult HcomGraphOptimizer::MemOutputForOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                  u32 i, int64_t &memSize) {
  ge::GeTensorDesc outputTensor = op->GetOutputDesc(i);
  ge::GeShape outputShape = outputTensor.GetShape();
  ge::Format format = outputTensor.GetFormat();
  ge::DataType dataType = outputTensor.GetDataType();
  // 获取内存大小
  bool bErr = (ge::GRAPH_SUCCESS != ge::TensorUtils::CalcTensorMemSize(outputShape, format, dataType, memSize));
  CHK_PRT_RET(bErr,
              HCCL_ERROR("[Set][OpOutputMemSize]In get output mem size, error outputSize because no"
                          "know shape, Format[%d], dataType[%d], outputSize[%lld], index[%u]",
                          format, dataType, memSize, i),
              HCCL_E_PARA);

  if (memSize == -1) {  // memsize 为-1 时，表示输入的shape不正确
    HCCL_ERROR(
        "[Set][OpOutputMemSize]In get output mem size, error outputSize because unknow shape,"
        "Format[%d], dataType[%d], outputSize[%lld], index[%u]",
        format, dataType, memSize, i);
    return HCCL_E_PARA;
  }

  // 根据 规则重新计算内存大小
  CHK_RET(CalcHCCLOutputMemSize(sCollectiveType, memSize));

  // 更新output Tensor
  if (op->UpdateOutputDesc(i, outputTensor) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR(
        "[Set][OpOutputMemSize]In get output mem size, update output desc error,"
        "Format[%d], dataType[%d], outputSize[%lld], index[%u]",
        format, dataType, memSize, i);
    return HCCL_E_PARA;
  }
  
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::GetMemOutPutForCountCalc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                        u32 dataTypeSize, u64& count) {
  constexpr u32 alignSize = 512;  // 以512字节对齐
  u64 totalSize = 0;
  for (u32 i = 0; i < op->GetOutputsSize(); i++) {
    int64_t memSize = 0;
    CHK_RET(MemOutputForOpDesc(op, sCollectiveType, i, memSize));
    // 对齐到512字节的倍数
    CHK_PRT_RET((static_cast<u64>(memSize) > INVALID_U64 - alignSize),
                HCCL_ERROR("[Set][OpOutputMemSize]op[%s] memSize[%lld] is overflow.",
                          op->GetName().c_str(), memSize),
                HCCL_E_PARA);
    memSize = (static_cast<u64>(memSize) + alignSize - 1) / alignSize * alignSize;
    totalSize += memSize;
  }
  count = totalSize / dataTypeSize;
  HCCL_INFO("[%s]In get output MemSize, sCollectiveType[%s], get count[%llu]", __func__, sCollectiveType.c_str(), count);
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::GetCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                  HcclDataType dataType, u64 &count) {
  HcclResult ret;
  u64 totalSize = 0;
  u32 dataTypeSize = 0;

  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[Get][CountFromOpDesc]dataType size is zero."), HCCL_E_PARA);

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    ret = GetHcomReceiveOpOutputSize(op, dataTypeSize, totalSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][Count]get op[%s] output size failed. ret[%d]", sCollectiveType.c_str(), ret), ret);
  } else {
    for (u64 i = 0; i < op->GetInputsSize(); i++) {
      u64 blockSize;
      CHK_SMART_PTR_NULL(op->GetInputDescPtr(i));
      if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
        // ReduceScatterV 算子的 count 为总数据的个数，count = input的size / dataTypeSize
        u64 shapeSize = 0;
        if (op->GetInputDescPtr(i)->GetShape().IsScalar()) {
          shapeSize = 1;
        } else {
          shapeSize = op->GetInputDescPtr(i)->GetShape().GetShapeSize();
        }
        CHK_PRT_RET((shapeSize > INVALID_U64 / dataTypeSize),
                    HCCL_ERROR("[Get][Count]op[%s] "
                               "shape size[%llu] is overflow.",
                               sCollectiveType.c_str(), shapeSize),
                    HCCL_E_PARA);
        const u32 paddingLen = 1024;  // 每个输入额外多申请 1024 bytes 的workspace memory。
        blockSize = (shapeSize * dataTypeSize + paddingLen);
      } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        s32 rankSize = 0;
        // ReduceScatter 算子的 count 为输出数据的个数，count = (input的size / rank_size) / dataTypeSize
        CHK_PRT_RET(
            (!ge::AttrUtils::GetInt(op, HCOM_ATTR_RANK_SIZE, rankSize)),
            HCCL_ERROR("[Get][Count]op[%s] get attr[%s] failed.", sCollectiveType.c_str(), HCOM_ATTR_RANK_SIZE.c_str()),
            HCCL_E_PARA);
        CHK_PRT_RET((rankSize <= 0),
                    HCCL_ERROR("[Get][Count]errNo[0x%016llx] in ReduceScatter op,"
                               "rank_size[%d] should be greater than 0.",
                               HCOM_ERROR_CODE(HCCL_E_PARA), rankSize),
                    HCCL_E_PARA);
        u64 shapeSize = 0;
        if (op->GetInputDescPtr(i)->GetShape().IsScalar()) {
          shapeSize = 1;
        } else {
          shapeSize = (u64)op->GetInputDescPtr(i)->GetShape().GetShapeSize();
        }
        CHK_PRT_RET((shapeSize > INVALID_U64 / dataTypeSize),
                    HCCL_ERROR("[Get][Count]op[%s] shape size[%llu]"
                               "is overflow.",
                               sCollectiveType.c_str(), shapeSize),
                    HCCL_E_PARA);
        // reduce-scatter 融合场景：reduce-scatter算子的每个输入tensor均有补齐处理。
        // mindspore 补齐规则：(size + 32  -1 + 512) / 512 * 512
        // 因此，此处每个输入额外多申请 1024 bytes 的workspace memory。
        const u32 paddingLen = 1024;  // 每个输入额外多申请 1024 bytes 的workspace memory。
        blockSize = (shapeSize * dataTypeSize + paddingLen) / rankSize;
      } else {
        const u32 alignSize = 512;  // 以512 Byte 对齐
        int64_t inputSize = 0;
        CHK_PRT_RET((ge::GRAPH_SUCCESS != ge::TensorUtils::GetSize(*op->GetInputDescPtr(i), inputSize)),
                    HCCL_ERROR("[Get][Count]errNo[0x%016llx] get workspace bytes failed. get size from TensorDesc"
                               "failed, op : %s"
                               ", input index : %llu",
                               HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), i),
                    HCCL_E_PARA);
        CHK_PRT_RET((static_cast<u64>(inputSize) > INVALID_U64 - alignSize),
                    HCCL_ERROR("[Get][Count]op[%s] input"
                               "size[%llu] is overflow.",
                               sCollectiveType.c_str(), static_cast<u64>(inputSize)),
                    HCCL_E_PARA);
        blockSize = (static_cast<u64>(inputSize) + alignSize - 1) / alignSize * alignSize;
      }
      totalSize = totalSize + blockSize;
    }
  }
  count = totalSize / dataTypeSize;
  HCCL_INFO("op[%s] get count[%llu] success.", sCollectiveType.c_str(), count);
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::GetCommFromOpDesc(const ge::OpDescPtr &op, int64_t &hcomComm, std::string &sGroup) {
  if (ge::AttrUtils::HasAttr(op, "comm")) {
    if (ge::AttrUtils::GetInt(op, "comm", hcomComm) == false) {
      HCCL_ERROR("[GetComm][OpDesc]errNo[0x%016llx]: get comm failed. get \"comm\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    } else if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      HCCL_INFO("[HcclCommGraph][Type]get comm equal to 0, should get group.");
      CHK_RET(HcomOpUtils::GetGroupFromOpDesc(op, sGroup));
    } else {
      HCCL_INFO("[HcclCommGraph][Type]get comm name[%lld] success.", hcomComm);
    }
  } else {
    CHK_RET(HcomOpUtils::GetGroupFromOpDesc(op, sGroup));
  }
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetOpOutputMemSize(ge::Node &node, const std::string &sCollectiveType) {
  ge::OpDescPtr op = node.GetOpDesc();
  for (u32 i = 0; i < op->GetOutputsSize(); i++) {
    ge::GeTensorDesc outputTensor = op->GetOutputDesc(i);
    int64_t memSize = 0;
    CHK_RET(MemOutputForOpDesc(op, sCollectiveType, i, memSize));
    // 将内存大小重新传给上层
    ge::TensorUtils::SetSize(outputTensor, memSize);
    HCCL_INFO("In set output MemSize, sCollectiveType[%s], opMemSize[%lld]", sCollectiveType.c_str(), memSize);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::CalcHCCLOutputMemSize(const std::string &sCollectiveType, int64_t &memSize) {
  HCCL_DEBUG("[HcomGraphOptimizer][CalcHCCLOutputMemSize] sCollectiveType[%s] memSize[%lld]", sCollectiveType.c_str(),
             memSize);
  const u32 MEMORY_ALIGN_RATIO = 2;  // GE要求内存需要32KB对齐后，再外加32KB. out = (in + 2 * 32 - 1) / 32 * 32
  const u32 MEMORY_ALIGN_SIZE = 32;  // GE要求内存需要32KB对齐后，再外加32KB. out = (in + 2 * 32 - 1) / 32 * 32
  // GE要求内存需要32KB对齐后，再外加32KB
  memSize = ((memSize + MEMORY_ALIGN_RATIO * MEMORY_ALIGN_SIZE - 1) / MEMORY_ALIGN_SIZE) * MEMORY_ALIGN_SIZE;
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetOpMemAttr(ge::Node &node, const std::string &sCollectiveType, const u64 &opMemSize) {
  bool bRet = false;

  // ATTENTION: 算子在IR定义时input/output同名场合（参考HcomRemoteRefRead算子）会隐式设置reference属性为TRUE,
  //   此处只对IR定义中input/output不同名且需要复用内存的算子，进行内存复用配置。
  //   后续有类似算子实现建议在IR定义时将input/output配置为相同name。
  // broadcast算子因为输入/输出为同一内存Ref属性为true
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    bRet = ge::AttrUtils::SetBool(node.GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx] op[%s]: set  reference attr[%d] to OpDesc"
                           "failed.",
                           HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), true),
                HCCL_E_PARA);
    bRet = node.GetOpDesc()->UpdateOutputName(node.GetOpDesc()->GetAllInputName());
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx] op[%s]: update output name failed.",
                           HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str()),
                HCCL_E_PARA);
    HCCL_INFO("node[%s] set attr [reference]: %u", node.GetName().c_str(), true);

    // 算子属性为reference时，为减少GE的内存分配，设置 ouput 复用 input 内存
    for (uint32_t i = 0; i < static_cast<uint32_t>(node.GetOpDesc()->GetOutputsSize()); i++) {
      auto outDescPtr = node.GetOpDesc()->MutableOutputDesc(i);
      CHK_SMART_PTR_NULL(outDescPtr);
      ge::TensorUtils::SetReuseInput(*outDescPtr, true);
      ge::TensorUtils::SetReuseInputIndex(*outDescPtr, i);
    }
  } else {
    HCCL_INFO("node[%s] set attr [reference]: skip", node.GetName().c_str());
  }

  string groupListString;

  std::string sGroup;
  CHK_RET(HcomOpUtils::GetGroupFromOpDesc(node.GetOpDesc(), sGroup));

  std::string socVersion;
  if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("[offline][compilation] get soc version failed.");
    return HCCL_E_INTERNAL;
  }

  bool withoutImplCompile =
      IsOfflineCompilation() ||
      (ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, groupListString) == ge::GRAPH_SUCCESS);

  // 针对310p duo卡 2p场景申请内存为普通内存，不需要单独设置，其余场景需要设置申请为p2p内存
  // 板卡推理不需要设置申请内存为p2p
  u32 memType = 0;
  u32 p2pMemType = RT_MEMORY_P2P_DDR;
  CHK_RET(HcomGetMemType(sGroup.c_str(), socVersion.c_str(), false, &memType, nullptr, withoutImplCompile));
  if (memType == p2pMemType) {
    vector<int64_t> memTypeInput(node.GetOpDesc()->GetInputsSize(), p2pMemType);
    vector<int64_t> memTypeOutput(node.GetOpDesc()->GetOutputsSize(), p2pMemType);
    vector<int64_t> memTypeWorkSpace(1, p2pMemType);
    bool ret = ge::AttrUtils::SetListInt(node.GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, memTypeInput);
    CHK_PRT_RET(!ret,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx]: Set input mem addr failed. op[%s]", HCCL_E_PARA,
                           sCollectiveType.c_str()),
                HCCL_E_PARA);

    ret = ge::AttrUtils::SetListInt(node.GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memTypeOutput);
    CHK_PRT_RET(!ret,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx]: Set output mem addr failed. op[%s]", HCCL_E_PARA,
                           sCollectiveType.c_str()),
                HCCL_E_PARA);

    if (opMemSize != 0) {
      ret = ge::AttrUtils::SetListInt(node.GetOpDesc(), ge::ATTR_NAME_WORKSPACE_TYPE_LIST, memTypeWorkSpace);
      CHK_PRT_RET(!ret,
                  HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx]: Set workspace mem addr failed. op[%s]", HCCL_E_PARA,
                             sCollectiveType.c_str()),
                  HCCL_E_PARA);
    }
    HCCL_INFO("[Set][OpMemAttr] Set memType p2p mem type");
  }

  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetOpAtomicInputIndex(ge::Node &node, const std::string &sCollectiveType) {
  // allreduce，reduce 算子设定atomic Input Index属性
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE) {
    vector<int64_t> atomicInputIndex(1, -1);  // 回传vector的值为-1，作为标志位
    if (!ge::AttrUtils::SetListInt(node.GetOpDesc(), "atomic_input_index", atomicInputIndex)) {
      HCCL_ERROR("[Set][OpAtomicInputIndex]errNo[0x%016llx]: set op[%s] atomic index failed.",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::CalcOpRunningParam(ge::Node &node, bool uknownShapeGraph) {
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
  CHK_PRT_RET(
      !node.GetOpDesc(),
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetOpDesc failed. null ptr.", HCOM_ERROR_CODE(HCCL_E_PTR)),
      HCCL_E_PTR);

  bool unknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(node, unknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Calc][OpRunningParam]node[%s] get node unknown status failed", node.GetName().c_str()),
              HCCL_E_PARA);
  if (unknownShapeNode) {
    HCCL_INFO("node:%s is unknown shape, does not need to Calc Op Running Param", node.GetName().c_str());
    HcomSetWorkflowMode(lastWorkflowMode);
    return HCCL_SUCCESS;
  }

  CHK_RET(HcomCalcOpRunningParam(node, uknownShapeGraph));
  HcomSetWorkflowMode(lastWorkflowMode);
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::GetHcomReceiveOpOutputSize(const ge::OpDescPtr &op, u32 dataTypeSize, u64 &outputSize) {
  CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[Receive][OpOutputSize]dataType size is zero."), HCCL_E_PARA);

  std::string sCollectiveType = op->GetType();
  CHK_PRT_RET(
      (!ge::AttrUtils::HasAttr(op, HCOM_ATTR_SHAPE)),
      HCCL_ERROR("[Receive][OpOutputSize]op[%s] has no attr[%s].", sCollectiveType.c_str(), HCOM_ATTR_SHAPE.c_str()),
      HCCL_E_PARA);

  vector<int64_t> shapeDims;
  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, HCOM_ATTR_SHAPE, shapeDims)),
              HCCL_ERROR("[Receive][OpOutputSize]op[%s] get attr[%s] failed.", sCollectiveType.c_str(),
                         HCOM_ATTR_SHAPE.c_str()),
              HCCL_E_PARA);

  u64 shapeSize = 0;
  if (shapeDims.empty()) {
    // HcomReceive算子标量的话将shapeSize设置为1
    shapeSize = 1;
  } else {
    shapeSize = static_cast<u64>(ge::Shape(shapeDims).GetShapeSize());
  }
  const u32 alignSize = 512;  // 以512 Byte 对齐
  CHK_PRT_RET(
      (shapeSize > (INVALID_U64 - alignSize) / dataTypeSize),
      HCCL_ERROR("[Receive][OpOutputSize]op[%s] shape size[%llu] is overflow.", sCollectiveType.c_str(), shapeSize),
      HCCL_E_PARA);
  outputSize = (static_cast<u64>(shapeSize * dataTypeSize) + alignSize - 1) / alignSize * alignSize;
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::HcomCalcOpRunningParam(ge::Node &node, bool uknownShapeGraph) {
  HCCL_INFO("calculate hccl runing parameters start.");

  HcclResult ret;
  HcomOpParam hcomOpParam;
  HcomResResponse hcomResResponse;
  std::string sCollectiveType;
  std::string sGroup;
  std::string socVersion;
  std::vector<int64_t> sendCountMatrix;
  std::vector<int64_t> sendCounts;
  std::vector<int64_t> sendDispls;
  std::vector<int64_t> recvCounts;
  std::vector<int64_t> recvDispls;
  std::vector<u32> curRanks;
  std::string rankTableStr;
  std::string rankTableM;

  CHK_RET(SetHcomOpParam(node, &hcomOpParam, sCollectiveType, sGroup, socVersion, sendCountMatrix, sendCounts,
                         sendDispls, recvCounts, recvDispls, curRanks, rankTableStr, rankTableM));

  if (IsOfflineCompilation() || hcomOpParam.groupListSize != 0) {
    CHK_RET(HcomCalcOpResOffline(&hcomOpParam, &hcomResResponse));
  } else {
    CHK_RET(HcomCalcOpOnline(&hcomOpParam, &hcomResResponse));
  }

  std::string nodeName = node.GetName();
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_SEND || sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE ||
      (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST && nodeName.find(NO_CALCULATION) != std::string::npos)) {
    // 重新刷新从流为0
    hcomResResponse.streamNum = 0;
  }

  if (ge::AttrUtils::SetInt(node.GetOpDesc(), "used_stream_num", hcomResResponse.streamNum) == false) {
    HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op[%s]: set stream number[%llu] to OpDesc failed.", HCCL_E_PARA,
               hcomOpParam.opType, hcomResResponse.streamNum);
    return HCCL_E_INTERNAL;
  }

  CHK_RET(SetOpWorkerSpaceForKnowShape(node, hcomResResponse.opMemSize));
  ret = SetOpMemAttr(node, node.GetOpDesc()->GetType(), hcomResResponse.opMemSize);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set node[%s] mem attr failed.", ret, node.GetName().c_str()),
      HCCL_E_INTERNAL);

  // 设置output size 大小
  ret = SetOpOutputMemSize(node, hcomOpParam.opType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set op[%s] output size failed.", ret, hcomOpParam.opType),
      HCCL_E_INTERNAL);

  // 设定atomic index参数
  ret = SetOpAtomicInputIndex(node, hcomOpParam.opType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set op[%s] atomic input index failed.", ret,
                         hcomOpParam.opType),
              HCCL_E_INTERNAL);

  HCCL_INFO(
      "[Calc][OpRunningParam] node[%s] calculate hccl runing parameters completed. stream num:[%llu], workspace "
      "size:[%llu]bytes",
      node.GetName().c_str(), hcomResResponse.streamNum, hcomResResponse.opMemSize);

  if (uknownShapeGraph) {  // 动态图+集合通信算子+send/recv
    // 计算清零task数量，累加到hcomResResponse算出的taskNum
    u32 taskNum = static_cast<u32>(hcomResResponse.taskNum);
    u32 cleanTaskNum = 0;
    CHK_RET(HcomOpUtils::GetTensorCleanTaskNum(node, sCollectiveType, cleanTaskNum));
    taskNum += cleanTaskNum;
    CHK_RET(CheckForceUnknown(node, taskNum));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::SetOpWorkerSpaceForKnowShape(ge::Node &node, u64 &opMemSize) {
  u32 shapeType = ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;
  CHK_RET(GetOriginalGraphShapeTypeFromDesc(node.GetOpDesc(), shapeType));
  if (shapeType == ORIGINAL_GRAPH_KNOWNSHAPE_TYPE) {
    std::vector<int64_t> workspaceBytes;
    workspaceBytes.push_back(opMemSize);
    node.GetOpDesc()->SetWorkspaceBytes(workspaceBytes);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::CheckForceUnknown(const ge::Node &node, u32 &taskNum) {
  /* 对动态shap下的集合通信known算子内存进行检查，如果大于ccl
   * buf，就把算子修改为unkown。GE会按照单算子模式调用 */
  uint64_t memSize = 0;
  if (node.GetOpDesc()->GetType() == HCCL_KERNEL_OP_TYPE_ALLGATHER ||
      node.GetOpDesc()->GetType() == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    CHK_RET(HcomOpUtils::GetAllOutputsTensorMemSize(node.GetOpDesc(), memSize));
  } else {
    CHK_RET(HcomOpUtils::GetAllInputsTensorMemSize(node.GetOpDesc(), memSize));
  }

  // 获取cclbuffer size
  u64 cclBuffSize;
  CHK_RET(GetCCLBufferAvailableSize(cclBuffSize));

  // 当用户内存大于ccl buff，会loop多次
  u32 loopTimes = (memSize + cclBuffSize - 1) / cclBuffSize;

  taskNum *= loopTimes;

  std::string sGroup;
  CHK_RET(HcomOpUtils::GetGroupFromOpDesc(node.GetOpDesc(), sGroup));
  u32 taskMaxNum = TASK_MAX_NUM_DEV_TYPE_V80;
  if (IsSocVersion910B(sGroup)) {
    taskMaxNum = TASK_MAX_NUM_DEV_TYPE_V71;
  }

  bool bRet = false;
  if (taskNum >= taskMaxNum) {
    HCCL_WARNING("[HcomGraphOptimizer][CheckForceUnknown] taskNum >= taskMaxNum set opbase mode");
    bRet = ge::AttrUtils::SetBool(node.GetOpDesc(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][UnknownShapeAttr]SetBool unknown shape failed, op type[%s]",
                           node.GetOpDesc()->GetType().c_str()),
                HCCL_E_PARA);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomGraphOptimizer::GetOriginalGraphShapeTypeFromDesc(const ge::OpDescPtr &op, u32 &shapeType) {
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

HcclResult HcomGraphOptimizer::SetHcomOpParam(const ge::Node &node, HcomOpParam *hcomOpParam,
                                              std::string &sCollectiveType, std::string &sGroup,
                                              std::string &socVersion, std::vector<int64_t> &sendCountMatrix,
                                              std::vector<int64_t> &sendCounts, std::vector<int64_t> &sendDispls,
                                              std::vector<int64_t> &recvCounts, std::vector<int64_t> &recvDispls,
                                              std::vector<u32> &curRanks, std::string &rankTableStr,
                                              std::string &rankTableM) {
  HcclResult ret;
  sCollectiveType = node.GetOpDesc()->GetType();
  ret = CheckSupportedOP(sCollectiveType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op type[%s] is not supported.", ret, sCollectiveType.c_str()),
      HCCL_E_NOT_SUPPORT);
  hcomOpParam->opType = const_cast<char *>(sCollectiveType.c_str());

  HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
  ret = HcomOpUtils::ConversionOpDataType(node.GetOpDesc(), sCollectiveType, dataType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: get data type failed. ret[%d]", sCollectiveType.c_str(), ret), ret);
  hcomOpParam->dataType = dataType;

  CHK_PRT_RET(ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS,
              HCCL_ERROR("[offline][compilation] get soc version failed."), HCCL_E_INTERNAL);
  hcomOpParam->socVersion = const_cast<char *>(socVersion.c_str());

  int64_t hcomComm = 0;
  ret = GetCommFromOpDesc(node.GetOpDesc(), hcomComm, sGroup);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: GetGroupFromOpDesc failed. ret[%d]", sCollectiveType.c_str(), ret),
      ret);
  if (hcomComm != static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &(hcomOpParam->group)));
  } else {
    hcomOpParam->group = const_cast<char *>(sGroup.c_str());
  }

  u32 rankSize = 0;
  if (!IsOfflineCompilation()) {
    if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomGetRankSize(sGroup.c_str(), &rankSize));
    } else {
      char *group = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
      CHK_RET(HcomGetRankSize(group, &rankSize));
    }
  } else {
    // 离线编译ranksize在HcomCalcOpResOffline中计算
  }
  if ((sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) || (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER)) {
    CHK_PRT_RET((!ge::AttrUtils::GetInt(node.GetOpDesc(), HCOM_ATTR_RANK_SIZE, rankSize)),
                HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s] get  attr[%s] failed.", sCollectiveType.c_str(),
                           HCOM_ATTR_RANK_SIZE.c_str()),
                HCCL_E_PARA);
    CHK_PRT_RET((rankSize <= 0),
                HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: rank_size[%d] should be "
                           "greater than 0.",
                           sCollectiveType.c_str(), rankSize),
                HCCL_E_PARA);
  }
  hcomOpParam->rankSize = rankSize;

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER ||
      sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
    HcclReduceOp reduction = HcclReduceOp::HCCL_REDUCE_SUM;
    CHK_RET(HcomOpUtils::GetReduction(node.GetOpDesc(), reduction));
    hcomOpParam->reduceOp = reduction;

    u8 deterministic = DETERMINISTIC_DISABLE;
    CHK_RET(GetDeterministic(deterministic));
    hcomOpParam->geDeterministic = deterministic;
  }

  u64 count = 0;
  ret = HcomGetAccuracyCountFromOpDesc(node.GetOpDesc(), sCollectiveType, dataType, count, rankSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: get count failed. ret[%d]", sCollectiveType.c_str(), ret),
              ret);
  hcomOpParam->count = count;

  // 获取aivCoreLimit，提供给HCCL，用于和Optype，count等参数一起选择具体的算法及判断是否是AIV模式
  uint32_t aivCoreLimit;
  CHK_RET(HcomOpUtils::GetAivCoreLimit(node.GetOpDesc(), sCollectiveType, aivCoreLimit));
  hcomOpParam->aivCoreLimit = aivCoreLimit;

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
    // reducescatterv复用HcomOpParam的All2AllDataDes字段
    CHK_RET(
        HcomOpUtils::GetReduceScatterVCountsDispl(const_cast<ge::Node &>(node), sendCounts, sendDispls, recvCounts));
    hcomOpParam->All2AllDataDes.sendCounts = static_cast<void *>(sendCounts.data());
    hcomOpParam->All2AllDataDes.sendDispls = static_cast<void *>(sendDispls.data());
    hcomOpParam->All2AllDataDes.recvCounts = static_cast<void *>(recvCounts.data());
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHERV) {
    // allgatherv复用HcomOpParam的All2AllDataDes字段
    CHK_RET(HcomOpUtils::GetAllGatherVCountsDispl(const_cast<ge::Node &>(node), sendCounts, recvCounts, recvDispls));
    hcomOpParam->All2AllDataDes.sendCounts = static_cast<void *>(sendCounts.data());
    hcomOpParam->All2AllDataDes.recvCounts = static_cast<void *>(recvCounts.data());
    hcomOpParam->All2AllDataDes.recvDispls = static_cast<void *>(recvDispls.data());
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV) {
    HcclDataType sendType;
    HcclDataType recvType;
    CHK_RET(HcomOpUtils::GetAlltoAllDataType(node.GetOpDesc(), sendType, recvType));

    auto op = node.GetOpDesc();
    if (ge::AttrUtils::HasAttr(op, "send_counts")) {
      CHK_RET(HcomOpUtils::GetAlltoAllCountsDispl(op, sendCounts, sendDispls, recvCounts, recvDispls));
    } else {
      CHK_RET(HcomOpUtils::GetAlltoAllCountsDispl(const_cast<ge::Node &>(node), sendCounts, sendDispls, recvCounts,
                                                  recvDispls));
    }

    if (sendCounts.size() < rankSize) {
      HCCL_ERROR("[sendCounts] size[%u] is invalid, expect size: %llu", sendCounts.size(), rankSize);
      return HCCL_E_PARA;
    }

    hcomOpParam->All2AllDataDes.sendType = sendType;
    hcomOpParam->All2AllDataDes.recvType = recvType;
    hcomOpParam->All2AllDataDes.sendCounts = static_cast<void *>(sendCounts.data());
    hcomOpParam->All2AllDataDes.sendDispls = static_cast<void *>(sendDispls.data());
    hcomOpParam->All2AllDataDes.recvCounts = static_cast<void *>(recvCounts.data());
    hcomOpParam->All2AllDataDes.recvDispls = static_cast<void *>(recvDispls.data());
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
    HcclDataType sendType;
    HcclDataType recvType;
    CHK_RET(HcomOpUtils::GetAlltoAllDataType(node.GetOpDesc(), sendType, recvType));

    if (!IsOfflineCompilation()) {
      CHK_RET(HcomOpUtils::CheckAlltoAllvcRank(node, hcomComm, sGroup));
    }

    auto op = node.GetOpDesc();
    if (ge::AttrUtils::HasAttr(op, "send_count_matrix")) {
      CHK_RET(HcomOpUtils::GetAlltoAllCountMatrix(op, sendCountMatrix));
    } else {
      CHK_RET(HcomOpUtils::GetAlltoAllCountMatrix(const_cast<ge::Node &>(node), sendCountMatrix));
    }
    if (sendCountMatrix.size() < rankSize * rankSize) {
      HCCL_ERROR("[sendCountMatrix] size[%u] is invalid, expect size: %llu", sendCountMatrix.size(),
                 rankSize * rankSize);
      return HCCL_E_PARA;
    }

    hcomOpParam->All2AllDataDes.sendType = sendType;
    hcomOpParam->All2AllDataDes.recvType = recvType;
    hcomOpParam->All2AllDataDes.sendCountMatrix = static_cast<void *>(sendCountMatrix.data());
  }

  std::string groupListString;
  if (ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, groupListString) == ge::GRAPH_SUCCESS) {
    try {
      nlohmann::json groupListConf;
      CHK_RET(SalParseInformation(groupListConf, groupListString));
      std::vector<nlohmann::json> groupList = groupListConf.get<std::vector<nlohmann::json>>();
      for (auto &groupInfo : groupList) {
        HCCL_DEBUG("groupInfo:%s", groupInfo.dump().c_str());
        std::string curGroupName = groupInfo["group_name"];
        HCCL_DEBUG("curGroupName:%s", curGroupName.c_str());
        if (curGroupName == sGroup) {
          curRanks = groupInfo["group_rank_list"].get<std::vector<u32>>();
          break;
        }
      }
    } catch (const std::exception &e) {
      HCCL_ERROR("[HcomCalcOpRunningParam] exception caught. err[%s]", e.what());
      return HCCL_E_INTERNAL;
    }
    hcomOpParam->groupList = static_cast<u32 *>(curRanks.data());
    hcomOpParam->groupListSize = curRanks.size();
  } else {
    HCCL_INFO("get groupListString failed");
  }

  if ((ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_TABLE, rankTableStr) == ge::GRAPH_SUCCESS) &&
      !rankTableStr.empty()) {
    hcomOpParam->rankTable = const_cast<char *>(rankTableStr.c_str());
  } else {
    HCCL_INFO("get rankTableStr failed");
  }

  std::string rankTablePath;
  if ((ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_TABLE_FILE, rankTablePath) == ge::GRAPH_SUCCESS) &&
      !rankTablePath.empty()) {
    ret = HcomLoadRanktableFile(rankTablePath, rankTableM);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[GetRanktable] rankTablePath[%s]"
                           "load rankTable error.",
                           rankTablePath.c_str()),
                HCCL_E_INTERNAL);
    hcomOpParam->rankTable = const_cast<char *>(rankTableM.c_str());
  } else {
    HCCL_INFO("get rankTablePath failed");
  }

  HCCL_INFO(
      "[SetHcomOpParam]HcomOpParam opType:[%s] dataType:[%s] count:[%llu] group:[%s] reduceOp:[%s] "
      "deterministic:[%d] socVersion:[%s] groupList:[%p] groupListSize:[%llu] ranktable:[%p]",
      hcomOpParam->opType, GetDataTypeEnumStr(hcomOpParam->dataType).c_str(), hcomOpParam->count, hcomOpParam->group,
      GetReduceOpEnumStr(hcomOpParam->reduceOp).c_str(), hcomOpParam->geDeterministic, hcomOpParam->socVersion,
      hcomOpParam->groupList, hcomOpParam->groupListSize, hcomOpParam->rankTable);
  return HCCL_SUCCESS;
}
}  // namespace hccl
