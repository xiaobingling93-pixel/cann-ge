/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_tuning_hcom_all_reduce_fusion.h"
#include "auto_tuning_hcom_ops_kernel_info_store.h"
#include "hcom_op_utils.h"
#include "hccl/hcom.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "mmpa/mmpa_api.h"
using namespace hccl;
using namespace std;

namespace hccl {
AutoTuningHcomAllReduceFusion::AutoTuningHcomAllReduceFusion() : isNotFoundHash_(false) {}

AutoTuningHcomAllReduceFusion::~AutoTuningHcomAllReduceFusion() {}

HcclResult AutoTuningHcomAllReduceFusion::Run(ge::ComputeGraph &graph, std::vector<GradientDataInfo> &recordInfos) {
  HcclResult ret = SetGradientInformation(graph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][AllReduceFusion]graph[%s]: GradientInformation Get failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);
  ret = RecordGradientDataInfo(graph, recordInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][AllReduceFusion]graph[%s]: RecordGradientDataInfo failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);
  FusionInfos fusionInfos;
  ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][AllReduceFusion]graph[%s]: get fusion HcomAllReduce ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  HCCL_INFO("there are %zu group to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of HcomAllReduce operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0), HCCL_INFO("NOT_CHANGED: the graph has no HcomAllReduce op."), HCCL_SUCCESS);

  for (auto iterFusionInfo = fusionInfos.begin(); iterFusionInfo != fusionInfos.end(); iterFusionInfo++) {
    HCCL_INFO("graph[%s] fusionlabel[%s]: there are %zu HcomAllreduce ops to be fused.", graph.GetName().c_str(),
              iterFusionInfo->first.c_str(), iterFusionInfo->second.size());
    ret = FuseOps(graph, iterFusionInfo->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] fusionlabel[%s]: fusion HcomAllreduce ops failed. "
                           "ret[%d]",
                           graph.GetName().c_str(), iterFusionInfo->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("fuse HcomAllReduce op end");

  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::RecordGradientDataInfo(ge::ComputeGraph &graph,
                                                                 std::vector<GradientDataInfo> &recordInfos) {
  HcclResult ret;
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("AutoTuningHcomAllReduceFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("AutoTuningHcomAllReduceFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }
    if (AutoTuneTargetNode(nodePtr)) {
      HCCL_INFO("record gradient info: strAllReduceNodeName[%s]...", nodePtr->GetName().c_str());
      if (opDescPtr->GetAllInputsDesc().size() != 1) {
        HCCL_ERROR("[Record][GradientDataInfo]AutoTuningHcomAllReduceFusion: node[%s] has %zu inputs.",
                   nodePtr->GetName().c_str(), opDescPtr->GetAllInputsDesc().size());
        return HCCL_E_INTERNAL;
      }
      GradientDataInfo gradientNodeInfo;
      ret = GetGradientDataInfo(graph, nodePtr, gradientNodeInfo);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Record][GradientDataInfo]GetGradientDataInfo failed, node[%s], ret[%u] ",
                             nodePtr->GetOpDesc()->GetName().c_str(), ret),
                  ret);

      ret = AddTraceNode(graph, nodePtr, gradientNodeInfo);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Record][GradientDataInfo]AddTraceNode failed, node[%s], ret[%u] ",
                             nodePtr->GetOpDesc()->GetName().c_str(), ret),
                  ret);
      recordInfos.push_back(gradientNodeInfo);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::AddTraceNode(ge::ComputeGraph &graph, ge::NodePtr &nodePtr,
                                                       const GradientDataInfo gradientNodeInfo) {
  auto gradientNodeOutDataAnchor = nodePtr->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto gradientNodePtr = gradientNodeOutDataAnchor->GetOwnerNode();
  ge::OpDescPtr traceNodeOpDescPtr = nullptr;
  EXECEPTION_CATCH(
      (traceNodeOpDescPtr = std::make_shared<ge::OpDesc>(gradientNodeInfo.traceNodeName, "TensorRedirect")),
      return HCCL_E_INTERNAL);
  ge::graphStatus geRet = traceNodeOpDescPtr->AddInputDesc(
      "x", gradientNodePtr->GetOpDesc()->GetOutputDesc(gradientNodeOutDataAnchor->GetIdx()));
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Add][TraceNode]Failed to add TensorRedirect input, node:%s .",
                         gradientNodeInfo.traceNodeName.c_str()),
              HCCL_E_INTERNAL);
  geRet = traceNodeOpDescPtr->AddOutputDesc(
      "output_x", gradientNodePtr->GetOpDesc()->GetOutputDesc(gradientNodeOutDataAnchor->GetIdx()));
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Add][TraceNode]Failed to add TensorRedirect output, node:%s .",
                         gradientNodeInfo.traceNodeName.c_str()),
              HCCL_E_INTERNAL);
  auto traceNodePtr = graph.AddNode(traceNodeOpDescPtr);
  CHK_PRT_RET(!traceNodePtr, HCCL_ERROR("[Add][TraceNode]create TensorRedirect node failed"), HCCL_E_INTERNAL);
  auto traceNodeInDataAnchor = traceNodePtr->GetInDataAnchor(0);
  CHK_PRT_RET(!traceNodeInDataAnchor, HCCL_ERROR("[Add][TraceNode]get TensorRedirect node in data anchor failed"),
              HCCL_E_INTERNAL);
  geRet = gradientNodeOutDataAnchor->LinkTo(traceNodeInDataAnchor);
  bool bErr = (geRet != ge::GRAPH_SUCCESS);
  CHK_PRT_RET(bErr,
              HCCL_ERROR("[Add][TraceNode]node[%s] link to node[%s] failed.", gradientNodeInfo.gradientNodeName.c_str(),
                         gradientNodeInfo.traceNodeName.c_str()),
              HCCL_E_INTERNAL);
  geRet = ge::GraphUtils::AddEdge(traceNodePtr->GetOutControlAnchor(), nodePtr->GetInControlAnchor());
  bErr = (geRet != ge::GRAPH_SUCCESS);
  CHK_PRT_RET(bErr,
              HCCL_ERROR("[Add][TraceNode]node[%s] add control dependence edge failed.",
                         gradientNodeInfo.allReduceNodeName.c_str()),
              HCCL_E_INTERNAL);

  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::GetGradientDataInfo([[maybe_unused]] ge::ComputeGraph &graph,
                                                              ge::NodePtr &nodePtr,
                                                              GradientDataInfo &gradientNodeInfo) {
  auto inputTensor = nodePtr->GetOpDesc()->GetInputDesc(0);
  auto shape = inputTensor.GetShape();
  auto format = inputTensor.GetFormat();
  auto dataType = inputTensor.GetDataType();
  bool bErr =
      (ge::GRAPH_SUCCESS != ge::TensorUtils::CalcTensorMemSize(shape, format, dataType, gradientNodeInfo.dataSize));
  CHK_PRT_RET(bErr,
              HCCL_ERROR("[Get][GradientDataInfo]In GetGradientDataInfo, CalcTensorMemSize failed,"
                         "Format[%d], dataType[%d], Size[%lld]",
                         format, dataType, gradientNodeInfo.dataSize),
              HCCL_E_PARA);
  HCCL_INFO("get data size[%lld] success.", gradientNodeInfo.dataSize);
  CHK_RET(GetDataTypeName(dataType, gradientNodeInfo.dataType));

  gradientNodeInfo.graphId = ge::GraphUtils::FindRootGraph(nodePtr->GetOwnerComputeGraph())->GetGraphID();
  CHK_RET(GetGroupName(nodePtr->GetOpDesc(), gradientNodeInfo.groupName));

  auto gradientNodeOutDataAnchor = nodePtr->GetInDataAnchor(0)->GetPeerOutAnchor();
  CHK_PRT_RET(
      (gradientNodeOutDataAnchor == nullptr),
      HCCL_ERROR("[Get][GradientDataInfo]Failed to get node[%s]'s peer out data anchor", nodePtr->GetName().c_str()),
      HCCL_E_INTERNAL);
  auto gradientNodePtr = gradientNodeOutDataAnchor->GetOwnerNode();
  CHK_PRT_RET(
      (gradientNodePtr == nullptr),
      HCCL_ERROR("[Get][GradientDataInfo]Failed to get node[%s]'s peer out data node", nodePtr->GetName().c_str()),
      HCCL_E_INTERNAL);
  gradientNodeInfo.gradientNodeName = gradientNodePtr->GetOpDesc()->GetName();
  gradientNodeInfo.allReduceNodeName = nodePtr->GetName();
  gradientNodeInfo.traceNodeName = gradientNodeInfo.allReduceNodeName + "_TensorRedirect";

  HCCL_INFO(
      "record gradient info: AllReduceNodeName[%s] traceNodeName[%s] GradientNodeName[%s] "
      "groupName[%s] dataType[%s] dataSize[%lld]",
      gradientNodeInfo.allReduceNodeName.c_str(), gradientNodeInfo.traceNodeName.c_str(),
      gradientNodeInfo.gradientNodeName.c_str(), gradientNodeInfo.groupName.c_str(), gradientNodeInfo.dataType.c_str(),
      gradientNodeInfo.dataSize);

  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::GetGroupName(const ge::OpDescPtr &op, std::string &group) {
  if (ge::AttrUtils::HasAttr(op, "group")) {
    if (ge::AttrUtils::GetStr(op, "group", group) == false) {
      HCCL_ERROR("[Get][GroupName]errNo[0x%016llx]: get group failed. get \"group\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
    CHK_PRT_RET(group.empty(),
                HCCL_ERROR("[Get][GroupName]errNo[0x%016llx] get group name failed. group from"
                           "opDesc is empty.",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);
  } else {
    group = HCCL_WORLD_GROUP;
  }
  HCCL_INFO("get group name[%s] success.", group.c_str());
  return HCCL_SUCCESS;
}

bool AutoTuningHcomAllReduceFusion::AutoTuneTargetNode(ge::NodePtr nodePtr) {
  auto opDescPtr = nodePtr->GetOpDesc();
  if (opDescPtr->GetType() != HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
    return false;
  }

  bool bUnknownShapeNode = false;
  if (ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, bUnknownShapeNode) != ge::GRAPH_SUCCESS) {
    return false;
  }
  if (bUnknownShapeNode) {
    return false;
  }
  if (ge::AttrUtils::HasAttr(opDescPtr, ge::ATTR_NAME_IS_UNKNOWN_SHAPE)) {
    if (!ge::AttrUtils::GetBool(opDescPtr, ge::ATTR_NAME_IS_UNKNOWN_SHAPE, bUnknownShapeNode)) {
      return false;
    }
    if (bUnknownShapeNode) {
      return false;
    }
  }

  std::string reduction;
  if (!ge::AttrUtils::GetStr(opDescPtr, ge::HCOM_ATTR_REDUCE_TYPE, reduction)) {
    return false;
  }
  if (reduction != "sum") {
    return false;
  }

  int64_t fusionAttr;
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION, fusionAttr)) {
    return false;
  }
  if (fusionAttr == HCOM_ATTR_FUSION_BY_SPLIT_STRATEGY) {
    return true;
  } else {
    return false;
  }
}

HcclResult AutoTuningHcomAllReduceFusion::GetDataTypeName(const ge::DataType dataType, std::string &dataTypeName) {
  std::map<ge::DataType, std::string> CONST_OP_HCOM_DATA_TYPE_MAP = {
      {ge::DT_FLOAT, "float32"}, {ge::DT_FLOAT16, "float16"}, {ge::DT_INT8, "int8"},
      {ge::DT_INT16, "int16"},   {ge::DT_INT32, "int32"},     {ge::DT_BF16, "bfloat16"},
  };

  auto iter = CONST_OP_HCOM_DATA_TYPE_MAP.find(dataType);
  CHK_PRT_RET((iter == CONST_OP_HCOM_DATA_TYPE_MAP.end()),
              HCCL_ERROR("[Get][DataTypeName]data type[%lld] is not supported, must be one of the following types: "
                         "int8, int16, int32, float16, float32.",
                         dataType),
              HCCL_E_PARA);
  dataTypeName = iter->second;
  HCCL_INFO("get data type[%s] success.", dataTypeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::GetGradSplitStrategy([[maybe_unused]] const std::string &modelName,
                                                               [[maybe_unused]] const std::string &sGroup,
                                                               const std::vector<ge::NodePtr> &fusionOps,
                                                               u32 &segmentNum, std::vector<u32> &segmentIndex) {
  segmentNum = 1;
  segmentIndex.clear();
  segmentIndex.push_back(fusionOps.size() - 1);
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  HcclResult ret;
  CHK_PRT_RET((fusionSection.size() <= 1), HCCL_INFO("the section has %u HcomAllReduce op.", fusionSection.size()),
              HCCL_SUCCESS);

  u32 segmentNum = 0;
  std::vector<u32> segmentIndex(HCCL_MAX_SEGMENT_NUM, 0);
  ret = GetFusionStrategy(graph, fusionSection, segmentNum, segmentIndex);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("graph[%s]  get HcomAllReduce ops fusion strategy failed. ret[%d]", graph.GetName().c_str(), ret),
      ret);

  ret = RunFusionOps(graph, fusionSection, segmentNum, segmentIndex);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret),
              ret);
  HCCL_INFO("graph[%s] fuse HcomAllReduce op end.", graph.GetName().c_str());
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::SetGradientInformation(ge::ComputeGraph &graph) {
  HcclResult ret;
  bool hashAllReduce = false;
  std::string fusionHash;
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("AutoTuningHcomAllReduceFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("AutoTuningHcomAllReduceFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }
    if (AutoTuneTargetNode(nodePtr)) {
      HCCL_INFO("AutoTuningHcomAllReduceFusion: AllReduce exists.");
      hashAllReduce = true;
      break;
    }
  }
  if (hashAllReduce) {
    HCCL_INFO("AutoTuningHcomAllReduceFusion: AllReduce exists.");
    ret = GetFusionhashFromGraph(graph, fusionHash);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] get fusionhash failed, ret[%d]", graph.GetName().c_str(), ret), ret);

    // 写入知识库
    ret = AddFusionMapInFusionJson(fusionHash);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] add Fusion Information in Lib failed, ret[%d]", graph.GetName().c_str(), ret),
                ret);
  } else {
    HCCL_INFO("AutoTuningHcomAllReduceFusion: Zero AllReduce exists.");
  }

  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::AddFusionMapInFusionJson(const std::string &fusionHash) {
  std::string workPath;
  std::string fileName;
  std::string socVerison;
  std::vector<uint64_t> defaultvalue{0, 0, 0, 0, 0};
  HCCL_INFO("[AddFusionInfo]Start add fusion value in library.");

  CHK_RET(GetFusionWorkPath(workPath));

  if (access(workPath.c_str(), F_OK) == -1) {
    HCCL_INFO("path is not exixts.");
    fstream file;
    file.open(workPath, ios::out | ios::app);
    nlohmann::json root;
    nlohmann::json model;
    model["modelhash"] = fusionHash;
    nlohmann::json value;
    value["gradientTime"] = {0, 0, 0, 0, 0};
    value["gradientSize"] = {0, 0, 0, 0, 0};
    model["modelvalue"] = value;
    root.push_back(model);
    file << root;
    file.close();
  } else {
    fstream file;
    nlohmann::json root;
    file.open(workPath, std::ios::in);
    bool emptyFile = false;
    if (file.peek() == std::ifstream::traits_type::eof()) {
      emptyFile = true;
    }
    file.close();
    if (emptyFile) {
      file.open(workPath, std::ios::out);
      nlohmann::json model;
      model["modelhash"] = fusionHash;
      nlohmann::json value;
      value["gradientTime"] = defaultvalue;
      value["gradientSize"] = defaultvalue;
      model["modelvalue"] = value;
      root.push_back(model);
      file << root;
      file.close();
      HCCL_INFO("[AddFusionInfo]File is null, add New model value in Fusion Library.");
    } else {
      CHK_RET(SetFusionModelInLibrary(workPath, fusionHash));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::SetFusionModelInLibrary(std::string &workPath,
                                                                  const std::string &fusionHash) {
  std::vector<uint64_t> defaultvalue{0, 0, 0, 0, 0};
  fstream file;
  nlohmann::json root;
  file.open(workPath, std::ios::in);
  file >> root;
  file.close();
  int32_t rootSize = root.size();
  for (auto i = 0; i < rootSize; i++) {
    if (root[i]["modelhash"] == fusionHash) {
      HCCL_INFO("[AddFusionInfo]The hash value already exists in the Fusion Library.");
      isNotFoundHash_ = false;
      break;
    } else if (root[i]["modelvalue"]["gradientTime"] == defaultvalue) {
      root[i]["modelhash"] = fusionHash;
      isNotFoundHash_ = false;
      break;
    } else {
      isNotFoundHash_ = true;
    }
  }
  HCCL_INFO("Add model value in fusion library.");
  if (isNotFoundHash_) {
    nlohmann::json model;
    model["modelhash"] = fusionHash;
    nlohmann::json value;
    value["gradientTime"] = defaultvalue;
    value["gradientSize"] = defaultvalue;
    model["modelvalue"] = value;
    root.push_back(model);
    HCCL_INFO("[AddFusionInfo]ModelHash out of exist, add new fusion value.");
  }
  file.open(workPath, std::ios::out | std::ios::trunc);
  file << root;
  file.close();
  isNotFoundHash_ = false;
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::GetFusionWorkPath(std::string &fusionPath) {
  HcclResult ret;
  char realFile[PATH_MAX] = {0};
  std::string ConfigVersion;
  std::string fileName;
  ret = HcomOpUtils::CreateFusionConfigVersion(ConfigVersion);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Atuo][Tune]GetSocVersion failed."), ret);
  fileName = ConfigVersion + "_gradient_fusion.json";

  std::string libPath;
  char *getEnvPath = nullptr;
  MM_SYS_GET_ENV(MM_ENV_TUNE_BANK_PATH, getEnvPath);
  if (getEnvPath != nullptr) {
    if (realpath(getEnvPath, realFile) == nullptr) {
      HCCL_ERROR(
          "[autotune][fusionPath]errNo[0x%016llx] path [%s] from env:TUNE_BANK_PATH is not a \
                valid real path",
          HCOM_ERROR_CODE(HCCL_E_PARA), getEnvPath);
      return HCCL_E_PARA;
    }
    libPath = getEnvPath;
    fusionPath = libPath + "/" + fileName;
  } else {
    HCCL_WARNING("ENV:TUNE_BANK_PATH is not set, use Default Path.");
    char *getDefPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HOME, getDefPath);
    if (getDefPath != nullptr) {
      libPath = getDefPath;
      libPath = libPath + "/Ascend/latest/data/aoe/custom/graph/" + ConfigVersion + "/";
      CHK_RET(CreateFusionDir(libPath));
      if (realpath(getDefPath, realFile) == nullptr) {
        HCCL_ERROR(
            "[autotune][fusionPath]errNo[0x%016llx] path [%s] from env:HOME is not a \
                    valid real path",
            HCOM_ERROR_CODE(HCCL_E_PARA), getDefPath);
        return HCCL_E_PARA;
      }
      libPath = libPath + "/" + fileName;
      fusionPath = libPath;
    } else {
      HCCL_ERROR("[auto][tune]find fusion library path failed.");
      return HCCL_E_NOT_FOUND;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomAllReduceFusion::CreateFusionDir(std::string &dir) {
  uint32_t beginCmpPath = 0;
  uint32_t endCmpPath = 0;

  std::string fullPath = "";
  if ('/' != dir[0]) {
    fullPath = getcwd(nullptr, 0);
    beginCmpPath = fullPath.size();
    fullPath = fullPath + "/" + dir;
  } else {
    fullPath = dir;
    beginCmpPath = static_cast<uint32_t>(CreateDir::HCCL_DIR_NUM_ONE);
  }

  if (fullPath[fullPath.size() - 1] != '/') {
    fullPath += "/";
  }
  endCmpPath = fullPath.size();
  for (uint32_t i = beginCmpPath; i < endCmpPath; i++) {
    if ('/' == fullPath[i]) {
      std::string curPath = fullPath.substr(0, i);
      if (access(curPath.c_str(), F_OK) != 0) {
        mkdir(curPath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
      }
    }
  }
  HCCL_INFO("[auto][tune]Create Fusion Dir success.");
  return HCCL_SUCCESS;
}
}  // namespace hccl
