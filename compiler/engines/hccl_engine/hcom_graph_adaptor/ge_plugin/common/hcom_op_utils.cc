/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_op_utils.h"
#include "hccl/hcom.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "framework/memory/memory_api.h"
#include "ge/ge_api_types.h"            // ge对内options
#include "framework/common/ge_types.h"  // ge对外options
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "mmpa/mmpa_api.h"
#include "offline_build_config_parse.h"
#include "graph/ge_context.h"

namespace hccl {
const std::string HCOM_ATTR_ROOT = "root_rank";
const std::string HCOM_ATTR_DTYPE = "dtype";
const std::string HCOM_ATTR_GROUP = "group";
const std::string HCOM_ATTR_REDUCTION = "reduction";

const std::string HCOM_ATTR_DEST_RANK = "dest_rank";
const std::string HCOM_ATTR_SR_TAG = "sr_tag";
const std::string HCOM_ATTR_SRC_RANK = "src_rank";

HcclResult HcomOpUtils::GetReduction(const ge::OpDescPtr &opDescPtr, HcclReduceOp &reduction) {
  std::string sReduction;
  CHK_PRT_RET((!ge::AttrUtils::GetStr(opDescPtr, HCOM_ATTR_REDUCTION, sReduction)),
              HCCL_ERROR("[Get][Reduction] get attr[%s] failed.", HCOM_ATTR_REDUCTION.c_str()), HCCL_E_PARA);

  auto iter = HCOM_REDUCE_TYPE_MAP.find(sReduction);
  CHK_PRT_RET((iter == HCOM_REDUCE_TYPE_MAP.end()),
              HCCL_ERROR("[Get][Reduction] reduction[%s] is not supported, must be one of the"
                         "following types: sum, prod, max, min.",
                         sReduction.c_str()),
              HCCL_E_PARA);
  reduction = iter->second;
  HCCL_DEBUG("get reduction[%d] success.", reduction);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetRoot(const ge::OpDescPtr &opDescPtr, int32_t &root) {
  CHK_PRT_RET((!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_ROOT, root)),
              HCCL_ERROR("[Get][Root] get attr[%s] failed.", HCOM_ATTR_ROOT.c_str()), HCCL_E_PARA);

  CHK_PRT_RET((root < 0), HCCL_ERROR("[Get][Root] root[%d] should be not less than 0.", root), HCCL_E_PARA);
  HCCL_DEBUG("get root[%d] success.", root);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetRankSize(const ge::OpDescPtr &opDescPtr, int32_t &rankSize) {
  CHK_PRT_RET((!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_RANK_SIZE, rankSize)),
              HCCL_ERROR("[Get][RankSize] get attr[%s] failed.", HCOM_ATTR_RANK_SIZE.c_str()), HCCL_E_PARA);

  CHK_PRT_RET((rankSize <= 0), HCCL_ERROR("[Get][RankSize] rankSize[%d] should be greater than 0.", rankSize),
              HCCL_E_PARA);
  HCCL_DEBUG("get rankSize[%d] success.", rankSize);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetGroup(const ge::OpDescPtr &opDescPtr, std::string &group) {
  if (ge::AttrUtils::HasAttr(opDescPtr, HCOM_ATTR_GROUP)) {
    if (ge::AttrUtils::GetStr(opDescPtr, HCOM_ATTR_GROUP, group) == false) {
      HCCL_ERROR("[Get][Group]errNo[0x%016llx]: get group failed. get attr \"%s\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA), HCOM_ATTR_GROUP.c_str());
      return HCCL_E_PARA;
    }
    CHK_PRT_RET(group.empty(),
                HCCL_ERROR("[Get][Group]errNo[0x%016llx] get group name failed. group"
                           "from opDesc is empty.",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);
  } else {
    group = HCCL_WORLD_GROUP;
  }
  HCCL_DEBUG("get group name[%s] success.", group.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetDataType(const ge::OpDescPtr &opDescPtr, HcclDataType &dataType) {
  // 优先顺序: 'dtype' atttr > input[0]
  ge::DataType geDataType = static_cast<ge::DataType>(0);
  if (ge::AttrUtils::HasAttr(opDescPtr, HCOM_ATTR_DTYPE)) {
    if (ge::AttrUtils::GetDataType(opDescPtr, HCOM_ATTR_DTYPE, geDataType) == false) {
      HCCL_ERROR("[Get][DataType]errNo[0x%016llx]: get data type failed. get \"dtype\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  } else {
    CHK_SMART_PTR_NULL(opDescPtr->GetInputDescPtr(0));
    // 指针指向的各个算子的DataType一致，获取第一个op的即可
    geDataType = opDescPtr->GetInputDescPtr(0)->GetDataType();
  }
  CHK_RET(TransformDataType(geDataType, dataType));
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetDataType(const ge::OpDescPtr &opDescPtr, std::string &dataType) {
  HcclDataType hcclDataType;
  HcclResult ret = GetDataType(opDescPtr, hcclDataType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][DataType]op[%s]: get data type failed.", opDescPtr->GetName().c_str()), HCCL_E_PARA);
  auto iter = HCOM_DATA_TYPE_STR_MAP.find(hcclDataType);
  CHK_PRT_RET((iter == HCOM_DATA_TYPE_STR_MAP.end()),
              HCCL_ERROR("[Get][Data]node[%s]: hccl data type[%s] transform failed.", opDescPtr->GetName().c_str(),
                         GetDataTypeEnumStr(hcclDataType).c_str()),
              HCCL_E_INTERNAL);
  dataType = iter->second;
  HCCL_DEBUG("get data type[%s] success.", dataType.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::TransformDataType(const ge::DataType geDataType, HcclDataType &hcclDataType) {
  auto iter = HCOM_DATA_TYPE_MAP.find(geDataType);
  CHK_PRT_RET((iter == HCOM_DATA_TYPE_MAP.end()),
              HCCL_ERROR("[Trans][DataType]errNo[0x%016llx] GeDataType[%lld] is not supported, must be one of the"
                         "following types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, "
                         "float64, bfloat16.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), geDataType),
              HCCL_E_PARA);
  hcclDataType = iter->second;
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAllInputsTensorMemSize(const ge::OpDescPtr &opDescPtr, uint64_t &tensorSize) {
  tensorSize = 0;
  for (uint32_t i = 0; i < opDescPtr->GetAllInputsSize(); i++) {
    auto inTensor = opDescPtr->GetInputDesc(i);
    uint64_t memSize = 0;
    HcclResult ret = GetTensorMemSize(inTensor, memSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][AllInputsTensorMemSize]node[%s] input[%u] GetTensorMemSize failed.",
                           opDescPtr->GetName().c_str(), i),
                ret);
    tensorSize += memSize;
    HCCL_DEBUG("node[%s] has %u inputs, input[%u] size %llu bytes.", opDescPtr->GetName().c_str(),
               opDescPtr->GetAllInputsSize(), i, tensorSize);
  }
  HCCL_DEBUG("node[%s] has %u inputs, total size %llu bytes.", opDescPtr->GetName().c_str(),
             opDescPtr->GetAllInputsSize(), tensorSize);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAllOutputsTensorMemSize(const ge::OpDescPtr &opDescPtr, uint64_t &tensorSize) {
  tensorSize = 0;
  for (uint32_t i = 0; i < opDescPtr->GetOutputsSize(); i++) {
    auto inTensor = opDescPtr->GetOutputDesc(i);
    uint64_t memSize = 0;
    HcclResult ret = GetTensorMemSize(inTensor, memSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][AllOutputsTensorMemSize]node[%s] output[%u] GetTensorMemSize failed.",
                           opDescPtr->GetName().c_str(), i),
                ret);
    tensorSize += memSize;
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetTensorMemSize(const ge::GeTensorDesc &tensorDesc, uint64_t &memSize) {
  const u32 memoryAlignRatio = 2;  // GE要求内存需要32B对齐后，再外加32B. out = (in + 2 * 32 - 1) / 32 * 32
  const u32 memoryAlignSize = 32;  // GE要求内存需要32B对齐后，再外加32B. out = (in + 2 * 32 - 1) / 32 * 32
  auto shape = tensorDesc.GetShape();
  auto format = tensorDesc.GetFormat();
  auto dataType = tensorDesc.GetDataType();
  int64_t size = 0;
  bool bErr = (ge::GRAPH_SUCCESS != ge::TensorUtils::CalcTensorMemSize(shape, format, dataType, size));
  CHK_PRT_RET((bErr) || (size < 0),
              HCCL_ERROR("[Get][TensorMemSize]In GetTensorMemSize, CalcTensorMemSize"
                         "failed, Format[%d], dataType[%d], size[%lld]",
                         format, dataType, size),
              HCCL_E_PARA);
  memSize = ((size + memoryAlignRatio * memoryAlignSize - 1) / memoryAlignSize) * memoryAlignSize;
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAllInputsTensorOriginSize(const ge::OpDescPtr &opDescPtr, uint64_t &tensorSize) {
  tensorSize = 0;
  for (uint32_t i = 0; i < opDescPtr->GetAllInputsSize(); i++) {
    auto inTensor = opDescPtr->GetInputDesc(i);
    uint64_t size = 0;
    HcclResult ret = GetTensorOriginSize(inTensor, size);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][AllInputsTensorOriginSize]node[%s] input[%u] GetTensorOriginSize failed.",
                           opDescPtr->GetName().c_str(), i),
                ret);
    HCCL_DEBUG("node[%s] has %u inputs, input[%u] size %llu bytes.", opDescPtr->GetName().c_str(),
               opDescPtr->GetAllInputsSize(), i, size);
    tensorSize += size;
  }
  HCCL_DEBUG("node[%s] has %u inputs, total size %llu bytes.", opDescPtr->GetName().c_str(),
             opDescPtr->GetAllInputsSize(), tensorSize);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetTensorOriginSize(const ge::GeTensorDesc &tensorDesc, uint64_t &size) {
  HcclDataType hcclDataType;
  auto geDataType = tensorDesc.GetDataType();
  CHK_RET(TransformDataType(geDataType, hcclDataType));
  uint32_t dataTypeSize;
  CHK_RET(SalGetDataTypeSize(hcclDataType, dataTypeSize));
  uint64_t shapeSize = static_cast<uint64_t>(tensorDesc.GetOriginShape().GetShapeSize());

  CHK_PRT_RET((shapeSize > INVALID_U64 / dataTypeSize),
              HCCL_ERROR("[Get][TensorOriginSize]shape size[%llu]"
                         "is overflow.",
                         shapeSize),
              HCCL_E_PARA);
  size = shapeSize * dataTypeSize;

  return HCCL_SUCCESS;
}
#ifndef HCOM_EXECUTOR
HcclResult HcomOpUtils::CreateFusionConfigVersion(std::string &configVersion) {
  uint32_t ret = 0;
  fe::PlatformInfoManager &configInst = fe::PlatformInfoManager::Instance();
  fe::PlatformInfo autoPlatInfo;
  fe::OptionalInfo autoOptinalInfo;
  ret = configInst.GetPlatformInfoWithOutSocVersion(autoPlatInfo, autoOptinalInfo);
  if (ret == 0) {
    configVersion = autoOptinalInfo.soc_version;
    HCCL_INFO("get soc version success, [%s], return [%u]", configVersion.c_str(), ret);
  } else {
    HCCL_ERROR("[Hcom][Optimizer]get soc version failed.");
  }
  return HCCL_SUCCESS;
}
#endif
HcclResult HcomOpUtils::GetPathFromEnv(char *getTunePath, std::string &fusionPath) {
  std::string libPath;
  if (getTunePath != nullptr) {
    libPath = getTunePath;
    fusionPath = libPath + "/";
    HCCL_INFO("Get Path from Env success.");
  } else {
    return HCCL_E_AGAIN;
  }
  return HCCL_SUCCESS;
}
#ifndef HCOM_EXECUTOR
HcclResult HcomOpUtils::GetFileNameFromPath(std::string &Path, std::string &fusionFile) {
  std::string fileName;
  std::string ConfigVersion;
  char getDefaultPath[PATH_MAX];
  if (realpath(Path.c_str(), getDefaultPath) == nullptr) {
    HCCL_WARNING("[Get][FusionConfigRealPath]path %s is not a valid path", Path.c_str());
    return HCCL_E_AGAIN;
  }

  CHK_RET(HcomOpUtils::CreateFusionConfigVersion(ConfigVersion));
  fileName = ConfigVersion + "_gradient_fusion.json";
  fusionFile = Path + fileName;
  return HCCL_SUCCESS;
}
#endif

bool HcomOpUtils::HcomOpIsSupportedBool(const std::string &opType) {
  const std::vector<std::string> hcomOpSupportBoolMap = {HCCL_KERNEL_OP_TYPE_BROADCAST, HCCL_KERNEL_OP_TYPE_ALLGATHER,
                                                         HCCL_KERNEL_OP_TYPE_SEND,      HCCL_KERNEL_OP_TYPE_RECEIVE,
                                                         HCCL_KERNEL_OP_TYPE_ALLTOALLV, HCCL_KERNEL_OP_TYPE_ALLTOALLVC,
                                                         HCCL_KERNEL_OP_TYPE_ALLTOALL};
  std::vector<std::string>::const_iterator it =
      std::find(hcomOpSupportBoolMap.begin(), hcomOpSupportBoolMap.end(), opType);
  return (it != hcomOpSupportBoolMap.end()) ? true : false;
}

HcclResult HcomOpUtils::ConversionOpDataType(const ge::OpDescPtr &op, const std::string &opType,
                                             HcclDataType &dataType) {
  // 获取GE数据类型
  ge::DataType geDataType = static_cast<ge::DataType>(0);
  if (ge::AttrUtils::HasAttr(op, HCOM_ATTR_DTYPE)) {
    if (ge::AttrUtils::GetDataType(op, HCOM_ATTR_DTYPE, geDataType) == false) {
      HCCL_ERROR("[Get][DataType]errNo[0x%016llx]: get data type failed. get \"dtype\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  } else {
    CHK_SMART_PTR_NULL(op->GetInputDescPtr(0));
    // 指针指向的各个算子的DataType一致，获取第一个op的即可
    geDataType = op->GetInputDescPtr(0)->GetDataType();
  }

  // 判断当前算子是否支持BOOL类型 且数据类型是否是BOOL,将GE的BOOL类型转换成HCCL的INT8
  if (HcomOpIsSupportedBool(opType) && (geDataType == ge::DT_BOOL)) {
    dataType = HCCL_DATA_TYPE_INT8;
  } else {
    auto iter = HCOM_DATA_TYPE_MAP.find(geDataType);
    CHK_PRT_RET((iter == HCOM_DATA_TYPE_MAP.end()),
                HCCL_ERROR("[Get][DataType]errNo[0x%016llx] node[%s]: data type[%lld] is not supported, must be "
                           "one of the following types: "
                           "int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float32, "
                           "float64.",
                           HCOM_ERROR_CODE(HCCL_E_PARA), op->GetName().c_str(), geDataType),
                HCCL_E_PARA);
    dataType = iter->second;
  }
  HCCL_INFO("[hcom][ConversionOpDataType]conversion opType[%s] geDataType[%lld] to data type[%s] success.",
            opType.c_str(), geDataType, GetDataTypeEnumStr(dataType).c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAlltoAllCountMatrix(const ge::OpDescPtr &op, std::vector<int64_t> &sendCountMatrix) {
  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, "send_count_matrix", sendCountMatrix)),
              HCCL_ERROR("[Set][AlltoAllVParams]op[%s] get attr[%s] failed.", HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(),
                         "send_count_matrix"),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAlltoAllCountMatrix(ge::Node &node, std::vector<int64_t> &sendCountMatrix) {
  // 1、判断是否是const 节点 NodeUtils::IsConst(*really_parent_node))
  // 是走老流程，不是判断是否是data节点，是去获取对应输入父节点，不是
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());
  auto sendCountMatrixTensor = ge::OpDescUtils::GetInputConstData(op, 1);
  if (sendCountMatrixTensor == nullptr) {
    CHK_RET(GetConstInputAcrossGraph(sendCountMatrixTensor, 1U, node));
  }

  CHK_RET(GetVectorFromTensor(sendCountMatrixTensor, sendCountMatrix));

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAlltoAllCountsDispl(const ge::OpDescPtr &op, std::vector<int64_t> &sendCounts,
                                               std::vector<int64_t> &sendDispls, std::vector<int64_t> &recvCounts,
                                               std::vector<int64_t> &recvDispls) {
  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, "send_counts", sendCounts)),
              HCCL_ERROR("[Set][AlltoAllVParams]op[%s] get attr[%s] failed.", HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(),
                         "send_counts"),
              HCCL_E_PARA);
  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, "send_displacements", sendDispls)),
              HCCL_ERROR("[Set][AlltoAllVParams]op[%s] get attr[%s] failed.", HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(),
                         "send_displacements"),
              HCCL_E_PARA);

  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, "recv_counts", recvCounts)),
              HCCL_ERROR("[Set][AlltoAllVParams]op[%s] get attr[%s] failed.", HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(),
                         "recv_counts"),
              HCCL_E_PARA);
  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, "recv_displacements", recvDispls)),
              HCCL_ERROR("[Set][AlltoAllVParams]op[%s] get attr[%s] failed.", HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(),
                         "recv_displacements"),
              HCCL_E_PARA);

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAlltoAllCountsDispl(ge::Node &node, std::vector<int64_t> &sendCounts,
                                               std::vector<int64_t> &sendDispls, std::vector<int64_t> &recvCounts,
                                               std::vector<int64_t> &recvDispls) {
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());

  std::vector<ge::ConstGeTensorPtr> alltoallvInputVec;
  const auto &op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::AttrUtils::GetListTensor(op_desc, "alltoallvInputVec", alltoallvInputVec);

  CHK_PRT_RET(alltoallvInputVec.size() != ALLTOALLV_INPUT_VEC_SIZE,
              HCCL_ERROR("Get AlltoallV input info from Operator invalid."), HCCL_E_PARA);

  auto sendCountsTensor = alltoallvInputVec[0U].get();
  auto sendDisplsTensor = alltoallvInputVec[1U].get();
  auto recvCountsTensor = alltoallvInputVec[2U].get();
  auto recvDisplsTensor = alltoallvInputVec[3U].get();

  CHK_RET(HcomOpUtils::GetVectorFromTensor(sendCountsTensor, sendCounts));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(sendDisplsTensor, sendDispls));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(recvCountsTensor, recvCounts));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(recvDisplsTensor, recvDispls));

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetReduceScatterVCountsDispl(ge::Node &node, std::vector<int64_t> &sendCounts,
                                                     std::vector<int64_t> &sendDispls,
                                                     std::vector<int64_t> &recvCount) {
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());

  std::vector<ge::ConstGeTensorPtr> vInputVec;
  const auto &op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::AttrUtils::GetListTensor(op_desc, "vInputVec", vInputVec);

  CHK_PRT_RET(vInputVec.size() != V_INPUT_VEC_SIZE, HCCL_ERROR("Get ReduceScatterV input info from Operator invalid."),
              HCCL_E_PARA);

  auto recvCountTensor = vInputVec[0U].get();
  auto sendCountsTensor = vInputVec[1U].get();
  auto sendDisplsTensor = vInputVec[2U].get();

  CHK_RET(HcomOpUtils::GetVectorFromTensor(recvCountTensor, recvCount));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(sendCountsTensor, sendCounts));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(sendDisplsTensor, sendDispls));

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAllGatherVCountsDispl(ge::Node &node, std::vector<int64_t> &sendCount,
                                                 std::vector<int64_t> &recvCounts, std::vector<int64_t> &recvDispls) {
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());

  std::vector<ge::ConstGeTensorPtr> vInputVec;
  const auto &op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::AttrUtils::GetListTensor(op_desc, "vInputVec", vInputVec);

  CHK_PRT_RET(vInputVec.size() != V_INPUT_VEC_SIZE, HCCL_ERROR("Get AllGatherV input info from Operator invalid."),
              HCCL_E_PARA);

  auto recvCountsTensor = vInputVec[0U].get();
  auto recvDisplsTensor = vInputVec[1U].get();
  auto sendCountTensor = vInputVec[2U].get();

  CHK_RET(HcomOpUtils::GetVectorFromTensor(sendCountTensor, sendCount));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(recvCountsTensor, recvCounts));
  CHK_RET(HcomOpUtils::GetVectorFromTensor(recvDisplsTensor, recvDispls));

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAlltoAllDataType(const ge::OpDescPtr &op, HcclDataType &sendType, HcclDataType &recvType) {
  auto opDescPtr = op->GetInputDescPtr(0);
  ge::DataType graphSendType = opDescPtr == nullptr ? ge::DT_INT8 : opDescPtr->GetDataType();  // 如果不发送则填int8类型
  auto iter = HCOM_DATA_TYPE_MAP.find(graphSendType);
  CHK_PRT_RET((iter == HCOM_DATA_TYPE_MAP.end()),
              HCCL_ERROR("[Get][DataType]errNo[0x%016llx] node[%s]: data type[%lld] is not supported, must be"
                         "one of the following types: "
                         "int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float32, "
                         "float64.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), op->GetName().c_str(), graphSendType),
              HCCL_E_PARA);
  sendType = iter->second;

  opDescPtr = op->GetOutputDescPtr(0);
  ge::DataType graphRecvType = opDescPtr == nullptr ? ge::DT_INT8 : opDescPtr->GetDataType();  // 如果不接收则填int8类型
  iter = HCOM_DATA_TYPE_MAP.find(graphRecvType);
  CHK_PRT_RET((iter == HCOM_DATA_TYPE_MAP.end()),
              HCCL_ERROR("[Get][DataType]errNo[0x%016llx] node[%s]: data type[%lld] is not supported, must be"
                         "one of the following types: "
                         "int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float32, "
                         "float64.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), op->GetName().c_str(), graphRecvType),
              HCCL_E_PARA);
  recvType = iter->second;

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetConstInputAcrossGraph(const ge::GeTensor *&tensor, u32 index, ge::Node &node) {
  ge::NodePtr really_parent_node = nullptr;
  if ((ge::NodeUtils::GetInNodeCrossPartionedCallNode(node.shared_from_this(), index, really_parent_node) ==
       ge::GRAPH_SUCCESS) &&
      (really_parent_node != nullptr) && (ge::NodeUtils::IsConst(*really_parent_node))) {
    std::vector<ge::GeTensorPtr> weight = ge::OpDescUtils::MutableWeights(really_parent_node);
    if (weight.size() != 0) {
      CHK_PTR_NULL(weight[0].get());
      tensor = weight[0].get();
    }
  } else {
    HCCL_ERROR("GetConstInputAcrossGraph failed");
    return HCCL_E_NOT_FOUND;
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetVectorFromTensor(const ge::GeTensor *tensor, std::vector<int64_t> &vector) {
  auto buff = tensor->GetData().GetData();
  const auto len = tensor->GetData().GetSize();
  auto buffTmp = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(buff));
  for (size_t i = 0UL; i < (len / sizeof(int64_t)); ++i) {
    vector.emplace_back(buffTmp[i]);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetRankId(const int64_t &hcomComm, const string &sGroup, u32 &rankId) {
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankId(sGroup.c_str(), &rankId));
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(HcomGetRankId(group, &rankId));
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetDestRank(const ge::OpDescPtr &opDescPtr, int32_t &destRank) {
  CHK_PRT_RET((!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_DEST_RANK, destRank)),
              HCCL_ERROR("[Get][RankSize] get attr[%s] failed.", HCOM_ATTR_DEST_RANK.c_str()), HCCL_E_PARA);

  CHK_PRT_RET((destRank < 0), HCCL_ERROR("[Get][destRank] destRank[%d] should not be negative.", destRank),
              HCCL_E_PARA);
  HCCL_DEBUG("get destRank[%d] success.", destRank);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetSrTag(const ge::OpDescPtr &opDescPtr, int32_t &srTag) {
  CHK_PRT_RET((!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_SR_TAG, srTag)),
              HCCL_ERROR("[Get][RankSize] get attr[%s] failed.", HCOM_ATTR_SR_TAG.c_str()), HCCL_E_PARA);

  CHK_PRT_RET((srTag < 0), HCCL_ERROR("[Get][srTag] srTag[%d] should not be negative.", srTag), HCCL_E_PARA);
  HCCL_DEBUG("get srTag[%d] success.", srTag);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetSrcRank(const ge::OpDescPtr &opDescPtr, int32_t &srcRank) {
  CHK_PRT_RET((!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_SRC_RANK, srcRank)),
              HCCL_ERROR("[Get][RankSize] get attr[%s] failed.", HCOM_ATTR_SRC_RANK.c_str()), HCCL_E_PARA);

  CHK_PRT_RET((srcRank < 0), HCCL_ERROR("[Get][destRank] destRank[%d] should not be negative.", srcRank), HCCL_E_PARA);
  HCCL_DEBUG("get srcRank[%d] success.", srcRank);

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::CheckAlltoAllvcRank(const ge::Node &node, const int64_t &hcomComm, const string &sGroup) {
  u32 rankId = 0;
  CHK_RET(GetRankId(hcomComm, sGroup, rankId));

  u32 alltoallvcRank = 0;
  bool bRet = ge::AttrUtils::GetInt(node.GetOpDesc(), "rank", alltoallvcRank);
  CHK_PRT_RET(
      !bRet,
      HCCL_ERROR("errNo[0x%016llx] get AlltoAllvc rank failed. no \"rank\" in opDesc", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);

  if (rankId != alltoallvcRank) {
    REPORT_PREDEFINED_ERR_MSG("EI0003", std::vector<const char *>({"ccl_op", "value", "parameter", "expect"}),
                              std::vector<const char *>({"CheckAlltoAllvcRank", std::to_string(alltoallvcRank).c_str(), "rankId",
                                                         std::to_string(rankId).c_str()}));
  }
  CHK_PRT_RET(rankId != alltoallvcRank,
              HCCL_ERROR("[%s][%s]errNo[0x%016llx] AlltoAllvc rank is invalid. "
                         "rank value %u, expect %u",
                         LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                         HCOM_ERROR_CODE(HCCL_E_PARA), alltoallvcRank, rankId),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetGroupFromOpDesc(const ge::OpDescPtr &op, std::string &sGroup) {
  if (ge::AttrUtils::HasAttr(op, "group")) {
    if (ge::AttrUtils::GetStr(op, "group", sGroup) == false) {
      HCCL_ERROR("[GetGroup][OpDesc]errNo[0x%016llx]: get group failed. get \"group\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
    CHK_PRT_RET(sGroup.empty(),
                HCCL_ERROR("[GetGroup][OpDesc]errNo[0x%016llx] get group name failed. group"
                           "from opDesc is empty.",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);
  } else {
    sGroup = HCCL_WORLD_GROUP;
  }
  HCCL_INFO("get group name[%s] success.", sGroup.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetTensorSize(const ge::GeTensorDesc &tensorDesc, int64_t &size) {
  auto shape = tensorDesc.GetShape();
  auto geDataType = tensorDesc.GetDataType();
  auto format = tensorDesc.GetFormat();

  int64_t sizeTemp = 0;
  bool bErr = (ge::TensorUtils::CalcTensorMemSize(shape, format, geDataType, sizeTemp) != ge::GRAPH_SUCCESS);
  CHK_PRT_RET((bErr) || (sizeTemp < 0),
              HCCL_ERROR("[Get][TensorMemSize]In GetTensorMemSize, CalcTensorMemSize"
                         "failed, Format[%d], dataType[%d], size[%lld]",
                         format, geDataType, sizeTemp),
              HCCL_E_PARA);

  size = sizeTemp;
  HCCL_DEBUG("[HcomOpsKernelBuilder][GetTensorSize]tensorSize %lld", size);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAllTensorSize(const ge::OpDescPtr &op, u32 tensorNum, std::vector<int64_t> &tensorSize) {
  for (size_t i = 0; i < tensorNum; i++) {
    auto inTensor = op->GetInputDesc(i);
    int64_t size = 0;
    HcclResult ret = GetTensorSize(inTensor, size);
    CHK_PRT_RET(
        ret, HCCL_ERROR("[Get][AllTensorSize]node[%s] input[%u] GetTensorSize failed.", op->GetName().c_str(), i), ret);
    HCCL_DEBUG("[HcomOpsKernelBuilder] node[%s] has %u inputs, input[%u] size %lld bytes.", op->GetName().c_str(),
               op->GetInputsSize(), i, size);
    tensorSize.push_back(size);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetAivCoreLimit(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                        u32 &aivCoreLimit) {
  aivCoreLimit = MAX_NUM_BLOCKS;
  if (ge::AttrUtils::HasAttr(op, "_op_vectorcore_num")) {  // 核数优先取算子级配置
    std::string aivNumStr;
    bool bRet = ge::AttrUtils::GetStr(op, "_op_vectorcore_num", aivNumStr);
    HCCL_DEBUG("[HcomOpUtils][GetAivCoreLimit] _op_vectorcore_num found [%s]", aivNumStr.c_str());
    if (bRet && aivNumStr != "") {
      s32 aivNum = 0;
      CHK_RET(SalStrToInt(aivNumStr, HCCL_BASE_DECIMAL, aivNum));
      HCCL_INFO("[HcomOpUtils][GetAivCoreLimit]node [%s] optype [%s] opVectorCoreNum [%u]", op->GetName().c_str(),
                sCollectiveType.c_str(), aivNum);
      if (aivNum == 0) {
        HCCL_ERROR("[HcomOpUtils][GetAivCoreLimit] node [%s] optype [%s] opVectorCoreNum 0 is illegel",
                   op->GetName().c_str(), sCollectiveType.c_str());
        return HCCL_E_PARA;
      }
      aivCoreLimit = aivNum;
    }
  } else {  // 算子级配置没有时尝试用GE全局配置
    std::string hardwareInfoStr;
    if (ge::GetContext().GetOption("ge.hardwareInfo", hardwareInfoStr) == ge::GRAPH_SUCCESS) {
      HCCL_DEBUG("[HcomOpUtils][GetAivCoreLimit] hardwareInfoStr [%s]", hardwareInfoStr.c_str());
      size_t pos = hardwareInfoStr.find("vector_core_cnt:");
      if (pos != string::npos) {
        pos += 16;  // "vector_core_cnt:"的长度是16
        size_t endPos = hardwareInfoStr.find(';', pos);
        if (endPos == string::npos) {
          endPos = hardwareInfoStr.size();
        }
        std::string aivNumStr = hardwareInfoStr.substr(pos, endPos - pos);
        if (aivNumStr != "") {
          s32 aivNum = 0;
          CHK_RET(SalStrToInt(aivNumStr, HCCL_BASE_DECIMAL, aivNum));
          HCCL_INFO("[HcomOpUtils][GetAivCoreLimit]node [%s] optype [%s] ge hardwareInfo [%u]", op->GetName().c_str(),
                    sCollectiveType.c_str(), aivNum);
          aivCoreLimit = aivNum > 0 ? aivNum : aivCoreLimit;
        }
      }
    } else {
      HCCL_DEBUG("[HcomOpUtils][GetAivCoreLimit] ge.hardwareInfo not found");
    }
  }
  HCCL_INFO("[HcomOpUtils][GetAivCoreLimit] op[%s] get aivCoreLimit[%u] success.", sCollectiveType.c_str(),
            aivCoreLimit);
  return HCCL_SUCCESS;
}

// 对比getcountfromopdesc接口，此接口更为完善，主要区分于allreduce算子的count计算方式
HcclResult HcomOpUtils::GetAccuracyCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                      HcclDataType dataType, u64 &count, u32 rankSize) {
  u32 dataTypeSize = 0;
  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  CHK_PRT_RET(dataTypeSize == 0, 
      HCCL_ERROR("[%s][Get][CountFromOpDesc]dataType size is zero.", __func__), HCCL_E_PARA);

  // Receive 算子不支持获取count
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    HCCL_RUN_WARNING("[%s][Get][Count] op[%s] get count failed. receive op not support get count.",
              __func__, sCollectiveType.c_str());
    return HCCL_SUCCESS;
  }

  // ALLREDUCE 特殊处理：使用 TensorUtils::GetSize 获取大小并立即返回
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
    CHK_RET(CalcAllReduceCount(op, sCollectiveType, dataTypeSize, count));
    return HCCL_SUCCESS;  
  }
  // BROADCAST 特殊处理
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    CHK_RET(CalcBroadcastCount(op, dataTypeSize, count));
    return HCCL_SUCCESS;  
  }
  // 其他算子通用处理
  CHK_RET(CalcCommonCount(op, sCollectiveType, dataTypeSize, rankSize, count));
  return HCCL_SUCCESS;
}

// ALLREDUCE 的count专用计算
HcclResult HcomOpUtils::CalcAllReduceCount(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                          u32 dataTypeSize, u64 &count) {
  constexpr u32 alignSize = 512; // 对齐大小为512字节的倍数
  u64 totalSize = 0;

  for (u64 i = 0; i < op->GetInputsSize(); i++) {
    int64_t tensorSize = 0;
    CHK_PRT_RET((ge::GRAPH_SUCCESS != ge::TensorUtils::GetSize(*op->GetInputDescPtr(i), tensorSize)),
        HCCL_ERROR("[Get][Count]errNo[0x%016llx] get workspace bytes failed. get size from TensorDesc"
                  "failed, op : %s, input index : %llu",
                  HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), i),
        HCCL_E_PARA);
    
    CHK_PRT_RET((static_cast<u64>(tensorSize) > INVALID_U64 - alignSize),
        HCCL_ERROR("op[%s] input size[%llu] is overflow.", sCollectiveType.c_str(), static_cast<u64>(tensorSize)),
        HCCL_E_PARA);

    totalSize += ((static_cast<u64>(tensorSize) + alignSize - 1) / alignSize * alignSize);
  }

  count = totalSize / dataTypeSize;
  HCCL_INFO("[%s]op[%s] get count[%llu] for allreduce success.", __func__, sCollectiveType.c_str(), count);
  return HCCL_SUCCESS;
}
 	 
// 除了allreduce算子以外的通用算子count计算
HcclResult HcomOpUtils::CalcCommonCount(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                        u32 dataTypeSize, u32 rankSize, u64 &count) {
  u64 totalSize = 0;

  for (u64 i = 0; i < op->GetInputsSize(); i++) {
    u64 shapeSize = static_cast<u64>(op->GetInputDescPtr(i)->GetShape().GetShapeSize());
    
    // 溢出检查
    CHK_PRT_RET(shapeSize > INVALID_U64 / dataTypeSize,
        HCCL_ERROR("op[%s] input size[%llu] * dataTypeSize[%u] is overflow.",
                  sCollectiveType.c_str(), shapeSize, dataTypeSize),
        HCCL_E_PARA);

    u64 inputSize = shapeSize * dataTypeSize;
    HCCL_INFO("[%s]op[%s] get inputSize[%llu] with dataTypeSize[%u] for index[%llu] success.",
              __func__, sCollectiveType.c_str(), inputSize, dataTypeSize, i);

    // 根据算子类型计算 blockSize
    u64 blockSize = 0;
    if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
      blockSize = inputSize / rankSize;
    } else {
      // ALLGATHER 和其他算子
      blockSize = inputSize;
    }
    
    // 溢出检查
    CHK_PRT_RET(totalSize > INVALID_U64 - blockSize,
        HCCL_ERROR("op[%s] totalSize[%llu] + blockSize[%llu] is overflow.",
                  sCollectiveType.c_str(), totalSize, blockSize),
        HCCL_E_PARA);

    totalSize += blockSize;
  }

  count = totalSize / dataTypeSize;
  HCCL_INFO("[%s]op[%s] get count[%llu] success.", __func__, sCollectiveType.c_str(), count);
  return HCCL_SUCCESS;
}

// broadcast等搬运算子在图编译阶段获取数据量需要做512对齐
HcclResult HcomOpUtils::CalcBroadcastCount(const ge::OpDescPtr &op, u32 dataTypeSize, u64 &count) {
  // 定义对齐大小为512字节的倍数
  constexpr u32 alignSize = 512;
  u64 totalSize = 0;

  // 遍历所有输入tensor
  for (u64 i = 0; i < op->GetInputsSize(); i++) {
    // 获取输入tensor的实际大小
    int64_t tensorSize = 0;
    CHK_PRT_RET((ge::GRAPH_SUCCESS != ge::TensorUtils::GetSize(*op->GetInputDescPtr(i), tensorSize)),
        HCCL_ERROR("[Calc][BroadcastCount]errNo[0x%016llx] get size from TensorDesc failed, "
                  "op: %s, input index: %llu",
                  HCOM_ERROR_CODE(HCCL_E_PARA), op->GetName().c_str(), i),
        HCCL_E_PARA);
    // 溢出检查：确保 tensorSize 不会超过 INVALID_U64 - alignSize
    CHK_PRT_RET((static_cast<u64>(tensorSize) > INVALID_U64 - alignSize),
        HCCL_ERROR("[Calc][BroadcastCount]op[%s] input size[%llu] is overflow.",
                  op->GetName().c_str(), static_cast<u64>(tensorSize)),
        HCCL_E_PARA);
    // 对输入大小进行512字节对齐
    u64 blockSize = (static_cast<u64>(tensorSize) + alignSize - 1) / alignSize * alignSize;
    // 溢出检查：确保 totalSize + blockSize 不会溢出
    CHK_PRT_RET(totalSize > INVALID_U64 - blockSize,
        HCCL_ERROR("[Calc][BroadcastCount]op[%s] totalSize[%llu] + blockSize[%llu] is overflow.",
                  op->GetName().c_str(), totalSize, blockSize),
        HCCL_E_PARA);
    
    totalSize += blockSize;
    
    HCCL_INFO("[Calc][BroadcastCount]op[%s] input[%llu]: tensorSize[%lld], blockSize[%llu], totalSize[%llu]",
              op->GetName().c_str(), i, tensorSize, blockSize, totalSize);
  }
  
  // 计算最终count（总大小除以数据类型大小）
  count = totalSize / dataTypeSize;
  HCCL_INFO("[Calc][BroadcastCount]op[%s] get count[%llu] success, dataTypeSize[%u], totalSize[%llu]",
            op->GetName().c_str(), count, dataTypeSize, totalSize);
  
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetCountFromOpDescSuperkernel(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                      HcclDataType dataType, u64 &count, u32 rankSize) {
  u64 totalSize = 0;
  u32 dataTypeSize = 0;

  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[Get][CountFromOpDesc]dataType size is zero."), HCCL_E_PARA);

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    return HCCL_E_PARA;
  } else {
    for (u64 i = 0; i < op->GetInputsSize(); i++) {
      u64 blockSize;
      int64_t inputSize = 0;
      inputSize = static_cast<u64>(op->GetInputDescPtr(i)->GetShape().GetShapeSize());
      inputSize = inputSize * dataTypeSize;
      if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        blockSize = static_cast<u64>(inputSize / rankSize);
      } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
        blockSize = static_cast<u64>(inputSize);
      } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        blockSize = static_cast<u64>(inputSize);
      } else {
        blockSize = static_cast<u64>(inputSize / rankSize);
      }
      totalSize = totalSize + blockSize;
    }
  }
  count = totalSize / dataTypeSize;
  HCCL_INFO("SPK op[%s] get count[%llu] success.", sCollectiveType.c_str(), count);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetTensorCleanTaskNum(const ge::Node &node, const std::string &sCollectiveType, u32 &taskNum) {
  u32 tensorNum = 0;
  // 获取tensor的数量
  CHK_RET(GetTensorNum(node, sCollectiveType, tensorNum));
  if (tensorNum != 0) {
    CHK_RET(GetTaskNumFromCrackSize(node, tensorNum, taskNum));
  }
  HCCL_DEBUG("[GetTensorCleanTaskNum] cur task num[%u].", taskNum);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetTensorNum(const ge::Node &node, const std::string &sCollectiveType, u32 &tensorNum) {
  constexpr const char *kCleanSeparately = "1";
  std::string atomic_clean_policy;
  bool needCleanSeparately =
      (ge::GetThreadLocalContext().GetOption(ge::ATOMIC_CLEAN_POLICY, atomic_clean_policy) == ge::GRAPH_SUCCESS) &&
      (atomic_clean_policy == kCleanSeparately);
  if (needCleanSeparately &&
      ((sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) || (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) ||
       (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE))) {
    // 获取Tensor的个数
    tensorNum = node.GetOpDesc()->GetInputsSize();
  } else {
    tensorNum = 0;
  }
  HCCL_DEBUG("[GetTensorNum] sCollectiveType[%s] tensorNum[%u].", sCollectiveType.c_str(), tensorNum);
  return HCCL_SUCCESS;
}

HcclResult HcomOpUtils::GetTaskNumFromCrackSize(const ge::Node &node, u32 tensorNum, u32 &taskNum) {
  bool crackSizeBigger32 = false;
  s64 tensorSize[tensorNum] = {0};
  s64 crackSize[tensorNum] = {0};
  std::vector<int64_t> tensorSizeTemp;

  auto op = node.GetOpDesc();
  CHK_PTR_NULL(op);
  // 获取tensor的大小
  CHK_RET(GetAllTensorSize(op, tensorNum, tensorSizeTemp));
  CHK_SAFETY_FUNC_RET(
      memcpy_s(tensorSize, tensorNum * sizeof(s64), tensorSizeTemp.data(), tensorSizeTemp.size() * sizeof(s64)));
  // 获取缝隙的大小
  for (u32 i = 0; i < tensorNum; i++) {
    s64 crackSizeTemp = 0;
    crackSizeTemp =
        (tensorSize[i] + TENSOR_ALIGNMENT_32 - 1) / TENSOR_ALIGNMENT_32 * TENSOR_ALIGNMENT_32 + TENSOR_ALIGNMENT_32;
    crackSizeTemp = (crackSizeTemp + TENSOR_ALIGNMENT_512 - 1) / TENSOR_ALIGNMENT_512 * TENSOR_ALIGNMENT_512;
    crackSizeTemp = crackSizeTemp - tensorSize[i];
    crackSize[i] = crackSizeTemp;
    if (crackSize[i] < AICORE_MIN_CLEAR_ZEOR_SIZE) {
      taskNum++;
    } else if (!crackSizeBigger32) {
      crackSizeBigger32 = true;
    }
  }
  if (crackSizeBigger32) {
    taskNum++;
  }
  return HCCL_SUCCESS;
}
}  // namespace hccl
