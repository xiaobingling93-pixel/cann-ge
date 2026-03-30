/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_graph_mc2.h"
#include "register/hidden_inputs_func_registry.h"
#include "register/op_tiling_info.h"
#include "hcom/hcom_topo_info.h"
#include "hccl/hcom.h"
#include "hcom_op_utils.h"
#include "hcom_plugin.h"
#include "graph/debug/ge_attr_define.h"
#include "hcom_graph_optimizer.h"
#include "hcom_acl_adapter.h"

namespace hccl {
static constexpr u32 NEW_TILING_VERSION = 2U;
const uint32_t GROUP_NAME_OFFSET = 5;
bool HcomGetGroupsByOpDesc(const ge::OpDescPtr &opdesc, std::vector<std::string> &groups) {
  std::string group;
  for (const auto& groupName : opdesc->GetAllAttrNames()) {
    HCCL_DEBUG("Get attr Name [%s]", groupName.c_str());
    if (groupName.substr(0, GROUP_NAME_OFFSET) == "group" && ge::AttrUtils::GetStr(opdesc, groupName, group)) {
      HCCL_INFO("Get group %s:%s of op %s.", groupName.c_str(), group.c_str(), opdesc->GetName().c_str());
      groups.push_back(group);
    }
  }

  return true;
}

u32 HcomGetTilingVersionByOpDesc(const ge::OpDescPtr &opdesc, std::string &tilingData) {
  const auto tiling = opdesc->GetExtAttr<std::shared_ptr<optiling::utils::OpRunInfo>>(ge::ATTR_NAME_OP_RUN_INFO);
  if (tiling == nullptr || *tiling == nullptr) {
    HCCL_WARNING("Failed to get tiling info of op %s.", opdesc->GetName().c_str());
    return 0U;
  }

  tilingData = (*tiling)->GetAllTilingData().str();
  const size_t tilingSize = tilingData.size();
  u32 expectedSize = sizeof(u32) + sizeof(u32);
  if (tilingSize < expectedSize) {
    HCCL_WARNING("Invalid tiling size(%zu) of op %s.", tilingSize, opdesc->GetName().c_str());
    return 0U;
  }

  return *reinterpret_cast<const u32 *>(tilingData.c_str());
}

rtStream_t HcomGetStreamByOpDesc(const ge::OpDescPtr &opdesc) {
  const auto rtList = opdesc->TryGetExtAttr<std::shared_ptr<std::vector<void *>>>("_rt_resource_list", nullptr);
  if (rtList == nullptr) {
    HCCL_ERROR("Failed to get rt list of op %s.", opdesc->GetName().c_str());
    return nullptr;
  }

  if (rtList->size() != 1) {
    HCCL_ERROR("Invalid rt list size(%zu) of op %s.", rtList->size(), opdesc->GetName().c_str());
    return nullptr;
  }

  return (*rtList)[0UL];
}

void *HcomGetContext(const rtStream_t stream, const void *tilingData, const char *groupName) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomGetContextV2(stream, tilingData, groupName);
  }
#endif
  HcclComm commHandle = nullptr;
  HcclResult ret = HcomGetCommHandleByGroup(groupName, &commHandle);
  CHK_PRT_RET(ret != HCCL_SUCCESS || commHandle == nullptr,
              HCCL_ERROR("[Get][CommHandle]Failed to get hcom commHandle, group[%s] errNo[0x%016llx].", groupName,
                         HCCL_ERROR_CODE(ret)),
              nullptr);

  void *context = nullptr;
  if (tilingData != nullptr) {
    ret = HcclAllocComResourceByTiling(commHandle, stream, const_cast<void *>(tilingData), &context);
  } else {
    uint64_t streamMode = 0;
    CHK_PRT_RET(hrtStreamGetMode(stream, &streamMode) != HCCL_SUCCESS, HCCL_ERROR("Failed to get stream mode."),
                nullptr);
    ret = HcomCreateComResourceByComm(commHandle, streamMode, true, &context);
  }
  CHK_PRT_RET(ret != HCCL_SUCCESS || context == nullptr,
              HCCL_ERROR("Failed to create ComResource by tiling, errNo[0x%016llx].", HCCL_ERROR_CODE(ret)), nullptr);

  ge::HcomTopoInfo::TopoInfo topoInfo;
  CHK_PRT_RET(!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(groupName, topoInfo),
              HCCL_ERROR("Failed to get topo info for group %s.", groupName), nullptr);

  rtStream_t opstream;
  ret = HcomGetAicpuOpStreamNotify(groupName, &opstream, 1, &topoInfo.notify_handle);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("Failed to get Aicpu Op Stream Notify, errNo[0x%016llx].", HCCL_ERROR_CODE(ret)), nullptr);

  HCCL_INFO("Set notify %p to group %s.", topoInfo.notify_handle, groupName);
  ge::HcomTopoInfo::Instance().SetGroupTopoInfo(groupName, topoInfo);

  HCCL_INFO("Create context(%s) for group %s successfully, address %p.", (tilingData != nullptr ? "v2" : "v1"),
            groupName, context);
  return context;
}

#ifndef OPEN_BUILD_PROJECT
void *HcomGetContextV2(const rtStream_t stream, const void *tilingData, const char *groupName) {
  HcclComm com = nullptr;
  HcclResult ret = HcomGetCommHandleByGroup(groupName, &com);

  void *context = nullptr;
  if (tilingData != nullptr) {
    ret = HcclAllocComResourceByTiling(com, stream, const_cast<void *>(tilingData), &context);
  } else {
    uint64_t streamMode = 0;
    CHK_PRT_RET(hrtStreamGetMode(stream, &streamMode) != HCCL_SUCCESS, HCCL_ERROR("Failed to get stream mode."),
                nullptr);
    ret = HcomCreateComResourceByComm(com, streamMode, true, &context);
  }
  CHK_PRT_RET(ret != HCCL_SUCCESS || context == nullptr,
              HCCL_ERROR("Failed to create ComResource by tiling, errNo[0x%016llx].", HCCL_ERROR_CODE(ret)), nullptr);

  ge::HcomTopoInfo::TopoInfo topoInfo;
  CHK_PRT_RET(!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(groupName, topoInfo),
              HCCL_ERROR("Failed to get topo info for group %s.", groupName), nullptr);

  HCCL_INFO("Set notify %p to group %s.", topoInfo.notify_handle, groupName);
  ge::HcomTopoInfo::Instance().SetGroupTopoInfo(groupName, topoInfo);

  HCCL_INFO("Create context(%s) for group %s successfully, address %p.", (tilingData != nullptr ? "v2" : "v1"),
            groupName, context);
  return context;
}
#endif

HcclResult GetCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType, HcclDataType dataType,
                              u64 &count, u32 rankSize) {
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

ge::graphStatus HcomCreateComResourceMC2(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts) {
  HCCL_INFO("[HcomCreateComResource]Hcomm create com resource of op %s, type %s", opdesc->GetName().c_str(),
            opdesc->GetType().c_str());

  std::vector<std::string> groups{};
  if (!HcomGetGroupsByOpDesc(opdesc, groups) || groups.empty()) {
    HCCL_ERROR("Failed to get groups of op %s", opdesc->GetName().c_str());
    return ge::GRAPH_FAILED;
  }

  const rtStream_t stream = HcomGetStreamByOpDesc(opdesc);
  if (stream == nullptr) {
    HCCL_ERROR("Failed to get attached stream of op %s", opdesc->GetName().c_str());
    return ge::GRAPH_FAILED;
  }

  std::string tilingData;
  const u32 version = HcomGetTilingVersionByOpDesc(opdesc, tilingData);
  HCCL_INFO("Tiling version of op %s is %u.", opdesc->GetName().c_str(), version);

  for (const auto& group : groups) {
    HCCL_INFO("HcomCreateComResourceMC2 group is %s.", group.c_str());
    if (group.empty()) {
      HCCL_RUN_INFO("[HcomCreateComResourceMC2] group is empty, push nullptr to context.");
      contexts.push_back(nullptr);
      continue;
    }
    void *context = HcomGetContext(stream, reinterpret_cast<const void *>(tilingData.c_str()), group.c_str());
    CHK_PRT_RET(context == nullptr, HCCL_ERROR("Failed to create ComResource by tiling."), ge::GRAPH_FAILED);
    contexts.push_back(context);
  }
  return ge::GRAPH_SUCCESS;
}
REG_HIDDEN_INPUTS_FUNC(ge::HiddenInputsType::HCOM, HcomCreateComResource);
ge::graphStatus HcomCreateComResource(const ge::OpDescPtr &opdesc, std::vector<void *> &contexts) {
  HCCL_INFO("[HcomCreateComResource]Hcom create com resource of op %s.", opdesc->GetName().c_str());
  ge::graphStatus gRet = ge::GRAPH_FAILED;
  std::string sCollectiveType = opdesc->GetType();

  auto iter = HCCL_OPTYPE_NAME_MAP.find(sCollectiveType);
  HcclCMDType opType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;
  if (opType == HcclCMDType::HCCL_CMD_INVALID) {
    HCCL_INFO("Select HcomCreateComResourceMC2.");
    gRet = HcomCreateComResourceMC2(opdesc, contexts);
  } else {
    HCCL_ERROR("[HcomCreateComResource]HcomCreateComResource failed, opType[%d] not support MC2, sCollectiveType[%s]",
               opType, sCollectiveType.c_str());
  }
  return gRet;
}
}  // namespace hccl
