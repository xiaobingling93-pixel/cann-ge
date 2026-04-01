/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
#define INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include "acl/acl_rt.h"
#include "runtime/rt.h"
#include "graph/op_desc.h"

namespace ge {
struct HcclDumpInfo {
  uint32_t task_id;
  uint32_t stream_id;
  uint32_t sub_task_type;
  void *input_addr;
  uint64_t input_size;
  void *output_addr;
  uint64_t output_size;
};

struct DvppInfo {
  OpDescPtr op_desc;
  std::vector<void *> io_addrs;
  uint32_t sqe[16];
};

// when need to eliminate GETaskKernelHcclInfo, so not need DAVINCI_TRAIN/DAVINCI_CLOUD
struct GETaskKernelHcclInfo {
  std::string input_name;
  std::string hccl_type;
  void *inputDataAddr;
  void *outputDataAddr;
  void *workSpaceAddr;
  int64_t count;
  int32_t dataType;
  int32_t opType;
  int64_t rootId;
  uint64_t workSpaceMemSize;
  std::vector<int64_t> dims;
  std::vector<aclrtStream> hcclStreamList;
  std::vector<HcclDumpInfo> hccl_dump_info;
  std::vector<void *> global_workspace_addr;
  uint32_t hcclQosCfg;
  std::vector<void *> inputDataAddrs;
  std::vector<void *> outputDataAddrs;
  std::vector<void *> workSpaceAddrs;
  std::vector<uint64_t> workSpaceMemSizes;
  std::vector<int32_t> inputZeroCopyFlags;
  std::vector<int32_t> outputZeroCopyFlags;
};

struct GETaskInfo {
  uint32_t id;
  uint16_t type;
  uint32_t streamID;
  void *stream;  // rtKernelLaunch input argument
  void *event;
  void *privateDef;
  uint32_t privateDefLen;
  void *opsKernelStorePtr;
  std::vector<GETaskKernelHcclInfo> kernelHcclInfo;
  DvppInfo dvpp_info;
  bool needRefresh{false};
  std::vector<void *> rt_attached_streams;
};

struct HcomRemoteAccessAddrInfo
{
  uint32_t remotetRankID;
  uint64_t remoteAddr;  // host embedding table address
  uint64_t localAddr;  // device HBM address
  uint64_t length;   // memory Length in Bytes
};


}  // namespace ge
#endif  // INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
