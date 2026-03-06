/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HCOM_COMM_H
#define OP_HCOM_COMM_H

#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/node.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "hccl/hccl_ex.h"
#include <nlohmann/json.hpp>

#include "hccl/hcom.h"
#include "hccl/base.h"

namespace hccl {
/* 原图shap类型配置 */
const std::string ORIGINAL_GRAPH_SHAPE_TYPE = "original_graph_shape_type";
constexpr std::int64_t ORIGINAL_GRAPH_KNOWNSHAPE_TYPE = 0;
constexpr std::int64_t ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE = 1;

constexpr u32 TENSOR_ALIGNMENT_512 = 512;
constexpr u32 TENSOR_ALIGNMENT_32 = 32;
constexpr u32 UNIQUE_TAG_MAX_LEN = 64;

constexpr u32 GROUP_NAME_MAX_LEN = 127;  // 最大的group name 长度
constexpr u64 INVALID_U64 = 0xFFFFFFFFFFFFFFFF;
constexpr u32 INVALID_UINT = 0xFFFFFFFF;
constexpr u32 MAX_MODULE_DEVICE_NUM = 32;  // 单server双模组时支持最大的设备数量
constexpr char HCCL_WORLD_GROUP[] = "hccl_world_group";
constexpr u32 MAX_NUM_BLOCKS = 48;
constexpr int HCCL_BASE_DECIMAL = 10;            // 10进制字符串转换
constexpr u32 RANK_TABLE_MAX_LEN = 4096 - 1;     // PATH_MAX=4096
constexpr s64 HCCL_ALIGN_SIZE = 4096;            // hccl  对齐方式， 按4KB来对齐
constexpr s64 HCCL_WORKSPACE_MEM_32_KB = 32768;  // hccl内存大小，暂定32KB
constexpr u32 HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE = 200;
constexpr u64 HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE = (1 * 1024 * 1024);
constexpr u64 CCL_COMM_INBUFFER_UNALIGNED_RESERVE_SIZE = (1 * 1024 * 1024);  // 1 * 1024 * 1024, 即1M

/* 公共模块函数返回值定义,跟业务层同步  */
const std::map<HcclDataType, std::string> HCOM_DATA_TYPE_STR_MAP{
    {HcclDataType::HCCL_DATA_TYPE_INT8, "int8"},       {HcclDataType::HCCL_DATA_TYPE_INT16, "int16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "int32"},     {HcclDataType::HCCL_DATA_TYPE_INT64, "int64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "uint64"},   {HcclDataType::HCCL_DATA_TYPE_FP16, "float16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "float32"},    {HcclDataType::HCCL_DATA_TYPE_UINT8, "uint8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "uint16"},   {HcclDataType::HCCL_DATA_TYPE_UINT32, "uint32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "float64"},    {HcclDataType::HCCL_DATA_TYPE_BFP16, "bfloat16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "int128"},   {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, "fp8e4m3"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, "fp8e5m2"}, {HcclDataType::HCCL_DATA_TYPE_RESERVED, "reserved"}};

const std::map<HcclReduceOp, std::string> HCOM_REDUCE_OP_STR_MAP{{HcclReduceOp::HCCL_REDUCE_SUM, "sum"},
                                                                 {HcclReduceOp::HCCL_REDUCE_PROD, "prod"},
                                                                 {HcclReduceOp::HCCL_REDUCE_MAX, "max"},
                                                                 {HcclReduceOp::HCCL_REDUCE_MIN, "min"},
                                                                 {HcclReduceOp::HCCL_REDUCE_RESERVED, "reserved"}};

inline std::string GetDataTypeEnumStr(HcclDataType dataType) {
  auto iter = HCOM_DATA_TYPE_STR_MAP.find(dataType);
  if (iter == HCOM_DATA_TYPE_STR_MAP.end()) {
    return "HcclDataType(" + std::to_string(dataType) + ")";
  } else {
    return iter->second;
  }
}

inline std::string GetDataTypeEnumStr(u32 dataType) {
  auto hcclDataType = static_cast<HcclDataType>(dataType);
  return GetDataTypeEnumStr(hcclDataType);
}

inline std::string GetReduceOpEnumStr(HcclReduceOp reduceOp) {
  auto iter = HCOM_REDUCE_OP_STR_MAP.find(reduceOp);
  if (iter == HCOM_REDUCE_OP_STR_MAP.end()) {
    return "HcclReduceOp(" + std::to_string(reduceOp) + ")";
  } else {
    return iter->second;
  }
}

using DeterministicEnableLevel = enum {
  DETERMINISTIC_DISABLE = 0,  // 不支持确定性
  DETERMINISTIC_ENABLE,       // 支持确定性，不支持规约保序
  DETERMINISTIC_STRICT        // 支持确定性以及规约保序
};

// 全局工作空间类型
enum class GlobalWorkSpaceType {
  OVERFLOW_DETECT_MODE = 0,
};

enum class HcclTopoLevel {
  HCCL_TOPO_L0 = 0,
  HCCL_TOPO_L1,
  HCCL_TOPO_MAX,
};

constexpr u32 SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(s8),
                                                     sizeof(s16),
                                                     sizeof(s32),
                                                     2,
                                                     sizeof(float),
                                                     sizeof(s64),
                                                     sizeof(u64),
                                                     sizeof(u8),
                                                     sizeof(u16),
                                                     sizeof(u32),
                                                     8,
                                                     2,
                                                     16,
                                                     2,
                                                     1,
                                                     1,
                                                     1,
                                                     1};

struct AlltoAllVParamsInfo {
  u64 sendCounts[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 sendDispls[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 recvCounts[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 recvDispls[ALLTOALLV_RANK_MAX_NUM] = {0};
  HcclDataType sendType = HCCL_DATA_TYPE_RESERVED;
  HcclDataType recvType = HCCL_DATA_TYPE_RESERVED;
};

struct AlltoAllVCParamsInfo {
  u64 sendCountMatrix[ALLTOALLVC_RANK_MAX_NUM * ALLTOALLVC_RANK_MAX_NUM] = {0};
  HcclDataType sendType = HCCL_DATA_TYPE_RESERVED;
  HcclDataType recvType = HCCL_DATA_TYPE_RESERVED;
};

struct ReduceScatterVParamsInfo {
  u64 sendCounts[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 sendDispls[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 recvCounts[ALLTOALLV_RANK_MAX_NUM] = {0};
};

struct AllGatherVParamsInfo {
  u64 sendCount[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 recvDispls[ALLTOALLV_RANK_MAX_NUM] = {0};
  u64 recvCounts[ALLTOALLV_RANK_MAX_NUM] = {0};
};

using HCCL_KERNEL_INFO_PRIVATE_DEF = struct hcclKernelInfoPrivateDef {
  u8 group[GROUP_NAME_MAX_LEN + 1] = {0};  // 1为结束符预留
  size_t nodeNameHash = 0;
  size_t tensorNum = 0;
  size_t privateDefSize = 0;
  u32 graphId = 0;
  u32 srcRank = 0;
  u32 destRank = 0;
  u32 selfRank = 0;
  u32 srTag = 0;
  u32 originalGraphShapeType = 0;
  int64_t comm = 0;
  HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
  bool needMapRank = false;
  bool isOfflineComp = false;                 // 是否是离线编译
  DevType devType = DevType::DEV_TYPE_COUNT;  // 只有离线编译时需要，在loadtask的时候校验一致性
  u32 aivCoreLimit = 0;
};

using HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF = struct hcclAlltoallvKernelInfoPrivateDef : HCCL_KERNEL_INFO_PRIVATE_DEF {
  AlltoAllVParamsInfo paramsInfo = AlltoAllVParamsInfo();
  AlltoAllVCParamsInfo cparamsInfo = AlltoAllVCParamsInfo();

  hcclAlltoallvKernelInfoPrivateDef(HCCL_KERNEL_INFO_PRIVATE_DEF KernelInfoPrivateDef) {
    for (u32 i = 0; i < GROUP_NAME_MAX_LEN + 1; i++) {
      this->group[i] = KernelInfoPrivateDef.group[i];
    }
    nodeNameHash = KernelInfoPrivateDef.nodeNameHash;
    tensorNum = KernelInfoPrivateDef.tensorNum;
    privateDefSize = KernelInfoPrivateDef.privateDefSize;
    graphId = KernelInfoPrivateDef.graphId;
    srcRank = KernelInfoPrivateDef.srcRank;
    destRank = KernelInfoPrivateDef.destRank;
    selfRank = KernelInfoPrivateDef.selfRank;
    srTag = KernelInfoPrivateDef.srTag;
    originalGraphShapeType = KernelInfoPrivateDef.originalGraphShapeType;
    comm = KernelInfoPrivateDef.comm;
    dataType = KernelInfoPrivateDef.dataType;
    needMapRank = KernelInfoPrivateDef.needMapRank;
    aivCoreLimit = KernelInfoPrivateDef.aivCoreLimit;
  };
};

using HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF =
    struct hcclReduceScatterVKernelInfoPrivateDef : HCCL_KERNEL_INFO_PRIVATE_DEF {
  ReduceScatterVParamsInfo paramsInfo = ReduceScatterVParamsInfo();

  hcclReduceScatterVKernelInfoPrivateDef(HCCL_KERNEL_INFO_PRIVATE_DEF KernelInfoPrivateDef) {
    for (u32 i = 0; i < GROUP_NAME_MAX_LEN + 1; i++) {
      this->group[i] = KernelInfoPrivateDef.group[i];
    }
    nodeNameHash = KernelInfoPrivateDef.nodeNameHash;
    tensorNum = KernelInfoPrivateDef.tensorNum;
    privateDefSize = KernelInfoPrivateDef.privateDefSize;
    graphId = KernelInfoPrivateDef.graphId;
    srcRank = KernelInfoPrivateDef.srcRank;
    destRank = KernelInfoPrivateDef.destRank;
    selfRank = KernelInfoPrivateDef.selfRank;
    srTag = KernelInfoPrivateDef.srTag;
    originalGraphShapeType = KernelInfoPrivateDef.originalGraphShapeType;
    comm = KernelInfoPrivateDef.comm;
    dataType = KernelInfoPrivateDef.dataType;
    needMapRank = KernelInfoPrivateDef.needMapRank;
    aivCoreLimit = KernelInfoPrivateDef.aivCoreLimit;
  };
};

using HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF =
    struct hcclAllGatherVKernelInfoPrivateDef : HCCL_KERNEL_INFO_PRIVATE_DEF {
  AllGatherVParamsInfo paramsInfo = AllGatherVParamsInfo();

  hcclAllGatherVKernelInfoPrivateDef(HCCL_KERNEL_INFO_PRIVATE_DEF KernelInfoPrivateDef) {
    for (u32 i = 0; i < GROUP_NAME_MAX_LEN + 1; i++) {
      this->group[i] = KernelInfoPrivateDef.group[i];
    }
    nodeNameHash = KernelInfoPrivateDef.nodeNameHash;
    tensorNum = KernelInfoPrivateDef.tensorNum;
    privateDefSize = KernelInfoPrivateDef.privateDefSize;
    graphId = KernelInfoPrivateDef.graphId;
    srcRank = KernelInfoPrivateDef.srcRank;
    destRank = KernelInfoPrivateDef.destRank;
    selfRank = KernelInfoPrivateDef.selfRank;
    srTag = KernelInfoPrivateDef.srTag;
    originalGraphShapeType = KernelInfoPrivateDef.originalGraphShapeType;
    comm = KernelInfoPrivateDef.comm;
    dataType = KernelInfoPrivateDef.dataType;
    needMapRank = KernelInfoPrivateDef.needMapRank;
    aivCoreLimit = KernelInfoPrivateDef.aivCoreLimit;
  };
};

const std::vector<std::string> HCOM_SUPPORTED_OP_TYPE = {
    HCCL_KERNEL_OP_TYPE_BROADCAST,      HCCL_KERNEL_OP_TYPE_REDUCE,     HCCL_KERNEL_OP_TYPE_ALLREDUCE,
    HCCL_KERNEL_OP_TYPE_ALLGATHER,      HCCL_KERNEL_OP_TYPE_ALLGATHERV, HCCL_KERNEL_OP_TYPE_REDUCESCATTER,
    HCCL_KERNEL_OP_TYPE_REDUCESCATTERV, HCCL_KERNEL_OP_TYPE_SEND,       HCCL_KERNEL_OP_TYPE_RECEIVE,
    HCCL_KERNEL_OP_TYPE_ALLTOALLV,      HCCL_KERNEL_OP_TYPE_ALLTOALLVC, HCCL_KERNEL_OP_TYPE_ALLTOALL};

HcclResult SetWorkspaceResource(const std::string &tag, const char *group, std::vector<rtStream_t> stream, void *memPtr,
                                u64 maxSize);
HcclResult GetCCLBufferAvailableSize(u64 &size);
HcclResult GetInCCLbuffer(const char *group, void *&buffer, u64 &size);
HcclResult GetOutCCLbuffer(const char *group, void *&buffer, u64 &size);

HcclResult HcomInitialize();
HcclResult GetOpDescIntAttr(const ge::OpDesc &op, const string &attr, s32 &output);
HcclResult GetOpDescStrAttr(const ge::OpDesc &op, const string &attr, string &output);
HcclResult InitGroup();

HcclResult HcomLoadRanktableFile(const std::string &rankTablePath, std::string &rankTableM);
bool CheckFilePath(const std::string &filePath, const std::string &fileType);
HcclResult ReadFile(const std::string &readFile, nlohmann::json &fileContent);
HcclResult HcomGetRanktableRealPath(const char *rankTable, std::string &realFilePath);
HcclResult GetJsonProperty(const nlohmann::json &obj, const char *propName, std::string &propValue);
void SalSleep(u32 sec);
HcclResult SalStrToInt(const std::string str, int base, s32 &val);
HcclResult SalStrToULong(const std::string str, int base, u32 &val);
HcclResult SalStrToULonglong(const std::string str, int base, u64 &val);
HcclResult SalGetDataTypeSize(HcclDataType dataType, u32 &dataTypeSize);
HcclResult SalParseInformation(nlohmann::json &parseInformation, const std::string &information);
void SetThreadName(const std::string &threadStr);

bool IsSocVersion91093(std::string socVersion);
bool IsSocVersion910B(std::string socVersion);
bool IsSocVersion910(std::string socVersion);

}  // namespace hccl
#endif
