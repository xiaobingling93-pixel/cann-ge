/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM__OP_UTILS_H
#define HCOM__OP_UTILS_H

#include <map>
#include <string>
#include "hccl/hccl_types.h"
#include "graph/op_desc.h"
#include "graph/utils/node_utils.h"
#include "platform/platform_info.h"
#include "hccl/base.h"
#include "ops_kernel_builder_base.h"
#include "graph/compute_graph.h"
#include "hcom_log.h"
#include "op_hcom_comm.h"

namespace hccl {
const std::string AUTOTUNE_HCCL_OPS_LIB_NAME = "ops_kernel_info_hccl_gradtune";
const std::string HCCL_OPS_LIB_NAME = "ops_kernel_info_hccl";

const std::string HCOM_ATTR_RANK_SIZE = "rank_size";
const std::string HCOM_ATTR_SHAPE = "shape";

const std::map<ge::DataType, HcclDataType> HCOM_DATA_TYPE_MAP = {{ge::DT_FLOAT, HCCL_DATA_TYPE_FP32},
                                                                 {ge::DT_FLOAT16, HCCL_DATA_TYPE_FP16},
                                                                 {ge::DT_INT8, HCCL_DATA_TYPE_INT8},
                                                                 {ge::DT_INT32, HCCL_DATA_TYPE_INT32},
                                                                 {ge::DT_INT16, HCCL_DATA_TYPE_INT16},
                                                                 {ge::DT_INT64, HCCL_DATA_TYPE_INT64},
                                                                 {ge::DT_UINT64, HCCL_DATA_TYPE_UINT64},
                                                                 {ge::DT_UINT8, HCCL_DATA_TYPE_UINT8},
                                                                 {ge::DT_UINT16, HCCL_DATA_TYPE_UINT16},
                                                                 {ge::DT_UINT32, HCCL_DATA_TYPE_UINT32},
                                                                 {ge::DT_DOUBLE, HCCL_DATA_TYPE_FP64},
                                                                 {ge::DT_BF16, HCCL_DATA_TYPE_BFP16},
#ifndef OPEN_BUILD_PROJECT
                                                                 {ge::DT_HIFLOAT8, HCCL_DATA_TYPE_HIF8},
                                                                 {ge::DT_FLOAT8_E5M2, HCCL_DATA_TYPE_FP8E5M2},
                                                                 {ge::DT_FLOAT8_E4M3FN, HCCL_DATA_TYPE_FP8E4M3},
                                                                 {ge::DT_FLOAT8_E8M0, HCCL_DATA_TYPE_FP8E8M0}
#endif
};

const std::map<std::string, HcclReduceOp> HCOM_REDUCE_TYPE_MAP{
    {"sum", HCCL_REDUCE_SUM},
    {"prod", HCCL_REDUCE_PROD},
    {"max", HCCL_REDUCE_MAX},
    {"min", HCCL_REDUCE_MIN},
};

enum class CreateDir {
  HCCL_DIR_NUM_NEG = -1,
  HCCL_DIR_NUM_ZEO = 0,
  HCCL_DIR_NUM_ONE = 1,
  HCCL_DIR_NUM_TWO = 2,
  HCCL_DIR_NUM_TRE = 3
};

// Ge适配的类
constexpr u32 AICORE_MIN_CLEAR_ZEOR_SIZE = 32;
constexpr u32 TASK_MAX_NUM_DEV_TYPE_V80 = 2042;
constexpr u32 TASK_MAX_NUM_DEV_TYPE_V71 = 4061;
constexpr u32 ALLTOALLV_INPUT_VEC_SIZE = 4;
constexpr u32 V_INPUT_VEC_SIZE = 3;

class HcomOpUtils {
 public:
  HcomOpUtils() = default;
  ~HcomOpUtils() = default;
  static HcclResult GetReduction(const ge::OpDescPtr &opDescPtr, HcclReduceOp &reduction);
  static HcclResult GetRoot(const ge::OpDescPtr &opDescPtr, int32_t &root);
  static HcclResult GetRankSize(const ge::OpDescPtr &opDescPtr, int32_t &rankSize);
  static HcclResult GetGroup(const ge::OpDescPtr &opDescPtr, std::string &group);
  static HcclResult GetDataType(const ge::OpDescPtr &opDescPtr, HcclDataType &dataType);
  static HcclResult GetDataType(const ge::OpDescPtr &opDescPtr, std::string &dataType);
  static HcclResult GetSrcRank(const ge::OpDescPtr &opDescPtr, int32_t &srcRank);
  static HcclResult GetDestRank(const ge::OpDescPtr &opDescPtr, int32_t &destRank);
  static HcclResult GetSrTag(const ge::OpDescPtr &opDescPtr, int32_t &srTag);
  static HcclResult TransformDataType(const ge::DataType geDataType, HcclDataType &hcclDataType);
  static HcclResult GetAllInputsTensorMemSize(const ge::OpDescPtr &opDescPtr, uint64_t &tensorSize);
  static HcclResult GetAllOutputsTensorMemSize(const ge::OpDescPtr &opDescPtr, uint64_t &tensorSize);
  static HcclResult GetAllInputsTensorOriginSize(const ge::OpDescPtr &opDescPtr, uint64_t &tensorSize);
  static HcclResult GetTensorMemSize(const ge::GeTensorDesc &tensorDesc, uint64_t &memSize);
  static HcclResult GetTensorOriginSize(const ge::GeTensorDesc &tensorDesc, uint64_t &size);
  static HcclResult GetPathFromEnv(char *getTunePath, std::string &fusionPath);
  static bool HcomOpIsSupportedBool(const std::string &opType);
  static HcclResult ConversionOpDataType(const ge::OpDescPtr &op, const std::string &opType, HcclDataType &dataType);
  static HcclResult GetAlltoAllCountMatrix(const ge::OpDescPtr &op, std::vector<int64_t> &sendCountMatrix);
  static HcclResult GetAlltoAllCountMatrix(ge::Node &node, std::vector<int64_t> &sendCountMatrix);
  static HcclResult GetAlltoAllCountsDispl(const ge::OpDescPtr &op, std::vector<int64_t> &sendCounts,
                                           std::vector<int64_t> &sendDispls, std::vector<int64_t> &recvCounts,
                                           std::vector<int64_t> &recvDispls);
  static HcclResult GetAlltoAllCountsDispl(ge::Node &node, std::vector<int64_t> &sendCounts,
                                           std::vector<int64_t> &sendDispls, std::vector<int64_t> &recvCounts,
                                           std::vector<int64_t> &recvDispls);
  static HcclResult GetReduceScatterVCountsDispl(ge::Node &node, std::vector<int64_t> &sendCounts,
                                                 std::vector<int64_t> &sendDispls, std::vector<int64_t> &recvCount);
  static HcclResult GetAllGatherVCountsDispl(ge::Node &node, std::vector<int64_t> &sendCount,
                                             std::vector<int64_t> &recvCounts, std::vector<int64_t> &recvDispls);
  static HcclResult GetAlltoAllDataType(const ge::OpDescPtr &op, HcclDataType &sendType, HcclDataType &recvType);
  static HcclResult GetConstInputAcrossGraph(const ge::GeTensor *&tensor, u32 index, ge::Node &node);
  static HcclResult GetVectorFromTensor(const ge::GeTensor *tensor, std::vector<int64_t> &vector);
  static HcclResult GetRankId(const int64_t &hcomComm, const string &sGroup, u32 &rankId);
  static HcclResult CheckAlltoAllvcRank(const ge::Node &node, const int64_t &hcomComm, const string &sGroup);
  static HcclResult GetGroupFromOpDesc(const ge::OpDescPtr &op, std::string &sGroup);
  static HcclResult GetTensorSize(const ge::GeTensorDesc &tensorDesc, int64_t &size);
  static HcclResult GetAllTensorSize(const ge::OpDescPtr &op, u32 tensorNum, std::vector<int64_t> &tensorSize);
  static HcclResult GetTaskNumFromCrackSize(const ge::Node &node, u32 tensorNum, u32 &taskNum);
  static HcclResult GetTensorCleanTaskNum(const ge::Node &node, const std::string &sCollectiveType, u32 &taskNum);
  static HcclResult GetTensorNum(const ge::Node &node, const std::string &sCollectiveType, u32 &tensorNum);
  static HcclResult GetAivCoreLimit(const ge::OpDescPtr &op, const std::string &sCollectiveType, u32 &aivCoreLimit);
  static HcclResult GetAccuracyCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                    HcclDataType dataType, u64 &count, u32 rankSize);
  static HcclResult CalcAllReduceCount(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                          u32 dataTypeSize, u64 &count);
  static HcclResult CalcCommonCount(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                        u32 dataTypeSize, u32 rankSize, u64 &count); 
  static HcclResult CalcBroadcastCount(const ge::OpDescPtr &op, u32 dataTypeSize, u64 &count);                                         
  static HcclResult GetCountFromOpDescSuperkernel(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                  HcclDataType dataType, u64 &count, u32 rankSize);
#ifndef HCOM_EXECUTOR
  static HcclResult CreateFusionConfigVersion(std::string &configVersion);
  static HcclResult GetFileNameFromPath(std::string &Path, std::string &fusionFile);
#endif
};
}  // namespace hccl

#endif  // end HCOM__OP_UTILS_H
