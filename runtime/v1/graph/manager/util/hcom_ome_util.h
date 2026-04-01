/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_
#define GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/opskernel/ge_task_info.h"
#include "framework/common/string_util.h"
#include "graph/op_desc.h"
#include "hccl/hcom.h"
#include "engines/hccl_engine/inc/hcom_executor.h"
#include "proto/task.pb.h"

namespace ge {
extern const std::map<int64_t, HcclDataType> kConstOpHcclDataType;


class HcomOmeUtil {
 public:
  /// @ingroup domi_ome
  /// @brief GetHcclDataType
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcclDataType(const ge::ConstOpDescPtr &op_desc,
                                std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  /// @ingroup domi_ome
  /// @brief GetHcclTypeSize
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcclTypeSize(const HcclDataType data_type, int32_t &size);

  /// @ingroup domi_ome
  /// @brief GetHcclCount
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcclCount(const ge::ConstOpDescPtr &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  /// @ingroup domi_ome
  /// @brief GetHcclOperationType
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcclOperationType(const ge::ConstOpDescPtr &op_desc, HcclReduceOp &op_type);

  /// @ingroup domi_ome
  /// @brief GetHcclRootId
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcclRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id);

  /// @ingroup domi_ome
  /// @brief GetAllRootId
  /// @return SUCCESS
  /// @return FAIL
  static Status GetAllRootId(const ge::ConstOpDescPtr &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  /// @ingroup domi_ome
  /// @brief check the op_type whether is hcom operator or not
  /// @return true
  /// @return false
  static bool IsHCOMOp(const std::string &op_type);

  /// @ingroup domi_ome
  /// @brief check the op_type whether is horovod operator or not
  /// @return true
  /// @return false
  static bool IsHorovodOp(const std::string &op_type);

  /// @ingroup domi_ome
  /// @brief GetHcclType
  /// @return void
  static void GetHcclType(const domi::TaskDef &task_def, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  /// @ingroup domi_ome
  /// @brief CheckKernelHcclInfo
  /// @return SUCCESS
  /// @return FAIL
  static Status CheckKernelHcclInfo(const ge::ConstOpDescPtr &op_desc,
                                    const std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  /// @ingroup domi_ome
  /// @brief GetHorovodInputs
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHorovodInputs(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  /// @ingroup domi_ome
  /// @brief GetHcomCount
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcomCount(const ge::ConstOpDescPtr &op_desc, const HcclDataType data_type, const bool is_allgather,
                             int64_t &count);
  static Status GetAlignedTensorSize(const ge::GeTensorDesc &tensor_desc, const int32_t align_size,
                                     int64_t &output_size);
  /// @ingroup domi_ome
  /// @brief GetHcomP2pCount
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcomP2pCount(const ge::GeTensorDesc &tensor_desc,
                                const HcclDataType data_type,
                                const HcomOperationType p2p_type,
                                int64_t &count);

  /// @ingroup domi_ome
  /// @brief GetHcomCount
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHcclDataType(const ge::DataType src_data_type, HcclDataType &hccl_data_type);
 private:
  /// @ingroup domi_ome
  /// @brief GetHorovodCount
  /// @return SUCCESS
  /// @return FAIL
  static Status GetHorovodCount(const ge::ConstOpDescPtr &op_desc,
                                std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_UTIL_HCOM_UTIL_H_
