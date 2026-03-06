/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_AICPU_EXT_INFO_H_
#define GE_HYBRID_AICPU_EXT_INFO_H_

#include "fwk_adpt_struct.h"
#include "ge/ge_api_error_codes.h"
#include "aicpu_engine_struct.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "acl/acl_rt.h"

namespace ge {
namespace hybrid {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AsyncWaitInfo = aicpu::FWKAdapter::AsyncWait;
using WorkSpaceInfo = aicpu::FWKAdapter::WorkSpaceInfo;
using AicpuSessionInfo = SessionInfo;

class AicpuExtInfoHandler {
 public:
  AicpuExtInfoHandler(const std::string &node_name, const uint32_t input_num, const uint32_t output_num,
                      const UnknowShapeOpType unknown_type)
      : node_name_(node_name), input_num_(input_num), output_num_(output_num), unknown_type_(unknown_type) {
  }

  ~AicpuExtInfoHandler() = default;

  uint8_t *GetExtInfo() const {
    return ext_info_.get();
  }
  size_t GetExtInfoLen() const {
    return ext_info_len_;
  }

  Status Parse(const std::string &ext_info);

  Status UpdateInputShapeAndType(const uint32_t input_index, const GeTensorDesc &input_desc);

  Status UpdateOutputShapeAndType(const uint32_t output_index, const GeTensorDesc &output_desc);

  Status UpdateInputShapeAndType(const size_t input_index, const GeTensorDesc &input_desc,
                                 const std::vector<int64_t> &input_dims);
  Status UpdateOutputShapeAndType(const size_t output_index, const GeTensorDesc &output_desc,
                                  const std::vector<int64_t> &output_dims);

  Status UpdateSessionInfo(const uint64_t session_id, const uint64_t kernel_id, const bool sess_flag) const;

  Status UpdateSessionInfoId(const uint64_t session_id) const;

  Status UpdateWorkSpaceInfo(const uint64_t workspace_length, const uint64_t workspace_addr) const;

  Status UpdateExecuteMode(const bool flag);

  Status UpdateEventId(const uint32_t event_id) const;

  Status GetOutputShapeAndType(const uint32_t output_index, GeShape &shape, DataType &data_type);

  int32_t GetDeployTypeFlag() const {
    return deploy_type_flag_;
  };
  uint32_t GeQosLevelFlag() const {
    return qos_level_flag_;
  };
  rtMemType_t GetMemType() const {
    return (deploy_type_flag_ == RT_KERNEL_HOST_ONLY) ? RT_MEMORY_HOST_SVM : RT_MEMORY_HBM;
  };
  aclrtMemcpyKind GetMemcpyKind() const {
    return (deploy_type_flag_ == RT_KERNEL_HOST_ONLY) ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
  };

  static uint64_t GenerateKernelId();

 private:
  Status ParseExtShapeType(AicpuExtInfo &aicpu_ext_info) const;
  Status ParseExtInputShape(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtOutputShape(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtSessionInfo(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtBitMap(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtUpdateAddr(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtTopicType(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtAsyncWait(AicpuExtInfo &aicpu_ext_info);
  Status ParseExtWorkSpaceInfo(AicpuExtInfo &aicpu_ext_info);

  Status UpdateShapeAndType(const std::vector<int64_t> &dims, const DataType data_type,
                            AicpuShapeAndType &shape_and_type) const;

  static Status UpdateShapeAndType(const GeShape &shape, const DataType data_type,
                                   AicpuShapeAndType *const shape_and_type);

  static void GetShapeAndType(const AicpuShapeAndType *const shape_and_type, GeShape &shape, DataType &data_type);

 private:
  static int32_t TopicTypeToRtsFlag(const int32_t topic_type);

  const std::string node_name_;
  const uint32_t input_num_;
  const uint32_t output_num_;
  UnknowShapeOpType unknown_type_;
  AicpuSessionInfo *session_info_ = nullptr;
  AsyncWaitInfo *async_wait_ = nullptr;
  WorkSpaceInfo *workspace_info_ = nullptr;
  uint64_t *bit_map_ = nullptr;
  uint32_t *update_addr_ = nullptr;
  int32_t deploy_type_flag_ = 0;  // default is device only
  uint32_t qos_level_flag_ = 0U;

  std::unique_ptr<uint8_t[]> ext_info_;
  size_t ext_info_len_ = 0U;
  std::vector<AicpuShapeAndType *> input_shape_and_type_;
  std::vector<AicpuShapeAndType *> output_shape_and_type_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_AICPU_EXT_INFO_H_
