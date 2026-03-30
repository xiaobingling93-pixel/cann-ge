/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_KERNEL_BUILDER__BASEH
#define OPS_KERNEL_BUILDER__BASEH

#include "hccl/base.h"
#include "common/opskernel/ge_task_info.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "hcom_ops_stores.h"
#include "graph/node.h"
#include "proto/task.pb.h"
#include "hcom_log.h"
#include "op_hcom_comm.h"

namespace hccl {
// Ge适配的类
class HCCLOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  HCCLOpsKernelBuilder();
  ~HCCLOpsKernelBuilder() override;
  // initialize opsKernelInfoStore
  ge::Status Initialize(const map<string, string> &options) override;
  // close opsKernelInfoStore
  ge::Status Finalize() override;
  // memory allocation requirement
  virtual ge::Status CalcOpRunningParam(ge::Node &node) = 0;
  // generate taskinfo for op
  virtual ge::Status GenerateTask(const ge::Node &node, ge::RunContext &runContext,
                                  std::vector<domi::TaskDef> &taskDefList) = 0;

 protected:
  virtual HcclResult GetSupportedOP([[maybe_unused]] std::vector<std::string> &hcclSupportOp) const {
    return HCCL_SUCCESS;
  };
  virtual HcclResult SetOpMemAttr([[maybe_unused]] ge::Node &node, [[maybe_unused]] const std::string &sCollectiveType,
                                  [[maybe_unused]] const u64 &opMemSize) {
    return HCCL_SUCCESS;
  };
  virtual HcclResult SetOpAtomicInputIndex([[maybe_unused]] ge::Node &node, [[maybe_unused]] const std::string &sCollectiveType) {
    return HCCL_SUCCESS;
  };
  virtual HcclResult CheckSupportedOP(const std::string &sCollectiveType) const;
  HcclResult SetOpOutputMemSize(ge::Node &node, const std::string &sCollectiveType);
  HcclResult CalcHCCLOutputMemSize(const std::string &sCollectiveType, int64_t &memSize);
};
}  // namespace hccl
#endif  // __OPS_KERNEL_INFO_STORE__BASEH__
