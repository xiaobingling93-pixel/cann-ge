/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include <functional>
#include "graph/tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"

#include "ops_kernel_builder_base.h"

namespace hccl {
HCCLOpsKernelBuilder::HCCLOpsKernelBuilder() {}

HCCLOpsKernelBuilder::~HCCLOpsKernelBuilder() {}

// close opsKernelInfoStore
ge::Status HCCLOpsKernelBuilder::Finalize() {
  // 直接返回, 有单独的销毁接口
  return ge::SUCCESS;
}

HcclResult HCCLOpsKernelBuilder::CheckSupportedOP(const std::string &sCollectiveType) const {
  std::vector<std::string>::const_iterator it;
  std::vector<std::string> hcclSupportOp;
  if (GetSupportedOP(hcclSupportOp) == HCCL_SUCCESS) {
    it = std::find(hcclSupportOp.begin(), hcclSupportOp.end(), sCollectiveType);
    return (it != hcclSupportOp.end()) ? HCCL_SUCCESS : HCCL_E_PARA;
  } else {
    return HCCL_E_PARA;
  }
}

// initialize opsKernelInfoStore
ge::Status HCCLOpsKernelBuilder::Initialize([[maybe_unused]] const map<string, string> &options) {
  // 直接返回, 有单独的初始化接口
  return ge::SUCCESS;
}

HcclResult HCCLOpsKernelBuilder::SetOpOutputMemSize(ge::Node &node, const std::string &sCollectiveType) {
  ge::OpDescPtr op = node.GetOpDesc();

  for (u32 i = 0; i < op->GetOutputsSize(); i++) {
    int64_t memSize = 0;
    ge::GeTensorDesc outputTensor = op->GetOutputDesc(i);
    ge::GeShape outputShape = outputTensor.GetShape();
    ge::Format format = outputTensor.GetFormat();
    ge::DataType dataType = outputTensor.GetDataType();
    // 获取内存大小
    bool bErr = (ge::GRAPH_SUCCESS != ge::TensorUtils::CalcTensorMemSize(outputShape, format, dataType, memSize));
    CHK_PRT_RET(bErr,
                HCCL_ERROR("[SetOp][OutputMemSize]In get output mem size, error outputSize because no"
                           "know shape, Format[%d], dataType[%d], outputSize[%lld], index[%u]",
                           format, dataType, memSize, i),
                HCCL_E_PARA);

    if (memSize == -1) {  // memsize 为-1 时，表示输入的shape不正确
      HCCL_ERROR(
          "[SetOp][OutputMemSize]In get output mem size, error outputSize because unknow shape,"
          "Format[%d], dataType[%d], outputSize[%lld], index[%u]",
          format, dataType, memSize, i);
      return HCCL_E_PARA;
    }

    // 根据 规则重新计算内存大小
    CHK_RET(CalcHCCLOutputMemSize(sCollectiveType, memSize));

    // 将内存大小重新传给上层
    ge::TensorUtils::SetSize(outputTensor, memSize);

    // 更新output Tensor
    if (op->UpdateOutputDesc(i, outputTensor)) {
      HCCL_ERROR(
          "[SetOp][OutputMemSize]In get output mem size, update output desc error,"
          "Format[%d], dataType[%d], outputSize[%lld], index[%u]",
          format, dataType, memSize, i);
      return HCCL_E_PARA;
    }
    HCCL_INFO("In set output MemSize, sCollectiveType[%s], opMemSize[%lld]", sCollectiveType.c_str(), memSize);
  }
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelBuilder::CalcHCCLOutputMemSize(const std::string &sCollectiveType, int64_t &memSize) {
  HCCL_DEBUG("[HCCLOpsKernelBuilder][CalcHCCLOutputMemSize] sCollectiveType[%s], memSize[%lld]",
             sCollectiveType.c_str(), memSize);
  const u32 MEMORY_ALIGN_RATIO = 2;  // GE要求内存需要32KB对齐后，再外加32KB. out = (in + 2 * 32 - 1) / 32 * 32
  const u32 MEMORY_ALIGN_SIZE = 32;  // GE要求内存需要32KB对齐后，再外加32KB. out = (in + 2 * 32 - 1) / 32 * 32
  // GE要求内存需要32KB对齐后，再外加32KB
  memSize = ((memSize + MEMORY_ALIGN_RATIO * MEMORY_ALIGN_SIZE - 1) / MEMORY_ALIGN_SIZE) * MEMORY_ALIGN_SIZE;
  return HCCL_SUCCESS;
}
}  // namespace hccl
