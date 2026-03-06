/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <iomanip>
#include "runtime/rt.h"
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/debug/kernel_tracing.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/checker.h"
#include "engine/aicore/fe_rt2_common.h"

using namespace ge;

namespace gert {
namespace {
constexpr size_t kDSAWorkspaceAddrSize = 2U;
constexpr size_t kDSAArgsInputAddrSize = 4U;
constexpr size_t kDSAStatelessAddrSize = 5U;
constexpr size_t k32Bits = 32U;
constexpr uint32_t kMask32Bits = 0xFFFFFFFFU;

enum class DsaInfo {
  kSqeData,  //
  kWorkspaceAddr,
  kWorkspaceSize,
  kStream,
  kSeedType,
  kCountType,
  kSeedTypeStr,
  kCountTypeStr,
  kInput1Type,
  kInput1ValueStr,
  kInput2ValueStr,
  kInputNum,
  kOutputNum,
  kNum
};

ge::graphStatus DsaCoreUpdateSqeSeedCountResult(const KernelContext *context,
                                                std::vector<gert::GertTensorData *> &input_tensor_datas,
                                                rtStarsDsaSqe_t *sqe_data,
                                                const size_t *input_num) {
  auto seed_type = context->GetInputPointer<uint32_t>(static_cast<size_t>(DsaInfo::kSeedType));
  FE_ASSERT_NOTNULL(seed_type);
  auto count_type = context->GetInputPointer<uint32_t>(static_cast<size_t>(DsaInfo::kCountType));
  FE_ASSERT_NOTNULL(count_type);
  auto seed_type_string = context->GetInputValue<char *>(static_cast<size_t>(DsaInfo::kSeedTypeStr));
  FE_ASSERT_NOTNULL(seed_type_string);
  auto count_type_string = context->GetInputValue<char *>(static_cast<size_t>(DsaInfo::kCountTypeStr));
  FE_ASSERT_NOTNULL(count_type_string);
  auto output_num = context->GetInputPointer<size_t>(static_cast<size_t>(DsaInfo::kOutputNum));
  FE_ASSERT_NOTNULL(output_num);

  for (size_t input_i = 0; input_i < *input_num; ++input_i) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(DsaInfo::kNum) + input_i);
    FE_ASSERT_NOTNULL(tensor_data);
    input_tensor_datas.emplace_back(tensor_data);
  }
  uint64_t dev_output_addr = 0;
  for (size_t output_i = 0; output_i < *output_num; ++output_i) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(DsaInfo::kNum) +
                                                                  *input_num + output_i);
    FE_ASSERT_NOTNULL(tensor_data);
    dev_output_addr = PtrToValue(reinterpret_cast<TensorAddress>(tensor_data->GetAddr()));
    sqe_data->dsaCfgResultAddrLow = static_cast<uint32_t>(dev_output_addr & kMask32Bits);
    sqe_data->dsaCfgResultAddrHigh = static_cast<uint32_t>(dev_output_addr >> k32Bits);
  }
  uint64_t seed_value_or_addr = 0;
  if (*seed_type == 0U && input_tensor_datas.size() > 1U) {
    seed_value_or_addr = PtrToValue(reinterpret_cast<TensorAddress>(input_tensor_datas[1U]->GetAddr()));
  } else {
    seed_value_or_addr = *(PtrToPtr<char_t, uint64_t>(seed_type_string));
  }
  sqe_data->dsaCfgSeedLow = static_cast<uint32_t>(seed_value_or_addr & kMask32Bits);
  sqe_data->dsaCfgSeedHigh = static_cast<uint32_t>(seed_value_or_addr >> k32Bits);

  uint64_t random_count_value_or_addr = 0;
  if (*count_type == 0U && !input_tensor_datas.empty()) {
    random_count_value_or_addr = PtrToValue(reinterpret_cast<TensorAddress>(input_tensor_datas[0U]->GetAddr()));
  } else {
    random_count_value_or_addr = *(PtrToPtr<char_t, uint64_t>(count_type_string));
  }
  sqe_data->dsaCfgNumberLow = static_cast<uint32_t>(random_count_value_or_addr & kMask32Bits);
  sqe_data->dsaCfgNumberHigh = static_cast<uint32_t>(random_count_value_or_addr >> k32Bits);
  GELOGD("DsaCoreUpdateSqeArg dump seed_type:%u, count_type:%u, seed_type_string:%s, count_type_string:%s, "
         "input_num:%zu, output_num:%zu, dev_output_addr:0x%lx, seed_value_or_addr:0x%lx,"
         "random_count_value_or_addr:0x%lx",
         *seed_type, *count_type, seed_type_string, count_type_string, *input_num, *output_num,
         dev_output_addr, seed_value_or_addr, random_count_value_or_addr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaCoreUpdateSqeArg(KernelContext *context) {
  GELOGD("DsaCoreUpdateSqeArg Enter.");
  auto sqe_data = context->GetInputValue<rtStarsDsaSqe_t *>(static_cast<size_t>(DsaInfo::kSqeData));
  FE_ASSERT_NOTNULL(sqe_data);
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(DsaInfo::kWorkspaceAddr));
  FE_ASSERT_NOTNULL(workspace);
  auto workspace_sizes = context->GetInputPointer<TypedContinuousVector<int64_t>>
                         (static_cast<size_t>(DsaInfo::kWorkspaceSize));
  FE_ASSERT_NOTNULL(workspace_sizes);
  auto input_num = context->GetInputPointer<size_t>(static_cast<size_t>(DsaInfo::kInputNum));
  FE_ASSERT_NOTNULL(input_num);
  auto input1_type = context->GetInputPointer<uint32_t>(static_cast<size_t>(DsaInfo::kInput1Type));
  FE_ASSERT_NOTNULL(input1_type);
  auto input1_value_string = context->GetInputValue<char *>(static_cast<size_t>(DsaInfo::kInput1ValueStr));
  FE_ASSERT_NOTNULL(input1_value_string);
  auto input2_value_string = context->GetInputValue<char *>(static_cast<size_t>(DsaInfo::kInput2ValueStr));
  FE_ASSERT_NOTNULL(input2_value_string);
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(DsaInfo::kStream));
  FE_ASSERT_NOTNULL(stream);

  std::vector<gert::GertTensorData *> input_tensor_datas;
  auto ret = DsaCoreUpdateSqeSeedCountResult(context, input_tensor_datas, sqe_data, input_num);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGI("DsaCoreUpdateSqeSeedCountResult did not succeed.");
    return ge::GRAPH_PARAM_INVALID;
  }
  FE_ASSERT_TRUE(workspace->GetSize() > 0U);
  FE_ASSERT_TRUE(input_tensor_datas.size() > 0U);
  auto addrs_data = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
  const uint64_t workspace_philox_count_addr = PtrToValue(const_cast<TensorAddress>(addrs_data[0U]->GetAddr()));
  if (workspace->GetSize() == kDSAWorkspaceAddrSize) {
    sqe_data->dsaCfgStateAddrLow = static_cast<uint32_t>(workspace_philox_count_addr & kMask32Bits);
    sqe_data->dsaCfgStateAddrHigh = static_cast<uint32_t>(workspace_philox_count_addr >> k32Bits);
  } else {
    const uint64_t counter_addr =
        PtrToValue(reinterpret_cast<TensorAddress>(input_tensor_datas[input_tensor_datas.size() - 1U]->GetAddr()));
    sqe_data->dsaCfgStateAddrLow = static_cast<uint32_t>(counter_addr & kMask32Bits);
    sqe_data->dsaCfgStateAddrHigh = static_cast<uint32_t>(counter_addr >> k32Bits);
  }

  const uint64_t workspace_input_addr =
      PtrToValue(const_cast<TensorAddress>(addrs_data[workspace->GetSize() - 1U]->GetAddr()));
  sqe_data->dsaCfgParamAddrLow = static_cast<uint32_t>(workspace_input_addr & kMask32Bits);
  sqe_data->dsaCfgParamAddrHigh = static_cast<uint32_t>(workspace_input_addr >> k32Bits);

  GELOGD("DsaCoreUpdateSqeArg dump workspace_philox_count_addr:0x%lx, workspace_input_addr:0x%lx, input1_type:%u",
         workspace_philox_count_addr, workspace_input_addr, *input1_type);
  FE_ASSERT_TRUE(workspace_sizes->GetSize() > 0U);
  auto ws_sizes_data = workspace_sizes->GetData();
  if (*input1_type == 0U) {
    FE_ASSERT_TRUE(input_tensor_datas.size() > 2U);
    vector<uint64_t> input_addr{ PtrToValue(reinterpret_cast<TensorAddress>(input_tensor_datas[2U]->GetAddr())) };
    if ((*input_num == kDSAArgsInputAddrSize && workspace->GetSize() == kDSAWorkspaceAddrSize) ||
        *input_num == kDSAStatelessAddrSize) {
      FE_ASSERT_TRUE(input_tensor_datas.size() > 3U);
      input_addr.push_back(PtrToValue(reinterpret_cast<TensorAddress>(input_tensor_datas[3U]->GetAddr())));
    }
    FE_ASSERT_TRUE((sizeof(uint64_t) * input_addr.size()) <=
                   static_cast<uint64_t>(ws_sizes_data[workspace->GetSize() - 1U]));
    FE_ASSERT_RT_OK(aclrtMemcpyAsync(const_cast<TensorAddress>(addrs_data[workspace->GetSize() - 1U]->GetAddr()),
        static_cast<uint64_t>(ws_sizes_data[workspace->GetSize() - 1U]), input_addr.data(),
        sizeof(uint64_t) * input_addr.size(), ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
  } else {
    uint64_t input_data[2] = {0U, 0U};
    FE_ASSERT_EOK(memcpy_s(&input_data[0], sizeof(uint64_t), input1_value_string, strlen(input1_value_string)));
    if ((*input_num == kDSAArgsInputAddrSize && workspace->GetSize() == 2U) || *input_num == 5U) {
      FE_ASSERT_EOK(memcpy_s(&input_data[1], sizeof(uint64_t), input2_value_string, strlen(input2_value_string)));
    }
    FE_ASSERT_TRUE(static_cast<size_t>(ws_sizes_data[workspace->GetSize() - 1U]) >= sizeof(input_data));
    FE_ASSERT_RT_OK(aclrtMemcpyAsync(const_cast<TensorAddress>(addrs_data[workspace->GetSize() - 1U]->GetAddr()),
        static_cast<uint64_t>(ws_sizes_data[workspace->GetSize() - 1U]), input_data,
        sizeof(input_data), ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(UpdateSqeArg).RunFunc(DsaCoreUpdateSqeArg);
}  // namespace
}  // namespace gert
