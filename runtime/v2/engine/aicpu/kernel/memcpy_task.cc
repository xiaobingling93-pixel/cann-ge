/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel/memory/mem_block.h"
#include "aicpu_engine_struct.h"
#include "runtime/mem.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"
#include "runtime/kernel.h"
#include "aicpu_task_struct.h"
#include "framework/common/taskdown_common.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/debug/kernel_tracing.h"
#include "framework/common/ge_types.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph_metadef/common/ge_common/util.h"
#include "acl/acl_rt.h"

namespace gert {
namespace kernel {
namespace {
enum class CopyInputs {
  kOutputNum,
  kReleaseFlag,
  kDataSize,
  kSrcAddr,
  kDstAddr
};
} // namespace
ge::graphStatus PrepareCopyInputs(KernelContext *context) {
  auto output_num = context->GetInputValue<size_t>(static_cast<size_t>(CopyInputs::kOutputNum));
  auto stream = context->GetInputValue<aclrtStream>(context->GetInputNum() - 1);
  std::vector<uint64_t> copy_input_release_flag;
  std::vector<uint64_t> copy_input_data_size;
  std::vector<uint64_t> copy_input_src;
  std::vector<uint64_t> copy_input_dst;

  size_t idx_start = static_cast<size_t>(CopyInputs::kDstAddr) + 1U;
  for (size_t i = 0U; i < output_num; ++i) {
    auto summary = context->GetInputPointer<aicpu::FWKAdapter::ResultSummary>(idx_start + i);
    auto output = context->GetInputValue<gert::GertTensorData *>(idx_start + output_num + i);
    // 2 means order is summary, output_addr, shape_buffer_addr
    auto shape_buffer = context->GetInputValue<gert::GertTensorData *>(idx_start + 2U * output_num + i);
    GE_ASSERT_NOTNULL(summary);
    GE_CHECK_NOTNULL(output);
    GE_CHECK_NOTNULL(shape_buffer);
    GELOGD("PrepareCopyInputs out[%zu], shape data=0x%lx, shape data size=%lu, raw data=0x%lx, raw data size=%lu.",
           i, summary->shape_data_ptr, summary->shape_data_size, summary->raw_data_ptr, summary->raw_data_size);

    copy_input_release_flag.emplace_back(ge::kReleaseFlag);
    copy_input_data_size.emplace_back(summary->raw_data_size);
    copy_input_src.emplace_back(summary->raw_data_ptr);
    copy_input_dst.emplace_back(ge::PtrToValue(output->GetAddr()));

    copy_input_release_flag.emplace_back(ge::kReleaseFlag);
    copy_input_data_size.emplace_back(summary->shape_data_size);
    copy_input_src.emplace_back(summary->shape_data_ptr);
    copy_input_dst.emplace_back(ge::PtrToValue(shape_buffer->GetAddr()));
  }
  auto release_flag = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(CopyInputs::kReleaseFlag));
  auto data_size = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(CopyInputs::kDataSize));
  auto src_addr = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(CopyInputs::kSrcAddr));
  auto dst_addr = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(CopyInputs::kDstAddr));
  GE_ASSERT_NOTNULL(release_flag);
  GE_ASSERT_NOTNULL(data_size);
  GE_ASSERT_NOTNULL(src_addr);
  GE_ASSERT_NOTNULL(dst_addr);
  GE_ASSERT_NOTNULL(release_flag->GetAddr());
  GE_ASSERT_NOTNULL(data_size->GetAddr());
  GE_ASSERT_NOTNULL(src_addr->GetAddr());
  GE_ASSERT_NOTNULL(dst_addr->GetAddr());

  // copy task need copy all output_data and output_shape, len is 2 * output_num
  const size_t copy_input_buf_len = output_num * 2U * sizeof(void *);
  GE_ASSERT_RT_OK(rtMemcpyAsync(release_flag->GetAddr(), copy_input_buf_len, &copy_input_release_flag[0U],
                                copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  GE_ASSERT_RT_OK(rtMemcpyAsync(data_size->GetAddr(), copy_input_buf_len, &copy_input_data_size[0U], copy_input_buf_len,
                                RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  GE_ASSERT_RT_OK(rtMemcpyAsync(src_addr->GetAddr(), copy_input_buf_len, &copy_input_src[0U], copy_input_buf_len,
                                RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  GE_ASSERT_RT_OK(rtMemcpyAsync(dst_addr->GetAddr(), copy_input_buf_len, &copy_input_dst[0U], copy_input_buf_len,
                                RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(PrepareCopyInputs).RunFunc(PrepareCopyInputs);

ge::graphStatus GetOutputShapeFromHbmBuffer(KernelContext *context) {
  size_t output_num = context->GetOutputNum();
  for (size_t i = 0U; i < output_num; ++i) {
    auto summary = context->GetInputPointer<aicpu::FWKAdapter::ResultSummary>(i);
    auto output_shape = context->GetOutputPointer<gert::StorageShape>(i);
    GE_ASSERT_NOTNULL(summary);
    GE_ASSERT_NOTNULL(output_shape);
    output_shape->MutableOriginShape().SetDimNum(0U);
    output_shape->MutableStorageShape().SetDimNum(0U);

    std::vector<int64_t> shape_dims;
    if (summary->shape_data_size > 0U) {
      GE_ASSERT_TRUE((summary->shape_data_size % sizeof(int64_t)) == 0U);
      const size_t dim_num = static_cast<size_t>(summary->shape_data_size) / sizeof(int64_t);
      GELOGD("Get output[%zu]th dim_num = %zu.", i, dim_num);

      auto shape_buffer = context->GetInputValue<gert::GertTensorData *>(output_num + i);
      GE_ASSERT_NOTNULL(shape_buffer);
      GE_ASSERT_NOTNULL(shape_buffer->GetAddr());
      vector<int64_t> host_shape_buffer(dim_num);
      GE_ASSERT_RT_OK(rtMemcpy(&host_shape_buffer[0], summary->shape_data_size, shape_buffer->GetAddr(),
                               summary->shape_data_size, RT_MEMCPY_DEVICE_TO_HOST));

      for (size_t dim_idx = 0U; dim_idx < dim_num; ++dim_idx) {
        output_shape->MutableOriginShape().AppendDim(host_shape_buffer[dim_idx]);
        output_shape->MutableStorageShape().AppendDim(host_shape_buffer[dim_idx]);
        GELOGD("Get output[%zu]th dim[%zu] = %ld.", i, dim_idx, host_shape_buffer[dim_idx]);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOutputsForHbmBuffer(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  GELOGD("Output number = %zd.", context->GetOutputNum());
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto av_holder = context->GetOutput(i);
    auto tensor = ge::MakeUnique<Tensor>();
    GE_ASSERT_NOTNULL(av_holder);
    GE_ASSERT_NOTNULL(tensor);
    av_holder->SetWithDefaultDeleter(tensor.release());
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetOutputShapeFromHbmBuffer)
    .RunFunc(GetOutputShapeFromHbmBuffer)
    .OutputsCreator(CreateOutputsForHbmBuffer);

ge::graphStatus GetHostSummary(KernelContext *context) {
  auto output_num = context->GetOutputNum();
  for (size_t i = 0U; i < output_num; ++i) {
    auto device_summary = context->GetInputValue<gert::GertTensorData *>(i);
    auto summary = context->GetOutputPointer<aicpu::FWKAdapter::ResultSummary>(i);
    GE_ASSERT_NOTNULL(device_summary);
    GE_ASSERT_NOTNULL(device_summary->GetAddr());
    GE_ASSERT_NOTNULL(summary);
    GE_ASSERT_RT_OK(rtMemcpy(summary, sizeof(aicpu::FWKAdapter::ResultSummary), device_summary->GetAddr(),
                             sizeof(aicpu::FWKAdapter::ResultSummary), RT_MEMCPY_DEVICE_TO_HOST));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOutputsForHostSummary(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto output_num = context->GetOutputNum();
  for (size_t i = 0U; i < output_num; i++) {
    auto av_summary = context->GetOutput(i);
    auto summary = ge::MakeUnique<aicpu::FWKAdapter::ResultSummary>();
    GE_ASSERT_NOTNULL(av_summary);
    GE_ASSERT_NOTNULL(summary);
    av_summary->SetWithDefaultDeleter(summary.release());
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetHostSummary).RunFunc(GetHostSummary).OutputsCreator(CreateOutputsForHostSummary);

ge::graphStatus GetSummaryDataSizes(KernelContext *context) {
  constexpr uint64_t kAlignBytes = 32U;
  auto output_num = context->GetOutputNum();
  for (size_t i = 0U; i < output_num; ++i) {
    auto summary = context->GetInputPointer<aicpu::FWKAdapter::ResultSummary>(i);
    auto data_size = context->GetOutputPointer<uint64_t>(i);
    GE_ASSERT_NOTNULL(summary);
    GE_ASSERT_NOTNULL(data_size);
    *data_size = ge::RoundUp(summary->raw_data_size, kAlignBytes) + kAlignBytes;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetSummaryDataSizes).RunFunc(GetSummaryDataSizes);

ge::graphStatus GetSummaryShapeSizes(KernelContext *context) {
  auto output_num = context->GetOutputNum();
  for (size_t i = 0U; i < output_num; ++i) {
    auto summary = context->GetInputPointer<aicpu::FWKAdapter::ResultSummary>(i);
    auto shape_size = context->GetOutputPointer<uint64_t>(i);
    GE_ASSERT_NOTNULL(summary);
    GE_ASSERT_NOTNULL(shape_size);
    *shape_size = summary->shape_data_size;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetSummaryShapeSizes).RunFunc(GetSummaryShapeSizes);
}  // namespace kernel
}  // namespace gert
