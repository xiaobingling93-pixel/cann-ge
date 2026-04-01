/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "graph/ge_error_codes.h"
#include "runtime/mem.h"
#include "register/kernel_registry_impl.h"
#include "common/checker.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "common/runtime_api_wrapper.h"
#include "graph/utils/tensor_utils.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/node.h"
#include "acl/acl_rt.h"

namespace gert {
namespace kernel {
ge::graphStatus InitCmoArgs(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateCmoLaunchArgs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  const uint32_t op_code = context->GetInputValue<uint32_t>(0UL);
  auto launch_result = context->GetOutput(0UL);
  GE_ASSERT_NOTNULL(launch_result);
  auto *cmo_task_info = new (std::nothrow) rtCmoTaskInfo_t();
  GE_ASSERT_NOTNULL(cmo_task_info);
  cmo_task_info->opCode = static_cast<uint16_t>(op_code);
  launch_result->SetWithDefaultDeleter(cmo_task_info);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(InitCmoArgs).RunFunc(InitCmoArgs).OutputsCreator(CreateCmoLaunchArgs);

ge::graphStatus UpdatePrefetchTaskInfo(KernelContext *const context) {
  auto cmo_args = context->MutableInputPointer<rtCmoTaskInfo_t>(0UL);
  GE_ASSERT_NOTNULL(cmo_args);
  auto tensor_data = context->GetInputPointer<GertTensorData>(1UL);
  GE_ASSERT_NOTNULL(tensor_data);

  auto storage_shape = context->GetInputPointer<StorageShape>(2UL);
  GE_ASSERT_NOTNULL(storage_shape);
  const auto shape_size = storage_shape->GetStorageShape().GetShapeSize();
  const auto dtype = context->GetInputValue<ge::DataType>(3UL);
  const auto max_len = context->GetInputValue<uint32_t>(4UL);
  const int64_t *offset = context->GetInputPointer<int64_t>(5UL);
  GE_ASSERT_NOTNULL(offset);
  int64_t tensor_len = ge::GetSizeInBytes(shape_size, dtype);
  GE_ASSERT_TRUE(tensor_len > 0);
  const bool is_offset_valid = (*offset >= 0) && (*offset < tensor_len);
  GE_ASSERT_TRUE(is_offset_valid, "The offset [%ld] should be within the range of [0, %ld).", *offset, tensor_len);
  tensor_len -= (*offset);
  cmo_args->lengthInner = std::min(static_cast<uint32_t>(tensor_len), max_len);
  cmo_args->sourceAddr = ge::PtrToValue(tensor_data->GetAddr()) + static_cast<uint64_t>(*offset);

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(UpdatePrefetchTaskInfo).RunFunc(UpdatePrefetchTaskInfo);

ge::graphStatus LaunchCmoTask(KernelContext *const context) {
  auto cmo_args = context->GetInputPointer<rtCmoTaskInfo_t>(0UL);
  GE_ASSERT_NOTNULL(cmo_args);
  auto stream = context->GetInputValue<aclrtStream>(1UL);
  GE_ASSERT_RT_OK(ge::rtCmoTaskLaunch(cmo_args, stream, 0U));
  return ge::GRAPH_SUCCESS;
}

static std::vector<std::string> LaunchCmoTracer(const KernelContext *context) {
  std::stringstream ss;
  auto cmo_args = context->GetInputPointer<rtCmoTaskInfo_t>(0UL);
  GE_ASSERT_NOTNULL(cmo_args);
  auto stream = context->GetInputValue<aclrtStream>(1UL);
  ss << "Launch cmo task with inner_len:" << cmo_args->lengthInner << ", src:" << cmo_args->sourceAddr
     << " on stream:" << stream;
  return {ss.str()};
}

REGISTER_KERNEL(LaunchCmoTask).RunFunc(LaunchCmoTask).TracePrinter(LaunchCmoTracer);
}  // namespace kernel
}  // namespace gert
