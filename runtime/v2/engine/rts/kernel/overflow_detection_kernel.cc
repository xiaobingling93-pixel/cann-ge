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
#include "runtime/mem.h"
#include "register/kernel_registry_impl.h"
#include "common/checker.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "common/runtime_api_wrapper.h"
#include "acl/acl_rt.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kNpuStatusStatusHostAddrSize = 64UL;
constexpr size_t kGetFloatDevArgsIdx = 0UL;
constexpr size_t kGetFloatHostArgsIdx = 1UL;
constexpr size_t kGetFloatStreamIdx = 2UL;
}  // namespace

struct NpuGetFloatStatusArgsHolder {
  void *output_holder{nullptr};
  size_t output_size{kNpuStatusStatusHostAddrSize};
};

ge::graphStatus NpuGetFloatStatus(KernelContext *const context) {
  auto args_dev = context->GetInputValue<gert::GertTensorData *>(kGetFloatDevArgsIdx);
  auto args_holder = context->GetInputValue<NpuGetFloatStatusArgsHolder *>(kGetFloatHostArgsIdx);
  auto stream = context->GetInputValue<rtStream_t>(kGetFloatStreamIdx);

  GE_ASSERT_NOTNULL(args_dev);
  GE_ASSERT_NOTNULL(args_holder);
  GE_ASSERT_RT_OK(aclrtMemcpyAsync(args_dev->GetAddr(), sizeof(void *), &args_holder->output_holder,
      sizeof(void *), ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
  GE_ASSERT_RT_OK(ge::rtNpuGetFloatStatus(args_dev->GetAddr(), args_holder->output_size, 0U, stream));

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(NpuGetFloatStatus).RunFunc(NpuGetFloatStatus);

ge::graphStatus NpuGetFloatStatusArgs(KernelContext *const context) {
  auto output_tensor = context->GetInputValue<gert::GertTensorData *>(0UL);
  auto args_holder = context->GetOutputPointer<NpuGetFloatStatusArgsHolder>(0UL);
  GE_ASSERT_NOTNULL(output_tensor);
  GE_ASSERT_NOTNULL(args_holder);
  args_holder->output_holder = output_tensor->GetAddr();

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateNpuGetFloatStatusArgs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto *args_holder = new (std::nothrow) NpuGetFloatStatusArgsHolder();
  GE_ASSERT_NOTNULL(args_holder);
  auto chain = context->GetOutput(0UL);
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(args_holder);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(NpuGetFloatStatusArgs).RunFunc(NpuGetFloatStatusArgs).OutputsCreator(CreateNpuGetFloatStatusArgs);

ge::graphStatus NpuClearFloatStatus(KernelContext *const context) {
  auto stream = context->GetInputValue<rtStream_t>(0UL);
  GE_ASSERT_RT_OK(ge::rtNpuClearFloatStatus(0UL, stream));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(NpuClearFloatStatus).RunFunc(NpuClearFloatStatus);

ge::graphStatus NpuGetFloatDebugStatus(KernelContext *const context) {
  auto args_dev = context->GetInputValue<gert::GertTensorData *>(kGetFloatDevArgsIdx);
  auto args_holder = context->GetInputValue<NpuGetFloatStatusArgsHolder *>(kGetFloatHostArgsIdx);
  auto stream = context->GetInputValue<rtStream_t>(kGetFloatStreamIdx);

  GE_ASSERT_NOTNULL(args_dev);
  GE_ASSERT_NOTNULL(args_holder);
  GE_ASSERT_RT_OK(aclrtMemcpyAsync(args_dev->GetAddr(), sizeof(void *), &args_holder->output_holder,
      sizeof(void *), ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
  GE_ASSERT_RT_OK(ge::rtNpuGetFloatDebugStatus(args_dev->GetAddr(), args_holder->output_size, 0U, stream));

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(NpuGetFloatDebugStatus).RunFunc(NpuGetFloatDebugStatus);

ge::graphStatus NpuClearFloatDebugStatus(KernelContext *const context) {
  auto stream = context->GetInputValue<rtStream_t>(0UL);
  GE_ASSERT_RT_OK(ge::rtNpuClearFloatDebugStatus(0UL, stream));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(NpuClearFloatDebugStatus).RunFunc(NpuClearFloatDebugStatus);

}  // namespace kernel
}  // namespace gert
