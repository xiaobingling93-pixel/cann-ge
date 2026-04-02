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
#include "runtime/kernel.h"
#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "core/debug/kernel_tracing.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/checker.h"
#include "common/runtime_api_wrapper.h"
#include "engine/aicore/fe_rt2_common.h"


using namespace ge;

namespace gert {
namespace kernel {
namespace {
enum class StarsLaunchCommon { kAddress, kLen, kStream };

std::vector<std::string> PrintStarsLaunchArgs(const KernelContext *context) {
  auto address = context->GetInputValue<void *>(static_cast<size_t>(StarsLaunchCommon::kAddress));
  auto len = context->GetInputValue<uint32_t>(static_cast<size_t>(StarsLaunchCommon::kLen));
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(StarsLaunchCommon::kStream));

  std::stringstream ss;
  ss << "Stars launch function arguments: "
     << "address " << address << ", len " << len << ", stream " << stream;
  return {ss.str()};
}

ge::graphStatus StarsTaskLaunchKernel(KernelContext *context) {
  auto address = context->GetInputValue<void *>(static_cast<size_t>(StarsLaunchCommon::kAddress));
  auto len = context->GetInputValue<uint32_t>(static_cast<size_t>(StarsLaunchCommon::kLen));
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(StarsLaunchCommon::kStream));

  FE_ASSERT_NOTNULL(address);
  FE_ASSERT_NOTNULL(stream);
  FE_ASSERT_RT_OK(ge::rtStarsTaskLaunch(address, len, stream));

  return SUCCESS;
}
REGISTER_KERNEL(StarsTaskLaunchKernel).RunFunc(StarsTaskLaunchKernel).TracePrinter(PrintStarsLaunchArgs);
}  // namespace
}  // namespace kernel
}  // namespace gert
