/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "acl/acl_rt.h"
#include "stub/gert_runtime_stub.h"
#include "runtime/rt.h"

namespace gert {
class GertRuntimeStubUT : public testing::Test {};

rtArgsEx_t BuildArgsEx(void *args, uint32_t argsSize, uint16_t tilingAddrOffset, uint16_t tilingDataOffset) {
  rtArgsEx_t ret_args;
  ret_args.args = args;
  ret_args.argsSize = argsSize;
  ret_args.tilingDataOffset = tilingDataOffset;
  ret_args.tilingAddrOffset = tilingAddrOffset;
  return ret_args;
}

TEST_F(GertRuntimeStubUT, test_rt_kernel_launch_with_handle_success) {
  GertRuntimeStub runtime;
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;

  ASSERT_EQ(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
}

TEST_F(GertRuntimeStubUT, test_rt_kernel_launch_with_flag_success) {
  GertRuntimeStub runtime;
  void *stubFunc;
  uint8_t args_data[1024];
  stubFunc = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;

  ASSERT_EQ(rtKernelLaunchWithFlag(stubFunc, 0, &args, &smDesc, stream, 0), RT_ERROR_NONE);
}

TEST_F(GertRuntimeStubUT, test_aclrt_unuse_stream_res_in_current_thread_success) {
  GertRuntimeStub runtime;
  aclrtStream stream = nullptr;

  ASSERT_EQ(aclrtUnuseStreamResInCurrentThread(stream), ACL_SUCCESS);
}

TEST_F(GertRuntimeStubUT, test_stub_kernel_launch_with_handle_return_failed) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;

  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtKernelLaunchWithHandle(void *handle, const uint64_t, uint32_t blockDim, rtArgsEx_t *args,
                                       rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) {
      return -1;
    }
  };
  GertRuntimeStub runtime(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime));
  ASSERT_NE(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_handle_check) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;
  GertRuntimeStub runtime;

  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle);

  ASSERT_EQ(launch_args, nullptr);
  ASSERT_EQ(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
  launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle);
  ASSERT_NE(launch_args, nullptr);
  ASSERT_EQ(launch_args->GetArgsEx()->argsSize, 48);
  ASSERT_EQ(launch_args->GetArgsEx()->tilingAddrOffset, 18);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_handle_again_is_null) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;
  GertRuntimeStub runtime;

  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle);
  ASSERT_EQ(launch_args, nullptr);
  ASSERT_EQ(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
  launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle);
  ASSERT_NE(launch_args, nullptr);
  launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle);
  ASSERT_EQ(launch_args, nullptr);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_handle_check_filer_by_dev_func) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;
  GertRuntimeStub runtime;

  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle);

  ASSERT_EQ(launch_args, nullptr);
  ASSERT_EQ(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
  launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsBy(handle, 1024);
  ASSERT_NE(launch_args, nullptr);
  ASSERT_EQ(launch_args->GetArgsEx()->argsSize, 48);
  ASSERT_EQ(launch_args->GetArgsEx()->tilingAddrOffset, 18);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_flag_check_filer_by_dev_func) {
  void *handle, *devFunc, *kernelInfo;
  uint8_t args_data[1024];
  devFunc = (void *)"1024";
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;
  GertRuntimeStub runtime;
  ASSERT_EQ(rtKernelLaunchWithFlag(handle, 0, &args, &smDesc, stream, 1), RT_ERROR_NONE);
  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsByStubFunc(handle);
  ASSERT_NE(launch_args, nullptr);
  ASSERT_EQ(launch_args->GetArgsEx()->argsSize, 48);
  ASSERT_EQ(launch_args->GetArgsEx()->tilingAddrOffset, 18);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_flag_check_filer_by_stub_num) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;
  GertRuntimeStub runtime;

  ASSERT_EQ(rtGetFunctionByName("stub_func", &handle), RT_ERROR_NONE);
  ASSERT_EQ(rtKernelLaunchWithFlag(handle, 0, &args, &smDesc, stream, 1), RT_ERROR_NONE);
  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsByStubName("stub_func");
  ASSERT_NE(launch_args, nullptr);
  ASSERT_EQ(launch_args->GetArgsEx()->argsSize, 48);
  ASSERT_EQ(launch_args->GetArgsEx()->tilingAddrOffset, 18);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_flag_check_filer_by_binary) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;

  rtDevBinary_t binary;
  uint64_t bin_data = 100;
  binary.data = &bin_data;
  GertRuntimeStub runtime;

  ASSERT_EQ(rtRegisterAllKernel(&binary, &handle), RT_ERROR_NONE);
  ASSERT_EQ(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsByBinary(&binary);
  ASSERT_NE(launch_args, nullptr);
  ASSERT_EQ(launch_args->GetArgsEx()->argsSize, 48);
  ASSERT_EQ(launch_args->GetArgsEx()->tilingAddrOffset, 18);
}

TEST_F(GertRuntimeStubUT, test_kernel_launch_with_flag_check_filer_by_binary_and_dev_func) {
  void *handle, *kernelInfo;
  uint8_t args_data[1024];
  int64_t devFunc = 1024;
  handle = (void *)10;
  rtStream_t stream;
  rtArgsEx_t args = BuildArgsEx(args_data, 48, 18, 24);
  rtSmDesc_t smDesc;

  rtDevBinary_t binary;
  uint64_t bin_data = 100;
  binary.data = &bin_data;
  GertRuntimeStub runtime;

  ASSERT_EQ(rtRegisterAllKernel(&binary, &handle), RT_ERROR_NONE);
  ASSERT_EQ(rtKernelLaunchWithHandle(handle, devFunc, 0, &args, &smDesc, stream, kernelInfo), RT_ERROR_NONE);
  auto launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsByBinary(&binary, 256);
  ASSERT_EQ(launch_args, nullptr);
  launch_args = runtime.GetRtsRuntimeStub().PopLaunchArgsByBinary(&binary, 1024);
  ASSERT_NE(launch_args, nullptr);
  ASSERT_EQ(launch_args->GetArgsEx()->argsSize, 48);
  ASSERT_EQ(launch_args->GetArgsEx()->tilingAddrOffset, 18);
}
}  // namespace gert
