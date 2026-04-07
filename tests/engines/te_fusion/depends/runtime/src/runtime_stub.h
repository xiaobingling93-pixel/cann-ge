/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_LLT_RUNTIME_STUB_H
#define __INC_LLT_RUNTIME_STUB_H

#include <vector>
#include "runtime/rt.h"
#include <memory>

namespace ge {
class RuntimeStub {
 public:
  virtual ~RuntimeStub() = default;

  static RuntimeStub* GetInstance();

  static void SetInstance(const std::shared_ptr<RuntimeStub> &instance) {
    instance_ = instance;
  }

  static void Install(RuntimeStub*);
  static void UnInstall(RuntimeStub*);

  static void Reset() {
    instance_.reset();
  }

  virtual rtError_t rtKernelLaunchEx(void *args, uint32_t args_size, uint32_t flags, rtStream_t stream) {
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtKernelLaunch(const void *stub_func,
                                   uint32_t block_dim,
                                   void *args,
                                   uint32_t args_size,
                                   rtSmDesc_t *sm_desc,
                                   rtStream_t stream) {
    return RT_ERROR_NONE;
  }
  virtual rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                   rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag) {
    return RT_ERROR_NONE;
  }
  virtual rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                             rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags,
                                             const rtTaskCfgInfo_t *cfgInfo) {
    return RT_ERROR_NONE;
  }
  virtual rtError_t rtKernelGetAddrAndPrefCntV2(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                                const uint32_t flag, rtKernelDetailInfo_t *kernelInfo) {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  }
  virtual rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t blockDim,
                                                const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                                uint32_t flags) {
    return RT_ERROR_NONE;
  }
  virtual rtError_t rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim,
                                                  const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                                  uint32_t flags) {
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtKernelLaunchWithHandle(void *handle, uint64_t devFunc, uint32_t blockDim, rtArgsEx_t *args,
                                     rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) {
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtGetIsHeterogenous(int32_t *heterogeneous) {
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtGetDeviceCount(int32_t *count) {
    *count = 1;
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtGetFunctionByName(const char *stub_name, void **stub_func) {
    *(char **)stub_func = (char *)("func");
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) {
    *handle = (void*)0x12345678;
    return RT_ERROR_NONE;
  }
  virtual rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle){
    return RT_ERROR_NONE;
  }

  virtual rtError_t rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind);

  virtual rtError_t rtMemcpyEx(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind);

  virtual rtError_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind,
                                  rtStream_t stream);

  virtual rtError_t rtMemcpyAsyncWithoutCheckKind(void *dst, uint64_t dest_max, const void *src, uint64_t count,
                                                  rtMemcpyKind_t kind, rtStream_t stream);

  virtual rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type);

  virtual rtError_t rtMallocV2(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId);

  virtual rtError_t rtFree(void *dev_ptr);

  virtual rtError_t rtEschedWaitEvent(int32_t device_id,
                                      uint32_t group_id,
                                      uint32_t thread_id,
                                      int32_t timeout,
                                      rtEschedEventSummary_t *event);

  virtual rtError_t rtRegTaskFailCallbackByModule(const char *moduleName, 
                                                  rtTaskFailCallback callback);

  virtual rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf);

  virtual rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf);

  virtual rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size);

  virtual rtError_t rtMemQueueEnQueue(int32_t dev_id, uint32_t qid, void *mem_buf);

  virtual rtError_t rtCpuKernelLaunch(const void *so_name, const void *kernel_name, uint32_t block_dim,
                                      const void *args, uint32_t args_size, rtSmDesc_t *sm_desc, rtStream_t stream);
 private:
  static std::shared_ptr<RuntimeStub> instance_;
  static RuntimeStub* fake_instance_;
};
}  // namespace ge

#ifdef __cplusplus
extern "C" {
#endif
void rtStubTearDown();

#define RTS_STUB_SETUP()    \
do {                        \
  rtStubTearDown();         \
} while (0)

#define RTS_STUB_TEARDOWN() \
do {                        \
  rtStubTearDown();         \
} while (0)

#define RTS_STUB_RETURN_VALUE(FUNC, TYPE, VALUE)                          \
do {                                                                      \
  g_Stub_##FUNC##_RETURN.emplace(g_Stub_##FUNC##_RETURN.begin(), VALUE);  \
} while (0)

#define RTS_STUB_OUTBOUND_VALUE(FUNC, TYPE, NAME, VALUE)                          \
do {                                                                              \
  g_Stub_##FUNC##_OUT_##NAME.emplace(g_Stub_##FUNC##_OUT_##NAME.begin(), VALUE);  \
} while (0)

extern std::string g_runtime_stub_mock;

#define RTS_STUB_RETURN_EXTERN(FUNC, TYPE) extern std::vector<TYPE> g_Stub_##FUNC##_RETURN;
#define RTS_STUB_OUTBOUND_EXTERN(FUNC, TYPE, NAME) extern std::vector<TYPE> g_Stub_##FUNC##_OUT_##NAME;

RTS_STUB_RETURN_EXTERN(rtGetDevice, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtGetDevice, int32_t, device)

RTS_STUB_RETURN_EXTERN(rtGetDeviceCapability, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtGetDeviceCapability, int32_t, value);

RTS_STUB_RETURN_EXTERN(rtGetRtCapability, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtGetRtCapability, int32_t, value);

RTS_STUB_RETURN_EXTERN(rtGetTsMemType, uint32_t);

RTS_STUB_RETURN_EXTERN(rtStreamWaitEvent, rtError_t);

RTS_STUB_RETURN_EXTERN(rtEventReset, rtError_t);

RTS_STUB_RETURN_EXTERN(rtEventCreate, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtEventCreate, rtEvent_t, event);

RTS_STUB_RETURN_EXTERN(rtGetEventID, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtEventCreate, uint32_t, event_id);

RTS_STUB_RETURN_EXTERN(rtQueryFunctionRegistered, rtError_t);

RTS_STUB_RETURN_EXTERN(rtGetAicpuDeploy, rtError_t);
RTS_STUB_OUTBOUND_EXTERN(rtGetAicpuDeploy, rtAicpuDeployType_t, value);

RTS_STUB_RETURN_EXTERN(rtProfilerTraceEx, rtError_t);

RTS_STUB_RETURN_EXTERN(rtNpuGetFloatStatus, rtError_t);

RTS_STUB_RETURN_EXTERN(rtNpuClearFloatStatus, rtError_t);
#ifdef __cplusplus
}
#endif
#endif // __INC_LLT_RUNTIME_STUB_H
