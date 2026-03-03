/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_ACL_ACL_RT_STUB_H_
#define INC_EXTERNAL_ACL_ACL_RT_STUB_H_

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <memory>
#include <mutex>
#include <set>
#include "mmpa/mmpa_api.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_dump.h"
#include "common/ge_common/ge_types.h"
#include "graph/small_vector.h"
#include "graph/any_value.h"

namespace ge {
class AclRuntimeStub {
public:
  virtual ~AclRuntimeStub() = default;

  static AclRuntimeStub* GetInstance();
  void SetDeviceId(int64_t device_id) {
    device_id_ = device_id;
  }
  static void SetErrorResultApiName(const std::string &stub_api_name);
  static void SetInstance(const std::shared_ptr<AclRuntimeStub> &instance) {
    instance_ = instance;
  }

  static void Install(AclRuntimeStub*);
  static void UnInstall(AclRuntimeStub*);

  static void Reset() {
    instance_.reset();
  }

  virtual aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream);
  virtual aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle,
                                                 uint64_t funcEntry,
                                                 aclrtFuncHandle *funcHandle);
  virtual aclError aclrtLaunchKernel(aclrtFuncHandle funcHandle,
                                     uint32_t blockDim,
                                     const void *argsData,
                                     size_t argsSize,
                                     aclrtStream stream);
  virtual aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId);
  virtual aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout);
  virtual aclError aclrtSetDevice(int32_t deviceId);
  virtual aclError aclrtResetDevice(int32_t deviceId);
  virtual aclError aclrtGetDevice(int32_t *deviceId);
  virtual aclError aclrtGetThreadLastTaskId(uint32_t *taskId);
  virtual aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);
  virtual aclError aclrtDestroyContext(aclrtContext context);
  virtual aclError aclrtSetCurrentContext(aclrtContext context);
  virtual aclError aclrtGetCurrentContext(aclrtContext *context);
  virtual aclError aclrtCreateEvent(aclrtEvent *event);
  virtual aclError aclrtDestroyEvent(aclrtEvent event);
  virtual aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
  virtual aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status);
  virtual aclError aclrtCreateStream(aclrtStream *stream);
  virtual aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);
  virtual aclError aclrtDestroyStream(aclrtStream stream);
  virtual aclError aclrtStreamAbort(aclrtStream stream);
  virtual aclError aclrtSynchronizeStream(aclrtStream stream);
  virtual aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout);
  virtual aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
  virtual aclError aclrtMallocHost(void **hostPtr, size_t size);
  virtual aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);
  virtual aclError aclrtFree(void *devPtr);
  virtual aclError aclrtFreeHost(void *devPtr);
  virtual aclError aclrtMemcpy(void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind);
  virtual aclError aclrtMemcpyAsync(void *dst,
                    size_t dest_max,
                    const void *src,
                    size_t src_count,
                    aclrtMemcpyKind kind,
                    aclrtStream stream);
  virtual aclError aclrtMemcpyAsyncWithCondition(void *dst,
                          size_t destMax,
                          const void *src,
                          size_t count,
                          aclrtMemcpyKind kind,
                          aclrtStream stream);
  virtual aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free_size, size_t *total);
  virtual const char* aclrtGetSocName();
  virtual aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value);
  virtual aclError aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t *const phyDevId);
  virtual aclError aclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes, size_t numBatches,
                                    aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexex, size_t numAttrs, size_t *failIndex);
  virtual aclError aclrtCheckArchCompatibility(const char *socVersion, int32_t *canCompatible);
  virtual aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode);
  virtual aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream);
  virtual aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream);
  virtual aclError aclrtDestroyLabel(aclrtLabel label);
  virtual aclError aclmdlRIDestroy(aclmdlRI modelRI);
  virtual aclError aclmdlRIUnbindStream(aclmdlRI modelRI, aclrtStream stream);
  virtual aclError aclrtDestroyLabelList(aclrtLabelList labelList);
  virtual aclError aclmdlRIBindStream(aclmdlRI modelRI, aclrtStream stream, uint32_t flag);
  virtual aclError aclmdlRIBuildEnd(aclmdlRI modelRI, void *reserve);
  virtual aclError aclrtPersistentTaskClean(aclrtStream stream);
  virtual aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback);
  virtual uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info);
  virtual uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info);
  virtual aclError aclrtGetUserDevIdByLogicDevId(const int32_t logicDevId, int32_t *const userDevid);
  virtual aclError aclrtSetTsDevice(aclrtTsId tsId);
  virtual aclError aclrtGetDeviceCount(uint32_t *count);
  virtual aclError aclrtGetDeviceCapability(int32_t deviceId, aclrtDevFeatureType devFeatureType, int32_t *value);
  virtual aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);
  virtual aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream);
  virtual aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout);
  virtual aclError aclrtMallocForTaskScheduler(void **devPtr, size_t size, aclrtMemMallocPolicy policy,
                                               aclrtMallocConfig *cfg);
  virtual aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr,
                                          uint64_t flags);
  virtual aclError aclrtReleaseMemAddress(void *virPtr);
  virtual aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop,
                                       uint64_t flags);
  virtual aclError aclrtFreePhysical(aclrtDrvMemHandle handle);
  virtual aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags);
  virtual aclError aclrtUnmapMem(void *virPtr);
  virtual aclError aclrtDestroyStreamForce(aclrtStream stream);
  virtual aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
  virtual aclError aclrtStreamWaitEventWithTimeout(aclrtStream stream, aclrtEvent event, int32_t timeout);
  virtual aclError aclrtSetOpWaitTimeout(uint32_t timeout);
  virtual aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode);
  virtual aclError aclrtDeviceGetBareTgid(int32_t *pid);
  virtual aclError aclrtSetOpExecuteTimeOut(uint32_t timeout);
  virtual aclError aclrtSetOpExecuteTimeOutWithMs(uint32_t timeout);
  virtual aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t *actualTimeout);
  virtual aclError aclrtGetStreamAvailableNum(uint32_t *streamCount);
  virtual aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId);
  virtual aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);
  virtual aclError aclrtGetEventAvailNum(uint32_t *eventCount);
  virtual aclError aclrtCreateLabel(aclrtLabel *label);
  virtual aclError aclrtSetLabel(aclrtLabel label, aclrtStream stream);
  virtual aclError aclrtCreateLabelList(aclrtLabel *labels, size_t num, aclrtLabelList *labelList);
  virtual aclError aclrtSwitchLabelByIndex(void *ptr, uint32_t maxValue, aclrtLabelList labelList, aclrtStream stream);
  virtual aclError aclrtSwitchStream(void *leftValue, aclrtCondition cond, void *rightValue,
                                     aclrtCompareDataType dataType, aclrtStream trueStream, aclrtStream falseStream,
                                     aclrtStream stream);
  virtual aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream);
  virtual aclError aclmdlRIExecute(aclmdlRI modelRI, int32_t timeout);
  virtual aclError aclmdlRIBuildBegin(aclmdlRI *modelRI, uint32_t flag);
  virtual aclError aclmdlRIEndTask(aclmdlRI modelRI, aclrtStream stream);
  virtual aclError aclmdlRISetName(aclmdlRI modelRI, const char *name);
  virtual aclError aclrtCtxGetFloatOverflowAddr(void **overflowAddr);
  virtual aclError aclrtGetHardwareSyncAddr(void **addr);
  virtual aclError aclrtTaskUpdateAsync(aclrtStream taskStream, uint32_t taskId, aclrtTaskUpdateInfo *info,
                                        aclrtStream execStream);
  virtual aclError aclmdlRIAbort(aclmdlRI modelRI);
  virtual aclError aclrtProfTrace(void *userdata, int32_t length, aclrtStream stream);
  virtual aclError aclrtCreateNotify(aclrtNotify *notify, uint64_t flag);
  virtual aclError aclrtDestroyNotify(aclrtNotify notify);

 private:
  static std::mutex mutex_;
  static std::shared_ptr<AclRuntimeStub> instance_;
  static thread_local AclRuntimeStub *fake_instance_;
  size_t reserve_mem_size_ = 200UL * 1024UL * 1024UL;
  std::mutex mtx_;
  int64_t device_id_{0L};
  std::vector<aclrtStream> model_bind_streams_;
  std::vector<aclrtStream> model_unbind_streams_;
  size_t input_mem_copy_batch_count_{0UL};
  int32_t cur_device_id = 0;
  int32_t batch_memcpy_device_id = 0;
};

class AclApiStub {
public:
  virtual ~AclApiStub() = default;

  static AclApiStub* GetInstance();

  static void SetInstance(const std::shared_ptr<AclApiStub> &instance) {
    instance_ = instance;
  }

  static void Install(AclApiStub*);
  static void UnInstall(AclApiStub*);

  static void Reset() {
    instance_.reset();
  }

  virtual aclError aclInit(const char *configPath);
  virtual aclError aclFinalize();
  virtual aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);
  virtual aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                    int numDims,
                                    const int64_t *dims,
                                    aclFormat format);
  virtual void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);
  virtual size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer);
  virtual aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize);
  virtual aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc);
  virtual size_t aclGetTensorDescNumDims(const aclTensorDesc *desc);
  virtual aclDataType aclGetTensorDescType(const aclTensorDesc *desc);
  virtual aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer);
  virtual aclmdlConfigHandle *aclmdlCreateConfigHandle();
  virtual aclmdlDataset *aclmdlCreateDataset();
  virtual aclmdlDesc *aclmdlCreateDesc();
  virtual aclError aclmdlDestroyDataset(const aclmdlDataset *dataset);
  virtual aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc);
  virtual aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);
  virtual aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index);
  virtual aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index);
  virtual aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId);
  virtual aclError aclmdlLoadFromMem(const void *model,  size_t modelSize, uint32_t *modelId);
  virtual aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId);
  virtual aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
                                    const void *attrValue, size_t valueSize);
  virtual aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset,
                                      aclTensorDesc *tensorDesc,
                                      size_t index);
  virtual aclError aclmdlSetExternalWeightAddress(aclmdlConfigHandle *handle, const char *weightFileName,
                                          void *devPtr, size_t size);
  virtual aclError aclmdlUnload(uint32_t modelId);
  virtual aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle);
  virtual void aclDestroyTensorDesc(const aclTensorDesc *desc);
  virtual aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);
  virtual const char* acldumpGetPath(acldumpType dumpType);

private:
  static std::mutex mutex_;
  static std::shared_ptr<AclApiStub> instance_;
  static thread_local AclApiStub *fake_instance_;
  std::mutex mtx_;
};
}

#ifdef __cplusplus
extern "C" {
#endif

extern std::string g_acl_stub_mock;
extern std::string g_acl_stub_mock_v2;

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_STUB_H_
