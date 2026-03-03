/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include "acl/acl_rt.h"
#include "acl_stub.h"
#include "acl/acl_base_rt.h"

struct TestAclDataBuffer {
    TestAclDataBuffer(void* const dataIn, const uint64_t len) : data(dataIn), length(len)
    {
    }

    ~TestAclDataBuffer() = default;
    void *data;
    uint64_t length;
};

aclDataBuffer *aclStub::aclCreateDataBuffer(void *data, size_t size)
{
    TestAclDataBuffer *buffer = new(std::nothrow) TestAclDataBuffer(data, size);
    return reinterpret_cast<aclDataBuffer *>(buffer);
}

void *aclStub::aclGetDataBufferAddr(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return nullptr;
    }
    TestAclDataBuffer *buffer = (TestAclDataBuffer *)dataBuffer;
    return buffer->data;
}

aclError aclStub::aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    delete (TestAclDataBuffer *)dataBuffer;
    dataBuffer = nullptr;
    return ACL_SUCCESS;
}

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size)
{
    TestAclDataBuffer *buffer = new(std::nothrow) TestAclDataBuffer(data, size);
    return reinterpret_cast<aclDataBuffer *>(buffer);
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    delete (TestAclDataBuffer *)dataBuffer;
    dataBuffer = nullptr;
    return ACL_SUCCESS;
}

void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferAddr(dataBuffer);
}

size_t aclStub::aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return 0;
    }
    return ((TestAclDataBuffer *)dataBuffer)->length;
}

uint32_t aclStub::aclGetDataBufferSize(const aclDataBuffer *dataBuffer)
{
    if (dataBuffer == nullptr) {
        return 0;
    }
    return ((TestAclDataBuffer *)dataBuffer)->length;
}

aclError aclStub::aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    *devPtr = malloc(size);
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtGetEventId(aclrtEvent event, uint32_t *eventId)
{
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_NONE);
}

aclError aclStub::aclrtDestroyEvent(aclrtEvent event)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtGetRunMode(aclrtRunMode *runMode)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtCreateStream(aclrtStream *stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtDestroyStream(aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtSynchronizeStream(aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtFree(void *devPtr)
{
    free(devPtr);
    devPtr = nullptr;
    return ACL_ERROR_NONE;
}

aclError aclStub::aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}


aclError aclStub::aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority)
{
    *leastPriority = 7;
    *greatestPriority = 0;
    return ACL_SUCCESS;
}

aclError aclStub::aclrtCtxGetCurrentDefaultStream(aclrtStream *stream)
{
    int tmp = 0x1;
    *stream = (aclrtStream)(&tmp);
    return ACL_SUCCESS;
}

aclError aclStub::aclrtRegStreamStateCallback(const char *regName, aclrtStreamStateCallback callback, void *args)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtRegDeviceStateCallback(const char *regName, aclrtDeviceStateCallback callback, void *args)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtLaunchCallback(aclrtCallback fn, void *userData,
    aclrtCallbackBlockType blockType, aclrtStream stream)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

aclError aclStub::aclrtGetDevice(int32_t *deviceId)
{
    return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);
}

size_t aclStub::aclDataTypeSize(aclDataType dataType)
{
    return 0;
}

aclError aclStub::aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtAllocatorGetByStream(aclrtStream stream,
                                aclrtAllocatorDesc *allocatorDesc,
                                aclrtAllocator *allocator,
                                aclrtAllocatorAllocFunc *allocFunc,
                                aclrtAllocatorFreeFunc *freeFunc,
                                aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclInitCallbackRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                        void *userData)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclInitCallbackUnRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclFinalizeCallbackRegister(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc, void *userData)
{
    return ACL_SUCCESS;
}
aclError aclStub::aclFinalizeCallbackUnRegister(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

const char *aclStub::aclrtGetSocName()
{
    return "";
}

aclError aclStub::aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclDumpSetCallbackUnRegister()
{
    return ACL_SUCCESS;
}

aclError aclStub::aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclDumpUnsetCallbackUnRegister()
{
    return ACL_SUCCESS;
}

aclError aclStub::aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtGetCurrentContext(aclrtContext *context)
{
    return ACL_SUCCESS;
}

aclError aclStub::aclrtSetCurrentContext(aclrtContext context)
{
    return ACL_SUCCESS;
}

MockFunctionTest::MockFunctionTest()
{
    ResetToDefaultMock();
}

MockFunctionTest& MockFunctionTest::aclStubInstance()
{
    static MockFunctionTest stub;
    return stub;
}

void MockFunctionTest::ResetToDefaultMock() {
    // delegates the default actions of the RTS methods to aclStub
    ON_CALL(*this, aclrtMalloc)
        .WillByDefault([this](void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
          return aclStub::aclrtMalloc(devPtr, size, policy);
        });
    ON_CALL(*this, aclrtGetSocName)
        .WillByDefault([this]() {
          return aclStub::aclrtGetSocName();
        });
    ON_CALL(*this, aclrtFree)
        .WillByDefault([this](void *devPtr) {
          return aclStub::aclrtFree(devPtr);
        });
    ON_CALL(*this, aclrtFree)
        .WillByDefault([this](void *devPtr) {
          return aclStub::aclrtFree(devPtr);
        });
    ON_CALL(*this, aclCreateDataBuffer)
        .WillByDefault([this](void *data, size_t size) {
          return aclStub::aclCreateDataBuffer(data, size);
        });
    ON_CALL(*this, aclGetDataBufferAddr)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferAddr(dataBuffer);
        });
    ON_CALL(*this, aclGetDataBufferAddr)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferAddr(dataBuffer);
        });
    ON_CALL(*this, aclDestroyDataBuffer)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclDestroyDataBuffer(dataBuffer);
        });
    ON_CALL(*this, aclGetDataBufferSizeV2)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferSizeV2(dataBuffer);
        });
    ON_CALL(*this, aclGetDataBufferSize)
        .WillByDefault([this](const aclDataBuffer *dataBuffer) {
          return aclStub::aclGetDataBufferSize(dataBuffer);
        });
    ON_CALL(*this, aclrtMemcpy)
        .WillByDefault([this](void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
          return aclStub::aclrtMemcpy(dst, destMax, src, count, kind);
        });
    ON_CALL(*this, aclrtCtxGetCurrentDefaultStream)
        .WillByDefault([this](aclrtStream *stream) {
          return aclStub::aclrtCtxGetCurrentDefaultStream(stream);
        });
}

aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    return MockFunctionTest::aclStubInstance().aclrtCreateEventWithFlag(event, flag);
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    return MockFunctionTest::aclStubInstance().aclrtMalloc(devPtr, size, policy);
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId)
{
    return MockFunctionTest::aclStubInstance().aclrtGetEventId(event, eventId);
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtResetEvent(event, stream);
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    return MockFunctionTest::aclStubInstance().aclrtDestroyEvent(event);
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    return MockFunctionTest::aclStubInstance().aclrtStreamWaitEvent(stream, event);
}

aclError aclrtGetRunMode(aclrtRunMode *runMode)
{
    return MockFunctionTest::aclStubInstance().aclrtGetRunMode(runMode);
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind)
{
    return MockFunctionTest::aclStubInstance().aclrtMemcpy(dst, destMax, src, count, kind);
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    return MockFunctionTest::aclStubInstance().aclrtCreateStream(stream);
}

aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtDestroyStream(stream);
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtSynchronizeStream(stream);
}

aclError aclrtFree(void *devPtr)
{
    return MockFunctionTest::aclStubInstance().aclrtFree(devPtr);
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId)
{
    return MockFunctionTest::aclStubInstance().aclrtGetNotifyId(notify, notifyId);
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtUnSubscribeReport(threadId, stream);
}

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtSubscribeReport(threadId, stream);
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    return MockFunctionTest::aclStubInstance().aclrtMemset(devPtr, maxCount, value, count);
}

aclError aclrtLaunchCallback(aclrtCallback fn, void *userData,
    aclrtCallbackBlockType blockType, aclrtStream stream)
{
    return MockFunctionTest::aclStubInstance().aclrtLaunchCallback(fn, userData, blockType, stream);
}

aclError aclrtGetDevice(int32_t *deviceId)
{
    return MockFunctionTest::aclStubInstance().aclrtGetDevice(deviceId);
}

size_t aclDataTypeSize(aclDataType dataType)
{
    return MockFunctionTest::aclStubInstance().aclDataTypeSize(dataType);
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    return MockFunctionTest::aclStubInstance().aclrtSynchronizeStreamWithTimeout(stream, timeout);
}

aclError aclrtAllocatorGetByStream(aclrtStream stream,
                                aclrtAllocatorDesc *allocatorDesc,
                                aclrtAllocator *allocator,
                                aclrtAllocatorAllocFunc *allocFunc,
                                aclrtAllocatorFreeFunc *freeFunc,
                                aclrtAllocatorAllocAdviseFunc *allocAdviseFunc,
                                aclrtAllocatorGetAddrFromBlockFunc *getAddrFromBlockFunc)
{
    return MockFunctionTest::aclStubInstance().aclrtAllocatorGetByStream(stream,
                                                        allocatorDesc, allocator, allocFunc, freeFunc, allocAdviseFunc, getAddrFromBlockFunc);
}

aclError aclInitCallbackRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc, void *userData)
{
    return MockFunctionTest::aclStubInstance().aclInitCallbackRegister(type, cbFunc, userData);
}

aclError aclInitCallbackUnRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclFinalizeCallbackRegister(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc, void *userData)
{
    return MockFunctionTest::aclStubInstance().aclFinalizeCallbackRegister(type, cbFunc, userData);
}

aclError aclFinalizeCallbackUnRegister(aclRegisterCallbackType type,
                                                            aclFinalizeCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferSizeV2(dataBuffer);
}

uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer)
{
    return MockFunctionTest::aclStubInstance().aclGetDataBufferSize(dataBuffer);
}

const char *aclrtGetSocName()
{
    return MockFunctionTest::aclStubInstance().aclrtGetSocName();
}

#ifdef __cplusplus
extern "C" {
#endif
aclError aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc)
{
    return MockFunctionTest::aclStubInstance().aclDumpSetCallbackRegister(cbFunc);
}

aclError aclDumpSetCallbackUnRegister()
{
    return ACL_SUCCESS;
}

aclError aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc)
{
    return ACL_SUCCESS;
}

aclError aclDumpUnsetCallbackUnRegister()
{
    return ACL_SUCCESS;
}
#ifdef __cplusplus
}
#endif


aclError aclrtGetCurrentContext(aclrtContext *context)
{
    return MockFunctionTest::aclStubInstance().aclrtGetCurrentContext(context);
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    return MockFunctionTest::aclStubInstance().aclrtSetCurrentContext(context);
}

aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority)
{
    *leastPriority = 7;
    *greatestPriority = 0;
    return MockFunctionTest::aclStubInstance().aclrtDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
}

aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream)
{
    return MockFunctionTest::aclStubInstance().aclrtCtxGetCurrentDefaultStream(stream);
}

aclError aclrtRegStreamStateCallback(const char *regName, aclrtStreamStateCallback callback, void *args)
{
    return MockFunctionTest::aclStubInstance().aclrtRegStreamStateCallback(regName, callback, args);
}

aclError aclrtRegDeviceStateCallback(const char *regName, aclrtDeviceStateCallback callback, void *args)
{
    return MockFunctionTest::aclStubInstance().aclrtRegDeviceStateCallback(regName, callback, args);
}