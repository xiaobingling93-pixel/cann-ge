/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <map>
#include <queue>
#include <iostream>
#include "ascendcl_stub.h"
#include "mmpa/mmpa_api.h"

extern std::string g_acl_stub_mock;
extern std::string g_acl_stub_mock_v2;
std::string g_acl_stub_mock = "";
std::string g_acl_stub_mock_v2 = "";
static char g_soc_version[50] = {0};

static int32_t g_free_stream_num = 2048;
static int32_t g_free_event_num = 2048;
static int32_t g_cnt_rtStreamSynchronize_over_flow = 0;
static int32_t g_cnt_rtStreamSynchronize_fail = 0;
static size_t reserve_mem_size_ = 200UL * 1024UL * 1024UL;

#define EVENT_LENTH 10
#define NOTIFY_LENTH 10

#define ACL_DELETE_AND_SET_NULL(var) \
    do { \
        if ((var) != nullptr) { \
            delete (var); \
            (var) = nullptr; \
        } \
    } \
    while (false)

#define ACL_DELETE_ARRAY_AND_SET_NULL(var) \
    do { \
        if ((var) != nullptr) { \
            delete[] (var); \
            (var) = nullptr; \
        } \
    } \
    while (false)

enum class AttrRangeTypeStub : std::uint8_t {
    RANGE_TYPE,
    VALUE_TYPE
};


struct aclTensorDescStub {
    aclTensorDescStub(const aclDataType aclTensorDataType, const std::initializer_list<int64_t> shape,
        const aclFormat aclTensorFormat);
    aclTensorDescStub(const aclDataType aclTensorDataType, const size_t numDims, const int64_t *const aclTensorDims,
        const aclFormat aclTensorFormat) : dataType(aclTensorDataType), format(aclTensorFormat) {
      for (size_t i = 0U; i < numDims; ++i) {
        this->dims.push_back(*(aclTensorDims + i));
      }
    }
    aclTensorDescStub(const aclTensorDescStub &tensorDesc);
    aclTensorDescStub &operator=(const aclTensorDescStub &tensorDesc);
    aclTensorDescStub() = default;
    ~aclTensorDescStub() = default;
    aclDataType dataType;
    aclFormat storageFormat = ACL_FORMAT_UNDEFINED;
    aclFormat format;
    ge::SmallVector<int64_t, static_cast<size_t>(8)> dims;
    ge::SmallVector<int64_t, static_cast<size_t>(8)> dimsBackup;
    ge::SmallVector<int64_t, static_cast<size_t>(8)> storageDims;
    ge::SmallVector<int64_t, static_cast<size_t>(8)> storageDimsBackup;
    std::string name;
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    std::vector<std::pair<int64_t, int64_t>> shapeRangeBackup;
    void *address = nullptr;
    std::string dynamicInputName;
    bool isConst = false;
    std::shared_ptr<void> constDataBuf;
    size_t constDataLen = 0U;
    bool isConstBackup = false;
    std::shared_ptr<void> constDataBufBackup;
    size_t constDataLenBackup = 0U;
    aclMemType memtype = ACL_MEMTYPE_DEVICE;
    // valRange is set from aclSetTensorValueRange
    std::vector<std::pair<int64_t, int64_t>> valRange;
    // for windows compile,use map ignore dvpp.so find the implementation GeAttrValue
    std::map<AttrRangeTypeStub, ge::GeAttrValue> valueRange;
    std::string DebugString() const;
    bool IsSameTensor(const aclTensorDescStub *const other) const;
    bool IsDynamicTensor() const;
    bool CheckShapeRange() const;
    bool IsConstTensor() const
    {
        return isConst;
    }
    bool IsHostMemTensor() const
    {
        return (memtype == ACL_MEMTYPE_HOST) || (memtype == ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
    }
    inline bool IsOptinalTensor() const
    {
        return (dataType == ACL_DT_UNDEFINED) && (format == ACL_FORMAT_UNDEFINED) && (dims.empty());
    }
    void Init(const aclTensorDescStub &tensorDesc);
    void UpdateTensorShape(const std::vector<int64_t> &shape);
    void UpdateTensorShapeRange(const std::vector<std::pair<int64_t, int64_t>> &ranges);
    inline bool CheckConstTensor(const bool needCheckHostMem) const
    {
        return isConst || (needCheckHostMem && (memtype == ACL_MEMTYPE_HOST));
    }

    bool operator==(const aclTensorDescStub *const other) const;
    void BackupDimsAndShapeRanges();
    void RecoverDimsAndShapeRanges();
    void BackupConst();
    void RecoverConst();

private:
    mutable std::string cachedKey;
    mutable std::string cachedShapeKey;
};

struct aclDataBufferStub {
    aclDataBufferStub(void* const dataIn, const uint64_t len) : data(dataIn), length(len)
    {
    }

    ~aclDataBufferStub() = default;
    void *data;
    uint64_t length;
};

struct AclModelTensorStub {
    AclModelTensorStub(aclDataBufferStub *const dataBufIn,
        aclTensorDescStub *const tensorDescIn) : dataBuf(dataBufIn), tensorDesc(tensorDescIn)
    {
    }

    ~AclModelTensorStub() = default;
    aclDataBufferStub *dataBuf;
    aclTensorDescStub *tensorDesc;
};

typedef void (*FnDestroyStub)(void *);

typedef struct {
  size_t itemSize;
  size_t size;
  size_t capacity;
  uint8_t *data;
  FnDestroyStub pfnDestroyItem;
} VectorStub;

struct aclmdlDescStub {
  uint32_t modelId;
  VectorStub inputDesc;
  VectorStub outputDesc;
};

struct aclmdlDatasetStub {
    aclmdlDatasetStub()
        : seq(0U),
          modelId(0U),
          timestamp(0U),
          timeout(0U),
          requestId(0U),
          dynamicBatchSize(0U),
          dynamicResolutionHeight(0U),
          dynamicResolutionWidth(0U) {}
    ~aclmdlDatasetStub() = default;
    uint32_t seq;
    uint32_t modelId;
    std::vector<AclModelTensorStub> blobs;
    uint32_t timestamp;
    uint32_t timeout;
    uint64_t requestId;
    uint64_t dynamicBatchSize;
    uint64_t dynamicResolutionHeight;
    uint64_t dynamicResolutionWidth;
    std::vector<uint64_t> dynamicDims;
};


struct aclmdlConfigHandleStub {
    aclmdlConfigHandleStub()
        : priority(0),
          mdlLoadType(0U),
          mdlAddr(nullptr),
          mdlSize(0U),
          workPtr(nullptr),
          workSize(0U),
          weightPtr(nullptr),
          weightSize(0U),
          inputQ(nullptr),
          inputQNum(0U),
          outputQ(nullptr),
          outputQNum(0U),
          reuseZeroCopy(0U) {}
    int32_t priority;
    size_t mdlLoadType;
    std::string loadPath;
    void *mdlAddr;
    size_t mdlSize;
    void *workPtr;
    size_t workSize;
    void *weightPtr;
    size_t weightSize;
    const uint32_t *inputQ;
    size_t inputQNum;
    const uint32_t *outputQ;
    size_t outputQNum;
    size_t reuseZeroCopy;
    std::string weightPath;
    std::set<aclmdlConfigAttr> attrState;
    std::vector<ge::FileConstantMem> fileConstantMem;
};

namespace ge {
struct aclrtContextStub {
    int32_t deviceId;
};

std::shared_ptr<AclRuntimeStub> AclRuntimeStub::instance_;
std::mutex AclRuntimeStub::mutex_;
thread_local AclRuntimeStub* AclRuntimeStub::fake_instance_;
void AclRuntimeStub::SetErrorResultApiName(const std::string &stub_api_name) {
  g_acl_stub_mock = stub_api_name;
}
AclRuntimeStub *AclRuntimeStub::GetInstance() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if(fake_instance_ != nullptr){
    return fake_instance_;
  }
  if (instance_ == nullptr) {
    instance_ = std::make_shared<AclRuntimeStub>();
  }
  return instance_.get();
}

void AclRuntimeStub::Install(AclRuntimeStub* instance){
  fake_instance_ = instance;
}

void AclRuntimeStub::UnInstall(AclRuntimeStub*){
  fake_instance_ = nullptr;
}

aclError AclRuntimeStub::aclrtRecordNotify(aclrtNotify notify, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle,
                                                       uint64_t funcEntry,
                                                       aclrtFuncHandle *funcHandle) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtLaunchKernel(aclrtFuncHandle funcHandle,
                                           uint32_t blockDim,
                                           const void *argsData,
                                           size_t argsSize,
                                           aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtStreamGetId(aclrtStream stream, int32_t *streamId) {
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return -1;
  }
  (void) stream;
  if (*streamId == 999) {
    return -1;
  }
  *streamId = 0;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetDevice(int32_t deviceId) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtResetDevice(int32_t deviceId) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetDevice(int32_t *deviceId) {
  if (__FUNCTION__ == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  *deviceId = 0;
  if (cur_device_id > 0) {
    *deviceId = cur_device_id;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetThreadLastTaskId(uint32_t *taskId) {
  if (*taskId == 999) {
    return -1;
  }
  *taskId = 0;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
  aclrtContextStub *ctxStub = new aclrtContextStub;
  ctxStub->deviceId = deviceId;
  *context = ctxStub;
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtDestroyContext(aclrtContext context) {
  if (context != nullptr) {
    delete (aclrtContextStub *)context;
  }
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtSetCurrentContext(aclrtContext context) {
  const char * const kEnvRecordPath = "SET_TRANS_VAR_DATA";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetCurrentContext(aclrtContext *context) {
  if (__FUNCTION__ == g_acl_stub_mock) {
    return -1;
  }
  uintptr_t x = 1;
  *context = (aclrtContext *)x;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateEvent(aclrtEvent *event) {
  if (__FUNCTION__ == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  if(g_free_event_num <= 0) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  g_free_event_num--;
  *event = new int[EVENT_LENTH];
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {
  if (__FUNCTION__ == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  if(g_free_event_num <= 0) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  g_free_event_num--;
  *event = new int[EVENT_LENTH];
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtDestroyEvent(aclrtEvent event) {
  g_free_event_num++;
  delete[](int *) event;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
  *status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateStream(aclrtStream *stream) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_4";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  if(g_free_stream_num <= 0) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  g_free_stream_num--;
  *stream = new uint32_t;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
  if(g_free_stream_num <= 0) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  g_free_stream_num--;
  *stream = new uint32_t;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtDestroyStream(aclrtStream stream) {
  if (stream != nullptr) {
    delete (uint32_t *)stream;
  }
  g_free_stream_num++;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtDestroyStreamForce(aclrtStream stream) {
  if (stream != nullptr) {
    delete (uint32_t *)stream;
  }
  g_free_stream_num++;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtStreamAbort(aclrtStream stream) {
  (void) stream;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSynchronizeStream(aclrtStream stream) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  const char * const kEnvPath = "END_OF_SEQUENCE";
  char env_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvPath, &env_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&env_path[0]).find("end") != std::string::npos) {
    return ACL_ERROR_RT_END_OF_SEQUENCE;
  }

  const char * const kEnvOverFlowPath = "ACL_ERROR_RT_OVER_FLOW";
  char over_flow_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvOverFlowPath, &over_flow_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&over_flow_path[0]).find("over_flow") != std::string::npos) {
    return ACL_ERROR_RT_OVER_FLOW;
  }

  const char * const kEnvPathSt = "MOCK_FAIL_ST";
  char env_path_st[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvPathSt, &env_path_st[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&env_path_st[0]).find("mock_st_fail") != std::string::npos) {
    g_cnt_rtStreamSynchronize_fail++;
    if (g_cnt_rtStreamSynchronize_fail == 3) {
      return ACL_ERROR_RT_INTERNAL_ERROR;
    }
  }

  const char * const kEnvOverFlowPathSt = "ACL_ERROR_RT_OVER_FLOW_ST";
  char over_flow_path_st[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvOverFlowPathSt, &over_flow_path_st[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&over_flow_path_st[0]).find("over_st_flow") != std::string::npos) {
    g_cnt_rtStreamSynchronize_over_flow++;
    if (g_cnt_rtStreamSynchronize_over_flow == 3) {
      return ACL_ERROR_RT_OVER_FLOW;
    }
  }

  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_9";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  const char * const kEnvPath = "END_OF_SEQUENCE";
  char env_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvPath, &env_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&env_path[0]).find("end") != std::string::npos) {
    return ACL_ERROR_RT_END_OF_SEQUENCE;
  }

  const char * const kEnvPathWithTimeout = "WITH_TIMEOUT_END_OF_SEQUENCE";
  char end_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvPathWithTimeout, &end_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&end_path[0]).find("end") != std::string::npos) {
    return ACL_ERROR_RT_END_OF_SEQUENCE;
  }

  const char * const kTimeoutEnvPath = "TIMEOUT";
  char timeout_env_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kTimeoutEnvPath, &timeout_env_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&timeout_env_path[0]).find("timeout") != std::string::npos) {
    return ACL_ERROR_RT_STREAM_SYNC_TIMEOUT;
  }
  const char * const kOverflowEnvPath = "SYNCSTREAM_OVERFLOW_RET";
  char overflow_env_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kOverflowEnvPath, &overflow_env_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&overflow_env_path[0]).find("aicore") != std::string::npos) {
    return ACL_ERROR_RT_AICORE_OVER_FLOW;
  }
  if (std::string(&overflow_env_path[0]).find("aicpu") != std::string::npos) {
    return ACL_ERROR_RT_OVER_FLOW;
  }
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_2";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  const char *const kEnvHybridProfiling = "HYBRID_PROFILING_LEVEL";
  char record_path1[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvHybridProfiling, &record_path1[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path1[0]).find("1") != std::string::npos) {
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return ACL_SUCCESS;
  }
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  if (size == 123) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  const char *const kEnvRecordPath_Huge = "MOCK_MEMCPY_HUGE";
  char record_path_Huge[MMPA_MAX_PATH] = {};
  int32_t ret = mmGetEnv(kEnvRecordPath_Huge, &record_path_Huge[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if ((ret == EN_OK) && (strlen(record_path_Huge) != 0)) {
    *devPtr = new uint8_t[size];
    memset_s(*devPtr, size, 0, size);
    return ACL_SUCCESS;
  }
  if (size > INT32_MAX) {
    *devPtr = new uint8_t[1024U];
    return ACL_SUCCESS;
  }
  *devPtr = new uint8_t[size];
  memset_s(*devPtr, size, 0, size);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMallocHost(void **hostPtr, size_t size) {
  return aclrtMalloc(hostPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclError AclRuntimeStub::aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) {
  if (maxCount == 321) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  memset_s(devPtr, maxCount, value, count);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtFree(void *devPtr) {
  delete[](uint8_t *) devPtr;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtFreeHost(void *devPtr) {
  delete[](uint8_t *) devPtr;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMemcpy(void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  const char *const kEnvRecordPath1 = "NPU_COLLECT_PATH_EXE";
  (void)mmGetEnv(kEnvRecordPath1, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (!std::string(&record_path[0]).empty()) {
    return ACL_SUCCESS;
  }

  if (__FUNCTION__ == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  if (dst != nullptr && src != nullptr) {
    dest_max = std::min(dest_max, reserve_mem_size_);
    memcpy_s(dst, dest_max, src, count);
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMemcpyAsync(void *dst,
              size_t dest_max,
              const void *src,
              size_t src_count,
              aclrtMemcpyKind kind,
              aclrtStream stream) {
  const char *const kEnvRecordPath = "MOCK_MEMCPY_HUGE";
  char record_path[MMPA_MAX_PATH] = {};
  int32_t ret = mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if ((ret != EN_OK) || (strlen(record_path) == 0)) {
    if (dst != nullptr && src != nullptr) {
      dest_max = std::min(dest_max, reserve_mem_size_);
      memcpy_s(dst, dest_max, src, src_count);
    }
    return ACL_SUCCESS;
  }
  size_t offset = 0U;
  size_t remain_size = src_count;
  do {
    size_t copy_size = (remain_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : remain_size;
    memcpy_s((dst + offset), copy_size, (src + offset), copy_size);
    offset += copy_size;
    remain_size -= copy_size;
  } while (remain_size > 0U);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMemcpyAsyncWithCondition(void *dst,
                                                      size_t destMax,
                                                      const void *src,
                                                      size_t count,
                                                      aclrtMemcpyKind kind,
                                                      aclrtStream stream) {
  return aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

aclError AclRuntimeStub::aclrtGetMemInfo(aclrtMemAttr attr, size_t *free_size, size_t *total) {
  *free_size = 64UL * 1024UL * 1024UL;
  *total = 128UL * 1024UL * 1024UL;
  return ACL_SUCCESS;
}

// no change for rt here
const char* AclRuntimeStub::aclrtGetSocName() {
  return g_soc_version;
}

aclError AclRuntimeStub::aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
  *value = 8;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t *const phyDevId) {
  *phyDevId = logicDevId;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes, size_t numBatches,
                          aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexex, size_t numAttrs, size_t *failIndex)
{
  *failIndex = static_cast<size_t>(0);
  if (__FUNCTION__ == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }

  if (dsts != nullptr && srcs != nullptr) {
    for (size_t i = 0; i < numBatches; i++) {
      memcpy_s(dsts[i], destMax[i], srcs[i], sizes[i]);
    }
  }
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtCheckArchCompatibility(const char *socVersion, int32_t *canCompatible) {
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return -1;
  }
  *canCompatible = 1;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtActiveStream(aclrtStream activeStream, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCtxGetCurrentDefaultStream(aclrtStream *stream) {
  if (__FUNCTION__ == g_acl_stub_mock) {
    return -1;
  }
  uintptr_t x = 1;
  *stream = (aclrtStream *)x;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtDestroyLabel(aclrtLabel label) {
  if (label != nullptr) {
    delete reinterpret_cast<int *>(label);
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIDestroy(aclmdlRI modelRI) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIUnbindStream(aclmdlRI modelRI, aclrtStream stream) {
  const std::lock_guard<std::mutex> lock(mtx_);
  model_unbind_streams_.emplace_back(stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtDestroyLabelList(aclrtLabelList labelList) {
  delete[](int *) labelList;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIBindStream(aclmdlRI modelRI, aclrtStream stream, uint32_t flag) {
  const std::lock_guard<std::mutex> lock(mtx_);
  model_bind_streams_.emplace_back(stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIBuildEnd(aclmdlRI modelRI, void *reserve) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_7";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtPersistentTaskClean(aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetTsDevice(aclrtTsId tsId) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) {
  return ACL_SUCCESS;
}
uint32_t AclRuntimeStub::aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info) {
  return 0U;
}

uint32_t AclRuntimeStub::aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info) {
  return 0U;
}

aclError AclRuntimeStub::aclrtGetUserDevIdByLogicDevId(const int32_t logicDevId, int32_t *const userDevid) {
  *userDevid = 0;
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtGetDeviceCount(uint32_t *count) {
  *count = 1U;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetDeviceCapability(int32_t deviceId,
                                                  aclrtDevFeatureType devFeatureType, int32_t *value) {
  *value = 1;
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return -1;
  }

  if(g_free_event_num <= 0) {
    return -1;
  }
  g_free_event_num--;
  *event = new int[EVENT_LENTH];
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtResetEvent(aclrtEvent event, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMallocForTaskScheduler(void **devPtr, size_t size, aclrtMemMallocPolicy policy,
                                                     aclrtMallocConfig *cfg) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr,
                                                uint64_t flags) {
  if (size < 200UL * 1024UL *1024UL) {
    *virPtr = new uint8_t[size];
    reserve_mem_size_ = size;
  } else {
    *virPtr = new uint8_t[reserve_mem_size_];
  }
  memset_s(*virPtr, reserve_mem_size_, 0, reserve_mem_size_);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtReleaseMemAddress(void *virPtr) {
  delete[] (uint8_t *)virPtr;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop,
                                             uint64_t flags) {
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return -1;
  }
  *handle = (aclrtDrvMemHandle) new uint8_t[8];
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtFreePhysical(aclrtDrvMemHandle handle) {
  if (handle != nullptr) {
    delete[] (uint8_t *)handle;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtUnmapMem(void *virPtr) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtStreamWaitEventWithTimeout(aclrtStream stream, aclrtEvent event, int32_t timeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetOpWaitTimeout(uint32_t timeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetOpExecuteTimeOut(uint32_t timeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtDeviceGetBareTgid(int32_t *pid) {
  *pid = static_cast<int32_t>(getpid());
  return ACL_ERROR_NONE;
}

aclError AclRuntimeStub::aclrtSetOpExecuteTimeOutWithMs(uint32_t timeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t *actualTimeout) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetStreamAvailableNum(uint32_t *streamCount) {
  const char *const kEnvRecordPath = "MOCK_AVAIL_STREAM_NUM";
  char record_path[8] = {};
  int32_t ret = mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(8));
  if ((ret != EN_OK) || (strlen(record_path) == 0)) {
    *streamCount = g_free_stream_num;
    return ACL_SUCCESS;
  }
  try {
    *streamCount = std::stoi(std::string(record_path));
    return ACL_SUCCESS;
  } catch (...) {
    return 1; // SOME ERROR
  }
  *streamCount = g_free_stream_num;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetEventId(aclrtEvent event, uint32_t *eventId) {
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return -1;
  }
  *eventId = 1;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetEventAvailNum(uint32_t *eventCount) {
  *eventCount = (uint32_t)g_free_event_num;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateLabel(aclrtLabel *label) {
  if (std::string(__FUNCTION__) == g_acl_stub_mock) {
    return -1;
  }
  *label = reinterpret_cast<aclrtLabel>(new int);
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSetLabel(aclrtLabel label, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCreateLabelList(aclrtLabel *labels, size_t num, aclrtLabelList *labelList) {
  *labelList = new int[num];
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSwitchLabelByIndex(void *ptr, uint32_t maxValue, aclrtLabelList labelList, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtSwitchStream(void *leftValue, aclrtCondition cond, void *rightValue,
                                           aclrtCompareDataType dataType, aclrtStream trueStream, aclrtStream falseStream,
                                           aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_8";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIExecute(aclmdlRI modelRI, int32_t timeout) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_8";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIBuildBegin(aclmdlRI *modelRI, uint32_t flag) {
  const char * const kEnvRecordPath = "CONSTANT_FOLDING_PASS_3";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));

  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIEndTask(aclmdlRI modelRI, aclrtStream stream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRISetName(aclmdlRI modelRI, const char *name) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtCtxGetFloatOverflowAddr(void **overflowAddr) {
  *overflowAddr = (void *)0x1;
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtGetHardwareSyncAddr(void **addr) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtTaskUpdateAsync(aclrtStream taskStream, uint32_t taskId, aclrtTaskUpdateInfo *info,
                                              aclrtStream execStream) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclmdlRIAbort(aclmdlRI modelRI) {
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtProfTrace(void *userdata, int32_t length, aclrtStream stream) {
  const char *const kEnvRecordPath = "CONSTANT_FOLDING_PASS";
  char record_path[MMPA_MAX_PATH] = {};
  (void)mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  if (std::string(&record_path[0]).find("mock_fail") != std::string::npos) {
    return -1;
  }
  return ACL_SUCCESS;
}
aclError AclRuntimeStub::aclrtCreateNotify(aclrtNotify *notify, uint64_t flag) {
  if (__FUNCTION__ == g_acl_stub_mock) {
    return ACL_ERROR_RT_INTERNAL_ERROR;
  }
  *notify = new int[NOTIFY_LENTH];
  return ACL_SUCCESS;
}

aclError AclRuntimeStub::aclrtDestroyNotify(aclrtNotify notify) {
  if (notify != nullptr) {
    delete[](int *) notify;
    notify = nullptr;
  }
  return ACL_SUCCESS;
}

std::shared_ptr<AclApiStub> AclApiStub::instance_;
std::mutex AclApiStub::mutex_;
thread_local AclApiStub* AclApiStub::fake_instance_;
AclApiStub *AclApiStub::GetInstance() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if(fake_instance_ != nullptr){
    return fake_instance_;
  }
  if (instance_ == nullptr) {
    instance_ = std::make_shared<AclApiStub>();
  }
  return instance_.get();
}


void AclApiStub::Install(AclApiStub* instance) {
  fake_instance_ = instance;
}

void AclApiStub::UnInstall(AclApiStub*) {
  fake_instance_ = nullptr;
}


aclError AclApiStub::aclInit(const char *configPath) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclFinalize() {
  return ACL_SUCCESS;
}

aclDataBuffer *AclApiStub::aclCreateDataBuffer(void *data, size_t size) {
  if (data == nullptr || size <= 0) {
    return nullptr;
  }
  aclDataBufferStub *buffer = new(std::nothrow) aclDataBufferStub(data, size);
  return reinterpret_cast<aclDataBuffer *>(buffer);
}

aclTensorDesc *AclApiStub::aclCreateTensorDesc(aclDataType dataType,
                                  int numDims,
                                  const int64_t *dims,
                                  aclFormat format) {
  if (numDims < 0) {
    return nullptr;
  }
  if ((numDims > 0) && (dims == nullptr)) {
    return nullptr;
  }
  aclTensorDescStub *tensor_desc = new(std::nothrow) aclTensorDescStub[1]{{dataType, static_cast<size_t>(numDims), dims, format}};
  return reinterpret_cast<aclTensorDesc *>(tensor_desc);
}

void *AclApiStub::aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) {
    return nullptr;
  }
  aclDataBufferStub *buffer = reinterpret_cast<aclDataBufferStub *>(const_cast<aclDataBuffer *>(dataBuffer));
  return buffer->data;
}

size_t AclApiStub::aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer) {
  if (dataBuffer == NULL) {
    return 0UL;
  }
  aclDataBufferStub *buffer = reinterpret_cast<aclDataBufferStub *>(const_cast<aclDataBuffer *>(dataBuffer));
  return static_cast<size_t>(buffer->length);
}

aclError AclApiStub::aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize) {
  if (desc == NULL) {
    return ACL_ERROR;
  }
  aclTensorDescStub *stub_desc = reinterpret_cast<aclTensorDescStub *>(const_cast<aclTensorDesc *>(desc));
  if (index >= stub_desc->dims.size()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *dimSize = stub_desc->dims[index];
  return ACL_SUCCESS;
}

aclFormat AclApiStub::aclGetTensorDescFormat(const aclTensorDesc *desc) {
  if (desc == nullptr) {
    return ACL_FORMAT_UNDEFINED;
  }
  aclTensorDescStub *stub_desc = reinterpret_cast<aclTensorDescStub *>(const_cast<aclTensorDesc *>(desc));
  return stub_desc->format;
}

size_t AclApiStub::aclGetTensorDescNumDims(const aclTensorDesc *desc) {
  if (desc == nullptr) {
    return 0U;
  }
  aclTensorDescStub *stub_desc = reinterpret_cast<aclTensorDescStub *>(const_cast<aclTensorDesc *>(desc));
  if ((stub_desc->dims.size() > 0U) && (stub_desc->dims[0U] == -2)) {
    return static_cast<size_t>(ACL_UNKNOWN_RANK);
  }
  return stub_desc->dims.size();
}

aclDataType AclApiStub::aclGetTensorDescType(const aclTensorDesc *desc) {
  if (desc == nullptr) {
    return ACL_DT_UNDEFINED;
  }
  aclTensorDescStub *stub_desc = reinterpret_cast<aclTensorDescStub *>(const_cast<aclTensorDesc *>(desc));
  return stub_desc->dataType;
}

aclError AclApiStub::aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer) {
  return ACL_SUCCESS;
}

aclmdlConfigHandle *AclApiStub::aclmdlCreateConfigHandle() {
  aclmdlConfigHandleStub *stub_handle = new(std::nothrow) aclmdlConfigHandleStub();
  return reinterpret_cast<aclmdlConfigHandle *>(stub_handle);
}

aclmdlDataset *AclApiStub::aclmdlCreateDataset() {
  aclmdlDatasetStub *stub_dataset = new(std::nothrow) aclmdlDatasetStub();
  return reinterpret_cast<aclmdlDataset *>(stub_dataset);
}

aclmdlDesc *AclApiStub::aclmdlCreateDesc() {
  aclmdlDescStub *stub_desc = new(std::nothrow) aclmdlDescStub();
  return reinterpret_cast<aclmdlDesc *>(stub_desc);
}

aclError AclApiStub::aclmdlDestroyDataset(const aclmdlDataset *dataset) {
  aclmdlDatasetStub *stub_dataset = reinterpret_cast<aclmdlDatasetStub *>(const_cast<aclmdlDataset *>(dataset));
  for (size_t i = 0U; i < stub_dataset->blobs.size(); ++i) {
    ACL_DELETE_ARRAY_AND_SET_NULL(stub_dataset->blobs[i].tensorDesc);
  }
  ACL_DELETE_AND_SET_NULL(stub_dataset);
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlDestroyDesc(aclmdlDesc *modelDesc) {
  aclmdlDescStub *stub_desc = reinterpret_cast<aclmdlDescStub *>(modelDesc);
  ACL_DELETE_AND_SET_NULL(stub_desc);
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output) {
  return ACL_SUCCESS;
}

aclDataBuffer *AclApiStub::aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index) {
  aclmdlDatasetStub *stub_dataset = reinterpret_cast<aclmdlDatasetStub *>(const_cast<aclmdlDataset *>(dataset));
  if ((stub_dataset == nullptr) || (index >= stub_dataset->blobs.size())) {
    return nullptr;
  }

  return reinterpret_cast<aclDataBuffer *>(stub_dataset->blobs[index].dataBuf);
}

aclTensorDesc *AclApiStub::aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index) {
  aclmdlDatasetStub *stub_dataset = reinterpret_cast<aclmdlDatasetStub *>(const_cast<aclmdlDataset *>(dataset));
  if (index >= stub_dataset->blobs.size()) {
    return nullptr;
  }
  return reinterpret_cast<aclTensorDesc *>(stub_dataset->blobs[index].tensorDesc);
}

aclError AclApiStub::aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlLoadFromMem(const void *model,  size_t modelSize, uint32_t *modelId) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
                                  const void *attrValue, size_t valueSize) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset,
                                    aclTensorDesc *tensorDesc,
                                    size_t index) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlSetExternalWeightAddress(aclmdlConfigHandle *handle, const char *weightFileName,
                                        void *devPtr, size_t size) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlUnload(uint32_t modelId) {
  return ACL_SUCCESS;
}

aclError AclApiStub::aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle) {
  aclmdlConfigHandleStub *stub_handle = reinterpret_cast<aclmdlConfigHandleStub *>(handle);
  if (stub_handle == nullptr) {
    return ACL_ERROR;
  }
  ACL_DELETE_AND_SET_NULL(stub_handle);
  return ACL_SUCCESS;
}

void AclApiStub::aclDestroyTensorDesc(const aclTensorDesc *desc) {
  aclTensorDescStub *stub_desc = reinterpret_cast<aclTensorDescStub *>(const_cast<aclTensorDesc *>(desc));
  ACL_DELETE_ARRAY_AND_SET_NULL(stub_desc);
}

aclError AclApiStub::aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
  aclDataBufferStub *stub_buffer = reinterpret_cast<aclDataBufferStub *>(const_cast<aclDataBuffer *>(dataBuffer));
  ACL_DELETE_AND_SET_NULL(stub_buffer);
  return ACL_SUCCESS;
}

const char* AclApiStub::acldumpGetPath(acldumpType dumpType) {
  (void) dumpType;
  return "";
}

}

#ifdef __cplusplus
extern "C" {
#endif

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtRecordNotify(notify, stream);
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle *funcHandle) {
  return ge::AclRuntimeStub::GetInstance()->aclrtBinaryGetFunctionByEntry(binHandle, funcEntry, funcHandle);
}

aclError aclrtLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, const void *argsData, size_t argsSize,
  aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtLaunchKernel(funcHandle, blockDim, argsData, argsSize, stream);
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtStreamGetId(stream, streamId);
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtWaitAndResetNotify(notify, stream, timeout);
}

aclError aclrtSetDevice(int32_t deviceId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetDevice(deviceId);
}

aclError aclrtResetDevice(int32_t deviceId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtResetDevice(deviceId);
}

aclError aclrtGetDevice(int32_t *deviceId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetDevice(deviceId);
}

aclError aclrtGetThreadLastTaskId(uint32_t *taskId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetThreadLastTaskId(taskId);
}

aclError aclrtSetCurrentContext(aclrtContext context) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetCurrentContext(context);
}

aclError aclrtGetCurrentContext(aclrtContext *context) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetCurrentContext(context);
}

aclError aclrtCreateEvent(aclrtEvent *event) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateEvent(event);
}

aclError aclrtDestroyEvent(aclrtEvent event) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyEvent(event);
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtRecordEvent(event, stream);
}

aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
  return ge::AclRuntimeStub::GetInstance()->aclrtQueryEventStatus(event, status);
}

aclError aclrtCreateStream(aclrtStream *stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateStream(stream);
}

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateStreamWithConfig(stream, priority, flag);
}

aclError aclrtDestroyStream(aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyStream(stream);
}

aclError aclrtDestroyStreamForce(aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyStreamForce(stream);
}

aclError aclrtStreamAbort(aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtStreamAbort(stream);
}

aclError aclrtSynchronizeStream(aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSynchronizeStream(stream);
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSynchronizeStreamWithTimeout(stream, timeout);
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMallocHost(void **hostPtr, size_t size) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMallocHost(hostPtr, size);
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMemset(devPtr, maxCount, value, count);
}

aclError aclrtFree(void *devPtr) {
  return ge::AclRuntimeStub::GetInstance()->aclrtFree(devPtr);
}

aclError aclrtFreeHost(void *devPtr) {
  return ge::AclRuntimeStub::GetInstance()->aclrtFreeHost(devPtr);
}

aclError aclrtMemcpy(void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMemcpy(dst, dest_max, src, count, kind);
}

aclError aclrtMemcpyAsync(void *dst,
                          size_t dest_max,
                          const void *src,
                          size_t src_count,
                          aclrtMemcpyKind kind,
                          aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMemcpyAsync(dst, dest_max, src, src_count, kind, stream);
}

aclError aclrtMemcpyAsyncWithCondition(void *dst,
                                        size_t destMax,
                                        const void *src,
                                        size_t count,
                                        aclrtMemcpyKind kind,
                                        aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMemcpyAsyncWithCondition(dst, destMax, src, count, kind, stream);
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free_size, size_t *total) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetMemInfo(attr, free_size, total);
}

const char* aclrtGetSocName() {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetSocName();
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetDeviceInfo(deviceId, attr, value);
}

aclError aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t *const phyDevId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetPhyDevIdByLogicDevId(logicDevId, phyDevId);
}

aclError aclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes, size_t numBatches,
                          aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexex, size_t numAttrs, size_t *failIndex)
{
  return ge::AclRuntimeStub::GetInstance()->aclrtMemcpyBatch(dsts, destMax, srcs, sizes, numBatches,
                                                              attrs, attrsIndexex, numAttrs, failIndex);
}

aclError aclrtCheckArchCompatibility(const char *socVersion, int32_t *canCompatible) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCheckArchCompatibility(socVersion, canCompatible);
}

aclError aclrtDestroyContext(aclrtContext context) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyContext(context);
}

aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetStreamFailureMode(stream, mode);
}

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtActiveStream(activeStream, stream);
}

aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCtxGetCurrentDefaultStream(stream);
}

aclError aclrtDestroyLabel(aclrtLabel label) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyLabel(label);
}

aclError aclmdlRIDestroy(aclmdlRI modelRI) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIDestroy(modelRI);
}

aclError aclmdlRIUnbindStream(aclmdlRI modelRI, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIUnbindStream(modelRI, stream);
}

aclError aclrtDestroyLabelList(aclrtLabelList labelList) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyLabelList(labelList);
}

aclError aclmdlRIBindStream(aclmdlRI modelRI, aclrtStream stream, uint32_t flag) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIBindStream(modelRI, stream, flag);
}

aclError aclmdlRIBuildEnd(aclmdlRI modelRI, void *reserve) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIBuildEnd(modelRI, reserve);
}

aclError aclrtPersistentTaskClean(aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtPersistentTaskClean(stream);
}

aclError aclrtSetTsDevice(aclrtTsId tsId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetTsDevice(tsId);
}

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateContext(context, deviceId);
}

aclError aclrtGetDeviceCount(uint32_t *count) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetDeviceCount(count);
}

aclError aclrtGetDeviceCapability(int32_t deviceId, aclrtDevFeatureType devFeatureType, int32_t *value) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetDeviceCapability(deviceId, devFeatureType, value);
}

aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateEventWithFlag(event, flag);
}

aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateEventExWithFlag(event, flag);
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtResetEvent(event, stream);
}

aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSynchronizeEventWithTimeout(event, timeout);
}

aclError aclrtMallocForTaskScheduler(void **devPtr, size_t size, aclrtMemMallocPolicy policy,
                                     aclrtMallocConfig *cfg) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMallocForTaskScheduler(devPtr, size, policy, cfg);
}

aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr,
                                uint64_t flags) {
  return ge::AclRuntimeStub::GetInstance()->aclrtReserveMemAddress(virPtr, size, alignment, expectPtr, flags);
}

aclError aclrtReleaseMemAddress(void *virPtr) {
  return ge::AclRuntimeStub::GetInstance()->aclrtReleaseMemAddress(virPtr);
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop,
                             uint64_t flags) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMallocPhysical(handle, size, prop, flags);
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  return ge::AclRuntimeStub::GetInstance()->aclrtFreePhysical(handle);
}

aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
  return ge::AclRuntimeStub::GetInstance()->aclrtMapMem(virPtr, size, offset, handle, flags);
}

aclError aclrtUnmapMem(void *virPtr) {
  return ge::AclRuntimeStub::GetInstance()->aclrtUnmapMem(virPtr);
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  return ge::AclRuntimeStub::GetInstance()->aclrtStreamWaitEvent(stream, event);
}

aclError aclrtStreamWaitEventWithTimeout(aclrtStream stream, aclrtEvent event, int32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtStreamWaitEventWithTimeout(stream, event, timeout);
}

aclError aclrtSetOpWaitTimeout(uint32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetOpWaitTimeout(timeout);
}

aclError aclrtSetOpExecuteTimeOut(uint32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetOpExecuteTimeOut(timeout);
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetDeviceSatMode(mode);
}

aclError aclrtDeviceGetBareTgid(int32_t *pid) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDeviceGetBareTgid(pid);
}

aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetExceptionInfoCallback(callback);
}

uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetDeviceIdFromExceptionInfo(info);
}

uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetErrorCodeFromExceptionInfo(info);
}

aclError aclrtGetUserDevIdByLogicDevId(const int32_t logicDevId, int32_t *const userDevid) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetUserDevIdByLogicDevId(logicDevId, userDevid);
}

aclError aclrtSetOpExecuteTimeOutWithMs(uint32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetOpExecuteTimeOutWithMs(timeout);
}

aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t *actualTimeout) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetOpExecuteTimeOutV2(timeout, actualTimeout);
}

aclError aclrtGetStreamAvailableNum(uint32_t *streamCount) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetStreamAvailableNum(streamCount);
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetEventId(event, eventId);
}

aclError aclrtGetEventAvailNum(uint32_t *eventCount) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetEventAvailNum(eventCount);
}

aclError aclrtCreateLabel(aclrtLabel *label) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateLabel(label);
}

aclError aclrtSetLabel(aclrtLabel label, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSetLabel(label, stream);
}

aclError aclrtCreateLabelList(aclrtLabel *labels, size_t num, aclrtLabelList *labelList) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateLabelList(labels, num, labelList);
}

aclError aclrtSwitchLabelByIndex(void *ptr, uint32_t maxValue, aclrtLabelList labelList, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSwitchLabelByIndex(ptr, maxValue, labelList, stream);
}

aclError aclrtSwitchStream(void *leftValue, aclrtCondition cond, void *rightValue,
                           aclrtCompareDataType dataType, aclrtStream trueStream, aclrtStream falseStream,
                           aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtSwitchStream(leftValue, cond, rightValue, dataType, trueStream, falseStream, stream);
}

aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIExecuteAsync(modelRI, stream);
}

aclError aclmdlRIExecute(aclmdlRI modelRI, int32_t timeout) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIExecute(modelRI, timeout);
}

aclError aclmdlRIBuildBegin(aclmdlRI *modelRI, uint32_t flag) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIBuildBegin(modelRI, flag);
}

aclError aclmdlRIEndTask(aclmdlRI modelRI, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIEndTask(modelRI, stream);
}

aclError aclmdlRISetName(aclmdlRI modelRI, const char *name) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRISetName(modelRI, name);
}

aclError aclrtCtxGetFloatOverflowAddr(void **overflowAddr) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCtxGetFloatOverflowAddr(overflowAddr);
}

aclError aclrtGetHardwareSyncAddr(void **addr) {
  return ge::AclRuntimeStub::GetInstance()->aclrtGetHardwareSyncAddr(addr);
}

aclError aclrtTaskUpdateAsync(aclrtStream taskStream, uint32_t taskId, aclrtTaskUpdateInfo *info,
                              aclrtStream execStream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtTaskUpdateAsync(taskStream, taskId, info, execStream);
}

aclError aclmdlRIAbort(aclmdlRI modelRI) {
  return ge::AclRuntimeStub::GetInstance()->aclmdlRIAbort(modelRI);
}

aclError aclrtProfTrace(void *userdata, int32_t length, aclrtStream stream) {
  return ge::AclRuntimeStub::GetInstance()->aclrtProfTrace(userdata, length, stream);
}

aclError aclrtCreateNotify(aclrtNotify *notify, uint64_t flag) {
  return ge::AclRuntimeStub::GetInstance()->aclrtCreateNotify(notify, flag);
}

aclError aclrtDestroyNotify(aclrtNotify notify) {
  return ge::AclRuntimeStub::GetInstance()->aclrtDestroyNotify(notify);
}

aclError aclInit(const char *configPath) {
  return ge::AclApiStub::GetInstance()->aclInit(configPath);
}

aclError aclFinalize() {
  return ge::AclApiStub::GetInstance()->aclFinalize();
}

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
  return ge::AclApiStub::GetInstance()->aclCreateDataBuffer(data, size);
}

aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                  int numDims,
                                  const int64_t *dims,
                                  aclFormat format) {
  return ge::AclApiStub::GetInstance()->aclCreateTensorDesc(dataType, numDims, dims, format);
}

void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) {
  return ge::AclApiStub::GetInstance()->aclGetDataBufferAddr(dataBuffer);
}

size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer) {
  return ge::AclApiStub::GetInstance()->aclGetDataBufferSizeV2(dataBuffer);
}

aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize) {
  return ge::AclApiStub::GetInstance()->aclGetTensorDescDimV2(desc, index, dimSize);
}

aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc) {
  return ge::AclApiStub::GetInstance()->aclGetTensorDescFormat(desc);
}

size_t aclGetTensorDescNumDims(const aclTensorDesc *desc) {
  return ge::AclApiStub::GetInstance()->aclGetTensorDescNumDims(desc);
}

aclDataType aclGetTensorDescType(const aclTensorDesc *desc) {
  return ge::AclApiStub::GetInstance()->aclGetTensorDescType(desc);
}

aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer) {
  return ge::AclApiStub::GetInstance()->aclmdlAddDatasetBuffer(dataset, dataBuffer);
}

aclmdlConfigHandle *aclmdlCreateConfigHandle() {
  return ge::AclApiStub::GetInstance()->aclmdlCreateConfigHandle();
}

aclmdlDataset *aclmdlCreateDataset() {
  return ge::AclApiStub::GetInstance()->aclmdlCreateDataset();
}

aclmdlDesc *aclmdlCreateDesc() {
  return ge::AclApiStub::GetInstance()->aclmdlCreateDesc();
}

aclError aclmdlDestroyDataset(const aclmdlDataset *dataset) {
  return ge::AclApiStub::GetInstance()->aclmdlDestroyDataset(dataset);
}

aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc) {
  return ge::AclApiStub::GetInstance()->aclmdlDestroyDesc(modelDesc);
}

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output) {
  return ge::AclApiStub::GetInstance()->aclmdlExecute(modelId, input, output);
}

aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index) {
  return ge::AclApiStub::GetInstance()->aclmdlGetDatasetBuffer(dataset, index);
}

aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index) {
  return ge::AclApiStub::GetInstance()->aclmdlGetDatasetTensorDesc(dataset, index);
}

aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
  return ge::AclApiStub::GetInstance()->aclmdlGetDesc(modelDesc, modelId);
}

aclError aclmdlLoadFromMem(const void *model,  size_t modelSize, uint32_t *modelId) {
  return ge::AclApiStub::GetInstance()->aclmdlLoadFromMem(model,  modelSize, modelId);
}

aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId) {
  return ge::AclApiStub::GetInstance()->aclmdlLoadWithConfig(handle, modelId);
}

aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
                                  const void *attrValue, size_t valueSize) {
  return ge::AclApiStub::GetInstance()->aclmdlSetConfigOpt(handle, attr, attrValue, valueSize);
}

aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset, aclTensorDesc *tensorDesc, size_t index) {
  return ge::AclApiStub::GetInstance()->aclmdlSetDatasetTensorDesc(dataset, tensorDesc, index);
}

aclError aclmdlSetExternalWeightAddress(aclmdlConfigHandle *handle, const char *weightFileName,
                                        void *devPtr, size_t size) {
  return ge::AclApiStub::GetInstance()->aclmdlSetExternalWeightAddress(handle, weightFileName, devPtr, size);
}

aclError aclmdlUnload(uint32_t modelId) {
  return ge::AclApiStub::GetInstance()->aclmdlUnload(modelId);
}

aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle) {
  return ge::AclApiStub::GetInstance()->aclmdlDestroyConfigHandle(handle);
}

void aclDestroyTensorDesc(const aclTensorDesc *desc) {
  return ge::AclApiStub::GetInstance()->aclDestroyTensorDesc(desc);
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
  return ge::AclApiStub::GetInstance()->aclDestroyDataBuffer(dataBuffer);
}

const char* acldumpGetPath(acldumpType dumpType) {
  return ge::AclApiStub::GetInstance()->acldumpGetPath(dumpType);
}
#ifdef __cplusplus
}
#endif