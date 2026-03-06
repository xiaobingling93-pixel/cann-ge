/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_RUNTIME_STUB_IMPL_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_RUNTIME_STUB_IMPL_H_
#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

#include "ge_fake_launch_args.h"
#include "ge_fake_rtmemcpy_args.h"
#include "ge_fake_rtswitch_args.h"
#include "runtime_stub.h"

namespace gert {
enum class TaskTypeOnStream {
  rtKernelLaunchWithHandle = 0,
  rtKernelLaunchWithFlag,
  rtKernelLaunchWithHandleV2,
  rtKernelLaunchWithFlagV2,
  rtAicpuKernelLaunchWithFlag,
  rtCpuKernelLaunch,
  rtMemcpyAsync,
  rtLaunchSqeUpdateTask,
  rtEventRecord,
  rtStreamWaitEvent,
  rtModelExecute,
  // add before
  kEnd
};
struct GertStreamStub {
  std::vector<rtStream_t> rt_streams;
  std::unordered_map<rtStream_t, size_t> rt_stream_2_index;
  std::unordered_map<rtStream_t, std::vector<TaskTypeOnStream>> streams_2_all_tasks;

  void LaunchTaskToStream(TaskTypeOnStream task_type, rtStream_t stream) {
    const std::lock_guard<std::mutex> lk(mtx_);
    streams_2_all_tasks[stream].emplace_back(task_type);
  }
  std::vector<TaskTypeOnStream> GetAllTaskOfStream(rtStream_t stream) {
    return streams_2_all_tasks[stream];
  }
  std::vector<rtStream_t> GetAllRtStreams() {
    return rt_streams;
  }
  void CreateStream(rtStream_t *stream) {
    const std::lock_guard<std::mutex>lock(mtx_);
    *stream = reinterpret_cast<rtStream_t>(++g_rtstream_id);
    rt_streams.emplace_back(*stream);
    rt_stream_2_index.insert(std::make_pair(*stream, rt_streams.size() - 1));
  }
  void DestoryStream(rtStream_t stream) {
    const std::lock_guard<std::mutex>lock(mtx_);
    if (rt_stream_2_index.find(stream) == rt_stream_2_index.cend()) {
      return;
    }
    auto index = rt_stream_2_index[stream];
    rt_streams[index] = nullptr;
  }
  void Clear() {
    rt_streams.clear();
    streams_2_all_tasks.clear();
    g_rtstream_id = 0;
  }

 private:
  int64_t stream_num = 0;
  int64_t g_rtstream_id;
  std::mutex mtx_;
};

struct GertEventStub {
  std::vector<rtEvent_t> events;
  std::map<rtEvent_t, rtStream_t> events_to_send_records;
  std::map<rtEvent_t, rtStream_t> events_to_wait_records;
  void LaunchEventRecordToStream(rtEvent_t event, rtStream_t stream) {
    events_to_send_records[event] = stream;
  }
  void LaunchEventWaitToStream(rtEvent_t event, rtStream_t stream) {
    events_to_wait_records[event] = stream;
  }
  void Clear() {
    events.clear();
    events_to_send_records.clear();
    events_to_wait_records.clear();
    g_rtevent_id = 0;
  }

 private:
  int64_t g_rtevent_id;
  std::mutex mtx_;
};
class RuntimeStubImpl : public ge::RuntimeStub {
 public:
  struct MemoryInfo {
    void *addr;
    size_t size;
    rtMemType_t rts_mem_type;
    uint16_t model_id;
  };
  using HandleArgsPtrList = std::list<ge::GeFakeLaunchArgs *>;
  const std::map<const void *, HandleArgsPtrList> &GetLaunchWithHandleArgs();
  void Clear();

  ge::GeFakeLaunchArgs *PopLaunchArgsBy(const void *handle);
  ge::GeFakeLaunchArgs *PopLaunchArgsBy(const void *handle, uint64_t devFunc);
  ge::GeFakeLaunchArgs *PopLaunchArgsByStubFunc(const void *stubFunc);
  ge::GeFakeLaunchArgs *PopLaunchArgsByStubName(const std::string &stub_name);
  ge::GeFakeLaunchArgs *PopLaunchArgsByBinary(const rtDevBinary_t *bin);
  ge::GeFakeLaunchArgs *PopLaunchArgsByBinary(const rtDevBinary_t *bin, uint64_t devFunc);
  ge::GeFakeLaunchArgs *PopCpuLaunchArgsByKernelName(const std::string &kernel_name);
  uintptr_t FindSrcAddrCpyToDst(uintptr_t dst_addr);
  const std::vector<ge::GeFakeRtMemcpyArgs> &GetRtMemcpyRecords() const;
  const std::vector<ge::GeFakeRtMemcpyArgs> &GetRtMemcpySyncRecords() const;
  const std::list<ge::GeFakeLaunchArgs> &GetAllLaunchArgs() const;
  const std::list<ge::GetAllSwitchArgs> &GetAllSwitchArgs() const;

  const std::unordered_map<void *, MemoryInfo> &GetAllocatedRtsMemory() const {
    return addrs_to_mem_info_;
  }
  const std::list<ge::GeLaunchSqeUpdateTaskArgs> &GetLaunchSqeUpdateTaskArgs() const {
    return all_launch_sqe_update_records_;
  }

  const std::vector<uintptr_t> &GetLiteEceptionArgs() const {
    return lite_exception_args_;
  }
  const std::map<rtEvent_t, std::vector<rtStream_t>> &GetRtEventRecordRecords() const {
    return events_to_record_records_;
  }
  const std::vector<TaskTypeOnStream> GetAllTaskOnStream(rtStream_t stream) {
    return stream_stub_.GetAllTaskOfStream(stream);
  }
  const std::vector<rtStream_t> GetAllRtStreams() {
    return stream_stub_.GetAllRtStreams();
  }

  void LaunchTaskToStream(TaskTypeOnStream task_type, rtStream_t stream) {
    stream_stub_.LaunchTaskToStream(task_type, stream);
  }
 protected:
  rtError_t rtKernelLaunchWithHandle(void *handle, uint64_t devFunc, uint32_t blockDim, rtArgsEx_t *args,
                                     rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) override;

  rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                   rtStream_t stream, uint32_t flag) override;

  rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                       rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo) override;

  rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                     rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo) override;
  rtError_t rtGetFunctionByName(const char *stub_name, void **stub_func) override;
  rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) override;
  rtError_t rtCpuKernelLaunchWithFlag(const void *so_name, const void *kernel_name, uint32_t block_dim,
                                      const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                      uint32_t flags) override;
  rtError_t rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launch_names, uint32_t blockDim,
                                        const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                        uint32_t flags) override;
  rtError_t rtGeneralCtrl(uintptr_t *ctrl, uint32_t num, uint32_t type) override;

  rtError_t rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind) override;

  rtError_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count, rtMemcpyKind_t kind,
                          rtStream_t stream) override;

  rtError_t rtMemcpyAsyncPtr(void *memcpy_addr_info, uint64_t dst_max, uint64_t count, rtMemcpyKind_t kind,
                             rtStream_t stream, uint32_t qos_cfg) override;

  rtError_t rtsMemcpyBatch(void **dsts, void **srcs, size_t *sizes, size_t count,
    rtMemcpyBatchAttr *attrs, size_t *attrs_idxs, size_t num_attrs, size_t *fail_idx) override;

  rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override;

  rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                             rtStream_t stream, rtSwitchDataType_t data_type) override;

  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override;
  rtError_t rtFree(void *dev_ptr) override;
  rtError_t rtLaunchSqeUpdateTask(uint32_t streamId, uint32_t taskId, void *src, uint64_t cnt, rtStream_t stm) override;
  rtError_t rtModelCreate(rtModel_t *model, uint32_t flag) override;
  rtError_t rtModelGetTaskId(void *handle, uint32_t *task_id, uint32_t *stream_id) override;
  rtError_t rtsStreamGetId(void *stm, int32_t *streamId) override;
  rtError_t rtsGetThreadLastTaskId(uint32_t *taskId) override;
  rtError_t rtsDeviceGetCapability(int32_t deviceId, int32_t devFeatureType, int32_t *val) override;
  rtError_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t * const sizeInfo) override;

  rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) override;
  rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream) override;
  rtError_t rtStreamWaitEvent(rtEvent_t event, rtStream_t stream) override;
  rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) override;
  rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) override;
  rtError_t rtStreamDestroyForce(rtStream_t stream) override;
  rtError_t rtStreamDestroy(rtStream_t stream) override;
  rtError_t rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag) override;
  rtError_t rtStreamTaskClean(rtStream_t stm) override;
  rtError_t rtsBinaryLoadFromFile(const char * const binPath, const rtLoadBinaryConfig_t *const optionalCfg,
                                  rtBinHandle *binHandle) override;
  rtError_t rtsFuncGetByName(const rtBinHandle binHandle, const char *kernelName,
                             rtFuncHandle *funcHandle) override;
  rtError_t rtsLaunchCpuKernel(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t st,
                               const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo) override;
  rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                        rtKernelLaunchCfg_t *cfg, void *hostArgs, uint32_t argsSize,
                                        rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum) override;
  rtError_t rtsLaunchKernelWithDevArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                       rtKernelLaunchCfg_t *cfg, const void *args, uint32_t argsSize, void *reserve) override;
  rtError_t rtsBinaryLoadFromData(const void * const data, const uint64_t length,
                                  const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle) override;
  rtError_t rtsRegisterCpuFunc(const rtBinHandle binHandle, const char_t * const funcName,
                               const char_t * const kernelName, rtFuncHandle *funcHandle) override;
  rtError_t rtsSetStreamResLimit(rtStream_t stm, const rtDevResLimitType_t type, const uint32_t value) override;
  rtError_t rtsUseStreamResInCurrentThread(const rtStream_t stm) override;
 public:
  rtError_t rtSetTaskTag(const char *taskTag) override;
  rtError_t rtGetAvailStreamNum(uint32_t streamType, uint32_t *const streamCount) override;

 private:
  using BinHandle = uint64_t;
  std::map<std::string, BinHandle> stub_names_to_handles_;

  struct BinData {
    explicit BinData(const rtDevBinary_t *binary)
        : magic(binary->magic),
          version(binary->version),
          data(reinterpret_cast<uint64_t>(binary->data)),
          length(binary->length) {}
    bool operator<(const BinData &rht) const {
      return this->data < rht.data;
    }
    uint32_t magic;    // magic number
    uint32_t version;  // version of binary
    uint64_t data;     // binary data
    uint64_t length;   // binary length
  };

  std::map<BinData, BinHandle> bin_data_to_handles_;
  std::list<ge::GeFakeLaunchArgs> all_launch_args_;
  std::map<const void *, HandleArgsPtrList> launch_with_handle_args_;
  std::map<std::string, HandleArgsPtrList> cpu_launch_args_;
  std::map<uintptr_t, uintptr_t> dst_addrs_to_src_addrs_;
  std::vector<ge::GeFakeRtMemcpyArgs> rt_memcpy_args_;
  std::vector<ge::GeFakeRtMemcpyArgs> rt_memcpy_sync_args_;
  std::list<ge::GetAllSwitchArgs> all_switch_args_;
  std::unordered_map<void *, MemoryInfo> addrs_to_mem_info_;
  std::list<ge::GeLaunchSqeUpdateTaskArgs> all_launch_sqe_update_records_;
  std::mutex mtx_;
  std::unique_ptr<std::string> last_tag_;
  // uint32_t task_id_{0}; // 同一个模型的task在一条流上分配，只用来区分不同task
  uint64_t last_stream_;
  std::map<uint64_t, uint32_t> stream_to_task_id_;
  std::vector<uintptr_t> lite_exception_args_;
  std::map<rtEvent_t, std::vector<rtStream_t>> events_to_record_records_;
  GertStreamStub stream_stub_;
  GertEventStub event_stub_;
};
}  // namespace gert

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_RUNTIME_STUB_IMPL_H_
