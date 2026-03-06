/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/runtime_stub_impl.h"
#include "stub/acl_runtime_stub_impl.h"
#include <algorithm>
#include <securec.h>
#include <iostream>
#include "mmpa/mmpa_api.h"

namespace gert {
void RuntimeStubImpl::Clear() {
  launch_with_handle_args_.clear();
  all_launch_args_.clear();
  all_switch_args_.clear();
  cpu_launch_args_.clear();
  rt_memcpy_args_.clear();
  rt_memcpy_sync_args_.clear();
  all_launch_sqe_update_records_.clear();
  events_to_record_records_.clear();
}

const std::map<const void *, HandleArgsPtrList> &RuntimeStubImpl::GetLaunchWithHandleArgs() {
  return launch_with_handle_args_;
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopLaunchArgsBy(const void *handle) {
  auto &args_list = launch_with_handle_args_[handle];
  if (args_list.empty()) {
    return nullptr;
  }
  auto ret = args_list.front();
  args_list.pop_front();
  return ret;
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopLaunchArgsByStubFunc(const void *stubFunc) {
  return PopLaunchArgsBy(stubFunc);
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopLaunchArgsByStubName(const std::string &stub_name) {
  auto handle = reinterpret_cast<void *>(&stub_names_to_handles_[stub_name]);
  return PopLaunchArgsBy(handle);
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopLaunchArgsByBinary(const rtDevBinary_t *bin) {
  void *handle = &bin_data_to_handles_[BinData(bin)];
  return PopLaunchArgsBy(handle);
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopLaunchArgsByBinary(const rtDevBinary_t *bin, uint64_t devFunc) {
  void *handle = &bin_data_to_handles_[BinData(bin)];
  return PopLaunchArgsBy(handle, devFunc);
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopLaunchArgsBy(const void *handle, uint64_t devFunc) {
  auto &args_list = launch_with_handle_args_[handle];
  if (args_list.empty()) {
    return nullptr;
  }
  auto iter = std::find_if(args_list.begin(), args_list.end(),
                           [&](ge::GeFakeLaunchArgs *args) { return args->GetDevFun() == devFunc; });
  if (iter == args_list.end()) {
    return nullptr;
  }
  auto ret = *iter;
  args_list.erase(iter);
  return ret;
}

ge::GeFakeLaunchArgs *RuntimeStubImpl::PopCpuLaunchArgsByKernelName(const std::string &kernel_name) {
  auto &args_list = cpu_launch_args_[kernel_name];
  if (args_list.empty()) {
    return nullptr;
  }
  auto iter = args_list.end();
  --iter;
  auto ret = *iter;
  args_list.erase(iter);
  return ret;
}
uintptr_t RuntimeStubImpl::FindSrcAddrCpyToDst(uintptr_t dst_addr) {
  return dst_addrs_to_src_addrs_[dst_addr];
}

rtError_t RuntimeStubImpl::rtKernelLaunchWithHandle(void *handle, uint64_t devFunc, uint32_t blockDim, rtArgsEx_t *args,
                                                    rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(handle, devFunc, blockDim, args, smDesc, stream, kernelInfo, std::move(last_tag_));
  launch_with_handle_args_[handle].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }

  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtKernelLaunchWithHandle, stream);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                                  rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(stubFunc, blockDim, argsInfo, smDesc, stream, flag, std::move(last_tag_));
  launch_with_handle_args_[stubFunc].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }

  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtKernelLaunchWithFlag, stream);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
                                                      rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                                      const rtTaskCfgInfo_t *cfgInfo) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo, std::move(last_tag_));
  launch_with_handle_args_[hdl].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stm));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }

  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtKernelLaunchWithHandleV2, stm);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                                    rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags,
                                                    const rtTaskCfgInfo_t *cfgInfo) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(stubFunc, blockDim, argsInfo, smDesc, stm, flags, std::move(last_tag_));
  launch_with_handle_args_[stubFunc].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stm));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }

  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtKernelLaunchWithFlagV2, stm);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launch_names, uint32_t blockDim,
                                                       const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                                       uint32_t flags) {
  const std::lock_guard<std::mutex> lk(mtx_);
  // todo : 当前只使用了args信息，其余参数未使用
  all_launch_args_.emplace_back(launch_names, blockDim, args, smDesc, stream, flags, std::move(last_tag_));
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }

  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtAicpuKernelLaunchWithFlag, stream);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtGetFunctionByName(const char *stub_name, void **stub_func) {
  *stub_func = &stub_names_to_handles_[std::string(stub_name)];
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) {
  *handle = &bin_data_to_handles_[BinData(bin)];
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtCpuKernelLaunchWithFlag(const void *so_name, const void *kernel_name, uint32_t block_dim,
                                                     const rtArgsEx_t *args, rtSmDesc_t *smDesc, rtStream_t stream,
                                                     uint32_t flags) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(kernel_name, block_dim, args->args, args->argsSize, stream, std::move(last_tag_));
  std::string kernel_name_str = reinterpret_cast<const char *>(kernel_name);
  cpu_launch_args_[kernel_name_str].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }

  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtGeneralCtrl(uintptr_t *ctrl, uint32_t num, uint32_t type) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(ctrl, num, type, std::move(last_tag_));
  cpu_launch_args_["test_haha"].emplace_back(&all_launch_args_.back());
  // task_id_++;
  return RT_ERROR_NONE;
}

const std::vector<ge::GeFakeRtMemcpyArgs> &RuntimeStubImpl::GetRtMemcpyRecords() const {
  return rt_memcpy_args_;
}

const std::vector<ge::GeFakeRtMemcpyArgs> &RuntimeStubImpl::GetRtMemcpySyncRecords() const {
  return rt_memcpy_sync_args_;
}

rtError_t RuntimeStubImpl::rtMemcpy(void *dst, uint64_t dest_max, const void *src, uint64_t count,
                                    rtMemcpyKind_t kind) {
  if (dst != nullptr && src != nullptr) {
    memcpy_s(dst, dest_max, src, count);
  }
  const std::lock_guard<std::mutex> lk(mtx_);
  dst_addrs_to_src_addrs_[reinterpret_cast<uintptr_t>(dst)] = reinterpret_cast<uintptr_t>(src);
  rt_memcpy_sync_args_.emplace_back(ge::GeFakeRtMemcpyArgs::RtMemcpy(dst, dest_max, src, count));
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src, uint64_t count,
                                         rtMemcpyKind_t kind, rtStream_t stream) {
  const std::lock_guard<std::mutex> lk(mtx_);
  rt_memcpy_args_.emplace_back(ge::GeFakeRtMemcpyArgs::RtMemcpyAsync(dst, dest_max, src, count, stream));
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtMemcpyAsync, stream);
  return RuntimeStub::rtMemcpyAsync(dst, dest_max, src, count, kind, stream);
}

rtError_t RuntimeStubImpl::rtsMemcpyBatch(void **dsts, void **srcs, size_t *sizes, size_t count,
    rtMemcpyBatchAttr *attrs, size_t *attrs_idxs, size_t num_attrs, size_t *fail_idx) {
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtMemcpyAsyncPtr(void *memcpy_addr_info, uint64_t dst_max, uint64_t count,
                                            rtMemcpyKind_t kind, rtStream_t stream, uint32_t qos_cfg) {
  const std::lock_guard<std::mutex> lk(mtx_);
  char soc_version[32] = {};
  mmGetEnv("MOCK_SOC_VERSION", &soc_version[0], sizeof(soc_version));

  // RTS demands the very memory is aligned to 64-byte boundary.
  if (reinterpret_cast<uintptr_t>(memcpy_addr_info) & (64 - 1)) {
    return -1;
  }

  // Simulate what RTS does.
  constexpr uint64_t magic = 0xdeadbeef'deadbeef; // See MemcpyAddrAsyncArgsParser.
  const auto u64 = strcmp(soc_version, "Ascend910D") == 0 ? std::vector<uint64_t>(8, magic)
                                                          : std::vector<uint64_t>(4, magic);
  memcpy(memcpy_addr_info, u64.data(), u64.size() * sizeof(uint64_t));

  all_launch_args_.emplace_back(memcpy_addr_info, std::move(last_tag_));
  return RuntimeStub::rtMemcpyAsyncPtr(memcpy_addr_info, dst_max, count, kind, stream, qos_cfg);
}

rtError_t RuntimeStubImpl::rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                                            rtStream_t stream, rtSwitchDataType_t data_type) {
  const std::lock_guard<std::mutex> lk(mtx_);

  all_switch_args_.emplace_back(ptr, value_ptr, std::move(last_tag_));
  return RuntimeStub::rtStreamSwitchEx(ptr, condition, value_ptr, true_stream, stream, data_type);
}

rtError_t RuntimeStubImpl::rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) {
  *free = 128UL * 1024UL * 1024UL;
  *total = 128UL * 1024UL * 1024UL;
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
  const std::lock_guard<std::mutex> lk(mtx_);
  auto ret = RuntimeStub::rtMalloc(dev_ptr, size, type, moduleId);
  if (ret == RT_ERROR_NONE) {
    addrs_to_mem_info_[*dev_ptr] = MemoryInfo{*dev_ptr, size, type, moduleId};
  }
  return ret;
}
rtError_t RuntimeStubImpl::rtFree(void *dev_ptr) {
  const std::lock_guard<std::mutex> lk(mtx_);
  auto iter = addrs_to_mem_info_.find(dev_ptr);
  if (iter != addrs_to_mem_info_.end()) {
    addrs_to_mem_info_.erase(iter);
  }
  return RuntimeStub::rtFree(dev_ptr);
}
rtError_t RuntimeStubImpl::rtLaunchSqeUpdateTask(uint32_t streamId, uint32_t taskId, void *src, uint64_t cnt,
                                                 rtStream_t stm) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_sqe_update_records_.emplace_back(ge::GeLaunchSqeUpdateTaskArgs{streamId, taskId, src, cnt, stm});
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtLaunchSqeUpdateTask, stm);
  for (auto &launch_arg : all_launch_args_) {
    if (launch_arg.GetStreamId() != streamId || launch_arg.GetTaskId() != taskId ||
        launch_arg.GetType() != RT_GNL_CTRL_TYPE_STARS_TSK_FLAG) {
      continue;
    }
    launch_arg.SetArgsAddr(src);
  }
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtSetTaskTag(const char *taskTag) {
  last_tag_ = std::make_unique<std::string>(taskTag);
  return RT_ERROR_NONE;
}

const std::list<ge::GeFakeLaunchArgs> &RuntimeStubImpl::GetAllLaunchArgs() const {
  return all_launch_args_;
}

const std::list<ge::GetAllSwitchArgs> &RuntimeStubImpl::GetAllSwitchArgs() const {
  return all_switch_args_;
}

rtError_t RuntimeStubImpl::rtModelCreate(rtModel_t *model, uint32_t flag) {
  const std::lock_guard<std::mutex> lk(mtx_);

  *model = new uint32_t;
  stream_to_task_id_.clear();

  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtModelGetTaskId(void *handle, uint32_t *task_id, uint32_t *stream_id) {
  const std::lock_guard<std::mutex> lk(mtx_);

  *task_id = stream_to_task_id_[last_stream_];
  *stream_id = static_cast<uint32_t>(last_stream_);

  // 更新最新的launch args的streamid 和taskid
  all_launch_args_.back().SetStreamId(*stream_id);
  all_launch_args_.back().SetTaskId(*task_id);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsStreamGetId(void *stm, int32_t *streamId)
{
  const std::lock_guard<std::mutex> lk(mtx_);
  (void) stm;

  *streamId = static_cast<uint32_t>(last_stream_);

  if (!all_launch_args_.empty()) {
    all_launch_args_.back().SetStreamId(*streamId);
  }
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsSetStreamResLimit(rtStream_t stm, const rtDevResLimitType_t type, const uint32_t value) {
  (void) stm;
  (void) type;
  (void) value;
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsUseStreamResInCurrentThread(const rtStream_t stm) {
  (void) stm;
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsGetThreadLastTaskId(uint32_t *taskId)
{
  const std::lock_guard<std::mutex> lk(mtx_);

  *taskId = stream_to_task_id_[last_stream_];

  auto it = stream_to_task_id_.find(last_stream_);
  if (it != stream_to_task_id_.end()) {
    *taskId = it->second;
  } else {
    *taskId = 0;
  }

  if (!all_launch_args_.empty()) {
    all_launch_args_.back().SetTaskId(*taskId);
  }
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsDeviceGetCapability(int32_t deviceId, int32_t devFeatureType, int32_t *val)
{
  (void) deviceId;
  (void) devFeatureType;
  *val = 16;
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtSetExceptionExtInfo(const rtArgsSizeInfo_t * const sizeInfo) {
  lite_exception_args_.emplace_back(reinterpret_cast<uintptr_t>(sizeInfo->infoAddr));
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtEventRecord(rtEvent_t event, rtStream_t stream) {
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtEventRecord, stream);
  events_to_record_records_[event].emplace_back(stream);
  event_stub_.LaunchEventRecordToStream(event, stream);
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) {
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtStreamWaitEvent, stream);
  event_stub_.LaunchEventWaitToStream(event, stream);
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtStreamCreate(rtStream_t *stream, int32_t priority) {
  stream_stub_.CreateStream(stream);
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) {
  stream_stub_.CreateStream(stream);
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtGetAvailStreamNum(uint32_t streamType, uint32_t *const streamCount) {
  const char *const kEnvRecordPath = "MOCK_AVAIL_STREAM_NUM";
  char record_path[8] = {};
  int32_t ret = mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(8));
  if ((ret != EN_OK) || (strlen(record_path) == 0)) {
    *streamCount = 2048;
    return RT_ERROR_NONE;
  }
  try {
    *streamCount = std::stoi(std::string(record_path));
    return RT_ERROR_NONE;
  } catch (...) {
  }
  *streamCount = 2048;
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtStreamDestroyForce(rtStream_t stream) {
  stream_stub_.DestoryStream(stream);
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtStreamDestroy(rtStream_t stream) {
  stream_stub_.DestoryStream(stream);
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag) {
  return RuntimeStub::rtEventCreateWithFlag(event, flag);
}

rtError_t RuntimeStubImpl::rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) {
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtModelExecute, stream);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtStreamTaskClean(rtStream_t stm) {
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stm));
  stream_to_task_id_.erase(last_stream_);
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsBinaryLoadFromFile(const char * const binPath, const rtLoadBinaryConfig_t *const optionalCfg,
                                                 rtBinHandle *binHandle)
{
  uint64_t stub_bin_addr = 0x1200;
  *binHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_bin_addr));
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtsFuncGetByName(const rtBinHandle binHandle, const char *kernelName,
                                            rtFuncHandle *funcHandle)
{
  uint64_t stub_func_addr = 0x1600;
  *funcHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_func_addr));
  return RT_ERROR_NONE;
}
rtError_t RuntimeStubImpl::rtsLaunchCpuKernel(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t st,
                                              const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo)
{
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(funcHandle, blockDim, st, cfg, argsInfo, std::move(last_tag_));
  cpu_launch_args_["cpu_new_args_launch"].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(st));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                                       rtKernelLaunchCfg_t *cfg, void *hostArgs, uint32_t argsSize,
                                                       rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum)
{
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(funcHandle, blockDim, stm, cfg, hostArgs, argsSize,
                                placeHolderArray, placeHolderNum, std::move(last_tag_));
  cpu_launch_args_["cpu_new_args_launch_with_place_holder"].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stm));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsLaunchKernelWithDevArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                                      rtKernelLaunchCfg_t *cfg, const void *args, uint32_t argsSize, void *reserve)
{
  const std::lock_guard<std::mutex> lk(mtx_);
  all_launch_args_.emplace_back(funcHandle, blockDim, stm, cfg, args, argsSize,
                                reserve, std::move(last_tag_));
  cpu_launch_args_["cpu_new_args_launch_with_place_holder"].emplace_back(&all_launch_args_.back());
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stm));
  if (stream_to_task_id_.find(last_stream_) != stream_to_task_id_.end()) {
    stream_to_task_id_[last_stream_]++;
  } else {
    stream_to_task_id_[last_stream_] = 0;
  }
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsBinaryLoadFromData(const void * const data, const uint64_t length,
                                                 const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle)
{
  uint64_t stub_bin_addr = 0x1200;
  *handle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_bin_addr));
  return RT_ERROR_NONE;
}

rtError_t RuntimeStubImpl::rtsRegisterCpuFunc(const rtBinHandle binHandle, const char_t * const funcName,
                                              const char_t * const kernelName, rtFuncHandle *funcHandle)
{
  uint64_t stub_func_addr = 0x1600;
  *funcHandle = reinterpret_cast<void *>(static_cast<uintptr_t>(stub_func_addr));
  return RT_ERROR_NONE;
}

// -----------------AclRuntimeStubImpl-----------------
void AclRuntimeStubImpl::Clear() {
  launch_with_handle_args_.clear();
  all_launch_args_.clear();
  all_switch_args_.clear();
  cpu_launch_args_.clear();
  rt_memcpy_args_.clear();
  rt_memcpy_sync_args_.clear();
  all_launch_sqe_update_records_.clear();
  events_to_record_records_.clear();
}

AclRuntimeStubImpl::AclRuntimeStubImpl() {
  const std::lock_guard<std::mutex> lk(mtx_);
  last_stream_ = 0UL;
  all_launch_args_.clear();
  launch_with_handle_args_.clear();
}

const std::map<const void *, HandleArgsPtrList> &AclRuntimeStubImpl::GetLaunchWithHandleArgs() {
  return launch_with_handle_args_;
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopLaunchArgsBy(const void *handle) {
  auto &args_list = launch_with_handle_args_[handle];
  if (args_list.empty()) {
    return nullptr;
  }
  auto ret = args_list.front();
  args_list.pop_front();
  return ret;
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopLaunchArgsByStubFunc(const void *stubFunc) {
  return PopLaunchArgsBy(stubFunc);
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopLaunchArgsByStubName(const std::string &stub_name) {
  auto handle = reinterpret_cast<void *>(&stub_names_to_handles_[stub_name]);
  return PopLaunchArgsBy(handle);
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopLaunchArgsByBinary(const rtDevBinary_t *bin) {
  void *handle = &bin_data_to_handles_[BinData(bin)];
  return PopLaunchArgsBy(handle);
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopLaunchArgsByBinary(const rtDevBinary_t *bin, uint64_t devFunc) {
  void *handle = &bin_data_to_handles_[BinData(bin)];
  return PopLaunchArgsBy(handle, devFunc);
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopLaunchArgsBy(const void *handle, uint64_t devFunc) {
  auto &args_list = launch_with_handle_args_[handle];
  if (args_list.empty()) {
    return nullptr;
  }
  auto iter = std::find_if(args_list.begin(), args_list.end(),
                           [&](ge::GeFakeLaunchArgs *args) { return args->GetDevFun() == devFunc; });
  if (iter == args_list.end()) {
    return nullptr;
  }
  auto ret = *iter;
  args_list.erase(iter);
  return ret;
}

ge::GeFakeLaunchArgs *AclRuntimeStubImpl::PopCpuLaunchArgsByKernelName(const std::string &kernel_name) {
  auto &args_list = cpu_launch_args_[kernel_name];
  if (args_list.empty()) {
    return nullptr;
  }
  auto iter = args_list.end();
  --iter;
  auto ret = *iter;
  args_list.erase(iter);
  return ret;
}
uintptr_t AclRuntimeStubImpl::FindSrcAddrCpyToDst(uintptr_t dst_addr) {
  return dst_addrs_to_src_addrs_[dst_addr];
}

const std::vector<ge::GeFakeRtMemcpyArgs> &AclRuntimeStubImpl::GetRtMemcpyRecords() const {
  return rt_memcpy_args_;
}

const std::vector<ge::GeFakeRtMemcpyArgs> &AclRuntimeStubImpl::GetRtMemcpySyncRecords() const {
  return rt_memcpy_sync_args_;
}

const std::list<ge::GeFakeLaunchArgs> &AclRuntimeStubImpl::GetAllLaunchArgs() const {
  return all_launch_args_;
}

const std::list<ge::GetAllSwitchArgs> &AclRuntimeStubImpl::GetAllSwitchArgs() const {
  return all_switch_args_;
}

aclError AclRuntimeStubImpl::aclrtStreamGetId(aclrtStream stream, int32_t *streamId) {
  const std::lock_guard<std::mutex> lk(mtx_);
  (void) stream;

  *streamId = static_cast<uint32_t>(last_stream_);

  if (!all_launch_args_.empty()) {
    all_launch_args_.back().SetStreamId(*streamId);
  }
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtGetThreadLastTaskId(uint32_t *taskId) {
  const std::lock_guard<std::mutex> lk(mtx_);
  *taskId = stream_to_task_id_[last_stream_];

  auto it = stream_to_task_id_.find(last_stream_);
  if (it != stream_to_task_id_.end()) {
    *taskId = it->second;
  } else {
    *taskId = 0;
  }

  if (!all_launch_args_.empty()) {
    all_launch_args_.back().SetTaskId(*taskId);
  }
  return RT_ERROR_NONE;
}
aclError AclRuntimeStubImpl::aclrtPersistentTaskClean(aclrtStream stream) {
  last_stream_ = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
  stream_to_task_id_.erase(last_stream_);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtCreateStream(aclrtStream *stream) {
  stream_stub_.CreateStream(stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
  stream_stub_.CreateStream(stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtSwitchStream(void *leftValue, aclrtCondition cond, void *rightValue,
  aclrtCompareDataType dataType, aclrtStream trueStream, aclrtStream falseStream,
  aclrtStream stream) {
  const std::lock_guard<std::mutex> lk(mtx_);
  all_switch_args_.emplace_back(leftValue, rightValue, std::move(last_tag_));
  return AclRuntimeStub::aclrtSwitchStream(leftValue, cond, rightValue, dataType, trueStream, falseStream, stream);
}

aclError AclRuntimeStubImpl::aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtEventRecord, stream);
  events_to_record_records_[event].emplace_back(stream);
  event_stub_.LaunchEventRecordToStream(event, stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtStreamWaitEvent, stream);
  event_stub_.LaunchEventWaitToStream(event, stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream) {
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtModelExecute, stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtDestroyStream(aclrtStream stream) {
  stream_stub_.DestoryStream(stream);
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtDestroyStreamForce(aclrtStream stream) {
  stream_stub_.DestoryStream(stream);
  return ACL_SUCCESS;
}
aclError AclRuntimeStubImpl::aclrtGetStreamAvailableNum(uint32_t *streamCount) {
  const char *const kEnvRecordPath = "MOCK_AVAIL_STREAM_NUM";
  char record_path[8] = {};
  int32_t ret = mmGetEnv(kEnvRecordPath, &record_path[0], static_cast<uint32_t>(8));
  if ((ret != EN_OK) || (strlen(record_path) == 0)) {
    *streamCount = 2048;
    return ACL_SUCCESS;
  }
  try {
    *streamCount = std::stoi(std::string(record_path));
    return ACL_SUCCESS;
  } catch (...) {
  }
  *streamCount = 2048;
  return ACL_SUCCESS;
}

aclError AclRuntimeStubImpl::aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  if (dst != nullptr && src != nullptr) {
    memcpy_s(dst, destMax, src, count);
  }
  const std::lock_guard<std::mutex> lk(mtx_);
  dst_addrs_to_src_addrs_[reinterpret_cast<uintptr_t>(dst)] = reinterpret_cast<uintptr_t>(src);
  rt_memcpy_sync_args_.emplace_back(ge::GeFakeRtMemcpyArgs::RtMemcpy(dst, destMax, src, count));
  return RT_ERROR_NONE;
}

aclError AclRuntimeStubImpl::aclrtMemcpyAsync(void *dst, size_t dest_max, const void *src, size_t src_count, aclrtMemcpyKind kind, aclrtStream stream) {
  const std::lock_guard<std::mutex> lk(mtx_);
  rt_memcpy_args_.emplace_back(ge::GeFakeRtMemcpyArgs::RtMemcpyAsync(dst, dest_max, src, src_count, stream));
  stream_stub_.LaunchTaskToStream(TaskTypeOnStream::rtMemcpyAsync, stream);
  return AclRuntimeStub::aclrtMemcpyAsync(dst, dest_max, src, src_count, kind, stream);
}

aclError AclRuntimeStubImpl::aclrtMalloc(void **dev_ptr, size_t size, aclrtMemMallocPolicy policy) {
  const std::lock_guard<std::mutex> lk(mtx_);
  auto ret = AclRuntimeStub::aclrtMalloc(dev_ptr, size, policy);
  if (ret == RT_ERROR_NONE) {
    addrs_to_mem_info_[*dev_ptr] = MemoryInfo{*dev_ptr, size, 0, 0};
  }
  return ret;
}

aclError AclRuntimeStubImpl::aclrtFree(void *dev_ptr) {
  const std::lock_guard<std::mutex> lk(mtx_);
  auto iter = addrs_to_mem_info_.find(dev_ptr);
  if (iter != addrs_to_mem_info_.end()) {
    addrs_to_mem_info_.erase(iter);
  }
  return AclRuntimeStub::aclrtFree(dev_ptr);
}

aclError AclRuntimeStubImpl::aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) {
  *free = 128UL * 1024UL * 1024UL;
  *total = 128UL * 1024UL * 1024UL;
  return ACL_SUCCESS;
}
}
