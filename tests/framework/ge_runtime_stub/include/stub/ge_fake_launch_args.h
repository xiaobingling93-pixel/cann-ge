/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BB00A46635AE45C5A244CDAF3274BB8B_H
#define BB00A46635AE45C5A244CDAF3274BB8B_H

#include <stdint.h>
#include "runtime_stub.h"
#include "acl/acl_rt.h"
#include <memory>

namespace ge {
struct GeFakeLaunchArgs {
  GeFakeLaunchArgs(void *handle, uint64_t devFunc, uint32_t blockDim, rtArgsEx_t *args,rtSmDesc_t *smDesc,
                   rtStream_t stream, const void *kernelInfo, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(const void *handle, uint32_t blockDim, rtArgsEx_t *args,
                   rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(const void *kernel_name, uint32_t block_dim, const void *args, uint32_t args_size,
                   rtStream_t stream, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(uintptr_t *ctrl, uint32_t num, uint32_t type, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(const rtKernelLaunchNames_t *launch_names, uint32_t blockDim, const rtArgsEx_t *args,
                   rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(void *args_addr, std::unique_ptr<std::string> tag)
    : args_addr_(args_addr), tag_name_(std::move(tag)) {}
  GeFakeLaunchArgs(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t st,
                   const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t stm,
                   const rtKernelLaunchCfg_t *cfg, void *hostArgs, uint32_t argsSize,
                   rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t stm,
                   const rtKernelLaunchCfg_t *cfg, const void *devArgs, uint32_t argsSize,
                   void *reserve, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
      aclrtStream stream, aclrtLaunchKernelCfg *cfg, void *hostArgs, size_t argsSize,
      aclrtPlaceHolderInfo *placeHolderArray, size_t placeHolderNum, std::unique_ptr<std::string> tag);
  GeFakeLaunchArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
      const void *argsData, size_t argsSize, aclrtLaunchKernelCfg *cfg,
      aclrtStream stream, std::unique_ptr<std::string> tag);
  uint64_t GetDevFun() const {
    return devFunc_;
  }

  rtStream_t GetStream() const {
    return stream_;
  }

  std::string GetKernelName() const {
    return kernel_name_;
  }

  std::string GetSerializeDumpInfo() const {
    return serialize_dump_info_;
  }

  size_t GetArgSize() const {
    return arg_size_;
  }

  const void **GetLaunchAddresses() const {
    return static_cast<const void **>(GetArgsEx()->args);
  }

  const rtArgsEx_t* GetArgsEx() const {
    return args_ex_;
  }

  const rtArgsEx_t* GetArgsExRaw() const {
    return args_ex_raw_;
  }

  template<class T>
  const T* GetArgsTilingData() const{
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(args_ex_->args) + args_ex_->tilingDataOffset);
  }
  const std::string *GetTag() const {
    return tag_name_.get();
  }

  void SetTaskId(uint32_t task_id) {
    task_id_ = task_id;
  }

  void SetStreamId(uint32_t stream_id) {
    stream_id_ = stream_id;
  }

  uint32_t GetTaskId() const {
    return task_id_;
  }

  uint32_t GetStreamId() const {
    return stream_id_;
  }

  uint32_t GetType() const {
    return type_;
  }

  void *GetArgsAddr() const {
    return args_addr_;
  }

  void SetArgsAddr(void *addr) {
    args_addr_ = addr;
  }

  private:
  void Init(const rtArgsEx_t *args, rtSmDesc_t *smDesc);
  void *args_addr_{nullptr};

 private:
  const void *handle_;
  uint64_t devFunc_;
  uint32_t blockDim_;
  rtArgsEx_t *args_ex_;
  rtArgsEx_t *args_ex_raw_{nullptr};
  rtSmDesc_t *smDesc_;
  rtStream_t stream_;
  const void *kernelInfo_;
  uint32_t  flag_;
  std::string serialize_dump_info_;
  size_t arg_size_{0UL};
  std::string kernel_name_;
  std::unique_ptr<std::string> tag_name_;
  uint32_t stream_id_;
  uint32_t task_id_;
  uint32_t type_ = UINT32_MAX;
 private:
  std::unique_ptr<uint8_t[]> args_holder_;
};

struct GeLaunchSqeUpdateTaskArgs {
  uint32_t stream_id;
  uint32_t task_id;
  void *src;
  uint64_t cnt;
  rtStream_t stm;
};
}  // namespace gert

#endif
