/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <iostream>
#include "aicpu_task_struct.h"
#include "op_mapping.pb.h"
#include "securec.h"
#include "stub/ge_fake_launch_args.h"
#include "graph/def_types.h"
using namespace std;


namespace ge {
GeFakeLaunchArgs::GeFakeLaunchArgs(void *handle, uint64_t devFunc, uint32_t blockDim, rtArgsEx_t *args,
                                   rtSmDesc_t *smDesc, rtStream_t stream, const void *kernelInfo,
                                   std::unique_ptr<std::string> tag)
    : handle_(handle),
      devFunc_(devFunc),
      blockDim_(blockDim),
      stream_(stream),
      kernelInfo_(kernelInfo),
      tag_name_(std::move(tag)) {
  Init(args, smDesc);
}
void GeFakeLaunchArgs::Init(const rtArgsEx_t *args, rtSmDesc_t *smDesc) {
  size_t data_size = sizeof(rtSmDesc_t) + sizeof(rtArgsEx_t) + args->argsSize + sizeof(rtArgsEx_t);
  args_holder_ = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[data_size]());
  smDesc_ = reinterpret_cast<rtSmDesc_t *>(args_holder_.get());
  args_ex_ = reinterpret_cast<rtArgsEx_t *>(args_holder_.get() + sizeof(rtSmDesc_t));
  if (smDesc != nullptr) {
    memcpy(smDesc_, smDesc, sizeof(rtSmDesc_t));
  }
  memcpy(args_ex_, args, sizeof(rtArgsEx_t));
  args_ex_->args = reinterpret_cast<void *>(args_ex_ + 1);
  memcpy(args_ex_->args, args->args, args->argsSize);

  args_ex_raw_ =
    reinterpret_cast<rtArgsEx_t *>(args_holder_.get() + sizeof(rtSmDesc_t)  + sizeof(rtArgsEx_t) + args->argsSize);
  memcpy(args_ex_raw_, args, sizeof(rtArgsEx_t));

  args_addr_ = args->args;
}

GeFakeLaunchArgs::GeFakeLaunchArgs(const void *handle, uint32_t blockDim, rtArgsEx_t *args, rtSmDesc_t *smDesc,
                                   rtStream_t stream, uint32_t flag, std::unique_ptr<std::string> tag)
    : handle_(handle),
      devFunc_(0),
      blockDim_(blockDim),
      stream_(stream),
      flag_(flag),
      tag_name_(std::move(tag)) {
  Init(args, smDesc);
}

GeFakeLaunchArgs::GeFakeLaunchArgs(const void *kernel_name, uint32_t block_dim, const void *args, uint32_t args_size,
                                   rtStream_t stream, std::unique_ptr<std::string> tag)
    : blockDim_(block_dim), stream_(stream), arg_size_(args_size), tag_name_(std::move(tag)) {
  if (args_size >= sizeof(aicpu::AicpuParamHead) + sizeof(uint64_t) * 2) {
    kernel_name_ = reinterpret_cast<const char *>(kernel_name);
    size_t args_pos = 0UL;
    args_pos += sizeof(aicpu::AicpuParamHead);
    auto cpu_args1 = *reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(args) + args_pos);
    args_pos += sizeof(uint64_t);
    auto cpu_args2 = *reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(args) + args_pos);
    auto mem = reinterpret_cast<void *>(cpu_args1);
    auto mem_size = *reinterpret_cast<uint64_t *>(cpu_args2);
    char serialize_dump_info[mem_size + 1];
    serialize_dump_info[mem_size] = '\0';
    (void)memcpy_s(serialize_dump_info, mem_size + 1, mem, mem_size);
    std::string tmp(serialize_dump_info, mem_size);
    serialize_dump_info_ = tmp;
  }
}

GeFakeLaunchArgs::GeFakeLaunchArgs(uintptr_t *ctrl, uint32_t num, uint32_t type, std::unique_ptr<std::string> tag)
    : type_(type), tag_name_(std::move(tag)) {
  if (type == RT_GNL_CTRL_TYPE_MEMCPY_ASYNC_CFG) {
    args_addr_ = reinterpret_cast<void *>(*(ctrl + 2));
    return;
  }

  if (num != 3 || type != RT_GNL_CTRL_TYPE_FFTS_PLUS_FLAG) {
    return;
  }

  rtFftsPlusTaskInfo_t *fftsPlusTaskInfo = reinterpret_cast<rtFftsPlusTaskInfo_t *>(*ctrl);
  const auto dump_info = fftsPlusTaskInfo->fftsPlusDumpInfo.loadDumpInfo;
  const auto dump_len = fftsPlusTaskInfo->fftsPlusDumpInfo.loadDumpInfolen;
  serialize_dump_info_ = std::string((char *)dump_info, dump_len);
}

GeFakeLaunchArgs::GeFakeLaunchArgs(const rtKernelLaunchNames_t *launch_names,
                                   uint32_t blockDim, const rtArgsEx_t *args,
                                   rtSmDesc_t *smDesc,
                                   rtStream_t stream, uint32_t flag, std::unique_ptr<std::string> tag)
    : devFunc_(0),
      blockDim_(blockDim),
      stream_(stream),
      flag_(flag),
      tag_name_(std::move(tag)) {
  (void)launch_names; // todo :launch_names当前未使用，没有保存相关信息
  Init(args, smDesc);
}

GeFakeLaunchArgs::GeFakeLaunchArgs(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t st,
                                   const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo,
                                   std::unique_ptr<std::string> tag)
    : blockDim_(blockDim), stream_(st), arg_size_(argsInfo->baseArgs.argsSize), tag_name_(std::move(tag)) {
  (void)cfg;
  (void)funcHandle;
}

GeFakeLaunchArgs::GeFakeLaunchArgs(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t stm,
                                   const rtKernelLaunchCfg_t *cfg, void *hostArgs, uint32_t argsSize,
                                   rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum,
                                   std::unique_ptr<std::string> tag)
    : blockDim_(blockDim), stream_(stm), arg_size_(argsSize), tag_name_(std::move(tag)) {
  (void)cfg;
  (void)funcHandle;
  args_addr_ = hostArgs;
  (void)placeHolderArray;
  (void)placeHolderNum;
}

GeFakeLaunchArgs::GeFakeLaunchArgs(const rtFuncHandle funcHandle, const uint32_t blockDim, rtStream_t stm,
                                   const rtKernelLaunchCfg_t *cfg, const void *devArgs, uint32_t argsSize,
                                   void *reserve, std::unique_ptr<std::string> tag)
    : blockDim_(blockDim), stream_(stm), arg_size_(argsSize), tag_name_(std::move(tag)) {
  (void)cfg;
  (void)funcHandle;
  args_addr_ = const_cast<void*>(devArgs);
  (void)reserve;
}
GeFakeLaunchArgs::GeFakeLaunchArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
    aclrtStream stream, aclrtLaunchKernelCfg *cfg, void *hostArgs, size_t argsSize,
    aclrtPlaceHolderInfo *placeHolderArray, size_t placeHolderNum, std::unique_ptr<std::string> tag)
    : blockDim_(numBlocks), stream_(stream), arg_size_(argsSize), tag_name_(std::move(tag)) {
  (void)cfg;
  (void)funcHandle;
  args_addr_ = hostArgs;
  (void)placeHolderArray;
  (void)placeHolderNum;
}

GeFakeLaunchArgs::GeFakeLaunchArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
    const void *argsData, size_t argsSize, aclrtLaunchKernelCfg *cfg,
    aclrtStream stream, std::unique_ptr<std::string> tag)
    : blockDim_(numBlocks), stream_(stream), arg_size_(argsSize), tag_name_(std::move(tag)) {
  (void)cfg;
  (void)funcHandle;
  args_addr_ = const_cast<void*>(argsData);
}
}  // namespace ge
