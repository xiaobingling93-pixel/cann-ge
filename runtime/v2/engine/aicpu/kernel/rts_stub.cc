/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_stub.h"
#include "graph/def_types.h"

rtError_t rtsLaunchCpuKernel(const rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                             const rtKernelLaunchCfg_t *cfg, rtCpuKernelArgs_t *argsInfo)
{
  (void)funcHandle;
  (void)blockDim;
  (void)stm;
  (void)cfg;
  (void)argsInfo;
  return RT_ERROR_NONE;
}

rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm,
                                      rtKernelLaunchCfg_t *cfg, void *hostArgs, uint32_t argsSize,
                                      rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum)
{
  (void)funcHandle;
  (void)blockDim;
  (void)stm;
  (void)cfg;
  (void)hostArgs;
  (void)argsSize;
  (void)placeHolderArray;
  (void)placeHolderNum;
  return RT_ERROR_NONE;
}

rtError_t rtsBinaryLoadFromFile(const char_t * const binPath, const rtLoadBinaryConfig_t * const optionalCfg,
                                rtBinHandle *handle)
{
  (void)binPath;
  (void)optionalCfg;
  uint64_t stub_bin_addr = 0x1200;
  *handle = ge::ValueToPtr(stub_bin_addr);
  return RT_ERROR_NONE;
}

rtError_t rtsBinaryLoadFromData(const void * const data, const uint64_t length,
                                const rtLoadBinaryConfig_t * const optionalCfg, rtBinHandle *handle)
{
  (void)data;
  (void)length;
  (void)optionalCfg;
  uint64_t stub_bin_addr = 0x1200;
  *handle = ge::ValueToPtr(stub_bin_addr);
  return RT_ERROR_NONE;
}

rtError_t rtsFuncGetByName(const rtBinHandle binHandle, const char_t *kernelName, rtFuncHandle *funcHandle)
{
  (void)binHandle;
  (void)kernelName;
  uint64_t stub_func_addr = 0x1600;
  *funcHandle = ge::ValueToPtr(stub_func_addr);
  return RT_ERROR_NONE;
}

rtError_t rtsRegisterCpuFunc(const rtBinHandle binHandle, const char_t * const funcName,
                             const char_t * const kernelName, rtFuncHandle *funcHandle)
{
  (void)binHandle;
  (void)funcName;
  (void)kernelName;
  uint64_t stub_func_addr = 0x1600;
  *funcHandle = ge::ValueToPtr(stub_func_addr);
  return RT_ERROR_NONE;
}