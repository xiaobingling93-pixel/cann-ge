/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AI_CORE_LAUNCH_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AI_CORE_LAUNCH_KERNEL_H_
#include <cstdint>
namespace gert {
namespace kernel {
enum class InputCommon {
  kStream,  //
  kBinHandle,
  kBlockDim,
  kWorkspaceAddr,
  kShapeBufferAddr,
  kCfg,
  kIoNum,
  kScheduleMode,
  kDfxArgs,
  kRtArg,
  kLocalMemSize,
  kNum
};
enum class WithHandle {
  kTilingKey = static_cast<int32_t>(InputCommon::kNum),  //
  kNodeInfo,
  kIoAddrs,
  kNum
};
enum class WithArgs {
  kIoAddrs = static_cast<int32_t>(InputCommon::kNum),  //
  kNum
};
enum class WithAtomic {
  kWorkspaceIndex = static_cast<int32_t>(InputCommon::kNum),  //
  kIoAddrs,
  kNum
};
enum class WithAtomicHandle {
  kTilingKey = static_cast<int32_t>(InputCommon::kNum), //
  kWorkspaceIndex, //
  kIoAddrs,
  kNum
};
constexpr uint64_t kAssertWorkFlag = 4U;
constexpr uint16_t kDumpTypeBitNum = 56U;
constexpr uint64_t kDumpSkipAddrNum = 1U;
constexpr uint16_t kDumpOffsetBitNum = 32U;

enum class L0DumpType {
  kNormal = 0,
  kFolded = 1,  // simple secondary level 2 pointer fold
  kFoldedWithDesc = 2,  // tensorlist
  kNum
};

enum class MixType {
  MIX_AICORE = 0,
  MIX_VECTOR_CORE
};

struct MixCoreArgs {
  MixType mix_type;
  uint32_t all_core_num;
  uint32_t vec_core_num;
  uint32_t ai_core_num;
};

}  // namespace kernel
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AI_CORE_LAUNCH_KERNEL_H_
