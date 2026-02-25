/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_NODE_TYPES_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_NODE_TYPES_H_
#include <string>
#include "common/types.h"
#include "graph/utils/node_utils.h"
#include "exe_graph/lowering/builtin_node_types.h"
#include "graph/fast_graph/fast_node.h"

namespace gert {
inline bool IsFeedType(const char *const node_type) {
  return (strcmp(node_type, ge::DATA_TYPE.c_str()) == 0) || (strcmp(node_type, ge::AIPP_DATA_TYPE.c_str()) == 0);
}
inline bool IsConstFeedType(const char *const node_type) {
  return (strcmp(node_type, kConstData) == 0);
}
inline bool IsInnerDataType(const char *const node_type) {
  return IsTypeInnerData(node_type);
}
inline bool IsInnerOutput(const char *const node_type) {
  return IsTypeInnerNetOutput(node_type);
}
inline bool IsOutputType(const char *const node_type) {
  return (strcmp(node_type, ge::NETOUTPUT) == 0);
}
inline bool IsConstType(const char *const node_type) {
  return (strcmp(node_type, ge::CONSTANT) == 0) || (strcmp(node_type, ge::CONSTANTOP) == 0);
}
inline bool IsIfOrCaseType(const char *const node_type) {
  auto func = [&node_type](const std::string &type) { return (strcmp(node_type, type.c_str()) == 0); };
  return std::any_of(ge::kIfOpTypes.begin(), ge::kIfOpTypes.end(), func) ||
         std::any_of(ge::kCaseOpTypes.begin(), ge::kCaseOpTypes.end(), func);
}
inline bool IsCaseType(const char *const node_type) {
  auto func = [&node_type](const std::string &type) { return (strcmp(node_type, type.c_str()) == 0); };
  return std::any_of(ge::kCaseOpTypes.begin(), ge::kCaseOpTypes.end(), func);
}
inline bool IsWhileType(const char *const node_type) {
  auto func = [&node_type](const std::string &type) { return (strcmp(node_type, type.c_str()) == 0); };
  return std::any_of(ge::kWhileOpTypes.begin(), ge::kWhileOpTypes.end(), func);
}
inline bool IsWaitAnyone(const char *const node_type) {
  return (strcmp(node_type, "WaitAnyone") == 0);
}
inline bool IsInferShapeNode(const char *const node_type) {
  return (strcmp(node_type, "InferShape") == 0) || (strcmp(node_type, "CompatibleInferShape") == 0);
}
inline bool IsPureInferShapeNode(const char *const node_type) {
  return (strcmp(node_type, "InferShape") == 0);
}
inline bool IsLaunchWithHandleNode(const char *const node_type) {
  return (strcmp(node_type, "LaunchKernelWithHandle") == 0) || (strcmp(node_type, "LaunchMixKernelWithHandle") == 0);
}
inline bool IsLaunchWithFlagNode(const char *const node_type) {
  return (strcmp(node_type, "LaunchKernelWithFlag") == 0) || (strcmp(node_type, "LaunchMixKernelWithFlag") == 0);
}
inline bool IsLaunchFFTSPlusTaskNode(const char *const node_type) {
  return (strcmp(node_type, "LaunchFFTSPlusTask") == 0) || (strcmp(node_type, "LaunchFFTSPlusTaskNoCopy") == 0);
}
inline bool IsFFTSUpdateAICoreArgsNode(const char *const node_type) {
  return (strcmp(node_type, "FFTSUpdateAICoreArgs") == 0);
}
inline bool IsAICoreUpdateContextNode(const char *const node_type) {
  return (strcmp(node_type, "AICoreUpdateContext") == 0);
}
inline bool IsAICpuUpdateContextNode(const char *const node_type) {
  return (strcmp(node_type, "AICpuUpdateContext") == 0);
}
inline bool IsFFTSUpdateAutoAICoreArgsNode(const char *const node_type) {
  return (strcmp(node_type, "FFTSUpdateAutoAICoreArgs") == 0);
}
inline bool IsStaAutoUpdateContext(const char *const node_type) {
  return (strcmp(node_type, "StaAutoUpdateContext") == 0);
}
inline bool IsMixL2UpdateContext(const char *const node_type) {
  return (strcmp(node_type, "MixL2UpdateContext") == 0);
}
inline bool IsUpdateContext(const char *const node_type) {
  return IsAICoreUpdateContextNode(node_type) || IsAICpuUpdateContextNode(node_type) ||
         IsStaAutoUpdateContext(node_type) || IsMixL2UpdateContext(node_type);
}
inline bool IsAtomicLaunchNode(const char *const node_type) {
  return (strcmp(node_type, "AtomicLaunchKernelWithFlag") == 0) || (strcmp(node_type, "AtomicLaunchKernelWithHandle") == 0);
}
inline bool IsAiCoreLaunchNode(const char *const node_type) {
  return IsLaunchWithHandleNode(node_type) || IsLaunchWithFlagNode(node_type) || IsAtomicLaunchNode(node_type);
}
inline bool IsAiCpuLaunchTfNode(const char *const node_type) {
  return (strcmp(node_type, "AicpuLaunchTfKernel") == 0);
}
inline bool IsAiCpuLaunchCCNode(const char *const node_type) {
  return (strcmp(node_type, "AicpuLaunchCCKernel") == 0);
}
inline bool IsHcomLaunchNode(const char *const node_type) {
  return (strcmp(node_type, "LaunchHcomKernel") == 0);
}
inline bool IsAiCpuLaunchNode(const char *const node_type) {
  return IsAiCpuLaunchTfNode(node_type) || IsAiCpuLaunchCCNode(node_type);
}
inline bool IsHostAicpuCpuLaunchNode(const char *const node_type) {
  return (strcmp(node_type, "AicpuHostCompute") == 0);
}
inline bool IsDavinciModelExecuteNode(const char *const node_type) {
  return (strcmp(node_type, "DavinciModelExecute") == 0);
}
inline bool IsSendEventsNode(const char *const node_type) {
  return (strcmp(node_type, "SendEvents") == 0);
}
inline bool IsExecuteOpFuncNode(const char *const node_type) {
  return (strcmp(node_type, "ExecuteOpFunc") == 0);
}

inline bool IsCustomOpFuncNode(const char *const node_type) {
  return (strcmp(node_type, "ExecuteCustomOp") == 0);
}

inline bool IsExecuteOpPrepareNode(const char *const node_type) {
  return (strcmp(node_type, "ExecuteOpPrepare") == 0);
}

inline bool IsExecuteOplaunchNode(const char *const node_type) {
  return (strcmp(node_type, "ExecuteOpLaunch") == 0);
}

inline bool IsBuildTensorNode(const char *const node_type) {
  return ((strcmp(node_type, "BuildTensor") == 0) || (strcmp(node_type, "BuildTensorStorage") == 0) ||
          (strcmp(node_type, "BuildTensorPureShape") == 0));
}

inline bool IsIdentityNode(const char *const node_type) {
    return ((strcmp(node_type, "IdentityAddr") == 0) || (strcmp(node_type, "IdentityShapeAndAddr") == 0));
}

inline bool IsSplitDataTensorNode(const char *const node_type) {
  return (strcmp(node_type, "SplitDataTensor") == 0);
}

inline bool IsSplitTensorNode(const char *const node_type) {
  return ((strcmp(node_type, "SplitDataTensor") == 0) || (strcmp(node_type, "SplitConstTensor") == 0));
}

inline bool IsLaunchNode(const char *const node_type) {
  return IsAiCoreLaunchNode(node_type) || IsAiCpuLaunchNode(node_type) ||
         (strcmp(node_type, "StarsTaskLaunchKernel") == 0) || IsLaunchFFTSPlusTaskNode(node_type) ||
         IsHcomLaunchNode(node_type) || IsDavinciModelExecuteNode(node_type) || IsExecuteOpFuncNode(node_type) ||
         IsExecuteOplaunchNode(node_type) || IsCustomOpFuncNode(node_type);
}

inline bool IsCalcSizeNode(const char *const node_type) {
  return (strcmp(node_type, "CalcTensorSizeFromShape") == 0) || (strcmp(node_type, "CalcTensorSizeFromStorage") == 0) ||
         (strcmp(node_type, "CalcUnalignedTensorSizeFromStorage") == 0);
}

inline bool IsModelOutZeroCopyEnabledAllocNode(const char *const node_type) {
  return (strcmp(node_type, "AllocMemory") == 0) || (strcmp(node_type, "AllocMemHbm") == 0) ||
         (strcmp(node_type, "AllocMemHost") == 0);
}

inline bool IsAllocNode(const char *const node_type) {
  static std::vector<const char *> kAllocKernels = {"AllocMemory",   "AllocMemHbm",     "AllocMemHost",
                                                    "AllocBatchHbm", "AllocateFftsMem", "AllocateBatchFftsMems",
                                                    "AllocFixedFeatureMemory"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kAllocKernels.begin(), kAllocKernels.end(), func);
}
inline bool IsAllocHostCpuOutputMemoryNode(const char *const node_type) {
  return (strcmp(node_type, "AllocHostCpuOutputMemory") == 0);
}
inline bool IsAllocHostNode(const char *const node_type) {
  static std::vector<const char *> kAllocKernels = {"AllocHostCpuOutputMemory", "AllocMemHost", "MakeSureTensorAtHost",
                                                    "CopyD2H", "CopyTensorDataH2H"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kAllocKernels.begin(), kAllocKernels.end(), func);
}
inline bool IsFreeNode(const char *const node_type) {
  static std::vector<const char *> kFreeKernels = {
      "FreeMemory",   "FreeMemHbm",        "FreeBatchHbm",           "FreeTensorMemory",
      "FreeFftsMem",  "FreeBatchFftsMems", "FreeFixedFeatureMemory", "FreeMemoryHoldAddr",
      "FreeMemHbmHoldAddr", "FreeBatchHbmHoldAddr"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kFreeKernels.begin(), kFreeKernels.end(), func);
}
inline bool IsInitNode(const char *const node_type) {
  return (strcmp(node_type, "Init") == 0);
}
inline bool IsTilingNode(const char *const node_type) {
  static std::vector<const char *> kTilingKernels = {"Tiling",           "FallibleTiling",
                                                     "CompatibleTiling", "FallibleCompatibleTiling",
                                                     "CacheableTiling",  "CacheableFallibleTiling",
                                                     "SymbolTiling",     "CacheableSymbolTiling"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kTilingKernels.begin(), kTilingKernels.end(), func);
}
inline bool IsTilingAppendWorkspace(const char *const node_type) {
  return (strcmp(node_type, "TilingAppendWorkspace") == 0);
}
inline bool IsSwitchNotifyNode(const char *const node_type) {
  return (strcmp(node_type, "SwitchNotify") == 0);
}
inline bool IsCondSwitchNotifyNode(const char *const node_type) {
  return (strcmp(node_type, "CondSwitchNotify") == 0);
}
inline bool IsWatcherPlaceholderNode(const char *const node_type) {
  return (strcmp(node_type, "WatcherPlaceholder") == 0);
}
inline bool IsBranchPivot(const char *const node_type) {
  return (strcmp(node_type, "BranchPivot") == 0);
}
inline bool IsBranchDone(const char *const node_type) {
  return (strcmp(node_type, "BranchDone") == 0);
}
inline bool IsSyncStreamNode(const char *const node_type) {
  return (strcmp(node_type, "SyncStream") == 0);
}
inline bool IsSubgraphCall(const char *const node_type) {
  return (strcmp(node_type, "SubgraphCall") == 0);
}
inline bool IsLaunchOrHasSubGraphNode(const ge::FastNode *const node) {
  if (IsLaunchNode(node->GetTypePtr())) {
    return true;
  }
  const auto &subgraph_names = node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
  return !subgraph_names.empty();
}

inline bool IsCopyAsyncNode(const ge::FastNode *const node) {
  const auto &node_type = node->GetTypePtr();
  static std::vector<const char *> kAllocKernels = {"CopyH2D",   "CopyD2H",
                                                    "MakeSureTensorAtDevice", "CopyD2D",
                                                    "EnsureTensorAtOutMemory", "CopyFlowLaunch"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kAllocKernels.begin(), kAllocKernels.end(), func);
}
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_NODE_TYPES_H_
