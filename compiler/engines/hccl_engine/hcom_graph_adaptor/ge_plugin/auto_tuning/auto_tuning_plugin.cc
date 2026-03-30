/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_tuning_plugin.h"
#include <sstream>
#include "ge/ge_api_types.h"            // ge对内options
#include "framework/common/ge_types.h"  // ge对外options
#include "hccl/hcom.h"
#include "hccl/hccl_types.h"
#include "hcom_op_utils.h"
#include "hcom_log.h"

namespace hccl {
AutoTuningPlugin::AutoTuningPlugin() : opsKernelInfoStorePtr_(nullptr), graphOptimizerPtr_(nullptr) {}

AutoTuningPlugin::~AutoTuningPlugin() {}

ge::Status AutoTuningPlugin::Finalize() {
  opsKernelInfoStorePtr_ = nullptr;
  graphOptimizerPtr_ = nullptr;
  return ge::SUCCESS;
}

void AutoTuningPlugin::GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &opKernInfos) {
  HCCL_INFO("get hcom kernel info store start.");

  if (opsKernelInfoStorePtr_ != nullptr) {
    opKernInfos.insert(std::make_pair(AUTOTUNE_HCCL_OPS_LIB_NAME, opsKernelInfoStorePtr_));
  } else {
    HCCL_ERROR("[Get][OpsKernelInfoStores]get hcom ops kernel info stores ptr failed for nullptr.");
  }
  HCCL_INFO("get hcom kernel info store finished.");
  return;
}

AutoTuningPlugin &AutoTuningPlugin::Instance() {
  static AutoTuningPlugin plugin;
  return plugin;
}

void AutoTuningPlugin::GetGraphOptimizerObjs([[maybe_unused]] map<string, GraphOptimizerPtr> &graphOptimizers) {
  HCCL_INFO("get hcom graph optimizer objs start.");
#ifndef HCOM_EXECUTOR
  if (graphOptimizerPtr_ != nullptr) {
    graphOptimizers.insert(std::make_pair(HCCL_GRAPH_OPTIMIZER_NAME, graphOptimizerPtr_));
  } else {
    HCCL_ERROR("[Get][GraphOptimizerObjs]get hcom graph optimizer objs failed for nullptr.");
  }
#endif
  HCCL_INFO("get hcom graph optimizer objs end.");
  return;
}

ge::Status AutoTuningPlugin::Initialize([[maybe_unused]] const std::map<string, string> &options) {
  EXECEPTION_CATCH((opsKernelInfoStorePtr_ = std::make_shared<hccl::AutoTuningHcomOpsKernelInfoStore>()),
                   return ge::INTERNAL_ERROR);
#ifndef HCOM_EXECUTOR
  EXECEPTION_CATCH((graphOptimizerPtr_ = std::make_shared<hccl::AutoTuningHcomGraphOptimizer>()),
                   return ge::INTERNAL_ERROR);
#endif
  HcomSetAutoTuneMode(true);
  HCCL_INFO("Auto Tuning hcom ops plugin init success.");
  return ge::SUCCESS;
}
}  // namespace hccl
