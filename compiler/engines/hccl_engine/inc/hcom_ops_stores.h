/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_OPS_STORES_H
#define HCOM_OPS_STORES_H

#include <string>
#include <map>
#include <memory>
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * @brief Initialize HCOM operators plugin.
 *
 * @param options Input parameter. Options must contain rank table path, deploy mode, rank id, pod name.
 * @return ge::SUCCESS success; others：fail.
 */
ge::Status Initialize(const std::map<std::string, std::string> &options);

/**
 * @brief Finalize HCOM operators plugin.
 *
 * @return ge::SUCCESS success; others: fail.
 */
ge::Status Finalize();

/**
 * @brief Get the information store of HCOM operators.
 *
 * @param opKernInfos A map identifying the information store of HCOM operators.
 */
void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &opKernInfos);

/**
 * @brief Get the graph optimizer of HCOM operators.
 *
 * @param graphOptimizers A map identifying the graph optimizer of HCOM operators.
 */
void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graphOptimizers);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // HCOM_OPS_STORES_H
