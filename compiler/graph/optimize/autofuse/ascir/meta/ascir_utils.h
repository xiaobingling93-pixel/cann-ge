/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCIR_UTILS_HPP
#define ASCIR_UTILS_HPP

#include <string>
#include "ascir.h"
#include "schedule_result.h"
#include "symbolizer/symbolic_utils.h"

namespace ascir::utils {
/**
 * @brief Dumps the graph to a pbtxt file
 */
void DumpGraph(const ascir::Graph &graph, const std::string &suffix, const uint32_t graph_id = 0U,
               const bool verbose = false);
/**
 * 不被环境变量管控, 只要调用就会落盘图
 * @param graph
 * @param suffix
 * @param graph_id
 * @param verbose
 */
void AlwaysDumpGraph(const Graph &graph, const string &suffix, const uint32_t graph_id = 0U,
                     const bool verbose = false);

void DumpComputeGraph(const ge::ComputeGraphPtr &compute_graph, const std::string &suffix, bool always_dump = false);

void DumpImplGraphs(const std::vector<ascir::ImplGraph> &graphs, const std::string &suffix);

void DumpPyCode(const ge::AscGraph &graph);
/**
 * @brief Prints nodes info in graph.
 */
std::string DebugStr(const ascir::Graph &graph, bool verbose = false);

std::string DebugHintGraphStr(const ascir::HintGraph &graph);
std::string DebugImplGraphStr(const ascir::ImplGraph &graph);

void DumpScheduleResult(const ascir::FusedScheduledResult &fused_scheduled_result, const std::string &suffix,
                        uint32_t graph_id = 0U, bool verbose = true);

std::string IdentifierToStr(ascir::Identifier id);

bool UseSmallTailConcatApi(const ge::AscNode &node, bool *output_need_align = nullptr);

bool IsConcatAllInputsAligned(const ge::AscNode &node);

ge::TriBool AreConcatInputShapesEqual(const ge::AscNodePtr &node);

bool AreAllInputsLoad(const ge::NodePtr &node);
}  // namespace ascir::utils

#endif
