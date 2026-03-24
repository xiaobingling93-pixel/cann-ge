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

// New MLIR-style dump format
// is_subgraph: true for subgraph (simplified mode, VIEW3/.api.unit/.mem are not displayed)
std::string DebugStrNew(const ascir::Graph &graph, bool verbose = false, bool is_subgraph = false);

void DumpScheduleResult(const ascir::FusedScheduledResult &fused_scheduled_result, const std::string &suffix,
                        uint32_t graph_id = 0U, bool verbose = true);

std::string IdentifierToStr(ascir::Identifier id);

bool UseSmallTailConcatApi(const ge::AscNode &node, bool *output_need_align = nullptr);

bool IsConcatAllInputsAligned(const ge::AscNode &node);

ge::TriBool AreConcatInputShapesEqual(const ge::AscNodePtr &node);

bool AreAllInputDistinct(const ge::NodePtr &node);

bool AreAllInputsFromPosition(const ge::AscNodePtr &node, Position position);

/**
 * @brief 设置当前 fused_graph 名称，用于按图分目录 dump
 * @param name fused_graph 的名称，空字符串表示清空
 * @return 返回之前的名称，用于恢复
 */
std::string SetCurrentFusedGraphName(const std::string &name);

void ResetDumpConfig();

/**
 * @brief RAII Guard，支持嵌套，作用域结束时自动恢复之前的 fused_graph_name
 */
class FusedGraphNameGuard {
public:
  explicit FusedGraphNameGuard(const std::string &name) : prev_name_(SetCurrentFusedGraphName(name)) {}
  ~FusedGraphNameGuard() {
    (void)SetCurrentFusedGraphName(prev_name_);
  }
  FusedGraphNameGuard(const FusedGraphNameGuard&) = delete;
  FusedGraphNameGuard& operator=(const FusedGraphNameGuard&) = delete;
private:
  std::string prev_name_;
};

}  // namespace ascir::utils

#endif
