/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPR_GEN_PIPE_PERF_EXPR_H_
#define EXPR_GEN_PIPE_PERF_EXPR_H_

#include "util/tenary_op.h"
#include "base/base_types.h"
#include "parser/tuning_space.h"
#include "set_operation.h"
#include "exe_time_pass.h"
#include "api_perf_register/ascendc_api_perf.h"

namespace att {
using ParentChildsMap = std::map<const SubAxis *, std::vector<std::set<const SubAxis *>>>;
using OrigAxisTree = std::map<std::vector<SubAxis *>, ParentChildsMap>;

// AddPerf/AddTailPerf 参数封装结构体
struct PerfAddContext {
  const std::map<PipeType, Expr> &node_perf;
  std::map<PipeType, Expr> &pipe_costs;
  std::map<Expr, TenaryOp, ExprCmp> &tenary_ops;
  const std::string &expr_prefix;

  PerfAddContext(const std::map<PipeType, Expr> &perf, std::map<PipeType, Expr> &costs,
                 std::map<Expr, TenaryOp, ExprCmp> &ops, const std::string &prefix)
      : node_perf(perf), pipe_costs(costs), tenary_ops(ops), expr_prefix(prefix) {}
};

class PipePerfExpr {
public:
  explicit PipePerfExpr(const TuningSpacePtr &tuning_space) : tuning_space_(tuning_space) {}
  ~PipePerfExpr() = default;
  ge::Status GetPerfExpr(std::map<PipeType, Expr> &pipe_costs, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                         Expr &head_cost);

private:
  // 把tensor信息转换为tensor shape
  ge::Status GetTensorShapes(const NodeInfo &node, std::vector<TensorShapeInfo> &input_dims,
                             std::vector<TensorShapeInfo> &output_dims, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                             bool tail_shape = false) const;
  // 将NodeInfo转换为性能公式使用的NodePerfInfo
  ge::Status ConvertToPerfInfo(const std::vector<NodeInfo> &node_infos, std::vector<NodePerfInfo> &node_perf_infos) const;

  // 获取node 性能计算表达式
  ge::Status GetNodePerf(const NodeInfo &node, std::map<PipeType, Expr> &node_perf,
                         std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape = false) const;

  // 获取node loop times
  ge::Status GetNodeExeTime(const NodeInfo &node, const ExeTimePassManager &exe_time_mgr, TenaryOp &cur_exe_time) const;

  // 获取尾块的loop times
  static ge::Status GetTailExeTime(const NodeInfo &node, const Expr &node_exe_times, Expr &tail_exe_times);

  static ge::Status AddPerf(const Expr &node_exe_times, const std::string &contrib_suffix,
                            PerfAddContext &ctx);
  ge::Status AddTailPerf(const Expr &tail_exe_time, const Expr &node_exe_times,
                         const std::map<PipeType, Expr> &node_tail_perf,
                         PerfAddContext &tail_ctx);

  // 获取节点性能（内部方法，包含VectorFunc特殊处理）
  ge::Status GetNodePerfInternal(const NodeInfo &node, std::map<PipeType, Expr> &node_perf,
                                  std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) const;
  // 添加节点性能到pipe_costs
  ge::Status AddNodePerfToPipeCost(const NodeInfo &node, const Expr &exe_var,
                                   const std::map<PipeType, Expr> &node_perf,
                                   std::map<PipeType, Expr> &pipe_costs,
                                   std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);

  Perf UpdateTilingScheduleConfigTable(const NodeInfo &node, bool tail_shape, PerfOutputInfo &perf_res) const;
  ge::Status UpdatePipeHead(std::map<PipeType, Expr> &pipe_costs, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) const;
  TuningSpacePtr tuning_space_;
};
std::vector<Expr> GetTensorTailRepeat(const TensorPtr &tensor, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);
ge::Status GetTensorShapeInfo(const TensorPtr &tensor, TensorShapeInfo &tensor_shape_info,
                              std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape = false);
}  // namespace att

#endif // EXPR_GEN_PIPE_PERF_EXPR_H_