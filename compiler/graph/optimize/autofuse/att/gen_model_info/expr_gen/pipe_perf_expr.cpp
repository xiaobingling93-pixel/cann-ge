/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pipe_perf_expr.h"
#include <unordered_set>
#include "base/att_const_values.h"
#include "arg_list_manager.h"
#include "common/checker.h"
#include "api_perf_register/utils/vf_perf_utils.h"
#include "api_perf_register/utils/api_perf_utils.h"
#include "api_perf_register/api_perf_factory.h"
#include "utils/stable_node_id.h"
#include "graph/compute_graph.h"

namespace att {
namespace {
constexpr int32_t kMaxRecursiveNum = 10;
void UpdateDim(const std::vector<Expr> &stride, std::vector<Expr> &dims) {
  for (size_t i = stride.size() - 1UL; i >= 1UL; --i) {
    if (stride[i] == 0) {
      continue;
    }
    size_t cur = i - 1UL;
    while (cur >= 1UL && stride[cur] == 0) {
      --cur;
    }
    if (stride[cur] != 0) {
      dims[i] = stride[cur];
    }
    break;
  }
}

// 辅助函数：处理 tenary_ops，生成变量名并更新
static void ProcessTenaryOps(const NodeExprId &node_expr_id, const std::string &annotation,
                             const PerfOutputInfo &perf_res, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                             std::vector<Expr> &update_vars, std::vector<std::pair<Expr, Expr>> &replace_vars) {
  std::string expr_prefix = node_expr_id.GetExprVarPrefix();
  std::string full_desc = node_expr_id.GetVarPrefix();
  Expr cur_expr;
  for (const auto &pair : perf_res.tenary_ops) {
    std::string var_name = expr_prefix + "_" + Str(pair.first) + annotation;
    std::string desc = full_desc + "_" + Str(pair.first) + annotation;
    GetPerfVar(var_name, cur_expr, tenary_ops);
    tenary_ops[cur_expr] = pair.second;
    tenary_ops[cur_expr].SetVariable(cur_expr);
    tenary_ops[cur_expr].SetDescription(desc);
    update_vars.emplace_back(cur_expr);
    replace_vars.emplace_back(std::make_pair(pair.first, cur_expr));
  }
}

// 辅助函数：应用变量替换
static void ApplyVariableReplacement(const std::vector<Expr> &update_vars,
                                     const std::vector<std::pair<Expr, Expr>> &replace_vars,
                                     std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) {
  for (const auto &var : update_vars) {
    tenary_ops[var].Replace(replace_vars);
    tenary_ops[var].UpdateRelatedVars(replace_vars);
    GELOGD("The value of [%s] is [%s]", Str(var).c_str(), tenary_ops[var].GetTenaryOpStr().c_str());
  }
}

// 辅助函数：处理 pipe_res，生成各pipe类型的性能变量
static void ProcessPipeRes(const NodeExprId &node_expr_id, const std::string &annotation,
                           const PerfOutputInfo &perf_res, std::map<PipeType, Expr> &node_perf,
                           std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                           const std::vector<std::pair<Expr, Expr>> &replace_vars) {
  std::string expr_prefix = node_expr_id.GetExprVarPrefix();
  std::string full_desc = node_expr_id.GetVarPrefix();
  Expr perf_expr;
  for (const auto &pair : perf_res.pipe_res) {
    auto it = PipeType2Str.find(pair.first);
    if (it == PipeType2Str.end()) {
      continue;
    }
    std::string var_name = expr_prefix + annotation + "_" + it->second + "_perf";
    std::string desc = full_desc + annotation + "_" + it->second + "_perf";
    GetPerfVar(var_name, perf_expr, tenary_ops);
    auto iter = perf_res.tenary_ops.find(pair.second);
    if (iter != perf_res.tenary_ops.end()) {
      // 复制 TenaryOp，但需要保留变量、替换和描述设置
      tenary_ops[perf_expr] = iter->second;
    } else {
      tenary_ops[perf_expr] = TenaryOp(pair.second);
    }
    // 使用传入的 replace_vars 进行变量替换
    node_perf[pair.first] = pair.second.Replace(replace_vars);
    tenary_ops[perf_expr].Replace(replace_vars);
    tenary_ops[perf_expr].UpdateRelatedVars(replace_vars);
    tenary_ops[perf_expr].SetVariable(perf_expr);
    // 必须在所有操作之后设置描述，因为 Replace/SetVariable/UpdateRelatedVars 可能影响 tenary_ops
    tenary_ops[perf_expr].SetDescription(desc);
  }
}

// 替换性能公式中使用的符号
ge::Status UpdatePerfRes(const NodeInfo &node, const PerfOutputInfo &perf_res, std::map<PipeType, Expr> &node_perf,
                         std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape = false) {
  NodeExprId node_expr_id = BuildNodeExprId(node);
  std::string annotation = tail_shape ? "_tail" : "";
  std::vector<Expr> update_vars;
  std::vector<std::pair<Expr, Expr>> replace_vars;

  GELOGI("Processing performance formula for node type %s, expr prefix: %s, full desc: %s",
         node.node_type.c_str(), node_expr_id.GetExprVarPrefix().c_str(),
         node_expr_id.GetVarPrefix().c_str());

  ProcessTenaryOps(node_expr_id, annotation, perf_res, tenary_ops, update_vars, replace_vars);
  ApplyVariableReplacement(update_vars, replace_vars, tenary_ops);
  ProcessPipeRes(node_expr_id, annotation, perf_res, node_perf, tenary_ops, replace_vars);

  return ge::SUCCESS;
}

ge::Status CheckOuterAxis(const SubAxis *cut_axis, const std::vector<SubAxis *> from_axes, bool &outer_inside_loop,
                          Expr &outer_axis, const int32_t recursive_num = 0) {
  GE_ASSERT_TRUE(recursive_num < kMaxRecursiveNum, "CheckOuterAxis failed, recursive_num = %d should < %d",
                 kMaxRecursiveNum);
  for (const auto &axis : from_axes) {
    if (axis->axis_type == AxisPosition::ORIGIN || axis->axis_type == AxisPosition::POSERR) {
      continue;
    } else if (axis->axis_type == AxisPosition::OUTER) {
      GE_ASSERT_TRUE(axis->parent_axis.size() == 1);
      if (cut_axis->parent_axis[0] == axis->parent_axis[0]) {
        GELOGD("Found outer axis of vectorized axis [%s] in [%s].", ge::SymbolicUtils::ToString(cut_axis->repeat).c_str(), ge::SymbolicUtils::ToString(axis->repeat).c_str());
        outer_inside_loop = true;
        outer_axis = axis->repeat;
        break;
      }
    } else {
      GELOGD("Check outer axis from [%s] (in [%zu] axes).", axis->name.c_str(), axis->parent_axis.size());
      GE_ASSERT_SUCCESS(CheckOuterAxis(cut_axis, axis->parent_axis, outer_inside_loop, outer_axis, recursive_num + 1));
      if (outer_inside_loop) {
        break;
      }
    }
  }
  return ge::SUCCESS;
}

Expr GetTailSize(const Expr &a, const Expr &b, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) {
  Expr tail_size;
  if (a.IsConstExpr() && b.IsConstExpr()) {
    int32_t a_value;
    int32_t b_value;
    a.GetConstValue(a_value);
    b.GetConstValue(b_value);
    if (a_value % b_value == 0) {
      tail_size = b;
    } else {
      tail_size = CreateExpr(a_value % b_value);
    }
  } else {
    tail_size = CreateExpr((Str(b) + "_tail").c_str());
    Expr status = ge::sym::Mod(a, b);
    TenaryOp tenary_op = TenaryOp(CondType::K_EQ, status, CreateExpr(0), b, status);
    tenary_op.SetVariable(tail_size);
    tenary_ops[tail_size] = tenary_op;
    GELOGD("The value of [%s] is [%s]", Str(tail_size).c_str(), tenary_op.GetTenaryOpStr().c_str());
  }
  return tail_size;
}

bool CheckInclude(const std::vector<std::string> &cur_orig, 
                  const std::vector<std::string> &check_orig) {
  std::map<std::string, int32_t> rec_num;
  if (check_orig.size() <= cur_orig.size()) {
    return false;
  }
  for (const auto &name : cur_orig) {
    rec_num[name]++;
  }
  for (const auto &name : check_orig) {
    rec_num[name]--;
  }
  for (const auto &pair : rec_num) {
    if (pair.second > 0) {
      return false;
    }
  }
  return true;
}

/*
此函数用于检测是否触发尾块计算逻辑：
1) 向量化轴中只存在一根切分轴
2) 存在某根循环轴，他的原始轴包含了这根切分轴所有的原始轴
注：
1. 对于同时切block和tile的场景，只会存在一个尾块
2. 存在一个尾块的场景，仅在核数为1的时候性能公式估计不准
3. 多核切分策略保证优先分核，而不会分出尾块，所以保证了不会出现上述情况
*/
bool CheckSingleCut(const NodeInfo &node) {
  std::set<const SubAxis*> cut_axes;
  for (const auto &input : node.inputs) {
    for (const auto &dim : input->dim_info) {
      if (dim->axis_type == AxisPosition::INNER) {
        cut_axes.insert(dim);
      }
    }
  }
  for (const auto &output : node.outputs) {
    for (const auto &dim : output->dim_info) {
      if (dim->axis_type == AxisPosition::INNER) {
        cut_axes.insert(dim);
      }
    }
  }
  GELOGD("Node [%s] has [%zu] cut.", node.name.c_str(), cut_axes.size());
  if (cut_axes.size() != 1) {
    GELOGD("Single cut unsatisfied for node[%s].", node.name.c_str());
    return false;
  }
  const SubAxis* cut_axis = *cut_axes.begin();
  for (const auto &axis : node.loop_axes) {
    if (CheckInclude(cut_axis->orig_axis_name, axis->orig_axis_name)) {
      GELOGD("The merged axis [%s] has the outer axis of [%s].", axis->name.c_str(), cut_axis->name.c_str());
      return true;
    }
  }
  GELOGD("Cannot find merged outer axis for [%s] within loop axes.", cut_axis->name.c_str());
  return false;
}
}

std::vector<Expr> GetTensorTailRepeat(const TensorPtr &tensor, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) {
  std::vector<Expr> ret;
  for (const auto &dim: tensor->dim_info) {
    if (dim->axis_type != AxisPosition::INNER) {
      ret.emplace_back(dim->repeat);
    } else {
      GE_ASSERT_TRUE(dim->parent_axis.size() == 1);
      Expr parent_size = dim->parent_axis[0]->repeat;
      ret.emplace_back(GetTailSize(parent_size, dim->repeat, tenary_ops));
    }
  }
  return ret;
}

ge::Status GetTensorShapeInfo(const TensorPtr &tensor, TensorShapeInfo &tensor_shape_info,
                              std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape) {
  GE_ASSERT_TRUE(tensor->repeat.size() == tensor->stride.size());
  if (tail_shape) {
    auto tail_repeat = GetTensorTailRepeat(tensor, tenary_ops);
    tensor_shape_info.repeats = tail_repeat;
    tensor_shape_info.dims = tail_repeat;
  } else {
    tensor_shape_info.repeats = tensor->repeat;
    tensor_shape_info.dims = tensor->repeat;
  }
  if (tensor_shape_info.dims.empty()) {
    tensor_shape_info.dims.emplace_back(ge::sym::kSymbolOne);
  } else {
    UpdateDim(tensor->stride, tensor_shape_info.dims);
  }
  tensor_shape_info.data_type = tensor->data_type;
  tensor_shape_info.loc = tensor->loc;
  tensor_shape_info.data_type_size = tensor->data_type_size;
  tensor_shape_info.strides = tensor->stride;
  tensor_shape_info.gm_strides = tensor->gm_stride;
  tensor_shape_info.origin_repeats = tensor->repeat;
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::GetTensorShapes(const NodeInfo &node, std::vector<TensorShapeInfo> &input_dims,
                                         std::vector<TensorShapeInfo> &output_dims,
                                         std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape) const {
  for (const auto &tensor : node.inputs) {
    TensorShapeInfo tensor_shape_info;
    GE_ASSERT_SUCCESS(GetTensorShapeInfo(tensor, tensor_shape_info, tenary_ops, tail_shape),
                       "Get node [%s] in tensor shape[%s] failed.", node.name.c_str(), tensor->name.c_str());
    input_dims.emplace_back(tensor_shape_info);
  }
  for (const auto &tensor : node.outputs) {
    TensorShapeInfo tensor_shape_info;
    GE_ASSERT_SUCCESS(GetTensorShapeInfo(tensor, tensor_shape_info, tenary_ops, tail_shape),
                       "Get node [%s] out tensor shape[%s] failed.", node.name.c_str(), tensor->name.c_str());
    output_dims.emplace_back(tensor_shape_info);
  }
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::ConvertToPerfInfo(const std::vector<NodeInfo> &node_infos,
                                           std::vector<NodePerfInfo> &node_perf_infos) const {
  std::vector<TensorShapeInfo> inputs;
  std::vector<TensorShapeInfo> outputs;
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;  // 当前暂不使用
  for (const auto &sub_node_info : node_infos) {
    NodePerfInfo node_perf;
    node_perf.optype = sub_node_info.node_type;
    if (!sub_node_info.inputs.empty()) {
      node_perf.input_dtype = sub_node_info.inputs[0]->data_type;
    }
    if (!sub_node_info.outputs.empty()) {
      node_perf.output_dtype = sub_node_info.outputs[0]->data_type;
      GE_ASSERT_SUCCESS(GetTensorShapes(sub_node_info, inputs, outputs, tenary_ops),
                        "Get tensor shape failed, node[%s].", sub_node_info.name.c_str());
      node_perf.dims = outputs[0].dims;
    }
    node_perf_infos.emplace_back(node_perf);
  }
  return ge::SUCCESS;
}

Perf PipePerfExpr::UpdateTilingScheduleConfigTable(const NodeInfo &node, bool tail_shape, PerfOutputInfo &perf_res) const {
  Perf perf_func = nullptr;
  const auto api_perf = GetApiPerf(node.node_type);
  if (api_perf != nullptr) {
    perf_func = api_perf->GetPerfFunc();
    perf_res.cache_line_config = tuning_space_->cache_line_config;
    if (tuning_space_->tiling_schedule_config_table == nullptr ||
        (api_perf->GetTilingScheduleConfigTable() != nullptr &&
         api_perf->GetTilingScheduleConfigTable()->GetConfigPriority() >
             tuning_space_->tiling_schedule_config_table->GetConfigPriority())) {
      GELOGD(
          "Replace node [%s] type [%s] tiling schedule config table with api perf tiling schedule config table, "
          "is_enable = %d, core_num_ratio = %s, ub_ratio = %s.",
          node.name.c_str(), node.node_type.c_str(),
          api_perf->GetTilingScheduleConfigTable()->GetTradeOffConfig().is_enable,
          Str(api_perf->GetTilingScheduleConfigTable()->GetTradeOffConfig().core_num_ratio).c_str(),
          Str(api_perf->GetTilingScheduleConfigTable()->GetTradeOffConfig().ub_ratio).c_str());
      tuning_space_->tiling_schedule_config_table = api_perf->GetTilingScheduleConfigTable();
    }
  }
  // tail_shape场景：仅整块需要进行cache line检查，尾块大小一定是小于整块的，不需要进行cache line大小检查
  if (tail_shape || tuning_space_->tiling_schedule_config_table == nullptr ||
      !tuning_space_->tiling_schedule_config_table->IsEnableCacheLineCheck()) {
    perf_res.cache_line_config = nullptr;
  }
  return perf_func;
}

ge::Status PipePerfExpr::GetNodePerf(const NodeInfo &node, std::map<PipeType, Expr> &node_perf,
                                     std::map<Expr, TenaryOp, ExprCmp> &tenary_ops, bool tail_shape) const {
  std::string tail_annotation;
  if (tail_shape) {
    tail_annotation = "tail ";
  }
  const auto &node_type = node.node_type;
  const auto &node_unit = node.node_unit;
  std::vector<TensorShapeInfo> inputs;
  std::vector<TensorShapeInfo> outputs;
  GE_ASSERT_SUCCESS(GetTensorShapes(node, inputs, outputs, tenary_ops, tail_shape), "Get tensor shape failed!");
  for (size_t i = 0; i < inputs.size(); i++) {
    GELOGD("node[%s, %s] input[%zu] %s shape: {%s}", node.name.c_str(), node.node_type.c_str(), i,
           tail_annotation.c_str(), inputs[i].GetDimExpr().c_str());
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    GELOGD("node[%s, %s] output[%zu] %s shape: {%s}", node.name.c_str(), node.node_type.c_str(), i,
           tail_annotation.c_str(), outputs[i].GetDimExpr().c_str());
  }
  PerfOutputInfo perf_res;
  Perf perf_func = UpdateTilingScheduleConfigTable(node, tail_shape, perf_res);
  // 获取pipe的兜底性能
  if (perf_func == nullptr) {
    GELOGD("Get node [%s] type [%s] perf func failed, node_unit = %s.", node.name.c_str(), node_type.c_str(),
           node_unit.c_str());
    perf_func = EvalCosts::Instance().GetFunc(node_unit);
  }
  GE_ASSERT_NOTNULL(perf_func, "Get node type [%s] perf func failed, node unit[%s].", node_type.c_str(),
                    node_unit.c_str());

  if (perf_func(inputs, outputs, node, perf_res) != ge::SUCCESS) {
    node_perf.clear();
  }
  GELOGD("node[%s, %s] perf res: {%s}", node.name.c_str(), node.node_type.c_str(), perf_res.ToString().c_str());
  GE_ASSERT_SUCCESS(UpdatePerfRes(node, perf_res, node_perf, tenary_ops, tail_shape));
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::GetNodeExeTime(const NodeInfo &node, const ExeTimePassManager &exe_time_mgr,
                                        TenaryOp &cur_exe_time) const {
  Expr exe_time = CreateExpr(1U);
  for (auto &loop_axis : node.loop_axes) {
    exe_time = ge::sym::Mul(exe_time, loop_axis->repeat);
  }
  GE_ASSERT_TRUE(IsValid(exe_time), "Get node exe times expr failed.");
  cur_exe_time = exe_time_mgr.UpdateNodeExeTime(node, exe_time);
  std::string exe_time_name = node.name + "_exe_time";
  cur_exe_time.SetVariable(CreateExpr(exe_time_name.c_str()));
  ArgListManager::GetInstance().SetArgExpr(exe_time_name, cur_exe_time.GetVariable());
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::AddTailPerf(const Expr &tail_exe_time, const Expr &node_exe_times,
                                     const std::map<PipeType, Expr> &node_perf,
                                     const std::map<PipeType, Expr> &node_tail_perf,
                                     std::map<PipeType, Expr> &pipe_costs) const {
  Expr pipe_cost;
  Expr core_exe_time = node_exe_times - tail_exe_time;
  GELOGD("The exe time of the tail block is [%s], the exe time of the other block is [%s].",
         ge::SymbolicUtils::ToString(tail_exe_time).c_str(), ge::SymbolicUtils::ToString(core_exe_time).c_str());
  GE_ASSERT_SUCCESS(AddPerf(core_exe_time, node_perf, pipe_costs));
  GE_ASSERT_SUCCESS(AddPerf(tail_exe_time, node_tail_perf, pipe_costs));
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::AddPerf(const Expr &node_exe_times, const std::map<PipeType, Expr> &node_perf,
                                 std::map<PipeType, Expr> &pipe_costs) const {
  Expr pipe_cost;
  for (const auto &pair : node_perf) {
    const auto &pipe_type_iter = PipeType2Str.find(pair.first);
    GE_ASSERT_TRUE(pipe_type_iter != PipeType2Str.end(), "Get pipe type str failed.");
    GELOGD("Get perf times [%s] at [%s], exe_time [%s]", pair.second.Str().get(), pipe_type_iter->second.c_str(),
           ge::SymbolicUtils::ToString(node_exe_times).c_str());
    pipe_cost = ge::sym::Mul(node_exe_times, pair.second);
    auto iter = pipe_costs.find(pair.first);
    if (iter == pipe_costs.end()) {
      pipe_costs[pair.first] = pipe_cost;
    } else {
      pipe_costs[pair.first] = pipe_cost + pipe_costs[pair.first];
    }
  }
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::GetTailExeTime(const NodeInfo &node, const Expr &node_exe_times, Expr &tail_exe_times) const {
  Expr outer_axis;
  const SubAxis *cut_axis = nullptr;
  bool outer_inside_loop = false;
  for (const auto &input : node.inputs) {
    for (const auto &dim : input->dim_info) {
      if (dim->axis_type == AxisPosition::INNER) {
        GE_ASSERT_TRUE(dim->parent_axis.size() == 1);
        GE_ASSERT_TRUE(cut_axis == nullptr || cut_axis == dim);
        cut_axis = dim;
      }
    }
  }
  for (const auto &output : node.outputs) {
    for (const auto &dim : output->dim_info) {
      if (dim->axis_type == AxisPosition::INNER) {
        GE_ASSERT_TRUE(dim->parent_axis.size() == 1);
        GE_ASSERT_TRUE(cut_axis == nullptr || cut_axis == dim);
        cut_axis = dim;
      }
    }
  }
  if (cut_axis != nullptr) {
    for (const auto &loop_axis : node.loop_axes) {
      GE_ASSERT_SUCCESS(CheckOuterAxis(cut_axis, loop_axis->parent_axis, outer_inside_loop, outer_axis));
    }
  }
  if (outer_inside_loop) {
    tail_exe_times = ge::sym::Ceiling(node_exe_times / outer_axis);
  } else {
    tail_exe_times = ge::sym::kSymbolOne;
  }
  GELOGD("Exe Time of tail block is [%s].", ge::SymbolicUtils::ToString(tail_exe_times).c_str());
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::UpdatePipeHead(std::map<PipeType, Expr> &pipe_costs,
                                        std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) {
  for (auto &pair : pipe_costs) {
    auto pipe_head_perf_func = GetPipeHeadPerfFunc(pair.first);
    GE_ASSERT_NOTNULL(pipe_head_perf_func);
    Expr pipe_head = pipe_head_perf_func(tuning_space_->node_infos, tenary_ops);
    pair.second = ge::sym::Add(pair.second, pipe_head);
  }
  return ge::SUCCESS;
}

ge::Status PipePerfExpr::GetPerfExpr(std::map<PipeType, Expr> &pipe_costs,
                                     std::map<Expr, TenaryOp, ExprCmp> &tenary_ops,
                                     Expr &head_cost) {
  ExeTimePassManager exe_time_mgr(tuning_space_);

  std::unordered_set<std::string> skip_node_types = {kData, kWorkspace, kOutput, kTbufData, kScalar};
  for (const auto &node : tuning_space_->node_infos) {
    if (skip_node_types.count(node.node_type) != 0U) {
      continue;
    }

    // 获取节点执行时间
    TenaryOp node_exe_times;
    GE_ASSERT_SUCCESS(GetNodeExeTime(node, exe_time_mgr, node_exe_times),
                      "Get node [%s] exec times failed.", node.name.c_str());
    Expr exe_var = node_exe_times.GetVariable();
    tenary_ops[exe_var] = node_exe_times;
    GELOGD("Get node [%s] exe times %s=[%s]", node.name.c_str(), exe_var.Serialize().get(),
           node_exe_times.GetTenaryOpStr().c_str());

    // 获取节点性能
    std::map<PipeType, Expr> node_perf;
    GE_ASSERT_SUCCESS(GetNodePerfInternal(node, node_perf, tenary_ops),
                      "Get node [%s][%s] perf failed.", node.name.c_str(), node.node_type.c_str());

    // 添加性能到总成本
    GE_ASSERT_SUCCESS(AddNodePerfToPipeCost(node, exe_var, node_perf, pipe_costs, tenary_ops));
  }

  GE_ASSERT_SUCCESS(UpdatePipeHead(pipe_costs, tenary_ops));
  GE_ASSERT_SUCCESS(GetOpHeadCost(head_cost));
  return ge::SUCCESS;
}

// 获取节点性能（内部方法，包含VectorFunc特殊处理）
ge::Status PipePerfExpr::GetNodePerfInternal(const NodeInfo &node, std::map<PipeType, Expr> &node_perf,
                                              std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) const {
  if (node.node_type == kVectorFunc) {
    // VectorFunc节点特殊处理
    std::vector<NodePerfInfo> node_perf_infos;
    GE_ASSERT_SUCCESS(ConvertToPerfInfo(node.sub_nodes_infos, node_perf_infos),
                      "Convert to perf info failed, node = %s %s", node.name.c_str(), node.node_type.c_str());
    Expr res;
    GE_ASSERT_SUCCESS(VfPerfUtils::GetVectorFunctionPerf(node_perf_infos, res),
                      "Get vector function perf failed, node = %s %s.", node.name.c_str(), node.node_type.c_str());

    // 为VectorFunc节点创建tenary_ops条目
    NodeExprId node_expr_id = BuildNodeExprId(node);
    std::string expr_prefix = node_expr_id.GetExprVarPrefix();
    std::string full_desc = node_expr_id.GetVarPrefix();

    auto it = PipeType2Str.find(PipeType::AIV_VEC);
    if (it != PipeType2Str.end()) {
      std::string var_name = expr_prefix + "_" + it->second + "_perf";
      std::string desc = full_desc + "_" + it->second + "_perf";
      Expr perf_expr;
      GetPerfVar(var_name, perf_expr, tenary_ops);
      tenary_ops[perf_expr] = TenaryOp(res);
      tenary_ops[perf_expr].SetVariable(perf_expr);
      tenary_ops[perf_expr].SetDescription(desc);
      node_perf[PipeType::AIV_VEC] = perf_expr;
    } else {
      node_perf[PipeType::AIV_VEC] = res;
    }
  } else {
    // 普通节点
    GE_ASSERT_SUCCESS(GetNodePerf(node, node_perf, tenary_ops),
                      "Get node [%s][%s] perf failed.", node.name.c_str(), node.node_type.c_str());
  }
  return ge::SUCCESS;
}

// 添加节点性能到pipe_costs
ge::Status PipePerfExpr::AddNodePerfToPipeCost(const NodeInfo &node, const Expr &exe_var,
                                               const std::map<PipeType, Expr> &node_perf,
                                               std::map<PipeType, Expr> &pipe_costs,
                                               std::map<Expr, TenaryOp, ExprCmp> &tenary_ops) {
  (void)tenary_ops;
  if (CheckSingleCut(node)) {
    GELOGD("Node with single cut, add tail perf.");
    Expr tail_exe_time;
    std::map<PipeType, Expr> node_tail_perf;
    GE_ASSERT_SUCCESS(GetNodePerf(node, node_tail_perf, tenary_ops, true),
                      "Get node [%s] tail perf failed.", node.name.c_str());
    GE_ASSERT_SUCCESS(GetTailExeTime(node, exe_var, tail_exe_time));
    GE_ASSERT_SUCCESS(AddTailPerf(tail_exe_time, exe_var, node_perf, node_tail_perf, pipe_costs));
  } else {
    GE_ASSERT_SUCCESS(AddPerf(exe_var, node_perf, pipe_costs));
  }
  return ge::SUCCESS;
}
}  // namespace att
