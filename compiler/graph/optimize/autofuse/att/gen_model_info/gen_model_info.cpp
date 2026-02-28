/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gen_model_info.h"
#include <fstream>
#include "common/util/mem_utils.h"
#include "common/autofuse_config/auto_fuse_config_utils.h"
#include "graph/utils/file_utils.h"
#include "expr_gen/generate_tiling_expr.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "parser/ascend_graph_parser.h"
#include "pass/pass_mgr.h"
#include "nlohmann/json.hpp"
#include "util/base_types_printer.h"
#include "util/duration.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "base/att_const_values.h"
#include "util/thread_local_context.h"
#include "gen_model_info/expr_gen/arg_list_reorder.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "schedule_result.h"
#include "ascgraph_info_complete.h"

namespace att {
namespace {
constexpr uint32_t kPathMax = 4096U;
const std::string kFileSeperator = "/";
const std::string kModelInfoFileName = "model_info.json";
constexpr uint32_t kConstType = 1U;
constexpr uint32_t kVarType = 2U;
constexpr uint32_t kDefaultAlignValue = 1U;
const std::string kModelInfoFilePath = "./";
}

ge::Status GenerateModelInfo(const ge::AscGraph &graph, ModelInfo &model_info, TuningSpacePtr &tuning_space,
                             const uint32_t tiling_case_id) {
  GELOGI("[DFX]Begin to generate model info for graph %s of tiling case id %u", graph.GetName().c_str(), tiling_case_id);
  DURATION_GUARD(DurationType::DURATION_GEN_MODEL_INFO);
  model_info.tiling_case_id = tiling_case_id;
  // step1: get tuningspace from compute graph
  GE_ASSERT_NOTNULL(tuning_space, "Create tuning space failed.");
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  GE_ASSERT_SUCCESS(ascend_graph_parser.GraphParser(graph), "Get tuning space failed.");
  // step2: get basic expr constraint
  att::GenerateTilingExpr tiling_expr(tuning_space);
  GE_ASSERT_SUCCESS(tiling_expr.Generate(model_info), "Get basic expr constraint failed.");
  // step3: call passes to get configs
  ATTConfig att_config;
  std::vector<PassFunc> pass_funcs;
  ATTPassMgr::Instance().GetPassList(pass_funcs);
  for (auto &pass_func : pass_funcs) {
    GE_ASSERT_NOTNULL(pass_func, "Get pass func failed.");
    std::map<std::string, std::string> config_str;
    GE_ASSERT_TRUE(pass_func(tuning_space, config_str), "Run pass failed.");
    for (auto &config : config_str) {
      GELOGD("Exe pass: config[%s], value[%s].", config.first.c_str(), config.second.c_str());
    }
    for (auto &config : config_str) {
      att_config.config_names.emplace_back(config.first);
      att_config.config_value.insert(config);
    }
  }
  GELOGI("[DFX]End to generate model info for graph %s of tiling case id %u", graph.GetName().c_str(), tiling_case_id);
  return ge::SUCCESS;
}

ge::Status CheckKeyValid(const std::vector<ge::AscGraph> &graph_list) {
  if (graph_list.size() > 0) {
    bool has_set_key = (graph_list[0].GetTilingKey() >= 0);
    for (auto &graph : graph_list) {
      GE_ASSERT_TRUE(has_set_key == (graph.GetTilingKey() >= 0),
                     "If the user has set tiling_case_id for a graph, the tiling_case_id of all graphs must be set.");
    }
  }
  return ge::SUCCESS;
}

ge::Status GenerateModelInfo(const std::vector<ge::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list) {
  GE_ASSERT_SUCCESS(CheckKeyValid(graph_list));
  uint32_t tiling_key = 0U;
  for (auto &graph : graph_list) {
    tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : tiling_key;
    ModelInfo model_info;
    TuningSpacePtr tuning_space = ge::MakeShared<TuningSpace>();
    GE_ASSERT_NOTNULL(tuning_space, "Make tuning space failed.");
    GE_ASSERT_SUCCESS(GenerateModelInfo(graph, model_info, tuning_space, tiling_key), "General model info failed.");
    model_info_list.emplace_back(model_info);
    tiling_key++;
  }
  GE_ASSERT_TRUE(!model_info_list.empty(), "No graph need convert.");
  return ge::SUCCESS;
}

void to_json(nlohmann::json &j, const SymInfoPtr &arg) {
  GE_CHECK_NOTNULL_JUST_RETURN(arg);
  auto type = 0U;
  uint32_t const_value = 0U;
  auto const_arg = std::dynamic_pointer_cast<SymConstInfo>(arg);
  if (const_arg != nullptr) {
    type = kConstType;
    const_value = const_arg->const_value;
  }

  Expr align = ge::Symbol(kDefaultAlignValue);
  std::vector<HardwareDef> related_scope;
  Expr max_value = ge::sym::kSymbolZero;
  auto var_arg = std::dynamic_pointer_cast<SymVarInfo>(arg);
  std::pair<int64_t, int64_t> value_range;
  std::string symbol_expr;
  if (var_arg != nullptr) {
    type = kVarType;
    max_value = var_arg->max_value;
    related_scope = var_arg->related_scope;
    align = var_arg->align;
    value_range = var_arg->value_range;
    symbol_expr = (!IsValid(arg->symbol_expr)) ? "" : Str(arg->symbol_expr);
  }
  const std::string max_value_str = (!IsValid(max_value)) ? "null" : Str(max_value);
  j = nlohmann::json{
      {"type", (type == kConstType) ? "const" : "var"},
      {"symbol_expr", symbol_expr},
      {"const_value", const_value},
      {"expr_max_value", max_value_str},
      {"related_scope", related_scope},
      {"align", align},
      {"min_value", value_range.first},
      {"max_value", value_range.second},
  };
}

void to_json(nlohmann::json &j, const std::shared_ptr<AttAxis> &arg) {
  GE_CHECK_NOTNULL_JUST_RETURN(arg);
  std::string orig_axis;
  for (auto axis : arg->orig_axis) {
    orig_axis += axis->name;
    orig_axis += ",";
  }
  std::string from_axis;
  for (auto axis : arg->from_axis) {
    from_axis += axis->name;
    from_axis += ",";
  }
  j = nlohmann::json{
      {"name", arg->name},
      {"axis_pos", arg->axis_pos},
      {"bind_multicore", arg->bind_multicore},
      {"is_last", arg->is_last},
      {"is_node_innerest_dim", arg->is_node_innerest_dim},
      {"size", arg->size},
      {"orig_axis", orig_axis},
      {"from_axis", from_axis},
  };
}

void to_json(nlohmann::json &j, const ATTConfig &info) {
  j = nlohmann::json{
      {"config_inputs", info.config_names},
      {"align_config", info.config_value},
  };
}

void to_json(nlohmann::json &j, const HardwareDef &type) {
  j = nlohmann::json{
      {HardwareType2Str.at(type)},
  };
}

void to_json(nlohmann::json &j, const AxisPosition &type) {
  j = nlohmann::json{
      {AxisType2Str.at(type)},
  };
}

void to_json(nlohmann::json &j, const PipeType &type) {
  j = nlohmann::json{
      {PipeType2Str.at(type)},
  };
}

void to_json(nlohmann::json &j, const ModelInfo &info) {
  j = nlohmann::json{
      {"tiling_case_id", info.tiling_case_id},
      {"hardware_cons", info.hardware_cons},
      {"eq_exprs", info.eq_exprs},
      {"leq_exprs", info.leq_exprs},
      {"objects", info.objects},
      {"arg_list", info.arg_list},
  };
}

std::string GetRealPath(const std::string &path) {
  if (path.empty() || (path.size() >= kPathMax)) {
    GELOGW("Path is size[%zu] exception.", path.size());
    return "";
  }
  std::string root_path = ge::RealPath(path.c_str());
  GE_ASSERT_TRUE(!root_path.empty(), "Invalid path: %s", path.c_str());
  if (!kFileSeperator.empty() && (root_path[root_path.size() - 1U] != kFileSeperator.back())) {
    root_path += kFileSeperator;
  }

  return root_path;
}

void DumpModelInfo(const std::vector<ModelInfo> &model_info_list, std::string dump_dir = "") {
  nlohmann::json j = nlohmann::json{
      {"model_info", model_info_list},
  };
  std::ofstream file_stream;
  std::string model_file = kModelInfoFileName;
  std::string output_path = dump_dir;
  if (output_path.empty()) {
    output_path = kModelInfoFilePath;
  }
  model_file = GetRealPath(output_path) + model_file;
  ge::char_t realpath_file[PATH_MAX] = {0x00};
  auto ret = realpath(model_file.c_str(), realpath_file);
  if (ret == nullptr) {
    GELOGD("Model file path [%s] unfound.", model_file.c_str());
    return;
  }
  file_stream.open(realpath_file, std::ios::out | std::ios::trunc);
  if (file_stream.is_open()) {
    file_stream << j.dump();
    file_stream.close();
  }
}

void ProcessTilingR(std::vector<ModelInfo> &model_info_list, const ModelInfo &model_info,
                    std::vector<AttAxisPtr> &tiling_R_arg_list) {
  if (tiling_R_arg_list.empty()) {
    return;
  }
  ModelInfo model_info_tiling_R = model_info;
  model_info_tiling_R.arg_list = tiling_R_arg_list;
  model_info_tiling_R.sub_case_tag = "R";
  model_info_list.emplace_back(model_info_tiling_R);
}

void ProcessGraphOriginalSizeVar(const ge::AscGraph &graph, ModelInfo &model_info) {
  optimize::SizeVarSet var_set;
  model_info.sizes.clear();
  optimize::AscGraphInfoComplete::AppendOriginalSizeVar(graph, var_set);
  for (const auto &var : var_set) {
    model_info.sizes.emplace_back(var);
  }
}

inline bool IsAxesReorderAlgorithm() {
  const auto res = AutoFuseConfig::MutableAttStrategyConfig().Init();
  if (res == ge::SUCCESS) {
    // 环境变量无配置则默认使用轴排序算法
    const auto &att_config = AutoFuseConfig::GetAttStrategyConfig();
    if (att_config.set_env_tiling_algorithm) {
      return att_config.tiling_algorithm == "AxesReorder";
    }
  }
  return true;
}

ge::Status GenerateModelInfo(const std::vector<ge::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list,
                             const std::map<std::string, std::string> &options,
                             bool enable_group_parallel) {
  GE_ASSERT_SUCCESS(CheckKeyValid(graph_list));
  uint32_t tiling_key = 0U;
  for (auto &graph : graph_list) {
    tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : tiling_key;
    ModelInfo model_info;
    model_info.graph_name = graph.GetName();
    model_info.tiling_case_id = tiling_key;
    model_info.enable_group_parallel = enable_group_parallel;
    std::vector<AttAxisPtr> tiling_R_arg_list;
    TuningSpacePtr tuning_space = ge::MakeShared<TuningSpace>();
    GE_ASSERT_NOTNULL(tuning_space, "Make tuning space failed.");
    tuning_space->cache_line_config = &model_info.cache_line_config;
    GetThreadLocalContext().SetOption(options);
    GE_ASSERT_SUCCESS(GenerateModelInfo(graph, model_info, tuning_space, tiling_key), "General model info failed.");
    if (IsAxesReorderAlgorithm()) {
      ArgListReorder arg_list_reorder(tuning_space);
      GE_ASSERT_SUCCESS(arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list), "Sort arg list failed.");
    }
    const auto iter = options.find(kTilingDataTypeName);
    const std::string tiling_data_type = (iter != options.cend()) ? iter->second : "";
    ApiTilingParams params{graph, tiling_data_type};
    GE_ASSERT_SUCCESS(GetApiTilingInfo(model_info.tiling_case_id, params, model_info.node_name_to_api_code),
                      "Generate api tiling info failed.");
    // 获取所有的原始size_var
    ProcessGraphOriginalSizeVar(graph, model_info);
    ProcessTilingR(model_info_list, model_info, tiling_R_arg_list);
    model_info_list.emplace_back(model_info);
    tiling_key++;
  }
  GE_ASSERT_TRUE(!model_info_list.empty(), "No graph need convert.");
  if (options.find(kDumpDebugInfo) != options.end()) {
    DumpModelInfo(model_info_list, options.at(kDumpDebugInfo).back() == '/' ? options.at(kDumpDebugInfo)
                                                                            : options.at(kDumpDebugInfo) + "/");
  }
  return ge::SUCCESS;
}

ge::Status MakeJsonIner(const std::vector<ModelInfo> &model_info_list, std::string &json_info) {
  nlohmann::json j = nlohmann::json{
      {"model_info", model_info_list},
  };
  json_info = j.dump();
  DumpModelInfo(model_info_list);
  return ge::SUCCESS;
}

ge::Status MakeJson(std::vector<ModelInfo> &model_info_list, std::string &json_info) {
  GE_ASSERT_SUCCESS(MakeJsonIner(model_info_list, json_info), "Make json failed.");
  return ge::SUCCESS;
}

ge::Status GetAllSubImplGraphs(const ascir::FusedScheduledResult &schedule_results,
                               std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> &all_graphs,
                               std::map<std::string, std::string> &all_graph_score_funcs) {
  bool has_none_graph = true;
  for (const auto &asc_graph : schedule_results.node_idx_to_scheduled_results) {
    std::vector<std::vector<std::vector<ge::AscGraph>>> cur_asc_graphs;
    for (const auto &schedule_result : asc_graph) {
      std::vector<std::vector<ge::AscGraph>> schedule_result_graphs;
      for (const auto &schedule_group : schedule_result.schedule_groups) {
        schedule_result_graphs.emplace_back(schedule_group.impl_graphs);
        all_graph_score_funcs.insert(schedule_group.graph_name_to_score_funcs.begin(),
                                     schedule_group.graph_name_to_score_funcs.end());
        if (!schedule_group.impl_graphs.empty()) {
          has_none_graph = false;
        }
      }
      cur_asc_graphs.emplace_back(schedule_result_graphs);
    }
    all_graphs.emplace_back(cur_asc_graphs);
  }
  GE_ASSERT_TRUE(!has_none_graph);
  return ge::SUCCESS;
}

namespace {
ge::Status ProcessAndSetScheduleGroupInfo(
    const std::vector<std::vector<ge::AscGraph>> &schedule_groups,
    const std::map<std::string, std::string> &all_graph_score_funcs,
    const ascir::FusedScheduledResult &schedule_results,
    const std::map<std::string, std::string> &options,
    att::ParsedScheduleResult &out_schedule_groups,
    size_t asc_graph_id, size_t impl_graph_id) {
  // 第三层表示schedule_group_id
  for (size_t schedule_group_id = 0UL; schedule_group_id < schedule_groups.size(); schedule_group_id++) {
    auto &model_info_list = out_schedule_groups.groups_tiling_model_info[schedule_group_id];
    GELOGI(
        "[DFX]Begin to gen model info for asc graph %zu, schedule result %zu, schedule group %zu, tiling_case size "
        "%zu, graph name %s.",
        asc_graph_id, impl_graph_id, schedule_group_id, schedule_groups[schedule_group_id].size(),
        !schedule_groups[schedule_group_id].empty() ? schedule_groups[schedule_group_id][0].GetName().c_str()
                                                    : "null");
    GE_ASSERT_SUCCESS(GenerateModelInfo(schedule_groups[schedule_group_id], model_info_list, options,
                                        out_schedule_groups.enable_group_parallel),
                      "Get model info failed, impl graph id = %ld, group id = %ld.", impl_graph_id,
                      schedule_group_id);
    for (auto &model_info : model_info_list) {
      model_info.schedule_group_ident.asc_graph_id = asc_graph_id;
      model_info.schedule_group_ident.impl_graph_id = impl_graph_id;
      model_info.schedule_group_ident.group_id = schedule_group_id;
      model_info.input_nodes = schedule_results.input_nodes;
      model_info.output_nodes = schedule_results.output_nodes;
      auto it = all_graph_score_funcs.find(model_info.graph_name);
      if (it != all_graph_score_funcs.end()) {
        model_info.score_func = it->second;
      }
    }
    const auto &ident = model_info_list[0].schedule_group_ident;
    GELOGI("[DFX]End to gen model info for %s tiling_case size %zu, graph name %s.", ident.GetItemPrefix().c_str(),
           schedule_groups[schedule_group_id].size(),
           !schedule_groups[schedule_group_id].empty() ? schedule_groups[schedule_group_id][0].GetName().c_str()
                                                       : "null");
    GE_ASSERT_SUCCESS(ReuseGroupUtils::InitReuseScheduleGroup(ident, model_info_list),
                      "Init reuse schedule group failed, impl_graph_id[%zu], group_ident[%s].", impl_graph_id,
                      ident.GetItemPrefix().c_str());
  }
  return ge::SUCCESS;
}
}

ge::Status GetModelInfoMap(const ascir::FusedScheduledResult &schedule_results,
                           const std::map<std::string, std::string> &options,
                           FusedParsedScheduleResult &out_all_model_infos) {
  std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> all_graphs_lists;
  std::map<std::string, std::string> all_graph_score_funcs;
  GE_ASSERT_SUCCESS(GetAllSubImplGraphs(schedule_results, all_graphs_lists, all_graph_score_funcs));
  // 最外层表示asc_graph_id
  for (size_t asc_graph_id = 0UL; asc_graph_id < all_graphs_lists.size(); asc_graph_id++) {
    auto &out_asc_graph_model_infos = out_all_model_infos[asc_graph_id];
    auto &asc_graph_list = all_graphs_lists[asc_graph_id];
    // 第二层表示impl_graph_id
    for (size_t impl_graph_id = 0UL; impl_graph_id < asc_graph_list.size(); impl_graph_id++) {
      if (asc_graph_list[impl_graph_id].empty() || asc_graph_list[impl_graph_id][0].empty()) {
        continue;
      }
      auto &schedule_groups = asc_graph_list[impl_graph_id];
      auto &out_schedule_groups = out_asc_graph_model_infos[impl_graph_id];
      const auto &schedule_result = schedule_results.node_idx_to_scheduled_results[asc_graph_id][impl_graph_id];
      GELOGD(
          "out_schedule_groups input values: score_func=%s, enable_group_parallel=%d, asc_graph_id=%zu, impl_graph_id=%zu",
          schedule_result.score_func.GetString(), schedule_result.enable_group_parallel, asc_graph_id, impl_graph_id);
      out_schedule_groups.score_func = schedule_result.score_func.GetString();
      out_schedule_groups.enable_group_parallel = schedule_result.enable_group_parallel;
      out_schedule_groups.asc_graph_id = asc_graph_id;
      out_schedule_groups.impl_graph_id = impl_graph_id;
      out_schedule_groups.var_relations = schedule_results.node_idx_to_scheduled_results[asc_graph_id][impl_graph_id].var_relations;
      GE_ASSERT_SUCCESS(ProcessAndSetScheduleGroupInfo(schedule_groups, all_graph_score_funcs, schedule_results,
                                                       options, out_schedule_groups, asc_graph_id, impl_graph_id));
    }
  }
  // 合并所有可以复用的group
  GE_ASSERT_SUCCESS(ReuseGroupUtils::MergeAllReusableGroups(all_graphs_lists, out_all_model_infos));
  return ge::SUCCESS;
}
}  // namespace att
