/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_code_gen_impl.h"
#include <fstream>
#include <set>
#include <queue>
#include <utility>
#include "args_manager.h"
#include "common/checker.h"
#include "util/duration.h"
#include "mmpa/mmpa_api.h"
#include "base_types_printer.h"
#include "base/att_const_values.h"
#include "generator_utils/tilingdata_gen_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "tiling_option_generator/tiling_option_code_generator.h"
#include "autofuse_config/auto_fuse_config.h"
#include "symbolizer/symbolic_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

namespace att {
 namespace {
 constexpr size_t kLogLength = 200;
 constexpr uint32_t kMaxDepth = 20U;
 constexpr ge::char_t kLogLevelStr[] = "ASCEND_GLOBAL_LOG_LEVEL";
 constexpr ge::char_t kEventEnableStr[] = "ASCEND_GLOBAL_EVENT_ENABLE";
 constexpr ge::char_t kInlineStr[] = "inline ";
 const std::unordered_map<int32_t, std::string> kGeLogLevelMap = {
     {DLOG_DEBUG, R"( GELOGD("[%s]" fmt, name, ##__VA_ARGS__))"},
     {DLOG_INFO, R"( GELOGI("[%s]" fmt, name, ##__VA_ARGS__))"},
     {DLOG_WARN, R"( GELOGW("[%s]" fmt, name, ##__VA_ARGS__))"},
     {DLOG_ERROR, R"( GELOGE(-1, "[%s]" fmt, name, ##__VA_ARGS__))"},
 };
 inline bool IsEventEnable() {
   ge::char_t env_path[MMPA_MAX_PATH] = {};
   bool enable = (mmGetEnv(kEventEnableStr, env_path, MMPA_MAX_PATH) == EN_OK) && (strcmp(env_path, "1") == 0);
   return enable;
 }
 
 inline int32_t GotLogLevel() {
   ge::char_t env_path[MMPA_MAX_PATH] = {};
   bool has_got = (mmGetEnv(kLogLevelStr, env_path, MMPA_MAX_PATH) == EN_OK);
   int32_t got_log_level = DLOG_ERROR;
   if (has_got) {
     got_log_level = std::atoi(env_path);
   }
   return got_log_level;
 }
 
 inline bool IsLogLevelEnable(const int64_t log_level) {
   return GotLogLevel() <= log_level;
 }
 
 inline const std::string &AddSlogExtend() {
   const static std::string kGeLogUtils = {
#include "ge_log_utils_src.h"
   };
   if (GotLogLevel() == DLOG_NULL) {
     const static std::string kNullStr;
     return kNullStr;
   }
   return kGeLogUtils;
 }
 template <typename T>
 ge::Status IsUpperBoundValid(const Expr &min_expr, const Expr &max_expr) {
   T min_value{};
   T max_value{};
   (void)min_expr.GetConstValue(min_value);
   (void)max_expr.GetConstValue(max_value);
   GE_ASSERT_TRUE(min_value <= max_value, "Args manager process failed, min[%u] can not be less than max[%u].",
                  min_value, max_value);
   return ge::SUCCESS;
 }
 
 inline const std::string &GetGeLogDefine(const int64_t log_level) {
   // event log is special, it will be enabled when event enable is true
   const bool is_enable1 = IsEventEnable();
   // other log is enabled when log level is enabled and log level is not event
   const bool is_enable2 = IsLogLevelEnable(log_level);
   const auto iter = kGeLogLevelMap.find(log_level);
   if ((is_enable1 || is_enable2) && (iter != kGeLogLevelMap.cend())) {
     return iter->second;
   }
   const static std::string NullStr;
   return NullStr;
 }
 
 void GenLogDefine(ge::CodePrinter &print) {
   const auto &slog_extend = AddSlogExtend();
   const auto &extend_define = slog_extend.empty() ? "\n" : slog_extend + "\n";
   std::string debug_log_define = std::string("#define OP_LOGD(name, fmt, ...)").append(GetGeLogDefine(DLOG_DEBUG));
   std::string info_log_define = std::string("#define OP_LOGI(name, fmt, ...)").append(GetGeLogDefine(DLOG_INFO));
   std::string warn_log_define = std::string("#define OP_LOGW(name, fmt, ...)").append(GetGeLogDefine(DLOG_WARN));
   std::string err_log_define = std::string("#define OP_LOGE(name, fmt, ...)").append(GetGeLogDefine(DLOG_ERROR));
   std::string event_log_define = std::string("#define OP_EVENT(name, fmt, ...)").append(GetGeLogDefine(DLOG_INFO));
   print.AddLine(extend_define);
   print.AddLine(debug_log_define);
   print.AddLine(info_log_define);
   print.AddLine(warn_log_define);
   print.AddLine(err_log_define);
   print.AddLine(event_log_define);
 }
 
 std::string GenParsePrint(const std::string &log_info,
                           const int32_t log_level) {
   std::string output;
   std::string log_level_str = (log_level == DLOG_ERROR) ? "E" : "I";
   for (size_t i = 0; i < log_info.size(); i += kLogLength) {
     output += "    OP_LOG" + log_level_str + "(OP_NAME, \"" + log_info.substr(i, kLogLength) + "\");\n";
   }
   return output;
 }
 
 std::string GenConsExprPrint(const ArgsManager &args_manager,
                                 const std::string &group_prefix,
                                 const int32_t log_level) {
   std::string output;
   std::string cur_log;
   for (const auto &pair : args_manager.GetTotalHardwareCons()) {
     cur_log = "Set " + BaseTypeUtils::DumpHardware(pair.first) + " for tiling case " + std::to_string(args_manager.GetTilingCaseId()) + " of " + group_prefix + " to " + Str(pair.second);
     output += GenParsePrint(cur_log, log_level);
   }
   return output;
 }
 
 std::string GenInputParamsPrint(const ArgsManager &args_manager, const std::string &group_prefix,
                                 const int32_t log_level) {
   std::string set_code;
   std::string param;
   for (const auto &arg : args_manager.GetInputVars()) {
     set_code.append(" ").append(Str(arg)).append(" = %u.");
     param.append(", tiling_data.get_").append(Str(arg)).append("()");
   }
   std::string output("    OP_LOG");
   std::string log_level_str = (log_level == DLOG_ERROR) ? "E" : "I";
   return output.append(log_level_str)
       .append("(OP_NAME, \"Set input params for tiling case ")
       .append(std::to_string(args_manager.GetTilingCaseId()))
       .append(" of ")
       .append(group_prefix)
       .append(". ")
       .append(set_code)
       .append("\"")
       .append(param)
       .append(");");
 }

 inline std::string GenScheduleResultFuncsDefine(
     const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map,
     const std::string &pgo = "") {
   std::string schedule_result_funcs_define("const std::array<ScheduleResultFunction" + pgo + ", ");
   schedule_result_funcs_define.append(std::to_string(namespace_map.size()))
       .append("> kScheduleResultFunctions" + pgo + " = {");
   for (size_t id = 0UL; id < namespace_map.size(); id++) {
     schedule_result_funcs_define.append("GetScheduleResult").append(std::to_string(id) + pgo).append(", ");
     if (id == (namespace_map.size() - 1UL)) {
       schedule_result_funcs_define.append("};");
     }
   }
   return schedule_result_funcs_define;
 }

 inline const std::string GenScheduleResultFuncTypeDefine(const std::string &tiling_data_name) {
   std::string schedule_result_func_define =
       "using ScheduleResultFunction = std::function<bool(const uint32_t ori_block_dim, const int32_t tiling_case_id, ";
   return schedule_result_func_define.append(tiling_data_name)
       .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim)>;");
 }

 inline const std::string GenPGOScheduleResultFuncTypeDefine(const std::string &tiling_data_name,
                                                             const std::string &input_output_def) {
   std::string schedule_result_func_define =
       "using ScheduleResultFunctionPGO = std::function<bool(std::vector<AutofuseTilingDataPerf>& tiling_data_list, const uint32_t ori_block_dim, const int32_t tiling_case_id, ";
  return schedule_result_func_define.append(tiling_data_name)
      .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim, " + input_output_def +
              "void* stream, uint32_t workspaceSize, std::vector<uint32_t*> block_dim_vec)>;");
}

inline const std::string GenPGOByCoreNumScheduleResultFuncTypeDefine() {
    std::string schedule_result_func_define =
      "using ScheduleResultFunctionPGOByCoreNum = std::function<bool(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData tiling_data)>;";
  return schedule_result_func_define;
}
 
 inline std::string GetScheduleResultPrefix(const size_t asc_graph_id, const size_t result_id) {
   return "AscGraph" + std::to_string(asc_graph_id) + "ScheduleResult" + std::to_string(result_id);
 }
 
 inline bool NeedGenScoreFunc(const ScoreFuncs &score_funcs) {
   // asc graph
   for (const auto &single_level_score_funcs : score_funcs) {
     // impl graph
     for (const auto &asc_graph_score_func : single_level_score_funcs.second) {
       for (const auto &impl_graph_score_func : asc_graph_score_func.second) {
         if (!impl_graph_score_func.second.empty()) {
           return true;
         }
       }
     }
   }
   return false;
 }

inline std::string GenPGOScheduleGroupDoTiling(const std::string &hardware_param,
                                               const std::string &schedule_result_prefix,
                                               const std::string &input_output) {
  return schedule_result_prefix + "::PGOSearchTilingKey(tiling_data_list_tmp, " + hardware_param + "_tiling_data, " +
         "tiling_case_id, &tiling_data, " + input_output + "stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list)";
}
 inline std::string GenGetScheduleGroupPerf(const std::string &namespace_prefix, const std::string &item_prefix) {
   return namespace_prefix + "::GetPerf(" + item_prefix + "_tiling_data)";
 }

 inline std::string GenUpdateCurPerfAndBlockByGroup() {
   return R"(
inline bool UpdateCurPerfAndBlockByGroup(std::pair<uint32_t, double> group_block_and_perf,
                                         const uint32_t limited_block,
                                         uint32_t &cur_block,
                                         double &cur_perf,
                                         double &cur_tmp_perf) {
  const auto &group_block = group_block_and_perf.first;
  const auto &group_perf = group_block_and_perf.second;
  if ((cur_block + group_block) > limited_block) {
    OP_LOGD(OP_NAME, "Cur block %u + group block %u > limited block %u, need to update cur perf %lf.",
             cur_block, group_block, limited_block, cur_tmp_perf);
    cur_block = group_block;
    cur_perf += cur_tmp_perf;
    cur_tmp_perf = group_perf;
    return true;
  } else {
    cur_block += group_block;
    cur_tmp_perf = Max(cur_tmp_perf, group_perf);
    return false;
  }
}
)";
 }

 inline std::string GenSumAllGroupsPerf(const std::vector<std::string> &groups_perf) {
   std::string sum_all_groups_perf;
   for (const auto &perf : groups_perf) {
     if (sum_all_groups_perf.empty()) {
       sum_all_groups_perf.append("      cur_perf = " + perf + ";\n");
     } else {
       sum_all_groups_perf.append("      cur_perf += " + perf + ";\n");
     }
   }
   return sum_all_groups_perf;
 }

 inline std::string GenGetCurBlockDim(const std::string &item_prefix) {
   return item_prefix + "_tiling_data.get_block_dim()";
 }

 inline std::string GenCurMaxBlockDim(const std::string &item_prefix, const std::vector<std::string> &block_num,
                                      std::string &cur_block) {
   cur_block = GenGetCurBlockDim(item_prefix);
   std::string call_max_block_dim = "Max(cur_block_dim, " + cur_block + ")";
   return "      cur_block_dim = " + (!block_num.empty() ? call_max_block_dim : cur_block) + ";";
 }

 inline bool HasSymbol(const Expr &expr) {
   return !expr.FreeSymbols().empty();
 }

 void GetRelatedInfo(const ArgsManager &args_manager, const Expr &expr, ExprExprMap &param_map, bool &related) {
   related = false;
   param_map.clear();
   ExprExprMap container_map = args_manager.GetContainerMap();
   for (const auto &arg : expr.FreeSymbols()) {
     auto iter = container_map.find(arg);
     if (iter != container_map.end()) {
       GELOGD("Add param map [%s] -> [%s].", Str(arg).c_str(), Str(iter->second).c_str());
       param_map[arg] = iter->second;
     } 
   }
   for (const auto &arg : args_manager.GetSearchableVars()) {
     if (expr.ContainVar(arg)) {
       GELOGD("Expr [%s] contain arg [%s].", Str(expr).c_str(), Str(arg).c_str());
       related = true;
     }
     for (const auto &pair : param_map) {
       if (expr.ContainVar(pair.first) && pair.second.ContainVar(arg)) {
         GELOGD("Expr [%s](%s) contain arg [%s].", Str(pair.first).c_str(), Str(pair.second).c_str(), Str(arg).c_str());
         related = true;
       }
     }
   }
 }

 ge::Status UpdateRelatedVars(const Expr &expr, const ExprExprMap &param_map, std::set<std::string> &related_vars, uint32_t depth) {
   GE_ASSERT_TRUE(depth <= kMaxDepth, "Out of max depth!");
   for (const auto &arg : expr.FreeSymbols()) {
     auto iter = param_map.find(arg);
     if (arg.GetExprType() == ge::ExprType::kExprVariable) {
       GELOGD("Analysis arg [%s].", Str(arg).c_str());
       if (iter == param_map.end()) {
         GELOGD("Arg [%s] is not a container.", Str(arg).c_str());
         related_vars.insert(Str(arg));
       } else {
         GE_ASSERT_SUCCESS(UpdateRelatedVars(iter->second, param_map, related_vars, depth + 1));
       }
     }  
   }
   return ge::SUCCESS;
 }

 bool CheckPerf(const std::string suffix, const std::string &var_name) {
   if (var_name.length() < suffix.length()) {
     return false;
   }
   return var_name.substr(var_name.length() - suffix.length()) == suffix;
 }

 inline std::string GenCallUpdateBetterTiling(bool is_uniq_group) {
   std::string workspace_param;
   if (is_uniq_group) {
     workspace_param = "";
   } else {
     workspace_param = ", workspace_map";
   }
   std::string func_params = std::string("tilingCaseImplPtr, tmp_tiling, tiling_data") +
       workspace_param +
       ", tilingCaseId";
   const std::string kUpdateBetterTilingCode = R"(
        UpdateBetterTiling()" + func_params + R"();
        sub_case_flag = is_sub_case;
        obj = cur_obj;
        ub_ratio = cur_ub_ratio;
  )";
   return kUpdateBetterTilingCode;
 }

 inline std::string GenScoreTilingCaseStruct() {
   return R"(
struct ScoreTilingCase {
  const char* sub_case_tag;
  int32_t tiling_case_id;
  TilingCaseImpl *tiling_case_ptr;
  ScoreTilingCase(const char *tag, int32_t case_id, TilingCaseImpl *case_ptr)
      :sub_case_tag(tag), tiling_case_id(case_id), tiling_case_ptr(case_ptr){}
};
)";
 }

 std::string GenTilingScoreFuncDefineHead(bool is_uniq_group) {
   std::string workspace_param = "";
   if (!is_uniq_group) {
     workspace_param = ", workspace_map";
   }
   std::string part1 = R"(  bool ret = false;
  for (const auto &s: score_map) {
    for (const auto &tiling: s.second) {
      OP_LOGD(OP_NAME, "Calculating the tiling data for tilingCaseId %s%d of score[%d]", tiling.sub_case_tag,
              tiling.tiling_case_id, s.first);
      ret |= (FindPerfBetterTilingbyCaseId(tiling.tiling_case_ptr, obj, ub_ratio, tmp_tiling, tiling_data)";

   std::string part2 = R"(, tiling.tiling_case_id, tiling.sub_case_tag[0] != 0 , sub_case_flag) || ret);
      OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId %s%d", tiling.sub_case_tag,
        tiling.tiling_case_id);
      tiling.tiling_case_ptr->~TilingCaseImpl();
    })";

   return part1 + workspace_param + part2;
 }
}  // namespace
 inline void SetTilingDefinition(const std::set<std::string> &var_names, const std::string &param_name,
                                 std::set<std::string> &tiling_data_vars,
                                 std::map<std::string, std::string> &type_name_to_definition) {
   ge::CodePrinter dumper;
   if (TilingDataGenUtils::NeedWrittenTilingData(var_names, tiling_data_vars)) {
     TilingDataGenUtils::WriteTilingDataElement(dumper, tiling_data_vars, var_names);
     type_name_to_definition[param_name] += dumper.GetOutputStr();
   }
 }
 
 inline std::vector<std::string> GetVarsNames(const std::vector<Expr> &vars) {
   std::vector<std::string> var_names;
   for (const auto &var : vars) {
     var_names.emplace_back(Str(var));
   }
   return var_names;
 }
 
 inline std::vector<std::string> GetHardwareNames(const std::map<HardwareDef, Expr> &scopes) {
   std::vector<std::string> scope_names;
   for (const auto &scope : scopes) {
     scope_names.emplace_back(BaseTypeUtils::DumpHardware(scope.first));
   }
   return scope_names;
 }
 
 inline std::vector<std::string> GetConstVarNames(const ExprUintMap &const_vars) {
   std::vector<std::string> const_var_names;
   for (const auto &const_var : const_vars) {
     const_var_names.emplace_back(GetSymbolName(const_var.first));
   }
   return const_var_names;
 }
 
 inline std::string DumpTilingData(const std::map<std::string, std::string> &tiling_data_elements) {
   std::string tiling_data_def;
   for (auto &tiling : tiling_data_elements) {
     if (tiling.second.empty()) {
       continue;
     }
     tiling_data_def += "  // definitions of " + tiling.first + "\n";
     tiling_data_def += tiling.second + "\n";
   }
   return tiling_data_def;
 }
 
 inline std::string RemoveSpace(std::string str) {
   str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
   return str;
 }

 ge::Status TilingCodeGenImpl::GenCastReuseTilingDataCode(const ReuseScheduleGroupInfo &reuse_info,
                                                          const ReuseScheduleGroupInfo &info) {
   GE_ASSERT_TRUE(reuse_info.reuse_input_axes.size() == info.reuse_input_axes.size(),
                  "reuse input axes size is not equal size: [%zu vs %zu]", reuse_info.reuse_input_axes.size(),
                  info.reuse_input_axes.size());
   GE_ASSERT_TRUE(reuse_info.reuse_search_axes.size() == info.reuse_search_axes.size(),
                  "reuse search axes size is not equal size: [%zu vs %zu]", reuse_info.reuse_search_axes.size(),
                  info.reuse_search_axes.size());
   GE_ASSERT_TRUE(reuse_info.tiling_keys.size() == info.tiling_keys.size(),
                  "reuse_keys size is not equal to info, size: [%zu vs %zu]", reuse_info.tiling_keys.size(),
                  info.tiling_keys.size());
   for (size_t i = 0UL; i < reuse_info.reuse_input_axes.size(); i++) {
     // reuse轴是原始轴
     tiling_func_.AddLine("  reuse_tiling_data.set_" + reuse_info.reuse_input_axes[i] + "(tiling_data.get_" +
                          info.reuse_input_axes[i] + "());");
   }
   for (size_t i = 0UL; i < reuse_info.tiling_keys.size(); i++) {
     std::string judge_cond = "if (tiling_data.get_tiling_key() == " + std::to_string(info.tiling_keys[i]) + ") {";
     if (i != 0UL) {
       judge_cond = " else " + judge_cond;
     }
     tiling_func_.AddLine("  " + judge_cond);
     tiling_func_.AddLine("    reuse_tiling_data.set_tiling_key(" + std::to_string(reuse_info.tiling_keys[i]) + ");");
     tiling_func_.AddLine("  }");
   }
   return ge::SUCCESS;
 }
 
 TilingCodeGenImpl::TilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                      const TilingModelInfo &tiling_model_info, const ScoreFuncs &score_funcs,
                                      const bool is_uniq_group)
     : op_name_(op_name),
       config_(config),
       tiling_data_manager_(tiling_model_info, extra_info_config_),
       extra_info_generator_(extra_info_config_, tiling_model_info, tiling_data_manager_),
       tiling_model_info_(tiling_model_info),
       is_uniq_group_(is_uniq_group),
       score_funcs_(score_funcs),
       operator_level_cache_gen_(std::make_unique<cache::OperatorLevelCacheGen>()),
       group_level_cache_gen_(std::make_unique<cache::GroupLevelCacheGen>()) {
   extra_info_config_.tiling_data_type_name = config_.tiling_data_type_name;
   if (config_.gen_extra_infos) {
     extra_info_config_.do_axes_calc = true;
     extra_info_config_.do_api_tiling = true;
   }

   // 读取编译态缓存配置
   const auto &att_config = AutoFuseConfig::GetAttStrategyConfig();
   config_.cache_enabled_at_compile_time = (att_config.enable_tiling_cache == "true");

   for (const auto &model_info : tiling_model_info) {
     const auto &hardware_cons = model_info.hardware_cons;
     if (hardware_cons.find(HardwareDef::UB) != hardware_cons.cend()) {
       hardware_has_ub_ = true;
       break;
     }
   }
   if (!config_.force_template_op_name.empty() && (config_.force_template_op_name != op_name_)) {
     config_.force_tiling_case.Clear();
     config_.force_schedule_result = -1L;
   }
   GELOGI("[DFX] Get tiling code gen config(%s)", config.Debug().c_str());
 }
 
 ge::Status TilingCodeGenImpl::GetRelatedHardware(std::map<std::string, std::string> &hardware_info) {
   std::string cur_code;
   for (const auto &model_info : tiling_model_info_) {
     ArgsManager args_manager(model_info);
     GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
     auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
     for (const auto &scope : scope_names) {
       if (hardware_info.find(scope) != hardware_info.end()) {
         continue;
       }
       auto iter = kCoreMemsizeMap.find(scope);
       if (iter != kCoreMemsizeMap.end()) {
         cur_code.clear();
         cur_code += "  uint64_t " + scope + ";\n";
         cur_code +=
             "  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::" + iter->second + ", " + scope + ");";
         hardware_info[scope] = cur_code;
       }
     }
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenDurationCommonCode() {
   const auto duration_common_code = DurationGenCommonCode();
   if (!duration_common_code.empty()) {
     tiling_head_.AddLine(duration_common_code);
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenDurationPrintCode(const std::string &indent) {
   const auto duration_print_code = DurationPrintGenCode();
   if (!duration_print_code.empty()) {
     tiling_func_.AddLine(indent + duration_print_code);
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenDurationClearCode(const std::string &indent) {
   const auto duration_clear_code = DurationClearGenCode();
   if (!duration_clear_code.empty()) {
     tiling_func_.AddLine(indent + duration_clear_code);
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenBaseTilingData(std::map<std::string, std::string> &type_name_to_definition) {
   std::set<std::string> tiling_data_vars;
   std::set<std::string> input_vars_set;
   std::set<std::string> searchable_vars_set;
   std::set<std::string> const_vars_set;
   std::set<std::string> hardware_vars_set;
   std::set<std::string> mem_vars_set;
   std::set<std::string> general_post_var_set;
   for (const auto &model_info : tiling_model_info_) {
     ArgsManager args_manager(model_info);
     GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
     auto input_vars = GetVarsNames(args_manager.GetInputVars());
     input_vars_set.insert(input_vars.begin(), input_vars.end());
     auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
     hardware_vars_set.insert(scope_names.begin(), scope_names.end());
 
     // 预留一个变量接口workspacesize，用于记录workspace信息
     std::string workspace_str = "workspaceSize";
     hardware_vars_set.insert(workspace_str);
 
     // const vars 如果没有出现在searchable vars中，那么归为InputParams。否则归为baseParams
     auto const_vars = GetConstVarNames(args_manager.GetConstVars());
     const_vars_set.insert(const_vars.begin(), const_vars.end());
     auto search_vars = GetVarsNames(args_manager.GetSearchableVars());
     searchable_vars_set.insert(search_vars.begin(), search_vars.end());
     const auto &post_datas = tiling_data_manager_.GetTilingDataWithAnnotation(
         model_info.tiling_case_id, TilingDataGenType::GENERAL_TILING_DATA_GEN);
     for (const auto &data_pair : post_datas) {
       general_post_var_set.insert(data_pair.first);
     }
     for (const auto &mem_pair : tiling_data_manager_.GetTilingDataWithAnnotation(TilingDataGenType::MEMORY_TILING_DATA_GEN)) {
       mem_vars_set.insert(mem_pair.first);
     }
   }
   for (auto &const_var : const_vars_set) {
     if (hardware_vars_set.count(const_var) == 0u) {
       input_vars_set.insert(const_var);
     }
   }
   SetTilingDefinition(input_vars_set, "InputParams", tiling_data_vars, type_name_to_definition);
   SetTilingDefinition(hardware_vars_set, "HardWareParams", tiling_data_vars, type_name_to_definition);
   SetTilingDefinition(searchable_vars_set, "BaseParams", tiling_data_vars, type_name_to_definition);
   SetTilingDefinition(general_post_var_set, "GeneralParams", tiling_data_vars, type_name_to_definition);
   SetTilingDefinition(mem_vars_set, "MemoryParams", tiling_data_vars, type_name_to_definition);
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenHeaderCodesHead() {
   std::string op_spec = RemoveSpace(op_name_);
   std::transform(op_spec.begin(), op_spec.end(), op_spec.begin(), ::toupper);
   tiling_data_.AddLine("#ifndef ATT_TILING_DATA_" + op_spec + "_H_");
   tiling_data_.AddLine("#define ATT_TILING_DATA_" + op_spec + "_H_");
   GE_ASSERT_SUCCESS(GenHeaderInclude(), "Generate tiling data head failed.");
   tiling_data_.AddLine("namespace optiling {");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenHeaderCodesTail() {
   tiling_data_.AddLine("REGISTER_TILING_DATA_CLASS(" + op_name_ + ", " + config_.tiling_data_type_name + ")");
   if(!config_.is_autofuse) {
      tiling_data_.AddLine("using AutofuseTilingData =  " + config_.tiling_data_type_name + ";\n");
      std::string pgo_perf_struct = {
      "struct AutofuseTilingDataPerf {\n"
      "  AutofuseTilingData tiling_data;\n"
      "  double best_perf;\n"
      "};\n"};
      tiling_data_.AddLine(pgo_perf_struct);
      tiling_data_.AddLine("typedef long int (*ProfilingCallback)(" + GenLaunchLikeInputOutputDef());
      tiling_data_.AddLine("void* stream, uint32_t workspaceSize, AutofuseTilingData* tiling_data, double* cost_time);");
      tiling_data_.AddLine("typedef long int (*ProfilingBatchCallback)(" + GenLaunchLikeInputOutputDef());
      tiling_data_.AddLine("void* stream, uint32_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);");
      tiling_data_.AddLine("class PgoConfig {");
      tiling_data_.AddLine("public:");
      tiling_data_.AddLine("  static PgoConfig& Instance() {");
      tiling_data_.AddLine("    static PgoConfig instance;");
      tiling_data_.AddLine("    return instance;");
      tiling_data_.AddLine("  }");
      tiling_data_.AddLine("  ProfilingCallback single_callback;");
      tiling_data_.AddLine("  ProfilingBatchCallback batch_callback;");
      tiling_data_.AddLine("  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num");
      tiling_data_.AddLine("  bool need_change_solver_run = false;");
      tiling_data_.AddLine("  size_t pgo_threshold_index = 0;");
      tiling_data_.AddLine("  constexpr static size_t pgo_threshold_list_size = 5;");
      tiling_data_.AddLine("  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};");
      tiling_data_.AddLine("  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};");
      tiling_data_.AddLine("private:");
      tiling_data_.AddLine("  PgoConfig() = default;");
      tiling_data_.AddLine("  ~PgoConfig() = default;");
      tiling_data_.AddLine("  PgoConfig(const PgoConfig &) = delete;");
      tiling_data_.AddLine("  PgoConfig &operator=(const PgoConfig &) = delete;");
      tiling_data_.AddLine("};");
   }
   GE_ASSERT_SUCCESS(GenExternFuncDef(), "Generate extern func definition failed.");
   tiling_data_.AddLine("} // namespace optiling");
   if (!config_.is_autofuse) {
    tiling_data_.AddLine("using optiling::AutofuseTilingData;");
    tiling_data_.AddLine("static uint32_t GetWorkspaceSize(const AutofuseTilingData &tiling_data) {return 0;}");
   }
   tiling_data_.AddLine("#endif");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenHeaderCodesBody() {
   GE_ASSERT_SUCCESS(tiling_data_manager_.Init());
   GE_ASSERT_SUCCESS(GenHeaderVarsDef(), "Generate vars definition failed.");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenHeaderVarsDef() {
   // 生成分的TilingData
   std::string tiling_data_def;
   std::map<std::string, std::string> base_tiling_data;
   GE_ASSERT_SUCCESS(GenBaseTilingData(base_tiling_data), "Generate base tiling data failed.");
   tiling_data_def += DumpTilingData(base_tiling_data);
   tiling_data_def += "\n";
   std::map<std::string, std::string> extra_tiling_data;
   if (config_.gen_extra_infos) {
     GE_ASSERT_SUCCESS(extra_info_generator_.GetExtraTilingDataDef(extra_tiling_data),
                       "Generate extra tiling data failed.");
   }
   tiling_data_def += DumpTilingData(extra_tiling_data);
   // 增加一个tiling_key的param
   ge::CodePrinter tiling_key_dumper;
   TilingDataGenUtils::AddElementDefinition(tiling_key_dumper, "uint32_t", "tiling_key");
   std::map<std::string, std::string> tiling_key_def = {{"TilingKeyParms", tiling_key_dumper.GetOutputStr()}};
   tiling_data_def += DumpTilingData(tiling_key_def) + "\n";
   std::string tiling_data_type_name;
   if (is_uniq_group_) {
     tiling_data_type_name = config_.tiling_data_type_name;
   } else {
     tiling_data_type_name = tiling_model_info_[0].schedule_group_ident.GetGroupPrefix() + "TilingData";
   }
   tiling_data_.AddLine(TilingDataGenUtils::StructElementDefine(tiling_data_type_name, tiling_data_def));
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GetReuseVarNames(std::map<std::string, std::string> &var_names_to_reuse_var_name) {
   std::set<ReuseScheduleGroupPtr> reuse_schedule_groups;
   for (const auto &model_info : tiling_model_info_) {
     reuse_schedule_groups.insert(model_info.reuse_schedule_group);
   }
   for (const auto &reuse_schedule_group : reuse_schedule_groups) {
     GE_ASSERT_NOTNULL(reuse_schedule_group);
     // 要求reuse的axes和schedule的axes一一对应
     for (auto &reuse_schedule : reuse_schedule_group->schedule_group_to_info) {
       for (size_t axis_id = 0UL; axis_id < reuse_schedule_group->info.reuse_input_axes.size(); axis_id++) {
         const auto &axis_name = reuse_schedule.second.reuse_input_axes[axis_id];
         const auto &reuse_axis_name = reuse_schedule_group->info.reuse_input_axes[axis_id];
         if (axis_name != reuse_axis_name) {
           var_names_to_reuse_var_name[axis_name] = reuse_axis_name;
         }
       }
     }
     for (auto &reuse_schedule : reuse_schedule_group->schedule_group_to_info) {
       for (size_t axis_id = 0UL; axis_id < reuse_schedule_group->info.reuse_search_axes.size(); axis_id++) {
         const auto &axis_name = reuse_schedule.second.reuse_search_axes[axis_id];
         const auto &reuse_axis_name = reuse_schedule_group->info.reuse_search_axes[axis_id];
         if (axis_name != reuse_axis_name) {
           var_names_to_reuse_var_name[axis_name] = reuse_axis_name;
         }
       }
     }
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenStructCopyDef() {
   std::set<std::string> tiling_data_vars;
   // 获取所有的var name和复用var的映射关系
   std::map<std::string, std::string> var_names_to_reuse_var_name;
   std::set<ReuseScheduleGroupPtr> reuse_schedule_groups;
   GE_ASSERT_SUCCESS(GetReuseVarNames(var_names_to_reuse_var_name));
   for (const auto &model_info : tiling_model_info_) {
     ArgsManager args_manager(model_info);
     GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
     auto input_vars = GetVarsNames(args_manager.GetInputVars());
     tiling_data_vars.insert(input_vars.begin(), input_vars.end());
     auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
     tiling_data_vars.insert(scope_names.begin(), scope_names.end());
     auto search_vars = GetVarsNames(args_manager.GetSearchableVars());
     tiling_data_vars.insert(search_vars.begin(), search_vars.end());
     const auto &post_datas = tiling_data_manager_.GetTilingDataWithAnnotation(
         model_info.tiling_case_id, TilingDataGenType::GENERAL_TILING_DATA_GEN);
     for (const auto &data_pair : post_datas) {
       tiling_data_vars.insert(data_pair.first);
     }
     for (const auto &data_pair : tiling_data_manager_.GetTilingDataWithAnnotation(TilingDataGenType::MEMORY_TILING_DATA_GEN)) {
       tiling_data_vars.insert(data_pair.first);
     }
     if (config_.gen_extra_infos) {
       std::set<std::string> extra_vars;
       auto extra_tiling_data_ret = extra_info_generator_.GetExtraTilingVars(model_info.tiling_case_id, extra_vars);
       if (extra_tiling_data_ret == ge::SUCCESS) {
         tiling_data_vars.insert(extra_vars.begin(), extra_vars.end());
       }
     }
   }
   tiling_data_vars.insert("tiling_key");
   tiling_data_vars.insert(BaseTypeUtils::DumpHardware(HardwareDef::CORENUM));
   if (config_.gen_extra_infos) {
     tiling_data_vars.insert("workspaceSize");
   }
   tiling_head_.AddLine("struct TilingDataCopy {");
   for (const auto &var : tiling_data_vars) {
     // 如果没有复用，则定义该变量
     std::string reuse_var = var;
     const auto &iter = var_names_to_reuse_var_name.find(var);
     if (iter == var_names_to_reuse_var_name.end()) {
       tiling_head_.AddLine("  uint32_t " + var + ";");
     } else {
       reuse_var = iter->second;
     }
     tiling_head_.AddLine("  void set_" + var + "(uint32_t val) { " + reuse_var + " = val; }");
     tiling_head_.AddLine("  inline uint32_t get_" + var + "() { return " + reuse_var + "; }");
   }
   tiling_head_.AddLine("};");
   return ge::SUCCESS;
 }

size_t TilingCodeGenImpl::CollectInputVarsSize() const {
  std::set<std::string> visited_var_names;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto input_vars = args_manager.GetInputVars();
    for (const auto &var : input_vars) {
      visited_var_names.insert(Str(var));
    }
  }
  return visited_var_names.size();
}

ge::Status TilingCodeGenImpl::GenCacheHashMapDef() {
  size_t input_vars_size = CollectInputVarsSize();

  cache::OperatorLevelCacheGen::GenConstantDefs(tiling_head_, input_vars_size);

  GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenFixedSizeHashMapDef(tiling_head_),
                    "Generate FixedSizeHashMap definition failed.");

  GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenOperatorCacheTypes(tiling_head_, config_.tiling_data_type_name),
                    "Generate Operator cache types failed.");

  if (config_.cache_enabled_at_compile_time) {
    GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenTilingCacheContext(tiling_head_, config_.tiling_data_type_name),
                      "Generate TilingCacheContext failed.");
  }

  return ge::SUCCESS;
}

 ge::Status TilingCodeGenImpl::GenHeaderInclude() {
   tiling_data_.AddLine("#include <stdint.h>");
   tiling_data_.AddLine("#include <vector>");
   tiling_data_.AddLine("#include <unordered_map>");
   tiling_data_.AddLine("#include \"register/tilingdata_base.h\"");
   tiling_data_.AddLine("#include \"tiling/tiling_api.h\"");
   return ge::SUCCESS;
 }
 
 std::vector<Expr> GetFromAxes(const Expr &hardware_arg, const ArgsManager &args_manager, const HardwareDef &hardware_def) {
  std::vector<Expr> from_axes;
  if (hardware_def == HardwareDef::UB) {
    from_axes = args_manager.GetAncestor(hardware_arg);
  } else if (hardware_def == HardwareDef::CORENUM) {
    from_axes = args_manager.GetParentVars(hardware_arg);
  }
  return from_axes;
 }

 void TilingCodeGenImpl::InitTilingUpperBound(const std::vector<Expr> &hardware_args, const ArgsManager &args_manager, 
                                               const HardwareDef &hardware_def, std::map<std::string, bool> &visited) {
   auto input_vars = GetVarsNames(args_manager.GetInputVars());
   auto const_vars = GetConstVarNames(args_manager.GetConstVars());
   for (const auto &hardware_arg : hardware_args) {
     if ((std::find(input_vars.begin(), input_vars.end(), Str(hardware_arg)) != input_vars.end()) ||
         (std::find(const_vars.begin(), const_vars.end(), Str(hardware_arg)) != const_vars.end())) {
       continue;
     }
     if (visited.find(Str(hardware_arg) + "_upper_bound") != visited.end()) {
       continue;
     }
     tiling_func_.AddLine("    int32_t " + Str(hardware_arg) + "_upper_bound = 1;");
     visited.insert({Str(hardware_arg) + "_upper_bound", true});
     std::vector<Expr> from_axes = GetFromAxes(hardware_arg, args_manager, hardware_def);
     std::string hardware_arg_value = "";
    
     for (uint32_t i = 0u; i < from_axes.size(); ++i) {
       auto primary_args = from_axes[i].FreeSymbols();
       if (primary_args.empty() && from_axes[i].IsConstExpr()) {
         hardware_arg_value += "      " + Str(hardware_arg) + "_upper_bound *= " + Str(from_axes[i]) + ";\n";
         continue;
       }
       for (uint32_t j = 0u; j < primary_args.size(); ++j) {
         auto pri_arg = primary_args[j];
         if (visited.find(Str(pri_arg)) == visited.end()) {
           hardware_arg_value += "      double " + Str(pri_arg) + " = tiling_data.get_" + Str(pri_arg) + "();\n";
           visited.insert({Str(pri_arg), true});
         } else {
           hardware_arg_value += "    " + Str(pri_arg) + " = tiling_data.get_" + Str(pri_arg) + "();\n";
         }
       }
       hardware_arg_value += "    " + Str(hardware_arg) + "_upper_bound *= " + Str(from_axes[i]) + ";\n";
     }
     tiling_func_.AddLine(hardware_arg_value);
     tiling_func_.AddLine("    tiling_data.set_" + Str(hardware_arg) + "(" + Str(hardware_arg) + "_upper_bound);");
   }
 }

 std::set<std::string> GetConsRelatedAncestors(HardwareDef hardware_def, const ArgsManager &args_manager, const std::map<HardwareDef, Expr> &hardware_cons) {
   std::set<std::string> cons_related_ancestor_vars;
   for (const auto &pair : hardware_cons) {
     if (pair.first != hardware_def) {
       continue;
     }
     auto hardware_expr = pair.second;
     auto hardware_args = hardware_expr.FreeSymbols();
     for (const auto &hardware_arg : hardware_args) {
       auto ancestor_vars = args_manager.GetAncestorNames(hardware_arg);
       for (const auto &ancestor_var : ancestor_vars) {
         GELOGD("cons_related_ancestor_vars: %s", ancestor_var.c_str());
         cons_related_ancestor_vars.insert(ancestor_var);
       }
     }
     break;
   }
   return cons_related_ancestor_vars;
 }
 
 bool TilingCodeGenImpl::HitSmallShapePattern(ArgsManager &args_manager) const {
   auto hardware_cons = args_manager.GetTotalHardwareCons(false);
   // 仅对vv融合做优化，涉及cube类不做此优化，另外ub或者多核约束缺失的不做优化
   if ((hardware_cons.find(HardwareDef::UB) == hardware_cons.end()) ||
       (hardware_cons.find(HardwareDef::CORENUM) == hardware_cons.end()) ||
       (hardware_cons.find(HardwareDef::L1) != hardware_cons.end()) ||
       (hardware_cons.find(HardwareDef::L0A) != hardware_cons.end()) ||
       (hardware_cons.find(HardwareDef::L0B) != hardware_cons.end()) ||
       (hardware_cons.find(HardwareDef::L0C) != hardware_cons.end())) {
     GELOGD("HitSmallShapePattern: not support this case");
     return false;
   }
   // 如果ub相关变量的原始轴和多核相关变量的原始轴不一致，那么不做优化
   // 例如： z0z1 is merged by [z0,z1], ub约束只和z1相关，和z0无关，z0z1用来做多核切分
   // 这个case里ub即使能全载，也不能直接返回，因为有一根轴z0和ub无关，但是和多核有关
   // 多核还有切分空间，不能直接返回1个核全载，否则会有严重的性能问题
   std::set<std::string> ub_cons_related_ancestor_vars = GetConsRelatedAncestors(HardwareDef::UB, args_manager, hardware_cons);
   std::set<std::string> mc_cons_related_ancestor_vars = GetConsRelatedAncestors(HardwareDef::CORENUM, args_manager, hardware_cons);
   if (mc_cons_related_ancestor_vars.empty() && ub_cons_related_ancestor_vars.empty()) {
     GELOGD("HitSmallShapePattern: ub_cons_related_ancestor_vars and mc_cons_related_ancestor_vars is empty");
     return false;
   }
   for (const auto &mc_var : mc_cons_related_ancestor_vars) {
    if (ub_cons_related_ancestor_vars.find(mc_var) == ub_cons_related_ancestor_vars.end()) {
      GELOGD("Cannot find ub_cons_related_ancestor_vars: %s", mc_var.c_str());
      return false;
    }
   }
   return true;
 }

 std::vector<Expr> TopoHardwareArgs(const Expr &hardware_arg, const ArgsManager &args_manager, const HardwareDef &hardware_def) {
  std::queue<Expr> expr_queue;
  std::vector<Expr> visited;
  expr_queue.push(hardware_arg);
  visited.emplace_back(hardware_arg);
  auto func = [&expr_queue, &visited](const std::vector<Expr> &primary_args) -> void {
    for (uint32_t j = 0u; j < primary_args.size(); ++j) {
      auto pri_arg = primary_args[j];
      if (std::find(visited.begin(), visited.end(), pri_arg) == visited.end()) {
        visited.emplace_back(pri_arg);
        expr_queue.push(pri_arg); 
      } 
    }
  };
  while (!expr_queue.empty()) {
    auto expr = expr_queue.front();
    expr_queue.pop();
    std::vector<Expr> from_axes = GetFromAxes(expr, args_manager, hardware_def);
    for (uint32_t i = 0u; i < from_axes.size(); ++i) {
     auto primary_args = from_axes[i].FreeSymbols();
     if (primary_args.empty() && from_axes[i].IsConstExpr()) {
       continue;
     }
     func(primary_args);
    }
  }
  return visited;
 }
 
 std::vector<Expr> ReorderHardwareArgs(const std::vector<Expr> &hardware_args, const ArgsManager &args_manager, const HardwareDef &hardware_def) {
  auto input_vars = GetVarsNames(args_manager.GetInputVars());
  auto const_vars = GetConstVarNames(args_manager.GetConstVars()); 
  std::vector<Expr> reordered_hardware_args;
   for (const auto &hardware_arg : hardware_args) {
     if ((std::find(input_vars.begin(), input_vars.end(), Str(hardware_arg)) != input_vars.end()) ||
          (std::find(const_vars.begin(), const_vars.end(), Str(hardware_arg)) != const_vars.end())) {
      continue;
     }
     std::vector<Expr> sorted_args = TopoHardwareArgs(hardware_arg, args_manager, hardware_def);
     for (int32_t i=sorted_args.size() - 1; i >= 0; --i) {
       if (std::find(hardware_args.begin(), hardware_args.end(), sorted_args[i]) == hardware_args.end()) {
         continue;
       }
       if (std::find(reordered_hardware_args.begin(), reordered_hardware_args.end(), sorted_args[i]) == reordered_hardware_args.end()) {
         reordered_hardware_args.emplace_back(sorted_args[i]); 
       }
     }
   }
   return reordered_hardware_args;
 }

 // 小shape场景如果不做tiling，原始shape全载ub，ubsize不超过设置的ub阈值大小，那么直接返回不做tiling切分
 ge::Status TilingCodeGenImpl::GenSmallShapeTiling(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   if (!HitSmallShapePattern(args_manager)) {
     return ge::SUCCESS;
   }
   tiling_func_.AddLine("  bool TrySmallShapeTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
   std::map<std::string, bool> visited;
   for (const auto &pair : args_manager.GetTotalHardwareCons()) {
     if (pair.first == HardwareDef::UB) {
       auto hardware_expr = pair.second;
       auto hardware_args = hardware_expr.FreeSymbols();
       auto reordered_hardware_args = ReorderHardwareArgs(hardware_args, args_manager, pair.first);
       InitTilingUpperBound(reordered_hardware_args, args_manager, pair.first, visited);
       // 如果设置了阈值，那么ubsize不超过阈值*原始UB大小，否则ubsize不超过原始UB大小
       if (AutoFuseConfig::GetAttStrategyConfig().enable_multicore_ub_tradeoff != "true") {
         tiling_func_.AddLine(
             "    if ((Getub_size(tiling_data) < 0) || (tiling_data.get_ub_size() < "
             "static_cast<double>(Getub_size(tiling_data)))) {");
       } else {
         tiling_func_.AddLine("    if ((Getub_size(tiling_data) < 0) || (tiling_data.get_ub_size() * " +
                              std::to_string(config_.ub_threshold) +
                              " < static_cast<double>(Getub_size(tiling_data)))) {");
       }
       tiling_func_.AddLine("      return false;");
       tiling_func_.AddLine("    }");
     }
   }
   for (const auto &pair : args_manager.GetTotalHardwareCons()) {
     if (pair.first == HardwareDef::CORENUM) {
       auto hardware_expr = pair.second;
       auto hardware_args = hardware_expr.FreeSymbols();
       auto reordered_hardware_args = ReorderHardwareArgs(hardware_args, args_manager, pair.first);
       InitTilingUpperBound(reordered_hardware_args, args_manager, pair.first, visited);
     }
   }
   tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"TilingCaseId[" + std::to_string(model_info.tiling_case_id) + "]Match small shape, apply small shape strategy.\");");
   tiling_func_.AddLine("    return true;");
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenGetTiling() {
   tiling_func_.AddLine(std::string("  bool GetTiling(") + config_.tiling_data_type_name + " &tiling_data" +
                        (hardware_has_ub_ ? ", double &cur_ub_ratio" : "") + ") {");
   tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Execute DoTiling.\");");
   if (config_.enable_small_shape_strategy) {
     tiling_func_.AddLine("    if (!TrySmallShapeTiling(tiling_data)) {");
     tiling_func_.AddLine(
         "      OP_LOGD(OP_NAME, \"The shape does not match small shape pattern. Turn to main tiling procedure\");");
   }
   tiling_func_.AddLine("    if (!DoTiling(tiling_data)) {");
   tiling_func_.AddLine("      OP_LOGW(OP_NAME, \"Failed to do tiling.\");");
   tiling_func_.AddLine("      return false;");
   tiling_func_.AddLine("    }");
   if (config_.enable_small_shape_strategy) {
     tiling_func_.AddLine("    }");
   }
   tiling_func_.AddLine("    if (is_empty_tensor_) {");
   tiling_func_.AddLine("      OP_LOGW(OP_NAME, \"Empty tensor, skip DoApiTiling and GeneralTiling.\");");
   tiling_func_.AddLine("      return true;");
   tiling_func_.AddLine("    }");
   tiling_func_.AddLine("    DoApiTiling(tiling_data);");
   tiling_func_.AddLine("    GeneralTiling(tiling_data);");
   if (config_.gen_extra_infos) {
     tiling_func_.AddLine("    GetWorkSpaceSize(tiling_data);");
     tiling_func_.AddLine("    ExtraTilingData(tiling_data);");
   }
   if (hardware_has_ub_) {
     tiling_func_.AddLine("    TilingSummary(tiling_data, cur_ub_ratio);");
   } else {
     tiling_func_.AddLine("    TilingSummary(tiling_data);");
   }
   tiling_func_.AddLine("    return true;");
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenProtectedVars() {
   tiling_func_.AddLine("  uint32_t corenum_;");
   tiling_func_.AddLine("  bool is_empty_tensor_{false};");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenTilingImplBaseClass() {
   std::string data_type = config_.tiling_data_type_name;
   tiling_func_.AddLine("class TilingCaseImpl {");
   tiling_func_.AddLine(" public:");
   tiling_func_.AddLine("  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}");
   tiling_func_.AddLine("  virtual ~TilingCaseImpl() = default;");
   GE_ASSERT_SUCCESS(GenTilingImplPublicFunc(), "Generate get tiling failed.");
   tiling_func_.AddLine(" protected:");
   if (config_.enable_small_shape_strategy) {
     tiling_func_.AddLine("  virtual bool TrySmallShapeTiling(" + data_type + " &tiling_data) { return false;}");
   }
   tiling_func_.AddLine("  virtual bool DoTiling(" + data_type + " &tiling_data) = 0;");
   tiling_func_.AddLine("  virtual void DoApiTiling(" + data_type + " &tiling_data) {}");
   tiling_func_.AddLine("  virtual void GeneralTiling(" + data_type + "& tiling_data) {}");
   if (config_.gen_extra_infos) {
     tiling_func_.AddLine("  virtual void GetWorkSpaceSize(" + data_type + "& tiling_data) {}");
     tiling_func_.AddLine("  virtual void ExtraTilingData(" + data_type + " &tiling_data) {}");
   }
   GE_ASSERT_SUCCESS(GenProtectedVars(), "Generate protected vars failed.");
   tiling_func_.AddLine("};");
   tiling_func_.AddLine("using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenCommonStruct() {
   const std::string kPipeType = R"(
enum class PipeType : uint8_t {
  AIC_MTE1 = 0,
  AIC_MTE2,
  AIC_FIXPIPE,
  AIC_MAC,
  AIV_MTE2,
  AIV_MTE3,
  AIV_VEC,
  AICORE_MTE1,
  AICORE_MTE2,
  AICORE_MTE3,
  AICORE_CUBE,
  AICORE_VEC,
  ALL,
};
)";
   const std::string kTilingOption = R"(
struct TilingOption {
  int32_t tiling_case_id{-1};
  int32_t algorithm_index{0};
};
static TilingOption tiling_option_default{};
)";
   tiling_head_.AddLine(kPipeType);
   tiling_head_.AddLine(kTilingOption);
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenCommonFrameWork() {
   GE_ASSERT_SUCCESS(GenToolFuncs(), "Generate tool funcs failed.");
   GE_ASSERT_SUCCESS(GenCommonStruct());
   GE_ASSERT_TRUE(!tiling_model_info_.empty(), "Tiling model info should not be empty.");
   GE_ASSERT_SUCCESS(GenSolverBaseClass(), "Generate base class failed.");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenHardwareCons(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   for (const auto &pair : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
     auto iter = kHardwareNameMap.find(pair.first);
     if (iter == kHardwareNameMap.end()) {
       continue;
     }
     tiling_func_.AddLine("  int Get" + iter->second + "(" + config_.tiling_data_type_name + "& tiling_data) {");
     tiling_func_.AddLine(GenBufRelatedVars(pair.second, args_manager.GetContainerMap()));
     tiling_func_.AddLine("  }");
     tiling_func_.AddLine("");
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenHardwareJudge(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   std::string name;
   std::string judge_code;
   std::set<std::string> related_vars;
   ExprExprMap param_map;
   for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
     bool related = false;
     name = BaseTypeUtils::DumpHardware(hardware.first);
     param_map.clear();
     GetRelatedInfo(args_manager, hardware.second, param_map, related);
     if (!related) {
       GELOGD("Size of param_map [%zu].", param_map.size());
       GELOGD("%s occupy is const, generating if codes.", name.c_str());
       GE_ASSERT_SUCCESS(UpdateRelatedVars(hardware.second, param_map, related_vars, 1U));
       for (const auto &pair : param_map) {
         judge_code += "    double " + Str(pair.first) + " = " + Str(pair.second) + ";\n";
       }
       std::string hardware_orig_expr = Str(hardware.second);
       judge_code.append("// ").append(name).append(" expr = ").append(hardware_orig_expr + "\n");
       Optimizer ast_optimizer;
       Parser parser(hardware_orig_expr);
       ASTPtr ast = parser.Parse();
       GE_ASSERT_NOTNULL(ast, "Parse expr failed: %s", hardware_orig_expr.c_str());
       ast_optimizer.Optimize(ast);
       std::string hardware_expr = ast_optimizer.RebuildExpr(*ast.get(), 1);
       judge_code.append(ast_optimizer.GenerateCode() + "\n");
       int64_t value = 0L;
       if (!hardware.second.IsConstExpr() || (!hardware.second.GetConstValue(value) || (value != 0UL))) {
         hardware_expr = hardware.second.IsConstExpr() ? hardware_expr.append("u") : hardware_expr;
         judge_code.append("    if (").append(hardware_expr).append(" > tiling_data.get_").append(name)
             .append("()) {\n");
         judge_code.append("      OP_LOGW(OP_NAME, \"").append(name).append(" cons unsatisfied!\");\n");
         judge_code.append("      return false;\n");
         judge_code.append("    }\n");
       }
     }
   }
   for (const auto &arg : related_vars) {
     GELOGD("Add related vars [%s].", arg.c_str());
     tiling_func_.AddLine("    uint32_t " + arg + " = tiling_data.get_" + arg + "();");
   }
   tiling_func_.AddLine(judge_code);
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenHardwareSummary(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   std::string name;
   std::string param;
   std::string set_code;
   for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
     name = BaseTypeUtils::DumpHardware(hardware.first);
     set_code += " " + name + " = %u.";
     if (hardware.first == HardwareDef::CORENUM) {
       param += ", corenum_";
     } else {
       param += ", tiling_data.get_" + name + "()";
     }
   }
   tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"Set hardware params." + set_code + "\"" + param + ");");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenInputSummary(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   tiling_func_.AddLine(GenInputParamsPrint(args_manager, model_info.schedule_group_ident.GetGroupPrefix(), DLOG_INFO));
   tiling_func_.AddLine(GenConsExprPrint(args_manager, model_info.schedule_group_ident.GetGroupPrefix(), DLOG_INFO));
   for (const auto &arg : args_manager.GetSearchableVars()) {
     Expr min_expr = args_manager.GetMinValue(arg);
     Expr max_expr = args_manager.GetMaxValue(arg);
     if (min_expr.IsConstExpr() && max_expr.IsConstExpr()) {
       if (max_expr.GetExprType() == ge::ExprType::kExprConstantRation) {
         GE_ASSERT_SUCCESS(IsUpperBoundValid<double>(min_expr, max_expr));
       } else {
         GE_ASSERT_SUCCESS(IsUpperBoundValid<uint64_t>(min_expr, max_expr));
       }
       GELOGD("Check upper bound %s and lower bound %s for %s", min_expr.Str().get(), max_expr.Str().get(),
              arg.Str().get());
     }
   }
   return ge::SUCCESS;
 }
 
 
 ge::Status TilingCodeGenImpl::GenTilingSummary(const ModelInfo &model_info) {
   std::string codes;
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   std::string case_info_str = " in " + model_info.schedule_group_ident.GetItemPrefix() + "_" + model_info.sub_case_tag + std::to_string(model_info.tiling_case_id);
   if (hardware_has_ub_) {
     tiling_func_.AddLine("  void TilingSummary(" + config_.tiling_data_type_name +
                          " &tiling_data, double& cur_ub_ratio) {");
   } else {
     tiling_func_.AddLine("  void TilingSummary(" + config_.tiling_data_type_name + " &tiling_data) {");
   }
   for (const auto &arg : args_manager.GetSearchableVars()) {
     tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + Str(arg) + " is %u" + case_info_str + ".\", tiling_data.get_" + Str(arg) + "());");
   }
   for (const auto &pair : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
     const auto &arg_name = BaseTypeUtils::DumpHardware(pair.first);
     tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + arg_name + " is %d" + case_info_str +
                          ".\", Get" + arg_name + "(tiling_data));");
   }
   for (const auto &var : model_info.container_exprs) {
     tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + var.first + " is %u" + case_info_str + ".\", tiling_data.get_" + var.first +
                          "());");
   }
   GE_ASSERT_SUCCESS(GenExtraSummaryInfo(model_info, args_manager, case_info_str), "Generate summary info failed.");
   if (hardware_has_ub_) {
     tiling_func_.AddLine(
         "    cur_ub_ratio = static_cast<double>(Getub_size(tiling_data) - " + Str(model_info.reserved_ub_size) + ") / tiling_data.get_ub_size();");
     tiling_func_.AddLine("    if (std::isnan(cur_ub_ratio)) {");
     tiling_func_.AddLine("      cur_ub_ratio = 1;");
     tiling_func_.AddLine("      OP_LOGI(OP_NAME, \"The ub ratio is NaN, set it to 1.\");");
     tiling_func_.AddLine("    }");
   }
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 } 
 
 ge::Status TilingCodeGenImpl::GenPostTiling(const ModelInfo &model_info) {
   GE_ASSERT_SUCCESS(GenDoApiTiling(model_info), "Generate do api tiling failed.");
   GE_ASSERT_SUCCESS(GenGeneralTiling(model_info), "Generate get block num failed.");
   GE_ASSERT_SUCCESS(GenEvalFunc(model_info), "Generate eval funcs failed.");
   GE_ASSERT_SUCCESS(GenMemoryParamCode(model_info), "Gen Mem param code failed.");
   if (config_.gen_extra_infos) {
     GE_ASSERT_SUCCESS(GenExtraTilingData(model_info), "Generate extra tiling data failed.");
   }
   GE_ASSERT_SUCCESS(GenTilingSummary(model_info), "Generate tiling summary failed.");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenImplPtr() {
   int idx = 0;
   tiling_func_.AddLine("TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {");
   tiling_func_.AddLine("  TilingCaseImplPtr tilingCaseImplPtr = nullptr;");
   // group_ident->case_id->sub_case_tags
   std::map<std::string, std::map<uint32_t, std::vector<std::string>>> tiling_case_id_map;
   for (const auto &model_info : tiling_model_info_) {
     tiling_case_id_map[model_info.schedule_group_ident.GetItemPrefix()][model_info.tiling_case_id].push_back(
         model_info.sub_case_tag);
   }
   for (const auto &model_info : tiling_model_info_) {
     const auto case_tags =
         tiling_case_id_map[model_info.schedule_group_ident.GetItemPrefix()][model_info.tiling_case_id];
     const auto force_sub_tag = config_.force_tiling_case.GetTag(model_info.schedule_group_ident.group_id);
     const bool is_exist_tag = std::find(case_tags.cbegin(), case_tags.cend(), force_sub_tag) != case_tags.cend();
     // 若对应tiling case中存在force_sub_tag，并且对应case不是该tag的话，则continue
     // 若不存在该force_case_tag，则继续生成该case
     if (is_exist_tag && (model_info.sub_case_tag != force_sub_tag)) {
       continue;
     }
     std::string tiling_id_str = std::to_string(model_info.tiling_case_id);
     if (idx == 0) {
       tiling_func_.AddLine("  if (tilingCaseId == " + tiling_id_str + "u) {");
     } else {
       tiling_func_.AddLine("  } else if (tilingCaseId == " + tiling_id_str + "u) {");
     }
     idx++;
     tiling_func_.AddLine("    tilingCaseImplPtr = std::make_shared<TilingCase" + model_info.sub_case_tag +
                          tiling_id_str + "Impl>(corenum);");
   }
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("  return tilingCaseImplPtr;");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::CheckImplPtr(const std::string &indent) {
   tiling_func_.AddLine(indent + "if (tilingCaseImplPtr == nullptr) {");
   GE_ASSERT_SUCCESS(GenOpLog(indent + "  ", "Pointer for tilingCaseId is null."));
   tiling_func_.AddLine(indent + "  return false;");
   tiling_func_.AddLine(indent + "}");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenOpLog(const std::string &indent, const std::string &log) {
   if (is_uniq_group_ && !config_.is_cube) {
     tiling_func_.AddLine(indent + "OP_LOGE(OP_NAME, \"" + log + "\");");
   } else {
     tiling_func_.AddLine(indent + "OP_LOGW(OP_NAME, \"" + log + "\");");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenOpLog(const std::string &indent, const std::string &uniq_log, const std::string &sched_log) {
   if (is_uniq_group_) {
     tiling_func_.AddLine(indent + "OP_LOGI(OP_NAME, \"" + uniq_log + "\");");
   } else {
     tiling_func_.AddLine(indent + "OP_LOGI(OP_NAME, \"" + sched_log + "\");");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenUsedTilingOption() {
   tiling_func_.AddLine("  TilingOption *tiling_option_used = nullptr;");
   tiling_func_.AddLine("  if (tiling_option == nullptr) {");
   tiling_func_.AddLine("    tiling_option_used = &tiling_option_default;");
   tiling_func_.AddLine("  } else {");
   tiling_func_.AddLine("    tiling_option_used = tiling_option;");
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenIsStaticShape() {
   bool is_static_graph{true};
   for (const auto &model_info : tiling_model_info_) {
     ArgsManager args_manager(model_info);
     GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
     const auto &input_vars = args_manager.GetInputVars();
     for (const auto &input_var : input_vars) {
       if (HasSymbol(input_var)) {
         GELOGD("Got dynamic shape model as input var: %s has symbol.", ge::SymbolicUtils::ToString(input_var).c_str());
         is_static_graph = false;
         break;
       }
     }
   }
   tiling_func_.AddLine(R"(extern "C" bool IsStaticShape() {)");
   std::string return_str("   return ");
   tiling_func_.AddLine(return_str + (is_static_graph ? "true" : "false") + ";");
   tiling_func_.AddLine("}");
   GELOGD("Gen IsStaticShape function success, is_static: %d", is_static_graph);
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenGetTilingImpl() {
   GE_ASSERT_SUCCESS(GenGetTilingWithOption());
   GE_ASSERT_SUCCESS(GenGetTilingWithCaseId());
   if (is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenGetTilingOptionRange());
     GE_ASSERT_SUCCESS(GenIsStaticShape());
   }
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenTilingFuncCallEntrance() {
   GE_ASSERT_SUCCESS(GenGetTilingImpl(), "Generate context tiling impl failed.");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenDurationBeginCode(const TilingFuncDurationType type, const std::string &indent) {
   const auto duration_begin_code = DurationBeginGenCode(type);
   if (!duration_begin_code.empty()) {
     tiling_func_.AddLine(indent + duration_begin_code);
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenDurationEndCode(const TilingFuncDurationType type, const std::string &indent) {
   const auto duration_end_code = DurationEndGenCode(type);
   if (!duration_end_code.empty()) {
     tiling_func_.AddLine(indent + duration_end_code);
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenExternFuncDef() {
   tiling_data_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                        " &tiling_data, int32_t tilingCaseId = -1);");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenExpressionMacro() {
   tiling_head_.AddLine("#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))");
   tiling_head_.AddLine("#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))");
   tiling_head_.AddLine("#define Log(a) (log((double)(a)))");
   tiling_head_.AddLine("#define Pow(a, b) pow(a, b)");
   tiling_head_.AddLine("#define Rational(a, b) ((double)(a) / (double)(b))");
   tiling_head_.AddLine("#define ExpectEq(a, b) ((a) == (b))");
   tiling_head_.AddLine("#define ExpectNe(a, b) ((a) != (b))");
   tiling_head_.AddLine("#define ExpectLe(a, b) ((a) <= (b))");
   tiling_head_.AddLine("#define ExpectLt(a, b) ((a) < (b))");
   tiling_head_.AddLine("#define LogicAnd(a, b) ((a) && (b))");
   tiling_head_.AddLine("#define LogicOr(a, b) ((a) || (b))");
   tiling_head_.AddLine("#define True true");
   tiling_head_.AddLine("#define False false");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenMacroInclude() {
   tiling_head_.AddLine("#include <cstdint>");
   tiling_head_.AddLine("#include <memory>");
   tiling_head_.AddLine("#include <cmath>");
   tiling_head_.AddLine("#include <cstdlib>");
   tiling_head_.AddLine("#include <memory.h>");
   tiling_head_.AddLine("#include <iostream>");
   tiling_head_.AddLine("#include <fstream>");
   tiling_head_.AddLine("#include <sstream>");
   tiling_head_.AddLine("#include <cfloat>");
   tiling_head_.AddLine("#include <algorithm>");
   tiling_head_.AddLine("#include <set>");
   tiling_head_.AddLine("#include <unordered_map>");
   tiling_head_.AddLine("#include <array>");
   tiling_head_.AddLine("#include <functional>");
   tiling_head_.AddLine("#include <chrono>");
   tiling_head_.AddLine("#include <cstdint>");
   tiling_head_.AddLine("#include <string>");
   std::set<std::string> uniq_head_files;
   for (const auto &model_info : tiling_model_info_) {
     for (const auto &node_param : model_info.node_name_to_api_code) {
       uniq_head_files.insert(node_param.second.head_files);
     }
   }
   for (const auto &head_file : uniq_head_files) {
     tiling_head_.AddLine(head_file);
   }
   GenLogDefine(tiling_head_);
   if (config_.gen_tiling_data) {
     // 如果是自己生成的tilingdata定义，需要在实现里面include 该头文件
     tiling_head_.AddLine("#include \"" + op_name_ + "_tiling_data.h\"");
   }
   GenExpressionMacro();
   tiling_head_.AddLine("#define MAX_SOLUTION 50");
   tiling_head_.AddLine("#define OP_NAME \"" + op_name_ + "\"");
   tiling_head_.AddLine("");
   GE_ASSERT_SUCCESS(GenDurationCommonCode(), "Generate duration common code failed.");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenToolFuncs() {
   tiling_head_.AddLine("inline bool IsEqual(double a, double b)");
   tiling_head_.AddLine("{");
   tiling_head_.AddLine("    const double epsilon = 1e-8;");
   tiling_head_.AddLine("    double abs = (a > b) ? (a - b) : (b - a);");
   tiling_head_.AddLine("    return abs < epsilon;");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("template<typename T1, typename T2>");
   tiling_head_.AddLine("inline double TenaryOp(bool cond, T1 a, T2 b)");
   tiling_head_.AddLine("{");
   tiling_head_.AddLine("    return static_cast<double>(cond ? a : b);");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("template<typename T>");
   tiling_head_.AddLine("inline T Ceiling(T a)");
   tiling_head_.AddLine("{");
   tiling_head_.AddLine("    T value = static_cast<T>(static_cast<int64_t>(a));");
   tiling_head_.AddLine("    return (IsEqual(value, a)) ? value : (value + 1);");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("template<typename T>");
   tiling_head_.AddLine("inline T Floor(T a)");
   tiling_head_.AddLine("{");
   tiling_head_.AddLine("    return static_cast<T>(static_cast<int64_t>(a));");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("template<typename T1, typename T2>");
   tiling_head_.AddLine("inline auto Mod(T1 a, T2 b)->decltype(a % b)");
   tiling_head_.AddLine("{");
   tiling_head_.AddLine("    return a % b;");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("template<typename T1, typename T2>");
   tiling_head_.AddLine(
       "inline auto Mod(T1 a, T2 b)->typename std::enable_if<std::is_floating_point<T1>::value || "
       "std::is_floating_point<T2>::value, decltype(std::fmod(a, b))>::type");
   tiling_head_.AddLine("{");
   tiling_head_.AddLine("    return std::fmod(a, b);");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("template<typename TI, typename TO>");
   tiling_head_.AddLine("inline TO &RefToRef(TI &ptr) {");
   tiling_head_.AddLine("  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&ptr)));");
   tiling_head_.AddLine("}");
   tiling_head_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenTilingImplPublicFunc() {
   std::string data_type = config_.tiling_data_type_name;
   GE_ASSERT_SUCCESS(GenGetTiling(), "Generate get tiling failed.");
   tiling_func_.AddLine("  virtual double GetPerf(" + data_type + " &tiling_data) { return 0.0; }");
   if (!is_uniq_group_) {
     tiling_func_.AddLine("  virtual std::string GetScheduleName() { return \"\"; }");
   }
   if (hardware_has_ub_) {
     tiling_func_.AddLine("  virtual void TilingSummary(" + data_type + " &tiling_data, double &cur_ub_ratio) = 0;");
   } else {
     tiling_func_.AddLine("  virtual void TilingSummary(" + data_type + " &tiling_data) = 0;");
   }
   if (config_.enable_autofuse_pgo) {
      tiling_func_.AddLine("  virtual bool ExecutePGOSolver(" + data_type +
                            " &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* "
                            "autofuse_tiling_data, " +
                            GenLaunchLikeInputOutputDef() + "void* stream, " +
                            "std::unordered_map<int64_t, uint64_t> &workspace_map, " +
                            "std::vector<uint32_t*> block_dim_vec={}) {");
      tiling_func_.AddLine("    return false;");
      tiling_func_.AddLine("  }");
   }
   tiling_func_.AddLine("  virtual int32_t CalcScore(const " + data_type + " &tiling_data) { return 0;}");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenVariableAnnotation(const ArgsManager &args_manager) {
   std::string tiling_id = std::to_string(args_manager.GetTilingCaseId());
   std::string annotations;
   auto variable_names = args_manager.GetContainerNames();
   auto variable_tenary_op = args_manager.GetTenaryOpReplaceVars();
   if (config_.do_variable_replace && !variable_names.empty()) {
     annotations += " Tensor used for tiling case " + tiling_id + " is:\n";
     for (const auto &pair : variable_names) {
       annotations += "  " + Str(pair.first) + ":" + pair.second + "\n";
     }
   }
   if (!variable_tenary_op.empty()) {
    annotations += " Exe time & Perf time used for tiling case " + tiling_id + " is:\n";
    for (const auto &pair : variable_tenary_op) {
      std::string variable_name = Str(pair.first);
      if (CheckPerf("_perf", variable_name) || CheckPerf("_exe_time", variable_name)) {
        annotations += "  " + variable_name + ":" + Str(pair.second) + "\n";
      }
    }
   }
   if (!annotations.empty()) {
    tiling_func_.AddLine("/*");
    tiling_func_.AddLine(annotations);
    tiling_func_.AddLine("*/");
   }
   return ge::SUCCESS;
 }

 std::string TilingCodeGenImpl::GenLaunchLikeInputOutputDef(bool is_define) {
  std::stringstream ss;
  std::string void_str = "";
  if (is_define) {
   void_str = "void* ";
  }
  int index = 0;
  for (auto input : tiling_model_info_[0].input_nodes) {
    ss << void_str << "input" << index++ << ", ";
  }
  index = 0;
  for (auto node : tiling_model_info_[0].output_nodes) {
    if (ge::ops::IsOps<ge::ascir_op::Output>(node)) {
      ss << void_str << "output" << index++ << ", ";
    }
  }
  return ss.str();
}

 ge::Status TilingCodeGenImpl::GenTilingCaseImpl(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   std::string tiling_id = std::to_string(args_manager.GetTilingCaseId());
   GE_ASSERT_SUCCESS(GenVariableAnnotation(args_manager));
   tiling_func_.AddLine("class TilingCase" + model_info.sub_case_tag + tiling_id + "Impl : public TilingCaseImpl {");
   tiling_func_.AddLine(" public:");
   tiling_func_.AddLine("  TilingCase" + model_info.sub_case_tag + tiling_id + "Impl(uint32_t corenum) : TilingCaseImpl(corenum) {\n");
   tiling_func_.AddLine("  }");
   std::string tiling_data_key_word = "AutofuseTilingData";
   if (!is_uniq_group_) {
     tiling_func_.AddLine("  std::string GetScheduleName() { return \"" +
                          model_info.schedule_group_ident.GetGroupPrefix() + "\"; }");
     tiling_data_key_word = model_info.schedule_group_ident.GetGroupPrefix() + "TilingData";
   }
   tiling_func_.AddLine(" protected:");
   tiling_func_.AddLine(" std::unordered_map<std::string, std::vector<" + tiling_data_key_word + ">> filter_map{};");
   GE_ASSERT_SUCCESS(GenPreTiling(model_info), "Generate pretiling failed.");
   if (config_.enable_small_shape_strategy) {
     GE_ASSERT_SUCCESS(GenSmallShapeTiling(model_info), "Generate small shape tiling failed.");
   }
   GE_ASSERT_SUCCESS(GenDoTiling(model_info), "Generate dotiling failed.");
   GE_ASSERT_SUCCESS(GenPostTiling(model_info), "Generate posttiling failed.");
   tiling_func_.AddLine("};");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenPreTiling(const ModelInfo &model_info) {
   (void)model_info;
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenDoApiTiling(const ModelInfo &model_info) {
   for (const auto &tiling_api_code : model_info.node_name_to_api_code) {
     tiling_func_.AddLine(tiling_api_code.second.function_impl);
     tiling_func_.AddLine("");
   }
   tiling_func_.AddLine("void DoApiTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
   for (const auto &tiling_api_code : model_info.node_name_to_api_code) {
     tiling_func_.AddLine(tiling_api_code.second.function_invoke);
   }
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenMemoryParamCode(const ModelInfo &model_info) {
   std::string func_call_code;
   std::string func_define_code;
   std::set<std::string> var_names;
   for (const auto &line :
        tiling_data_manager_.GetTilingFuncImpl(model_info.tiling_case_id, TilingDataGenType::MEMORY_TILING_DATA_GEN)) {
     tiling_func_.AddLine(line);
   }
   tiling_func_.AddLine("  void ComputeMemoryParam(" + config_.tiling_data_type_name + " &tiling_data) {");
   tiling_func_.AddLine(
       tiling_data_manager_.GetTilingFuncInvoke(model_info.tiling_case_id, TilingDataGenType::MEMORY_TILING_DATA_GEN));
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenExtraTilingFuncInvoke(const ModelInfo &model_info) {
   tiling_func_.AddLine(
       tiling_data_manager_.GetTilingFuncInvoke(model_info.tiling_case_id, TilingDataGenType::AXES_TILING_DATA_GEN));
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenGeneralTiling(const ModelInfo &model_info) {
   std::string impl_code;
   std::set<std::string> used_vars;
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   tiling_func_.AddLine("  void GeneralTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
   auto all_cons = args_manager.GetTotalHardwareCons(config_.do_variable_replace);
   if (all_cons.find(HardwareDef::CORENUM) != all_cons.end()) {
     auto expr = all_cons.at(HardwareDef::CORENUM);
     for (const auto &var : expr.FreeSymbols()) {
       if (!var.IsConstExpr()) {
         used_vars.insert(Str(var));
       }
     }
     impl_code += "    tiling_data.set_block_dim(" + Str(expr) + ");";
     for (const auto &var : used_vars) {
       tiling_func_.AddLine("    double " + var + " = static_cast<double>(tiling_data.get_" + var + "());");
     }
     tiling_func_.AddLine(impl_code);
   } else {
     GELOGW("Did not apply block split.");
     tiling_func_.AddLine("    tiling_data.set_block_dim(1);");
   }
   tiling_func_.AddLine("    ComputeMemoryParam(tiling_data);");
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenExtraTilingData(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   GE_ASSERT_SUCCESS(GenExtraTilingFuncImpl(model_info), "Gen extra tiling func failed.");
 
   std::string param = config_.tiling_data_type_name + " &tiling_data";
   tiling_func_.AddLine("  void ExtraTilingData(" + param + ") {");
   tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Start executing extra tiling for tilingCaseId " +
                        std::to_string(model_info.tiling_case_id) + ".\");");
   GE_ASSERT_SUCCESS(GenExtraTilingFuncInvoke(model_info), "Gen extra tiling invoke failed.");
   tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Execute extra tiling for tilingCaseId " +
                        std::to_string(model_info.tiling_case_id) + " successfully.\");");
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenExtraEvalFunc(const ModelInfo &model_info) {
   GE_ASSERT_SUCCESS(GenPipeTypeObj(model_info), "Generate PipeTypeObj failed.");
   GE_ASSERT_SUCCESS(GenGetObj(model_info), "Generate GetObj failed.");
   GE_ASSERT_SUCCESS(GenCalcScore(model_info), "Generate GetObj failed, graph name %s, tiling case %u",
                     model_info.graph_name.c_str(), model_info.tiling_case_id);
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenEvalFunc(const ModelInfo &model_info) {
   GE_ASSERT_SUCCESS(GenHardwareCons(model_info), "Generate HardwareCons failed.");
   GE_ASSERT_SUCCESS(GenExtraEvalFunc(model_info), "Generate ExtraEvalFunc failed.");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenExtraTilingFuncImpl(const ModelInfo &model_info) {
   for (auto &axes_tiling_data_impl :
        tiling_data_manager_.GetTilingFuncImpl(model_info.tiling_case_id, TilingDataGenType::AXES_TILING_DATA_GEN)) {
     tiling_func_.AddLine(axes_tiling_data_impl);
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenPipeTypeObj(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   auto tiling_id_str = std::to_string(args_manager.GetTilingCaseId());
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   for (const auto &pair : args_manager.GetObjectFunc()) {
     auto iter = kPipetypeNameMap.find(pair.first);
     if (iter == kPipetypeNameMap.end()) {
       continue;
     }
     tiling_func_.AddLine("  double Get" + iter->second + "(" + config_.tiling_data_type_name + "& tiling_data) {");
     tiling_func_.AddLine(GenRelatedVars({pair.second}, args_manager.GetContainerMap(), args_manager.GetTenaryOpRelatedVars()));
     tiling_func_.AddLine("    return " + Str(pair.second.Replace(args_manager.GetTenaryOpReplaceVars())) + ";");
     tiling_func_.AddLine("  }");
     tiling_func_.AddLine("");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenExtraSummaryInfo(const ModelInfo &model_info, const ArgsManager &args_manager, std::string &case_info_str) {
   (void)model_info;
   for (const auto &pair : args_manager.GetObjectFunc()) {
     auto iter = kPipetypeNameMap.find(pair.first);
     if (iter != kPipetypeNameMap.end()) {
       tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + iter->second + " is %f" + case_info_str +
           ".\", Get" + iter->second + "(tiling_data));");
     }
   }
   tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The objective value of the tiling data is %f" + case_info_str +
       ".\", GetPerf(tiling_data));");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenScheduleGroupTilingHead() {
   if (config_.gen_tiling_data) {
     GE_ASSERT_SUCCESS(GenHeaderCodesBody(), "Generate tiling data head failed.");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenExtraParamCode(const ModelInfo &model_info, std::string &pass_code) {
   std::set<std::string> tiling_vars;
   auto extra_tiling_data_ret = extra_info_generator_.GetExtraTilingVars(model_info.tiling_case_id, tiling_vars);
   if (extra_tiling_data_ret == ge::SUCCESS) {
     for (const auto &var : tiling_vars) {
       pass_code += "    to_tiling.set_" + var + "(from_tiling.get_" + var + "());\n";
     }
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenGetObj(const ModelInfo &model_info) {
   Expr expression;
   std::vector<Expr> funcs;
   Expr expr;
   std::string codes;
   ArgsManager args_manager(model_info);
   GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
   Expr head_cost = args_manager.GetHeadCost();
   tiling_func_.AddLine("  double GetPerf(" + config_.tiling_data_type_name + "& tiling_data) {");
   for (const auto &pair : args_manager.GetObjectFunc()) {
     auto iter = kPipetypeNameMap.find(pair.first);
     if (iter != kPipetypeNameMap.end()) {
       funcs.emplace_back(pair.second);
       expression = CreateExpr(iter->second.c_str());
       codes += "    double " + Str(expression) + " = " + Str(pair.second.Replace(args_manager.GetTenaryOpReplaceVars())) + ";\n";
       expr = (!IsValid(expr)) ? expression : ge::sym::Max(expr, expression);
     }
   }
   funcs.emplace_back(head_cost);
   tiling_func_.AddLine(GenRelatedVars(funcs, args_manager.GetContainerMap(), args_manager.GetTenaryOpRelatedVars()));
   tiling_func_.AddLine(codes);
   if (!IsValid(expr)) {
     tiling_func_.AddLine("    return 0;");
   } else {
     expr = ge::sym::Add(expr, head_cost);
     tiling_func_.AddLine("    return " + Str(expr) + ";");
   }
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenCalcScore(const ModelInfo &model_info) {
   if (!model_info.score_func.empty()) {
     tiling_func_.AddLine(model_info.score_func);
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenGetSetTilingImpl(const ModelInfo &model_info) {
   ArgsManager args_manager(model_info);
   args_manager.Process(false);
   std::string set_codes;
   std::string data_type = config_.tiling_data_type_name;
   for (const auto &arg : args_manager.GetInputVars()) {
     set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
   }
   for (const auto &arg : args_manager.GetSearchableVars()) {
     set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
   }
   for (const auto &var : model_info.container_exprs) {
     set_codes += "    to_tiling.set_" + var.first + "(from_tiling.get_" + var.first + "());\n";
   }
   auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
   set_codes += "    to_tiling.set_" + core_num + "(from_tiling.get_" + core_num + "());\n";
   if (config_.gen_extra_infos) {
     std::string additional_code;
     GE_ASSERT_SUCCESS(GenExtraParamCode(model_info, additional_code), "Gen ExtraParamCode failed.");
     set_codes += additional_code;
   }
   set_codes += "    to_tiling.set_tiling_key(from_tiling.get_tiling_key());\n";
 
   tiling_func_.AddLine("  void GetTilingData(TilingDataCopy &from_tiling, " + data_type + " &to_tiling) {");
   tiling_func_.AddLine(set_codes);
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("  void SetTilingData(" + data_type + " &from_tiling, TilingDataCopy &to_tiling) {");
   tiling_func_.AddLine(set_codes);
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("  void SetWorkspaceSize(" + data_type +
   " &tiling_data, std::unordered_map<int64_t, uint64_t> &workspace_map) {");
   tiling_func_.AddLine(GenWorkspaceRelatedVars(model_info.workspace_size_map, args_manager.GetContainerMap()));
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenSaveCaseNumInfo(uint32_t case_num) {
   if (IsProfilingEnabled()) {
     tiling_func_.AddLine("    SaveCaseNumInfo(" + std::to_string(case_num) + ");");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenerateInputParamsAndTiling(){
   tiling_func_.AddLine("  } else {");
   tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Calculating the tiling data for tilingCaseId %u.\", tilingCaseId);");
   GE_ASSERT_SUCCESS(GenSaveCaseNumInfo(1), "Gen SaveCaseNumInfo failed.");
   tiling_func_.AddLine("    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);");
   GE_ASSERT_SUCCESS(CheckImplPtr("    "), "Generate implptr check failed!");
   if (is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING, "    "),
                       "Generate begin code!");
   }
   tiling_func_.AddLine(std::string("    ret = tilingCaseImplPtr->GetTiling(tiling_data") +
                        (hardware_has_ub_ ? ", ub_ratio" : "") + ");");
   tiling_func_.AddLine("    tiling_data.set_tiling_key(tilingCaseId);");
   tiling_func_.AddLine(
       "    OP_LOGD(OP_NAME, \"Finish calculating the tiling data for tilingCaseId %u.\", tilingCaseId);");
   if (is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING, "    "),
                       "Generate end code!");
   }
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenDoTilingCommon(const ModelInfo &model_info,
                                                 const std::pair<std::string, std::string> &codes) {
   tiling_func_.AddLine(codes.first);
   tiling_func_.AddLine("  bool DoTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
   GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenInputSummary(model_info),
                     "Generate input summary failed, group[%s], case[%u,%s].",
                     model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                     model_info.sub_case_tag.c_str());
   GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareSummary(model_info),
                     "Generate hardware summary failed, group[%s], case[%u,%s].",
                     model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                     model_info.sub_case_tag.c_str());
   GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareJudge(model_info),
                     "Generate hardware judge failed, group[%s], case[%u,%s].",
                     model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                     model_info.sub_case_tag.c_str());
   tiling_func_.AddLine(codes.second);
   tiling_func_.AddLine("    return true;");
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenGetTilingDataFromCopy() {
  std::string set_codes;
  bool first_model_info = true;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    if (first_model_info){
      for (const auto &arg : args_manager.GetInputVars()) {
        set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
      }
      first_model_info = false;
    }
    for (const auto &arg : args_manager.GetSearchableVars()) {
      set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
    }
    for (const auto &arg : model_info.container_exprs) {
      set_codes += "    to_tiling.set_" + arg.first + "(from_tiling.get_" + arg.first + "());\n";
    }
  }
  auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  set_codes += "    to_tiling.set_" + core_num + "(from_tiling.get_" + core_num + "());\n";
  set_codes += "    to_tiling.set_tiling_key(from_tiling.get_tiling_key());";
  std::string data_type = config_.tiling_data_type_name;
  tiling_func_.AddLine("void GetScheduleGroupTilingData(TilingDataCopy &from_tiling, "+data_type+" &to_tiling) {");
  tiling_func_.AddLine(set_codes);
  tiling_func_.AddLine("}");
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenFindCacheAndSaveCache() {
  size_t input_vars_size = CollectInputVarsSize();

  GE_ASSERT_SUCCESS(group_level_cache_gen_->GenGroupCacheTypes(tiling_head_, input_vars_size, cache_capacity_),
                    "Generate Group cache types failed.");

  GE_ASSERT_SUCCESS(group_level_cache_gen_->GenGroupCacheFunctions(tiling_func_, config_.tiling_data_type_name),
                    "Generate Group cache functions failed.");

  return ge::SUCCESS;
}

void TilingCodeGenImpl::GenCalcScoreVarsDefine() {
  tiling_func_.AddLine(GenScoreTilingCaseStruct());
  std::string function_signature =
      "bool GetTilingCaseScoreFunc(const std::map<int32_t, std::vector<ScoreTilingCase>, greater<int32_t>> "
      "&score_map, double &obj, double &ub_ratio, TilingDataCopy &tmp_tiling, bool &sub_case_flag, " +
      config_.tiling_data_type_name + " &tiling_data" + (is_uniq_group_ ? "" : ", std::unordered_map<int64_t, uint64_t> &workspace_map");
  const char_t *cache_args =
      with_reuse_info_ ? ", std::array<uint32_t, kInputShapeSize> &input_shapes, GroupLevelCache *cache = nullptr" : "";
  tiling_func_.AddLine(function_signature.append(cache_args).append(") {"));
  tiling_func_.AddLine(GenTilingScoreFuncDefineHead(is_uniq_group_));
  tiling_func_.AddLine("    if (ret) {");
  if (with_reuse_info_) {
    tiling_func_.AddLine("      if (cache != nullptr) {");
    tiling_func_.AddLine("        SaveGroupCache(input_shapes, tmp_tiling, *cache);");
    tiling_func_.AddLine("      }");
  }
  tiling_func_.AddLine("      OP_LOGI(OP_NAME, \"[PROF]The score_map[%d] has been processed, tiling case %s%u of " +
                       tiling_model_info_[0].schedule_group_ident.GetItemPrefix() + " is the best choice.\",");
  tiling_func_.AddLine(R"(        s.first, sub_case_flag ? "R" : "", tiling_data.get_tiling_key());
      break;
    }
  }
  return ret;
}
)");
}

ge::Status TilingCodeGenImpl::GenAllSameScoreTilingCases(
    std::map<std::string, std::vector<const ModelInfo *>> &same_args_name_to_graphs,
    const std::vector<std::string> &ordered_assemble_args_name) {
  bool is_first_same_args_graph = true;
  for (const auto &assemble_args_name : ordered_assemble_args_name) {
    auto &same_args_graphs = same_args_name_to_graphs[assemble_args_name];
    std::sort(same_args_graphs.begin(), same_args_graphs.end(), [](const ModelInfo *a, const ModelInfo *b) -> bool {
      return (a->tiling_case_id < b->tiling_case_id) || (a->sub_case_tag < b->sub_case_tag);
    });
    if (is_first_same_args_graph) {
      std::string kInitVar = R"(    TilingCaseImpl *tilingCaseImplPtr;
    TilingDataCopy tmp_tiling;
    int32_t score = 0;
    std::map<int32_t, std::vector<ScoreTilingCase>, greater<int32_t>> score_map;)";
      tiling_func_.AddLine(kInitVar);
      is_first_same_args_graph = false;
    } else {
      tiling_func_.AddLine("    score = 0;");
      tiling_func_.AddLine("    score_map.clear();");
    }
    for (const auto &models : same_args_graphs) {
      auto model_info = models;
      std::string case_id_str = model_info->sub_case_tag + std::to_string(model_info->tiling_case_id);
      tiling_func_.AddLine(std::string("    TilingCase")
                               .append(case_id_str)
                               .append("Impl case")
                               .append(case_id_str)
                               .append("(corenum);"));
      tiling_func_.AddLine("    tilingCaseImplPtr = &case" + case_id_str + ";");
      tiling_func_.AddLine("    score = tilingCaseImplPtr->CalcScore(tiling_data);");
      tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"tiling case" + case_id_str + " of " +
                           model_info->schedule_group_ident.GetGroupPrefixSnakeCase() + " score is %d\", score);");
      tiling_func_.AddLine("    score_map[score].emplace_back(\"" + model_info->sub_case_tag + "\", " +
                           std::to_string(model_info->tiling_case_id) + ", tilingCaseImplPtr);");
    }
    std::string call_str =
        "    ret |= GetTilingCaseScoreFunc(score_map, obj, ub_ratio, tmp_tiling, sub_case_flag, tiling_data";
    std::string workspace_str = is_uniq_group_ ? "" : ", workspace_map";
    std::string cache_str = with_reuse_info_ ? ", input_shapes, cache" : "";
    tiling_func_.AddLine(call_str.append(workspace_str).append(cache_str).append(");"));
  }
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenGroupCacheLookupCode() {
  ArgsManager args_manager(tiling_model_info_[0]);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  auto input_vars = args_manager.GetInputVars();

  std::string input_shapes_init = "  std::array<uint32_t, kInputShapeSize> input_shapes = {";
  for (size_t i = 0; i < input_vars.size(); ++i) {
    if (i > 0) input_shapes_init += ", ";
    input_shapes_init += "tiling_data.get_" + Str(input_vars[i]) + "()";
  }
  input_shapes_init += "};";
  tiling_func_.AddLine(input_shapes_init);

  tiling_func_.AddLine("  if (cache != nullptr) {");
  tiling_func_.AddLine("    if (FindGroupCache(input_shapes, tiling_data, *cache)) {");
  tiling_func_.AddLine(
      "      OP_LOGD(OP_NAME, \"" + config_.tiling_data_type_name + " find cache for this shape.\");");
  tiling_func_.AddLine("      return true;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine(
      "  OP_LOGD(OP_NAME, \"" + config_.tiling_data_type_name + " find no cache, turn to main tiling procedure.\");");
  tiling_func_.AddLine("");
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenTemplateIterationLogic() {
  tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"The user didn't specify tilingCaseId, iterate all templates.\");");
  GE_ASSERT_SUCCESS(GenSaveCaseNumInfo(tiling_model_info_.size()), "Gen SaveCaseNumInfo failed.");
  // 1.对所有model info内的切分轴按照字符顺序进行排序
  std::map<std::string, std::vector<std::string>> graph_name_to_arg_list;
  for (const auto &i : tiling_model_info_) {
    std::vector<AttAxisPtr> copy_args = i.arg_list;
    std::sort(copy_args.begin(), copy_args.end(),
              [](const AttAxisPtr &a, const AttAxisPtr &b) { return a->name < b->name; });
    for (const auto &arg : copy_args) {
      graph_name_to_arg_list[i.graph_name].emplace_back(arg->name);
    }
  }
  // 2.取所有model info判断切分轴名字是否一致
  std::map<std::string, std::vector<const ModelInfo *>> same_args_name_to_graphs;
  std::vector<std::string> ordered_assemble_args_name;
  for (const auto &i : tiling_model_info_) {
    auto args_name = graph_name_to_arg_list[i.graph_name];
    std::string assemble_args_name;
    if (!args_name.empty()) {
      for (const auto &arg : args_name) {
        assemble_args_name.append(arg).append(",");
      }
    }
    auto &same_args_name_to_graph = same_args_name_to_graphs[assemble_args_name];
    if (same_args_name_to_graph.empty()) {
      ordered_assemble_args_name.emplace_back(assemble_args_name);
    }
    same_args_name_to_graph.emplace_back(&i);
  }
  // 3.若切分轴一致，则将所有model info的切分轴加入同一个score_map，保证顺序为model info的顺序为tiling case id的顺序
  GE_ASSERT_SUCCESS(GenAllSameScoreTilingCases(same_args_name_to_graphs, ordered_assemble_args_name));
  tiling_func_.AddLine("    if (ret) {");
  tiling_func_.AddLine("      OP_LOGI(OP_NAME, \"[PROF]Among the templates, tiling case %s%u of " +
                       tiling_model_info_[0].schedule_group_ident.GetItemPrefix() +
                       R"( is the best choice.", sub_case_flag ? "R" : "", tiling_data.get_tiling_key());)");
  tiling_func_.AddLine("    }");
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenGetTilingbyCaseId() {
  tiling_func_.AddLine("  if (tilingCaseId == -1) {");
  // Group级缓存查询
  if (with_reuse_info_) {
    GE_ASSERT_SUCCESS(GenGroupCacheLookupCode(), "Gen group cache lookup failed.");
  }
  GE_ASSERT_SUCCESS(GenTemplateIterationLogic(), "Gen template iteration failed.");
  GE_ASSERT_SUCCESS(GenerateInputParamsAndTiling(), "Gen GenerateInputParamsAndTiling failed.");
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenPGODefaultTiling() {
  tiling_func_.AddLine("    TilingDataCopy tmp_tiling;");
  tiling_func_.AddLine("    size_t malloc_size = 0;");
  
  GE_ASSERT_SUCCESS(GenSaveCaseNumInfo(tiling_model_info_.size()), "Gen SaveCaseNumInfo failed.");
  
  for (const auto &model_info : tiling_model_info_) {
      tiling_func_.AddLine("    malloc_size = Max(malloc_size, sizeof(TilingCase" + model_info.sub_case_tag +
                          std::to_string(model_info.tiling_case_id) + "Impl));");
  }
  
  tiling_func_.AddLine("    void* memory = malloc(malloc_size);");
  tiling_func_.AddLine("    TilingCaseImpl *tilingCaseImplPtr;");
  tiling_func_.AddLine("    double best_perf = DBL_MAX;");
  tiling_func_.AddLine("    double cur_perf = DBL_MAX;");
  tiling_func_.AddLine("    AutofuseTilingData autofuse_tiling_data_tmp;");
  tiling_func_.AddLine("    AutofuseTilingData autofuse_tiling_data_best = *autofuseTilingData;");
  
  return ge::SUCCESS;
}

 ge::Status TilingCodeGenImpl::GenPGOTilingCase(const ModelInfo& model_info) {
  std::string tiling_id_str = std::to_string(model_info.tiling_case_id);
  
  tiling_func_.AddLine("    tilingCaseImplPtr = new (memory) TilingCase" + model_info.sub_case_tag +
                      tiling_id_str + "Impl(corenum);");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Calculating the tiling data for tilingCaseId " +
                       model_info.sub_case_tag + tiling_id_str + ".\");");
  tiling_func_.AddLine("    autofuse_tiling_data_tmp = *autofuseTilingData;");
  
  if (!is_uniq_group_) {
      tiling_func_.AddLine("    autofuse_tiling_data_tmp." + 
                          model_info.schedule_group_ident.GetItemPrefix() +
                          "_tiling_data.set_tiling_key(" + tiling_id_str + ");");
  } else {
      tiling_func_.AddLine("    autofuse_tiling_data_tmp.set_tiling_key(" + tiling_id_str + ");");
  }
  
  tiling_func_.AddLine("    ret = (SearchAllTilingbyCaseId(tilingCaseImplPtr, tiling_data, tiling_data_list, " +
                        tiling_id_str + "u, &autofuse_tiling_data_tmp, " +
                        GenLaunchLikeInputOutputDef(false) + "stream, workspace_map, block_dim_vec) || ret);");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Finish calculating the tiling data for tilingCaseId " +
                       model_info.sub_case_tag + tiling_id_str + ".\");");
  tiling_func_.AddLine("    tilingCaseImplPtr->~TilingCaseImpl();");
  
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenEnableGroupParallelPgoInvoke(const std::string &tiling_name, bool is_pointer,
                                                              const std::string &indent, std::string &invoke_code) {
  std::map<std::string, std::set<std::string>> hardware_map;
  FusedGraphNamespaceMap namespace_map;
  GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
  std::string access;
  std::string obj_arg;
  if (is_pointer) {
    access = "->";
    obj_arg = std::to_string('*') + tiling_name;
  } else {
    access = ".";
    obj_arg = tiling_name;
  }

  std::stringstream ss;
  for (const auto &asc_graph_map_iter : namespace_map) {
    const auto &asc_graph_id = asc_graph_map_iter.first;
    const auto &asc_graph_namespace_map = asc_graph_map_iter.second;
    for (const auto &result_id_and_groups : asc_graph_namespace_map) {
      const auto &result_id = result_id_and_groups.first;
      if (enable_group_parallels_[asc_graph_id][result_id]) {
        ss << indent << "if (" << tiling_name << access << "get_graph" << asc_graph_id
           << "_tiling_key() == " << result_id << ") {" << std::endl;
        ss << indent << "  ArrangeBlockOffsetsAscGraph" << asc_graph_id << "Result" << result_id
           << "(" << obj_arg << ", " << tiling_name << access << "get_block_dim());" << std::endl;
        ss << indent << "}" << std::endl;
      }
    }
  }
  invoke_code = ss.str();
  return ge::SUCCESS;
}

 ge::Status TilingCodeGenImpl::GenPGOGetTilingbyCaseId() {
  GE_ASSERT_SUCCESS(GenPGODefaultTiling(), "Gen default tiling failed.");
  
  for (const auto &model_info : tiling_model_info_) {
      std::string tiling_id_str = std::to_string(model_info.tiling_case_id);
      GE_ASSERT_SUCCESS(GenPGOTilingCase(model_info), 
                       "Gen tiling case %s failed.", tiling_id_str.c_str());
  }
  
  return ge::SUCCESS;
}
 
 ge::Status TilingCodeGenImpl::GenUpdateBetterTiling() {
   tiling_func_.AddLine("void UpdateBetterTiling(TilingCaseImpl *tilingCaseImplPtr, TilingDataCopy &tmp_tiling, "
                            + config_.tiling_data_type_name + " &tiling_data" +
       (is_uniq_group_ ?  "" : ", std::unordered_map<int64_t, uint64_t> &workspace_map") + ", uint32_t tilingCaseId) {");
   tiling_func_.AddLine("  OP_LOGD(OP_NAME, \"The solution for tilingCaseId %u is better, updating the tiling data.\", tilingCaseId);");
   tiling_func_.AddLine("  tiling_data.set_tiling_key(tilingCaseId);");
   tiling_func_.AddLine("  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);");
   if (!is_uniq_group_) {
     tiling_func_.AddLine("  tilingCaseImplPtr->SetWorkspaceSize(tiling_data, workspace_map);");
   }
   tiling_func_.AddLine("  OP_LOGD(OP_NAME, \"Set the output tiling data.\");");
   tiling_func_.AddLine("  OP_LOGD(OP_NAME, \"Updated the best tilingCaseId to %u.\", tilingCaseId);");
   tiling_func_.AddLine("}");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenSelectBetterTilingBasedOnObjAndUbRatio() {
   double ub_threshold_perf_val_effect = 0.0;
   double perf_effect_val = 0.0;
   GE_ASSERT_TRUE(!tiling_model_info_.empty());
   if (tiling_model_info_[0].tiling_schedule_config_table != nullptr) {
     ub_threshold_perf_val_effect = tiling_model_info_[0].tiling_schedule_config_table->GetUbThresholdPerfValEffect();
     perf_effect_val = tiling_model_info_[0].tiling_schedule_config_table->GetPerfEffectVal();
     std::string perf_effect_val_str = std::to_string(perf_effect_val);
     tiling_func_.AddLine("    if (obj < 0) {");
     tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_));
     tiling_func_.AddLine("      return true;");
     tiling_func_.AddLine("    }");
     tiling_func_.AddLine("    double ub_ratio_diff = cur_ub_ratio > ub_ratio ? (cur_ub_ratio - ub_ratio) : (ub_ratio - cur_ub_ratio);");
     tiling_func_.AddLine("    if ((cur_obj - obj > " + perf_effect_val_str + ")) {\n");
     tiling_func_.AddLine("        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
     tiling_func_.AddLine("    } else if ((obj - cur_obj > " + perf_effect_val_str + ")) {");
     tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_));
     tiling_func_.AddLine("    } else if (cur_ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) +
                          " && ub_ratio >= " + std::to_string(ub_threshold_perf_val_effect) + ") {");
     tiling_func_.AddLine("        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
     tiling_func_.AddLine("    } else if (cur_ub_ratio >= " + std::to_string(ub_threshold_perf_val_effect) +
                          " && ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) + ") {");
     tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_));
     tiling_func_.AddLine("    } else if (cur_ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) +
                          " && ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) +
                          " && !IsEqual(cur_ub_ratio, ub_ratio)) {");
     tiling_func_.AddLine("        if (cur_ub_ratio > ub_ratio) {");
     tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_));
     tiling_func_.AddLine("        } else {");
     tiling_func_.AddLine("          tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
     tiling_func_.AddLine("        }");
     tiling_func_.AddLine("    } else {");
     tiling_func_.AddLine("      if (cur_obj < obj) {");
     tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_));
     tiling_func_.AddLine("      } else {");
     tiling_func_.AddLine("        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
     tiling_func_.AddLine("      }");
     tiling_func_.AddLine("    }");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenFindPerfBetterTilingbyCaseId() {
   tiling_func_.AddLine(
       "bool FindPerfBetterTilingbyCaseId(TilingCaseImpl *tilingCaseImplPtr, double &obj, double &ub_ratio, "
       "TilingDataCopy &tmp_tiling, " +
       config_.tiling_data_type_name + " &tiling_data,\n  " +
       (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
       "uint32_t tilingCaseId, bool is_sub_case, bool &sub_case_flag) {");
   tiling_func_.AddLine("  double cur_obj;");
   if (hardware_has_ub_) {
     tiling_func_.AddLine("  double cur_ub_ratio;");
     GE_ASSERT_SUCCESS(CheckImplPtr("  "), "Generate implptr check failed!");
     tiling_func_.AddLine("  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);");
     tiling_func_.AddLine(std::string("  if (tilingCaseImplPtr->GetTiling(tiling_data, cur_ub_ratio)) {"));
     tiling_func_.AddLine("    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);");
     if (!is_uniq_group_) {
       tiling_func_.AddLine("    std::string schedule_name = tilingCaseImplPtr->GetScheduleName();");
       tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"The ub ratio for tilingCaseId %u of %s is %f.\", tilingCaseId, schedule_name.c_str(), cur_ub_ratio);");
       tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"The optimal objection for tilingCaseId %u of %s is %f.\", tilingCaseId, schedule_name.c_str(), cur_obj);");
     } else {
       tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"The ub ratio for tilingCaseId %u is %f.\", tilingCaseId, cur_ub_ratio);");
       tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"The optimal objection for tilingCaseId %u is %f.\", tilingCaseId, cur_obj);");
     }
     GenSelectBetterTilingBasedOnObjAndUbRatio();
   } else {
     GE_ASSERT_SUCCESS(CheckImplPtr("  "), "Generate implptr check failed!");
     tiling_func_.AddLine("  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);");
     tiling_func_.AddLine(std::string("  if (tilingCaseImplPtr->GetTiling(tiling_data)) {"));
     tiling_func_.AddLine("    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);");
     tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"The optimal objection for tilingCaseId %u is %f.\", tilingCaseId, cur_obj);");
     tiling_func_.AddLine("    if (obj < 0 || cur_obj < obj) {");
     tiling_func_.AddLine(std::string("      UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data") + (is_uniq_group_ ? "" : ", workspace_map") + ", tilingCaseId);");
     tiling_func_.AddLine("      sub_case_flag = is_sub_case;");
     tiling_func_.AddLine("      obj = cur_obj;");
     tiling_func_.AddLine("    } else {");
     tiling_func_.AddLine("      tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
     tiling_func_.AddLine("    }");
   }
   tiling_func_.AddLine("    return true;");
   tiling_func_.AddLine("  } else {");
   tiling_func_.AddLine("    tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("  return false;");
   tiling_func_.AddLine("}");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenSearchAllTilingbyCaseId() {
   tiling_func_.AddLine("bool SearchAllTilingbyCaseId(TilingCaseImpl *tilingCaseImplPtr, " +
                        config_.tiling_data_type_name + " &tiling_data" +
                        ", std::vector<AutofuseTilingDataPerf>& tiling_data_list" +
                        ", uint32_t tilingCaseId, AutofuseTilingData* autofuseTilingData, " +
                        GenLaunchLikeInputOutputDef() + "void* stream, " +
                        "std::unordered_map<int64_t, uint64_t> &workspace_map, " +
                        "std::vector<uint32_t*> block_dim_vec={}) {");

   tiling_func_.AddLine("    tiling_data.set_tiling_key(tilingCaseId);");
   tiling_func_.AddLine(
       "    if (!tilingCaseImplPtr->ExecutePGOSolver(tiling_data, tiling_data_list, autofuseTilingData, " +
       GenLaunchLikeInputOutputDef(false) + "stream, workspace_map, block_dim_vec)) {");
   tiling_func_.AddLine(
       "      OP_LOGW(OP_NAME, \"Failed to execute PGO solver for tilingCaseId %d .\", tilingCaseId);");
   tiling_func_.AddLine("      return false;");
   tiling_func_.AddLine("    }");
   tiling_func_.AddLine(
       "    OP_LOGD(OP_NAME, \"Execute PGO solver for tilingCaseId %d successfully.\", tilingCaseId);");
   tiling_func_.AddLine("    return true;");
   tiling_func_.AddLine("}");
   tiling_func_.AddLine("");
   return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::ValidateSingleResultAndGroup() {
  tiling_func_.AddLine("  if (!ret) {");
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    int32_t log_level = (is_uniq_group_ && !config_.is_cube) ? DLOG_ERROR : DLOG_INFO;
    tiling_func_.AddLine(
        GenInputParamsPrint(args_manager, model_info.schedule_group_ident.GetGroupPrefix(), log_level));
  }
  GE_ASSERT_SUCCESS(GenOpLog("    ", "Failed to execute tiling func."));
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenGetTilingKey() {
   if (with_reuse_info_) {
     GE_ASSERT_SUCCESS(GenGetTilingDataFromCopy(), "Gen GetTilingDataFromCopy failed.");
     GE_ASSERT_SUCCESS(GenFindCacheAndSaveCache(), "Gen FindCacheAndSaveCache failed.");
   }
   GE_ASSERT_SUCCESS(GenUpdateBetterTiling(), "Gen UpdateBetterTiling failed.");
   GE_ASSERT_SUCCESS(GenFindPerfBetterTilingbyCaseId(), "Gen FindPerfBetterTilingbyCaseId failed.");
   std::string params = config_.tiling_data_type_name + " &tiling_data" +
                        ( is_uniq_group_ ? "" : ", std::unordered_map<int64_t, uint64_t> &workspace_map") +
                        ", int32_t tilingCaseId = -1";
   const ge::char_t *cache_str = (with_reuse_info_) ? ", GroupLevelCache *cache = nullptr" : "";
   GenCalcScoreVarsDefine();
   tiling_func_.AddLine("bool GetTilingKey(" + params + cache_str + ") {");
   tiling_func_.AddLine("  bool ret = false;");
   tiling_func_.AddLine("  bool sub_case_flag = false;");
   tiling_func_.AddLine("  double obj = -1;");
   tiling_func_.AddLine("  double ub_ratio = -1;");
   auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
   tiling_func_.AddLine("  uint32_t corenum = tiling_data.get_" + core_num + "();");
   GE_ASSERT_SUCCESS(GenGetTilingbyCaseId(), "Gen GetTilingbyCaseId failed.");
   GE_ASSERT_SUCCESS(ValidateSingleResultAndGroup(), "Gen ValidateSingleResultAndGroup failed.");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenPGOSearchTilingKey() {
   GE_ASSERT_SUCCESS(GenSearchAllTilingbyCaseId(), "Gen SearchAllTilingbyCaseId failed.");
   std::string params = config_.tiling_data_type_name + " &tiling_data" +
                        ", int32_t tilingCaseId";
   tiling_head_.AddLine("bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params +
                        ", AutofuseTilingData* autofuseTilingData," + GenLaunchLikeInputOutputDef() +
                        "void* stream, uint32_t workspaceSize, double& out_best_perf, std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec={});");
   tiling_func_.AddLine("bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params +
                        ", AutofuseTilingData* autofuseTilingData," + GenLaunchLikeInputOutputDef() +
                        "void* stream, uint32_t workspaceSize, double& out_best_perf, std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec) {");
   tiling_func_.AddLine("  bool ret = false;");
   tiling_func_.AddLine("  double obj = -1;");
   tiling_func_.AddLine("  double ub_ratio = -1;");
   auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
   tiling_func_.AddLine("  uint32_t corenum = tiling_data.get_" + core_num + "();");
   GE_ASSERT_SUCCESS(GenPGOGetTilingbyCaseId(), "Gen GetTilingbyCaseId failed.");
   if (is_uniq_group_) {
     tiling_func_.AddLine("  workspaceSize = 0;");
     tiling_func_.AddLine("  for (const auto &tiling_data_perf : tiling_data_list) {");
     tiling_func_.AddLine("    auto workspaceSizeTmp = GetWorkspaceSize(tiling_data_perf.tiling_data);");
     tiling_func_.AddLine("    if (workspaceSizeTmp > workspaceSize) {");
     tiling_func_.AddLine("      workspaceSize = workspaceSizeTmp;");
     tiling_func_.AddLine("    }");
     tiling_func_.AddLine("  }");
     tiling_func_.AddLine("  workspaceSize += 16 * 1024 * 1024;");
     tiling_func_.AddLine("  PgoConfig::Instance().batch_callback(" + GenLaunchLikeInputOutputDef(false) + "stream, workspaceSize, &tiling_data_list);");
     tiling_func_.AddLine("  for (const auto &tiling_data_perf : tiling_data_list) {");
     tiling_func_.AddLine("    if (best_perf > tiling_data_perf.best_perf) {");
     tiling_func_.AddLine("      best_perf = tiling_data_perf.best_perf;");
     tiling_func_.AddLine("    }");
     tiling_func_.AddLine("  }");
   }
   GE_ASSERT_SUCCESS(ValidateSingleResultAndGroup(), "Gen ValidateSingleResultAndGroup failed.");
   return ge::SUCCESS;
 }

  ge::Status TilingCodeGenImpl::GenPGOByCoreNumSearchTilingKeyCollectTilingData(FusedGraphNamespaceMap namespace_map) {
    for (auto &asc_graph_map_iter : namespace_map) {
      size_t asc_graph_id = asc_graph_map_iter.first;
      tiling_func_.AddLine("    for (auto ascgraph_tiling_data_" + std::to_string(asc_graph_id) + " : vec" +
                           std::to_string(asc_graph_id) + ") {");
    }

    tiling_func_.AddLine("      AutofuseTilingData tiling_data_tmp;");
    tiling_func_.AddLine("      tiling_data_tmp = *tiling_data; // 用于初始化部分常量参数");

    for (auto &asc_graph_map_iter : namespace_map) {
      auto &asc_graph_map = asc_graph_map_iter.second;
      size_t asc_graph_id = asc_graph_map_iter.first;
      tiling_func_.AddLine("      tiling_data_tmp.set_graph" + std::to_string(asc_graph_id) +
                           "_tiling_key(ascgraph_tiling_data_" + std::to_string(asc_graph_id) + ".get_graph" +
                           std::to_string(asc_graph_id) + "_tiling_key());");
      for (auto &graph_info_map : asc_graph_map) {
        auto graph_info = graph_info_map.second;
        for (auto &group_info : graph_info) {
          auto schedule_result_prefix = group_info.second.second;
          tiling_func_.AddLine("      tiling_data_tmp." + schedule_result_prefix + "_tiling_data = ascgraph_tiling_data_"
                               + std::to_string(asc_graph_id) + "." + schedule_result_prefix + "_tiling_data" + ";");
        }
      }
    }

    tiling_func_.AddLine("      tiling_data_list.push_back(tiling_data_tmp);");
    for ([[maybe_unused]] size_t i = 0; i < namespace_map.size(); ++i) {
      tiling_func_.AddLine("    }");
    }
    return ge::SUCCESS;
  }

  ge::Status TilingCodeGenImpl::GenPGOByCoreNumSearchTilingKeySingleGroup() {
    for (auto model_info : tiling_model_info_) {
      tiling_func_.AddLine("    tiling_case = " + std::to_string(model_info.tiling_case_id) + ";");
      tiling_func_.AddLine("    tiling_data->set_block_dim(block_dim_i);");
      tiling_func_.AddLine("    tiling_data->set_tiling_key(tiling_case);");
      tiling_func_.AddLine("    if (GetTiling(*tiling_data, tiling_case)) {");
      tiling_func_.AddLine("      tiling_data_tmp = *tiling_data;");
      tiling_func_.AddLine("      tiling_data_list.push_back(tiling_data_tmp);");
      tiling_func_.AddLine("    }");
    }
    return ge::SUCCESS;
  }

  ge::Status TilingCodeGenImpl::GenPGOByCoreNumSearchTilingKey() {
    tiling_func_.AddLine("bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* tiling_data, uint32_t max_block_dim) {");
    tiling_func_.AddLine("  bool ret = true;");
    tiling_func_.AddLine("  for (uint32_t block_dim_i=1; block_dim_i <= max_block_dim; block_dim_i++) {");
    tiling_func_.AddLine("    int32_t tiling_case;");
    tiling_func_.AddLine("    AutofuseTilingData tiling_data_tmp;");
    if (is_uniq_group_) {
      GenPGOByCoreNumSearchTilingKeySingleGroup();
    }
    tiling_func_.AddLine("   }");
    tiling_func_.AddLine("  return ret;");
    tiling_func_.AddLine("}");
    return ge::SUCCESS;
  }

 ge::Status TilingCodeGenImpl::GenHeaderCodesSummaryBody() {
   // add fixed tiling data
   ge::CodePrinter dumper;
   std::set<std::string> fixed_var_name;
   std::set<size_t> tiling_keys;
   std::set<uint32_t> case_id_set;
   for (const auto &model_info : tiling_model_info_) {
     if (!case_id_set.insert(model_info.tiling_case_id).second) {
       continue;
     }
     tiling_keys.insert(model_info.schedule_group_ident.asc_graph_id);
   }
   for (const size_t tiling_key : tiling_keys) {
     fixed_var_name.insert("graph" + std::to_string(tiling_key) + "_tiling_key");
   }
   std::set<std::string> keep_uniq;
   for (const auto &model_info : tiling_model_info_) {
     ArgsManager args_manager(model_info);
     GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
     auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
     fixed_var_name.insert(scope_names.begin(), scope_names.end());
   }
   TilingDataGenUtils::WriteTilingDataElement(dumper, keep_uniq, fixed_var_name);
   // add struct tiling data
   std::map<std::string, std::string> struct_set;
   for (const auto &model_info : tiling_model_info_) {
     struct_set[model_info.schedule_group_ident.GetGroupPrefix() + "TilingData"] =
         model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data";
   }
   for (const auto &pair : struct_set) {
     TilingDataGenUtils::WriteTilingDataStruct(dumper, keep_uniq, pair.first, pair.second);
   }
   tiling_data_.AddLine(TilingDataGenUtils::StructElementDefine(config_.tiling_data_type_name, dumper.GetOutputStr()));
   return ge::SUCCESS;
 }

 void TilingCodeGenImpl::GenTilingHeadMultiGroup() {
   std::string params = config_.tiling_data_type_name + " &tiling_data, int32_t tilingCaseId";
   if (config_.enable_autofuse_pgo) {
      tiling_head_.AddLine(
          "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params +
          ", AutofuseTilingData* autofuseTilingData," + GenLaunchLikeInputOutputDef() +
          "void* stream, uint32_t workspaceSize, double& best_perf);");
  }
 }
 
 ge::Status TilingCodeGenImpl::GenTilingHead(std::map<std::string, std::string> &tiling_res,
                                            const EnableGroupParallels &enable_group_parallels) {
   enable_group_parallels_ = enable_group_parallels;
   std::map<std::string, std::set<std::string>> hardware_map;
   FusedGraphNamespaceMap namespace_map;
   GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
   // 1、生成总TilingData
   tiling_head_.Reset();
   tiling_func_.Reset();
   tiling_data_.Reset();
   tiling_func_.AddLine("#include \"" + kDefaultTilingHeadFileName + "\"");
   GE_ASSERT_SUCCESS(tiling_data_manager_.Init());
   if (config_.gen_tiling_data) {
     GE_ASSERT_SUCCESS(GenHeaderCodesHead(), "Generate tiling data head failed.");
   }
   // 2、生成公共的TilingFunc代码
   GE_ASSERT_SUCCESS(GenMacroInclude(), "Generate macro include failed.");
   tiling_head_.AddLine("namespace optiling{};");
   tiling_head_.AddLine("using namespace optiling;");
   tiling_head_.AddLine("uint32_t GetWorkspaceSize(const AutofuseTilingData &tiling_data);");
   tiling_head_.AddLine("namespace optiling {");
   tiling_func_.AddLine("namespace optiling {");
   if (!is_uniq_group_) {
      GenTilingHeadMultiGroup();
   }
   if (config_.enable_autofuse_pgo) {
      tiling_head_.AddLine("bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, "
                           "AutofuseTilingData* tiling_data, uint32_t max_block_dim);");
   }
   tiling_head_.AddLine("using namespace std;");
   for (const auto &asc_graph_map_iter : namespace_map) {
     const auto &asc_graph_id = asc_graph_map_iter.first;
     const auto &asc_graph_namespace_map = asc_graph_map_iter.second;
     for (const auto &result_id_and_groups : asc_graph_namespace_map) {
        const auto &result_id = result_id_and_groups.first;
        if (enable_group_parallels_[asc_graph_id][result_id]) {
          tiling_head_.AddLine("void ArrangeBlockOffsetsAscGraph" + std::to_string(asc_graph_id) +
                                "Result" + std::to_string(result_id) + "(AutofuseTilingData &t, uint32_t aiv_num);");
        }
     }
   }
   GE_ASSERT_SUCCESS(GenCommonFrameWork(), "Generate common framework failed.");
   tiling_func_.AddLine("} // namespace optiling");
   if (config_.gen_tiling_data) {
     tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
   }
   tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
   tiling_res[kTilingSolverIdentify] += tiling_func_.GetOutputStr();
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::ObtainInnerParams(std::map<std::string, std::set<std::string>> &hardware_map,
                                                 FusedGraphNamespaceMap &namespace_map) {
   std::string obj_name;
   for (const auto &model_info : tiling_model_info_) {
     obj_name = model_info.schedule_group_ident.GetItemPrefix();
     auto &asc_graph_namespace_map = namespace_map[model_info.schedule_group_ident.asc_graph_id];
     auto &schedule_result_namespace_map = asc_graph_namespace_map[model_info.schedule_group_ident.impl_graph_id];
     auto &schedule_group_namespace_map = schedule_result_namespace_map[model_info.schedule_group_ident.group_id];
     schedule_group_namespace_map = std::make_pair(model_info.schedule_group_ident.GetGroupPrefix(), obj_name);
     ArgsManager args_manager(model_info);
     GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
     for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
       hardware_map[obj_name].insert(BaseTypeUtils::DumpHardware(hardware.first));
     }
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenGetResultSummary(const size_t asc_graph_id) {
   tiling_func_.AddLine("bool GetResultSummary(const double best_perf, " + config_.tiling_data_type_name +
                        " &tiling_data) {");
   tiling_func_.AddLine("  if (IsEqual(best_perf, -1)) {");
   tiling_func_.AddLine("    OP_LOGE(OP_NAME, \"GetTiling Failed.\");");
   tiling_func_.AddLine("    return false;");
   tiling_func_.AddLine("  }");
   std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
   tiling_func_.AddLine(
       "  OP_LOGI(OP_NAME, \"[PROF]Among all schedule results, " + tiling_key_prefix + "result%u is the best choice.\", "
       "tiling_data.get_" +
       tiling_key_prefix + "tiling_key());");
   tiling_func_.AddLine("    return true;");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenGetTilingForAllInitLines(bool pgo) {
   tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"Start GetTiling.\");");
   tiling_func_.AddLine("  double cur_perf;");
   if (!pgo) {
     tiling_func_.AddLine("  double best_perf = -1;");
   }
   tiling_func_.AddLine("  uint32_t cur_block_dim = 1;");
   tiling_func_.AddLine("  uint32_t ori_block_dim = tiling_data.get_block_dim();");
   if (!pgo) {
     GenUsedTilingOption();
   }
   return ge::SUCCESS;
 }

 void TilingCodeGenImpl::GenCacheInit() {
   if (with_reuse_info_) {
     std::unordered_set<std::string> declared_cache_types_; // 防止重复声明
     for (const auto &pair : cache_reuse_info_) {
       if (declared_cache_types_.find(pair.second) == declared_cache_types_.end()) {
         tiling_func_.AddLine("  " + pair.second + "::GroupLevelCache " + pair.second + "_Cache;");
         declared_cache_types_.insert(pair.second);
       }
     }
   }
 }

 inline ge::Expression GetInputVarFromSrcVarExpr(const ge::Expression &src_var_expr, const std::string &src_tiling_data_name) {
    std::unordered_set<std::string> contain_vars;
    for (const auto &arg : src_var_expr.FreeSymbols()) {
      if (arg.GetExprType() == ge::ExprType::kExprVariable) {
        contain_vars.insert(Str(arg));
      }
    }
    std::vector<std::pair<Expr, Expr>> var_replacement;
    for (auto &var : contain_vars) {
      var_replacement.emplace_back(std::make_pair(CreateExpr(var.c_str()), CreateExpr(("static_cast<double>(" + src_tiling_data_name + "_tiling_data.get_" + var +"())").c_str())));
    }
    return src_var_expr.Replace(var_replacement);
 }

 inline ge::Expression GetInputVarFromSrcVarExprWithPrefix(const ge::Expression &src_var_expr, const std::string &src_tiling_data_name) {
    std::unordered_set<std::string> contain_vars;
    for (const auto &arg : src_var_expr.FreeSymbols()) {
      if (arg.GetExprType() == ge::ExprType::kExprVariable) {
        contain_vars.insert(Str(arg));
      }
    }
    std::vector<std::pair<Expr, Expr>> var_replacement;
    for (auto &var : contain_vars) {
      var_replacement.emplace_back(std::make_pair(CreateExpr(var.c_str()), CreateExpr(("static_cast<double>(tiling_data." + src_tiling_data_name + "_tiling_data.get_" + var +"())").c_str())));
    }
    return src_var_expr.Replace(var_replacement);
 }

  inline std::pair<std::string, bool> ProcessVarRelations(const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
      const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &var_relation, size_t group_id) {
    std::string input_vars_set_code;
    bool need_update = false;
    auto it = var_relation.find(group_id);
    if (it != var_relation.end()) {
      for (const auto &var_expr_pair : it->second) {
        size_t src_id = var_expr_pair.first;
        for (const auto &pair : var_expr_pair.second) {
          need_update = true;
          auto src_it = graph_info.find(src_id);
          auto dst_it = graph_info.find(group_id);
          if (src_it != graph_info.end() && dst_it != graph_info.end()) {
            auto dst_expr = GetInputVarFromSrcVarExpr(pair.second, src_it->second.second);
            input_vars_set_code += dst_it->second.second + "_tiling_data.set_" + pair.first + "(" +
            Str(dst_expr) + "), ";
          }
        }
      }
    }
    return {input_vars_set_code, need_update};
  }

  inline std::pair<std::string, bool> ProcessVarRelationsStatement(const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                                                   const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &var_relation,
                                                                   size_t group_id, const std::string &prefix) {
    std::string input_vars_set_code;
    bool need_update = false;
    auto it = var_relation.find(group_id);
    if (it != var_relation.end()) {
      for (const auto &var_expr_pair : it->second) {
        size_t src_id = var_expr_pair.first;
        for (const auto &pair : var_expr_pair.second) {
          auto src_it = graph_info.find(src_id);
          auto dst_it = graph_info.find(group_id);
          if (src_it != graph_info.end() && dst_it != graph_info.end()) {
            auto dst_expr = GetInputVarFromSrcVarExprWithPrefix(pair.second, src_it->second.second);
            input_vars_set_code += prefix + dst_it->second.second + "_tiling_data.set_" + pair.first + "(" +
            Str(dst_expr) + ");\n";
            need_update = true;
          }
        }
      }
    }
    return {input_vars_set_code, need_update};
  }

  void TilingCodeGenImpl::GenSetHardwareCodes(const std::string& group_prefix, const std::set<std::string>& hardware_names) {
    for (const auto& hardware_name : hardware_names) {
      std::string set_code("  ");
      set_code.append(group_prefix).append("_tiling_data.set_").append(hardware_name);
      bool is_block_dim = (hardware_name == "block_dim");
      std::string hardware_val = is_block_dim
                                 ? "(ori_block_dim);"
                                 : "(tiling_data.get_" + hardware_name + "());";
      tiling_func_.AddLine(set_code.append(hardware_val));
    }
  }

  void TilingCodeGenImpl::GenGetScheduleResultTail(const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
    for (const auto &group_info : graph_info) {
      tiling_func_.AddLine("  " + group_info.second.second + "_tiling_data.set_block_dim(0);");
    }
    tiling_func_.AddLine("  return false;");
    tiling_func_.AddLine("}");
  }

  void TilingCodeGenImpl::GenUpdateWorkspace(const size_t asc_graph_id, const size_t impl_graph_id) {
    for (const auto &tensor_id : workspace_tensor_id_set_[asc_graph_id][impl_graph_id]) {
      auto tensor_id_str = to_string(tensor_id);
      tiling_func_.AddLine("      auto it" + tensor_id_str + " = workspace_map.find(" + tensor_id_str + ");");
      tiling_func_.AddLine("      if (it" + tensor_id_str + " != workspace_map.end()) {");
      tiling_func_.AddLine("        tiling_data.set_workspace" + tensor_id_str + "(it" + tensor_id_str + "->second);");
      tiling_func_.AddLine("      }");
    }
  }

  ge::Status TilingCodeGenImpl::GenUpdatePerf(const size_t asc_graph_id, const size_t impl_graph_id,
                                              const std::vector<std::string> &groups_perf,
                                              const std::vector<std::string> &groups_block_num,
                                              const std::vector<std::string> &assign_max_block_num) {
    if (!IsScheduleResultEnableParallel(asc_graph_id, impl_graph_id)) {
      tiling_func_.AddLine(GenSumAllGroupsPerf(groups_perf));
    } else {
      if (groups_perf.size() == 1UL) {
        tiling_func_.AddLine("  cur_perf = " + groups_perf[0] + ";");
      } else {
        std::string update_code("    cur_perf = 0.0;\n");
        (void)update_code.append("    bool has_update = false;\n")
            .append("    auto cur_tmp_perf = ")
            .append(groups_perf[0])
            .append(";\n")
            .append("    auto cur_block = ")
            .append(groups_block_num[0])
            .append(";\n");
        for (size_t id = 1UL; id < groups_perf.size(); ++id) {
          (void)update_code.append("    has_update = UpdateCurPerfAndBlockByGroup({")
              .append(groups_block_num[id])
              .append(", ")
              .append(groups_perf[id])
              .append("}, ori_block_dim, cur_block, cur_perf, cur_tmp_perf);\n");
        }
        (void)update_code.append("    OP_LOGD(OP_NAME, \"Begin to add group perf %lf\", cur_tmp_perf);\n")
            .append("    cur_perf += cur_tmp_perf;\n");
        tiling_func_.AddLine(update_code);
      }
    }
    tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"The value of graph" + std::to_string(asc_graph_id) + "_result" +
                         std::to_string(impl_graph_id) + " is %lf\", cur_perf);");
    tiling_func_.AddLine("    if (IsEqual(best_perf, -1) || cur_perf < best_perf) {");
    tiling_func_.AddLine("      best_perf = cur_perf;");
    for (const auto &code : assign_max_block_num) {
      tiling_func_.AddLine(code);
    }
    tiling_func_.AddLine("      tiling_data.set_block_dim(cur_block_dim);");
    std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
    tiling_func_.AddLine("      tiling_data.set_" + tiling_key_prefix + "tiling_key(" + std::to_string(impl_graph_id) +
                         ");");
    tiling_func_.AddLine(
        "      OP_LOGI(OP_NAME, \"Update best perf to %lf, tiling key = %u, block dim = %u\", best_perf, "
        "tiling_data.get_" +
        tiling_key_prefix + "tiling_key(), cur_block_dim);");
    GenUpdateWorkspace(asc_graph_id, impl_graph_id);
    tiling_func_.AddLine("      return true;");
    tiling_func_.AddLine("    }");
    tiling_func_.AddLine("    return true;");
    tiling_func_.AddLine("  }");
    return ge::SUCCESS;
  }

  ge::Status TilingCodeGenImpl::GenScheduleGroupDoTiling(std::string &check_cond, const std::string &hardware_param,
                                                         const std::string &schedule_result_prefix) {
    std::string tiling_hyphens = "&&";
    if (check_cond.empty()) {
      tiling_hyphens = "";
    }
    std::string first_param = hardware_param + "_tiling_data, ";
    std::string cache_param;
    const auto &key = schedule_result_prefix;
    for (const auto &pair : cache_reuse_info_) {
      if (pair.first == key || pair.second == key) {
        // 看是否为复用方/被复用方
        cache_param = ", &" + pair.second + "_Cache";
        break;
      }
    }
    std::string workspace_param;
    if (!is_uniq_group_) {
      workspace_param = "workspace_map, ";
    }
    std::string tiling = "(" + schedule_result_prefix + "::GetTiling(" + first_param + workspace_param +
                         "tiling_case_id" + cache_param + "))";
    check_cond += (tiling_hyphens + tiling);
    return ge::SUCCESS;
  }

  ge::Status TilingCodeGenImpl::GenGetScheduleResult(
      const size_t asc_graph_id, const size_t impl_graph_id,
      const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
      const std::map<std::string, std::set<std::string>> &hardware_map) {
    const auto var_relation = var_relations_[asc_graph_id][impl_graph_id];
    std::string check_cond;
    std::string func_define(kInlineStr);
    func_define.append("bool GetScheduleResult")
        .append(std::to_string(impl_graph_id))
        .append("(const uint32_t ori_block_dim, const int32_t tiling_case_id,")
        .append(config_.tiling_data_type_name)
        .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim) {");
    tiling_func_.AddLine(func_define);
    tiling_func_.AddLine("  std::unordered_map<int64_t, uint64_t> workspace_map{};");
    GenCacheInit();
    std::vector<std::string> assign_max_block_num;
    std::vector<std::string> groups_perf;
    std::vector<std::string> groups_block_num;
    for (const auto &group_info : graph_info) {
      auto [input_vars_set_code, need_update_second_group_input_vars] =
          ProcessVarRelations(graph_info, var_relation, group_info.first);
      std::string cur_block;
      tiling_func_.AddLine("  auto &" + group_info.second.second + "_tiling_data = tiling_data." +
                           group_info.second.second + "_tiling_data;");
      const auto &hardware_iter = hardware_map.find(group_info.second.second);
      if (hardware_iter != hardware_map.cend()) {
        GenSetHardwareCodes(group_info.second.second, hardware_iter->second);
        if (need_update_second_group_input_vars) {
          std::string tiling_hyphens = check_cond.empty() ? "" : "&&";
          check_cond += (tiling_hyphens + "(" + input_vars_set_code + "true)");
        }
        GE_ASSERT_SUCCESS(GenScheduleGroupDoTiling(check_cond, group_info.second.second, group_info.second.first),
                          "Gen schedule group do tiling failed, graph id[%zu], impl id[%zu]", asc_graph_id,
                          impl_graph_id);
        groups_perf.emplace_back(GenGetScheduleGroupPerf(group_info.second.first, group_info.second.second));
        assign_max_block_num.emplace_back(GenCurMaxBlockDim(group_info.second.second, groups_block_num, cur_block));
        groups_block_num.emplace_back(GenGetCurBlockDim(group_info.second.second));
      }
    }
    GE_ASSERT_TRUE(groups_perf.size() > 0UL, "groups_perf size of asc_graph_id %zu impl_graph_id %zu is 0",
                   asc_graph_id, impl_graph_id);
    GE_ASSERT_EQ(groups_block_num.size(), groups_perf.size());
    tiling_func_.AddLine("  if (" + (check_cond.empty() ? "true" : check_cond) + ") {");
    GE_ASSERT_SUCCESS(GenUpdatePerf(asc_graph_id, impl_graph_id, groups_perf, groups_block_num, assign_max_block_num),
                      "Gen update perf failed, asc_graph_id %zu impl_graph_id %zu", asc_graph_id, impl_graph_id);
    GenGetScheduleResultTail(graph_info);
    return ge::SUCCESS;
  }

  void TilingCodeGenImpl::GenPGOByCoreNumDoTiling(const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                                                 const uint32_t group_index, const size_t asc_graph_id, const size_t impl_graph_id) {
   auto hard_ware_param = group_info.second.second;
   auto schedule_result_prefex = group_info.second.first;
   auto group_id = group_info.first;
   uint32_t index = 0U;

   for (auto &model_info : tiling_model_info_) {
     if (model_info.schedule_group_ident.asc_graph_id != asc_graph_id || model_info.schedule_group_ident.impl_graph_id != impl_graph_id || model_info.schedule_group_ident.group_id != group_id) {
       continue;
     }

     tiling_func_.AddLine("    "+ config_.tiling_data_type_name +" tiling_data_tmp" + std::to_string(index) + "= tiling_data;");
     tiling_func_.AddLine("    auto sub_tiling_data_tmp" + std::to_string(index) + "= tiling_data_tmp" + std::to_string(index) + "." + hard_ware_param + "_tiling_data" + ";");
     tiling_func_.AddLine("    sub_tiling_data_tmp" + std::to_string(index) + ".set_tiling_key(" + std::to_string(model_info.tiling_case_id) + ");");
     tiling_func_.AddLine("    if (" + schedule_result_prefex + "::GetTiling(sub_tiling_data_tmp" + std::to_string(index) + ", workspace_map, " +std::to_string(model_info.tiling_case_id) + ")) { ");
     std::string tiling_data_add("      ");
     tiling_data_add.append("tiling_data_tmp" + std::to_string(index) + ".").append(hard_ware_param).append("_tiling_data=sub_tiling_data_tmp" + std::to_string(index) +";");
     tiling_func_.AddLine(tiling_data_add);
     tiling_func_.AddLine("      tiling_data_list_tmp" + std::to_string(group_index) + ".push_back(tiling_data_tmp" + std::to_string(index) + ");");
     tiling_func_.AddLine("    }");
     index++;
   }
 }

 void TilingCodeGenImpl::GenPGOByCoreNumGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id, 
                                                          const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                                          const std::map<std::string, std::set<std::string>> &hardware_map,
                                                          const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &var_relation) {
   std::string func_define(kInlineStr);
   func_define.append("bool GetScheduleResult")
       .append(std::to_string(impl_graph_id) + "PGOByCoreNum")
       .append("(std::vector<" + config_.tiling_data_type_name +">& tiling_data_list, ")
       .append(config_.tiling_data_type_name + " tiling_data")
       .append(") {");

   tiling_func_.AddLine(func_define);
   tiling_func_.AddLine("  std::unordered_map<int64_t, uint64_t> workspace_map{};");
   uint32_t group_index = 0U;
   tiling_func_.AddLine("  " + config_.tiling_data_type_name + " tiling_data_tmp = tiling_data;");
   tiling_func_.AddLine("  std::vector<" + config_.tiling_data_type_name +"> tiling_data_list_tmp0 = {tiling_data_tmp};");

   for (const auto &group_info : graph_info) {
     group_index++;
     tiling_func_.AddLine("  std::vector<" + config_.tiling_data_type_name +"> tiling_data_list_tmp" + std::to_string(group_index) + ";");
     tiling_func_.AddLine("  for (auto &tiling_data : tiling_data_list_tmp" + std::to_string(group_index - 1) + ") {");
     auto [input_vars_set_code, need_update_second_group_input_vars] =
         ProcessVarRelationsStatement(graph_info, var_relation, group_info.first, "    ");
     std::string tiling_item_name = group_info.second.second + "_tiling_data";
     tiling_func_.AddLine("    auto &" + tiling_item_name + " = tiling_data." + tiling_item_name + ";");
     const auto &hardware_iter = hardware_map.find(group_info.second.second);
     if (hardware_iter != hardware_map.cend()) {
       for (const auto &hardware_name : hardware_iter->second) {
         std::string set_hardware_code("    ");
         set_hardware_code.append(tiling_item_name).append(".set_").append(hardware_name);
         std::string hardware_val = "(tiling_data.get_" + hardware_name  + "());";
         tiling_func_.AddLine(set_hardware_code.append(hardware_val));
       }
       if (need_update_second_group_input_vars) {
         tiling_func_.AddLine(input_vars_set_code);
       }
       GenPGOByCoreNumDoTiling(group_info, group_index, asc_graph_id, impl_graph_id);
     }
     tiling_func_.AddLine("  }");
     tiling_func_.AddLine("");
   }
   tiling_func_.AddLine("  for (auto &tiling_data : tiling_data_list_tmp" + std::to_string(group_index) + ") {");
   GenPGOUpdateTilingInfo(asc_graph_id, impl_graph_id);
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("  tiling_data_list.insert(tiling_data_list.end(), tiling_data_list_tmp" +
                        std::to_string(group_index) + ".begin(), tiling_data_list_tmp" + std::to_string(group_index) +
                        ".end());");
   tiling_func_.AddLine("  return true;");
   tiling_func_.AddLine("}");
 }

void TilingCodeGenImpl::GenPGOUpdateTilingInfo(const size_t asc_graph_id, const size_t impl_graph_id) {
  GenUpdateWorkspace(asc_graph_id, impl_graph_id);
  if (enable_group_parallels_[asc_graph_id][impl_graph_id]) {
    tiling_func_.AddLine("      ArrangeBlockOffsetsAscGraph" + std::to_string(asc_graph_id) + "Result" +
                         std::to_string(impl_graph_id) + "(tiling_data, tiling_data.get_block_dim());");
  }
}

ge::Status TilingCodeGenImpl::GenPGOGetScheduleResultPerGroup(const size_t asc_graph_id, const size_t impl_graph_id,
                                             const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                             const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                                             const std::map<std::string, std::set<std::string>> &hardware_map) {
  tiling_func_.AddLine("    bool has_solution = true;");
  tiling_func_.AddLine("    for (auto &tiling_data_perf : tiling_data_list_tmp) {");
  tiling_func_.AddLine("      auto &tiling_data = tiling_data_perf.tiling_data;");
  tiling_func_.AddLine("      std::unordered_map<int64_t, uint64_t> workspace_map;");
  tiling_func_.AddLine("      workspace_map.reserve(workspace_map_filter_use.size());");
  tiling_func_.AddLine("      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());");
  auto current_group_iter = graph_info.find(group_info.first);
  GE_ASSERT_TRUE(current_group_iter != graph_info.end(), "Current graph id not found in graph info.");
  for (auto group_iter = std::next(current_group_iter); group_iter != graph_info.end(); ++group_iter) {
    const auto &hardware_iter = hardware_map.find(group_iter->second.second);
    if (hardware_iter != hardware_map.cend()) {
      GenSetHardwareCodes(std::string("    tiling_data.") + group_iter->second.second, hardware_iter->second);
    } else {
      GELOGW("Hardware info not found for group %s.", group_iter->second.second.c_str());
    }
    auto [input_vars_set_code, need_update_second_group_input_vars] =
        ProcessVarRelationsStatement(graph_info, var_relations_[asc_graph_id][impl_graph_id], group_iter->first, "      tiling_data.");
    if (need_update_second_group_input_vars) {
      tiling_func_.AddLine(input_vars_set_code);
    }
    tiling_func_.AddLine("      has_solution = " + group_iter->second.first + "::GetTiling(tiling_data." + group_iter->second.second + "_tiling_data, workspace_map, -1);");
    tiling_func_.AddLine("      if (!has_solution) {");
    tiling_func_.AddLine("        OP_LOGI(OP_NAME, \"No solution for " + group_info.second.second + " at " + group_iter->second.second + "\");");
    tiling_func_.AddLine("        continue;");
    tiling_func_.AddLine("      }");
  }
  GenPGOUpdateTilingInfo(asc_graph_id, impl_graph_id);
  tiling_func_.AddLine("      auto workspaceSizeTmp = GetWorkspaceSize(tiling_data);");
  tiling_func_.AddLine("      if (workspaceSizeTmp > workspaceSize) {");
  tiling_func_.AddLine("        workspaceSize = workspaceSizeTmp;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    workspaceSize += 16 * 1024 * 1024;");
  tiling_func_.AddLine("    PgoConfig::Instance().batch_callback(" + GenLaunchLikeInputOutputDef(false) + "stream, workspaceSize, &tiling_data_list_tmp);");
  tiling_func_.AddLine("    for (auto &tiling_data_perf : tiling_data_list_tmp) {");
  tiling_func_.AddLine("      tiling_data_list.push_back(tiling_data_perf);");
  tiling_func_.AddLine("      if (tiling_data_perf.best_perf < best_perf) {");
  tiling_func_.AddLine("        tiling_data = tiling_data_perf.tiling_data;");
  tiling_func_.AddLine("        best_perf = tiling_data_perf.best_perf;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("    }");
  return ge::SUCCESS;
}

ge::Status TilingCodeGenImpl::GenPGOGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                                             const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                             const std::map<std::string, std::set<std::string>> &hardware_map) {
  std::string check_cond;
  std::string cal_perf;
  std::vector<std::string> block_num;
  std::string func_define(kInlineStr);
  func_define.append("bool GetScheduleResult")
      .append(std::to_string(impl_graph_id) + "PGO")
      .append("(std::vector<AutofuseTilingDataPerf>& tiling_data_list, const uint32_t ori_block_dim, const int32_t tiling_case_id,")
      .append(config_.tiling_data_type_name)
      .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim,")
      .append(GenLaunchLikeInputOutputDef())
      .append("void* stream, uint32_t workspaceSize, std::vector<uint32_t*> multi_group_block_dim_list = {}) {");
  tiling_func_.AddLine(func_define);
  uint32_t group_index = 0U;
  tiling_func_.AddLine("  std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp{};");
  tiling_func_.AddLine("  workspaceSize = 0;");
  tiling_func_.AddLine("  std::unordered_map<int64_t, uint64_t> workspace_map_filter_use{};");
  std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
  tiling_func_.AddLine("  tiling_data.set_" + tiling_key_prefix + "tiling_key(" + std::to_string(impl_graph_id) + ");");
  for (const auto &group_info : graph_info) {
    std::string tiling_item_name = group_info.second.second + "_tiling_data";
    tiling_func_.AddLine("  auto &" + tiling_item_name + " = tiling_data." + tiling_item_name + ";");
    const auto &hardware_iter = hardware_map.find(group_info.second.second);
    if (hardware_iter == hardware_map.cend()) {
      continue;
    }
    GenSetHardwareCodes(group_info.second.second, hardware_iter->second);
    std::string result_name = "result" + std::to_string(group_index);
    tiling_func_.AddLine("  auto " + result_name + " = " +
                         GenPGOScheduleGroupDoTiling(group_info.second.second, group_info.second.first,
                         GenLaunchLikeInputOutputDef(false)) + ";");
    tiling_func_.AddLine("  if (" + result_name + ") {");
    GE_ASSERT_SUCCESS(GenPGOGetScheduleResultPerGroup(asc_graph_id, impl_graph_id, graph_info, group_info, hardware_map));
    tiling_func_.AddLine("  }");
    group_index++;
  }
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
  return ge::SUCCESS;
}

 void TilingCodeGenImpl::GenGetScoreFuncs(const size_t asc_graph_id,
     const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map) {
   auto &schedule_results_score_func = score_funcs_[kModelInfoLevel::K_SCHEDULE_RESULT_LEVEL][asc_graph_id];
   for (size_t i = 0UL; i < namespace_map.size(); i++) {
     tiling_func_.AddLine("namespace " + GetScheduleResultPrefix(asc_graph_id, i) + " {");
     auto &score_func = schedule_results_score_func[i];
     if (score_func.empty()) {
       score_func = "int32_t CalcScore(" + config_.tiling_data_type_name + " &tiling_data) { return 0;}";
     }
     tiling_func_.AddLine(schedule_results_score_func[i]);
     tiling_func_.AddLine("}");
   }
 }
 
 void TilingCodeGenImpl::GenGetScoreFuncsCalling(const size_t asc_graph_id,
     const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map) {
   tiling_func_.AddLine("  int32_t scores[" + std::to_string(namespace_map.size()) + "]{};");
   for (size_t i = 0UL; i < namespace_map.size(); i++) {
     tiling_func_.AddLine("  scores[" + std::to_string(i) + "] = " + GetScheduleResultPrefix(asc_graph_id, i) +
                          "::CalcScore(tiling_data);");
   }
 }
 
 void TilingCodeGenImpl::GenGetMaxScoreIndex(const AscGraphNamepspaceMap &namespace_map) {
   tiling_func_.AddLine("  int32_t max_index = 0L;");
   if (namespace_map.size() > 1) {
     tiling_func_.AddLine("   for (int32_t index = 1; index < " + std::to_string(namespace_map.size()) + "; index++) {");
     tiling_func_.AddLine("    if (scores[index] > scores[max_index]) {");
     tiling_func_.AddLine("      max_index = index;");
     tiling_func_.AddLine("    }");
     tiling_func_.AddLine("  }");
   }
 }
 
 void TilingCodeGenImpl::GenScheduleResultGetTilingCalling(const std::string &index, const std::string &ident) {
   tiling_func_.AddLine(ident + "  if (kScheduleResultFunctions[" + index +
                        "](ori_block_dim, tiling_option->tiling_case_id, tiling_data, cur_perf, best_perf, "
                        "cur_block_dim)) {");
   tiling_func_.AddLine(ident + "    auto res = GetResultSummary(best_perf, tiling_data);");
   GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, ident + "    ");
   GenDurationPrintCode(ident + "    ");
   GenDurationClearCode(ident + "    ");
   tiling_func_.AddLine(ident + "    return res;");
   tiling_func_.AddLine(ident + "  }");
 }

 ge::Status TilingCodeGenImpl::GenGetAllSchedulesResults(const AscGraphNamepspaceMap &namespace_map) {
   std::string chosen_index = (config_.force_schedule_result < 0) ? "max_index" : std::to_string(config_.force_schedule_result);
   if (NeedGenScoreFunc(score_funcs_)) {
     GenScheduleResultGetTilingCalling(chosen_index);
   }
   if (config_.force_schedule_result >= 0) {
     GELOGI("Force schedule result %ld for op %s", config_.force_schedule_result,
            config_.tiling_data_type_name.c_str());
     GE_ASSERT_TRUE(config_.force_schedule_result < static_cast<int32_t>(namespace_map.size()), "Force schedule "
                    "result[%ld] should less than result size[%zu]", config_.force_schedule_result,
                    namespace_map.size());
     tiling_func_.AddLine("  auto got_result = kScheduleResultFunctions[" + chosen_index +
       "](ori_block_dim, tiling_option->tiling_case_id, tiling_data, cur_perf, "
                          "best_perf, cur_block_dim);");
     tiling_func_.AddLine("  if (!got_result) {");
     tiling_func_.AddLine("    OP_LOGW(OP_NAME, \"Schedule result" + std::to_string(config_.force_schedule_result) +
                          " can not found for op\");");
     tiling_func_.AddLine("    return false;");
     tiling_func_.AddLine("  }");
     return ge::SUCCESS;
   }
   tiling_func_.AddLine("  for (int32_t index = 0; index < " + std::to_string(namespace_map.size()) + "; index++) {");
   if (NeedGenScoreFunc(score_funcs_)) {
     tiling_func_.AddLine("    if (max_index == index) {");
     tiling_func_.AddLine("      continue;");
     tiling_func_.AddLine("    }");
   }
   tiling_func_.AddLine(
       "    (void)kScheduleResultFunctions[index](ori_block_dim, tiling_option->tiling_case_id, tiling_data, cur_perf, "
       "best_perf, cur_block_dim);");
   tiling_func_.AddLine("  }");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenEnableGroupParallelFunctions(const FusedGraphNamespaceMap &namespace_map) {
   size_t asc_graph_id = 0UL;
   for (const auto &asc_graph_namespace_map : namespace_map) {
     std::stringstream ss;
     for (const auto &result_id_and_groups : asc_graph_namespace_map.second) {
       const auto &groups = result_id_and_groups.second;
       if (enable_group_parallels_[asc_graph_id][result_id_and_groups.first]) {
         // 先简单按顺序遍历，非最优
         ss << "void ArrangeBlockOffsetsAscGraph" << asc_graph_id << "Result" << result_id_and_groups.first
            << "(AutofuseTilingData &t, uint32_t aiv_num) {" << std::endl;
         ss << "  uint32_t block_offset = 0U;" << std::endl;
         ss << "  uint32_t block_dim = 0U;" << std::endl;
         ss << "  uint32_t max_block_dim = aiv_num;" << std::endl;
         ss << "  uint32_t actual_max_block_dim = t.get_block_dim();" << std::endl;
         for (const auto &group_id_and_names : groups) {
           const auto group_id = group_id_and_names.first;
           const auto &sub_tiling_data = "t." + group_id_and_names.second.second + "_tiling_data";
           const auto var_name = std::string("sub_tiling_data_") + std::to_string(group_id);
           ss << "  block_dim = " << sub_tiling_data << ".get_block_dim();" << std::endl;
           ss << "  " << sub_tiling_data << ".set_ub_size(block_offset); // reuse ub_size as block_offset" << std::endl;
           ss << "  block_offset += block_dim;" << std::endl;
           // block不够则回绕
           ss << "  if (block_offset > max_block_dim) {" << std::endl;
           ss << "    block_offset = block_offset - max_block_dim;" << std::endl;
           ss << "    actual_max_block_dim = max_block_dim;" << std::endl;
           ss << "  }" << std::endl;
           ss << "  actual_max_block_dim = std::max(actual_max_block_dim, block_offset);" << std::endl;
         }
         ss << "  t.set_block_dim(actual_max_block_dim);" << std::endl;
         ss << "}" << std::endl;
       }
     }
     tiling_func_.AddLine(ss.str());
     asc_graph_id++;
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenEnableGroupParallelInvoke(size_t asc_graph_id,
                                                            const AscGraphNamepspaceMap &asc_graph_namespace_map) {
   for (const auto &result_id_and_groups : asc_graph_namespace_map) {
     const auto result_id = result_id_and_groups.first;
     if (enable_group_parallels_[asc_graph_id][result_id]) {
       std::stringstream ss;
       ss << "  if (tiling_data.get_graph" << asc_graph_id << "_tiling_key() == " << result_id << ") {"
          << std::endl;
       ss << "    ArrangeBlockOffsetsAscGraph" << asc_graph_id << "Result" << result_id
          << "(tiling_data, org_block_dim);" << std::endl;
       ss << "  }" << std::endl;
       tiling_func_.AddLine(ss.str());
     }
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map) {
   tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, TilingOption *option) {");
   tiling_head_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, TilingOption *option);");
   size_t asc_graph_id = 0UL;
   for (const auto &asc_graph_namespace_map : namespace_map) {
     if (asc_graph_id == 0UL) {
       tiling_func_.AddLine("  uint32_t max_block_dim = 0U;");
       tiling_func_.AddLine("  uint32_t org_block_dim = tiling_data.get_block_dim();");
     }
     const std::string &asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
     tiling_func_.AddLine("  if (!" + asc_graph_namespace + "::GetTiling(tiling_data, option)) {");
     tiling_func_.AddLine("    OP_LOGE(OP_NAME, \"Failed to get tiling of " + asc_graph_namespace + ".\");");
     tiling_func_.AddLine("    return false;");
     tiling_func_.AddLine("  }");
     GenEnableGroupParallelInvoke(asc_graph_id, asc_graph_namespace_map.second);
     tiling_func_.AddLine(
         "  max_block_dim = (tiling_data.get_block_dim() > max_block_dim) ? tiling_data.get_block_dim() : "
         "max_block_dim;");
     asc_graph_id++;
    }
    tiling_func_.AddLine("  tiling_data.set_block_dim(max_block_dim);");
    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End GetTiling.\");");
    tiling_func_.AddLine("  return true;");
    tiling_func_.AddLine("}");
    return ge::SUCCESS;
  }

  ge::Status TilingCodeGenImpl::GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map) {
    tiling_func_.AddLine("bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* tiling_data, uint32_t max_block_dim) {");
    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"Start PGOSearchTilingKey root.\");");
    tiling_func_.AddLine("  bool ret = true;");
    tiling_func_.AddLine("  for (uint32_t block_dim_i=1; block_dim_i <= max_block_dim; block_dim_i++) {");
    for (const auto &asc_graph_namespace_map : namespace_map) {
      tiling_func_.AddLine("    std::vector<AutofuseTilingData> vec" + std::to_string(asc_graph_namespace_map.first) +";");
    }
    auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
    tiling_func_.AddLine( "    tiling_data->set_"+core_num + "(block_dim_i);");
    size_t asc_graph_id = 0UL;
    for (const auto &asc_graph_namespace_map : namespace_map) {
      const std::string &asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
      tiling_func_.AddLine("    if (!" + asc_graph_namespace +
                           "::PGOByCoreNumSearchTilingKey(vec"+ std::to_string(asc_graph_namespace_map.first) +", *tiling_data)) {");
      tiling_func_.AddLine("      OP_LOGE(OP_NAME, \"Failed to get tiling of " + asc_graph_namespace + ".\");");
      tiling_func_.AddLine("      continue;");
      tiling_func_.AddLine("    }");
      asc_graph_id++;
    }
    GenPGOByCoreNumSearchTilingKeyCollectTilingData(namespace_map);
    tiling_func_.AddLine("  }");

    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey root.\");");

    tiling_func_.AddLine("  return ret;");
    tiling_func_.AddLine("}");
    return ge::SUCCESS;
  }

  ge::Status TilingCodeGenImpl::GenPGOFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map) {
    tiling_func_.AddLine("bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " +
                         config_.tiling_data_type_name + " &tiling_data, " +
                         " int32_t tilingCaseId, AutofuseTilingData* tilingData," + GenLaunchLikeInputOutputDef() +
                         "void* stream, uint32_t workspaceSize, double& best_perf) {");
    size_t asc_graph_id = 0UL;
    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"Start PGOSearchTilingKey root.\");");
    tiling_func_.AddLine("  double cur_perf = DBL_MAX;");
    tiling_func_.AddLine("  uint32_t cur_block_dim = 1;");
    tiling_func_.AddLine("  uint32_t ori_block_dim = tiling_data.get_block_dim();");
    tiling_func_.AddLine("  AutofuseTilingData tilingTmp;");
    tiling_func_.AddLine("  tilingTmp = tiling_data;");
    tiling_func_.AddLine("  uint32_t max_block_dim = 0U;");
    tiling_func_.AddLine("  std::vector<uint32_t*> multi_group_block_dim_list;");
    for (const auto &asc_graph_namespace_map : namespace_map) {
      for (auto &asc_graph_map_iter : asc_graph_namespace_map.second) {
        auto &asc_graph_map = asc_graph_map_iter.second;
        for (const auto &graph_info : asc_graph_map) {
          auto &graph_info_map = graph_info.second;
          std::string tiling_item_name = graph_info_map.second + "_tiling_data";
          tiling_func_.AddLine("  multi_group_block_dim_list.push_back(tilingTmp." + tiling_item_name +
                               ".get_addr_block_dim());");
        }
      }
      asc_graph_id++;
    }

    asc_graph_id = 0UL;
    for (const auto &asc_graph_namespace_map : namespace_map) {
      const std::string &asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
      tiling_func_.AddLine("  if (!" + asc_graph_namespace +
                           "::PGOSearchTilingKey(tiling_data_list, tilingTmp, tilingCaseId, &tilingTmp, " +
                           GenLaunchLikeInputOutputDef(false) + "stream, workspaceSize, cur_perf, multi_group_block_dim_list)) {");
      tiling_func_.AddLine("    OP_LOGE(OP_NAME, \"Failed to get tiling of " + asc_graph_namespace + ".\");");
      tiling_func_.AddLine("    return false;");
      tiling_func_.AddLine("  }");
      tiling_func_.AddLine("  if (best_perf > cur_perf) {");
      tiling_func_.AddLine("    tiling_data = tilingTmp;");
      tiling_func_.AddLine("    best_perf = cur_perf;");
      tiling_func_.AddLine("  }");
      asc_graph_id++;
    }

    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey root.\");");
    tiling_func_.AddLine("  return true;");
    tiling_func_.AddLine("}");
    return ge::SUCCESS;
  }

  void TilingCodeGenImpl::GenPGOByCoreNumGetAllSchedulesResults(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map) {
    std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
    tiling_func_.AddLine("  for (int32_t index = 0; index < " + std::to_string(namespace_map.size()) + "; index++) {");
    tiling_func_.AddLine("    tiling_data.set_" + tiling_key_prefix + "tiling_key(index);");
    for (const auto &result_id_and_groups : namespace_map) {
      for (const auto &group_info : result_id_and_groups.second) {
        tiling_func_.AddLine("    tiling_data." + group_info.second.second + "_tiling_data = {};");
      }
    }
    tiling_func_.AddLine("    TilingOption option;");
    tiling_func_.AddLine("    option.tiling_case_id = index;");
    tiling_func_.AddLine("    (void)kScheduleResultFunctionsPGOByCoreNum[index](tiling_data_list, tiling_data);");
    tiling_func_.AddLine("  }");
  } 

  void TilingCodeGenImpl::GenPGOGetAllSchedulesResults(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map) {
    std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";

    tiling_func_.AddLine("  AutofuseTilingData tilingTmp;");
    tiling_func_.AddLine("  for (int32_t index = 0; index < " + std::to_string(namespace_map.size()) + "; index++) {");
    tiling_func_.AddLine("    tilingTmp = tiling_data;");
    tiling_func_.AddLine("    tilingTmp.set_" + tiling_key_prefix + "tiling_key(index);");
    tiling_func_.AddLine("    TilingOption option;");
    tiling_func_.AddLine("    option.tiling_case_id = index;");
    tiling_func_.AddLine("    AscGraph" + std::to_string(asc_graph_id) + "::GetTiling(tilingTmp, &option);");
    tiling_func_.AddLine("    (void)kScheduleResultFunctionsPGO[index](tiling_data_list, ori_block_dim, tilingCaseId, tilingTmp, cur_perf, "
                         "best_perf, cur_block_dim, " +
                         GenLaunchLikeInputOutputDef(false) + "stream, workspaceSize, block_dim_vec);");
    tiling_func_.AddLine("    workspaceSize = GetWorkspaceSize(*tilingData);");
    if (!config_.is_inductor_scene) {
      tiling_func_.AddLine("    workspaceSize += 16 * 1024 * 1024;");
    }
    tiling_func_.AddLine("    PgoConfig::Instance().single_callback(" + GenLaunchLikeInputOutputDef(false) +
                         "stream, workspaceSize, tilingData, &cur_perf);");
    tiling_func_.AddLine("    AutofuseTilingDataPerf tiling_perf;");
    tiling_func_.AddLine("    tiling_perf.tiling_data = *tilingData;");
    tiling_func_.AddLine("    tiling_perf.best_perf = cur_perf;");
    tiling_func_.AddLine("    tiling_data_list.push_back(tiling_perf);");
    tiling_func_.AddLine("    if (best_perf > cur_perf) {");
    tiling_func_.AddLine("      *tilingData = tilingTmp;");
    tiling_func_.AddLine("      best_perf = cur_perf;");
    tiling_func_.AddLine("    }");
    tiling_func_.AddLine("  }");
 }

 ge::Status TilingCodeGenImpl::GenGetTilingForAllSchedulesResults(const uint32_t asc_graph_id,
                                                                  const AscGraphNamepspaceMap &asc_graph_map) {
   tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                        "TilingOption *tiling_option) {");
   tiling_head_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                        "TilingOption *tiling_option);");
   GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                     "Generate begin code!");
   GE_ASSERT_SUCCESS(GenGetTilingForAllInitLines());
   if (NeedGenScoreFunc(score_funcs_)) {
     GenGetScoreFuncsCalling(asc_graph_id, asc_graph_map);
     GenGetMaxScoreIndex(asc_graph_map);
   }
   GE_ASSERT_SUCCESS(GenGetAllSchedulesResults(asc_graph_map));
   tiling_func_.AddLine("  GetResultSummary(best_perf, tiling_data);");
   GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                     "Generate end code!");
   GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
   GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
   tiling_func_.AddLine("  return true;");
   tiling_func_.AddLine("}");
   tiling_func_.AddLine("} // namespace AscGraph" + std::to_string(asc_graph_id) + " {");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenGetTilingForScheduleResult() {
   std::map<std::string, std::set<std::string>> hardware_map;
   FusedGraphNamespaceMap namespace_map;
   GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
   for (auto &asc_graph_map_iter : namespace_map) {
     auto &asc_graph_map = asc_graph_map_iter.second;
     size_t asc_graph_id = asc_graph_map_iter.first;
     tiling_func_.AddLine("namespace AscGraph" + std::to_string(asc_graph_id) + " {");
     if (NeedGenScoreFunc(score_funcs_)) {
       GenGetScoreFuncs(asc_graph_id, asc_graph_map);
     }
     GE_ASSERT_SUCCESS(GenGetResultSummary(asc_graph_id),
                       "Gen GetResultSummary failed, asc_graph_id = %zu, tiling data name = %s.", asc_graph_id,
                       config_.tiling_data_type_name.c_str());
     const bool enable_groups_parallel = GenUpdateCurPerfAndBlockByGroupIfNeeded(asc_graph_id, asc_graph_map);
     if (enable_groups_parallel) {
       tiling_func_.AddLine(GenUpdateCurPerfAndBlockByGroup());
     }
     for (const auto &graph_info : asc_graph_map) {
       GenGetScheduleResult(asc_graph_id, graph_info.first, graph_info.second, hardware_map);
     }
     tiling_func_.AddLine(GenScheduleResultFuncTypeDefine(config_.tiling_data_type_name));
     tiling_func_.AddLine(GenScheduleResultFuncsDefine(asc_graph_map));
     GE_ASSERT_SUCCESS(GenGetTilingForAllSchedulesResults(asc_graph_id, asc_graph_map),
                       "Generate GetTiling for all schedules results failed, asc_graph_id = %zu.", asc_graph_id);
   }
   GE_ASSERT_SUCCESS(GenEnableGroupParallelFunctions(namespace_map));
   GE_ASSERT_SUCCESS(GenFusedScheduleResultsGetTilingDefine(namespace_map));
   GE_ASSERT_SUCCESS(GenGetTilingWithCaseId(true));
   GE_ASSERT_SUCCESS(GenGetTilingOptionRange());
   GE_ASSERT_SUCCESS(GenIsStaticShape());
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenPGOGetTilingForAll() {
   std::map<std::string, std::set<std::string>> hardware_map;
   FusedGraphNamespaceMap namespace_map;
   GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
   for (auto &asc_graph_map_iter : namespace_map) {
     auto &asc_graph_map = asc_graph_map_iter.second;
     size_t asc_graph_id = asc_graph_map_iter.first;
     tiling_func_.AddLine("namespace AscGraph" + std::to_string(asc_graph_id) + " {");
     for (const auto &graph_info : asc_graph_map) {
       GE_ASSERT_SUCCESS(GenPGOGetScheduleResult(asc_graph_id, graph_info.first, graph_info.second, hardware_map));
     }
     tiling_func_.AddLine(
         GenPGOScheduleResultFuncTypeDefine(config_.tiling_data_type_name, GenLaunchLikeInputOutputDef()));
     tiling_func_.AddLine(GenScheduleResultFuncsDefine(asc_graph_map, "PGO"));
     tiling_func_.AddLine("bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + config_.tiling_data_type_name + " &tiling_data, " +
                          " int32_t tilingCaseId, AutofuseTilingData* tilingData," + GenLaunchLikeInputOutputDef() +
                          "void* stream, uint32_t workspaceSize, double& best_perf, std::vector<uint32_t*> block_dim_vec={}) {");
     GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                       "Generate begin code!");
     GE_ASSERT_SUCCESS(GenGetTilingForAllInitLines(true));
     GenPGOGetAllSchedulesResults(asc_graph_id, asc_graph_map);
     tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey in AscGraph.\");");
     GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                       "Generate end code!");
     GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
     GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
     tiling_func_.AddLine("  return true;");
     tiling_func_.AddLine("}");
     tiling_func_.AddLine("} // namespace AscGraph" + std::to_string(asc_graph_id) + " {");
   }
   GE_ASSERT_SUCCESS(GenPGOFusedScheduleResultsGetTilingDefine(namespace_map));
   return ge::SUCCESS;
 }

  ge::Status TilingCodeGenImpl::GenPGOByCoreNumTilingForAll() {
    std::map<std::string, std::set<std::string>> hardware_map;
    FusedGraphNamespaceMap namespace_map;
    GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
    for (auto &asc_graph_map_iter : namespace_map) {
      auto &asc_graph_map = asc_graph_map_iter.second;
      size_t asc_graph_id = asc_graph_map_iter.first;
      tiling_func_.AddLine("namespace AscGraph" + std::to_string(asc_graph_id) + " {");
      for (const auto &graph_info : asc_graph_map) {
        GenPGOByCoreNumGetScheduleResult(asc_graph_id, graph_info.first, graph_info.second, hardware_map, var_relations_[asc_graph_id][graph_info.first]);
      }
      tiling_func_.AddLine(GenPGOByCoreNumScheduleResultFuncTypeDefine());
      tiling_func_.AddLine(GenScheduleResultFuncsDefine(asc_graph_map, "PGOByCoreNum"));
      tiling_func_.AddLine("bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData tiling_data) {");
      GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "), "Generate begin code!");

      GenPGOByCoreNumGetAllSchedulesResults(asc_graph_id, asc_graph_map);

      tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey in AscGraph.\");");
      GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "), "Generate end code!");
      GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
      GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
      tiling_func_.AddLine("  return true;");
      tiling_func_.AddLine("}");
      tiling_func_.AddLine("} // namespace AscGraph" + std::to_string(asc_graph_id));
    }
    GE_ASSERT_SUCCESS(GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(namespace_map));
    return ge::SUCCESS;
  }

 ge::Status TilingCodeGenImpl::GenGetTilingOptionRange() {
   TilingOptionCodeGenerator generator;
   TilingOptionRangeData case_id_option_data{
       kTilingOptionType::kTilingCaseId,
       kTilingOptionRangeType::kEnumRange,
   };
   if (is_uniq_group_) {
     for (const auto &model_info : tiling_model_info_) {
       case_id_option_data.range_vals.emplace_back(model_info.tiling_case_id);
     }
   } else {
     std::unordered_set<size_t> schedule_results_ids;
     for (const auto &model_info : tiling_model_info_) {
       schedule_results_ids.insert(model_info.schedule_group_ident.impl_graph_id);
     }
     for (const auto &result_id : schedule_results_ids) {
       case_id_option_data.range_vals.emplace_back(result_id);
     }
   }
   GE_ASSERT_TRUE(case_id_option_data.range_vals.size() <= kMaxTilingOptionEnumNum, "schedule result %zu is over %d",
                  case_id_option_data.range_vals.size(), kMaxTilingOptionEnumNum);
   generator.AddTilingOptionRange(std::move(std::make_unique<TilingOptionRange>(case_id_option_data)));
   GE_ASSERT_SUCCESS(generator.GenFunctionDefine());
   tiling_func_.AddLine(generator.GetOutputStr());
   return ge::SUCCESS;
 }

ge::Status TilingCodeGenImpl::GenGetTilingWithCaseId(bool is_tail) {
  bool use_cache = (!is_tail && with_reuse_info_);
  bool use_workspace = !(is_uniq_group_ || is_tail);
  int32_t min_tiling_case_size = INT32_MAX;
  std::map<string, int32_t> group_tiling_case_ids;
  for (auto &model : tiling_model_info_) {
    group_tiling_case_ids[model.schedule_group_ident.GetItemPrefix()]++;
  }
  for (auto &group_tiling_case : group_tiling_case_ids) {
    min_tiling_case_size = std::min(group_tiling_case.second, min_tiling_case_size);
  }
  GE_ASSERT_SUCCESS(ValidateForceTilingCase(group_tiling_case_ids, min_tiling_case_size));

  std::string tiling_case = (config_.force_tiling_case.is_single_mode && config_.force_tiling_case.single_case < 0)
                              ? "tilingCaseId"
                              : std::to_string(config_.force_tiling_case.single_case);
  const ge::char_t *cache_define_head = use_cache ? (", GroupLevelCache *cache = nullptr") : "";
  const ge::char_t *cache_define_func = use_cache ? (", GroupLevelCache *cache") : "";
  const ge::char_t *cache_used = use_cache ? (", cache") : "";
  const ge::char_t *workspace_define = use_workspace
                                         ? (", std::unordered_map<int64_t, uint64_t> &workspace_map")
                                         : "";
  const ge::char_t *workspace_used = use_workspace ? (", workspace_map") : "";
  GE_ASSERT_TRUE(!tiling_model_info_.empty());
  tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                       " &tiling_data" + workspace_define + ", int32_t tilingCaseId" + cache_define_func + ") {");
  tiling_head_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                       " &tiling_data" + workspace_define + ", int32_t tilingCaseId" + cache_define_head + ");");
  bool need_operator_cache = is_tail || (!is_tail && is_uniq_group_);
  // 第一级：算子级缓存查询（在GetTilingKey之前）
   if (need_operator_cache) {
     GE_ASSERT_SUCCESS(
         cache::OperatorLevelCacheGen::GenInitAndQueryCacheCode(tiling_func_, tiling_model_info_, config_),
         "Generate init and query cache code failed.");
   }
  tiling_func_.AddLine("  tiling_option_default.tiling_case_id = " + tiling_case + ";");
  tiling_func_.AddLine(
      std::string("  auto ret = GetTiling(tiling_data") + workspace_used + ", &tiling_option_default" + cache_used +
      ");");
  if (need_operator_cache) {
    GE_ASSERT_SUCCESS(cache::OperatorLevelCacheGen::GenSaveCacheCalls(tiling_func_, tiling_model_info_, config_),
                      "Generate save cache calls failed.");
  }
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  return ge::SUCCESS;
}
 
 ge::Status TilingCodeGenImpl::GenGetTilingWithOption() {
   const ge::char_t *cache_define_func = with_reuse_info_ ? (", GroupLevelCache* cache") : "";
   tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                        " &tiling_data, " + (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                        "TilingOption *tiling_option" + cache_define_func + ") {");
   tiling_head_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                        " &tiling_data, " + (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                        "TilingOption *tiling_option" + cache_define_func + ");");
   if (is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                       "Generate duration begin code failed.");
   }
   GE_ASSERT_SUCCESS(GenUsedTilingOption());
   GE_ASSERT_SUCCESS(GenOpLog(
       "  ", "Start GetTiling.",
       "Start tiling for sched group " + tiling_model_info_[0].schedule_group_ident.GetGroupPrefix() + "."));
   const ge::char_t *cache_used = with_reuse_info_ ? (", cache") : "";
   int32_t force_case_id = config_.force_tiling_case.GetCase(tiling_model_info_[0].schedule_group_ident.group_id).first;
   std::string tiling_case = (force_case_id < 0) ? "tiling_option_used->tiling_case_id" : std::to_string(force_case_id);
   tiling_func_.AddLine(std::string("  if (!GetTilingKey(tiling_data, ") +
                        (is_uniq_group_ ? "" : "workspace_map, ") + tiling_case + cache_used + ")) {");
   GE_ASSERT_SUCCESS(GenOpLog("    ", "GetTiling Failed."));
   tiling_func_.AddLine("    return false;");
   tiling_func_.AddLine("  }");
   GE_ASSERT_SUCCESS(GenOpLog("  ", "End GetTiling.", "End tiling for sched group."));
   if (is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                       "Generate duration end code failed.");
     GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
     GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
   }
   tiling_func_.AddLine("  return true;");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::ValidateForceTilingCase(
     const std::map<string, int32_t> &group_tiling_case_ids,
     int32_t min_tiling_case_size) const {
   // 如果配置了 force_schedule_result，但当前 tiling_model_info_ 不包含该 result，
   // 说明当前是 GenTilingBody 在处理单个 result，跳过该校验（留给 GenTilingTail 处理）
   if (config_.force_schedule_result >= 0) {
     bool has_force_result = false;
     for (const auto &model : tiling_model_info_) {
       if (static_cast<int64_t>(model.schedule_group_ident.impl_graph_id) == config_.force_schedule_result) {
         has_force_result = true;
         break;
       }
     }
     if (!has_force_result) {
       GELOGD("Skip force tiling case validation: current model info does not contain result[%ld]",
              config_.force_schedule_result);
       return ge::SUCCESS;
     }
   }

   GE_ASSERT_SUCCESS(ValidateSingleModeForceTilingCase(min_tiling_case_size));
   if (!config_.force_tiling_case.is_single_mode) {
     GE_ASSERT_SUCCESS(ValidateGroupModeForceTilingCase(group_tiling_case_ids));
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::ValidateSingleModeForceTilingCase(int32_t min_tiling_case_size) const {
   int32_t force_case = config_.force_tiling_case.single_case;

   // 如果 force_case 为默认值 -1，跳过校验
   if (force_case == -1) {
     GELOGD("Force tiling case is not set, skip validation");
     return ge::SUCCESS;
   }

   // 如果 force_case < min_tiling_case_size，按 case_id 校验
   if (force_case < min_tiling_case_size) {
     GE_ASSERT_TRUE(force_case >= 0,
                    "Force tiling case[%d] should be non-negative", force_case);
     return ge::SUCCESS;
   }

   // 否则，检查是否为有效的 tiling_key
   bool tiling_key_found = false;
   for (const auto &model : tiling_model_info_) {
     if (static_cast<int32_t>(model.tiling_case_id) == force_case) {
       tiling_key_found = true;
       break;
     }
   }

   GE_ASSERT_TRUE(tiling_key_found,
                  "Force tiling case[%d] not found as tiling_key in model info", force_case);
   return ge::SUCCESS;
 }

ge::Status TilingCodeGenImpl::ValidateGroupModeForceTilingCase(
    const std::map<string, int32_t> &group_tiling_case_ids) const {
  // 遍历当前传入的 tiling_model_info_，只校验这些 group
  // 因为 GenTilingBody 每次只传入单个 group，而不是所有 groups
  for (const auto &model : tiling_model_info_) {
    size_t cur_group_id = model.schedule_group_ident.group_id;
    int64_t cur_result_id = static_cast<int64_t>(model.schedule_group_ident.impl_graph_id);

    // 如果配置了 force_schedule_result，跳过非指定 result 的 group
    if (config_.force_schedule_result >= 0 && cur_result_id != config_.force_schedule_result) {
      continue;
    }

    // 查找该 group 是否有强制指定的 case
    auto it = config_.force_tiling_case.group_cases.find(cur_group_id);
    if (it == config_.force_tiling_case.group_cases.end()) {
      continue;  // 该 group 没有强制指定，跳过
    }

    int32_t force_case_id = it->second.first;

    // 获取该 group 的 case 数量
    auto case_it = group_tiling_case_ids.find(model.schedule_group_ident.GetItemPrefix());
    GE_ASSERT_TRUE(case_it != group_tiling_case_ids.end(), "Group[%zu] in result[%ld] not found in "
                                                          "group_tiling_case_ids", cur_group_id, cur_result_id);
    size_t group_case_size = case_it->second;

    GELOGD("Validate force tiling case: result[%ld] group[%zu] case[%d] < size[%zu]",
           cur_result_id, cur_group_id, force_case_id, group_case_size);

    // 校验 case_id 或 tiling_key
    if (force_case_id < static_cast<int32_t>(group_case_size)) {
      // case_id 在范围内，校验通过
      GELOGD("Validate force tiling case by case_id: result[%ld] group[%zu] case[%d]",
             cur_result_id, cur_group_id, force_case_id);
    } else {
      // 检查是否为有效的 tiling_key
      bool tiling_key_found = false;
      for (const auto &m : tiling_model_info_) {
        if (m.schedule_group_ident.group_id == cur_group_id &&
            static_cast<int64_t>(m.schedule_group_ident.impl_graph_id) == cur_result_id &&
            static_cast<int32_t>(m.tiling_case_id) == force_case_id) {
          tiling_key_found = true;
          break;
        }
      }
      GE_ASSERT_TRUE(tiling_key_found,
                     "Force tiling case[%d] for group[%zu] result[%ld] not found as tiling_key",
                     force_case_id, cur_group_id, cur_result_id);
    }
  }
  return ge::SUCCESS;
}

 ge::Status TilingCodeGenImpl::GenScheduleGroupTilingTail() {
   if (config_.gen_tiling_data) {
     if (!is_uniq_group_) {
       GE_ASSERT_SUCCESS(GenHeaderCodesSummaryBody(), "Generate tiling data summary body failed.");
     }
     GE_ASSERT_SUCCESS(GenHeaderCodesTail(), "Generate tiling data tail failed.");
   }
   if (!is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenGetTilingForScheduleResult());
     if (config_.enable_autofuse_pgo) {
        GE_ASSERT_SUCCESS(GenPGOGetTilingForAll());
        GE_ASSERT_SUCCESS(GenPGOByCoreNumTilingForAll());
     }
   }
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenTilingTail(std::map<std::string, std::string>& tiling_res,
                                            GenTilingTailImplExtParams ext_params) {
   var_relations_ = std::move(ext_params.var_relations);
   enable_group_parallels_ = std::move(ext_params.enable_group_parallels);
   workspace_tensor_id_set_ = std::move(ext_params.workspace_tensor_id_set);
   if (!ext_params.cache_reuse_info.empty()) {
     cache_reuse_info_ = std::move(ext_params.cache_reuse_info);
     with_reuse_info_ = true;
   }
   tiling_func_.Reset();
   tiling_head_.Reset();
   tiling_data_.Reset();
   tiling_func_.AddLine("#include \"" + kDefaultTilingHeadFileName + "\"");
   tiling_func_.AddLine("namespace optiling {");
   GE_ASSERT_SUCCESS(GenScheduleGroupTilingTail(), "Generate tiling data tail inner failed.");

  // 生成TilingCacheContext静态成员变量定义（必须在cpp文件中，否则会链接错误）
  if (config_.cache_enabled_at_compile_time) {
    GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenTilingCacheContextStaticDefs(tiling_func_),
                      "Generate TilingCacheContext static defs failed.");
  }
   tiling_head_.AddLine("} // namespace optiling");
   tiling_func_.AddLine("} // namespace optiling");
   tiling_res[kTilingScheduleGroupTailIdentify] += tiling_func_.GetOutputStr();
   tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
   if (config_.gen_tiling_data) {
     tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenGetPerf() {
   tiling_func_.AddLine("double GetPerf(" + config_.tiling_data_type_name + " &tiling_data) {");
   tiling_head_.AddLine("double GetPerf(" + config_.tiling_data_type_name + " &tiling_data);");
   tiling_func_.AddLine(
       "  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tiling_data.get_tiling_key(), "
       "tiling_data.get_block_dim());");
   tiling_func_.AddLine("  return tilingCaseImplPtr->GetPerf(tiling_data);");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenGetSummary() {
   tiling_func_.AddLine("void GetSummary(" + config_.tiling_data_type_name +
     " &tiling_data) {");
   tiling_head_.AddLine("void GetSummary(" + config_.tiling_data_type_name +
     " &tiling_data);");
   tiling_func_.AddLine("  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tiling_data.get_tiling_key(), tiling_data.get_block_dim());");
   tiling_func_.AddLine("  if (tilingCaseImplPtr == nullptr) {");
   tiling_func_.AddLine("    return;");
   tiling_func_.AddLine("  }");
   if (hardware_has_ub_) {
     tiling_func_.AddLine("  double ub_radio;");
     tiling_func_.AddLine("  tilingCaseImplPtr->TilingSummary(tiling_data, ub_radio);");
   } else {
     tiling_func_.AddLine("  tilingCaseImplPtr->TilingSummary(tiling_data);");
   }
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenTilingKeyFunc()
 {
   GE_ASSERT_SUCCESS(GenTilingImplBaseClass(), "Generate base class failed.");
   for (const auto &model_info : tiling_model_info_) {
     GE_ASSERT_SUCCESS(GenSolverTiling(model_info), "Generate do op tiling failed.");
     GE_ASSERT_SUCCESS(GenTilingCaseImpl(model_info), "Generate solver definition failed.");
   }
   GE_ASSERT_SUCCESS(GenImplPtr(), "Generate func call entrance failed.");
   GE_ASSERT_SUCCESS(GenGetTilingKey(), "Generate func call entrance failed.");
   if (config_.enable_autofuse_pgo) {
      GE_ASSERT_SUCCESS(GenPGOSearchTilingKey(), "Generate func call entrance failed.");
   }
   GE_ASSERT_SUCCESS(GenTilingFuncCallEntrance(), "Generate func call entrance failed.");
   if (config_.enable_autofuse_pgo) {
      GE_ASSERT_SUCCESS(GenPGOByCoreNumSearchTilingKey(), "Generate pgo by core num func call entrance failed.");
   }
   return ge::SUCCESS;
 }
 
 ge::Status TilingCodeGenImpl::GenTiling(std::map<std::string, std::string> &tiling_res,
                                         std::unordered_map<std::string, std::string> cache_reuse_info,
                                         uint32_t cache_capacity,
                                         const EnableGroupParallels &enable_group_parallels) {
   enable_group_parallels_ = enable_group_parallels;
   if (config_.enable_autofuse_pgo) {
      GE_ASSERT_SUCCESS(GenEnableGroupParallelPgoInvoke("autofuse_tiling_data", true, "      ", arrange_code_));
   }
   cache_capacity_ = cache_capacity;
   if (!(cache_reuse_info.empty())) {
     cache_reuse_info_ = cache_reuse_info;
     with_reuse_info_ = true;
   }
   // make sure input model info is valid
   tiling_head_.Reset();
   tiling_func_.Reset();
   tiling_data_.Reset();
   GE_ASSERT_TRUE(!tiling_model_info_.empty());
   GE_ASSERT_SUCCESS(tiling_data_manager_.Init());
   GE_ASSERT_SUCCESS(GenScheduleGroupTilingHead());
   const auto &cur_ident = tiling_model_info_[0].schedule_group_ident;
   tiling_func_.AddLine("#include \"" + kDefaultTilingHeadFileName + "\"");
   tiling_func_.AddLine("namespace optiling{");
   if (!is_uniq_group_) {
     tiling_head_.AddLine("namespace " + cur_ident.GetGroupPrefix() + " {");
     tiling_func_.AddLine("namespace " + cur_ident.GetGroupPrefix() + " {");
   }
   GELOGD("Generate tiling code for %s of %s reuse_ident is %s.", cur_ident.GetGroupPrefix().c_str(),
          op_name_.c_str(), tiling_model_info_[0].reuse_schedule_group->reuse_group_ident.GetGroupPrefix().c_str());
   if (tiling_model_info_[0].reuse_schedule_group->IsReuseGroup(cur_ident)) {
     if (config_.enable_autofuse_pgo) {
        GE_ASSERT_SUCCESS(GenPGOReuseGroupTilingWrapper(), "Generate func call entrance failed.");
     }
     return GenReuseGroupTilingWrapper(tiling_res);
   }
   GE_ASSERT_SUCCESS(GenTilingKeyFunc());
   if (!is_uniq_group_) {
     GE_ASSERT_SUCCESS(GenGetPerf(), "Generate getperf failed.");
     GE_ASSERT_SUCCESS(GenGetSummary(), "Generate getsummary failed.");
     tiling_head_.AddLine("} // namespace " + cur_ident.GetGroupPrefix());
     tiling_func_.AddLine("} // namespace " + cur_ident.GetGroupPrefix());
   }
   tiling_func_.AddLine("} // namespace optiling");
   if (config_.gen_tiling_data) {
     tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
   }
   tiling_res[cur_ident.GetGroupPrefixSnakeCase()] = tiling_func_.GetOutputStr();
   tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenReuseGroupTilingWrapperGetTiling(
     const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
     std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter) {
   if (with_reuse_info_) {
     tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                          " &tiling_data, " + (is_uniq_group_
                                                 ? ""
                                                 : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                          "int32_t tilingCaseId, " +
                          reuse_prefix + "::GroupLevelCache* cache) {");
     tiling_head_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                          " &tiling_data, " + (is_uniq_group_
                                                 ? ""
                                                 : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                          "int32_t tilingCaseId, " +
                          reuse_prefix + "::GroupLevelCache* cache = nullptr);");
   } else {
     tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                          (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                          "int32_t tilingCaseId) {");
     tiling_head_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                          (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                          "int32_t tilingCaseId);");
   }
   auto reuse_tiling_data = "  auto reuse_tiling_data = RefToRef<" + cur_prefix + "TilingData, " + reuse_prefix +
                            "TilingData>(tiling_data);";
   tiling_func_.AddLine(reuse_tiling_data);
   GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
   tiling_func_.AddLine("  auto ret = " + reuse_prefix + "::GetTiling(reuse_tiling_data, " +
                        (is_uniq_group_ ? "" : "workspace_map, ") + "tilingCaseId" + (with_reuse_info_ ? ", cache" : "")
                        + ");");
   tiling_func_.AddLine("  tiling_data = RefToRef<" + reuse_prefix + "TilingData, " + cur_prefix +
                        "TilingData>(reuse_tiling_data);");
   tiling_func_.AddLine("  return ret;");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenReuseGroupTilingWrapperGetPerf(
     const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
     std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter) {
   tiling_func_.AddLine("double GetPerf(" + config_.tiling_data_type_name + " &tiling_data) {");
   tiling_head_.AddLine("double GetPerf(" + config_.tiling_data_type_name + " &tiling_data);");
   auto reuse_tiling_data = "  auto reuse_tiling_data = RefToRef<" + cur_prefix + "TilingData, " + reuse_prefix +
                            "TilingData>(tiling_data);";
   tiling_func_.AddLine(reuse_tiling_data);
   GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
   tiling_func_.AddLine("  return " + reuse_prefix + "::GetPerf(reuse_tiling_data);");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenReuseGroupTilingWrapperGetSummary(
     const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
     std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter) {
   tiling_func_.AddLine("void GetSummary(" + config_.tiling_data_type_name + " &tiling_data) {");
   tiling_head_.AddLine("void GetSummary(" + config_.tiling_data_type_name + " &tiling_data);");
   auto reuse_tiling_data = "  auto reuse_tiling_data = RefToRef<" + cur_prefix + "TilingData, " + reuse_prefix +
                            "TilingData>(tiling_data);";
   tiling_func_.AddLine(reuse_tiling_data);
   GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
   tiling_func_.AddLine(reuse_prefix + "::GetSummary(reuse_tiling_data);");
   tiling_func_.AddLine("}");
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenReuseGroupTilingWrapper(std::map<std::string, std::string> &tiling_res) {
   const auto &reuse_ident = tiling_model_info_[0].reuse_schedule_group->reuse_group_ident;
   const auto &cur_ident = tiling_model_info_[0].schedule_group_ident;
   const auto &reuse_prefix = reuse_ident.GetGroupPrefix();
   const auto &cur_prefix = cur_ident.GetGroupPrefix();
   GELOGD("Cast reuse group %s to %s of %s.", reuse_prefix.c_str(), cur_prefix.c_str(), op_name_.c_str());
   const auto iter = tiling_model_info_[0].reuse_schedule_group->schedule_group_to_info.find(cur_ident);
   GE_ASSERT_TRUE(iter!= tiling_model_info_[0].reuse_schedule_group->schedule_group_to_info.cend(),
                  "Find reuse group %s failed.", cur_prefix.c_str());
   const auto &reuse_info = tiling_model_info_[0].reuse_schedule_group->info;
   const auto &reuse_input_axes = reuse_info.reuse_input_axes;
   GE_ASSERT_TRUE(iter->second.reuse_input_axes.size() == reuse_input_axes.size(),
                  "Reuse group %s input axes size %zu not equal to current axes size %zu.", cur_prefix.c_str(),
                  iter->second.reuse_input_axes.size(), reuse_input_axes.size());
   const auto &reuse_search_axes = reuse_info.reuse_search_axes;
   GE_ASSERT_TRUE(iter->second.reuse_search_axes.size() == reuse_search_axes.size(),
                  "Reuse group %s search axes size %zu not equal to current axes size %zu.", cur_prefix.c_str(),
                  iter->second.reuse_search_axes.size(), reuse_search_axes.size());
   // Gen GetTiling
   GE_ASSERT_SUCCESS(GenReuseGroupTilingWrapperGetTiling(cur_prefix, reuse_prefix, reuse_info, iter));
   // Gen GetPerf
   GE_ASSERT_SUCCESS(GenReuseGroupTilingWrapperGetPerf(cur_prefix, reuse_prefix, reuse_info, iter));
   // Gen GetSummary
   GE_ASSERT_SUCCESS(GenReuseGroupTilingWrapperGetSummary(cur_prefix, reuse_prefix, reuse_info, iter));
   tiling_head_.AddLine("} // namespace " + cur_prefix);
   tiling_func_.AddLine("} // namespace " + cur_prefix);
   tiling_func_.AddLine("} // namespace optiling");
   if (config_.gen_tiling_data) {
     tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
   }
   tiling_res[cur_ident.GetGroupPrefixSnakeCase()] = tiling_func_.GetOutputStr();
   tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
   GELOGD("Generate reuse group tiling wrapper for %s of %s success.", cur_prefix.c_str(), op_name_.c_str());
   return ge::SUCCESS;
 }

 ge::Status TilingCodeGenImpl::GenPGOReuseGroupTilingWrapper() {
   const auto &reuse_ident = tiling_model_info_[0].reuse_schedule_group->reuse_group_ident;
   const auto &cur_ident = tiling_model_info_[0].schedule_group_ident;
   const auto &reuse_prefix = reuse_ident.GetGroupPrefix();
   const auto &cur_prefix = cur_ident.GetGroupPrefix();
   const auto &reuse_item_prefix = reuse_ident.GetItemPrefix();
   const auto &cur_item_prefix = cur_ident.GetItemPrefix();

   GELOGD("Cast pgo reuse group %s to %s of %s.", reuse_prefix.c_str(), cur_prefix.c_str(), op_name_.c_str());

   // Gen PGOSearchTilingKey
   std::string params = config_.tiling_data_type_name + " &tiling_data, int32_t tilingCaseId";
   tiling_head_.AddLine(
       "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params +
       ", AutofuseTilingData* autofuseTilingData," + GenLaunchLikeInputOutputDef() +
       "void* stream, uint32_t workspaceSize, double& best_perf, " +
       "std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec={});");
   tiling_func_.AddLine(
       "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params +
       ", AutofuseTilingData* autofuseTilingData," + GenLaunchLikeInputOutputDef() +
       "void* stream, uint32_t workspaceSize, double& best_perf, " +
       "std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec) {");
   tiling_func_.AddLine("  double cur_perf = DBL_MAX;");
   tiling_func_.AddLine("  AutofuseTilingData autofuse_tiling_data_tmp = *autofuseTilingData;");
   auto reuse_tiling_data = "  auto reuse_tiling_data = RefToRef<" + reuse_prefix + "TilingData, " + cur_prefix +
                            "TilingData>(autofuse_tiling_data_tmp." + reuse_item_prefix + "_tiling_data);";
   tiling_func_.AddLine(reuse_tiling_data);
   tiling_func_.AddLine("  autofuse_tiling_data_tmp." + cur_item_prefix + "_tiling_data = reuse_tiling_data;");
   tiling_func_.AddLine("  workspaceSize = GetWorkspaceSize(autofuse_tiling_data_tmp);");
   if (!config_.is_inductor_scene) {
      tiling_func_.AddLine("  workspaceSize += 16 * 1024 * 1024;");
   }
   std::string invoke_code;
   GE_ASSERT_SUCCESS(GenEnableGroupParallelPgoInvoke("autofuse_tiling_data_tmp", false, "      ", invoke_code));
   tiling_func_.AddLine("  PgoConfig::Instance().single_callback(" + GenLaunchLikeInputOutputDef(false) +
                        "stream, workspaceSize, &autofuse_tiling_data_tmp, &cur_perf);");
   tiling_func_.AddLine("  AutofuseTilingDataPerf tiling_perf;");
   tiling_func_.AddLine("  tiling_perf.tiling_data = autofuse_tiling_data_tmp;");
   tiling_func_.AddLine("  tiling_perf.best_perf = cur_perf;");
   tiling_func_.AddLine("  tiling_data_list.push_back(tiling_perf);");
   tiling_func_.AddLine("  if (best_perf > cur_perf) {");
   tiling_func_.AddLine("    *autofuseTilingData = autofuse_tiling_data_tmp;");
   tiling_func_.AddLine("    best_perf = cur_perf;");
   tiling_func_.AddLine("  }");
   tiling_func_.AddLine("  return true;");
   tiling_func_.AddLine("}");
   GELOGD("Generate pgo reuse group tiling wrapper for %s of %s success.", cur_prefix.c_str(), op_name_.c_str());
   return ge::SUCCESS;
 }

 bool TilingCodeGenImpl::IsScheduleResultEnableParallel(const size_t asc_graph_id, const size_t impl_graph_id) const {
   bool enable_group_parallel = false;
   for (const auto &info : tiling_model_info_) {
     if (info.schedule_group_ident.asc_graph_id == asc_graph_id &&
         info.schedule_group_ident.impl_graph_id == impl_graph_id) {
       enable_group_parallel = info.enable_group_parallel;
       break;
     }
   }
   GELOGD("Enable parallel flag of graph%d_result%d is: %d", asc_graph_id, impl_graph_id, enable_group_parallel);
   return enable_group_parallel;
 }

 bool TilingCodeGenImpl::GenUpdateCurPerfAndBlockByGroupIfNeeded(const size_t asc_graph_id,
                                                                 const AscGraphNamepspaceMap &asc_graph_map) const {
   for (const auto &graph_info : asc_graph_map) {
     if (IsScheduleResultEnableParallel(asc_graph_id, graph_info.first)) {
       GenUpdateCurPerfAndBlockByGroup();
       return true;
     }
   }
   return false;
 }
 }  // namespace att
 
