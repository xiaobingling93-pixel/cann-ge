/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling_data.h"
#include <sstream>

#include "common_utils.h"
#include "common/ge_common/debug/log.h"

using namespace ascgen_utils;

codegen::TilingData::TilingData(const std::string &kernel, const std::string &name_class)
    : class_name(name_class), kernel_name(kernel){}

std::string codegen::TilingData::macros_and_includes = { // 不却分是否const
    "#include <stdint.h>\n"
    "#include \"kernel_tiling/kernel_tiling.h\"\n"
    "#define BEGIN_TILING_DATA_DEF_T(name) struct name {\n"
    "#define TILING_DATA_FIELD_DEF_T(type, name) \\\n"
    "  type name; \\\n"
    "  inline void set_##name(type value) { name = value; } \\\n"
    "  inline type get_##name() { return name; } \\\n"
    "  inline type* get_addr_##name() {return &name;}\n"
    "#define END_TILING_DATA_DEF_T };\n"
    "#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \\\n"
    "  struct_type filed_name;\n"};

std::string codegen::TilingData::common_tiling_filed = { // 非const模式
    "  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);\n"
    "  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);\n"
    "  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);\n"
    "  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);"};

std::string codegen::TilingData::GenGenTilingDataFieldConstDefFunc() const {
  std::stringstream ss;
  ss << "std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint32_t value) {" << std::endl;
  ss << "  std::stringstream ss_mid;" << std::endl;
  ss << "  ss_mid << \"const uint32_t \";" << std::endl;
  ss << "  ss_mid << f_name << \" = \" << std::to_string(value) << \";\" << std::endl;" << std::endl;
  ss << "  return ss_mid.str();" <<  std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string codegen::TilingData::GenGenTilingDataFieldConstValueFunc() const {
  std::stringstream ss;
  ss << "std::string GenTilingDataFieldConstValueFunc(uint32_t value) {" << std::endl;
  ss << "  std::stringstream ss_mid;" << std::endl;
  ss << "  ss_mid << std::to_string(value) << std::endl;" << std::endl;
  ss << "  return ss_mid.str();" <<  std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string codegen::TilingData::GetCommonTilingField(bool is_group,
                                                      const ascir::FusedScheduledResult& fused_schedule_result) {
  std::stringstream ss;
  std::vector<ascir::TensorId> workspace_tensor_id = GetWorkspaceTensorIdListInOneScheduleResult(fused_schedule_result);
  std::vector<std::string> common_tiling_fileds = {"block_dim", "corenum", "ub_size", "hbm_size"};
  for (auto tId : workspace_tensor_id) {
    common_tiling_fileds.push_back("workspace" + std::to_string(tId));
  }
  if (!const_mode_) {
    // 非const模式
    ss << common_tiling_filed << std::endl;
    for (auto tId : workspace_tensor_id) {
      ss << "  TILING_DATA_FIELD_DEF_T(uint32_t, workspace" << std::to_string(tId) << ");" << std::endl;
    }
    if (is_group || ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) &&
        (fused_schedule_result.node_idx_to_scheduled_results[0].size() == 1) &&
        (fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1))) {
      ss << "  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);";
      return ss.str();
    }
    for (size_t i = 0U;i < fused_schedule_result.node_idx_to_scheduled_results.size();i++) {
      ss << "  TILING_DATA_FIELD_DEF_T(uint32_t, " << "graph" << std::to_string(i) << "_tiling_key);";
      if (i < (fused_schedule_result.node_idx_to_scheduled_results.size() - 1U)) {
        ss << std::endl;
      }
    }
    return ss.str();
  }

  // const模式
  if (is_group || ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) &&
        (fused_schedule_result.node_idx_to_scheduled_results[0].size() == 1) &&
        (fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1))) {
    common_tiling_fileds.push_back("tiling_key");
  } else {
    for (size_t i = 0U;i < fused_schedule_result.node_idx_to_scheduled_results.size();i++) {
      common_tiling_fileds.push_back("graph" + std::to_string(i) + "_tiling_key");
    }
  }
  uint32_t idx = 0U;
  for (auto &field : common_tiling_fileds) {
    std::string field_func_str = GetNameOfGenTilingDataFieldConstDefFunc(field);
    std::string field_func_str_simple = GetNameOfGenTilingDataFieldConstDefFuncSimple(field);
    std::string field_def = field_func_str + "_def";
    pre_var_ss << "  std::string " << field_def << " = " << field_func_str_simple << ";" << std::endl;
    ss << "  " << field_def;
    // 最后一个不加换行，在外面加换行
    if (idx < (common_tiling_fileds.size() - 1U)) {
      ss << std::endl;
    }
    field_var_defs_.push_back(field_def);
    idx++;
  }

  return ss.str();
}

std::string codegen::TilingData::pgo_perf_struct = {
    "struct AutofuseTilingDataPerf {\n"
    "  AutofuseTilingData tiling_data;\n"
    "  double best_perf;\n"
    "};\n"};

ge::Status codegen::TilingData::ProcessCubeFusionResult(ascir::FusedScheduledResult &schedule_result) {
  if (ascgen_utils::IsCubeUBFusedScheduled(schedule_result)) {
    GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionResult(schedule_result));
  } else if (ascgen_utils::IsCubeCommonFusedScheduled(schedule_result)) {
    GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionCommonResult(schedule_result));
  }
  return ge::SUCCESS;
}

std::string codegen::TilingData::Generate(const ascir::FusedScheduledResult& fused_schedule_result) {
  std::stringstream ss;
  std::stringstream ss1;    // ss1 是最外层的tilingData结构体定义
  std::stringstream ss2;    // ss1 是最内层的子tilingData结构体定义

  ss << "#ifndef __" << this->kernel_name << "_Tiling_Data_H__" << std::endl;
  ss << "#define __" << this->kernel_name << "_Tiling_Data_H__" << std::endl;
  ss << macros_and_includes << std::endl;

  auto generate_footer = [this, &ss, &ss1, &ss2]() {
    ss1 << this->ClassEnd() << std::endl << std::endl;
    ss << ss2.str() << ss1.str();
    std::string input_type = this->kernel_name + this->class_name;
    if (input_type != "AutofuseTilingData") {
      ss << "using AutofuseTilingData = " << input_type << ";" << std::endl;
    }
    ss << pgo_perf_struct;
    ss << "#endif" << std::endl;
  };
  if (ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    GE_ASSERT(fused_schedule_result.node_idx_to_scheduled_results.size() == 1U, "Cube Fixpip results just one.");
    GE_ASSERT(fused_schedule_result.node_idx_to_scheduled_results[0].size() == 1U,
              "Cube Fixpip scheduled_results just one.");
    GE_ASSERT(fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1U,
              "Cube Fixpip schedule groups just one.");
    ss1 << this->ClassBegin(this->kernel_name, this->class_name) << std::endl;
    ss1 << GetCommonTilingField(false, fused_schedule_result) << std::endl;
    this->ProcessSingleGroup(fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0], ss1);
    GELOGI("TilingCaseId:ProcessSingleGroup\n");
    generate_footer();
    return ss.str();
  }
  ascir::FusedScheduledResult elemwise_schedule_result = fused_schedule_result;
  GE_ASSERT_SUCCESS(ProcessCubeFusionResult(elemwise_schedule_result));

  ss1 << this->ClassBegin(this->kernel_name, this->class_name) << std::endl;
  ss1 << GetCommonTilingField(false, elemwise_schedule_result) << std::endl;

  if ((elemwise_schedule_result.node_idx_to_scheduled_results.size() == 1U) &&
      (elemwise_schedule_result.node_idx_to_scheduled_results[0].size() == 1U) &&
      (elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1U)) {
    this->ProcessSingleGroup(elemwise_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0], ss1);
    GELOGI("TilingCaseId:ProcessSingleGroup\n");
  } else {
    for (size_t i = 0U; i < elemwise_schedule_result.node_idx_to_scheduled_results.size(); i++) {
      auto scheduled_results = elemwise_schedule_result.node_idx_to_scheduled_results[i];
      for (size_t j = 0U; j < scheduled_results.size(); j++) {
        this->ProcessMultiGroup(j, i, scheduled_results[j].schedule_groups, ss1, ss2);
        GELOGI("TilingCaseId:ProcessMultiGroup\n");
      }
    }
  }
  generate_footer();
  return ss.str();
}

std::string codegen::TilingData::ClassBegin(
  const std::string& begin_kernel_name, const std::string& begin_class_name) const {
  std::stringstream ss;
  ss << "BEGIN_TILING_DATA_DEF_T(" << begin_kernel_name << begin_class_name << ")";
  return ss.str();
}

std::string codegen::TilingData::DataFieldDefine(ascir::SizeVar &size) const {
  std::stringstream ss;
  ss << "TILING_DATA_FIELD_DEF_T(uint32_t, " << std::string(size.expr.Str().get()) << ");";
  return ss.str();
}

std::string codegen::TilingData::DataFieldConstDefine(ascir::SizeVar &size) {
  std::stringstream ss;

  std::string field = std::string(size.expr.Str().get());
  std::string field_func_str = GetNameOfGenTilingDataFieldConstDefFunc(field);
  std::string field_func_str_simple = GetNameOfGenTilingDataFieldConstDefFuncSimple(field);
  std::string field_def = field_func_str + "_def";
  pre_var_ss << "  std::string " << field_def << " = " << field_func_str_simple << ";" << std::endl;
  ss << field_def;
  field_var_defs_.push_back(field_def);

  return ss.str();
}

std::string codegen::TilingData::StructDataFiledDefine(const std::string& type_name,
                                                       const std::string& filed_name) const {
  std::stringstream ss;
  ss << "TILING_DATA_FIELD_DEF_T_STRUCT(" << type_name << ", " << filed_name << ");";
  return ss.str();
}

std::string codegen::TilingData::ClassEnd() const {
  std::stringstream ss;
  ss << "END_TILING_DATA_DEF_T;";
  return ss.str();
}

std::string codegen::TilingData::ClassRegister() {
  std::stringstream ss;
  ss << "REGISTER_TILING_DATA_CLASS(" << this->kernel_name << ", " << this->class_name << ")";
  return ss.str();
}

ge::Status codegen::TilingData::GetApiTilingDataName(const ascir::NodeView& node, std::vector<std::string>& api_tiling_data_names) {
  // transpose api tiling data包含的字段：
  // param0, param1, param2, ... param17
  const std::vector<std::string> transpose_params = {"param0", "param1", "param2", "param3", "param4", "param5", "param6", "param7",
                                               "param8", "param9", "param10", "param11", "param12", "param13",
                                               "param14", "param15", "param16", "param17"};
  const std::vector<std::string> pad_params = {"srcHeight", "srcWidth", "srcOriWidth", "widthWithoutLastBlock", "blocksPerRow",
                                         "heightTiling", "heightFractal", "heightFractalTail", "mainLoopOffset",
                                         "tailBlockOffset", "tmpBuffer1BlockNum", "tmpBuffer1RowNum",
                                         "tmpBuffer2Offset", "widthTiling", "widthFractal", "widthFractalTail",
                                         "widthFractalTailAlingned", "brcbTiling", "brcbFractal", "brcbFractalTail",
                                         "maxRepeatTimes", "brcbTilingRepeatTimes", "brcbTilingRepeatTimesTail",
                                         "brcbFractalTailRepeatTimes", "brcbFractalTailRepeatTimesTail", "reserved"};
  std::map<std::string, std::vector<std::string>> node_with_api_tiling = {{"Transpose", transpose_params},
                                                                          {"Pad", pad_params}};
  auto it = node_with_api_tiling.find(node->GetType());
  if (it == node_with_api_tiling.end()) {
    GELOGE(ge::FAILED, "not supported const api tilingdata node type:%s.", node->GetType().c_str());
    return ge::FAILED;
  }

  api_tiling_data_names.assign(it->second.begin(), it->second.end());
  return ge::SUCCESS;
}

std::string codegen::TilingData::ConstApiTilingDataFiledDefine(std::string &type_name, std::string &field_name,
                                                               const ascir::NodeView& node) {
  std::vector<std::string> node_with_api_tiling;
  if (GetApiTilingDataName(node, node_with_api_tiling) != ge::SUCCESS) {
    return "";
  }

  std::stringstream ss;
  bool is_first = true;
  for (auto &param : node_with_api_tiling) {
    std::string param_func_str = GetNameOfGenTilingDataFieldConstDefFunc(param);
    std::string param_func_str_simple = GetNameOfGenTilingDataFieldConstValueFuncSimple(param);
    std::string param_def = param_func_str + "_field_def";
    pre_var_ss << "  std::string " << param_def << " = " << param_func_str_simple << ";" << std::endl;
    field_var_defs_.push_back(param_def);

    if (is_first) {
      ss << "const " << type_name << " " << field_name << " = {"
      << param_def;
      is_first = false;
    } else {
      ss << ", " << param_def;
    }
  }

  ss << "};" << std::endl;
  return ss.str();
}

void codegen::TilingData::AddApiTilingData(const ge::AscGraph &graph, std::stringstream &ss, uint32_t tiling_case_id)
{
  for (const auto &node : graph.GetAllNodes()) {
    std::string device_type_name;
    std::string host_type_name;
    std::string field_name;
    if (ge::SUCCESS == GetApiTilingTypeName(node, device_type_name) && (ge::SUCCESS == GetApiTilingFieldName(node, field_name))) {
      host_type_name = "optiling::" + device_type_name;
      field_name = field_name + "_" + std::to_string(tiling_case_id);
      const_tiling_data_field.push_back(field_name);

      if (const_mode_) {
        std::string host_api_tiling_data_def = this->ConstApiTilingDataFiledDefine(device_type_name, field_name, node);
        ss << "  " << host_api_tiling_data_def << std::endl;
      } else {
        std::string dev_api_tiling_data_def = this->StructDataFiledDefine(device_type_name, field_name);
        ss << "  " << dev_api_tiling_data_def << std::endl;
        }
      ConstTilingDataFieldPopBack();
    }
  }
}

void codegen::TilingData::GetTqueAndTbufId(const ge::AscGraph& graph, std::set<int64_t>& q_ids, std::set<int64_t>& b_ids) {
  for (auto node : graph.GetAllNodes()) {
    for (auto out : node->outputs()) {
      int64_t q_id = out->attr.que.id;
      int64_t b_id = out->attr.buf.id;
      if (q_ids.find(q_id) == q_ids.end()) {
        q_ids.insert(q_id);
      }
      if (b_ids.find(b_id) == b_ids.end()) {
        b_ids.insert(b_id);
      }
    }
  }
}

void codegen::TilingData::GetTmpBufName(const ge::AscGraph& graph, std::set<int64_t>& b_ids) {
  for (auto node : graph.GetAllNodes()) {
    for (auto &tmp_buffer : node->attr.tmp_buffers) {
      GELOGD("Get tmp buffer [%ld, %s] for node %s.", tmp_buffer.buf_desc.life_time_axis_id,
             tmp_buffer.buf_desc.size.Str().get(), node->GetNamePtr());
      if (tmp_buffer.id == -1L) {
        continue;
      }
      b_ids.insert(tmp_buffer.id);
    }
  }
}

void codegen::TilingData::GenTqueTbufTmpBufFunc(const std::set<int64_t>& q_ids, const std::set<int64_t>& b_ids, std::stringstream& ss) {
  for (const auto& q_id : q_ids) {
    if (q_id < 0) {
      continue;
    }
    std::string field_def = const_mode_ ? this->TqueOrTbufDataFieldConstDefine(q_id, "q") : this->TqueOrTbufDataFieldDefine(q_id, "q");
    ss << "  " << field_def << std::endl;
  }
  for (const auto& b_id : b_ids) {
    if (b_id < 0) {
      continue;
    }
    std::string field_def = const_mode_ ? this->TqueOrTbufDataFieldConstDefine(b_id, "b") : this->TqueOrTbufDataFieldDefine(b_id, "b");
    ss << "  " << field_def << std::endl;
  }
}

void codegen::TilingData::ProcessSingleGroup(const ascir::ScheduleGroup &schedule_group, std::stringstream &ss) {
  std::unordered_set<std::string> size_var_names;
  std::set<int64_t> q_ids;
  std::set<int64_t> b_ids;
  for (size_t i = 0U; i < schedule_group.impl_graphs.size(); i++) {
    auto &graph = schedule_group.impl_graphs[i];
    for (auto size : graph.GetAllSizeVar()) {
      if (size->expr.IsConstExpr()) {
        continue;
      }
      if (size_var_names.find(std::string(size->expr.Str().get())) == size_var_names.end()) {
        std::string field_def = const_mode_ ? this->DataFieldConstDefine(*size) : this->DataFieldDefine(*size);
        ss << "  " << field_def << std::endl;
        size_var_names.emplace(std::string(size->expr.Str().get()));
      }
    }
    GetTqueAndTbufId(graph, q_ids, b_ids);
    GetTmpBufName(graph, b_ids);
    AddApiTilingData(graph, ss, i);
    GELOGI("TilingCaseId:ProcessSingleGroup, tilingcaseNum:%d\n", schedule_group.impl_graphs.size());
  }
  GenTqueTbufTmpBufFunc(q_ids, b_ids, ss);
  return;
}

void codegen::TilingData::ProcessMultiGroup(uint64_t pos, const int graph_id,
                                            const std::vector<ascir::ScheduleGroup> &schedule_groups,
                                            std::stringstream &ss1, std::stringstream &ss2) {
  for (uint64_t i = 0; i < schedule_groups.size(); i++) {
    std::stringstream struct_name;
    struct_name << "AscGraph" << std::to_string(graph_id) << "Schedule";
    std::stringstream struct_name_tail;
    struct_name_tail << "Result" << std::to_string(pos) << "G" << std::to_string(i);
    struct_name << struct_name_tail.str();
    std::string filed_name = "graph" + std::to_string(graph_id) + "_" + CamelToLowerSneak(struct_name_tail.str() + this->class_name);
    ss1 << "  " << this->StructDataFiledDefine(struct_name.str() + this->class_name, filed_name) << std::endl;
    const_tiling_data_field.push_back(filed_name);
    std::unordered_set<std::string> size_var_names;
    ss2 << this->ClassBegin(struct_name.str(), this->class_name) << std::endl;
    ss2 << GetCommonTilingField(true, ascir::FusedScheduledResult()) << std::endl;

    std::set<int64_t> q_ids;
    std::set<int64_t> b_ids;
    for (uint32_t j = 0; j <  schedule_groups[i].impl_graphs.size(); j++) {
      auto &graph =  schedule_groups[i].impl_graphs[j];
      for (auto size : graph.GetAllSizeVar()) {
        if (size->expr.IsConstExpr()) {
          continue;
        }
        if (size_var_names.find(std::string(size->expr.Str().get())) == size_var_names.end()) {
          std::string field_def = const_mode_ ? this->DataFieldConstDefine(*size) : this->DataFieldDefine(*size);
          ss2 << "  " << field_def << std::endl;
          size_var_names.emplace(std::string(size->expr.Str().get()));
        }
      }
      GetTqueAndTbufId(graph, q_ids, b_ids);
      GetTmpBufName(graph, b_ids);
      AddApiTilingData(graph, ss2 , j);
      GELOGI("TilingCaseId:ProcessMultiGroup, i_%d, i_num:%d, j_%d, j_num:%d\n", i, schedule_groups.size(), j, schedule_groups[i].impl_graphs.size());
    }
    GenTqueTbufTmpBufFunc(q_ids, b_ids, ss2);
    ss2 << this->ClassEnd() << std::endl;
    ConstTilingDataFieldPopBack();
    ss2 << std::endl;
  }
  return;
}

std::string codegen::TilingData::GenStringReplaceFunc() const {
  std::stringstream ss;
  ss << "void replaceSubstring(std::string& ori_str, ";
  ss << "const std::string& old_sub_str, ";
  ss << "const std::string& new_sub_str) {" << std::endl;
  ss << "  size_t pos = ori_str.find(old_sub_str);" << std::endl;
  ss << "  if (pos != std::string::npos) {" << std::endl;
  ss << "    ori_str.replace(pos, old_sub_str.length(), new_sub_str);" << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string codegen::TilingData::GenConstGenResultReplace() {
  std::stringstream ss;

  for (auto &field_var : field_var_defs_) {
    ss << "  replaceSubstring(tiling_data_const_gen_result, \"" << field_var << "\","  << field_var << ");" << std::endl;
  }

  return ss.str();
}

void codegen::TilingData::ConstTilingDataFieldPopBack() {
  if (const_tiling_data_field.size() > 0) {
    const_tiling_data_field.pop_back();
  } else {
    // todo: tilingData的生成过程中遇错终止, 此处是内部逻辑错误，先打印一条Error日志
    GELOGE(ge::FAILED, "The const_tiling_data_field is empty.");
  }
}

std::string codegen::TilingData::GenCVConstTilingData(const std::string &tiling_data_struct_name,
    bool is_inductor_scene) {
  std::stringstream ss;
  ss << "  int32_t basen_basem_align = compute_basen_basem_align();" << std::endl;
  ss << "  set_g_basen_basem_align(basen_basem_align);" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"basen_basem_align=%d, set_g_basen_basem_align=%d\", ";
  ss << "basen_basem_align, get_g_basen_basem_align());" << std::endl;
  ss << "  auto ret = AutofuseTilingWithConfig(config_file, &" << tiling_data_struct_name;
  ss << ", &workspace_size, &block_dim, ";
  ss << (is_inductor_scene ? "nullptr, 0);" : "&limit, 0);") << std::endl;
  ss << "  if (ret == -1) {" << std::endl;
  ss << "    uint32_t basen_basem_align_tmp = (uint32_t)basen_basem_align;" << std::endl;
  ss << "    // ub_size必大于 basen_basem_align_tmp" << std::endl;
  ss << "    limit.ub_size = limit.ub_size - basen_basem_align_tmp * cube_output_type_size;" << std::endl;
  ss << "    set_g_basen_basem_align(1);" << std::endl;
  ss << "    OP_LOGI(OP_NAME, \"set_g_basen_basem_align=%d, ub_size=%u\", get_g_basen_basem_align(), ub_size);"
      << std::endl;
  ss << "    (void)AutofuseTilingWithConfig(config_file, &" << tiling_data_struct_name;
  ss << ", &workspace_size, &block_dim, ";
  ss << (is_inductor_scene ? "nullptr, 1);" : "&limit, 1);") << std::endl;
  ss << "  }" << std::endl;
  return ss.str();
}

// GenerateConst生成的信息放在tiling func .cpp中
std::string codegen::TilingData::GenerateConst(const ascir::FusedScheduledResult& fused_schedule_result,
                                               bool is_inductor_scene) {
  if (!IsStaticSchedResult(fused_schedule_result)) {
    return "";
  }

  const_mode_ = true;
  // 生成GenConstTilingData, GenConstTilingData实现对tilingfunc的调用得到的tilingData为初值，初始化生成的常量TilingData
  std::stringstream ss;
  std::stringstream global_pre_def_ss;
  std::stringstream const_gen_ss;

  global_pre_def_ss << "std::string tiling_data_const_gen_result;" << std::endl;
  std::string tiling_data_struct_name = "TilingDataValue";
  const_tiling_data_field.push_back(tiling_data_struct_name);
  global_pre_def_ss << "AutofuseTilingData " << tiling_data_struct_name << ";" << std::endl << std::endl;;
  global_pre_def_ss << GenStringReplaceFunc() << std::endl;  // 生成一个字符串替换接口

  const_gen_ss << "extern \"C\" const char* GenConstTilingData(char* config_file, int aiv_num, int ub_size) {"
      << std::endl;
  const_gen_ss << "  uint32_t workspace_size;" << std::endl;
  const_gen_ss << "  uint32_t block_dim;" << std::endl;
  const_gen_ss << "  ResLimit limit;" << std::endl;
  const_gen_ss << "  limit.aiv_num = aiv_num;" << std::endl;
  const_gen_ss << "  limit.ub_size = ub_size - 256;" << std::endl;

  if (IsCubeFusedScheduled(fused_schedule_result)) {
    const_gen_ss << GenCVConstTilingData(tiling_data_struct_name, is_inductor_scene);
  } else {
    const_gen_ss << "  (void)AutofuseTilingWithConfig(config_file, &" << tiling_data_struct_name;
    if (is_inductor_scene) {
      const_gen_ss << ", &workspace_size, &block_dim, nullptr);" << std::endl;
    } else {
      const_gen_ss << ", &workspace_size, &block_dim, &limit);" << std::endl;
    }
  }

  pre_func_ss << GenGenTilingDataFieldConstDefFunc() << std::endl;
  pre_func_ss << GenGenTilingDataFieldConstValueFunc() << std::endl;
  std::string g_str = Generate(fused_schedule_result);
  global_pre_def_ss << pre_func_ss.str() << std::endl;   // 一些前置函数定义放在前面,

  const_gen_ss << pre_var_ss.str() << std::endl;     // 前置函数的调用，生成"const声明"放在这里
  const_gen_ss << "  tiling_data_const_gen_result = R\"(" << g_str << ")\";" << std::endl;
  const_gen_ss << GenConstGenResultReplace() << std::endl;
  const_gen_ss << "  return tiling_data_const_gen_result.c_str();" << std::endl;
  const_gen_ss << "}" << std::endl;
  ConstTilingDataFieldPopBack();

  ss << global_pre_def_ss.str();
  ss << const_gen_ss.str() << std::endl;

  const_mode_ = false;
  return ss.str();
}

std::string codegen::TilingData::GenTingDataField(std::string field_name) {
  if (!const_mode_) {
    return "";
  }

  std::stringstream ss;
  for (auto &field : const_tiling_data_field) {
    ss << field << ".";
  }
  ss << field_name;

  return ss.str();
}

std::string codegen::TilingData::GetNameOfGenTilingDataFieldConstDefFunc(const std::string field_name) {
  if (!const_mode_) {
    return "";
  }

  std::stringstream ss;
  ss << "Gen";
  for (auto &field : const_tiling_data_field) {
    ss << field << "_";
  }
  ss << field_name << "_field_DeclareFunc";

  return ss.str();
}

std::string codegen::TilingData::GetNameOfGenTilingDataFieldConstDefFuncSimple(const std::string field_name) {
  if (!const_mode_) {
    return "";
  }

  std::string complete_fields = GenTingDataField(field_name);
  std::stringstream ss;
  ss << "GenTilingDataFieldConstDefFunc(\"" << field_name << "\", " << complete_fields << ")";
  return ss.str();
}

std::string codegen::TilingData::GetNameOfGenTilingDataFieldConstValueFuncSimple(const std::string field_name) {
  if (!const_mode_) {
    return "";
  }

  std::string complete_fields = GenTingDataField(field_name);
  std::stringstream ss;
  ss << "GenTilingDataFieldConstValueFunc(" << complete_fields << ")";
  return ss.str();
}

std::string codegen::TilingData::DataFieldConstDefine(const std::string& buf_name) {
  std::stringstream ss;
  std::string field = buf_name + "_size";
  std::string field_func_str = GetNameOfGenTilingDataFieldConstDefFunc(field);
  std::string field_func_str_simple = GetNameOfGenTilingDataFieldConstDefFuncSimple(field);
  std::string field_def = field_func_str + "_def";
  pre_var_ss << "  std::string " << field_def << " = " << field_func_str_simple << ";" << std::endl;
  ss << field_def;
  field_var_defs_.push_back(field_def);

  return ss.str();
}

std::string codegen::TilingData::TqueOrTbufDataFieldDefine(int64_t index, const std::string& que_or_buf) const {
  std::stringstream ss;
  ss << "TILING_DATA_FIELD_DEF_T(uint32_t, " << que_or_buf << std::to_string(index) << "_size);";
  return ss.str();
}

std::string codegen::TilingData::TqueOrTbufDataFieldConstDefine(int64_t index, const std::string& que_or_buf) {
  return DataFieldConstDefine(que_or_buf + std::to_string(index));
}

std::string codegen::TilingData::TmpBufDataFieldDefine(const std::string& tmp_tbuf_name) const {
  std::stringstream ss;
  ss << "TILING_DATA_FIELD_DEF_T(uint32_t, " << tmp_tbuf_name << "_size);";
  return ss.str();
}

std::string codegen::TilingData::TmpBufDataFieldConstDefine(const std::string& tmp_tbuf_name) {
  return DataFieldConstDefine(tmp_tbuf_name);
}