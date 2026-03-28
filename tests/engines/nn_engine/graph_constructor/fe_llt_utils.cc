/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fe_llt_utils.h"
#include <iostream>
#include "common/util/json_util.h"
#include "ge/ge_api_types.h"
#include "itf_handler/itf_handler.h"
#include "mmpa/mmpa_api.h"

#define protected public
#define private public
#include "platform/platform_info.h"
#include "graph/ge_local_context.h"
#include "graph/utils/op_desc_utils.h"
#undef private
#undef protected

using namespace std;
namespace fe {

std::string GetCodeDir() {
  static std::string gCachedCodeDir;
  if (gCachedCodeDir.empty()) {
    const char *code_path_ptr = std::getenv("AIR_CODE_DIR");
    if (code_path_ptr != nullptr) {
      gCachedCodeDir = string(code_path_ptr);
    }
  }
  return gCachedCodeDir;
}

std::string GetGraphPath(const std::string &graph_name) {
  char_t env_value[MMPA_MAX_PATH];
  mmGetEnv("FE_ST_PATH", env_value, MMPA_MAX_PATH);
  return string(env_value) + "/net/" + graph_name;
}

uint32_t InitPlatformInfo(const std::string &soc_version, const bool is_force) {
  if (fe::PlatformInfoManager::Instance().init_flag_ && !is_force) {
    return 0;
  }
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  fe::PlatformInfoManager::Instance().init_flag_ = true;
  fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.Init();
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
  fe::PlatformInfoManager::GeInstance().platform_info_map_.clear();
  fe::PlatformInfoManager::GeInstance().platform_infos_map_.clear();
  init_ret = fe::PlatformInfoManager::GeInstance().LoadConfigFile(real_path);
  fe::PlatformInfoManager::GeInstance().init_flag_ = true;
  fe::PlatformInfoManager::GeInstance().opti_compilation_info_.soc_version = soc_version;
  fe::PlatformInfoManager::GeInstance().opti_compilation_infos_.Init();
  fe::PlatformInfoManager::GeInstance().opti_compilation_infos_.SetSocVersion(soc_version);
  return init_ret;
}
void SetPlatformSocVersion(const std::string &soc_version) {
  fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
  fe::PlatformInfoManager::GeInstance().opti_compilation_info_.soc_version = soc_version;
  fe::PlatformInfoManager::GeInstance().opti_compilation_infos_.SetSocVersion(soc_version);
}

void SetPrecisionMode(const std::string &precision_mode) {
  if (!precision_mode.empty()) {
    ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = precision_mode;
  }
}

void SetContextOption(const std::string &key, const std::string &value) {
  if (!key.empty() && !value.empty()) {
    ge::GetThreadLocalContext().graph_options_[key] = value;
  }
}

void InitWithSocVersion(const std::string &soc_version, const std::string &precision_mode) {
  ge::GetThreadLocalContext().graph_options_.clear();
  ge::GetThreadLocalContext().session_options_.clear();
  ge::GetThreadLocalContext().global_options_.clear();
  if (!precision_mode.empty()) {
    ge::GetThreadLocalContext().graph_options_.emplace(ge::PRECISION_MODE, precision_mode);
  }
  fe::InitPlatformInfo(soc_version, true);
  map<string, string> options;
  options.emplace(ge::SOC_VERSION, soc_version);
  Initialize(options);
}

void InitWithOptions(const std::map<std::string, std::string> &options) {
  ge::GetThreadLocalContext().graph_options_ = options;
  ge::GetThreadLocalContext().session_options_.clear();
  ge::GetThreadLocalContext().global_options_.clear();

  auto iter = options.find(ge::SOC_VERSION);
  if (iter == options.end()) {
    return;
  }
  fe::InitPlatformInfo(iter->second, true);
  Initialize(options);
}

void FillWeightValue(const ge::ComputeGraphPtr &graph) {
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != "Const") {
      continue;
    }
    std::vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(node);
    if (weights.empty()) {
      ge::ConstGeTensorDescPtr out_tensor = node->GetOpDesc()->GetOutputDescPtr(0);
      int64_t shape_size = out_tensor->GetShape().GetShapeSize();
      if (shape_size <= 0) {
        continue;
      }
      ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(*out_tensor);
      if (out_tensor->GetDataType() == ge::DT_UINT32 || out_tensor->GetDataType() == ge::DT_INT32 ||
          out_tensor->GetDataType() == ge::DT_FLOAT) {
        vector<int32_t> data_vec(shape_size, 1);
        weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int32_t));
      }
      if (out_tensor->GetDataType() == ge::DT_UINT64 || out_tensor->GetDataType() == ge::DT_INT64 ||
          out_tensor->GetDataType() == ge::DT_DOUBLE) {
        vector<int64_t> data_vec(shape_size, 1);
        weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int64_t));
      }
      if (out_tensor->GetDataType() == ge::DT_UINT16 || out_tensor->GetDataType() == ge::DT_INT16 ||
          out_tensor->GetDataType() == ge::DT_FLOAT16) {
        vector<int16_t> data_vec(shape_size, 1);
        weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int16_t));
      }
      if (out_tensor->GetDataType() == ge::DT_UINT8 || out_tensor->GetDataType() == ge::DT_INT8) {
        vector<int8_t> data_vec(shape_size, 1);
        weight->SetData(reinterpret_cast<uint8_t *>(data_vec.data()), shape_size * sizeof(int8_t));
      }
      ge::OpDescUtils::SetWeights(node->GetOpDesc(), weight);
    }
  }
}
void FillNodeParaType(const ge::NodePtr &node, fe::OpParamType type) {
  size_t in_num = node->GetOpDesc()->GetAllInputsSize();
  size_t out_num = node->GetOpDesc()->GetOutputsSize();
  std::vector<uint32_t> input_type_list(in_num, static_cast<uint32_t>(type));
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kInputParaTypeList, input_type_list);
  std::vector<std::string> input_name_list;
  for (size_t i = 0; i < in_num; ++i) {
    input_name_list.emplace_back("__input" + std::to_string(i));
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), kInputNameList, input_name_list);
  std::vector<uint32_t> output_type_list(out_num, static_cast<uint32_t>(type));
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kOutputParaTypeList, output_type_list);
  std::vector<std::string> output_name_list;
  for (size_t i = 0; i < out_num; ++i) {
    output_name_list.emplace_back("__output" + std::to_string(i));
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), kOutputNameList, output_name_list);
}
void FillGraphNodeParaType(const ge::ComputeGraphPtr &graph, fe::OpParamType type) {
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    FillNodeParaType(node, type);
  }
}

void CreateDir(const std::string &path) {
  std::string real_path = RealPath(path);
  if (real_path.empty()) {
    std::string command = "mkdir -p " + path;
    system((char*) command.c_str());
  }
  return;
}

void CreateFileAndFillContent(const std::string fileName, nlohmann::json json_obj, const bool flag) {
  std::ofstream outfile(fileName, std::ios::out);
  if (!outfile.is_open()) {
        return;
  }
  if (flag && !json_obj.empty()) {
    outfile << json_obj.dump(4);
  }
  outfile.close();
  return;
}

void CreateAndCopyJsonFile() {
  std::string ascend_opp_path = getenv("ASCEND_OPP_PATH");
  std::string fusion_config_path = ascend_opp_path + "/lib64/plugin/opskernel/fusion_pass/config/fusion_config.json";
  std::cout << "fusion_config_path is " << fusion_config_path.c_str() << std::endl;
  std::string graph_rule_path = ascend_opp_path + "/lib64/plugin/opskernel/fusion_rules/ai_core/built_in_graph_rules.json";
  std::cout << "graph_rule_path is " << graph_rule_path.c_str() << std::endl;
  char *current_path = new(std::nothrow) char[1024];
  getcwd(current_path, 1024);
  std::string cur_path(current_path);
  std::string cur_air_path = cur_path + "/air";
  std::string real_path = RealPath(cur_air_path);
  if (real_path.empty()) {
    cur_air_path = cur_path + "/ge-dev";
  }
  std::string new_config_path = cur_air_path + "/build/tests/engines/nn_engine/st/plugin/opskernel/fusion_pass/config/";
  std::cout << "new_config_path is " << new_config_path.c_str() << std::endl;
  std::string new_rule_path = cur_air_path + "/build/tests/engines/nn_engine/st/plugin/opskernel/fusion_rules/ai_core/";
  std::cout << "new_rule_path is " << new_rule_path.c_str() << std::endl;
  system(("mkdir -p " + new_config_path).c_str());
  system(("mkdir -p " + new_rule_path).c_str());
  std::string cmd = "cp -r " + fusion_config_path + " " + new_config_path;
  std::string cmd1= "cp -r " + graph_rule_path + " " + new_rule_path;
  system(cmd.c_str());
  system(cmd1.c_str());
  delete[] current_path;
}

void DelJsonFile() {
  cout << "Begin to rm fusion_pass and fusion_rules directory" << endl;
  char *current_path = new(std::nothrow) char[1024];
  getcwd(current_path, 1024);
  std::string cur_path(current_path);
  std::string cur_air_path = cur_path + "/air";
  std::string real_path = RealPath(cur_air_path);
  if (real_path.empty()) {
    cur_air_path = cur_path + "/ge-dev";
  }
  std::string fusion_pass_path = cur_air_path + "/build/tests/engines/nn_engine/st/plugin/opskernel/fusion_pass/";
  std::string fusion_rules_path = cur_air_path + "/build/tests/engines/nn_engine/st/plugin/opskernel/fusion_rules/";
  system(("rm -rf " + fusion_pass_path).c_str());
  system(("rm -rf " + fusion_rules_path).c_str());
  delete[] current_path;
}
}
