/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_FE_LLT_UTILS_H_
#define AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_FE_LLT_UTILS_H_

#include <string>
#include <nlohmann/json.hpp>
#include "graph/compute_graph.h"
#include "common/aicore_util_types.h"
#include "mmpa/mmpa_api.h"

namespace fe {
class EnvVarGuard {
public:
    EnvVarGuard(const mmEnvId var_name, const char* new_value) 
        : var_name_(var_name), saved_value_(nullptr) {
        MM_SYS_GET_ENV(var_name_, saved_value_);
        int32_t err = 0;
        MM_SYS_SET_ENV(var_name_, new_value, 1, err);
    }
    
    void Restore() {
        if (!restored_) {
            int32_t err = 0;
            if (saved_value_ != nullptr) {
                MM_SYS_SET_ENV(var_name_, saved_value_, 1, err);
            } else {
                MM_SYS_SET_ENV(var_name_, "", 1, err);
            }
            restored_ = true;
        }
    }
    
    ~EnvVarGuard() {
        Restore();
    }
    
    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;
    
private:
    mmEnvId var_name_;
    const char_t* saved_value_;
    bool restored_ = false;
};
std::string GetCodeDir();
std::string GetGraphPath(const std::string &graph_name);
uint32_t InitPlatformInfo(const std::string &soc_version, const bool is_force = false);
void SetPlatformSocVersion(const std::string &soc_version);
void SetPrecisionMode(const std::string &precision_mode);
void SetContextOption(const std::string &key, const std::string &value);
void InitWithSocVersion(const std::string &soc_version, const std::string &precision_mode);
void InitWithOptions(const std::map<std::string, std::string> &options);
void FillWeightValue(const ge::ComputeGraphPtr &graph);
void FillGraphNodeParaType(const ge::ComputeGraphPtr &graph, fe::OpParamType type = fe::OpParamType::REQUIRED);
void FillNodeParaType(const ge::NodePtr &node, fe::OpParamType type = fe::OpParamType::REQUIRED);
void CreateDir(const std::string &path);
void CreateFileAndFillContent(const std::string fileName,
                              nlohmann::json json_obj = nlohmann::json::object(), const bool flag = false);
void CreateAndCopyJsonFile();
void DelJsonFile();
}
#endif  // AIR_TEST_ENGINES_NNENG_GRAPH_CONSTRUCTOR_FE_LLT_UTILS_H_
