/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_kb_api_stub.h"

using namespace fe::CannKb;

namespace fe {
CannKBUtils::CannKBUtils() {
  init_flag_ = false;
  cann_kb_plugin_manager_ = nullptr;
  cann_kb_init_func_ = nullptr;
  cann_kb_finalize_func_ = nullptr;
}
 
CannKBUtils::~CannKBUtils() {
  if (!init_flag_) {
    return;
  }
  if (cann_kb_plugin_manager_ != nullptr) {
    cann_kb_plugin_manager_->CloseHandle();
  }
}

CannKBUtils &CannKBUtils::Instance() {
  static CannKBUtils cann_kb_utils;
  return cann_kb_utils;
}

bool CannKBUtils::InitCannKb() {
  return true;
}

CannKb::CANN_KB_STATUS CannKBUtils::CannKbInit(const std::map<std::string, std::string> &sysConfig,
                                    const std::map<std::string, std::string> &loadConfig) {
  return CannKb::CANN_KB_STATUS::CANN_KB_SUCC;
}

CannKb::CANN_KB_STATUS CannKBUtils::CannKbFinalize() {
  return CannKb::CANN_KB_STATUS::CANN_KB_SUCC;
}
                                        
CannKb::CANN_KB_STATUS CannKBUtils::RunCannKbSearch(const std::string &infoDict,
                                         const std::map<std::string, std::string> &searchConfig,
                                         std::vector<std::map<std::string, std::string>> &searchResult) const {
if (infoDict == "ScatterNdUpdate") {
    std::map<std::string, std::string> knowledge_map;
    knowledge_map.emplace("knowledge", "{\"dynamic_compile_static\":\"false\", \"op_impl_switch\":\"dsl\"}");
    searchResult.push_back(knowledge_map);
  }
  if (infoDict == "AA") {
    std::map<std::string, std::string> knowledge_map;
    knowledge_map.emplace("knowledge", "{\"input0\":\"NHWC\", \"output0\":\"NHWC\"}");
    searchResult.push_back(knowledge_map);
  }
  if (infoDict == "BB") {
    std::map<std::string, std::string> knowledge_map;
    knowledge_map.emplace("knowledge", "{\"input0\":\"NHWC\"}");
    searchResult.push_back(knowledge_map);
  }
  if (infoDict == "BBB") {
    std::map<std::string, std::string> knowledge_map;
    knowledge_map.emplace("knowledge", "");
    searchResult.push_back(knowledge_map);
  }
  if (infoDict == "BBBB") {
    std::map<std::string, std::string> knowledge_map;
    knowledge_map.emplace("knowledge", "{\"input0\":\"NCHW\", \"output0\":\"NCHW\"}");
    searchResult.push_back(knowledge_map);
  }
  if (infoDict == "BBBBB") {
    std::map<std::string, std::string> knowledge_map;
    knowledge_map.emplace("knowledge", "{\"input0\":\"NCHW\", \"output0\":\"NCHW\",(),[]}");
    searchResult.push_back(knowledge_map);
  }
  return CannKb::CANN_KB_STATUS::CANN_KB_SUCC;
}
}