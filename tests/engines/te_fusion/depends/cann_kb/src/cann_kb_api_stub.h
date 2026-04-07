/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_CANN_KB_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_CANN_KB_UTILS_H_
#include<map>
#include "common/plugin_manager.h"
 
namespace fe {
namespace CannKb {
enum class CANN_KB_STATUS : int {              /* < CANN_KB Status Define > */
    CANN_KB_SUCC = 0,                          // CANN_KB SUCCESS
    CANN_KB_FAILED,                            // CANN_KB FAILED
    CANN_KB_CHECK_FAILED,                      // CANN_KB check failed
    CANN_KB_INIT_ERR,                          // CANN_KB init ERROR
    CANN_KB_GET_PARAM_ERR,                        // CANN_KB get param ERROR
    CANN_KB_PY_NULL,                     // CANN_KB python return null
    CANN_KB_PY_FAILED,                         // CANN_KB python return failed
};
}
 
using CannKbInitFunc = std::function<CannKb::CANN_KB_STATUS(const std::map<std::string, std::string> &,const std::map<std::string, std::string> &)>;
 
using CannKbFinalizeFunc = std::function<CannKb::CANN_KB_STATUS()>;

using CannKbSearchFunc = std::function<CannKb::CANN_KB_STATUS(const std::string &,
    const std::map<std::string, std::string> &, std::vector<std::map<std::string, std::string>> &)>;
 
class CannKBUtils {
 public:
  CannKBUtils();
  ~CannKBUtils();
  static CannKBUtils &Instance();
  bool InitCannKb();
 
  CannKb::CANN_KB_STATUS CannKbInit(const std::map<std::string, std::string> &sysConfig,
                                    const std::map<std::string, std::string> &loadConfig);
 
  CannKb::CANN_KB_STATUS CannKbFinalize();

  CannKb::CANN_KB_STATUS RunCannKbSearch(const std::string &infoDict,
                                         const std::map<std::string, std::string> &searchConfig,
                                         std::vector<std::map<std::string, std::string>> &searchResult) const;
 
 private:
  PluginManagerPtr cann_kb_plugin_manager_;
  CannKbInitFunc cann_kb_init_func_;
  CannKbFinalizeFunc cann_kb_finalize_func_;
  CannKbSearchFunc cann_kb_search_func_;
  bool init_flag_;
};
}  // namespace fe
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_CANN_KB_UTILS_H_
