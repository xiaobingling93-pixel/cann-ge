/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opt_info/ge_opt_info.h"

#include <string>
#include <map>
#include "graph/ge_local_context.h"
#include "ge/ge_api_types.h"
#include "common/debug/ge_log.h"
#include "common/checker.h"
#include "register/optimization_option_registry.h"
#include "common/option_supportion_checker/option_supportion_checker.h"

namespace ge {
Status GeOptInfo::SetOptInfo() {
  GE_ASSERT_GRAPH_SUCCESS(GetThreadLocalContext().GetOo().Initialize(
      GetThreadLocalContext().GetAllOptions(), OptionRegistry::GetInstance().GetRegisteredOptTable(), GetAllGeOptionNames()));

  std::string soc_ver;
  graphStatus ret = GetThreadLocalContext().GetOption(SOC_VERSION, soc_ver);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get soc version failed.");
    GELOGE(FAILED, "[Get][SocVersion]Get soc version failed.");
    return FAILED;
  }
  GELOGD("Soc version:%s.", soc_ver.c_str());
  // opt_module set all open default
  static std::map<std::string, std::string> opt_info = {{"opt_module.pass", "ALL"},
                                                        {"opt_module.fe", "ALL"},
                                                        {"opt_module.op_tune", "ALL"},
                                                        {"opt_module.rl_tune", "ALL"},
                                                        {"opt_module.aoe", "ALL"}};

  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  for (const auto &itr : opt_info) {
    graph_options.emplace(itr.first, itr.second);
    GELOGI("Get optional information success, key:%s, value:%s.", itr.first.c_str(), itr.second.c_str());
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
  return SUCCESS;
}
}  // namespace ge
