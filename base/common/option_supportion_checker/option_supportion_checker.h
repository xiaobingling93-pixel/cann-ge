/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_OPTION_SUPPORTION_CHECKER_H_
#define GE_GRAPH_COMMON_OPTION_SUPPORTION_CHECKER_H_

#include <map>
#include <unordered_set>
#include "graph/types.h"

namespace ge {
  using Status = uint32_t;
  Status IrbuildCheckSupportedGlobalOptions(const std::map<std::string, std::string> &input_options);
  Status GEAPICheckSupportedGlobalOptions(const std::map<std::string, std::string> &input_options);
  Status GEAPICheckSupportedSessionOptions(const std::map<std::string, std::string> &input_options);
  Status GEAPICheckSupportedGraphOptions(const std::map<std::string, std::string> &input_options);
  const std::unordered_set<std::string> &GetAllGeOptionNames();
  Status CheckAllowParallelCompile(const std::map<std::string, std::string> &options);
} // namespace ge
#endif