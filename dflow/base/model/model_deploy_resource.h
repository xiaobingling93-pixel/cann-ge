/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_BASE_MODEL_MODEL_DEPLOY_RESOURCE_H_
#define DFLOW_BASE_MODEL_MODEL_DEPLOY_RESOURCE_H_
#include <string>
#include <vector>
#include <map>

namespace ge {
struct ModelDeployResource {
  std::string resource_type;
  bool is_heavy_load = false;
   // key is resource name, such as cpu_num shared_memory
  std::map<std::string, int64_t> resource_list;
};

struct ModelCompileResource {
  std::string host_resource_type;
  std::map<std::string, std::string> logic_dev_id_to_res_type;
  std::map<std::string, std::vector<std::pair<std::string, int64_t>>> dev_to_resource_list;
  bool IsEmpty() const {
    return host_resource_type.empty() &&
           logic_dev_id_to_res_type.empty() &&
           dev_to_resource_list.empty();
  }
};
}  // namespace ge

#endif  // DFLOW_BASE_MODEL_MODEL_DEPLOY_RESOURCE_H_
