/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GEN_ESB_OPTIONS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GEN_ESB_OPTIONS_H_

namespace ge {
namespace es {
struct GenEsbOptions {
  std::string mode;
  std::string output_dir;
  std::string module_name;
  std::string h_guard_prefix;
  std::string exclude_ops;
  std::string history_registry;
  std::string release_version;
  std::string release_date;
  std::string branch_name;
};
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GEN_ESB_OPTIONS_H_
