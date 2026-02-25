/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_WRITER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_WRITER_H_

#include <string>
#include <vector>

#include "graph/op_desc.h"

namespace ge {
namespace es {
namespace history {
class HistoryRegistryWriter {
 public:
  static void WriteRegistry(const std::string &output_dir, const std::string &release_version,
                            const std::string &release_date, const std::string &branch_name,
                            const std::vector<OpDescPtr> &all_ops);

 private:
  static void WriteIndexJson(const std::string &output_dir, const std::string &release_version,
                             const std::string &release_date);
  static void WriteMetadataJson(const std::string &version_dir, const std::string &release_version,
                                const std::string &branch_name);
  static void WriteOperatorsJson(const std::string &version_dir, const std::vector<OpDescPtr> &all_ops);
};
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_WRITER_H_
