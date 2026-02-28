/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_INTERFACE_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_INTERFACE_H_

#include <string>
#include <vector>

#include "history_registry_types.h"

namespace ge {
namespace es {
namespace history {
/**
 * 历史版本链
 * versions 与 proto_chain 一一对应，按 release_date 升序
 */
struct HistoryContext {
  std::vector<VersionMeta> versions;
  std::vector<IrOpProto> proto_chain;
};

/**
 * 加载历史窗口版本列表（不含具体算子原型）
 *
 * @param pkg_dir 历史原型库分包目录
 * @param baseline_version 基线版本，若为空则以当前日期为锚点
 * @param window_versions 输出窗口内版本，按 release_date 升序
 * @param error_msg 失败时返回错误信息
 * @return true 成功，false 失败
 */
bool LoadHistoryWindowVersions(const std::string &pkg_dir,
                               const std::string &baseline_version,
                               std::vector<VersionMeta> &window_versions,
                               std::string &error_msg);

/**
 * 加载历史原型链
 *
 * @param pkg_dir 历史原型库分包目录
 * @param window_versions 窗口内版本，按 release_date 升序
 * @param op_type 算子类型
 * @param warnings 记录跳过原因或校验失败原因
 * @return 通过校验的版本及对应 IrOpProto
 */
HistoryContext LoadHistoryChain(const std::string &pkg_dir,
                                const std::vector<VersionMeta> &window_versions,
                                const std::string &op_type,
                                std::vector<std::string> &warnings);

} // namespace history
} // namespace es
} // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_INTERFACE_H_
