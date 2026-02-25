/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_READER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_READER_H_

#include <string>
#include <vector>

#include "history_registry_types.h"

namespace ge {
namespace es {
namespace history {
class HistoryRegistryReader {
 public:
  /**
   * 读取历史原型库版本列表
   * @param pkg_dir 历史原型库分包目录
   * @return 版本列表，按 release_date 升序
   */
  static std::vector<VersionMeta> LoadIndex(const std::string &pkg_dir);

  /**
   * 读取历史原型库版本元数据
   * @param pkg_dir 历史原型库分包目录
   * @param version 版本号
   * @return 元信息
   * @throws std::runtime_error 文件加载失败
   */
  static VersionMeta LoadMetadata(const std::string &pkg_dir, const std::string &version);

  /**
   * 从历史原型库提取单个算子原型
   * @param pkg_dir 历史原型库分包目录
   * @param version 版本号
   * @param op_type 算子类型
   * @return IrOpProto
   * @throws std::runtime_error 文件加载失败或 op_type 未找到、不唯一、解析失败
   */
  static IrOpProto LoadOpProto(const std::string &pkg_dir, const std::string &version,
                              const std::string &op_type);

  /**
   * 在版本列表中筛选窗口内版本
   * @param all_versions 全量版本，按 release_date 升序
   * @param current_version 当前版本号。为空时以当前日期为基准筛选历史窗口
   * @param window_days 窗口天数
   * @return 窗口内版本，按 release_date 升序
   */
  static std::vector<VersionMeta> SelectWindowVersions(const std::vector<VersionMeta> &all_versions,
                                                       const std::string &current_version,
                                                       int window_days = 365);
};
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_READER_H_
