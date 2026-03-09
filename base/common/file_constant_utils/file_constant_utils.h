/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTILS_H
#define INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTILS_H

#include <map>
#include <string>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "common/ge_common/ge_types.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "nlohmann/json.hpp"

namespace ge {
// external_weight option values
constexpr const char *kExternalWeightDisabled = "0";           // 禁用外置权重
constexpr const char *kExternalWeightEnabled = "1";            // 启用外置权重，每个权重单独导出
constexpr const char *kExternalWeightCombined = "2";           // 启用外置权重，所有权重合并导出到同一文件

using ConstNodeWeightHashMap = std::unordered_map<NodePtr, std::pair<GeTensorPtr, std::string>>;

struct FileIdToFilePath {
  std::string value_bin_file_id;
  std::string value_bin_file_path;
};

struct OptionInfo {
  std::vector<FileIdToFilePath> info;
};

void from_json(const nlohmann::json &j, FileIdToFilePath &info);
void from_json(const nlohmann::json &j, OptionInfo &option_info);

/* FileConstant算子的权重文件路径允许通过三种方式保存和获取:
 * 1.通过可选IR属性file_path直接设置或者获取外置权重文件的路径；
 * 2.通过可选IR属性file_id设置外置权重文件的唯一标识，并通过option ge.exec.value_bins设置file id到file path的映射；
 * 3.通过私有属性location获取外置权重路径，该属性存在于两种场景：
 *   ① parser模块解析onnx模型的外置权重时，权重路径会被写在节点的location属性上；
 *   ② 开启权重外置功能（ge.externalWeight）时，生成的外置权重文件的路径会被写在location属性上。
 */
class FileConstantUtils {
 public:
  /// @brief get file id to file path map from option ge.exec.value_bins
  /// @param [out] file_id_to_path_map
  /// @return Status
  static Status GetFileIdToPathMapFromOption(std::map<std::string, std::string> &file_id_to_path_map);
  static Status CopyOneWeightFromFileWithFilehandler(const void *const curr_dev_ptr, const std::string &file_path,
                                                     const size_t offset, const size_t file_constant_size,
                                                     size_t &left_size, std::ifstream &ifs);

  /// @brief load one weight from file to device memory
  /// @param [in] fileconstant memory addr on device
  /// @param [in] file path
  /// @param [in] offset
  /// @param [in] weight size
  /// @param [in] fileconstant memory size
  /// @return Status
  static Status CopyOneWeightFromFile(const void *const curr_dev_ptr, const std::string &file_path, const size_t offset,
                                      const size_t file_constant_size, size_t &left_size);

  /// @brief get weight file path
  /// @param [in] op_desc
  /// @param [in] file_id_to_path_map
  /// @param [out] file_path
  /// @param [out] offset
  /// @param [out] length
  /// @return Status
  static Status GetFilePath(const OpDescPtr &op_desc, const std::map<std::string, std::string> &file_id_to_path_map,
                            std::string &file_path, size_t &offset, size_t &length);

  /// @brief get fileconstant info
  /// @param [in] op_desc
  /// @return fileconstant info
  static FileConstantInfo GetFileConstantInfo(const OpDescPtr &op_desc);

  /// @brief get dir name to save external weight from om path
  /// @param [in] om_path
  /// @param [out] file_constant_weight_dir
  /// @return Status
  static Status GetExternalWeightDirFromOmPath(const std::string &om_path, string &file_constant_weight_dir);

  /// @brief get dir name to save external weight from model data
  /// @param [in] model_data
  /// @param [out] file_constant_weight_dir
  /// @return Status
  static Status GetExternalWeightDir(const ge::ModelData &model_data, string &file_constant_weight_dir);

  /// @brief set absolute file path for one fileconstant node
  /// @param [in] op_desc
  /// @param [in] weight_dir
  /// @return Status
  static Status SetExternalPath(const OpDescPtr &op_desc, const std::string &weight_dir);

  /// @brief set absolute file path for all fileconstant nodes in graph
  /// @param [in] compute_graph
  /// @param [in] weight_dir
  /// @return Status
  static Status SetExternalPath(const ComputeGraphPtr &compute_graph, const std::string &weight_dir);

  /// @brief load one weight from file to host memory
  /// @param [in] file_path
  /// @param [in] offset
  /// @param [in] file_length
  /// @param [in] bin_buff
  /// @return Status
  static Status ReadExternalWeightFromFile(const std::string &file_path, const size_t offset, const size_t file_length,
                                           char_t *const bin_buff);

  /// @brief convert all fileconstant nodes to const nodes in graph
  /// @param [in] compute_graph
  /// @return Status
  static Status ConvertFileConstToConst(const ComputeGraphPtr &compute_graph);

  /// @brief convert single fileconstant to const
  /// @param [in] fileconstant node
  /// @return Status
  static Status ConvertFileConstToConst(const NodePtr &node);

  /// @brief convert all const nodes to fileconstant nodes in graph
  /// @param [in] compute_graph
  /// @param [in] all_in_one
  /// @return Status
  static Status ConvertConstToFileConst(const ComputeGraphPtr &compute_graph, bool all_in_one = false);

  /// @brief convert single const to fileconstant
  /// @param [in] const node
  /// @return Status
  static Status ConvertConstToFileConst(const NodePtr &node);

  /// @brief get a map of graph, [key]:const node, [value]:a pair of weight and hash
  /// @param [in] compute_graph
  /// @return ConstNodeWeightHashMap
  static ConstNodeWeightHashMap GetAllConstNodesAndWeightHash(const ComputeGraphPtr &compute_graph);

  /// @brief move weight files from tmp_weight to om_path/weight
  /// @param [in] compute_graph
  /// @param [in] om_path
  /// @return Status
  static Status ChangeFilePath(const ComputeGraphPtr &compute_graph, const std::string &om_path);

  /// @brief get tmp weight dir
  /// @param [in] pid
  /// @param [in] session_id
  /// @return string
  static std::string GetTmpWeightDir(const int32_t pid, const uint64_t session_id);

  /// @brief set weight file path to attr location(private attribute)
  /// @param [in] op_desc
  /// @param [in] file_path
  /// @param [in] offset
  /// @param [in] length
  /// @return void
  static void SetFileConstantPath(const OpDescPtr &op_desc, const std::string &file_path, const int64_t offset = 0,
                                  const int64_t length = 0);

  /// @brief refresh relative weight file path for fileconstant
  /// @param [in] compute_graph
  /// @return Status
  static Status RefreshRelativePath(const ComputeGraphPtr &compute_graph);

 private:
  friend class ExternalWeightManager;

  /// @brief convert const to fileconstant
  /// @param [in] const_to_weight_hash_map
  /// @param [in] external_weight_dir
  /// @param [in] graph_name
  /// @param [in] meta
  /// @param [in] all_in_one
  /// @return Status
  static Status ConvertToFileConstants(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                       const std::string &weight_dir, FileConstantMeta &meta, const bool all_in_one=false);
  /// @brief save all const weight to file with multi threads
  /// @param [in] const_to_weight_hash_map
  /// @param [in] external_weight_dir
  /// @param [in] meta
  /// @return Status
  static Status SaveWeightToFileWithReuse(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                          const std::string &weight_dir, FileConstantMeta &meta);
  /// @brief save all const weight to one file
  /// @param [in] const_to_weight_hash_map
  /// @param [in] external_weight_dir
  /// @param [in] graph_name
  /// @param [in] meta
  /// @return Status
  static Status SaveWeightToOneFileWithReuse(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                          const std::string &weight_dir, FileConstantMeta &meta);
  /// @brief save weights and update metadata
  /// @param [in] const_to_weight_hash_map
  /// @param [in,out] meta
  /// @param [in,out] ofs
  /// @param [in] weight_path
  /// @param [in,out] offset
  /// @param [out] new_weights_written
  /// @param [out] reused_weights_count
  /// @return Status
  static Status SaveWeightsAndMetadata(const ConstNodeWeightHashMap &const_to_weight_hash_map,
                                       FileConstantMeta &meta, std::ofstream &ofs,
                                       const std::string &weight_path, size_t &offset,
                                       size_t &new_weights_written, size_t &reused_weights_count);

  /// @brief write weight data with padding to align 512 bytes
  /// @param [in,out] ofs
  /// @param [in] weight_data_ptr
  /// @param [in] size
  /// @param [in,out] offset
  /// @return Status
  static Status WriteWeightWithPadding(std::ofstream &ofs, const uint8_t *weight_data_ptr,
                                       const size_t size, size_t &offset);

  /// @brief change all fileconstant nodes attr location in graph
  /// @param [in] compute_graph
  /// @param [in] om_path
  /// @param [out] old_file_to_new_file
  /// @return Status
  static Status ChangeFilePathAttr(const ComputeGraphPtr &compute_graph, const std::string &om_path,
                                   std::map<std::string, std::string> &old_file_to_new_file);

  /// @brief move weight file from old path to new path
  /// @param [in] old_file_to_new_file
  /// @return Status
  static Status MoveFilePath(const std::map<std::string, std::string> &old_file_to_new_file);

  /// @brief get dir_name + file_name
  /// @param [in] dir_name
  /// @param [in] file_name
  /// @param [out] full_name
  /// @return void
  static void GetValidFullPath(const std::string &dir_name, const std::string &file_name, std::string &full_name);
};
}

#endif // INC_FRAMEWORK_COMMON_FILE_CONSTANT_UTILS_H
