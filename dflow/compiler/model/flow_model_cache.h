/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_MODEL_FLOW_MODEL_CACHE_H_
#define AIR_COMPILER_PNE_MODEL_FLOW_MODEL_CACHE_H_

#include <mutex>
#include "ge/ge_api_types.h"
#include "dflow/inc/data_flow/model/flow_model.h"
#include "nlohmann/json.hpp"

namespace ge {
struct CacheFileIndex {
  std::string graph_key;
  std::string cache_file_name;
};


struct CacheConfig {
  bool cache_manual_check = false;
  bool cache_debug_mode = false;
};

struct CacheCompileResult {
  std::map<std::string, std::string> compile_bin_info;  // key:runnable_res_type, value: compile_release_path;
  std::map<std::string, int64_t> running_resources_info;  // key:res_type, value: res_num;
};

class FlowModelCache {
 public:
   FlowModelCache() = default;
   ~FlowModelCache();
  /**
   * @brief Init submodel cache.
   * @return Status SUCCESS:init cache success.
   *                Other:failed.
   */
  Status InitSubmodelCache(const ComputeGraphPtr &root_graph,
                           const std::string &cache_dir,
                           const std::string &graph_key);

  /**
   * @brief Init root model cache.
   * @return Status SUCCESS:init cache success.
   *                Other:failed.
   */
  Status Init(const ComputeGraphPtr &root_graph);

  static std::string GetCacheDirFromContext();
  static std::string GetGraphKeyFromContext();

  /**
   * @brief try load flow model from cache.
   * if return success, but flow_model is null, means no cache.
   * @param root_graph root graph
   * @param flow_model loaded flow model.
   * @return Status SUCCESS:load cache success or no cache.
   *                Other:failed.
   */
  Status TryLoadFlowModelFromCache(const ComputeGraphPtr &root_graph, FlowModelPtr &flow_model);

  /**
   * @brief try cache flow model.
   * @param flow_model cache flow model.
   * @return Status SUCCESS:cache success or no need cache
   *                Other: failed.
   */
  Status TryCacheFlowModel(const FlowModelPtr &flow_model);

  static void TryRecordSuspendGraph(const std::string &suspend_graph_name);

  bool EnableCache() const {
    return cache_enable_;
  }

  bool ManualCheck() const {
    return cache_manual_check_;
  }

  bool DebugMode() const {
    return cache_debug_mode_;
  }

  bool TryLoadCompileResultFromCache(CacheCompileResult &cache_compile_result) const;

 private:
  static Status SaveFlowModelToCachedData(const FlowModelPtr &flow_model, const std::string &cache_file_name);
  static bool IsMatchFileName(const std::string &str);
  static std::string GenNormalizeName(const std::string &name);
  static bool CheckFileExist(const std::string &file_path);
  static Status CreateIndexFile(const std::string &index_file);
  Status ReadIndex(const std::string &index_file, std::vector<CacheFileIndex> &cache_file_list) const;
  Status InitCacheFileInfo();
  Status InitCacheFileByIdx(const std::string &cache_path);
  Status SaveCacheIndexFile() const;
  static Status GetRealFileName(std::string &file_name);
  void GenerateCacheFile();
  static std::string NormalizeDirPath(const std::string &dir_path);
  static Status CreateDir(const std::string &dir_path);
  static Status ReadJsonFile(const std::string &file_path, nlohmann::json &json_obj);
  static Status WriteJsonFile(const std::string &file_path, const nlohmann::json &json_obj);
  Status TryMatchCacheForSubGraph(bool &is_match) const;
  Status TryAddCacheForUdfGraph() const;
  Status TryMatchCacheForUdfSubGraph(bool &is_match) const;
  Status CheckCacheFile(bool &need_load);
  Status CacheBuildInfo() const;
  std::string GetSubmodelPneId() const;
  static Status ReadCacheConfig(const std::string &config_file, CacheConfig &cache_config);
  static Status FormatCacheCompilerResult(const ge::NamedAttrs &compile_results, CacheCompileResult &result);
  Status GetSplitOmDataBaseDir(std::string &split_om_data_base_dir) const;
  static Status UpdateFlowModelCache(const std::set<PneModelPtr> &refreshed_models);
  bool cache_enable_ = false;
  std::string cache_dir_;
  CacheFileIndex cache_index_;
  std::string cache_file_prefix_;
  std::string cache_file_time_str_;
  std::string index_file_;
  std::vector<CacheFileIndex> matched_file_list_;
  uint64_t session_id_ = UINT64_MAX;
  uint64_t graph_id_ = UINT32_MAX;
  int32_t lock_file_fd_ = -1;

  bool is_subgraph_cache_ = false;
  std::string build_info_path_;
  ComputeGraphPtr root_graph_ = nullptr;
  bool cache_debug_mode_ = false;
  bool cache_manual_check_ = false;
  static std::mutex cache_map_mutex_;
  static std::map<std::string, std::string> suspend_graph_name_to_cache_dir_;
  static std::map<std::string, std::string> suspend_graph_name_to_graph_key_;
};
void from_json(const nlohmann::json &json_obj, CacheFileIndex &cache_file_index);
void to_json(nlohmann::json &json_obj, const CacheFileIndex &cache_file_index);
}  // namespace ge

#endif  // AIR_COMPILER_PNE_MODEL_FLOW_MODEL_CACHE_H_
