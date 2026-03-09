/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/model/flow_model_cache.h"
#include <string>
#include <regex>
#include <fstream>
#include <iomanip>
#include <sys/file.h>
#include "mmpa/mmpa_api.h"
#include "common/debug/log.h"
#include "common/proto_util/proto_util.h"
#include "common/helper/file_saver.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "dflow/base/model/flow_model_om_loader.h"
#include "dflow/base/model/flow_model_om_saver.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/thread_pool/thread_pool.h"

namespace ge {
namespace {
constexpr const char *kFileNamePattern = R"(^[A-Za-z0-9_\-]{1,128}$)";
constexpr const char *kJsonFieldCacheFileList = "cache_file_list";
constexpr const char *kJsonFieldCacheFileName = "cache_file_name";
constexpr const char *kJsonFieldGraphKey = "graph_key";
constexpr const char *kJsonFieldCacheManualCheck = "cache_manual_check";
constexpr const char *kJsonFieldCacheDebugMode = "cache_debug_mode";
constexpr int32_t kIndentationLen = 4;
constexpr const char *kFinalWeightDirName = "/weight/";
constexpr const char *kLockFileName = ".lock";
constexpr const char *kSuspendGraphOriginalName = "_suspend_graph_original_name";
constexpr const char *kAttrNameDataFlowCompilerResult = "_dflow_compiler_result";
constexpr const char *kAttrNameDataFlowRunningResourceInfo = "_dflow_running_resource_info";
constexpr const char *kAttrNameDataFlowRunnableResource = "_dflow_runnable_resource";
constexpr size_t kMaxUpdateCacheThreadPoolSize = 8U;

Status TryLockFile(const std::string &lock_file, int32_t &fd) {
  fd = mmOpen2(lock_file.c_str(), M_CREAT | M_WRONLY, M_IRUSR | M_IWUSR);
  if (fd == -1) {
    const int32_t error_code = mmGetErrorCode();
    GELOGE(FAILED, "open file[%s] failed, error=%d, error msg=%s.", lock_file.c_str(), error_code,
           GetErrorNumStr(error_code).c_str());
    return FAILED;
  }
  if (flock(fd, LOCK_EX) != 0) {
    const int32_t error_code = mmGetErrorCode();
    GELOGE(FAILED, "Failed to lock file[%s], error msg[%s].", lock_file.c_str(), GetErrorNumStr(error_code).c_str());
    (void)close(fd);
    fd = -1;
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace

std::map<std::string, std::string> FlowModelCache::suspend_graph_name_to_cache_dir_;
std::map<std::string, std::string> FlowModelCache::suspend_graph_name_to_graph_key_;
std::mutex FlowModelCache::cache_map_mutex_;

FlowModelCache::~FlowModelCache() {
  if (lock_file_fd_ != -1) {
    (void)flock(lock_file_fd_, LOCK_UN);
    (void)close(lock_file_fd_);
    lock_file_fd_ = -1;
  }
}

Status FlowModelCache::InitSubmodelCache(const ComputeGraphPtr &root_graph,
                                         const std::string &cache_dir,
                                         const std::string &graph_key) {
  cache_enable_ = false;
  GE_CHECK_NOTNULL(root_graph);
  if (root_graph->TryGetExtAttr<bool>("_is_built_in_for_data_flow", false)) {
    GELOGI("The graph[%s] is no need cache.", root_graph->GetName().c_str());
    return SUCCESS;
  }
  cache_dir_ = NormalizeDirPath(cache_dir);
  cache_index_.graph_key = graph_key;
  if (cache_dir_.empty() || cache_index_.graph_key.empty()) {
    GELOGI("Cache is disabled, no need load cache.");
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(CheckFileExist(cache_dir_), PARAM_INVALID,
                         "Init cache failed, as cache dir[%s] does not exist.", cache_dir_.c_str());
  GE_CHK_BOOL_RET_STATUS(IsMatchFileName(cache_index_.graph_key), FAILED,
                         "The graph_key[%s] can't be used as file name, should match pattern[%s].",
                         cache_index_.graph_key.c_str(), kFileNamePattern);
  auto cache_config_file = cache_dir_ + "cache.conf";
  CacheConfig cache_config;
  GE_CHK_STATUS_RET(ReadCacheConfig(cache_config_file, cache_config), "Failed to read cache config file.");
  cache_manual_check_ = cache_config.cache_manual_check;

  std::string sub_cache_dir = cache_dir_ + cache_index_.graph_key;
  GE_CHK_STATUS_RET(CreateDir(sub_cache_dir), "Failed to create dir[%s].", sub_cache_dir.c_str());
  root_graph_ = root_graph;
  session_id_ = root_graph->GetSessionID();
  graph_id_ = root_graph->GetGraphID();
  std::string graph_name = GenNormalizeName(root_graph->GetName());
  sub_cache_dir = sub_cache_dir + "/" + graph_name + "/";
  GE_CHK_STATUS_RET(CreateDir(sub_cache_dir), "Failed to create dir[%s].", sub_cache_dir.c_str());
  cache_index_.cache_file_name = sub_cache_dir + graph_name + "_root.om";
  build_info_path_ = sub_cache_dir + "buildinfo";
  cache_enable_ = true;
  is_subgraph_cache_ = true;
  GELOGI("Cache init end, cache_dir=%s, graph_key=%s, cache_file=%s, cache manual check=%d.",
         sub_cache_dir.c_str(), cache_index_.graph_key.c_str(),
         cache_index_.cache_file_name.c_str(), static_cast<int32_t>(cache_manual_check_));
  return SUCCESS;
}

Status FlowModelCache::Init(const ComputeGraphPtr &root_graph) {
  cache_enable_ = false;
  GE_CHECK_NOTNULL(root_graph);
  std::string original_graph_name;
  if (AttrUtils::GetStr(root_graph, kSuspendGraphOriginalName, original_graph_name) &&
      (!original_graph_name.empty())) {
    // init graph will be cache during other graph caching. Get graph key and cache dir by static map
    const std::lock_guard<std::mutex> lock(cache_map_mutex_);
    const auto iter_dir = suspend_graph_name_to_cache_dir_.find(original_graph_name);
    if (iter_dir != suspend_graph_name_to_cache_dir_.cend()) {
      cache_dir_ = NormalizeDirPath(iter_dir->second);
      suspend_graph_name_to_cache_dir_.erase(iter_dir);
    }
    const auto iter_key = suspend_graph_name_to_graph_key_.find(original_graph_name);
    if (iter_key != suspend_graph_name_to_graph_key_.cend()) {
      cache_index_.graph_key = iter_key->second;
      suspend_graph_name_to_graph_key_.erase(iter_key);
    }
    GELOGD("Current graph %s with original name %s is suspended.",
           root_graph->GetName().c_str(), original_graph_name.c_str());
  } else {
    cache_dir_ = NormalizeDirPath(GetCacheDirFromContext());
    cache_index_.graph_key = GetGraphKeyFromContext();
  }

  if (cache_dir_.empty() || cache_index_.graph_key.empty()) {
    GELOGI("Cache is disable due to cache dir[%s] or graph key[%s] is empty. No need to load cache.",
           cache_dir_.c_str(), cache_index_.graph_key.c_str());
    return SUCCESS;
  }
  GELOGI("Cache is enable, cache_dir=%s, graph_key=%s.", cache_dir_.c_str(), cache_index_.graph_key.c_str());
  if (!CheckFileExist(cache_dir_)) {
    REPORT_PREDEFINED_ERR_MSG("E13026", std::vector<const char_t *>({"pathname", "reason"}),
                       std::vector<const char_t *>({cache_dir_.c_str(), "The cache directory does not exist."}));
    GELOGE(PARAM_INVALID, "Init cache failed, as cache dir[%s] does not exist.", cache_dir_.c_str());
    return PARAM_INVALID;
  }
  auto cache_config_file = cache_dir_ + "cache.conf";
  CacheConfig cache_config;
  GE_CHK_STATUS_RET(ReadCacheConfig(cache_config_file, cache_config), "Failed to read cache config file.");
  cache_debug_mode_ = cache_config.cache_debug_mode;
  cache_manual_check_ = cache_config.cache_manual_check;
  // check graph key
  if (IsMatchFileName(cache_index_.graph_key)) {
    cache_file_prefix_ = cache_index_.graph_key;
    index_file_ = cache_dir_ + cache_file_prefix_ + ".idx";
    cache_file_time_str_ = CurrentTimeInStr();
  } else {
    GELOGE(PARAM_INVALID, "The graph_key[%s] can't be used as file name, should match pattern[%s].",
           cache_index_.graph_key.c_str(), kFileNamePattern);
    return PARAM_INVALID;
  }
  const std::string lock_file = cache_dir_ + cache_index_.graph_key + kLockFileName;
  GE_CHK_STATUS_RET(TryLockFile(lock_file, lock_file_fd_), "Try lock cache dir by locking file failed.");
  GE_CHK_STATUS_RET(InitCacheFileInfo(), "Fail to init cache file info, cache_dir=%s, graph_key=%s.",
                    cache_dir_.c_str(), cache_index_.graph_key.c_str());
  session_id_ = root_graph->GetSessionID();
  graph_id_ = root_graph->GetGraphID();
  cache_enable_ = true;
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  GE_CHECK_NOTNULL(external_weight_manager);
  external_weight_manager->SetWeightPath(cache_dir_ + kFinalWeightDirName);
  GELOGI("Init end, cache_dir=%s, graph_key=%s, cache_file=%s, index_file=%s, cache manual check = %d.",
         cache_dir_.c_str(), cache_index_.graph_key.c_str(), cache_index_.cache_file_name.c_str(),
         index_file_.c_str(), static_cast<int32_t>(cache_manual_check_));
  return SUCCESS;
}

std::string FlowModelCache::NormalizeDirPath(const std::string &dir_path) {
  std::string normalize_path = dir_path;
  if (!dir_path.empty() && (dir_path.at(dir_path.length() - 1) != '/')) {
    (void)normalize_path.append("/");
  }
  return normalize_path;
}

Status FlowModelCache::InitCacheFileInfo() {
  // if index does not exist, try match om name.
  if (!CheckFileExist(index_file_)) {
    const std::string om_name = cache_dir_ + cache_file_prefix_ + ".om";
    if (CheckFileExist(om_name)) {
      cache_index_.cache_file_name = om_name;
      matched_file_list_.emplace_back(cache_index_);
      GELOGI("No index file[%s] found in cache dir[%s], matched om[%s] directly.",
             index_file_.c_str(), cache_dir_.c_str(), om_name.c_str());
      return SUCCESS;
    }
  }
  return InitCacheFileByIdx(cache_dir_);
}

Status FlowModelCache::ReadIndex(const std::string &index_file,
                                 std::vector<CacheFileIndex> &cache_file_list) const {
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ReadJsonFile(index_file, json_obj), "Failed to read cache index file[%s].", index_file.c_str());
  try {
    cache_file_list = json_obj[kJsonFieldCacheFileList].get<std::vector<CacheFileIndex>>();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read cache index file[%s], err msg: %s", index_file.c_str(), e.what());
    return FAILED;
  }
  for (auto &idx : cache_file_list) {
    if (idx.cache_file_name.find("/") == std::string::npos) {
      idx.cache_file_name = cache_dir_ + idx.cache_file_name;
      GELOGD("Add cache dir to cache file name[%s].", idx.cache_file_name.c_str());
    }
  }
  return SUCCESS;
}

void FlowModelCache::GenerateCacheFile() {
  cache_index_.cache_file_name = cache_dir_ + cache_file_prefix_ + "_" + cache_file_time_str_ + ".om";
}

Status FlowModelCache::InitCacheFileByIdx(const std::string &cache_path) {
  // if index is exist, gen file name.
  if (CheckFileExist(index_file_)) {
    std::vector<CacheFileIndex> cache_file_list;
    GE_CHK_STATUS_RET(ReadIndex(index_file_, cache_file_list),
                      "Failed to read cache index list from file:%s", index_file_.c_str());
    for (const auto &idx : cache_file_list) {
      if (idx.graph_key == cache_index_.graph_key) {
        GE_CHK_BOOL_RET_STATUS(CheckFileExist(idx.cache_file_name), FAILED,
                               "cache file[%s] in cache index file[%s] does not exist.",
                               idx.cache_file_name.c_str(), index_file_.c_str());
        GELOGI("Matched graph_key[%s] success, cache om file = %s, cache dir = %s.",
               cache_index_.graph_key.c_str(), idx.cache_file_name.c_str(),
               cache_path.c_str());
        matched_file_list_.emplace_back(idx);
      }
    }
  }

  if (matched_file_list_.empty()) {
    // not found, need generate a new file.
    GenerateCacheFile();
    GELOGI("No graph_key[%s] found in cache index file[%s], generate cache om file[%s]",
           cache_index_.graph_key.c_str(), index_file_.c_str(),
           cache_index_.cache_file_name.c_str());
  }
  return SUCCESS;
}

void FlowModelCache::TryRecordSuspendGraph(const std::string &suspend_graph_name) {
  const std::string cache_dir = GetCacheDirFromContext();
  const std::string graph_key = GetGraphKeyFromContext();
  if (cache_dir.empty() || graph_key.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(cache_map_mutex_);
  suspend_graph_name_to_cache_dir_[suspend_graph_name] = cache_dir;
  suspend_graph_name_to_graph_key_[suspend_graph_name] = graph_key;
}

bool FlowModelCache::TryLoadCompileResultFromCache(CacheCompileResult &cache_compile_result) const {
  auto buildinfo_exist = CheckFileExist(build_info_path_);
  if (!cache_enable_ || !buildinfo_exist) {
    GELOGI("Can not load compile result from cache, enable cache = %d, buildinfo exist = %d.",
           static_cast<int32_t>(cache_enable_), static_cast<int32_t>(buildinfo_exist));
    return false;
  }
  GE_TRACE_START(LoadCompileResult);
  bool match_cache = false;
  nlohmann::json json_obj;
  auto ret = ReadJsonFile(build_info_path_, json_obj);
  if (ret == SUCCESS) {
    try {
      auto iter = json_obj.find("bin_info");
      if (iter != json_obj.end()) {
        cache_compile_result.compile_bin_info = iter.value().get<std::map<std::string, std::string>>();
        match_cache = true;
      }
      iter = json_obj.find("resource_info");
      if (iter != json_obj.end()) {
        cache_compile_result.running_resources_info = iter.value().get<std::map<std::string, int64_t>>();
        match_cache = true;
      }
    } catch (const nlohmann::json::exception &e) {
      GELOGE(FAILED, "Failed to read cache info from json file[%s], err msg[%s].", build_info_path_.c_str(), e.what());
      match_cache = false;
    }
    GE_COMPILE_TRACE_TIMESTAMP_END(LoadCompileResult, "loading compile result cache");
  }
  return match_cache;
}


Status FlowModelCache::UpdateFlowModelCache(const std::set<PneModelPtr> &refreshed_models) {
  if (refreshed_models.empty()) {
    return SUCCESS;
  }
  GE_TRACE_START(UpdateFlowModelCache);
  size_t pool_size = refreshed_models.size() > kMaxUpdateCacheThreadPoolSize ?
                                               kMaxUpdateCacheThreadPoolSize :
                                               refreshed_models.size();
  ThreadPool pool("ge_upd_cch", static_cast<uint32_t>(pool_size), true);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &model : refreshed_models) {
    if (model->GetSavedModelPath().empty()) {
      continue;
    }
    auto fut = pool.commit([model]() -> Status {
      ModelBufferData serialize_buff{};
      GE_CHK_STATUS_RET(model->SerializeModel(serialize_buff),
                        "Failed to serialize model, model_name = %s", model->GetModelName().c_str());
      const auto &saved_model_path = model->GetSavedModelPath();
      GE_ASSERT_GRAPH_SUCCESS(SaveBinToFile(reinterpret_cast<char_t *>(serialize_buff.data.get()),
                                            serialize_buff.length, saved_model_path),
                              "Failed to save model data to file %s.", saved_model_path.c_str());
      GEEVENT("Update model cache success, model_name = %s, path = %s",
              model->GetModelName().c_str(), saved_model_path.c_str());
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }

  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to update model cache");
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(UpdateFlowModelCache, "Update flow model cache cost");
  return SUCCESS;
}

Status FlowModelCache::TryLoadFlowModelFromCache(const ComputeGraphPtr &root_graph, FlowModelPtr &flow_model) {
  if (!cache_enable_) {
    GELOGD("cache is disable, no need load cache.");
    return SUCCESS;
  }

  bool is_need_load = true;
  GE_CHK_STATUS_RET(CheckCacheFile(is_need_load), "Check file can be loaded failed.");
  if (!is_need_load) {
    GELOGD("Cache not matched or submodel cache is udf.");
    return SUCCESS;
  }
  flow_model = nullptr;
  GE_TRACE_START(LoadFlowModel);
  std::string split_om_data_base_dir;
  GE_CHK_STATUS_RET(GetSplitOmDataBaseDir(split_om_data_base_dir), "Get split om data file dir failed.");
  GE_CHK_STATUS_RET(FlowModelHelper::LoadToFlowModel(cache_index_.cache_file_name, flow_model, split_om_data_base_dir),
                    "Failed to load flow model, cache_file:%s", cache_index_.cache_file_name.c_str());

  GE_CHK_STATUS_RET(
      FlowModelOmLoader::RefreshModel(flow_model, cache_dir_ + "/weight/", session_id_, graph_id_),
      "Failed to assign constant mem for cache model, cache_file:%s", cache_index_.cache_file_name.c_str());
  std::string session_graph_id;
  if (AttrUtils::GetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GE_CHK_STATUS_RET(FlowModelHelper::UpdateSessionGraphId(flow_model, session_graph_id),
                      "Failed to update flow model session graph id, session_graph_id:%s", session_graph_id.c_str());
  }

  const std::string trace_log = "loading flow model cache by key[" + cache_index_.graph_key + "]";
  GE_COMPILE_TRACE_TIMESTAMP_END(LoadFlowModel, trace_log.c_str());
  GELOGI("load flow model from cache file:%s success.", cache_index_.cache_file_name.c_str());
  return SUCCESS;
}

Status FlowModelCache::CheckCacheFile(bool &need_load) {
  need_load = false;
  if (is_subgraph_cache_) {
    bool is_matched = false;
    GE_CHK_STATUS_RET(TryMatchCacheForSubGraph(is_matched), "Failed to match graph[%s].",
                      root_graph_->GetName().c_str());
    if (!is_matched) {
      if (GetSubmodelPneId() == PNE_ID_UDF) {
        return TryAddCacheForUdfGraph();
      }
      return SUCCESS;
    }
    // for udf submodel, om is tar package which should not be deserialized into memory
    if (GetSubmodelPneId() == PNE_ID_UDF) {
        return SUCCESS;
    }
  } else {
      if (matched_file_list_.empty()) {
        GELOGD("cache file does not exist, no need load cache.");
        return SUCCESS;
      } else {
        cache_index_ = matched_file_list_[0];
      }
  }
  need_load = true;
  return SUCCESS;
}

Status FlowModelCache::TryCacheFlowModel(const FlowModelPtr &flow_model) {
  if (!cache_enable_ || cache_debug_mode_) {
    GELOGD("cache is disable or in debug mode, no need cache.");
    return SUCCESS;
  }
  if (!is_subgraph_cache_) {
    GenerateCacheFile();
  } else {
    if (GetSubmodelPneId() == PNE_ID_UDF && !cache_manual_check_) {
      GELOGI("The udf no need to cache om.");
      GE_CHK_STATUS_RET(CacheBuildInfo(), "Failed to cache build info for graph[%s].", root_graph_->GetName().c_str());
      return SUCCESS;
    }
    const std::string offline_session_graph_id = "-1_" + std::to_string(graph_id_);
    GE_CHK_STATUS_RET(FlowModelHelper::UpdateSessionGraphId(flow_model, offline_session_graph_id),
                      "Failed to update flow model session graph id, offline_session_graph_id:%s",
                      offline_session_graph_id.c_str());
  }

  if ((flow_model->GetModelRelation() == nullptr)) {
    // cache by old om.
    auto ret = SaveFlowModelToCachedData(flow_model, cache_index_.cache_file_name);
    GE_CHK_STATUS_RET(ret, "Failed to cache no model relation flow model, cache_file:%s",
                      cache_index_.cache_file_name.c_str());
  } else {
    std::string split_om_dir = cache_dir_ + cache_index_.graph_key;
    if (!CheckFileExist(split_om_dir)) {
      GE_CHK_STATUS_RET(CreateDir(split_om_dir), "Failed to create dir[%s].", split_om_dir.c_str());
    }
    std::string split_om_data_base_dir;
    GE_CHK_STATUS_RET(GetSplitOmDataBaseDir(split_om_data_base_dir), "Failed to get split om data dir.");
    FlowModelOmSaver om_saver(flow_model);
    auto ret = om_saver.SaveToOm(cache_index_.cache_file_name, split_om_data_base_dir);
    GE_CHK_STATUS_RET(ret, "Failed to cache flow model, cache_file:%s", cache_index_.cache_file_name.c_str());
  }
  GELOGI("cache flow model success, cache file:%s", cache_index_.cache_file_name.c_str());

  if (!is_subgraph_cache_) {
    GE_CHK_STATUS_RET(SaveCacheIndexFile(), "Failed to save cache file index, cache_file:%s, cache file index:%s",
                      cache_index_.cache_file_name.c_str(), index_file_.c_str());
  }
  if (is_subgraph_cache_ &&
      ((GetSubmodelPneId() != PNE_ID_UDF) || (GetSubmodelPneId() == PNE_ID_UDF && cache_manual_check_))) {
    GE_CHK_STATUS_RET(CacheBuildInfo(), "Failed to cache build info for graph[%s].", root_graph_->GetName().c_str());
  }
  return SUCCESS;
}

Status FlowModelCache::GetSplitOmDataBaseDir(std::string &split_om_data_base_dir) const {
  std::string om_data_base_dir = cache_dir_ + cache_index_.graph_key;
  split_om_data_base_dir = RealPath(om_data_base_dir.c_str());
  if (!split_om_data_base_dir.empty()) {
    split_om_data_base_dir += "/";
    GELOGD("Get split om data file base path[%s].", split_om_data_base_dir.c_str());
  } else {
    GELOGI("There is no om data file path result of current model may not flow model.");
  }
  return SUCCESS;
}

std::string FlowModelCache::GetCacheDirFromContext() {
  std::string cache_dir;
  graphStatus ret = GetThreadLocalContext().GetOption(OPTION_GRAPH_COMPILER_CACHE_DIR, cache_dir);
  if (ret != GRAPH_SUCCESS) {
    GELOGD("option[%s] does not exist, build cache is disable.", OPTION_GRAPH_COMPILER_CACHE_DIR);
    return cache_dir;
  }
  GELOGD("compile cache dir is %s.", cache_dir.c_str());
  return cache_dir;
}

std::string FlowModelCache::GetGraphKeyFromContext() {
  std::string graph_key;
  graphStatus ret = GetThreadLocalContext().GetOption(OPTION_GRAPH_KEY, graph_key);
  if (ret != GRAPH_SUCCESS) {
    GELOGD("option[%s] does not exist, build cache is disable.", OPTION_GRAPH_KEY);
    return graph_key;
  }
  // just graph unique mark, log is ok.
  GELOGD("graph key is %s.", graph_key.c_str());
  return graph_key;
}

Status FlowModelCache::ReadCacheConfig(const std::string &config_file, CacheConfig &cache_config) {
  auto real_path = RealPath(config_file.c_str());
  if (real_path.empty()) {
    return SUCCESS;
  }
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ReadJsonFile(config_file, json_obj), "Failed to read cache config file[%s].", config_file.c_str());
  try {
    auto iter = json_obj.find(kJsonFieldCacheManualCheck);
    if (iter != json_obj.end()) {
      cache_config.cache_manual_check = iter.value().get<bool>();
    }
    iter = json_obj.find(kJsonFieldCacheDebugMode);
    if (iter != json_obj.end()) {
      cache_config.cache_debug_mode = iter.value().get<bool>();
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read cache config file[%s], err msg: %s", config_file.c_str(), e.what());
    return FAILED;
  }
  GELOGI("Read cache config file success, cache manual check = %d, cache debug mode = %d.",
         static_cast<int32_t>(cache_config.cache_manual_check), static_cast<int32_t>(cache_config.cache_debug_mode));
  return SUCCESS;
}

Status FlowModelCache::SaveFlowModelToCachedData(const FlowModelPtr &flow_model, const std::string &cache_file_name) {
  const auto &submodels = flow_model->GetSubmodels();
  if (submodels.size() != 1U) {
    GELOGE(FAILED, "save ge root model must be only one submodel, but size=%zu.", submodels.size());
    return FAILED;
  }
  const auto &submodel = submodels.cbegin()->second;
  ModelBufferData model_buff;
  auto ret = submodel->SerializeModel(model_buff);
  GE_CHK_STATUS_RET(ret, "Failed to serialize model, model_name=%s, model_type=%s", submodel->GetModelName().c_str(),
                    submodel->GetModelType().c_str());
  ret = FileSaver::SaveToFile(cache_file_name, model_buff.data.get(), model_buff.length);
  GE_CHK_STATUS_RET(ret, "Failed to save model, model_name=%s, model_type=%s, file_name=%s, length=%lu.",
                    submodel->GetModelName().c_str(), submodel->GetModelType().c_str(), cache_file_name.c_str(),
                    model_buff.length);
  GELOGD("save to ge root model success, model_name=%s, file_name=%s.", submodel->GetModelName().c_str(),
         cache_file_name.c_str());
  return SUCCESS;
}

bool FlowModelCache::IsMatchFileName(const std::string &str) {
  const std::regex file_name_regex(kFileNamePattern);
  return std::regex_match(str, file_name_regex);
}

std::string FlowModelCache::GenNormalizeName(const std::string &name) {
  std::stringstream ss;
  for (const char &element : name) {
    if ((!isalpha(element)) && (!isdigit(element)) && (element != '_') && (element != '-')) {
      ss << std::hex << static_cast<uint32_t>(element);
    } else {
      ss << element;
    }
  }
  return ss.str();
}

bool FlowModelCache::CheckFileExist(const std::string &file_path) {
  return mmAccess(file_path.c_str()) == EN_OK;
}

Status FlowModelCache::GetRealFileName(std::string &file_name) const {
  file_name = file_name.substr(file_name.rfind("/") + 1UL, file_name.length());
  GE_ASSERT_TRUE(!file_name.empty());
  GELOGD("Get real file name[%s].", file_name.c_str());
  return SUCCESS;
}

Status FlowModelCache::SaveCacheIndexFile() const {
  if (index_file_.empty()) {
    GELOGD("hash index file is empty, no need save cache index file.");
    return SUCCESS;
  }
  std::vector<CacheFileIndex> cache_file_list;
  bool file_exist = CheckFileExist(index_file_);
  if (file_exist) {
    GE_CHK_STATUS_RET(ReadIndex(index_file_, cache_file_list),
                      "Failed to read cache index list from file:%s", index_file_.c_str());
  }
  CacheFileIndex new_cache_index = cache_index_;
  GE_CHK_STATUS_RET(GetRealFileName(new_cache_index.cache_file_name), "Get real cache file name failed by[%s].",
                    cache_index_.cache_file_name.c_str());
  cache_file_list.emplace_back(new_cache_index);
  nlohmann::json json_obj;
  try {
    json_obj[kJsonFieldCacheFileList] = cache_file_list;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to set cache file to json, err msg: %s", e.what());
    return FAILED;
  }

  if (!file_exist) {
    GE_CHK_STATUS_RET(CreateIndexFile(index_file_), "Failed to create index file:%s",
                      index_file_.c_str());
  }
  GE_CHK_STATUS_RET(WriteJsonFile(index_file_, json_obj), "Failed to write cache index file[%s].", index_file_.c_str());
  GELOGI("save cache file index success, index file:%s", index_file_.c_str());
  return SUCCESS;
}

Status FlowModelCache::CreateIndexFile(const std::string &index_file) {
  auto open_mode = static_cast<mmMode_t>(M_IRUSR | M_IWUSR);
  auto open_flag = M_RDWR | M_CREAT;
  int32_t fd = mmOpen2(index_file.c_str(), open_flag, open_mode);
  GE_CHK_BOOL_RET_STATUS(((fd != EN_ERROR) && (fd != EN_INVALID_PARAM)), FAILED,
                         "Create index file[%s] failed, fd=%d.", index_file.c_str(), fd);
  (void)mmClose(fd);
  return SUCCESS;
}

void from_json(const nlohmann::json &json_obj, CacheFileIndex &cache_file_index) {
  json_obj.at(kJsonFieldGraphKey).get_to(cache_file_index.graph_key);
  json_obj.at(kJsonFieldCacheFileName).get_to(cache_file_index.cache_file_name);
}

void to_json(nlohmann::json &json_obj, const CacheFileIndex &cache_file_index) {
  json_obj = nlohmann::json{{kJsonFieldGraphKey, cache_file_index.graph_key},
                            {kJsonFieldCacheFileName, cache_file_index.cache_file_name}};
}

Status FlowModelCache::CreateDir(const std::string &dir_path) {
  static std::mutex mu;
  const std::lock_guard<std::mutex> lock(mu);
  if (CheckFileExist(dir_path)) {
    GELOGD("The dir[%s] already exists, not need create.", dir_path.c_str());
    return SUCCESS;
  }
  // 700
  constexpr auto mkdir_mode = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR) |
                                                    static_cast<uint32_t>(M_IXUSR));
  GE_CHK_BOOL_RET_STATUS(mmMkdir(dir_path.c_str(), mkdir_mode) == 0, FAILED, "Failed to mkdir[%s]", dir_path.c_str());
  return SUCCESS;
}

Status FlowModelCache::ReadJsonFile(const std::string &file_path, nlohmann::json &json_obj) {
  std::ifstream file_stream(file_path);
  GE_CHK_BOOL_RET_STATUS(file_stream.is_open(), FAILED, "Failed to open json file[%s]", file_path.c_str());
  try {
    file_stream >> json_obj;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read json file[%s], err msg: %s", file_path.c_str(), e.what());
    return FAILED;
  }
  // check whether write file error.
  GE_CHK_BOOL_RET_STATUS(file_stream.good(), FAILED, "Failed to read json file[%s], error msg = %s", file_path.c_str(),
                         strerror(errno));
  GELOGD("Read json file[%s] success, content is:%s", file_path.c_str(), json_obj.dump().c_str());
  return SUCCESS;
}

Status FlowModelCache::WriteJsonFile(const std::string &file_path, const nlohmann::json &json_obj) {
  std::ofstream out_stream(file_path);
  GE_CHK_BOOL_RET_STATUS(out_stream.is_open(), FAILED, "Failed to open json file:%s", file_path.c_str());
  try {
    out_stream << std::setw(kIndentationLen) << json_obj;
  } catch (const std::exception &e) {
    GELOGE(FAILED, "Failed to write json file:%s, err msg: %s", file_path.c_str(), e.what());
    return FAILED;
  }
  GE_CHK_BOOL_RET_STATUS(out_stream.good(), FAILED, "Failed to write json file[%s], error msg = %s", file_path.c_str(),
                         strerror(errno));
  return SUCCESS;
}

Status FlowModelCache::TryMatchCacheForUdfSubGraph(bool &is_match) const {
  const std::string release_file = StringUtils::ReplaceAll(cache_index_.cache_file_name, "_root.om", "_release.om");
  auto cache_file_exist = CheckFileExist(release_file);
  if (cache_manual_check_ && cache_file_exist) {
    GE_CHK_BOOL_RET_STATUS(root_graph_->SetExtAttr("_cache_skip_release_info_check", true),
      FAILED, "Failed to set cache graph info for graph[%s].", root_graph_->GetName().c_str());
    const std::string release_real_path = RealPath(release_file.c_str());
    GE_ASSERT_TRUE(!release_real_path.empty(), " Get empty real path by [%s].", release_file.c_str());
    GE_CHK_BOOL_RET_STATUS(root_graph_->SetExtAttr("_cache_graph_udf_om_file", release_real_path),
        FAILED, "Failed to set cache graph info for graph[%s].", root_graph_->GetName().c_str());
    GEEVENT("Match cache successfully for udf graph[%s].", root_graph_->GetName().c_str());
    is_match = true;
  } else {
    GELOGI("Can not match cache for udf graph[%s], cache need manual check = %d, cache exist = %d.",
           root_graph_->GetName().c_str(),
           static_cast<int32_t>(cache_manual_check_), static_cast<int32_t>(cache_file_exist));
    is_match = false;
  }
  return SUCCESS;
}

Status FlowModelCache::TryMatchCacheForSubGraph(bool &is_match) const {
  if (GetSubmodelPneId() == PNE_ID_UDF) {
    return TryMatchCacheForUdfSubGraph(is_match);
  }
  is_match = false;
  if (!CheckFileExist(cache_index_.cache_file_name)) {
    GELOGI("No cache for graph[%s].", root_graph_->GetName().c_str());
    return SUCCESS;
  }

  if (!CheckFileExist(build_info_path_)) {
    if (cache_manual_check_) {
      is_match = true;
      GELOGI("No need to match buildinfo for nn cache, graph = %s.", root_graph_->GetName().c_str());
    } else {
      GELOGI("Failed to match cache for graph[%s], no buildinfo in simple cache mode.",
             root_graph_->GetName().c_str());
    }
    return SUCCESS;
  }

  std::string cache_graph_info;
  std::map<std::string, std::string> cache_build_options;
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ReadJsonFile(build_info_path_, json_obj),
      "Failed to read build info json file[%s].", build_info_path_.c_str());
  try {
    auto iter = json_obj.find("graph_info");
    if (iter != json_obj.end()) {
      cache_graph_info = iter.value().get<std::string>();
    }
    iter = json_obj.find("build_options");
    if (iter != json_obj.end()) {
      cache_build_options = iter.value().get<std::map<std::string, std::string>>();
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read cache info from json file[%s], err msg[%s].", build_info_path_.c_str(), e.what());
    return FAILED;
  }
  const std::string graph_info = root_graph_->TryGetExtAttr<std::string>("_graph_info_for_data_flow_cache", "");
  const std::map<std::string, std::string> build_options =
      root_graph_->TryGetExtAttr<std::map<std::string, std::string>>("_graph_build_options_for_data_flow_cache", {});
  if (cache_manual_check_) {
    if (((!cache_graph_info.empty()) && (!graph_info.empty()) && (cache_graph_info != graph_info)) ||
        ((!cache_build_options.empty()) && (!build_options.empty()) && (cache_build_options != build_options))) {
      GELOGI("The cache is not match for graph[%s], graph_info or buildinfo is not equal in cache manual check mode.",
             root_graph_->GetName().c_str());
      return SUCCESS;
    }
  } else {
    if (cache_graph_info != graph_info || cache_build_options != build_options) {
      GELOGI("The cache is not match for graph[%s], graph_info or buildinfo is not equal in simple cache mode.",
             root_graph_->GetName().c_str());
      return SUCCESS;
    }
  }
  is_match = true;
  GEEVENT("Match cache successfully for nn graph[%s].", root_graph_->GetName().c_str());
  return SUCCESS;
}

Status FlowModelCache::TryAddCacheForUdfGraph() const {
  if (!CheckFileExist(build_info_path_)) {
    GELOGD("No cache for graph[%s].", root_graph_->GetName().c_str());
    return SUCCESS;
  }
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ReadJsonFile(build_info_path_, json_obj),
      "Failed to read build info json file[%s].", build_info_path_.c_str());
  std::string cache_graph_info;
  std::string om_file;
  try {
    auto iter = json_obj.find("graph_info");
    if (iter != json_obj.end()) {
      cache_graph_info = iter.value().get<std::string>();
    }
    iter = json_obj.find("om_file_path");
    if (iter != json_obj.end()) {
      om_file = iter.value().get<std::string>();
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read cache info from json file[%s], err msg[%s].", build_info_path_.c_str(), e.what());
    return FAILED;
  }
  if (!om_file.empty()) {
    std::string split_om_data_base_dir;
    GE_CHK_STATUS_RET(GetSplitOmDataBaseDir(split_om_data_base_dir), "Get split om data file dir failed.");
    if (!CheckFileExist(split_om_data_base_dir + om_file)) {
      GELOGI("The cache om file[%s] does not exist. Graph [%s] need to be compile again.",
             (split_om_data_base_dir + om_file).c_str(), root_graph_->GetName().c_str());
      return SUCCESS;
    }
    GE_CHK_BOOL_RET_STATUS(root_graph_->SetExtAttr("_cache_graph_udf_om_file", split_om_data_base_dir + om_file),
        FAILED, "Failed to set cache graph info for graph[%s].", root_graph_->GetName().c_str());
  }
  if (!cache_graph_info.empty()) {
    GE_CHK_BOOL_RET_STATUS(root_graph_->SetExtAttr("_cache_graph_info_for_data_flow_cache", cache_graph_info), FAILED,
                           "Failed to set cache graph info for graph[%s].", root_graph_->GetName().c_str());
  }
  return SUCCESS;
}

Status FlowModelCache::FormatCacheCompilerResult(const ge::NamedAttrs &compile_results, CacheCompileResult &result) {
  ge::NamedAttrs runnable_resources_info;
  GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetNamedAttrs(compile_results, kAttrNameDataFlowRunnableResource,
                                                      runnable_resources_info),
                         FAILED, "Get graph's attr[%s] failed.",
                         kAttrNameDataFlowRunnableResource);
  auto runnable_infos = ge::AttrUtils::GetAllAttrs(runnable_resources_info);
  for (const auto &runnable_info : runnable_infos) {
    std::string release_path;
    GE_CHK_STATUS_RET(runnable_info.second.GetValue<std::string>(release_path), "Failed to get release_path.");
    result.compile_bin_info[runnable_info.first] = release_path;
  }
  ge::NamedAttrs running_resource_info;
  GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetNamedAttrs(compile_results, kAttrNameDataFlowRunningResourceInfo,
                                                      running_resource_info),
                         FAILED, "Get graph's attr[%s] failed.",
                         kAttrNameDataFlowRunningResourceInfo);
  auto running_infos = ge::AttrUtils::GetAllAttrs(running_resource_info);
  for (const auto &running_info : running_infos) {
    int64_t res_num = 0;
    GE_CHK_STATUS_RET(running_info.second.GetValue<int64_t>(res_num), "Failed to get res_num.");
    result.running_resources_info[running_info.first] = res_num;
  }
  return SUCCESS;
}

Status FlowModelCache::CacheBuildInfo() const {
  const std::string graph_info = root_graph_->TryGetExtAttr<std::string>("_graph_info_for_data_flow_cache", "");
  const std::string om_model_file = root_graph_->TryGetExtAttr<std::string>("_udf_om_file_for_data_flow_cache", "");
  const std::map<std::string, std::string> build_options =
      root_graph_->TryGetExtAttr<std::map<std::string, std::string>>("_graph_build_options_for_data_flow_cache", {});
  ge::NamedAttrs compile_results;
  auto has_result = ge::AttrUtils::GetNamedAttrs(root_graph_, kAttrNameDataFlowCompilerResult, compile_results);
  if (graph_info.empty() && build_options.empty() && (!has_result)) {
    GELOGD("No graph info, build options and compiler result need cache.");
    return SUCCESS;
  }
  nlohmann::json json_obj;
  try {
    if (!graph_info.empty()) {
      json_obj["graph_info"] = graph_info;
    }
    if (GetSubmodelPneId() != PNE_ID_UDF) {
      json_obj["build_options"] = build_options;
    }
    if (has_result) {
      CacheCompileResult result = {};
      GE_CHK_STATUS_RET(FormatCacheCompilerResult(compile_results, result), "Failed to format compiler result.");
      json_obj["bin_info"] = result.compile_bin_info;
      json_obj["resource_info"] = result.running_resources_info;
    }
    if (!om_model_file.empty()) {
      json_obj["om_file_path"] = om_model_file;
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to cache build info for graph[%s], err msg[%s].", root_graph_->GetName().c_str(), e.what());
    return FAILED;
  }
  return WriteJsonFile(build_info_path_, json_obj);
}

std::string FlowModelCache::GetSubmodelPneId() const {
  std::string pne_id = PNE_ID_NPU;
  (void)AttrUtils::GetStr(root_graph_, ATTR_NAME_PROCESS_NODE_ENGINE_ID, pne_id);
  return pne_id;
}
}  // namespace ge
