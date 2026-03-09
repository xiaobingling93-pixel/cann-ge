/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_cache.h"
#include <string>
#include <regex>
#include <fstream>
#include <iomanip>
#include <sys/file.h>
#include "mmpa/mmpa_api.h"
#include "common/debug/log.h"
#include "common/proto_util/proto_util.h"
#include "common/helper/file_saver.h"
#include "common/helper/model_parser_base.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/common/helper/model_helper.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/thread_pool/thread_pool.h"

namespace ge {
namespace {
constexpr const char *kFileNamePattern = R"(^[A-Za-z0-9_\-]{1,128}$)";
constexpr const char *kJsonFieldCacheFileList = "cache_file_list";
constexpr const char *kJsonFieldCacheFileName = "cache_file_name";
constexpr const char *kJsonFielVarDescFileName = "var_desc_file_name";
constexpr const char *kJsonFieldGraphKey = "graph_key";
constexpr const char *kJsonFielCacheManualCheck = "cache_manual_check";
constexpr const char *kJsonFieldCacheDebugMode = "cache_debug_mode";
constexpr int32_t kIndentationLen = 4;
constexpr const char *kFinalWeightDirName = "/weight/";
constexpr const char *kLockFileName = ".lock";
constexpr const char *kSuspendGraphOriginalName = "_suspend_graph_original_name";
std::unordered_set<std::string> kDataUnChangedNodeType;
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

ModelCache::~ModelCache() {
  if (lock_file_fd_ != -1) {
    (void)flock(lock_file_fd_, LOCK_UN);
    (void)close(lock_file_fd_);
    lock_file_fd_ = -1;
  }
}

Status ModelCache::Init(const ComputeGraphPtr &root_graph, GraphRebuildStateCtrl *ctrl) {
  cache_enable_ = false;
  GE_CHECK_NOTNULL(root_graph);
  cache_dir_ = NormalizeDirPath(GetCacheDirFromContext());
  cache_index_.graph_key = GetGraphKeyFromContext();
  var_accelerate_ctrl_ = ctrl;
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
  GE_WARN_ASSERT_GRAPH_SUCCESS(InitCacheFileInfo(), "Fail to init cache file info, cache_dir=%s, graph_key=%s.",
      cache_dir_.c_str(), cache_index_.graph_key.c_str());
  session_id_ = root_graph->GetSessionID();
  graph_id_ = root_graph->GetGraphID();
  cache_enable_ = true;
  const auto &external_weight_manager =
      ExternalWeightManagerPool::Instance().GetManager(GetContext().SessionId());
  GE_CHECK_NOTNULL(external_weight_manager);
  std::string weight_dir = ExternalWeightManager::GetWeightPathFromOption();
  if(weight_dir.empty()){
    weight_dir = cache_dir_ + kFinalWeightDirName;
  }
  external_weight_manager->SetWeightPath(weight_dir);
  GELOGI("Init end, cache_dir=%s, graph_key=%s, cache_file=%s, index_file=%s, cache manual check=%d, weight_dir=%s.",
         cache_dir_.c_str(), cache_index_.graph_key.c_str(), cache_index_.cache_file_name.c_str(),
         index_file_.c_str(), static_cast<int32_t>(cache_manual_check_), weight_dir.c_str());
  return SUCCESS;
}

std::string ModelCache::NormalizeDirPath(const std::string &dir_path) {
  std::string normalize_path = dir_path;
  if (!dir_path.empty() && (dir_path.at(dir_path.length() - 1) != '/')) {
    (void)normalize_path.append("/");
  }
  return normalize_path;
}

Status ModelCache::InitCacheFileInfo() {
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

Status ModelCache::ReadIndex(const std::string &index_file,
                             std::vector<CacheFileIdx> &cache_file_list) const {
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ReadJsonFile(index_file, json_obj), "Failed to read cache index file[%s].", index_file.c_str());
  try {
    cache_file_list = json_obj[kJsonFieldCacheFileList].get<std::vector<CacheFileIdx>>();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read cache index file[%s], err msg: %s", index_file.c_str(), e.what());
    return FAILED;
  }
  for (auto &idx : cache_file_list) {
    if (idx.cache_file_name.find("/") == std::string::npos) {
      idx.cache_file_name = cache_dir_ + idx.cache_file_name;
      GELOGD("Add cache dir to cache file name[%s].", idx.cache_file_name.c_str());
    }
    if ((!idx.var_desc_file_name.empty()) && (idx.var_desc_file_name.find("/") == std::string::npos)) {
      idx.var_desc_file_name = cache_dir_ + idx.var_desc_file_name;
      GELOGD("Add cache dir to cache var desc file name[%s].", idx.var_desc_file_name.c_str());
    }
  }
  return SUCCESS;
}

void ModelCache::GenerateCacheFile() {
  cache_index_.cache_file_name = cache_dir_ + cache_file_prefix_ + "_" + cache_file_time_str_ + ".om";
  cache_index_.var_desc_file_name = cache_dir_ + cache_file_prefix_ + "_" + cache_file_time_str_ + ".rdcpkt";
}

Status ModelCache::InitCacheFileByIdx(const std::string &cache_path) {
  // if index is exist, gen file name.
  if (CheckFileExist(index_file_)) {
    std::vector<CacheFileIdx> cache_file_list;
    GE_CHK_STATUS_RET(ReadIndex(index_file_, cache_file_list),
                      "Failed to read cache index list from file:%s", index_file_.c_str());
    for (const auto &idx : cache_file_list) {
      if (idx.graph_key == cache_index_.graph_key) {
        GE_WARN_ASSERT(CheckFileExist(idx.cache_file_name),
            "cache file[%s] in cache index file[%s] is not exists.",
            idx.cache_file_name.c_str(), index_file_.c_str());
        GE_WARN_ASSERT((idx.var_desc_file_name.empty() || CheckFileExist(idx.cache_file_name)),
            "var desc file[%s] in cache index file[%s] is not exists.",
            idx.var_desc_file_name.c_str(), index_file_.c_str());
        GELOGI("Matched graph_key[%s] success, cache om file = %s, cache var desc file = %s, cache dir = %s.",
               cache_index_.graph_key.c_str(), idx.cache_file_name.c_str(),
               idx.var_desc_file_name.c_str(), cache_path.c_str());
        matched_file_list_.emplace_back(idx);
      }
    }
  }

  if (matched_file_list_.empty()) {
    // not found, need generate a new file.
    GenerateCacheFile();
    GELOGI("No graph_key[%s] found in cache index file[%s], generate cache om file[%s], var desc file[%s]",
           cache_index_.graph_key.c_str(), index_file_.c_str(),
           cache_index_.cache_file_name.c_str(), cache_index_.var_desc_file_name.c_str());
  }
  return SUCCESS;
}

Status ModelCache::RenewVarDesc(uint64_t session_id, const VarDescCache &update_var_desc) const {
  for (const auto &it : update_var_desc.staged_var_desc_map) {
    const auto &var_name = it.first;
    GE_CHECK_NOTNULL(VarManager::Instance(session_id));
    GE_CHK_STATUS_RET(ge::VarManager::Instance(session_id)->RenewCurVarDesc(var_name, it.second),
                      "[Renew][Descriptor] for node:%s failed, session_id:%lu",
                      var_name.c_str(), session_id);
  }
  return SUCCESS;
}

Status ModelCache::RenewVarDesc(uint64_t session_id,
                                const std::string &var_name,
                                const VarTransRoad &fusion_road) const {
  if (kDataUnChangedNodeType.empty()) {
    kDataUnChangedNodeType = {RESHAPE, REFORMAT, SQUEEZEV2, UNSQUEEZEV2};
  }

  // renew var desc if the trans_road is all reshape or reformat
  for (const auto &road : fusion_road) {
    if (kDataUnChangedNodeType.count(road.node_type) == 0) {
      return SUCCESS;
    }
  }

  if (!fusion_road.empty()) {
    auto end_iter = fusion_road.rbegin();
    GE_CHK_STATUS_RET(VarManager::Instance(session_id)->RenewCurVarDesc(var_name, end_iter->output),
                      "Failed to refresh trans_roads of var[%s]", var_name.c_str());
  }
  return SUCCESS;
}

Status ModelCache::RefreshVariableDesc(const ComputeGraphPtr &root_graph,
                                       const VarDescCache &update_var_desc) {
  const uint64_t session_id = root_graph->GetSessionID();
  const uint32_t graph_id = root_graph->GetGraphID();
  GE_CHK_STATUS_RET(RenewVarDesc(session_id, update_var_desc), "Failed to renew var desc.");
  GE_CHECK_NOTNULL(var_accelerate_ctrl_);
  for (const auto &var_name : update_var_desc.changed_var_names) {
    auto trans_roads_it = update_var_desc.var_trans_roads_map.find(var_name);
    GE_CHK_BOOL_RET_STATUS(trans_roads_it != update_var_desc.var_trans_roads_map.end(), FAILED,
                           "Failed to get var[%s] trans road from cache.", var_name.c_str());
    const auto &desc_it = update_var_desc.var_desc_map.find(var_name);
    GE_CHK_BOOL_RET_STATUS(desc_it != update_var_desc.var_desc_map.end(), FAILED,
                           "Failed to get var[%s] desc from cache.", var_name.c_str());
    const auto &trans_roads = trans_roads_it->second;
    GE_CHECK_NOTNULL(VarManager::Instance(session_id));
    GE_CHK_STATUS_RET(VarManager::Instance(session_id)->SetTransRoad(var_name, trans_roads),
                      "Failed to refresh trans_roads of var[%s]", var_name.c_str());
    GE_CHK_STATUS_RET_NOLOG(VarManager::Instance(session_id)->SetChangedGraphId(var_name, graph_id));
    var_accelerate_ctrl_->SetStateChanged(var_name);
    GE_CHK_STATUS_RET(RenewVarDesc(session_id, var_name, trans_roads),
                      "Failed to refresh trans_roads of var[%s]", var_name.c_str());
    GELOGI("Refresh var[%s] desc and trans_roads success.", var_name.c_str());
  }
  return SUCCESS;
}

void ModelCache::InitVarDescFromProto(const deployer::VarDescInfo &desc_proto, VarDescCache &var_desc) {
  for (const auto &x : desc_proto.cur_var_tensor_desc_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second, tensor_desc);
    var_desc.var_desc_map[x.first] = tensor_desc;
  }

  for (const auto &x : desc_proto.var_to_trans_road()) {
    std::vector<TransNodeInfo> trans_node_info_vec;
    for (int32_t i = 0; i < x.second.node_info_size(); i++) {
      TransNodeInfo trans_node_info;
      trans_node_info.node_type = x.second.node_info(i).node_type();
      const proto::TensorDescriptor &input_tensor_desc = x.second.node_info(i).input();
      const proto::TensorDescriptor &output_tensor_desc = x.second.node_info(i).output();
      GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&input_tensor_desc, trans_node_info.input);
      GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&output_tensor_desc, trans_node_info.output);
      trans_node_info_vec.emplace_back(trans_node_info);
    }
    var_desc.var_trans_roads_map[x.first] = trans_node_info_vec;
  }

  for (const auto &x : desc_proto.changed_var_names()) {
    var_desc.changed_var_names.emplace_back(x);
  }

  for (const auto &x : desc_proto.staged_var_tensor_desc_map()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&x.second, tensor_desc);
    var_desc.staged_var_desc_map[x.first] = tensor_desc;
  }
  GELOGI("Var match info, var desc size = %zu, trans road size = %zu, changed var names size = %zu",
         desc_proto.cur_var_tensor_desc_map().size(),
         desc_proto.var_to_trans_road().size(),
         var_desc.changed_var_names.size());
}

bool ModelCache::CompareVarDesc(uint64_t session_id, const VarDescCache &var_descs) {
  for (const auto &var_desc_it : var_descs.var_desc_map) {
    const auto &var_name = var_desc_it.first;
    const auto &var_desc = var_desc_it.second;
    GeTensorDesc current_tensor_desc;
    GE_CHECK_NOTNULL(VarManager::Instance(session_id));
    if (VarManager::Instance(session_id)->GetCurVarDesc(var_name, current_tensor_desc) != SUCCESS) {
      GELOGI("The var[%s] does not exist in var manager, can not load from cache.", var_name.c_str());
      return false;
    }

    if (var_desc == current_tensor_desc) {
      continue;
    } else {
      GELOGI("The var[%s] desc of var manager is not the same with cache, can not load from cache.", var_name.c_str());
      return false;
    }
  }
  return true;
}

Status ModelCache::MatchVariableDesc(uint64_t session_id,
                                     const std::string &var_desc_file_name,
                                     bool &matched,
                                     VarDescCache &update_var_desc) {
  if (var_desc_file_name.empty()) {
    GELOGD("The file of var desc[%s] does not exist, no need match var desc.", var_desc_file_name.c_str());
    matched = true;
    return SUCCESS;
  }

  std::string real_path = RealPath(var_desc_file_name.c_str());
  GE_CHK_BOOL_RET_STATUS(!real_path.empty(), FAILED,
                         "The path[%s] of var desc cache file is invalid.", var_desc_file_name.c_str());
  std::ifstream file_stream(real_path, std::ifstream::in);
  GE_CHK_BOOL_RET_STATUS(file_stream.is_open(), FAILED, "[Open][File] %s failed.", real_path.c_str());
  deployer::VarMatchInfo var_match_info;
  GE_CHK_BOOL_RET_STATUS(var_match_info.ParseFromIstream(&file_stream), FAILED, "[Parse][VarDesc] failed.");

  VarDescCache var_desc_after_compile;
  InitVarDescFromProto(var_match_info.desc_info_after_compile(), var_desc_after_compile);
  GELOGI("Try match var desc with after compile cache, after compile descs size = %zu",
         var_desc_after_compile.var_desc_map.size());
  matched = CompareVarDesc(session_id, var_desc_after_compile);
  if (matched) {
    GELOGI("After compiled var desc compare success, cache var desc size = %zu.",
           var_desc_after_compile.var_desc_map.size());
    return SUCCESS;
  }

  VarDescCache var_desc_before_compile;
  InitVarDescFromProto(var_match_info.desc_info_before_compile(), var_desc_before_compile);
  GELOGI("Try match var desc with before compile cache, before compile descs size = %zu",
         var_desc_before_compile.var_desc_map.size());
  GeTensorDesc var_tensor_desc;
  matched = CompareVarDesc(session_id, var_desc_before_compile);
  if (matched) {
    GELOGI("Before compiled var desc compare success, cache var desc size = %zu.",
           var_desc_before_compile.var_desc_map.size());
    update_var_desc = std::move(var_desc_after_compile);
  }
  return SUCCESS;
}

Status ModelCache::TryMatchVarDescWithCache(bool &is_matched, VarDescCache &update_var_desc) {
  is_matched = false;
  if (matched_file_list_.empty()) {
    GELOGD("cache file does not exist, no need load cache.");
    return SUCCESS;
  }

  for (const auto &index : matched_file_list_) {
    GE_CHK_STATUS_RET(MatchVariableDesc(session_id_,
                                        index.var_desc_file_name,
                                        is_matched,
                                        update_var_desc),
                      "Failed to match var desc.");
    if (is_matched) {
      cache_index_ = index;
      GELOGI("Match variable tensor descs success, load from cache[%s].", cache_index_.cache_file_name.c_str());
      break;
    }
  }
  return SUCCESS;
}

Status ModelCache::TryLoadModelFromCache(const ComputeGraphPtr &root_graph, GeRootModelPtr &ge_root_model) {
  if (!cache_enable_) {
    GELOGD("cache is disable, no need load cache.");
    return SUCCESS;
  }

  GE_DISMISSABLE_GUARD(record_var_desc, ([this]() {
    if (VarManager::Instance(session_id_) != nullptr) {
      GE_CHK_STATUS(VarManager::Instance(session_id_)->VarDescInfoToSerial(session_id_, var_desc_before_compile_));
    }
  }));
  bool is_need_load = true;
  GE_CHK_STATUS_RET(CheckCacheFile(root_graph, is_need_load), "Check file can be loaded failed.");
  if (!is_need_load) {
    GELOGD("Cache not matched or submodel cache is udf.");
    return SUCCESS;
  }
  ge_root_model = nullptr;
  GE_TRACE_START(LoadModel);
  GE_CHK_STATUS_RET(LoadToGeRootModel(cache_index_.cache_file_name, ge_root_model),
                    "Failed to load model, cache_file:%s", cache_index_.cache_file_name.c_str());
  std::string option_external_weight_dir = ExternalWeightManager::GetWeightPathFromOption();
  if(option_external_weight_dir.empty()){
    option_external_weight_dir = cache_dir_ + "/weight/";
  }
  GE_CHK_STATUS_RET(AssignConstantVarMem(ge_root_model, option_external_weight_dir, session_id_, graph_id_),
      "Failed to assign constant mem for cache model, cache_file:%s", cache_index_.cache_file_name.c_str());
  GE_ASSERT_SUCCESS(UpdateGeModelSessionId(ge_root_model, session_id_), "Failed to update ge model session id");
  std::string session_graph_id;
  if (AttrUtils::GetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GE_CHK_STATUS_RET(UpdateSessionGraphId(ge_root_model, session_graph_id),
                      "Failed to update model session graph id, session_graph_id:%s", session_graph_id.c_str());
  }
  const std::string trace_log = "loading model cache by key[" + cache_index_.graph_key + "]";
  GE_COMPILE_TRACE_TIMESTAMP_END(LoadModel, trace_log.c_str());
  GE_DISMISS_GUARD(record_var_desc);
  GELOGI("load model from cache file:%s success.", cache_index_.cache_file_name.c_str());
  return SUCCESS;
}

Status ModelCache::LoadToGeRootModel(const std::string &model_path, GeRootModelPtr &ge_root_model) {
  ge_root_model = nullptr;
  ModelData model;
  // Load model from file, default 0 priority.
  GE_CHK_STATUS_RET_NOLOG(ModelParserBase::LoadFromFile(model_path.c_str(), 0, model));
  GE_MAKE_GUARD(model_guard, [&model]() {
    if (model.model_data != nullptr) {
      delete[] static_cast<char *>(model.model_data);
      model.model_data = nullptr;
    }
  });

  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.LoadRootModel(model), "[Load][RootModel] failed.");
  ge_root_model = model_helper.GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  GELOGD("load ge root model success, model_name=%s, graph=%s.", ge_root_model->GetModelName().c_str(),
         ge_root_model->GetRootGraph()->GetName().c_str());
  // ge root model dynamic model name is empty, need set by graph name.
  if (ge_root_model->GetModelName().empty()) {
    ge_root_model->SetModelName(ge_root_model->GetRootGraph()->GetName());
  }
  return SUCCESS;
}

Status ModelCache::AssignConstantVarMem(const GeRootModelPtr &ge_root_model,
    const std::string &model_path, const uint64_t session_id, const uint32_t graph_id) {
  const auto root_graph = ge_root_model->GetRootGraph();
  std::set<ComputeGraph *> refreshed_graphs;
  GE_CHK_STATUS_RET(FileConstantUtils::SetExternalPath(root_graph, model_path), "Failed to set external path:%s.",
                    model_path.c_str());
  root_graph->SetSessionID(session_id);
  root_graph->SetGraphID(graph_id);
  GE_CHK_STATUS_RET(ModelHelper::UpdateGeRootModelTaskAddr(ge_root_model, root_graph, refreshed_graphs, true),
                    "Update model task address failed, graph name %s", root_graph->GetName().c_str());
  return SUCCESS;
}

Status ModelCache::UpdateGeModelSessionId(const GeRootModelPtr &ge_root_model, const uint64_t session_id) {
  const auto &ge_models = ge_root_model->GetSubgraphInstanceNameToModel();
  for (const auto &pair : ge_models) {
    const auto &ge_model = pair.second;
    uint64_t old_session_id = std::numeric_limits<uint64_t>::max();
    (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_SESSION_ID, old_session_id);
    if (old_session_id != session_id) {
      GE_ASSERT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));
      GELOGI("update ge model[%s] session id from %llu to %llu",
             ge_model->GetName().c_str(), old_session_id, session_id);
    }
  }
  return SUCCESS;
}

Status ModelCache::UpdateSessionGraphId(const GeRootModelPtr &ge_root_model,
                                        const std::string &session_graph_id) {
  const auto &root_graph = ge_root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph, ", model root graph is null");
  std::string old_session_graph_id;
  if (AttrUtils::GetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, old_session_graph_id) &&
      (old_session_graph_id == session_graph_id)) {
    GELOGD("session graph id is same, no need update.");
    return SUCCESS;
  }
  // -1 is offline session, no need to update
  if (old_session_graph_id.find("-1") != std::string::npos) {
    GELOGI("old session graph id[%s] is offline, no need update.", old_session_graph_id.c_str());
    return SUCCESS;
  }
  GELOGI("need update graph[%s] session graph id from %s to %s.", root_graph->GetName().c_str(),
         old_session_graph_id.c_str(), session_graph_id.c_str());
  bool refreshed;
  GE_CHK_STATUS_RET(ModelHelper::UpdateSessionGraphId(root_graph, session_graph_id, refreshed),
                    "update graph[%s] session graph id failed",
                    root_graph->GetName().c_str());
  return SUCCESS;
}

Status ModelCache::CheckCacheFile(const ComputeGraphPtr &root_graph, bool &need_load) {
  need_load = false;
  bool is_matched = false;
  VarDescCache update_var_desc;
  GE_CHK_STATUS_RET(TryMatchVarDescWithCache(is_matched, update_var_desc), "Failed to match Var desc with cache.");
  if (!is_matched) {
    GELOGI("cache var desc file is not matched, can not load cache.");
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(RefreshVariableDesc(root_graph, update_var_desc),
                    "Failed to refresh variable desc.");
  need_load = true;
  return SUCCESS;
}

Status ModelCache::TryCacheModel(GeRootModelPtr &ge_root_model) {
  if (!cache_enable_ || cache_debug_mode_) {
    GELOGD("cache is disable or in debug mode, no need cache.");
    return SUCCESS;
  }
  GenerateCacheFile();
  auto ret = SaveModelToGeRootModel(ge_root_model, cache_index_.cache_file_name);
  GE_CHK_STATUS_RET(ret, "Failed to cache no model relation model, cache_file:%s",
                    cache_index_.cache_file_name.c_str());
  GELOGI("cache model success, cache file:%s", cache_index_.cache_file_name.c_str());

  GE_CHK_STATUS_RET(SaveVarDescToFile(), "Failed to save var desc to file");
  GE_CHK_STATUS_RET(SaveCacheIndexFile(), "Failed to save cache file index, cache_file:%s, cache file index:%s",
                    cache_index_.cache_file_name.c_str(), index_file_.c_str());
  return SUCCESS;
}

Status ModelCache::SaveVarDescToFile() {
  deployer::VarDescInfo var_desc_after_compile;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  GE_CHK_STATUS_RET(VarManager::Instance(session_id_)->VarDescInfoToSerial(session_id_, var_desc_after_compile));
  // var manager is empty, no need to save var cache
  if (var_desc_before_compile_.cur_var_tensor_desc_map().size() == 0 &&
      var_desc_after_compile.cur_var_tensor_desc_map().size() == 0) {
    cache_index_.var_desc_file_name.clear();
    GELOGI("The number of var is 0, no need to cache var desc.");
    return SUCCESS;
  }
  deployer::VarMatchInfo match_info;
  *match_info.mutable_desc_info_before_compile() = std::move(var_desc_before_compile_);
  *match_info.mutable_desc_info_after_compile() = std::move(var_desc_after_compile);
  auto changed_var_names = VarManager::Instance(session_id_)->GetChangedVarNames(graph_id_);
  for (const auto &var_name : changed_var_names) {
    match_info.mutable_desc_info_after_compile()->add_changed_var_names(var_name);
  }
  auto changed_var_descs = VarManager::Instance(session_id_)->GetStagedVarDescs(graph_id_);
  for (const auto &info : changed_var_descs) {
    proto::TensorDescriptor tensor_desc_proto;
    GeTensorSerializeUtils::GeTensorDescAsProto(info.second, &tensor_desc_proto);
    (void)match_info.mutable_desc_info_after_compile()->mutable_staged_var_tensor_desc_map()->insert(
        {info.first, tensor_desc_proto});
  }

  std::string match_info_str;
  GE_CHK_BOOL_RET_STATUS(match_info.SerializeToString(&match_info_str), FAILED,
                         "VarMatchInfo serialize to string failed.");

  const mmMode_t kAccess = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR));
  const int32_t fd = mmOpen2(cache_index_.var_desc_file_name.c_str(),
                             static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) |
                                                  static_cast<uint32_t>(M_CREAT) |
                                                  static_cast<uint32_t>(O_TRUNC)),
                             kAccess);
  GE_CHK_BOOL_RET_STATUS(fd >= 0, FAILED, "Failed to open file, path = %s", cache_index_.var_desc_file_name.c_str());
  (void) mmClose(fd);
  GE_DISMISSABLE_GUARD(file_guard, [this]() {
    (void) std::remove(cache_index_.var_desc_file_name.c_str());
  });
  std::ofstream file_stream(cache_index_.var_desc_file_name, std::ios::out | std::ios::binary);
  GE_CHK_BOOL_RET_STATUS(file_stream.good(),
                         FAILED,
                         "Failed to open file for write, path = %s",
                         cache_index_.var_desc_file_name.c_str());
  file_stream << match_info_str;
  GE_CHK_BOOL_RET_STATUS(file_stream.good(), FAILED,
                         "Failed to write cache desc file[%s], error msg = %s",
                         cache_index_.var_desc_file_name.c_str(), strerror(errno));
  GE_DISMISS_GUARD(file_guard);
  GELOGI("Save var descs to cache file[%s] success, before cache var desc size = %zu, "
         "after cache var desc size = %zu, cache_size = %zu",
         cache_index_.var_desc_file_name.c_str(),
         var_desc_before_compile_.cur_var_tensor_desc_map().size(),
         var_desc_after_compile.cur_var_tensor_desc_map().size(),
         match_info_str.size());
  return SUCCESS;
}

std::string ModelCache::GetCacheDirFromContext() {
  std::string cache_dir;
  graphStatus ret = GetThreadLocalContext().GetOption(OPTION_GRAPH_COMPILER_CACHE_DIR, cache_dir);
  if (ret != GRAPH_SUCCESS) {
    GELOGD("option[%s] does not exist, build cache is disable.", OPTION_GRAPH_COMPILER_CACHE_DIR);
    return cache_dir;
  }
  GELOGD("compile cache dir is %s.", cache_dir.c_str());
  return cache_dir;
}

std::string ModelCache::GetGraphKeyFromContext() {
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

Status ModelCache::ReadCacheConfig(const std::string &config_file, CacheConfig &cache_config) {
  auto real_path = RealPath(config_file.c_str());
  if (real_path.empty()) {
    return SUCCESS;
  }
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ReadJsonFile(config_file, json_obj), "Failed to read cache config file[%s].", config_file.c_str());
  try {
    auto iter = json_obj.find(kJsonFielCacheManualCheck);
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
  GELOGI("Read cache config file success, cache manuak check = %d, cache debug mode = %d.",
         static_cast<int32_t>(cache_config.cache_manual_check), static_cast<int32_t>(cache_config.cache_debug_mode));
  return SUCCESS;
}

Status ModelCache::SaveModelToGeRootModel(GeRootModelPtr &ge_root_model, const std::string &cache_file_name) {
  GE_ASSERT_NOTNULL(ge_root_model);
  ModelBufferData model_buff;
  auto ret = SerializeModel(ge_root_model, model_buff);
  GE_CHK_STATUS_RET(ret, "Failed to serialize model, model_name=%s", ge_root_model->GetModelName().c_str());
  ret = FileSaver::SaveToFile(cache_file_name, model_buff.data.get(), model_buff.length);
  GE_CHK_STATUS_RET(ret, "Failed to save model, model_name=%s, file_name=%s, length=%lu.",
                    ge_root_model->GetModelName().c_str(), cache_file_name.c_str(),
                    model_buff.length);
  GELOGI("save to ge root model success, model_name=%s, file_name=%s.", ge_root_model->GetModelName().c_str(),
         cache_file_name.c_str());
  return SUCCESS;
}

Status ModelCache::SerializeModel(const GeRootModelPtr &ge_root_model, ModelBufferData &model_buff) {
  bool is_unknown_shape = false;
  (void) ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);
  GE_CHK_STATUS_RET(model_helper.SaveToOmRootModel(ge_root_model, "no-output.om", model_buff, is_unknown_shape),
                    "[Serialize][Submodel] failed, model_name = [%s]", ge_root_model->GetModelName().c_str());
  GELOGD("[Serialize][Submodel] succeeded, model_name = [%s], size = %lu", ge_root_model->GetModelName().c_str(),
         model_buff.length);
  return SUCCESS;
}

bool ModelCache::IsMatchFileName(const std::string &str) {
  const std::regex file_name_regex(kFileNamePattern);
  return std::regex_match(str, file_name_regex);
}

bool ModelCache::CheckFileExist(const std::string &file_path) {
  return mmAccess(file_path.c_str()) == EN_OK;
}

Status ModelCache::GetRealFileName(std::string &cache_file_name) const {
  cache_file_name = cache_file_name.substr(cache_file_name.rfind("/") + 1UL, cache_file_name.length());
  GE_ASSERT_TRUE(!cache_file_name.empty());
  GELOGD("Get real file name[%s].", cache_file_name.c_str());
  return SUCCESS;
}

Status ModelCache::SaveCacheIndexFile() const {
  if (index_file_.empty()) {
    GELOGD("hash index file is empty, no need save cache index file.");
    return SUCCESS;
  }
  std::vector<CacheFileIdx> cache_file_list;
  bool file_exist = CheckFileExist(index_file_);
  if (file_exist) {
    GE_CHK_STATUS_RET(ReadIndex(index_file_, cache_file_list),
                      "Failed to read cache index list from file:%s", index_file_.c_str());
  }
  CacheFileIdx new_cache_index = cache_index_;
  GE_CHK_STATUS_RET(GetRealFileName(new_cache_index.cache_file_name), "Get real cache file name failed by[%s].",
                    cache_index_.cache_file_name.c_str());
  if (!new_cache_index.var_desc_file_name.empty()) {
    GE_CHK_STATUS_RET(GetRealFileName(new_cache_index.var_desc_file_name),
                      "Get real cache file name failed by[%s].",
                      cache_index_.var_desc_file_name.c_str());
  }
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

Status ModelCache::CreateIndexFile(const std::string &index_file) {
  auto open_mode = static_cast<mmMode_t>(M_IRUSR | M_IWUSR);
  auto open_flag = M_RDWR | M_CREAT;
  int32_t fd = mmOpen2(index_file.c_str(), open_flag, open_mode);
  GE_CHK_BOOL_RET_STATUS(((fd != EN_ERROR) && (fd != EN_INVALID_PARAM)), FAILED,
                         "Create index file[%s] failed, fd=%d.", index_file.c_str(), fd);
  (void)mmClose(fd);
  return SUCCESS;
}

Status ModelCache::ReadJsonFile(const std::string &file_path, nlohmann::json &json_obj) {
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

Status ModelCache::WriteJsonFile(const std::string &file_path, const nlohmann::json &json_obj) {
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

void from_json(const nlohmann::json &json_obj, CacheFileIdx &cache_file_index) {
  json_obj.at(kJsonFieldGraphKey).get_to(cache_file_index.graph_key);
  json_obj.at(kJsonFieldCacheFileName).get_to(cache_file_index.cache_file_name);
  auto iter = json_obj.find(kJsonFielVarDescFileName);
  if (iter != json_obj.end()) {
    cache_file_index.var_desc_file_name = iter.value().get<std::string>();
  }
}

void to_json(nlohmann::json &json_obj, const CacheFileIdx &cache_file_index) {
  json_obj = nlohmann::json{{kJsonFieldGraphKey, cache_file_index.graph_key},
                            {kJsonFieldCacheFileName, cache_file_index.cache_file_name}};
  if (!cache_file_index.var_desc_file_name.empty()) {
    json_obj[kJsonFielVarDescFileName] = cache_file_index.var_desc_file_name;
  }
}
}  // namespace ge
