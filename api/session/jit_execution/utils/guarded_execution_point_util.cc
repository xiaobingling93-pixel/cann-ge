/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include "guarded_execution_point_util.h"
#include "graph/debug/ge_attr_define.h"
#include "compiler/graph/build/model_cache.h"
#include "compiler/graph/manager/util/graph_rebuild_state_ctrl.h"

namespace ge {
namespace {
static Status TryLoadCompiledGraphFromCache(const ComputeGraphPtr &root_graph, ComputeGraphPtr &compiled_graph_cached) {
  ModelCache model_cache;
  GeRootModelPtr root_model;
  GraphRebuildStateCtrl ctrl;
  GE_WARN_ASSERT_GRAPH_SUCCESS(model_cache.Init(root_graph, &ctrl));
  GE_CHK_STATUS_RET(model_cache.TryLoadModelFromCache(root_graph, root_model), "Failed to load model from cache.");
  GE_WARN_ASSERT(root_model != nullptr);
  compiled_graph_cached = root_model->GetRootGraph();
  return SUCCESS;
}
}

static void to_json(nlohmann::json &json_obj, const GuardedExecutionPointInfo &info) {
  json_obj = Json();
  json_obj[kGuardedExecutionPointInfoKeyName] = info.gep_graph_key;
}

static void to_json(nlohmann::json &json_obj, const GuardedExecutionPointInfoList &list) {
  json_obj = Json();
  json_obj[kGuardedExecutionPointListKeyName] = list.gep_list;
}

static Status SaveGepListJsonFile(const std::string &gep_list_file, const GuardedExecutionPointInfoList &gep_info_list) {
  nlohmann::json json_obj;
  try {
    to_json(json_obj, gep_info_list); // transfer gep_info_list to JSON object
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to create gep_list.json file: %s", e.what());
    return FAILED;
  }
  GE_CHK_STATUS_RET(ModelCache::WriteJsonFile(gep_list_file, json_obj)); // write new file
  GELOGI("Write new gep_list.json file in: %s.", gep_list_file.c_str());
  return SUCCESS;
}

static std::string GetCurTimeInNs() {
  const auto cur_time = std::chrono::system_clock::now();
  const auto cur_time_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(cur_time);
  const auto value_ns = cur_time_ns.time_since_epoch().count();
  return std::to_string(value_ns);
}

Status GuardedExecutionPointUtil::AddGuardedExecutionPointGraphKey(std::string root_dir, std::string user_graph_key, const GuardedExecutionPoint *gep) {
  if (user_graph_key.empty() || root_dir.empty()) {
    GELOGD("user_graph_key = %s and root_dir = %s, will not invoke AddGuardedExecutionPointGraphKey.",
      user_graph_key.c_str(), root_dir.c_str());
    return SUCCESS;
  }

  const std::lock_guard<std::mutex> lock(gep_util_mutex_);
  GE_ASSERT_NOTNULL(gep);
  GE_CHK_BOOL_RET_STATUS(gep_to_gep_graph_key_.find(gep) == gep_to_gep_graph_key_.end(), FAILED, "The GEP has been added to CompiledModelCache.");
  const std::string cur_timestamp = GetCurTimeInNs();
  std::string gep_graph_key = user_graph_key + "_" +
    std::to_string(gep->GetOwnerEp()->GetId()) + "_" + cur_timestamp;
  GELOGI("Generate new gep_graph_key: %s", gep_graph_key.c_str());
  gep_to_gep_graph_key_.emplace(std::make_pair(gep, gep_graph_key)); // update gep_to_gep_graph_key
  return SUCCESS;
}

Status GuardedExecutionPointUtil::EmplaceGuardedExecutionPointOption(std::string root_dir, std::string user_graph_key, const GuardedExecutionPoint *gep,
                                                                     std::map<std::string, std::string> &options) {
  if (user_graph_key.empty() || root_dir.empty()) {
    GELOGD("user_graph_key = %s and root_dir = %s, will not invoke EmplaceGuardedExecutionPointOption.",
      user_graph_key.c_str(), root_dir.c_str());
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(gep);
  GE_CHK_BOOL_RET_STATUS(options.find(OPTION_GRAPH_KEY) == options.end(), FAILED, "Current options already have the graphKey option.");
  GE_CHK_BOOL_RET_STATUS(options.find(OPTION_GRAPH_COMPILER_CACHE_DIR) == options.end(), FAILED, "Current options already have the cache_dir option.");
  std::string gep_graph_key;
  GE_CHK_STATUS_RET(GetGuardedExecutionPointGraphKey(gep, gep_graph_key), "Failed to get gep graph key.");
  options.emplace(std::make_pair(OPTION_GRAPH_KEY, gep_graph_key)); // add gep_graph_key to option
  options.emplace(std::make_pair(OPTION_GRAPH_COMPILER_CACHE_DIR, root_dir)); // add cache_dir to option
  return SUCCESS;
}

Status GuardedExecutionPointUtil::HasGuardedExecutionPointRegistered(const GuardedExecutionPoint *gep, bool &has_registered) {
  GE_ASSERT_NOTNULL(gep);
  has_registered = true;
  const std::lock_guard<std::mutex> lock(gep_util_mutex_);
  const auto gep_iter = gep_to_gep_graph_key_.find(gep);
  if (gep_iter == gep_to_gep_graph_key_.end()) {
    has_registered = false;
    GELOGD("The GEP has not been registered in CMC. Should invoke AddGuardedExecutionPointGraphKey to add it");
  }
  return SUCCESS;
}

Status GuardedExecutionPointUtil::GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key) {
  const std::lock_guard<std::mutex> lock(gep_util_mutex_);
  GE_ASSERT_NOTNULL(gep);
  const auto gep_iter = gep_to_gep_graph_key_.find(gep);
  GE_CHK_BOOL_RET_STATUS(gep_iter != gep_to_gep_graph_key_.end(), FAILED, "Get GEP graph key failed. Cannot find the GEP in the CompiledModelCache.");
  gep_graph_key = gep_iter->second;
  return SUCCESS;
}

Status GuardedExecutionPointUtil::RestoreGuardedExecutionPoint(const std::string root_dir, const GuardedExecutionPointInfo &info,
		                                               ExecutionPoint &exec_point, GuardedExecutionPoint *gep) {
  /* Reconstruct the gep_to_gep_graph_key mapping */
  GE_CHK_BOOL_RET_STATUS(gep_to_gep_graph_key_.find(gep) == gep_to_gep_graph_key_.end(), FAILED, "Failed to restore gep_to_gep_graph_key, the gep has existed.");
  gep_to_gep_graph_key_.emplace(gep, info.gep_graph_key);

  /* update the gep_graph_key to the context for the upcoming TryLoadFlowModelFromCache */
  auto cur_graph_options = GetThreadLocalContext().GetAllGraphOptions();
  /**
   * when save cache, call EmplaceGuardedExecutionPointOption before jit executor CompileAndLoad and slice_graph add
   * & flow_model_cache init, at the same, restore cache also need update the graph option before using flow_model_cache.
   **/
  cur_graph_options.erase(OPTION_GRAPH_KEY);
  cur_graph_options.emplace(OPTION_GRAPH_KEY, info.gep_graph_key);
  // update cache_dir to option
  cur_graph_options.erase(OPTION_GRAPH_COMPILER_CACHE_DIR);
  cur_graph_options.emplace(std::make_pair(OPTION_GRAPH_COMPILER_CACHE_DIR, root_dir));
  GetThreadLocalContext().SetGraphOption(cur_graph_options);

  /*
   * 2025.5.28
   * todo: compiled_设成false，在jit_executor中对该gep执行AddGraph和CompileGraph操作，注册graphNode以及从om中读取FlowModel。
   * 优点：
   *    1）实现简单，借用已有CompileGraph流程，无需考虑CompileGraph中对graphNode等对象一系列的状态设置；
   *    2）借用CompileGraph内部的FlowModelCache读取流程，在正确设置options后可以读取对应的FlowModel，跳过编译。
   * 缺点：
   *    1）在om文件会被加载两次（第一次在cmc中仅读取guard函数，第二次在CompileGraph中读取FlowModel；（可能的改进方式：将guard函数
   *    二进制字符串单独存储）；
   *    2）编译缓存为在小图执行时按需从磁盘加载，而不是在初始化RestoreCache时一把从磁盘中读取。
   * 后续可进一步讨论改进方案。
   */
  /* load the flow_model from the .om file, and restore the Guard func stored in the .om file */
  gep->compiled_ = false;
  ComputeGraphPtr compiled_graph_cached;
  GE_WARN_ASSERT_GRAPH_SUCCESS(TryLoadCompiledGraphFromCache(exec_point.GetSlicedGraph(), compiled_graph_cached),
      "Failed to load compiled graph from cache.");
  if (compiled_graph_cached == nullptr) {
    return SUCCESS;
  }
  gep->compiled_graph_ = compiled_graph_cached; // set compiled_graph_ in the gep

  /*
   * 2025.5.28
   * todo: 上一个GE进程中，AddGraph会将图的ATTR_NAME_GRAPH_HAS_BEEN_ADDED属性设为true。结束前，没有地方将ComputeGraph的
   * ATTR_NAME_GRAPH_HAS_BEEN_ADDED属性设为false（考虑可能是bug），导致新GE进程起来并加载该图后ATTR_NAME_GRAPH_HAS_BEEN_ADDED
   * 属性依旧是true，在GraphManager::CheckGraphAdded函数中会出错，报告该图已加载。因此此处需将图的该属性刷成false
   */
  (void)AttrUtils::SetBool(*gep->compiled_graph_, ATTR_NAME_GRAPH_HAS_BEEN_ADDED, false);

  // load guard check func
  GE_CHK_STATUS_RET(gep->matcher_.LoadGuardCheckFunc(compiled_graph_cached));

  return SUCCESS;
}

Status GuardedExecutionPointUtil::SaveGuardedExecutionPoint(std::string user_graph_root_dir, const ExecutionPoint &exec_point,
		                                            const std::vector<std::unique_ptr<GuardedExecutionPoint>> &geps) {
  GuardedExecutionPointInfoList gep_info_list;
  for (auto &gep : geps) { // iterate geps and construct gep_info_list
    auto it = gep_to_gep_graph_key_.find(gep.get());
    GE_CHK_BOOL_RET_STATUS(it != gep_to_gep_graph_key_.end(), FAILED, "Failed to get gep_graph_key.");
    const std::string gep_graph_key = it->second;
    GuardedExecutionPointInfo gep_info = {gep_graph_key}; // construct gep_info
    gep_info_list.gep_list.emplace_back(gep_info);
  }

  // save gep_info_list to the gep_list.json file of this ep
  const std::string sid = std::to_string(exec_point.GetId());
  const std::string slice_graph_sub_dir = user_graph_root_dir + sid + "/";
  GE_CHK_STATUS_RET(CreateDirectory(slice_graph_sub_dir), "Failed to gen directory[%s].", slice_graph_sub_dir.c_str());
  GELOGI("Generated directory: %s.", slice_graph_sub_dir.c_str());

  const std::string gep_list_file = slice_graph_sub_dir + kGEPListJsonFileName;
  GE_CHK_STATUS_RET(SaveGepListJsonFile(gep_list_file, gep_info_list),
		  "Failed to save gep list json file[%s].", gep_list_file.c_str());
  GELOGI("Saved gep_list.json file for slice graph id %lld.", exec_point.GetId());
  return SUCCESS;
}
} // namespace ge