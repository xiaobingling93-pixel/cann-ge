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
#include "execution_order_util.h"
#include "compiler/graph/build/model_cache.h"
#include "compiler/graph/manager/util/graph_rebuild_state_ctrl.h"

namespace ge {

static void to_json(nlohmann::json &json_obj, const SliceGraphInfo &info) {
  json_obj = Json();
  json_obj[kSliceGraphInfoSliceGraphIDKeyName] = info.slice_graph_id;
}

static void to_json(nlohmann::json &json_obj, const SlicingResult &result) {
  json_obj = Json();
  json_obj[kSlicingResultUserGraphKeyKeyName] = result.user_graph_key;
  json_obj[kSlicingResultUserGraphIDKeyName] = result.user_graph_id;
  json_obj[kSlicingResultSliceGraphListKeyName] = result.slice_graph_infos;
}

static void from_json(const nlohmann::json &json_obj, SliceGraphInfo &info) {
  auto iter = json_obj.find(kSliceGraphInfoSliceGraphIDKeyName);
  if (iter != json_obj.end()) {
    info.slice_graph_id = iter.value().get<int64_t>();
  }
}

static void from_json(const nlohmann::json &json_obj, SlicingResult &result) {
  auto iter = json_obj.find(kSlicingResultUserGraphKeyKeyName);
  if (iter != json_obj.end()) {
    result.user_graph_key = iter.value().get<std::string>();
  }
  iter = json_obj.find(kSlicingResultUserGraphIDKeyName);
  if (iter != json_obj.end()) {
    result.user_graph_id = iter.value().get<std::uint32_t>();
  }
  iter = json_obj.find(kSlicingResultSliceGraphListKeyName);
  if (iter != json_obj.end()) {
    result.slice_graph_infos = iter.value().get<std::vector<SliceGraphInfo>>();
  }
}

static Status ReadSlicingResultFromFile(const std::string &slicing_result_file, SlicingResult &slicing_result) {
  nlohmann::json slicing_result_json_obj;
  GE_CHK_STATUS_RET(ModelCache::ReadJsonFile(slicing_result_file, slicing_result_json_obj),
    "Failed to read json file file[%s].", slicing_result_file.c_str());
  try {
    from_json(slicing_result_json_obj, slicing_result);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read slice result file[%s], err msg: %s", slicing_result_file.c_str(), e.what());
    return FAILED;
  }
  GE_CHK_BOOL_RET_STATUS(!slicing_result.slice_graph_infos.empty(), FAILED, "The slicing result is empty.");
  return SUCCESS;
}

static Status SaveSlicingResultToFile(const std::string &slicing_result_file, const SlicingResult &slicing_result) {
  nlohmann::json json_obj;
  try {
    to_json(json_obj, slicing_result);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to create json file: %s", e.what());
    return FAILED;
  }
  GE_CHK_STATUS_RET(ModelCache::WriteJsonFile(slicing_result_file, json_obj));
  return SUCCESS;
}

Status ExecutionOrderUtil::GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key) {
  return ep_util_.GetGuardedExecutionPointGraphKey(gep, gep_graph_key);
}

Status ExecutionOrderUtil::CreateKeyOptionForGuardedExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		                                 const GuardedExecutionPoint *gep, std::map<std::string, std::string> &options) {
  return ep_util_.CreateKeyOptionForGuardedExecutionPoint(root_dir, user_graph_key, gep, options);
}

Status ExecutionOrderUtil::RestoreExecutionOrder(const std::string root_dir, const std::string user_graph_key, ExecutionOrder &order) {
  // user_graph_root_dir example: ./cache_dir/jit/slicing_hierarchy/userGraphKey0/
  const std::string user_graph_root_dir = root_dir + kSlicingHierarchySubDirName + "/" + user_graph_key + "/";
  // do not have previous cache result of this user_graph. Maybe see the graph the first time
  // or previous caching result has been deleted (because they are invalid). Then skip restoration
  if (!ModelCache::CheckFileExist(user_graph_root_dir)) {
    GELOGI("Cannot find the caching directory of the user graph key: %s [path: %s]. Will skip RestoreCache.",
            user_graph_key.c_str(), user_graph_root_dir.c_str());
    return SUCCESS;
  }

  // In subsequent codes, if it returns FAILED, it means the cache results are not valid.
  // slicing_result_file_path example: ./cache_dir/jit/slicing_hierarchy/userGraphKey0/slicing_result.json
  const std::string slicing_result_file_path = user_graph_root_dir + kSlicingResultJsonFileName;
  GE_CHK_BOOL_RET_STATUS(ModelCache::CheckFileExist(slicing_result_file_path), FAILED,
      "Cannot find the slicing result file of the user graph key: %s.",
      user_graph_key.c_str());

   /* deserialize slicing_result.json */
  SlicingResult result;
  GE_CHK_STATUS_RET(ReadSlicingResultFromFile(slicing_result_file_path, result),
		  "Failed to read slicing result json file[%s].", slicing_result_file_path.c_str());

  /* restore the execution points and add them to the EO */
  for (SliceGraphInfo &info : result.slice_graph_infos) {
    std::unique_ptr<ExecutionPoint> exec_point_ptr;
    GE_WARN_ASSERT_GRAPH_SUCCESS(ep_util_.RestoreExecutionPoint(root_dir, user_graph_key, info, exec_point_ptr),
		    "Failed to restore ep.");
    GELOGD("Slice graph %lld restoration success.", exec_point_ptr->GetId());
    order.slice_graphs_.emplace_back(std::move(exec_point_ptr));
  }
  return SUCCESS;
}

Status ExecutionOrderUtil::SaveExecutionOrder(const std::string root_dir, const std::string user_graph_key, uint32_t user_graph_id,
		                              ExecutionOrder &order) {
  // construct user_graph_root_dir example: ./cache_dir/jit/slicing_hierarchy/userGraphKey0/
  const std::string user_graph_root_dir = root_dir + kSlicingHierarchySubDirName + "/" + user_graph_key + "/";
  GE_CHK_STATUS_RET(CreateDirectory(user_graph_root_dir));
  GELOGI("Generated directory: %s for user graph[%u].", user_graph_root_dir.c_str(), user_graph_id);

  /* construct slicing_result.json */
  const std::string slicing_result_file = user_graph_root_dir + kSlicingResultJsonFileName;
  SlicingResult slicing_result;
  slicing_result.user_graph_id = user_graph_id;
  slicing_result.user_graph_key = user_graph_key;

  /* iterate all the slice graphs */
  for (auto &epPtr : order.slice_graphs_) {
    SliceGraphInfo slice_graph_info = {epPtr->GetId(), 0};
    slicing_result.slice_graph_infos.push_back(slice_graph_info); // update slicing_result
    GE_CHK_STATUS_RET(ep_util_.SaveExecutionPoint(root_dir, user_graph_key, epPtr),
		    "Failed to save ep."); // save slice_graph and rem_graph to .pb files
  }

  GE_CHK_STATUS_RET(SaveSlicingResultToFile(slicing_result_file, slicing_result));
  return SUCCESS;
}
} // namespace ge