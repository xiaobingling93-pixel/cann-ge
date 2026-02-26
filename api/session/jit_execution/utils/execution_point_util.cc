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
#include "execution_point_util.h"
#include "compiler/graph/build/model_cache.h"
#include "compiler/graph/manager/util/graph_rebuild_state_ctrl.h"

namespace ge {
static void from_json(const nlohmann::json &json_obj, GuardedExecutionPointInfo &info) {
  auto iter = json_obj.find(kGuardedExecutionPointInfoKeyName);
  if (iter != json_obj.end()) {
    info.gep_graph_key = iter.value().get<std::string>();
  }
}

static void from_json(const nlohmann::json &json_obj, GuardedExecutionPointInfoList &list) {
  auto iter = json_obj.find(kGuardedExecutionPointListKeyName);
  if (iter != json_obj.end()) {
    list.gep_list = iter.value().get<std::vector<GuardedExecutionPointInfo>>();
  }
}

static Status ReadGEPListFromFile(const std::string &gep_list_file, GuardedExecutionPointInfoList &gep_info_list) {
  nlohmann::json json_obj;
  GE_CHK_STATUS_RET(ModelCache::ReadJsonFile(gep_list_file, json_obj), "Failed to read gep_list file[%s]", gep_list_file.c_str());
  try {
    from_json(json_obj, gep_info_list);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to read gep_list file[%s], err msg: %s", gep_list_file.c_str());
    return FAILED;
  }
  if (gep_info_list.gep_list.empty()) {
    GELOGE(FAILED, "The gep_list is empty");
    return FAILED;
  }
  return SUCCESS;
}

Status ExecutionPointUtil::GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key) {
  return gep_util_.GetGuardedExecutionPointGraphKey(gep, gep_graph_key);
}

Status ExecutionPointUtil::CreateKeyOptionForGuardedExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		                                 const GuardedExecutionPoint *gep, std::map<std::string, std::string> &options) {
  bool has_gep_registered;
  GE_CHK_STATUS_RET(gep_util_.HasGuardedExecutionPointRegistered(gep, has_gep_registered));
  if (!has_gep_registered) { // add the new gep to cmc
    GE_CHK_STATUS_RET(gep_util_.AddGuardedExecutionPointGraphKey(root_dir, user_graph_key, gep));
    GE_ASSERT_NOTNULL(gep);
    const ExecutionPoint *ep = gep->GetOwnerEp();
    updated_eps_.insert(ep); // update updated_eps
  }
  GE_ASSERT_SUCCESS(gep_util_.EmplaceGuardedExecutionPointOption(root_dir, user_graph_key, gep, options));
  return SUCCESS;
}

Status ExecutionPointUtil::RestoreExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		                                 const SliceGraphInfo &slice_graph_info, std::unique_ptr<ExecutionPoint> &exec_point_ptr) {
  // slice_graph_sub_dir example: ./cache_dir/jit/slicing_hierarchy/userGraphKey0/1/
  const std::string slice_graph_sub_dir = root_dir + kSlicingHierarchySubDirName + "/" + user_graph_key + "/" +
	                                  std::to_string(slice_graph_info.slice_graph_id) + "/";
  GE_CHK_BOOL_RET_STATUS(ModelCache::CheckFileExist(slice_graph_sub_dir), FAILED,
    "Cannot find the caching directory of the slice graph ID: %ld, the path is: %s",
    slice_graph_info.slice_graph_id, slice_graph_sub_dir.c_str());

  // gep_list_file_path example: ./cache_dir/jit/slicing_hierarchy/userGraphKey0/1/gep_list.json
  const std::string gep_list_file_path = slice_graph_sub_dir + kGEPListJsonFileName;
  GE_CHK_BOOL_RET_STATUS(ModelCache::CheckFileExist(gep_list_file_path), FAILED, "Cannot find the gep list file, the path is: %s", gep_list_file_path.c_str());

  /* read and deserialize gep_list.json */
  GuardedExecutionPointInfoList gep_info_list;
  GE_CHK_STATUS_RET(ReadGEPListFromFile(gep_list_file_path, gep_info_list));

  // restore the slice garph and remaining graph from slice_graph_sub_dir
  // slice_graph_file example:  ./cache_dir/jit/slicing_hierarchy/userGraphKey0/1/slice_graph.pb
  // rem_graph_file example:  ./cache_dir/jit/slicing_hierarchy/userGraphKey0/1/rem_graph.pb
  Graph slice_graph_tmp, remaining_graph_tmp;
  ComputeGraphPtr slice_graph = nullptr;
  ComputeGraphPtr remaining_graph = nullptr;
  const std::string slice_graph_file = slice_graph_sub_dir + kSliceGraphPbFileName;
  const std::string rem_graph_file = slice_graph_sub_dir + kRemGraphPbFileName;
  GE_CHK_BOOL_RET_STATUS(ModelCache::CheckFileExist(slice_graph_file), FAILED, "The slice graph file[%s] does not exist.", slice_graph_file.c_str());
  GE_CHK_GRAPH_STATUS_RET(slice_graph_tmp.LoadFromFile(slice_graph_file.c_str())); // load .pb file and generate Graph
  GELOGD("Loading slice_graph.pb success.");
  slice_graph = GraphUtilsEx::GetComputeGraph(slice_graph_tmp); // generate the compute_graph

  /* Graph sedes does not save output size, need to compute again
   * todo: do we need to restore more attrs?
   */
  auto netout_node = slice_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  slice_graph->SetOutputSize(netout_node->GetInDataNodesSize());

  if (ModelCache::CheckFileExist(rem_graph_file)) {
    GE_CHK_GRAPH_STATUS_RET(remaining_graph_tmp.LoadFromFile(rem_graph_file.c_str())); // load .pb file and generate Graph
    GELOGD("Loading rem_graph.pb success.");
    remaining_graph = GraphUtilsEx::GetComputeGraph(remaining_graph_tmp); // generate the compute_graph
  } else {
    GELOGD("The remaining graph file[%s] does not exist.", rem_graph_file.c_str());
  }

  /* construct the ExecutionPoint, EP */
  exec_point_ptr = MakeUnique<ExecutionPoint>(slice_graph_info.slice_graph_id, slice_graph, remaining_graph);
  GE_ASSERT_NOTNULL(exec_point_ptr, "Failed to create execution point.");

  // restore the GEPs in this EP
  for (const GuardedExecutionPointInfo &info: gep_info_list.gep_list) {
    GuardedExecutionPoint *gep = new GuardedExecutionPoint(&(*exec_point_ptr));
    GE_ASSERT_NOTNULL(gep);
    GE_WARN_ASSERT_GRAPH_SUCCESS(gep_util_.RestoreGuardedExecutionPoint(root_dir, info, *exec_point_ptr, gep));
    (*exec_point_ptr).models_.GetCache().emplace_back(gep);
    GELOGD("Restoring GEP success, gep_graph_key = %s", info.gep_graph_key.c_str());
  }
  return SUCCESS;
}

Status ExecutionPointUtil::SaveExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		                              const std::unique_ptr<ExecutionPoint> &exec_point_ptr) {
  // construct slice_graph_id_x dir
  // slice_graph_dir example: ./root_dir/jit/slicing_hierarchy/userGraphKey0/1/
  const std::string sid = std::to_string((*exec_point_ptr).GetId());
  const std::string slice_graph_dir = root_dir + kSlicingHierarchySubDirName + "/" + user_graph_key + "/" + sid + "/";
  GE_CHK_STATUS_RET(CreateDirectory(slice_graph_dir));
  GELOGI("Generated directory: %s for user graph[%u].", slice_graph_dir.c_str());

  // save slice_graph.pb
  ComputeGraphPtr slice_graph_ptr = (*exec_point_ptr).GetSlicedGraph();
  if (slice_graph_ptr == nullptr) {
    GELOGI("The slice graph is nullptr, skip save cache.");
    return SUCCESS;
  }
  const std::string slice_graph_file = slice_graph_dir + kSliceGraphPbFileName;
  const Graph slice_graph = GraphUtilsEx::CreateGraphFromComputeGraph(slice_graph_ptr);
  if (!ModelCache::CheckFileExist(slice_graph_file)) {
    GE_CHK_STATUS_RET(slice_graph.SaveToFile(slice_graph_file.c_str()),
                      "Save slice graphs to pb failed, file_path:%s", slice_graph_file.c_str());
    GELOGI("Saved slice_graph.pb to dir: %s", slice_graph_file.c_str());
  } else {
    GELOGI("slice_graph.pb already exists in %s, skip file generation.", slice_graph_file.c_str());
  }

  // save rem_graph.pb
  ComputeGraphPtr rem_graph_ptr = (*exec_point_ptr).GetRemainingGraph();
  if (rem_graph_ptr != nullptr) {
    const std::string rem_graph_file = slice_graph_dir + kRemGraphPbFileName;
    const Graph rem_graph = GraphUtilsEx::CreateGraphFromComputeGraph(rem_graph_ptr);
    if (!ModelCache::CheckFileExist(rem_graph_file)) {
      GE_CHK_STATUS_RET(rem_graph.SaveToFile(rem_graph_file.c_str()),
                      "Save remaining graphs to pb failed, file_path:%s", rem_graph_file.c_str());
      GELOGI("Saved rem_graph.pb to dir: %s", rem_graph_file.c_str());
    } else {
      GELOGI("rem_graph.pb already exists in %s, skip file generation.", rem_graph_file.c_str());
    }
  } else {
    GELOGD("The remaining graph is nullptr.");
  }

  std::string user_graph_root_dir = root_dir + kSlicingHierarchySubDirName + "/" + user_graph_key + "/";
  if (updated_eps_.count(exec_point_ptr.get()) <= 0) {
    GELOGI("Slice graph with id %ld does not change, skip it.", exec_point_ptr->GetId());
    return SUCCESS;
  }
  // the ep has newly generated geps
  auto &geps = exec_point_ptr->models_.GetCache();
  GE_CHK_BOOL_RET_STATUS(!geps.empty(), FAILED, "EP with slice_id %ld has newly generated GEPs, but the guard cache is empty.", exec_point_ptr->GetId());
  GELOGI("Update slice graph with id %lld", exec_point_ptr->GetId());
  GE_CHK_STATUS_RET(gep_util_.SaveGuardedExecutionPoint(user_graph_root_dir, *exec_point_ptr, geps));
  return SUCCESS;
}
} // namespace ge