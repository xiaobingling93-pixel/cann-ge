/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_COMMON_CONFIG_NUMA_CONFIG_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_COMMON_CONFIG_NUMA_CONFIG_MANAGER_H_

#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "ge_common/ge_api_types.h"

namespace ge {
struct ItemDeviceInfo {
  int32_t device_id = -1;  // 必选
};

struct ItemDef {
  std::string item_type;  // 必选
  std::string memory;  // 必选
  std::string aic_type;  // 必选
  std::string links_mode;  // 可选
  std::string resource_type; // 可选
  std::vector<ItemDeviceInfo> device_list;  // 可选
};

struct LinkPair {
  int32_t id = -1;  // 必选
  int32_t pair_id = -1;  // 必选
};

struct Plane {
  int32_t plane_id = -1;  // 必选
  std::vector<int32_t> devices;  // 必选
};

struct ItemTopology {
  std::string links_mode;  // 必选
  std::vector<LinkPair> links;  // 必选
};

struct NodeDef {
  std::string node_type;  // 必选
  std::string resource_type; // 可选
  std::string support_links;  // 必选
  std::string item_type;  // 必选
  std::string h2d_bw;  // 可选
  std::vector<ItemTopology> item_topology;  // 可选
};

struct NodesTopology {
  std::string type;  // 必选
  std::string protocol;  // 可选
  std::vector<Plane> topos;  // 必选
};

struct ItemInfo {
  int32_t item_id = -1;  // 必选
  int32_t device_id = -1;
};

struct ClusterNode {
  int32_t node_id = -1;  // 必选
  std::string node_type;  // 必选
  int32_t memory = -1;  // 可选，单位GB
  bool is_local = false;  // 可选
  std::vector<ItemInfo> item_list;  // 必选
};

struct ClusterInfo {
  std::vector<ClusterNode> cluster_nodes;  // 必选
  bool has_nodes_topology = false;
  NodesTopology nodes_topology;  // 可选
};

struct NumaConfig {
  std::vector<ClusterInfo> cluster; // 必选
  std::vector<NodeDef> node_def;  // 必选
  std::vector<ItemDef> item_def;  // 必选
};

class NumaConfigManager {
 public:
  static Status InitNumaConfig();

 private:
  static std::string ToJsonString(const NumaConfig &numa_config);

  static Status InitServerNumaConfig(NumaConfig &numa_config);

  static bool ExportOptionSupported();
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_COMMON_CONFIG_NUMA_CONFIG_MANAGER_H_
