/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_parser.h"
#include <exception>
#include <nlohmann/json.hpp>
#include "runtime/dev.h"
#include "runtime/config.h"
#include "framework/common/debug/ge_log.h"
#include "common/debug/log.h"
#include "dflow/base/utils/process_utils.h"
#include "common/subprocess/subprocess_manager.h"
#include "json_parser.h"
#include "common/utils/deploy_location.h"
#include "deploy/resource/device_info.h"
#include "graph/ge_context.h"

namespace ge {
namespace {
const std::string kConfigNetName = "eth0";
const std::string kConfigCluster = "cluster";
const std::string kConfigClusterNodes = "cluster_nodes";
const std::string kConfigNodeDef = "node_def";
const std::string kConfigItemDef = "item_def";
const std::string kConfigResourceType = "resource_type";
const std::string kConfigItem = "item";
const std::string kConfigItemType = "item_type";
const std::string kConfigNodeType = "node_type";
const std::string kConfigIsLocal = "is_local";
const std::string kConfigAuthLibPath = "auth_lib_path";
const std::string kConfigIpaddr = "ipaddr";
const std::string kConfigDataPanel = "data_panel";
const std::string kConfigAvailPorts = "avail_ports";
const std::string kConfigPort = "port";
const std::string kConfigDeviceList = "item_list";
const std::string kConfigItemId = "item_id";
const std::string kConfigDeviceId = "device_id";
const std::string kConfigNodeId = "node_id";
const std::string kConfigProtocol = "protocol";
const std::string kConfigDeployResPath = "deploy_res_path";
const std::string kConfigNodesTopology = "nodes_topology";
const std::string kNodesTopologyType = "star";
const std::string kResoureTypeX86 = "X86";
const std::string kResoureTypeAscend = "Ascend";
const std::string kResoureTypeAarch = "Aarch";
const std::string kProtocolTypeTcp = "TCP";
const std::string kProtocolTypeRdma = "RDMA";
const std::string kDefaultAvailPorts = "16666~32767";
const char_t *const kNetworkModeCtrlDefaultPorts = "10023";
const char_t *const kNetworkModeDataDefaultPorts = "18000~22000";

template <typename T>
inline void AssignOptionalField(T &varible, const string &key, const nlohmann::json &json_read) {
  auto iter = json_read.find(key);
  if (iter != json_read.end()) {
    varible = iter.value().get<T>();
  }
}

template <typename T>
inline void AssignOptionalField(T &varible,
                                const string &key,
                                const nlohmann::json &json_read,
                                const T &default_value) {
  auto iter = json_read.find(key);
  if (iter != json_read.end()) {
    try {
      varible = iter.value().get<T>();
    }  catch (const nlohmann::json::exception &e) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Failed to read config[%s] from json[%s], err msg: %s.",
             key.c_str(), iter.value().dump().c_str(), e.what());
      throw e;
    }
  } else {
    varible = default_value;
  }
}

template <typename T>
inline void AssignRequiredField(T &varible, const string &key, const nlohmann::json &json_read) {
  try {
    json_read.at(key).get_to(varible);
  }  catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Failed to read config[%s], err msg: %s.", key.c_str(), e.what());
    throw e;
  }
}
}

struct ClusterNodesTopology {
  std::string protocol = kProtocolTypeRdma;
};

struct ClusterConfig {
  std::vector<NodeConfig> node_list;
  ClusterNodesTopology nodes_topology;
};

struct ItemDefConfig {
  std::string item_type;
  std::string resource_type;
};

struct ItemConfig {
  std::vector<int32_t> item_id_list;
  std::string item_type;
};

struct NodeDefConfig {
  std::string node_type;
  std::string resource_type;
  std::vector<ItemConfig> item_config_list;
};

static void from_json(const nlohmann::json &j, ItemDefConfig &item_def_config) {
  AssignOptionalField(item_def_config.item_type, kConfigItemType, j);
  AssignOptionalField(item_def_config.resource_type, kConfigResourceType, j, kResoureTypeAscend);
}

static void from_json(const nlohmann::json &j, ItemConfig &item_config) {
  AssignOptionalField(item_config.item_type, kConfigItemType, j);
}

static void from_json(const nlohmann::json &j, NodeDefConfig &node_def_config) {
  AssignOptionalField(node_def_config.node_type, kConfigNodeType, j);
  auto default_resource_type = DeployLocation::IsX86() ? kResoureTypeX86 : kResoureTypeAarch;
  AssignOptionalField(node_def_config.resource_type, kConfigResourceType, j, default_resource_type);
  AssignOptionalField(node_def_config.item_config_list, kConfigItem, j);
}

static void from_json(const nlohmann::json &j, ClusterNodesTopology &nodes_topology) {
  AssignOptionalField(nodes_topology.protocol, kConfigProtocol, j, kProtocolTypeRdma);
  const auto &pos = nodes_topology.protocol.find(':');
  nodes_topology.protocol = nodes_topology.protocol.substr(0, pos);
}

static void from_json(const nlohmann::json &j, ClusterConfig &cluster_config) {
  AssignRequiredField(cluster_config.node_list, kConfigClusterNodes, j);
  std::sort(cluster_config.node_list.begin(), cluster_config.node_list.end(),
            [](const NodeConfig &lhs, const NodeConfig &rhs) -> bool {
              return (lhs.node_id < rhs.node_id);
            });
  AssignOptionalField(cluster_config.nodes_topology, kConfigNodesTopology, j);
}

static void from_json(const nlohmann::json &j, NetworkInfo &network_info) {
  AssignOptionalField(network_info.available_ports, kConfigAvailPorts, j, kDefaultAvailPorts);
}

static void from_json(const nlohmann::json &j, NodeConfig &node_config) {
  AssignRequiredField(node_config.node_id, kConfigNodeId, j);
  AssignRequiredField(node_config.node_type, kConfigNodeType, j);
  AssignRequiredField(node_config.ipaddr, kConfigIpaddr, j);
  AssignRequiredField(node_config.port, kConfigPort, j);
  AssignOptionalField(node_config.is_local, kConfigIsLocal, j);
  AssignOptionalField(node_config.auth_lib_path, kConfigAuthLibPath, j);
  AssignOptionalField(node_config.deploy_res_path, kConfigDeployResPath, j);
  NetworkInfo network = {};
  network.available_ports = kDefaultAvailPorts;
  AssignOptionalField(network, kConfigDataPanel, j);
  node_config.available_ports = network.available_ports;
  GEEVENT("Get node config success, node_id = %d, ipaddr = %s, port = %d, is_local = %d, auth_lib_path %s",
         node_config.node_id,
         node_config.ipaddr.c_str(),
         node_config.port,
         static_cast<int32_t>(node_config.is_local),
         node_config.auth_lib_path.empty() ? "is not set, skip authentication"
                                           : (std::string("= ") + node_config.auth_lib_path).c_str());
  AssignRequiredField(node_config.device_list, kConfigDeviceList, j);
  int32_t os_id = 0;
  for (size_t i = 0; i < node_config.device_list.size(); ++i) {
    auto &device_config = node_config.device_list[i];
    device_config.device_index = static_cast<int32_t>(i);
    device_config.os_id = os_id++;
    node_config.proxy_device_ids.emplace_back(device_config.device_id);
  }
  DeviceConfig host_cpu = {};
  host_cpu.device_type = CPU;
  host_cpu.ipaddr = node_config.ipaddr;
  host_cpu.device_id = 0;
  host_cpu.device_index = -1;
  host_cpu.os_id = os_id++;
  host_cpu.support_hcom = false;
  host_cpu.phy_device_id = 0;
  node_config.device_list.emplace_back(host_cpu);
}

static void from_json(const nlohmann::json &j, DeviceConfig &device_config) {
  AssignRequiredField(device_config.ipaddr, kConfigIpaddr, j);
  AssignRequiredField(device_config.device_id, kConfigItemId, j);
  AssignRequiredField(device_config.phy_device_id, kConfigDeviceId, j);
  device_config.hcom_device_id = device_config.device_id;
  GELOGI("Get device config success, ipaddr = %s, device_id = %d, phy_device_id = %d",
         device_config.ipaddr.c_str(),
         device_config.device_id,
         device_config.phy_device_id);
}

void ConfigParser::InitNetWorkInfo(DeployerConfig &deployer_config) {
  deployer_config.host_info.ctrl_panel.ipaddr = deployer_config.node_config.ipaddr;
  deployer_config.host_info.ctrl_panel.available_ports = kNetworkModeCtrlDefaultPorts;
  deployer_config.host_info.data_panel.ipaddr = deployer_config.node_config.ipaddr;
  deployer_config.host_info.data_panel.available_ports = kNetworkModeDataDefaultPorts;
}

Status ConfigParser::GetResourceType(const std::vector<NodeDefConfig> &node_defs,
                                     const std::vector<ItemDefConfig> &item_defs,
                                     std::map<std::string, std::string> &node_type_to_node_resource_type,
                                     std::map<std::string, std::string> &node_type_to_item_resource_type) {
  std::map<std::string, std::string> item_type_to_resource_type;
  const static std::set<std::string> kSupportResourceTypeList = {kResoureTypeX86,
                                                                 kResoureTypeAscend,
                                                                 kResoureTypeAarch};
  for (const auto &item_def : item_defs) {
    const auto &resource_type = item_def.resource_type;
    const auto &it = kSupportResourceTypeList.find(resource_type);
    GE_CHK_BOOL_RET_STATUS(it != kSupportResourceTypeList.cend(), ACL_ERROR_GE_PARAM_INVALID,
                           "The resourceType[%s] of config is not supported, only support %s, %s or %s.",
                           resource_type.c_str(),
                           kResoureTypeX86.c_str(),
                           kResoureTypeAscend.c_str(),
                           kResoureTypeAarch.c_str());
    item_type_to_resource_type[item_def.item_type] = resource_type;
    GELOGI("Parse item resource success, item type = %s, resource type = %s",
           item_def.item_type.c_str(), item_def.resource_type.c_str());
  }
  for (const auto &node_def : node_defs) {
    const auto &resource_type = node_def.resource_type;
    const auto &type_it = kSupportResourceTypeList.find(resource_type);
    GE_CHK_BOOL_RET_STATUS(type_it != kSupportResourceTypeList.cend(), ACL_ERROR_GE_PARAM_INVALID,
                           "The resourceType[%s] of config is not supported, only support %s, %s or %s.",
                           resource_type.c_str(),
                           kResoureTypeX86.c_str(),
                           kResoureTypeAscend.c_str(),
                           kResoureTypeAarch.c_str());
    node_type_to_node_resource_type[node_def.node_type] = resource_type;
    GELOGI("Parse node resource success, node type = %s, resource type = %s",
           node_def.node_type.c_str(), resource_type.c_str());
    for (const auto &item : node_def.item_config_list) {
      const auto &it = item_type_to_resource_type.find(item.item_type);
      // all item are the same resource type
      if (it != item_type_to_resource_type.cend()) {
        node_type_to_item_resource_type[node_def.node_type] = it->second;
        GELOGI("Parse resource success, node type = %s, item resource type = %s",
               node_def.node_type.c_str(), it->second.c_str());
      }
    }
  }
  return SUCCESS;
}

Status ConfigParser::InitNodeResourceType(const std::map<std::string, std::string> &node_type_to_node_resource_type,
                                          const std::map<std::string, std::string> &node_type_to_item_resource_type,
                                          NodeConfig &node_config) {
  const auto &node_it = node_type_to_node_resource_type.find(node_config.node_type);
  node_config.resource_type = DeployLocation::IsX86() ? kResoureTypeX86 : kResoureTypeAarch;
  if (node_it != node_type_to_node_resource_type.cend()) {
    node_config.resource_type = node_it->second;
  }
  std::string item_resource_type = kResoureTypeAscend;
  const auto &item_it = node_type_to_item_resource_type.find(node_config.node_type);
  if (item_it != node_type_to_item_resource_type.cend()) {
    item_resource_type = item_it->second;
  }
  for (auto &device_config : node_config.device_list) {
    if (device_config.device_type == CPU) {
      device_config.resource_type = node_config.resource_type;
    } else {
      device_config.resource_type = item_resource_type;
    }
  }
  return SUCCESS;
}

Status ConfigParser::CheckProtocolType(const std::string &protocol) {
  const static std::set<std::string> kSupportProtocolList = {kProtocolTypeRdma, kProtocolTypeTcp};
  const auto &it = kSupportProtocolList.find(protocol);
  GE_CHK_BOOL_RET_STATUS(it != kSupportProtocolList.cend(), ACL_ERROR_GE_PARAM_INVALID,
                         "The protocol[%s] of config is not supported, only support %s or %s.",
                         protocol.c_str(),
                         kProtocolTypeRdma.c_str(),
                         kProtocolTypeTcp.c_str());
  GELOGI("Parse protocol success, type = %s.", protocol.c_str());
  return SUCCESS;
}

Status ConfigParser::InitAllNodeConfig(const std::vector<ClusterConfig> &clusters,
                                       const std::vector<NodeDefConfig> &node_defs,
                                       const std::vector<ItemDefConfig> &item_defs,
                                       std::vector<NodeConfig> &node_configs) {
  std::map<std::string, std::string> node_type_to_node_resource_type;
  std::map<std::string, std::string> node_type_to_item_resource_type;
  GE_CHK_STATUS_RET(GetResourceType(node_defs,
                                    item_defs,
                                    node_type_to_node_resource_type,
                                    node_type_to_item_resource_type),
                    "Failed to get resource type.");
  bool local_node_matched = false;
  int32_t cluster_index = -1;
  int32_t node_id = 0;
  std::map<std::string, uint32_t> node_to_index;
  int32_t support_host_flowgw = 0;
  for (const auto &cluster : clusters) {
    cluster_index++;
    int32_t node_index = -1;
    const auto &protocol = cluster.nodes_topology.protocol;
    for (const auto &node : cluster.node_list) {
      node_index++;
      auto new_node = node;
      if (new_node.is_local && new_node.device_list.size() > 1) {
        const auto device_id = new_node.device_list[0].device_id;
        (void) rtGetDeviceCapability(device_id, RT_MODULE_TYPE_SYSTEM, FEATURE_TYPE_MEMQ_EVENT_CROSS_DEV,
                                     &support_host_flowgw);
        bool has_flowgw = false;
        GE_CHK_STATUS_RET(SubprocessManager::HasFlowGw(has_flowgw), "Failed to check has flowgw");
        GEEVENT("Check has host flowgw success, support_flowgw = %d, has_flowgw = %d.",
                support_host_flowgw, static_cast<int32_t>(has_flowgw));
        support_host_flowgw = static_cast<int32_t>((support_host_flowgw != 0) && has_flowgw);
      }
      GE_CHK_STATUS_RET(CheckProtocolType(protocol), "Failed to check protocol.");
      GE_CHK_STATUS_RET(InitNodeResourceType(node_type_to_node_resource_type,
                                             node_type_to_item_resource_type,
                                             new_node),
                        "Failed to init node resource type.");
      new_node.node_id = node_id++;
      new_node.node_mesh_index.emplace_back(cluster_index);
      new_node.node_mesh_index.emplace_back(node_index);
      new_node.protocol = protocol;
      if (node.is_local) {
        local_node_matched = true;
      }
      node_to_index[node.ipaddr] = node_configs.size();
      node_configs.emplace_back(new_node);
    }
  }
  for (auto &node : node_configs) {
    int32_t tmp_host_flowgw = support_host_flowgw;
    if (!node.is_local && node.device_list.size() == 1 && node.device_list[0].device_type == CPU) {
      tmp_host_flowgw = 0;
    }
    for (auto &dev_config : node.device_list) {
      dev_config.support_flowgw = (dev_config.device_type == CPU && tmp_host_flowgw == 0) ? false : true;
    }
  }

  if (!local_node_matched) {
    std::string local_addr;
    GE_CHK_STATUS_RET_NOLOG(ProcessUtils::GetIpaddr(kConfigNetName, local_addr));
    const auto &it = node_to_index.find(local_addr);
    GE_CHK_BOOL_RET_STATUS(it != node_to_index.cend(), ACL_ERROR_GE_PARAM_INVALID,
                           "Get local node failed, local ipaddr = %s.", local_addr.c_str());
    node_configs[it->second].is_local = true;
  }
  return SUCCESS;
}

Status ConfigParser::InitDeployerConfig(const std::vector<ClusterConfig> &clusters,
                                        const std::vector<NodeDefConfig> &node_defs,
                                        const std::vector<ItemDefConfig> &item_defs,
                                        DeployerConfig &deployer_config) {
  std::vector<NodeConfig> node_configs;
  GE_CHK_STATUS_RET(InitAllNodeConfig(clusters, node_defs, item_defs, node_configs),
                    "Failed to init all node config.");
  for (const auto &node : node_configs) {
    if (node.is_local) {
      deployer_config.node_config = node;
    } else {
      deployer_config.remote_node_config_list.emplace_back(node);
    }
    GELOGI("Get node success, node index = %s, is_local = %d",
           ToString(node.node_mesh_index).c_str(),
           static_cast<int32_t>(node.is_local));
  }
  InitNetWorkInfo(deployer_config);
  return SUCCESS;
}

Status ConfigParser::ParseServerInfo(const std::string &file_path, DeployerConfig &deployer_config) {
  GE_CHK_BOOL_RET_STATUS(!file_path.empty(), ACL_ERROR_GE_PARAM_INVALID, "File path is null.");
  GELOGI("Get config json path[%s]successfully", file_path.c_str());

  nlohmann::json json_config;
  GE_CHK_STATUS_RET(JsonParser::ReadConfigFile(file_path, json_config),
                    "Read config file:%s failed",
                    file_path.c_str());
  std::vector<ClusterConfig> clusters;
  std::vector<NodeDefConfig> node_defs;
  std::vector<ItemDefConfig> item_defs;
  try {
    auto clusters_json = json_config.find(kConfigCluster.c_str());
    GE_CHK_BOOL_RET_STATUS(clusters_json != json_config.end(), ACL_ERROR_GE_PARAM_INVALID,
                           "Json config[%s] is empty.", kConfigCluster.c_str());
    clusters = clusters_json->get<std::vector<ClusterConfig>>();
    auto node_defs_json = json_config.find(kConfigNodeDef.c_str());
    if (node_defs_json != json_config.end()) {
      node_defs = node_defs_json->get<std::vector<NodeDefConfig>>();
    }
    auto item_defs_json = json_config.find(kConfigItemDef.c_str());
    if (item_defs_json != json_config.end()) {
      item_defs = item_defs_json->get<std::vector<ItemDefConfig>>();
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid json, file path:%s, exception:%s", file_path.c_str(), e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_CHK_STATUS_RET_NOLOG(InitDeployerConfig(clusters, node_defs, item_defs, deployer_config));
  return SUCCESS;
}

Status ConfigParser::ParseTopologyLinks(const nlohmann::json &json_link, LinkPair &link_pair) {
  const auto &link_vec = json_link.get<std::vector<int32_t>>();
  GE_CHK_BOOL_RET_STATUS(link_vec.size() == 2U, ACL_ERROR_GE_PARAM_INVALID, "Invalid link pair");
  link_pair.id = link_vec[0];
  link_pair.pair_id = link_vec[1];
  return SUCCESS;
}

Status ConfigParser::ParseClusterNode(const nlohmann::json &json_cluster_node,
                                      const int32_t local_node_id,
                                      ClusterNode &cluster_node) {
  try {
    cluster_node.node_id = json_cluster_node.at("node_id").get<int32_t>();
    cluster_node.node_type = json_cluster_node.at("node_type").get<std::string>();
    if (cluster_node.node_id == local_node_id) {
      cluster_node.is_local = true;
    }
    if (json_cluster_node.contains("memory")) {
      cluster_node.memory = json_cluster_node.at("memory").get<int32_t>();
    }
    nlohmann::json json_item_list = json_cluster_node.at("item_list").get<nlohmann::json>();
    for (const auto &json_item : json_item_list) {
      ItemInfo item_info;
      item_info.item_id = json_item.at("item_id").get<int32_t>();
      item_info.device_id = json_item.at("device_id").get<int32_t>();
      cluster_node.item_list.emplace_back(item_info);
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid json, exception:%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status ConfigParser::ParseTopologyPlanes(const nlohmann::json &json_topo, Plane &plane) {
  plane.plane_id = json_topo.at("plane_id").get<int32_t>();
  plane.devices = json_topo.at("devices").get<std::vector<int32_t>>();
  return SUCCESS;
}

Status ConfigParser::ParseNodesTopology(const nlohmann::json &json_topology, NodesTopology &nodes_topology) {
  try {
    if (json_topology.contains("type")) {
      nodes_topology.type = json_topology.at("type").get<std::string>();
    } else {
      nodes_topology.type = kNodesTopologyType;
    }

    if (json_topology.contains("protocol")) {
      nodes_topology.protocol = json_topology.at("protocol").get<std::string>();
    }
    nlohmann::json json_topos = json_topology.at("topos").get<nlohmann::json>();
    for (const auto &json_topo : json_topos) {
      Plane plane;
      GE_CHK_STATUS_RET_NOLOG(ParseTopologyPlanes(json_topo, plane));
      nodes_topology.topos.emplace_back(plane);
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid json, exception:%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status ConfigParser::ParseClusterInfo(const nlohmann::json &json_config, ClusterInfo &cluster_info) {
  try {
    nlohmann::json json_cluster = json_config.at("cluster").get<nlohmann::json>();
    for (const auto &json_cluster_nodes : json_cluster) {
      nlohmann::json json_cluster_info = json_cluster_nodes.at("cluster_nodes").get<nlohmann::json>();
      const auto local_node_id = Configurations::GetInstance().GetHostInformation().node_config.node_id;
      for (const auto &json_cluster_node : json_cluster_info) {
        ClusterNode cluster_node;
        GE_CHK_STATUS_RET_NOLOG(ParseClusterNode(json_cluster_node, local_node_id, cluster_node));
        cluster_info.cluster_nodes.emplace_back(cluster_node);
      }
      if (json_cluster_nodes.contains("nodes_topology")) {
        nlohmann::json json_nodes_topology = json_cluster_nodes.at("nodes_topology").get<nlohmann::json>();
        GE_CHK_STATUS_RET_NOLOG(ParseNodesTopology(json_nodes_topology, cluster_info.nodes_topology));
        cluster_info.has_nodes_topology = true;
      }
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid json, exception:%s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "Invalid json, exception:%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status ConfigParser::ParseNodeDef(const nlohmann::json &json_config, std::vector<NodeDef> &nodes_def) {
  try {
    nlohmann::json json_nodes_def = json_config.at("node_def").get<nlohmann::json>();
    for (const auto &json_node_def : json_nodes_def) {
      NodeDef node_def;
      node_def.node_type = json_node_def.at("node_type").get<std::string>();
      if (json_node_def.contains("resource_type")) {
        node_def.resource_type = json_node_def.at("resource_type").get<std::string>();
      }
      node_def.support_links = json_node_def.at("support_links").get<std::string>();
      node_def.item_type = json_node_def.at("item_type").get<std::string>();
      if (json_node_def.contains("h2d_bw")) {
        node_def.h2d_bw = json_node_def.at("h2d_bw").get<std::string>();
      }
      if (json_node_def.contains("item_topology")) {
        nlohmann::json json_item_topology = json_node_def.at("item_topology").get<nlohmann::json>();
        for (const auto &json_item : json_item_topology) {
          ItemTopology item_topology;
          item_topology.links_mode = json_item.at("links_mode").get<std::string>();
          nlohmann::json json_links = json_item.at("links").get<nlohmann::json>();
          for (const auto &json_link : json_links) {
            LinkPair link_pair;
            GE_CHK_STATUS_RET_NOLOG(ParseTopologyLinks(json_link, link_pair));
            item_topology.links.emplace_back(link_pair);
          }
          node_def.item_topology.emplace_back(item_topology);
        }
      }
      nodes_def.emplace_back(node_def);
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid json, exception:%s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "Invalid json, exception:%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status ConfigParser::ParseItemDef(const nlohmann::json &json_config, std::vector<ItemDef> &items_def) {
  try {
    nlohmann::json json_items_def = json_config.at("item_def").get<nlohmann::json>();
    for (const auto &json_item_def : json_items_def) {
      ItemDef item_def;
      item_def.item_type = json_item_def.at("item_type").get<std::string>();
      if (json_item_def.contains("resource_type")) {
        item_def.resource_type = json_item_def.at("resource_type").get<std::string>();
      }
      item_def.memory = json_item_def.at("memory").get<std::string>();
      item_def.aic_type = json_item_def.at("aic_type").get<std::string>();
      items_def.emplace_back(item_def);
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Invalid json, exception:%s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "Invalid json, exception:%s", e.what());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status ConfigParser::InitNumaConfig(const std::string &file_path, NumaConfig &numa_config) {
  GE_CHK_BOOL_RET_STATUS(!file_path.empty(), ACL_ERROR_GE_PARAM_INVALID, "File path is null.");
  GELOGI("Get config json path[%s]successfully", file_path.c_str());

  nlohmann::json json_config;
  GE_CHK_STATUS_RET(JsonParser::ReadConfigFile(file_path, json_config),
                    "Read config file:%s failed",
                    file_path.c_str());
  ClusterInfo cluster_info;
  GE_CHK_STATUS_RET(ParseClusterInfo(json_config, cluster_info), "Failed to parse cluster info from json file:%s",
                    file_path.c_str());
  numa_config.cluster.emplace_back(cluster_info);
  std::vector<NodeDef> nodes_def;
  GE_CHK_STATUS_RET(ParseNodeDef(json_config, nodes_def), "Failed to parse node def from json file:%s",
                    file_path.c_str());
  numa_config.node_def = std::move(nodes_def);
  std::vector<ItemDef> items_def;
  GE_CHK_STATUS_RET(ParseItemDef(json_config, items_def), "Failed to parse item def from json file:%s",
                    file_path.c_str());
  numa_config.item_def = std::move(items_def);
  return SUCCESS;
}
}  // namespace ge
