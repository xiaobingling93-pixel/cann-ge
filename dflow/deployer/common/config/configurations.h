/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_COMMON_CONFIG_CONFIGURATIONS_H_
#define AIR_RUNTIME_COMMON_CONFIG_CONFIGURATIONS_H_

#include <cstdint>
#include <string>
#include "ge/ge_api_types.h"

namespace ge {
struct DeviceConfig {
  DeviceType device_type = NPU;
  int32_t device_id = -1;
  int32_t phy_device_id = -1;
  int32_t hcom_device_id = 0;
  int32_t device_index = -1;
  std::string ipaddr;
  std::string resource_type;
  int32_t os_id = 0;
  int64_t super_device_id = -1;
  bool support_hcom = true;
  bool support_flowgw = true;
};

struct NodeConfig {
  int32_t node_id = -1;
  std::string node_type;
  std::string ipaddr;
  int32_t port = -1;
  bool is_local = false;
  std::vector<DeviceConfig> device_list;
  std::string resource_type;
  std::string deploy_res_path;
  bool lazy_connect = true;
  std::vector<int32_t> node_mesh_index;
  uint32_t chip_count = 0U;
  std::string protocol;
  std::vector<int32_t> proxy_device_ids;
  std::string available_ports;
  bool need_port_preemption = false;
  std::string auth_lib_path;
  std::string server_id;
  bool operator<(const NodeConfig& other) const {
    return ipaddr.compare(other.ipaddr);
  }
};

struct NetworkInfo {
  std::string mode;
  std::string mask;
  std::string ipaddr;
  std::string available_ports;
};

struct HostInfo {
  NetworkInfo ctrl_panel;
  NetworkInfo data_panel;
};

struct DeployerConfig {
  HostInfo host_info;
  std::string mode;
  NodeConfig node_config;
  std::vector<NodeConfig> remote_node_config_list;
  std::string working_dir;
};

class Configurations {
 public:
  static Configurations &GetInstance();

  void Finalize();

  Status InitInformation();

  std::vector<NodeConfig> GetAllNodeConfigs() const;

  const std::vector<NodeConfig> &GetRemoteNodeConfigs() const;

  const DeployerConfig& GetHostInformation() const {
    return information_;
  }

  const NodeConfig &GetLocalNode() const;

  std::string GetDeployResDir();

  static Status GetResourceConfigPath(std::string &config_dir);

  static Status GetConfigDir(std::string &config_dir);

  static std::vector<std::string> GetHeterogeneousEnvs();

 private:
  Status GetWorkingDir(std::string &working_dir) const;
  static std::string GetHostDirByEnv();
  DeployerConfig information_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_COMMON_CONFIG_CONFIGURATIONS_H_
