/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_AUTO_DEPLOYER_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_AUTO_DEPLOYER_H

#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"

namespace ge {
using AutoDeployFunc = Status (*)(const std::map<std::string, std::string> &,
                                  const ComputeGraphPtr &);

class DataFlowGraphAutoDeployer {
 public:
  static Status AutoDeployDataFlowGraph(const DataFlowGraph &data_flow_graph, const std::string &deploy_info_path);
  static Status UpdateFlowFuncDeployInfo(const DataFlowGraph &data_flow_graph);

 private:
  static Status GetConfigDeployInfo(std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
    std::map<std::string, std::pair<uint32_t, uint32_t>> &device_id_to_mem_cfg,
    bool &dynamic_schedule_enable,
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> &invoke_deploy_map,
    const std::string &deploy_info_str);
  static Status GetDeployLogicDeviceForInvoke(
    std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> &invoke_deploy_map,
    const std::vector<CompileConfigJson::InvokeDeployInfo> &invoke_deploy_infos,
    const std::string &flow_node_name, const bool dynamic_schedule_enable);
  static Status GetDeployLogicDeviceFromBatchInfo(
      std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
      std::set<std::string> &logic_device_ids,
      std::map<std::string, std::vector<std::pair<std::string, std::string>>> &invoke_deploy_map,
      const std::vector<CompileConfigJson::FlowNodeBatchDeployInfo> &batch_deploy_info_list,
      const bool dynamic_schedule_enable);
  static Status GetExpandLogicDeviceIds(const std::string &origin_logic_device_id_list,
                                        std::string &resolved_logic_device_id_list);
  static Status GetSortedLogicDeviceIds(const std::string &origin_logic_device_id_list,
                                        std::string &resolved_logic_device_id_list);
  static Status CheckAndExpandLogicDeviceIds(const std::string &logic_device_id_list,
                                             std::vector<std::string> &expand_logic_device_id_list);
  static Status ExpandRangeConfig(std::vector<CompileConfigJson::FlowNodeBatchDeployInfo> &batch_deploy_info_list);
  /**
   * @brief expand an range config to singe logic device id list.
   * example:config "0:1~3:0~1" expand result is{0:1:0,0:2:0,0:3:0,0:1:1,0:2:1,0:3:1}.
   * @param logic_device_id_range range config
   * @param logic_device_ids single device id list
   * @return Status Success:success, other failed.
   */
  static Status ExpandToSingleLogicDevice(const std::string &logic_device_id_range,
                                          std::vector<std::string> &logic_device_ids);
  static Status UpdateFlowNodeSubGraphDeployInfo(const OpDescPtr &flow_node_op_desc, const DataFlowGraph &graph);
  static Status UpdateFlowNodeSubGraphDeployInfo(const OpDescPtr &flow_node_op_desc, const DataFlowGraph &graph,
                                                 bool is_redundant);
  static Status SelectResourceType(const std::vector<std::string> &runnable_resources_type,
                                   const std::string &logic_device_id, std::string &resources_type, bool is_heavy_load);
  static Status CheckAndProcessMemCfg(const std::vector<CompileConfigJson::FlowNodeBatchMemCfg> &mem_size_cfg,
                                      const std::set<std::string> &logic_dev_ids,
                                      std::map<std::string, std::pair<uint32_t, uint32_t>> &device_id_to_mem_cfg);
  static Status SetMemCfgRecord(const uint32_t &std_mem_size, const uint32_t &shared_mem_size,
                                const std::vector<std::string> &expand_logic_device_ids,
                                std::map<std::string, std::pair<uint32_t, uint32_t>> &device_id_to_mem_cfg);
  static Status ExpandDeployInfoStr(const std::string &deploy_info_str, bool is_sub_dataflow,
                                    std::string &resolved_logic_device_id_list,
                                    std::string &resolved_redundant_logic_device_id_list, bool &is_expand);
  static Status GetNodeDeployName(const DataFlowGraph &data_flow_graph, const OpDescPtr &op_desc,
                                  std::string &node_deploy_name, int32_t depth);
  static Status GetInvokeDeployInfos(const std::vector<std::pair<std::string, std::string>> &invoke_names,
                    const std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
                    std::vector<std::string> &invoke_deploy_infos);
  static Status HandleInvokedSubgraph(const ComputeGraphPtr &subgraph, const size_t logic_dev_num,
                                      const std::string &subgraph_deploy_info, bool is_redundant,
                                      std::string &assign_logic_device_id);
};
}  // namespace ge
#endif  // AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_AUTO_DEPLOYER_H
