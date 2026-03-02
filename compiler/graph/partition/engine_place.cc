/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/engine_place.h"
#include <mutex>

#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/thread_pool.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"
#include "api/gelib/gelib.h"
#include "graph/partition/optimizer/dynamic_data_flow_engine_reassign_pass.h"
#include "graph/partition/optimizer/hostcpu_engine_update_pass.h"

namespace ge {
namespace {
std::mutex check_support_cost_mutex;
constexpr uint32_t kDefaultThreadNum = 16;
}
Status EnginePlacer::Check() const {
  if (compute_graph_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "compute_graph_ is nullptr, check invalid.");
    GELOGE(GE_GRAPH_NULL_INPUT, "[Check][Param] compute_graph_ is nullptr.");
    return FAILED;
  }
  std::shared_ptr<GELib> instance_ptr = GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    REPORT_INNER_ERR_MSG("E19999", "GELib instance is nullptr or it is not InitFlag, check invalid.");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Run enginePlacer failed, because GELib is invalid.");
    return FAILED;
  }
  return SUCCESS;
}

Status EnginePlacer::SelectEngine(const NodePtr &node, const std::set<std::string> &exclude_engines,
                                  bool &is_check_support_success, OpInfo &matched_op_info) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::string engine_name;
  std::string kernel_name;
  // Check if this node has assigned engine
  bool has_engine_attr =
      AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, engine_name) && !engine_name.empty();
  bool has_kernel_attr =
      AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kernel_name) && !kernel_name.empty();

  if (op_desc->GetOpEngineName().empty()) {
    if (has_kernel_attr && has_engine_attr) {
      UpdateOpdescWithAttr(op_desc, engine_name, kernel_name, matched_op_info);
    } else {
      // Call placer cost model to get the "best" engine for this node
      std::string selected_engine_name =
          DNNEngineManager::GetInstance().GetDNNEngineName(node, exclude_engines, matched_op_info);
      // If can't get op's engine name, keep check support finish and return failed
      if (selected_engine_name.empty()) {
        is_check_support_success = false;
        REPORT_PREDEFINED_ERR_MSG(
            "EZ3003", std::vector<const char *>({"opname", "optype"}),
            std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str()}));
        GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[Check][Param] Can not find engine of op name %s type %s",
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
      }
    }
  } else {
    GetOpInfoByOpDesc(op_desc, engine_name, kernel_name, matched_op_info);
    matched_op_info.flagAsync = false;
  }

  // Record the node assigned atomic_engine name
  GELOGD("Assigning DNNEngine %s to node %s, op type %s", op_desc->GetOpEngineName().c_str(), node->GetName().c_str(),
         node->GetType().c_str());
  std::lock_guard<std::mutex> lock(mutex_);
  node_atomic_engine_map_.emplace(node, op_desc->GetOpEngineName());
  return SUCCESS;
}

void EnginePlacer::GetOpInfoByOpDesc(const OpDescPtr &op_desc, const std::string &attr_engine_name,
                                     const std::string &attr_kernel_name, OpInfo &matched_op_info) {
  if (attr_engine_name.empty()) {
    matched_op_info.engine = op_desc->GetOpEngineName();
  } else {
    if (attr_engine_name != op_desc->GetOpEngineName()) {
      GELOGW("[%s] of attr[%s] in op[%s] is not equal to engine[%s]", attr_engine_name.c_str(),
             ATTR_NAME_ENGINE_NAME_FOR_LX.c_str(), op_desc->GetName().c_str(), op_desc->GetOpEngineName().c_str());
    }
  }

  if (attr_kernel_name.empty()) {
    matched_op_info.opKernelLib = op_desc->GetOpKernelLibName();
  } else {
    if (attr_kernel_name != op_desc->GetOpKernelLibName()) {
      GELOGW("[%s] of attr[%s] in op[%s] is not equal to [%s]", attr_kernel_name.c_str(),
             ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX.c_str(), op_desc->GetName().c_str(),
             op_desc->GetOpKernelLibName().c_str());
    }
  }
}

void EnginePlacer::UpdateOpdescWithAttr(const OpDescPtr &op_desc, const std::string &attr_engine_name,
                                        const std::string &attr_kernel_name, OpInfo &matched_op_info) {
  op_desc->SetOpEngineName(attr_engine_name);
  if (op_desc->GetOpKernelLibName().empty()) {
    matched_op_info.opKernelLib = attr_kernel_name;
  } else {
    if (op_desc->GetOpKernelLibName() != attr_kernel_name) {
      GELOGW("kernel[%s] in op[%s] is not equal to [%s] of attr[%s]", op_desc->GetOpKernelLibName().c_str(),
             op_desc->GetName().c_str(), attr_kernel_name.c_str(), ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX.c_str());
    }
  }
}

Status EnginePlacer::RunAllSubgraphs() {
  // Assign engine for each node in the graph
  DNNEngineManager::GetInstance().InitPerformanceStatistic();
  GE_ASSERT_SUCCESS(Run(false), "RunAllSubgraphs failed, graph=%s.", compute_graph_->GetName().c_str());
  DNNEngineManager::GetInstance().LogCheckSupportCost();
  return SUCCESS;
}

Status EnginePlacer::Run(bool direct_node_flag) {
  std::lock_guard<std::mutex> lock(check_support_cost_mutex);
  GELOGD("Engine placer starts.");
  if (Check() != SUCCESS) {
    return FAILED;
  }
  ThreadPool thread_pool("ge_fdengine", kDefaultThreadNum, false);
  std::vector<std::future<Status>> fut_rets;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  const auto &context = GetThreadLocalContext();
  const auto &err_msg_ctx = error_message::GetErrMgrContext();
  std::map<NodePtr, OpInfo> nodes_op_infos;
  for (const auto &node_ptr : compute_graph_->GetNodes(direct_node_flag)) {
    GE_CHECK_NOTNULL(node_ptr);
    auto fut = thread_pool.commit(
        [this, node_ptr, &exclude_engines, &context, &err_msg_ctx, &nodes_op_infos]() -> Status {
          error_message::SetErrMgrContext(err_msg_ctx);
          bool is_check_support_success = true;
          GetThreadLocalContext() = context;
          OpInfo op_info;
          GE_ASSERT_SUCCESS(SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info),
                            "Failed to select engine for [%s][%s].", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
          if ((!op_info.engine.empty()) || (!op_info.opKernelLib.empty())) {
            std::lock_guard<std::mutex> inner_lock(mutex_);
            nodes_op_infos.emplace(node_ptr, op_info);
          }
          GE_ASSERT_TRUE(is_check_support_success);
          return SUCCESS;
        });
    fut_rets.emplace_back(std::move(fut));
  }
  Status result = SUCCESS;
  for (auto &fut : fut_rets) {
    Status ret = SUCCESS;
    // 多线程可能存在异常，增加try catch和相应的日志为了辅助定位
    try {
      ret = fut.get();
    } catch(std::runtime_error &e) {
      GELOGE(INTERNAL_ERROR, "Got an exception, reason[%s].", e.what());
      ret = INTERNAL_ERROR;
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to SelectEngine.");
      result = ret;
    }
  }
  DNNEngineManager::UpdateOpDescsWithOpInfos(nodes_op_infos);
  GELOGD("Engine placer ends.");
  return result;
}

Status EnginePlacer::AssignCompositeEngine() {
  if (OpsKernelManager::GetInstance().GetCompositeEngines().empty()) {
    GELOGI("No composite engine registers, ignore assign composite engine");
    return SUCCESS;
  }
  std::vector<ComputeGraphPtr> subgraphs;
  if (GraphUtils::GetSubgraphsRecursively(compute_graph_, subgraphs) != GRAPH_SUCCESS) {
    GE_CHECK_NOTNULL(compute_graph_);
    GELOGE(FAILED, "[Get][Subgraphs] Get subgraphs contained in graph %s failed", compute_graph_->GetName().c_str());
    return FAILED;
  }
  for (const auto &subgraph : subgraphs) {
    (void)subgraph->DelAttr(ATTR_NAME_COMPOSITE_ENGINE_NAME);
  }
  std::reverse(subgraphs.begin(), subgraphs.end());
  subgraphs.emplace_back(compute_graph_);
  for (const auto &subgraph : subgraphs) {
    for (const auto &node : subgraph->GetDirectNode()) {
      std::string composite_engine_name = DNNEngineManager::GetInstance().GetCompositeEngineName(node, 1);
      GELOGD("Assign composite engine %s to node %s, op type %s", composite_engine_name.c_str(),
             node->GetName().c_str(), node->GetType().c_str());
      node_composite_engine_map_.insert(std::make_pair(node, composite_engine_name));
    }
  }
  return SUCCESS;
}

const NodeEngineMap &EnginePlacer::GetNodeEngineMap(bool is_composite_engine_mode) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_composite_engine_mode ? node_composite_engine_map_ : node_atomic_engine_map_;
}

Status EnginePlacer::RunSinglePass(const std::string &pass_name,
                                   const std::shared_ptr<EngineReAssignPass> &pass) {
  GE_CHECK_NOTNULL(pass);
  Status ret;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ret = pass->Run(compute_graph_, node_atomic_engine_map_, node_composite_engine_map_);
  }
  if (ret == SUCCESS) {
    GELOGD("Engine reassign pass %s return SUCCESS.", pass_name.c_str());
  } else if (ret == NOT_CHANGED) {
    GELOGD("Engine reassign pass %s return NOT_CHANGED.", pass_name.c_str());
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Engine reassign pass %s run failed.", pass_name.c_str());
    GELOGE(ret, "[Call][Run] Engine reassign pass %s failed.", pass_name.c_str());
    return ret;
  }
  return ret;
}

Status EnginePlacer::RunHostcpuEngineUpdatePass() {
  auto hostcpu_pass = MakeShared<HostcpuEngineUpdatePass>();
  return RunSinglePass("HostcpuEngineUpdatePass", hostcpu_pass);
}

Status EnginePlacer::ReAssignEngine() {
  std::vector<std::pair<std::string, std::shared_ptr<EngineReAssignPass>>> passes;
  passes.emplace_back(
      std::make_pair("DynamicDataFlowEngineReassignPass", MakeShared<DynamicDataFlowEngineReassignPass>()));
  passes.emplace_back(std::make_pair("HostcpuEngineUpdatePass", MakeShared<HostcpuEngineUpdatePass>()));

  for (const auto &pass_item : passes) {
    Status ret = RunSinglePass(pass_item.first, pass_item.second);
    if (ret != SUCCESS && ret != NOT_CHANGED) {
      return ret;
    }
  }
  return SUCCESS;
}
}  // namespace ge
