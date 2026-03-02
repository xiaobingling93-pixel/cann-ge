/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/manager/engine_manager/dnnengine_manager.h"

#include <cstdio>
#include <map>

#include "framework/common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "framework/common/debug/ge_log.h"
#include "analyzer/analyzer.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "api/gelib/gelib.h"
#include "base/err_msg.h"

namespace {
const char *const kSchedulerUnits = "schedule_units";
const char *const kId = "id";
const char *const kName = "name";
const char *const kExAttrs = "ex_attrs";
const char *const kIndependent = "independent";
const char *const kSkipAssignStream = "skip_assign_stream";
const char *const kCalEngines = "cal_engines";
const char *const kAttach = "attach";
const char *const kVectorCore = "VectorCore";
const char *const kVectorEngine = "VectorEngine";
const char *const kAIcoreEngine = "AIcoreEngine";
const char *const kHostCpuEngineName = "DNN_VM_HOST_CPU";
const char *const kHostCpuOpKernelLibName = "DNN_VM_HOST_CPU_OP_STORE";
const char *const kNoSupportReason = "Dynamic shape is not supported on this chip!";

const std::map<std::string, std::string> kAcceleratorToEngineName = {
    {"AiCore", "AIcoreEngine"},
    {"AiVector", "VectorEngine"},
    {"Dsa", "DSAEngine"},
    {"Dvpp", "DNN_VM_DVPP"},
    {"AiCpuAscend", "DNN_VM_AICPU_ASCEND"},
    {"AiCpu", "DNN_VM_AICPU"},
    {"HostCpu", "DNN_VM_HOST_CPU"},
    {"Hccl", "DNN_HCCL"},
    {"FftsPlus", "ffts_plus"}};
}  // namespace

namespace ge {
namespace {
const char *const kGetDNNEngineObjs = "GetDNNEngineObjs";
const char *const kInvalidCompositeEngineName = "InvalidCompositeEngineName";
constexpr uint32_t kMaxRecursiveDepth = 25U;

bool ExecOnHostCpu(const OpDescPtr &op_desc) {
  if (!ge::GetContext().GetHostExecFlag()) {
    return false;
  }
  const std::string type = op_desc->GetType();
  bool is_not_cpu_op = OpTypeUtils::IsDataNode(type) || (type == ge::CONSTANT) || (type == ge::CONSTANTOP) ||
                       OpTypeUtils::IsVarLikeNode(type) || (type == ge::FILECONSTANT) || (type == ge::NETOUTPUT) ||
                       (type == ge::PARTITIONEDCALL);
  return (!is_not_cpu_op);
}
}  // namespace

DNNEngineManager::DNNEngineManager() : init_flag_(false) {}
DNNEngineManager::~DNNEngineManager() {
  engines_attrs_map_.clear();
  schedulers_.clear();
  engines_map_.clear();
  atomic_2_composite_.clear();
  checksupport_cost_.clear();
}

DNNEngineManager &DNNEngineManager::GetInstance() {
  static DNNEngineManager instance;
  return instance;
}

Status DNNEngineManager::Initialize(const std::map<std::string, std::string> &options) {
  // Multiple initializations are not supported
  if (init_flag_) {
    GELOGW("DNNEngineManager has been initialized.");
    return SUCCESS;
  }

  // Load engine so
  std::string so_path = "plugin/nnengine/";
  std::string path = GetModelPath();
  path.append(so_path);
  std::string so_api_func = kGetDNNEngineObjs;
  std::vector<std::string> so_func{so_api_func};
  Status status = plugin_mgr_.Load(path, so_func);
  if (status != SUCCESS) {
    GELOGE(status, "[Load][EngineSo]Failed, lib path %s", path.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Load engine so failed, lib path %s", path.c_str());
    return status;
  }

  status = plugin_mgr_.InvokeAll<std::map<std::string, DNNEnginePtr> &>(so_api_func, engines_map_);
  if (status != SUCCESS) {
    GELOGE(status, "[Get][DNNEngineObjs]Failed, so_api_func %s", so_api_func.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Get DNNEngineObjs failed, so_api_func %s", so_api_func.c_str());
    return status;
  }

  GELOGI("The number of DNNEngineObjs is %zu.", engines_map_.size());

  // Engines initialize
  for (auto iter = engines_map_.cbegin(); iter != engines_map_.cend(); ++iter) {
    if (iter->second == nullptr) {
      GELOGI("Engine: %s point to nullptr", (iter->first).c_str());
      continue;
    }

    GELOGI("DNNEngine name: %s.", (iter->first).c_str());

    const uint64_t start = ge::GetCurrentTimestamp();
    status = iter->second->Initialize(options);
    const uint64_t end = ge::GetCurrentTimestamp();
    GEEVENT("[GEPERFTRACE] The time cost of DNNEngineManager::Initialize[%s] is [%lu] micro seconds.",
            (iter->first).c_str(), (end - start));
    if (status != SUCCESS) {
      GELOGE(status, "[Init][Engine]Failed, engine %s", (iter->first).c_str());
      REPORT_INNER_ERR_MSG("E19999", "Initialize engine %s failed", (iter->first).c_str());
      return status;
    }


    // Check engines' attribute
    DNNEngineAttribute attrs;
    iter->second->GetAttributes(attrs);
    if (attrs.runtime_type == RuntimeType::DEVICE) {
      if ((attrs.mem_type.size()) != 1 ||
          ((attrs.mem_type.size() > 1) && (attrs.mem_type[0] != GE_ENGINE_ATTR_MEM_TYPE_HBM))) {
        GELOGE(GE_ENG_MEMTYPE_ERROR, "[Check][Param]Engine %s in aicore, but the memory type is "
               "not HBM, mem_type_size %lu", (iter->first).c_str(), attrs.mem_type.size());
        REPORT_INNER_ERR_MSG("E19999", "Engine %s in aicore, but the memory type is not HBM, mem_type_size %lu",
                           (iter->first).c_str(), attrs.mem_type.size());
        return GE_ENG_MEMTYPE_ERROR;
      }
    }
  }

  status = ParserJsonFile();
  if (status != SUCCESS) {
    GELOGE(status, "[Parse][JsonFile]Failed");
    return status;
  }

  status = CheckJsonFile();
  if (status != SUCCESS) {
    GELOGE(status, "[Check][JsonFile]Failed");
    return status;
  }

  init_flag_ = true;

  return SUCCESS;
}

Status DNNEngineManager::Finalize() {
  // Finalize is not allowed, initialize first is necessary
  if (!init_flag_) {
    GELOGW("DNNEngineManager has been finalized.");
    return SUCCESS;
  }

  for (auto iter = engines_map_.cbegin(); iter != engines_map_.cend(); ++iter) {
    if (iter->second != nullptr) {
      GELOGI("DNNEngine name: %s.", (iter->first).c_str());
      Status status = iter->second->Finalize();
      if (status != SUCCESS) {
        GELOGE(status, "[Finalize][Engine]Failed, engine %s", (iter->first).c_str());
        REPORT_INNER_ERR_MSG("E19999", "Finalize engine %s failed", (iter->first).c_str());
        return status;
      }
    }
  }
  init_flag_ = false;
  engines_map_.clear();
  atomic_2_composite_.clear();
  engines_attrs_map_.clear();
  schedulers_.clear();
  std::lock_guard<std::mutex> lock(mutex_);
  checksupport_cost_.clear();
  return SUCCESS;
}

std::shared_ptr<ge::DNNEngine> DNNEngineManager::GetEngine(const std::string &name) const {
  auto iter = engines_map_.find(name);
  if (iter != engines_map_.end()) {
    return iter->second;
  }

  GELOGW("Failed to get engine object by engine name. %s.", name.c_str());
  return nullptr;
}

bool DNNEngineManager::IsEngineRegistered(const std::string &name) {
  std::map<std::string, DNNEnginePtr>::const_iterator iter = engines_map_.find(name);
  if (iter != engines_map_.cend()) {
    return true;
  }
  GELOGW("Engine:[%s] is not registered", name.c_str());
  return false;
}

void DNNEngineManager::InitPerformanceStatistic() {
  std::lock_guard<std::mutex> lock(mutex_);
  checksupport_cost_.clear();
}

void DNNEngineManager::LogCheckSupportCost() const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &it : checksupport_cost_) {
    GEEVENT("The time cost of %s::CheckSupported is [%lu] micro seconds.", it.first.c_str(), it.second);
  }
}

void DNNEngineManager::GetOpInfos(std::vector<OpInfo> &op_infos, const OpDescPtr &op_desc,
                                  bool &is_op_specified_engine) const {
  std::string engine_name;
  std::string kernel_name;
  bool has_engine_attr = AttrUtils::GetStr(op_desc, ATTR_NAME_OP_SPECIFIED_ENGINE_NAME, engine_name) &&
                         !engine_name.empty();
  bool has_kernel_attr = AttrUtils::GetStr(op_desc, ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME, kernel_name) &&
                         !kernel_name.empty();
  is_op_specified_engine = has_engine_attr && has_kernel_attr;
  if (is_op_specified_engine) {
    OpInfo temp_op_info{engine_name, kernel_name, 0, false, false, false, "", ""};
    GELOGD("GetDNNEngineName:get op specified engine:%s and kernel:%s", engine_name.c_str(), kernel_name.c_str());
    op_infos.emplace_back(std::move(temp_op_info));
  } else {
    // make sure op_infos has been copy
    std::lock_guard<std::mutex> lock(mutex_);
    op_infos = OpsKernelManager::GetInstance().GetOpsKernelInfo(op_desc->GetType());
  }
}

void DNNEngineManager::GetExcludeEngines(std::set<std::string> &exclude_engines) {
  // option CORE_TYPE
  std::string ge_core_type;
  const auto ret = ge::GetContext().GetOption(ge::CORE_TYPE, ge_core_type);
  const std::string &exclude_core_Type = (ge_core_type == kVectorCore) ? kAIcoreEngine : kVectorEngine;
  if (ret == GRAPH_SUCCESS) {
    GELOGI("option [%s] is set to [%s], engine type will exclude %s",
           CORE_TYPE.c_str(), ge_core_type.c_str(), exclude_core_Type.c_str());
  }

  // option EXCLUDE_ENGINES
  const auto &graph_option = GetThreadLocalContext().GetAllGraphOptions();
  const auto it = graph_option.find(EXCLUDE_ENGINES);
  if (it != graph_option.end()) {
    GetExcludeEngines(it->second, exclude_engines);
  }
  exclude_engines.insert(exclude_core_Type);
}

void DNNEngineManager::GetExcludeEngines(const std::string &option, std::set<std::string> &engine_names) {
  std::ostringstream oss;
  auto accelerators = StringUtils::Split(option, '|');
  for (size_t i = 0U; i < accelerators.size(); ++i) {
    const auto &after_trim = StringUtils::Trim(accelerators[i]);
    if (after_trim.empty()) {
      continue;
    }
    const auto iter = kAcceleratorToEngineName.find(after_trim);
    if (iter != kAcceleratorToEngineName.end()) {
      engine_names.insert(iter->second);
      oss << iter->second << ",";
    } else {
      GELOGW("can not get engine name. please check value of option %s. option value[%s], failed engine[%s] ",
             EXCLUDE_ENGINES.c_str(), option.c_str(), after_trim.c_str());
    }
  }
  GELOGI("graph option %s is [%s], engine name is [%s]",
         EXCLUDE_ENGINES.c_str(), option.c_str(), oss.str().c_str());
}

void DNNEngineManager::UpdateOpDescWithOpInfo(const OpDescPtr &op_desc, const OpInfo &op_info) {
  if (!op_info.opKernelLib.empty()) {
    op_desc->SetOpKernelLibName(op_info.opKernelLib);
  }
  if (ExecOnHostCpu(op_desc)) {
    return;
  }
  // set attrs for taking information when load txt to graph object
  if (op_info.flagAsync) {
    GELOGD("Set aicpu blocking op:%s attribute(is_blocking_op):true", op_desc->GetName().c_str());
    (void)AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
  }
  (void) AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, op_info.engine);
  (void) AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, op_info.opKernelLib);
}

void DNNEngineManager::UpdateOpDescsWithOpInfos(const std::map<NodePtr, OpInfo> &nodes_op_infos) {
  for (const auto &it : nodes_op_infos) {
    const auto &op_desc = it.first->GetOpDesc();
    const auto &op_info = it.second;
    UpdateOpDescWithOpInfo(op_desc, op_info);
  }
}

std::string DNNEngineManager::GetDNNEngineName(const ge::NodePtr &node_ptr) {
  std::set<std::string> exclude_engines;
  GetExcludeEngines(exclude_engines);
  OpInfo matched_op_info;
  const auto &engine = GetDNNEngineName(node_ptr, exclude_engines, matched_op_info);
  if (!engine.empty()) {
    UpdateOpDescWithOpInfo(node_ptr->GetOpDesc(), matched_op_info);
  }
  return engine;
}

std::string DNNEngineManager::GetDNNEngineName(const ge::NodePtr &node_ptr,
                                               const std::set<std::string> &exclude_engines,
                                               OpInfo &matched_op_info) {
  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GE_CLI_GE_NOT_INITIALIZED, "DNNEngineManager: node_ptr is nullptr");
                  return "");
  auto op_desc = node_ptr->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GE_CLI_GE_NOT_INITIALIZED, "DNNEngineManager: op_desc is nullptr");
                  return "");
  // Use the OpsKernelManager to get the opInfos for this opCode
  std::vector<OpInfo> op_infos;
  bool is_op_specified_engine = false;
  GetOpInfos(op_infos, op_desc, is_op_specified_engine);
  if (op_infos.empty()) {
    GELOGI("DNNEngineManager: Can not get op info by op type %s", op_desc->GetType().c_str());
    return "";
  }
  GE_IF_BOOL_EXEC(ExecOnHostCpu(op_desc), return GetHostCpuEngineName(op_infos, op_desc, matched_op_info));
  std::map<std::string, std::string> unsupported_reasons;
  for (const auto &it : op_infos) {
    if ((exclude_engines.find(it.engine) != exclude_engines.end()) && (!is_op_specified_engine)) {
      continue;
    }
    const auto &kernel_name = it.opKernelLib;
    auto kernel_info_store = OpsKernelManager::GetInstance().GetOpsKernelInfoStore(kernel_name);
    if (kernel_info_store == nullptr) {
      GELOGW("DNNEngineManager:Can not find any supported ops kernel info store by kernel_name %s, op type is %s, "
             "op name is %s", kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
      return "";
    }
    std::string unsupported_reason;
    // It will be replaced by engine's check support
    uint64_t start_time = GetCurrentTimestamp();
    CheckSupportFlag flag = CheckSupportFlag::kDefault;
    bool check_support_res = true;
    // 多线程可能存在异常，增加try catch和相应的日志为了辅助定位
    try {
      check_support_res = kernel_info_store->CheckSupported(node_ptr, unsupported_reason, flag);
    } catch (std::runtime_error &e) {
      std::ostringstream oss;
      oss << "got an exception, op type[" << op_desc->GetType() << "],ops kernel[" << kernel_name << "], reason["
          << e.what() << "]";
      GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "%s", oss.str().c_str());
      unsupported_reasons.emplace(kernel_name, oss.str());
      break;
    }
    if (check_support_res) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        checksupport_cost_[kernel_name] += GetCurrentTimestamp() - start_time;
      }
      op_desc->SetOpEngineName(it.engine);
      matched_op_info = it;
      GELOGD("DNNEngineManager:Set kernel_lib %s, atomic engine %s, to node %s", kernel_name.c_str(), it.engine.c_str(),
             op_desc->GetName().c_str());
      return it.engine;
    } else {
      if (flag == CheckSupportFlag::kNotSupportDynamicShape) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ3002", std::vector<const char *>({"optype", "opskernel", "reason"}),
            std::vector<const char *>({op_desc->GetType().c_str(), kernel_name.c_str(), kNoSupportReason}));
        GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[Check][OpSupported]Op type %s of kernel %s is unsupported, reason: %s",
               op_desc->GetType().c_str(), kernel_name.c_str(), kNoSupportReason);
        return "";
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        checksupport_cost_[kernel_name] += GetCurrentTimestamp() - start_time;
      }
      unsupported_reasons.emplace(kernel_name, unsupported_reason);
      GELOGI("DNNEngineManager: op not support, kernel_name is %s, op type is %s, op name is %s",
             kernel_name.c_str(), op_desc->GetType().c_str(), op_desc->GetName().c_str());
      if (!op_desc->HasAttr("_is_ge_op")) {
        REPORT_PREDEFINED_ERR_MSG("W11001", std::vector<const char *>({"opname"}),
                                  std::vector<const char *>({op_desc->GetName().c_str()}));
      }
    }
  }

  // concat unsupported reasons analyzed data selection
  std::string reason;
  for (const auto &it : unsupported_reasons) {
    reason += it.first + ":" + it.second + ";";
    REPORT_PREDEFINED_ERR_MSG(
        "EZ3002", std::vector<const char *>({"optype", "opskernel", "reason"}),
        std::vector<const char *>({op_desc->GetType().c_str(), it.first.c_str(), it.second.c_str()}));
    GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED,
           "[Check][OpSupported]Op type %s of ops kernel %s "
           "is unsupported, reason : %s",
           op_desc->GetType().c_str(), it.first.c_str(), it.second.c_str());
  }

  auto root_graph = ge::GraphUtils::FindRootGraph(node_ptr->GetOwnerComputeGraph());
  if (root_graph != nullptr) {
    analyzer::DataInfo analyze_info{root_graph->GetSessionID(), root_graph->GetGraphID(),
                                    analyzer::CHECKSUPPORT, node_ptr, reason};
    // do not change original process
    (void)Analyzer::GetInstance()->DoAnalyze(analyze_info);
  }

  REPORT_PREDEFINED_ERR_MSG(
      "EZ3003", std::vector<const char *>({"opname", "optype"}),
      std::vector<const char *>({op_desc->GetName().c_str(), op_desc->GetType().c_str()}));
  GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED, "[Get][DNNEngineName]Can't find any supported ops kernel "
         "and engine of %s, type is %s",
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return "";
}

std::string DNNEngineManager::GetCompositeEngineName(const ge::NodePtr &node_ptr, uint32_t recursive_depth) {
  // op_desc of node should not be null
  const auto &op_desc = node_ptr->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(GE_CLI_GE_NOT_INITIALIZED, "DNNEngineManager: op_desc is nullptr");
                  return "");
  if (recursive_depth > kMaxRecursiveDepth) {
    REPORT_INNER_ERR_MSG("E19999", "Get CompositeEngineName will be terminated because too many nesting levels(%u) of "
                                 "subgraphs, last node is %s", recursive_depth, op_desc->GetName().c_str());
    GELOGE(PARAM_INVALID,
           "[Check][Param] Get CompositeEngineName will be terminated because too many nesting levels(%u) of subgraphs, "
           "last node is %s", recursive_depth, op_desc->GetName().c_str());
    return "";
  }

  if (OpsKernelManager::GetInstance().GetCompositeEngines().empty() ||
      OpsKernelManager::GetInstance().GetCompositeEngineKernelLibNames().empty()) {
    return "";
  }

  // composite engine name exist
  std::string composite_engine_name;
  (void)AttrUtils::GetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_NAME, composite_engine_name);
  std::string composite_engine_kernel_lib_name;
  (void)AttrUtils::GetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME, composite_engine_kernel_lib_name);
  if (!composite_engine_name.empty() && !composite_engine_kernel_lib_name.empty()) {
    return composite_engine_name;
  }

  bool recursive_mode = (op_desc->GetType() == PARTITIONEDCALL)
      ? (op_desc->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) || op_desc->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH))
      : !op_desc->GetSubgraphInstanceNames().empty();
  GELOGD("Node: %s, Recursive mode: %d", op_desc->GetName().c_str(), static_cast<int32_t>(recursive_mode));
  return recursive_mode ? GetCompositeEngine(node_ptr, recursive_depth) : GetCompositeEngine(node_ptr);
}

std::string DNNEngineManager::GetCompositeEngine(const NodePtr &node) {
  std::string composite_engine_name;
  if (IsNoTask(node)) {
    bool in_diff_flag = false;
    std::string in_composite_engine_name = kInvalidCompositeEngineName;
    for (const auto &in_node : node->GetInAllNodes()) {
      std::string tmp_composite_engine_name;
      (void)AttrUtils::GetStr(in_node->GetOpDesc(), ATTR_NAME_COMPOSITE_ENGINE_NAME, tmp_composite_engine_name);
      if (in_composite_engine_name == kInvalidCompositeEngineName) {
        in_composite_engine_name = tmp_composite_engine_name;
      } else if (in_composite_engine_name != tmp_composite_engine_name) {
        in_diff_flag = true;
        break;
      }
    }
    if (!in_diff_flag &&
        (in_composite_engine_name != kInvalidCompositeEngineName) &&
        !in_composite_engine_name.empty()) {
      composite_engine_name = in_composite_engine_name;
    }
  }
  // op_desc of node should not be null
  const auto &op_desc = node->GetOpDesc();
  if (composite_engine_name.empty()) {
    auto atomic_engine_name = op_desc->GetOpEngineName().empty() ? GetDNNEngineName(node) : op_desc->GetOpEngineName();
    composite_engine_name = (op_desc->GetType() == NETOUTPUT) ? "" : GetCompositeEngineName(atomic_engine_name);
  }
  const auto &composite_engine_kernel_lib_name = GetCompositeEngineKernelLibName(composite_engine_name);
  if (composite_engine_name.empty() || composite_engine_kernel_lib_name.empty()) {
    (void)op_desc->DelAttr(ATTR_NAME_COMPOSITE_ENGINE_NAME);
    (void)op_desc->DelAttr(ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME);
  } else {
    GELOGI("Assign composite engine %s, kernel lib name %s for node %s.", composite_engine_name.c_str(),
           composite_engine_kernel_lib_name.c_str(), op_desc->GetName().c_str());
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_NAME, composite_engine_name);
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME, composite_engine_kernel_lib_name);
  }
  return composite_engine_name;
}

std::string DNNEngineManager::GetCompositeEngine(const NodePtr &func_node, uint32_t recursive_depth) {
  // op_desc of node should not be null
  const auto &op_desc = func_node->GetOpDesc();
  bool graph_diff_composite_engine_flag = false;
  std::string graph_composite_engine_name = kInvalidCompositeEngineName;
  std::vector<ComputeGraphPtr> subgraphs;
  if (NodeUtils::GetDirectSubgraphs(func_node, subgraphs) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get subgraphs of node %s failed", op_desc->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] Get subgraphs of node %s failed", op_desc->GetName().c_str());
    return "";
  }
  for (const auto &subgraph : subgraphs) {
    std::string cur_graph_composite_engine_name = GetCompositeEngine(subgraph, recursive_depth);
    if (graph_composite_engine_name == kInvalidCompositeEngineName) {
      graph_composite_engine_name = cur_graph_composite_engine_name;
    } else if (graph_composite_engine_name != cur_graph_composite_engine_name) {
      graph_diff_composite_engine_flag = true;
      break;
    }
  }

  std::string composite_engine_name;
  std::string composite_engine_kernel_lib_name = GetCompositeEngineKernelLibName(graph_composite_engine_name);
  if (!graph_diff_composite_engine_flag &&
      (graph_composite_engine_name != kInvalidCompositeEngineName) &&
      !graph_composite_engine_name.empty() &&
      !composite_engine_kernel_lib_name.empty()) {
    composite_engine_name = graph_composite_engine_name;
    GELOGI("Assign composite engine %s, kernel lib name %s for node %s.", composite_engine_name.c_str(),
           composite_engine_kernel_lib_name.c_str(), op_desc->GetName().c_str());
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_NAME, composite_engine_name);
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME, composite_engine_kernel_lib_name);
  } else {
    (void)op_desc->DelAttr(ATTR_NAME_COMPOSITE_ENGINE_NAME);
    (void)op_desc->DelAttr(ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME);
  }

  return composite_engine_name;
}

std::string DNNEngineManager::GetCompositeEngine(const ComputeGraphPtr &subgraph, uint32_t recursive_depth) {
  std::string graph_composite_engine_name;
  (void)AttrUtils::GetStr(subgraph, ATTR_NAME_COMPOSITE_ENGINE_NAME, graph_composite_engine_name);
  // if subgraph has been assigned
  if (!graph_composite_engine_name.empty()) {
    return graph_composite_engine_name;
  }

  bool node_diff_composite_engine_flag = false;
  std::string node_composite_engine_name = kInvalidCompositeEngineName;
  uint32_t assigned_node_num = 0;
  for (const auto &cur_node : subgraph->GetDirectNode()) {
    if (IsNoTask(cur_node) || cur_node->GetType() == NETOUTPUT) {
      continue;
    }
    assigned_node_num++;
    std::string cur_node_composite_engine_name = GetCompositeEngineName(cur_node, recursive_depth + 1);
    if (node_composite_engine_name == kInvalidCompositeEngineName) {
      node_composite_engine_name = cur_node_composite_engine_name;
    } else if (node_composite_engine_name != cur_node_composite_engine_name) {
      node_diff_composite_engine_flag = true;
      break;
    }
  }
  if (assigned_node_num == 0) {
    GELOGD("all nodes in subgraph %s belongs to ge_local engine", subgraph->GetName().c_str());
    return "";
  }
  if (!node_diff_composite_engine_flag &&
      (node_composite_engine_name != kInvalidCompositeEngineName) &&
      !node_composite_engine_name.empty()) {
    GELOGI("Assign composite engine %s for subgraph %s.", node_composite_engine_name.c_str(), subgraph->GetName().c_str());
    (void)AttrUtils::SetStr(subgraph, ATTR_NAME_COMPOSITE_ENGINE_NAME, node_composite_engine_name);
    graph_composite_engine_name = node_composite_engine_name;
  } else {
    (void)subgraph->DelAttr(ATTR_NAME_COMPOSITE_ENGINE_NAME);
  }

  return graph_composite_engine_name;
}

std::string DNNEngineManager::GetCompositeEngineName(const std::string &atomic_engine_name) {
  if (atomic_2_composite_.empty()) {
    InitAtomicCompositeMapping();
  }
  const std::map<std::string, std::string>::const_iterator &iter = atomic_2_composite_.find(atomic_engine_name);
  if (iter == atomic_2_composite_.cend()) {
    GELOGW("Composite engine which contains atomic engine %s is not registered", atomic_engine_name.c_str());
    return "";
  }
  return iter->second;
}

std::string DNNEngineManager::GetCompositeEngineKernelLibName(const std::string &composite_engine_name) const {
  const auto &composite_engine_2_kernel_lib_name = OpsKernelManager::GetInstance().GetCompositeEngineKernelLibNames();
  const auto &iter = composite_engine_2_kernel_lib_name.find(composite_engine_name);
  if (iter == composite_engine_2_kernel_lib_name.end()) {
    GELOGW("Kernel lib name of composite engine %s is not registered", composite_engine_name.c_str());
    return "";
  }
  return iter->second;
}

std::string DNNEngineManager::GetHostCpuEngineName(const std::vector<OpInfo> &op_infos,
                                                   const OpDescPtr &op_desc,
                                                   OpInfo &matched_op_info) const {
  for (const auto &it : op_infos) {
    if ((it.engine == kHostCpuEngineName) && (it.opKernelLib == kHostCpuOpKernelLibName)) {
      op_desc->SetOpEngineName(kHostCpuEngineName);
      matched_op_info.opKernelLib = kHostCpuOpKernelLibName;
      GELOGI("DNNEngineManager: Set OpKernelLibName %s and OpEngineName %s to %s",
             kHostCpuOpKernelLibName, kHostCpuEngineName, op_desc->GetName().c_str());
      return kHostCpuEngineName;
    }
  }
  GELOGE(FAILED, "[Get][HostCpuEngineName]Failed, HostCpuEngine not support [%s, %s]",
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  REPORT_INNER_ERR_MSG("E19999", "Get HostCpuEngineName failed, HostCpuEngine not support [%s, %s]",
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return "";
}

const std::map<std::string, SchedulerConf> &DNNEngineManager::GetSchedulers() const { return schedulers_; }

Status DNNEngineManager::ParserJsonFile() {
  GELOGI("Begin to parse json file");
  std::string json_file_path = "plugin/nnengine/ge_config/engine_conf.json";
  std::string path = GetModelPath();
  path.append(json_file_path);
  nlohmann::json scheduler_json_file;
  Status status = ReadJsonFile(path, &scheduler_json_file);
  if (status != SUCCESS) {
    GELOGE(FAILED, "[Read][JsonFile]Failed, file %s", path.c_str());
    return FAILED;
  }
  if (scheduler_json_file.is_null()) {
    // when engine_conf.json does not exist, just return success
    GELOGW("Json file is null");
    return SUCCESS;
  }

  try {
    nlohmann::json scheduler_utils_json = scheduler_json_file[kSchedulerUnits];
    if (scheduler_utils_json.is_null()) {
      GELOGE(FAILED, "[Check[Param]Find scheduler units failed, the message is null, file %s", path.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Find scheduler units failed, the message is null, file %s", path.c_str());
      return FAILED;
    }
    if (!scheduler_utils_json.is_array()) {
      GELOGE(FAILED, "[Check][Param]The message of kSchedulerUnits is not array and "
             "the file path is %s", path.c_str());
      REPORT_INNER_ERR_MSG("E19999", "The message of kSchedulerUnits is not array and "
                        "the file path is %s", path.c_str());
      return FAILED;
    }
    auto size = scheduler_json_file[kSchedulerUnits].size();
    for (size_t i = 0; i < size; i++) {
      SchedulerConf scheduler_conf;
      std::map<std::string, EngineConfPtr> engine_conf_map;
      nlohmann::json engines_json_map = scheduler_utils_json[i][kCalEngines];
      if (engines_json_map.is_null()) {
        GELOGE(FAILED, "[Check][Param]The message of cal_engines is null, file %s", path.c_str());
        REPORT_INNER_ERR_MSG("E19999", "The message of cal_engines is null, file %s", path.c_str());
        return FAILED;
      }
      std::string scheduler_id_temp = scheduler_utils_json[i][kId];
      if (!scheduler_id_temp.empty()) {
        scheduler_conf.id = scheduler_id_temp;
      } else {
        GELOGE(FAILED, "[Check][Param]Scheduler ID is null, file %s", path.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Scheduler ID is null, file %s", path.c_str());
        return FAILED;
      }
      status = ParserEngineMessage(engines_json_map, scheduler_id_temp, engine_conf_map);
      if (status != SUCCESS) {
        GELOGE(FAILED, "[Parse][EngineMessage]Failed, scheduler_id_temp %s", scheduler_id_temp.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Parse engine message failed, scheduler_id_temp %s",
                          scheduler_id_temp.c_str());
        return FAILED;
      }
      scheduler_conf.name = scheduler_utils_json[i][kName];
      scheduler_conf.ex_attrs = scheduler_utils_json[i][kExAttrs];
      scheduler_conf.cal_engines = engine_conf_map;
      std::map<std::string, SchedulerConf>::const_iterator it = schedulers_.find(scheduler_id_temp);
      if (it != schedulers_.cend()) {
        GELOGW("[Check][Param]There are the same scheduler ts %s in the json file",
               scheduler_id_temp.c_str());
        continue;
      }
      schedulers_.emplace(scheduler_id_temp, scheduler_conf);
    }
  } catch (const nlohmann::detail::type_error &e) {
    GELOGE(FAILED, "[Parse][JsonFile]Failed, file %s, reason %s", path.c_str(), e.what());
    REPORT_INNER_ERR_MSG("E19999", "Parse json file %s failed, reason %s", path.c_str(), e.what());
    return FAILED;
  }

  GELOGI("Parser json file SUCCESS");
  return SUCCESS;
}

Status DNNEngineManager::ParserEngineMessage(const json engines_json, const std::string &scheduler_mark,
                                             std::map<std::string, EngineConfPtr> &engines) const {
  GELOGI("Begin to parser engine massage");
  if (engines_json.is_null()) {
    GELOGE(FAILED, "[Check][Param]The message of cal_engines is null");
    REPORT_INNER_ERR_MSG("E19999", "The message of cal_engines is null");
    return FAILED;
  }
  try {
    if (engines_json.is_array()) {
      for (size_t i = 0; i < engines_json.size(); i++) {
        nlohmann::json engines_elems = engines_json[i];
        EngineConfPtr engine_conf_ptr = MakeShared<EngineConf>();
        if (engine_conf_ptr == nullptr) {
          return FAILED;
        }
        std::string engine_id = engines_elems[kId];
        if (!engine_id.empty()) {
          engine_conf_ptr->id = engine_id;
        } else {
          GELOGE(FAILED, "[Check][Param]Engine ID is null");
          REPORT_INNER_ERR_MSG("E19999", "Engine ID is null");
          return FAILED;
        }
        if (engines_elems.find(kName) != engines_elems.end()) {
          engine_conf_ptr->name = engines_elems[kName];
        } else {
          GELOGW("The engine %s name is null", engine_id.c_str());
        }
        if (engines_elems.find(kIndependent) != engines_elems.end()) {
          engine_conf_ptr->independent = engines_elems[kIndependent];
        }

        if (engines_elems.find(kAttach) != engines_elems.end()) {
          engine_conf_ptr->attach = engines_elems[kAttach];
        }

        if (engines_elems.find(kSkipAssignStream) != engines_elems.end()) {
          engine_conf_ptr->skip_assign_stream = engines_elems[kSkipAssignStream];
        }
        engine_conf_ptr->scheduler_id = scheduler_mark;
        std::map<std::string, EngineConfPtr>::const_iterator it = engines.find(engine_id);
        if (it != engines.cend()) {
          GELOGE(FAILED, "[Check][Param]There are the same engine %s message in the json file",
                 engine_id.c_str());
          REPORT_INNER_ERR_MSG("E19999", "There are the same engine %s message in the json file",
                             engine_id.c_str());
          return FAILED;
        }
        engines.emplace(engine_id, engine_conf_ptr);
      }
    } else {
      GELOGE(FAILED, "[Check][Param]The message of cal_engines is not array in the json file");
      REPORT_INNER_ERR_MSG("E19999", "The message of cal_engines is not array in the json file");
      return FAILED;
    }
  } catch (const json::exception &e) {
    GELOGE(FAILED, "[Construct][JsonContent]Failed, reason %s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "Construct json content failed, reason %s", e.what());
    return FAILED;
  }
  GELOGI("Parser engine massage success");
  return SUCCESS;
}

Status DNNEngineManager::ReadJsonFile(const std::string &file_path, JsonHandle handle) {
  GELOGD("Begin to read json file");
  if (file_path.empty()) {
    GELOGE(FAILED, "[Check][Param]Json path is empty");
    REPORT_INNER_ERR_MSG("E19999", "Json path is empty");
    return FAILED;
  }
  nlohmann::json *json_file = reinterpret_cast<nlohmann::json *>(handle);
  if (json_file == nullptr) {
    GELOGE(FAILED, "[Check][Param]Json file is nullptr");
    REPORT_INNER_ERR_MSG("E19999", "Json file is nullptr");
    return FAILED;
  }
  const char *file = file_path.data();
  if ((mmAccess2(file, M_F_OK)) != EN_OK) {
    if (engines_map_.size() != 0) {
      GELOGE(FAILED, "[Check][Param]The json file %s does not exist, err %s",
             file_path.c_str(), strerror(errno));
      REPORT_INNER_ERR_MSG("E19999", "Json file %s does not exist, err %s",
                        file_path.c_str(), strerror(errno));
      return FAILED;
    } else {
      GELOGW("The json file %s is not needed.", file_path.c_str());
      return SUCCESS;
    }
  }

  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    GELOGE(FAILED, "[Open][JsonFile]Failed, file %s", file_path.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Open json file %s failed", file_path.c_str());
    return FAILED;
  }

  try {
    ifs >> *json_file;
  } catch (const json::exception &e) {
    GELOGE(FAILED, "[Read][JsonFile]Failed, reason %s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "Read json file failed, reason %s", e.what());
    ifs.close();
    return FAILED;
  }
  ifs.close();
  GELOGD("Read json file success");
  return SUCCESS;
}

Status DNNEngineManager::CheckJsonFile() const {
  GELOGD("Begin to check json file");
  for (auto &it : engines_map_) {
    std::string engine_name = it.first;
    int32_t count = 0;
    for (auto &iter : schedulers_) {
      auto engine_map = iter.second.cal_engines;
      std::map<std::string, EngineConfPtr>::const_iterator iter_engine_name = engine_map.find(engine_name);
      if (iter_engine_name != engine_map.cend()) {
        count++;
      }
    }
    if (count == 0) {
      GELOGE(FAILED, "[Check][JsonFile]The engine message %s is not found in the json file",
             engine_name.c_str());
      REPORT_INNER_ERR_MSG("E19999", "The engine message %s is not found in the json file",
                         engine_name.c_str());
      return FAILED;
    }
    if (count > 1) {
      GELOGE(FAILED, "[Check][JsonFile]The same engine message %s exists in the json file",
             engine_name.c_str());
      REPORT_INNER_ERR_MSG("E19999", "The same engine message %s exists in the json file",
                         engine_name.c_str());
      return FAILED;
    }
  }
  GELOGD("Check json file success");
  return SUCCESS;
}

void DNNEngineManager::InitAtomicCompositeMapping() {
  for (const auto &item : OpsKernelManager::GetInstance().GetCompositeEngines()) {
    const auto &composite_engine = GetEngine(item.first);
    if ((composite_engine == nullptr) || composite_engine->IsAtomic()) {
      GELOGW("Composite engine %s is not registered", item.first.c_str());
    }
    for (const auto &atomic_engine_name : item.second) {
      const auto &atomic_engine = GetEngine(atomic_engine_name);
      if ((atomic_engine == nullptr) || !atomic_engine->IsAtomic()) {
        GELOGW("Atomic engine %s is not registered", atomic_engine_name.c_str());
        continue;
      }
      std::map<std::string, std::string>::const_iterator iter = atomic_2_composite_.find(atomic_engine_name);
      if (iter != atomic_2_composite_.cend()) {
        GELOGW("Atomic engine %s has been contained in composite engine %s, and will be overwritten by engine %s",
               atomic_engine_name.c_str(), iter->second.c_str(), item.first.c_str());
      }
      atomic_2_composite_[atomic_engine_name] = item.first;
    }
  }
}

bool DNNEngineManager::IsNoTask(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  // op_desc of node should not be null
  if (op_desc->HasAttr(ATTR_NAME_NOTASK)) {
    return true;
  }
  return IsStreamAssignSkip(node) && op_desc->GetSubgraphInstanceNames().empty();
}

bool DNNEngineManager::IsStreamAssignSkip(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  // op_desc of node should not be null
  const auto &engine_name = op_desc->GetOpEngineName().empty() ? GetDNNEngineName(node) : op_desc->GetOpEngineName();
  return IsStreamAssignSkip(engine_name);
}

bool DNNEngineManager::IsStreamAssignSkip(const std::string &engine_name) {
  // Only one scheduler has been supported by now
  for (const auto &scheduler : schedulers_) {
    const auto &iter = scheduler.second.cal_engines.find(engine_name);
    if (iter == scheduler.second.cal_engines.end()) {
      GELOGW("No engine found within name %s", engine_name.c_str());
      continue;
    }
    if (iter->second == nullptr) {
      GELOGW("engine configuration of engine %s is null", engine_name.c_str());
      continue;
    }
    return iter->second->skip_assign_stream;
  }
  return false;
}
}  // namespace ge
