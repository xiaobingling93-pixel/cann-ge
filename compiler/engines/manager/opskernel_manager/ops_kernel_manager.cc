/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/manager/opskernel_manager/ops_kernel_manager.h"

#include "api/gelib/gelib.h"
#include "proto/optimizer_priority.pb.h"
#include "common/proto_util/proto_util.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/ge_context.h"
#include "graph/types.h"
#include "common/checker.h"
#include "engines/custom_engine/custom_ops_kernel_info_store.h"
#include "engines/custom_engine/custom_graph_optimizer.h"
#include "common/util/mem_utils.h"
#include "common/ge_common/ge_types.h"

namespace ge {
namespace {
const char *const kInitialize = "Initialize";
const char *const kGetOpsKernelInfoStores = "GetOpsKernelInfoStores";
const char *const kGetGraphOptimizerObjs = "GetGraphOptimizerObjs";
const char *const kFinalize = "Finalize";
const char *const kGetCompositeEngines = "GetCompositeEngines";
const ge::char_t *const kGetFftsEnableFlag = "GetFFTSPlusSwitch";
const size_t kSlogOverflowThreshold = 1024u;
std::mutex ops_kernel_info_mutex;

Status InsertCustomOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_store) {
  OpsKernelInfoStorePtr custom_kernel_info_store_ptr = MakeShared<custom::CustomOpsKernelInfoStore>();
  GE_ASSERT_NOTNULL(custom_kernel_info_store_ptr);
  ops_kernel_store.emplace(std::make_pair(kCustomOpKernelLibName, custom_kernel_info_store_ptr));
  return SUCCESS;
}

Status InsertCustomGraphOptimizers(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  GraphOptimizerPtr custom_graph_optimizer_ptr = MakeShared<CustomGraphOptimizer>();
  GE_ASSERT_NOTNULL(custom_graph_optimizer_ptr);
  graph_optimizers.emplace(std::make_pair(kCustomGraphOptimizer, custom_graph_optimizer_ptr));
  return SUCCESS;
}
}  // namespace
OpsKernelManager::OpsKernelManager()
    : plugin_manager_(),
      op_tiling_manager_(),
      graph_optimize_utility_(),
      init_flag_(false) {}

OpsKernelManager::~OpsKernelManager() {
  graph_optimizers_.clear();
  ops_kernel_store_.clear();
  atomic_graph_optimizers_.clear();
  composite_graph_optimizers_.clear();
  atomic_graph_optimizers_by_priority_.clear();
  atomic_first_optimizers_by_priority_.clear();
  composite_engines_.clear();
  ops_kernel_info_.clear();
}

OpsKernelManager &OpsKernelManager::GetInstance() {
  static OpsKernelManager instance;
  return instance;
}

Status OpsKernelManager::Initialize(const std::map<std::string, std::string> &init_options) {
  if (init_flag_) {
    GELOGW("OpsKernelManager has been initialized.");
    return SUCCESS;
  }
  init_flag_ = true;
  std::map<std::string, std::string> options(init_options);
  std::vector<std::string> func_check_list = {kInitialize, kGetOpsKernelInfoStores, kGetGraphOptimizerObjs, kFinalize};
  const std::map<std::string, std::string>::const_iterator it = options.find(OPTION_EXEC_IS_USEHCOM);
  if (it == options.cend()) {
    GELOGI("OPTION_EXEC_IS_USEHCOM is not set, default is single P");
    options.emplace("ge.exec.isUseHcom", to_string(0));
  }
  const std::map<std::string, std::string>::const_iterator iter = options.find(OPTION_EXEC_IS_USEHVD);
  if (iter == options.cend()) {
    GELOGI("OPTION_EXEC_IS_USEHVD is not set, default is single P");
    options.emplace("ge.exec.isUseHvd", to_string(0));
  }
  std::string extern_engine_path;
  GetExternalEnginePath(extern_engine_path, options);
  GELOGI("OPTION_EXEC_EXTERN_PLUGIN_PATH=%s.", extern_engine_path.c_str());
  op_tiling_manager_.LoadSo();
  GE_TRACE_START(LoadPluginManagerSo);
  Status ret = plugin_manager_.LoadSo(extern_engine_path, func_check_list);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "OpsKernelManager::Initialize failed for not find any valid so file.");
  GE_INIT_TRACE_TIMESTAMP_END(LoadPluginManagerSo, "OpsKernelManager::LoadPluginManagerSo");
  initialize_ = options;
  GE_CHK_BOOL_RET_STATUS((plugin_manager_.InvokeAll<std::map<std::string, std::string> &, Status>(kInitialize,
      initialize_) == SUCCESS), GE_OPS_GET_NO_VALID_SO, "PluginManager InvokeAll failed.");
  if (plugin_manager_.InvokeAll<std::map<std::string, OpsKernelInfoStorePtr> &>(kGetOpsKernelInfoStores,
                                                                                ops_kernel_store_) != SUCCESS) {
    GELOGW("Initialize OpsKernelInfo failed.");
  }
  if (plugin_manager_.InvokeAll<std::map<std::string, GraphOptimizerPtr> &>(kGetGraphOptimizerObjs,
                                                                            graph_optimizers_) != SUCCESS) {
    GELOGW("Initialize GraphOptimizerObjs failed.");
  }
  GE_ASSERT_SUCCESS(InsertCustomOpsKernelInfoStores(ops_kernel_store_));
  GE_ASSERT_SUCCESS(InsertCustomGraphOptimizers(graph_optimizers_));
  plugin_manager_.
    OptionalInvokeAll<std::map<std::string, std::set<std::string>> &, std::map<std::string, std::string> &>(
      kGetCompositeEngines, composite_engines_, composite_engine_kernel_lib_names_);
  plugin_manager_.OptionalInvokeAll<bool &>(kGetFftsEnableFlag, enable_ffts_flag_);
  GE_CHK_STATUS_RET_NOLOG(CheckPluginPtr());
  GE_CHK_STATUS_RET_NOLOG(InitOpKernelInfoStores(options));
  InitOpsKernelInfo();
  GE_CHK_STATUS_RET_NOLOG(InitGraphOptimizers(options));
  ClassifyGraphOptimizers();
  InitGraphOptimizerPriority();
  return SUCCESS;
}

void OpsKernelManager::GetExternalEnginePath(std::string &extern_engine_path,
    const std::map<std::string, std::string>& options) const {
  GELOGI("Enter get external engine so path schedule");
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_ENGINE_PATH, path_env);
  if (path_env != nullptr) {
    extern_engine_path = path_env;
    GELOGI("OpsKernelManager get external engine so path from env.");
    return;
  }
  std::string path_base = GetModelPath();
  std::string so_path = "plugin/opskernel/";
  std::string path = path_base + so_path;
  extern_engine_path = (path + "libfe.so" + ":") + (path + "libge_local_engine.so" + ":") +
                       (path + "librts_engine.so" + ":") + (path + "libaicpu_ascend_engine.so" + ":") +
                       (path + "libhost_cpu_engine.so" + ":") + (path + "libaicpu_tf_engine.so" + ":") +
                       (path + "libffts.so" + ":") + (path + "libdvpp_engine.so" + ":");
  auto iter = options.find(OPTION_EXEC_HCCL_FLAG);
  if (iter == options.end() || iter->second != "0") {
    extern_engine_path += (path_base + "libhcom_graph_adaptor.so");
  }
}

Status OpsKernelManager::CheckPluginPtr() const {
  for (auto iter = ops_kernel_store_.begin(); iter != ops_kernel_store_.end(); ++iter) {
    if (iter->second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][PluginPtr] OpsKernelInfoStorePtr key=%s is null", iter->first.c_str());
      REPORT_INNER_ERR_MSG("E19999", "CheckPluginPtr OpsKernelInfoStorePtr key=%s is null", iter->first.c_str());
      return FAILED;
    }
  }
  for (auto iter1 = graph_optimizers_.begin(); iter1 != graph_optimizers_.end(); ++iter1) {
    if (iter1->second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][PluginPtr] GraphOptimizerPtr key=%s is null", iter1->first.c_str());
      REPORT_INNER_ERR_MSG("E19999", "GraphOptimizerPtr key=%s is null", iter1->first.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpsKernelManager::InitOpKernelInfoStores(const std::map<std::string, std::string> &options) {
  FuncPerfScope func_perf_scope("OpsKernelManager", __FUNCTION__);
  GELOGI("The number of OpKernelInfoStoreObjs are %lu.", ops_kernel_store_.size());
  for (const auto &it : ops_kernel_store_) {
    GELOGI("OpKernelInfoStore name: %s.", (it.first).c_str());
    const uint64_t start = ge::GetCurrentTimestamp();
    Status ret = it.second->Initialize(options);
    const uint64_t end = ge::GetCurrentTimestamp();
    GEEVENT("[GEPERFTRACE] The time cost of InitOpKernelInfoStores::Initialize[%s] is [%lu] micro seconds.",
            (it.first.c_str()), (end - start));
    if (ret != SUCCESS) {
      GELOGE(GE_OPS_KERNEL_STORE_INIT_FAILED,
             "[Init][OpKernelLib]OpKernelInfoStore: %s initialize failed.", (it.first).c_str());
      REPORT_INNER_ERR_MSG("E19999", "OpKernelInfoStore: %s initialize failed.", (it.first).c_str());
      return GE_OPS_KERNEL_STORE_INIT_FAILED;
    }
  }

  return SUCCESS;
}

void OpsKernelManager::InitOpsKernelInfo() {
  FuncPerfScope func_perf_scope("OpsKernelManager", __FUNCTION__);
  ops_kernel_info_.clear();
  for (const auto &it : ops_kernel_store_) {
    std::map<std::string, OpInfo> op_infos{};
    const uint64_t start = ge::GetCurrentTimestamp();
    it.second->GetAllOpsKernelInfo(op_infos);
    const uint64_t end = ge::GetCurrentTimestamp();
    GEEVENT("[GEPERFTRACE] The time cost of InitOpsKernelInfo::GetAllOpsKernelInfo[%s] is [%lu] micro seconds.",
            (it.first.c_str()), (end - start));
    for (const auto &op_info_it : op_infos) {
      auto op_info_copy = op_info_it.second;
      // flush ops kernel
      op_info_copy.opKernelLib = it.first;
      ops_kernel_info_[op_info_it.first].emplace_back(op_info_copy);
      GELOGD("OpKernelInfoStore name: %s, found op type is %s, engine name is %s, opkernel name is %s",
             (it.first).c_str(), op_info_it.first.c_str(), op_info_it.second.engine.c_str(),
             op_info_it.second.opKernelLib.c_str());
    }
  }
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib]malloc instance_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "InitOpsKernelInfo failed for new GELib.");
    return;
  }
  // sort op_info of ops_kernel_info_
  for (auto &it : ops_kernel_info_) {
    if (it.second.empty()) {
      continue;
    }
    auto comp_func = [&instance_ptr](const OpInfo &op_a, const OpInfo &op_b) -> bool {
      const std::string &a = op_a.engine;
      const std::string &b = op_b.engine;
      // check if a or b is registered
      if (!(instance_ptr->DNNEngineManagerObj().IsEngineRegistered(a))) {
        return false;
      }
      if (!(instance_ptr->DNNEngineManagerObj().IsEngineRegistered(b))) {
        return true;
      }
      // compare compute cost of a and b, IsEngineRegistered make sure engine is not nullptr
      auto engine_a = instance_ptr->DNNEngineManagerObj().GetEngine(a);
      auto engine_b = instance_ptr->DNNEngineManagerObj().GetEngine(b);
      DNNEngineAttribute attr_a, attr_b;
      engine_a->GetAttributes(attr_a);
      engine_b->GetAttributes(attr_b);
      return attr_a.compute_cost < attr_b.compute_cost;
    };
    // Sort the OpInfos based on the compute cost of the engine
    std::sort(it.second.begin(), it.second.end(), comp_func);
  }
  GELOGI("Init opsKernelInfo finished, size is %zu", ops_kernel_info_.size());
}

Status OpsKernelManager::InitGraphOptimizers(const std::map<std::string, std::string> &options) {
  FuncPerfScope func_perf_scope("OpsKernelManager", __FUNCTION__);
  GELOGI("Init graph optimizers options count %zu", options.size());
  for (const auto &option : options) {
    // slog日志截断阈值，若发生截断，跳过打印
    if ((option.second.length() > kSlogOverflowThreshold) || (option.second.empty())) {
      continue;
    }
    GELOGI("Init graph optimizers option %s: %s", option.first.c_str(), option.second.c_str());
  }
  GELOGI("The number of GraphOptimizerObjs are %zu.", graph_optimizers_.size());

  for (const auto &it : graph_optimizers_) {
    GELOGI("GraphOptimizer name: %s.", (it.first).c_str());
    GraphOptimizerAttribute attrs;
    GE_CHK_STATUS_RET(it.second->GetAttributes(attrs));
    if (!DNNEngineManager::GetInstance().IsEngineRegistered(attrs.engineName)) {
      GELOGW("Engine: %s is not registered.", attrs.engineName.c_str());
      continue;
    }
    const uint64_t start = ge::GetCurrentTimestamp();
    if (it.second->Initialize(options, &graph_optimize_utility_) != SUCCESS) {
      GELOGE(GE_OPS_GRAPH_OPTIMIZER_INIT_FAILED,
             "[Init][GraphOptimizer] GraphOptimizer: %s initialize failed.", (it.first).c_str());
      REPORT_INNER_ERR_MSG("E19999", "InitGraphOptimizers failed. %s initialize failed.", (it.first).c_str());
      return GE_OPS_GRAPH_OPTIMIZER_INIT_FAILED;
    }
    const uint64_t end = ge::GetCurrentTimestamp();
    GEEVENT("[GEPERFTRACE] The time cost of InitGraphOptimizers::Initialize[%s] is [%lu] micro seconds.",
            (it.first.c_str()), (end - start));
  }
  return SUCCESS;
}

Status OpsKernelManager::Finalize() {
  if (!init_flag_) {
    GELOGW("Finalize is not allowed, initialize first is necessary.");
    return SUCCESS;
  }
  GELOGI("free ops kernel resource.");
  for (auto iter = ops_kernel_store_.cbegin(); iter != ops_kernel_store_.cend(); ++iter) {
    GELOGI("OpsKernelStore finalize, name: %s.", (iter->first).c_str());
    Status status = iter->second->Finalize();
    if (status != SUCCESS) {
      GELOGE(status, "[Check][Status]OpsKernelStore finalize failed, name: %s.", (iter->first).c_str());
      REPORT_INNER_ERR_MSG("E19999", "OpsKernelStore finalize failed, name: %s.", (iter->first).c_str());
      return status;
    }
  }
  for (auto iter = graph_optimizers_.cbegin(); iter != graph_optimizers_.cend(); ++iter) {
    GELOGI("GraphOptimizer finalize, name: %s.", (iter->first).c_str());
    Status status = iter->second->Finalize();
    if (status != SUCCESS) {
      GELOGE(status, "[Check][Status] GraphOptimizer finalize failed, name: %s.", (iter->first).c_str());
      REPORT_INNER_ERR_MSG("E19999", "GraphOptimizer finalize failed, name: %s.", (iter->first).c_str());
      return status;
    }
  }

  Status ret = FinalizeOpsKernel();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Free][Ops Kernel Resource] failed.");
    return ret;
  }
  graph_optimizers_.clear();
  ops_kernel_store_.clear();
  atomic_graph_optimizers_.clear();
  composite_graph_optimizers_.clear();
  atomic_graph_optimizers_by_priority_.clear();
  atomic_first_optimizers_by_priority_.clear();
  composite_engines_.clear();
  ops_kernel_info_.clear();
  init_flag_ = false;
  return SUCCESS;
}

std::vector<OpInfo> OpsKernelManager::GetOpsKernelInfo(const std::string &op_type) {
  std::lock_guard<std::mutex> lock(ops_kernel_info_mutex);

  std::map<std::string, std::vector<OpInfo>>::const_iterator find = ops_kernel_info_.find(op_type);
  if (find != ops_kernel_info_.cend()) {
    return find->second;
  } else {
    InitOpsKernelInfo();
    find = ops_kernel_info_.find(op_type);
    if (find != ops_kernel_info_.end()) {
      return find->second;
    }
    GELOGW("Failed to get opsKernelInfo object by type: %s.", op_type.c_str());
    return empty_op_info_;
  }
}

const std::map<std::string, std::vector<OpInfo>> &OpsKernelManager::GetAllOpsKernelInfo() const {
  std::lock_guard<std::mutex> lock(ops_kernel_info_mutex);
  return ops_kernel_info_;
}

OpsKernelInfoStorePtr OpsKernelManager::GetOpsKernelInfoStore(const std::string &name) const {
  auto find = ops_kernel_store_.find(name);
  if (find != ops_kernel_store_.end()) {
    return find->second;
  }

  GELOGW("Failed to get opsKernelInfoStore object by name. OpKernelLibName is %s", name.c_str());
  return nullptr;
}

const std::map<std::string, OpsKernelInfoStorePtr> &OpsKernelManager::GetAllOpsKernelInfoStores() const {
  return ops_kernel_store_;
}

const std::map<std::string, GraphOptimizerPtr> &OpsKernelManager::GetAllGraphOptimizerObjs() const {
  return graph_optimizers_;
}

void OpsKernelManager::GetGraphOptimizerByEngine(const std::string &engine_name,
                                                 std::vector<GraphOptimizerPtr> &graph_optimizer) {
  for (const auto &it : graph_optimizers_) {
    GraphOptimizerAttribute attrs;
    if (it.second->GetAttributes(attrs) != SUCCESS) {
      GELOGW("Get GraphOptimizer name: %s attributes failed.", (it.first).c_str());
      continue;
    }
    if (attrs.engineName == engine_name) {
      GELOGD("GraphOptimizer name: %s, engineName: %s", (it.first).c_str(), engine_name.c_str());
      graph_optimizer.push_back(it.second);
    }
  }
}

void OpsKernelManager::ClassifyGraphOptimizers() {
  FuncPerfScope func_perf_scope("OpsKernelManager", __FUNCTION__);
  if (composite_engines_.empty()) {
    GELOGI("No composite engine registers");
    atomic_graph_optimizers_ = graph_optimizers_;
    composite_graph_optimizers_.clear();
    return;
  }
  for (const auto &item : graph_optimizers_) {
    GraphOptimizerAttribute attrs;
    const uint64_t start = ge::GetCurrentTimestamp();
    if (item.second->GetAttributes(attrs) != SUCCESS) {
      GELOGW("Get GraphOptimizer attributes failed, name: %s.", (item.first).c_str());
      continue;
    }
    const uint64_t end = ge::GetCurrentTimestamp();
    GEEVENT("[GEPERFTRACE] The time cost of ClassifyGraphOptimizers::GetAttributes[%s] is [%lu] micro seconds.",
            (item.first.c_str()), (end - start));
    if (composite_engines_.find(attrs.engineName) != composite_engines_.end()) {
      GELOGI("Engine of optimizer %s is %s, which is composited.", item.first.c_str(), attrs.engineName.c_str());
      composite_graph_optimizers_.emplace(item);
    } else {
      GELOGI("Engine of optimizer %s is %s, which is atomic.", item.first.c_str(), attrs.engineName.c_str());
      atomic_graph_optimizers_.emplace(item);
    }
  }
}

void OpsKernelManager::InitGraphOptimizerPriority() {
  FuncPerfScope func_perf_scope("OpsKernelManager", __FUNCTION__);
  std::string priority_conf_path = "plugin/opskernel/optimizer_priority.pbtxt";
  std::string path = GetModelPath();
  path.append(priority_conf_path);

  optimizers::Priority optimizerPriority;
  if (!ReadProtoFromText(path.c_str(), &optimizerPriority)) {
    GELOGW("Read priority file failed. Follow loading sequence.");
    return;
  }
  auto priorities = optimizerPriority.optimizer();
  if (priorities.empty()) {
    GELOGI("No priority file config. Follow loading sequence.");
    return;
  }
  // sort optimizer map by priority
  std::stringstream priority_seq;
  for (const auto &optimizer_name : priorities) {
    auto name_to_optimizer_pair = atomic_graph_optimizers_.find(optimizer_name);
    if (name_to_optimizer_pair != atomic_graph_optimizers_.end()) {
      atomic_graph_optimizers_by_priority_.emplace_back(*name_to_optimizer_pair);
      priority_seq << optimizer_name.c_str() << ' ';
    } else {
      GELOGW("Unknown optimizer %s show up in priority config file. Please check.", optimizer_name.c_str());
    }
  }
  GELOGI("Atomic graph Optimizers priority initialized. The sequence will follow : %s.", priority_seq.str().c_str());
  atomic_first_optimizers_by_priority_ = atomic_graph_optimizers_by_priority_;
  for (const auto &item : composite_graph_optimizers_) {
    atomic_first_optimizers_by_priority_.emplace_back(std::make_pair(item.first, item.second));
  }
}

Status OpsKernelManager::FinalizeOpsKernel() {
  GELOGI("ge invoke ops kernel finalize.");
  Status ret = plugin_manager_.InvokeAll<Status>(kFinalize);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Finalize][Check][Status] invoke Fe finalize failed.");
    REPORT_INNER_ERR_MSG("E19999", "PluginManager InvokeAll failed.");
    return ret;
  }

  return SUCCESS;
}
}  // namespace ge
