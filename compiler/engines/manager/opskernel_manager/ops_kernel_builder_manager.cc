/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"

#include "framework/common/debug/log.h"
#include "register/ops_kernel_builder_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
const std::vector<std::string> kBasicBuilderLibs = {
    "libge_local_opskernel_builder.so",
    "libhost_cpu_opskernel_builder.so",
    "librts_kernel_builder.so",
    "libaicpu_ascend_builder.so",
    "libaicpu_tf_builder.so",
    "libdvpp_engine.so"
};

const std::vector<std::string> kHcclBuilderLibs = {
    "libhcom_opskernel_builder.so",
    "libhvd_opskernel_builder.so",
    "libhcom_gradtune_opskernel_builder.so"
};
}  // namespace
OpsKernelBuilderManager::~OpsKernelBuilderManager() {
}

OpsKernelBuilderManager &OpsKernelBuilderManager::Instance() {
  static OpsKernelBuilderManager instance;
  return instance;
}

Status OpsKernelBuilderManager::Initialize(const std::map<std::string, std::string> &options,
                                           const std::string &path_base, const bool is_train) {
  if (is_train) {
    std::string lib_paths;
    GE_CHK_STATUS_RET_NOLOG(GetLibPaths(options, path_base, lib_paths));
    plugin_manager_ = MakeUnique<PluginManager>();
    GE_CHECK_NOTNULL(plugin_manager_);
    GE_CHK_STATUS_RET(plugin_manager_->LoadSo(lib_paths),
        "[Load][Libs]Failed, lib_paths=%s.", lib_paths.c_str());
  }

  auto &kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  GELOGI("[Show][OpsKernelBuilderNum]Number of OpBuild = %zu", kernel_builders.size());

  for (const auto &it : kernel_builders) {
    const std::string &kernel_lib_name = it.first;
    GELOGI("Initialize ops kernel util for %s", kernel_lib_name.c_str());
    GE_CHECK_NOTNULL(it.second);
    GE_CHK_STATUS_RET(it.second->Initialize(options),
        "[Invoke][Initialize]failed, kernel lib name = %s", kernel_lib_name.c_str());
  }

  return SUCCESS;
}

Status OpsKernelBuilderManager::Finalize() {
  const auto &ops_kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  for (const auto &it : ops_kernel_builders) {
    const std::string &kernel_lib_name = it.first;
    GELOGI("Finalize ops kernel util for %s", kernel_lib_name.c_str());
    const auto ret = it.second->Finalize();
    if (ret != SUCCESS) {
      GELOGW("Failed to invoke Finalize, kernel lib name = %s",
             kernel_lib_name.c_str());
    }
  }

  plugin_manager_.reset();
  return SUCCESS;
}

const std::map<std::string, OpsKernelBuilderPtr> &OpsKernelBuilderManager::GetAllOpsKernelBuilders() const {
  return OpsKernelBuilderRegistry::GetInstance().GetAll();
}

OpsKernelBuilderPtr OpsKernelBuilderManager::GetOpsKernelBuilder(const std::string &name) const {
  const auto &ops_kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  const auto it = ops_kernel_builders.find(name);
  if (it != ops_kernel_builders.end()) {
    return it->second;
  }

  GELOGW("Failed to get opsKernelInfoStore object by name. OpKernelLibName is %s", name.c_str());
  return nullptr;
}

Status OpsKernelBuilderManager::GetLibPaths(const std::map<std::string, std::string> &options,
                                            const std::string &path_base, std::string &lib_paths) const {
  GELOGD("Start to execute GetLibPaths");
  const std::string so_path = "plugin/opskernel/";
  const std::string path = path_base + so_path;
  std::string all_lib_paths;
  for (const auto &lib_name : kBasicBuilderLibs) {
    all_lib_paths += (path + lib_name + ":");
  }

  const auto iter = options.find(OPTION_EXEC_HCCL_FLAG);
  if ((iter == options.end()) || (iter->second != "0")) {
    for (const auto &lib_name : kHcclBuilderLibs) {
      all_lib_paths += (path + lib_name + ":");
    }
  }

  lib_paths = std::move(all_lib_paths);
  GELOGI("Get lib paths by default. paths = %s", lib_paths.c_str());
  return SUCCESS;
}

Status OpsKernelBuilderManager::CalcOpRunningParam(Node &node) const {
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const std::string &lib_name = op_desc->GetOpKernelLibName();
  const auto &ops_kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  const auto it = ops_kernel_builders.find(lib_name);
  if (it == ops_kernel_builders.end()) {
    GELOGE(INTERNAL_ERROR, "[Find][LibName] fail for libName = %s, node = %s.", lib_name.c_str(),
           op_desc->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999",
                       "find LibName for CalcOpRunningParam failed, libName = %s, node = %s does not exist.",
                       lib_name.c_str(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("To invoke CalcOpRunningParam, node = %s, lib name = %s", op_desc->GetName().c_str(), lib_name.c_str());
  GE_CHK_STATUS_RET(it->second->CalcOpRunningParam(node),
      "[Invoke][CalcOpRunningParam]failed, libName = %s, node = %s", lib_name.c_str(), op_desc->GetName().c_str());
  GELOGD("Done invoking CalcOpRunningParam successfully");
  return SUCCESS;
}

Status OpsKernelBuilderManager::GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks,
                                             const bool atomic_engine_flag) const {
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::string lib_name;
  if (atomic_engine_flag) {
    lib_name = op_desc->GetOpKernelLibName();
  } else {
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME, lib_name);
  }
  const auto &ops_kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  const auto it = ops_kernel_builders.find(lib_name);
  if (it == ops_kernel_builders.end()) {
    GELOGE(INTERNAL_ERROR, "[Find][LibName]fail for libName = %s, node:%s", lib_name.c_str(),
           op_desc->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "find LibName for GenerateTask failed, libName = %s, node = %s does not exist",
                       lib_name.c_str(), op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GELOGD("To invoke GenerateTask, node = %s, lib name = %s", op_desc->GetName().c_str(), lib_name.c_str());
  GE_CHK_STATUS_RET(it->second->GenerateTask(node, context, tasks),
      "[Invoke][GenerateTask]failed, libName = %s, node = %s", lib_name.c_str(), op_desc->GetName().c_str());
  GELOGD("Done invoking GenerateTask successfully");
  return SUCCESS;
}

Status OpsKernelBuilderManager::UpdateTask(const Node &node, std::vector<domi::TaskDef> &tasks) const {
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::string lib_name = op_desc->GetOpKernelLibName();
  const auto &ops_kernel_builders = OpsKernelBuilderRegistry::GetInstance().GetAll();
  const auto it = ops_kernel_builders.find(lib_name);
  if (it == ops_kernel_builders.end()) {
    GELOGD("node %s doesn't find lib, maybe this is ffts plus engine", op_desc->GetNamePtr());
    return SUCCESS;
  }

  GELOGD("To invoke UpdateTask, node = %s, lib name = %s", op_desc->GetName().c_str(), lib_name.c_str());
  auto task_size = tasks.size();
  GE_CHK_STATUS_RET(it->second->UpdateTask(node, tasks),
                    "[Invoke][UpdateTask]failed, libName = %s, node = %s", lib_name.c_str(), op_desc->GetName().c_str());
  if (tasks.size() != task_size) {
    // 流拆分后StreamSwitch、StreamActive的激活流数量可能变化，进而导致task数量变化
    if (op_desc->HasAttr(ATTR_NAME_ACTIVE_STREAM_LIST)) {
      GELOGI("node has active stream attr, libName = %s, node = %s, old task size %zu, new task size %zu",
             lib_name.c_str(), op_desc->GetName().c_str(), task_size, tasks.size());
    } else {
      GELOGE(FAILED, "[Invoke][UpdateTask]failed, libName = %s, node = %s, old task size %zu, new task size %zu",
             lib_name.c_str(), op_desc->GetName().c_str(), task_size, tasks.size());
      return FAILED;
    }
  }
  GELOGD("Done invoking UpdateTask successfully");
  return SUCCESS;
}
}  // namespace ge
