/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/hybrid_model_rt_v2_executor.h"

#include <sys/syscall.h>
#include <sys/mman.h>
#include <memory>
#include "lowering/model_converter.h"
#include "framework/runtime/model_v2_executor.h"
#include "exe_graph/runtime/tensor_data.h"

#include "common/dump/dump_manager.h"
#include "common/memory/tensor_trans_utils.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/host_mem_allocator.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/utils/type_utils.h"
#include "graph/runtime_inference_context.h"
#include "graph/manager/trans_var_data_utils.h"
#include "hybrid/executor/runtime_v2/rt_v2_executor_factory.h"
#include "hybrid/executor/runtime_v2/rt_v2_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "common/profiling/profiling_manager.h"
#include "ge/ge_api_types.h"
#include "utils/utils.h"
#include "core/utils/tensor_utils.h"
#include "common/global_variables/diagnose_switch.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/model/external_allocator_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "acl/acl_rt.h"
#include "register/core_num_utils.h"

namespace ge {
namespace hybrid {
namespace {
constexpr uint32_t kMaxThreadNum = 16U;
constexpr uint64_t kMaxStringSize = 1024U;
const std::string kGuardCheckSoName = "libguard_check.so";
constexpr char_t const *kGuardCheckSoDataResult = "_guard_check_so_data";
const std::string kVectorcoreNum = "ge.vectorcoreNum";

struct VarMgrNodes {
  std::vector<ge::NodePtr> variables;
  std::vector<ge::NodePtr> shared_constants;
  std::vector<ge::NodePtr> file_constants;
};
Status GetModelSessionId(const GeRootModelPtr &model, uint64_t &session_id) {
  GE_ASSERT_NOTNULL(model);
  GE_ASSERT(!model->GetSubgraphInstanceNameToModel().empty());
  const auto &first_model = model->GetSubgraphInstanceNameToModel().begin()->second;
  if (!AttrUtils::GetInt(first_model, MODEL_ATTR_SESSION_ID, session_id)) {
    const static uint64_t kDefaultSessionId = 0U;
    GELOGI("No session attr named %s on model %s, use default session id %lu", MODEL_ATTR_SESSION_ID.c_str(),
           first_model->GetGraph()->GetName().c_str(), kDefaultSessionId);
    session_id = kDefaultSessionId;
  }
  return ge::SUCCESS;
}

Status GetTensorDescSize(const GeTensorDesc &desc, int64_t &size) {
  if (desc.GetDataType() == DT_STRING) {
    GE_ASSERT_GRAPH_SUCCESS(TensorUtils::GetSize(desc, size));
    return SUCCESS;
  }
  GE_ASSERT_GRAPH_SUCCESS(TensorUtils::GetTensorMemorySizeInBytes(desc, size));
  return SUCCESS;
}

bool IsVarPlacedOnRdma(const NodePtr &node) {
  uint32_t type_key = 0U;
  return AttrUtils::GetInt(node->GetOpDesc(), ATTR_OUTPUT_MEMORY_TYPE, type_key) && (type_key == 1U);
}

bool IsVarPlacedOnHost(const NodePtr &node) {
  const std::string *placement = AttrUtils::GetStr(node->GetOpDesc(), ATTR_VARIABLE_PLACEMENT);
  if ((placement == nullptr) || placement->empty()) {
    return false;
  }
  return *placement == "host";
}

void CollectGraphVarMgrNodes(const ge::ComputeGraphPtr &graph, VarMgrNodes &var_mgr_nodes) {
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == VARIABLE) {
      var_mgr_nodes.variables.push_back(node);
    } else if (node->GetType() == CONSTANTOP) {
      var_mgr_nodes.shared_constants.push_back(node);
    } else if (node->GetType() == FILECONSTANT) {
      var_mgr_nodes.file_constants.push_back(node);
    } else {
      continue;
    }
    GELOGD("Collect %s node %s", node->GetType().c_str(), node->GetName().c_str());
  }
}

Status EnsureModelVarMemoryMalloced(const GeRootModelPtr &model, const std::shared_ptr<ge::VarManager> &var_manager,
                                    uint32_t device_id) {
  GE_ASSERT_NOTNULL(model);
  uint64_t session_id = 0U;
  GE_ASSERT_SUCCESS(GetModelSessionId(model, session_id));
  const auto &root_graph = model->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  auto graph_id = root_graph->GetGraphID();

  if (!var_manager->HasMemoryManager()) {
    GELOGI("Model %u set memory manager for var manager of session %lu", graph_id, session_id);
    var_manager->SetMemManager(&ge::MemManager::Instance());
  }
  GELOGI("[Init] Variable mem auto malloc, no need to malloc var max size mem.");
  const auto page_size = VarManager::IsVariableUse1gHugePage() ? kDrv1GPageSize : kDrvPageSize;
  auto allocator =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(session_id, device_id,
                                                                                       RT_MEMORY_HBM, page_size);
  (void) var_manager->InitExpandableMemoryAllocator(allocator);
  return SUCCESS;
}

ge::Status DoRtStreamSyncWithTimeout(rtStream_t stream) {
  auto timeout = ge::GetContext().StreamSyncTimeout();
  auto rt_ret = rtStreamSynchronizeWithTimeout(stream, timeout);
  if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
    GELOGE(rt_ret, "[Invoke][rtStreamSynchronizeWithTimeout] failed, stream synchronize timeout:%d, ret:%d.", timeout,
           rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%d, ret:%d.",
                      timeout, rt_ret);
    return ge::FAILED;
  } else if (rt_ret == ACL_ERROR_RT_END_OF_SEQUENCE) {
    GELOGD("SyncStream return END_OF_SEQUENCE");
    return ge::END_OF_SEQUENCE;
  }
  GE_ASSERT_RT_OK(rt_ret, "SyncStream return %d", rt_ret);
  return ge::SUCCESS;
}

bool IsInputPlacementOnDeviceHbm() {
  std::string input_placement;
  (void)ge::GetThreadLocalContext().GetOption("ge.inputPlacement", input_placement);
  return input_placement == "DeviceHbm";
}
}  // namespace

std::string GraphVarVisitor::Variable::DebugString() const {
  std::stringstream ss;
  ss << "desc " << ::ge::hybrid::DebugString(desc) << ", placement " << ReadablePlacement(placement) << ", address "
     << reinterpret_cast<uintptr_t>(addr) << ", size " << size;
  return ss.str();
}

std::string GraphVarVisitor::Variable::ReadablePlacement(const VariablePlacement placement) {
  switch (placement) {
    case VariablePlacement::kOnHost:
      return "host";
    case VariablePlacement::kOnDeviceHbm:
      return "device hbm";
    case VariablePlacement::kOnDeviceRdma:
      return "device rdma";
    default:
      return "undefined";
  }
}

Status GraphVarVisitor::AssignVarLogicalMemory(const ge::NodePtr &node, bool &assigned) {
  const std::string &var_name = node->GetName();
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto &var_desc = node->GetOpDesc()->GetOutputDescPtr(0U);
  GE_ASSERT_NOTNULL(var_desc, "Output desc of variable %s is nullptr", node->GetName().c_str());
  if (session_var_manager_->IsVarExist(var_name, *var_desc)) {
    GELOGD("Skip initialized %s %s with desc %s", node->GetType().c_str(), node->GetName().c_str(),
           DebugString(*var_desc).c_str());
    assigned = false;
    return SUCCESS;
  }
  GELOGI("Assign %s %s logical memory with desc %s", node->GetType().c_str(), var_name.c_str(),
         DebugString(*var_desc).c_str());
  GE_ASSERT_SUCCESS(session_var_manager_->AssignVarMem(var_name, node->GetOpDesc(), *var_desc, RT_MEMORY_HBM));
  assigned = true;
  return SUCCESS;
}

Status GraphVarVisitor::AssignVarLogicalMemory(const ge::NodePtr &node) {
  bool unused = false;
  return AssignVarLogicalMemory(node, unused);
}

Status GraphVarVisitor::GetVarDeviceInstance(const ge::NodePtr &node, Variable &var_instance) {
  const std::string &var_name = node->GetName();
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto &var_desc = node->GetOpDesc()->GetOutputDescPtr(0U);
  GE_ASSERT_NOTNULL(var_desc, "Output desc of %s %s is nullptr", node->GetType().c_str(), node->GetName().c_str());
  var_instance.desc = *var_desc;

  uint8_t *var_logic = nullptr;
  GE_ASSERT_SUCCESS(session_var_manager_->GetVarAddr(var_name, *var_desc, var_logic));

  rtMemType_t placement = IsVarPlacedOnRdma(node) ? RT_MEMORY_RDMA_HBM : RT_MEMORY_HBM;
  var_instance.placement =
      (placement == RT_MEMORY_HBM) ? VariablePlacement::kOnDeviceHbm : VariablePlacement::kOnDeviceRdma;
  var_instance.addr = session_var_manager_->GetVarMemoryAddr(var_logic, placement, device_id_);
  // Empty var hold non-null address same with next var as var memory length is 0
  GE_ASSERT_NOTNULL(var_instance.addr, "Get %s %s memory[type:%u] with desc %s on device %u failed",
                    node->GetType().c_str(), var_name.c_str(), static_cast<uint32_t>(var_instance.placement),
                    DebugString(*var_desc).c_str(), device_id_);

  GE_ASSERT_SUCCESS(GetTensorDescSize(*var_desc, var_instance.size));
  GE_ASSERT(var_instance.size >= 0U);
  return SUCCESS;
}

Status GraphVarVisitor::AssembleVariables(const std::vector<ge::NodePtr> &variables) {
  std::vector<ge::NodePtr> host_variables;
  std::vector<ge::NodePtr> device_variables;
  for (const auto &node : variables) {
    if (IsVarPlacedOnHost(node)) {
      GELOGI("Collect host %s %s", node->GetType().c_str(), node->GetName().c_str());
      host_variables.emplace_back(node);
      continue;
    }
    GELOGI("Collect device %s %s", node->GetType().c_str(), node->GetName().c_str());
    device_variables.emplace_back(node);
  }
  GE_ASSERT_SUCCESS(AssembleHostVariables(host_variables));
  GE_ASSERT_SUCCESS(AssembleDeviceVariables(device_variables));
  return ge::SUCCESS;
}

void *GraphVarVisitor::GetOrCreateVarMem(const string &var_name,
                                         const OpDescPtr &var_desc,
                                         const rtMemType_t memory_type) const {
  const GeTensorDesc &output_tensor = var_desc->GetOutputDesc(0U);
  if (session_var_manager_->IsVarExist(var_name, output_tensor)) {
    GELOGD("Skip initialized %s %s with out[0]", var_desc->GetType().c_str(), var_desc->GetName().c_str());
  } else {
    GELOGI("Assign %s %s logical memory with out[0].", var_desc->GetType().c_str(), var_name.c_str());
    GE_ASSERT_SUCCESS(session_var_manager_->AssignVarMem(var_name, var_desc, output_tensor, memory_type),
                      "[Assign][VarMem] for %s failed.", var_name.c_str());
  }
  uint8_t *var_addr = nullptr;
  GE_ASSERT_SUCCESS(session_var_manager_->GetVarAddr(var_name, output_tensor, var_addr),
                    "[Get][VarAddr] failed, var name[%s]", var_name.c_str());
  return static_cast<void *>(var_addr);
}

Status GraphVarVisitor::AssembleHostVariables(const std::vector<ge::NodePtr> &variables) {
  for (auto &node : variables) {
    const std::string &var_name = node->GetName();
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    const auto &var_desc = node->GetOpDesc()->GetOutputDescPtr(0U);
    GE_ASSERT_NOTNULL(var_desc, "Output desc of variable %s is nullptr", node->GetName().c_str());
    auto &var_instance = named_variables_[node->GetName()];
    var_instance.desc = *var_desc;
    GE_ASSERT_SUCCESS(GetTensorDescSize(*var_desc, var_instance.size));
    GE_ASSERT(var_instance.size >= 0U);

    ge::SharedMemInfo shared_memory_info;
    auto &mem_instance = MemManager::Instance().HostMemInstance(RT_MEMORY_HBM);
    if (HostMemManager::Instance().QueryVarMemInfo(var_name, shared_memory_info)) {
      GE_ASSERT(static_cast<uint64_t>(var_instance.size) <= shared_memory_info.mem_size,
                "Malloced shm %s with size %lu not enough for variable %s", var_name.c_str(),
                shared_memory_info.mem_size, var_instance.DebugString().c_str());
      var_instance.addr = static_cast<uint8_t *>(
          mem_instance.Malloc(shared_memory_info.host_aligned_ptr, static_cast<size_t>(var_instance.size)));
      GE_ASSERT_NOTNULL(var_instance.addr);
    } else {
      var_instance.addr = static_cast<uint8_t *>(GetOrCreateVarMem(var_name, node->GetOpDesc(), RT_MEMORY_HOST));
    }
    var_instance.placement = VariablePlacement::kOnHost;
    GELOGD("After assemble type %s, name %s, %s", node->GetType().c_str(), node->GetName().c_str(),
           var_instance.DebugString().c_str());
  }
  return ge::SUCCESS;
}

Status GraphVarVisitor::AssembleDeviceVariables(const std::vector<ge::NodePtr> &variables) {
  for (const auto &node : variables) {
    bool assigned_here = false;
    GE_ASSERT_SUCCESS(AssignVarLogicalMemory(node, assigned_here));
    if (assigned_here) {
      GELOGD("Set variable %s allocated graph id %u", node->GetName().c_str(), graph_id_);
      GE_ASSERT_SUCCESS(session_var_manager_->SetAllocatedGraphId(node->GetName(), graph_id_));
    }
  }
  // 这里RT1的流程就有个潜在的BUG,TransAllVarData中默认申请非RDMA的HBM内存，RDMA的Variable(推荐网络)无法正常Trans。
  // 当前从场景上认为没有需要TransRoad的RDMA变量
  GE_ASSERT_SUCCESS(ge::TransVarDataUtils::TransAllVarData(variables, session_id_, graph_id_, device_id_));
  for (const auto &node : variables) {
    auto &var_instance = named_variables_[node->GetName()];
    GE_ASSERT_SUCCESS(GetVarDeviceInstance(node, var_instance));
    GELOGD("After assemble type %s, name %s, %s", node->GetType().c_str(), node->GetName().c_str(),
           var_instance.DebugString().c_str());
  }
  return ge::SUCCESS;
}

Status GraphVarVisitor::AssembleDeviceSharedConstants(const vector<ge::NodePtr> &shared_constants) {
  GE_TIMESTAMP_START(AssembleSharedConstants);
  std::unordered_map<uint32_t, std::vector<SharedConstantCopyHelper>> helpers;
  uint32_t cur_thread = 0U;
  for (const auto &node : shared_constants) {
    const auto &constant_name = node->GetName();
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const auto &weights = ModelUtils::GetWeights(op_desc);
    GE_ASSERT(weights.size() == 1U, "Constant %s has unexpected num %zu weights", constant_name.c_str(),
              weights.size());
    const auto &weight = weights[0U];
    GE_ASSERT_NOTNULL(weight);

    GE_ASSERT_SUCCESS(AssignVarLogicalMemory(node));
    auto &var_instance = named_variables_[constant_name];
    GE_ASSERT_SUCCESS(GetVarDeviceInstance(node, var_instance));

    if (weight->GetData().size() == 0U) {
      GELOGD("Constant node[%s] weight size is 0, expected > 0, skip assembling.", node->GetName().c_str());
      continue;
    }
    helpers[cur_thread].emplace_back(SharedConstantCopyHelper{var_instance.size, var_instance.addr, weight, op_desc});
    ++cur_thread;
    cur_thread %= kMaxThreadNum;
    GELOGD("After assemble type %s, name %s, %s", node->GetType().c_str(), node->GetName().c_str(),
           var_instance.DebugString().c_str());
  }
  GE_ASSERT_SUCCESS(MultiThreadSharedConstantCopy(helpers));
  GE_TIMESTAMP_EVENT_END(AssembleSharedConstants, "AssembleSharedConstants");
  return SUCCESS;
}

Status GraphVarVisitor::CopySharedConstant(const std::shared_ptr<ge::VarManager> &var_manager, uint32_t device_id,
                                           const std::vector<SharedConstantCopyHelper> &helpers) const {
  // every thread needs to rtSetDevice
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(device_id)));
  GE_MAKE_GUARD(reset_device, [device_id]() {
      GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(device_id)));
  });

  for (const auto &helper : helpers) {
    if (!var_manager->CheckAndSetVarLoaded(helper.op_desc, device_id)) {
      GELOGD("Copy constant %s size %ld to device addr %p", helper.op_desc->GetNamePtr(), helper.size, helper.addr);
      GE_ASSERT_RT_OK(rtMemcpy(helper.addr, helper.size, helper.weight->GetData().data(),
                               helper.weight->GetData().size(), RT_MEMCPY_HOST_TO_DEVICE));
    } else {
      GELOGD("Constant %s size %ld has been loaded to device addr %p, no need to reload.", helper.op_desc->GetNamePtr(),
             helper.size, helper.addr);
    }
  }

  return SUCCESS;
}

using SharedConstantCopyFunc = std::function<Status(const GraphVarVisitor *, const std::shared_ptr<ge::VarManager> &,
                                                    const uint32_t, const std::vector<SharedConstantCopyHelper> &)>;

Status GraphVarVisitor::MultiThreadSharedConstantCopy(
    const std::unordered_map<uint32_t, std::vector<SharedConstantCopyHelper>> &helpers) const {
  GELOGD("Start to copy shared constant, num: %zu.", helpers.size());
  if (helpers.empty()) {
    return SUCCESS;
  }

  ThreadPool executor("ge_cpconst", static_cast<uint32_t>(helpers.size()), false);
  std::vector<std::future<Status>> vector_future;
  SharedConstantCopyFunc func = &GraphVarVisitor::CopySharedConstant;
  const auto &var_manager = VarManager::Instance(session_id_);
  for (auto &helper : helpers) {
    std::future<Status> f = executor.commit(func, this, var_manager, device_id_, helper.second);
    if (f.valid()) {
      vector_future.emplace_back(std::move(f));
    }
  }

  for (auto &i : vector_future) {
    GE_ASSERT_SUCCESS(i.get(), "Shared constant copy failed.");
  }
  GELOGD("Success to copy shared constant.");
  return SUCCESS;
}

Status GraphVarVisitor::AssembleHostSharedConstants(const vector<ge::NodePtr> &shared_constants) {
  for (const auto &node : shared_constants) {
    auto constant_name = node->GetName();
    const auto &weights = ModelUtils::GetWeights(node->GetOpDesc());
    GE_ASSERT(weights.size() == 1U, "Constant %s has unexpected num %zu weights", constant_name.c_str(),
              weights.size());
    const auto &weight = weights[0U];
    GE_ASSERT_NOTNULL(weight);

    const auto &constant_desc = node->GetOpDesc()->GetOutputDescPtr(0U);
    GE_ASSERT_NOTNULL(constant_desc, "Output desc of variable %s is nullptr", node->GetName().c_str());
    auto &var_instance = named_variables_[constant_name];
    var_instance.desc = *constant_desc;
    var_instance.size = weight->GetData().size();
    GE_ASSERT(var_instance.size >= 0U);

    if (weight->GetData().size() == 0U) {
      GELOGD("Constant node[%s] weight size is 0, expected > 0, skip assembling.", node->GetName().c_str());
      continue;
    }
    // todo here directly use weight addr, later move to host resource manager
    GE_ASSERT_NOTNULL(weight->GetData().GetData());
    var_instance.addr = const_cast<uint8_t *>(weight->GetData().GetData());
    var_instance.placement = VariablePlacement::kOnHost;
    // todo ATTR_VARIABLE_PLACEMENT属性和Host exec flag option语义重复了，最好归一
    // HostExecFlag若为true，ATTR_VARIABLE_PLACEMENT属性一定为host
    // 如编译时只判断Host exec flag，加载时将flag转换为var/constantop上的属性
    (void) AttrUtils::SetStr(node->GetOpDesc(), ATTR_VARIABLE_PLACEMENT, "host");
    GELOGD("After assemble type %s, name %s, %s", node->GetType().c_str(), node->GetName().c_str(),
           var_instance.DebugString().c_str());
  }
  return SUCCESS;
}

Status GraphVarVisitor::AssembleSharedConstants(const std::vector<ge::NodePtr> &shared_constants) {
  if (is_visitor_for_host_) {  // Keep constant placed on host when execute on host-cpu
    GELOGD("Assemble constant for host execute graph %u", graph_id_);
    return AssembleHostSharedConstants(shared_constants);
  }
  return AssembleDeviceSharedConstants(shared_constants);
}

ge::Status GraphVarVisitor::LoadFileConstantToAddr(const ge::OpDescPtr &op_desc, const Variable &var_instance) const {
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  std::map<std::string, std::string> file_id_and_path_map;
  GE_ASSERT_SUCCESS(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map),
                    "Failed to get FILE_CONSTANT_PATH option.");
  GE_ASSERT_SUCCESS(FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map, file_path, offset, length),
                    "Failed to get file path.");
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(session_id_);
  GE_ASSERT_NOTNULL(external_weight_manager);
  if (!external_weight_manager->CheckAndSetWeightLoaded(file_path + ":" + std::to_string(offset), device_id_)) {
    int64_t weight_size = 0;
    GE_ASSERT_GRAPH_SUCCESS(TensorUtils::GetTensorSizeInBytes(var_instance.desc, weight_size),
                            "Failed to get file constant size by tensor desc.");
    GE_ASSERT_TRUE(weight_size <= var_instance.size, "Weight size[%ld] can not be larger than mem size[%ld].",
                   weight_size, var_instance.size);
    const size_t file_length = (length == 0U ? static_cast<size_t>(weight_size) : length);
    size_t left_size = static_cast<size_t>(var_instance.size);
    const auto dev_addr = static_cast<void *>(var_instance.addr);
    GE_ASSERT_SUCCESS(FileConstantUtils::CopyOneWeightFromFile(dev_addr, file_path, offset, file_length, left_size));
    GELOGD("Load file constant [%s] weight size [%ld] to addr [%p] success.", op_desc->GetName().c_str(),
           var_instance.size, var_instance.addr);
  } else {
    GELOGD("File constant [%s] weight size [%ld] has been loaded to addr [%p], no need to reload.",
           op_desc->GetName().c_str(), var_instance.size, var_instance.addr);
  }
  return ge::SUCCESS;
}

Status GraphVarVisitor::AssembleFileConstantsOnHost(const std::vector<ge::NodePtr> &file_constants) {
  GE_TIMESTAMP_START(AssembleFileConstantsOnHost);
  // 传入host memorytype，使用AssignVarMem构建var_addr_mgr_map_，并将var_addr_mgr_map_的物理地址拿出
  // host memorytype中的var_addr_mgr_map_存的就是物理地址
  GELOGD("Exec file constant on host.");
  for (const auto &file_constant_node : file_constants) {
    const auto &var_desc = file_constant_node->GetOpDesc()->GetOutputDescPtr(0U);
    GE_ASSERT_NOTNULL(var_desc, "Output desc of file constant %s is nullptr.", file_constant_node->GetName().c_str());
    auto &var_instance = named_variables_[file_constant_node->GetName()];
    var_instance.desc = *var_desc;
    GE_ASSERT_SUCCESS(GetTensorDescSize(*var_desc, var_instance.size));
    GE_ASSERT(var_instance.size >= 0U);
    var_instance.addr = static_cast<uint8_t *>(
        GetOrCreateVarMem(file_constant_node->GetName(), file_constant_node->GetOpDesc(), RT_MEMORY_HOST));
    var_instance.placement = VariablePlacement::kOnHost;

    GE_ASSERT_SUCCESS(LoadFileConstantToAddr(file_constant_node->GetOpDesc(), var_instance));
    GELOGD("After assemble type %s, name %s, %s", file_constant_node->GetType().c_str(),
           file_constant_node->GetName().c_str(), var_instance.DebugString().c_str());
  }
  GE_TIMESTAMP_EVENT_END(AssembleFileConstantsOnHost, "AssembleFileConstantsOnHost");
  return ge::SUCCESS;
}

Status GraphVarVisitor::PreLoadFileConstant(const ge::OpDescPtr &op_desc, const Variable &var_instance,
                                            const std::map<std::string, std::string> &file_id_path,
                                            H2DCopyHelper &helper) const {
  helper.node_name = op_desc->GetName();
  helper.dev_addr = static_cast<void *>(var_instance.addr);
  size_t length = 0U;
  GE_ASSERT_SUCCESS(FileConstantUtils::GetFilePath(op_desc, file_id_path, helper.file_path, helper.offset, length),
                    "Failed to get file path.");
  int64_t weight_size = 0;
  GE_ASSERT_GRAPH_SUCCESS(TensorUtils::GetTensorSizeInBytes(var_instance.desc, weight_size),
                          "Failed to get file constant size by tensor desc.");
  GE_ASSERT_TRUE(weight_size <= var_instance.size, "Weight size[%ld] can not be larger than mem size[%ld].",
                 weight_size, var_instance.size);
  helper.file_length = (length == 0U ? static_cast<size_t>(weight_size) : length);
  helper.left_size = static_cast<size_t>(var_instance.size);
  GELOGD("Finish to prepare for loading file constant for %s.", op_desc->GetNamePtr());

  return SUCCESS;
}

Status GraphVarVisitor::LoadFileConstantToDevice(const ExternalWeightManagerPtr &manager, const uint32_t device_id,
                                                 std::vector<H2DCopyHelper> &node_infos) const {
  // every thread needs to rtSetDevice
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(device_id)));
  GE_MAKE_GUARD(reset_device, [device_id]() {
    GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(device_id)));
  });

  for (auto &helper : node_infos) {
    // different graphs' nodes may use same files
    if (manager->CheckAndSetWeightLoaded(helper.file_path + ":" + std::to_string(helper.offset), device_id)) {
      GELOGD("File constant [%s], file path [%s] weight size [%ld] has been loaded to addr [%p], no need to reload.",
             helper.node_name.c_str(), helper.file_path.c_str(), helper.file_length, helper.dev_addr);
      continue;
    }
    GE_ASSERT_SUCCESS(FileConstantUtils::CopyOneWeightFromFile(helper.dev_addr, helper.file_path, helper.offset,
                                                               helper.file_length, helper.left_size));
    GELOGD("Load file constant [%s] file path [%s] weight size [%ld] to addr [%p] success.", helper.node_name.c_str(),
           helper.file_path.c_str(), helper.file_length, helper.dev_addr);
  }

  return SUCCESS;
}

using H2DCopyFunc = std::function<Status(GraphVarVisitor *, const ExternalWeightManagerPtr &, const uint32_t,
                                         std::vector<H2DCopyHelper> &)>;

Status GraphVarVisitor::MultiThreadH2DCopy(std::unordered_map<uint32_t, std::vector<H2DCopyHelper>> &multi_node_infos) {
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(session_id_);
  GE_ASSERT_NOTNULL(external_weight_manager);

  ThreadPool executor("ge_ldfconst", static_cast<uint32_t>(multi_node_infos.size()), false);
  std::vector<std::future<Status>> vector_future;
  H2DCopyFunc func = &GraphVarVisitor::LoadFileConstantToDevice;
  for (auto &node_infos : multi_node_infos) {
    std::future<Status> f = executor.commit(func, this, external_weight_manager, device_id_, node_infos.second);
    if (f.valid()) {
      vector_future.emplace_back(std::move(f));
    }
  }

  for (auto &i : vector_future) {
    GE_ASSERT_SUCCESS(i.get(), "H2D copy failed.");
  }
  return SUCCESS;
}

Status GraphVarVisitor::AssembleFileConstantsOnDevice(const std::vector<ge::NodePtr> &file_constants) {
  GE_TIMESTAMP_START(AssembleFileConstantsOnDevice);
  GELOGD("Exec file constant on device.");
  std::map<std::string, std::string> file_id_path;
  GE_ASSERT_SUCCESS(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_path),
                    "Failed to get FILE_CONSTANT_PATH option.");
  std::unordered_map<uint32_t, std::vector<H2DCopyHelper>> multi_node_infos;
  std::unordered_map<std::string, GeTensorDesc> name_descs;
  std::set<std::string> file_paths;
  uint32_t cur_thread = 0U;
  for (const auto &file_constant_node : file_constants) {
    bool assigned_here = false;
    GE_ASSERT_SUCCESS(AssignVarLogicalMemory(file_constant_node, assigned_here));
    const auto &node_name = file_constant_node->GetName();
    if (assigned_here) {
      GELOGD("Set variable %s allocated graph id %u", node_name.c_str(), graph_id_);
      GE_ASSERT_SUCCESS(session_var_manager_->SetAllocatedGraphId(node_name, graph_id_));
    }

    auto &var_instance = named_variables_[node_name];
    GE_ASSERT_SUCCESS(GetVarDeviceInstance(file_constant_node, var_instance));
    if (!session_var_manager_->IsVarReady(node_name, var_instance.desc, device_id_)) {
      name_descs.emplace(std::make_pair(node_name, var_instance.desc));
      H2DCopyHelper helper;
      GE_ASSERT_SUCCESS(PreLoadFileConstant(file_constant_node->GetOpDesc(), var_instance, file_id_path, helper));
      if (file_paths.emplace(helper.file_path).second) {
        multi_node_infos[cur_thread].emplace_back(std::move(helper));
        ++cur_thread;
        cur_thread %= kMaxThreadNum;
      }
    }
    GELOGD("After assemble type %s, name %s, %s", file_constant_node->GetTypePtr(), node_name.c_str(),
           var_instance.DebugString().c_str());
  }
  GE_ASSERT_SUCCESS(MultiThreadH2DCopy(multi_node_infos));
  for (const auto &name_desc : name_descs) {
    session_var_manager_->SetVarIsReady(name_desc.first, name_desc.second, device_id_);
  }
  GE_TIMESTAMP_EVENT_END(AssembleFileConstantsOnDevice, "AssembleFileConstantsOnDevice");
  return ge::SUCCESS;
}

Status GraphVarVisitor::AssembleFileConstants(const std::vector<ge::NodePtr> &file_constants) {
  // 根据fileconstant是否在host上执行，使用不同的memtype进行AssignVarMem内存共享
  return (is_visitor_for_host_) ? AssembleFileConstantsOnHost(file_constants)
                                : AssembleFileConstantsOnDevice(file_constants);
}

Status GraphVarVisitor::Init(const std::shared_ptr<ge::VarManager> &var_manager, uint32_t device_id,
                             uint64_t session_id, uint32_t graph_id) {
  session_var_manager_ = var_manager;
  GE_ASSERT_NOTNULL(session_var_manager_);
  session_id_ = session_id;
  device_id_ = device_id;
  graph_id_ = graph_id;
  is_visitor_for_host_ = ge::GetContext().GetHostExecFlag();
  return SUCCESS;
}

Status GraphVarVisitor::GetVarShapeAndMemory(const std::string &id, gert::StorageShape &shape,
                                             gert::TensorData &memory) const {
  auto iter = named_variables_.find(id);
  GE_ASSERT(iter != named_variables_.end(), "Variable %s not found", id.c_str());

  GeShapeAsRtShape(iter->second.desc.GetOriginShape(), shape.MutableOriginShape());
  GeShapeAsRtShape(iter->second.desc.GetShape(), shape.MutableStorageShape());

  memory.SetPlacement(iter->second.placement == VariablePlacement::kOnHost ? gert::TensorPlacement::kOnHost
                                                                           : gert::TensorPlacement::kOnDeviceHbm);
  memory.SetAddr(iter->second.addr, nullptr);
  memory.SetSize(iter->second.size);
  return SUCCESS;
}

HybridModelRtV2Executor::HybridModelRtV2Executor(HybridModel *const model, uint32_t device_id, const rtStream_t stream)
    : HybridModelExecutor(model, device_id, stream), num_inputs_(0U), num_outputs_(0U) {
}

Status HybridModelRtV2Executor::RunCtx::Init(HybridModel *model) {
  GE_TIMESTAMP_START(RunCtxInit);
  GE_ASSERT_NOTNULL(model);
  device_id_ = model->device_id_;
  GE_ASSERT_NOTNULL(model->ge_root_model_);
  GE_ASSERT_SUCCESS(GetModelSessionId(model->ge_root_model_, session_id_));
  const auto &root_graph = model->ge_root_model_->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  graph_id_ = root_graph->GetGraphID();
  graph_name_ = root_graph->GetName();
  auto var_manager = ge::VarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(var_manager);
  GE_ASSERT_SUCCESS(EnsureModelVarMemoryMalloced(model->ge_root_model_, var_manager, device_id_));

  GE_ASSERT_SUCCESS(graph_var_visitor_.Init(var_manager, device_id_, session_id_, graph_id_));

  VarMgrNodes var_mgr_nodes;
  CollectGraphVarMgrNodes(root_graph, var_mgr_nodes);

  GELOGD("Start assemble %zu variables of graph %s", var_mgr_nodes.variables.size(), graph_name_.c_str());
  GE_ASSERT_SUCCESS(graph_var_visitor_.AssembleVariables(var_mgr_nodes.variables));
  GELOGD("Start assemble %zu shared constants of graph %s", var_mgr_nodes.shared_constants.size(), graph_name_.c_str());
  GE_ASSERT_SUCCESS(graph_var_visitor_.AssembleSharedConstants(var_mgr_nodes.shared_constants));
  GELOGD("Start assemble %zu file constants of graph %s", var_mgr_nodes.file_constants.size(), graph_name_.c_str());
  GE_ASSERT_SUCCESS(graph_var_visitor_.AssembleFileConstants(var_mgr_nodes.file_constants));
  session_.SetSessionId(session_id_);
  session_.SetVarManager(&graph_var_visitor_);
  HybridModelExecutor::ParserContextOption(OPTION_EXEC_DYNAMIC_EXECUTE_MODE, execute_mode_);
  HybridModelExecutor::ParserContextOption(OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, is_copy_output_addr_);
  std::string iterations_per_loop;
  if (model->ge_root_model_->GetRootGraph()->GetNeedIteration()) {
    HybridModelExecutor::ParserContextOption(ATTR_NAME_ITERATORS_PER_LOOP, iterations_per_loop);
  }
  const int32_t kDecimalBase = 10;
  const int64_t kOneStep = 1;
  auto loop_config = strtol(iterations_per_loop.c_str(), nullptr, kDecimalBase);
  iterations_per_loop_ = static_cast<size_t>(std::max<int64_t>(loop_config, kOneStep));
  host_exec_flag_ = ge::GetContext().GetHostExecFlag();

  string enable_input_batch_cpy_str;
  (void)GetThreadLocalContext().GetOption(configure_option::INPUT_BATCH_CPY, enable_input_batch_cpy_str);
  enable_input_batch_cpy_ = (enable_input_batch_cpy_str == "1");

  bool need_set_stream_core_limits = false;
  ge::AttrUtils::GetBool(model->ge_root_model_->GetRootGraph(), "need_set_stream_core_limits", need_set_stream_core_limits);
  if (need_set_stream_core_limits) {
    ParserContextOption(AICORE_NUM, aicore_num_str_);
    ParserContextOption(kVectorcoreNum, vectorcore_num_str_);
  }

  GELOGI("Get npu iterations per loop [%s] and will loop %zu for graph [%s], input_batch_cpy_:%d.", iterations_per_loop.c_str(),
         iterations_per_loop_, graph_name_.c_str(), static_cast<int32_t>(enable_input_batch_cpy_));
  GE_TIMESTAMP_EVENT_END(RunCtxInit, "RunCtxInit::ALL");
  return ge::SUCCESS;
}

static void InitTraningTraceInfo(const ComputeGraphPtr compute_graph,
                                 std::unordered_map<std::string, gert::TraceAttr> &node_names_to_attrs) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    bool is_fp = false;
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_INSERT_FP_PROFILILNG_TASK, is_fp)) {
      node_names_to_attrs[node->GetName()].is_fp = is_fp;
    }
    bool is_bp = false;
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_INSERT_BP_PROFILILNG_TASK, is_bp)) {
      node_names_to_attrs[node->GetName()].is_bp = is_bp;
    }
    int64_t log_id = -1;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID, log_id) && log_id != -1) {
      node_names_to_attrs[node->GetName()].start_log_id = log_id;
    }
    node_names_to_attrs[node->GetName()].logic_stream_id = node->GetOpDescBarePtr()->GetStreamId();
    GELOGD("InitPrivateAttrs, node type: %s, is_fp: %d, is_bp: %d, log id: %lld, logic_stream_id: %lld",
           node->GetType().c_str(), is_fp, is_bp, log_id, node_names_to_attrs[node->GetName()].logic_stream_id);
  }
}

Status HybridModelRtV2Executor::CheckInputIsOnDevice() {
  // 如果输入Data是kOnDeviceHbm，后续rt2 lowering时就会减少构图，避免执行时判断是否需要H2D
  if (!run_ctx_.host_exec_flag_ && IsInputPlacementOnDeviceHbm()) {
    for (size_t i = 0U; i < num_inputs_; i++) {
      GE_ASSERT_TRUE(rt_inputs_[i]->GetPlacement() == gert::kOnDeviceHbm);
    }
  }
  return SUCCESS;
}

Status HybridModelRtV2Executor::InitCtx() {
  GE_ASSERT_SUCCESS(run_ctx_.Init(model_));
  if (run_ctx_.host_exec_flag_ && gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump) != 0UL) {
    diagnoseSwitch::DisableDumper();
    diagnoseSwitch::EnableHostDump();
  }

  context_.ge_context = &ge_context_;
  std::map<std::string, std::string> options;
  std::string static_memory_policy;
  HybridModelExecutor::ParserContextOption(STATIC_MEMORY_POLICY, static_memory_policy);
  if (!static_memory_policy.empty()) {
    options.insert(std::make_pair(STATIC_MEMORY_POLICY, static_memory_policy));
  }
  ge_context_.SetGlobalOption(options);
  return ge::SUCCESS;
}

Status HybridModelRtV2Executor::LoadGuardFunc(const ge::ComputeGraphPtr &graph) {
  const std::string *buffer = AttrUtils::GetStr(graph, kGuardCheckSoDataResult);
  if ((buffer == nullptr) || buffer->empty()) {
    return ge::SUCCESS;
  }
  GELOGI("Start load guard check func");
  if (guard_check_info_.guard_so_fd == -1) {
    guard_check_info_.guard_so_fd =
        static_cast<int32_t>(syscall(__NR_memfd_create, kGuardCheckSoName.c_str(), 0));
  }
  const auto write_count = mmWrite(guard_check_info_.guard_so_fd,
      const_cast<char_t *>(buffer->c_str()), buffer->size());
  GE_ASSERT_TRUE(((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR)),
      "Write data failed, errno: %lld", write_count);
  (void)lseek(static_cast<int32_t>(guard_check_info_.guard_so_fd), 0, SEEK_SET);
  std::string so_path = "/proc/self/fd/" + std::to_string(guard_check_info_.guard_so_fd);
  // 通过/proc访问文件描述符对应的"文件"
  guard_check_info_.guard_handle = mmDlopen(so_path.c_str(), static_cast<int32_t>(MMPA_RTLD_NOW));
  GE_ASSERT_NOTNULL(guard_check_info_.guard_handle);
  guard_check_info_.guard_check_func =
      reinterpret_cast<GuardCheckFunc>(mmDlsym(guard_check_info_.guard_handle, "GuardCheckFunc"));
  GE_ASSERT_NOTNULL(guard_check_info_.guard_check_func);
  return ge::SUCCESS;
}

Status HybridModelRtV2Executor::CheckGuard() {
  if (guard_check_info_.guard_check_func == nullptr) {
    return ge::SUCCESS;
  }
  GELOGI("Start Check guard");
  char_t reason[kMaxStringSize];
  GE_ASSERT_TRUE(guard_check_info_.guard_check_func(rt_inputs_.data(), rt_inputs_.size(),
      reason, kMaxStringSize), reason);
  return SUCCESS;
}

Status HybridModelRtV2Executor::Init(CallbackManager *const callback_manager) {
  (void)callback_manager;
  GE_ASSERT_SUCCESS(InitCtx());
  auto ge_root_model = model_->ge_root_model_;
  GE_ASSERT_NOTNULL(ge_root_model);
  auto root_graph = ge_root_model->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  name_ = root_graph->GetName();
  model_id_ = model_->GetModelId();
  ge_root_model->SetCurModelId(model_id_);
  GELOGI("Current id of hybrid model %s is %u.", ge_root_model->GetModelName().c_str(), ge_root_model->GetCurModelId());
  for (auto &named_model : ge_root_model->GetSubgraphInstanceNameToModel()) {
    GE_ASSERT_NOTNULL(named_model.second, "Compiled model of graph %s is nullptr", named_model.first.c_str());
    named_model.second->SetModelId(model_id_);
  }

  executor_ =
      gert::RtV2ExecutorFactory::Create(ge_root_model, run_ctx_.dev_resource_allocator_, &run_ctx_.session_);
  GE_ASSERT_NOTNULL(executor_, "Failed create rt2 executor for model %s", name_.c_str());
  profiler_collector_ =
      std::unique_ptr<ProfilerCollector>(new (std::nothrow) ProfilerCollector(model_id_, run_ctx_.graph_id_));
  GE_ASSERT_NOTNULL(profiler_collector_);
  if (gert::GlobalProfilingWrapper::GetInstance()->GetEnableFlags() > 0UL) {
    gert::SubscriberExtendInfo extend_info;
    InitTraningTraceInfo(ge_root_model->GetFlattenGraph(), extend_info.node_names_to_attrs);
    extend_info.model_id = model_id_;
    extend_info.stream = stream_;
    executor_->AddSubscriber(extend_info);
  }

  auto model_input_desc = executor_->GetAllInputsDesc(num_inputs_);
  auto model_output_desc = executor_->GetAllOutputsDesc(num_outputs_);
  bool enable_dynamic_batch = false;
  (void)ge::AttrUtils::GetBool(root_graph, "_enable_dynamic_batch", enable_dynamic_batch);
  if (enable_dynamic_batch) {
    GELOGI("Hybrid model of graph %s is multi batch, should exclude the last input", name_.c_str());
    num_inputs_--;
  }
  GELOGI("Hybrid model of graph %s has %zu inputs and %zu outputs", name_.c_str(), num_inputs_, num_outputs_);
  inputs_holder_.resize(num_inputs_);
  for (size_t i = 0U; i < num_inputs_; i++) {
    auto &holder = inputs_holder_[i];
    holder.SetPlacement(gert::kTensorPlacementEnd);
    holder.SetDataType(static_cast<ge::DataType>(model_input_desc[i].GetDataType()));
    holder.SetOriginFormat(model_input_desc[i].GetOriginFormat());
    holder.SetStorageFormat(model_input_desc[i].GetStorageFormat());
    rt_inputs_.emplace_back(&holder);
    GELOGI("Input %zu %s", i, DebugString(holder, false).c_str());
  }

  outputs_holder_.resize(num_outputs_);
  output_descs_.reserve(num_outputs_);
  for (size_t i = 0U; i < num_outputs_; i++) {
    auto dtype = static_cast<ge::DataType>(model_output_desc[i].GetDataType());
    auto origin_format = model_output_desc[i].GetOriginFormat();
    auto storage_format = model_output_desc[i].GetStorageFormat();

    auto &holder = outputs_holder_[i];
    holder.SetData(gert::TensorData(nullptr));
    holder.SetDataType(dtype);
    holder.SetOriginFormat(origin_format);
    holder.SetStorageFormat(storage_format);
    holder.SetPlacement(gert::kTensorPlacementEnd);
    rt_outputs_.emplace_back(&holder);
    GELOGI("  Output %zu %s", i, DebugString(holder, false).c_str());

    output_descs_.emplace_back(MakeShared<GeTensorDesc>(ge::GeShape(), storage_format, dtype));
    GE_ASSERT_NOTNULL(output_descs_.back(), "Failed create output %zu tensor desc", i);
    output_descs_.back()->SetOriginFormat(origin_format);
  }
  GELOGI("Load model of graph: %s, session: %lu, device: %u", run_ctx_.graph_name_.c_str(), run_ctx_.session_id_,
         run_ctx_.device_id_);
  gert::ModelExecuteArg model_args;
  model_args.stream = stream_;
  model_args.external_stream_allocator = &(run_ctx_.dev_resource_allocator_.stream_allocator);
  model_args.external_event_allocator = &(run_ctx_.dev_resource_allocator_.event_allocator);
  model_args.external_notify_allocator = &(run_ctx_.dev_resource_allocator_.notify_allocator);
  const auto allocators = allocator_manager_.GetAllocator(run_ctx_.graph_name_, model_args.stream);
  GE_ASSERT_NOTNULL(allocators, "Failed get scalable allocators");
  model_args.external_allocator = allocators;
  gert::ModelLoadArg load_args(&run_ctx_.session_);
  GE_ASSERT_GRAPH_SUCCESS(executor_->Load(model_args, load_args), "Failed load rt v2 model for graph %s",
                          name_.c_str());
  return ge::SUCCESS;
}

Status HybridModelRtV2Executor::ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
                                                   std::shared_ptr<ModelListener> listener) {
  GELOGI("HybridModel will execute in rt2.0 mode");
  HybridModelExecutor::CtrlArgs ctr_args;
  GE_TIMESTAMP_START(Execute);
  std::vector<gert::Tensor> outputs;
  auto execute_status = Execute(inputs, outputs, ctr_args);
  GE_TIMESTAMP_END(Execute, "HybridModelRtV2Executor::Execute");
  iterator_count_++;
  GELOGI("run iterator count is %lu, model_id: %u", iterator_count_, model_->GetModelId());
  ge::ScopeGuard guarder([this]() { StepDoneV2(); });
  GE_TIMESTAMP_START(HandleResult);
  execute_status = HandleResult(execute_status, 0U, ctr_args, outputs, listener);
  GE_TIMESTAMP_END(HandleResult, "HybridModelRtV2Executor::HandleResult");
  return execute_status;
}

Status HybridModelRtV2Executor::HandleResult(const Status exec_ret,
                                             const uint32_t data_id,
                                             HybridModelExecutor::CtrlArgs &ctrl_args,
                                             std::vector<gert::Tensor> &outputs,
                                             std::shared_ptr<ModelListener> listener) const {
  GELOGI("Start to handle result. model id = %u, data index = %u, execution ret = %u", model_id_, data_id, exec_ret);
  std::vector<gert::Tensor> host_outputs;
  if (ctrl_args.is_eos) {
    RecycleOutputs(outputs, stream_);
    GELOGI("End of sequence, model id = %u.", model_id_);
    GE_CHK_STATUS_RET_NOLOG(OnComputeDone(data_id, END_OF_SEQUENCE, host_outputs, listener));
    return SUCCESS;
  }

  if (exec_ret != SUCCESS) {
    RecycleOutputs(outputs, stream_);
    GELOGE(exec_ret, "[Check][Param:Status] failed to execute graph. model_id = %u", model_id_);
    REPORT_INNER_ERR_MSG("E19999", "failed to execute graph. model_id = %u", model_id_);
    return OnComputeDone(data_id, INTERNAL_ERROR, host_outputs, listener);
  }

  const auto ret = CopyOutputs(outputs, host_outputs);
  if (ret != SUCCESS) {
    RecycleOutputs(outputs, stream_);
    (void)OnComputeDone(data_id, INTERNAL_ERROR, host_outputs, listener);
    return INTERNAL_ERROR;
  }
  // OnComputeDone会调用tf（或用户）的回调函数，可能会触发下一个迭代的执行，因此要在OnComputeDone前释放内存，并触发allocator回收。
  RecycleOutputs(outputs, stream_);
  GELOGI("Executed graph successfully, model id = %u, data_index = %u.", model_id_, data_id);
  return OnComputeDone(data_id, SUCCESS, host_outputs, listener);
}

/*
 * 为什么不调用StepDone？
 * StepDone中遍历rt_outputs_，释放内存。
 * rt_outputs_保存gert::Tensor裸指针，指向outputs中的对象。outputs.clear()会触发内存释放，
 * 因此rt_outputs_中变成了无效指针，在为外部StepDoneV2中会赋值为null。
 */
Status HybridModelRtV2Executor::RecycleOutputs(std::vector<gert::Tensor> &outputs, const rtStream_t stream) const {
  GELOGI("recycle outputs. graph %s, stream: %p", name_.c_str(), stream);
  outputs.clear();
  GE_ASSERT_SUCCESS(AllocatorRecycle(stream_));
  return SUCCESS;
}

Status HybridModelRtV2Executor::AllocatorRecycle(const rtStream_t stream) const {
  const auto allocators = const_cast<ScalableAllocatorManager *>(&allocator_manager_)->GetAllocator("", stream);
  if (allocators != nullptr) {
    auto *allocator =
        allocators->GetAllocator(gert::kOnDeviceHbm, static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput));
    if (allocator != nullptr) {
      auto mem_synchronizer = dynamic_cast<gert::memory::MemSynchronizer*>(allocator);
      if (mem_synchronizer != nullptr) {
        GELOGI("caching memory allocator recycle. graph %s, stream: %p", name_.c_str(), stream);
        mem_synchronizer->Recycle();
      }
    }
  }
  return SUCCESS;
}

Status HybridModelRtV2Executor::PrepareInputData(InputData &current_data,
                                                 const HybridModelExecutor::ExecuteArgs &args) const {
  if (args.input_desc.size() != args.inputs.size()) {
    GELOGE(PARAM_INVALID, "[Check][Size]Input size mismatches, Input desc size %zu and input size %zu are not same",
           args.input_desc.size(), args.inputs.size());
    REPORT_INNER_ERR_MSG("E19999", "Input size mismatches, Input desc size %zu and input size %zu are not same.",
                       args.input_desc.size(), args.inputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0UL; i < args.inputs.size(); ++i) {
    DataBuffer buffer;
    buffer.data = const_cast<void *>(args.inputs[i].GetData());
    GE_ASSERT_NOTNULL(buffer.data, "Failed get inputs data blob");
    buffer.length = args.inputs[i].GetSize();
    buffer.placement = ((args.inputs[i].GetMemType() == MemStorageType::HBM))
                           ? static_cast<uint32_t>(Placement::kPlacementDevice)
                           : static_cast<uint32_t>(Placement::kPlacementHost);
    current_data.blobs.emplace_back(buffer);
    std::vector<int64_t> shape_value;
    for (size_t j = 0UL; j < args.input_desc[i]->GetShape().GetDimNum(); ++j) {
      shape_value.emplace_back(args.input_desc[i]->GetShape().GetDim(j));
    }
    current_data.shapes.emplace_back(shape_value);
    shape_value.clear();
  }
  return SUCCESS;
}

Status HybridModelRtV2Executor::PostProcResult(std::vector<GeTensor> &outputs) const {
  GE_ASSERT_EQ(outputs.size(), num_outputs_);
  for (size_t i = 0U; i < num_outputs_; ++i) {
    auto &rt_output = rt_outputs_[i];
    GELOGI("  Output %zu %s", i, DebugString(*rt_output).c_str());
    auto placement = (gert::TensorPlacementUtils::IsOnDevice(rt_output->GetPlacement())) ? Placement::kPlacementDevice
                                                                                   : Placement::kPlacementHost;
    auto &desc = output_descs_[i];
    // output with string datatype will get -1 * shapeSize
    size_t buffer_size = 0UL;
    if (desc->GetDataType() == ge::DT_STRING) {
      buffer_size = rt_output->GetSize();
    } else {
      buffer_size = GetSizeInBytes(rt_output->GetShapeSize(), desc->GetDataType());
    }
    auto output_holder = MakeShared<gert::TensorData>();
    GE_ASSERT_NOTNULL(output_holder);
    output_holder->ShareFrom(rt_output->MutableTensorData());
    const auto deleter = [output_holder](uint8_t *data) {
      (void) data;
      output_holder->Free();
    };
    GE_ASSERT_GRAPH_SUCCESS(
        outputs[i].MutableData().SetData(ge::PtrToPtr<void, uint8_t>(rt_output->GetAddr()), buffer_size, deleter));
    GELOGI("Output index %zu data type is %d. Get or calculate buffer size %zu",
           i, static_cast<int32_t>(desc->GetDataType()), buffer_size);
    RtShapeAsGeShape(rt_output->GetStorageShape(), outputs[i].MutableTensorDesc().MutableShape());
    RtShapeAsGeShape(rt_output->GetOriginShape(), outputs[i].MutableTensorDesc().MutableOriginShape());
    outputs[i].MutableTensorDesc().SetFormat(rt_output->GetStorageFormat());
    outputs[i].MutableTensorDesc().SetOriginFormat(rt_output->GetOriginFormat());
    outputs[i].MutableTensorDesc().SetPlacement(placement);
    outputs[i].MutableTensorDesc().SetDataType(desc->GetDataType());
  }
  return ge::SUCCESS;
}

static Status InputTensorValidate(const std::vector<gert::Tensor> &inputs, size_t inputs_num,
                                  bool host_exec_flag, uint8_t logLevel) {
  GE_ASSERT_EQ(inputs.size(), inputs_num);
  for (size_t i = 0U; i < inputs_num; ++i) {
    size_t size = inputs[i].GetSize();
    auto address = reinterpret_cast<const void*>(inputs[i].GetAddr());
    if (host_exec_flag) {
      GE_ASSERT_EQ(inputs[i].GetPlacement(), gert::kOnHost);
    } else if (gert::TensorPlacementUtils::IsOnDevice(inputs[i].GetPlacement())) {
      if (logLevel <= DLOG_DEBUG) {
        GELOGD(
            "input[%zu] addres = %p, size = %zu, placement = %u, which is on device, no need do alloc memory and "
            "rtmemcpy", i, address, size, inputs[i].GetPlacement());
      }
    } else {
      GE_ASSERT(gert::TensorPlacementUtils::IsOnHostNotFollowing(inputs[i].GetPlacement()),
                "Input %zu has unexpected placement %d", i, inputs[i].GetPlacement());
    }
    if (logLevel <= DLOG_INFO) {
      GELOGI("Input %zu %s", i, DebugString(inputs[i]).c_str());
    }
  }
  return SUCCESS;
}

Status HybridModelRtV2Executor::TryUpdateStreamCoreLimits(const rtStream_t stream) {
  bool update_stream_core_num = false;
  if (!run_ctx_.aicore_num_str_.empty()) {
    int32_t aicore_num = -1;
    GE_CHK_STATUS_RET(CoreNumUtils::ParseAndValidateCoreNum(ge::GetContext().GetReadableName(AICORE_NUM), run_ctx_.aicore_num_str_, 0, INT32_MAX, aicore_num));
    if (aicore_num > 0) {
      GE_CHK_RT_RET(rtsSetStreamResLimit(stream, RT_DEV_RES_CUBE_CORE, static_cast<uint32_t>(aicore_num)));
      update_stream_core_num = true;
    }
  }

  if (!run_ctx_.vectorcore_num_str_.empty()) {
    int32_t vectorcore_num = -1;
    GE_CHK_STATUS_RET(CoreNumUtils::ParseAndValidateCoreNum(ge::GetContext().GetReadableName(kVectorcoreNum), run_ctx_.vectorcore_num_str_, 0, INT32_MAX, vectorcore_num));
    if (vectorcore_num > 0) {
      GE_CHK_RT_RET(rtsSetStreamResLimit(stream, RT_DEV_RES_VECTOR_CORE, static_cast<uint32_t>(vectorcore_num)));
      update_stream_core_num = true;
    }
  }

  if (update_stream_core_num) {
    GE_CHK_RT_RET(rtsUseStreamResInCurrentThread(stream));
    GELOGI("Update stream core limits success.");
  }

  return SUCCESS;
}

Status HybridModelRtV2Executor::ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                                       std::vector<gert::Tensor> &outputs, const rtStream_t stream) {
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Start execute ExecuteWithStreamAsync with rtv2 executor of graph %s", name_.c_str());
  }
  gert::ModelExecuteArg model_args;
  model_args.stream = stream == nullptr ? stream_ : stream;
  const auto allocators = allocator_manager_.GetAllocator(run_ctx_.graph_name_, model_args.stream);
  GE_ASSERT_NOTNULL(allocators, "Failed get scalable allocators");
  auto *allocator =
      allocators->GetAllocator(gert::kOnDeviceHbm, static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput));
  GE_ASSERT_NOTNULL(allocator, "Failed get scalable allocator");
  GE_ASSERT_SUCCESS(InputTensorValidate(inputs, num_inputs_, run_ctx_.host_exec_flag_, logLevel_));

  model_args.external_allocator = allocators;
  ProfilerCollector *profiler_collector = nullptr;
  if (gert::GlobalProfilingWrapper::GetInstance()->GetEnableFlags() > 0UL) {
    profiler_collector = profiler_collector_.get();
  }
  model_args.external_stream_allocator = &(run_ctx_.dev_resource_allocator_.stream_allocator);
  model_args.external_event_allocator = &(run_ctx_.dev_resource_allocator_.event_allocator);
  model_args.external_notify_allocator = &(run_ctx_.dev_resource_allocator_.notify_allocator);
  const auto config =
      gert::RtV2ExecutorInterface::RunConfig(run_ctx_.iterations_per_loop_, profiler_collector);
  for (size_t i = 0U; i < num_inputs_; i++) {
    rt_inputs_[i] = const_cast<gert::Tensor*>(&inputs[i]);
  }

  GE_ASSERT_SUCCESS(CheckInputIsOnDevice());

  if (outputs.empty()) {
    outputs.resize(num_outputs_);
  }
  GE_ASSERT_EQ(outputs.size(), num_outputs_);
  for (size_t i = 0U; i < num_outputs_; i++) {
    rt_outputs_[i] = &outputs[i];
  }

  TryUpdateStreamCoreLimits(model_args.stream);

  const auto ret = executor_->Execute(model_args, rt_inputs_.data(), rt_inputs_.size(), rt_outputs_.data(),
                                      rt_outputs_.size(), config);

  GE_ASSERT_SUCCESS(ret, "Failed to execute rt v2 model for graph %s, model_id %u.", name_.c_str(), model_id_);

  for (size_t i = 0U; i < num_outputs_; i++) {
    auto &desc = output_descs_[i];
    if (desc->GetDataType() != ge::DT_STRING) {
      rt_outputs_[i]->SetSize(GetSizeInBytes(rt_outputs_[i]->GetShapeSize(), desc->GetDataType()));
    }
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("  Output %zu %s", i, DebugString(*rt_outputs_[i]).c_str());
    }
  }

  return ge::SUCCESS;
}

Status HybridModelRtV2Executor::ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs,
                                                       std::vector<GeTensor> &outputs, const rtStream_t stream) {
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Start execute ExecuteWithStreamAsync with rtv2 executor of graph %s", name_.c_str());
  }
  gert::ModelExecuteArg model_args;
  model_args.stream = stream == nullptr ? stream_ : stream;
  const auto allocators = allocator_manager_.GetAllocator(run_ctx_.graph_name_, model_args.stream);
  GE_ASSERT_NOTNULL(allocators, "Failed get scalable allocators");
  auto *allocator =
      allocators->GetAllocator(gert::kOnDeviceHbm, static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput));
  GE_ASSERT_NOTNULL(allocator, "Failed get scalable allocator");
  GE_ASSERT_EQ(inputs.size(), num_inputs_);
  for (size_t i = 0U; i < num_inputs_; ++i) {
    auto &rt_input = rt_inputs_[i];
    if (inputs[i].GetTensorDesc().IsOriginShapeInitialized()) {
      SmallVecDimsAsRtShape(inputs[i].GetTensorDesc().GetOriginShape().GetMutableDims(),
                            rt_input->MutableOriginShape());
    } else {
      SmallVecDimsAsRtShape(inputs[i].GetTensorDesc().GetShape().GetMutableDims(), rt_input->MutableOriginShape());
    }
    SmallVecDimsAsRtShape(inputs[i].GetTensorDesc().GetShape().GetMutableDims(), rt_input->MutableStorageShape());
    // 对于同一个输入而言，当前不允许用户两次RungraphWithStreamAsync时，传递不同的placement
    // 由于从TensorDesc上获取Placement的代价较高，因此只在第一次时做placement转换处理，并且不做校验,只打印INFO日志
    if (rt_input->GetPlacement() == gert::kTensorPlacementEnd) {
      ge::Placement placement = inputs[i].GetTensorDesc().GetPlacement();
      gert::TensorPlacement rt_placement =
          placement == ge::kPlacementHost ? gert::TensorPlacement::kOnHost : gert::TensorPlacement::kOnDeviceHbm;
      rt_input->SetPlacement(rt_placement);
    }
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("input %zu has placement %s, and executor expect placement %s", i,
            DebugString(inputs[i].GetTensorDesc().GetPlacement()).c_str(),
            DebugString(rt_input->GetPlacement()).c_str());
    }

    size_t size = inputs[i].GetData().size();
    auto address = reinterpret_cast<const void*>(inputs[i].GetData().data());

    if (run_ctx_.host_exec_flag_) {
      rt_input->MutableTensorData().SetPlacement(gert::kOnHost);
    } else if (gert::TensorPlacementUtils::IsOnDevice(rt_input->GetPlacement())) {
      GELOGD(
          "input[%zu] addres = %p, size = %zu, placement = %u, which is on device, no need do alloc memory and "
          "rtmemcpy", i, address, size, rt_input->GetPlacement());
    } else {
      GE_ASSERT(gert::TensorPlacementUtils::IsOnHostNotFollowing(rt_input->GetPlacement()),
                "Input %zu has unexpected placement %d", i, rt_input->GetPlacement());
      GE_ASSERT(rt_input->GetOriginShape().GetDimNum() <= 1U, "Input %zu %s on host must be scalar or list-scalar", i,
                DebugString(*rt_input).c_str());
    }
    GE_ASSERT_GRAPH_SUCCESS(rt_input->MutableTensorData().SetAddr(const_cast<void *>(address), nullptr));
    rt_input->MutableTensorData().SetSize(size);
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("Input %zu %s", i, DebugString(*rt_input).c_str());
    }
  }

  if (outputs.empty()) {
    outputs.resize(num_outputs_);
  }
  GE_ASSERT_EQ(outputs.size(), rt_outputs_.size());
  for (size_t i = 0UL; i < num_outputs_; ++i) {
    if (outputs[i].IsTensorDataValid()) {
      auto address = ValueToPtr(PtrToValue(outputs[i].GetData().data()));
      size_t size = outputs[i].GetData().size();
      if (rt_outputs_[i]->GetPlacement() == gert::TensorPlacement::kTensorPlacementEnd) {
        const auto placement =
            (outputs[i].GetTensorDesc().GetPlacement() == kPlacementDevice) ? gert::kOnDeviceHbm : gert::kOnHost;
        rt_outputs_[i]->SetPlacement(placement);
      }

      if (logLevel_ <= DLOG_DEBUG) {
        GELOGD("The user did specify output memory when index = %zu, address = %p, size = %zu, placement = %d",
              i, address, size, static_cast<int32_t>(rt_outputs_[i]->GetPlacement()));
      }
      GE_ASSERT_GRAPH_SUCCESS(rt_outputs_[i]->MutableTensorData().SetAddr(address, nullptr));
      rt_outputs_[i]->MutableTensorData().SetSize(size);
    } else {
      rt_outputs_[i]->SetPlacement(gert::kOnDeviceHbm);
    }
  }

  model_args.external_allocator = allocators;
  ProfilerCollector *profiler_collector = nullptr;
  if (gert::GlobalProfilingWrapper::GetInstance()->GetEnableFlags() > 0UL) {
    profiler_collector = profiler_collector_.get();
  }
  model_args.external_stream_allocator = &(run_ctx_.dev_resource_allocator_.stream_allocator);
  model_args.external_event_allocator = &(run_ctx_.dev_resource_allocator_.event_allocator);
  model_args.external_notify_allocator = &(run_ctx_.dev_resource_allocator_.notify_allocator);
  const auto config =
      gert::RtV2ExecutorInterface::RunConfig(run_ctx_.iterations_per_loop_, profiler_collector);

  TryUpdateStreamCoreLimits(model_args.stream);

  const auto ret = executor_->Execute(model_args, rt_inputs_.data(), rt_inputs_.size(), rt_outputs_.data(),
                                      rt_outputs_.size(), config);

  GE_ASSERT_SUCCESS(ret, "Failed to execute rt v2 model for graph %s, model_id %u.", name_.c_str(), model_id_);

  // Reference count decrement 1 when ExecuteWithStreamAsync finished
  ge::ScopeGuard guarder([this]() { StepDone(); });
  // move executor results to usr
  GE_ASSERT_SUCCESS(PostProcResult(outputs));
  return ge::SUCCESS;
}

Status HybridModelRtV2Executor::Execute(ExecuteArgs &args) {
  InputData input_data;
  // prepare input_data by args.inputs
  GE_CHK_STATUS_RET(PrepareInputData(input_data, args),
                    "[Invoke][PrepareInputData]Failed to copy input data to model, model_id = %u", model_id_);
  GELOGD("Done parser input data successfully.");
  return Execute(input_data, args);
}

void HybridModelRtV2Executor::ResetMemcpyBatchParams() {
  memcpy_batch_params_.dsts.clear();
  memcpy_batch_params_.dst_aligned_sizes.clear();
  memcpy_batch_params_.srcs.clear();
  memcpy_batch_params_.src_sizes.clear();
  memcpy_batch_params_.attrs.clear();
  memcpy_batch_params_.attr_idxs.clear();
  memcpy_batch_params_.device_id = 0;
}

Status HybridModelRtV2Executor::Execute(const InputData &input_data, ExecuteArgs &args) {
  gert::ModelExecuteArg model_args;
  model_args.stream = args.ctrl_args.stream != nullptr ? args.ctrl_args.stream : stream_;
  GELOGI("Start execute hybrid model v2 executor of graph %s, stream: %p", name_.c_str(), model_args.stream);
  const auto allocators = allocator_manager_.GetAllocator(run_ctx_.graph_name_, model_args.stream);
  GE_ASSERT_NOTNULL(allocators, "Failed get scalable allocators");
  auto *allocator =
      allocators->GetAllocator(gert::kOnDeviceHbm, static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput));
  GE_ASSERT_NOTNULL(allocator, "Failed get scalable allocator");
  auto mem_synchronizer = dynamic_cast<gert::memory::MemSynchronizer*>(allocator);
  std::vector<MemBlock *> input_mem_block;
  auto free_mem_block_callback = [&input_mem_block] () {
    for (auto &mem_block : input_mem_block) {
      mem_block->Free();
    }
  };
  GE_MAKE_GUARD(free_mem, free_mem_block_callback);
  GE_CHECK_LE(num_inputs_, input_data.shapes.size());

  int32_t cur_device_id = -1;
  if (run_ctx_.enable_input_batch_cpy_) {
    ResetMemcpyBatchParams();
    GE_CHK_RT_RET(rtGetDevice(&cur_device_id));
  }
  size_t idx = 0;
  for (size_t i = 0U; i < num_inputs_; ++i) {
    auto &input = rt_inputs_[i];
    DimsAsShape(input_data.shapes[i], input->MutableOriginShape());
    DimsAsShape(input_data.shapes[i], input->MutableStorageShape());

    if (run_ctx_.host_exec_flag_) {
      input->SetData(gert::TensorData(input_data.blobs[i].data, nullptr, input_data.blobs[i].length, gert::kOnHost));
    } else if (input_data.blobs[i].placement == static_cast<uint32_t>(Placement::kPlacementDevice)) {
      GELOGD("Construct RT2 input index[%u], placement: %u, no need to execute rtMemcpy.", i,
             input_data.blobs[i].placement);
      input->SetData(
          gert::TensorData(input_data.blobs[i].data, nullptr, input_data.blobs[i].length, gert::kOnDeviceHbm));
    } else {
      MemBlock *mem_block_to_keep = nullptr;
      memcpy_batch_params_.device_id = cur_device_id;
      if (run_ctx_.enable_input_batch_cpy_) {
        auto ge_tensor_length = input_data.blobs[i].length;
        size_t data_size = 0U;
        TensorTransUtils::AllocDeviceMemory(allocator, ge_tensor_length, *input, mem_block_to_keep, data_size);
        if (ge_tensor_length <= 0) {
          GELOGD("Skip input[%zu] with length %zu, no need to execute rtMemcpy.", i, input_data.blobs[i].length);
          input_mem_block.emplace_back(mem_block_to_keep);
          continue;
        }
        MemcpyParam memcpy_param {input->GetAddr(), data_size, input_data.blobs[i].data, input_data.blobs[i].length, idx++};
        TensorTransUtils::AddMemcpyBatchParam(memcpy_param, memcpy_batch_params_);
      } else {
        GE_ASSERT_SUCCESS(TensorTransUtils::HostTensorToDeviceGertTensor(allocator, input_data.blobs[i].data,
                                                                       input_data.blobs[i].length, *input, mem_block_to_keep));
        GE_ASSERT_NOTNULL(mem_block_to_keep);
      }
      input_mem_block.emplace_back(mem_block_to_keep);
    }
    GELOGI("Input %zu %s", i, DebugString(*input).c_str());
  }
  GE_ASSERT_SUCCESS(CheckInputIsOnDevice());

  if (!memcpy_batch_params_.dsts.empty()) {
    GE_ASSERT_SUCCESS(TensorTransUtils::TryBatchMemcpy(memcpy_batch_params_));
  }

  if (!args.outputs.empty()) {
    GE_ASSERT_EQ(args.outputs.size(), rt_outputs_.size());
    for (size_t i = 0UL; i < num_outputs_; ++i) {
      const auto placement = (args.outputs[i].GetMemType() == MemStorageType::HOST_DDR) ?
                             gert::kOnHost : gert::kOnDeviceHbm;
      rt_outputs_[i]->SetData(gert::TensorData(const_cast<gert::TensorAddress>(args.outputs[i].GetData()), nullptr,
                                               args.outputs[i].GetSize(), placement));
    }
  }

  ProfilerCollector *profiler_collector = nullptr;
  model_args.external_allocator = allocators;
  if (gert::GlobalProfilingWrapper::GetInstance()->GetEnableFlags() > 0UL) {
    profiler_collector = profiler_collector_.get();
    profiler_collector->host_cpu_flag_ = run_ctx_.host_exec_flag_;
  }
  const auto config = gert::RtV2ExecutorInterface::RunConfig(run_ctx_.iterations_per_loop_, profiler_collector);

  model_args.external_stream_allocator = &(run_ctx_.dev_resource_allocator_.stream_allocator);
  model_args.external_event_allocator = &(run_ctx_.dev_resource_allocator_.event_allocator);
  model_args.external_notify_allocator = &(run_ctx_.dev_resource_allocator_.notify_allocator);

  TryUpdateStreamCoreLimits(model_args.stream);

  const auto ret = executor_->Execute(model_args, rt_inputs_.data(), rt_inputs_.size(), rt_outputs_.data(),
                                      rt_outputs_.size(), config);
  free_mem_block_callback();
  if (ret == END_OF_SEQUENCE) {
    args.ctrl_args.is_eos = true;
    (void)DoRtStreamSyncWithTimeout(model_args.stream);
    if (mem_synchronizer != nullptr) {
      mem_synchronizer->Recycle();
    }
    return ge::SUCCESS;
  } else {
    GE_ASSERT_SUCCESS(ret, "Failed to execute rt v2 model for graph %s, model_id %u.", name_.c_str(), model_id_);
  }
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(model_args.stream));
  GELOGI("Execute sync rt v2 model for graph %s succeed", name_.c_str());
  if (mem_synchronizer != nullptr) {
    GELOGI("start to recycle memory. graph %s, stream: %p", name_.c_str(), model_args.stream);
    mem_synchronizer->Recycle();
  }
  args.outputs.clear();
  args.outputs.reserve(num_outputs_);
  args.output_desc.assign(output_descs_.begin(), output_descs_.end());

  for (size_t i = 0U; i < num_outputs_; ++i) {
    auto &output = rt_outputs_[i];
    GELOGI("  Output %zu %s", i, DebugString(*output).c_str());
    auto placement = gert::TensorPlacementUtils::IsOnDevice(output->GetPlacement()) ? MemStorageType::HBM
                                                                                    : MemStorageType::HOST_DDR;
    auto &desc = output_descs_[i];
    // output with string datatype will get -1*shapeSize
    size_t buffer_size = 0UL;
    if (desc->GetDataType() == ge::DT_STRING) {
      buffer_size = output->GetSize();
    } else {
      buffer_size = GetSizeInBytes(output->GetShapeSize(), desc->GetDataType());
    }
    GELOGI("Output index %zu data type is %d. Get or calculate buffer size %zu",
           i, static_cast<int32_t>(desc->GetDataType()), buffer_size);

    args.outputs.emplace_back(output->GetAddr(), buffer_size, placement);
    RtShapeAsGeShape(output->GetStorageShape(), desc->MutableShape());
  }
  return ge::SUCCESS;
}

/*
 * 输入内存说明：
 * inputs 为输入数据，可能位于host或device, 但是模型执行一般是要求输入位于device，因此有了rt_inputs_，
 * rt_inputs_只是引用内存，不负责释放。
 *  如果args.inputs[i]为host内存，则新申请device内存，并把数据拷贝过去，内存释放由input_mem_block负责。rt_inputs_[i]引用新申请内存
 *  如果args.inputs[i]为device内存，rt_inputs_[i]引用args.inputs[i]，内存由args.inputs[i]负责释放。
 *
 * 输出内存说明：
 * 输出由执行器申请，保存在outputs，一般是device内存，后续需要调用CopyOutputs拷贝到host上，返回给用户
 */
Status HybridModelRtV2Executor::Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) {
  gert::ModelExecuteArg model_args;
  model_args.stream = ctrl_args.stream != nullptr ? ctrl_args.stream : stream_;
  GELOGI("Start execute hybrid model v2 executor of graph %s, stream: %p", name_.c_str(), model_args.stream);
  const auto allocators = allocator_manager_.GetAllocator(run_ctx_.graph_name_, model_args.stream);
  GE_ASSERT_NOTNULL(allocators, "Failed get scalable allocators");
  auto *allocator =
      allocators->GetAllocator(gert::kOnDeviceHbm, static_cast<size_t>(gert::AllocatorUsage::kAllocNodeOutput));
  GE_ASSERT_NOTNULL(allocator, "Failed get scalable allocator");
  auto mem_synchronizer = dynamic_cast<gert::memory::MemSynchronizer*>(allocator);
  std::vector<MemBlock *> input_mem_block;
  auto free_mem_block_callback = [&input_mem_block] () {
    for (auto &mem_block : input_mem_block) {
      mem_block->Free();
    }
  };
  GE_MAKE_GUARD(free_mem, free_mem_block_callback);
  GE_CHECK_LE(num_inputs_, inputs.size());

  int32_t cur_device_id = -1;
  if (run_ctx_.enable_input_batch_cpy_) {
    ResetMemcpyBatchParams();
    GE_CHK_RT_RET(rtGetDevice(&cur_device_id));
  }
  size_t idx = 0;
  for (size_t i = 0U; i < num_inputs_; ++i) {
    // 内存所有权仍然在arg_input，ref_input只是引用内存地址，不负责释放内存
    const auto &arg_input = inputs.at(i);
    auto &ref_input = rt_inputs_.at(i);

    // update shape
    ref_input->MutableOriginShape() = arg_input.GetOriginShape();
    ref_input->MutableStorageShape() = arg_input.GetStorageShape();
    // ge.exec.placement, not public, tensorflow may set
    if (run_ctx_.host_exec_flag_) {
      GE_ASSERT_TRUE(arg_input.GetPlacement() == gert::TensorPlacement::kOnHost,
                     "host exec, but ref_input[%zu] is on device", i);
      ref_input->SetData(gert::TensorData(const_cast<void *>(arg_input.GetAddr()), nullptr,
        arg_input.GetSize(), gert::kOnHost));
    } else if (arg_input.GetPlacement() == gert::TensorPlacement::kOnDeviceHbm) {
      GELOGD("Construct RT2 ref_input index[%u], placement: %u, no need to execute rtMemcpy.", i,
             gert::TensorPlacement::kOnDeviceHbm);
      ref_input->SetData(gert::TensorData(const_cast<void *>(arg_input.GetAddr()), nullptr,
        arg_input.GetSize(), gert::kOnDeviceHbm));
    } else {
      MemBlock *mem_block_to_keep = nullptr;
      memcpy_batch_params_.device_id = cur_device_id;
      if (run_ctx_.enable_input_batch_cpy_) {
        const auto src_tensor_length = arg_input.GetSize();
        size_t data_size = 0U;
        GE_ASSERT_SUCCESS(TensorTransUtils::AllocDeviceMemory(allocator, src_tensor_length, *ref_input, mem_block_to_keep, data_size));
        if (src_tensor_length == 0) {
          GELOGD("Skip ref_input[%zu] with length %zu, no need to execute rtMemcpy.", i, arg_input.GetSize());
          input_mem_block.emplace_back(mem_block_to_keep);
          continue;
        }
        MemcpyParam memcpy_param {ref_input->GetAddr(), data_size, const_cast<void *>(arg_input.GetAddr()),
                              src_tensor_length, idx++};
        TensorTransUtils::AddMemcpyBatchParam(memcpy_param, memcpy_batch_params_);
      } else {
        GE_ASSERT_SUCCESS(TensorTransUtils::HostTensorToDeviceGertTensor(allocator, arg_input.GetAddr(),
            arg_input.GetSize(), *ref_input, mem_block_to_keep));
        GE_ASSERT_NOTNULL(mem_block_to_keep);
      }

      input_mem_block.emplace_back(mem_block_to_keep);
    }
    GELOGI("Input %zu %s", i, DebugString(*ref_input).c_str());
  }
  GE_ASSERT_SUCCESS(CheckInputIsOnDevice());

  if (!memcpy_batch_params_.dsts.empty()) {
    GE_ASSERT_SUCCESS(TensorTransUtils::TryBatchMemcpy(memcpy_batch_params_));
  }

  outputs.clear();
  outputs.resize(outputs_holder_.size());
  for (size_t i = 0UL; i < outputs_holder_.size(); ++i) {
    outputs[i].MutableFormat() = outputs_holder_[i].GetFormat();
    outputs[i].MutableStorageShape() = outputs_holder_[i].GetStorageShape();
    outputs[i].MutableOriginShape() = outputs_holder_[i].GetOriginShape();
    outputs[i].SetDataType(outputs_holder_[i].GetDataType());
    outputs[i].SetPlacement(gert::TensorPlacement::kTensorPlacementEnd);
    outputs[i].SetData(gert::TensorData(nullptr));
    rt_outputs_[i] = &outputs[i];
  }

  ProfilerCollector *profiler_collector = nullptr;
  model_args.external_allocator = allocators;
  if (gert::GlobalProfilingWrapper::GetInstance()->GetEnableFlags() > 0UL) {
    profiler_collector = profiler_collector_.get();
    profiler_collector->host_cpu_flag_ = run_ctx_.host_exec_flag_;
  }
  const auto config = gert::RtV2ExecutorInterface::RunConfig(run_ctx_.iterations_per_loop_, profiler_collector);

  model_args.external_stream_allocator = &(run_ctx_.dev_resource_allocator_.stream_allocator);
  model_args.external_event_allocator = &(run_ctx_.dev_resource_allocator_.event_allocator);
  model_args.external_notify_allocator = &(run_ctx_.dev_resource_allocator_.notify_allocator);

  TryUpdateStreamCoreLimits(model_args.stream);

  const auto ret = executor_->Execute(model_args, rt_inputs_.data(), rt_inputs_.size(), rt_outputs_.data(),
                                      rt_outputs_.size(), config);
  free_mem_block_callback();
  if (ret == END_OF_SEQUENCE) {
    ctrl_args.is_eos = true;
    (void)DoRtStreamSyncWithTimeout(model_args.stream);
    if (mem_synchronizer != nullptr) {
      mem_synchronizer->Recycle();
    }
    return ge::SUCCESS;
  } else {
    GE_ASSERT_SUCCESS(ret, "Failed to execute rt v2 model for graph %s, model_id %u.", name_.c_str(), model_id_);
  }
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(model_args.stream));
  GELOGI("Execute sync rt v2 model for graph %s succeed", name_.c_str());
  if (mem_synchronizer != nullptr) {
    GELOGI("start to recycle memory. graph %s, stream: %p", name_.c_str(), model_args.stream);
    mem_synchronizer->Recycle();
  }
  return ge::SUCCESS;
}

Status HybridModelRtV2Executor::StepDone() const {
  const static gert::TensorData kClearedTensorData;
  for (auto &output : rt_outputs_) {
    // MutableTensorData().Free() is not enough as it maybe not owned the output memory
    GELOGI("Clear model output holder with block %p", output->GetAddr());
    output->MutableTensorData().ShareFrom(kClearedTensorData);
  }
  return ge::SUCCESS;
}

void HybridModelRtV2Executor::StepDoneV2() {
  for (auto &ref : rt_outputs_) {
    ref = nullptr;
  }
}

void HybridModelRtV2Executor::Stop() {}

bool HybridModelRtV2Executor::NeedBuildDeviceTensorAsOutput() const {
  return ((run_ctx_.execute_mode_ == kLazyRecompile) && (run_ctx_.is_copy_output_addr_ == kIsCopyOuputAddr));
}

Status HybridModelRtV2Executor::BuildDeviceTensor(TensorValue &output_tensor, GeTensorDesc &ge_tensor_desc,
                                                  const int64_t output_size, vector<ge::Tensor> &outputs) const {
  ge_tensor_desc.SetPlacement(kPlacementDevice);
  GeTensor ge_tensor(ge_tensor_desc);
  auto tensor = TensorAdapter::AsTensor(ge_tensor);
  auto output_holder = MakeShared<gert::TensorData>();
  GE_ASSERT_NOTNULL(output_holder);
  for (auto &output : outputs_holder_) {
    if (output.GetAddr() == output_tensor.GetData()) {
      GELOGD("Add ref count for tensor with details [%s]", DebugString(output, true).c_str());
      output_holder->ShareFrom(output.GetTensorData());
    }
  }
  const auto deleter = [output_holder](uint8_t *const device_data) {
    (void)device_data;
    output_holder->Free();
  };
  GELOGD("Build device tensor successfully with details [%s]", output_tensor.DebugString().c_str());
  GE_CHK_STATUS_RET(
      tensor.SetData(PtrToPtr<void, uint8_t>(output_tensor.Release()), static_cast<size_t>(output_size), deleter));
  outputs.emplace_back(std::move(tensor));
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
