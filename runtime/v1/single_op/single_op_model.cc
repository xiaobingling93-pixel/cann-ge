/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/single_op_model.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "framework/generator/ge_generator.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "single_op/task/aicpu_task_builder.h"
#include "single_op/task/aicpu_c_c_task_builder.h"
#include "single_op/task/dsa_task_builder.h"
#include "single_op/task/rts_kernel_task_builder.h"
#include "single_op/task/tbe_task_builder.h"
#include "hybrid/node_executor/node_executor.h"
#include "common/ge_inner_attrs.h"
#include "common/profiling/profiling_manager.h"
#include "common/utils/executor_utils.h"
#include "runtime/subscriber/global_profiler.h"
#include "graph/args_format_desc.h"

namespace ge {
namespace {
constexpr size_t kDataOutputNum = 1U;
constexpr uint32_t kInputIndexOfData = 0U;
constexpr size_t kNumTaskWithMemCpyTask = 2U;
constexpr int64_t kMemTypeHost = 1;
constexpr int64_t kMemTypeHostCompileIndependent = 2;
constexpr char_t const *kMallocWeightPurpose = "malloc weights memory on model execute.";
std::atomic<std::uint64_t> aicpu_kernel_id(0U);
using StaticTaskBuildFunc =
    std::function<Status(const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &stream_resource)>;
using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

bool IsGeLocalTaskWithoutHybrid(const std::string &type) {
  return OpTypeUtils::IsDataNode(type) || (type == ge::CONSTANT) || (type == ge::CONSTANTOP) ||
         OpTypeUtils::IsVarLikeNode(type) || (type == ge::NETOUTPUT);
}

Status CheckHostMem(const std::vector<std::string> &dependencies, const NodePtr &node, bool &is_host_mem) {
  const auto op_desc = node->GetOpDesc();
  for (const auto &input_name : dependencies) {
    const int32_t input_index = op_desc->GetInputIndexByName(input_name);
    if (input_index < 0) {
      GELOGE(INTERNAL_ERROR, "[Get][InputIndex]failed, node:[%s] inputname: %s.",
             node->GetName().c_str(), input_name.c_str());
      REPORT_INNER_ERR_MSG("E19999", "GetInputIndexByName failed, node:[%s] inputname: %s.",
                        node->GetName().c_str(), input_name.c_str());
      return INTERNAL_ERROR;
    }

    const auto &src_node = NodeUtils::GetInDataNodeByIndex(*node, input_index);
    GE_CHECK_NOTNULL(src_node);
    const auto src_op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    if (OpTypeUtils::IsDataNode(src_op_desc->GetType())) {
      const auto tensor = src_op_desc->MutableInputDesc(kInputIndexOfData);
      GE_CHECK_NOTNULL(tensor);
      int64_t mem_type = 0;
      if (AttrUtils::GetInt(tensor, ATTR_NAME_PLACEMENT, mem_type) &&
          ((mem_type == kMemTypeHost) || (mem_type == kMemTypeHostCompileIndependent))) {
        GELOGD("Get hostmem from node %s, inputname: %s, mem_type = %" PRId64 ".",
               src_node->GetName().c_str(), input_name.c_str(), mem_type);
        continue;
      }
    }
    is_host_mem = false;
    return SUCCESS;
  }
  is_host_mem = true;
  return SUCCESS;
}

Status CheckInferDepend(const ComputeGraphPtr &comp_graph, bool &is_infer_depend, bool &is_host_mem) {
  for (const auto &node : comp_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &depends = op_desc->GetOpInferDepends();
    bool support_dynamic_shape = false;
    (void)AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, support_dynamic_shape);
    if ((!depends.empty()) && support_dynamic_shape) {
      is_infer_depend = true;
      const auto ret = CheckHostMem(depends, node, is_host_mem);
      return ret;
    }
  }
  return SUCCESS;
}

Status CheckGeLocalNeedHybrid(const ComputeGraphPtr &comp_graph, bool &is_ge_local_need_hybrid) {
  for (const auto &node : comp_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &lib_name = op_desc->GetOpKernelLibName();
    const auto &op_type = op_desc->GetType();
    GELOGD("op name is %s, op kernel name is %s, op type is %s.", op_desc->GetName().c_str(),
        lib_name.c_str(), op_type.c_str());
    if ((lib_name == kEngineNameGeLocal) && (!IsGeLocalTaskWithoutHybrid(op_type))) {
      GELOGD("op name is %s, use GE local task with hybrid execute", op_desc->GetName().c_str());
      is_ge_local_need_hybrid = true;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

Status GetAicoreTask(const std::vector<domi::TaskDef> &task_defs, std::vector<domi::TaskDef> &aicore_task_defs) {
  for (size_t i = 0UL; i < task_defs.size(); ++i) {
    const domi::TaskDef &task_def = task_defs[i];
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    if (task_type == ModelTaskType::MODEL_TASK_FFTS_PLUS) {
      aicore_task_defs.emplace_back(task_def);
      continue;
    }

    if ((task_type == ModelTaskType::MODEL_TASK_KERNEL) || (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL)) {
      const auto &context = (task_type == ModelTaskType::MODEL_TASK_KERNEL) ? task_def.kernel().context() :
                                                                  task_def.kernel_with_handle().context();
      const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
      if (kernel_type == ccKernelType::TE) {
        aicore_task_defs.emplace_back(task_def);
      }
    }
  }
  if (aicore_task_defs.empty()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node size must larger then 0, but get %zu.",
           aicore_task_defs.size());
    REPORT_INNER_ERR_MSG("E19999", "[Check][Size]task_defs size must larger then 0, but get %zu.",
                       aicore_task_defs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}
}  // namespace

SingleOpModel::SingleOpModel(const std::string &model_name, const void *const model_data, const uint32_t model_size)
    : model_name_(model_name), ori_model_data_(model_data), ori_model_size_(model_size) {}

Status SingleOpModel::Init() {
  GE_CHK_STATUS_RET_NOLOG(InitModel());
  return LoadRootGraph();
}

Status SingleOpModel::InitModel() {
  ge::ModelData model;
  model.model_len = ori_model_size_;
  model.model_data = ValueToPtr(PtrToValue(ori_model_data_));

  const auto ret = model_helper_.LoadRootModel(model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed.");
    REPORT_INNER_ERR_MSG("E19999", "InitModel fail for ModelHelper LoadModel failed.");
    return ret;
  }
  const std::shared_ptr<GeRootModel> &root_model = model_helper_.GetGeRootModel();
  GE_CHECK_NOTNULL(root_model);
  root_graph_ = root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph_);
  root_ge_model_ = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(root_ge_model_);

  return SUCCESS;
}

Status SingleOpModel::ParseOpModelParams() {
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  GE_ASSERT_SUCCESS(
      ModelUtils::InitRuntimeParams(root_ge_model_, model_params_.runtime_param, static_cast<uint32_t>(device_id)));
  model_params_.runtime_param.session_id = UINT64_MAX;
  model_params_.runtime_param.is_single_op = true;
  (void)AttrUtils::GetInt(root_ge_model_, ATTR_MODEL_CORE_TYPE, model_params_.core_type);
  GE_CHK_STATUS_RET_NOLOG(model_helper_.HandleDeviceInfo(model_params_.platform_infos));

  // load so from model_helper_
  const auto root_model = model_helper_.GetGeRootModel();
  if (root_model == nullptr) {
    GELOGW("Invalid root model!");
  } else {
    GE_ASSERT_SUCCESS(
        ge::ModelUtils::GetSpaceRegistries(model_helper_.GetGeRootModel(), model_params_.space_registries_));
  }

  GELOGI("ParseOpModelParams(): core_type = %" PRId64 ", runtime_param = %s.", model_params_.core_type,
         model_params_.runtime_param.ToString().c_str());
  return SUCCESS;
}

Status SingleOpModel::InitOverflowAddr(StreamResource &resource) const {
  return resource.InitOverflowMemory();
}

Status SingleOpModel::InitModelMem(StreamResource &resource) {
  GE_CHK_STATUS_RET(InitOverflowAddr(resource), "[Init][OverflowAddr] failed.");
  GE_CHK_STATUS_RET(ParseOpModelParams(), "Parse op model params failed.");
  GE_CHK_STATUS_RET(MallocWeight(resource));
  return resource.MallocExMem(0U, model_params_.runtime_param);
}

Status SingleOpModel::MallocWeight(StreamResource &resource) {
  if ((model_params_.runtime_param.weight_size > 0U) && has_weight_) {
    uint8_t * const weight_base = resource.MallocWeight(kMallocWeightPurpose, model_params_.runtime_param.weight_size);
    if (weight_base == nullptr) {
      // no need to free memory, for that was handled by StreamResources
      return ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED;
    }
    GELOGI("To copy weight to device. weight size = %zu.", root_ge_model_->GetWeightSize());
    GE_CHK_RT_RET(rtMemcpy(weight_base, model_params_.runtime_param.weight_size, root_ge_model_->GetWeightData(),
                           root_ge_model_->GetWeightSize(), RT_MEMCPY_HOST_TO_DEVICE));
    model_params_.runtime_param.weight_base = reinterpret_cast<uintptr_t>(weight_base);
  }
  return SUCCESS;
}

Status SingleOpModel::ParseInputNode(const OpDescPtr &op_desc) {
  const std::vector<int64_t> offsets = op_desc->GetOutputOffset();
  if (offsets.size() != kDataOutputNum) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Parse][InputNode]Data op should have only one output, but got %zu, op_name:%s, op_type:%s.",
           op_desc->GetOutputOffset().size(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "ParseInputNode fail for Data op should have only one output, but got %zu,"
                       "op_name:%s, op_type:%s.", op_desc->GetOutputOffset().size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  const auto output_desc = op_desc->GetOutputDescPtr(0U);
  GE_CHECK_NOTNULL(output_desc);
  int64_t tensor_size = 0;
  (void)TensorUtils::GetSize(*output_desc, tensor_size);
  input_offset_list_.emplace_back(offsets[0U]);
  input_sizes_.emplace_back(tensor_size);
  GELOGI("[%s] parse input node: %s, size = %" PRId64 ", offset = %" PRId64 ".",
    model_name_.c_str(), op_desc->GetName().c_str(), tensor_size, offsets[0U]);
  return SUCCESS;
}

void SingleOpModel::ParseOutputNode(const OpDescPtr &op_desc) {
  const std::vector<int64_t> offsets = op_desc->GetInputOffset();
  for (uint32_t k = 0U; k < static_cast<uint32_t>(offsets.size()); ++k) {
    const auto input_desc = op_desc->GetInputDescPtr(k);
    if (input_desc == nullptr) {
      continue;
    }
    int64_t tensor_size = 0;
    (void)TensorUtils::GetSize(*input_desc, tensor_size);
    output_offset_list_.emplace_back(offsets[static_cast<size_t>(k)]);
    output_sizes_.emplace_back(tensor_size);
    GELOGI("[%s] parse output node: %s, size = %" PRId64 ", offset = %u.", model_name_.c_str(),
      op_desc->GetName().c_str(), tensor_size, static_cast<uint32_t>(offsets[static_cast<size_t>(k)]));
  }
}

Status SingleOpModel::LoadRootGraph() {
  model_id_ = root_ge_model_->GetModelId();
  const auto nodes = root_graph_->GetDirectNode();
  GELOGI("[%s] node size = %zu.", model_name_.c_str(), nodes.size());

  for (const auto &node : nodes) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_list_[static_cast<uint32_t>(op_desc->GetId())] = node;
    const auto op_type = op_desc->GetType();
    GELOGI("[%s] node = %s, type = %s.", model_name_.c_str(), node->GetName().c_str(), op_type.c_str());

    if (OpTypeUtils::IsDataNode(op_type)) {
      size_t index = data_ops_.size();
      AttrUtils::GetInt(op_desc, "index", index);
      GELOGD("Get Index = %zu, data_ops_ size = %zu.", index, data_ops_.size());
      if (index >= data_ops_.size()) {
        data_ops_.resize(index + 1U);
      }
      data_ops_[index] = op_desc;
      const auto tensor = op_desc->MutableInputDesc(0U);
      int64_t mem_type = 0;
      if (AttrUtils::GetInt(tensor, ATTR_NAME_PLACEMENT, mem_type) &&
          ((mem_type == kMemTypeHost) || (mem_type == kMemTypeHostCompileIndependent))) {
        int32_t index_new = 0;
        (void)AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index_new);
        GELOGD("Node %s, index %d, has host mem, mem_type = %" PRId64 ".", node->GetName().c_str(), index_new, mem_type);
        op_with_hostmem_[index_new] = node;
      }
      continue;
    }

    if ((op_type == CONSTANT) || (op_type == CONSTANTOP)) {
      has_weight_ = true;
      continue;
    }

    if (op_type == NETOUTPUT) {
      netoutput_op_ = op_desc;
      continue;
    }

    root_ge_model_->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(op_desc);
    root_ge_model_->GetCustAICPUKernelStore().LoadCustAICPUKernelBinToOpDesc(op_desc);
    GE_CHK_STATUS_RET(ExecutorUtils::LoadAtomicWorkspace(op_desc), "[LoadAtomicWorkSpace]failed for [%s(%s)].",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }

  return SUCCESS;
}

Status SingleOpModel::ParseInputsAndOutputs() {
  for (auto &op_desc : data_ops_) {
    GE_CHK_STATUS_RET_NOLOG(ParseInputNode(op_desc));
  }
  if (netoutput_op_ != nullptr) {
    ParseOutputNode(netoutput_op_);
  }
  return SUCCESS;
}

Status SingleOpModel::SetInputsAndOutputs(SingleOpImpl &single_op) {
  size_t arg_index = 0U;
  for (size_t i = 0UL; i < input_offset_list_.size(); ++i) {
    uint8_t *addr = PtrToPtr<void, uint8_t>(
        ValueToPtr(model_params_.runtime_param.mem_base + static_cast<uint64_t>(input_offset_list_[i])));
    (void)model_params_.addr_mapping_.emplace(PtrToValue(addr), arg_index++);
    single_op.input_sizes_.emplace_back(input_sizes_[i]);
    single_op.input_addr_list_.emplace_back(addr);
  }

  for (size_t i = 0UL; i < output_offset_list_.size(); ++i) {
    uint8_t *addr = PtrToPtr<void, uint8_t>(
        ValueToPtr(model_params_.runtime_param.mem_base + static_cast<uint64_t>(output_offset_list_[i])));
    (void)model_params_.addr_mapping_.emplace(PtrToValue(addr), arg_index++);
    single_op.output_sizes_.emplace_back(output_sizes_[i]);
    single_op.output_addr_list_.emplace_back(addr);
  }

  single_op.args_.resize(arg_index);
  return SUCCESS;
}

Status SingleOpModel::BuildMixL2KernelTask(const domi::TaskDef &task_def, OpTask *&op_task,
                                           StreamResource &stream_resource) {
  const auto &ffts_plus_task_def = task_def.ffts_plus_task();
  const auto &iter = op_list_.find(ffts_plus_task_def.op_index());
  if (iter == op_list_.cend()) {
    GELOGE(FAILED, "Model [%s] does not have node with index:[%u].",
           model_name_.c_str(), ffts_plus_task_def.op_index());
    return FAILED;
  }

  const auto &node = iter->second;
  GE_CHECK_NOTNULL(node);
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (!op_desc->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    GELOGE(FAILED, "Node [%s] does not have attr [_alias_engine_name].", op_desc->GetName().c_str());
    return FAILED;
  }
  std::unique_ptr<MixL2OpTask> task = MakeUnique<MixL2OpTask>(node);
  GE_CHECK_NOTNULL(task);
  SetHostMemInputFlagToTask(node->GetOpDesc(), *task);
  task->stream_resource_ = &stream_resource;

  auto builder = MixL2TaskBuilder(model_name_, node, task_def);
  GE_CHK_STATUS_RET(builder.BuildMixL2Task(*task, model_params_), "Failed to build task.");
  op_task = task.release();

  return SUCCESS;
}

Status SingleOpModel::BuildTEKernelAndTask(const domi::TaskDef &task_def, OpTask *&op_task,
                                           StreamResource &stream_resource) {
  uint32_t op_index = 0U;
  if (!ExecutorUtils::GetOpIndex(task_def, op_index)) {
    GELOGE(FAILED, "Get op_index failed.");
    return FAILED;
  }
  const auto op_iter = op_list_.find(op_index);
  GE_CHK_BOOL_RET_STATUS(op_iter != op_list_.end(), FAILED, "Failed to get node by op_index: %u.", op_index);

  auto &node = op_iter->second;
  GE_CHECK_NOTNULL(node);
  const auto node_iter = node_tasks_.find(node);
  GE_CHK_BOOL_RET_STATUS(node_iter != node_tasks_.end(), FAILED,
                         "Failed to get task by node %s.", node->GetName().c_str());
  auto &tasks = node_iter->second;

  if (tasks.size() == 1U) {
    return BuildKernelTask(task_def, PtrToPtr<OpTask *, TbeOpTask *>(&op_task), stream_resource);
  }

  if (tasks.size() == kNumTaskWithAtomicAddrCleanTask) {
    if (built_nodes_.count(node) > 0U) {
      GELOGD("Node: %s has already built task.", node->GetName().c_str());
      return SUCCESS;
    }
    GELOGI("Node: %s has 2 tasks, include static atomic kernel and dynamic tbe kernel.", node->GetName().c_str());
    const auto &tbe_task = tasks.back();
    const auto tbe_op_task = PtrToPtr<OpTask *, TbeOpTask *>(&op_task);
    GE_CHK_STATUS_RET_NOLOG(BuildKernelTask(tbe_task, tbe_op_task, stream_resource));
    const auto &atomic_task_def = tasks.front();
    AtomicAddrCleanOpTask *atomic_task = nullptr;
    GE_CHK_STATUS_RET_NOLOG(BuildAtomicTask(atomic_task_def, &atomic_task, stream_resource));
    GE_CHK_STATUS_RET_NOLOG(atomic_task->InitAtomicAddrCleanIndices());
    (*tbe_op_task)->SetAtomicAddrCleanTask(atomic_task);
    (void) built_nodes_.emplace(node);
  }

  return SUCCESS;
}

Status SingleOpModel::BuildKernelAndExTask(const domi::TaskDef &task_def, OpTask *&op_task,
                                           StreamResource &stream_resource) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  if (task_type == ModelTaskType::MODEL_TASK_KERNEL_EX) {
    const uint64_t singleop_kernel_id = aicpu_kernel_id++;
    GELOGI("Build singleOp TfTask, kernel_id = %" PRIu64, singleop_kernel_id);
    return BuildKernelExTask(task_def.kernel_ex(), PtrToPtr<OpTask *, AiCpuTask *>(&op_task), singleop_kernel_id);
  }

  // ModelTaskType::MODEL_TASK_KERNEL or ModelTaskType::MODEL_TASK_ALL_KERNEL
  const auto &context =
      (task_type == ModelTaskType::MODEL_TASK_KERNEL) ? task_def.kernel().context() : task_def.kernel_with_handle().context();
  const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type == ccKernelType::TE) {
    GELOGD("Building TBE task");
    GE_CHK_STATUS_RET_NOLOG(BuildTEKernelAndTask(task_def, op_task, stream_resource));
  } else if ((kernel_type == ccKernelType::AI_CPU) || (kernel_type == ccKernelType::CUST_AI_CPU)) {
    const uint64_t singleop_kernel_id = aicpu_kernel_id++;
    GELOGI("Build singleOp CCTask, kernel_id = %" PRIu64, singleop_kernel_id);
    GE_CHK_STATUS_RET_NOLOG(
        BuildCpuKernelTask(task_def.kernel(), PtrToPtr<OpTask *, AiCpuCCTask *>(&op_task), singleop_kernel_id));
  } else {
    GELOGE(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID,
           "[Check][KernelType]Only TBE, AI_CPU, CUST_AI_CPU kernel are supported, but got %u", context.kernel_type());
    REPORT_INNER_ERR_MSG("E19999",
                       "BuildTaskList fail for %u not supported, Only TBE, AI_CPU, CUST_AI_CPU kernel are supported.",
                       context.kernel_type());
    return ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID;
  }
  return SUCCESS;
}

Status SingleOpModel::BuildTaskList(StreamResource &stream_resource, SingleOpImpl &single_op) {
  const GetOpDescFunc get_op_desc_func = [this](const uint32_t op_index, OpDescPtr &op_desc) -> Status {
    const auto &node = op_list_[op_index];
    GE_CHECK_NOTNULL(node);
    op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    return SUCCESS;
  };

  std::map<ModelTaskType, StaticTaskBuildFunc> convert_map = {
      {ModelTaskType::MODEL_TASK_KERNEL, std::bind(&SingleOpModel::BuildKernelAndExTask, this, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_ALL_KERNEL, std::bind(&SingleOpModel::BuildKernelAndExTask, this, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_KERNEL_EX, std::bind(&SingleOpModel::BuildKernelAndExTask, this, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_FFTS_PLUS, std::bind(&SingleOpModel::BuildMixL2KernelTask, this, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_DSA, std::bind(&DsaTaskBuilder::BuildDsaTask, get_op_desc_func, model_params_, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_MEMCPY_ASYNC, std::bind(&RtsKernelTaskBuilder::BuildMemcpyAsyncTask,
                                             get_op_desc_func, model_params_, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_MEMCPY_ADDR_ASYNC, std::bind(&RtsKernelTaskBuilder::BuildMemcpyAsyncTask,
                                                  get_op_desc_func, model_params_, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_NPU_CLEAR_FLOAT_STATUS, std::bind(&RtsKernelTaskBuilder::BuildNpuClearFloatStatusTask,
                                                       get_op_desc_func, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_NPU_GET_FLOAT_STATUS, std::bind(&RtsKernelTaskBuilder::BuildNpuGetFloatStatusTask,
                                                     get_op_desc_func, model_params_, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_NPU_CLEAR_DEBUG_FLOAT_STATUS, std::bind(&RtsKernelTaskBuilder::BuildNpuClearFloatDebugStatusTask,
                                                             get_op_desc_func, _1, _2, _3)},
      {ModelTaskType::MODEL_TASK_NPU_GET_DEBUG_FLOAT_STATUS, std::bind(&RtsKernelTaskBuilder::BuildNpuGetFloatDebugStatusTask,
                                                           get_op_desc_func, model_params_, _1, _2, _3)},
  };

  single_op.arg_table_.resize(single_op.input_sizes_.size() + single_op.output_sizes_.size());
  const auto &tasks = root_ge_model_->GetModelTaskDefPtr()->task();
  for (int32_t i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    GELOGI("[%s] Task[%d], type = %u, DebugString = %s", model_name_.c_str(), i, task_def.type(),
           task_def.DebugString().c_str());
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    const auto iter = convert_map.find(task_type);
    if (iter != convert_map.cend()) {
      OpTask *op_task = nullptr;
      GE_CHK_STATUS_RET(iter->second(task_def, op_task, stream_resource));
      if (op_task == nullptr) {
        GELOGD("Current task has built.");
        continue;
      }
      ParseArgTable(op_task, single_op);
      op_task->SetModelArgs(model_name_, model_id_);
      single_op.tasks_.emplace_back(op_task);
    } else {
      GELOGW("Unsupported task type: %d", static_cast<int32_t>(task_type));
    }
  }
  return SUCCESS;
}

void SingleOpModel::ParseArgTable(OpTask *const task, SingleOpImpl &op) {
  if (task == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Parse][ArgTable] fail for input OpTask is nullptr.");
    REPORT_INNER_ERR_MSG("E19999", "ParseArgTable fail for input OpTask is nullptr.");
    return;
  }
  // for exception dump
  task->SaveForL0ExceptionDump();
  // args: addr1, addr2, addr3 ...
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0U;
  task->GetIoAddr(arg_base, arg_num);
  const std::vector<bool> &v_is_input_const = task->GetOpdesc()->GetIsInputConst();
  constexpr size_t ptr_size = sizeof(uintptr_t);
  for (size_t i = 0U; i < arg_num; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      continue;
    }
    uintptr_t *ptr_to_addr = PtrToPtr<void, uintptr_t>(
        ValueToPtr(PtrToValue(arg_base) + static_cast<uint64_t>(ptr_size * i)));
    const uintptr_t addr = *ptr_to_addr;
    const auto &iter = model_params_.addr_mapping_.find(addr);
    if (iter != model_params_.addr_mapping_.cend()) {
      const int32_t arg_index = iter->second;
      GELOGI("%s args[%zu] mapped to user designated args[%d]", task->GetOpdesc()->GetName().c_str(), i, arg_index);
      op.arg_table_[static_cast<size_t>(iter->second)].emplace_back(ptr_to_addr);
    }
  }
}

Status SingleOpModel::BuildKernelTask(const domi::TaskDef &task_def, TbeOpTask **const task,
                                      StreamResource &stream_resource) {
  GE_CHECK_NOTNULL(task);
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  if (task_type == ModelTaskType::MODEL_TASK_FFTS_PLUS) {
    return BuildMixL2KernelTask(task_def, *PtrToPtr<TbeOpTask *, OpTask *>(task), stream_resource);
  }

  const auto &context = (task_type == ModelTaskType::MODEL_TASK_KERNEL) ? task_def.kernel().context() :
                                                              task_def.kernel_with_handle().context();
  const auto &iter = op_list_.find(context.op_index());
  if (iter == op_list_.cend()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Param:TaskDef]op desc not found. op index = %u", context.op_index());
    REPORT_INNER_ERR_MSG("E19999", "BuildKernelTask fail for op desc not found. op index = %u", context.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  std::unique_ptr<TbeOpTask> tbe_task = MakeUnique<TbeOpTask>(iter->second);
  if (tbe_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][TbeOpTask]failed.");
    REPORT_INNER_ERR_MSG("E19999", "BuildKernelTask fail for new TbeOpTask.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  if (!context.args_format().empty()) {
    std::vector<ArgDesc> arg_descs;
    GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(iter->second->GetOpDesc(), context.args_format(), arg_descs),
                      "Formatted args [%s] parsed failed.", context.args_format().c_str());
    if (!arg_descs.empty() && (arg_descs[0].addr_type == AddrType::FFTS_ADDR)) {
      tbe_task->ffts_addr_num_ = 1UL;
    }
  }

  SetHostMemInputFlagToTask(iter->second->GetOpDesc(), *tbe_task);
  tbe_task->SetOverflowAddr(stream_resource.GetOverflowAddr());
  auto builder = TbeTaskBuilder(model_name_, iter->second, task_def);
  const auto ret = builder.BuildTask(*tbe_task, model_params_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][TbeOpTask]failed.");
    REPORT_INNER_ERR_MSG("E19999", "[Build][TbeOpTask]failed.");
    return ret;
  }

  if (tbe_task->need_tiling_) {
    GELOGD("tiling buffer is not nullptr.");
    tbe_task->stream_resource_ = &stream_resource;
  }

  *task = tbe_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildAtomicTask(const domi::TaskDef &task_def, AtomicAddrCleanOpTask **const task,
                                      const StreamResource &stream_resource) {
  GE_CHECK_NOTNULL(task);
  const auto &context = task_def.kernel().context();
  const auto &iter = op_list_.find(context.op_index());
  if (iter == op_list_.cend()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Param:TaskDef]op desc not found. op index = %u", context.op_index());
    REPORT_INNER_ERR_MSG("E19999", "BuildKernelTask fail for op desc not found. op index = %u", context.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  std::unique_ptr<AtomicAddrCleanOpTask> atomic_task = MakeUnique<AtomicAddrCleanOpTask>();
  if (atomic_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][AtomicAddrCleanOpTask]failed.");
    REPORT_INNER_ERR_MSG("E19999", "BuildKernelTask fail for new AtomicAddrCleanOpTask.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto builder = AtomicAddrCleanTaskBuilder(model_name_, iter->second, task_def);
  atomic_task->SetOverflowAddr(stream_resource.GetOverflowAddr());
  const auto ret = builder.BuildTask(*atomic_task, model_params_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][AtomicAddrCleanOpTask]failed.");
    REPORT_INNER_ERR_MSG("E19999", "[Build][AtomicAddrCleanOpTask]failed.");
    return ret;
  }

  *task = atomic_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildKernelExTask(const domi::KernelExDef &kernel_def, AiCpuTask **const task,
                                        const uint64_t kernel_id) {
  const auto &iter = op_list_.find(kernel_def.op_index());
  if (iter == op_list_.cend()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
        "[Check][Param:KernelExDef]op not found. op index = %u", kernel_def.op_index());
    REPORT_INNER_ERR_MSG("E19999",
        "BuildKernelExTask fail for param kernel_def, because op of kernel_def not found, op index:%u.",
        kernel_def.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  std::unique_ptr<AiCpuTask> aicpu_task = MakeUnique<AiCpuTask>();
  if (aicpu_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][AiCpuTask] failed.");
    REPORT_INNER_ERR_MSG("E19999", "BuildKernelExTask fail for new AiCpuTask, model_name:%s.", model_name_.c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  SetHostMemInputFlagToTask(iter->second->GetOpDesc(), *aicpu_task);
  const auto builder = AiCpuTaskBuilder(iter->second->GetOpDesc(), kernel_def);
  const auto ret = builder.BuildTask(*aicpu_task, model_params_, kernel_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Task] failed, kernel_id:%" PRIu64 ".", kernel_id);
    return ret;
  }

  *task = aicpu_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildCpuKernelTask(const domi::KernelDef &kernel_def, AiCpuCCTask **const task,
                                         const uint64_t kernel_id) {
  const auto &context = kernel_def.context();
  const auto &iter = op_list_.find(context.op_index());
  if (iter == op_list_.cend()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
        "[Check][Param:KernelDef] op desc not found. op index = %u", context.op_index());
    REPORT_INNER_ERR_MSG("E19999",
        "BuildCpuKernelTask fail for kernel_def is invalid, because op of kernel_def not found, op index:%u.",
        context.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  std::unique_ptr<AiCpuCCTask> aicpucc_task = MakeUnique<AiCpuCCTask>();
  if (aicpucc_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][AiCpuCCTask] failed");
    REPORT_INNER_ERR_MSG("E19999", "BuildCpuKernelTask fail for new AiCpuCCTask, model_name:%s.", model_name_.c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  SetHostMemInputFlagToTask(iter->second->GetOpDesc(), *aicpucc_task);
  const auto builder = AiCpuCCTaskBuilder(iter->second->GetOpDesc(), kernel_def);
  const auto ret = builder.BuildTask(*aicpucc_task, kernel_id, model_params_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][AiCpuCCTask]failed, kernel_id:%" PRIu64 ".", kernel_id);
    REPORT_INNER_ERR_MSG("E19999", "BuildCpuKernelTask fail for build AiCpuTask, kernel_id:%" PRIu64 ".", kernel_id);
    return ret;
  }
  *task = aicpucc_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildOp(StreamResource &resource, SingleOpImpl &single_op) {
  GE_CHK_STATUS_RET_NOLOG(ParseInputsAndOutputs());
  GE_CHK_STATUS_RET_NOLOG(InitModelMem(resource));
  single_op.model_param_ = MakeUnique<SingleOpModelParam>(model_params_);
  GE_CHECK_NOTNULL(single_op.model_param_);
  single_op.root_graph_ = root_graph_;
  GE_CHK_STATUS_RET_NOLOG(SetInputsAndOutputs(single_op));
  std::string single_op_type;
  if (AttrUtils::GetStr(root_ge_model_, kAttrNameSingleOpType, single_op_type)) {
    single_op.profiling_node_type_index_ = static_cast<int64_t>(
        gert::GlobalProfilingWrapper::GetInstance()->RegisterString(
            single_op_type));
  } else {
    GELOGW("Can not find single op type from GeModel");
  }

  GE_CHK_STATUS_RET(ParseTasks(), "[Parse][Tasks] failed.");
  return BuildTaskList(resource, single_op);
}

Status SingleOpModel::BuildTaskListForDynamicOp(StreamResource &stream_resource, DynamicSingleOpImpl &single_op) {
  const auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);

  const auto compute_graph = ge_model->GetGraph();
  GE_CHECK_NOTNULL(compute_graph);
  single_op.compute_graph_ = compute_graph;

  if (node_tasks_.size() != 1U) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node size must be 1, but get %" PRIu64 ".", node_tasks_.size());
    REPORT_INNER_ERR_MSG("E19999", "[Check][Size]Node size must be 1, but get %zu.", node_tasks_.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  const auto iter = node_tasks_.cbegin();
  const auto node = iter->first;
  const auto &task_defs = iter->second;
  if (task_defs.size() <= 0U) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node size must larger then 0, but get %zu.", task_defs.size());
    REPORT_INNER_ERR_MSG("E19999", "[Check][Size]task_defs size must larger then 0, but get %zu.", task_defs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(node);
  for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    const auto peer_node = out_data_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    single_op.input_node_anchor_map_[in_data_anchor->GetIdx()] =
        {peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx()};
  }

  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto &lib_name = op_desc->GetOpKernelLibName();
  // avoid memory leak when exiting if-branch
  std::unique_ptr<OpTask> task_ptr;
  if (lib_name == kEngineNameAiCore) {
    GELOGD("Building TBE task.");
    std::vector<domi::TaskDef> aicore_task_defs;
    GE_CHK_STATUS_RET_NOLOG(GetAicoreTask(task_defs, aicore_task_defs));
    const auto &task_def = aicore_task_defs.back();
    TbeOpTask *tbe_task = nullptr;
    GE_CHK_STATUS_RET_NOLOG(BuildKernelTask(task_def, &tbe_task, stream_resource));
    task_ptr.reset(tbe_task);
    tbe_task->SetModelArgs(model_name_, model_id_);
    if (aicore_task_defs.size() == kNumTaskWithAtomicAddrCleanTask) {
      const auto &atomic_task_def = aicore_task_defs.front();
      AtomicAddrCleanOpTask *atomic_task = nullptr;
      GE_CHK_STATUS_RET_NOLOG(BuildAtomicTask(atomic_task_def, &atomic_task, stream_resource));
      GE_CHK_STATUS_RET_NOLOG(atomic_task->InitAtomicAddrCleanIndices());
      tbe_task->SetAtomicAddrCleanTask(atomic_task);
    }
    single_op.op_task_.reset(task_ptr.release());
  } else if (lib_name == kEngineNameAiCpu) {
    const auto &task_def = task_defs[0U];
    GELOGD("Building AICPU_CC task");
    AiCpuCCTask *task = nullptr;
    const uint64_t dynamic_singleop_kernel_id = aicpu_kernel_id++;
    GELOGI("Build dynamic singleOp CCTask, kernel_id = %" PRIu64, dynamic_singleop_kernel_id);
    GE_CHK_STATUS_RET_NOLOG(BuildCpuKernelTask(task_def.kernel(), &task, dynamic_singleop_kernel_id));
    task_ptr.reset(task);
    if (task->GetUnknownType() == DEPEND_COMPUTE) {
      if (task_defs.size() < kNumTaskWithMemCpyTask) {
        GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Task]The copy task of the fourth operator was not found.");
        REPORT_INNER_ERR_MSG("E19999", "The copy task of the fourth operator was not found.");
        return ACL_ERROR_GE_PARAM_INVALID;
      }
      const domi::TaskDef &copy_task_def = task_defs[1U];
      GE_CHK_STATUS_RET_NOLOG(task->SetMemCopyTask(copy_task_def.kernel()));
    }
    task->SetModelArgs(model_name_, model_id_);
    single_op.op_task_.reset(task_ptr.release());
  } else if (lib_name == kEngineNameAiCpuTf) {
    const auto &task_def = task_defs[0U];
    GELOGD("Building AICPU_TF task");
    AiCpuTask *aicpu_task = nullptr;
    const uint64_t dynamic_singleop_kernel_id = aicpu_kernel_id++;
    GELOGI("Build dynamic singleOp TfTask, kernel_id = %" PRIu64, dynamic_singleop_kernel_id);
    GE_CHK_STATUS_RET_NOLOG(BuildKernelExTask(task_def.kernel_ex(), &aicpu_task, dynamic_singleop_kernel_id));
    task_ptr.reset(aicpu_task);
    if (aicpu_task->GetUnknownType() == DEPEND_COMPUTE) {
      if (task_defs.size() < kNumTaskWithMemCpyTask) {
        GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Task]The copy task of the fourth operator was not found.");
        REPORT_INNER_ERR_MSG("E19999", "The copy task of the fourth operator was not found.");
        return ACL_ERROR_GE_PARAM_INVALID;
      }
      const domi::TaskDef &copy_task_def = task_defs[1U];
      GE_CHK_STATUS_RET_NOLOG(aicpu_task->SetMemCopyTask(copy_task_def.kernel_ex()));
    }
    aicpu_task->SetModelArgs(model_name_, model_id_);
    single_op.op_task_.reset(task_ptr.release());
  } else {
    // something
  }
  single_op.InjectRuntimeContext();
  return SUCCESS;
}

Status SingleOpModel::NeedHybridModel(bool &need_hybrid_model) {
  bool is_infer_depend = false;
  bool is_host_mem = false;
  GE_CHK_STATUS_RET(CheckInferDepend(root_graph_, is_infer_depend, is_host_mem), "[Check][InferDepend] failed.");
  const bool need_d2h_cpy = is_infer_depend && (!is_host_mem);

  // Check if GE local task with executor
  bool is_ge_local_need_hybrid = false;
  GE_CHK_STATUS_RET(CheckGeLocalNeedHybrid(root_graph_, is_ge_local_need_hybrid));

  const bool has_multi_model = (model_helper_.GetGeRootModel()->GetSubgraphInstanceNameToModel().size() > 1U);
  need_hybrid_model = need_d2h_cpy || (node_tasks_.size() > 1U) || is_ge_local_need_hybrid || has_multi_model;
  return SUCCESS;
}

Status SingleOpModel::ParseTasks() {
  const auto &tasks = root_ge_model_->GetModelTaskDefPtr()->task();
  for (int32_t i = 0; i < tasks.size(); ++i) {
    const domi::TaskDef &task_def = tasks[i];
    GELOGI("[%s] Task[%d], type = [%u], DebugString = [%s].", model_name_.c_str(), i, task_def.type(),
           task_def.DebugString().c_str());
    uint32_t op_index = 0U;
    if (!ExecutorUtils::GetOpIndex(task_def, op_index)) {
      continue;
    }
    const auto iter = op_list_.find(op_index);
    if (iter == op_list_.end()) {
      GELOGE(INTERNAL_ERROR, "[Find][Node]Failed to get node by op_index = %u", op_index);
      REPORT_INNER_ERR_MSG("E19999", "Failed to get node by op_index = %u.", op_index);
      return INTERNAL_ERROR;
    }
    auto &node = iter->second;
    node_tasks_[node].emplace_back(task_def);
  }
  return SUCCESS;
}

Status SingleOpModel::BuildDynamicOp(StreamResource &resource, DynamicSingleOpImpl &single_op) {
  single_op.num_inputs_ = data_ops_.size();
  single_op.num_outputs_ = netoutput_op_->GetAllInputsSize();
  GE_CHK_STATUS_RET(InitModelMem(resource), "[Init][ModelMem] failed.");
  model_params_.runtime_param.mem_size = UINT64_MAX;
  model_params_.graph_is_dynamic = true;
  GE_CHK_STATUS_RET(ParseTasks(), "[Parse][Tasks] failed.");

  const std::string* single_op_type = AttrUtils::GetStr(root_ge_model_, kAttrNameSingleOpType);
  if (single_op_type != nullptr) {
    ProfilingManager::Instance().RegisterElement(single_op.profiling_node_type_index_, *single_op_type);
  } else {
    single_op.profiling_node_type_index_ = -1;
    GELOGW("Can not find single op type from GeModel");
  }

  bool need_hybrid_model = false;
  GE_CHK_STATUS_RET(NeedHybridModel(need_hybrid_model), "[Check][NeedHybridModel] failed.");
  if (need_hybrid_model) {
    GELOGD("Build single op HybridModel.");
    GE_CHK_STATUS_RET(hybrid::NodeExecutorManager::GetInstance().EnsureInitialized(),
                      "[Ensure][NodeExecutor Initialized(] failed.");
    SetHostMemTensorAndNode(single_op);
    GE_CHK_STATUS(SetHostMemNode(single_op.node_with_hostmem_), "[Init][HostMem]Failed.");
    const auto root_model = model_helper_.GetGeRootModel();
    GE_CHECK_NOTNULL(root_model);
    root_model->SetRootGraph(root_graph_);
    root_model->SetSubgraphInstanceNameToModel(root_graph_->GetName(), root_ge_model_);
    single_op.hybrid_model_ = MakeUnique<hybrid::HybridModel>(root_model);
    GE_CHECK_NOTNULL(single_op.hybrid_model_);
    GE_CHK_STATUS_RET(single_op.hybrid_model_->SetOverflowAddr(resource.GetOverflowAddr(),
                                                               static_cast<uint64_t>(resource.GetOverflowSize())),
                      "[Set][OverflowAddr]failed.");
    GE_CHK_STATUS_RET(single_op.hybrid_model_->Init(true), "[Init][HybridModel]Failed.");
    int32_t device_id = 0;
    GE_CHK_RT_RET(rtGetDevice(&device_id));
    ThreadPool *thread_pool = nullptr;
    GE_CHK_STATUS_RET_NOLOG(resource.GetThreadPool(&thread_pool));
    single_op.hybrid_model_executor_ = MakeUnique<hybrid::HybridModelRtV1Executor>(single_op.hybrid_model_.get(),
                                                                               device_id,
                                                                               resource.GetStream(), thread_pool);
    GE_CHECK_NOTNULL(single_op.hybrid_model_executor_);
    hybrid::CallbackManager *callback_manager = nullptr;
    GE_CHK_STATUS_RET_NOLOG(resource.GetCallbackManager(&callback_manager));
    GE_CHK_STATUS_RET(single_op.hybrid_model_executor_->Init(callback_manager),
                      "[Init][HybridModelRtV1Executor]Failed.");
    return SUCCESS;
  }
  return BuildTaskListForDynamicOp(resource, single_op);
}

void SingleOpModel::SetHostMemInputFlagToTask(const OpDescPtr &op_desc, OpTask &task) const {
  if (ExecutorUtils::HasHostMemInput(op_desc)) {
    task.SetHostMemInputFlag(true);
    GELOGD("node[%s] has host mem", op_desc->GetName().c_str());
  }
}

void SingleOpModel::SetHostMemTensorAndNode(DynamicSingleOpImpl &single_op) const {
  for (const auto &node_map : op_with_hostmem_) {
    const auto node = node_map.second;
    single_op.hostmem_node_id_map_[node_map.first] = node->GetOpDesc()->GetId();
  }
}

Status SingleOpModel::SetHostMemNode(std::vector<NodePtr> &node_with_hostmem) {
  for (const auto &node_map : op_with_hostmem_) {
    const NodePtr node = node_map.second;
    const int32_t idx = node_map.first;
    const auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    const auto in_anchors = out_anchor->GetPeerInDataAnchorsPtr();

    for (const auto &anchor : in_anchors) {
      GE_CHECK_NOTNULL(anchor);
      auto output_node = anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(output_node);

      node_with_hostmem.emplace_back(output_node);
      GELOGD("Get %d th input tensor desc of %s by %d data node: %s.", anchor->GetIdx(),
             output_node->GetName().c_str(), idx, node->GetName().c_str());
    }
  }
  return SUCCESS;
}
}  // namespace ge
