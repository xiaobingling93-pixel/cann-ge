/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/ge/custom_task_info.h"

#include "common/checker.h"
#include "graph/debug/ge_util.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/node_utils.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "graph/load/model_manager/sink_only_allocator.h"

namespace {
bool IsInputDescValid(const ge::GeTensorDesc &input_desc, size_t &invalid_index_num) {
  if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
    if (invalid_index_num < std::numeric_limits<size_t>::max()) {
      invalid_index_num++;
    }
    return false;
  }
  return true;
}

void GetStorageShape(const ge::GeTensorDesc &input_desc, gert::StorageShape &storage_shape) {
  const auto &dims = input_desc.GetShape().GetDims();
  GELOGI("shape is %s", input_desc.GetShape().ToString().c_str());
  for (const auto &dim : dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
}
// inputs layout is input tensors
std::vector<void *> GetHoldersRawPtr(const std::vector<std::unique_ptr<uint8_t[]>> &holders) {
  std::vector<void *> holderRawPtr;
  holderRawPtr.reserve(holders.size());
  for (const auto &holder : holders) {
    (void)holderRawPtr.emplace_back(holder.get());
  }
  return holderRawPtr;
}
}

namespace ge {
Status CustomTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                              TaskRunParam &task_run_param) {
  GELOGI("CustomTaskInfo  ParseTaskRunParam start");
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();

  GE_CHECK_NOTNULL(davinci_model);
  op_desc_ = davinci_model->GetOpByIndex(context.op_index());
  GE_CHECK_NOTNULL(op_desc_);

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  input_data_addrs_ = ModelUtils::GetInputAddrsValue(rts_param, op_desc_, input_mem_types_);
  output_data_addrs_ = ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_);
  workspace_addrs_ = ModelUtils::GetWorkspaceDataAddrsValue(rts_param, op_desc_, workspace_mem_types_);
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({input_data_addrs_[i], input_mem_types_[i], false, {0}});
  }
  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({output_data_addrs_[i], output_mem_types_[i], false, {0}});
  }
  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back({workspace_addrs_[i], workspace_mem_types_[i], false, {0}});
  }
  const auto mem_size = input_data_addrs_.size() + output_data_addrs_.size() + workspace_addrs_.size();
  task_run_param.args_descs.push_back(
      {static_cast<int64_t>(MemSizeAlign(mem_size, sizeof(uintptr_t))), args_placement_});
  GELOGI(
       "Get args size[%u] of op[%s], is known node[%d], task_type: %d, placement: %d.",
       mem_size, op_desc_->GetName().c_str(),
       static_cast<int32_t>(davinci_model->IsFeatureBaseRefreshable()), static_cast<int32_t>(static_cast<ModelTaskType>(task_def.type())),
       args_placement_);
  return SUCCESS;
}

Status CustomTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                 const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                 const IowAddrs &iow_addrs) {
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHECK_NOTNULL(op_desc_);
  GELOGI("CustomTaskInfo Init Start, op: %s", op_desc_->GetNamePtr());

  (void)persistent_workspace;
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  UpdateIoAndWorkspaceAddrs(iow_addrs);
  GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement_)].dev_addr != 0U),
                 "[Check][Param] Op:%s, dev addr is nullptr.", op_desc_->GetName().c_str());
  auto mem_block_manager_allocator = davinci_model_->GetAllocator();
  sink_only_allocator_ = ComGraphMakeShared<gert::memory::SinkOnlyAllocator>();
  sink_only_allocator_->SetAllocator(mem_block_manager_allocator);

  GELOGI("CustomTaskInfo Init Success, node: %s, logic stream id: %u, stream: %p.",
    op_desc_->GetName().c_str(), task_def.stream_id(), stream_);
  return SUCCESS;
}

void CustomTaskInfo::UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs) {
  // todo: model args manager功能适配完毕后, 此处新增input_data_addrs_和iow_addrs.input_logic_addrs相等的校验
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] = (iow_addrs.input_logic_addrs.empty())
                           ? input_data_addrs_[i] : iow_addrs.input_logic_addrs[i].logic_addr;
    input_mem_types_[i] = (iow_addrs.input_logic_addrs.empty())
                          ? input_mem_types_[i] : iow_addrs.input_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] = (iow_addrs.output_logic_addrs.empty())
                            ? output_data_addrs_[i] : iow_addrs.output_logic_addrs[i].logic_addr;
    output_mem_types_[i] = (iow_addrs.output_logic_addrs.empty())
                           ? output_mem_types_[i] : iow_addrs.output_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    workspace_addrs_[i] = (iow_addrs.workspace_logic_addrs.empty())
                          ? workspace_addrs_[i] : iow_addrs.workspace_logic_addrs[i].logic_addr;
    workspace_mem_types_[i] = (iow_addrs.workspace_logic_addrs.empty())
                              ? workspace_mem_types_[i] : iow_addrs.workspace_logic_addrs[i].memory_type;
  }
}
Status CustomTaskInfo::ConstructCustomKernelContextInputsOutputs(
    const ge::OpDescPtr &op_desc, std::vector<std::unique_ptr<uint8_t[]>> &inputs,
    std::vector<std::unique_ptr<uint8_t[]>> &outputs) const {
  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    if (!IsInputDescValid(op_desc->GetInputDesc(static_cast<uint32_t>(i)), invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }
    gert::StorageShape storage_shape;
    auto input_desc = op_desc->MutableInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    GetStorageShape(*input_desc, storage_shape);
    // init tensor address, if can not get const tensor input, set it to nullptr
    const size_t instance_index = i - invalid_index_num;
    GE_ASSERT_TRUE((input_data_addrs_.size() > instance_index),
                   "instance_index %zu is invalid, %zu - %zu, total input size %zu",
                   instance_index, i, invalid_index_num, input_data_addrs_.size() );
    gert::TensorAddress address = ValueToPtr(input_data_addrs_[instance_index]);
    std::unique_ptr<uint8_t[]> tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder inputs failed.");
    new (tensor_holder.get())
        gert::Tensor(storage_shape, {input_desc->GetOriginFormat(), input_desc->GetFormat(), {}},
                     gert::kOnDeviceHbm, input_desc->GetDataType(), address);
    (void)inputs.emplace_back(std::move(tensor_holder));
  }
  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); i++) {
    gert::StorageShape storage_shape;
    auto output_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_desc);
    GetStorageShape(*output_desc, storage_shape);
    GE_ASSERT_TRUE((output_data_addrs_.size() > i),
                   "output index %zu is invalid, total output size %zu", i, output_data_addrs_.size() );
    gert::TensorAddress address = ValueToPtr(output_data_addrs_[i]);
    std::unique_ptr<uint8_t[]> tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed.");
    new (tensor_holder.get())
        gert::Tensor(storage_shape, {output_desc->GetOriginFormat(), output_desc->GetFormat(), {}},
                     gert::kOnDeviceHbm, output_desc->GetDataType(), address);
    (void)outputs.emplace_back(std::move(tensor_holder));
  }
  return SUCCESS;
}

Status CustomTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("CustomTaskInfo Distribute Start, op: %s", op_desc_->GetName().c_str());
  const TaskProfGuarder prof_guarder(this);

  AscendString op_type(op_desc_->GetType().c_str());
  auto custom_op_ptr = CustomOpFactory::CreateCustomOp(op_type);
  GE_ASSERT_NOTNULL(custom_op_ptr);
  std::vector<std::unique_ptr<uint8_t[]>> inputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> outputs_holder;
  GE_ASSERT_SUCCESS(ConstructCustomKernelContextInputsOutputs(op_desc_, inputs_holder, outputs_holder));

  std::vector<void *> ws_vec;
  eager_context_holder_ = gert::KernelRunContextBuilder()
      .Inputs(GetHoldersRawPtr(inputs_holder))
      .Inputs({sink_only_allocator_.get(), stream_})
      .Outputs(GetHoldersRawPtr(outputs_holder))
      .Outputs({&ws_vec})
      .Build(op_desc_);
  auto eager_context = reinterpret_cast<gert::EagerOpExecutionContext *>(eager_context_holder_.context_);
  GE_ASSERT_SUCCESS(custom_op_ptr->Execute(eager_context));
  GELOGI(
      "CustomTaskInfo Distribute Success, node: %s, stream_id: %u, stream: %p, task_id: %u",
      op_desc_->GetName().c_str(), stream_id_, stream_, task_id_);
  return SUCCESS;
}

Status CustomTaskInfo::Release() {
  rtContext_t ctx = nullptr;
  GE_CHK_RT(rtCtxGetCurrent(&ctx));
  sink_only_allocator_.reset();
  return SUCCESS;
}

int64_t CustomTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();
  return static_cast<int64_t>(context.op_index());
}

void CustomTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext& context = kernel_def.context();
  davinci_model_->SaveDfxInfo(context.op_index(), task_def, *this);
}


REGISTER_TASK_INFO(MODEL_TASK_CUSTOM_KERNEL, CustomTaskInfo);
}  // namespace ge