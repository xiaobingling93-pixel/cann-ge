/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/aicpu/kernel_ex_task_info.h"

#include "common/checker.h"
#include "graph/utils/math_util.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "graph/load/model_manager/kernel/kernel_register_info_builder.h"
#include "acl/acl_rt.h"

namespace {
const std::string kAttrAicpuAllshape = "_AllShape";
}  // namespace

namespace ge {
bool KernelExTaskInfo::NeedUpdateAddr(const OpDescPtr &op_desc) const {
  return davinci_model_->IsFeatureBaseRefreshable() || davinci_model_->HasZeroCopyAddr(op_desc);
}

Status KernelExTaskInfo::InitTaskExtInfo(const std::string &ext_info, const OpDescPtr &op_desc) {
  if (ext_info.empty()) {
    return SUCCESS;
  }
  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  const auto unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  const uint32_t num_inputs = static_cast<uint32_t>(op_desc->GetInputsSize());
  const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());

  hybrid::AicpuExtInfoHandler ext_handle(op_desc->GetName(), num_inputs, num_outputs, unknown_type);
  GE_CHK_STATUS_RET(ext_handle.Parse(ext_info), "[Parse][KernelExtInfo] failed, ext_info_size=%zu", ext_info.size());
  const bool need_update = NeedUpdateAddr(op_desc);
  // 1 means static(no need update), 0 means dynamic(need_update)
  GE_CHK_STATUS_RET(ext_handle.UpdateExecuteMode(!need_update), "[Update][ExecuteMode] failed.");
  GELOGD("Update aicpu_task ext_info bit_map execute mode to %d.", static_cast<int32_t>(!need_update));
  deploy_type_flag_ = ext_handle.GetDeployTypeFlag();
  mem_type_ = ext_handle.GetMemType();
  memcpy_kind_ = ext_handle.GetMemcpyKind();
  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAttrAicpuAllshape, all_shape);
  if (all_shape) {
    GELOGD("Aicpu all_shape kernel need to update io shape.");
    for (uint32_t i = 0U; i < num_inputs; i++) {
      const auto input_desc = op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET(ext_handle.UpdateInputShapeAndType(i, *input_desc),
                        "[Call][UpdateInputShapeAndType] Input[%u] update input shape failed, op:%s.",
                        i, op_desc->GetName().c_str());
    }
    if (unknown_type != DEPEND_COMPUTE) {
      for (uint32_t i = 0U; i < num_outputs; ++i) {
        const auto output_desc = op_desc->MutableOutputDesc(i);
        GE_CHECK_NOTNULL(output_desc);
        GE_CHK_STATUS_RET(ext_handle.UpdateOutputShapeAndType(i, *output_desc),
                          "[Call][UpdateOutputShapeAndType] Output[%u] update output shape failed, op:%s.",
                          i, op_desc->GetName().c_str());
      }
    }
  }

  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op_);
  GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc->GetName().c_str(),
         static_cast<int32_t>(is_blocking_aicpu_op_));

  if (UpdateEventIdForAicpuBlockingOp(op_desc, ext_handle) != SUCCESS) {
    GELOGE(FAILED, "[Call][UpdateEventIdForAicpuBlockingOp] failed for op:%s(%s)",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  ext_info_addr_ = davinci_model_->MallocDynamicMemory(ext_handle.GetExtInfoLen(), mem_type_);
  GE_ASSERT_NOTNULL(ext_info_addr_);
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_, ext_handle.GetExtInfoLen(), ext_handle.GetExtInfo(),
                         ext_handle.GetExtInfoLen(), memcpy_kind_));
  GELOGD("Op %s use %s mem %p for ext info with flag %d", op_desc->GetName().c_str(),
         mem_type_ == RT_MEMORY_HOST_SVM ? "host" : "device", ext_info_addr_, deploy_type_flag_);
  return SUCCESS;
}

Status KernelExTaskInfo::InitKernelBufferAddr() {
  kernel_buf_size_ = static_cast<uint32_t>(sizeof(STR_FWK_OP_KERNEL));
  kernel_buf_ = davinci_model_->MallocDynamicMemory(kernel_buf_size_, mem_type_);
  GE_ASSERT_NOTNULL(kernel_buf_);
  GELOGD("Op %s use %s mem %p for kernel_buf with flag %d", op_desc_->GetName().c_str(),
         mem_type_ == RT_MEMORY_HOST_SVM ? "host" : "device", kernel_buf_, deploy_type_flag_);
  return SUCCESS;
}

Status KernelExTaskInfo::InitInputOutputAddr(const PisToArgs &args, const IowAddrs &iow_addrs) {
  // todo: model args manager功能适配完毕后, 此处新增input_data_addrs_和iow_addrs.input_logic_addrs相等的校验
  for (size_t i = 0U; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] = ValueToPtr(iow_addrs.input_logic_addrs[i].logic_addr);
    input_addr_mem_types_[i] = iow_addrs.input_logic_addrs[i].memory_type;
  }

  for (size_t i = 0U; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] = ValueToPtr(iow_addrs.output_logic_addrs[i].logic_addr);
    output_addr_mem_types_[i] = iow_addrs.output_logic_addrs[i].memory_type;
  }

  (void)io_addrs_.insert(io_addrs_.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
  (void)io_addrs_.insert(io_addrs_.cend(), output_data_addrs_.cbegin(), output_data_addrs_.cend());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), input_addr_mem_types_.cbegin(),
      input_addr_mem_types_.cend());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), output_addr_mem_types_.cbegin(),
      output_addr_mem_types_.cend());

  addrs_size_ = sizeof(uint64_t) * io_addrs_.size();
  if (deploy_type_flag_ == static_cast<int32_t>(RT_KERNEL_HOST_ONLY)) {
    input_output_addr_ = ValueToPtr(args[static_cast<uint32_t>(ArgsPlacement::kArgsPlacementHostSvm)].dev_addr);
  } else {
    input_output_addr_ = ValueToPtr(args[static_cast<uint32_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr);
  }
  GELOGD("Op %s deploy_type_flag %d, input_output_addr %p, addrs_size %zu",
         op_desc_->GetName().c_str(), deploy_type_flag_, input_output_addr_, addrs_size_);

  GE_ASSERT_TRUE(input_output_addr_ != nullptr);

  return SUCCESS;
}

Status KernelExTaskInfo::AssembleKernelBuffer(const STR_FWK_OP_KERNEL * const fwk_op_kernel) const {
  GE_CHK_RT_RET(rtMemcpy(kernel_buf_, kernel_buf_size_, PtrToPtr<STR_FWK_OP_KERNEL, void>(fwk_op_kernel),
                         kernel_buf_size_, memcpy_kind_));
  GELOGD("Op %s use %s mem %p for kernel_buf with flag %d", op_desc_->GetName().c_str(),
         mem_type_ == RT_MEMORY_HOST_SVM ? "host" : "device", kernel_buf_, deploy_type_flag_);
  return SUCCESS;
}

Status KernelExTaskInfo::AssembleInputOutputAddr() {
  if (addrs_size_ <= 0UL) {
    return SUCCESS;
  }
  auto io_addrs = VPtrToValue(io_addrs_);
  const auto zero_copy_args_index = davinci_model_->GetZeroCopyArgsIndex(io_addrs);
  if (!zero_copy_args_index.empty()) {
    const std::vector<bool> input_raw_data_list = ModelUtils::GetInputTensorNeedRawData(op_desc_);
    std::vector<bool> need_raw_data_list;
    std::map<uintptr_t, std::set<size_t>> zero_copy_args_offset;
    for (const auto &args_index : zero_copy_args_index) {
      const uintptr_t io_addr = static_cast<uintptr_t>(PtrToValue(io_addrs_[args_index]));
      (void)zero_copy_args_offset[io_addr].insert(args_index * sizeof(uint64_t));
      if (args_index < input_raw_data_list.size()) {
        need_raw_data_list.push_back(input_raw_data_list[args_index]);
      }
    }
    need_raw_data_list.resize(zero_copy_args_index.size(), false);
    GE_CHK_STATUS_RET(
        davinci_model_->Mapping2BundleZeroCopy(op_desc_, zero_copy_args_offset, need_raw_data_list, addrs_size_,
                                               io_addrs_.data(), input_output_addr_, own_args_memory_),
        "Failed mapping zero copy task for %s to bundle task", op_desc_->GetName().c_str());
  }
  InitDumpFlag(op_desc_);
  InitDumpArgs(input_output_addr_, op_desc_);

  return SUCCESS;
}

aclrtFuncHandle KernelExTaskInfo::GetFuncHandle() {
  auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kAicpu);
  GE_ASSERT_NOTNULL(kernel_handles_manager);
  GE_ASSERT_NOTNULL(op_desc_);
  KernelRegisterInfo register_info;
  GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructAicpuRegisterInfo(op_desc_->GetType(),
      "libtf_kernels.so", "TFOperateAPI", "TFKernel", register_info));
  const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
  auto bin_handle = kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  GE_ASSERT_NOTNULL(bin_handle);
  return KernelHandleUtils::GetFuncHandle(bin_handle, op_desc_->GetType());
}

Status KernelExTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                              const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                              const IowAddrs &iow_addrs) {
  GELOGI("KernelExTaskInfo Init Start.");
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  // 1. Copy context from kernelExDef.private to workspace
  const auto &kernel_ex_def = task_def.kernel_ex();
  const uint32_t op_index = kernel_ex_def.op_index();
  const OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  op_desc_ = op_desc;
  func_handle_ = GetFuncHandle();
  GE_ASSERT_NOTNULL(func_handle_);
  // 2. Reconstruct kernelExDef.args to STR_FWK_OP_KERNEL
  STR_FWK_OP_KERNEL fwk_op_kernel = {};
  const size_t args_size = kernel_ex_def.args().size();
  if (args_size > sizeof(STR_FWK_OP_KERNEL)) {
    REPORT_INNER_ERR_MSG("E19999", "Param kernel_ex_def.args().size():%zu > sizeof(STR_FWK_OP_KERNEL):%zu, check invalid",
                       args_size, sizeof(STR_FWK_OP_KERNEL));
    GELOGE(FAILED, "[Check][Param] kernel_ex_def.args().size():%zu > sizeof(STR_FWK_OP_KERNEL):%zu",
           args_size, sizeof(STR_FWK_OP_KERNEL));
    return FAILED;
  }
  const errno_t sec_ret = memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), kernel_ex_def.args().data(), args_size);
  if (sec_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "Call memcpy_s fail, size:%zu, ret:%d", sizeof(STR_FWK_OP_KERNEL), sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%zu, ret: %d", sizeof(STR_FWK_OP_KERNEL), sec_ret);
    return FAILED;
  }

  const auto &ext_info = kernel_ex_def.kernel_ext_info();
  GE_CHK_STATUS_RET(InitTaskExtInfo(ext_info, op_desc),
                    "[Init][TaskExtInfo] failed, ext_info size=%zu, op:%s",
                    ext_info.size(), op_desc->GetName().c_str());

  GELOGI("Node[%s] type[%s] kernel_ext_info size=%zu, ext_info_addr_=%p, deploy_type_flag %d",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), ext_info.size(), ext_info_addr_, deploy_type_flag_);

  // 2.1 get SessionId and loop variable for tensor array write
  const uint64_t session_id = davinci_model_->GetSessionId();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = session_id;

  // 2.2 Collect aicpu kernel
  const uint64_t kernel_id = hybrid::AicpuExtInfoHandler::GenerateKernelId();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = kernel_id;
  ModelManager::GetInstance().CreateAicpuKernel(session_id, davinci_model->Id(),
                                                davinci_model->SubModelId(), kernel_id);

  // 2.3 Create session
  GE_CHK_STATUS_RET_NOLOG(ModelManager::GetInstance().CreateAicpuSession(session_id));
  // init kernel buffer addr
  GE_CHK_STATUS_RET(InitKernelBufferAddr(), "[Init][Param] failed for [%s].", op_desc_->GetName().c_str());
  // init inputOutputDataAddr
  GE_ASSERT_TRUE((iow_addrs.input_logic_addrs.size() == input_data_addrs_.size()),
                 "[Check][Param] Op:%s(%s) input logic addrs list size:%zu != input data addr list size:%zu",
                 op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                 iow_addrs.input_logic_addrs.size(), input_data_addrs_.size());

  GE_ASSERT_TRUE((iow_addrs.output_logic_addrs.size() == output_data_addrs_.size()),
                 "[Check][Param] Op:%s(%s) output logic addrs list size:%zu != output data addr list size:%zu",
                 op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                 iow_addrs.output_logic_addrs.size(), output_data_addrs_.size());
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  GE_CHK_STATUS_RET(InitInputOutputAddr(args, iow_addrs), "[Init][Param] failed for [%s].",
                    op_desc_->GetName().c_str());
  // Assemble workspaceaddr
  GE_CHK_STATUS_RET(AssembleWorkSpaceAddr(kernel_ex_def, rts_param, op_desc),
                    "[Assemble][WorkSpace] to workspace failed, op:%s.", op_desc->GetName().c_str());
  // Assemble inputOutputDataAddr
  GE_CHK_STATUS_RET(AssembleInputOutputAddr(), "[Assemble][Param] failed for [%s].", op_desc_->GetName().c_str());
  // Assemble fwk_op_kernel
  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = PtrToValue(workspace_data_addrs_[0U]);
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = PtrToValue(input_output_addr_);
  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = davinci_model_->GetGlobalStep();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = ext_info.size();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = PtrToValue(ext_info_addr_);

  // Assemble kernel buffer
  GE_CHK_STATUS_RET(AssembleKernelBuffer(&fwk_op_kernel), "[Assemble][Param] failed for [%s].",
                    op_desc_->GetName().c_str());

  io_addr_mem_types_.resize(io_addrs_.size(), kFixMemType);
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), VPtrToValue(io_addrs_),
      io_addr_mem_types_, {op_desc->GetName(), op_desc->GetType()}));

  GELOGI("KernelExTaskInfo Init Success, node %s, session id: %" PRIu64 ", logic stream id: %u, stream: %p",
    op_desc_->GetName().c_str(), session_id, task_def.stream_id(), stream_);
  return SUCCESS;
}

void KernelExTaskInfo::InitDumpFlag(const OpDescPtr &op_desc) {
  if (davinci_model_->OpNeedDump(op_desc) || davinci_model_->OpNeedPrint(op_desc)
    || davinci_model_->OpNeedSetDumpFlagOnWatcherModel(op_desc->GetName())) {
    GELOGD("Op %s need init dump flag in kernel ex task info", op_desc->GetName().c_str());
    is_data_dump_ = true;
  }
}

void KernelExTaskInfo::InitDumpArgs(void *const addr, const OpDescPtr &op_desc) {
  if (davinci_model_->OpNeedDump(op_desc) || davinci_model_->OpNeedDumpOnWatcherModel(op_desc->GetName())) {
    GELOGD("Op %s need dump in kernel ex task info", op_desc->GetName().c_str());
    dump_args_ = addr;
  }
  if (davinci_model_->GetOpDugReg()) {
    GELOGD("Op debug is open in kernel ex task info");
    dump_args_ = addr;
  }
}

Status KernelExTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                           TaskRunParam &task_run_param) {
  (void)task_run_param;
  const auto &kernel_ex_def = task_def.kernel_ex();
  const uint32_t op_index = kernel_ex_def.op_index();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Can't get op_desc from davinci_model by index:%u", op_index);
    GELOGE(INTERNAL_ERROR, "[Get][Op] By Index, index:%u is out of range!", op_index);
    return INTERNAL_ERROR;
  }
  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  input_data_addrs_ = ModelUtils::GetInputAddrs(rts_param, op_desc, input_addr_mem_types_);
  output_data_addrs_ = ModelUtils::GetOutputAddrs(rts_param, op_desc, output_addr_mem_types_);
  for (size_t i = 0U; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({PtrToValue(input_data_addrs_[i]), input_addr_mem_types_[i],
                                                true, {0}});
  }
  for (size_t i = 0U; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({PtrToValue(output_data_addrs_[i]), output_addr_mem_types_[i],
                                                 true, {0}});
  }

  const size_t inputs_size = op_desc->GetInputsSize();
  const size_t outputs_size = op_desc->GetOutputsSize();
  REQUIRE_COMPAT_UINT32(sizeof(uint64_t) * (inputs_size + outputs_size));

  uint32_t mem_size = static_cast<uint32_t>(sizeof(uint64_t) * (inputs_size + outputs_size));
  const uint32_t mem_size_t =
    static_cast<uint32_t>(sizeof(uint64_t) * (input_data_addrs_.size() + output_data_addrs_.size()));

  GELOGD("mem_size %u, inputs_size %zu, outputs_size %zu, input_data_addrs size %zu, output_data_addrs size %zu.",
      mem_size, inputs_size, outputs_size, input_data_addrs_.size(), output_data_addrs_.size());

  mem_size = (mem_size > mem_size_t) ? mem_size :mem_size_t;

  int32_t deploy_type_flag = static_cast<int32_t>(RT_KERNEL_DEVICE_FIRST);
  const auto &ext_info = kernel_ex_def.kernel_ext_info();
  if (!ext_info.empty()) {
    int32_t unknown_shape_type_val = 0;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
    const auto unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
    const uint32_t num_inputs = static_cast<uint32_t>(op_desc->GetInputsSize());
    const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());

    hybrid::AicpuExtInfoHandler ext_handle(op_desc->GetName(), num_inputs, num_outputs, unknown_type);
    GE_CHK_STATUS_RET(ext_handle.Parse(ext_info), "[Parse][KernelExtInfo] failed, ext_info_size=%zu", ext_info.size());
    deploy_type_flag = ext_handle.GetDeployTypeFlag();
  }
  pls_ = (deploy_type_flag == static_cast<int32_t>(RT_KERNEL_HOST_ONLY))
      ? ArgsPlacement::kArgsPlacementHostSvm : ArgsPlacement::kArgsPlacementHbm;
  task_run_param.args_descs.push_back({static_cast<int64_t>(mem_size) + 8, pls_});
  GELOGI("kernel task name %s, args_size %u, args_size_t %u pls %u, deploy_type_flag %d", op_desc->GetName().c_str(),
      mem_size, mem_size_t, static_cast<uint32_t>(pls_), deploy_type_flag);

  return SUCCESS;
}

Status KernelExTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                        const size_t host_args_max_len) {
  GELOGI("KernelExTaskInfo::UpdateArgs in.");
  std::vector<uint64_t> io_addrs_updated;
  io_addrs_updated.reserve(io_addrs_.size());
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_max_len));
  uint64_t *host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
  GE_ASSERT_TRUE(io_addrs_.size() * sizeof(uint64_t) <= host_args_max_len);
  for (size_t index = 0; index < io_addrs_.size(); ++index) {
    io_addrs_updated.push_back(*host_args_tmp++);
  }
  davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, io_addrs_updated);
  GELOGI("KernelExTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status KernelExTaskInfo::UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) {
  GELOGI("KernelExTaskInfo::UpdateDumpInfos in.");
  std::vector<uint64_t> io_addrs_updated;
  io_addrs_updated.reserve(io_addrs_.size());
  uint64_t *host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
  GE_ASSERT_TRUE(io_addrs_.size() * sizeof(uint64_t) <= host_args_max_len);
  for (size_t index = 0; index < io_addrs_.size(); ++index) {
    io_addrs_updated.push_back(*host_args_tmp++);
  }
  davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, io_addrs_updated);
  GELOGI("KernelExTaskInfo::UpdateDumpInfos success.");
  return SUCCESS;
}

Status KernelExTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, pls_);
  return SUCCESS;
}

Status KernelExTaskInfo::AssembleWorkSpaceAddr(const domi::KernelExDef &kernel_def, const RuntimeParam &rts_param,
                                               const OpDescPtr &op_desc) {
  (void)rts_param;
  // todo: workspace地址不能用编译态分配的结果，因为workspace来自于fm，fm可刷新场景下，该task不支持刷新workspace地址
  // 故此处用独立申请的地址，后面修改为persistent workspace
  void *workspace_base_addr = davinci_model_->MallocDynamicMemory(kernel_def.task_info().size(), mem_type_);
  GE_ASSERT_NOTNULL(workspace_base_addr);
  GE_CHK_RT_RET(rtMemcpy(workspace_base_addr, kernel_def.task_info().size(), kernel_def.task_info().data(),
                         kernel_def.task_info().size(), memcpy_kind_));
  workspace_data_addrs_.emplace_back(workspace_base_addr);

  GELOGI("Op %s use %s mem %p for workspace_base_addr with flag %d", op_desc->GetName().c_str(),
         mem_type_ == RT_MEMORY_HOST_SVM ? "host" : "device", workspace_base_addr, deploy_type_flag_);

  return SUCCESS;
}

Status KernelExTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("KernelExTaskInfo %s Distribute Start.", op_desc_->GetNamePtr());
  if (davinci_model_ != nullptr && davinci_model_->IsDumpOpWithAdump()) {
    GELOGD("Both overflow detection and persistent stream unlimited enabled, disable dump for op %s",
            op_desc_ ? op_desc_->GetName().c_str() : "unknown");
    is_data_dump_ = false;
  }
  const TaskProfGuarder prof_guarder(this);
  SetTaskTag(op_desc_->GetName().c_str());
  LaunchKernelParam launch_kernel_param;
  launch_kernel_param.args = kernel_buf_;
  launch_kernel_param.args_size = kernel_buf_size_;
  launch_kernel_param.block_dim = 1U;
  launch_kernel_param.stream = stream_;
  launch_kernel_param.launch_config.is_data_dump = is_data_dump_;
  GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(func_handle_, launch_kernel_param));
  GE_CHECK_NOTNULL(davinci_model_);
  GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
  GE_CHK_RT_RET(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)));

  GELOGI("KernelExTaskInfo %s Distribute Success. task id: %u, stream id: %u, stream: %p.",
         op_desc_->GetNamePtr(), task_id_, stream_id_, stream_);
  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp() != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] Call DistributeWaitTaskForAicpuBlockingOp failed");
      return FAILED;
    }
  }

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
  }
  is_support_redistribute_ = true;

  return SUCCESS;
}

void KernelExTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  const auto &kernel_ex_def = task_def.kernel_ex();
  davinci_model_->SaveDfxInfo(kernel_ex_def.op_index(), task_def, *this);
}

Status KernelExTaskInfo::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const {
  int32_t device_id = 0;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));
  int32_t value = 0;
  GE_CHK_RT_RET(rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value));

  if ((value != RT_AICPU_BLOCKING_OP_NOT_SUPPORT) && (value != RT_AICPU_BLOCKING_OP_SUPPORT)) {
    REPORT_INNER_ERR_MSG("E19999", "Value should be %d or %d but %d",
                       RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    GELOGE(FAILED, "[Check][Value] Value should be %d or %d but %d",
           RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    return FAILED;
  }
  is_support = (value == RT_AICPU_BLOCKING_OP_SUPPORT);
  return SUCCESS;
}

Status KernelExTaskInfo::UpdateEventIdForAicpuBlockingOp(const OpDescPtr &op_desc,
                                                         const hybrid::AicpuExtInfoHandler &ext_handle) const {
  if (is_blocking_aicpu_op_) {
    bool is_support = false;
    if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
      GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOp failed");
      return FAILED;
    }
    if (!is_support) {
      GELOGD("Device not support blocking aicpu op process");
      return SUCCESS;
    }
    uint32_t event_id = 0U;
    if (davinci_model_->GetEventIdForBlockingAicpuOp(op_desc, stream_, event_id) != SUCCESS) {
      GELOGE(FAILED, "[Get][EventId] Get event id failed for op:%s(%s)", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return FAILED;
    }
    if (ext_handle.UpdateEventId(event_id) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Update event id failed for op:%s(%s).", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str());
      GELOGE(FAILED, "[Update][EventId] Update event id failed for op:%s(%s)", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return FAILED;
    }
    GELOGI("Update event_id=%u success", event_id);
  }
  return SUCCESS;
}

Status KernelExTaskInfo::DistributeWaitTaskForAicpuBlockingOp() const {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOp failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("Device not support blocking aicpu op process.");
    return SUCCESS;
  }
  GELOGD("Distribute wait task begin");
  aclrtEvent rt_event = nullptr;
  if (davinci_model_->GetEventByStream(stream_, rt_event) != SUCCESS) {
    GELOGE(FAILED, "[Call][GetEventByStream] Call GetEventByStream failed");
    return FAILED;
  }
  uint32_t timeout = 0xffffffff;
  (void)AttrUtils::GetInt(op_desc_, ATTR_NAME_BLOCKING_OP_TIMEOUT, timeout);
  GE_CHK_RT_RET(rtStreamWaitEventWithTimeout(stream_, rt_event, timeout));
  GE_CHK_RT_RET(aclrtResetEvent(rt_event, stream_));

  return SUCCESS;
}

Status KernelExTaskInfo::Release() {
  kernel_buf_ = nullptr;
  input_output_addr_ = nullptr;
  ext_info_addr_ = nullptr;

  ext_args_.clear();
  workspace_data_addrs_.clear();
  return SUCCESS;
}

int64_t KernelExTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const auto &kernel_ex_def = task_def.kernel_ex();
  return static_cast<int64_t>(kernel_ex_def.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_KERNEL_EX, KernelExTaskInfo);
}  // namespace ge
