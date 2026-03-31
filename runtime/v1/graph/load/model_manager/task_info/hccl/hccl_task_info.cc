/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/hccl/hccl_task_info.h"

#include "common/checker.h"
#include "graph/ge_context.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "acl/acl_rt.h"
#include "acl/acl_mdl.h"

namespace {
const ge::char_t *const kDumpOutput = "output";
const ge::char_t *const kDumpInput = "input";
}
namespace ge {
constexpr uint32_t kZeroIndex = 0U;
constexpr int32_t kValidTypeSize = 1;
constexpr size_t kAddressLen = sizeof(uint64_t);
constexpr int32_t kSupportZeroCopy = 1;
std::mutex HcclTaskInfo::hccl_follow_stream_mutex_;

HcclTaskInfo::~HcclTaskInfo() {
  if (private_def_ != nullptr) {
    const rtError_t ret = rtFreeHost(private_def_);
    if (ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtFreeHost failed, ret:%d", ret);
      GELOGE(RT_FAILED, "[Call][RtFree] Fail, ret = %d.", ret);
    }
    private_def_ = nullptr;
  }
  davinci_model_ = nullptr;
  ops_kernel_store_ = nullptr;
  args_ = nullptr;
}

Status HcclTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                          const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                          const IowAddrs &iow_addrs) {
  GELOGI("HcclTaskInfo Init Start.");
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;

  logic_stream_id_ = task_def.stream_id();
  GE_CHK_STATUS_RET_NOLOG(SetStream(logic_stream_id_, davinci_model_->GetStreamList()));
  // 保存hccl所在的stream
  davinci_model_->SetHcclTaskStream(stream_);
  stream_flag_ = davinci_model_->GetStreamFlagById(logic_stream_id_);
  GELOGI("Hccl stream id %u, stream flag %u.", logic_stream_id_, stream_flag_);

  static std::atomic<uint32_t> hccl_task_id(0U);
  id_ = hccl_task_id.fetch_add(1U);

  const auto &hccl_def = task_def.kernel_hccl();
  const uint32_t op_index = hccl_def.op_index();

  // Get HCCL op
  hccl_op_desc_ = davinci_model_->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(hccl_op_desc_);
  GELOGI("HcclTaskInfo Init, logical stream id: %u, op_index is: %u, op:%s(%s)", logic_stream_id_, op_index,
         hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
  GetPrivateDefByTaskDef(hccl_op_desc_, task_def);

  // Create the kernel hccl infos
  CreateKernelHcclInfo(hccl_op_desc_);

  // Initialize the hccl_type of all kernel hccl info
  HcomOmeUtil::GetHcclType(task_def, kernel_hccl_infos_);

  // Only in Horovod scenario should get the inputName and GeShape
  auto ret = HcomOmeUtil::GetHorovodInputs(hccl_op_desc_, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][HorovodInputs] fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return ret;
  }
  Status dmrt = HcomOmeUtil::GetHcclDataType(hccl_op_desc_, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    GELOGE(dmrt, "[Get][HcomDataType] fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return dmrt;
  }
  dmrt = HcomOmeUtil::GetHcclCount(hccl_op_desc_, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call GetHcclCount fail for op:%s(%s)",
                      hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    GELOGE(dmrt, "[Get][HcomCount] fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return dmrt;
  }
  // Only HCOMBROADCAST, HCOMGATHER and HVDCALLBACKBROADCAST need to get the rootId
  dmrt = HcomOmeUtil::GetAllRootId(hccl_op_desc_, kernel_hccl_infos_);
  if (dmrt != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call GetAllRootId fail for op:%s(%s)",
                      hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    GELOGE(dmrt, "[Get][RootId] fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return dmrt;
  }

  // GE's new process: hccl declares the number of streams required, creates a stream by GE, and sends it to hccl
  ret = SetFollowStream(hccl_op_desc_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Stream] Fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return ret;
  }
  UpdateIoAndWorkspaceAddrs(iow_addrs);

  const auto args_placement = ((args_mem_type_ & RT_MEMORY_TS) == 0U)
      ? ArgsPlacement::kArgsPlacementHbm : ArgsPlacement::kArgsPlacementTs;
  GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement)].dev_addr != 0U),
                 "[Check][Param] Op:%s, args_placement:%d, dev addr is nullptr.",
                 hccl_op_desc_->GetName().c_str(), args_placement);
  args_ = ValueToPtr(args[static_cast<size_t>(args_placement)].dev_addr);
  GELOGI("Known node %s args addr %p.", hccl_op_desc_->GetName().c_str(), args_);

  GE_CHK_STATUS_RET(InitZeroCopyInfos(hccl_op_desc_, hccl_def),
                    "Init ZeroCopyInfos failed, node:%s(%s).",
                    hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());

  ret = SetAddrs(hccl_op_desc_, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Addrs] Fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return ret;
  }
  // GE's new process: hccl declares the need for Workspace size, and GE allocates Workspace
  ret = SetWorkspace(hccl_op_desc_, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Workspace] Fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return ret;
  }

  // set hccl op overflow detection addr
  ret = SetOverflowAddrs(hccl_op_desc_, kernel_hccl_infos_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][OverflowAddrs] Fail for op:%s(%s)",
        hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
    return ret;
  }

  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(),
      io_addrs_, io_addr_mem_types_, {hccl_op_desc_->GetName(), hccl_op_desc_->GetType()}));
  GE_CHECK_NOTNULL(ops_kernel_store_);
  GETaskInfo ge_task;
  TransToGETaskInfo(ge_task);
  const auto result = ops_kernel_store_->PrepareTaskAsync(ge_task);
  GE_CHK_BOOL_RET_STATUS(result == HCCL_SUCCESS, INTERNAL_ERROR, "[Prepare][Task] fail, return ret:%u", result);
  GELOGI("HcclTaskInfo Init Success, prepare task success for op:%s(%s), logic stream id: %u, stream: %p.",
      hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str(), task_def.stream_id(), stream_);
  return SUCCESS;
}

Status HcclTaskInfo::GetTaskIowPaRemapInfos(std::vector<IowPaRemapInfo> &infos) {
  const auto input_size_list = ModelUtils::GetInputSize(hccl_op_desc_);
  const auto output_size_list = ModelUtils::GetOutputSize(hccl_op_desc_);
  const auto workspace_size_list = ModelUtils::GetWorkspaceSize(hccl_op_desc_);

  GE_ASSERT_EQ(input_data_addrs_.size(), input_size_list.size());
  GE_ASSERT_EQ(output_data_addrs_.size(), output_size_list.size());
  GE_ASSERT_EQ(workspace_addrs_.size(), workspace_size_list.size());

  std::vector<int64_t> io_tensor_size;
  (void)io_tensor_size.insert(io_tensor_size.cend(), input_size_list.cbegin(), input_size_list.cend());
  (void)io_tensor_size.insert(io_tensor_size.cend(), output_size_list.cbegin(), output_size_list.cend());
  (void)io_tensor_size.insert(io_tensor_size.cend(), workspace_size_list.cbegin(), workspace_size_list.cend());

  std::vector<MemAllocationAndOffset> mem_allocation_and_offset;
  args_io_addrs_updater_.GetArgsMemAllocationAndOffset(mem_allocation_and_offset);
  GE_ASSERT_EQ(mem_allocation_and_offset.size(), io_tensor_size.size());

  for (size_t i = 0U; i < mem_allocation_and_offset.size(); i++) {
    IowPaRemapInfo iow_pa_remap_info{};
    iow_pa_remap_info.allocation_id = mem_allocation_and_offset[i].id;
    iow_pa_remap_info.allocation_offset = mem_allocation_and_offset[i].offset;
    iow_pa_remap_info.tensor_size = io_tensor_size[i];

    iow_pa_remap_info.policy = PaRemapPolicy::KSupport;
    if (!(IsFeatureBaseRefreshable(davinci_model_)) ||
      (mem_allocation_and_offset[i].type == MemAllocation::Type::ABSOLUTE)) {
      iow_pa_remap_info.policy = PaRemapPolicy::KNoSupport;
      infos.emplace_back(std::move(iow_pa_remap_info));
    }
  }

  return SUCCESS;
}

Status HcclTaskInfo::SetFollowStream(const ConstOpDescPtr &op_desc) {
  if (!HcomOmeUtil::IsHCOMOp(op_desc->GetType())) {
    GELOGI("Node %s Optye %s no need to create slave streams.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return SUCCESS;
  }

  int64_t hccl_stream_num = 0;
  if ((!AttrUtils::GetInt(op_desc, "used_stream_num", hccl_stream_num)) || (hccl_stream_num < 0)) {
    GELOGI("op_desc has no attr used_stream_num or is invalid!");
  }

  const std::lock_guard<std::mutex> lock(hccl_follow_stream_mutex_);
  const int64_t main_stream_id = op_desc->GetStreamId();
  const std::map<int64_t, std::vector<rtStream_t>> &main_follow_stream_mapping = davinci_model_->GetHcclFolowStream();

  if (main_follow_stream_mapping.find(main_stream_id) != main_follow_stream_mapping.end()) {
    const std::vector<rtStream_t> &follow_stream_usage = main_follow_stream_mapping.at(main_stream_id);
    GE_CHECK_GE(hccl_stream_num, 0);
    if (static_cast<size_t>(hccl_stream_num) <= follow_stream_usage.size()) {
      GELOGI("capacity of follow stream is enough to be reused.");
      for (size_t i = 0UL; i < static_cast<size_t>(hccl_stream_num); i++) {
        hccl_stream_list_.emplace_back(follow_stream_usage.at(i));
      }
    } else {
      GELOGI("need to reuse follow stream and create new follow stream.");
      const size_t created_stream_num = follow_stream_usage.size();
      for (const auto &stream : follow_stream_usage) {
        hccl_stream_list_.emplace_back(stream);
      }
      const auto ret = CreateStream(hccl_stream_num - static_cast<int64_t>(created_stream_num), main_stream_id);
      if (ret != SUCCESS) {
        GELOGE(RT_FAILED, "[Create][Stream] for %s failed, stream id:%" PRId64 ", stream num:%" PRIu64 ".",
               op_desc->GetName().c_str(), main_stream_id, static_cast<uint64_t>(hccl_stream_num) - created_stream_num);
        return RT_ERROR_TO_GE_STATUS(ret);
      }
    }
    GELOGI("Initialize hccl slave stream success, hcclStreamNum =%" PRId64, hccl_stream_num);
  } else {
    GELOGI("need to create follow stream for %s with new mainstream %" PRId64 ".",
      op_desc->GetName().c_str(), main_stream_id);
    const auto ret = CreateStream(hccl_stream_num, main_stream_id);
    if (ret != SUCCESS) {
      GELOGE(RT_FAILED, "[Create][Stream] for %s failed, stream id:%" PRId64 ", stream num:%" PRId64 ".",
             op_desc->GetName().c_str(), main_stream_id, hccl_stream_num);
      return RT_ERROR_TO_GE_STATUS(ret);
    }
  }
  return SUCCESS;
}

Status HcclTaskInfo::CreateStream(const int64_t stream_num, const int64_t main_stream_id) {
  GELOGI("Start to create %" PRId64 " hccl stream.", stream_num);
  const bool isOverflowDetectionOpen = GetContext().IsOverflowDetectionOpen();
  GE_ASSERT_NOTNULL(davinci_model_->GetReusableStreamAllocator());
  // task num of follow stream can not exceed that of main stream
  const int32_t task_num = davinci_model_->GetTaskNumOfStream(logic_stream_id_);
  for (int64_t i = 0; i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    uint32_t stream_flags = static_cast<uint32_t>(RT_STREAM_PERSISTENT) | static_cast<uint32_t>(RT_STREAM_FORCE_COPY);
    if (isOverflowDetectionOpen) {
      stream_flags |= RT_STREAM_OVERFLOW;
    }
    davinci_model_->GetReusableStreamAllocator()->GetOrCreateRtStream(
        stream, davinci_model_->GetRuntimeModelId(), davinci_model_->Priority(), stream_flags, task_num);
    hccl_stream_list_.emplace_back(stream);
    davinci_model_->PushHcclStream(stream);

    // Create slave stream, inactive by default, activated by hccl
    GE_CHK_RT_RET(rtModelBindStream(davinci_model_->GetRtModelHandle(), stream,
                                     static_cast<uint32_t>(RT_MODEL_WAIT_ACTIVE_STREAM)));
    GELOGD("hccl_stream addr is=%p", stream);
    davinci_model_->SaveHcclFollowStream(main_stream_id, stream);
  }
  GELOGI("CreateStream success.");
  return SUCCESS;
}

Status HcclTaskInfo::InsertDumpOp(const std::string &dump_mode) {
  if (!davinci_model_->OpNeedDump(hccl_op_desc_->GetName())) {
    return SUCCESS;
  }
  GELOGI("Data Dump is on, dump op fo node: %s, type: %s, stream flag: %u.",
         hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str(), stream_flag_);
  auto hccl_dump_properties = davinci_model_->GetDumpProperties();
  DumpOp *dump_op = nullptr;
  if (dump_mode == kDumpInput) {
    if (hccl_dump_properties.GetDumpMode() == kDumpOutput) {
      return SUCCESS;
    }
    GELOGI("Insert input dump op fo node: %s, type: %s.",
           hccl_op_desc_->GetName().c_str(),
           hccl_op_desc_->GetType().c_str());
    hccl_dump_properties.ClearOpDebugFlag();
    hccl_dump_properties.SetDumpMode(kDumpInput);
    dump_op = &input_hccl_dump_;
  } else if (dump_mode == kDumpOutput) {
    if (hccl_dump_properties.GetDumpMode() == kDumpInput) {
      return SUCCESS;
    }
    GELOGI("Insert output dump op fo node: %s, type: %s.",
           hccl_op_desc_->GetName().c_str(),
           hccl_op_desc_->GetType().c_str());
    hccl_dump_properties.ClearOpDebugFlag();
    hccl_dump_properties.SetDumpMode(kDumpOutput);
    dump_op = &output_hccl_dump_;
  } else {
    return SUCCESS;
  }

  std::vector<uintptr_t> input_addrs;
  std::vector<uintptr_t> output_addrs;
  for (size_t i = 0UL; i < hccl_op_desc_->GetInputsSize(); i++) {
    input_addrs.push_back(static_cast<uintptr_t>(io_addrs_[i]));
  }
  for (size_t i = hccl_op_desc_->GetInputsSize();
      i < hccl_op_desc_->GetOutputsSize() + hccl_op_desc_->GetInputsSize(); i++) {
    output_addrs.push_back(static_cast<uintptr_t>(io_addrs_[i]));
  }
  dump_op->SetDumpInfo(hccl_dump_properties, hccl_op_desc_, input_addrs, output_addrs, stream_);
  if (davinci_model_->IsKnownNode()) {
    dump_op->SetLoopAddr(davinci_model_->GetGlobalStep(), 0U, 0U);
  } else {
    dump_op->SetLoopAddr(davinci_model_->GetGlobalStep(),
                         davinci_model_->GetLoopPerIter(),
                         davinci_model_->GetLoopCond());
  }
  dump_op->SetDynamicModelInfo(davinci_model_->GetDumpModelName(),
                               davinci_model_->GetOmName(),
                               davinci_model_->GetDumpModelId());
  return dump_op->LaunchDumpOp(false, (stream_flag_ & RT_STREAM_FORCE_COPY) == 0U);
}

Status HcclTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(hccl_op_desc_);
  GELOGI("HcclTaskInfo Distribute Start, op %s, type %s",
      hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
  if (ops_kernel_store_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Check param ops_kernel_store_ nullptr");
    GELOGE(INTERNAL_ERROR, "[Check][Param] ops kernel store is null.");
    return INTERNAL_ERROR;
  }
  const TaskProfGuarder prof_guarder(this);
  GETaskInfo ge_task;
  TransToGETaskInfo(ge_task);
  GE_ASSERT_SUCCESS(AssembleAttachedRtStream(ge_task));
  if (InsertDumpOp(kDumpInput) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Insert hccl input dump op fail");
    GELOGE(INTERNAL_ERROR, "Insert hccl input dump op fail");
    return INTERNAL_ERROR;
  }
  const auto result = ops_kernel_store_->LoadTask(ge_task);
  GE_CHK_BOOL_RET_STATUS((result == HCCL_SUCCESS), INTERNAL_ERROR, "call hccl op:%s(%s) load task fail",
      hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str());
  if (InsertDumpOp(kDumpOutput) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Insert hccl output dump op fail");
    GELOGE(INTERNAL_ERROR, "Insert hccl input dump op fail");
    return INTERNAL_ERROR;
  }
  if (ge_task.kernelHcclInfo.size() > kZeroIndex) {
    hccl_dump_infos_ = ge_task.kernelHcclInfo[kZeroIndex].hccl_dump_info;
  }

  is_support_redistribute_ = true;
  GELOGI("HcclTaskInfo %s(%s) Distribute Success, stream: %p.",
    hccl_op_desc_->GetNamePtr(), hccl_op_desc_->GetTypePtr(), stream_);
  return SUCCESS;
}

Status HcclTaskInfo::GetTypeSizeByDataType(const ge::DataType data_type, int64_t &type_size) const {
  type_size = static_cast<int64_t>(GetSizeByDataType(data_type));
  GE_CHECK_GE(type_size, kValidTypeSize);
  GELOGD("type size is %d", type_size);
  return SUCCESS;
}

void HcclTaskInfo::HcclWatcherModeProcess(const ModelTaskType task_type) {
  if (davinci_model_->OpNeedDumpOnWatcherModel(hccl_op_desc_->GetName())) {
    std::vector<uintptr_t> args;
    for (size_t i = 0U; i < io_addrs_.size(); i++) {
      args.emplace_back(static_cast<uintptr_t>(io_addrs_[i]));
    }
    FirstLevelAddressInfo first_level_address_info{true, args};

    davinci_model_->SaveDumpTask({0U, 0U, 0U, 0U}, hccl_op_desc_, static_cast<uintptr_t>(PtrToValue(args.data())),
                                 first_level_address_info, {}, task_type, stream_);
    GELOGI("Save hccl dump watcher op %s, op type: %s, input size: %zu, output size: %zu",
           hccl_op_desc_->GetName().c_str(), hccl_op_desc_->GetType().c_str(), hccl_op_desc_->GetInputsSize(),
           hccl_op_desc_->GetOutputsSize());
  }

  if (davinci_model_->OpNoNeedDumpOnWatcherModel(hccl_op_desc_->GetName())) {
    GELOGW("Set hccl op in DUMP_LAYER_OP_MODEL is not support", hccl_op_desc_->GetName().c_str());
  }
  return;
}

void HcclTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  const auto &hccl_def = task_def.kernel_hccl();
  const uint32_t op_index = hccl_def.op_index();
  GELOGI("HcclTaskInfo PostProcess, op_index is: %u", op_index);
  // Get HCCL op
  const auto op_desc = davinci_model_->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  ge::DataType src_data_type = ge::DT_FLOAT;
  if (op_desc->GetType() == HCOMRECEIVE) {
    (void)ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, src_data_type);
  } else {
    const auto input_desc_ptr = op_desc->GetInputDescPtr(0U);
    GE_CHECK_NOTNULL_JUST_RETURN(input_desc_ptr);
    src_data_type = input_desc_ptr->GetDataType();
  }
  int64_t type_size = 0;
  if (GetTypeSizeByDataType(src_data_type, type_size) != SUCCESS) {
    return;
  }
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  if (davinci_model_->GetOpDugReg() || davinci_model_->OpNeedDump(op_desc->GetName())
    || davinci_model_->OpNeedDumpOnWatcherModel(op_desc->GetName())) {
    for (size_t i = 0U; i < hccl_dump_infos_.size(); i++) {
      const auto sdma_op_desc = MakeShared<OpDesc>(op_desc->GetName(), op_desc->GetType());
      GE_CHECK_NOTNULL_JUST_RETURN(sdma_op_desc);

      // sdma dump format always are FORMAT_ND
      const std::vector<int64_t> input_dims = { static_cast<int64_t>(hccl_dump_infos_[i].input_size) / type_size };
      GeTensorDesc input_desc(GeShape(input_dims), FORMAT_ND, src_data_type);
      TensorUtils::SetSize(input_desc, static_cast<int64_t>(hccl_dump_infos_[i].input_size));
      (void)sdma_op_desc->AddInputDesc(input_desc);
      const std::vector<int64_t> output_dims = { static_cast<int64_t>(hccl_dump_infos_[i].output_size) / type_size };
      GeTensorDesc output_desc(GeShape(output_dims), FORMAT_ND, src_data_type);
      TensorUtils::SetSize(output_desc, static_cast<int64_t>(hccl_dump_infos_[i].output_size));
      (void)sdma_op_desc->AddOutputDesc(output_desc);

      const std::vector<uintptr_t> args{ static_cast<uintptr_t>(PtrToValue(hccl_dump_infos_[i].input_addr)),
                                         static_cast<uintptr_t>(PtrToValue(hccl_dump_infos_[i].output_addr)) };
      FirstLevelAddressInfo first_level_address_info{true, args};

      davinci_model_->SaveDumpTask({hccl_dump_infos_[i].task_id, hccl_dump_infos_[i].stream_id, 0U, 0U}, sdma_op_desc,
                                   static_cast<uintptr_t>(PtrToValue(args.data())), first_level_address_info, {},
                                   task_type, stream_);
      GELOGD("Save dump op %s, op type: %s, task id: %u, stream id: %u, input size: %zu, output size: %zu",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), hccl_dump_infos_[i].task_id,
             hccl_dump_infos_[i].stream_id, hccl_dump_infos_[i].input_size, hccl_dump_infos_[i].output_size);
    }
  }

  HcclWatcherModeProcess(task_type);

  davinci_model_->SaveProfilingTaskDescInfo(op_desc, *this, task_def);
}

Status HcclTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                       TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  const auto &hccl_def = task_def.kernel_hccl();
  const uint32_t op_index = hccl_def.op_index();
  GELOGI("HcclTaskInfo Init, op_index is: %u", op_index);
  // Get HCCL op
  const auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  // Only need the number of addr to allocate args memory
  const auto input_size = op_desc->GetInputsSize();
  const auto output_size = op_desc->GetOutputsSize();
  const auto workspace_size = op_desc->GetWorkspaceBytes().size();
  const uint32_t args_size = static_cast<uint32_t>(sizeof(void *) * (input_size + output_size + workspace_size));
  args_mem_type_ =
    davinci_model->IsArgsUpdateByDeviceAicpu() ? RT_MEMORY_HBM : rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, args_size);
  GELOGI("memory_type: %u", args_mem_type_);
  pls_ = ((args_mem_type_ & RT_MEMORY_TS) == 0U)
      ? ArgsPlacement::kArgsPlacementHbm : ArgsPlacement::kArgsPlacementTs;
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size), pls_});
  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  input_data_addrs_ = ModelUtils::GetInputAddrs(rts_param, op_desc, input_mem_types_);
  output_data_addrs_ = ModelUtils::GetOutputAddrs(rts_param, op_desc, output_mem_types_);
  workspace_addrs_ = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc, workspace_mem_types_);
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({PtrToValue(input_data_addrs_[i]), input_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({PtrToValue(output_data_addrs_[i]), output_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back({PtrToValue(workspace_addrs_[i]), workspace_mem_types_[i],
                                                    true, {0}});
  }

  return SUCCESS;
}

void HcclTaskInfo::UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs) {
  // todo: model args manager功能适配完毕后, 此处新增input_data_addrs_和iow_addrs.input_logic_addrs相等的校验
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] = (iow_addrs.input_logic_addrs.empty())
        ? input_data_addrs_[i] : ValueToPtr(iow_addrs.input_logic_addrs[i].logic_addr);
    input_mem_types_[i] = (iow_addrs.input_logic_addrs.empty())
        ? input_mem_types_[i] : iow_addrs.input_logic_addrs[i].memory_type;
    is_refresh_addr_op_ |= ModelUtils::IsSuppoprtAddrRefreshable(input_mem_types_[i]);
  }

  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] = (iow_addrs.output_logic_addrs.empty())
        ? output_data_addrs_[i] : ValueToPtr(iow_addrs.output_logic_addrs[i].logic_addr);
    output_mem_types_[i] = (iow_addrs.output_logic_addrs.empty())
        ? output_mem_types_[i] : iow_addrs.output_logic_addrs[i].memory_type;
    is_refresh_addr_op_ |= ModelUtils::IsSuppoprtAddrRefreshable(output_mem_types_[i]);
  }

  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    workspace_addrs_[i] = (iow_addrs.workspace_logic_addrs.empty())
        ? workspace_addrs_[i] : ValueToPtr(iow_addrs.workspace_logic_addrs[i].logic_addr);
    workspace_mem_types_[i] = (iow_addrs.workspace_logic_addrs.empty())
        ? workspace_mem_types_[i] : iow_addrs.workspace_logic_addrs[i].memory_type;
    is_refresh_addr_op_ |= ModelUtils::IsSuppoprtAddrRefreshable(workspace_mem_types_[i]);
  }

  std::vector<uint64_t> temp_input_addrs = VPtrToValue(input_data_addrs_);
  (void)io_addrs_.insert(io_addrs_.cend(), temp_input_addrs.cbegin(), temp_input_addrs.cend());
  std::vector<uint64_t> temp_output_addrs = VPtrToValue(output_data_addrs_);
  (void)io_addrs_.insert(io_addrs_.cend(), temp_output_addrs.cbegin(), temp_output_addrs.cend());
  std::vector<uint64_t> temp_workspace_addrs = VPtrToValue(workspace_addrs_);
  (void)io_addrs_.insert(io_addrs_.cend(), temp_workspace_addrs.cbegin(), temp_workspace_addrs.cend());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), input_mem_types_.cbegin(), input_mem_types_.cend());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), output_mem_types_.cbegin(), output_mem_types_.cend());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), workspace_mem_types_.cbegin(),
                                  workspace_mem_types_.cend());
  GELOGI("HcclTaskInfo::is refresh addr op:%d.", is_refresh_addr_op_);
}

Status HcclTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                                    void *const host_args, const size_t host_args_max_len) {
  GELOGI("HcclTaskInfo::UpdateArgs in.");
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_max_len));
  if (davinci_model_->OpNeedDump(hccl_op_desc_->GetName())) {
    std::vector<uintptr_t> input_addrs;
    std::vector<uintptr_t> output_addrs;
    uint64_t *const host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
    for (size_t i = 0UL; i < hccl_op_desc_->GetInputsSize(); i++) {
      input_addrs.push_back(static_cast<uintptr_t>(host_args_tmp[i]));
    }
    for (size_t i = hccl_op_desc_->GetInputsSize();
         i < hccl_op_desc_->GetOutputsSize() + hccl_op_desc_->GetInputsSize(); i++) {
      output_addrs.push_back(static_cast<uintptr_t>(host_args_tmp[i]));
    }
    GE_CHK_STATUS_RET(input_hccl_dump_.UpdateAddrs(input_addrs, {}),
                      "[Update][HcclDumpAddrs] fail! op:%s", hccl_op_desc_->GetName().c_str());
    GE_CHK_STATUS_RET(output_hccl_dump_.UpdateAddrs({}, output_addrs),
                      "[Update][HcclDumpAddrs] fail! op:%s", hccl_op_desc_->GetName().c_str());
  }
  GELOGI("HcclTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status HcclTaskInfo::UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) {
  if (davinci_model_->OpNeedDump(hccl_op_desc_->GetName())) {
    GELOGI("HcclTaskInfo::UpdateDumpInfos in.");
    std::vector<uintptr_t> input_addrs;
    std::vector<uintptr_t> output_addrs;
    const auto input_size = hccl_op_desc_->GetInputsSize();
    const auto output_size = hccl_op_desc_->GetOutputsSize();
    GE_ASSERT_TRUE(host_args_max_len >= (sizeof(uint64_t) * (input_size + output_size)));
    uint64_t *const host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
    for (size_t i = 0UL; i < input_size; i++) {
      input_addrs.push_back(static_cast<uintptr_t>(host_args_tmp[i]));
    }
    for (size_t i = input_size; i < (output_size + input_size); i++) {
      output_addrs.push_back(static_cast<uintptr_t>(host_args_tmp[i]));
    }
    GE_CHK_STATUS_RET(input_hccl_dump_.UpdateAddrs(input_addrs, {}),
                      "[Update][HcclDumpAddrs] fail! op:%s", hccl_op_desc_->GetName().c_str());
    GE_CHK_STATUS_RET(output_hccl_dump_.UpdateAddrs({}, output_addrs),
                      "[Update][HcclDumpAddrs] fail! op:%s", hccl_op_desc_->GetName().c_str());
    GELOGI("HcclTaskInfo::UpdateDumpInfos success.");
  }

  return SUCCESS;
}

Status HcclTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, pls_);
  return SUCCESS;
}

bool HcclTaskInfo::IsReduceOp(const std::string &hccl_type) const {
  return ((hccl_type == HCOMALLREDUCE) || (hccl_type == HCOMREDUCESCATTER) || (hccl_type == HCOMREDUCESCATTERV) ||
          (hccl_type == HVDCALLBACKALLREDUCE) || (hccl_type == HCOMREDUCE));
}

bool HcclTaskInfo::UpdateOutputAddr(const std::string &hccl_type) const {
  return ((hccl_type == HCOMALLGATHER) || (hccl_type == HCOMRECEIVE) || (hccl_type == HVDCALLBACKALLGATHER) ||
          (hccl_type == HCOMALLTOALLV) || (hccl_type == HCOMALLTOALLVC)  || (hccl_type == HCOMALLTOALL) ||
          (hccl_type == HCOMALLGATHERV) || (hccl_type == HCOMGATHER));
}

Status HcclTaskInfo::SetAddrs(const std::shared_ptr<OpDesc> &op_desc,
                              std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_STATUS_RET(HcomOmeUtil::CheckKernelHcclInfo(op_desc, kernel_hccl_infos),
                    "[Check][Param] HcomOmeUtil:: the number of GETaskKernelHcclInfo is invalid, node:%s(%s).",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
  GELOGI("Set hccl task input output address, node[%s], type[%s] kernel_hccl_infos.size[%zu].",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }

  HcclReduceOp op_type = HCCL_REDUCE_SUM;
  GE_CHECK_NOTNULL(davinci_model_);
  GELOGI("Calc opType[%s] input address before. Node name[%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  void *input_data_addr = nullptr;
  void *output_data_addr = nullptr;
  // initialize every kernel_hccl_info inputDataAddr
  for (size_t i = 0U; i < kernel_hccl_infos.size(); i++) {
    const std::string hccl_type = kernel_hccl_infos[i].hccl_type;
    if (IsFeatureBaseRefreshable(davinci_model_)) {
      input_data_addr = ValueToPtr(PtrToValue(args_) + (i * sizeof(uint64_t)));
      output_data_addr = ValueToPtr(PtrToValue(args_) + ((op_desc->GetInputsSize() + i) * sizeof(uint64_t)));
      GELOGI("Hccl task info known input addr %p, output addr %p.", input_data_addr, output_data_addr);
    } else {
      input_data_addr = input_data_addrs_.empty() ? nullptr : input_data_addrs_[i];
      output_data_addr = output_data_addrs_.empty() ? nullptr : output_data_addrs_[i];
    }
    kernel_hccl_infos[i].inputDataAddr = input_data_addr;
    if (UpdateOutputAddr(hccl_type)) {
      kernel_hccl_infos[i].outputDataAddr = output_data_addr;
    } else if (IsReduceOp(hccl_type)) {
      GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclOperationType(op_desc, op_type),
                        "[Get][HcomOperationType] fail! op:%s", op_desc->GetName().c_str());
      kernel_hccl_infos[i].outputDataAddr = output_data_addr;
      kernel_hccl_infos[i].opType = op_type;
    } else {
      // do nothing
    }

    if (!IsFeatureBaseRefreshable(davinci_model_)) {
      kernel_hccl_infos[i].inputDataAddrs.assign(input_data_addrs_.begin(), input_data_addrs_.end());
      kernel_hccl_infos[i].outputDataAddrs.assign(output_data_addrs_.begin(), output_data_addrs_.end());
    }

    if (!support_zero_copy_) {
      GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(input_data_addr));
      GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(output_data_addr));
    }
  }
  return SetZeroCopyAddrs(op_desc, kernel_hccl_infos);
}

void HcclTaskInfo::TransToGETaskInfo(GETaskInfo &ge_task) const {
  ge_task.id = id_;
  ge_task.type = static_cast<uint16_t>(ModelTaskType::MODEL_TASK_HCCL);
  ge_task.stream = stream_;
  ge_task.kernelHcclInfo = kernel_hccl_infos_;
  ge_task.privateDef = private_def_;
  ge_task.privateDefLen = private_def_len_;
  ge_task.opsKernelStorePtr = ops_kernel_store_;
  ge_task.needRefresh = IsFeatureBaseRefreshable(davinci_model_);
  GELOGI("Hccl task id:%zu, need refresh:%d", ge_task.id, ge_task.needRefresh);
  for (size_t i = 0U; i < ge_task.kernelHcclInfo.size(); i++) {
    ge_task.kernelHcclInfo[i].hcclStreamList = hccl_stream_list_;
  }
}

void HcclTaskInfo::GetPrivateDefByTaskDef(const OpDescPtr &op_desc, const domi::TaskDef &task) {
  // Get privateDef and opsKernelStorePtr from taskDef and save them in taskInfo
  GELOGI("get custom info in modelTaskDef.");
  ops_kernel_store_ = op_desc->TryGetExtAttr<OpsKernelExecutor *>("OpsKernelInfoStorePtr", nullptr);
  if ((ops_kernel_store_ == nullptr) &&
      (OpsKernelExecutorManager::GetInstance().GetExecutor(op_desc->GetOpKernelLibName(), ops_kernel_store_) !=
          SUCCESS)) {
    return;
  }
  const std::string &private_def_temp = task.private_def();
  if ((!private_def_temp.empty()) && (private_def_temp.size() <= static_cast<size_t>(UINT32_MAX))) {
    private_def_len_ = static_cast<uint32_t>(private_def_temp.size());
    GE_CHK_RT_EXEC(rtMallocHost(&private_def_, static_cast<uint64_t>(private_def_len_), GE_MODULE_NAME_U16), return);
    GE_CHK_RT_EXEC(rtMemcpy(private_def_, static_cast<uint64_t>(private_def_len_), task.private_def().c_str(),
                   static_cast<uint64_t>(private_def_len_), RT_MEMCPY_HOST_TO_HOST), return);
    GELOGI("The first address of the custom info, privateDef=%p.", private_def_);
  }
}

void HcclTaskInfo::CreateKernelHcclInfo(const ConstOpDescPtr &op_desc) {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  if (HcomOmeUtil::IsHCOMOp(op_desc->GetType())) {
    GETaskKernelHcclInfo kernel_hccl_info;
    kernel_hccl_infos_.emplace_back(kernel_hccl_info);
  } else if (HcomOmeUtil::IsHorovodOp(op_desc->GetType())) {
    // Horovod wait do not have any input, but create a GETaskKernelHcclInfo to record hccl_type.
    // Other Operator need to check that the number of GETaskKernelHcclInfo must equals to number of inputs
    if (op_desc->GetType() == HVDWAIT) {
      GETaskKernelHcclInfo kernel_hccl_info;
      kernel_hccl_infos_.emplace_back(kernel_hccl_info);
      return;
    }
    for (size_t i = 0U; i < op_desc->GetInputsSize(); i++) {
      GETaskKernelHcclInfo kernel_hccl_info;
      kernel_hccl_infos_.emplace_back(kernel_hccl_info);
    }
  } else {
    // do nothing
  }
}

Status HcclTaskInfo::SetWorkspace(const std::shared_ptr<OpDesc> &op_desc,
                                  std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(davinci_model_);
  GELOGI("SetWorkspace Node[%s] opType[%s] set workspace.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  uint64_t workspace_mem_size = 0U;
  void *workspace_addr = nullptr;
  const auto workspace_bytes = op_desc->GetWorkspaceBytes();
  if (!workspace_bytes.empty()) {
    const uint64_t workspace_mem_size_tmp = static_cast<uint64_t>(workspace_bytes[0U]);
    GELOGI("hccl need workSpaceMemSize=%" PRIu64, workspace_mem_size_tmp);
    if (workspace_mem_size_tmp != 0U) {
      workspace_mem_size = workspace_mem_size_tmp;
      if (IsFeatureBaseRefreshable(davinci_model_)) {
        workspace_addr = ValueToPtr(PtrToValue(args_) +
                                    ((op_desc->GetInputsSize() + op_desc->GetOutputsSize()) * sizeof(uint64_t)));
      } else {
        workspace_addr = workspace_addrs_.empty() ? nullptr : workspace_addrs_[0U];
      }
    }
  }
  for (size_t i = 0U; i < kernel_hccl_infos.size(); i++) {
    kernel_hccl_infos[i].workSpaceMemSize = workspace_mem_size;
    kernel_hccl_infos[i].workSpaceAddr = workspace_addr;
  }
  return SUCCESS;
}

Status HcclTaskInfo::SetOverflowAddrs(const std::shared_ptr<OpDesc> &op_desc,
                                      std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  for (size_t i = 0U; i < kernel_hccl_infos.size(); i++) {
    void *const globalworkspace_overflow_addr = davinci_model_->GetOverflowAddr();
    if ((globalworkspace_overflow_addr != nullptr) && (AttrUtils::HasAttr(op_desc, GLOBALWORKSPACE_TYPE))) {
      global_workspace_addr_.emplace_back(globalworkspace_overflow_addr);
      kernel_hccl_infos[i].global_workspace_addr = global_workspace_addr_;
    }
  }
  return SUCCESS;
}

Status HcclTaskInfo::Release() {
  GELOGI("HcclTaskInfo unload Start. begin to call function unloadTask in hccl.");
  if (ops_kernel_store_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Check param ops_kernel_store_ nullptr");
    GELOGE(INTERNAL_ERROR, "[Check][Param] ops kernel store is null.");
    return INTERNAL_ERROR;
  }
  GETaskInfo ge_task;
  TransToGETaskInfo(ge_task);
  const auto result = ops_kernel_store_->UnloadTask(ge_task);
  if (result != HCCL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call ops_kernel_info_store unloadTask fail");
    GELOGE(INTERNAL_ERROR, "[UnLoad][Task] fail, return ret:%u", result);
    return INTERNAL_ERROR;
  }
  GELOGI("HcclTaskInfo unload Success.");
  return SUCCESS;
}

Status HcclTaskInfo::InitZeroCopyInfos(const OpDescPtr &op_desc, const domi::KernelHcclDef &hccl_def) {
  if (!hccl_def.input_zero_copy_flag().empty()) {
    GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(hccl_def.input_zero_copy_flag_size()) == op_desc->GetInputsSize(),
                           FAILED, "input_zero_copy_flag_size %d is not equal with op input size %zu",
                           hccl_def.input_zero_copy_flag_size(), op_desc->GetInputsSize());
    for (int32_t idx = 0; idx < hccl_def.input_zero_copy_flag_size(); ++idx) {
      int32_t input_zero_copy_flag = hccl_def.input_zero_copy_flag(idx);
      if (input_zero_copy_flag == kSupportZeroCopy) {
        support_zero_copy_ = true;
      }
      GELOGD("Get HCCL input zero copy flag: %d", input_zero_copy_flag);
      input_zero_copy_flag_.emplace_back(input_zero_copy_flag);
    }
  }

  if (!hccl_def.output_zero_copy_flag().empty()) {
    GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(hccl_def.output_zero_copy_flag_size()) == op_desc->GetOutputsSize(),
                           FAILED, "output_zero_copy_flag_size %d is not equal with op output size %zu",
                           hccl_def.output_zero_copy_flag_size(), op_desc->GetOutputsSize());
    for (int32_t idx = 0; idx < hccl_def.output_zero_copy_flag_size(); ++idx) {
      int32_t output_zero_copy_flag = hccl_def.output_zero_copy_flag(idx);
      if (output_zero_copy_flag == kSupportZeroCopy) {
        support_zero_copy_ = true;
      }
      GELOGD("Get HCCL output zero copy flag: %d", output_zero_copy_flag);
      output_zero_copy_flag_.emplace_back(output_zero_copy_flag);
    }
  }
  return SUCCESS;
}

Status HcclTaskInfo::SetZeroCopyAddrs(const std::shared_ptr<OpDesc> &op_desc,
                                      std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  if (!support_zero_copy_) {
    return SUCCESS;
  }

  GELOGI("Hccl task support zero copy: op:%s", op_desc->GetName().c_str());
  const auto input_size = op_desc->GetInputsSize();
  const auto output_size = op_desc->GetOutputsSize();
  const auto input_addrs_value = VPtrToValue(input_data_addrs_);
  const auto output_addrs_value = VPtrToValue(output_data_addrs_);
  if (IsFeatureBaseRefreshable(davinci_model_)) {
    kernel_hccl_infos[0].inputDataAddrs.assign(input_data_addrs_.begin(), input_data_addrs_.end());
    kernel_hccl_infos[0].outputDataAddrs.assign(output_data_addrs_.begin(), output_data_addrs_.end());
  }
  if (!input_zero_copy_flag_.empty()) {
    kernel_hccl_infos[0].inputZeroCopyFlags.assign(input_zero_copy_flag_.begin(), input_zero_copy_flag_.end());
    davinci_model_->SetZeroCopyAddr(op_desc, input_addrs_value, input_addrs_value.data(),
                                    static_cast<uintptr_t>(PtrToValue(args_)), input_size * kAddressLen, 0U, {});
    if (input_zero_copy_flag_[0] == kSupportZeroCopy) {
      kernel_hccl_infos[0].inputDataAddr = args_;
    }
    for (size_t i = 0U; i < input_zero_copy_flag_.size(); i++) {
      if (input_zero_copy_flag_[i] == kSupportZeroCopy) {
        kernel_hccl_infos[0].inputDataAddrs[i] = ValueToPtr(PtrToValue(args_) + (i * kAddressLen));
      } else {
        GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(kernel_hccl_infos[0].inputDataAddrs[i]));
      }
    }
  }

  if (!output_zero_copy_flag_.empty()) {
    kernel_hccl_infos[0].outputZeroCopyFlags.assign(output_zero_copy_flag_.begin(), output_zero_copy_flag_.end());
    davinci_model_->SetZeroCopyAddr(op_desc, output_addrs_value, output_addrs_value.data(),
                                    static_cast<uintptr_t>(PtrToValue(args_) + (input_size * kAddressLen)),
                                    output_size * kAddressLen, 0U, {});
    if (output_zero_copy_flag_[0] == kSupportZeroCopy) {
      kernel_hccl_infos[0].outputDataAddr = ValueToPtr(PtrToValue(args_) + (input_size * kAddressLen));
    }
    for (size_t i = 0U; i < output_zero_copy_flag_.size(); i++) {
      if (output_zero_copy_flag_[i] == kSupportZeroCopy) {
        kernel_hccl_infos[0].outputDataAddrs[i] = ValueToPtr(PtrToValue(args_) + ((input_size + i) * kAddressLen));
      } else {
        GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(kernel_hccl_infos[0].outputDataAddrs[i]));
      }
    }
  }
  return SUCCESS;
}

int64_t HcclTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const auto &hccl_def = task_def.kernel_hccl();
  return static_cast<int64_t>(hccl_def.op_index());
}

bool HcclTaskInfo::IsFeatureBaseRefreshable(const ge::DavinciModel *const davinci_model) const {
  return !davinci_model->IsStaticAddrFixed() && davinci_model->IsFeatureBaseRefreshable() && is_refresh_addr_op_;
}

Status HcclTaskInfo::AssembleAttachedRtStream(GETaskInfo &ge_task_info) const {
  GE_CHECK_NOTNULL(hccl_op_desc_);
  GE_CHECK_NOTNULL(davinci_model_);
  const std::vector<rtStream_t> &stream_list = davinci_model_->GetStreamList();
  const auto stream_ids = hccl_op_desc_->GetAttachedStreamIds();
  for (const auto stream_id : stream_ids) {
    if (stream_id < 0) {
      continue;
    }
    GE_ASSERT(static_cast<size_t>(stream_id) < stream_list.size(), "[%s]'s attached stream_id [%" PRId64 "] is invalid.",
              hccl_op_desc_->GetNamePtr(), stream_id);
    ge_task_info.rt_attached_streams.push_back(stream_list[stream_id]);
    GELOGI("[%s] get rt stream [%p] by logic stream [%" PRId64 "] successfully", hccl_op_desc_->GetNamePtr(),
           stream_list[stream_id], stream_id);
  }
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_HCCL, HcclTaskInfo);
}  // namespace ge
