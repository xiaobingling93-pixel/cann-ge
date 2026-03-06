/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/dsa_task_info.h"

#include "common/runtime_api_wrapper.h"
#include "framework/common/ge_types.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace ge {
namespace {
constexpr uint32_t kMask32Bits = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111
constexpr int64_t  kSqeArgsLen = 40L;
}

Status DSATaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                         const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                         const IowAddrs &iow_addrs) {
  GELOGI("DSATaskInfo Init Start.");
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  const domi::DSATaskDef &dsa_task = task_def.dsa_task();
  op_index_ = dsa_task.op_index();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(op_index_);
  GE_CHECK_NOTNULL(op_desc);
  op_desc_ = op_desc;
  InitDsaDumpInfo(op_desc);
  hbm_workspace_args_ = args[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)].dev_addr;

  GE_ASSERT_TRUE((iow_addrs.input_logic_addrs.size() == input_data_addrs_.size()),
                 "[Check][Param] Op:%s(%s) input logic addrs list size:%zu != input data addr list size:%zu",
                 op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                 iow_addrs.input_logic_addrs.size(), input_data_addrs_.size());

  GE_ASSERT_TRUE((iow_addrs.output_logic_addrs.size() == output_data_addrs_.size()),
                 "[Check][Param] Op:%s(%s) output logic addrs list size:%zu != output data addr list size:%zu",
                 op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                 iow_addrs.output_logic_addrs.size(), output_data_addrs_.size());

  GE_ASSERT_TRUE((iow_addrs.workspace_logic_addrs.size() == workspace_data_addrs_.size()),
                 "[Check][Param] Op:%s(%s) workspace logic addrs list size:%zu != workspace data addr list size:%zu",
                 op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                 iow_addrs.workspace_logic_addrs.size(), workspace_data_addrs_.size());

  GetAddrs(iow_addrs);
  GE_ASSERT_SUCCESS(InitSqe(op_desc, dsa_task));

  (void)workspace_io_addrs_.insert(workspace_io_addrs_.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
  (void)workspace_io_addrs_.insert(workspace_io_addrs_.cend(), output_data_addrs_.cbegin(), output_data_addrs_.cend());
  (void)hbm_args_refresh_flags_.insert(hbm_args_refresh_flags_.cend(), input_addr_refresh_.cbegin(),
      input_addr_refresh_.cend());
  (void)hbm_args_refresh_flags_.insert(hbm_args_refresh_flags_.cend(), output_addr_refresh_.cbegin(),
      output_addr_refresh_.cend());

  if (support_refresh_) {
    // args updater init
    GE_ASSERT_SUCCESS(sqe_args_updater_.Init(davinci_model_->GetLogicalMemAllocation(), sqe_io_addrs_,
        sqe_args_refresh_flags_, {op_desc->GetName(), op_desc->GetType()}));
    GE_ASSERT_SUCCESS(workspace_args_updater_.Init(davinci_model_->GetLogicalMemAllocation(), workspace_io_addrs_,
        hbm_args_refresh_flags_, {op_desc->GetName(), op_desc->GetType()}));
  }

  GELOGI("DSATaskInfo %s Init Success, hbm_workspace_args is 0x%llx, logic stream id: %u, stream: %p.",
    op_desc->GetNamePtr(), hbm_workspace_args_, task_def.stream_id(), stream_);
  return SUCCESS;
}

Status DSATaskInfo::InitWorkspace(const OpDescPtr &op_desc, const domi::DSATaskDef &dsa_task) {
  hbm_args_refresh_flags_.clear();
  workspace_io_addrs_.clear();
  if (dsa_task.input1_value_or_ptr() == kDSASetInputAddr) {
    hbm_args_refresh_flags_.push_back(input_addr_refresh_[2U]);
    workspace_io_addrs_.push_back(input_data_addrs_[2U]);
    if ((input_data_addrs_.size() == kDSAStateInputAddrSize) ||
    ((input_data_addrs_.size() == kDSAArgsInputAddrSize) && (workspace_data_addrs_.size() == kDSAWorkspaceAddrSize))) {
      hbm_args_refresh_flags_.push_back(input_addr_refresh_[3U]);
      workspace_io_addrs_.push_back(input_data_addrs_[3U]);
    }
  } else {
    uint64_t input_data0 = 0UL;
    uint64_t input_data1 = 0UL;
    const std::string &input1 = dsa_task.args().input1_value_or_addr();
    auto mem_ret = memcpy_s(&input_data0, sizeof(uint64_t), input1.c_str(), input1.size());
    GE_ASSERT_EOK(mem_ret, "dsa input data memcpy failed.");
    if ((input_data_addrs_.size() == kDSAStateInputAddrSize) ||
    ((input_data_addrs_.size() == kDSAArgsInputAddrSize) && (workspace_data_addrs_.size() == kDSAWorkspaceAddrSize))) {
      const std::string &input2 = dsa_task.args().input2_value_or_addr();
      mem_ret = memcpy_s(&input_data1, sizeof(uint64_t), input2.c_str(), input2.size());
      GE_ASSERT_EOK(mem_ret, "dsa input data memcpy failed.");
    }
    workspace_io_addrs_.push_back(input_data0);
    workspace_io_addrs_.push_back(input_data1);
    hbm_args_refresh_flags_.resize(2U, 0U);
  }

  // todo: hbm_args要替换成init接口传过来的args地址
  const uint64_t hbm_args = (hbm_workspace_args_ == 0UL)
      ? workspace_data_addrs_[workspace_data_addrs_.size() - 1U] : hbm_workspace_args_;
  const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
  GE_CHECK_GE(workspace_size.size(), workspace_data_addrs_.size());
  uint64_t dev_size = static_cast<uint64_t>(workspace_size[workspace_data_addrs_.size() - 1U]);
  GE_ASSERT_TRUE(static_cast<size_t>(dev_size) >= (sizeof(uint64_t) * workspace_io_addrs_.size()));

  // todo: 后面修改成静态图不可刷新场景, 在此拷贝, 采用model; 注意不支持刷新的也需要在此拷贝
  if ((!davinci_model_->IsFeatureBaseRefreshable()) || (!support_refresh_)) {
    GE_CHK_RT_RET(rtMemcpy(ValueToPtr(hbm_args), dev_size, workspace_io_addrs_.data(),
                           sizeof(uint64_t) * workspace_io_addrs_.size(), RT_MEMCPY_HOST_TO_DEVICE));
  }

  dev_size = static_cast<uint64_t>(MemSizeAlign(static_cast<size_t>(dev_size),
      static_cast<uint32_t>(sizeof(uint64_t))));
  workspace_io_addrs_.resize((dev_size / sizeof(uint64_t)), 0UL);
  hbm_args_refresh_flags_.resize(dev_size / sizeof(uint64_t), 0U);
  dump_args_ = ((hbm_workspace_args_ != 0UL) && (dump_flag_ == RT_KERNEL_DUMPFLAG)) ? (hbm_args + dev_size) : 0UL;

  GELOGI("opType[%s] Node name[%s], hbm_args addr 0x%llx, dump args addr 0x%llx, dev_size 0x%llx.",
      op_desc->GetType().c_str(), op_desc->GetName().c_str(), hbm_args, dump_args_, dev_size);

  return SUCCESS;
}

Status DSATaskInfo::InitSqe(const OpDescPtr &op_desc, const domi::DSATaskDef &dsa_task) {
  dsa_sqe_.sqeHeader.type = static_cast<uint8_t>(dsa_task.sqe_type());
  dsa_sqe_.start = dsa_task.start();
  dsa_sqe_.functionType = dsa_task.distribution_type();
  dsa_sqe_.dataType = dsa_task.data_type();
  dsa_sqe_.algoType = dsa_task.alg_type();
  dsa_sqe_.paramVldBitmap = dsa_task.input_vld();
  dsa_sqe_.paramAddrValBitmap = dsa_task.input_value_addr_flag();
  dsa_sqe_.kernelCredit = 100U;

  // 说明：以下几个地址封装顺序不能变更，封装顺序需要和字段在dsa sqe的格式顺序一致
  // 1.dsaCfgResultAddr
  const uint64_t dev_output_addr = output_data_addrs_[0U];
  dsa_sqe_.dsaCfgResultAddrLow = static_cast<uint32_t>(dev_output_addr & kMask32Bits);
  dsa_sqe_.dsaCfgResultAddrHigh = static_cast<uint32_t>(dev_output_addr >> k32Bits);
  sqe_args_refresh_flags_.push_back(output_addr_refresh_[0U]);
  sqe_io_addrs_.push_back(dev_output_addr);

  // 2.dsaCfgStateAddr
  if (workspace_data_addrs_.size() == kDSAWorkspaceAddrSize) {
    const uint64_t workspace_philox_count_addr = workspace_data_addrs_[0U];
    sqe_args_refresh_flags_.push_back(workspace_addr_refresh_[0U]);
    stateful_workspace_idx_ = sqe_io_addrs_.size();
    sqe_io_addrs_.push_back(workspace_philox_count_addr);
  } else {
    sqe_args_refresh_flags_.push_back(input_addr_refresh_[input_addr_refresh_.size() - 1U]);
    sqe_io_addrs_.push_back(input_data_addrs_[input_data_addrs_.size() - 1U]);
  }
  dsa_sqe_.dsaCfgStateAddrLow = static_cast<uint32_t>(sqe_io_addrs_[sqe_io_addrs_.size() - 1U] & kMask32Bits);
  dsa_sqe_.dsaCfgStateAddrHigh = static_cast<uint32_t>(sqe_io_addrs_[sqe_io_addrs_.size() - 1U] >> k32Bits);

  // 3.dsaCfgParamAddr指向一片连续的内存，内存中存放的可能是地址，也可能是指，由param_addr_val_bitmap指定
  GE_ASSERT_SUCCESS(InitWorkspace(op_desc, dsa_task));
  const uint64_t workspace_input_addr = workspace_data_addrs_[workspace_data_addrs_.size() - 1U];
  const uint64_t cfg_param_addr = (hbm_workspace_args_ == 0UL) ? workspace_input_addr : hbm_workspace_args_;
  sqe_args_refresh_flags_.push_back((hbm_workspace_args_ == 0UL)
      ? workspace_addr_refresh_[workspace_addr_refresh_.size() - 1U] : 0U);
  sqe_io_addrs_.push_back(cfg_param_addr);
  dsa_sqe_.dsaCfgParamAddrLow = static_cast<uint32_t>(cfg_param_addr & kMask32Bits);
  dsa_sqe_.dsaCfgParamAddrHigh = static_cast<uint32_t>(cfg_param_addr >> k32Bits);

  // 4.dsaCfgSeed
  const uint64_t seed_value_or_addr = (dsa_task.seed_value_or_ptr() == kDSASetInputAddr) ? input_data_addrs_[1U] :
                                      *(PtrToPtr<char_t, uint64_t>(dsa_task.args().seed_value_or_addr().c_str()));
  sqe_args_refresh_flags_.push_back((dsa_task.seed_value_or_ptr() == kDSASetInputAddr) ? input_addr_refresh_[1U] : 0U);
  sqe_io_addrs_.push_back(seed_value_or_addr);
  dsa_sqe_.dsaCfgSeedLow = static_cast<uint32_t>(seed_value_or_addr & kMask32Bits);
  dsa_sqe_.dsaCfgSeedHigh = static_cast<uint32_t>(seed_value_or_addr >> k32Bits);

  // 5.dsaCfgNumber
  const uint64_t random_count_value_or_addr = (dsa_task.random_count_value_or_ptr() == kDSASetInputAddr) ?
      input_data_addrs_[0U] : *(PtrToPtr<char_t, uint64_t>(dsa_task.args().random_count_value_or_addr().c_str()));

  sqe_args_refresh_flags_.push_back((dsa_task.random_count_value_or_ptr() == kDSASetInputAddr)
                                    ? input_addr_refresh_[0U] : 0U);
  sqe_io_addrs_.push_back(random_count_value_or_addr);
  dsa_sqe_.dsaCfgNumberLow = static_cast<uint32_t>(random_count_value_or_addr & kMask32Bits);
  dsa_sqe_.dsaCfgNumberHigh = static_cast<uint32_t>(random_count_value_or_addr >> k32Bits);

  return SUCCESS;
}

void DSATaskInfo::GetAddrs(const IowAddrs &iow_addrs) {
  for (size_t i = 0U; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] = iow_addrs.input_logic_addrs[i].logic_addr;
    input_addr_refresh_[i] = BoolToUint8(iow_addrs.input_logic_addrs[i].support_refresh);
  }
  for (size_t i = 0U; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] = iow_addrs.output_logic_addrs[i].logic_addr;
    output_addr_refresh_[i] = BoolToUint8(iow_addrs.output_logic_addrs[i].support_refresh);
  }
  for (size_t i = 0U; i < workspace_data_addrs_.size(); i++) {
    workspace_data_addrs_[i] = iow_addrs.workspace_logic_addrs[i].logic_addr;
    workspace_addr_refresh_[i] = BoolToUint8(iow_addrs.workspace_logic_addrs[i].support_refresh);
  }

  // todo: dsa 独立申请wordspace地址
  if (davinci_model_->IsFeatureBaseRefreshable()) {
    for (size_t i = 0U; i < workspace_data_addrs_.size(); i++) {
      workspace_data_addrs_[i] =  PtrToValue(self_wkspace_base_) + static_cast<uint64_t>(workspace_offset_[i]);
      workspace_addr_refresh_[i] = 0U;
    }
  }

  return;
}

Status DSATaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                      TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  const domi::DSATaskDef &dsa_task = task_def.dsa_task();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(dsa_task.op_index());
  GE_CHECK_NOTNULL(op_desc);

  // 解析 task run param
  int64_t value = 0L;
  const rtError_t rt_ret = rtGetRtCapability(FEATURE_TYPE_UPDATE_SQE, UPDATE_SQE_SUPPORT_DSA, &value);
  GE_ASSERT_TRUE((rt_ret == RT_ERROR_NONE), "[Call][RtGetRtCapability] failed, ret = 0x%x.", rt_ret);

  // todo: 后面优化目标是地址和刷新信息不保存到成员变量上
  support_refresh_ = (value == static_cast<int64_t>(RT_CAPABILITY_SUPPORT));
  GELOGI("support_refresh:%d, value:%" PRId64, static_cast<int32_t>(support_refresh_), value);
  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  std::vector<uint64_t> input_addr_mem_types;
  input_data_addrs_ = ModelUtils::GetInputDataAddrsValue(rts_param, op_desc, input_addr_mem_types);
  GE_ASSERT_TRUE((input_data_addrs_.size() >= kDSAInputAddrSize), "Node %s input addr size %zu is wrong",
                 op_desc->GetName().c_str(), input_data_addrs_.size());
  std::vector<uint64_t> output_addr_mem_types;
  output_data_addrs_ = ModelUtils::GetOutputDataAddrsValue(rts_param, op_desc, output_addr_mem_types);
  GE_ASSERT_TRUE((output_data_addrs_.size() == kDSAOutputAddrSize), "Node %s output addr size %zu is wrong",
                 op_desc->GetName().c_str(), output_data_addrs_.size());
  std::vector<uint64_t> wkspace_addr_mem_types;
  workspace_data_addrs_ = ModelUtils::GetWorkspaceDataAddrsValue(rts_param, op_desc, wkspace_addr_mem_types);
  GE_ASSERT_TRUE((!workspace_data_addrs_.empty()), "Node %s workspace addr size %zu is wrong",
                 op_desc->GetName().c_str(), workspace_data_addrs_.size());
  for (size_t i = 0U; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({input_data_addrs_[i], input_addr_mem_types[i], support_refresh_, {0}});
  }
  for (size_t i = 0U; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({output_data_addrs_[i], output_addr_mem_types[i],
                                                 support_refresh_, {0}});
  }
  for (size_t i = 0U; i < workspace_data_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back({workspace_data_addrs_[i], wkspace_addr_mem_types[i],
                                                    support_refresh_, {0}});
  }
  input_addr_refresh_.resize(input_data_addrs_.size(), (support_refresh_ ? 1U : 0U));
  output_addr_refresh_.resize(output_data_addrs_.size(), (support_refresh_ ? 1U : 0U));
  workspace_addr_refresh_.resize(workspace_data_addrs_.size(), (support_refresh_ ? 1U : 0U));
  const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
  GE_CHECK_GE(workspace_size.size(), workspace_data_addrs_.size());
  // todo: 如果dump打开了, 才申请一个dump args, 追加到hbm args尾部
  if (support_refresh_) {
    hbm_args_len_ = static_cast<int64_t>(MemSizeAlign(
        static_cast<size_t>(workspace_size[workspace_data_addrs_.size() - 1U]),
        static_cast<uint32_t>(sizeof(uint64_t)))) +
        static_cast<int64_t>(sizeof(uint64_t) * (input_data_addrs_.size() + output_data_addrs_.size()));
    task_run_param.args_descs.push_back({hbm_args_len_, ArgsPlacement::kArgsPlacementHbm});
    task_run_param.args_descs.push_back({kSqeArgsLen, ArgsPlacement::kArgsPlacementSqe});
    GELOGI("opType[%s] Node name[%s], sqe args len is %" PRId64 ", hbm_args len is %" PRId64 ", "
        "workspace_size %" PRId64 ", io size %zu.",
        op_desc->GetType().c_str(), op_desc->GetName().c_str(), kSqeArgsLen, hbm_args_len_,
        workspace_size[workspace_data_addrs_.size() - 1U],
        (sizeof(uint64_t) * (input_data_addrs_.size() + output_data_addrs_.size())));
  }
  if (!davinci_model->IsFeatureBaseRefreshable()) {
    return SUCCESS;
  }

  const std::vector<int64_t> workspace_bytes = op_desc->GetWorkspaceBytes();
  if (workspace_bytes.empty()) {
    GELOGE(INTERNAL_ERROR, "Node %s workspace size %zu is wrong", op_desc->GetName().c_str(),
           workspace_bytes.size());
    return INTERNAL_ERROR;
  }

  int64_t wkspace_size = 0;
  for (size_t i = 0U; i < workspace_bytes.size(); ++i) {
    workspace_offset_.push_back(wkspace_size);
    wkspace_size += workspace_bytes[i];
  }
  self_wkspace_base_ = davinci_model->MallocDynamicMemory(static_cast<size_t>(wkspace_size));
  GE_ASSERT_NOTNULL(self_wkspace_base_);

  return SUCCESS;
}

Status DSATaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("DSATaskInfo Distribute Start.");
  SetTaskTag(op_desc_->GetName().c_str());

  const TaskProfGuarder prof_guarder(this);
  GE_CHK_RT_RET(ge::rtStarsTaskLaunchWithFlag(&dsa_sqe_, static_cast<uint32_t>(sizeof(dsa_sqe_)), stream_, dump_flag_));
  GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
  GE_CHK_RT_RET(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)));
  GELOGI("DSATaskInfo %s Distribute TaskId[%u], stream id [%u], dumpflag [%u] Success.",
         op_desc_->GetNamePtr(), task_id_, stream_id_, dump_flag_);

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
  }
  is_support_redistribute_ = true;

  GELOGI("DSATaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

Status DSATaskInfo::Release() {
  self_wkspace_base_ = nullptr;
  return SUCCESS;
}

void DSATaskInfo::InitDsaDumpInfo(const OpDescPtr &op_desc) {
  if ((davinci_model_->OpNeedDump(op_desc->GetName()) ||
    davinci_model_->OpNeedSetDumpFlagOnWatcherModel(op_desc->GetName()))) {
    dump_flag_ = RT_KERNEL_DUMPFLAG;
  }
  return;
}

void DSATaskInfo::PostProcess(const domi::TaskDef &task_def) {
  if (dump_flag_ == RT_KERNEL_DUMPFLAG) {
    PostDumpProcess(task_def);
    GELOGI("PostDumpProcess Success");
  }
  PostProfilingProcess(task_def);
  return;
}

void DSATaskInfo::PostProfilingProcess(const domi::TaskDef &task_def) {
  const domi::DSATaskDef &dsa_task = task_def.dsa_task();
  GE_CHK_RT_EXEC(rtsGetThreadLastTaskId(&task_id_), return);
  GE_CHK_RT_EXEC(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)), return);
  davinci_model_->SaveDfxInfo(dsa_task.op_index(), task_def, *this);
}

void DSATaskInfo::PostDumpProcess(const domi::TaskDef &task_def) {
  const auto &dsatask_def = task_def.dsa_task();
  const uint32_t op_index = dsatask_def.op_index();
  const auto op_desc = davinci_model_->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  if (davinci_model_->OpNoNeedDumpOnWatcherModel(op_desc->GetName())) {
    GELOGI("No need to save common dump task in watcher mode");
    return;
  }

  std::vector<uint64_t> dump_io_addr;

  (void)dump_io_addr.insert(dump_io_addr.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
  (void)dump_io_addr.insert(dump_io_addr.cend(), output_data_addrs_.cbegin(), output_data_addrs_.cend());

  GELOGI("DSATaskInfo PostDumpProcess TaskId[%u], stream id [%u], op_index [%u] args[%p] Success.",
         task_id_, stream_id_, op_index, dump_io_addr.data());

  std::vector<uintptr_t> args;
  for (size_t i = 0U; i < dump_io_addr.size(); i++) {
    args.emplace_back(static_cast<uintptr_t>(dump_io_addr[i]));
  }
  const OpDescInfoId id{task_id_, stream_id_, 0U, 0U};
  const FirstLevelAddressInfo first_level_address_info{true, args};
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  GELOGI("Start to SaveDumpTask for op[%s], task_type[%u]", op_desc->GetName().c_str(),
         static_cast<uint32_t>(task_type));
  if ((!support_refresh_) || (!davinci_model_->IsFeatureBaseRefreshable())) {
      // 兼容性考虑 support_refresh_ 为false表示是老的drv包是老包, 或者不支持刷新的场景下, 走一级指针dump
      davinci_model_->SaveDumpTask(id, op_desc, static_cast<uintptr_t>(PtrToValue(args.data())),
                                   first_level_address_info, {}, task_type);
  } else {
    // support_refresh_ 为true表示是dsa支持可刷新, 走二级指针dump流程, 仅纯静态图在此拷贝, 其他走args table拷贝
    (void)rtMemcpy(ValueToPtr(dump_args_), sizeof(uint64_t) * dump_io_addr.size(), dump_io_addr.data(),
                   sizeof(uint64_t) * dump_io_addr.size(), RT_MEMCPY_HOST_TO_DEVICE);
    // Dump of second-level addresses
    davinci_model_->SaveDumpTask(id, op_desc, static_cast<uintptr_t>(dump_args_), {false, {}}, {}, task_type);
  }

  return;
}

int64_t DSATaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::DSATaskDef &dsa_task = task_def.dsa_task();
  return static_cast<int64_t>(dsa_task.op_index());
}

Status DSATaskInfo::UpdateHostArgsWithSqePlacement(const std::vector<uint64_t> &active_mem_base_addr,
                                   void *const host_args,
                                   const size_t host_args_len) const {
  return sqe_args_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_len);
}

Status DSATaskInfo::UpdateHostArgsWithHbmPlacement(const std::vector<uint64_t> &active_mem_base_addr,
                                   void *const host_args,
                                   const size_t host_args_len) const {
  return workspace_args_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_len);
}

Status DSATaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                                   const std::vector<HostArg> &host_args) {
GE_ASSERT_TRUE((host_args.size() == 2U), "[Update][HostArgs]args does not support, host args size is %zu.",
               host_args.size());

  for (auto &arg : host_args) {
    GE_ASSERT_TRUE(((arg.placement == ArgsPlacement::kArgsPlacementSqe) ||
                   (arg.placement == ArgsPlacement::kArgsPlacementHbm)), "[Update][HostArgs]invalid placement %d.",
                   arg.placement);
    if (arg.placement == ArgsPlacement::kArgsPlacementSqe) {
      GE_ASSERT_SUCCESS(UpdateHostArgsWithSqePlacement(active_mem_base_addr, arg.addr, static_cast<size_t>(arg.len)));
    } else {
      GE_ASSERT_SUCCESS(UpdateHostArgsWithHbmPlacement(active_mem_base_addr, arg.addr, static_cast<size_t>(arg.len)));
    }
  }

  return SUCCESS;
}

Status DSATaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  if (support_refresh_) {
    sqe_args_updater_.GenArgsRefreshInfos(infos, 0UL, ArgsPlacement::kArgsPlacementSqe);
    workspace_args_updater_.GenArgsRefreshInfos(infos, 0UL, ArgsPlacement::kArgsPlacementHbm);
  }
  return SUCCESS;
}

Status DSATaskInfo::GetTaskIowPaRemapInfos(std::vector<IowPaRemapInfo> &infos) {
  if (!support_refresh_) {  // uce remap需求匹配的drv版本已支持dsa刷新
    GELOGW("Dsa task no support get task remap info.");
    return SUCCESS;
  }

  if ((davinci_model_->IsFeatureBaseRefreshable()) || (workspace_data_addrs_.size() != kDSAWorkspaceAddrSize)) {
    return SUCCESS;
  }

  const OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index_);
  GE_CHECK_NOTNULL(op_desc);
  const std::vector<int64_t> workspace_bytes = op_desc->GetWorkspaceBytes();
  GE_ASSERT_TRUE(workspace_bytes.size() == kDSAWorkspaceAddrSize);
  std::vector<MemAllocationAndOffset> mem_allocation_id_and_offset;
  sqe_args_updater_.GetArgsMemAllocationAndOffset(mem_allocation_id_and_offset);
  GE_ASSERT_TRUE(stateful_workspace_idx_ < mem_allocation_id_and_offset.size());
  infos.push_back({this, static_cast<uint32_t>(mem_allocation_id_and_offset[stateful_workspace_idx_].id),
                   mem_allocation_id_and_offset[stateful_workspace_idx_].offset,
                   static_cast<uint64_t>(workspace_bytes[0UL]), PaRemapPolicy::KNoSupport, ""});
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_DSA, DSATaskInfo);
}  // namespace ge
