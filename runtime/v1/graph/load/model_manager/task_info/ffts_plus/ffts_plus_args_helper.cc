/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffts_plus_args_helper.h"
#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"
namespace {
constexpr size_t kDescBufAlignedBytes = 128UL;
}
namespace ge {

void FftsPlusArgsHelper::AppendAbsoluteAddrs(const uint64_t rt_addr, const std::string &addr_type) {
  GELOGD("AppendAbsoluteAddrs addr_type:[%s] ctx_op:[%s] ctx_id:[%d] idx:[%zu], "
         "logic_addr:[%" PRIx64 "] mem_type:[%" PRIx64 "]",
         addr_type.c_str(), ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, io_addrs_.size(), rt_addr,
         kAbsoluteMemType);
  io_addrs_.push_back(rt_addr);
  io_mem_types_.push_back(kAbsoluteMemType);
  ctx_args_size_[ctx_info_.ctx_id] += sizeof(uint64_t);
}

void FftsPlusArgsHelper::AppendIoAddrs(const uint64_t logic_addr) {
  uint64_t logic_addr_to_push{logic_addr};
  uint64_t mem_type = kAbsoluteMemType;
  if (mode_addr_idx_.count(io_addrs_.size()) == 0UL) {
    uint8_t *mem_addr = nullptr;
    if (ModelUtils::GetRtAddress(runtime_param_, static_cast<uintptr_t>(logic_addr), mem_addr, mem_type) == SUCCESS) {
      logic_addr_to_push = PtrToValue(mem_addr);
    }
  }
  GELOGD("AppendIoAddrs ctx_op:[%s] ctx_id:[%d] idx:[%zu], logic_addr:[%" PRIx64 "] iow_mem_type:[%" PRIx64 "]",
         ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, io_addrs_.size(), logic_addr_to_push, mem_type);
  io_addrs_.push_back(logic_addr_to_push);
  io_mem_types_.push_back(mem_type);
  ctx_args_size_[ctx_info_.ctx_id] += sizeof(uint64_t);
}

void FftsPlusArgsHelper::AppendRtIoAddrs(const uint64_t rt_addr, const uint64_t mem_type) {
  GELOGD("AppendRtIoAddrs ctx_op:[%s] ctx_id:[%d] idx:[%zu], rt_addr:[%" PRIx64 "] iow_mem_type:[%" PRIx64 "]",
         ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, io_addrs_.size(), rt_addr, mem_type);
  io_addrs_.push_back(rt_addr);
  io_mem_types_.push_back(mem_type);
  ctx_args_size_[ctx_info_.ctx_id] += sizeof(uint64_t);
}

void FftsPlusArgsHelper::AppendCtxLevel1Addrs(const uint64_t logic_addr, const uint64_t iow_mem_type,
                                              const CtxAddrInfo &ctx_addr_info) {
  GELOGD("AppendCtxLevel1Addrs ctx_op:[%s] ctx_id:[%d], idx:[%zu], addr:[%" PRIx64 "] iow_mem_type:[%" PRIx64 "]",
         ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, level1_logic_heads_.size(), logic_addr, iow_mem_type);
  level1_logic_heads_.push_back(logic_addr);
  level1_mem_types_.push_back(iow_mem_type);
  level1_ctx_addr_infos_.push_back(ctx_addr_info);
}

Status FftsPlusArgsHelper::AppendAicpuAddrs(const uint64_t rt_addr, const uint64_t io_mem_type,
                                            const size_t relevant_offset) {
  // thread offset for auto thread mode, default is 0.
  GELOGD("AppendAicpuAddrs ctx_op:[%s] ctx_id:[%d] idx:[%zu], rt_addr:[%" PRIx64 "] iow_mem_type:[%" PRIx64 "]",
         ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, aicpu_logic_heads_.size(), rt_addr, io_mem_type);
  aicpu_logic_heads_.push_back(rt_addr);
  aicpu_logic_heads_mem_types_.push_back(io_mem_type);
  size_t ret{0UL};
  GE_ASSERT_TRUE(!ge::AddOverflow(relevant_offset, used_bin_args_size_, ret));
  args_relevent_offsets_.push_back(ret);
  ctx_args_size_[ctx_info_.ctx_id] += sizeof(uint64_t);
  return SUCCESS;
}

Status FftsPlusArgsHelper::AppendBinArgs(const uint8_t *const args_addr, const size_t args_size) {
  GE_ASSERT(bin_args_size_ >= used_bin_args_size_, "bin args size [%zu] is overflow", args_size);
  uint8_t *const host_addr_begin = bin_args_host_ + used_bin_args_size_;
  const size_t bin_size_left = static_cast<size_t>(bin_args_size_ - used_bin_args_size_);
  if (memcpy_s(host_addr_begin, bin_size_left, args_addr, args_size) != EOK) {
    GELOGE(FAILED, "Bin args memcpy_s failed size_left:[%zu], copy_size:[%zu]", bin_size_left, args_size);
    return FAILED;
  }
  used_bin_args_size_ += args_size;
  ctx_args_size_[ctx_info_.ctx_id] += args_size;
  return SUCCESS;
}

Status FftsPlusArgsHelper::UpdateIoAddrByIndex(const size_t index, const uint64_t rt_addr) {
  GE_ASSERT_TRUE(((index < io_addrs_.size()) && (index < io_mem_types_.size())),
    "ctx_op:[%s] ctx_id:[%d] idx:[%zu] is greater than io addr size:[%zu] or io mem size:[%zu]",
    ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, index, io_addrs_.size(), io_mem_types_.size());
  io_addrs_[index] = rt_addr;
  GELOGD("UpdateIoAddrByIndex ctx_op:[%s] ctx_id:[%d] idx:[%zu], logic_addr:[%" PRIx64 "] mem_type:[%" PRIx64 "]",
         ctx_info_.ctx_op->GetNamePtr(), ctx_info_.ctx_id, index, io_addrs_[index], io_mem_types_[index]);
  return SUCCESS;
}

Status FftsPlusArgsHelper::InitArgsBase(uint8_t *const pis_args_host_base, uint8_t *const args_dev,
                                        uint8_t *const args_host, const size_t args_size, const size_t bin_args_size) {
  GE_ASSERT(args_size >= bin_args_size, "bin args size [%zu] is greater than args_size:[%zu]", bin_args_size,
            args_size);
  pis_args_host_base_ = pis_args_host_base;
  args_host_ = args_host;
  args_dev_ = args_dev;
  args_size_ = args_size;
  bin_args_size_ = bin_args_size;
  bin_args_host_ = args_host_ + args_size_ - bin_args_size_;

  GELOGI(
      "Args init successfully, with details {args: %p args_host: %p bin_args_host: %p args_size: %zu bin_args_size: "
      "%zu}",
      args_dev_, args_host_, bin_args_host_, args_size_, bin_args_size_);
  return SUCCESS;
}

Status FftsPlusArgsHelper::PlanUpdaterArgslayOut(DavinciModel *davinci_model) {
  // append level1 to io_addrs
  (void)io_addrs_.insert(io_addrs_.cend(), level1_logic_heads_.cbegin(), level1_logic_heads_.cend());
  (void)io_mem_types_.insert(io_mem_types_.cend(), level1_mem_types_.cbegin(), level1_mem_types_.cend());
  // append ascend aicpu to io_addrs
  (void)io_addrs_.insert(io_addrs_.cend(), aicpu_logic_heads_.cbegin(), aicpu_logic_heads_.cend());
  (void)io_mem_types_.insert(io_mem_types_.cend(), aicpu_logic_heads_mem_types_.cbegin(),
                             aicpu_logic_heads_mem_types_.cend());

  std::vector<uint8_t> refreshable_flags(io_addrs_.size(), 1U);
  GE_ASSERT_EQ(io_addrs_.size(), io_mem_types_.size());
  for (size_t i = 0U; i < io_mem_types_.size(); ++i) {
    refreshable_flags[i] = static_cast<uint8_t>(ModelUtils::IsFeatureMapOrModelIoType(io_mem_types_[i]));
  }
  (void)args_io_addrs_updater_.Init(davinci_model->GetLogicalMemAllocation(), io_addrs_, refreshable_flags,
                                    {op_desc_->GetName(), op_desc_->GetType()});
  level1_logic_heads_.clear();
  level1_mem_types_.clear();
  aicpu_logic_heads_.clear();
  aicpu_logic_heads_mem_types_.clear();
  return SUCCESS;
}

Status FftsPlusArgsHelper::InitRuntimeAddr(DavinciModel *davinci_model) {
  const size_t addr_size = sizeof(uint64_t) * io_addrs_.size();
  if ((args_dev_ == nullptr) || (args_host_ == nullptr) || (args_size_ < addr_size)) {
    GELOGE(FAILED, "[Check][Param] Invalid args: args size: %zu, adds size: %zu", args_size_, addr_size);
    return FAILED;
  }
  aicaiv_addr_size_ = io_addrs_.size();
  level1_addr_size_ = level1_logic_heads_.size();
  const size_t aicpu_logic_heads_size = aicpu_logic_heads_.size();
  GE_ASSERT_SUCCESS(PlanUpdaterArgslayOut(davinci_model));
  GELOGI(
      "Op %s is ready for copy to addr %p, addr_size=%" PRIu64 ", len=%" PRIu64 ", aicaiv_addr_size=%zu, "
      "level1_addr_size=%zu, ascend_aicpu_addr_size=%zu.",
      op_desc_->GetNamePtr(), args_dev_, args_size_, addr_size, aicaiv_addr_size_, level1_addr_size_,
      aicpu_logic_heads_size);
  return SUCCESS;
}

const std::string FftsPlusArgsHelper::GetMemCheckInfo(const std::string &op_name) const {
  const auto iter = memcheck_infos_.find(op_name);
  if (iter != memcheck_infos_.end()) {
    return iter->second;
  }
  return "";
}

bool FftsPlusArgsHelper::CheckAndGetLevel2Offset(const uint32_t ctx_id, int64_t &sub_offset) {
  const auto iter = data_ctx_to_level2_offset_.find(ctx_id);
  if (iter != data_ctx_to_level2_offset_.cend()) {
    sub_offset = iter->second;
    return true;
  }
  return false;
}

bool FftsPlusArgsHelper::CheckAndGetArgsFormats(int64_t op_id, ArgsFormatHolder &holder) {
  auto iter = op_to_format_holder_.find(op_id);
  if (iter != op_to_format_holder_.end()) {
    holder = iter->second;
    return true;
  }
  return false;
}

void FftsPlusArgsHelper::SaveOffsetPairs(const uint64_t cust_offset, const uint64_t relevant_offset) {
  GELOGD("Save cust_offset [%" PRIu64 "] to relevant offset:[%" PRIu64 "]", cust_offset, relevant_offset);
  cust_to_relevant_offset_[cust_offset] = relevant_offset;
}

Status FftsPlusArgsHelper::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos,
                                                   FftsPlusProtoTransfer &ffts_plus_proto_transfer) {
  // 获取args table refresh info
  std::vector<TaskArgsRefreshInfo> args_refresh_infos;
  uint64_t args_table_offset =
    PtrToValue(PtrToPtr<uint8_t, void>(args_host_)) - PtrToValue(PtrToPtr<uint8_t, void>(pis_args_host_base_));
  (void)args_io_addrs_updater_.GenArgsRefreshInfos(args_refresh_infos,
                                                   args_table_offset,
                                                   ArgsPlacement::kArgsPlacementHbm);

  // 获取contex level1 refresh info
  std::vector<TaskArgsRefreshInfo> ctx_level1_args_fresh_infos;
  GE_CHK_STATUS_RET(ffts_plus_proto_transfer.GenCtxLevel1RefreshInfo(level1_ctx_addr_infos_,
                                                                     args_refresh_infos,
                                                                     aicaiv_addr_size_,
                                                                     level1_addr_size_,
                                                                     ctx_level1_args_fresh_infos));

  // 获取aicpu refresh info
  std::vector<TaskArgsRefreshInfo> aicpu_args_fresh_infos;
  GE_CHK_STATUS_RET(GenAicpuRefreshInfos(args_refresh_infos,
                                         aicaiv_addr_size_ + level1_addr_size_,
                                         aicpu_args_fresh_infos));

  // 只保留args table表的参数， 不包含context leve1 和 aicpu
  infos.insert(infos.end(), args_refresh_infos.begin(),  args_refresh_infos.begin() + aicaiv_addr_size_);
  infos.insert(infos.end(), ctx_level1_args_fresh_infos.begin(), ctx_level1_args_fresh_infos.end());
  infos.insert(infos.end(), aicpu_args_fresh_infos.begin(), aicpu_args_fresh_infos.end());

  return SUCCESS;
}

Status FftsPlusArgsHelper::UpdateAddrsWithIOZcpy(const std::vector<uint64_t> &active_mem_base_addr,
                                                 FftsPlusProtoTransfer &ffts_plus_proto_transfer) {
  std::vector<uint64_t> io_addrs = io_addrs_;
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(
      active_mem_base_addr, PtrToPtr<uint64_t, void>(io_addrs.data()), io_addrs.size() * sizeof(uint64_t)));

  const size_t io_addr_size = static_cast<size_t>(args_size_ - bin_args_size_);
  // update i,o,w,t for aic,aiv,fmk aicpu
  if ((aicaiv_addr_size_ != 0U) &&
      (memcpy_s(args_host_, io_addr_size, io_addrs.data(), aicaiv_addr_size_ * sizeof(uint64_t)) != EOK)) {
    GELOGE(FAILED, "copy data failed, args_size:%zu, ioaddr_size:%zu.", io_addr_size, aicaiv_addr_size_);
    return FAILED;
  }
  // update level_1 pointer from ctx
  GE_CHK_STATUS_RET(ffts_plus_proto_transfer.UpdateCtxLevel1Addrs(level1_ctx_addr_infos_, io_addrs, aicaiv_addr_size_,
                                                                  level1_addr_size_));
  // update noncontinuous args for asecnd aicpu ctx
  GE_CHK_STATUS_RET(UpdateAicpuAddrs(io_addrs, aicaiv_addr_size_ + level1_addr_size_));
  GELOGD("Op %s update addr successfully.", op_desc_->GetNamePtr());
  return SUCCESS;
}

size_t FftsPlusArgsHelper::GetCtxArgsSize(const int32_t ctx_id) const {
  const auto &iter = ctx_args_size_.find(ctx_id);
  if (iter != ctx_args_size_.end()) {
    return iter->second;
  }
  return 0UL;
}

Status FftsPlusArgsHelper::GenAicpuRefreshInfos(const std::vector<TaskArgsRefreshInfo> &args_fresh_info,
                                                const size_t start_idx,
                                                std::vector<TaskArgsRefreshInfo> &aicpu_args_fresh_info) {
  GE_ASSERT_TRUE(start_idx + args_relevent_offsets_.size() == args_fresh_info.size(), "Aicpu ioaddr size mismatch.");

  for (size_t idx = 0UL; idx < args_relevent_offsets_.size(); ++idx) {
    GE_ASSERT(args_relevent_offsets_[idx] + sizeof(uint64_t) <= bin_args_size_);
    TaskArgsRefreshInfo info = {
        args_fresh_info[start_idx + idx].id,
        args_fresh_info[start_idx + idx].offset,
        0UL,
        PtrToValue(PtrToPtr<uint8_t, void>(&bin_args_host_[args_relevent_offsets_[idx]])) -
          PtrToValue(PtrToPtr<uint8_t, void>(pis_args_host_base_)),
        ArgsPlacement::kArgsPlacementHbm,
        ArgsFormatPolicy::kAddrAll,
    };
    aicpu_args_fresh_info.emplace_back(std::move(info));
  }

  return SUCCESS;
}

Status FftsPlusArgsHelper::UpdateAicpuAddrs(const std::vector<uint64_t> &io_addrs, const size_t start_idx) {
  if (start_idx + args_relevent_offsets_.size() != io_addrs.size()) {
    GELOGE(FAILED, "Aicpu ioaddr size mismatch.");
    return FAILED;
  }
  for (size_t idx = 0UL; idx < args_relevent_offsets_.size(); ++idx) {
    GE_ASSERT(args_relevent_offsets_[idx] + sizeof(uint64_t) <= bin_args_size_);
    uint64_t *const tmp_addr = PtrToPtr<void, uint64_t>(&bin_args_host_[args_relevent_offsets_[idx]]);
    *tmp_addr = io_addrs[start_idx + idx];
  }
  return SUCCESS;
}

Status FftsPlusArgsHelper::AssembleTilingData() const {
  if (tiling_data_len_ == 0U) {
    return SUCCESS;
  }
  GE_ASSERT_RT_OK(rtMemcpy(PtrToPtr<uint8_t, void>(tiling_data_dev_),
                           static_cast<uint64_t>(tiling_data_len_ + kDescBufAlignedBytes),
                           PtrToPtr<uint8_t, const void>(tiling_data_host_), static_cast<uint64_t>(tiling_data_len_),
                           RT_MEMCPY_HOST_TO_DEVICE));
  GELOGI("Memcpy to tiling addr: %p, len: %zu successfully.", tiling_data_dev_, tiling_data_len_);
  return SUCCESS;
}
}  // namespace ge
