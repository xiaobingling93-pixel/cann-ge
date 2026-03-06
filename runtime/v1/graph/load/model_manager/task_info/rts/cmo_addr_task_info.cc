/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/cmo_addr_task_info.h"

#include "runtime/mem.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/args_format_desc.h"
namespace ge {
constexpr uint32_t kMaxPrefetchLen = 120U * 1024U * 1024U;
constexpr uint64_t kAlignedBytes = 64U;
constexpr char_t const *kAttrMaxSize = "max_size";
constexpr char_t const *kAttrAddrOffset = "offset";

Status CmoAddrTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                          TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  const auto &cmo_addr_task = task_def.cmo_addr_task();
  op_desc_ = davinci_model->GetOpByIndex(cmo_addr_task.op_index());
  GE_CHECK_NOTNULL(op_desc_);
  auto format_str = task_def.cmo_addr_task().args_format();
  if (format_str.empty()) {
    const GeTensorDesc &tensor_desc = op_desc_->GetInputDesc(0U);
    int64_t num_cnt = tensor_desc.GetShape().IsScalar() ? 1 : tensor_desc.GetShape().GetShapeSize();
    int64_t shape_len = GetSizeInBytes(num_cnt, tensor_desc.GetDataType());
    GE_ASSERT_TRUE(shape_len > 0);
    int64_t offset{0};
    (void)AttrUtils::GetInt(op_desc_, kAttrAddrOffset, offset);
    if ((offset < 0) || (offset >= shape_len)) {
      REPORT_INNER_ERR_MSG("E19999", "The offset %" PRId64 " should be within the range of [0, %" PRId64 ").", offset,
                           shape_len);
      GELOGE(ge::PARAM_INVALID, "The offset [%" PRId64 "] should be within the range of [0, %" PRId64 ").", offset,
             shape_len);
      return ge::PARAM_INVALID;
    }
    shape_len -= offset;

    uint32_t max_size{0U};
    (void)AttrUtils::GetInt(op_desc_, kAttrMaxSize, max_size);
    if (max_size == 0) {
      max_size = kMaxPrefetchLen;
    }
    // 把len_inner值赋给format_str
    uint32_t len_inner = std::min(static_cast<uint32_t>(shape_len), max_size);
    format_str = "{}{.32b}{#.32b" + std::to_string(len_inner) + "}{i_instance0*}{}";
    GELOGI("Generating format_str for op: %s, shape_len: %" PRId64 ", offset: %" PRId64
           ", max_size: %u, len_inner: %u, format_str: %s",
           op_desc_->GetName().c_str(), shape_len, offset, max_size, len_inner, format_str.c_str());
  }
  GE_ASSERT_GRAPH_SUCCESS(ArgsFormatDesc::FromString(format_, op_desc_, format_str));
  GELOGI("op:%s, args format: %s", op_desc_->GetName().c_str(), format_str.c_str());

  GE_ASSERT_GRAPH_SUCCESS(format_.GetArgsSize(op_desc_, format_args_size_));

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  std::vector<uint64_t> mem_type;
  std::vector<uint64_t> input_addrs = ModelUtils::GetInputAddrsValue(rts_param, op_desc_, mem_type);
  GE_ASSERT_TRUE(input_addrs.size() == 1UL, "Input_addr size [%zu] is invalid, op: %s", input_addrs.size(),
                 op_desc_->GetNamePtr());
  GE_ASSERT_TRUE(mem_type.size() == 1UL, "Input_addr size [%zu] is invalid, op: %s", mem_type.size(),
                 op_desc_->GetNamePtr());
  args_size_ = format_args_size_ + kAlignedBytes;
  task_run_param.parsed_input_addrs.push_back({input_addrs[0UL], mem_type[0UL], true, {0}});
  const uint32_t args_mem_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, static_cast<uint32_t>(args_size_));
  args_placement_ =
      ((args_mem_type & RT_MEMORY_TS) == 0U) ? ArgsPlacement::kArgsPlacementHbm : ArgsPlacement::kArgsPlacementTs;
  GELOGI("args mem type:%u, args size:%" PRIu64 ", args placement:%d", args_mem_type, args_size_, args_placement_);
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size_), args_placement_});
  GELOGI("CmoAddrTaskInfo::ParseTaskRunParam success, op: %s", op_desc_->GetNamePtr());
  return SUCCESS;
}

Status CmoAddrTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                             const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  (void) persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));
  GE_CHECK_NOTNULL(op_desc_);

  const uint64_t args_base = args[static_cast<size_t>(args_placement_)].dev_addr;
  GE_ASSERT_TRUE((args_base != 0UL), "[Check][Param] Op:%s, args_placement:%d, dev addr is nullptr.",
                 op_desc_->GetNamePtr(), args_placement_);
  const uint64_t align_addr = ((args_base / kAlignedBytes) + 1U) * kAlignedBytes;
  const size_t align_offset = static_cast<size_t>(align_addr - args_base);

  args_ = ValueToPtr(align_addr);
  GE_ASSERT(static_cast<size_t>(args[static_cast<size_t>(args_placement_)].len) >= args_size_);
  host_args_ = ValueToPtr(PtrToValue(args[static_cast<size_t>(args_placement_)].host_addr) + align_offset);

  const auto &cmo_addr_task = task_def.cmo_addr_task();
  cmo_op_code_ = static_cast<rtCmoOpCode_t>(cmo_addr_task.cmo_op_code());

  io_addrs_.clear();
  io_addr_mem_types_.clear();
  size_t io_offset = 0;
  bool io_encountered = false;
  for (const auto &iter : format_) {
    if ((iter.addr_type == AddrType::INPUT_INSTANCE) && !io_encountered) {
      GELOGI("align_offset: %zu, io_offset: %zu", align_offset, io_offset);
      io_align_offset_ = align_offset + io_offset;
      io_encountered = true;
    }
    if (iter.addr_type == AddrType::INPUT_INSTANCE) {
      GE_ASSERT_TRUE(static_cast<size_t>(iter.ir_idx) < iow_addrs.input_logic_addrs.size());
      uint64_t base_addr = iow_addrs.input_logic_addrs[iter.ir_idx].logic_addr;
      io_addrs_.push_back(base_addr);
      io_addr_mem_types_.push_back(iow_addrs.input_logic_addrs[iter.ir_idx].memory_type);
      vector<uint64_t> addrs = {base_addr};
      davinci_model_->SetZeroCopyAddr(op_desc_, addrs, io_addrs_.data(), static_cast<uintptr_t>(args_base),
                                      sizeof(uint64_t), io_align_offset_, {});
    }
    GE_ASSERT_SUCCESS(ArgsFormatDesc::GetArgSize(op_desc_, iter, io_offset));
  }
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), io_addrs_,
                                                io_addr_mem_types_, {op_desc_->GetName(), op_desc_->GetType()}),
                    "args io addrs updater init failed.");

  GELOGI("CmoAddrTaskInfo Init Success, logic stream id: %u, stream: %p", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status CmoAddrTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                       const size_t host_args_max_len) {
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr,
                                                         ValueToPtr(PtrToValue(host_args) + io_align_offset_),
                                                         static_cast<size_t>(host_args_max_len - io_align_offset_)));
  GELOGI("CmoAddrTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status CmoAddrTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, static_cast<uint64_t>(io_align_offset_), args_placement_);
  GELOGI("CmoAddrTaskInfo::GetTaskArgsRefreshInfos success.");
  return SUCCESS;
}

Status CmoAddrTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("CmoAddrTaskInfo Distribute Start, op:[%s].", op_desc_->GetNamePtr());
  SetTaskTag(op_desc_->GetNamePtr());

  GE_CHK_RT_RET(rtCmoAddrTaskLaunch(args_, format_args_size_, cmo_op_code_, stream_, 0U));
  GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
  GE_CHK_RT_RET(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)));

  GE_CHK_RT_RET(aclrtMemcpy(host_args_, format_args_size_, args_, format_args_size_, ACL_MEMCPY_DEVICE_TO_HOST));
  uintptr_t host_addr = PtrToValue(host_args_);
  for (const auto &iter : format_) {
    if (iter.addr_type == AddrType::CUSTOM_VALUE) {
      uint64_t u64 = *reinterpret_cast<const uint64_t *>(iter.reserved);
      if (iter.ir_idx == static_cast<int32_t>(ArgsFormatWidth::BIT32)) {
        *reinterpret_cast<uint32_t *>(host_addr) = u64;
      } else {
        *reinterpret_cast<uint64_t *>(host_addr) = u64;
      }
      GELOGI("Write custom value[%" PRIu64 "] to host_addr[%p]", u64, host_addr);
    }
    GE_ASSERT_SUCCESS(ArgsFormatDesc::GetArgSize(op_desc_, iter, host_addr));
  }
  is_support_redistribute_ = true;
  GELOGI("CmoAddrTaskInfo Distribute Success, op: %s, stream: %p, stream_id: %d, task_id: %d.", op_desc_->GetNamePtr(),
         stream_, stream_id_, task_id_);
  return SUCCESS;
}

int64_t CmoAddrTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::CmoAddrTaskDef &cmo_addr_task = task_def.cmo_addr_task();
  return static_cast<int64_t>(cmo_addr_task.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_CMO_ADDR, CmoAddrTaskInfo);
}  // namespace ge
