/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/memcpy_addr_async_task_info.h"

#include "runtime/mem.h"
#include "graph/args_format_desc.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace {
constexpr size_t kAlignment = 64U;
} // anonymous namespace

namespace ge {
Status MemcpyAddrAsyncTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                                  TaskRunParam &task_run_param) {
  GE_ASSERT_NOTNULL(davinci_model);
  const auto &memcpy_async = task_def.memcpy_async();
  op_desc_ = davinci_model->GetOpByIndex(memcpy_async.op_index());
  GE_ASSERT_NOTNULL(op_desc_);

  // TODO: RTS暂时不能从Torino分支回合，故为空args_format的场景提供默认值
  const auto &format_str = !memcpy_async.args_format().empty() ? memcpy_async.args_format()
                                                               : "{}{}{i_instance0*}{o_instance0*}";
  GE_ASSERT_GRAPH_SUCCESS(ArgsFormatDesc::FromString(format_, op_desc_, format_str));
  GELOGD("args format: %s", format_str.c_str());

  dst_max_ = memcpy_async.dst_max();
  count_ = memcpy_async.count();
  kind_ = static_cast<rtMemcpyKind_t>(memcpy_async.kind());
  GE_ASSERT_GRAPH_SUCCESS(format_.GetArgsSize(op_desc_, args_size_));
  pls_ = (rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, args_size_ + kAlignment) & RT_MEMORY_TS) != 0 ?
         ArgsPlacement::kArgsPlacementTs : ArgsPlacement::kArgsPlacementHbm;
  GELOGI("size: %zu (with extra %zu alignment addend), placement: %d, dst_max: %" PRIu64 ", count: %" PRIu64 ", kind: %d",
         args_size_, kAlignment, static_cast<int32_t>(pls_), dst_max_, count_, static_cast<int32_t>(kind_));

  uint8_t *src = nullptr;
  uint8_t *dst = nullptr;
  uint64_t src_mem_type = kFixMemType;
  uint64_t dst_mem_type = kFixMemType;
  const auto &rts_param = davinci_model->GetRuntimeParam();

  GE_ASSERT_SUCCESS(ModelUtils::GetRtAddress(rts_param, memcpy_async.src(), src, src_mem_type));
  GE_ASSERT_SUCCESS(ModelUtils::GetRtAddress(rts_param, memcpy_async.dst(), dst, dst_mem_type));

  task_run_param.parsed_input_addrs.push_back({PtrToValue(src), src_mem_type, true, {0U}});
  task_run_param.parsed_output_addrs.push_back({PtrToValue(dst), dst_mem_type, true, {0U}});
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size_ + kAlignment), pls_});
  return SUCCESS;
}

Status MemcpyAddrAsyncTaskInfo::HandleZeroCopy(DavinciModel *const davinci_model, uintptr_t base,
                                               uint64_t logic_addr) {
  // AICPU scheduler cannot access TS memory.
  if (davinci_model->IsArgsUpdateByDeviceAicpu() && pls_ == ArgsPlacement::kArgsPlacementTs) {
    GE_ASSERT_SUCCESS(davinci_model->DisableZeroCopy(ValueToPtr(logic_addr)));
  } else {
    std::vector<uint64_t> addrs = {logic_addr};
    davinci_model->SetZeroCopyAddr(op_desc_, addrs, addrs.data(), base, sizeof(uint64_t), 0, {});
  }
  return SUCCESS;
}

Status MemcpyAddrAsyncTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                     const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                     const IowAddrs &iow_addrs) {
  (void)persistent_workspace;
  GE_ASSERT_NOTNULL(davinci_model);
  GE_ASSERT_TRUE(!iow_addrs.input_logic_addrs.empty(), "Op:%s, empty input", op_desc_->GetName().c_str());
  GE_ASSERT_TRUE(!iow_addrs.output_logic_addrs.empty(), "Op:%s, empty output", op_desc_->GetName().c_str());
  GE_ASSERT_SUCCESS(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  const auto &arg = args[static_cast<size_t>(pls_)];
  size_t align_offset = (arg.dev_addr + kAlignment - 1) / kAlignment * kAlignment - arg.dev_addr;
  device_args_aligned_ = arg.dev_addr + align_offset;
  host_args_aligned_ = ValueToPtr(PtrToValue(arg.host_addr) + align_offset);
  GELOGI("arg.dev_addr: %p, device_args_aligned: %p, arg.host_addr: %p, host_args_aligned: %p",
         arg.dev_addr, device_args_aligned_, arg.host_addr, host_args_aligned_);

  // 此处假设RTS提供的args_format中输入与输出字段相邻排布，中间不能有其他字段
  // 首个IO地址的偏移量将被记录，其与align_offset之和作为后续地址刷新的总偏移
  // + ------------ + --------- + ------- + --- +
  // | align_offset | io_offset | src|dst | ... |
  // + ------------ + --------- + ------- + --- +
  // |<-  aligned_io_offset_  ->|
  std::vector<uint64_t> io_addrs;
  std::vector<uint64_t> io_addr_mem_types;
  size_t io_offset = 0;
  bool io_encountered = false;
  for (const auto &iter : format_) {
    if ((iter.addr_type == AddrType::INPUT_INSTANCE || iter.addr_type == AddrType::OUTPUT_INSTANCE) &&
        !io_encountered) {
      GELOGI("align_offset: %zu, io_offset: %zu", align_offset, io_offset);
      aligned_io_offset_ = align_offset + io_offset;
      io_encountered = true;
    }
    if (iter.addr_type == AddrType::INPUT_INSTANCE) {
      GE_ASSERT_TRUE(static_cast<size_t>(iter.ir_idx) < iow_addrs.input_logic_addrs.size());
      io_addrs.push_back(iow_addrs.input_logic_addrs[iter.ir_idx].logic_addr);
      io_addr_mem_types.push_back(iow_addrs.input_logic_addrs[iter.ir_idx].memory_type);
      GE_ASSERT_SUCCESS(HandleZeroCopy(davinci_model, device_args_aligned_ + io_offset,
                                       iow_addrs.input_logic_addrs[iter.ir_idx].logic_addr));
    } else if (iter.addr_type == AddrType::OUTPUT_INSTANCE) {
      GE_ASSERT_TRUE(static_cast<size_t>(iter.ir_idx) < iow_addrs.output_logic_addrs.size());
      io_addrs.push_back(iow_addrs.output_logic_addrs[iter.ir_idx].logic_addr);
      io_addr_mem_types.push_back(iow_addrs.output_logic_addrs[iter.ir_idx].memory_type);
      GE_ASSERT_SUCCESS(HandleZeroCopy(davinci_model, device_args_aligned_ + io_offset,
                                       iow_addrs.output_logic_addrs[iter.ir_idx].logic_addr));
    }
    GE_ASSERT_SUCCESS(ArgsFormatDesc::GetArgSize(op_desc_, iter, io_offset));
  }

  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model->GetLogicalMemAllocation(), io_addrs,
                                                io_addr_mem_types, {op_desc_->GetName(), op_desc_->GetType()}),
                    "args io addrs updater init failed.");

  GELOGI("MemcpyAddrAsyncTaskInfo Init Success, node :%s, logic stream id: %u, stream: %p.",
    op_desc_->GetName().c_str(), task_def.stream_id(), stream_);

  return SUCCESS;
}

Status MemcpyAddrAsyncTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, aligned_io_offset_, pls_);
  return SUCCESS;
}

Status MemcpyAddrAsyncTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("MemcpyAddrAsyncTaskInfo Distribute Start, op %s, dst_max:%" PRIu64 ", count:%" PRIu64 ", kind:%u",
         op_desc_->GetName().c_str(), dst_max_, count_, kind_);
  SetTaskTag(op_desc_->GetName().c_str());

  const auto rt_ret = rtMemcpyAsyncPtr(ValueToPtr(device_args_aligned_), dst_max_, count_, kind_,
                                       stream_, qosCfg_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpyAsyncWithCfg failed, size:%" PRIu64 ",ret:%d", dst_max_, rt_ret);
    GELOGE(RT_FAILED, "[Call][rtMemcpyAsyncWithCfg] failed, size:%" PRIu64 ", ret:%d", dst_max_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GE_CHK_RT_RET(aclrtMemcpy(host_args_aligned_, args_size_, ValueToPtr(device_args_aligned_), args_size_,
      ACL_MEMCPY_DEVICE_TO_HOST));

  uintptr_t host_addr = PtrToValue(host_args_aligned_);
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

  GELOGI("MemcpyAddrAsyncTaskInfo Distribute Success, op %s, dst_max:%" PRIu64 ", count:%" PRIu64
    ", kind:%us, stream: %p.", op_desc_->GetNamePtr(), dst_max_, count_, kind_, stream_);

  is_support_redistribute_ = true;

  return SUCCESS;
}

int64_t MemcpyAddrAsyncTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  return static_cast<int64_t>(memcpy_async.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_MEMCPY_ADDR_ASYNC, MemcpyAddrAsyncTaskInfo);
}  // namespace ge
