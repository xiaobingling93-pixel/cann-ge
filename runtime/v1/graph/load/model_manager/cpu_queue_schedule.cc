/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/cpu_queue_schedule.h"
#include "securec.h"
#include "framework/common/runtime_tensor_desc.h"
#include "framework/common/debug/log.h"

namespace {
constexpr uint32_t kCoreDim = 1U;  // for rtCpuKernelLaunch
const std::string kCpuTaskModelEnqueue = "modelEnqueue";
const std::string kCpuTaskWaitEndGraph = "modelWaitEndGraph";
const std::string kCpuTaskPrepareOutput = "bufferPrepareOutput";
const std::string kCpuTaskPrepareOutputWithTensorDesc = "bufferPrepareOutputWithTensorDesc";
const std::string kCpuTaskStaticOutputPostProcess = "modelPrepareOutput";
const std::string kCpuTaskStaticOutputPostProcessWithTensorDesc = "modelPrepareOutputWithTensorDesc";
const std::string kCpuTaskDynOutputPostProcess = "dynOutputPostProcess";
const std::string kCpuTaskModelDequeue = "modelDequeue";
const std::string kCpuTaskModelBatchDequeue = "modelBatchDequeue";
const std::string kCpuTaskModelGatherDequeue = "gatherDequeue";
const std::string kCpuTaskMarkStep = "markStep";
const std::string kCpuTaskModelReportStatus = "modelReportStatus";
const std::string kCpuTaskModelRepeat = "modelRepeat";
const std::string kCpuTaskZeroCopy = "zeroCpy";
const std::string kCpuTaskZeroCopyV2 = "zeroCpyV2";
const std::string KCpuTaskMemCopyInput = "modelPrepareNonZeroCopyInput";
const std::string KCpuTaskCheckInputTensorDesc = "checkInputTensorDesc";
}  // namespace

namespace ge {
CpuTaskInfo::CpuTaskInfo(aclrtStream const stream) : TaskInfo() {
  stream_ = stream;
}

CpuTaskInfo::~CpuTaskInfo() {
  GE_FREE_RT_LOG(args_);
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
/// @param [in] queue_id: input queue id from user.
/// @param [out] in_mbuf: input mbuf addr for input data.
/// @return: 0 for success / others for failed
///
Status CpuTaskModelDequeue::Init(const uint32_t queue_id, uintptr_t &in_mbuf) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(MbufQueueInfo) + sizeof(uintptr_t);  // sizeof(uintptr_t) for save in_mbuf.
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  in_mbuf = PtrToValue(args_) + sizeof(MbufQueueInfo);
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);

  MbufQueueInfo queue_info;
  queue_info.queue_id = queue_id;
  queue_info.in_mbuf = in_mbuf;  // Placeholder, input mbuf addr will save to this place.
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &queue_info, sizeof(MbufQueueInfo),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status CpuTaskModelDequeue::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskModelDequeue));
  GELOGI("Cpu kernel launch model dequeue task success.");
  return SUCCESS;
}

Status CpuTaskZeroCopy::InitAddrs(std::vector<uintptr_t> &mbuf_list,
                                  const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
                                  const vector_bit_t &is_no_tiling_list,
                                  const ZeroCpyArgs &cpy_args) {
  // init src_addrs/dst_addrs/no_tilings_
  for (const auto &addrs : outside_addrs) {
    const size_t data_idx = addrs.first;
    const auto &addrs_mapping_list = addrs.second.GetOutsideAddrs();
    std::map<uintptr_t, bool> outside_addrs_is_tiling_map = addrs.second.GetOutsideAddrsIsTiling();
    if (addrs_mapping_list.empty()) {
      GELOGE(FAILED, "[Check][Param] not set outside_addrs");
      return PARAM_INVALID;
    }
    for (size_t count = 0U; count < addrs_mapping_list.size(); count++) {
      const std::map<uintptr_t, std::vector<uintptr_t>> &virtual_args_addrs = addrs_mapping_list[count];
      for (const auto &virtual_args_addr : virtual_args_addrs) {
        for (size_t i = 0U; i < virtual_args_addr.second.size(); ++i) {
          const uintptr_t &virtual_addr = virtual_args_addr.second.at(i);
          const bool is_no_tiling = is_no_tiling_list.at(data_idx);
          const bool dest_is_tiling = outside_addrs_is_tiling_map[virtual_addr];
          const int32_t fusion_offset = cpy_args.fusion_offsets.at(data_idx);
          GELOGI("Init addr, index:[%zu], cpy_type:[%d], is notiling:[%d], dest is tiling[%d], fusion offset = %d, "
                 "addr[%p] will be replaced.",
                 data_idx,
                 static_cast<int32_t>(cpy_args.cpy_type),
                 static_cast<int32_t>(is_no_tiling),
                 static_cast<int32_t>(dest_is_tiling),
                 fusion_offset,
                 reinterpret_cast<void *>(virtual_addr));
          SetAddrs(mbuf_list.at(data_idx), cpy_args, is_no_tiling, virtual_addr, dest_is_tiling, fusion_offset);
        }
      }
    }
  }
  return SUCCESS;
}

void CpuTaskZeroCopy::SetAddrs(uintptr_t src_addr, const ZeroCpyArgs &cpy_args, const bool is_no_tiling,
                               const uintptr_t virtual_addr, const bool dest_is_tiling, const int32_t fusion_offset) {
  if (cpy_args.cpy_type == ZeroCpyType::kAllStatic) {
    if (!is_no_tiling) {
      addr_num_++;
      src_addrs_.emplace_back(src_addr);
      dst_addrs_.push_back(virtual_addr);
      no_tilings_.emplace_back(static_cast<int32_t>(is_no_tiling));
      dest_is_tilings_.emplace_back(static_cast<int32_t>(dest_is_tiling));
      fusion_offsets_.emplace_back(fusion_offset);
    }
  } else if (cpy_args.cpy_type == ZeroCpyType::kAllDynamic) {
    if (is_no_tiling) {
      addr_num_++;
      src_addrs_.emplace_back(src_addr);
      dst_addrs_.push_back(virtual_addr);
      no_tilings_.emplace_back(static_cast<int32_t>(is_no_tiling));
      dest_is_tilings_.emplace_back(static_cast<int32_t>(dest_is_tiling));
      fusion_offsets_.emplace_back(fusion_offset);
    }
  } else {
    addr_num_++;
    src_addrs_.emplace_back(src_addr);
    dst_addrs_.push_back(virtual_addr);
    no_tilings_.emplace_back(static_cast<int32_t>(is_no_tiling));
    dest_is_tilings_.emplace_back(static_cast<int32_t>(dest_is_tiling));
    fusion_offsets_.emplace_back(fusion_offset);
  }
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, zero copy.
/// @param [in] mbuf_list: input/output mbuf addr list for input/output data.
/// @param [in] outside_addrs: model input/output memory addr
/// @param [in] is_no_tiling_list: model input/output is notiling
/// @param [out] cpy_args: cpy args
/// @return: 0 for success / others for failed
///
Status CpuTaskZeroCopy::Init(std::vector<uintptr_t> &mbuf_list,
                             const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
                             const std::vector<bool> &is_no_tiling_list,
                             ZeroCpyArgs &cpy_args) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  has_tensor_desc_ = cpy_args.has_tensor_desc;
  cpy_args.need_distribute = true;
  GE_CHK_STATUS_RET_NOLOG(InitAddrs(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args));
  GELOGI("addr_num is %u, outside_addrs size is %zu, is_no_tiling_list size is %zu",
         addr_num_, outside_addrs.size(), is_no_tiling_list.size());
  if (addr_num_ == 0U) {
    cpy_args.need_distribute = false;
    GELOGI("addr_num is 0, no need to distribute task");
    return SUCCESS;
  }

  // malloc mem for src_addrs/dst_addrs, and copy data of src_addrs/dst_addrs
  GE_CHK_RT_RET(rtMalloc(&src_addr_, src_addrs_.size() * sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(src_addr_, src_addrs_.size() * sizeof(uint64_t), src_addrs_.data(),
                         src_addrs_.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));

  GE_CHK_RT_RET(rtMalloc(&dst_addr_, dst_addrs_.size() * sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(dst_addr_, dst_addrs_.size() * sizeof(uint64_t), dst_addrs_.data(),
                         dst_addrs_.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));

  // src_addr_list is init to src_addr, which is the point to src_addrs
  const void *args = nullptr;
  AddrMapInfo addr_map_info;
  // AddrMapInfoV2 + AddrMapInfoV2.extend_info
  constexpr size_t buff_size = sizeof(AddrMapInfoV2) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t);
  std::vector<uint8_t> buff(buff_size, 0);
  if (!has_tensor_desc_) {
    addr_map_info.addr_num = addr_num_;
    addr_map_info.src_addr_list = PtrToValue(src_addr_);
    addr_map_info.dst_addr_list = PtrToValue(dst_addr_);
    args = &addr_map_info;
    args_size_ = static_cast<uint32_t>(sizeof(AddrMapInfo));
    GELOGI("src_addr_list is %" PRIu64 ", dst_addr_list is %" PRIu64,
      addr_map_info.src_addr_list, addr_map_info.dst_addr_list);
  } else {
    AddrMapInfoV2 *const addr_map_info_v2 = PtrToPtr<uint8_t, AddrMapInfoV2>(&buff[0]);
    GE_CHK_RT_RET(rtMalloc(&no_tiling_addr_, no_tilings_.size() * sizeof(int32_t), RT_MEMORY_HBM,
                           GE_MODULE_NAME_U16));
    GE_CHK_RT_RET(rtMemcpy(no_tiling_addr_, no_tilings_.size() * sizeof(int32_t), no_tilings_.data(),
                           no_tilings_.size() * sizeof(int32_t), RT_MEMCPY_HOST_TO_DEVICE));

    GE_CHK_RT_RET(rtMalloc(&dest_is_tiling_addr_, dest_is_tilings_.size() * sizeof(int32_t), RT_MEMORY_HBM,
                           GE_MODULE_NAME_U16));
    GE_CHK_RT_RET(rtMemcpy(dest_is_tiling_addr_, dest_is_tilings_.size() * sizeof(int32_t), dest_is_tilings_.data(),
                           dest_is_tilings_.size() * sizeof(int32_t), RT_MEMCPY_HOST_TO_DEVICE));
    GE_CHK_RT_RET(rtMalloc(&fusion_offsets_addr_, fusion_offsets_.size() * sizeof(int32_t), RT_MEMORY_HBM,
                           GE_MODULE_NAME_U16));
    GE_CHK_RT_RET(rtMemcpy(fusion_offsets_addr_, fusion_offsets_.size() * sizeof(int32_t), fusion_offsets_.data(),
                           fusion_offsets_.size() * sizeof(int32_t), RT_MEMCPY_HOST_TO_DEVICE));
    addr_map_info_v2->addr_num = addr_num_;
    addr_map_info_v2->src_addr_list = PtrToValue(src_addr_);
    addr_map_info_v2->dst_addr_list = PtrToValue(dst_addr_);
    addr_map_info_v2->is_no_tiling_list = PtrToValue(no_tiling_addr_);
    addr_map_info_v2->len = static_cast<uint32_t>(sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t));
    size_t extend_offset = 0U;
    uint32_t *const skip_size = PtrToPtr<char, uint32_t>(&addr_map_info_v2->extend_info[0]);
    *skip_size = static_cast<uint32_t>(sizeof(RuntimeTensorDesc));
    extend_offset += sizeof(uint32_t);
    uint64_t *const dest_is_tiling_list =
        PtrToPtr<char, uint64_t>(&addr_map_info_v2->extend_info[0] + extend_offset);
    *dest_is_tiling_list = PtrToValue(dest_is_tiling_addr_);
    extend_offset += sizeof(uint64_t);
    uint64_t *const fusion_offsets =
        PtrToPtr<char, uint64_t>(&addr_map_info_v2->extend_info[0] + extend_offset);
    *fusion_offsets = PtrToValue(fusion_offsets_addr_);
    args = PtrToPtr<uint8_t, void>(&buff[0]);
    args_size_ = static_cast<uint32_t>(sizeof(AddrMapInfoV2) + addr_map_info_v2->len);
    GELOGI("src_addr_list is %" PRIu64 ", dst_addr_list is %" PRIu64 ", "
           "is_no_tiling_list is %" PRIu64 ", len %u, args_size_ is %u",
           addr_map_info_v2->src_addr_list, addr_map_info_v2->dst_addr_list, addr_map_info_v2->is_no_tiling_list,
           addr_map_info_v2->len, args_size_);
  }

  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), args, static_cast<uint64_t>(args_size_),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status CpuTaskZeroCopy::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  const std::string zero_cpy_task = (has_tensor_desc_ ? kCpuTaskZeroCopyV2 : kCpuTaskZeroCopy);
  GE_CHK_RT_RET(LaunchCpuKernel(zero_cpy_task));

  GELOGI("Cpu kernel launch zero copy task success.");
  return SUCCESS;
}

CpuTaskZeroCopy::~CpuTaskZeroCopy() {
  GE_FREE_RT_LOG(src_addr_);
  GE_FREE_RT_LOG(dst_addr_);
  GE_FREE_RT_LOG(no_tiling_addr_);
  GE_FREE_RT_LOG(dest_is_tiling_addr_);
  GE_FREE_RT_LOG(fusion_offsets_addr_);
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] addr: NetOutput Op input tensor address.
/// @param [in] size: NetOutput Op input tensor size.
/// @param [in] in_mbuf: input mbuf addr for input data.
/// @param [out] out_mbuf: output mbuf addr for output data.
/// @return: 0 for success / others for failed
///
Status CpuTaskProcessOutput::Init(const uintptr_t addr, const uint32_t size, const uintptr_t in_mbuf,
                                  uintptr_t &out_mbuf, const InputOutputDescInfo *const output_desc) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ =
      static_cast<uint32_t>(sizeof(ProcessOutputInfo) + sizeof(uintptr_t));  // sizeof(uintptr_t) for save out_mbuf.
  if (output_desc != nullptr) {
    args_size_ += static_cast<uint32_t>(sizeof(RuntimeTensorDesc));
    GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    out_mbuf = PtrToValue(args_) + static_cast<uint64_t>(sizeof(ProcessOutputInfo)) +
               static_cast<uint64_t>(sizeof(RuntimeTensorDesc));
    RuntimeTensorDesc tensor_desc{};
    tensor_desc.dtype = static_cast<int64_t>(output_desc->data_type);
    GE_CHK_BOOL_RET_STATUS(static_cast<int64_t>(output_desc->shape_info.dims.size()) <= kMaxDimSize, FAILED,
                           "Shape dim size:%zu must less than max dim size:%" PRId64 ".",
                           output_desc->shape_info.dims.size(), kMaxDimSize);
    tensor_desc.shape[0] = static_cast<int64_t>(output_desc->shape_info.dims.size());
    tensor_desc.original_shape[0] = tensor_desc.shape[0];
    for (size_t i = 0UL; i < output_desc->shape_info.dims.size(); ++i) {
      tensor_desc.shape[i + 1UL] = output_desc->shape_info.dims[i];
      tensor_desc.original_shape[i + 1UL] = output_desc->shape_info.dims[i];
    }
    tensor_desc.data_size = static_cast<uint64_t>(size);
    GELOGD("Tensordesc type = %d, shape = original shape = %s, data size = %u", static_cast<int32_t>(tensor_desc.dtype),
           ToString(output_desc->shape_info.dims).c_str(), size);
    GE_CHK_RT_RET(rtMemcpy(ValueToPtr(PtrToValue(args_) + sizeof(ProcessOutputInfo)),
                           (static_cast<uint64_t>(args_size_) - static_cast<uint64_t>(sizeof(ProcessOutputInfo))),
                           &tensor_desc, static_cast<uint64_t>(sizeof(RuntimeTensorDesc)), RT_MEMCPY_HOST_TO_DEVICE));
  } else {
    GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    out_mbuf = PtrToValue(args_) + sizeof(ProcessOutputInfo);
    GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);
  }

  // Get NetOutput Input address and bind to queue.
  ProcessOutputInfo process;
  process.data_size = size;
  process.data_addr = addr;
  process.in_mbuf = in_mbuf;
  process.out_mbuf = out_mbuf;  // Placeholder, output mbuf addr will save to this place.
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &process, sizeof(ProcessOutputInfo),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status CpuTaskProcessOutput::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  std::string kernel_name;
  if (stage_ == ProcessStage::kPrepare) {
    kernel_name = kCpuTaskPrepareOutput;
    if (out_has_tensor_desc_) {
      kernel_name = kCpuTaskPrepareOutputWithTensorDesc;
    }
  } else if (stage_ == ProcessStage::kPostStatic) {
    kernel_name = kCpuTaskStaticOutputPostProcess;
    if (out_has_tensor_desc_) {
      kernel_name = kCpuTaskStaticOutputPostProcessWithTensorDesc;
    }
  } else {
    kernel_name = kCpuTaskDynOutputPostProcess;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(kernel_name));
  GELOGI("Cpu kernel launch %s output task success.", kernel_name.c_str());
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] queue_id: output queue id from user.
/// @param [in] out_mbuf: mbuf for output data.
/// @return: 0 for success / others for failed
///
Status CpuTaskModelEnqueue::Init(const uint32_t queue_id, const uintptr_t out_mbuf) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  // Get NetOutput Input address and bind to queue.
  args_size_ = sizeof(MbufQueueInfo);
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);

  MbufQueueInfo queue_info;
  queue_info.queue_id = queue_id;
  queue_info.in_mbuf = out_mbuf;
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &queue_info, static_cast<uint64_t>(args_size_),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status CpuTaskModelEnqueue::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_ is 0 or stream_ is nullptr, arg_size:%u,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskModelEnqueue));

  GELOGI("Cpu kernel launch model enqueue task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, active entry stream.
/// @param [in] stream: stream to be active.
/// @return: 0 for success / others for failed
///
Status CpuTaskActiveEntry::Init(aclrtStream const stream) {
  if (stream == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param stream is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Task active stream not valid");
    return FAILED;
  }

  active_stream_ = stream;
  return SUCCESS;
}

Status CpuTaskActiveEntry::Distribute() {
  if ((active_stream_ == nullptr) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param stream is nullptr or active_stream_ is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  GE_CHK_RT_RET(aclrtActiveStream(active_stream_, stream_));

  GELOGI("Cpu kernel launch active entry task success.");
  return SUCCESS;
}

Status CpuTaskMarkStep::Init(const GroupInfo &group_info, const std::string &dump_step,
                             const uintptr_t step_id_addr, bool is_head) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = static_cast<uint32_t>(sizeof(MarkStepInfo));
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);

  MarkStepInfo mark_step_info{};
  mark_step_info.group_total_count = group_info.group_total_count;
  mark_step_info.group_index = group_info.group_index;
  mark_step_info.group_policy = group_info.group_policy;
  mark_step_info.is_head = is_head ? 0U : 1U;
  const auto ret = strcpy_s(mark_step_info.dump_step, sizeof(mark_step_info.dump_step), dump_step.c_str());
  if (ret != EOK) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"dump_step", dump_step.c_str(), "Dump step is too long."}));
    GELOGE(FAILED, "[Call][strcpy_s] strcpy failed, result: %d, dump_step: %s", ret, dump_step.c_str());
    return FAILED;
  }

  void * const step_id = reinterpret_cast<void *>(step_id_addr);
  if (step_id != nullptr) {
    GE_CHK_RT_RET(rtMemset(step_id, sizeof(uint64_t), 0U, sizeof(uint64_t)));
  }
  mark_step_info.step_id_addr = step_id_addr;
  GELOGI("[MarkStep] group_total_count[%u], group_index[%u], step_id_addr: 0x%" PRIx64 ", dump_step: %s, is_head: %d.",
         mark_step_info.group_total_count, mark_step_info.group_index, mark_step_info.step_id_addr,
         mark_step_info.dump_step, static_cast<int32_t>(mark_step_info.is_head));

  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &mark_step_info, sizeof(MarkStepInfo),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status CpuTaskMarkStep::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0U) && (stream_ != nullptr),
                         FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskMarkStep));
  GELOGI("Cpu kernel launch mark dump step task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
/// @param [in] model_id: model id for wait end graph.
/// @return: 0 for success / others for failed
///
Status CpuTaskWaitEndGraph::Init(const uint32_t model_id) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(model_id);
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);

  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &model_id, static_cast<uint64_t>(args_size_),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status CpuTaskWaitEndGraph::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskWaitEndGraph));
  GELOGI("Cpu kernel launch wait end task success.");
  return SUCCESS;
}

Status CpuTaskModelReportStatus::Init(const uint32_t model_uuid,
                                      const QueueAttrs &status_output_queue,
                                      const std::vector<QueueAttrs> &input_queues) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }
  args_size_ = static_cast<uint32_t>(sizeof(ReportStatusInfo) + (sizeof(QueueAttrs) * input_queues.size()));
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);

  ReportStatusInfo report_status_info;
  report_status_info.model_uuid = model_uuid;
  report_status_info.status_output_queue = status_output_queue;
  report_status_info.input_num = static_cast<uint32_t>(input_queues.size());
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_),
                         &report_status_info, sizeof(ReportStatusInfo),
                         RT_MEMCPY_HOST_TO_DEVICE));
  QueueAttrs * const input_queues_ptr = PtrToPtr<void, QueueAttrs>(ValueToPtr(PtrToValue(args_) +
    sizeof(ReportStatusInfo)));
  GE_CHK_RT_RET(rtMemcpy(input_queues_ptr, static_cast<uint64_t>(args_size_ - sizeof(ReportStatusInfo)),
                         input_queues.data(), sizeof(QueueAttrs) * input_queues.size(),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status CpuTaskModelReportStatus::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskModelReportStatus));
  GELOGI("Cpu kernel launch repeat task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
/// @param [in] model_id: model id for repeat run.
/// @return: 0 for success / others for failed
///
Status CpuTaskModelRepeat::Init(const uint32_t model_id) {
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(model_id);
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);

  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &model_id, static_cast<uint64_t>(args_size_),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status CpuTaskModelRepeat::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskModelRepeat));
  GELOGI("Cpu kernel launch repeat task success.");
  return SUCCESS;
}

Status CpuTaskModelBatchDequeue::Init(const uint32_t align_interval,
                                      const std::vector<uint32_t> &queue_ids,
                                      const std::vector<uint32_t> &align_offsets,
                                      std::vector<uintptr_t> &in_mbufs) {
  GE_CHK_BOOL_RET_STATUS(args_ == nullptr,
                         FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
  GE_CHK_BOOL_RET_STATUS(queue_ids.size() == align_offsets.size(),
                         FAILED, "number of align_offsets mismatches that of queue ids ");
  const uint32_t num_inputs = static_cast<uint32_t>(queue_ids.size());
  // kernel_args|mbuf_addr_buffer|queue_ids_buffer|align_offsets_buffer
  BatchEnqueueKernelArgs kernel_args{};
  kernel_args.num_inputs = num_inputs;
  kernel_args.align_interval = align_interval;
  args_size_ = static_cast<uint32_t>(sizeof(kernel_args));
  const uint32_t mbuf_addrs_offset = args_size_;
  const size_t mbuf_addrs_size = sizeof(uint64_t) * num_inputs;
  args_size_+= static_cast<uint32_t>(mbuf_addrs_size);
  const uint32_t queue_ids_offset = args_size_;
  const size_t queue_ids_size = sizeof(uint32_t) * num_inputs;
  args_size_+= static_cast<uint32_t>(queue_ids_size);
  const uint32_t align_offsets_offset = args_size_;
  const size_t align_offsets_size = sizeof(uint32_t) * num_inputs;
  args_size_+= static_cast<uint32_t>(sizeof(uint32_t) * align_offsets_size);
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  kernel_args.align_offsets_addr = PtrToValue(args_) + align_offsets_offset;
  kernel_args.queue_ids_addr = PtrToValue(args_) + queue_ids_offset;
  kernel_args.mbuf_addrs_addr = PtrToValue(args_) + mbuf_addrs_offset;
  for (uint32_t i = 0U; i < num_inputs; ++i) {
    in_mbufs.emplace_back(kernel_args.mbuf_addrs_addr + sizeof(uintptr_t) * i);
  }

  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);
  GE_CHK_RT_RET(rtMemcpy(args_, args_size_,
                         &kernel_args, sizeof(kernel_args), RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.align_offsets_addr), align_offsets_size,
                         align_offsets.data(), align_offsets_size, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.queue_ids_addr), queue_ids_size,
                         queue_ids.data(), queue_ids_size, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.mbuf_addrs_addr), mbuf_addrs_size,
                         in_mbufs.data(), mbuf_addrs_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status CpuTaskModelBatchDequeue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0U) && (stream_ != nullptr),
                         FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskModelBatchDequeue));
  GELOGI("Cpu kernel launch model batch dequeue task success.");
  return SUCCESS;
}

Status CpuTaskModelGatherDequeue::Init(const std::vector<QueueAttrs> &queues,
                                       const InputAlignAttrs &align_attrs,
                                       std::vector<uintptr_t> &in_mbufs) {
  GE_CHK_BOOL_RET_STATUS(args_ == nullptr, FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
  const auto queue_num = queues.size();
  // memory layout: kernel_args | queue_ids_addr | mbufs_buffer_ptr_addr |
  //                queue_device_buff | queue_device_type_buff | mbuff_buffs_ptrs
  GatherDequeueKernelArgs kernel_args{};
  args_size_ = static_cast<uint32_t>(sizeof(kernel_args));
  kernel_args.input_nums = static_cast<uint32_t>(queue_num);
  kernel_args.inputs_align_timeout = align_attrs.align_timeout;
  kernel_args.inputs_align_max_cache_num = align_attrs.align_max_cache_num;
  kernel_args.inputs_align_drop_out = static_cast<uint32_t>(align_attrs.drop_when_not_align);

  const uint64_t queue_ids_offset = args_size_;
  const size_t queue_id_addrs_size = sizeof(uint32_t) * queue_num;
  args_size_ += static_cast<uint32_t>(queue_id_addrs_size);

  const uint64_t mbuf_addrs_offset = args_size_;
  const uint64_t mbuf_addrs_size = static_cast<uint64_t>(sizeof(uint64_t) * queue_num);
  args_size_ += static_cast<uint32_t>(mbuf_addrs_size);

  const uint64_t device_ids_offset = args_size_;
  const size_t device_ids_size = sizeof(uint32_t) * queue_num;
  args_size_ += static_cast<uint32_t>(device_ids_size);

  const uint64_t device_type_offset = args_size_;
  const size_t device_type_size = sizeof(uint32_t) * queue_num;
  args_size_ += static_cast<uint32_t>(device_type_size);

  const uint64_t mbufs_offset = args_size_;
  const size_t mbuff_size = sizeof(uint64_t) * queue_num;
  args_size_ += static_cast<uint32_t>(mbuff_size);

  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);
  kernel_args.queue_ids_addr = PtrToValue(args_) + queue_ids_offset;
  kernel_args.mbuf_addrs_addr = PtrToValue(args_) + mbuf_addrs_offset;
  kernel_args.queue_device_ids_addr = PtrToValue(args_) + device_ids_offset;
  kernel_args.queue_device_type_addr = PtrToValue(args_) + device_type_offset;
  const uint64_t mbufs_addr = PtrToValue(args_) + mbufs_offset;
  std::vector<uint32_t> queue_ids;
  std::vector<uint32_t> device_ids;
  std::vector<uint32_t> device_types;
  
  for (size_t i = 0UL; i < queue_num; ++i) {
    in_mbufs.emplace_back(mbufs_addr + static_cast<uint64_t>(sizeof(uint64_t)) * static_cast<uint64_t>(i));
    queue_ids.emplace_back(queues[i].queue_id);
    device_ids.emplace_back(queues[i].device_id);
    device_types.emplace_back(queues[i].device_type);
  }
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_),
                         &kernel_args, sizeof(kernel_args), RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.queue_ids_addr), queue_id_addrs_size,
                         queue_ids.data(), queue_id_addrs_size, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.mbuf_addrs_addr), mbuf_addrs_size,
                         in_mbufs.data(), in_mbufs.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.queue_device_ids_addr), device_ids_size,
                         device_ids.data(), device_ids_size, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(kernel_args.queue_device_type_addr), device_type_size,
                         device_types.data(), device_type_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status CpuTaskModelGatherDequeue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0U) && (stream_ != nullptr),
                         FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kCpuTaskModelGatherDequeue));
  GELOGI("Cpu kernel launch model gather dequeue task success.");
  return SUCCESS;
}

Status CpuTaskProcessInputsMemCopy::Init(const std::vector<uintptr_t> &mbuf_list,
                                         const std::vector<uintptr_t> &data_addr_list,
                                         const std::vector<uint64_t> &length_list,
                                         const std::vector<int32_t> &input_fusion_offset_list) {
  // check params
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }
  if (mbuf_list.empty() || data_addr_list.empty() || length_list.empty()) {
    GELOGI("[Check][Param] Params of CpuTaskProcessInputsMemCopy task is emtpy.");
    return SUCCESS;
  }
  if ((mbuf_list.size() != data_addr_list.size()) ||
      (length_list.size() != data_addr_list.size()) ||
      (input_fusion_offset_list.size() != data_addr_list.size())) {
    GELOGE(FAILED, "[Check][Param] The size of mubf list:%zu, data addr list:%zu, "
           "length list:%zu and input fusion offset list:%zu should be same",
           mbuf_list.size(), data_addr_list.size(), length_list.size(), input_fusion_offset_list.size());
    return FAILED;
  }

  // construct InputCopyAddrMapInfo and copy data to device
  GE_CHK_RT_RET(rtMalloc(&src_addr_, mbuf_list.size() * sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(src_addr_, mbuf_list.size() * sizeof(uint64_t), mbuf_list.data(),
                         mbuf_list.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));

  GE_CHK_RT_RET(rtMalloc(&dst_addr_, data_addr_list.size() * sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(dst_addr_, data_addr_list.size() * sizeof(uint64_t), data_addr_list.data(),
                         data_addr_list.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));

  GE_CHK_RT_RET(rtMalloc(&len_list_, length_list.size() * sizeof(uint64_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(len_list_, length_list.size() * sizeof(uint64_t), length_list.data(),
                         length_list.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMalloc(&input_fusion_offset_list_,
                         input_fusion_offset_list.size() * sizeof(int32_t),
                         RT_MEMORY_HBM,
                         GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(input_fusion_offset_list_,
                         input_fusion_offset_list.size() * sizeof(int32_t),
                         input_fusion_offset_list.data(),
                         input_fusion_offset_list.size() * sizeof(int32_t),
                         RT_MEMCPY_HOST_TO_DEVICE));
  InputCopyAddrMapInfo addr_map_info;
  addr_map_info.addr_num = static_cast<uint32_t>(mbuf_list.size());
  addr_map_info.src_addr_list = PtrToValue(src_addr_);
  addr_map_info.dst_addr_list = PtrToValue(dst_addr_);
  addr_map_info.data_len_list = PtrToValue(len_list_);
  addr_map_info.input_fusion_offset_list = PtrToValue(input_fusion_offset_list_);
  args_size_ = static_cast<uint32_t>(sizeof(InputCopyAddrMapInfo));
  GELOGI("src_addr_list is 0x%" PRIx64 ", dst_addr_list is 0x%" PRIx64 ", data_len_addr is 0x%" PRIx64,
         addr_map_info.src_addr_list, addr_map_info.dst_addr_list, addr_map_info.data_len_list);
  GE_CHK_RT_RET(rtMalloc(&args_, static_cast<uint64_t>(args_size_), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_);
  GE_CHK_RT_RET(rtMemcpy(args_, static_cast<uint64_t>(args_size_), &addr_map_info, static_cast<uint64_t>(args_size_),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status CpuTaskProcessInputsMemCopy::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(KCpuTaskMemCopyInput));
  GELOGI("Cpu kernel launch input memory copy task success.");
  return SUCCESS;
}

CpuTaskProcessInputsMemCopy::~CpuTaskProcessInputsMemCopy() {
  GE_FREE_RT_LOG(src_addr_);
  GE_FREE_RT_LOG(dst_addr_);
  GE_FREE_RT_LOG(len_list_);
  GE_FREE_RT_LOG(input_fusion_offset_list_);
}

Status CpuTaskProcessInputsShapeCheck::Init(const std::vector<uintptr_t> &mbuf_list,
                                            const std::vector<int32_t> &input_fusion_offset_list) {
  // check params
  if ((args_ != nullptr) || (args_size_ > 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }
  if (mbuf_list.empty()) {
    GELOGI("[Check][Param] Params of CpuTaskProcessInputsMemCopy task is emtpy.");
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS((mbuf_list.size() == input_fusion_offset_list.size()), FAILED,
                         "[Check][Param] The size of mubf list:%zu, input fusion offset list:%zu should be same",
                         mbuf_list.size(), input_fusion_offset_list.size());
  std::vector<ShapeValidation> shape_validation;
  for (size_t i = 0U; i < mbuf_list.size(); i++) {
    ShapeValidation validation = {};
    validation.mbuf_addr = static_cast<uint64_t>(mbuf_list[i]);
    validation.offset = static_cast<uint64_t>(input_fusion_offset_list[i]);
    shape_validation.emplace_back(validation);
  }
  GE_CHK_RT_RET(rtMalloc(&shape_validation_addr_, sizeof(ShapeValidation) * shape_validation.size(), RT_MEMORY_HBM,
                         GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(shape_validation_addr_, sizeof(ShapeValidation) * shape_validation.size(),
                         shape_validation.data(), sizeof(ShapeValidation) * shape_validation.size(),
                         RT_MEMCPY_HOST_TO_DEVICE));
  ShapeValidationInfo shape_validation_info = {};
  shape_validation_info.validation_num = shape_validation.size();
  shape_validation_info.validation_info_device_addr = PtrToValue(shape_validation_addr_);
  GELOGI("Addr of shape validation info is 0x%" PRIx64 ".", shape_validation_addr_);
  GE_CHK_RT_RET(rtMalloc(&args_, sizeof(ShapeValidationInfo), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(args_, sizeof(ShapeValidationInfo), &shape_validation_info, sizeof(ShapeValidationInfo),
                         RT_MEMCPY_HOST_TO_DEVICE));
  args_size_ = static_cast<uint32_t>(sizeof(ShapeValidationInfo));
  return SUCCESS;
}

Status CpuTaskProcessInputsShapeCheck::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0U) || (stream_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid",
                       args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }
  GE_CHK_RT_RET(LaunchCpuKernel(KCpuTaskCheckInputTensorDesc));
  GELOGI("Cpu kernel launch input shape check task success.");
  return SUCCESS;
}

CpuTaskProcessInputsShapeCheck::~CpuTaskProcessInputsShapeCheck() {
  GE_FREE_RT_LOG(shape_validation_addr_);
}

Status CpuTaskInfo::LaunchCpuKernel(const std::string &kernel_name) const {
  rtArgsEx_t args_info = {};
  args_info.args = args_;
  args_info.argsSize = args_size_;
  args_info.isNoNeedH2DCopy = 1U;
  GE_CHK_RT_RET(rtCpuKernelLaunchWithFlag(nullptr,
      kernel_name.data(), kCoreDim, &args_info, nullptr, stream_, RT_KERNEL_DEFAULT));
  return SUCCESS;
}
}  // namespace ge
