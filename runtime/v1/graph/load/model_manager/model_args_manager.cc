/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/model_args_manager.h"

#include <numeric>

#include "common/checker.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/runtime_api_wrapper.h"
#include "common/utils/executor_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/ge_context.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "memory_app_type_classifier.h"
#include "model_args_layout_planner.h"
#include "runtime/mem.h"
#include "task_args_refresh_type_classifier.h"
#include "task_node_map.h"

namespace {
  constexpr uint32_t k32BitsMask = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111
  constexpr uint64_t kUnknown = 0U;
  constexpr uint64_t kSupport = 1UL << 0U;
  constexpr uint64_t kNoSupport = 1UL << 1U;
  constexpr uint32_t kKiloByte = 1024U;
  constexpr uint32_t kTilingThreshold1 = 96U;
  constexpr uint32_t kTilingThreshold2 = 4096U;
  constexpr uint32_t kTilingFactor1 = 8U;
  constexpr uint32_t kTilingFactor2 = 2U;
  constexpr uint32_t kTilingFactor3 = 6U;
  constexpr uint32_t kAlign256B = 64;
  constexpr uint32_t kUBLen = 183 * 1024;
  constexpr uint32_t kRtsLitePcieBarCopySize = 1024U;
  constexpr uint32_t kKernelLaunchArgOffset2 = 16;
  constexpr uint32_t kBufferNum = 2;
  constexpr uint32_t kBufferFactor = 31;
  constexpr uint32_t kUpdateVersionH2d = 2;
  constexpr uint32_t kUpdateVersionKernelLaunch = 3;
  constexpr const ge::char_t *kCoreTypeAIV = "AIV";
  const std::string kAddrRefreshOpName = "UpdateModelParam_static_bin";
  const std::string kAddrRefreshOpType = "Data";
  constexpr uint32_t kModelLoadStage = 0;
}
namespace ge {
rtMemType_t GetRtsMemoryType(const ArgsPlacement placement, const int64_t size) {
  switch (placement) {
    case ArgsPlacement::kArgsPlacementHbm:
      return RT_MEMORY_HBM;
    case ArgsPlacement::kArgsPlacementTs: {
      if (!IntegerChecker<uint32_t>::Compat(size)) {
        return std::numeric_limits<rtMemType_t>::max();
      }
      const auto mem_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, static_cast<uint32_t>(size));
      GELOGI("TS memory_type: %u, size %lld", mem_type, size);
      return mem_type;
    }
    case ArgsPlacement::kArgsPlacementHostSvm: {
      return RT_MEMORY_HOST_SVM;
    }
    default:
      GELOGE(INTERNAL_ERROR, "Unexpected args placement %d", static_cast<int32_t>(placement));
      return std::numeric_limits<rtMemType_t>::max();
  }
}

namespace {
Status PlanFixedMemoryLayout(const TaskNodeMap &task_node_map,
                             const TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs, int64_t &total_len,
                             std::vector<int64_t> &offsets) {
  offsets.resize(fixed_addrs.size());
  for (size_t i = 0U; i < fixed_addrs.size(); ++i) {
    offsets[i] = total_len;

    const auto &fixed_addr = fixed_addrs[i].at(0);
    auto &node_info = task_node_map.FindNodeByTaskIndex(fixed_addr.task_index);
    GE_ASSERT_TRUE(node_info.node_id != -1);  // 这里不可能为-1，因为在TaskArgsRefreshTypeClassifier做过了一次检查
    const auto op_desc = node_info.node->GetOpDesc();

    switch (fixed_addr.iow_index_type) {
      case TaskArgsRefreshTypeClassifier::kInput: {
        const auto td = op_desc->GetInputDescPtr(static_cast<uint32_t>(fixed_addr.iow_index));
        GE_ASSERT_NOTNULL(td, "Failed to calculate fixed address for task %zu, op %s, null input, index %zu",
                          fixed_addr.task_index, op_desc->GetName().c_str(), fixed_addr.iow_index);
        int64_t size{0};
        GE_ASSERT_GRAPH_SUCCESS(TensorUtils::GetTensorMemorySizeInBytes(*td, size));
        GE_ASSERT_TRUE(!AddOverflow(total_len, size, total_len));
        break;
      }
      case TaskArgsRefreshTypeClassifier::kOutput: {
        const auto td = op_desc->GetOutputDescPtr(static_cast<uint32_t>(fixed_addr.iow_index));
        GE_ASSERT_NOTNULL(td, "Failed to calculate fixed address for task %zu, op %s, null output, index %zu",
                          fixed_addr.task_index, op_desc->GetName().c_str(), fixed_addr.iow_index);
        int64_t size{0};
        GE_ASSERT_GRAPH_SUCCESS(TensorUtils::GetTensorMemorySizeInBytes(*td, size));
        GE_ASSERT_TRUE(!AddOverflow(total_len, size, total_len));
        break;
      }
      case TaskArgsRefreshTypeClassifier::kWorkspace: {
        auto ws_sizes = op_desc->GetWorkspaceBytes();
        GE_ASSERT_TRUE(
            fixed_addr.iow_index < ws_sizes.size(),
            "Failed to calculate fixed address for task %zu, op %s, workspace index out of range %zu, max %zu",
            fixed_addr.task_index, op_desc->GetName().c_str(), fixed_addr.iow_index, ws_sizes.size());
        GE_ASSERT_TRUE(!AddOverflow(total_len, ws_sizes.at(fixed_addr.iow_index), total_len));
        break;
      }
      default:
        GELOGE(INTERNAL_ERROR, "Failed to calculate fixed address for task %zu, op %s, unexpected iow type %d",
               fixed_addr.task_index, op_desc->GetName().c_str(), static_cast<int32_t>(fixed_addr.iow_index_type));
        return FAILED;
    }
  }
  return SUCCESS;
}
void DebugLogTaskRunParam(const size_t task_index, const int64_t op_index, const TaskRunParam &param,
                          const OpDescPtr &op_desc) {
  std::stringstream ss;
  ss << "Task index " << task_index << " op index " << op_index << ", args num " << param.args_descs.size() << ',';
  if (!param.args_descs.empty()) {
    ss << " len/placement: ";
    for (const auto &args_desc : param.args_descs) {
      ss << args_desc.args_len << '/' << GetArgsPlacementStr(args_desc.placement) << ',';
    }
  }

  ss << " inputs num " << param.parsed_input_addrs.size() << ','
      << " outputs num " << param.parsed_output_addrs.size() << ','
      << " workspaces num " << param.parsed_workspace_addrs.size() << ','
      << " persistent workspaces num " << param.persistent_workspace_descs.size() << ',';
  if (!param.persistent_workspace_descs.empty()) {
    ss << " len/placement: ";
    for (const auto &pw_desc : param.persistent_workspace_descs) {
      ss << pw_desc.args_len << '/' << GetArgsPlacementStr(pw_desc.placement) << ',';
    }
  }

  if (op_desc != nullptr) {
    ss << " op type " << op_desc->GetType().c_str() << ',' << " op name "<< op_desc->GetName().c_str() << '.';
  }
  GELOGD("%s", ss.str().c_str());
}
constexpr const char *kUpdatePolicyStr[ModelArgsManager::kUpdatePolicyEnd + 1] = {
    "no_need_update",   // kNoNeedUpdate
    "host_input",       // KUpdateHostInput
    "model-io",         // kUpdateModelIo
    "fm-and-model-io",  // kUpdateFmAndModelIo
    "all-one-time",     // kInitOneTime
    "unknown"};
const char *GetUpdatePolicyStr(ModelArgsManager::UpdatePolicy up) {
  if (up > ModelArgsManager::kUpdatePolicyEnd) {
    up = ModelArgsManager::kUpdatePolicyEnd;
  }
  return kUpdatePolicyStr[up];
}
void UseMin(uint64_t new_dev_addr, void *new_host_addr, uint64_t &dev_addr, void *&host_addr) {
  if (dev_addr > new_dev_addr) {
    dev_addr = new_dev_addr;
    host_addr = new_host_addr;
  }
}
Status GetOverlapRange(const std::pair<uint64_t, uint64_t> range1,
                       const std::pair<uint64_t, uint64_t> range2,
                       std::pair<uint64_t, uint64_t> &overlap_range) {
    overlap_range.second = std::min(range1.second, range2.second);  // [va, va + len)
    overlap_range.first = std::max(range1.first, range2.first);
    if (overlap_range.first >= overlap_range.second) {
      return FAILED;
    }

    return SUCCESS;
}
uint32_t MathCeil(uint32_t num1, uint32_t num2)
{
    return (num2 == 0) ? num1 : ((num1 + num2 - 1) / num2);
}
uint32_t MathFloor(uint32_t num1, uint32_t num2)
{
    return (num2 == 0) ? num1 : (num1 / num2);
}
uint32_t AlignUp(uint32_t num1, uint32_t num2)
{
    return MathCeil(num1, num2) * num2;
}
uint32_t AlignDown(uint32_t num1, uint32_t num2)
{
    return MathFloor(num1, num2) * num2;
}
Status GetPlatformVectorNum(uint32_t &vec_core_num) {
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos optional_info;
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_info) ==
                 SUCCESS, "Get platform failed.");
  platform_infos.SetCoreNumByCoreType(kCoreTypeAIV);
  vec_core_num = platform_infos.GetCoreNum();
  GE_ASSERT_TRUE(vec_core_num != 0U, "Vector num:%u is invalid.", vec_core_num);
  return SUCCESS;
}
Status CalculateBlockDim(uint32_t index_len, uint32_t &block_dim)
{
  uint32_t vec_core_num = 0;
  GE_ASSERT_TRUE(GetPlatformVectorNum(vec_core_num) == SUCCESS, "GetPlatformVectorNum failed.");
  uint32_t k_total_len = index_len / kKiloByte; // 计算有多少K的数据，以Byte为单位
  if (k_total_len <= kTilingThreshold1) {
      block_dim = std::max(k_total_len / kTilingFactor1, 1U);
  } else if (k_total_len <= kTilingThreshold2) {
      block_dim = std::lround(kTilingFactor2 * std::log2(static_cast<double>(k_total_len)));
  } else {
      block_dim = std::min(k_total_len * kTilingFactor3 / kKiloByte, static_cast<uint32_t>(vec_core_num));
  }
  return SUCCESS;
}
}  // namespace

// todo variable直连输出的场景，会把输出影响为不可刷新
ModelArgsManager::~ModelArgsManager() noexcept = default;

Status ModelArgsManager::Init(domi::ModelTaskDef &model_task_def, std::vector<TaskInfoPtr> *task_list_ptr) {
  GE_ASSERT_NOTNULL(task_list_ptr);
  task_list_ptr_ = task_list_ptr;
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  return InitTaskInfoV2(model_task_def);
}

Status ModelArgsManager::GenModelArgsRefreshInfosForTask(std::vector<TaskArgsRefreshInfo> &infos,
                                                         PisToArgs &pls_to_args, const NodePtr &node) {
  for (const auto &info : infos) {
    ModelArgsRefreshInfo m_info;
    const size_t pls = static_cast<size_t>(info.placement);
    m_info.id = info.id;
    m_info.offset = info.offset;
    GE_ASSERT_TRUE(info.placement < ArgsPlacement::kEnd);
    GE_ASSERT_TRUE(info.args_offset < static_cast<uint64_t>(pls_to_args[pls].len),
        "op_name:%s, op_type:%s, args offset:%" PRIu64 " is more than pls:%zu, len:%d, task args refresh info:[%s]",
        node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str(),
        info.args_offset, pls, pls_to_args[pls].len, info.ToString().c_str());
    GE_ASSERT_TRUE(pls_to_args[pls].host_addr != nullptr);
    m_info.host_args_addr = ValueToPtr(PtrToValue(pls_to_args[pls].host_addr) + info.args_offset);
    m_info.device_args_addr = pls_to_args[pls].dev_addr + info.args_offset;
    GELOGI("[Args][Init] op_name:%s, op_type:%s, pls:%zu, pls host addr:0x%llx, pls dev addr:0x%llx, "
      "task args refresh info:[%s], after transfer, model args refresh info:[%s].",
      node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str(), pls,
      PtrToValue(pls_to_args[pls].host_addr), pls_to_args[pls].dev_addr,
      info.ToString().c_str(), m_info.ToString().c_str());
    if (info.args_format_policy == ArgsFormatPolicy::kAddrAll) {
      allocation_ids_to_model_args_refresh_infos_addr_all[m_info.id].emplace_back(std::move(m_info));
    } else if (info.args_format_policy == ArgsFormatPolicy::kAddrLow32Bit) {
      allocation_ids_to_model_args_refresh_infos_addr_low_32bit[m_info.id].emplace_back(std::move(m_info));
    } else if (info.args_format_policy == ArgsFormatPolicy::kAddrHigh32Bit) {
      allocation_ids_to_model_args_refresh_infos_addr_high_32bit[m_info.id].emplace_back(std::move(m_info));
    }
  }

  return SUCCESS;
}

Status ModelArgsManager::GenAllocationToIowPaRemapInfos(TaskInfoPtr task_info,
                                                        const NodePtr &node,
                                                        std::vector<IowPaRemapInfo> pa_remap_infos) {
  for (auto &info : pa_remap_infos) {
    info.task_info = task_info.get();
    info.op_name = node->GetOpDesc()->GetName();
    allocation_ids_to_iow_pa_remap_infos_[info.allocation_id].insert(info);
    GELOGI("Iow pa remap info:[%s].", info.ToString().c_str());
  }
  return SUCCESS;
}

Status ModelArgsManager::PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len,
                                    std::vector<std::pair<uint64_t, uint64_t>> &overlap_range) {
  (void)new_pa;
  std::pair<uint64_t, uint64_t> va_range(va, va + len); // [va, va + len)
  pa_remap_match_support_num_ = 0UL;
  pa_remap_match_nosupport_num_ = 0UL;
  uint64_t flag = kUnknown;
  GE_ASSERT_TRUE(((last_bases_.size()) > 0U && (id_to_len_.size() == last_bases_.size())),
    "len list size %zu, base list size %zu", id_to_len_.size(), last_bases_.size());
  const size_t active_mem_base_len = last_bases_.size();
  auto active_mem_base_addr = GetActivateMemBaseAddrs();
  GE_ASSERT_NOTNULL(active_mem_base_addr);
  for (size_t i = 0; i < active_mem_base_len; i++) {
    if (last_bases_[i] != active_mem_base_addr[i]) {
      last_bases_[i] = active_mem_base_addr[i];
    }
  }
  size_t absolute_mem_id = id_to_len_.size() - 1U;
  for (size_t id = 0U; id < id_to_len_.size(); id++) {
    std::pair<uint64_t, uint64_t> allocation_range(last_bases_[id], last_bases_[id] + id_to_len_[id]); // 左闭右开
    std::pair<uint64_t, uint64_t> allocation_and_va_overlap_range;
    if (GetOverlapRange(va_range, allocation_range, allocation_and_va_overlap_range) != SUCCESS) {
      continue;
    }

    std::pair<uint64_t, uint64_t> allocation_and_va_overlap_offset_range(
      allocation_and_va_overlap_range.first - last_bases_[id],
      allocation_and_va_overlap_range.second - last_bases_[id]);

    IowPaRemapInfo pa_remap_info {};
    pa_remap_info.allocation_offset = allocation_and_va_overlap_offset_range.first;
    auto it = allocation_ids_to_iow_pa_remap_infos_[id].upper_bound(pa_remap_info);
    for (; it != allocation_ids_to_iow_pa_remap_infos_[id].end(); it++) {
      if (it->allocation_offset >= allocation_and_va_overlap_offset_range.second) {
        break;
      }

      std::pair<uint64_t, uint64_t> tensor_offset_range(it->allocation_offset, it->allocation_offset + it->tensor_size);
      std::pair<uint64_t, uint64_t> tensor_and_va_offset_overlap_range;
      if (GetOverlapRange(allocation_and_va_overlap_offset_range,
        tensor_offset_range, tensor_and_va_offset_overlap_range) != SUCCESS) {
        continue;
      }

      if (it->policy != PaRemapPolicy::KSupport) {
        flag |= kNoSupport;
        pa_remap_match_nosupport_num_++;
        GELOGI("Iow no support remap, active mem base:[0x%" PRIx64 "], len:[0x%" PRIx64 "], task info:[%s], "
          "va:[0x%" PRIx64 "], va len:[0x%" PRIx64 "],"
          "overlap_range start:[0x%" PRIx64 "], overlap_range end:[0x%" PRIx64 "]", last_bases_[id], id_to_len_[id],
          it->ToString().c_str(), va, len, last_bases_[id] + tensor_and_va_offset_overlap_range.first,
          last_bases_[id] + tensor_and_va_offset_overlap_range.second - 1U);
      }

      // absolute段，包含const var，保存交叉的tensor记录，当前需求只支持fm/io段，和absolute段应无交叉
      if (id == absolute_mem_id) {
        overlap_range.emplace_back(
          std::pair<uint64_t, uint64_t>(last_bases_[id] + tensor_and_va_offset_overlap_range.first,
            last_bases_[id] + tensor_and_va_offset_overlap_range.second - 1U)); // [lower, upper - 1U]
      }
    }

    if (id == absolute_mem_id) {
      break;
    }

    // 非absolute 段，如果地址已经分配给GE，但是并未使用, 识别为可恢复
    if (flag == kUnknown) {
      flag |= kSupport;
      pa_remap_match_support_num_++;
      GELOGI("Iow support remap, active mem base:[0x%" PRIx64 "], len:[0x%" PRIx64 "], allocation id:[%u], "
        "va:[0x%" PRIx64 "], va len:[0x%" PRIx64 "], "
        "overlap_range start:[0x%" PRIx64 "], overlap_range end:[0x%" PRIx64 "].",
        last_bases_[id], id_to_len_[id], id, va, len,
        allocation_and_va_overlap_range.first, allocation_and_va_overlap_range.second - 1U);
    }

    // 非absolute 段，记录交叉的段区间
    overlap_range.emplace_back(std::pair<uint64_t, uint64_t>(
      allocation_and_va_overlap_range.first, allocation_and_va_overlap_range.second - 1U)); // [lower, upper - 1U]
  }

  if (flag == kUnknown) {  // 全部是未识别
    GELOGI("va unkown, va:[0x%" PRIx64 "], va len:[0x%" PRIx64 "].", va, len);
    return PARAM_INVALID;
  } else if ((flag & (kNoSupport)) == kNoSupport) { // 只要有一个不支持remap，返回失败
    GELOGI("va no support remap, match support num %" PRIu64 ", match no support num %" PRIu64 ".",
      pa_remap_match_support_num_, pa_remap_match_nosupport_num_);
    return FAILED;
  } else {
    // 1)所有都支持remap 2) 部分支持+部分未识别，这两种场景未做区分，session层面根据返回的交叉区间区分是否存在未识别场景
    GELOGI("va support remap, match support num %" PRIu64 ", match no support num %" PRIu64 ".",
      pa_remap_match_support_num_, pa_remap_match_nosupport_num_);
    return SUCCESS;
  }
}

Status ModelArgsManager::CalculateUpdateModelParamTiling(uint32_t active_base_len, uint32_t index_len,
    uint32_t &block_dim, UpdateModelParamTilingData &tiling) const {
  GE_ASSERT_TRUE(CalculateBlockDim(index_len, block_dim) == SUCCESS, "CalculateBlockDim failed.");
  tiling.totalActiveBaseTblCnt = AlignUp(active_base_len, kAlign32B) / sizeof(uint32_t);
  uint32_t total_cnt = index_len / sizeof(uint32_t);
  uint32_t total_buffer_len = kUBLen - active_base_len * kBufferNum;
  /* 每一轮最多计算的uint32_t数据个数 */
  uint32_t max_tile_cnt = AlignDown(total_buffer_len / kBufferFactor, kAlign256B); // 每轮计算256字节 即64个uint32_t

  tiling.blockCnt = AlignUp(MathCeil(total_cnt, block_dim), kAlign256B);
  block_dim = MathCeil(total_cnt, tiling.blockCnt);

  uint32_t last_block_cnt_ori = total_cnt - tiling.blockCnt * (block_dim - 1);
  uint32_t last_block_cnt = AlignUp(total_cnt - tiling.blockCnt * (block_dim - 1), kAlign256B);
  tiling.tileNum = static_cast<uint16_t>(MathCeil(tiling.blockCnt, max_tile_cnt));
  tiling.tileCnt = AlignUp(MathCeil(tiling.blockCnt, tiling.tileNum), kAlign256B);
  tiling.lastTileNum = static_cast<uint16_t>(MathCeil(last_block_cnt, tiling.tileCnt));

  tiling.tailCnt = tiling.blockCnt - tiling.tileCnt * (tiling.tileNum - 1U);
  tiling.lastTailCnt = last_block_cnt - tiling.tileCnt * (tiling.lastTileNum - 1U);
  tiling.lastTailCntOri = last_block_cnt_ori - tiling.tileCnt * (tiling.lastTileNum - 1U);

  return SUCCESS;
}

Status ModelArgsManager::GenAddrRefreshIndexAndOffset(const uint64_t &offset_num) {
  uint64_t host_input_copy_num =
    (host_input_size_ > 0U && is_pcie_bar_copy_) ? MathCeil(host_input_size_, sizeof(uint64_t)) : 0U;
  uint64_t args_offset_num = offset_num + host_input_copy_num;

  // 申请offset 和 index表的host地址
  // index为uint32类型，offset为uint64，在算子侧统一uint32计算，因此index需要乘2获取正确的offset地址
  uint8_t index_mutiples = sizeof(uint64_t) / sizeof(uint32_t);
  auto model_args_host_offset = ge::MakeUnique<uint64_t[]>(static_cast<size_t>(args_offset_num));
  auto model_args_host_index = ge::MakeUnique<uint32_t[]>(index_mutiples * static_cast<size_t>(args_offset_num));

  // offset的device表
  model_args_device_offset_ = davinci_model_->MallocDynamicMemory(static_cast<size_t>(args_offset_num) * sizeof(uint64_t));
  GE_ASSERT_NOTNULL(model_args_device_offset_);

  // index的device表
  model_args_device_index_ = davinci_model_->MallocDynamicMemory(static_cast<size_t>(args_offset_num) * sizeof(uint64_t));
  GE_ASSERT_NOTNULL(model_args_device_index_);

  // 算子的workspace参数
  workspace_addr_device_ = davinci_model_->MallocDynamicMemory(sizeof(uint64_t));
  GE_ASSERT_NOTNULL(workspace_addr_device_);
  std::set<uint64_t> io_index_set;

  // index和offset的host表赋值，可刷新部分赋值
  for (auto item : allocation_ids_to_model_args_refresh_infos_addr_all) {
    for (const auto &info : item) {
      if (info.device_args_addr % sizeof(uint64_t) != 0) {
        GELOGW("info.device_args_addr %llu is not uint64 offset.", info.device_args_addr);
        return FAILED;
      }
      uint64_t io_index =
          (info.device_args_addr - model_args_[0].model_args_device_addr) / sizeof(uint64_t);
      if (io_index <= offset_num) {
        model_args_host_offset[io_index] = info.offset;
        // index为uint32_t类型，且在算子侧，以uint8_t进行计算，也因此需要转换为uint8_t的index并压栈两次
        model_args_host_index[index_mutiples * io_index] = index_mutiples * info.id * sizeof(uint32_t);
        model_args_host_index[index_mutiples * io_index + 1] = (index_mutiples * info.id + 1) * sizeof(uint32_t);
        io_index_set.insert(io_index);
      }
    }
  }

  for (auto item : allocation_ids_to_model_args_refresh_infos_addr_low_32bit) {
    for (const auto &info : item) {
     if (info.device_args_addr % sizeof(uint64_t) != 0) {
       GELOGW("info.device_args_addr %llu is not uint64 offset.", info.device_args_addr);
       return FAILED;
     }
      uint64_t io_index =
         (info.device_args_addr - model_args_[0].model_args_device_addr) / sizeof(uint64_t);
      if (io_index <= offset_num) {
        // index为uint32_t类型，且在算子侧，以uint8_t进行计算，也因此需要转换为uint8_t的index并压栈两次
        model_args_host_index[index_mutiples * io_index] = index_mutiples * info.id * sizeof(uint32_t);
        io_index_set.insert(io_index);
      }
    }
  }

  for (auto item : allocation_ids_to_model_args_refresh_infos_addr_high_32bit) {
    for (const auto &info : item) {
      uint64_t io_index =
          (info.device_args_addr - model_args_[0].model_args_device_addr) / sizeof(uint64_t);
      if (io_index <= offset_num) {
        model_args_host_offset[io_index] = info.offset;
        // index压栈两次
        model_args_host_index[index_mutiples * io_index + 1] = (index_mutiples * info.id + 1) * sizeof(uint32_t);
      }
    }
  }
  // index和offset的host表赋值，不可刷新部分赋值
  for (size_t i = 0; i < offset_num; i++) {
    if (io_index_set.count(i) == 0) {
      model_args_host_index[index_mutiples * i] = index_mutiples * (allocation_ids_to_model_args_refresh_infos_addr_all.size() - 1) * sizeof(uint32_t);
      model_args_host_index[index_mutiples * i + 1] = (index_mutiples * allocation_ids_to_model_args_refresh_infos_addr_all.size() - 1) * sizeof(uint32_t);
      model_args_host_offset[i] = *(reinterpret_cast<uint64_t *>(model_args_[0].model_args_host_addr.get()) + i);
    }
  }

  // host_input_copy_num部分赋值
  uint32_t active_mem_base_addr_len = davinci_model_->GetLogicalMemAllocation().size() * sizeof(uint64_t);
  uint32_t active_mem_base_addr_len_align32b = AlignUp(active_mem_base_addr_len, kAlign32B);
  uint32_t active_mem_base_offset = active_mem_base_addr_len_align32b/sizeof(uint64_t);
  for (size_t i = offset_num; i < args_offset_num; i++) {
    if (io_index_set.count(i) == 0) {
      model_args_host_index[index_mutiples * i] = index_mutiples * active_mem_base_offset * sizeof(uint32_t);
      model_args_host_index[index_mutiples * i + 1] = (index_mutiples * active_mem_base_offset + 1) * sizeof(uint32_t);
      model_args_host_offset[i] = 0;
      active_mem_base_offset++;
    }
  }

  // 拷贝index和offset的device表
  GE_ASSERT_RT_OK(rtMemcpy(model_args_device_offset_, args_offset_num * sizeof(uint64_t),
      model_args_host_offset.get(), args_offset_num * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));
  GE_ASSERT_RT_OK(rtMemcpy(model_args_device_index_, args_offset_num * sizeof(uint64_t),
      model_args_host_index.get(), args_offset_num * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));

  // 补充args构成
  kernel_launch_args_ptr_->model_offset_args_device_addr = PtrToValue(model_args_device_offset_);
  kernel_launch_args_ptr_->model_index_args_device_addr = PtrToValue(model_args_device_index_);
  kernel_launch_args_ptr_->model_args_table_addr = model_args_[0].model_args_device_addr;
  kernel_launch_args_ptr_->workspace_addr = PtrToValue(workspace_addr_device_);
  if (logLevel_ <= DLOG_INFO) {
    for (size_t i = 0; i < args_offset_num; i++) {
      GELOGI("Print GenAddrRefreshIndexAndOffset result, model args table Index is:%d, "
             "active mem base index is:%d, addr offset is:0x%" PRIx64, i,
             model_args_host_index[index_mutiples * i] / sizeof(uint64_t), model_args_host_offset[i]);
    }
  }

  return SUCCESS;
}

Status ModelArgsManager::GenKernelLaunchArgs(uint64_t &offset_num) {
  // gen tiling data
  uint64_t active_mem_base_addr_len = davinci_model_->GetLogicalMemAllocation().size() * sizeof(uint64_t);
  uint32_t active_mem_base_addr_len_align32b = AlignUp(active_mem_base_addr_len, kAlign32B);
  // host_input_size_已做过32字节对齐
  uint64_t args_active_mem_base_and_host_input_size = active_mem_base_addr_len_align32b + host_input_size_;
  uint64_t args_size = sizeof(KernelLaunchOpArgs) + args_active_mem_base_and_host_input_size;
  uint64_t args_offset_num = offset_num;

  addr_update_op_args_.tilingDataOffset = kAddrRefreshOpParamOffset;
  args_input_info_ptr_ = ge::MakeUnique<rtHostInputInfo_t>();
  if (args_size <= kRtsLitePcieBarCopySize) {
    // offset device addr | index device addr | active mem base device addr | model args table addr
    // | workspace addr | tiling addr | tiling data host addr | kernel launch args | active mem base host addr |
    // active mem base append
    args_input_info_ptr_->addrOffset = kKernelLaunchArgOffset2;
    args_input_info_ptr_->dataOffset = sizeof(KernelLaunchOpArgs);
    addr_update_op_args_.hostInputInfoPtr = args_input_info_ptr_.get();
    addr_update_op_args_.hostInputInfoNum = 1;
    addr_update_op_args_.argsSize = args_size;
    is_pcie_bar_copy_ = true;
    if (host_input_size_ > 0U) {
      // pcie bar场景通过算子刷新来更新host内存，offset需要做扩展,partition分配内存时已做64字节对齐，此处不会越界
      args_offset_num += MathCeil(host_input_size_, sizeof(uint64_t));
    }
    GE_ASSERT_SUCCESS(
      CalculateUpdateModelParamTiling(args_active_mem_base_and_host_input_size, args_offset_num * sizeof(uint64_t),
                                      block_dim_, kernel_launch_args_ptr_->tiling_data));
    GELOGI("kernel launch is pcie bar copy, args size:%" PRIu64 ", tiling active mem base and host input size:%" PRIu64
           ", active mem base align size:%u, host input size:%" PRIu64 ", tiling args offset num:%" PRIu64
           ", model args offset num:%" PRIu64,
           args_size, args_active_mem_base_and_host_input_size,
           active_mem_base_addr_len_align32b, host_input_size_, args_offset_num, offset_num);
  } else {
    activate_mem_base_device_addrs_dev_ = davinci_model_->MallocDynamicMemory(args_active_mem_base_and_host_input_size);
    // offset device addr | index device addr | active mem base device addr | model args table addr
    // | workspace addr | tiling addr | tiling data host addr | kernel launch args
    GE_ASSERT_NOTNULL(activate_mem_base_device_addrs_dev_);
    kernel_launch_args_ptr_->active_mem_base_device_addr = PtrToValue(activate_mem_base_device_addrs_dev_);
    host_input_device_ptr_ = kernel_launch_args_ptr_->active_mem_base_device_addr + active_mem_base_addr_len_align32b;
    addr_update_op_args_.hostInputInfoNum = 0;
    addr_update_op_args_.argsSize = sizeof(KernelLaunchOpArgs);
    GE_ASSERT_SUCCESS(CalculateUpdateModelParamTiling(
      active_mem_base_addr_len_align32b, args_offset_num * sizeof(uint64_t),
      block_dim_, kernel_launch_args_ptr_->tiling_data));

    GELOGI("kernel launch is not pcie bar copy, tiling active mem base size:%" PRIu64 ", tiling args offset num:%" PRIu64
           ", args size:%" PRIu64 ", host input size:%" PRIu64 ", malloc active mem and host input size:%" PRIu64,
           active_mem_base_addr_len_align32b, args_offset_num, addr_update_op_args_.argsSize,
           host_input_size_, args_active_mem_base_and_host_input_size);
  }

  GE_ASSERT_NOTNULL(launched_args_unique_ptr_);
  addr_update_op_args_.args = static_cast<void*>(launched_args_unique_ptr_.get());
  // tiling addr在的偏移地址
  addr_update_op_args_.tilingAddrOffset = kAddrRefreshOpParamOffset - sizeof(uint64_t);
  addr_update_op_args_.hasTiling = 1;
  addr_update_op_args_.isNoNeedH2DCopy = 0;

  return SUCCESS;
}

Status ModelArgsManager::InitTaskInfoV2(domi::ModelTaskDef &model_task_def) {
  if (model_task_def.task_size() == 0) {
    GELOGW("No task defs in model task def");
    return SUCCESS;
  }
  GELOGI("Begin to init all task info, task count %zu", model_task_def.task_size());
  allocation_ids_to_model_args_refresh_infos_addr_all.resize(davinci_model_->GetLogicalMemAllocation().size());
  allocation_ids_to_model_args_refresh_infos_addr_low_32bit.resize(davinci_model_->GetLogicalMemAllocation().size());
  allocation_ids_to_model_args_refresh_infos_addr_high_32bit.resize(davinci_model_->GetLogicalMemAllocation().size());
  allocation_ids_to_iow_pa_remap_infos_.resize(davinci_model_->GetLogicalMemAllocation().size());
  const size_t task_size = static_cast<size_t>(model_task_def.task_size());
  std::vector<TaskRunParam> task_indexes_to_run_param(task_size);
  TaskNodeMap task_node_map;
  GE_ASSERT_SUCCESS(task_node_map.Init(davinci_model_->GetCompiledComputeGraph(), task_size));
  GE_ASSERT_SUCCESS(ParseModelTaskDef(model_task_def, task_indexes_to_run_param, task_node_map));

  // todo 逻辑地址与memory type的对应关系，看起来通过task_info返回有些重复了，因为不同的task
  //      info可能返回同一个逻辑地址，而一个逻辑地址是什么memory type是确定的，没必要在每个task info中都返回一
  const auto logical_addrs_to_memory_type = MemoryAppTypeClassifier(davinci_model_->GetLogicalMemAllocation(),
                                                                    davinci_model_->GetFmMemAllocationsStartId())
                                          .ClassifyByTaskRunParams(task_indexes_to_run_param);

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> task_indexes_to_refresh_type;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  GE_ASSERT_SUCCESS(TaskArgsRefreshTypeClassifier(task_node_map, logical_addrs_to_memory_type,
                                                  davinci_model_->IsFeatureBaseRefreshable())
                        .ClassifyMultiTasks(task_indexes_to_run_param, task_indexes_to_refresh_type, fixed_addrs,
                                            davinci_model_->GetPhysicalMemoryRefreshable()));

  ModelArgsLayoutPlannedResult planned_model_args_layout_result;
  GE_ASSERT_SUCCESS(ModelArgsLayoutPlanner(task_indexes_to_refresh_type, task_indexes_to_run_param, host_input_size_)
                        .Plan(planned_model_args_layout_result, AddrUseFor::kAddrUseForArgs));
  GE_ASSERT_SUCCESS(AllocModelArgs(planned_model_args_layout_result, model_args_, model_args_len_, op_refresh_placement_));
  std::vector<PisToArgs> task_indexes_to_args;
  GE_ASSERT_SUCCESS(ConstructUpdateData(task_node_map, planned_model_args_layout_result, task_indexes_to_run_param,
                                        task_indexes_to_args));

  GE_ASSERT_SUCCESS(AllocFixedAddrs(task_node_map, fixed_addrs));

  std::vector<IowAddrs> task_indexes_to_init_param;
  GE_ASSERT_SUCCESS(ConstructTaskInitParams(task_indexes_to_refresh_type, logical_addrs_to_memory_type,
                                            std::move(task_indexes_to_run_param), task_indexes_to_init_param));

  for (size_t i = 0UL; i < task_list_ptr_->size(); ++i) {
    const auto task_info = task_list_ptr_->at(i);
    // todo persistent workspaces not set yet
    GE_ASSERT_SUCCESS(task_info->Init(model_task_def.task(static_cast<int32_t>(i)), davinci_model_,
                                      task_indexes_to_args.at(i), {}, task_indexes_to_init_param.at(i)),
                      "Failed to init task index %zu, related node %s", i,
                      task_node_map.FindNodeByTaskIndex(i).node->GetName().c_str());

    std::vector<TaskArgsRefreshInfo> infos;
    GE_ASSERT_SUCCESS(task_info->GetTaskArgsRefreshInfos(infos),
                    "Failed to get task args refresh infos, task index %zu, related node %s", i,
                    task_node_map.FindNodeByTaskIndex(i).node->GetName().c_str());

    GE_ASSERT_SUCCESS(GenModelArgsRefreshInfosForTask(infos, task_indexes_to_args[i],
                      task_node_map.FindNodeByTaskIndex(i).node));

    std::vector<IowPaRemapInfo> pa_remap_infos;
    GE_ASSERT_SUCCESS(task_info->GetTaskIowPaRemapInfos(pa_remap_infos),
                     "Failed to get task iow pa remap infos, task index %zu, related node %s", i,
                     task_node_map.FindNodeByTaskIndex(i).node->GetName().c_str());

    GE_ASSERT_SUCCESS(GenAllocationToIowPaRemapInfos(task_info,
                      task_node_map.FindNodeByTaskIndex(i).node, pa_remap_infos));
  }

  /*
   * todo: davinci model中存在编译时即返回的不支持零拷贝的输入输出，这部分信息需要被利用
   */

  /*
   * todo: 如何识别不能零拷贝的输入输出？
   * 如果一个模型的输入输出内存的逻辑地址出现在了fm内，说明该内存参与完整版本的fm的内存复用，
   * 这意味着，这块内存可能作为一个子block参与复用。这意味着这块输入/输出内存，无法进行零拷贝，即不支持被ModelIo刷新
   * 例如：一块model input内存被作为子block复用在了PhonyConcat的输入上，
   * 那么当输入地址变化时，PhonyConcat的输出地址没法变化。导致PhonyConcat的输出内存错误（少了模型输入的部分）
   *
   * 判断一个输入/输出是否可以零拷贝：当模型的输入/输出地址不是modelio段时，本输入/输出不可以零拷贝。
   * todo: 不可以零拷贝的段被识别后，需要返回给davinci model，在模型执行前/后，做显式的拷贝动作
   */
  GE_CHK_RT_RET(rtNeedDevVA2PA(&need_dev_va_2_pa_));
  if (update_version_ != 1) {
    InitForUpdate();
  }
  return SUCCESS;
}

void ModelArgsManager::InitForUpdate() {
  const size_t size = davinci_model_->GetLogicalMemAllocation().size();
  last_bases_.resize(size, UINT64_MAX);
  id_to_plicy_.resize(size);

  id_to_len_.resize(size);
  const auto logical_mem_allocations = davinci_model_->GetLogicalMemAllocation();
  for (size_t id = 0U; id < size; id++) {
    id_to_len_[id] = logical_mem_allocations[id].data_size;
  }

  const uint32_t absolute_mem_id = static_cast<uint32_t>(size - 1U);
  id_to_plicy_[absolute_mem_id] = static_cast<uint32_t>(kInitOneTime);

  const size_t fm_start_id = davinci_model_->GetFmMemAllocationsStartId();
  const size_t fm_size = davinci_model_->GetFmMemAllocationsSize();
  for (size_t id = 0U; id < absolute_mem_id; id++) {
    if ((id >= fm_start_id) && (id < (fm_start_id + fm_size))) {
      id_to_plicy_[id] = static_cast<uint32_t>(kUpdateFmAndModelIo);
    } else {
      id_to_plicy_[id] = static_cast<uint32_t>(kUpdateModelIo);
    }
  }
}

Status ModelArgsManager::TaskArgsVa2PaAssociatedWithModelIO(rtStream_t const stm) const {
  auto &model_update_data = update_policies_to_model_data_[kUpdateModelIo];
  GE_ASSERT_NOTNULL(model_update_data, "Failed to exe model args va 2 pa, policy %s does not exist",
                    GetUpdatePolicyStr(kUpdateModelIo));

  for (const auto &cp_data : model_update_data->h2d_copy_datas) {
    GE_ASSERT_RT_OK(rtDevVA2PA(cp_data.device_addr, cp_data.len, stm, davinci_model_->GetAsyncMode()));
  }

  return SUCCESS;
}

void ModelArgsManager::UpdateHostArgs(uint64_t* active_mem_base_addr) {
  dfx_info_.update_addr_num = 0UL;
  const size_t size = davinci_model_->GetLogicalMemAllocation().size();
  for (size_t id = 0UL; id < size; id++) {
    if (active_mem_base_addr[id] == last_bases_[id]) {
      continue;
    }

    for (const auto &info : allocation_ids_to_model_args_refresh_infos_addr_all[id]) {
      *(PtrToPtr<void, uint64_t>(info.host_args_addr)) = active_mem_base_addr[id] + info.offset;
      GELOGI("[Args][Updater] update model args refresh info:[%s], active addr:0x%llx.",
        info.ToString().c_str(), *(PtrToPtr<void, uint64_t>(info.host_args_addr)));
    }
    dfx_info_.update_addr_num += allocation_ids_to_model_args_refresh_infos_addr_all[id].size();

    for (const auto &info : allocation_ids_to_model_args_refresh_infos_addr_low_32bit[id]) {
      *(PtrToPtr<void, uint32_t>(info.host_args_addr)) =
        static_cast<uint32_t>((active_mem_base_addr[id] + info.offset) & k32BitsMask);
      GELOGI("[Args][Updater] update model args refresh info:[%s], active addr:0x%x",
        info.ToString().c_str(), *(PtrToPtr<void, uint32_t>(info.host_args_addr)));
    }
    dfx_info_.update_addr_num += allocation_ids_to_model_args_refresh_infos_addr_low_32bit[id].size();

    for (const auto &info : allocation_ids_to_model_args_refresh_infos_addr_high_32bit[id]) {
      *(PtrToPtr<void, uint32_t>(info.host_args_addr)) =
        static_cast<uint32_t>((active_mem_base_addr[id] + info.offset) >> k32Bits);
      GELOGI("[Args][Updater] update model args refresh info:%s, active addr:0x%x",
        info.ToString().c_str(), *(PtrToPtr<void, uint32_t>(info.host_args_addr)));
    }

    last_bases_[id] = active_mem_base_addr[id];
  }
}

void ModelArgsManager::GenModelArgsAaddrAfterDistributed() {
  // 满足以下条件才用算子刷新
  // 1、地址刷新算子已加载
  // 2、只有一个placememt需要刷新且placememt有效(即只kernel launch一次算子)
  if (func_handle_ != nullptr && model_args_.size() == 1 &&
      op_refresh_placement_ == ArgsPlacement::kArgsPlacementHbm) {
    uint64_t offset_num = (model_args_len_[0] - host_input_partition_len_) / sizeof(uint64_t) ;
    // args table表的长度在这边扩展
    if (offset_num > 0 && GenKernelLaunchArgs(offset_num) == SUCCESS &&
        GenAddrRefreshIndexAndOffset(offset_num) == SUCCESS) {
        update_version_ = kUpdateVersionKernelLaunch;
    }
    GELOGI("update_version:%d, model args offset num:%llu", update_version_, offset_num);
  } else {
    GELOGI("update_version:%d, func_handle_:%p, model args size:%zu, op_refresh_placement:%d",
      update_version_, func_handle_, model_args_.size(), static_cast<int32_t>(op_refresh_placement_));
  }
  GELOGI("model args manager update version %d", update_version_);
  return;
}

Status ModelArgsManager::PrintKernelLaunchArgsDfxInfo(rtStream_t const stm) {
  uint32_t active_mem_base_addr_size = davinci_model_->GetLogicalMemAllocation().size();
  uint32_t active_mem_base_addr_len_align32b = AlignUp(active_mem_base_addr_size * sizeof(uint64_t), kAlign32B);
  active_mem_base_addr_len_align32b = active_mem_base_addr_len_align32b + host_input_size_;
  active_mem_base_addr_size = active_mem_base_addr_len_align32b/sizeof(uint64_t);
  // 此处添加校验
  uint64_t *active_mem_base_addr = GetActivateMemBaseAddrs();
  for (size_t i = 0; i < active_mem_base_addr_size; i++) {
    GELOGI("Print Kernel Launch Args, host active mem base Index is:%d, active mem base addr is:0x%" PRIx64,
            i, active_mem_base_addr[i]);
  }

  GE_CHK_RT_RET(rtStreamSynchronize(stm));
  std::vector<uint64_t> model_args_device_addrs(model_args_len_[0] / sizeof(uint64_t), 0);
  (void)rtMemcpy(model_args_device_addrs.data(), model_args_len_[0], ValueToPtr(model_args_[0].model_args_device_addr),
                  model_args_len_[0], RT_MEMCPY_DEVICE_TO_HOST);
  UpdateModelParamTilingData update_model_param_tiling_data_temp = {};
  (void)rtMemcpy(static_cast<void*>(&update_model_param_tiling_data_temp), sizeof(UpdateModelParamTilingData),
                 ValueToPtr(kernel_launch_args_ptr_->tiling_addr), sizeof(UpdateModelParamTilingData), RT_MEMCPY_DEVICE_TO_HOST);
  GELOGI("Print device Tiling Data. tiling.totalActivateBaseTblCnt: %u, tiling.blockCnt:%u, tiling.tileCnt: %u , tiling.tileNum: %u, "
    "tiling.tailCnt: %u, tiling.lastTileNum: %u, tiling.lastTailCnt: %u, block dim is %u.",
    update_model_param_tiling_data_temp.totalActiveBaseTblCnt,
    update_model_param_tiling_data_temp.blockCnt, update_model_param_tiling_data_temp.tileCnt,
    update_model_param_tiling_data_temp.tileNum, update_model_param_tiling_data_temp.tailCnt,
    update_model_param_tiling_data_temp.lastTileNum, update_model_param_tiling_data_temp.lastTailCnt, block_dim_);

  std::vector<uint32_t> device_index_table;
  std::vector<uint64_t> device_offset_table;

  uint64_t model_args_refresh_len_ = model_args_len_[0] - host_input_partition_len_;
  if (host_input_size_ > 0U && is_pcie_bar_copy_) {
    model_args_refresh_len_ += host_input_size_;
  }

  device_index_table.resize(model_args_refresh_len_ / sizeof(uint32_t));
  device_offset_table.resize(model_args_refresh_len_ / sizeof(uint64_t));
  (void)rtMemcpy(device_offset_table.data(), model_args_refresh_len_,
                 ValueToPtr(kernel_launch_args_ptr_->model_offset_args_device_addr), model_args_refresh_len_, RT_MEMCPY_DEVICE_TO_HOST);
  (void)rtMemcpy(device_index_table.data(), model_args_refresh_len_,
                 ValueToPtr(kernel_launch_args_ptr_->model_index_args_device_addr), model_args_refresh_len_, RT_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < model_args_refresh_len_ / sizeof(uint64_t); i++) {
    GELOGI("Print device offset table. index:%" PRId64 ", offset is:%" PRId64 ", index is %d, %d.",
          i, device_offset_table[i], device_index_table[2 * i], device_index_table[2 * i + 1]);
  }

  std::vector<uint64_t> device_active_mem_table;
  device_active_mem_table.resize(active_mem_base_addr_size);
  (void)rtMemcpy(device_active_mem_table.data(), active_mem_base_addr_size * sizeof(uint64_t),
                 ValueToPtr(kernel_launch_args_ptr_->active_mem_base_device_addr), active_mem_base_addr_size * sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < active_mem_base_addr_size; i++) {
    GELOGI("Print active mem base. index:%" PRId64 ", value is:%" PRId64 ".", i, device_active_mem_table[i]);
  }

  GELOGI("Print kernelLaunch Op args is: model_offset_args_device_addr is:0x%" PRIx64 ", "
         "model_index_args_device_addr is:0x%" PRIx64 ", active_mem_base_device_addr: 0x%" PRIx64 ", "
         "model_args_table_addr:0x%" PRIx64 ", workspace_addr:0x%" PRIx64 ", tiling_addr:0x%" PRIx64,
         kernel_launch_args_ptr_->model_offset_args_device_addr,
         kernel_launch_args_ptr_->model_index_args_device_addr, kernel_launch_args_ptr_->active_mem_base_device_addr,
         kernel_launch_args_ptr_->model_args_table_addr, kernel_launch_args_ptr_->workspace_addr,
         kernel_launch_args_ptr_->tiling_addr);

  for (size_t j = 0; j < model_args_len_[0] / sizeof(uint64_t); j++) {
    GELOGI("Print model args host table, model args index is:%d, model args host tensor data addr is:0x%" PRIx64 ","
            "model device_args_addr is 0x%" PRIx64 ".",
          j, *(reinterpret_cast<uint64_t*>(static_cast<void*>(model_args_[0].model_args_host_addr.get())) + j),
          model_args_[0].model_args_host_addr.get() + j * sizeof(uint64_t));
    if (model_args_device_addrs[j] !=
        *(reinterpret_cast<uint64_t*>(static_cast<void*>(model_args_[0].model_args_host_addr.get())) + j)) {
      GELOGI("Print different args. Index: %" PRId64 ", device addr is is :%" PRId64 ", host addr is: %" PRId64,
        j, model_args_device_addrs[j],
        *(reinterpret_cast<uint64_t*>(static_cast<void*>(model_args_[0].model_args_host_addr.get())) + j));
    }
  }
  return SUCCESS;
}

Status ModelArgsManager::ReportKernelLaunchOpProfilingData(const uint64_t begin_time) const {
  thread_local const int32_t tid = mmGetTid();
  const uint64_t end_time = MsprofSysCycleTime();
  const uint64_t op_name_hash = MsprofGetHashId(kAddrRefreshOpName.c_str(), kAddrRefreshOpName.length());
  (void)gert::GlobalProfilingWrapper::ReportApiInfo(begin_time, end_time, op_name_hash,
                                                    MSPROF_REPORT_NODE_LAUNCH_TYPE);

  if (!gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kDevice)) {
    return ge::SUCCESS;
  }

  MsprofCompactInfo node_basic_info{};
  const uint64_t op_type_hash = MsprofGetHashId(kAddrRefreshOpType.c_str(), kAddrRefreshOpType.length());
  uint32_t task_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV);

  auto &prof_node_basic_info = node_basic_info.data.nodeBasicInfo;
  prof_node_basic_info.opName = op_name_hash;
  prof_node_basic_info.opType = op_type_hash;
  prof_node_basic_info.taskType = task_type;
  prof_node_basic_info.blockDim = block_dim_;
  prof_node_basic_info.opFlag = 0;

  node_basic_info.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
  node_basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  node_basic_info.timeStamp = end_time;
  node_basic_info.threadId = static_cast<uint32_t>(tid);

  GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(static_cast<uint32_t>(true), &node_basic_info,
                                              static_cast<uint32_t>(sizeof(MsprofCompactInfo))));

  return ge::SUCCESS;
}

 Status ModelArgsManager::UpdateForExecute(uint32_t &up, rtStream_t const stm, const uint32_t model_execute_stage) {
  GetStageTimeInfo(kStageCalcUpdatePolicyBegin);
  uint64_t active_mem_base_addr_len = davinci_model_->GetLogicalMemAllocation().size() * sizeof(uint64_t);
  uint64_t *active_mem_base_addr = GetActivateMemBaseAddrs();
  GE_ASSERT_NOTNULL(active_mem_base_addr);
   if (update_version_ == 1) {
    std::vector<uint64_t> active_mem_base_addr_vec;
    for (size_t i = 0; i < davinci_model_->GetLogicalMemAllocation().size(); i++) {
      active_mem_base_addr_vec.emplace_back(active_mem_base_addr[i]);
    }
    up_ = CalcUpdatePolicy(active_mem_base_addr_vec);
    GELOGI("Begin to update model args, policy %s, fm_hit_count 0x%" PRIx64 ", "
          "zero_copy_model_io_hit_count:0x%" PRIx64 ", va_2_pa:%d.",
          GetUpdatePolicyStr(up_), fm_hit_count_, model_io_hit_count_, static_cast<int32_t>(need_dev_va_2_pa_));
    GE_ASSERT_TRUE(up_ < kUpdatePolicyEnd);
    GetStageTimeInfo(kStageUpdateHostArgsBegin);
    if (up_ == kNoNeedUpdate) {
      if (need_dev_va_2_pa_ && (model_io_hit_count_ != 0UL)) {
        GE_ASSERT_SUCCESS(TaskArgsVa2PaAssociatedWithModelIO(stm));
      }
      return SUCCESS;
    }
    auto &model_update_data = update_policies_to_model_data_[up_];
    GE_ASSERT_NOTNULL(model_update_data, "Failed to update model args, policy %s does not exist",
                      GetUpdatePolicyStr(up_));

    for (const auto &update_data : model_update_data->update_datas) {
      GE_ASSERT_SUCCESS(update_data.task_info->UpdateHostArgs(active_mem_base_addr_vec, update_data.host_args));
    }
  } else if (update_version_ == kUpdateVersionH2d || model_execute_stage == kModelLoadStage
    || davinci_model_->GetForbiddenStreamFlag()) {
    GetStageTimeInfo(kStageUpdateHostArgsBegin);
    up_ = static_cast<ModelArgsManager::UpdatePolicy>(up);
    if (SECUREC_UNLIKELY(!has_args_)) {
      up_ = kNoNeedUpdate;
    }
    if (up_ != kNoNeedUpdate) {
      UpdateHostArgs(active_mem_base_addr);
    }
    up_ = ((model_io_hit_count_ == 0U) && (up_ == kUpdateModelIo)) ? kNoNeedUpdate : up_;
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("Begin to update model args, policy %s, fm_hit_count 0x%" PRIx64 ", model_io_hit_count:0x%" PRIx64
        ", update_addr_num:%" PRIu64 ", va_2_pa:%d.", GetUpdatePolicyStr(up_), fm_hit_count_, model_io_hit_count_,
        dfx_info_.update_addr_num, static_cast<int32_t>(need_dev_va_2_pa_));
    }

    GE_ASSERT_TRUE(up_ < kUpdatePolicyEnd);
    if (up_ == kNoNeedUpdate) {
      if (need_dev_va_2_pa_ && (model_io_hit_count_ != 0UL)) {
        GE_ASSERT_SUCCESS(TaskArgsVa2PaAssociatedWithModelIO(stm));
      }
      return SUCCESS;
    }

    // 更新dump info
    if (davinci_model_->ModelNeedDump() ||            // data dump include L0
        davinci_model_->IsDumpLayerOpModelEnable() || // data dump include L0
        davinci_model_->GetOpDugReg() ||              // overflow dump include L0
        gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) { // exception dump
      auto &model_update_data = update_policies_to_model_data_[up_];
      GE_ASSERT_NOTNULL(model_update_data, "Failed to update model args, policy %s does not exist",
                        GetUpdatePolicyStr(up_));
      for (const auto &update_data : model_update_data->update_datas) {
        GE_ASSERT_SUCCESS(update_data.task_info->UpdateDumpInfos(update_data.host_args));
      }
    }
  } else {
    GetStageTimeInfo(kStageUpdateHostArgsBegin);
    up_ = static_cast<ModelArgsManager::UpdatePolicy>(up);
    up_ = ((model_io_hit_count_ == 0U) && (up_ == kUpdateModelIo)) ? kNoNeedUpdate : up_;
    if (SECUREC_UNLIKELY(!has_args_)) {
      up_ = kNoNeedUpdate;
    }
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("Begin to update model args, policy %s, fm_hit_count 0x%" PRIx64 ", model_io_hit_count:0x%" PRIx64
        ", update_addr_num:%" PRIu64 ", va_2_pa:%d.", GetUpdatePolicyStr(up_), fm_hit_count_, model_io_hit_count_,
        dfx_info_.update_addr_num, static_cast<int32_t>(need_dev_va_2_pa_));
    }
    GE_ASSERT_TRUE(up_ < kUpdatePolicyEnd);
    if (up_ == kNoNeedUpdate) {
      if (need_dev_va_2_pa_ && (model_io_hit_count_ != 0UL)) {
        GE_ASSERT_SUCCESS(TaskArgsVa2PaAssociatedWithModelIO(stm));
      }
      return SUCCESS;
    }

    // 更新dump info
    if (davinci_model_->ModelNeedDump() ||            // data dump include L0
        davinci_model_->IsDumpLayerOpModelEnable() || // data dump include L0
        davinci_model_->GetOpDugReg() ||              // overflow dump include L0
        gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) { // exception dump
      UpdateHostArgs(active_mem_base_addr);
      auto &model_update_data = update_policies_to_model_data_[up_];
      GE_ASSERT_NOTNULL(model_update_data, "Failed to update model args, policy %s does not exist",
                        GetUpdatePolicyStr(up_));
      for (const auto &update_data : model_update_data->update_datas) {
        GE_ASSERT_SUCCESS(update_data.task_info->UpdateDumpInfos(update_data.host_args));
      }
    }

    GetStageTimeInfo(kStageActiveMembaseMemcpyBegin);
    uint32_t active_mem_base_addr_len_align32b = AlignUp(active_mem_base_addr_len, kAlign32B);
    uint64_t args_active_mem_base_size = active_mem_base_addr_len_align32b + host_input_size_;
    if (args_active_mem_base_size >
      kRtsLitePcieBarCopySize - kAddrRefreshOpParamOffset - sizeof(UpdateModelParamTilingData)) {
      if (up_ == KUpdateHostInput) {
        uint64_t host_input_device_addr =
            PtrToValue(activate_mem_base_device_addrs_dev_) + active_mem_base_addr_len_align32b;
        uint64_t host_input_host_addr = PtrToValue(active_mem_base_addr) + active_mem_base_addr_len_align32b;
        GE_ASSERT_RT_OK(rtMemcpyAsync(ValueToPtr(host_input_device_addr), host_input_size_,
            ValueToPtr(host_input_host_addr), host_input_size_, RT_MEMCPY_HOST_TO_DEVICE_EX, stm));
        return SUCCESS;
      }
      // 加载阶段会做一次model args table表的全量刷新，本次刷新未使用地址刷新算子，所以未刷新device侧active mem base表
      // 需要在首次使用地址刷新算子时，完成对device侧active mem base表全量刷新，新增active_mem_base_table_h2d_copy_flag_区分是否部分拷贝active mem base表
      if (up > static_cast<uint32_t>(kUpdateModelIo) || !active_mem_base_table_h2d_copy_flag_) {
        GE_ASSERT_RT_OK(rtMemcpyAsync(activate_mem_base_device_addrs_dev_, args_active_mem_base_size,
          static_cast<void*>(active_mem_base_addr), args_active_mem_base_size, RT_MEMCPY_HOST_TO_DEVICE_EX, stm));
        active_mem_base_table_h2d_copy_flag_ = true;
      } else {
        args_active_mem_base_size = args_active_mem_base_size - davinci_model_->GetNoFrozenInputAllocationBaseId() * sizeof(uint64_t);
        GE_ASSERT_RT_OK(rtMemcpyAsync(ValueToPtr(PtrToValue(activate_mem_base_device_addrs_dev_) +
	    davinci_model_->GetNoFrozenInputAllocationBaseId() * sizeof(uint64_t)), args_active_mem_base_size,
            static_cast<void*>(active_mem_base_addr + davinci_model_->GetNoFrozenInputAllocationBaseId()),
	    args_active_mem_base_size, RT_MEMCPY_HOST_TO_DEVICE_EX, stm));
      }
    }

    GetStageTimeInfo(kStageKernelLaunchBegin);
    bool l0_prof_enable = gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime);
    uint64_t kernel_launch_prof_begin_time = 0;
    GE_IF_BOOL_EXEC(l0_prof_enable, kernel_launch_prof_begin_time = MsprofSysCycleTime());
    GE_IF_BOOL_EXEC(dfx_info_.get_model_args_device_table_flag, GE_CHK_RT_RET(rtStreamSynchronize(stm)));

    LaunchKernelParam launch_kernel_param;
    launch_kernel_param.stream = stm;
    launch_kernel_param.block_dim = block_dim_;
    launch_kernel_param.args = addr_update_op_args_.args;
    launch_kernel_param.args_size = addr_update_op_args_.argsSize;
    if (addr_update_op_args_.hostInputInfoPtr != nullptr) {
      RefreshAddrInfo input_output_addr_info;
      input_output_addr_info.addrOffset = addr_update_op_args_.hostInputInfoPtr->addrOffset;
      input_output_addr_info.dataOffset = addr_update_op_args_.hostInputInfoPtr->dataOffset;
      launch_kernel_param.refresh_add_infos.emplace_back(input_output_addr_info);
    }
    RefreshAddrInfo tiling_addr_info;
    tiling_addr_info.addrOffset = addr_update_op_args_.tilingAddrOffset;
    tiling_addr_info.dataOffset = addr_update_op_args_.tilingDataOffset;
    launch_kernel_param.refresh_add_infos.emplace_back(tiling_addr_info);

    launch_kernel_param.is_host_args = true;
    GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(func_handle_, launch_kernel_param));
    GE_IF_BOOL_EXEC(l0_prof_enable, ReportKernelLaunchOpProfilingData(kernel_launch_prof_begin_time));
    if (dfx_info_.get_model_args_device_table_flag && logLevel_ <= DLOG_INFO) {
      UpdateHostArgs(active_mem_base_addr);
      GE_ASSERT_SUCCESS(PrintKernelLaunchArgsDfxInfo(stm));
    }
  }

  GetStageTimeInfo(kStageHostArgsH2dBegin);
  auto &model_update_data = update_policies_to_model_data_[up_];
  GE_ASSERT_NOTNULL(model_update_data, "Failed to update model args, policy %s does not exist",
                    GetUpdatePolicyStr(up_));
  if (update_version_ != kUpdateVersionKernelLaunch  || model_execute_stage == kModelLoadStage
    || davinci_model_->GetForbiddenStreamFlag()) {
    for (const auto &cp_data : model_update_data->h2d_copy_datas) {
      if (davinci_model_->GetAsyncMode()) {
        GE_ASSERT_RT_OK(rtMemcpyAsync(ValueToPtr(cp_data.device_addr), cp_data.len, cp_data.host_addr, cp_data.len,
            RT_MEMCPY_HOST_TO_DEVICE_EX, stm));
      } else {
        GE_ASSERT_RT_OK(rtMemcpy(ValueToPtr(cp_data.device_addr), cp_data.len, cp_data.host_addr, cp_data.len,
            RT_MEMCPY_HOST_TO_DEVICE));
      }
      if (need_dev_va_2_pa_) {
        GE_ASSERT_RT_OK(rtDevVA2PA(cp_data.device_addr, cp_data.len, stm, davinci_model_->GetAsyncMode()));
      }
    }
  }

  GetStageTimeInfo(kStageUpdateDsaSqeBegin);
  for (const auto &sqe_ud : model_update_data->seq_update_datas) {
    GE_ASSERT_RT_OK(
        rtLaunchSqeUpdateTask(sqe_ud.stream_id, sqe_ud.task_id, ValueToPtr(sqe_ud.dev_addr), sqe_ud.len, stm));
  }
  return SUCCESS;
}

void ModelArgsManager::InitDfxStage1Begin() {
  if (!dfx_info_.enable_flag) {
    return;
  }
  dfx_info_.stage_time_info[kStagePrepareBegin] = ge::GetCurrentTimestamp();
}

void ModelArgsManager::InitDfxStatsticsEnd() {
  if (!dfx_info_.enable_flag) {
    return;
  }
  dfx_info_.stage_time_info[kStageStatisticsEnd] = ge::GetCurrentTimestamp();
}

void ModelArgsManager::GetStageTimeInfo(ModelArgsManagerStage stage) {
  if (!dfx_info_.enable_flag) {
    return;
  }
  dfx_info_.stage_time_info[stage] = ge::GetCurrentTimestamp();
}

void ModelArgsManager::CalculateDfxTime(std::stringstream &ss, const uint32_t model_execute_stage) {
  if (!dfx_info_.enable_flag) {
    return;
  }

  if (up_ == kNoNeedUpdate) {
    dfx_info_.stage_time_info[kStageHostArgsH2dBegin] = dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
    dfx_info_.stage_time_info[kStageUpdateDsaSqeBegin] = dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
    dfx_info_.stage_time_info[kStageActiveMembaseMemcpyBegin] = dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
    dfx_info_.stage_time_info[kStageKernelLaunchBegin] = dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
  }

  if (update_version_ == kUpdateVersionH2d || model_execute_stage == kModelLoadStage) {
    dfx_info_.stage_time_info[kStageActiveMembaseMemcpyBegin] = dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
    dfx_info_.stage_time_info[kStageKernelLaunchBegin] = dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
  }

  if ((update_version_ == kUpdateVersionKernelLaunch) && (up_ == KUpdateHostInput)) {
    dfx_info_.stage_time_info[kStageHostArgsH2dBegin] = dfx_info_.stage_time_info[kStageKernelLaunchBegin];
    dfx_info_.stage_time_info[kStageUpdateDsaSqeBegin] = dfx_info_.stage_time_info[kStageKernelLaunchBegin];
  }

  const uint64_t stage1_t =
    dfx_info_.stage_time_info[kStageCalcUpdatePolicyBegin] - dfx_info_.stage_time_info[kStagePrepareBegin];
  const uint64_t stage2_t =
    dfx_info_.stage_time_info[kStageUpdateHostArgsBegin] - dfx_info_.stage_time_info[kStageCalcUpdatePolicyBegin];

  const uint64_t stage3_t =
    dfx_info_.stage_time_info[kStageActiveMembaseMemcpyBegin] - dfx_info_.stage_time_info[kStageUpdateHostArgsBegin];
  const uint64_t stage4_t =
    dfx_info_.stage_time_info[kStageKernelLaunchBegin] - dfx_info_.stage_time_info[kStageActiveMembaseMemcpyBegin];

  const uint64_t stage5_t =
    dfx_info_.stage_time_info[kStageHostArgsH2dBegin] - dfx_info_.stage_time_info[kStageKernelLaunchBegin];
  const uint64_t stage6_t =
    dfx_info_.stage_time_info[kStageUpdateDsaSqeBegin] - dfx_info_.stage_time_info[kStageHostArgsH2dBegin];
  const uint64_t stage7_t =
    dfx_info_.stage_time_info[kStageStatisticsEnd] - dfx_info_.stage_time_info[kStageUpdateDsaSqeBegin];

  const uint64_t avg_a_addr_update_time =
    (dfx_info_.update_addr_num == 0UL) ? 0UL : (((stage5_t + stage4_t + stage3_t) * 1000UL) / dfx_info_.update_addr_num);

  ss << "update_version:" << update_version_ << ", updatepolicy:" << static_cast<int32_t>(up_)
    << ", active_mem_base_addr_len:" << dfx_info_.active_mem_base_addr_len
    << ", actual_update_addr_num:" << dfx_info_.update_addr_num
    << ", stage_2_1-7_time[" << stage1_t << "," << stage2_t << "," << stage3_t << ","
    << stage4_t << "," << stage5_t << "," << stage6_t << "," << stage7_t
    << "]us, avg_a_addr_update_time[" << avg_a_addr_update_time << "]ns";
}

void ModelArgsManager::PrintDfxStatistics(const uint32_t model_execute_stage) {
  if (!dfx_info_.enable_flag) {
    return;
  }

  std::stringstream ss;
  CalculateDfxTime(ss, model_execute_stage);

  GEEVENT(
    "[ArgsUpdate] graph_name:%s, graph_id:%u, model_id:%u, fm_refreshable:%d, known:%d, update_policy:%s, "
    "fm_hit_cnt:%" PRIu64 ", mdl_io_hit_cnt:%" PRIu64 ", %s",
    dfx_info_.graph_name.c_str(), dfx_info_.graph_id, dfx_info_.model_id,
    static_cast<int32_t>(dfx_info_.fm_refreshable), static_cast<int32_t>(dfx_info_.known), GetUpdatePolicyStr(up_),
    fm_hit_count_, model_io_hit_count_, ss.str().c_str());
}

Status ModelArgsManager::AllocModelArgs(const ModelArgsLayoutPlannedResult &layout,
                                        std::vector<ModelArgs> &model_args, std::vector<uint64_t> &model_args_len,
                                        ArgsPlacement &pls) {
  model_args.reserve(static_cast<size_t>(ArgsPlacement::kEnd));
  for (size_t pli = 0; pli < static_cast<size_t>(ArgsPlacement::kEnd); ++pli) {
    int64_t len = 0;
    ModelArgs placed_model_args;
    placed_model_args.placement = static_cast<ArgsPlacement>(pli);

    // model_args_partitions assignment
    for (size_t pai = 0; pai < static_cast<size_t>(UpdateTriggerType::kEnd); ++pai) {
      const auto partition_len = layout.placements_to_partitions_to_len[pli][pai];
      if (partition_len == 0) {
        continue;
      }

      if ((pli == static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)) &&
        (pai == static_cast<size_t>(UpdateTriggerType::KTriggerByHostInput))) {
        host_input_partition_len_ = partition_len;
      }

      placed_model_args.model_args_partitions.push_back({static_cast<UpdateTriggerType>(pai), len, partition_len});
      GE_ASSERT_TRUE(!AddOverflow(len, partition_len, len));
    }

    if (len == 0) {
      continue;
    }

    // host and device memory allocation and assignment
    placed_model_args.model_args_host_addr = ge::MakeUnique<uint8_t[]>(static_cast<size_t>(len));
    GE_ASSERT_NOTNULL(placed_model_args.model_args_host_addr, "Failed to alloc args %d at host, size %lld", pli, len);

    const auto memory_type = GetRtsMemoryType(placed_model_args.placement, len);
    const auto model_args_device_addr = davinci_model_->MallocDynamicMemory(static_cast<size_t>(len), memory_type);
    GE_ASSERT_NOTNULL(model_args_device_addr);
    placed_model_args.model_args_device_addr = PtrToValue(model_args_device_addr);

    GELOGI("Alloc model args len %lld, placement %s, addr 0x%llx for model %u(%s)", len,
           GetArgsPlacementStr(placed_model_args.placement), placed_model_args.model_args_device_addr,
           davinci_model_->GetModelId(), davinci_model_->GetOmName().c_str());

    model_args.emplace_back(std::move(placed_model_args));
    model_args_len.emplace_back(len);
    pls = placed_model_args.placement;
  }

  return SUCCESS;
}
Status ModelArgsManager::ConstructUpdateData(const TaskNodeMap &task_node_map,
                                             const ModelArgsLayoutPlannedResult &layout,
                                             const std::vector<TaskRunParam> &task_indexes_to_param,
                                             std::vector<PisToArgs> &task_indexes_to_args) {
  // step 1. prepare query data
  const bool need_debug_log = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  auto trigger_types_to_update_policies = GenerateTriggerTypesToCorrespondingUpdatePolicies();

  std::array<const ModelArgs *, static_cast<size_t>(ArgsPlacement::kEnd)> pis_to_model_args{nullptr};
  for (const auto &placed_model_arg : model_args_) {
    pis_to_model_args[static_cast<size_t>(placed_model_arg.placement)] = &placed_model_arg;
  }

  // step 2. construct task-level update data, e.g. update_datas, seq_update_datas
  const auto task_size = layout.task_indexes_to_arg_results.size();
  task_indexes_to_args.resize(task_size);
  for (size_t i = 0U; i < task_size; ++i) {
    const auto &task_arg_results = layout.task_indexes_to_arg_results[i];
    if (task_arg_results.empty()) {
      continue;
    }

    OneTaskUpdateData one_task_update_data{{i, task_list_ptr_->at(i).get(), {}},  // UpdateHostArgsArg
                                           false,                                 // has_sqe_placement
                                           {},                                    // SqeUpdateArg
                                           &task_indexes_to_args};
    GE_ASSERT_SUCCESS(ConstructOneTaskUpdateData(i, task_arg_results, task_indexes_to_param, pis_to_model_args,
                                                 one_task_update_data, AddrUseFor::kAddrUseForArgs));
    const auto &upis = trigger_types_to_update_policies.at(static_cast<size_t>(task_arg_results.at(0).trigger_type));
    if (need_debug_log) {
      DebugLogTaskUpdatePolicies(task_node_map, upis, i);
    }
    GE_ASSERT_SUCCESS(AddToTaskUpdateDataToPolicies(i, upis, one_task_update_data));
  }

  // 增加host_input的updatda
  if (host_input_size_ > 0U) {
      update_policies_to_model_data_[KUpdateHostInput] = MakeUnique<ArgsUpdateData>();
      GE_ASSERT_NOTNULL(update_policies_to_model_data_[KUpdateHostInput]);
  }

  // step 3. construct policy level update data
  for (int32_t i = 0; i < kUpdatePolicyEnd; ++i) {
    const auto model_update_data = update_policies_to_model_data_[static_cast<size_t>(i)].get();
    if (model_update_data == nullptr) {
      continue;
    }
    for (const auto &model_arg : model_args_) {
      H2DCopyArg cp_arg{};
      const auto ret = ConstructH2DCopyParams(model_arg, static_cast<UpdatePolicy>(i), cp_arg);
      if (ret == GE_GRAPH_GRAPH_NOT_EXIST) {
        continue;
      } else if (ret == SUCCESS) {
        model_update_data->h2d_copy_datas.emplace_back(cp_arg);
      } else {
        return ret;
      }
    }
  }

  return SUCCESS;
}

void ModelArgsManager::DebugLogTaskUpdatePolicies(const TaskNodeMap &task_node_map, const TriggerPolicies &upis,
                                                  size_t task_index) const {
  std::stringstream ss;
  for (const auto upi : upis) {
    ss << GetUpdatePolicyStr(upi) << ",";
  }
  std::string node_name = "unknown";
  auto node_info = task_node_map.FindNodeByTaskIndex(task_index);
  if (node_info.node != nullptr) {
    node_name = node_info.node->GetName();
  }
  GELOGD("The args of node %s task index %zu will be updated in policies %s", node_name.c_str(), task_index,
         ss.str().c_str());
}

Status ModelArgsManager::ConstructOneTaskUpdateData(
    const size_t task_index, const OneTaskArgsLayoutResult &task_arg_results,
    const std::vector<TaskRunParam> &task_indexes_to_param,
    const std::array<const ModelArgsManager::ModelArgs *, static_cast<size_t>(ArgsPlacement::kEnd)> &pis_to_model_args,
    OneTaskUpdateData &task_update_data,
    const AddrUseFor addr_use_for) const {
  for (size_t j = 0UL; j < task_arg_results.size(); ++j) {
    const auto &task_arg_ret = task_arg_results[j];
    auto &args_desc = (addr_use_for == AddrUseFor::kAddrUseForArgs) ?
                                task_indexes_to_param[task_index].args_descs[j] :
                                task_indexes_to_param[task_index].persistent_workspace_descs[j];
    // store placement与require placement的区别：
    // require placement是task返回的placement，代表args的真实位置
    // store placement是经过layout planner规划后的placement，代表args在device上的存储位置
    // 在大部分场景下，这两个placement应该是相同的，但是例如placement sqe，该placement在device上与hbm一起存储
    const auto store_placement = task_arg_ret.placement;
    const auto require_placement = args_desc.placement;
    const auto placed_model_args = pis_to_model_args[static_cast<size_t>(store_placement)];

    void *host_addr = nullptr;
    uint64_t device_addr = 0UL;
    // ModelArgsManager允许长度为0的args，当一个policy中所有的args长度都是0时，placed_model_args
    // 不会被创建，此处会拿到空指针
    if (placed_model_args != nullptr) {
      host_addr = placed_model_args->model_args_host_addr.get() + task_arg_ret.offset;
      device_addr = placed_model_args->model_args_device_addr + static_cast<uint64_t>(task_arg_ret.offset);
    }

    task_update_data.update_data.host_args.emplace_back(HostArg{host_addr, args_desc.args_len, require_placement});
    (*task_update_data.task_indexes_to_args)[task_index][static_cast<size_t>(require_placement)] = {
        device_addr, host_addr, args_desc.args_len};

    if (require_placement == ArgsPlacement::kArgsPlacementSqe) {
      GE_ASSERT_TRUE(!task_update_data.has_sqe_placement,
                     "More than one placement-sqe tasks found in task %zu, not support yet", task_index);
      task_update_data.has_sqe_placement = true;
      task_update_data.sqe_update_arg.stream_id = std::numeric_limits<uint32_t>::max();  // update in OnTaskDistributed
      task_update_data.sqe_update_arg.task_id = std::numeric_limits<uint32_t>::max();    // update in OnTaskDistributed
      task_update_data.sqe_update_arg.dev_addr = device_addr;
      task_update_data.sqe_update_arg.len = static_cast<uint64_t>(args_desc.args_len);
    }
  }
  return SUCCESS;
}
Status ModelArgsManager::AddToTaskUpdateDataToPolicies(
    const size_t task_index,
    const SmallVector<ModelArgsManager::UpdatePolicy, ModelArgsManager::kUpdatePolicyEnd> &upis,
    const OneTaskUpdateData &one_task_update_data) {
  for (const auto upi : upis) {
    GE_ASSERT_TRUE(
        upi < kUpdatePolicyEnd,
        "Failed to construct update data, found trigger by fm partition when fm refresh disabled, task index %zu",
        task_index);
    if (update_policies_to_model_data_[upi] == nullptr) {
      update_policies_to_model_data_[upi] = MakeUnique<ArgsUpdateData>();
      GE_ASSERT_NOTNULL(update_policies_to_model_data_[upi]);
    }
    auto model_update_data = update_policies_to_model_data_[upi].get();
    model_update_data->update_datas.emplace_back(one_task_update_data.update_data);

    if (one_task_update_data.has_sqe_placement) {
      auto &sqe_update_datas = model_update_data->seq_update_datas;
      const auto sqe_index = model_update_data->seq_update_datas.size();
      task_indexes_to_update_data_appenders_on_distributed_[task_index].emplace_back(
          [&sqe_update_datas, sqe_index](const TaskInfo *task_info) {
            sqe_update_datas[sqe_index].stream_id = task_info->GetStreamId();
            sqe_update_datas[sqe_index].task_id = task_info->GetTaskID();
          });
      sqe_update_datas.emplace_back(one_task_update_data.sqe_update_arg);
    }
  }
  return SUCCESS;
}
Status ModelArgsManager::ConstructH2DCopyParams(const ModelArgsManager::ModelArgs &model_arg,
                                                const ModelArgsManager::UpdatePolicy up,
                                                ModelArgsManager::H2DCopyArg &cp_arg) {
  switch (up) {
    // 需要适配一下新的策略
    case KUpdateHostInput: {
      for (const auto &partition : model_arg.model_args_partitions) {
        if (partition.partition_type == UpdateTriggerType::KTriggerByHostInput) {
          cp_arg.len = static_cast<uint64_t>(partition.len);
          cp_arg.device_addr = model_arg.model_args_device_addr + static_cast<uint64_t>(partition.offset);
          cp_arg.host_addr =
            ValueToPtr(PtrToValue(model_arg.model_args_host_addr.get()) + static_cast<uint64_t>(partition.offset));
          return SUCCESS;
        }
      }
      return GE_GRAPH_GRAPH_NOT_EXIST;
    }
    case kUpdateModelIo: {
      bool has_partition = false;
      cp_arg.len = 0UL;
      cp_arg.device_addr = std::numeric_limits<uint64_t>::max();
      for (const auto &partition : model_arg.model_args_partitions) {
        if ((partition.partition_type == UpdateTriggerType::kTriggerByFmAndIo)||
            (partition.partition_type == UpdateTriggerType::KTriggerByHostInput)) {
          cp_arg.len += static_cast<uint64_t>(partition.len);
          UseMin(model_arg.model_args_device_addr + static_cast<uint64_t>(partition.offset),
                 ValueToPtr(PtrToValue(model_arg.model_args_host_addr.get()) + static_cast<uint64_t>(partition.offset)),
                 cp_arg.device_addr, cp_arg.host_addr);
          has_partition = true;
        }
      }
      return has_partition ? SUCCESS : GE_GRAPH_GRAPH_NOT_EXIST;
    }
    case kUpdateFmAndModelIo: {
      bool has_partition = false;
      cp_arg.len = 0UL;
      cp_arg.device_addr = std::numeric_limits<uint64_t>::max();
      for (const auto &partition : model_arg.model_args_partitions) {
        if ((partition.partition_type == UpdateTriggerType::kTriggerByFmAndIo) ||
            (partition.partition_type == UpdateTriggerType::kTriggerByFm ) ||
            (partition.partition_type == UpdateTriggerType::KTriggerByHostInput)) {
          cp_arg.len += static_cast<uint64_t>(partition.len);
          UseMin(model_arg.model_args_device_addr + static_cast<uint64_t>(partition.offset),
                 ValueToPtr(PtrToValue(model_arg.model_args_host_addr.get()) + static_cast<uint64_t>(partition.offset)),
                 cp_arg.device_addr, cp_arg.host_addr);
          has_partition = true;
        }
      }
      return has_partition ? SUCCESS : GE_GRAPH_GRAPH_NOT_EXIST;
    }
    case kInitOneTime:
      cp_arg.len = 0UL;
      cp_arg.device_addr = model_arg.model_args_device_addr;
      cp_arg.host_addr = model_arg.model_args_host_addr.get();
      for (const auto &partition : model_arg.model_args_partitions) {
        cp_arg.len += static_cast<uint64_t>(partition.len);
      }
      GE_ASSERT_TRUE(cp_arg.len > 0UL, "Placement %s does not have a partition",
                     GetArgsPlacementStr(model_arg.placement));
      return SUCCESS;
    default:
      GELOGE(INTERNAL_ERROR, "unexpected update policy %d", static_cast<int32_t>(up));
      return FAILED;
  }
}
Status ModelArgsManager::AllocFixedAddrs(const TaskNodeMap &task_node_map,
                                         const TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs) {
  std::vector<int64_t> offsets;
  int64_t total_len = 0;
  GE_ASSERT_SUCCESS(PlanFixedMemoryLayout(task_node_map, fixed_addrs, total_len, offsets));
  if (total_len == 0) {
    GELOGD("No need to alloc fixed memory for model %u(%s)", davinci_model_->GetModelId(),
           davinci_model_->GetOmName().c_str());
    return SUCCESS;
  }

  // 历史上的处理，fixed地址直接申请ts内存，仍然延续这个逻辑
  const auto mem_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, static_cast<uint32_t>(total_len));
  fixed_addr_bulk_.device_addr = davinci_model_->MallocDynamicMemory(static_cast<size_t>(total_len), mem_type);
  GE_ASSERT_NOTNULL(fixed_addr_bulk_.device_addr, "Failed to alloc fixed memory, rts memory type %u, size %lld",
                    mem_type, total_len);
  GELOGI("Alloc fixed memory size %lld, rts type %u, addr %p for model %u(%s)", total_len, mem_type,
         fixed_addr_bulk_.device_addr, davinci_model_->GetModelId(), davinci_model_->GetOmName().c_str());

  // 本来也不多，不需要太精确，尽可能不要出现扩容就行了，万一扩容了也没关系
  fixed_addr_bulk_.pieces.reserve(offsets.size() * 2UL);
  for (size_t i = 0U; i < offsets.size(); ++i) {
    for (const auto &fixed_addr : fixed_addrs.at(i)) {
      fixed_addr_bulk_.pieces.push_back({fixed_addr,
                                         PtrToValue(fixed_addr_bulk_.device_addr) + static_cast<uint64_t>(offsets[i])});
    }
  }

  return SUCCESS;
}

Status ModelArgsManager::ConstructTaskInitParams(
    const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
    const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_mem_app_type,
    std::vector<TaskRunParam> &&task_indexes_to_param, std::vector<IowAddrs> &task_indexes_to_init_param) const {
  // update refresh_type and memory_app_type of all addrs
  task_indexes_to_init_param.reserve(task_indexes_to_param.size());
  for (size_t i = 0UL; i < task_indexes_to_refresh_type.size(); ++i) {
    auto &param = task_indexes_to_param[i];
    IowAddrs init_param = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs),
                           std::move(param.parsed_workspace_addrs)};
    for (size_t j = 0UL; j < init_param.input_logic_addrs.size(); ++j) {
      auto &addr = init_param.input_logic_addrs[j];
      addr.support_refresh = static_cast<bool>(task_indexes_to_refresh_type[i].input_refresh_types[j]);
      addr.memory_type = static_cast<uint64_t>(
          logical_addrs_to_mem_app_type.at(std::pair<uint64_t, uint64_t>(addr.memory_type, addr.logic_addr)));
    }
    for (size_t j = 0UL; j < init_param.output_logic_addrs.size(); ++j) {
      auto &addr = init_param.output_logic_addrs[j];
      addr.support_refresh = static_cast<bool>(task_indexes_to_refresh_type[i].output_refresh_types[j]);
      addr.memory_type = static_cast<uint64_t>(
          logical_addrs_to_mem_app_type.at(std::pair<uint64_t, uint64_t>(addr.memory_type, addr.logic_addr)));
    }
    for (size_t j = 0UL; j < init_param.workspace_logic_addrs.size(); ++j) {
      auto &addr = init_param.workspace_logic_addrs[j];
      addr.support_refresh = static_cast<bool>(task_indexes_to_refresh_type[i].workspace_refresh_types[j]);
      addr.memory_type = static_cast<uint64_t>(
          logical_addrs_to_mem_app_type.at(std::pair<uint64_t, uint64_t>(addr.memory_type, addr.logic_addr)));
    }
    task_indexes_to_init_param.emplace_back(std::move(init_param));
  }

  // update fixed addrs
  for (const auto &fap : fixed_addr_bulk_.pieces) {  // fap: fixed addr piece
    AddrDesc *addr_desc;
    switch (fap.desc.iow_index_type) {
      case TaskArgsRefreshTypeClassifier::kInput:
        addr_desc = &(task_indexes_to_init_param.at(fap.desc.task_index).input_logic_addrs.at(fap.desc.iow_index));
        break;
      case TaskArgsRefreshTypeClassifier::kOutput:
        addr_desc = &(task_indexes_to_init_param.at(fap.desc.task_index).output_logic_addrs.at(fap.desc.iow_index));
        break;
      case TaskArgsRefreshTypeClassifier::kWorkspace:
        addr_desc = &(task_indexes_to_init_param.at(fap.desc.task_index).workspace_logic_addrs.at(fap.desc.iow_index));
        break;
      default:
        GELOGE(INTERNAL_ERROR, "Unexpected iow type %d when init task infos",
               static_cast<int32_t>(fap.desc.iow_index_type));
        return FAILED;
    }
    addr_desc->logic_addr = fap.device_addr;
    addr_desc->memory_type = static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix);
    addr_desc->support_refresh = false;
  }

  return SUCCESS;
}
Status ModelArgsManager::ValidateTaskRunParam(const std::vector<TaskArgsDesc> &args_descs) const {
  std::map<ArgsPlacement, int32_t> placement_counts;
  for (const auto &args_desc : args_descs) {
    GE_ASSERT_TRUE((++placement_counts[args_desc.placement] <= 1),
                   "Placement %d has multiple records", static_cast<int32_t>(args_desc.placement));
  }
  return SUCCESS;
}
Status ModelArgsManager::ParseModelTaskDef(domi::ModelTaskDef &model_task_def,
                                           std::vector<TaskRunParam> &task_indexes_to_run_param,
                                           TaskNodeMap &task_node_map) {
  const auto need_log = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  const size_t task_size = static_cast<size_t>(model_task_def.task_size());
  task_list_ptr_->resize(task_size);

  davinci_model_->ResetDumpFsmState();
  for (size_t i = 0UL; i < task_size; ++i) {
    domi::TaskDef *const task_def = model_task_def.mutable_task(static_cast<int32_t>(i));
    auto &task_info = task_list_ptr_->at(i);  
    task_info = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task_def->type()));
    GE_ASSERT_NOTNULL(task_info, "Failed to create task info from type %d, task index %zu", task_def->type(), i);
    GE_ASSERT_SUCCESS(task_info->ParseTaskRunParam(*task_def, davinci_model_, task_indexes_to_run_param[i]),
                      "task index:%zu ParseTaskRunParam failed", i);
    GE_ASSERT_SUCCESS(ValidateTaskRunParam(task_indexes_to_run_param[i].args_descs),
                      "task index %zu occurred multiple placement, task_type is %d", i, task_def->type());
    has_args_ = (has_args_) || (!task_indexes_to_run_param[i].args_descs.empty());

    const auto op_index = task_info->ParseOpIndex(*task_def);
    GE_ASSERT_SUCCESS(task_node_map.AddRelation(i, op_index));  // logged in function when error

    const OpDescPtr op_desc = davinci_model_->GetOpByIndex(static_cast<uint32_t>(op_index));
    if (op_desc != nullptr) {
      GE_ASSERT_SUCCESS(
        davinci_model_->SetDumpFsmState(op_desc->GetName(),static_cast<ModelTaskType>(task_def->type())));
    }

    if (need_log) {
      DebugLogTaskRunParam(i, op_index, task_indexes_to_run_param[i], op_desc);
    }
  }
  if (!has_args_) {
    GELOGW("There no args need be managed in model");
  }
  return SUCCESS;
}

const std::vector<ModelArgsManager::ModelArgs> &ModelArgsManager::GetModelArgs() const {
  return model_args_;
}
const ModelArgsManager::FixedAddrBulk &ModelArgsManager::GetFixedAddrBulk() const {
  return fixed_addr_bulk_;
}
ModelArgsManager::UpdatePolicy ModelArgsManager::CalcUpdatePolicy(const vector<uint64_t> &active_mem_base_addr) {
  if (SECUREC_UNLIKELY(!has_args_)) {
    return kNoNeedUpdate;
  }
  if (SECUREC_UNLIKELY(last_bases_.empty())) {
    last_bases_ = active_mem_base_addr;
    return kInitOneTime;
  }
  if (SECUREC_UNLIKELY(last_bases_.size() != active_mem_base_addr.size())) {
    GELOGE(INTERNAL_ERROR, "Failed to calc update policy, last base num %zu not equal with current %zu",
           last_bases_.size(), active_mem_base_addr.size());
    return kUpdatePolicyEnd;
  }

  const size_t fm_start_id = davinci_model_->GetFmMemAllocationsStartId();
  const size_t fm_size = davinci_model_->GetFmMemAllocationsSize();
  if (SECUREC_UNLIKELY(fm_size + fm_start_id > active_mem_base_addr.size())) {
    GELOGE(INTERNAL_ERROR, "Failed to calc update policy, fm_size %zu sub fm_start_id %u "
           "should less than %zu", fm_size, fm_start_id,
           active_mem_base_addr.size());
    return kUpdatePolicyEnd;
  }

  auto reset_last_base = [this, &active_mem_base_addr] (size_t start_id, size_t end_id) {
    for (size_t i = start_id; i < end_id; ++i) {
      if (last_bases_[i] != active_mem_base_addr[i]) {
        last_bases_ = active_mem_base_addr;
        return true;
      }
    }
    return false;
  };

  if (reset_last_base(fm_start_id, fm_start_id + fm_size)) {
    return kUpdateFmAndModelIo;
  }

  // index fm_size..n: model io memory base
  // index 0...fm_start_id: model io memory base, fusion io
  if ((model_io_hit_count_ != 0UL) &&
      ((reset_last_base(fm_start_id + fm_size, active_mem_base_addr.size())) || (reset_last_base(0U, fm_start_id)))) {
    return kUpdateModelIo;
  }

  return kNoNeedUpdate;
}
Status ModelArgsManager::OnTaskDistributed(const size_t task_index, const TaskInfo *task_info) {
  const auto iter = task_indexes_to_update_data_appenders_on_distributed_.find(task_index);
  if (iter != task_indexes_to_update_data_appenders_on_distributed_.end()) {
    for (const auto &func : iter->second) {
      func(task_info);
    }
  }
  return SUCCESS;
}
ModelArgsManager::TriggerTypesToPolicies ModelArgsManager::GenerateTriggerTypesToCorrespondingUpdatePolicies() const {
  if (davinci_model_->IsFeatureBaseRefreshable()) {
    return {
        // kNoNeedUpdate 不管哪种trigger type，在初始化时总要被初始化一次，因此kInitOneTime是在所有trigger types中都有的
        SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kInitOneTime},

        // kTriggerByFm：仅在kUpdateFmAndModelIo策略中被使用
        SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdateFmAndModelIo, kInitOneTime},

        // kTriggerByFmAndIo：在kUpdateModelIo和kUpdateFmAndModelIo两种策略中被使用
        SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdateModelIo, kUpdateFmAndModelIo, kInitOneTime},

        // KTriggerByHostInput: 在kUpdateModelIo和kUpdateFmAndModelIo和KUpdateHostInput三种策略中被使用
        SmallVector<UpdatePolicy, kUpdatePolicyEnd>{KUpdateHostInput, kUpdateModelIo, kUpdateFmAndModelIo, kInitOneTime}};
  } else {
    return {SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kInitOneTime},
            // kTriggerByFm： 在fm不支持刷新时，不可能出现，给一个错误值，如果出现了在外层报错
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdatePolicyEnd},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdateModelIo, kInitOneTime},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{KUpdateHostInput, kUpdateModelIo, kInitOneTime}};
  }
}

Status ModelArgsManager::GetHostInputMem(uint64_t &host_addr, uint64_t &device_addr, uint64_t &len) {
  if (host_input_size_ == 0U) {
    return SUCCESS;
  }

  if (update_version_ == kUpdateVersionH2d ) {
    auto &model_update_data = update_policies_to_model_data_[KUpdateHostInput];
    GE_ASSERT_TRUE((model_update_data != nullptr) && (model_update_data->h2d_copy_datas.size() == 1));
    host_addr = PtrToValue(model_update_data->h2d_copy_datas[0].host_addr);
    device_addr = model_update_data->h2d_copy_datas[0].device_addr;
    len = model_update_data->h2d_copy_datas[0].len;
    GELOGI("host input mem from model args table, model_id:%u, "
      "host addr:0x%" PRIx64 ", device addr:0x%" PRIx64 ", len:%" PRIu64 ", update_version:%u, is_pcie_bar_copy:%s",
      davinci_model_->GetModelId(), host_addr, device_addr, len, update_version_, is_pcie_bar_copy_ ? "true" : "false");
  } else if (update_version_ == kUpdateVersionKernelLaunch && is_pcie_bar_copy_) {
    auto &model_update_data = update_policies_to_model_data_[KUpdateHostInput];
    GE_ASSERT_TRUE((model_update_data != nullptr) && (model_update_data->h2d_copy_datas.size() == 1));
    device_addr = model_update_data->h2d_copy_datas[0].device_addr;
    GE_ASSERT_TRUE(host_input_size_ <= model_update_data->h2d_copy_datas[0].len,
      "host_input_size:%" PRIu64 ", update len:%" PRIu64, host_input_size_, model_update_data->h2d_copy_datas[0].len);
    // 使用active membase里的长度
    len = host_input_size_;
    host_addr = PtrToValue(host_input_host_ptr_);
    GELOGI("host input mem from model args table, model_id:%u, "
      "host addr:0x%" PRIx64 ", device addr:0x%" PRIx64 ", len:%" PRIu64 ", update_version:%u, is_pcie_bar_copy:%s",
      davinci_model_->GetModelId(), host_addr, device_addr, len, update_version_, is_pcie_bar_copy_ ? "true" : "false");
  } else if (update_version_ == kUpdateVersionKernelLaunch) {
    host_addr = PtrToValue(host_input_host_ptr_);
    device_addr = host_input_device_ptr_;
    len = host_input_size_;
    GELOGI("host input mem from active mem base, model_id:%u, host addr:0x%" PRIx64 ", device addr:0x%" PRIx64
      ", len:%" PRIu64, davinci_model_->GetModelId(), host_addr, device_addr, len);
  }

  GE_ASSERT_TRUE((host_addr != 0U) && (device_addr != 0U));
  return SUCCESS;
}

}  // namespace ge
