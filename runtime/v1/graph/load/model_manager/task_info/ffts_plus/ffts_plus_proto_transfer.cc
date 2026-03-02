/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"

#include "securec.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "aicpu_task_struct.h"
#include "graph/detail/attributes_holder.h"
#include "common/checker.h"
#include "common/dump/dump_utils.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_context_adapter.h"
#include "graph/utils/op_desc_utils.h"
#include "mmpa/mmpa_api.h"

namespace {
constexpr int32_t kRequiredUserDataNum = 6;
constexpr int32_t kArgsAddrLIndex = 2;
constexpr int32_t kArgsAddrHIndex = 3;
constexpr int32_t kSoNameAddrLIndex = 0;
constexpr int32_t kSoNameAddrHIndex = 1;
constexpr int32_t kKernelNameAddrLIndex = 4;
constexpr int32_t kKernelNameAddrHIndex = 5;
constexpr size_t kHostPidIndex = 6UL;

constexpr int32_t kManualIndex = 0;
constexpr int32_t kManualAicAivCtxPcNum = 1;
constexpr int32_t kAutoNonTailIndex = 0;
constexpr int32_t kAutoTailIndex = 1;
constexpr int32_t kAutoAicAivCtxPcNum = 2;
constexpr int32_t kAutoNonTailAicCtxIndexVal0 = 0;
constexpr int32_t kAutoTailAicCtxIndex = 1;
constexpr int32_t kAutoNonTailAivCtxIndexVal2 = 2;
constexpr int32_t kAutoTailAivCtxIndex = 3;
constexpr int32_t kAutoMixAicAivCtxPcNum = 4;
constexpr size_t kAicAivCtxPcAddrNum = 1U;

constexpr uint32_t k2BitsMask = 0x00000003U;   // 2  bits, 0000,0011
constexpr uint32_t k5BitsMask = 0x0000001FU;   // 5  bits, 0001,1111

constexpr uint32_t k16BitsMask = 0x0000FFFFU;  // 16 bits, 1111,1111,1111,1111

constexpr uint32_t k17BitsMask = 0x0001FFFFU;  // 17 bits, 0000,0000,0000,0001,1111,1111,1111,1111
constexpr uint32_t k32BitsMask = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111

constexpr size_t KDumpListArgsTableSizeIndex = 1U;
constexpr size_t KDumpListSizeNumIndex = 2U;
constexpr size_t kDumpListSizeInfoIndex = 3U; // context_id | args tabe size | size num | size info

constexpr uint64_t kBitFlag8 = 0x00FFFFFFFFFFFFFFUL;
constexpr uint64_t kLevel2BitFlagWithShape = 0x0200000000000000UL;
constexpr uint64_t kLevel2BitFlagTilingData = 0x0300000000000000UL;
constexpr uint64_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr char_t const *kMaxTilingSize = "op_para_size";

const std::string kAicpuAllshape = "_AllShape";
constexpr uint32_t kFwkAicpuKernelType = 1U;
constexpr uint32_t kCustomAicpuKernelType = 4U;
const std::string kMixl2PrefixMixAic = "_mix_aic";
const std::string kMixl2PrefixMixAiv = "_mix_aiv";
const std::string kAttrCtxIdList = "_tensor_ctx_id";

const std::set<rtFftsPlusContextType_t> kSaveArgsCtxType = {
    RT_CTX_TYPE_AICORE,
    RT_CTX_TYPE_AIV,
    RT_CTX_TYPE_MIX_AIC,
    RT_CTX_TYPE_MIX_AIV,
};
constexpr uint32_t kModeInArgsFirstFieldVal0 = 0U; // mode addr at args field
constexpr uint32_t kArgsSkipFirstField = 1U; // mix ctx args first addr is not input/output addr
constexpr uint32_t kSaveTaskAddr = 1U;
}  // namespace

namespace ge {
void CleanRtFftsPlusTask(rtFftsPlusTaskInfo_t &ffts_plus_task_info) noexcept {
  if (ffts_plus_task_info.descAddrType == RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_HOST) {
    delete[] PtrToPtr<const rtFftsPlusSqe_t, const uint8_t>(ffts_plus_task_info.fftsPlusSqe);
  } else {
    delete ffts_plus_task_info.fftsPlusSqe;
  }
  ffts_plus_task_info.fftsPlusSqe = nullptr;
  ffts_plus_task_info.descBuf = nullptr;
}
std::map<rtFftsPlusContextType_t, FftsPlusProtoTransfer::CtxHandle> FftsPlusProtoTransfer::init_ctx_fun_ {
  { RT_CTX_TYPE_AICORE, &FftsPlusProtoTransfer::InitAicAivCtx },
  { RT_CTX_TYPE_AIV, &FftsPlusProtoTransfer::InitAicAivCtx },
  { RT_CTX_TYPE_WRITE_VALUE, &FftsPlusProtoTransfer::InitWriteValueCtx },
  { RT_CTX_TYPE_MIX_AIC, &FftsPlusProtoTransfer::InitMixAicAivCtx },
  { RT_CTX_TYPE_MIX_AIV, &FftsPlusProtoTransfer::InitMixAicAivCtx },
  { RT_CTX_TYPE_SDMA, &FftsPlusProtoTransfer::InitSdmaCtx },
  { RT_CTX_TYPE_FLUSH_DATA, &FftsPlusProtoTransfer::InitDataCtx },
  { RT_CTX_TYPE_INVALIDATE_DATA, &FftsPlusProtoTransfer::InitDataCtx },
  { RT_CTX_TYPE_WRITEBACK_DATA, &FftsPlusProtoTransfer::InitDataCtx },
  { RT_CTX_TYPE_AICPU, &FftsPlusProtoTransfer::InitAicpuCtx },
  { RT_CTX_TYPE_COND_SWITCH, &FftsPlusProtoTransfer::InitCondSwitchCtx },
  { RT_CTX_TYPE_CASE_SWITCH, &FftsPlusProtoTransfer::InitCaseCtx },
  { RT_CTX_TYPE_DSA, &FftsPlusProtoTransfer::InitDsaCtx }
};

Status FftsPlusProtoTransfer::Transfer(const OpDescPtr &op_desc, const domi::FftsPlusTaskDef &ffts_plus_task_def,
                                       rtFftsPlusTaskInfo_t &ffts_plus_task_info, uint8_t *const desc_buf,
                                       const size_t desc_buf_len) {
  GE_CHECK_NOTNULL(find_node_handle_);
  GE_CHECK_NOTNULL(op_desc);
  logic_stream_id_ = static_cast<uint32_t>(op_desc->GetStreamId());

  const int32_t ctx_num = ffts_plus_task_def.ffts_plus_ctx_size();
  size_t ctx_total_size = 0U;
  GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(rtFftsPlusComCtx_t), static_cast<size_t>(ctx_num), ctx_total_size));
  tiling_addr_base_dev_ = const_cast<uint8_t *>(ffts_plus_args_helper_->GetTilingDataDev());
  const size_t tiling_data_len = ffts_plus_args_helper_->GetTilingDataLen();
  tiling_addr_base_.resize(tiling_data_len);
  ffts_plus_args_helper_->SetTilingDataHost(tiling_addr_base_.data());

  rtFftsPlusSqe_t *ffts_plus_sqe = nullptr;
  if (desc_buf == nullptr) {
    ffts_plus_task_info.descAddrType = RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_HOST;
    ffts_plus_task_info.descBufLen = ctx_total_size;
    size_t sqe_ctx_size = 0U;
    GE_ASSERT_TRUE(!ge::AddOverflow(sizeof(rtFftsPlusSqe_t), ffts_plus_task_info.descBufLen, sqe_ctx_size));
    auto *const sqe_ctx = new (std::nothrow) uint8_t[sqe_ctx_size]{};
    GE_CHECK_NOTNULL(sqe_ctx);

    ffts_plus_sqe = PtrToPtr<uint8_t, rtFftsPlusSqe_t>(sqe_ctx);
    FftsPlusContextAdapter::InitFftsPlusSqe(ffts_plus_task_def.ffts_plus_sqe(), *ffts_plus_sqe);
    ffts_plus_task_info.fftsPlusSqe = ffts_plus_sqe;
    ffts_plus_task_info.descBuf = &sqe_ctx[sizeof(rtFftsPlusSqe_t)];
    GE_CHK_STATUS_RET_NOLOG(InitFftsPlusCtx(ffts_plus_task_def, &sqe_ctx[sizeof(rtFftsPlusSqe_t)], ctx_num));
  } else {
    ffts_plus_sqe = new (std::nothrow) rtFftsPlusSqe_t{};
    GE_CHECK_NOTNULL(ffts_plus_sqe);
    FftsPlusContextAdapter::InitFftsPlusSqe(ffts_plus_task_def.ffts_plus_sqe(), *ffts_plus_sqe);
    ffts_plus_task_info.fftsPlusSqe = ffts_plus_sqe;
    const size_t buffer_len = ctx_total_size;
    GE_CHECK_GE(desc_buf_len, buffer_len);
    GELOGI("Init ctx begin, node %s, args_base=0x%" PRIx64 ", ctx_num=%d",
      op_desc->GetName().c_str(), args_base_, ctx_num);
    GE_CHK_STATUS_RET_NOLOG(InitFftsPlusCtx(ffts_plus_task_def, desc_buf, ctx_num));
  }

  if (op_desc->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME) && (ctx_num == 1)) {
    const domi::FftsPlusCtxDef &ctx_def = ffts_plus_task_def.ffts_plus_ctx(0);
    ffts_plus_sqe->subType = static_cast<uint8_t>(ctx_def.context_type());
  }

  return SUCCESS;
}

static bool IsAicpuFwkCtxType(const domi::FftsPlusCtxDef &ctx_def) {
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
  if (ctx_type != RT_CTX_TYPE_AICPU) {
    return false;
  }
  const domi::FftsPlusAicpuCtxDef &aicpu_ctx_def = ctx_def.aicpu_ctx();
  if (aicpu_ctx_def.kernel_type() == kFwkAicpuKernelType) {
    GELOGD("Get aicpu ctx type.");
    return true;
  }
  return false;
}

static bool NeedSaveArgsCtxDirectly(const domi::FftsPlusCtxDef &ctx_def) {
  return (IsAicpuFwkCtxType(ctx_def)) ||
         (kSaveArgsCtxType.count(static_cast<rtFftsPlusContextType_t>(ctx_def.context_type())) > 0U);
}

void FftsPlusProtoTransfer::InitFftsPlusArgs(const domi::FftsPlusCtxDef &ctx_def) {
  if ((ctx_def.op_type() != domi::FftsPlusCtxDef::ATOMIC) && (save_ctx_args_handle_ != nullptr) &&
      (NeedSaveArgsCtxDirectly(ctx_def))) {
    size_t dump_args_offset = ffts_plus_args_helper_->GetIoAddrSize();
    const auto it = ctx_additional_data_.find(kModeInArgsFirstFieldVal0);
    if ((it != ctx_additional_data_.cend()) && (it->second.count(ctx_def.context_id()) > 0U)) {
      dump_args_offset += kArgsSkipFirstField;
    }
    GELOGD("save ctx args, op idx:%u, ctx type:%u, ctx id:%u", ctx_def.op_index(), ctx_def.context_type(),
           ctx_def.context_id());
    save_ctx_args_handle_(op_desc_, 0U, dump_args_offset * sizeof(void *), {});
  }
}

Status FftsPlusProtoTransfer::InitFftsPlusCtx(const domi::FftsPlusTaskDef &task_def, uint8_t *const ctx,
                                              const int32_t num) {
  InitAdditionalData(task_def);
  rtFftsPlusComCtx_t *const com_ctx = PtrToPtr<uint8_t, rtFftsPlusComCtx_t>(ctx);
  GE_CHECK_NOTNULL(com_ctx);
  size_t tiling_data_pos = 0U;
  for (int32_t i = 0; i < num; ++i) {
    const domi::FftsPlusCtxDef &ctx_def = task_def.ffts_plus_ctx(i);
    const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
    op_desc_ = find_node_handle_(ctx_def.op_index());
    GE_ASSERT_NOTNULL(op_desc_);
    ffts_plus_args_helper_->UpdateCtxInfo({i, op_desc_});
    if (ctx_def.op_type() == domi::FftsPlusCtxDef::ATOMIC) {
      is_atomic_op_type_ = true;
    } else {
      is_atomic_op_type_ = false;
    }
    block_dim_ = 0U;
    if (op_desc_ != nullptr) {
      GE_CHK_STATUS_RET_NOLOG(InitOpRunInfo(tiling_data_pos, tiling_addr_base_.data()));
      std::vector<std::string> orig_names;
      if (AttrUtils::GetListStr(op_desc_, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, orig_names) &&
          (!orig_names.empty())) {
        FusionOpInfo fusion_op_info;
        fusion_op_info.stream_id = logic_stream_id_;
        fusion_op_info.op_index = ctx_def.op_index();
        fusion_op_info.original_op_names = orig_names;
        fusion_op_info.op_name = op_desc_->GetName();
        fusion_op_info_.emplace_back(fusion_op_info);
      }
      InitFftsPlusArgs(ctx_def);
    }

    GELOGI("Init ctx %d in FftsPlusTask, context_type=%u, op_index=%u, name=%s", i, ctx_type, ctx_def.op_index(),
           op_desc_->GetName().c_str());

    l0_dump_list_.clear();
    auto *const temp_ctx = PtrAdd<rtFftsPlusComCtx_t>(com_ctx, static_cast<size_t>(num), static_cast<size_t>(i));
    GE_ASSERT_SUCCESS(FftsPlusContextAdapter::ParseCtxByType(ctx_type, ctx_def, temp_ctx),
                      "Parse context type [%u] failed, index:[%d]", static_cast<uint32_t>(ctx_type), i);
    const auto it = init_ctx_fun_.find(ctx_type);
    if (it != init_ctx_fun_.end()) {
      GE_CHK_STATUS_RET(it->second(this, ctx_def, temp_ctx), "Init ffts_plus ctx failed, ctx_index:%d, type:[%u]", i,
                        static_cast<uint32_t>(ctx_type));
    }

    // 通过回调，将l0_dump_list_赋给task_info
    if ((l0_dump_list_.size() > 0U) && (save_l0_dump_info_handle_ != nullptr)) {
      save_l0_dump_info_handle_(l0_dump_list_);
    }
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitOpRunInfo(size_t &tiling_data_pos, uint8_t *tiling_addr) {
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  if (is_atomic_op_type_) {
    tiling_info = op_desc_->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, tiling_info);
  } else {
    tiling_info = op_desc_->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, tiling_info);
  }
  if (tiling_info != nullptr) {
    const std::string memcheck_info = ffts_plus_args_helper_->GetMemCheckInfo(op_desc_->GetName());
    auto &memcheck_info_cache = tiling_info_addr_pos_[tiling_info];
    const auto iter = memcheck_info_cache.find(memcheck_info);
    if (iter != memcheck_info_cache.end()) {
      GELOGD("Has cache tiling data with pos %" PRIu64, iter->second);
      return SUCCESS;
    }
    std::stringstream &tiling_stream = tiling_info->GetAllTilingData();
    const size_t tiling_data_len = static_cast<size_t>(tiling_stream.str().size());
    if (tiling_data_len == 0U) {
      return SUCCESS;
    }
    const size_t memcheck_info_size = memcheck_info.size();
    if (tiling_addr == nullptr) {
      tiling_addr_base_.resize(tiling_data_len + memcheck_info_size);
      tiling_addr = tiling_addr_base_.data();
      ffts_plus_args_helper_->SetTilingDataHost(tiling_addr_base_.data());
    }
    auto *const cur_data_buf = reinterpret_cast<char_t *>(tiling_addr + tiling_data_pos);
    (void)tiling_stream.rdbuf()->pubseekoff(0U, std::ios_base::beg);  // rewind
    const auto rd_len = tiling_stream.rdbuf()->sgetn(cur_data_buf, static_cast<int64_t>(tiling_data_len));
    if (static_cast<size_t>(rd_len) != tiling_data_len) {
      GELOGE(INTERNAL_ERROR, "Copy tiling data failed, data pos: %zu, tiling size: %zu, rd_len: %" PRId64 ".",
        tiling_data_pos, tiling_data_len, rd_len);
      return INTERNAL_ERROR;
    }
    GELOGD("Copy tiling data, op_name is %s, tiling addr is %p, data pos: %zu, tiling size: %zu.",
           op_desc_->GetName().c_str(), tiling_addr, tiling_data_pos, tiling_data_len);
    if (!memcheck_info.empty()) {
      char_t *const memcheck_info_buf = cur_data_buf + tiling_data_len;
      GE_ASSERT_TRUE(tiling_addr_base_.size() >= (tiling_data_pos + tiling_data_len),
                     "Tiling addr base size: %zu should not less than tiling data pos: %zu add tiling data len: %zu",
                     tiling_addr_base_.size(), tiling_data_pos, tiling_data_len);
      const size_t buffer_size = tiling_addr_base_.size() - tiling_data_pos - tiling_data_len;
      GE_ASSERT_RT_OK(rtMemcpy(memcheck_info_buf, buffer_size, memcheck_info.data(), memcheck_info_size,
                               RT_MEMCPY_HOST_TO_HOST),
                               "Copy memcheck info to tiling buf failed, buffer_size: %zu, memcheck_info_size: %zu",
                               buffer_size, memcheck_info_size);
      GELOGD("Copy memcheck info, op_name is %s, memcheck addr is %p, data pos: %zu, memcheck size: %zu.",
          op_desc_->GetName().c_str(), memcheck_info_buf, tiling_data_pos + tiling_data_len, memcheck_info_size);
    }
    memcheck_info_cache.emplace(std::make_pair(memcheck_info, tiling_data_pos));
    block_dim_ = tiling_info->GetBlockDim();
    tiling_data_pos += MemSizeAlign(tiling_data_len + memcheck_info_size);
  }
  return SUCCESS;
}

void FftsPlusProtoTransfer::GetScheduleModeFromRunInfo(uint32_t &schedule_mode) const {
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  if (is_atomic_op_type_) {
    tiling_info = op_desc_->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, tiling_info);
  } else {
    tiling_info = op_desc_->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, tiling_info);
  }
  if (tiling_info != nullptr) {
    schedule_mode = tiling_info->GetScheduleMode();
  }
}

Status FftsPlusProtoTransfer::InitAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusAicAivCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAicAivCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusAicAivCtxDef &ctx_def = task_def.aic_aiv_ctx();
  if (((task_def.op_type() != domi::FftsPlusCtxDef::ATOMIC) && (is_dump_ != nullptr) && (is_dump_(op_desc_))) ||
      davinci_model_->OpNeedPrint(op_desc_)) {
    ctx->dumpSwitch = static_cast<uint8_t>(true);
  }

  uint32_t schedule_mode = ctx_def.schem();
  GetScheduleModeFromRunInfo(schedule_mode);
  ctx->schem = static_cast<uint16_t>(schedule_mode & k2BitsMask);
  GELOGD("OpName: %s set schedule mode: %u", op_desc_->GetNamePtr(), static_cast<uint32_t>(ctx->schem));
  if (block_dim_ != 0U) {
    ctx->nonTailBlockdim = static_cast<uint16_t>(block_dim_);
    ctx->tailBlockdim = static_cast<uint16_t>(block_dim_);
  }

  const uint64_t task_param_ptr_base =
      static_cast<uint64_t>(args_base_ + (sizeof(void *) * ffts_plus_args_helper_->GetIoAddrSize()));
  GELOGD("FftsPlusAicAivCtxDef: task param addr is %" PRIu64 ".", task_param_ptr_base);
  ctx->taskParamPtrBaseL = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->taskParamPtrBaseH = static_cast<uint16_t>((task_param_ptr_base >> 32U) & k16BitsMask);

  if (ctx->threadDim == 0U) {
    GELOGD("Context thread dim is zero, Dynamic shape mode.");
    return SUCCESS;
  }

  return (ctx->atm == 0U) ? InitManualAicAivCtx(task_def, *ctx) : InitAutoAicAivCtx(ctx_def, *ctx);
}

Status FftsPlusProtoTransfer::SetTilingDataAddr(size_t &tiling_data_len) {
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  if (is_atomic_op_type_) {
    tiling_info = op_desc_->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, tiling_info);
  } else {
    tiling_info = op_desc_->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, tiling_info);
  }
  if (tiling_info != nullptr) {
    tiling_data_len = tiling_info->GetAllTilingData().str().size();
    if (tiling_data_len == 0U) {
      return FAILED;
    }
    const std::string memcheck_info = ffts_plus_args_helper_->GetMemCheckInfo(op_desc_->GetName());
    tiling_data_len += memcheck_info.size();
    const auto &memcheck_info_cache = tiling_info_addr_pos_[tiling_info];
    const auto iter = memcheck_info_cache.find(memcheck_info);
    if (iter != memcheck_info_cache.end()) {
      const uint64_t tiling_addr = reinterpret_cast<uint64_t>(tiling_addr_base_dev_) + iter->second;
      GELOGD("set tiling addr, op name is %s, tiling addr is %" PRIu64 ", tiling data pos is %" PRIu64,
             op_desc_->GetName().c_str(), tiling_addr, iter->second);
      ffts_plus_args_helper_->AppendAbsoluteAddrs(tiling_addr, "tiling");
      return SUCCESS;
    }
    return FAILED;
  } else {
    return FAILED;
  }
}

Status FftsPlusProtoTransfer::InitManualAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusAicAivCtx_t &ctx) {
  const domi::FftsPlusAicAivCtxDef &ctx_def = task_def.aic_aiv_ctx();
  size_t tiling_data_len = 0U;
  for (int32_t i = 0; i < ctx_def.task_addr_size(); ++i) {
    GELOGD("index %d, task addr is 0x%" PRIx64 ", size:%u", i, ctx_def.task_addr(i), ctx_def.task_addr_size());
    if (i == (ctx_def.task_addr_size() - 1)) {
      if (SetTilingDataAddr(tiling_data_len) == SUCCESS) {
        break;
      }
    }
    ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(i));
  }

  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (ctx_def.kernel_name_size() != kManualAicAivCtxPcNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
                       kManualAicAivCtxPcNum, ctx_def.kernel_name_size());
    GELOGE(FAILED, "[Check][Param] Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
           kManualAicAivCtxPcNum, ctx_def.kernel_name_size());
    return FAILED;
  }
  GELOGD("op name is %s, kernel name is %s", op_desc_->GetName().c_str(), ctx_def.kernel_name(0).c_str());
  uint32_t i_cache_prefetch_cnt = 0U;
  void *task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    if (is_atomic_op_type_) {
      ge::NodePtr memset_node = nullptr;
      memset_node = op_desc_->TryGetExtAttr(ATTR_NAME_MEMSET_NODE, memset_node);
      if (memset_node != nullptr) {
        op_desc_ = memset_node->GetOpDesc();
      }
    }
    std::vector<std::pair<void *, uint32_t>> addr_pref_cnt;
    GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(kManualIndex), "", addr_pref_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kManualIndex).c_str());
    GE_ASSERT_EQ(addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    task_start_pc = addr_pref_cnt[0].first;
    i_cache_prefetch_cnt = addr_pref_cnt[0].second;
  }
  ctx.nonTailTaskStartPcL = static_cast<uint32_t>(PtrToValue(task_start_pc) & k32BitsMask);
  ctx.nonTailTaskStartPcH = static_cast<uint16_t>((PtrToValue(task_start_pc) >> 32U) & k16BitsMask);
  ctx.tailTaskStartPcL = static_cast<uint32_t>(PtrToValue(task_start_pc) & k32BitsMask);
  ctx.tailTaskStartPcH = static_cast<uint16_t>((PtrToValue(task_start_pc) >> 32U) & k16BitsMask);
  ctx.icachePrefetchCnt = static_cast<uint16_t>(i_cache_prefetch_cnt & k5BitsMask);

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAutoAicAivCtx(const domi::FftsPlusAicAivCtxDef &ctx_def,
                                                rtFftsPlusAicAivCtx_t &ctx) const {
  if (ctx_def.save_task_addr() == kSaveTaskAddr) {
    GE_RETURN_IF_ERROR(InitThreadIoAddrs(ctx_def, ctx.threadDim));
  }
  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (ctx_def.kernel_name_size() != kAutoAicAivCtxPcNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
                       kAutoAicAivCtxPcNum, ctx_def.kernel_name_size());
    GELOGE(FAILED, "[Check][Param] Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
           kAutoAicAivCtxPcNum, ctx_def.kernel_name_size());
    return FAILED;
  }
  uint32_t non_tail_i_cache_prefetch_cnt = 0U;
  void *non_tail_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    std::vector<std::pair<void *, uint32_t>> no_tail_addr_pref_cnt;
    GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(kAutoNonTailIndex), "", no_tail_addr_pref_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kAutoNonTailIndex).c_str());
    GE_ASSERT_EQ(no_tail_addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    non_tail_task_start_pc = no_tail_addr_pref_cnt[0].first;
    non_tail_i_cache_prefetch_cnt = no_tail_addr_pref_cnt[0].second;
  }
  ctx.nonTailTaskStartPcL = static_cast<uint32_t>(PtrToValue(non_tail_task_start_pc) & k32BitsMask);
  ctx.nonTailTaskStartPcH = static_cast<uint16_t>((PtrToValue(non_tail_task_start_pc) >> 32U) & k16BitsMask);
  uint32_t tail_i_cache_prefetch_cnt = 0U;
  void *tail_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    std::vector<std::pair<void *, uint32_t>> tail_addr_pref_cnt;
    GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(kAutoTailIndex), "", tail_addr_pref_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kAutoTailIndex).c_str());
    GE_ASSERT_EQ(tail_addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    tail_task_start_pc = tail_addr_pref_cnt[0].first;
    tail_i_cache_prefetch_cnt = tail_addr_pref_cnt[0].second;
  }

  ctx.tailTaskStartPcL = static_cast<uint32_t>(PtrToValue(tail_task_start_pc) & k32BitsMask);
  ctx.tailTaskStartPcH = static_cast<uint16_t>((PtrToValue(tail_task_start_pc) >> 32U) & k16BitsMask);
  const uint32_t i_cache_prefetch_cnt = std::min(non_tail_i_cache_prefetch_cnt, tail_i_cache_prefetch_cnt);
  ctx.icachePrefetchCnt = static_cast<uint16_t>(i_cache_prefetch_cnt & k5BitsMask);

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitWriteValueCtx(const domi::FftsPlusCtxDef &task_def,
                                                rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusWriteValueCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusWriteValueCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusWriteValueCtxDef &ctx_def = task_def.write_value_ctx();

  uint8_t *write_addr_base = nullptr;
  uint64_t mem_type = kAbsoluteMemType;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.write_addr_base()), write_addr_base, mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic write addr base is 0x%" PRIx64 ".",
      ctx_def.write_addr_base());
    return INTERNAL_ERROR;
  }
  ctx->writeAddressBaseL = static_cast<uint32_t>(PtrToValue(write_addr_base) & k32BitsMask);
  ctx->writeAddressBaseH = static_cast<uint32_t>((PtrToValue(write_addr_base) >> 32U) & k17BitsMask);

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitMixAicAivCtx(const domi::FftsPlusCtxDef &task_def,
                                               rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusMixAicAivCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusMixAicAivCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusMixAicAivCtxDef &ctx_def = task_def.mix_aic_aiv_ctx();
  if ((is_dump_ != nullptr && is_dump_(op_desc_)) || davinci_model_->OpNeedPrint(op_desc_)) {
    ctx->dumpSwitch = static_cast<uint8_t>(true);
  }
  uint32_t schedule_mode = ctx_def.schem();
  GetScheduleModeFromRunInfo(schedule_mode);
  ctx->schem = static_cast<uint16_t>(schedule_mode & k2BitsMask);
  GELOGD("OpName: %s set schedule mode: %u", op_desc_->GetNamePtr(), static_cast<uint32_t>(ctx->schem));
  if (block_dim_ != 0U) {
    ctx->nonTailBlockdim = static_cast<uint16_t>(block_dim_);
    ctx->tailBlockdim = static_cast<uint16_t>(block_dim_);
  }
  const uint64_t task_param_ptr_base =
      static_cast<uint64_t>(args_base_ + (sizeof(void *) * ffts_plus_args_helper_->GetIoAddrSize()));
  GELOGD("FftsPlusMixAicAivCtxDef: task param addr is %" PRIu64 ".", task_param_ptr_base);
  ctx->aicTaskParamPtrL = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->aicTaskParamPtrH = static_cast<uint16_t>((task_param_ptr_base >> 32U) & k16BitsMask);
  ctx->aivTaskParamPtrL = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->aivTaskParamPtrH = static_cast<uint16_t>((task_param_ptr_base >> 32U) & k16BitsMask);
  int32_t start_addr = 0;
  const auto it = ctx_additional_data_.find(kModeInArgsFirstFieldVal0);
  if ((it != ctx_additional_data_.cend()) && (it->second.count(task_def.context_id()) > 0U)) {
    uint64_t mode_addr = 0U;
    uint32_t len = 0U;
    start_addr = 1;
    GE_CHK_RT_RET(rtGetC2cCtrlAddr(&mode_addr, &len));
    ffts_plus_args_helper_->MarkModelIOAddrIndex();
    ffts_plus_args_helper_->AppendIoAddrs(mode_addr);
    GELOGD("save mode addr:0x%" PRIx64 " to mix_aic/mix_aiv args.", mode_addr);
  }

  if (ctx->threadDim == 0U) {
    GELOGD("Context thread dim is zero, Dynamic shape mode.");
    return SUCCESS;
  }
  if (ctx->atm == 0U) {
    std::vector<std::string> name_prefixes;
    (void)AttrUtils::GetListStr(op_desc_, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefixes);
    GE_CHK_STATUS_RET(InitManualMixAicAivCtx(ctx_def, name_prefixes, *ctx, start_addr),
                      "Init MixAicAivCtx in manual mode failed, node:[%s].", op_desc_->GetName().c_str());
  } else {
    GE_CHK_STATUS_RET(InitAutoMixAicAivCtx(ctx_def, *ctx, start_addr), "Init MixAicAivCtx in auto mode failed");
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::ConstructArgsFromFormat(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                                                      const ArgsFormatHolder &format_holder) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = format_holder.ir_input_2_range;
  const std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = format_holder.ir_output_2_range;
  size_t addr_idx = 0UL;
  size_t workspace_idx = 0UL;
  std::vector<std::pair<size_t, uint64_t>> dynamic_addr_start_idxes;  // start index in task_addr for each dynamic io
  const size_t begin_offset = ffts_plus_args_helper_->GetIoAddrSize();

  std::vector<size_t> level0_addr_idx;
  std::vector<size_t> level1_addr_cnt;
  std::vector<void *> context_addrs;
  for (const auto &arg_desc : format_holder.arg_descs) {
    const size_t ir_idx = static_cast<size_t>(arg_desc.ir_idx);
    switch (arg_desc.addr_type) {
      case AddrType::INPUT_DESC: {
        auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end());
        level0_addr_idx.push_back(ffts_plus_args_helper_->GetIoAddrSize());
        ffts_plus_args_helper_->AppendAbsoluteAddrs(
            0, "ptr_to_shape_desc_" + std::to_string(static_cast<int64_t>(arg_desc.addr_type))); // place holder
        dynamic_addr_start_idxes.emplace_back(addr_idx, iter->second.first);
        level1_addr_cnt.push_back(iter->second.second);
        addr_idx += iter->second.second;  // 动态输入的数量

        // level2_addr
        uint64_t level_num = iter->second.second & kBitFlag8;
        level_num |= kLevel2BitFlagWithShape;
        l0_dump_list_.push_back(level_num);
        // level1
        for (size_t i = 0UL; i < iter->second.second; ++i) {
          l0_dump_list_.push_back(iter->second.first + i);
        }

        break;
      }
      case AddrType::OUTPUT_DESC: {
        auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end());
        level0_addr_idx.push_back(ffts_plus_args_helper_->GetIoAddrSize());
        ffts_plus_args_helper_->AppendAbsoluteAddrs(
            0, "ptr_to_shape_desc_" + std::to_string(static_cast<int64_t>(arg_desc.addr_type)));  // place holder
        dynamic_addr_start_idxes.emplace_back(addr_idx, iter->second.first + op_desc_->GetInputsSize());
        addr_idx += iter->second.second;  // 动态输入的数量
        level1_addr_cnt.push_back(iter->second.second);

        // level2_addr
        uint64_t level_num = iter->second.second & kBitFlag8;
        level_num |= kLevel2BitFlagWithShape;
        l0_dump_list_.push_back(level_num);
        // level1
        for (size_t i = 0UL; i < iter->second.second; ++i) {
          l0_dump_list_.push_back(op_desc_->GetInputsSize() + iter->second.first + i);
        }
        break;
      }
      case AddrType::TILING:
      case AddrType::TILING_FFTS: {
        size_t tiling_data_len = 0U;
        if (SetTilingDataAddr(tiling_data_len) == SUCCESS) {
          tiling_data_len |= kLevel2BitFlagTilingData;
          l0_dump_list_.push_back(tiling_data_len);
          ++addr_idx;
        }
        break;
      }
      case AddrType::FFTS_ADDR: {
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        ++addr_idx;
        break;
      }
      case AddrType::HIDDEN_INPUT: {
        if (*reinterpret_cast<const HiddenInputsType *>(arg_desc.reserved) == HiddenInputsType::HCOM) {
          if (context_addrs.empty()) {
            GE_ASSERT_SUCCESS(ArgsFormatUtils::GetHcomHiddenInputs(op_desc_, *davinci_model_, context_addrs));
          }
          GE_ASSERT_TRUE(ir_idx < context_addrs.size());
          ffts_plus_args_helper_->AppendAbsoluteAddrs(
              reinterpret_cast<uint64_t>(context_addrs[ir_idx]), "hidden input");
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        }
        ++addr_idx;
        break;
      }
      case AddrType::INPUT: {
        auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end());
        const auto &range_pair = iter->second;
        if (range_pair.second == 0UL) {
          // optional input placeholder
          ffts_plus_args_helper_->AppendRtIoAddrs(0, kAbsoluteMemType);
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
          ++addr_idx;
          break;
        }
        size_t begin_idx = range_pair.first;
        for (size_t i = 0UL; i < range_pair.second; ++i) {
          const int32_t idx = static_cast<int32_t>(addr_idx);
          GE_ASSERT(idx < ctx_def.task_addr_size(), "Invalid index %d, addrs size %zu.", idx, ctx_def.task_addr_size());
          const size_t cur_offset = ffts_plus_args_helper_->GetIoAddrSize();
          l0_dump_list_.push_back(begin_idx);
          ffts_plus_args_helper_->SaveOffsetPairs(
              begin_idx++,
              static_cast<uint64_t>(cur_offset - begin_offset));  // start at ffts_addr
          ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(idx));
          ++addr_idx;
        }
        break;
      }
      case AddrType::OUTPUT: {
        auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end());
        const auto &range_pair = iter->second;
        size_t begin_idx = range_pair.first + op_desc_->GetInputsSize(); // 添加了input的偏移的
        for (size_t i = 0UL; i < range_pair.second; ++i) {
          const int32_t idx = static_cast<int32_t>(addr_idx);
          GE_ASSERT(idx < ctx_def.task_addr_size(), "Invalid index %d, addrs size %zu.", idx, ctx_def.task_addr_size());
          const size_t cur_offset = ffts_plus_args_helper_->GetIoAddrSize();
          l0_dump_list_.push_back(begin_idx);
          ffts_plus_args_helper_->SaveOffsetPairs(
              begin_idx++,
              static_cast<uint64_t>(cur_offset - begin_offset));  // start at ffts_addr
          ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(idx));
          ++addr_idx;
        }
        break;
      }
      case AddrType::PLACEHOLDER: {
        ffts_plus_args_helper_->AppendAbsoluteAddrs(0UL, "placeholder.");
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        ++addr_idx;
        break;
      }
      case AddrType::TILING_CONTEXT: {
        GE_ASSERT_SUCCESS(AssembleTilingContextArgs(arg_desc));
        // 和传统保持一致
        uint64_t tiling_data_size = kMaxTilingDataSize;
        int64_t max_size = -1;
        if (ge::AttrUtils::GetInt(op_desc_, kMaxTilingSize, max_size) && max_size > 0) {
          tiling_data_size = static_cast<uint64_t>(max_size);
        }
        tiling_data_size &= kBitFlag8;
        tiling_data_size |= kLevel2BitFlagTilingData;
        l0_dump_list_.push_back(tiling_data_size);
        ++addr_idx;
        break;
      }
      case AddrType::CUSTOM_VALUE: {
        ffts_plus_args_helper_->AppendAbsoluteAddrs(
            *reinterpret_cast<const uint64_t *>(arg_desc.reserved), "custom_value");
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        ++addr_idx;
        break;
      }
      case AddrType::INPUT_INSTANCE: {
        const int32_t idx = static_cast<int32_t>(addr_idx);
        GE_ASSERT(idx < ctx_def.task_addr_size(), "Invalid index %d, addrs size %zu.", idx, ctx_def.task_addr_size());
        const size_t cur_offset = ffts_plus_args_helper_->GetIoAddrSize();
        l0_dump_list_.push_back(ir_idx);
        ffts_plus_args_helper_->SaveOffsetPairs(ir_idx, static_cast<uint64_t>(cur_offset - begin_offset));
        ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(idx));
        ++addr_idx;
        break;
      }
      case AddrType::OUTPUT_INSTANCE: {
        const int32_t idx = static_cast<int32_t>(addr_idx);
        GE_ASSERT(idx < ctx_def.task_addr_size(), "Invalid index %d, addrs size %zu.", idx, ctx_def.task_addr_size());
        const size_t cur_offset = ffts_plus_args_helper_->GetIoAddrSize();
        size_t begin_idx = op_desc_->GetInputsSize() + ir_idx;
        l0_dump_list_.push_back(begin_idx);
        ffts_plus_args_helper_->SaveOffsetPairs(begin_idx, static_cast<uint64_t>(cur_offset - begin_offset));
        ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(idx));
        ++addr_idx;
        break;
      }
      default: {
        // ws
        const int32_t idx = static_cast<int32_t>(addr_idx);
        GE_ASSERT(idx < ctx_def.task_addr_size(), "Invalid index %d, addrs size %zu.", idx, ctx_def.task_addr_size());
        ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(idx));
        const size_t input_output_size = op_desc_->GetInputsSize() + op_desc_->GetOutputsSize();
        l0_dump_list_.push_back(input_output_size + workspace_idx); // index
        ++addr_idx;
        ++workspace_idx;
        break;
      }
    }
  }

  GE_ASSERT(format_holder.shape_infos.size() == dynamic_addr_start_idxes.size());
  GE_ASSERT(level1_addr_cnt.size() == dynamic_addr_start_idxes.size());
  GE_ASSERT(level0_addr_idx.size() == dynamic_addr_start_idxes.size());
  for (size_t idx = 0UL; idx < dynamic_addr_start_idxes.size(); ++idx) {
    const size_t ptr_offset_idx = ffts_plus_args_helper_->GetIoAddrSize();
    const uint64_t offset_addr = static_cast<uint64_t>(args_base_ + ptr_offset_idx * sizeof(uint64_t));
    // addr to ptr offset
    GE_ASSERT_SUCCESS(ffts_plus_args_helper_->UpdateIoAddrByIndex(level0_addr_idx[idx], offset_addr));

    auto &shape_info = format_holder.shape_infos[idx];
    for (auto &info : shape_info) {
      ffts_plus_args_helper_->AppendAbsoluteAddrs(info, "shape_info");
    }

    GELOGI("Node[%s] io desc idx[%zu], io desc val[%" PRIu64 "], offset idx[%zu], level1 addr cnt[%zu]",
      op_desc_->GetNamePtr(), level0_addr_idx[idx], offset_addr, ptr_offset_idx, level1_addr_cnt[idx]);
    for (size_t dyn_idx = 0UL; dyn_idx < level1_addr_cnt[idx]; ++dyn_idx) {
      const int32_t dyn_addr_idx = static_cast<int32_t>(dyn_idx + dynamic_addr_start_idxes[idx].first);
      GE_ASSERT(dyn_addr_idx < ctx_def.task_addr_size(), "addr index:[%d] is out of range", dyn_addr_idx);
      const size_t cur_offset = ffts_plus_args_helper_->GetIoAddrSize();
      ffts_plus_args_helper_->SaveOffsetPairs(dynamic_addr_start_idxes[idx].second + dyn_idx,
                                              static_cast<uint64_t>(cur_offset - begin_offset));  // start at ffts_addr
      ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(dyn_addr_idx));
    }
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::AssembleTilingContextArgs(const ArgDesc &arg_desc) {
  std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
  std::shared_ptr<TilingContextAddr> tiling_context_addr =
      op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
  GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
  const TilingContextSubType sub_type = static_cast<TilingContextSubType>(arg_desc.ir_idx);
  GE_ASSERT_TRUE(sub_type == TilingContextSubType::TILING_DATA, "sub idx [%d] is invalid.", arg_desc.ir_idx);
  ffts_plus_args_helper_->AppendAbsoluteAddrs(tiling_context_addr->tiling_data_addr, "sink tiling data.");

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitManualMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                                                     const std::vector<std::string> &kernel_name_prefixes,
                                                     rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx) {
  CtxInfo ctx_info = {};
  ffts_plus_args_helper_->GetCtxInfo(ctx_info);
  l0_dump_list_.push_back(ctx_info.ctx_id); // ctx_id
  l0_dump_list_.push_back(0U); // args table size place holder
  l0_dump_list_.push_back(0U); // size num place holder

  ArgsFormatHolder format_holder;
  size_t tiling_data_len = 0U;
  if (ffts_plus_args_helper_->CheckAndGetArgsFormats(op_desc_->GetId(), format_holder)) {
    GE_ASSERT_SUCCESS(ConstructArgsFromFormat(ctx_def, format_holder), "ConstructArgsFormFormat failed.");
  } else {
    // 设置了ffts_addr
    if (start_idx == 1) {
      l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
    }

    int32_t index = 0;
    for (int32_t i = start_idx; i < ctx_def.task_addr_size(); ++i) {
      GELOGD("index %u, task addr is 0x%" PRIx64, i, ctx_def.task_addr(i));
      if (i == (ctx_def.task_addr_size() - 1)) {
        if (SetTilingDataAddr(tiling_data_len) == SUCCESS) {
          tiling_data_len |= kLevel2BitFlagTilingData;
          l0_dump_list_.push_back(static_cast<uint64_t>(tiling_data_len));
          break;
        }
      }
      ffts_plus_args_helper_->AppendIoAddrs(ctx_def.task_addr(i));
      l0_dump_list_.push_back(static_cast<uint64_t>(index));
      index++;
    }
  }

  const size_t args_table_size = ffts_plus_args_helper_->GetCtxArgsSize(ctx_info.ctx_id);
  l0_dump_list_[KDumpListArgsTableSizeIndex] = args_table_size;
  l0_dump_list_[KDumpListSizeNumIndex] = l0_dump_list_.size() - kDumpListSizeInfoIndex;
  UpdateL0ExceptionDumpInfoSize(op_desc_, l0_dump_list_, kDumpListSizeInfoIndex);

  const size_t prefix_size = kernel_name_prefixes.size();
  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (static_cast<size_t>(ctx_def.kernel_name_size()) != prefix_size) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusMixAicAivCtxDef should be %zu, but %d exactly",
                       prefix_size, ctx_def.kernel_name_size());
    GELOGE(FAILED, "[Check][Param] Size of kernel_name in FftsPlusMixAicAivCtxDef should be %zu, but %d exactly",
           prefix_size, ctx_def.kernel_name_size());
    return FAILED;
  }
  uint32_t aic_i_cache_prefetch_cnt = 0U;
  void *aic_task_start_pc = nullptr;
  uint32_t aiv_i_cache_prefetch_cnt = 0U;
  void *aiv_task_start_pc = nullptr;

  // set default pc for tiling sink context
  std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
  std::shared_ptr<TilingContextAddr> tiling_context_addr =
      op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
  if (tiling_context_addr != nullptr) {
    GELOGI("Node [%s] set with default pc", op_desc_->GetNamePtr());
    ctx.nonTailAicTaskStartPcL = 0U;
    ctx.nonTailAicTaskStartPcH = 0U;
    ctx.tailAicTaskStartPcL = 0U;
    ctx.tailAicTaskStartPcH = 0U;
    ctx.aicIcachePrefetchCnt = 0U;

    ctx.nonTailAivTaskStartPcL = 0U;
    ctx.nonTailAivTaskStartPcH = 0U;
    ctx.tailAivTaskStartPcL = 0U;
    ctx.tailAivTaskStartPcH = 0U;
    ctx.aivIcachePrefetchCnt = 0U;
    return SUCCESS;
  }

  if (addr_pref_handle_ != nullptr) {
    for (size_t i = 0UL; i < prefix_size; ++i) {
      if (kernel_name_prefixes[i] == kMixl2PrefixMixAic) {
        std::vector<std::pair<void *, uint32_t>> addr_pref_cnt;
        GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(static_cast<int32_t>(i)),
                                            kernel_name_prefixes[i], addr_pref_cnt),
                          "Get addr and pref cnt failed, kernel_name=%s",
                          ctx_def.kernel_name(static_cast<int32_t>(i)).c_str());
        GELOGI("Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(static_cast<int32_t>(i)).c_str());
        GE_ASSERT_EQ(addr_pref_cnt.size(), kMixMultiKernelPcAddrCnt);
        aic_task_start_pc = addr_pref_cnt[0].first;
        aic_i_cache_prefetch_cnt = addr_pref_cnt[0].second;
      } else if (kernel_name_prefixes[i] == kMixl2PrefixMixAiv) {
        std::vector<std::pair<void *, uint32_t>> addr_pref_cnt;
        GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(static_cast<int32_t>(i)),
                                            kernel_name_prefixes[i], addr_pref_cnt),
                          "Get addr and pref cnt failed, kernel_name=%s",
                          ctx_def.kernel_name(static_cast<int32_t>(i)).c_str());
        GE_ASSERT_EQ(addr_pref_cnt.size(), kMixMultiKernelPcAddrCnt);
        aiv_task_start_pc = addr_pref_cnt[0].first;
        aiv_i_cache_prefetch_cnt = addr_pref_cnt[0].second;
      } else {
        std::vector<std::pair<void *, uint32_t>> addr_pref_cnt;
        // for _mix_enhanced
        GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(static_cast<int32_t>(i)),
                                            kernel_name_prefixes[i], addr_pref_cnt),
                          "Get addr and pref cnt failed, kernel_name=%s",
                          ctx_def.kernel_name(static_cast<int32_t>(i)).c_str());
        ExpandMixOnlyInfos(*op_desc_, addr_pref_cnt);
        GE_ASSERT_EQ(addr_pref_cnt.size(), kMixSingleKernelPcAddrCnt);
        aic_task_start_pc = addr_pref_cnt[kMixSingleKernelAicPcIndex].first;
        aic_i_cache_prefetch_cnt = addr_pref_cnt[kMixSingleKernelAicPcIndex].second;
        aiv_task_start_pc = addr_pref_cnt[kMixSingleKernelAivPcIndex].first;
        aiv_i_cache_prefetch_cnt = addr_pref_cnt[kMixSingleKernelAivPcIndex].second;
      }
    }
  }

  ctx.nonTailAicTaskStartPcL = static_cast<uint32_t>(PtrToValue(aic_task_start_pc) & k32BitsMask);
  ctx.nonTailAicTaskStartPcH = static_cast<uint16_t>((PtrToValue(aic_task_start_pc) >> 32U) & k16BitsMask);
  ctx.tailAicTaskStartPcL = static_cast<uint32_t>(PtrToValue(aic_task_start_pc) & k32BitsMask);
  ctx.tailAicTaskStartPcH = static_cast<uint16_t>((PtrToValue(aic_task_start_pc) >> 32U) & k16BitsMask);
  ctx.aicIcachePrefetchCnt = static_cast<uint16_t>(aic_i_cache_prefetch_cnt & k5BitsMask);

  ctx.nonTailAivTaskStartPcL = static_cast<uint32_t>(PtrToValue(aiv_task_start_pc) & k32BitsMask);
  ctx.nonTailAivTaskStartPcH = static_cast<uint16_t>((PtrToValue(aiv_task_start_pc) >> 32U) & k16BitsMask);
  ctx.tailAivTaskStartPcL = static_cast<uint32_t>(PtrToValue(aiv_task_start_pc) & k32BitsMask);
  ctx.tailAivTaskStartPcH = static_cast<uint16_t>((PtrToValue(aiv_task_start_pc) >> 32U) & k16BitsMask);
  ctx.aivIcachePrefetchCnt = static_cast<uint16_t>(aiv_i_cache_prefetch_cnt & k5BitsMask);

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAutoMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                                                   rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx) const {
  if (ctx_def.save_task_addr() == kSaveTaskAddr) {
    GE_RETURN_IF_ERROR(InitThreadIoAddrs(ctx_def, ctx.threadDim, start_idx));
  }

  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (ctx_def.kernel_name_size() != kAutoMixAicAivCtxPcNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusMixAicAivCtxDef should be %d, but %d exactly",
                       kAutoMixAicAivCtxPcNum, ctx_def.kernel_name_size());
    GELOGE(FAILED, "[Check][Param] Size of kernel_name in FftsPlusMixAicAivCtxDef should be %d, but %d exactly",
           kAutoMixAicAivCtxPcNum, ctx_def.kernel_name_size());
    return FAILED;
  }

  uint32_t non_tail_aic_i_cache_prefetch_cnt = 0U;
  void *non_tail_aic_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    std::vector<std::pair<void *, uint32_t>> non_tail_aic_addr_pref_cnt;
    GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(kAutoNonTailAicCtxIndexVal0), "",
                                        non_tail_aic_addr_pref_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s",
                      ctx_def.kernel_name(kAutoNonTailAicCtxIndexVal0).c_str());
    GE_ASSERT_EQ(non_tail_aic_addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    non_tail_aic_task_start_pc = non_tail_aic_addr_pref_cnt[0].first;
    non_tail_aic_i_cache_prefetch_cnt = non_tail_aic_addr_pref_cnt[0].second;
  }

  ctx.nonTailAicTaskStartPcL = static_cast<uint32_t>(PtrToValue(non_tail_aic_task_start_pc) & k32BitsMask);
  ctx.nonTailAicTaskStartPcH = static_cast<uint16_t>((PtrToValue(non_tail_aic_task_start_pc) >> 32U) & k16BitsMask);
  uint32_t tail_aic_i_cache_prefetch_cnt = 0U;
  void *tail_aic_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    std::vector<std::pair<void *, uint32_t>> tail_aic_addr_pref_cnt;
    GE_CHK_STATUS_RET(
        addr_pref_handle_(op_desc_, ctx_def.kernel_name(kAutoTailAicCtxIndex), "", tail_aic_addr_pref_cnt),
        "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kAutoTailAicCtxIndex).c_str());
    GE_ASSERT_EQ(tail_aic_addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    tail_aic_task_start_pc = tail_aic_addr_pref_cnt[0].first;
    tail_aic_i_cache_prefetch_cnt = tail_aic_addr_pref_cnt[0].second;
  }

  ctx.tailAicTaskStartPcL = static_cast<uint32_t>(PtrToValue(tail_aic_task_start_pc) & k32BitsMask);
  ctx.tailAicTaskStartPcH = static_cast<uint16_t>((PtrToValue(tail_aic_task_start_pc) >> 32U) & k16BitsMask);
  const uint32_t aic_i_cache_prefetch_cnt = std::min(non_tail_aic_i_cache_prefetch_cnt, tail_aic_i_cache_prefetch_cnt);
  ctx.aicIcachePrefetchCnt = static_cast<uint16_t>(aic_i_cache_prefetch_cnt & k5BitsMask);

  uint32_t non_tail_aiv_i_cache_prefetch_cnt = 0U;
  void *non_tail_aiv_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    std::vector<std::pair<void *, uint32_t>> non_tail_aiv_addr_pref_cnt;
    GE_CHK_STATUS_RET(addr_pref_handle_(op_desc_, ctx_def.kernel_name(kAutoNonTailAivCtxIndexVal2), "",
                                        non_tail_aiv_addr_pref_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s",
                      ctx_def.kernel_name(kAutoNonTailAivCtxIndexVal2).c_str());
    GE_ASSERT_EQ(non_tail_aiv_addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    non_tail_aiv_task_start_pc = non_tail_aiv_addr_pref_cnt[0].first;
    non_tail_aiv_i_cache_prefetch_cnt = non_tail_aiv_addr_pref_cnt[0].second;
  }

  ctx.nonTailAivTaskStartPcL = static_cast<uint32_t>(PtrToValue(non_tail_aiv_task_start_pc) & k32BitsMask);
  ctx.nonTailAivTaskStartPcH = static_cast<uint16_t>((PtrToValue(non_tail_aiv_task_start_pc) >> 32U) & k16BitsMask);
  uint32_t tail_aiv_i_cache_prefetch_cnt = 0U;
  void *tail_aiv_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    std::vector<std::pair<void *, uint32_t>> tail_aiv_addr_pref_cnt;
    GE_CHK_STATUS_RET(
        addr_pref_handle_(op_desc_, ctx_def.kernel_name(kAutoTailAivCtxIndex), "", tail_aiv_addr_pref_cnt),
        "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kAutoTailAivCtxIndex).c_str());
    GE_ASSERT_EQ(tail_aiv_addr_pref_cnt.size(), kAicAivCtxPcAddrNum);
    tail_aiv_task_start_pc = tail_aiv_addr_pref_cnt[0].first;
    tail_aiv_i_cache_prefetch_cnt = tail_aiv_addr_pref_cnt[0].second;
  }

  ctx.tailAivTaskStartPcL = static_cast<uint32_t>(PtrToValue(tail_aiv_task_start_pc) & k32BitsMask);
  ctx.tailAivTaskStartPcH = static_cast<uint16_t>((PtrToValue(tail_aiv_task_start_pc) >> 32U) & k16BitsMask);
  const uint32_t aiv_i_cache_prefetch_cnt = std::min(non_tail_aiv_i_cache_prefetch_cnt, tail_aiv_i_cache_prefetch_cnt);
  ctx.aivIcachePrefetchCnt = static_cast<uint16_t>(aiv_i_cache_prefetch_cnt & k5BitsMask);

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitSdmaCtx(const domi::FftsPlusCtxDef &task_def,
                                          rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusSdmaCtxDef &ctx_def = task_def.sdma_ctx();

  uint8_t *src_addr_base = nullptr;
  uint64_t src_mem_type = kAbsoluteMemType;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.src_addr_base()), src_addr_base, src_mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic src addr is 0x%" PRIx64 ".", ctx_def.src_addr_base());
    return INTERNAL_ERROR;
  }
  ctx->sourceAddressBaseL = static_cast<uint32_t>(PtrToValue(src_addr_base) & k32BitsMask);
  ctx->sourceAddressBaseH = static_cast<uint32_t>(PtrToValue(src_addr_base) >> 32U);

  uint8_t *dst_addr_base = nullptr;
  uint64_t dst_mem_type = kAbsoluteMemType;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.dst_addr_base()), dst_addr_base, dst_mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic dst addr is 0x%" PRIx64 ".", ctx_def.dst_addr_base());
    return INTERNAL_ERROR;
  }
  ctx->destinationAddressBaseL = static_cast<uint32_t>(PtrToValue(dst_addr_base) & k32BitsMask);
  ctx->destinationAddressBaseH = static_cast<uint32_t>(PtrToValue(dst_addr_base) >> 32U);

  ffts_plus_args_helper_->AppendCtxLevel1Addrs(PtrToValue(PtrToPtr<uint8_t, uint64_t>(src_addr_base)), src_mem_type,
                                               {Level1AddrType::SDMA_SRC, task_def, com_ctx, op_desc_});
  ffts_plus_args_helper_->AppendCtxLevel1Addrs(PtrToValue(PtrToPtr<uint8_t, uint64_t>(dst_addr_base)), dst_mem_type,
                                               {Level1AddrType::SDMA_DST, task_def, com_ctx, op_desc_});
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitDataCtx(const domi::FftsPlusCtxDef &task_def,
                                          rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusDataCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);

  const domi::FftsPlusDataCtxDef &ctx_def = task_def.data_ctx();
  auto logic_base = ctx_def.addr_base();
  int64_t addr_offset{0};
  if (ffts_plus_args_helper_->CheckAndGetLevel2Offset(task_def.context_id(), addr_offset)) {
    logic_base += static_cast<uint64_t>(addr_offset);  // has memory reuse
    ctx->addressOffset = 0U;
  }

  uint8_t *addr_base = nullptr;
  uint64_t mem_type = kAbsoluteMemType;
  if (run_addr_handle_ != nullptr) {
    GE_ASSERT_SUCCESS(run_addr_handle_(static_cast<uintptr_t>(logic_base), addr_base, mem_type),
                      "[Check][GetRtAddress] failed, logic addr base is 0x%" PRIx64, logic_base);
  }
  ctx->addressBaseL = static_cast<uint32_t>(PtrToValue(addr_base) & k32BitsMask);
  ctx->addressBaseH = static_cast<uint32_t>(PtrToValue(addr_base) >> 32U);
  ffts_plus_args_helper_->AppendCtxLevel1Addrs(PtrToValue(PtrToPtr<uint8_t, void>(addr_base)), mem_type,
                                               {Level1AddrType::CMO_ADDR, task_def, com_ctx, op_desc_});

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAicpuCtx(const domi::FftsPlusCtxDef &task_def,
                                           rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusAiCpuCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAiCpuCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusAicpuCtxDef &ctx_def = task_def.aicpu_ctx();
  if (is_dump_ != nullptr && is_dump_(op_desc_)) {
    ctx->dumpSwitch = static_cast<uint8_t>(true);
  }

  const size_t user_data_len = sizeof(ctx->usrData) / sizeof(uint32_t);
  if (user_data_len < static_cast<uint64_t>(kRequiredUserDataNum)) {
    REPORT_INNER_ERR_MSG("E19999", "Length of user_data in rtFftsPlusAiCpuCtx_t should not < %d, but %" PRIu64 " exactly",
                       kRequiredUserDataNum, static_cast<uint64_t>(user_data_len));
    GELOGE(FAILED, "[Check][Param] Length of user_data in rtFftsPlusAiCpuCtx_t should not < %d, but %" PRIu64 " exactly",
           kRequiredUserDataNum, user_data_len);
    return FAILED;
  }
  ctx->usrDataLength = static_cast<uint32_t>(kRequiredUserDataNum);
  const auto &kernel = ctx_def.kernel();

  // copy so_name
  const auto &so_name = kernel.so_name();
  const size_t so_name_len = so_name.size() + 1U;  // Need copy terminate byte '\0' to device.
  void *so_name_addr = davinci_model_->MallocDynamicMemory(so_name_len);
  GE_ASSERT_NOTNULL(so_name_addr);
  ext_info_addrs_.emplace_back(so_name_addr);
  GE_CHK_RT_RET(rtMemcpy(so_name_addr, so_name_len, so_name.data(), so_name_len, RT_MEMCPY_HOST_TO_DEVICE));
  ctx->usrData[kSoNameAddrLIndex] = static_cast<uint32_t>(PtrToValue(so_name_addr) & k32BitsMask);
  ctx->usrData[kSoNameAddrHIndex] = static_cast<uint32_t>(PtrToValue(so_name_addr) >> 32U);

  // get args addr
  void *args_addr = nullptr;
  GE_CHK_STATUS_RET(InitAicpuInfo(op_desc_, ctx_def, args_addr));
  ctx->usrData[kArgsAddrLIndex] = static_cast<uint32_t>(PtrToValue(args_addr) & k32BitsMask);
  ctx->usrData[kArgsAddrHIndex] = static_cast<uint32_t>(PtrToValue(args_addr) >> 32U);

  // copy kernel_name
  const auto &kernel_name = kernel.kernel_name();
  const size_t kernel_name_len = kernel_name.size() + 1U;  // Need copy terminate byte '\0' to device.
  void *kernel_name_addr = davinci_model_->MallocDynamicMemory(kernel_name_len);
  GE_ASSERT_NOTNULL(kernel_name_addr);
  ext_info_addrs_.emplace_back(kernel_name_addr);
  GE_CHK_RT_RET(
      rtMemcpy(kernel_name_addr, kernel_name_len, kernel_name.data(), kernel_name_len, RT_MEMCPY_HOST_TO_DEVICE));
  ctx->usrData[kKernelNameAddrLIndex] = static_cast<uint32_t>(PtrToValue(kernel_name_addr) & k32BitsMask);
  ctx->usrData[kKernelNameAddrHIndex] = static_cast<uint32_t>(PtrToValue(kernel_name_addr) >> 32U);
  // 适配aicpu vf场景，把userdata6刷新成hostpid
  ctx->usrData[kHostPidIndex] = static_cast<uint32_t>(mmGetPid());
  GELOGD("Init aicpu ctx user data success, node is [%s], type is [%s], so name is [%s], kernel name is [%s]",
         op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), so_name.c_str(), kernel_name.c_str());
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAicpuInfo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def,
                                            void *&addr) const {
  // aicpu fwk op
  if (ctx_def.kernel_type() == kFwkAicpuKernelType) {
    return InitAicpuFwkExtInfo(op_desc, ctx_def, addr);
  }
  if (ctx_def.kernel_type() == kCustomAicpuKernelType) {
    load_cust_aicpu_so_(op_desc, ctx_def);
  }
  return InitAicpuExtInfo(op_desc, ctx_def, addr);
}

Status FftsPlusProtoTransfer::InitAicpuFwkAddrInfo(const OpDescPtr &op_desc, uint8_t *const ori_args_addr,
                                                   const size_t args_size) const {
  const std::vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(runtime_param_, op_desc);
  if (workspace_data_addrs.empty()) {
    GELOGI("[Check][Param] workspace_data_addrs is empty in op:%s(%s), take for dynamic shape.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return SUCCESS;
  }

  const auto fwk_op_kernel = PtrToPtr<uint8_t, STR_FWK_OP_KERNEL>(ori_args_addr);
  const auto task_info_addr = PtrAdd(ori_args_addr, args_size, sizeof(STR_FWK_OP_KERNEL));
  const size_t task_info_addr_size = args_size - sizeof(STR_FWK_OP_KERNEL);
  const auto ret = CopyTaskInfoToWorkspace(op_desc, static_cast<const void *>(task_info_addr), task_info_addr_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "[copy][TaskInfo] to workspace failed, op:%s.", op_desc->GetName().c_str());
    return ret;
  }

  const uint64_t task_param_base =
      static_cast<uint64_t>(args_base_ + (sizeof(uint64_t) * ffts_plus_args_helper_->GetIoAddrSize()));
  std::vector<uint64_t> mem_types;
  std::vector<uint64_t> io_addrs = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc, mem_types);
  const std::vector<uint64_t> output_addrs = ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc, mem_types);
  (void)io_addrs.insert(io_addrs.cend(), output_addrs.cbegin(), output_addrs.cend());
  GE_ASSERT_EQ(mem_types.size(), io_addrs.size());
  for (size_t i = 0UL; i < io_addrs.size(); ++i) {
    ffts_plus_args_helper_->AppendRtIoAddrs(io_addrs[i], mem_types[i]);
  }

  fwk_op_kernel->fwkKernelBase.fwk_kernel.workspaceBaseAddr = PtrToValue(workspace_data_addrs[0U]);
  fwk_op_kernel->fwkKernelBase.fwk_kernel.inputOutputAddr = task_param_base;
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAicpuFwkExtInfo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def,
                                                  void *&addr) const {
  GELOGI("Begin to init aicpu fwk op ext info.");
  const auto &kernel = ctx_def.kernel();

  // create host args
  const size_t args_size = kernel.args().size();
  std::vector<uint8_t> ori_args_addr(kernel.args().data(), &kernel.args()[args_size]);

  // 1 get loop cond variable for tensor array
  GE_CHECK_GE(args_size, sizeof(STR_FWK_OP_KERNEL));
  const auto fwk_op_kernel = PtrToPtr<uint8_t, STR_FWK_OP_KERNEL>(ori_args_addr.data());
  GE_CHK_STATUS_RET_NOLOG(create_aicpu_session_(*fwk_op_kernel));

  const auto &ext_info = kernel.kernel_ext_info();
  std::shared_ptr<hybrid::AicpuExtInfoHandler> ext_handle;
  GE_CHK_STATUS_RET_NOLOG(InitAicpuTaskExtInfo(op_desc, ext_info, ext_handle));
  if (ext_handle == nullptr) {
    return SUCCESS; // ext_info is empty.
  }

  void *aicpu_ext_info_addr = davinci_model_->MallocDynamicMemory(ext_handle->GetExtInfoLen());
  GE_ASSERT_NOTNULL(aicpu_ext_info_addr);
  ext_info_addrs_.emplace_back(aicpu_ext_info_addr);
  GE_CHK_RT_RET(rtMemcpy(aicpu_ext_info_addr, ext_handle->GetExtInfoLen(), ext_handle->GetExtInfo(),
                         ext_handle->GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE));

  GELOGI("Node[%s] type[%s] kernel_ext_info size=%zu, aicpu_ext_info_addr=%p", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), ext_info.size(), aicpu_ext_info_addr);

  // 4 set workspace addr and input output data addr
  fwk_op_kernel->fwkKernelBase.fwk_kernel.extInfoLen = ext_info.size();
  fwk_op_kernel->fwkKernelBase.fwk_kernel.extInfoAddr = PtrToValue(aicpu_ext_info_addr);
  GE_CHK_STATUS_RET(InitAicpuFwkAddrInfo(op_desc, ori_args_addr.data(), args_size),
                    "[Init][InitAicpuFwkAddrInfo] failed, ext info size is %zu, op is %s", args_size,
                    op_desc->GetName().c_str());

  // 5. Return result
  addr = davinci_model_->MallocDynamicMemory(sizeof(STR_FWK_OP_KERNEL));
  GE_ASSERT_NOTNULL(addr);
  ext_info_addrs_.emplace_back(addr);
  GE_CHK_RT_RET(rtMemcpy(addr, sizeof(STR_FWK_OP_KERNEL), static_cast<void *>(fwk_op_kernel),
                         sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE));
  GELOGI("Init aicpu fwk op context info Success. session id: %" PRIu64,
    fwk_op_kernel->fwkKernelBase.fwk_kernel.sessionID);
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAicpuExtInfo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def,
                                               void *&addr) const {
  GELOGI("Begin to init aicpu op ext info, node:[%s], thread dim is %u.", op_desc->GetNamePtr(), ctx_def.thread_dim());
  if (ctx_def.thread_dim() == 0U) {  // Zero mark for dynamic.
    return save_aicpu_ctx_handle_(op_desc, ctx_def.kernel());
  }

  const auto &kernel = ctx_def.kernel();
  // copy args to new host addr
  const size_t args_size = kernel.args().size();
  std::vector<uint8_t> ori_args_addr(kernel.args().data(), &kernel.args()[args_size]);

  // ctx_def.atm() == 0 is aicpu manual mode(only once looping), else is auto mode
  const auto thread_num = ctx_def.thread_dim();
  const size_t task_param_offset = static_cast<size_t>(ctx_def.task_param_offset());
  GELOGI("[Init ffts plus aicpu ext info] thread num is %u, atm is %u, task_param_offset is %" PRIu64, thread_num,
         ctx_def.atm(), task_param_offset);
  const auto &ext_info = kernel.kernel_ext_info();
  std::shared_ptr<hybrid::AicpuExtInfoHandler> ext_handle;
  GE_CHK_STATUS_RET_NOLOG(InitAicpuTaskExtInfo(op_desc, ext_info, ext_handle));
  if (ext_handle == nullptr) {
    return SUCCESS;  // ext_info is empty.
  }
  size_t aicpu_ext_malloc_size = 0UL;
  GE_ASSERT_TRUE(!ge::MulOverflow(ext_handle->GetExtInfoLen(), thread_num, aicpu_ext_malloc_size));
  void *aicpu_ext_info_addr = davinci_model_->MallocDynamicMemory(aicpu_ext_malloc_size);
  GE_ASSERT_NOTNULL(aicpu_ext_info_addr);
  ext_info_addrs_.emplace_back(aicpu_ext_info_addr);
  size_t aicpu_ext_info_len = 0UL;
  GE_ASSERT_TRUE(!ge::MulOverflow((thread_num - 1U), task_param_offset, aicpu_ext_info_len));
  size_t total_len = 0UL;
  GE_ASSERT_TRUE(!ge::AddOverflow(aicpu_ext_info_len, sizeof(aicpu::AicpuParamHead), total_len));
  if (total_len >= args_size) {
    GELOGE(PARAM_INVALID, "product of (thread_num - 1U)[%u] and task_param_offset[%zu] >= args_size[%zu]",
           (thread_num - 1U), task_param_offset, args_size);
    return PARAM_INVALID;
  }
  // Dynamic shape mode will skip for thread dim is zero.
  size_t addr_inner_offset{0UL};
  for (size_t i = 0UL; i < thread_num; ++i) {
    void *const ctx_ext_info_addr = ValueToPtr(PtrToValue(aicpu_ext_info_addr) + (ext_handle->GetExtInfoLen() * i));
    GE_CHK_RT_RET(rtMemcpy(ctx_ext_info_addr, ext_handle->GetExtInfoLen(), ext_handle->GetExtInfo(),
                           ext_handle->GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE));
    GELOGI("Node[%s] type[%s] kernel_ext_info size=%zu, current thread aicpu ext_info_addr=%p",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), ext_info.size(), ctx_ext_info_addr);
    const auto aicpu_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(&ori_args_addr[i * task_param_offset]);
    aicpu_param_head->extInfoAddr = PtrToValue(ctx_ext_info_addr);
    aicpu_param_head->extInfoLength = static_cast<uint32_t>(ext_info.size());
    addr_inner_offset = i * task_param_offset + sizeof(aicpu::AicpuParamHead);
    const uint64_t io_addr = PtrToValue(&ori_args_addr[addr_inner_offset]);
    const size_t size_left = args_size - addr_inner_offset;
    GE_CHK_STATUS_RET(InitAicpuIoAddrs(op_desc, io_addr, size_left, addr_inner_offset),
                      "Init aicpu[%s] io addrs failed", op_desc->GetName().c_str());
  }
  ffts_plus_args_helper_->GetBinArsDevAddr(addr);
  size_t dump_args_offset = ffts_plus_args_helper_->GetBinArsDevOffset();
  GE_CHK_STATUS_RET(ffts_plus_args_helper_->AppendBinArgs(ori_args_addr.data(), args_size));
  dump_args_offset += sizeof(aicpu::AicpuParamHead);
  if (save_ctx_args_handle_ != nullptr) {
    save_ctx_args_handle_(op_desc, 0U, dump_args_offset, {});
  }
  GELOGI("Init aicpu op context info success, arg_base:[%p]", addr);
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAicpuTaskExtInfo(const OpDescPtr &op_desc, const std::string &ext_info,
                                                   std::shared_ptr<hybrid::AicpuExtInfoHandler> &ext_handle) const {
  if (ext_info.empty()) {
    return SUCCESS;
  }

  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  const UnknowShapeOpType unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  const uint32_t num_inputs = static_cast<uint32_t>(op_desc->GetInputsSize());
  const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());
  ext_handle = MakeShared<hybrid::AicpuExtInfoHandler>(op_desc->GetName(), num_inputs, num_outputs, unknown_type);
  GE_CHK_BOOL_RET_STATUS(ext_handle != nullptr, FAILED, "[Malloc][Memory] for aicpu_ext_handle failed!");
  GE_CHK_STATUS_RET(ext_handle->Parse(ext_info),
                    "[Parse][KernelExtInfo] failed, kernel_ext_info_size=%zu, op:%s.",
                    ext_info.size(), op_desc->GetName().c_str());
  GE_CHK_STATUS_RET(ext_handle->UpdateSessionInfoId(aicpu_get_session_id_()),
                    "[Update][SessionInfoSessionId] failed, op:%s", op_desc->GetName().c_str());
  GELOGD("Update aicpu_task ext_info session_info session_id is %" PRIu64, aicpu_get_session_id_());
  // update bit map if ioaddrs shall be updated.
  const bool need_update = davinci_model_->IsFeatureBaseRefreshable() || davinci_model_->HasZeroCopyAddr(op_desc);
  // 1 means static(no need update), 0 means dynamic(need_update)
  GE_CHK_STATUS_RET(ext_handle->UpdateExecuteMode(!need_update), "[Update][ExecuteMode] failed.");
  GELOGD("Node [%s] update aicpu_task ext_info bit_map execute mode to %d.",
         op_desc->GetNamePtr(), static_cast<int32_t>(!need_update));
  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAicpuAllshape, all_shape);
  if (all_shape) {
    GELOGD("Aicpu all_shape kernel need to update io shape.");
    for (uint32_t i = 0U; i < num_inputs; i++) {
      const auto input_desc = op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET(ext_handle->UpdateInputShapeAndType(i, *input_desc),
                        "[Call][UpdateInputShapeAndType] Input[%u] update input shape failed, op:%s.",
                        i, op_desc->GetName().c_str());
    }
    for (uint32_t j = 0U; j < num_outputs; j++) {
      const auto output_desc = op_desc->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);
      GE_CHK_STATUS_RET(ext_handle->UpdateOutputShapeAndType(j, *output_desc),
                        "[Call][UpdateOutputShapeAndType] Output[%u] update output shape failed, op:%s.",
                        j, op_desc->GetName().c_str());
    }
  }
  bool is_blocking_aicpu_op = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op);
  GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc->GetNamePtr(),
         static_cast<int32_t>(is_blocking_aicpu_op));
  if (!is_blocking_aicpu_op) {
    return SUCCESS;
  }

  if (UpdateEventIdForAicpuBlockingOp(op_desc, ext_handle) != SUCCESS) {
    GELOGE(FAILED, "[Call][UpdateEventIdForAicpuBlockingOp] failed for op:%s(%s)",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const {
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
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

Status FftsPlusProtoTransfer::UpdateEventIdForAicpuBlockingOp(
    const OpDescPtr &op_desc, const std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle) const {
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
  if (get_event_id_(op_desc, event_id) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get event id failed for op:%s(%s).", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][EventId] Get event id failed for op:%s(%s)", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return FAILED;
  }
  if (ext_handle->UpdateEventId(event_id) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Update event id failed for op:%s(%s).", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str());
    GELOGE(FAILED, "[Update][EventId] Update event id failed for op:%s(%s)", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return FAILED;
  }
  GELOGI("Update event_id=%u success", event_id);
  return SUCCESS;
}

Status FftsPlusProtoTransfer::CopyTaskInfoToWorkspace(const OpDescPtr &op_desc, const void *const task_info_addr,
                                                      const size_t task_info_addr_size) const {
  // Userspace copy need virtual address.
  const std::vector<int64_t> workspace_data_sizes = ModelUtils::GetWorkspaceSize(op_desc);
  const std::vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(runtime_param_, op_desc);
  if (workspace_data_addrs.empty() || workspace_data_sizes.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) workspace addr:%zu or size:%zu empty, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       workspace_data_addrs.size(), workspace_data_sizes.size());
    GELOGE(FAILED, "[Check][Param] Node:%s invalid workspace, addrs is %zu, size is %zu.", op_desc->GetName().c_str(),
           workspace_data_addrs.size(), workspace_data_sizes.size());
    return FAILED;
  }

  if (workspace_data_addrs[0U] == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) workspace addr is nullptr, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Node:%s workspace addrs is null.", op_desc->GetName().c_str());
    return FAILED;
  }

  if ((workspace_data_sizes[0U] < static_cast<int64_t>(task_info_addr_size)) ||
      (workspace_data_sizes[0U] > static_cast<int64_t>(runtime_param_.mem_size))) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Node:%s(%s) workspace size:%" PRId64 " < task info size:%zu or workspace size > total mem "
                       "size %" PRIu64 ", check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), workspace_data_sizes[0U],
                       task_info_addr_size, runtime_param_.mem_size);
    GELOGE(FAILED,
           "[Check][Param] Node:%s workspace size is %" PRId64 ", task info size is %zu or workspace size > total mem "
           "size %" PRIu64 ", check invalid",
           op_desc->GetName().c_str(), workspace_data_sizes[0U], task_info_addr_size, runtime_param_.mem_size);
    return FAILED;
  }

  GE_CHK_RT_RET(rtMemcpy(workspace_data_addrs[0U], static_cast<uint64_t>(workspace_data_sizes[0U]),
                         task_info_addr, static_cast<uint64_t>(task_info_addr_size), RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitAicpuIoAddrs(const OpDescPtr &op_desc, const uint64_t io_addr,
                                               const size_t io_size, const size_t addr_inner_offset) const {
  std::vector<uint64_t> mem_types;
  std::vector<uint64_t> io_addrs = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc, mem_types);
  GE_ASSERT_EQ(mem_types.size(), io_addrs.size());
  const std::vector<uint64_t> output_addrs = ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc, mem_types);
  (void)io_addrs.insert(io_addrs.cend(), output_addrs.cbegin(), output_addrs.cend());
  if ((io_addrs.size() * sizeof(uint64_t)) > io_size) {
    GELOGE(PARAM_INVALID, "io_addrs size mismatch, ioaddr num[%zu], args_size_left:[%zu]", io_addrs.size(), io_size);
    return PARAM_INVALID;
  }
  GE_ASSERT_EQ(mem_types.size(), io_addrs.size());
  const uint64_t *const thread_offset = PtrToPtr<void, uint64_t>(ValueToPtr(io_addr));
  for (size_t i = 0UL; i < io_addrs.size(); ++i) {
    const uint64_t step = thread_offset[i];
    GELOGI("Node[%s] type[%s] index[%zu], input addr=%" PRIx64 ", thread_offset:[%" PRIu64 "]", op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), i, io_addrs[i], step);
    io_addrs[i] += step;
    GE_CHK_STATUS_RET(ffts_plus_args_helper_->AppendAicpuAddrs(io_addrs[i], mem_types[i],
                                                               addr_inner_offset + i * sizeof(uint64_t)));
  }

  GELOGI("Node[%s] type[%s] io_addrs size is [%zu]", op_desc->GetNamePtr(), op_desc->GetTypePtr(), io_addrs.size());
  if (!io_addrs.empty()) {
    const size_t addrs_size = sizeof(uint64_t) * io_addrs.size();
    const errno_t sec_ret = memcpy_s(ValueToPtr(io_addr), io_size, io_addrs.data(), addrs_size);
    if (sec_ret != EOK) {
      REPORT_INNER_ERR_MSG("E19999", "Call memcpy_s fail, io_size:%zu, addrs_size:%" PRIu64 ", ret:%d",
                        io_size, static_cast<uint64_t>(addrs_size), sec_ret);
      GELOGE(FAILED, "[Call][Memcpy] failed, io_size:%zu, addrs_size:%" PRIu64 ", ret:%d", io_size, addrs_size, sec_ret);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitCondSwitchCtx(const domi::FftsPlusCtxDef &task_def,
                                                rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusCondSwitchCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCondSwitchCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusCondSwitchCtxDef &ctx_def = task_def.cond_switch_ctx();
  uint8_t *addr_base_0 = nullptr;
  uint64_t mem_type = kAbsoluteMemType;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.load_addr0_base()), addr_base_0, mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic load addr0 base is 0x%" PRIx64 ".",
      ctx_def.load_addr0_base());
    return INTERNAL_ERROR;
  }
  ctx->loadAddress0BaseL = static_cast<uint32_t>(PtrToValue(addr_base_0) & k32BitsMask);
  ctx->loadAddress0BaseH = static_cast<uint32_t>((PtrToValue(addr_base_0) >> 32U) & k17BitsMask);

  uint8_t *addr_base_1 = nullptr;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.load_addr1_base()), addr_base_1, mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic load addr1 base is 0x%" PRIx64 ".",
      ctx_def.load_addr1_base());
    return INTERNAL_ERROR;
  }
  ctx->loadAddress1BaseL = static_cast<uint32_t>(PtrToValue(addr_base_1) & k32BitsMask);
  ctx->loadAddress1BaseH = static_cast<uint32_t>((PtrToValue(addr_base_1) >> 32U) & k17BitsMask);

  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitCaseCtx(const domi::FftsPlusCtxDef &task_def,
                                          rtFftsPlusComCtx_t *const com_ctx) const {
  if (!task_def.has_case_switch_ctx()) {
    return SUCCESS;
  }

  rtFftsPlusCaseSwitchCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCaseSwitchCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusCaseSwitchCtxDef &ctx_def = task_def.case_switch_ctx();

  uint8_t *addr_base_0 = nullptr;
  uint64_t mem_type = kAbsoluteMemType;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.load_addr0_base()), addr_base_0, mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic load addr0 base is 0x%" PRIx64 ".",
      ctx_def.load_addr0_base());
    return INTERNAL_ERROR;
  }
  ctx->loadAddress0BaseL = static_cast<uint32_t>(PtrToValue(addr_base_0) & k32BitsMask);
  ctx->loadAddress0BaseH = static_cast<uint32_t>((PtrToValue(addr_base_0) >> 32U) & k17BitsMask);

  uint8_t *addr_base_1 = nullptr;
  if ((run_addr_handle_ != nullptr) &&
      (run_addr_handle_(static_cast<uintptr_t>(ctx_def.load_addr1_base()), addr_base_1, mem_type) != SUCCESS)) {
    GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic load addr1 base is 0x%" PRIx64 ".",
      ctx_def.load_addr1_base());
    return INTERNAL_ERROR;
  }
  ctx->loadAddress1BaseL = static_cast<uint32_t>(PtrToValue(addr_base_1) & k32BitsMask);
  ctx->loadAddress1BaseH = static_cast<uint32_t>((PtrToValue(addr_base_1) >> 32U) & k17BitsMask);

  return SUCCESS;
}

void FftsPlusProtoTransfer::InitAdditionalData(const domi::FftsPlusTaskDef &task_def) {
  GELOGD("init additional data start, size:%d", task_def.additional_data_size());
  for (int32_t i = 0; i < task_def.additional_data_size(); ++i) {
    const domi::AdditionalDataDef &additional_data = task_def.additional_data(i);
    auto &additional_context = ctx_additional_data_[additional_data.data_type()];
    for (int32_t j = 0; j < additional_data.context_id_size(); ++j) {
      (void)additional_context.emplace(additional_data.context_id(j));
      GELOGD("additional data type:%u, context id:%u", additional_data.data_type(), additional_data.context_id(j));
    }
  }
}

template <typename T>
Status FftsPlusProtoTransfer::GetThreadIoAddr(ge::AttrHolder &obj, const T &ctx_def, std::vector<uint64_t> &addr_offset,
                                              const uint32_t thread_id, const int32_t idx,
                                              const int32_t start_idx) const {
  if (((start_idx + idx) >= ctx_def.task_addr_size()) || (idx >= ctx_def.task_addr_offset_size())) {
    GELOGE(FAILED, "task_addr start_idx[%d], size[%d], task_addr_offset size[%d], but need read count[%i]", start_idx,
           ctx_def.task_addr_size(), ctx_def.task_addr_offset_size(), idx);
    return FAILED;
  }
  std::vector<int64_t> sub_offsets;
  uint64_t logic_addr = 0;
  int64_t sub_offset = 0;
  std::string set_info;
  (void)ge::AttrUtils::GetListInt(&obj, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_OFFSETS, sub_offsets);
  if (static_cast<size_t>(thread_id) < sub_offsets.size()) {
    sub_offset = sub_offsets[static_cast<size_t>(thread_id)];
    set_info = "from ge";
    std::vector<uint32_t> ctx_id_vec;
    (void) AttrUtils::GetListInt(obj, kAttrCtxIdList, ctx_id_vec);
    if (static_cast<size_t>(thread_id) < ctx_id_vec.size()) {
      ffts_plus_args_helper_->SaveLevel2Offset(ctx_id_vec[static_cast<size_t>(thread_id)], sub_offset);
    }
  } else {
    sub_offset = static_cast<int64_t>(thread_id * ctx_def.task_addr_offset(idx));
    set_info = "from engine";
  }

  if (thread_id == 1U) {
    addr_offset.emplace_back(static_cast<uint64_t>(sub_offset));
  }
  logic_addr = static_cast<uint64_t>(ctx_def.task_addr(start_idx + idx) + static_cast<uint64_t>(sub_offset));
  GELOGD("task base addr is %" PRIu64 ", offset is %" PRId64 " set by %s, thread id is %u, logic addr is 0x%" PRIx64,
         ctx_def.task_addr(start_idx + idx), sub_offset, set_info.c_str(), thread_id, logic_addr);

  ffts_plus_args_helper_->AppendIoAddrs(logic_addr);
  return SUCCESS;
}

template <typename T>
Status FftsPlusProtoTransfer::InitThreadIoAddrs(const T &ctx_def, const uint32_t thread_dim,
                                                const int32_t start_idx) const {
  vector<uint64_t> task_addr_offset;
  for (uint32_t thread_id = 0U; thread_id < thread_dim; ++thread_id) {
    int32_t idx = 0;
    for (size_t i = 0U; i < op_desc_->GetAllInputsSize(); ++i) {
      const GeTensorDescPtr tensor_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
      if (tensor_desc != nullptr) {
        GE_RETURN_IF_ERROR(GetThreadIoAddr(*tensor_desc, ctx_def, task_addr_offset, thread_id, idx, start_idx));
        idx++;
      }
    }

    for (size_t i = 0U; i < op_desc_->GetOutputsSize(); ++i) {
      const GeTensorDescPtr tensor_desc = op_desc_->MutableOutputDesc(static_cast<uint32_t>(i));
      if (tensor_desc != nullptr) {
        GE_RETURN_IF_ERROR(GetThreadIoAddr(*tensor_desc, ctx_def, task_addr_offset, thread_id, idx, start_idx));
        idx++;
      }
    }

    const std::vector<int64_t> v_workspace_num = op_desc_->GetWorkspace();
    for (size_t i = 0U; i < v_workspace_num.size(); ++i) {
      GE_RETURN_IF_ERROR(GetThreadIoAddr(*op_desc_, ctx_def, task_addr_offset, thread_id, idx, start_idx));
      idx++;
    }
  }

  if (!task_addr_offset.empty()) {
    (void)op_desc_->SetExtAttr("task_addr_offset", task_addr_offset);
  }
  return SUCCESS;
}

void FftsPlusProtoTransfer::SaveFirstLevelAddressDumpInfo(const OpDescPtr &op_desc,
                                                          const std::vector<uint64_t> &input_data_addrs,
                                                          const std::vector<uint64_t> &output_data_addrs) const {
  std::vector<uint64_t> dump_args;
  (void)dump_args.insert(dump_args.cend(), input_data_addrs.cbegin(), input_data_addrs.cend());
  (void)dump_args.insert(dump_args.cend(), output_data_addrs.cbegin(), output_data_addrs.cend());
  std::vector<uintptr_t> args;
  for (size_t i = 0U; i < dump_args.size(); i++) {
    args.emplace_back(static_cast<uintptr_t>(dump_args[i]));
  }
  save_ctx_args_handle_(op_desc, 0U, 0U, args);
}

Status FftsPlusProtoTransfer::GetAndCheckDsaAddr(
    const OpDescPtr &op_desc, std::vector<std::pair<uint64_t, uint64_t>> &input_data_addrs,
    std::vector<std::pair<uint64_t, uint64_t>> &output_data_addrs,
    std::vector<std::pair<uint64_t, uint64_t>> &workspace_data_addrs) const {
  std::vector<uint64_t> iow_mem_types;
  std::vector<uint64_t> iow_addrs;
  iow_addrs = ModelUtils::GetInputDataAddrsValue(runtime_param_, op_desc, iow_mem_types);
  if (iow_addrs.size() < kDSAInputAddrSize) {
    GELOGE(INTERNAL_ERROR, "Node %s input addr size %zu is wrong", op_desc->GetName().c_str(), iow_addrs.size());
    return INTERNAL_ERROR;
  }
  GE_ASSERT_EQ(iow_mem_types.size(), iow_addrs.size());
  for (size_t i = 0U; i < iow_mem_types.size(); ++i) {
    input_data_addrs.emplace_back(iow_addrs[i], iow_mem_types[i]);
  }
  iow_mem_types.clear();
  iow_addrs.clear();

  iow_addrs = ModelUtils::GetOutputDataAddrsValue(runtime_param_, op_desc, iow_mem_types);
  if (iow_addrs.size() != kDSAOutputAddrSize) {
    GELOGE(INTERNAL_ERROR, "Node %s output addr size %zu is wrong", op_desc->GetName().c_str(), iow_addrs.size());
    return INTERNAL_ERROR;
  }
  GE_ASSERT_EQ(iow_mem_types.size(), iow_addrs.size());
  for (size_t i = 0U; i < iow_mem_types.size(); ++i) {
    output_data_addrs.emplace_back(iow_addrs[i], iow_mem_types[i]);
  }
  iow_mem_types.clear();
  iow_addrs.clear();

  iow_addrs = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param_, op_desc, iow_mem_types);
  if (iow_addrs.empty()) {
    GELOGE(INTERNAL_ERROR, "Node %s workspace addr size %zu is wrong", op_desc->GetName().c_str(), iow_addrs.size());
    return INTERNAL_ERROR;
  }
  GE_ASSERT_EQ(iow_mem_types.size(), iow_addrs.size());
  for (size_t i = 0U; i < iow_mem_types.size(); ++i) {
    workspace_data_addrs.emplace_back(iow_addrs[i], iow_mem_types[i]);
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::InitDsaWorkSpace(const OpDescPtr &op_desc,
                                               const domi::FftsPlusDsaCtxDef &ctx_def,
                                               const std::vector<std::pair<uint64_t, uint64_t>> &input_datas) const {
  const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
  GE_CHECK_SIZE(workspace_size.size());

  if (ctx_def.input1_value_or_ptr() == kDSASetInputAddr) {
    ffts_plus_args_helper_->AppendRtIoAddrs(input_datas[kDSAArgsInputAddrSize - 2U].first,
                                            input_datas[kDSAArgsInputAddrSize - 2U].second);
    if ((input_datas.size() == kDSAStateInputAddrSize) ||
        ((input_datas.size() == kDSAArgsInputAddrSize) && (workspace_size.size() == kDSAWorkspaceAddrSize))) {
      ffts_plus_args_helper_->AppendRtIoAddrs(input_datas[kDSAArgsInputAddrSize - 1U].first,
                                              input_datas[kDSAArgsInputAddrSize - 1U].second);
    }
  } else {
    uint64_t input_data[2] = {0U, 0U};
    const std::string &input1 = ctx_def.args().input1_value_or_addr();
    auto mem_ret = memcpy_s(&input_data[0U], sizeof(uint64_t), input1.c_str(), input1.size());
    GE_ASSERT_EOK(mem_ret, "dsa input data memcpy failed.");
    if ((input_datas.size() == kDSAStateInputAddrSize) ||
        ((input_datas.size() == kDSAArgsInputAddrSize) && (workspace_size.size() == kDSAWorkspaceAddrSize))) {
      const std::string &input2 = ctx_def.args().input2_value_or_addr();
      mem_ret = memcpy_s(&input_data[1U], sizeof(uint64_t), input2.c_str(), input2.size());
      GE_ASSERT_EOK(mem_ret, "dsa input data memcpy failed.");
    }

    // io_addrs_的排布需要和host_args里iow的排布保持一致
    ffts_plus_args_helper_->AppendAbsoluteAddrs(input_data[0], "int dsa workspace first");
    ffts_plus_args_helper_->AppendAbsoluteAddrs(input_data[1], "int dsa workspace second");
  }
  return SUCCESS;
}

Status FftsPlusProtoTransfer::AssembleDsaWorkSpaceByInput(const OpDescPtr &op_desc,
                                                          const domi::FftsPlusDsaCtxDef &ctx_def,
                                                          const std::vector<uint64_t> &input_data_addrs,
                                                          const uint64_t workspace_input_addr) const {
  const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
  GE_CHECK_SIZE(workspace_size.size());

  if (ctx_def.input1_value_or_ptr() == kDSASetInputAddr) {
    vector<uint64_t> input_addr{input_data_addrs[kDSAArgsInputAddrSize - 2U]};
    if ((input_data_addrs.size() == kDSAStateInputAddrSize) ||
        ((input_data_addrs.size() == kDSAArgsInputAddrSize) && (workspace_size.size() == kDSAWorkspaceAddrSize))) {
      input_addr.push_back(input_data_addrs[kDSAArgsInputAddrSize - 1U]);
    }
    GE_CHK_RT_RET(rtMemcpy(ValueToPtr(workspace_input_addr),
                           static_cast<uint64_t>(workspace_size[workspace_size.size() -1U]),
                           input_addr.data(), sizeof(uint64_t) * input_addr.size(), RT_MEMCPY_HOST_TO_DEVICE));
  } else {
    uint64_t input_data[2] = {0U, 0U};
    const std::string &input1 = ctx_def.args().input1_value_or_addr();
    auto mem_ret = memcpy_s(&input_data[0U], sizeof(uint64_t), input1.c_str(), input1.size());
    if (mem_ret != EOK) {
      GELOGE(INTERNAL_ERROR, "dsa input data memcpy failed.");
      return INTERNAL_ERROR;
    }
    if ((input_data_addrs.size() == kDSAStateInputAddrSize) ||
        ((input_data_addrs.size() == kDSAArgsInputAddrSize) && (workspace_size.size() == kDSAWorkspaceAddrSize))) {
      const std::string &input2 = ctx_def.args().input2_value_or_addr();
      mem_ret = memcpy_s(&input_data[1U], sizeof(uint64_t), input2.c_str(), input2.size());
      if (mem_ret != EOK) {
        GELOGE(INTERNAL_ERROR, "dsa input data memcpy failed.");
        return INTERNAL_ERROR;
      }
    }
    GE_CHK_RT_RET(rtMemcpy(ValueToPtr(workspace_input_addr),
                           static_cast<uint64_t>(workspace_size[workspace_size.size() -1U]), &input_data[0U],
                           sizeof(input_data), RT_MEMCPY_HOST_TO_DEVICE));
  }
  return SUCCESS;
}

void FftsPlusProtoTransfer::AppendDsaCtxLevel1RrefreshInfo(const OpDescPtr &op_desc,
                                                           const domi::FftsPlusDsaCtxDef &ctx_def,
                                                           rtFftsPlusDsaCtx_t *const ctx) {
  AppendCtxLevel1RrefreshInfo(dsa_update_ctx_helper_.output_data_refresh_infos[0],
                              &(ctx->dsaCfgResultAddrHigh),
                              &(ctx->dsaCfgResultAddrLow),
                              op_desc,
                              Level1AddrType::DSA_OUTPUT);

  if (dsa_update_ctx_helper_.workspace_data_refresh_infos.size() == kDSAWorkspaceAddrSize) {
    AppendCtxLevel1RrefreshInfo(dsa_update_ctx_helper_.workspace_data_refresh_infos[0],
                                &(ctx->dsaCfgStateAddrHigh),
                                &(ctx->dsaCfgStateAddrLow),
                                op_desc,
                                Level1AddrType::DSA_WORKSPACE);
  } else {
    AppendCtxLevel1RrefreshInfo(
      dsa_update_ctx_helper_.input_data_refresh_infos[dsa_update_ctx_helper_.input_data_refresh_infos.size() - 1U],
      &(ctx->dsaCfgStateAddrHigh),
      &(ctx->dsaCfgStateAddrLow),
      op_desc,
      Level1AddrType::DSA_INPUT);
  }

  if (ctx_def.seed_value_or_ptr() == kDSASetInputAddr) {
    AppendCtxLevel1RrefreshInfo(dsa_update_ctx_helper_.input_data_refresh_infos[1],
                                &(ctx->dsaCfgSeedHigh),
                                &(ctx->dsaCfgSeedLow),
                                op_desc,
                                Level1AddrType::DSA_INPUT);
  }

  if (ctx_def.random_count_value_or_ptr() == kDSASetInputAddr) {
    AppendCtxLevel1RrefreshInfo(dsa_update_ctx_helper_.input_data_refresh_infos[0],
                                &(ctx->dsaCfgNumberHigh),
                                &(ctx->dsaCfgNumberLow),
                                op_desc,
                                Level1AddrType::DSA_INPUT);
  }

  GELOGI("%s %s append dsa context level1 refresh ino success.",
     op_desc->GetName().c_str(), op_desc->GetType().c_str());
}

void FftsPlusProtoTransfer::AssembleDsaCtxByRealAddr(const OpDescPtr &op_desc, const domi::FftsPlusDsaCtxDef &ctx_def,
                                                     const std::vector<uint64_t> &input_data_addrs,
                                                     const std::vector<uint64_t> &output_data_addrs,
                                                     const std::vector<uint64_t> &workspace_data_addrs,
                                                     rtFftsPlusDsaCtx_t *const ctx) const {
  // dump addr need real addr
  SaveFirstLevelAddressDumpInfo(op_desc, input_data_addrs, output_data_addrs);
  // update ctx addr info
  const uint64_t dev_output_addr = output_data_addrs[0U];
  ctx->dsaCfgResultAddrLow = static_cast<uint32_t>(dev_output_addr & k32BitsMask);
  ctx->dsaCfgResultAddrHigh = static_cast<uint32_t>(dev_output_addr >> k32Bits);

  if (workspace_data_addrs.size() == kDSAWorkspaceAddrSize) {
    const uint64_t workspace_philox_count_addr = workspace_data_addrs[0U];
    ctx->dsaCfgStateAddrLow = static_cast<uint32_t>(workspace_philox_count_addr & k32BitsMask);
    ctx->dsaCfgStateAddrHigh = static_cast<uint32_t>(workspace_philox_count_addr >> k32Bits);
  } else {
    ctx->dsaCfgStateAddrLow = static_cast<uint32_t>(input_data_addrs[input_data_addrs.size() - 1U] & k32BitsMask);
    ctx->dsaCfgStateAddrHigh = static_cast<uint32_t>(input_data_addrs[input_data_addrs.size() - 1U] >> k32Bits);
  }

  const uint64_t seed_value_or_addr = (ctx_def.seed_value_or_ptr() == kDSASetInputAddr)
                                          ? input_data_addrs[1U]
                                          : *(PtrToPtr<char_t, uint64_t>(ctx_def.args().seed_value_or_addr().c_str()));
  ctx->dsaCfgSeedLow = static_cast<uint32_t>(seed_value_or_addr & k32BitsMask);
  ctx->dsaCfgSeedHigh = static_cast<uint32_t>(seed_value_or_addr >> k32Bits);

  const uint64_t random_count_value_or_addr =
      (ctx_def.random_count_value_or_ptr() == kDSASetInputAddr)
          ? input_data_addrs[0U]
          : *(PtrToPtr<char_t, uint64_t>(ctx_def.args().random_count_value_or_addr().c_str()));
  ctx->dsaCfgNumberLow = static_cast<uint32_t>(random_count_value_or_addr & k32BitsMask);
  ctx->dsaCfgNumberHigh = static_cast<uint32_t>(random_count_value_or_addr >> k32Bits);
  GELOGI("%s %s update ctx by real addr success.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
}

Status FftsPlusProtoTransfer::InitDsaCtx(const domi::FftsPlusCtxDef &task_def,
                                         rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusDsaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDsaCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const auto &op_desc = find_node_handle_(task_def.op_index());
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Init DsaCtx for %s.", op_desc->GetName().c_str());
  // for data dump
  if ((is_dump_ != nullptr) && (is_dump_(op_desc))) {
    ctx->dumpSwitch = static_cast<uint8_t>(true);
  }
  std::vector<std::pair<uint64_t, uint64_t>> input_datas;
  std::vector<uint64_t> input_data_addrs;
  std::vector<std::pair<uint64_t, uint64_t>> output_datas;
  std::vector<uint64_t> output_data_addrs;
  std::vector<std::pair<uint64_t, uint64_t>> workspace_datas;
  std::vector<uint64_t> workspace_data_addrs;
  GE_CHK_STATUS_RET(GetAndCheckDsaAddr(op_desc, input_datas, output_datas, workspace_datas),
                    "GetAndCheckDsaAddr failed for %s", op_desc->GetName().c_str());

  // workspace的值使用args tabel表的固定地址
  const uint64_t task_param_ptr_base =
      static_cast<uint64_t>(args_base_ + (sizeof(void *) * ffts_plus_args_helper_->GetIoAddrSize()));
  ctx->dsaCfgParamAddrLow = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->dsaCfgParamAddrHigh = static_cast<uint32_t>(task_param_ptr_base >> k32Bits);
  const domi::FftsPlusDsaCtxDef &ctx_def = task_def.dsa_ctx();
  (void)InitDsaWorkSpace(op_desc, ctx_def, input_datas);

  for (const auto &addr_and_type : input_datas) {
    ffts_plus_args_helper_->AppendCtxLevel1Addrs(addr_and_type.first, addr_and_type.second,
                                                 {Level1AddrType::DSA_INPUT, task_def, com_ctx, op_desc_});
    input_data_addrs.push_back(addr_and_type.first);
  }
  for (const auto &addr_and_type : workspace_datas) {
    // 保存全量信息，只在workspace_data_addrs的size为2的场景下，使用workspace_data_addrs[0],
    // 其它场景不使用workspace_data_addrs
    ffts_plus_args_helper_->AppendCtxLevel1Addrs(addr_and_type.first, addr_and_type.second,
                                                 {Level1AddrType::DSA_WORKSPACE, task_def, com_ctx, op_desc_});
    workspace_data_addrs.push_back(addr_and_type.first);
  }
  for (const auto &addr_and_type : output_datas) {
    ffts_plus_args_helper_->AppendCtxLevel1Addrs(addr_and_type.first, addr_and_type.second,
                                                 {Level1AddrType::DSA_OUTPUT, task_def, com_ctx, op_desc_});
    output_data_addrs.push_back(addr_and_type.first);
  }

  return SUCCESS;
}

void FftsPlusProtoTransfer::AppendCtxLevel1RrefreshInfo(const TaskArgsRefreshInfo &info,
                                                        const uint32_t *addr_high,
                                                        const uint32_t *addr_low,
                                                        const OpDescPtr &op_desc,
                                                        const Level1AddrType addr_type) {
  TaskArgsRefreshInfo info_high_32_bit = {
      info.id,
      info.offset,
      0UL,
      PtrToValue(PtrToPtr<uint32_t, void>(addr_high)) - static_cast<uint64_t>(pis_args_host_base_),
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrHigh32Bit
  };
  GELOGI("%s %s addr type:%d, high32 level1 refresh info:[%s]", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), static_cast<int16_t>(addr_type), info_high_32_bit.ToString().c_str());
  ctx_level1_refresh_info_list_.emplace_back(std::move(info_high_32_bit));

  TaskArgsRefreshInfo info_low_32_bit = {
      info.id,
      info.offset,
      0UL,
      PtrToValue(PtrToPtr<uint32_t, void>(addr_low)) - static_cast<uint64_t>(pis_args_host_base_),
      ArgsPlacement::kArgsPlacementHbm,
      ArgsFormatPolicy::kAddrLow32Bit
  };
  GELOGI("%s %s addr type:%d, low32 level1 refresh info:[%s]", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), static_cast<int16_t>(addr_type), info_low_32_bit.ToString().c_str());
  ctx_level1_refresh_info_list_.emplace_back(std::move(info_low_32_bit));
}

Status FftsPlusProtoTransfer::GenCtxLevel1RefreshInfo(const AddrType2CtxAddrInfo &addr_type_2_ctx_addr_infos,
                                                      const std::vector<TaskArgsRefreshInfo> &args_fresh_infos,
                                                      const size_t start_idx,
                                                      const size_t real_addr_size,
                                                      std::vector<TaskArgsRefreshInfo> &ctx_level1_args_fresh_infos) {
  GE_ASSERT_TRUE(addr_type_2_ctx_addr_infos.size() == real_addr_size,
                 "Level1 addr_size [%zu] mismatches with ctx size:[%zu].", real_addr_size,
                 addr_type_2_ctx_addr_infos.size());

  for (size_t i = 0UL; i < addr_type_2_ctx_addr_infos.size(); ++i) {
    const size_t index = start_idx + i;
    const auto args_refresh_info = args_fresh_infos[index];  // index must be valid
    const auto &ctx_addr_info = addr_type_2_ctx_addr_infos[i];
    const auto &op_desc = ctx_addr_info.op;
    GE_CHECK_NOTNULL(op_desc);
    switch (ctx_addr_info.level_1_addr_type) {
      case Level1AddrType::CMO_ADDR: {
        rtFftsPlusDataCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(ctx_addr_info.rts_ctx);
        GE_CHECK_NOTNULL(ctx);
        AppendCtxLevel1RrefreshInfo(args_refresh_info,
          &(ctx->addressBaseH), &(ctx->addressBaseL), op_desc, Level1AddrType::CMO_ADDR);
        break;
      }
      case Level1AddrType::SDMA_SRC: {
        rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(ctx_addr_info.rts_ctx);
        GE_CHECK_NOTNULL(ctx);
        AppendCtxLevel1RrefreshInfo(args_refresh_info,
          &(ctx->sourceAddressBaseH), &(ctx->sourceAddressBaseL), op_desc, Level1AddrType::SDMA_SRC);
        break;
      }
      case Level1AddrType::SDMA_DST: {
        rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(ctx_addr_info.rts_ctx);
        AppendCtxLevel1RrefreshInfo(args_refresh_info,
          &(ctx->destinationAddressBaseH), &(ctx->destinationAddressBaseL), op_desc, Level1AddrType::SDMA_DST);
        break;
      }
      case Level1AddrType::DSA_INPUT: {
        dsa_update_ctx_helper_.input_data_refresh_infos.emplace_back(args_refresh_info);
        GELOGI("%s %s get input data refresh info, allocation id %u, offset %" PRIu64,
               op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_refresh_info.id, args_refresh_info.offset);
        break;
      }
      case Level1AddrType::DSA_WORKSPACE: {
        dsa_update_ctx_helper_.workspace_data_refresh_infos.emplace_back(args_refresh_info);
        GELOGI("%s %s get workspace data real refresh info, allocation id %u, offset %" PRIu64,
               op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_refresh_info.id, args_refresh_info.offset);
        break;
      }
      case Level1AddrType::DSA_OUTPUT: {
        dsa_update_ctx_helper_.output_data_refresh_infos.emplace_back(args_refresh_info);
        GELOGI("%s %s get output data refresh info, allocation id %u, offset %" PRIu64,
               op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_refresh_info.id, args_refresh_info.offset);
        if (dsa_update_ctx_helper_.IsAllRealAddrReady()) {
          rtFftsPlusDsaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDsaCtx_t>(ctx_addr_info.rts_ctx);
          GE_CHECK_NOTNULL(ctx);
          AppendDsaCtxLevel1RrefreshInfo(op_desc, ctx_addr_info.ctx_def.dsa_ctx(), ctx);
          dsa_update_ctx_helper_.Refresh();
        }
        break;
      }
      default:
        GELOGE(FAILED, "Invalid level addr type:[%u] for op [%s %s].",
               static_cast<uint32_t>(ctx_addr_info.level_1_addr_type), op_desc->GetName().c_str(),
               op_desc->GetType().c_str());
        return FAILED;
    }
  }

  ctx_level1_args_fresh_infos.insert(ctx_level1_args_fresh_infos.end(),
                                    ctx_level1_refresh_info_list_.begin(),
                                    ctx_level1_refresh_info_list_.end());
  ctx_level1_refresh_info_list_.clear();
  return SUCCESS;
}

Status FftsPlusProtoTransfer::UpdateCtxLevel1Addrs(const AddrType2CtxAddrInfo &addr_type_2_ctx_addr_infos,
                                                   const std::vector<uint64_t> &real_addrs, const size_t start_idx,
                                                   const size_t real_addr_size) {
  if (addr_type_2_ctx_addr_infos.size() != real_addr_size) {
    GELOGE(FAILED, "Level1 addr_size [%zu] mismatches with ctx size:[%zu].", real_addr_size,
           addr_type_2_ctx_addr_infos.size());
    return FAILED;
  }

  for (size_t i = 0UL; i < addr_type_2_ctx_addr_infos.size(); ++i) {
    const size_t index = start_idx + i;
    const auto real_addr = real_addrs[index];  // index must be valid
    const uint32_t addr_high = static_cast<uint32_t>(real_addr >> 32U);
    const uint32_t addr_low = static_cast<uint32_t>(real_addr & k32BitsMask);
    const auto &ctx_addr_info = addr_type_2_ctx_addr_infos[i];
    const auto &op_desc = ctx_addr_info.op;
    GE_CHECK_NOTNULL(op_desc);
    switch (ctx_addr_info.level_1_addr_type) {
      case Level1AddrType::CMO_ADDR: {
        rtFftsPlusDataCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(ctx_addr_info.rts_ctx);
        GE_CHECK_NOTNULL(ctx);
        ctx->addressBaseH = addr_high;
        ctx->addressBaseL = addr_low;
        break;
      }
      case Level1AddrType::SDMA_SRC: {
        rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(ctx_addr_info.rts_ctx);
        GE_CHECK_NOTNULL(ctx);
        ctx->sourceAddressBaseH = addr_high;
        ctx->sourceAddressBaseL = addr_low;
        break;
      }
      case Level1AddrType::SDMA_DST: {
        rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(ctx_addr_info.rts_ctx);
        GE_CHECK_NOTNULL(ctx);
        ctx->destinationAddressBaseH = addr_high;
        ctx->destinationAddressBaseL = addr_low;
        break;
      }
      case Level1AddrType::DSA_INPUT: {
        dsa_update_ctx_helper_.input_data_addrs.emplace_back(real_addr);
        GELOGI("[IMAS] %s %s get real input data addr %" PRIu64, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
               real_addr);
        break;
      }
      case Level1AddrType::DSA_WORKSPACE: {
        dsa_update_ctx_helper_.workspace_data_addrs.emplace_back(real_addr);
        GELOGI("[IMAS] %s %s get real workspace addr %" PRIu64, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
               real_addr);
        break;
      }
      case Level1AddrType::DSA_OUTPUT: {
        dsa_update_ctx_helper_.output_data_addrs.emplace_back(real_addr);
        GELOGI("[IMAS] %s %s get real output data addr %" PRIu64, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
               real_addr);
        if (dsa_update_ctx_helper_.IsAllRealAddrReady()) {
          rtFftsPlusDsaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDsaCtx_t>(ctx_addr_info.rts_ctx);
          GE_CHECK_NOTNULL(ctx);
          AssembleDsaCtxByRealAddr(op_desc, ctx_addr_info.ctx_def.dsa_ctx(), dsa_update_ctx_helper_.input_data_addrs,
                                   dsa_update_ctx_helper_.output_data_addrs,
                                   dsa_update_ctx_helper_.workspace_data_addrs, ctx);
          dsa_update_ctx_helper_.Refresh();
        }

        break;
      }
      default:
        GELOGE(FAILED, "Invalid addr type:[%u] for op [%s %s].", static_cast<uint32_t>(ctx_addr_info.level_1_addr_type),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
    }
  }
  return SUCCESS;
}

void FftsPlusProtoTransfer::ExpandMixOnlyInfos(const OpDesc &op_desc,
                                               std::vector<std::pair<void *, uint32_t>> &addr_pref_cnt) {
  if (addr_pref_cnt.size() != kMixSingleOnlyKernelPcAddrCnt) {
    return;
  }
  (void)addr_pref_cnt.insert(addr_pref_cnt.cend(), addr_pref_cnt.cbegin(), addr_pref_cnt.cend());
  GELOGI("%s %s is mix only, double pc and pref_cnt.", op_desc.GetName().c_str(), op_desc.GetType().c_str());
}
}  // namespace ge
