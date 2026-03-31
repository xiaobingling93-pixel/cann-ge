/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_task_info.h"

#include <sstream>
#include <iomanip>
#include "graph/def_types.h"
#include "common/checker.h"
#include "common/dump/dump_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "aicpu_task_struct.h"
#include "framework/common/types.h"
#include "common/runtime_api_wrapper.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "common/op_tiling/tiling_memcheck.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph/args_format_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"

namespace {
constexpr size_t kDescBufAlignedBytes = 128UL;
constexpr size_t kContextLen = 128UL;
constexpr size_t kDescBufLen = 32UL;
constexpr int32_t kHexDumpWidth = 8;
constexpr uint32_t kAicpuTfKernel = 1U;
constexpr int64_t kDefaultDimInfo = 0x100000001;
constexpr uint64_t kDefaultShapeNum = 0x100000000U;
constexpr uint64_t kDsaWorkspaceMaxSize = 16U; // dsa workspace数组最后一个值指向的内存最多保存两个地址
const std::set<rtFftsPlusContextType_t> kCalTilingSizeCtxType = {
    RT_CTX_TYPE_AICORE,
    RT_CTX_TYPE_AIV,
    RT_CTX_TYPE_MIX_AIC,
    RT_CTX_TYPE_MIX_AIV,
};
const std::set<rtFftsPlusContextType_t> kUnsupportedDumpCtxTypes = {
    RT_CTX_TYPE_FLUSH_DATA,
    RT_CTX_TYPE_INVALIDATE_DATA,
    RT_CTX_TYPE_WRITEBACK_DATA,
};

void DumpFftsPlusTask(const void *const desc_buf, const size_t desc_buf_len, const uint32_t stream_id,
                      const uint32_t task_id) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }
  GELOGD("========Dump FftsPlusTask-begin-context, descBufLen =%" PRIu64 "========.stream_id:%u, task_id:%u",
         desc_buf_len, stream_id, task_id);
  std::stringstream ss;
  for (size_t i = 0UL; i < (desc_buf_len / kContextLen); ++i) {
    const uint32_t *const buf = ge::PtrToPtr<const void, const uint32_t>(desc_buf) + (i * kDescBufLen);
    for (size_t j = 0UL; j < kDescBufLen; ++j) {
      if (buf[j] > 0U) {
        ss << "idx:[" << &(std::dec) << j << "]=[0x" << std::setfill('0') << std::setw(kHexDumpWidth) << &(std::hex)
           << buf[j] << "]";
      }
    }
    GELOGD("========Dump FftsPlusTask-The %zu context: [%s]", i, ss.str().c_str());
    ss.clear();
    ss.str("");
  }
  GELOGD("========Dump FftsPlusTask-end-context========");
}

void GetCtxSliceThreadInfo(const domi::FftsPlusCtxDef &ctx_def, size_t &thread_id, size_t &thread_dim,
                           size_t &thread_offset) {
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
  switch (ctx_type) {
    case RT_CTX_TYPE_MIX_AIV:
    case RT_CTX_TYPE_MIX_AIC:
      thread_id = ctx_def.mix_aic_aiv_ctx().thread_id();
      thread_dim = ctx_def.mix_aic_aiv_ctx().thread_dim();
      if (ctx_type == RT_CTX_TYPE_MIX_AIC) {
        thread_offset = ctx_def.mix_aic_aiv_ctx().aic_task_param_ptr_offset();
      } else {
        thread_offset = ctx_def.mix_aic_aiv_ctx().aiv_task_param_ptr_offset();
      }
      break;
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      thread_id = ctx_def.aic_aiv_ctx().thread_id();
      thread_dim = ctx_def.aic_aiv_ctx().thread_dim();
      thread_offset = ctx_def.aic_aiv_ctx().task_param_ptr_offset();
      break;
    case RT_CTX_TYPE_AICPU:
      thread_id = ctx_def.aicpu_ctx().thread_id();
      thread_dim = ctx_def.aicpu_ctx().thread_dim();
      thread_offset = 0U;
      break;
    case RT_CTX_TYPE_DSA:
      thread_id = ctx_def.dsa_ctx().thread_id();
      thread_dim = ctx_def.dsa_ctx().thread_dim();
      thread_offset = 0U;
      break;
    default:
      break;
  }
}

void AppendShapeDesc(const ge::GeTensorDesc &tensor_desc, std::vector<int64_t> &shape_infos) {
  const auto &shape = tensor_desc.GetShape();
  if (shape.IsScalar()) {
    shape_infos.push_back(kDefaultDimInfo);
    shape_infos.push_back(0x1);  // shape value [1]
  } else {
    uint64_t dim_info{kDefaultShapeNum};
    dim_info |= (static_cast<uint64_t>(shape.GetDimNum()));
    shape_infos.push_back(static_cast<int64_t>(dim_info));
    for (const int64_t dim : shape.GetDims()) {
      shape_infos.push_back(dim);
    }
  }
}
}  // namespace

namespace ge {
Status FftsPlusTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                              const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  (void)persistent_workspace;
  (void)iow_addrs;
  (void)davinci_model;
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("FftsPlusTaskInfo Init Start, node: %s.", op_desc_->GetNamePtr());
  GE_ASSERT_SUCCESS(InitArgsBaseInfo(args));
  GE_ASSERT_NOTNULL(ffts_flus_args_helper_);
  GE_ASSERT_SUCCESS(AssembleOtherArgsByArgsBase(task_def.ffts_plus_task()));
  ffts_proto_transfer_ =
      MakeUnique<FftsPlusProtoTransfer>(static_cast<uintptr_t>(PtrToValue(args_)), ffts_flus_args_helper_.get(),
                                        davinci_model_->GetRuntimeParam(), ext_info_addrs_);
  GE_CHECK_NOTNULL(ffts_proto_transfer_);
  SetTransferCallback(*ffts_proto_transfer_);
  ffts_proto_transfer_->SetPisArgsHostBase(
    static_cast<uintptr_t>(PtrToValue(PtrToPtr<uint8_t, void>(pis_args_host_base_))));

  InitDescBufInfo();
  GE_ASSERT_SUCCESS(ffts_proto_transfer_->Transfer(op_desc_, task_def.ffts_plus_task(), ffts_plus_task_info_,
                                                   desc_buf_host_, desc_buffer_len_));
  fusion_op_info_.clear();
  fusion_op_info_ = ffts_proto_transfer_->GetAllFusionOpInfo();
  GE_ASSERT_SUCCESS(ffts_flus_args_helper_->InitRuntimeAddr(davinci_model_));
  GE_ASSERT_SUCCESS(ffts_flus_args_helper_->AssembleTilingData());

  // adapter zero copy for helper
  const std::vector<bool> need_raw_data_list = ModelUtils::GetInputTensorNeedRawData(op_desc_);
  davinci_model_->SetZeroCopyAddr(op_desc_, ffts_flus_args_helper_->GetIoAddr(),
                                  ffts_flus_args_helper_->GetIoAddr().data(),
                                  static_cast<uintptr_t>(PtrToValue(args_)),
                                  args_size_, 0UL, need_raw_data_list);
  GELOGI("FftsPlusTaskInfo Init Success, node: %s.", op_desc_->GetNamePtr());
  return SUCCESS;
}

Status FftsPlusTaskInfo::InitShapeInfosArgs(const domi::FftsPlusCtxDef &ctx_def) const {
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
  if ((ctx_type != RT_CTX_TYPE_MIX_AIC) && (ctx_type != RT_CTX_TYPE_MIX_AIV)) {
    return SUCCESS;
  }
  const auto &op_desc = davinci_model_->GetOpByIndex(ctx_def.op_index());
  GE_ASSERT_NOTNULL(op_desc);
  ArgsFormatHolder holder;
  if (!ffts_flus_args_helper_->CheckAndGetArgsFormats(ctx_def.op_index(), holder)) {
    return SUCCESS;
  }
  auto input_descs = op_desc->GetAllInputsDescPtr();
  for (const auto &arg_desc : holder.arg_descs) {
    if (arg_desc.addr_type == AddrType::INPUT_DESC) {
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      GE_ASSERT((arg_desc.ir_idx >= 0) && (static_cast<size_t>(arg_desc.ir_idx) < holder.ir_input_2_range.size()));
      const auto &ir_range = holder.ir_input_2_range[static_cast<size_t>(arg_desc.ir_idx)];
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        const size_t instance_idx = static_cast<size_t>(ir_range.first + idx);
        GE_ASSERT_TRUE(instance_idx < input_descs.size(), "Instance index [%zu] is out of range, max_size:[%zu].",
                       instance_idx, input_descs.size());
        AppendShapeDesc(*input_descs.at(instance_idx), shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      for (size_t idx = 0UL; idx < shape_info.size(); idx++) {
        GELOGI("Node [%s] idx[%zu] shape info[%" PRId64 "]", op_desc->GetNamePtr(), idx, shape_info[idx]);
      }
      holder.shape_infos.push_back(shape_info);
    }
    if (arg_desc.addr_type == AddrType::OUTPUT_DESC) {
      GE_ASSERT((arg_desc.ir_idx >= 0) && (static_cast<size_t>(arg_desc.ir_idx) < holder.ir_output_2_range.size()));
      const auto &ir_range = holder.ir_output_2_range[static_cast<size_t>(arg_desc.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(ir_range.first + idx));
        GE_ASSERT_NOTNULL(output_desc);
        AppendShapeDesc(*output_desc, shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      for (size_t idx = 0UL; idx < shape_info.size(); idx++) {
        GELOGI("Node [%s] idx[%zu] shape info[%" PRId64 "]", op_desc->GetNamePtr(), idx, shape_info[idx]);
      }
      holder.shape_infos.push_back(shape_info);
    }
  }

  ffts_flus_args_helper_->SaveArgsFormats(static_cast<int64_t>(ctx_def.op_index()), holder);
  return SUCCESS;
}

Status FftsPlusTaskInfo::ParseArgsFormat(const domi::FftsPlusCtxDef &ctx_def) const {
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
  if ((ctx_type != RT_CTX_TYPE_MIX_AIC) && (ctx_type != RT_CTX_TYPE_MIX_AIV)) {
    return SUCCESS;
  }
  const domi::FftsPlusMixAicAivCtxDef &mix_ctx = ctx_def.mix_aic_aiv_ctx();
  if (mix_ctx.args_format().empty()) {
    return SUCCESS;
  }
  const auto &op_desc = davinci_model_->GetOpByIndex(ctx_def.op_index());
  GE_ASSERT_NOTNULL(op_desc);
  GELOGI("Node [%s] has formatted args [%s]", op_desc->GetNamePtr(), mix_ctx.args_format().c_str());
  ArgsFormatHolder holder;
  GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(op_desc, mix_ctx.args_format(), holder.arg_descs),
                    "Formatted args [%s] parsed failed.", mix_ctx.args_format().c_str());
  GE_ASSERT_GRAPH_SUCCESS(OpDescUtils::GetIrInputInstanceDescRange(op_desc_, holder.ir_input_2_range));
  GE_ASSERT_GRAPH_SUCCESS(OpDescUtils::GetIrOutputDescRange(op_desc_, holder.ir_output_2_range));
  ffts_flus_args_helper_->SaveArgsFormats(static_cast<int64_t>(ctx_def.op_index()), holder);
  return SUCCESS;
}

Status FftsPlusTaskInfo::AssembleOtherArgsByArgsBase(const domi::FftsPlusTaskDef &ffts_plus_task_def) {
  const int32_t ctx_num = ffts_plus_task_def.ffts_plus_ctx_size();
  std::vector<int64_t> desc_val;
  for (int32_t i = 0; i < ctx_num; ++i) {
    const domi::FftsPlusCtxDef &ctx_def = ffts_plus_task_def.ffts_plus_ctx(i);
    GE_ASSERT_SUCCESS(InitShapeInfosArgs(ctx_def), "InitShapeInfosArgs failed.");
  }

  args_ = pis_args_dev_base_ + desc_buf_aligned_size_;
  args_host_ = pis_args_host_base_ + desc_buf_aligned_size_;
  GE_CHK_STATUS_RET(
      ffts_flus_args_helper_->InitArgsBase(pis_args_host_base_, args_, args_host_, args_size_, bin_args_size_));
  return SUCCESS;
}

Status FftsPlusTaskInfo::HandleSoftSyncOp(const uint32_t op_index, const OpDescPtr &op_desc) const {
  GE_CHECK_NOTNULL(op_desc);
  bool is_soft_sync_op = false;
  if ((!AttrUtils::GetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync_op)) || (!is_soft_sync_op)) {
    return SUCCESS;
  }

  const auto op = davinci_model_->GetOperatorByIndex(op_index);
  GE_CHECK_NOTNULL(op);
  const auto run_info = MakeShared<optiling::utils::OpRunInfo>(0, false, 0);
  GE_CHECK_NOTNULL(run_info);
  GE_CHK_STATUS_RET(
      optiling::SoftSyncOpRtParseAndTiling(
          *op, davinci_model_->MutablePlatformInfo(), *run_info,
          davinci_model_->GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))),
      "Recall tiling for soft sync op: %s failed.", op_desc->GetName().c_str());
  GE_ASSERT_TRUE(op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info));
  GELOGI("Success to set extra attr: %s to %s.", ATTR_NAME_OP_RUN_INFO.c_str(), op_desc->GetName().c_str());

  return SUCCESS;
}

void FftsPlusTaskInfo::SetTransferCallback(FftsPlusProtoTransfer &transfer) {
  transfer.SetDavinciModel(davinci_model_);
  const auto &rts_param = davinci_model_->GetRuntimeParam();
  const auto ffts_run_addr_handle = [&rts_param](const uintptr_t logic_addr, uint8_t *&mem_addr,
                                                 uint64_t &mem_type) -> Status {
    return ModelUtils::GetRtAddress(rts_param, logic_addr, mem_addr, mem_type);
  };

  const auto ffts_addr_pref_handle = [this](const OpDescPtr &op_desc, const std::string &kernel_name,
                                            const std::string &prefix,
                                            std::vector<std::pair<void *, uint32_t>> &addr_pref_cnt) -> Status {
    return davinci_model_->GetAddrAndPrefCnt(op_desc, kernel_name, prefix, addr_pref_cnt);
  };

  const auto ffts_find_node_handle = [this](const uint32_t index) -> OpDescPtr {
    return davinci_model_->GetOpByIndex(index);
  };

  const auto ffts_save_ctx_args_handle = [this](const OpDescPtr &descriptor, const uintptr_t op_args,
                                                const size_t args_offset,
                                                const std::vector<uintptr_t> &first_level_args) {
    InitDumpArgs(descriptor, op_args, args_offset, first_level_args);
  };

  const auto ffts_get_session_id = [this]() -> uint64_t {
    return davinci_model_->GetSessionId();
  };

  const auto ffts_create_aicpu_session = [this](STR_FWK_OP_KERNEL &fwk_op_kernel) -> Status {
    return this->CreateAicpuSession(fwk_op_kernel);
  };

  const auto ffts_get_event_id = [this](const OpDescPtr &op_desc, uint32_t &event_id) -> Status {
    return davinci_model_->GetEventIdForBlockingAicpuOp(op_desc, stream_, event_id);
  };

  const auto ffts_load_cust_aicpu_so = [this](const OpDescPtr &op_desc,
                                              const domi::FftsPlusAicpuCtxDef &ctx_def) -> Status {
    return this->LoadCustAicpuSo(op_desc, ctx_def);
  };

  const auto ffts_is_dump = [this](const OpDescPtr &op_desc) -> bool {
    return davinci_model_->OpNeedDump(op_desc);
  };

  const auto ffts_save_l0_dump_info_handle = [this](const std::vector<uint64_t> &l0_dump_list) {
    l0_dump_list_.insert(l0_dump_list_.end(), l0_dump_list.begin(), l0_dump_list.end());
  };

  transfer.SetRunAddrHandle(ffts_run_addr_handle);
  transfer.SetAddrPrefHandle(ffts_addr_pref_handle);
  transfer.SetFindNodeHandle(ffts_find_node_handle);
  transfer.SetSaveCtxArgsHandle(ffts_save_ctx_args_handle);
  transfer.SetGetSessionId(ffts_get_session_id);
  transfer.SetCreateAicpuSession(ffts_create_aicpu_session);
  transfer.SetLoadCustAicpuSo(ffts_load_cust_aicpu_so);
  transfer.SetGetEventIdHandle(ffts_get_event_id);
  transfer.SetIsDumpHandle(ffts_is_dump);
  transfer.SetSaveL0DumpInfoHandle(ffts_save_l0_dump_info_handle);
}

Status FftsPlusTaskInfo::CreateAicpuSession(STR_FWK_OP_KERNEL &fwk_op_kernel) const {
  // 0 Global Step
  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = davinci_model_->GetGlobalStep();

  // 1 Session Id
  const uint64_t session_id = davinci_model_->GetSessionId();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = session_id;

  // 2 Collect aicpu kernel
  fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = hybrid::AicpuExtInfoHandler::GenerateKernelId();
  ModelManager::GetInstance().CreateAicpuKernel(session_id, davinci_model_->Id(), davinci_model_->SubModelId(),
                                                fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID);

  // 3 Create session
  const auto ret = ModelManager::GetInstance().CreateAicpuSession(session_id);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "CreateAicpuSession fail, session_id:%" PRIu64 "", session_id);
    GELOGE(ret, "[Create][AicpuSession] error. session id:%" PRIu64, session_id);
    return ret;
  }
  return SUCCESS;
}

Status FftsPlusTaskInfo::LoadCustAicpuSo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def) const {
  bool loaded = false;
  const auto &kernel = ctx_def.kernel();
  auto &model_mgr = ModelManager::GetInstance();
  GE_CHK_STATUS_RET(model_mgr.LoadCustAicpuSo(davinci_model_->GetCustAICPUKernel(op_desc), kernel.so_name(), loaded),
                    "[Launch][CustAicpuSo] failed.");
  return SUCCESS;
}

void FftsPlusTaskInfo::CalculateAscendAicpuKernelSize(const domi::FftsPlusCtxDef &ctx_dex) {
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_dex.context_type());
  if (ctx_type != RT_CTX_TYPE_AICPU) {
    return;
  }
  const domi::FftsPlusAicpuCtxDef &aicpu_ctx = ctx_dex.aicpu_ctx();
  if (aicpu_ctx.kernel_type() != kAicpuTfKernel) {
    bin_args_size_ += static_cast<size_t>(aicpu_ctx.kernel().args().size());
  }
}

Status FftsPlusTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                           TaskRunParam &task_run_param) {
  davinci_model_ = davinci_model;
  GE_ASSERT_SUCCESS(PrePareForTransfer(task_def));
  task_run_param.args_descs.push_back({static_cast<int64_t>(pis_args_size_), args_placement_});

  const auto logical_mem_allocations = davinci_model_->GetLogicalMemAllocation();
  const size_t max_index =
      (logical_mem_allocations.size() > 0U) ? (logical_mem_allocations.size() - 1UL) : logical_mem_allocations.size();
  for (size_t i = 0UL; i < max_index; i++) {
    task_run_param.parsed_input_addrs.push_back({logical_mem_allocations[i].logical_addr, kFmMemType, true, {0U}});
  }
  return SUCCESS;
}

Status FftsPlusTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  return ffts_flus_args_helper_->GetTaskArgsRefreshInfos(infos, *ffts_proto_transfer_);
}

Status FftsPlusTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                        const size_t host_args_max_len) {
  (void)host_args;
  (void)host_args_max_len;
  GELOGD("FftsPlusTaskInfo::UpdateHostArgs in.");
  GE_CHECK_NOTNULL(ffts_flus_args_helper_);
  GE_CHECK_NOTNULL(ffts_proto_transfer_);
  GE_CHK_STATUS_RET(ffts_flus_args_helper_->UpdateAddrsWithIOZcpy(active_mem_base_addr, *ffts_proto_transfer_));
  DumpFftsPlusTask(desc_buf_host_, desc_buffer_len_, stream_id_, task_id_);
  GELOGD("FftsPlusTaskInfo::UpdateHostArgs success.");
  return SUCCESS;
}

Status FftsPlusTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("FftsPlusTaskInfo %s Distribute Start.", op_desc_->GetNamePtr());
  const TaskProfGuarder prof_guarder(this);

  // 只有mix l2算子支持l0 exception dump, ffts+场景不走该流程
  if (op_desc_->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    GE_ASSERT_SUCCESS(SetL0ExceptionSizeInfo(op_desc_, l0_dump_list_));
  }

  GE_CHK_RT_RET(ge::rtFftsPlusTaskLaunchWithFlag(&ffts_plus_task_info_, stream_, dump_flag_));
  GE_CHECK_NOTNULL(davinci_model_);
  GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
  GE_CHK_RT_RET(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)));

  std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
  std::shared_ptr<TilingContextAddr> tiling_context_addr =
      op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
  if (tiling_context_addr != nullptr) {
    auto sink_info = MakeShared<TilingSinkTaskInfo>();
    GE_ASSERT_NOTNULL(sink_info);
    sink_info->task_id = task_id_;
    sink_info->stream = stream_;
    sink_info->ffts_task_handle = &ffts_plus_task_info_;
    GE_ASSERT_TRUE(op_desc_->SetExtAttr(kTilingSinkTaskInfo, sink_info));
  }

  GELOGI("FftsPlusTaskInfo Distribute Success, node_name:[%s], subgraph_name:[%s] stream_id:%u, task_id:%u.",
         op_desc_->GetNamePtr(), op_desc_->GetSubgraphInstanceName(0).c_str(), stream_id_, task_id_);
  DumpFftsPlusTask(desc_buf_host_, desc_buffer_len_, stream_id_, task_id_);

  return SUCCESS;
}

Status FftsPlusTaskInfo::Release() {
  ext_info_addrs_.clear();
  CleanRtFftsPlusTask(ffts_plus_task_info_);
  return SUCCESS;
}

Status FftsPlusTaskInfo::SetCachePersistentWay(const OpDescPtr &op_desc) const {
  const vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(davinci_model_->GetRuntimeParam(), op_desc);
  const size_t input_addr_size = input_addrs.size();
  size_t j = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      continue;
    }
    ++j;

    uint32_t persistent_id = std::numeric_limits<uint32_t>::max();
    (void)AttrUtils::GetInt(tensor_desc, ATTR_NAME_CACHE_PERSIST, persistent_id);
    if (persistent_id == std::numeric_limits<uint32_t>::max()) {
      continue;
    }

    GE_ASSERT_TRUE(j <= input_addr_size, "Invalid addr index %zu, max %zu", j, input_addr_size);

    int64_t tensor_size = 0;
    (void)TensorUtils::GetSize(*tensor_desc, tensor_size);
    constexpr uint32_t advise = 0U;
    const rtError_t ret = rtMemAdvise(input_addrs[j - 1UL], static_cast<uint64_t>(tensor_size), advise);
    GE_ASSERT_TRUE(ret == RT_ERROR_NONE,
                   "Failed to advise memory, persis id %u, error code %d, ptr addr %p, size %" PRIu64 ".",
                   persistent_id, ret, input_addrs[j - 1UL], static_cast<uint64_t>(tensor_size));
  }
  return SUCCESS;
}

Status FftsPlusTaskInfo::CalculateTilingDataSize(const OpDescPtr &op_desc, const bool is_atomic_op_type) {
  GE_CHECK_NOTNULL(op_desc);
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  if (is_atomic_op_type) {
    tiling_info = op_desc->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, tiling_info);
  } else {
    tiling_info = op_desc->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, tiling_info);
  }
  if (tiling_info != nullptr) {
    const size_t memcheck_size = ffts_flus_args_helper_->GetMemCheckInfo(op_desc->GetName()).size();
    // memcheck场景下需要附加一段size
    const size_t tiling_data_size = tiling_info->GetAllTilingData().str().size() + memcheck_size;
    tiling_data_len_ += MemSizeAlign(tiling_data_size);
  }
  GELOGD("op %s tiling data is %zu, is_atomic_op_type:%u", op_desc->GetName().c_str(), tiling_data_len_,
         is_atomic_op_type);
  return SUCCESS;
}

void FftsPlusTaskInfo::InitDumpArgs(const OpDescPtr &op_desc, const uintptr_t op_args, const size_t args_offset,
                                    const std::vector<uintptr_t> &first_level_args) {
  if (davinci_model_->OpNeedDump(op_desc) || davinci_model_->OpNeedPrint(op_desc)) {
    GELOGD("Op:%s(%s) need dump in ffts plus task info", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    dump_flag_ = RT_KERNEL_FFTSPLUS_STATIC_SHAPE_DUMPFLAG;
  }

  const std::string &op_name = op_desc->GetName();
  if (first_level_args.empty()) {
    if (op_args != 0U) {
      dump_op_args_[op_name] = op_args;
    } else {
      dump_args_offset_[op_name] = args_offset;
    }
  } else {
    dump_op_2_first_level_args_[op_name] = first_level_args;
  }
}

uintptr_t FftsPlusTaskInfo::FindDumpArgs(const std::string &op_name) const {
  const std::map<std::string, size_t>::const_iterator iter = dump_args_offset_.find(op_name);
  if (iter != dump_args_offset_.end()) {
    return static_cast<uintptr_t>(PtrToValue(args_) + iter->second);
  }

  const std::map<std::string, uintptr_t>::const_iterator iter1 = dump_op_args_.find(op_name);
  if (iter1 != dump_op_args_.end()) {
    return iter1->second;
  }
  GELOGD("op:%s not save args", op_name.c_str());
  return 0U;
}

bool FftsPlusTaskInfo::OpNeedDump(const OpDescPtr &op_desc) const {
  return (davinci_model_->GetOpDugReg() || (davinci_model_->OpNeedDump(op_desc) && CallSaveDumpInfo()));
}

void FftsPlusTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  if (davinci_model_->OpNeedSetDumpFlagOnWatcherModel(op_desc_->GetName())) {
    GELOGW("fftsplus task no support dump watcher model.");
  }

  const domi::FftsPlusTaskDef &ffts_plus_task_def = task_def.ffts_plus_task();
  const int32_t ctx_num = ffts_plus_task_def.ffts_plus_ctx_size();
  for (int32_t i = 0; i < ctx_num; ++i) {
    const domi::FftsPlusCtxDef &ctx_def = ffts_plus_task_def.ffts_plus_ctx(i);
    const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
    if (kUnsupportedDumpCtxTypes.count(ctx_type) > 0UL) {
      GELOGD("ctx op type %u is not supported, ctx id:%u, op index:%u", ctx_type,
             ctx_def.context_id(), ctx_def.op_index());
      continue;
    }
    const OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_def.op_index());
    if (op_desc == nullptr) {
      GELOGD("ctx op is nullptr, ctx id:%u, ctx type:%u, op index:%u", ctx_def.context_id(), ctx_type,
             ctx_def.op_index());
      continue;
    }
    const auto context_id = ctx_def.context_id();
    (void)AttrUtils::SetInt(op_desc, "current_context_id", static_cast<int64_t>(context_id));
    const auto &op_name = op_desc->GetName();
    uintptr_t dump_args = 0U;
    size_t args_size = 0UL;
    if (ctx_def.op_type() != domi::FftsPlusCtxDef::ATOMIC) {
      dump_args = FindDumpArgs(op_name);
      args_size = ffts_flus_args_helper_->GetCtxArgsSize(static_cast<int32_t>(ctx_def.context_id()));
    }

    davinci_model_->SaveFftsExceptionDumpInfo(ctx_def, op_desc, *this, std::make_pair(dump_args, args_size),
                                              ffts_flus_args_helper_->GetCustToRelevantOffset());
    davinci_model_->SaveFftsDfxInfo(ctx_def, op_desc, *this);
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    SavePrintOrDumpTask(op_desc, ctx_def, dump_args, task_type);
    GELOGD("ffts dump, ctx id:%u, ctx type:%u, op index:%u, op name:%s, task id:%u, stream id:%u, args_size:[%zu]",
           context_id, ctx_type, ctx_def.op_index(), op_name.c_str(), task_id_, stream_id_, args_size);
  }

  davinci_model_->SaveFftsPlusProfilingTask(task_def, *this);
}

void FftsPlusTaskInfo::SavePrintOrDumpTask(const OpDescPtr &op_desc, const domi::FftsPlusCtxDef &ctx_def,
                                           const uintptr_t &dump_args, const ModelTaskType task_type) {
  bool need_dump = OpNeedDump(op_desc);
  bool need_print = davinci_model_->OpNeedPrint(op_desc);
  if (!need_dump && !need_print) {
    GELOGD("Op %s is not in the dump list.", op_desc->GetName().c_str());
    return;
  }
  size_t thread_id = 0U;
  size_t thread_dim = 0U;
  size_t thread_offset = 0U;
  size_t real_ctx_num;
  GetCtxSliceThreadInfo(ctx_def, thread_id, thread_dim, thread_offset);
  std::vector<uint32_t> ctx_ids;
  if (ge::AttrUtils::GetListInt(op_desc, gert::kContextIdList, ctx_ids) && !ctx_ids.empty()) {
    real_ctx_num = std::min(thread_dim, ctx_ids.size());
  } else {
    real_ctx_num = 1U;
    thread_offset = 0U;
  }
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
  const auto context_id = ctx_def.context_id();
  for (size_t num = 0U; num < real_ctx_num; ++num) {
    const size_t slice_idx = thread_id + num * real_ctx_num;
    const auto &iter = dump_op_2_first_level_args_.find(op_desc->GetName());
    FirstLevelAddressInfo first_level_address_info{};
    uintptr_t arg_addr;
    if (iter != dump_op_2_first_level_args_.end()) {
      arg_addr = static_cast<uintptr_t>(PtrToValue(iter->second.data()));
      first_level_address_info.address_type = true;
      first_level_address_info.address.assign(iter->second.begin(), iter->second.end());
    } else if (dump_args != 0U) {
      arg_addr = dump_args + static_cast<uintptr_t>(thread_offset * slice_idx);
    } else {
      GELOGW("Op %s will not be dumped because of invalid arg address.", op_desc->GetName().c_str());
      continue;
    }

    if (need_dump) {
      davinci_model_->SaveDumpTask({task_id_, stream_id_, context_id, static_cast<uint32_t>(slice_idx)}, op_desc,
          arg_addr, first_level_address_info, ffts_flus_args_helper_->GetCustToRelevantOffset(), task_type, stream_);
    }

    if (need_print) {
      davinci_model_->SavePrintDumpTask({task_id_, stream_id_, context_id, static_cast<uint32_t>(slice_idx)}, op_desc,
          arg_addr, first_level_address_info, task_type, stream_);
      davinci_model_->SavePrintWorkInfo(op_desc);
    }

    GELOGD("Save ctx op %s, ctx id:%u, ctx type:%u, op index:%u, task id:%u, stream id:%u, thread id:%u",
           op_desc->GetName().c_str(), context_id, ctx_type, ctx_def.op_index(), task_id_, stream_id_, slice_idx);
  }
}

int64_t FftsPlusTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::FftsPlusTaskDef &ffts_plus_task_def = task_def.ffts_plus_task();
  return static_cast<int64_t>(ffts_plus_task_def.op_index());
}

Status FftsPlusTaskInfo::InitTilingInfo() {
  ffts_flus_args_helper_->SetTilingDataLen(tiling_data_len_);
  void *tiling_data_dev = davinci_model_->MallocDynamicMemory(
      static_cast<size_t>(tiling_data_len_ + kDescBufAlignedBytes));
  GE_ASSERT_NOTNULL(tiling_data_dev);
  ext_info_addrs_.emplace_back(tiling_data_dev);
  ffts_flus_args_helper_->SetTilingDataDev(tiling_data_dev);
  return SUCCESS;
}

Status FftsPlusTaskInfo::TilingDataHandle(const domi::FftsPlusCtxDef &ctx_def,
                                          const OpDescPtr &op_desc) const {
  GE_CHECK_NOTNULL(op_desc);
  const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
  GE_ASSERT_SUCCESS(HandleSoftSyncOp(ctx_def.op_index(), op_desc),
                    "Handle soft sync op %s failed.", op_desc->GetNamePtr());
  if ((ctx_type == RT_CTX_TYPE_MIX_AIC) || (ctx_type == RT_CTX_TYPE_MIX_AIV)) {
    std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
    tiling_info = op_desc->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, tiling_info);
    // 没有tiling信息时啥都不处理
    if (tiling_info != nullptr) {
      ArgsFormatHolder args_format_holder;
      (void)ffts_flus_args_helper_->CheckAndGetArgsFormats(static_cast<int64_t>(ctx_def.op_index()),
          args_format_holder);
      std::string memcheck_info;
      GE_ASSERT_SUCCESS(optiling::TilingMemCheck::ConstructMemCheckInfo(op_desc, *tiling_info,
          args_format_holder.arg_descs, memcheck_info),
          "Append Memcheck info to tiling data: %s failed.", op_desc->GetNamePtr());
      ffts_flus_args_helper_->SaveMemCheckInfo(op_desc->GetName(), memcheck_info);
    }
  }
  return SUCCESS;
}

Status FftsPlusTaskInfo::PrePareForTransfer(const domi::TaskDef &task_def) {
  GE_ASSERT_NOTNULL(davinci_model_);
  GE_ASSERT_SUCCESS(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  const domi::FftsPlusTaskDef &ffts_plus_task_def = task_def.ffts_plus_task();
  op_desc_ = davinci_model_->GetOpByIndex(ffts_plus_task_def.op_index());
  GE_ASSERT_NOTNULL(op_desc_);
  ffts_flus_args_helper_ = MakeUnique<FftsPlusArgsHelper>(davinci_model_->GetRuntimeParam());
  GE_CHECK_NOTNULL(ffts_flus_args_helper_);
  ffts_flus_args_helper_->SetOpDesc(op_desc_);
  const int32_t ctx_num = ffts_plus_task_def.ffts_plus_ctx_size();
  size_t dsa_ctx_num = 0U;
  for (int32_t i = 0; i < ctx_num; ++i) {
    const domi::FftsPlusCtxDef &ctx_def = ffts_plus_task_def.ffts_plus_ctx(i);
    CalculateAscendAicpuKernelSize(ctx_def);
    GE_ASSERT_SUCCESS(ParseArgsFormat(ctx_def));
    bool is_atomic_op_type = false;
    if (ctx_def.op_type() == domi::FftsPlusCtxDef::ATOMIC) {
      is_atomic_op_type = true;
    }
    const OpDescPtr sub_op_desc = davinci_model_->GetOpByIndex(ctx_def.op_index());
    if (!is_atomic_op_type) {
      GE_ASSERT_SUCCESS(TilingDataHandle(ctx_def, sub_op_desc));
    }
    const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
    if (kCalTilingSizeCtxType.count(ctx_type) > 0U) {
      GE_CHECK_NOTNULL(sub_op_desc);
      GE_ASSERT_SUCCESS(CalculateTilingDataSize(sub_op_desc, is_atomic_op_type),
                        "[Calc][CalculateTilingSize] failed, node:%s", sub_op_desc->GetName().c_str());
    }
    if (ctx_type == RT_CTX_TYPE_DSA) {
        dsa_ctx_num++;
    }
    if ((sub_op_desc != nullptr) && (!davinci_model_->IsFeatureBaseRefreshable())) {
      GE_ASSERT_SUCCESS(SetCachePersistentWay(sub_op_desc), "[Call][SetCachePersistentWay] failed, node:%s.",
                        sub_op_desc->GetName().c_str());
    }
  }
  GE_ASSERT_SUCCESS(InitTilingInfo());
  GE_ASSERT_TRUE(!ge::MulOverflow(dsa_ctx_num, kDsaWorkspaceMaxSize, dsa_workspace_size_));
  GELOGI("Prepare for task transfer-ing success, node: %s, ctx num: %d, dsa workspace size: %zu.",
    op_desc_->GetNamePtr(), ctx_num, dsa_workspace_size_);

  GE_ASSERT_TRUE(!ge::MulOverflow(sizeof(void *), ffts_plus_task_def.addr_size(), args_size_));
  GE_ASSERT_TRUE(!ge::AddOverflow(args_size_, dsa_workspace_size_, args_size_));
  desc_buffer_len_ = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(ctx_num);
  desc_buf_aligned_size_ = desc_buffer_len_ + kDescBufAlignedBytes;
  GE_ASSERT_TRUE(!ge::AddOverflow(desc_buf_aligned_size_, args_size_, pis_args_size_));
  GELOGI("After parser, args is %zu, pis args size is %zu", args_size_, pis_args_size_);
  return SUCCESS;
}

void FftsPlusTaskInfo::InitDescBufInfo() {
  uint8_t *const aligned_base = PtrToPtr<void, uint8_t>(ValueToPtr(static_cast<uint64_t>(
      MemSizeAlign(static_cast<size_t>(PtrToValue(PtrToPtr<uint8_t, void>(pis_args_dev_base_))),
                   kDescBufAlignedBytes))));
  ffts_plus_task_info_.descBuf = aligned_base;
  desc_buf_host_ = aligned_base - pis_args_dev_base_ + pis_args_host_base_;
  GELOGI("Descbuf addr:[%p], 128-aligned addr:[%p], aligned_host:[%p].", pis_args_dev_base_,
         ffts_plus_task_info_.descBuf, desc_buf_host_);
  ffts_plus_task_info_.descBufLen = desc_buffer_len_;
  ffts_plus_task_info_.descAddrType = RT_FFTS_PLUS_CTX_DESC_ADDR_TYPE_DEVICE;
}

Status FftsPlusTaskInfo::InitArgsBaseInfo(const PisToArgs &args) {
  pis_args_dev_base_ = PtrToPtr<void, uint8_t>(ValueToPtr(args[static_cast<size_t>(args_placement_)].dev_addr));
  pis_args_host_base_ = PtrToPtr<void, uint8_t>(args[static_cast<size_t>(args_placement_)].host_addr);
  GE_CHECK_NOTNULL(pis_args_dev_base_);
  GE_CHECK_NOTNULL(pis_args_host_base_);
  GELOGI("Args base init successfully, with details {pis_args_dev_base: %p pis_args_host_base: %p}", pis_args_dev_base_,
         pis_args_host_base_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_FFTS_PLUS, FftsPlusTaskInfo);
}  // namespace ge
