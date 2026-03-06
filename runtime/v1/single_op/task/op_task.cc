/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/task/op_task.h"
#include <chrono>
#include <thread>

#include "aicpu_task_struct.h"
#include "common/dump/dump_manager.h"
#include "common/profiling/profiling_manager.h"
#include "formats/formats.h"
#include "common/math/math_util.h"
#include "common/runtime_api_wrapper.h"
#include "common/utils/executor_utils.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"
#include "single_op/task/build_task_utils.h"
#include "framework/common/profiling_definitions.h"
#include "runtime/subscriber/global_profiler.h"
#include "common/checker.h"
#include "common/dump/kernel_tracing_utils.h"
#include "adump_pub.h"
#include "runtime/kernel.h"
#include "common/dump/dump_utils.h"
#include "common/error_tracking/error_tracking.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"

namespace ge {
namespace {
constexpr size_t kMemcpyArgCount = 2U;
constexpr size_t kCopyNum = 2U;
constexpr int8_t kInputIsConst = 1;
constexpr uint32_t kMask32Bits = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111
constexpr uint32_t k2BitsMask = 0x00000003U;   // 2  bits, 0000,0011

const std::string kLocalMemorySize = "local_memory_size";
const std::string kAttrTaskRatio = "_task_ratio";
const std::string kAttrIsAiv = "_mix_is_aiv";
const std::string kAttrIsFFTSTask = "_is_fftsplus_task";  // fftsplus task

void FreeHbm(void *const var) {
  if (var != nullptr) {
    (void)rtFree(var);
  }
}
}  // namespace

Status OpTask::SaveExceptionDumpInfo() {
  if (gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) {
    uintptr_t *arg_base = nullptr;
    size_t arg_num = 0UL;
    GetIoAddr(arg_base, arg_num);
    ExtraOpInfo extra_op_info{};
    const auto input_size = op_desc_->GetInputsSize();
    for (size_t i = 0UL; i < input_size; i++) {
      extra_op_info.input_addrs.emplace_back(reinterpret_cast<void *>(*arg_base));
      ++arg_base;
    }
    const auto output_size = op_desc_->GetOutputsSize();
    for (size_t j = 0UL; j < output_size; j++) {
      extra_op_info.output_addrs.emplace_back(reinterpret_cast<void *>(*arg_base));
      ++arg_base;
    }
    GetTilingKeyAndData(extra_op_info.tiling_key, extra_op_info.tiling_data);
    const std::vector<int64_t> v_workspace_bytes = op_desc_->GetWorkspaceBytes();
    if (arg_num < (input_size + output_size + v_workspace_bytes.size())) {
      GELOGW("Args num is invalid, no workspace saved.");
    } else {
      for (const auto &v_workspace_byte : v_workspace_bytes) {
        extra_op_info.workspace_info.emplace_back(reinterpret_cast<uintptr_t>(*arg_base), v_workspace_byte);
        ++arg_base;
      }
    }
    GetHostArgsAndSize(extra_op_info.args, extra_op_info.args_size);
    extra_op_info.is_host_args = true;
    std::stringstream ss;
    ss << "args before execute: ";
    gert::PrintHex(reinterpret_cast<void **>(extra_op_info.args), extra_op_info.args_size / sizeof(void *), ss);
    extra_op_info.args_before_execute = ss.str();
    int32_t dev_id = 0;
    GE_CHK_RT_RET(rtGetDevice(&dev_id));
    ge::OpDescInfoId id(task_id_, stream_id_, dev_id);
    gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->SaveDumpOpInfo(op_desc_, extra_op_info, id, true);
  }
  return SUCCESS;
}

void OpTask::GetHostArgsAndSize(uintptr_t &args, size_t &arg_size) {
  args = reinterpret_cast<uintptr_t>(nullptr);
  arg_size = 0UL;
}

Status OpTask::OpenDump(rtStream_t const stream) {
  if (DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsSingleOpNeedDump()) {
    GELOGI("Dump is open in single op, start to set dump info");
    std::vector<uintptr_t> input_addrs;
    std::vector<uintptr_t> output_addrs;
    const auto input_size = op_desc_->GetInputsSize();
    const auto output_size = op_desc_->GetOutputsSize();
    uintptr_t *arg_base = nullptr;
    size_t arg_num = 0U;
    GetIoAddr(arg_base, arg_num);
    if (arg_num < (input_size + output_size)) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
          "[Check][Size]io_addrs_for_dump_ size %zu is not equal input and output size %zu",
          arg_num, input_size + output_size);
      REPORT_INNER_ERR_MSG("E19999", "io_addrs_for_dump_ size %zu is not equal input and output size %zu",
          arg_num, input_size + output_size);
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }

    for (size_t i = 0U; i < input_size; i++) {
      input_addrs.emplace_back(*arg_base);
      ++arg_base;
    }
    for (size_t j = 0U; j < output_size; j++) {
      output_addrs.emplace_back(*arg_base);
      ++arg_base;
    }
    dump_op_.SetDumpInfo(DumpManager::GetInstance().GetDumpProperties(kInferSessionId),
                         op_desc_, input_addrs, output_addrs, stream);
    const auto status = dump_op_.LaunchDumpOp(true);
    if (status != SUCCESS) {
      GELOGE(status, "[Launch][DumpOp] failed in single op.");
      return status;
    }
    return SUCCESS;
  }
  GELOGI("Dump is not open in single op");
  return SUCCESS;
}

Status OpTask::GetTaskIdAndStreamId(rtStream_t const stream) {
  if (ProfilingManager::Instance().ProfilingModelLoadOn()) {
    GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
    GE_CHK_RT_RET(rtsStreamGetId(stream, reinterpret_cast<int32_t*>(&stream_id_)));
  }
  return SUCCESS;
}

void OpTask::SetTaskTag() const {
  if (op_desc_ != nullptr) {
    const rtError_t rt_set_tag = rtSetTaskTag(op_desc_->GetName().c_str());
    if (rt_set_tag != RT_ERROR_NONE) {
      GELOGW("[Call][rtSetTaskTag] failed, ret:0x%X", rt_set_tag);
    }
  }
}

Status OpTask::PostProcess(rtStream_t const stream) {
  GE_CHK_STATUS_RET(OpenDump(stream), "[Open][Dump]failed, single op:%s.",
                    GetOpdesc()->GetName().c_str());
  GE_ASSERT_RT_OK(rtsGetThreadLastTaskId(&task_id_));
  GE_ASSERT_RT_OK(rtsStreamGetId(stream, reinterpret_cast<int32_t*>(&stream_id_)));
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(op_desc_, task_id_, stream_id_);
  GE_CHK_STATUS(SaveExceptionDumpInfo(), "Save Exception dump failed.");
  ResetDumperResource();
  return SUCCESS;
}

void TbeOpTask::SetStubFunc(const std::string &name, const void *const stub_func) {
  this->stub_name_ = name;
  this->stub_func_ = stub_func;
  this->task_name_ = name;
}

void TbeOpTask::SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, const size_t arg_size,
                              const uint32_t block_dim, const OpDescPtr &op_desc) {
  args_ = std::move(args);
  arg_size_ = arg_size;
  block_dim_ = block_dim;
  op_desc_ = op_desc;
  (void)AttrUtils::GetInt(op_desc, kLocalMemorySize, cfg_.localMemorySize);
}

void TbeOpTask::SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, const size_t arg_size,
                              const uint32_t block_dim, const OpDescPtr &op_desc,
                              const domi::KernelDef &kernel_def) {
  SetKernelArgs(std::move(args), arg_size, block_dim, op_desc);
  cfg_.schemMode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
  (void)AttrUtils::GetInt(op_desc, kLocalMemorySize, cfg_.localMemorySize);
  GELOGD("OpName: %s set schedule mode from kernel def: %u, block dim: %u, local memory size: %u",
      op_desc->GetName().c_str(), static_cast<uint32_t>(cfg_.schemMode), block_dim, cfg_.localMemorySize);
}

void TbeOpTask::SetKernelWithHandleArgs(std::unique_ptr<uint8_t[]> &&args, const size_t arg_size,
                                        const uint32_t block_dim, const OpDescPtr &op_desc,
                                        const domi::KernelDefWithHandle &kernel_def_with_handle) {
  SetKernelArgs(std::move(args), arg_size, block_dim, op_desc);
  node_info_ = kernel_def_with_handle.node_info();
  cfg_.schemMode = static_cast<uint8_t>(kernel_def_with_handle.schedule_mode() & k2BitsMask);
  (void)AttrUtils::GetInt(op_desc, kLocalMemorySize, cfg_.localMemorySize);
  GELOGD("OpName: %s set schedule mode from kernel def: %u, block dim:%u, local memory size: %u",
      op_desc->GetName().c_str(), static_cast<uint32_t>(cfg_.schemMode), block_dim, cfg_.localMemorySize);
}

void OpTask::SetModelArgs(const std::string &model_name, const uint32_t model_id) {
  model_name_ = model_name;
  model_id_ = model_id;
}

const std::string OpTask::GetOpType() const {
  if (op_desc_ == nullptr) {
    return "";
  } else {
    return op_desc_->GetType();
  }
}

static const std::map<std::string, MsprofGeTaskType> kCoreTypeToTaskTypes {
  {"AI_CORE", MSPROF_GE_TASK_TYPE_AI_CORE},
  {"AI_CPU", MSPROF_GE_TASK_TYPE_AI_CPU},
  {"MIX_AIC", MSPROF_GE_TASK_TYPE_MIX_AIC},
  {"MIX_AIV", MSPROF_GE_TASK_TYPE_MIX_AIV},
  {"AIV", MSPROF_GE_TASK_TYPE_AIV},
  {"DSA", MSPROF_GE_TASK_TYPE_DSA},
  {"WRITE_BACK", MSPROF_GE_TASK_TYPE_WRITE_BACK},
  {"INVALID", MSPROF_GE_TASK_TYPE_INVALID}
};

Status OpTask::ReportProfAdditionalInfo(const uint64_t end_time, const uint64_t op_name_hash, const int32_t tid) const {
  if (!gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kDevice)) {
    return ge::SUCCESS;
  }
  GE_CHECK_NOTNULL(op_desc_);

  MsprofCompactInfo node_basic_info{};
  const uint64_t op_type_hash = MsprofGetHashId(GetOpType().c_str(), GetOpType().length());
  uint32_t task_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE);
  const auto task_type_str = GetTaskType();
  if (task_type_str != kTaskTypeInvalid) {
    const auto iter = kCoreTypeToTaskTypes.find(task_type_str);
    if (iter != kCoreTypeToTaskTypes.end()) {
      task_type = static_cast<uint32_t>(iter->second);
    }
  }

  uint32_t block_dim = block_dim_;
  uint32_t task_ratio = 0;
  bool is_fftsplus_task = false;
  if ((AttrUtils::GetBool(op_desc_, kAttrIsFFTSTask, is_fftsplus_task) && is_fftsplus_task &&
       AttrUtils::GetInt(op_desc_, kAttrTaskRatio, task_ratio))) {
    GELOGI("Op %s is fftsplus task, task type: %d, block dim: %u, task ratio: %u", op_desc_->GetName().c_str(),
           static_cast<uint32_t>(task_type), block_dim, task_ratio);
    // 针对mix算子，低16位为主加速器blockdim，高16位为从加速器的ratio值，由工具解析
    block_dim = ((block_dim & 0xFFFFU) | (task_ratio << 16U));
  }

  gert::GlobalProfilingWrapper::BuildNodeBasicInfo(op_desc_, block_dim, {op_name_hash, op_type_hash},
                                                   task_type, node_basic_info);
  gert::GlobalProfilingWrapper::BuildCompactInfo(end_time, node_basic_info);
  GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(static_cast<uint32_t>(true), &node_basic_info,
                                              static_cast<uint32_t>(sizeof(MsprofCompactInfo))));
  TaskDescInfo task_desc_info{};
  task_desc_info.op_name = op_desc_->GetName();
  ProfilingManager::Instance().GetOpInputOutputInfo(op_desc_, task_desc_info);
  task_desc_info.prof_time = end_time;
  GELOGD("[Cann Profiling] node name is %s, node type is %s, blcokDim is %u, op name hash is %llu, "
         "op type hash is %llu, task type is %u, endtime is %llu, tid is %d",
         op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), block_dim, op_name_hash, op_type_hash,
         task_type, end_time, tid);
  GE_ASSERT_TRUE(tid > 0);
  GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::ReportTensorInfo(static_cast<uint32_t>(tid), true, task_desc_info));
  GE_ASSERT_SUCCESS(ReportProfExtendInfo(end_time, op_name_hash, tid));
  return SUCCESS;
}

Status OpTask::ReportProfilingData(const uint64_t begin_time) const {
  thread_local const int32_t tid = mmGetTid();
  const uint64_t end_time = MsprofSysCycleTime();
  const uint64_t op_name_hash = MsprofGetHashId(op_desc_->GetName().c_str(), op_desc_->GetName().length());
  (void)gert::GlobalProfilingWrapper::ReportApiInfo(begin_time, end_time, op_name_hash,
                                                    MSPROF_REPORT_NODE_LAUNCH_TYPE);
  GE_ASSERT_SUCCESS(ReportProfAdditionalInfo(end_time, op_name_hash, tid));
  return ge::SUCCESS;
}

Status OpTask::UpdateRunInfo() {
  return UNSUPPORTED;
}

Status OpTask::DoUpdateArgTable(const SingleOpModelParam &param, const bool keep_workspace) {
  const auto addresses = BuildTaskUtils::GetAddresses(op_desc_, param, keep_workspace);
  const auto all_addresses = BuildTaskUtils::JoinAddresses(addresses);
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0U;
  GetIoAddr(arg_base, arg_num);
  if (arg_num < all_addresses.size()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
        "[Check][Size][%s] arg number mismatches, expect at least = %zu, but got = %zu.",
        op_desc_->GetName().c_str(), all_addresses.size(), arg_num);
    REPORT_INNER_ERR_MSG("E19999", "%s arg number mismatches, expect at least = %zu, but got = %zu.",
                         op_desc_->GetName().c_str(), all_addresses.size(), arg_num);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  for (void *const addr : all_addresses) {
    *arg_base = static_cast<uintptr_t>(PtrToValue(addr));
    arg_base++;
  }
  return SUCCESS;
}

Status OpTask::UpdateArgTable(const SingleOpModelParam &param) {
  return DoUpdateArgTable(param, true);
}

Status OpTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                            const std::vector<DataBuffer> &input_buffers,
                            std::vector<GeTensorDesc> &output_desc,
                            std::vector<DataBuffer> &output_buffers,
                            rtStream_t const stream) {
  (void)input_desc;
  (void)input_buffers;
  (void)output_desc;
  (void)output_buffers;
  (void)stream;
  return UNSUPPORTED;
}

const std::string &OpTask::GetTaskType() const { return kTaskTypeInvalid; }

void OpTask::SetNeedHostMemOpt(const bool need_host_mem_opt) {
  need_host_mem_opt_ = need_host_mem_opt;
}

void OpTask::SetHostMemInputFlag(const bool has_host_mem_input) {
  extend_args_for_host_input_ = has_host_mem_input;
}

bool OpTask::GetNeedTiling() const {
  return need_tiling_;
}

void OpTask::SetRuntimeContext(RuntimeInferenceContext *const context) {
  if (op_ != nullptr) {
    OpDescUtils::SetRuntimeContextToOperator(*op_, context);
  }
}

Status OpTask::UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  (void)inputs;
  (void)outputs;
  return SUCCESS;
}

TbeOpTask::~TbeOpTask() noexcept {
  if (sm_desc_ != nullptr) {
    (void)rtMemFreeManaged(sm_desc_);
  }
}

const std::string &TbeOpTask::GetTaskType() const {
  bool is_fftsplus_task = false;
  if ((AttrUtils::GetBool(op_desc_, kAttrIsFFTSTask, is_fftsplus_task) && is_fftsplus_task)) {
    bool is_mix_aiv = false;
    (void)AttrUtils::GetBool(op_desc_, kAttrIsAiv, is_mix_aiv);
    return is_mix_aiv ? kTaskTypeMixAiv : kTaskTypeMixAic;
  }

  std::string cube_vector_core_type = kTaskTypeAicore;
  (void)AttrUtils::GetStr(op_desc_, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, cube_vector_core_type);
  if (cube_vector_core_type == kTaskTypeAiv) {
    return kTaskTypeAiv;
  }
  return kTaskTypeAicore;
}

void TbeOpTask::SetHandle(void *const handle) {
  this->handle_ = handle;
}

void TbeOpTask::ResetDumperResource() {
  // tiling data need to be cleared after use
  if (run_info_ != nullptr) {
    run_info_->GetAllTilingData().str("");
  }
}

Status TbeOpTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("To invoke rtKernelLaunch. task = %s, block_dim = %u", this->stub_name_.c_str(), block_dim_);

  bool is_soft_sync = false;
  GE_ASSERT_NOTNULL(op_desc_);
  if (AttrUtils::GetBool(op_desc_, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync) && is_soft_sync) {
    GE_CHECK_NOTNULL(op_);
    GE_ASSERT_TRUE(static_cast<size_t>(op_desc_->GetOppImplVersion()) < space_registries_->size());
    GE_CHK_STATUS_RET(
        optiling::SoftSyncOpRtParseAndTiling(*op_, platform_infos_, *run_info_,
                                             space_registries_->at(static_cast<size_t>(op_desc_->GetOppImplVersion()))),
        "Recall tiling for soft sync op: %s failed.", op_desc_->GetName().c_str());
    GE_CHK_STATUS_RET(ExecutorUtils::AssembleReuseBinaryArgs(op_desc_, *run_info_, args_ex_), "Refresh Args failed.");
    GE_CHK_STATUS_RET(UpdateRunInfoByTilingResult(), "[Update][RunInfo] by tiling result failed.");
    has_overflow_attr_ = has_overflow_attr_ && (overflow_addr_ != nullptr);
    size_t new_size = 0U;
    UpdateArgsItemOffset((ffts_addr_num_ + input_num_ + output_num_) * sizeof(uintptr_t),
                         workspaces_.size() * sizeof(uintptr_t), new_size);
  }
  const auto ret = DoLaunchKernel(stream);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Invoke][RtKernelLaunch] failed. ret = %u, task = %s", ret, this->stub_name_.c_str());
    REPORT_INNER_ERR_MSG("E19999", "invoke rtKernelLaunch failed, ret = %u, task = %s", ret, this->stub_name_.c_str());
    return ret;
  }
  GELOGI("[TASK_INFO] %s", this->stub_name_.c_str());

  return SUCCESS;
}

Status TbeOpTask::PreProcess(uint64_t &launch_begin_time) {
  launch_begin_time = MsprofSysCycleTime();
  GE_ASSERT_SUCCESS(ReportL0ExceptionDumpInfo(op_desc_, l0_dump_list_), "[%s] report l0 exception dump addr failed",
                    op_desc_->GetNamePtr());
  return ge::SUCCESS;
}

void TbeOpTask::SaveForL0ExceptionDump() {
  // aclop流程单算子不会演进，因此没有占位、二级指针等场景，用默认顺序即可
  const size_t iow_total_size =
      op_desc_->GetInputsSize() + op_desc_->GetOutputsSize() + op_desc_->GetWorkspaceBytes().size();
  l0_dump_list_.resize(iow_total_size + ffts_addr_num_, 0);
  for (size_t i = 0UL; i < iow_total_size; ++i) {
    l0_dump_list_[i + ffts_addr_num_] = i;
  }
}

Status TbeOpTask::CalcTilingInfo() {
  GE_CHECK_NOTNULL(op_);
  const auto ret = optiling::OpParaCalculateV2(*op_, *run_info_);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Invoke][OpParaCalculate] failed, ret = %u.", ret);
    REPORT_INNER_ERR_MSG("E19999", "invoke OpParaCalculate failed, ret = %u.", ret);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status TbeOpTask::UpdateRunInfo() {
  PROFILING_SCOPE(-1, profiling::kTiling);
  // invoke OpParaCalculate
  GELOGD("Start to invoke OpParaCalculate.");
  run_info_->ResetWorkspace();
  constexpr size_t ptr_size = sizeof(void *);
  void *const tiling_data_addr = ValueToPtr(PtrToValue(args_.get()) +
                                            static_cast<uint64_t>(ptr_size * tiling_data_idx_));
  run_info_->ResetAddrBase(tiling_data_addr, static_cast<uint64_t>(max_tiling_size_));
  GE_CHK_STATUS_RET(CalcTilingInfo(), "[Calc][TilingInfo]failed.");

  GE_CHK_STATUS_RET(UpdateRunInfoByTilingResult(), "Update run info by tiling result failed.");
  GELOGD("Invoking OpParaCalculate successfully. block_dim is %u, tiling_key is %" PRIu64 ".",
         block_dim_, tiling_key_);
  return SUCCESS;
}

Status TbeOpTask::UpdateRunInfoByTilingResult() {
  block_dim_ = run_info_->GetBlockDim();
  tiling_key_ = run_info_->GetTilingKey();
  clear_atomic_ = run_info_->GetClearAtomic();
  cfg_.schemMode = static_cast<uint8_t>(run_info_->GetScheduleMode() & k2BitsMask);
  cfg_.localMemorySize = run_info_->GetLocalMemorySize();
  const auto workspaces = run_info_->GetAllWorkspaces();
  GELOGI("Update run info of %s, block dim: %u, tiling key: %" PRIu64 ","
         "clear atomic: %d, workspace size: %zu, schedule_mode: %u, local memory size: %u.",
         op_desc_->GetName().c_str(), block_dim_, tiling_key_,
         static_cast<int32_t>(clear_atomic_), workspaces.size(),
         static_cast<uint32_t>(cfg_.schemMode), cfg_.localMemorySize);

  return SUCCESS;
}

Status TbeOpTask::UpdateNodeByShape(const std::vector<GeTensorDesc> &input_desc,
                                    const std::vector<GeTensorDesc> &output_desc) const {
  PROFILING_SCOPE(-1, profiling::kUpdateShape);
  const auto op_desc = node_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // Set runtime shape to node
  for (size_t i = 0UL; i < input_desc.size(); ++i) {
    const auto tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(tensor_desc);
    tensor_desc->SetShape(input_desc[i].GetShape());
    tensor_desc->SetOriginShape(input_desc[i].GetOriginShape());
  }

  for (size_t i = 0UL; i < output_desc.size(); ++i) {
    const auto tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(tensor_desc);
    tensor_desc->SetShape(output_desc[i].GetShape());
    tensor_desc->SetOriginShape(output_desc[i].GetOriginShape());
  }

  return SUCCESS;
}

void TbeOpTask::EnableDynamicSupport(const NodePtr &node, const uint32_t max_tiling_size) {
  node_ = node;
  constexpr uint32_t size_of_uintptr = static_cast<uint32_t>(sizeof(uintptr_t));
  max_tiling_size_ = (max_tiling_size + size_of_uintptr - 1U) / size_of_uintptr * size_of_uintptr;
  need_tiling_ = max_tiling_size > 0U;
}

Status TbeOpTask::CheckAndExecuteAtomic(const std::vector<GeTensorDesc> &input_desc,
                                        const std::vector<DataBuffer> &input_buffers,
                                        std::vector<GeTensorDesc> &output_desc,
                                        std::vector<DataBuffer> &output_buffers,
                                        rtStream_t const stream) {
  (void)stream;
  if (clear_atomic_ && (atomic_task_ != nullptr)) {
    RT2_PROFILING_SCOPE_CONST(gert::profiling::kAtomic, gert::profiling::kOpExecute);
    atomic_task_->SetWorkSpaceAddr(workspaces_);
    return atomic_task_->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream);
  }
  return SUCCESS;
}

Status TbeOpTask::UpdateArgsItem(const std::vector<DataBuffer> &inputs,
                                 const std::vector<DataBuffer> &outputs) {
  size_t new_size = 0U;
  has_overflow_attr_ = has_overflow_attr_ && (overflow_addr_ != nullptr);
  UpdateArgsItemOffset((ffts_addr_num_ + input_num_ + output_num_) * sizeof(uintptr_t),
                       workspaces_.size() * sizeof(uintptr_t), new_size);
  GE_CHK_STATUS_RET(ExtendArgSizeIfNeed(new_size), "[Extend][ArgSizeIfNeed] failed.");
  GE_CHK_STATUS_RET(UpdateTilingArgs(), "[Update][TilingArgs] failed.");
  // workspace must be after tiling updating
  UpdateWorkspaceArgs();
  UpdateOverflowAddr();
  GE_CHK_STATUS_RET(UpdateHostMemInputArgs(inputs, outputs),
                    "[Update][Args] failed of %s.", node_->GetName().c_str());
  return SUCCESS;
}

void TbeOpTask::UpdateArgsItemOffset(const size_t io_size, const size_t workspace_addr_size, size_t &arg_size) {
  arg_size = io_size;
  args_item_offsets_.overflow_addr_offset = 0U;
  args_item_offsets_.workspace_addr_offset = 0U;
  args_item_offsets_.tiling_addr_offset = 0U;
  args_item_offsets_.tiling_data_offset = 0U;
  args_item_offsets_.host_input_data_offset = 0U;

  if (workspace_addr_size != 0U) {
    args_item_offsets_.workspace_addr_offset = arg_size;
    arg_size +=  workspace_addr_size;
  }
  if (need_tiling_) {
    args_item_offsets_.tiling_addr_offset = arg_size;
    arg_size += sizeof(void *);
  }
  if (has_overflow_attr_) {
    args_item_offsets_.overflow_addr_offset = arg_size;
    arg_size += sizeof(void *);
  }
  if (need_tiling_) {
    args_item_offsets_.tiling_data_offset = arg_size;
    arg_size += max_tiling_size_;
  }
  if (extend_args_for_host_input_) {
    args_item_offsets_.host_input_data_offset = arg_size;
    arg_size += kMaxHostMemInputLen;
  }

  if (arg_size > io_size) {
    GELOGD("args size is extended frome %zu to %zu. overflow flag = %u, tiling flag = %u, host mem flag = %u,"
        " max tiling size = %u, workspace_addr_size = %zu, host_input_data_offset = %zu", io_size, arg_size,
        ((overflow_addr_ != nullptr) ? 1U : 0U), (need_tiling_ ? 1U : 0U), (extend_args_for_host_input_ ? 1U : 0U),
        max_tiling_size_, workspace_addr_size, args_item_offsets_.host_input_data_offset);
  }
}

Status TbeOpTask::ExtendArgSizeIfNeed(size_t new_size) {
  if (arg_size_ >= new_size) {
    return SUCCESS;
  }
  GELOGD("Need to reset size of args_ from %zu to %zu.", arg_size_, new_size);
  std::unique_ptr<uint8_t[]> args = MakeUnique<uint8_t[]>(new_size);
  GE_CHECK_NOTNULL(args);
  if (memcpy_s(args.get(), new_size, args_.get(), arg_size_) != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][KernelArgs] failed for [%s].", node_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "update kernel args failed for %s.", node_->GetName().c_str());
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  args_ = std::move(args);
  arg_size_ = new_size;
  args_ex_.args = args_.get();
  args_ex_.argsSize = static_cast<uint32_t>(arg_size_);
  args_ex_.isNoNeedH2DCopy = 0U;
  return SUCCESS;
}

void TbeOpTask::UpdateOverflowAddr() const {
  if (has_overflow_attr_) {
    uintptr_t *const addr = PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(args_ex_.args) +
                                                         args_item_offsets_.overflow_addr_offset));
    *addr = static_cast<uintptr_t>(PtrToValue(overflow_addr_));
  }
}

void TbeOpTask::UpdateWorkspaceArgs() {
  if (workspaces_.empty()) {
    return;
  }
  void *const arg_base = args_ex_.args;
  uintptr_t *arg_workspace = PtrToPtr<void, uintptr_t>(ValueToPtr(
      PtrToValue(arg_base) +
      static_cast<uint64_t>(args_item_offsets_.workspace_addr_offset)));
  for (size_t i = 0UL; i < workspaces_.size(); ++i) {
    *arg_workspace = static_cast<uintptr_t>(PtrToValue(workspaces_[i]));
    arg_workspace++;
  }
}

Status TbeOpTask::UpdateTilingArgs() {
  RT2_PROFILING_SCOPE_CONST(gert::profiling::kUnknownName, gert::profiling::kKernelLaunchPrepare);
  args_ex_.hasTiling = need_tiling_;
  if (!need_tiling_) {
    return SUCCESS;
  }

  const size_t tiling_data_index = args_item_offsets_.tiling_data_offset / sizeof(void *);
  if (tiling_data_index > tiling_data_idx_) {
    GELOGD("[%s] Start to copy tiling info.", node_->GetName().c_str());
    uintptr_t *const tiling_arg = PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(args_.get()) +
                                                                 args_item_offsets_.tiling_addr_offset));
    void *const tiling_data = ValueToPtr(PtrToValue(args_.get()) + args_item_offsets_.tiling_data_offset);
    void *const tiling_data_old = ValueToPtr(PtrToValue(args_.get()) +
                                             static_cast<uint64_t>(tiling_data_idx_ * sizeof(void *)));
    if (memmove_s(tiling_data, static_cast<size_t>(max_tiling_size_),
                  tiling_data_old, static_cast<size_t>(max_tiling_size_)) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][Args] failed of %s.", node_->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "update kernel args failed of %s.", node_->GetName().c_str());
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    // must after tiling data copy
    *tiling_arg = static_cast<uintptr_t>(PtrToValue(tiling_data));

    tiling_data_idx_ = tiling_data_index;
    args_ex_.tilingAddrOffset = static_cast<uint32_t>(args_item_offsets_.tiling_addr_offset);
    args_ex_.tilingDataOffset = static_cast<uint32_t>(args_item_offsets_.tiling_data_offset);
  }
  return SUCCESS;
}

Status TbeOpTask::SetArgIndex() {
  const std::vector<bool> v_is_input_const = op_desc_->GetIsInputConst();
  size_t input_index = 0UL;
  for (size_t i = 0UL; i < op_desc_->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGD("SingleOp: %s, Index: %zu, has no input", op_desc_->GetName().c_str(), i);
      continue;
    }
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      GELOGD("SingleOp: %s, Index: %zu, input is const", op_desc_->GetName().c_str(), i);
      input_index++;
      continue;
    }
    arg_index_.emplace_back(input_index);
    input_index++;
  }
  return SUCCESS;
}

Status TbeOpTask::UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs,
                                         const std::vector<DataBuffer> &outputs) {
  (void)outputs;
  args_ex_.hostInputInfoPtr = nullptr;
  args_ex_.hostInputInfoNum = 0U;
  if (!need_host_mem_opt_) {
    return SUCCESS;
  }

  vector<rtHostInputInfo_t> host_inputs;
  GE_CHK_STATUS_RET_NOLOG(ExecutorUtils::UpdateHostMemInputArgs(inputs, *this, args_.get(), arg_size_, host_inputs));
  host_inputs_info_ = MakeUnique<rtHostInputInfo_t[]>(host_inputs.size());
  GE_CHECK_NOTNULL(host_inputs_info_);
  size_t idx = 0U;
  for (auto &host_input : host_inputs) {
    host_inputs_info_[idx] = host_input;
    idx++;
  }
  args_ex_.hostInputInfoPtr = host_inputs_info_.get();
  args_ex_.hostInputInfoNum = static_cast<uint16_t>(host_inputs.size());
  return SUCCESS;
}

Status TbeOpTask::UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  RT2_PROFILING_SCOPE_CONST(gert::profiling::kUnknownName, gert::profiling::kKernelLaunchPrepare);
  if (arg_index_.size() != inputs.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size] Args size is %zu, but get input size is %zu.",
           arg_index_.size(), inputs.size());
    REPORT_INNER_ERR_MSG("E19999", "[Check][Size] Args size is %zu, but get input size is %zu.",
                       arg_index_.size(), inputs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  constexpr size_t ptr_size = sizeof(uintptr_t);
  uintptr_t *const arg_base =
      PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(args_.get()) + ffts_addr_num_ * ptr_size));
  for (size_t i = 0UL; i < arg_index_.size(); ++i) {
    uintptr_t *const arg_tiling_pre = PtrToPtr<void, uintptr_t>(
        ValueToPtr(PtrToValue(arg_base) +
                   static_cast<uint64_t>(ptr_size * arg_index_[i])));
    *arg_tiling_pre = static_cast<uintptr_t>(PtrToValue(inputs[i].data));
  }

  uintptr_t *arg_output = PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(arg_base) +
                                                               static_cast<uint64_t>(ptr_size * input_num_)));
  for (size_t i = 0UL; i < op_desc_->GetOutputsSize(); ++i) {
    *arg_output = static_cast<uintptr_t>(PtrToValue(outputs[i].data));
    ++arg_output;
  }

  return SUCCESS;
}

Status TbeOpTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                               const std::vector<DataBuffer> &input_buffers,
                               std::vector<GeTensorDesc> &output_desc,
                               std::vector<DataBuffer> &output_buffers,
                               rtStream_t const stream) {
  GELOGD("[%s] Start to launch kernel", node_->GetName().c_str());
  GE_CHK_STATUS_RET(UpdateIoAddr(input_buffers, output_buffers), "[Update][IoAddr] failed.");
  GE_CHK_STATUS_RET_NOLOG(UpdateNodeByShape(input_desc, output_desc));
  GE_CHK_STATUS_RET_NOLOG(UpdateRunInfo());
  GE_CHK_STATUS_RET(CheckAndExecuteAtomic(input_desc, input_buffers, output_desc, output_buffers, stream),
                    "[Execute][AtomicTask] failed.");
  GE_CHK_STATUS_RET(UpdateArgsItem(input_buffers, output_buffers),
                    "[Update][ArgsItem] failed of %s", node_->GetName().c_str());

  GELOGD("[%s] Start to invoke rtKernelLaunch", node_->GetName().c_str());
  GE_CHK_STATUS_RET(DoLaunchKernel(stream), "Failed to do launch kernel.");
  return SUCCESS;
}

Status TbeOpTask::DoLaunchKernel(rtStream_t const stream) {
  GE_PROFILING_START(kRt2tKernelLaunch);
  GE_CHK_RT_RET(static_cast<rtError_t>(DoLaunchKernelWithArgsEx(stream)));
  GE_PROFILING_END(gert::profiling::kUnknownName, gert::profiling::kStaticSingleOpKernelLaunch, kRt2tKernelLaunch);
  GE_CHK_STATUS_RET_NOLOG(GetTaskIdAndStreamId(stream));
  return SUCCESS;
}

Status TbeOpTask::DoLaunchKernelWithArgsEx(rtStream_t const stream) {
  auto *const sm_desc = PtrToPtr<void, rtSmDesc_t>(sm_desc_);
  cfg_.dumpflag = 0U;
  if (handle_ == nullptr) {
    GE_CHK_RT_RET(rtKernelLaunchWithFlagV2(stub_func_, block_dim_, &args_ex_, sm_desc, stream, 0U, &cfg_));
  } else {
    SetTaskTag();
    GE_CHK_RT_RET(rtKernelLaunchWithHandleV2(handle_, tiling_key_, block_dim_, &args_ex_,
                                             sm_desc, stream, &cfg_));
  }
  return SUCCESS;
}

void TbeOpTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = PtrToPtr<uint8_t, uintptr_t>(args_.get());
  arg_base += ffts_addr_num_;
  arg_count = (arg_size_ - max_tiling_size_) / sizeof(void *);  // for tiling data
  arg_count -= ffts_addr_num_;
  if (need_tiling_) {
    --arg_count;  // for tiling arg
  }
}

Status TbeOpTask::ReportProfExtendInfo(const uint64_t end_time, const uint64_t op_name_hash, const int32_t tid) const {
  bool is_fftsplus_task = false;
  (void)AttrUtils::GetBool(op_desc_, kAttrIsFFTSTask, is_fftsplus_task);
  if (!is_fftsplus_task) {
    return SUCCESS;
  }
  // for mix op, report ctxIdNum = 1, ctxIds[0] = 0
  GELOGI("[Cann Profiling] node name %s is fftsplus task.", op_desc_->GetName().c_str());

  MsprofAdditionalInfo context_info;
  context_info.level = MSPROF_REPORT_NODE_LEVEL;
  context_info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  context_info.threadId = static_cast<uint32_t>(tid);
  context_info.dataLen = static_cast<uint32_t>(sizeof(MsprofContextIdInfo));
  context_info.timeStamp = end_time;
  auto context_data = PtrToPtr<uint8_t, MsprofContextIdInfo>(context_info.data);
  GE_ASSERT_NOTNULL(context_data);
  context_data->opName = op_name_hash;
  context_data->ctxIdNum = 1U;
  context_data->ctxIds[0] = 0U;
  GE_ASSERT_MSPROF_OK(
      MsprofReportAdditionalInfo(true, &context_info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  return SUCCESS;
}

Status AtomicAddrCleanOpTask::UpdateNodeByShape(const std::vector<GeTensorDesc> &input_desc,
                                                const std::vector<GeTensorDesc> &output_desc) const {
  (void)input_desc;
  (void)output_desc;
  return SUCCESS;
}

const std::string AtomicAddrCleanOpTask::GetOpType() const {
  return kAtomicOpType;
}

Status AtomicAddrCleanOpTask::UpdateIoAddr(const std::vector<DataBuffer> &inputs,
                                           const std::vector<DataBuffer> &outputs) {
  (void)inputs;
  uintptr_t *arg_base = PtrToPtr<uint8_t, uintptr_t>(args_.get());
  for (const int32_t atomic_output_index : atomic_output_indices_) {
    if (atomic_output_index >= static_cast<int32_t>(outputs.size())) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Update][Args] failed, atomic index must smaller then data size.");
      REPORT_INNER_ERR_MSG("E19999", "[Update][Args] failed, atomic index must smaller then data size.");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    auto &output_buffer = outputs[static_cast<size_t>(atomic_output_index)];
    *arg_base = static_cast<uintptr_t>(PtrToValue(output_buffer.data));
    ++arg_base;

    const auto tensor_desc = op_desc_->MutableOutputDesc(static_cast<uint32_t>(atomic_output_index));
    GE_ASSERT_NOTNULL(tensor_desc);
    int64_t size = 0;
    const graphStatus graph_status = TensorUtils::GetTensorMemorySizeInBytes(*tensor_desc, size);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get tensor size in bytes failed!");
      GELOGE(graph_status, "[Get][TensorMemorySize] In Bytes failed!");
      return FAILED;
    }
    TensorUtils::SetSize(*tensor_desc, size);
  }
  for (const int32_t atomic_ws_index : atomic_workspace_indices_) {
    if (atomic_ws_index >= static_cast<int32_t>(workspaces_.size())) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[Update][Args] failed, workspace atomic index must smaller then workspace size.");
      REPORT_INNER_ERR_MSG("E19999", "[Update][Args] failed, workspace atomic index must smaller then workspace size.");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    *arg_base = static_cast<uintptr_t>(PtrToValue(workspaces_[static_cast<size_t>(atomic_ws_index)]));
    ++arg_base;
  }

  return SUCCESS;
}

Status AtomicAddrCleanOpTask::UpdateTilingArgs() {
  return SUCCESS;
}

void AtomicAddrCleanOpTask::UpdateOverflowAddr() const {
  // do nothing
}

Status AtomicAddrCleanOpTask::CalcTilingInfo() {
  const auto ret = optiling::OpAtomicCalculateV2(*node_, *run_info_);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Invoke][OpAtomicCalculate] failed, ret = %u.", ret);
    REPORT_INNER_ERR_MSG("E19999", "invoke OpAtomicCalculate failed, ret = %u.", ret);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status AtomicAddrCleanOpTask::InitAtomicAddrCleanIndices() {
  GELOGD("[%s] Start to setup AtomicAddrClean task.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET(
      ExecutorUtils::InitAtomicAddrCleanIndices(op_desc_, atomic_output_indices_, atomic_workspace_indices_),
      "Init atomic addr clean indices of %s failed.", op_desc_->GetName().c_str());

  const size_t arg_count = atomic_workspace_indices_.size() + atomic_output_indices_.size();
  uintptr_t *arg_base = nullptr;
  size_t max_arg_size = 0U;
  GetIoAddr(arg_base, max_arg_size);
  if (arg_count > max_arg_size) {
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] atomic_output_indices invalid. atomic_output_indices size is %zu,"
           "arg size is %zu.", op_desc_->GetName().c_str(), arg_count, arg_size_);
    REPORT_INNER_ERR_MSG("E19999", "[%s] atomic_output_indices invalid. atomic_output_indices size is %zu,"
                       "arg size is %zu.", op_desc_->GetName().c_str(), arg_count, arg_size_);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

AiCpuBaseTask::~AiCpuBaseTask() noexcept {
  if (ext_info_addr_dev_ != nullptr) {
    (void)rtFree(ext_info_addr_dev_);
  }
  if (rt_event_ != nullptr) {
    (void)rtEventDestroy(rt_event_);
  }
  FreeHbm(copy_input_release_flag_dev_);
  FreeHbm(copy_input_data_size_dev_);
  FreeHbm(copy_input_src_dev_);
  FreeHbm(copy_input_dst_dev_);
  for (const auto summary : output_summary_) {
    FreeHbm(summary);
  }
  for (const auto out_shape : out_shape_hbm_) {
    FreeHbm(out_shape);
  }
}

Status AiCpuBaseTask::UpdateEventIdForBlockingAicpuOp() {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOpProcess failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("Device not support blocking aicpu op process");
    return SUCCESS;
  }
  uint32_t event_id = 0U;
  auto rt_ret = rtEventCreateWithFlag(&rt_event_, RT_EVENT_WITH_FLAG);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtEventCreateWithFlag failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtEventCreateWithFlag] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtGetEventID(rt_event_, &event_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtGetEventID failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetEventID] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (aicpu_ext_handle_->UpdateEventId(event_id) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Update event id=%u failed.", event_id);
    GELOGE(FAILED, "[Update][EventId] Update event id=%u failed", event_id);
    return FAILED;
  }
  GELOGI("Update event_id=%u success", event_id);
  return SUCCESS;
}

Status AiCpuBaseTask::SetExtInfoAndType(const std::string &kernel_ext_info, const uint64_t kernel_id) {
  if (kernel_ext_info.empty()) {
    GELOGI("Kernel_ext_info is empty, no need copy to device.");
    return SUCCESS;
  }

  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(op_desc_, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  GELOGD("Get unknown_type is %d.", unknown_shape_type_val);
  unknown_type_ = static_cast<UnknowShapeOpType>(unknown_shape_type_val);

  (void)AttrUtils::GetBool(op_desc_, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op_);
  GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc_->GetName().c_str(),
         static_cast<int32_t>(is_blocking_aicpu_op_));

  aicpu_ext_handle_ = MakeUnique<::ge::hybrid::AicpuExtInfoHandler>(op_desc_->GetName(),
                                                                    num_inputs_,
                                                                    num_outputs_,
                                                                    unknown_type_);
  GE_CHK_BOOL_RET_STATUS(aicpu_ext_handle_ != nullptr, ACL_ERROR_GE_MEMORY_ALLOCATION,
                         "[Malloc][Memory] failed for aicpu_ext_handle!");

  const Status ret = aicpu_ext_handle_->Parse(kernel_ext_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Param:kernel_ext_info] failed, kernel_ext_info_size=%zu.", kernel_ext_info.size());
    REPORT_INNER_ERR_MSG("E19999",
        "Parse Param:kernel_ext_info failed, kernel_ext_info_size=%zu.", kernel_ext_info.size());
    return ret;
  }
  deploy_type_flag_ = aicpu_ext_handle_->GetDeployTypeFlag();
  mem_type_ = aicpu_ext_handle_->GetMemType();
  memcpy_kind_ = aicpu_ext_handle_->GetMemcpyKind();
  GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateSessionInfo(std::numeric_limits<uint64_t>::max(), kernel_id, false),
                    "[Update][SessionInfo] failed.");

  if (is_blocking_aicpu_op_) {
    if (UpdateEventIdForBlockingAicpuOp() != SUCCESS) {
      GELOGE(FAILED, "[Call][UpdateEventIdForBlockingAicpuOp] Call UpdateEventIdForBlockingAicpuOp failed");
      return FAILED;
    }
  }

  GE_CHK_RT_RET(rtMalloc(&ext_info_addr_dev_, aicpu_ext_handle_->GetExtInfoLen(), RT_MEMORY_HBM,
                         GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(ext_info_addr_dev_, aicpu_ext_handle_->GetExtInfoLen(),
                         aicpu_ext_handle_->GetExtInfo(), aicpu_ext_handle_->GetExtInfoLen(),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuBaseTask::SetInputConst() {
  input_is_const_.clear();
  const std::vector<bool> v_is_input_const = op_desc_->GetIsInputConst();
  for (size_t i = 0U; i < op_desc_->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGD("SingleOp: %s, Index: %zu, has no input", op_desc_->GetName().c_str(), i);
      continue;
    }
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      GELOGD("SingleOp: %s, Index: %zu, input is const", op_desc_->GetName().c_str(), i);
      input_is_const_.push_back(kInputIsConst);
      continue;
    }
    input_is_const_.push_back(0);
  }
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateExtInfo(const std::vector<GeTensorDesc> &input_desc,
                                    const std::vector<GeTensorDesc> &output_desc,
                                    rtStream_t const stream) {
  GELOGI("Update ext info begin, unknown_type is %d.", unknown_type_);
  GE_CHECK_NOTNULL(aicpu_ext_handle_);
  GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateExecuteMode(false), "[Update][ExecuteMode] failed.");

  if ((num_inputs_ == 0U) && (num_outputs_ == 0U)) {
    GELOGI("No input and output, no need update ext info.");
    return SUCCESS;
  }

  size_t non_const_index = 0U;
  for (size_t input_index = 0U; input_index < num_inputs_; input_index++) {
    if ((input_index < input_is_const_.size()) && (input_is_const_[input_index] == kInputIsConst)) {
      // get input_desc from op_desc if const input, num_inputs_ is op_desc_ input_size
      const auto const_input_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(input_index));
      GE_CHECK_NOTNULL(const_input_desc);
      GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateInputShapeAndType(
          static_cast<uint32_t>(input_index), *const_input_desc),
          "[Update][InputShapeAndType] failed, input_index:%zu.", input_index);
      continue;
    }
    GE_CHK_BOOL_RET_STATUS(non_const_index < input_desc.size(), ACL_ERROR_GE_PARAM_INVALID,
        "[Check][Size]Input_desc size is %zu, but get non_const_index is %zu", input_desc.size(), non_const_index);
    GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateInputShapeAndType(static_cast<uint32_t>(input_index),
        input_desc[non_const_index]), "[Update][InputShapeAndType]failed, input_index:%zu.", input_index);
    if (DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsSingleOpNeedDump()) {
      GE_CHK_STATUS_RET(op_desc_->UpdateInputDesc(static_cast<uint32_t>(input_index),
          input_desc[non_const_index]), "AiCpuTask Update [%zu]th input desc failed.", input_index);
    }
    non_const_index++;
  }

  if (unknown_type_ != DEPEND_COMPUTE) {
    for (size_t j = 0U; j < num_outputs_; ++j) {
      GE_CHK_STATUS_RET(aicpu_ext_handle_->UpdateOutputShapeAndType(static_cast<uint32_t>(j),
          output_desc[j]), "[Update][OutputShapeAndType] failed, Output:%zu.", j);
      if (DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsSingleOpNeedDump()) {
        GE_CHK_STATUS_RET(op_desc_->UpdateOutputDesc(static_cast<uint32_t>(j), output_desc[j]),
                          "AiCpuTask Update [%zu]th output desc failed.", j);
    }
    }
  }
  // aicpu_ext_handle_->GetExtInfoLen() 已校验过非空
  GE_CHK_RT_RET(rtMemcpyAsync(ext_info_addr_dev_,
                              aicpu_ext_handle_->GetExtInfoLen(), // check size
                              aicpu_ext_handle_->GetExtInfo(),
                              aicpu_ext_handle_->GetExtInfoLen(),
                              RT_MEMCPY_HOST_TO_DEVICE_EX,
                              stream));

  GELOGI("Update ext info end.");
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateOutputShape(std::vector<GeTensorDesc> &output_desc) {
  if (num_outputs_ == 0U) {
    GELOGD("AiCpuBaseTask output_num is 0, no need update output shape.");
    return SUCCESS;
  }
  GELOGD("Start to update DEPEND_SHAPE_RANGE AiCpuBaseTask outputshape.");

  GE_CHK_RT_RET(rtMemcpy(aicpu_ext_handle_->GetExtInfo(), aicpu_ext_handle_->GetExtInfoLen(), ext_info_addr_dev_,
                         aicpu_ext_handle_->GetExtInfoLen(), RT_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0U; i < num_outputs_; ++i) {
    GeShape shape;
    DataType data_type;
    (void)aicpu_ext_handle_->GetOutputShapeAndType(static_cast<uint32_t>(i), shape, data_type);
    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(shape, output_desc[i]),
        "[Update][ShapeToOutputDesc] failed, output:%zu. datatype:%d.", i, data_type);
    if (DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsSingleOpNeedDump()) {
      GE_CHK_STATUS_RET(op_desc_->UpdateOutputDesc(static_cast<uint32_t>(i), output_desc[i]),
                        "[Update][OutputDesc] failed, output:%zu.", i);
    }
  }
  GELOGD("Update DEPEND_SHAPE_RANGE AiCpuBaseTask outputshape finished.");
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateShapeToOutputDesc(const GeShape &shape_new, GeTensorDesc &output_desc) const {
  const auto shape_old = output_desc.GetShape();
  output_desc.SetShape(shape_new);
  GELOGD("Update AiCpuBaseTask shape from %s to %s", shape_old.ToString().c_str(), shape_new.ToString().c_str());

  const auto origin_shape_old = output_desc.GetOriginShape();
  const auto origin_format = output_desc.GetOriginFormat();
  const auto format = output_desc.GetFormat();
  if (origin_format == format) {
    output_desc.SetOriginShape(shape_new);
    return SUCCESS;
  }

  std::vector<int64_t> origin_dims_new;
  const Status trans_ret =
      formats::TransTensorShape(format, shape_new.GetDims(), output_desc.GetDataType(), origin_format, origin_dims_new);
  GE_CHK_STATUS_RET(trans_ret,
                    "[Trans][Shape] failed, AiCpuTask originFormat[%d] is not same as format[%d], shape=%s.",
                    origin_format, format, shape_new.ToString().c_str());

  const auto origin_shape_new = GeShape(origin_dims_new);
  output_desc.SetOriginShape(origin_shape_new);
  GELOGD("AiCpuTask originFormat[%d] is not same as format[%d], need update from %s ro %s.",
         origin_format, format, origin_shape_old.ToString().c_str(), origin_shape_new.ToString().c_str());
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0U;
  GetIoAddr(arg_base, arg_num);

  // input number and output number was check in ValidateParams
  size_t non_const_index = 0U;
  for (size_t input_index = 0U; input_index < num_inputs_; input_index++) {
    if ((input_index < input_is_const_.size()) && (input_is_const_[input_index] == kInputIsConst)) {
      // const input no need update addr
      GE_CHECK_NOTNULL(arg_base);
      GELOGD("AICpuTask input[%zu] addr = %" PRIu64, input_index, *arg_base);
      arg_base++;
      continue;
    }
    GE_CHK_BOOL_RET_STATUS(non_const_index < inputs.size(), ACL_ERROR_GE_PARAM_INVALID,
        "[Check][Size] Input size is %zu, but get non_const_index is %zu", inputs.size(), non_const_index);
    const auto addr = inputs[non_const_index].data;
    const uint64_t length = inputs[non_const_index].length;
    if ((length != 0U) && (addr == nullptr)) {
      GELOGE(PARAM_INVALID, "[Check][Addr]AiCpuTask input[%zu] addr is nullptr, length = %" PRIu64,
        input_index, length);
      return PARAM_INVALID;
    }
    GELOGD("AICpuTask input[%zu] addr = %p, length = %" PRIu64 ".", input_index, addr, length);
    *arg_base = static_cast<uintptr_t>(PtrToValue(addr));
    ++arg_base;
    non_const_index++;
  }

  for (size_t i = 0U; i < outputs.size(); ++i) {
    const auto addr = outputs[i].data;
    const uint64_t length = outputs[i].length;
    if ((length != 0U) && (addr == nullptr)) {
      GELOGE(PARAM_INVALID, "[Check][Addr]AiCpuTask output[%zu] addr is nullptr, length = %" PRIu64, i, length);
      return PARAM_INVALID;
    }
    GELOGD("AICpuTask output[%zu] addr = %p, length = %" PRIu64 ".", i, addr, length);
    *arg_base = static_cast<uintptr_t>(PtrToValue(addr));
    ++arg_base;
  }
  GE_CHK_STATUS_RET(UpdateHostMemInputArgs(inputs, outputs),
                    "[Update][Args] failed of %s.", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status AiCpuBaseTask::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const {
  int32_t device_id = 0;
  auto rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtGetDevice failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetDevice] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  int32_t value = 0;
  rt_ret = rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtGetDeviceCapability failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetDeviceCapability] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
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

Status AiCpuBaseTask::DistributeWaitTaskForAicpuBlockingOp(rtStream_t const stream) {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOpProcess failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("Device not support blocking aicpu op process.");
    return SUCCESS;
  }
  GELOGI("Distribute queue task begin");
  if (rt_event_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "rt_event_ is nullptr");
    GELOGE(FAILED, "[Check][rt_event_] rt_event_ is nullptr");
    return FAILED;
  }
  SetTaskTag();
  auto rt_ret = rtStreamWaitEvent(stream, rt_event_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtStreamWaitEvent failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  SetTaskTag();
  rt_ret = rtEventReset(rt_event_, stream);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtEventReset failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

AiCpuTask::~AiCpuTask() noexcept {
  FreeHbm(args_);
  FreeHbm(io_addr_);
  FreeHbm(workspace_addr_);
  FreeHbm(copy_workspace_buf_);
  FreeHbm(copy_ioaddr_dev_);
  FreeHbm(copy_task_args_buf_);
}

Status AiCpuTask::UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs,
                                         const std::vector<DataBuffer> &outputs) {
  (void)outputs;
  if (!need_host_mem_opt_) {
    return SUCCESS;
  }
  vector<rtHostInputInfo_t> host_inputs;
  GE_CHK_STATUS_RET_NOLOG(ExecutorUtils::UpdateHostMemInputArgs(inputs, *this, io_addr_host_.data(),
                                                                io_addr_host_.size() * sizeof(void *), host_inputs));
  for (auto &host_input : host_inputs) {
    const size_t index = host_input.addrOffset / sizeof(void *);
    io_addr_host_[index] = ValueToPtr(PtrToValue(io_addr_) + host_input.dataOffset);
  }
  return SUCCESS;
}

Status AiCpuTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("Start to launch kernel. task = %s", this->op_type_.c_str());
  SetTaskTag();
  const tagRtMemcpyKind memcpy_kind =
      ((deploy_type_flag_ == static_cast<int32_t>(RT_KERNEL_HOST_ONLY)) ? memcpy_kind_ : RT_MEMCPY_HOST_TO_DEVICE_EX);
  auto ret = RT_ERROR_NONE;
  if (io_addr_host_.size() > 0U) {
    ret = rtMemcpyAsync(io_addr_, io_addr_size_, io_addr_host_.data(), io_addr_host_.size() * sizeof(void *),
                             memcpy_kind, stream);
  }
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[MemcpyAsync][Date] failed. ret = %d, task = %s", ret, this->op_type_.c_str());
    REPORT_INNER_ERR_MSG("E19999", "rtMemcpyAsync data failed, ret = %d, task = %s", ret, this->op_type_.c_str());
    return RT_ERROR_TO_GE_STATUS(ret);
  }

  GELOGI("To invoke rtKernelLaunchEx. task = %s", this->op_type_.c_str());
  SetTaskTag();
  ret = rtKernelLaunchEx(args_, static_cast<uint32_t>(arg_size_), static_cast<uint32_t>(deploy_type_flag_), stream);
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Invoke][rtKernelLaunch] failed. ret = %d, task = %s", ret, this->op_type_.c_str());
    REPORT_INNER_ERR_MSG("E19999", "invoke rtKernelLaunchEx failed, ret = %d, task = %s", ret, this->op_type_.c_str());
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  GE_CHK_STATUS_RET_NOLOG(GetTaskIdAndStreamId(stream));
  GELOGI("[TASK_INFO] %" PRIu64 "/%s", kernel_id_, op_type_.c_str());

  GELOGD("Done launch kernel successfully. task = %s", this->op_type_.c_str());

  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp(stream) != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] Call DistributeWaitTaskForAicpuBlockingOp failed");
      return FAILED;
    }
  }

  return SUCCESS;
}

Status AiCpuBaseTask::PrepareCopyInputs(const std::vector<DataBuffer> &outputs) {
  std::vector<uint64_t> copy_input_release_flag;
  std::vector<uint64_t> copy_input_data_size;
  std::vector<uint64_t> copy_input_src;
  std::vector<uint64_t> copy_input_dst;

  for (size_t i = 0U; i < num_outputs_; ++i) {
    const auto &summary = output_summary_host_[i];
    GELOGI("Node out[%zu] summary, shape data=0x%" PRIx64 ", shape data size=%" PRIu64 ", "
           "raw data=0x%" PRIx64 ", raw data size=%" PRIu64 ".",
           i, summary.shape_data_ptr, summary.shape_data_size,
           summary.raw_data_ptr, summary.raw_data_size);
    const auto output = outputs[i];
    copy_input_release_flag.emplace_back(kReleaseFlag);
    if (summary.raw_data_size != 0U) {
      copy_input_data_size.emplace_back(output.length);
    } else {
      copy_input_data_size.emplace_back(summary.raw_data_size);
    }
    copy_input_src.emplace_back(summary.raw_data_ptr);
    copy_input_dst.emplace_back(PtrToValue(output.data));

    const auto &shape_buffer = out_shape_hbm_[i];
    copy_input_release_flag.emplace_back(kReleaseFlag);
    copy_input_data_size.emplace_back(summary.shape_data_size);
    copy_input_src.emplace_back(summary.shape_data_ptr);
    copy_input_dst.emplace_back(PtrToValue(shape_buffer));
  }

  const size_t copy_input_buf_len = num_outputs_ * kCopyNum * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMemcpy(copy_input_release_flag_dev_, copy_input_buf_len,
                         copy_input_release_flag.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_data_size_dev_, copy_input_buf_len,
                         copy_input_data_size.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_src_dev_, copy_input_buf_len,
                         copy_input_src.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(copy_input_dst_dev_, copy_input_buf_len,
                         copy_input_dst.data(), copy_input_buf_len, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuBaseTask::ReadResultSummaryAndPrepareMemory() {
  for (size_t i = 0U; i < num_outputs_; ++i) {
    auto &result_summary = output_summary_host_[i];

    GE_CHK_RT_RET(rtMemcpy(&result_summary, sizeof(aicpu::FWKAdapter::ResultSummary),
                           output_summary_[i], sizeof(aicpu::FWKAdapter::ResultSummary),
                           RT_MEMCPY_DEVICE_TO_HOST));
    const size_t shape_data_size = result_summary.shape_data_size;
    void *shape_buffer = nullptr;
    if (shape_data_size > 0U) {
      GE_CHK_RT_RET(rtMalloc(&shape_buffer, shape_data_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    }
    out_shape_hbm_.emplace_back(shape_buffer);
  }
  return SUCCESS;
}

Status AiCpuCCTask::CopyDataToHbm(std::vector<DataBuffer> &outputs,
                                  rtStream_t const stream) {
  GE_CHK_STATUS_RET_NOLOG(PrepareCopyInputs(outputs));
  rtArgsEx_t args_ex = {};
  args_ex.args = memcpy_args_.get();
  args_ex.argsSize = memcpy_args_size_;
  const auto ret = rtCpuKernelLaunchWithFlag(static_cast<const void *>(memcpy_so_name_.data()),
                                             static_cast<const void *>(memcpy_kernel_name_.data()),
                                             block_dim_, &args_ex,
                                             nullptr, stream, RT_KERNEL_DEFAULT);
  GE_CHK_RT_RET(ret);
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  return SUCCESS;
}

Status AiCpuTask::CopyDataToHbm(std::vector<DataBuffer> &outputs,
                                rtStream_t const stream) {
  GE_CHK_STATUS_RET_NOLOG(PrepareCopyInputs(outputs));

  GE_CHK_RT_RET(rtKernelLaunchEx(copy_task_args_buf_, static_cast<uint32_t>(sizeof(STR_FWK_OP_KERNEL)),
                                 RT_KERNEL_DEFAULT, stream));
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateShapeByHbmBuffer(std::vector<GeTensorDesc> &output_desc) {
  for (size_t i = 0U; i < num_outputs_; ++i) {
    const auto &result_summary = output_summary_host_[i];
    std::vector<int64_t> shape_dims;
    if (result_summary.shape_data_size > 0U) {
      const auto &shape_hbm = out_shape_hbm_[i];

      const uint32_t dim_num = static_cast<uint32_t>(result_summary.shape_data_size / sizeof(int64_t));
      const std::unique_ptr<int64_t[]> shape_addr = MakeUnique<int64_t[]>(static_cast<size_t>(dim_num));
      GE_CHECK_NOTNULL(shape_addr);
      GE_CHK_RT_RET(rtMemcpy(shape_addr.get(), result_summary.shape_data_size, shape_hbm,
                             result_summary.shape_data_size, RT_MEMCPY_DEVICE_TO_HOST));

      for (size_t dim_idx = 0U; dim_idx < dim_num; ++dim_idx) {
        shape_dims.emplace_back(shape_addr[dim_idx]);
        GELOGD("Node [%zu]th output dim[%zu]=%" PRId64 ".", i, dim_idx, shape_addr[dim_idx]);
      }
    }

    GE_CHK_STATUS_RET(UpdateShapeToOutputDesc(GeShape(shape_dims), output_desc[i]),
        "[Update][ShapeToOutputDesc] failed , output:%zu.", i);
    if (DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsSingleOpNeedDump()) {
      GE_CHK_STATUS_RET(op_desc_->UpdateOutputDesc(static_cast<uint32_t>(i), output_desc[i]),
          "[Update][OutputDesc] failed, output:%zu.", i);
    }
  }
  return SUCCESS;
}


Status AiCpuBaseTask::UpdateShapeAndDataByResultSummary(std::vector<GeTensorDesc> &output_desc,
                                                        std::vector<DataBuffer> &outputs,
                                                        rtStream_t const stream) {
  if (num_outputs_ == 0U) {
    GELOGI("Output num is 0, there is no need to update the output and size.");
    return SUCCESS;
  }

  GELOGI("Update shape and data by result summary begin.");

  for (const auto out_shape : out_shape_hbm_) {
    FreeHbm(out_shape);
  }
  out_shape_hbm_.clear();
  GE_CHK_STATUS_RET(ReadResultSummaryAndPrepareMemory(),
                    "[Read][ResultSummaryAndPrepareMemory] failed.");

  GE_CHK_STATUS_RET(CopyDataToHbm(outputs, stream),
                    "[Copy][DataToHbm] failed.");

  GE_CHK_STATUS_RET(UpdateShapeByHbmBuffer(output_desc),
                    "[Update][ShapeByHbmBuffer] failed.");

  for (const auto out_shape : out_shape_hbm_) {
    FreeHbm(out_shape);
  }
  out_shape_hbm_.clear();

  GELOGI("Update shape and data by result summary end.");
  return SUCCESS;
}

Status AiCpuTask::InitForSummaryAndCopy() {
  if ((unknown_type_ != DEPEND_COMPUTE) || (num_outputs_ == 0U)) {
    GELOGI("Unknown_type is %d, output num is %zu.", unknown_type_, num_outputs_);
    return SUCCESS;
  }

  output_summary_.resize(num_outputs_);
  for (size_t i = 0U; i < num_outputs_; ++i) {
    constexpr size_t result_summary_size = sizeof(aicpu::FWKAdapter::ResultSummary);
    GE_CHK_RT_RET(rtMalloc(&output_summary_[i], result_summary_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  }
  output_summary_host_.resize(num_outputs_);

  const size_t copy_input_buf_len = num_outputs_ * kCopyNum * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMalloc(&copy_input_release_flag_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_input_data_size_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_input_src_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_input_dst_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_task_args_buf_, sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM, GE_MODULE_NAME_U16));

  std::vector<uint64_t> copy_io_addr;
  copy_io_addr.emplace_back(PtrToValue(copy_input_release_flag_dev_));
  copy_io_addr.emplace_back(PtrToValue(copy_input_data_size_dev_));
  copy_io_addr.emplace_back(PtrToValue(copy_input_src_dev_));
  copy_io_addr.emplace_back(PtrToValue(copy_input_dst_dev_));

  const uint64_t copy_io_addr_size = sizeof(uint64_t) * static_cast<uint64_t>(copy_io_addr.size());

  GE_CHK_RT_RET(rtMalloc(&copy_ioaddr_dev_, copy_io_addr_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));

  GE_CHK_RT_RET(rtMemcpy(copy_ioaddr_dev_, copy_io_addr_size,
                         copy_io_addr.data(), copy_io_addr_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuTask::SetMemCopyTask(const domi::KernelExDef &kernel_def) {
  if (kernel_def.args_size() > sizeof(STR_FWK_OP_KERNEL)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]sizeof STR_FWK_OP_KERNEL is: %" PRIu64 ", but args_size is: %u",
        sizeof(STR_FWK_OP_KERNEL), kernel_def.args_size());
    REPORT_INNER_ERR_MSG("E19999", "[sizeof STR_FWK_OP_KERNEL is: %" PRIu64 ", but args_size is: %u",
        static_cast<uint64_t>(sizeof(STR_FWK_OP_KERNEL)), kernel_def.args_size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_CHK_RT_RET(rtMalloc(&copy_workspace_buf_, static_cast<uint64_t>(kernel_def.task_info_size()),
                         RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHECK_GE(kernel_def.task_info().size(), static_cast<size_t>(kernel_def.task_info_size()));
  GE_CHK_RT_RET(rtMemcpy(copy_workspace_buf_, static_cast<uint64_t>(kernel_def.task_info_size()),
                         kernel_def.task_info().data(), static_cast<uint64_t>(kernel_def.task_info_size()),
                         RT_MEMCPY_HOST_TO_DEVICE));

  STR_FWK_OP_KERNEL aicpu_task = {};
  const auto sec_ret = memcpy_s(&aicpu_task, sizeof(STR_FWK_OP_KERNEL),
                                kernel_def.args().data(), kernel_def.args().size());
  if (sec_ret != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][TaskArgs] failed, ret: %d", sec_ret);
    REPORT_INNER_ERR_MSG("E19999", "update STR_FWK_OP_KERNEL args failed because memcpy_s return %d.", sec_ret);
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  aicpu_task.fwkKernelBase.fwk_kernel.inputOutputAddr = PtrToValue(copy_ioaddr_dev_);
  aicpu_task.fwkKernelBase.fwk_kernel.workspaceBaseAddr = PtrToValue(copy_workspace_buf_);
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoAddr = 0U;
  aicpu_task.fwkKernelBase.fwk_kernel.extInfoLen = 0U;

  GE_CHK_RT_RET(rtMemcpy(copy_task_args_buf_, sizeof(STR_FWK_OP_KERNEL),
                         &aicpu_task, sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status AiCpuTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                               const std::vector<DataBuffer> &input_buffers,
                               std::vector<GeTensorDesc> &output_desc,
                               std::vector<DataBuffer> &output_buffers,
                               rtStream_t const stream) {
  GE_CHK_STATUS_RET_NOLOG(UpdateExtInfo(input_desc, output_desc, stream));
  if (unknown_type_ == DEPEND_COMPUTE) {
    std::vector<DataBuffer> summary_buffers;
    for (size_t i = 0U; i < num_outputs_; ++i) {
      summary_buffers.emplace_back(output_summary_[i], sizeof(aicpu::FWKAdapter::ResultSummary), false);
    }
    GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, summary_buffers));
  } else {
    GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, output_buffers));
  }

  GE_CHK_STATUS_RET_NOLOG(LaunchKernel(stream));
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateOutputShape(output_desc));
  } else if (unknown_type_ == DEPEND_COMPUTE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateShapeAndDataByResultSummary(output_desc, output_buffers, stream));
  } else {
    // something else
  }

  return SUCCESS;
}

Status AiCpuCCTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                                 const std::vector<DataBuffer> &input_buffers,
                                 std::vector<GeTensorDesc> &output_desc,
                                 std::vector<DataBuffer> &output_buffers,
                                 rtStream_t const stream) {
  GE_CHK_STATUS_RET_NOLOG(UpdateExtInfo(input_desc, output_desc, stream));
  if (unknown_type_ == DEPEND_COMPUTE) {
    std::vector<DataBuffer> summary_buffers;
    for (size_t i = 0U; i < num_outputs_; ++i) {
      summary_buffers.emplace_back(output_summary_[i], sizeof(aicpu::FWKAdapter::ResultSummary), false);
    }
    GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, summary_buffers));
  } else {
    GE_CHK_STATUS_RET_NOLOG(UpdateIoAddr(input_buffers, output_buffers));
  }

  GE_CHK_STATUS_RET_NOLOG(LaunchKernel(stream));
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateOutputShape(output_desc));
  } else if (unknown_type_ == DEPEND_COMPUTE) {
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GE_CHK_STATUS_RET_NOLOG(UpdateShapeAndDataByResultSummary(output_desc, output_buffers, stream));
  } else {
    // something else
  }

  return SUCCESS;
}

Status AiCpuCCTask::InitForSummaryAndCopy() {
  if ((unknown_type_ != DEPEND_COMPUTE) || (num_outputs_ == 0U)) {
    GELOGI("Unknown_type is %d, output num is %zu.", unknown_type_, num_outputs_);
    return SUCCESS;
  }

  output_summary_.resize(num_outputs_);
  for (size_t i = 0U; i < num_outputs_; ++i) {
    constexpr size_t result_summary_size = sizeof(aicpu::FWKAdapter::ResultSummary);
    GE_CHK_RT_RET(rtMalloc(&output_summary_[i], result_summary_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  }
  output_summary_host_.resize(num_outputs_);

  const size_t copy_input_buf_len = num_outputs_ * kCopyNum * sizeof(uint64_t);

  GE_CHK_RT_RET(rtMalloc(&copy_input_release_flag_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_input_data_size_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_input_src_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMalloc(&copy_input_dst_dev_, copy_input_buf_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));

  copy_io_addr_.emplace_back(PtrToValue(copy_input_release_flag_dev_));
  copy_io_addr_.emplace_back(PtrToValue(copy_input_data_size_dev_));
  copy_io_addr_.emplace_back(PtrToValue(copy_input_src_dev_));
  copy_io_addr_.emplace_back(PtrToValue(copy_input_dst_dev_));
  return SUCCESS;
}

Status AiCpuCCTask::SetMemCopyTask(const domi::KernelDef &kernel_def) {
  auto &memcpy_args = kernel_def.args();
  memcpy_args_size_ = kernel_def.args_size();
  memcpy_so_name_ = kernel_def.so_name();
  memcpy_kernel_name_ = kernel_def.kernel_name();
  if (memcpy_args.size() != memcpy_args_size_) {
    REPORT_INNER_ERR_MSG("E19999", "MemCopy task def args.size=%zu, but args_size=%u not equal.",
                       memcpy_args.size(), memcpy_args_size_);
    GELOGE(FAILED, "[Check][Size]MemCopy task def args.size=%zu, but args_size=%u not equal.",
           memcpy_args.size(), memcpy_args_size_);
    return FAILED;
  }
  if (memcpy_args_size_ < sizeof(aicpu::AicpuParamHead)) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Task def args_size=%u is less than aicpu param head len=%zu.",
                       memcpy_args_size_, sizeof(aicpu::AicpuParamHead));
    GELOGE(FAILED,
           "[Check][Size] Task def args_size=%u is less than aicpu param head len=%zu.",
           memcpy_args_size_, sizeof(aicpu::AicpuParamHead));
    return FAILED;
  }

  memcpy_args_ = MakeUnique<uint8_t[]>(static_cast<size_t>(memcpy_args_size_));
  if (memcpy_args_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "new memory failed for Node[MemCopy], task_size[%u].",
                       memcpy_args_size_);
    GELOGE(FAILED, "[Malloc][Memory] failed for Node[MemCopy], task_size[%u].",
           memcpy_args_size_);
    return FAILED;
  }

  const errno_t sec_ret = memcpy_s(memcpy_args_.get(), static_cast<size_t>(memcpy_args_size_),
                                   memcpy_args.c_str(), memcpy_args.size());
  if (sec_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999",
                       "memcpy_s argc_ failed for Node[MemCopy], ret: %d", sec_ret);
    GELOGE(INTERNAL_ERROR,
           "[Update][args] failed for Node[MemCopy], ret: %d", sec_ret);
    return FAILED;
  }
  const auto memcpy_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(memcpy_args_.get());
  const uint32_t memcpy_io_num = memcpy_param_head->ioAddrNum;
  const auto memcpy_io_addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(memcpy_args_.get()) +
                                                                 sizeof(aicpu::AicpuParamHead)));
  // if has input and output, need copy to ioaddr
  const int32_t cpy_ret = memcpy_s(memcpy_io_addr,
                                   static_cast<size_t>(memcpy_args_size_ - sizeof(aicpu::AicpuParamHead)),
                                   &copy_io_addr_[0U], sizeof(uint64_t) * memcpy_io_num);
  if (cpy_ret != 0) {
    REPORT_INNER_ERR_MSG("E19999", "Node[Memcpoy] memcpy io addr to AicpuParamHead failed,"
        "ret=%d, args_size=%u, io nums=%u.", cpy_ret, memcpy_args_size_, memcpy_io_num);
    GELOGE(INTERNAL_ERROR, "[Update][io_addr]Node[MemCopy] memcpy io addr to AicpuParamHead failed,"
        "ret=%d, args_size=%u, io nums=%u.", cpy_ret, memcpy_args_size_, memcpy_io_num);
    return INTERNAL_ERROR;
  }
  GELOGD("Set memcpy task for node[MemCopy] successfully.");
  return SUCCESS;
}

Status AiCpuBaseTask::UpdateArgTable(const SingleOpModelParam &param) {
  // aicpu do not have workspace, for now
  return DoUpdateArgTable(param, false);
}

const std::string &AiCpuBaseTask::GetTaskType() const { return kTaskTypeAicpu; }

void AiCpuTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = PtrToPtr<void *, uintptr_t>(io_addr_host_.data());
  arg_count = io_addr_host_.size();
}

void AiCpuCCTask::SetKernelArgs(std::unique_ptr<uint8_t[]> args, const size_t arg_size) {
  args_ = std::move(args);
  arg_size_ = arg_size;
  args_ex_.args = args_.get();
  args_ex_.argsSize = static_cast<uint32_t>(arg_size_);
  args_ex_.isNoNeedH2DCopy = 0U;
}

void AiCpuCCTask::SetSoName(const std::string &so_name) { so_name_ = so_name; }

void AiCpuCCTask::SetkernelName(const std::string &kernel_Name) { kernel_name_ = kernel_Name; }

void AiCpuCCTask::SetIoAddr(uintptr_t *const io_addr) { io_addr_ = io_addr; }

AiCpuCCTask::~AiCpuCCTask() noexcept = default;

Status AiCpuCCTask::UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs,
                                           const std::vector<DataBuffer> &outputs) {
  (void)outputs;
  args_ex_.hostInputInfoPtr = nullptr;
  args_ex_.hostInputInfoNum = 0U;
  if (!need_host_mem_opt_) {
    return SUCCESS;
  }
  vector<rtHostInputInfo_t> host_inputs;
  GE_CHK_STATUS_RET_NOLOG(ExecutorUtils::UpdateHostMemInputArgs(inputs, *this, io_addr_,
                                                                arg_size_ - sizeof(aicpu::AicpuParamHead),
                                                                host_inputs));
  host_inputs_info_ = MakeUnique<rtHostInputInfo_t[]>(host_inputs.size());
  GE_CHECK_NOTNULL(host_inputs_info_);
  size_t idx = 0U;
  for (auto &host_input : host_inputs) {
    host_input.dataOffset += static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead));
    host_input.addrOffset += static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead));
    host_inputs_info_[idx] = host_input;
    idx++;
  }
  args_ex_.hostInputInfoPtr = host_inputs_info_.get();
  args_ex_.hostInputInfoNum = static_cast<uint16_t>(host_inputs.size());
  return SUCCESS;
}

Status AiCpuCCTask::LaunchKernel(rtStream_t const stream) {
  GELOGI("To invoke rtCpuKernelLaunch. block_dim = %u, so_name is %s, kernel_name is %s", block_dim_, so_name_.data(),
         kernel_name_.data());
  // sm_desc is nullptr, because l2 buffer does not support
  auto *const sm_desc = PtrToPtr<void, rtSmDesc_t>(sm_desc_);
  SetTaskTag();
  dump_flag_ = dump_flag_ | static_cast<uint32_t>(deploy_type_flag_);
  const auto ret = rtCpuKernelLaunchWithFlag(static_cast<const void *>(so_name_.data()),
                                             static_cast<const void *>(kernel_name_.data()), block_dim_, &args_ex_,
                                             sm_desc, stream, dump_flag_);
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Invoke][rtCpuKernelLaunchWithFlag] failed. ret = %d.", ret);
    REPORT_INNER_ERR_MSG("E19999", "invoke rtCpuKernelLaunchWithFlag failed, ret:%d.", ret);
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  GE_CHK_STATUS_RET_NOLOG(GetTaskIdAndStreamId(stream));
  GELOGI("[TASK_INFO] %" PRIu64 "/%s", kernel_id_, op_type_.c_str());
  GELOGD("Invoke rtCpuKernelLaunch succeeded");

  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp(stream) != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] Call DistributeWaitTaskForAicpuBlockingOp failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

void AiCpuCCTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = io_addr_;
  arg_count = io_addr_num_;
}

Status MemcpyAsyncTask::LaunchKernel(rtStream_t const stream) {
  const auto src_addr = ValueToPtr(static_cast<uint64_t>(addresses_[0]));
  const auto dst_addr = ValueToPtr(static_cast<uint64_t>(addresses_[1]));
  kind_ = (kind_ == RT_MEMCPY_ADDR_DEVICE_TO_DEVICE) ? RT_MEMCPY_DEVICE_TO_DEVICE : kind_;
  SetTaskTag();
  MemcpyAsyncMemInfo memcpy_info;
  memcpy_info.dst = dst_addr;
  memcpy_info.destMax = dst_max_;
  memcpy_info.src = src_addr;
  memcpy_info.cnt = count_;

  // |----mpamid:bit4~11(0~127)|------qos:bit0~3(0~7)|
  const uint32_t qos_cfg =
          ((static_cast<uint32_t>(cfg_.qos) & 0XFU) | ((static_cast<uint32_t>(cfg_.partId) & 0xFFU) << 4U));
  GE_CHK_RT_RET(ge::rtMemcpyAsyncWithCfg(memcpy_info, static_cast<rtMemcpyKind_t>(kind_), stream, qos_cfg));
  GE_CHK_STATUS_RET_NOLOG(GetTaskIdAndStreamId(stream));
  return SUCCESS;
}

void MemcpyAsyncTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = &addresses_[0];
  arg_count = kMemcpyArgCount;
}

Status MixL2OpTask::ReportProfExtendInfo(const uint64_t end_time, const uint64_t op_name_hash,
                                         const int32_t tid) const {
  const auto ctx_num = context_ids_.size();
  GELOGD("[Cann Profiling] ctx num is %u, node name is %s, node type is %s",
         ctx_num, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
  if ((ctx_num > 0U) && (ctx_num <= MSPROF_CTX_ID_MAX_NUM)) {
    MsprofAdditionalInfo context_info;
    context_info.level = MSPROF_REPORT_NODE_LEVEL;
    context_info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    context_info.threadId = static_cast<uint32_t>(tid);
    context_info.dataLen = static_cast<uint32_t>(sizeof(MsprofContextIdInfo));
    context_info.timeStamp = end_time;
    auto context_data = reinterpret_cast<MsprofContextIdInfo *>(context_info.data);
    context_data->opName = op_name_hash;
    context_data->ctxIdNum = static_cast<uint32_t>(ctx_num);
    for (size_t index = 0U; index < ctx_num; index++) {
      context_data->ctxIds[index] = context_ids_[index];
    }
    GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(true, &context_info,
        static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  }

  return SUCCESS;
}


Status MixL2OpTask::PreProcess(uint64_t &launch_begin_time) {
  launch_begin_time = MsprofSysCycleTime();
  GE_ASSERT_SUCCESS(SetL0ExceptionSizeInfo(op_desc_, l0_dump_list_), "[%s] report l0 exception dump addr failed",
                    op_desc_->GetNamePtr());
  return ge::SUCCESS;
}

void MixL2OpTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = &host_args_[args_addr_base_idx_];
  arg_count = args_addr_cnt_;
  if (need_tiling_) {
    --arg_count;  // for tiling arg
  }
}

void MixL2OpTask::GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) {
  const auto op_desc = GetOpdesc();
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  tiling_info = op_desc->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  if (tiling_info != nullptr) {
    tiling_key = static_cast<uint32_t>(tiling_info->GetTilingKey());
    const size_t tiling_data_size = tiling_info->GetAllTilingData().str().size();
    const size_t tiling_data_addr_idx = op_desc->GetAllInputsDescPtr().size() + op_desc->GetWorkspaceBytes().size() +
                                        static_cast<size_t>(op_desc->GetAllOutputsDescSize()) + args_addr_base_idx_;
    if (tiling_data_addr_idx >= host_args_.size()) {
      return;
    }
    const uintptr_t tiling_addr = host_args_[tiling_data_addr_idx];
    auto tiling_data_holder = MakeUnique<uint8_t[]>(tiling_data_size);
    GE_CHECK_NOTNULL_JUST_RETURN(tiling_data_holder);
    const auto ret = rtMemcpy(tiling_data_holder.get(), static_cast<uint64_t>(tiling_data_size),
                              reinterpret_cast<void *>(tiling_addr), static_cast<uint64_t>(tiling_data_size),
                              RT_MEMCPY_DEVICE_TO_HOST);
    if (ret == RT_ERROR_NONE) {
      std::stringstream ss;
      gert::PrintHex(tiling_data_holder.get(), tiling_data_size, ss);
      tiling_data = ss.str();
    }
  }
}

void MixL2OpTask::GetHostArgsAndSize(uintptr_t &args, size_t &arg_size) {
  args = reinterpret_cast<uintptr_t>(host_args_.data());
  arg_size = host_args_.size() * sizeof(uintptr_t);
}

Status MixL2OpTask::UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs,
                                           const std::vector<DataBuffer> &outputs) {
  (void)outputs;
  if (!need_host_mem_opt_) {
    return SUCCESS;
  }
  // |tiling data|host mem data|mode addrs|input addrs|output addrs|workspace addrs|tiling addr|
  size_t max_dst_len = kMaxHostMemInputLen;
  uint8_t *host_mem_base_addr = PtrToPtr<uintptr_t, uint8_t>(&host_args_[max_tiling_size_ / sizeof(uintptr_t)]);
  uint64_t device_data_offset = PtrToValue(device_args_) + max_tiling_size_;
  for (size_t i = 0UL; i < inputs.size(); ++i) {
    if (inputs[i].placement == kHostMemType) {
      GE_CHECK_LE(inputs[i].length, SIZE_MAX);
      const size_t aligned_len = (static_cast<size_t>(inputs[i].length) + sizeof(uintptr_t) - 1U) /
                           sizeof(uintptr_t) * sizeof(uintptr_t);
      GE_CHECK_LE(aligned_len, max_dst_len);
      // Copy tensor data to host args
      if (memcpy_s(host_mem_base_addr, max_dst_len, inputs[i].data, inputs[i].length) != EOK) {
        GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
               "[Update][HostMemInputArgs]failed, dst length is %zu, src length is %zu.",
               max_dst_len, inputs[i].length);
        REPORT_INNER_ERR_MSG("E19999", "update kernel args failed");
        return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      }
      // Update addr for host memory
      host_args_[args_addr_base_idx_ + arg_index_[i]] = static_cast<uintptr_t>(device_data_offset);
      host_mem_base_addr = PtrAdd(host_mem_base_addr, max_dst_len, aligned_len);
      device_data_offset += aligned_len;
      max_dst_len -= aligned_len;
    }
  }
  return SUCCESS;
}

Status MixL2OpTask::UpdateIoAddr(const std::vector<DataBuffer> &inputs,
                                 const std::vector<DataBuffer> &outputs) {
  const auto data_size = arg_index_.size();
  if (data_size != inputs.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size] Args size is %zu, but get input size is %zu.",
           data_size, inputs.size());
    REPORT_INNER_ERR_MSG("E19999", "[Check][Size] Args size is %zu, but get input size is %zu.",
                       data_size, inputs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (size_t i = 0UL; i < data_size; ++i) {
    const uintptr_t data_base = static_cast<uintptr_t>(PtrToValue(inputs[i].data));
    host_args_[args_addr_base_idx_ + arg_index_[i]] = data_base;
  }

  for (size_t i = 0UL; i < output_num_; ++i) {
    const uintptr_t data_base = static_cast<uintptr_t>(PtrToValue(outputs[i].data));
    host_args_[args_addr_base_idx_ + input_num_ + i] = data_base;
  }
  return SUCCESS;
}

Status MixL2OpTask::DoLaunchKernel(rtStream_t const stream) {
  return LaunchKernel(stream);
}

Status MixL2OpTask::LaunchKernel(rtStream_t const stream) {
  // single op mode
  if (!host_args_.empty()) {
    GE_CHK_RT(rtMemcpyAsync(device_args_, arg_size_, host_args_.data(), host_args_.size() * sizeof(uintptr_t),
                            RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  }

  GELOGD("Start to call rtFftsPlusTaskLaunch.");
  GE_CHK_RT_RET(ge::rtFftsPlusTaskLaunch(&ffts_plus_task_info_, stream));
  GELOGD("Call rtFftsPlusTaskLaunch succeeded.");
  return SUCCESS;
}

Status MixL2OpTask::LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                                 const std::vector<DataBuffer> &input_buffers, std::vector<GeTensorDesc> &output_desc,
                                 std::vector<DataBuffer> &output_buffers, rtStream_t const stream) {
  (void)input_desc;
  (void)input_buffers;
  (void)output_desc;
  (void)output_buffers;
  (void)stream;
  REPORT_INNER_ERR_MSG("E19999", "V1 dynamic shape process does not support mixl2.");
  GELOGE(ge::PARAM_INVALID, "V1 dynamic shape process does not support mixl2.");
  return FAILED;
};

MixL2OpTask::~MixL2OpTask() noexcept {
  for (auto &addr : ext_args_) {
    GE_FREE_RT_LOG(addr);
  }
  if (device_args_ != nullptr) {
    GE_FREE_RT_LOG(device_args_);
  }
  CleanRtFftsPlusTask(ffts_plus_task_info_);
}

Status NpuGetFloatStatusTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("NpuGetFloatStatusTask launch in.");
  GE_CHK_RT_RET(rtMemcpyAsync(args_, args_size_, &output_addr_, args_size_, RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  GE_CHK_RT_RET(ge::rtNpuGetFloatStatus(args_, output_size_, mode_, stream));
  return SUCCESS;
}

void NpuGetFloatStatusTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = reinterpret_cast<uintptr_t *>(&output_addr_);
  arg_count = 1UL;
}

NpuGetFloatStatusTask::~NpuGetFloatStatusTask() noexcept {
  if (args_ != nullptr) {
    GE_FREE_RT_LOG(args_);
  }
}

Status NpuClearFloatStatusTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("NpuClearFloatStatusTask launch in.");
  GE_CHK_RT_RET(ge::rtNpuClearFloatStatus(mode_, stream));
  return SUCCESS;
}

Status NpuGetFloatDebugStatusTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("NpuGetFloatDebugStatusTask launch in.");
  GE_CHK_RT_RET(rtMemcpyAsync(args_, args_size_, &output_addr_, args_size_, RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  GE_CHK_RT_RET(ge::rtNpuGetFloatDebugStatus(args_, output_size_, mode_, stream));
  return SUCCESS;
}

void NpuGetFloatDebugStatusTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = reinterpret_cast<uintptr_t *>(&output_addr_);
  arg_count = 1UL;
}

NpuGetFloatDebugStatusTask::~NpuGetFloatDebugStatusTask() noexcept {
  if (args_ != nullptr) {
    GE_FREE_RT_LOG(args_);
  }
}

Status NpuClearFloatDebugStatusTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("NpuClearFloatDebugStatusTask launch in.");
  GE_CHK_RT_RET(ge::rtNpuClearFloatDebugStatus(mode_, stream));
  return SUCCESS;
}

void DsaTask::GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) {
  arg_base = PtrToPtr<void *, uintptr_t>(io_addr_.data());
  arg_count = io_addr_.size();
}

Status DsaTask::UpdateDsaSqe(rtStream_t const stream) {
  if (io_addr_.size() != (input_size_ + output_size_ + workspace_size_)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "io_addr size is not right");
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  const uint64_t dev_output_addr = PtrToValue(io_addr_[input_size_]);
  dsa_sqe_.dsaCfgResultAddrLow = static_cast<uint32_t>(dev_output_addr & kMask32Bits);
  dsa_sqe_.dsaCfgResultAddrHigh = static_cast<uint32_t>(dev_output_addr >> k32Bits);

  if (workspace_size_ == kDSAWorkspaceAddrSize) {
    const uint64_t workspace_philox_count_addr = PtrToValue(io_addr_[input_size_ + output_size_]);
    dsa_sqe_.dsaCfgStateAddrLow = static_cast<uint32_t>(workspace_philox_count_addr & kMask32Bits);
    dsa_sqe_.dsaCfgStateAddrHigh = static_cast<uint32_t>(workspace_philox_count_addr >> k32Bits);
  } else {
    dsa_sqe_.dsaCfgStateAddrLow = static_cast<uint32_t>(PtrToValue(io_addr_[input_size_ - 1U]) & kMask32Bits);
    dsa_sqe_.dsaCfgStateAddrHigh = static_cast<uint32_t>(PtrToValue(io_addr_[input_size_ - 1U]) >> k32Bits);
  }

  const uint64_t workspace_input_addr = PtrToValue(io_addr_[input_size_ + output_size_ + workspace_size_ - 1U]);
  dsa_sqe_.dsaCfgParamAddrLow = static_cast<uint32_t>(workspace_input_addr & kMask32Bits);
  dsa_sqe_.dsaCfgParamAddrHigh = static_cast<uint32_t>(workspace_input_addr >> k32Bits);

  if (input1_value_or_ptr_ == kDSASetInputAddr) {
    std::vector<uint64_t> input_addr{PtrToValue(io_addr_[2U])};
    if ((input_size_ == kDSAStateInputAddrSize) ||
        ((input_size_ == kDSAArgsInputAddrSize) && (workspace_size_ == kDSAWorkspaceAddrSize))) {
      input_addr.push_back(PtrToValue(io_addr_[3U]));
    }
    GELOGD("Try to do async memory copy, dst_addr = %p, dst_size = %zu, src_addr = %d, src_size = %zu, stream = %p",
           workspace_input_addr, sizeof(uint64_t) * 2U, input_addr.data(), sizeof(uint64_t) * input_addr.size(),
           stream);
    // 此处无需校验，可以保证原地址非空且src_size > 0
    GE_CHK_RT_RET(rtMemcpyAsync(ValueToPtr(workspace_input_addr), sizeof(uint64_t) * 2U, input_addr.data(),
                                sizeof(uint64_t) * input_addr.size(), RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  } else {
    GELOGD("Try to do async memory copy, dst_addr = %p, dst_size = %zu, src_addr = %d, src_size = %zu, stream = %p",
           workspace_input_addr, sizeof(uint64_t) * 2U, input_data_, sizeof(input_data_), stream);
    // 此处无需校验，可以保证原地址非空且src_size > 0
    GE_CHK_RT_RET(rtMemcpyAsync(ValueToPtr(workspace_input_addr), sizeof(uint64_t) * 2U, input_data_,
                                sizeof(input_data_), RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  }

  if (seed_value_or_ptr_ == kDSASetInputAddr) {
    dsa_sqe_.dsaCfgSeedLow = static_cast<uint32_t>(PtrToValue(io_addr_[1U]) & kMask32Bits);
    dsa_sqe_.dsaCfgSeedHigh = static_cast<uint32_t>(PtrToValue(io_addr_[1U]) >> k32Bits);
  }

  if (random_count_value_or_ptr_ == kDSASetInputAddr) {
    dsa_sqe_.dsaCfgNumberLow = static_cast<uint32_t>(PtrToValue(io_addr_[0U]) & kMask32Bits);
    dsa_sqe_.dsaCfgNumberHigh = static_cast<uint32_t>(PtrToValue(io_addr_[0U]) >> k32Bits);
  }
  return SUCCESS;
}

Status DsaTask::LaunchKernel(rtStream_t const stream) {
  GELOGD("DSATask Distribute Start.");
  GE_CHK_STATUS_RET(UpdateDsaSqe(stream), "Update dsa sqe failed");
  GE_CHK_RT_RET(ge::rtStarsTaskLaunchWithFlag(&dsa_sqe_, static_cast<uint32_t>(sizeof(dsa_sqe_)), stream, 0U));
  return SUCCESS;
}
}  // namespace ge
