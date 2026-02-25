/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include "runtime/subscriber/global_profiler.h"
#include "base/err_msg.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/profiling_definitions.h"
#include "common/profiling/profiling_properties.h"
#include "common/global_variables/diagnose_switch.h"
#include "framework/runtime/device_memory_recorder.h"
#include "runtime/dev.h"
#include "common/scope_guard.h"
#include "common/util.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "utils/extern_math_util.h"
#include "framework/runtime/device_memory_recorder.h"
#include "graph/compute_graph.h"
#include "aprof_pub.h"
#include "graph/debug/ge_attr_define.h"

namespace gert {
const std::unordered_map<std::string, GeProfInfoType> kNamesToProfTypes = {
    {"ModelExecute", GeProfInfoType::kModelExecute},
    {"ModelLoad", GeProfInfoType::kModelLoad},
    {"InputCopy", GeProfInfoType::kInputCopy},
    {"OutputCopy", GeProfInfoType::kOutputCopy},
    {"InferShape", GeProfInfoType::kInferShape},
    {"CompatibleInferShape", GeProfInfoType::kCompatibleInferShape},
    {"Tiling", GeProfInfoType::kTiling},
    {"CompatibleTiling", GeProfInfoType::kCompatibleTiling},
    {"StreamSync", GeProfInfoType::kStreamSync},
    {"step_info", GeProfInfoType::kStepInfo},
    {"isGraphNeedRebuild", GeProfInfoType::kIsGraphNeedRebuild},
    {"RemoveGraph", GeProfInfoType::kRemoveGraph},
    {"AddGraph", GeProfInfoType::kAddGraph},
    {"BuildGraph", GeProfInfoType::kBuildGraph},
    {"RunGraphAsync", GeProfInfoType::kRunGraphAsync},
    {"GEInitialize", GeProfInfoType::kGEInitialize},
    {"GEFinalize", GeProfInfoType::kGEFinalize},
    {"AicpuHostCompute", GeProfInfoType::kHostOpExec}
  };

namespace {
constexpr char_t kVersionSingleThread[] = "2.0-SingleThread";
constexpr uint32_t kIdOffset = 16U;
constexpr uint32_t kFusionOpInfoCap = 52U;
constexpr uint32_t kHashOffset = 8U;
const std::string kOpImplModeEnum = "_op_impl_mode_enum";
constexpr uint32_t kEnableHf32 = 0x40U;
constexpr uint32_t kOpImplHf32Mode = 1U;

REGISTER_PROF_TYPE(LaunchHcomKernel);
REGISTER_PROF_TYPE(LaunchKernelWithHandle);
REGISTER_PROF_TYPE(LaunchKernelWithFlag);
REGISTER_PROF_TYPE(AtomicLaunchKernelWithFlag);
REGISTER_PROF_TYPE(AtomicLaunchKernelWithHandle);
REGISTER_PROF_TYPE(AicpuLaunchTfKernel);
REGISTER_PROF_TYPE(AicpuLaunchCCKernel);
REGISTER_PROF_TYPE(StarsTaskLaunchKernel);
REGISTER_PROF_TYPE(LaunchFFTSPlusTask);
REGISTER_PROF_TYPE(LaunchFFTSPlusTaskNoCopy);
REGISTER_PROF_TYPE(AicpuHostCompute);
REGISTER_PROF_TYPE(LaunchMixKernelWithHandle);
REGISTER_PROF_TYPE(LaunchMixKernelWithFlag);
REGISTER_PROF_TYPE(ExecuteCustomOp);
REGISTER_PROF_NON_LAUNCH_TYPE(AICoreUpdateContext);
REGISTER_PROF_NON_LAUNCH_TYPE(AICpuUpdateContext);
REGISTER_PROF_NON_LAUNCH_TYPE(StaAutoUpdateContext);
REGISTER_PROF_NON_LAUNCH_TYPE(AtomicUpdateContext);

void DumpEventType(const ExecutorEvent et, std::ostream &out_stream) {
  switch (et) {
    case kExecuteStart:
      out_stream << "Start";
      break;
    case kExecuteEnd:
      out_stream << "End";
      break;
    default:
      out_stream << "UNKNOWN(" << static_cast<int64_t>(et) << ")";
      break;
  }
}

void DumpE2eEvent(const int64_t thread_id, const ExecutorEvent et,
                  const std::chrono::time_point<std::chrono::system_clock> timestamp, std::ostream &out_stream) {
  out_stream << std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count() << ' ';
  out_stream << thread_id << ' ';
  out_stream << "[Model]";
  out_stream << ' ';
  out_stream << "[Execute]";
  out_stream << ' ';
  switch (et) {
    case kModelStart:
      out_stream << "Start";
      break;
    case kModelEnd:
      out_stream << "End";
      break;
    default:
      break;
  }
  out_stream << std::endl;
}

void InitProfTensorDesc(const ge::TaskDescInfo &task_desc_info, const size_t index, const uint64_t offset_idx,
                        MsprofTensorInfo *const tensor_info) {
  const auto BuildTensor = [&offset_idx, &tensor_info](
                               const MsprofGeTensorType tensor_type, const std::vector<ge::Format> &format_lst,
                               const std::vector<ge::DataType> &data_type_lst,
                               const std::vector<std::vector<int64_t>> &shape_lst, const size_t tensor_index) {
    tensor_info->tensorData[offset_idx].tensorType = tensor_type;
    // when enum Format is changed, profiling analyze needs to be synchronized
    tensor_info->tensorData[offset_idx].format = static_cast<uint32_t>(format_lst[tensor_index]);
    // when enum DataType is changed, profiling analyze needs to be synchronized
    const ge::DataType data_type = data_type_lst[tensor_index];
    const uint32_t prof_dtype = (static_cast<uint32_t>(data_type) < static_cast<uint32_t>(ge::DT_MAX))
                                    ? static_cast<uint32_t>(data_type)
                                    : static_cast<uint32_t>(ge::DT_UNDEFINED);
    tensor_info->tensorData[offset_idx].dataType = prof_dtype;
    const auto shape_size = shape_lst[tensor_index].size();
    const size_t src_size = std::min(static_cast<size_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), shape_size);
    for (size_t i = 0UL; i < src_size; ++i) {
      tensor_info->tensorData[offset_idx].shape[i] = static_cast<uint32_t>(shape_lst[tensor_index][i]);
    }
    if (shape_size < static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN)) {
      tensor_info->tensorData[offset_idx].shape[shape_size] = 0U;
    }
  };

  const size_t input_size = task_desc_info.input_shape.size();
  if (index < input_size) {
    // when current index is smaller than input_size, build tensor by input tensor
    BuildTensor(MSPROF_GE_TENSOR_TYPE_INPUT, task_desc_info.input_format, task_desc_info.input_data_type,
                task_desc_info.input_shape, index);
  } else {
    // when current index is bigger than input_size, build tensor by output tensor, use index - input_size as
    // index of output tensor
    BuildTensor(MSPROF_GE_TENSOR_TYPE_OUTPUT, task_desc_info.output_format, task_desc_info.output_data_type,
                task_desc_info.output_shape, index - input_size);
  }
}
}  // namespace

void GlobalProfiler::Dump(std::ostream &out_stream, std::vector<std::string> &idx_to_str) const {
  size_t print_size = GetCount();
  std::ofstream fs;
  GE_MAKE_GUARD(close, [&fs]() -> void {
    fs.flush();
    fs.close();
  });
  const auto out_buf = out_stream.rdbuf();
  GE_CHECK_NOTNULL_JUST_RETURN(out_buf);
  if (&out_stream == &std::cout) {
    std::string ascend_work_path;
    GE_CHK_BOOL_EXEC(ge::GetAscendWorkPath(ascend_work_path) == ge::SUCCESS, return, "Failed to get ASCEND_WORK_PATH");
    std::string ge_profiling_path;
    if (!ascend_work_path.empty()) {
      ge_profiling_path = ascend_work_path + "/ge_profiling_" + std::to_string(mmGetPid()) + ".txt";
    } else {
      ge_profiling_path = "ge_profiling_" + std::to_string(mmGetPid()) + ".txt";
    }
    fs.open(ge_profiling_path, std::ios::out | std::ios::app);
    if (fs.is_open()) {
      auto f_buf = fs.rdbuf();
      GE_CHECK_NOTNULL_JUST_RETURN(f_buf);
      (void)out_stream.rdbuf(f_buf);
    }
  }

  out_stream << "ExecutorProfiler version: " << kVersionSingleThread << ", dump start, records num: " << print_size
             << std::endl;
  if (print_size > kProfilingDataCap) {
    out_stream << "Too many records(" << print_size << "), the records after " << kProfilingDataCap
               << " will be dropped" << std::endl;
    print_size = kProfilingDataCap;
  }
  for (size_t i = 0UL; i < print_size; ++i) {
    auto &rec = records_[i];
    if ((rec.event == kModelStart) || (rec.event == kModelEnd)) {
      DumpE2eEvent(rec.thread_id, rec.event, rec.timestamp, out_stream);
      continue;
    }
    // in format: <timestamp> <thread-id> <node-name> <kernel-type> <event-type>
    out_stream << std::chrono::duration_cast<std::chrono::nanoseconds>(rec.timestamp.time_since_epoch()).count() << ' ';
    out_stream << rec.thread_id << ' ';
    out_stream << '[' << idx_to_str[rec.name_idx] << ']';
    out_stream << ' ';
    out_stream << '[' << idx_to_str[rec.type_idx] << ']';
    out_stream << ' ';
    DumpEventType(rec.event, out_stream);
    out_stream << std::endl;
  }
  out_stream << "Profiling dump end" << std::endl;
  (void)out_stream.rdbuf(out_buf);
}

void GlobalProfilingWrapper::Init(const uint64_t enable_flags) {
  SetEnableFlags(enable_flags);
  if (IsEnabled((ProfilingType::kCannHost))) {
    RegisterBuiltInString();
    return;
  }

  if (IsEnabled(ProfilingType::kGeHost) && (global_profiler_ == nullptr)) {
    global_profiler_ = ge::MakeUnique<GlobalProfiler>();
    if (global_profiler_ == nullptr) {
      GELOGE(ge::FAILED, "Init global profiling failed.");
    }
    RegisterBuiltInString();
  }
}

thread_local uint32_t GlobalProfilingWrapper::current_model_id_ = 0U;
thread_local uint32_t GlobalProfilingWrapper::current_step_id_ = 0U;

void GlobalProfilingWrapper::RegisterBuiltInString() {
  if ((is_builtin_string_registered_)) {
    return;
  }
  idx_to_str_.resize(kInitSize);
  idx_to_str_[profiling::kModel] = "Model";
  idx_to_str_[profiling::kExecute] = "Execute";
  idx_to_str_[profiling::kAclCreateTensorDesc] = "AclCreateTensorDesc";
  idx_to_str_[profiling::kAclSetTensorFormat] = "AclSetTensorFormat";
  idx_to_str_[profiling::kAclSetTensorPlacement] = "AclSetTensorPlacement";
  idx_to_str_[profiling::kAclSetTensorShape] = "AclSetTensorShape";
  idx_to_str_[profiling::kAclSetTensorDescName] = "AclSetTensorDescName";
  idx_to_str_[profiling::kAclCreateDataBuffer] = "AclCreateDataBuffer";
  idx_to_str_[profiling::kAclRtMalloc] = "AclRtMalloc";
  idx_to_str_[profiling::kAclRtFree] = "AclRtFree";
  idx_to_str_[profiling::kAclRtMemcpyAsync] = "AclRtMemcpyAsync";
  idx_to_str_[profiling::kAclRtMemcpy] = "AclRtMemcpy";
  idx_to_str_[profiling::kAclRtSynchronizeStream] = "AclRtSynchronizeStream";
  idx_to_str_[profiling::kAclRtStreamWaitEvent] = "AclRtStreamWaitEvent";
  idx_to_str_[profiling::kAclRtSynchronizeDevice] = "AclRtSynchronizeDevice";
  idx_to_str_[profiling::kAclRtDestoryEvent] = "AclRtDestoryEvent";
  idx_to_str_[profiling::kAclRtRecordEvent] = "AclRtRecordEvent";
  idx_to_str_[profiling::kAclRtSynchronizeEvent] = "AclRtSynchronizeEvent";
  idx_to_str_[profiling::kAclRtCreateEventWithFlag] = "AclRtCreateEventWithFlag";
  idx_to_str_[profiling::kAclRtEventWaitStatus] = "AclRtEventWaitStatus";
  idx_to_str_[profiling::kAclRtEventRecordedStatus] = "AclRtEventRecordedStatus";
  idx_to_str_[profiling::kAclRtQueryEventStatus] = "AclRtQueryEventStatus";
  idx_to_str_[profiling::kAclCompileAndExecute] = "AclCompileAndExecute";
  idx_to_str_[profiling::kAclCompileAndExecuteV2] = "AclCompileAndExecuteV2";
  idx_to_str_[profiling::kAclMatchOpModel] = "AclMatchOpModel";
  idx_to_str_[profiling::kAclMatchStaticOpModel] = "AclMatchStaticOpModel";
  idx_to_str_[profiling::kAclMatchDynamicOpModel] = "AclMatchDynamicOpModel";
  idx_to_str_[profiling::kAclExecuteAsync] = "AclExecuteAsync";
  idx_to_str_[profiling::kAclExecuteSync] = "AclExecuteSync";
  idx_to_str_[profiling::kAclLoadSingleOp] = "AclLoadSingleOp";
  idx_to_str_[profiling::kAclBuildOpModel] = "AclBuildOpModel";
  idx_to_str_[profiling::kStaticSingleOpExecute] = "StaticSingleOpExecute";
  idx_to_str_[profiling::kStaticSingleOpKernelLaunch] = "StaticSingleOpKernelLaunch";
  idx_to_str_[profiling::kModelExecute] = "RT1_ModelExecute";
  idx_to_str_[profiling::kInitInferShapeContext] = "RT1_InitInferShapeContext";
  idx_to_str_[profiling::kTiling] = "RT1_Tiling";
  idx_to_str_[profiling::kUpdateShape] = "RT1_UpdateShape";
  idx_to_str_[profiling::kAllocMem] = "RT1_AllocMem";
  idx_to_str_[profiling::kAtomic] = "RT1_Atomic";
  idx_to_str_[profiling::kOpExecute] = "RT1_Atomic";
  idx_to_str_[profiling::kKernelLaunchPrepare] = "RT1_KernelLaunchPrepare";
  idx_to_str_[profiling::kInitHybridExecuteArgs] = "RT1_InitHybridExecuteArgs";
  idx_to_str_[profiling::kKnownGetAddrAndPrefCnt] = "RT1_KnownGetAddrAndPrefCnt";
  idx_to_str_[profiling::kKernelGetAddrAndPrefCnt] = "RT1_KernelGetAddrAndPrefCnt";
  idx_to_str_[profiling::kUpdateAddrAndPrefCnt] = "RT1_UpdateAddrAndPrefCnt";
  idx_to_str_[profiling::kRtEventCreateRecord] = "RT1_RtEventCreateRecord";
  idx_to_str_[profiling::kRtEventSync] = "RT1_RtEventSync";
  idx_to_str_[profiling::kRtEventDestroy] = "RT1_RtEventDestroy";
  idx_to_str_[profiling::kKernelGetAddrAndPrefCnt] = "RT1_KernelGetAddrAndPrefCnt";
  idx_to_str_[profiling::kStaticSingleOpCopyH2D] = "StaticSingleOpCopyH2D";
  idx_to_str_[profiling::kStaticGraphExecute] = "kStaticGraphExecute";
  idx_to_str_[profiling::kDavinciModelCopyH2D] = "kDavinciModelCopyH2D";
  idx_to_str_[profiling::kRtModelExecute] = "kRtModelExecute";
  idx_to_str_[profiling::kUnknownName] = "UNKNOWNNAME";
  str_idx_ = static_cast<uint64_t>(profiling::kProfilingIndexEnd);
  is_builtin_string_registered_ = true;
}

ge::Status GlobalProfilingWrapper::RegisterProfType() const {
  for (const auto &name_to_type : kNamesToProfTypes) {
    if (name_to_type.second < GeProfInfoType::kModelLevelEnd) {
      GE_ASSERT_MSPROF_OK(MsprofRegTypeInfo(MSPROF_REPORT_MODEL_LEVEL, static_cast<uint32_t>(name_to_type.second),
                                            name_to_type.first.c_str()));
    } else if (name_to_type.second < GeProfInfoType::kNodeLevelEnd) {
      GE_ASSERT_MSPROF_OK(MsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, static_cast<uint32_t>(name_to_type.second),
                                            name_to_type.first.c_str()));
    } else {
      GE_ASSERT_MSPROF_OK(MsprofRegTypeInfo(MSPROF_REPORT_ACL_LEVEL, static_cast<uint32_t>(name_to_type.second),
                                            name_to_type.first.c_str()));
    }
  }

  return ge::SUCCESS;
}

uint64_t GlobalProfilingWrapper::RegisterString(const std::string &name) {
  const std::lock_guard<std::mutex> lk(register_mutex_);
  RegisterBuiltInString();
  const auto iter = std::find(idx_to_str_.begin(), idx_to_str_.end(), name);
  if (iter == idx_to_str_.end()) {
    idx_to_str_[str_idx_] = name;
    ++str_idx_;
    if (str_idx_ >= idx_to_str_.size()) {
      idx_to_str_.resize(idx_to_str_.size() * kDouble);
    }
    if (RegisterExtendProfType(name, static_cast<uint32_t>(str_idx_) - 1U) != ge::SUCCESS) {
      return static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    }
    return str_idx_ - 1UL;
  } else {
    return static_cast<uint64_t>(iter - idx_to_str_.begin());
  }
}

ge::Status GlobalProfilingWrapper::RegisterExtendProfType(const std::string &name, const uint32_t idx) const {
  uint32_t prof_idx = 0U;
  GE_ASSERT_SUCCESS(ge::AddOverflow(idx, static_cast<uint32_t>(GeProfInfoType::kNodeLevelEnd), prof_idx));
  GE_ASSERT_MSPROF_OK(MsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, prof_idx, name.c_str()));
  return ge::SUCCESS;
}

GlobalProfilingWrapper::GlobalProfilingWrapper() {
  ge::diagnoseSwitch::MutableProfiling().RegisterHandler(this, {this, GlobalProfilingWrapper::OnGlobalProfilingSwitch});
}

void GlobalProfilingWrapper::OnGlobalProfilingSwitch(void *ins, uint64_t enable_flags) {
  if (ins == nullptr) {
    return;
  }
  GELOGI("enable flags = %lu", enable_flags);
  const auto global_prof_wrapper = static_cast<GlobalProfilingWrapper *>(ins);
  if (enable_flags != 0ULL) {
    global_prof_wrapper->Init(enable_flags);
    return;
  }
  if (enable_flags == 0ULL) {
    global_prof_wrapper->DumpAndFree(std::cout);
  }
}

ge::Status GlobalProfilingWrapper::ReportEvent(const uint64_t item_id, const uint32_t request_id,
                                               const GeProfInfoType type, MsprofEvent &prof_single_event) {
  prof_single_event.level = MSPROF_REPORT_MODEL_LEVEL;
  prof_single_event.type = static_cast<uint32_t>(type);
  prof_single_event.requestId = request_id;
  prof_single_event.itemId = item_id;
  prof_single_event.timeStamp = MsprofSysCycleTime();
  prof_single_event.threadId = static_cast<uint32_t>(mmGetTid());
  GE_ASSERT_MSPROF_OK(MsprofReportEvent(true, &prof_single_event));
  return ge::SUCCESS;
}

ge::Status GlobalProfilingWrapper::ReportDefaultEventForRt2MultiThread(const GeProfInfoType type, const uint32_t thread_id,
                                                      MsprofEvent &prof_single_event) const {
  prof_single_event.level = MSPROF_REPORT_MODEL_LEVEL;
  prof_single_event.type = static_cast<uint32_t>(type);
  prof_single_event.requestId = current_step_id_;
  prof_single_event.itemId = current_model_id_;
  prof_single_event.timeStamp = MsprofSysCycleTime();
  prof_single_event.threadId = thread_id;
  GE_ASSERT_MSPROF_OK(MsprofReportEvent(true, &prof_single_event));
  return ge::SUCCESS;
}

void GlobalProfilingWrapper::SetModelIdStepId(const uint32_t model_id, const uint32_t step_id) {
  current_model_id_ = model_id;
  current_step_id_ = step_id;
}

ge::Status GlobalProfilingWrapper::ReportApiInfo(const uint64_t begin_time, const uint64_t end_time,
                                                 const uint64_t item_id, const uint32_t api_type) {
  MsprofApi api_info{};
  BuildApiInfo({begin_time, end_time}, api_type, item_id, api_info);
  GE_ASSERT_MSPROF_OK(MsprofReportApi(true, &api_info));
  return ge::SUCCESS;
}

ge::Status GlobalProfilingWrapper::ReportApiInfoModelLevel(const uint64_t begin_time, const uint64_t end_time,
                                                           const uint64_t item_id, const uint32_t api_type) {
  MsprofApi api_info{};
  BuildApiInfo({begin_time, end_time}, api_type, item_id, api_info);
  api_info.level = MSPROF_REPORT_MODEL_LEVEL;
  GE_ASSERT_MSPROF_OK(MsprofReportApi(true, &api_info));
  return ge::SUCCESS;
}

void GlobalProfilingWrapper::BuildNodeBasicInfo(const ge::OpDescPtr &op_desc, const uint32_t block_dim,
                                                const std::pair<uint64_t, uint64_t> &op_name_and_type_hash,
                                                const uint32_t task_type, MsprofCompactInfo &node_basic_info) {
  auto &prof_node_basic_info = node_basic_info.data.nodeBasicInfo;
  prof_node_basic_info.opName = op_name_and_type_hash.first;
  prof_node_basic_info.opType = op_name_and_type_hash.second;
  prof_node_basic_info.taskType = task_type;
  prof_node_basic_info.blockDim = block_dim;

  uint32_t op_impl_mode = 0U;
  (void)ge::AttrUtils::GetInt(op_desc, kOpImplModeEnum, op_impl_mode);
  prof_node_basic_info.opFlag = (op_impl_mode == kEnableHf32) ? kOpImplHf32Mode : 0U;
}

void GlobalProfilingWrapper::BuildCompactInfo(const uint64_t prof_time, MsprofCompactInfo &node_basic_info) {
  node_basic_info.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
  node_basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  node_basic_info.timeStamp = prof_time;
  thread_local const auto tid = mmGetTid();
  node_basic_info.threadId = static_cast<uint32_t>(tid);
}

void GlobalProfilingWrapper::BuildApiInfo(const std::pair<uint64_t, uint64_t> &prof_time, const uint32_t api_type,
                                          const uint64_t item_id, MsprofApi &api) {
  api.itemId = item_id;
  api.beginTime = prof_time.first;
  api.endTime = prof_time.second;
  api.type = api_type;
  api.level = MSPROF_REPORT_NODE_LEVEL;
  thread_local const auto tid = mmGetTid();
  api.threadId = static_cast<uint32_t>(tid);
}

void GlobalProfilingWrapper::BuildContextIdInfo(const uint64_t prof_time, const std::vector<uint32_t> &context_ids,
                                                const std::string &op_name, std::vector<ContextIdInfoWrapper> &infos) {
  const size_t index = context_ids.size() / kMaxContextIdNum;
  const auto op_name_hash = MsprofGetHashId(op_name.c_str(), op_name.length());
  for (size_t i = 0UL; i < index; ++i) {
    ContextIdInfoWrapper info{};
    info.op_name = op_name;
    reinterpret_cast<MsprofContextIdInfo *>(info.context_id_info.data)->opName = op_name_hash;
    BuildSingleContextIdInfo(prof_time, context_ids, i, kMaxContextIdNum, info.context_id_info);
    (void)infos.emplace_back(info);
  }

  const size_t remain_index = context_ids.size() % kMaxContextIdNum;
  ContextIdInfoWrapper info{};
  info.op_name = op_name;
  reinterpret_cast<MsprofContextIdInfo *>(info.context_id_info.data)->opName = op_name_hash;
  BuildSingleContextIdInfo(prof_time, context_ids, index, remain_index, info.context_id_info);
  (void)infos.emplace_back(info);
}

void GlobalProfilingWrapper::BuildSingleContextIdInfo(const uint64_t prof_time, const vector<uint32_t> &context_ids,
                                                      const size_t index, const size_t context_id_num,
                                                      MsprofAdditionalInfo &info) {
  info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.timeStamp = prof_time;
  thread_local const auto tid = mmGetTid();
  info.threadId = static_cast<uint32_t>(tid);
  info.dataLen = static_cast<uint32_t>(context_ids.size() * sizeof(uint32_t));
  auto context_id_info = reinterpret_cast<MsprofContextIdInfo *>(info.data);
  context_id_info->ctxIdNum = static_cast<uint32_t>(context_id_num);
  for (size_t j = 0UL; j < context_id_num; ++j) {
    context_id_info->ctxIds[j] = context_ids[index * kMaxContextIdNum + j];
  }
}

ge::Status GlobalProfilingWrapper::RecordAndReportMallocTaskMemoryInfo(const void *const addr, const size_t size,
                                                                       const std::string &model_name) {
  DeviceMemoryRecorder::SetRecorder(addr, static_cast<int64_t>(size));
  return GlobalProfilingWrapper::ReportTaskMemoryInfo(model_name);
}

ge::Status GlobalProfilingWrapper::RecordAndReportFreeTaskMemoryInfo(const void *const addr, const size_t size,
                                                                     const std::string &model_name) {
  const int64_t free_size = static_cast<int64_t>(size) * (-1);
  DeviceMemoryRecorder::SetRecorder(addr, free_size);
  return GlobalProfilingWrapper::ReportTaskMemoryInfo(model_name);
}

ge::Status GlobalProfilingWrapper::ReportTaskMemoryInfo(const std::string &model_name) {
  if (!gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kMemory)) {
    return ge::SUCCESS;
  }
  if (DeviceMemoryRecorder::IsRecorderEmpty()) {
    return ge::SUCCESS;
  }
  MsprofAdditionalInfo task_memory_info;
  task_memory_info.threadId = static_cast<uint32_t>(mmGetTid());
  task_memory_info.type = MSPROF_REPORT_NODE_TASK_MEMORY_TYPE;
  task_memory_info.level = MSPROF_REPORT_NODE_LEVEL;
  task_memory_info.dataLen = static_cast<uint32_t>(sizeof(MsprofMemoryInfo));
  task_memory_info.timeStamp = MsprofSysCycleTime();
  auto memory_info_data = reinterpret_cast<MsprofMemoryInfo *>(task_memory_info.data);
  memory_info_data->nodeId = MsprofGetHashId(model_name.c_str(), model_name.size());
  int32_t device_id = 0;
  (void)rtGetDevice(&device_id);
  memory_info_data->deviceId = static_cast<uint32_t>(device_id);
  memory_info_data->deviceType = 0U;
  while (!DeviceMemoryRecorder::IsRecorderEmpty()) {
    const MemoryRecorder record_memory_info = DeviceMemoryRecorder::GetRecorder();
    memory_info_data->size = record_memory_info.size;
    memory_info_data->addr = record_memory_info.addr;
    memory_info_data->totalAllocateMemory = record_memory_info.total_allocate_memory;
    memory_info_data->totalReserveMemory = record_memory_info.total_reserve_memory;
    GELOGD(
        "[ReportTaskMemoryInfo]Report memory info: node_id: %llu, "
        "addr: %llu, size: %lld, total allocate size: %llu, total reserve size: %lld"
        "time stamp: %llu",
        memory_info_data->nodeId, memory_info_data->addr, memory_info_data->size, memory_info_data->totalAllocateMemory,
        memory_info_data->totalReserveMemory, task_memory_info.timeStamp);
    GE_ASSERT_MSPROF_OK(
        MsprofReportAdditionalInfo(true, &task_memory_info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  }
  return ge::SUCCESS;
}

ge::Status GlobalProfilingWrapper::ReportTensorInfo(const uint32_t tid, const bool is_aging,
                                                    const ge::TaskDescInfo &task_desc_info) {
  const size_t total_num = task_desc_info.input_shape.size() + task_desc_info.output_shape.size();
  GELOGD("[Cann Profiling]tensor size is %zu, is_aging %u", total_num, static_cast<uint32_t>(is_aging));
  const size_t index = total_num / static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  for (size_t j = 0UL; j < index; ++j) {
    MsprofAdditionalInfo tensor_info{};
    BuildSingleProfTensorInfo(tid, task_desc_info, j, static_cast<uint32_t>(MSPROF_GE_TENSOR_DATA_NUM), tensor_info);
    GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(static_cast<uint32_t>(is_aging), &tensor_info,
                                                   static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  }

  const size_t remain_index = total_num % static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  if (remain_index == 0UL) {
    return ge::SUCCESS;
  }
  MsprofAdditionalInfo tensor_info{};
  BuildSingleProfTensorInfo(tid, task_desc_info, index, static_cast<uint32_t>(remain_index), tensor_info);
  GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(static_cast<uint32_t>(is_aging), &tensor_info,
                                                 static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  return ge::SUCCESS;
}

void GlobalProfilingWrapper::BuildSingleProfTensorInfo(const uint32_t tid, const ge::TaskDescInfo &task_desc_info,
                                                       const size_t index, const uint32_t tensor_num,
                                                       MsprofAdditionalInfo &tensor_info) {
  tensor_info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensor_info.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
  tensor_info.timeStamp = task_desc_info.prof_time;
  tensor_info.threadId = tid;
  tensor_info.dataLen = kTensorInfoBytesWithCap + (static_cast<uint32_t>(tensor_num) - 1U) * kTensorInfoBytes;
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
  const auto op_name_hash = MsprofGetHashId(task_desc_info.op_name.c_str(), task_desc_info.op_name.length());
  prof_tensor_data->opName = op_name_hash;
  prof_tensor_data->tensorNum = tensor_num;
  for (size_t k = 0UL; k < static_cast<size_t>(tensor_num); ++k) {
    const size_t tensor_index = (index * static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM)) + k;
    InitProfTensorDesc(task_desc_info, tensor_index, k, prof_tensor_data);
  }
}

ge::Status GlobalProfilingWrapper::ReportGraphIdMap(const uint64_t prof_time, const uint32_t tid,
                                                    const std::pair<uint32_t, uint32_t> graph_id_and_model_id,
                                                    const bool is_aging, const size_t model_name) {
  MsprofAdditionalInfo graph_id_info{};
  graph_id_info.level = MSPROF_REPORT_MODEL_LEVEL;
  graph_id_info.type = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE;
  graph_id_info.timeStamp = prof_time;
  graph_id_info.threadId = tid;
  graph_id_info.dataLen = kIdOffset;
  reinterpret_cast<MsprofGraphIdInfo *>(graph_id_info.data)->graphId = graph_id_and_model_id.first;
  reinterpret_cast<MsprofGraphIdInfo *>(graph_id_info.data)->modelName = model_name;
  reinterpret_cast<MsprofGraphIdInfo *>(graph_id_info.data)->modelId = graph_id_and_model_id.second;
  GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(static_cast<uint32_t>(is_aging), &graph_id_info,
                                                 static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  return ge::SUCCESS;
}

ge::Status GlobalProfilingWrapper::ProfileStepTrace(const uint64_t step_id, const uint32_t model_id,
                                                    const uint16_t tag_id, const rtStream_t stream) {
  {
    const auto subscribe_graph_id = ge::ProfilingProperties::Instance().GetSubscribeGraphId();
    const bool is_this_model_unsubscribed = (subscribe_graph_id.find(model_id) == subscribe_graph_id.end());
    // 开关关闭 且 （订阅数量为0 或 （订阅数量不为0 且 该模型不在订阅范围内）） -》 return
    if ((GlobalProfilingWrapper::GetInstance()->GetEnableFlags() == 0UL) &&
        ((!ge::ProfilingProperties::Instance().ProfilingSubscribeOn()) || is_this_model_unsubscribed)) {
      GELOGD("Profiling is not turned on, no need to profile step info.");
      return ge::SUCCESS;
    }
  }
  GELOGD("Profiling Step Info TraceTask execute async start, step_id = %lu, model_id = %u, tag_id = %u", step_id,
         model_id, static_cast<uint32_t>(tag_id));
  const auto begin_time = MsprofSysCycleTime();
  const rtError_t rt_ret = rtProfilerTraceEx(step_id, static_cast<uint64_t>(model_id), tag_id, stream);
  const auto end_time = MsprofSysCycleTime();
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(ge::RT_FAILED, "[Call][rtProfilerTraceEx]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtProfilerTraceEx failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return ReportApiInfo(begin_time, end_time, static_cast<uint64_t>(tag_id),
                       static_cast<uint32_t>(GeProfInfoType::kStepInfo));
}

void GlobalProfilingWrapper::BuildProfFusionInfoBase(const ProfFusionMemSize &mem_size, const size_t fusion_op_num,
                                                     const size_t op_name, ProfFusionOpInfo *prof_fusion_data) {
  prof_fusion_data->opName = op_name;
  prof_fusion_data->fusionOpNum = static_cast<uint32_t>(fusion_op_num);
  prof_fusion_data->inputMemsize = mem_size.input_mem_size;
  prof_fusion_data->outputMemsize = mem_size.output_mem_size;
  prof_fusion_data->workspaceMemSize = mem_size.workspace_mem_size;
  prof_fusion_data->weightMemSize = mem_size.weight_mem_size;
  prof_fusion_data->totalMemSize =
      mem_size.weight_mem_size + mem_size.workspace_mem_size + mem_size.output_mem_size + mem_size.input_mem_size;
}

void GlobalProfilingWrapper::BuildFusionOpInfo(const ProfFusionMemSize &mem_size,
                                               const std::vector<std::string> &origin_op_names, const size_t op_name,
                                               std::vector<MsprofAdditionalInfo> &infos) {
  thread_local const auto tid = mmGetTid();
  const auto prof_time = MsprofSysCycleTime();
  const size_t slice_index = origin_op_names.size() / static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM);
  for (size_t k = 0UL; k < slice_index; ++k) {
    MsprofAdditionalInfo info{};
    info.level = MSPROF_REPORT_NODE_LEVEL;
    info.type = MSPROF_REPORT_NODE_FUSION_OP_INFO_TYPE;
    info.timeStamp = prof_time;
    info.threadId = static_cast<uint32_t>(tid);
    info.dataLen = kFusionOpInfoCap + static_cast<uint32_t>(MSPROF_GE_FUSION_OP_NUM) * kHashOffset;
    BuildProfFusionInfoBase(mem_size, static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM), op_name,
                            reinterpret_cast<ProfFusionOpInfo *>(info.data));
    for (size_t j = 0UL; j < static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM); ++j) {
      const size_t origin_op_index = (k * static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM)) + j;
      const auto origin_op_name =
          MsprofGetHashId(origin_op_names[origin_op_index].c_str(), origin_op_names[origin_op_index].length());
      reinterpret_cast<ProfFusionOpInfo *>(info.data)->fusionOpId[j] = origin_op_name;
    }
    (void)infos.emplace_back(info);
  }

  const size_t remain_index = origin_op_names.size() % static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM);
  if (remain_index == 0UL) {
    return;
  }
  MsprofAdditionalInfo info{};
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.type = MSPROF_REPORT_NODE_FUSION_OP_INFO_TYPE;
  info.timeStamp = prof_time;
  info.threadId = static_cast<uint32_t>(tid);
  info.dataLen = kFusionOpInfoCap + static_cast<uint32_t>(remain_index) * kHashOffset;
  BuildProfFusionInfoBase(mem_size, remain_index, op_name, reinterpret_cast<ProfFusionOpInfo *>(info.data));
  for (size_t k = 0UL; k < remain_index; ++k) {
    const size_t origin_op_index = static_cast<size_t>(slice_index * static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM) + k);
    const auto origin_op_name =
        MsprofGetHashId(origin_op_names[origin_op_index].c_str(), origin_op_names[origin_op_index].length());
    reinterpret_cast<ProfFusionOpInfo *>(info.data)->fusionOpId[k] = origin_op_name;
  }
  (void)infos.emplace_back(info);
}

static ge::Status ReportOneLogicStreamInfo(const std::pair<uint32_t, std::set<uint32_t>> &ids_pair,
                                           const uint64_t timestamp, const uint32_t tid, const uint16_t aging_flag) {
  const size_t info_count = (ids_pair.second.size() + static_cast<size_t>(MSPROF_PHYSIC_STREAM_ID_MAX_NUM) - 1UL) /
                            static_cast<size_t>(MSPROF_PHYSIC_STREAM_ID_MAX_NUM);
  std::vector<MsprofAdditionalInfo> logic_stream_infos{info_count, MsprofAdditionalInfo{}};
  std::vector<uint32_t> physic_stream_id;

  for (const auto elem : ids_pair.second) {
    physic_stream_id.push_back(elem);
  }
  for (size_t index = 0UL; index < info_count; ++index) {
    auto &logic_stream_info = logic_stream_infos[index];
    logic_stream_info.threadId = tid;
    logic_stream_info.level = MSPROF_REPORT_MODEL_LEVEL;
    logic_stream_info.type = MSPROF_REPORT_MODEL_LOGIC_STREAM_TYPE;
    logic_stream_info.timeStamp = timestamp;
    auto prof_logic_stream_info = reinterpret_cast<MsprofLogicStreamInfo *>(logic_stream_info.data);
    prof_logic_stream_info->logicStreamId = ids_pair.first;
    size_t slice_size = static_cast<size_t>(MSPROF_PHYSIC_STREAM_ID_MAX_NUM);
    if ((index == (info_count - 1UL)) && (info_count * slice_size > ids_pair.second.size())) {
      slice_size = ids_pair.second.size() % static_cast<size_t>(MSPROF_PHYSIC_STREAM_ID_MAX_NUM);
    }
    prof_logic_stream_info->physicStreamNum = static_cast<uint32_t>(slice_size);
    for (size_t i = 0UL; i < slice_size; ++i) {
      prof_logic_stream_info->physicStreamId[i] =
          physic_stream_id[index * static_cast<size_t>(MSPROF_PHYSIC_STREAM_ID_MAX_NUM) + i];
    }
    GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(aging_flag, &logic_stream_info,
                                                   static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  }
  return ge::SUCCESS;
};

ge::Status GlobalProfilingWrapper::ReportLogicStreamInfo(
    const uint64_t timestamp, const uint32_t tid,
    const std::unordered_map<uint32_t, std::set<uint32_t>> &logic_stream_ids_to_physic_stream_ids,
    const uint16_t is_aging) {
  for (const auto &ids_pair : logic_stream_ids_to_physic_stream_ids) {
    GE_ASSERT_SUCCESS(ReportOneLogicStreamInfo(ids_pair, timestamp, tid, is_aging));
  }
  return ge::SUCCESS;
}

// if op_desc is not null, mem_size is the op mem size, otherwise mem_size is the total mem size of the graph
ge::Status GlobalProfilingWrapper::ReportStaticOpMemInfo(const ge::ComputeGraphPtr &graph, const ge::OpDescPtr &op_desc,
                                                         const uint64_t mem_size, const uint64_t life_start,
                                                         const uint64_t life_end) {
  constexpr uint32_t aging = 0U;
  MsprofAdditionalInfo info{};
  static_assert(sizeof(MsprofStaticOpMem) <= MSPROF_ADDTIONAL_INFO_DATA_LENGTH,
                "size of MsprofStaticOpMem is bigger than MSPROF_ADDTIONAL_INFO_DATA_LENGTH");

  // don't report static op mem info in single op scene
  bool is_single_op = false;
  (void)ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op);
  GE_IF_BOOL_EXEC(is_single_op, return ge::SUCCESS);

  MsprofStaticOpMem *mem_info = reinterpret_cast<MsprofStaticOpMem *>(info.data);
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.type = MSPROF_REPORT_NODE_STATIC_OP_MEM_TYPE;
  info.timeStamp = MsprofSysCycleTime();
  info.threadId = static_cast<uint32_t>(mmGetTid());
  info.dataLen = sizeof(MsprofStaticOpMem);

  auto GetHashID = [](const std::string &str) -> uint64_t { return MsprofGetHashId(str.c_str(), str.size()); };

  const std::string op_name = (op_desc == nullptr) ? "" : op_desc->GetName();
  const bool is_dyn_op = (graph->GetParentNodeBarePtr() == nullptr);
  const std::string dyn_op_name = (is_dyn_op) ? "" : graph->GetParentNodeBarePtr()->GetName();

  mem_info->size = mem_size;
  mem_info->opName = (op_desc == nullptr) ? 0U : GetHashID(op_name);
  mem_info->lifeStart = life_start;
  mem_info->lifeEnd = life_end;
  mem_info->dynOpName = (is_dyn_op ? 0U : GetHashID(dyn_op_name));
  mem_info->graphId = graph->GetGraphID();

  GELOGD("graph: %s, op: %s %lu, size: %lu, life_start: %lu, life_end: %lu, dynOpName: %s %lu",
         graph->GetName().c_str(), op_name.c_str(), mem_info->opName, mem_info->size, life_start, life_end,
         dyn_op_name.c_str(), mem_info->dynOpName);

  GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(aging, &info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  return ge::SUCCESS;
}

uint32_t GlobalProfilingWrapper::GetProfModelId() const {
  return model_id_generator_.load();
}

void GlobalProfilingWrapper::IncProfModelId() {
  ++model_id_generator_;
}

void ProfilerRegistry::SaveRegistryType(const std::string &type, const bool launch_flag) {
  const std::lock_guard<std::mutex> lk(mutex_);
  if (launch_flag) {
    (void)register_prof_launch_type_.emplace_back(type);
  } else {
    (void)register_prof_non_launch_type_.emplace_back(type);
  }
}

ProfilerRegistry &ProfilerRegistry::GetInstance() {
  static ProfilerRegistry prof_registry;
  return prof_registry;
}

bool ProfilerRegistry::IsProfLaunchType(const std::string &kernel_type, const bool launch_flag) {
  const std::lock_guard<std::mutex> lk(mutex_);
  if (launch_flag) {
    return (std::find(register_prof_launch_type_.cbegin(), register_prof_launch_type_.cend(), kernel_type) !=
            register_prof_launch_type_.cend());
  } else {
    return (std::find(register_prof_non_launch_type_.cbegin(), register_prof_non_launch_type_.cend(), kernel_type) !=
            register_prof_non_launch_type_.cend());
  }
}

bool ProfilerRegistry::IsProfDavinciModelExecuteType(const std::string &kernel_type) const {
  return kernel_type == std::string("DavinciModelExecute");
}
}  // namespace gert
