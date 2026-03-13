/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/davinci_model.h"

#include <numeric>
#include <regex>
#include <sstream>
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/omg_util/omg_util.h"
#include "common/profiling/profiling_manager.h"
#include "common/utils/executor_utils.h"
#include "common/runtime_api_wrapper.h"
#include "framework/common/runtime_tensor_desc.h"
#include "rt_error_codes.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/ge_context.h"
#include "base/err_mgr.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/manager/util/hcom_ome_util.h"
#include "graph/model_serialize.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/checker.h"
#include "common/dump/kernel_tracing_utils.h"
#include "runtime/subscriber/global_profiler.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "mmpa/mmpa_api.h"
#include "common/error_tracking/error_tracking.h"
#include "common/model/executor.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "common/platform_info_util/platform_info_util.h"
#include "hcom/hcom_topo_info.h"
#include "common/memory/tensor_trans_utils.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"
#include "platform/soc_spec.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "graph/load/model_manager/kernel/kernel_register_info_builder.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"

namespace ge {
namespace {
constexpr uint32_t kDataIndex = 0U;
constexpr uint32_t kTrueBranchStreamCount = 1U;
constexpr uint32_t kGetDynamicDimsCount = 1U;
constexpr uint32_t kAddrSize = static_cast<uint32_t>(sizeof(uint64_t));
constexpr int32_t kDecimalRadix = 10;
constexpr int64_t kDataMemAlignSizeCompare = 64;
constexpr uint32_t kModelL1FusionOpMByteSize = 2097152U;   // 2 * 1024 * 1024
constexpr uint32_t kModelFlagOfL1Fusion = 0U;
constexpr int32_t kInvalidStream = -1;
constexpr int32_t kDefaultTaskNum = 1;
constexpr int32_t kDefaultEventWaitTaskNum = 2;
constexpr int32_t kNoTask = 0;
constexpr int64_t kInvalidOpIndex = -1;
constexpr int32_t kSinkModelEndOfSequence = 0x0704000A;
constexpr int32_t kSinkModelEndOfSequenceNew = 507005;
constexpr int32_t kSinkModelAbortNormal = 0x0704000E;
constexpr int32_t kSinkModelAbortNormalNew = 507024;
constexpr uint32_t kPlaceHostData = 0U;
constexpr uint32_t kDefaultThreadNum = 16U;
constexpr size_t kMemAlignment = 64U;
constexpr uint32_t kManualThreadMode = 0U;
constexpr uint32_t kInValidThreadMode = 3U;
constexpr int32_t kDynamicOutInfoMinSize = 2; // 0 is output index, 1 is branch index
constexpr int32_t kBase = 10;
constexpr int32_t kDefaultTimeout = -1;
const std::string kGetDynamicDimsName = "ascend_mbatch_get_dynamic_dims_node";
const std::string kAddrRefreshOpPath = "/UpdateModelParam_dav";
const std::string kMultiBatchNodePostfix = "_ascend_mbatch_batch_";
const std::string kAddrRefreshOpBinId = "UpdateModelParam";
const std::string kAicpuCustLoadPlatformInfo = "LoadCustPlatform";
const std::string kMallocPurpose = "davinci_model_load";
const std::string kPlatformSoPathSuffix= "/lib64/device/lib64/libkernel_load_platform.so";
constexpr const char_t *kUndefinedOptype = "Undefined";
constexpr const char_t *kStaticModelAddrFixed = "ge.exec.static_model_addr_fixed";
constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";

constexpr const char* K_INPUT = "Input";
constexpr const char* K_OUTPUT = "Output";
constexpr int64_t kOverflowUserSize = INT64_MAX - kDataMemAlignSizeCompare;
constexpr uint32_t kTilingSinkBlockDim = 0xFFFFFFFF;
constexpr uint32_t kUpdatePolicyAllOneTime = 4U;
constexpr uint32_t kModelLoadStage = 0;
constexpr uint8_t kConstructInputLogicalAllcationLoop = 2; // frozen input index loop
const std::set<std::string> kIoNodeTypes { QUEUE_DATA, NETOUTPUT };

const std::set<std::string> hccl_op_types({ge::HCOMBROADCAST, ge::HCOMALLGATHER, ge::HCOMALLREDUCE,
                                           ge::HCOMREDUCESCATTER, ge::HCOMREDUCE, ge::HCOMALLTOALLV,
                                           ge::HCOMGATHERALLTOALLV, ge::HCOMALLTOALLVC, ge::HCOMALLTOALL});

constexpr const char *kModelProfStageStr[kMdlProfStageNameEnd + 1] = {
    "InitModelMem",
    "InitIoNodes",
    "TransAllVarData",
    "InitNodes",
    "DoTaskSink",
    "CopyModelData",
    "rtModelExecute",
    "CopyOutputData",
    "unknown"};

const std::string kPurpose("feature map,used for op input and output");
const std::string kCoreTypeMix = "MIX";
const std::string kAttrTaskRatio = "_task_ratio";
const std::string kAttrIsAiv = "_mix_is_aiv";
const std::string kAttrIsFFTSTask = "_is_fftsplus_task";  // fftsplus task

const std::string kCustPlatformInfoPurpose("cust platform info memory, used for tiling sink");
const std::string kCustPlatformInfoMemoryKey("cust_platform_info_memory");
const uint64_t kStopOnFailure = 1U;

const std::string kPlatformInfoPurpose("platform info memory, used for tiling sink");
const std::string kPlatformInfoMemoryKey("platform_info_memory");

const char *GetModelProfStageStr(ModelProfStage stage_name) {
  if (stage_name > kMdlProfStageNameEnd) {
    stage_name = kMdlProfStageNameEnd;
  }
  return kModelProfStageStr[stage_name];
}
const std::map<rtFftsPlusContextType_t, MsprofGeTaskType> ctx_type_to_task_types{
    {RT_CTX_TYPE_AICORE, MSPROF_GE_TASK_TYPE_AI_CORE},
    {RT_CTX_TYPE_AIV, MSPROF_GE_TASK_TYPE_AIV},
    {RT_CTX_TYPE_MIX_AIC, MSPROF_GE_TASK_TYPE_MIX_AIC},
    {RT_CTX_TYPE_MIX_AIV, MSPROF_GE_TASK_TYPE_MIX_AIV},
    {RT_CTX_TYPE_AICPU, MSPROF_GE_TASK_TYPE_AI_CPU},
    {RT_CTX_TYPE_WRITEBACK_DATA, MSPROF_GE_TASK_TYPE_WRITE_BACK},
    {RT_CTX_TYPE_INVALIDATE_DATA, MSPROF_GE_TASK_TYPE_INVALID},
    {RT_CTX_TYPE_DSA, MSPROF_GE_TASK_TYPE_DSA}};

const std::map<ModelTaskType, MsprofGeTaskType> model_task_type_to_task_types{
    {ModelTaskType::MODEL_TASK_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_VECTOR_KERNEL, MSPROF_GE_TASK_TYPE_AIV},
    {ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL, MSPROF_GE_TASK_TYPE_AIV},
    {ModelTaskType::MODEL_TASK_KERNEL_EX, MSPROF_GE_TASK_TYPE_AI_CPU},
    {ModelTaskType::MODEL_TASK_DSA, MSPROF_GE_TASK_TYPE_DSA},
    {ModelTaskType::MODEL_TASK_HCCL, MSPROF_GE_TASK_TYPE_HCCL},
    {ModelTaskType::MODEL_TASK_ALL_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_SUPER_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_FUSION_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_KERNEL_LAUNCH_V2, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_CUSTOM_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE}};

inline bool IsNoTaskAndDumpNeeded(const OpDescPtr &op_desc) {
  bool save_dump_info = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NO_TASK_AND_DUMP_NEEDED, save_dump_info);
  return save_dump_info;
}

inline rtMemcpyKind_t GetRtMemcpyKindByPlacement(const uint32_t placement, const bool out2model) {
  if (out2model) {
    return (placement == kPlaceHostData) ? RT_MEMCPY_HOST_TO_DEVICE : RT_MEMCPY_DEVICE_TO_DEVICE;
  }
  return (placement == kPlaceHostData) ? RT_MEMCPY_DEVICE_TO_HOST : RT_MEMCPY_DEVICE_TO_DEVICE;
}

bool IsEventWaitNode(const ge::OpDescPtr &op_desc) {
  const auto &node_type = op_desc->GetTypePtr();
  return (strcmp(node_type, "Recv") == 0) || (strcmp(node_type, "RecvMem") == 0);
}

bool IsInputOfNetoutputCanZeroCopy(const NodePtr &node, const int32_t anchor_idx) {
  if ((node->GetInDataAnchor(anchor_idx) == nullptr) ||
      (node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor() == nullptr) ||
      (node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) ||
      (node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr)) {
    GELOGE(PARAM_INVALID, "Peer node of net-output %s input %d is invalid", node->GetName().c_str(), anchor_idx);
    return false;
  }

  const auto src_node = node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetOwnerNode();
  const int32_t src_output_index = node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetIdx();
  const auto output_desc = src_node->GetOpDesc()->GetOutputDescPtr(static_cast<uint32_t>(src_output_index));

  bool is_zero_copy_block = false;
  const bool determinate =
      (output_desc != nullptr) && AttrUtils::GetBool(output_desc, ATTR_IS_ZERO_COPY_BLOCK, is_zero_copy_block);

  GELOGI("Net-output %s input %d from %s output %d can zero copy: %s", node->GetName().c_str(), anchor_idx,
         src_node->GetName().c_str(), src_output_index,
         (determinate ? (is_zero_copy_block ? "true" : "false") : "indeterminate"));

  return is_zero_copy_block;
}

void InitMemoryInfo(const OpDescPtr &op_desc, uint64_t &input_mem_size, uint64_t &output_mem_size,
                    uint64_t &workspace_mem_size, uint64_t &weight_mem_size) {
  const auto input_size = ModelUtils::GetInputSize(op_desc);
  const auto output_size = ModelUtils::GetOutputSize(op_desc);
  const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
  const auto weight_size = ModelUtils::GetWeightSize(op_desc);
  input_mem_size = static_cast<uint64_t>(std::accumulate(input_size.begin(), input_size.end(), 0));
  output_mem_size = static_cast<uint64_t>(std::accumulate(output_size.begin(), output_size.end(), 0));
  workspace_mem_size = static_cast<uint64_t>(std::accumulate(workspace_size.begin(), workspace_size.end(), 0));
  weight_mem_size = static_cast<uint64_t>(std::accumulate(weight_size.begin(), weight_size.end(), 0));
}

MsprofGeTaskType GetProfilingTaskType(const OpDescPtr &op_desc, const domi::TaskDef &task_def, std::string &core_type) {
  if (AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type) && core_type == kTaskTypeAiv) {
    return MSPROF_GE_TASK_TYPE_AIV;
  }

  bool is_fftsplus_task = false;
  if ((AttrUtils::GetBool(op_desc, kAttrIsFFTSTask, is_fftsplus_task) && is_fftsplus_task)) {
    bool is_mix_aiv = false;
    (void)AttrUtils::GetBool(op_desc, kAttrIsAiv, is_mix_aiv);
    return is_mix_aiv ? MSPROF_GE_TASK_TYPE_MIX_AIV : MSPROF_GE_TASK_TYPE_MIX_AIC;
  }

  const auto model_task_type = static_cast<ModelTaskType>(task_def.type());
  const auto it = model_task_type_to_task_types.find(model_task_type);
  if (it == model_task_type_to_task_types.end()) {
    GELOGW("Skip unsupported model task type: %d", static_cast<int32_t>(model_task_type));
    return MSPROF_GE_TASK_TYPE_INVALID;
  }

  MsprofGeTaskType task_type;
  if (model_task_type == ModelTaskType::MODEL_TASK_KERNEL) {
    const auto &context = task_def.kernel().context();
    const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
    if (kernel_type == ccKernelType::TE) {
      task_type = MSPROF_GE_TASK_TYPE_AI_CORE;
    } else if ((kernel_type == ccKernelType::AI_CPU) || (kernel_type == ccKernelType::CUST_AI_CPU) ||
               (kernel_type == ccKernelType::AI_CPU_KFC)) {
      task_type = MSPROF_GE_TASK_TYPE_AI_CPU;
    } else {
      task_type = MSPROF_GE_TASK_TYPE_AI_CORE;
    }
    GELOGD("Kernel type: %u, task type: %u", static_cast<uint32_t>(kernel_type), static_cast<uint32_t>(task_type));
  } else {
    task_type = it->second;
    GELOGD("Model kernel type: %u, task type: %u",
           static_cast<uint32_t>(model_task_type), static_cast<uint32_t>(task_type));
  }

  return task_type;
}

std::string GetPlatformSoPath() {
  char path_env[MMPA_MAX_PATH] = {0};
  int32_t ret = mmGetEnv(kAscendHomePath, path_env, MMPA_MAX_PATH);
  if ((ret != EN_OK) && (strlen(path_env) == 0)) {
    GELOGI("Get platform so path failed because %s is not set", kAscendHomePath);
    return "";
  }
  std::string so_path = std::string(path_env) + kPlatformSoPathSuffix;
  return ge::RealPath(so_path.c_str());
}

bool CheckPlatformSoExist(const std::string &platform_so_path) {
  if (platform_so_path.empty()) {
    return false;
  }
  if (mmAccess2(platform_so_path.c_str(), M_F_OK) != EN_OK) {
    return false;
  }
  return true;
}

Status ReadPlatformSo(const std::string &platform_so_path,
    std::unique_ptr<char_t []> &buf, uint32_t &buf_len) {
  // 读取platform so到buf
  std::ifstream file(platform_so_path.c_str(), std::ios::binary | std::ios::in);
  GE_ASSERT_TRUE(file.is_open(), "File: %s does not exist or is unaccessible.", platform_so_path.c_str());
  GE_MAKE_GUARD(file_guard, [&file]() {
    (void)file.close();
  });
  const std::streampos begin = file.tellg();
  (void)file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  buf_len = static_cast<uint32_t>(end - begin);
  GE_ASSERT_TRUE(static_cast<int32_t>(buf_len) > 0, "file: %s data is empty.", platform_so_path.c_str());
  buf = MakeUnique<char_t []>(buf_len);
  GE_ASSERT_NOTNULL(buf);
  (void)file.seekg(0, std::ios::beg);
  (void)file.read(buf.get(), buf_len);
  return SUCCESS;
}

struct PlatformInfosLaunchArgs {
  uint64_t proto_ptr{0UL};
  uint64_t proto_len{0UL};
  uint64_t platform_infos_addr{0UL};
};

struct LoadCustPlatformInfosArgs {
  uint64_t args{0UL};
  uint64_t args_size{0UL};
};

// 兼容处理，引擎不设置sqe_num则表示只有1个任务
int32_t GetTaskSqeNum(const domi::TaskDef &task) {
  size_t sqe_num = static_cast<int32_t>(task.sqe_num());
  return sqe_num == 0 ? 1 : sqe_num;
}

// trans string to map, "0:0,1:0,2:1" to {{0,0}, {1,0}, {2,1}}
Status TransStrToMap(const std::string map_str, std::map<int64_t, int64_t> &result) {
  std::stringstream ss(map_str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t pos = item.find(':');
    if (pos != std::string::npos) {
      int64_t key = -1L;
      GE_ASSERT_SUCCESS(ge::ConvertToInt64(item.substr(0, pos), key));
      int64_t value = -1L;
      GE_ASSERT_SUCCESS(ge::ConvertToInt64(item.substr(pos + 1), value));
      result[key] = value;
    }
  }
  return SUCCESS;
}

}  // namespace

DavinciModel::DavinciModel(const int32_t priority, const std::shared_ptr<ModelListener> &listener)
    : listener_(listener), priority_(priority), data_dumper_(&runtime_param_) {
  op_list_.clear();
  operator_list_.clear();
  support_extend_memory_full_ = VarManager::IsGeUseExtendSizeMemoryFull();
  args_manager_.Init(this);
}

DavinciModel::~DavinciModel() noexcept {
  // static model has not finalized
  if (!has_finalized_) {
    GELOGI("Npu model: %u starts to finalize.", model_id_);
    // clear exception dump info before stream release
    exception_dumper_.Clear();
    UnbindTaskSinkStream();
    DestroyStream();
    DestroyResources();
    GELOGI("Npu model: %u success to finalize.", model_id_);

    if ((reusable_stream_allocator_ != nullptr) && (!is_outer_allocator_)) {
      delete reusable_stream_allocator_;
      reusable_stream_allocator_ = nullptr;
    }
  }
}

void DavinciModel::DestroyResources() {
  GELOGD("Npu model: %u start to destroy resources.", model_id_);
  GE_CHK_STATUS(ModelRunStop());
  data_dumper_.UnloadDumpInfo();

  LogModelDevMemInfo();

  ClearTaskAddrs();

  (void)ModelManager::GetInstance().ClearBuiltinAicpuSo(device_id_);
  ProfilingManager::Instance().RemoveUnloadedModelId(model_id_);
  if (!known_node_) {
    ErrorTracking::GetInstance().ClearUnloadedModelOpdescInfo(model_id_);
  }

  op_list_.clear();
  operator_list_.clear();
  // check rt ctx is exist. rt api call will cause error log when ctx does not exist
  rtContext_t current_ctx = nullptr;
  if (rtCtxGetCurrent(&current_ctx) == RT_ERROR_NONE) {
    for (size_t i = 0U; i < label_list_.size(); ++i) {
      if (label_list_[i] != nullptr) {
        GE_LOGW_IF(rtLabelDestroy(label_list_[i]) != RT_ERROR_NONE, "Destroy label failed, index: %zu.", i);
      }
    }

    for (size_t i = 0U; i < notify_list_.size(); ++i) {
      GE_LOGW_IF(rtNotifyDestroy(notify_list_[i]) != RT_ERROR_NONE, "Destroy notify failed, index: %zu", i);
    }

    for (size_t i = 0U; i < event_list_.size(); ++i) {
      GE_LOGW_IF(rtEventDestroySync(event_list_[i]) != RT_ERROR_NONE, "Destroy event failed, index: %zu", i);
    }

    for (size_t i = 0U; i < hccl_group_ordered_event_list_.size(); ++i) {
      GE_LOGW_IF(rtEventDestroy(hccl_group_ordered_event_list_[i]) != RT_ERROR_NONE,
        "Destroy hccl group ordered event failed, index: %zu", i);
    }

    for (const auto &it : stream_2_event_) {
      GE_LOGW_IF(rtEventDestroySync(it.second) != RT_ERROR_NONE, "Destroy event failed");
    }

    ReleaseTask();

    FreeWeightsMem();

    FreeFileConstantMem();

    FreeFeatureMapMem();

    FreeExMem();

    FreeDynamicWorkspaceMemory();

    OpDebugUnRegister();

    l1_fusion_addr_ = nullptr;

    if (rt_model_handle_ != nullptr) {
      GE_CHK_RT(rtModelDestroy(rt_model_handle_));
      rt_model_handle_ = nullptr;
    }
  }
  model_kernel_handles_manager_.ClearAllHandle();
  bin_kernel_handle_.CleanTbeHandle();
  aicpu_resources_.ReleaseResources();

  var_mem_base_ = 0U;
  global_step_addr_ = 0U;
  for (auto &it : platform_infos_addr_) {
    if (it.second != nullptr) {
      it.second = nullptr;
    }
  }

  for (auto &it : cust_platform_infos_addr_) {
    if (it.second != nullptr) {
      it.second = nullptr;
    }
  }

  for (auto &it : cust_platform_infos_addr_to_launch_) {
    if (it.second.first != nullptr) {
      it.second.first = nullptr;
    }
  }

  has_finalized_ = true;
  GELOGD("Npu model: %u success to destroy resources.", model_id_);
}

void DavinciModel::ClearTaskAddrs() {
  saved_task_addrs_.clear();
}

void DavinciModel::UnbindHcomStream() {
  for (size_t i = 0U; i < all_hccl_stream_list_.size(); ++i) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, all_hccl_stream_list_[i]) != RT_ERROR_NONE,
               "Unbind hccl stream from model failed, Index: %zu", i);
  }
}

void DavinciModel::ReleaseTask() {
  for (const auto &task : cpu_task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "[Release][Task] failed, model id:%u.", model_id_);
    }
  }
  cpu_task_list_.clear();

  for (const auto &task : task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "[Release][Task] failed.");
    }
  }
  task_list_.clear();
  label_goto_args_.clear();
}

void DavinciModel::Assign(const GeModelPtr &ge_model) {
  if (ge_model == nullptr) {
    GELOGW("Assign null to ge_model");
  }
  ge_model_ = ge_model;
}

///
/// @ingroup ge
/// @brief Reduce memory usage after task sink.
/// @return: void
///
void DavinciModel::Shrink() {
  DumperShrink();
  ge_model_.reset();  // delete object.
  op_list_.clear();
  operator_list_.clear();
  ClearTaskAddrs();
  // some profiling is reported in init, so clear its cache here
  if (need_clear_dfx_cache_) {
    GELOGI("clear profiling cache");
    ClearProfilingDataCache();
  }
}

Status DavinciModel::InitWeightMem(const uintptr_t mem_ptr, const uintptr_t weight_ptr, const size_t weight_size) {
  if (is_weight_mem_has_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "Call InitWeightMem more than once, model_id:%u, check invalid", model_id_);
    GELOGE(FAILED, "[Check][Param] call InitWeightMem more than once, model id:%u.", model_id_);
    return FAILED;
  }
  is_weight_mem_has_inited_ = true;

  const auto weights_size = ge_model_->GetWeightSize();
  if ((weight_ptr != 0U) && (weight_size < weights_size)) {
    const std::string reason = "The weight size of model " + name_ + " is " + std::to_string(weights_size) +
                               ", but the input size set by user is " + std::to_string(weight_size);
    const std::vector<const char_t *> key{"reason"};
    const std::vector<const char_t *> val{reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("E13025", key, val);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] Invalid mem param. %s", reason.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  weights_mem_base_ = mem_ptr;
  is_inner_weight_base_ = false;

  if (weights_size != 0U) {
    weights_mem_base_ = weight_ptr;
    is_inner_weight_base_ = false;
    if (weight_ptr == 0U) {
      weights_mem_base_ = static_cast<uintptr_t>(
          PtrToValue(ModelManager::MallocWeightsMem(GetWeightsMemId(), device_id_, weights_size)));
      if (weights_mem_base_ == 0U) {
        REPORT_INNER_ERR_MSG("E19999", "MallocWeightsMem fail, weights_size:%zu, model_id:%u, check invalid",
                          weights_size, model_id_);
        GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Alloc][Memory] for weight failed. size:%zu, model_id:%u",
               weights_size, model_id_);
        return ACL_ERROR_GE_MEMORY_ALLOCATION;
      }
      is_inner_weight_base_ = true;
    }
    GELOGI("[IMAS]InitWeightMem graph_%u MallocMemory type[W] memaddr[0x%" PRIx64 "] mem_size[%zu]",
      runtime_param_.graph_id, weights_mem_base_, weights_size);
    GE_CHK_RT_RET(rtMemcpy(ValueToPtr(weights_mem_base_), weights_size, ge_model_->GetWeightData(), weights_size,
                           RT_MEMCPY_HOST_TO_DEVICE));
    GELOGI("copy weights data to device");
  }
  dev_mem_statistic_.alloc_size += weights_size;
  runtime_param_.weight_base = weights_mem_base_;
  return SUCCESS;
}

Status DavinciModel::InitFixedFeatureMap(const uintptr_t fixed_mem_ptr, const size_t fixed_mem_size) {
  if (fixed_mem_ptr == 0U) {
    return SUCCESS;
  }

  uintptr_t mem_base = fixed_mem_ptr;
  fixed_mem_size_ = 0U;
  for (auto &mem_info : runtime_param_.fixed_fm_memory_infos) {
    fixed_mem_size_ += static_cast<size_t>(mem_info.memory_size);
    GE_ASSERT_TRUE(fixed_mem_size >= fixed_mem_size_,
      "fix size:%zu less than fix mem size:%zu", fixed_mem_size, fixed_mem_size_);
    mem_info.memory_base = (PtrToPtr<void, uint8_t>(reinterpret_cast<void *>(mem_base)));
    (void)runtime_param_.sorted_memory_infos.insert(mem_info);
    mem_base += static_cast<uintptr_t>(mem_info.memory_size);
    GELOGI("Update one fixed sub feature map memory info with details: [%s]", mem_info.ToString().c_str());
  }

  // 使用用户实际申请内存的大小
  io_mem_base_ =
    (io_mem_base_ < (fixed_mem_ptr + fixed_mem_size)) ? (fixed_mem_ptr + fixed_mem_size) : io_mem_base_;

  GELOGI("fixed_mem_size_: %zu, io_mem_base_: %zu ", fixed_mem_size_, io_mem_base_);
  return SUCCESS;
}

Status DavinciModel::InitFeatureMapAndP2PMem(const uintptr_t mem_ptr, const size_t mem_size, void *outer_fm_mem) {
  if (is_feature_map_mem_has_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "InitFeatureMapMem is called more than once, model_id:%u, check invalid", model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] InitFeatureMapMem is called more than once, model_id:%u", model_id_);
    return PARAM_INVALID;
  }
  is_feature_map_mem_has_inited_ = true;
  if (!runtime_param_.memory_infos.empty()) {
    GE_CHK_STATUS_RET(MallocExMem(), "MallocExMem failed.");
  }
  size_t data_size = TotalMemSize();
  int64_t non_zero_copy_memory_size = 0;
  GE_ASSERT_SUCCESS(GetTotalMemSizeExcludeZeroCopy(non_zero_copy_memory_size));
  const size_t required_minimum_size = static_cast<size_t>(non_zero_copy_memory_size);
  if (!is_static_model_addr_fixed_ && known_node_) {
    mem_base_size_ = required_minimum_size;
    return SUCCESS;
  }

  if (ModelUtils::IsReuseZeroCopyMemory()) {
    data_size = static_cast<size_t>(non_zero_copy_memory_size);
    GELOGI("Model %u need %zu/%zu for feature-map without zero-copyable memory", model_id_, data_size, TotalMemSize());
  }

  if ((mem_ptr != 0U) && (mem_size + fixed_mem_size_ < required_minimum_size)) {
    REPORT_INNER_ERR_MSG("E19999", "Param mem_ptr is nullptr or mem_size:%zu + fixed_size:%zu < ge_model.mem_size:%zu, "
                       "model_id:%u, check invalid", mem_size, fixed_mem_size_, required_minimum_size, model_id_);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param] Invalid mem param: mem_size=%zu, fixed_mem_size=%zu, required_minimum_size=%zu, "
           "model_id:%u.", mem_size, fixed_mem_size_, required_minimum_size, model_id_);

    return ACL_ERROR_GE_PARAM_INVALID;
  }

  mem_base_ = mem_ptr;
  mem_base_size_ = mem_size;
  is_inner_mem_base_ = false;

  if ((data_size != 0U) && (mem_base_ == 0U)) {
    if (outer_fm_mem != nullptr) {
      mem_base_ = static_cast<uintptr_t>(PtrToValue(outer_fm_mem));
      is_inner_mem_base_ = false;
    } else if (data_size > fixed_mem_size_) { // refresh fm的内存需要分配
      mem_base_ = static_cast<uintptr_t>(PtrToValue(MallocFeatureMapMem(data_size)));
      is_inner_mem_base_ = true;

      if (mem_base_ == 0U) {
        REPORT_INNER_ERR_MSG("E19999",
                          "MallocFeatureMapMem fail, data_size:%zu, fixed data size%zu, model_id:%u, check invalid",
                          data_size, fixed_mem_size_, model_id_);
        GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
              "[Alloc][Memory] for feature map failed. size:%zu, fixed size:%zu, model_id:%u",
              data_size, fixed_mem_size_, model_id_);
        return ACL_ERROR_GE_MEMORY_ALLOCATION;
      }
    }
    GEEVENT("[IMAS]InitFeatureMapAndP2PMem graph_%u MallocMemory type[F] memaddr[0x%" PRIx64 "] mem_size[%zu]Bytes",
            runtime_param_.graph_id, mem_base_, data_size);

    mem_base_size_ = data_size;
  } else if (mem_base_ != 0U && runtime_param_.fixed_mem_base != 0) { // refresh fm设置，fix fm设置，使用分段
      size_t used_mem_size = 0U;
      GE_ASSERT_SUCCESS(UpdateHbmFmMemBases(mem_base_, mem_size, used_mem_size, true));
      GELOGI("Update feature memory base success, model_id:%u, mem_base:%#lx, mem_size:%zu, used_mem_size:%zu",
        model_id_, mem_base_, mem_size, used_mem_size);

      io_mem_base_ =
        (io_mem_base_ < (mem_base_ + mem_size)) ? (mem_base_ + mem_size) : io_mem_base_;

      // 添加零拷贝信息， 外部设置refresh fm地址时，io内存为外部分配，该行为继承原来fm 地址设置接口
      MemInfo fm_info(runtime_param_.mem_size - static_cast<uint64_t>(runtime_param_.zero_copy_size),
        runtime_param_.zero_copy_size, PtrToPtr<void, uint8_t>(ValueToPtr(io_mem_base_)));
      (void) runtime_param_.sorted_memory_infos.insert(std::move(fm_info));

      // fm段实际使用的内存大小
      mem_base_size_ = used_mem_size + fixed_mem_size_;
  }

  GE_CHK_STATUS_RET(InitVariableMem(), "[Init][VariableMemory] failed, model_id: %u", model_id_);
  GE_ASSERT_SUCCESS(UpdateRuntimeParamBase());
  runtime_param_.weight_base = weights_mem_base_;
  return SUCCESS;
}

Status DavinciModel::InitVariableMem() {
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  GELOGI("[Init][Var] Variable max size do not set, no need to malloc max size mem.");
  runtime_param_.var_base = 0U;
  const auto page_size = VarManager::IsVariableUse1gHugePage() ? kDrv1GPageSize : kDrvPageSize;
  auto allocator =
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(session_id_,
      GetDeviceId(), RT_MEMORY_HBM, page_size);
  (void) VarManager::Instance(session_id_)->InitExpandableMemoryAllocator(allocator);
  return SUCCESS;
}

Status DavinciModel::InitRuntimeParams() {
  GE_ASSERT_SUCCESS(ModelUtils::InitRuntimeParams(ge_model_, runtime_param_, device_id_));
  session_id_ = runtime_param_.session_id;
  const auto &var_manager = VarManager::Instance(session_id_);
  if ((var_manager != nullptr) && (!var_manager->HasMemoryManager())) {
    var_manager->SetMemManager(&MemManager::Instance());
  }
  GELOGI("InitRuntimeParams: %s.", runtime_param_.ToString().c_str());
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Make active stream list and bind to model.
/// @return: 0 for success / others for fail
///
Status DavinciModel::BindModelStream() {
  // Stream not in active_stream_indication_ is active stream.
  bool is_software_queue = !is_hw_q_ && ((!input_queue_attrs_.empty()) || (!output_queue_attrs_.empty()));
  if (is_software_queue) {
    for (size_t i = 0UL; i < stream_list_.size(); ++i) {
      if (active_stream_indication_.count(static_cast<uint32_t>(i)) == 0U) {
        active_stream_list_.push_back(stream_list_[i]);
        (void)active_stream_indication_.insert(static_cast<uint32_t>(i));  // deactive all model stream.
      }
    }
  }
  std::map<uint32_t, uint32_t> first_task_id_to_stream;
  GE_ASSERT_TRUE(stream_to_first_task_id_.size() <= stream_list_.size(),
                 "stream_to_first_node_id size %zu more than stream size %zu", stream_to_first_task_id_.size(),
                 stream_list_.size());
  for (const auto &pair : stream_to_first_task_id_) {
    first_task_id_to_stream[pair.second] = pair.first;
    GELOGI("stream id %" PRId64 ", first task id %" PRId64, pair.first, pair.second);
  }
  for (const auto &pair : first_task_id_to_stream) {
    auto stream_id = pair.second;
    const auto bind_flag =
        (active_stream_indication_.count(stream_id) == 0U) ? RT_HEAD_STREAM : RT_INVALID_FLAG;
    GELOGI("rtModelBindStream[%zu] stream: %p, flag: %#x", stream_id, stream_list_[stream_id], static_cast<uint32_t>(bind_flag));
    GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[stream_id], static_cast<uint32_t>(bind_flag)));
  }
  is_stream_list_bind_ = true;
  return SUCCESS;
}

// 对于纯静态图，在load阶段完成fm相关连的args刷新
Status DavinciModel::UpdateStaticModelArgsByFm() {
  // 首次刷新为all-one-time
  uint32_t ret_up = kUpdatePolicyAllOneTime;
  if ((!input_queue_attrs_.empty()) || (!output_queue_attrs_.empty())) {
    GELOGI("Update cpu model args, model_id:%u", model_id_);
    args_manager_.InitDfxStage1Begin();
    ConstructActiveMemBaseAddrs();
    // 此时已经确定了执行时是否走算子化刷新，使用正确的device地址来 更新对应io的device地址
    GE_ASSERT_SUCCESS(InitCopyHostInputInfos());
    rtStream_t stream = nullptr;
    GE_CHK_RT_RET(rtStreamCreate(&stream, 0));
    GE_MAKE_GUARD_RTSTREAM(stream);
    // 加载阶段同老流程走全量model args h2d拷贝
    GE_ASSERT_SUCCESS(args_manager_.UpdateForExecute(ret_up, stream, kModelLoadStage));
    args_manager_.InitDfxStatsticsEnd();
    args_manager_.PrintDfxStatistics(kModelLoadStage);
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GELOGI("Sync stream successfully, model_id: %u", model_id_);
  } else if ((!feature_base_refreshable_) || (host_input_size_ > 0U)) {
    GELOGI("Update static model args, model_id: %u", model_id_);
    args_manager_.InitDfxStage1Begin();
    ConstructActiveMemBaseAddrs();
    // 此时已经确定了执行时是否走算子化刷新，使用正确的device地址来 更新对应io的device地址
    GE_ASSERT_SUCCESS(InitCopyHostInputInfos());
    // 加载阶段同老流程走全量model args h2d拷贝
    GE_ASSERT_SUCCESS(args_manager_.UpdateForExecute(ret_up, rt_model_stream_, kModelLoadStage));
    args_manager_.InitDfxStatsticsEnd();
    args_manager_.PrintDfxStatistics(kModelLoadStage);
  }
  return SUCCESS;
}

Status DavinciModel::CreateHcclGroupOrderedEvent() {
  int32_t device_id = -1;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));
  for (const auto &group_id : hccl_group_id_set_) {
    rtStream_t stream = nullptr;
    if (HcomTopoInfo::Instance().GetGroupOrderedStream(device_id, group_id.c_str(), stream) != GRAPH_SUCCESS) {
      GELOGW("Cannot get hccl group order stream by deviceid:%d, group id:%s", device_id, group_id.c_str());
    } else {
      GE_ASSERT_NOTNULL(stream);
      GELOGI("Get device id:%d, group id:%s, hccl group ordered stream:%p", device_id, group_id.c_str(), stream);
      hccl_group_ordered_stream_list_.emplace_back(stream);
    }
  }

  uint32_t i = 0U;
  while (i < hccl_group_ordered_stream_list_.size()) {
    rtEvent_t rt_event = nullptr;
    int32_t stream_id = 0;
    GE_CHK_RT_RET(rtEventCreateExWithFlag(&rt_event, static_cast<uint32_t>(RT_EVENT_WITH_FLAG)));
    hccl_group_ordered_event_list_.push_back(rt_event);
    (void)rtGetStreamId(hccl_group_ordered_stream_list_[i], &stream_id);
    GELOGI("hccl group ordered stream id:%d, stream:%p", stream_id, hccl_group_ordered_stream_list_[i]);
    ++i;
  }
  return SUCCESS;
}

Status DavinciModel::InitCopyHostInputInfos() {
  if (host_input_size_ == 0U) {
    return SUCCESS;
  }

  uint64_t host_addr = 0U;
  uint64_t device_addr = 0U;
  uint64_t len = 0U;
  GE_ASSERT_SUCCESS(args_manager_.GetHostInputMem(host_addr, device_addr, len));
  std::vector<uint32_t> copy_host_input_indexes_vec(copy_host_input_indexes_.begin(), copy_host_input_indexes_.end());
  std::sort(copy_host_input_indexes_vec.begin(), copy_host_input_indexes_vec.end());
  for (const auto &index : copy_host_input_indexes_vec) {
    if (input_indexes_to_copy_info_.find(index) != input_indexes_to_copy_info_.end()) {
      continue;
    }
    auto &copy_info = copy_host_input_infos_.at(index);
    GE_ASSERT(len >= copy_info.tensor_size,
      "host input index:%u, tensro size:%" PRIu64 ", len:%" PRIu64, index, copy_info.tensor_size, len);
    copy_info.host_addr = ValueToPtr(host_addr);
    copy_info.device_addr = device_addr;
    // 更新io 段的基地址
    GE_ASSERT(index < input_index_to_allocation_ids_.size());
    uint32_t allcation_id = input_index_to_allocation_ids_[index];
    allocation_ids_to_active_base_addr_[allcation_id] = copy_info.device_addr;
    GELOGI(
      "[ActiveMemBase], host input index:%u, model_id:%u, id:%zu, active mem base:0x%" PRIx64 ", host mem base:0x%" PRIx64
      ", len:%" PRIu64, index, model_id_, allcation_id, allocation_ids_to_active_base_addr_[allcation_id], host_addr,
      copy_info.tensor_size);

    host_addr += copy_info.tensor_size;
    device_addr += copy_info.tensor_size;
    len -= copy_info.tensor_size;
  }
  return SUCCESS;
}

Status DavinciModel::DoTaskSink() {
  // task sink is supported as model_task_def is set
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(BindModelStream(), "[Bind][ModelStream] failed, model_id: %u.", model_id_);

  GE_CHK_STATUS_RET(InitL1DataDumperArgs(), "[Init][L1DataDumperArgs] failed, model_id: %u.", model_id_);

  GE_CHK_STATUS_RET(InitTaskInfo(*model_task_def), "[Init][TaskInfo] failed, model_id: %u.", model_id_);

  GE_CHK_STATUS_RET(LoadWithQueue(), "[Init][LoadWithQueue] failed, model_id: %u.", model_id_);

  GE_CHK_STATUS_RET(LaunchCustPlatformInfos(), "[Launch][CustPlatform] failed, model_id: %u.", model_id_);
  auto &model_mgr = ModelManager::GetInstance();
  GE_CHK_STATUS_RET(model_mgr.LaunchBuiltinAicpuSo(device_id_),
                    "[Launch][BuiltinAicpuSo] failed, model_id: %u.", model_id_);
  GE_CHK_STATUS_RET(model_mgr.CheckAicpuOpList(ge_model_), "[Check][AicpuOpList] failed, model_id: %u.", model_id_);

  GE_CHK_STATUS_RET(DistributeTask(*model_task_def), "[Distribute][Task] failed, model_id: %u.", model_id_);

  GE_CHK_STATUS_RET(CreateHcclGroupOrderedEvent());

  CalculateMemAllocationsHitInfo();

  GE_CHK_STATUS_RET(UpdateStaticModelArgsByFm());

  GE_CHK_RT_RET(rtModelLoadComplete(rt_model_handle_));

  GE_CHK_STATUS_RET(LoadWithHardwareQueue(), "[Init][LoadWithHardwareQueue] failed, model_id: %u.", model_id_);

  GE_ASSERT_SUCCESS(SetCopyOnlyOutput());

  return SUCCESS;
}

Status DavinciModel::RecoverModel() {
  uint32_t task_id = 0U;

  for(const auto& pair : stream_to_task_index_list_) {
    for (auto &task_index : pair.second) {
      const auto &task_info = task_list_.at(static_cast<size_t>(task_index));
      GE_ASSERT_NOTNULL(task_info);
      task_id = task_info->GetTaskID();

      GE_ASSERT_TRUE(task_info->IsSupportReDistribute(),
        "task index: %zu, task id: %u no support redestribute", task_index, task_id);
    }
  }

  for(const auto& pair : stream_to_task_index_list_) {
    GELOGI("recover stream:%" PRIu64 ", task num:%zu", pair.first, pair.second.size());
    // 从流清理
    if (main_follow_stream_mapping_.find(pair.first) != main_follow_stream_mapping_.end()) {
      for (auto &follow_stream : main_follow_stream_mapping_[pair.first]) {
        GE_CHK_RT_RET(rtStreamTaskClean(follow_stream));
      }
    }

    GE_CHK_RT_RET(rtStreamTaskClean(ge::ValueToPtr(pair.first)));
    for (auto &task_index : pair.second) {
      const auto &task_info = task_list_.at(static_cast<size_t>(task_index));
      GE_ASSERT_NOTNULL(task_info);
      task_id = task_info->GetTaskID();
      GE_CHK_STATUS_RET(task_info->Distribute(), "[Call][Distribute] for Task[%d] fail", task_index);
      uint32_t new_task_id = task_info->GetTaskID();
      GELOGI("raw task id %u task index %u, after redistribute task id %u", task_id, task_index, task_info->GetTaskID());
      if (task_id == new_task_id) {
        GELOGI("skip update, old task id %u equals new task id", task_id);
        continue;
      }
      ErrorTracking::GetInstance().UpdateTaskId(task_id, new_task_id, task_info->GetStreamId(), model_id_);
      args_manager_.OnTaskDistributed(static_cast<size_t>(task_index), task_info.get());
    }
  }
  return SUCCESS;
}

void DavinciModel::CalculateMemAllocationsHitInfo() {
  for (const auto &it : logical_mem_allocations_) {
    if ((it.type == ge::MemAllocation::Type::INPUT) || (it.type == ge::MemAllocation::Type::OUTPUT)) {
      model_io_hit_count_ += it.hit_count;
    } else if (it.type == ge::MemAllocation::Type::FEATURE_MAP) {
      fm_hit_count_ += it.hit_count;
    }
    GELOGI("[mem allocation] model_id %u, %s.", model_id_, it.ToString().c_str());
  }

  args_manager_.SetAllocationHitCount(fm_hit_count_, model_io_hit_count_);
  GELOGD("[mem allocation] model_id %u, fm_hit_count:0x%" PRIx64 ", model_io_hit_count:0x%" PRIx64 ".",
         model_id_, fm_hit_count_, model_io_hit_count_);
};

// set device use aicore(0) or vectorcore(1)
Status DavinciModel::SetTSDevice() {
  int64_t value = 0;
  const bool ret = AttrUtils::GetInt(ge_model_, ATTR_MODEL_CORE_TYPE, value);
  const uint32_t core_type = ret ? static_cast<uint32_t>(value) : 0U;
  GELOGD("Set TSDevice: %u.", core_type);
  GE_CHK_RT_RET(aclrtSetTsDevice(static_cast<aclrtTsId>(core_type)));
  return SUCCESS;
}

Status DavinciModel::OpDebugRegister() {
  if (GetDumpProperties().IsOpDebugOpen() && (!is_op_debug_reg_)) {
    const uint32_t op_debug_mode = GetDumpProperties().GetOpDebugMode();
    const auto ret = opdebug_register_.RegisterDebugForModel(rt_model_handle_, op_debug_mode, data_dumper_);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][RegisterDebugForModel] Register known shape op debug failed, ret: 0x%X", ret);
      return ret;
    }
    is_op_debug_reg_ = true;
  }
  return SUCCESS;
}

void DavinciModel::OpDebugUnRegister() {
  if (is_op_debug_reg_) {
    opdebug_register_.UnregisterDebugForModel(rt_model_handle_);
    is_op_debug_reg_ = false;
  }
  return;
}

Status DavinciModel::SetModelConfig() const {
  if (need_model_config_) {
    AiCpuResources::AiCpuModelConfig config = {};
    config.model_id = model_id_;
    config.runtime_model_id = runtime_model_id_;
    config.abnormal_exist = 0;
    config.abnormal_enqueue = 1;
    config.req_msg_queue = -1;
    config.resp_msg_queue = -1;
    GE_CHK_STATUS_RET_NOLOG(aicpu_resources_.SetModelConfig(config));
  }
  return SUCCESS;
}

void DavinciModel::SetStaticModelShapeConfig() {
  if (model_queue_param_.need_check_inputs) {
    AiCpuResources::AiCpuModelShapeConfig config = {};
    config.model_id = model_id_;
    config.runtime_model_id = runtime_model_id_;
    (void)aicpu_resources_.SetStaticModelShapeConfig(config, origin_input_descs_);
  }
}

std::string DavinciModel::FindKernelInPath(const std::string& path, const std::string& npu_arch) const {
  std::regex pattern(R"((?:.*/)?(cann-[^/]+/lib64|runtime/lib64)(?:/.*)?)", std::regex::icase);
  std::smatch match;
  if (!std::regex_match(path, match, pattern)) {
    return "";
  }

  std::string matched_dir = match[1].str();
  size_t pos = path.find(matched_dir);
  if (pos != std::string::npos) {
    matched_dir = path.substr(0, pos + matched_dir.length());
  }

  while (!matched_dir.empty() && matched_dir.back() == '/') {
    matched_dir.pop_back();
  }

  std::string full_path = matched_dir + kAddrRefreshOpPath + "_" + npu_arch + ".o";
  std::ifstream file(full_path.c_str());
  if (!file.good()) {
    GELOGW("Addr refresh kernel file not found at: %s", full_path.c_str());
    return "";
  }

  return full_path;
}

std::string DavinciModel::FindAddrRefreshKernelFile(const std::string& npu_arch) const {
  const char_t* lib_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, lib_path);

  if (lib_path == nullptr || std::string(lib_path).empty()) {
    GELOGW("Environment variable MM_ENV_LD_LIBRARY_PATH is not set or empty.");
    return "";
  }

  std::string env_path = lib_path;
  std::vector<std::string> search_paths;
  size_t start = 0, end = 0;

  while ((end = env_path.find(':', start)) != std::string::npos) {
    if (end > start) {
      search_paths.emplace_back(env_path.substr(start, end - start));
    }
    start = end + 1;
  }

  if (start < env_path.length()) {
    search_paths.emplace_back(env_path.substr(start));
  }

  for (const auto& path : search_paths) {
    std::string found_file = FindKernelInPath(path, npu_arch);
    if (!found_file.empty()) {
      return found_file;
    }
  }

  GELOGW("Cannot find addr refresh kernel file in any matching directory of MM_ENV_LD_LIBRARY_PATH.");
  return "";
}

Status DavinciModel::LoadAndRegisterAddrRefreshKernel(const std::string& file_path) {
  uint8_t *buf = nullptr;
  int32_t buf_len = 0U;

  if (!ReadBytesFromBinaryFile(file_path.c_str(), PtrToPtr<uint8_t *, char_t *>(&buf), buf_len)) {
    GELOGW("Read Addr Refresh Op failed from: %s", file_path.c_str());
    return SUCCESS;
  }

  GE_ASSERT_NOTNULL(buf);
  std::vector<char> data(buf, PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(buf) + static_cast<uint64_t>(buf_len))));
  delete[] buf;

  const TBEKernelPtr addr_refresh_kernel = MakeShared<OpKernelBin>(file_path, std::move(data));
  KernelRegisterInfo register_info;
  AicoreRegisterInfo aicore_register_info;
  aicore_register_info.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  aicore_register_info.kernel_bin = addr_refresh_kernel;
  aicore_register_info.kernel_bin_name = kAddrRefreshOpBinId;
  register_info = aicore_register_info;
  auto kernel_handles_manager = GetKernelHandlesManager(KernelHandleType::kAicore);
  GE_ASSERT_NOTNULL(kernel_handles_manager);
  const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
  auto bin_handle = kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  GE_ASSERT_NOTNULL(bin_handle);
  auto func_handle = KernelHandleUtils::GetFuncHandle(bin_handle, kAddrRefreshOpBinId);
  GE_ASSERT_NOTNULL(func_handle);
  args_manager_.SetFuncHandle(func_handle);
  return SUCCESS;
}

Status DavinciModel::InitAddrRefreshKernelBin() {
  if (gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump)) {
    GELOGI("Npu model do not register UpdateModelParam op when overflow enabled.");
    return SUCCESS;
  }

  std::string npu_arch;
  (void)PlatformInfoUtil::GetSocSpec("version", "NpuArch", npu_arch);
  GELOGI("Current NpuArch is [%s]", npu_arch.c_str());
  if (npu_arch != NPUARCH_TO_STR(NpuArch::DAV_2201)) {
    GELOGW("Npu arch [%s] does not support address-refresh op", npu_arch.c_str());
    return SUCCESS;
  }

  std::string kernel_file_path = FindAddrRefreshKernelFile(npu_arch);
  if (kernel_file_path.empty()) {
    return SUCCESS;
  }

  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Npu model add addr refresh kernel: %s", kernel_file_path.c_str());
  }
  return LoadAndRegisterAddrRefreshKernel(kernel_file_path);
}


Status DavinciModel::InitSpaceRegistry(const GeRootModelPtr &root_model) {
  if (space_registries_ != nullptr && space_registries_->at(static_cast<size_t>(OppImplVersion::kOpp)) != nullptr) {
    GELOGD("Space registry already exist.");
    return SUCCESS;
  }
  GELOGD("Load space registry from so in root model.");
  GE_ASSERT_SUCCESS(ge::ModelUtils::GetSpaceRegistries(root_model, space_registries_));
  return SUCCESS;
}

// initialize op sequence and call initialization function of each op respectively
Status DavinciModel::Init(const ModelParam &param, void *outer_fm_mem) {
  InitModelProf(); // 优先初始化
  GELOGI("Priority is %d.", priority_);
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  SetProfileTime(ModelProcStage::MODEL_LOAD_START, MsprofSysCycleTime());
  is_dump_to_std_enable_ = profiling::ProfilingContext::IsDumpToStdEnabled();
  if ((priority_ < 0) || (priority_ > 7)) {
    GELOGE(FAILED, "[Check][Param] Priority must between 0-7, now is %d.", priority_);
    return PARAM_INVALID;
  }
  GE_CHK_BOOL_RET_STATUS(ge_model_ != nullptr, PARAM_INVALID, "[Check][Param] GeModel is nullptr.");
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  GE_CHK_BOOL_RET_STATUS(compute_graph != nullptr, INTERNAL_ERROR, "[Get][ComputeGraph] failed, ret is nullptr.");
  isGraphLevelSat_ = GetContext().IsGraphLevelSat();
  // Initializing runtime_param_
  runtime_param_.fixed_mem_base = param.fixed_mem_base; // memory type hbm
  runtime_param_.fixed_mem_size = param.fixed_mem_size; // memory type hbm
  runtime_param_.p2p_fixed_mem_base = param.p2p_fixed_mem_base;
  runtime_param_.p2p_fixed_mem_size = param.p2p_fixed_mem_size;

  GE_ASSERT_SUCCESS(InitRuntimeParams());

  // 临时方案：根据环境变量判断hccl算子地址是否可刷新，环境变量开启代表地址不可刷新，待1230 hccl正式方案上库后删除
  std::string is_addr_fixed_opt;
  (void)ge::GetContext().GetOption(kStaticModelAddrFixed, is_addr_fixed_opt);
  is_static_model_addr_fixed_ = is_addr_fixed_opt.empty() ? false : true;
  // 分段信息清除条件
  // 1）fix环境变量设置, todo:临时方案，后续删除
  // 2) 传入外置内存, todo:临时方案，后续删除
  // 3) 外部设置可刷新fm地址且未设置fixed fm地址
  // 4) 纯静态图没有设置fixed fm且未设置扩展模式
  if ((is_static_model_addr_fixed_) ||
      (outer_fm_mem != nullptr) ||
      (param.mem_base != 0U && param.fixed_mem_base == 0U) ||
      ((!ModelUtils::IsGeUseExtendSizeMemory()) && (param.fixed_mem_base == 0U) && (!known_node_))) {
    runtime_param_.fm_memory_infos.clear();
    MemInfo fm_info(0L, static_cast<int64_t>(runtime_param_.mem_size) - runtime_param_.zero_copy_size, nullptr);
    runtime_param_.fm_memory_infos.emplace_back(fm_info);
    GELOGI("clear fm sub memory info.");
  }

  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.HandleDeviceInfo(platform_infos_), "Fail to handle device info.");
  GE_CHK_STATUS_RET(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_and_path_map_), "Failed to get file path.");

  // RTS set aicore or vectorcore
  GE_CHK_STATUS_RET(SetTSDevice(), "[Set][TSDevice] failed, graph[%s].", compute_graph->GetName().c_str());

  version_ = ge_model_->GetVersion();
  name_ = ge_model_->GetName();

  // inference will use default graph_id 0;
  runtime_param_.graph_id = compute_graph->GetGraphID();
  runtime_param_.graph_name = compute_graph->GetName();

  GetStageTimestampStart(kInitMdlMem);
  GELOGI("Known node:%d, refreshable:%d, model_id:%u, graph_id:%u, graph_name:%s.", static_cast<int32_t>(known_node_),
         static_cast<int32_t>(feature_base_refreshable_), model_id_, runtime_param_.graph_id,
         compute_graph->GetName().c_str());

  args_manager_.InitDfx(mdl_prof_.enable_flag, compute_graph->GetName(), runtime_param_.graph_id, model_id_,
                        feature_base_refreshable_, known_node_, mdl_prof_.get_model_args_device_table_flag);

  GE_CHK_STATUS_RET_NOLOG(InitWeightMem(param.mem_base, param.weight_base, param.weight_size));
  GE_CHK_STATUS_RET_NOLOG(InitFixedFeatureMap(param.fixed_mem_base, param.fixed_mem_size));
  GE_CHK_STATUS_RET_NOLOG(InitFeatureMapAndP2PMem(param.mem_base, param.mem_size, outer_fm_mem));
  fixed_mem_base_ = mem_base_;
  GetStageTimestampEnd(kInitMdlMem);

  GE_CHK_STATUS_RET(PreProcessFileConstants(compute_graph, param), "[PreProcess][FileConstant] failed, graph: %s.",
                    compute_graph->GetName().c_str());

  std::vector<NodePtr> variable_nodes;
  GetStageTimestampStart(kInitIoNodes);
  GE_CHK_STATUS_RET(InitIoNodes(compute_graph, variable_nodes), "[Init][InitIoNodes] failed, name: %s", name_.c_str());
  // recover variables
  GE_ASSERT_SUCCESS(RestoreDeviceVarMem(variable_nodes, param));

  GetStageTimestampEnd(kInitIoNodes);

  GE_ASSERT_SUCCESS(InitStreamInfoOfTask(compute_graph));
  GE_CHK_STATUS_RET_NOLOG(InitRuntimeResource());
  GE_CHK_STATUS_RET_NOLOG(InitSupplyResource());

  GetStageTimestampStart(kTransAllData);
  GE_CHK_STATUS_RET_NOLOG(TransAllVarData(compute_graph, variable_nodes));
  GetStageTimestampEnd(kTransAllData);

  GetStageTimestampStart(kInitAllNodes);
  GE_CHK_STATUS_RET(InitNodes(compute_graph), "[Init][Nodes] failed, graph: %s.", compute_graph->GetName().c_str());
  GetStageTimestampEnd(kInitAllNodes);

  GetStageTimestampStart(kDoTaskSink);
  GE_CHK_STATUS_RET(DoTaskSink(), "[Call][DoTaskSink] failed, model_id: %u.", model_id_);
  GetStageTimestampEnd(kDoTaskSink);
  (void)CopyOnlyAddrCheck();

  SetProfileTime(ModelProcStage::MODEL_LOAD_END);

  // collect profiling for ge
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
    GE_CHK_STATUS_RET(ReportProfilingData(), "[Report][ProfilingData] failed, model_id: %u.", model_id_);
    SetReportedProfCount(gert::GlobalProfilingWrapper::GetInstance()->GetProfCount());
  }
  ProfilingManager::Instance().RecordLoadedModelId(model_id_);
  // get stream sync time
  std::string stream_sync_timeout;
  (void)ge::GetContext().GetOption(OPTION_EXEC_STREAM_SYNC_TIMEOUT, stream_sync_timeout);
  stream_sync_timeout_ = stream_sync_timeout.empty()
                             ? kDefaultTimeout
                             : static_cast<int32_t>(std::strtol(stream_sync_timeout.c_str(), nullptr, kBase));

  // Parse output reuse input memory indexes from model attribute
  std::string reuse_indexes_str;
  if (ge::AttrUtils::GetStr(ge_model_, ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, reuse_indexes_str)) {
    ge::ParseOutputReuseInputMemIndexes(reuse_indexes_str, io_same_addr_pairs_);
    GELOGI("Parsed output reuse input mem indexes, pairs count: %zu", io_same_addr_pairs_.size());
  }

  Shrink();
  PrintfModelProfOfModelLoad();
  return SUCCESS;
}

void DavinciModel::InitModelProf() {
  mdl_prof_.init_begin = ge::GetCurrentTimestamp();
  const char_t *davinci_model_profiling = nullptr;
  MM_SYS_GET_ENV(MM_ENV_GE_DAVINCI_MODEL_PROFILING, davinci_model_profiling);
  mdl_prof_.enable_flag = ((davinci_model_profiling != nullptr) && (strcmp(davinci_model_profiling, "1") == 0));
  mdl_prof_.get_model_args_device_table_flag =
    ((davinci_model_profiling != nullptr) && (strcmp(davinci_model_profiling, "2") == 0));
  GELOGI("Init davinci_model profiler flag:%d, get device model args table flag:%d.",
    static_cast<int32_t>(mdl_prof_.enable_flag), static_cast<int32_t>(mdl_prof_.get_model_args_device_table_flag));
}

void DavinciModel::InitModelExecuteProf() {
  if (!mdl_prof_.enable_flag) {
    return;
  }

  mdl_prof_.stage_to_timestamp.clear();
  mdl_prof_.execute_begin = ge::GetCurrentTimestamp();
}

void DavinciModel::GetCurTimestamp(uint64_t &cur_time) {
  if (!mdl_prof_.enable_flag) {
    return;
  }
  cur_time = ge::GetCurrentTimestamp();
}

void DavinciModel::GetStageTimestampStart(ModelProfStage stage) {
  if (!mdl_prof_.enable_flag) {
    return;
  }

  mdl_prof_.stage_to_timestamp[stage] = ge::GetCurrentTimestamp();
}

void DavinciModel::GetStageTimestampEnd(ModelProfStage stage) {
  if (!mdl_prof_.enable_flag) {
    return;
  }

  mdl_prof_.stage_to_timestamp[stage] = ge::GetCurrentTimestamp() - mdl_prof_.stage_to_timestamp[stage] ;
}

void DavinciModel::UpdateTaskTypeStat(const uint32_t task_type, const uint64_t start_t, const uint64_t end_t) {
  if (!mdl_prof_.enable_flag) {
    return;
  }
  auto it = mdl_prof_.task_type_to_distribute_time.find(task_type);
  if (it == mdl_prof_.task_type_to_distribute_time.cend()) {
    mdl_prof_.task_type_to_distribute_time[task_type] = 0UL;
    mdl_prof_.task_type_to_distribute_num[task_type] = 0UL;
  }
  mdl_prof_.task_type_to_distribute_time[task_type] += (end_t - start_t);
  mdl_prof_.task_type_to_distribute_num[task_type] += 1UL;
}

void DavinciModel::PrintfModelProfOfModelLoad() {
  // 打印一条加载所需总时间, GraphLoader是日志关键字
  mdl_prof_.init_end = ge::GetCurrentTimestamp();
  GEEVENT("[GEPERFTRACE] The time cost of GraphLoader::DavinciModel::Init is [%" PRIu64 "] micro seconds, "
          "name[%s]model_id[%u]graph_id[%u].", (mdl_prof_.init_end - mdl_prof_.init_begin), name_.c_str(), model_id_,
          runtime_param_.graph_id);

  if (!mdl_prof_.enable_flag) {
    return;
  }

  for (auto &it : mdl_prof_.stage_to_timestamp) {
      GEEVENT("[GEPERFTRACE] The time cost of GraphLoader::%s is [%" PRIu64 "] micro seconds.",
              GetModelProfStageStr(it.first), it.second);
  }

  for (auto &it : mdl_prof_.task_type_to_distribute_time) {
      GEEVENT("[GEPERFTRACE] The time cost of GraphLoader::Distribute of task_type[%u] is [%" PRIu64 "] micro seconds,"
      " task_num:%" PRIu64 ".", it.first, it.second, mdl_prof_.task_type_to_distribute_num[it.first]);
  }
}

void DavinciModel::PrintfModelProfOfModelExecute() {
  if (!mdl_prof_.enable_flag) {
    return;
  }
  mdl_prof_.execute_end = ge::GetCurrentTimestamp();

  uint64_t nnexecute_t = mdl_prof_.execute_end - mdl_prof_.execute_begin;

  std::stringstream ss;
  for (auto &it : mdl_prof_.stage_to_timestamp) {
    ss << GetModelProfStageStr(it.first) << "_time[" << it.second << "]us,";
  }

  std::stringstream ss_args_update;
  args_manager_.CalculateDfxTime(ss_args_update);
  GEEVENT("[GEPERFTRACE] graph_name:%s, graph_id:%u, model_id:%u, nnexecute_time[%" PRIu64 "]us, %s %s",
    name_.c_str(), runtime_param_.graph_id, model_id_, nnexecute_t,
    ss.str().c_str(), ss_args_update.str().c_str());

  // args manager 单独打印
  args_manager_.PrintDfxStatistics();
}

int32_t DavinciModel::GetTaskNumOfStream(const uint32_t stream_id) const {
  const auto iter = stream_task_num_.find(stream_id);
  GE_ASSERT_TRUE(iter != stream_task_num_.cend());
  return iter->second;
}

Status DavinciModel::GetTaskNumOfTaskdef(
    const domi::TaskDef &task_def, int32_t &task_num,
    std::map<std::pair<int64_t, int32_t>, std::set<uint32_t>> &taskdef_task_num) const {
  const auto &task_info = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task_def.type()));
  GE_ASSERT_NOTNULL(task_info);
  const int64_t op_index = task_info->ParseOpIndex(task_def);
  if (op_index == kInvalidOpIndex) {
    task_num = kDefaultTaskNum;
    return SUCCESS;
  }

  const auto iter = op_list_.find(op_index);
  GE_ASSERT_TRUE(iter != op_list_.cend());
  const auto &op_desc = iter->second;
  if (AttrUtils::GetInt(op_desc, ATTR_NAME_NODE_SQE_NUM, task_num)) {
    if(IsEventWaitNode(op_desc)) {
      task_num++;
    }
    GELOGD("Node: %s, type: %s, sqe num: %d.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), task_num);
  } else if ((hccl_op_types.find(op_desc->GetType()) != hccl_op_types.cend()) &&
             AttrUtils::GetInt(op_desc, ATTR_NAME_HCCL_TASK_NUM, task_num)) {
    GELOGD("Node: %s, type: %s, hccl task num: %d.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), task_num);
  } else {
    auto task_type = static_cast<ModelTaskType>(task_def.type());
    if (task_type == ModelTaskType::MODEL_TASK_EVENT_WAIT) {
      task_num = kDefaultEventWaitTaskNum;
    } else {
      task_num = GetTaskSqeNum(task_def);
    }
  }

  const uint32_t stream_id = task_def.stream_id();
  const auto task_iter = taskdef_task_num.find(std::make_pair(op_index, task_num));
  if (task_iter == taskdef_task_num.cend()) {
    taskdef_task_num[std::make_pair(op_index, task_num)].emplace(stream_id);
  } else {
    auto &stream_list = task_iter->second;
    if (!stream_list.emplace(stream_id).second) {
      task_num = kNoTask;
      GELOGD("Node: %s, type: %s, stream id: %u, has been calculated, current task num is 0.", op_desc->GetNamePtr(),
             op_desc->GetTypePtr(), stream_id);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitStreamInfoOfTask(const ComputeGraphPtr &compute_graph) {
  for (const auto node : compute_graph->GetAllNodesPtr()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_list_[op_desc->GetId()] = op_desc;
  }

  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_CHECK_NOTNULL(model_task_def);
  const int32_t task_size = model_task_def->task_size();
  // <<node id, task num>, <stream id>>
  std::map<std::pair<int64_t, int32_t>, std::set<uint32_t>> taskdef_task_num;
  for (int32_t i = 0; i < task_size; ++i) {
    const auto &task_def = model_task_def->task(i);
    int32_t task_num = 0;
    GE_ASSERT_SUCCESS(GetTaskNumOfTaskdef(task_def, task_num, taskdef_task_num));
    stream_task_num_[task_def.stream_id()] += task_num;
    auto stream_id = task_def.stream_id();
    if (stream_to_first_task_id_.find(stream_id) == stream_to_first_task_id_.end()) {
      stream_to_first_task_id_[stream_id] = static_cast<uint32_t>(i);
    }
  }
  GELOGD("task_size: %d, stream num: %zu.", task_size, stream_task_num_.size());

  // nullptr means it is static model
  if (reusable_stream_allocator_ == nullptr) {
    reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    GE_CHECK_NOTNULL(reusable_stream_allocator_);
  }

  return SUCCESS;
}

Status DavinciModel::InitRuntimeResource() {
  // create model_handle to load model
  GE_CHK_RT_RET(rtModelCreate(&rt_model_handle_, 0U));
  GE_CHK_RT_RET(rtSetModelName(rt_model_handle_, name_.c_str()));
  GE_CHK_RT_RET(rtModelGetId(rt_model_handle_, &runtime_model_id_));
  std::vector<int64_t> huge_stream_list;
  (void)AttrUtils::GetListInt(ge_model_, ATTR_MODEL_HUGE_STREAM_LIST, huge_stream_list);
  const auto graph = ge_model_->GetGraph();
  GE_ASSERT_NOTNULL(graph);
  std::map<int64_t, int64_t> stream_id_to_logic_stream_id;
  std::string split_stream_2_logical_stream_str;
  const std::string *split_stream_2_logical_stream_str_ptr = AttrUtils::GetStr(graph, "_split_logic_stream_2_origin_logic_stream");
  if (split_stream_2_logical_stream_str_ptr != nullptr) {
    split_stream_2_logical_stream_str = *split_stream_2_logical_stream_str_ptr;
  }
  GE_ASSERT_SUCCESS(
      TransStrToMap(split_stream_2_logical_stream_str, split_logic_stream_2_origin_logic_stream_));
  const std::set<int64_t> huge_streams(huge_stream_list.begin(), huge_stream_list.end());

  const bool isOverflowDetectionOpen = GetContext().IsOverflowDetectionOpen();
  GE_CHECK_NOTNULL(reusable_stream_allocator_);
  for (uint32_t i = 0U; i < runtime_param_.stream_num; ++i) {
    uint32_t stream_flags = RT_STREAM_PERSISTENT;
    if (huge_streams.count(static_cast<int32_t>(i)) > 0U) {
      GELOGI("Stream %u is huge stream.", i);
      stream_flags |= RT_STREAM_HUGE;
    }
    if (hcom_streams_.count(i) > 0U || hcom_attach_streams_.count(i) > 0U) {
      stream_flags |= RT_STREAM_FORCE_COPY;
    }
    if (isOverflowDetectionOpen) {
      stream_flags |= RT_STREAM_OVERFLOW;
    }

    rtStream_t stream = nullptr;
    uint32_t task_num = 0U;
    // some graphs may have no task such as variable graph
    if (stream_task_num_.find(i) != stream_task_num_.end()) {
      task_num = stream_task_num_[i];
      GELOGD("Stream id: %u, task num: %u.", i, task_num);
    }
    GE_ASSERT_SUCCESS(
        reusable_stream_allocator_->GetOrCreateRtStream(stream, runtime_model_id_, priority_, stream_flags, task_num));

    stream_list_.push_back(stream);
    stream_flag_list_.push_back(stream_flags);
    int32_t rt_stream_id = kInvalidStream;
    (void)rtGetStreamId(stream, &rt_stream_id);
    GELOGI("Logical stream index: %u, rtstream: %d, model: %u, stream flag: %u.",
           i, rt_stream_id, model_id_, stream_flags);
  }

  uint32_t i = 0U;
  if (runtime_param_.notify_types.empty()) {
    runtime_param_.notify_types.resize(runtime_param_.notify_num, RT_NOTIFY_DEFAULT);
  }
  GE_ASSERT_EQ(runtime_param_.notify_num, static_cast<uint32_t>(runtime_param_.notify_types.size()));
  while (i < runtime_param_.notify_num) {
    rtNotify_t rt_notify = nullptr;
    GE_ASSERT_RT_OK(rtNotifyCreate(static_cast<int32_t>(device_id_), &rt_notify));
    notify_list_.push_back(rt_notify);
    ++i;
  }
  i = 0U;
  while (i < runtime_param_.event_num) {
    rtEvent_t rt_event = nullptr;
    GE_CHK_RT_RET(rtEventCreateWithFlag(&rt_event, static_cast<uint32_t>(RT_EVENT_WITH_FLAG)));
    event_list_.push_back(rt_event);
    ++i;
  }
  label_list_.resize(static_cast<size_t>(runtime_param_.label_num), nullptr);

  GE_CHK_STATUS_RET(SetModelConfig(), "[Call][SetModelConfig] failed, model_id: %u.", model_id_);
  SetStaticModelShapeConfig();
  return SUCCESS;
}

Status DavinciModel::InitSupplyResource() {
  // op debug register
  GE_CHK_STATUS_RET(OpDebugRegister(), "[Call][OpDebugRegister] failed, model_id: %u.", model_id_);

  // malloc mem for overflow detetcion
  GE_CHK_RT_RET(rtCtxGetOverflowAddr(&globalworkspace_overflow_addr_));
  return SUCCESS;
}

// save specify attr values of op, such as ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
// it will save more attr values in the future
void DavinciModel::SaveSpecifyAttrValues(const OpDescPtr &op_desc) {
  std::vector<std::string> value;
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value)) {
    op_name_to_attrs_[op_desc->GetName()] = { {ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value} };
    GELOGD("Get op:%s attr:%s success.", op_desc->GetName().c_str(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str());
  }
}

Status DavinciModel::UpdateSessionId(const uint64_t session_id) {
  GE_CHECK_NOTNULL(ge_model_);
  if (!AttrUtils::SetInt(ge_model_, MODEL_ATTR_SESSION_ID, static_cast<int64_t>(session_id))) {
    GELOGW("Set attr[%s] failed in updating session_id.", MODEL_ATTR_SESSION_ID.c_str());
  }
  bin_kernel_handle_.SetSessionId(session_id);
  GELOGD("Update session id[%" PRIu64 "].", session_id);
  return SUCCESS;
}

Status DavinciModel::RestoreDeviceVarMem(const std::vector<NodePtr> &variable_nodes, const ModelParam &param) {
  if (variable_nodes.empty() || domi::GetContext().is_online_model) {
    return SUCCESS;
  }
  const auto &rt_var_manager = gert::ModelRtVarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(rt_var_manager);
  GE_ASSERT_SUCCESS(rt_var_manager->Init(device_id_, runtime_param_.logic_var_base, runtime_param_.var_size,
                                         param.external_var_addr_, param.external_var_size_),
                    "Failed to init runtime var_manager.");
  GE_ASSERT_SUCCESS(
      rt_var_manager->RestoreDeviceVariables(variable_nodes, runtime_param_.graph_id, GetDeviceId(), false),
      "Restore device variables failed.");
  return SUCCESS;
}

Status DavinciModel::InitIoNodes(const ComputeGraphPtr &compute_graph, std::vector<NodePtr> &variable_nodes) {
  uint32_t data_op_index = 0U;
  std::map<uint32_t, OpDescPtr> index_to_data;
  std::vector<OpDescPtr> output_op_list;
  std::set<uint64_t> input_outside_addrs;
  std::set<uint64_t> output_outside_addrs;
  std::vector<NodePtr> queue_data_nodes;
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      GE_CHK_STATUS_RET_NOLOG(InitDataOp(compute_graph, node, data_op_index, index_to_data, input_outside_addrs));
      data_dumper_.SaveDumpInput(node);
    } else if (op_desc->GetType() == NETOUTPUT) {
      GE_CHK_STATUS_RET_NOLOG(InitNetOutput(compute_graph, node, output_op_list, output_outside_addrs));
      GE_CHK_STATUS_RET_NOLOG(InitRealSizeAndShapeInfo(compute_graph, node));
    } else if ((op_desc->GetType() == VARIABLE) || (op_desc->GetType() == CONSTANTOP) || (op_desc->GetType() == CONSTPLACEHOLDER)) {
      variable_nodes.emplace_back(node);
    } else if (op_desc->GetType() == QUEUE_DATA) {
      queue_data_nodes.emplace_back(node);
    } else if (HcomOmeUtil::IsHCOMOp(op_desc->GetType()) ||
               HcomOmeUtil::IsHorovodOp(op_desc->GetType())) { // CheckHasHcomOp
      CollectHcomRelatedStreams(op_desc);
    } else {
      // do nothing
    }
  }

  DelDependentHcclStreams(compute_graph);
  GE_CHK_STATUS_RET_NOLOG(GenInputOutputInfo(index_to_data, output_op_list));
  GE_CHK_STATUS_RET_NOLOG(InitQueueDataNodes(queue_data_nodes, data_op_index, input_outside_addrs));
  GE_CHK_STATUS_RET_NOLOG(GenMemAllocations(index_to_data, output_op_list));
  return SUCCESS;
}

Status DavinciModel::SetExternalPath(const ComputeGraphPtr &compute_graph) {
  if (file_constant_weight_dir_.empty()) {
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(FileConstantUtils::SetExternalPath(compute_graph, file_constant_weight_dir_),
                    "Failed to set external path:%s.", file_constant_weight_dir_.c_str());
  return SUCCESS;
}

Status DavinciModel::InitVarResourceIfNeeded(const ModelParam &param, bool &var_resource_inited) {
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  if (var_resource_inited || VarManager::Instance(session_id_)->IsVarResourceInited()) {
    return SUCCESS;
  }

  const auto &rt_var_manager = gert::ModelRtVarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(rt_var_manager);
  GE_ASSERT_SUCCESS(rt_var_manager->Init(device_id_, runtime_param_.logic_var_base, runtime_param_.var_size,
                                         param.external_var_addr_, param.external_var_size_),
                    "Failed to init runtime var_manager.");
  var_resource_inited = true;
  return SUCCESS;
}

Status DavinciModel::ProcessFileConstantNode(const NodePtr &node, const ModelParam &param, bool &var_resource_inited,
                                             bool &is_weight_combined, std::string &combined_weight_file,
                                             std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size) {
  GE_CHECK_NOTNULL(node);
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetType() != FILECONSTANT) {
    return SUCCESS;
  }

  if (!var_resource_inited) {
    GE_CHK_STATUS_RET(InitVarResourceIfNeeded(param, var_resource_inited));
  }

  if (VarManager::Instance(session_id_)->IsVarExist(op_desc->GetName(), op_desc->GetOutputDesc(0U))) {
    return SUCCESS;
  }

  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  if (v_output_offset.size() != 1U) {
    GELOGE(PARAM_INVALID, "FileConstant %s output offsets invalid: v_output_offset size = %zu, expect 1.",
           op_desc->GetNamePtr(), v_output_offset.size());
    return PARAM_INVALID;
  }

  const auto &tensor_desc = op_desc->GetOutputDescPtr(0U);
  GE_CHECK_NOTNULL(tensor_desc);
  int64_t weights_size = 0;
  GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weights_size),
                    "Failed to get file constant tensor size, node: %s.", op_desc->GetNamePtr());

  const auto v_output_size = ModelUtils::GetOutputSize(op_desc);
  if (v_output_size.empty()) {
    GELOGE(PARAM_INVALID, "Output size is empty");
    return PARAM_INVALID;
  }

  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  GE_CHK_STATUS_RET(FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map_, file_path, offset, length),
                    "Failed to get file path.");
  if (combined_weight_file.empty()) {
    combined_weight_file = file_path;
    is_weight_combined = true;
  } else if (!combined_weight_file.empty() && combined_weight_file != file_path) {
    is_weight_combined = false;
  }

  const int64_t alloc_size = std::max(v_output_size[0], std::max(weights_size, static_cast<int64_t>(length)));
  GELOGD("FileConstant weight size is:%" PRId64 " output size:%zu alloc_size:%" PRId64 "",
         std::max(weights_size, static_cast<int64_t>(length)), v_output_size[0], alloc_size);
  node_to_offset_and_size[node] = std::make_pair(offset, alloc_size);

  return SUCCESS;
}

Status DavinciModel::AllocateCombinedWeightMemory(const std::string &combined_weight_file,
                                                  const void *&real_dev_addr,
                                                  int64_t &file_size,
                                                  bool &is_user_mem) {
  // 获取归一权重文件名用于查找用户内存
  std::string file_constant_file_name;
  std::string file_dir;
  SplitFilePath(combined_weight_file, file_dir, file_constant_file_name);
  (void)file_dir;

  // 查询用户是否为归一权重文件提供了内存
  const auto user_device_mem = GetFileConstantUserDeviceMem(file_constant_file_name);
  if (user_device_mem != nullptr) {
    // 用户提供了内存，使用用户内存
    GE_ASSERT_NOTNULL(user_device_mem->device_mem,
                      "Error: The address set by the user via aclmdlSetExternalWeightAddress"
                      " for the file %s is a null pointer.", file_constant_file_name.c_str());
    file_size = user_device_mem->mem_size;
    real_dev_addr = user_device_mem->device_mem;
    is_user_mem = true;
    GELOGI("GE found user device memory from file: %s, addr: %p", file_constant_file_name.c_str(), real_dev_addr);
    return SUCCESS;
  }

  // 用户未提供内存，GE自行分配
  const std::string real_path = RealPath(combined_weight_file.c_str());
  std::ifstream ifs(real_path, std::ifstream::binary);
  if (!ifs.is_open()) {
    GELOGE(FAILED, "[Open][File] %s failed.", real_path.c_str());
    REPORT_PREDEFINED_ERR_MSG("E13001", std::vector<const char *>({"file", "errmsg"}),
                              std::vector<const char *>({real_path.c_str(), "Open file failed"}));
    return FAILED;
  }
  ifs.seekg(0, std::ios::end);
  file_size = static_cast<int64_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  real_dev_addr = MallocFileConstantMem(static_cast<size_t>(file_size));
  if (real_dev_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "MallocFileConstantMem fail, weights_size:%" PRId64 ", model_id:%u, check invalid",
                         file_size, model_id_);
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Alloc][Memory] for weight failed. size:%" PRId64 ", model_id:%u",
           file_size, model_id_);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  dev_mem_statistic_.alloc_size += file_size;
  is_user_mem = false;
  GELOGI("GE allocated device memory for combined weights, addr: %p, size: %" PRId64, real_dev_addr, file_size);

  size_t left_size = file_size;
  Status ret = FileConstantUtils::CopyOneWeightFromFileWithFilehandler(real_dev_addr, real_path, 0, file_size, left_size, ifs);
  ifs.close();
  if (ret != SUCCESS) {
    // 复制失败，需要释放已分配的内存
    auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
    GE_CHK_STATUS(mem_instance.FreeMemory(const_cast<void*>(real_dev_addr), device_id_),
                  "Failed to free memory after copy failure");
    dev_mem_statistic_.alloc_size -= file_size;  // 从统计中移除
    GELOGE(ret, "copy weight from file[%s] to addr[%p] failed, file size[%" PRId64 "]", real_path.c_str(), real_dev_addr, file_size);
    return ret;
  }

  return SUCCESS;
}

Status DavinciModel::MapNodeAddressesToCombinedWeight(const std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size,
                                                      const void *base_addr,
                                                      int64_t file_size,
                                                      const std::string &file_path) {
  for (const auto &item : node_to_offset_and_size) {
    const auto &op_desc = item.first->GetOpDesc();
    const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
    if (v_output_offset.size() != 1U) {
      GELOGE(PARAM_INVALID, "FileConstant %s output offsets invalid: v_output_offset size = %zu, expect 1.",
             op_desc->GetNamePtr(), v_output_offset.size());
      return PARAM_INVALID;
    }
    int64_t weights_size;
    int64_t weights_offset = item.second.first;
    const auto &tensor_desc = op_desc->GetOutputDescPtr(0U);
    GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weights_size),
                      "Failed to get file constant tensor size, node: %s.", op_desc->GetNamePtr());
    bool not_exceeds = (file_size >= weights_offset) && (file_size - weights_offset >= weights_size);
    GE_CHK_BOOL_RET_STATUS(not_exceeds, PARAM_INVALID,
                          "Weight offset[%" PRId64 "], size[%" PRId64 "] exceeds the file size[%" PRId64 "] for file: %s.",
                          weights_offset, weights_size, file_size, file_path.c_str());
    runtime_param_.fileconstant_addr_mapping[v_output_offset[0]] = reinterpret_cast<uintptr_t>(base_addr) + weights_offset;
    VarManager::Instance(session_id_)->SetVarIsReady(op_desc->GetName(), *tensor_desc, device_id_);
    GELOGI("FileConstant node %s malloc device memory (addr: %p, offset: %" PRId64 ")",
           op_desc->GetNamePtr(), runtime_param_.fileconstant_addr_mapping[v_output_offset[0]], v_output_offset[0]);
  }
  return SUCCESS;
}

Status DavinciModel::HandleCombinedWeights(const std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size,
                                           const std::string &combined_weight_file) {
  GELOGI("Handle combined weights in file[%s].", combined_weight_file.c_str());
  if (node_to_offset_and_size.empty()) {
    GELOGI("There is no node need alloc memory.");
    return SUCCESS;
  }

  const void *real_dev_addr = nullptr;
  bool is_user_mem = false;
  int64_t file_size = 0;
  Status ret = AllocateCombinedWeightMemory(combined_weight_file, real_dev_addr, file_size, is_user_mem);
  GE_CHK_STATUS_RET(ret, "Failed to allocate combined weight memory for file: %s", combined_weight_file.c_str());

  std::string file_constant_file_name;
  std::string file_dir;
  SplitFilePath(combined_weight_file, file_dir, file_constant_file_name);

  ret = MapNodeAddressesToCombinedWeight(node_to_offset_and_size, real_dev_addr, file_size, file_constant_file_name);
  if (ret != SUCCESS) {
    if (!is_user_mem && real_dev_addr != nullptr) {
      // 复制失败，需要释放已分配的内存
      auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
      (void)mem_instance.FreeMemory(const_cast<void*>(real_dev_addr), device_id_);
      dev_mem_statistic_.alloc_size -= file_size;  // 从统计中移除
    }
    GELOGE(ret, "Failed to map node addresses to combined weight for file: %s", combined_weight_file.c_str());
    return ret;
  }

  auto device_id = device_id_;
  if (is_user_mem) {
    external_weight_combined_mem_addr_ = std::unique_ptr<void, std::function<void(void*)>>(
      const_cast<void*>(real_dev_addr),
      [file_constant_file_name](void* ptr) {
        GELOGI("User-managed combined weight memory at %p for file %s, no need to free by GE.",
               ptr, file_constant_file_name.c_str());
      });
  } else {
    external_weight_combined_mem_addr_ = std::unique_ptr<void, std::function<void(void*)>>(
      const_cast<void*>(real_dev_addr),
      [device_id](void* ptr) {
        if (ptr != nullptr) {
          auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
          (void)mem_instance.FreeMemory(ptr, device_id);
          GELOGI("Freed GE-managed combined weight memory at %p", ptr);
        }
      });
  }
  return SUCCESS;
}

Status DavinciModel::HandleIndividualWeights(const std::map<NodePtr, std::pair<size_t, int64_t>> &node_to_offset_and_size) {
  for (const auto &item : node_to_offset_and_size) {
    const auto &op_desc = item.first->GetOpDesc();
    const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
    if (v_output_offset.size() != 1U) {
      GELOGE(PARAM_INVALID, "FileConstant %s output offsets invalid: v_output_offset size = %zu, expect 1.",
             op_desc->GetNamePtr(), v_output_offset.size());
      return PARAM_INVALID;
    }
    const int64_t orig_output_offset = v_output_offset[0];
    const auto &tensor_desc = op_desc->GetOutputDescPtr(0U);
    GE_CHECK_NOTNULL(tensor_desc);
    int64_t weights_size = 0;
    GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weights_size),
                      "Failed to get file constant tensor size, node: %s.", op_desc->GetNamePtr());
    void *user_device_mem = nullptr;
    GE_ASSERT_SUCCESS(GetUserDeviceMemForFileConstant(op_desc, weights_size, user_device_mem),
                      "node %s get user address failed", op_desc->GetName().c_str());

    if (user_device_mem != nullptr) {
      runtime_param_.fileconstant_addr_mapping[v_output_offset[0]] = reinterpret_cast<uintptr_t>(user_device_mem);
      VarManager::Instance(session_id_)->SetVarIsReady(op_desc->GetName(), *tensor_desc, device_id_);
      GELOGI("FileConstant node %s found user device memory (addr:%p, logic addr:%" PRId64 "), no need to copy weight,"
             " set var ready", op_desc->GetNamePtr(), user_device_mem, v_output_offset[0]);
      continue;
    }

    const auto alloc_size = item.second.second;
    const uintptr_t real_dev_addr =
        static_cast<uintptr_t>(PtrToValue(MallocFileConstantMem(static_cast<size_t>(alloc_size))));
    if (real_dev_addr == 0U) {
      REPORT_INNER_ERR_MSG("E19999", "MallocFileConstantMem fail, weights_size:%" PRId64 ", model_id:%u, check invalid",
                           alloc_size, model_id_);
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Alloc][Memory] for weight failed. size:%" PRId64 ", model_id:%u",
             alloc_size, model_id_);
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
    dev_mem_statistic_.alloc_size += static_cast<uint64_t>(alloc_size);
    runtime_param_.fileconstant_addr_mapping[orig_output_offset] = real_dev_addr;
    GELOGI("FileConstant node %s malloc device memory (addr: %p, offset: %" PRId64 ")",
           op_desc->GetNamePtr(), real_dev_addr, orig_output_offset);
  }

  return SUCCESS;
}

Status DavinciModel::PreProcessFileConstants(const ComputeGraphPtr &compute_graph, const ModelParam &param) {
  GE_CHK_STATUS_RET(SetExternalPath(compute_graph));

  bool var_resource_inited = false;
  const auto &nodes = compute_graph->GetAllNodes();
  bool is_weight_combined = false;
  std::string combined_weight_file;
  std::map<NodePtr, std::pair<size_t, int64_t>> node_to_offset_and_size;

  for (size_t i = 0UL; i < nodes.size(); ++i) {
    GE_CHK_STATUS_RET(ProcessFileConstantNode(nodes.at(i), param, var_resource_inited, is_weight_combined,
                      combined_weight_file, node_to_offset_and_size));
  }

  if (is_weight_combined) {
    GE_CHK_STATUS_RET(HandleCombinedWeights(node_to_offset_and_size, combined_weight_file));
  } else {
    GE_CHK_STATUS_RET(HandleIndividualWeights(node_to_offset_and_size));
  }

  return SUCCESS;
}

Status DavinciModel::GetUserDeviceMemForFileConstant(const OpDescPtr &op_desc, size_t weights_size,
                                                     void *&user_mem) const {
  user_mem = nullptr;
  if (file_constant_user_device_mems_.empty()) {
    return SUCCESS;
  }
  const auto tensor_desc = op_desc->GetOutputDescPtr(0U);
  GE_ASSERT_NOTNULL(tensor_desc);
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  GE_CHK_STATUS_RET(FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map_, file_path, offset, length),
                    "Failed to get file path, FileConstant node: %s.", op_desc->GetNamePtr());
  std::string file_constant_file_name;
  std::string file_dir;
  SplitFilePath(file_path, file_dir, file_constant_file_name);
  (void)file_dir;
  const auto user_device_mem = GetFileConstantUserDeviceMem(file_constant_file_name);
  if (user_device_mem == nullptr) {
    GELOGI("No user device memory found for FileConstant node %s. File name: %s", op_desc->GetNamePtr(),
           file_constant_file_name.c_str());
    return SUCCESS;
  }
  // It's unlikely, since aclmdlSetExternalWeightAddress has already verified the device_mem.
  GE_ASSERT_NOTNULL(user_device_mem->device_mem, "Error: The address set by the user via aclmdlSetExternalWeightAddress"
                    " for the file %s is a null pointer.", file_constant_file_name.c_str());
  // The offset is non-zero only when multiple FileConstants share one weight file,
  // which is a rarely used pattern.
  GE_CHECK_GE(user_device_mem->mem_size, offset);
  if (user_device_mem->mem_size - offset < weights_size) {
    std::string reason =
        "The device memory size set by the user via "
        "aclmdlSetExternalWeightAddress for the external weight file is insufficient. "
        "Required: " +
        std::to_string(weights_size) + " bytes, Provided: " + std::to_string(user_device_mem->mem_size - offset) +
        " bytes. External weight - Shape: [" + tensor_desc->GetShape().ToString().c_str() + "], Data type: [" +
        TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str() + "], Offset: [" +
        std::to_string(offset) + "], File name: [" + file_constant_file_name + "], Node name: [" +
        op_desc->GetNamePtr() + "].";
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"aclmdlSetExternalWeightAddress",
                                                         std::to_string(weights_size).c_str(), reason.c_str()}));

    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param] The device memory size set by the user via "
           "aclmdlSetExternalWeightAddress for the external weight file is insufficient. "
           "Required: %zu bytes, Provided: %zu bytes. External weight - Shape: [%s], Data type: [%s], Offset: [%zu], "
           "File name: [%s], Node name: [%s].", weights_size, user_device_mem->mem_size - offset,
           tensor_desc->GetShape().ToString().c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(), offset,
           file_constant_file_name.c_str(), op_desc->GetNamePtr());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  user_mem = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(user_device_mem->device_mem) + offset);
  GELOGI("FileConstant node %s found user device memory (addr: %p, size: %zu). file name: %s, offset: %zu, length: %zu,"
      " weight size: %zu", op_desc->GetNamePtr(), user_mem, user_device_mem->mem_size,
      file_constant_file_name.c_str(), offset, length, weights_size);
  return SUCCESS;
}

void DavinciModel::PrintHcclOps(const std::vector<std::pair<std::string, std::string>> &hccl_ops) const {
  if (!hccl_ops.empty()) {
    std::stringstream ss;
    ss << "print model_id [" << model_id_ << "] all hccl ops, start: ";
    for (const auto &it : hccl_ops) {
      ss << it.first << "[" << it.second << "], ";
    }
    ss << "end.";
    // in case of being truncated out of log limit 1024, set up limit 800
    const size_t max_log_string_len = 800U;
    size_t index = 0U;
    while (index < ss.str().length()) {
      GEEVENT("%s", ss.str().substr(index, max_log_string_len).c_str());
      index += max_log_string_len;
    }
  }
}

///
/// @ingroup ge
/// @brief Travel all nodes and do some init.
/// @param [in] compute_graph: ComputeGraph to load.
/// @return Status
///
Status DavinciModel::InitNodes(const ComputeGraphPtr &compute_graph) {
  GE_TIMESTAMP_START(InitNodes);
  GE_TIMESTAMP_CALLNUM_START(InitTbeHandle);
  using OpDescCall = std::function<Status(DavinciModel *, const OpDescPtr &)>;
  static const std::map<std::string, OpDescCall> op_desc_handle = {
      {STREAMACTIVE, &DavinciModel::InitStreamActive},
      {STREAMSWITCH, &DavinciModel::InitStreamSwitch},
      {LABELSET, &DavinciModel::InitLabelSet},
      {CASE, &DavinciModel::InitCase},
  };
  std::vector<NodePtr> nodes_init_by_thread;
  std::map<std::string, OpDescPtr> variable_by_name;
  std::vector<std::pair<std::string, std::string>> hccl_ops;
  // 建立KernelSo与opName之间的映射关系
  GE_ASSERT_SUCCESS(ge_model_->GetCustAICPUKernelStore().BuildKernelSoToOpNameMap(compute_graph));
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    ge_model_->GetCustAICPUKernelStore().LoadCustAICPUKernelBinToOpDesc(op_desc);
    ge_model_->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(op_desc);
    if (op_desc->GetOpKernelLibName() == ge::kEngineNameHccl) {
      has_hccl_task_ = true;
      hccl_ops.emplace_back(std::make_pair(op_desc->GetName(), op_desc->GetType()));
    }

    std::vector<std::string> hccl_group_id_list;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_HCCL_GROUP_ID_LIST, hccl_group_id_list) && !hccl_group_id_list.empty()) {
      hccl_group_id_set_.insert(hccl_group_id_list.begin(), hccl_group_id_list.end());
    }

    SaveSpecifyAttrValues(op_desc);
    GE_ASSERT_SUCCESS(DisableZeroCopyNode(op_desc));
    operator_list_[op_desc->GetId()] = MakeShared<Operator>(OpDescUtils::CreateOperatorFromNode(node));

    const auto &op_type = op_desc->GetType();
    // 跳过Data类型及IO类型节点
    if (OpTypeUtils::IsDataNode(op_type) || (kIoNodeTypes.count(op_type) > 0U)) {
      continue;
    }

    // 处理 SuperKernel
    if (op_type == "SuperKernel") {
      GELOGI("find sk node %s", op_desc->GetNamePtr());
      ComputeGraphPtr sub_ext_graph = nullptr;
      sub_ext_graph = op_desc->TryGetExtAttr("_sk_sub_graph", sub_ext_graph);
      if (sub_ext_graph == nullptr) {
        ComputeGraphPtr sub_graph = nullptr;
        GELOGI("find sk %s has setgraph", op_desc->GetNamePtr());
        GE_ASSERT_TRUE(AttrUtils::GetGraph(op_desc, "_sk_sub_graph", sub_graph));
        if (sub_graph != nullptr) {
          GELOGI("set extattr to sk %s", op_desc->GetNamePtr());
          GE_ASSERT_TRUE(op_desc->SetExtAttr("_sk_sub_graph", sub_graph));
        }
      }
    }

    // 处理变量节点
    if (op_type == VARIABLE) {
      GE_CHK_STATUS_RET_NOLOG(InitVariable(op_desc, variable_by_name));
      continue;
    }

    // FILECONSTANT/CONSTANTOP节点先on-thread处理
    if ((op_type == FILECONSTANT) || (op_type == CONSTANTOP)) {
      const auto &tensor_desc = op_desc->GetOutputDescPtr(0U);
      GE_CHECK_NOTNULL(tensor_desc);
      GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
      if (op_type == FILECONSTANT && VarManager::Instance(session_id_)->IsVarReady(node->GetName(), *tensor_desc, device_id_)) {
        GELOGD("file constant op:%s is ready", node->GetName().c_str());
        continue;
      }
      nodes_init_by_thread.emplace_back(node);
      continue;
    }

    GE_CHK_STATUS_RET_NOLOG(AllocateResource(*node));

    // 特定算子类型op_desc_handle处理
    const auto it = op_desc_handle.find(op_type);
    if (it != op_desc_handle.end()) {
      if ((it->second)(this, op_desc) != SUCCESS) {
        GELOGE(FAILED, "[Init][Node] failed, Name:%s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      continue;
    }

    GE_CHK_STATUS_RET(ExecutorUtils::LoadAtomicWorkspace(op_desc), "LoadAtomicWorkspace failed for [%s(%s)].",
                      op_desc->GetName().c_str(), op_type.c_str());

    GE_CHK_STATUS_RET(InitNoTaskAndDumpNeededNode(op_desc), "Init no task and dump needed node: %s failed.",
                      op_desc->GetName().c_str());
    GE_TIMESTAMP_RESTART(InitTbeHandle);
    // 临时修改 ffts+ 是日落特性，走老的注册流程，特性下线后删除
    bool is_ffts = false;
    (void)AttrUtils::GetBool(op_desc, "_mix_with_enhanced_kernel", is_ffts);
    if (is_ffts) {
      GE_CHK_STATUS_RET(InitTbeHandle(op_desc), "[Init][TbeHandle] failed. op: %s", op_desc->GetName().c_str());
    }
    GE_TIMESTAMP_ADD(InitTbeHandle);
  }

  // 临时修改 由于fusionTask暂时不支持aclrtlaunch接口需要使用InitTbeHandle做初始化，aclrtlaunch接口支持后删除
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  if (model_task_def != nullptr) {
    for (int32_t task_index = 0; task_index < model_task_def->task_size(); ++task_index) {
      const auto &task_def = model_task_def->task(task_index);
      if (task_def.type() == static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FUSION_KERNEL)) {
        auto op_desc = GetOpByIndex(task_def.fusion_task().op_index());
        GE_CHECK_NOTNULL(op_desc);
        GE_CHK_STATUS_RET(InitTbeHandle(op_desc), "[Init][TbeFusionTaskHandle] failed. op: %s",
                          op_desc->GetName().c_str());
      }
    }
  }

  if (!nodes_init_by_thread.empty()) {
    ThreadPool thread_pool("ge.fileconst", kDefaultThreadNum, false);
    std::vector<std::future<Status>> fut_rets;
    auto thread_local_context = GetThreadLocalContext();
    auto error_manager_context = error_message::GetErrMgrContext();

    for(const auto &node : nodes_init_by_thread) {
      auto op_type = node->GetType();
      auto fut = thread_pool.commit([this, node, op_type, thread_local_context, error_manager_context]() -> Status {
        GetThreadLocalContext() = thread_local_context;
        error_message::SetErrMgrContext(error_manager_context);
        GE_CHK_RT_RET(aclrtSetDevice(device_id_));
        GE_MAKE_GUARD(reset_device, [this]() { GE_CHK_RT(aclrtResetDevice(device_id_)); });
        if (op_type == FILECONSTANT) {
          GE_CHK_STATUS_RET_NOLOG(InitFileConstant(node));
        } else if (op_type == CONSTANTOP) {
          auto op_desc = node->GetOpDesc();
          GE_CHK_STATUS_RET_NOLOG(InitConstant(op_desc));
        }
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    for (auto &fut : fut_rets) {
      GE_CHK_STATUS_RET(fut.get(), "Failed to init nodes, graph:%s", compute_graph->GetName().c_str());
    }
  }

  PrintHcclOps(hccl_ops);
  GE_CHK_STATUS_RET(SetDataDumperArgs(compute_graph, variable_by_name), "[Set][DataDumperArgs] failed, graph: %s",
                    compute_graph->GetName().c_str());
  GE_TIMESTAMP_CALLNUM_EVENT_END(InitTbeHandle, "GraphLoader::InitTbeHandle");
  GE_TIMESTAMP_EVENT_END(InitNodes, "DavinciModel::InitNodes");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Data Op Initialize.
/// @param [in] ComputeGraphPtr: root graph of the model.
/// @param [in] NodePtr: Data Op.
/// @param [in/out] data_op_index: index of courrent count.
/// @param [in/out] index_to_data: Data ordered by index.
/// @return Status
///
Status DavinciModel::InitDataOp(const ComputeGraphPtr &graph, const NodePtr &node, uint32_t &data_op_index,
                                std::map<uint32_t, OpDescPtr> &index_to_data, std::set<uint64_t> &input_outside_addrs) {
  // op_desc Checked by Init: Data, valid.
  const auto op_desc = node->GetOpDesc();
  if (node->GetOwnerComputeGraph() != graph) {
    GELOGI("Skip Data node: %s in subgraph.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (node->GetOwnerComputeGraphBarePtr()->GetParentNode() != nullptr) {
    if (std::strcmp(node->GetTypePtr(),REFDATA) == 0) {
      GELOGD("Skip RefData node: %s in subgraph %s.", op_desc->GetName().c_str(),
        node->GetOwnerComputeGraphBarePtr()->GetName().c_str());
      return SUCCESS;
    }
  }

  uint32_t data_index = data_op_index++;
  const auto &index_attr = (GraphUtils::FindRootGraph(graph) == graph) ? ATTR_NAME_INDEX : ATTR_NAME_PARENT_NODE_INDEX;
  if (AttrUtils::GetInt(op_desc, index_attr, data_index)) {
    GELOGD("Get new index %u, old %u", data_index, data_op_index - 1U);
  }
  GELOGI("Init data node: %s, index: %u.", op_desc->GetName().c_str(), data_index);

  const auto &anchor = node->GetOutDataAnchor(0);
  if ((anchor != nullptr) && (anchor->GetFirstPeerAnchor() != nullptr) &&
      (anchor->GetFirstPeerAnchor()->GetOwnerNode() != nullptr)) {
    const auto &node_desc = anchor->GetFirstPeerAnchor()->GetOwnerNode()->GetOpDesc();
    const size_t anchor_idx = static_cast<size_t>(anchor->GetFirstPeerAnchor()->GetIdx());
    std::vector<int64_t> op_max_size;
    if (AttrUtils::GetListInt(node_desc, "_op_max_size", op_max_size) && (op_max_size.size() > anchor_idx)) {
      (void)AttrUtils::SetInt(op_desc, "_op_max_size", op_max_size[anchor_idx]);
    }
  }

  index_to_data[data_index] = op_desc;
  if (feature_base_refreshable_ && !support_extend_memory_full_) {
    return SUCCESS;
  }

  // todo: RefData 被识别不能零拷贝的可定位手段
  if (op_desc->GetType() == REFDATA) {
    const auto &offsets = op_desc->GetOutputOffset();
    GE_ASSERT_TRUE(!offsets.empty());
    const auto virtual_addr_list = ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc);
    GE_ASSERT_TRUE(!virtual_addr_list.empty());
    const uint64_t virtual_addr = virtual_addr_list[kDataIndex];
    (void)copy_only_addrs_.refdata_virtual_addrs.insert(virtual_addr);
    GELOGI("Refdata %s output offset %lld, virtual_addr:%p", op_desc->GetName().c_str(), offsets.at(0),
           ValueToPtr(virtual_addr));
  }

  NamedAttrs align_attr;
  if (AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr)) {
    GELOGD("Input[%u] has aligned attr", data_index);
    align_attrs_[data_index] = align_attr;
  }

  GE_CHK_STATUS_RET_NOLOG(InitInputZeroCopy(op_desc, data_index, input_outside_addrs));
  return SUCCESS;
}

Status DavinciModel::InitNoTaskAndDumpNeededNode(const OpDescPtr &op_desc) {
  if (IsNoTaskAndDumpNeeded(op_desc)) {
    GELOGD("node[%s] has no task, saving op_desc and addr for dump.", op_desc->GetName().c_str());
    const auto input_data_addrs = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc);
    const auto output_data_addrs = ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc);
    const auto workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param_, op_desc);
    std::vector<uint64_t> device_addrs;
    (void)device_addrs.insert(device_addrs.cend(), input_data_addrs.cbegin(), input_data_addrs.cend());
    (void)device_addrs.insert(device_addrs.cend(), output_data_addrs.cbegin(), output_data_addrs.cend());
    (void)device_addrs.insert(device_addrs.cend(), workspace_data_addrs.cbegin(), workspace_data_addrs.cend());
    const size_t addr_size = kAddrSize * device_addrs.size();
    void *addr = MallocDynamicMemory(addr_size);
    GE_ASSERT_NOTNULL(addr);
    saved_task_addrs_[op_desc] = addr;

    GE_CHK_RT_RET(rtMemcpy(addr, addr_size, device_addrs.data(), addr_size, RT_MEMCPY_HOST_TO_DEVICE));
  }

  return SUCCESS;
}

Status DavinciModel::InitInputZeroCopy(const OpDescPtr &op_desc,
                                       const uint32_t data_index,
                                       std::set<uint64_t> &input_outside_addrs) {
  // Make information for copy input data.
  const auto output_size_list = ModelUtils::GetOutputSize(op_desc);
  const auto virtual_addr_list = ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc);
  const auto virtual_addr_list_no_offset = ModelUtils::GetOutputDataAddrsValue(runtime_param_, op_desc);
  const auto output_offset_list = op_desc->GetOutputOffset();
  if (output_size_list.empty() || virtual_addr_list.empty() ||
      virtual_addr_list_no_offset.empty() ||
      (output_size_list.size() != virtual_addr_list.size()) ||
      (output_offset_list.size() != virtual_addr_list.size())) {
    REPORT_INNER_ERR_MSG(
        "E19999", "Check data fail in op:%s(%s), output_desc size:%zu output addr size:%zu output offset size:%zu "
        "not equal or has empty, model_id:%u",
        op_desc->GetName().c_str(), op_desc->GetType().c_str(),
        output_size_list.size(), virtual_addr_list.size(), output_offset_list.size(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] Data[%s] init failed: output size is %zu, "
           "virtual_addr size is %zu, offset size is %zu.", op_desc->GetName().c_str(), output_size_list.size(),
           virtual_addr_list.size(), output_offset_list.size());
    return PARAM_INVALID;
  }

  bool fusion_flag = false;
  ZeroCopyOffset zero_copy_offset;
  const int64_t data_size = output_size_list[kDataIndex];
  const auto ret = zero_copy_offset.InitInputDataInfo(data_size, ValueToPtr(virtual_addr_list[kDataIndex]),
                                                      op_desc, fusion_flag);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Init][DataInfo] of input_info %s failed.", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  const uint64_t virtual_addr = virtual_addr_list[kDataIndex];
  if (input_outside_addrs.count(virtual_addr) == 0U) {
    const int64_t output_offset = output_offset_list.at(kDataIndex);
    GE_RETURN_IF_ERROR(zero_copy_offset.SetInputOutsideAddrs(output_offset, static_cast<uintptr_t>(virtual_addr),
                                                             fusion_flag, real_virtual_addrs_));
    (void)input_outside_addrs.insert(virtual_addr);
  }
  const uint64_t virtual_addr_no_offset = virtual_addr_list_no_offset[kDataIndex];
  if (input_outside_addrs.count(virtual_addr_no_offset) == 0U) {
    const int64_t output_offset = output_offset_list.at(kDataIndex);
    GE_RETURN_IF_ERROR(zero_copy_offset.SetInputOutsideAddrs(output_offset,
                                                             static_cast<uintptr_t>(virtual_addr_no_offset),
                                                             fusion_flag, real_virtual_addrs_));
    (void)input_outside_addrs.insert(virtual_addr_no_offset);
  }
  input_data_info_[data_index] = zero_copy_offset;

  if (ModelUtils::IsReuseZeroCopyMemory()) {
    bool is_zero_copy_block = false;
    (void)ge::AttrUtils::GetBool(op_desc->GetOutputDescPtr(kDataIndex), ATTR_IS_ZERO_COPY_BLOCK, is_zero_copy_block);
    if (!is_zero_copy_block) {
      GELOGI("[ZCPY][disable zero copy] addrs %" PRIu64 " of data %s, model_id:%u.",
          virtual_addr, op_desc->GetName().c_str(), model_id_);
      const std::lock_guard<std::mutex> lk(outside_addrs_mutex_);
      GE_ASSERT_SUCCESS(copy_only_addrs_.Insert(virtual_addr));
    }
  }

  bool is_no_tiling = false;
  (void)AttrUtils::GetBool(op_desc->GetOutputDescPtr(kDataIndex), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
  input_no_tiling_flag_.push_back(is_no_tiling);
  if (is_no_tiling) {
    has_no_tiling_input_ = true;
  }
  return SUCCESS;
}

Status DavinciModel::GenFmMemAllocations() {
  fm_mem_allocations_start_id_ = logical_mem_allocations_.size();

  for (const auto &mem_info : runtime_param_.fm_memory_infos) {
    refreshable_fm_index_and_allocation_ids_.emplace_back(
        std::make_pair(static_cast<uint32_t>(logical_fm_mem_allocations_size_),
        static_cast<uint32_t>(logical_mem_allocations_.size())));

    MemAllocation fm_mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                       PtrToValue(mem_info.memory_base),
                                       static_cast<uint64_t>(mem_info.memory_size),
                                       ge::MemAllocation::Type::FEATURE_MAP,
                                       static_cast<uint32_t>(logical_fm_mem_allocations_size_),
                                       kFmMemType, 0UL, 0UL};
    GELOGI("[mem allocation][feature map] model_id %u, %s.", model_id_, fm_mem_allocation.ToString().c_str());
    logical_mem_allocations_.emplace_back(fm_mem_allocation);
    ++logical_fm_mem_allocations_size_;
  }
  return SUCCESS;
}

Status DavinciModel::GenFixedFmMemAllocations() {
  fixed_fm_mem_allocations_start_id_ = logical_mem_allocations_.size();
  for (const auto &mem_info : runtime_param_.fixed_fm_memory_infos) {
    fixed_fm_index_and_allocation_ids_.emplace_back(
        std::make_pair(static_cast<uint32_t>(logical_fixed_fm_mem_allocations_size_),
        static_cast<uint32_t>(logical_mem_allocations_.size())));
    MemAllocation fm_mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                       PtrToValue(mem_info.memory_base),
                                       static_cast<uint64_t>(mem_info.memory_size),
                                       ge::MemAllocation::Type::FIXED_FEATURE_MAP,
                                       static_cast<uint32_t>(logical_fixed_fm_mem_allocations_size_),
                                       kFmMemType, 0UL, 0UL};
    GELOGI("[mem allocation][fixed feature map] model_id %u, %s.", model_id_, fm_mem_allocation.ToString().c_str());
    logical_mem_allocations_.emplace_back(fm_mem_allocation);
    ++logical_fixed_fm_mem_allocations_size_;
  }
  return SUCCESS;
}

void DavinciModel::ConstructFixedFmActiveMemBase() {
  for (const auto &it : fixed_fm_index_and_allocation_ids_) {
    allocation_ids_to_active_base_addr_[static_cast<size_t>(it.second)] =
        PtrToValue(runtime_param_.fixed_fm_memory_infos[static_cast<size_t>(it.first)].memory_base);
    GELOGI("[ActiveMemBase][FIXED], model_id:%u, id:%u, active mem base:0x%" PRIx64 ", fm_idx:%u.",
           model_id_, it.second, allocation_ids_to_active_base_addr_[static_cast<size_t>(it.second)], it.first);
  }
}

///
/// @ingroup ge
/// @brief Sort Data op list by index.
/// @param [in] index_to_data: map of Data Op.
/// @param [in] output_op_list: list of NetOutput op.
/// @return Status
///
Status DavinciModel::GenInputOutputInfo(const std::map<uint32_t, OpDescPtr> &index_to_data,
                                        const std::vector<OpDescPtr> &output_op_list) {
  GELOGD("Data node size: %zu, NetOutput node size: %zu.", index_to_data.size(), output_op_list.size());
  for (auto &item : index_to_data) {
    const auto output_addrs = ModelUtils::GetOutputAddrsValue(runtime_param_, item.second);
    GELOGD("Data node is: %s, output addr size: %zu", item.second->GetName().c_str(), output_addrs.size());
    input_addrs_list_.emplace_back(output_addrs);

    GE_CHK_STATUS_RET(InitAippInfo(item.first, item.second),
                      "[Init][AippInfo] failed, node: %s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(InitAippType(item.first, item.second, index_to_data),
                      "[Init][AippType] failed, node: %s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(AippUtils::SetAippInputOutputInfoFromOpDesc(
                      item.second, item.first, orig_input_info_, aipp_dims_info_),
                      "[Init][SetAippInputOutputInfoFromOpDesc] failed, node: %s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(InitInputDescInfo(item.second),
                      "[Init][InputDescInfo] failed, node: %s", item.second->GetName().c_str());
    if (item.second->GetType() == AIPP_DATA_TYPE) {
      GELOGI("This is dynamic aipp model, Node: %s", item.second->GetName().c_str());
      is_dynamic_aipp_ = true;
    }
  }

  std::vector<std::string> out_node_name;
  (void)AttrUtils::GetListStr(ge_model_, ATTR_MODEL_OUT_NODES_NAME, out_node_name);
  GELOGD("Output node size: %zu, out nodes name is: %zu", output_op_list.size(), out_node_name.size());
  for (const auto &op_desc : output_op_list) {
    const auto input_addrs = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc);
    GELOGD("NetOutput node is: %s, input addr size: %zu", op_desc->GetName().c_str(), input_addrs.size());
    output_addrs_list_.emplace_back(input_addrs);

    bool getnext_sink_dynamic = false;
    if (AttrUtils::GetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, getnext_sink_dynamic) && getnext_sink_dynamic) {
      GELOGI("ATTR_GETNEXT_SINK_DYNMAIC has been set and is true, node: %s", op_desc->GetName().c_str());
      is_getnext_sink_dynamic_ = true;
    }

    std::vector<std::string> shape_info;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      (void)dynamic_output_shape_info_.insert(dynamic_output_shape_info_.cend(), shape_info.cbegin(),
                                              shape_info.cend());
    }

    if (InitOutputTensorInfo(op_desc) != SUCCESS) {
      return INTERNAL_ERROR;
    }

    GE_CHK_STATUS_RET(InitOutputDescInfo(op_desc, out_node_name),
                      "[Init][OutputDescInfo] failed, node: %s", op_desc->GetName().c_str());
  }

  return SUCCESS;
}

void DavinciModel::PrintNoFrozenInputIndexes() {
  std::string input_indexes_nofrozen = "";
  std::string refreshable_ids_nofrozen_str = "";
  for (size_t i = 0; i < zero_copy_input_indexes_no_frozen_.size(); i++) {
    input_indexes_nofrozen += std::to_string(zero_copy_input_indexes_no_frozen_[i]);
    input_indexes_nofrozen += ", ";
    refreshable_ids_nofrozen_str += "(";
    refreshable_ids_nofrozen_str += std::to_string(refreshable_input_index_no_frozen_and_allocation_ids_[i].first);
    refreshable_ids_nofrozen_str += ", ";
    refreshable_ids_nofrozen_str += std::to_string(refreshable_input_index_no_frozen_and_allocation_ids_[i].second);
    refreshable_ids_nofrozen_str += "), ";
  }
  GELOGI("[Gen][FrozenInputIndexes], zero_copy_input_indexes_no_frozen is: %s",
         input_indexes_nofrozen.c_str());
  GELOGI("[Gen][RefreshableFrozenInputIndexes], refreshable_input_index_no_frozen_and_allocation_ids is: %s",
         refreshable_ids_nofrozen_str.c_str());
}

Status DavinciModel::GenInputMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data) {
  GE_ASSERT_SUCCESS(ParseHostInputIndexOption(index_to_data.size()));
  copy_host_input_infos_.clear();
  copy_host_input_infos_.resize(index_to_data.size());

  input_index_to_allocation_ids_.resize(index_to_data.size(), UINT32_MAX);
  uint32_t input_base_allocation_id = logical_mem_allocations_.size();
  // 两次轮询，先放frozen index部分，后续排放no frozen部分
  for (size_t construct_input_logical_allcation_loop = 0;
       construct_input_logical_allcation_loop < kConstructInputLogicalAllcationLoop; construct_input_logical_allcation_loop++) {
    uint32_t input_index = 0U;
    for (const auto &item : index_to_data) {
      if ((construct_input_logical_allcation_loop == 0 && frozen_input_indexes_.count(input_index) == 0) ||
      (construct_input_logical_allcation_loop != 0 && frozen_input_indexes_.count(input_index) != 0)){
        input_index++;
        continue;
      }
      std::vector<uint64_t> mem_types;
      const auto virtual_addr_list = ModelUtils::GetOutputAddrsValue(runtime_param_, item.second, mem_types);
      const auto output_size_list = ModelUtils::GetOutputSize(item.second);

      GELOGD("Data node is: %s, output size is %zu, virtual_addr size is %zu.", item.second->GetName().c_str(),
            output_size_list.size(), virtual_addr_list.size());
      GE_ASSERT_EQ(output_size_list.size(), virtual_addr_list.size());
      GE_ASSERT_EQ(virtual_addr_list.size(), mem_types.size());
      if (virtual_addr_list.empty() || output_size_list.empty()) {
        GELOGE(PARAM_INVALID, "[Check][Param] Data[%s] failed: output size is %zu, virtual_addr size is %zu.",
              item.second->GetName().c_str(), output_size_list.size(), virtual_addr_list.size());
        return PARAM_INVALID;
      }

      const uint64_t logical_addr = virtual_addr_list[kDataIndex];
      const uint64_t data_size = static_cast<uint64_t>(output_size_list[kDataIndex]);
      MemAllocationAndOffset mem_allocation_and_offset{};
      if (GetMemAllocationByLogicAddr(logical_addr, mem_allocation_and_offset) == SUCCESS) {
        // id 0 indicates that the input address is within the feature map address range
        input_indexes_to_copy_info_[input_index] =
          {static_cast<uint32_t>(mem_allocation_and_offset.id), mem_allocation_and_offset.offset, data_size};
        GELOGW("[mem allocation][input] model_id %u, input_index %u, op_name %s op_type %s not support zero copy, %s.",
              model_id_, input_index, item.second->GetName().c_str(), item.second->GetType().c_str(),
              input_indexes_to_copy_info_[input_index].ToString().c_str());
        // RefData 被识别不能零拷贝的可定位手段
        GE_ASSERT_TRUE((item.second->GetType() != REFDATA),
            "model_id %u, input_index %u, op_name %s op_type %s not support zero copy",
            model_id_, input_index, item.second->GetName().c_str(), item.second->GetType().c_str());

        // host input index随路拷贝只支持零拷贝场景
        if (copy_host_input_indexes_.count(input_index) != 0U) {
          GELOGW("model_id %u, host_input_index %u, op_name %s op_type %s not support zero copy",
              model_id_, input_index, item.second->GetName().c_str(), item.second->GetType().c_str());
        }

        input_index++;
        continue;
      }

      refreshable_input_index_and_allocation_ids_.emplace_back(
          std::make_pair(input_index, static_cast<uint32_t>(logical_mem_allocations_.size())));

      uint64_t tensor_size = data_size;
      int64_t size = 0L;
      const OpDescPtr &op_desc = item.second;
      const auto tensor_desc = op_desc->GetOutputDescPtr(kDataIndex);
      if ((tensor_desc != nullptr) && (TensorUtils::GetTensorSizeInBytes(*tensor_desc, size) == GRAPH_SUCCESS)) {
        tensor_size = static_cast<uint64_t>(size);
      }

      MemAllocation mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()), logical_addr, data_size,
                                      ge::MemAllocation::Type::INPUT, input_index, mem_types[kDataIndex], 0UL, 0UL};
      mem_allocation.tensor_size = tensor_size;
      GELOGI("[mem allocation][input] model_id %u, input_index %u, op_name %s op_type %s, %s, tensor_size %" PRIu64,
            model_id_, input_index, item.second->GetName().c_str(), item.second->GetType().c_str(),
            mem_allocation.ToString().c_str(), tensor_size);
      logical_mem_allocations_.emplace_back(mem_allocation);
      input_index_to_allocation_ids_[input_index] = mem_allocation.id;
      zero_copy_input_indexes_.push_back(input_index);
      // 保存随路拷贝的io的索引以及长度，预留保存device地址的成员，只有支持零拷贝的走该流程
      if (copy_host_input_indexes_.count(input_index) > 0U) {
        GE_ASSERT_TRUE((item.second->GetType() != REFDATA),
            "model_id %u, input_index %u, op_name %s op_type %s not support host input index ",
            model_id_, input_index, item.second->GetName().c_str(), item.second->GetType().c_str());
        CopyHostInputInfo copy_host_input = {};
        copy_host_input.input_index = input_index;
        copy_host_input.tensor_size = tensor_size;
        copy_host_input_infos_[input_index] = std::move(copy_host_input);
        host_input_size_ += tensor_size;
      }

      if (frozen_input_indexes_.count(input_index) == 0) {
        refreshable_input_index_no_frozen_and_allocation_ids_.push_back(std::make_pair(input_index,
          mem_allocation.id));
        zero_copy_input_indexes_no_frozen_.push_back(input_index);
      }
      input_index++;
    }
  }

  if (host_input_size_ > 0U) {
    host_input_size_ = ge::MemSizeAlign(host_input_size_, kAlign32B); // 后续可能会通过算子计算方式刷新，此处先做32字节对齐
  }

  no_frozen_input_allocation_base_id_ = frozen_input_indexes_.size() + input_base_allocation_id;
  InitModelInputsMergeCopyHostMem();
  InitBatchMemcpyH2d();
  if (logLevel_ <= DLOG_INFO) {
    PrintNoFrozenInputIndexes();
  }
  return SUCCESS;
}

Status DavinciModel::GenOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list) {
  uint32_t output_index = 0U;

  for (const auto &op_desc : output_op_list) {
    std::vector<uint64_t> mem_types;
    const std::vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
    const std::vector<uint64_t> virtual_addr_list = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc, mem_types);

    GELOGD("NetOutput node is: %s, input size is %zu, virtual_addr size is %zu.", op_desc->GetName().c_str(),
           input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(virtual_addr_list.size(), mem_types.size());

    size_t actual_output_size = virtual_addr_list.size();
    if (is_getnext_sink_dynamic_) {
      actual_output_size -= kGetDynamicDimsCount;
      GELOGD(
          "In getnext sink dynamic scene, output size will minus 1 as GetNextDynamic is not model output, "
          "actual output size:%zu",
          actual_output_size);
    }

    for (size_t i = 0UL; i < actual_output_size; ++i) {
      int64_t data_size;
      const auto &tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
      GE_ASSERT_NOTNULL(tensor_desc);
      GE_ASSERT_SUCCESS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, data_size));
      output_indexes_to_tensor_size_[output_index] = static_cast<uint64_t>(data_size);
      const uint64_t logical_addr = virtual_addr_list[i];
      if (output_data_to_slice_flag_[output_index]) {
        output_index++;
        continue;
      }

      MemAllocationAndOffset mem_allocation_and_offset{};
      const auto ret = GetMemAllocationByLogicAddr(logical_addr, mem_allocation_and_offset);
      GELOGI("[mem allocation][output] model_id %u, output_index %u, logical_addr 0x%" PRIx64
        " ret %u id %u offset 0x%" PRIx64 ".", model_id_, output_index, logical_addr, ret,
        mem_allocation_and_offset.id, mem_allocation_and_offset.offset);
      if ((mem_types[i] == kVarMemType) || (ret == SUCCESS)) {
        // id = 0 indicates that the output address is within the feature map address range
        // id = 0xFFFFFFFFU indicates that the output will be copy after model execute
        uint32_t id = 0xFFFFFFFFU;
        uint64_t offset = logical_addr;
        if (mem_types[i] != kVarMemType) {
          GE_ASSERT_TRUE(ret == SUCCESS, "not find 0x%" PRIx64 " in allocatin table", logical_addr);
          id = static_cast<uint32_t>(mem_allocation_and_offset.id);
          offset = mem_allocation_and_offset.offset;
        }

        output_indexes_to_copy_info_[output_index] = {id, offset, static_cast<uint64_t>(data_size)};
        GELOGI("[mem allocation][output] model_id %u, output_index %u, add output copy info, %s.",
               model_id_, output_index, output_indexes_to_copy_info_[output_index].ToString().c_str());
        output_index++;
        continue;
      }

      refreshable_output_index_and_allocation_ids_.emplace_back(
          std::make_pair(output_index, static_cast<uint32_t>(logical_mem_allocations_.size())));
      MemAllocation mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()), virtual_addr_list[i],
                                      static_cast<uint64_t>(input_size_list[i]), ge::MemAllocation::Type::OUTPUT,
                                      output_index, mem_types[i], 0UL, 0UL};
      GELOGI("[mem allocation][output] model_id %u, %s.", model_id_, mem_allocation.ToString().c_str());
      logical_mem_allocations_.emplace_back(mem_allocation);
      output_index_to_allocation_ids_[output_index] = mem_allocation.id;
      zero_copy_output_indexes_.push_back(output_index);
      output_index++;
    }
  }

  return SUCCESS;
}

Status DavinciModel::GenSliceOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list) {
  int64_t fm_mem_size = 0;
  GE_ASSERT_SUCCESS(GetTotalMemSizeExcludeZeroCopy(fm_mem_size));
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  GE_CHECK_NOTNULL(compute_graph);

  uint32_t output_index = 0U;
  for (const auto &op_desc : output_op_list) {
    const auto node = compute_graph->FindNode(op_desc->GetName());
    GE_CHECK_NOTNULL(node);
    std::vector<uint64_t> mem_types;
    const std::vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
    const std::vector<uint64_t> virtual_addr_list = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc, mem_types);
    const std::vector<int64_t> v_input_offset = op_desc->GetInputOffset();

    GELOGD("NetOutput node is: %s, input size is %zu, virtual_addr size is %zu.", op_desc->GetName().c_str(),
           input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(virtual_addr_list.size(), mem_types.size());
    GE_ASSERT(virtual_addr_list.size() <= v_input_offset.size());

    size_t actual_output_size = virtual_addr_list.size();
    if (is_getnext_sink_dynamic_) {
      actual_output_size -= kGetDynamicDimsCount;
      GELOGD(
          "In getnext sink dynamic scene, output size will minus 1 as GetNextDynamic is not model output, "
          "actual output size:%zu",
          actual_output_size);
    }

    output_index_to_allocation_ids_.resize(actual_output_size, UINT32_MAX);
    for (size_t i = 0UL; i < actual_output_size; ++i) {
      const uint64_t logical_addr = virtual_addr_list[i];
      const int64_t offset = v_input_offset[i];
      if (IsInputOfNetoutputCanZeroCopy(node, static_cast<int32_t>(i)) && (offset >= 0) && (offset < fm_mem_size)) {
        output_data_to_slice_flag_[output_index] = true;
      }

      if (!output_data_to_slice_flag_[output_index]) {
        output_index++;
        continue;
      }

      // fusion场景, 识别成可以零拷贝的输出
      refreshable_output_index_and_allocation_ids_.emplace_back(
          std::make_pair(output_index, static_cast<uint32_t>(logical_mem_allocations_.size())));
      MemAllocation mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()), logical_addr,
                                      static_cast<uint64_t>(input_size_list[i]), ge::MemAllocation::Type::OUTPUT,
                                      output_index, mem_types[i], 0UL, 0UL};
      GELOGI("[mem allocation][output][slice] model_id %u, %s.", model_id_, mem_allocation.ToString().c_str());
      logical_mem_allocations_.emplace_back(mem_allocation);
      output_index_to_allocation_ids_[output_index] = mem_allocation.id;
      zero_copy_output_indexes_.push_back(output_index);
      output_index++;
    }
  }

  return SUCCESS;
}

Status DavinciModel::GenMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data,
                                       const std::vector<OpDescPtr> &output_op_list) {
  GE_ASSERT_SUCCESS(GenSliceOutputMemAllocations(output_op_list));

  // feature map mem allocation
  GE_ASSERT_SUCCESS(GenFmMemAllocations());

  // fixed feature map mem allocation
  GE_ASSERT_SUCCESS(GenFixedFmMemAllocations());

  // input mem allocation
  GE_ASSERT_SUCCESS(GenInputMemAllocations(index_to_data));

  // output mem allocation
  GE_ASSERT_SUCCESS(GenOutputMemAllocations(output_op_list));

  // absolute mem allocation
  MemAllocation not_change_mem_item = {static_cast<uint32_t>(logical_mem_allocations_.size()), 0U, UINT64_MAX,
                                       ge::MemAllocation::Type::ABSOLUTE, 0U, kAbsoluteMemType, 0UL, 0UL};
  GELOGI("[mem allocation][absolute] model_id %u, %s.", model_id_, not_change_mem_item.ToString().c_str());
  logical_mem_allocations_.emplace_back(not_change_mem_item);
  input_index_to_active_mem_base_addrs_.resize(input_index_to_allocation_ids_.size(), 0UL);
  output_index_to_active_mem_base_addrs_.resize(output_index_to_allocation_ids_.size(), 0UL);

  // active mem base的内存归一到args里
  GE_ASSERT_SUCCESS(args_manager_.AllocKernelLaunchArgsHostMem(
    logical_mem_allocations_.size(), host_input_size_));
  allocation_ids_to_active_base_addr_ = args_manager_.GetActivateMemBaseAddrs();
  GE_CHECK_NOTNULL(allocation_ids_to_active_base_addr_);
  ConstructFixedFmActiveMemBase();
  return SUCCESS;
}

void DavinciModel::InitModelInputsMergeCopyHostMem() {
  uint64_t input_fusion_size = ge::GetContext().GetInputFusionSize();
  if (has_no_tiling_input_ || (input_fusion_size == 0U)) {
    GELOGI("[InputMergeCopy] has_no_tiling_input: %d, input fusion size is 0, no need to merge copy.",
           has_no_tiling_input_);
    return;
  }

  // record input index and logical address for merge copy
  std::vector<std::pair<uint32_t, uint64_t>> input_index_and_logical_addr;
  for (auto idx : zero_copy_input_indexes_) {
    const auto id = input_index_to_allocation_ids_[idx];
    const auto input_size = logical_mem_allocations_[id].tensor_size;
    if (input_size > input_fusion_size) {
      GELOGI("[InputMergeCopy]Input[%u] size %" PRIu64 " is bigger than input fusion size %" PRIu64
             ", no need merge-copy.",
             idx, input_size, input_fusion_size);
      continue;
    }
    const auto mem_size = logical_mem_allocations_[id].data_size;
    const auto logical_addr = logical_mem_allocations_[id].logical_addr;
    input_index_and_logical_addr.emplace_back(std::make_pair(idx, logical_addr));
    GELOGI("[InputMergeCopy]input index %u, logic addr %" PRIu64 ", mem size %" PRIu64 ", tensor size %" PRIu64, idx,
           logical_addr, mem_size, input_size);
  }

  if (input_index_and_logical_addr.size() <= 1U) {
    GELOGI("[InputMergeCopy]no need merge h2d copy, fusion input num: %zu", input_index_and_logical_addr.size());
    return;
  }

  // sort by logical address
  std::sort(input_index_and_logical_addr.begin(), input_index_and_logical_addr.end(),
            [](const std::pair<uint32_t, uint64_t> &a, const std::pair<uint32_t, uint64_t> &b) {
              return a.second < b.second;
            });

  const auto &first_input = input_index_and_logical_addr.front();
  const auto &last_input = input_index_and_logical_addr.back();
  const auto last_input_size = logical_mem_allocations_[input_index_to_allocation_ids_[last_input.first]].data_size;
  fisrt_input_index_of_merge_copy_ = first_input.first;

  // init host buff for merge copy, if fail just return and run with no merge copy
  input_merge_copy_mem_size_ = last_input.second + last_input_size - first_input.second;
  void *host_mem = nullptr;
  const rtError_t rt_ret = rtMallocHost(&host_mem, input_merge_copy_mem_size_, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    input_merge_copy_mem_base_.reset();
    GELOGW("[InputMergeCopy][rtMallocHost] host buffer alloc failed, size:%" PRIu64 ", ret:%d",
           input_merge_copy_mem_size_, static_cast<int32_t>(rt_ret));
    return;
  }
  if (host_mem == nullptr) {
    input_merge_copy_mem_base_.reset();
    GELOGW("[InputMergeCopy][rtMallocHost] host buffer is nullptr, size:%" PRIu64 ", ret:%d",
           input_merge_copy_mem_size_, static_cast<int32_t>(rt_ret));
    return;
  }
  input_merge_copy_mem_base_.reset(static_cast<uint8_t *>(host_mem), [](uint8_t *ptr) {
    if (ptr == nullptr) {
      return;
    }
    const rtError_t free_ret = rtFreeHost(ptr);
    if (free_ret != RT_ERROR_NONE) {
      GELOGW("[InputMergeCopy][rtFreeHost] host buffer free failed, ptr:%p, ret:%d", ptr,
             static_cast<int32_t>(free_ret));
    }
  });
  (void)memset_s(input_merge_copy_mem_base_.get(), input_merge_copy_mem_size_, 0U, input_merge_copy_mem_size_);

  // record offset for fusion copy input
  for (auto iter : input_index_and_logical_addr) {
    input_index_to_merge_copy_offset_[iter.first] = iter.second - first_input.second;
  }

  GELOGI("[InputMergeCopy]host base: %p, first input index: %u, fusion input num: %zu, size:%" PRIu64,
         input_merge_copy_mem_base_.get(), fisrt_input_index_of_merge_copy_, input_index_to_merge_copy_offset_.size(),
         input_merge_copy_mem_size_);
  return;
}

Status DavinciModel::GetMemAllocationByLogicAddr(const uint64_t addr, MemAllocationAndOffset &allocation_info) const {
  for (const auto &item : logical_mem_allocations_) {
    if ((addr >= item.logical_addr) && (addr < (item.logical_addr + item.data_size))) {
      allocation_info.id = static_cast<size_t>(item.id);
      allocation_info.offset = (addr - item.logical_addr);
      return SUCCESS;
    }
  }
  return INTERNAL_ERROR;
}

bool DavinciModel::IsGetNextSinkDynamic(const OpDescPtr &op_desc) const {
  bool getnext_sink_dynamic = false;
  if (AttrUtils::GetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, getnext_sink_dynamic) && getnext_sink_dynamic) {
    GELOGI("ATTR_GETNEXT_SINK_DYNMAIC has been set and is true.");
    return true;
  }
  return false;
}

/// @ingroup ge
/// @brief NetOutput Op Initialize.
/// @param [in] ComputeGraphPtr: root graph of the model.
/// @param [in] NodePtr: NetOutput Op.
/// @param [in/out] std::vector<OpDescPtr>: All NetOutput node in model.
/// @return Status
Status DavinciModel::InitNetOutput(const ComputeGraphPtr &graph, const NodePtr &node,
                                   std::vector<OpDescPtr> &output_op_list,
                                   std::set<uint64_t> &output_outside_addrs) {
  // node->GetOpDesc Checked by Init: NetOutput, valid.
  const auto op_desc = node->GetOpDesc();
  // excludes the function op sub graph, e.g. case,if
  if (node->GetOwnerComputeGraph() != graph) {
    GELOGI("Skip subgraph NetOutput node: %s.", op_desc->GetName().c_str());
    (void)op_list_.erase(op_desc->GetId());
    (void)operator_list_.erase(op_desc->GetId());
    return SUCCESS;
  }

  GELOGI("Init NetOutput node: %s, model id:%u", op_desc->GetName().c_str(), model_id_);
  output_op_list.push_back(op_desc);
  has_output_node_ = true;
  if (feature_base_refreshable_ && !support_extend_memory_full_) {
    return SUCCESS;
  }

  // Make information for copy output data.
  const std::vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
  const std::vector<uint64_t> virtual_addr_list = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc);
  const std::vector<int64_t> input_offset_list = op_desc->GetInputOffset();
  GE_IF_BOOL_EXEC(input_offset_list.size() != virtual_addr_list.size(),
                  REPORT_INNER_ERR_MSG("E19999", "Check data fail in op:%s(%s), input addr size:%zu "
                                     "input offset size:%zu not equal, model_id:%u", op_desc->GetName().c_str(),
                                     op_desc->GetType().c_str(), virtual_addr_list.size(), input_offset_list.size(),
                                     model_id_);
                  GELOGE(PARAM_INVALID, "[Check][Param] virtual_addr size:%zu should be equal to offset size:%zu, "
                         "op:%s(%s), model id:%u", virtual_addr_list.size(), input_offset_list.size(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
                  return PARAM_INVALID);
  if (input_size_list.empty() && virtual_addr_list.empty()) {
    GELOGI("NetOutput[%s] is empty.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (input_size_list.empty() || (input_size_list.size() != virtual_addr_list.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Check data fail in op:%s(%s), input_desc size:%zu input addr size:%zu "
                       "not equal or has empty, model_id:%u", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       input_size_list.size(), virtual_addr_list.size(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] NetOutput[%s] init failed: Input size is %zu, Input addr is %zu",
           op_desc->GetName().c_str(), input_size_list.size(), virtual_addr_list.size());
    return PARAM_INVALID;
  }

  const size_t num = output_data_info_.size();
  size_t input_count = input_size_list.size();
  is_getnext_sink_dynamic_ = false;
  if (IsGetNextSinkDynamic(op_desc)) {
    input_count = input_size_list.size() - kGetDynamicDimsCount;
    is_getnext_sink_dynamic_ = true;
  }
  const bool is_reuse_zero_copy_memory = ModelUtils::IsReuseZeroCopyMemory();
  for (size_t idx = 0UL; idx < input_count; ++idx) {
    ZeroCopyOffset zero_copy_offset;
    bool fusion_flag = false;
    const auto ret = zero_copy_offset.InitOutputDataInfo(input_size_list, virtual_addr_list, op_desc, idx, fusion_flag);
    GE_IF_BOOL_EXEC(ret != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Init][DataInfo] of input_info %s failed.", op_desc->GetName().c_str());
                    return PARAM_INVALID);
    const uint64_t addr = virtual_addr_list.at(idx);
    const int64_t input_offset = input_offset_list.at(idx);
    if (output_outside_addrs.count(addr) > 0UL) {
      GELOGI("same output_tensor_addr %" PRIu64 " to different input_tensor of %s", addr, op_desc->GetName().c_str());
      // TEMP SOLUTION: refdata can not disable zero copy. This code will be discard when static execution
      // refactoring
      if (!copy_only_addrs_.IsRefDataAddr(addr)) {
        GE_ASSERT_SUCCESS(DisableZeroCopy(ValueToPtr(addr)));
      }
    } else {
      std::vector<uint64_t> tensor_addrs;
      GE_RETURN_IF_ERROR(zero_copy_offset.SetOutputOutsideAddrs(input_offset, fusion_flag,
                                                                static_cast<uintptr_t>(addr), tensor_addrs));
      (void)output_outside_addrs.insert(addr);

      if (!fusion_flag) {
        GE_ASSERT_SUCCESS(DisableZeroCopyInReuseMemoryMode(node, idx, ValueToPtr(addr)));
      }
      if (!is_reuse_zero_copy_memory) {
        for (const auto &real_addr : tensor_addrs) {
          // TEMP SOLUTION: refdata can not disable zero copy. This code will be discard when static execution
          // refactoring
          if (copy_only_addrs_.IsRefDataAddr(real_addr)) {
            GELOGW("Refdata op %s virtual addr:[%p] as netoutput can not disable zero copy,"
                   " may cause precision problem.",
                   op_desc->GetName().c_str(), ValueToPtr(real_addr));
            continue;
          }
          GE_ASSERT_SUCCESS(DisableZeroCopy(ValueToPtr(real_addr), fusion_flag));
          (void)real_virtual_addrs_.insert(real_addr);
        }
      }
    }
    output_data_info_[static_cast<uint32_t>(num + idx)] = zero_copy_offset;
    output_data_to_slice_flag_[static_cast<uint32_t>(num + idx)] = fusion_flag;
  }
  return SUCCESS;
}

Status DavinciModel::InitRealSizeAndShapeInfo(const ComputeGraphPtr &compute_graph, const NodePtr &node) {
  if (node->GetName().find(kMultiBatchNodePostfix) != std::string::npos) {
    GELOGD("No need to get size and shape of netoutput in subgraph.");
    return SUCCESS;
  }
  if (all_gears_info_.empty()) {
    all_gears_info_ = run_context_.dynamic_shape_dims;
    is_online_infer_dynamic_ = (!run_context_.dynamic_shape_dims.empty());
  }
  GELOGD("Start to initialize real size and shape info of %s.", node->GetName().c_str());
  if (is_getnext_sink_dynamic_) {
    if (GetDynamicDimsNodeInfo(node) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Get][Info] of getdynamicdims node:%s failed.", node->GetName().c_str());
      return PARAM_INVALID;
    }
  }
  if (is_online_infer_dynamic_) {
    if (GetGearAndRealOutSizeInfo(compute_graph, node) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Call][GetGearAndRealOutSizeInfo] failed, node:%s.", node->GetName().c_str());
      return PARAM_INVALID;
    }
    if (GetGearAndRealOutShapeInfo(node) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Call][GetGearAndRealOutShapeInfo] failed, node:%s.", node->GetName().c_str());
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status DavinciModel::GetDynamicDimsNodeInfo(const NodePtr &node) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const size_t input_count = node->GetAllInDataAnchors().size();
  GELOGI("input_anchor count of %s is %zu.", node->GetName().c_str(), input_count);
  GE_CHECK_GE(input_count, kGetDynamicDimsCount);
  const size_t get_dynamic_dims_index = input_count - kGetDynamicDimsCount;
  const auto in_anchor = node->GetAllInDataAnchors().at(get_dynamic_dims_index);
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);

  const auto peer_node = peer_out_anchor->GetOwnerNode();
  const auto op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if ((op_desc->GetName() == kGetDynamicDimsName) && (op_desc->GetType() == GETDYNAMICDIMS)) {
    GELOGD("Start get info of %s.", op_desc->GetName().c_str());
    const auto input_addr = ModelUtils::GetInputAddrsValue(runtime_param_, node->GetOpDesc());
    const auto input_size = ModelUtils::GetInputSize(node->GetOpDesc());
    if (input_addr.empty() || input_size.empty() || (input_addr.size() != input_size.size())) {
      REPORT_INNER_ERR_MSG("E19999", "input addr size:%zu or input size:%zu in op:%s(%s) has empty, model_id:%u "
                         "check invalid", input_addr.size(), input_size.size(),
                         node->GetName().c_str(), node->GetType().c_str(), model_id_);
      GELOGE(PARAM_INVALID, "[Check][Param] input addr size:%zu or input size:%zu in op:%s(%s) is empty, model_id:%u",
             input_addr.size(), input_size.size(), node->GetName().c_str(), node->GetType().c_str(), model_id_);
      return PARAM_INVALID;
    }
    const auto input_desc = node->GetOpDesc()->GetInputDescPtr(static_cast<uint32_t>(get_dynamic_dims_index));
    GE_CHECK_NOTNULL(input_desc);
    if (input_desc->GetShape().GetDims().empty()) {
      REPORT_INNER_ERR_MSG("E19999", "input_desc_index:%zu in op:%s(%s) shape dim is empty, model_id:%u, check invalid",
                         get_dynamic_dims_index, node->GetName().c_str(), node->GetType().c_str(), model_id_);
      GELOGE(PARAM_INVALID, "[Check][Param] input_desc_index:%zu in op:%s(%s) shape dim is empty, model_id:%u",
             get_dynamic_dims_index, node->GetName().c_str(), node->GetType().c_str(), model_id_);
      return PARAM_INVALID;
    }
    netoutput_last_input_addr_ = ValueToPtr(input_addr[get_dynamic_dims_index]);
    netoutput_last_input_size_ = input_size[get_dynamic_dims_index];
    shape_of_cur_dynamic_dims_ = static_cast<size_t>(input_desc->GetShape().GetDims().at(0U));
    GELOGD("Shape of cur dynamic dims is %zu, size is %" PRId64 ", addr is %p.", shape_of_cur_dynamic_dims_,
           netoutput_last_input_size_, netoutput_last_input_addr_);
  }
  return SUCCESS;
}

Status DavinciModel::GetGearAndRealOutSizeInfo(const ComputeGraphPtr &graph, const NodePtr &node) {
  GELOGD("Start get gear and real output size info of %s.", node->GetName().c_str());
  merge_nodes_gear_and_real_out_size_info_.clear();
  size_t idx = 0U;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      const auto peer_node = peer_out_anchor->GetOwnerNode();
      const auto op_desc = peer_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if ((peer_node->GetType() == CASE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
        if (GetRealOutputSizeOfCase(graph, idx, peer_node) != SUCCESS) {
          GELOGE(PARAM_INVALID, "[Get][RealOutputSizeOfCase] %s failed.", peer_node->GetName().c_str());
          return PARAM_INVALID;
        }
      }
      idx++;
    }
  }
  return SUCCESS;
}

Status DavinciModel::GetRealOutputSizeOfCase(const ComputeGraphPtr &graph, const size_t input_index,
                                             const NodePtr &case_node) {
  GELOGD("Start to get output size of %s, which is %zu input to netoutput", case_node->GetName().c_str(), input_index);
  const auto &func_desc = case_node->GetOpDesc();
  GE_CHECK_NOTNULL(func_desc);
  auto &gear_and_real_out_size_info = merge_nodes_gear_and_real_out_size_info_[input_index];
  for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
    const auto &subgraph = graph->GetSubgraph(name);
    if (subgraph == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Get name:%s subgraph in graph:%s fail, model_id:%u, check invalid",
                         name.c_str(), graph->GetName().c_str(), model_id_);
      GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][Subgraph] %s in graph:%s failed, model_id:%u.",
             name.c_str(), graph->GetName().c_str(), model_id_);
      return GE_GRAPH_EMPTY_SUBGRAPH;
    }
    for (auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() == NETOUTPUT) {
        const auto op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        std::string batch_label;
        if (AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
          size_t batch_index = 0U;
          try {
            batch_index = static_cast<size_t>(stoi(batch_label.substr(batch_label.rfind('_') + 1U)));
          } catch (std::invalid_argument &) {
            GELOGE(PARAM_INVALID, "invalid batch lable %s.", batch_label.c_str());
            return PARAM_INVALID;
          } catch (std::out_of_range &) {
            GELOGE(PARAM_INVALID, "batch lable %s transform to size_t failed.", batch_label.c_str());
            return PARAM_INVALID;
          }
          GELOGD("Batch index of %s is %zu.", op_desc->GetName().c_str(), batch_index);
          if (batch_index >= all_gears_info_.size()) {
            REPORT_INNER_ERR_MSG("E19999", "Batch_index:%zu in op:%s(%s) > all_gears_info.size:%zu, model_id:%u, "
                               "check invalid", batch_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                               all_gears_info_.size(), model_id_);
            GELOGE(PARAM_INVALID, "[Check][Param] Batch_index:%zu in op:%s(%s) > all_gears_info.size:%zu, model_id:%u.",
                   batch_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                   all_gears_info_.size(), model_id_);
            return PARAM_INVALID;
          }

          const std::vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
          const auto tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(input_index));
          GE_CHECK_NOTNULL(tensor_desc);
          int64_t data_size = 0;
          if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, data_size) != GRAPH_SUCCESS) {
            REPORT_INNER_ERR_MSG("E19999", "Get input TensorSize in op:%s(%s) failed, input_index:%zu, model_id:%u",
                               op_desc->GetName().c_str(), op_desc->GetType().c_str(), input_index, model_id_);
            GELOGE(FAILED, "[Get][TensorSize] in op:%s(%s) failed, input_index:%zu, model_id:%u",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str(), input_index, model_id_);
            return FAILED;
          }
          gear_and_real_out_size_info[all_gears_info_[batch_index]] = data_size;
          GELOGD("Get real gear index is: %zu, gear info is %s, size is %" PRId64 ", tensor size is %" PRId64,
                 batch_index, ToString(all_gears_info_[batch_index]).c_str(), input_size_list[input_index], data_size);
        }
        break;
      }
    }
  }
  return SUCCESS;
}

Status DavinciModel::GetGearAndRealOutShapeInfo(const NodePtr &node) {
  GELOGD("Start to get dynamic output dims of %s", node->GetName().c_str());
  merge_nodes_gear_and_real_out_shape_info_.clear();
  size_t idx = 0U;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      const auto peer_node = peer_out_anchor->GetOwnerNode();
      const auto op_desc = peer_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if ((peer_node->GetType() == CASE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
        std::vector<std::string> dynamic_output_shape_info;
        if (!AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_shape_info)) {
          GELOGD("Can not get dynamic output dims attr from %s", node->GetName().c_str());
          return SUCCESS;
        }
        GELOGI("Dynamic output shape info is %s", ToString(dynamic_output_shape_info).c_str());
        std::vector<std::vector<int64_t>> dynamic_output_shape;
        ParseDynamicOutShape(dynamic_output_shape_info, dynamic_output_shape);
        auto &gear_and_real_out_shape_info = merge_nodes_gear_and_real_out_shape_info_[idx];
        for (auto &it : dynamic_output_shape) {
          GE_CHECK_GE(it.size(), static_cast<size_t>(kDynamicOutInfoMinSize));
          const int64_t gear_index = it[0U];
          GE_ASSERT_TRUE(gear_index >= 0,
              "gear index is less than 0, node NetOutput may have input not from node Case");
          GE_ASSERT_TRUE(gear_index < static_cast<int64_t>(all_gears_info_.size()),
                   "[Check][Param] gear index:%zu in op:%s(%s) > all_gears_info.size:%zu in model:%u.",
                   gear_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), all_gears_info_.size(),
                   model_id_);

          if (static_cast<size_t>(it[1U]) == idx) {
            std::vector<int64_t> output_shape;
            for (size_t i = 2U; i < it.size(); ++i) {
              output_shape.emplace_back(it[i]);
            }
            gear_and_real_out_shape_info[all_gears_info_[gear_index]] = output_shape;
            GELOGD("Get real gear index is: %zu, gear info is %s, output shape is %s",
                   gear_index, ToString(all_gears_info_[gear_index]).c_str(), ToString(output_shape).c_str());
          }
        }
      }
      idx++;
    }
  }
  return SUCCESS;
}

void DavinciModel::ParseDynamicOutShape(const std::vector<std::string> &str_info,
                                        std::vector<std::vector<int64_t>> &vec_info) const {
  for (size_t i = 0U; i < str_info.size(); ++i) {
    std::vector<int64_t> shape;
    const auto dims = StringUtils::Split(str_info[i], ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shape.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimalRadix));
    }
    GELOGI("Shape from attr is %s", ToString(shape).c_str());
    vec_info.emplace_back(shape);
  }
}

Status DavinciModel::GetLabelGotoAddr(const uint32_t label_index, const rtMemType_t mem_type,
                                      void *&arg_addr, uint32_t &arg_size) {
  const std::lock_guard<std::mutex> lk(label_args_mutex_);
  const auto it = label_goto_args_.find(label_index);
  if (it != label_goto_args_.cend()) {
    arg_addr = it->second.first;
    arg_size = it->second.second;
    return SUCCESS;
  }

  if (label_index >= label_list_.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Param label index:%u >= label_list_.size:%zu in model:%u, check invalid",
                       label_index, label_list_.size(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Param label index:%u >= label_list_.size:%zu in model:%u",
           label_index, label_list_.size(), model_id_);
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL(label_list_[static_cast<size_t>(label_index)]);
  std::vector<rtLabel_t> label_used = { label_list_[static_cast<size_t>(label_index)] };

  arg_size = static_cast<uint32_t>(label_used.size() * sizeof(rtLabelDevInfo));
  arg_addr = MallocDynamicMemory(static_cast<uint64_t>(arg_size), mem_type);
  GE_ASSERT_NOTNULL(arg_addr);
  label_goto_args_[label_index] = { arg_addr, arg_size };

  GE_CHK_RT_RET(rtLabelListCpy(label_used.data(), static_cast<uint32_t>(label_used.size()), arg_addr, arg_size));

  return SUCCESS;
}

void DavinciModel::SetModelQueueParam(const ModelQueueParam &model_queue_param) {
  model_queue_param_ = model_queue_param;
}

void DavinciModel::SetGlobalStep(const uintptr_t step_addr, const uint64_t step_size) {
  global_step_addr_ = step_addr;
  global_step_size_ = step_size;
}

void* DavinciModel::GetMemEventIdAddr(const uint32_t mem_event_id) {
  auto it = mem_event_id_mem_map_.find(mem_event_id);
  if (it != mem_event_id_mem_map_.end()) {
    auto ret = it->second;
    GELOGI("get mem_event_id %u, addr %p", mem_event_id, ret);
    return ret;
  } else {
    const size_t mem_event_size = 8;
    void *cur_mem = MallocDynamicMemory(mem_event_size);
    GE_ASSERT_NOTNULL(cur_mem);
    (void)rtMemset(cur_mem, mem_event_size, 0, mem_event_size);
    mem_event_id_mem_map_[mem_event_id] = cur_mem;
    GELOGI("append mem_event_id %u, addr is %p", mem_event_id, cur_mem);
    return cur_mem;
  }
}

/// @ingroup ge
/// @brief Get Op rtStream.
/// @param [in] op_desc: Op descriptor.
/// @param [in] stream_id: Logical stream id.
/// @param [out] stream: rt stream.
/// @return Status
Status DavinciModel::GetOpStream(const OpDescPtr &op_desc, const size_t stream_id, rtStream_t &stream) {
  if (stream_list_.size() == 1U) {
    stream = stream_list_[0U];
  } else if (stream_list_.size() > stream_id) {
    stream = stream_list_[stream_id];
  } else {
    REPORT_INNER_ERR_MSG("E19999", "stream_id:%zu in op:%s(%s) >= stream size:%zu in model:%u, check invalid",
        stream_id, op_desc->GetName().c_str(), op_desc->GetType().c_str(), stream_list_.size(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] stream_id:%zu in op:%s(%s) >= stream size:%zu in model:%u",
           stream_id, op_desc->GetName().c_str(), op_desc->GetType().c_str(), stream_list_.size(), model_id_);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

/// @ingroup ge
/// @brief LabelSet Op Initialize.
/// @param [in] op_desc: LabelSet Op descriptor.
/// @return Status
Status DavinciModel::InitLabelSet(const OpDescPtr &op_desc) {
  uint32_t label_index = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail, model_id:%u, check invalid",
                       ATTR_NAME_LABEL_SWITCH_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       model_id_);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) fail, model_id:%u",
           ATTR_NAME_LABEL_SWITCH_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return INTERNAL_ERROR;
  }
  if (label_index >= runtime_param_.label_num) {
    REPORT_INNER_ERR_MSG("E19999", "label_switch_index:%u in Node:%s >= label_num:%u in model:%u, check invalid",
                       label_index, op_desc->GetName().c_str(), runtime_param_.label_num, model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] label_switch_index:%u in Node:%s >= label_num:%u in model:%u",
           label_index, op_desc->GetName().c_str(), runtime_param_.label_num, model_id_);
    return INTERNAL_ERROR;
  }
  if (label_id_indication_.count(label_index) > 0U) {
    REPORT_INNER_ERR_MSG("E19999", "label_switch_index:%u in op:%s(%s) is already used  in model:%u, check invalid",
                       label_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] label_switch_index:%u in op: %s(%s) is already used in model: %u",
           label_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return INTERNAL_ERROR;
  }

  rtStream_t stream = nullptr;
  const size_t stream_id = static_cast<size_t>(op_desc->GetStreamId());
  GE_CHK_STATUS_RET_NOLOG(GetOpStream(op_desc, stream_id, stream));

  rtLabel_t rt_label = nullptr;
  GE_CHK_RT_RET(rtLabelCreateExV2(&rt_label, rt_model_handle_, stream));

  GELOGI("InitLabelSet: label[%u]=%p stream[%zu]=%p", label_index, rt_label, stream_id, stream);
  (void)label_id_indication_.insert(label_index);
  label_list_[static_cast<size_t>(label_index)] = rt_label;
  return SUCCESS;
}

Status DavinciModel::InitVariable(const OpDescPtr &op_desc, std::map<std::string, OpDescPtr> &variable_by_name) {
  if (!known_node_) {
    if (op_desc->GetName() == NODE_NAME_GLOBAL_STEP) {
      const auto output_sizes = ModelUtils::GetOutputSize(op_desc);
      if (!output_sizes.empty()) {
        global_step_size_ = static_cast<uint64_t>(output_sizes[0U]);
      }
      const auto output_addrs = ModelUtils::GetOutputAddrs(runtime_param_, op_desc);
      if (!output_addrs.empty()) {
        global_step_addr_ = static_cast<uintptr_t>(PtrToValue(output_addrs[0U]));
      }
    }
  }

  variable_by_name[op_desc->GetName()] = op_desc;
  return SUCCESS;
}

/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_queue_attrs: input queue from user, nums equal Data Op.
/// @param [in] output_queue_attrs: input queue from user, nums equal NetOutput Op.
/// @return: 0 for success / others for failed
Status DavinciModel::SetQueIds(const std::vector<QueueAttrs> &input_queue_attrs,
                               const std::vector<QueueAttrs> &output_queue_attrs) {
  if (input_queue_attrs.empty() && output_queue_attrs.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Param input_queue_attrs.size:%zu and output_queue_attrs.size:%zu is empty,"
                       "model_id:%u, check invalid", input_queue_attrs.size(), output_queue_attrs.size(),
                       model_id_);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "[Check][Param] Param is empty, model_id:%u", model_id_);
    return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  input_queue_attrs_ = input_queue_attrs;
  output_queue_attrs_ = output_queue_attrs;
  return SUCCESS;
}

Status DavinciModel::SetQueueType() {
    std::vector<QueueAttrs> all_queues;
    all_queues.reserve(input_queue_attrs_.size() + output_queue_attrs_.size());
    std::copy(input_queue_attrs_.begin(), input_queue_attrs_.end(), std::back_inserter(all_queues));
    std::copy(output_queue_attrs_.begin(), output_queue_attrs_.end(), std::back_inserter(all_queues));

    if (all_queues.empty()) {
        GELOGD("No queues to check, return success directly.");
        return SUCCESS;
    }

    uint32_t first_entity_type = 0;
    uint32_t out_len = sizeof(uint32_t);
    GE_CHK_RT_RET(rtMemQueueQuery(device_id_, RT_MQ_QUERY_QUES_ATTR_ENTITY_TYPE,
                                  &all_queues[0].queue_id, sizeof(uint32_t),
                                  &first_entity_type, &out_len));
    bool first_is_hwq = (first_entity_type != 0);
    for (size_t i = 1; i < all_queues.size(); ++i) {
        uint32_t current_entity_type = 0;
        GE_CHK_RT_RET(rtMemQueueQuery(device_id_, RT_MQ_QUERY_QUES_ATTR_ENTITY_TYPE,
                                      &all_queues[i].queue_id, sizeof(uint32_t),
                                      &current_entity_type, &out_len));
        bool current_is_hwq = (current_entity_type != 0);

        // 类型不一致则打印错误日志并返回失败
        if (current_is_hwq != first_is_hwq) {
            GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "Queue types are not consistent. "
                  "Entity type of queue (ID: %u) is %u, but entity type of queue (ID: %u) is %u.",
                  all_queues[0].queue_id,
                  first_entity_type,
                  all_queues[i].queue_id,
                  current_entity_type);
            return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
        }
    }
    is_hw_q_ = first_is_hwq;
    GELOGI("All %zu queues are %s queue.",  all_queues.size(), is_hw_q_ ? "hardware" : "software");

    return SUCCESS;
}

void DavinciModel::SetStatusQueue(const QueueAttrs &status_output_queue) {
  status_output_queue_ = status_output_queue;
}

void DavinciModel::SetModelUuid(const uint32_t model_uuid) {
  model_uuid_ = model_uuid;
}

void DavinciModel::SetNeedModelConfig(const bool flag) {
  need_model_config_ = flag;
}

void DavinciModel::SetNeedReportStatus(bool need_report_status) {
  need_report_status_ = need_report_status;
}

void DavinciModel::SetInputFusionOffsets(const std::vector<int32_t> &fusion_offsets) {
  input_fusion_offsets_ = fusion_offsets;
}

///
/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_que_ids: input queue ids from user, nums equal Data Op.
/// @param [in] output_que_ids: input queue ids from user, nums equal NetOutput Op.
/// @return: 0 for success / others for failed
///
Status DavinciModel::LoadWithQueue() {
  if (is_hw_q_) {
    GELOGI("This is hardware queue.");
    return SUCCESS;
  }
  if (input_queue_attrs_.empty() && output_queue_attrs_.empty()) {
    return SUCCESS;
  }

  if ((input_fusion_offsets_.empty()) && (!input_queue_attrs_.empty())) {
    input_fusion_offsets_.resize(input_queue_attrs_.size());
  }

  use_control_input_queue_ = input_data_info_.empty() && (input_queue_attrs_.size() == 1U);
  use_control_output_queue_ = output_data_info_.empty() && (!output_queue_attrs_.empty());
  if ((!use_control_input_queue_) &&
      ((input_queue_attrs_.size() != input_data_info_.size()) ||
          (input_fusion_offsets_.size() != input_queue_attrs_.size()))) {
    REPORT_INNER_ERR_MSG("E19999", "Param input_queue_attrs_.size:%zu != input_data_info_.size:%zu or "
                       "input_queue_attrs_.size:%zu != input_fusion_offsets_.size:%zu, model_id:%u, check invalid",
                       input_queue_attrs_.size(), input_data_info_.size(),
                       input_queue_attrs_.size(), input_fusion_offsets_.size(),
                       model_id_);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "[Check][Param] Input queue ids not match model: "
           "input_queue=%zu input_data=%zu, model_id:%u",
           input_queue_attrs_.size(), input_data_info_.size(), model_id_);
    return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  if ((!use_control_output_queue_) && (output_queue_attrs_.size() != output_data_info_.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Param output_queue_attrs_.size:%zu != output_data_info_.size:%zu, model_id:%u,"
                       "check invalid", output_queue_attrs_.size(), output_data_info_.size(), model_id_);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID,
           "[Check][Param] Output queue ids not match model: output_queue=%zu output_data=%zu, model_id:%u",
           output_queue_attrs_.size(), output_data_info_.size(), model_id_);
    return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  GE_CHK_STATUS_RET(AddHeadStream(), "[Add][HeadStream] failed, model_id: %u", model_id_);
  if (model_queue_param_.mark_dump_step) {
    const auto &dump_step = GetDumpProperties().GetDumpStep();
    GE_CHK_STATUS_RET(CpuMarkStep(model_queue_param_.group_total_count, model_queue_param_.group_index,
                                  model_queue_param_.group_policy, dump_step),
                      "[Call][CpuMarkStep] failed, model_id: %u", model_id_);
  }
  if (use_control_input_queue_) {
    GE_CHK_STATUS_RET(BindControlInputQueue(), "[Bind][ControlInputQueue] failed, model_id: %u", model_id_);
  } else {
    // Binding input_queue and Data Op.
    GE_CHK_STATUS_RET(BindInputQueue(), "[Bind][InputQueue] failed, model_id: %u", model_id_);
    ZeroCpyArgs input_args{};
    input_args.cpy_type = ZeroCpyType::kMixedCpy;
    input_args.has_tensor_desc = has_no_tiling_input_;
    input_args.fusion_offsets = input_fusion_offsets_;
    if (model_queue_param_.io_with_tensor_desc) {
      input_args.has_tensor_desc = true;
    }
    input_args.need_distribute = true;
    GE_CHK_STATUS_RET(CpuTaskModelZeroCopy(input_mbuf_list_, input_data_info_, input_no_tiling_flag_, input_args),
                      "[Call][CpuTaskModelZeroCopy] failed, model_id: %u", model_id_);
    // only helper need this task now
    GE_CHK_STATUS_RET(CpuStaticInputShapeValidate(),
                      "[Call][CpuStaticInputShapeValidate] failed, model_id: %u", model_id_);
    if (model_queue_param_.copy_inputs_for_non_zero_copy) {
      GE_CHK_STATUS_RET(CpuInputCopyProcess(), "[Call][CpuInputCopyProcess] failed, model_id: %u", model_id_);
    }
  }

  // Binding output_queue and NetOutput Op.
  if (use_control_output_queue_) {
    GE_CHK_STATUS_RET(BindControlOutputQueue(), "[Bind][ControlOutputQueue] failed, model_id: %u", model_id_);
  } else {
    GE_CHK_STATUS_RET(BindOutputQueue(), "[Bind][OutputQueue] failed, model_id: %u", model_id_);
    ZeroCpyArgs output_args{};
    output_args.cpy_type = ZeroCpyType::kAllStatic;
    output_args.has_tensor_desc = has_no_tiling_output_;
    output_args.fusion_offsets = std::vector<int32_t>(output_queue_attrs_.size());
    if (model_queue_param_.io_with_tensor_desc) {
      output_args.has_tensor_desc = true;
    }
    output_args.need_distribute = true;
    const auto zero_copy_addrs = FilterZeroCopyAddrs();
    GE_CHK_STATUS_RET(CpuTaskModelZeroCopy(output_mbuf_list_, zero_copy_addrs, output_no_tiling_flag_, output_args),
                      "[Call][CpuTaskModelZeroCopy] failed, model_id: %u", model_id_);
  }

  GE_CHK_STATUS_RET(CpuActiveStream(), "[Call][CpuActiveStream] failed, model_id: %u", model_id_);
  GE_CHK_STATUS_RET(CpuWaitEndGraph(), "[Call][CpuWaitEndGraph] failed, model_id: %u", model_id_);
  if (!use_control_output_queue_) {
    GE_CHK_STATUS_RET(CpuPostProcess(), "[Call][CpuPostProcess] failed, model_id: %u", model_id_);
  }
  GE_CHK_STATUS_RET(BindEnqueue(), "[Call][BindEnqueue] failed, model_id: %u", model_id_);
  if (need_report_status_) {
    GE_CHK_STATUS_RET(CpuModelReportStatus(), "[Call][CpuModelReportStatus] failed, model_id: %u", model_id_);
  }
  GE_CHK_STATUS_RET(CpuModelRepeat(), "[Call][CpuModelRepeat] failed, model_id: %u", model_id_);

  return SUCCESS;
}

Status DavinciModel::LaunchDqsTask(rtDqsTaskType task_type, void* cfg) {
  rtDqsTaskCfg_t task_cfg {};
  task_cfg.type = task_type;
  task_cfg.cfg = cfg;
  GE_ASSERT_RT_OK(rtLaunchDqsTask(rt_entry_stream_, &task_cfg));
  return SUCCESS;
}

Status DavinciModel::LoadWithHardwareQueue() {
  if (!is_hw_q_) {
    return SUCCESS;
  }
  GELOGI("this is hardware queue, use runtime api");
  // create control stream
  GE_ASSERT_SUCCESS(reusable_stream_allocator_->GetOrCreateRtStream(rt_entry_stream_, runtime_model_id_, priority_,
                                                                    RT_STREAM_DQS_CTRL));
  // single input or multi input
  rtDqsSchedConfig_t cfg {};
  cfg.type = static_cast<uint8_t>(RT_DQS_SCHED_TYPE_NN);
  cfg.reserve = 0;
  cfg.inputQueueNum = input_queue_attrs_.size();
  cfg.outputQueueNum = output_queue_attrs_.size();
  for (size_t i = 0; i < input_queue_attrs_.size(); ++i) {
    GELOGI("input %zu queue id is %u", i, input_queue_attrs_[i].queue_id);
    cfg.inputQueueIds[i] = input_queue_attrs_[i].queue_id;
  }
  for (size_t i = 0; i < output_queue_attrs_.size(); ++i) {
    GELOGI("output %zu queue id is %u", i, output_queue_attrs_[i].queue_id);
    cfg.outputQueueIds[i] = output_queue_attrs_[i].queue_id;
  }

  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_SCHED_CONFIG, &cfg));
  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_NOTIFY_WAIT));
  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_DEQUEUE));

  GE_ASSERT_SUCCESS(LaunchInputZeroCpyCfg());
  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_PREPARE_OUT));
  GE_ASSERT_SUCCESS(LaunchOutputZeroCpyCfg());

  GE_ASSERT_RT_OK(rtModelExecute(rt_model_handle_, rt_entry_stream_, 0));
  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_ENQUEUE));
  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_FREE));
  GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_SCHED_END));
  return SUCCESS;
}

Status DavinciModel::GetZcpyReplaceAddrsMap(const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
    std::map<size_t, std::vector<uint64_t>> &replace_addrs_map) {
  for (const auto &addrs : outside_addrs) {
    const size_t data_idx = addrs.first;
    const auto &addrrs_mapping_list = addrs.second.GetOutsideAddrs();
    GE_ASSERT_TRUE(!addrrs_mapping_list.empty());
    for (size_t count = 0U; count < addrrs_mapping_list.size(); ++count) {
      for (const auto &virtual_args : addrrs_mapping_list[count]) {
        for (size_t i = 0U; i < virtual_args.second.size(); ++i) {
          const uintptr_t virtual_addr = virtual_args.second[i];
          replace_addrs_map[data_idx].emplace_back(virtual_addr);
          GELOGI("Index[%zu] addr %p will be repalced", data_idx, ValueToPtr(virtual_addr));
        }
      }
    }
  }
  return SUCCESS;
}

Status DavinciModel::LaunchInputZeroCpyCfg() {
  std::map<size_t, std::vector<uint64_t>> replace_addrs_map;
  GE_ASSERT_SUCCESS(GetZcpyReplaceAddrsMap(input_data_info_, replace_addrs_map));
  for (auto &ele : replace_addrs_map) {
    const size_t input_id = ele.first;
    GE_ASSERT_TRUE(input_id < input_queue_attrs_.size());
    rtDqsZeroCopyCfg_t input_zcpy_cfg {};
    input_zcpy_cfg.copyType = RT_DQS_ZERO_COPY_INPUT;
    input_zcpy_cfg.queueId = input_queue_attrs_[input_id].queue_id;
    input_zcpy_cfg.count = ele.second.size();
    input_zcpy_cfg.dest = ele.second.data();
    std::vector<uint64_t> offset(input_zcpy_cfg.count, 0);
    input_zcpy_cfg.offset = offset.data();
    GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_ZERO_COPY, &input_zcpy_cfg));
  }
  return SUCCESS;
}

Status DavinciModel::LaunchOutputZeroCpyCfg() {
  const auto zero_copy_addrs = FilterZeroCopyAddrs();
  std::map<size_t, std::vector<uint64_t>> replace_addrs_map;
  GE_ASSERT_SUCCESS(GetZcpyReplaceAddrsMap(zero_copy_addrs, replace_addrs_map));
  for (auto &ele : replace_addrs_map) {
    const size_t output_id = ele.first;
    GE_ASSERT_TRUE(output_id < output_queue_attrs_.size());
    rtDqsZeroCopyCfg_t output_zcpy_cfg {};
    output_zcpy_cfg.copyType = RT_DQS_ZERO_COPY_OUTPUT;
    output_zcpy_cfg.queueId = output_queue_attrs_[output_id].queue_id;
    output_zcpy_cfg.count = ele.second.size();
    output_zcpy_cfg.dest = ele.second.data();
    std::vector<uint64_t> offset(output_zcpy_cfg.count, 0);
    output_zcpy_cfg.offset = offset.data();
    GE_ASSERT_RT_OK(LaunchDqsTask(RT_DQS_TASK_ZERO_COPY, &output_zcpy_cfg));
  }
  return SUCCESS;
}

Status DavinciModel::BindControlInputQueue() {
  const uint32_t queue_id = input_queue_attrs_[0U].queue_id;
  GELOGI("using control input queue, queue_id = %u", queue_id);
  GE_CHK_RT_RET(rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_INPUT_QUEUE));
  GE_CHK_STATUS_RET_NOLOG(CpuModelDequeue(queue_id));
  return SUCCESS;
}

/// @ingroup ge
/// @brief queue schedule, Bind  input queue to Data output address.
/// @return: 0 for success / others for failed
Status DavinciModel::BindInputQueue() {
  // Caller checked: input_queue_attrs_.size() == input_size_list_.size() != input_addr_list_.size()
  std::set<uint32_t> unique_inputs;
  for (size_t i = 0U; i < input_queue_attrs_.size(); ++i) {
    const auto it = input_data_info_.find(static_cast<uint32_t>(i));
    if (it == input_data_info_.end()) {
      GELOGE(FAILED, "[Check][Param] Input not match: tensor num=%zu, Queue id index=%zu", input_data_info_.size(), i);
      return FAILED;
    }

    const uint32_t queue_id = input_queue_attrs_[i].queue_id;
    if (unique_inputs.find(queue_id) != unique_inputs.end()) {
      continue;
    }
    (void)unique_inputs.emplace(queue_id);

    if (it->second.GetDataInfo().empty()) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] the %zu input_queue not set data_info.", i);
      return INTERNAL_ERROR;
    }
    const uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0U).first);
    const uintptr_t data_addr = static_cast<uintptr_t>(it->second.GetDataInfo().at(0U).second);
    GELOGI("BindInputToQueue: graph_%u index[%zu] queue id[%u] output addr[0x%" PRIx64 "] output size[%u]",
           runtime_param_.graph_id, i, queue_id, data_addr, data_size);

    GE_CHK_RT_RET(rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_INPUT_QUEUE));
  }

  GE_CHK_STATUS_RET_NOLOG(CpuModelDequeue());
  return SUCCESS;
}

Status DavinciModel::CpuModelDequeue() {
  if (!align_attrs_.empty()) {
    return CpuModelBatchDequeue();
  }
  if ((model_queue_param_.input_align_attrs.align_max_cache_num != 0U) &&
      (input_queue_attrs_.size() > 1UL)) {
    return CpuModelGatherDequeue();
  }

  std::map<uint32_t, size_t> unique_qid_to_idx;
  for (size_t i = 0; i < input_queue_attrs_.size(); ++i) {
    const uint32_t queue_id = input_queue_attrs_[i].queue_id;
    const auto &it = unique_qid_to_idx.find(queue_id);
    if (it != unique_qid_to_idx.cend()) {
      GELOGI("Dequeue task has been added already, queue id = %u, index = %zu.", queue_id, i);
      input_mbuf_list_.push_back(input_mbuf_list_[it->second]);
      continue;
    }
    unique_qid_to_idx[queue_id] = i;
    GE_CHK_STATUS_RET_NOLOG(CpuModelDequeue(queue_id));
  }
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
/// @param [in] queue_id: input queue id from user.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelDequeue(const uint32_t queue_id) {
  GELOGI("Set CpuKernel model dequeue task enter.");
  const auto dequeue_task = MakeShared<CpuTaskModelDequeue>(rt_entry_stream_);
  GE_CHECK_NOTNULL(dequeue_task);

  // Get DataOp Output address and bind to queue.
  uintptr_t in_mbuf = 0U;
  const Status status = dequeue_task->Init(queue_id, in_mbuf);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(dequeue_task);
  input_mbuf_list_.push_back(in_mbuf);
  GELOGI("Set CpuKernel model dequeue task success.");
  return SUCCESS;
}

Status DavinciModel::CpuModelBatchDequeue() {
  GELOGI("Set CpuKernel model batch dequeue task enter.");
  uint32_t align_interval = 0U;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(align_attrs_.cbegin()->second,
                                           ATTR_NAME_INPUTS_ALIGN_INTERVAL,
                                           align_interval),
                         FAILED, "Failed to attr: %s", ATTR_NAME_INPUTS_ALIGN_INTERVAL.c_str());
  GELOGD("align interval = %u", align_interval);
  std::vector<uint32_t> aligned_offsets(input_queue_attrs_.size());
  for (size_t i = 0U; i < input_queue_attrs_.size(); ++i) {
    GE_CHK_BOOL_RET_STATUS(
        AttrUtils::GetInt(align_attrs_[static_cast<uint32_t>(i)], ATTR_NAME_INPUTS_ALIGN_OFFSET, aligned_offsets[i]),
        FAILED, "Failed to attr: %s, input_index = %zu", ATTR_NAME_INPUTS_ALIGN_OFFSET.c_str(), i);
    GELOGD("Input index = %zu, align_offset = %u", i, aligned_offsets[i]);
  }
  const auto dequeue_task = MakeShared<CpuTaskModelBatchDequeue>(rt_entry_stream_);
  GE_CHECK_NOTNULL(dequeue_task);

  // Get DataOp Output address and bind to queue.
  std::vector<uint32_t> input_queue_ids(input_queue_attrs_.size());
  for (size_t index = 0U; index < input_queue_attrs_.size(); index++) {
    input_queue_ids[index] = input_queue_attrs_[index].queue_id;
  }
  GE_CHK_STATUS_RET_NOLOG(dequeue_task->Init(align_interval, input_queue_ids, aligned_offsets, input_mbuf_list_));
  cpu_task_list_.push_back(dequeue_task);
  GELOGI("Set CpuKernel model batch dequeue task success.");
  return SUCCESS;
}

Status DavinciModel::CpuModelGatherDequeue() {
  GELOGI("Set CpuKernel model gather dequeue task enter.");
  const auto dequeue_task = MakeShared<CpuTaskModelGatherDequeue>(rt_entry_stream_);
  GE_CHECK_NOTNULL(dequeue_task);
  GE_CHK_STATUS_RET_NOLOG(dequeue_task->Init(input_queue_attrs_, model_queue_param_.input_align_attrs,
                                             input_mbuf_list_));
  cpu_task_list_.push_back(dequeue_task);
  GELOGI("Set CpuKernel model gather dequeue task success.");
  return SUCCESS;
}

Status DavinciModel::CpuTaskModelZeroCopy(std::vector<uintptr_t> &mbuf_list,
                                          const std::map<uint32_t, ZeroCopyOffset> &outside_addrs,
                                          const std::vector<bool> &is_no_tiling_list,
                                          ZeroCpyArgs &cpy_args) {
  GELOGI("Set CpuKernel model zero_copy task enter.");
  const auto zero_copy = MakeShared<CpuTaskZeroCopy>(rt_entry_stream_);
  GE_CHECK_NOTNULL(zero_copy);

  // mdc zero_copy not support l2 fusion
  const Status status = zero_copy->Init(mbuf_list, outside_addrs, is_no_tiling_list, cpy_args);
  if (status != SUCCESS) {
    return status;
  }

  if (cpy_args.need_distribute) {
    cpu_task_list_.push_back(zero_copy);
    GELOGI("Set CpuKernel model zero_copy task success.");
  }
  return SUCCESS;
}

Status DavinciModel::BindControlOutputQueue() {
  // Caller checked: output_queue_attrs_.size() == output_size_list_.size() != output_addr_list_.size()
  output_mbuf_list_.resize(output_queue_attrs_.size());
  for (size_t i = 0U; i < output_queue_attrs_.size(); ++i) {
    const uint32_t queue_id = output_queue_attrs_[i].queue_id;
    if (queue_id == UINT32_MAX) {
      continue;
    }
    GE_CHK_RT_RET(rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_OUTPUT_QUEUE));
    GELOGD("BindOutputQueue: graph_%u index[%zu] bind queue[%u] success", runtime_param_.graph_id, i, queue_id);
    constexpr uint32_t kControlSize = 1U;
    // add empty desc for control output
    output_descs_.emplace_back(InputOutputDescInfo{});
    GE_CHK_STATUS_RET_NOLOG(CpuModelPrepareOutput(i, 0, kControlSize));
  }
  return SUCCESS;
}

/// @ingroup ge
/// @brief queue schedule, bind output queue to NetOutput input address.
/// @return: 0 for success / others for failed
Status DavinciModel::BindOutputQueue() {
  // Caller checked: input_queue_attrs.size() == input_size_list_.size() != input_addr_list_.size()
  output_mbuf_list_.resize(output_queue_attrs_.size());
  for (size_t i = 0U; i < output_queue_attrs_.size(); ++i) {
    const uint32_t queue_id = output_queue_attrs_[i].queue_id;
    if (queue_id == UINT32_MAX) {
      // dummy output.
      continue;
    }
    GE_CHK_RT_RET(rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_OUTPUT_QUEUE));
    GELOGD("BindOutputQueue: graph_%u index[%zu] bind queue[%u] success", runtime_param_.graph_id, i, queue_id);

    if (output_no_tiling_flag_[i]) {
      GELOGI("BindOutputQueue: output[%zu] support no tiling", i);
      continue;
    }

    const auto it = output_data_info_.find(static_cast<uint32_t>(i));
    if (it == output_data_info_.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Can't find in output_data_info, size:%zu, index:%zu, model_id:%u, check invalid",
                         output_data_info_.size(), i, model_id_);
      GELOGE(FAILED, "[Check][Param] Can't find in output_data_info, size:%zu, Index:%zu, model_id:%u",
             output_data_info_.size(), i, model_id_);
      return FAILED;
    }

    if (it->second.GetDataInfo().empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%zu out_data_info in model:%u is empty, check invalid", i, model_id_);
      GELOGE(INTERNAL_ERROR, "[Check][Param] Index:%zu out_data_info in model:%u is empty, check invalid",
             i, model_id_);
      return INTERNAL_ERROR;
    }
    const uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0U).first);
    const uint64_t data_ptr = it->second.GetDataInfo().at(0U).second;
    const uintptr_t data_addr = static_cast<uintptr_t>(data_ptr);
    GELOGI("BindOutputToQueue: graph_%u index[%zu] queue id[%u] input addr[0x%" PRIx64 "] input size[%u]",
           runtime_param_.graph_id, i, queue_id, data_ptr, data_size);

    if (copy_only_addrs_.Count(data_ptr) != 0) {
      GELOGI("BindOutputQueue: output[%zu] doesn't support zero copy", i);
      continue;
    }

    const Status status = CpuModelPrepareOutput(i, data_addr, data_size);
    if (status != SUCCESS) {
      return status;
    }
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] output_idx: output index.
/// @param [in] addr: NetOutput Op input tensor address.
/// @param [in] size: NetOutput Op input tensor size.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelPrepareOutput(const size_t output_idx, const uintptr_t addr, const uint32_t data_size) {
  GELOGI("Set CpuKernel model enqueue task enter.");
  if (input_mbuf_list_.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "input_mbuf_list_ is empty, model_id:%u, check invalid", model_id_);
    GELOGE(FAILED, "[Check][Param] input_mbuf_list_ is empty, model_id:%u", model_id_);
    return FAILED;
  }

  const auto prepare_output = MakeShared<CpuTaskProcessOutput>(rt_entry_stream_, ProcessStage::kPrepare,
                                                               model_queue_param_.io_with_tensor_desc);
  GE_CHECK_NOTNULL(prepare_output);

  uintptr_t out_mbuf = 0U;
  const InputOutputDescInfo *tensor_desc = nullptr;
  if (model_queue_param_.io_with_tensor_desc) {
    tensor_desc = &output_descs_[output_idx];
  }
  GE_CHK_STATUS_RET_NOLOG(prepare_output->Init(addr, data_size, input_mbuf_list_.front(), out_mbuf, tensor_desc));
  cpu_task_list_.push_back(prepare_output);
  output_mbuf_list_[output_idx] = out_mbuf;
  GELOGI("Set CpuKernel model enqueue task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, active original model stream.
/// @return: 0 for success / others for failed
///
Status DavinciModel::CpuActiveStream() {
  GELOGI("Set CpuKernel active stream task enter.");
  const auto active_entry = MakeShared<CpuTaskActiveEntry>(rt_entry_stream_);
  GE_CHECK_NOTNULL(active_entry);

  const Status status = active_entry->Init(rt_head_stream_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(active_entry);
  GELOGI("Set CpuKernel active stream task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, mark dump step.
/// @return: 0 for success / others for fail
Status DavinciModel::CpuMarkStep(const uint32_t group_total_count,
                                 const uint32_t group_index,
                                 const uint32_t group_policy,
                                 const std::string &dump_step) {
  GELOGD("Set CpuKernel mark step task enter.");
  const auto mark_step = MakeShared<CpuTaskMarkStep>(rt_entry_stream_);
  GE_CHECK_NOTNULL(mark_step);
  GroupInfo group_info{};
  group_info.group_total_count = group_total_count;
  group_info.group_index = group_index;
  group_info.group_policy = group_policy;
  GE_CHK_STATUS_RET(mark_step->Init(group_info, dump_step, global_step_addr_, model_queue_param_.is_head),
                    "CpuKernel mark step task init failed.");

  cpu_task_list_.push_back(mark_step);
  GELOGD("Set CpuKernel mark step task success.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuWaitEndGraph() {
  GELOGI("Set CpuKernel wait end graph task enter.");
  const auto wait_endgraph = MakeShared<CpuTaskWaitEndGraph>(rt_entry_stream_);
  GE_CHECK_NOTNULL(wait_endgraph);

  const Status status = wait_endgraph->Init(runtime_model_id_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(wait_endgraph);
  GELOGI("Set CpuKernel wait end graph task success.");
  return SUCCESS;
}

Status DavinciModel::CpuPostProcess() {
  for (size_t i = 0U; i < output_queue_attrs_.size(); ++i) {
    if (output_queue_attrs_[i].queue_id == UINT32_MAX) {
      continue;
    }
    const auto it = output_data_info_.find(static_cast<uint32_t>(i));
    if (it == output_data_info_.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%zu can't find in output_data_info_ size:%zu in model_id:%u, check invalid",
                         i, output_data_info_.size(), model_id_);
      GELOGE(FAILED, "[Check][Param] Index:%zu can't find in output_data_info_ size:%zu in model_id:%u",
             i, output_data_info_.size(), model_id_);
      return FAILED;
    }

    if (it->second.GetDataInfo().empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%zu out_data_info in model:%u is empty, check invalid", i, model_id_);
      GELOGE(INTERNAL_ERROR, "[Check][Param] Index:%zu out_data_info in model:%u is empty, check invalid",
             i, model_id_);
      return INTERNAL_ERROR;
    }
    const uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0U).first);
    const uint64_t data_addr = it->second.GetDataInfo().at(0U).second;
    const uintptr_t data_ptr = static_cast<uintptr_t>(data_addr);
    GELOGI("CpuPostProcess: graph_%u index[%zu] input addr[0x%" PRIx64 "] input size[%u]",
           runtime_param_.graph_id, i, data_addr, data_size);
    ProcessStage stage;
    if (output_no_tiling_flag_[i]) {
      GELOGI("CpuPostProcess: output[%zu] support no tiling", i);
      stage = ProcessStage::kPostDynamic;
    } else if (copy_only_addrs_.Count(data_addr) != 0) {
      GELOGI("CpuPostProcess: output[%zu] need to non zero copy", i);
      stage = ProcessStage::kPostStatic;
    } else {
      GELOGI("CpuPostProcess: output[%zu] support zero copy, do nothing", i);
      continue;
    }
    GE_CHK_STATUS_RET_NOLOG(CpuModelPostProcess(i, data_ptr, data_size, stage));
  }

  return SUCCESS;
}

Status DavinciModel::CpuModelPostProcess(const size_t output_idx, const uintptr_t addr, const uint32_t data_size,
                                         const ProcessStage stage) {
  GELOGI("Set CpuKernel model post process task enter.");
  if (input_mbuf_list_.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "input_mbuf_list_ is empty, model_id:%u, check invalid", model_id_);
    GELOGE(FAILED, "[Check][Param] input_mbuf_list_ is empty, model_id:%u", model_id_);
    return FAILED;
  }

  const auto post_process = MakeShared<CpuTaskProcessOutput>(rt_entry_stream_, stage,
                                                             model_queue_param_.io_with_tensor_desc);
  GE_CHECK_NOTNULL(post_process);

  uintptr_t out_mbuf = 0U;
  const InputOutputDescInfo *tensor_desc = nullptr;
  if (model_queue_param_.io_with_tensor_desc && (!output_no_tiling_flag_[output_idx])) {
    tensor_desc = &output_descs_[output_idx];
  }
  GE_CHK_STATUS_RET_NOLOG(post_process->Init(addr, data_size, input_mbuf_list_.front(), out_mbuf, tensor_desc));

  cpu_task_list_.push_back(post_process);
  output_mbuf_list_[output_idx] = out_mbuf;
  GELOGI("Set CpuKernel model post process task success.");
  return SUCCESS;
}

Status DavinciModel::BindEnqueue() {
  for (size_t i = 0U; i < output_queue_attrs_.size(); ++i) {
    const uint32_t queue_id = output_queue_attrs_[i].queue_id;
    if (queue_id == UINT32_MAX) {
      continue;
    }
    if (CpuModelEnqueue(queue_id, output_mbuf_list_[i]) != SUCCESS) {
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status DavinciModel::CpuModelEnqueue(const uint32_t queue_id, const uintptr_t out_mbuf) {
  GELOGI("Set CpuKernel model enqueue task enter.");
  const auto model_enqueue = MakeShared<CpuTaskModelEnqueue>(rt_entry_stream_);
  GE_CHECK_NOTNULL(model_enqueue);

  const Status status = model_enqueue->Init(queue_id, out_mbuf);
  if (status != SUCCESS) {
    return status;
  }
  cpu_task_list_.push_back(model_enqueue);
  GELOGI("Set CpuKernel model enqueue task success.");
  return SUCCESS;
}

Status DavinciModel::CpuModelReportStatus() {
  GELOGI("Set CpuKernel report status task enter.");
  const auto model_report_status = MakeShared<CpuTaskModelReportStatus>(rt_entry_stream_);
  GE_CHECK_NOTNULL(model_report_status);

  const Status status = model_report_status->Init(model_uuid_,
                                                  status_output_queue_,
                                                  input_queue_attrs_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(model_report_status);
  GELOGI("Set CpuKernel report status task success.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelRepeat() {
  GELOGI("Set CpuKernel repeat task enter.");
  const auto model_repeat = MakeShared<CpuTaskModelRepeat>(rt_entry_stream_);
  GE_CHECK_NOTNULL(model_repeat);

  const Status status = model_repeat->Init(runtime_model_id_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(model_repeat);
  GELOGI("Set CpuKernel repeat task success.");
  return SUCCESS;
}

Status DavinciModel::CpuStaticInputShapeValidate() {
  if ((!aicpu_resources_.GetStaticModelShapeConfigRet()) || (!model_queue_param_.need_check_inputs)) {
    GELOGI("GetStaticModelShapeConfigRet is not supported or no need check inputs, skip, need_check_inputs[%d].",
           static_cast<int32_t>(model_queue_param_.need_check_inputs));
    return SUCCESS;
  }

  const std::function<bool(std::vector<int64_t>)> is_static_shape = [](const std::vector<int64_t> &dims) -> bool {
    GELOGD("Input shape is %s", ToString(dims).c_str());
    return std::all_of(dims.begin(), dims.end(), [](int64_t dim) -> bool { return dim >= 0; });
  };
  for (const auto &input_desc : origin_input_descs_) {
    const std::vector<int64_t> &shape = input_desc.shape_info.dims;
    if (!is_static_shape(shape)) {
      GELOGI("Input [%s] is not static shape, skip.", input_desc.name.c_str());
      return SUCCESS;
    }
  }

  std::vector<uintptr_t> mbuf_list;
  std::vector<int32_t> input_fusion_offset_list;
  for (size_t i = 0UL; i < input_queue_attrs_.size(); ++i) {
    const auto &is_queue_data_iter = is_queue_data_.find(static_cast<uint32_t>(i));
    if (is_queue_data_iter == is_queue_data_.cend() || is_queue_data_iter->second == false) {
      const auto iter = input_data_info_.find(static_cast<uint32_t>(i));
      GE_CHK_BOOL_RET_STATUS((iter != input_data_info_.end()), INTERNAL_ERROR,
                            "[Check][Param] Index:%zu can't find in input_data_info_ size:%zu in model_id:%u", i,
                            input_data_info_.size(), model_id_);
      GE_CHK_BOOL_RET_STATUS((static_cast<uint64_t>(iter->first) < input_mbuf_list_.size()), INTERNAL_ERROR,
                            "[Check][Param] Data index:%u shold in range of mbuf size:%zu", iter->first,
                            input_mbuf_list_.size());
      GE_CHK_BOOL_RET_STATUS((static_cast<uint64_t>(iter->first) < input_fusion_offsets_.size()), INTERNAL_ERROR,
                            "[Check][Param] Data index:%u shold in range of input_fusion_offsets_ size:%zu",
                            iter->first, input_fusion_offsets_.size());
      mbuf_list.emplace_back(input_mbuf_list_.at(static_cast<size_t>(iter->first)));
      const int32_t input_fusion_offset = input_fusion_offsets_.at(static_cast<size_t>(iter->first));
      input_fusion_offset_list.emplace_back(input_fusion_offset);
      GELOGI("Copy input process task: index:%zu, src muff addr:0x%" PRIx64 ", input fusion offset:%d.", i,
             static_cast<uint64_t>(input_mbuf_list_.at(static_cast<size_t>(iter->first))), input_fusion_offset);
    }
  }
  const auto input_shape_check = MakeShared<CpuTaskProcessInputsShapeCheck>(rt_entry_stream_);
  GE_CHECK_NOTNULL(input_shape_check);
  GE_CHK_STATUS_RET(input_shape_check->Init(mbuf_list, input_fusion_offset_list),
                    "CpuTaskProcessInputsShapeCheck task init failed.");
  cpu_task_list_.push_back(input_shape_check);
  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                            std::vector<InputOutputDescInfo> &output_desc) const {
  if (input_addrs_list_.empty() || (input_addrs_list_[0U].size() != 1U)) {
    GELOGI("data_op_list_ is empty or input_desc size is not 1.");
  } else {
    std::vector<uint32_t> input_formats;
    GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats, false),
                      "[Get][InputDescInfo] failed, model_id: %u", model_id_);
  }

  std::vector<uint32_t> output_formats;
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats),
                    "[Get][OutputDescInfo] failed, model_id: %u", model_id_);
  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                            std::vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &input_formats,
                                            std::vector<uint32_t> &output_formats, const bool by_dims) const {
  if (input_addrs_list_.empty() || (input_addrs_list_[0U].size() != 1U)) {
    GELOGI("data_op_list_ is empty or input_desc size is not 1.");
  } else {
    GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats, by_dims),
                      "[Get][InputDescInfo] failed, model_id: %u", model_id_);
  }
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats),
                    "[Get][OutputDescInfo] failed, model_id: %u", model_id_);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [out] batch_info
/// @param [out] dynamic_type
/// @return execute result
///
Status DavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const {
  dynamic_type = dynamic_type_;
  batch_info = batch_info_;

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [out] batch_info
/// @return None
///
void DavinciModel::GetCombinedDynamicDims(std::vector<std::vector<int64_t>> &batch_info) const {
  batch_info = combined_batch_info_;
}

///
/// @ingroup ge
/// @brief Get user designate shape order
/// @param [out] user_input_shape_order
/// @return None
///
void DavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const {
  user_input_shape_order = user_designate_shape_order_;
}

///
/// @ingroup ge
/// @brief Get AIPP input info
/// @param [in] index
/// @param [int] OpDescPtr
/// @return execute result
///
Status DavinciModel::InitAippInfo(const uint32_t index, const OpDescPtr &op_desc) {
  if (!op_desc->HasAttr(ATTR_NAME_AIPP)) {
    GELOGW("There is not AIPP related with index %u", index);
    return SUCCESS;
  }

  domi::AippOpParams aipp_params;
  NamedAttrs aipp_attr;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr), ACL_ERROR_GE_AIPP_NOT_EXIST,
                         "[Get][NamedAttrs] Data node:%s do not contain param aipp!", op_desc->GetName().c_str());
  GE_CHK_STATUS_RET(OpUtils::ConvertAippParams(aipp_attr, aipp_params),
                    "[Convert][AippParams] get aipp params failed, op: %s", op_desc->GetName().c_str());
  GELOGI("Node data: %s, type: %s, current index: %u, current node related input rank: %u",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), index, aipp_params.related_input_rank());

  AippConfigInfo aipp_info;
  GE_CHK_STATUS_RET(AippUtils::ConvertAippParams2AippInfo(aipp_params, aipp_info),
                    "[Call][ConvertAippParams2AippInfo] failed, op: %s", op_desc->GetName().c_str());

  aipp_info_list_[index] = aipp_info;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get AIPP input info
/// @param [in] index
/// @param [out] aipp_info
/// @return execute result
///
Status DavinciModel::GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const {
  const auto it = aipp_info_list_.find(index);
  if (it == aipp_info_list_.end()) {
    GELOGW("there is not AIPP related with index %u", index);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }

  aipp_info = it->second;
  return SUCCESS;
}

Status DavinciModel::InitAippType(const uint32_t index, const OpDescPtr &op_desc,
                                  const std::map<uint32_t, OpDescPtr> &data_list) {
  if (!op_desc->HasAttr(ATTR_DATA_RELATED_AIPP_MODE)) {
    GELOGW("There is no aipp related info with index %u", index);
    return SUCCESS;
  }

  // Set default value
  InputAippType aipp_type = InputAippType::DATA_WITHOUT_AIPP;
  std::string data_mode;
  (void)AttrUtils::GetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, data_mode);
  if (data_mode == "static_aipp") {
    aipp_type = InputAippType::DATA_WITH_STATIC_AIPP;
  } else if (data_mode == "dynamic_aipp") {
    aipp_type = InputAippType::DATA_WITH_DYNAMIC_AIPP;
  } else if (data_mode == "dynamic_aipp_conf") {
    aipp_type = InputAippType::DYNAMIC_AIPP_NODE;
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s data_mode:%s in op:%s(%s), model_id:%u, check invalid",
                       ATTR_DATA_RELATED_AIPP_MODE.c_str(), data_mode.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(ACL_ERROR_GE_AIPP_MODE_INVALID, "[Get][Attr] %s data_mode:%s in op:%s(%s), model_id:%u, check invalid",
           ATTR_DATA_RELATED_AIPP_MODE.c_str(), data_mode.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return ACL_ERROR_GE_AIPP_MODE_INVALID;
  }

  size_t aipp_index = 0xFFFFFFFFUL;  // default invalid value
  if (aipp_type == InputAippType::DATA_WITH_DYNAMIC_AIPP) {
    std::string releated_name;
    (void)AttrUtils::GetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, releated_name);
    for (const auto &item : data_list) {
      if (item.second->GetName() == releated_name) {
        GELOGI("Find aipp_data [%s] index %u from index %u", releated_name.c_str(), item.first, index);
        aipp_index = item.first;
      }
    }

    if (aipp_index == 0xFFFFFFFFU) {
      GELOGW("Can not find aipp data node from index %u", index);
      return SUCCESS;
    }
  }

  aipp_type_list_[index] = { aipp_type, aipp_index };
  return SUCCESS;
}

Status DavinciModel::GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_index) const {
  GE_CHK_BOOL_RET_STATUS(index < input_addrs_list_.size(), PARAM_INVALID,
                         "[Check][Param] Index %u is invalid", index);
  const auto it = aipp_type_list_.find(index);
  if (it == aipp_type_list_.end()) {
    GELOGW("There is no aipp releated info with index %u", index);
    aipp_type = InputAippType::DATA_WITHOUT_AIPP;
    aipp_index = 0xFFFFFFFFU;
    return SUCCESS;
  }

  aipp_type = it->second.first;
  aipp_index = it->second.second;
  return SUCCESS;
}

void DavinciModel::SetDynamicSize(const std::vector<uint64_t> &batch_num, const int32_t dynamic_type) {
  batch_size_.clear();
  if (batch_num.empty()) {
    GELOGD("User has not set dynammic data");
  }
  for (size_t i = 0U; i < batch_num.size(); ++i) {
    batch_size_.emplace_back(batch_num[i]);
  }

  dynamic_type_ = dynamic_type;
}

void DavinciModel::GetCurrentShape(std::vector<int64_t> &batch_info, int32_t &dynamic_type) const {
  if (batch_size_.empty()) {
    GELOGD("User does not set dynamic size");
  }
  for (size_t i = 0U; i < batch_size_.size(); ++i) {
    GELOGI("Start to get current shape");
    batch_info.emplace_back(batch_size_[i]);
  }

  dynamic_type = dynamic_type_;
}

Status DavinciModel::GetNodeAttr(const std::string &op_name, const std::string &attr_name,
                                 std::string &attr_info) const {
  const auto itr = op_name_to_attrs_.find(op_name);
  if (itr == op_name_to_attrs_.end()) {
    GELOGW("Did not save op:%s attr", op_name.c_str());
    return SUCCESS;
  }
  const auto attr_itr = itr->second.find(attr_name);
  if (attr_itr == itr->second.end()) {
    GELOGW("Did not save attr:%s of op:%s", attr_name.c_str(), op_name.c_str());
    return SUCCESS;
  }
  for (const auto &attr : attr_itr->second) {
    attr_info += "[" + std::to_string(attr.size()) + "]" + attr;
  }
  GELOGD("Get attr:%s of op:%s success, attr value:%s", attr_name.c_str(), op_name.c_str(), attr_info.c_str());
  return SUCCESS;
}

void DavinciModel::GetOutputShapeInfo(std::vector<std::string> &out_shape_info) const {
  (void)out_shape_info.insert(out_shape_info.cend(),
                              dynamic_output_shape_info_.cbegin(),
                              dynamic_output_shape_info_.cend());
}

void DavinciModel::SetInputDimsInfo(const std::vector<int64_t> &input_dims, const Format format,
                                    ShapeDescription &shape_info) const {
  const size_t n = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_N : NCHW_DIM_N);
  const size_t c = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_C : NCHW_DIM_C);
  const size_t h = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_H : NCHW_DIM_H);
  const size_t w = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_W : NCHW_DIM_W);

  if (input_dims.size() == static_cast<size_t>(NORMAL_TENSOR_SIZE)) {
    shape_info.num = input_dims[n];
    shape_info.height = input_dims[h];
    shape_info.width = input_dims[w];
    shape_info.channel = input_dims[c];
  }
  for (size_t k = 0U; k < input_dims.size(); ++k) {
    shape_info.dims.push_back(input_dims[k]);
  }
}

void DavinciModel::CreateInputDimsInfo(const OpDescPtr &op_desc, const Format format,
                                       ShapeDescription &shape_info, ShapeDescription &dims_info) const {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc->GetInputDescPtr(0U));
  // judge if this data is linked dynamic aipp first, multiply batch has been considered
  if (op_desc->HasAttr(ATTR_DYNAMIC_AIPP_INPUT_DIMS)) {
    std::vector<int64_t> dynamic_aipp_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, dynamic_aipp_input_dims);
    SetInputDimsInfo(dynamic_aipp_input_dims, format, shape_info);
  } else {
    // judge if this data is multiply batch
    if (!op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
      const std::vector<int64_t> input_dims = op_desc->GetInputDescPtr(0U)->GetShape().GetDims();
      SetInputDimsInfo(input_dims, format, shape_info);
    } else {
      std::vector<int64_t> origin_input_dims;
      (void)AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
      SetInputDimsInfo(origin_input_dims, format, shape_info);
    }
  }

  if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    // When static aipp is set, need to get the model input dims which processed by aipp
    std::vector<int64_t> model_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    SetInputDimsInfo(model_input_dims, format, dims_info);
  } else {
    dims_info = shape_info;
  }
}

Status DavinciModel::InitInputDescInfo(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(0U));
  GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(0U));

  InputOutputDescInfo input;
  input.data_type = op_desc->GetInputDescPtr(0U)->GetDataType();
  input.name = op_desc->GetName();
  int64_t input_size = 0;
  if (AttrUtils::GetInt(*op_desc->GetOutputDescPtr(0U), ATTR_NAME_SPECIAL_INPUT_SIZE, input_size) &&
      (input_size > 0)) {
    GELOGI("data[%s] output has special size [%" PRId64 "]", op_desc->GetName().c_str(), input_size);
  } else {
    GE_CHK_STATUS_RET(TensorUtils::GetSize(*op_desc->GetInputDescPtr(0U), input_size),
                      "[Get][InputSize] failed in op: %s.", op_desc->GetName().c_str());
  }
  input.size = static_cast<uint64_t>(input_size);

  const Format format = op_desc->GetInputDescPtr(0U)->GetFormat();
  const std::vector<int64_t> input_dims = op_desc->GetInputDescPtr(0U)->GetShape().GetDims();
  InputOutputDescInfo origin_input = input;
  SetInputDimsInfo(input_dims, format, origin_input.shape_info);
  origin_input_descs_.push_back(origin_input);
  ShapeDescription dims_info;
  CreateInputDimsInfo(op_desc, format, input.shape_info, dims_info);

  input_formats_.push_back(format);
  input_descs_.push_back(input);

  input.shape_info = dims_info;
  input_descs_dims_.push_back(input);
  return SUCCESS;
}

Status DavinciModel::GetInputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                      std::vector<uint32_t> &input_format, const bool by_dims) const {
  const std::vector<InputOutputDescInfo> &input_desc_info = by_dims ? input_descs_dims_ : input_descs_;
  (void)input_desc.insert(input_desc.cend(), input_desc_info.cbegin(), input_desc_info.cend());
  (void)input_format.insert(input_format.cend(), input_formats_.cbegin(), input_formats_.cend());

  return SUCCESS;
}

void DavinciModel::CreateOutput(const size_t index, const OpDescPtr &op_desc, InputOutputDescInfo &output,
                                uint32_t &format_result) const {
  /// netoutput input tensor desc
  const auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(index));
  GE_IF_BOOL_EXEC(input_desc == nullptr,
      REPORT_INNER_ERR_MSG("E19999", "input_desc index:%zu in op:%s(%s) does not exist, model_id:%u, check invalid",
                         index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      GELOGE(FAILED, "[Get][InputDescPtr] input_desc index:%zu in op:%s(%s) does not exist, model_id:%u",
             index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      return);
  const auto format = input_desc->GetFormat();
  const auto shape = input_desc->GetShape();

  int64_t dims[] = {1, 1, 1, 1};
  format_result = format;
  if (format == FORMAT_ND) {  // for ND tensor
    for (size_t i = 0U; (i < shape.GetDimNum()) && (i < (sizeof(dims) / sizeof(dims[0]))); ++i) {
      dims[i] = shape.GetDim(i);
    }
  } else {                                                                    // FOR FORMAT_NHWC or FORMAT_NCHW
    dims[0] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_N : NCHW_DIM_N));  // 0: first dim
    dims[1] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_C : NCHW_DIM_C));  // 1: second dim
    dims[2] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_H : NCHW_DIM_H));  // 2: third dim
    dims[3] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_W : NCHW_DIM_W));  // 3: forth dim
  }
  output.shape_info.num = dims[0];      // 0: first dim
  output.shape_info.channel = dims[1];  // 1: second dim
  output.shape_info.height = dims[2];   // 2: third dim
  output.shape_info.width = dims[3];    // 3: forth dim

  if (input_desc->GetFormat() == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    const int64_t k = shape.GetDim(0U);      // 0: first dim
    const int64_t c = shape.GetDim(1U);      // 1: second dim
    const int64_t h = shape.GetDim(2U);      // 2: third dim
    const int64_t w = shape.GetDim(3U);      // 3: forth dim
    output.shape_info.dims.push_back(h);
    output.shape_info.dims.push_back(w);
    output.shape_info.dims.push_back(c);
    output.shape_info.dims.push_back(k);
    format_result = FORMAT_HWCN;
  } else {
    for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
      output.shape_info.dims.push_back(shape.GetDim(j));
    }
  }

  int64_t tensor_size = 0;
  if (AttrUtils::GetInt(input_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size) && (tensor_size > 0)) {
    GELOGI("netoutput[%s] [%zu]th input has special size [%" PRId64 "]", op_desc->GetName().c_str(), index, tensor_size);
  } else {
    (void)TensorUtils::GetTensorSizeInBytes(*input_desc, tensor_size);  // no need to check value
  }
  output.size = static_cast<uint64_t>(tensor_size);
  output.data_type = static_cast<uint32_t>(input_desc->GetDataType());
}

Status DavinciModel::InitOutputDescInfo(const OpDescPtr &op_desc, const std::vector<std::string> &out_node_name) {
  const size_t out_size = op_desc->GetInputsSize();
  for (size_t i = 0U; i < out_size; ++i) {
    std::string output_name;
    InputOutputDescInfo output;
    uint32_t format_result;
    CreateOutput(i, op_desc, output, format_result);

    const auto src_name = op_desc->GetSrcName();
    const auto src_index = op_desc->GetSrcIndex();
    GE_CHK_BOOL_RET_STATUS((src_name.size() > i) && (src_index.size() > i), INTERNAL_ERROR,
                           "[Check][Param] construct output failed, as index:%zu >= src name size:%zu, "
                           "or index >= src index size:%zu, op:%s.",
                           i, src_name.size(), src_index.size(), op_desc->GetName().c_str());
    // forward compatbility, if old om has no out_node_name, need to return output follow origin way
    if (out_size == out_node_name.size()) {
      // neweast plan, the index will add to name during generate model.
      const bool contains_colon = out_node_name[i].find(":") != std::string::npos;
      output_name = contains_colon ? out_node_name[i] : out_node_name[i] + ":" + std::to_string(src_index[i]);
    } else {
      output_name = std::string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
    }
    output.name = output_name;
    output_descs_.push_back(output);
    output_formats_.push_back(format_result);
  }

  return SUCCESS;
}

Status DavinciModel::GetOutputDescInfo(std::vector<InputOutputDescInfo> &output_desc,
                                       std::vector<uint32_t> &output_format) const {
  (void)output_desc.insert(output_desc.cend(), output_descs_.cbegin(), output_descs_.cend());
  (void)output_format.insert(output_format.cend(), output_formats_.cbegin(), output_formats_.cend());
  return SUCCESS;
}

static Status CopyInputForNoTiling(const InputData &input_data, const size_t data_idx, void *&mem_addr) {
  RuntimeTensorDesc tensor_desc;
  // copy data_addr from tensor_desc addr
  GE_CHK_RT_RET(rtMemcpy(&tensor_desc, sizeof(RuntimeTensorDesc), mem_addr, sizeof(RuntimeTensorDesc),
      RT_MEMCPY_DEVICE_TO_HOST));
  if (data_idx >= input_data.shapes.size()) {
    GELOGE(PARAM_INVALID, "invalid index[%zu], input shape size[%zu]", data_idx, input_data.shapes.size());
    return PARAM_INVALID;
  }
  const auto &shape = input_data.shapes[data_idx];
  if (shape.size() > static_cast<size_t>(kMaxDimSize)) {
    GELOGE(PARAM_INVALID, "invalid InputData, input shape[%zu]'s dim size[%zu] > kMaxDimSize[%" PRId64 "]",
           data_idx, shape.size(), kMaxDimSize);
    return PARAM_INVALID;
  }

  tensor_desc.shape[0] = static_cast<int64_t>(shape.size());
  for (size_t i = 0U; i < shape.size(); i++) {
    tensor_desc.shape[i + 1U] = shape[i];
  }
  // fill actual shape and copy to tensor_desc addr
  GE_CHK_RT_RET(
      rtMemcpy(mem_addr, sizeof(RuntimeTensorDesc), &tensor_desc, sizeof(RuntimeTensorDesc),
      RT_MEMCPY_HOST_TO_DEVICE));
  mem_addr = ValueToPtr(tensor_desc.data_addr);
  GELOGD("copy tensor desc for no tiling, data_addr:%p, dim:%" PRId64, mem_addr, tensor_desc.shape[0]);
  return SUCCESS;
}

Status DavinciModel::CopyInputData(const InputData &input_data) {
  const std::vector<DataBuffer> &blobs = input_data.blobs;

  int32_t cur_device_id = -1;
  if (enable_input_batch_cpy_) {
    if (input_data_info_.size() == 1) {
      enable_input_batch_cpy_ = false;
      GELOGW("The switch of input_batch_cpy is open but only one input exists, not enable batch memcpy");
    } else {
      ResetMemcpyBatchParams();
      GE_CHK_RT_RET(aclrtGetDevice(&cur_device_id));
    }
  }
  size_t idx = 0;

  for (const auto &data_info : input_data_info_) {
    const size_t data_idx = data_info.first;
    if (data_idx >= blobs.size()) {
      const std::string reason = "The required input " + std::to_string(data_idx) +
                                 " is not provided by user, while the total input data num is " + std::to_string(blobs.size());
      REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                std::vector<const char_t *>({reason.c_str()}));
      GELOGE(FAILED, "[Check][Param] Blobs not match: blobs=%zu, model input num=%zu, required index=%u, op_name(%s)",
             blobs.size(), input_data_info_.size(), data_idx, data_info.second.GetOpName().c_str());
      return FAILED;
    }

    const DataBuffer &data_buf = blobs.at(data_idx);
    if (data_buf.length == 0U) {
      GELOGW("No data need to copy, index=%u", data_idx);
      continue;
    }

    const uint64_t data_size = static_cast<uint64_t>(data_info.second.GetDataSize());
    GE_CHK_BOOL_RET_STATUS(data_size >= data_buf.length, PARAM_INVALID,
                           "[Check][Param] input data size(%" PRIu64 ") is bigger than model required size(%" PRIu64
                           "), op_name(%s)",
                           data_buf.length, data_size, data_info.second.GetOpName().c_str());
    void *mem_addr = data_info.second.GetBasicAddr();
    bool is_no_tiling = false;
    if (data_idx < input_no_tiling_flag_.size()) {
      is_no_tiling = input_no_tiling_flag_[data_idx];
    } else {
      GELOGW("[Check][Param]invalid input_no_tiling_flag_ size[%zu], index[%u]",
             input_no_tiling_flag_.size(), data_idx);
    }
    if (is_no_tiling) {
      // mem_addr will be changed to data addr here
      GE_CHK_STATUS_RET_NOLOG(CopyInputForNoTiling(input_data, data_idx, mem_addr));
    }

    GELOGI("CopyPlainData memcpy graph_%u type[F] input[%s] rank[%u] dst[%p] src[%p] "
           "mem_size[%" PRIu64 "] datasize[%" PRIu64 "]",
           runtime_param_.graph_id, data_info.second.GetOpName().c_str(), data_idx, mem_addr, data_buf.data,
           data_size, data_buf.length);

    const auto kind = GetRtMemcpyKindByPlacement(data_buf.placement, true);
    // 目前只有开启了批拷贝开关+H2D场景支持batch memcpy
    memcpy_batch_params_.device_id = cur_device_id;
    if (!enable_input_batch_cpy_ || kind != RT_MEMCPY_HOST_TO_DEVICE) {
      GE_CHK_RT_RET(rtMemcpy(mem_addr, data_size, data_buf.data, data_buf.length, kind));
    } else {
      MemcpyParam memcpy_param {mem_addr, data_size, data_buf.data, data_buf.length, idx++};
      TensorTransUtils::AddMemcpyBatchParam(memcpy_param, memcpy_batch_params_);
    }
  }

  return TensorTransUtils::TryBatchMemcpy(memcpy_batch_params_);
}

Status DavinciModel::CopyInputDataWithMergeH2D(const InputData &input_data) {
  const std::vector<DataBuffer> &blobs = input_data.blobs;
  std::vector<size_t> non_merge_copy_indexs;
  void *input_merge_copy_device_addr = nullptr;

  for (const auto &data_info : input_data_info_) {
    const size_t data_idx = data_info.first;
    if (data_idx >= blobs.size()) {
      const std::string reason = "The required input " + std::to_string(data_idx) +
                                 " is not provided by user, while the total input data num is " + std::to_string(blobs.size());
      REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                std::vector<const char_t *>({reason.c_str()}));
      GELOGE(FAILED, "[Check][Param] Blobs not match: blobs=%zu, model input num=%zu, required index=%zu, op_name(%s)",
             blobs.size(), input_data_info_.size(), data_idx, data_info.second.GetOpName().c_str());
      return FAILED;
    }
    // find device addr for merge copy
    input_merge_copy_device_addr =
        (data_idx == fisrt_input_index_of_merge_copy_) ? data_info.second.GetBasicAddr() : input_merge_copy_device_addr;

    const DataBuffer &data_buf = blobs.at(data_idx);
    if (data_buf.length == 0U) {
      GELOGW("No data need to copy, index=%u", data_idx);
      continue;
    }
    const uint64_t data_size = static_cast<uint64_t>(data_info.second.GetDataSize());
    GE_CHK_BOOL_RET_STATUS(data_size >= data_buf.length, PARAM_INVALID,
                           "[Check][Param] input data size(%" PRIu64 ") is bigger than model required size(%" PRIu64
                           "), index: %zu, op_name(%s)",
                           data_buf.length, data_size, data_idx, data_info.second.GetOpName().c_str());

    const auto kind = GetRtMemcpyKindByPlacement(data_buf.placement, true);
    const auto &merge_copy_offset = input_index_to_merge_copy_offset_.find(data_idx);

    if ((kind != RT_MEMCPY_HOST_TO_DEVICE) || (merge_copy_offset == input_index_to_merge_copy_offset_.end())) {
      GELOGD("index[%zu] push back to non_merge_copy_indexs", data_idx);
      non_merge_copy_indexs.push_back(data_idx);
      continue;
    }

    // copy input to host buffer, h2h
    const auto host_offset = merge_copy_offset->second;
    GELOGI("[InputMergeCopy] Copy2H graph_%u type[F] input[%s] index[%zu] offset[%" PRIu64
           "] src[%p] mem_size[%" PRIu64 "] datasize[%" PRIu64 "]",
           runtime_param_.graph_id, data_info.second.GetOpName().c_str(), data_idx, host_offset, data_buf.data,
           data_size, data_buf.length);
    const auto mem_ret = GeMemcpy(input_merge_copy_mem_base_.get() + host_offset,
                                  input_merge_copy_mem_size_ - host_offset,
                                  reinterpret_cast<uint8_t *>(data_buf.data),
                                  data_buf.length);
    GE_CHK_BOOL_RET_STATUS(mem_ret == SUCCESS, FAILED,
                           "memcpy fail, graph %u, index %zu, data len:%" PRIu64 ", buffer size:%" PRIu64
                           ", offset:%" PRIu64,
                           runtime_param_.graph_id, data_idx, data_buf.length, input_merge_copy_mem_size_, host_offset);
  }

  // merge copy input to device buffer, h2d
  GELOGI("[InputMergeCopy]CopyPlainData graph_%u type[F] dst[%p] src[%p] mem_size[%" PRIu64 "].",
         runtime_param_.graph_id, input_merge_copy_device_addr, input_merge_copy_mem_base_.get(),
         input_merge_copy_mem_size_);
  GE_CHECK_NOTNULL(input_merge_copy_device_addr,
                   "invalid input_merge_copy_device_addr value, input_merge_copy_device_addr is nullptr");
  GE_CHK_RT_RET(rtMemcpy(input_merge_copy_device_addr, input_merge_copy_mem_size_, input_merge_copy_mem_base_.get(),
                         input_merge_copy_mem_size_, RT_MEMCPY_HOST_TO_DEVICE));
  // copy non merge copy input

  int32_t cur_device_id = -1;
  if (enable_input_batch_cpy_) {
    GE_CHK_RT_RET(aclrtGetDevice(&cur_device_id));
    ResetMemcpyBatchParams();
  }
  size_t idx = 0;

  for (const auto &data_idx : non_merge_copy_indexs) {
    const DataBuffer &data_buf = blobs.at(data_idx);
    const auto &data_info = input_data_info_[data_idx];
    const uint64_t data_size = static_cast<uint64_t>(data_info.GetDataSize());
    void *mem_addr = data_info.GetBasicAddr();
    GELOGI("[InputMergeCopy]CopyPlainData graph_%u type[F] input[%s] rank[%u] dst[%p] src[%p] mem_size[%" PRIu64 "]"
           " datasize[%" PRIu64 "]",
           runtime_param_.graph_id, data_info.GetOpName().c_str(), data_idx, mem_addr, data_buf.data,
           data_size, data_buf.length);
    const auto kind = GetRtMemcpyKindByPlacement(data_buf.placement, true);
    // 目前只有H2D场景支持batch memcpy
    memcpy_batch_params_.device_id = cur_device_id;
    if (!enable_input_batch_cpy_ || kind != RT_MEMCPY_HOST_TO_DEVICE) {
      GELOGD("Call rtMemcpy for non_merge_copy_indexs");
      GE_CHK_RT_RET(rtMemcpy(mem_addr, data_size, data_buf.data, data_buf.length, kind));
    } else {
      MemcpyParam memcpy_param {mem_addr, data_size, data_buf.data, data_buf.length, idx++};
      TensorTransUtils::AddMemcpyBatchParam(memcpy_param, memcpy_batch_params_);
    }
  }

  return TensorTransUtils::TryBatchMemcpy(memcpy_batch_params_);
}

void DavinciModel::ResetMemcpyBatchParams() {
  memcpy_batch_params_.dsts.clear();
  memcpy_batch_params_.dst_aligned_sizes.clear();
  memcpy_batch_params_.srcs.clear();
  memcpy_batch_params_.src_sizes.clear();
  memcpy_batch_params_.attrs.clear();
  memcpy_batch_params_.attr_idxs.clear();
  memcpy_batch_params_.device_id = 0;
}

void DavinciModel::InitBatchMemcpyH2d() {
  std::string input_batch_cpy_str;
  (void)GetThreadLocalContext().GetOption(configure_option::INPUT_BATCH_CPY, input_batch_cpy_str);
  enable_input_batch_cpy_ = (!input_batch_cpy_str.empty() && input_batch_cpy_str == "1");
  GELOGI("Init davinci_model input_batch_cpy_:%d.", static_cast<int32_t>(enable_input_batch_cpy_));
}

Status DavinciModel::HandleInputData(InputData &input_data) {
  GE_TIMESTAMP_START(Model_SyncVarData);
  if (UpdateStepInfoWithStream() != SUCCESS) {
    return FAILED;
  }
  if (is_first_execute_) {
    GE_TIMESTAMP_EVENT_END(Model_SyncVarData, "Model Run SyncVarData");
  }

  GELOGI("Copy input data, model id:%u", model_id_);
  const bool dynamic_shape_data = is_online_infer_dynamic_ && (!is_getnext_sink_dynamic_);
  if (dynamic_shape_data) {
    cur_dynamic_dims_.clear();
    if (GetCurDynamicDims(input_data.shapes, cur_dynamic_dims_) != SUCCESS) {
      return INTERNAL_ERROR;
    }

    DataBuffer data;
    data.data = cur_dynamic_dims_.data();
    data.length = static_cast<uint32_t>(cur_dynamic_dims_.size() * sizeof(int32_t));
    input_data.blobs.push_back(data);
  }

  Status status = SUCCESS;
  GE_TIMESTAMP_START(CopyInputData);

  if (has_no_tiling_input_ || (input_merge_copy_mem_base_ == nullptr)) {
    status = CopyInputData(input_data);
  } else {
    status = CopyInputDataWithMergeH2D(input_data);
  }
  GE_TIMESTAMP_EVENT_END_WITH_FLAG(CopyInputData, "CopyInputData", mdl_prof_.enable_flag);

  if (dynamic_shape_data) {
    input_data.blobs.pop_back();
  }

  return status;
}

Status DavinciModel::InitFusionProfiling(const FusionOpInfo &fusion_op_info) {
  const auto &op_desc = GetOpByIndex(fusion_op_info.op_index);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Get op by index failed, as index:%u out of range", fusion_op_info.op_index);
    GELOGE(FAILED, "[Get][Op] failed, as index:%u out of range", fusion_op_info.op_index);
    return FAILED;
  }

  ProfileInfo profile;
  profile.fusion_info = fusion_op_info;
  // save fusion op info into MsprofGeProfFusionData list
  SaveFusionOpInfo(op_desc, profile);
  profile_list_.emplace_back(profile);
  GELOGD("Add fusion task, profile info size: %zu", profile_list_.size());
  return SUCCESS;
}

Status DavinciModel::ReportProfilingData(const uint32_t graph_id) {
  auto &prof_mgr = ProfilingManager::Instance();

  // davinci model report only one time during the training
  if (prof_mgr.IsGraphProfReported(graph_id)) {
    GELOGD("[Profiling] graph id %u has been reported.", graph_id);
    return SUCCESS;
  }
  // Report profiling data
  const auto ret = ReportProfilingData();
  // graph id is UINT32_MAX on execution
  if ((ret == SUCCESS) && (graph_id != UINT32_MAX)) {
    prof_mgr.InsertReportedGraphId(graph_id);
  }
  return ret;
}

void DavinciModel::SetClearDfxCacheFlagAfterInit(const bool clear_cache) {
  GELOGI("set clear_cache %d", static_cast<int32_t>(clear_cache));
  need_clear_dfx_cache_ = clear_cache;
}

bool DavinciModel::NeedClearDfxCacheFlagAfterInit() const {
  return need_clear_dfx_cache_;
}

bool DavinciModel::HasZeroCopyAddr(const OpDescPtr &op_desc) const {
  const auto input_addrs = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc);
  const auto output_addrs = ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc);
  std::vector<uint64_t> io_addrs;
  (void)io_addrs.insert(io_addrs.cend(), input_addrs.begin(), input_addrs.end());
  (void)io_addrs.insert(io_addrs.cend(), output_addrs.begin(), output_addrs.end());

  const auto zero_copy_args_index = GetZeroCopyArgsIndex(io_addrs);
  return !zero_copy_args_index.empty();
}

Status DavinciModel::ReportTaskTimeL1Info() {
  if (!gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kDevice)) {
    GELOGI("Do not report l1 info.");
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(ReportFusionOpInfo(), "Report profiling fusion op info failed");
  for (auto &node_basic_info : node_basic_infos_) {
    if (node_basic_info.node_basic_info.data.nodeBasicInfo.opName == 0UL) {
      node_basic_info.node_basic_info.data.nodeBasicInfo.opName =
          MsprofGetHashId(node_basic_info.op_name.c_str(), node_basic_info.op_name.length());
    }
    if (node_basic_info.node_basic_info.data.nodeBasicInfo.opType == 0UL) {
      node_basic_info.node_basic_info.data.nodeBasicInfo.opType =
          MsprofGetHashId(node_basic_info.op_type.c_str(), node_basic_info.op_type.length());
    }
    GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(static_cast<uint32_t>(false), &node_basic_info.node_basic_info,
                                                static_cast<uint32_t>(sizeof(MsprofCompactInfo))));
  }

  for (const auto &task_desc_info : task_desc_info_) {
    GE_ASSERT_SUCCESS(
        gert::GlobalProfilingWrapper::ReportTensorInfo(model_load_event_.threadId, false, task_desc_info));
  }
  return SUCCESS;
}

Status DavinciModel::ReportTaskTimeL0Info(const uint32_t prof_model_id) {
  if (!gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
    GELOGI("Do not report l0 info.");
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(gert::GlobalProfilingWrapper::ReportLogicStreamInfo(load_end_time_, model_load_event_.threadId,
                                                                        logic_stream_ids_to_physic_stream_ids_,
                                                                        static_cast<uint16_t>(false)));
  GE_CHK_STATUS_RET(ReportModelExtInfo(model_load_event_.threadId, prof_model_id),
                    "Report profiling model ext info failed");
  for (auto &context_info_id : context_id_infos_) {
    auto prof_context_info = reinterpret_cast<MsprofContextIdInfo *>(context_info_id.context_id_info.data);
    if (prof_context_info->opName == 0UL) {
      prof_context_info->opName = MsprofGetHashId(context_info_id.op_name.c_str(), context_info_id.op_name.length());
    }
    GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(static_cast<uint32_t>(false), &context_info_id.context_id_info,
                                                   static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
  }

  for (auto &launch_api : prof_launch_apis_) {
    if (launch_api.api.itemId == 0UL) {
      launch_api.api.itemId = MsprofGetHashId(launch_api.op_name.c_str(), launch_api.op_name.length());
    }
    GE_ASSERT_MSPROF_OK(MsprofReportApi(static_cast<uint32_t>(false), &launch_api.api));
  }
  return SUCCESS;
}

Status DavinciModel::ReportProfilingData() {
  GELOGI("Report profiling data.");
  const uint32_t prof_model_id = (model_id_ == std::numeric_limits<uint32_t>::max()
                                      ? gert::GlobalProfilingWrapper::GetInstance()->GetProfModelId()
                                      : model_id_);
  model_load_event_.type = static_cast<uint32_t>(gert::GeProfInfoType::kModelLoad);
  model_load_event_.itemId = prof_model_id;
  model_load_event_.level = MSPROF_REPORT_MODEL_LEVEL;
  model_load_event_.timeStamp = load_begin_time_;
  model_load_event_.requestId = 0U;
  GE_ASSERT_MSPROF_OK(MsprofReportEvent(static_cast<uint32_t>(false), &model_load_event_));
  GE_ASSERT_SUCCESS(ReportTaskTimeL0Info(prof_model_id), "Report profiling task time l0 info failed.");
  GE_ASSERT_SUCCESS(ReportTaskTimeL1Info(), "Report profiling task time l1 info failed");
  model_load_event_.timeStamp = load_end_time_;
  GE_ASSERT_MSPROF_OK(MsprofReportEvent(static_cast<uint32_t>(false), &model_load_event_));
  return SUCCESS;
}

void DavinciModel::ClearProfilingDataCache() {
  GELOGI("clear profiling data cache start.");
  /* clear Task Time L1Info */
  node_basic_infos_.clear();
  task_desc_info_.clear();
  /* clear Task Time L0Info */
  context_id_infos_.clear();
  prof_launch_apis_.clear();
  GELOGI("clear profiling data cache end.");
  return;
}

void DavinciModel::SaveFusionOpInfo(const OpDescPtr &op_desc, ProfileInfo &profile) const {
  // init memory info
  gert::ProfFusionMemSize fusion_mem{};
  InitMemoryInfo(op_desc, fusion_mem.input_mem_size, fusion_mem.output_mem_size, fusion_mem.workspace_mem_size,
                 fusion_mem.weight_mem_size);
  const size_t op_name = MsprofGetHashId(op_desc->GetName().c_str(), op_desc->GetName().length());
  for (auto iter = profile.fusion_info.original_op_names.begin();
       iter != profile.fusion_info.original_op_names.end();) {
    if ((*iter).empty()) {
      iter = profile.fusion_info.original_op_names.erase(iter);
    } else {
      ++iter;
    }
  }
  gert::GlobalProfilingWrapper::BuildFusionOpInfo(fusion_mem, profile.fusion_info.original_op_names, op_name,
                                                  profile.prof_fusion_data_lst);
}

Status DavinciModel::ReportFusionOpInfo() {
  GELOGD("Report profiling fusion_op_info, model id is %u", model_id_);
  for (auto &profile : profile_list_) {
    for (size_t i = 0UL; i < profile.prof_fusion_data_lst.size(); ++i) {
      auto fusion_op_info = reinterpret_cast<ProfFusionOpInfo *>(profile.prof_fusion_data_lst[i].data);
      if (fusion_op_info->opName == 0UL) {
        fusion_op_info->opName =
            MsprofGetHashId(profile.fusion_info.op_name.c_str(), profile.fusion_info.op_name.length());
        for (size_t j = 0UL; j < fusion_op_info->fusionOpNum; ++j) {
          const size_t origin_op_idx = i * static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM) + j;
          GE_ASSERT_TRUE(origin_op_idx < profile.fusion_info.original_op_names.size());
          fusion_op_info->fusionOpId[j] =
              MsprofGetHashId(profile.fusion_info.original_op_names[origin_op_idx].c_str(),
                              profile.fusion_info.original_op_names[origin_op_idx].length());
        }
      }
      GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(static_cast<uint32_t>(false), &profile.prof_fusion_data_lst[i],
                                                     static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))));
    }
  }
  return SUCCESS;
}

Status DavinciModel::ReportModelExtInfo(const uint32_t tid, const uint32_t model_id) {
  const std::string model_name = om_name_.empty() ? name_ : om_name_;
  const size_t model_name_hash = MsprofGetHashId(model_name.c_str(), model_name.length());
  GELOGD("Report profiling id map info. model name is %s, hash is %zu", model_name.c_str(), model_name_hash);
  // if it is not online model, there is no graph id
  if (!domi::GetContext().is_online_model) {
    GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::ReportGraphIdMap(
        load_end_time_, tid, {std::numeric_limits<uint32_t>::max(), model_id}, false, model_name_hash));
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::ReportGraphIdMap(load_end_time_, tid, {GetGraphId(), model_id}, false,
                                                                   model_name_hash));
  return ge::SUCCESS;
}

void DavinciModel::SinkTimeProfile(const uint32_t data_index, const uint64_t request_id) {
  if (!gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
    return;
  }
  GELOGI("Report profiling model execute info in, model id is %u", model_id_);
  (void)data_index;
  thread_local const auto tid = mmGetTid();
  input_data_pre_.threadId = static_cast<uint32_t>(tid);
  input_data_pre_.level = MSPROF_REPORT_MODEL_LEVEL;
  input_data_pre_.type = static_cast<uint32_t>(static_cast<uint32_t>(gert::GeProfInfoType::kInputCopy));
  input_data_pre_.itemId = model_id_;
  (void)MsprofReportApi(static_cast<uint32_t>(true), &input_data_pre_);
  MsprofEvent model_execute{};
  model_execute.level = MSPROF_REPORT_MODEL_LEVEL;
  model_execute.itemId = model_id_;
  model_execute.threadId = static_cast<uint32_t>(tid);
  model_execute.type = static_cast<uint32_t>(gert::GeProfInfoType::kModelExecute);
  model_execute.timeStamp = execute_start_time_;
  model_execute.requestId = static_cast<uint32_t>(request_id);
  (void)MsprofReportEvent(static_cast<uint32_t>(true), &model_execute);
  model_execute.timeStamp = execute_end_time_;
  (void)MsprofReportEvent(static_cast<uint32_t>(true), &model_execute);
  output_data_pre_.threadId = static_cast<uint32_t>(tid);
  output_data_pre_.level = MSPROF_REPORT_MODEL_LEVEL;
  output_data_pre_.type = static_cast<uint32_t>(gert::GeProfInfoType::kOutputCopy);
  output_data_pre_.itemId = model_id_;
  (void)MsprofReportApi(static_cast<uint32_t>(true), &output_data_pre_);
  GELOGI("Report profiling model execute info out, model id is %u", model_id_);
}

void DavinciModel::SetProfileTime(const ModelProcStage stage, const uint64_t end_time) {
  uint64_t prof_time = end_time;
  if (prof_time == 0U) {
    prof_time = MsprofSysCycleTime();
  }
  thread_local const auto tid = mmGetTid();
  switch (stage) {
    case ModelProcStage::MODEL_LOAD_START:
      load_begin_time_ = prof_time;
      model_load_event_.threadId = static_cast<uint32_t>(tid);
      break;
    case ModelProcStage::MODEL_LOAD_END:
      load_end_time_ = prof_time;
      break;
    case ModelProcStage::MODEL_PRE_PROC_START:
      input_data_pre_.beginTime = prof_time;
      break;
    case ModelProcStage::MODEL_PRE_PROC_END:
      input_data_pre_.endTime = prof_time;
      break;
    case ModelProcStage::MODEL_INFER_START:
      execute_start_time_ = prof_time;
      break;
    case ModelProcStage::MODEL_INFER_END:
      execute_end_time_ = prof_time;
      break;
    case ModelProcStage::MODEL_AFTER_PROC_START:
      output_data_pre_.beginTime = prof_time;
      break;
    case ModelProcStage::MODEL_AFTER_PROC_END:
      output_data_pre_.endTime = prof_time;
      break;
    default:
      break;
  }
}

static Status CheckBufferSizeValid(const bool is_dynamic, const bool is_no_tiling, const uint64_t buffer_length,
                                   const uint64_t data_size) {
  if (is_dynamic || is_no_tiling) {
    GELOGI("No need to check output data size.");
    return SUCCESS;
  }
  if (buffer_length < data_size) {
    GELOGE(PARAM_INVALID, "invalid output buffer length[%zu], data size[%zu].", buffer_length, data_size);
    return FAILED;
  }
  if (buffer_length > data_size) {
    GELOGW("Tensor data size=%" PRIu64 ", buffer size=%" PRIu64, data_size, buffer_length);
  }
  return SUCCESS;
}

Status DavinciModel::CopyOutputForNoZeroCopy(const std::vector<GeTensor> &output_tensor,
                                             const std::vector<DataBuffer> &blobs,
                                             const std::map<uint32_t, MemAllocationSlice> &copy_infos) {
  uint32_t blobs_size = blobs.size();
  bool isBlobsEmpty = blobs.size() == 0U;
  for (const auto &item : copy_infos) {
    const uint32_t output_idx = item.first;
    size_t id = static_cast<size_t>(item.second.id);
    id = (id == 0xFFFFFFFFU) ? (logical_mem_allocations_.size() - 1U) : id;
    const uint64_t offset = item.second.offset;
    uint64_t data_size = item.second.data_size;

    GE_ASSERT_TRUE((output_idx < (!isBlobsEmpty ? blobs_size : output_tensor.size())),
                  "invalid output_index:%u, blobs size:%zu", output_idx,
                  !isBlobsEmpty ? blobs_size : output_tensor.size());

    uint64_t buffer_length = !isBlobsEmpty ? blobs.at(static_cast<size_t>(output_idx)).length :
        output_tensor.at(output_idx).GetData().size();
    uint32_t buffer_placement = !isBlobsEmpty ? blobs.at(static_cast<size_t>(output_idx)).placement :
        static_cast<uint32_t>(Placement::kPlacementDevice);
    void *data = !isBlobsEmpty ? blobs.at(static_cast<size_t>(output_idx)).data :
        ValueToPtr(PtrToValue(output_tensor[output_idx].GetData().data()));

    std::vector<int64_t> output_shape;
    int64_t data_size_t = -1;
    GE_CHK_STATUS_RET(BuildOutputShapeInfo(static_cast<size_t>(output_idx), output_shape, data_size_t),
                      "BuildOutputShapeInfo failed, output_idx:[%" PRId64 "].", output_idx);
    data_size = (data_size_t == -1) ? data_size : static_cast<uint64_t>(data_size_t);
    GE_ASSERT_TRUE((data_size <= buffer_length), "invalid buffer_length:%u, data size:%zu", buffer_length, data_size);
    if ((buffer_length == 0U) || (data_size == 0U)) {
      GELOGI("Length of data is zero, No need copy. output tensor index=%u", output_idx);
      continue;
    }

    GE_ASSERT_TRUE((id < logical_mem_allocations_.size()), "invalid id:%zu, active_allocations size:%zu",
                   id, logical_mem_allocations_.size());
    const void *const src_addr = ValueToPtr(allocation_ids_to_active_base_addr_[id] + offset);
    const auto kind = (buffer_placement == kPlaceHostData) ? RT_MEMCPY_DEVICE_TO_HOST : RT_MEMCPY_DEVICE_TO_DEVICE;
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[ForceCopy%s] model_id:%u, output_index:%u, id:%zu, offset:0x%" PRIx64 ", base:0x%" PRIx64 ", dst_addr:%p, "
             "dst_size:%" PRIu64 ", src_addr:%p, src_size:%" PRIu64 ", async_mode:%d, kind:%d",
             ((src_addr == data) ? " skip" : ""),
             model_id_, output_idx, id, offset, allocation_ids_to_active_base_addr_[id], data, buffer_length,
             src_addr, data_size, static_cast<int32_t>(is_async_mode_), static_cast<int32_t>(kind));
    }
    if (src_addr == data) {
      continue;
    }

    if (is_async_mode_) {
      GE_CHK_RT_RET(rtMemcpyAsyncWithoutCheckKind(data, buffer_length, src_addr, data_size,
                                                  kind, rt_model_stream_));
    } else {
      GE_CHK_RT_RET(rtMemcpy(data, buffer_length, src_addr, data_size, kind));
    }
  }

  return SUCCESS;
}

Status DavinciModel::CopyOutputForNoZeroCopy(const std::vector<gert::Tensor> &output_tensor,
                                             const std::map<uint32_t, MemAllocationSlice> &copy_infos) {
  for (const auto &item : copy_infos) {
    const uint32_t output_idx = item.first;
    size_t id = static_cast<size_t>(item.second.id);
    id = (id == 0xFFFFFFFFU) ? (logical_mem_allocations_.size() - 1U) : id;
    const uint64_t offset = item.second.offset;
    uint64_t data_size = item.second.data_size;

    GE_ASSERT_TRUE((output_idx < output_tensor.size()),
                  "invalid output_index:%u, blobs size:%zu", output_idx,
                  output_tensor.size());

    uint64_t buffer_length = output_tensor.at(output_idx).GetSize();
    uint32_t buffer_placement = output_tensor.at(output_idx).GetPlacement();
    void *data = ValueToPtr(PtrToValue(output_tensor[output_idx].GetAddr()));

    std::vector<int64_t> output_shape;
    int64_t data_size_t = -1;
    GE_CHK_STATUS_RET(BuildOutputShapeInfo(static_cast<size_t>(output_idx), output_shape, data_size_t),
                      "BuildOutputShapeInfo failed, output_idx:[%" PRId64 "].", output_idx);
    data_size = (data_size_t == -1) ? data_size : static_cast<uint64_t>(data_size_t);
    GE_ASSERT_TRUE((data_size <= buffer_length), "invalid buffer_length:%u, data size:%zu", buffer_length, data_size);
    if ((buffer_length == 0U) || (data_size == 0U)) {
      GELOGI("Length of data is zero, No need copy. output tensor index=%u", output_idx);
      continue;
    }

    GE_ASSERT_TRUE((id < logical_mem_allocations_.size()), "invalid id:%zu, active_allocations size:%zu",
                   id, logical_mem_allocations_.size());
    const void *const src_addr = ValueToPtr(allocation_ids_to_active_base_addr_[id] + offset);
    const auto kind = (buffer_placement == static_cast<uint32_t>(gert::kOnDeviceHbm)) ? RT_MEMCPY_DEVICE_TO_DEVICE : RT_MEMCPY_DEVICE_TO_HOST;
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[ForceCopy%s] model_id:%u, output_index:%zu, id:%zu, offset:0x%" PRIx64 ", base:0x%" PRIx64 ", "
             "dst_addr:%p, dst_size:%" PRIu64 ", src_addr:%p, src_size:%" PRIu64 ", async_mode:%d, kind:%d",
             ((src_addr == data) ? " skip" : ""),
             model_id_, output_idx, id, offset, allocation_ids_to_active_base_addr_[id], data, buffer_length,
             src_addr, data_size, static_cast<int32_t>(is_async_mode_), static_cast<int32_t>(kind));
    }
    if (src_addr == data) {
      continue;
    }

    if (is_async_mode_) {
      GE_CHK_RT_RET(rtMemcpyAsyncWithoutCheckKind(data, buffer_length, src_addr, data_size,
                                                  kind, rt_model_stream_));
    } else {
      GE_CHK_RT_RET(rtMemcpy(data, buffer_length, src_addr, data_size, kind));
    }
  }

  return SUCCESS;
}

Status DavinciModel::UpdateStepInfoWithStream() {
  // iterator_count_ used both in tran and inferance, to get(or manager) resouces between diffrence run times
  if ((global_step_addr_ != 0U) && (global_step_size_ != 0U)) {
    if (is_async_mode_) {
      GE_CHK_RT_RET(rtMemcpyAsync(ValueToPtr(static_cast<uint64_t>(global_step_addr_)), global_step_size_,
                                  &iterator_count_, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE_EX, rt_model_stream_));
    } else {
      GE_CHK_RT_RET(rtMemcpy(ValueToPtr(static_cast<uint64_t>(global_step_addr_)), global_step_size_,
                             &iterator_count_, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));
    }
  }
  return SUCCESS;
}

Status DavinciModel::CopyOutputData(const std::vector<gert::Tensor> &output_tensor) {
  if (!has_output_node_) {
    return UpdateStepInfoWithStream();
  }

  GE_ASSERT_SUCCESS(CopyOutputForNoZeroCopy(output_tensor, output_indexes_to_copy_info_));
  if (host_pls_output_indexes_to_copy_info_.size() != 0U) {
    GE_ASSERT_SUCCESS(CopyOutputForNoZeroCopy(output_tensor, host_pls_output_indexes_to_copy_info_));
    host_pls_output_indexes_to_copy_info_.clear();
  }

  return SUCCESS;
}

Status DavinciModel::CopyOutputData(const OutputData &output_data,
    const std::vector<GeTensor> &output_tensor) {
  if (!has_output_node_) {
    return UpdateStepInfoWithStream();
  }

  GE_ASSERT_SUCCESS(CopyOutputForNoZeroCopy(output_tensor, output_data.blobs, output_indexes_to_copy_info_));
  if (host_pls_output_indexes_to_copy_info_.size() != 0U) {
    GE_ASSERT_SUCCESS(CopyOutputForNoZeroCopy(output_tensor, output_data.blobs, host_pls_output_indexes_to_copy_info_));
    host_pls_output_indexes_to_copy_info_.clear();
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] data_id: the index of output_data
/// @param [in/out] output_data: real user output_data
/// @param [in] kind: the kind of rtMemcpy
/// @return Status result
/// @author
///
Status DavinciModel::CopyOutputDataLegacy(const OutputData &output_data) {
  if (!has_output_node_) {
    return UpdateStepInfoWithStream();
  }

  if (output_data.blobs.size() != output_data_info_.size()) {
    REPORT_INNER_ERR_MSG("E19999", "output_data.blobs.size:%zu != output_data_info.size:%zu, model_id:%u, check invalid",
                       output_data.blobs.size(), output_data_info_.size(), model_id_);
    GELOGE(FAILED, "[Check][Param] output_data.blobs.size:%zu != output_data_info.size:%zu, model_id:%u",
           output_data.blobs.size(), output_data_info_.size(), model_id_);
    return FAILED;
  }

  const std::vector<DataBuffer> &blobs = output_data.blobs;
  for (const auto &output : output_data_info_) {
    const size_t output_idx = output.first;
    if (output_idx >= blobs.size()) {
      REPORT_INNER_ERR_MSG("E19999", "index:%u in output_data_info_ >= output_data.blobs.size:%zu, model_id:%u, "
                         "check invalid", output.first, blobs.size(), model_id_);
      GELOGE(FAILED, "[Check][Param] index:%u in output_data_info_ >= output_data.blobs.size:%zu, model_id:%u",
             output.first, blobs.size(), model_id_);
      return FAILED;
    }

    void *output_addr = output.second.GetBasicAddr();
    const DataBuffer &buffer = blobs.at(output_idx);
    auto kind = GetRtMemcpyKindByPlacement(buffer.placement, false);
    const bool feed_by_zero_copy = (kind == RT_MEMCPY_DEVICE_TO_DEVICE) &&
                                   (copy_only_addrs_.Count(PtrToValue(output_addr)) == 0);
    if (feed_by_zero_copy) {
      continue;  // Skip: Feed by zero copy.
    }

    const uint64_t mem_size = static_cast<uint64_t>(output.second.GetDataSize());
    if ((buffer.length == 0U) || (mem_size == 0U)) {
      GELOGI("Length of data is zero, No need copy. output tensor index=%u", output.first);
      continue;
    }
    const bool is_no_tiling = (output_idx < output_no_tiling_flag_.size()) ? output_no_tiling_flag_[output_idx] : false;
    GE_CHK_STATUS_RET(CheckBufferSizeValid(is_dynamic_, is_no_tiling, buffer.length, mem_size),
                      "[Check][Param] Buffer.length:%" PRIu64 " in output blob < data_size:%" PRIu64
                      " in output_data_info, index:%u, "
                      "model_id:%u.", buffer.length, mem_size, output.first, model_id_);
    // refresh copied data len
    int64_t data_size = output.second.GetDataSize();
    if (is_no_tiling || is_online_infer_dynamic_) {
      std::vector<int64_t> output_shape;
      GE_CHK_STATUS_RET(BuildOutputShapeInfo(output_idx, output_shape, data_size),
                        "BuildOutputShapeInfo failed, output_idx:[%" PRId64 "].", output_idx);
      const auto iter = output_no_tiling_data_addr_.find(output.first);
      if (iter != output_no_tiling_data_addr_.end()) {
        output_addr = ValueToPtr(iter->second);
      }
    }
    const uint64_t copied_size = static_cast<uint64_t>(data_size);
    kind = (is_async_mode_ && (kind == RT_MEMCPY_HOST_TO_DEVICE)) ? RT_MEMCPY_HOST_TO_DEVICE_EX : kind;
    GELOGI("CopyPlainData memcpy %s graph_%u type[F] output[%u] dst[%p] memaddr[%p] "
           "mem_size[%" PRIu64 "] datasize[%" PRIu64 "]",
           (is_async_mode_ ? "async" : "sync"), runtime_param_.graph_id, output.first, buffer.data, output_addr,
           copied_size, buffer.length);
    if (is_async_mode_) {
      GE_CHK_RT_RET(rtMemcpyAsync(buffer.data, buffer.length, output_addr,
                                  copied_size, kind, rt_model_stream_));
    } else {
      GE_CHK_RT_RET(rtMemcpy(buffer.data, buffer.length, output_addr, copied_size, kind));
    }
  }
  return SUCCESS;
}

Status DavinciModel::InitOutputTensorInfo(const OpDescPtr &op_desc) {
  size_t input_num = op_desc->GetInputsSize();
  if (is_getnext_sink_dynamic_) {
    GE_CHECK_GE(input_num, kGetDynamicDimsCount);
    input_num = input_num - kGetDynamicDimsCount;
  }

  for (size_t i = 0U; i < input_num; ++i) {
    int64_t size = 0;
    const auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(input_desc);
    const auto ret = TensorUtils::GetTensorSizeInBytes(*input_desc, size);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get input TensorSize in op:%s(%s) failed, input_index:%zu, model_id:%u",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), i, model_id_);
      GELOGE(ret, "[Get][InputTensorSize] in op:%s(%s) failed, input_index:%zu, model_id:%u",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), i, model_id_);
      return ret;
    }
    const GeShape &shape = input_desc->GetShape();
    bool is_no_tiling = false;
    (void)AttrUtils::GetBool(input_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
    GELOGI("Output size is %" PRId64 ", output shape is %s, no tiling is %d.",
           size, ToString(shape.GetDims()).c_str(), static_cast<int32_t>(is_no_tiling));
    output_buffer_size_.emplace_back(size);
    output_shape_info_.emplace_back(shape);
    output_no_tiling_flag_.push_back(is_no_tiling);
    if (is_no_tiling) {
      has_no_tiling_output_ = true;
    }
  }

  return SUCCESS;
}

Status DavinciModel::BuildOutputShapeInfo(const size_t output_idx, std::vector<int64_t> &output_shape,
                                          int64_t &output_size) {
  if ((output_idx < output_no_tiling_flag_.size()) && output_no_tiling_flag_[output_idx]) {
    const auto output = output_data_info_.find(static_cast<uint32_t>(output_idx));
    if (output == output_data_info_.end()) {
      GELOGE(FAILED, "output_data_info_[%zu] is empty.", output_idx);
      return FAILED;
    }
    RuntimeTensorDesc tensor_desc;
    GE_CHK_RT_RET(rtMemcpy(&tensor_desc, sizeof(RuntimeTensorDesc), output->second.GetBasicAddr(),
                           sizeof(RuntimeTensorDesc), RT_MEMCPY_DEVICE_TO_HOST));
    const int64_t dim_num = tensor_desc.shape[0];
    for (int64_t dim_loop = 0; dim_loop < dim_num; dim_loop++) {
      output_shape.emplace_back(tensor_desc.shape[dim_loop + 1]);
    }
    GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSize(GeShape(output_shape),
                                                     static_cast<Format>(tensor_desc.format),
                                                     static_cast<DataType>(tensor_desc.dtype), output_size),
                      "tensor[%zu] CalcTensorMemSize failed.", output_idx);
    output_no_tiling_data_addr_[static_cast<uint32_t>(output_idx)] = tensor_desc.data_addr;
    GELOGD("Output [%zu] is no tiling, out desc addr[%p], data addr[%#lx].",
           output_idx, output->second.GetBasicAddr(), tensor_desc.data_addr);
  } else {
    output_shape = output_shape_info_[output_idx].GetDims();
    if (is_online_infer_dynamic_) {
      const auto it0 = merge_nodes_gear_and_real_out_size_info_.find(output_idx);
      if (it0 != merge_nodes_gear_and_real_out_size_info_.end()) {
        const auto size_it = it0->second.find(cur_dynamic_dims_);
        output_size = (size_it != it0->second.end()) ? size_it->second : 0;
        is_dynamic_ = true;
      }
      const auto it1 = merge_nodes_gear_and_real_out_shape_info_.find(output_idx);
      if (it1 != merge_nodes_gear_and_real_out_shape_info_.end()) {
        const auto shape_it = it1->second.find(cur_dynamic_dims_);
        output_shape = (shape_it != it1->second.end()) ? shape_it->second : std::vector<int64_t>{};
      }
    }
  }
  return SUCCESS;
}

void DavinciModel::GeShapeAsRtShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) const {
  gert_shape.SetDimNum(ge_shape.GetDims().size());
  for (size_t i = 0U; i < ge_shape.GetMutableDims().size(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
}

std::vector<int64_t> DavinciModel::GetGertTensorDims(gert::Shape &gert_shape) const {
  std::vector<int64_t> dims;
  size_t dim_num = gert_shape.GetDimNum();
  dims.resize(dim_num);
  for (size_t i = 0U; i < dim_num; i++) {
    dims[i] = gert_shape[i];
  }
  return dims;
}

void DavinciModel::UpdateOutputTensorShape(std::vector<gert::Tensor> &output_tensor) {
  for (size_t output_idx = 0UL; output_idx < output_tensor.size(); ++output_idx) {
    if ((output_idx < output_no_tiling_flag_.size()) && output_no_tiling_flag_[output_idx]) {
      GeShapeAsRtShape(output_shape_info_[output_idx],
          output_tensor[output_idx].MutableStorageShape());
      GELOGI("Output index[%zu] shape update to [%s]", output_idx,
             ToString(output_shape_info_[output_idx].GetDims()).c_str());
      continue;
    }
    const auto it1 = merge_nodes_gear_and_real_out_shape_info_.find(output_idx);
    if (it1 != merge_nodes_gear_and_real_out_shape_info_.cend()) {
      const auto shape_it = it1->second.find(cur_dynamic_dims_);
      if (shape_it != it1->second.cend()) {
        const auto before_dims = GetGertTensorDims(output_tensor[output_idx].MutableStorageShape());
        GeShapeAsRtShape(GeShape(shape_it->second),
            output_tensor[output_idx].MutableStorageShape());
        const auto after_dims = GetGertTensorDims(output_tensor[output_idx].MutableStorageShape());
        GELOGI("Output index[%zu] shape update from [%s] to [%s].",
               output_idx, ToString(before_dims).c_str(), ToString(after_dims).c_str());
      }
    }
  }
}

void DavinciModel::UpdateOutputTensorShape(std::vector<GeTensor> &output_tensor) {
  if ((!is_online_infer_dynamic_) && (!has_no_tiling_output_)) {
    GELOGI("No need to update output tensor shape.");
    return;
  }
  for (size_t output_idx = 0UL; output_idx < output_tensor.size(); ++output_idx) {
    if ((output_idx < output_no_tiling_flag_.size()) && output_no_tiling_flag_[output_idx]) {
      output_tensor[output_idx].MutableTensorDesc().SetShape(output_shape_info_[output_idx]);
      GELOGI("Output index[%zu] shape update to [%s]", output_idx,
             ToString(output_shape_info_[output_idx].GetDims()).c_str());
      continue;
    }
    const auto it1 = merge_nodes_gear_and_real_out_shape_info_.find(output_idx);
    if (it1 != merge_nodes_gear_and_real_out_shape_info_.cend()) {
      const auto shape_it = it1->second.find(cur_dynamic_dims_);
      if (shape_it != it1->second.cend()) {
        const auto before_dims = output_tensor[output_idx].GetTensorDesc().GetShape().GetDims();
        output_tensor[output_idx].MutableTensorDesc().SetShape(GeShape(shape_it->second));
        const auto after_dims = output_tensor[output_idx].GetTensorDesc().GetShape().GetDims();
        GELOGI("Output index[%zu] shape update from [%s] to [%s].",
               output_idx, ToString(before_dims).c_str(), ToString(after_dims).c_str());
      }
    }
  }
}

Status DavinciModel::GenOutputTensorInfo(OutputData &output_data, std::vector<gert::Tensor> &outputs) {
  if (!output_data.blobs.empty()) {
    GELOGI("No need to generate output tensor info, model id:%u", model_id_);
    return SUCCESS;
  }

  std::vector<int64_t> output_size_info;
  std::vector<std::vector<int64_t>> output_shape_info;
  const size_t output_num = output_buffer_size_.size();
  for (size_t i = 0U; i < output_num; ++i) {
    if (output_no_tiling_flag_[i] && (output_data_info_.find(static_cast<uint32_t>(i)) == output_data_info_.end())) {
      GELOGW("output_data_info_[%zu] is empty.", i);
      continue;
    }

    std::vector<int64_t> output_shape;
    int64_t output_size = output_buffer_size_[i];
    GE_CHK_STATUS_RET_NOLOG(BuildOutputShapeInfo(i, output_shape, output_size));

    GELOGI("Output size is %" PRId64 ", output shape is %s.", output_size, ToString(output_shape).c_str());
    output_size_info.push_back(output_size);
    output_shape_info.push_back(output_shape);
  }

  GELOGI("Output blobs size:%zu, model id:%u", output_size_info.size(), model_id_);
  for (size_t i = 0U; i < output_size_info.size(); ++i) {
    const auto output_buffer = MakeShared<AlignedPtr>(output_size_info[i], kMemAlignment);
    GE_CHECK_NOTNULL(output_buffer);
    const GeTensorDesc tensor_desc(GeShape(output_shape_info[i]), FORMAT_ND, DT_FLOAT);
    GeTensor ge_tensor(tensor_desc);
    ge_tensor.SetData(output_buffer, static_cast<size_t>(output_size_info[i]));

    void *const data_ptr = output_buffer->MutableGet();
    output_data.blobs.push_back({data_ptr, static_cast<uint64_t>(output_size_info[i])});
    gert::Tensor gert_tensor;
    GE_ASSERT_SUCCESS(TensorTransUtils::GeTensor2GertTensor(ge_tensor, gert_tensor));
    outputs.emplace_back(std::move(gert_tensor));
    GELOGD("Output index:%zu, output dims is %s, data length:%" PRIu64 ".",
           i, ToString(output_shape_info[i]).c_str(), output_size_info[i]);
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] data_id: the index of output_data
/// @param [in] rslt_flg: result flag
/// @param [in] seq_end_flag: sequence end flag
/// @param [out] output_data: real user output_data
/// @return Status result
/// @author
///
void DavinciModel::AssembleListenerOutput(const std::shared_ptr<RunArgs> &args, const uint32_t data_id,
                                          std::vector<gert::Tensor> &outputs) {
  if (listener_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "listener_ is nullptr, check invalid.");
    GELOGE(PARAM_INVALID, "listener_ is nullptr, check invalid, model id: %u", model_id_);
    return;
  }

  if (is_getnext_sink_dynamic_) {
    GELOGD("Reinit cur dynamic dims when getnext sink dynamic.");
    cur_dynamic_dims_.clear();
    cur_dynamic_dims_.resize(shape_of_cur_dynamic_dims_);
    const auto ret = rtMemcpy(cur_dynamic_dims_.data(), shape_of_cur_dynamic_dims_ * sizeof(int32_t),
                              netoutput_last_input_addr_, static_cast<uint64_t>(netoutput_last_input_size_),
                              RT_MEMCPY_DEVICE_TO_HOST);
    GE_CHK_RT_EXEC(ret, return);
  }
  OutputData output_data;
  GELOGD("Cur dynamic dims is %s.", ToString(cur_dynamic_dims_).c_str());
  if (GenOutputTensorInfo(output_data, outputs) != SUCCESS) {
    return;
  }

  if (CopyOutputDataLegacy(output_data) != SUCCESS) {
    OnComputeDoneWithResultCallback(args, data_id, INTERNAL_ERROR, outputs);
  }
}

void DavinciModel::ReturnSequenceResult(const std::shared_ptr<RunArgs> &args, const uint32_t data_id,
  bool seq_end_flag) {
  std::vector<gert::Tensor> outputs;
  OutputData output_data;
  if (!seq_end_flag || !has_output_node_) {
    OnComputeDoneWithResultCallback(args, data_id, INTERNAL_ERROR, outputs);
  } else {
    AssembleListenerOutput(args, data_id, outputs);
    GELOGW("End of sequence, model id: %u", model_id_);
    OnComputeDoneWithResultCallback(args, data_id, END_OF_SEQUENCE, outputs);
  }
}

void DavinciModel::OnComputeDoneWithResultCallback(const std::shared_ptr<RunArgs> &args, const uint32_t data_id,
                                                   uint32_t result, std::vector<gert::Tensor> &outputs) {
  // 静态shape和动态shape都启用了扩展模式
  std::shared_ptr<ActiveMemoryAllocator> mem_allocator = nullptr;
  if (support_extend_memory_full_) {
    mem_allocator =
        SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(session_id_, GetDeviceId());
    if (mem_allocator != nullptr) {
      mem_allocator->Recycle(active_memorys_);
    }
  }
  if ((args != nullptr) && (args->callback != nullptr)) {
    args->callback(result, outputs);
  } else {
    OnComputeDoneWithResult(data_id, result, outputs);
  }
}

void DavinciModel::OnComputeDoneWithResult(const uint32_t data_id, uint32_t result,
  std::vector<gert::Tensor> &outputs) {
  if (listener_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "listener_ is nullptr, check invalid.");
    GELOGE(PARAM_INVALID, "[Check][Param]listener_ is nullptr, check invalid.");
    return;
  }

  GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, result, outputs),
                "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
}

uint32_t DavinciModel::GetResultCode() {
  GE_CHK_BOOL_EXEC(listener_ != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "listener_ is nullptr, check invalid.");
                   return PARAM_INVALID, "[Check][Param] listener_ is null!");
  return listener_->GetResultCode();
}

Status DavinciModel::ResetResult() {
  GE_CHK_BOOL_EXEC(listener_ != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "listener_ is nullptr, check invalid.");
                   return PARAM_INVALID, "[Check][Param] listener_ is null!");
  return listener_->ResetResult();
}

void DavinciModel::ConstructActiveMemBaseAddrs() {
  bool need_dev_va_2_pa = false;
  (void)rtNeedDevVA2PA(&need_dev_va_2_pa);
  for (size_t i = 0U; i < logical_mem_allocations_.size(); i++) {
    if (need_dev_va_2_pa && ModelUtils::IsReuseZeroCopyMemory() &&
        ((logical_mem_allocations_[i].type == ge::MemAllocation::Type::INPUT) ||
         (logical_mem_allocations_[i].type == ge::MemAllocation::Type::OUTPUT))) {
      allocation_ids_to_active_base_addr_[i] = 0U;
    } else {
      allocation_ids_to_active_base_addr_[i] = logical_mem_allocations_[i].logical_addr;
    }
    GELOGI("[ActiveMemBase], model_id:%u, id:%zu, active mem base:0x%" PRIx64, model_id_, i,
           allocation_ids_to_active_base_addr_[i]);
  }
}

void DavinciModel::Run() {
  SET_THREAD_NAME(pthread_self(), "ge_davidmdlrun");
  const uint32_t run_dev_id = device_id_;
  error_message::SetErrMgrContext(error_message::GetErrMgrContext());

  GELOGI("Model Run thread start, model_id:%u.", model_id_);
  ModelUtils::SetDevice(run_dev_id);

  // set graphLevelSat
  if (isGraphLevelSat_) {
    (void)ge::rtSetStreamTag(rt_model_stream_, model_id_);
  }
  // DeviceReset before thread run finished!
  GE_MAKE_GUARD(reset_device, [run_dev_id]() {
    GE_CHK_STATUS(ModelUtils::ResetDevice(run_dev_id));
  });

  while (run_flg_) {
    // Model hasn't truly started running before received data
    const bool is_prof_enabled = gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime);
    SetRunningFlag(false);
    std::vector<gert::Tensor> outputs;
    std::shared_ptr<RunArgs> args;
    Status ret = data_inputer_.Pop(args);
    const bool close_terminated = (args == nullptr) || (ret != SUCCESS) ||
      (!run_flg_);
    if (close_terminated) {
      GELOGW("args is null or data queue closed, exit!");
      break;
    }

    if (!copy_host_input_indexes_.empty()) {
      GELOGW("ge.exec.hostInputIndexes does not support RunGraphAsync!");
    }

    SetRunningFlag(true);
    InputData current_data;
    const std::vector<gert::Tensor> &inputs = args->input_tensor;
    current_data.blobs.reserve(inputs.size());
    current_data.shapes.reserve(inputs.size());
    for (size_t i = 0U; i < inputs.size(); ++i) {
      current_data.shapes.emplace_back(TensorTransUtils::GetDimsFromGertShape(inputs[i].GetStorageShape()));
      DataBuffer data_blob;
      data_blob.data = ValueToPtr(PtrToValue(inputs[i].GetAddr()));
      data_blob.length = inputs[i].GetSize();
      data_blob.placement = static_cast<uint32_t>(gert::TensorPlacementUtils::IsOnDevice(inputs[i].GetPlacement()) ?
        ge::Placement::kPlacementDevice : ge::Placement::kPlacementHost);
      current_data.blobs.push_back(data_blob);
    }
    if (MallocPhysicalMemory() != SUCCESS) {
      OnComputeDoneWithResultCallback(args, 0U, INTERNAL_ERROR, outputs);
      return;
    }

    // 当前流程模型的所有输入都走强制拷贝, 此处只需刷新1次就可以
    args_manager_.InitDfxStage1Begin();
    if (!is_first_time_model_execute_) {
      // 首次刷新为all-one-time
      uint32_t ret_up = kUpdatePolicyAllOneTime;
      ConstructActiveMemBaseAddrs();
      if (rt_model_stream_!=nullptr) {
        GELOGW("model stream is null");
      }
      if (args_manager_.UpdateForExecute(ret_up, rt_model_stream_) != SUCCESS) {
        GELOGE(FAILED, "UpdateForExecute, model_id:%u.", model_id_);
        OnComputeDoneWithResultCallback(args, 0U, INTERNAL_ERROR, outputs);
        return;
      }
      is_first_time_model_execute_ = true;
    }
    args_manager_.InitDfxStatsticsEnd();
    args_manager_.PrintDfxStatistics();

    // Model run indeedly start after received data.
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_PRE_PROC_START));
    GELOGI("Model thread Run begin, model id:%u, data index:%u.", model_id_, 0U);
    ret = HandleInputData(current_data);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Call][HandleInputData] handle input data failed, model_id:%u.", model_id_);
      OnComputeDoneWithResultCallback(args, 0U, INTERNAL_ERROR, outputs);
      continue;
    }
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_PRE_PROC_END));
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_INFER_START));
    GE_TIMESTAMP_START(rtModelExecute);
    GELOGI("rtModelExecute start, model id:%u.", model_id_);
    CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 0U, rt_model_stream_);
    auto rt_ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0U);
    if (rt_ret != RT_ERROR_NONE) {
      OnComputeDoneWithResultCallback(args, 0U, INTERNAL_ERROR, outputs);
      continue;
    }
    GELOGI("rtModelExecute end, model id:%u.", model_id_);
    CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 1U, rt_model_stream_);
    GE_IF_BOOL_EXEC(is_first_execute_, GE_TIMESTAMP_EVENT_END(rtModelExecute, "rtModelExecute"));
    iterator_count_++;

    GE_TIMESTAMP_START(rtStreamSynchronizeWithTimeout);
    GELOGI("rtStreamSynchronizeWithTimeout start, model id:%u.", model_id_);
    rt_ret = rtStreamSynchronizeWithTimeout(rt_model_stream_, stream_sync_timeout_);
    if (rt_ret == ACL_ERROR_RT_SOCKET_CLOSE) {
      GELOGI("connect lost to model exec, befause socket closed, model_id:%u", model_id_);
      ModelManager::GetInstance().SetSocketCloseStatus(true);
    }
    if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      is_stream_sync_timeout_ = true;
      GE_LOGW_IF(rtModelAbort(rt_model_handle_) != RT_ERROR_NONE, "Abort model failed!");
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%dms, ret:%d.",
                        stream_sync_timeout_, rt_ret);
      GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, timeout:%dms, ret:%d.", stream_sync_timeout_,
             rt_ret);
      OnComputeDoneWithResultCallback(args, 0U, INTERNAL_ERROR, outputs);
      return;
    }
    const bool model_abort = ((rt_ret == kSinkModelAbortNormal) || (rt_ret == kSinkModelAbortNormalNew));
    if ((!model_abort) && (rt_ret != RT_ERROR_NONE)) {
      const bool seq_end_flag = ((rt_ret == kSinkModelEndOfSequence) || (rt_ret == kSinkModelEndOfSequenceNew));
      GELOGI("seq_end_flg: %d", static_cast<int32_t>(seq_end_flag));
      ReturnSequenceResult(args, 0U, seq_end_flag);
      continue;
    }
    GELOGI("rtStreamSynchronizeWithTimeout end, model id:%u, status:%s.", model_id_, model_abort ? "abort" : "normal");
    GE_IF_BOOL_EXEC(is_first_execute_,
                    GE_TIMESTAMP_EVENT_END(rtStreamSynchronizeWithTimeout, "Wait for rtStreamSynchronizeWithTimeout"));
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_INFER_END));
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_AFTER_PROC_START));
    GE_TIMESTAMP_START(ReturnResult);
    if (has_output_node_) {
      AssembleListenerOutput(args, 0U, outputs);
    }
    GE_IF_BOOL_EXEC(is_first_execute_, GE_TIMESTAMP_EVENT_END(ReturnResult, "CopyDataFromDeviceToHost"));
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_AFTER_PROC_END));
    GE_IF_BOOL_EXEC(is_prof_enabled, SinkTimeProfile(0U, iterator_count_ - 1UL));
    OnComputeDoneWithResultCallback(args, 0U, SUCCESS, outputs);
    is_first_execute_ = false;
    // model run finished
    GELOGI("run iterator count is %" PRIu64 ", model_id:%u", iterator_count_, model_id_);
  }
  GELOGI("Model run end, model id:%u", model_id_);
}

///
/// @ingroup ge
/// @brief call API provided by data inputer to destroy thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::DestroyThread() {
  run_flg_ = false;

  data_inputer_.Stop();
  if (thread_id_.joinable()) {
    thread_id_.join();
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief create model std::thread,
/// @brief start to execute Model
/// @param [in] no
/// @return Status create model thread and execute result
/// @author
///
Status DavinciModel::ModelRunStart() {
  const std::unique_lock<std::mutex> lk(mux_run_flg_);
  GE_CHK_BOOL_RET_STATUS(!run_flg_, INTERNAL_ERROR, "[Check][Param] Model already started, model id:%u.", model_id_);
  run_flg_ = true;

  // create stream instance which rt_model_handel is running on
  const uint32_t stream_flags = (GetContext().IsOverflowDetectionOpen()) ? RT_STREAM_OVERFLOW : RT_STREAM_DEFAULT;
  GE_CHECK_NOTNULL(reusable_stream_allocator_);
  GE_ASSERT_SUCCESS(
      reusable_stream_allocator_->GetOrCreateRtStream(rt_model_stream_, runtime_model_id_, priority_, stream_flags));
  is_inner_model_stream_ = true;
  GE_CHK_RT_RET(rtStreamSetMode(rt_model_stream_, kStopOnFailure));
  error_context_ = error_message::GetErrMgrContext();
  thread_id_ = std::thread(&DavinciModel::Run, this);

  GELOGI("model thread create success, model id:%u.", model_id_);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief call API provided by data inputer and destroy model Thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::ModelRunStop() {
  const std::unique_lock<std::mutex> lk(mux_run_flg_);
  GE_CHK_STATUS_RET(DestroyThread(), "[Destoy][Thead] failed, model id: %u.", model_id_);
  return SUCCESS;
}

void DavinciModel::UnbindTaskSinkStream() {
  GELOGD("Npu model: %u start to unbind streams.", model_id_);
  // check rt ctx is exist. rt api call will cause error log when ctx does not exist
  rtContext_t current_ctx = nullptr;
  if (rtCtxGetCurrent(&current_ctx) != RT_ERROR_NONE) {
    return;
  }

  // unbinding hcom stream
  UnbindHcomStream();
  if (is_stream_list_bind_) {
    for (const auto &iter : stream_to_first_task_id_) {
      // unbind rt_model_handle and streams
      auto stream_id = iter.first;
      GE_LOGE_IF(static_cast<size_t>(stream_id) >= stream_list_.size(), "stream id %zu is invalid", stream_id);
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, stream_list_[stream_id]) != RT_ERROR_NONE,
                 "Unbind stream from model failed! Index: %zu", stream_id);
    }
  }

  if (is_inner_model_stream_) {
    if ((!input_queue_attrs_.empty()) || (!output_queue_attrs_.empty()) || is_stream_sync_timeout_) {
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_model_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    }
  }

  if (is_pure_head_stream_ && (rt_head_stream_ != nullptr)) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_head_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
  }

  if (rt_entry_stream_ != nullptr) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_entry_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
  }
  GELOGD("Npu model: %u success to unbind streams.", model_id_);
}

void DavinciModel::DestroyStream() {
  GELOGD("Npu model: %u start to destroy stream.", model_id_);
  // check rt ctx is exist. rt api call will cause error log when ctx does not exist
  rtContext_t current_ctx = nullptr;
  if (rtCtxGetCurrent(&current_ctx) != RT_ERROR_NONE) {
    return;
  }
  for (size_t i = 0U; i < all_hccl_stream_list_.size(); ++i) {
    (void)reusable_stream_allocator_->DestroyStream(all_hccl_stream_list_[i]);
    all_hccl_stream_list_[i] = nullptr;
  }
  if (is_inner_model_stream_) {
    // destroy stream that is bound with rt_model
    (void)reusable_stream_allocator_->DestroyStream(rt_model_stream_, true);
    rt_model_stream_ = nullptr;
  }
  if (is_pure_head_stream_ && (rt_head_stream_ != nullptr)) {
    (void)reusable_stream_allocator_->DestroyStream(rt_head_stream_);
    rt_head_stream_ = nullptr;
  }
  if (rt_entry_stream_ != nullptr) {
    (void)reusable_stream_allocator_->DestroyStream(rt_entry_stream_);
    rt_entry_stream_ = nullptr;
  }
  for (size_t i = 0U; i < stream_list_.size(); ++i) {
    if (stream_list_[i] != nullptr) {
      (void)reusable_stream_allocator_->DestroyStream(stream_list_[i]);
      stream_list_[i] = nullptr;
    }
  }
  if (rt_stream_to_destroy_ != nullptr) {
    GELOGD("Npu model: %u rt_stream_to_destroy_ %p.", model_id_, rt_stream_to_destroy_);
    (void)reusable_stream_allocator_->DestroyStream(rt_stream_to_destroy_);
    rt_stream_to_destroy_ = nullptr;
  }
  GELOGD("Npu model: %u start to destroy stream.", model_id_);
}

Status DavinciModel::UpdateKnownNodeArgs(const std::vector<uint64_t> &inputs,
                                         const std::vector<uint64_t> &outputs) {
  if (known_node_) {
    args_manager_.InitDfxStage1Begin();
  }

  uint32_t ret_up = 0;
  GE_ASSERT_SUCCESS(ConstructActiveMemBaseAddrsForKnownNode(ret_up, inputs, outputs));
  GE_ASSERT_SUCCESS(args_manager_.UpdateForExecute(ret_up, rt_model_stream_));

  if (known_node_) {
    args_manager_.InitDfxStatsticsEnd();
    args_manager_.PrintDfxStatistics(); // 动态shape的静态子图的地址更新耗时统计在此处打印
  }
  return SUCCESS;
}

Status DavinciModel::InitTaskInfo(domi::ModelTaskDef &model_task_def) {
  GELOGI("InitTaskInfo in, task size %d", model_task_def.task().size());
  GE_CHK_STATUS_RET(InitAddrRefreshKernelBin(),
      "[Call][InitAddrRefreshKernelBin] failed, model_id: %u.", model_id_);
  GE_CHK_STATUS_RET(args_manager_.Init(model_task_def, &task_list_),
                    "model args manager init failed, model_id: %u.", model_id_);
  GELOGI("InitTaskInfo out");
  return SUCCESS;
}

void DavinciModel::SaveNodeBasicProfInfo(const OpDescPtr &op_desc, NodeBasicInfoWrapper &node_basic_info,
                                         const TaskProfInfo &prof_api, uint32_t block_dim, uint32_t task_type) {
  const auto name_hash = MsprofGetHashId(node_basic_info.op_name.c_str(), node_basic_info.op_name.length());
  const auto type_hash = MsprofGetHashId(node_basic_info.op_type.c_str(), node_basic_info.op_type.length());
  gert::GlobalProfilingWrapper::BuildNodeBasicInfo(op_desc, block_dim, {name_hash, type_hash},
                                                   task_type, node_basic_info.node_basic_info);
  gert::GlobalProfilingWrapper::BuildCompactInfo(prof_api.end_time, node_basic_info.node_basic_info);
  node_basic_infos_.emplace_back(node_basic_info);
  GELOGD("Add op %s to basic info list.", node_basic_info.op_name.c_str());
}

void DavinciModel::SaveNodeApiProfInfo(const std::string &op_name, const TaskProfInfo &prof_api) {
  ApiInfoWrapper api{};
  api.op_name = op_name;
  gert::GlobalProfilingWrapper::BuildApiInfo(
      {prof_api.begin_time, prof_api.end_time}, MSPROF_REPORT_NODE_LAUNCH_TYPE,
      MsprofGetHashId(op_name.c_str(), op_name.length()), api.api);
  prof_launch_apis_.emplace_back(api);
  GELOGD("Add op %s to api list.", op_name.c_str());
}

void DavinciModel::SaveTaskProfInfo(const std::string &op_name, const OpDescPtr &op_desc,
                                    const TaskProfInfo &prof_api) {
  const auto &prof_mgr = ProfilingManager::Instance();
  TaskDescInfo task_desc_info{};
  task_desc_info.op_name = op_name;
  task_desc_info.prof_time = prof_api.end_time;
  prof_mgr.GetOpInputOutputInfo(op_desc, task_desc_info);
  task_desc_info_.emplace_back(task_desc_info);
  GELOGD("Add op %s to task list.", op_name.c_str());
}

void DavinciModel::SaveProfilingTaskDescInfo(const OpDescPtr &op_desc, const TaskInfo &task_info,
                                             const domi::TaskDef &task_def) {
  const auto &prof_api = task_info.GetProfApi();
  if (prof_api.begin_time == 0UL) {
    GELOGW("Node %s No api info in this task and do not save profiling task info.", op_desc->GetName().c_str());
    return;
  }
  int64_t stream_id = static_cast<int64_t>(task_def.stream_id());
  int64_t logic_stream_id = -1;
  if (stream_id == op_desc->GetStreamId()) {
    // 主流上的task
    if (AttrUtils::GetInt(op_desc, "_logic_stream_id", logic_stream_id) && (logic_stream_id >= 0)) {
      logic_stream_ids_to_physic_stream_ids_[static_cast<uint32_t>(logic_stream_id)].insert(prof_api.stream_id);
    }
  } else {
    // 从流上的task
    std::vector<int64_t> attached_stream_ids;
    if (!AttrUtils::GetInt(op_desc, "_logic_attached_stream_id", logic_stream_id) &&
        AttrUtils::GetListInt(op_desc, "_logic_attached_stream_ids", attached_stream_ids)) {
      const auto &iter = split_logic_stream_2_origin_logic_stream_.find(stream_id);
      if ((iter != split_logic_stream_2_origin_logic_stream_.end()) && (iter->second >= 0)) {
        logic_stream_id = iter->second;
      }
    }
    if (logic_stream_id >= 0) {
      logic_stream_ids_to_physic_stream_ids_[static_cast<uint32_t>(logic_stream_id)].insert(prof_api.stream_id);
    }
  }

  std::string core_type;
  const MsprofGeTaskType task_type = GetProfilingTaskType(op_desc, task_def, core_type);
  if (task_type == MSPROF_GE_TASK_TYPE_INVALID) {
    return;
  }

  // mix op, report context_id to 0
  bool is_fftsplus_task = false;
  if ((AttrUtils::GetBool(op_desc, kAttrIsFFTSTask, is_fftsplus_task) && is_fftsplus_task)) {
    gert::GlobalProfilingWrapper::BuildContextIdInfo(prof_api.end_time, {0}, op_desc->GetName(), context_id_infos_);
  }

  uint32_t block_dim = GetBlockDim(static_cast<ModelTaskType>(task_def.type()), task_def, op_desc);
  GELOGI("Save task info of op name %s, op type %s, task type %u, core type %s, block dim %u, is_fftsplus_task: %d",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), static_cast<uint32_t>(task_type), core_type.c_str(),
         block_dim, static_cast<int32_t>(is_fftsplus_task));

  NodeBasicInfoWrapper node_basic_info{};
  node_basic_info.op_name = op_desc->GetName();
  node_basic_info.op_type = op_desc->GetType();
  SaveNodeBasicProfInfo(op_desc, node_basic_info, prof_api, block_dim, static_cast<uint32_t>(task_type));
  SaveNodeApiProfInfo(node_basic_info.op_name, prof_api);

  if (task_type == MSPROF_GE_TASK_TYPE_HCCL) {
    return;
  }
  SaveTaskProfInfo(op_desc->GetName(), op_desc, prof_api);
}

Status DavinciModel::SaveProfilingInfoByContext(const domi::FftsPlusCtxDef &ctx_def, const OpDescPtr &sgt_node,
                                                const TaskProfInfo &prof_api, bool ffts_flag) {
  const auto it = ctx_type_to_task_types.find(static_cast<rtFftsPlusContextType_t>(ctx_def.context_type()));
  if (it == ctx_type_to_task_types.end()) {
    GELOGW("Task type %u is invalid, do not save profiling task info.", ctx_def.context_type());
    return ge::SUCCESS;
  }

  auto op_desc = GetOpByIndex(ctx_def.op_index());
  GE_ASSERT_NOTNULL(op_desc);

  NodeBasicInfoWrapper ctx_node_basic_info{};
  ctx_node_basic_info.op_name = ctx_def.uniq_ctx_name();
  if (ctx_node_basic_info.op_name.empty()) {
    ctx_node_basic_info.op_name = op_desc->GetName();
  }

  if (ffts_flag && std::find_if(node_basic_infos_.begin(), node_basic_infos_.end(),
                                [&ctx_node_basic_info] (const NodeBasicInfoWrapper &info) -> bool {
                                  return (ctx_node_basic_info.op_name == info.op_name);
                                }) != node_basic_infos_.end()) {
    GELOGD("Skip ffts+ op %s which has already been saved.", ctx_node_basic_info.op_name.c_str());
    return ge::SUCCESS;
  }

  uint32_t block_dim = 0U;
  auto prof_op_desc = sgt_node;
  const auto ctx_type = ctx_def.context_type();
  if ((ctx_type == RT_CTX_TYPE_FLUSH_DATA) || (ctx_type == RT_CTX_TYPE_INVALIDATE_DATA) ||
    (ctx_type == RT_CTX_TYPE_WRITEBACK_DATA)) {
    // 3种CMO类型算子
    ctx_node_basic_info.op_type = kUndefinedOptype;
  } else if (ctx_def.op_type() == domi::FftsPlusCtxDef::ATOMIC) {
    // ATOMIC类算子
    ctx_node_basic_info.op_type = "MemSet";
    const domi::FftsPlusAicAivCtxDef &aic_aiv_ctx_def = ctx_def.aic_aiv_ctx();
    block_dim = aic_aiv_ctx_def.non_tail_block_dim();
  } else {
    ctx_node_basic_info.op_type = op_desc->GetType();
    block_dim = GetBlockDim(ctx_def);
    TaskDescInfo ctx_desc_info{};
    ctx_desc_info.op_name = ctx_node_basic_info.op_name;
    ctx_desc_info.prof_time = prof_api.end_time;
    ProfilingManager::Instance().GetOpInputOutputInfo(op_desc, ctx_desc_info);
    task_desc_info_.emplace_back(ctx_desc_info);
    prof_op_desc = op_desc;
  }

  gert::GlobalProfilingWrapper::BuildContextIdInfo(prof_api.end_time, {ctx_def.context_id()},
                                                   ctx_node_basic_info.op_name, context_id_infos_);
  SaveNodeBasicProfInfo(prof_op_desc, ctx_node_basic_info, prof_api, block_dim, it->second);
  if (!ffts_flag) {
    SaveNodeApiProfInfo(ctx_node_basic_info.op_name, prof_api);
  }
  return ge::SUCCESS;
}

void DavinciModel::SaveProfilingInfoByPartitionCall(const domi::FftsPlusTaskDef &ffts_plus_task_def,
                                                    const OpDescPtr &sgt_node, const TaskProfInfo &prof_api) {
  const auto &op_name = om_name_.empty() ? name_ : om_name_;
  NodeBasicInfoWrapper node_basic_info{};
  node_basic_info.op_type = kTaskTypeFftsPlus;
  node_basic_info.op_name = op_name;
  SaveNodeBasicProfInfo(
      sgt_node, node_basic_info, prof_api, ffts_plus_task_def.ffts_plus_sqe().sqe_header().block_dim(),
      static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_FFTS_PLUS));
  SaveNodeApiProfInfo(op_name, prof_api);
  SaveTaskProfInfo(op_name, sgt_node, prof_api);
  GELOGD("Add partition call node %s to prof list.", op_name.c_str());
}

void DavinciModel::SaveFftsPlusProfilingTask(const domi::TaskDef &task_def, const TaskInfo &task_info) {
  const domi::FftsPlusTaskDef &ffts_plus_task_def = task_def.ffts_plus_task();
  const auto &sgt_node = GetOpByIndex(ffts_plus_task_def.op_index());
  if (sgt_node == nullptr) {
    GELOGW("FftsPlus node not found for index: %u", ffts_plus_task_def.op_index());
    return;
  }

  const auto &prof_api = task_info.GetProfApi();
  if (prof_api.begin_time == 0UL) {
    GELOGW("No api info in this task and do not save profiling task info.");
    return;
  }

  const bool ffts_flag = sgt_node->HasAttr(ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH);
  if (ffts_flag) {
    SaveProfilingInfoByPartitionCall(ffts_plus_task_def, sgt_node, prof_api);
  }
  for (int32_t i = 0; i < ffts_plus_task_def.ffts_plus_ctx_size(); ++i) {
    const auto &ctx_def = ffts_plus_task_def.ffts_plus_ctx(i);
    if (SaveProfilingInfoByContext(ctx_def, sgt_node, prof_api, ffts_flag) != ge::SUCCESS) {
      return;
    }
  }
}

MsprofGeTaskType DavinciModel::GetTaskType(const domi::FftsPlusCtxDef &ctx_def) const {
  const auto it = ctx_type_to_task_types.find(static_cast<rtFftsPlusContextType_t>(ctx_def.context_type()));
  if (it == ctx_type_to_task_types.end()) {
    return MSPROF_GE_TASK_TYPE_INVALID;
  }
  return it->second;
}

uint32_t DavinciModel::GetBlockDim(const ModelTaskType type, const domi::TaskDef &task_def,
  const OpDescPtr &op_desc) const {
  uint32_t block_dim = 0;
  bool is_tiling_depend = false;
  ccKernelType kernel_type = ge::ccKernelType::INVALID;

  if ((type == ModelTaskType::MODEL_TASK_KERNEL) || (type == ModelTaskType::MODEL_TASK_VECTOR_KERNEL) ||
      (type == ModelTaskType::MODEL_TASK_SUPER_KERNEL)) {
    const auto &context = task_def.kernel().context();
    kernel_type = static_cast<ccKernelType>(context.kernel_type());
    block_dim = task_def.kernel().block_dim();
  }
  if ((type == ModelTaskType::MODEL_TASK_ALL_KERNEL) || (type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL)) {
    const auto &context = task_def.kernel_with_handle().context();
    kernel_type = static_cast<ccKernelType>(context.kernel_type());
    block_dim = task_def.kernel_with_handle().block_dim();
  }

  if (type == ModelTaskType::MODEL_TASK_HCCL) {
    block_dim = task_def.kernel_hccl().aiv_block_dim();
  }

  // mix op
  uint32_t task_ratio = 0;
  bool is_fftsplus_task = false;
  if ((AttrUtils::GetBool(op_desc, kAttrIsFFTSTask, is_fftsplus_task) && is_fftsplus_task &&
       AttrUtils::GetInt(op_desc, kAttrTaskRatio, task_ratio))) {
    GELOGI("Op %s is fftsplus task, kernel type: %d, block dim: %u, task ratio: %u", op_desc->GetName().c_str(),
           static_cast<uint32_t>(kernel_type), block_dim, task_ratio);
    // 针对mix算子，低16位为主加速器blockdim，高16位为从加速器的ratio值，由工具解析
    block_dim = ((block_dim & 0xFFFFU) | (task_ratio << 16U));
  }

  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, is_tiling_depend);
  if (is_tiling_depend && ModelUtils::IsAICoreKernel(kernel_type)) {
    GELOGD("Op %s is tiling dependent, set block dim to maximum value.", op_desc->GetName().c_str());
    return kTilingSinkBlockDim;
  }

  return block_dim;
}

uint32_t DavinciModel::GetBlockDim(const domi::FftsPlusCtxDef &ctx_def) const {
  std::string op_name = "";
  const auto &ctx_op_desc = GetOpByIndex(ctx_def.op_index());
  if (ctx_op_desc != nullptr) {
    op_name = ctx_op_desc->GetName();
  }
  const MsprofGeTaskType task_type = GetTaskType(ctx_def);
  uint32_t block_dim;
  if ((task_type == MSPROF_GE_TASK_TYPE_MIX_AIC) || (task_type == MSPROF_GE_TASK_TYPE_MIX_AIV)) {
    bool is_tiling_depend = false;
    (void)ge::AttrUtils::GetBool(ctx_op_desc, ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, is_tiling_depend);
    if (is_tiling_depend) {
      block_dim = kTilingSinkBlockDim;
    } else {
      const auto &mixaicaiv_ctx_def = ctx_def.mix_aic_aiv_ctx();
      const uint32_t mix_non_tail_block_dim = mixaicaiv_ctx_def.non_tail_block_dim();
      const uint32_t mix_tail_block_ratio = mixaicaiv_ctx_def.non_tail_block_ratio_n();
      // 针对mix算子，低16位为主加速器blockdim，高16位为从加速器的ratio值，由工具解析
      block_dim = ((mix_non_tail_block_dim & 0xFFFFU) | (mix_tail_block_ratio << 16U));
    }
    GELOGI("FftsPlus mixaicaiv profiling blockdim: %u, op_name: %s", block_dim, op_name.c_str());
  } else if ((task_type == MSPROF_GE_TASK_TYPE_AI_CORE) || (task_type == MSPROF_GE_TASK_TYPE_AIV)) {
    const auto &aicaiv_ctx_def = ctx_def.aic_aiv_ctx();
    block_dim = aicaiv_ctx_def.non_tail_block_dim();
    GELOGI("FftsPlus aicaiv profiling blockdim: %u, op_name: %s", block_dim, op_name.c_str());
  } else if (task_type == MSPROF_GE_TASK_TYPE_AI_CPU) {
    const auto &aicpu_ctx_def = ctx_def.aicpu_ctx();
    block_dim = aicpu_ctx_def.non_tail_block_dim();
    GELOGI("FftsPlus aicpu profiling blockdim: %u, op_name: %s", block_dim, op_name.c_str());
  } else {
    block_dim = 0U;
  }
  return block_dim;
}

uint32_t DavinciModel::GetThreadId(const domi::FftsPlusCtxDef &ctx_def) const {
  const MsprofGeTaskType task_type = GetTaskType(ctx_def);
  uint32_t tid;
  if ((task_type == MSPROF_GE_TASK_TYPE_MIX_AIC) || (task_type == MSPROF_GE_TASK_TYPE_MIX_AIV)) {
    const auto &mixaicaiv_ctx_def = ctx_def.mix_aic_aiv_ctx();
    tid = mixaicaiv_ctx_def.thread_id();
  } else if ((task_type == MSPROF_GE_TASK_TYPE_AI_CORE) || (task_type == MSPROF_GE_TASK_TYPE_AIV)) {
    const auto &aicaiv_ctx_def = ctx_def.aic_aiv_ctx();
    tid = aicaiv_ctx_def.thread_id();
  } else if (task_type == MSPROF_GE_TASK_TYPE_AI_CPU) {
    const auto &aicpu_ctx_def = ctx_def.aicpu_ctx();
    tid = aicpu_ctx_def.thread_id();
  } else {
    tid = 0U;
  }
  return tid;
}

Status DavinciModel::SetStreamLockOrUnlocK(rtStream_t stm, const bool is_lock) const {
  if (stream_list_.size() != 1UL) { // only single physical stream can set lock or unlock currently
    return SUCCESS;
  }
  GELOGI("try to set lock flag %d to stream %p", static_cast<int32_t>(is_lock), stm);
  if (is_lock) {
    GE_CHK_RT_RET(rtSetStreamSqLock(stm));
  } else {
    GE_CHK_RT_RET(rtSetStreamSqUnlock(stm));
  }
  return SUCCESS;
}

Status DavinciModel::DistributeTask(const domi::ModelTaskDef &model_task_def) {
  GELOGI("DistributeTask in: model task: %d, cpu task: %zu", model_task_def.task().size(), cpu_task_list_.size());
  if (!stream_list_.empty()) {
    GE_ASSERT_SUCCESS(SetStreamLockOrUnlocK(stream_list_[0], true));
  }
  for (auto &task : cpu_task_list_) {
    GE_CHECK_NOTNULL(task);
    GE_CHK_STATUS_RET(task->Distribute());
  }

  task_desc_info_.clear();
  const auto &prof_mgr = ProfilingManager::Instance();
  for (int32_t task_index = 0; task_index < model_task_def.task_size(); ++task_index) {
    const auto &task_def = model_task_def.task(task_index);
    const auto &task_info = task_list_.at(static_cast<size_t>(task_index));
    GE_CHECK_NOTNULL(task_info);
    uint64_t start_time = 0UL;
    uint64_t end_time = 0UL;
    GetCurTimestamp(start_time);
    GE_CHK_STATUS_RET(task_info->Distribute(), "[Call][Distribute] for Task[%d] fail", task_index);
    GetCurTimestamp(end_time);

    if (hccl_task_stream_set_.count(task_info->GetTaskStream()) > 0) {
      stream_to_task_index_list_[task_info->GetTaskStream()].emplace_back(static_cast<size_t>(task_index));
      GELOGI("logic task index: %d is On stream: %" PRIu64 " with hccl tasks", task_index, task_info->GetTaskStream());
    }

    UpdateTaskTypeStat(static_cast<uint32_t>(task_def.type()), start_time, end_time);

    if (prof_mgr.ProfilingModelLoadOn()) {
      for (auto fusion_op_info : task_info->GetAllFusionOpInfo()) {
        if (!fusion_op_info.original_op_names.empty()) {
          fusion_op_info.task_id = task_info->GetTaskID();
          GELOGD("task id is %u, op num is %zu", fusion_op_info.task_id, fusion_op_info.original_op_names.size());
          GE_CHK_STATUS_RET(InitFusionProfiling(fusion_op_info), "[Init][Profiling] failed, model_id: %u.", model_id_);
        }
      }
    }

    // 在task launch后，通过额外信息
    GE_ASSERT_SUCCESS(args_manager_.OnTaskDistributed(static_cast<size_t>(task_index), task_info.get()));

    // 保存stream id task id 与opdesc的 信息
    const auto op_index = task_info->ParseOpIndex(task_def);

    SaveDfxInfo(op_index, *task_info);
    ModelManager::GetInstance().SetCallBackFuncForDumpManager();
    // for profiling and data dump
    task_info->PostProcess(task_def);
  }

  hccl_task_stream_set_.clear();

  if (model_task_def.task_size() != 0) {
    args_manager_.GenModelArgsAaddrAfterDistributed();
  }

  // launch dump kernel to aicpu
  GE_CHK_STATUS_RET(data_dumper_.LoadDumpInfo(), "[Load][DumpInfo] failed, model_id: %u.", model_id_);
  GELOGI("DistributeTask out");
  return SUCCESS;
}

void DavinciModel::SaveDfxInfo(const int64_t op_idx, const TaskInfo &task_info) {
  const auto stream_id = task_info.GetStreamId();
  const auto task_id = task_info.GetTaskID();

  if (!known_node_) {
    GELOGD("Save dfx info, task_id=%u, stream_id=%u, op_idx=%lld.", task_id, stream_id, op_idx);
    if ((stream_id == 0xFFFFFFFFU) || (task_id == 0xFFFFFFFFU) || (op_idx == -1)) {
      return;
    }
    const auto &op_desc = GetOpByIndex(static_cast<uint32_t>(op_idx));

    ErrorTracking::GetInstance().SaveGraphTaskOpdescInfo(op_desc, task_id, stream_id, model_id_);
  }
  return;
}

void DavinciModel::SaveFftsDfxInfo(const domi::FftsPlusCtxDef &ctx_def, const OpDescPtr &op_desc,
                                       const TaskInfo &task_info) {
  uint32_t context_id;
  if (!known_node_) {
    (void)AttrUtils::GetInt(op_desc, "current_context_id", context_id);
    const TaskKey key(task_info.GetTaskID(), task_info.GetStreamId(), context_id, GetThreadId(ctx_def));
    ErrorTracking::GetInstance().SaveGraphTaskOpdescInfo(op_desc, key, model_id_);
  }
}

void DavinciModel::InitExceptionDumpInfo(const OpDescPtr &op_desc, uintptr_t args, size_t arg_size,
                                         const std::map<uint64_t, uint64_t> &cust_to_relevant_offset,
                                         ExtraOpInfo &extra_dump_info) const {
  extra_dump_info.args = args;
  extra_dump_info.args_size = arg_size;
  extra_dump_info.input_addrs = ModelUtils::GetInputAddrs(runtime_param_, op_desc);
  extra_dump_info.output_addrs = ModelUtils::GetOutputAddrs(runtime_param_, op_desc);
  const auto workspaces = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param_, op_desc);
  const auto workspace_bytes = ModelUtils::GetWorkspaceSize(op_desc);
  if (workspaces.size() == workspace_bytes.size()) {
    for (size_t i = 0U; i < workspaces.size(); ++i) {
      extra_dump_info.workspace_info.emplace_back(static_cast<uintptr_t>(workspaces[i]), workspace_bytes[i]);
    }
  }
  extra_dump_info.cust_to_relevant_offset_ = cust_to_relevant_offset;
  if (gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) {
    extra_dump_info.RecordArgsBefore();
  }
}

void DavinciModel::SaveFftsExceptionDumpInfo(const domi::FftsPlusCtxDef &ctx_def, const OpDescPtr &op_desc,
                                             const TaskInfo &task_info, const std::pair<uintptr_t, size_t> &args,
                                             const std::map<uint64_t, uint64_t> &cust_to_relevant_offset) {
  if (args.first == 0UL) {
    return;
  }

  ExtraOpInfo extra_dump_info;
  InitExceptionDumpInfo(op_desc, args.first, args.second, cust_to_relevant_offset, extra_dump_info);

  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  tiling_info = op_desc->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  if (tiling_info != nullptr) {
    extra_dump_info.tiling_key = static_cast<uint32_t>(tiling_info->GetTilingKey());
    extra_dump_info.tiling_data = tiling_info->GetAllTilingData().str();
  }

  const OpDescInfoId id(task_info.GetTaskID(), task_info.GetStreamId(), UINT32_MAX, GetThreadId(ctx_def),
                        GetDeviceId());
  exception_dumper_.SaveDumpOpInfo(op_desc, extra_dump_info, id, false);
}

bool DavinciModel::OpNeedDump(const OpDescPtr &op_desc) {
  if (OpNeedDump(op_desc->GetName())) {
    return true;
  }
  std::vector<std::string> original_names;
  if (ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names) &&
    !original_names.empty()) {
    for (const auto &name : original_names) {
      if (OpNeedDump(name)) {
        return true;
      }
    }
  }
  return false;
}

bool DavinciModel::OpNeedPrint(const OpDescPtr &op_desc) const {
  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxPrintf = "printf";
  std::vector<std::string> dfx_opts;
  if (!ge::AttrUtils::GetListStr(op_desc, kOpDfxOptions, dfx_opts) ||
      (std::find(dfx_opts.begin(), dfx_opts.end(), kOpDfxPrintf) == dfx_opts.end())) {
    GELOGD("op[%s] does not have print dfx option", op_desc->GetName().c_str());
    return false;
  }
  return true;
}

void DavinciModel::SaveExceptionDumpInfo(const OpDescPtr &op_desc, const TaskInfo &task_info) {
  ExtraOpInfo extra_dump_info{};
  InitExceptionDumpInfo(op_desc, task_info.GetArgs(), task_info.GetArgSize(), task_info.GetCustToRelevantOffset(),
                        extra_dump_info);
  task_info.GetTilingKeyAndData(extra_dump_info.tiling_key, extra_dump_info.tiling_data);
  ge::OpDescInfoId id(task_info.GetTaskID(), task_info.GetStreamId(), GetDeviceId());
  exception_dumper_.SaveDumpOpInfo(op_desc, extra_dump_info, id, false);
}

void DavinciModel::SaveDfxInfo(const uint32_t op_idx, const domi::TaskDef &task_def, const TaskInfo &task_info) {
  const OpDescPtr &op_desc = GetOpByIndex(op_idx);
  if (op_desc == nullptr) {
    GELOGW("Node not found for profiling, index: %u", op_idx);
    return;
  }
  GELOGD("Start to SaveDfxinfo for op[%s]", op_desc->GetName().c_str());

  if (task_info.GetDumpArgs() != 0U) {
    const bool call_dump = (OpNeedDump(op_desc) || OpNeedDumpOnWatcherModel(op_desc->GetName())) &&
      task_info.CallSaveDumpInfo();
    if (call_dump || is_op_debug_reg_) {
      const auto task_type = static_cast<ModelTaskType>(task_def.type());
      GELOGI("Start to SaveDumpTask for op[%s], task_type[%u]", op_desc->GetName().c_str(),
             static_cast<uint32_t>(task_type));
      SaveDumpTask({task_info.GetTaskID(), task_info.GetStreamId(), 0U, 0U}, op_desc, task_info.GetDumpArgs(),
                   {}, task_info.GetCustToRelevantOffset(), task_type);
    }
  }

  if (OpNoNeedDumpOnWatcherModel(op_desc->GetName())) {
    LayerOpOnWatcherModeInfo op_info = {task_info.GetTaskID(), task_info.GetStreamId(), op_desc};
    SaveLayerOpInfoOnWatcherMode(op_info);
  }

  if (GetAiCpuCustFlag()) {
    SaveWorkInfo(op_desc);
  }

  if (OpNeedPrint(op_desc)) {
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    GELOGI("Start to SavePrintDumpTask for op[%s], task_type[%u]", op_desc->GetName().c_str(),
           static_cast<uint32_t>(task_type));
    SavePrintDumpTask({task_info.GetTaskID(), task_info.GetStreamId(), 0U, 0U}, op_desc, task_info.GetDumpArgs(), {},
                      task_type);
    SavePrintWorkInfo(op_desc);
  }

  // save task info for exception dump
  if ((gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) || !need_clear_dfx_cache_) {
    GELOGI("save exception info");
    SaveExceptionDumpInfo(op_desc, task_info);
  }

  // save task info for profiling
  if ((gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) || !need_clear_dfx_cache_) {
    GELOGI("save profiling info");
    SaveProfilingTaskDescInfo(op_desc, task_info, task_def);
  }
}

bool DavinciModel::ModelNeedDump() const {
  const auto all_dump_model = GetDumpProperties().GetAllDumpModel();
  return (all_dump_model.find(DUMP_ALL_MODEL) != all_dump_model.end()) ||
         (all_dump_model.find(dump_model_name_) != all_dump_model.end()) ||
         (all_dump_model.find(om_name_) != all_dump_model.end());
}

void DavinciModel::SetEndGraphId(const uint32_t task_id, const uint32_t stream_id) {
  if (ModelNeedDump()) {
    GELOGI("start save end_graph_info to dumper, task_id is %u, stream_id is %u", task_id, stream_id);
    data_dumper_.SaveEndGraphId(task_id, stream_id);
  }
}

///
/// @ingroup ge
/// @brief Set copy only for No task feed NetOutput address.
/// @return None.
///
Status DavinciModel::SetCopyOnlyOutput() {
  for (const auto &output_outside_addrs : output_data_info_) {
    const ZeroCopyOffset &output_outside = output_outside_addrs.second;
    if (!output_outside.IsRelativeOffsetValid()) {
      return SUCCESS;
    }
    for (size_t out_count = 0U; out_count < output_outside.GetAddrCount(); ++out_count) {
      const auto &addrs_mapping_list = output_outside.GetOutsideAddrs();
      const std::map<uintptr_t, std::vector<uintptr_t>> &virtual_args_addrs = addrs_mapping_list[out_count];
      for (const auto &virtual_args_addr : virtual_args_addrs) {
        const auto &args_addrs = virtual_args_addr.second;
        if (args_addrs.empty()) {
          // No task feed Output addr, Need copy directly.
          GELOGI("[ZCPY][disable zero copy] addrs 0x%" PRIx64 ", no task feed this output, model_id:%u.",
              virtual_args_addr.first, model_id_);
          if (!copy_only_addrs_.IsRefDataAddr(static_cast<uint64_t>(virtual_args_addr.first))) {
            GE_ASSERT_SUCCESS(copy_only_addrs_.Insert(static_cast<uint64_t>(virtual_args_addr.first)));
          }
        }
      }
    }
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set disabled input zero copy addr.
/// @param [in] const void *addr: address of task
/// @return None.
///
Status DavinciModel::DisableZeroCopy(const void *const addr, const bool fusion_flag) {
  if ((real_virtual_addrs_.find(PtrToValue(addr)) == real_virtual_addrs_.end()) || fusion_flag) {
    return SUCCESS;
  }

  // Data link to RTS Op directly.
  const std::lock_guard<std::mutex> lk(outside_addrs_mutex_);
  GELOGI("[ZCPY][disable zero copy] addrs %p, model_id:%u.", addr, model_id_);
  GE_ASSERT_SUCCESS(copy_only_addrs_.Insert(PtrToValue(addr)));
  return SUCCESS;
}

Status DavinciModel::DisableZeroCopyInReuseMemoryMode(const NodePtr &node, const size_t idx, const void *const addr) {
  if (!IsInputOfNetoutputCanZeroCopy(node, static_cast<int32_t>(idx))) {
    for (const auto &item : input_data_info_) {
      if (item.second.GetBasicAddr() == addr) {
        GELOGI("Addr %p reference or reused from input and was determined by data node", addr);
        return SUCCESS;
      }
    }
    GELOGI("[ZCPY][disable zero copy] addrs %p of %s input %zu, model_id:%u.", addr, node->GetName().c_str(),
        idx, model_id_);
    const std::lock_guard<std::mutex> lk(outside_addrs_mutex_);
    GE_ASSERT_SUCCESS(copy_only_addrs_.Insert(PtrToValue(addr)));
  }
  return SUCCESS;
}

Status DavinciModel::DisableZeroCopyNode(const OpDescPtr &op_desc) {
  const auto disable_addrs = [this](const std::vector<uint64_t> &all_data_addrs) -> Status {
    for (const uint64_t addr : all_data_addrs) {
      GE_ASSERT_SUCCESS(DisableZeroCopy(ValueToPtr(addr)));
    }
    return SUCCESS;
  };

  const auto &node_type = op_desc->GetType();
  if (node_type == VARIABLE) {
    disable_addrs(ModelUtils::GetOutputAddrsValue(runtime_param_, op_desc));
  }
  return SUCCESS;
}

void DavinciModel::DelDependentHcclStreams(const ComputeGraphPtr &compute_graph) {
  // remove stream_id from hcom_streams which hccl_stream has op type except hcom
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    const uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
    if (hcom_streams_.count(stream_id) > 0U) {
      if (!(HcomOmeUtil::IsHCOMOp(op_desc->GetType()) || HcomOmeUtil::IsHorovodOp(op_desc->GetType()) ||
            IsSendRecvOp(op_desc->GetType()))) {
        (void)hcom_streams_.erase(stream_id);
        GELOGI("delete stream_id:%u from hcom_streams, node:%s, op_type:%s.",
               stream_id,
               node->GetName().c_str(),
               op_desc->GetType().c_str());
      }
    }
  }
}

///
/// @ingroup ge
/// @brief Save outside address used info for ZeroCopy.
/// @param [in] const OpDescPtr &op_desc: current op desc
/// @param [in] const std::vector<void *> &outside_addrs: address of task
/// @param [in] const void *info: task args
/// @param [in] const char *args: task args
/// @param [in] size_t size: size of task args
/// @param [in] size_t offset: offset of task args
/// @return None.
///
void DavinciModel::SetZeroCopyAddr(const OpDescPtr &op_desc, const std::vector<uint64_t> &outside_addrs,
                                   const void *const args_info, const uintptr_t args_base, const size_t args_size,
                                   const size_t offset, const std::vector<bool> &io_tiling_list) {
  (void)args_info;
  (void)args_size;
  // Internal call has ensured that op_desc is not nullptr
  GELOGD("[ZCPY] SetZeroCopyAddr for node %s.", op_desc->GetName().c_str());
  const size_t nums = outside_addrs.size();

  for (size_t i = 0U; i < nums; ++i) {
    const std::lock_guard<std::mutex> lk(outside_addrs_mutex_);
    const bool is_tiling = (i >= io_tiling_list.size() ? false : io_tiling_list[i]);
    for (auto &input_outside_addrs : input_data_info_) {
      ZeroCopyOffset &input_outside = input_outside_addrs.second;
      input_outside.SetOutsideAddrsValue(static_cast<uintptr_t>(outside_addrs[i]), is_tiling,
                                         args_base, offset + (i * kAddrSize));
    }

    for (auto &output_outside_addrs : output_data_info_) {
      ZeroCopyOffset &output_outside = output_outside_addrs.second;
      output_outside.SetOutsideAddrsValue(static_cast<uintptr_t>(outside_addrs[i]), is_tiling,
                                          args_base, offset + (i * kAddrSize));
    }
  }
}

///
/// @ingroup ge
/// @brief Copy Check input size and model op size.
/// @param [in] const int64_t &input_size: input size.
/// @param [in] const int64_t &op_size: model op size.
/// @param [in] is_dynamic: dynamic batch input flag.
/// @return true if success
///
bool DavinciModel::CheckUserAndModelSize(const int64_t size, const int64_t op_size, const char *model_io_type) const {
  if (is_dynamic_) {  // dynamic is max size.
    GELOGI("No need to check user input and model size.", model_io_type);
    return true;
  }

  if (size > op_size) {
    if (logLevel_ <= DLOG_WARN) {
      GELOGW(
        "User %s size(bytes) [%" PRId64 "] is bigger than om size [%" PRId64 "], "
        "may cause inference problem, please check model input",
        model_io_type, size, op_size);
    }
  }

  if (is_dynamic_aipp_) {
    GELOGI("This is dynamic aipp model, no need to judge smaller user size");
    return true;
  }
  // Judge overflow first
  if (size > (kOverflowUserSize)) {
    GELOGI("The user %s size [%" PRId64 "] is smaller than model size [%" PRId64 "] and is in the range of 64 bytes",
            model_io_type, size, op_size);
    return true;
  }
  // The input and model input size can not be exactly equal because user input is not definite.
  if ((size + kDataMemAlignSizeCompare) < op_size) {
    std::string model_id_type_str = model_io_type;
    const std::string reason = "The input memory size set by the user is invalid.The provided " + std::to_string(size) +
                                " bytes of buffer size plus the aligned " + std::to_string(kDataMemAlignSizeCompare) +
                                " bytes is less than the tensor size " + std::to_string(op_size) + " bytes required by the model";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({reason.c_str()}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] %s size:%" PRId64 " "
           "from user add align:%" PRId64 " < op_size:%" PRId64 " in model:%u",
           model_io_type, size, kDataMemAlignSizeCompare, op_size, model_id_);
    return false;
  }
  return true;
}

void DavinciModel::CreateMultiBatchDataBuffer(std::vector<DataBuffer> &blobs) {
  DataBuffer data;
  data.length = static_cast<uint32_t>(cur_dynamic_dims_.size() * sizeof(int32_t));
  data.data = cur_dynamic_dims_.data();
  blobs.push_back(data);
}

void DavinciModel::GetGeTensorBlobs(InputData &input_data,
  const std::vector<GeTensor> &input_tensor) const {
  if (input_data.blobs.size() == 0) {
      input_data.blobs.resize(input_tensor.size());
      for (size_t i = 0U; i < input_tensor.size(); i++) {
        input_data.blobs[i].data = ValueToPtr(PtrToValue(input_tensor[i].GetData().data()));
        input_data.blobs[i].length = input_tensor[i].GetData().size();
        input_data.blobs[i].isDataSupportMemShare = false;
        // In case the user does not set the placement.
        input_data.blobs[i].placement = static_cast<uint32_t>(Placement::kPlacementDevice);
      }
    }
}

std::vector<int64_t> DavinciModel::GetTensorDims(const gert::Shape &shape) const {
  std::vector<int64_t> dims;
  size_t dimNum = shape.GetDimNum();
  dims.resize(dimNum);

  for (size_t i = 0U; i < dimNum; i++) {
    dims[i] = shape[i];
  }

  return dims;
}

void DavinciModel::GetGeTensorBlobs(InputData &input_data,
  const std::vector<gert::Tensor> &input_tensor) const {
  if (input_data.blobs.size() == 0) {
      input_data.blobs.resize(input_tensor.size());
      for (size_t i = 0U; i < input_tensor.size(); i++) {
        input_data.blobs[i].data = ValueToPtr(PtrToValue(input_tensor[i].GetAddr()));
        input_data.blobs[i].length = input_tensor[i].GetSize();
        input_data.blobs[i].isDataSupportMemShare = false;
        // In case the user does not set the placement.
        input_data.blobs[i].placement =
          static_cast<uint32_t>(input_tensor[i].GetPlacement() == gert::kOnHost ?
          Placement::kPlacementHost : Placement::kPlacementDevice);
      }
    }
}

Status DavinciModel::CopyModelData(const std::vector<gert::Tensor> &input_tensor,
                                   const std::vector<gert::Tensor> &output_tensor) {
  const bool dynamic_shape_data = is_online_infer_dynamic_ && (!is_getnext_sink_dynamic_);
  InputData input_data;
  OutputData output_data;
  if (dynamic_shape_data) {
    cur_dynamic_dims_.clear();
    for (size_t i = 0U; i < input_tensor.size(); i++) {
      input_data.shapes.emplace_back(GetTensorDims(input_tensor[i].GetStorageShape()));
    }
    if (GetCurDynamicDims(input_data.shapes, cur_dynamic_dims_) != SUCCESS) {
      return INTERNAL_ERROR;
    }

    GetGeTensorBlobs(input_data, input_tensor);

    // 整图分档会多生成一个data用来命中挡位，此处需要为他构造数据
    CreateMultiBatchDataBuffer(input_data.blobs);
  }

  args_manager_.InitDfxStage1Begin();
  GE_ASSERT_SUCCESS(UpdateAllNodeArgs(input_data, output_data, input_tensor, output_tensor));
  args_manager_.InitDfxStatsticsEnd();

  GE_ASSERT_SUCCESS(CopyInputForNoZeroCopy(input_data.blobs, input_indexes_to_copy_info_, input_tensor));
  if (host_pls_input_indexes_to_copy_info_.size() != 0U) {
    GE_ASSERT_SUCCESS(CopyInputForNoZeroCopy(input_data.blobs, host_pls_input_indexes_to_copy_info_,
                                             input_tensor));
    host_pls_input_indexes_to_copy_info_.clear();
  }

  return SUCCESS;
}

Status DavinciModel::CopyModelData(InputData &input_data, OutputData &output_data,
    const std::vector<GeTensor> &input_tensor, const std::vector<GeTensor> &output_tensor) {
  const bool dynamic_shape_data = is_online_infer_dynamic_ && (!is_getnext_sink_dynamic_);
  if (dynamic_shape_data) {
    cur_dynamic_dims_.clear();
    for (size_t i = 0U; i < input_tensor.size(); i++) {
      input_data.shapes.emplace_back(input_tensor[i].GetTensorDesc().GetShape().GetDims());
    }
    if (GetCurDynamicDims(input_data.shapes, cur_dynamic_dims_) != SUCCESS) {
      return INTERNAL_ERROR;
    }

    GetGeTensorBlobs(input_data, input_tensor);

    // 整图分档会多生成一个data用来命中挡位，此处需要为他构造数据
    CreateMultiBatchDataBuffer(input_data.blobs);
  }

  args_manager_.InitDfxStage1Begin();
  GE_ASSERT_SUCCESS(UpdateAllNodeArgs(input_data, output_data, input_tensor, output_tensor));
  args_manager_.InitDfxStatsticsEnd();

  GE_ASSERT_SUCCESS(CopyInputForNoZeroCopy(input_data.blobs, input_indexes_to_copy_info_, input_tensor));
  if (host_pls_input_indexes_to_copy_info_.size() != 0U) {
    GE_ASSERT_SUCCESS(CopyInputForNoZeroCopy(input_data.blobs, host_pls_input_indexes_to_copy_info_,
                                             input_tensor));
    host_pls_input_indexes_to_copy_info_.clear();
  }

  if (dynamic_shape_data) {
    input_data.blobs.pop_back();
  }
  return SUCCESS;
}

Status DavinciModel::CopyInputForNoZeroCopy(const std::vector<DataBuffer> &blobs,
                                            const std::map<uint32_t, MemAllocationSlice> &copy_infos,
                                            const std::vector<gert::Tensor> &tensors) {
  uint32_t blobs_size = blobs.size();
  bool isBlobsEmpty = false;
  if (blobs_size == 0U) {
    isBlobsEmpty = true;
  }
  for (auto &item : copy_infos) {
    const size_t input_idx = static_cast<size_t>(item.first);
    const size_t id = static_cast<size_t>(item.second.id);
    const uint64_t offset = item.second.offset;
    const uint64_t data_size = item.second.data_size;

    if (!isBlobsEmpty) {
      GE_ASSERT_TRUE((input_idx < blobs_size), "invalid user input index:%zu, while model blobs size:%zu",
                    input_idx, blobs_size);
    } else {
      GE_ASSERT_TRUE((input_idx < tensors.size()), "invalid user input index:%zu, while model tensors size:%zu",
                    input_idx, tensors.size());
    }

    uint64_t buffer_length = !isBlobsEmpty ? blobs.at(input_idx).length :
        tensors.at(input_idx).GetSize();
    uint32_t buffer_placement = !isBlobsEmpty ? blobs.at(input_idx).placement :
        static_cast<uint32_t>(tensors[input_idx].GetPlacement() == gert::kOnHost ?
        Placement::kPlacementHost : Placement::kPlacementDevice);
    void *data = !isBlobsEmpty ? blobs.at(input_idx).data :
        ValueToPtr(PtrToValue(tensors[input_idx].GetAddr()));

    if ((buffer_length == 0U) || (data_size == 0U)) {
      GELOGI("Length of data is zero, No need copy. intput tensor index=%zu", input_idx);
      continue;
    }
    GE_ASSERT_TRUE(CheckUserAndModelSize(static_cast<int64_t>(buffer_length),
                                         static_cast<int64_t>(data_size), K_INPUT));

    GE_ASSERT_TRUE((id < logical_mem_allocations_.size()),
                   "invalid user input id:%zu, active mem base size:%zu", id,
                   logical_mem_allocations_.size());
    void *const des_addr = ValueToPtr(allocation_ids_to_active_base_addr_[id] + offset);
    const auto kind = (buffer_placement == kPlaceHostData) ? RT_MEMCPY_HOST_TO_DEVICE_EX : RT_MEMCPY_DEVICE_TO_DEVICE;
    const auto src_len = buffer_length > data_size ? data_size : buffer_length;
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[ForceCopy] model_id:%u, input_index:%zu, id:%zu, offset:0x%" PRIx64 ", base:0x%" PRIx64 ", dst_addr:%p, "
           "dst_size:%" PRIu64 ", src_addr:%p, src_size:%" PRIu64 ", async_mode:%d, kind:%d", model_id_, input_idx, id,
           offset, allocation_ids_to_active_base_addr_[id], des_addr, data_size, data, buffer_length,
           static_cast<int32_t>(is_async_mode_), static_cast<int32_t>(kind));
    }
    if (is_async_mode_) {
      GE_CHK_RT_RET(rtMemcpyAsync(des_addr, data_size, data, src_len, kind, rt_model_stream_));
    } else {
      GE_CHK_RT_RET(rtMemcpy(des_addr, data_size, data, src_len, kind));
    }
  }

  return SUCCESS;
}

Status DavinciModel::CopyInputForNoZeroCopy(const std::vector<DataBuffer> &blobs,
                                            const std::map<uint32_t, MemAllocationSlice> &copy_infos,
                                            const std::vector<GeTensor> &tensors) {
  uint32_t blobs_size = blobs.size();
  bool isBlobsEmpty = false;
  if (blobs_size == 0U) {
    isBlobsEmpty = true;
  }
  for (auto &item : copy_infos) {
    const size_t input_idx = static_cast<size_t>(item.first);
    const size_t id = static_cast<size_t>(item.second.id);
    const uint64_t offset = item.second.offset;
    const uint64_t data_size = item.second.data_size;

    if (!isBlobsEmpty) {
      GE_ASSERT_TRUE((input_idx < blobs_size), "invalid user input index:%zu, while model blobs size:%zu",
                    input_idx, blobs_size);
    } else {
      GE_ASSERT_TRUE((input_idx < tensors.size()), "invalid user input index:%zu, while model tensors size:%zu",
                    input_idx, tensors.size());
    }

    uint64_t buffer_length = !isBlobsEmpty ? blobs.at(input_idx).length :
        tensors.at(input_idx).GetData().size();
    uint32_t buffer_placement = !isBlobsEmpty ? blobs.at(input_idx).placement :
        (!copy_host_input_indexes_.empty() ? static_cast<uint32_t>(tensors[input_idx].GetTensorDesc().GetPlacement()):
        static_cast<uint32_t>(Placement::kPlacementDevice));
    void *data = !isBlobsEmpty ? blobs.at(input_idx).data :
        ValueToPtr(PtrToValue(tensors[input_idx].GetData().data()));

    if ((buffer_length == 0U) || (data_size == 0U)) {
      GELOGI("Length of data is zero, No need copy. intput tensor index=%zu", input_idx);
      continue;
    }
    GE_ASSERT_TRUE(CheckUserAndModelSize(static_cast<int64_t>(buffer_length),
                                         static_cast<int64_t>(data_size), K_INPUT));

    GE_ASSERT_TRUE((id < logical_mem_allocations_.size()),
                   "invalid user input id:%zu, active mem base size:%zu", id,
                   logical_mem_allocations_.size());
    void *const des_addr = ValueToPtr(allocation_ids_to_active_base_addr_[id] + offset);
    const auto kind = (buffer_placement == kPlaceHostData) ? RT_MEMCPY_HOST_TO_DEVICE_EX : RT_MEMCPY_DEVICE_TO_DEVICE;
    const auto src_len = buffer_length > data_size ? data_size : buffer_length;
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[ForceCopy] model_id:%u, input_index:%zu, id:%zu, offset:0x%" PRIx64 ", base:0x%" PRIx64 ", dst_addr:%p, "
           "dst_size:%" PRIu64 ", src_addr:%p, src_size:%" PRIu64 ", async_mode:%d, kind:%d", model_id_, input_idx, id,
           offset, allocation_ids_to_active_base_addr_[id], des_addr, data_size, data, buffer_length,
           static_cast<int32_t>(is_async_mode_), static_cast<int32_t>(kind));
    }

    if (is_async_mode_) {
      GE_CHK_RT_RET(rtMemcpyAsync(des_addr, data_size, data, src_len, kind, rt_model_stream_));
    } else {
      GE_CHK_RT_RET(rtMemcpy(des_addr, data_size, data, src_len, kind));
    }
  }

  return SUCCESS;
}

Status DavinciModel::ConstructZeroCopyIoActiveBaseAddrs(std::vector<std::pair<uint32_t, uint32_t>> &refreshable_index_to_allocation_ids,
                                                        const std::vector<DataBuffer> &blobs,
                                                        const std::vector<GeTensor> &tensors,
                                                        bool is_input, uint32_t &ret_up,
                                                        std::vector<uint32_t>& id_to_plicy) {
  bool isBlobsEmpty = false;
  if (blobs.size() == 0U) {
    isBlobsEmpty = true;
  }

  for (const auto &item : refreshable_index_to_allocation_ids) {
    const auto &io_idx = item.first;
    const uint32_t id = item.second;

    uint64_t buffer_length = !isBlobsEmpty ? blobs.at(static_cast<size_t>(io_idx)).length :
        tensors.at(io_idx).GetData().size();

    uint32_t pls = !isBlobsEmpty ? blobs[io_idx].placement :
        (!copy_host_input_indexes_.empty() ? static_cast<uint32_t>(tensors[io_idx].GetTensorDesc().GetPlacement()):
        static_cast<uint32_t>(Placement::kPlacementDevice));

    void *data = !isBlobsEmpty ? ValueToPtr(PtrToValue(blobs.at(static_cast<size_t>(io_idx)).data)) :
        ValueToPtr(PtrToValue(tensors[io_idx].GetData().data()));

    GE_ASSERT_TRUE(id != UINT32_MAX);
    GE_ASSERT_TRUE(CheckUserAndModelSize(static_cast<int64_t>(buffer_length),
                                         static_cast<int64_t>(logical_mem_allocations_[id].data_size),
                                         is_input ? K_INPUT : K_OUTPUT),
                   "Check %s size failed, index %u, user size %llu, op size %llu.", (is_input ? "input" : "output"),
                   io_idx, buffer_length, logical_mem_allocations_[id].data_size);
    if (pls == kPlaceHostData) {
      // 如果是随路拷贝的内存，做h2h的拷贝, 并设置更新策略
      // 随路拷贝不会更新args table表中input地址，不会触发io段的更新
      if (is_input && (copy_host_input_indexes_.count(io_idx) > 0)) {
        GE_ASSERT_TRUE(copy_host_input_infos_[io_idx].host_addr != nullptr);
        GE_ASSERT_SUCCESS(
          GeMemcpy(reinterpret_cast<uint8_t *>(copy_host_input_infos_[io_idx].host_addr),
            copy_host_input_infos_[io_idx].tensor_size,
            reinterpret_cast<const uint8_t *>(data), buffer_length), "io index:%u, dst host addr:%p, dst len:%" PRIu64 ", "
            "src host addr:%p, src len:%" PRIu64,
            io_idx, copy_host_input_infos_[io_idx].host_addr, copy_host_input_infos_[io_idx].tensor_size,
            data, buffer_length);
        ret_up = std::max(ret_up, static_cast<uint32_t>(ModelArgsManager::KUpdateHostInput));
      } else {
        // 支持零拷贝，但用户给的host内存，要校验fm内存有没有申请零拷贝段
        if (mem_base_size_ < TotalMemSize()) {
          const std::string reason = "Zero-copy memory reuse mode is enabled, requiring all I/O tensors to be allocated in device memory, but " +
                                      std::string(is_input ? "input " : "output ") + std::to_string(io_idx) +
                                      " is located in host memory, and the model's reusable device memory is insufficient(available: " +
                                      std::to_string(mem_base_size_) + ", required: " + std::to_string(TotalMemSize()) + ")";

          REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                    std::vector<const char_t *>({reason.c_str()}));
          GELOGE(ACL_ERROR_GE_PARAM_INVALID,
              "[Check][Param] %s[%u] placement is host when ge.exec.reuseZeroCopyMemory=1, "
              "no enough memory for zero copy, mem_size:%u while required total_size:%u.",
              is_input ? "input ":" output", io_idx, mem_base_size_, TotalMemSize());
          return ACL_ERROR_GE_PARAM_INVALID;
        }
        if (is_input) {
          host_pls_input_indexes_to_copy_info_[io_idx] = {id, 0U, logical_mem_allocations_[id].data_size};
        } else {
          host_pls_output_indexes_to_copy_info_[io_idx] = {id, 0U, output_indexes_to_tensor_size_[io_idx]};
        }

        if (allocation_ids_to_active_base_addr_[id] != logical_mem_allocations_[id].logical_addr) {
          allocation_ids_to_active_base_addr_[id] = logical_mem_allocations_[id].logical_addr;
          ret_up = std::max(ret_up, id_to_plicy[id]);
        }
      }
    } else {
      if (allocation_ids_to_active_base_addr_[id] != ge::PtrToValue(data)) {
        allocation_ids_to_active_base_addr_[id] = ge::PtrToValue(data);
        ret_up = std::max(ret_up, id_to_plicy[id]);
      }
    }

    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[%s] index:%u, user_addr:0x%" PRIx64 ", active_base_addr:0x%" PRIx64 ", pls:%s, is copy host:%zu.",
        (is_input ? "Input" : "Output"), io_idx, ge::PtrToValue(data), allocation_ids_to_active_base_addr_[id],
        ((pls == kPlaceHostData) ? "host" : "device"),
        ((pls == kPlaceHostData) ? copy_host_input_indexes_.count(io_idx) : 0u));
    }
  }

  return SUCCESS;
}

Status DavinciModel::ConstructZeroCopyIoActiveBaseAddrs(const std::vector<std::pair<uint32_t, uint32_t>> &refreshable_index_to_allocation_ids,
                                                 const std::vector<DataBuffer> &blobs,
                                                 const std::vector<gert::Tensor> &tensors,
                                                 bool is_input, uint32_t &ret_up,
                                                 std::vector<uint32_t>& id_to_plicy) {
  bool isBlobsEmpty = false;
  if (blobs.size() == 0U) {
    isBlobsEmpty = true;
  }

  for (const auto& item : refreshable_index_to_allocation_ids) {
    const auto &io_idx = item.first;
    const uint32_t id = item.second;
    GE_ASSERT_TRUE(io_idx < tensors.size());
    uint64_t buffer_length = !isBlobsEmpty ? blobs.at(static_cast<size_t>(io_idx)).length :
        tensors.at(io_idx).GetSize();
    uint32_t pls = !isBlobsEmpty ? blobs[io_idx].placement :
         static_cast<uint32_t>(tensors[io_idx].GetPlacement() == gert::kOnHost ?
         Placement::kPlacementHost : Placement::kPlacementDevice);

    void *data = !isBlobsEmpty ? ValueToPtr(PtrToValue(blobs.at(static_cast<size_t>(io_idx)).data)) :
        ValueToPtr(PtrToValue(tensors[io_idx].GetAddr()));

    GE_ASSERT_TRUE(id != UINT32_MAX);
    GE_ASSERT_TRUE(CheckUserAndModelSize(static_cast<int64_t>(buffer_length),
                                         static_cast<int64_t>(logical_mem_allocations_[id].data_size),
                                         is_input ? K_INPUT : K_OUTPUT),
                   "Check %s size failed, index %u, user size %llu, op size %llu.", (is_input ? "input" : "output"),
                   io_idx, buffer_length, logical_mem_allocations_[id].data_size);

    if (pls == kPlaceHostData) {
      // 如果时随路拷贝的内存，做h2h的拷贝, 并设置更新策略
      // 随路拷贝不会更新args table表中input地址，不会触发io段的更新
      if (is_input && (copy_host_input_indexes_.count(io_idx) > 0)) {
        GE_ASSERT_TRUE(copy_host_input_infos_[io_idx].host_addr != nullptr);
        GE_ASSERT_SUCCESS(
          GeMemcpy(reinterpret_cast<uint8_t *>(copy_host_input_infos_[io_idx].host_addr),
            copy_host_input_infos_[io_idx].tensor_size,
            reinterpret_cast<const uint8_t *>(data), buffer_length), "io index:%u, dst host addr:%p, dst len:%" PRIu64 ", "
            "src host addr:%p, src len:%" PRIu64,
            io_idx, copy_host_input_infos_[io_idx].host_addr, copy_host_input_infos_[io_idx].tensor_size,
            data, buffer_length);
        ret_up = std::max(ret_up, static_cast<uint32_t>(ModelArgsManager::KUpdateHostInput));
      } else {
        // 支持零拷贝，但用户给的host内存，要校验fm内存有没有申请零拷贝段
        if (mem_base_size_ < TotalMemSize()) {
          const std::string reason = "Zero-copy memory reuse mode is enabled, requiring all I/O tensors to be allocated in device memory, but " +
                                      std::string(is_input ? "input " : "output ") + std::to_string(io_idx) +
                                      " is located in host memory, and the model's reusable device memory is insufficient(available: " +
                                      std::to_string(mem_base_size_) + ", required: " + std::to_string(TotalMemSize()) + ")";
          REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                                    std::vector<const char_t *>({reason.c_str()}));
          GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                "[Check][Param] %s[%u] placement is host when ge.exec.reuseZeroCopyMemory=1, "
                "no enough memory for zero copy, mem_size:%u while required total_size:%u.",
                is_input ? "input ":" output", io_idx, mem_base_size_, TotalMemSize());
          return ACL_ERROR_GE_PARAM_INVALID;
        }
        if (is_input) {
          host_pls_input_indexes_to_copy_info_[io_idx] = {id, 0U, logical_mem_allocations_[id].data_size};
        } else {
          host_pls_output_indexes_to_copy_info_[io_idx] = {id, 0U, output_indexes_to_tensor_size_[io_idx]};
        }
        if (allocation_ids_to_active_base_addr_[id] != logical_mem_allocations_[id].logical_addr) {
          allocation_ids_to_active_base_addr_[id] = logical_mem_allocations_[id].logical_addr;
          ret_up = std::max(ret_up, id_to_plicy[id]);
        }
      }
    } else {
      if (allocation_ids_to_active_base_addr_[id] != ge::PtrToValue(data)) {
        allocation_ids_to_active_base_addr_[id] = ge::PtrToValue(data);
        ret_up = std::max(ret_up, id_to_plicy[id]);
      }
    }
    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[%s] index:%u, user_addr:0x%" PRIx64 ", active_base_addr:0x%" PRIx64 ", pls:%s, is copy host:%zu.",
        (is_input ? "Input" : "Output"), io_idx, ge::PtrToValue(data), allocation_ids_to_active_base_addr_[id],
        ((pls == kPlaceHostData) ? "host" : "device"),
        ((pls == kPlaceHostData) ? copy_host_input_indexes_.count(io_idx) : 0u));
    }
  }

  return SUCCESS;
}

void DavinciModel::ConstructFmActiveMemBaseAddrs(uint32_t &ret_up, std::vector<uint32_t> &active_mem_base_id_to_plicy) {
  for (const auto &it : refreshable_fm_index_and_allocation_ids_) {
    if (allocation_ids_to_active_base_addr_[it.second] !=
        PtrToValue(PtrToPtr<uint8_t, void>(runtime_param_.fm_memory_infos[it.first].memory_base))) {
      allocation_ids_to_active_base_addr_[it.second] =
        PtrToValue(PtrToPtr<uint8_t, void>(runtime_param_.fm_memory_infos[it.first].memory_base));
      ret_up = std::max(ret_up, active_mem_base_id_to_plicy[it.second]);
    }

    if (logLevel_ <= DLOG_INFO) {
      GELOGI("[ActiveMemBase][FM], model_id:%u, id:%u, active mem base:0x%" PRIx64 ", fm_idx:%u.",
            model_id_, it.second, allocation_ids_to_active_base_addr_[it.second], it.first);
    }
  }
}

Status DavinciModel::UpdateAllNodeArgs(const InputData &input_data, const OutputData &output_data,
                                      const std::vector<gert::Tensor> &input_tensor,
                                      const std::vector<gert::Tensor> &output_tensor) {
  if ((input_data.blobs.size() != 0 &&
      input_data.blobs.size() != input_index_to_allocation_ids_.size()) ||
      (output_data.blobs.size() != output_index_to_allocation_ids_.size() &&
      output_data.blobs.size() != 0)) {
    const std::string reason = "The number of inputs or outputs provided by the user is inconsistent with that required by the model";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                               std::vector<const char_t *>({reason.c_str()}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
          "[Check][Param] data size:%zu from model and input blobs.size:%zu are not equal. "
          "Or output size:%zu from model and output blobs size:%zu are not equal. ",
          input_index_to_allocation_ids_.size(), input_data.blobs.size(),
          output_index_to_allocation_ids_.size(), output_data.blobs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if ((input_data.blobs.size() == 0 && input_tensor.size() != input_index_to_allocation_ids_.size()) ||
      (output_data.blobs.size() == 0 && output_tensor.size() != output_index_to_allocation_ids_.size())) {
    const std::string reason = "The number of inputs or outputs provided by the user is inconsistent with that required by the model";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                               std::vector<const char_t *>({reason.c_str()}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
          "[Check][Param] input size:%zu from model and input tensor size:%zu are not equal. "
          "Or output size:%zu from model and output tensor size:%zu are not equal.",
          input_index_to_allocation_ids_.size(), input_tensor.size(),
          output_index_to_allocation_ids_.size(), output_tensor.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  GE_ASSERT_SUCCESS(CheckIoReuseAddrs(input_data.blobs, output_data.blobs, input_tensor, output_tensor),
                    "[Check][IoReuseAddrs] failed, model_id:%u.", model_id_);

  uint32_t ret_up = 0;
  std::vector<uint32_t>& active_mem_base_id_to_plicy = args_manager_.GetId2Policy();
  if (!active_mem_base_id_to_plicy.empty()) {
    GE_ASSERT_SUCCESS(ConstructZeroCopyIoActiveBaseAddrs(
      is_first_time_model_execute_ ?
        refreshable_input_index_no_frozen_and_allocation_ids_ : refreshable_input_index_and_allocation_ids_,
      input_data.blobs, input_tensor, true, ret_up, active_mem_base_id_to_plicy));

    // 非首次刷新零拷贝地址后，使用不变的zero_copy_no_frozen
    if (!is_first_time_model_execute_ && !zero_copy_input_indexes_no_frozen_.empty()
        && !refreshable_input_index_no_frozen_and_allocation_ids_.empty()) {
      is_first_time_model_execute_ = true;
    }

    GE_ASSERT_SUCCESS(ConstructZeroCopyIoActiveBaseAddrs(refreshable_output_index_and_allocation_ids_,
      output_data.blobs, output_tensor, false, ret_up, active_mem_base_id_to_plicy));
    ConstructFmActiveMemBaseAddrs(ret_up, active_mem_base_id_to_plicy);
    ret_up = allocation_ids_to_active_base_addr_[logical_mem_allocations_.size() - 1] == 0 ?
      ret_up : std::max(ret_up, active_mem_base_id_to_plicy[logical_mem_allocations_.size() - 1]);
    allocation_ids_to_active_base_addr_[logical_mem_allocations_.size() - 1] = 0;
  }

  GE_ASSERT_SUCCESS(args_manager_.UpdateForExecute(ret_up, rt_model_stream_));
  return SUCCESS;
}

Status DavinciModel::UpdateAllNodeArgs(const InputData &input_data, const OutputData &output_data,
    const std::vector<GeTensor> &input_tensor, const std::vector<GeTensor> &output_tensor) {
  if ((input_data.blobs.size() != 0 &&
      input_data.blobs.size() != input_index_to_allocation_ids_.size()) ||
      (output_data.blobs.size() != output_index_to_allocation_ids_.size() &&
      output_data.blobs.size() != 0)) {
    const std::string reason = "The number of inputs or outputs provided by the user is inconsistent with that required by the model";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                               std::vector<const char_t *>({reason.c_str()}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
          "[Check][Param] data size:%zu from model and input blobs.size:%zu are not equal. "
          "Or output size:%zu from model and output blobs size:%zu are not equal. ",
          input_index_to_allocation_ids_.size(), input_data.blobs.size(),
          output_index_to_allocation_ids_.size(), output_data.blobs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if ((input_data.blobs.size() == 0 && input_tensor.size() != input_index_to_allocation_ids_.size()) ||
      (output_data.blobs.size() == 0 && output_tensor.size() != output_index_to_allocation_ids_.size())) {
    const std::string reason = "The number of inputs or outputs provided by the user is inconsistent with that required by the model";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                               std::vector<const char_t *>({reason.c_str()}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
          "[Check][Param] input size:%zu from model and input tensor size:%zu are not equal. "
          "Or output size:%zu from model and output tensor size:%zu are not equal.",
          input_index_to_allocation_ids_.size(), input_tensor.size(),
          output_index_to_allocation_ids_.size(), output_tensor.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  GE_ASSERT_SUCCESS(CheckIoReuseAddrs(input_data.blobs, output_data.blobs, input_tensor, output_tensor),
                    "[Check][IoReuseAddrs] failed, model_id:%u.", model_id_);

  uint32_t ret_up = 0;
  std::vector<uint32_t>& active_mem_base_id_to_plicy = args_manager_.GetId2Policy();
  if (!active_mem_base_id_to_plicy.empty()) {
    GE_ASSERT_SUCCESS(ConstructZeroCopyIoActiveBaseAddrs(refreshable_input_index_and_allocation_ids_,
                                                       input_data.blobs, input_tensor, true, ret_up,
                                                       active_mem_base_id_to_plicy));
    GE_ASSERT_SUCCESS(ConstructZeroCopyIoActiveBaseAddrs(refreshable_output_index_and_allocation_ids_,
                                                        output_data.blobs, output_tensor, false, ret_up,
                                                        active_mem_base_id_to_plicy));
    ConstructFmActiveMemBaseAddrs(ret_up, active_mem_base_id_to_plicy);
    if (allocation_ids_to_active_base_addr_[logical_mem_allocations_.size() - 1] != 0) {
      allocation_ids_to_active_base_addr_[logical_mem_allocations_.size() - 1] = 0;
      ret_up = std::max(ret_up, active_mem_base_id_to_plicy[logical_mem_allocations_.size() - 1]);
    }
  }

  GE_ASSERT_SUCCESS(args_manager_.UpdateForExecute(ret_up, rt_model_stream_));
  return SUCCESS;
}

Status DavinciModel::CheckIoReuseAddrs(const std::vector<DataBuffer> &input_blobs,
                                       const std::vector<DataBuffer> &output_blobs,
                                       const std::vector<gert::Tensor> &input_tensors,
                                       const std::vector<gert::Tensor> &output_tensors) const {
  if (io_same_addr_pairs_.empty()) {
    return ge::GRAPH_SUCCESS;
  }

  AddrGetter input_getter;
  size_t input_num = 0;
  if (!input_blobs.empty()) {
    input_num = input_blobs.size();
    input_getter = [&](size_t i) { return input_blobs[i].data; };
  } else if (!input_tensors.empty()) {
    input_num = input_tensors.size();
    input_getter = [&](size_t i) { return input_tensors[i].GetAddr(); };
  } else {
    return SUCCESS;
  }

  AddrGetter output_getter;
  size_t output_num = 0;
  if (!output_blobs.empty()) {
    output_num = output_blobs.size();
    output_getter = [&](size_t i) { return output_blobs[i].data; };
  } else if (!output_tensors.empty()) {
    output_num = output_tensors.size();
    output_getter = [&](size_t i) { return output_tensors[i].GetAddr(); };
  } else {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(ge::CheckIoReuseAddrPairs(io_same_addr_pairs_, input_getter, input_num, output_getter, output_num));
  return SUCCESS;
}

Status DavinciModel::CheckIoReuseAddrs(const std::vector<DataBuffer> &input_blobs,
                                       const std::vector<DataBuffer> &output_blobs,
                                       const std::vector<GeTensor> &input_tensors,
                                       const std::vector<GeTensor> &output_tensors) const {
  if (io_same_addr_pairs_.empty()) {
    return ge::GRAPH_SUCCESS;
  }

  AddrGetter input_getter;
  size_t input_num = 0;
  std::vector<const void *> input_addrs;
  if (!input_blobs.empty()) {
    input_num = input_blobs.size();
    input_getter = [&](size_t i) { return input_blobs[i].data; };
  } else if (!input_tensors.empty()) {
    input_num = input_tensors.size();
    input_getter = [&](size_t i) { return input_tensors[i].GetData().GetData(); };
  } else {
    return SUCCESS;
  }

  AddrGetter output_getter;
  size_t output_num = 0;
  std::vector<const void *> output_addrs;
  if (!output_blobs.empty()) {
    output_num = output_blobs.size();
    output_getter = [&](size_t i) { return output_blobs[i].data; };
  } else if (!output_tensors.empty()) {
    output_num = output_tensors.size();
    output_getter = [&](size_t i) { return output_tensors[i].GetData().GetData(); };
  } else {
    GELOGD("Skip check io reuse addrs, output blobs and tensors are both empty");
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(ge::CheckIoReuseAddrPairs(io_same_addr_pairs_, input_getter, input_num, output_getter, output_num));
  return SUCCESS;
}

void DavinciModel::FreeInnerFeatureMapMem() {
  if ((mem_base_ != 0) && is_inner_mem_base_) {
    GELOGD("Start to free inner feature mem:0x%" PRIx64, mem_base_);
    const auto rt_ret = rtStreamSynchronizeWithTimeout(rt_model_stream_, stream_sync_timeout_);
    if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      is_stream_sync_timeout_ = true;
      GE_LOGE_IF(rtModelAbort(rt_model_handle_) != RT_ERROR_NONE, "Abort model failed!");
      GELOGW("[Invoke][rtStreamSynchronizeWithTimeout] failed, timeout:%dms, ret:%d.", stream_sync_timeout_, rt_ret);
      FreeFeatureMapMem();
      return;
    }
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
      return;
    }
    FreeFeatureMapMem();
  }
  return;
}

Status DavinciModel::ConstructActiveMemBaseAddrsForKnownNode(uint32_t &ret_up,
                                                 const std::vector<uint64_t> &inputs,
                                                 const std::vector<uint64_t> &outputs) {
  std::vector<uint32_t> &active_mem_base_id_to_plicy = args_manager_.GetId2Policy();
  for (const auto &it : refreshable_fm_index_and_allocation_ids_) {
    if (allocation_ids_to_active_base_addr_[it.second] !=
        PtrToValue(PtrToPtr<uint8_t, void>(runtime_param_.fm_memory_infos[it.first].memory_base))) {
      allocation_ids_to_active_base_addr_[it.second] =
        PtrToValue(PtrToPtr<uint8_t, void>(runtime_param_.fm_memory_infos[it.first].memory_base));
      ret_up = std::max(ret_up, active_mem_base_id_to_plicy[it.second]);
      if (logLevel_ <= DLOG_INFO) {
        GELOGI("[ActiveMemBase][FM], model_id:%u, id:%u, active mem base:0x%" PRIx64 ", fm_idx:%u.",
              model_id_, it.second, allocation_ids_to_active_base_addr_[it.second], it.first);
      }
    }
  }

  for (const auto &it : is_first_time_model_execute_ ? refreshable_input_index_no_frozen_and_allocation_ids_ :
      refreshable_input_index_and_allocation_ids_) {
    if (allocation_ids_to_active_base_addr_[it.second] != inputs[it.first]) {
      allocation_ids_to_active_base_addr_[it.second] = inputs[it.first];
      ret_up = std::max(ret_up, active_mem_base_id_to_plicy[it.second]);
      if (logLevel_ <= DLOG_INFO) {
        GELOGI("[ActiveMemBase][INPUT], model_id:%u, id:%u, active mem base:0x%" PRIx64 ", input_idx:%u.",
              model_id_, it.second, inputs[it.first], it.first);
      }
    }
  }

  // 非首次刷新零拷贝地址后，使用不变的zero_copy_no_frozen
  if (!is_first_time_model_execute_ && !zero_copy_input_indexes_no_frozen_.empty()
      && !refreshable_input_index_no_frozen_and_allocation_ids_.empty()) {
    is_first_time_model_execute_ = true;
  }

  for (const auto &it : refreshable_output_index_and_allocation_ids_) {
    if (allocation_ids_to_active_base_addr_[it.second] != outputs[it.first]) {
      allocation_ids_to_active_base_addr_[it.second] = outputs[it.first];
      ret_up = std::max(ret_up, active_mem_base_id_to_plicy[it.second]);
      if (logLevel_ <= DLOG_INFO) {
        GELOGI("[ActiveMemBase][OUTPUT], model_id:%u, id:%u, active mem base:0x%" PRIx64 ", output_idx:%u",
              model_id_, it.second, outputs[it.first], it.first);
      }
    }
  }

  if (allocation_ids_to_active_base_addr_[logical_mem_allocations_.size() - 1] != 0) {
    allocation_ids_to_active_base_addr_[logical_mem_allocations_.size() - 1] = 0;
    ret_up = std::max(ret_up, active_mem_base_id_to_plicy[logical_mem_allocations_.size() - 1]);
  }
  return SUCCESS;
}

std::set<size_t> DavinciModel::GetZeroCopyArgsIndex(const std::vector<uint64_t> &arg_logical_addrs) const {
  std::set<size_t> zero_copy_args_index;

  const auto addr_is_outline = [](const std::map<uint32_t, ZeroCopyOffset> &outline_addrs_info,
                                  const uintptr_t addr) -> bool {
    for (const auto &outline_addr_info : outline_addrs_info) {
      for (const auto &logical2outline : outline_addr_info.second.GetOutsideAddrs()) {
        if (logical2outline.find(addr) != logical2outline.end()) {
          return true;
        }
      }
    }
    return false;
  };

  for (size_t i = 0U; i < arg_logical_addrs.size(); i++) {
    if (addr_is_outline(input_data_info_, static_cast<uintptr_t>(arg_logical_addrs[i])) ||
        addr_is_outline(output_data_info_, static_cast<uintptr_t>(arg_logical_addrs[i]))) {
      (void)zero_copy_args_index.insert(i);
    }
  }

  return zero_copy_args_index;
}

void DavinciModel::SetLogicalOutsideAddrs(const std::map<uintptr_t, std::set<size_t>> &args_offset,
                                          const std::vector<bool> &tiling_list, const uintptr_t args_device_addr) {
  size_t index = 0U;
  for (const auto &logical_offsets : args_offset) {
    const bool is_tiling = (index >= tiling_list.size() ? false : tiling_list[index]);
    for (const auto &offset : logical_offsets.second) {
      GELOGD("Set logical outside device addr for 0x%" PRIx64 ", is tiling:%d, offset:%" PRIu64 ".",
        logical_offsets.first, static_cast<int32_t>(is_tiling), offset);
      for (auto &info : input_data_info_) {
        info.second.SetLogicalOutsideAddrs(logical_offsets.first, is_tiling,
                                           static_cast<uintptr_t>(args_device_addr + offset));
      }
      for (auto &info : output_data_info_) {
        info.second.SetLogicalOutsideAddrs(logical_offsets.first, is_tiling,
                                           static_cast<uintptr_t>(args_device_addr + offset));
      }
    }
    index++;
  }
}

///
/// @ingroup ge
/// @brief set model id
/// @return model ID
///
void DavinciModel::SetId(const uint32_t model_id) {
  model_id_ = model_id;
  bin_kernel_handle_.SetModelId(model_id);
}

const std::string DavinciModel::GetBinHandleKey(const OpDesc &op_desc, const std::string &prefix,
                                                const bool is_atomic_kernel) const {
  return bin_kernel_handle_.GetBinHandleKey(op_desc, prefix, is_atomic_kernel);
}

Status DavinciModel::Mapping2BundleZeroCopy(const OpDescPtr &op_desc,
                                            const std::map<uintptr_t, std::set<size_t>> &args_offset,
                                            const std::vector<bool> &tiling_list, const size_t args_size,
                                            const void *const args_host_copy, void *&args_device_addr,
                                            const bool &own_memory, const bool is_all_kernel) {
  (void)op_desc;
  (void)args_size;
  (void)args_host_copy;
  (void)own_memory;
  (void)is_all_kernel;

  // todo: args_device_addr 这个参数怎么赋值
  SetLogicalOutsideAddrs(args_offset, tiling_list, static_cast<uintptr_t>(PtrToValue(args_device_addr)));

  return ge::SUCCESS;
}

///
/// @ingroup ge
/// @brief Constant Op Init.
/// @return Status
///
Status DavinciModel::InitConstant(const OpDescPtr &op_desc) {
  const auto v_weights = ModelUtils::GetWeights(op_desc);
  const auto v_output_size = ModelUtils::GetOutputSize(op_desc);
  const auto v_output_addr = ModelUtils::GetOutputAddrs(runtime_param_, op_desc);
  if (v_weights.empty() || v_output_size.empty() || v_output_addr.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "weight.size:%zu size.size:%zu addr.size:%zu in Node:%s has empty, check invalid",
        v_weights.size(), v_output_size.size(), v_output_addr.size(), op_desc->GetName().c_str());
    GELOGE(PARAM_INVALID, "const op:%s not set output", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  const GeTensor *const tensor = v_weights[0U].get();
  GE_ASSERT_NOTNULL(tensor);
  if (static_cast<size_t>(v_output_size[0U]) < tensor->GetData().size()) {
    REPORT_INNER_ERR_MSG("E19999", "Output size:%" PRId64 " < weight size:%zu in op:%s(%s) model_id:%u, check invalid",
        v_output_size[0U], tensor->GetData().size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] Output size:%" PRId64 " < weight size:%zu in op:%s(%s), model_id:%u",
        v_output_size[0U], tensor->GetData().size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return PARAM_INVALID;
  }

  if (tensor->GetData().size() == 0U) {
    GELOGW("const op:%s has no weight data.", op_desc->GetName().c_str());
    return SUCCESS;
  }

  GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] "
         "mem_size[%" PRIu64 "] datasize[%zu].",
         runtime_param_.graph_id, op_desc->GetName().c_str(), 0, v_output_addr[0U], v_output_size[0U],
         tensor->GetData().size());
  const auto &var_manager = VarManager::Instance(session_id_);
  GE_CHECK_NOTNULL(var_manager);
  if (!var_manager->CheckAndSetVarLoaded(op_desc, device_id_)) {
    GELOGD("Copy weight to device, node:%s, weight size:%zu", op_desc->GetName().c_str(), tensor->GetData().size());
    GE_CHK_RT_RET(rtMemcpy(v_output_addr[0U], static_cast<uint64_t>(v_output_size[0U]), tensor->GetData().data(),
                           tensor->GetData().size(), RT_MEMCPY_HOST_TO_DEVICE));
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Constant Op Init.
/// @return Status
///
Status DavinciModel::InitFileConstant(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  const auto &tensor_desc = op_desc->GetOutputDescPtr(0U);
  GE_CHECK_NOTNULL(tensor_desc);
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  if (VarManager::Instance(session_id_)->IsVarReady(node->GetName(), *tensor_desc, device_id_)) {
    GELOGI("file constant op:%s is ready", node->GetName().c_str());
    return SUCCESS;
  }
  const auto v_output_addr = ModelUtils::GetOutputAddrs(runtime_param_, op_desc);
  GE_ASSERT_TRUE(!v_output_addr.empty());
  int64_t weight_size;
  GE_ASSERT_SUCCESS(TensorUtils::GetSize(*tensor_desc, weight_size));
  if (weight_size == 0) {
    GELOGW("const op:%s has no weight data.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  GELOGI("[IMAS]Init FileConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] mem_size[%" PRId64 "]",
         runtime_param_.graph_id, op_desc->GetName().c_str(), 0, v_output_addr[0U], weight_size);
  std::string file_path;
  size_t offset = 0U;
  size_t length = 0U;
  GE_CHK_STATUS_RET(FileConstantUtils::GetFilePath(op_desc, file_id_and_path_map_, file_path, offset, length),
                    "Failed to get file path.");
  size_t left_size = static_cast<size_t>(weight_size);
  const auto &external_weight_manager = ExternalWeightManagerPool::Instance().GetManager(session_id_);
  GE_CHECK_NOTNULL(external_weight_manager);
  if (!external_weight_manager->CheckAndSetWeightLoaded(file_path + ":" + std::to_string(offset), device_id_)) {
    const size_t file_length = (length == 0U ? static_cast<size_t>(weight_size) : length);
    GE_CHK_STATUS_RET(
        FileConstantUtils::CopyOneWeightFromFile(v_output_addr[0U], file_path, offset, file_length, left_size),
        "Failed to copy data to file constant.");
    GELOGD("Load file constant [%s] file path [%s] weight size [%zu] to addr [%p] success.", node->GetName().c_str(),
           file_path.c_str(), file_length, v_output_addr[0U]);
  }
  VarManager::Instance(session_id_)->SetVarIsReady(node->GetName(), *tensor_desc, device_id_);
  GELOGI("Finish to copy data to device memory of file constant.");
  return SUCCESS;
}

CustAICPUKernelPtr DavinciModel::GetCustAICPUKernel(const OpDescPtr &op_desc) const {
  CustAICPUKernelPtr aicpu_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
  if (aicpu_kernel != nullptr) {
    return aicpu_kernel;
  }

  // Called by TaskInfo::Init, ge_model_ always valid.
  aicpu_kernel = ge_model_->GetCustAICPUKernelStore().FindKernel(op_desc->GetName());
  if (aicpu_kernel != nullptr) {
    GELOGI("Get cust aicpu so success, will add to extend attr");
    op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, aicpu_kernel);
  }

  return aicpu_kernel;
}

///
/// @ingroup ge
/// @brief TVM Op Init.
/// @return Status
///
Status DavinciModel::InitTbeHandle(const OpDescPtr &op_desc) {
  if (!IsTbeTask(op_desc)) {
    return SUCCESS;
  }
  uint32_t thread_mode = kInValidThreadMode;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_THREAD_MODE, thread_mode);
  // ffts mode only has ATTR_NAME_THREAD_SCOPE_ID attr.
  // ffts plus auto mode has ATTR_NAME_THREAD_SCOPE_ID attr and thread mode is 1.
  // ffts plus manual mode has ATTR_NAME_THREAD_SCOPE_ID attr and thread mode is 0.
  // normal mode do not have ATTR_NAME_THREAD_SCOPE_ID and ATTR_NAME_THREAD_MODE attr.
  // only ffts mode and ffts plus auto mode enter the branch.
  if (op_desc->HasAttr(ATTR_NAME_THREAD_SCOPE_ID) && (thread_mode != kManualThreadMode)) {
    return bin_kernel_handle_.RegisterAutoThreadHandle(op_desc, ge_model_->GetTBEKernelStore());
  }

  const bool is_dynamic = IsAllKernelTask(op_desc);
  GELOGD("kernel name: %s, is dynamic: %d.", op_desc->GetNamePtr(), is_dynamic);
  std::vector<std::string> names_prefix;
  (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  if (!names_prefix.empty()) {
    if (is_dynamic) {
      for (const auto &prefix : names_prefix) {
        GE_CHK_STATUS_RET_NOLOG(
            bin_kernel_handle_.RegisterDynamicKernel(op_desc, prefix, ge_model_->GetTBEKernelStore()));
      }
    } else {
      for (const auto &prefix : names_prefix) {
        GE_CHK_STATUS_RET_NOLOG(
            bin_kernel_handle_.RegisterStaticHandle(op_desc, prefix, ge_model_->GetTBEKernelStore()));
      }
    }
  } else {
    if (is_dynamic) {
      GE_CHK_STATUS_RET_NOLOG(bin_kernel_handle_.RegisterDynamicKernel(op_desc, "", ge_model_->GetTBEKernelStore()));
    } else {
      GE_CHK_STATUS_RET_NOLOG(bin_kernel_handle_.RegisterStaticHandle(op_desc, "", ge_model_->GetTBEKernelStore()));
    }
  }

  std::string atomic_kernel_name;
  if (IsNeedAtomicCleanTask(op_desc) &&
      AttrUtils::GetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, atomic_kernel_name)) {
    GE_CHK_STATUS_RET_NOLOG(
        bin_kernel_handle_.RegisterStaticHandle(op_desc, kAtomicPrefix, ge_model_->GetTBEKernelStore(), true));
  }
  return SUCCESS;
}

Status DavinciModel::GetAddrAndPrefCnt(const OpDescPtr &op_desc, const std::string &kernel_name,
    const std::string &prefix,
    std::vector<std::pair<void *, uint32_t>> &addr_pref_cnt) const {
  addr_pref_cnt.clear();
  return bin_kernel_handle_.GetAddrAndPrefCnt(op_desc, kernel_name, prefix, addr_pref_cnt);
}

///
/// @ingroup ge
/// @brief insert active_stream_indication_
/// @return Status
///
Status DavinciModel::InitStreamActive(const OpDescPtr &op_desc) {
  if (op_desc->HasAttr(ATTR_NAME_SWITCH_BRANCH_NODE_LABEL)) {
    std::vector<uint32_t> active_stream_list;
    if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) {
      REPORT_INNER_ERR_MSG("E19999", "[Get][Attr] active_stream_list in op:%s(%s) failed, model_id:%u.",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      GELOGE(INTERNAL_ERROR, "[Get][Attr] active_stream_list in op:%s(%s) failed, model_id:%u.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      return INTERNAL_ERROR;
    }

    for (size_t j = 0U; j < active_stream_list.size(); ++j) {
      (void)active_stream_indication_.insert(active_stream_list[j]);
      GELOGI("flowctrl_op_index_map node:%s, active_stream_id=%u.", op_desc->GetName().c_str(), active_stream_list[j]);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitStreamSwitch(const OpDescPtr &op_desc) {
  std::vector<uint32_t> active_stream_list;
  GE_LOGI_IF(!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
             "GetInt active_stream_list failed.");
  if (active_stream_list.size() != kTrueBranchStreamCount) {
    REPORT_INNER_ERR_MSG("E19999", "[Check][Param] Attr: active_stream_list.size:%zu in op:%s(%s) != 1, model_id:%u, "
        "check invalid", active_stream_list.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Attr: active_stream_list.size:%zu in op:%s(%s) != 1, model_id:%u",
           active_stream_list.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return INTERNAL_ERROR;
  }

  const uint32_t true_stream_id = active_stream_list.front();
  (void)active_stream_indication_.insert(true_stream_id);
  GELOGI("flowctrl_op_index_map node:%s, true_stream_id=%u.", op_desc->GetName().c_str(), true_stream_id);

  return SUCCESS;
}

Status DavinciModel::SetDynamicBatchInfo(const OpDescPtr &op_desc, const uint32_t batch_num) {
  batch_info_.clear();
  combined_batch_info_.clear();

  (void)AttrUtils::GetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type_);
  (void)AttrUtils::GetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order_);
  for (uint32_t i = 0U; i < batch_num; ++i) {
    std::vector<int64_t> batch_shape;
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::GetListInt(op_desc, attr_name, batch_shape)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) fail, model_id:%u", attr_name.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) fail, model_id:%u", attr_name.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      batch_info_.clear();
      return FAILED;
    }
    batch_info_.emplace_back(batch_shape);
    batch_shape.clear();
    const std::string attr_combined_batch = ATTR_NAME_COMBINED_BATCH + "_" + std::to_string(i);
    if (AttrUtils::GetListInt(op_desc, attr_combined_batch, batch_shape)) {
      combined_batch_info_.emplace_back(batch_shape);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitCase(const OpDescPtr &op_desc) {
  uint32_t batch_num = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGI("Not multi-batch Node: %s", op_desc->GetName().c_str());
    return SUCCESS;
  }

  return SetDynamicBatchInfo(op_desc, batch_num);
}

///
/// @ingroup ge
/// @brief Init model stream for NN model.
/// @param [in] stream   user input model stream.
/// @return Status
///
Status DavinciModel::InitModelStream(rtStream_t const stream) {
  const ExecuteMode curr_mode = is_async_mode_ ? ExecuteMode::ASYNCHRONIZATION : ExecuteMode::SYNCHRONIZATION;
  GE_CHK_BOOL_RET_STATUS((curr_mode == last_execute_mode_) || (last_execute_mode_ == ExecuteMode::INITIALIZATION),
                         INTERNAL_ERROR, "[Check][Param] NnExecute not support mix execute.");
  last_execute_mode_ = curr_mode;

  // asynchronize mode, use user input stream.
  if (is_async_mode_) {
    if (is_inner_model_stream_) {
      // destroy stream that is bound with rt_model
      (void)reusable_stream_allocator_->DestroyStream(rt_model_stream_, true);
      rt_model_stream_ = nullptr;
    }
    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  // synchronize mode, use forbidden stream.
  if (stream != nullptr) {
    if ((rt_model_stream_ != nullptr) && is_inner_model_stream_) {
      rt_stream_to_destroy_ = rt_model_stream_;
    }

    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  if (rt_model_stream_ == nullptr) {
    uint32_t stream_flags = RT_STREAM_FORBIDDEN_DEFAULT;
    if (GetContext().IsOverflowDetectionOpen()) {
      stream_flags |= RT_STREAM_OVERFLOW;
    }
    // only static model need to create model stream, rt2 is certain to pass stream
    GE_CHECK_NOTNULL(reusable_stream_allocator_);
    GE_ASSERT_SUCCESS(
        reusable_stream_allocator_->GetOrCreateRtStream(rt_model_stream_, runtime_model_id_, priority_, stream_flags));
    is_inner_model_stream_ = true;
    is_forbidden_stream_ = true;
  }

  return SUCCESS;
}

void DavinciModel::RecordProfileTime() {
  gert::GlobalProfilingWrapper::GetInstance()->Record(gert::profiling::kUnknownName,
      gert::profiling::kDavinciModelCopyH2D, ExecutorEvent::kExecuteStart, davinci_model_stage_time_[kStageBeforeH2D]);
  gert::GlobalProfilingWrapper::GetInstance()->Record(gert::profiling::kUnknownName,
    gert::profiling::kDavinciModelCopyH2D, ExecutorEvent::kExecuteEnd, davinci_model_stage_time_[kStageBeforeRtExecute]);
  gert::GlobalProfilingWrapper::GetInstance()->Record(gert::profiling::kUnknownName,
    gert::profiling::kRtModelExecute, ExecutorEvent::kExecuteStart, davinci_model_stage_time_[kStageBeforeRtExecute]);
  gert::GlobalProfilingWrapper::GetInstance()->Record(gert::profiling::kUnknownName,
    gert::profiling::kRtModelExecute, ExecutorEvent::kExecuteEnd, davinci_model_stage_time_[kStageAfterRtExecute]);
}

Status DavinciModel::CheckRtStreamSynchronize(rtError_t rt_ret) {
  if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
    is_stream_sync_timeout_ = true;
    GE_LOGW_IF(rtModelAbort(rt_model_handle_) != RT_ERROR_NONE, "Abort model failed!");
    GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, timeout:%dms, ret:%d.", stream_sync_timeout_,
            rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%dms, ret:%d.",
                      stream_sync_timeout_, rt_ret);
    return FAILED;
  }
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
    return FAILED;
  }
  return SUCCESS;
}

Status DavinciModel::NnExecute(rtStream_t const stream, const bool async_mode,
                               const std::vector<gert::Tensor> &input_tensor,
                               std::vector<gert::Tensor> &output_tensor) {
  InitModelExecuteProf();
  is_async_mode_ = async_mode;
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Model Run start, model id:%u, data index:%u, flag:%d.",
         model_id_, 0, static_cast<int32_t>(is_async_mode_));
  }

  GE_CHK_STATUS_RET(InitModelStream(stream), "[Init][ModelStream] failed, model_id:%u.", model_id_);
  GE_ASSERT_TRUE(!(is_forbidden_stream_ && host_input_size_ != 0U),
    "forbidden stream no support host input tensor option");
  is_dynamic_ = is_online_infer_dynamic_;

  const bool is_prof_enabled = gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime);
  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_PRE_PROC_START));
  GE_IF_BOOL_EXEC(is_dump_to_std_enable_, davinci_model_stage_time_[kStageBeforeH2D] = std::chrono::system_clock::now());

  GetStageTimestampStart(kCopyMdlData);
  Status ret = CopyModelData(input_tensor, output_tensor);
  GetStageTimestampEnd(kCopyMdlData);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Copy][ModelData] failed. model id: %u", model_id_);
    return ret;
  }

  GE_IF_BOOL_EXEC(is_dump_to_std_enable_, davinci_model_stage_time_[kStageBeforeRtExecute] = std::chrono::system_clock::now());
  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_PRE_PROC_END));
  if (!task_list_.empty()) {
    // used for debug resource manager
    if (GetDumpProperties().IsDumpOpen() || GetDumpProperties().IsOpDebugOpen()) {
      GELOGD("update step[%" PRIu64 "] info to device", iterator_count_);
      GE_CHK_STATUS_RET(UpdateStepInfoWithStream(), "UpdateStepInfoWithStream failed");
    }

    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_INFER_START));
    CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 0U, rt_model_stream_);
    GetStageTimestampStart(kMdlExecute);
    rtError_t rt_ret;
    if (is_forbidden_stream_ && is_inner_model_stream_) {
      // MDC场景:使用的是forbidden流+模型执行时设置了超时时间时，调用rtModelExecuteSync接口，其内部会做流同步，超时会abort model
      const int32_t stream_sync_timeout_exe = GetContext().StreamSyncTimeout();
      GELOGD("[NnExecute] Get stream_sync_timeout_exe: %dms.", stream_sync_timeout_exe);
      rt_ret = rtModelExecuteSync(rt_model_handle_, rt_model_stream_, 0U, stream_sync_timeout_exe);
    } else {
      rt_ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0U);
    }
    GetStageTimestampEnd(kMdlExecute);
    CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 1U, rt_model_stream_);
    GE_CHK_RT_EXEC(rt_ret, return RT_ERROR_TO_GE_STATUS(rt_ret));

    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_INFER_END));
    iterator_count_++;
  }
  if (is_inner_model_stream_ &&
    (is_prof_enabled || (ProfilingManager::Instance().ProfilingSubscribeOn()) || (!is_forbidden_stream_))) {
    const auto rt_ret = rtStreamSynchronizeWithTimeout(rt_model_stream_, stream_sync_timeout_);
    if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      is_stream_sync_timeout_ = true;
      GE_LOGW_IF(rtModelAbort(rt_model_handle_) != RT_ERROR_NONE, "Abort model failed!");
      GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, timeout:%dms, ret:%d.", stream_sync_timeout_,
             rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%dms, ret:%d.",
                        stream_sync_timeout_, rt_ret);
      return FAILED;
    }
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
      return FAILED;
    }
  }
  GE_IF_BOOL_EXEC(is_dump_to_std_enable_, davinci_model_stage_time_[kStageAfterRtExecute] = std::chrono::system_clock::now());
  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_AFTER_PROC_START));
  GetStageTimestampStart(kCopyOutputData);
  ret = CopyOutputData(output_tensor);
  GetStageTimestampEnd(kCopyOutputData);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Copy][OutputData] to user failed, model_id:%u.", model_id_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_AFTER_PROC_END));
  // report model time data
  if (!task_list_.empty() && is_prof_enabled) {
    SinkTimeProfile(0, iterator_count_ - 1UL);
  }

  if (is_online_infer_dynamic_ || has_no_tiling_output_) {
    UpdateOutputTensorShape(output_tensor);
  }
  GE_IF_BOOL_EXEC(is_dump_to_std_enable_, RecordProfileTime());

  GE_ASSERT_SUCCESS(LaunchEventForHcclGroupOrderedStream(rt_model_stream_));
  PrintfModelProfOfModelExecute();
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief ACL case, do not start  new thread, return execute result.
/// @param [in] stream   execute model stream.
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data.
/// @param [out] output_data  model output data.
///
Status DavinciModel::NnExecute(rtStream_t const stream, const bool async_mode, const InputData &input_data,
                               OutputData &output_data, const std::vector<GeTensor> &input_tensor,
                               const std::vector<GeTensor> &output_tensor) {
  InitModelExecuteProf();
  is_async_mode_ = async_mode;
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Model Run start, model id:%u, data index:%u, flag:%d.",
         model_id_, input_data.index, static_cast<int32_t>(is_async_mode_));
  }
  GE_CHK_STATUS_RET(InitModelStream(stream), "[Init][ModelStream] failed, model_id:%u.", model_id_);
  GE_ASSERT_TRUE(!(is_forbidden_stream_ && host_input_size_ != 0U),
    "forbidden stream no support host input tensor option");
  is_dynamic_ = ((input_data.is_dynamic_batch) || (is_online_infer_dynamic_));

  const bool is_prof_enabled = gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime);
  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_PRE_PROC_START));
  GetStageTimestampStart(kCopyMdlData);
  Status ret = CopyModelData(const_cast<InputData &>(input_data), output_data,
                             input_tensor, output_tensor);
  GetStageTimestampEnd(kCopyMdlData);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Copy][ModelData] failed. model id: %u", model_id_);
    return ret;
  }

  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_PRE_PROC_END));

  if (!task_list_.empty()) {
    // used for debug resource manager
    if (GetDumpProperties().IsDumpOpen() || GetDumpProperties().IsOpDebugOpen()) {
      if (logLevel_ <= DLOG_DEBUG) {
        GELOGD("update step[%" PRIu64 "] info to device", iterator_count_);
      }
      GE_CHK_STATUS_RET(UpdateStepInfoWithStream(), "UpdateStepInfoWithStream failed");
    }
    // tag_id 0 means step begin, 1 meas step end.
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_INFER_START));
    CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 0U, rt_model_stream_);

    if (is_forbidden_stream_ && is_inner_model_stream_) {
      // MDC场景:使用的是forbidden流+模型执行时设置了超时时间时，调用rtModelExecuteSync接口，其内部会做流同步，超时会abort model
      const int32_t stream_sync_timeout_exe = GetContext().StreamSyncTimeout();
      GELOGD("[NnExecute] Get stream_sync_timeout_exe: %dms.", stream_sync_timeout_exe);
      GetStageTimestampStart(kMdlExecute);
      const rtError_t rt_ret = rtModelExecuteSync(rt_model_handle_, rt_model_stream_, 0U, stream_sync_timeout_exe);
      GetStageTimestampEnd(kMdlExecute);
      CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 1U, rt_model_stream_);
      GE_CHK_RT_EXEC(rt_ret, return RT_ERROR_TO_GE_STATUS(rt_ret));
    } else {
      const rtError_t rt_ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0U);
      CANN_PROFILING_STEP_TRACE(model_id_, iterator_count_, 1U, rt_model_stream_);
      GE_CHK_RT_EXEC(rt_ret, return RT_ERROR_TO_GE_STATUS(rt_ret));
    }
    GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_INFER_END));
    iterator_count_++;
  }
  if ((is_prof_enabled || (ProfilingManager::Instance().ProfilingSubscribeOn()) || (!is_forbidden_stream_)) &&
      is_inner_model_stream_) {
    const auto rt_ret = rtStreamSynchronizeWithTimeout(rt_model_stream_, stream_sync_timeout_);
    if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      is_stream_sync_timeout_ = true;
      GE_LOGW_IF(rtModelAbort(rt_model_handle_) != RT_ERROR_NONE, "Abort model failed!");
      GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, timeout:%dms, ret:%d.", stream_sync_timeout_,
             rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%dms, ret:%d.",
                        stream_sync_timeout_, rt_ret);
      return FAILED;
    }
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
      return FAILED;
    }
  }

  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_AFTER_PROC_START));
  output_data.index = input_data.index;
  output_data.model_id = model_id_;
  GetStageTimestampStart(kCopyOutputData);
  ret = CopyOutputData(output_data, output_tensor);
  GetStageTimestampEnd(kCopyOutputData);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Copy][OutputData] to user failed, model_id:%u.", model_id_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  GE_IF_BOOL_EXEC(is_prof_enabled, SetProfileTime(ModelProcStage::MODEL_AFTER_PROC_END));
  // report model time data
  if (!task_list_.empty() && is_prof_enabled) {
    SinkTimeProfile(input_data.index, iterator_count_ - 1UL);
  }

  GE_ASSERT_SUCCESS(LaunchEventForHcclGroupOrderedStream(rt_model_stream_));
  PrintfModelProfOfModelExecute();
  return SUCCESS;
}

std::shared_ptr<MemoryBlockManager> DavinciModel::GetAllocator() {
  auto &allocator = mem_type_to_allocator_[RT_MEMORY_HBM];
  if (allocator == nullptr) {
    allocator = ge::MakeShared<MemoryBlockManager>(RT_MEMORY_HBM, kHugePagesize);
    GE_ASSERT_NOTNULL(allocator);
  }
  return allocator;
}

// Add active entry stream for special env.
Status DavinciModel::AddHeadStream() {
  if (active_stream_list_.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "active_stream_list is empty in model:%u, check invalid", model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] active_stream_list is empty in model:%u, check invalid", model_id_);
    return INTERNAL_ERROR;
  }

  if (active_stream_list_.size() == 1U) {
    GELOGI("Just one active stream, take as head stream.");
    rt_head_stream_ = active_stream_list_[0U];
    is_pure_head_stream_ = false;
  } else {
    // Create stream which rt_model_handel running on, this is S0, TS stream.
    GELOGI("Multiple active stream: %zu, create head stream.", active_stream_list_.size());
    GE_CHECK_NOTNULL(reusable_stream_allocator_);
    GE_ASSERT_SUCCESS(reusable_stream_allocator_->GetOrCreateRtStream(rt_head_stream_, runtime_model_id_, priority_,
                                                                      RT_STREAM_PERSISTENT));
    GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, rt_head_stream_, static_cast<uint32_t>(RT_INVALID_FLAG)));
    is_pure_head_stream_ = true;

    for (const auto &s : active_stream_list_) {
      const auto active_entry = MakeShared<CpuTaskActiveEntry>(rt_head_stream_);
      GE_CHECK_NOTNULL(active_entry);

      const Status status = active_entry->Init(s);
      if (status != SUCCESS) {
        return status;
      }

      cpu_task_list_.emplace_back(active_entry);
    }
  }

  // Create entry stream active head stream. AICPU stream.
  GE_ASSERT_SUCCESS(reusable_stream_allocator_->GetOrCreateRtStream(rt_entry_stream_, runtime_model_id_, priority_,
                                                                    RT_STREAM_AICPU));
  GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, rt_entry_stream_, static_cast<uint32_t>(RT_HEAD_STREAM)));
  return SUCCESS;
}

uint8_t *DavinciModel::MallocFeatureMapMem(const size_t data_size) {
  uint8_t *temp_mem_base = nullptr;
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  if ((!is_static_model_addr_fixed_) && ModelUtils::IsGeUseExtendSizeMemory()) {
    auto mem_allocator =
        SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(session_id_, GetDeviceId());
    GE_ASSERT_NOTNULL(mem_allocator);
    LogicalMemorys logical_memorys;
    for (auto &info : runtime_param_.fm_memory_infos) {
      logical_memorys.emplace_back(info.logic_memory_base, info.memory_size);
    }

    // 添加零拷贝信息
    if ((runtime_param_.zero_copy_size != 0) && (runtime_param_.mem_size >= data_size)) {
      if (runtime_param_.mem_size > data_size) {
        logical_memorys.emplace_back(data_size, runtime_param_.zero_copy_size, false, true);
      } else {
        logical_memorys.emplace_back(runtime_param_.mem_size - static_cast<uint64_t>(runtime_param_.zero_copy_size),
                                     static_cast<uint64_t>(runtime_param_.zero_copy_size), true, true);
      }
    }

    temp_mem_base = mem_allocator->MallocMemory(kPurpose, logical_memorys, active_memorys_, GetDeviceId());
    support_extend_memory_full_ = (!is_async_mode_) && mem_allocator->IsSupportExpandableMemoryFull();
    runtime_param_.fm_memory_infos.clear();
    for (const auto &info : logical_memorys) {
      // 更新合并后内存分段信息
      MemInfo fm_info(info.logical_addr, info.memory_size, info.active_addr);
      dev_mem_statistic_.shared_size += info.memory_size;
      // 零拷贝部分去掉
      if (!info.is_zero_copy) {
        runtime_param_.fm_memory_infos.emplace_back(fm_info);
      } else {
        // IO内存为外部分配, 逻辑地址取当前已分配内存最大地址，防止地址交叉
        if (static_cast<size_t>(info.logical_addr) == data_size) {
          io_mem_base_ = (io_mem_base_ < static_cast<uintptr_t>(PtrToValue(info.active_addr))) ?
            static_cast<uintptr_t>(PtrToValue(info.active_addr)) : io_mem_base_;
          fm_info.memory_base = PtrToPtr<void, uint8_t>(ValueToPtr(io_mem_base_));
        }
      }
      (void) runtime_param_.sorted_memory_infos.insert(std::move(fm_info));
    }

    if (static_cast<size_t>(mem_allocator->MemorySize()) < data_size - fixed_mem_size_) {
      REPORT_INNER_ERR_MSG("E19999", "Malloc feature map memory fail. malloced_memory_size[%" PRId64 "] < mem_size[%zu],"
                        " device_id[%u]", mem_allocator->MemorySize(), data_size, GetDeviceId());
      GELOGE(ge::INTERNAL_ERROR, "Malloc feature map memory fail. malloced_memory_size[%" PRId64 "] < mem_size[%zu],"
                                 " device_id[%u]", mem_allocator->MemorySize(), data_size, GetDeviceId());
      if (mem_allocator->FreeMemory(GetDeviceId()) != SUCCESS) {
        GELOGE(ge::INTERNAL_ERROR, "Free feature map memory fail. device_id[%u]", GetDeviceId());
      }
      return nullptr;
    }
    // allocator里已经完成rtMemset，避免后面再次rtMemset
    return temp_mem_base;
  } else {
    GE_ASSERT_TRUE(data_size > fixed_mem_size_, "data size:%zu less than or equal fix mem size:%zu",
      data_size, fixed_mem_size_);
    const size_t mem_size = data_size - fixed_mem_size_;

    temp_mem_base = mem_instance.MallocMemory(kPurpose, mem_size, GetDeviceId());
    GE_ASSERT_NOTNULL(temp_mem_base);
    auto ret = rtMemset(temp_mem_base, mem_size, 0U, mem_size);
    if (ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[RtMemset][Memory] failed, ret = %d", ret);
      GE_ASSERT_SUCCESS(mem_instance.FreeMemory(temp_mem_base, GetDeviceId()),
        "Free hbm feature map memory fail. device_id[%u]", GetDeviceId());
      return nullptr;
    }
    dev_mem_statistic_.alloc_size += mem_size;

    // 非扩展模式，fix fm设置场景，fm为多段需要单独赋值，该场景默认IO内存外部分配, ge内部分配了并不使用
    if (runtime_param_.fixed_mem_base != 0U) {
      if (UpdateHbmFmMemBasesWithInnerMemory(temp_mem_base, mem_size, data_size) != SUCCESS) {
        GE_ASSERT_SUCCESS(mem_instance.FreeMemory(temp_mem_base, GetDeviceId()),
          "Free hbm feature map memory fail. device_id[%u]", GetDeviceId());
        return nullptr;
      }
    }

    return temp_mem_base;
  }
}

Status DavinciModel::UpdateHbmFmMemBasesWithInnerMemory(uint8_t *mem_base, const size_t mem_size,
                                                        const size_t data_size) {
  size_t used_mem_size = 0U;
  const uintptr_t hbm_mem_base = static_cast<uintptr_t>(PtrToValue(mem_base));
  GE_ASSERT_SUCCESS(UpdateHbmFmMemBases(hbm_mem_base, mem_size, used_mem_size, true));
  GELOGI("Update feature memory base success, model_id:%u, mem_base:%#lx, mem_size:%zu, used_mem_size:%zu",
    model_id_, hbm_mem_base, mem_size, used_mem_size);

  if (runtime_param_.mem_size == data_size) { // GE分配了IO段, IO段使用实际分配的地址作为逻辑地址
    GE_ASSERT_TRUE(mem_size >= (used_mem_size + static_cast<size_t>(runtime_param_.zero_copy_size)),
      "mem size:%zu less than sum of refresh fm size and zero copy size :%zu",
      mem_size, used_mem_size + runtime_param_.zero_copy_size);
    MemInfo fm_info(static_cast<int64_t>(runtime_param_.mem_size) - runtime_param_.zero_copy_size,
      runtime_param_.zero_copy_size, mem_base + used_mem_size);
    (void) runtime_param_.sorted_memory_infos.insert(std::move(fm_info));
  } else {  // GE未分配IO段，IO段的逻辑地址使用当前已分配的最大的物理地址
    io_mem_base_ = (io_mem_base_ < (static_cast<uintptr_t>(PtrToValue(mem_base)) + mem_size)) ?
      (static_cast<uintptr_t>(PtrToValue(mem_base)) + mem_size) : io_mem_base_;

    MemInfo fm_info(runtime_param_.mem_size - static_cast<uint64_t>(runtime_param_.zero_copy_size),
      runtime_param_.zero_copy_size, PtrToPtr<void, uint8_t>(ValueToPtr(io_mem_base_)));
    (void) runtime_param_.sorted_memory_infos.insert(std::move(fm_info));
  }
  return SUCCESS;
}

Status DavinciModel::MallocExMem() {
  GE_CHK_STATUS_RET(ModelUtils::MallocExMem(GetDeviceId(), runtime_param_),
                    "MallocExMem fail, model_id:%u", model_id_);
  for (auto &it : runtime_param_.memory_infos) {
    const auto mem_size = it.second.memory_size;
    const rtMemType_t  mem_type = static_cast<rtMemType_t>(it.second.memory_type);
    if ((mem_type != RT_MEMORY_HOST) && (mem_type != RT_MEMORY_HOST_SVM) && (mem_size > 0)) {
      dev_mem_statistic_.alloc_size += static_cast<uint64_t>(mem_size);
    }
  }
  return SUCCESS;
}

/*
 * 加载时申请的内存，单独申请，统一释放。
 */
void* DavinciModel::MallocDynamicMemory(const size_t size, const rtMemType_t mem_type) {
  auto &allocator = mem_type_to_allocator_[mem_type];
  const auto block_size = ((mem_type == RT_MEMORY_TS) ? kSmallPagesize : kHugePagesize);
  if (allocator == nullptr) {
    allocator = ge::MakeShared<MemoryBlockManager>(mem_type, block_size);
    GE_ASSERT_NOTNULL(allocator);
  }

  auto ptr = allocator->Malloc(kMallocPurpose, size);
  GE_ASSERT_NOTNULL(ptr, "malloc failed, size: %zu, mem_type: %u", size, mem_type);
  GELOGI("malloc success, ptr: %p, size: %zu, mem_type: %u, block_size: %zu", ptr, size, mem_type, block_size);
  return ptr;
}

void DavinciModel::FreeDynamicWorkspaceMemory() {
  for (auto &iter : mem_type_to_allocator_) {
    if (iter.second != nullptr) {
      iter.second->Release();
    }
  }
  mem_type_to_allocator_.clear();
  GELOGI("free dynamic memory.");
}

uint8_t *DavinciModel::MallocWeightsMem(const size_t weights_size) const {
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  const std::string purpose("weights memory in inference network");
  return mem_instance.MallocMemory(purpose, weights_size, GetDeviceId());
}

uint8_t *DavinciModel::MallocFileConstantMem(const size_t weights_size) const {
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  const std::string purpose("fileconstant memory in inference network");
  return mem_instance.MallocMemory(purpose, weights_size, GetDeviceId());
}

void DavinciModel::FreeFeatureMapMem() {
  if (!is_inner_mem_base_) {
    return;
  }

  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  if (ModelUtils::IsGeUseExtendSizeMemory()) {
    auto mem_allocator =
        SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(session_id_, GetDeviceId());
    if (mem_allocator != nullptr) {
      // 正常执行时，执行后会释放物理内存，如果一次都未执行，这里需要先释放物理内存
      if (support_extend_memory_full_ && (is_first_execute_ || (iterator_count_ == 0U))){
        mem_allocator->Recycle(active_memorys_);
      }
      GE_CHK_STATUS(mem_allocator->FreeMemory(GetDeviceId()), "failed to free FeatureMap");
      GELOGD("Succeed to free extend-size static featuremap memory.");
    }
  } else {
    if (mem_base_ != 0U) {
      GE_CHK_STATUS(mem_instance.FreeMemory(ValueToPtr(static_cast<uint64_t>(mem_base_)), GetDeviceId()),
                    "failed to free FeatureMap");
      GELOGD("Succeed to free featuremap memory.");
    }
  }
  mem_base_ = 0U;
}

void DavinciModel::FreeExMem() {
  return ModelUtils::FreeExMem(GetDeviceId(), runtime_param_, session_id_,
                               domi::GetContext().is_online_model);
}

void DavinciModel::FreeWeightsMem() {
  if ((weights_mem_base_ != 0U) && (weights_mem_base_ != mem_base_) && is_inner_weight_base_) {
    GE_CHK_STATUS(
        ModelManager::FreeWeightsMem(GetWeightsMemId(), GetDeviceId(), reinterpret_cast<uint8_t *>(weights_mem_base_)),
        "failed to free Weight.");
    weights_mem_base_ = 0U;
  }
}

void DavinciModel::FreeFileConstantMem() {
  if (runtime_param_.fileconstant_addr_mapping.size() == 0UL) {
    GELOGI("Need not to free fileconstant memory.");
    return;
  }

  // 外置权重归一场景：智能指针会在析构函数中自动释放内存（通过自定义deleter）
  if (external_weight_combined_mem_addr_ != nullptr) {
    GELOGI("Combined external weight memory will be automatically freed by smart pointer.");
    return;
  }

  // 非归一场景：遍历各个FileConstant节点，检查是否为用户内存，仅释放GE分配的内存
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  for (const auto &addr_pair : runtime_param_.fileconstant_addr_mapping) {
    if (IsUserDeviceMemForFileConstant(addr_pair.second)) {
      // 用户管理的内存，跳过释放
      GELOGI("Skipping user-managed FileConstant memory at addr:0x%" PRIx64, addr_pair.second);
      continue;
    }
    if (addr_pair.second != 0UL) {
      GE_CHK_STATUS(mem_instance.FreeMemory(ValueToPtr(static_cast<uint64_t>(addr_pair.second)), GetDeviceId()),
                    "failed to free fileconstant");
      GELOGD("Finish to free fileconstant memory. offset:%" PRId64 " addr:0x%" PRIx64,
             addr_pair.first, addr_pair.second);
    }
  }
}

Status DavinciModel::TransAllVarData(const ComputeGraphPtr &graph, const std::vector<NodePtr> &variable_nodes) const {
  const uint32_t device_id = GetDeviceId();
  GE_CHK_STATUS_RET(TransVarDataUtils::TransAllVarData(variable_nodes, session_id_, runtime_param_.graph_id, device_id),
                    "[Call][TransAllVarData] failed, graph:%s, session_id:%" PRIu64 ", graph_id:%u, device_id:%u",
                    graph->GetName().c_str(), session_id_, runtime_param_.graph_id, device_id);

  GE_CHK_STATUS_RET(TransVarDataUtils::CopyVarData(graph, variable_nodes, session_id_, device_id),
                    "[Copy][CopyVarData] failed, graph:%s, session_id:%" PRIu64 ", graph_id:%u, device_id:%u",
                    graph->GetName().c_str(), session_id_, runtime_param_.graph_id, device_id);
  return SUCCESS;
}

uint32_t DavinciModel::GetDumpModelId() const {
  if (CheckModelNoInputAndOutput()) {
    return model_id_;
  }
  return runtime_model_id_;
}

Status DavinciModel::SetDataDumperArgs(const ComputeGraphPtr &graph,
                                       const std::map<std::string, OpDescPtr> &variable_by_name) {
  if (dump_model_name_.empty()) {
    dump_model_name_ = name_;
  }
  data_dumper_.SetModelName(dump_model_name_);
  data_dumper_.SetModelId(GetDumpModelId());
  data_dumper_.SetOmName(om_name_);
  data_dumper_.SetComputeGraph(graph);
  data_dumper_.SetRefInfo(saved_task_addrs_);

  int32_t tmp_device_id = -1;
  GE_CHK_RT_RET(aclrtGetDevice(&tmp_device_id));
  data_dumper_.SetDeviceId(static_cast<uint32_t>(tmp_device_id));

  const auto get_var_addr = [&variable_by_name, this](const std::string &var_name) -> uintptr_t {
    const auto it = variable_by_name.find(var_name);
    if (it != variable_by_name.end()) {
      const auto output_sizes = ModelUtils::GetOutputSize(it->second);
      const auto output_addrs = ModelUtils::GetOutputAddrs(runtime_param_, it->second);
      if (output_sizes.empty() || output_addrs.empty()) {
        return 0U;
      }
      return static_cast<uintptr_t>(PtrToValue(output_addrs[0U]));
    }
    GELOGD("op: %s is null.", var_name.c_str());
    return 0U;
  };
  // prepare for inference, tran will be setted by SetGlobalStep
  if (ValueToPtr(static_cast<uint64_t>(global_step_addr_)) == nullptr) {
    void *malloc_mem = MallocDynamicMemory(sizeof(uint64_t));
    GE_ASSERT_NOTNULL(malloc_mem);
    global_step_addr_ = static_cast<uintptr_t>(PtrToValue(malloc_mem));
    GELOGI("Malloc global_step_addr: 0x%" PRIx64, global_step_addr_);
    global_step_size_ = sizeof(uint64_t);
    need_free_global_step_addr_ = true;
  }

  if (known_node_) {
    data_dumper_.SetLoopAddr(global_step_addr_, 0U, 0U);
    GELOGI("[Set][KnownNodeLoopAddr] succeed, global_step_addr: 0x%" PRIx64, global_step_addr_);
  } else {
    // set loop count addr
    GELOGI("[Set][LoopAddr] succeed, global_step_addr: 0x%" PRIx64, global_step_addr_);
    loop_per_iter_addr_ = get_var_addr(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
    loop_cond_addr_ = get_var_addr(NODE_NAME_FLOWCTRL_LOOP_COND);
    data_dumper_.SetLoopAddr(global_step_addr_,
                             loop_per_iter_addr_,
                             loop_cond_addr_);
  }

  return SUCCESS;
}

uint32_t DavinciModel::GetFlowctrlIndex(const uint32_t op_index) {
  const std::lock_guard<std::mutex> lk(flowctrl_op_index_internal_map_mutex_);
  ++flowctrl_op_index_internal_map_[op_index];
  return (flowctrl_op_index_internal_map_[op_index]) - 1U;
}

void DavinciModel::PushHcclStream(rtStream_t const hccl_stream) {
  const std::lock_guard<std::mutex> lk(all_hccl_stream_list_mutex_);
  all_hccl_stream_list_.push_back(hccl_stream);
}

void DavinciModel::SetHcclTaskStream(rtStream_t const hccl_stream) {
  const std::lock_guard<std::mutex> lk(hccl_task_stream_set_mutex_);
  (void) hccl_task_stream_set_.insert(ge::PtrToValue(hccl_stream));
}

void DavinciModel::SaveHcclFollowStream(const int64_t main_stream_id, rtStream_t stream) {
  const std::lock_guard<std::mutex> lk(capacity_of_stream_mutex_);
  main_follow_stream_mapping_[main_stream_id].emplace_back(stream);
}

Status DavinciModel::GetOrigInputInfo(const uint32_t index, OriginInputInfo &orig_input_info) const {
  return AippUtils::GetOrigInputInfo(orig_input_info_, index, orig_input_info);
}

Status DavinciModel::GetAllAippInputOutputDims(const uint32_t index, std::vector<InputOutputDims> &input_dims,
                                               std::vector<InputOutputDims> &output_dims) const {
  return AippUtils::GetAllAippInputOutputDims(aipp_dims_info_, index, input_dims, output_dims);
}

Status DavinciModel::InitL1DataDumperArgs() {
  if (ModelNeedDump()) {
    // malloc 2M for dump l1fusion op
    l1_fusion_addr_ = MallocDynamicMemory(kModelL1FusionOpMByteSize);
    GE_ASSERT_NOTNULL(l1_fusion_addr_);

    // send l1fusion dump addr to rts
    if (rtDumpAddrSet(rt_model_handle_, l1_fusion_addr_, kModelL1FusionOpMByteSize, kModelFlagOfL1Fusion) !=
        RT_ERROR_NONE) {
      // l1_fusion_addr_ will be free when DavinciModel destruct
      REPORT_INNER_ERR_MSG("E19999", "Call rtDumpAddrSet failed, model_id:%u", model_id_);
      GELOGE(FAILED, "[Call][RtDumpAddrSet] failed, model_id:%u", model_id_);
      return FAILED;
    }

    // set addr for l1 data dump
    data_dumper_.SetL1FusionAddr(static_cast<uintptr_t>(PtrToValue(l1_fusion_addr_)));
  }
  return SUCCESS;
}

void DavinciModel::UpdateOpIOAddrs(const uint32_t task_id, const uint32_t stream_id,
                                   const std::vector<uint64_t> &io_addrs) {
  if (!gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump)) {
    GELOGD("Disable exception dump, no need update.");
    return;
  }

  if (fixed_mem_base_ == mem_base_) {
    GELOGD("[Update][OpIOAddrs] No need to update op input output addr.");
    return;
  }

  OpDescInfo *const op_desc_info = exception_dumper_.MutableOpDescInfo(task_id, stream_id);
  if (op_desc_info == nullptr) {
    GELOGW("[Update][OpIOAddrs] Find op desc failed, task_id: %u, stream_id: %u.", task_id, stream_id);
    return;
  }
  const size_t input_size = op_desc_info->input_addrs.size();
  const size_t output_size = op_desc_info->output_addrs.size();
  if ((input_size + output_size) != io_addrs.size()) {
    GELOGW("[Update][OpIOAddrs] Op[%s] input size[%zu] and output size[%zu] is not equal to io addr size[%zu]",
           op_desc_info->op_name.c_str(), input_size, output_size, io_addrs.size());
    return;
  }

  std::vector<void *> input_addrs;
  std::vector<void *> output_addrs;
  for (size_t i = 0UL; i < io_addrs.size(); ++i) {
    const uint64_t addr = io_addrs[i];
    if (i < input_size) {
      input_addrs.emplace_back(ValueToPtr(addr));
    } else {
      output_addrs.emplace_back(ValueToPtr(addr));
    }
  }
  op_desc_info->input_addrs = input_addrs;
  op_desc_info->output_addrs = output_addrs;
  GELOGD("[Update][OpIOAddrs] Op [%s] update input output addr success.", op_desc_info->op_name.c_str());
}

///
/// @ingroup ge
/// @brief Get total useful size, in known subgraph, no need to allocate zero copy memory during initialization.
/// @param [in] total_useful_size: total mem size - zero copy size.
/// @return Status
///
Status DavinciModel::GetTotalMemSizeExcludeZeroCopy(int64_t &total_useful_size) {
  if (runtime_param_.mem_size < static_cast<uint64_t>(runtime_param_.zero_copy_size)) {
    REPORT_INNER_ERR_MSG("E19999", "total mem size[%" PRIu64 "] is less than zero copy size["
		      "%" PRId64 "] ", runtime_param_.mem_size, runtime_param_.zero_copy_size);
    GELOGE(FAILED, "[Check][TotalMemSizeExcludeZeroCopy] failed, total mem size[%" PRIu64 "] is less than "
      "zero copy size[%" PRId64 "]", runtime_param_.mem_size, runtime_param_.zero_copy_size);
    return FAILED;
  }
  total_useful_size = (static_cast<int64_t>(runtime_param_.mem_size) - runtime_param_.zero_copy_size);
  return SUCCESS;
}

Status DavinciModel::GetEventIdForBlockingAicpuOp(const OpDescPtr &op_desc, rtStream_t const stream,
                                                  uint32_t &event_id) {
  GELOGI("Get event id for aicpu blocking op:%s", op_desc->GetName().c_str());
  const auto it = stream_2_event_.find(stream);
  if (it != stream_2_event_.end()) {
    GE_CHK_RT_RET(rtGetEventID(it->second, &event_id));
  } else {
    rtEvent_t rt_event = nullptr;
    GE_CHK_RT_RET(rtEventCreateWithFlag(&rt_event, RT_EVENT_WITH_FLAG));
    const rtError_t rt_ret = rtGetEventID(rt_event, &event_id);
    if (rt_ret != RT_ERROR_NONE) {
      (void)rtEventDestroy(rt_event);
      REPORT_INNER_ERR_MSG("E19999", "Call rtGetEventID fail, ret: %d", rt_ret);
      GELOGE(ge::RT_FAILED, "Call rtGetEventID failed, ret: %d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    stream_2_event_[stream] = rt_event;
  }
  return SUCCESS;
}

Status DavinciModel::GetEventByStream(rtStream_t const stream, rtEvent_t &rt_event) {
  const auto it = stream_2_event_.find(stream);
  if (it == stream_2_event_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Get event failed");
    GELOGE(FAILED, "[Get][Event] Get event failed");
    return FAILED;
  }
  rt_event = it->second;
  return SUCCESS;
}

Status DavinciModel::AllocateQueueResource(const Node &node, const OpDescPtr &op_desc, const NamedAttrs &resource) {
  int32_t queue_idx = -1;
  uint32_t queue_id = std::numeric_limits<uint32_t>::max();
  GE_CHK_STATUS_RET_NOLOG(aicpu_resources_.AllocateQueueResource(op_desc, resource, queue_idx, queue_id));
  const auto &src_node = NodeUtils::GetInDataNodeByIndex(node, queue_idx);
  GE_CHECK_NOTNULL(src_node);
  if (!NodeUtils::IsConst(*src_node)) {
    GELOGE(PARAM_INVALID,
           "[%s] Queue id index is not a const (actually is %s), cannot update value",
           op_desc->GetName().c_str(), src_node->GetType().c_str());
    return PARAM_INVALID;
  }
  GE_CHK_STATUS_RET_NOLOG(UpdateOpInputValue(op_desc, queue_idx, queue_id));
  GELOGD("[%s] Input [%d] updated with queue id [%u]", op_desc->GetName().c_str(), queue_idx, queue_id);
  return SUCCESS;
}

Status DavinciModel::AllocateDvppChlResource(const OpDescPtr &op_desc) {
  rtStream_t stream = nullptr;
  const size_t stream_id = static_cast<size_t>(op_desc->GetStreamId());
  GE_CHK_STATUS_RET_NOLOG(GetOpStream(op_desc, stream_id, stream));
  int32_t rt_stream_id = kInvalidStream;
  (void)rtGetStreamId(stream, &rt_stream_id);
  GE_CHK_STATUS_RET_NOLOG(aicpu_resources_.AllocateChannelResource(op_desc, rt_stream_id));
  GELOGD("[%s] Channel resource allocation with stream id [%d] is complete",
         op_desc->GetName().c_str(), rt_stream_id);
  return SUCCESS;
}

Status DavinciModel::AllocateVdecChlResource(const OpDescPtr &op_desc) {
  rtStream_t stream = nullptr;
  const size_t stream_id = static_cast<size_t>(op_desc->GetStreamId());
  GE_CHK_STATUS_RET_NOLOG(GetOpStream(op_desc, stream_id, stream));
  int32_t rt_stream_id = kInvalidStream;
  (void)rtGetStreamId(stream, &rt_stream_id);
  GE_CHK_STATUS_RET_NOLOG(aicpu_resources_.AllocateVdecChannelResource(op_desc, rt_stream_id));
  GELOGD("[%s] Vdec channel resource allocation with stream id [%d] is complete",
         op_desc->GetName().c_str(), rt_stream_id);
  return SUCCESS;
}

Status DavinciModel::AllocateResource(const Node &node) {
  const auto &op_desc = node.GetOpDesc();
  if (!op_desc->HasAttr(ATTR_NAME_RESOURCE_LIST)) {
    return SUCCESS;
  }

  std::vector<NamedAttrs> resource_list;
  if (!AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_RESOURCE_LIST, resource_list)) {
    GELOGE(INTERNAL_ERROR, "Failed to get resource list, node name = %s", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (const auto &resource : resource_list) {
    std::string resource_type;
    if (!AttrUtils::GetStr(resource, "resource_type", resource_type)) {
      GELOGE(PARAM_INVALID, "[%s] Failed to get resource type", op_desc->GetName().c_str());
      return PARAM_INVALID;
    }

    Status ret;
    if (resource_type == AiCpuResources::ResourceTypeQueue()) {
      ret = AllocateQueueResource(node, op_desc, resource);
    } else if (resource_type == AiCpuResources::ResourceTypeChannel()) {
      ret = AllocateDvppChlResource(op_desc);
    } else if (resource_type == AiCpuResources::ResourceTypeVdecChannel()) {
      ret = AllocateVdecChlResource(op_desc);
    } else {
      GELOGE(UNSUPPORTED, "Unsupported resource type: %s", resource_type.c_str());
      return UNSUPPORTED;
    }
    if (ret != SUCCESS) {
      GELOGE(PARAM_INVALID, "[%s] Failed to Alloce %s", op_desc->GetName().c_str(), resource_type.c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status DavinciModel::UpdateOpInputValue(const OpDescPtr &op_desc, const int32_t input_index,
                                        const uint32_t queue_id) const {
  const auto &input_addresses = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc);
  if (static_cast<size_t>(input_index) >= input_addresses.size()) {
    GELOGE(PARAM_INVALID,
           "[%s] Invalid queue_id_idx: %d, number of inputs = %zu",
           op_desc->GetName().c_str(), input_index, input_addresses.size());
    return PARAM_INVALID;
  }
  const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(input_index));
  GE_CHECK_NOTNULL(input_desc);
  int64_t tensor_size = 0;
  (void)TensorUtils::GetSize(*input_desc, tensor_size);
  GE_CHK_RT_RET(rtMemcpy(ValueToPtr(input_addresses[static_cast<size_t>(input_index)]),
                         static_cast<uint64_t>(tensor_size),
                         &queue_id,
                         sizeof(queue_id),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief Get cur_dynamic_dims for all input.
/// @param [in] std::vector<vector<int64_t>> &tensor_input_dims: dims info of all user_inputs.
/// @param [out] std::vector<int32_t> &cur_dynamic_dims: real dims gather, where the index of -1.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
///
Status DavinciModel::GetCurDynamicDims(const std::vector<std::vector<int64_t>> &tensor_input_dims,
                                       std::vector<int32_t> &cur_dynamic_dims) const {
  // parse inputs.dims to std::vector<std::vector<uint64_t>> dynamic_dims
  std::vector<vector<int64_t>> user_real_input_dims;
  if (ParseInputsDims(tensor_input_dims, user_real_input_dims) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  const auto &user_input_dims = run_context_.user_input_dims;
  if (user_real_input_dims.size() != user_input_dims.size()) {
    const std::string reason = "The number of tensors " + std::to_string(user_input_dims.size()) +
                               " configured by the user is not equal to number of tensors " +
                               std::to_string(user_real_input_dims.size()) + " in the graph";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                               std::vector<const char_t *>({reason.c_str()}));
    GELOGE(INTERNAL_ERROR, "[Check][Param] The input count of user:%zu should be equal to the data count of graph:%zu",
           user_real_input_dims.size(), user_input_dims.size());
    return INTERNAL_ERROR;
  }

  for (size_t i = 0UL; i < user_input_dims.size(); ++i) {
    const auto &user_input_dim = user_input_dims.at(i).second;
    for (size_t j = 0U; j < user_input_dim.size(); ++j) {
      if (user_input_dim.at(j) < 0) {
        // 校验写在里面的原因
        // 当前分档只支持对静态输入使能私有格式，此时user_input_dims中静态data的shape为原始shape，而user_real_input_dims中shape为运行时shape。
        // 考虑到此校验为保证动态输入轴pos的安全编码，因此将校验移动到动态判断内层
        GE_ASSERT_TRUE(user_real_input_dims[i].size() == user_input_dim.size(),
                       "[Check][Param] The shape size:%zu of dynamic input:%s should equal to input shape:%zu",
                       user_real_input_dims[i].size(), user_input_dims[i].first.c_str(), user_input_dim.size());
        cur_dynamic_dims.emplace_back(static_cast<int32_t>(user_real_input_dims[i][j]));
      }
    }
  }
  if (logLevel_ <= DLOG_DEBUG) {
    GELOGD("Cur dynamic dims is %s.", ToString(cur_dynamic_dims).c_str());
  }

  for (const auto &dynamic_dim : run_context_.dynamic_shape_dims) {
    if (dynamic_dim == cur_dynamic_dims) {
      return SUCCESS;
    }
  }

  const std::string reason = "The input tensor dimension " + ToString(cur_dynamic_dims) + " is not in the dynamic dimension list configured by dynamic_dims";
  REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char_t *>({"reason"}),
                               std::vector<const char_t *>({reason.c_str()}));
  GELOGE(INTERNAL_ERROR, "[Check][Param] Cur dynamic dims is %s, does not exist in options.",
         ToString(cur_dynamic_dims).c_str());
  return INTERNAL_ERROR;
}

void DavinciModel::ParseInputsDimsForData(const std::vector<std::vector<int64_t>> &tensor_input_dims,
                                          std::vector<std::vector<int64_t>> &real_input_dims) const {
  GELOGD("Start parse input dims from data.");
  for (const auto &shape_dims : tensor_input_dims) {
    GELOGD("Input tensor dims is %s.", ToString(shape_dims).c_str());
    real_input_dims.emplace_back(shape_dims);
  }
}

Status DavinciModel::ParseInputsDimsForGetNextNoSinkAndData(const std::vector<NodePtr> &dynamic_nodes,
                                                            const std::vector<std::vector<int64_t>> &tensor_input_dims,
                                                            std::vector<std::vector<int64_t>> &real_input_dims) const {
  GELOGD("Start parse inputs dims when coexist data and getnext sink.");
  for (size_t i = 0UL; i < dynamic_nodes.size(); ++i) {
    const auto &op_desc = dynamic_nodes.at(i)->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    uint64_t index = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) fail", ATTR_NAME_INDEX.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(PARAM_INVALID, "[Get][Attr] %s from op:%s(%s) fail", ATTR_NAME_INDEX.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
    if (static_cast<size_t>(index) >= tensor_input_dims.size()) {
      REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) index:%" PRIu64 " >= param input_tensor.size:%zu, check invalid",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), index, tensor_input_dims.size());
      GELOGE(PARAM_INVALID, "[Check][Param] Node:%s(%s) index:%zu >= param input_tensor.size:%zu",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), index, tensor_input_dims.size());
      return PARAM_INVALID;
    }

    const auto &shape_dims = tensor_input_dims.at(static_cast<size_t>(index));
    GELOGI("Shape dims of %" PRIu64 " data is %s.", index, ToString(shape_dims).c_str());
    real_input_dims.emplace_back(shape_dims);
  }
  return SUCCESS;
}

Status DavinciModel::ParseInputsDims(const std::vector<std::vector<int64_t>> &tensor_input_dims,
                                     std::vector<std::vector<int64_t>> &real_input_dims) const {
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Start parse input dims of %zu input tensor.", tensor_input_dims.size());
  }
  if (run_context_.dynamic_node_type.empty()) {
    return SUCCESS;
  }

  const std::vector<NodePtr> &input_nodes = run_context_.data_nodes;
  const std::vector<NodePtr> &getnext_nodes = run_context_.getnext_nosink_nodes;
  GELOGD("Data nodes count = %zu, getnext nosink nodes count = %zu.", input_nodes.size(), getnext_nodes.size());
  if (run_context_.dynamic_node_type == DATA) {
    if (getnext_nodes.empty()) {
      // just data or data+getnext_sink
      ParseInputsDimsForData(tensor_input_dims, real_input_dims);
    } else {
      // data+getnext_nosink, but only need to get shape_dims of data
      if (ParseInputsDimsForGetNextNoSinkAndData(input_nodes, tensor_input_dims, real_input_dims) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Parse][Dims] from data failed, when data coexist with getnext nosink.");
        return PARAM_INVALID;
      }
    }
  } else {
    if (getnext_nodes.empty()) {
      // just getnext_sink or getnext_sink+data, need to get shape_dims from aicpu op
      GELOGI("Need to get dims from aicpu op: GETDYNAMICDIMS.");
      return SUCCESS;
    }

    if (input_nodes.empty()) {
      // just getnext_nosink
      ParseInputsDimsForData(tensor_input_dims, real_input_dims);
    } else {
      // getnext_nosink + data, but only need to get shape_dims of getnext_nosink
      if (ParseInputsDimsForGetNextNoSinkAndData(getnext_nodes, tensor_input_dims, real_input_dims) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Parse][Dims] from getnext nosink failed, when data coexist with getnext nosink");
        return PARAM_INVALID;
      }
    }
  }

  GELOGI("Parse %zu inputs dims success.", real_input_dims.size());
  return SUCCESS;
}

Status DavinciModel::InitQueueDataNodes(const std::vector<NodePtr> &queue_data_nodes,
                                        const uint32_t data_index,
                                        std::set<uint64_t> &input_outside_addrs) {
  if (queue_data_nodes.empty()) {
    GELOGD("No QueueData node");
    return SUCCESS;
  }

  if (queue_data_nodes.size() > 1U) {
    GELOGE(UNSUPPORTED, "Only supported single QueueData, actual number = %zu", queue_data_nodes.size());
    return UNSUPPORTED;
  }

  if (input_queue_attrs_.empty() && output_queue_attrs_.empty()) {
    GELOGE(UNSUPPORTED, "Only supported by LoadModelWithQueue");
    return UNSUPPORTED;
  }

  is_queue_data_[data_index] = true;
  auto &queue_data_node = queue_data_nodes[0U];
  std::string queue_name;
  (void) AttrUtils::GetStr(queue_data_node->GetOpDesc(), "queue_name", queue_name);
  if (queue_name.empty()) {
    GELOGE(PARAM_INVALID, "Queue name not set, node = %s", queue_data_node->GetName().c_str());
    return PARAM_INVALID;
  }
  GE_CHK_STATUS_RET_NOLOG(InitInputZeroCopy(queue_data_node->GetOpDesc(), data_index, input_outside_addrs));
  return SUCCESS;
}

bool DavinciModel::CheckModelNoInputAndOutput() const {
  return (input_queue_attrs_.empty() && output_queue_attrs_.empty());
}

bool DavinciModel::IsSendRecvOp(const std::string &op_type) {
  return (op_type == "SendMem") || (op_type == "RecvMem") || (op_type == SEND) || (op_type == RECV) || (op_type == SENDNOTIFY) ||
         (op_type == RECVNOTIFY);
}

std::map<uint32_t, ZeroCopyOffset> DavinciModel::FilterZeroCopyAddrs() const {
  std::map<uint32_t, ZeroCopyOffset> zero_copy_addrs;
  for (const auto &output : output_data_info_) {
    const size_t output_index = output.first;
    if (output_index < output_queue_attrs_.size()) {
      const uint32_t queue_id = output_queue_attrs_[output_index].queue_id;
      if (queue_id == UINT32_MAX) {
        continue;
      }
    }
    const auto output_addr = output.second.GetBasicAddr();
    const bool feed_by_zero_copy = copy_only_addrs_.Count(PtrToValue(output_addr)) == 0;
    if (feed_by_zero_copy) {
      (void)zero_copy_addrs.emplace(output.first, output.second);
      GELOGD("output [%zu] can perform zero-copy", output_index);
    } else {
      GELOGD("output [%zu] is copy-only", output_index);
    }
  }
  return zero_copy_addrs;
}

Status DavinciModel::CpuInputCopyProcess() {
  std::vector<uintptr_t> mbuf_list;
  std::vector<uintptr_t> data_addr_list;
  std::vector<uint64_t> length_list;
  std::vector<int32_t> input_fusion_offset_list;
  for (size_t i = 0UL; i < input_queue_attrs_.size(); ++i) {
    const auto iter = input_data_info_.find(static_cast<uint32_t>(i));
    if (iter == input_data_info_.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%zu can't find in input_data_info_ size:%zu in model_id:%u, check invalid",
                         i, input_data_info_.size(), model_id_);
      GELOGE(FAILED, "[Check][Param] Index:%zu can't find in input_data_info_ size:%zu in model_id:%u",
             i, input_data_info_.size(), model_id_);
      return INTERNAL_ERROR;
    }
    if (iter->second.GetDataInfo().empty() || iter->second.GetOutsideAddrs().empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%zu out_data_info in model:%u is empty, check invalid", i, model_id_);
      GELOGE(INTERNAL_ERROR, "[Check][Param] Index:%zu out_data_info in model:%u is empty, check invalid",
             i, model_id_);
      return INTERNAL_ERROR;
    }

    uint64_t data_ptr;
    const uint64_t data_size = static_cast<uint64_t>(iter->second.GetDataInfo().at(0U).first);
    // size 1: data info is data ptr; size 2: data info is RuntimeTensorDesc ptr
    if (iter->second.GetOutsideAddrs().size() == 1UL) {
      GELOGD("Input id:%u is data partition", iter->first);
      data_ptr = iter->second.GetDataInfo().at(0U).second;
    } else {
      GELOGD("Input id:%u is tensor desc partition.", iter->first);
      RuntimeTensorDesc *const tensor_desc =
          PtrToPtr<void, RuntimeTensorDesc>(ValueToPtr(iter->second.GetDataInfo().at(0U).second));
      GE_CHECK_NOTNULL(tensor_desc);
      GE_CHK_RT_RET(rtMemcpy(&data_ptr, sizeof(data_ptr), tensor_desc, sizeof(data_ptr), RT_MEMCPY_DEVICE_TO_HOST));
    }
    if (copy_only_addrs_.Count(data_ptr) == 0) {
      continue;
    }

    if (static_cast<uint64_t>(iter->first) >= input_mbuf_list_.size()) {
      GELOGE(FAILED, "[Check][Param] Data index:%u shold in range of mbuf size:%zu",
             iter->first, input_mbuf_list_.size());
      return INTERNAL_ERROR;
    }
    mbuf_list.emplace_back(input_mbuf_list_.at(static_cast<size_t>(iter->first)));
    data_addr_list.emplace_back(static_cast<uintptr_t>(data_ptr));
    length_list.emplace_back(data_size);
    auto input_fusion_offset = input_fusion_offsets_.at(static_cast<size_t>(iter->first));
    input_fusion_offset_list.emplace_back(input_fusion_offset);
    GELOGI("Copy input process task: index:%zu, src muff addr:0x%" PRIx64 ", "
           "dst addr:0x%" PRIx64 ", data size:%" PRIu64 ", input fusion offset:%d.",
           i, static_cast<uint64_t>(input_mbuf_list_.at(iter->first)), data_ptr, data_size, input_fusion_offset);
  }
  if (!length_list.empty()) {
    const auto input_memcp_task = MakeShared<CpuTaskProcessInputsMemCopy>(rt_entry_stream_);
    GE_CHECK_NOTNULL(input_memcp_task);
    GE_CHK_STATUS_RET(input_memcp_task->Init(mbuf_list, data_addr_list, length_list, input_fusion_offset_list),
                      "CpuTaskProcessInputsMemCopy task init failed.");

    cpu_task_list_.push_back(input_memcp_task);
  } else {
    GELOGI("All inputs support zero copy");
  }
  return SUCCESS;
}

bool DavinciModel::IsFeatureBaseRefreshable() const {
  return feature_base_refreshable_;
}

Status DavinciModel::UpdateHbmFmMemBases(const uintptr_t mem_base, const size_t size,
                                         size_t &used_size, const bool is_init) {
  std::vector<uint8_t *> hbm_fm_mem_bases;
  size_t mem_size = 0U;
  uintptr_t fm_mem_base = mem_base;
  for (const auto &mem_info : runtime_param_.fm_memory_infos) {
    mem_size += static_cast<size_t>(mem_info.memory_size);
    GE_ASSERT_TRUE(size >= mem_size, "size:%zu less than mem size:%zu", size, mem_size);
    hbm_fm_mem_bases.push_back(PtrToPtr<void, uint8_t>(reinterpret_cast<void *>(fm_mem_base)));
    fm_mem_base += static_cast<uintptr_t>(mem_info.memory_size);
  }
  used_size = mem_size; // 实际使用的长度

  GE_ASSERT_SUCCESS(UpdateHbmFmMemBases(hbm_fm_mem_bases, used_size, is_init));
  return SUCCESS;
}

Status DavinciModel::UpdateHbmFmMemBases(const vector<uint8_t *> &hbm_fm_mem_bases,
                                         const size_t size, const bool is_init) {
  // fixed memory可能会导致fm_memory_infos为空，hbm_fm_mem_bases 为空
  if (hbm_fm_mem_bases.empty()) {
    GELOGI("No need Update feature memory base.");
    return SUCCESS;
  }

  if (is_inner_mem_base_) {
    FreeInnerFeatureMapMem();
    is_inner_mem_base_ = false;
  }

  GE_ASSERT_EQ(hbm_fm_mem_bases.size(), runtime_param_.fm_memory_infos.size());
  for (size_t i = 0U; i < hbm_fm_mem_bases.size(); ++i) {
    auto &mem_info = runtime_param_.fm_memory_infos[i];
    mem_info.memory_base = hbm_fm_mem_bases[i];

    // init流程中调用，多段场景，已分配物理地址，需要保存物理地址和逻辑地址的对应关系
    // task GetAddr时可以通过逻辑地址获取到物理地址并返回给model args manager
    // model args manager使用物理地址来匹配到对应的fm allocation段
    if (is_init) {
      (void) runtime_param_.sorted_memory_infos.insert(mem_info);
    }

    GELOGI("Update one sub feature map memory info with details: [%s]", mem_info.ToString().c_str());
  }
  // 兼容静态图feature map的可刷新流程
  mem_base_ = reinterpret_cast<uintptr_t>(PtrToPtr<uint8_t, void>(hbm_fm_mem_bases[0U]));
  mem_base_size_ = (size == SIZE_MAX) ? mem_base_size_ : size + fixed_mem_size_;
  GE_ASSERT_SUCCESS(UpdateRuntimeParamBase());
  return SUCCESS;
}

Status DavinciModel::UpdateRuntimeParamBase() {
 // 该变量未使用
  runtime_param_.mem_base = mem_base_;

  // fixed memory 可能回导致fm_memory_infos为空
  if (runtime_param_.fm_memory_infos.empty()) {
    return SUCCESS;
  }

  // 静态shape场景下， fm_memory_infos只有一个元素, 同步更新他
  runtime_param_.fm_memory_infos[0U].memory_base = PtrToPtr<void, uint8_t>(reinterpret_cast<void *>(mem_base_));
  return SUCCESS;
}

bool DavinciModel::IsKnownNode() const {
  return known_node_;
}

void DavinciModel::LogModelDevMemInfo() const {
  GEEVENT("model_metrics:name=%s, alloc_dev_mem=%" PRIu64 " B, shared_dev_mem=%" PRIu64
          " B, device_id=%u, rts_model_id=%u",
          runtime_param_.graph_name.c_str(), dev_mem_statistic_.alloc_size,
          dev_mem_statistic_.shared_size, device_id_, runtime_model_id_);
}

bool DavinciModel::CopyOnlyAddrCheck() const {
  GELOGI("The num of copy only addr is %zu", copy_only_addrs_.copy_only_addrs.size());
  bool check_ret = true;
  for (const auto &copy_only_addr : copy_only_addrs_.copy_only_addrs) {
    const auto check_input_func = [this, &copy_only_addr]() -> bool {
      for (const auto &iter : input_indexes_to_copy_info_) {
        if (static_cast<size_t>(iter.second.id) >= logical_mem_allocations_.size()) {
          continue;
        }
        const uint64_t logical_input_addr =
            logical_mem_allocations_[static_cast<size_t>(iter.second.id)].logical_addr + iter.second.offset;
        if ((copy_only_addr >= logical_input_addr) && (copy_only_addr < (logical_input_addr + iter.second.data_size))) {
          GELOGI("[CopyOnlyAddrCheck][input] copy only addr:0x%" PRIx64 " match to copy info, input index:%u, %s",
                 copy_only_addr, iter.first, iter.second.ToString().c_str());
          return true;
        }
      }
      return false;
    };
    if (check_input_func()) {
      continue;
    }

    const auto &check_output_func = [this, &copy_only_addr]() -> bool {
      for (const auto &iter : output_indexes_to_copy_info_) {
        if (static_cast<size_t>(iter.second.id) >= logical_mem_allocations_.size()) {
          continue;
        }
        const uint64_t logical_output_addr =
            logical_mem_allocations_[static_cast<size_t>(iter.second.id)].logical_addr + iter.second.offset;
        if ((copy_only_addr >= logical_output_addr)
            && (copy_only_addr < (logical_output_addr + iter.second.data_size))) {
          GELOGI("[CopyOnlyAddrCheck][output] copy only addr:0x%" PRIx64 " match to copy info, output index:%u, %s",
                 copy_only_addr, iter.first, iter.second.ToString().c_str());
          return true;
        }
      }
      return false;
    };
    if (check_output_func()) {
      continue;
    }

    const auto &check_allocations_func = [&]() -> void {
      for (const auto &iter : logical_mem_allocations_) {
        if ((copy_only_addr >= iter.logical_addr) && (copy_only_addr < (iter.logical_addr + iter.data_size))) {
          GELOGW("[CopyOnlyAddrCheck][allocations] copy only addr:0x%" PRIx64 " match to logical mem allocations table, "
            "%s", copy_only_addr, iter.ToString().c_str());
          return;
        }
      }
      GELOGW(
          "[CopyOnlyAddrCheck][null] copy only addr:0x%" PRIx64 " can not find in copy info and logical mem allocations table!",
          copy_only_addr);
      return;
    };
    check_allocations_func();
    check_ret = false;
  }
  return check_ret;
}

uint32_t DavinciModel::GetStreamFlagById(uint32_t stream_id) const {
  if (stream_flag_list_.size() == 1U) {
    return stream_flag_list_[0U];
  } else if (stream_flag_list_.size() > stream_id) {
    return stream_flag_list_[stream_id];
  } else {
    GELOGE(INTERNAL_ERROR, "Invalid stream id %u.", stream_id);
    return 0U;
  }
}

bool DavinciModel::NeedUpdateCoreCountWithOpDesc(const NodePtr &node, fe::PlatFormInfos &platform_infos, std::string &addr_key_out) const {
  const std::string soc_info = "SoCInfo";
  std::map<std::string, std::string> res;
  GE_ASSERT_TRUE(platform_infos.GetPlatformResWithLock(soc_info, res));

  std::string aic_cnt_value;
  std::string vec_core_cnt_value;
  std::string ai_core_cnt_global = res["ai_core_cnt"];
  std::string vector_core_cnt_global = res["vector_core_cnt"];
  bool need_update = false;

  if (ge::AttrUtils::GetStr(node->GetOpDesc(), "_op_aicore_num", aic_cnt_value)) {
    need_update = true;
  }
  if (ge::AttrUtils::GetStr(node->GetOpDesc(), "_op_vectorcore_num", vec_core_cnt_value)) {
    need_update = true;
  }

  if (need_update) {
    addr_key_out = aic_cnt_value + "|" + vec_core_cnt_value;
    GELOGD("Need to update, build addr_key with op desc, addr_key is [%s]", addr_key_out.c_str());
  } else {
    addr_key_out = ai_core_cnt_global + "|" + vector_core_cnt_global;
    GELOGD("Not need to update, build addr_key with existing platform infos, addr_key is [%s]", addr_key_out.c_str());
  }

  return need_update;
}

bool DavinciModel::UpdateCoreCountWithOpDesc(const NodePtr &node, fe::PlatFormInfos &platform_infos) const {
  const std::string soc_info = "SoCInfo";
  std::map<std::string, std::string> res;
  GE_ASSERT_TRUE(platform_infos.GetPlatformResWithLock(soc_info, res));

  std::string aic_cnt_value;
  std::string vec_core_cnt_value;
  std::string ai_core_cnt_global = res["ai_core_cnt"];
  std::string vector_core_cnt_global = res["vector_core_cnt"];

  if (ge::AttrUtils::GetStr(node->GetOpDesc(), "_op_aicore_num", aic_cnt_value)) {
    res["ai_core_cnt"] = aic_cnt_value;
    res["cube_core_cnt"] = aic_cnt_value;
  }
  if (ge::AttrUtils::GetStr(node->GetOpDesc(), "_op_vectorcore_num", vec_core_cnt_value)) {
    res["vector_core_cnt"] = vec_core_cnt_value;
  }

  platform_infos.SetPlatformResWithLock(soc_info, res);
  GELOGD("Update core count with op desc to: ai_core_cnt[%s], vector_core_cnt[%s]", aic_cnt_value.c_str(), vec_core_cnt_value.c_str());
  return true;
}

Status DavinciModel::UpdatePlatformInfos(const NodePtr &node, fe::PlatFormInfos &platform_infos) const {
  int32_t device_id = -1;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));

  fe::PlatFormInfos platform_infos_bak;
  auto ret = fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(
      static_cast<uint32_t>(device_id), platform_infos_bak, true);
  GE_ASSERT_TRUE(ret == 0, "Get runtime platformInfos by device failed, deviceId = %d", device_id);

  GE_ASSERT_TRUE(UpdateCoreCountWithOpDesc(node, platform_infos_bak));
  platform_infos = platform_infos_bak; // 更新引用
  return SUCCESS;
}

void* DavinciModel::AllocPlatformInfosMem(size_t total_size, bool need_update_op_desc, bool is_custom) {
  if (need_update_op_desc) {
    // 算子级控核场景：使用模型级的PlatFormInfo内存
    return MallocDynamicMemory(total_size);
  }

  // 全局控核场景：使用进程级的PlatFormInfo内存
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  const std::string &memory_key = is_custom ? kCustPlatformInfoMemoryKey : kPlatformInfoMemoryKey;
  const std::string &purpose = is_custom ? kCustPlatformInfoPurpose : kPlatformInfoPurpose;
  return mem_instance.MallocMemory(purpose, memory_key, total_size, GetDeviceId());
}

Status DavinciModel::SerializeAndCopyToDevice(fe::PlatFormInfos &platform_infos, void *dev_addr, size_t copy_size,
                                              size_t total_size) const {
  const std::string serialized_str = platform_infos.SaveToBuffer();
  const size_t serialized_size = serialized_str.length() + 1UL; // '\0'

  std::unique_ptr<uint8_t[]> host_addr = ge::MakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(host_addr);
  GE_ASSERT_EOK(memset_s(host_addr.get(), total_size, 0, total_size), "Failed to memset host context buffer.");

  GE_ASSERT_EOK(memcpy_s(&host_addr[sizeof(fe::PlatFormInfos)], serialized_size, serialized_str.data(), serialized_size));

  PlatformInfosLaunchArgs *args = reinterpret_cast<PlatformInfosLaunchArgs *>(&host_addr[copy_size]);
  GE_ASSERT_NOTNULL(args);
  args->proto_ptr = PtrToValue(dev_addr) + sizeof(fe::PlatFormInfos);
  args->proto_len = serialized_str.length();
  args->platform_infos_addr = PtrToValue(dev_addr);

  GE_CHK_RT_RET(rtMemcpy(dev_addr, total_size, host_addr.get(), total_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}


//  |platform_infos|serialized_proto|launch_args|
// 如果是通过 LaunchPlatformInfos 接口过来的，plat_form_info_ptr默认为platform_infos_，如果配置了算子级核数，platform_infos为装载了核数信息的temp_info
// 如果是通过 LoadCustPlatformInfos 接口过来的，plat_form_info_ptr为platform_infos_，因此本接口内要判断是否配置了算子级核数
Status DavinciModel::LoadPlatformInfos(const fe::PlatFormInfos *const plat_form_info_ptr, size_t &copy_size, void *&dev_addr,
                                       bool is_custom, const NodePtr &node) {
  fe::PlatFormInfos platform_infos_to_load = *plat_form_info_ptr;

  std::string addr_key;
  bool need_update_with_op_desc = NeedUpdateCoreCountWithOpDesc(node, platform_infos_to_load, addr_key);
  if (is_custom) {
    auto it = cust_platform_infos_addr_.find(addr_key);
    if (it != cust_platform_infos_addr_.end()) {
      dev_addr = it->second;
      return SUCCESS;
    }

    if (need_update_with_op_desc) {
      GE_CHK_STATUS_RET(UpdatePlatformInfos(node, platform_infos_to_load));
    }

    auto it_to_launch = cust_platform_infos_addr_to_launch_.find(addr_key);
    if (it_to_launch != cust_platform_infos_addr_to_launch_.end()) {
      dev_addr = it_to_launch->second.first;
      return SUCCESS;
    }
  }

  const std::string serialized_str = platform_infos_to_load.SaveToBuffer();
  const size_t serialized_size = serialized_str.length() + 1UL; // '\0'
  copy_size = sizeof(fe::PlatFormInfos) + serialized_size;
  const size_t launched_size = sizeof(PlatformInfosLaunchArgs);
  const size_t total_size = copy_size + launched_size;

  // 分配内存
  dev_addr = AllocPlatformInfosMem(total_size, need_update_with_op_desc, is_custom);
  GE_ASSERT_NOTNULL(dev_addr);

  // 序列化与拷贝
  GE_CHK_STATUS_RET(SerializeAndCopyToDevice(platform_infos_to_load, dev_addr, copy_size, total_size));

  GELOGD("load platform_infos_addr = %p", dev_addr);

  // 更新缓存
  if (is_custom) {
    cust_platform_infos_addr_to_launch_[addr_key] = std::make_pair(dev_addr, copy_size);
  }
  return SUCCESS;
}

Status DavinciModel::LoadCustPlatformInfos(void *&cust_platform_infos_addr, const NodePtr &node) {
  size_t copy_size = 0U;
  GE_CHK_STATUS_RET(LoadPlatformInfos(&platform_infos_, copy_size, cust_platform_infos_addr, true, node),
                    "Failed to load platform infos");
  GELOGI("Succeed to load cust platform infos.");
  return SUCCESS;
}

Status DavinciModel::LaunchCustPlatformInfos() {
  if (cust_platform_infos_addr_to_launch_.empty()) {
    GELOGD("No custom platform infos need to launch.");
    return SUCCESS;
  }
  std::string platform_so_path = GetPlatformSoPath();
  bool is_platform_so_exist = CheckPlatformSoExist(platform_so_path);
  if (is_platform_so_exist) {
    return LaunchFromPlatformSo(platform_so_path);
  }
  return LaunchFromOpMasterSo();
}

Status DavinciModel::LaunchFromPlatformSo(const std::string &platform_so_path) {
  GELOGI("Launch cust platform infos from platform so.");
  std::unique_ptr<char_t []> buf = nullptr;
  uint32_t buf_len = 0UL;
  GE_ASSERT_SUCCESS(ReadPlatformSo(platform_so_path, buf, buf_len));
  GE_ASSERT_NOTNULL(buf);
  auto kernel_handles_manager = GetKernelHandlesManager(KernelHandleType::kCustAicpu);
  GE_ASSERT_NOTNULL(kernel_handles_manager);
  KernelRegisterInfo register_info;
  CustAicpuRegisterInfo tiling_device_register_info;
  tiling_device_register_info.cust_aicpu_kernel_bin = MakeShared<OpKernelBin>("platform_so_bin", std::vector<char_t>(buf.get(), buf.get() + buf_len));
  register_info = tiling_device_register_info;
  const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
  auto bin_handle = kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  auto func_handle = KernelHandleUtils::GetCustAicpuFuncHandle(bin_handle,
      "PlatformInfos", kAicpuCustLoadPlatformInfo.c_str());
  GE_ASSERT_NOTNULL(func_handle);
  for (const auto &it : cust_platform_infos_addr_to_launch_) {
    const std::string &addr_key = it.first;
    auto cust_platform_infos_addr = it.second.first;
    GELOGI("Launch custom platform infos: addr_key[%s], addr[%p].", addr_key.c_str(), cust_platform_infos_addr);
    rtStream_t stream = nullptr;
    const std::function<void()> callback = [&stream]() {
      if (stream != nullptr) {
        GE_CHK_RT(rtStreamDestroy(stream));
      }
    };
    GE_MAKE_GUARD(release, callback);
    GE_CHK_RT_RET(rtStreamCreate(&stream, 0));
    LaunchKernelParam launch_param;
    launch_param.block_dim = 1U;
    launch_param.stream = stream;
    launch_param.is_host_args = true;
    LaunchKernelConfig launch_config;
    launch_param.launch_config = launch_config;
    LoadCustPlatformInfosArgs load_args = {};
    load_args.args = PtrToValue(cust_platform_infos_addr) + it.second.second;
    load_args.args_size = static_cast<uint64_t>(sizeof(PlatformInfosLaunchArgs));
    GELOGD("Load cust platform infos args is %lu, args size is %lu", load_args.args, load_args.args_size);
    launch_param.args = static_cast<void *>(&load_args);
    launch_param.args_size = static_cast<uint32_t>(sizeof(LoadCustPlatformInfosArgs));
    GE_ASSERT_RT_OK(KernelHandleUtils::LaunchKernel(func_handle, launch_param));
    GELOGI("Launch custom platform infos: so_path[%s], kernel_name[%s], stream[%" PRIu64 "].",
           platform_so_path.c_str(), kAicpuCustLoadPlatformInfo.c_str(), PtrToValue(stream));
    cust_platform_infos_addr_[addr_key] = cust_platform_infos_addr;
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GELOGI("Succeed to launch custom platform infos task.");
  }
  cust_platform_infos_addr_to_launch_.clear();
  return SUCCESS;
}

Status DavinciModel::LaunchFromOpMasterSo() {
  GELOGI("Launch cust platform infos from op master so.");
  rtBinHandle platform_bin_handle = ModelManager::GetInstance().GetPlatformBinHandle();
  GE_ASSERT_NOTNULL(platform_bin_handle, "Failed to get platform infos bin handle");
  rtFuncHandle platform_func_handle = KernelHandleUtils::GetCustAicpuFuncHandle(platform_bin_handle,
      "CustPlatformInfo", kAicpuCustLoadPlatformInfo);
  GE_ASSERT_NOTNULL(platform_func_handle);
  for (const auto &it : cust_platform_infos_addr_to_launch_) {
    const std::string &addr_key = it.first;
    auto cust_platform_infos_addr = it.second.first;
    GELOGI("Launch custom platform infos: addr_key[%s], addr[%p].", addr_key.c_str(), cust_platform_infos_addr);
    rtStream_t stream = nullptr;
    const std::function<void()> callback = [&stream]() {
      if (stream != nullptr) {
        GE_CHK_RT(rtStreamDestroy(stream));
      }
    };
    GE_MAKE_GUARD(release, callback);
    GE_CHK_RT_RET(rtStreamCreate(&stream, 0));
    LoadCustPlatformInfosArgs load_args = {};
    load_args.args = PtrToValue(cust_platform_infos_addr) + it.second.second;
    load_args.args_size = static_cast<uint64_t>(sizeof(PlatformInfosLaunchArgs));
    GELOGD("Load cust platform infos args is %lu, args size is %lu", load_args.args, load_args.args_size);
    LaunchKernelParam launch_kernel_param;
    launch_kernel_param.args = static_cast<void *>(&load_args);
    launch_kernel_param.args_size = static_cast<uint32_t>(sizeof(LoadCustPlatformInfosArgs));
    launch_kernel_param.block_dim = 1U;
    launch_kernel_param.stream = stream;
    launch_kernel_param.is_host_args = true;
    GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(platform_func_handle, launch_kernel_param));
    cust_platform_infos_addr_[addr_key] = cust_platform_infos_addr;
    GELOGI("Launch custom platform infos: kernel_name[%s], stream[%" PRIu64 "].",
        kAicpuCustLoadPlatformInfo.c_str(), PtrToValue(stream));
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
    GELOGI("Succeed to launch custom platform infos task.");
  }
  cust_platform_infos_addr_to_launch_.clear();
  return SUCCESS;
}

Status DavinciModel::LaunchPlatformInfos(void *&platform_infos_addr, const NodePtr &node) {
  int32_t device_id = -1;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));
  fe::PlatFormInfos platform_infos_bak;
  GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(
                 static_cast<uint32_t>(device_id), platform_infos_bak, true) == 0,
                 "Get runtime platformInfos by device failed, deviceId = %d", device_id);

  std::string addr_key;
  bool need_update_with_op_desc = NeedUpdateCoreCountWithOpDesc(node, platform_infos_bak, addr_key);

  const auto &it = platform_infos_addr_.find(addr_key);
  if (it != platform_infos_addr_.cend()) {
    GELOGI("Find exists platform infos address[%p] for addr_key [%s]", it->second, addr_key.c_str());
    platform_infos_addr = it->second;
    return SUCCESS;
  }

  size_t copy_size = 0U;
  void *dev_addr = nullptr;
  // platform_infos_在初始化过程中被rts/device级/global级/session级核数配置刷新过
  fe::PlatFormInfos *plat_form_info_ptr = &platform_infos_;
  if (need_update_with_op_desc) {
    GE_ASSERT_TRUE(UpdateCoreCountWithOpDesc(node, platform_infos_bak));
    plat_form_info_ptr = &platform_infos_bak;
  }
  GE_CHK_STATUS_RET(LoadPlatformInfos(plat_form_info_ptr, copy_size, dev_addr, false, node), "Failed to load platform infos");
  // rtcpulaunch
  const size_t launched_size = sizeof(PlatformInfosLaunchArgs);
  void *const args_addr = ValueToPtr(PtrToValue(dev_addr) + copy_size);
  GE_ASSERT_RT_OK(rtAicpuInfoLoad(args_addr, launched_size), "Init device platform infos failed.");

  platform_infos_addr_[addr_key] = dev_addr;
  platform_infos_addr = dev_addr;
  GELOGI("Succeed to launch platform infos load task.");
  return SUCCESS;
}

void DavinciModel::CollectHcomRelatedStreams(const OpDescPtr &op_desc) {
  const uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
  (void)hcom_streams_.emplace(stream_id);
  GELOGI("hcom stream id: %u.", stream_id);
  const auto attach_stream_ids = op_desc->GetAttachedStreamIds();
  for (const auto attach_stream_id : attach_stream_ids) {
    if (attach_stream_id < 0) {
      continue;
    }
    (void)hcom_attach_streams_.emplace(static_cast<uint32_t>(attach_stream_id));
    GELOGI("hcom valid attached stream id: %" PRId64 ".", attach_stream_id);
  }
}

bool DavinciModel::GetPhysicalMemoryRefreshable() const {
  return support_extend_memory_full_;
}

Status DavinciModel::MallocPhysicalMemory() {
  // 静态shape和动态shape都启用了扩展模式，加载时已经分配好内存，首次执行不要再分配
  if ((!is_first_execute_) && support_extend_memory_full_) {
    auto mem_allocator =
        SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(session_id_, GetDeviceId());
    GE_ASSERT_NOTNULL(mem_allocator);
    return mem_allocator->MallocPhysicalMemory(kPurpose, active_memorys_);
  }
  return SUCCESS;
}

uint32_t DavinciModel::GetGraphId() const {
  return known_node_ ? runtime_param_.root_graph_id : runtime_param_.graph_id;
}

std::string DavinciModel::GetWeightsMemId() const {
  return to_string(runtime_param_.session_id) + "_" + to_string(GetGraphId()) + "_" + runtime_param_.graph_name;
}

Status DavinciModel::LaunchEventForHcclGroupOrderedStream(rtStream_t const stream) {
  if (!is_async_mode_) {
    GELOGI("model[%u] is sync execute.", model_id_);
    return SUCCESS;
  }

  rtError_t rt_ret = RT_ERROR_NONE;
  for (size_t i = 0U ; i < hccl_group_ordered_event_list_.size(); i++) {
    // 执行流下发event record
    rt_ret = rtEventRecord(hccl_group_ordered_event_list_[i], stream);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtEventRecord failed, ret:%d", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtEventRecord] failed, ret:%d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    // kfc流下发event wait
    rt_ret = rtStreamWaitEventWithTimeout(hccl_group_ordered_stream_list_[i], hccl_group_ordered_event_list_[i], 0U);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtStreamWaitEvent failed, ret:%d", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtStreamWaitEvent] failed, ret:%d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    GELOGI("model[%u] launch hccl group ordered event, record event stream:%p, wait event stream:%p",
      model_id_, stream, hccl_group_ordered_stream_list_[i]);
  }

  return SUCCESS;
}

Status DavinciModel::ParseHostInputIndexOption(const size_t input_num) {
  copy_host_input_indexes_.clear();
  string copy_host_inputs;
  (void)ge::GetContext().GetOption(OPTION_EXEC_HOST_INPUT_INDEXES, copy_host_inputs);
  if (copy_host_inputs.empty()) {
    GELOGI("host input indexes is empty.");
    return SUCCESS;
  }

  // copy host input indexes: ids(1;2;4;5)
  std::vector<std::string> copy_host_input_vec = StringUtils::Split(copy_host_inputs, ';');
  for (auto &input : copy_host_input_vec) {
    int32_t input_index;
    GE_ASSERT_SUCCESS(ConvertToInt32(input, input_index));
    GE_ASSERT_TRUE((input_index >= 0) && static_cast<uint32_t>(input_index) < input_num,
      "host input index:%d no less than input num:%zu", input_index, input_num);
    GELOGI("model:%u, host input index:%d", model_id_, input_index);
    (void)copy_host_input_indexes_.insert(input_index);
  }

  return SUCCESS;
}
}  // namespace ge
