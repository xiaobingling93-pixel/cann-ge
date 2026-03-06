/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/executor/ge_executor.h"
#include <ctime>
#include "framework/common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_properties.h"
#include "common/dump/dump_manager.h"
#include "graph/execute/graph_executor.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "graph/manager/host_mem_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/ge_context.h"
#include "single_op/single_op_manager.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/opsproto_manager.h"
#include "base/registry/opp_package_utils.h"
#include "register/op_lib_register_impl.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "runtime/base.h"
#include "runtime/config.h"
#include "common/profiling/command_handle.h"
#include "common/profiling_definitions.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "common/plugin/runtime_plugin_loader.h"
#include "runtime/subscriber/global_profiler.h"
#include "common/ge_model_inout_types.h"
#include "common/error_tracking/error_tracking.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_callback.h"

namespace {
constexpr size_t kDynamicBatchSizeVecSize = 1U;
constexpr size_t kStaticBatchInfoSize = 1U;
constexpr size_t kDynamicImageSizeVecSize = 2U;
constexpr size_t kDynamicImageSizeInputSize = 2U;
const std::string kBatchLabel = "Batch_";
constexpr uint32_t kDefaultOfflinePlacement = 1U;

void GetGeTensorDescFromDomiInfo(std::vector<ge::TensorDesc> &ge_descs,
                                 const std::vector<ge::InputOutputDescInfo> &domi_descs,
                                 const std::vector<uint32_t> &formats_vec) {
  size_t idx = 0U;
  for (const auto &desc_item : domi_descs) {
    ge::TensorDesc ge_desc;
    ge_desc.SetName(desc_item.name.c_str());
    ge_desc.SetDataType(static_cast<ge::DataType>(desc_item.data_type));
    ge_desc.SetFormat(static_cast<ge::Format>(formats_vec[idx]));
    std::vector<int64_t> shape_dims;
    for (const auto &dim : desc_item.shape_info.dims) {
      shape_dims.push_back(dim);
    }
    const ge::Shape ge_shape(shape_dims);
    ge_desc.SetShape(ge_shape);
    ge_desc.SetSize(static_cast<int64_t>(desc_item.size));
    (void)ge_desc.SetShapeRange(desc_item.shape_info.shape_ranges);
    ge_descs.emplace_back(ge_desc);
    ++idx;
  }
}

void GetDomiInputData(const ge::RunModelData &input_data, ge::InputData &inputs) {
  inputs.index = input_data.index;
  inputs.model_id = input_data.modelId;
  inputs.timestamp = input_data.timestamp;
  inputs.timeout = input_data.timeout;
  inputs.request_id = input_data.request_id;
  for (const auto &data_item : input_data.blobs) {
    inputs.blobs.emplace_back(
        data_item.data, data_item.length, data_item.isDataSupportMemShare, kDefaultOfflinePlacement);
  }
}

void GetDomiOutputData(const ge::RunModelData &output_data, ge::OutputData &outputs) {
  outputs.index = output_data.index;
  outputs.model_id = output_data.modelId;
  for (const auto &data_item : output_data.blobs) {
    outputs.blobs.emplace_back(
        data_item.data, data_item.length, data_item.isDataSupportMemShare, data_item.placement);
  }
}

void SetDynamicInputDataFlag(const ge::RunModelData &input_data, const std::vector<std::vector<int64_t>> &batch_info,
                             ge::InputData &inputs) {
  inputs.is_dynamic_batch = true;
  std::string batch_label;
  size_t match_idx = 0U;
  for (size_t i = 0U; i < batch_info.size(); ++i) {
    // dynamic_dims
    if ((input_data.dynamic_dims.size() != 0U) && (input_data.dynamic_dims.size() <= batch_info[i].size())) {
      bool is_match = true;
      for (size_t j = 0U; j < static_cast<size_t>(input_data.dynamic_dims.size()); ++j) {
        if (static_cast<uint64_t>(batch_info[i][j]) != input_data.dynamic_dims[j]) {
          is_match = false;
          break;
        }
      }
      if (is_match) {
        match_idx = i;
        break;
      }
    } else {
      // dynamic_batch_size
      if ((batch_info[i].size() == kDynamicBatchSizeVecSize) &&
          (batch_info[i][0U] == static_cast<int64_t>(input_data.dynamic_batch_size))) {
        match_idx = i;
        break;
      }
      // dynamic_image_size
      if ((batch_info[i].size() == kDynamicImageSizeVecSize) &&
          (batch_info[i][0U] == static_cast<int64_t>(input_data.dynamic_image_height)) &&
          (batch_info[i][1U] == static_cast<int64_t>(input_data.dynamic_image_width))) {
        match_idx = i;
        break;
      }
    }
  }
  batch_label = kBatchLabel + std::to_string(match_idx);
  inputs.batch_label = batch_label;
  GELOGI("current batch label:%s", batch_label.c_str());
}

bool IsDynamicBatchSizeMatchModel(const uint64_t batch_size, const std::vector<std::vector<int64_t>> &batch_info) {
  if (batch_info.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "param Dynamic batch info is empty, check invalid.");
    GELOGE(ge::FAILED, "[Check][Param] Dynamic batch info is empty.");
    return false;
  }

  for (const auto &batch : batch_info) {
    if (batch.size() != kDynamicBatchSizeVecSize) {
      REPORT_INNER_ERR_MSG("E19999", "Dynamic batch param num is %zu, current batch size is %zu.",
                         kDynamicBatchSizeVecSize, batch.size());
      GELOGE(ge::FAILED, "[Check][Param] Dynamic batch param num is %zu, current batch size is %zu.",
             kDynamicBatchSizeVecSize, batch.size());
      return false;
    }
    if (batch[0U] == static_cast<int64_t>(batch_size)) {
      return true;
    }
  }
  REPORT_INNER_ERR_MSG("E19999", "Dynamic batch %" PRIu64 " can not match the gear of model.", batch_size);
  GELOGE(ge::FAILED, "[Check][Param] Dynamic batch %" PRIu64 " can not match the gear of model.", batch_size);
  return false;
}

bool IsDynamicImageSizeMatchModel(const uint64_t image_height, const uint64_t image_width,
                                  const std::vector<std::vector<int64_t>> &batch_info) {
  if (batch_info.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "ParamDynamic batch info is empty. check invalid");
    GELOGE(ge::FAILED, "[Check][Param] Dynamic batch info is empty.");
    return false;
  }

  for (const auto &resolution : batch_info) {
    if (resolution.size() != kDynamicImageSizeVecSize) {
      REPORT_INNER_ERR_MSG("E19999", "Dynamic resolution param num is %zu, current resolution size is %zu.",
                         kDynamicImageSizeVecSize, resolution.size());
      GELOGE(ge::FAILED, "[Check][Param] Dynamic resolution param num is %zu, current resolution size is %zu.",
             kDynamicImageSizeVecSize, resolution.size());
      return false;
    }
    if ((resolution[0U] == static_cast<int64_t>(image_height)) &&
        (resolution[1U] == static_cast<int64_t>(image_width))) {
      return true;
    }
  }
  REPORT_INNER_ERR_MSG("E19999", "Dynamic resolution (%" PRIu64 ",%" PRIu64 ") can not match the gear of model.",
                     image_height, image_width);
  GELOGE(ge::FAILED, "[Check][Param]Dynamic resolution (%" PRIu64 ",%" PRIu64 ") can not match the gear of model.",
         image_height, image_width);
  return false;
}

bool IsDynmaicDimsSizeMatchModel(const std::vector<uint64_t> &cur_dynamic_dims,
                                 const std::vector<std::vector<int64_t>> &batch_info) {
  if (batch_info.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "param batch_info is empty, check invalid");
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] Dynamic batch info is empty.");
    return false;
  }

  bool find_match = false;
  for (const auto &resolution : batch_info) {
    if (cur_dynamic_dims.size() != resolution.size()) {
      REPORT_INNER_ERR_MSG("E19999", "Cur dynamic dims param num is %zu, current resolution size is %zu.",
                         cur_dynamic_dims.size(), resolution.size());
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[Check][Param] Cur dynamic dims param num is %zu, current resolution size is %zu.",
             cur_dynamic_dims.size(), resolution.size());
      return false;
    }
    bool flag = true;
    for (size_t i = 0U; i < resolution.size(); ++i) {
      if (cur_dynamic_dims[i] != static_cast<uint64_t>(resolution[i])) {
        flag = false;
        break;
      }
    }
    if (flag) {
      find_match = true;
      break;
    }
  }
  if (!find_match) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] choose dynamic dims can not match the gear of model.");
  }
  return find_match;
}

// With a dynamic-shaped model, a caller cannot allocate the buffer if no valid output size can be calculated.
// In this case the buffer can be set to NULL and the executor would allocate and cache it till the next invocation.
// Update output buffer back to run_output_data
void UpdateOutputBuffer(const bool is_async,
                        const std::vector<ge::DataBuffer> &execute_outputs,
                        std::vector<ge::DataBuffer> &user_outputs) {
  if (is_async) {
    return;
  }
  if (execute_outputs.size() != user_outputs.size()) {
    GELOGW("Output number mismatches, before execute: %zu, after execute: %zu",
           user_outputs.size(), execute_outputs.size());
    return;
  }
  for (size_t i = 0U; i < execute_outputs.size(); ++i) {
    auto &data_buffer = user_outputs[i];
    if (data_buffer.data == nullptr) {
      data_buffer.data = execute_outputs[i].data;
      data_buffer.length = execute_outputs[i].length;
      data_buffer.placement = execute_outputs[i].placement;
    }
  }
}
}  // namespace

namespace ge {
std::atomic_bool GeExecutor::is_inited_{false};

static void InitOpsProtoManager() {
  std::string opsproto_path;
  const Status ret = PluginManager::GetOpsProtoPath(opsproto_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get ops proto path!");
  }
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  std::map<std::string, std::string> option_tmp;
  (void)option_tmp.emplace(std::pair<std::string, std::string>(std::string("ge.opsProtoLibPath"), opsproto_path));
  (void)OpsProtoManager::Instance()->Initialize(option_tmp);
}

GeExecutor::GeExecutor() {}

Status GeExecutor::Initialize(const std::map<std::string, std::string> &options) {
  if (is_inited_) {
    GELOGW("Already initialized, no need to be initialized again.");
    return SUCCESS;
  }

  GELOGI("Init GeExecutor begin.");
  GE_ASSERT_GRAPH_SUCCESS(OpLibRegistry::GetInstance().PreProcessForCustomOp());
  OpTilingManager::GetInstance().LoadSo();

  const std::string path_base = GetModelPath();
  const Status init_hostcpu_engine_status = HostCpuEngine::GetInstance().Initialize(path_base);
  if (init_hostcpu_engine_status != SUCCESS) {
    GELOGE(init_hostcpu_engine_status, "[initialize][HostCpuEngine] failed");
    return init_hostcpu_engine_status;
  }

  const Status rt_plugin_status = ge::RuntimePluginLoader::GetInstance().Initialize(path_base);
  if (rt_plugin_status != SUCCESS) {
    GELOGE(rt_plugin_status, "[Init][RTv2]Failed to initialize rtv2 plugin.");
    return rt_plugin_status;
  }

  GE_CHK_STATUS_RET_NOLOG(OpsKernelExecutorManager::GetInstance().Initialize(options));
  InitOpsProtoManager();
  gert::OppPackageUtils::LoadAllOppPackage();

  GE_CHK_STATUS_RET(HostMemManager::Instance().Initialize());

  const std::vector<rtMemType_t> mem_type{RT_MEMORY_HBM, RT_MEMORY_P2P_DDR};
  Status status = MemManager::Instance().Initialize(mem_type);
  if (status != SUCCESS) {
    GELOGE(status, "[Init][MemManager] MemoryAllocatorManager initialize failed.");
    REPORT_INNER_ERR_MSG("E19999", "MemManager initialize failed.");
    return status;
  }

  //注册dump回调
  GELOGI("Register dump callbacks.");
  if (!ge::DumpCallbackManager::GetInstance().RegisterDumpCallbacks(GE_MODULE_NAME)) {
    GELOGW("Register dump callbacks failed, but continue initialization.");
  } else {
    GELOGI("Register dump callbacks successfully.");
  }

  status = DumpManager::GetInstance().Init(options);
  if (status != SUCCESS) {
    return status;
  }

  // 注册回调
  GE_ASSERT_SUCCESS(RegErrorTrackingCallBack());

  ProfilingProperties::Instance().SetExecuteProfiling(options);

  is_inited_.store(true);
  GELOGI("Init GeExecutor over.");
  return SUCCESS;
}

Status GeExecutor::FinalizeEx() {
  if (!is_inited_) {
    GELOGW("GeExecutor has not been initialized.");
    return SUCCESS;
  }

  hybrid::NpuMemoryAllocator::Finalize();
  // Stop profiling
  GELOGI("Begin to finalize ge executor. Report uninit to MsProf while profiling status[%d] is on",
         static_cast<int32_t>(ProfilingProperties::Instance().ProfilingOn()));
  if (ProfilingProperties::Instance().ProfilingOn()) {
    ProfilingProperties::Instance().ClearProperties();
  }

  OpsKernelExecutorManager::GetInstance().Finalize();
  HostMemManager::Instance().Finalize();

  MemManager::Instance().Finalize();
  DumpManager::GetInstance().Finalize();

  GELOGI("Uninit GeExecutor begin.");
  is_inited_.store(false);
  GELOGI("Uninit GeExecutor over.");
  return SUCCESS;
}

Status GeExecutor::Initialize() {
  // job id need to be set, the value is meaningless;
  const std::map<std::string, std::string> options({
      {OPTION_EXEC_JOB_ID, "1"}, {OPTION_EXEC_PROFILING_MODE, ""}, {OPTION_EXEC_PROFILING_OPTIONS, ""}
  });

  rtProfCtrlHandle const callback = &ge::ProfCtrlHandle;
  const int32_t ret = MsprofRegisterCallback(GE, callback);
  if (ret != MSPROF_ERROR_NONE) {
    GELOGE(FAILED, "register func failed");
    return FAILED;
  }

  return GeExecutor::Initialize(options);
}

Status GeExecutor::Finalize() {
  return GeExecutor::FinalizeEx();
}

Status GeExecutor::SetDynamicBatchSize(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                                       const uint64_t batch_size) {
  if (dynamic_input_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param dynamic_input_addr is nullptr, check invalid, model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID,
           "[Check][Param] Dynamic input addr is nullptr, model id:%u", model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID;
  }

  uint64_t size = sizeof(uint32_t);
  if (length < size) {
    REPORT_INNER_ERR_MSG("E19999", "Dynamic input size [%" PRIu64 "] is less than ["
		       "%" PRIu64 "], check invalid, model id:%u", length, size, model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
           "[Check][Param] Dynamic input size [%" PRIu64 "] is less than [%" PRIu64 "], model id:%u",
           length, size, model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
  }
  if (length >= sizeof(uint64_t)) {
    size = sizeof(uint64_t);
  }

  // Verify whether the input dynamic batch matches the model gear
  std::vector<std::vector<int64_t>> batch_info;
  const std::vector<uint64_t> batch_num{batch_size};
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::FIXED);
  Status ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "get dynamic batch info failed, model id:%u", model_id);
    GELOGE(ret, "[Get][DynamicBatchInfo] failed, model id:%u.", model_id);
    return ret;
  }

  if (!IsDynamicBatchSizeMatchModel(batch_size, batch_info)) {
    GELOGE(ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID,
           "[Check][Param] The current dynamic input does not match the gear of the model(id:%u).", model_id);
    return ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID;
  }

  ret = GraphExecutor::SetDynamicSize(model_id, batch_num, static_cast<int32_t>(DynamicInputType::DYNAMIC_BATCH));
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "set dynamic size failed, model id:%u, dynamic_type:1", model_id);
    GELOGE(ret, "[Set][DynamicSize] failed, model id:%u, dynamic_type:1", model_id);
    return ret;
  }
  // memcpy dynamic_batch_size from host to device
  const aclError rt_ret = aclrtMemcpy(dynamic_input_addr, length, &batch_size, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy, size:%" PRIu64 " ret:%d", length, rt_ret);
    GELOGE(RT_FAILED, "[Call][aclrtMemcpy] memcpy dynamic batch input data failed! size:%" PRIu64 " ret:%d",
      length, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status GeExecutor::SetDynamicImageSize(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                                       const uint64_t image_height, const uint64_t image_width) {
  if (dynamic_input_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param dynamic_input_addr is nullptr, check invalid, model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID,
           "[Check][Param] Dynamic input addr is nullptr, model id:%u", model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID;
  }

  constexpr uint64_t dynamic_input_size = kDynamicImageSizeInputSize * sizeof(uint32_t);
  if (length < dynamic_input_size) {
    REPORT_INNER_ERR_MSG("E19999", "Dynamic input size [%" PRIu64 "] is less than [%" PRIu64 "], "
		       "check invalid, model id:%u", length, dynamic_input_size, model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
           "[Check][Param] Dynamic input size [%" PRIu64 "] is less than [%" PRIu64 "], model id:%u",
           length, dynamic_input_size, model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
  }
  uint64_t size = sizeof(uint32_t);
  if (length >= (kDynamicImageSizeInputSize * sizeof(uint64_t))) {
    size = sizeof(uint64_t);
  }
  // Verify whether the input dynamic resolution matches the model gear
  std::vector<std::vector<int64_t>> batch_info;
  const std::vector<uint64_t> batch_num{image_height, image_width};
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::FIXED);
  Status ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get dynamic input info failed, model id:%u.", model_id);
    GELOGE(ret, "[Get][DynamicBatchInfo] failed, model id:%u.", model_id);
    return ret;
  }

  if (!IsDynamicImageSizeMatchModel(image_height, image_width, batch_info)) {
    GELOGE(ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID,
           "[Check][Param] The current dynamic input does not match the gear of the model, "
           "image_height:%" PRIu64 ", image_width:%" PRIu64 ".", image_height, image_width);
    return ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID;
  }

  ret = GraphExecutor::SetDynamicSize(model_id, batch_num, static_cast<int32_t>(DynamicInputType::DYNAMIC_IMAGE));
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Set dynamic size failed, model id:%u,", model_id);
    GELOGE(ret, "[Set][DynamicSize] failed, model id:%u", model_id);
    return ret;
  }

  // Memcpy dynamic resolution height from host to device
  aclError rt_ret =
      aclrtMemcpy(dynamic_input_addr, size, &image_height, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed! size:%" PRIu64 ", ret:%d, model id:%u",
                      size, rt_ret, model_id);
    GELOGE(RT_FAILED, "[Call][aclrtMemcpy] memcpy dynamic resolution input data failed! size:%" PRIu64 ", "
      "ret:%d, model id:%u", size, rt_ret, model_id);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  const uint64_t remain_size = length - size;
  // Memcpy dynamic resolution width from host to device
  rt_ret = aclrtMemcpy(ValueToPtr(PtrToValue(dynamic_input_addr) + size), remain_size, &image_width,
      size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed! size:%" PRIu64 ", ret:%d, model id:%u",
                      remain_size, rt_ret, model_id);
    GELOGE(RT_FAILED, "[Call][aclrtMemcpy] memcpy dynamic resolution input data failed! size:%" PRIu64 ", "
      "ret:%d, model id:%u", remain_size, rt_ret, model_id);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status GeExecutor::SetDynamicDims(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                                  const std::vector<uint64_t> &dynamic_dims) {
  if (dynamic_input_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param dynamic_input_addr is nullptr, check invalid, "
		       "model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID,
           "[Check][Param] Dynamic input addr is nullptr, model id:%u", model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID;
  }

  std::vector<uint64_t> cur_dynamic_dims;
  Status ret = GetCurDynamicDims(model_id, dynamic_dims, cur_dynamic_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][CurDynamicDims] failed, model id:%u", model_id);
    return ret;
  }
  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::FIXED);
  ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get dynamic input info failed, model id:%u.", model_id);
    GELOGE(ret, "[Get][DynamicBatchInfo] failed, model id:%u.", model_id);
    return ret;
  }

  if (!IsDynmaicDimsSizeMatchModel(cur_dynamic_dims, batch_info)) {
    GELOGE(ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID,
           "[Check][Param] The current dynamic input does not match the gear of the model, id:%u.", model_id);
    return ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID;
  }

  ret = GraphExecutor::SetDynamicSize(model_id, cur_dynamic_dims, static_cast<int32_t>(DYNAMIC_DIMS));
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Set dynamic size failed, model id:%u", model_id);
    GELOGE(ret, "[Set][DynamicSize] failed, model id:%u", model_id);
    return ret;
  }

  const size_t dynamic_dim_num = cur_dynamic_dims.size();
  const uint64_t dynamic_input_size = static_cast<uint64_t>(dynamic_dim_num * sizeof(uint32_t));
  if (length < dynamic_input_size) {
    REPORT_INNER_ERR_MSG("E19999", "input dynamic size [%" PRIu64 "] is less than [%" PRIu64 "], model id:%u",
                       length, dynamic_input_size, model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
           "[Check][Param] Dynamic input size [%" PRIu64 "] is less than [%" PRIu64 "], model id:%u",
           length, dynamic_input_size, model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
  }
  uint64_t size = sizeof(uint32_t);
  if (length >= static_cast<uint64_t>(dynamic_dim_num * sizeof(uint64_t))) {
    size = sizeof(uint64_t);
  }
  aclError rt_ret;
  for (size_t i = 0U; i < dynamic_dim_num; ++i) {
    // Memcpy dynamic dim[i] from host to device
    rt_ret = aclrtMemcpy(ValueToPtr(PtrToValue(dynamic_input_addr) + (size * i)),
        length - (size * i), &cur_dynamic_dims[i], size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != ACL_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, size:%" PRIu64 ", ret:%d",
          (length - (size * i)), rt_ret);
      GELOGE(RT_FAILED, "[Call][aclrtMemcpy] memcpy dynamic resolution input data failed! size:%" PRIu64 ", ret:%d",
             (length - (size * i)), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  return SUCCESS;
}

Status GeExecutor::GetCurDynamicDims(const uint32_t model_id, const std::vector<uint64_t> &dynamic_dims,
                                     std::vector<uint64_t> &cur_dynamic_dims) {
  cur_dynamic_dims.clear();
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
  auto ret = GetModelDescInfo(model_id, input_desc, output_desc);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][ModelDescInfo] failed, model id:%u.", model_id);
    return ret;
  }
  std::vector<std::string> user_designate_shape_order;
  std::vector<int64_t> all_data_dims;
  ret = GetUserDesignateShapeOrder(model_id, user_designate_shape_order);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GetUserDesignateShapeOrder] failed, model id:%u.", model_id);
    return ret;
  }
  for (auto &data_name : user_designate_shape_order) {
    for (auto &desc : input_desc) {
      AscendString get_name;
      (void) desc.GetName(get_name);
      if (get_name.GetString() == data_name) {
        const auto dims = desc.GetShape().GetDims();
        (void)std::copy(dims.begin(), dims.end(), std::back_inserter(all_data_dims));
        break;
      }
    }
  }
  if (dynamic_dims.size() != all_data_dims.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Dynamic input size [%" PRIu64 "] is not equal with all data dims size [%" PRIu64 "]!",
                       static_cast<uint64_t>(dynamic_dims.size()), static_cast<uint64_t>(all_data_dims.size()));
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
           "[Check][Param] Dynamic input size [%" PRIu64 "] is not equal with all data dims size [%" PRIu64 "]!",
           dynamic_dims.size(), all_data_dims.size());
    return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
  }
  for (size_t i = 0U; i < all_data_dims.size(); ++i) {
    if (all_data_dims[i] < 0) {
      cur_dynamic_dims.push_back(dynamic_dims[i]);
    } else {
      if (static_cast<uint64_t>(all_data_dims[i]) != dynamic_dims[i]) {
        REPORT_INNER_ERR_MSG("E19999", "Static dims should be same, index:%zu value:%" PRIu64 " should be %" PRId64 "",
                           i, dynamic_dims[i], all_data_dims[i]);
        GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
               "[Check][Param] Static dims should be same, index:%zu value:%" PRIu64 " should be %" PRId64,
               i, dynamic_dims[i], all_data_dims[i]);
        return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status GeExecutor::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type) {
  GELOGI("Begin to get current shape");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized, model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized, model id:%u", model_id);
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  const Status ret = GraphExecutor::GetCurrentShape(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get Cur Shape failed, model id:%u", model_id);
    GELOGE(ret, "[Get][CurShape] failed, model id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GeExecutor::SetDynamicAippData(const uint32_t model_id, void *const dynamic_input_addr, const uint64_t length,
                                      const std::vector<kAippDynamicBatchPara> &aipp_batch_para,
                                      const kAippDynamicPara &aipp_parms) {
  GELOGI("Enter to SetDynamicAippData.");
  if (dynamic_input_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param dynamic_input_addr is nullptr, check invalid, model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID,
           "[Check][Param] Dynamic aipp input addr is nullptr, model id:%u", model_id);
    return ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID;
  }
  if (aipp_batch_para.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Param aipp_batch_para is empty, check invalid, model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_AIPP_BATCH_EMPTY, "[Check][Param] aipp_batch_para is empty, model id:%u", model_id);
    return ACL_ERROR_GE_AIPP_BATCH_EMPTY;
  }
  const uint64_t batch_num = aipp_batch_para.size();
  constexpr uint64_t real_aippParms_size = sizeof(kAippDynamicPara) - sizeof(kAippDynamicBatchPara);
  const uint64_t struct_len = (batch_num * sizeof(kAippDynamicBatchPara)) + real_aippParms_size;
  GELOGI("Get acl input dynamic aipp data, model_id is %u, length is %" PRIu64 ", batch num is %" PRIu64 ", "
    "struct_len is %" PRIu64, model_id, length, batch_num, struct_len);
  if (struct_len > length) {
    REPORT_INNER_ERR_MSG("E19999", "input dynamic aipp param len:%" PRIu64 " is larger than aipp_data size:%" PRIu64 "",
                       struct_len, length);
    GELOGE(ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID,
           "[Check][Param] input dynamic aipp param len [%" PRIu64 "] is larger than aipp_data size [%" PRIu64 "]",
           struct_len, length);
    return ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID;
  }
  // Memcpy real kAippDynamicBatchPara from host to device
  aclError rt_ret = aclrtMemcpy(dynamic_input_addr, length, &aipp_parms,
      real_aippParms_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, size:%" PRIu64 ", ret:%d", length, rt_ret);
    GELOGE(RT_FAILED, "[Call][aclrtMemcpy] memcpy aipp_parms failed! size:%" PRIu64 ", ret:%d", length, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  const uint64_t remain_len = length - real_aippParms_size;
  const uint64_t aipp_batch_para_dev = PtrToValue(dynamic_input_addr) + real_aippParms_size;

  for (uint64_t i = 0U; i < batch_num; ++i) {
    rt_ret = aclrtMemcpy(ValueToPtr(aipp_batch_para_dev + (i * sizeof(kAippDynamicBatchPara))),
        (remain_len - (i * sizeof(kAippDynamicBatchPara))), &(aipp_batch_para[i]),
        sizeof(kAippDynamicBatchPara), ACL_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != ACL_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret:%d", rt_ret);
      GELOGE(RT_FAILED, "[Call][aclrtMemcpy] memcpy kAippDynamicBatchPara input data failed! ret:%d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  return SUCCESS;
}

Status GeExecutor::UnloadModel(const uint32_t model_id) {
  GELOGD("unload model %u begin.", model_id);
  Status ret = ModelManager::GetInstance().DestroyAicpuSessionForInfer(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Destroy][AicpuSession] For Infer failed. model id:%u", model_id);
    return ret;
  }

  const auto hybrid_davinci_model = ModelManager::GetInstance().GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    const uint64_t session_id = hybrid_davinci_model->GetSessionId();
    VarManagerPool::Instance().RemoveVarManager(session_id);
    ExternalWeightManagerPool::Instance().RemoveManager(session_id);
    SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(session_id,
                                                                                     GetContext().DeviceId());
    SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id,
                                                                                  GetContext().DeviceId());
    SessionMemAllocator<ActiveMemoryAllocator>::Instance().RemoveAllocator(session_id, GetContext().DeviceId());
  } else {
    const auto davinci_model = ModelManager::GetInstance().GetModel(model_id);
    // if session is shared, can not destroy resource here
    if ((davinci_model != nullptr) && (!ModelManager::GetInstance().IsModelSharedSession(model_id))) {
      const uint64_t session_id = davinci_model->GetSessionId();
      VarManagerPool::Instance().RemoveVarManager(session_id);
      gert::RtVarManagerPool::Instance().RemoveRtVarManager(session_id);
      ExternalWeightManagerPool::Instance().RemoveManager(session_id);
      SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(session_id,
                                                                                       GetContext().DeviceId());
      SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id,
                                                                                    GetContext().DeviceId());
      SessionMemAllocator<ActiveMemoryAllocator>::Instance().RemoveAllocator(session_id, GetContext().DeviceId());
    }
  }
  ret = GraphLoader::UnloadModel(model_id);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "unload model failed, model id:%u", model_id);
    GELOGE(ret, "[Unload][Model] failed. model id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GeExecutor::RecoverAllModel(const int32_t device_id) const {
    return ModelManager::GetInstance().RecoverAllModel(device_id);
}

Status GeExecutor::GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info) const {
  return GraphLoader::GetModelDescInfoFromMem(model_data, info);
}

// Get input and output descriptor
Status GeExecutor::GetModelDescInfo(const uint32_t model_id, std::vector<TensorDesc> &input_desc,
                                    std::vector<TensorDesc> &output_desc, const bool new_model_desc) {
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized, model id:%u", model_id);
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized, model id:%u", model_id);
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  std::vector<InputOutputDescInfo> input_desc_infos;
  std::vector<InputOutputDescInfo> output_desc_infos;
  std::vector<uint32_t> input_formats;
  std::vector<uint32_t> output_formats;

  const auto ret = GraphExecutor::GetInputOutputDescInfo(model_id, input_desc_infos, output_desc_infos, input_formats,
                                                         output_formats, new_model_desc);
  GE_CHK_BOOL_RET_STATUS((ret == ge::SUCCESS), ret,
                         "[Get][InputOutputDescInfo] failed. ret = %u, model id:%u", ret, model_id);

  if (input_formats.size() != input_desc_infos.size()) {
    REPORT_INNER_ERR_MSG("E19999", "input_formats size %zu is not equal to input_desc_infos size %zu, model id:%u.",
                       input_formats.size(), input_desc_infos.size(), model_id);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param] input_formats size %zu is not equal to input_desc_infos size %zu, model id:%u.",
           input_formats.size(), input_desc_infos.size(), model_id);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (output_formats.size() != output_desc_infos.size()) {
    REPORT_INNER_ERR_MSG("E19999", "output_formats size %zu is not equal to output_desc_infos size %zu, model id:%u.",
                       output_formats.size(), output_desc_infos.size(), model_id);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param] output_formats size %zu is not equal to output_desc_infos size %zu, model id:%u.",
           output_formats.size(), output_desc_infos.size(), model_id);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  // Transfer data to TensorDesc
  GetGeTensorDescFromDomiInfo(input_desc, input_desc_infos, input_formats);
  GetGeTensorDescFromDomiInfo(output_desc, output_desc_infos, output_formats);

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @param [out] dynamic_type
/// @return execute result
///
Status GeExecutor::GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                       int32_t &dynamic_type) {
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const auto ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Get Dynamic BatchInfo failed, model id:%u.", model_id);
    GELOGE(ret, "[Get][DynamicBatchInfo] failed, model id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status GeExecutor::GetCombinedDynamicDims(const uint32_t model_id,
                                          std::vector<std::vector<int64_t>> &batch_info) {
  GELOGI("Begin to get combined dynamic dims info.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const auto ret = GraphExecutor::GetCombinedDynamicDims(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][CombinedDynamicDims] failed, model id:%u.", model_id);
    return ret;
  }

  GELOGI("Get combined dynamic dims succ.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get user designeate shape order
/// @param [in] model_id
/// @param [out] user_designate_shape_order
/// @return execute result
///
Status GeExecutor::GetUserDesignateShapeOrder(const uint32_t model_id,
                                              std::vector<std::string> &user_designate_shape_order) {
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const auto ret = GraphExecutor::GetUserDesignateShapeOrder(model_id, user_designate_shape_order);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GetUserDesignateShapeOrder] failed, model id:%u.", model_id);
    return ret;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get AIPP input format
/// @param [in] model_id
/// @param [in] index
/// @param [out] input_format
/// @return execute result
///
Status GeExecutor::GetAIPPInfo(const uint32_t model_id, const uint32_t index, AippConfigInfo &aipp_info) {
  GELOGI("Begin to GetAIPPInfo.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor not inited yet!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  const auto ret = GraphExecutor::GetAippInfo(model_id, index, aipp_info);
  if (ret != SUCCESS) {
    GELOGW("GetAIPPInfo is not success.");
    return ret;
  }
  GELOGI("GetAIPPInfo succ.");
  return SUCCESS;
}

Status GeExecutor::GetAippType(const uint32_t model_id, const uint32_t index, InputAippType &type,
                               size_t &aipp_index) {
  GELOGI("Begin to get aipp type.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "GeExecutor has not inited yet!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  const auto ret = GraphExecutor::GetAippType(model_id, index, type, aipp_index);
  if (ret != SUCCESS) {
    GELOGW("Get aipp type is not success.");
    return ret;
  }
  GELOGI("Get aipp type success.");
  return SUCCESS;
}

Status GeExecutor::GetOpAttr(const uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                             std::string &attr_value) {
  GELOGI("Begin to get op attr.");
  if (!is_inited_) {
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Init][GeExecutor]Ge executor not inited yet!");
    REPORT_INNER_ERR_MSG("E19999", "Ge executor not inited yet!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  const auto ret = GraphExecutor::GetNodeAttr(model_id, op_name, attr_name, attr_value);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OpAttr]Get op:%s attr:%s failed, model id:%u.",
           op_name.c_str(), attr_name.c_str(), model_id);
    REPORT_INNER_ERR_MSG("E19999", "Get op:%s attr:%s failed, model id:%u",
                      op_name.c_str(), attr_name.c_str(), model_id);
    return ret;
  }
  return SUCCESS;
}

Status GeExecutor::GetModelAttr(const uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info) {
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not inited yet!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not inited yet!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  const auto ret = GraphExecutor::GetOutputShapeInfo(model_id, dynamic_output_shape_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][ModelAttr] failed, model id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GeExecutor::CommandHandle(const Command &command) const {
  const Status ret = ModelManager::GetInstance().HandleCommand(command);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "call CommandHandle failed, ret:%u", ret);
    GELOGE(ACL_ERROR_GE_COMMAND_HANDLE, "[Call][CommandHandle] failed, ret:%u", ret);
    return ACL_ERROR_GE_COMMAND_HANDLE;
  }
  return SUCCESS;
}

Status GeExecutor::GetMaxUsedMemory(const uint32_t model_id, uint32_t &max_size) {
  GELOGI("Get max used memory begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  uint64_t max_mem_size = 0U;
  const auto ret = ModelManager::GetInstance().GetMaxUsedMemory(model_id, max_mem_size);
  max_size = static_cast<uint32_t>(max_mem_size);
  return ret;
}

/**
 * @ingroup ge
 * @brief Load data from model file to memory
 * @param [in] const std::string &path: Offline model file path
 * @param [out] domi::ModelData &model_data: Offline model memory data
 * @return SUCCESS handle successfully / others handle failed
 */
Status GeExecutor::LoadDataFromFile(const std::string &path, ModelData &model_data) {
  GELOGI("Load data from file begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const std::string filePath = RealPath(path.c_str());
  if (filePath.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E13026", std::vector<const char_t *>({"pathname", "reason"}),
        std::vector<const char_t *>({path.c_str(), "It is not a real path. Please check your model path."}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID,
           "[Call][RealPath] File path is invalid. please check your text file '%s'.", path.c_str());
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }
  GELOGI("load modelData from file: %s.", path.c_str());
  constexpr int32_t priority = 0;
  const auto ret = GraphLoader::LoadDataFromFile(path, priority, model_data);
  if (ret != SUCCESS) {
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char_t *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  }
  return ret;
}

/**
* @ingroup ge
* @brief Load model from offline model memory data
* @param [in] domi::ModelData &model_data: Offline model data
              void *dev_ptr: Input/Output memory start address
              size_t memsize: Input/Output memory length
              void *weight_ptr: Weight memory start address
              size_t weightsize: Weight memory length
* @param [out] uint32_t &model_id: identification after model loading
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *const dev_ptr,
                                     const size_t mem_size, void *const weight_ptr, const size_t weight_size) {
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not inited yet!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const uintptr_t mem_base = PtrToValue(dev_ptr);
  const uintptr_t weight_base = PtrToValue(weight_ptr);
  const ModelParam model_param { model_data.priority, mem_base, mem_size, weight_base, weight_size };
  return GraphLoader::LoadModelFromData(model_data, model_param, model_id);
}

Status GeExecutor::LoadModelFromDataWithArgs(uint32_t &model_id, const ModelData &model_data,
                                             const ModelLoadArg &load_arg) {
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not inited yet!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  void *external_var_addr = nullptr;
  uint64_t external_var_size = 0;
  if (load_arg.rt_session != nullptr) {
    load_arg.rt_session->GetExternalVar(external_var_addr, external_var_size);
  }

  const ModelParam model_param{model_data.priority, reinterpret_cast<uintptr_t>(load_arg.dev_ptr), load_arg.mem_size,
                               reinterpret_cast<uintptr_t>(load_arg.weight_ptr), load_arg.weight_size,
                               &load_arg.file_constant_mems, external_var_addr, external_var_size,
                               load_arg.need_clear_dfx_cache};
  return ModelManager::GetInstance().LoadModelOffline(model_data, model_param, model_id, load_arg.rt_session);
}

/**
 * @ingroup ge
 * @brief Load task list from ModelData with queue.
 * @param [out] model_id: model id allocate from manager.
 * @param [in] ge_model_data: Model data load from offline model.
 * @param [in] input_queue_ids: input queue ids create from user.
 * @param [in] output_queue_ids: input queue ids create from user.
 * @return: 0 for success / others for fail
 */
Status GeExecutor::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                  const std::vector<uint32_t> &input_queue_ids,
                                  const std::vector<uint32_t> &output_queue_ids) {
  GELOGI("Load model with queue begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  ModelQueueArg args{.input_queue_ids = input_queue_ids, .output_queue_ids = output_queue_ids,
                     .file_constant_mems = {}};
  return GraphLoader::LoadModelWithQ(model_id, model_data, args);
}

/**
 * @ingroup ge
 * @brief Load task list from ModelData with queue.
 * @param [out] model_id: model id allocate from manager.
 * @param [in] ge_model_data: Model data load from offline model.
 * @param [in] args: input/output queue ids, and file constant mems
 * @return: 0 for success / others for fail
 */
Status GeExecutor::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                  const ModelQueueArg &args) {
  GELOGI("Load model with queue using model queue arg begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  return GraphLoader::LoadModelWithQ(model_id, model_data, args);
}

/**
 * @ingroup ge
 * @brief Load task list from GeRootModel with queue and param.
 * @param [out] model_id: model id allocate from manager.
 * @param [in] root_model: Instance of GeRootModel.
 * @param [in] model_queue_param: params and queue ids and create from user.
 * @return: 0 for success / others for fail
 */
Status GeExecutor::LoadModelWithQ(uint32_t &model_id,
                                  const shared_ptr<GeRootModel> &root_model,
                                  const ModelQueueParam &model_queue_param) const {
  return GraphLoader::LoadModelWithQueueParam(model_id, root_model, model_queue_param, false);
}

Status GeExecutor::LoadModelWithQueueParam(uint32_t &model_id, const ModelData &model_data,
                                           const ModelQueueParam &model_queue_param) const {
  GELOGI("Load model with queue param begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  return GraphLoader::LoadModelWithQueueParam(model_id, model_data, model_queue_param);
}

/**
 * @ingroup ge
 * @brief Load task list from root_model without queue.
 * @param [out] model_id: model id allocate from manager.
 * @param [in] root_model:model of root
 * @return: 0 for success / others for fail
 */
Status GeExecutor::LoadModelWithoutQ(uint32_t &model_id, const shared_ptr<GeRootModel> &root_model) const {
  return GraphLoader::LoadModelWithoutQ(model_id, root_model);
}

/**
* @ingroup ge
* @brief Synchronous execution of offline model(Do not create thread)
* @param [in] uint32_t model_id: Model ID to execute
              void* stream: stream to execute
              const domi::InputData *input_data: Model input data
              bool async_mode: is asynchronize mode.
* @param [out] domi::OutputData *output_data: Model output data
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::ExecModel(const uint32_t model_id, void *const stream, const RunModelData &input_data,
                             RunModelData &output_data, const bool async_mode) {
  std::vector<GeTensorDesc> output_desc;
  return ExecModel(model_id, stream, input_data, {}, output_data, output_desc, async_mode);
}

/**
* @ingroup ge
* @brief Synchronous execution of offline model(Do not create thread)
* @param [in] uint32_t model_id: Model ID to execute
              void* stream: stream to execute
              const domi::InputData *input_data: Model input data
              const std::vector<GeTensorDesc> &input_desc: Description of model input data
              bool async_mode: is asynchronize mode
* @param [out] domi::OutputData *output_data: Model output data
* @param [out] std::vector<GeTensorDesc> &output_desc: Description of model output data
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::ExecModel(const uint32_t model_id, void *const stream, const RunModelData &run_input_data,
                             const std::vector<GeTensorDesc> &input_desc, RunModelData &run_output_data,
                             std::vector<GeTensorDesc> &output_desc, const bool async_mode) {
  RT2_PROFILING_SCOPE_CONST(gert::profiling::kUnknownName, gert::profiling::kModelExecute);
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  InputData input_data;
  OutputData output_data;
  GetDomiInputData(run_input_data, input_data);
  GetDomiOutputData(run_output_data, output_data);

  if ((run_input_data.dynamic_batch_size != 0U) || (run_input_data.dynamic_image_width != 0U) ||
      (run_input_data.dynamic_image_height != 0U) || (run_input_data.dynamic_dims.size() != 0U)) {
    std::vector<std::vector<int64_t>> batch_info;
    int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::FIXED);
    const auto ret = GraphExecutor::GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "get dynamic batch info failed, model id:%u.", model_id);
      GELOGE(ret, "[Get][DynamicBatchInfo] failed, model id:%u.", model_id);
      return ret;
    }
    if (!batch_info.empty()) {
      SetDynamicInputDataFlag(run_input_data, batch_info, input_data);
    }
  }

  GE_CHK_STATUS_RET_NOLOG(GraphLoader::ExecuteModel(model_id,
                                                    stream,
                                                    async_mode,
                                                    input_data,
                                                    input_desc,
                                                    output_data,
                                                    output_desc));
  UpdateOutputBuffer(async_mode, output_data.blobs, run_output_data.blobs);
  return SUCCESS;
}

/**
* @ingroup ge
* @brief Get weight memory size from model file
* @param [in] const std::string &path: Offline model file path
* @param [out] size_t &mem_size Execution memory size
               size_t &weight_size Weight memory space size
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::GetMemAndWeightSize(const std::string &path, size_t &mem_size, size_t &weight_size) {
  GELOGI("Get memory and weight size from file begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  ModelData model;
  Status ret = GraphLoader::LoadDataFromFile(path, 0, model);
  if ((ret != SUCCESS) || (model.model_data == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "load data from file failed, ret = %u", ret);
    GELOGE(ret, "[Load][Data] from file failed. ret = %u", ret);
    return ret;
  }

  ret = ModelManager::GetModelMemAndWeightSize(model, mem_size, weight_size);

  delete[] static_cast<char_t *>(model.model_data);
  model.model_data = nullptr;

  return ret;
}

/**
* @ingroup ge
* @brief Get weight memory size from model file
* @param [in] const void *model_data Offline model buffer
              size_t model_size Offline model buffer length
* @param [out] size_t &mem_size Execution memory size
               size_t &weight_size Weight memory space size
* @return SUCCESS handle successfully / others handle failed
*/
Status GeExecutor::GetMemAndWeightSize(const void *const model_data, const size_t model_size, size_t &mem_size,
                                       size_t &weight_size) {
  GELOGI("Get memory and weight size from data begin.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  if (model_data == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param model_data is nullptr, check invalid!");
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID, "[Check][Param] invalid model data!");
    return ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID;
  }

  ModelData model;
  model.model_data = ValueToPtr(PtrToValue(model_data));
  model.model_len = model_size;

  return ModelManager::GetModelMemAndWeightSize(model, mem_size, weight_size);
}

Status GeExecutor::LoadSingleOp(const std::string &model_name, const ModelData &model_data, void *const stream,
                                SingleOp **const single_op) {
  return LoadSingleOpV2(model_name, model_data, stream, single_op, 0U);
}

Status GeExecutor::LoadSingleOpV2(const std::string &model_name, const ModelData &model_data, void *const stream,
                                  SingleOp **const single_op, const uint64_t model_id) {
  return SingleOpManager::GetInstance().GetOpFromModel(model_name, model_data, stream, single_op, model_id);
}

Status GeExecutor::LoadDynamicSingleOp(const std::string &model_name, const ModelData &model_data, void *const stream,
                                       DynamicSingleOp **const single_op) {
  return LoadDynamicSingleOpV2(model_name, model_data, stream, single_op, 0U);
}

Status GeExecutor::LoadDynamicSingleOpV2(const std::string &model_name, const ModelData &model_data, void *const stream,
                                         DynamicSingleOp **const single_op, const uint64_t model_id) {
  return SingleOpManager::GetInstance().GetDynamicOpFromModel(model_name, model_data, stream, single_op, model_id);
}

Status GeExecutor::UnloadSingleOp(const uint64_t op_id) {
  return SingleOpManager::GetInstance().DeleteSingleOp(op_id);
}

Status GeExecutor::UnloadDynamicSingleOp(const uint64_t op_id) {
  return SingleOpManager::GetInstance().DeleteDynamicSingleOp(op_id);
}

Status GeExecutor::ExecuteAsync(SingleOp *const executor, const std::vector<DataBuffer> &inputs,
                                std::vector<DataBuffer> &outputs) {
  if (executor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param executor is nullptr, check invalid");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] param executor is nullptr");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }
  GE_PROFILING_START(kStaticSingleOpExecute);
  const auto ret = executor->ExecuteAsync(inputs, outputs);
  GE_PROFILING_END(static_cast<uint64_t>(executor->GetProfilingNodeIndex()), gert::profiling::kStaticSingleOpExecute,
                   kStaticSingleOpExecute);
  return ret;
}

Status GeExecutor::ExecuteAsync(DynamicSingleOp *const executor, const std::vector<GeTensorDesc> &input_desc,
                                const std::vector<DataBuffer> &inputs, std::vector<GeTensorDesc> &output_desc,
                                std::vector<DataBuffer> &outputs) {
  GE_CHECK_NOTNULL(executor);
  PROFILING_SCOPE(executor->GetProfilingNodeIndex(), profiling::kOpExecute);
  const auto ret = executor->ExecuteAsync(input_desc, inputs, output_desc, outputs);
  return ret;
}

Status GeExecutor::ReleaseSingleOpResource(void *const stream) {
  return SingleOpManager::GetInstance().ReleaseResource(stream);
}

Status GeExecutor::ClearCustomAicpuSo(const uint32_t device_id) {
  int32_t cur_device_id = -1;
  GE_CHK_RT_RET(rtCtxGetDevice(&cur_device_id));
  if (device_id != static_cast<uint32_t>(cur_device_id)) {
    GELOGW("given device_id[%u] is not equal to cur_device_id[%i], skip clear so", device_id, cur_device_id);
    return SUCCESS;
  }
  return ModelManager::GetInstance().ClearAicpuSo();
}

Status GeExecutor::GetDeviceIdByModelId(const uint32_t model_id, uint32_t &device_id) {
  const auto davinci_model = ModelManager::GetInstance().GetModel(model_id);
  if (davinci_model == nullptr) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
           "[Get][Model] failed, Model id:%u is invaild or model is not loaded.", model_id);
    return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
  }

  device_id = davinci_model->GetDeviceId();
  return SUCCESS;
}

Status GeExecutor::GetBatchInfoSize(const uint32_t model_id, size_t &shape_count) {
  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = static_cast<int32_t>(DynamicInputType::FIXED);
  const auto ret = GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][DynamicBatchInfo] failed. ret = %d, model id:%u", ret, model_id);
    return ret;
  }
  if (batch_info.empty()) {
    shape_count = kStaticBatchInfoSize;
  } else {
    shape_count = batch_info.size();
  }
  return SUCCESS;
}

Status GeExecutor::GetOrigInputInfo(const uint32_t model_id, const uint32_t index, OriginInputInfo &orig_input_info) {
  GELOGI("Begin to GetOrigInputInfo.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const auto ret = GraphExecutor::GetOrigInputInfo(model_id, index, orig_input_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OrigInputInfo] failed, model id:%u.", model_id);
    return ret;
  }

  GELOGI("GetOrigInputInfo succ.");
  return SUCCESS;
}

Status GeExecutor::GetAllAippInputOutputDims(const uint32_t model_id, const uint32_t index,
                                             std::vector<InputOutputDims> &input_dims,
                                             std::vector<InputOutputDims> &output_dims) {
  GELOGI("Begin to GetAllAippInputOutputDims.");
  if (!is_inited_) {
    REPORT_INNER_ERR_MSG("E19999", "GeExecutor has not been initialized!");
    GELOGE(ACL_ERROR_GE_EXEC_NOT_INIT, "[Check][Param] GeExecutor has not been initialized!");
    return ACL_ERROR_GE_EXEC_NOT_INIT;
  }

  const auto ret = GraphExecutor::GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][AllAippInputOutputDims] failed, model id:%u.", model_id);
    return ret;
  }

  GELOGI("GetAllAippInputOutputDims succ.");
  return SUCCESS;
}

Status GeExecutor::GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                                 OpDescInfo &op_desc_info) {
  GELOGI("Begin to GetOpDescInfo.");
  const auto ret = GraphExecutor::GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "get opdesc info failed, device_id:%u, stream_id:%u, task_id:%u.",
                      device_id, stream_id, task_id);
    GELOGE(ret, "[Get][OpDescInfo] failed, device_id:%u, stream_id:%u, task_id:%u.",
           device_id, stream_id, task_id);
    return ret;
  }
  GELOGI("GetOpDescInfo succ.");
  return SUCCESS;
}

Status GeExecutor::SetDump(const DumpConfig &dump_config) {
  GELOGI("Start to set dump config");
  const auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][DumpConf] failed, ret:%d", ret);
    return ret;
  }
  GELOGI("Set dump config successfully");
  return SUCCESS;
}

Status GeExecutor::SetAllocator(void *const stream, Allocator *const external_allocator) {
  return SingleOpManager::GetInstance().SetAllocator(stream, external_allocator);
}

Status GeExecutor::ReleaseResource(const uint32_t device_id) {
  MemManager::Instance().ReleaseResource(device_id);
  return SUCCESS;
}

Status GeExecutor::ReleaseResource() {
  return ReleaseResource(0U);
}

Status GeExecutor::GetRuntimeModelId(const uint32_t model_id,
                                     uint32_t &model_runtime_id) {
  return GraphLoader::GetRuntimeModelId(model_id, model_runtime_id);
}
}  // namespace ge
