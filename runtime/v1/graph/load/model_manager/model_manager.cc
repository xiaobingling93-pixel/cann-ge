/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/model_manager.h"

#include "aicpu_engine_struct.h"
#include "aicpu_op_type_list.h"
#include "common/helper/model_parser_base.h"
#include "common/dump/dump_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/file_constant_utils.h"
#include "graph/ge_context.h"
#include "common/model/external_allocator_manager.h"
#include "graph/utils/op_type_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/profiling_definitions.h"
#include "common/proto_util.h"
#include "graph/utils/math_util.h"
#include "graph/ge_global_options.h"
#include "common/checker.h"
#include "framework/runtime/rt_session.h"
#include "common/op_so_store/op_so_store_utils.h"
#include "common/global_variables/diagnose_switch.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/manager/mem_manager.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "graph/load/model_manager/kernel/kernel_register_info_builder.h"
#include "common/kernel_handles_manager/aicpu_kernel_handles_manager.h"

namespace ge {
namespace {
constexpr size_t kCmdParSize = 2U;
constexpr size_t kDumpCmdPairSize = 2U;
constexpr size_t kProfCmdParaMaxSize = 1000U;
constexpr size_t kProfStartCmdParaSize = 2U;
constexpr size_t kOpNameMaxSize = 100U;
constexpr size_t kKernelNameMaxSize = 100U;
constexpr size_t kSoNameMaxSize = 100U;
constexpr uint32_t kNullSoNameOffset = 0xFFFFFFFF;
constexpr uint16_t kNeverTimeout = 0xFFFF;
constexpr uint64_t kOfflineSessionId = 0U;
constexpr uint32_t kTriggerScanIntervalMs = 5U; // millisconds
constexpr uint32_t kRecordIntervalMs = 60000U; // millisconds
constexpr uint32_t kRecordTimes = 4U;

const std::string kCmdTypeDump = "dump";
const std::string kCmdTypeProfInit = "prof_init";
const std::string kCmdTypeProfFinalize = "prof_finalize";
const std::string kCmdTypeProfStart = "prof_start";
const std::string kCmdTypeProfStop = "prof_stop";
const std::string kCmdTypeProfModelSubscribe = "prof_model_subscribe";
const std::string kCmdTypeProfModelUnsubscribe = "prof_model_cancel_subscribe";
const std::string kStreamResource = "stream";
const std::string kEventResource = "event";
const std::string kIsCopyOutputAddr = "1";
const std::string kBatchLoadBuf = "batchLoadsoFrombuf";
const std::string kDeleteCustOp = "deleteCustOp";
const std::string kLoadBuiltinSo = "RunAicpuPreprocessLoadSoLaunch";
const std::string kUnloadBuiltinSo = "RunAicpuPreprocessUnloadSoLaunch";
const std::string kLibAicpuExtendKernelsSo = "libaicpu_extend_kernels.so";
const std::string kTriggerFile = "exec_record_trigger";
const std::string kRecordFilePrefix = "exec_record_";
const std::string kPathSeparator = "/";
constexpr const char_t *const kReloadDumpFuncName = "Load";
constexpr const char_t *const kUnloadDumpFuncName = "Unload";

#pragma pack(push, 1)
struct CustAicpuSoBuf {
  uint64_t kernelSoBuf;
  uint32_t kernelSoBufLen;
  uint64_t kernelSoName;
  uint32_t kernelSoNameLen;
};

struct BatchLoadOpFromBufArgs {
  uint32_t soNum;
  uint64_t args;
  char_t kernel_name[kKernelNameMaxSize];
};

struct LoadSoFromBufArgs {
  uint64_t kernelSoBuf;
  uint32_t kernelSoBufLen;
  uint64_t kernelSoName;
  uint32_t kernelSoNameLen;
  char_t so_name[kSoNameMaxSize];
  char_t kernel_name[kKernelNameMaxSize];
};
#pragma pack(pop)

Status LoadTask(const DumpProperties &dump_properties) {
  return ModelManager::GetInstance().LoadTaskForDavinciModel(dump_properties);
}

Status UnloadTask(const DumpProperties &dump_properties) {
  return ModelManager::GetInstance().UnloadTaskForDavinciModel(dump_properties);
}

void ConstructLoadSoFromBufArgs(const void *const d_aicpu_data, const size_t aicpu_data_len,
                                const void *const d_so_name, const size_t so_name_len, LoadSoFromBufArgs &so_buf) {
  so_buf.kernelSoBuf = PtrToValue(d_aicpu_data);
  so_buf.kernelSoBufLen = static_cast<uint32_t>(aicpu_data_len);
  so_buf.kernelSoName = PtrToValue(d_so_name);
  so_buf.kernelSoNameLen = static_cast<uint32_t>(so_name_len);
}

int32_t GetGraphMaxParallelModelNum() {
  int32_t max_num = -1;
  std::string opt;
  if (GetContext().GetOption(GRAPH_MAX_PARALLEL_MODEL_NUM, opt) == GRAPH_SUCCESS) {
    GE_ASSERT_SUCCESS(ge::ConvertToInt32(opt, max_num), "option %s, value %s is not int",
                      GetContext().GetReadableName(GRAPH_MAX_PARALLEL_MODEL_NUM).c_str(), opt.c_str());
  }
  GELOGI("graphMaxParallelModelNum is %d", max_num);
  return max_num;
}
}  // namespace

Status SetNetOutputTensorInfo(const GraphId &graph_id, const GraphNodePtr &graph_node) {
  if (graph_node->IsSavedNetOutputTensorInfoFlag()) {
    return SUCCESS;
  }
  auto compute_graph = graph_node->GetComputeGraph();
  auto net_output_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_node, "netoutput node null, graph_id:%u.", graph_id);

  auto ge_tensor_descs = net_output_node->GetOpDesc()->GetAllInputsDescPtr();
  for (auto &ge_tensor_desc : ge_tensor_descs) {
    size_t tensor_size = 0UL;
    GeShape shape(ge_tensor_desc->GetShape().GetDims());
    GE_ASSERT_SUCCESS(CalcTensorSizeByShape(shape, ge_tensor_desc->GetDataType(), tensor_size));
    graph_node->SetTensorSize(tensor_size);
    graph_node->SetGeTensorDescPtr(ge_tensor_desc);
  }
  graph_node->SetSavedNetOutputTensorInfoFlag(true);
  return SUCCESS;
}

Status MallocConstMemory(const GraphId &graph_id, const GraphNodePtr &graph_node,
                         const AllocatorPtr external_allocator) {
  // If the app is set, do not use external allocator
  if (graph_node->IsAppRefreshConstMemory()) {
    GELOGI("Const memory base has been set by app, graph_id:%u.", graph_id);
    return SUCCESS;
  }

  // const memory only set once
  auto const_mem = graph_node->GetConstMemoryBase();
  if (const_mem.first != nullptr) {
    GELOGI("Const memory base has been set by external allocator, graph_id:%u.", graph_id);
    return SUCCESS;
  }

  const size_t const_size = graph_node->GetConstMemoryBase().second;
  auto const_block = external_allocator->Malloc(const_size);
  GE_ASSERT((const_block != nullptr), "malloc const memory failed by external allocator, graph_id:%u.", graph_id);

  auto memory = const_block->GetAddr();
  GE_ASSERT((memory != nullptr), "Get const address failed by external allocator, graph_id:%u.", graph_id);

  graph_node->SetConstMemoryBase(memory, const_size);
  graph_node->SetConstMemBlock(const_block);

  GELOGI("Set graph const memory base success by external allocator, const_block:%p, memory:%p, size:%zu, graph_id:%u",
         const_block, memory, const_size, graph_id);
  return SUCCESS;
}

Status MallocFeatureMemory(const GraphId &graph_id, const uint32_t model_id, const GraphNodePtr &graph_node,
                           const AllocatorPtr external_allocator) {
  // If the app is set, do not use external allocator
  if (graph_node->IsAppRefreshFeatureMemory()) {
    GELOGI("Feature memory base has been set by app, graph_id:%u.", graph_id);
    return SUCCESS;
  }

  const bool is_refreshable = graph_node->IsFeatureBaseRefreshable();
  if (graph_node->GetLoadFlag() && (!is_refreshable)) {
    GELOGI("Feature memory base only set once when ge.featureBaseRefreshable disabled, graph_id:%u.", graph_id);
    return SUCCESS;
  }

  const size_t feature_mem_size = graph_node->GetFeatureMemoryBase().second;
  const size_t refreshable_feature_mem_size = graph_node->GetRefreshableFeatureMemoryBase().second;

  /*
   * 1 如果用户设置了refreshable_feature_memory，优先使用refreshable_feature_memory地址和size.
   * 2 其次如果设置了feature_memory，就使用feature_memory地址和size
   * 3 如果上面两个都没有设置，就使用feature_memory_size申请内存
   */
  size_t size = feature_mem_size;
  auto feature_mem = graph_node->GetFeatureMemoryBase();
  if (graph_node->GetRefreshableFeatureMemoryBase().first != nullptr) {
    size = refreshable_feature_mem_size;
    feature_mem = graph_node->GetRefreshableFeatureMemoryBase();
  }

  MemBlock *mem_block = nullptr;
  if (feature_mem.first != nullptr) {
    mem_block = external_allocator->MallocAdvise(size, const_cast<void *>(feature_mem.first));
    GE_ASSERT((mem_block != nullptr),
              "malloc advise feature memory failed by external allocator, graph_id:%u.", graph_id);
  } else {
    mem_block = external_allocator->Malloc(size);
    GE_ASSERT((mem_block != nullptr), "malloc feature memory failed by external allocator, graph_id:%u.", graph_id);
  }

  auto memory = mem_block->GetAddr();
  GE_ASSERT((memory != nullptr), "Get feature address failed by external allocator, graph_id:%u.", graph_id);

  if (graph_node->GetLoadFlag()) {
    GE_ASSERT_SUCCESS(ModelManager::GetInstance().UpdateFeatureMemoryBase(model_id, PtrToValue(memory), size),
                      "Failed to update feature memory base, graph_id = %u, model_id = %u", graph_id, model_id);
  }
  if (graph_node->GetRefreshableFeatureMemoryBase().first != nullptr) {
    graph_node->SetRefreshableFeatureMemoryBase(memory, size);
  } else {
    graph_node->SetFeatureMemoryBase(memory, size);
  }
  graph_node->SetFeatureMemBlock(mem_block);
  GELOGI("Update graph feature memory base success by external allocator, block:%p, memory:%p, size:%zu, graph_id:%u, "
         "refreshable_feature_mem_size:%zu, feature_mem_size:%zu",
         mem_block, memory, size, graph_id, refreshable_feature_mem_size, feature_mem_size);
  return SUCCESS;
}

Status CalcTensorSizeByShape(const GeShape &shape, DataType data_type, size_t &ret_tensor_size) {
  constexpr uint64_t kAlignBytes = 32U;
  auto shape_size = shape.GetShapeSize();
  int64_t cal_size = 0;
  if (data_type == ge::DT_STRING) {
    uint32_t type_size = 0U;
    GE_ASSERT_TRUE(ge::TypeUtils::GetDataTypeLength(data_type, type_size));
    if (ge::MulOverflow(shape_size, static_cast<int64_t>(type_size), cal_size)) {
      GELOGE(ge::GRAPH_FAILED, "[Calc][TensorSizeByShape] shape_size[%" PRId64 "] "
        "multiplied by type_size[%u] overflowed!", shape_size, type_size);
      return ge::FAILED;
    }
  } else {
    cal_size = ge::GetSizeInBytes(shape_size, data_type);
  }
  if (cal_size < 0) {
    GELOGE(ge::GRAPH_FAILED, "[Calc][TensorSizeByShape] shape_size[%" PRId64 "] data_type[%s] failed", shape_size,
           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ge::FAILED;
  }

  // 不可能溢出，因为ret最大值也只有int64的最大值
  ret_tensor_size = ge::RoundUp(static_cast<uint64_t>(cal_size), kAlignBytes) + kAlignBytes;
  return ge::SUCCESS;
}

Status CreateGertTensor(const GeTensorDescPtr ge_tensor_desc, gert::Tensor &gert_tensor)  {
  gert_tensor.MutableStorageShape().SetDimNum(ge_tensor_desc->GetShape().GetDimNum());
  for (size_t i = 0U; i < ge_tensor_desc->GetShape().GetMutableDims().size(); ++i) {
    gert_tensor.MutableStorageShape().SetDim(i, ge_tensor_desc->GetShape().GetDim(i));
  }

  gert_tensor.MutableOriginShape().SetDimNum(ge_tensor_desc->GetOriginShape().GetDimNum());
  for (size_t i = 0U; i < ge_tensor_desc->GetOriginShape().GetMutableDims().size(); ++i) {
    gert_tensor.MutableOriginShape().SetDim(i, ge_tensor_desc->GetOriginShape().GetDim(i));
  }

  gert_tensor.SetPlacement(gert::kOnDeviceHbm);
  gert_tensor.SetDataType(ge_tensor_desc->GetDataType());
  gert_tensor.SetOriginFormat(ge_tensor_desc->GetOriginFormat());
  gert_tensor.SetStorageFormat(ge_tensor_desc->GetFormat());

  return SUCCESS;
}

Status MallocOutputsMemory(const GraphId &graph_id, const GraphNodePtr &graph_node,
                           const AllocatorPtr external_allocator, std::vector<gert::Tensor> &outputs) {
  // If the app is set, do not use external allocator
  if (graph_node->IsAppRefreshFeatureMemory() || graph_node->IsAppRefreshConstMemory()) {
    GELOGI("Outputs memory base has been set by app, graph_id:%u.", graph_id);
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(SetNetOutputTensorInfo(graph_id, graph_node));
  auto ret_tensor_size = graph_node->GetTensorSize();
  auto ge_tensor_descs = graph_node->GetGeTensorDescPtr();
  GE_ASSERT((ret_tensor_size.size() == ge_tensor_descs.size()), "tensor info invalid, graph_id:%u.", graph_id);
  auto HbmManager = [](void *block, gert::TensorOperateType operate_type, void **out) -> ge::graphStatus {
    GE_ASSERT_NOTNULL(block);
    auto mem_block = reinterpret_cast<ge::MemBlock *>(block);
    GE_ASSERT((operate_type == gert::kGetTensorAddress || operate_type == gert::kFreeTensor ||
      operate_type == gert::kPlusShareCount), "Unexpected operate type %d", static_cast<int32_t>(operate_type));
    if (operate_type == gert::kGetTensorAddress) {
      GE_ASSERT_NOTNULL(out);
      *out = mem_block->GetAddr();
    }
    if (operate_type == gert::kFreeTensor) {
      mem_block->Free();
    }
    if (operate_type == gert::kPlusShareCount) {
      mem_block->AddCount();
    }
    return ge::GRAPH_SUCCESS;
  };
  // the outputs is empty
  if (outputs.size() == 0U) {
    outputs.resize(ge_tensor_descs.size());
    for (size_t i = 0UL; i < ge_tensor_descs.size(); i++) {
      auto &gert_tensor = outputs[i];
      CreateGertTensor(ge_tensor_descs[i], gert_tensor);
      auto mem_block = external_allocator->Malloc(ret_tensor_size[i]);
      GE_ASSERT((mem_block != nullptr), "malloc output memory failed by external allocator, graph_id:%u.", graph_id);
      GE_ASSERT((mem_block->GetAddr() != nullptr),
        "malloc output memory failed by external allocator, output memory addr is null, graph_id:%u.", graph_id);
      gert_tensor.SetData(gert::TensorData{mem_block, HbmManager, mem_block->GetSize(), gert_tensor.GetPlacement()});
      GELOGI("Alloc output memory success by external allocator, mem_block:%p, addr:%p, size:%u.", mem_block,
              mem_block->GetAddr(), ret_tensor_size[i]);
    }
    return SUCCESS;
  }

  GE_ASSERT((ret_tensor_size.size() == outputs.size()), "tensor info invalid, graph_id:%u.", graph_id);
  // just device memory don't alloc
  for (size_t i = 0UL; i < outputs.size(); i++) {
    if (outputs[i].GetSize() != 0 && outputs[i].GetAddr() != nullptr) {
      continue;
    }
    // for outputs not init
    if (outputs[i].GetFormat().GetOriginFormat() == FORMAT_RESERVED) {
      auto &gert_tensor = outputs[i];
      CreateGertTensor(ge_tensor_descs[i], gert_tensor);
    }
    auto mem_block = external_allocator->Malloc(ret_tensor_size[i]);
    GE_ASSERT((mem_block != nullptr), "malloc output memory failed by external allocator, graph_id:%u.", graph_id);
    GE_ASSERT((mem_block->GetAddr() != nullptr),
        "malloc output memory failed by external allocator, output memory addr is null, graph_id:%u.", graph_id);
    outputs[i].SetData(gert::TensorData{mem_block, HbmManager, mem_block->GetSize(), gert::kOnDeviceHbm});
    GELOGI("Alloc output memory success by external allocator, mem_block:%p, addr:%p, size:%u.", mem_block,
           mem_block->GetAddr(), ret_tensor_size[i]);
  }
  return SUCCESS;
}

Status MallocOutputsMemory(const GraphId &graph_id, const GraphNodePtr &graph_node,
                           const AllocatorPtr external_allocator, std::vector<GeTensor> &outputs) {
  // If the app is set, do not use external allocator
  if (graph_node->IsAppRefreshFeatureMemory() || graph_node->IsAppRefreshConstMemory()) {
    GELOGI("Outputs memory base has been set by app, graph_id:%u.", graph_id);
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(SetNetOutputTensorInfo(graph_id, graph_node));
  auto ret_tensor_size = graph_node->GetTensorSize();
  auto ge_tensor_descs = graph_node->GetGeTensorDescPtr();
  GE_ASSERT((ret_tensor_size.size() == ge_tensor_descs.size()), "tensor info invalid, graph_id:%u.", graph_id);

  // the outputs is empty
  if (outputs.size() == 0U) {
    for (size_t i = 0UL; i < ge_tensor_descs.size(); i++) {
      GeTensor ge_tensor(*(ge_tensor_descs[i]));
      auto mem_block = external_allocator->Malloc(ret_tensor_size[i]);
      GE_ASSERT((mem_block != nullptr), "malloc output memory failed by external allocator, graph_id:%u.", graph_id);
      GE_ASSERT((mem_block->GetAddr() != nullptr),
        "malloc output memory failed by external allocator, output memory addr is null, graph_id:%u.", graph_id);
      const auto deleter = [mem_block](uint8_t *device_data) {
        (void)device_data;
        GELOGI("Free output memory which alloc by external allocator, mem_block:%p, addr:%p.", mem_block,
               mem_block->GetAddr());
        mem_block->Free();
      };
      GE_ASSERT_SUCCESS(ge_tensor.SetData(static_cast<uint8_t *>(mem_block->GetAddr()), ret_tensor_size[i],
                                          deleter));
      GELOGI("Alloc output memory success by external allocator, mem_block:%p, addr:%p, size:%u.", mem_block,
             mem_block->GetAddr(), ret_tensor_size[i]);
      outputs.emplace_back(std::move(ge_tensor));
    }
    return SUCCESS;
  }

  GE_ASSERT((ret_tensor_size.size() == outputs.size()), "tensor info invalid, graph_id:%u.", graph_id);
  // just device memory don't alloc
  for (size_t i = 0UL; i < outputs.size(); i++) {
    if (outputs[i].GetData().GetSize() != 0) {
      continue;
    }
    auto mem_block = external_allocator->Malloc(ret_tensor_size[i]);
    GE_ASSERT((mem_block != nullptr), "malloc output memory failed by external allocator, graph_id:%u.", graph_id);
    GE_ASSERT((mem_block->GetAddr() != nullptr),
        "malloc output memory failed by external allocator, output memory addr is null, graph_id:%u.", graph_id);
    const auto deleter = [mem_block](uint8_t *device_data) {
      (void)device_data;
      GELOGI("Free output memory which alloc by external allocator, mem_block:%p, addr:%p.", mem_block,
             mem_block->GetAddr());
      mem_block->Free();
    };
    GE_ASSERT_SUCCESS(outputs[i].SetData(static_cast<uint8_t *>(mem_block->GetAddr()), ret_tensor_size[i], deleter));
    GELOGI("Alloc output memory success by external allocator, mem_block:%p, addr:%p, size:%u.", mem_block,
           mem_block->GetAddr(), ret_tensor_size[i]);
  }
  return SUCCESS;
}

void FreeFeatureMemory(const ge::GraphNodePtr &graph_node) {
  auto mem_block = graph_node->GetFeatureMemBlock();
  if (mem_block == nullptr) {
    return;
  }
  mem_block->Free();
  graph_node->SetFeatureMemBlock(nullptr);
  return;
}

std::string ModelManager::record_file_name_;
std::mutex ModelManager::weights_mem_mtx_;
std::unordered_map<std::string, ModelManager::SharedWeightAddrInfo>
    ModelManager::weights_mem_ids_to_addr_info_;

ModelManager &ModelManager::GetInstance() {
  static const std::shared_ptr<ModelManager> instance_ptr =
      std::shared_ptr<ModelManager>(new (std::nothrow) ModelManager, [](ModelManager *){});
  // 老代码可能存在单例析构顺序问题，所以使用shared_ptr保证不提前释放。
  // 下面增加空指针校验，instance_ptr为空指针时，再返回局部静态变量（这应该算极端场景）。
  CHECK_FALSE_EXEC(instance_ptr != nullptr, static ModelManager instance; return instance);
  return *instance_ptr;
}

ModelManager::ModelManager() {
  CreateMonitorThread();
}

Status ModelManager::KernelLaunchEx(const aicpu::FWKAdapter::FWKOperateType op_type, const uint64_t session_id,
                                    const uint32_t model_id, const uint32_t sub_model_id) {
  rtStream_t stream = nullptr;
  std::vector<void *> allocated_mem;
  GE_MAKE_GUARD(kernel_launch_release, [&]() {
    for (auto &mem : allocated_mem) {
      GE_CHK_RT(rtFree(mem));
    }
    if (stream != nullptr) {
      GE_CHK_RT(rtStreamDestroy(stream));
    }
  });

  STR_FWK_OP_KERNEL param_base{};
  constexpr uint32_t kKernelType = 0U;
  param_base.fwkKernelType = kKernelType;
  param_base.fwkKernelBase.fwk_kernel.opType = op_type;
  param_base.fwkKernelBase.fwk_kernel.sessionID = session_id;
  if (op_type == aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY) {
    const std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id) + "_" +
                                  std::to_string(sub_model_id);
    const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
    const auto iter = model_aicpu_kernel_.find(model_key);
    if (iter != model_aicpu_kernel_.end()) {
      GELOGD("kernel destroy session_id %" PRIu64 ", model_id %u, sub_model_id %u.", session_id, model_id, sub_model_id);
      std::vector<uint64_t> v_aicpu_kernel = model_aicpu_kernel_.at(model_key);
      // Insert size of aicpu kernel vector in the first element
      (void)v_aicpu_kernel.insert(v_aicpu_kernel.cbegin(), v_aicpu_kernel.size());

      const uint64_t kernel_size = sizeof(uint64_t) * (v_aicpu_kernel.size());
      void *aicpu_kernel_addr = nullptr;
      GE_CHK_RT_RET(rtMalloc(&aicpu_kernel_addr, kernel_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
      allocated_mem.emplace_back(aicpu_kernel_addr);

      GE_CHK_RT_RET(rtMemcpy(aicpu_kernel_addr, kernel_size, v_aicpu_kernel.data(), kernel_size,
                             RT_MEMCPY_HOST_TO_DEVICE));
      param_base.fwkKernelBase.fwk_kernel.kernelID = PtrToValue(aicpu_kernel_addr);
      // In the scene of loading once and running many times, the kernel needs to be destroyed many times,
      // and connot be removed from kernel map.
    }
  }

  void *device_base = nullptr;
  constexpr size_t op_kernel_size = sizeof(STR_FWK_OP_KERNEL);
  GE_CHK_RT_RET(rtMalloc(&device_base, op_kernel_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  allocated_mem.emplace_back(device_base);
  GE_CHK_RT_RET(rtMemcpy(device_base, op_kernel_size, &param_base, op_kernel_size, RT_MEMCPY_HOST_TO_DEVICE));

  GE_CHK_RT_RET(rtStreamCreate(&stream, 0));
  KernelRegisterInfo register_info;
  GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructAicpuRegisterInfo("TfSessionTask",
      "libtf_kernels.so", "TFOperateAPI", "TFKernel", register_info));
  AicpuKernelHandlesManager aicpu_kernel_handles_manager;
  const auto bin_name = aicpu_kernel_handles_manager.GenerateKey(register_info);
  auto bin_handle = aicpu_kernel_handles_manager.GetOrRegisterKernel(register_info, bin_name);
  GE_ASSERT_NOTNULL(bin_handle);
  auto func_handle = KernelHandleUtils::GetFuncHandle(bin_handle, "TfSessionTask");
  GE_ASSERT_NOTNULL(func_handle);

  LaunchKernelParam launch_kernel_param;
  launch_kernel_param.args = device_base;
  launch_kernel_param.args_size = op_kernel_size;
  launch_kernel_param.block_dim = 1U;
  launch_kernel_param.stream = stream;
  GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(func_handle, launch_kernel_param));
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  return SUCCESS;
}

Status ModelManager::DestroyAicpuSessionForDevice(const uint64_t session_id,
                                                  const uint32_t device_id,
                                                  const bool need_set_device) {
  GELOGI("DestroyAicpuSession device id:%u", device_id);
  if (need_set_device) {
    GELOGI("Set device %u.", device_id);
    GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(device_id)));
  }

  const auto ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_SESSION_DESTROY, session_id, 0U, 0U);
  if (ret != SUCCESS) {
    GELOGW("The session:%" PRIu64 " destroy failed, device id:%u", session_id, device_id);
  }

  if (need_set_device) {
    GELOGI("Reset device %u.", device_id);
    GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(device_id)));
  }
  return ret;
}

void ModelManager::DestroyAicpuSession(const uint64_t session_id, const bool single_device, const uint32_t device_id) {
  // when model execute failed because socket close, destroy aicpu session will block all process, must skip
  if (IsSocketClose()) {
    GELOGI("socket is closed, skip destroy aicpu session, session_id:%" PRIu64, session_id);
    return;
  }
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  const auto &iter = sess_id_to_device_ids_.find(session_id);
  if (iter == sess_id_to_device_ids_.cend()) {
    GELOGI("The session : %" PRIu64 " not created", session_id);
    return;
  }
  rtContext_t ctx = nullptr;
  const bool has_ctx = (rtCtxGetCurrent(&ctx) == RT_ERROR_NONE);
  if (single_device) {
    if (iter->second.find(device_id) == iter->second.cend()) {
      GELOGI("The session:%" PRIu64 " not create on device:%u", session_id, device_id);
      return;
    }
    if (DestroyAicpuSessionForDevice(session_id, device_id, !has_ctx) == SUCCESS) {
      GELOGI("The session: %" PRIu64 " destroyed, device id:%u", session_id, device_id);
      (void) iter->second.erase(device_id);
      if (iter->second.empty()) {
        (void) sess_id_to_device_ids_.erase(iter);
      }
    }
  } else {
    for (auto it = iter->second.cbegin(); it != iter->second.cend();) {
      if (DestroyAicpuSessionForDevice(session_id, *it, true) == SUCCESS) {
        GELOGI("The session: %" PRIu64 " destroyed, device id:%u", session_id, *it);
        it = iter->second.erase(it);
      } else {
        ++it;
      }
    }
    if (iter->second.empty()) {
      (void) sess_id_to_device_ids_.erase(iter);
    }
    if (has_ctx) {
      (void) rtCtxSetCurrent(ctx);
    }
  }
}

Status ModelManager::DestroyAicpuSessionForInfer(const uint32_t model_id) {
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  const auto hybrid_it = hybrid_model_map_.find(model_id);
  if (hybrid_it != hybrid_model_map_.end()) {
    DestroyAicpuSession(hybrid_it->second->GetSessionId(), true, hybrid_it->second->GetDeviceId());
    return SUCCESS;
  }

  const auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Param model_id:%u can't find in model_map, check invalid", model_id);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "[Check][Param] model id %u does not exist.", model_id);
    return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
  }

  DestroyAicpuSession(it->second->GetSessionId(), true, it->second->GetDeviceId());
  return SUCCESS;
}

Status ModelManager::DestroyAicpuKernel(const uint64_t session_id, const uint32_t model_id,
                                        const uint32_t sub_model_id) {
  GELOGD("destroy aicpu kernel in session id %" PRIu64 ", model_id %u.", session_id, model_id);
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  const std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id) + "_" +
                                std::to_string(sub_model_id);
  if (model_aicpu_kernel_.find(model_key) != model_aicpu_kernel_.end()) {
    const Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY, session_id, model_id,
                                      sub_model_id);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call KernelLaunchEx fail, model_id:%u, sub_model_id:%u, session_id:%" PRIu64 "",
                        model_id, sub_model_id, session_id);
      GELOGE(FAILED, "[Call][KernelLaunchEx] fail, model_id:%u, sub_model_id:%u, session_id:%" PRIu64,
             model_id, sub_model_id, session_id);
      return FAILED;
    }
  }
  return SUCCESS;
}

void ModelManager::CreateAicpuKernel(const uint64_t session_id, const uint32_t model_id, const uint32_t sub_model_id,
                                     const uint64_t kernel_id) {
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  const std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id) + "_" +
                                std::to_string(sub_model_id);
  const auto it = model_aicpu_kernel_.find(model_key);
  if (it != model_aicpu_kernel_.cend()) {
    it->second.push_back(kernel_id);
  } else {
    model_aicpu_kernel_[model_key] = { kernel_id };
  }
}

ModelManager::~ModelManager() {
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  model_map_.clear();
  hybrid_model_map_.clear();
  model_aicpu_kernel_.clear();
  sess_id_to_device_ids_.clear();
  cust_aicpu_so_.clear();
  cust_op_master_so_names_to_bin_.clear();
  cust_op_master_so_names_to_unique_name_.clear();
  cust_op_master_so_datas_to_name_.clear();
  builtin_aicpu_so_.clear();
  built_in_op_master_so_names_to_bin_.clear();
  ClearMonitorThread();
}

Status ModelManager::GetRuntimeModelId(const uint32_t model_id, uint32_t &model_runtime_id) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHECK_NOTNULL(davinci_model);
  model_runtime_id = davinci_model->GetRuntimeModelId();
  return SUCCESS;
}

Status ModelManager::SetDynamicSize(const uint32_t model_id, const std::vector<uint64_t> &batch_num,
                                    const int32_t dynamic_type) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->SetDynamicSize(batch_num, dynamic_type);
  return SUCCESS;
}

Status ModelManager::DoLoadHybridModelOnline(const uint32_t model_id,
                                             const ModelData &model,
                                             const uint32_t device_id,
                                             const GeRootModelPtr &ge_root_model,
                                             const std::shared_ptr<ModelListener> &listener,
                                             const rtStream_t stream) {
  auto hybrid_model = hybrid::HybridDavinciModel::Create(ge_root_model);
  GE_CHECK_NOTNULL(hybrid_model);
  hybrid_model->SetListener(listener);
  hybrid_model->SetModelId(model_id);
  hybrid_model->SetDeviceId(device_id);
  hybrid_model->SetOmName(model.om_name);
  GE_CHECK_NOTNULL(ge_root_model);
  hybrid_model->SetFileConstantWeightDir(ge_root_model->GetFileConstantWeightDir());
  hybrid_model->SetLoadStream(stream);
  GE_CHK_STATUS_RET(hybrid_model->Init(), "[Init][HybridModel] failed. model_id = %u", model_id);
  const auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model.release());
  InsertModel(model_id, shared_model);
  return SUCCESS;
}

bool ModelManager::IsNeedHybridLoad(const GeRootModel &ge_root_model) const {
  const auto root_graph = ge_root_model.GetRootGraph();
  GE_RT_FALSE_CHECK_NOTNULL(root_graph);

  bool is_dsp_partitioned_graph = false;
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dsp_partitioned_graph);
  return root_graph->GetGraphUnknownFlag() || is_dsp_partitioned_graph || GetContext().GetHostExecFlag();
}

///
/// @ingroup domi_ome
/// @brief load model online
/// @return Status run result
///
Status ModelManager::LoadModelOnline(uint32_t &model_id, const GeRootModelPtr &ge_root_model,
                                     const GraphNodePtr &graph_node, const uint32_t device_id,
                                     const rtStream_t stream) {
  std::shared_ptr<ModelListener> listener;
  if (graph_node->IsAsync()) {
    listener = MakeShared<RunAsyncListener>();
  } else {
    listener = MakeShared<GraphModelListener>();
  }
  GE_CHK_BOOL_RET_STATUS(listener.get() != nullptr, PARAM_INVALID, "[Check][Param] Param incorrect, listener is null");
  if (model_id == INVALID_MODEL_ID) {
    GenModelId(model_id);
    GELOGD("Generate new model_id:%u", model_id);
  }

  ProfilingManager::Instance().SetGraphIdToModelMap(graph_node->GetGraphId(), model_id);
  GELOGI("Set graph id to model map, graph id: %u, model id: %u.", graph_node->GetGraphId(), model_id);

  domi::GetContext().is_online_model = true;

  GE_ASSERT_SUCCESS(InitOpMasterDeviceSo(model_id, ge_root_model), "Init model [%u] op master device failed", model_id);

  if (IsNeedHybridLoad(*ge_root_model)) {
    const ModelData model_data;
    return DoLoadHybridModelOnline(model_id, model_data, device_id, ge_root_model, listener, stream);
  }
  GE_ASSERT_SUCCESS(ExternalAllocatorMalloc(graph_node->GetGraphId(), model_id, graph_node, stream));
  const auto &root_graph = ge_root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);

  const auto &name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
  const auto it = name_to_model.find(root_graph->GetName());
  const GeModelPtr ge_model = (it != name_to_model.end()) ? it->second : nullptr;
  GE_CHECK_NOTNULL(ge_model);

  const auto davinci_model = MakeShared<DavinciModel>(0, listener);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->SetId(model_id);
  davinci_model->SetDeviceId(device_id);
  davinci_model->SetRunContext(graph_node->GetOmeContext());
  davinci_model->SetFeatureBaseRefreshable(graph_node->IsFeatureBaseRefreshable());

  GE_TIMESTAMP_START(Assign);
  davinci_model->Assign(ge_model);
  GE_TIMESTAMP_END(Assign, "GraphLoader::ModelAssign");
  const uint64_t session_id = GetContext().SessionId();

  const DumpProperties &dump_properties = DumpManager::GetInstance().GetDumpProperties(session_id);
  davinci_model->SetDumpProperties(dump_properties);
  {
    const std::lock_guard<std::mutex> lk(dump_properties_mutex_);
    dump_properties_ = dump_properties;
  }

  GE_CHK_STATUS_RET(davinci_model->InitSpaceRegistry(ge_root_model), "Get space registry failed!");
  GE_TIMESTAMP_START(Init);
  const auto feature_mem = graph_node->GetFeatureMemoryBase();
  const auto const_mem = graph_node->GetConstMemoryBase();
  const auto refreshable_feature_mem = graph_node->GetRefreshableFeatureMemoryBase();
  ModelParam param{};
  if (refreshable_feature_mem.first != nullptr) {
    // fixed Feature Memory addr set
    param.mem_base = reinterpret_cast<uintptr_t>(refreshable_feature_mem.first);
    param.mem_size = refreshable_feature_mem.second;
    GELOGI("use refreshable feature memory");
  } else {
    // fixed Feature Memory addr not set, update feature mem
    param.mem_base = reinterpret_cast<uintptr_t>(feature_mem.first);
    param.mem_size = feature_mem.second;
  }
  const auto &fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  const auto &hbm_fixed_mem_iter = fixed_feature_mem.find(RT_MEMORY_HBM);
  const auto &p2p_fixed_mem_iter = fixed_feature_mem.find(RT_MEMORY_P2P_DDR);
  if (hbm_fixed_mem_iter != fixed_feature_mem.end()) {
    param.fixed_mem_base = reinterpret_cast<uintptr_t>(hbm_fixed_mem_iter->second.addr);
    param.fixed_mem_size = hbm_fixed_mem_iter->second.size;
  }
  if (p2p_fixed_mem_iter != fixed_feature_mem.end()) {
    param.p2p_fixed_mem_base = reinterpret_cast<uintptr_t>(p2p_fixed_mem_iter->second.addr);
    param.p2p_fixed_mem_size = p2p_fixed_mem_iter->second.size;
  }

  param.weight_base = reinterpret_cast<uintptr_t>(const_mem.first);
  param.weight_size = const_mem.second;
  GELOGI("param init: mem_base:[0x%llx], mem_size[%zu], weight_base:[0x%llx], weight_size[%zu],"
         " fixed_feature_memory[0x%llx], fixed_mem_size[%zu], p2p_fixed_feature_memory[0x%llx],"
         " p2p_fixed_mem_size[%zu].", param.mem_base, param.mem_size, param.weight_base, param.weight_size,
         param.fixed_mem_base, param.fixed_mem_size, param.p2p_fixed_mem_base, param.p2p_fixed_mem_size);
  davinci_model->SetNoFrozenInputIndexes(graph_node->GetFrozenInputIndex());
  const Status result = davinci_model->Init(param);
  if (result != SUCCESS) {
    GELOGE(result, "DavinciModel Init failed.");
    return result;
  }
  GE_TIMESTAMP_END(Init, "GraphLoader::ModelInit");

  InsertModel(model_id, davinci_model);
  GELOGI("Parse model %u success.", model_id);

  return SUCCESS;
}


void ModelManager::InsertModel(const uint32_t model_id, const std::shared_ptr<DavinciModel> &davinci_model) {
  GE_CHK_BOOL_EXEC(davinci_model != nullptr, return, "[Check][Param] davinci_model ptr is null, id:%u", model_id);
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  model_map_[model_id] = davinci_model;
}

void ModelManager::InsertModel(const uint32_t model_id, const shared_ptr<hybrid::HybridDavinciModel> &hybrid_model) {
  GE_CHK_BOOL_EXEC(hybrid_model != nullptr, return, "[Check][Param] hybrid_model ptr is null, id:%u", model_id);
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  hybrid_model_map_[model_id] = hybrid_model;
}

Status ModelManager::DeleteModel(const uint32_t id) {
  // 使用临时变量延长davinci model对象的生命周期，在lk析构后再做davinci model对象的析构，较少时间占用锁资源
  std::shared_ptr<DavinciModel> davinci_model_delay_destruction;
  std::shared_ptr<hybrid::HybridDavinciModel> hybrid_davinci_model_delay_destruction;
  // These two pointers are used to unbind erase() and model destruction process.
  {
    const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
    const auto it = model_map_.find(id);
    if (it != model_map_.end()) {
      const uint64_t session_id = it->second->GetSessionId();
      const std::string model_key = std::to_string(session_id) + "_" + std::to_string(id)  + "_" +
                                    std::to_string(it->second->SubModelId());
      (void)model_aicpu_kernel_.erase(model_key);
      davinci_model_delay_destruction = it->second;
      (void)model_map_.erase(it);
      return SUCCESS;
    }

    const auto hybrid_model_it = hybrid_model_map_.find(id);
    if (hybrid_model_it != hybrid_model_map_.end()) {
      hybrid_davinci_model_delay_destruction = hybrid_model_it->second;
      (void)hybrid_model_map_.erase(hybrid_model_it);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "model_id:%u does not exist in model_map, check invalid", id);
      GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "model id %u does not exist.", id);
      return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
    }
  }
  DeleteSharedSessionModel(id);
  return SUCCESS;
}

std::shared_ptr<DavinciModel> ModelManager::GetModel(const uint32_t id) {
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);

  const auto it = model_map_.find(id);
  return (it == model_map_.end()) ? nullptr : it->second;
}

Status ModelManager::RecoverAllModel(const int32_t device_id) {
  // 校验device_id是否正确
  int32_t current_device_id = -1;
  GE_ASSERT_RT_OK(rtGetDevice(&current_device_id));
  GE_ASSERT_TRUE(current_device_id == device_id);
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);

  // 只支持在线场景
  GE_ASSERT_TRUE(domi::GetContext().is_online_model);

  // 只处理静态图
  for (auto &model : model_map_) {
    const auto davinci_model = model.second;
    GE_ASSERT_SUCCESS(davinci_model->RecoverModel());
  }

  return SUCCESS;
}

std::shared_ptr<hybrid::HybridDavinciModel> ModelManager::GetHybridModel(const uint32_t id) {
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);

  const auto it = hybrid_model_map_.find(id);
  return (it == hybrid_model_map_.end()) ? nullptr : it->second;
}

bool ModelManager::IsModelSharedSession(const uint32_t model_id) {
  const std::lock_guard<std::mutex> lk(model_shared_session_mutex_);
  return (model_shared_session_.find(model_id) != model_shared_session_.end());
}

void ModelManager::AddSharedSessionModel(const uint32_t model_id) {
  const std::lock_guard<std::mutex> lk(model_shared_session_mutex_);
  model_shared_session_.insert(model_id);
}
void ModelManager::DeleteSharedSessionModel(const uint32_t model_id) {
  const std::lock_guard<std::mutex> lk(model_shared_session_mutex_);
  (void)model_shared_session_.erase(model_id);
}

Status ModelManager::Unload(const uint32_t model_id) {
  GE_CHK_STATUS_RET(DeleteModel(model_id), "[Delete][Model] failed, model id:%u", model_id);
  GELOGI("Unload model %u success.", model_id);

  if (!domi::GetContext().is_online_model) {
    (void)MsprofUnsetDeviceIdByGeModelIdx(model_id, 0U);
  }

  ProfilingManager::Instance().RemoveFromGraphIdMap(model_id);
  return SUCCESS;
}

Status ModelManager::SyncExecuteModel(const uint32_t model_id, const std::vector<gert::Tensor> &inputs,
                                      std::vector<gert::Tensor> &outputs) {
  GELOGI("SyncExecuteModel execute in, model_id:[%u].", model_id);
  const auto &model = GetModel(model_id);
  GE_CHECK_NOTNULL(model);
  const auto device_id = model->GetDeviceId();
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(device_id)));
  GE_MAKE_GUARD(reset_device, [device_id]() {
    GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(device_id)));
  });
  GE_CHK_STATUS_RET(model->NnExecute(nullptr, false, inputs, outputs));
  GELOGI("Execute model %u success, device id is %d.", model_id, device_id);
  GELOGI("UpdateOutputTensorShape for model:[%u] success.", model_id);
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief load Input and output TensorInfo for Model
/// @return Status run result
/// @author
///
Status ModelManager::DataInputTensor(const uint32_t model_id, const std::shared_ptr<RunArgs> &args) {
  const auto &hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    GE_CHECK_NOTNULL(args);
    GE_CHK_STATUS_RET(hybrid_model->EnqueueData(args),
                    "[Enqueue][Data] Data queue is full, please call again later, model_id:%u", model_id);
    return SUCCESS;
  }
  const auto &model = GetModel(model_id);
  GE_CHECK_NOTNULL(model);
  GE_CHK_STATUS_EXEC(model->Push(args), return domi::DATA_QUEUE_ISFULL,
                     "[Call][Push] Data queue is full, please call again later, model_id:%u", model_id);

  GELOGD("Data input success, model_id:%u", model_id);
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief create model thread, start to execute model
/// @param [in] model_id Model ID to be started
/// @return Status model run result
/// @author
///
Status ModelManager::Start(const uint32_t model_id) {
  const auto &hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(hybrid_model->ModelRunStart());
    GELOGI("Start hybrid model %u success.", model_id);
    return SUCCESS;
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u to start! ", model_id);

  const auto status = davinci_model->ModelRunStart();
  if (status == SUCCESS) {
    GELOGI("Start model %u success.", model_id);
  }

  return status;
}

///
/// @ingroup domi_ome
/// @brief Model ID stop
/// @only when unloaded
/// @param [in] model_id Model ID to be stopped
/// @return Status model stop result
/// @author
///
Status ModelManager::Stop(const uint32_t model_id) {
  const auto &hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(hybrid_model->ModelRunStop());
    GELOGI("Stop hybrid model %u success.", model_id);
    return SUCCESS;
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u to stop!", model_id);

  const auto status = davinci_model->ModelRunStop();
  if (status == SUCCESS) {
    GELOGI("Stop model %u success.", model_id);
  }

  return status;
}

///
/// @ingroup domi_ome
/// @brief Command handle
/// @iterator 1 only Ieference, Debug 2 modes
/// @param [in] command command to handle
/// @return Status command handle result
/// @author
///
Status ModelManager::HandleCommand(const Command &cmd_info) {
  static const std::map<std::string, std::function<uint32_t(const Command &)>> cmds = {
      {kCmdTypeDump, &HandleDumpCommand}, {kCmdTypeProfInit, &HandleProfInitCommand},
      {kCmdTypeProfFinalize, &HandleProfFinalizeCommand}, {kCmdTypeProfStart, &HandleProfStartCommand},
      {kCmdTypeProfStop, &HandleProfStopCommand},
      {kCmdTypeProfModelSubscribe, &HandleProfModelSubscribeCommand},
      {kCmdTypeProfModelUnsubscribe, &HandleProfModelUnsubscribeCommand}};

  const auto iter = cmds.find(cmd_info.cmd_type);
  if (iter == cmds.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported command:%s check", cmd_info.cmd_type.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Unsupported command:%s", cmd_info.cmd_type.c_str());
    return PARAM_INVALID;
  } else {
    return iter->second(cmd_info);
  }
}

Status ModelManager::GetModelIdByCmd(const Command &cmd_info, uint32_t &model_id) {
  if (cmd_info.cmd_params.size() < kCmdParSize) {
    REPORT_INNER_ERR_MSG("E19999", "Command::cmd_params.size:%zu < kCmdParSize:%zu, Command::cmd_type:%s, check invalid",
                       cmd_info.cmd_params.size(), kCmdParSize, cmd_info.cmd_type.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is '%s', the size of cmd_params must larger than 2.",
           cmd_info.cmd_type.c_str());
    return PARAM_INVALID;
  }

  if (cmd_info.cmd_params[0U] != PROFILE_MODEL_ID) {
    REPORT_INNER_ERR_MSG("E19999", "Fisrt cmd_param not %s, check invalid", PROFILE_MODEL_ID.c_str());
    GELOGE(FAILED, "[Check][Param] The model_id parameter is not found in the command.");
    return FAILED;
  }

  int32_t tmp_value = 0;
  GE_CHK_STATUS_RET_NOLOG(ConvertToInt32(cmd_info.cmd_params[1U], tmp_value));
  model_id = static_cast<uint32_t>(tmp_value);
  return SUCCESS;
}

Status ModelManager::HandleProfModelSubscribeCommand(const Command &cmd_info) {
  uint32_t model_id = std::numeric_limits<uint32_t>::max();
  const Status ret = GetModelIdByCmd(cmd_info, model_id);
  if (ret != SUCCESS) {
    return ret;
  }

  // set graph id to UINT32_MAX to determine whether execution
  return ModelManager::GetInstance().ProfModelSubscribe(cmd_info.module_index, model_id, UINT32_MAX);
}

Status ModelManager::HandleProfModelUnsubscribeCommand(const Command &cmd_info) {
  uint32_t model_id = std::numeric_limits<uint32_t>::max();
  const Status ret = GetModelIdByCmd(cmd_info, model_id);
  if (ret != SUCCESS) {
    return ret;
  }

  const auto hybrid_davinci_model = ModelManager::GetInstance().GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return ProfilingManager::Instance().ProfModelUnsubscribe(hybrid_davinci_model->GetDeviceId(), model_id);
  }

  const auto davinci_model = ModelManager::GetInstance().GetModel(model_id);
  if (davinci_model != nullptr) {
    return ProfilingManager::Instance().ProfModelUnsubscribe(davinci_model->GetDeviceId(), model_id);
  }
  return FAILED;
}

Status ModelManager::HandleProfInitCommand(const Command &cmd_info) {
  if (profiling::ProfilingContext::IsDumpToStdEnabled()) {
    return SUCCESS;
  }

  const uint64_t module_index = cmd_info.module_index;
  if (ProfilingManager::Instance().ProfInit(module_index) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfInit] failed, module_index:%" PRIu64 ".", module_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelManager::HandleProfFinalizeCommand(const Command &cmd_info) {
  if (ProfilingManager::Instance().ProfFinalize() != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfFinalize] failed, moduld index: %" PRIu64 ".", cmd_info.module_index);
    return FAILED;
  }
  return SUCCESS;
}
/*
 * cmd para when prof start
 * "devNums:2"
 * "devIdList:1,2"
 * "profilingOption:PROF_OP_TRACE"
 * "aicoreMetrics:AICORE_ARITHMATIC_THROUGHPUT"
 */
Status ModelManager::HandleProfStartCommand(const Command &cmd_info) {
  if (profiling::ProfilingContext::IsDumpToStdEnabled()) {
    diagnoseSwitch::EnableGeHostProfiling();
    profiling::ProfilingContext::GetInstance().SetEnable();
    GELOGI("GE host profiling is on");
    return SUCCESS;
  }

  if (cmd_info.cmd_params.size() < kProfStartCmdParaSize) {
    REPORT_INNER_ERR_MSG("E19999", "Command::cmd_params.size:%zu < %zu, check invalid",
                       cmd_info.cmd_params.size(), kProfStartCmdParaSize);
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is 'profile start', "
           "the size:%zu of cmd_params must larger than 2.", cmd_info.cmd_params.size());
    return PARAM_INVALID;
  }
  if (cmd_info.cmd_params.size() > kProfCmdParaMaxSize) {
    REPORT_INNER_ERR_MSG("E19999", "Command::cmd_params.size:%zu > %zu, check invalid",
                       cmd_info.cmd_params.size(), kProfCmdParaMaxSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Command param size[%zu] larger than max[1000].", cmd_info.cmd_params.size());
    return PARAM_INVALID;
  }

  std::map<std::string, std::string> cmd_params_map;
  constexpr uint32_t step = 2U;
  for (size_t i = 0U; i < cmd_info.cmd_params.size(); i += step) {
    if ((i + 1U) >= cmd_info.cmd_params.size()) {
      continue;
    }
    cmd_params_map[cmd_info.cmd_params[i]] = cmd_info.cmd_params[i + 1U];
  }
  const uint64_t module_index = cmd_info.module_index;
  if (ProfilingManager::Instance().ProfStartProfiling(module_index, cmd_params_map, cmd_info.cache_flag) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfStartProfiling] failed, module_index:%" PRIu64 ".", module_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelManager::HandleProfStopCommand(const Command &cmd_info) {
  if (profiling::ProfilingContext::IsDumpToStdEnabled()) {
    profiling::ProfilingContext::GetInstance().DumpToStdOut();
    profiling::ProfilingContext::GetInstance().Reset();
    profiling::ProfilingContext::GetInstance().SetDisable();
    diagnoseSwitch::DisableProfiling();
    return SUCCESS;
  }

  if (cmd_info.cmd_params.size() < kProfStartCmdParaSize) {
    REPORT_INNER_ERR_MSG("E19999", "Command::cmd_params.size:%zu < %zu, check invalid",
                       cmd_info.cmd_params.size(), kProfStartCmdParaSize);
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is 'profile stop', "
           "the size:%zu of cmd_params must larger than 2.", cmd_info.cmd_params.size());
    return PARAM_INVALID;
  }
  if (cmd_info.cmd_params.size() > kProfCmdParaMaxSize) {
    REPORT_INNER_ERR_MSG("E19999", "Command::cmd_params.size:%zu > %zu, check invalid",
                       cmd_info.cmd_params.size(), kProfCmdParaMaxSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Command param size[%zu] larger than max[1000].", cmd_info.cmd_params.size());
    return PARAM_INVALID;
  }

  std::map<std::string, std::string> cmd_params_map;
  constexpr uint32_t step = 2U; // cmd params saved as: { key1, val1, key2, val2, key3, val3 }
  for (size_t i = 0U; i < cmd_info.cmd_params.size(); i += step) {
    if ((i + 1U) >= cmd_info.cmd_params.size()) {  // +1 for value.
      break;
    }
    cmd_params_map[cmd_info.cmd_params[i]] = cmd_info.cmd_params[i + 1U];
  }
  const uint64_t module_index = cmd_info.module_index;
  if (ProfilingManager::Instance().ProfStopProfiling(module_index, cmd_params_map) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfStopProfiling] failed, module_index:%" PRIu64 ".", module_index);
    return FAILED;
  }
  return SUCCESS;
}

static Status ParserPara(const Command &cmd_info, const std::string &dump_key, std::string &dump_value) {
  auto iter = std::find(cmd_info.cmd_params.begin(), cmd_info.cmd_params.end(), dump_key);
  if (iter != cmd_info.cmd_params.end()) {
    ++iter; // cmd params saved as: { key1, val1, key2, val2, key3, val3 }
    if (iter == cmd_info.cmd_params.end()) {
      REPORT_INNER_ERR_MSG("E19999", "dump_key:%s can't find in Command::cmd_param, check invalid", dump_key.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] dump_key:%s can't find in cmd_param, check invalid", dump_key.c_str());
      return PARAM_INVALID;
    }
    dump_value = *iter;
  }
  return SUCCESS;
}

Status ModelManager::HandleDumpCommand(const Command &cmd_info) {
  if ((cmd_info.cmd_params.size() % kDumpCmdPairSize) != 0U) {
    REPORT_INNER_ERR_MSG("E19999", "Command::cmd_params.size:%zu MOD 2 != 0, check invalid", cmd_info.cmd_params.size());
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is 'dump', "
           "the size:%zu of cmd_params must be a even number.", cmd_info.cmd_params.size());
    return PARAM_INVALID;
  }

  std::string dump_status("off");
  std::string dump_model(DUMP_ALL_MODEL);
  std::string dump_path("/");
  std::string dump_mode("output");
  std::set<std::string> dump_layers;

  auto ret = ParserPara(cmd_info, DUMP_STATUS, dump_status);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpStatus] failed, ret:%d", ret);
    return FAILED;
  }
  GELOGI("dump status = %s.", dump_status.c_str());

  ret = ParserPara(cmd_info, DUMP_MODEL, dump_model);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpModel] failed, ret:%d", ret);
    return FAILED;
  }
  GELOGI("dump model = %s.", dump_model.c_str());

  auto &ref_dump_properties = ModelManager::GetInstance().dump_properties_;
  if ((dump_status == "off") || (dump_status == "OFF")) {
    ref_dump_properties.DeletePropertyValue(dump_model);
    return SUCCESS;
  }

  for (size_t i = 0U; i < (cmd_info.cmd_params.size() / kDumpCmdPairSize); ++i) {
    if (cmd_info.cmd_params.at(i * kDumpCmdPairSize).find(DUMP_LAYER) != std::string::npos) {
      GELOGI("dump layer: %s.", cmd_info.cmd_params.at((i * kDumpCmdPairSize) + 1U).c_str());
      (void)dump_layers.insert(cmd_info.cmd_params.at((i * kDumpCmdPairSize) + 1U));
    }
  }

  ret = ParserPara(cmd_info, DUMP_FILE_PATH, dump_path);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpPath] failed, ret:%d", ret);
    return FAILED;
  }
  if ((!dump_path.empty()) && (dump_path[dump_path.size() - 1U] != '/')) {
    dump_path = dump_path + "/";
  }
  dump_path = (dump_path + CurrentTimeInStr()) + "/";
  GELOGI("dump path = %s.", dump_path.c_str());

  ret = ParserPara(cmd_info, DUMP_MODE, dump_mode);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpMode] failed, ret:%d", ret);
    return FAILED;
  }
  GELOGI("dump mode = %s", dump_mode.c_str());

  ref_dump_properties.AddPropertyValue(dump_model, dump_layers);
  ref_dump_properties.SetDumpPath(dump_path);
  ref_dump_properties.SetDumpMode(dump_mode);
  return SUCCESS;
}

Status ModelManager::GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size) {
  const auto &hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    max_size = 0U;
    return SUCCESS;
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id:%u!", model_id);

  max_size = davinci_model->TotalMemSize();
  return SUCCESS;
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                            std::vector<InputOutputDescInfo> &output_desc) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u!", model_id);

  return davinci_model->GetInputOutputDescInfo(input_desc, output_desc);
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                            std::vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &outputFormats,
                                            const bool new_model_desc) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    hybrid_davinci_model->SetModelDescVersion(new_model_desc);
    return hybrid_davinci_model->GetInputOutputDescInfo(input_desc, output_desc, inputFormats, outputFormats);
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid model id %u!", model_id);

  return davinci_model->GetInputOutputDescInfo(input_desc, output_desc, inputFormats, outputFormats, new_model_desc);
}

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status ModelManager::GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                         int32_t &dynamic_type) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->GetDynamicBatchInfo(batch_info, dynamic_type);
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] failed, Invalid model id %u!", model_id);

  return davinci_model->GetDynamicBatchInfo(batch_info, dynamic_type);
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status ModelManager::GetCombinedDynamicDims(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);

  davinci_model->GetCombinedDynamicDims(batch_info);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get user designate shape order
/// @param [in] model_id
/// @param [out] user_input_shape_order
/// @return execute result
///
Status ModelManager::GetUserDesignateShapeOrder(const uint32_t model_id,
                                                std::vector<std::string> &user_input_shape_order) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    hybrid_davinci_model->GetUserDesignateShapeOrder(user_input_shape_order);
    return SUCCESS;
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);
  davinci_model->GetUserDesignateShapeOrder(user_input_shape_order);
  return SUCCESS;
}

Status ModelManager::GetCurrentShape(const uint32_t model_id, std::vector<int64_t> &batch_info,
                                     int32_t &dynamic_type) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);
  davinci_model->GetCurrentShape(batch_info, dynamic_type);
  return SUCCESS;
}

Status ModelManager::GetNodeAttr(const uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                                 std::string &attr_info) {
  const auto &davinci_model = GetModel(model_id);
  if (davinci_model != nullptr) {
    return davinci_model->GetNodeAttr(op_name, attr_name, attr_info);
  }
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->GetOpAttr(op_name, attr_name, attr_info);
  }
  GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "[Get][Model]Get model failed, invalid model id:%u.", model_id);
  REPORT_INNER_ERR_MSG("E19999", "Get model failed, invalid model id:%u.", model_id);
  return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
}

Status ModelManager::GetOutputShapeInfo(const uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    hybrid_davinci_model->GetOutputShapeInfo(dynamic_output_shape_info);
    return SUCCESS;
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);
  davinci_model->GetOutputShapeInfo(dynamic_output_shape_info);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get AIPP info
/// @param [in] model_id
/// @param [in] index
/// @param [out] aipp_info
/// @return execute result
///
Status ModelManager::GetAippInfo(const uint32_t model_id, const uint32_t index, AippConfigInfo &aipp_info) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->GetAippInfo(index, aipp_info);
  }
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
      "[Get][Model] failed, invalid model_id is %u.", model_id);
  return davinci_model->GetAippInfo(index, aipp_info);
}

Status ModelManager::GetAippType(const uint32_t model_id, const uint32_t index,
                                 InputAippType &type, size_t &aipp_index) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->GetAippType(index, type, aipp_index);
  }
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
      "[Get][Model] failed, invalid model_id is %u.", model_id);
  return davinci_model->GetAippType(index, type, aipp_index);
}

Status ModelManager::LoadModelOffline(const ModelData &model, const ModelParam &model_param, uint32_t &model_id,
                                      const gert::RtSession *rt_session) {
  domi::GetContext().is_online_model = false;
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(true);
  GenModelId(model_id);

  ModelHelper model_helper;
  // offline load can share weight, no need to copy
  model_helper.SetSharedWeightFlag(true);
  Status ret = model_helper.LoadRootModel(model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][RootModel] failed, ret:%d, model_id:%u.", ret, model_id);
    return ret;
  }
  GE_CHK_STATUS_RET(FileConstantUtils::RefreshRelativePath(model_helper.GetGeRootModel()->GetRootGraph()),
                    "Failed to refresh relative path, model_id:%u.", model_id);
  int32_t device_id = -1;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  (void)MsprofSetDeviceIdByGeModelIdx(model_id, static_cast<uint32_t>(device_id));
  /// In multi-threaded inference,  using the same session_id among multiple threads may cause some threads to fail.
  /// These session_ids come from the same model, so the values of session_id are the same.
  /// Update session_id for infer in load model to avoid the same session_id.
  uint64_t new_session_id;
  if (rt_session != nullptr) {
    new_session_id = rt_session->GetSessionId();
    AddSharedSessionModel(model_id);
  } else {
    new_session_id = SessionIdManager::GetNextSessionId();
  }

  ProfilingManager::Instance().SetGraphIdToModelMap(model_id, model_id);
  GELOGI("Set graph id to model map, model_id: %u, runtime session_id: %" PRIu64 ".", model_id, new_session_id);

  GE_ASSERT_SUCCESS(InitOpMasterDeviceSo(model_id, model_helper.GetGeRootModel()),
                    "Init model [%u] op master device failed", model_id);

  if (model_helper.GetModelType()) {
    bool is_shape_unknown = false;
    GE_CHK_STATUS_RET(model_helper.GetGeRootModel()->CheckIsUnknownShape(is_shape_unknown),
                      "[Check][IsUnknownShape] failed, model id:%u", model_id);
    if (is_shape_unknown || GetContext().GetHostExecFlag()) {
      const auto &ge_models = model_helper.GetGeRootModel()->GetSubgraphInstanceNameToModel();
      for (auto iter = ge_models.begin(); iter != ge_models.end(); ++iter) {
        (void)AttrUtils::SetInt(iter->second, MODEL_ATTR_SESSION_ID, static_cast<int64_t>(new_session_id));
      }

      return DoLoadHybridModelOnline(model_id, model, static_cast<uint32_t>(device_id), model_helper.GetGeRootModel(),
                                     nullptr);
    }
  }

  const GeModelPtr &ge_model = model_helper.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  const auto davinci_model = MakeShared<DavinciModel>(model.priority, nullptr);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->Assign(ge_model);
  davinci_model->SetId(model_id);
  davinci_model->SetDeviceId(static_cast<uint32_t>(device_id));
  davinci_model->SetOmName(model.om_name);
  const auto &ge_root_model = model_helper.GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  davinci_model->SetFileConstantWeightDir(ge_root_model->GetFileConstantWeightDir());
  const DumpProperties &dump_properties = DumpManager::GetInstance().GetDumpProperties(kOfflineSessionId);
  davinci_model->SetDumpProperties(dump_properties);
  if (model_param.file_constant_mems != nullptr) {
    davinci_model->SetFileConstantUserDeviceMem(*model_param.file_constant_mems);
  }
  davinci_model->SetClearDfxCacheFlagAfterInit(model_param.need_clear_dfx_cache_);

  GE_CHK_STATUS_RET(davinci_model->UpdateSessionId(new_session_id), "Update SessionId failed");
  GE_CHK_STATUS_RET(davinci_model->InitSpaceRegistry(model_helper.GetGeRootModel()), "Get space registry failed!");
  ret = davinci_model->Init(model_param);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Init][DavinciModel] failed, ret:%d.", ret);
    return ret;
  }
  // avoid wild pointer
  ge_model->ClearWeightDataBuf();

  InsertModel(model_id, davinci_model);
  GELOGI("Parse model %u success.", model_id);
  return SUCCESS;
}

/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [out] model_id: model id for manager.
/// @param [in] model_data: Model data load from offline model file.
/// @param [in] arg: input/output queue ids from user, num equals Data Op, and file constant mems
/// @return: 0 for success / others for fail
Status ModelManager::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data, const ModelQueueArg &arg) {
  ModelHelper model_helper;
  const Status ret = model_helper.LoadModel(model_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed.");
    return ret;
  }
  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = arg.input_queue_ids;
  model_queue_param.output_queues = arg.output_queue_ids;
  model_queue_param.input_queues_attrs.resize(arg.input_queue_ids.size());
  for (size_t index = 0U; index < arg.input_queue_ids.size(); index++) {
    model_queue_param.input_queues_attrs[index].queue_id = arg.input_queue_ids[index];
  }
  model_queue_param.output_queues_attrs.resize(arg.output_queue_ids.size());
  for (size_t index = 0U; index < arg.output_queue_ids.size(); index++) {
    model_queue_param.output_queues_attrs[index].queue_id = arg.output_queue_ids[index];
  }
  model_queue_param.file_constant_mems = &arg.file_constant_mems;
  model_queue_param.need_clear_dfx_cache = arg.need_clear_dfx_cache;

  return LoadModelWithQueueParam(model_id, model_helper.GetGeRootModel(), model_queue_param, model_data.priority);
}

Status ModelManager::LoadModelWithQueueParam(uint32_t &model_id,
                               const ModelData &model_data,
                               const ModelQueueParam &model_queue_param) {
  ModelHelper model_helper;
  const Status ret = model_helper.LoadModel(model_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed.");
    return ret;
  }
  return LoadModelWithQueueParam(model_id, model_helper.GetGeRootModel(), model_queue_param, model_data.priority);
}

Status ModelManager::LoadModelWithQueueParam(uint32_t &model_id,
                                             const GeRootModelPtr &root_model,
                                             const ModelQueueParam &model_queue_param,
                                             const int32_t priority,
                                             const bool need_update_session_id) {
  if (IsNeedHybridLoad(*root_model)) {
    REPORT_INNER_ERR_MSG("E19999", "Dynamic shaped graphs are not currently supported by LoadModelWithQ");
    GELOGE(UNSUPPORTED, "Dynamic shaped graphs are not currently supported by LoadModelWithQ, model_id: %u", model_id);
    return UNSUPPORTED;
  }
  const auto &root_graph = root_model->GetRootGraph();
  const auto &submodels = root_model->GetSubgraphInstanceNameToModel();
  const auto it = submodels.find(root_graph->GetName());
  if (it == submodels.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to get GeModel");
    GELOGE(INTERNAL_ERROR, "Failed to get GeModel, name = %s", root_graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  GE_ASSERT_SUCCESS(InitOpMasterDeviceSo(model_id, root_model), "Init model [%u] op master device failed", model_id);
  const auto &ge_model = it->second;
  GE_CHECK_NOTNULL(ge_model);
  const auto davinci_model = MakeShared<DavinciModel>(priority, nullptr);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->Assign(ge_model);

  Status ret = SUCCESS;
  if (need_update_session_id) {
    /// In multi-threaded inference,  using the same session_id among multiple threads may cause some threads to fail.
    /// These session_ids come from the same model, so the values of session_id are the same.
    /// Update session_id for infer in load model to avoid the same session_id.
    uint64_t new_session_id = SessionIdManager::GetNextSessionId();
    ret = davinci_model->UpdateSessionId(new_session_id);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Update][SessionId] for infer failed, SessionId:%" PRIu64 ".", new_session_id);
      return ret;
    }
  }

  GenModelId(model_id);
  davinci_model->SetId(model_id);
  int32_t device_id = -1;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  GELOGD("Get device_id %d success", device_id);
  davinci_model->SetDeviceId(static_cast<uint32_t>(device_id));
  ret = davinci_model->SetQueIds(model_queue_param.input_queues_attrs, model_queue_param.output_queues_attrs);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Ids] for model queue failed, ret:%d, model_id:%u.", ret, model_id);
    return ret;
  }
  GE_CHK_STATUS_RET(davinci_model->SetQueueType(), "Set queue type failed!");
  // set dynamic sched info
  davinci_model->SetNeedModelConfig(model_queue_param.need_model_config);
  davinci_model->SetModelUuid(model_queue_param.model_uuid);
  davinci_model->SetStatusQueue(model_queue_param.status_output_queue);
  davinci_model->SetNeedReportStatus((model_queue_param.is_dynamic_sched &&
    model_queue_param.need_report_status));

  davinci_model->SetInputFusionOffsets(model_queue_param.input_fusion_offsets);
  SetDumpProperties(davinci_model);
  davinci_model->SetModelQueueParam(model_queue_param);
  if (model_queue_param.file_constant_mems != nullptr) {
    davinci_model->SetFileConstantUserDeviceMem(*model_queue_param.file_constant_mems);
  }
  davinci_model->SetClearDfxCacheFlagAfterInit(model_queue_param.need_clear_dfx_cache);

  GE_CHK_STATUS_RET(davinci_model->InitSpaceRegistry(root_model), "Get space registry failed!");
  ret = davinci_model->Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][Model] failed, ret:%d, model_id:%u.", ret, model_id);
    return ret;
  }

  InsertModel(model_id, davinci_model);
  GELOGI("Parse model %u success.", model_id);

  return SUCCESS;
}

void ModelManager::SetDumpProperties(const std::shared_ptr<DavinciModel> &davinci_model) {
  if (DumpManager::GetInstance().GetDumpProperties(kOfflineSessionId).IsDumpOpen()) {
    davinci_model->SetDumpProperties(DumpManager::GetInstance().GetDumpProperties(kOfflineSessionId));
  } else {
    davinci_model->SetDumpProperties(dump_properties_);
  }
}

Status ModelManager::LoadModelWithoutQ(uint32_t &model_id, const GeRootModelPtr &root_model, const int32_t priority) {
  GE_CHECK_NOTNULL(root_model);
  if (IsNeedHybridLoad(*root_model)) {
    REPORT_INNER_ERR_MSG("E19999", "Dynamic shaped graphs are not currently supported by LoadModelWithQ");
    GELOGE(UNSUPPORTED, "Dynamic shaped graphs are not currently supported by LoadModelWithQ, model_id: %u", model_id);
    return UNSUPPORTED;
  }
  const auto &root_graph = root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  const auto &submodels = root_model->GetSubgraphInstanceNameToModel();
  const auto it = submodels.find(root_graph->GetName());
  if (it == submodels.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to get GeModel");
    GELOGE(INTERNAL_ERROR, "Failed to get GeModel, name = %s, model_id: %u", root_graph->GetName().c_str(), model_id);
    return INTERNAL_ERROR;
  }

  GE_ASSERT_SUCCESS(InitOpMasterDeviceSo(model_id, root_model), "Init model [%u] op master device failed", model_id);
  const auto &ge_model = it->second;
  GE_CHECK_NOTNULL(ge_model);
  const auto davinci_model = MakeShared<DavinciModel>(priority, nullptr);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->Assign(ge_model);

  GenModelId(model_id);
  davinci_model->SetId(model_id);
  davinci_model->SetDumpProperties(dump_properties_);
  davinci_model->SetNeedModelConfig(true);
  int32_t device_id = -1;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  GELOGD("Get device_id %d success", device_id);
  davinci_model->SetDeviceId(static_cast<uint32_t>(device_id));
  GE_CHK_STATUS_RET(davinci_model->InitSpaceRegistry(root_model), "Get space registry failed!");
  GELOGD("Begin to init model %u.", model_id);
  GE_CHK_STATUS_RET(davinci_model->Init(), "[Init][Model] failed, model_id:%u.", model_id);

  InsertModel(model_id, davinci_model);
  GELOGI("Parse model %u success.", model_id);

  if (davinci_model->CheckModelNoInputAndOutput()) {
    GELOGI("need model %u execute.", model_id);
    const std::vector<GeTensor> input_tensor;
    std::vector<GeTensor> output_tensor;
    GE_CHK_STATUS_RET(ExecuteModel(model_id, nullptr, false, input_tensor, output_tensor),
                      "[Execute][Model] failed, model_id:%u.", model_id);
    GELOGI("Finish model %u execute.", model_id);
  }

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief  ACL case, not start new thread, return result
/// @param [in] model_id  mode id
/// @param [in] stream   model stream
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  input data
/// @param [in] input_desc  description of input data
/// @param [out] output_data  output data
/// @param [out] output_desc  description of output data
///
Status ModelManager::ExecuteModel(const uint32_t model_id, const rtStream_t stream, const bool async_mode,
                                  const InputData &input_data, const std::vector<GeTensorDesc> &input_desc,
                                  OutputData &output_data, std::vector<GeTensorDesc> &output_desc,
                                  const std::vector<GeTensor> &input_tensor,
                                  const std::vector<GeTensor> &output_tensor) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->Execute(input_data.blobs, input_desc, output_data.blobs, output_desc, stream);
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Invalid model id %u, check whether model has been loaded or not.", model_id);

  return davinci_model->NnExecute(stream, async_mode, input_data, output_data, input_tensor, output_tensor);
}

Status ModelManager::UpdateFeatureMemoryBase(const uint32_t model_id, const uintptr_t mem_base, const size_t size) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  GE_ASSERT_TRUE(hybrid_davinci_model == nullptr, "[Check][Model] Not support dynamic model, model_id:%u.", model_id);

  const auto &davinci_model = GetModel(model_id);
  GE_ASSERT_NOTNULL(davinci_model, "[Get][Model] Invalid model id %u, please check model loaded.", model_id);

  size_t used_size = 0U;
  GE_ASSERT_SUCCESS(davinci_model->UpdateHbmFmMemBases(mem_base, size, used_size));
  GELOGI("Update feature memory base success, model_id:%u, mem_base:%#lx, mem_size:%zu, used_mem_size:%zu",
    model_id, mem_base, size, used_size);
  return SUCCESS;
}

Status ModelManager::PaRemapped(const uint32_t model_id, const uint64_t va, const uint64_t new_pa,
                                const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) {
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  GE_IF_BOOL_EXEC(hybrid_davinci_model != nullptr,
                  GELOGW("[Check][Model] Not support dynamic model, model_id:%u", model_id);
                  return FAILED);

  const auto &davinci_model = GetModel(model_id);
  GE_ASSERT_NOTNULL(davinci_model);

  GELOGI("PaRemap start model_id:%u va:%" PRIu64 ", new_pa:%" PRIu64 ", len:%" PRIu64 ".", model_id, va, new_pa, len);
  return davinci_model->PaRemapped(va, new_pa, len, cross_ranges);
}

namespace {
void GetGeTensorBlobs(const std::vector<GeTensor> &tensors, std::vector<DataBuffer> &blobs) {
  blobs.resize(tensors.size());
  for (size_t i = 0U; i < tensors.size(); i++) {
    blobs[i].data = ValueToPtr(PtrToValue(tensors[i].GetData().data()));
    blobs[i].length = tensors[i].GetData().size();
    blobs[i].isDataSupportMemShare = false;
    // In case the user does not set the placement.
    blobs[i].placement = static_cast<uint32_t>(Placement::kPlacementDevice);
  }
}

void GetGeTensorShapes(const std::vector<GeTensor> &tensors, std::vector<std::vector<int64_t>> &shapes) {
  for (size_t i = 0U; i < tensors.size(); i++) {
    shapes.emplace_back(tensors[i].GetTensorDesc().GetShape().GetDims());
  }
}

void GetGeTensorDescs(const std::vector<GeTensor> &tensors, std::vector<GeTensorDesc> &descs) {
  descs.reserve(tensors.size());
  const auto accum_tensors = [&descs](const GeTensor &tensor) -> bool {
    // update origin shape for user input
    if (!tensor.GetTensorDesc().IsOriginShapeInitialized()) {
      auto tensor_desc = tensor.GetTensorDesc();
      tensor_desc.SetOriginShape(tensor_desc.GetShape());
      descs.emplace_back(tensor_desc);
      return true;
    }
    descs.emplace_back(tensor.GetTensorDesc());
    return true;
  };
  (void)std::all_of(tensors.cbegin(), tensors.cend(), accum_tensors);
}
}

Status ModelManager::ExecuteModelAsync(const uint32_t model_id, const rtStream_t stream, const bool async_mode,
                                       const std::vector<GeTensor> &input_tensor,
                                       std::vector<GeTensor> &output_tensor) {
  InputData input_buffer;
  input_buffer.index = 0U;
  input_buffer.model_id = model_id;
  OutputData output_buffer;
  output_buffer.index = 0U;
  output_buffer.model_id = model_id;

  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;
  Status status = SUCCESS;
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr && input_tensor.size() != 0) {
    GetGeTensorDescs(input_tensor, input_desc);
    GetGeTensorDescs(output_tensor, output_desc);
    GetGeTensorBlobs(input_tensor, input_buffer.blobs);
    GetGeTensorBlobs(output_tensor, output_buffer.blobs);
    GetGeTensorShapes(input_tensor, input_buffer.shapes);
    status = hybrid_davinci_model->Execute(input_buffer.blobs, input_desc, output_buffer.blobs, output_desc, stream);
    if (status == SUCCESS) {
      GELOGI("Execute model %u success.", model_id);
    }
    return status;
  }

  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Invalid model id %u, check whether model has been loaded or not.", model_id);

  status = davinci_model->NnExecute(stream, async_mode, input_buffer, output_buffer, input_tensor, output_tensor);
  if (status == SUCCESS) {
    GELOGD("Execute model %u success.", model_id);
  }
  return status;
}

Status ModelManager::ExecuteModel(const uint32_t model_id, const rtStream_t stream, const bool async_mode,
                                  const std::vector<GeTensor> &input_tensor,
                                  std::vector<GeTensor> &output_tensor) {
  GE_ASSERT_SUCCESS(ExecuteModelAsync(model_id, stream, async_mode,
      input_tensor, output_tensor));
  auto model = GetModel(model_id);
  if (model != nullptr) {
    model->UpdateOutputTensorShape(output_tensor);
    GELOGD("UpdateOutputTensorShape for model:[%u] success.", model_id);
  }
  return SUCCESS;
}

Status ModelManager::ExecuteModelWithStream(const uint32_t model_id, const rtStream_t stream, const bool async_mode,
                                  const std::vector<gert::Tensor> &input_tensor,
                                  std::vector<gert::Tensor> &output_tensor) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Invalid model id %u, check whether model has been loaded or not.", model_id);
  GE_ASSERT_SUCCESS(davinci_model->NnExecute(stream, async_mode, input_tensor, output_tensor));
  return SUCCESS;
}

Status ModelManager::ExecuteModelWithStreamAsync(const uint32_t model_id, const GraphNodePtr &graph_node,
                                                 const std::vector<GeTensor> &input_tensor,
                                                 std::vector<GeTensor> &output_tensor, const rtStream_t stream) {
  // ExecuteModelWithStreamAsync: dynamic model should return output tensor to usr
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->ExecuteWithStreamAsync(input_tensor, output_tensor, stream);
  }

  bool is_refreshable = false;
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator != nullptr) {
    is_refreshable = graph_node->IsFeatureBaseRefreshable();
    if (is_refreshable) {
      // malloc feature memory by external allocator
      GE_ASSERT_SUCCESS(
          MallocFeatureMemory(graph_node->GetGraphId(), model_id, graph_node, external_allocator),
          "malloc feature memory failed, graph_id:%u.", graph_node->GetGraphId());
    }
    // malloc outputs memory by external allocator
    GE_ASSERT_SUCCESS(MallocOutputsMemory(graph_node->GetGraphId(), graph_node, external_allocator, output_tensor),
                      "malloc outputs memory failed, graph_id:%u.", graph_node->GetGraphId());
  }
  // ExecuteModelWithStreamAsync: static model use default
  Status result = ExecuteModel(model_id, stream, true, input_tensor, output_tensor);
  if (is_refreshable) {
    FreeFeatureMemory(graph_node);
  }
  return result;
}

Status ModelManager::ExecuteModelWithStreamAsync(const uint32_t model_id, const GraphNodePtr &graph_node,
                                                 const std::vector<gert::Tensor> &input_tensor,
                                                 std::vector<gert::Tensor> &output_tensor, const rtStream_t stream) {
  // ExecuteModelWithStreamAsync: dynamic model should return output tensor to usr
  const auto &hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->ExecuteWithStreamAsync(input_tensor, output_tensor, stream);
  }
  // ExecuteModelWithStreamAsync: static model use default
  bool is_refreshable = false;
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator != nullptr) {
    is_refreshable = graph_node->IsFeatureBaseRefreshable();
    if (is_refreshable) {
      // malloc feature memory by external allocator
      GE_ASSERT_SUCCESS(MallocFeatureMemory(graph_node->GetGraphId(), model_id, graph_node, external_allocator),
                        "malloc feature memory failed, graph_id:%u.", graph_node->GetGraphId());
    }
    // malloc outputs memory by external allocator
    GE_ASSERT_SUCCESS(MallocOutputsMemory(graph_node->GetGraphId(), graph_node, external_allocator, output_tensor),
                      "malloc outputs memory failed, graph_id:%u.", graph_node->GetGraphId());
  }

  Status result = ExecuteModelWithStream(model_id, stream, true, input_tensor, output_tensor);
  if (is_refreshable) {
    FreeFeatureMemory(graph_node);
  }
  return result;
}

Status ModelManager::CreateAicpuSession(const uint64_t session_id) {
  const std::lock_guard<std::recursive_mutex> lk(map_mutex_);
  auto &device_ids = sess_id_to_device_ids_[session_id];
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  GELOGI("CreateAicpuSession device id:%d", device_id);
  const auto &it = device_ids.find(static_cast<uint32_t>(device_id));
  // never been created by any model
  if (it == device_ids.cend()) {
    const Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_SESSION_CREATE, session_id, 0U, 0U);
    if (ret == SUCCESS) {
      (void) device_ids.insert(static_cast<uint32_t>(device_id));
      GELOGI("The session:%" PRIu64 " create success on device:%d", session_id, device_id);
    }
    return ret;
  }
  return SUCCESS;
}

Status ModelManager::LoadCustAicpuSo(const OpDescPtr &op_desc, const std::string &so_name, bool &loaded) {
  GELOGD("LoadCustAicpuSo in, op name %s, so name %s", op_desc->GetName().c_str(), so_name.c_str());
  const CustAICPUKernelPtr aicpu_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
  return LoadCustAicpuSo(aicpu_kernel, so_name, loaded);
}

Status ModelManager::LoadCustAicpuSo(const CustAICPUKernelPtr &aicpu_kernel, const std::string &so_name, bool &loaded) {
  if (aicpu_kernel == nullptr) {
    GELOGI("cust aicpu has no corresponding kernel: %s!", so_name.c_str());
    return SUCCESS;
  }

  // get current context
  rtContext_t rt_cur_ctx = nullptr;
  GE_CHK_RT_RET(rtCtxGetCurrent(&rt_cur_ctx));

  // use current context as resource key
  const std::lock_guard<std::mutex> lk(cust_aicpu_mutex_);
  const uintptr_t resource_id = static_cast<uintptr_t>(PtrToValue(rt_cur_ctx));
  auto &iter_so = cust_aicpu_so_[resource_id];
  const auto it_so_name = iter_so.find(so_name);
  if (it_so_name == iter_so.end()) {
    iter_so[so_name] = {aicpu_kernel, 0UL};
    loaded = false;
    GELOGD("LoadCustAicpuSo add aicpu so name %s, resource id %" PRIu64, so_name.c_str(), resource_id);
    return SUCCESS;
  }
  loaded = true;
  GELOGD("LoadCustAicpuSo so name %s has been loaded, resource id %" PRIu64 ".", so_name.c_str(), resource_id);
  return SUCCESS;
}

Status ModelManager::GetCustAicpuSo(const std::string &so_name, CustAICPUKernelPtr &aicpu_kernel) {
  const std::lock_guard<std::mutex> lk(cust_aicpu_mutex_);
  for (const auto &iter_so : cust_aicpu_so_) {
    const auto &iter_so_name = iter_so.second.find(so_name);
    if (iter_so_name != iter_so.second.end()) {
      GELOGI("Get cust so for %s success", so_name.c_str());
      aicpu_kernel = iter_so_name->second.kernel_ptr;
      return SUCCESS;
    }
  }

  aicpu_kernel = nullptr;
  return FAILED;
}

Status ModelManager::LaunchKernelCustAicpuSo(const std::string &kernel_name) {
  GELOGD("Aicpu kernel launch task in, kernel name %s.", kernel_name.c_str());
  const std::lock_guard<std::mutex> lk(cust_aicpu_mutex_);
  if (cust_aicpu_so_.empty()) {
    return SUCCESS;
  }
  // get current context
  rtContext_t rt_cur_ctx = nullptr;
  GE_CHK_RT_RET(rtCtxGetCurrent(&rt_cur_ctx));

  const uintptr_t resource_id = static_cast<uintptr_t>(PtrToValue(rt_cur_ctx));
  const auto it = cust_aicpu_so_.find(resource_id);
  if (it == cust_aicpu_so_.end()) {
    GELOGI("Cust aicpu so map is empty, context id %" PRIu64, resource_id);
    return SUCCESS;
  }

  if (kernel_name == kDeleteCustOp) {
    bool skip_flag = false;
    for (auto &it_so : it->second) {
      if (it_so.second.launch_count > 0UL) {
        if ((--it_so.second.launch_count) > 0UL) {
          skip_flag = true;
        }
      }
    }
    if (skip_flag) {
      GELOGD("Skip repeated delete.");
      return SUCCESS;
    }
  }

  rtStream_t stream = nullptr;
  std::vector<void *> allocated_mem;
  const std::function<void()> callback = [&stream, &allocated_mem]() {
    for (auto &mem : allocated_mem) {
      GE_CHK_RT(rtFree(mem));
    }
    if (stream != nullptr) {
      GE_CHK_RT(rtStreamDestroy(stream));
    }
  };
  GE_MAKE_GUARD(release, callback);

  std::vector<CustAicpuSoBuf> v_cust_so;
  for (auto &it_so : it->second) {
    if (kernel_name == kBatchLoadBuf) {
      ++it_so.second.launch_count;
      if (it_so.second.launch_count > 1UL) {
        GELOGD("Skip repeated load with so:[%s].", it_so.first.c_str());
        continue;
      }
    }
    const void *const aicpu_data = it_so.second.kernel_ptr->GetBinData();
    const size_t aicpu_data_length = it_so.second.kernel_ptr->GetBinDataSize();
    const std::string so_name = it_so.first;
    void *d_aicpu_data = nullptr;
    void *d_so_name = nullptr;

    GE_CHK_RT_RET(rtMalloc(&d_aicpu_data, aicpu_data_length, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    allocated_mem.push_back(d_aicpu_data);
    GE_CHK_RT_RET(rtMalloc(&d_so_name, so_name.size(), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    allocated_mem.push_back(d_so_name);
    GE_CHK_RT(rtMemcpy(d_aicpu_data, aicpu_data_length, aicpu_data, aicpu_data_length, RT_MEMCPY_HOST_TO_DEVICE));
    GE_CHK_RT(rtMemcpy(d_so_name, so_name.size(), so_name.c_str(), so_name.size(), RT_MEMCPY_HOST_TO_DEVICE));

    CustAicpuSoBuf cust_aicpu_so_buf;
    cust_aicpu_so_buf.kernelSoBuf = PtrToValue(d_aicpu_data);
    cust_aicpu_so_buf.kernelSoBufLen = static_cast<uint32_t>(aicpu_data_length);
    cust_aicpu_so_buf.kernelSoName = PtrToValue(d_so_name);
    cust_aicpu_so_buf.kernelSoNameLen = static_cast<uint32_t>(so_name.size());
    v_cust_so.push_back(cust_aicpu_so_buf);
  }

  if (v_cust_so.empty()) {
    GELOGD("Cust so is empty skip do launch with kernel [%s].", kernel_name.c_str());
    return SUCCESS;
  }

  void *args = nullptr;
  const size_t args_size = sizeof(CustAicpuSoBuf) * v_cust_so.size();
  GE_CHK_RT_RET(rtMalloc(&args, args_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  allocated_mem.push_back(args);
  GE_CHK_RT(rtMemcpy(args, args_size, v_cust_so.data(), args_size, RT_MEMCPY_HOST_TO_DEVICE));

  BatchLoadOpFromBufArgs batch_cust_so;
  batch_cust_so.soNum = static_cast<uint32_t>(v_cust_so.size());
  batch_cust_so.args = PtrToValue(args);
  GE_ASSERT_EOK(strcpy_s(batch_cust_so.kernel_name, sizeof(batch_cust_so.kernel_name), kernel_name.c_str()));

  constexpr uint32_t batch_args_size = static_cast<uint32_t>(sizeof(BatchLoadOpFromBufArgs));
  GE_CHK_RT(rtStreamCreate(&stream, 0));
  rtAicpuArgsEx_t args_info = {};
  args_info.args = const_cast<void *>(static_cast<void *>(&batch_cust_so));
  args_info.argsSize = batch_args_size;
  args_info.isNoNeedH2DCopy = 0U;
  // 临时方案，offset设置为最大值表示为空
  args_info.soNameAddrOffset = kNullSoNameOffset;
  args_info.kernelNameAddrOffset = batch_args_size - static_cast<uint32_t>(sizeof(batch_cust_so.kernel_name));
  // 临时方案，超时时间设置为最大值表示永不超时
  args_info.timeout = kNeverTimeout;
  GE_CHK_RT(rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU, kernel_name.c_str(), 1U, &args_info,
                                          nullptr, stream, RT_KERNEL_USE_SPECIAL_TIMEOUT));
  GELOGI("Load cust so, soNameAddrOffset %u, kernelNameAddrOffset %u, timeout %u", args_info.soNameAddrOffset,
         args_info.kernelNameAddrOffset, args_info.timeout);

  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  GELOGI("Cpu kernel launch task success.");
  return SUCCESS;
}

Status ModelManager::ClearAicpuSo() {
  GE_CHK_STATUS_RET(LaunchKernelCustAicpuSo(kDeleteCustOp),
                    "[Call][LaunchKernelCustAicpuSo] delete cust op so failed.");
  return SUCCESS;
}

Status ModelManager::LaunchCustAicpuSo() {
  GE_CHK_STATUS_RET(LaunchKernelCustAicpuSo(kBatchLoadBuf),
                    "[Call][LaunchKernelCustAicpuSo] launch cust op so failed.");
  return SUCCESS;
}

Status ModelManager::GetPlatformInfosSoName(std::string &so_name) {
  // get current context
  rtContext_t rt_cur_ctx = nullptr;
  GE_CHK_RT_RET(rtCtxGetCurrent(&rt_cur_ctx));
  const uintptr_t resource_id = static_cast<uintptr_t>(PtrToValue(rt_cur_ctx));
  std::vector<std::string> v_so_name;
  {
    const std::lock_guard<std::mutex> lk(cust_aicpu_mutex_);
    for (auto &it : cust_aicpu_so_[resource_id]) {
      if (it.second.launch_count > 0) {
        v_so_name.emplace_back(it.first);
      }
    }
  }
  {
    const std::lock_guard<std::mutex> opmaster_lk(op_master_device_mutex_);
    for (auto &it : v_so_name) {
      if (cust_op_master_so_names_to_bin_.find(it) != cust_op_master_so_names_to_bin_.end()) {
        so_name = it;
        break;
      }
    }
  }
  return SUCCESS;
}

std::string ModelManager::GetCustTilingDeviceUniqueSoName(const uint32_t model_id, const std::string &so_name) {
  const auto &query_so_name = ModelUtils::GetOpMasterDeviceKey(model_id, so_name);
  const std::lock_guard<std::mutex> lk(op_master_device_mutex_);
  return cust_op_master_so_names_to_unique_name_[query_so_name];
}

KernelBinPtr ModelManager::GetCustTilingDeviceSoBin(const std::string &unique_so_name) {
  OpSoBinPtr op_so_bin = nullptr;
  {
    const std::lock_guard<std::mutex> lk(op_master_device_mutex_);
    op_so_bin = cust_op_master_so_names_to_bin_[unique_so_name];
  }
  GE_ASSERT_NOTNULL(op_so_bin, "find so name %s from cust_op_master_so_names_to_bin failed", unique_so_name.c_str());
  const auto data = reinterpret_cast<char_t *>(op_so_bin->MutableBinData());
  GE_ASSERT_NOTNULL(data);
  return MakeShared<OpKernelBin>(unique_so_name, std::vector<char_t>(data, data + op_so_bin->GetBinDataSize()));
}

std::string ModelManager::GetBuiltinTilingDeviceSoName(const std::string &so_name) const {
  const auto &pos = so_name.find_last_of("/");
  GE_ASSERT_TRUE(pos != std::string::npos);
  return so_name.substr(pos + 1UL);
}

KernelBinPtr ModelManager::GetBuiltinTilingDeviceSoBin(const std::string &so_name) {
  OpSoBinPtr op_so_bin = nullptr;
  {
    const std::lock_guard<std::mutex> lk(op_master_device_mutex_);
    op_so_bin = built_in_op_master_so_names_to_bin_[so_name];
  }
  GE_ASSERT_NOTNULL(op_so_bin, "find so name %s of built_in op_master_so failed", so_name.c_str());
  const auto data = reinterpret_cast<char_t *>(op_so_bin->MutableBinData());
  GE_ASSERT_NOTNULL(data);
  return MakeShared<OpKernelBin>(so_name, std::vector<char_t>(data, data + op_so_bin->GetBinDataSize()));
}

Status ModelManager::LoadCustAicpuSoAndUpdateSoName(const uint32_t model_id, std::string &so_name) {
  GELOGD("Start to load custom so %s", so_name.c_str());
  const auto unique_so_name = GetCustTilingDeviceUniqueSoName(model_id, so_name);
  GELOGI("[OpMasterDevice][Custom]The so [%s] will be replaced by [%s].", so_name.c_str(), unique_so_name.c_str());
  so_name = unique_so_name;
  const auto aicpu_kernel = GetCustTilingDeviceSoBin(so_name);
  GE_ASSERT_NOTNULL(aicpu_kernel);
  bool loaded = false;
  return LoadCustAicpuSo(aicpu_kernel, so_name, loaded);
}

Status ModelManager::LoadBuiltinAicpuSoAndUpdateSoName(const uint32_t device_id, std::string &so_name) {
  GELOGD("Start to load built-in so %s", so_name.c_str());
  so_name = GetBuiltinTilingDeviceSoName(so_name);
  GELOGI("[OpMasterDevice][BuiltIn]The so name will be replaced by [%s].", so_name.c_str());
  const auto aicpu_kernel = GetBuiltinTilingDeviceSoBin(so_name);
  GE_ASSERT_NOTNULL(aicpu_kernel);
  return LoadBuiltinAicpuSo(aicpu_kernel, device_id, so_name);
}

Status ModelManager::LoadBuiltinAicpuSo(const KernelBinPtr &aicpu_kernel, const uint32_t device_id,
                                        const std::string &so_name) {
  GE_ASSERT_NOTNULL(aicpu_kernel);
  // use device id as resource key
  const std::lock_guard<std::mutex> lk(builtin_aicpu_mutex_);
  auto &iter_so = builtin_aicpu_so_[device_id];
  if (iter_so.find(so_name) == iter_so.end()) {
    iter_so[so_name] = {aicpu_kernel, 0UL};
    GELOGI("[OpMasterDevice][BuiltIn]Add so name %s, device id %u", so_name.c_str(), device_id);
    return SUCCESS;
  }
  GELOGI("[OpMasterDevice][BuiltIn]So name %s has been loaded.", so_name.c_str());
  return SUCCESS;
}

const std::map<std::string, AICPUKernelHolder> ModelManager::CollectWorkingBuiltinAicpuSo(
    const std::string &kernel_name, const uint32_t device_id) {
  const std::lock_guard<std::mutex> lk(builtin_aicpu_mutex_);
  const auto &it = builtin_aicpu_so_.find(device_id);
  if (it == builtin_aicpu_so_.end()) {
    GELOGI("Builtin aicpu so map is empty for device id[%u]", device_id);
    return std::map<std::string, AICPUKernelHolder>{};
  }

  std::map<std::string, AICPUKernelHolder> working_list;
  for (auto &it_so : it->second) {
    if ((kernel_name == kLoadBuiltinSo) && (++it_so.second.launch_count) == 1UL) {
      working_list.emplace(it_so);
      GELOGI("[OpMasterDevice][BuiltIn]Collect load so [%s] for device id [%u].", it_so.first.c_str(), device_id);
    }
    if ((kernel_name == kUnloadBuiltinSo) && (it_so.second.launch_count != 0) && (--it_so.second.launch_count) == 0UL) {
      working_list.emplace(it_so);
      GELOGI("[OpMasterDevice][BuiltIn]Collect unload so [%s] for device id [%u].", it_so.first.c_str(), device_id);
    }
  }
  return working_list;
}

Status ModelManager::LaunchKernelBuiltinAicpuSo(const std::string &kernel_name, const uint32_t device_id) {
  const auto working_builtin_aicpu_so = CollectWorkingBuiltinAicpuSo(kernel_name, device_id);
  if (working_builtin_aicpu_so.empty()) {
    GELOGI("Builtin aicpu so map is empty for device id[%u]", device_id);
    return SUCCESS;
  }

  rtStream_t stream = nullptr;
  void *d_aicpu_data = nullptr;
  void *d_so_name = nullptr;
  const std::function<void()> callback = [&stream, &d_aicpu_data, &d_so_name]() -> void {
    if (d_aicpu_data != nullptr) {
      GE_CHK_RT(rtFree(d_aicpu_data));
    }
    if (d_so_name != nullptr) {
      GE_CHK_RT(rtFree(d_so_name));
    }
    if (stream != nullptr) {
      GE_CHK_RT(rtStreamDestroy(stream));
    }
  };
  GE_MAKE_GUARD(release, callback);

  std::vector<LoadSoFromBufArgs> v_builtin_so;
  for (auto &it_so : working_builtin_aicpu_so) {
    const void *const aicpu_data = it_so.second.kernel_ptr->GetBinData();
    const size_t aicpu_data_len = it_so.second.kernel_ptr->GetBinDataSize();
    const std::string &so_name = it_so.first;
    if (kernel_name == kLoadBuiltinSo) {
      GE_CHK_RT_RET(rtMalloc(&d_aicpu_data, aicpu_data_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
      GE_CHK_RT_RET(rtMemcpy(d_aicpu_data, aicpu_data_len, aicpu_data, aicpu_data_len, RT_MEMCPY_HOST_TO_DEVICE));
    }
    GE_CHK_RT_RET(rtMalloc(&d_so_name, so_name.size(), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
    GE_CHK_RT_RET(rtMemcpy(d_so_name, so_name.size(), so_name.c_str(), so_name.size(), RT_MEMCPY_HOST_TO_DEVICE));

    LoadSoFromBufArgs aicpu_so_buf;
    ConstructLoadSoFromBufArgs(d_aicpu_data, aicpu_data_len, d_so_name, so_name.length(), aicpu_so_buf);
    GE_ASSERT_EOK(strcpy_s(aicpu_so_buf.so_name, sizeof(aicpu_so_buf.so_name), kLibAicpuExtendKernelsSo.c_str()));
    GE_ASSERT_EOK(strcpy_s(aicpu_so_buf.kernel_name, sizeof(aicpu_so_buf.kernel_name), kernel_name.c_str()));

    GE_CHK_RT(rtStreamCreate(&stream, 0));
    rtAicpuArgsEx_t args_info = {};
    args_info.args = static_cast<void *>(&aicpu_so_buf);
    args_info.argsSize = static_cast<uint32_t>(sizeof(LoadSoFromBufArgs));
    args_info.isNoNeedH2DCopy = 0U;
    auto aicpu_so_buf_args_size = static_cast<uint32_t>(sizeof(LoadSoFromBufArgs));
    auto so_name_size = static_cast<uint32_t>(sizeof(aicpu_so_buf.so_name));
    auto kernel_name_size = static_cast<uint32_t>(sizeof(aicpu_so_buf.kernel_name));
    args_info.soNameAddrOffset = aicpu_so_buf_args_size - so_name_size - kernel_name_size;
    args_info.kernelNameAddrOffset = aicpu_so_buf_args_size - so_name_size;
    // 临时方案，超时时间设置为最大值表示永不超时
    args_info.timeout = kNeverTimeout;
    GE_CHK_RT_RET(rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU, kernel_name.c_str(), 1U, &args_info,
                                                nullptr, stream, RT_KERNEL_USE_SPECIAL_TIMEOUT));
    GELOGI("[OpMasterDevice][BuiltIn]Launch so[%s], kernel_name[%s], stream[%" PRIu64 "].", so_name.c_str(),
           kernel_name.c_str(), PtrToValue(stream));
    GELOGI("Load build in so, soNameAddrOffset %u, kernelNameAddrOffset %u, timeout %u", args_info.soNameAddrOffset,
           args_info.kernelNameAddrOffset, args_info.timeout);
    GE_CHK_RT_RET(rtStreamSynchronize(stream));
  }
  return SUCCESS;
}

Status ModelManager::ClearBuiltinAicpuSo(const uint32_t device_id) {
  GE_ASSERT_SUCCESS(LaunchKernelBuiltinAicpuSo(kUnloadBuiltinSo, device_id),
                    "[Call][LaunchKernelBuiltinAicpuSo] clear builtin op so failed.");
  return SUCCESS;
}

Status ModelManager::LaunchBuiltinAicpuSo(const uint32_t device_id) {
  GE_ASSERT_SUCCESS(LaunchKernelBuiltinAicpuSo(kLoadBuiltinSo, device_id),
                    "[Call][LaunchKernelBuiltinAicpuSo] launch builtin op so failed.");
  return SUCCESS;
}


///
/// @ingroup ge
/// @brief get model memory size and weight
/// @param [in] const ModelData model: model type
/// @param [out] size_t memSize: model memory usage
///           size_t weightSize: model weight and memory size
/// @return SUCCESS success / others failure
///
Status ModelManager::GetModelMemAndWeightSize(const ModelData &model, size_t &mem_size, size_t &weight_size) {
  uint8_t *model_data = nullptr;
  uint64_t model_len = 0UL;
  Status ret = ModelParserBase::ParseModelContent(model, model_data, model_len);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Parse][ModelContent] failed!");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const ModelFileHeader * const mdl_file_header = PtrToPtr<void, ModelFileHeader>(model.model_data);
  const bool is_dynamic_model = ModelParserBase::IsDynamicModel(*mdl_file_header);
  const size_t model_num = is_dynamic_model ? mdl_file_header->model_num : 1U;
  // load partition table
  OmFileLoadHelper om_load_helper;
  GE_ASSERT_NOTNULL(model_data);
  ret = om_load_helper.Init(model_data, model_len, static_cast<uint32_t>(model_num), mdl_file_header);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Init][OmFileHelper] failed, ret:%u", ret);
    return ret;
  }
  // avoid simulated mode for static model
  const auto root_partition_table = *(reinterpret_cast<ModelPartitionTable *>(model_data));
  if ((!is_dynamic_model) && (root_partition_table.num == 1U)) {
    const std::string reason =
        "The model cannot be executed. For static models, the table number must be greater than 1.";
    REPORT_PREDEFINED_ERR_MSG("E10055", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({reason.c_str()}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] om model is error, please use executable om model");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  // query weight size
  ModelPartition partition_weight;
  weight_size = 0U;
  for (size_t idx = 0U; idx < model_num; ++idx) {
    ret = om_load_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition_weight, idx);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Get][ModelPartition] failed. ret = %u", ret);
      return ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED;
    }
    GE_ASSERT_TRUE(!ge::AddOverflow(weight_size, partition_weight.size, weight_size),
                   "Fail to query weight size, as weight size overflow!");
    partition_weight.size = 0U;
  }
  // query work size for dynamic model
  if (is_dynamic_model) {
    mem_size = 0U;
    GELOGD("Query size for dynamic model success, model num:%zu, weight size:%zu bytes, work size:0 byte.", model_num,
           weight_size);
    return SUCCESS;
  }
  // query work size for static model
  ModelPartition task_partition;
  if (om_load_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition, 0U) != SUCCESS) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, "[Get][ModelPartition] failed.");
    return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
  }

  domi::ModelTaskDef model_task_def;
  if (task_partition.size != 0U) {
    if (!ReadProtoFromArray(task_partition.data, static_cast<int32_t>(task_partition.size), &model_task_def)) {
      GELOGE(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, "[Read][Proto] From Array failed.");
      return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
    }
  }
  mem_size = model_task_def.memory_size();
  GELOGD("Query size for static model success, model num:%zu, weight size:%zu bytes, work size:%zu bytes.",
         model_num, weight_size, mem_size);
  return SUCCESS;
}

void ModelManager::GenModelId(uint32_t &id) {
  id = max_model_id_.fetch_add(1);
}

Status ModelManager::GetOrigInputInfo(const uint32_t model_id, const uint32_t index,
                                      OriginInputInfo &orig_input_info) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] failed, invalid model_id is %u.", model_id);

  return davinci_model->GetOrigInputInfo(index, orig_input_info);
}

Status ModelManager::GetAllAippInputOutputDims(const uint32_t model_id, const uint32_t index,
                                               std::vector<InputOutputDims> &input_dims,
                                               std::vector<InputOutputDims> &output_dims) {
  const auto &davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] failed, invalid model_id is %u.", model_id);

  return davinci_model->GetAllAippInputOutputDims(index, input_dims, output_dims);
}

bool ModelManager::IsDynamicShape(const uint32_t model_id) {
  const auto &model = GetHybridModel(model_id);
  return model != nullptr;
}

Status ModelManager::SyncExecuteHybridModel(const uint32_t model_id, const std::vector<gert::Tensor> &inputs,
                                      std::vector<gert::Tensor> &outputs) {
  const auto &model = GetHybridModel(model_id);
  GE_ASSERT_NOTNULL(model);
  const auto device_id = model->GetDeviceId();
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(device_id)));
  GE_MAKE_GUARD(reset_device, [device_id]() {
    GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(device_id)));
  });
  return model->Execute(inputs, outputs);
}

Status ModelManager::GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                                   OpDescInfo &desc_info) const {
  for (const auto &model : model_map_) {
    const auto davinci_model = model.second;
    if (davinci_model->GetDeviceId() == device_id) {
      GELOGI("[Get][OpDescInfo] Start to GetOpDescInfo of device_id %u in davinci model.", device_id);
      if (davinci_model->GetOpDescInfo(stream_id, task_id, desc_info)) {
        GELOGI("[Get][OpDescInfo] Find specific node of stream_id %u, task_id %u in davinci model.",
               stream_id, task_id);
        return SUCCESS;
      }
    }
  }
  for (const auto &model : hybrid_model_map_) {
    const auto hybrid_model = model.second;
    if (hybrid_model->GetDeviceId() == device_id) {
      GELOGI("[Get][OpDescInfo] Start to GetOpDescInfo of device_id %u in hybrid model.", device_id);
      if (hybrid_model->GetOpDescInfo(stream_id, task_id, desc_info)) {
        GELOGI("[Get][OpDescInfo] Find specific node of stream_id: %u, task_id: %u in hybrid model.",
               stream_id, task_id);
        return SUCCESS;
      }
    }
  }
  GELOGE(FAILED, "can not find exception info from device_id %u, stream_id %u, task_id %u, please check these params",
         device_id, stream_id, task_id);
  return FAILED;
}

Status ModelManager::LaunchKernelCheckAicpuOp(const std::vector<std::string> &aicpu_optype_list,
                                              const std::vector<std::string> &aicpu_tf_optype_list) {
  const std::string kernel_name = "checkOpType";
  GELOGI("LaunchKernelCheckAicpuOpType in, kernel name is %s", kernel_name.c_str());
  const std::lock_guard<std::mutex> lk(cust_aicpu_mutex_);
  std::vector<SysOpInfo> req_aicpu_op_info_list;

  if (aicpu_optype_list.empty() && aicpu_tf_optype_list.empty()) {
    GELOGI("No need to check aicpu op type because the list is empty.");
    return SUCCESS;
  }

  std::vector<void *> allocated_mem;

  const size_t aicpu_op_nums = aicpu_optype_list.size();
  const size_t tf_op_nums = aicpu_tf_optype_list.size();
  const size_t op_nums = aicpu_op_nums + tf_op_nums;
  const std::function<void()> callback = [&allocated_mem]() {
    for (auto &mem : allocated_mem) {
      GE_CHK_RT(rtFree(mem));
    }
  };
  GE_MAKE_GUARD(release, callback);

  // malloc sysOpInfoList in SysOpCheckInfo
  void *d_req_op_list = nullptr;
  GE_CHK_RT_RET(rtMalloc(&d_req_op_list, op_nums * sizeof(SysOpInfo), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  allocated_mem.push_back(d_req_op_list);

  // malloc sysOpInfoList in SysOpCheckResp
  void *d_res_op_list = nullptr;
  GE_CHK_RT_RET(rtMalloc(&d_res_op_list, op_nums * sizeof(SysOpInfo), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  allocated_mem.push_back(d_res_op_list);

  // malloc returnCodeList in SysOpCheckResp
  void *d_ret_code_list = nullptr;
  GE_CHK_RT_RET(rtMalloc(&d_ret_code_list, op_nums * sizeof(ReturnCode), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  allocated_mem.push_back(d_ret_code_list);

  for (const auto &op_type : aicpu_optype_list) {
    SysOpInfo op_info;
    // malloc op_type name in SysOpInfo
    void *d_op_type_name = nullptr;
    GE_CHK_RT_RET(rtMalloc(&d_op_type_name, op_type.length(), RT_MEMORY_HBM, GE_MODULE_NAME_U16));

    allocated_mem.push_back(d_op_type_name);
    GE_CHK_RT(rtMemcpy(d_op_type_name, op_type.length(), op_type.c_str(), op_type.length(), RT_MEMCPY_HOST_TO_DEVICE));
    op_info.opType = PtrToValue(d_op_type_name);
    op_info.opLen = op_type.length();
    op_info.kernelsType = CPU_KERNEL;
    req_aicpu_op_info_list.emplace_back(op_info);
  }

  for (const auto &op_type : aicpu_tf_optype_list) {
    SysOpInfo op_info;
    // malloc op_type name in SysOpInfo
    void *d_op_type_name = nullptr;
    GE_CHK_RT_RET(rtMalloc(&d_op_type_name, op_type.length(), RT_MEMORY_HBM, GE_MODULE_NAME_U16));

    allocated_mem.push_back(d_op_type_name);
    GE_CHK_RT(rtMemcpy(d_op_type_name, op_type.size(), op_type.c_str(), op_type.size(), RT_MEMCPY_HOST_TO_DEVICE));
    op_info.opType = PtrToValue(d_op_type_name);
    op_info.opLen = op_type.size();
    op_info.kernelsType = TF_KERNEL;
    req_aicpu_op_info_list.emplace_back(op_info);
  }
  GELOGI("Need check aicpu op nums from attr:[%zu], real nums:[%zu].", op_nums, req_aicpu_op_info_list.size());
  GE_CHK_RT(rtMemcpy(d_req_op_list, sizeof(SysOpInfo) * req_aicpu_op_info_list.size(), req_aicpu_op_info_list.data(),
                     sizeof(SysOpInfo) * req_aicpu_op_info_list.size(), RT_MEMCPY_HOST_TO_DEVICE));

  SysOpCheckInfo op_check_info_req{};
  SysOpCheckResp op_check_info_res{};
  op_check_info_req.opListNum = op_nums;
  op_check_info_req.offSetLen = sizeof(SysOpCheckInfo);
  op_check_info_req.sysOpInfoList = PtrToValue(d_req_op_list);

  op_check_info_res.opListNum = 0U;
  op_check_info_res.isWithoutJson = false;
  op_check_info_res.returnCodeList = PtrToValue(d_ret_code_list);
  op_check_info_res.sysOpInfoList = PtrToValue(d_res_op_list);

  constexpr uint32_t args_size = static_cast<uint32_t>(sizeof(SysOpCheckInfo) + sizeof(SysOpCheckResp));
  const std::unique_ptr<uint8_t[]> args = MakeUnique<uint8_t[]>(static_cast<size_t>(args_size));
  GE_CHECK_NOTNULL(args);
  auto ret = memcpy_s(args.get(), sizeof(SysOpCheckInfo), &op_check_info_req, sizeof(SysOpCheckInfo));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "[Memcpy] Call memcpy failed from src op_check_info_req, ret=%d", ret);
  ret = memcpy_s(args.get() + op_check_info_req.offSetLen, sizeof(SysOpCheckResp), &op_check_info_res,
                 sizeof(SysOpCheckResp));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "[Memcpy] Call memcpy failed from src op_check_info_res, ret=%d", ret);

  rtStream_t stream = nullptr;
  GE_CHK_RT_RET(rtStreamCreate(&stream, 0));
  GE_MAKE_GUARD(stream_guard, [&stream]() {
    GE_CHK_RT(rtStreamDestroy(stream));
  });
  rtArgsEx_t args_info = {};
  args_info.args = const_cast<void *>(static_cast<void *>(args.get()));
  args_info.argsSize = args_size;
  args_info.isNoNeedH2DCopy = 0U;
  GE_CHK_RT(rtCpuKernelLaunchWithFlag(nullptr,
      kernel_name.c_str(), 1U, &args_info, nullptr, stream, RT_KERNEL_DEFAULT));

  GE_CHK_RT_RET(rtStreamSynchronize(stream));

  // Check the response
  const void *const d_op_check_info_res = ValueToPtr(PtrToValue(args.get()) + op_check_info_req.offSetLen);
  ret = memcpy_s(&op_check_info_res, sizeof(SysOpCheckResp),
                 d_op_check_info_res, sizeof(SysOpCheckResp));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "copy check info res failed");

  if (op_check_info_res.isWithoutJson) {
    GELOGI("No need to check aicpu in this scenoria.");
    return SUCCESS;
  }
  const uint64_t res_op_nums = op_check_info_res.opListNum;
  GELOGI("Check aicpu type, is without json: %d, res op num: %" PRIu64 ".",
         static_cast<int32_t>(op_check_info_res.isWithoutJson), res_op_nums);
  if (res_op_nums != 0U) {
    std::vector<ReturnCode> res_ret_code_list(res_op_nums);
    std::vector<SysOpInfo> res_aicpu_op_info_list(res_op_nums);
    GE_CHK_RT(rtMemcpy(res_ret_code_list.data(), sizeof(ReturnCode) * res_op_nums,
                       ValueToPtr(op_check_info_res.returnCodeList),
                       sizeof(ReturnCode) * res_op_nums, RT_MEMCPY_DEVICE_TO_HOST));
    GE_CHK_RT(rtMemcpy(res_aicpu_op_info_list.data(), sizeof(SysOpInfo) * res_op_nums,
                       ValueToPtr(op_check_info_res.sysOpInfoList),
                       sizeof(SysOpInfo) * res_op_nums, RT_MEMCPY_DEVICE_TO_HOST));
    std::string fail_reason;
    for (uint64_t i = 0U; i < res_op_nums; i++) {
      const SysOpInfo &aicpu_info = res_aicpu_op_info_list.at(i);
      GELOGI("Not support aicpu op type: %" PRIu64 ", kernel_type:%d, opLen:%" PRIu64 ", ret_code:%d", aicpu_info.opType,
             aicpu_info.kernelsType, aicpu_info.opLen, res_ret_code_list.at(i));
      std::vector<char> op_name(kOpNameMaxSize);
      GE_CHK_RT(rtMemcpy(op_name.data(), kOpNameMaxSize, ValueToPtr(aicpu_info.opType), aicpu_info.opLen,
                         RT_MEMCPY_DEVICE_TO_HOST));
      const std::string kernel_type = (aicpu_info.kernelsType == TF_KERNEL) ? "TF_KERNEL" : "CPU_KERNEL";
      fail_reason += "op_type: " + std::string(op_name.data()) + " kernel_type: " + kernel_type +
                     " ret code:" + std::to_string(res_ret_code_list.at(i)) + "<0: op_type, 1: format, 2: datatype>\n";
    }
    fail_reason += "not support.";
    REPORT_INNER_ERR_MSG("E19999", "Check aicpu op_type failed, details:%s", fail_reason.c_str());
    GELOGE(FAILED, "[Check][Param] Check aicpu op_type failed. details:%s", fail_reason.c_str());
    return FAILED;
  }

  GELOGI("Cpu kernel launch check optype task success.");
  return SUCCESS;
}

Status ModelManager::CheckAicpuOpList(const GeModelPtr &ge_model) {
  std::vector<std::string> aicpu_optype_list;
  std::vector<std::string> aicpu_tf_optype_list;
  const bool aicpu_need_check = AttrUtils::GetListStr(ge_model, "needCheckCpu", aicpu_optype_list);
  const bool tf_need_check = AttrUtils::GetListStr(ge_model, "needCheckTf", aicpu_tf_optype_list);
  if ((!aicpu_need_check) && (!tf_need_check)) {
    GELOGI("Model:%s No need to check aicpu optype.", ge_model->GetName().c_str());
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(LaunchKernelCheckAicpuOp(aicpu_optype_list, aicpu_tf_optype_list),
                    "[Call][LaunchKernelCheckAicpuOp] failed.");
  return SUCCESS;
}

uint32_t ModelManager::GetRunningFlag(const uint32_t model_id) {
  const auto hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    return static_cast<uint32_t>(hybrid_model->GetRunningFlag());
  }

  const auto davinci_model = GetModel(model_id);
  return (davinci_model != nullptr) ? static_cast<uint32_t>(davinci_model->GetRunningFlag()) : 0U;
}

uint32_t ModelManager::GetDataInputerSize(const uint32_t model_id) {
  const auto hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    return hybrid_model->GetDataInputerSize();
  }

  const auto davinci_model = GetModel(model_id);
  return (davinci_model != nullptr) ? davinci_model->GetDataInputerSize() : 0U;
}

Status ModelManager::SetCallbackHybridLoad(const uint32_t model_id, const GeRootModelPtr &ge_root_model,
                                           const RunAsyncCallbackV2 &callback) {
  if (IsNeedHybridLoad(*ge_root_model)) {
    const auto model = GetHybridModel(model_id);
    GE_CHECK_NOTNULL(model);
    return model->SetRunAsyncListenerCallback(callback);
  }
  return SUCCESS;
}

Status ModelManager::ModelSubscribe(const uint32_t graph_id) {
  const auto &subcribe_info = ProfilingProperties::Instance().GetSubscribeInfo();
  if (!subcribe_info.is_subscribe || (subcribe_info.graph_id != graph_id)) {
    return SUCCESS;
  }
  auto &prof_mgr = ProfilingManager::Instance();
  uint32_t model_id = 0U;
  if (prof_mgr.GetModelIdFromGraph(graph_id, model_id) != SUCCESS) {
    return FAILED;
  }

  return ProfModelSubscribe(subcribe_info.prof_switch, model_id, graph_id);
}

Status ModelManager::ProfModelSubscribe(const uint64_t module, const uint32_t model_id, const uint32_t graph_id) {
  auto &prof_mgr = ProfilingManager::Instance();
  const auto hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(prof_mgr.CheckInitForSubscribe(module, hybrid_davinci_model->GetDeviceId(), model_id));
    return hybrid_davinci_model->ReportProfilingData();
  }

  const auto davinci_model = GetModel(model_id);
  if (davinci_model != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(prof_mgr.CheckInitForSubscribe(module, davinci_model->GetDeviceId(), model_id));
    return davinci_model->ReportProfilingData(graph_id);
  }

  GELOGE(FAILED, "[Call][GetModel] failed, model_id:%u", model_id);
  return FAILED;
}

void ModelManager::RecordTsSnapshot() {
  SET_THREAD_NAME(pthread_self(), "ge_rectssnap");
  while (!stop_monitor_) {
    const INT32 trigger_fd = mmOpen(trigger_file_name_.c_str(), M_RDONLY);
    if (trigger_fd >= 0) {
      (void)mmClose(trigger_fd);
      for (uint32_t i = 0U; (i <= kRecordTimes) && (!stop_monitor_); ++i) {
        GE_CHK_RT(rtGetDevMsg(RT_GET_DEV_RUNNING_STREAM_SNAPSHOT_MSG, &getDevMsgCallback));

        const uint32_t wait_interval = (i < kRecordTimes) ? kRecordIntervalMs : kTriggerScanIntervalMs;
        const std::chrono::milliseconds wait_for_ms(static_cast<uint32_t>(wait_interval));
        std::unique_lock<std::mutex> lk(monitor_mtx_);
        (void)monitor_cv_.wait_for(lk, wait_for_ms,
                                   [this]() -> bool { return stop_monitor_; });
      }
      (void)mmUnlink(trigger_file_name_.c_str());
    } else {
      std::unique_lock<std::mutex> lk(monitor_mtx_);
      (void)monitor_cv_.wait_for(lk, std::chrono::milliseconds(kTriggerScanIntervalMs),
                                 [this]() -> bool { return stop_monitor_; });
    }
  }

  GELOGI("Finish recording snapshot because of stopping");
}

void ModelManager::CreateMonitorThread() {
  const char_t *record_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, record_path);
  if ((record_path != nullptr) && (!std::string(record_path).empty())) {
    GELOGI("Create thread to monitor snapshot");
    stop_monitor_ = false;
    trigger_file_name_ = std::string(record_path) + kPathSeparator + kTriggerFile;

    (void)CreateDirectory(record_path);
    record_file_name_ = std::string(record_path) + kPathSeparator + kRecordFilePrefix + std::to_string(mmGetPid());
    monitor_thread_ = std::thread(&ModelManager::RecordTsSnapshot, this);
  } else {
    GELOGI("Abandon creating thread to monitor snapshot for there's no valid env: NPU_COLLECT_PATH");
  }
}

void ModelManager::ClearMonitorThread() {
  GELOGI("Start to clear monitor thread");
  {
    const std::unique_lock<std::mutex> lk(monitor_mtx_);
    stop_monitor_ = true;
    monitor_cv_.notify_all();
  }

  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }
  GELOGI("Finish to clear monitor thread");
}

void ModelManager::getDevMsgCallback(const char_t *const msg, const uint32_t len) {
  constexpr int32_t open_flags = static_cast<int32_t>(static_cast<uint32_t>(M_CREAT) |
                                                      static_cast<uint32_t>(M_WRONLY) |
                                                      static_cast<uint32_t>(M_APPEND));
  const INT32 record_fd = mmOpen(record_file_name_.c_str(), open_flags);
  if ((record_fd == EN_INVALID_PARAM) || (record_fd == EN_ERROR)) {
    GELOGE(FAILED, "Fail to open file[%s] to record snapshot", record_file_name_.c_str());
    return;
  }

  const mmSsize_t write_count = mmWrite(record_fd, ValueToPtr(PtrToValue(msg)), len);
  if ((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR)) {
    static char_t kRecordSeperate[] = "@@@\n";
    (void)mmWrite(record_fd, static_cast<void *>(&kRecordSeperate[0]), static_cast<uint32_t>(sizeof(kRecordSeperate)));
  }
  (void)mmClose(record_fd);
}

Status ModelManager::LoadTaskForDavinciModel(const DumpProperties &dump_properties) {
  const std::lock_guard<std::mutex> lk(dump_regis_mutex_);
  for (const auto &model : model_map_) {
    const auto davinci_model = model.second;
    davinci_model->SetDumpProperties(dump_properties);
    davinci_model->ReLoadDumpInfo();
  }
  return SUCCESS;
}

Status ModelManager::UnloadTaskForDavinciModel(const DumpProperties &dump_properties) {
  (void)dump_properties;
  const std::lock_guard<std::mutex> lk(dump_regis_mutex_);
  for (const auto &model : model_map_) {
    const auto davinci_model = model.second;
    int32_t device_id = 0;
    bool is_set = false;
    if ((rtGetDevice(&device_id) != RT_ERROR_NONE) || (device_id < 0)) {
      device_id = static_cast<int32_t>(davinci_model->GetDeviceId());
      GE_CHK_RT_RET(rtSetDevice(device_id));
      is_set = true;
    }
    davinci_model->UnloadDumpInfo();
    if (is_set) {
      GE_CHK_RT(rtDeviceReset(device_id));
    }
  }
  return SUCCESS;
}

Status ModelManager::SetCallBackFuncForDumpManager() {
  const std::lock_guard<std::mutex> lk(dump_regis_mutex_);
  if (!DumpManager::GetInstance().CheckIfAclDumpOpen() || is_dump_registered_) {
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(DumpManager::GetInstance().RegisterCallBackFunc(kReloadDumpFuncName, LoadTask),
                    "Register Load func failed");
  GE_CHK_STATUS_RET(DumpManager::GetInstance().RegisterCallBackFunc(kUnloadDumpFuncName, UnloadTask),
                    "Register Unload func failed");
  is_dump_registered_ = true;
  GELOGI("Finish to set funcs for dump manager.");
  return SUCCESS;
}

Status ModelManager::ExternalAllocatorMalloc(const GraphId graph_id, const uint32_t model_id,
                                             const GraphNodePtr &graph_node, const rtStream_t stream) {
  if (stream == nullptr) {
    return SUCCESS;
  }

  bool is_refreshable = false;
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator != nullptr) {
    GELOGD("RunGraphWithStreamAsync with external allocator = %p", external_allocator.get());
    is_refreshable = graph_node->IsFeatureBaseRefreshable();

    // malloc const memory by external allocator
    GE_ASSERT_SUCCESS(MallocConstMemory(graph_id, graph_node, external_allocator),
                      "malloc const memory failed, graph_id:%u.", graph_id);
    if (!is_refreshable) {
      // malloc feature memory by external allocator
      GE_ASSERT_SUCCESS(MallocFeatureMemory(graph_id, model_id, graph_node, external_allocator),
                        "malloc feature memory failed, graph_id:%u.", graph_id);
    }
  }
  return SUCCESS;
}

Status ModelManager::InitOpMasterDeviceSo(const uint32_t &model_id, const GeRootModelPtr &ge_root_model) {
  GE_ASSERT_SUCCESS(ge_root_model->CheckAndSetNeedSoInOM());
  GELOGI("so in om flag:0x%x", ge_root_model->GetSoInOmFlag());
  if (!OpSoStoreUtils::IsSoBinType(ge_root_model->GetSoInOmFlag(), SoBinType::kOpMasterDevice)) {
    return SUCCESS;
  }
  std::unordered_map<std::string, OpSoBinPtr> built_in_so_bins;
  std::unordered_map<std::string, OpSoBinPtr> cust_so_bins;
  GE_ASSERT_SUCCESS(ModelUtils::GetOpMasterDevice(model_id, ge_root_model, built_in_so_bins, cust_so_bins));

  const std::lock_guard<std::mutex> lk(op_master_device_mutex_);
  // 内置AscendXXX-VXX-libopmaster.so通过类型和版本号保证唯一
  for (const auto &item : built_in_so_bins) {
    const auto &so_name = item.first;
    if (built_in_op_master_so_names_to_bin_.find(so_name) == built_in_op_master_so_names_to_bin_.end()) {
      built_in_op_master_so_names_to_bin_.emplace(so_name, item.second);
      GELOGI("[OpMasterDevice][BuiltIn]Save so [%s].", item.first.c_str());
    } else {
      GELOGI("[OpMasterDevice][BuiltIn]The so [%s] has already be saved, will be ignored.", so_name.c_str());
    }
  }

  // 自定义libcust_opmaster.so通过so内容去重保证唯一
  for (const auto &item : cust_so_bins) {
    const auto &so_name = item.first;
    const auto &so_bin = item.second;
    GE_ASSERT_NOTNULL(so_bin);
    auto so_data = so_bin->GetBinData();
    GE_ASSERT_NOTNULL(so_data);
    const auto &so_len = so_bin->GetBinDataSize();
    std::string str_so_data(so_data, so_data + so_len);
    const auto &iter = cust_op_master_so_datas_to_name_.find(str_so_data);
    if (iter == cust_op_master_so_datas_to_name_.end()) {
      cust_op_master_so_names_to_bin_.emplace(so_name, so_bin);
      cust_op_master_so_names_to_unique_name_.emplace(so_name, so_name);
      cust_op_master_so_datas_to_name_.emplace(str_so_data, so_name);
      GELOGI("[OpMasterDevice][Custom] Save so [%s].", so_name.c_str());
    } else {
      cust_op_master_so_names_to_unique_name_.emplace(so_name, iter->second);
      GELOGI("[OpMasterDevice][Custom] The so [%s] will be replaced by [%s] as th same content.", so_name.c_str(),
             iter->second.c_str());
    }
  }
  return SUCCESS;
}

uint8_t *ModelManager::MallocWeightsMem(const std::string &weights_mem_id, const uint32_t device_id,
                                        const size_t weights_size) {
  const std::lock_guard<std::mutex> lock(weights_mem_mtx_);
  uint8_t *weight_mem_base = nullptr;
  if (weights_mem_ids_to_addr_info_.find(weights_mem_id) != weights_mem_ids_to_addr_info_.end()) {
    weights_mem_ids_to_addr_info_[weights_mem_id].shared_num++;
    weight_mem_base = weights_mem_ids_to_addr_info_[weights_mem_id].weight_addr_;
    GELOGI("[WeightsMem][Reuse] weights mem id[%s] reuse memory addr[0x%" PRIx64 "] mem_size[%zu]",
      weights_mem_id.c_str(), weight_mem_base, weights_size);
  } else {
    auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
    const std::string purpose("weights memory in inference network");
    weight_mem_base = mem_instance.MallocMemory(purpose, weights_size, device_id);
    GELOGI("[WeightsMem][Malloc] weights mem id[%s] malloc memory addr[0x%" PRIx64 "] mem_size[%zu]",
      weights_mem_id.c_str(), weight_mem_base, weights_size);
    if (GetGraphMaxParallelModelNum() > 1) {
      weights_mem_ids_to_addr_info_[weights_mem_id].shared_num++;
      weights_mem_ids_to_addr_info_[weights_mem_id].weight_addr_ = weight_mem_base;
      GELOGI("[WeightsMem][Store] weights mem id[%s] store memory addr[0x%" PRIx64 "] mem_size[%zu]",
        weights_mem_id.c_str(), weight_mem_base, weights_size);
    }
  }
  return weight_mem_base;
}

Status ModelManager::FreeWeightsMem(const std::string &weights_mem_id, const uint32_t device_id,
                                    uint8_t *weights_mem_base) {
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  const std::lock_guard<std::mutex> lock(weights_mem_mtx_);
  auto iter = weights_mem_ids_to_addr_info_.find(weights_mem_id);
  if (iter != weights_mem_ids_to_addr_info_.end()) {
    GE_ASSERT_EQ(iter->second.weight_addr_, weights_mem_base);
    auto &shared_num = iter->second.shared_num;
    if (shared_num == 1UL) {
      GE_ASSERT_SUCCESS(mem_instance.FreeMemory(PtrToPtr<uint8_t, void>(weights_mem_base), device_id),
                        "failed to free Weight.");
      GELOGI("[WeightsMem][Free] weights mem id[%s] free memory addr[0x%" PRIx64 "] as shared num is %" PRIu64,
             weights_mem_id.c_str(), weights_mem_base, shared_num);
      weights_mem_ids_to_addr_info_.erase(weights_mem_id);
    } else {
      GELOGI("[WeightsMem][NotFree] weights mem id[%s] no need free memory addr[0x%" PRIx64 "] as shared num is %" PRIu64,
             weights_mem_id.c_str(), weights_mem_base, shared_num);
      --shared_num;
    }
  } else {
    GE_ASSERT_SUCCESS(mem_instance.FreeMemory(PtrToPtr<uint8_t, void>(weights_mem_base), device_id),
                      "failed to free Weight.");
    GELOGI("[WeightsMem][Free] weights mem id[%s] free memory addr[0x%" PRIx64 "]",
      weights_mem_id.c_str(), weights_mem_base);
  }
  return SUCCESS;
}
}  // namespace ge
