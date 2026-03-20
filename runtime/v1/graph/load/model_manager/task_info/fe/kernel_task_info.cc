/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/kernel_task_info.h"

#include <securec.h>
#include "common/checker.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "aicpu_task_struct.h"
#include "framework/common/types.h"
#include "graph/load/model_manager/memory_app_type_classifier.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "common/op_tiling/tiling_memcheck.h"
#include "common/op_tiling/tiling_dfx.h"
#include "common/dump/dump_utils.h"
#include "runtime/kernel.h"
#include "register/op_tiling_registry.h"
#include "adump_pub.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "graph/load/model_manager/kernel/kernel_register_info_builder.h"
#include "acl/acl_rt.h"

namespace ge {
namespace {
const std::string kAllShapeInAicpu = "_AllShape";
const std::string kAttrNameAtomicWspMode = "wspMode";
const std::string kWspFoldedMode = "folded";
const std::string kWspUnfoldedMode = "unfolded";
constexpr char_t const *kMaxTilingSize = "op_para_size";
constexpr uint64_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr char_t const *kMaxAtomicCleanTilingSize = "atomic_op_para_size";
const std::string kLocalMemorySize = "local_memory_size";

constexpr uint32_t kAddressLen = static_cast<uint32_t>(sizeof(uint64_t));
constexpr uint32_t kUBAlignedLen = 32U;
constexpr size_t kArgsInputDesc = 0U;
constexpr size_t kArgsInputAddr = 1U;
constexpr size_t kArgsOutputDesc = 2U;
constexpr size_t kArgsOutputAddr = 3U;
constexpr size_t kArgsAttrHandle = 4U;
constexpr uint32_t k2BitsMask = 0x00000003U;   // 2  bits, 0000,0011

constexpr int64_t kDefaultDimInfo = 0x100000001;
constexpr uint64_t kDefaultShapeNum = 0x100000000U;

constexpr uint64_t kBitFlag8 = 0x00FFFFFFFFFFFFFFUL;
constexpr uint64_t kLevel2BitFlagCustom = 0x0100000000000000UL;
constexpr uint64_t kLevel2BitFlagWithShape = 0x0200000000000000UL;
constexpr uint64_t kLevel2BitFlagTilingData = 0x0300000000000000UL;

bool IsAllKernel(const ModelTaskType task_type) {
  return (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) || (task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL);
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

// 为了做到tensor.GetTensorData().GetAddr()返回地址的可刷新，这里需要做两次8字节对齐，即TensorData首地址对其以及Tensor大小的对齐
void GetAddrAlignedGertTensorSize(size_t &io_aligned_offset, size_t &double_aliged_tensor_size) {
  gert::Tensor tensor;
  tensor.MutableTensorData();
  const size_t raw_addr_offset =
      static_cast<size_t>(ge::PtrToValue(&tensor.MutableTensorData()) - ge::PtrToValue(&tensor));
  io_aligned_offset = ge::MemSizeAlign(raw_addr_offset, static_cast<uint32_t>(sizeof(uint64_t)));
  double_aliged_tensor_size = sizeof(gert::Tensor) + io_aligned_offset;
  double_aliged_tensor_size = ge::MemSizeAlign(double_aliged_tensor_size, static_cast<uint32_t>(sizeof(uint64_t)));
}

ge::Status GetMemCheckStartSize(const ge::OpDescPtr &op_desc,
                                const int64_t origin_tiling_data_size,
                                int64_t &memcheck_start_size) {
  int64_t ori_param_size = 0LL;
  (void)ge::AttrUtils::GetInt(op_desc, optiling::kOriOpParaSize, ori_param_size);
  if (ori_param_size > 0LL) {
    // tik场景下TilingAppendMem添加的数据需要从偏移为ori_param_size的地址开始添加，此处需要将DataSize设置成ori_param_size
    GE_ASSERT_TRUE(origin_tiling_data_size <= ori_param_size);
    GELOGI("Current tiling data size: %zu, set ori_para_size to %lld by attr, op_name: %s",
          origin_tiling_data_size, ori_param_size, op_desc->GetNamePtr());
  } else {
    ori_param_size = ((origin_tiling_data_size + sizeof(int64_t) - 1UL) / sizeof(int64_t)) * sizeof(int64_t);
    GELOGI("Current tiling data size: %zu, set ori_param_size to %lld by aligned by %zu, op_name: %s",
          origin_tiling_data_size, ori_param_size, sizeof(int64_t), op_desc->GetNamePtr());
  }
  memcheck_start_size = ori_param_size - origin_tiling_data_size;
  return ge::SUCCESS;
}

void AppendShapeInfo(const ge::GeShape &shape, std::vector<int64_t> &shape_info_vec) {
  const auto dim_num = shape.GetDimNum();
  shape_info_vec.push_back(dim_num);
  GELOGD("[AppendShapeInfo] Append shape num: %zu", dim_num);
  if (dim_num > 0) {
    const auto dims = shape.GetDims();
    for (size_t i = 0; i < dim_num; i++) {
      shape_info_vec.push_back(dims[i]);
    }
  }
}

void FormatArgsException(uint64_t *host_addr, const std::vector<int64_t> &args_size,
                         const std::vector<int64_t> &shape_size, uint64_t atomic_index) {
  uint64_t *addr = host_addr;
  for (size_t index = 0UL; index < args_size.size(); index++) {
    *addr = static_cast<uint64_t>(args_size[index]);
    GELOGI("[TilingAppendDfxInfo] size idx[%zu], val[%llu], atomic index[%llu]", index, *addr, atomic_index);
    addr++;
  }

  for (size_t index = 0UL; index < shape_size.size(); index++) {
    *addr = static_cast<uint64_t>(shape_size[index]);
    GELOGI("[TilingAppendDfxInfo] shape idx[%zu], val[%llu], atomic index[%llu]",
      args_size.size() + index, *addr, atomic_index);
    addr++;
  }

  return;
}

ge::Status UpdateDfxArgsAndShapeSize(const ge::OpDescPtr &op_desc,
                                     const std::vector<optiling::ArgsIndexToIoIndex> &args_idx_to_io_idx_vec,
                                     std::vector<int64_t> &args_size_vec,
                                     std::vector<int64_t> &shape_size_vec) {
  auto input_descs = op_desc->GetAllInputsDescPtr();

 // 更新size以及shape
  for (size_t i = 0U; i < args_idx_to_io_idx_vec.size(); i++) {
    size_t io_index = args_idx_to_io_idx_vec[i].io_index;
    size_t args_index = args_idx_to_io_idx_vec[i].args_index;
    GE_ASSERT(args_index < args_size_vec.size(),
      "args index [%zu] not less than args list size [%zu]", args_index, args_size_vec.size());

    if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kInput) {
      const auto tensor = input_descs.at(io_index);
      GE_ASSERT_NOTNULL(tensor);
      int64_t tensor_size = 0;
      GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(*tensor, tensor_size));
      GELOGI("Update input tensor size, node[%s], index:%zu, args index: %zu, io index: %zu, tensor size: %lld",
        op_desc->GetNamePtr(), i, args_index, io_index, tensor_size);
      args_size_vec[args_index] = tensor_size;
      // shape
      AppendShapeInfo(tensor->GetShape(), shape_size_vec);
    } else if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kOutput) {
      const auto tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(io_index));
      int64_t tensor_size = 0L;
      GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(tensor, tensor_size));
      GELOGI("Update output tensor size, node[%s], index:%zu, args index: %zu, io index: %zu, tensor size: %lld",
          op_desc->GetNamePtr(), i, args_index, io_index, tensor_size);
      args_size_vec[args_index] = tensor_size;
      // shape
      shape_size_vec.push_back(0);
    }
  }
  return ge::SUCCESS;
}

ge::Status ConstructDfxInfo(const ge::OpDescPtr &op_desc,
                            const optiling::OpRunInfoV2 &run_info,
                            const std::vector<ge::ArgDesc> &arg_descs,
                            std::string &dfx_info) {
  bool is_mem_check_enable = false;
  (void)ge::AttrUtils::GetBool(op_desc, optiling::kMemoryCheck, is_mem_check_enable);
  bool is_args_exception_enable = Adx::AdumpGetDumpSwitch(Adx::DumpType::ARGS_EXCEPTION);
  if (!is_mem_check_enable && !is_args_exception_enable) {
    return ge::SUCCESS;
  }

  GE_ASSERT_NOTNULL(op_desc);
  auto input_descs = op_desc->GetAllInputsDescPtr();

  // 获取size
  std::vector<int64_t> args_size_vec;
  std::vector<optiling::ArgsIndexToIoIndex> args_idx_to_io_idx_vec;
  if (!arg_descs.empty()) {
    GE_ASSERT_SUCCESS(
      optiling::TilingDfx::GetArgsSizeWithArgsFormat(op_desc, arg_descs, args_size_vec, args_idx_to_io_idx_vec));
  } else  {
    GELOGI("OP [%s] not has formatted args_format. input desc size [%zu], out desc size [%zu]",
      op_desc->GetNamePtr(),input_descs.size(), op_desc->GetOutputsSize());
    GE_ASSERT_SUCCESS(optiling::TilingDfx::GetArgsSizeWithoutArgsFormat(input_descs.size(),
      op_desc->GetOutputsSize(), args_size_vec, args_idx_to_io_idx_vec));
  }

  std::vector<int64_t> shape_size_vec;
  GE_ASSERT_SUCCESS(UpdateDfxArgsAndShapeSize(op_desc, args_idx_to_io_idx_vec, args_size_vec, shape_size_vec));
  (void)args_size_vec.insert(args_size_vec.cend(),
      run_info.GetAllWorkspaces().cbegin(), run_info.GetAllWorkspaces().cend());

  // tiling data为0的场景 或者args Size 为0的场景，直接返回
  const int64_t tiling_data_size = static_cast<int64_t>(run_info.GetAllTilingData().str().size());
  if ((tiling_data_size == 0) || (args_size_vec.size() == 0U)) {
    return ge::SUCCESS;
  }

  uint64_t atomic_index = 0UL;
  if (is_args_exception_enable) {
    size_t total_size = args_size_vec.size() + shape_size_vec.size();
    uint64_t *host_addr = ge::PtrToPtr<void, uint64_t>(Adx::AdumpGetDFXInfoAddrForStatic(total_size, atomic_index));
    GE_ASSERT_NOTNULL(host_addr, "total size[%zu]", total_size);
    FormatArgsException(host_addr, args_size_vec, shape_size_vec, atomic_index);
  }

  if (is_mem_check_enable) {
    int64_t max_size = -1;
    if (!ge::AttrUtils::GetInt(op_desc, kMaxTilingSize, max_size) || max_size < 0) {
      GELOGI("No max tiling size in opdesc.");
      max_size = static_cast<int64_t>(kMaxTilingDataSize);
    }

    const auto memcheck_info_capcity = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
    GELOGI("Get memcheck info capcity: %zu, op_name: %s", memcheck_info_capcity, op_desc->GetNamePtr());
    const auto memcheck_data_holder = gert::TilingData::CreateCap(memcheck_info_capcity);
    auto memcheck_data = reinterpret_cast<gert::TilingData *>(memcheck_data_holder.get());
    int64_t memcheck_start_size = 0L;
    GE_ASSERT_SUCCESS(GetMemCheckStartSize(op_desc, tiling_data_size, memcheck_start_size));
    memcheck_data->SetDataSize(static_cast<size_t>(memcheck_start_size));

    // append size
    for (size_t i = 0U; i < args_size_vec.size(); i++) {
      GELOGI("[TilingAppendDfxInfo] size idx[%zu], val[%lld]", i, args_size_vec[i]);
    }
    GE_ASSERT_SUCCESS(memcheck_data->Append(args_size_vec.data(), args_size_vec.size()));
    GELOGI("Op name[%s] memcheck info size: %lld, start size: %lld",
      op_desc->GetNamePtr(), memcheck_data->GetDataSize(), memcheck_start_size);
    dfx_info = std::string(reinterpret_cast<ge::char_t *>(memcheck_data->GetData()), memcheck_data->GetDataSize());
  }

  if (is_args_exception_enable) {
    dfx_info += std::string(reinterpret_cast<ge::char_t *>(&atomic_index), sizeof(uint64_t));
  }

  return ge::SUCCESS;
}
}  // namespace

static bool IsSeparatelyCleanTask(const OpDescPtr &op_desc, const std::string &kernel_name) {
  const std::string attr_key_kernel_name = op_desc->GetName() + "_atomic_kernelname";
  std::string attr_val_kernel_name;
  bool has_set_atomic_kernel_name =
      ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, "_atomic_kernelname", attr_val_kernel_name) &&
      (kernel_name.compare(attr_val_kernel_name) == 0);
  // fe set atomic clean/memset kernel name into corresponding nodes
  // as this, ge can use this attr to find separate atomic clean/memset op
  has_set_atomic_kernel_name =
      has_set_atomic_kernel_name && (op_desc->GetType() != ge::ATOMICADDRCLEAN) && (op_desc->GetType() != ge::MEMSET);
  if (!has_set_atomic_kernel_name) {
    return false;
  }
  const bool is_need_atomic_clean = IsNeedAtomicCleanTask(op_desc);
  GELOGD("Node: %s has_set_atomic_kernel_name: %d, is_need_atomic_clean: %d.", op_desc->GetNamePtr(),
         static_cast<int32_t>(has_set_atomic_kernel_name), static_cast<int32_t>(is_need_atomic_clean));
  return is_need_atomic_clean;
}

static bool IsWspAddrFolded(const OpDescPtr &op_desc) {
  std::string wsp_mode = kWspUnfoldedMode;
  return ge::AttrUtils::GetStr(op_desc, kAttrNameAtomicWspMode, wsp_mode) && (wsp_mode == kWspFoldedMode);
}

void KernelTaskInfo::UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs) {
  // todo: model args manager功能适配完毕后, 此处新增input_data_addrs_和iow_addrs.input_logic_addrs相等的校验
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] = (iow_addrs.input_logic_addrs.empty())
        ? input_data_addrs_[i] : iow_addrs.input_logic_addrs[i].logic_addr;
    input_mem_types_[i] = (iow_addrs.input_logic_addrs.empty())
        ? input_mem_types_[i] : iow_addrs.input_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] = (iow_addrs.output_logic_addrs.empty())
        ? output_data_addrs_[i] : iow_addrs.output_logic_addrs[i].logic_addr;
    output_mem_types_[i] = (iow_addrs.output_logic_addrs.empty())
        ? output_mem_types_[i] : iow_addrs.output_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    workspace_addrs_[i] = (iow_addrs.workspace_logic_addrs.empty())
        ? workspace_addrs_[i] : iow_addrs.workspace_logic_addrs[i].logic_addr;
    workspace_mem_types_[i] = (iow_addrs.workspace_logic_addrs.empty())
        ? workspace_mem_types_[i] : iow_addrs.workspace_logic_addrs[i].memory_type;
  }
}

rtBinHandle KernelTaskInfo::GetBinHandle(const domi::TaskDef &task_def) const {
  if (IsAllKernel(task_type_) || ModelUtils::IsAICoreKernel(kernel_type_)) {
    auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kAicore);
    GE_ASSERT_NOTNULL(kernel_handles_manager);
    KernelRegisterInfo register_info;
    GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructAicoreRegisterInfo(op_desc_,
        is_separately_clean_task_, davinci_model_->GetModelId(), register_info));
    const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
    return kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  }
  if (task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL) {
    auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kCustAicpu);
    GE_ASSERT_NOTNULL(kernel_handles_manager);
    KernelRegisterInfo register_info;
    GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructTilingDeviceRegisterInfo(task_def.kernel().so_name(),
        davinci_model_->GetModelId(), register_info));
    const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
    auto tiling_device_bin_handle = kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
    ModelManager::GetInstance().SetPlatformBinHandle(tiling_device_bin_handle);
    return tiling_device_bin_handle;
  }
  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kCustAicpu);
    GE_ASSERT_NOTNULL(kernel_handles_manager);
    KernelRegisterInfo register_info;
    GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructCustAicpuRegisterInfo(op_desc_, register_info));
    const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
    return kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  }
  if ((kernel_type_ == ccKernelType::AI_CPU_KFC) || (kernel_type_ == ccKernelType::AI_CPU)) {
    auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kAicpu);
    GE_ASSERT_NOTNULL(kernel_handles_manager);
    KernelRegisterInfo register_info;
    const std::string op_kernel_lib = (kernel_type_ == ccKernelType::AI_CPU_KFC) ? "KFCKernel" : "AICPUKernel";
    GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructAicpuRegisterInfo(op_desc_->GetType(),
        task_def.kernel().so_name(), task_def.kernel().kernel_name(), op_kernel_lib, register_info));
    const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
    return kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  }
  GELOGW("[%s][%s] task type: %u, kernel type: %u is not support bin handle.",
      op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), task_type_, kernel_type_);
  return nullptr;
}

void KernelTaskInfo::SetExceptionCallback(rtBinHandle bin_handle) {
  const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
    static_cast<gert::OppImplVersionTag>(op_desc_->GetOppImplVersion()));
  if (space_registry == nullptr) {
    GELOGW("GetSpaceRegistry returned nullptr for node %s", op_desc_->GetNamePtr());
    return;
  }
  const auto op_impl = space_registry->GetOpImpl(op_desc_->GetType().c_str());
  if (op_impl != nullptr) {
    auto exception_func = op_impl->exception_func;
    if (exception_func != nullptr) {
      GE_CHK_RT_EXEC(rtBinarySetExceptionCallback(bin_handle, reinterpret_cast<rtOpExceptionCallback>(exception_func), nullptr), return);
      GELOGI("Set exception callback for node %s.", op_desc_->GetNamePtr());
    } else {
      GELOGW("No exception callback found for node %s.", op_desc_->GetNamePtr());
    }
  } else {
    GELOGW("Failed to get op registry func node %s.", op_desc_->GetNamePtr());
  }
}

rtFuncHandle KernelTaskInfo::GetFuncHandle(const domi::TaskDef &task_def) {
  auto bin_handle = GetBinHandle(task_def);
  GE_ASSERT_NOTNULL(bin_handle);
  GE_ASSERT_NOTNULL(op_desc_);
  SetExceptionCallback(bin_handle);
  if (IsAllKernel(task_type_)) {
    return KernelHandleUtils::GetFuncHandle(bin_handle, tiling_key_);
  }
  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    return KernelHandleUtils::GetCustAicpuFuncHandle(bin_handle,
        op_desc_->GetType(), task_def.kernel().kernel_name());
  }
  if (ModelUtils::IsAICoreKernel(kernel_type_)) {
    std::string attr_kernel_name;
    if (is_separately_clean_task_) {
      attr_kernel_name = kAtomicPrefix + op_desc_->GetName() + "_kernelname";
    } else {
      attr_kernel_name = op_desc_->GetName() + "_kernelname";
    }
    std::string kernel_name;
    (void)AttrUtils::GetStr(op_desc_, attr_kernel_name, "_kernelname", kernel_name);
    GELOGD("[%s][%s] get kernel name: %s from attr: %s.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(),
        kernel_name.c_str(), attr_kernel_name.c_str());
    return KernelHandleUtils::GetFuncHandle(bin_handle, kernel_name);
  }
  if ((kernel_type_ == ccKernelType::AI_CPU_KFC) || (kernel_type_ == ccKernelType::AI_CPU)) {
    return KernelHandleUtils::GetFuncHandle(bin_handle, op_desc_->GetType());
  }
  return nullptr;
}

Status KernelTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                            const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                            const IowAddrs &iow_addrs) {
  GE_CHECK_NOTNULL(davinci_model);
  (void)persistent_workspace;
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  UpdateIoAndWorkspaceAddrs(iow_addrs);
  if (IsAllKernel(task_type_)) {
    GE_CHK_STATUS_RET_NOLOG(InitKernelWithHandle(task_def, args));
  } else {
    GE_CHK_STATUS_RET_NOLOG(InitKernel(task_def, args));
  }
  // om兼容场景，新流程已经归一到custaicpu
  if ((task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL) && (kernel_type_ == ccKernelType::AI_CPU)) {
    GELOGI("[%s][%s] Tiling device built in kernel no need init func handle",
        op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  } else {
    func_handle_ = GetFuncHandle(task_def);
    GE_ASSERT_NOTNULL(func_handle_);
  }
  io_addr_mem_types_.resize(io_addrs_.size(), static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), io_addrs_,
      io_addr_mem_types_, {op_desc_->GetName(), op_desc_->GetType()}));

  if ((kernel_type_ == ccKernelType::AI_CPU) ||
      (kernel_type_ == ccKernelType::CUST_AI_CPU) ||
      (kernel_type_ == ccKernelType::AI_CPU_KFC)) {
    uint32_t pls = static_cast<uint32_t>(args_placement_);
    GE_ASSERT_TRUE(args[pls].len >= args_offset_from_pls_);
    const errno_t sec_ret = memcpy_s(ValueToPtr(PtrToValue(args[pls].host_addr) + args_offset_from_pls_), args[pls].len - args_offset_from_pls_,
      args_addr_.data(), static_cast<size_t>(args_size_));
    GE_ASSERT_EOK(sec_ret, "[Call][Memcpy] failed, size:%zu, ret:%d", args[pls].len, sec_ret);
  }

  GELOGI("KernelTaskInfo Init Success, node :%s, logic stream id: %u, stream: %p.",
    op_desc_->GetName().c_str(), task_def.stream_id(), stream_);
  return SUCCESS;
}

Status KernelTaskInfo::InitKernel(const domi::TaskDef &task_def, const PisToArgs &args) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  op_index_ = context.op_index();
  op_desc_ = (op_desc_ != nullptr) ? op_desc_ : davinci_model_->GetOpByIndex(op_index_);
  GE_CHECK_NOTNULL(op_desc_);
  (void)AttrUtils::GetBool(op_desc_, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync_op_);
  (void)AttrUtils::GetInt(op_desc_, kLocalMemorySize, local_memory_size_);

  // Old model will not take this value, its default value is 0,need to convert to the real default value 1.
  block_dim_ = (kernel_def.block_dim() == 0U) ? 1U : kernel_def.block_dim();
  cfg_.blockDimOffset = kernel_def.block_dim_offset();
  is_block_task_prefetch_ = kernel_def.is_block_task_prefetch();
  is_separately_clean_task_ = IsSeparatelyCleanTask(op_desc_, kernel_def.kernel_name());

  is_addrs_folded_ = IsWspAddrFolded(op_desc_);
  GE_CHK_STATUS_RET_NOLOG(InitKernelByContext(task_def, context, args));
  Status ret = FAILED;
  if (ModelUtils::IsAICoreKernel(kernel_type_)) {
    ret = InitTVMTask(kernel_def);
  } else if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    ret = InitAICPUCustomTask(op_desc_, kernel_def);
  } else if (kernel_type_ == ccKernelType::AI_CPU_KFC) {
    ret = InitAicpuKfcTask(kernel_def);
  } else if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    ret = InitAicpuTask(op_desc_, kernel_def);
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Node op:%s(%s) kernel type invalid", op_desc_->GetName().c_str(),
                       op_desc_->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Node op:%s(%s) kernel type invalid", op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str());
    return ret;
  }
  if (!is_separately_clean_task_) {
    InitDumpArgs(io_addr_offset_);
  }
  GELOGD("KernelTaskInfo %s init finish, result=%u.", op_desc_->GetNamePtr(), ret);
  return ret;
}

Status KernelTaskInfo::InitKernelWithHandle(const domi::TaskDef &task_def, const PisToArgs &args) {
  const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
  const domi::KernelContext &context = kernel_def.context();
  op_index_ = context.op_index();
  op_desc_ = (op_desc_ != nullptr) ? op_desc_ : davinci_model_->GetOpByIndex(op_index_);
  GE_CHECK_NOTNULL(op_desc_);
  (void)AttrUtils::GetBool(op_desc_, ge::ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync_op_);
  (void)AttrUtils::GetInt(op_desc_, kLocalMemorySize, local_memory_size_);

  const auto tiling_info = op_desc_->GetExtAttr<std::shared_ptr<optiling::utils::OpRunInfo>>(ge::ATTR_NAME_OP_RUN_INFO);
  if ((tiling_info != nullptr) && (*tiling_info != nullptr)) {
    tiling_key_ = (*tiling_info)->GetTilingKey();
  }

  // Old model will not take this value, its default value is 0,need to convert to the real default value 1.
  block_dim_ = (kernel_def.block_dim() == 0U) ? 1U : kernel_def.block_dim();
  cfg_.blockDimOffset = kernel_def.block_dim_offset();
  is_block_task_prefetch_ = kernel_def.is_block_task_prefetch();
  GE_CHK_STATUS_RET_NOLOG(InitKernelByContext(task_def, context, args));

  if (!ModelUtils::IsAICoreKernel(kernel_type_)) {
    GELOGE(FAILED, "Op[%s] kernel type[%d] invalid.", op_desc_->GetName().c_str(), static_cast<int32_t>(kernel_type_));
    return FAILED;
  }
  GE_CHK_STATUS_RET_NOLOG(InitTVMTask(kernel_def));

  InitDumpArgs(io_addr_offset_);
  GELOGD("KernelTaskInfo %s init with handle finish.", op_desc_->GetNamePtr());
  return SUCCESS;
}

Status KernelTaskInfo::InitKernelByContext(const domi::TaskDef &task_def, const domi::KernelContext &context,
                                           const PisToArgs &args) {
  operator_ = (sk_sub_operator_ != nullptr) ? sk_sub_operator_ : davinci_model_->GetOperatorByIndex(context.op_index());
  GE_CHECK_NOTNULL(operator_);

  if (context.origin_op_index_size() > CC_FUSION_OP_MAX) {
    REPORT_INNER_ERR_MSG("E19999", "origin_op_index_size:%d is more than CC_FUSION_OP_MAX(%d), op:%s(%s), check invalid",
        context.origin_op_index_size(), CC_FUSION_OP_MAX, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]invalid, origin_op_index_size:%d is more than CC_FUSION_OP_MAX(%d), op:%s(%s)",
           context.origin_op_index_size(), CC_FUSION_OP_MAX, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return PARAM_INVALID;
  }

  kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
  GELOGD("KernelTaskInfo init start, kernel_type: %d.", static_cast<int32_t>(kernel_type_));
  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    args_offset_from_pls_ = ge::MemSizeAlign(sizeof(aicpu::AicpuParamHead), sizeof(uintptr_t)) - sizeof(aicpu::AicpuParamHead);
  }
  ctx_.opIndex = context.op_index();

  GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement_)].dev_addr != 0U),
                 "[Check][Param] Op:%s, dev addr is nullptr.", op_desc_->GetName().c_str());
  args_ = ValueToPtr(args[static_cast<size_t>(args_placement_)].dev_addr + args_offset_from_pls_);

  GE_ASSERT_SUCCESS(CopyTilingDataIfNeeded(), "Copy tiling data to device failid.");

  const bool assemble_by_args_manager =
      (!args_format_holder_.arg_descs.empty()) && (kernel_type_ != ccKernelType::CUSTOMIZED) && (!is_addrs_folded_);
  if (assemble_by_args_manager) {
    GE_ASSERT_SUCCESS(AssembleIoByArgsFormat(), "[Assemble][Addresses] failed, op = %s.", op_desc_->GetNamePtr());
  } else {
    GE_ASSERT_SUCCESS(SetIoAddrs(), "[Set][Addresses] failed, op = %s.", op_desc_->GetName().c_str());
  }
  InitFusionDumpInfo(op_desc_, task_def);

  return SUCCESS;
}

void KernelTaskInfo::UpdateTaskId() {
  if (davinci_model_ != nullptr) {
    GE_CHK_RT_EXEC(rtsGetThreadLastTaskId(&task_id_), return);
    GE_CHK_RT_EXEC(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)), return);
    GELOGD("UpdateTaskId:UpdateTaskId [%u], stream id [%u]:", task_id_, stream_id_);
  }
}

Status KernelTaskInfo::DistributeTask() {
  // call rtKernelLaunch for current task
  const string op_name = op_desc_->GetName();
  GELOGD("Start to launch kernel of %s.", op_name.c_str());
  SetTaskTag(op_name.c_str());
  LaunchKernelParam launch_kernel_param;
  launch_kernel_param.args = args_;
  launch_kernel_param.args_size =
      customized_args_info_.customized_aligned ? customized_args_info_.kernel_def_args_size : args_size_;
  // aicore和aicpu_kfc算子设置成block_dim，其他类型的算子设置为1
  if (IsAllKernel(task_type_) || ModelUtils::IsAICoreKernel(kernel_type_) || kernel_type_ == ccKernelType::AI_CPU_KFC) {
    launch_kernel_param.block_dim = block_dim_;
    GE_ASSERT_SUCCESS(ReportL0ExceptionDumpInfo(op_desc_, l0_dump_list_), "[%s] report l0 exception dump addr failed",
                      op_desc_->GetNamePtr());
  } else {
    launch_kernel_param.block_dim = 1U;
  }
  launch_kernel_param.stream = stream_;
  launch_kernel_param.launch_config.schedule_mode = cfg_.schemMode;
  launch_kernel_param.launch_config.local_memory_size = local_memory_size_;
  launch_kernel_param.launch_config.block_dim_offset = cfg_.blockDimOffset;
  launch_kernel_param.launch_config.is_block_task_prefetch = is_block_task_prefetch_;
  launch_kernel_param.launch_config.is_data_dump = is_data_dump_;
  if ((task_type_ == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL) || (task_type_ == ModelTaskType::MODEL_TASK_VECTOR_KERNEL)) {
    launch_kernel_param.launch_config.engine_type = RT_ENGINE_TYPE_AIV;
  }
  bool op_exec_never_timeout = false;
  if (AttrUtils::GetBool(op_desc_, public_attr::OP_EXEC_NEVER_TIMEOUT, op_exec_never_timeout)
      && op_exec_never_timeout) {
    launch_kernel_param.launch_config.time_out = 0;
    GELOGI("op %s type %s set never timeout", op_name.c_str(), op_desc_->GetTypePtr());
  }
  GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(func_handle_, launch_kernel_param));
  // set for task_id_
  UpdateTaskId();
  if (IsAllKernel(task_type_) || ModelUtils::IsAICoreKernel(kernel_type_)) {
    std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
    std::shared_ptr<TilingContextAddr> tiling_context_addr = op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
    if (tiling_context_addr != nullptr) {
      auto sink_info = MakeShared<TilingSinkTaskInfo>();
      GE_ASSERT_NOTNULL(sink_info);
      sink_info->task_id = task_id_;
      sink_info->ffts_task_handle = nullptr;
      sink_info->stream = stream_;
      GE_ASSERT_TRUE(op_desc_->SetExtAttr(kTilingSinkTaskInfo, sink_info));
      is_support_redistribute_ = false;
      GELOGW("Redistribute is not supported in tiling, is_support_redistribute_ set to false");
      return SUCCESS;
    }
  }
  is_support_redistribute_ = true;
  return SUCCESS;
}

Status KernelTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("KernelTaskInfo Distribute Start, op: %s", op_desc_->GetName().c_str());
  const TaskProfGuarder prof_guarder(this);
  const bool is_built_tiling_device = (kernel_type_ == ccKernelType::AI_CPU)
      && (task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL);
  if (is_built_tiling_device) {
    // Use the 5th and 6th bits of dump_flag_ indicate the value of topic_type.
    // xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
    // xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
    // xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
    // xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
    dump_flag_ = dump_flag_ | static_cast<uint32_t>(deploy_type_flag_);
    // Use the 9th-11th bits of dump_flag_ indicate the value of qos. 12th indicate qos on/off
    // xxxxxxxx xxxxxxxx xxxx0000 xxxxxxxx: qos off
    // xxxxxxxx xxxxxxxx xxxx1000 xxxxxxxx: qos on, level=0(min level)
    // xxxxxxxx xxxxxxxx xxxx1111 xxxxxxxx: qos on, level=7(max level)
    dump_flag_ = dump_flag_ | qos_level_flag_;
    GELOGI("distribute task info kernel_type: %d, flag: %u", static_cast<int32_t>(kernel_type_), dump_flag_);
    GE_RETURN_IF_ERROR(AssembleKernelNamesAndLaunch());
  } else {
    GE_ASSERT_SUCCESS(DistributeTask());
  }
  call_save_dump_ = true;
  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp() != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] Call DistributeWaitTaskForAicpuBlockingOp failed");
      return FAILED;
    }
  }
  GELOGI("KernelTaskInfo Distribute Success, op: %s, taskid: %d, stubfunc: %p, blockdim: %d, "
         "streamid: %u, stream: %p, dump_flag: %u",
         op_desc_->GetName().c_str(), task_id_, stub_func_, block_dim_, stream_id_, stream_, dump_flag_);
  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
    operator_.reset();
    super_kernel_op_desc_.reset();
    sk_sub_operator_.reset();
  }
  return SUCCESS;
}

Status KernelTaskInfo::UpdateRunInfoByTilingResult(const optiling::utils::OpRunInfo *const run_info) {
  block_dim_ = run_info->GetBlockDim();
  tiling_key_ = run_info->GetTilingKey();
  clear_atomic_ = run_info->GetClearAtomic();
  local_memory_size_ = run_info->GetLocalMemorySize();
  const auto workspaces = run_info->GetAllWorkspaces();
  op_desc_->SetWorkspaceBytes(workspaces);
  GELOGI(
      "Update run info of %s, block dim: %u, tiling key: %" PRIu64 ", clear atomic: %d, workspace size: %zu, "
      "local memory size: %u.",
      op_desc_->GetName().c_str(), block_dim_, tiling_key_, static_cast<int32_t>(clear_atomic_), workspaces.size(),
      local_memory_size_);

  return SUCCESS;
}

void KernelTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  if (has_memory_log_ && (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    davinci_model_->SetAiCpuCustFlag(true);
  }

  if (!IsAllKernel(task_type_)) {
    if (!is_separately_clean_task_) {
      const auto &context_def = task_def.kernel().context();
      davinci_model_->SaveDfxInfo(context_def.op_index(), task_def, *this);
    }
  } else {
    const auto &context_def = task_def.kernel_with_handle().context();
    davinci_model_->SaveDfxInfo(context_def.op_index(), task_def, *this);
  }

  if (has_memory_log_ && (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    davinci_model_->SetAiCpuCustFlag(false);
  }
  ResetArgsEx();
}

Status KernelTaskInfo::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const {
  int32_t device_id = 0;
  GE_CHK_RT_RET(aclrtGetDevice(&device_id));

  int32_t val = 0;
  GE_CHK_RT_RET(rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &val));
  if ((val != RT_AICPU_BLOCKING_OP_NOT_SUPPORT) && (val != RT_AICPU_BLOCKING_OP_SUPPORT)) {
    REPORT_INNER_ERR_MSG("E19999", "Value should be %d or %d but %d", RT_AICPU_BLOCKING_OP_NOT_SUPPORT,
                       RT_AICPU_BLOCKING_OP_SUPPORT, val);
    GELOGE(FAILED, "[Check][Value] Value should be %d or %d but %d", RT_AICPU_BLOCKING_OP_NOT_SUPPORT,
           RT_AICPU_BLOCKING_OP_SUPPORT, val);
    return FAILED;
  }
  is_support = (val == RT_AICPU_BLOCKING_OP_SUPPORT);
  return SUCCESS;
}

Status KernelTaskInfo::UpdateEventIdForAicpuBlockingOp(const hybrid::AicpuExtInfoHandler &ext_handle) const {
  if (is_blocking_aicpu_op_) {
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
    if (davinci_model_->GetEventIdForBlockingAicpuOp(op_desc_, stream_, event_id) != SUCCESS) {
      GELOGE(FAILED, "[Get][EventId] Get event id failed for op:%s(%s)", op_desc_->GetName().c_str(),
             op_desc_->GetType().c_str());
      return FAILED;
    }
    if (ext_handle.UpdateEventId(event_id) != SUCCESS) {
      GELOGE(FAILED, "[Update][EventId] Update event id failed for op:%s(%s)", op_desc_->GetName().c_str(),
             op_desc_->GetType().c_str());
      return FAILED;
    }
    GELOGI("Update event_id=%u success", event_id);
  }
  return SUCCESS;
}

Status KernelTaskInfo::DistributeWaitTaskForAicpuBlockingOp() const {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOp failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("device not support blocking aicpu op process.");
    return SUCCESS;
  }
  GELOGD("Distribute wait task begin");
  aclrtEvent rt_event = nullptr;
  if (davinci_model_->GetEventByStream(stream_, rt_event) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call GetEventByStream failed");
    GELOGE(FAILED, "[Call][GetEventByStream] Call GetEventByStream failed");
    return FAILED;
  }

  uint32_t timeout = 0xffffffff;
  (void)AttrUtils::GetInt(op_desc_, ATTR_NAME_BLOCKING_OP_TIMEOUT, timeout);
  GE_CHK_RT_RET(rtStreamWaitEventWithTimeout(stream_, rt_event, timeout));
  GE_CHK_RT_RET(aclrtResetEvent(rt_event, stream_));

  return SUCCESS;
}

void KernelTaskInfo::GetAtomicOutAddrs(const std::vector<uint64_t> &output_data_addrs,
                                       std::vector<uint64_t> &atomic_output_data_addrs) const {
  std::vector<uint64_t> output_addr_mem_types;
  std::vector<uint64_t> atomic_output_addr_mem_types;
  output_addr_mem_types.reserve(output_data_addrs.size());
  GetAtomicOutAddrs(output_data_addrs, output_addr_mem_types, atomic_output_data_addrs, atomic_output_addr_mem_types);
}

void KernelTaskInfo::GetAtomicOutAddrs(const std::vector<uint64_t> &output_data_addrs,
                                       const std::vector<uint64_t> &output_addr_mem_types,
                                       std::vector<uint64_t> &atomic_output_data_addrs,
                                       std::vector<uint64_t> &atomic_output_addr_mem_types) const {
  std::vector<int64_t> atomic_output_indices;
  (void) AttrUtils::GetListInt(op_desc_, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  for (const int64_t output_index : atomic_output_indices) {
    if (output_index < static_cast<int64_t>(output_data_addrs.size())) {
      atomic_output_data_addrs.push_back(output_data_addrs[static_cast<size_t>(output_index)]);
      atomic_output_addr_mem_types.push_back(output_addr_mem_types[static_cast<size_t>(output_index)]);
    }
  }
}


void KernelTaskInfo::GetAtomicWorkspaceAddrs(const std::vector<uint64_t> &workspace_data_addrs,
                                             std::vector<uint64_t> &atomic_workspace_data_addrs) const {
  std::vector<uint64_t> workspace_addr_types;
  std::vector<uint64_t> atomic_workspace_addr_types;
  workspace_addr_types.reserve(workspace_data_addrs.size());
  GetAtomicWorkspaceAddrs(workspace_data_addrs, workspace_addr_types, atomic_workspace_data_addrs,
                          atomic_workspace_addr_types);
}

void KernelTaskInfo::GetAtomicWorkspaceAddrs(const std::vector<uint64_t> &workspace_data_addrs,
                                             const std::vector<uint64_t> &workspace_addr_types,
                                             std::vector<uint64_t> &atomic_workspace_data_addrs,
                                             std::vector<uint64_t> &atomic_workspace_addr_types) const {
  GeAttrValue::NAMED_ATTRS workspaces;
  if (AttrUtils::GetNamedAttrs(op_desc_, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces)) {
    std::vector<int64_t> value;
    const std::string &op_name = op_desc_->GetName();
    (void) AttrUtils::GetListInt(workspaces, op_name, value);
    for (auto &index : value) {
      if (index < static_cast<int64_t>(workspace_data_addrs.size())) {
        atomic_workspace_data_addrs.push_back(workspace_data_addrs[static_cast<size_t>(index)]);
        atomic_workspace_addr_types.push_back(workspace_addr_types[static_cast<size_t>(index)]);
      }
    }
  }
}

Status KernelTaskInfo::SetIoAddrsForCustomized() {
  if (kernel_type_ != ccKernelType::CUSTOMIZED) {
    return SUCCESS;
  }
  std::vector<uint64_t> mem_types;
  std::vector<uint64_t> tensor_device_addrs;
  const size_t kernel_def_args_size_align =
      MemSizeAlign(static_cast<size_t>(customized_args_info_.kernel_def_args_size), kAddressLen);
  const size_t args_num = kernel_def_args_size_align / kAddressLen;
  (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), args_num, 0);
  (void)mem_types.insert(mem_types.cend(), args_num, static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
  GELOGD("customized has kernel_def_args_size:%u, after align:%u, args num:%zu",
         customized_args_info_.kernel_def_args_size, kernel_def_args_size_align, args_num);

  (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
  (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), output_data_addrs_.cbegin(), output_data_addrs_.cend());
  (void)mem_types.insert(mem_types.cend(), input_mem_types_.cbegin(), input_mem_types_.cend());
  (void)mem_types.insert(mem_types.cend(), output_mem_types_.cbegin(), output_mem_types_.cend());

  io_addrs_.resize(tensor_device_addrs.size());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), mem_types.cbegin(), mem_types.cend());
  size_t args_size = 0UL;
  GE_ASSERT_TRUE(!ge::MulOverflow(io_addrs_.size(), kAddressLen, args_size));
  GE_ASSERT_SUCCESS(InitArgsAddr(tensor_device_addrs, PtrToPtr<uint64_t, uint8_t>(io_addrs_.data()),
                                 io_addr_mem_types_, args_size));
  return SUCCESS;
}

Status KernelTaskInfo::SetIoAddrs() {
  if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    return SetIoAddrsForCustomized();
  }

  std::vector<uint64_t> mem_types;
  std::vector<uint64_t> tensor_device_addrs;
  if (!is_separately_clean_task_) {
    (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
    (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), output_data_addrs_.cbegin(),
                                     output_data_addrs_.cend());
    (void)mem_types.insert(mem_types.cend(), input_mem_types_.cbegin(), input_mem_types_.cend());
    (void)mem_types.insert(mem_types.cend(), output_mem_types_.cbegin(), output_mem_types_.cend());
  } else {
    GetAtomicOutAddrs(output_data_addrs_, output_mem_types_, tensor_device_addrs, mem_types);
  }

  if (ModelUtils::IsAICoreKernel(kernel_type_)) {
    if (!is_separately_clean_task_) {
      (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
      (void)mem_types.insert(mem_types.cend(), workspace_mem_types_.cbegin(), workspace_mem_types_.cend());
    } else {
      GetAtomicWorkspaceAddrs(workspace_addrs_, workspace_mem_types_, tensor_device_addrs, mem_types);
    }
  }
  SaveL0DumpList(tensor_device_addrs.size());

  if (has_tiling_) {
    tensor_device_addrs.emplace_back(PtrToValue(tiling_data_addr_));
    mem_types.push_back(static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
    GELOGD("Node: %s needs to reserve a tiling data address.", op_desc_->GetName().c_str());
  }

  // Refresh the address for overflow detection
  if (KernelTaskInfo::HasOverflowAddr(op_desc_)) {
    tensor_device_addrs.push_back(PtrToValue(davinci_model_->GetOverflowAddr()));
    mem_types.push_back(static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
  }

  size_t io_addrs_element_num = tensor_device_addrs.size();
  if (is_addrs_folded_) {
    io_addrs_element_num += 1UL;
  }
  io_addrs_.resize(io_addrs_element_num);
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), mem_types.cbegin(), mem_types.cend());
  size_t args_size = 0UL;
  GE_ASSERT_TRUE(!ge::MulOverflow(io_addrs_.size(), kAddressLen, args_size));
  GE_ASSERT_SUCCESS(InitArgsAddr(tensor_device_addrs, PtrToPtr<uint64_t, uint8_t>(io_addrs_.data()),
                                 io_addr_mem_types_, args_size));
  return SUCCESS;
}

void KernelTaskInfo::AppendIoAddr(const uint64_t addr, const uint64_t addr_type) {
  io_addrs_.push_back(addr);
  io_addr_mem_types_.push_back(addr_type);
}

void KernelTaskInfo::SaveL0DumpList(const size_t io_addr_size) {
  if (is_addrs_folded_) {
    uint64_t level_num = static_cast<uint64_t>(io_addr_size) & kBitFlag8;
    level_num |= kLevel2BitFlagCustom;
    l0_dump_list_.push_back(level_num);
  }
  if (!is_separately_clean_task_) {
    const size_t level1_num = input_data_addrs_.size() + output_data_addrs_.size() + workspace_addrs_.size();
    for (uint64_t i = 0UL; i < level1_num; ++i) {
      l0_dump_list_.push_back(i);
    }
  } else {
    size_t relevant_idx = input_data_addrs_.size();
    std::vector<int64_t> atomic_output_indices;
    (void)AttrUtils::GetListInt(op_desc_, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
    for (const int64_t idx : atomic_output_indices) {
      if (static_cast<size_t>(idx) < output_data_addrs_.size()) {
        l0_dump_list_.push_back(relevant_idx + static_cast<uint64_t>(idx));
      }
    }
    relevant_idx += output_data_addrs_.size();
    GeAttrValue::NAMED_ATTRS workspaces;
    (void)AttrUtils::GetNamedAttrs(op_desc_, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces);
    std::vector<int64_t> value;
    const std::string &op_name = op_desc_->GetName();
    (void)AttrUtils::GetListInt(workspaces, op_name, value);
    for (const int64_t idx : value) {
      if (static_cast<size_t>(idx) < workspace_addrs_.size()) {
        l0_dump_list_.push_back(relevant_idx + static_cast<uint64_t>(idx));
      }
    }
  }
}

Status KernelTaskInfo::ParseArgsFormat(uint32_t op_index, DavinciModel *const davinci_model) {
  (void)OpDescUtils::GetIrInputInstanceDescRange(op_desc_, args_format_holder_.ir_input_2_range);
  (void)OpDescUtils::GetIrOutputDescRange(op_desc_, args_format_holder_.ir_output_2_range);
  auto &arg_descs = args_format_holder_.arg_descs;
  auto input_descs = op_desc_->GetAllInputsDescPtr();
  for (const auto &arg_format : arg_descs) {
    if (arg_format.addr_type == AddrType::INPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_holder_.ir_input_2_range.size());
      const auto &ir_range = args_format_holder_.ir_input_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        const size_t instance_idx = static_cast<size_t>(ir_range.first + idx);
        GE_ASSERT_TRUE(instance_idx < input_descs.size(), "Instance index [%zu] is out of range, max_size:[%zu].",
                       instance_idx, input_descs.size());
        AppendShapeDesc(*input_descs.at(instance_idx), shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      args_format_holder_.level1_addr_cnt += ir_range.second + shape_info.size();
      args_format_holder_.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::OUTPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_holder_.ir_output_2_range.size());
      const auto &ir_range = args_format_holder_.ir_output_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      args_format_holder_.level1_addr_cnt += ir_range.second;
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        auto output_desc = op_desc_->MutableOutputDesc(static_cast<uint32_t>(ir_range.first + idx));
        GE_ASSERT_NOTNULL(output_desc);
        AppendShapeDesc(*output_desc, shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      args_format_holder_.level1_addr_cnt += ir_range.second + shape_info.size();
      args_format_holder_.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::TILING_CONTEXT &&
               arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_CONTEXT)) {
      const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
          static_cast<gert::OppImplVersionTag>(op_desc_->GetOppImplVersion()));
      const auto op_impl = space_registry->GetOpImpl(op_desc_->GetType().c_str());
      GE_ASSERT_NOTNULL(op_impl, "Failed to get op registry func for node %s.", op_desc_->GetNamePtr());
      for (size_t i = 0UL; i < op_desc_->GetInputsSize(); ++i) {
        size_t ir_index = 0UL;
        GE_ASSERT_SUCCESS(ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc_, i, ir_index));
        if (op_impl->IsTilingInputDataDependency(ir_index)) {
          GELOGI("Node [%s]'s [%zu]th input has tiling dependency.", op_desc_->GetNamePtr(), i);
          args_format_holder_.tiling_depends_input_idx.push_back(i);
        }
      }
    } else if ((arg_format.addr_type == AddrType::TILING_CONTEXT) &&
               (arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_DATA))) {
      // ifa 算子，保留其args desc，aicpu 在拼装tilling的时候使用
      davinci_model->SetTilingSinkTaskArgDescs(op_index, arg_descs);
    } else {
      // misra
    }
  }
  return SUCCESS;
}

size_t KernelTaskInfo::GetArgsSizeByFormat() const {
  const auto &arg_descs = args_format_holder_.arg_descs;
  size_t tmp_size = 0U;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(op_desc_, arg_desc, tmp_size);
  }
  return tmp_size;
}

Status KernelTaskInfo::AssembleShapeInfoAddrs(const std::vector<ArgDesc> &dynamic_args_desc,
                                              const std::vector<size_t> &level2_addr_idx) {
  std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_holder_.ir_input_2_range;
  std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_holder_.ir_output_2_range;
  // append additional level1 addr
  GE_ASSERT(dynamic_args_desc.size() == args_format_holder_.shape_infos.size());
  for (size_t i = 0UL; i < dynamic_args_desc.size(); ++i) {
    auto &shape_info = args_format_holder_.shape_infos[i];
    const size_t ptr_offset_idx = io_addrs_.size();
    GE_ASSERT(level2_addr_idx[i] < io_addrs_.size());
    // addr to ptr offset
    io_addrs_[level2_addr_idx[i]] = PtrToValue(args_) + static_cast<uint64_t>(ptr_offset_idx * sizeof(uint64_t));
    GELOGD("Set ptr_offset idx:[%zu], addr:[%" PRIx64 "] io index:[%zu]",
      ptr_offset_idx, io_addrs_[level2_addr_idx[i]], level2_addr_idx[i]);
    // copy shape_infos
    (void)io_addrs_.insert(io_addrs_.cend(), shape_info.cbegin(), shape_info.cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), shape_info.size(), kAbsoluteMemType);

    if (dynamic_args_desc[i].addr_type == AddrType::INPUT_DESC) {
      const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
      const auto &range_pair = ir_input_2_range[ir_idx];
      size_t begin_idx = range_pair.first;
      for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
        GE_ASSERT(begin_idx < input_data_addrs_.size(),
                  "ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx, begin_idx,
                  input_data_addrs_.size());
        cust_to_relevant_offset_[begin_idx] = io_addrs_.size();
        AppendIoAddr(input_data_addrs_[begin_idx], input_mem_types_[begin_idx]);
        ++begin_idx;
      }
    } else if (dynamic_args_desc[i].addr_type == AddrType::OUTPUT_DESC) {
      const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
      const auto &range_pair = ir_output_2_range[ir_idx];
      size_t begin_idx = range_pair.first;
      for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
        GE_ASSERT(begin_idx < output_data_addrs_.size(),
                  "ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx, begin_idx,
                  output_data_addrs_.size());
        cust_to_relevant_offset_[begin_idx + input_data_addrs_.size()] = io_addrs_.size();
        AppendIoAddr(output_data_addrs_[begin_idx], output_mem_types_[begin_idx]);
        ++begin_idx;
      }
    } else {
      // misra
    }
  }
  return SUCCESS;
}

Status KernelTaskInfo::GetTilingSinkAtomicIndex(bool &is_args_exception_enable, uint64_t &atomic_index) {
  if (Adx::AdumpGetDumpSwitch(Adx::DumpType::ARGS_EXCEPTION) == 0) {
    GELOGI("args exception is not enable");
    return SUCCESS;
  }

  // 获取args desc, ffts+ 走老的l0 exception dump流程
  std::vector<ArgDesc> arg_descs;
  if(davinci_model_->GetAndEraseTilingSinkTaskArgDescs(op_index_, arg_descs) != SUCCESS) {
    GELOGW("op index: %u cannot find sink task args descs.", op_index_);
    return SUCCESS;
  }

  GE_ASSERT_TRUE(!arg_descs.empty());

  std::vector<int64_t> args_size_vec;
  std::vector<optiling::ArgsIndexToIoIndex> args_idx_to_io_idx_vec;
  GE_ASSERT_SUCCESS(
    optiling::TilingDfx::GetArgsSizeWithArgsFormat(op_desc_, arg_descs, args_size_vec, args_idx_to_io_idx_vec));

  std::vector<int64_t> shape_size_vec;
  GE_ASSERT_SUCCESS(UpdateDfxArgsAndShapeSize(op_desc_, args_idx_to_io_idx_vec, args_size_vec, shape_size_vec));

  // 添加workspace
  const std::vector<int64_t> ws_bytes = op_desc_->GetWorkspaceBytes();
  for (size_t idx = 0UL; idx < arg_descs.size(); ++idx) {
    switch (arg_descs[idx].addr_type) {
      case AddrType::WORKSPACE: {
        if (arg_descs[idx].ir_idx < 0) {
           args_size_vec.insert(args_size_vec.cend(), ws_bytes.cbegin(), ws_bytes.cend());
        } else {
          const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
          GE_ASSERT(ir_idx < ws_bytes.size(), "workspace ir_idx:[%zu] is output of range, max_size:[%zu]",
                    ir_idx, ws_bytes.size());
          args_size_vec.push_back(ws_bytes[ir_idx]);
        }
        break;
      }
      default:
        break;
	  }
  }

  if (args_size_vec.size() == 0U ) {
    GELOGW("op index: %u args size is 0.", op_index_);
    return SUCCESS;
  }

  size_t total_size = args_size_vec.size() + shape_size_vec.size();
  uint64_t *host_addr = ge::PtrToPtr<void, uint64_t>(Adx::AdumpGetDFXInfoAddrForStatic(total_size, atomic_index));
  GE_ASSERT_NOTNULL(host_addr, "total size[%zu], automic index[%" PRIu64 "].", total_size, atomic_index);
  FormatArgsException(host_addr, args_size_vec, shape_size_vec, atomic_index);
  is_args_exception_enable = true;

  return SUCCESS;
}

Status KernelTaskInfo::AssembleTilingContextArgs(const ArgDesc &arg_desc,
                                                 std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor) {
  std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
  std::shared_ptr<TilingContextAddr> tiling_context_addr =
      op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
  const TilingContextSubType sub_type = static_cast<TilingContextSubType>(arg_desc.ir_idx);
  switch (sub_type) {
    case TilingContextSubType::TILING_CONTEXT:
      if (tiling_context_addr == nullptr) {
        ConstNodePtr const_node = NodeUtilsEx::GetNodeFromOperator(*operator_);
        GE_ASSERT_NOTNULL(const_node);
        auto node = std::const_pointer_cast<Node>(const_node);
        GE_ASSERT_NOTNULL(node);

        // init platform info on device
        void *platform_infos_addr{nullptr};
        if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
          GE_ASSERT_SUCCESS(davinci_model_->LoadCustPlatformInfos(platform_infos_addr, node));
        } else {
          GE_ASSERT_SUCCESS(davinci_model_->LaunchPlatformInfos(platform_infos_addr, node));
        }
        GE_ASSERT_NOTNULL(platform_infos_addr, "Please check platform_infos_addr.");
        GELOGD("platform_infos_addr = %" PRIu64, reinterpret_cast<uint64_t>(platform_infos_addr));

        // 添加atomic index 信息
        bool is_args_exception_enable = false;
        uint64_t atomic_index = 0UL;
        GE_ASSERT_SUCCESS(GetTilingSinkAtomicIndex(is_args_exception_enable, atomic_index));

        GE_ASSERT_SUCCESS(ArgsFormatUtils::SinkTilingContext(node, *davinci_model_, index_to_tensor,
                          platform_infos_addr, is_args_exception_enable, atomic_index));
        tiling_context_addr = op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
        GE_ASSERT_NOTNULL(tiling_context_addr, "Failed to sink tiling context.");
      }
      AppendIoAddr(tiling_context_addr->tiling_context_addr, kAbsoluteMemType);
      break;
    case TilingContextSubType::TILING_DATA:
      GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
      AppendIoAddr(tiling_context_addr->tiling_data_addr, kAbsoluteMemType);
      break;
    case TilingContextSubType::TILING_KEY:
      GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
      AppendIoAddr(tiling_context_addr->tiling_key_addr, kAbsoluteMemType);
      break;
    case TilingContextSubType::BLOCK_DIM:
      GE_ASSERT_NOTNULL(tiling_context_addr, "Please check whether tiling task has been launched.");
      AppendIoAddr(tiling_context_addr->block_dim_addr, kAbsoluteMemType);
      break;
    default:
      GELOGE(FAILED, "ir index [%d] is invalid.", arg_desc.ir_idx);
      break;
  }
  return SUCCESS;
}

Status KernelTaskInfo::AssembleTilingSinkTensors(std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor) {
  if (args_format_holder_.tiling_depends_input_idx.empty()) {
    return SUCCESS;
  }

  size_t rt_tensor_offset{0UL};
  size_t rt_tensor_size{0UL};
  GetAddrAlignedGertTensorSize(rt_tensor_offset, rt_tensor_size);
  GELOGI("IoAddr Offset:[%zu] double aligned tensor size:[%zu].", rt_tensor_offset, rt_tensor_size);
  args_format_holder_.sink_tensor_size = rt_tensor_size * args_format_holder_.tiling_depends_input_idx.size();
  const size_t addr_num = args_format_holder_.sink_tensor_size / sizeof(uint64_t);
  io_addrs_.resize(addr_num);
  io_addr_mem_types_.resize(addr_num, kAbsoluteMemType);
  size_t tensor_cnt = 0UL;
  const auto args = (task_type_ != ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL)
                        ? PtrToValue(args_)
                        : (PtrToValue(args_) + sizeof(aicpu::AicpuParamHead));
  GELOGI("tiling sink task[%u] args[0x%" PRIx64 "] for [%s]",
    static_cast<uint32_t>(task_type_), args, op_desc_->GetNamePtr());
  for (auto tiling_idx : args_format_holder_.tiling_depends_input_idx) {
    index_to_tensor[tiling_idx].device_addr = args + rt_tensor_size * tensor_cnt + rt_tensor_offset;
    gert::Tensor *host_tensor =
        reinterpret_cast<gert::Tensor *>(PtrToValue(io_addrs_.data()) + rt_tensor_size * tensor_cnt + rt_tensor_offset);
    GE_ASSERT_NOTNULL(host_tensor);
    GE_ASSERT(tiling_idx < input_data_addrs_.size(), "Input index [%zu] is invalid, inputs size:[%zu]", tiling_idx,
              input_data_addrs_.size());
    host_tensor->MutableTensorData().SetAddr(ValueToPtr(input_data_addrs_[tiling_idx]), nullptr);
    const size_t addr_offset =
        static_cast<size_t>(PtrToValue(&host_tensor->MutableTensorData()) - PtrToValue(io_addrs_.data()));
    const size_t addr_idx = addr_offset / sizeof(uintptr_t);
    GE_ASSERT(addr_idx < io_addr_mem_types_.size(), "Tensor addr index [%zu] is invalid, io mem type size:[%zu].",
              addr_idx, io_addr_mem_types_.size());
    io_addr_mem_types_[addr_idx] = input_mem_types_[tiling_idx];
    GELOGI("Set tensor addr index [%zu] memory type with [%" PRIu64 "] by input idx:[%zu]", addr_idx,
           input_mem_types_[tiling_idx], tiling_idx);

    index_to_tensor[tiling_idx].host_addr = host_tensor;
    ++tensor_cnt;
  }

  return SUCCESS;
}

Status KernelTaskInfo::AppendWorkspaceAddr(int32_t ir_idx) {
  if (ir_idx < 0) {
    (void)io_addrs_.insert(io_addrs_.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), workspace_mem_types_.cbegin(),
                                    workspace_mem_types_.cend());
  } else {
    const size_t idx = static_cast<size_t>(ir_idx);
    GE_ASSERT(idx < workspace_addrs_.size(), "workspace index[%zu] is output of workspace addrs range[%zu]",
              idx, workspace_addrs_.size());
    AppendIoAddr(workspace_addrs_[idx], workspace_mem_types_[idx]);
    GELOGI("op[%s], workspace_addrs_[%zu] = 0x%" PRIx64 ", workspace_mem_types_[%zu] = %" PRIu64,
      op_desc_->GetName().c_str(), idx, workspace_addrs_[idx], idx, workspace_mem_types_[idx]);
    if (task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL && kernel_type_ == ccKernelType::CUST_AI_CPU) {
      const std::vector<int64_t> v_workspace_bytes = op_desc_->GetWorkspaceBytes();
      GE_ASSERT(idx < v_workspace_bytes.size(), "workspace index[%zu] is output of workspace bytes range[%zu]",
                idx, v_workspace_bytes.size());
      AppendIoAddr(v_workspace_bytes[idx], kAbsoluteMemType);
      GELOGI("preprocess custom op[%s], v_workspace_bytes[%zu] = %" PRId64, op_desc_->GetName().c_str(), idx,
             v_workspace_bytes[idx]);
    }
  }
  return SUCCESS;
}

Status KernelTaskInfo::AppendInputOutputAddr(size_t ir_idx, bool is_input) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_2_range =
      is_input ? args_format_holder_.ir_input_2_range : args_format_holder_.ir_output_2_range;
  const auto iter = ir_2_range.find(ir_idx);
  GE_ASSERT(iter != ir_2_range.end(), "Ir idx [%zu] is not found, input flag %u.", ir_idx, is_input);
  const auto &range_pair = iter->second;
  if (is_input && range_pair.second == 0UL) {
    // optional input placeholder
    AppendIoAddr(0UL, kAbsoluteMemType);
    return SUCCESS;
  }
  size_t begin_idx = range_pair.first;
  std::vector<uint64_t> &addrs = is_input ? input_data_addrs_ : output_data_addrs_;
  std::vector<uint64_t> &types = is_input ? input_mem_types_ : output_mem_types_;
  const size_t cust_offset = is_input ? 0U : input_data_addrs_.size();
  for (size_t i = 0UL; i < range_pair.second; ++i, ++begin_idx) {
    GE_ASSERT(begin_idx < addrs.size(), "ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].",
              ir_idx, begin_idx, addrs.size());
    cust_to_relevant_offset_[begin_idx + cust_offset] = io_addrs_.size();
    AppendIoAddr(addrs[begin_idx], types[begin_idx]);
  }
  return SUCCESS;
}

Status KernelTaskInfo::AppendInputOutputAddrByInstanceIndex(size_t ins_idx, bool is_input) {
  if (is_input) {
    GE_ASSERT_TRUE(ins_idx < input_data_addrs_.size(),
                   "Instance idx [%zu] is invalid, input_size:[%zu]",
                   ins_idx, input_data_addrs_.size());
    cust_to_relevant_offset_[ins_idx] = io_addrs_.size();
    AppendIoAddr(input_data_addrs_[ins_idx], input_mem_types_[ins_idx]);
  } else {
    GE_ASSERT_TRUE(ins_idx < output_data_addrs_.size(),
                   "Instance idx [%zu] is invalid, output_size:[%zu]",
                   ins_idx, output_data_addrs_.size());
    cust_to_relevant_offset_[input_data_addrs_.size() + ins_idx] = io_addrs_.size();
    AppendIoAddr(output_data_addrs_[ins_idx], output_mem_types_[ins_idx]);
  }
  return SUCCESS;
}

Status KernelTaskInfo::AssembleIoByArgsFormat() {
  const auto &arg_descs = args_format_holder_.arg_descs;
  io_addrs_.reserve(arg_descs.size());
  io_addr_mem_types_.reserve(arg_descs.size());
  std::map<size_t, gert::AddrRefreshedTensor> idx_to_sink_tensor_map;
  GE_ASSERT_SUCCESS(AssembleTilingSinkTensors(idx_to_sink_tensor_map));
  std::vector<ArgDesc> dynamic_args_desc;
  std::vector<size_t> level_addr_idx;
  std::vector<void *> context_addrs;
  for (const auto &arg_format : arg_descs) {
    switch (arg_format.addr_type) {
      case AddrType::INPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(static_cast<size_t>(arg_format.ir_idx), true));
        break;
      }
      case AddrType::OUTPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(static_cast<size_t>(arg_format.ir_idx), false));
        break;
      }
      case AddrType::INPUT_DESC:
      case AddrType::OUTPUT_DESC: {
        level_addr_idx.push_back(io_addrs_.size());
        dynamic_args_desc.push_back(arg_format);
        AppendIoAddr(0UL, kAbsoluteMemType);
        break;
      }
      case AddrType::INPUT: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddr(static_cast<size_t>(arg_format.ir_idx), true));
        break;
      }
      case AddrType::OUTPUT: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddr(static_cast<size_t>(arg_format.ir_idx), false));
        break;
      }
      case AddrType::WORKSPACE: {
        GE_ASSERT_SUCCESS(AppendWorkspaceAddr(arg_format.ir_idx));
        break;
      }
      case AddrType::HIDDEN_INPUT: {
        if (*reinterpret_cast<const HiddenInputsType *>(arg_format.reserved) == HiddenInputsType::HCOM) {
          if (context_addrs.empty()) {
            GE_ASSERT_SUCCESS(ArgsFormatUtils::GetHcomHiddenInputs(op_desc_, *davinci_model_, context_addrs));
          }
          const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
          GE_ASSERT_TRUE(ir_idx < context_addrs.size());
          AppendIoAddr(reinterpret_cast<uint64_t>(context_addrs[ir_idx]), kAbsoluteMemType);
        }
        if (*reinterpret_cast<const HiddenInputsType *>(arg_format.reserved) == HiddenInputsType::TILEFWK) {
          if (context_addrs.empty()) {
            GE_ASSERT_SUCCESS(ArgsFormatUtils::GetTileFwkHiddenInputs(op_desc_, *davinci_model_, context_addrs));
          }
          const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
          GE_ASSERT_TRUE(ir_idx < context_addrs.size());
          AppendIoAddr(reinterpret_cast<uint64_t>(context_addrs[ir_idx]), kAbsoluteMemType);
        }
        break;
      }
      case AddrType::TILING: {
        AppendIoAddr(PtrToValue(tiling_data_addr_), kAbsoluteMemType);
        GELOGD("Node: %s needs to reserve a tiling data addr [%p].", op_desc_->GetName().c_str(), tiling_data_addr_);
        break;
      }
      case AddrType::OVERFLOW_ADDR: {
        if (KernelTaskInfo::HasOverflowAddr(op_desc_)) {
          AppendIoAddr(PtrToValue(davinci_model_->GetOverflowAddr()),
                       static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
        }
        break;
      }
      case AddrType::PLACEHOLDER: {
        AppendIoAddr(0UL, kAbsoluteMemType);
        break;
      }
      case AddrType::OP_TYPE: {
        std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
        std::shared_ptr<TilingContextAddr> tiling_context_addr =
            op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
        GE_ASSERT_NOTNULL(tiling_context_addr);
        AppendIoAddr(tiling_context_addr->op_type_addr, kAbsoluteMemType);
        break;
      }
      case AddrType::TILING_CONTEXT: {
        GE_ASSERT_SUCCESS(AssembleTilingContextArgs(arg_format, idx_to_sink_tensor_map));
        break;
      }
      case AddrType::CUSTOM_VALUE: {
        AppendIoAddr(*reinterpret_cast<const uint64_t *>(arg_format.reserved), kAbsoluteMemType);
        break;
      }
      case AddrType::FFTS_ADDR: {
        uint64_t mode_addr = 0U;
        uint32_t len = 0U;
        GE_CHK_RT_RET(rtGetC2cCtrlAddr(&mode_addr, &len));
        AppendIoAddr(mode_addr, kAbsoluteMemType);
        break;
      }
      case (AddrType::EVENT_ADDR): {
        const uint32_t mem_event_id = static_cast<uint32_t>(arg_format.ir_idx);
        AppendIoAddr(PtrToValue(davinci_model_->GetMemEventIdAddr(mem_event_id)), kAbsoluteMemType);
        break;
      }
      default:
        break;
    }
  }
  GE_ASSERT_SUCCESS(AssembleShapeInfoAddrs(dynamic_args_desc, level_addr_idx));
  // l0 exception dump
  GE_ASSERT_SUCCESS(SaveL0DumpListWithArgsFormat());

  return SUCCESS;
}

Status KernelTaskInfo::SaveL0DumpListWithArgsFormat() {
  if (!ModelUtils::IsAICoreKernel(kernel_type_)) {
    return SUCCESS;
  }
  const auto &arg_descs = args_format_holder_.arg_descs;
  if (arg_descs.empty()) {
    return SUCCESS;
  }

  const std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_holder_.ir_input_2_range;
  const std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_holder_.ir_output_2_range;
  for (size_t idx = 0UL; idx < arg_descs.size(); ++idx) {
    switch (arg_descs[idx].addr_type) {
      case AddrType::INPUT_DESC: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end(), "input ir idx [%zu] is not found", ir_idx);
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
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end(), "input ir idx [%zu] is not found", ir_idx);
        // level2_addr
        uint64_t level_num = iter->second.second & kBitFlag8;
        level_num |= kLevel2BitFlagWithShape;
        l0_dump_list_.push_back(level_num);
        // level1
        for (size_t i = 0UL; i < iter->second.second; ++i) {
          l0_dump_list_.push_back(input_data_addrs_.size() + iter->second.first + i);
        }
        break;
      }
      case AddrType::INPUT_INSTANCE: {
        l0_dump_list_.push_back(static_cast<size_t>(arg_descs[idx].ir_idx));
        break;
      }
      case AddrType::OUTPUT_INSTANCE: {
        l0_dump_list_.push_back(static_cast<size_t>(arg_descs[idx].ir_idx) + input_data_addrs_.size());
        break;
      }
      case AddrType::INPUT: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end(), "input ir idx [%zu] is not found", ir_idx);
        const auto &range_pair = iter->second;
        if (range_pair.second == 0UL) {
          // optional input placeholder
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
          break;
        }
        size_t begin_idx = range_pair.first;
        for (size_t i = 0UL; i < range_pair.second; ++i) {
          GE_ASSERT(begin_idx < input_data_addrs_.size(),
                    "ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx, begin_idx,
                    input_data_addrs_.size());
          l0_dump_list_.push_back(begin_idx);
          ++begin_idx;
        }
        break;
      }
      case AddrType::OUTPUT: {
        const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
        const auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end(), "output ir idx [%zu] is not found", ir_idx);
        const auto &range_pair = iter->second;
        size_t begin_idx = range_pair.first;
        for (size_t i = 0UL; i < range_pair.second; ++i) {
          GE_ASSERT(begin_idx < output_data_addrs_.size(),
                    "ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx, begin_idx,
                    output_data_addrs_.size());
          l0_dump_list_.push_back(begin_idx + input_data_addrs_.size());
          ++begin_idx;
        }
        break;
      }
      case AddrType::WORKSPACE: {
        const size_t input_output_size = input_data_addrs_.size() + output_data_addrs_.size();
        if (arg_descs[idx].ir_idx < 0) {
          for (size_t i = 0UL; i < workspace_addrs_.size(); ++i) {
            l0_dump_list_.push_back(input_output_size + i);
          }
        } else {
          const size_t ir_idx = static_cast<size_t>(arg_descs[idx].ir_idx);
          GE_ASSERT(ir_idx < workspace_addrs_.size(), "workspace ir_idx:[%zu] is output of range, max_size:[%zu]",
                    ir_idx, workspace_addrs_.size());
          l0_dump_list_.push_back(input_output_size + ir_idx);
        }
        break;
      }

      case AddrType::TILING_CONTEXT: {
        if (arg_descs[idx].ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_DATA)) {
          uint64_t tiling_data_size = kMaxTilingDataSize;
          int64_t max_size = -1;
          if (ge::AttrUtils::GetInt(op_desc_, kMaxTilingSize, max_size) && max_size > 0) {
            tiling_data_size = static_cast<uint64_t>(max_size);
          }
          tiling_data_size &= kBitFlag8;
          tiling_data_size |= kLevel2BitFlagTilingData;
          l0_dump_list_.push_back(tiling_data_size);
        } else {
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        }
        break;
      }
      default:
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        break;
    }
  }

  return SUCCESS;
}

Status KernelTaskInfo::GetNoncontinuousArgsRefreshInfo(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, io_addr_offset_ + args_offset_from_pls_, args_placement_);
  return SUCCESS;
}

Status KernelTaskInfo::GetcontinuousArgsRefreshInfo(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, args_placement_);
  return SUCCESS;
}

Status KernelTaskInfo::UpdateNoncontinuousArgs(const size_t offset, const std::vector<uint64_t> &active_mem_base_addr,
                                               void *const host_args,
                                               const size_t host_args_len) {
  GELOGD("kernel_type:%d, task_id:%u, offset:%zu, io_addrs_size:%zu, args_size:%u, host_args_len:%zu.",
      static_cast<int32_t>(kernel_type_), task_id_, offset, io_addrs_.size(), args_size_, host_args_len);
  GE_ASSERT_TRUE(static_cast<size_t>(args_size_) > offset);
  const size_t addr_size = static_cast<size_t>(args_size_) - offset;
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr, &args_addr_[offset], addr_size));
  const errno_t sec_ret = memcpy_s(host_args, host_args_len, args_addr_.data(), static_cast<size_t>(args_size_));
  GE_ASSERT_EOK(sec_ret, "[Call][Memcpy] failed, size:%zu, ret:%d", addr_size, sec_ret);

  return SUCCESS;
}

Status KernelTaskInfo::UpdateContinuousArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                            const size_t host_args_len) {
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_len));
  std::vector<uint64_t> io_addrs_updated;
  io_addrs_updated.reserve(io_addrs_.size());
  uint64_t *host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
  GE_ASSERT_TRUE(io_addrs_.size() * sizeof(uint64_t) <= host_args_len);
  for (size_t index = 0; index < io_addrs_.size(); ++index) {
    io_addrs_updated.push_back(*host_args_tmp++);
  }
  davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, io_addrs_updated);
  return SUCCESS;
}

Status KernelTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  GELOGI("KernelTaskInfo::GetTaskArgsRefreshInfos in.");
  GE_CHECK_NOTNULL(davinci_model_);
  if (ModelUtils::IsAICoreKernel(kernel_type_)) {
    return GetcontinuousArgsRefreshInfo(infos);
  }

  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU) ||
      (kernel_type_ == ccKernelType::AI_CPU_KFC)) {
    return GetNoncontinuousArgsRefreshInfo(infos);
  }

  if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    return GetcontinuousArgsRefreshInfo(infos);
  }
  return SUCCESS;
}

Status KernelTaskInfo::UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) {
  GE_CHECK_NOTNULL(davinci_model_);
  if (ModelUtils::IsAICoreKernel(kernel_type_) || kernel_type_ == ccKernelType::CUSTOMIZED) {
    std::vector<uint64_t> io_addrs_updated;
    io_addrs_updated.reserve(io_addrs_.size());
    uint64_t *host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
    GE_ASSERT_TRUE(io_addrs_.size() * sizeof(uint64_t) <= host_args_max_len);
    for (size_t index = 0; index < io_addrs_.size(); ++index) {
      io_addrs_updated.push_back(*host_args_tmp++);
    }
    davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, io_addrs_updated);
  }
  return SUCCESS;
}

Status KernelTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                                      void *const host_args,
                                      const size_t host_args_max_len) {
  GELOGI("KernelTaskInfo::UpdateArgs in.");
  GE_CHECK_NOTNULL(davinci_model_);
  if (ModelUtils::IsAICoreKernel(kernel_type_)) {
    return UpdateContinuousArgs(active_mem_base_addr, host_args, host_args_max_len);
  }

  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU) ||
      (kernel_type_ == ccKernelType::AI_CPU_KFC)) {
    return UpdateNoncontinuousArgs(io_addr_offset_, active_mem_base_addr, host_args, host_args_max_len);
  }

  if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    return UpdateContinuousArgs(active_mem_base_addr, host_args, host_args_max_len);
  }
  return SUCCESS;
}

Status KernelTaskInfo::Release() {
  aclrtContext ctx = nullptr;
  GE_CHK_RT(aclrtGetCurrentContext(&ctx));

  args_ = nullptr;
  kernel_name_arg_ = nullptr;
  launch_addr_ = nullptr;
  custom_info_.input_descs = nullptr;
  custom_info_.output_descs = nullptr;
  custom_info_.attr_handle = nullptr;
  aicpu_ext_info_addr_ = nullptr;

  ctx_.argsOffset.clear();

  return SUCCESS;
}

Status KernelTaskInfo::CopyTilingDataIfNeeded() {
  std::shared_ptr<optiling::utils::OpRunInfo> default_tiling = nullptr;
  std::shared_ptr<optiling::utils::OpRunInfo> run_info = nullptr;
  if (is_separately_clean_task_) {
    run_info = op_desc_->TryGetExtAttr(ge::ATTR_NAME_ATOMIC_OP_RUN_INFO, default_tiling);
  } else {
    run_info = op_desc_->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, default_tiling);
  }

  if (is_soft_sync_op_ && IsAllKernel(task_type_)) {
    run_info = MakeShared<optiling::utils::OpRunInfo>(0, false, 0);
    GE_CHECK_NOTNULL(run_info);
    GE_CHK_STATUS_RET(optiling::SoftSyncOpRtParseAndTiling(
                          *operator_, davinci_model_->MutablePlatformInfo(), *run_info,
                          davinci_model_->GetSpaceRegistry(
                              static_cast<gert::OppImplVersionTag>(op_desc_->GetOppImplVersion()))),
                      "Recall tiling for soft sync op: %s failed.", op_desc_->GetNamePtr());
    GE_CHK_STATUS_RET(UpdateRunInfoByTilingResult(run_info.get()), "Update run info by tiling result failed.");
    // 软同步操作后会刷新runInfo，需要重新memcheck
    GELOGD("OpName: %s update schedule mode from tiling info: %u", op_desc_->GetNamePtr(),
           static_cast<uint32_t>(cfg_.schemMode));
  }

  if (run_info != nullptr) {
    local_memory_size_ = run_info->GetLocalMemorySize();
    cfg_.schemMode = static_cast<uint8_t>(run_info->GetScheduleMode() & k2BitsMask);
    if (run_info->GetAllTilingData().str().empty()) {
      GELOGD("Tiling data of %s is empty.", op_desc_->GetNamePtr());
      return SUCCESS;
    }
    has_tiling_ = true;
    std::string tiling_data = run_info->GetAllTilingData().str();
    if (!is_separately_clean_task_) {
      std::string dfx_info;
      GE_CHK_STATUS_RET(ConstructDfxInfo(op_desc_, *run_info, args_format_holder_.arg_descs, dfx_info),
          "Append memcheck data for node: %s failed.", op_desc_->GetNamePtr());
      tiling_data += dfx_info;
    }
    tiling_data_size_ = tiling_data.size();
    tiling_data_addr_ = davinci_model_->MallocDynamicMemory(tiling_data_size_);
    GE_CHECK_NOTNULL(tiling_data_addr_);
    GE_CHK_RT_RET(rtMemcpy(tiling_data_addr_, tiling_data_size_, tiling_data.data(), tiling_data.size(),
                           RT_MEMCPY_HOST_TO_DEVICE));
    GELOGI("Success to update tiling data to io_addr of %s, addr: %p, size: %zu.", op_desc_->GetNamePtr(),
           tiling_data_addr_, tiling_data.size());
  }
  return SUCCESS;
}

Status KernelTaskInfo::FindSkSubNode(const OpDescPtr &sk_op, const int32_t id,  NodePtr &sub_node) const {
  GE_ASSERT_NOTNULL(sk_op);
  ComputeGraphPtr sub_graph = nullptr;
  sub_graph = sk_op->TryGetExtAttr("_sk_sub_graph", sub_graph);
  GE_ASSERT_NOTNULL(sub_graph);
  for (const auto &node : sub_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOpDesc()->GetId() == static_cast<int64_t>(id)) {
      sub_node = node;
      GELOGI("find %d sub node %s from sk node %s", id, node->GetNamePtr(), sk_op->GetNamePtr());
      return SUCCESS;
    }
  }
  GELOGE(FAILED, "can not find %d sub node from sk node %s", id, sk_op->GetNamePtr());
  return FAILED;
}

Status KernelTaskInfo::PreprocessForSkNode() {
  if (super_kernel_op_desc_ == nullptr) {
    return SUCCESS;
  }
  std::vector<ArgDesc> sub_arg_descs;
  auto &arg_descs = args_format_holder_.arg_descs;
  int32_t sub_node_id = -1;
  for (const auto &arg_format : arg_descs) {
    ArgDesc tmp_arg_desc = arg_format;
    if (arg_format.addr_type == AddrType::SUPER_KERNEL_SUB_NODE) {
      const SkArgDesc *sk_args_desc = reinterpret_cast<const SkArgDesc *>(&arg_format);
      tmp_arg_desc.addr_type = sk_args_desc->sub_addr_type;
      tmp_arg_desc.ir_idx = sk_args_desc->sub_idx;
      tmp_arg_desc.folded = sk_args_desc->folded;
      sub_node_id = sk_args_desc->ir_idx;
    }
    sub_arg_descs.emplace_back(tmp_arg_desc);
  }
  GE_ASSERT_TRUE(sub_node_id != -1);
  NodePtr sub_node;
  GE_ASSERT_SUCCESS(FindSkSubNode(super_kernel_op_desc_, sub_node_id, sub_node));
  sk_sub_operator_ = MakeShared<Operator>(OpDescUtils::CreateOperatorFromNode(sub_node));
  op_desc_ = sub_node->GetOpDesc();
  args_format_holder_.arg_descs = sub_arg_descs;
  GELOGI("current sk node %s, inner op_desc replace to %s",
         super_kernel_op_desc_->GetNamePtr(), op_desc_->GetNamePtr());
  return SUCCESS;
}

Status KernelTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                         TaskRunParam &task_run_param) {
  task_type_ = static_cast<ModelTaskType>(task_def.type());
  GE_CHECK_NOTNULL(davinci_model);
  domi::KernelContext context;
  size_t extra_name_size = 0U;
  if (IsAllKernel(task_type_)) {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    args_size_ = static_cast<uint32_t>(kernel_def.args().size());
    context = kernel_def.context();
    kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
  } else {
    const domi::KernelDef &kernel_def = task_def.kernel();
    args_size_ = static_cast<uint32_t>(kernel_def.args().size());
    context = kernel_def.context();
    kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
    if (kernel_type_ == ccKernelType::AI_CPU_KFC) {
      extra_name_size = kernel_def.so_name().size() + 1UL + kernel_def.kernel_name().size() + 1UL;
    }
  }

  uint32_t op_index = context.op_index();
  // Called before Init. Assign necessary vaiables.
  op_desc_ = (op_desc_ != nullptr) ? op_desc_ : davinci_model->GetOpByIndex(context.op_index());
  GE_CHECK_NOTNULL(op_desc_);
  super_kernel_op_desc_ = (op_desc_->GetType() == "SuperKernel") ? op_desc_ : nullptr;
  if (super_kernel_op_desc_ != nullptr) {
    GE_ASSERT_TRUE(!context.args_format().empty());
  }
  if (!context.args_format().empty()) {
    GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(op_desc_, context.args_format(), args_format_holder_.arg_descs),
                      "Formatted args [%s] parsed failed.", context.args_format().c_str());
    GE_ASSERT_SUCCESS(PreprocessForSkNode());
    GE_ASSERT_SUCCESS(ParseArgsFormat(op_index, davinci_model),
      "ParseArgsFormat failed, op:[%s].", op_desc_->GetNamePtr());
    const size_t format_args_size = GetArgsSizeByFormat() + extra_name_size;
    args_size_ = std::max(args_size_, static_cast<uint32_t>(format_args_size));
    if (task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL && kernel_type_ == ccKernelType::CUST_AI_CPU) {
      // Used for the length of the workspace where logs are dumped by the custom sink operator.
      args_size_ += sizeof(uint64_t);
    }
    GELOGI("OP [%s] has formatted args_format:[%s], args size by format is [%" PRIu64 "], final size is [%u]",
           op_desc_->GetNamePtr(), context.args_format().c_str(), format_args_size, args_size_);
  }

  const size_t extra_args_size = GetExtraArgsSize(*davinci_model, op_desc_, kernel_type_);
  GELOGD("Op:[%s] args size from_task:[%u], extra_size:[%zu]", op_desc_->GetNamePtr(), args_size_, extra_args_size);
  GE_ASSERT_TRUE(!AddOverflow(args_size_, static_cast<uint32_t>(extra_args_size), args_size_));

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  input_data_addrs_ = ModelUtils::GetInputAddrsValue(rts_param, op_desc_, input_mem_types_);
  if (!context.args_format().empty()) {
    // args format场景，GetOutputAddrsValue返回的地址数量和输出anchor数量保持一致，可选输出需要使用空地址占位
    output_data_addrs_ = ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_, true);
  } else {
    output_data_addrs_ = ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_);
  }
  workspace_addrs_ = ModelUtils::GetWorkspaceDataAddrsValue(rts_param, op_desc_, workspace_mem_types_);
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({input_data_addrs_[i], input_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({output_data_addrs_[i], output_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back({workspace_addrs_[i], workspace_mem_types_[i], true, {0}});
  }

  size_t append_size = 0U;
  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    std::unique_ptr<hybrid::AicpuExtInfoHandler> ex_handle = nullptr;
    const auto &kernel_def = task_def.kernel();
    const auto &ext_info = kernel_def.kernel_ext_info();
    GE_ASSERT_SUCCESS(ParseAicpuExtInfoHandler(op_desc_, ext_info, ex_handle));
    if ((ex_handle != nullptr) && (ex_handle->GetDeployTypeFlag() == static_cast<int32_t>(RT_KERNEL_HOST_ONLY))) {
      args_placement_ = ArgsPlacement::kArgsPlacementHostSvm;
    }
    append_size = sizeof(uintptr_t); //多申请8字节，用来做aicpuhead结构体的对齐
  } else if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    customized_args_info_.kernel_def_args_size = args_size_;
    GE_ASSERT_SUCCESS(UpdateArgsSizeWithCustomized(op_desc_));
  }
  task_run_param.args_descs.push_back({static_cast<int64_t>(MemSizeAlign(static_cast<size_t>(args_size_),
      static_cast<uint32_t>(sizeof(uintptr_t))) + append_size), args_placement_});
  const bool is_wsp_addr_folded = IsWspAddrFolded(op_desc_);
  GELOGD(
      "Get args size[%u] of op[%s], is_wsp_addr_folded[%d], is known node[%d], task_type: %d, placement: %d.",
      args_size_, op_desc_->GetName().c_str(), static_cast<int32_t>(is_wsp_addr_folded),
      static_cast<int32_t>(davinci_model->IsFeatureBaseRefreshable()), static_cast<int32_t>(task_type_),
      args_placement_);
  return SUCCESS;
}

Status KernelTaskInfo::InitTVMContext(const domi::KernelContext &context) {
  if ((context.args_offset().size() / sizeof(uint16_t)) < 1U) {
    REPORT_INNER_ERR_MSG("E19999", "args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s), check invalid",
                       context.args_offset().size(), op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    GELOGE(FAILED, "[Check][Param]invalid, args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s)",
           context.args_offset().size(), op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return FAILED;
  }

  uint16_t args_offset = 0U;
  GE_ASSERT_EOK(memcpy_s(&args_offset, sizeof(uint16_t), context.args_offset().data(), sizeof(uint16_t)));
  GE_CHECK_LE(args_offset, args_size_);
  io_addr_offset_ = static_cast<size_t>(args_offset);
  GELOGD("Get args_offset[%u] of op[%s]", static_cast<uint32_t>(args_offset), op_desc_->GetName().c_str());
  return SUCCESS;
}

void KernelTaskInfo::UpdateAtomicCleanArgs(std::vector<uint64_t> &input_data_addrs,
                                           std::vector<uint64_t> &output_data_addrs,
                                           std::vector<uint64_t> &workspace_data_addrs) const {
  input_data_addrs.clear();
  std::vector<uint64_t> output_data_addrs_atomic;
  GetAtomicOutAddrs(output_data_addrs, output_data_addrs_atomic);
  output_data_addrs.clear();
  (void)output_data_addrs.insert(output_data_addrs.cend(), output_data_addrs_atomic.cbegin(),
                                 output_data_addrs_atomic.cend());

  std::vector<uint64_t> atomic_workspace_data_addrs;
  GetAtomicWorkspaceAddrs(workspace_data_addrs, atomic_workspace_data_addrs);
  workspace_data_addrs.clear();
  (void)workspace_data_addrs.insert(workspace_data_addrs.cend(), atomic_workspace_data_addrs.cbegin(),
                                    atomic_workspace_data_addrs.cend());
}

Status KernelTaskInfo::InitArgsAddr(const std::vector<uint64_t> &tensor_device_addrs, uint8_t *io_addr,
                                    std::vector<uint64_t> &io_addr_mem_types, const size_t args_size) {
  if (tensor_device_addrs.empty()) {
    return SUCCESS;
  }
  const size_t args_size_max = static_cast<size_t>(args_size) - io_addr_offset_;
  size_t addr_size_max = kAddressLen * tensor_device_addrs.size();
  if ((args_size <= io_addr_offset_) || (args_size_max < addr_size_max)) {
    REPORT_INNER_ERR_MSG("E19999", "offset:%zu >= argsSize:%zu or content:%zu beyond applied memory:%zu, check invalid",
                       io_addr_offset_, args_size, addr_size_max, args_size_max);
    GELOGE(FAILED, "[Check][Param] offset:%zu >= argsSize:%u or content:%zu beyond applied memory:%zu, check invalid",
           io_addr_offset_, args_size, addr_size_max, args_size_max);
    return FAILED;
  }
  errno_t sec_ret;
  const bool has_overflow_addr = KernelTaskInfo::HasOverflowAddr(op_desc_);
  GELOGI(
      "Init op[%s], type[%s] args addr start, is_addrs_folded[%d], device_addrs size[%zu], has_overflow_addr[%d], "
      "is_separately_clean_task[%d].",
      op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), static_cast<int32_t>(is_addrs_folded_),
      tensor_device_addrs.size(), static_cast<int32_t>(has_overflow_addr),
      static_cast<int32_t>(is_separately_clean_task_));
  if (!is_addrs_folded_) {
    // unfolded mode(clear addr size < 192):
    // ----------------------------------
    // | wsl addr list | over flow addr |
    // ----------------------------------
    sec_ret = memcpy_s(&io_addr[io_addr_offset_], args_size_max, tensor_device_addrs.data(), addr_size_max);
  } else {
    // folded mode(clear addr size >= 192):
    // --------------------------------------------------------------------------------------------------
    // | point to wsl addr list | tiling data addr(optional) | over flow addr(optional) | wsl addr list |
    // |          |_______________________________________________________________________|
    // ---------------------------------------------------------------------------------------------------
    const auto address_to_addrs_table = PtrToPtr<void, uint64_t>(static_cast<void *>(&io_addr[io_addr_offset_]));
    // offset the first address for address_to_addrs_table(device memory address)
    size_t addrs_table_offset = io_addr_offset_ + kAddressLen;
    int32_t inner_offset = 0;
    constexpr uint64_t mem_type = static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix);
    (void)io_addr_mem_types.insert(io_addr_mem_types.cbegin(), mem_type);
    if (has_tiling_) {
      // offset the second address for tiling data
      inner_offset++;
      *(address_to_addrs_table + inner_offset) = PtrToValue(tiling_data_addr_);
      addrs_table_offset += kAddressLen;
      addr_size_max -= kAddressLen;
      (void)io_addr_mem_types.insert(io_addr_mem_types.cbegin(), mem_type);
    }
    if (has_overflow_addr) {
      // offset the third address for overflow address
      inner_offset++;
      *(address_to_addrs_table + inner_offset) = PtrToValue(davinci_model_->GetOverflowAddr());
      addrs_table_offset += kAddressLen;
      addr_size_max -= kAddressLen;
      (void)io_addr_mem_types.insert(io_addr_mem_types.cbegin(), mem_type);
    }
    const auto dev_addr_table = PtrToPtr<uint8_t, void>(
        PtrAdd(PtrToPtr<void, uint8_t>(args_), static_cast<size_t>(args_size_), addrs_table_offset));
    *address_to_addrs_table = PtrToValue(dev_addr_table);
    sec_ret = memcpy_s(&io_addr[addrs_table_offset], args_size - addrs_table_offset,
                       tensor_device_addrs.data(),
                       addr_size_max);
  }
  GE_ASSERT_TRUE(sec_ret == EOK, "[Call][Memcpy] failed, size:%zu, ret:%#x, has_overflow_addr:%d.",
                 args_size_max, sec_ret, static_cast<int32_t>(has_overflow_addr));
  return SUCCESS;
}

Status KernelTaskInfo::InitTVMTask(const domi::KernelDefWithHandle &kernel_def) {
  GELOGD("Do InitTVMTask with handle of %s.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InitTVMContext(kernel_def.context()));
  node_info_ = kernel_def.node_info() + "/";
  cfg_.schemMode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
  GELOGD("OpName: %s set schedule mode from kernel def: %u",
      op_desc_->GetName().c_str(), static_cast<uint32_t>(cfg_.schemMode));
  return InitTVMTask();
}

Status KernelTaskInfo::InitTVMTask(const domi::KernelDef &kernel_def) {
  GELOGD("Do InitTVMTask of %s.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InitTVMContext(kernel_def.context()));
  GELOGI("io_addrs_size:%zu, args_size:%zu", io_addrs_.size(), kernel_def.args().size() / kAddressLen);
  if ((io_addrs_.size() * kAddressLen) < kernel_def.args().size()) {
    const size_t offset = io_addrs_.size() * kAddressLen;
    const size_t len = kernel_def.args().size() - offset;
    io_addrs_.resize(MemSizeAlign(static_cast<size_t>(kernel_def.args().size()), kAddressLen) / kAddressLen);
    uint8_t *dst_addr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(io_addrs_.data())) + offset;
    uint8_t *src_addr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(kernel_def.args().data())) + offset;
    const errno_t sec_ret = memcpy_s(dst_addr, len, src_addr, len);
    GE_ASSERT_TRUE(sec_ret == EOK);
    io_addr_mem_types_.resize(io_addrs_.size(), static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
  }

  cfg_.schemMode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
  GELOGD("OpName: %s set schedule mode from kernel def: %u",
      op_desc_->GetName().c_str(), static_cast<uint32_t>(cfg_.schemMode));
  return InitTVMTask();
}

bool KernelTaskInfo::HasOverflowAddr(const OpDescPtr &op_desc) const {
  return (davinci_model_->GetOverflowAddr() != nullptr) &&
         AttrUtils::HasAttr(op_desc, GLOBALWORKSPACE_TYPE);
}

Status KernelTaskInfo::InitTVMTask() {
  GE_CHECK_NOTNULL(davinci_model_);
  if (is_separately_clean_task_) {
    UpdateAtomicCleanArgs(input_data_addrs_, output_data_addrs_, workspace_addrs_);
  }

  GE_CHK_STATUS_RET(SetTvmTaskZeroCopy(op_desc_, io_addrs_));
  GELOGD("Do InitTVMTask end");
  return SUCCESS;
}

Status KernelTaskInfo::SetTvmTaskZeroCopy(const OpDescPtr &op_desc, const std::vector<uint64_t> &virtual_io_addrs) {
  const auto zero_copy_args_index = davinci_model_->GetZeroCopyArgsIndex(virtual_io_addrs);
  if (!zero_copy_args_index.empty()) {
    const std::vector<bool> input_raw_data_list = ModelUtils::GetInputTensorNeedRawData(op_desc);
    std::vector<bool> need_raw_data_list;
    std::map<uintptr_t, std::set<size_t>> zero_copy_args_offset;
    const auto &input_offsets = op_desc->GetInputOffset();
    for (const auto args_index : zero_copy_args_index) {
      if (args_index < input_offsets.size()) {
        int64_t inner_offset = 0;
        (void)AttrUtils::GetInt(op_desc->GetInputDesc(static_cast<uint32_t>(args_index)), ATTR_NAME_INNER_OFFSET,
                                inner_offset);
        const uint64_t session_id = davinci_model_->GetRuntimeParam().session_id;
        if ((davinci_model_->GetRuntimeParam().var_size > 0U) &&
            VarManager::Instance(session_id)->IsVarAddr(input_offsets[args_index] - inner_offset)) {
          GELOGI("Node:%s input:%" PRIu64 " is var, no need zero copy refresh.",
            op_desc->GetName().c_str(), args_index);
          continue;
        }
      }
      const size_t args_offset_tmp = (args_index * kAddressLen) + io_addr_offset_;
      (void)zero_copy_args_offset[static_cast<size_t>(virtual_io_addrs[args_index])].insert(args_offset_tmp);
      if (args_index < input_raw_data_list.size()) {
        need_raw_data_list.push_back(input_raw_data_list[args_index]);
      }
    }
    need_raw_data_list.resize(zero_copy_args_index.size(), false);
    GE_CHK_STATUS_RET(davinci_model_->Mapping2BundleZeroCopy(op_desc, zero_copy_args_offset, need_raw_data_list,
                                                             static_cast<size_t>(args_size_), args_addr_.data(), args_,
                                                             own_args_memory_,
                                                             IsAllKernel(task_type_)),
                      "Failed mapping zero copy task for %s to bundle task", op_desc->GetName().c_str());
  }
  return SUCCESS;
}

bool KernelTaskInfo::IsL1OrUBFusionOp(const OpDescPtr &op_desc) const {
  std::vector<int64_t> input_memory_type;
  (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_type);
  for (const int64_t i : input_memory_type) {
    if ((i == static_cast<int64_t>(RT_MEMORY_L1)) || (i == static_cast<int64_t>(kRtMemoryUB))) {
      return true;
    }
  }

  std::vector<int64_t> output_memory_type;
  (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_type);
  for (const int64_t type : output_memory_type) {
    if ((type == static_cast<int64_t>(RT_MEMORY_L1)) || (type == static_cast<int64_t>(kRtMemoryUB))) {
      return true;
    }
  }
  return false;
}

Status KernelTaskInfo::InitAICPUCustomTask(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAICPUCustomTask");
  const domi::KernelContext &context = kernel_def.context();
  constexpr uint32_t kCustomAicpuArgsLen = 5U;
  ctx_.argsOffset.resize(kCustomAicpuArgsLen);

  if ((context.args_offset().size() / sizeof(uint16_t)) < kCustomAicpuArgsLen) {
    REPORT_INNER_ERR_MSG("E19999", "context.args_offset().size():%zu / sizeof(uint16_t) is less than "
                       "kCustomAicpuArgsLen:%u, op:%s(%s), check invalid", context.args_offset().size(),
                       kCustomAicpuArgsLen, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] context.args_offset().size():%zu / sizeof(uint16_t) is less than "
           "kCustomAicpuArgsLen:%u, op:%s(%s)", context.args_offset().size(), kCustomAicpuArgsLen,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  GE_ASSERT_EOK(memcpy_s(ctx_.argsOffset.data(), (sizeof(uint16_t) * kCustomAicpuArgsLen), context.args_offset().data(),
      (sizeof(uint16_t) * kCustomAicpuArgsLen)));

  const Status ret = StoreInputOutputTensor(input_data_addrs_, output_data_addrs_, ModelUtils::GetInputDescs(op_desc),
                                            ModelUtils::GetOutputDescs(op_desc));
  if (ret != SUCCESS) {
    GELOGE(ret, "[Store][InputOutputTensor] Failed, op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ret;
  }

  // attrHandle
  Buffer buffer;
  if (!AttrUtils::GetBytes(op_desc, ATTR_NAME_OPATTR, buffer)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail", ATTR_NAME_OPATTR.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s in op:%s(%s) fail", ATTR_NAME_OPATTR.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  const uint64_t op_attr_size = static_cast<uint64_t>(buffer.GetSize());
  if (op_attr_size == 0U) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s in op:%s(%s) size is 0, check invalid",
                       ATTR_NAME_OPATTR.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] param op_attr_size is out of range, op:%s", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  custom_info_.attr_handle = davinci_model_->MallocDynamicMemory(op_attr_size);
  GE_ASSERT_NOTNULL(custom_info_.attr_handle);
  GE_CHK_RT_RET(rtMemcpy(custom_info_.attr_handle, op_attr_size, buffer.GetData(), op_attr_size,
                         RT_MEMCPY_HOST_TO_DEVICE));

  GE_ASSERT_TRUE((io_addrs_.size() * kAddressLen) >= kernel_def.args().size());
  const errno_t sec_ret =
      memcpy_s(io_addrs_.data(), kernel_def.args().size(), kernel_def.args().data(), kernel_def.args().size());
  if (sec_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "Call memcpy_s fail, size:%zu, ret:%d", kernel_def.args().size(), sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%zu, ret:%d", kernel_def.args().size(), sec_ret);
    return FAILED;
  }

  for (uint32_t i = 0U; i < kCustomAicpuArgsLen; ++i) {
    if (kernel_def.args().size() < (static_cast<size_t>(ctx_.argsOffset[static_cast<size_t>(i)]) + sizeof(uint64_t))) {
      REPORT_INNER_ERR_MSG("E19999", "ctx.argsOffset[%u]: %u + sizeof(uint64_t): %zu >= kernelDef.args().size():%zu, "
                         "op:%s(%s) check invalid", i, static_cast<uint32_t>(ctx_.argsOffset[static_cast<size_t>(i)]),
                         sizeof(uint64_t), kernel_def.args().size(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] ctx.argsOffset[%u]:%u + sizeof(uint64_t):%zu >= kernelDef.args().size():%zu", i,
             static_cast<uint32_t>(ctx_.argsOffset[static_cast<size_t>(i)]), sizeof(uint64_t),
             kernel_def.args().size());
      return FAILED;
    }
  }
  // arg 0
  const size_t input_desc_pos = static_cast<size_t>(ctx_.argsOffset[kArgsInputDesc]) / kAddressLen;
  io_addrs_[input_desc_pos] = PtrToValue(custom_info_.input_descs);
  // arg 1
  const size_t input_addr_pos = static_cast<size_t>(ctx_.argsOffset[kArgsInputAddr]) / kAddressLen;
  io_addrs_[input_addr_pos] = PtrToValue(custom_info_.input_addrs);
  // arg 2
  const size_t output_desc_pos = static_cast<size_t>(ctx_.argsOffset[kArgsOutputDesc]) / kAddressLen;
  io_addrs_[output_desc_pos] = PtrToValue(custom_info_.output_descs);
  // arg 3
  const size_t output_addr_pos = static_cast<size_t>(ctx_.argsOffset[kArgsOutputAddr]) / kAddressLen;
  io_addrs_[output_addr_pos] = PtrToValue(custom_info_.output_addrs);
  // arg 4
  const size_t attr_handle_pos = static_cast<size_t>(ctx_.argsOffset[kArgsAttrHandle]) / kAddressLen;
  io_addrs_[attr_handle_pos] = PtrToValue(custom_info_.attr_handle);
  std::stringstream ss;
  ss << "customized io_addrs info are size:" << io_addrs_.size()
     << ", arg_0 offset/pos/addr:" << ctx_.argsOffset[kArgsInputDesc] << "/" << input_desc_pos << "/"
     << io_addrs_[input_desc_pos]
     << ", arg_1 offset/pos/addr:" << ctx_.argsOffset[kArgsInputAddr] << "/" << input_addr_pos << "/"
     << io_addrs_[input_addr_pos]
     << ", arg_2 offset/pos/addr:" << ctx_.argsOffset[kArgsOutputDesc] << "/" << output_desc_pos << "/"
     << io_addrs_[output_desc_pos]
     << ", arg_3 offset/pos/addr:" << ctx_.argsOffset[kArgsOutputAddr] << "/" << output_addr_pos << "/"
     << io_addrs_[output_addr_pos]
     << ", arg_4 offset/pos/addr:" << ctx_.argsOffset[kArgsAttrHandle] << "/" << attr_handle_pos << "/"
     << io_addrs_[attr_handle_pos];
  GELOGD("%s ", ss.str().c_str());

  const std::vector<bool> need_raw_data_list = ModelUtils::GetInputTensorNeedRawData(op_desc);
  davinci_model_->SetZeroCopyAddr(op_desc, input_data_addrs_, input_data_addrs_.data(),
                                  static_cast<uintptr_t>(PtrToValue(custom_info_.input_addrs)),
                                  input_data_addrs_.size() * kAddressLen, 0U,
                                  need_raw_data_list);
  davinci_model_->SetZeroCopyAddr(op_desc, output_data_addrs_, output_data_addrs_.data(),
                                  static_cast<uintptr_t>(PtrToValue(custom_info_.output_addrs)),
                                  output_data_addrs_.size() * kAddressLen, 0U,
                                  {});
  return SUCCESS;
}

Status KernelTaskInfo::InitPreprocessTask(const OpDescPtr &op_desc) {
  if (task_type_ != ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL) {
    return SUCCESS;
  }
  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    const bool has_space_type = SetHasMemoryLog();
    GELOGD("op[%s] has_space_type[%u]", op_desc->GetName().c_str(), static_cast<uint32_t>(has_space_type));
  } else if (kernel_type_ == ccKernelType::AI_CPU) {
    GE_ASSERT_SUCCESS(
        ModelManager::GetInstance().LoadBuiltinAicpuSoAndUpdateSoName(davinci_model_->GetDeviceId(), so_name_),
        "[OpMasterDevice][BuiltIn]Load so failed for node: %s", op_desc->GetNamePtr());
  } else {
    REPORT_INNER_ERR_MSG("E19999", "kernel type:%u not supported", static_cast<uint32_t>(kernel_type_));
    GELOGE(FAILED, "kernel type:%u not supported", static_cast<uint32_t>(kernel_type_));
    return FAILED;
  }
  args_ = ValueToPtr(PtrToValue(args_) + args_format_holder_.sink_tensor_size + sizeof(aicpu::AicpuParamHead));
  GELOGI("Update preprocess task args[0x%" PRIx64 "] so_name[%s] for op[%s]", args_, so_name_.c_str(),
         op_desc->GetName().c_str());
  return SUCCESS;
}

Status KernelTaskInfo::InitAicpuTask(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAicpuTask");
  so_name_ = kernel_def.so_name();
  kernel_name_ = kernel_def.kernel_name();
  GE_CHECK_GE(args_size_, sizeof(aicpu::AicpuParamHead));
  io_addr_offset_ = sizeof(aicpu::AicpuParamHead);
  GELOGI("node[%s] so name %s, kernel name %s", op_desc->GetName().c_str(), so_name_.c_str(), kernel_name_.c_str());

  GE_ASSERT_SUCCESS(InitPreprocessTask(op_desc), "Init preprocess task of op[%s] failed.", op_desc->GetName().c_str());

  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    bool loaded = false;
    auto &model_mgr = ModelManager::GetInstance();
    GE_CHK_STATUS_RET(model_mgr.LoadCustAicpuSo(davinci_model_->GetCustAICPUKernel(op_desc), so_name_, loaded),
                      "[Launch][CustAicpuSo] failed, node: %s", op_desc->GetName().c_str());
  }

  // copy args to new host memory
  args_addr_.resize(static_cast<size_t>(args_size_));
  GE_PRINT_DYNAMIC_MEMORY(new, "cce task physical memory.", sizeof(uint8_t) * args_size_);
  const errno_t sec_ret = memcpy_s(args_addr_.data(), static_cast<size_t>(args_size_), kernel_def.args().data(),
                                   static_cast<size_t>(kernel_def.args().size()));
  if (sec_ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "Call memcpy_s fail, size:%u, ret:%d", args_size_, sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%u, ret:%d", args_size_, sec_ret);
    return FAILED;
  }
  const auto aicpu_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(args_addr_.data());
  const auto &ext_info = kernel_def.kernel_ext_info();
  const auto init_ret = InitAicpuTaskExtInfo(ext_info);
  if (init_ret != SUCCESS) {
    GELOGE(init_ret, "[Init][AicpuTaskExtInfo] failed, ext_info size=%zu", ext_info.size());
    return init_ret;
  }
  GELOGI("Node[%s] type[%s] kernel_ext_info size=%zu, aicpu_ext_info_addr_=%p", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), ext_info.size(), aicpu_ext_info_addr_);

  aicpu_param_head->extInfoAddr = PtrToValue(aicpu_ext_info_addr_);
  aicpu_param_head->extInfoLength = static_cast<uint32_t>(ext_info.size());

  if (!io_addrs_.empty()) {
    // refresh io addrs
    const size_t addrs_size = sizeof(uint64_t) * io_addrs_.size();
    const size_t args_size_max = static_cast<size_t>(args_size_) - io_addr_offset_;
    GE_ASSERT_EOK(memcpy_s(&(args_addr_[io_addr_offset_]), args_size_max, io_addrs_.data(), addrs_size));
  }

  GE_CHK_STATUS_RET(AssembleArgs(io_addrs_), "[Assemble][Addrs] failed, node: %s", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status KernelTaskInfo::InitAicpuKfcTask(const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAicpuKfcTask, op:[%s].", op_desc_->GetNamePtr());
  so_name_ = kernel_def.so_name();
  kernel_name_ = kernel_def.kernel_name();
  GELOGI("node[%s] so name %s, kernel name %s", op_desc_->GetName().c_str(), so_name_.c_str(), kernel_name_.c_str());

  // copy args to new host memory
  args_addr_.resize(static_cast<size_t>(args_size_));
  args_addr_.assign(kernel_def.args().begin(), kernel_def.args().end());

  size_t cur_offset = 0UL;
  size_t args_size_max = static_cast<size_t>(args_size_);
  if (!io_addrs_.empty()) {
    // refresh io addrs
    const size_t addrs_size = sizeof(uint64_t) * io_addrs_.size();
    GE_ASSERT_EOK(memcpy_s(&(args_addr_[cur_offset]), args_size_max, io_addrs_.data(), addrs_size));
    cur_offset += addrs_size;
    args_size_max -= addrs_size;
  }
  // kernel_name
  aicpu_args_ex_.kernelNameAddrOffset = static_cast<uint32_t>(cur_offset - args_format_holder_.sink_tensor_size);
  const size_t kernel_name_size = kernel_name_.size() + 1UL;
  GE_ASSERT(args_size_max >= kernel_name_size);
  GE_ASSERT_EOK(strncpy_s(reinterpret_cast<char_t *>(&(args_addr_[cur_offset])), args_size_max, kernel_name_.c_str(),
                          kernel_name_.size()));
  cur_offset += kernel_name_size;
  args_size_max -= kernel_name_size;

  // so_name
  aicpu_args_ex_.soNameAddrOffset = static_cast<uint32_t>(cur_offset - args_format_holder_.sink_tensor_size);
  const size_t so_name_size = so_name_.size() + 1UL;
  GE_ASSERT(args_size_max >= so_name_size);
  GE_ASSERT_EOK(strncpy_s(reinterpret_cast<char_t *>(&(args_addr_[cur_offset])), args_size_max, so_name_.c_str(),
                          so_name_.size()));
  cur_offset += so_name_size;
  args_size_max -= so_name_size;

  args_ = ValueToPtr(PtrToValue(args_) + args_format_holder_.sink_tensor_size);

  GELOGD(
      "Init AicpuKfc Kernel successfully, Op:[%s], kernel_name_offset:[%u], so_name_offset:[%u], "
      "sink_tensor_size:[%zu]. ",
      op_desc_->GetNamePtr(), aicpu_args_ex_.kernelNameAddrOffset,
      aicpu_args_ex_.soNameAddrOffset, args_format_holder_.sink_tensor_size);
  GE_CHK_STATUS_RET(AssembleArgs(io_addrs_), "[Assemble][Addrs] failed, node: %s", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status KernelTaskInfo::AssembleArgs(const std::vector<uint64_t> &io_addrs) {
  std::vector<bool> need_raw_data_list = ModelUtils::GetInputTensorNeedRawData(op_desc_);
  need_raw_data_list.resize(io_addrs.size(), false);
  // copy args memory to device
  davinci_model_->SetZeroCopyAddr(op_desc_, io_addrs, args_addr_.data(), static_cast<uintptr_t>(PtrToValue(args_)),
                                  static_cast<size_t>(args_size_), io_addr_offset_, need_raw_data_list);
  GELOGI("op %s use device mem %p for arg info with flag %d", op_desc_->GetName().c_str(), args_, deploy_type_flag_);
  return SUCCESS;
}

void KernelTaskInfo::InitFusionDumpInfo(const OpDescPtr &op_desc, const domi::TaskDef &task_def) {
  // fusion_op_info
  std::vector<std::string> original_op_names;
  (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);
  if (!original_op_names.empty()) {
    FusionOpInfo fusion_op_info;
    fusion_op_info.stream_id = task_def.stream_id();
    fusion_op_info.op_index = ctx_.opIndex;
    fusion_op_info.original_op_names = original_op_names;
    fusion_op_info.op_name = op_desc->GetName();
    fusion_op_info_.emplace_back(fusion_op_info);
  }

  if ((davinci_model_->OpNeedDump(op_desc) && (!is_separately_clean_task_)) || davinci_model_->OpNeedPrint(op_desc)
    || davinci_model_->OpNeedSetDumpFlagOnWatcherModel(op_desc->GetName())) {
    GELOGD("Op %s init dump flag", op_desc->GetName().c_str());
    dump_flag_ = IsL1OrUBFusionOp(op_desc) ? static_cast<uint32_t>(RT_FUSION_KERNEL_DUMPFLAG)
                                       : static_cast<uint32_t>(RT_KERNEL_DUMPFLAG);
    is_data_dump_ = true;
  }
}

void KernelTaskInfo::InitDumpArgs(const size_t offset) {
  static const std::set<ccKernelType> dump_able{ccKernelType::TE,          ccKernelType::AI_CPU,
                                                ccKernelType::CUST_AI_CPU, ccKernelType::AI_CPU_KFC,
                                                ccKernelType::MIX_AICORE,  ccKernelType::MIX_VECTOR_CORE};
  if (dump_able.count(kernel_type_) == 0U) {
    return;
  }

  if (!args_format_holder_.arg_descs.empty() && cust_to_relevant_offset_.empty()) {
    // tiling下沉的aicpu task通过上报dump信息打印device侧log
    dump_flag_ |= ((kernel_type_ == ccKernelType::CUST_AI_CPU) ? RT_KERNEL_CUSTOM_AICPU : RT_KERNEL_DEFAULT);
  }

  if (davinci_model_->OpNeedDump(op_desc_) || davinci_model_->OpNeedDumpOnWatcherModel(op_desc_->GetName())) {
    GELOGD("Op %s need dump in task info", op_desc_->GetName().c_str());
    dump_args_ = PtrAdd(PtrToPtr<void, uint8_t>(args_), static_cast<size_t>(args_size_), offset);
  }

  if (davinci_model_->GetOpDugReg()) {
    GELOGD("Op debug is open in kernel task info");
    dump_args_ = PtrAdd(PtrToPtr<void, uint8_t>(args_), static_cast<size_t>(args_size_), offset);
  }
  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    dump_flag_ |= RT_KERNEL_CUSTOM_AICPU;
  }
  GELOGD("Op[%s], dump_flag_[%u], dump_args_[0x%" PRIx64 "]", op_desc_->GetName().c_str(), dump_flag_, dump_args_);
}

bool KernelTaskInfo::SetHasMemoryLog() {
  std::vector<int64_t> space_type;
  const bool has_space_type = ge::AttrUtils::GetListInt(op_desc_, ATTR_NAME_AICPU_WORKSPACE_TYPE, space_type);
  if (has_space_type) {
    const auto result = std::find(space_type.begin(), space_type.end(), ge::AicpuWorkSpaceType::CUST_LOG);
    if (result != space_type.end()) {
      has_memory_log_ = true;
    }
  }
  return has_space_type;
}

Status KernelTaskInfo::InitAicpuTaskExtInfo(const std::string &ext_info) {
  if (ext_info.empty()) {
    return SUCCESS;
  }
  std::unique_ptr<hybrid::AicpuExtInfoHandler> ex_handle = nullptr;
  GE_ASSERT_SUCCESS(ParseAicpuExtInfoHandler(op_desc_, ext_info, ex_handle));
  GE_ASSERT_NOTNULL(ex_handle);
  GE_CHK_STATUS_RET(ex_handle->UpdateSessionInfoId(davinci_model_->GetSessionId()),
                    "[Update][SessionInfoSessionId] failed, op:%s", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET(ex_handle->UpdateExecuteMode(true), "[Update][ExecuteMode] failed, op:%s",
                    op_desc_->GetName().c_str());

  deploy_type_flag_ = ex_handle->GetDeployTypeFlag();
  qos_level_flag_ = ex_handle->GeQosLevelFlag();

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc_, kAllShapeInAicpu, all_shape);
  if (all_shape) {
    for (uint32_t i = 0U; i < static_cast<uint32_t>(op_desc_->GetInputsSize()); i++) {
      const auto input_desc = op_desc_->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET(ex_handle->UpdateInputShapeAndType(i, *input_desc),
                        "[Call][UpdateInputShapeAndType] Input[%u] update input shape failed, op:%s.",
                        i, op_desc_->GetName().c_str());
    }
    for (uint32_t j = 0U; j < static_cast<uint32_t>(op_desc_->GetOutputsSize()); j++) {
      const auto output_desc = op_desc_->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);
      GE_CHK_STATUS_RET(ex_handle->UpdateOutputShapeAndType(j, *output_desc),
                        "[Call][UpdateOutputShapeAndType] Output[%u] update output shape failed, op:%s.",
                        j, op_desc_->GetName().c_str());
    }
  }

  const bool has_space_type = SetHasMemoryLog();
  if (has_memory_log_) {
    const std::vector<int64_t> v_workspace_size = op_desc_->GetWorkspaceBytes();
    if ((!workspace_addrs_.empty()) && (!v_workspace_size.empty())) {
      // workspace地址不支持刷新，io复用或者fm可刷新场景下无法获取到执行时实际地址，使用固定内存
      rtMemType_t mem_type = ex_handle->GetMemType();
      size_t workspace_size = static_cast<size_t>(v_workspace_size[0]);
      void *workspace_addr = davinci_model_->MallocDynamicMemory(workspace_size, mem_type);
      GE_ASSERT_NOTNULL(workspace_addr);
      GELOGI("workspace_info addr:%p, size:%zu, mem type:%u.", workspace_addr, workspace_size, mem_type);
      GE_CHK_STATUS_RET(
          ex_handle->UpdateWorkSpaceInfo(static_cast<uint64_t>(workspace_size), PtrToValue(workspace_addr)),
          "[Call][UpdateWorkSpaceInfo] failed, op:%s.", op_desc_->GetName().c_str());
    }
  }

  (void)AttrUtils::GetBool(op_desc_, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op_);
  GELOGI(
      "op:%s ext_info: session_id: %" PRIu64 ", deploy_type_flag: %d, qos_level_flag: %u, all_shape: %d, "
      "aicpu_workspace_type: %d, is_blocking_op:%d",
      op_desc_->GetName().c_str(), davinci_model_->GetSessionId(), deploy_type_flag_, qos_level_flag_,
      static_cast<int32_t>(all_shape), static_cast<int32_t>(has_space_type),
      static_cast<int32_t>(is_blocking_aicpu_op_));

  if (UpdateEventIdForAicpuBlockingOp(*ex_handle) != SUCCESS) {
    GELOGE(FAILED, "[Call][UpdateEventIdForAicpuBlockingOp] failed for op:%s(%s)", op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str());
    return FAILED;
  }
  GE_CHK_STATUS_RET(UpdateExtraInfo(*ex_handle), "[Call][UpdateExtraInfo] failed, op:%s.", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status KernelTaskInfo::UpdateExtraInfo(const hybrid::AicpuExtInfoHandler &ext_handle) {
  if (deploy_type_flag_ == static_cast<int32_t>(RT_KERNEL_HOST_ONLY)) {
    aicpu_ext_info_addr_ = davinci_model_->MallocDynamicMemory(ext_handle.GetExtInfoLen(), RT_MEMORY_HOST_SVM);
    GE_ASSERT_NOTNULL(aicpu_ext_info_addr_);
    GE_CHK_RT_RET(rtMemcpy(aicpu_ext_info_addr_, ext_handle.GetExtInfoLen(), ext_handle.GetExtInfo(),
                           ext_handle.GetExtInfoLen(), RT_MEMCPY_HOST_TO_HOST));
    GELOGI("op %s use host mem %p for ext info", op_desc_->GetName().c_str(), aicpu_ext_info_addr_);
  } else {
    aicpu_ext_info_addr_ = davinci_model_->MallocDynamicMemory(ext_handle.GetExtInfoLen());
    GE_ASSERT_NOTNULL(aicpu_ext_info_addr_);
    GE_CHK_RT_RET(rtMemcpy(aicpu_ext_info_addr_, ext_handle.GetExtInfoLen(), ext_handle.GetExtInfo(),
                           ext_handle.GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE));
    GELOGI("op %s use device mem %p for ext info with flag %d", op_desc_->GetName().c_str(), aicpu_ext_info_addr_,
           deploy_type_flag_);
  }
  return SUCCESS;
}

Status KernelTaskInfo::StoreInputOutputTensor(const std::vector<uint64_t> &input_data_addrs,
                                              const std::vector<uint64_t> &output_data_addrs,
                                              const std::vector<ccAICPUTensor> &input_descs,
                                              const std::vector<ccAICPUTensor> &output_descs) {
  if (!input_data_addrs.empty()) {
    const auto input_size = input_descs.size();
    const size_t total_desc_size = sizeof(ccAICPUTensor) * input_size;
    // inputDescs
    custom_info_.input_descs = davinci_model_->MallocDynamicMemory(total_desc_size);
    GE_ASSERT_NOTNULL(custom_info_.input_descs);
    GE_CHK_RT_RET(rtMemcpy(custom_info_.input_descs, total_desc_size, input_descs.data(), total_desc_size,
                           RT_MEMCPY_HOST_TO_DEVICE));

    // inputAddrs
    GE_ASSERT_TRUE(customized_args_info_.input_addr_offset <= args_size_);
    uint64_t *const args_tmp = PtrToPtr<void, uint64_t>(args_);
    custom_info_.input_addrs = ValueToPtr(args_tmp[customized_args_info_.input_addr_offset / kAddressLen]);
  }

  if (!output_data_addrs.empty()) {
    const auto output_size = output_descs.size();
    const size_t total_desc_size = sizeof(ccAICPUTensor) * output_size;
    // outputDescs
    custom_info_.output_descs = davinci_model_->MallocDynamicMemory(total_desc_size);
    GE_ASSERT_NOTNULL(custom_info_.output_descs);
    GE_CHK_RT_RET(rtMemcpy(custom_info_.output_descs, total_desc_size, output_descs.data(),
                           sizeof(ccAICPUTensor) * output_size, RT_MEMCPY_HOST_TO_DEVICE));

    // outputAddrs
    GE_ASSERT_TRUE(customized_args_info_.output_addr_offset <= args_size_);
    uint64_t *const args_tmp = PtrToPtr<void, uint64_t>(args_);
    custom_info_.output_addrs = ValueToPtr(args_tmp[customized_args_info_.output_addr_offset / kAddressLen]);
  }

  return SUCCESS;
}

Status KernelTaskInfo::AssembleKernelNamesAndLaunch() {
  rtKernelLaunchNames_t launch_name{};
  const std::string op_name(op_desc_->GetName());
  if (deploy_type_flag_ == static_cast<int32_t>(RT_KERNEL_HOST_ONLY)) {
    if (ge::CheckSizeTAddOverflow(so_name_.size(), kernel_name_.size()) != ge::SUCCESS) {
      return FAILED;
    }
    size_t total_launch_size = so_name_.size() + kernel_name_.size();
    if (ge::CheckSizeTAddOverflow(total_launch_size, op_name.size()) != ge::SUCCESS) {
      return FAILED;
    }
    total_launch_size += op_name.size();
    const std::string launch_info = so_name_ + kernel_name_ + op_name;
    if (launch_addr_ == nullptr) {
      launch_addr_ = davinci_model_->MallocDynamicMemory(total_launch_size, RT_MEMORY_HOST_SVM);
    }
    GE_ASSERT_NOTNULL(launch_addr_);
    GE_CHK_RT_RET(
        rtMemcpy(launch_addr_, launch_info.size(), launch_info.c_str(), launch_info.size(), RT_MEMCPY_HOST_TO_HOST));
    launch_name.soName = PtrToPtr<void, const char>(launch_addr_);
    launch_name.kernelName = PtrAdd(PtrToPtr<void, const char>(launch_addr_), total_launch_size, so_name_.size());
    launch_name.opName =
        PtrAdd(PtrToPtr<void, const char>(launch_addr_), total_launch_size, so_name_.size() + kernel_name_.size());

    if (kernel_name_arg_ == nullptr) {
      kernel_name_arg_ = davinci_model_->MallocDynamicMemory(sizeof(rtKernelLaunchNames_t), RT_MEMORY_HOST_SVM);
    }
    GE_ASSERT_NOTNULL(kernel_name_arg_);
    GELOGI("Using host mem info: kernel_name_arg_ %p, so_name_host_ %p, kernel_name_host_ %p, op_name_host_ %p",
           kernel_name_arg_, launch_name.soName, launch_name.kernelName, launch_name.opName);
    GE_CHK_RT_RET(rtMemcpy(kernel_name_arg_, sizeof(rtKernelLaunchNames_t),
                           PtrToPtr<rtKernelLaunchNames_t, void>(&launch_name), sizeof(rtKernelLaunchNames_t),
                           RT_MEMCPY_HOST_TO_HOST));
  } else {
    launch_name.soName = so_name_.c_str();
    launch_name.kernelName = kernel_name_.c_str();
    launch_name.opName = op_name.c_str();
    kernel_name_arg_ = PtrToPtr<rtKernelLaunchNames_t, void>(&launch_name);
  }
  SetTaskTag(op_name.c_str());
  args_ex_.args = args_;
  args_ex_.argsSize = args_size_;
  args_ex_.isNoNeedH2DCopy = 1;
  // blockDim is reserved parameter, set to 1
  GE_CHK_RT_RET(rtAicpuKernelLaunchWithFlag(PtrToPtr<void, rtKernelLaunchNames_t>(kernel_name_arg_), block_dim_,
                                            &args_ex_, nullptr, stream_, dump_flag_));
  is_support_redistribute_ = true;
  return SUCCESS;
}

int64_t KernelTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  domi::KernelContext context;
  if (!IsAllKernel(task_type)) {
    const domi::KernelDef &kernel_def = task_def.kernel();
    context = kernel_def.context();
  } else {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    context = kernel_def.context();
  }
  return static_cast<int64_t>(context.op_index());
}

size_t KernelTaskInfo::GetExtraArgsSize(const DavinciModel &davinci_model, const OpDescPtr &op_desc,
                                        const ccKernelType kernel_type) {
  size_t extra_size = 0UL;
  int32_t max_tiling_len{-1};
  (void)AttrUtils::GetInt(op_desc, kMaxTilingSize, max_tiling_len);
  int32_t max_atomic_tiling_len{-1};
  (void)AttrUtils::GetInt(op_desc, kMaxAtomicCleanTilingSize, max_atomic_tiling_len);
  if ((max_tiling_len > 0) || (max_atomic_tiling_len > 0)) {
    extra_size += kAddressLen;
  }

  if ((davinci_model.GetOverflowAddr() != nullptr) && AttrUtils::HasAttr(op_desc, GLOBALWORKSPACE_TYPE)) {
    extra_size += kAddressLen;
  }

  if (kernel_type == ccKernelType::TE) {
    const auto is_wsp_addr_folded = IsWspAddrFolded(op_desc);
    if (is_wsp_addr_folded) {
      // kAddressLen: if folded mode, need add a memory for point to wsl addr list
      // kUBAlignedLen:
      // reserved 32B for aligned start with wsl addr list
      // -----------------------------------------------------------
      // | point to wsl addr list | over flow addr | wsl addr list |
      // -----------------------------------------------------------
      extra_size += kAddressLen + kUBAlignedLen;
    }
  }

  // level2 addr
  const size_t shape_info_size = args_format_holder_.level1_addr_cnt * sizeof(int64_t);
  extra_size += shape_info_size;

  // tiling sink tensor size
  size_t rt_tensor_offset{0UL};
  size_t rt_tensor_size{0UL};
  GetAddrAlignedGertTensorSize(rt_tensor_offset, rt_tensor_size);
  const size_t total_size = rt_tensor_size * args_format_holder_.tiling_depends_input_idx.size();
  extra_size += total_size * args_format_holder_.tiling_depends_input_idx.size();

  return extra_size;
}

Status KernelTaskInfo::UpdateArgsSizeWithCustomized(const OpDescPtr &op_desc) {
  GE_ASSERT_NOTNULL(op_desc);
  args_size_ = static_cast<uint32_t>(MemSizeAlign(static_cast<size_t>(args_size_)));
  customized_args_info_.customized_aligned = true;

  GE_ASSERT_TRUE(
      !ge::MulOverflow(ModelUtils::GetInputDescs(op_desc).size(), kAddressLen, customized_args_info_.input_addr_size));
  customized_args_info_.input_addr_offset = args_size_;
  GE_ASSERT_TRUE(!AddOverflow(args_size_, customized_args_info_.input_addr_size, args_size_));

  GE_ASSERT_TRUE(
      !ge::MulOverflow(ModelUtils::GetOutputDescs(op_desc).size(),
                       kAddressLen, customized_args_info_.output_addr_size));
  customized_args_info_.output_addr_offset = args_size_;
  GE_ASSERT_TRUE(!AddOverflow(args_size_, customized_args_info_.input_addr_size, args_size_));

  std::stringstream ss;
  ss << "customized_args_info: args/after_align:" << customized_args_info_.kernel_def_args_size << "/ " << args_size_
     << ", is aligned: " << customized_args_info_.customized_aligned << ", input_addr_size/offset is "
     << customized_args_info_.input_addr_size << " / " << customized_args_info_.input_addr_offset
     << ", output_addr_size/offset is " << customized_args_info_.output_addr_size << " / "
     << customized_args_info_.output_addr_offset;
  GELOGD("%s ", ss.str().c_str());
  return SUCCESS;
}

Status KernelTaskInfo::ParseAicpuExtInfoHandler(const OpDescPtr &op_desc, const string &ext_info,
                                                std::unique_ptr<hybrid::AicpuExtInfoHandler> &ex_handle) const {
  if (ext_info.empty()) {
    return SUCCESS;
  }
  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  const auto unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  const uint32_t num_inputs = static_cast<uint32_t>(op_desc->GetInputsSize());
  const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());

  ex_handle = MakeUnique<hybrid::AicpuExtInfoHandler>(op_desc->GetName(), num_inputs, num_outputs, unknown_type);
  GE_CHECK_NOTNULL(ex_handle);
  GE_CHK_STATUS_RET(ex_handle->Parse(ext_info), "[Parse][KernelExtInfo] failed, kernel_ext_info_size=%zu, op:%s.",
                    ext_info.size(), op_desc_->GetName().c_str());
  return SUCCESS;
};

REGISTER_TASK_INFO(MODEL_TASK_KERNEL, KernelTaskInfo);
REGISTER_TASK_INFO(MODEL_TASK_ALL_KERNEL, KernelTaskInfo);
REGISTER_TASK_INFO(MODEL_TASK_VECTOR_KERNEL, KernelTaskInfo);
REGISTER_TASK_INFO(MODEL_TASK_VECTOR_ALL_KERNEL, KernelTaskInfo);
REGISTER_TASK_INFO(MODEL_TASK_PREPROCESS_KERNEL, KernelTaskInfo);
}  // namespace ge
