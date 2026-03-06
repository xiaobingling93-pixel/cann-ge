/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/task/tbe_task_builder.h"
#include <mutex>
#include <vector>

#include "graph/utils/math_util.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/def_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/rt.h"
#include "single_op/task/build_task_utils.h"

namespace ge {
namespace {
const std::string kSupportDynamicShape = "support_dynamicshape";
const std::string kOpParamSize = "op_para_size";
const std::string kAtomicOpParamSize = "atomic_op_para_size";
const std::string kTbeCoreTypeMixAic = "MIX_AIC";
const std::string kTbeCoreTypeMixAiv = "MIX_AIV";
const std::set<std::string> kTbeCoreTypeMix = { "MIX_AIC", "MIX_AIV", "MIX"};
constexpr size_t kBufAlignedBytes = 128UL;
std::mutex g_reg_mutex;
}  // namespace

TbeTaskBuilder::TbeTaskBuilder(const std::string &model_name, const NodePtr &node, const domi::TaskDef &task_def)
    : node_(node),
      op_desc_(node->GetOpDesc()),
      task_def_(task_def),
      kernel_def_(task_def.kernel()),
      kernel_def_with_handle_(task_def.kernel_with_handle()),
      model_name_(model_name) {}

TBEKernelPtr TbeTaskBuilder::GetTbeKernel(const OpDescPtr &op_desc) const {
  return op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
}

void TbeTaskBuilder::GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const {
  (void)AttrUtils::GetStr(op_desc, op_desc->GetName() + "_kernelname", "_kernelname", kernel_name);
}

Status TbeTaskBuilder::DoRegisterBinary(const OpKernelBin &kernel_bin, void **const bin_handle,
                                        const SingleOpModelParam &param) const {
  rtDevBinary_t binary;
  binary.version = 0U;
  binary.data = kernel_bin.GetBinData();
  binary.length = kernel_bin.GetBinDataSize();
  GE_CHK_STATUS_RET_NOLOG(GetMagic(binary.magic));
  rtError_t ret = RT_ERROR_NONE;
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    ret = rtRegisterAllKernel(&binary, bin_handle);
  } else {
    ret = rtDevBinaryRegister(&binary, bin_handle);
  }
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[DoRegister][Binary] failed, bin key = %s, core_type = %" PRId64 ", rt ret = %d", stub_name_.c_str(),
        param.core_type, static_cast<int32_t>(ret));
    REPORT_INNER_ERR_MSG("E19999", "DoRegisterBinary failed, bin key = %s, core_type = %" PRId64 ", rt ret = %d",
        stub_name_.c_str(), param.core_type, static_cast<int32_t>(ret));
    return FAILED;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterMeta(void *const bin_handle) const {
  std::string meta_data;
  const std::string* meta_data_ptr = AttrUtils::GetStr(op_desc_, GetKeyForTvmMetaData());
  if (meta_data_ptr != nullptr) {
    meta_data = *meta_data_ptr;
  }
  GELOGI("TBE: meta data: %s", meta_data.c_str());
  if (!meta_data.empty()) {
    const auto rt_ret = rtMetadataRegister(bin_handle, meta_data.c_str());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[Invoke][rtMetadataRegister] failed. bin key = %s, meta_data = %s, rt ret = %d",
          stub_name_.c_str(), meta_data.c_str(), rt_ret);
      REPORT_INNER_ERR_MSG("E19999", "rtMetadataRegister failed, bin key = %s, meta_data = %s, rt ret = %d",
          stub_name_.c_str(), meta_data.c_str(), rt_ret);
      return FAILED;
    }
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterFunction(void *const bin_handle, const char_t *const stub_name,
                                          const char_t *const kernel_name) {
  const auto rt_ret = rtFunctionRegister(bin_handle, stub_name, stub_name, kernel_name, FUNC_MODE_NORMAL);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "[Invoke][rtFunctionRegister] failed. bin key = %s, kernel name = %s, rt ret = %d",
        stub_name, kernel_name, static_cast<int32_t>(rt_ret));
    REPORT_INNER_ERR_MSG("E19999", "rtFunctionRegister failed. bin key = %s, kernel name = %s, rt ret = %d",
        stub_name, kernel_name, static_cast<int32_t>(rt_ret));
    return static_cast<uint32_t>(rt_ret);
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterKernel(const ge::OpKernelBin &tbe_kernel, const char_t *const bin_file_key,
                                        void **const bin_handle, const SingleOpModelParam &param) {
  void *handle = nullptr;
  auto ret = DoRegisterBinary(tbe_kernel, &handle, param);
  if (ret != SUCCESS) {
    return ret;
  }
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    *bin_handle = handle;
    return SUCCESS;
  }

  ret = DoRegisterMeta(handle);
  if (ret != SUCCESS) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle));
    return ret;
  }

  std::string kernel_name;
  GetKernelName(op_desc_, kernel_name);
  ret = DoRegisterFunction(handle, bin_file_key, kernel_name.c_str());
  if (ret != SUCCESS) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle));
    return ret;
  }

  GELOGI("Register function succeeded: kernel_name = %s", kernel_name.c_str());
  *bin_handle = handle;
  return SUCCESS;
}

Status TbeTaskBuilder::RegisterKernel(TbeOpTask &task, const SingleOpModelParam &param) {
  KernelBinRegistry &registry = KernelBinRegistry::GetInstance();
  // check if already registered
  const char_t *stub_func = registry.GetStubFunc(stub_name_);
  if (stub_func != nullptr) {
    task.SetStubFunc(stub_name_, stub_func);
    return SUCCESS;
  }

  // to avoid repeat register
  const std::lock_guard<std::mutex> lock(g_reg_mutex);
  // check again
  stub_func = registry.GetStubFunc(stub_name_);
  if (stub_func == nullptr) {
    stub_func = registry.GetUnique(stub_name_);
    GELOGI("RegisterKernel begin, stub_func = %s", stub_func);
    const auto tbe_kernel = GetTbeKernel(op_desc_);
    if (tbe_kernel == nullptr) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TbeKernel] fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
          op_desc_->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "GetTbeKernel fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
          op_desc_->GetName().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }

    auto holder = MakeUnique<KernelHolder>(stub_func, tbe_kernel);
    if (holder == nullptr) {
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][KernelHodler] failed.");
      REPORT_INNER_ERR_MSG("E19999", "Create KernelHodler failed.");
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }

    void *bin_handle = nullptr;
    const auto ret = DoRegisterKernel(*tbe_kernel, stub_func, &bin_handle, param);
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Register][Kernel] failed. stub name = %s", stub_name_.c_str());
      REPORT_INNER_ERR_MSG("E19999", "DoRegisterKernel failed, stub name = %s", stub_name_.c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    holder->SetBinHandle(bin_handle);
    if (!registry.AddKernel(stub_name_, std::move(holder))) {
      // should not happen. only one thread can reach here
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Add][Kernel] failed. stub name = %s", stub_name_.c_str());
      REPORT_INNER_ERR_MSG("E19999", "AddKernel failed. stub name = %s", stub_name_.c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
  }

  task.SetStubFunc(stub_name_, stub_func);
  return SUCCESS;
}

Status TbeTaskBuilder::RegisterKernelWithHandle(const SingleOpModelParam &param) {
  GELOGD("RegisterKernelWithHandle begin.");
  HandleRegistry &registry = HandleRegistry::GetInstance();
  const auto tbe_kernel = GetTbeKernel(op_desc_);
  if (tbe_kernel == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TbeKernel] fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
        op_desc_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "GetTbeKernel fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
        op_desc_->GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  void *bin_handle = nullptr;
  const auto ret = DoRegisterKernel(*tbe_kernel, nullptr, &bin_handle, param);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Register][Kernel] failed. node name = %s", op_desc_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "DoRegisterKernel failed, node name = %s", op_desc_->GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  handle_ = bin_handle;
  auto holder = MakeUnique<HandleHolder>(handle_);
  if (holder == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][HandleHolder] failed.");
    REPORT_INNER_ERR_MSG("E19999", "Create HandleHolder failed.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  if (!registry.AddHandle(std::move(holder))) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Add][Handle] failed. node name = %s", op_desc_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "AddHandle failed, node name = %s", op_desc_->GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::InitKernelArgs(void *const args_addr, const size_t arg_size, const SingleOpModelParam &param) {
  // copy args
  std::vector<void *> tensor_device_addr_vec = BuildTaskUtils::GetKernelArgs(op_desc_, param);
  if (!tensor_device_addr_vec.empty()) {
    void *const src_addr = reinterpret_cast<void *>(tensor_device_addr_vec.data());
    const size_t src_len = sizeof(void *) * tensor_device_addr_vec.size();
    GE_CHK_RT_RET(rtMemcpy(args_addr, arg_size, src_addr, src_len, RT_MEMCPY_HOST_TO_HOST));
  }
  return SUCCESS;
}

Status TbeTaskBuilder::UpdateTilingArgs(TbeOpTask &task, const size_t index, const size_t tiling_arg_index) const {
  if (task.need_tiling_) {
    GE_CHECK_GE(tiling_arg_index, (index * sizeof(void *)));
    REQUIRE_COMPAT_UINT32(tiling_arg_index);
    REQUIRE_COMPAT_UINT32(tiling_arg_index - (index * sizeof(void *)));
    task.args_ex_.tilingAddrOffset = static_cast<uint32_t>(tiling_arg_index - (index * sizeof(void *)));
    task.args_ex_.tilingDataOffset = static_cast<uint32_t>(tiling_arg_index);
    task.args_ex_.hasTiling = true;
    task.tiling_data_idx_ = static_cast<uint32_t>(tiling_arg_index / sizeof(void *));
  }
  return SUCCESS;
}

Status TbeTaskBuilder::SetKernelArgs(TbeOpTask &task, const SingleOpModelParam &param, const OpDescPtr &op_desc) {
  bool is_dynamic = false;
  (void)AttrUtils::GetBool(op_desc_, kSupportDynamicShape, is_dynamic);
  if (is_dynamic) {
    GE_CHK_STATUS_RET_NOLOG(InitTilingInfo(task));
  }

  const auto task_type = static_cast<ModelTaskType>(task_def_.type());
  const bool is_task_all_kernel = (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL);
  size_t arg_size = 0U;
  size_t kernel_def_arg_size = 0U;
  // distance of tiling_addr to the end
  task.has_overflow_attr_ = task.has_overflow_attr_ && (task.overflow_addr_ != nullptr);
  const size_t default_addr_index = (task.has_overflow_attr_ ? 2UL : 1UL);

  std::unique_ptr<uint8_t[]> args = nullptr;
  const void *kernel_def_args = nullptr;
  if (is_task_all_kernel) {
    GELOGD("SetKernelArgs of %s in branch of ModelTaskType::MODEL_TASK_ALL_KERNEL.", op_desc->GetName().c_str());
    kernel_def_arg_size = kernel_def_with_handle_.args_size();
    GE_CHECK_GE(kernel_def_with_handle_.args().size(), kernel_def_arg_size);
    kernel_def_args = kernel_def_with_handle_.args().data();
  } else {
    GELOGD("SetKernelArgs of %s in branch of ModelTaskType::MODEL_TASK_KERNEL.", op_desc->GetName().c_str());
    kernel_def_arg_size = kernel_def_.args_size();
    GE_CHECK_GE(kernel_def_.args().size(), kernel_def_arg_size);
    kernel_def_args = kernel_def_.args().data();
  }

  const size_t len = (task.extend_args_for_host_input_ ? kMaxHostMemInputLen : 0U);
  arg_size = kernel_def_arg_size + task.max_tiling_size_ + len;
  REQUIRE_COMPAT_UINT16(arg_size);
  args = MakeUnique<uint8_t[]>(arg_size);
  GE_CHECK_NOTNULL(args);
  GE_CHK_RT_RET(rtMemcpy(args.get(), arg_size, kernel_def_args, kernel_def_arg_size,
                         RT_MEMCPY_HOST_TO_HOST));
  if (task.has_overflow_attr_) {
    GE_CHECK_GE(kernel_def_arg_size, sizeof(void *));
    const size_t argsize_idx_with_overflow = kernel_def_arg_size - sizeof(void *);
    GE_CHK_RT_RET(rtMemcpy(args.get() + argsize_idx_with_overflow, sizeof(void *), &(task.overflow_addr_),
                           sizeof(void *), RT_MEMCPY_HOST_TO_HOST));
  }
  const domi::KernelContext &context = (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) ?
                                       kernel_def_with_handle_.context() : kernel_def_.context();
  const auto *const args_offset_tmp = PtrToPtr<const char_t, const uint16_t>(context.args_offset().data());
  uint16_t offset = *args_offset_tmp;
  GE_CHECK_GE(arg_size, offset + task.ffts_addr_num_ * sizeof(uint64_t));
  // add ffts_addr after offset
  if (task.ffts_addr_num_ == 1UL) {
    uint64_t mode_addr = 0U;
    uint32_t model_len = 0U;
    GE_CHK_RT_RET(rtGetC2cCtrlAddr(&mode_addr, &model_len));
    GE_CHK_RT_RET(
        rtMemcpy(args.get() + offset, sizeof(uint64_t), &mode_addr, sizeof(uint64_t), RT_MEMCPY_HOST_TO_HOST));
    offset += sizeof(uint64_t);
  }

  GE_CHK_STATUS_RET_NOLOG(InitKernelArgs(args.get() + static_cast<int32_t>(offset), arg_size - offset, param));

  if (is_task_all_kernel) {
    task.SetKernelWithHandleArgs(std::move(args), arg_size, kernel_def_with_handle_.block_dim(), op_desc,
                                 kernel_def_with_handle_);
  } else {
    task.SetKernelArgs(std::move(args), arg_size, kernel_def_.block_dim(), op_desc, kernel_def_);
  }

  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = static_cast<uint32_t>(arg_size);
  GE_CHK_STATUS_RET_NOLOG(UpdateTilingArgs(task, default_addr_index, kernel_def_arg_size));
  if (task.extend_args_for_host_input_) {
    task.args_item_offsets_.host_input_data_offset = kernel_def_arg_size;
  }

  task.run_info_ = MakeUnique<optiling::utils::OpRunInfo>(0, true, 0);
  GE_CHECK_NOTNULL(task.run_info_);
  return SUCCESS;
}

Status TbeTaskBuilder::BuildTask(TbeOpTask &task, const SingleOpModelParam &param) {
  GELOGD("Build tbe task begin.");
  task.has_overflow_attr_ = AttrUtils::HasAttr(op_desc_, GLOBALWORKSPACE_TYPE);
  task.input_num_ = op_desc_->GetInputsSize();
  task.output_num_ = op_desc_->GetOutputsSize();
  task.SetOpDesc(op_desc_);
  auto ret = SetKernelArgs(task, param, op_desc_);
  if (ret != SUCCESS) {
    return ret;
  }

  const auto task_type = static_cast<ModelTaskType>(task_def_.type());
  if (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    stub_name_ = model_name_ + "/" + node_->GetName() + "_tvmbin";
    ret = RegisterKernelWithHandle(param);
  } else {
    const domi::KernelDef &kernel_def = task_def_.kernel();
    stub_name_ = model_name_ + "/" + kernel_def.stub_func() + "_tvmbin";
    ret = RegisterKernel(task, param);
  }

  task.SetHandle(handle_);
  if (ret != SUCCESS) {
    return ret;
  }

  const auto task_info = BuildTaskUtils::GetTaskInfo(op_desc_);
  GELOGI("[TASK_INFO] %s %s", stub_name_.c_str(), task_info.c_str());

  if (task_type != ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    void *stub_func = nullptr;
    const auto rt_ret = rtGetFunctionByName(stub_name_.c_str(), &stub_func);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(FAILED, "[Get][FunctionByName] failed. stub_name:%s.", stub_name_.c_str());
      REPORT_INNER_ERR_MSG("E19999", "rtGetFunctionByName failed, stub_name:%s.", stub_name_.c_str());
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    task.SetStubFunc(stub_name_, stub_func);
  }
  GE_CHK_STATUS_RET(task.SetArgIndex(), "[Set][ArgTable] failed.");

  return SUCCESS;
}

Status TbeTaskBuilder::InitTilingInfo(TbeOpTask &task) {
  GELOGD("Start alloc tiling data of node %s.", op_desc_->GetName().c_str());
  int64_t max_size = -1;
  (void)AttrUtils::GetInt(op_desc_, GetKeyForOpParamSize(), max_size);
  GELOGD("Got op param size by key: %s, ret = %" PRId64, GetKeyForOpParamSize().c_str(), max_size);
  if (max_size < 0) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Get][Int] %s Invalid op_param_size: %" PRId64 ".",
        op_desc_->GetName().c_str(), max_size);
    REPORT_INNER_ERR_MSG("E19999", "AttrUtils::GetInt failed, %s Invalid op_param_size: %" PRId64 ".",
        op_desc_->GetName().c_str(), max_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_CHECK_LE(max_size, static_cast<int64_t>(UINT32_MAX));
  task.EnableDynamicSupport(node_, static_cast<uint32_t>(max_size));
  return SUCCESS;
}

Status TbeTaskBuilder::GetMagic(uint32_t &magic) const {
  std::string json_string;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_, TVM_ATTR_NAME_MAGIC, json_string),
                  GELOGD("Get original type of session_graph_id."));
  if (json_string == "RT_DEV_BINARY_MAGIC_ELF") {
    magic = RT_DEV_BINARY_MAGIC_ELF;
  } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
    magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s in op:%s(%s), value:%s check invalid",
                       TVM_ATTR_NAME_MAGIC.c_str(), op_desc_->GetName().c_str(),
                       op_desc_->GetType().c_str(), json_string.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in op:%s(%s), value:%s check invalid",
           TVM_ATTR_NAME_MAGIC.c_str(), op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str(), json_string.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

std::string TbeTaskBuilder::GetKeyForOpParamSize() const {
  return kOpParamSize;
}

std::string TbeTaskBuilder::GetKeyForTvmMetaData() const {
  return TVM_ATTR_NAME_METADATA;
}

Status AtomicAddrCleanTaskBuilder::InitKernelArgs(void *const args_addr, const size_t arg_size,
                                                  const SingleOpModelParam &param) {
  (void)args_addr;
  (void)arg_size;
  (void)param;
  return SUCCESS;
}

std::string AtomicAddrCleanTaskBuilder::GetKeyForOpParamSize() const {
  return kAtomicOpParamSize;
}

std::string AtomicAddrCleanTaskBuilder::GetKeyForTvmMetaData() const {
  return ATOMIC_ATTR_TVM_METADATA;
}

void AtomicAddrCleanTaskBuilder::GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const {
  (void)AttrUtils::GetStr(op_desc, op_desc->GetName() + "_atomic_kernelname", "_atomic_kernelname", kernel_name);
}

TBEKernelPtr AtomicAddrCleanTaskBuilder::GetTbeKernel(const OpDescPtr &op_desc)  const {
  return op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, TBEKernelPtr());
}

Status MixL2TaskBuilder::BuildMixL2Task(MixL2OpTask &task, SingleOpModelParam &param) {
  // init args
  task.input_num_ = op_desc_->GetInputsSize();
  task.output_num_ = op_desc_->GetOutputsSize();
  task.SetOpDesc(op_desc_);

  const auto ctx_num = ffts_plus_task_def_.ffts_plus_ctx_size();
  for (int32_t i = 0; i < ctx_num; ++i) {
    const auto &ctx_def = ffts_plus_task_def_.ffts_plus_ctx(i);
    const auto type = static_cast<tagFftsPlusContextType>(ctx_def.context_type());
    if (type == RT_CTX_TYPE_MIX_AIC || type == RT_CTX_TYPE_MIX_AIV) {
      task.ctx_type_ = type;
      const domi::FftsPlusMixAicAivCtxDef &aic_aiv_ctx_def = ctx_def.mix_aic_aiv_ctx();
      const uint32_t mix_non_tail_block_dim = aic_aiv_ctx_def.non_tail_block_dim();
      const uint32_t mix_tail_block_ratio = aic_aiv_ctx_def.non_tail_block_ratio_n();
      // 针对mix算子，低16位为主加速器blockdim，高16位为从加速器的ratio值，由工具解析
      task.block_dim_ = ((mix_non_tail_block_dim & 0xFFFFU) | (mix_tail_block_ratio << 16U));
      GELOGI("Op %s get context type %u by ctx idx %d, blockdim %u, master blockdim %u, slave ratio %u.",
             op_desc_->GetName().c_str(), type, i, task.block_dim_, mix_non_tail_block_dim, mix_tail_block_ratio);
      break;
    }
  }

  for (int32_t i = 0; i < ctx_num; ++i) {
    const auto &ctx_def = ffts_plus_task_def_.ffts_plus_ctx(i);
    const auto context_id = ctx_def.context_id();
    task.context_ids_.push_back(context_id);
  }

  bool is_dynamic = false;
  (void)AttrUtils::GetBool(op_desc_, kSupportDynamicShape, is_dynamic);
  if (is_dynamic) {
    GE_CHK_STATUS_RET(InitTilingInfo(task));
  }

  const size_t kernel_def_arg_size = sizeof(void *) * static_cast<size_t>(ffts_plus_task_def_.addr_size());
  const size_t len = task.extend_args_for_host_input_ ? kMaxHostMemInputLen : 0U;
  task.arg_size_ = kernel_def_arg_size + task.max_tiling_size_ + len;
  if (task.arg_size_ > 0UL) {
    task.host_args_.resize(task.arg_size_ / sizeof(uintptr_t));
    GE_CHK_RT_RET(rtMalloc(&task.device_args_, task.arg_size_, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  }

  // Init Mode addr
  const size_t addr_base = (task.max_tiling_size_ + len) / sizeof(uintptr_t);

  // Register Kernel and do transfer
  GE_CHK_STATUS_RET(InitMixKernelArgs(task, addr_base, param), "Init mix kernel args failed.");

  // Init mode addr after transfer
  for (const auto &addr_idx : task.mode_addr_idx_) {
    if (addr_idx >= task.io_addrs_from_taskdef_.size()) {
      GELOGE(FAILED, "Index [%zu] greater than [%zu] is invalid.", addr_idx, task.io_addrs_from_taskdef_.size());
      return FAILED;
    }
    task.host_args_[addr_base + task.mode_addr_cnt_] = static_cast<uintptr_t>(task.io_addrs_from_taskdef_[addr_idx]);
    task.mode_addr_cnt_++;
  }
  task.args_addr_base_idx_ = addr_base + task.mode_addr_cnt_;
  const size_t addr_len = kernel_def_arg_size - task.mode_addr_cnt_ * sizeof(uintptr_t);
  task.args_addr_cnt_ = addr_len / sizeof(uintptr_t);
  GELOGD("Node: %s, input num: %zu, output num: %zu, workspace num: %zu, is_dynamic: %d, kernel_def_arg_size: %zu, "
         "len: %zu, max tiling size: %u, args size: %zu, host args size: %zu, addr_base: %zu, mode_addr_cnt: %zu, "
         "args_addr_base_idx: %zu, addr_len: %zu, args_addr_cnt: %zu, tiling data size: %zu.",
         op_desc_->GetName().c_str(), task.input_num_, task.output_num_, op_desc_->GetWorkspaceBytes().size(),
         static_cast<int32_t>(is_dynamic), kernel_def_arg_size, len, task.max_tiling_size_, task.arg_size_,
         task.host_args_.size(), addr_base, task.mode_addr_cnt_, task.args_addr_base_idx_, addr_len,
         task.args_addr_cnt_, tiling_data_size_);

  // Init IO/workspace addr
  GE_CHK_STATUS_RET(InitKernelArgs(&task.host_args_[task.args_addr_base_idx_], addr_len, param));
  GE_CHK_STATUS_RET(InitTilingDataAddrToArgs(task), "Init tiling data addr to args failed.");
  GE_CHK_STATUS_RET(task.SetArgIndex(), "Set argtable failed.");

  return SUCCESS;
}

Status MixL2TaskBuilder::InitTilingDataAddrToArgs(MixL2OpTask &task) const {
  if (tiling_data_size_ == 0U) {
    GELOGD("No need to init tiling data addr of %s.", op_desc_->GetName().c_str());
    return SUCCESS;
  }

  const size_t tiling_data_idx = op_desc_->GetAllInputsDescPtr().size() + op_desc_->GetWorkspaceBytes().size() +
                                 static_cast<size_t>(op_desc_->GetAllOutputsDescSize());
  GE_CHECK_GE(task.host_args_.size(), (task.args_addr_base_idx_ + tiling_data_idx + 1U));
  GE_CHK_RT_RET(rtMemcpy(&task.host_args_[task.args_addr_base_idx_ + tiling_data_idx], sizeof(uintptr_t),
                         &tiling_data_addr_, sizeof(uintptr_t), RT_MEMCPY_HOST_TO_HOST));
  GELOGI("Init tiling data addr of %s, tiling_data_idx: %zu.", op_desc_->GetName().c_str(), tiling_data_idx);
  return SUCCESS;
}

Status MixL2TaskBuilder::InitMixKernelArgs(MixL2OpTask &task, const size_t addr_base_offset,
                                           SingleOpModelParam &param) {
  // Register kernel
  std::string core_type;
  const std::string* core_type_ptr = AttrUtils::GetStr(op_desc_, ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  if (core_type_ptr != nullptr) {
    core_type = *core_type_ptr;
  }
  if (kTbeCoreTypeMix.count(core_type) > 0U) {
    (void)AttrUtils::GetListStr(op_desc_, ATTR_NAME_KERNEL_NAMES_PREFIX, task.names_prefix_);
    for (const auto &prefix : task.names_prefix_) {
      GE_CHK_STATUS_RET(task.bin_kernel_handle_.Register(op_desc_, prefix));
    }
  }

  FftsPlusArgsHelper helper(param.runtime_param);
  const uintptr_t args_base = static_cast<uintptr_t>(PtrToValue(task.device_args_) +
                                                     static_cast<uint64_t>(addr_base_offset * sizeof(uintptr_t)));
  FftsPlusProtoTransfer transfer(args_base, &helper, param.runtime_param, task.ext_args_);
  // Add handle
  transfer.SetFindNodeHandle([this](const uint32_t idx_object) -> OpDescPtr {
    (void) idx_object;
    return op_desc_;
  });
  GE_CHK_STATUS_RET(HandleSoftSyncOp(task, param), "Handle soft sync op %s failed.", task.op_desc_->GetName().c_str());
  if (tiling_data_size_ > 0U) {
    static const std::string kPurpose("malloc tiling data memory for soft sync op.");
    GE_CHECK_NOTNULL(task.stream_resource_);
    tiling_data_addr_ = task.stream_resource_->MallocMemory(kPurpose, tiling_data_size_ + kBufAlignedBytes);
    GE_CHECK_NOTNULL(tiling_data_addr_);
    helper.SetTilingDataLen(tiling_data_size_);
    helper.SetTilingDataDev(tiling_data_addr_);
  }

  transfer.SetAddrPrefHandle([&task](const OpDescPtr &op_desc, const std::string &kernel_name,
                                     const std::string &prefix,
                                     std::vector<std::pair<void *, uint32_t>> &addr_and_pref_cnt) -> Status {
    return task.bin_kernel_handle_.GetAddrAndPrefCnt(op_desc, kernel_name, prefix, addr_and_pref_cnt);
  });

  transfer.SetSaveL0DumpInfoHandle([&task](const std::vector<uint64_t> &l0_dump_list) {
      task.l0_dump_list_.insert(task.l0_dump_list_.end(), l0_dump_list.begin(), l0_dump_list.end());
  });

  GE_CHK_STATUS_RET(transfer.Transfer(op_desc_, ffts_plus_task_def_, task.ffts_plus_task_info_),
                    "Do transfer filed.");
  task.io_addrs_from_taskdef_ = helper.GetIoAddr();
  task.mode_addr_idx_ = helper.GetModeAddrIdx();
  GE_ASSERT_SUCCESS(helper.AssembleTilingData());
  return SUCCESS;
}

Status MixL2TaskBuilder::HandleSoftSyncOp(MixL2OpTask &task, SingleOpModelParam &param) {
  tiling_data_size_ = 0U;
  GE_CHECK_NOTNULL(task.op_desc_);
  GE_CHECK_NOTNULL(task.op_);
  bool is_soft_sync_op = false;
  if ((!AttrUtils::GetBool(task.op_desc_, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync_op)) ||
      (!is_soft_sync_op)) {
    return SUCCESS;
  }

  const auto run_info = MakeShared<optiling::utils::OpRunInfo>(0, false, 0);
  GE_CHECK_NOTNULL(run_info);
  GE_ASSERT_TRUE(static_cast<size_t>(task.op_desc_->GetOppImplVersion()) < param.space_registries_->size());
  GE_CHK_STATUS_RET(
      optiling::SoftSyncOpRtParseAndTiling(
          *task.op_, param.platform_infos, *run_info,
          param.space_registries_->at(static_cast<size_t>(task.op_desc_->GetOppImplVersion()))),
                    "Recall tiling for soft sync op: %s failed.", op_desc_->GetName().c_str());
  if (task.op_desc_->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info)) {
    GELOGI("Success to set extra attr: %s to %s.", ATTR_NAME_OP_RUN_INFO.c_str(), task.op_desc_->GetName().c_str());
  }

  tiling_data_size_ = run_info->GetAllTilingData().str().size();
  GELOGI("Node: %s, tiling data size: %zu.", op_desc_->GetName().c_str(), tiling_data_size_);

  return SUCCESS;
}
}  // namespace ge
