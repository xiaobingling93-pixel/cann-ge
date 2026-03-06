/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/fusion_task_info.h"

#include <securec.h>
#include "common/checker.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "framework/common/types.h"
#include "graph/load/model_manager/memory_app_type_classifier.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/task_info/hccl/hccl_util.h"
#include "graph/manager/graph_var_manager.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "common/dump/dump_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "runtime/kernel.h"
#include "common/op_tiling/tiling_memcheck.h"
#include "common/op_tiling/tiling_dfx.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "register/op_tiling_registry.h"
#include "adump_api.h"
#include "register/op_tiling/op_tiling_constants.h"

namespace {
const std::string kLocalMemorySize = "local_memory_size";
constexpr int64_t kDefaultDimInfo = 0x100000001;
constexpr uint64_t kDefaultShapeNum = 0x100000000U;
constexpr uint32_t kAddressLen = static_cast<uint32_t>(sizeof(uint64_t));
constexpr char_t const *kMaxTilingSize = "op_para_size";
constexpr uint64_t kMaxTilingDataSize = 16UL * 1024UL;

constexpr uint64_t kBitFlag8 = 0x00FFFFFFFFFFFFFFUL;
constexpr uint64_t kLevel2BitFlagWithShape = 0x0200000000000000UL;

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

uint32_t DomiFusionTypeToRtFusionType(const domi::FusionSubTaskInfo::FusionType fusion_type) {
  uint32_t rt_fusion_type = static_cast<uint32_t>(RT_FUSION_END);
  static const std::map<domi::FusionSubTaskInfo::FusionType, rtFusionType_t> domi_type_to_rt_type = {
      {domi::FusionSubTaskInfo::HCOM_CPU, RT_FUSION_HCOM_CPU},
      {domi::FusionSubTaskInfo::AICPU, RT_FUSION_AICPU},
      {domi::FusionSubTaskInfo::AICORE, RT_FUSION_AICORE},
      {domi::FusionSubTaskInfo::CCU, RT_FUSION_CCU}};
  const auto iter = domi_type_to_rt_type.find(fusion_type);
  if (iter != domi_type_to_rt_type.end()) {
    rt_fusion_type = static_cast<uint32_t>(iter->second);
  }
  GELOGD("domi type:[%u], rt_type:[%u]", static_cast<uint32_t>(fusion_type), rt_fusion_type);
  return rt_fusion_type;
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
  GE_ASSERT_TRUE(!arg_descs.empty());
  std::vector<int64_t> args_size_vec;
  std::vector<optiling::ArgsIndexToIoIndex> args_idx_to_io_idx_vec;
  GE_ASSERT_SUCCESS(
    optiling::TilingDfx::GetArgsSizeWithArgsFormat(op_desc, arg_descs, args_size_vec, args_idx_to_io_idx_vec));

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

namespace ge {
size_t FusionTaskInfo::GetArgsSizeByFormat(ArgsFormatInfo &args_format_info) const {
  const auto &arg_descs = args_format_info.arg_descs;
  size_t tmp_size = 0U;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(op_desc_, arg_desc, tmp_size);
  }
  return tmp_size;
}

Status FusionTaskInfo::ParseArgsFormat(ArgsFormatInfo &args_format_info) {
  (void)OpDescUtils::GetIrInputInstanceDescRange(op_desc_, args_format_info.ir_input_2_range);
  (void)OpDescUtils::GetIrOutputDescRange(op_desc_, args_format_info.ir_output_2_range);
  auto &arg_descs = args_format_info.arg_descs;
  auto input_descs = op_desc_->GetAllInputsDescPtr();
  // tiling下沉不支持
  for (const auto &arg_format : arg_descs) {
    if (arg_format.addr_type == AddrType::INPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_info.ir_input_2_range.size());
      const auto &ir_range = args_format_info.ir_input_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        const size_t instance_idx = static_cast<size_t>(ir_range.first + idx);
        GE_ASSERT_TRUE(instance_idx < input_descs.size(), "Instance index [%zu] is out of range, max_size:[%zu].",
                       instance_idx, input_descs.size());
        AppendShapeDesc(*input_descs.at(instance_idx), shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      args_format_info.level1_addr_cnt += ir_range.second + shape_info.size();
      args_format_info.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::OUTPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_info.ir_output_2_range.size());
      const auto &ir_range = args_format_info.ir_output_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      args_format_info.level1_addr_cnt += ir_range.second;
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        auto output_desc = op_desc_->MutableOutputDesc(static_cast<uint32_t>(ir_range.first + idx));
        GE_ASSERT_NOTNULL(output_desc);
        AppendShapeDesc(*output_desc, shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size() * sizeof(uintptr_t));
      args_format_info.level1_addr_cnt += ir_range.second + shape_info.size();
      args_format_info.shape_infos.push_back(shape_info);
    } else {
      // misra
    }
  }
  return SUCCESS;
}

Status FusionTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                         TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  task_type_ = static_cast<ModelTaskType>(task_def.type());

  GE_ASSERT_TRUE(static_cast<uint32_t>(task_def.fusion_task().fusion_sub_task_info_size()) <= FUSION_SUB_TASK_MAX_NUM);
  op_desc_ = davinci_model->GetOpByIndex(task_def.fusion_task().op_index());
  GE_CHECK_NOTNULL(op_desc_);

  GE_ASSERT_TRUE(!task_def.fusion_task().args_format().empty());
  GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(op_desc_, task_def.fusion_task().args_format(), args_format_info_.arg_descs),
                    "Aicore formatted args [%s] parsed failed.", task_def.fusion_task().args_format().c_str());
  GE_ASSERT_SUCCESS(ParseArgsFormat(args_format_info_));
  args_size_ = GetArgsSizeByFormat(args_format_info_);

  for (const domi::FusionSubTaskInfo &fusion_sub_task_info : task_def.fusion_task().fusion_sub_task_info()) {
    const domi::FusionSubTaskDef &task = fusion_sub_task_info.task();
    switch (fusion_sub_task_info.type()) {
      case domi::FusionSubTaskInfo::AICORE: {
        GELOGI("Fusion task support sub aicore task kernel.");
        const auto &aicore_task = task.aicore_fusion_task_info();
        is_all_kernel_ = aicore_task.is_all_kernel();
        domi::KernelContext context = aicore_task.context();
        ccKernelType kernel_type = static_cast<ccKernelType>(context.kernel_type());
        GE_ASSERT_TRUE(ccKernelType::MIX_AICORE == kernel_type);
        break;
      }
      case domi::FusionSubTaskInfo::AICPU: {
        GELOGW("Fusion task no support sub aicpu kernel. ");
        break;
      }
      case domi::FusionSubTaskInfo::CCU: {
        GELOGI("Fusion task support sub ccu task kernel.");
        break;
      }
      default:
        break;
    }
  }

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  input_data_addrs_ = ModelUtils::GetInputAddrsValue(rts_param, op_desc_, input_mem_types_);
  output_data_addrs_ = ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_, true);
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

  task_run_param.args_descs.push_back({static_cast<int64_t>(MemSizeAlign(static_cast<size_t>(args_size_),
                                                                         static_cast<uint32_t>(sizeof(uintptr_t)))),
                                       args_placement_});

  GELOGI("Get args size[%u] of op[%s], is fm refresh[%d], task_type: %d, placement: %d, "
         "args format: %s, is all kernel: %d.", args_size_, op_desc_->GetName().c_str(),
         static_cast<int32_t>(davinci_model->IsFeatureBaseRefreshable()), static_cast<int32_t>(task_type_),
         args_placement_, task_def.fusion_task().args_format().c_str(), is_all_kernel_);

  return SUCCESS;
}

Status FusionTaskInfo::AppendInputOutputAddr(size_t ir_idx, bool is_input, const ArgsFormatInfo &args_format_info) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_2_range =
      is_input ? args_format_info.ir_input_2_range : args_format_info.ir_output_2_range;
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
    GE_ASSERT(begin_idx < addrs.size(), "ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx,
              begin_idx, addrs.size());
    l0_dump_list_.push_back(begin_idx + cust_offset);
    cust_to_relevant_offset_[begin_idx + cust_offset] = io_addrs_.size();
    AppendIoAddr(addrs[begin_idx], types[begin_idx]);
  }
  return SUCCESS;
}

Status FusionTaskInfo::AppendInputOutputAddrByInstanceIndex(size_t ins_idx, bool is_input) {
  if (is_input) {
    GE_ASSERT_TRUE(ins_idx < input_data_addrs_.size(), "Instance idx [%zu] is invalid, input_size:[%zu]", ins_idx,
                   input_data_addrs_.size());
    l0_dump_list_.push_back(ins_idx);
    cust_to_relevant_offset_[ins_idx] = io_addrs_.size();
    AppendIoAddr(input_data_addrs_[ins_idx], input_mem_types_[ins_idx]);
  } else {
    GE_ASSERT_TRUE(ins_idx < output_data_addrs_.size(), "Instance idx [%zu] is invalid, output_size:[%zu]", ins_idx,
                   output_data_addrs_.size());
    l0_dump_list_.push_back(input_data_addrs_.size() + ins_idx);
    cust_to_relevant_offset_[input_data_addrs_.size() + ins_idx] = io_addrs_.size();
    AppendIoAddr(output_data_addrs_[ins_idx], output_mem_types_[ins_idx]);
  }
  return SUCCESS;
}

Status FusionTaskInfo::AppendWorkspaceAddr(int32_t ir_idx) {
  const size_t input_output_size = input_data_addrs_.size() + output_data_addrs_.size();
  if (ir_idx < 0) {
    for (size_t i = 0UL; i < workspace_addrs_.size(); ++i) {
      l0_dump_list_.push_back(input_output_size + i);
    }

    (void)io_addrs_.insert(io_addrs_.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), workspace_mem_types_.cbegin(),
                                    workspace_mem_types_.cend());
  } else {
    const size_t idx = static_cast<size_t>(ir_idx);
    GE_ASSERT(idx < workspace_addrs_.size(), "workspace index[%zu] is output of workspace addrs range[%zu]", idx,
              workspace_addrs_.size());
    l0_dump_list_.push_back(input_output_size + ir_idx);
    AppendIoAddr(workspace_addrs_[idx], workspace_mem_types_[idx]);
    GELOGI("op[%s], workspace_addrs_[%zu] = 0x%lx, workspace_mem_types_[%zu] = %" PRIu64 "", op_desc_->GetName().c_str(), idx,
           workspace_addrs_[idx], idx, workspace_mem_types_[idx]);
  }
  return SUCCESS;
}

void FusionTaskInfo::AppendIoAddr(const uint64_t addr, const uint64_t addr_type) {
  io_addrs_.push_back(addr);
  io_addr_mem_types_.push_back(addr_type);
}

Status FusionTaskInfo::AssembleShapeInfoAddrs(const std::vector<ArgDesc> &dynamic_args_desc,
                                              const std::vector<size_t> &level2_addr_idx,
                                              const ArgsFormatInfo &args_format_info) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_info.ir_input_2_range;
  const std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_info.ir_output_2_range;
  // append additional level1 addr
  GE_ASSERT(dynamic_args_desc.size() == args_format_info.shape_infos.size());
  for (size_t i = 0UL; i < dynamic_args_desc.size(); ++i) {
    auto &shape_info = args_format_info.shape_infos[i];
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
      const auto &range_pair = ir_input_2_range.at(ir_idx);
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
      const auto &range_pair = ir_output_2_range.at(ir_idx);
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

bool FusionTaskInfo::HasOverflowAddr(const OpDescPtr &op_desc) const {
  return (davinci_model_->GetOverflowAddr() != nullptr) && AttrUtils::HasAttr(op_desc, GLOBALWORKSPACE_TYPE);
}

Status FusionTaskInfo::AssembleIoByArgsFormat(const ArgsFormatInfo &args_format_info) {
  const auto &arg_descs = args_format_info.arg_descs;
  io_addrs_.reserve(arg_descs.size());
  io_addr_mem_types_.reserve(arg_descs.size());
  std::vector<ArgDesc> dynamic_args_desc;
  std::vector<size_t> level_addr_idx;
  std::vector<void *> context_addrs;
  const std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_info.ir_input_2_range;
  const std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_info.ir_output_2_range;
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
      case AddrType::INPUT_DESC: {
        // l0 exception dump处理
        const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
        const auto iter = ir_input_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_input_2_range.end(),
                  "node[%s] input ir idx[%zu] is not found", op_desc_->GetName().c_str(), ir_idx);
        // level2_addr
        uint64_t level_num = iter->second.second & kBitFlag8;
        level_num |= kLevel2BitFlagWithShape;
        l0_dump_list_.push_back(level_num);
        // level1
        for (size_t i = 0UL; i < iter->second.second; ++i) {
          l0_dump_list_.push_back(iter->second.first + i);;
        }

        level_addr_idx.push_back(io_addrs_.size());
        dynamic_args_desc.push_back(arg_format);
        AppendIoAddr(0UL, kAbsoluteMemType);
        break;
      }
      case AddrType::OUTPUT_DESC: {
        const size_t ir_idx = static_cast<size_t>(arg_format.ir_idx);
        const auto iter = ir_output_2_range.find(ir_idx);
        GE_ASSERT(iter != ir_output_2_range.end(),
                  "node[%s] input ir idx [%zu] is not found", op_desc_->GetName().c_str(), ir_idx);
        // level2_addr
        uint64_t level_num = iter->second.second & kBitFlag8;
        level_num |= kLevel2BitFlagWithShape;
        l0_dump_list_.push_back(level_num);
        // level1
        for (size_t i = 0UL; i < iter->second.second; ++i) {
          l0_dump_list_.push_back(input_data_addrs_.size() + iter->second.first + i);
        }

        level_addr_idx.push_back(io_addrs_.size());
        dynamic_args_desc.push_back(arg_format);
        AppendIoAddr(0UL, kAbsoluteMemType);
        break;
      }
      case AddrType::INPUT: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddr(static_cast<size_t>(arg_format.ir_idx), true, args_format_info));
        break;
      }
      case AddrType::OUTPUT: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddr(static_cast<size_t>(arg_format.ir_idx), false, args_format_info));
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
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
          AppendIoAddr(reinterpret_cast<uint64_t>(context_addrs[ir_idx]), kAbsoluteMemType);
        }
        break;
      }
      case AddrType::TILING: {
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        AppendIoAddr(PtrToValue(tiling_data_addr_), kAbsoluteMemType);
        GELOGD("Node: %s needs to reserve a tiling data addr [%p].", op_desc_->GetName().c_str(), tiling_data_addr_);
        break;
      }
      case AddrType::OVERFLOW_ADDR: {
        if (HasOverflowAddr(op_desc_)) {
          l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
          AppendIoAddr(PtrToValue(davinci_model_->GetOverflowAddr()),
                       static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
        }
        break;
      }
      case AddrType::PLACEHOLDER: {
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        AppendIoAddr(0UL, kAbsoluteMemType);
        break;
      }
      case AddrType::OP_TYPE: {
        GELOGW("Node: %s no support op_type.", op_desc_->GetName().c_str());
        break;
      }
      case AddrType::TILING_CONTEXT: {
        GELOGW("Node: %s no support tiling context.", op_desc_->GetName().c_str());
        break;
      }
      case AddrType::CUSTOM_VALUE: {
        l0_dump_list_.push_back(std::numeric_limits<uint64_t>::max());
        AppendIoAddr(*reinterpret_cast<const uint64_t *>(arg_format.reserved), kAbsoluteMemType);
        break;
      }
      case AddrType::FFTS_ADDR: {
        GELOGW("Node: %s no support ffts addr.", op_desc_->GetName().c_str());
        break;
      }
      default:
        break;
    }
  }
  GE_ASSERT_SUCCESS(AssembleShapeInfoAddrs(dynamic_args_desc, level_addr_idx, args_format_info));
  return SUCCESS;
}

Status FusionTaskInfo::SetTvmTaskZeroCopy(const OpDescPtr &op_desc, const std::vector<uint64_t> &virtual_io_addrs,
                                          void *args) const {
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
          GELOGI("Node:%s input:%" PRIu64 "is var, no need zero copy refresh.", op_desc->GetName().c_str(), args_index);
          continue;
        }
      }
      const size_t args_offset_tmp = (args_index * kAddressLen);
      (void)zero_copy_args_offset[static_cast<size_t>(virtual_io_addrs[args_index])].insert(args_offset_tmp);
      if (args_index < input_raw_data_list.size()) {
        need_raw_data_list.push_back(input_raw_data_list[args_index]);
      }
    }
    need_raw_data_list.resize(zero_copy_args_index.size(), false);
    GE_CHK_STATUS_RET(davinci_model_->Mapping2BundleZeroCopy(op_desc, zero_copy_args_offset, need_raw_data_list, 0,
                                                             nullptr, args, false, false),
                      "Failed mapping zero copy task for %s to bundle task", op_desc->GetName().c_str());
  }
  return SUCCESS;
}

Status FusionTaskInfo::SetAicoreTaskHandle(rtAicoreFusionInfo_t &aicore_fusion_info) {
  std::string kernel_handle_name = davinci_model_->GetBinHandleKey(*op_desc_, "", false);
  GE_ASSERT_TRUE(TBEHandleStore::GetInstance().FindTBEHandle(kernel_handle_name, aicore_fusion_info.hdl),
                 "Kernel bin is not found for op :[%s] with  name:[%s]", op_desc_->GetNamePtr(),
                 kernel_handle_name.c_str());
  aicore_fusion_info.tilingKey = tiling_key_;
  return SUCCESS;
}

Status FusionTaskInfo::SetAicoreTaskStubFunc(rtAicoreFusionInfo_t &aicore_fusion_info) {
  // 获取stub func + 老的零拷贝处理
  const std::string bin_handle_key = davinci_model_->GetBinHandleKey(*op_desc_, "", false);
  const rtError_t rt_ret = rtGetFunctionByName(bin_handle_key.c_str(), &stub_func_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtGetFunctionByName failed op:%s(%s), bin_file_key:%s, ret:%d",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), bin_handle_key.c_str(), rt_ret);
    GELOGE(RT_FAILED, "[Execute][RtGetFunctionByName] failed for op:%s(%s), bin_file_key:%s",
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), bin_handle_key.c_str());
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  aicore_fusion_info.stubFunc = stub_func_;
  return SUCCESS;
}

Status FusionTaskInfo::SetAIcoreLaunchAttrs(const domi::LaunchConfig &cfg, rtAicoreFusionInfo_t &aicore_fusion_info) {
  launch_attr_list_.clear();
  for (const domi::LaunchAttribute &attr : cfg.launch_attribute()) {
    switch (attr.id()) {
      case domi::LaunchAttribute::BLOCKDIM: {
        rtLaunchAttribute_t launch_attr;
        launch_attr.id = RT_LAUNCH_ATTRIBUTE_BLOCKDIM;
        launch_attr.value.blockDim = attr.value().block_dim();
        launch_attr_list_.push_back(std::move(launch_attr));
        break;
      }
      case domi::LaunchAttribute::BLOCKDIM_OFFSET: {
        rtLaunchAttribute_t launch_attr;
        launch_attr.id = RT_LAUNCH_ATTRIBUTE_BLOCKDIM_OFFSET;
        launch_attr.value.blockDimOffset = attr.value().block_dim_offset();
        launch_attr_list_.push_back(std::move(launch_attr));
        break;
      }
      case domi::LaunchAttribute::SCHEMMODE: {
        rtLaunchAttribute_t launch_attr;
        launch_attr.id = RT_LAUNCH_ATTRIBUTE_SCHEMMODE;
        launch_attr.value.schemMode = static_cast<uint8_t>(attr.value().schem_model());
        launch_attr_list_.push_back(std::move(launch_attr));
        break;
      }
      default:
        break;
    }
  }

  // dump 开关
  rtLaunchAttribute_t launch_attr;
  launch_attr.id = RT_LAUNCH_ATTRIBUTE_DUMPFLAG;
  launch_attr.value.dumpflag = static_cast<uint8_t>(dump_flag_);
  launch_attr_list_.push_back(std::move(launch_attr));

  rt_launch_config_.numAttrs = launch_attr_list_.size();
  rt_launch_config_.attrs = (rt_launch_config_.numAttrs > 0U ? launch_attr_list_.data() : nullptr);
  aicore_fusion_info.config = &rt_launch_config_;

  if (is_all_kernel_) {
    GE_ASSERT_SUCCESS(SetAicoreTaskHandle(aicore_fusion_info));
  } else {
    GE_ASSERT_SUCCESS(SetAicoreTaskStubFunc(aicore_fusion_info));
  }
  return SUCCESS;
}

Status FusionTaskInfo::CopyTilingDataIfNeeded() {
  std::shared_ptr<optiling::utils::OpRunInfo> default_tiling = nullptr;
  std::shared_ptr<optiling::utils::OpRunInfo> run_info = nullptr;
  run_info = op_desc_->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, default_tiling);
  if (run_info == nullptr || run_info->GetAllTilingData().str().empty()) {
      GELOGD("Tiling data of %s is empty.", op_desc_->GetNamePtr());
      return SUCCESS;
  }

  has_tiling_ = true;
  tiling_data_host_ = run_info->GetAllTilingData().str();
  std::string dfx_info;
  GE_CHK_STATUS_RET(ConstructDfxInfo(op_desc_, *run_info, args_format_info_.arg_descs, dfx_info),
      "Append memcheck data for node: %s failed.", op_desc_->GetNamePtr());
  tiling_data_host_ += dfx_info;

  tiling_data_size_ = tiling_data_host_.size();
  tiling_data_addr_ = davinci_model_->MallocDynamicMemory(tiling_data_size_);
  GE_CHECK_NOTNULL(tiling_data_addr_);
  GE_CHK_RT_RET(aclrtMemcpy(tiling_data_addr_, tiling_data_size_,
      tiling_data_host_.data(), tiling_data_host_.size(), ACL_MEMCPY_HOST_TO_DEVICE));

  GELOGI("Success to update tiling data to io_addr of %s, device addr: %p, size: %zu, host tiling data addr: %p",
          op_desc_->GetNamePtr(), tiling_data_addr_, tiling_data_host_.size(), tiling_data_host_.data());

  return SUCCESS;
}

Status FusionTaskInfo::InitArgs(const PisToArgs &args) {
  io_addrs_.clear();
  io_addr_mem_types_.clear();
  GE_ASSERT_SUCCESS(CopyTilingDataIfNeeded());
  GE_ASSERT_SUCCESS(AssembleIoByArgsFormat(args_format_info_), "[Assemble][Addresses] failed, op = %s.",
                    op_desc_->GetNamePtr());

  uint32_t pls = static_cast<uint32_t>(args_placement_);
  const errno_t sec_ret = memcpy_s(ValueToPtr(PtrToValue(args[pls].host_addr)), args[pls].len ,
    io_addrs_.data(), sizeof(uint64_t) * io_addrs_.size());
  GE_ASSERT_EOK(sec_ret, "[Call][Memcpy] failed, size:%zu, ret:%d", args[pls].len, sec_ret);
  return SUCCESS;
}

Status FusionTaskInfo::InitKernel(const domi::TaskDef &task_def, const PisToArgs &args) {
  uint32_t pls = static_cast<uint32_t>(args_placement_);
  GE_ASSERT_TRUE((args[pls].dev_addr != 0U), "[Check][Param] Op:%s, dev addr is nullptr.", op_desc_->GetName().c_str());
  args_ = ValueToPtr(args[static_cast<size_t>(args_placement_)].dev_addr);

  if ((davinci_model_->OpNeedDump(op_desc_) || davinci_model_->OpNeedPrint(op_desc_))
    || davinci_model_->GetOpDugReg()) {
    GELOGI("Op %s need dump or print in task info", op_desc_->GetName().c_str());
    dump_args_ = args_;
    dump_flag_ = RT_KERNEL_DUMPFLAG;
  }

  InitArgs(args);

  GE_CHK_STATUS_RET(SetTvmTaskZeroCopy(op_desc_, io_addrs_, args_));

  size_t index = 0U;
  for (const domi::FusionSubTaskInfo &fusion_sub_task_info : task_def.fusion_task().fusion_sub_task_info()) {
    GE_ASSERT_TRUE(index < FUSION_SUB_TASK_MAX_NUM);
    const domi::FusionSubTaskDef &task = fusion_sub_task_info.task();
    switch (fusion_sub_task_info.type()) {
      case domi::FusionSubTaskInfo::AICORE: {
        rt_fusion_task_.subTask[index].type =
            static_cast<rtFusionType_t>(DomiFusionTypeToRtFusionType(fusion_sub_task_info.type()));
        GE_ASSERT_TRUE(rt_fusion_task_.subTask[index].type != RT_FUSION_END);

        if (is_all_kernel_) {
          const auto tiling_info =
            op_desc_->GetExtAttr<std::shared_ptr<optiling::utils::OpRunInfo>>(ge::ATTR_NAME_OP_RUN_INFO);
          if ((tiling_info != nullptr) && (*tiling_info != nullptr)) {
            tiling_key_ = (*tiling_info)->GetTilingKey();
            GELOGI("Op %s tiling key %" PRIu64 "", op_desc_->GetName().c_str(), tiling_key_);
          }
        }

        const auto &aicore_task = task.aicore_fusion_task_info();
        GE_ASSERT_SUCCESS(SetAIcoreLaunchAttrs(aicore_task.config(), rt_fusion_task_.subTask[index].task.aicoreInfo));
        rt_fusion_task_.subTaskNum++;
        break;
      }
      case domi::FusionSubTaskInfo::AICPU: {
        GELOGW("Fusion task no support sub aicpu kernel. ");
        break;
      }
      case domi::FusionSubTaskInfo::CCU: {
        rt_fusion_task_.subTask[index].type =
            static_cast<rtFusionType_t>(DomiFusionTypeToRtFusionType(fusion_sub_task_info.type()));
        GE_ASSERT_TRUE(rt_fusion_task_.subTask[index].type != RT_FUSION_END);
        const auto &ccu_task_group = task.ccu_task_group();
        GE_ASSERT(ccu_task_group.group_size() == 1U);
        string hcom_group_attr_name = ccu_task_group.group(0);
        string hcom_group;
        GE_ASSERT(AttrUtils::GetStr(op_desc_, hcom_group_attr_name, hcom_group),
          "Op %s get hcom group attr name %s", op_desc_->GetName().c_str(), hcom_group_attr_name.c_str());
        GE_ASSERT_SUCCESS(HcclDllHcomMgr::GetInstance().HcomGetCcuTaskInfoFunc(
          hcom_group, tiling_data_host_.data(), &(rt_fusion_task_.subTask[index].task)));
        GELOGI("Op %s hcom group attr name %s group %s",
          op_desc_->GetName().c_str(), hcom_group_attr_name.c_str(), hcom_group.c_str());
        rt_fusion_task_.subTaskNum++;
        break;
      }
      default:
        break;
    }
    index++;
  }

  return SUCCESS;
}

void FusionTaskInfo::UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs) {
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] = iow_addrs.input_logic_addrs[i].logic_addr;
    input_mem_types_[i] = iow_addrs.input_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] = iow_addrs.output_logic_addrs[i].logic_addr;
    output_mem_types_[i] = iow_addrs.output_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    workspace_addrs_[i] = iow_addrs.workspace_logic_addrs[i].logic_addr;
    workspace_mem_types_[i] = iow_addrs.workspace_logic_addrs[i].memory_type;
  }
}

Status FusionTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                            const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHECK_NOTNULL(op_desc_);

  GELOGI("FusionTaskInfo Init Start, op: %s", op_desc_->GetNamePtr());
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));
  UpdateIoAndWorkspaceAddrs(iow_addrs);

  GE_CHK_STATUS_RET_NOLOG(InitKernel(task_def, args));

  io_addr_mem_types_.resize(io_addrs_.size(), static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix));
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), io_addrs_,
                                                io_addr_mem_types_, {op_desc_->GetName(), op_desc_->GetType()}));

  GELOGI("FusionTaskInfo init finish.");
  return SUCCESS;
}

void FusionTaskInfo::UpdateTaskId() {
  if (davinci_model_ != nullptr) {
    GE_CHK_RT_EXEC(rtsGetThreadLastTaskId(&task_id_), return);
    GE_CHK_RT_EXEC(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)), return);
    GELOGD("UpdateTaskId:UpdateTaskId [%u], stream id [%u]:", task_id_, stream_id_);
  }
}

Status FusionTaskInfo::Distribute() {
  GE_CHECK_NOTNULL(op_desc_);
  GE_ASSERT_SUCCESS(ReportL0ExceptionDumpInfo(op_desc_, l0_dump_list_), "[%s] report l0 exception dump addr failed",
                    op_desc_->GetNamePtr());

  const TaskProfGuarder prof_guarder(this);
  const string op_name = op_desc_->GetName();
  GELOGI("Start to launch kernel of %s.", op_name.c_str());
  SetTaskTag(op_name.c_str());

  // 构造参数下发
  rt_args_ex_.args = args_;
  rt_args_ex_.argsSize = args_size_;
  rt_args_ex_.isNoNeedH2DCopy = 1;
  GE_CHK_RT_RET(rtFusionLaunch(&rt_fusion_task_, stream_, &rt_args_ex_));

  call_save_dump_ = true;

  UpdateTaskId();
  GELOGI("FusionTaskInfo Distribute Success, op: %s, taskid: %d, stub func: %p, tiling key: %" PRIu64 ", "
         "streamid: %u, stream: %p, dump_flag: %u",
         op_desc_->GetName().c_str(), task_id_, stub_func_, tiling_key_, stream_id_, stream_, dump_flag_);

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
  }

  return SUCCESS;
}

void FusionTaskInfo::GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) const {
  if (!has_tiling_) {
    return;
  }
  tiling_key = static_cast<uint32_t>(tiling_key_);
  const auto tiling_data_holder = MakeUnique<uint8_t[]>(static_cast<size_t>(tiling_data_size_));
  GE_CHECK_NOTNULL_JUST_RETURN(tiling_data_holder);
  if (aclrtMemcpy(tiling_data_holder.get(), static_cast<uint64_t>(tiling_data_size_), tiling_data_addr_,
      static_cast<uint64_t>(tiling_data_size_), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    return;
  }
  std::stringstream ss;
  gert::PrintHex(tiling_data_holder.get(), static_cast<size_t>(tiling_data_size_), ss);
  tiling_data = ss.str();
}

Status FusionTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, args_placement_);
  return SUCCESS;
}

int64_t FusionTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::FusionTaskDef &fusion_task = task_def.fusion_task();
  return static_cast<int64_t>(fusion_task.op_index());
}

void FusionTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  const domi::FusionTaskDef &fusion_task = task_def.fusion_task();
  davinci_model_->SaveDfxInfo(fusion_task.op_index(), task_def, *this);
  ResetArgsEx();
}

REGISTER_TASK_INFO(MODEL_TASK_FUSION_KERNEL, FusionTaskInfo);
}  // namespace ge