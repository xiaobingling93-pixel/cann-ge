/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/memory_app_type_classifier.h"

#include <string>
#include "mmpa/mmpa_api.h"

#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/runtime_tensor_desc.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr int32_t kSessionNoReuse = 1;
constexpr uint64_t kSessionScopeMemoryMask = 0x100000000UL;
constexpr uint64_t kMemoryTypeMask = 0xFFFFFFFFUL;
constexpr uint32_t kInvalidDeviceId = UINT32_MAX;
constexpr char_t const *kUsedStreamNum = "used_stream_num";
constexpr char_t const *kWorkSpace = "workspace";
constexpr char_t const *kInner = "built-in";
constexpr char_t const *kVendors = "vendors";
constexpr char_t const *kOpImpl = "op_impl";
constexpr char_t const *kLegacySoSuffix = "_legacy.so";
static thread_local uint32_t load_so_count = 0;
uint64_t GetWorkspaceMemTypeByPriority(const bool is_p2p_memory, const bool is_l1_memory, const bool is_ub_memory,
                                       const bool session_scope_memory) {
  if (is_p2p_memory) {
    return RT_MEMORY_P2P_DDR;
  }
  if (is_l1_memory) {
    return RT_MEMORY_L1;
  }
  if (is_ub_memory) {
    return kRtMemoryUB;
  }
  if (session_scope_memory) {
    return kSessionScopeMemoryMask | RT_MEMORY_HBM;
  }
  return RT_MEMORY_HBM;
}

Status GetMaxVarMemSize(const RuntimeParam &runtime_param, uint64_t &max_var_mem_size) {
    const auto var_manager = VarManager::Instance(runtime_param.session_id);
    GE_CHECK_NOTNULL(var_manager);
    max_var_mem_size = static_cast<uint64_t>(var_manager->GetVarMemSize(RT_MEMORY_HBM)) + \
                       static_cast<uint64_t>(var_manager->GetVarConstPlaceHolderMemSize(RT_MEMORY_HBM));
    max_var_mem_size = (max_var_mem_size == 0LU) ? kMemoryVarAddressSize : max_var_mem_size;
    GELOGI("GetMaxVarMemSize max_var_mem_size = %" PRIu64 ".", max_var_mem_size);
    return SUCCESS;
}

size_t GetVarConstPlaceHolderMemSize(const RuntimeParam &runtime_param) {
    size_t var_cph_mem_size = 0L;
    const auto var_manager = VarManager::Instance(runtime_param.session_id);
    if (var_manager != nullptr) {
        var_cph_mem_size = static_cast<size_t>(var_manager->GetVarConstPlaceHolderMemSize(RT_MEMORY_HBM));
    }
    GELOGI("GetVarConstPlaceHolderMemSize var_cph_mem_size = %zu.", var_cph_mem_size);
    return var_cph_mem_size;
}

void FreeP2pMem(const uint32_t device_id, RuntimeParam &runtime_param,
                std::pair<const uint64_t, MemInfo> &mem_type_info) {
  const auto memory_type = static_cast<rtMemType_t>(mem_type_info.first & kMemoryTypeMask);
  if ((mem_type_info.second.memory_base != nullptr)
      && (mem_type_info.second.memory_base != PtrToPtr<void, uint8_t>(ValueToPtr(runtime_param.p2p_fixed_mem_base)))) {
    auto &mem_instance = MemManager::Instance().MemInstance(memory_type);
    GE_CHK_STATUS(mem_instance.FreeMemory(mem_type_info.second.memory_base, device_id), "failed to free memory");
    mem_type_info.second.memory_base = nullptr;
  }
}

const std::map<OppImplVersion, std::string> kVersion2VendorPath{{OppImplVersion::kOpp, "/opp/"},
                                                                {OppImplVersion::kOppKernel, "/opp_latest/"}};

void GetSpecificSoBins(const ge::GeRootModelPtr &root_model, const SoBinType so_bin_type,
                       std::vector<OpSoBinPtr> &so_list) {
  const auto &all_so_list = root_model->GetAllSoBin();
  std::for_each(all_so_list.begin(), all_so_list.end(), [&so_list, so_bin_type](const OpSoBinPtr &so_bin) -> void {
    if (so_bin->GetSoBinType() == so_bin_type) {
      so_list.emplace_back(so_bin);
    }
  });
  // sort, move "_legacy.so" to the end
  std::stable_partition(so_list.begin(), so_list.end(), [](const OpSoBinPtr &so_bin) {
    const std::string so_bin_name = so_bin->GetSoName();
    return so_bin_name.size() < std::strlen(kLegacySoSuffix) ||
           so_bin_name.compare((so_bin_name.size() - std::strlen(kLegacySoSuffix)), std::strlen(kLegacySoSuffix), kLegacySoSuffix) != 0;
  });
}
}  // namespace

bool ModelUtils::ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                                  const int64_t size) {
  if (CheckInt64AddOverflow(offset, size) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Int64 %" PRId64 " and %" PRId64 " addition can result in overflow!", offset, size);
    return false;
  }
  const int64_t mem_range = offset + size;
  if (total_size < static_cast<uint64_t>(mem_range)) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) memory out of range, offset:%" PRId64 ", size:"
                       "%" PRId64 ", exceed total size:%" PRIu64 ".", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str(), offset, size, total_size);
    GELOGE(OUT_OF_MEMORY, "[Check][Param]Node:%s(%s) memory out of range, offset:%" PRId64
           ", size:%" PRId64 ", exceed total size:%" PRIu64 ".",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), offset, size, total_size);
    return false;
  }
  return true;
}

Status ModelUtils::CreateOmOppDir(std::string &opp_dir) {
  opp_dir.clear();
  GE_ASSERT_SUCCESS(ge::GetAscendWorkPath(opp_dir));
  if (opp_dir.empty()) {
    const ge::char_t *path_env = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HOME, path_env);
    GE_ASSERT_NOTNULL(path_env);
    GE_ASSERT_TRUE(strnlen(path_env, static_cast<size_t>(MMPA_MAX_PATH)) > 0U);

    const std::string file_path = ge::RealPath(path_env);
    GE_ASSERT_TRUE(!file_path.empty());
    opp_dir = file_path;
  }
  if (opp_dir.back() != '/') {
    opp_dir += '/';
  }
  opp_dir += ".ascend_temp/.om_exe_data/"
      + std::to_string(mmGetPid())
      + "_" + std::to_string(mmGetTid())
      + "_" + std::to_string(load_so_count++)
      + "/";
  GELOGD("opp_dir is %s", opp_dir.c_str());

  GE_ASSERT_TRUE(mmAccess2(opp_dir.c_str(), M_F_OK) != EN_OK);
  GE_ASSERT_TRUE(ge::CreateDir(opp_dir) == 0);

  return ge::GRAPH_SUCCESS;
}

Status ModelUtils::RmOmOppDir(const std::string &opp_dir) {
  GELOGD("Start to rm opp dir %s", opp_dir.c_str());
  if (!opp_dir.empty()) {
    GE_ASSERT_TRUE(mmRmdir(opp_dir.c_str()) == 0, "remove dir [%s] failed!", opp_dir.c_str());
  }
  GELOGI("Remove dir success, opp_dir is %s", opp_dir.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelUtils::SaveToFile(const std::shared_ptr<ge::OpSoBin> &so_bin, const std::string &opp_path) {
  constexpr mmMode_t kAccess = static_cast<mmMode_t>(static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR) |
                                                     static_cast<uint32_t>(M_UMASK_USREXEC));
  const int32_t fd = mmOpen2(opp_path.c_str(),
                             static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) | static_cast<uint32_t>(M_CREAT) |
                                                  static_cast<uint32_t>(O_TRUNC)),
                             kAccess);
  GELOGD("Prepare to save so: [%s], the save path: [%s]", so_bin->GetSoName().c_str(), opp_path.c_str());
  GE_MAKE_GUARD(close_opp_path, [&fd]() -> void { mmClose(fd); });
  GE_ASSERT(fd >= 0, "open file [%s] failed!", opp_path.c_str());
  const int32_t write_count =
      mmWrite(fd, const_cast<uint8_t *>(so_bin->GetBinData()), static_cast<uint32_t>(so_bin->GetBinDataSize()));
  GE_ASSERT_TRUE((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR), "write file [%s] failed!",
                 opp_path.c_str());
  return ge::GRAPH_SUCCESS;
}


///
/// @ingroup ge
/// @brief Get input size.
/// @return std::vector<int64_t>
///
std::vector<int64_t> ModelUtils::GetInputSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_input_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_size);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  for (size_t i = 0U; i < inputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
      TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
      GELOGI("Tensor has no size, op: %s, input index: %zu", op_desc->GetName().c_str(), i);
      continue);

    GELOGI("GetInputSize op:[%s], index:[%zu], size:[%" PRId64 "]", op_desc->GetName().c_str(), i, tensor_size);
    v_input_size.push_back(tensor_size);
  }

  return v_input_size;
}

///
/// @ingroup ge
/// @brief Get output size.
/// @return std::vector<int64_t>
///
std::vector<int64_t> ModelUtils::GetOutputSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_output_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_size);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_size);

  for (size_t i = 0U; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t string_max_size = 0;
    if ((tensor_desc->GetDataType() == DT_STRING) && AttrUtils::GetInt(op_desc, "_op_max_size", string_max_size)) {
      GELOGI("Get op max size value = %" PRId64, string_max_size);
      v_output_size.push_back(string_max_size);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
      TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
      GELOGI("Tensor has no size, op: %s, output index: %zu", op_desc->GetName().c_str(), i);
      continue);

    GELOGD("GetOutputSize op:[%s], index:[%zu], size:[%" PRId64 "]", op_desc->GetName().c_str(), i, tensor_size);
    v_output_size.push_back(tensor_size);
  }

  return v_output_size;
}

///
/// @ingroup ge
/// @brief Get workspace size.
/// @return std::vector<int64_t>
///
std::vector<int64_t> ModelUtils::GetWorkspaceSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_workspace_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_size);

  const std::vector<int64_t> v_workspace_num = op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_num.size() != v_workspace_bytes.size()) {
    GELOGW("workspace_num[%zu]!= workspace_bytes[%zu]", v_workspace_num.size(), v_workspace_bytes.size());
    return v_workspace_size;
  }

  return v_workspace_bytes;
}

///
/// @ingroup ge
/// @brief Get weight size.
/// @return std::vector<int64_t>
///
std::vector<int64_t> ModelUtils::GetWeightSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_weight_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weight_size);

  // const op, get weight directly
  const std::string type_name = op_desc->GetType();
  if ((type_name == CONSTANT) || (type_name == CONSTANTOP)) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weight_size.push_back(static_cast<int64_t>(TensorUtils::GetWeightSize(weight)));
    }

    return v_weight_size;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0U; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      int64_t tensor_size = 0;
      (void)TensorUtils::GetSize(*tensor_desc, tensor_size);
      v_weight_size.push_back(tensor_size);
    }
  }

  return v_weight_size;
}

///
/// @ingroup ge
/// @brief Get weights.
/// @return std::vector<ConstGeTensorPtr>
///
std::vector<ConstGeTensorPtr> ModelUtils::GetWeights(const ConstOpDescPtr &op_desc) {
  std::vector<ConstGeTensorPtr> v_weights;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weights);

  // const op, get weight directly
  const std::string op_type = op_desc->GetType();
  if ((op_type == CONSTANT) || (op_type == CONSTANTOP)) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weights.push_back(weight);
    }

    return v_weights;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();

  for (size_t i = 0U; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      ConstGeTensorPtr weight = nullptr;
      if (AttrUtils::GetTensor(*tensor_desc, ATTR_NAME_WEIGHTS, weight)) {
        v_weights.push_back(weight);
      }
    }
  }

  return v_weights;
}

///
/// @ingroup ge
/// @brief Get AiCpuOp Input descriptor.
/// @return std::vector<ccAICPUTensor>
///
std::vector<ccAICPUTensor> ModelUtils::GetInputDescs(const ConstOpDescPtr &op_desc) {
  // AiCpuOp::GetInputDescs
  std::vector<ccAICPUTensor> v_input_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_descs);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();

  for (size_t i = 0U; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {  // skip Const input node
      continue;
    }

    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0U;
    if (TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) != GRAPH_SUCCESS) {
      GELOGW("Get dim_cnt unsuccessful");
      continue;
    }

    ccAICPUTensor tmp{};
    tmp.format = static_cast<tagOpTensorFormat>(tensor_desc->GetFormat());
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    tmp.data_type = static_cast<tagOpDataType>(tensor_desc->GetDataType());

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      const int64_t tensor_dim = tensor_desc->GetShape().GetDim(static_cast<size_t>(j));
      if (tensor_dim > INT32_MAX) {
        GELOGW("Op[%s], input tensor[%zu], dim[%d]: tensor_dim[%" PRId64 "] is greater than INT32_MAX[%d]",
               op_desc->GetName().c_str(), i, j, tensor_dim, INT32_MAX);
      }
      tmp.dim[j] = (j < tmp.dim_cnt) ? static_cast<int32_t>(tensor_dim) : 1;
    }

    v_input_descs.push_back(tmp);
  }

  return v_input_descs;
}

///
/// @ingroup ge
/// @brief Get AiCpuOp Output descriptor.
/// @return std::vector<ccAICPUTensor>
///
std::vector<ccAICPUTensor> ModelUtils::GetOutputDescs(const ConstOpDescPtr &op_desc) {
  // AiCpuOp::GetOutputDescs
  std::vector<ccAICPUTensor> v_output_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_descs);

  // init op output ccAICPUTensor struct
  const size_t output_num = op_desc->GetOutputsSize();
  for (size_t i = 0UL; i < output_num; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0U;
    if (TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) != GRAPH_SUCCESS) {
      GELOGW("Get dim_cnt failed");
      continue;
    }

    ccAICPUTensor tmp{};
    tmp.format = static_cast<tagOpTensorFormat>(tensor_desc->GetFormat());
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    tmp.data_type = static_cast<tagOpDataType>(tensor_desc->GetDataType());

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      const int64_t tensor_dim = tensor_desc->GetShape().GetDim(static_cast<size_t>(j));
      if (tensor_dim > INT32_MAX) {
        GELOGW("Op[%s], output tensor[%zu], dim[%d]: tensor_dim[%" PRId64 "] is greater than INT32_MAX[%d]",
               op_desc->GetName().c_str(), i, j, tensor_dim, INT32_MAX);
      }
      tmp.dim[j] = (j < tmp.dim_cnt) ? static_cast<int32_t>(tensor_dim) : 1;
    }

    v_output_descs.push_back(tmp);
  }

  return v_output_descs;
}

///
/// @ingroup ge
/// @brief Get input address.
/// @return std::vector<void*>
///
std::vector<void *> ModelUtils::GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                              std::vector<uint64_t> &mem_type) {
  GELOGD("Start GetInputAddrs: op_name[%s].", op_desc->GetName().c_str());
  auto v_input_addr = GetInputDataAddrs(model_param, op_desc, mem_type);
  if (GetInputOutputDescAddrs(model_param, op_desc, op_desc->GetAllInputsDescPtr(), v_input_addr) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check] GetInputOutputDescAddrs failed: op_name[%s]", op_desc->GetName().c_str());
    return {};
  }

  return v_input_addr;
}

///
/// @ingroup ge
/// @brief Get input address value.
/// @return std::vector<uint64_t>
///
std::vector<uint64_t> ModelUtils::GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                     std::vector<uint64_t> &mem_type) {
  GELOGD("Start GetInputAddrsValue: op_name[%s]", op_desc->GetName().c_str());
  return VPtrToValue(GetInputAddrs(model_param, op_desc, mem_type));
}

Status ModelUtils::RefreshAddressByMemType(const RuntimeParam &model_param, const NodeMemInfo &node_mem_info,
                                           void *&mem_addr) {
  switch (node_mem_info.mem_type_) {
    case RT_MEMORY_L1:  // fusion
    case kRtMemoryUB:
      mem_addr = ValueToPtr(static_cast<uint64_t>(node_mem_info.logical_offset_));
      break;
    case RT_MEMORY_TS:
      // The input size and peer output size may be not consecutive, therefore, the tensor_size is not been checked.
      if (!ValidateMemRange(node_mem_info.op_desc_, model_param.mem_size, node_mem_info.logical_offset_, 0)) {
        return FAILED;
      }
      mem_addr =
          model_param.ts_mem_mall->Acquire(node_mem_info.logical_offset_, static_cast<uint64_t>(node_mem_info.size_));
      break;
    case kSessionScopeMemoryMask | RT_MEMORY_HBM:
    case RT_MEMORY_HOST:
    case RT_MEMORY_HOST_SVM:
    case RT_MEMORY_P2P_DDR: {
      const auto &mem_info =
          model_param.memory_infos.at(node_mem_info.mem_type_);  // Init by InitRuntimeParams, key is valid.
      mem_addr = mem_info.GetMemory(node_mem_info.logical_offset_, node_mem_info.size_);
      break;
    }
    case RT_MEMORY_HBM:
    case RT_MEMORY_L2: // l2 also malloc hbm for datadump
    case RT_MEMORY_DEFAULT:
      // size can be 0 and need update addr for input and output
      if ((node_mem_info.size_ <= 0) && (node_mem_info.io_type_ == kWorkSpace)) {
        return SUCCESS;
      }
      // The input node_mem_info.size_ and peer output size may be not consecutive, therefore, the tensor_size is not
      // been checked.
      if (!ValidateMemRange(node_mem_info.op_desc_, model_param.mem_size, node_mem_info.logical_offset_, 0)) {
        return FAILED;
      }
      mem_addr = model_param.GetMemAddr(node_mem_info.logical_offset_);
      break;
    default:
      GELOGE(FAILED, "mem_type %" PRIu64 " is not supported for now.", node_mem_info.mem_type_);
      return FAILED;
  }
  return SUCCESS;
}
///
/// @ingroup ge
/// @brief Get input data address value.
/// @return std::vector<uint64_t>
///
std::vector<uint64_t> ModelUtils::GetInputDataAddrsValue(const RuntimeParam &model_param,
                                                         const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputDataAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetInputDataAddrsValue(const RuntimeParam &model_param,
                                                         const ConstOpDescPtr &op_desc,
                                                         std::vector<uint64_t> &mem_type) {
  return VPtrToValue(GetInputDataAddrs(model_param, op_desc, mem_type));
}

///
/// @ingroup ge
/// @brief Get input data address.
/// @return std::vector<void*>
///
std::vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetInputDataAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                  std::vector<uint64_t> &mem_type) {
  std::vector<void *> v_input_data_addr;  // init as:buf_base + op_def_->input(i));
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_data_addr);
  const uint64_t session_id = model_param.session_id;
  GELOGD("Print Session Id:%" PRIu64 ", op name[%s]", session_id, op_desc->GetName().c_str());
  GE_CHECK_NOTNULL_EXEC(VarManager::Instance(session_id), return v_input_data_addr);
  const size_t inputs_size = op_desc->GetInputsSize();
  const std::vector<int64_t> v_input_offset = op_desc->GetInputOffset();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();

  size_t non_const_index = 0UL;
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  const bool check_failed = has_mem_type_attr && (v_memory_type.size() != inputs_size);
  if (check_failed) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return v_input_data_addr;
  }

  v_input_data_addr.reserve(inputs_size);
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_IF_BOOL_EXEC(tensor_desc == nullptr, GELOGD("Op: %s, Index: %zu, has no input", op_desc->GetName().c_str(), i);
                    continue);
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      // Add weights address to input
      int64_t data_offset = 0;
      GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));
      int64_t weight_size = 0;
      // The reason why GetTensorSizeInBytes is used here is that the weight is allocated based on the size of
      // TensorData in function AdjustConstWeightSize. and the size is zero when the tensor is empty.
      GE_CHK_STATUS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weight_size));
      GE_IF_BOOL_EXEC(!ValidateMemRange(op_desc, model_param.weight_size, data_offset, weight_size), return {});
      void *const weight_addr = ValueToPtr(model_param.weight_base + static_cast<uint64_t>(data_offset));
      v_input_data_addr.push_back(weight_addr);
      mem_type.push_back(kWeightMemType);
      GELOGI("[IMAS]GetInputDataAddrs graph_%u type[C] name[%s] input[%zu] size[%" PRId64 "] memaddr[%p]",
        model_param.graph_id, op_desc->GetName().c_str(), i, weight_size, weight_addr);
      non_const_index++;
      continue;
    }

    GE_IF_BOOL_EXEC(non_const_index >= v_input_offset.size(), break);

    const int64_t input_offset = v_input_offset[non_const_index];
    const auto iter = model_param.fileconstant_addr_mapping.find(input_offset);
    if (iter != model_param.fileconstant_addr_mapping.end()) {
      v_input_data_addr.push_back(reinterpret_cast<void *>(iter->second));
      mem_type.push_back(kConstantMemType);
      non_const_index++;
      continue;
    }

    non_const_index++;
    int64_t inner_offset = 0;
    (void)AttrUtils::GetInt(op_desc->MutableInputDesc(static_cast<uint32_t>(i)), ATTR_NAME_INNER_OFFSET, inner_offset);
    const bool is_check_var_manager = (GetVarConstPlaceHolderMemSize(model_param) > 0U || model_param.var_size > 0U);
    if (is_check_var_manager && (VarManager::Instance(session_id)->IsVarAddr(input_offset - inner_offset))) {
      void *variable_addr = nullptr;
      if (CheckInt64AddOverflow(tensor_size, inner_offset) != SUCCESS) {
          return {};
      }
      if (GetVarAddr(model_param, input_offset - inner_offset, variable_addr) != SUCCESS) {
          return {};
      }
      variable_addr = ValueToPtr(PtrToValue(variable_addr) + static_cast<uint64_t>(inner_offset));
      v_input_data_addr.push_back(variable_addr);
      mem_type.push_back(kVarMemType);
      GELOGI("[IMAS]GetInputDataAddrs graph_%u type[V] name[%s] input[%zu] memaddr[%p]",
             model_param.graph_id, op_desc->GetName().c_str(), i, variable_addr);
      continue;
    }

    int64_t tensor_mem_type = -1;
    const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
    uint64_t memory_type(RT_MEMORY_DEFAULT);
    if (tensor_has_mem_type) {
      memory_type = static_cast<uint64_t>(tensor_mem_type);
    } else if (v_memory_type.size() > i) {
      memory_type = static_cast<uint64_t>(v_memory_type[i]);
    } else {  // do nothing, use default type
    }
    const NodeMemInfo node_mem_info{memory_type, op_desc, i, "input", tensor_size, input_offset};
    void *mem_addr = nullptr;
    if (RefreshAddressByMemType(model_param, node_mem_info, mem_addr) != SUCCESS) {
      GELOGE(FAILED, "[IMAS]get failed for graph_%u %s", model_param.graph_id,
             node_mem_info.ToString().c_str());
      return {};
    }
    GELOGI("[IMAS]graph_%u %s memaddr[%p]", model_param.graph_id, node_mem_info.ToString().c_str(), mem_addr);
    v_input_data_addr.push_back(mem_addr);
    mem_type.push_back(memory_type);
  }

  return v_input_data_addr;
}

///
/// @ingroup ge
/// @brief Get variable address.
/// @return Status
///
Status ModelUtils::GetVarAddr(const RuntimeParam &model_param, const int64_t offset, void *&var_addr) {
  Status ret = SUCCESS;
  GE_CHECK_NOTNULL(VarManager::Instance(model_param.session_id));
  const rtMemType_t mem_type = VarManager::Instance(model_param.session_id)->GetVarMemType(offset);
  switch (mem_type) {
    case RT_MEMORY_RDMA_HBM: {
      if (offset < 0) {
        REPORT_INNER_ERR_MSG("E19999", "Param offset:%" PRId64 " < 0, check invalid", offset);
        GELOGE(PARAM_INVALID, "[Check][Param] Param offset:%" PRId64 " cannot be negative", offset);
        ret = PARAM_INVALID;
        break;
      }
      var_addr = ValueToPtr(static_cast<uint64_t>(offset));
      break;
    }
    case RT_MEMORY_HBM: {
      const auto &var_manager = VarManager::Instance(model_param.session_id);
      GE_CHECK_NOTNULL(var_manager);
      uint8_t *var_logic_addr = nullptr;
      var_logic_addr = PtrToPtr<void, uint8_t>(ValueToPtr(static_cast<uint64_t>(offset)));
      var_addr = var_manager->GetVarMemoryAddr(model_param.graph_name, var_logic_addr,
                                               RT_MEMORY_HBM, model_param.device_id);
      GE_CHK_BOOL_RET_STATUS(var_addr != nullptr, INTERNAL_ERROR, "[Get][VarAddr] failed.");
      break;
    }
    default: {
      REPORT_INNER_ERR_MSG("E19999", "Get mem_type:%u for offset:%" PRId64 " is unsupported, "
		         "check invalid", mem_type, offset);
      GELOGE(PARAM_INVALID, "[Check][Param] Get mem_type:%d for offset:%" PRId64 " is unsupported, check invalid",
             mem_type, offset);
      ret = PARAM_INVALID;
      break;
    }
  }
  GE_CHECK_NOTNULL(var_addr);
  return ret;
}

///
/// @ingroup ge
/// @brief Get output address.
/// @return std::vector<void*>
///
std::vector<void *> ModelUtils::GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc)  {
  std::vector<uint64_t> mem_type;
  return GetOutputAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                               std::vector<uint64_t> &mem_type,
                                               const bool has_optional_addr) {
  GELOGD("Start GetOutputAddrs: op_name[%s].", op_desc->GetName().c_str());
  auto v_output_addr = GetOutputDataAddrs(model_param, op_desc, mem_type, has_optional_addr);
  if (GetInputOutputDescAddrs(
    model_param, op_desc, op_desc->GetAllOutputsDescPtr(), v_output_addr, has_optional_addr) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check] GetInputOutputDescAddrs failed: op_name[%s]", op_desc->GetName().c_str());
    return {};
  }
  return v_output_addr;
}

///
/// @ingroup ge
/// @brief Get output address value.
/// @return std::vector<uint64_t>
///
std::vector<uint64_t> ModelUtils::GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                      std::vector<uint64_t> &mem_type,
                                                      const bool has_optional_addr) {
  GELOGD("Start GetOutputAddrsValue: op_name[%s].", op_desc->GetName().c_str());
  return VPtrToValue(GetOutputAddrs(model_param, op_desc, mem_type, has_optional_addr));
}
///
/// @ingroup ge
/// @brief Get output address value.
/// @return std::vector<uint64_t>
///
std::vector<uint64_t> ModelUtils::GetOutputDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputDataAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetOutputDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc,
                                                          std::vector<uint64_t> &mem_type) {
  return VPtrToValue(GetOutputDataAddrs(model_param, op_desc, mem_type));
}

///
/// @ingroup ge
/// @brief Get output data address.
/// @return Status
///
std::vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetOutputDataAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                   std::vector<uint64_t> &mem_type,
                                                   const bool has_optional_addr) {
  std::vector<void *> v_output_data_addr;  // init as:buf_base + op_def_->output(i)
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_data_addr);
  GELOGD("Start GetOutputDataAddrs: op_name[%s]", op_desc->GetName().c_str());
  const uint64_t session_id = model_param.session_id;
  GE_CHECK_NOTNULL_EXEC(VarManager::Instance(session_id), return v_output_data_addr);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_data_addr);
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return v_output_data_addr;
  }

  v_output_data_addr.reserve(outputs_size);
  for (size_t i = 0U; i < outputs_size; ++i) {
    const auto iter = model_param.fileconstant_addr_mapping.find(v_output_offset[i]);
    if (iter != model_param.fileconstant_addr_mapping.end()) {
      v_output_data_addr.push_back(reinterpret_cast<void *>(iter->second));
      mem_type.push_back(kConstantMemType);
      GELOGI("Find mapping existed. index:%zu key offset:%" PRId64 ", dev addr:%" PRIx64,
             i, v_output_offset[i], iter->second);
      continue;
    }

    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    // skip some addr
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }
    int32_t calc_type = 0;
    (void)AttrUtils::GetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (calc_type == static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY)) {
      if (has_optional_addr) {
        v_output_data_addr.push_back(nullptr);
        mem_type.push_back(kFixMemType);
      }
      GELOGD("%s is an optional output, has option addr:%d.",
        op_desc->GetName().c_str(), static_cast<int32_t>(has_optional_addr));
      continue;
    }
    // var addr
    int64_t inner_offset = 0;
    (void)AttrUtils::GetInt(op_desc->MutableOutputDesc(static_cast<uint32_t>(i)), ATTR_NAME_INNER_OFFSET, inner_offset);
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    const bool is_check_var_manager = (GetVarConstPlaceHolderMemSize(model_param) > 0U || model_param.var_size > 0U);
    if (is_check_var_manager && (VarManager::Instance(session_id)->IsVarAddr(v_output_offset[i] - inner_offset))) {
      void *variable_addr = nullptr;
      if (CheckInt64AddOverflow(tensor_size, inner_offset) != SUCCESS) {
          return {};
      }
      if (GetVarAddr(model_param, v_output_offset[i] - inner_offset, variable_addr) != SUCCESS) {
          return {};
      }
      variable_addr = ValueToPtr(PtrToValue(variable_addr) + static_cast<uint64_t>(inner_offset));
      v_output_data_addr.push_back(variable_addr);
      mem_type.push_back(kVarMemType);
      GELOGI("[IMAS]graph_%u type[V] name[%s] output[%zu] memaddr[%p]",
              model_param.graph_id, op_desc->GetName().c_str(), i, variable_addr);
      continue;
    }

    int64_t tensor_mem_type = -1;
    const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
    uint64_t memory_type(RT_MEMORY_DEFAULT);
    if (tensor_has_mem_type) {
      memory_type = static_cast<uint64_t>(tensor_mem_type);
    } else if (has_mem_type_attr) {
      memory_type = static_cast<uint64_t>(v_memory_type[i]);
    } else {  // do nothing, use default type
    }
    const NodeMemInfo node_mem_info{memory_type, op_desc, i, "output", tensor_size, v_output_offset[i]};
    void *mem_addr = nullptr;
    if (RefreshAddressByMemType(model_param, node_mem_info, mem_addr) != SUCCESS) {
      GELOGE(FAILED, "[IMAS]get failed for graph_%u %s", model_param.graph_id,
             node_mem_info.ToString().c_str());
      return {};
    }
    GELOGI("[IMAS]graph_%u %s memaddr[%p]", model_param.graph_id, node_mem_info.ToString().c_str(), mem_addr);
    v_output_data_addr.push_back(mem_addr);
    mem_type.push_back(memory_type);
  }
  return v_output_data_addr;
}

static Status FillSinkTensorDesc(RuntimeTensorDesc &sink_tensor_desc, const GeTensorDescPtr &tensor_desc,
                                 const uint64_t data_addr) {
  sink_tensor_desc.data_addr = data_addr;
  sink_tensor_desc.dtype = static_cast<int64_t>(tensor_desc->GetDataType());
  sink_tensor_desc.format = static_cast<int64_t>(tensor_desc->GetFormat());
  const auto shape = tensor_desc->GetShape();
  const int64_t dim_num = static_cast<int64_t>(shape.GetDimNum());
  sink_tensor_desc.shape[0] = dim_num;
  if (dim_num > kMaxDimSize) {
    GELOGE(PARAM_INVALID, "shape dim size[%" PRId64 "] out of range[%zu]", dim_num, kMaxDimSize);
    return FAILED;
  }
  for (int64_t i = 0; i < dim_num; i++) {
    sink_tensor_desc.shape[i + 1] = shape.GetDim(static_cast<size_t>(i));
  }
  const auto ori_shape = tensor_desc->GetOriginShape();
  const int64_t ori_dim_num = static_cast<int64_t>(ori_shape.GetDimNum());
  sink_tensor_desc.original_shape[0] = ori_dim_num;
  if (ori_dim_num > kMaxDimSize) {
    GELOGE(PARAM_INVALID, "original shape dim size[%" PRId64 "] out of range[%zu]", ori_dim_num, kMaxDimSize);
    return FAILED;
  }
  for (int64_t i = 0; i < ori_dim_num; i++) {
    sink_tensor_desc.original_shape[i + 1] = ori_shape.GetDim(static_cast<size_t>(i));
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get tensor description address, and copy data address to tensor description memory.
/// @return Status
///
Status ModelUtils::GetInputOutputDescAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                           const OpDesc::Vistor<GeTensorDescPtr> &tensor_desc_visitor,
                                           std::vector<void *> &v_addrs,
                                           const bool has_optional_addr) {
  std::vector<int64_t> v_data_mem_type;
  (void) AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_data_mem_type);
  size_t tensor_cnt = 0UL;
  for (const auto &tensor_desc : tensor_desc_visitor) {
    int32_t calc_type = 0;
    const bool ret = AttrUtils::GetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (ret && (calc_type == static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY))) {
      if (has_optional_addr) {
        tensor_cnt++;
      }
      GELOGD("%s is an optional output, has option addr:%d.",
        op_desc->GetName().c_str(), static_cast<int32_t>(has_optional_addr));
      continue;
    }
    int64_t mem_offset;
    const bool has_offset_attr = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, mem_offset);
    if (!has_offset_attr) {
      tensor_cnt++;
      continue;
    }

    constexpr size_t size = sizeof(struct RuntimeTensorDesc);
    GE_IF_BOOL_EXEC(!ValidateMemRange(op_desc, model_param.mem_size, mem_offset, static_cast<int64_t>(size)),
                    return FAILED);
    void *mem_addr = nullptr;
    if ((v_data_mem_type.size() > tensor_cnt) && (v_data_mem_type[tensor_cnt] == static_cast<int64_t>(RT_MEMORY_TS))) {
      mem_addr = model_param.ts_mem_mall->Acquire(mem_offset, size);
    } else {
      mem_addr = model_param.GetMemAddr(mem_offset);
    }

    if (tensor_cnt >= v_addrs.size()) {
      GELOGE(FAILED, "[Check] update tensor desc addr failed, tensor_cnt:%zu, size:%zu", tensor_cnt, v_addrs.size());
      return FAILED;
    }
    RuntimeTensorDesc sink_tensor_desc;
    GE_CHK_STATUS_RET_NOLOG(FillSinkTensorDesc(sink_tensor_desc, tensor_desc, PtrToValue(v_addrs[tensor_cnt])));
    const rtError_t rt_ret = rtMemcpy(mem_addr, size, &sink_tensor_desc, size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpy failed, size:%zu, ret:%d", size, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] copy data_addr failed, size:%zu, ret:%d", size, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    v_addrs[tensor_cnt] = mem_addr;
    GELOGD("Calc op[%s] tenser[%zu] desc addr[%p] ok", op_desc->GetName().c_str(), tensor_cnt, mem_addr);
    tensor_cnt++;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get workspace data address value.
/// @return std::vector<uint64_t>
///
std::vector<uint64_t> ModelUtils::GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                             const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetWorkspaceDataAddrsValue(model_param, op_desc, mem_type);
}

std::vector<uint64_t> ModelUtils::GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                             const ConstOpDescPtr &op_desc,
                                                             std::vector<uint64_t> &mem_type) {
  return VPtrToValue(GetWorkspaceDataAddrs(model_param, op_desc, mem_type));
}
///
/// @ingroup ge
/// @brief Get workspace data address.
/// @return std::vector<void*>
///
std::vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc) {
  std::vector<uint64_t> mem_type;
  return GetWorkspaceDataAddrs(model_param, op_desc, mem_type);
}

std::vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                      std::vector<uint64_t> &mem_type) {
  std::vector<void *> v_workspace_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_data_addr);
  GELOGD("Start GetWorkspaceDataAddrs: op_name[%s].", op_desc->GetName().c_str());
  const std::vector<int64_t> v_workspace_offset = op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_offset.size() != v_workspace_bytes.size()) {
    GELOGW("v_workspace_offset.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_offset.size(),
           v_workspace_bytes.size());
    return v_workspace_data_addr;
  }

  vector_bit_t workspace_reuse_flag;
  const bool has_workspace_reuse = AttrUtils::GetListBool(op_desc, "workspace_reuse_flag", workspace_reuse_flag);
  std::vector<int64_t> v_memory_type;
  std::vector<int64_t> workspace_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  const bool has_mem_type_workspace =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_memory_type);
  if ((has_mem_type_attr && (v_memory_type.size() != v_workspace_offset.size())) ||
      (has_mem_type_workspace && (workspace_memory_type.size() != v_workspace_offset.size()))) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
                       "same, op:%s(%s), check invalid",
                       TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(),
                       ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(), workspace_memory_type.size(), v_workspace_offset.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID,
           "[Check][Param] Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
           "same, op:%s(%s), check invalid",
           TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
           workspace_memory_type.size(), v_workspace_offset.size(), op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_workspace_data_addr;
  }
  std::vector<int32_t> workspace_no_reuse_scope;
  const bool has_workspace_no_reuse_scope = AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE,
                                                                  workspace_no_reuse_scope);
  v_workspace_data_addr.reserve(v_workspace_bytes.size());
  for (size_t i = 0U; i < v_workspace_bytes.size(); ++i) {
    // Temporary solution, the aicpu workspace of multiple images cannot be shared.
    const bool aicpu_work_space = (has_workspace_reuse && (i < workspace_reuse_flag.size()) &&
        (!workspace_reuse_flag[i]) && (!model_param.is_single_op));
    if (aicpu_work_space) {
      void *const mem_addr = model_param.aicpu_mem_mall->Acquire(v_workspace_offset[i],
                                                                 static_cast<uint64_t>(v_workspace_bytes[i]));
      v_workspace_data_addr.push_back(mem_addr);
      mem_type.push_back(kAicpuMemMallMemType);
      GELOGI("[IMAS]graph_%u type[F] name[%s] aicpu workspace[%zu] offset[%" PRId64 "] bytes[%" PRId64 "] memaddr[%p]",
             model_param.graph_id, op_desc->GetName().c_str(), i, v_workspace_offset[i], v_workspace_bytes[i],
             mem_addr);
      continue;
    }
    const bool session_scope_memory = (has_workspace_no_reuse_scope) && (i < workspace_no_reuse_scope.size()) &&
        (workspace_no_reuse_scope[i] == kSessionNoReuse);
    const bool is_p2p_memory =
        has_mem_type_workspace && (static_cast<uint64_t>(workspace_memory_type[i]) == RT_MEMORY_P2P_DDR);
    const bool is_l1_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == RT_MEMORY_L1);
    const bool is_ub_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == kRtMemoryUB);
    const uint64_t memory_type = GetWorkspaceMemTypeByPriority(is_p2p_memory, is_l1_memory, is_ub_memory,
                                                               session_scope_memory);
    const NodeMemInfo node_mem_info{memory_type, op_desc, i, kWorkSpace, v_workspace_bytes[i], v_workspace_offset[i]};
    void *mem_addr = nullptr;
    if (RefreshAddressByMemType(model_param, node_mem_info, mem_addr) != SUCCESS) {
      GELOGE(FAILED, "[IMAS]get failed for graph_%u %s", model_param.graph_id, node_mem_info.ToString().c_str());
      return {};
    }
    GELOGI("[IMAS]graph_%u %s memaddr[%p]", model_param.graph_id, node_mem_info.ToString().c_str(), mem_addr);
    v_workspace_data_addr.push_back(mem_addr);
    mem_type.push_back(memory_type);
  }

  return v_workspace_data_addr;
}

///
/// @ingroup ge
/// @brief Get runtime memory address.
/// @return Status
///
Status ModelUtils::GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr)  {
  uint64_t mem_type = kFixMemType;
  return GetRtAddress(param, logic_addr, mem_addr, mem_type);
}

Status ModelUtils::GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr,
                                uint64_t &mem_type) {
  if (logic_addr == std::numeric_limits<uintptr_t>::max()) {
    GELOGI("Got placeholder logic addr.");
    mem_addr = nullptr;
    return SUCCESS;
  }
  void *runtime_base_addr = nullptr;
  uint64_t max_logic_offset = 0U;
  uint64_t max_var_mem_size = 0U;
  GE_CHK_STATUS_RET(GetMaxVarMemSize(param, max_var_mem_size), "Failed to get MaxVarMemSize");
  bool is_check_var_manager = (GetVarConstPlaceHolderMemSize(param) > 0U || param.var_size > 0U);
  if ((param.logic_mem_base <= logic_addr) && (logic_addr < (param.logic_mem_base + param.mem_size))) {
    mem_type = kFmMemType;
    const size_t logical_offset = logic_addr - param.logic_mem_base;
    mem_addr = reinterpret_cast<uint8_t *>(param.GetMemAddr(static_cast<int64_t>(logical_offset)));
    return SUCCESS;
  } else if ((param.logic_weight_base <= logic_addr) && (logic_addr < (param.logic_weight_base + param.weight_size))) {
    mem_type = kWeightMemType;
    runtime_base_addr = ValueToPtr(param.weight_base - param.logic_weight_base);
    max_logic_offset = param.logic_weight_base + param.weight_size;
    GELOGI("The logic addr:0x%" PRIx64 " is weight address, base:0x%" PRIx64 ", size:%" PRIu64 ", mem_type:%" PRIu64 ".",
      logic_addr, param.logic_weight_base, param.weight_size, mem_type);
  } else if (is_check_var_manager && (param.logic_var_base <= logic_addr)
             && (logic_addr < (param.logic_var_base + max_var_mem_size))) {
    const auto &iter = param.fileconstant_addr_mapping.find(static_cast<int64_t>(logic_addr));
    if (iter != param.fileconstant_addr_mapping.cend()) {
      GELOGI("Find mapping existed, logic_addr:%" PRId64 ", dev_addr:0x%" PRIx64,
        static_cast<int64_t>(logic_addr), iter->second);
      mem_addr = PtrToPtr<void, uint8_t>(ValueToPtr(iter->second));
      GE_CHECK_NOTNULL(mem_addr);
      mem_type = kConstantMemType;
      return SUCCESS;
    }
    const auto &var_manager = VarManager::Instance(param.session_id);
    GE_CHECK_NOTNULL(var_manager);
    uint8_t *var_logic_addr = nullptr;
    var_logic_addr = PtrToPtr<void, uint8_t>(ValueToPtr(logic_addr));
    mem_addr = var_manager->GetVarMemoryAddr(var_logic_addr, RT_MEMORY_HBM, param.device_id);
    GE_CHECK_NOTNULL(mem_addr);
    mem_type = kVarAutoMemType;
    return SUCCESS;
  } else if (logic_addr != 0U) {
    for (const auto &iter : param.memory_infos) {
      const auto &mem_info = iter.second;
      GE_ASSERT_TRUE(mem_info.logic_memory_base >= 0);
      const uint64_t logic_begin = mem_info.memory_type == RT_MEMORY_P2P_DDR
                                       ? param.logic_mem_base + param.mem_size
                                       : static_cast<uint64_t>(mem_info.logic_memory_base);
      GE_ASSERT_TRUE(mem_info.memory_size >= 0);
      if ((logic_begin <= logic_addr) && (logic_addr < logic_begin + static_cast<uint64_t>(mem_info.memory_size))) {
        mem_addr = mem_info.memory_base + (logic_addr - logic_begin);
        mem_type = mem_info.memory_type;
        GELOGI("The logic addr:0x%" PRIx64 " matches type [%" PRIu64 "] address, logic base:0x%" PRIx64 ", size:%" PRIu64
          ", mem_addr:%p", logic_addr, mem_type, logic_begin, mem_info.memory_size, mem_addr);
        return SUCCESS;
      }
    }
    mem_addr = nullptr;
    REPORT_INNER_ERR_MSG("E19999", "Check param logic addr:0x%" PRIx64 " abnormal", static_cast<uint64_t>(logic_addr));
    GELOGE(PARAM_INVALID, "[Check][Param] The logic addr:0x%" PRIx64 " is abnormal", logic_addr);
    return PARAM_INVALID;
  } else {
    GELOGW("The logic addr is:0x%" PRIx64 ", base:0x%" PRIx64 ", size:%" PRIu64,
      logic_addr, param.logic_var_base, param.var_size);
  }

  mem_addr = PtrAdd<uint8_t>(static_cast<uint8_t *>(runtime_base_addr), static_cast<size_t>(max_logic_offset),
                             static_cast<size_t>(logic_addr));
  GELOGI("The logic addr:0x%" PRIx64 " matches type [%" PRIu64 "] address, mem_addr:%p", logic_addr, mem_type, mem_addr);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set device.
/// @return Status
///
Status ModelUtils::SetDevice(const uint32_t device_id) {
  GE_ASSERT_TRUE(device_id != kInvalidDeviceId);
  GE_ASSERT_RT_OK(rtSetDevice(static_cast<int32_t>(device_id)), "Call rtSetDevice failed, device_id:%u", device_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Reset device.
/// @return Status
///
Status ModelUtils::ResetDevice(const uint32_t device_id) {
  const rtError_t rt_ret = rtDeviceReset(static_cast<int32_t>(device_id));
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtSetDevice failed, device_id:%u, ret:%d", device_id, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtSetDevice] failed, device_id:%u, ret:%d", device_id, rt_ret);
    return RT_FAILED;
  }
  return SUCCESS;
}

Status ModelUtils::CalculateAicpuBlockingEventNum(const GeModelPtr &ge_model, uint32_t &aicpu_blocking_event_num) {
  const auto compute_graph = ge_model->GetGraph().get();
  GE_CHECK_NOTNULL(compute_graph);

  std::unordered_set<int64_t> stream_bloking;
  for (const auto &node : compute_graph->GetAllNodes()) {
    bool is_blocking_aicpu_op = false;
    const OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op);
    if (!is_blocking_aicpu_op) {
      continue;
    }

    GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc->GetName().c_str(),
           static_cast<int32_t>(is_blocking_aicpu_op));
    (void)stream_bloking.insert(op_desc->GetStreamId());
  }

  aicpu_blocking_event_num = static_cast<uint32_t>(stream_bloking.size());
  return SUCCESS;
}

Status ModelUtils::CalculateHcclGroupOrderedEventNum(const GeModelPtr &ge_model,
  uint32_t &hccl_group_ordered_event_num) {
  const auto &model_def = ge_model->GetModelTaskDefPtr();
  GE_CHECK_NOTNULL(model_def);
  const auto compute_graph = ge_model->GetGraph().get();
  GE_CHECK_NOTNULL(compute_graph);

  std::unordered_set<std::string > hccl_group_id_set;
  for (const auto &node : compute_graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    std::vector<std::string> hccl_group_id_list;
    const bool has_gropu_id_list = (AttrUtils::GetListStr(op_desc, ATTR_NAME_HCCL_GROUP_ID_LIST, hccl_group_id_list) &&
      !hccl_group_id_list.empty());
    if (has_gropu_id_list) {
      for (const auto &group_id : hccl_group_id_list) {
        GELOGI("Get op:%s attribute(hccl_group_id_list) group id:%s", op_desc->GetName().c_str(), group_id.c_str());
        hccl_group_id_set.insert(group_id);
      }
    }
  }

  hccl_group_ordered_event_num = static_cast<uint32_t>(hccl_group_id_set.size());
  return SUCCESS;
}

Status ModelUtils::CalculateFollowStream(const GeModelPtr &ge_model, uint64_t &hccl_fellow_stream_num) {
  const auto &model_def = ge_model->GetModelTaskDefPtr();
  GE_CHECK_NOTNULL(model_def);
  const auto compute_graph = ge_model->GetGraph().get();
  GE_CHECK_NOTNULL(compute_graph);

  std::map<uint32_t, OpDescPtr> op_list;
  for (const auto &node : compute_graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    (void)op_list.emplace(op_desc->GetId(), op_desc);
  }

  std::multimap<int64_t, uint64_t> main_follow_num;
  for (int32_t i = 0; i < model_def->task_size(); i++) {
    const domi::TaskDef &task = model_def->task(i);
    if (static_cast<ModelTaskType>(task.type()) == ModelTaskType::MODEL_TASK_HCCL) {
      auto const &hccl_def = task.kernel_hccl();
      const auto it = op_list.find(hccl_def.op_index());
      GE_CHK_BOOL_RET_STATUS(it != op_list.end(), FAILED, "Failed to find op index %u in op list",
                             hccl_def.op_index());
      const OpDescPtr hccl_op_desc = it->second;
      int64_t main_stream_id = hccl_op_desc->GetStreamId();
      int64_t follow_stream_num = 0;
      if (!AttrUtils::GetInt(hccl_op_desc, kUsedStreamNum, follow_stream_num)) {
        GELOGW("Get used stream num failed, op is %s", hccl_op_desc->GetName().c_str());
      }
      (void)main_follow_num.emplace(main_stream_id, follow_stream_num);
    }
  }
  hccl_fellow_stream_num = CalFollowStreamSum(main_follow_num);
  return SUCCESS;
}

uint64_t ModelUtils::CalFollowStreamSum(const std::multimap<int64_t, uint64_t> &hccl_stream_map) {
  std::map<int64_t, uint64_t> max_follow_stream_map;
  for (const auto &it : hccl_stream_map) {
    const std::map<int64_t, uint64_t>::const_iterator max_it =
        static_cast<std::map<int64_t, uint64_t>::const_iterator>(max_follow_stream_map.find(it.first));
    if (max_it == max_follow_stream_map.cend()) {
      (void)max_follow_stream_map.emplace(it.first, it.second);
      continue;
    }
    if (it.second > max_it->second) {
      max_follow_stream_map.at(max_it->first) = it.second;
    }
  }
  uint64_t need_follow_stream_num = 0U;
  for (const auto &follow_it : max_follow_stream_map) {
    need_follow_stream_num = need_follow_stream_num + follow_it.second;
  }
  GELOGD("Need follow num is %" PRIu64, need_follow_stream_num);
  return need_follow_stream_num;
}

bool ModelUtils::IsReuseZeroCopyMemory() {
  static const std::string kEnabled = "1";
  std::string reuse_zero_copy_memory;
  (void)ge::GetContext().GetOption(OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, reuse_zero_copy_memory);
  return (reuse_zero_copy_memory == kEnabled);
}

bool ModelUtils::IsGeUseExtendSizeMemory(bool dynamic_graph) {
  return VarManager::IsGeUseExtendSizeMemory(dynamic_graph);
}

vector_bit_t ModelUtils::GetInputTensorNeedRawData(const OpDescPtr &op_desc) {
  vector_bit_t need_raw_data_list;
  for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
    bool is_no_tiling = false;
    (void)ge::AttrUtils::GetBool(input_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
    need_raw_data_list.push_back(!is_no_tiling);
  }
  return need_raw_data_list;
}

Status ModelUtils::InitRuntimeParams(const GeModelPtr &ge_model, RuntimeParam &runtime_param,
                                     const uint32_t device_id) {
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, runtime_param.mem_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, runtime_param.weight_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_STREAM_NUM, runtime_param.stream_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, runtime_param.notify_num);
  (void)AttrUtils::GetListInt(ge_model, ATTR_MODEL_NOTIFY_TYPES, runtime_param.notify_types);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_EVENT_NUM, runtime_param.event_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_LABEL_NUM, runtime_param.label_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_BATCH_NUM, runtime_param.batch_num);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, runtime_param.logic_mem_base);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, runtime_param.logic_weight_base);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_SESSION_ID, runtime_param.session_id);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, runtime_param.logic_var_base);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_VAR_SIZE, runtime_param.var_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, runtime_param.zero_copy_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, runtime_param.host_mem_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, runtime_param.host_logic_mem_base);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, runtime_param.host_svm_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, runtime_param.host_svm_logic_mem_base);
  runtime_param.device_id = device_id;

  // 多次init调用的时候，先清空
  runtime_param.fm_memory_infos.clear();
  runtime_param.fixed_fm_memory_infos.clear();
  runtime_param.memory_infos.clear();

  // 外部传入fix mem base时，fix地址优先功能生效
  bool is_fixed_prior_fm = (runtime_param.fixed_mem_base != 0U);
  GELOGD("runtime_param.fixed_mem_base:0x%" PRIx64 ", is_fixed_prior_fm:%d",
    runtime_param.fixed_mem_base, is_fixed_prior_fm);

  int64_t total_hbm_size = 0;
  const auto &memory_info_vec = GetAllMemoryTypeSize(ge_model);
  for (auto &i : memory_info_vec) {
    if (i.memory_type == RT_MEMORY_HBM) {
      if (is_fixed_prior_fm && i.is_fixed_addr_prior) {
        runtime_param.fixed_fm_memory_infos.push_back(i);
      } else {
        runtime_param.fm_memory_infos.push_back(i);
      }

      total_hbm_size += i.memory_size;
      continue;
    }
    runtime_param.memory_infos[i.memory_type] = i;
  }
  GE_ASSERT_EQ(ge::IntegerChecker<int64_t>::Compat(runtime_param.mem_size), true);
  GE_ASSERT_EQ(total_hbm_size, (static_cast<int64_t>(runtime_param.mem_size) - runtime_param.zero_copy_size));
  runtime_param.fileconstant_addr_mapping.clear();
  return SUCCESS;
}

Status ModelUtils::GetHbmFeatureMapMemInfo(const GeModelPtr &ge_model, std::vector<MemInfo> &all_mem_info,
                                           bool get_zero_copy) {
  std::vector<std::vector<int64_t>> sub_memory_infos;
  (void)AttrUtils::GetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_memory_infos);
  if (sub_memory_infos.empty()) {
    MemInfo default_mem_info{};
    int64_t zero_copy_size = 0;
    (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, default_mem_info.memory_size);
    (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size);
    default_mem_info.memory_size -= zero_copy_size;
    default_mem_info.memory_type = RT_MEMORY_HBM;
    GELOGD("Get feature map memory info with details: [%s]", default_mem_info.ToString().c_str());
    all_mem_info.emplace_back(std::move(default_mem_info));
    return SUCCESS;
  }

  const size_t fm_memory_info_size = sub_memory_infos.size() - 1U;  // last is zero copy mem info
  for (size_t index = 0; index < sub_memory_infos.size(); ++index) {
    if ((index == (fm_memory_info_size)) && (!get_zero_copy)) {
      continue;
    }
    const auto &sub_memory_info = sub_memory_infos[index];
    // memory_type, logic_memory_base, memory_size
    GE_ASSERT_TRUE(sub_memory_info.size() >= 3U);
    GE_ASSERT_EQ(sub_memory_info[0U], static_cast<int64_t>(RT_MEMORY_HBM));
    MemInfo one_fm_mem_info;
    one_fm_mem_info.memory_type = RT_MEMORY_HBM;
    one_fm_mem_info.logic_memory_base = sub_memory_info[1U];
    one_fm_mem_info.memory_size = sub_memory_info[2U];
    one_fm_mem_info.memory_base = reinterpret_cast<uint8_t *>(one_fm_mem_info.logic_memory_base);
    one_fm_mem_info.is_fixed_addr_prior = ((sub_memory_info.size() > 3U) ? sub_memory_info[3U] : false);
    GELOGD("Get one sub feature map memory info with details: [%s]", one_fm_mem_info.ToString().c_str());
    all_mem_info.emplace_back(std::move(one_fm_mem_info));
  }
  std::sort(all_mem_info.begin(), all_mem_info.end());
  return SUCCESS;
}

std::vector<MemInfo> ModelUtils::GetAllMemoryTypeSize(const GeModelPtr &ge_model) {
  std::vector<MemInfo> all_mem_info;
  GE_ASSERT_SUCCESS(GetHbmFeatureMapMemInfo(ge_model, all_mem_info));

  MemInfo p2p_mem_info{};
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_mem_info.memory_size);
  p2p_mem_info.memory_type = RT_MEMORY_P2P_DDR;
  p2p_mem_info.memory_key = "_p";
  all_mem_info.emplace_back(std::move(p2p_mem_info));

  MemInfo session_scope_mem_info{};
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, session_scope_mem_info.memory_size);
  session_scope_mem_info.memory_type = (kSessionScopeMemoryMask | RT_MEMORY_HBM);
  all_mem_info.emplace_back(std::move(session_scope_mem_info));

  MemInfo host_mem_info{};
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, host_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, host_mem_info.logic_memory_base);
  host_mem_info.memory_type = RT_MEMORY_HOST;
  host_mem_info.memory_key = "_h";
  all_mem_info.emplace_back(std::move(host_mem_info));

  MemInfo host_svm_mem_info{};
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, host_svm_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, host_svm_mem_info.logic_memory_base);
  host_svm_mem_info.memory_type = RT_MEMORY_HOST_SVM;
  host_svm_mem_info.memory_key = "_svm";
  all_mem_info.emplace_back(std::move(host_svm_mem_info));
  return all_mem_info;
}

Status ModelUtils::MallocExMem(const uint32_t device_id, RuntimeParam &runtime_param) {
  for (auto &it : runtime_param.memory_infos) {
    const size_t mem_size = static_cast<size_t>(it.second.memory_size);
    if (mem_size == 0U) {
      continue;
    }
    const bool sessoion_scope = ((kSessionScopeMemoryMask & it.first) == kSessionScopeMemoryMask);
    rtMemType_t memory_type = static_cast<rtMemType_t>(it.first & kMemoryTypeMask);
    const rtMemType_t  mem_type_from_infos = static_cast<rtMemType_t>(it.second.memory_type);
    if (mem_type_from_infos == RT_MEMORY_HOST) {
      memory_type = mem_type_from_infos;
      it.second.memory_base = PtrToPtr<void, uint8_t>(MemManager::Instance().HostMemInstance().Malloc(mem_size));
      runtime_param.host_mem_base = PtrToValue(it.second.memory_base);
    } else if (mem_type_from_infos == RT_MEMORY_HOST_SVM) {
      memory_type = mem_type_from_infos;
      it.second.memory_base =
          PtrToPtr<void, uint8_t>(MemoryAllocator(RT_MEMORY_HOST_SVM).MallocMemory(it.second.memory_key, mem_size));
      runtime_param.host_svm_mem_base = PtrToValue(it.second.memory_base);
    } else if (sessoion_scope) {
      auto &mem_instance = MemManager::Instance().SessionScopeMemInstance(memory_type);
      it.second.memory_base = mem_instance.Malloc(mem_size, runtime_param.session_id);
    } else if ((memory_type == RT_MEMORY_P2P_DDR) && (runtime_param.p2p_fixed_mem_base != 0U)) {
      GE_ASSERT_TRUE(runtime_param.p2p_fixed_mem_size >= mem_size,
                     "runtime_param.p2p_fixed_mem_size: %zu, mem_size:%zu", runtime_param.p2p_fixed_mem_size, mem_size);
      it.second.memory_base = PtrToPtr<void, uint8_t>(ValueToPtr(runtime_param.p2p_fixed_mem_base));
      GELOGI(
          "graph_%u use p2p_fixed_mem_base, mem_type[RT_MEMORY_P2P_DDR] "
          "mem_type_origin[%" PRIu64 "] mem_addr[%p] size[%zu].",
          runtime_param.graph_id, it.first, it.second.memory_base, mem_size);
      continue;
    } else {
      const std::string purpose = MemTypeUtils::ToString(memory_type);
      auto &mem_instance = MemManager::Instance().MemInstance(memory_type);
      it.second.memory_base = mem_instance.MallocMemory(purpose, mem_size, device_id);
    }
    if (it.second.memory_base == nullptr) {
      GELOGW("Alloc extra memory failed, type:%u size: %zu", memory_type, mem_size);
    }
    GEEVENT(
        "[IMAS]InitFeatureMap graph_%u MallocMemory type[F] mem_type[%u] "
        "mem_type_origin[%" PRIu64 "] mem_addr[%p] size[%zu].",
        runtime_param.graph_id, memory_type, it.first, it.second.memory_base, mem_size);
  }
  return SUCCESS;
}

void ModelUtils::FreeExMem(const uint32_t device_id, RuntimeParam &runtime_param,
                           const uint64_t session_id, const bool is_online) {
  for (auto &it : runtime_param.memory_infos) {
    if ((kSessionScopeMemoryMask & it.first) == kSessionScopeMemoryMask) {
      if ((!is_online) && (it.second.memory_base != nullptr)) {
        MemManager::Instance().FreeSessionMemory(session_id);  // reasoning process releases memory
        it.second.memory_base = nullptr;
      }
      continue;  // free when session destory
    }
    if (it.second.memory_type == RT_MEMORY_HOST) {
      if (it.second.memory_base != nullptr) {
        (void)MemManager::Instance().HostMemInstance().Free(PtrToPtr<uint8_t, void>(it.second.memory_base));
        it.second.memory_base = nullptr;
        continue;
      }
    }
    if (it.second.memory_type == RT_MEMORY_HOST_SVM) {
      if (it.second.memory_base != nullptr) {
        (void)MemoryAllocator(RT_MEMORY_HOST_SVM).FreeMemory(it.second.memory_base);
        it.second.memory_base = nullptr;
        continue;
      }
    }
    if (it.second.memory_type == RT_MEMORY_P2P_DDR) {
      FreeP2pMem(device_id, runtime_param, it);
      continue;
    }
    const rtMemType_t memory_type = static_cast<rtMemType_t>(it.first & kMemoryTypeMask);
    auto &mem_instance = MemManager::Instance().MemInstance(memory_type);
    if (it.second.memory_base != nullptr) {
      GE_CHK_STATUS(mem_instance.FreeMemory(it.second.memory_base, device_id), "failed to free memory");
      it.second.memory_base = nullptr;
    }
  }
}

bool ModelUtils::IsSuppoprtAddrRefreshable(const uint64_t mem_type) {
  return (mem_type == static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)) ||
         (mem_type == static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo));
}

void ModelUtils::GetAddrRefreshableFlagsByMemTypes(const std::vector<uint64_t> &mem_types,
                                                   std::vector<uint8_t> &flags) {
  for (const auto &mem_type : mem_types) {
      const bool refresh = IsSuppoprtAddrRefreshable(mem_type);
      flags.push_back(refresh ? 1U : 0U);
  }
}

bool ModelUtils::IsFeatureMapOrModelIoType(const uint64_t mem_type) {
  return ((mem_type == kFmMemType) ||
          (mem_type == static_cast<uint64_t>(RT_MEMORY_HBM)) || (mem_type == static_cast<uint64_t>(RT_MEMORY_L2)) ||
          (mem_type == static_cast<uint64_t>(RT_MEMORY_DEFAULT)));
}

Status ModelUtils::GetSpaceRegistries(const ge::GeRootModelPtr &root_model,
                                      std::shared_ptr<gert::OpImplSpaceRegistryV2Array> &space_registries) {
  GE_ASSERT_NOTNULL(root_model);
  if (space_registries == nullptr) {
    space_registries = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
    GE_ASSERT_NOTNULL(space_registries);
  }

  std::vector<OpSoBinPtr> so_list{};
  GetSpecificSoBins(root_model, SoBinType::kSpaceRegistry, so_list);
  if (!so_list.empty()) {
    std::map<OppImplVersion, std::vector<ge::OpSoBinPtr>> version_2_so_lists;
    for (const auto &so_within_om : so_list) {
      const auto &vendor_name = so_within_om->GetVendorName();
      GELOGD("find so in om vendor path %s", so_within_om->GetVendorName().c_str());
      bool found = false;
      for (const auto &vendor_path_2_version : kVersion2VendorPath) {
        if (vendor_name.find(vendor_path_2_version.second)  != std::string::npos) {
          GELOGI("Add opp version:[%zu] so from root model!", vendor_path_2_version.first);
          version_2_so_lists[vendor_path_2_version.first].emplace_back(so_within_om);
          found = true;
          break;
        }
      }
      if (!found) {
        GELOGI("vendor_name:[%s] without path identifier, and default is opp version.", vendor_name.c_str());
        version_2_so_lists[OppImplVersion::kOpp].emplace_back(so_within_om);
      }
    }

    for (const auto &version_2_so_list : version_2_so_lists) {
      GELOGI("Load opp version:[%zu] so from root model, so size = %zu!", version_2_so_list.first,
             version_2_so_list.second.size());
      auto space_registry = ge::MakeShared<gert::OpImplSpaceRegistryV2>();
      GE_ASSERT_NOTNULL(space_registry);
      {
        std::vector<gert::OppSoDesc> opp_so_desc_list;
        std::vector<std::string> opp_dir_list;
        // 在出作用域，析构opp_dir_list时一并删除创建的临时so目录
        GE_MAKE_GUARD(clear_tmp_opp_dir_list, [&opp_dir_list]() -> void {
          for (const auto& opp_dir : opp_dir_list) {
            GELOGD("clear_tmp_opp_dir_list opp dir: %s", opp_dir.c_str());
            (void)RmOmOppDir(opp_dir);
          }
        });
        for (const auto &so_bin : version_2_so_list.second) {
          GE_ASSERT_NOTNULL(so_bin);
          GE_ASSERT_TRUE(so_bin->GetBinDataSize() <= kGByteSize);
          std::string opp_dir;
          GE_ASSERT_SUCCESS(CreateOmOppDir(opp_dir));
          opp_dir_list.emplace_back(opp_dir);
          const auto &so_path = opp_dir + so_bin->GetSoName();
          GE_ASSERT_GRAPH_SUCCESS(SaveToFile(so_bin, so_path));
          opp_so_desc_list.emplace_back(gert::OppSoDesc(std::vector<ge::AscendString>{ge::AscendString(so_path.c_str())},
                                                        so_bin->GetSoName().c_str()));
        }
        // 因子包中nn对common_legacy有依赖，需先全部落盘再加载
        for (const auto &opp_so_desc : opp_so_desc_list) {
          GELOGD("Prepare to AddSoToRegistry, so name: %s", opp_so_desc.GetPackageName().GetString());
          const auto is_opp_depend_common_so = opp_so_desc.GetPackageName().Find("libophost_comm_legacy.so");
          // 正式方案需由aoe整改，当前规避方案由GE跳过加载
          if (is_opp_depend_common_so != std::string::npos) {
            GELOGD("AddSoToRegistry skiped, the specify so: libophost_comm_legacy.so");
            continue;
          }
          GE_ASSERT_GRAPH_SUCCESS(space_registry->AddSoToRegistry(opp_so_desc));
        }
      }
      space_registries->at(static_cast<size_t>(version_2_so_list.first)) = space_registry;
    }
  } else {
    GELOGI("No space registries in root model, use default so!");
    // for 1911 minirc and helper, space_registry is null
    GE_ASSERT_TRUE(space_registries->size() == static_cast<size_t>(gert::OppImplVersionTag::kVersionEnd));
    for (size_t i = 0U; i < static_cast<size_t>(gert::OppImplVersionTag::kVersionEnd); i++) {
      auto space_registry =
          gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(i));
      if (space_registry == nullptr) {
        continue;
      }
      space_registries->at(i) = space_registry;
    }
  }

  return ge::SUCCESS;
}

bool ModelUtils::IsAICoreKernel(const ge::ccKernelType kernel_type) {
  static std::set<ge::ccKernelType> aicore_kernel_type{ge::ccKernelType::TE, ge::ccKernelType::MIX_AICORE,
                                                       ge::ccKernelType::MIX_VECTOR_CORE};
  return aicore_kernel_type.count(kernel_type) > 0UL;
}

std::string ModelUtils::GetOpMasterDeviceKey(const uint32_t &model_id, const std::string &so_path) {
  if (so_path.find(kInner) != std::string::npos) {
    const auto &pos = so_path.find_last_of("/");
    GE_ASSERT_TRUE(pos != std::string::npos);
    return so_path.substr(pos + 1UL);
  } else {
    return GetOpMasterDeviceCustKey(model_id, so_path);
  }
}

// 自定义算子命名规则： model_id + "vendors" + 厂商名称 + so名称
std::string ModelUtils::GetOpMasterDeviceCustKey(const uint32_t &model_id, const std::string &so_path) {
  const auto vendor_pos = so_path.find(kVendors);
  GE_ASSERT_TRUE(vendor_pos != std::string::npos);
  const auto op_impl_pos = so_path.find(kOpImpl);
  GE_ASSERT_TRUE(op_impl_pos != std::string::npos);
  const auto so_pos = so_path.find_last_of("/");
  GE_ASSERT_TRUE(so_pos != std::string::npos);
  auto cust_key = std::to_string(model_id)
                      .append("/")
                      .append(so_path.substr(vendor_pos, op_impl_pos - vendor_pos))
                      .append(so_path.substr(so_pos));
  const auto &new_end = std::unique(cust_key.begin(), cust_key.end(), [](const char_t c1, const char_t c2) -> bool {
    return (c1 == '/') && (c2 == '/');
  });
  cust_key.erase(new_end, cust_key.end());
  std::replace(cust_key.begin(), cust_key.end(), '/', '_');
  GELOGD("Get op master device cust key is %s.", cust_key.c_str());
  return cust_key;
}

Status ModelUtils::GetOpMasterDevice(const uint32_t &model_id, const ge::GeRootModelPtr &root_model,
                                     std::unordered_map<std::string, OpSoBinPtr> &built_in_so_bins,
                                     std::unordered_map<std::string, OpSoBinPtr> &cust_so_bins) {
  GE_ASSERT_NOTNULL(root_model);
  std::vector<OpSoBinPtr> so_list{};
  GetSpecificSoBins(root_model, SoBinType::kOpMasterDevice, so_list);
  if (!so_list.empty()) {
    for (const auto &op_so_bin : so_list) {
      const auto &path = op_so_bin->GetVendorName();
      const auto &name = op_so_bin->GetSoName();
      GELOGD("find so in om vendor path %s, so name %s", path.c_str(), name.c_str());
      // 兼容老流程
      if (path.find(kInner) != std::string::npos) {
        built_in_so_bins.emplace(name, op_so_bin);
        GELOGI("[OpMasterDevice][BuiltIn]Get so [%s] in model [%u].", name.c_str(), model_id);
      }
      const auto &cust_key = GetOpMasterDeviceKey(model_id, path + "/" + name);
      cust_so_bins.emplace(cust_key, op_so_bin);
      GELOGI("[OpMasterDevice][Custom]Get so [%s]->[%s] in model [%u].",
          name.c_str(), cust_key.c_str(), model_id);
    }
  } else {
    GE_ASSERT_SUCCESS(GetOpMasterDeviceFromOppPackage(model_id, built_in_so_bins, cust_so_bins));
  }
  GELOGI("[OpMasterDevice]Get so num BuiltIn:[%zu] and Custom:[%zu].", built_in_so_bins.size(), cust_so_bins.size());
  return SUCCESS;
}

Status ModelUtils::GetOpMasterDeviceFromOppPackage(const uint32_t &model_id,
                                                   std::unordered_map<std::string, OpSoBinPtr> &built_in_so_bins,
                                                   std::unordered_map<std::string, OpSoBinPtr> &cust_so_bins) {
  std::string op_master_device_path;
  GE_ASSERT_SUCCESS(PluginManager::GetOpMasterDeviceSoPath(op_master_device_path));
  std::vector<std::string> path_vec;
  PluginManager::SplitPath(op_master_device_path, path_vec);
  for (const auto &path : path_vec) {
    const auto &is_built_in = path.find(kInner) != std::string ::npos;
    std::vector<std::string> file_list;
    PluginManager::GetFileListWithSuffix(path, ".so", file_list);
    for (const auto &file : file_list) {
      uint32_t bin_len = 0U;
      auto op_so_bin = GetBinDataFromFile(file, bin_len);
      GE_ASSERT_NOTNULL(op_so_bin, "open so fail, path=%s", file.c_str());
      const auto &pos = file.find_last_of("/");
      GE_ASSERT_TRUE(pos != std::string::npos);
      const auto &so_name = file.substr(pos + 1UL);
      const auto &vendor_name = file.substr(0, pos);
      const auto proto_bin =
          ge::MakeShared<OpSoBin>(so_name, vendor_name, std::move(op_so_bin), bin_len, SoBinType::kOpMasterDevice);
      GE_ASSERT_NOTNULL(proto_bin);
      // 兼容老cann包
      if (is_built_in) {
        built_in_so_bins.emplace(so_name, proto_bin);
        GELOGI("[OpMasterDevice][BuiltIn]Get so [%s] from opp path.", so_name.c_str());
      }
      const auto &cust_key = GetOpMasterDeviceKey(model_id, file);
      cust_so_bins.emplace(cust_key, proto_bin);
      GELOGI("[OpMasterDevice][Custom] Get so [%s]->[%s] from opp path.",
          so_name.c_str(), cust_key.c_str());
    }
  }
  return SUCCESS;
}
}  // namespace ge
