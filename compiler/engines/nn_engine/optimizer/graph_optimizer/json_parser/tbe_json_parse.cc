/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/json_parser/tbe_json_parse.h"
#include <fstream>
#include <memory>
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/aicore_util_constants.h"
#include "common/string_utils.h"
#include "common/configuration.h"
#include "common/util/json_util.h"
#include "common/aicore_util_attr_define.h"
#include "common/op_tensor_utils.h"
#include "common/fe_context_utils.h"
#include "common/platform_utils.h"
#include "graph/op_kernel_bin.h"
#include "graph/tuning_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "register/op_ext_gentask_registry.h"
#include "graph/utils/op_type_utils.h"
#include "common/fe_gentask_utils.h"

namespace fe {
namespace {
const std::string kStageParseTvmMgc = "[SubGraphOpt][ParseJson][ParseTvmMgc]";
const std::unordered_map<std::string, std::string> kCoreTypeMapping {
    {VECTOR_CORE_TYPE, kCoreTypeAIV},
    {CUBE_CORE_TYPE, kCoreTypeAIC},
    {AI_CORE_TYPE, kCoreTypeAIC},
    {kCoreTypeMixVectorCore, kCoreTypeMixVectorCore},
    {kCoreTypeMixAICore, kCoreTypeMixAICore}
};
const std::string kShortSocA2 = "Ascend910B";
constexpr char const *kMemoryCheckKey = "oom";

bool IsThirdClassOp(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return false;
  }
  int32_t unknown_shape_type_val = 0;
  (void) ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  return static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val) == ge::DEPEND_SHAPE_RANGE;
}

bool IsCustomGenTaskOp(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return false;
  }
  if (PlatformUtils::Instance().GetShortSocVersion() != kShortSocA2 &&
      (OpExtGenTaskRegistry::GetInstance().GetExtTaskType(op_desc->GetType()) == ExtTaskType::kAicoreTask)) {
    FE_LOGI("Op[%s,%s] would generate an aicore taskdef instead of ffts+.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  // 这些算子自行定义了gentask方法，依赖ffts+流程，注册关键字IMPL_OP_CT
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(ge::OppImplVersion::kOpp));
  bool has_space_registry_func = false;
  uint32_t impl_version = 0;
  if (space_registry != nullptr) {
    auto space_registry_func = space_registry->GetOpImpl(op_desc->GetType().c_str());
    has_space_registry_func = space_registry_func != nullptr && space_registry_func->gen_task != nullptr;
    if (has_space_registry_func) { impl_version = space_registry_func->version; }
  }
  if (has_space_registry_func && impl_version > DEFAULT_OP_IMPL_MAIN_VERSION) {
    FE_LOGI("Op[%s,%s]IMPL version = %zu, would generate aicore taskdef",
            op_desc->GetNamePtr(), op_desc->GetTypePtr(), impl_version);
    return false;
  }
  auto register_func = OpExtGenTaskRegistry::GetInstance().FindRegisterFunc(op_desc->GetType());
  if(has_space_registry_func == false && register_func == nullptr) {
    return false;
  }
  FE_LOGI("Op[%s,%s] is a custom gen task op, space_registry[%d] register_func[%d], using ffts+", op_desc->GetNamePtr(),
          op_desc->GetTypePtr(), has_space_registry_func, register_func != nullptr);
  return true;
}
} // namespace

TbeJsonFileParse::TbeJsonFileParse(ge::Node& node) : node_(node), op_desc_(node.GetOpDesc()) {
  json_parser_impl_ = std::unique_ptr<TbeJsonFileParseImpl>(new (std::nothrow) TbeJsonFileParseImpl());
  ffts_related_thread_nodes_ =
      op_desc_->TryGetExtAttr<std::shared_ptr<std::vector<ge::NodePtr>>>(kAttrRelatedThreadsNodes, nullptr);
}

bool TbeJsonFileParse::SetRelatedNodesListInt(const std::string &attr_name, const std::vector<int64_t> &value) {
  bool ret = ge::AttrUtils::SetListInt(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetListInt(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesListInt(const std::string &attr_name, const std::vector<int32_t> &value) {
  bool ret = ge::AttrUtils::SetListInt(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetListInt(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesListStr(const std::string &attr_name, const std::vector<string> &value) {
  bool ret = ge::AttrUtils::SetListStr(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetListStr(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesInt(const std::string &attr_name, const int64_t &value) {
  bool ret = ge::AttrUtils::SetInt(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetInt(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesBool(const std::string &attr_name, const bool &value) {
  bool ret = ge::AttrUtils::SetBool(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetBool(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesStr(const string &attr_name, const string &value) {
  bool ret = ge::AttrUtils::SetStr(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetStr(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::ClearRelatedNodesStr([[maybe_unused]] const string &attr_name) {
  bool ret = op_desc_->DelAttr(GetAttrPrefix() + ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ele->GetOpDesc()->DelAttr(GetAttrPrefix() + ATTR_NAME_KERNEL_LIST_FIRST_NAME);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesStrPrefixWithOpName(const string &prefix,
                                                          const string &attr_name, const string &value) {
  bool ret = ge::AttrUtils::SetStr(op_desc_, prefix + op_desc_->GetName() + attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetStr(ele->GetOpDesc(), prefix + ele->GetName() + attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesBytes(const string &attr_name, const ge::Buffer &value) {
  bool ret = ge::AttrUtils::SetBytes(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetBytes(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesListDataType(const string &attr_name,
                                                   const std::vector<ge::DataType> &value) {
  bool ret = ge::AttrUtils::SetListDataType(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetListDataType(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

bool TbeJsonFileParse::SetRelatedNodesListFloat(const string &attr_name, const std::vector<float> &value) {
  bool ret = ge::AttrUtils::SetListFloat(op_desc_, attr_name, value);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ret &= ge::AttrUtils::SetListFloat(ele->GetOpDesc(), attr_name, value);
    }
  }
  return ret;
}

void TbeJsonFileParse::SetRelatedNodesWorkspace(const std::vector<int64_t> &tvm_workspace_sizes) {
  op_desc_->SetWorkspaceBytes(tvm_workspace_sizes);
  if (ffts_related_thread_nodes_ != nullptr) {
    for (auto &ele : *ffts_related_thread_nodes_) {
      ele->GetOpDesc()->SetWorkspaceBytes(tvm_workspace_sizes);
    }
  }
}

Status TbeJsonFileParse::ParseTvmBlockDim() {
  int32_t block_dim = 1;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyBlockDim, block_dim, block_dim) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmBlkDim] Failed to get the block dimension for node[%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if ((block_dim < -1) || (block_dim > kMaxBlockDim)) {
    FE_LOGE("Op[%s], blockDim[%d] is out of range, range is (-1, 65535).", op_desc_->GetName().c_str(), block_dim);
    return FAILED;
  }

  if (block_dim == -1) {
    block_dim = 1;
  }
  FE_LOGD("Op[%s], ParseTvmBlockDim: %d.", op_desc_->GetName().c_str(), block_dim);
  SetRelatedNodesInt(ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseSuperKernelSupportFlag() {
  int64_t sk_supp = -1;
  if (json_parser_impl_->ParseJsonAttr(false, kSupportSuperKernel, sk_supp, sk_supp) == FAILED) {
    REPORT_FE_ERROR("[ParseJson][ParseSuperKernelSupportFlag] Get supportSuperKernel for node[%s] failed.",
                    op_desc_->GetNamePtr());
    return FAILED;
  }
  FE_LOGD("Op[%s], parsing supportSuperKernel flag: %d.", op_desc_->GetNamePtr(), sk_supp);
  if (sk_supp != -1) {
    SetRelatedNodesInt(kSupportSuperKernel, sk_supp);
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseLocalMemSize() {
  int64_t local_mem_size = 0;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyLocalMemSize, local_mem_size, local_mem_size) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseLocalMemSize] Failed to get the local memory size for node[%s].",
                    op_desc_->GetNamePtr());
    return FAILED;
  }
  if (local_mem_size <= 0 || local_mem_size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    FE_LOGD("Op[%s], local_mem_size[%ld] is invalid.", op_desc_->GetNamePtr(), local_mem_size);
    return SUCCESS;
  }
  uint32_t real_size = static_cast<uint32_t>(local_mem_size);
  FE_LOGD("Op[%s], local memory size: %u.", op_desc_->GetNamePtr(), real_size);
  SetRelatedNodesInt(kLocalMemorySize, real_size);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseScheduleMode() {
  int64_t schedule_mode = 0;
  (void)json_parser_impl_->ParseJsonAttr(false, kKeyScheduleMode, static_cast<int64_t>(0), schedule_mode);
  if (schedule_mode < 0 || schedule_mode > UINT32_MAX) {
    REPORT_FE_ERROR("Schedule mode [%ld] from op [%s, %s]'s json is invalid.",
                    schedule_mode, op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  if (schedule_mode > 0) {
    (void)SetRelatedNodesInt(kAttrScheduleMode, schedule_mode);
    FE_LOGD("Set schedule mode attr[%ld] for op[%s, %s].",
            schedule_mode, op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseBatchBindOnly() {
  int32_t batch_bind_only = 0;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyBatchBindOnly, batch_bind_only, batch_bind_only) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseBtcBindOnly] Failed to get batch_bind_only for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  FE_LOGD("Op[%s], Parsing batch_bind_only[%d].", op_desc_->GetName().c_str(), batch_bind_only);
  if (!SetRelatedNodesBool(ge::ATTR_N_BATCH_SPILT, batch_bind_only)) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseBtcBindOnly] Failed to set attribute _is_n_batch_split for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmMagic() {
  std::string magic{"RT_DEV_BINARY_MAGIC_ELF"};
  if (json_parser_impl_->ParseJsonAttr(true, kKeyMagic, magic, magic) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmMgc] Failed to get the magic value for node[%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  auto iter = std::find(kBinaryMagicTypesVec.begin(), kBinaryMagicTypesVec.end(), magic);
  if (iter == kBinaryMagicTypesVec.end()) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmMgc] The magic value [%s] is not valid.", magic.c_str());
    REPORT_FE_ERROR("%s Only support RT_DEV_BINARY_MAGIC_(ELF_AICPU/ELF/ELF_AIVEC/ELF_AICUBE/ELF_MIX_AIC/ELF_MIX_AIV).",
                    kStageParseTvmMgc.c_str());
    return FAILED;
  }
  (void)SetRelatedNodesStr(ge::TVM_ATTR_NAME_MAGIC, magic);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmOldCoreType() {
  std::string core_type;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyOldCoreType, core_type, core_type) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmOldCoreType] Failed to get the core type for node[%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  if (core_type.empty()) {
    return SUCCESS;
  }

  if ((core_type != "AIC" && core_type != "AIV")) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmOldCoreType]Core type %s is invalid, which should be AIC or AIV.",
                    core_type.c_str());
    return FAILED;
  }
  (void)SetRelatedNodesStr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  return SUCCESS;
}

void TbeJsonFileParse::ProcMixCoreType() {
  bool tile_fwk_op_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_, kAttrTileFwkOpStr, tile_fwk_op_flag);
  if (tile_fwk_op_flag) {
    FE_LOGI("Op[%s] is tile fwk op, use aicore.", op_desc_->GetNamePtr());
    return;
  }
  bool is_mix_aic = true;
  if ((cube_ratio_ == 0) || (vector_ratio_ != 0 && cube_ratio_ > vector_ratio_)) {
    (void)SetRelatedNodesBool(kMixIsAiv, true); // for mix profiling
    is_mix_aic = false;
  }
  if (PlatformUtils::Instance().GetFftsMode() != FFTS_MODE_FFTS_PLUS) {
    (void)SetRelatedNodesBool(kFftsplusTask, true);
    FE_LOGI("Op[%s] has core_type as MIX, with is_mix_aic[%d].", op_desc_->GetNamePtr(), is_mix_aic);
    return;
  }
  bool ffts_node = false;
  (void)ge::AttrUtils::GetBool(op_desc_, kTypeFFTSPlus, ffts_node);
  if (!ffts_node && !IsCustomGenTaskOp(op_desc_) && !CheckTilingSink(node_)) {
    (void)SetRelatedNodesBool(kFftsplusTask, true); // for mix profiling
    FE_LOGI("Op[%s] has core_type as MIX, is_mix_aic[%d], platform supports ffts+, using aicore",
            op_desc_->GetNamePtr(), is_mix_aic);
    return; // return core_type=MIX
  }
  attr_prefix_ = kTbeMixEnhancedPrefix;
  (void)SetRelatedNodesListStr(kKernelNamesPrefix, {kTbeMixEnhancedPrefix});
  if (!is_mix_aic) {
    (void)SetRelatedNodesStr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixAIV);
  } else {
    (void)SetRelatedNodesStr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixAIC);
  }
  (void)SetRelatedNodesBool(kMixEnhancedKernel, true);
  (void)ge::AttrUtils::GetBool(op_desc_, kTypeFFTSPlus, ffts_node);
  if (!ffts_node) {
    (void)SetRelatedNodesStr(ATTR_NAME_ALIAS_ENGINE_NAME, "ffts_plus");
  }
  FE_LOGI("Op[%s] has core_type as MIX, is_mix_aic[%d], platform supports ffts+, using ffts+",
          op_desc_->GetNamePtr(), is_mix_aic);
  return;
}

Status TbeJsonFileParse::ParseTvmCoreType() {
  std::string core_type;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyCoreType, core_type, core_type) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmCoreType] Failed to get the core_type for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("Core type from json file of op[%s] is [%s].", op_desc_->GetNamePtr(), core_type.c_str());
  if (core_type.empty()) {
    return SUCCESS;
  }
  const auto iter = kCoreTypeMapping.find(core_type);
  if (iter != kCoreTypeMapping.end()) {
    (void)SetRelatedNodesStr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, iter->second);
    FE_LOGD("Set core type attr[%s] for op[%s].", iter->second.c_str(), op_desc_->GetNamePtr());
  } else if (core_type == kCoreTypeMixEnhance) {
    (void)SetRelatedNodesStr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixEnhance);
    ProcMixCoreType();
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmTaskRatio() {
  bool dyn_ratio = false;
  if (json_parser_impl_->ParseTvmTaskRatio(cube_ratio_, vector_ratio_, dyn_ratio) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmTaskRatio] Failed to get task_ratio for node[%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (cube_ratio_ == 0 && vector_ratio_ == 0) {
    return SUCCESS;
  }
  if (dyn_ratio) {
    (void)SetRelatedNodesBool(kMixDynamicRatio, true);
    TilingWithRatio tiling_ratio;
    if (json_parser_impl_->ParseListTilingRatio(tiling_ratio) == FAILED) {
      REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmTaskRatio]ParseListTaskRatio get ratio failed, node[%s].",
                      op_desc_->GetName().c_str());
      return FAILED;
    }
    if (!tiling_ratio.tiling_key_vec.empty()) {
      ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
      tiling_with_ratio.SetAttr(kDynRatioTiling, ge::GeAttrValue::CreateFrom<std::vector<std::string>>(
          tiling_ratio.tiling_key_vec));
      tiling_with_ratio.SetAttr(kDynRatioCRatio, ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(
          tiling_ratio.c_ratio_vec));
      tiling_with_ratio.SetAttr(kDynRatioVRatio, ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(
          tiling_ratio.v_ratio_vec));
      (void)ge::AttrUtils::SetNamedAttrs(op_desc_, kDynRatioAttr, tiling_with_ratio);
    }
    return SUCCESS;
  }
  int32_t ratio = 0;
  if (cube_ratio_ == 0 || vector_ratio_ == 0) {
    ratio = 0;
  } else if (cube_ratio_ > vector_ratio_) {
    ratio = static_cast<int32_t>(cube_ratio_ / vector_ratio_);
  } else {
    ratio = static_cast<int32_t>(vector_ratio_ / cube_ratio_);
  }
  (void)SetRelatedNodesInt(kTaskRadio, ratio);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmModeInArgsFirstField() {
  uint32_t mode = 0;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyModeInArgsFirstField, mode, mode) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmModeInArgs] get modeInArgsFirstField for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  FE_LOGD("Op[%s] Parse modeInArgsFirstField[%u].", op_desc_->GetName().c_str(), mode);
  if (!SetRelatedNodesInt(kModeInArgsFirstField, mode)) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmModeInArgs] Failed to set attribute modeInArgsFirstField for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmInterCoreSync() {
  int32_t inter_core_sync = 0;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyIntercoreSync, inter_core_sync, inter_core_sync) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmModeInArgs] get IntercoreSync for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  FE_LOGD("Op[%s] Parsed IntercoreSync[%d].", op_desc_->GetName().c_str(), inter_core_sync);
  if (!SetRelatedNodesBool(kAttrIntercoreSync, static_cast<bool>(inter_core_sync))) {
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmKernelList() {
  std::string kernel_list_first;
  if (json_parser_impl_->ParseTvmKernelList(kernel_list_first) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmKernList] ParseTvmKernelList get kernelList failed, node[%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (!kernel_list_first.empty()) {
    FE_LOGD("Op[%s] Set the attribute kernel_list_first", op_desc_->GetName().c_str());
    (void)SetRelatedNodesStr(GetAttrPrefix() + ATTR_NAME_KERNEL_LIST_FIRST_NAME, kernel_list_first);
  } else {
    FE_LOGD("Op[%s] Delete the attribute kernel_list_first", op_desc_->GetName().c_str());
    (void)ClearRelatedNodesStr(GetAttrPrefix() + ATTR_NAME_KERNEL_LIST_FIRST_NAME);
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmWorkSpace() {
  std::vector<int64_t> tvm_workspace_types;
  std::vector<int64_t> tvm_workspace_sizes;
  if (json_parser_impl_->ParseTvmWorkSpace(tvm_workspace_sizes, tvm_workspace_types) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmWorkSpace] Failed to get workspace for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (!tvm_workspace_sizes.empty()) {
    SetRelatedNodesWorkspace(tvm_workspace_sizes);
  }

  if (!tvm_workspace_types.empty()) {
    (void)SetRelatedNodesListInt(ge::TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmParameters() {
  std::vector<int64_t> parameters_index;
  AtomicInitInfo atomic_init_info;
  if (json_parser_impl_->ParseTvmParameters(parameters_index, atomic_init_info) == FAILED) {
  REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmParm] Failed to get parameters for node [%s].",
                  op_desc_->GetName().c_str());
  return FAILED;
  }

  if (parameters_index.empty()) {
    return SUCCESS;
  }

  return SetAtomicInfo(parameters_index, atomic_init_info);
}

Status TbeJsonFileParse::ParseTvmWspMode() {
  bool wsp_mode = false;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyWspMode, wsp_mode, wsp_mode) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmWspMode] get the wspMode for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  const string folded_mode = "folded";
  const string unfolded_mode = "unfolded";
  (void)SetRelatedNodesStr(TBE_OP_ATOMIC_WSP_MODE, wsp_mode ? folded_mode : unfolded_mode);
  return SUCCESS;
}

void TbeJsonFileParse::GetWorkspaceAtomicFlagAndOutputIndexFlag(const std::vector<int64_t> &parameters_index,
                                                                const NodeBaseInfo &node_info,
                                                                std::vector<int64_t> &output_index,
                                                                std::vector<int64_t> &workspace_index,
                                                                bool &workspace_atomic_flag,
                                                                bool &output_index_flag) const {
  bool is_third_op = IsThirdClassOp(op_desc_);
  FE_LOGD("Op[%s] third_op_flag: %d.", op_desc_->GetName().c_str(), is_third_op);
  size_t parameters_index_size = parameters_index.size();
  for (size_t i = 0; i < node_info.workspace_num; ++i) {
    size_t index = node_info.input_num + node_info.output_num + i;
    // third op: input_num + output_num + 1(addition output) + workspace
    index = is_third_op ? (index + 1) : index;
    if (index >= parameters_index_size) {
      continue;
    }
    workspace_index.emplace_back(parameters_index[index]);
    if (parameters_index[index] != 0) {
      workspace_atomic_flag = true;
    }
  }
  for (size_t i = 0; i < node_info.output_num; ++i) {
    size_t index = node_info.input_num + i;
    if (index >= parameters_index_size) {
      continue;
    }
    output_index.emplace_back(parameters_index[index]);
    if (parameters_index[index] != 0) {
      output_index_flag = true;
    }
  }
}

Status TbeJsonFileParse::SetAtomicInfo(std::vector<int64_t> &parameters_index, AtomicInitInfo &atomic_init_info) {
  // need modify
  std::vector<int64_t> output_index;
  std::vector<int64_t> workspace_index;
  NodeBaseInfo node_base_info;
  node_base_info.input_num = op_desc_->GetInputsSize();
  node_base_info.workspace_num = op_desc_->GetWorkspaceBytes().size();
  node_base_info.output_num = op_desc_->GetOutputsSize();
  node_base_info.offset_index = 0;
  uint32_t mode = 0;
  (void) ge::AttrUtils::GetInt(op_desc_, kModeInArgsFirstField, mode);
  bool inter_core_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc_, kAttrIntercoreSync, inter_core_sync);
  if (mode == 1 || inter_core_sync) {
    node_base_info.offset_index = 1;
  }
  size_t total_size = node_base_info.input_num + node_base_info.output_num + node_base_info.workspace_num;
  bool is_support_dynamic_shape = false;
  if (ge::AttrUtils::GetBool(op_desc_, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, is_support_dynamic_shape) &&
      is_support_dynamic_shape) {
    total_size = total_size + 1;
  }
  if (total_size >= parameters_index.size()) {
    size_t loss_num = total_size - parameters_index.size();
    for (size_t i = 0; i < loss_num; ++i) {
      parameters_index.emplace_back(0);
    }
  } else {
    FE_LOGD("parameters size is larger than input&output&workspace.");
    FE_LOGD("inputNum:%zu, outputNum:%zu, workspaceSize:%zu, offsetIndex:%zu, paraSize:%zu name:%s",
            node_base_info.input_num, node_base_info.output_num, node_base_info.workspace_num,
            node_base_info.offset_index, parameters_index.size(), op_desc_->GetName().c_str());
  }
  total_param_size_ = parameters_index.size() + node_base_info.offset_index;
  (void)SetRelatedNodesListInt("ub_atomic_params", parameters_index);
  // in parameters data sort as offset_index->input->output->workspace
  bool output_index_flag = false;
  bool workspace_atomic_flag = false;
  GetWorkspaceAtomicFlagAndOutputIndexFlag(parameters_index, node_base_info, output_index, workspace_index,
                                           workspace_atomic_flag, output_index_flag);

  SetAtomicInitInfo(output_index_flag, workspace_atomic_flag, output_index, workspace_index, atomic_init_info);

  return SUCCESS;
}

void TbeJsonFileParse::SetAtomicInitInfo(const bool &output_index_flag, bool &workspace_atomic_flag,
                                         std::vector<int64_t> &output_index, std::vector<int64_t> &workspace_index,
                                         AtomicInitInfo &atomic_init_info) {
  (void)SetRelatedNodesInt(TBE_OP_ATOMIC_WORKSPACE_FLAG, static_cast<int>(workspace_atomic_flag));

  if (output_index_flag) {
    (void)SetRelatedNodesListInt(TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);
  }
  if (workspace_atomic_flag) {
    (void)SetRelatedNodesListInt(TBE_OP_ATOMIC_WORKSPACE_INDEX, workspace_index);
  }

  if (!atomic_init_info.dtype_list.empty()) {
    (void)SetRelatedNodesListInt(TBE_OP_ATOMIC_DTYPES, atomic_init_info.dtype_list);
  }
  if (!atomic_init_info.init_value_int64_list.empty()) {
    (void)SetRelatedNodesListInt(TBE_OP_ATOMIC_INT64_VALUES, atomic_init_info.init_value_int64_list);
  }
  if (!atomic_init_info.init_value_float_list.empty()) {
    (void)SetRelatedNodesListFloat(TBE_OP_ATOMIC_FLOAT_VALUES, atomic_init_info.init_value_float_list);
  }
}

Status TbeJsonFileParse::ParseTvmMetaData() {
  string meta_data;
  if (json_parser_impl_->ParseTvmMetaData(meta_data) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmMetaData] get attr metadata for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  if (meta_data.empty()) {
    return SUCCESS;
  }
  (void)SetRelatedNodesStr(GetAttrPrefix() + ge::TVM_ATTR_NAME_METADATA, meta_data);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmKernelBinId() {
  string kernel_bin_id;
  if (json_parser_impl_->ParseTvmKernelBinId(kernel_bin_id) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmKernelBinId] get attr kernel bin id for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (kernel_bin_id.empty()) {
    return SUCCESS;
  }
  (void)SetRelatedNodesStr(kAttrKernelBinId, kernel_bin_id);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseGlobleWorkspaceStatus() {
  ge::ComputeGraphPtr graph = node_.GetOwnerComputeGraph();
  if (graph == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseGlobleWorkspaceStatus] get graph for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("[SubGraphOpt][ParseJson][ParseGlobalWorkspaceStatus] Successfully retrieved graph [%s] for node [%s].",
          graph->GetName().c_str(), node_.GetName().c_str());
  KeyGlobalWorkspaceSpecWorkspace global_work_space = {0, 0};
  Status ret = json_parser_impl_->ParseGlobleWorkspaceStatus(global_work_space);
  if (ret == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseGlobleWorkspaceStatus] "
                    "get globleworkspace_status for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (ret == NOT_CHANGED) {
    return SUCCESS;
  }
  if (!ge::AttrUtils::HasAttr(graph, kGlobalworkspaceBytes)) {
    (void)ge::AttrUtils::SetInt(graph, kGlobalworkspaceBytes, global_work_space.size);
    FE_LOGD("[SubGraphOpt][Compile][ParseGlobalWorkspaceStatus] graph[%s] op[%s] set [%s:%ld] successfully",
            graph->GetName().c_str(), op_desc_->GetName().c_str(),
            kGlobalworkspaceBytes.c_str(), global_work_space.size);
  }
  if (!ge::AttrUtils::HasAttr(graph, kKeyGlobalWorkspace)) {
    (void)ge::AttrUtils::SetInt(graph, kKeyGlobalWorkspace, 0);
  }
  SetRelatedNodesStr(kGlobalWorkspaceRef, kKeyGlobalWorkspace);
  SetRelatedNodesInt(kGlobalworkspaceType, global_work_space.type);
  SetRelatedNodesInt(kGlobalworkspaceSize, global_work_space.size);
  FE_LOGD("[SubGraphOpt][Compile][ParseGlobalWorkspaceStatus] Set attr globalworkspace_type[%ld], "
          "globalworkspace_size[%ld] for op[%s].", global_work_space.type, global_work_space.size,
          op_desc_->GetName().c_str());
  return SUCCESS;
}

Status TbeJsonFileParse::ParseOptionalInputMode() {
  if (op_desc_->GetType() == kSuperKernelType) {
    return SUCCESS;
  }
  string opt_input_mode;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyOptionalInputMode, opt_input_mode, opt_input_mode) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalInputMode] Failed to parse optionalInputMode for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (opt_input_mode.empty()) {
    opt_input_mode = kNoPlaceholder;
  }
  FE_LOGI("[SubGraphOpt][ParseJson][ParseOptionalInputMode] set attr optionalInputMode[val: %s] for node[%s].",
          opt_input_mode.c_str(), op_desc_->GetName().c_str());
  SetRelatedNodesStr(kAttrOptionalInputMode, opt_input_mode);
  if (opt_input_mode == kGenPlaceholder) {
    int64_t imply_type = -1;
    if (!ge::AttrUtils::GetInt(op_desc_, FE_IMPLY_TYPE, imply_type)) {
      REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalInputMode] Op[name=%s, type=%s] failed to get op implementation type.",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return FAILED;
    }
    OpImplType op_impl_type = static_cast<OpImplType>(imply_type);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(node_.GetOpDesc()->GetOpEngineName())
        .GetOpKernelInfoByOpType(op_impl_type, node_.GetType());
    if (op_kernel_info_ptr == nullptr) {
      REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalInputMode] Op[name=%s,type=%s] get kernel info failed.",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return FAILED;
    }
    size_t all_input_size = op_kernel_info_ptr->GetAllInputInfo().size();
    SetRelatedNodesInt(kOpKernelAllInputSize, all_input_size);
    FE_LOGD("Op[name=%s, type=%s]: set all in size[%zu].", op_desc_->GetName().c_str(),
            op_desc_->GetType().c_str(), all_input_size);
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseOptionalOutputMode() {
  if (op_desc_->GetType() == kSuperKernelType) {
    return SUCCESS;
  }
  string opt_output_mode;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyOptionalOutputMode, string(kNoPlaceholder), opt_output_mode) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalOutputMode]Failed to parse optionalOutputMode for op[%s:%s]",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  if (opt_output_mode == kNoPlaceholder) {
    return SUCCESS;
  }
  if (opt_output_mode != kGenPlaceholder) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalOutputMode]Op[%s:%s] optionalOutputMode is not supported",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  FE_LOGD("[SubGraphOpt][ParseJson][ParseOptionalOutputMode] set attr optionalOutputMode[val: %s] for op[%s:%s]",
          opt_output_mode.c_str(), op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  SetRelatedNodesStr(kAttrOptionalOutputMode, opt_output_mode);
  int64_t imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc_, FE_IMPLY_TYPE, imply_type)) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalOutputMode] Op [%s:%s] failed to get op impl type.",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  OpImplType op_impl_type = static_cast<OpImplType>(imply_type);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(node_.GetOpDesc()->GetOpEngineName())
      .GetOpKernelInfoByOpType(op_impl_type, node_.GetType());
  if (op_kernel_info_ptr == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOptionalOutputMode] Failed to get kernel info for Op [%s:%s].",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  size_t all_output_size = op_kernel_info_ptr->GetAllOutputInfo().size();
  SetRelatedNodesInt(kOpKernelAllOutputSize, all_output_size);
  FE_LOGD("Op[%s:%s]: set all out size[%zu].", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), all_output_size);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseDynamicParamMode() {
  if (op_desc_->GetType() == kSuperKernelType) {
    return SUCCESS;
  }
  string dy_input_mode;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyDynamicParamMode, dy_input_mode, dy_input_mode) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseDynamicInputMode] Failed to parse dynamicInputMode for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (dy_input_mode.empty()) {
    return SUCCESS;
  }
  FE_LOGI("[SubGraphOpt][ParseJson][ParseDynamicInputMode] Set attr dynamicInputMode[val: %s] for node[%s].",
          dy_input_mode.c_str(), op_desc_->GetName().c_str());
  SetRelatedNodesStr(kAttrDynamicParamMode, dy_input_mode);
  return SUCCESS;
}

Status TbeJsonFileParse::ParseOpKBHitrate() {
  string graph_node_name;
  string session_graph_id;
  ge::ComputeGraphPtr graph = node_.GetOwnerComputeGraph();
  if (graph != nullptr) {
    (void)ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    graph_node_name = session_graph_id;
    graph_node_name += "_";
    graph_node_name += op_desc_->GetName();
  } else {
    graph_node_name = op_desc_->GetName();
  }

  bool kb_hit = false;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyKBHit, kb_hit, kb_hit) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseOpKBHitrate] Failed to get the op_hitrate for node [%s].",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  std::string kernel_name;
  (void)ge::AttrUtils::GetStr(op_desc_, GetAttrPrefix() + kKernelName, kernel_name);
  // kb_hit log can not be deleted, other components are already in use.
  FE_LOGI("[op_kb_hit][%s][%d][%s]", graph_node_name.c_str(), kb_hit, kernel_name.c_str());
  return SUCCESS;
}

Status TbeJsonFileParse::ParseTvmKernelName() {
  // maste json donot have attr kernel name
  std::string kernel_name;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyKernelName, kernel_name, kernel_name) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmKernNm] get attr kernel name for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (!kernel_name.empty()) {
    SetRelatedNodesStrPrefixWithOpName(GetAttrPrefix(), kKernelName, kernel_name);
    SetRelatedNodesStr(GetAttrPrefix() + kKernelName, kernel_name);
    int64_t spk_scope = -1;
    if (ge::AttrUtils::GetInt(op_desc_, kAscendcSuperKernelScope, spk_scope) && spk_scope != -1) {
      SetRelatedNodesStr(kSuperKernelPrefix + kKernelName, kernel_name);
    }
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseConvCompressParameters() {
  std::vector<int64_t> compress_param_vec;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyCompressParameters, compress_param_vec,
                                       compress_param_vec) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmKernNm] get the compress_parameters for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  if (compress_param_vec.empty()) {
    return SUCCESS;
  }

  if (!SetRelatedNodesListInt(ATTR_NAME_COMPRESS_PARAMETERS, compress_param_vec)) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseTvmKernNm] Failed to set attribute compress_weight for node %s.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status TbeJsonFileParse::ParseWeightRepeat() {
  int64_t weight_repeat = INT64_MAX;
  if (json_parser_impl_->ParseJsonAttr(false, kKeyWeightRepeat, weight_repeat, weight_repeat) == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseWgtRepeat] get the weight_repeat for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }

  if (weight_repeat == INT64_MAX) {
    return SUCCESS;
  }

  FE_LOGI("The weight repeat of node [%s] is %ld times.", op_desc_->GetName().c_str(), weight_repeat);
  if (!SetRelatedNodesInt(ATTR_NAME_WEIGHT_REPEAT, weight_repeat)) {
    FE_LOGE("Failed to set attribute weight_repeat for node [%s].", op_desc_->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseSupportInfo() {
  nlohmann::json support_info;
  (void)json_parser_impl_->ParseJsonAttr(false, kKeySupportInfo, support_info, support_info);
  if (support_info.empty()) {
    return SUCCESS;
  }

  std::string op_debug_config;
  (void)json_parser_impl_->ParseJsonAttr(support_info, false, kOpDebugConfig, op_debug_config, op_debug_config);
  if (op_debug_config.empty()) {
    return SUCCESS;
  }

  std::vector<std::string> op_debug_config_vec = StringUtils::Split(op_debug_config, ',');
  for (size_t i = 0; i < op_debug_config_vec.size(); i++) {
    if (fe::StringUtils::Trim(op_debug_config_vec[i]) == kMemoryCheckKey) {
      SetRelatedNodesBool(kMemoryCheck, true);
    }
  }

  return SUCCESS;
}

Status TbeJsonFileParse::ParseOpParaSize() {
  bool op_debug_compile = false;
  ge::AttrUtils::GetBool(op_desc_, kOpDebugCompile, op_debug_compile);
  if (Configuration::Instance(AI_CORE_NAME).GetMemoryCheckSwitch() && op_debug_compile) {
    FE_LOGD("The node[%s] is labeled with memcheck.", op_desc_->GetNamePtr());
    SetRelatedNodesBool(kMemoryCheck, true);
  }
  int64_t op_para_size = 0;
  (void)json_parser_impl_->ParseJsonAttr(false, kKeyOpParaSize, op_para_size, op_para_size);
  FE_LOGD("Op[%s] Parse OpParaSize: %ld.", op_desc_->GetNamePtr(), op_para_size);
  (void)SetRelatedNodesInt(OP_PARA_SIZE, op_para_size);

  uint64_t ori_op_para_size = 0;
  (void)json_parser_impl_->ParseJsonAttr(false, kKeyOriOpParaSize, ori_op_para_size, ori_op_para_size);
  if (ori_op_para_size != 0) {
    FE_LOGD("The ori_op_para_size of node [%s] is %lu.", op_desc_->GetNamePtr(), ori_op_para_size);
    (void)SetRelatedNodesInt(ORI_OP_PARA_SIZE, ori_op_para_size);
  }
  return SUCCESS;
}

Status TbeJsonFileParse::PackageTvmBinFile() {
  ge::OpKernelBinPtr tbe_kernel_ptr = json_parser_impl_->GetOpKernelBinPtr();
  if (tbe_kernel_ptr == nullptr) {
    vector<char> buffer;
    if (json_parser_impl_->PackageTvmBinFile(buffer) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][ParseJson][PkgTvmBinFile] Read bin file failed.");
      return PARAM_INVALID;
    }

    std::string kernel_name;
    (void)ge::AttrUtils::GetStr(op_desc_, GetAttrPrefix() + kKernelName, kernel_name);
    FE_MAKE_SHARED(tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(kernel_name, std::move(buffer)), return FAILED);
  }

  SetRelatedNodesExtAttr(GetAttrPrefix() + ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  (void)SetRelatedNodesStr(GetAttrPrefix() + ge::ATTR_NAME_TBE_KERNEL_NAME,
                           tbe_kernel_ptr->GetName());
  if (FEContextUtils::IsOpTuneMode()) {
    FE_LOGI("Tuning mode, need save bytes bin.");
    ge::Buffer tbe_kernel_buffer(tbe_kernel_ptr->GetBinDataSize());
    tbe_kernel_buffer = ge::Buffer::CopyFrom(tbe_kernel_ptr->GetBinData(), tbe_kernel_ptr->GetBinDataSize());
    (void)SetRelatedNodesBytes(GetAttrPrefix() + ge::ATTR_NAME_TBE_KERNEL_BUFFER,
                               tbe_kernel_buffer);
  }
  size_t tbe_kernel_size = tbe_kernel_ptr->GetBinDataSize();
  (void)SetRelatedNodesInt(GetAttrPrefix() + ATTR_NAME_TBE_KERNEL_SIZE, tbe_kernel_size);
  (void)SetRelatedNodesExtAttr(ge::ATTR_NAME_OP_FILE_PATH, json_parser_impl_->GetTvmDirPath());
  FE_LOGD("node[%s]'s tbe kernel buffer size is %zu.", op_desc_->GetName().c_str(), tbe_kernel_size);
  return SUCCESS;
}

Status TbeJsonFileParse::PackageHeadFilePath() {
  std::string head_file_path;
  (void)json_parser_impl_->ParseHeadFilePath(head_file_path);
  if (!head_file_path.empty()) {
    SetRelatedNodesStr(kAttrHeadFilePath, head_file_path);
    FE_LOGD("Head file path for node[%s] is [%s].", op_desc_->GetName().c_str(), head_file_path.c_str());
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseCompileResult() {
  std::string compile_info_json;
  std::string compile_info_key;
  std::string core_type;
  if (ge::AttrUtils::GetStr(op_desc_, COMPILE_INFO_JSON, compile_info_json)) {
    (void)SetRelatedNodesStr(COMPILE_INFO_JSON, compile_info_json);
  }
  if (ge::AttrUtils::GetStr(op_desc_, COMPILE_INFO_KEY, compile_info_key)) {
    (void)SetRelatedNodesStr(COMPILE_INFO_KEY, compile_info_key);
  }
  if (ge::AttrUtils::GetStr(op_desc_, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type)) {
    (void)SetRelatedNodesStr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  }
  return SUCCESS;
}

Status TbeJsonFileParse::ParseOpDfxOptions() {
  std::vector<std::string> opt_list;
  int64_t buffer_size = 0;
  if (json_parser_impl_->ParseOpDfxOptions(opt_list, buffer_size) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseDfxDebOptions]Failed to parse dfx option.");
    return PARAM_INVALID;
  }
  if (!opt_list.empty() && buffer_size != 0) {
    SetRelatedNodesListStr(kOpDfxOptions, opt_list);
    SetRelatedNodesInt(kOpDfxBufferSize, buffer_size);
  }
  return SUCCESS;
}

bool TbeJsonFileParse::IsModelBinaryReuse(const CompileResultInfo &compile_result) {
  if (compile_result.bin_file_path.empty()) {
    return false;
  }
  size_t pos = compile_result.bin_file_path.rfind('.');
  if (pos == std::string::npos) {
    return false;
  }
  bool ret = compile_result.bin_file_path.substr(pos) == kModelBinFileSuffix;
  if (ret) {
    SetRelatedNodesStr(ge::ATTR_NAME_OM_BINARY_PATH, compile_result.bin_file_path);
  }
  return ret;
}

const std::string& TbeJsonFileParse::GetAttrPrefix() const {
  return attr_prefix_;
}

Status TbeJsonFileParse::PackageTvmJsonInfo(const CompileResultInfo &compile_result) {
  FE_LOGD("Start to parse op_json_file %s for node [%s, %s].", compile_result.json_file_path.c_str(),
          op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  FE_CHECK_NOTNULL(json_parser_impl_);
  Status status = json_parser_impl_->Initialize(compile_result.json_file_path,
                                                compile_result.json_ptr, compile_result.bin_ptr);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][InitialJsonParserImpl] Initialization failed for node [%s, %s].",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return status;
  }

  if (IsModelBinaryReuse(compile_result)) {
    return SUCCESS;
  }

  for (auto &parseFunc : parse_func_map_) {
    if ((this->*(parseFunc.second))() != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][ParseJson][PkgTvmJsInfo] Failed to parse %s, the JSON file is: %s.",
                      parseFunc.first.c_str(), compile_result.json_file_path.c_str());
      return FAILED;
    }
  }
  if (ge::OpTypeUtils::IsAutofuseNode(op_desc_)) {
    SetRelatedNodesStr("bin_file_path", compile_result.bin_file_path);
    FE_LOGD("Set autofuse bin file path: %s.", compile_result.bin_file_path.c_str());
  }
  return SUCCESS;
}

void TbeJsonFileParse::HexDecode(const std::string &hex_str, std::vector<unsigned char> &bytes) const {
  if (hex_str.size() % kNumTwo != 0) {
    FE_LOGE("Hex string length must be even.");
    return;
  }
  for (size_t i = 0; i < hex_str.size(); i += kNumTwo) {
    std::string byte_str = hex_str.substr(i, 2);
    uint8_t byte = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    bytes.emplace_back(byte);
  }
}

bool TbeJsonFileParse::ParseAndSetTilingData(const std::string &tiling_data_str, RunInfoPtr &tiling_info) {
  FE_LOGI("Tiling data string in JSON is [%s].", tiling_data_str.c_str());
  std::vector<unsigned char> byte_array;
  try {
    HexDecode(tiling_data_str, byte_array);
  } catch (...) {
    FE_LOGE("Decode hex str[%s] failed.", tiling_data_str.c_str());
    return false;
  }
  char* char_array = reinterpret_cast<char*>(byte_array.data());
  size_t data_size = byte_array.size();
  tiling_info->AddTilingData(reinterpret_cast<ge::char_t *>(char_array), data_size);
  return true;
}

Status TbeJsonFileParse::ParseAndSetTilingInfo() {
  RunInfoPtr run_info = nullptr;
  FE_MAKE_SHARED(run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0), return FAILED);
  // mc2 static compile json has attr tilingData
  FE_CHECK_NOTNULL(json_parser_impl_);
  OpTilingInfo tiling_info;
  bool has_run_info = false;
  if (json_parser_impl_->ParseRunInfo(tiling_info, has_run_info) != SUCCESS) {
    return FAILED;
  }
  if (!has_run_info) {
    return SUCCESS;
  }
  run_info->SetTilingKey(tiling_info.tiling_key);
  run_info->SetBlockDim(tiling_info.block_dim);
  run_info->SetAicpuBlockDim(tiling_info.aicpu_block_dim);
  run_info->SetClearAtomic(tiling_info.clear_atomic);
  run_info->SetWorkspaces(tiling_info.tvm_workspace_sizes);
  run_info->SetTilingCond(tiling_info.tiling_cond);
  run_info->SetScheduleMode(tiling_info.schedule_mode);
  run_info->SetLocalMemorySize(tiling_info.local_memory_size);
  if (!ParseAndSetTilingData(tiling_info.tiling_data_str, run_info)) {
    FE_LOGE("Node [%s, %s] failed to parse tiling data.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  op_desc_->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, run_info);
  (void)ge::AttrUtils::SetStr(op_desc_, kAttrTilingDataStr, run_info->GetAllTilingData().str());
  (void)ge::AttrUtils::SetInt(op_desc_, ge::TVM_ATTR_NAME_BLOCKDIM, run_info->GetBlockDim());
  (void)ge::AttrUtils::SetInt(op_desc_, kAicpuBlockDim, run_info->GetAicpuBlockDim());
  (void)ge::AttrUtils::SetInt(op_desc_, kAttrScheduleMode, run_info->GetScheduleMode());
  (void)ge::AttrUtils::SetInt(op_desc_, kLocalMemorySize, run_info->GetLocalMemorySize());
  FE_LOGD("Set attr block dim[%u], aicpu blockdim[%u] and schedule mode[%u] for op[%s, %s].",
          run_info->GetBlockDim(), run_info->GetAicpuBlockDim(), run_info->GetScheduleMode(),
          op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  FE_LOGI("Node [%s, %s] set tiling info success.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  return SUCCESS;
}

Status TbeJsonFileParse::ParseFatbinInfo() {
  bool tile_fwk_op_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_, kAttrTileFwkOpStr, tile_fwk_op_flag);
  if (!tile_fwk_op_flag) {
    return SUCCESS;
  }
  std::string op_type = op_desc_->GetType();
  if (TileFwkOpInfo::Instance().CheckFatbinInfo(op_type)) {
    FE_LOGD("Node[%s, %s]: hit fatbin info cache, no need to parse fatbin.",
            op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return SUCCESS;
  }
  ge::OpKernelBinPtr fatbin = nullptr;
  fatbin = op_desc_->TryGetExtAttr<ge::OpKernelBinPtr>(GetAttrPrefix() + ge::OP_EXTATTR_NAME_TBE_KERNEL, fatbin);
  FE_CHECK_NOTNULL(fatbin);
  FatbinKernelInfoMap fatbin_kernel_info_map;
  if (json_parser_impl_->ParseFatbin(fatbin, fatbin_kernel_info_map)  == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseFatbinInfo] parse fatbin for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  if (json_parser_impl_->ParseFatbinJson(fatbin_kernel_info_map)  == FAILED) {
    REPORT_FE_ERROR("[SubGraphOpt][ParseJson][ParseFatbinInfo] parse fatbin json for node[%s] failed.",
                    op_desc_->GetName().c_str());
    return FAILED;
  }
  TileFwkOpInfo::Instance().SetFatbinInfo(op_type, fatbin_kernel_info_map);
  return SUCCESS;
}
}  // namespace fe
