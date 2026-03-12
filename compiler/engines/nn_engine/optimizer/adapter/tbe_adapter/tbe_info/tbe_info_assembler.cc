/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include <sstream>
#include <set>
#include "common/aicore_util_constants.h"
#include "common/configuration.h"
#include "common/dump_util.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/lxfusion_json_util.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/string_utils.h"
#include "common/unknown_shape_util.h"
#include "common/platform_utils.h"
#include "ge/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "adapter/tbe_adapter/tbe_info/execution_time_estimator.h"
#include "framework/common/ge_types.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "register/op_impl_kernel_registry.h"
#include "common/fe_context_utils.h"
#include "graph/tuning_utils.h"

namespace fe {
namespace {
const string kDebugMode = "2";
const string kNeedEstimationCoreNum = "1";
const string kEnableSuperkernelPlus = "enable_superkernel_plus";
const string kOpDebugConfigTe = "op_debug_config_te";
const string kAicCntKeyOp = "_op_aicore_num";
const string kAivCntKeyOp = "_op_vectorcore_num";

const std::map<JitCompile, te::JitCompileType> kJitCompileMap {
    {JitCompile::DEFAULT, te::JitCompileType::DEFAULT},
    {JitCompile::ONLINE, te::JitCompileType::ONLINE},
    {JitCompile::REUSE_BINARY, te::JitCompileType::REUSE_BINARY},
    {JitCompile::STATIC_BINARY_DYNAMIC_ONLINE, te::JitCompileType::STATIC_BINARY_DYNAMIC_ONLINE},
    {JitCompile::STATIC_BINARY_DYNAMIC_BINARY, te::JitCompileType::STATIC_BINARY_DYNAMIC_BINARY}
};

const std::map<DynamicRankType, te::DynamicRankType> kDynamicRankTypeMap {
        {DynamicRankType::SUPPORT, te::DynamicRankType::SUPPORT},
        {DynamicRankType::NOT_SUPPORT, te::DynamicRankType::NOT_SUPPORT},
        {DynamicRankType::UPGRADE_TO_SUPPORT, te::DynamicRankType::UPGRADE_TO_SUPPORT}
};

const std::map<RangeLimitType, te::RangeLimitType> kRangeLimitTypeMap {
        {RangeLimitType::LIMITED, te::RangeLimitType::LIMITED},
        {RangeLimitType::UNLIMITED, te::RangeLimitType::UNLIMITED},
        {RangeLimitType::DYNAMIC, te::RangeLimitType::DYNAMIC}
};

const std::map<VectorCoreType, te::VectorCoreType> kVectorCoreTypeMap {
        {VectorCoreType::ENABLE, te::VectorCoreType::ENABLE},
        {VectorCoreType::DISABLE, te::VectorCoreType::DISABLE}
};

const std::map<OpPattern, std::string> kOpPatternStrMap = {
        std::make_pair(OP_PATTERN_OP_KERNEL, "opKernel"),
        std::make_pair(OP_PATTERN_OP_CUSTOMIZE, "opCustomize"),
        std::make_pair(OP_PATTERN_FORMAT_AGNOSTIC, "formatAgnostic"),
        std::make_pair(OP_PATTERN_BROADCAST, "broadcast"),
        std::make_pair(OP_PATTERN_REDUCE, "reduce"),
        std::make_pair(OP_PATTERN_RANGE_AGNOSTIC, "rangeAgnostic"),
        std::make_pair(OP_PATTERN_BROADCAST_ENHANCED, "broadcastEnhanced")
};

void SetTbeTensorValueRange(const ge::OpDesc &op_desc, const ge::GeTensorDesc &tensor_desc,
                            te::TbeOpTensor &tbe_tensor) {
  std::vector<std::pair<int64_t, int64_t>> value_range = GetValueRange(tensor_desc);
  FE_LOGD("[SubGraphOpt][SetTbeTensor][SetTValueRange] Value range of op [name:%s, type:%s] is [%s].",
          op_desc.GetName().c_str(), op_desc.GetType().c_str(), ShapeRangeToStr(value_range).c_str());
  tbe_tensor.SetValueRange(value_range);
}

bool GetAddTensorFlag(const ge::OpDesc &op_desc, const vector<uint32_t> &specific_input_index_vec) {
  bool res = false;
  if (!specific_input_index_vec.empty()) {
    auto input_desc = op_desc.GetInputDescPtr(specific_input_index_vec.at(0));
    if (input_desc == nullptr) { return false; }
    auto primary_format = ge::GetPrimaryFormat(input_desc->GetFormat());
    res = primary_format != ge::FORMAT_RESERVED && input_desc->GetDataType() != ge::DT_UNDEFINED
            && input_desc->GetDataType() < ge::DT_MAX;
  }
  return res;
}

bool GetAddTensorFlag(const ge::OpDescPtr &op_desc, const vector<uint32_t> &specific_input_index_vec) {
  return GetAddTensorFlag(*op_desc.get(), specific_input_index_vec);
}

void SetCDimReshapeValue(ge::GeTensorDesc &tensor_desc, te::TbeOpTensor &tensor) {
  int64_t c_dim = -1;
  (void)ge::AttrUtils::GetInt(tensor_desc, fe::ATTR_NAME_RESHAPE_CXVALUE, c_dim);
  if (c_dim == -1) {
    c_dim = GetAxisValueByName('C', tensor_desc);
    FE_LOGD("Cdim = -1 not come from reshape and cdim value %lld exist for 5hd or 4d.", c_dim);
  }
  tensor.SetCAxisValue(c_dim);
  return;
}

void SetTbeTensorShape(const ge::OpDesc &op_desc, const TensorDescAndIndex &tensor_info, te::TbeOpTensor &tbe_tensor) {
  bool stc_to_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kStaticToDynamicSoftSyncOp, stc_to_dyn_soft_sync);
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(op_desc, kDynamicTilingDependOp, stc_tiling_depend);
  if (UnknownShapeUtils::IsUnknownShapeOp(op_desc) || stc_to_dyn_soft_sync || stc_tiling_depend) {
    std::vector<std::pair<int64_t, int64_t>> shape_range = GetShapeRange(*tensor_info.tensor_desc_ptr.get());
    std::vector<std::pair<int64_t, int64_t>> ori_shape_range;
    (void)tensor_info.tensor_desc_ptr->GetOriginShapeRange(ori_shape_range);
    FE_LOGD("Shape range of op[name:%s,type:%s] is %s, origin range is %s.", op_desc.GetName().c_str(),
            op_desc.GetType().c_str(), ShapeRangeToStr(shape_range).c_str(), ShapeRangeToStr(ori_shape_range).c_str());
    tbe_tensor.SetShapeRange(shape_range);
    tbe_tensor.SetOriginShapeRange(ori_shape_range);
    SetTbeTensorValueRange(op_desc, *tensor_info.tensor_desc_ptr.get(), tbe_tensor);
  }

  ge::Format primary_origin_format =
          static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_info.tensor_desc_ptr->GetOriginFormat()));
  tbe_tensor.SetOriginFormat(ge::TypeUtils::FormatToSerialString(primary_origin_format));
  std::vector<int64_t> dyn_ori_shape_vec;
  std::vector<int64_t> ori_shape_vec = tensor_info.tensor_desc_ptr->GetOriginShape().GetDims();
  (void)ge::AttrUtils::GetListInt(tensor_info.tensor_desc_ptr, kSoftSyncDynOriShape, dyn_ori_shape_vec);
  if (!dyn_ori_shape_vec.empty()) {
    ori_shape_vec = dyn_ori_shape_vec;
  }
  tbe_tensor.SetOriginShape(ori_shape_vec);
  SetCDimReshapeValue(*tensor_info.tensor_desc_ptr.get(), tbe_tensor);
  FE_LOGD("Op[name=%s,type=%s]: origin %s format is [%s], the index_in_op_kernel is [%zu].", op_desc.GetName().c_str(),
          op_desc.GetType().c_str(), IS_INPUT_TO_STRING(tensor_info.is_input),
          ge::TypeUtils::FormatToSerialString(tensor_info.tensor_desc_ptr->GetOriginFormat()).c_str(),
          tensor_info.index_in_op_kernel);
  string origin_shape_dims = GetShapeDims(ori_shape_vec);
  FE_LOGD("Op[name=%s,type=%s]: origin %s shape is [%s].", op_desc.GetName().c_str(), op_desc.GetType().c_str(),
          IS_INPUT_TO_STRING(tensor_info.is_input), origin_shape_dims.c_str());
}

Status SetTbeTensor(const ge::OpDesc &op_desc, const TensorDescAndIndex &tensor_info, te::TbeOpTensor &input_tensor) {
  Status ret = CreateTbeTensor(op_desc, tensor_info, input_tensor);
  if (ret != SUCCESS) {
    return ret;
  }

  if (tensor_info.is_input) {
    input_tensor.SetFirstLayer(tensor_info.is_first_layer_conv);
  }

  /* Set original format and dtype */
  SetTbeTensorShape(op_desc, tensor_info, input_tensor);
  return SUCCESS;
}

bool CheckAOETuning(const ge::Node *node) {
  ge::ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    FE_LOGW("Node[%s] can not get owner graph.", node->GetName().c_str());
    return false;
  }
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (build_mode_value == ge::BUILD_MODE_TUNING) {
    FE_LOGI("Node[%s]'s graph has build_mode_value BUILD_MODE_TUNING.", node->GetName().c_str());
    return true;
  }
  FE_LOGI("Node[%s]'s graph check not aoe tuning.", node->GetName().c_str());
  return false;
}
}

const std::map<ge::DataType, SetConstValueWithDtypePtr> TbeInfoAssembler::set_const_value_func_map = {
    {ge::DT_FLOAT16, std::make_shared<SetConstValueWithDtype>(SetConstValueWithFloat16)},
    {ge::DT_BF16, std::make_shared<SetConstValueWithDtype>(SetConstValueWithBf16)},
    {ge::DT_FLOAT, std::make_shared<SetConstValueWithDtype>(SetConstValue<float>)},
    {ge::DT_DOUBLE, std::make_shared<SetConstValueWithDtype>(SetConstValue<double>)},
    {ge::DT_INT8, std::make_shared<SetConstValueWithDtype>(SetConstValue<int8_t>)},
    {ge::DT_UINT8, std::make_shared<SetConstValueWithDtype>(SetConstValue<uint8_t>)},
    {ge::DT_INT16, std::make_shared<SetConstValueWithDtype>(SetConstValue<int16_t>)},
    {ge::DT_UINT16, std::make_shared<SetConstValueWithDtype>(SetConstValue<uint16_t>)},
    {ge::DT_INT32, std::make_shared<SetConstValueWithDtype>(SetConstValue<int32_t>)},
    {ge::DT_UINT32, std::make_shared<SetConstValueWithDtype>(SetConstValue<uint32_t>)},
    {ge::DT_INT64, std::make_shared<SetConstValueWithDtype>(SetConstValue<int64_t>)},
    {ge::DT_UINT64, std::make_shared<SetConstValueWithDtype>(SetConstValue<uint64_t>)},
    {ge::DT_BOOL, std::make_shared<SetConstValueWithDtype>(SetConstValue<bool>)}};

/* The map of estimated execution time to amplified blocknum.
 * {4800, 8} means if execution time is less than 4800ns,
 * the block num should be 8. */
const std::map<uint32_t, std::vector<std::pair<uint64_t, string>>> TbeInfoAssembler::time_to_core_num_ = {
    {8, {{UINT64_MAX, "8"}}},
    {10, {{201610, "10"}, {UINT64_MAX, "40"}}}
};

static std::map<ffts::AtomicType, std::string> AtomicTypeStr = {
    {ffts::AtomicType::None, "None"},
    {ffts::AtomicType::ADD, "add"},
    {ffts::AtomicType::SUB, "sub"},
    {ffts::AtomicType::MUL, "mul"},
    {ffts::AtomicType::DIV, "div"},
};

Status TbeInfoAssembler::Initialize() {
  return PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(all_plat_info_.platform_info,
                                                                          all_plat_info_.opti_compilation_info);
}

/*
 *  @ingroup fe
 *  @brief   set parameter infos to tbe_op_info
 *  @param   [in]      node           op node pointer
 *  @param   [in]      input_info_ptr   op info store pointer
 *  @param   [in/out]  input          global temp param
 *  @param   [in]      input_tensor    tensor from const node
 *  @param   [in/out]  op_info         tbe data item
 *  @param   [in]      input_size     number of inputs or outputs of op
 *  @param   [in]      i             index to input
 *  @param   [in]      is_input_or_output input or output
 *  @return  SUCCESS or FAILED
 */
Status TbeInfoAssembler::FeedParameterInfoForInput(const ge::Node *node, const InputOrOutputInfoPtr &info_ptr,
                                                   int index_in_opdesc, bool last_item_flag,
                                                   te::TbeOpTensor &tbe_op_tensor, te::TbeOpParam &tbe_op_param,
                                                   te::TbeOpInfo &tbe_op_info) const {
  OpParamType input_type = info_ptr->GetParamType();
  tbe_op_param.SetType(te::TensorType(input_type));
  OpConstValueDepend value_depend = info_ptr->GetConstValueDepend();
  auto iter = VALUE_DEPEND_MAP.find(value_depend);
  if (iter != VALUE_DEPEND_MAP.end()) {
    tbe_op_param.SetValueDepend(iter->second);
  } else {
    FE_LOGW("Node[type=%s,name=%s]: ValueDepend is %d, is invalid.", node->GetOpDesc()->GetType().c_str(),
            node->GetOpDesc()->GetName().c_str(), value_depend);
    tbe_op_param.SetValueDepend(te::VALUE_DEPEND_IGNORE);
  }

  auto input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(index_in_opdesc));
  if (input_desc != nullptr && input_type == OpParamType::REQUIRED &&
      input_desc->GetOriginDataType() == ge::DT_UNDEFINED) {
    FE_LOGW("Node[type=%s,name=%s]Input %s original data type is invalid.", node->GetType().c_str(),
            node->GetType().c_str(), info_ptr->GetName().c_str());
  }

  // dynamic input contains multiple sub-inputs, should be packaged together
  if (input_type == OpParamType::DYNAMIC) {
    tbe_op_param.SetTensor(tbe_op_tensor);
    if (last_item_flag) {
      tbe_op_info.AddInput(tbe_op_param);
      tbe_op_param.Clear();
    }
  } else {
    bool need_tensor =
        ((input_type == OpParamType::OPTIONAL || input_type == OpParamType::REQUIRED) && (index_in_opdesc >= 0) &&
         (node->GetInDataAnchor(index_in_opdesc) != nullptr &&
          node->GetInDataAnchor(index_in_opdesc)->GetPeerOutAnchor() != nullptr));
    if (need_tensor) {
      tbe_op_param.SetTensor(tbe_op_tensor);
    }
    tbe_op_info.AddInput(tbe_op_param);
    tbe_op_param.Clear();
  }
  return SUCCESS;
}

Status TbeInfoAssembler::FeedParameterInfoForOutput(const ge::OpDesc &op_desc, const ge::GeTensorDesc &output_desc,
                                                    const InputOrOutputInfoPtr &info_ptr, bool last_item_flag,
                                                    te::TbeOpTensor &tbe_op_tensor, te::TbeOpParam &tbe_op_param,
                                                    te::TbeOpInfo &tbe_op_info) const {
  OpParamType output_type = info_ptr->GetParamType();
  tbe_op_param.SetType(te::TensorType(output_type));
  OpConstValueDepend value_depend = info_ptr->GetConstValueDepend();
  auto iter = VALUE_DEPEND_MAP.find(value_depend);
  if (iter != VALUE_DEPEND_MAP.end()) {
    tbe_op_param.SetValueDepend(iter->second);
  } else {
    FE_LOGW("Node[type=%s,name=%s]: ValueDepend is %d, is invalid.", op_desc.GetType().c_str(),
            op_desc.GetName().c_str(), value_depend);
    tbe_op_param.SetValueDepend(te::VALUE_DEPEND_IGNORE);
  }
  if (output_type == OpParamType::REQUIRED && output_desc.GetOriginDataType() == ge::DT_UNDEFINED) {
    FE_LOGD("Node[type=%s,name=%s]Output %s original data type is invalid.", op_desc.GetType().c_str(),
            op_desc.GetName().c_str(), info_ptr->GetName().c_str());
  }

  // dynamic input contains multiple sub-inputs, should be packaged together
  if (output_type == OpParamType::DYNAMIC) {
    tbe_op_param.SetTensor(tbe_op_tensor);
    if (last_item_flag) {
      tbe_op_info.AddOutput(tbe_op_param);
      tbe_op_param.Clear();
    }
  } else {
    bool need_tensor =
        (output_type == OpParamType::OPTIONAL && !IsMemoryEmpty(output_desc) && !HasNullableOutput(output_desc))
        || output_type == OpParamType::REQUIRED;
    if (need_tensor) {
      tbe_op_param.SetTensor(tbe_op_tensor);
    } else {
      FE_LOGI("Node[type=%s,name=%s]: the output %s needs an empty tensor, the paramType=%d.",
              op_desc.GetType().c_str(), op_desc.GetName().c_str(), info_ptr->GetName().c_str(), output_type);
    }
    tbe_op_info.AddOutput(tbe_op_param);
    tbe_op_param.Clear();
  }
  return SUCCESS;
}

Status TbeInfoAssembler::FeedParameterInfoForNotFound(const InputOrOutputInfoPtr &info_ptr,
                                                      const string &is_input_or_output, te::TbeOpParam &tbe_op_param,
                                                      te::TbeOpInfo &tbe_op_info) const {
  tbe_op_param.SetType(te::TensorType(info_ptr->GetParamType()));
  if (is_input_or_output == STR_INPUT_LOWERCASE) {
    tbe_op_info.AddInput(tbe_op_param);
  } else if (is_input_or_output == STR_OUTPUT_LOWERCASE) {
    tbe_op_info.AddOutput(tbe_op_param);
  }
  tbe_op_param.Clear();
  return SUCCESS;
}

Status TbeInfoAssembler::ConvertParameterInfoForInput(InputOrOutputInfoPtr info_ptr, te::TbeOpParam &input,
                                                      te::TbeOpTensor &input_tensor, te::TbeOpInfo &op_info,
                                                      bool last_item_flag) const {
  OpConstValueDepend value_depend = info_ptr->GetConstValueDepend();
  auto iter = VALUE_DEPEND_MAP.find(value_depend);
  if (iter == VALUE_DEPEND_MAP.end()) {
    FE_LOGW("Node[input name=%s]: ValueDepend is %d, is invalid.", info_ptr->GetName().c_str(), value_depend);
    input.SetValueDepend(te::VALUE_DEPEND_IGNORE);
  } else {
    input.SetValueDepend(iter->second);
  }

  OpParamType input_type = info_ptr->GetParamType();
  input.SetType(te::TensorType(input_type));
  if (input_type == OpParamType::DYNAMIC) {
    input.SetTensor(input_tensor);
    if (last_item_flag) {
      op_info.AddInput(input);
      input.Clear();
    }
  } else {
    bool need_tensor = input_type == OpParamType::OPTIONAL || input_type == OpParamType::REQUIRED;
    if (need_tensor) {
      input.SetTensor(input_tensor);
    }
    op_info.AddInput(input);
    input.Clear();
  }
  return SUCCESS;
}

void TbeInfoAssembler::FeedL1InputTensor(const ToOpStructPtr &l1_info, const ge::OpDescPtr &op_desc,
                                         IndexNameMap &input_idx_name_map, const uint32_t &index_in_opdesc,
                                         te::TbeOpTensor &input_tensor) const {
  // update opinfo
  input_tensor.SetL1WorkspaceFlag(l1_info->op_l1_workspace_flag);
  input_tensor.SetL1WorkspaceSize(l1_info->op_l1_workspace_size);
  input_tensor.SetTotalShape(l1_info->total_shape);
  input_tensor.SetSplitIndex(l1_info->split_index);
  FE_LOGD("FeedL1InputTensor begin, op name is %s.", op_desc->GetName().c_str());
  if (!l1_info->op_l1_fusion_type.empty()) {
    input_tensor.SetL1FusionType(l1_info->op_l1_fusion_type[0]);
  }

  size_t input_idx_name_size = input_idx_name_map.size();
  size_t l1_slice_input_shape_size = l1_info->slice_input_shape.size();
  if (l1_slice_input_shape_size == input_idx_name_size) {
    if (index_in_opdesc >= l1_slice_input_shape_size) {
      FE_LOGW("index_in_opdesc > l1_slice_input_shape_size, index_in_opdesc:%u, l1_slice_input_shape_size:%zu.",
              index_in_opdesc, l1_slice_input_shape_size);
      return;
    }

    input_tensor.SetValidShape(l1_info->slice_input_shape[index_in_opdesc]);
  }

  size_t l1_slice_input_offset_size = l1_info->slice_input_offset.size();
  if (l1_slice_input_offset_size == input_idx_name_size) {
    if (index_in_opdesc >= l1_slice_input_offset_size) {
      FE_LOGW("index_in_opdesc >= l1_slice_input_offset_size, index_in_opdesc:%u, l1_slice_input_offset_size:%zu.",
              index_in_opdesc, l1_slice_input_offset_size);
      return;
    }
    input_tensor.SetSliceOffset(l1_info->slice_input_offset[index_in_opdesc]);
  }

  /* Set addr  offset */
  size_t input_num = op_desc->GetInputsSize();
  vector<int64_t> input_offsets = op_desc->GetInputOffset();
  size_t input_offset_size = input_offsets.size();
  if (input_num > input_offset_size) {
    FE_LOGW("input_num > input_offset_size, input_num:%lu, input_offset_size:%lu.", input_num, input_offset_size);
  } else {
    if (index_in_opdesc >= input_offset_size) {
      FE_LOGW("index_in_opdesc >= input_offsets_size, index_in_opdesc:%u, input_offset_size:%zu.", index_in_opdesc,
              input_offset_size);
      return;
    }
    input_tensor.SetAddrOffset(input_offsets[index_in_opdesc]);
  }
}

void TbeInfoAssembler::FeedL2InputTensor(const ToOpStructPtr &l2_info, const ge::OpDescPtr &op_desc,
                                         IndexNameMap &input_idx_name_map, const uint32_t &index_in_opdesc,
                                         te::TbeOpTensor &input_tensor) const {
  if (l2_info == nullptr) {
    return;
  }
  // update opinfo
  size_t input_idx_name_size = input_idx_name_map.size();
  size_t l2_slice_input_shape_size = l2_info->slice_input_shape.size();
  if (l2_slice_input_shape_size == input_idx_name_size) {
    if (index_in_opdesc >= l2_slice_input_shape_size) {
      FE_LOGW("index_in_opdesc >= l2_slice_input_shape_size, index_in_opdesc:%u, l2_slice_input_shape_size:%zu.",
              index_in_opdesc, l2_slice_input_shape_size);
      return;
    }
    input_tensor.SetValidShape(l2_info->slice_input_shape[index_in_opdesc]);
  }

  size_t l2_slice_input_offset_size = l2_info->slice_input_offset.size();
  if (l2_slice_input_offset_size == input_idx_name_size) {
    if (index_in_opdesc >= l2_slice_input_offset_size) {
      FE_LOGW("index_in_opdesc >= l2_slice_input_offset_size, index_in_opdesc:%u, l2_slice_input_offset_size:%zu.",
              index_in_opdesc, l2_slice_input_offset_size);
      return;
    }
    input_tensor.SetSliceOffset(l2_info->slice_input_offset[index_in_opdesc]);
  }
  /* Set addr  offset */
  size_t input_num = op_desc->GetInputsSize();
  vector<int64_t> input_offsets = op_desc->GetInputOffset();
  size_t input_offset_size = input_offsets.size();
  if (input_num > input_offset_size) {
    FE_LOGW("intput_desc_size > input_offset_size, input_desc_size:%lu, input_offset_size:%lu.", input_num,
            input_offset_size);
  } else {
    if (index_in_opdesc >= input_offset_size) {
      FE_LOGW("index_in_opdesc >= input_offsets_size, index_in_opdesc:%u, input_offset_size:%zu.", index_in_opdesc,
              input_offset_size);
      return;
    }
    input_tensor.SetAddrOffset(input_offsets[index_in_opdesc]);
  }
}

Status TbeInfoAssembler::SetInputTensorBaseInfo(const ge::OpDescPtr &op_desc,
                                                const uint32_t &index_in_opdesc,
                                                te::TbeOpTensor &input_tensor) const {
  if (op_desc->MutableInputDesc(index_in_opdesc) == nullptr) {
    return SUCCESS;
  }
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  auto input_i = op_desc->MutableInputDesc(index_in_opdesc);
  auto &shape = input_i->MutableShape();
  std::vector<int64_t> dim_vec;
  (void)ge::AttrUtils::GetListInt(input_i, kSoftSyncDynShape, dim_vec);
  if (dim_vec.empty()) {
    dim_vec = shape.GetDims();
  }
  ge::DataType dtype = input_i->GetDataType();
  auto format = input_i->GetFormat();
  auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(format));
  auto sub_format = static_cast<ge::Format>(ge::GetSubFormat(format));
  std::string dtype_str = "";

  FE_CHECK(TransDtypeToString(dtype, dtype_str) != SUCCESS,
           FE_LOGW("Current data type[%u] of input index = %u of op (name [%s], type [%s]) is not found.", dtype,
                  index_in_opdesc, op_name.c_str(), op_type.c_str()), return FAILED);
  FE_LOGD("Op[name=%s,type=%s]: index_in_opdesc is [%u], input format is [%s].", op_name.c_str(), op_type.c_str(),
          index_in_opdesc, ge::TypeUtils::FormatToSerialString(format).c_str());
  // If empty shape of scalar, the op will compile fail. So need set {1} to shape.
  if (input_i->MutableShape().IsScalar()) {
    dim_vec = {1};
  }
  FE_LOGD("Op[name=%s,type=%s]: current input shape is [%s].",
          op_name.c_str(), op_type.c_str(), GetShapeDims(dim_vec).c_str());
  string primary_format_str = ge::TypeUtils::FormatToSerialString(primary_format);
  // set full shape to StridedRead
  bool is_strided_read = op_desc->GetType() == STRIDEDREAD &&
      static_cast<ge::Format>(ge::GetPrimaryFormat(op_desc->GetInputDesc(0).GetFormat())) == ge::FORMAT_NC1HWC0;
  if (is_strided_read && dim_vec.size() >= 2) {
    int64_t c1 = dim_vec[1];
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_STRIDE_ATTR_STRIDE, c1);
    dim_vec[1] = c1;
  }
  // AiCore do not support bool.
  input_tensor.SetShape(dim_vec);
  input_tensor.SetType(dtype_str);
  input_tensor.SetFormat(primary_format_str);
  input_tensor.SetSubFormat(sub_format);
  bool stc_to_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kStaticToDynamicSoftSyncOp, stc_to_dyn_soft_sync);
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(op_desc, kDynamicTilingDependOp, stc_tiling_depend);
  if (stc_to_dyn_soft_sync || stc_tiling_depend || UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    std::vector<std::pair<int64_t, int64_t>> shape_range = GetShapeRange(*input_i);
    std::vector<std::pair<int64_t, int64_t>> ori_shape_range;
    (void)input_i->GetOriginShapeRange(ori_shape_range);
    FE_LOGD("Shape range of op[name:%s,type:%s] is %s, origin range is %s.", op_name.c_str(), op_type.c_str(),
            ShapeRangeToStr(shape_range).c_str(), ShapeRangeToStr(ori_shape_range).c_str());
    input_tensor.SetShapeRange(shape_range);
    input_tensor.SetOriginShapeRange(ori_shape_range);
    SetTbeTensorValueRange(*op_desc, *input_i, input_tensor);
  }
  return SUCCESS;
}

void TbeInfoAssembler::GetOpInputL1Attr(const ge::OpDescPtr &op_desc, std::vector<int64_t> &op_input_l1_flag,
                                        std::vector<int64_t> &op_input_l1_addr,
                                        std::vector<int64_t> &op_input_l1_valid_size) const {
  if (!ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OP_INPUT_L1_FLAG, op_input_l1_flag)) {
    FE_LOGD("Get attribute op_input_l1_flag of op[%s, %s] unsuccessful.", op_desc->GetName().c_str(),
            op_desc->GetType().c_str());
    return;
  }
  bool is_l1_flag = false;
  if (std::any_of(op_input_l1_flag.begin(), op_input_l1_flag.end(),
                  [](int64_t input_l1_flag) { return input_l1_flag >= 0; })) {
    is_l1_flag = true;
  }
  if (is_l1_flag) {
    if (!ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OP_INPUT_L1_ADDR, op_input_l1_addr)) {
      FE_LOGD("Get attribute op_input_l1_addr of op[%s, %s] unsuccessful.", op_desc->GetName().c_str(),
              op_desc->GetType().c_str());
      return;
    }
    if (!ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OP_INPUT_L1_VALID_SIZE, op_input_l1_valid_size)) {
      FE_LOGD("Get attr op_input_l1_valid_size of op[%s, %s] unsuccessful.", op_desc->GetName().c_str(),
              op_desc->GetType().c_str());
      return;
    }
  }
  FE_LOGD("The value of OP_INPUT_L1_FLAG, OP_INPUT_L1_ADDR, OP_INPUT_L1_VALID_SIZE of node[%s] are [%s], [%s], [%s].",
          op_desc->GetName().c_str(), StringUtils::IntegerVecToString(op_input_l1_flag).c_str(),
          StringUtils::IntegerVecToString(op_input_l1_addr).c_str(),
          StringUtils::IntegerVecToString(op_input_l1_valid_size).c_str());
}

void TbeInfoAssembler::SetL1Info(te::TbeOpTensor &input_tensor, const std::vector<int64_t> &op_input_l1_flag,
                                 const std::vector<int64_t> &op_input_l1_addr,
                                 const std::vector<int64_t> &op_input_l1_valid_size,
                                 const uint32_t &index_in_opdesc) const {
  if (op_input_l1_flag.size() > index_in_opdesc) {
    input_tensor.SetL1AddrFlag(op_input_l1_flag[index_in_opdesc]);
  }
  if (op_input_l1_addr.size() > index_in_opdesc) {
    input_tensor.SetAddrOffset(op_input_l1_addr[index_in_opdesc]);
  }
  if (op_input_l1_valid_size.size() > index_in_opdesc) {
    input_tensor.SetL1ValidSize(op_input_l1_valid_size[index_in_opdesc]);
  }
}

/*
 *  @ingroup fe
 *  @brief   set inputs to tbe_op_info
 *  @param   [in]  node            input node pointer
 *  @param   [in]  input_idx_name_map map with input index as key and input name as
 * value
 *  @param   [in]  op_kernel_info_ptr tensor from const node
 *  @param   [in/out]  op_info      tbe data item
 *  @return  SUCCESS or FAILED
 */
Status TbeInfoAssembler::FeedInputsToTbeOpInfo(const ge::Node *node, IndexNameMap &input_idx_name_map,
                                               OpKernelInfoPtr op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  auto op_desc = node->GetOpDesc();
  auto &input_info_in_opkernel = op_kernel_info_ptr->GetAllInputInfo();
  auto input_size_in_op_kernel = input_info_in_opkernel.size();
  vector<int32_t> memery_type_vec;
  ToOpStructPtr l1_info;
  FE_MAKE_SHARED(l1_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  ToOpStructPtr l2_info;
  FE_MAKE_SHARED(l2_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  (void)ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, memery_type_vec);
  GetL1ToOpStructFromJson(op_desc, l1_info);
  GetL2ToOpStructFromJson(op_desc, l2_info);
  GetStridedToOpStructFromJson(op_desc, optimize_info, kSplitCOptimizeInfoPtr, kSplitCOptimizeInfoStr);

  std::vector<int64_t> op_input_l1_flag;
  std::vector<int64_t> op_input_l1_addr;
  std::vector<int64_t> op_input_l1_valid_size;
  GetOpInputL1Attr(op_desc, op_input_l1_flag, op_input_l1_addr, op_input_l1_valid_size);

  auto memery_type_vec_size = memery_type_vec.size();
  auto input_size = op_desc->GetAllInputsSize();
  te::TbeOpParam input;
  for (size_t index_in_op_kernel = 0; index_in_op_kernel < input_size_in_op_kernel; index_in_op_kernel++) {
    auto input_info_ptr = input_info_in_opkernel.at(index_in_op_kernel);
    FE_CHECK(input_info_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][FeedInputs] InputInfoPtr is nullptr."),
             return FAILED);
    string input_name_in_op_kernel = input_info_ptr->GetName();

    vector<uint32_t> specific_input_index_vec_in_op_desc;
    if (GetSpecificIndex(*op_desc.get(), input_idx_name_map, input_name_in_op_kernel, true,
                         specific_input_index_vec_in_op_desc) != SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][PreCompileOp][FeedInputs] Node [%s, %s]: failed to obtain input name %s.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), input_name_in_op_kernel.c_str());
      return FAILED;
    }

    auto size_of_this_input = specific_input_index_vec_in_op_desc.size();
    bool add_tensor_info = GetAddTensorFlag(op_desc, specific_input_index_vec_in_op_desc);
    if (add_tensor_info) {
      uint32_t count = 0;
      for (uint32_t index_in_opdesc : specific_input_index_vec_in_op_desc) {
        auto input_i = op_desc->GetInputDesc(index_in_opdesc);
        vector<int64_t> dims;
        te::TbeOpTensor input_tensor(input_name_in_op_kernel, dims, "", "");
        if (SetInputTensorBaseInfo(op_desc, index_in_opdesc, input_tensor) != SUCCESS) {
          REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][FeedInputs][Op %s, type %s]: failed to setInputTensorBaseInfo.",
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
          return FAILED;
        }
        FE_LOGD("Get attr fusion userinfo size %zu, input size %zu.", memery_type_vec_size, input_size);
        bool add_addr_type = (memery_type_vec_size == input_size && index_in_opdesc < memery_type_vec_size);
        if (add_addr_type) {
          input_tensor.SetAddrType(memery_type_vec[index_in_opdesc]);
        }
        FE_LOGD("get addr_type %zu", input_tensor.GetAddrType());
        if (l1_info != nullptr) {
          FeedL1InputTensor(l1_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
          SetL1Info(input_tensor, op_input_l1_flag, op_input_l1_addr, op_input_l1_valid_size, index_in_opdesc);
        }

        FeedL2InputTensor(l2_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
        bool no_l2_info = l2_info == nullptr ||
                          l2_info->slice_input_shape.empty() || l2_info->slice_input_shape.at(0).empty();
        bool only_optimize_info = (l1_info == nullptr && no_l2_info && optimize_info != nullptr);
        if (only_optimize_info) {
          FeedL2InputTensor(optimize_info, op_desc, input_idx_name_map, index_in_opdesc, input_tensor);
        }

        /* Set original format and dtype */
        ge::Format primary_origin_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_i.GetOriginFormat()));
        input_tensor.SetOriginFormat(ge::TypeUtils::FormatToSerialString(primary_origin_format));
        std::vector<int64_t> ori_shape_vec;
        (void)ge::AttrUtils::GetListInt(input_i, kSoftSyncDynOriShape, ori_shape_vec);
        if (ori_shape_vec.empty()) {
          ori_shape_vec = input_i.GetOriginShape().GetDims();
        }
        input_tensor.SetOriginShape(ori_shape_vec);
        FE_LOGD("Op[name=%s,type=%s]: origin input format is [%s], the index_in_op_kernel is [%zu].",
                op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                ge::TypeUtils::FormatToSerialString(input_i.GetOriginFormat()).c_str(), index_in_op_kernel);
        string input_origin_shape_dims = GetShapeDims(ori_shape_vec);
        SetCDimReshapeValue(input_i, input_tensor);
        FE_LOGD("Op[name=%s,type=%s]: origin input shape is [%s].", op_desc->GetName().c_str(),
                op_desc->GetType().c_str(), input_origin_shape_dims.c_str());
        SetInputDdrBaseProp(node, index_in_opdesc, input_tensor);
        FE_LOGD("Ddr base prop of op[%s, %s]'s input[%u] is [%d].",
                op_desc->GetName().c_str(), op_desc->GetType().c_str(), index_in_opdesc,
                static_cast<int32_t>(input_tensor.GetDdrBaseProp()));
        if (SetTensorConstValue(node, index_in_opdesc, input_info_ptr, input_tensor) != SUCCESS) {
          REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][FeedInputs][Op %s, type %s]: Failed to set value for input[%u].",
                          op_desc->GetName().c_str(), op_desc->GetType().c_str(), index_in_opdesc);
          return FAILED;
        }
        // take care that danamic input have multiple sub-inputs.
        if (FeedParameterInfoForInput(node, input_info_ptr, static_cast<int>(index_in_opdesc),
                                      (count == size_of_this_input - 1), input_tensor, input, op_info) != SUCCESS) {
          REPORT_FE_ERROR(
              "[SubGraphOpt][PreCompileOp][FeedInputs] Node [%s, %s]: failed to FeedParameterInfoForInput.",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
          return FAILED;
        }
        count++;
      }
    } else {
      if (FeedParameterInfoForNotFound(input_info_ptr, STR_INPUT_LOWERCASE, input, op_info) != SUCCESS) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][PreCompileOp][FeedInputs] Node[%s, %s]: failed to FeedParameterInfoForNotFound.",
            op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
      }
      FE_LOGD("Optional input %u named [%s] is missing in Opdesc, we set a empty TbeOpTensor when feeds input.",
              index_in_op_kernel, input_info_ptr->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status TbeInfoAssembler::FeedAutoFuseInoutputsToTbeOpInfo(const ge::Node *node, const bool is_input,
                                                      te::TbeOpInfo &tbe_op_info) const {
  FE_LOGD("Feed autofuse input %u info.", is_input);
  auto op = node->GetOpDesc();
  auto all_inout_desc = (is_input ? op->GetAllInputsDescPtr() : op->GetAllOutputsDescPtr());
  for (const auto &inoutput : all_inout_desc) {
    te::TbeOpTensor inoutput_tensor(inoutput->GetName(), inoutput->GetShape().GetDims(), "", "");

    ge::DataType dtype = inoutput->GetDataType();
    auto format = inoutput->GetFormat();
    auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(format));
    auto sub_format = static_cast<ge::Format>(ge::GetSubFormat(format));

    std::string dtype_str = "";
    FE_CHECK(TransDtypeToString(dtype, dtype_str) != SUCCESS,
             FE_LOGW("Current data type[%u] of op (name [%s], type [%s]) is not found.", dtype,
              op->GetName().c_str(), op->GetType().c_str()), return FAILED);
    FE_LOGD("AssembleAutoFuseTbeInfo Op[name=%s,type=%s]: format is [%s], dtype is [%s], shape is [%s].",
            op->GetName().c_str(), op->GetType().c_str(), ge::TypeUtils::FormatToSerialString(format).c_str(),
            dtype_str.c_str(), GetShapeDims(inoutput->GetShape().GetDims()).c_str());
    string primary_format_str = ge::TypeUtils::FormatToSerialString(primary_format);
    inoutput_tensor.SetType(dtype_str);
    inoutput_tensor.SetFormat(primary_format_str);
    inoutput_tensor.SetSubFormat(sub_format);

    te::TbeOpParam tbe_op_param;
    tbe_op_param.SetType(te::TensorType::TT_REQ);
    tbe_op_param.SetTensor(inoutput_tensor);

    if (is_input) {
      tbe_op_info.AddInput(tbe_op_param);
    } else {
      tbe_op_info.AddOutput(tbe_op_param);
    }
  }
  return SUCCESS;
}

Status TbeInfoAssembler::FindAndCheckEndNodeForConstValue(const ge::Node *node, const uint32_t &tensor_index,
                                                          InputOrOutputInfoPtr tensor_info_ptr,
                                                          ge::NodePtr &other_end_node, bool &is_const_node) const {
  // find the node of other end
  is_const_node = FeGraphUtils::IsPeerOutConst(node, static_cast<int>(tensor_index), other_end_node);
  bool required_check_invalid = (tensor_info_ptr->GetConstValueDepend() == CONST_REQUIRED && !is_const_node);
  if (required_check_invalid) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][PreCompileOp][SetTensorConstVal] Node [%s, %s]: failed to obtain constant value for input [%u].",
        node->GetName().c_str(), node->GetType().c_str(), tensor_index);
    return FAILED;
  }
  return SUCCESS;
}

void TbeInfoAssembler::SetInputDdrBaseProp(const ge::Node *node, const uint32_t &tensor_index,
                                           te::TbeOpTensor &input_tensor) const {
  if (!PlatformUtils::Instance().IsSpecifiedMemBase()) {
    return;
  }
  FE_CHECK(node == nullptr, FE_LOGI("Node is null."), return);
  if (node->GetInDataAnchor(tensor_index) == nullptr ||
      node->GetInDataAnchor(tensor_index)->GetPeerOutAnchor() == nullptr ||
      node->GetInDataAnchor(tensor_index)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
    FE_LOGI("The peer node of op[%s, %s]'s input[%u] is null.",
            node->GetName().c_str(), node->GetType().c_str(), tensor_index);
    return;
  }
  ge::NodePtr peer_node = node->GetInDataAnchor(tensor_index)->GetPeerOutAnchor()->GetOwnerNode();
  FE_LOGD("Peer node of op[%s, %s]'s input[%u] is [%s, %s].", node->GetName().c_str(), node->GetType().c_str(),
          tensor_index, peer_node->GetName().c_str(), peer_node->GetType().c_str());
  std::string parent_op_type;
  FeGraphUtils::FindPeerOpType(peer_node, true, parent_op_type);
  FE_LOGD("Parent op type of op[%s, %s]'s input[%u] is [%s].", node->GetName().c_str(), node->GetType().c_str(),
          tensor_index, parent_op_type.c_str());
  auto tensor_desc = node->GetOpDesc()->MutableInputDesc(tensor_index);
  if (tensor_desc == nullptr) {
    FE_LOGW("Node[%s, %s], unexpected tensor desc nullptr, input idx is: %u.", node->GetNamePtr(),
            node->GetTypePtr(), tensor_index);
  }
  int64_t input_memory_scope = 0;
  (void)ge::AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_TENSOR_MEMORY_SCOPE, input_memory_scope);
  if (input_memory_scope != 0) {
    input_tensor.SetDdrBaseProp(static_cast<te::DdrBaseType>(input_memory_scope));
    FE_LOGD("The ddr base prop of op[%s, %s]'s input[%u] is [%ld].",
            node->GetNamePtr(), node->GetTypePtr(), tensor_index, input_memory_scope);
    return;
  }
  if (parent_op_type == CONSTANT || parent_op_type == CONSTANTOP) {
    // set weight
    input_tensor.SetDdrBaseProp(te::DdrBaseType::WEIGHT);
    FE_LOGD("The ddr base type of op[%s, %s]'s input[%u] is weight.",
            node->GetName().c_str(), node->GetType().c_str(), tensor_index);
    return;
  }
  if (IsRootGraphData(parent_op_type)) {
    // set net edge
    input_tensor.SetDdrBaseProp(te::DdrBaseType::NET_EDGE);
    FE_LOGD("The ddr base type of op[%s, %s]'s input[%u] is data.",
            node->GetName().c_str(), node->GetType().c_str(), tensor_index);
    return;
  }
  for (auto &peer_peer_in_anchor : node->GetInDataAnchor(tensor_index)->GetPeerOutAnchor()->GetPeerInDataAnchors()) {
    if (peer_peer_in_anchor == nullptr || peer_peer_in_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr peer_peer_in_node = peer_peer_in_anchor->GetOwnerNode();
    FE_LOGD("Peer peer in node of op[%s, %s]'s input[%u] is [%s, %s].", node->GetName().c_str(),
            node->GetType().c_str(), tensor_index, peer_peer_in_node->GetName().c_str(),
            peer_peer_in_node->GetType().c_str());
    std::string peer_peer_in_node_parent_op_type;
    FeGraphUtils::FindPeerOpType(peer_peer_in_node, false, peer_peer_in_node_parent_op_type);
    FE_LOGD("Peer peer in node parent op type of op[%s, %s]'s input[%u] is [%s].", node->GetName().c_str(),
            node->GetType().c_str(), tensor_index, peer_peer_in_node_parent_op_type.c_str());
    if (peer_peer_in_node_parent_op_type == NETOUTPUT) {
      // set net_edge
      input_tensor.SetDdrBaseProp(te::DdrBaseType::NET_EDGE);
      FE_LOGD("The ddr base type of op[%s, %s]'s input[%u] is NetOutput.",
              node->GetName().c_str(), node->GetType().c_str(), tensor_index);
      break;
    }
  }
}

void TbeInfoAssembler::SetOutputDdrBaseProp(const ge::Node *node, const uint32_t &tensor_index,
                                            te::TbeOpTensor &output_tensor) const {
  if (!PlatformUtils::Instance().IsSpecifiedMemBase()) {
    return;
  }
  FE_CHECK(node == nullptr, FE_LOGI("Node is null."), return);
  if (node->GetOutDataAnchor(tensor_index) == nullptr) {
    FE_LOGI("The peer node of op[%s, %s]'s output[%u] is null.",
            node->GetName().c_str(), node->GetType().c_str(), tensor_index);
    return;
  }
  auto tensor_desc = node->GetOpDesc()->MutableOutputDesc(tensor_index);
  if (tensor_desc == nullptr) {
    FE_LOGW("Node[%s, %s], unexpected tensor desc nullptr, output idx is: %u.", node->GetNamePtr(),
            node->GetTypePtr(), tensor_index);
  }
  int64_t output_memory_scope = 0;
  (void)ge::AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_TENSOR_MEMORY_SCOPE, output_memory_scope);
  if (output_memory_scope != 0) {
    output_tensor.SetDdrBaseProp(static_cast<te::DdrBaseType>(output_memory_scope));
    FE_LOGD("The ddr base prop of op[%s, %s]'s output[%u] is [%ld].",
            node->GetNamePtr(), node->GetTypePtr(), tensor_index, output_memory_scope);
    return;
  }
  for (const ge::InDataAnchorPtr &in_data_anchor_ptr : node->GetOutDataAnchor(tensor_index)->GetPeerInDataAnchors()) {
    if (in_data_anchor_ptr == nullptr || in_data_anchor_ptr->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr peer_node = in_data_anchor_ptr->GetOwnerNode();
    FE_LOGD("Peer node of op[%s, %s]'s output[%u] is [%s, %s].", node->GetName().c_str(), node->GetType().c_str(),
            tensor_index, peer_node->GetName().c_str(), peer_node->GetType().c_str());
    std::string parent_op_type;
    FeGraphUtils::FindPeerOpType(peer_node, false, parent_op_type);
    FE_LOGD("Parent op type of op[%s, %s]'s output[%u] is [%s].", node->GetName().c_str(), node->GetType().c_str(),
            tensor_index, parent_op_type.c_str());
    if (parent_op_type == NETOUTPUT) {
      // set net_edge
      output_tensor.SetDdrBaseProp(te::DdrBaseType::NET_EDGE);
      FE_LOGD("The ddr base type of op[%s, %s]'s output[%u] is NetOutput.",
              node->GetName().c_str(), node->GetType().c_str(), tensor_index);
      break;
    }
  }
}

void TbeInfoAssembler::SetIsNullOutputFlag(const ge::Node *node, const uint32_t &tensor_index,
 	                                            te::TbeOpTensor &output_tensor) const {
 	FE_CHECK(node == nullptr, FE_LOGI("Node is null."), return);
 	auto tensor_desc = node->GetOpDesc()->MutableOutputDesc(tensor_index);
 	if (tensor_desc == nullptr) {
 	  FE_LOGW("Node[%s, %s], unexpected tensor desc nullptr, output idx is: %u.", node->GetNamePtr(),
 	          node->GetTypePtr(), tensor_index);
 	}
 	bool is_null_output = false;
 	(void)ge::AttrUtils::GetBool(tensor_desc, ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
 	if(is_null_output) {
 	  FE_LOGD("The peer node of op[%s, %s]'s output[%u] has attribute is_null_output %d.",
 	    node->GetName().c_str(), node->GetType().c_str(), tensor_index, is_null_output);
 	  output_tensor.SetIsNullOutput(is_null_output);
 	}
}

void TbeInfoAssembler::SetIsConstInputFlag(const ge::Node *node, const ge::OpDesc &op_desc,
                                           const uint32_t &tensor_index, te::TbeOpTensor &input_tensor) const {
  if (CheckAOETuning(node)) {
    FE_LOGD("op[%s, %s] will not set input_const for the aoe_tuning scenario.",
      op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return;
  }
  ge::NodePtr peer_out_node = nullptr;
  bool is_const_node = FeGraphUtils::IsPeerOutConst(node, static_cast<int>(tensor_index), peer_out_node);
  if (is_const_node) {
    input_tensor.SetIsInputConst(1);
  }
  input_tensor.SetIsInputConst(0);
}

Status TbeInfoAssembler::SetTensorConstValue(const ge::Node *node, const uint32_t &tensor_index,
                                             InputOrOutputInfoPtr tensor_info_ptr, te::TbeOpTensor &op_tensor) const {
  auto op_desc = node->GetOpDesc();
  FE_LOGD("Begin to set const value for input[%s] of node[%s, %s].", tensor_info_ptr->GetName().c_str(),
          node->GetName().c_str(), node->GetType().c_str());

  ge::GeTensorPtr const_tensor_ptr;
  auto tensor_desc = op_desc->MutableInputDesc(tensor_index);
  FE_CHECK_NOTNULL(tensor_desc);
  // fuzz build, GE set const value to op_desc by tensor attr, change const node to data node
  if (ge::AttrUtils::MutableTensor(tensor_desc, ge::ATTR_NAME_VALUE, const_tensor_ptr)) {
    if (AssembleConstValue(const_tensor_ptr, op_desc, op_tensor) == SUCCESS) {
      FE_LOGD("Set index[%u] of node[%s] const value successfully.", tensor_index, node->GetName().c_str());
      return SUCCESS;
    }
  }

  if (tensor_info_ptr->GetConstValueDepend() == CONST_IGNORE) {
    return SUCCESS;
  }

  // find the node of other end
  ge::NodePtr other_end_node = nullptr;
  bool is_const_node = false;
  if (FindAndCheckEndNodeForConstValue(node, tensor_index, tensor_info_ptr, other_end_node, is_const_node) != SUCCESS) {
    return FAILED;
  }

  bool optional_check_invalid = (tensor_info_ptr->GetConstValueDepend() == CONST_OPTIONAL && !is_const_node);
  if (optional_check_invalid) {
    FE_LOGD("The const value input[%u] in node[%s, %s] is optional, and the input is not linked to a const node.",
            tensor_index, node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  FE_CHECK_NOTNULL(other_end_node);
  auto const_op_desc = other_end_node->GetOpDesc();
  FE_CHECK_NOTNULL(const_op_desc);
  FE_LOGD("Begin to get const data from node[%s, %s].",
          other_end_node->GetName().c_str(), other_end_node->GetType().c_str());
  vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(other_end_node);
  if (weights.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleConstVal][Op %s,type %s]: Const node does not have weight.",
                    const_op_desc->GetName().c_str(), const_op_desc->GetType().c_str());
    return FAILED;
  }
  const_tensor_ptr = weights[0];
  if (AssembleConstValue(const_tensor_ptr, const_op_desc, op_tensor) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

Status TbeInfoAssembler::AssembleConstValue(ge::GeTensorPtr const_tensor_ptr, const ge::OpDescPtr &op_desc,
                                            te::TbeOpTensor &op_tensor) const {
  FE_LOGD("Begin AssembleConstValue of node[%s].", op_desc->GetName().c_str());
  string tensor_name;
  (void)op_tensor.GetName(tensor_name);
  FE_CHECK_NOTNULL(const_tensor_ptr);
  auto &out_tensor_desc = const_tensor_ptr->GetTensorDesc();
  FE_CHECK_NOTNULL(const_tensor_ptr->GetData().GetData());

  ge::DataType data_type = out_tensor_desc.GetDataType();
  auto iter_set_const_value_func = set_const_value_func_map.find(data_type);
  if (iter_set_const_value_func == set_const_value_func_map.end()) {
    FE_LOGW(
        "[SubGraphOpt][PreCompileOp][AssembleConstVal] Node[%s, %s]: data type %s is not supported yet.",
        op_desc->GetName().c_str(), op_desc->GetType().c_str(), DTypeToStr(data_type).c_str());
    return FAILED;
  }

  SetConstValueWithDtypePtr set_const_value_func = nullptr;
  FE_MAKE_SHARED(set_const_value_func = iter_set_const_value_func->second, return FAILED);
  FE_CHECK_NOTNULL(set_const_value_func);

  Status status = (*set_const_value_func)(const_tensor_ptr, tensor_name, op_tensor);
  if (status != SUCCESS) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][PreCompileOp][AssembleConstVal] Failed to set constant value for node [%s, %s] with data type %s.",
        op_desc->GetName().c_str(), op_desc->GetType().c_str(), DTypeToStr(data_type).c_str());
    return FAILED;
  }

  FE_LOGD("Op tensor[%s] has been set const value whose data type is %s.", tensor_name.c_str(),
          DTypeToStr(data_type).c_str());
  return SUCCESS;
}

Status CreateTbeTensor(const ge::OpDesc &op_desc, const TensorDescAndIndex &tensor_info, te::TbeOpTensor &tbe_tensor) {
  auto &shape = tensor_info.tensor_desc_ptr->MutableShape();
  ge::DataType dtype = tensor_info.tensor_desc_ptr->GetDataType();
  std::string dtype_str = "";
  Status ret = TransDtypeToString(dtype, dtype_str);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG(EM_INNER_ERROR.c_str(), "Current data type[%d] of %s %u of op (name [%s], type [%s]) is not found.",
                       dtype, IS_INPUT_TO_STRING(tensor_info.is_input), tensor_info.index_in_opdesc,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return FAILED;
  }

  auto format = tensor_info.tensor_desc_ptr->GetFormat();
  auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(format));
  auto sub_format = ge::GetSubFormat(format);
  if (tensor_info.propagat_heavy_format != ge::FORMAT_RESERVED) {
    sub_format = tensor_info.propagat_sub_format;
  }
  string primary_format_str = ge::TypeUtils::FormatToSerialString(primary_format);
  FE_LOGD("Node[%s, %s]: current primary_format of %s is %s. Sub format is %d. The index in op kernel is %zu.",
          op_desc.GetName().c_str(), op_desc.GetType().c_str(), IS_INPUT_TO_STRING(tensor_info.is_input),
          primary_format_str.c_str(), sub_format, tensor_info.index_in_op_kernel);
  std::vector<int64_t> shape_vec;
  (void)ge::AttrUtils::GetListInt(tensor_info.tensor_desc_ptr, kSoftSyncDynShape, shape_vec);
  if (shape_vec.empty()) {
    shape_vec = shape.GetDims();
  }
  FE_LOGD("Op[name=%s,type=%s]: current %s shape is [%s], tensor shape is [%s].", op_desc.GetName().c_str(),
          op_desc.GetType().c_str(), IS_INPUT_TO_STRING(tensor_info.is_input),
          GetShapeDims(shape_vec).c_str(), GetShapeDims(shape.GetDims()).c_str());

  // If dim_num is 0, the op will compile fail. So need set {1} to shape.
  if (shape.IsScalar()) {
    shape_vec = {1};
  }

  tbe_tensor = te::TbeOpTensor(tensor_info.name_in_op_kernel, shape_vec, dtype_str, primary_format_str);
  tbe_tensor.SetSubFormat(sub_format);
  return SUCCESS;
}

Status TbeInfoAssembler::ConvertInputsToTbeOpInfo(const ge::NodePtr &node, IndexNameMap &input_idx_name_map,
                                                  OpKernelInfoPtr op_kernel_info_ptr,
                                                  const HeavyFormatInfo &heavy_format_info,
                                                  te::TbeOpInfo &op_info) const {
  ge::OpDesc &op_desc = *(node->GetOpDesc().get());
  auto &input_info_in_opkernel = op_kernel_info_ptr->GetAllInputInfo();
  auto input_size_in_op_kernel = input_info_in_opkernel.size();
  bool is_first_layer_conv = false;
  (void)ge::AttrUtils::GetBool(op_desc, IS_FIRST_LAYER_CONV_FOR_OP, is_first_layer_conv);

  te::TbeOpParam input;
  for (size_t index_in_op_kernel = 0; index_in_op_kernel < input_size_in_op_kernel; index_in_op_kernel++) {
    auto input_info_ptr = input_info_in_opkernel.at(index_in_op_kernel);
    FE_CHECK(input_info_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleConstVal]inputInfoPtr is nullptr."),
             return FAILED);
    string input_name_in_op_kernel = input_info_ptr->GetName();

    vector<uint32_t> specific_input_index_vec_in_op_desc;
    if (GetSpecificIndex(op_desc, input_idx_name_map, input_name_in_op_kernel, true,
                         specific_input_index_vec_in_op_desc) != SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][PreCompileOp][AssembleConstVal] Node[%s, %s]: fail to get input name %s.",
          op_desc.GetName().c_str(), op_desc.GetType().c_str(), input_name_in_op_kernel.c_str());
      return FAILED;
    }

    auto size_of_this_input = specific_input_index_vec_in_op_desc.size();
    bool add_tensor_info = GetAddTensorFlag(op_desc, specific_input_index_vec_in_op_desc);
    if (add_tensor_info) {
      uint32_t count = 0;
      for (uint32_t index_in_opdesc : specific_input_index_vec_in_op_desc) {
        te::TbeOpTensor input_tensor;
        auto input_i = op_desc.MutableInputDesc(index_in_opdesc);
        if (input_i == nullptr) {
          FE_LOGD("[SubGraphOpt][PreCompileOp][AssembleConstVal]input_i is nullptr.");
          continue;
        }
        TensorDescAndIndex tensor_info = {input_i,
                                          input_name_in_op_kernel,
                                          index_in_op_kernel,
                                          index_in_opdesc,
                                          true,
                                          heavy_format_info.expected_heavy_format,
                                          heavy_format_info.sub_format,
                                          is_first_layer_conv};
        Status ret = SetTbeTensor(op_desc, tensor_info, input_tensor);
        if (ret != SUCCESS) {
          return ret;
        }
        SetInputDdrBaseProp(node.get(), index_in_opdesc, input_tensor);
        FE_LOGD("Ddr base prop of op[%s, %s]'s input[%u] is [%d].",
                op_desc.GetName().c_str(), op_desc.GetType().c_str(), index_in_opdesc,
                static_cast<int32_t>(input_tensor.GetDdrBaseProp()));
        SetIsConstInputFlag(node.get(), op_desc, index_in_opdesc, input_tensor);
        if (SetTensorConstValue(node.get(), index_in_opdesc, input_info_ptr, input_tensor) != SUCCESS) {
          REPORT_FE_ERROR(
              "[SubGraphOpt][PreCompileOp][AssembleConstVal][Op %s, type %s]: Fail to set value for input[%u].",
              op_desc.GetName().c_str(), op_desc.GetType().c_str(), index_in_opdesc);
          return FAILED;
        }

        // take care that danamic input have multiple sub-inputs.
        if (ConvertParameterInfoForInput(input_info_ptr, input, input_tensor, op_info,
                                         (count == size_of_this_input - 1)) != SUCCESS) {
          REPORT_FE_ERROR(
              "[SubGraphOpt][PreCompileOp][AssembleConstVal] Node [%s, %s]: failed to feed normal input %zu.",
              op_desc.GetName().c_str(), op_desc.GetType().c_str(), index_in_op_kernel);
          return FAILED;
        }
        count++;
      }
    } else {
      if (FeedParameterInfoForNotFound(input_info_ptr, STR_INPUT_LOWERCASE, input, op_info) != SUCCESS) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][PreCompileOp][AssembleConstVal] Node[%s, %s]: fail to feed dummy input %zu.",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), index_in_op_kernel);
        return FAILED;
      }
      FE_LOGI("Optional input %zu named [%s] is missing in Opdesc, we set a empty TbeOpTensor.", index_in_op_kernel,
              input_info_ptr->GetName().c_str());
    }
  }
  return SUCCESS;
}

void TbeInfoAssembler::FeedFusionOutputTensor(const ToOpStructPtr &fusion_info, const ge::OpDescPtr &op_desc,
                                              IndexNameMap &output_idx_name_map, const uint32_t &index_in_opdesc,
                                              te::TbeOpTensor &output_tensor) const {
  // update opinfo
  if (!fusion_info->op_l1_fusion_type.empty()) {
    output_tensor.SetL1FusionType(fusion_info->op_l1_fusion_type[0]);
  }

  size_t output_idx_name_size = output_idx_name_map.size();
  size_t slice_output_shape_size = fusion_info->slice_output_shape.size();
  if (slice_output_shape_size == output_idx_name_size) {
    if (index_in_opdesc >= slice_output_shape_size) {
      FE_LOGW("index_in_opdesc >= valid_output_shape_size, index_in_opdesc:%u, valid_output_shape_size:%zu.",
              index_in_opdesc, slice_output_shape_size);
      return;
    }
    output_tensor.SetValidShape(fusion_info->slice_output_shape[index_in_opdesc]);
  }

  size_t slice_output_offset_size = fusion_info->slice_output_offset.size();
  if (slice_output_offset_size == output_idx_name_size) {
    if (index_in_opdesc >= slice_output_offset_size) {
      FE_LOGW("index_in_opdesc >= slice_output_offset_size, index_in_opdesc:%u, slice_output_offset_size:%zu.",
              index_in_opdesc, slice_output_offset_size);
      return;
    }
    output_tensor.SetSliceOffset(fusion_info->slice_output_offset[index_in_opdesc]);
  }

  /* Set addr  offset */
  size_t output_num = op_desc->GetOutputsSize();
  vector<int64_t> output_offsets = op_desc->GetOutputOffset();
  size_t output_offset_size = output_offsets.size();
  if (output_num > output_offset_size) {
    FE_LOGW("output_desc_size > output_offset_size, output_desc_size:%lu, output_offset_size:%lu.", output_num,
            output_offset_size);
  } else {
    if (index_in_opdesc >= output_offset_size) {
      FE_LOGW("index_in_opdesc >= output_offset_size, index_in_opdesc:%u, output_offset_size:%lu.", index_in_opdesc,
              output_offset_size);
      return;
    }
    output_tensor.SetAddrOffset(output_offsets[index_in_opdesc]);
  }
}

void TbeInfoAssembler::FeedOutputTensorAtomicAttr(const ffts::ThreadSliceMapPtr &slice_info,
                                                  te::TbeOpTensor &output_tensor,
                                                  const uint32_t &index_in_opdesc) const {
  if (slice_info->atomic_types.empty()) {
    return;
  }
  if (index_in_opdesc >= slice_info->atomic_types.size()) {
    return;
  }
  ffts::AtomicType atomic_type = slice_info->atomic_types[index_in_opdesc];
  if (atomic_type == ffts::AtomicType::None) {
    return;
  }
  output_tensor.SetAtomicType(AtomicTypeStr[atomic_type]);
  return;
}
Status TbeInfoAssembler::FeedOutputsToTbeOpInfo(const ge::Node *node, IndexNameMap &output_idx_name_map,
                                                OpKernelInfoPtr op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  auto op = node->GetOpDesc();
  auto &output_info_in_opkernel = op_kernel_info_ptr->GetAllOutputInfo();
  auto output_size_in_op_kernel = output_info_in_opkernel.size();
  vector<int32_t> memery_type_vec;
  ToOpStructPtr l1_info;
  FE_MAKE_SHARED(l1_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  ToOpStructPtr l2_info;
  FE_MAKE_SHARED(l2_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  ToOpStructPtr optimize_info = nullptr;
  FE_MAKE_SHARED(optimize_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::GetListInt(op, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memery_type_vec);
  auto memery_type_vec_size = memery_type_vec.size();
  (void)GetL1ToOpStructFromJson(op, l1_info);
  (void)GetL2ToOpStructFromJson(op, l2_info);
  GetStridedToOpStructFromJson(op, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);

  te::TbeOpParam output;
  auto output_size = op->GetOutputsSize();
  for (uint32_t index_in_op_kernel = 0; index_in_op_kernel < output_size_in_op_kernel; index_in_op_kernel++) {
    auto output_info_ptr = output_info_in_opkernel.at(index_in_op_kernel);
    FE_CHECK(output_info_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][FeedOutputs] OutputInfoPtr is nullptr."),
             return FAILED);
    string output_name_in_op_kernel = output_info_ptr->GetName();

    vector<uint32_t> specific_output_index_vec_in_op_desc;
    if (GetSpecificIndex(*op.get(), output_idx_name_map, output_name_in_op_kernel, false,
                         specific_output_index_vec_in_op_desc) != SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][PreCompileOp][FeedOutputs] Node [%s, %s]: failed to get output name %s.",
          op->GetName().c_str(), op->GetType().c_str(), output_name_in_op_kernel.c_str());
      return FAILED;
    }
    auto size_of_this_output = specific_output_index_vec_in_op_desc.size();
    // dynamic, required, optional
    if (size_of_this_output != 0) {
      uint32_t count = 0;
      for (uint32_t index_in_opdesc : specific_output_index_vec_in_op_desc) {
        te::TbeOpTensor output_tensor;
        auto output_desc = op->MutableOutputDesc(index_in_opdesc);
        if (output_desc == nullptr) {
          continue;
        }
        TensorDescAndIndex tensor_info = {output_desc, output_name_in_op_kernel, index_in_op_kernel, index_in_opdesc,
                                          false};
        Status ret = CreateTbeTensor(*op.get(), tensor_info, output_tensor);
        if (ret != SUCCESS) {
          return ret;
        }
        FE_LOGD("Get attr l1 userinfo, l1 info size %zu, output size %zu.", memery_type_vec_size, output_size);

        if (memery_type_vec_size == output_size) {
          output_tensor.SetAddrType(memery_type_vec[index_in_opdesc]);
        }
        if (l1_info != nullptr) {
          FeedFusionOutputTensor(l1_info, op, output_idx_name_map, index_in_opdesc, output_tensor);
        }

        if (l2_info != nullptr) {
          FeedFusionOutputTensor(l2_info, op, output_idx_name_map, index_in_opdesc, output_tensor);
        }

        bool no_l2_info = l2_info == nullptr ||
                          l2_info->slice_output_shape.empty() || l2_info->slice_output_shape.at(0).empty();
        bool only_optimize_info = (l1_info == nullptr && no_l2_info && optimize_info != nullptr);
        if (only_optimize_info) {
          FeedFusionOutputTensor(optimize_info, op, output_idx_name_map, index_in_opdesc, output_tensor);
        }

        if (slice_info_ptr != nullptr) {
          FeedOutputTensorAtomicAttr(slice_info_ptr, output_tensor, index_in_opdesc);
        }

        SetOutputDdrBaseProp(node, index_in_opdesc, output_tensor);
        FE_LOGD("Ddr base prop of op[%s, %s]'s output[%u] is [%d].", op->GetName().c_str(), op->GetType().c_str(),
                index_in_opdesc, static_cast<int32_t>(output_tensor.GetDdrBaseProp()));
        SetIsNullOutputFlag(node, index_in_opdesc, output_tensor);
        SetTbeTensorShape(*op.get(), tensor_info, output_tensor);
        // take care that danamic output have multiple sub-outputs.
        if (FeedParameterInfoForOutput(*(node->GetOpDesc().get()), *(output_desc.get()), output_info_ptr,
                                       (count == size_of_this_output - 1), output_tensor, output, op_info) != SUCCESS) {
          REPORT_FE_ERROR(
              "[SubGraphOpt][PreCompileOp][FeedOutputs] Node[%s, %s]: FeedDynamicOutputsToTbeOpInfo failed.",
              op->GetName().c_str(), op->GetType().c_str());
          return FAILED;
        }
        count++;
      }
    } else {
      if (FeedParameterInfoForNotFound(output_info_ptr, STR_OUTPUT_LOWERCASE, output, op_info) != SUCCESS) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][PreCompileOp][FeedOutputs] Node[%s, %s]: FeedDynamicOutputsToTbeOpInfo failed.",
            op->GetName().c_str(), op->GetType().c_str());
        return FAILED;
      }
      FE_LOGD("Optional output %u named [%s] is missing in Opdesc, we set a empty TbeOpTensor.", index_in_op_kernel,
              output_info_ptr->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status TbeInfoAssembler::ConvertOutputsToTbeOpInfo(const ge::NodePtr &node, IndexNameMap &output_idx_name_map,
                                                   OpKernelInfoPtr op_kernel_info_ptr,
                                                   const HeavyFormatInfo &heavy_format_info,
                                                   te::TbeOpInfo &op_info) const {
  auto &output_info_in_opkernel = op_kernel_info_ptr->GetAllOutputInfo();
  auto output_size_in_op_kernel = output_info_in_opkernel.size();

  te::TbeOpParam output;
  for (uint32_t index_in_op_kernel = 0; index_in_op_kernel < output_size_in_op_kernel; index_in_op_kernel++) {
    auto output_info_ptr = output_info_in_opkernel.at(index_in_op_kernel);
    FE_CHECK(output_info_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][ConvertOutputs] OutputInfoPtr is nullptr."),
             return FAILED);
    string output_name = output_info_ptr->GetName();
    auto op_desc = *(node->GetOpDesc());
    vector<uint32_t> specific_output_index_vec_in_op_desc;
    if (GetSpecificIndex(op_desc, output_idx_name_map, output_name, false, specific_output_index_vec_in_op_desc) !=
        SUCCESS) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][PreCompileOp][ConvertOutputs] Node[%s, %s]: failed to obtain output name %s.",
          op_desc.GetName().c_str(), op_desc.GetType().c_str(), output_name.c_str());
      return FAILED;
    }
    auto size_of_this_output = specific_output_index_vec_in_op_desc.size();
    if (size_of_this_output != 0) {
      uint32_t count = 0;
      for (uint32_t index_in_opdesc : specific_output_index_vec_in_op_desc) {
        te::TbeOpTensor output_tensor;
        auto output_desc = op_desc.MutableOutputDesc(index_in_opdesc);
        if (output_desc == nullptr) {
          continue;
        }
        TensorDescAndIndex tensor_info = {output_desc,
                                          output_name,
                                          index_in_op_kernel,
                                          index_in_opdesc,
                                          false,
                                          heavy_format_info.expected_heavy_format,
                                          heavy_format_info.sub_format};
        Status ret = SetTbeTensor(op_desc, tensor_info, output_tensor);
        if (ret != SUCCESS) {
          return ret;
        }
        SetOutputDdrBaseProp(node.get(), index_in_opdesc, output_tensor);
        // take care that danamic output have multiple sub-outputs.
        if (FeedParameterInfoForOutput(op_desc, *(output_desc.get()), output_info_ptr,
                                       (count == size_of_this_output - 1), output_tensor, output, op_info) != SUCCESS) {
          REPORT_FE_ERROR(
              "[SubGraphOpt][PreCompileOp][ConvertOutputs][Op %s,type %s]: FeedDynamicOutputsToTbeOpInfo failed.",
              op_desc.GetName().c_str(), op_desc.GetType().c_str());
          return FAILED;
        }
        count++;
      }
    } else {
      if (FeedParameterInfoForNotFound(output_info_ptr, STR_OUTPUT_LOWERCASE, output, op_info) != SUCCESS) {
        REPORT_FE_ERROR(
            "[SubGraphOpt][PreCompileOp][ConvertOutputs] Node[%s, %s]: FeedDynamicOutputsToTbeOpInfo failed.",
            op_desc.GetName().c_str(), op_desc.GetType().c_str());
        return FAILED;
      }
      FE_LOGI("Optional output %u named [%s] is missing in Opdesc, we set a empty TbeOpTensor.", index_in_op_kernel,
              output_info_ptr->GetName().c_str());
    }
  }
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   set Attrs to tbe_op_info
 *  @param   [in]  op              op desc
 *  @param   [in]  op_kernel_info_ptr op kernel info
 *  @param   [in/out]  op_info      tbe data item
 *  @return  SUCCESS or FAILED
 */
Status TbeInfoAssembler::FeedAttrsToTbeOpInfo(const ge::OpDesc &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                                              te::TbeOpInfo &op_info) const {
  // load op info store and get all attr list_fe_ops_kernel_info_store
  const std::vector<AttrInfoPtr> &attrs_info = op_kernel_info_ptr->GetVecAttrInfo();
  // loop over attr list and set each of them to TbeOpInfo
  for (const AttrInfoPtr &iter : attrs_info) {
    te::TbeAttrValue attr_value;
    string attr_name = iter->GetAttrName();
    auto func = k_attr_get_funcs.find(iter->GetAttrDType());
    if (func == k_attr_get_funcs.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][FeedAttr][Op %s, type=%s]: dtype %d of attr %s is invalid.",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str(), iter->GetAttrDType(), attr_name.c_str());
      return FAILED;
    } else {
      if (func->second(op_desc, attr_name, attr_value, iter) != SUCCESS) {
        return FAILED;
      }
    }
    attr_value.SetAttrSupAllFlag(iter->GetSupportAllValue());
    attr_value.SetIsRequiredAttr(iter->GetIsRequired());
    op_info.AddAttrValue(attr_value);
  }
  // Set op private attrs
  std::vector<te::TbeAttrValue> private_attrs_list = {};
  GetPrivateAttrsList(op_desc, private_attrs_list);
  op_info.SetPrivateAttrs(private_attrs_list);
  return SUCCESS;
}

void TbeInfoAssembler::GenerateTbePrivateAttrValue(const ge::OpDesc &op_desc,
                                                   const ge::AnyValue &value_type,
                                                   te::TbeAttrValue &tbe_attr_value, const string &attr_name) const {
  auto func = k_private_attr_get_funcs.find(value_type.GetValueType());
  if (func == k_private_attr_get_funcs.end()) {
    FE_LOGW("Current not support");
  } else {
    func->second(op_desc, value_type, tbe_attr_value, attr_name);
  }
}

void TbeInfoAssembler::GetPrivateAttrsList(const ge::OpDesc &op_desc,
                                             std::vector<te::TbeAttrValue> &private_attrs_list) const {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
                              static_cast<gert::OppImplVersionTag>(op_desc.GetOppImplVersion()));
  if (space_registry == nullptr) {
    FE_LOGE("space_registry is nullptr");
    return;
  }
  const auto op_impl_func_v2 = space_registry->GetOpImpl(op_desc.GetType().c_str());
  if (op_impl_func_v2 == nullptr) {
    return;
  }
  gert::OpImplKernelRegistry::PrivateAttrList private_attrs = op_impl_func_v2->private_attrs;
  for (auto &private_attr : private_attrs) {
    auto private_attr_name = private_attr.first.GetString();
    if (!private_attr.second.IsEmpty()) {
      te::TbeAttrValue private_attr_value;
      GenerateTbePrivateAttrValue(op_desc, private_attr.second, private_attr_value, private_attr_name);
      private_attrs_list.push_back(private_attr_value);
      continue;
    } else {
      FE_LOGW("Can not find the private attr %s from node %s",
            private_attr_name, op_desc.GetName().c_str());
    }
  }
}

Status TbeInfoAssembler::JudgeShapeToSetFlag(const ge::OpDescPtr &op_desc, const bool &is_input,
                                             te::TbeOpInfo &op_info, bool &flag) const {
  if (flag) {
    return SUCCESS;
  }
  size_t size = 0;
  if (is_input) {
    size = op_desc->GetInputsSize();
  } else {
    size = op_desc->GetOutputsSize();
  }
  for (size_t i = 0; i < size; i++) {
    auto in_out_desc = is_input ? op_desc->MutableInputDesc(i) : op_desc->MutableOutputDesc(i);
    if (in_out_desc == nullptr) {
      continue;
    }

    auto &shape = in_out_desc->MutableShape();
    ge::DataType data_type = in_out_desc->GetDataType();
    FE_CHECK(data_type == ge::DT_UNDEFINED || data_type >= ge::DT_MAX,
             REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][JudgeShape] dataType UNDEFINED."),
             return FAILED);
    int64_t sum = 1;
    int64_t max_int32 = INT32_MAX;
    for (size_t index = 0; index < shape.GetDimNum(); index++) {
      FE_MUL_OVERFLOW(sum, shape.GetDim(index), sum);
      if (sum > max_int32) {
        flag = true;
        op_info.SetFlagUseInt64(true);
        break;
      }
    }
    FE_LOGD("Shape multiplication of %d-%zu is %ld.", is_input, i, sum);

    if (flag) {
      break;
    }
  }
  return SUCCESS;
}

Status TbeInfoAssembler::FeedFlagInt64ToTbeOpInfo(const ge::Node *node, te::TbeOpInfo &op_info) const {
  auto op_desc = node->GetOpDesc();
  bool flag = false;
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    FE_LOGD("[Node: %s] is UnKnownShape.", node->GetName().c_str());
    op_info.SetFlagUseInt64(true);
  } else {
    (void)JudgeShapeToSetFlag(op_desc, true, op_info, flag);
    (void)JudgeShapeToSetFlag(op_desc, false, op_info, flag);
  }
  FE_LOGD("[Node: %s] AssembleTbeInfo FlagUseInt64: %d.", node->GetName().c_str(), op_info.GetFlagUseInt64());
  return SUCCESS;
}

Status TbeInfoAssembler::FeedIsUnknownShapeToTbeOpInfo(const ge::OpDesc &op_desc,
                                                       te::TbeOpInfo &op_info) const {
  bool is_unknown_shape = UnknownShapeUtils::IsUnknownShapeOp(op_desc);
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(op_desc, kDynamicTilingDependOp, stc_tiling_depend);
  FE_LOGD("Op[name=%s,type=%s]: is_unknown_shape flag is %d, tiling depend flag %d.", op_desc.GetName().c_str(),
          op_desc.GetType().c_str(), is_unknown_shape, stc_tiling_depend);
  if (is_unknown_shape || stc_tiling_depend) {
    op_info.SetIsUnknownShape(true);
  }
  return SUCCESS;
}

void TbeInfoAssembler::SetTbeInfoLimitedRange(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  const std::map<RangeLimitType, te::RangeLimitType>::const_iterator it =
          kRangeLimitTypeMap.find(op_kernel_info_ptr->GetRangeLimitType());
  if (it != kRangeLimitTypeMap.cend()) {
    op_info.SetRangeLimitType(it->second);
    FE_LOGD("[AssembleTbeInfo] Limited range has been set to[%d].", it->second);
  }
}

void TbeInfoAssembler::SetTbeInfoVectorCore(const ge::OpDescPtr op_desc_ptr,
                                            const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  if (Configuration::Instance(AI_CORE_NAME).IsEnableVirtualType()) {
    return;
  }
  if (ge::AttrUtils::HasAttr(op_desc_ptr, ge::ATTR_NAME_DISABLE_ATTACHED_RESOURCE)) {
    op_info.SetVectorCoreType(te::VectorCoreType::DISABLE);
    FE_LOGD("Node[%s] set disable vector.", op_desc_ptr->GetNamePtr());
    return;
  }
  const std::map<VectorCoreType, te::VectorCoreType>::const_iterator it =
          kVectorCoreTypeMap.find(op_kernel_info_ptr->GetEnableVectorCore());
  if (it != kVectorCoreTypeMap.cend()) {
    op_info.SetVectorCoreType(it->second);
    FE_LOGD("[AssembleTbeInfo] Vector core type has been set to[%d].", it->second);
  }
}

void TbeInfoAssembler::SetExtraParams(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const {
  std::string extra_params;
  (void)ge::AttrUtils::GetStr(op_desc, kAttrExtraParams, extra_params);
  auto param_builder = op_desc.GetExtAttr<std::function<std::string()>>("_extra_param_builder");
  if (extra_params.empty() && param_builder != nullptr) {
    extra_params = (*param_builder)();
  }
  if (!extra_params.empty()) {
    FE_LOGD("Node[%s] has extra_params[%s].", op_desc.GetName().c_str(), extra_params.c_str());
  }
  op_info.SetExtraParams(extra_params);
}

void TbeInfoAssembler::SetCustCoreNum(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const {
  std::string opAicoreNum;
  if (ge::AttrUtils::GetStr(op_desc, kAicCntKeyOp, opAicoreNum)) {
    op_info.SetCustAicNum(opAicoreNum);
    FE_LOGI("Node[%s] has _op_aicore_num[%s].", op_desc.GetName().c_str(), opAicoreNum.c_str());
  }

  std::string opVectorCoreNum;
  if (ge::AttrUtils::GetStr(op_desc, kAivCntKeyOp, opVectorCoreNum)) {
    op_info.SetCustAivNum(opVectorCoreNum);
    FE_LOGI("Node[%s] has _op_vectorcore_num[%s].", op_desc.GetName().c_str(), opVectorCoreNum.c_str());
  }
}

void TbeInfoAssembler::SetHashedExtraParams(const ge::OpDesc &op_desc, te::TbeOpInfo &op_info) const {
  std::string hashed_extra_params = "";
  hashed_extra_params = op_desc.TryGetExtAttr("_hashed_extra_param_builder", hashed_extra_params);
  if (!hashed_extra_params.empty()) {
    FE_LOGD("Node[%s] has hashed_extra_params[%s].", op_desc.GetName().c_str(), hashed_extra_params.c_str());
  } else {
    FE_LOGW("Node[%s] hashed_extra_params is null.", op_desc.GetName().c_str());
  }
  op_info.SetHashedExtraParams(hashed_extra_params);
}

void TbeInfoAssembler::SetOpImplMode(const std::string &engine_name, const ge::OpDescPtr &op_desc_ptr,
                                     te::TbeOpInfo &op_info) const {
  int64_t op_impl_mode_enum = 0;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, OP_IMPL_MODE_ENUM, op_impl_mode_enum)) {
    (void)ge::AttrUtils::GetInt(op_desc_ptr, OP_CUSTOM_IMPL_MODE_ENUM, op_impl_mode_enum);
    if (op_impl_mode_enum != 0) {
      FE_LOGW("Op[name=%s, type=%s] get _op_custom_impl_mode_enum 0x%llx from node attr.",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), op_impl_mode_enum);
    }
  }
  if (op_impl_mode_enum != 0) {
    auto iter = kOpImplIntToStr.find(op_impl_mode_enum);
    if (iter != kOpImplIntToStr.end()) {
      op_info.SetOpImplMode(iter->second);
      FE_LOGD("Op[name=%s, type=%s] set op_impl_mode %s by node attr.",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), iter->second.c_str());
      return;
    }
  }
  std::string op_impl_mode;
  if (!Configuration::Instance(engine_name)
      .GetOpImplMode(op_desc_ptr->GetName(), op_desc_ptr->GetType(), op_impl_mode)) {
    FE_LOGD("Op[name=%s, type=%s] can't get op_impl_mode from config.", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    return;
  }
  auto iter_impl_mode = kOpImplStrToInt.find(op_impl_mode);
  if (iter_impl_mode != kOpImplStrToInt.end()) {
    (void)ge::AttrUtils::SetInt(op_desc_ptr, OP_IMPL_MODE_ENUM, iter_impl_mode->second);
  }
  op_info.SetOpImplMode(op_impl_mode);
  FE_LOGD("Op[name=%s, type=%s] set op_impl_mode %s by config.",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), op_impl_mode.c_str());
  return;
}

void TbeInfoAssembler::SetSingleOpScene(const ge::Node *node, te::TbeOpInfo &op_info) const {
  ge::ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    FE_LOGW("Node[%s] can not get owner graph.", node->GetName().c_str());
    return;
  }
  bool is_single_op_scene = IsSingleOpGraphWithCache(*owner_graph);
  FE_LOGD("Node[%s] single op flag is %d.", node->GetName().c_str(), is_single_op_scene);
  op_info.SetSingleOpSceneFlag(is_single_op_scene);

  // graph fusion pass will add node to graph, get SingleOpScene from root graph
  const auto &src_graph = owner_graph->TryGetExtAttr(kPartSrcGraph, ge::ComputeGraphPtr());
  ge::ComputeGraphPtr root_graph = nullptr;
  if (src_graph == nullptr) {
    root_graph = ge::GraphUtils::FindRootGraph(owner_graph);
  } else {
    root_graph = ge::GraphUtils::FindRootGraph(src_graph);
  }
  if (root_graph == nullptr) {
    return;
  }
  bool is_origin_single_op_scene = false;
  ge::AttrUtils::GetBool(root_graph, ge::ATTR_SINGLE_OP_SCENE, is_origin_single_op_scene);
  FE_LOGD("Node[%s] graph single op flag is %d.", node->GetName().c_str(), is_origin_single_op_scene);
  op_info.SetOriSingleOpSceneFlag(is_origin_single_op_scene);
}

void TbeInfoAssembler::SetOpDynamicRank(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  const std::map<DynamicRankType, te::DynamicRankType>::const_iterator it =
          kDynamicRankTypeMap.find(op_kernel_info_ptr->GetDynamicRankType());
  if (it != kDynamicRankTypeMap.cend()) {
    op_info.SetDynamicRankType(it->second);
    FE_LOGD("[AssembleTbeInfo] Dynamic_rank_type has been set to[%d].", it->second);
  }
}

void TbeInfoAssembler::SetOpStorePattern(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  const std::map<OpPattern, std::string>::const_iterator iter =
      kOpPatternStrMap.find(op_kernel_info_ptr->GetOpPattern());
  if (iter == kOpPatternStrMap.end()) {
    FE_LOGW("[AssembleTbeInfo] Failed to set op pattern");
  } else {
    const std::string &pattern_str = iter->second;
    FE_LOGD("[AssembleTbeInfo] Succeed to set op pattern[%s].", pattern_str.c_str());
    op_info.SetOpStorePattern(pattern_str);
  }
}

void TbeInfoAssembler::SetOpImplSwitch(const ge::OpDescPtr &op_desc_ptr, te::TbeOpInfo &tbe_op_info) const {
  std::string op_impl_switch;
  if (ge::AttrUtils::GetStr(op_desc_ptr, kAttrOpImplSwitchValue, op_impl_switch)) {
    FE_LOGD("Attr op impl switch of op[%s, %s] is [%s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), op_impl_switch.c_str());
    tbe_op_info.SetOpImplSwitch(op_impl_switch);
  }
}

void TbeInfoAssembler::SetOpJitCompile(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const {
  const std::map<JitCompile, te::JitCompileType>::const_iterator iter =
          kJitCompileMap.find(op_kernel_info_ptr->GetJitCompileType());
  if (iter != kJitCompileMap.end()) {
    op_info.SetOpJitCompile(iter->second);
    FE_LOGD("[AssembleTbeInfo] Op jit compile has been set[%d].", iter->second);
  }
}

void TbeInfoAssembler::SetNeedPreCompile(const ge::OpDescPtr &op_desc_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                         te::TbeOpInfo &op_info) const
{
  if (op_kernel_info_ptr->GetCoreTypeVec().size() > 1 ||
      (op_kernel_info_ptr->GetCoreTypeVec().size() == 0 && IsCustomOp(*op_desc_ptr))) {
    op_info.SetNeedPreCompile(true);
    FE_LOGD("Set need pre compile for op[%s, %s].", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
  }
}

void TbeInfoAssembler::SetOpDebugConfig(const std::string &engine_name, const ge::OpDescPtr &op_desc_ptr,
                                        te::TbeOpInfo &op_info) const
{
  bool is_debug_compile = false;
  if ((ge::AttrUtils::GetBool(op_desc_ptr, kOpDebugCompile, is_debug_compile) && is_debug_compile) ||
      Configuration::Instance(engine_name).IsConfigDebugListOp(op_desc_ptr)) {
    op_info.SetOpDebugConfig(Configuration::Instance(engine_name).GetOpDebugConfig());
    FE_LOGD("Set op debug config[%s] for op[%s, %s].", Configuration::Instance(engine_name).GetOpDebugConfig().c_str(),
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
  }
}

Status TbeInfoAssembler::AssembleTbeInfo(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                         const std::string &engine_name, te::TbeOpInfo &tbe_op_info) {
  HeavyFormatInfo heavy_format_info;
  return AssembleTbeInfo(node, op_kernel_info_ptr, heavy_format_info, engine_name, tbe_op_info);
}

void TbeInfoAssembler::TransIOIrIndxToRealIndex(const std::vector<size_t> &output_real_idex,
                                                const std::vector<size_t> &input_real_idex,
                                                vector<vector<int64_t>> &real_output_inplace) const{
  for (auto &real_out : output_real_idex) {
    for (auto &real_input : input_real_idex) {
      real_output_inplace.push_back({static_cast<int64_t>(real_out), static_cast<int64_t>(real_input)});
    }
  }
}

bool TbeInfoAssembler::SetOutputRealIndexInplaceAttr(const vector<vector<int64_t>> &output_inplace,
                                                     std::map<size_t, std::pair<size_t, size_t>> &input_ir_real_index,
                                                     std::map<size_t, std::pair<size_t, size_t>> &output_ir_real_index,
                                                     vector<vector<int64_t>> &real_output_inplace) const{
  for (auto &inplace_pair : output_inplace) {
    size_t output_ir_indx = static_cast<size_t>(inplace_pair[0]);
    size_t input_ir_indx = static_cast<size_t>(inplace_pair[1]);

    // check ir index to real index
    auto output_it = output_ir_real_index.find(output_ir_indx);
    if (output_it == output_ir_real_index.end()) {
      FE_LOGW("The current out ir index[%u] is invalid.", output_ir_indx);
      return false;
    }

    auto input_it = input_ir_real_index.find(input_ir_indx);
    if (input_it == input_ir_real_index.end()) {
      FE_LOGW("The current input ir index[%u] is invalid.", input_ir_indx);
      return false;
    }

    // trans ir index to real index
    std::vector<size_t> output_real_idex;
    for (size_t i = 0; i < output_it->second.second; i++) {
      output_real_idex.emplace_back(output_it->second.first + i);
    }
    FE_LOGD("The output ir_index is[%lu], trans to real_index is [%s].",
            output_ir_indx, StringUtils::IntegerVecToString(output_real_idex).c_str());

    std::vector<size_t> input_real_idex;
    for (size_t i = 0; i < input_it->second.second; i++) {
      input_real_idex.emplace_back(input_it->second.first + i);
    }
    FE_LOGD("The input ir_index[%lu], trans to real_index is [%s].",
            input_ir_indx, StringUtils::IntegerVecToString(input_real_idex).c_str());
    TransIOIrIndxToRealIndex(output_real_idex, input_real_idex, real_output_inplace);
  }
  return true;
}

void TbeInfoAssembler::SetOutputInplaceAttr(ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr) const {
  vector<vector<int64_t>> output_inplace = op_kernel_info_ptr->GetOutputIplaceInfo();
  if (output_inplace.size() > 0) {
    std::map<size_t, std::pair<size_t, size_t>> input_ir_real_index;
    std::map<size_t, std::pair<size_t, size_t>> output_ir_real_index;
    vector<vector<int64_t>> real_output_inplace;
    GetIrIdexInstance(op_desc, input_ir_real_index, output_ir_real_index);
    if (SetOutputRealIndexInplaceAttr(output_inplace, input_ir_real_index, output_ir_real_index, real_output_inplace)) {
      ge::AttrUtils::SetListListInt(op_desc, kAttrOutputInplaceAbility, real_output_inplace);
      PrintOutputInplace(op_desc, real_output_inplace);
    }
  } else {
    FE_LOGD("Op[%s, %s] without outputInplaceAbility.", op_desc->GetTypePtr(), op_desc->GetNamePtr());
  }
}

void TbeInfoAssembler::SetInplaceAttr(const ge::OpDescPtr &op_desc, const IndexNameMap &input_map,
                                      const IndexNameMap &output_map) const {
  if (op_desc->HasAttr(kReusedParam) || op_desc->HasAttr(kRelationReusedParam)) {
    FE_LOGD("Node [%s, %s] already has the param_reused attr.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return;
  }
  std::shared_ptr<std::map<int32_t, int32_t>> relation_map_ptr = nullptr;
  FE_MAKE_SHARED((relation_map_ptr = std::make_shared<std::map<int32_t, int32_t>>()), return);
  for (const auto &out : output_map) {
    for (const auto &in : input_map) {
      FE_LOGD("Node [%s, %s] SetInplaceAttr out:%s-%u, input:%s-%u.", op_desc->GetName().c_str(),
              op_desc->GetType().c_str(), out.second.c_str(), out.first, in.second.c_str(), in.first);
      if (out.second == in.second) {
        relation_map_ptr->emplace(std::make_pair(out.first, in.first));
        break;
      }
    }
  }
  FE_LOGD("Node [%s, %s] set inplace attr success.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  (void)op_desc->SetExtAttr(kRelationReusedParam, relation_map_ptr);
}

Status TbeInfoAssembler::AssembleTbeInfo(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                         const HeavyFormatInfo &heavy_format_info, const std::string &engine_name,
                                         te::TbeOpInfo &tbe_op_info) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  ge::OpDesc &op_desc = *(op_desc_ptr.get());
  // 1. set the opFileName and the op_func_name
  string opFileName = op_kernel_info_ptr->GetOpInfo().opFileName;
  string opFuncName = op_kernel_info_ptr->GetOpInfo().opFuncName;
  tbe_op_info.SetOpFileName(opFileName);
  tbe_op_info.SetOpFuncName(opFuncName);
  FE_LOGD("Op[name=%s,type=%s]: tbe_op_info.opFileName=[%s], tbe_op_info.opFuncName=[%s].", op_desc.GetName().c_str(),
          op_desc.GetType().c_str(), opFileName.c_str(), opFuncName.c_str());

  SetOpDynamicRank(op_kernel_info_ptr, tbe_op_info);
  SetOpStorePattern(op_kernel_info_ptr, tbe_op_info);
  SetOpImplSwitch(op_desc_ptr, tbe_op_info);
  SetOpJitCompile(op_kernel_info_ptr, tbe_op_info);
  SetNeedPreCompile(op_desc_ptr, op_kernel_info_ptr, tbe_op_info);
  SetOpDebugConfig(engine_name, op_desc_ptr, tbe_op_info);

  IndexNameMap input_map;
  IndexNameMap output_map;
  if (GetInputOutputNameMap(op_desc, op_kernel_info_ptr, input_map, output_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s,type %s]: GetInputOutputNameMap failed.",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return FAILED;
  }

  // 2. feed all inputs to TbeOpInfo
  if (ConvertInputsToTbeOpInfo(node, input_map, op_kernel_info_ptr, heavy_format_info, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to assemble input info.",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return FAILED;
  }

  // 3. feed all outputs to TbeOpInfo
  if (ConvertOutputsToTbeOpInfo(node, output_map, op_kernel_info_ptr, heavy_format_info, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to assemble output info.",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return FAILED;
  }

  // 4. feed all attrs to TbeOpInfo
  if (FeedAttrsToTbeOpInfo(op_desc, op_kernel_info_ptr, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to feedAttrsToTbeOpInfo.",
                    op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return FAILED;
  }

  // feed all options to TbeOpInfo
  map<std::string, std::string> options = GetAllOptionsForTBE(op_desc, engine_name, op_kernel_info_ptr);
  tbe_op_info.SetOptions(options);
  SetOpImplMode(engine_name, op_desc_ptr, tbe_op_info);
  SetExtraParams(*op_desc_ptr, tbe_op_info);
  SetTbeInfoLimitedRange(op_kernel_info_ptr, tbe_op_info);
  SetTbeInfoVectorCore(op_desc_ptr, op_kernel_info_ptr, tbe_op_info);
  SetSingleOpScene(node.get(), tbe_op_info);

  if (IsFuzzBuildOp(op_desc)) {
    tbe_op_info.SetBuildType(te::FUZZILY_BUILD);
  }

  return SUCCESS;
}

bool TbeInfoAssembler::NeedCalibration(const string &engine_name,
                                       map<std::string, std::string> &options,
                                       string &default_core_num_str) {
  std::string soc_version = all_plat_info_.opti_compilation_info.GetSocVersion();
  if (engine_name == VECTOR_CORE_NAME) {
    all_plat_info_.platform_info.GetPlatformRes(kCfgSocInfo, kCfgVectorCoreCnt, default_core_num_str);
    return false;
  } else if (!PlatformUtils::Instance().IsDCSoc()) {
    return false;
  } else {
    const auto iter = options.find(ge::VIRTUAL_TYPE);
    if (iter == options.end() || iter->second != kNeedEstimationCoreNum) {
      return false;
    }

    auto original_core_num =
        static_cast<uint32_t>(BasicEstimator::GetUintParam(all_plat_info_.platform_info, kCfgSocInfo, kCfgAiCoreCnt));
    if (time_to_core_num_.find(original_core_num) == time_to_core_num_.end()) {
      FE_LOGD("Original core num %u does not need to be amplified.", original_core_num);
      return false;
    }
  }
  return true;
}

void TbeInfoAssembler::FindAmplifiedCoreNum(uint64_t exec_time,
                                            string &final_core_num_str) {
  auto original_core_num =
      static_cast<uint32_t>(BasicEstimator::GetUintParam(all_plat_info_.platform_info, kCfgSocInfo, kCfgAiCoreCnt));

  const std::map<uint32_t, std::vector<std::pair<uint64_t, string>>>::const_iterator iter =
      time_to_core_num_.find(original_core_num);
  if (iter != time_to_core_num_.end()) {
    for (const auto &time_to_core_num : iter->second) {
      if (exec_time < time_to_core_num.first) {
        final_core_num_str = time_to_core_num.second;
        return;
      }
    }
  }
}

void TbeInfoAssembler::CalibrateCoreNum(const ge::OpDesc &op_desc, const string &engine_name,
                                        const OpKernelInfoPtr &op_kernel_info_ptr,
                                        map<std::string, std::string> &options) {
  if (options[ge::AICORE_NUM].empty()) {
    uint32_t split_core_num = all_plat_info_.opti_compilation_info.GetAICoreNum();
    uint32_t original_core_num =
        static_cast<uint32_t>(BasicEstimator::GetUintParam(all_plat_info_.platform_info, kCfgSocInfo, kCfgAiCoreCnt));
    if (split_core_num != 0 && split_core_num < original_core_num) {
      options[ge::AICORE_NUM] = std::to_string(split_core_num);
    }
  }
  string core_num_str;
  (void)ge::GetContext().GetOption(ge::AICORE_NUM, core_num_str);
  if (core_num_str.empty() && options.find("ai_core_cnt") != options.end()) {
    options[ge::AICORE_NUM] = options["ai_core_cnt"];
    return;
  }
  bool is_soft_sync_op = false;
  bool dynamic_compile_static = false;
  if (op_kernel_info_ptr != nullptr) {
    is_soft_sync_op = op_kernel_info_ptr->IsSoftSyncOp();
    dynamic_compile_static = op_kernel_info_ptr->IsDynamicCompileStatic();
  }
  if (Configuration::Instance(engine_name).IsEnableVirtualType() && is_soft_sync_op && core_num_str.empty()) {
    if (dynamic_compile_static) {
      std::string origin_core_num;
      (void)all_plat_info_.platform_info.GetPlatformRes(kCfgSocInfo, kCfgAiCoreCnt, origin_core_num);
      options[ge::AICORE_NUM] = origin_core_num;
    } else {
      options[ge::AICORE_NUM] = "1";
    }
    return;
  }
  // if have no core_num, not do super kernel
  if (options[ge::AICORE_NUM].empty()) {
    return;
  }
  string default_core_num_str;
  bool need_calibration = NeedCalibration(engine_name, options, default_core_num_str);
  string final_core_num_str(default_core_num_str);
  if (need_calibration) {
    uint64_t exec_time = 0;
    Status ret = ExecutionTimeEstimator::GetExecTime(all_plat_info_.platform_info, op_desc,
                                                     op_kernel_info_ptr, exec_time);
    if (ret == SUCCESS) {
      FE_LOGD("The estimated execution time of op %s is %lu.", op_desc.GetName().c_str(), exec_time);
      FindAmplifiedCoreNum(exec_time, final_core_num_str);
    }
    FE_LOGD("Amplified core number string is %s.", final_core_num_str.c_str());
  } else {
    FE_LOGD("Default core number string is %s.", default_core_num_str.c_str());
  }

  if (!final_core_num_str.empty()) {
    options[ge::AICORE_NUM] = final_core_num_str;
  }
}

map<std::string, std::string> TbeInfoAssembler::GetAllOptionsForTBE(const ge::OpDesc &op_desc,
                                                                    const string &engine_name,
                                                                    const OpKernelInfoPtr &op_kernel_info_ptr) {
  map<std::string, std::string> options;
  if (Configuration::Instance(AI_CORE_NAME).IsEnableSuperkernelPlus()) {
    options.insert(std::pair<string, string>(kEnableSuperkernelPlus, kStrTrue));
    FE_LOGD("Set enable superkernel plus for op[%s].", op_desc.GetName().c_str());
  }

  CalibrateCoreNum(op_desc, engine_name, op_kernel_info_ptr, options);
  return options;
}

Status TbeInfoAssembler::AssembleTbeInfo(ge::Node *node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                         te::TbeOpInfo &tbe_op_info, const string &engine_name) {
  // set op_info
  tbe_op_info.SetOpFileName(op_kernel_info_ptr->GetOpInfo().opFileName);
  tbe_op_info.SetOpFuncName(op_kernel_info_ptr->GetOpInfo().opFuncName);

  IndexNameMap input_map;
  IndexNameMap output_map;
  auto op = node->GetOpDesc();
  if (fe::Configuration::Instance(AI_CORE_NAME).IsConfigDebugListOp(op) &&
      !ge::AttrUtils::HasAttr(op, kOpDebugCompile)) {
      FE_LOGD("op[%s, %s] set op_debug_compile in assemble tbeinfo",
              op->GetName().c_str(), op->GetType().c_str());
      ge::AttrUtils::SetBool(op, kOpDebugCompile, true);
  }

  if (GetInputOutputNameMap(*(op.get()), op_kernel_info_ptr, input_map, output_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to getInputOutputNameMap.",
                    op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // feed all inputs to TbeOpInfo
  if (FeedInputsToTbeOpInfo(node, input_map, op_kernel_info_ptr, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to feedInputsToTbeOpInfo.",
                    op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // feed all outputs to TbeOpInfo
  if (FeedOutputsToTbeOpInfo(node, output_map, op_kernel_info_ptr, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to feedOutputsToTbeOpInfo.",
                    op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // feed all outputs to TbeOpInfo
  if (FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput][Op %s, type %s]: failed to feedAttrsToTbeOpInfo.",
                    op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  if (FeedFlagInt64ToTbeOpInfo(node, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][PreCompileOp][AssembleInput] Node[%s, %s] AssembleTbeInfo FeedFlagInt64ToTbeOpInfo failed.",
        op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // feed all IsUnknownShape to TbeOpInfo
  if (FeedIsUnknownShapeToTbeOpInfo(*(op.get()), tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][PreCompileOp][AssembleInput] Node[%s, %s]: failed to feedIsUnknownShapeToTbeOpInfo.",
        op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // feed all options to TbeOpInfo
  const map<std::string, std::string> &options = GetAllOptionsForTBE(*op, engine_name, op_kernel_info_ptr);
  ge::AttrUtils::SetStr(op, kAttrEngineType, engine_name);
  FE_LOGD("Set Node %s's engine_name as %s.", op->GetName().c_str(), engine_name.c_str());
  tbe_op_info.SetOptions(options);
  SetOpImplMode(engine_name, op, tbe_op_info);
  SetExtraParams(*op, tbe_op_info);
  SetTbeInfoLimitedRange(op_kernel_info_ptr, tbe_op_info);
  SetTbeInfoVectorCore(op, op_kernel_info_ptr, tbe_op_info);
  SetSingleOpScene(node, tbe_op_info);
  SetCustCoreNum(*op, tbe_op_info);
  SetOpDynamicRank(op_kernel_info_ptr, tbe_op_info);
  SetOpStorePattern(op_kernel_info_ptr, tbe_op_info);
  SetOpImplSwitch(op, tbe_op_info);
  SetOpJitCompile(op_kernel_info_ptr, tbe_op_info);
  SetNeedPreCompile(op, op_kernel_info_ptr, tbe_op_info);
  SetOpDebugConfig(engine_name, op, tbe_op_info);
  DumpOpInfo(tbe_op_info);
  uint32_t ub_fusion_space_size = 0;
  ub_fusion_space_size = op->TryGetExtAttr(ATTR_NAME_UB_FUSION_SPACE_SIZE, ub_fusion_space_size);
  if (ub_fusion_space_size != 0) {
    tbe_op_info.SetUBSpaceSize(static_cast<uint64_t>(ub_fusion_space_size));
    FE_LOGD("Set Node %s's UBSpaceSize %u", op->GetName().c_str(), ub_fusion_space_size);
  }

  ToOpStructPtr l1_info;
  FE_MAKE_SHARED(l1_info = std::make_shared<ToOpStruct_t>(), return FAILED);
  ToOpStructPtr l2_info;
  FE_MAKE_SHARED(l2_info = std::make_shared<ToOpStruct_t>(), return FAILED);

  GetL1ToOpStructFromJson(op, l1_info);
  GetL2ToOpStructFromJson(op, l2_info);

  if (l1_info != nullptr) {
    FE_LOGD("Get attr l1 userinfo, op name = %s, op type = %s, op_l1_space = %ld.", op->GetName().c_str(),
            op->GetType().c_str(), l1_info->op_l1_space);
    if (!l1_info->op_l1_fusion_type.empty()) {
      FE_LOGD("OpL1fusionType = %ld.", l1_info->op_l1_fusion_type[0]);
    }
    tbe_op_info.SetL1Space(l1_info->op_l1_space);
    DumpL1Attr(node);
    DumpOpInfo(tbe_op_info);
  } else {
    FE_LOGD("L1Info is null_ptr, op name = %s, op type = %s.", op->GetName().c_str(), op->GetType().c_str());
  }
  if (l2_info != nullptr) {
    DumpL2Attr(node);
    DumpOpInfo(tbe_op_info);
  }
  SetInplaceAttr(op, input_map, output_map);
  SetOutputInplaceAttr(op, op_kernel_info_ptr);
  return SUCCESS;
}

Status TbeInfoAssembler::AssembleAutoFuseTbeInfo(ge::Node *node, te::TbeOpInfo &tbe_op_info) const {
  // set op_info
  tbe_op_info.SetOpFileName("asc_codegen_compile");
  tbe_op_info.SetOpFuncName("asc_codegen_compile");
  auto op = node->GetOpDesc();
  if (fe::Configuration::Instance(AI_CORE_NAME).IsConfigDebugListOp(op) &&
      !ge::AttrUtils::HasAttr(op, kOpDebugCompile)) {
      FE_LOGD("Op[%s, %s] set op_debug_compile in assemble tbeinfo.",
              op->GetName().c_str(), op->GetType().c_str());
      ge::AttrUtils::SetBool(op, kOpDebugCompile, true);
  }

  // input info
  if (FeedAutoFuseInoutputsToTbeOpInfo(node, true, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleAutofuseInput][Op %s, type %s]: failed to feedInputs.",
                    op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }
  // output info
  if (FeedAutoFuseInoutputsToTbeOpInfo(node, false, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleAutofuseInput][Op %s, type %s]: failed to feedInputs.",
                    op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  if (FeedFlagInt64ToTbeOpInfo(node, tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][PreCompileOp][AssembleInput] Node[%s, %s] AssembleTbeInfo FeedFlagInt64 failed.",
        op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // feed all IsUnknownShape to TbeOpInfo
  if (FeedIsUnknownShapeToTbeOpInfo(*(op.get()), tbe_op_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][AssembleInput] Node[%s, %s]: failed to feedIsUnknown.",
        op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  SetExtraParams(*op, tbe_op_info);
  SetHashedExtraParams(*op, tbe_op_info);
  SetSingleOpScene(node, tbe_op_info);
  SetCustCoreNum(*op, tbe_op_info);
  SetOpImplSwitch(op, tbe_op_info);
  return SUCCESS;
}

Status TbeInfoAssembler::GetSpecificIndex(const ge::OpDesc &op_desc, const IndexNameMap &name_map,
                                          const std::string &input_name_in_op_kernel, bool is_input,
                                          vector<uint32_t> &specific_input_index) const {
  for (const auto &inOrOutDesc : name_map) {
    if (inOrOutDesc.second == input_name_in_op_kernel) {
      if (is_input && inOrOutDesc.first < op_desc.GetAllInputsSize() &&
          op_desc.MutableInputDesc(inOrOutDesc.first) != nullptr) {
        specific_input_index.push_back(inOrOutDesc.first);
      }
      if (!is_input && inOrOutDesc.first < op_desc.GetAllOutputsDescSize() &&
          op_desc.MutableOutputDesc(inOrOutDesc.first) != nullptr) {
        specific_input_index.push_back(inOrOutDesc.first);
      }
    }
  }
  return SUCCESS;
}

template <typename T>
void TbeInfoAssembler::GetConstValueVec(ge::GeTensorPtr &const_tensor_ptr, vector<T> &const_data_vec) {
  const T *const_data_ptr = reinterpret_cast<const T *>(const_tensor_ptr->GetData().GetData());
  if (const_data_ptr == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][GetConstVal] const data ptr is null.");
    return;
  }
  size_t size = const_tensor_ptr->GetData().GetSize() / sizeof(T);
  for (size_t i = 0; i < size; ++i) {
    T const_data = *(const_data_ptr + i);
    const_data_vec.push_back(const_data);
  }
}

Status TbeInfoAssembler::SetConstValueWithFloat16(ge::GeTensorPtr tensor_ptr, const std::string &tensor_name,
                                                  te::TbeOpTensor &op_tensor) {
  const uint16_t *const_data_ptr = reinterpret_cast<const uint16_t *>(tensor_ptr->GetData().GetData());
  FE_CHECK_NOTNULL(const_data_ptr);
  size_t size = tensor_ptr->GetData().GetSize() / sizeof(uint16_t);
  vector<float> const_data_vec;
  for (size_t i = 0; i < size; ++i) {
    uint16_t const_data = *(const_data_ptr + i);
    float const_data_fp32 = Uint16ToFloat(const_data);
    const_data_vec.push_back(const_data_fp32);
    FE_LOGD("Float16 value of const data[%zu] is %f.", i, const_data_fp32);
  }
  te::TbeAttrValue const_value(tensor_name, const_data_vec);
  op_tensor.SetConstValue(const_value);
  return SUCCESS;
}

Status TbeInfoAssembler::SetConstValueWithBf16(ge::GeTensorPtr tensor_ptr, const std::string &tensor_name,
                                               te::TbeOpTensor &op_tensor) {
  const uint16_t *const_data_ptr = reinterpret_cast<const uint16_t *>(tensor_ptr->GetData().GetData());
  FE_CHECK_NOTNULL(const_data_ptr);
  size_t size = tensor_ptr->GetData().GetSize() / sizeof(uint16_t);
  vector<float> const_data_vec;
  for (size_t i = 0; i < size; ++i) {
    uint16_t const_data = *(const_data_ptr + i);
    float const_data_fp32 = Bf16ToFloat(const_data);
    const_data_vec.push_back(const_data_fp32);
    FE_LOGD("Bf16 value of const data[%zu] is %f.", i, const_data_fp32);
  }
  te::TbeAttrValue const_value(tensor_name, const_data_vec);
  op_tensor.SetConstValue(const_value);
  return SUCCESS;
}

template <typename T>
Status TbeInfoAssembler::SetConstValue(ge::GeTensorPtr tensor_ptr, const std::string &tensor_name,
                                       te::TbeOpTensor &op_tensor) {
  vector<T> const_data_vec;
  GetConstValueVec(tensor_ptr, const_data_vec);
  te::TbeAttrValue const_value(tensor_name, const_data_vec);
  op_tensor.SetConstValue(const_value);
  return SUCCESS;
}
}  // namespace fe
