/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/aipp_utils.h"

#include <sstream>
#include "framework/common/debug/log.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/checker.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr size_t kInvalidIdx = 0xFFFFFFFFUL;
const std::string kDataModeStatic = "static_aipp";
const std::string kDataModeDynamic = "dynamic_aipp";
const std::string kDataModeDynamicConf = "dynamic_aipp_conf";
constexpr int32_t kDecimalRadix = 10;
constexpr uint32_t kDataIndex = 0U;
}

void AippUtils::SetMatrixInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info) {
  if (aipp_params.matrix_r0c0_size() > 0) {
    aipp_info.matrix_r0c0 = aipp_params.matrix_r0c0(0);
  }
  if (aipp_params.matrix_r0c1_size() > 0) {
    aipp_info.matrix_r0c1 = aipp_params.matrix_r0c1(0);
  }
  if (aipp_params.matrix_r0c2_size() > 0) {
    aipp_info.matrix_r0c2 = aipp_params.matrix_r0c2(0);
  }
  if (aipp_params.matrix_r1c0_size() > 0) {
    aipp_info.matrix_r1c0 = aipp_params.matrix_r1c0(0);
  }
  if (aipp_params.matrix_r1c1_size() > 0) {
    aipp_info.matrix_r1c1 = aipp_params.matrix_r1c1(0);
  }
  if (aipp_params.matrix_r1c2_size() > 0) {
    aipp_info.matrix_r1c2 = aipp_params.matrix_r1c2(0);
  }
  if (aipp_params.matrix_r2c0_size() > 0) {
    aipp_info.matrix_r2c0 = aipp_params.matrix_r2c0(0);
  }
  if (aipp_params.matrix_r2c1_size() > 0) {
    aipp_info.matrix_r2c1 = aipp_params.matrix_r2c1(0);
  }
  if (aipp_params.matrix_r2c2_size() > 0) {
    aipp_info.matrix_r2c2 = aipp_params.matrix_r2c2(0);
  }
}

void AippUtils::SetBiasInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info) {
  if (aipp_params.output_bias_0_size() > 0) {
    aipp_info.output_bias_0 = aipp_params.output_bias_0(0);
  }
  if (aipp_params.output_bias_1_size() > 0) {
    aipp_info.output_bias_1 = aipp_params.output_bias_1(0);
  }
  if (aipp_params.output_bias_2_size() > 0) {
    aipp_info.output_bias_2 = aipp_params.output_bias_2(0);
  }
  if (aipp_params.input_bias_0_size() > 0) {
    aipp_info.input_bias_0 = aipp_params.input_bias_0(0);
  }
  if (aipp_params.input_bias_1_size() > 0) {
    aipp_info.input_bias_1 = aipp_params.input_bias_1(0);
  }
  if (aipp_params.input_bias_2_size() > 0) {
    aipp_info.input_bias_2 = aipp_params.input_bias_2(0);
  }
}

void AippUtils::SetChnInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info) {
  aipp_info.mean_chn_0 = aipp_params.mean_chn_0();
  aipp_info.mean_chn_1 = aipp_params.mean_chn_1();
  aipp_info.mean_chn_2 = aipp_params.mean_chn_2();
  aipp_info.mean_chn_3 = aipp_params.mean_chn_3();
  aipp_info.min_chn_0 = aipp_params.min_chn_0();
  aipp_info.min_chn_1 = aipp_params.min_chn_1();
  aipp_info.min_chn_2 = aipp_params.min_chn_2();
  aipp_info.min_chn_3 = aipp_params.min_chn_3();
  if (aipp_params.var_reci_chn_0_size() > 0) {
    aipp_info.var_reci_chn_0 = aipp_params.var_reci_chn_0(0);
  }
  if (aipp_params.var_reci_chn_1_size() > 0) {
    aipp_info.var_reci_chn_1 = aipp_params.var_reci_chn_1(0);
  }
  if (aipp_params.var_reci_chn_2_size() > 0) {
    aipp_info.var_reci_chn_2 = aipp_params.var_reci_chn_2(0);
  }
  if (aipp_params.var_reci_chn_3_size() > 0) {
    aipp_info.var_reci_chn_3 = aipp_params.var_reci_chn_3(0);
  }
}

Status AippUtils::ConvertAippParams2AippInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info) {
  aipp_info.aipp_mode = static_cast<int8_t>(aipp_params.aipp_mode());
  aipp_info.input_format = static_cast<int8_t>(aipp_params.input_format());
  aipp_info.related_input_rank = aipp_params.related_input_rank();
  aipp_info.src_image_size_w = aipp_params.src_image_size_w();
  aipp_info.src_image_size_h = aipp_params.src_image_size_h();
  aipp_info.crop = static_cast<int8_t>(aipp_params.crop());
  aipp_info.load_start_pos_w = aipp_params.load_start_pos_w();
  aipp_info.load_start_pos_h = aipp_params.load_start_pos_h();
  aipp_info.crop_size_w = aipp_params.crop_size_w();
  aipp_info.crop_size_h = aipp_params.crop_size_h();
  aipp_info.resize = static_cast<int8_t>(aipp_params.resize());
  aipp_info.resize_output_w = aipp_params.resize_output_w();
  aipp_info.resize_output_h = aipp_params.resize_output_h();
  aipp_info.padding = static_cast<int8_t>(aipp_params.padding());
  aipp_info.left_padding_size = aipp_params.left_padding_size();
  aipp_info.right_padding_size = aipp_params.right_padding_size();
  aipp_info.top_padding_size = aipp_params.top_padding_size();
  aipp_info.bottom_padding_size = aipp_params.bottom_padding_size();
  aipp_info.csc_switch = static_cast<int8_t>(aipp_params.csc_switch());
  aipp_info.rbuv_swap_switch = static_cast<int8_t>(aipp_params.rbuv_swap_switch());
  aipp_info.ax_swap_switch = static_cast<int8_t>(aipp_params.ax_swap_switch());
  aipp_info.single_line_mode = static_cast<int8_t>(aipp_params.single_line_mode());
  aipp_info.support_rotation = static_cast<int8_t>(aipp_params.support_rotation());
  aipp_info.max_src_image_size = aipp_params.max_src_image_size();

  SetMatrixInfo(aipp_params, aipp_info);
  SetBiasInfo(aipp_params, aipp_info);
  SetChnInfo(aipp_params, aipp_info);

  return SUCCESS;
}

Status AippUtils::SetAippInfoAndTypeFromOpDesc(const std::map<std::string, uint32_t> &data_index_map,
                                               const OpDescPtr &op_desc, const uint32_t index,
                                               std::map<uint32_t, AippConfigInfo> &aipp_infos,
                                               std::map<uint32_t, std::pair<InputAippType, size_t>> &aipp_types) {
  GE_CHECK_NOTNULL(op_desc);
  NamedAttrs aipp_attr;
  const bool get_attr_aipp = AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr);
  const std::string *data_mode_ptr = AttrUtils::GetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE);
  const bool get_attr_mode = (data_mode_ptr != nullptr);
  std::string data_mode = "";
  if (get_attr_mode) {
    data_mode = *data_mode_ptr;
  }
  if ((!get_attr_aipp) && (!get_attr_mode)) {
    GELOGD("There is not AIPP related with op:%s, index:%u.", op_desc->GetName().c_str(), index);
    return SUCCESS;
  }
  if ((!get_attr_aipp) || (!get_attr_mode)) {
    std::stringstream error_message;
    error_message << "Both ATTR_NAME_AIPP and ATTR_DATA_RELATED_AIPP_MODE attributes are needed on the data node, "
                     "but only " << (get_attr_aipp ? "ATTR_NAME_AIPP" : "ATTR_DATA_RELATED_AIPP_MODE")
                  << " is obtained at the time.";
    REPORT_INNER_ERR_MSG("E19999", "Op:%s, index:%u, error message:%s", op_desc->GetName().c_str(), index,
                       error_message.str().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attrs]Op:%s, index:%u, error message:%s", op_desc->GetName().c_str(), index,
           error_message.str().c_str());
    return INTERNAL_ERROR;
  }

  auto ret = SetAippInfoImpl(aipp_attr, op_desc, index, aipp_infos);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][AippInfo]Failed to set aipp info, op:%s, index:%u.", op_desc->GetName().c_str(), index);
    return ret;
  }
  ret = SetAippTypeImpl(data_index_map, data_mode, op_desc, index, aipp_types);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][AippType]Failed to set aipp type, op:%s, index:%u.", op_desc->GetName().c_str(), index);
    return ret;
  }
  return SUCCESS;
}

Status AippUtils::SetAippInfoImpl(const NamedAttrs &aipp_attr, const OpDescPtr &op_desc, const uint32_t index,
                                  std::map<uint32_t, AippConfigInfo> &aipp_infos) {
  domi::AippOpParams aipp_params;
  Status ret = OpUtils::ConvertAippParams(aipp_attr, aipp_params);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Convert][AippParams] Failed to convert aipp params, op:%s.", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GELOGI("Add aipp info for node:%s, type:%s, current index:%u, current node related input rank:%u",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), index, aipp_params.related_input_rank());

  AippConfigInfo aipp_info;
  ret = AippUtils::ConvertAippParams2AippInfo(aipp_params, aipp_info);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Convert][AippInfo]Failed to convert params to info, op:%s.", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  aipp_infos[index] = aipp_info;
  return SUCCESS;
}

Status AippUtils::SetAippTypeImpl(const std::map<std::string, uint32_t> &data_index_map,
                                  const std::string &data_mode, const OpDescPtr &op_desc, const uint32_t index,
                                  std::map<uint32_t, std::pair<InputAippType, size_t>> &aipp_types) {
  InputAippType aipp_type = DATA_WITHOUT_AIPP;
  if (data_mode == kDataModeStatic) {
    aipp_type = DATA_WITH_STATIC_AIPP;
  } else if (data_mode == kDataModeDynamic) {
    aipp_type = DATA_WITH_DYNAMIC_AIPP;
  } else if (data_mode == kDataModeDynamicConf) {
    aipp_type = DYNAMIC_AIPP_NODE;
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Get invalid mode:%s, op:%s, type:%s.", data_mode.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr]Get invalid mode:%s, op:%s, type:%s.", data_mode.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  size_t aipp_data_index = kInvalidIdx;
  if (aipp_type == DATA_WITH_DYNAMIC_AIPP) {
    const std::string *releated_name = AttrUtils::GetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP);
    if (releated_name == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Failed to get attr:%s, op:%s, type:%s.", ATTR_DATA_AIPP_DATA_NAME_MAP.c_str(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][Attr]Failed to get attr:%s, op:%s, type:%s.", ATTR_DATA_AIPP_DATA_NAME_MAP.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return INTERNAL_ERROR;
    }
    const auto iter = data_index_map.find(*releated_name);
    if (iter != data_index_map.end()) {
      aipp_data_index = iter->second;
      GELOGI("Find AippData:%s of index:%zu for op:%s, index:%u", releated_name->c_str(), aipp_data_index,
             op_desc->GetName().c_str(), index);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Can not find AippData node for index:%u, op:%s", index, op_desc->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Find][AippData]Can not find AippData node for index:%u, op:%s",
             index, op_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  aipp_types[index] = {aipp_type, aipp_data_index};
  return SUCCESS;
}

Status AippUtils::GetAippInfo(const std::map<uint32_t, AippConfigInfo> &aipp_infos, const uint32_t index,
                              AippConfigInfo &aipp_info) {
  const auto it = aipp_infos.find(index);
  if (it == aipp_infos.end()) {
    GELOGD("There is not AIPP related info with index:%u", index);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }
  aipp_info = it->second;
  return SUCCESS;
}

Status AippUtils::GetAippType(const std::map<uint32_t, std::pair<InputAippType, size_t>> &aipp_types,
                              const uint32_t index, InputAippType &aipp_type, size_t &aipp_data_index) {
  const auto it = aipp_types.find(index);
  if (it == aipp_types.end()) {
    GELOGD("There is not AIPP releated type with index:%u, return default value.", index);
    aipp_type = DATA_WITHOUT_AIPP;
    aipp_data_index = kInvalidIdx;
    return SUCCESS;
  }
  aipp_type = it->second.first;
  aipp_data_index = it->second.second;
  return SUCCESS;
}

Status AippUtils::ParseAIPPInfo(const std::string &in_out_info, InputOutputDims &dims_info) {
  GELOGI("ParseAIPPInfo: origin str: %s", in_out_info.c_str());
  const std::vector<std::string> infos = StringUtils::Split(in_out_info, ':');
  if (infos.size() != kAippInfoNum) {
    REPORT_INNER_ERR_MSG("E19999", "in_out_info:%s size:%zu != kAippInfoNum:%u check invalid",
        in_out_info.c_str(), infos.size(), kAippInfoNum);
    GELOGE(ACL_ERROR_GE_AIPP_MODE_INVALID, "[Check][Param] in_out_info:%s size:%zu != kAippInfoNum:%u",
        in_out_info.c_str(), infos.size(), kAippInfoNum);
    return ACL_ERROR_GE_AIPP_MODE_INVALID;
  }
  dims_info.name = infos[kAippInfoTensorName];
  dims_info.size = static_cast<uint32_t>(std::strtol(infos[kAippInfoTensorSize].c_str(), nullptr, kDecimalRadix));
  dims_info.dim_num = static_cast<size_t>(std::strtol(infos[kAippInfoDimNum].c_str(), nullptr, kDecimalRadix));

  const std::vector<std::string> dims = StringUtils::Split(infos[kAippInfoShape], ',');
  for (const auto &dim : dims) {
    if (dim.empty()) {
      continue;
    }
    dims_info.dims.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimalRadix));
  }
  return SUCCESS;
}

void AippUtils::SetOrigInputInfo(const std::string &input, const uint32_t index,
    std::map<uint32_t, OriginInputInfo> &orig_input_info_map) {
  GELOGI("origin input str: %s.", input.c_str());
  const std::vector<std::string> infos = StringUtils::Split(input, ':');
  OriginInputInfo input_info{};
  input_info.format = TypeUtils::SerialStringToFormat(infos[kAippInfoFormat]);
  input_info.data_type = TypeUtils::SerialStringToDataType(infos[kAippInfoDataType]);
  input_info.dim_num = static_cast<uint32_t>(std::strtol(infos[kAippInfoDimNum].c_str(), nullptr, kDecimalRadix));
  orig_input_info_map[index] = input_info;
  return;
}

Status AippUtils::SetAippInputOutputInfoFromOpDesc(const OpDescPtr &op_desc, const uint32_t index,
      std::map<uint32_t, OriginInputInfo> &orig_input_info_map,
      std::map<uint32_t, std::pair<std::vector<InputOutputDims>, std::vector<InputOutputDims>>> &aipp_dims_info) {
  if ((!op_desc->HasAttr(ATTR_NAME_AIPP_INPUTS)) || (!op_desc->HasAttr(ATTR_NAME_AIPP_OUTPUTS))) {
    GELOGI("there is not AIPP related with index %u, node: %s.", index, op_desc->GetName().c_str());
    return SUCCESS;
  }

  std::vector<std::string> inputs;
  std::vector<InputOutputDims> input_dims;
  orig_input_info_map[index] = { FORMAT_RESERVED, DT_UNDEFINED, 0U };
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs) && (!inputs.empty())) {
    GELOGI("Data: %s has %zu related aippInfo.", op_desc->GetName().c_str(), inputs.size());
    for (const auto &it : inputs) {
      InputOutputDims input_info;
      GE_CHK_STATUS_RET_NOLOG(ParseAIPPInfo(it, input_info));
      input_dims.emplace_back(input_info);
      GELOGD("Aipp origin input dims info: %s", it.c_str());

      const auto data_input_desc = op_desc->GetInputDescPtr(kDataIndex);
      GE_CHECK_NOTNULL(data_input_desc);
      int64_t data_input_size = 0;
      (void)TensorUtils::GetSize(*(data_input_desc), data_input_size);
      GELOGD("Related Data[%d]: tensor_name: %s, dim_num: %zu, tensor_size: %zu, format: %s, data_type: %s, shape: %s.",
          index, op_desc->GetName().c_str(), data_input_desc->GetShape().GetDimNum(), data_input_size,
          TypeUtils::FormatToSerialString(data_input_desc->GetFormat()).c_str(),
          TypeUtils::DataTypeToSerialString(data_input_desc->GetDataType()).c_str(),
          ToString(data_input_desc->GetShape().GetDims()).c_str());
    }
    SetOrigInputInfo(inputs[kAippOriginInputIndex], index, orig_input_info_map);
  }

  std::vector<std::string> outputs;
  std::vector<InputOutputDims> output_dims;
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs) && (!outputs.empty())) {
    for (const auto &it : outputs) {
      InputOutputDims output_info;
      GE_CHK_STATUS_RET_NOLOG(ParseAIPPInfo(it, output_info));
      output_dims.emplace_back(output_info);
      GELOGD("Aipp output dims info: %s", it.c_str());
    }
  }

  aipp_dims_info[index] = { input_dims, output_dims };
  return SUCCESS;
}

Status AippUtils::GetOrigInputInfo(const std::map<uint32_t, OriginInputInfo> &orig_input_info_map, const uint32_t index,
    OriginInputInfo &orig_input_info) {
  const auto it = orig_input_info_map.find(index);
  if (it == orig_input_info_map.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Get index:%u from orig_input_info_map fail", index);
    GELOGE(ACL_ERROR_GE_AIPP_NOT_EXIST, "[Check][Param] Get index:%u from orig_input_info_map fail", index);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }

  const OriginInputInfo &input_info = it->second;
  if ((input_info.format != FORMAT_RESERVED) || (input_info.data_type != DT_UNDEFINED)) {
    orig_input_info = input_info;
  }

  return SUCCESS;
}

Status AippUtils::GetAllAippInputOutputDims(
      const std::map<uint32_t,
      std::pair<std::vector<InputOutputDims>, std::vector<InputOutputDims>>> &aipp_dims_info,
      const uint32_t index, std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims) {
  const auto it = aipp_dims_info.find(index);
  if (it == aipp_dims_info.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Get index:%u from aipp_dims_info fail", index);
    GELOGE(ACL_ERROR_GE_AIPP_NOT_EXIST, "[Check][Param] Get index:%u from aipp_dims_info fail", index);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }

  input_dims = it->second.first;
  output_dims = it->second.second;
  return SUCCESS;
}
}  // namespace ge
