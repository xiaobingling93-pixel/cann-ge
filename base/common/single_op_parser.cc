/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op_parser.h"

#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <regex>

#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_factory_impl.h"
#include "formats/utils/formats_trans_utils.h"
#include "base/err_msg.h"

using Json = nlohmann::json;

namespace ge {
namespace {
constexpr char_t const *kKeyOp = "op";
constexpr char_t const *kKeyInputDesc = "input_desc";
constexpr char_t const *kKeyOutputDesc = "output_desc";
constexpr char_t const *kKeyAttr = "attr";
constexpr char_t const *kKeyName = "name";
constexpr char_t const *kKeyType = "type";
constexpr char_t const *kKeyShape = "shape";
constexpr char_t const *kKeyOriginShape = "origin_shape";
constexpr char_t const *kKeyShapeRange = "shape_range";
constexpr char_t const *kKeyValue = "value";
constexpr char_t const *kKeyFormat = "format";
constexpr char_t const *kKeyOriginFormat = "origin_format";
constexpr char_t const *kKeyIsConst = "is_const";
constexpr char_t const *kKeyConstValue = "const_value";
constexpr char_t const *kFileSuffix = ".om";
constexpr char_t const *kKeyDynamicInput = "dynamic_input";
constexpr char_t const *kKeyCompileFlag = "compile_flag";
constexpr int32_t kDumpJsonIndent = 2;
constexpr uint32_t kShapeRangePairSize = 2U;
constexpr uint32_t kShapeRangeLow = 0U;
constexpr uint32_t kShapeRangeHigh = 1U;
constexpr uint32_t kMaxFileNameLen = 128U;

map<std::string, GeAttrValue::ValueType> kAttrTypeDict = {
    {"bool", GeAttrValue::VT_BOOL},
    {"int", GeAttrValue::VT_INT},
    {"float", GeAttrValue::VT_FLOAT},
    {"string", GeAttrValue::VT_STRING},
    {"list_bool", GeAttrValue::VT_LIST_BOOL},
    {"list_int", GeAttrValue::VT_LIST_INT},
    {"list_float", GeAttrValue::VT_LIST_FLOAT},
    {"list_string", GeAttrValue::VT_LIST_STRING},
    {"list_list_int", GeAttrValue::VT_LIST_LIST_INT},
    {"data_type", GeAttrValue::VT_DATA_TYPE},
};

map<std::string, DataType> kDataTypeDict = {
    {"bool", DT_BOOL},
    {"int8", DT_INT8},
    {"uint8", DT_UINT8},
    {"int16", DT_INT16},
    {"uint16", DT_UINT16},
    {"int32", DT_INT32},
    {"uint32", DT_UINT32},
    {"int64", DT_INT64},
    {"uint64", DT_UINT64},
    {"float16", DT_FLOAT16},
    {"half", DT_FLOAT16},
    {"fp16", DT_FLOAT16},
    {"float", DT_FLOAT},
    {"float32", DT_FLOAT},
    {"double", DT_DOUBLE},
    {"complex32", DT_COMPLEX32},
    {"complex64", DT_COMPLEX64},
    {"complex128", DT_COMPLEX128},
    {"uint1", DT_UINT1},
    {"bfloat16", DT_BF16},
    {"int4", DT_INT4},
    {"hifloat8", DT_HIFLOAT8},
    {"float8_e5m2", DT_FLOAT8_E5M2},
    {"float8_e4m3fn", DT_FLOAT8_E4M3FN},
};

map<std::string, Format> kFormatDict = {
    {"nchw", FORMAT_NCHW},
    {"nhwc", FORMAT_NHWC},
    {"nd", FORMAT_ND},
    {"nc1hwc0", FORMAT_NC1HWC0},
    {"fractal_z", FORMAT_FRACTAL_Z},
    {"nc1c0hwpad", FORMAT_NC1C0HWPAD},
    {"nhwc1c0", FORMAT_NHWC1C0},
    {"fsr_nchw", FORMAT_FSR_NCHW},
    {"fractal_deconv", FORMAT_FRACTAL_DECONV},
    {"c1hwnc0", FORMAT_C1HWNC0},
    {"fractal_deconv_transpose", FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {"fractal_deconv_sp_stride_trans", FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {"nc1hwc0_c04", FORMAT_NC1HWC0_C04},
    {"fractal_z_c04", FORMAT_FRACTAL_Z_C04},
    {"chwn", FORMAT_CHWN},
    {"deconv_sp_stride8_trans", FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {"nc1khkwhwc0", FORMAT_NC1KHKWHWC0},
    {"bn_weight", FORMAT_BN_WEIGHT},
    {"filter_hwck", FORMAT_FILTER_HWCK},
    {"hwcn", FORMAT_HWCN},
    {"lookup_lookups", FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {"lookup_keys", FORMAT_HASHTABLE_LOOKUP_KEYS},
    {"lookup_value", FORMAT_HASHTABLE_LOOKUP_VALUE},
    {"lookup_output", FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {"lookup_hits", FORMAT_HASHTABLE_LOOKUP_HITS},
    {"md", FORMAT_MD},
    {"c1hwncoc0", FORMAT_C1HWNCoC0},
    {"fractal_nz", FORMAT_FRACTAL_NZ},
    {"ndhwc", FORMAT_NDHWC},
    {"ncdhw", FORMAT_NCDHW},
    {"dhwcn", FORMAT_DHWCN},
    {"dhwnc", FORMAT_DHWNC},
    {"ndc1hwc0", FORMAT_NDC1HWC0},
    {"fractal_z_3d", FORMAT_FRACTAL_Z_3D},
    {"fractal_z_3d_transpose", FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {"cn", FORMAT_CN},
    {"nc", FORMAT_NC},
    {"fractal_zn_lstm", FORMAT_FRACTAL_ZN_LSTM},
    {"fractal_z_g", FORMAT_FRACTAL_Z_G}
};

bool CheckFileNameIsValid(const std::string &file_name) {
  if ((file_name == ".") || (file_name == "..")) {
    return false;
  }
  const std::regex r("[a-zA-Z0-9\\._-]+");
  return std::regex_match(file_name, r);
}

std::string GenerateFileName(const SingleOpDesc &single_op_desc, const int32_t index) {
  std::string file_name = single_op_desc.name;
  if (file_name.length() > kMaxFileNameLen) {
    GELOGW("[GenerateFileName]Trim file name for it is too long, origin file name = %s", file_name.c_str());
    file_name = file_name.substr(0, kMaxFileNameLen);
  }
  if (CheckFileNameIsValid(file_name)) {
    file_name += kFileSuffix;
    GELOGI("Output om file name is from name field in json file, which is: %s", file_name.c_str());
    return file_name;
  }

  if (file_name.empty()) {
    GELOGI("There is no name field in json file, or name field is empty.");
  } else {
    GELOGW("[GenerateFileName]name field '%s' is invalid, valid file name can only contain 'a-z,A-Z,0-9,.,-,_', "
           "and can not be '.' nor '..'", file_name.c_str());
  }

  std::stringstream file_name_ss;
  file_name_ss << index;
  file_name_ss << "_" << single_op_desc.op;
  for (const auto &desc : single_op_desc.input_desc) {
    file_name_ss << "_" << desc.type << "_" << desc.format;
    for (const auto &dim : desc.dims) {
      file_name_ss << "_" << dim;
    }
  }

  for (const auto &desc : single_op_desc.output_desc) {
    file_name_ss << "_" << desc.type << "_" << desc.format;
    for (const auto &dim : desc.dims) {
      file_name_ss << "_" << dim;
    }
  }

  file_name = file_name_ss.str();
  if (file_name.length() > kMaxFileNameLen) {
    GELOGI("Trim file name for it is too long, origin file name = %s", file_name.c_str());
    file_name = file_name.substr(0, kMaxFileNameLen);
  }
  file_name += kFileSuffix;
  GELOGI("Om file name is: %s", file_name.c_str());
  return file_name;
}

bool AttrValueIsString(const Json &j, const std::string &key) {
  try {
    const std::string tmp_str = j.at(key).get<std::string>();
    return true;
  } catch (Json::type_error &) {
    return false;
  }
}

template<typename T>
void JsonConstToDescConst(const Json &j, SingleOpTensorDesc &desc) {
  const std::vector<T> json_const_value = j.at(kKeyConstValue).get<std::vector<T>>();
  desc.const_value_size = static_cast<uint64_t>(json_const_value.size() * sizeof(T));
  desc.const_value = std::shared_ptr<uint8_t> (new (std::nothrow) uint8_t[desc.const_value_size],
                                               std::default_delete<uint8_t[]>());
  const auto ret =
      memcpy_s(desc.const_value.get(), desc.const_value_size, json_const_value.data(), desc.const_value_size);
  if (ret != EOK) {
    GELOGW("[JsonConstToDescConst] memcpy failed.");
  }
}

template<typename T>
auto GetValue(const std::map<std::string, T> &dict, std::string &key, T default_val) -> T {
  (void)transform(key.begin(), key.end(), key.begin(), &::tolower);
  const auto it = dict.find(key);
  if (it == dict.end()) {
    return default_val;
  }

  return it->second;
}

template<typename T>
void SetAttrValue(const Json &j, SingleOpAttr &attr) {
  // when attr type is "data_type", we support two kinds of attr value.
  // 1. value: "DT_FLOAT", "DT_INT32", "DT_INT8" ...
  // 2. value: 1, 3 ...
  if ((j.at(kKeyType).get<std::string>() == "data_type") && AttrValueIsString(j, kKeyValue)) {
    const std::string type_str = j.at(kKeyValue).get<std::string>();
    const DataType dtype = TypeUtils::SerialStringToDataType(type_str);
    (void)attr.value.SetValue<DataType>(dtype);
    return;
  }
  (void)attr.value.SetValue<T>(j.at(kKeyValue).get<T>());
}
}  // namespace

void TransConstValue(const std::string &type_str, const Json &j, SingleOpTensorDesc &desc) {
  auto it = j.find(kKeyConstValue);
  if (it != j.end() && j.at(kKeyConstValue).is_array()) {
    switch (desc.type) {
      case DT_INT8:
      case DT_UINT8:
        JsonConstToDescConst<uint8_t>(j, desc);
        break;
      case DT_INT16:
      case DT_UINT16:
        JsonConstToDescConst<uint16_t>(j, desc);
        break;
      case DT_INT32:
      case DT_UINT32:
        JsonConstToDescConst<uint32_t>(j, desc);
        break;
      case DT_INT64:
      case DT_UINT64:
        JsonConstToDescConst<uint64_t>(j, desc);
        break;
      case DT_FLOAT16:
        JsonConstToDescConst<fp16_t>(j, desc);
        break;
      case DT_FLOAT:
        JsonConstToDescConst<float>(j, desc);
        break;
      case DT_DOUBLE:
        JsonConstToDescConst<double>(j, desc);
        break;
      default:
        GELOGE(UNSUPPORTED, "[Find][JsonAttr] name=%s, type=%s failed for Unsupported type.",
            kKeyType, type_str.c_str());
        REPORT_INNER_ERR_MSG("E19999", "[Find][JsonAttr] name=%s, type=%s failed for Unsupported type.",
            kKeyType, type_str.c_str());
        break;
    }
  }
}

void from_json(const Json &j, fp16_t &fp16) {
  const float32_t float32 = j.get<float>();
  fp16 = float32;
}

void from_json(const Json &j, SingleOpTensorDesc &desc) {
  bool is_tensor_valid = true;
  desc.dims = j.at(kKeyShape).get<std::vector<int64_t>>();
  auto it = j.find(kKeyShapeRange);
  if (it != j.end()) {
    desc.dim_ranges = j.at(kKeyShapeRange).get<std::vector<std::vector<int64_t>>>();
  }
  it = j.find(kKeyOriginShape);
  if (it != j.end()) {
    desc.ori_dims = j.at(kKeyOriginShape).get<std::vector<int64_t>>();
  }
  std::string format_str = j.at(kKeyFormat).get<std::string>();
  std::string type_str = j.at(kKeyType).get<std::string>();
  desc.format = GetValue(kFormatDict, format_str, FORMAT_RESERVED);
  desc.type = GetValue(kDataTypeDict, type_str, DT_UNDEFINED);
  is_tensor_valid = is_tensor_valid && ge::TypeUtilsInner::IsFormatValid(format_str);
  is_tensor_valid = is_tensor_valid && ge::TypeUtilsInner::IsDataTypeValid(type_str);
  it = j.find(kKeyOriginFormat);
  if (it != j.end()) {
    std::string origin_format_str = j.at(kKeyOriginFormat).get<string>();
    is_tensor_valid = is_tensor_valid && ge::TypeUtilsInner::IsFormatValid(origin_format_str);
    desc.ori_format = GetValue(kFormatDict, origin_format_str, FORMAT_RESERVED);
  }
  auto tensor_name = j.find(kKeyName);
  if (tensor_name != j.end()) {
    desc.name = tensor_name->get<string>();
  }
  auto dynamic_input_name = j.find(kKeyDynamicInput);
  if (dynamic_input_name != j.end()) {
    desc.dynamic_input_name = dynamic_input_name->get<string>();
  }
  desc.is_valid = is_tensor_valid ? desc.is_valid : is_tensor_valid;
  it = j.find(kKeyIsConst);
  if (it != j.end()) {
    desc.is_const = j.at(kKeyIsConst).get<bool>();
  }

  TransConstValue(type_str, j, desc);
}

void from_json(const Json &j, SingleOpAttr &attr) {
  attr.name = j.at(kKeyName).get<std::string>();
  attr.type = j.at(kKeyType).get<std::string>();
  const map<std::string, GeAttrValue::ValueType>::const_iterator it = kAttrTypeDict.find(attr.type);
  if (it == kAttrTypeDict.cend()) {
    GELOGE(UNSUPPORTED, "[Find][JsonAttr] name=%s, type=%s failed for Unsupported type.",
        attr.name.c_str(), attr.type.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Find jsonattr name=%s, type=%s failed for Unsupported type.",
        attr.name.c_str(), attr.type.c_str());
    return;
  }

  switch (it->second) {
    case GeAttrValue::VT_BOOL:
      SetAttrValue<bool>(j, attr);
      break;
    case GeAttrValue::VT_INT:
      SetAttrValue<int64_t>(j, attr);
      break;
    case GeAttrValue::VT_FLOAT:
      SetAttrValue<float>(j, attr);
      break;
    case GeAttrValue::VT_STRING:
      SetAttrValue<std::string>(j, attr);
      break;
    case GeAttrValue::VT_LIST_BOOL:
      SetAttrValue<std::vector<bool>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_INT:
      SetAttrValue<std::vector<int64_t>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_FLOAT:
      SetAttrValue<std::vector<float>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_STRING:
      SetAttrValue<std::vector<std::string>>(j, attr);
      break;
    case GeAttrValue::VT_LIST_LIST_INT:
      SetAttrValue<std::vector<std::vector<int64_t>>>(j, attr);
      break;
    case GeAttrValue::VT_DATA_TYPE:
      SetAttrValue<DataType>(j, attr);
      break;
    default:
      GELOGE(UNSUPPORTED, "[Find][JsonAttr] name=%s, type=%s failed for Unsupported type.",
          attr.name.c_str(), attr.type.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Find jsonattr name=%s, type=%s failed for Unsupported type.",
          attr.name.c_str(), attr.type.c_str());
      break;
  }
}

void from_json(const Json &j, SingleOpDesc &desc) {
  const auto op = j.find(kKeyOp);
  if (op != j.end()) {
    desc.op = j.at(kKeyOp).get<std::string>();
  }

  const auto name = j.find(kKeyName);
  if (name != j.end()) {
    desc.name = j.at(kKeyName).get<std::string>();
  }

  const auto input_desc = j.find(kKeyInputDesc);
  if (input_desc != j.end()) {
    desc.input_desc = input_desc->get<std::vector<SingleOpTensorDesc>>();
  }

  const auto output_desc = j.find(kKeyOutputDesc);
  if (output_desc != j.end()) {
    desc.output_desc = output_desc->get<std::vector<SingleOpTensorDesc>>();
  }

  const auto attr_field = j.find(kKeyAttr);
  if (attr_field != j.end()) {
    desc.attrs = attr_field->get<std::vector<SingleOpAttr>>();
  }

  const auto compile_flag = j.find(kKeyCompileFlag);
  if (compile_flag != j.end()) {
    desc.compile_flag = compile_flag->get<int32_t>();
  }
}

Status SingleOpParser::ReadJsonFile(const std::string &file, Json &json_obj) {
  std::string real_path = RealPath(file.c_str());
  if (real_path.empty()) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10023", std::vector<const char *>({"value"}),
                              std::vector<const char *>({file.c_str()}));
    GELOGE(FAILED, "[Read][JsonFile]Input parameter[--singleop]'s value[%s] is not a valid path.", file.c_str());
    return INTERNAL_ERROR;
  }

  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10024", std::vector<const char *>({"value"}),
                              std::vector<const char *>({file.c_str() }));
    GELOGE(FAILED, "[Open][JsonFile] failed for file[%s] provided in input parameter[--singleop].", file.c_str());
    return FAILED;
  }
  try {
    ifs >> json_obj;
  } catch (const std::exception &e) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10025", std::vector<const char *>({"realpath", "errmsg"}),
                              std::vector<const char *>({real_path.c_str(), e.what()}));
    GELOGE(PARAM_INVALID,
        "[Parse][JsonFile] fail for file[%s] provided in input parameter[--singleop], exception = %s.",
        real_path.c_str(), e.what());
    return PARAM_INVALID;
  }

  ifs.close();
  return SUCCESS;
}

bool SingleOpParser::Validate(const SingleOpDesc &op_desc) {
  if (op_desc.op.empty()) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10026", std::vector<const char *>({}), std::vector<const char *>({}));
    GELOGE(PARAM_INVALID, "[Check][Param] fail for name of input SingleOpDesc is empty.");
    return false;
  }
  int32_t index = 0;
  const auto report_command_err = [&index, &op_desc](const char_t *const in_out, const char_t *const dt_ft) -> bool {
    REPORT_PREDEFINED_ERR_MSG(
        "E10027", std::vector<const char *>({"op_name", "input_output", "attr", "index"}),
        std::vector<const char *>({op_desc.op.c_str(), in_out, dt_ft, std::to_string(index).c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param] The attribute [%s] of [%s] tensor[%d] for Op [%s] is invalid!",
        dt_ft, in_out, index, op_desc.op.c_str());
    return false;
  };
  for (auto &tensor_desc : op_desc.input_desc) {
    if ((tensor_desc.type == DT_UNDEFINED) && (tensor_desc.format != FORMAT_RESERVED)) {
      return report_command_err("input", "datatype");
    }
    if ((tensor_desc.type != DT_UNDEFINED) && (tensor_desc.format == FORMAT_RESERVED)) {
      return report_command_err("input", "format");
    }
    if (!tensor_desc.is_valid) {
      return report_command_err("input", "dataType or format");
    }
    ++index;
  }
  index = 0;
  for (auto &tensor_desc : op_desc.output_desc) {
    if (tensor_desc.type == DT_UNDEFINED) {
      return report_command_err("output", "datatype");
    }
    if (tensor_desc.format == FORMAT_RESERVED) {
      return report_command_err("output", "format");
    }
    if (!tensor_desc.is_valid) {
      return report_command_err("output", "dataType or format");
    }
    ++index;
  }
  for (auto &attr : op_desc.attrs) {
    if (attr.name.empty()) {
      (void)REPORT_PREDEFINED_ERR_MSG("E10029", std::vector<const char *>({"op_name"}),
                                std::vector<const char *>({op_desc.op.c_str()}));
      GELOGE(PARAM_INVALID, "[Parse][Attr]attr name is empty");
      return false;
    }
    if (attr.value.IsEmpty()) {
      (void)REPORT_PREDEFINED_ERR_MSG("E10030", std::vector<const char *>({"op_name", "attrname"}),
                                std::vector<const char *>({op_desc.op.c_str(), attr.name.c_str()}));
      GELOGE(PARAM_INVALID, "[Parse][Attr] fail for vale of attr name:\"%s\" is empty. ", attr.name.c_str());
      return false;
    }
  }
  return true;
}

std::unique_ptr<OpDesc> SingleOpParser::CreateOpDesc(const std::string &name, const std::string &op_type) {
  const auto &ret = name.empty() ? op_type : name;
  return MakeUnique<OpDesc>(ret, op_type);
}

Status SingleOpParser::UpdateDynamicTensorName(std::vector<SingleOpTensorDesc> &desc) {
  std::map<std::string, int32_t> dynamic_name_map;
  for (auto &tensor : desc) {
    if (tensor.dynamic_input_name.empty()) {
      continue;
    }
    if (dynamic_name_map.find(tensor.dynamic_input_name) == dynamic_name_map.end()) {
      dynamic_name_map[tensor.dynamic_input_name] = 0;
    } else {
      dynamic_name_map[tensor.dynamic_input_name]++;
    }
    tensor.name = tensor.dynamic_input_name + std::to_string(dynamic_name_map[tensor.dynamic_input_name]);
  }
  GELOGD("Update dynamic tensor name success!");
  return SUCCESS;
}

Status SingleOpParser::ConvertToBuildParam(int32_t index,
                                           const SingleOpDesc &single_op_desc,
                                           SingleOpBuildParam &build_param) {
  auto op_desc = CreateOpDesc(single_op_desc.name, single_op_desc.op);
  GE_CHECK_NOTNULL(op_desc);

  for (auto &desc : single_op_desc.input_desc) {
    GeTensorDesc ge_tensor_desc(GeShape(desc.dims),
                                desc.format,
                                desc.type);
    const auto ori_format_to_set = desc.ori_format != FORMAT_RESERVED ? desc.ori_format : desc.format;
    const auto ori_dims = !desc.ori_dims.empty() ? desc.ori_dims : desc.dims;
    ge_tensor_desc.SetOriginFormat(ori_format_to_set);
    ge_tensor_desc.SetOriginShape(GeShape(ori_dims));
    GE_CHK_STATUS_RET_NOLOG(SetShapeRange(op_desc->GetName(), desc, ge_tensor_desc));
    TensorUtils::SetRealDimCnt(ge_tensor_desc, static_cast<uint32_t>(ori_dims.size()));
    TensorUtils::SetInputTensor(ge_tensor_desc, true);
    TensorUtils::SetOutputTensor(ge_tensor_desc, false);
    if (desc.is_const) {
      (void)AttrUtils::SetBool(ge_tensor_desc, kKeyIsConst, desc.is_const);
      GeTensorDesc value_desc(GeShape(desc.dims), desc.format, desc.type);
      const GeTensorPtr value_tensor = ge::MakeShared<GeTensor>(value_desc, desc.const_value.get(),
                                                                desc.const_value_size);
      GE_CHECK_NOTNULL(value_tensor);
      if (!AttrUtils::SetTensor(ge_tensor_desc, kKeyValue, value_tensor)) {
        GELOGW("[SetTensor] Set attr name %s failed", kKeyValue);
      }
    }
    if (desc.name.empty()) {
      (void)op_desc->AddInputDesc(ge_tensor_desc);
    } else {
      ge_tensor_desc.SetName(desc.name);
      (void)op_desc->AddInputDesc(desc.name, ge_tensor_desc);
    }
    (void)build_param.inputs.emplace_back(ge_tensor_desc);
  }

  for (auto &desc : single_op_desc.output_desc) {
    GeTensorDesc ge_tensor_desc(GeShape(desc.dims),
                                desc.format,
                                desc.type);
    const auto ori_format_to_set = desc.ori_format != FORMAT_RESERVED ? desc.ori_format : desc.format;
    const auto ori_dims = !desc.ori_dims.empty() ? desc.ori_dims : desc.dims;
    ge_tensor_desc.SetOriginFormat(ori_format_to_set);
    ge_tensor_desc.SetOriginShape(GeShape(ori_dims));
    GE_CHK_STATUS_RET_NOLOG(SetShapeRange(op_desc->GetName(), desc, ge_tensor_desc));
    TensorUtils::SetRealDimCnt(ge_tensor_desc, static_cast<uint32_t>(ori_dims.size()));
    TensorUtils::SetInputTensor(ge_tensor_desc, false);
    TensorUtils::SetOutputTensor(ge_tensor_desc, true);
    if (desc.name.empty()) {
      (void)op_desc->AddOutputDesc(ge_tensor_desc);
    } else {
      ge_tensor_desc.SetName(desc.name);
      (void)op_desc->AddOutputDesc(desc.name, ge_tensor_desc);
    }
    (void)build_param.outputs.emplace_back(ge_tensor_desc);
  }

  for (const auto &attr : single_op_desc.attrs) {
    (void)op_desc->SetAttr(attr.name, attr.value);
  }

  if (VerifyOpInputOutputSizeByIr(*op_desc) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Verify][OpInputOutputSize] fail for input op [%s] invalid.", op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  build_param.file_name = GenerateFileName(single_op_desc, index);
  build_param.op_desc.reset(op_desc.release());
  return SUCCESS;
}

Status SingleOpParser::VerifyOpInputOutputSizeByIr(const OpDesc &current_op_desc) {
  const ge::Operator operator_ir = ge::OperatorFactory::CreateOperator("tmp_operator",
                                                                       current_op_desc.GetType().c_str());
  if (!operator_ir.IsEmpty()) {
    const auto opdesc_ir = ge::OpDescUtils::GetOpDescFromOperator(operator_ir);
    GE_CHECK_NOTNULL(opdesc_ir);
    const size_t current_opdesc_inputs_num = current_op_desc.GetInputsSize();
    const size_t ir_opdesc_inputs_num = opdesc_ir->GetInputsSize();
    if (current_opdesc_inputs_num < ir_opdesc_inputs_num) {
      const std::string reason = "The number of actual input in operator is less than the required number of inputs";
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
          std::vector<const char *>({current_op_desc.GetName().c_str(), "input size",
                                     std::to_string(current_opdesc_inputs_num).c_str(), reason.c_str()}));
      GELOGE(PARAM_INVALID,
          "[Verify][OpInputOutputSize]This op:%s input size %zu is smaller than the ir needed input size %zu",
          current_op_desc.GetName().c_str(), current_opdesc_inputs_num, ir_opdesc_inputs_num);
      return PARAM_INVALID;
    }
    const size_t current_opdesc_outputs_num = current_op_desc.GetOutputsSize();
    const size_t ir_opdesc_outputs_num = opdesc_ir->GetOutputsSize();
    if (current_opdesc_outputs_num < ir_opdesc_outputs_num) {
      const std::string reason = "The number of actual output in operator is less than the required number of outputs";
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
          std::vector<const char *>({current_op_desc.GetName().c_str(), "output size",
                                     std::to_string(current_opdesc_outputs_num).c_str(), reason.c_str()}));
      GELOGE(PARAM_INVALID,
          "[Verify][OpInputOutputSize]This op:%s output size %zu is smaller than the ir needed output size %zu",
          current_op_desc.GetName().c_str(), current_opdesc_outputs_num, ir_opdesc_outputs_num);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status SingleOpParser::SetShapeRange(const std::string &op_name,
                                     const SingleOpTensorDesc &tensor_desc,
                                     GeTensorDesc &ge_tensor_desc) {
  const auto num_shape_ranges = tensor_desc.dim_ranges.size();
  GELOGD("Number of shape ranges = %zu", num_shape_ranges);
  const auto it = std::find(tensor_desc.dims.begin(), tensor_desc.dims.end(), ge::UNKNOWN_DIM_NUM);
  if (it != tensor_desc.dims.end()) {
    if (tensor_desc.dims != ge::UNKNOWN_RANK) {
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
          std::vector<const char *>({op_name.c_str(), "shape", formats::JoinToString(tensor_desc.dims).c_str(),
                                     "The tensor has unknown rank, so its shape must be set to {-2}"}));
      GELOGE(PARAM_INVALID, "[Set][ShapeRange]Invalid tensor shape:%s.",
          ge_tensor_desc.MutableShape().ToString().c_str());
      return PARAM_INVALID;
    }
    if (!tensor_desc.dim_ranges.empty()) {
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
          std::vector<const char *>({op_name.c_str(), "shape range size",
                                     std::to_string(tensor_desc.dim_ranges.size()).c_str(),
                                     "The tensor has dynamic dimensions, but the shape range is not empty"}));
      GELOGE(PARAM_INVALID, "[Set][ShapeRange]Shape range is not needed while the rank the shape is unknown.");
      return PARAM_INVALID;
    }

    GELOGD("Shape is unknown rank, do not set shape range");
    return SUCCESS;
  }

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  size_t range_index = 0;
  for (int64_t dim : tensor_desc.dims) {
    if (dim >= 0) {
      (void)shape_range.emplace_back(dim, dim);
      GELOGD("Adding shape range: [%ld, %ld]", dim, dim);
    } else {
      GELOGD("To get shape range by index = %zu", range_index);
      if (range_index >= num_shape_ranges) {
        ++range_index;
        std::string reason =
            "The number of dimensions with a configured shape range is inconsistent with the actual number of dynamic "
            "dimensions";
        (void)REPORT_PREDEFINED_ERR_MSG(
            "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
            std::vector<const char *>(
                {op_name.c_str(), "shape range num", std::to_string(num_shape_ranges).c_str(), reason.c_str()}));
        GELOGE(PARAM_INVALID, "[Set][ShapeRange]The number of shape_range mismatches that of unknown dims.");
        return PARAM_INVALID;
      }

      auto &range = tensor_desc.dim_ranges[range_index];
      if (range.size() != kShapeRangePairSize) {
        std::string reason = "The format of shape range " + std::to_string(range_index) +
                             " is invalid. The expected number of shape ranges is 2, but actual number is " +
                             std::to_string(range.size());
        (void)REPORT_PREDEFINED_ERR_MSG(
            "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
            std::vector<const char *>({op_name.c_str(),
                                       ("shape range " + std::to_string(range_index) + " size").c_str(),
                                       std::to_string(range.size()).c_str(), reason.c_str()}));
        GELOGE(PARAM_INVALID, "[Set][ShapeRange]Invalid shape range entry. index = %zu, size = %zu",
            range_index, range.size());
        return PARAM_INVALID;
      }

      (void)shape_range.emplace_back(range[kShapeRangeLow], range[kShapeRangeHigh]);
      GELOGD("Adding shape range: [%ld, %ld]", range[kShapeRangeLow], range[kShapeRangeHigh]);
      ++range_index;
    }
  }

  if (num_shape_ranges != range_index) {
    std::string reason = "The number of dimensions with a configured shape range is inconsistent with the actual number of dynamic dimensions";
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
        std::vector<const char *>(
            {op_name.c_str(), "shape range num", std::to_string(num_shape_ranges).c_str(), reason.c_str()}));
    GELOGE(PARAM_INVALID,
        "[Set][ShapeRange]The number of shape_range(%zu) mismatches that of unknown dims(%zu).",
        num_shape_ranges, range_index);
    return PARAM_INVALID;
  }

  if (range_index > 0) {
    (void)ge_tensor_desc.SetShapeRange(shape_range);
  }

  return SUCCESS;
}

Status SingleOpParser::ParseSingleOpList(const std::string &file, std::vector<SingleOpBuildParam> &op_list) {
  int32_t index = 0;
  try {
    Json single_op_list_json;
    auto ret = ReadJsonFile(file, single_op_list_json);
    if (ret != SUCCESS) {
      return ret;
    }
    // 这个定义不能放在循环里，因为多个op共享一个compile_flag
    int32_t compile_flag = 0;
    for (const Json &single_op_json : single_op_list_json) {
      const std::string dump_info = single_op_json.dump(kDumpJsonIndent);
      GELOGI("Parsing op[%d], jsonStr: %s", index, dump_info.c_str());
      SingleOpDesc single_op_desc;
      from_json(single_op_json, single_op_desc);
      GELOGD("Compile flag is: %d.", single_op_desc.compile_flag);
      if (single_op_desc.compile_flag == 1) {
        compile_flag = single_op_desc.compile_flag;
        continue;
      }

      (void)UpdateDynamicTensorName(single_op_desc.input_desc);  // Never failed

      if (!Validate(single_op_desc)) {
        GELOGE(PARAM_INVALID,
            "[Check][OpDesc]Validate the index[%d] of op failed when read json file[%s].", index, file.c_str());
        return PARAM_INVALID;
      }

      SingleOpBuildParam param;
      ret = ConvertToBuildParam(index, single_op_desc, param);
      if (ret != SUCCESS) {
        return ret;
      }
      param.compile_flag = compile_flag;

      (void)op_list.emplace_back(param);
      GELOGI("Parse the index[%d] of op[%s] success", index, single_op_desc.op.c_str());
      index += 1;
    }
  } catch (const nlohmann::json::exception &e) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10032", std::vector<const char *>({"file_name", "reason"}),
                                    std::vector<const char *>({file.c_str(), e.what()}));
    GELOGE(PARAM_INVALID, "[Parse][OpList] the index:%d of op failed when read json file:%s, exception:%s",
        index, file.c_str(), e.what());
    return PARAM_INVALID;
  }

  return SUCCESS;
}
} // namespace ge

