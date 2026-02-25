/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api/aclgrph/option_utils.h"

#include "base/err_msg.h"

#include <iostream>
#include <string>
#include <regex>
#include "common/helper/file_saver.h"
#include "base/err_msg.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/checker.h"
#include "common/screen_printer.h"
#include "ge/ge_api_types.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/ge_context.h"
#include "common/checker.h"
#include "graph/compute_graph.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_type_utils.h"
#include "register/optimization_option_registry.h"
#include "graph/option/optimization_option.h"
#include "base/err_msg.h"

namespace ge {
namespace {
const int64_t kDynamicInputDim = -1;
const int64_t kDynamicImageSizeNum = 2;
const constexpr size_t kLeastStrElementNum = 2UL;
const int32_t kBase = 10;
// datatype/formats from user to GE, Unified to util interface file later
const std::map<std::string, ge::DataType> kOutputTypeSupportDatatype = {
    {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}, {"INT8", ge::DT_INT8},
    {"HIF8", ge::DT_HIFLOAT8}, {"FP8E5M2", ge::DT_FLOAT8_E5M2}, {"FP8E4M3FN", ge::DT_FLOAT8_E4M3FN},
};
const char *const kOutputTypeSupport =
    "The current value is not within the valid range. Only support FP32, FP16, UINT8, INT8, HIF8, FP8E5M2, FP8E4M3FN.";
const std::set<std::string> kBufferOptimizeSupportOption = {"l1_optimize", "l2_optimize", "off_optimize",
                                                            "l1_and_l2_optimize"};
// The function is incomplete. Currently, only l2_optimize, off_optimize is supported.
const char *const kBufferOptimizeSupport = "The value must be  l2_optimize or off_optimize.";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT = "high_performance";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_PRECISON = "high_precision";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PRECISION_FOR_ALL = "high_precision_for_all";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PERFORMANCE_FOR_ALL = "high_performance_for_all";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_ALLOW_FP32 = "enable_float_32_execution";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_ALLOW_HI_FP32 = "enable_hi_float_32_execution";
const char *const kInputShapeSample1 = "\"input_name1:n1,c1,h1,w1\"";
const char *const kInputShapeSample2 = "\"input_name1:1,3,224,224\"";
const char *const kSplitError1 = "The shape must contain two parts: name and value";
const char *const kEmptyError = "The shape has a parameter name, whose value cannot be empty";
const char *const kFloatNumError = "The float number is unsupported";
const char *const kDigitError = "It is not a digit";
const char *const kCompressWeightError =
    "Parameters --op_select_implmode and --optypelist_for_implmode must be set at the same time.";
const char *const kSelectImplmodeError = "The value must be high_performance, high_precision, "
                                         "high_precision_for_all, high_performance_for_all.";
const char *const kDynamicBatchSizeError = "It can only contain digits and \",\".";
const char *const kDynamicImageSizeError = "It can only contain digits, \",\", \" \" and \";\".";
const char *const kKeepDtypeError = "File defined by keep_dtype is not found.";
const char *const kModifyMixlistPrecisonModeError =
    "Modify_mixlist is set. Please ensure that precision_mode is set to any of {allow_mix_precision, "
    "allow_mix_precision_fp16, allow_mix_precision_bf16(if available}.";
const char *const kModifyMixlistPrecisonModeV2Error =
    "Modify_mixlist is set. Please ensure that precision_mode_v2 is set to any of {mixed_float16, mixed_bfloat16, "
    "mixed_hif8(if available)}.";
const char *const kModifyMixlistError = "modify_mixlist is assigned, "
    "Please ensure that precision_mode only can be assigned to 'allow_mix_precision' or "
    "'allow_mix_precision_fp16' or 'allow_mix_precision_bf16'(if available), "
    "precision_mode_v2 only can be assigned to 'mixed_float16' or "
    "'mixed_bfloat16' or 'mixed_hif8'(if available).";
const char *const kInValidValueRange = "The current value is not within the valid range.";

// parser input_shape_range
const size_t kSquareBracketsSize = 2;
const size_t kRangePairSize = 2;
const size_t kShapeRangeSize = 2;
const size_t kShapeRangeStrIndex = 2;
const size_t kShapeRangeStrSize = 1;
const size_t kShapeRangeVecNameIndex = 0;
const size_t kShapeRangeVecValueIndex = 1;
const size_t kShapeRangeVecSize = 2;
const char *const kInvalidReasonBrackets = "The value can only contains a pair of '[]'";
const char *const kInputShapeRangeInvalid = "The format of the shape range is invalid";
const char *const kInputShapeRangeSizeInvalid = "The shape range size less than 2, and it is invalid";
const char *const kShapeRangeValueConvertError = "The current string cannot be converted to a number";
const char *const kInputShapeRangeSample1 = "\"input_name1:[n1~n2,c1,h1,w1]\"";
const char *const kInputShapeRangeSample2 = "\"1~20\"";
const char *const kInputShapeRangeSample3 = "\"[1~20,3,3~6,-1]\"";
const char *const kInputShapeRangeSample4 = "\"input_name1:[1~20,3,3~6,-1];input_name2:[1~20,3,3~6,-1]\"";
const char *const kInputShapeRangeSample5 = "\"16\"";
const char *const kInputShapeRangeSample6 = "\"input_name1:n1~n2,c1,h1,w1\"";
const char *const kInputShapeRangeSample7 = "\"n1~n2,c1,h1,w1;n3,c2,h2,w2\"";
const char *const kHintInputShape = "ge.inputHintShape";
const std::vector<int64_t> kDummyShape = {-3};

const std::unordered_set<std::string> kSupportedPrintMode = {"enable", "disable"};

constexpr const char *kExternalWeightDisabled = "0";           // 禁用外置权重
constexpr const char *kExternalWeightEnabled = "1";            // 启用外置权重，每个权重单独导出
constexpr const char *kExternalWeightCombined = "2";           // 启用外置权重，所有权重合并导出到同一文件

vector<std::string> SplitInputShape(const std::string &input_shape) {
  std::vector<std::string> shape_pair_vec;
  size_t pos = input_shape.rfind(":");
  if (pos != std::string::npos) {
    shape_pair_vec.emplace_back(input_shape.substr(0, pos));
    shape_pair_vec.emplace_back(input_shape.substr(pos + 1, input_shape.size() - pos));
  }
  return shape_pair_vec;
}

Status ConstructShapeFromStr(const std::string &shape_str, GeShape &shape) {
  // Shape的字符串至少有2个[]字符, 校验shape是否由[]包括
  GE_ASSERT_TRUE(shape_str.length() >= kLeastStrElementNum && shape_str.front() == '[' && shape_str.back() == ']');
  // 去除中括号
  auto shape_dims_str = shape_str.substr(1, shape_str.length() - kLeastStrElementNum);
  auto dim_strs = ge::StringUtils::Split(shape_dims_str, ',');
  shape.SetDimNum(0UL);
  for (auto &str : dim_strs) {
    if (str.empty()) {
      continue;
    }
    int64_t dim = -1;
    GE_ASSERT_SUCCESS(ConvertToInt64(ge::StringUtils::Trim(str), dim),
        "Shape: %s is invalid in option %s", shape_str.c_str(), kHintInputShape);
    GE_ASSERT_TRUE(dim >= 0L, "Shape in ge.inputHintOption should not less than 0, but get: %lld.", dim);
    shape.AppendDim(dim);
  }
  return GRAPH_SUCCESS;
}

static bool StringToLongNoThrow(const std::string &str, long &val) {
  std::string val_str(str);
  std::stringstream ss(StringUtils::Trim(val_str));
  ss >> val;
  if (ss.fail() || !ss.eof()) {
    REPORT_PREDEFINED_ERR_MSG("E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
                       std::vector<const char *>({str.c_str(), kShapeRangeValueConvertError, kInputShapeRangeSample5}));
    GELOGE(PARAM_INVALID, "[Parse][Parameter] failed, reason: %s, str: \"%s\", correct sample is %s.",
           kShapeRangeValueConvertError, str.c_str(), kInputShapeRangeSample5);
    return false;
  }
  return true;
}

static bool ParseShapeRangePair(const std::string &shape_range,
                                const std::vector<std::string> &range_pair_set,
                                std::pair<int64_t, int64_t> &range_pair) {
  if (range_pair_set.size() == 1) {
    long range_value = 0;
    if (!StringToLongNoThrow(range_pair_set.at(0), range_value)) {
      return false;
    }
    if (range_value < 0) {
      range_pair = std::make_pair(SHAPE_RANGE_LOWER_LIMIT, range_value);
    } else {
      range_pair = std::make_pair(range_value, range_value);
    }
  } else if (range_pair_set.size() == kRangePairSize) {
    // unknown dim, should get range.
    long range_left = 0;
    if (!StringToLongNoThrow(range_pair_set.at(0), range_left)) {
      return false;
    }
    long range_right = 0;
    if (!StringToLongNoThrow(range_pair_set.at(1), range_right)) {
      return false;
    }
    if ((range_left < 0) || (range_right < 0)) {
      REPORT_PREDEFINED_ERR_MSG("E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
                         std::vector<const char *>({shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample3}));
      GELOGE(PARAM_INVALID,
             "[Parse][InputParameter] [--input_shape_range]'s shape range[%s] failed,"
             "reason: %s, correct sample is %s.",
             shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample3);
      return false;
    }
    range_pair = std::make_pair(range_left, range_right);
  } else {
    REPORT_PREDEFINED_ERR_MSG("E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
                       std::vector<const char *>({shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample3}));
    GELOGE(PARAM_INVALID, "[Parse][Parameter]shape_range:%s invalid, reason: %s, correct sample is %s.",
           shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample3);
    return false;
  }
  return true;
}

std::string StringSetToString(const std::unordered_set<std::string> &allowed_set) {
  const std::string delim = ", ";
  std::string debug;
  for (const auto &item : allowed_set) {
    if (item.empty()) {
      continue;
    }
    debug.append(item + delim);
  }
  if (!debug.empty()) {
    debug.erase(debug.rfind(delim));
  }
  return "{" + debug + "}";
}

std::string GetOptionValue(const std::map<std::string, std::string>& options, const std::string& option_name) {
  auto it = options.find(option_name);
  if (it != options.end()) {
    return it->second;
  }
  return "";
}
}  // namespace

Status CheckInputFormat(const std::string &input_format) {
  if (input_format.empty()) {
    return ge::SUCCESS;
  }
  if (!ge::TypeUtilsInner::IsFormatValid(input_format.c_str())) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"--input_format", input_format.c_str(), kInValidValueRange}));
    GELOGE(ge::PARAM_INVALID, "[Check][InputFormat] --input_format[%s] is invalid!", input_format.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

bool CheckDynamicBatchSizeInputShapeValid(std::map<std::string, std::vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size) {
  int32_t size = 0;
  for (auto iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    std::vector<int64_t> shape = iter->second;
    if (shape.empty()) {
      REPORT_PREDEFINED_ERR_MSG("E10012", std::vector<const char *>({}), std::vector<const char *>({}));
      GELOGE(ge::PARAM_INVALID,
          "[Check][DynamicBatchSizeInputShape] shape size can not be less than 1 when set --dynamic_batch_size.");
      return false;
    }

    if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    }

    bool ret = multibatch::CheckDynamicBatchShape(shape, iter->first);
    if (ret) {
      size++;
    }
  }

  if (size == 0) {
    REPORT_PREDEFINED_ERR_MSG("E10031", std::vector<const char *>({}), std::vector<const char *>({}));
    GELOGE(ge::PARAM_INVALID,
        "[Check][DynamicBatchSizeInputShape]At least one batch n must be equal to -1 when set dynamic_batch_size.");
    return false;
  }

  for (char c : dynamic_batch_size) {
    if (!isdigit(c) && (c != ',')) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({"dynamic_batch_size", dynamic_batch_size.c_str(), kDynamicBatchSizeError}));
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicBatchSizeInputShape] --dynamic_batch_size:%s is invalid. reason: %s",
          dynamic_batch_size.c_str(), kDynamicBatchSizeError);
      return false;
    }
  }
  if (dynamic_batch_size.back() == ',') {
    dynamic_batch_size.erase(dynamic_batch_size.end() - 1);
  }
  return true;
}

bool CheckDynamicImagesizeInputShapeValid(std::map<std::string, std::vector<int64_t>> shape_map,
                                          const std::string &input_format, std::string &dynamic_image_size) {
  if (!input_format.empty() && !ge::TypeUtilsInner::IsFormatValid(input_format.c_str())) {
    GELOGE(ge::PARAM_INVALID,
        "[Check][DynamicImagesizeInputShape] input_format [%s] invalid, can not support now.", input_format.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10003", std::vector<const char *>({"parameter", "value", "reason"}),
                       std::vector<const char *>({"input_format", input_format.c_str(), "This format is not supported."}));
    return false;
  }
  int32_t size = 0;
  for (auto iter = shape_map.cbegin(); iter != shape_map.cend(); ++iter) {
    std::vector<int64_t> shape = iter->second;
    // only support four dim
    if (shape.size() != DIM_DEFAULT_SIZE) {
      if (std::count(shape.begin(), shape.end(), kDynamicInputDim) > 0) {
        REPORT_PREDEFINED_ERR_MSG("E10019", std::vector<const char *>({}), std::vector<const char *>({}));
        GELOGE(ge::PARAM_INVALID, "[Check][DynamicImagesizeInputShape] --input_shape invalid,"
            " only height and width can be -1 when set --dynamic_image_size.");
        return false;
      }
      continue;
    }

    if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    }
    auto ret = multibatch::CheckDynamicImageSizeShape(shape, input_format);
    if (ret) {
      size++;
    } else {
      return ret;
    }
  }
  if (size == 0) {
    REPORT_PREDEFINED_ERR_MSG("E10019", std::vector<const char *>({}), std::vector<const char *>({}));
    GELOGE(ge::PARAM_INVALID, "[Check][DynamicImagesizeInputShape]--input shape invalid, "
        "only height and width can be -1 when set --dynamic_image_size.");
    return false;
  }

  EraseEndSemicolon(dynamic_image_size);
  for (char c : dynamic_image_size) {
    bool is_char_valid = isdigit(c) || (c == ',') || (c == ' ') || (c == ';');
    if (!is_char_valid) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({"dynamic_image_size", dynamic_image_size.c_str(), kDynamicImageSizeError}));
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicImageSizeInputShape] --dynamic_image_size:%s is invalid. reason: %s",
             dynamic_image_size.c_str(), kDynamicImageSizeError);
      return false;
    }
  }
  // Different parameter sets are split std::string by ';'
  std::vector<std::string> split_set = StringUtils::Split(dynamic_image_size, ';');
  // Different dimensions are split by ','
  std::vector<std::string> split_dim;
  for (auto str : split_set) {
    split_dim = StringUtils::Split(str, ',');
    if (split_dim.size() != static_cast<size_t>(kDynamicImageSizeNum)) {
      REPORT_PREDEFINED_ERR_MSG("E10020", std::vector<const char *>({"dynamic_image_size"}),
                         std::vector<const char *>({dynamic_image_size.c_str()}));
      GELOGE(ge::PARAM_INVALID,
          "[Check][DynamicImagesizeInputShape] invalid value:%s number of dimensions of each group must be %ld.",
          dynamic_image_size.c_str(), kDynamicImageSizeNum);
      return false;
    }
  }

  return true;
}

bool CheckDynamicDimsInputShapeValid(const std::map<std::string, std::vector<int64_t>> &shape_map,
                                     std::string &dynamic_dims) {
  int32_t dynamic_dim = 0;
  for (auto &info_shapes : shape_map) {
    auto &shapes = info_shapes.second;
    dynamic_dim += std::count(shapes.begin(), shapes.end(), kDynamicInputDim);
  }
  if (dynamic_dim == 0) {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({"--input_shape's dynamic dim num", "0",
                                                         "At least one dim should be -1 when dynamic_dims is set."}));
    GELOGE(ge::PARAM_INVALID,
           "[Check][DynamicDimsInputShape]--input_shape invalid,"
           "at least one dim should be -1 when set dynamic_dims.");
    return false;
  }

  if (!CheckAndParseDynamicDims(dynamic_dim, dynamic_dims)) {
    GELOGE(ge::PARAM_INVALID, "[CheckAndParse][DynamicDims]failed, %s invalid.", dynamic_dims.c_str());
    return false;
  }

  return true;
}

bool CheckAndParseDynamicDims(int32_t dynamic_dim_num, std::string &dynamic_dims) {
  EraseEndSemicolon(dynamic_dims);
  if (dynamic_dims.empty()) {
    REPORT_PREDEFINED_ERR_MSG("E10058", std::vector<const char *>({"parameter"}),
                              std::vector<const char *>({"--dynamic_dims"}));
    GELOGE(ge::PARAM_INVALID, "[CheckAndParse][DynamicDims]--dynamic_dims can not be empty.");
    return false;
  }
  // Different parameter sets are split by ';'
  std::vector<std::string> split_set = StringUtils::Split(dynamic_dims, ';');
  for (auto split_dim : split_set) {
    std::vector<std::string> one_set = StringUtils::Split(split_dim, ',');
    if (one_set.size() != static_cast<size_t>(dynamic_dim_num)) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({"dynamic_dims", dynamic_dims.c_str(),
            "Each setting needs to be consistent with the number of -1 in the input shape."}));
      GELOGE(ge::PARAM_INVALID, "[CheckAndParse][DynamicDims] --dynamic_dims:%s invalid. "
          "reason: Each gear setting needs to be consistent with the number of -1 in the inputshape.",
          dynamic_dims.c_str());
      return false;
    }
    for (auto dim : one_set) {
      for (auto c : dim) {
        if (!isdigit(c)) {
          REPORT_PREDEFINED_ERR_MSG(
              "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
              std::vector<const char *>({"--dynamic_dims", dim.c_str(), "Dynamic dims must be a positive integer."}));
          GELOGE(ge::PARAM_INVALID,
              "[CheckAndParse][DynamicDims]--dynamic_dims:%s parameter must be positive integer.",
              dynamic_dims.c_str());
          return false;
        }
      }
    }
  }
  return true;
}

bool ParseSingleShapeRange(std::string &shape_range, std::vector<std::pair<int64_t, int64_t>> &shape_range_vec) {
  std::vector<char> square_brackets;
  for (auto ch : shape_range) {
    if (ch == '[' || ch == ']') {
      square_brackets.push_back(ch);
    }
  }
  bool is_square_brackets = (square_brackets.size() == kSquareBracketsSize) &&
                            (square_brackets[0] == '[') && (square_brackets[1] == ']');
  if (!is_square_brackets) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
        std::vector<const char *>({shape_range.c_str(), kInvalidReasonBrackets, kInputShapeRangeSample1}));
    GELOGE(PARAM_INVALID, "[Parse][Parameter] shape_range:%s invalid, reason: %s, correct sample is %s.",
        shape_range.c_str(), kInvalidReasonBrackets, kInputShapeRangeSample1);
    return false;
  }
  // trim start bytes, after that, single input should be "1~20,3,3~6,-1"
  if (ge::StringUtils::StartWith(shape_range, "[")) {
    shape_range = shape_range.substr(1, shape_range.size() - 1);
  }
  if ((shape_range.size() > 0) && (shape_range[shape_range.size() - 1] == ']')) {
    shape_range = shape_range.substr(0, shape_range.size() - 1);
  }
  // parse shape_range of single input. eg. "1~20,3,3~6,-1"
  uint32_t shape_range_index = 1UL;
  std::vector<std::string> dim_range_set = ge::StringUtils::Split(shape_range, ',');
  for (const auto &range_pair_str : dim_range_set) {
    std::vector<std::string> range_pair_set = ge::StringUtils::Split(range_pair_str, '~');
    std::pair<int64_t, int64_t> range_pair;
    if (!ParseShapeRangePair(shape_range, range_pair_set, range_pair)) {
      GELOGE(PARAM_INVALID, "[Parse][RangePair] one of the ranges parse failed, range is: %s,"
             "there are %zu parts in the range, the invalid part of range is:[%u], value: %s.",
             shape_range.c_str(), dim_range_set.size(), shape_range_index, range_pair_str.c_str());
      return false;
    }
    shape_range_vec.emplace_back(range_pair);
    ++shape_range_index;
  }
  return true;
}

/**
 * Parser shape_range from std::string to map
 * shape_range from option normally is "input1:[1~20,3,3~6,-1];input2:[1~20,3,3~6,-1]"
 * @param shape_range
 */
Status ParseInputShapeRange(const std::string &shape_range,
                            std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> &shape_range_map) {
  GELOGD("Input shape range: %s", shape_range.c_str());

  std::vector<std::string> shape_range_vec = StringUtils::Split(shape_range, ';');
  uint32_t shape_range_index = 1UL;
  for (const auto &shape_range_item : shape_range_vec) {
    std::vector<std::string> shape_range_pair_vec = SplitInputShape(shape_range_item);
    if (shape_range_pair_vec.size() != kShapeRangeVecSize) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
          std::vector<const char *>({shape_range.c_str(), kSplitError1, kInputShapeRangeSample1}));
      GELOGE(PARAM_INVALID, "[Parse][Parameter]--input_shape_range invalid: \"%s\" , reason: %s, correct sample is %s.",
          shape_range.c_str(), kSplitError1, kInputShapeRangeSample1);
      return PARAM_INVALID;
    }
    if (shape_range_pair_vec[kShapeRangeVecValueIndex].empty()) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10048", std::vector<const char *>({"shape", "reason", "sample"}),
          std::vector<const char *>({shape_range.c_str(), kEmptyError, kInputShapeRangeSample1}));
      GELOGE(PARAM_INVALID, "[Parse][Parameter]invalid shape_range: \"%s\", reason: %s, correct sample is %s.",
          shape_range.c_str(), kEmptyError, kInputShapeRangeSample1);
      return PARAM_INVALID;
    }

    std::string shape_range_str = shape_range_pair_vec[kShapeRangeVecValueIndex];
    std::vector<std::pair<int64_t, int64_t>> shape_range_val;
    if (!ParseSingleShapeRange(shape_range_str, shape_range_val)) {
      GELOGE(PARAM_INVALID, "[Parse][Parameter] shape_range[%u], value is invalid, name: %s, value: %s.",
             shape_range_index, shape_range_pair_vec[kShapeRangeVecNameIndex].c_str(),
             shape_range_pair_vec[kShapeRangeVecValueIndex].c_str());
      return PARAM_INVALID;
    }
    shape_range_map.emplace(make_pair(StringUtils::Trim(shape_range_pair_vec[0]), shape_range_val));
    ++shape_range_index;
  }
  return SUCCESS;
}

/**
 * Parser shape_range from std::string to vector
 * shape_range from option normally is "[1~20,3,3~6,-1],[1~20,3,3~6,-1]"
 * @param shape_range
 */
Status ParseInputShapeRange(const std::string &shape_range,
                            std::vector<std::vector<std::pair<int64_t, int64_t>>> &range) {
  GELOGD("Input shape range %s", shape_range.c_str());

  if (shape_range.size() < kShapeRangeSize) {
    REPORT_PREDEFINED_ERR_MSG("E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
                       std::vector<const char *>({shape_range.c_str(), kInputShapeRangeSizeInvalid, kInputShapeRangeSample4}));
    GELOGE(PARAM_INVALID, "[Parse][ShapeRange] str:%s invalid, reason: %s, correct sample is %s.",
           shape_range.c_str(), kInputShapeRangeSizeInvalid, kInputShapeRangeSample4);
    return PARAM_INVALID;
  }
  // different shape_range of single input are split by ']'
  std::vector<std::string> shape_range_set = ge::StringUtils::Split(shape_range, ']');
  if (shape_range_set.empty()) {
    REPORT_PREDEFINED_ERR_MSG("E10048", std::vector<const char *>({"shape_range", "reason", "sample"}),
                       std::vector<const char *>({shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample4}));
    GELOGE(PARAM_INVALID, "[Parse][ShapeRange] str:%s invalid, reason: %s, correct sample is %s.",
           shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample4);
    return PARAM_INVALID;
  }
  for (auto &shape_range_str : shape_range_set) {
    if (shape_range_str.size() < kShapeRangeStrSize) {
      // shape_range_str should be "[2~3,1"
      // or ",[2~3,1". because we should trim '[' or ',['.
      // For scaler input, shape range should be "[]"
      // so shape_range_str.size() < 1 is invalid
      continue;
    }
    // trim start bytes, after that, single input should be "1~20,3,3~6,-1"
    if (ge::StringUtils::StartWith(shape_range_str, "[")) {
      shape_range_str = shape_range_str.substr(1, shape_range_str.size());
    }
    if (ge::StringUtils::StartWith(shape_range_str, ",")) {
      shape_range_str = shape_range_str.substr(kShapeRangeStrIndex, shape_range_str.size());
    }

    // parse shape_range of single input. eg. "1~20,3,3~6,-1"
    std::vector<std::pair<int64_t, int64_t>> range_of_single_input;
    std::vector<std::string> dim_range_set = ge::StringUtils::Split(shape_range_str, ',');
    for (const auto &range_pair_str : dim_range_set) {
      if (range_pair_str.empty()) {
        // for scaler input ,range is empty. use [0,0] as scaler range.
        range_of_single_input.emplace_back(std::make_pair(0, 0));
        continue;
      }
      std::vector<std::string> range_pair_set = ge::StringUtils::Split(range_pair_str, '~');
      std::pair<int64_t, int64_t> range_pair;
      if (!ParseShapeRangePair(shape_range_str, range_pair_set, range_pair)) {
        GELOGE(PARAM_INVALID, "[Parse][RangePair] Parse range pair failed.");
        return PARAM_INVALID;
      }
      range_of_single_input.emplace_back(range_pair);
    }
    range.emplace_back(range_of_single_input);
  }
  return SUCCESS;
}

Status CheckInputShapeValueValid(const std::string &input_shape_value) {
  for (auto input_char : input_shape_value) {
    if (((input_char < '0') || (input_char > '9')) &&
        (input_char != ' ') && (input_char != ',') &&
        (input_char != '~') && (input_char != '-')) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10002", std::vector<const char *>({"shape", "reason", "sample"}),
          std::vector<const char *>({input_shape_value.c_str(), kSplitError1, kInputShapeRangeSample6}));
      GELOGW("the parameter [--input_shape] Parse failed, value: \"%s\", reason: %s, correct sample is %s.",
          input_shape_value.c_str(), kSplitError1, kInputShapeRangeSample6);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status CheckHintShapeConflictWithDynamicParam(std::string &hint_shape, std::string &dynamic_batch_size,
                                              std::string &dynamic_image_size, std::string &dynamic_dims) {
  bool is_enable_dynamic_param = !dynamic_batch_size.empty() || !dynamic_image_size.empty() || !dynamic_dims.empty();
  if (!hint_shape.empty() && is_enable_dynamic_param) {
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                std::vector<const char *>({"--input_hint_shape", hint_shape.c_str(),
                                "input_hint_shape cannot be used with dynamic_batch_size, dynamic_image_size or dynamic_dims."}));
      GELOGE(PARAM_INVALID, "input_hint_shape cannot be used with dynamic_batch_size, dynamic_image_size or dynamic_dims.");
      return PARAM_INVALID;
  }
  return SUCCESS;
}

Status ParserShapeRangeByIndex(std::string &input_shape, std::string &input_shape_range) {
  input_shape_range.clear();
  std::vector<std::string> temp_input_shape_vec = StringUtils::Split(input_shape, ';');
  for (size_t i = 0UL; i < temp_input_shape_vec.size(); i++) {
    if (CheckInputShapeValueValid(temp_input_shape_vec[i]) != SUCCESS) {
      return PARAM_INVALID;
    }
    input_shape_range += "[";
    input_shape_range += temp_input_shape_vec[i];
    input_shape_range += "]";
    if (i < temp_input_shape_vec.size() - 1UL) {
      input_shape_range += ",";
    }
  }
  input_shape.clear();
  return SUCCESS;
}

Status ParserShapeRangeByName(std::string &input_shape, std::string &input_shape_range) {
  input_shape_range.clear();
  std::vector<std::string> temp_input_shape_vec = StringUtils::Split(input_shape, ';');
  for (size_t i = 0UL; i < temp_input_shape_vec.size(); i++) {
    std::vector<std::string> input_name_and_value = SplitInputShape(temp_input_shape_vec[i]);
    if (input_name_and_value.size() != kShapeRangeVecSize) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10002", std::vector<const char *>({"shape", "reason", "sample"}),
          std::vector<const char *>({temp_input_shape_vec[i].c_str(), kSplitError1, kInputShapeRangeSample6}));
      GELOGE(PARAM_INVALID, "[Parse][Parameter]--input_shape invalid: \"%s\" , reason: %s, correct sample is %s or %s.",
          temp_input_shape_vec[i].c_str(), kSplitError1, kInputShapeRangeSample6, kInputShapeRangeSample7);
      return PARAM_INVALID;
    }
    if (input_name_and_value[kShapeRangeVecNameIndex].empty()) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10002", std::vector<const char *>({"shape", "reason", "sample"}),
          std::vector<const char *>({temp_input_shape_vec[i].c_str(), "The input name is empty", kInputShapeRangeSample6}));
      GELOGE(PARAM_INVALID,
          "[Parse][Parameter]invalid input_shape: %s, reason: Input name is empty, correct sample is %s.",
          temp_input_shape_vec[i].c_str(), kInputShapeRangeSample6);
      return PARAM_INVALID;
    }
    if (input_name_and_value[kShapeRangeVecValueIndex].empty()) {
      GELOGI("Input: %s value is empty, shape is scalar.", input_name_and_value[kShapeRangeVecNameIndex].c_str());
      continue;
    }
    if (CheckInputShapeValueValid(input_name_and_value[kShapeRangeVecValueIndex]) != SUCCESS) {
      return PARAM_INVALID;
    }
    input_shape_range += input_name_and_value[kShapeRangeVecNameIndex];
    input_shape_range += ":";
    input_shape_range += "[";
    input_shape_range += input_name_and_value[kShapeRangeVecValueIndex];
    input_shape_range += "]";
    input_shape_range += ";";
  }
  // 去掉最后一个;
  input_shape_range = input_shape_range.substr(0, input_shape_range.size() - 1);
  input_shape.clear();
  return SUCCESS;
}

Status CheckAndTransferInputShapeToRange(std::string &input_shape, std::string &input_shape_range,
                                         std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                         std::string &dynamic_dims) {
  if (!input_shape_range.empty()) {
    SCREEN_LOG("WARNING: Option input_shape_range is deprecated and will be removed in future version,"
               "please use input_shape instead");
    if (!input_shape.empty()) {
      SCREEN_LOG("WARNING: Option input_shape cannot use with option input_shape_range,"
                 "it will be override by input_shape_range");
      input_shape.clear();
    }
    return SUCCESS;
  }

  if ((!dynamic_batch_size.empty()) || (!dynamic_image_size.empty()) || (!dynamic_dims.empty())) {
    if (input_shape.find('~') != std::string::npos) {
      REPORT_PREDEFINED_ERR_MSG("E10040", std::vector<const char *>(), std::vector<const char *>());
      GELOGE(PARAM_INVALID, "[Check][Param] --input_shape cannot have '~' and must have -1 "
          "when user set --dynamic_batch_size, --dynamic_image_size or --dynamic_dims");
      return ge::PARAM_INVALID;
    }
    GELOGI("Option dynamic_batch_size, dynamic_image_size or dynamic_dims is set,"
           "no need transfer input shape to input shape range");
    return SUCCESS;
  }

  if (input_shape.empty()) {
    GELOGI("Input shape is empty, no need turn shape range");
    return SUCCESS;
  }

  if ((input_shape.find('~') == std::string::npos) &&
      (input_shape.find("-1") == std::string::npos)) {
    GELOGI("Input shape is static shape, no need turn shape range");
    return SUCCESS;
  }
  // shape with index: -1,1~2,3,4;2,3,4~7,-1
  if (input_shape.find(":") == std::string::npos) {
    return ParserShapeRangeByIndex(input_shape, input_shape_range);
  }
  // shape with name: data:-1.1~2,3,4;data2:1,2,-1,3~4
  return ParserShapeRangeByName(input_shape, input_shape_range);
}

Status CheckDynamicInputParamValid(std::string &dynamic_batch_size, std::string &dynamic_image_size,
                                   std::string &dynamic_dims, const std::string &input_shape,
                                   const std::string &input_shape_range, const std::string &input_format,
                                   bool &is_dynamic_input) {
  int32_t param_size = static_cast<int32_t>(!dynamic_batch_size.empty()) +
      static_cast<int32_t>(!dynamic_image_size.empty()) + static_cast<int32_t>(!dynamic_dims.empty());
  // dynamicDimBatch should not use with dynamic shape
  if (param_size + static_cast<int32_t>(!input_shape_range.empty())  > 1) {
    REPORT_PREDEFINED_ERR_MSG("E10009", std::vector<const char *>(), std::vector<const char *>());
    GELOGE(ge::PARAM_INVALID, "[Parse][Parameter]These parameters are mutually exclusive: "
           "dynamic_batch_size, dynamic_image_size, dynamic_dims, input_shape_range");
    return ge::PARAM_INVALID;
  }

  if (param_size == 0) {
    if (input_shape_range.find(":") != std::string::npos) {
      std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> shape_range_map;
      if (ParseInputShapeRange(input_shape_range, shape_range_map) != SUCCESS) {
        GELOGE(ge::PARAM_INVALID, "[Parse][InputShapeRange] failed, range: %s", input_shape_range.c_str());
        return ge::PARAM_INVALID;
      }
    }
    return ge::SUCCESS;
  }
  std::map<std::string, std::vector<int64_t>> shape_map;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_shape_map;
  is_dynamic_input = true;
  if (input_shape.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10004", std::vector<const char *>({"parameter"}), std::vector<const char *>({"input_shape"}));
    GELOGE(ge::PARAM_INVALID,
           "[Check][Parameter:input_shape]The input_shape can not be empty in dynamic input size scenario.");
    return ge::PARAM_INVALID;
  }
  if (!ParseInputShape(input_shape, shape_map, user_shape_map, is_dynamic_input)) {
    GELOGE(ge::PARAM_INVALID, "[Parse][InputShape]input_shape: %s invalid.", input_shape.c_str());
    return ge::PARAM_INVALID;
  }
  if (!dynamic_batch_size.empty()) {
    if (!CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicBatchSizeInputShape] input_shape: %s invalid.", input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }
  if (!dynamic_image_size.empty()) {
    if (!CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicImagesizeInputShape] %s invalid. dynamic_image_size:%s ",
             input_shape.c_str(), dynamic_image_size.c_str());
      return ge::PARAM_INVALID;
    }
  }
  if (!dynamic_dims.empty()) {
    if (!CheckDynamicDimsInputShapeValid(shape_map, dynamic_dims)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicDimsInputShape]: %s of input shape: %s failed.", dynamic_dims.c_str(),
             input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }
  return ge::SUCCESS;
}

bool ParseInputShape(const std::string &input_shape, std::map<std::string, std::vector<int64_t>> &shape_map,
                     std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map, bool is_dynamic_input) {
  std::vector<std::string> shape_vec = StringUtils::Split(input_shape, ';');
  const int32_t DEFAULT_SHAPE_PAIR_SIZE = 2;
  for (const auto &shape : shape_vec) {
    std::vector<std::string> shape_pair_vec = SplitInputShape(shape);
    if (shape_pair_vec.size() != DEFAULT_SHAPE_PAIR_SIZE) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10002", std::vector<const char *>({"shape", "reason", "sample"}),
          std::vector<const char *>({shape.c_str(), kSplitError1, kInputShapeSample1}));
      GELOGW("the parameter [--input_shape] Parse failed, value: \"%s\", reason: %s, correct sample is %s.",
             shape.c_str(), kSplitError1, kInputShapeSample1);
      return false;
    }
    if (shape_pair_vec[1].empty()) {
      std::vector<int64_t> empty_shape_value;
      auto shape_name = StringUtils::Trim(shape_pair_vec[0]);
      shape_map.emplace(std::make_pair(shape_name, empty_shape_value));
      user_shape_map.emplace_back(std::make_pair(shape_name, empty_shape_value));
      continue;
    }

    std::vector<std::string> shape_value_strs = StringUtils::Split(shape_pair_vec[1], ',');
    std::vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      // stoul: The method may throw an exception: invalid_argument/out_of_range
      if (shape_value_str.find('.') != std::string::npos) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10002", std::vector<const char *>({"shape", "reason", "sample"}),
            std::vector<const char *>({shape.c_str(), kFloatNumError, kInputShapeSample2}));
        GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
               shape.c_str(), kFloatNumError, kInputShapeSample2);
        return false;
      }

      long left_result = 0;
      try {
        left_result = stol(StringUtils::Trim(shape_value_str));
        if (!shape_value_str.empty() && (shape_value_str.front() == '-')) {
          // The value maybe dynamic shape [-1], need substr it and verify isdigit.
          shape_value_str = shape_value_str.substr(1);
        }
        for (char c : shape_value_str) {
          if (!isdigit(c)) {
            REPORT_PREDEFINED_ERR_MSG(
                "E10002", std::vector<const char *>({"shape", "reason", "sample"}),
                std::vector<const char *>({shape.c_str(), kDigitError, kInputShapeSample2}));
            GELOGE(PARAM_INVALID, "[Check][Param]--input_shape's shape value[%s] is not digit",
                   shape_value_str.c_str());
            return false;
          }
        }
      } catch (const std::out_of_range &) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10013", std::vector<const char *>({"parameter", "value"}),
            std::vector<const char *>({"input_shape", shape_value_str.c_str()}));
        GELOGW("Input parameter[--input_shape]'s value[%s] cause out of range execption!", shape_value_str.c_str());
        return false;
      } catch (const std::invalid_argument &) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10014", std::vector<const char *>({"parameter", "value"}),
            std::vector<const char *>({"input_shape", shape_value_str.c_str()}));
        GELOGW("Input parameter[--input_shape]'s value[%s] cause invalid argument!", shape_value_str.c_str());
        return false;
      } catch (...) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10014", std::vector<const char *>({"parameter", "value"}),
            std::vector<const char *>({"input_shape", shape_value_str.c_str()}));
        GELOGW("Input parameter[--input_shape]'s value[%s] cause unkown execption!", shape_value_str.c_str());
        return false;
      }
      int64_t result = left_result;
      // - 1 is not currently supported
      if (!is_dynamic_input && result <= 0) {
        REPORT_PREDEFINED_ERR_MSG(
            "E10011", std::vector<const char *>({"shape", "result"}),
            std::vector<const char *>({shape.c_str(), std::to_string(result).c_str()}));
        GELOGW(
            "Input parameter[--input_shape]'s shape value[%s] is invalid, "
            "expect positive integer, but value is %ld.",
            shape.c_str(), result);
        return false;
      }
      shape_values.push_back(result);
    }

    shape_map.emplace(make_pair(StringUtils::Trim(shape_pair_vec[0]), shape_values));
    user_shape_map.push_back(make_pair(StringUtils::Trim(shape_pair_vec[0]), shape_values));
  }

  return true;
}

Status CheckOutputTypeParamValid(const std::string &output_type) {
  if ((!output_type.empty()) && (kOutputTypeSupportDatatype.find(output_type) == kOutputTypeSupportDatatype.end())) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"--output_type", output_type.c_str(), kOutputTypeSupport}));
    GELOGE(ge::PARAM_INVALID,
           "[Check][Param]Invalid value for --output_type[%s], %s.", output_type.c_str(), kOutputTypeSupport);
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckBufferOptimizeParamValid(const std::string &buffer_optimize) {
  if ((!buffer_optimize.empty()) &&
      (kBufferOptimizeSupportOption.find(buffer_optimize) == kBufferOptimizeSupportOption.end())) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"--buffer_optimize", buffer_optimize.c_str(), kBufferOptimizeSupport}));
    GELOGE(ge::PARAM_INVALID,
           "[Check][BufferOptimize]Invalid value for [%s], %s.", buffer_optimize.c_str(), kBufferOptimizeSupport);
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckCompressWeightParamValid(const std::string &enable_compress_weight,
                                     const std::string &compress_weight_conf) {
  if ((!compress_weight_conf.empty()) &&
      (!CheckInputPathValid(compress_weight_conf, "--compress_weight_conf"))) {
    GELOGE(ge::PARAM_INVALID, "[Check][InputPath]compress weight config file not found, file_name:%s",
           compress_weight_conf.c_str());
    return ge::PARAM_INVALID;
  }
  if ((enable_compress_weight != "") && (enable_compress_weight != "true") && (enable_compress_weight != "false")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10005", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"enable_compress_weight", enable_compress_weight.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param:enable_compress_weight]"
           "Input parameter[--enable_compress_weight]'s value:%s must be true or false.",
           enable_compress_weight.c_str());
    return ge::PARAM_INVALID;
  }

  if ((enable_compress_weight == "true") && (!compress_weight_conf.empty())) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10047", std::vector<const char *>({"parameter0", "parameter1"}),
        std::vector<const char *>({"enable_compress_weight", "compress_weight_conf"}));
    GELOGE(ge::PARAM_INVALID,
           "[Check][CompressWeight]enable_compress_weight and compress_weight_conf can not both exist!!");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckSparseParamValid(const std::string &sparsity) {
  if ((sparsity != "0") && (sparsity != "1")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10006", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"--sparsity", sparsity.c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param:sparsity]Input parameter[--sparsity]'s value:%s must be 1 or 0.",
           sparsity.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status CheckKeepTypeParamValid(const std::string &keep_dtype) {
  if ((!keep_dtype.empty()) && (!CheckInputPathValid(keep_dtype, "--keep_dtype"))) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"--keep_dtype", keep_dtype.c_str(), kKeepDtypeError}));
    GELOGE(ge::PARAM_INVALID, "[Check][InputPath::--keep_dtype] file not found, file_name:%s", keep_dtype.c_str());
    return ge::PARAM_INVALID;
  }

  return ge::SUCCESS;
}

int32_t CheckLogParamValidAndSetLogLevel(const std::string &log) {
  int32_t ret = -1;
  const char_t *npu_collect_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, npu_collect_path);
  if (npu_collect_path != nullptr && log == "null") {
    return 0;
  }

  if (log == "default") {
    ret = 0;
  } else if (log == "null") {
    ret = dlog_setlevel(-1, DLOG_NULL, 0);
  } else if (log == "debug") {
    ret = dlog_setlevel(-1, DLOG_DEBUG, 1);
  } else if (log == "info") {
    ret = dlog_setlevel(-1, DLOG_INFO, 1);
  } else if (log == "warning") {
    ret = dlog_setlevel(-1, DLOG_WARN, 1);
  } else if (log == "error") {
    ret = dlog_setlevel(-1, DLOG_ERROR, 1);
  } else {
    GELOGE(ge::PARAM_INVALID,
           "[Check][LogParam]log:%s invalid, only support debug, info, warning, error, null", log.c_str());
    return ret;
  }
  if (ret != 0) {
    GELOGE(ge::PARAM_INVALID, "[Set][LogLevel] fail, level:%s.", log.c_str());
  }
  return ret;
}

// option的格式为：0:[1, 2];1:[2, 3]
Status ParseHintInputShape(std::vector<GeShape> &option_shape) {
  std::string input_option;
  (void)ge::GetContext().GetOption(INPUT_HINT_SHAPE, input_option);
  if (input_option.empty()) {
    GELOGI("Option %s is not set, skip parse hint shape.", INPUT_HINT_SHAPE);
    return GRAPH_SUCCESS;
  }
  GELOGI("Option %s is set, value: %s.", INPUT_HINT_SHAPE, input_option.c_str());
  std::vector<std::string> input_option_strs = ge::StringUtils::Split(input_option, ';');

  std::vector<pair<int64_t, GeShape>> parse_shape;
  parse_shape.reserve(input_option_strs.size());
  std::set<int64_t> index_set;
  int64_t max_index = 0L;
  for (size_t i = 0U; i < input_option_strs.size(); i++) {
    auto &input_option_local = StringUtils::Trim(input_option_strs[i]);
    // 如果配置的input是空跳过解析
    if (input_option_local.empty()) {
      GELOGW("Options[%s] is invalid, Input[%u] is empty.", INPUT_HINT_SHAPE);
      continue;
    }
    std::vector<std::string> index_and_shape_str = ge::StringUtils::Split(input_option_local, ':');
    // 的左右两边必须是有元素的，key和value元素
    if (index_and_shape_str.size() != kLeastStrElementNum) {
      REPORT_PREDEFINED_ERR_MSG(
        "E10014", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"input_hint_shape", input_option.c_str()}));
      GELOGE(PARAM_INVALID, "Options[--input_hint_shape] is invalid, input[%u][%s] not match pattern: input_index:[n,c,h,w]",
        i, input_option_local.c_str());
      return PARAM_INVALID;
    }

    int64_t index = -1;
    if (ConvertToInt64(index_and_shape_str.front(), index) != SUCCESS ||
      index < 0 ||
      !index_set.insert(index).second) {
      REPORT_PREDEFINED_ERR_MSG(
        "E10014", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"input_hint_shape", input_option.c_str()}));
      GELOGE(PARAM_INVALID, "Option[--input_hint_shape] is invalid, input[%u][%s] check index fail.",
        i, input_option_local.c_str());
      return PARAM_INVALID;
    }
    max_index = index > max_index ? index : max_index;

    GeShape shape;
    if (ConstructShapeFromStr(StringUtils::Trim(index_and_shape_str.back()), shape) != GRAPH_SUCCESS) {
      REPORT_PREDEFINED_ERR_MSG(
        "E10014", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"input_hint_shape", input_option.c_str()}));
      GELOGE(PARAM_INVALID, "Option[--input_hint_shape] is invalid, Input[%u] parse shape[%s] failed.",
        i, input_option_local.c_str());
      return PARAM_INVALID;
    }
    parse_shape.emplace_back(std::make_pair(index, shape));
  }

  option_shape.resize(max_index + 1, GeShape(kDummyShape));
  for (const auto &shape : parse_shape) {
    option_shape[shape.first] = shape.second;
  }
  return GRAPH_SUCCESS;
}

std::string GetAutofuseFlagValue(const std::string &option) {
  // 自动融合新的环境变量
  const char_t *auto_fuse_options = nullptr;
  MM_SYS_GET_ENV(MM_ENV_AUTOFUSE_FLAGS, auto_fuse_options);
  if (auto_fuse_options == nullptr) {
    GELOGI("Env variable AUTOFUSE_FLAGS is not set.");
    return "";
  }
  std::vector<std::string> option_strs = ge::StringUtils::Split(std::string(auto_fuse_options), ';');
  for (const auto &opt : option_strs) {
    if (opt.find(option) != std::string::npos) {
      std::vector<std::string> key_and_values = ge::StringUtils::Split(std::string(opt), '=');
      constexpr const size_t options_key_and_value_num = 2UL;
      GE_ASSERT_TRUE(key_and_values.size() == options_key_and_value_num,
          "Options in env AUTOFUSE_FLAGS is invalid, which is %s", opt.c_str());
      if (key_and_values.front() == option) {
        GELOGI("Find option: %s in env AUTOFUSE_FLAGS, value: %s.",
            key_and_values.front().c_str(), key_and_values.back().c_str());
        return key_and_values.back();
      }
    }
  }
  GELOGI("Option %s is not set in env AUTOFUSE_FLAGS", option.c_str());
  return "";
}

Status CheckInsertOpConfParamValid(const std::string &insert_op_conf) {
  if ((!insert_op_conf.empty()) &&
      (!CheckInputPathValid(insert_op_conf, "--insert_op_conf"))) {
    GELOGE(ge::PARAM_INVALID, "[Check][InputPath]file not found: %s", insert_op_conf.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckDisableReuseMemoryParamValid(const std::string &disable_reuse_memory) {
  if ((disable_reuse_memory != "") && (disable_reuse_memory != "0") && (disable_reuse_memory != "1")) {
    REPORT_PREDEFINED_ERR_MSG("E10006", std::vector<const char *>({"parameter", "value"}),
                       std::vector<const char *>({"disable_reuse_memory", disable_reuse_memory.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][DisableReuseMemory]disable_reuse_memory must be 1 or 0.");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckEnableSingleStreamParamValid(const std::string &enable_single_stream) {
  if ((enable_single_stream != "") && (enable_single_stream != "true") && (enable_single_stream != "false")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10005", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"enable_single_stream", enable_single_stream.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param:--enable_single_stream] value:%s must be true or false.",
           enable_single_stream.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckExternalWeightParamValid(const std::string &enable_external_weight) {
  if ((enable_external_weight != "") && (enable_external_weight != kExternalWeightDisabled) &&
      (enable_external_weight != kExternalWeightEnabled) && (enable_external_weight != kExternalWeightCombined)) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10006", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"external_weight", enable_external_weight.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param:--external_weight] value:%s must be 0, 1 or 2.",
           enable_external_weight.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckIsWeightClipParamValid(const std::string &is_weight_clip) {
  if ((is_weight_clip != "0") && (is_weight_clip != "1")) {
    REPORT_PREDEFINED_ERR_MSG("E10006", std::vector<const char *>({"parameter", "value"}),
                       std::vector<const char *>({"is_weight_clip", is_weight_clip.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param:--is_weight_clip]is_weight_clip must be 1 or 0.");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckAcParallelEnableParamValid(const std::string &ac_parallel_enable) {
  if ((ac_parallel_enable != "") && (ac_parallel_enable != "0") && (ac_parallel_enable != "1")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10006", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"ac_parallel_enable", ac_parallel_enable.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param: ac_parallel_enable] value:%s must be 0 or 1.",
           ac_parallel_enable.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckTilingScheduleOptimizeParamValid(const std::string &tiling_schedule_optimize) {
  if ((tiling_schedule_optimize != "") && (tiling_schedule_optimize != "0") && (tiling_schedule_optimize != "1")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10006", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"tiling_schedule_optimize", tiling_schedule_optimize.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param: tiling_schedule_optimize] value:%s must be 0 or 1.",
           tiling_schedule_optimize.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckQuantDumpableParamValid(const std::string &quant_dumpable) {
  if ((quant_dumpable != "") && (quant_dumpable != "0") && (quant_dumpable != "1")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10006", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"quant_dumpable", quant_dumpable.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param: quant_dumpable] value:%s must be 0 or 1.",
           quant_dumpable.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckImplmodeParamValid(const std::string &optypelist_for_implmode, std::string &op_select_implmode) {
  // only appointed op_select_implmode, can user appoint optypelist_for_implmode
  if (optypelist_for_implmode != "" && op_select_implmode == "") {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"--op_select_implmode", op_select_implmode.c_str(),
                                  kCompressWeightError}));
    GELOGE(ge::PARAM_INVALID, "[Check][Param:--op_select_implmode]value:%s invalid, %s.",
           op_select_implmode.c_str(), kCompressWeightError);
    return ge::PARAM_INVALID;
  }
  // op_select_implmode default value is high_performance
  if (op_select_implmode == "") {
    op_select_implmode = IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT;
  } else {
    if (op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_PRECISON &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PRECISION_FOR_ALL &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PERFORMANCE_FOR_ALL &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_ALLOW_FP32 &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_ALLOW_HI_FP32) {
      REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({"--op_select_implmode", op_select_implmode.c_str(),
                                    kSelectImplmodeError}));
      GELOGE(ge::PARAM_INVALID, "[Check][Implmode]Invalid value for --op_select_implmode[%s], %s.",
             op_select_implmode.c_str(), kSelectImplmodeError);
      return ge::PARAM_INVALID;
    }
  }

  return ge::SUCCESS;
}

Status CheckPrecisionModeParamValid(const std::map<std::string, std::string> &options) {
  const std::string precision_mode = GetOptionValue(options, ge::PRECISION_MODE);
  const std::string precision_mode_v2 = GetOptionValue(options, ge::PRECISION_MODE_V2);
  GE_ASSERT_SUCCESS(CheckPrecisionModeParamValid(precision_mode));
  GE_ASSERT_SUCCESS(CheckPrecisionModeV2ParamValid(precision_mode_v2));
  GE_ASSERT_SUCCESS(CheckPrecisionModeV2Conflict(precision_mode, precision_mode_v2));
  return ge::SUCCESS;
}

Status CheckPrecisionModeParamValid(const std::string &precision_mode) {
  static const std::unordered_set<std::string> allowed_set = {
      "", "force_fp16", "force_fp32", "cube_fp16in_fp32out", "allow_mix_precision", "allow_fp32_to_fp16",
      "must_keep_origin_dtype", "allow_mix_precision_fp16", "allow_mix_precision_bf16", "allow_fp32_to_bf16"};
  static const std::string reason =
      "The current value is not within the valid range. Valid values are: " + StringSetToString(allowed_set) + ".";
  if (allowed_set.find(precision_mode) == allowed_set.end()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.precision_mode").c_str(), precision_mode.c_str(), reason.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][PrecisionMode]precision_mode is invalid, allowed value is {%s}.",
           StringSetToString(allowed_set).c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckPrecisionModeV2ParamValid(const std::string &precision_mode_v2) {
  static const std::unordered_set<std::string> allowed_set = {
    "", "fp16", "origin", "cube_fp16in_fp32out", "mixed_float16", "mixed_bfloat16",
    "cube_hif8", "mixed_hif8",
  };
  static const std::string reason =
      "The current value is not within the valid range. Valid values are: " + StringSetToString(allowed_set) + ".";
  if (allowed_set.find(precision_mode_v2) == allowed_set.end()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.precision_mode_v2").c_str(),
                                   precision_mode_v2.c_str(), reason.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][PrecisionMode]precision_mode_v2 is invalid, allowed value is {%s}.",
           StringSetToString(allowed_set).c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckPrecisionModeV2Conflict(const std::string &precision_mode, const std::string &precision_mode_v2) {
  if (precision_mode != "" && precision_mode_v2 != "") {
    REPORT_PREDEFINED_ERR_MSG("E10056", std::vector<const char *>({"parameter1", "parameter2"}),
                       std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.precision_mode").c_str(),
                                                 ge::GetContext().GetReadableName("ge.exec.precision_mode_v2").c_str()}));
    GELOGE(ge::PARAM_INVALID,
           "Cannot config both parameters \"precision_mode=%s\" and \"precision_mode_v2=%s\""
           " simultaneously.",
           precision_mode.c_str(), precision_mode_v2.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckModifyMixlistParamValid(const std::map<std::string, std::string> &options) {
  const std::string precision_mode = GetOptionValue(options, ge::PRECISION_MODE);
  const std::string precision_mode_v2 = GetOptionValue(options, ge::PRECISION_MODE_V2);
  const std::string modify_mixlist = GetOptionValue(options, ge::MODIFY_MIXLIST);
  if (CheckModifyMixlistParamValid(precision_mode, precision_mode_v2, modify_mixlist) != ge::SUCCESS) {
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckModifyMixlistParamValid(const std::string &precision_mode, const std::string &precision_mode_v2,
                                    const std::string &modify_mixlist) {
  std::string error_message = "";
  const bool check_precision_mode = (precision_mode == "allow_mix_precision") ||
                                    (precision_mode == "allow_mix_precision_fp16") ||
                                    (precision_mode == "allow_mix_precision_bf16");
  const bool check_precision_mode_v2 = (precision_mode_v2 == "mixed_float16") ||
                                       (precision_mode_v2 == "mixed_bfloat16") ||
                                       (precision_mode_v2 == "mixed_hif8");
  if (!modify_mixlist.empty() && !precision_mode.empty() && !check_precision_mode && !check_precision_mode_v2) {
    error_message = kModifyMixlistPrecisonModeError;
  }
  if (!modify_mixlist.empty() && !precision_mode_v2.empty() && !check_precision_mode && !check_precision_mode_v2) {
    error_message = kModifyMixlistPrecisonModeV2Error;
  }
  if (!modify_mixlist.empty() && precision_mode.empty() && precision_mode_v2.empty()) {
    error_message = kModifyMixlistError;
  }
  if (!error_message.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"--modify_mixlist", modify_mixlist.c_str(), error_message.c_str()}));
    GELOGE(ge::PARAM_INVALID, "[Check][ModifyMixlist] Failed, %s", error_message.c_str());
    return ge::PARAM_INVALID;
  }
  GELOGI("Option set successfully, option_key=%s, option_value=%s", ge::MODIFY_MIXLIST.c_str(), modify_mixlist.c_str());

  return ge::SUCCESS;
}

Status CheckAllowHF32ParamValid(const std::string &allow_hf32) {
  if ((!allow_hf32.empty()) && (allow_hf32 != "true") && (allow_hf32 != "false")) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10005", std::vector<const char *>({"parameter", "value"}),
        std::vector<const char *>({"allow_hf32", allow_hf32.c_str()}));
    GELOGE(PARAM_INVALID, "[Check][allow_hf32] Input parameter[--allow_hf32]'s value:%s must be true or false.",
           allow_hf32.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

void PrintOptionMap(std::map<std::string, std::string> &options, std::string tips) {
  for (auto iter = options.cbegin(); iter != options.cend(); iter++) {
    std::string key = iter->first;
    std::string option_name = iter->second;
    GELOGD("%s set successfully, option_key=%s, option_value=%s", tips.c_str(), key.c_str(), option_name.c_str());
  }
}

void EraseEndSemicolon(std::string &param) {
  if (param.empty()) {
    return;
  }
  if (param.back() == ';') {
    param.erase(param.end() - 1);
  }
}

Status UpdateDataOpShape(const OpDescPtr &op, std::map<std::string, std::vector<int64_t>> &shape_map) {
  GE_CHECK_NOTNULL(op);
  if (shape_map.empty()) {
    GELOGI("Shape map of data op [%s] is empty, no need to update.", op->GetName().c_str());
    return SUCCESS;
  }

  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_input);
  GE_CHECK_NOTNULL(tensor_output);
  std::string data_op_name = op->GetName();
  std::map<std::string, std::vector<int64_t>>::const_iterator iter = shape_map.find(data_op_name);
  if (iter != shape_map.cend()) {
    tensor_input->SetShape(ge::GeShape(iter->second));
    tensor_input->SetOriginShape(ge::GeShape(iter->second));
    tensor_output->SetShape(ge::GeShape(iter->second));
    tensor_output->SetOriginShape(ge::GeShape(iter->second));
    GELOGI("Update input [%s] shape info", data_op_name.c_str());
  } else {
    GELOGI("No need update input [%s] attr because not found from input_shape.", data_op_name.c_str());
  }

  return SUCCESS;
}

void UpdateDataOpFormat(const OpDescPtr &op, const std::string &format) {
  if (format.empty()) {
    GELOGD("OpName[%s] format is empty", op->GetName().c_str());
    return;
  }
  GELOGD("OpName[%s] format is %s", op->GetName().c_str(), format.c_str());
  auto format_type = ge::TypeUtils::DataFormatToFormat(format);
  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  tensor_input->SetFormat(format_type);
  tensor_input->SetOriginFormat(format_type);
  tensor_output->SetFormat(format_type);
  tensor_output->SetOriginFormat(format_type);
}

Status UpdateDataOpShapeRange(const OpDescPtr &op, const std::map<std::string,
                              std::vector<std::pair<int64_t, int64_t>>> &name_shape_range_map) {
  GE_CHECK_NOTNULL(op);
  if (name_shape_range_map.empty()) {
    GELOGI("Shape range name map of data op [%s] is empty.", op->GetName().c_str());
    return SUCCESS;
  }

  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_input);
  GE_CHECK_NOTNULL(tensor_output);
  std::string data_op_name = op->GetName();
  auto origin_shape = tensor_input->GetShape();
  auto iter = name_shape_range_map.find(data_op_name);
  if (iter != name_shape_range_map.end()) {
    auto cur_shape_range = iter->second;
    if (TensorUtils::CheckShapeByShapeRange(origin_shape, cur_shape_range) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Check][OpDescPtr] Check shape by shape range failed for op:%s.", data_op_name.c_str());
      return PARAM_INVALID;
    }
    std::vector<int64_t> dims;
    for (size_t idx = 0; idx < cur_shape_range.size(); ++idx) {
      auto left_range = cur_shape_range[idx].first;
      auto right_range = cur_shape_range[idx].second;
      if (left_range != right_range) {
        dims.push_back(UNKNOWN_DIM);
      } else {
        dims.push_back(left_range);
      }
    }
    origin_shape = GeShape(dims);
    tensor_input->SetShape(origin_shape);
    tensor_input->SetOriginShape(origin_shape);
    tensor_input->SetShapeRange(cur_shape_range);
    tensor_output->SetShape(origin_shape);
    tensor_output->SetOriginShape(origin_shape);
    tensor_output->SetShapeRange(cur_shape_range);
    GELOGI("Update input [%s] shape range and shape [%s] info success.",
           data_op_name.c_str(), origin_shape.ToString().c_str());
  } else {
    GELOGI("No need to update input [%s] attr because not found from input_shape_range.", data_op_name.c_str());
  }

  return SUCCESS;
}

Status UpdateDataOpShapeRange(const OpDescPtr &op,
                              const std::vector<std::vector<std::pair<int64_t, int64_t>>> &index_shape_range_map) {
  GE_CHECK_NOTNULL(op);
  if (index_shape_range_map.empty()) {
    GELOGI("Shape range index map of data op [%s] is empty.", op->GetName().c_str());
    return SUCCESS;
  }

  int64_t index = 0;
  if (!AttrUtils::GetInt(op, ATTR_NAME_INDEX, index)) {
    GELOGW("[%s] Get index from data attr failed.", op->GetName().c_str());
    return SUCCESS;
  }

  if ((index < 0) || (static_cast<size_t>(index) >= index_shape_range_map.size())) {
    std::string situation = "data op index[" + std::to_string(index) + "]";
    std::string reason = "The attribute index of the operator is less than 0 or exceeds the shape range entered by the user";
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const char *>({"situation", "reason"}),
                       std::vector<const char *>({situation.c_str(), reason.c_str()}));
    GELOGE(PARAM_INVALID, "user_input size = %zu, graph data op index = %ld.", index_shape_range_map.size(), index);
    return FAILED;
  }

  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_input);
  GE_CHECK_NOTNULL(tensor_output);
  std::string data_op_name = op->GetName();
  auto origin_shape = tensor_input->GetShape();
  auto cur_shape_range = index_shape_range_map[index];
  if (TensorUtils::CheckShapeByShapeRange(origin_shape, cur_shape_range) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][OpDescPtr] Check shape by shape range failed for op:%s.", data_op_name.c_str());
    return PARAM_INVALID;
  }
  std::vector<int64_t> dims;
  for (size_t idx = 0; idx < cur_shape_range.size(); ++idx) {
    auto left_range = cur_shape_range[idx].first;
    auto right_range = cur_shape_range[idx].second;
    if (left_range != right_range) {
      dims.push_back(UNKNOWN_DIM);
    } else {
      dims.push_back(left_range);
    }
  }
  origin_shape = GeShape(dims);
  tensor_input->SetShape(origin_shape);
  tensor_input->SetOriginShape(origin_shape);
  tensor_input->SetShapeRange(cur_shape_range);
  tensor_output->SetShape(origin_shape);
  tensor_output->SetOriginShape(origin_shape);
  tensor_output->SetShapeRange(cur_shape_range);
  GELOGI("Update input [%s] shape range and shape [%s] info success.",
         data_op_name.c_str(), origin_shape.ToString().c_str());

  return SUCCESS;
}

static Status CheckInputShapeRangeNode(const ComputeGraphPtr &compute_graph,
                                       const std::map<std::string,
                                       std::vector<std::pair<int64_t, int64_t>>> &shape_range_map) {
  for (const auto &it : shape_range_map) {
    std::string node_name = it.first;
    ge::NodePtr node = compute_graph->FindNode(node_name);
    if (node == nullptr) {
      REPORT_PREDEFINED_ERR_MSG("E10016", std::vector<const char *>({"parameter", "opname"}),
                         std::vector<const char *>({"input_shape", node_name.c_str()}));
      GELOGE(PARAM_INVALID, "[Check][InputNode]Input parameter[--input_shape]'s opname[%s] does not exist in model",
             node_name.c_str());
      return PARAM_INVALID;
    }
    if (!OpTypeUtils::IsDataNode(node->GetType())) {
      REPORT_PREDEFINED_ERR_MSG("E10017", std::vector<const char *>({"parameter", "opname"}),
                         std::vector<const char *>({"input_shape", node_name.c_str()}));
      GELOGE(PARAM_INVALID, "[Check][InputNode]Input parameter[--input_shape]'s opname[%s] is not a input opname",
             node_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status UpdateDynamicInputShapeRange(const ge::ComputeGraphPtr &compute_graph, const std::string &input_shape_range) {
  if (input_shape_range.empty()) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(compute_graph);

  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> shape_range_map;
  if (ParseInputShapeRange(input_shape_range, shape_range_map) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parse][InputShapeRange] input_shape_range:%s invalid.", input_shape_range.c_str());
    return PARAM_INVALID;
  }

  if (CheckInputShapeRangeNode(compute_graph, shape_range_map) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][InputShapeRange]check input shape range:%s failed.", input_shape_range.c_str());
    return PARAM_INVALID;
  }

  for (NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (OpTypeUtils::IsDataNode(op->GetType())) {
      if (UpdateDataOpShapeRange(op, shape_range_map) != SUCCESS) {
        GELOGE(FAILED, "[Update][InputShapeRange] fail for op:%s.", op->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

std::string SupportedHostEnvOsList(std::unordered_map<std::string, std::unordered_set<std::string>>
                                   &opp_supported_os_cpu) {
  std::string opp_supported_os;
  for (const auto &it : opp_supported_os_cpu) {
    opp_supported_os.append(it.first);
    opp_supported_os.append(" / ");
  }

  return opp_supported_os;
}

std::string SupportedHostEnvCpuList(std::unordered_set<std::string> &supported_os_cpu) {
  std::string opp_supported_cpu;
  for (const auto &it : supported_os_cpu) {
    opp_supported_cpu.append(it);
    opp_supported_cpu.append(" / ");
  }

  return opp_supported_cpu;
}

Status CheckHostEnvOsAndHostEnvCpuValid(const std::string &host_env_os, const std::string &host_env_cpu) {
  GE_RETURN_WITH_LOG_IF_TRUE(host_env_os.empty() || host_env_cpu.empty(),
      "os[%s] or cpu[%s] is empty", host_env_os.c_str(), host_env_cpu.c_str());

  std::unordered_map<std::string, std::unordered_set<std::string>> opp_supported_os_cpu;
  PluginManager::GetOppSupportedOsAndCpuType(opp_supported_os_cpu);
  auto cpu_os_iter = opp_supported_os_cpu.find(host_env_os);
  if (cpu_os_iter == opp_supported_os_cpu.end()) {
    const std::string &opp_supported_os = SupportedHostEnvOsList(opp_supported_os_cpu);
    std::stringstream reason;
    reason << "The OS " << host_env_os << " is not within the support list of {" << opp_supported_os
           << "}, and the OS type must be consistent with that of the opp.";
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"value", "parameter", "reason"}),
                       std::vector<const char *>({host_env_os.c_str(), "--host_env_os", reason.str().c_str()}));
    GELOGE(FAILED, "%s", reason.str().c_str());
    return FAILED;
  }

  if (cpu_os_iter->second.count(host_env_cpu) == 0U) {
    const std::string &opp_supported_cpu = SupportedHostEnvCpuList(cpu_os_iter->second);
    std::stringstream reason;
    reason << "The CPU " << host_env_cpu << " is not within the support list of {" << opp_supported_cpu
           << "}, and the CPU type must be consistent with that of the opp.";
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"value", "parameter", "reason"}),
                       std::vector<const char *>({host_env_cpu.c_str(), "--host_env_cpu", reason.str().c_str()}));
    GELOGE(FAILED, "%s", reason.str().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void SetDefaultHostEnvOsAndHostEnvCpu(std::string &host_env_os, std::string &host_env_cpu) {
  std::string cur_env_os;
  std::string cur_env_cpu;
  PluginManager::GetCurEnvPackageOsAndCpuType(cur_env_os, cur_env_cpu);
  if (!host_env_os.empty() || !host_env_cpu.empty()) {
    FileSaver::SetHostPlatformParamInitialized(true);
  }
  if (host_env_os.empty()) {
    host_env_os = cur_env_os;
    GELOGI("Set host env os with default:%s", host_env_os.c_str());
  }
  if (host_env_cpu.empty()) {
    host_env_cpu = cur_env_cpu;
    GELOGI("Set host env cpu with default:%s", host_env_cpu.c_str());
  }
  return;
}

Status CheckOptionValidValues(const std::map<std::string, std::string> &options, const std::string &key,
                              const std::unordered_set<std::string> &valid_values) {
  const auto iter = options.find(key);
  if (iter == options.cend()) {
    return SUCCESS;
  }
  if (valid_values.count(iter->second) == 0UL) {
    GELOGE(PARAM_INVALID, "[Check][Option]option %s=%s is invalid", key.c_str(), iter->second.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                       std::vector<const char *>({key.c_str(), iter->second.c_str(), kInValidValueRange}));
    return FAILED;
  }
  GELOGI("Get option key[%s] value[%s].", key.c_str(), iter->second.c_str());
  return SUCCESS;
}

Status CheckValidValueRange(const std::string &key, const std::string &value, const int64_t min, const int64_t max) {
  std::regex remove_brackets("-?\\b[0-9]+\\b");
  std::smatch match_result;
  GE_ASSERT_TRUE((!value.empty()) && std::regex_match(value, match_result, remove_brackets), "Invalid value[%s]",
                 value.c_str());
  std::istringstream val(value);
  int64_t value_got;
  val >> value_got;
  if ((value_got < min) || (value_got > max)) {
    std::string valid_range_msg = "The current value is not within the valid range. The valid range is [" +
                                  std::to_string(min) + "," + std::to_string(max) + "].";
    GELOGE(PARAM_INVALID, "[Check][Parameter] failed, option[%s] value[%s] is invalid, %s", key.c_str(),
           value.c_str(), valid_range_msg.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                              std::vector<const char *>({key.c_str(), value.c_str(), valid_range_msg.c_str()}));
    return FAILED;
  }
  GELOGI("Get option key[%s] value[%s].", key.c_str(), value.c_str());
  return SUCCESS;
}

Status CheckOptionValidThreshold(const std::map<std::string, std::string> &options, const std::string &str) {
  const auto iter = options.find(str);
  if (iter == options.cend()) {
    return SUCCESS;
  }

  GELOGI("[host_scheduling_max_threshold_option] max threshold:%s.", iter->second.c_str());
  GE_ASSERT_SUCCESS(CheckValidValueRange(str, iter->second, 0L, INT64_MAX));
  return SUCCESS;
}

Status CheckOutputReuseMemIndexesOption(const std::map<std::string, std::string> &options,
                                        bool &has_output_set_reuse_mem) {
  has_output_set_reuse_mem = false;
  const auto iter = options.find(OPTION_OUTPUT_REUSE_MEM_INDEXES);
  if (iter == options.cend()) {
    return SUCCESS;
  }

  GELOGI("[io_reuse_mem_option] output indexes:%s.", iter->second.c_str());
  const auto output_indexes = StringUtils::Split(iter->second, ',');
  std::set<int32_t> indexes_in_option;
  for (const auto &index : output_indexes) {
    GE_ASSERT_TRUE((StringUtils::IsSignedInt32(index)), "[Check][Option]Check option failed because option %s=%s is "
                   "invalid. Please input in decimal format separated by commas.",
                   OPTION_OUTPUT_REUSE_MEM_INDEXES, iter->second.c_str());
      const int32_t idx = std::stoi(index);
    GE_ASSERT_TRUE(((idx >= 0) && (indexes_in_option.count(idx) == 0U)), "[Check][Option]Check option failed because "
                   "option %s=%s is invalid. Output_index must be greater than or equal to 0 and cannot be repeated.",
                   OPTION_OUTPUT_REUSE_MEM_INDEXES, iter->second.c_str());
    indexes_in_option.insert(idx);
  }

  if (indexes_in_option.size() > 0U) {
    has_output_set_reuse_mem = true;
  }

  return SUCCESS;
}

Status CheckInputReuseMemIndexesOption(const std::map<std::string, std::string> &options,
                                       bool &has_input_set_reuse_mem) {
  has_input_set_reuse_mem = false;
  const auto iter = options.find(OPTION_INPUT_REUSE_MEM_INDEXES);
  if (iter == options.cend()) {
    return SUCCESS;
  }

  GELOGI("[io_reuse_mem_option] input indexes:%s.", iter->second.c_str());
  const auto input_indexes = StringUtils::Split(iter->second, ',');
  std::set<int32_t> indexes_in_option;
  for (const auto &index : input_indexes) {
    GE_ASSERT_TRUE((StringUtils::IsSignedInt32(index)), "[Check][Option]Check option failed because option %s=%s is "
                   "invalid. Please input in decimal format separated by commas.",
                   OPTION_INPUT_REUSE_MEM_INDEXES, iter->second.c_str());
    const int32_t idx = std::stoi(index);
    GE_ASSERT_TRUE(((idx >= 0) && (indexes_in_option.count(idx) == 0U)),"[Check][Option]Check option failed because "
                   "option %s=%s is invalid. Input_index must be greater than or equal to 0 and cannot be repeated.",
                   OPTION_INPUT_REUSE_MEM_INDEXES, iter->second.c_str());
    indexes_in_option.insert(idx);
  }

  if (indexes_in_option.size() > 0U) {
    has_input_set_reuse_mem = true;
  }

  return SUCCESS;
}

Status CheckOutputReuseInputMemIndexesOption(const ComputeGraphPtr &compute_graph,
                                             const std::map<std::string, std::string> &options) {
  const auto iter = options.find(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES);
  if (iter == options.cend()) {
    return SUCCESS;
  }

  const std::string &reuse_indexes_str = iter->second;
  if (reuse_indexes_str.empty()) {
    return SUCCESS;
  }

  // Get input count
  const size_t input_num = compute_graph->GetInputNodes().size();
  const auto netoutput_node = compute_graph->GetOrUpdateNetOutputNode();
  size_t output_num;
  if (netoutput_node == nullptr) {
    output_num = 0;
  } else {
    output_num = netoutput_node->GetAllInDataAnchorsSize();
  }

  // Parse using common function
  std::vector<std::pair<size_t, size_t>> io_same_addr_pairs;
  ParseOutputReuseInputMemIndexes(reuse_indexes_str, io_same_addr_pairs);

  // Validate parsed pairs (pair is <input_idx, output_idx>)
  for (const auto &pair : io_same_addr_pairs) {
    const size_t input_idx = pair.first;
    const size_t output_idx = pair.second;

    GE_ASSERT_TRUE((input_idx < input_num),
                   "[Check][Option]Check option failed because option %s=%s is invalid. "
                   "Input_index %zu out of range, model has %zu inputs (valid range: 0-%zu).",
                   ge::GetContext().GetReadableName(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES).c_str(), reuse_indexes_str.c_str(),
                   input_idx, input_num, input_num > 0U ? input_num - 1U : 0U);
    GE_ASSERT_TRUE((output_idx < output_num),
                   "[Check][Option]Check option failed because option %s=%s is invalid. "
                   "Output_index %zu out of range, model has %zu outputs (valid range: 0-%zu).",
                   ge::GetContext().GetReadableName(OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES).c_str(), reuse_indexes_str.c_str(),
                   output_idx, output_num, output_num > 0U ? output_num - 1U : 0U);
  }

  GELOGD("[Check][Option]Check output reuse input mem indexes option passed.");
  return SUCCESS;
}

Status CheckIoReuseMemIndexesOption(const ComputeGraphPtr &compute_graph,
                                    const std::map<std::string, std::string> &options) {
  bool has_input_set_reuse_mem = false;
  bool has_output_set_reuse_mem = false;
  GE_ASSERT_SUCCESS(CheckInputReuseMemIndexesOption(options, has_input_set_reuse_mem));
  GE_ASSERT_SUCCESS(CheckOutputReuseMemIndexesOption(options, has_output_set_reuse_mem));
  GE_ASSERT_SUCCESS(CheckOutputReuseInputMemIndexesOption(compute_graph, options));

  if ((!has_input_set_reuse_mem) && (!has_output_set_reuse_mem)) {
    return SUCCESS;
  }

  std::string alloc_mode;
  (void)ge::GetContext().GetOption(OPTION_GRAPH_IO_MEM_ALLOC_MODE, alloc_mode);
  if (alloc_mode == "ByGE") {
    return SUCCESS;
  }
  return SUCCESS;
}

Status CheckScreenPrinterOption(const std::map<std::string, std::string> &options) {
  const auto iter = options.find(OPTION_SCREEN_PRINT_MODE);
  if (iter == options.end()) {
    return SUCCESS;
  }
  if (kSupportedPrintMode.count(iter->second) == 0U) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({OPTION_SCREEN_PRINT_MODE, iter->second.c_str(),
                                   "This value is not supported. It only supports enable or disable."}));
    GELOGE(ge::PARAM_INVALID, "[Check][Option] option[%s] value[%s] invalid, not support.", OPTION_SCREEN_PRINT_MODE,
           iter->second.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status CheckOptimizationOptionValid(const std::map<std::string, std::string> &options) {
  const auto &registered_opt_table = OptionRegistry::GetInstance().GetRegisteredOptTable();
  for (const auto &ge_opt : options) {
    if (strcmp(ge_opt.first.c_str(), OO_LEVEL) == 0) {
      GE_ASSERT_GRAPH_SUCCESS(OptimizationOption::IsOoLevelValid(ge_opt.second));
      continue;
    }
    const auto iter = registered_opt_table.find(ge_opt.first);
    if (iter == registered_opt_table.end()) {
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(OptimizationOption::IsOptionValueValid(ge_opt.first, ge_opt.second, iter->second.checker));
  }
  return SUCCESS;
}
bool EnableSliceSchedule() {
  static const bool kSliceSheduleEnable = ((ge::GetAutofuseFlagValue(kAutoFuseEnableOption) == "true") &&
      (ge::GetAutofuseFlagValue(kSliceScheduleOption) == "true"));
  return kSliceSheduleEnable;
}
}  // namespace ge
