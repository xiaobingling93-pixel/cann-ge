/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir_proto_codec.h"

#include <checker.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "../utils.h"

namespace ge {
namespace es {
namespace history {

namespace {
template <typename IrType>
struct IrTypeToJsonMap {
  IrType type;
  const char *name;
};

constexpr IrTypeToJsonMap<IrInputType> kInputTypeMap[] = {
  {kIrInputRequired, "INPUT"},
  {kIrInputOptional, "OPTIONAL_INPUT"},
  {kIrInputDynamic, "DYNAMIC_INPUT"},
};

constexpr IrTypeToJsonMap<IrOutputType> kOutputTypeMap[] = {
  {kIrOutputRequired, "OUTPUT"},
  {kIrOutputDynamic, "DYNAMIC_OUTPUT"},
};

constexpr IrTypeToJsonMap<SubgraphType> kSubgraphTypeMap[] = {
  {kStatic, "STATIC"},
  {kDynamic, "DYNAMIC"},
};

template <typename IrType, size_t N>
const char *IrTypeToJson(IrType type, const IrTypeToJsonMap<IrType> (&map)[N], const std::string &error_prefix) {
  for (size_t i = 0; i < N; ++i) {
    if (map[i].type == type) {
      return map[i].name;
    }
  }
  throw std::runtime_error(error_prefix + std::to_string(static_cast<int64_t>(type)));
}

template <typename IrType, size_t N>
IrType JsonToIrType(const std::string &type_str, const IrTypeToJsonMap<IrType> (&map)[N], const std::string &error_prefix) {
  for (size_t i = 0; i < N; ++i) {
    if (type_str == map[i].name) {
      return map[i].type;
    }
  }
  throw std::runtime_error(error_prefix + type_str);
}

#define GET_ATTR_JSON_FUNC(AttrUtilType, CppType)                                              \
  static std::string Get##AttrUtilType##Json(const OpDescPtr &op_desc, const std::string &attr_name) { \
    CppType value{};                                                                           \
    if (!AttrUtils::Get##AttrUtilType(op_desc, attr_name, value)) {                            \
      throw std::runtime_error("Failed to get default value for attr: " + attr_name); \
    }                                                                                          \
    return nlohmann::json(value).dump();                                                             \
  }

GET_ATTR_JSON_FUNC(Int, int64_t);
GET_ATTR_JSON_FUNC(Float, float);
GET_ATTR_JSON_FUNC(Str, std::string);
GET_ATTR_JSON_FUNC(Bool, bool);
GET_ATTR_JSON_FUNC(DataType, ge::DataType);
GET_ATTR_JSON_FUNC(ListInt, std::vector<int64_t>);
GET_ATTR_JSON_FUNC(ListFloat, std::vector<float>);
GET_ATTR_JSON_FUNC(ListBool, std::vector<bool>);
GET_ATTR_JSON_FUNC(ListListInt, std::vector<std::vector<int64_t>>);
GET_ATTR_JSON_FUNC(ListStr, std::vector<std::string>);
GET_ATTR_JSON_FUNC(ListDataType, std::vector<ge::DataType>);
GET_ATTR_JSON_FUNC(Tensor, ge::ConstGeTensorPtr);

#undef GET_ATTR_JSON_FUNC

using AttrDefaultJsonHandler = std::string (*)(const OpDescPtr &, const std::string &);
std::string GetDefaultValueJsonLiteral(const OpDescPtr &op_desc, const IrAttrInfo &attr_info) {
  static const std::unordered_map<std::string, AttrDefaultJsonHandler> av_types_to_default = {
    {"VT_INT", GetIntJson},
    {"VT_FLOAT", GetFloatJson},
    {"VT_STRING", GetStrJson},
    {"VT_BOOL", GetBoolJson},
    {"VT_DATA_TYPE", GetDataTypeJson},
    {"VT_TENSOR", GetTensorJson},
    {"VT_LIST_INT", GetListIntJson},
    {"VT_LIST_FLOAT", GetListFloatJson},
    {"VT_LIST_BOOL", GetListBoolJson},
    {"VT_LIST_DATA_TYPE", GetListDataTypeJson},
    {"VT_LIST_LIST_INT", GetListListIntJson},
    {"VT_LIST_STRING", GetListStrJson},
  };

  const std::string &attr_name = attr_info.name;
  const std::string &av_type = attr_info.type_info.av_type;
  auto it = av_types_to_default.find(av_type);
  if (it != av_types_to_default.end()) {
    return it->second(op_desc, attr_name);
  }
  throw std::runtime_error("Failed to get default value for attr: " + attr_name + " with invalid av_type: " + av_type);
}

std::string GetRequiredStringField(const nlohmann::json &obj, const char *key, const std::string &context = "") {
  const std::string error_prefix = context.empty() ? key : context + "." + key;
  if (!obj.contains(key) || !obj.at(key).is_string()) {
    throw std::runtime_error(error_prefix + " is required and must be a string");
  }
  std::string value = obj.at(key).get<std::string>();
  if (value.empty()) {
    throw std::runtime_error(error_prefix + " cannot be empty");
  }
  return value;
}

bool GetRequiredBoolField(const nlohmann::json &obj, const char *key, const std::string &context) {
  if (!obj.contains(key) || !obj.at(key).is_boolean()) {
    throw std::runtime_error(context + "." + key + " is required and must be a boolean");
  }
  return obj.at(key).get<bool>();
}

std::vector<IrInput> ParseInputsFromJson(const nlohmann::json &op_json) {
  std::vector<IrInput> inputs;
  auto it = op_json.find("inputs");
  if (it == op_json.end()) {
    return inputs;
  }
  if (!it->is_array()) {
    throw std::runtime_error("inputs field is not an array");
  }
  for (size_t i = 0; i < it->size(); ++i) {
    const auto &input_json = (*it)[i];
    const std::string ctx = "inputs[" + std::to_string(i) + "]";
    IrInput input;
    input.name = GetRequiredStringField(input_json, "name", ctx);
    input.type = JsonToIrType(GetRequiredStringField(input_json, "type", ctx), kInputTypeMap,
                              ctx + ".type invalid: ");
    auto dtype_it = input_json.find("dtype");
    if (dtype_it != input_json.end()) {
      if (!dtype_it->is_string()) {
        throw std::runtime_error(ctx + ".dtype is not a string");
      }
      input.dtype = dtype_it->get<std::string>();
    }
    inputs.emplace_back(std::move(input));
  }
  return inputs;
}

std::vector<IrOutput> ParseOutputsFromJson(const nlohmann::json &op_json) {
  std::vector<IrOutput> outputs;
  auto it = op_json.find("outputs");
  if (it == op_json.end()) {
    return outputs;
  }
  if (!it->is_array()) {
    throw std::runtime_error("outputs field is not an array");
  }
  for (size_t i = 0; i < it->size(); ++i) {
    const auto &output_json = (*it)[i];
    const std::string ctx = "outputs[" + std::to_string(i) + "]";
    IrOutput output;
    output.name = GetRequiredStringField(output_json, "name", ctx);
    output.type = JsonToIrType(GetRequiredStringField(output_json, "type", ctx), kOutputTypeMap,
                               ctx + ".type invalid: ");
    auto dtype_it = output_json.find("dtype");
    if (dtype_it != output_json.end()) {
      if (!dtype_it->is_string()) {
        throw std::runtime_error(ctx + ".dtype is not a string");
      }
      output.dtype = dtype_it->get<std::string>();
    }
    outputs.emplace_back(std::move(output));
  }
  return outputs;
}

std::vector<IrSubgraph> ParseSubgraphsFromJson(const nlohmann::json &op_json) {
  std::vector<IrSubgraph> subgraphs;
  auto it = op_json.find("subgraphs");
  if (it == op_json.end()) {
    return subgraphs;
  }
  if (!it->is_array()) {
    throw std::runtime_error("subgraphs field is not an array");
  }
  for (size_t i = 0; i < it->size(); ++i) {
    const auto &subgraph_json = (*it)[i];
    const std::string ctx = "subgraphs[" + std::to_string(i) + "]";
    IrSubgraph subgraph;
    subgraph.name = GetRequiredStringField(subgraph_json, "name", ctx);
    subgraph.type = JsonToIrType(GetRequiredStringField(subgraph_json, "type", ctx), kSubgraphTypeMap,
                                 ctx + ".type invalid: ");
    subgraphs.emplace_back(std::move(subgraph));
  }
  return subgraphs;
}

std::vector<IrAttr> ParseAttrsFromJson(const nlohmann::json &op_json) {
  std::vector<IrAttr> attrs;
  auto it = op_json.find("attrs");
  if (it == op_json.end()) {
    return attrs;
  }
  if (!it->is_array()) {
    throw std::runtime_error("attrs field is not an array");
  }
  for (size_t i = 0; i < it->size(); ++i) {
    const auto &attr_json = (*it)[i];
    const std::string ctx = "attrs[" + std::to_string(i) + "]";
    IrAttr attr;
    attr.name = GetRequiredStringField(attr_json, "name", ctx);
    attr.av_type = GetRequiredStringField(attr_json, "type", ctx);
    attr.required = GetRequiredBoolField(attr_json, "required", ctx);
    auto default_it = attr_json.find("default_value");
    if (default_it != attr_json.end() && !default_it->is_null()) {
      if (!default_it->is_string()) {
        throw std::runtime_error(ctx + ".default_value is not a string");
      }
      attr.default_value = default_it->get<std::string>();
    }
    attrs.emplace_back(std::move(attr));
  }
  return attrs;
}
}  // namespace

IrOpProto IrProtoCodec::FromOpDesc(const OpDescPtr &op_desc) {
  GE_ASSERT_NOTNULL(op_desc);
  IrOpProto proto;
  proto.op_type = op_desc->GetType();
  std::vector<std::string> inputs_dtypes;
  GE_ASSERT_GRAPH_SUCCESS(OpDescUtils::GetIrInputDtypeSymIds(op_desc, inputs_dtypes));
  const auto &ir_inputs = op_desc->GetIrInputs();
  GE_ASSERT_EQ(inputs_dtypes.size(), ir_inputs.size());
  for (size_t i = 0U; i < ir_inputs.size(); ++i) {
    const auto &ir_input = ir_inputs[i];
    IrInput input;
    input.name = ir_input.first;
    input.type = ir_input.second;
    input.dtype = inputs_dtypes[i];
    proto.inputs.emplace_back(std::move(input));
  }

  const auto &ir_outputs = op_desc->GetIrOutputs();
  std::vector<std::string> outputs_dtypes;
  GE_ASSERT_GRAPH_SUCCESS(OpDescUtils::GetIrOutputDtypeSymIds(op_desc, outputs_dtypes));
  GE_ASSERT_EQ(outputs_dtypes.size(), ir_outputs.size());
  for (size_t i = 0U; i < ir_outputs.size(); ++i) {
    const auto &ir_output = ir_outputs[i];
    IrOutput output;
    output.name = ir_output.first;
    output.type = ir_output.second;
    output.dtype = outputs_dtypes[i];
    proto.outputs.emplace_back(std::move(output));
  }

  for (const auto &ir_subgraph : op_desc->GetOrderedSubgraphIrNames()) {
    IrSubgraph subgraph;
    subgraph.name = ir_subgraph.first;
    subgraph.type = ir_subgraph.second;
    proto.subgraphs.emplace_back(std::move(subgraph));
  }

  const auto ir_attrs = ge::es::GetAllIrAttrsNamesAndTypeInIrOrder(op_desc);
  proto.attrs.reserve(ir_attrs.size());
  for (const auto &ir_attr : ir_attrs) {
    IrAttr attr;
    attr.name = ir_attr.name;
    attr.av_type = ir_attr.type_info.ir_type;
    attr.required = ir_attr.is_required;
    if (!attr.required) {
      attr.default_value = GetDefaultValueJsonLiteral(op_desc, ir_attr);
    }
    proto.attrs.emplace_back(std::move(attr));
  }

  return proto;
}

IrOpProto IrProtoCodec::FromJson(const nlohmann::json &op_json) {
  IrOpProto proto;
  proto.op_type = GetRequiredStringField(op_json, "op_type");
  proto.inputs = ParseInputsFromJson(op_json);
  proto.outputs = ParseOutputsFromJson(op_json);
  proto.subgraphs = ParseSubgraphsFromJson(op_json);
  proto.attrs = ParseAttrsFromJson(op_json);
  return proto;
}

nlohmann::json IrProtoCodec::ToJson(const IrOpProto &op_proto) {
  nlohmann::json op_json;
  op_json["op_type"] = op_proto.op_type;

  nlohmann::json inputs_json = nlohmann::json::array();
  for (const auto &input : op_proto.inputs) {
    nlohmann::json input_json;
    input_json["name"] = input.name;
    input_json["type"] = IrTypeToJson(input.type, kInputTypeMap, "invalid input ir type: ");
    if (!input.dtype.empty()) {
      input_json["dtype"] = input.dtype;
    }
    inputs_json.emplace_back(std::move(input_json));
  }
  op_json["inputs"] = std::move(inputs_json);

  nlohmann::json outputs_json = nlohmann::json::array();
  for (const auto &output : op_proto.outputs) {
    nlohmann::json output_json;
    output_json["name"] = output.name;
    output_json["type"] = IrTypeToJson(output.type, kOutputTypeMap, "invalid output ir type: ");
    if (!output.dtype.empty()) {
      output_json["dtype"] = output.dtype;
    }
    outputs_json.emplace_back(std::move(output_json));
  }
  op_json["outputs"] = std::move(outputs_json);

  nlohmann::json subgraphs_json = nlohmann::json::array();
  for (const auto &subgraph : op_proto.subgraphs) {
    nlohmann::json subgraph_json;
    subgraph_json["name"] = subgraph.name;
    subgraph_json["type"] = IrTypeToJson(subgraph.type, kSubgraphTypeMap, "invalid subgraph ir type: ");
    subgraphs_json.emplace_back(std::move(subgraph_json));
  }
  op_json["subgraphs"] = std::move(subgraphs_json);

  nlohmann::json attrs_json = nlohmann::json::array();
  for (const auto &attr : op_proto.attrs) {
    nlohmann::json attr_json;
    attr_json["name"] = attr.name;
    attr_json["type"] = attr.av_type;
    attr_json["required"] = attr.required;
    if (!attr.default_value.empty()) {
      attr_json["default_value"] = attr.default_value;
    }
    attrs_json.emplace_back(std::move(attr_json));
  }
  op_json["attrs"] = std::move(attrs_json);

  return op_json;
}

}  // namespace history
}  // namespace es
}  // namespace ge
