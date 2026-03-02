/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_file_constant_parser.h"
#include <vector>

#include "graph/ge_tensor.h"
#include "parser/common/op_parser_factory.h"
#include "parser/onnx/onnx_util.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "base/err_msg.h"

using ge::onnx::NodeProto;
using ge::onnx::TensorProto;
using domi::ONNX;
using GeShape = ge::GeShape;
using GeTensorDesc = ge::GeTensorDesc;
using namespace ge::parser;

namespace {
const char *const kAttrShape = "shape";
const char *const kAttrDataType = "dtype";
const char *const kFileConstantPath = "_file_constant_path";
const char *const kLocation = "location";
const char *const kOffset = "offset";
const char *const kLength = "length";
const char *const kFileConstant = "FileConstant";
}
namespace ge {
Status OnnxFileConstantParser::ParseParams(const Message *op_src, ge::Operator &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node = PtrToPtr<const Message, const ge::onnx::NodeProto>(op_src);
  GELOGD("Onnx op node name = %s, op type= %s, parse params", node->name().c_str(), node->op_type().c_str());

  ge::onnx::TensorProto tensor_proto;
  if (GetTensorProto(*node, tensor_proto) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "node[%s] get tensor failed", node->name().c_str());
    GELOGE(domi::PARAM_INVALID, "[Get][TensorProto] node[%s] get tensor failed", node->name().c_str());
    return FAILED;
  }
  if (ParseDataType(tensor_proto, op_def) != SUCCESS) {
    GELOGE(domi::PARAM_INVALID, "[Parse][Shape] node[%s] parse data type failed", node->name().c_str());
    return FAILED;
  }
  if (ParsePath(tensor_proto, op_def) != SUCCESS) {
    GELOGE(domi::PARAM_INVALID, "[Parse][Shape] node[%s] parse file path failed", node->name().c_str());
    return FAILED;
  }
  ParseShape(tensor_proto, op_def);
  return SUCCESS;
}

Status OnnxFileConstantParser::GetTensorProto(const ge::onnx::NodeProto &node_proto,
                                              ge::onnx::TensorProto &tensor_proto) const {
  for (const auto &it : node_proto.attribute()) {
    if (it.name() != ge::kAttrNameValue) {
      continue;
    }
    tensor_proto = it.t();
    return SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E19999", "node_proto[%s] get value failed", node_proto.name().c_str());
  GELOGE(ge::PARAM_INVALID, "[Get][TensorProto] node_proto[%s] get value failed", node_proto.name().c_str());
  return FAILED;
}

void OnnxFileConstantParser::ParseShape(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const {
  std::vector<int64_t> tmp_shape;
  for (int i = 0; i < tensor_proto.dims_size(); i++) {
    tmp_shape.push_back(tensor_proto.dims(i));
  }
  op_def.SetAttr(kAttrShape, tmp_shape);
}

Status OnnxFileConstantParser::ParseDataType(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const {
  int64_t data_type = tensor_proto.data_type();
  ge::DataType type = ge::OnnxUtil::ConvertOnnxDataType(data_type);
  if (type >= ge::DataType::DT_UNDEFINED) {
    REPORT_INNER_ERR_MSG("E19999", "tensor_proto date type %ld is undefined.", data_type);
    GELOGE(domi::PARAM_INVALID, "[Check][Param] tensor_proto date type %ld is undefined.", data_type);
    return FAILED;
  }

  op_def.SetAttr(kAttrDataType, type);
  return SUCCESS;
}

Status OnnxFileConstantParser::ParsePath(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const {
  for (int32_t i = 0; i < tensor_proto.external_data_size(); ++i) {
    const ge::onnx::StringStringEntryProto &string_proto = tensor_proto.external_data(i);
    if (SetPathAttr(string_proto, op_def) != SUCCESS) {
      GELOGE(domi::PARAM_INVALID, "external tensor proto[%s] parse attrs failed.", tensor_proto.name().c_str());
      return FAILED;
    }
  }

  std::string location;
  (void)op_def.GetAttr(kLocation, location);
  if (location.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "external tensor proto[%s] must contain location.", tensor_proto.name().c_str());
    GELOGE(domi::PARAM_INVALID, "external tensor proto[%s] must contain location.", tensor_proto.name().c_str());
    return FAILED;
  }
  GELOGD("The weight file of Op[%s] is: [%s].", tensor_proto.name().c_str(), location.c_str());
  return SUCCESS;
}

Status OnnxFileConstantParser::SetPathAttr(const ge::onnx::StringStringEntryProto &string_proto,
                                           ge::Operator &op_def) const {
  if (string_proto.key() == kLocation) {
    op_def.SetAttr(kLocation, string_proto.value());
  } else {
    int64_t value;
    try {
      value = stol(string_proto.value());
    } catch (const std::exception &e) {
      REPORT_INNER_ERR_MSG("E19999", "Convert %s to int64_t value failed:%s", string_proto.value().c_str(), e.what());
      GELOGE(domi::PARAM_INVALID, "Convert %s to int64_t value failed:%s", string_proto.value().c_str(), e.what());
      return FAILED;
    }
    if (string_proto.key() == kOffset) {
      op_def.SetAttr(kOffset, value);
    } else {
      op_def.SetAttr(kLength, value);
    }
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(ONNX, kFileConstant, OnnxFileConstantParser);
}  // namespace ge
