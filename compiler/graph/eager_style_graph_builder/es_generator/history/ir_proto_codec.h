/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_IR_PROTO_CODEC_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_IR_PROTO_CODEC_H_

#include <nlohmann/json.hpp>

#include "ge_attr_value.h"
#include "graph/op_desc.h"
#include "graph/utils/type_utils.h"
#include "history_registry_types.h"

namespace ge {

inline void to_json(nlohmann::json &j, const DataType &v) {
  j = TypeUtils::DataTypeToSerialString(v);
}
inline void from_json(const nlohmann::json &j, DataType &v) {
  v = TypeUtils::SerialStringToDataType(j.get<std::string>());
}

inline void to_json(nlohmann::json &j, const ConstGeTensorPtr &v) {
  j = v == nullptr ? nullptr : "Tensor()";
}
inline void from_json(const nlohmann::json &j, ConstGeTensorPtr &v) {
  if (j.is_string() && j.get<std::string>() == "Tensor()") {
    v = std::make_shared<GeTensor>(GeTensor());
  } else {
    v = nullptr;
  }
}

namespace es {
namespace history {
class IrProtoCodec {
 public:
  static IrOpProto FromOpDesc(const OpDescPtr &op_desc);
  static IrOpProto FromJson(const nlohmann::json &op_json);
  static nlohmann::json ToJson(const IrOpProto &op_proto);
};
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_IR_PROTO_CODEC_H_
