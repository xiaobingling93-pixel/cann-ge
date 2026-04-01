/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_AUTO_MAPPING_UTIL_H_
#define COMMON_AUTO_MAPPING_UTIL_H_

#include <vector>
#include "framework/common/debug/ge_log.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "register/tensor_assign.h"

namespace ge {

class AutoMappingUtil {
 public:
  static bool FindAttrValue(const domi::tensorflow::NodeDef *const nodeDef, const string &attr_name,
                            domi::tensorflow::AttrValue &attr_value);
  static void ConvertShape(const domi::tensorflow::TensorShapeProto &shape, std::vector<int64_t> &shape_dims);
  static graphStatus ConvertTensor(const domi::tensorflow::TensorProto &tensor, ge::GeTensorPtr &weight);
  static void ConvertFunc(const domi::tensorflow::NameAttrList &tf_func, ge::NamedAttrs &ge_func,
                          const int32_t recursive_depth = 0);

  static void ConvertDataTypeList(const domi::tensorflow::AttrValue_ListValue &list, std::vector<ge::DataType> &vec);
  static void ConvertShapeList(const domi::tensorflow::AttrValue_ListValue &list, std::vector<vector<int64_t>> &vec);
  static void ConvertTensorList(const domi::tensorflow::AttrValue_ListValue &list, std::vector<ge::GeTensorPtr> &vec);
  static void ConvertFuncList(const domi::tensorflow::AttrValue_ListValue &list, std::vector<ge::NamedAttrs> &vec,
                              const int32_t recursive_depth = 0);

  // Get the attribute list list of tensorflow and save it to obj according to the key
  template<typename T>
  static void ConvertList(const std::string &key, const domi::tensorflow::AttrValue &value, T &obj,
                          const int32_t recursive_depth = 0) {
    const domi::tensorflow::AttrValue_ListValue &list = value.list();
    if (list.s_size() > 0) {
      std::vector<std::string> vec;
      for (const auto &e : list.s()) {
        vec.push_back(e);
      }
      (void) ge::AttrUtils::SetListStr(obj, key, vec);
    } else if (list.i_size() > 0) {
      std::vector<int64_t> vec;
      for (const int64_t e : list.i()) {
        vec.push_back(e);
      }
      (void) ge::AttrUtils::SetListInt(obj, key, vec);
    } else if (list.f_size() > 0) {
      std::vector<float32_t> vec;
      for (const float32_t e : list.f()) {
        vec.push_back(e);
      }
      (void) ge::AttrUtils::SetListFloat(obj, key, vec);
    } else if (list.b_size() > 0) {
      std::vector<bool> vec;
      for (const bool e : list.b()) {
        vec.push_back(e);
      }
      (void) ge::AttrUtils::SetListBool(obj, key, vec);
    } else if (list.type_size() > 0) {
      std::vector<ge::DataType> vec;
      ConvertDataTypeList(list, vec);
      (void) ge::AttrUtils::SetListDataType(obj, key, vec);
    } else if (list.shape_size() > 0) {
      std::vector<std::vector<int64_t>> shape_dims_vec;
      ConvertShapeList(list, shape_dims_vec);
      (void) ge::AttrUtils::SetListListInt(obj, key, shape_dims_vec);
    } else if (list.tensor_size() > 0) {
      std::vector<ge::GeTensorPtr> vec;
      ConvertTensorList(list, vec);
      (void) ge::AttrUtils::SetListTensor(obj, key, vec);
    } else if (list.func_size() > 0) {
      std::vector<ge::NamedAttrs> vec;
      ConvertFuncList(list, vec, recursive_depth + 1);
      (void) ge::AttrUtils::SetListNamedAttrs(obj, key, vec);
    } else {
      GELOGD("The list has no value, key is %s.", key.c_str());
    }
  }

  // According to the property type of tensorflow, set it to the corresponding property of obj
  template<typename T>
  static void ConvertValue(const std::string &key, const domi::tensorflow::AttrValue &value, T &obj,
                           const int32_t recursive_depth = 0) {
    switch (value.value_case()) {
      case domi::tensorflow::AttrValue::kS:
        (void) ge::AttrUtils::SetStr(obj, key, value.s());
        break;
      case domi::tensorflow::AttrValue::kI:
        (void) ge::AttrUtils::SetInt(obj, key, static_cast<int64_t>(value.i()));
        break;
      case domi::tensorflow::AttrValue::kF:
        (void) ge::AttrUtils::SetFloat(obj, key, static_cast<float32_t>(value.f()));
        break;
      case domi::tensorflow::AttrValue::kB:
        (void) ge::AttrUtils::SetBool(obj, key, static_cast<bool>(value.b()));
        break;
      case domi::tensorflow::AttrValue::kType: {
        const ge::DataType ge_data_type =
            domi::TensorAssign::ConvertTensorflowDataType(static_cast<uint32_t>(value.type()));
        (void) ge::AttrUtils::SetDataType(obj, key, ge_data_type);
        break;
      }
      case domi::tensorflow::AttrValue::kList:
        ConvertList(key, value, obj, recursive_depth + 1);
        break;
      case domi::tensorflow::AttrValue::kShape: {
        std::vector<int64_t> shape_dims;
        ConvertShape(value.shape(), shape_dims);
        (void) ge::AttrUtils::SetListInt(obj, key, shape_dims);
        break;
      }
      case domi::tensorflow::AttrValue::kTensor: {
        ge::GeTensorPtr ge_tensor = nullptr;
        if (ConvertTensor(value.tensor(), ge_tensor) != GRAPH_SUCCESS) {
          GE_LOGE("Convert ge tensor failed, key is %s.", key.c_str());
          return;
        }
        (void) ge::AttrUtils::SetTensor(obj, key, ge_tensor);
        break;
      }
      case domi::tensorflow::AttrValue::kFunc: {
        ge::NamedAttrs func;
        ConvertFunc(value.func(), func, recursive_depth + 1);
        (void) ge::AttrUtils::SetNamedAttrs(obj, key, func);
        break;
      }
      case domi::tensorflow::AttrValue::kPlaceholder:
        (void) ge::AttrUtils::SetStr(obj, key, value.placeholder());
        break;
      case domi::tensorflow::AttrValue::VALUE_NOT_SET:
        GELOGD("the attr value of %s is not set.", key.c_str());
        break;
      default:
        GE_LOGE("the attr value type(%d) is invalid.", static_cast<int32_t>(value.value_case()));
        break;
    }
  }
};
}  // namespace ge
#endif  // COMMON_AUTO_MAPPING_UTIL_H_
