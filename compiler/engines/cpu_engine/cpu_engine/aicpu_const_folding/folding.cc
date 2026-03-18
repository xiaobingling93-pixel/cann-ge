/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "folding.h"

#include <vector>
#include <set>
#include "cpu_kernel_register.h"
#include "cpu_context.h"
#include "proto/aicpu/cpu_attr.pb.h"
#include "proto/aicpu/cpu_node_def.pb.h"
#include "proto/aicpu/cpu_tensor.pb.h"
#include "proto/aicpu/cpu_tensor_shape.pb.h"
#include "util/log.h"
#include "graph/types.h"

namespace {
const char* const kVtString = "VT_STRING";
const char* const kVtListString = "VT_LIST_STRING";
const char* const kVtFloat = "VT_FLOAT";
const char* const kVtListFloat = "VT_LIST_FLOAT";
const char* const kVtInt = "VT_INT";
const char* const kVtListInt = "VT_LIST_INT";
const char* const kVtListListInt = "VT_LIST_LIST_INT";
const char* const kVtBool = "VT_BOOL";
const char* const kVtListBool = "VT_LIST_BOOL";
const char* const kVtDataType = "VT_DATA_TYPE";
const char* const kVtListDataType = "VT_LIST_DATA_TYPE";
const char* const kVtTensor = "VT_TENSOR";
const char* const kVtListTensor = "VT_LIST_TENSOR";
using AttrValueMap = google::protobuf::Map<string, aicpuops::AttrValue>;

void ConvertGeToAicpuTensor(const ge::GeTensorDesc &tensor_desc,
                            const std::string &tensor_name,
                            const ge::Tensor &ge_tensor,
                            aicpuops::Tensor *aicpu_tensor) {
  aicpu_tensor->set_name(tensor_name);
  aicpu_tensor->set_tensor_type(tensor_desc.GetDataType());
  aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
  aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
  auto shape = aicpu_tensor->mutable_tensor_shape();
  if (shape != nullptr) {
    shape->clear_dim();
    std::vector<int64_t> dims = tensor_desc.GetShape().GetDims();
    for (size_t i = 0; i < dims.size(); i++) {
      aicpuops::TensorShape_Dim *aicpu_dims = shape->add_dim();
      if (aicpu_dims != nullptr) {
        aicpu_dims->set_size(dims[i]);
      }
    }
    shape->set_data_format(static_cast<ge::Format>(tensor_desc.GetFormat()));
  }
  AICPUE_LOGI("Op set tensor[%s], tensor info[type:%d, data:%p, size:%llu].",
         tensor_name.c_str(), static_cast<int>(tensor_desc.GetDataType()),
         ge_tensor.GetData(), ge_tensor.GetSize());
}

int32_t AddStringAttrToNodeDef(const ge::Operator &op, const char *name,
                               [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::string s;
  ge::graphStatus ret = op.GetAttr(name, s);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_s(s);

  AICPUE_LOGD("Finish add string attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListStringAttrToNodeDef(const ge::Operator &op, const char *name,
                                   [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<std::string> list_s;
  ge::graphStatus ret = op.GetAttr(name, list_s);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (std::string value : list_s) {
    array->add_s(value);
  }

  AICPUE_LOGD("Finish add list string attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddFloatAttrToNodeDef(const ge::Operator &op, const char *name,
                              [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  float f = 0;
  ge::graphStatus ret = op.GetAttr(name, f);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_f(f);

  AICPUE_LOGD("Finish add float attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListFloatAttrToNodeDef(const ge::Operator &op, const char *name,
                                  [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<float> list_f;
  ge::graphStatus ret = op.GetAttr(name, list_f);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (float value : list_f) {
    array->add_f(value);
  }

  AICPUE_LOGD("Finish add list float attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddBoolAttrToNodeDef(const ge::Operator &op, const char *name,
                             [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  bool b = false;
  ge::graphStatus ret = op.GetAttr(name, b);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_b(b);

  AICPUE_LOGD("Finish add bool attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListBoolAttrToNodeDef(const ge::Operator &op, const char *name,
                                 [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<bool> list_b;
  ge::graphStatus ret = op.GetAttr(name, list_b);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (bool value : list_b) {
    array->add_b(value);
  }

  AICPUE_LOGD("Finish add list bool attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddIntAttrToNodeDef(const ge::Operator &op, const char *name,
                            [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  int64_t i = 0;
  ge::graphStatus ret = op.GetAttr(name, i);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_i(i);

  AICPUE_LOGD("Finish add int attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListIntAttrToNodeDef(const ge::Operator &op, const char *name,
                                [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<int64_t> list_i;
  ge::graphStatus ret = op.GetAttr(name, list_i);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (int64_t value : list_i) {
    array->add_i(value);
  }

  AICPUE_LOGD("Finish add list int attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListListIntAttrToNodeDef(const ge::Operator &op, const char *name,
                                    [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<std::vector<int64_t>> list_i;
  ge::graphStatus ret = op.GetAttr(name, list_i);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_list_list_int();
  if (array == nullptr) {
    return -1;
  }

  array->clear_list_list_i();
  for (const std::vector<int64_t> &i : list_i) {
    const auto list_i = array->add_list_list_i();
    for (const int64_t val : i) {
      list_i->add_list_i(val);
    }
  }

  AICPUE_LOGD("Finish add list int int attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddDataTypeAttrToNodeDef(const ge::Operator &op, const char *name,
                                 [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::graphStatus ret = op.GetAttr(name, data_type);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  attr_value.set_type(data_type);

  AICPUE_LOGD("Finish add datatype attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListDataTypeAttrToNodeDef(const ge::Operator &op, const char *name,
                                     [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<ge::DataType> list_type;
  ge::graphStatus ret = op.GetAttr(name, list_type);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (ge::DataType value : list_type) {
    array->add_type(value);
  }

  AICPUE_LOGD("Finish add list datatype attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddTensorAttrToNodeDef(const ge::Operator &op, const char *name,
                               [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  ge::Tensor ge_tensor;
  ge::graphStatus ret = op.GetAttr(name, ge_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto aicpu_tensor = attr_value.mutable_tensor();
  if (aicpu_tensor == nullptr) {
    return -1;
  }

  ge::TensorDesc ge_tensor_desc = ge_tensor.GetTensorDesc();
  aicpu_tensor->set_tensor_type(ge_tensor_desc.GetDataType());
  aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
  aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
  auto shape = aicpu_tensor->mutable_tensor_shape();
  if (shape == nullptr) {
    return -1;
  }

  shape->clear_dim();
  std::vector<int64_t> dims = ge_tensor_desc.GetShape().GetDims();
  for (size_t i = 0; i < dims.size(); i++) {
    aicpuops::TensorShape_Dim *aicpu_dims = shape->add_dim();
    if (aicpu_dims == nullptr) {
      return -1; 
    }
    aicpu_dims->set_size(dims[i]);
  }

  AICPUE_LOGD("Finish add tensor attr to neod def, name[%s].", name);                           
  return 0;
}

int32_t AddListTensorAttrToNodeDef(const ge::Operator &op, const char *name,
                                   [[maybe_unused]] aicpuops::NodeDef node_def, aicpuops::AttrValue &attr_value) {
  std::vector<ge::Tensor> ge_list_tensor;
  ge::graphStatus ret = op.GetAttr(name, ge_list_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    return -1;
  }

  auto array = attr_value.mutable_array();
  if (array == nullptr) {
    return -1;
  }

  for (const ge::Tensor &ge_tensor : ge_list_tensor) {
    auto aicpu_tensor = array->add_tensor();
    if (aicpu_tensor == nullptr) {
      return -1;
    }
  
    ge::TensorDesc ge_tensor_desc = ge_tensor.GetTensorDesc();
    aicpu_tensor->set_tensor_type(ge_tensor_desc.GetDataType());
    aicpu_tensor->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(ge_tensor.GetData())));
    aicpu_tensor->set_data_size(static_cast<uint64_t>(ge_tensor.GetSize()));
    auto shape = aicpu_tensor->mutable_tensor_shape();
    if (shape == nullptr) {
      return -1;
    }

    shape->clear_dim();
    std::vector<int64_t> dims = ge_tensor_desc.GetShape().GetDims();
    for (size_t i = 0; i < dims.size(); i++) {
      aicpuops::TensorShape_Dim *aicpu_dims = shape->add_dim();
      if (aicpu_dims == nullptr) {
        return -1; 
      }
      aicpu_dims->set_size(dims[i]);
    }
  }

  AICPUE_LOGD("Finish add list tensor attr to neod def, name[%s].", name);
  return 0;
}

int32_t AddListAttrToNodeDef(const ge::Operator &op, const char *name,
                             const std::string &type,
                             aicpuops::NodeDef node_def,
                             aicpuops::AttrValue &attr_value) {
  int32_t ret = 0;
  if (type == kVtListString) {
    ret = AddListStringAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListFloat) {
    ret = AddListFloatAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListInt) {
    ret = AddListIntAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListListInt) {
    ret = AddListListIntAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListBool) {
    ret = AddListBoolAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListDataType) {
    ret = AddListDataTypeAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtListTensor) {
    ret = AddListTensorAttrToNodeDef(op, name, node_def, attr_value);
  } else {
    AICPUE_LOGW("Attr type is unsuported, name: [%s], type: [%s].",
           name, type.c_str());
  }
  return ret;
}

int32_t AddAttrToNodeDef(const ge::Operator &op, const char *name,
                         const std::string type, aicpuops::NodeDef node_def,
                         aicpuops::AttrValue &attr_value) {
  int32_t ret = 0;
  if (type.empty() || type[0] == '_') {
    return ret;
  }
  if (type == kVtString) {
    ret = AddStringAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtFloat) {
    ret = AddFloatAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtInt) {
    ret = AddIntAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtBool) {
    ret = AddBoolAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtDataType) {
    ret = AddDataTypeAttrToNodeDef(op, name, node_def, attr_value);
  } else if (type == kVtTensor) {
    ret = AddTensorAttrToNodeDef(op, name, node_def, attr_value);
  } else {
    ret = AddListAttrToNodeDef(op, name, type, node_def, attr_value);
  }
  return ret;
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) int32_t InitCpuConstantFoldingNew(
    ge::HostCpuOp *(*create_fn)()) {
  AICPUE_LOGI("Init cpu constant folding begin.");
  std::set<std::string> black_list = {"Assign", "NoOp", "TruncatedNormal"};
  std::vector<std::string> ops =
      aicpu::CpuKernelRegister::Instance().GetAllRegisteredOpTypes();
  AICPUE_LOGI("Number of registered ops: %llu", static_cast<uint64_t>(ops.size()));
  for (const std::string &op_type : ops) {
    if (black_list.find(op_type) != black_list.end()) {
      continue;
    }
    AICPUE_LOGI("Register op[%s].", op_type.c_str());
    ::ge::HostCpuOpRegistrar registrar __attribute__((unused)) =
        ::ge::HostCpuOpRegistrar(op_type.c_str(), create_fn);
  }
  return 0;
}

__attribute__((visibility("default"))) int32_t CpuConstantFoldingComputeNew(
    const ge::Operator &op, const std::map<std::string, const ge::Tensor> &inputs,
    std::map<std::string, ge::Tensor> outputs) {
  ge::AscendString op_type;
  if (op.GetOpType(op_type) != ge::GRAPH_SUCCESS) {
    return -1;
  }
  AICPUE_LOGI("Enter cpu op[%s].", op_type.GetString());
  std::string op_type_str(op_type.GetString());
  auto kernel = aicpu::CpuKernelRegister::Instance().GetCpuKernel(op_type_str);
  if (kernel == nullptr) {
    AICPUE_LOGW("Op[%s] is not registered in cpu kernels.", op_type.GetString());
    return -1;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPUE_LOGW("Op[%s] get op desc from operator failed.", op_type.GetString());
    return -1;
  }
  aicpuops::NodeDef node_def;
  node_def.set_op(op_type_str);
  uint32_t input_size = static_cast<uint32_t>(op_desc->GetAllInputsSize());
  for (uint32_t i = 0; i < input_size; ++i) {
    ge::GeTensorDescPtr input_desc = op_desc->MutableInputDesc(i);
    if (input_desc == nullptr) {
      continue;
    }
    std::string input_name = op_desc->GetInputNameByIndex(i);
    auto iter = inputs.find(input_name);
    if (iter == inputs.end()) {
      AICPUE_LOGW("Op[%s] input tensor[%s] is not found in inputs.",
             op_type.GetString(), input_name.c_str());
      return -1;
    }

    aicpuops::Tensor *input_tensor = node_def.add_inputs();
    if (input_tensor == nullptr) {
      return -1;
    }
    ConvertGeToAicpuTensor(*input_desc, input_name, iter->second, input_tensor);
  }

  uint32_t output_size = static_cast<uint32_t>(op_desc->GetOutputsSize());
  for (uint32_t i = 0; i < output_size; ++i) {
    ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(i);
    std::string output_name = op_desc->GetOutputNameByIndex(i);
    auto iter = outputs.find(output_name);
    if (iter == outputs.end()) {
      AICPUE_LOGW("Op[%s] output tensor[%s] is not found in outputs.",
             op_type.GetString(), output_name.c_str());
      return -1;
    }

    aicpuops::Tensor *output_tensor = node_def.add_outputs();
    if (output_tensor == nullptr) {
      return -1;
    }
    ConvertGeToAicpuTensor(output_desc, output_name, iter->second,
                           output_tensor);
  }

  std::map<ge::AscendString, ge::AscendString> attrs;
  if (op.GetAllAttrNamesAndTypes(attrs) != ge::GRAPH_SUCCESS) {
    return -1;
  }
  for (const auto &attr : attrs) {
    const char *name = attr.first.GetString();
    std::string type = std::string(attr.second.GetString());
    aicpuops::AttrValue attr_value;
    int32_t ret = AddAttrToNodeDef(op, name, type, node_def, attr_value);
    if (ret != 0) {
      return ret;
    }

    auto node_def_attrs = node_def.mutable_attrs();
    if (node_def_attrs == nullptr) {
      return -1;    
    }

    auto pair = node_def_attrs->insert(AttrValueMap::value_type(std::string(name), attr_value));
    if (!pair.second) {
      return -1;    
    }
  }

  aicpu::CpuKernelContext ctx(aicpu::HOST);
  int32_t ret = ctx.Init(&node_def);
  if (ret != 0) {
    return -1;
  }

  ret = aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx);
  if (ret != 0) {
    return -1;
  }

  AICPUE_LOGI("Finish cpu op[%s].", op_type.GetString());
  return 0;
}
}
