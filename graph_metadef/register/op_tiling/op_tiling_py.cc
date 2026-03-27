/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_local_context.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "common/util/tiling_utils.h"
#include "platform/platform_info.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ge_common/ge_api_types.h"
#include "graph_metadef/common/ge_common/util.h"
#include "base/err_mgr.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common/checker.h"
#include "graph/utils/math_util.h"
#include "hcom/hcom_topo_info.h"
#include "register/core_num_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
__attribute__((unused)) void to_json(nlohmann::json &j, const HcomTopoInfo::TopoLevelDesc &desc) {
  j = nlohmann::json{
      {"comm_sets", desc.comm_sets},
      {"rank_size", desc.rank_size}
  };
}

__attribute__((unused)) void from_json(const nlohmann::json &j, HcomTopoInfo::TopoLevelDesc &desc) {
  (void) j.at("comm_sets").get_to(desc.comm_sets);
  (void) j.at("rank_size").get_to(desc.rank_size);
}

__attribute__((unused)) void to_json(nlohmann::json &j, const HcomTopoInfo::TopoInfo &info) {
  j = nlohmann::json{
      {"rank_size", info.rank_size},
      {"topo_level_descs", nlohmann::json::array()},
      {"local_window_size", info.local_window_size}
  };

  for (const auto &topo_level_desc : info.topo_level_descs) {
    j["topo_level_descs"].push_back(topo_level_desc);
  }
}

__attribute__((unused)) void from_json(const nlohmann::json &j, HcomTopoInfo::TopoInfo &info) {
  if (j.contains("rank_size")) {
    (void)j.at("rank_size").get_to(info.rank_size);
  }
  if (j.contains("local_window_size")) {
    (void)j.at("local_window_size").get_to(info.local_window_size);
  }
  const auto &arr = j.at("topo_level_descs");
  if (arr.size() != static_cast<size_t>(HcomTopoInfo::TopoLevel::MAX)) {
    std::ostringstream oss;
    oss << "Invalid topo_level_descs array length " << arr.size() << ", should be "
        << static_cast<size_t>(HcomTopoInfo::TopoLevel::MAX);
    throw std::out_of_range(oss.str());
  }

  for (size_t i = 0; i < static_cast<size_t>(HcomTopoInfo::TopoLevel::MAX); ++i) {
    (void) arr[i].get_to(info.topo_level_descs[i]);
  }
}
}
namespace optiling {
using ParseAttrFunc = std::function<bool(ge::OpDescPtr &, const nlohmann::json &, const std::string &)>;
using CopyConstDataFunc = std::function<bool(const nlohmann::json &, const size_t, std::unique_ptr<uint8_t[]> &)>;

class FuncTable {
public:
  FuncTable() = default;
  FuncTable &Init() {
    funcs_.resize(ge::DT_MAX, nullptr);
    return *this;
  }

  FuncTable &Insert(ge::DataType index, CopyConstDataFunc func) {
    funcs_[index] = func;
    return *this;
  }

  CopyConstDataFunc Find(ge::DataType index) const {
    return funcs_[index];
  }

private:
  std::vector<CopyConstDataFunc> funcs_;
};

namespace {
constexpr uint32_t kRightShiftBits = 4;
constexpr uint32_t kAndBits = 15;
const std::string kHexDigits = "0123456789ABCDEF";
constexpr size_t kSize = 4UL;
constexpr size_t kTilingCtxFixedInputSize = 5UL;
constexpr size_t kDeterministicOffset = 3UL;
constexpr size_t kDeterministicLevelOffset = 4UL;
const std::string kMaxTilingSize = "op_para_size";
constexpr size_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr size_t kWorkspaceHolerSize = 8UL;
const std::string kAttrGroup = "group";
const std::string kIsNullOutput = "_is_null_output";

struct ContextComponent {
  std::vector<gert::StorageShape> storage_shapes;
  std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> index_to_tensors;
  ge::OpDescPtr op_desc {nullptr};
  std::unique_ptr<uint8_t[]> tiling_data;
  std::unique_ptr<uint8_t[]> workspace_size;
  bool atomic_flag = true;
  int32_t tiling_cond = 0;
  uint32_t schedule_mode = 0;
};


bool FindImplFuncs(const ge::char_t *op_type, const gert::OpImplKernelRegistry::OpImplFunctionsV2 *&funcs) {
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) {
      GELOGW("Failed to find implfuncs in 2.0 way, registery is null. op type is %s.", op_type);
      return false;
    }
    const std::string op_type_str(op_type);
    funcs = registry->GetOpImpl(op_type_str.c_str());
    if (funcs == nullptr || funcs->tiling == nullptr || funcs->tiling_parse == nullptr) {
      std::string default_impl_str("DefaultImpl");
      funcs = registry->GetOpImpl(default_impl_str.c_str());
      if (funcs == nullptr || funcs->tiling == nullptr || funcs->tiling_parse == nullptr) {
        GELOGW("failed to find implfuncs in 2.0 way, funcs/tiling/tiling_parse is null. op type is %s.", op_type);
        return false;
      }
      GELOGD("Finding default implfuncs in 2.0 way, op type is %s.", op_type);
      return true;
    }
    GELOGD("Finding implfuncs using the 2.0 method, op type is %s.", op_type);
    return true;
}

template<typename T>
bool ParseValueNullDesc(const nlohmann::json &value_null_desc, std::vector<T> &data) {
  GE_ASSERT_TRUE(!value_null_desc.is_null(), "value_null desc is null");
  std::string null_desc = value_null_desc.get<std::string>();
  if (std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::has_quiet_NaN) {
    if (null_desc == "inf") {
      (void)data.emplace_back(std::numeric_limits<T>::infinity());
    } else if (null_desc == "-inf") {
      (void)data.emplace_back(static_cast<T>(0) - std::numeric_limits<T>::infinity());
    } else if (null_desc == "nan") {
      (void)data.emplace_back(std::numeric_limits<T>::quiet_NaN());
    } else {
      GELOGE(ge::GRAPH_PARAM_INVALID, "value_null desc: %s is not supported", null_desc.c_str());
      return false;
    }
  } else {
    GELOGE(ge::GRAPH_PARAM_INVALID, "this type doesn't support infinity and nan");
    return false;
  }
  return true;
}

bool ParseAndSetFloatAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  const auto value = attr["value"];
  std::vector<float> data;
  const auto value_null_desc = attr.find("value_null_desc");
  if (value_null_desc == attr.end()) {
    (void)data.emplace_back(value.get<float>());
  } else {
    if (value.is_null()) {
      GE_ASSERT_TRUE(ParseValueNullDesc(value_null_desc.value(), data));
    } else {
      GELOGE(ge::GRAPH_PARAM_INVALID, "value_null_desc is set, but value is not null");
      return false;
    }
  }
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<float>(data.front()));
  return true;
}

template<typename T>
bool ParseAndSetAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  const T attr_value = attr["value"].get<T>();
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<T>(attr_value));
  return true;
}

bool ParseAndSetFloatListAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  const auto value = attr["value"];
  std::vector<float> data;
  const auto value_null_desc =  attr.find("value_null_desc");
  if (value_null_desc == attr.end()) {
    data = value.get<std::vector<float>>();
  } else {
    GE_ASSERT_TRUE(value.size() == value_null_desc->size(), "value size is not equal to value_null_desc size");
    for (size_t i = 0U; i < value.size(); ++i) {
      if (value.at(i).is_null()) {
        GE_ASSERT_TRUE(ParseValueNullDesc(value_null_desc->at(i), data));
      } else {
        (void)data.emplace_back(value.at(i).get<float>());
      }
    }
  }
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<float>>(data));
  return true;
}

template<typename T>
bool ParseAndSetListAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<T> attr_value = attr["value"].get<std::vector<T>>();
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<T>>(attr_value));
  return true;
}

bool ParseAndSetListListAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<std::vector<int32_t>> attr_value_int32 = attr["value"].get<std::vector<std::vector<int32_t>>>();
  std::vector<std::vector<int64_t>> attr_value_int64;
  std::vector<int64_t> temp_int64_vec;
  for (const auto &vec_int32 : attr_value_int32) {
    for (const auto &item : vec_int32) {
      int64_t tmp = static_cast<int64_t>(item);
      (void)temp_int64_vec.emplace_back(tmp);
    }
    (void)attr_value_int64.emplace_back(temp_int64_vec);
    temp_int64_vec.clear();
  }
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(attr_value_int64));
  return true;
}

bool ParseAndSetListListInt64Attr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<std::vector<int64_t>> attr_value_int64 = attr["value"].get<std::vector<std::vector<int64_t>>>();
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(attr_value_int64));
  return true;
}

template<typename T>
bool GetConstData(const nlohmann::json &json_array, const size_t total_size,
                  std::unique_ptr<uint8_t[]> &tensor_holder) {
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  std::vector<T> value;
  const auto const_value = json_array["const_value"];
  const auto const_value_null_desc = json_array.find("const_value_null_desc");
  if (const_value_null_desc == json_array.end()) {
    value = const_value.get<std::vector<T>>();
  } else {
    GE_ASSERT_TRUE(const_value.size() == const_value_null_desc->size(),
                   "const_value size is not equal to const_value_null_desc size");
    for (size_t i = 0U; i < const_value.size(); ++i) {
      if (const_value.at(i).is_null()) {
        GE_ASSERT_TRUE(ParseValueNullDesc(const_value_null_desc->at(i), value));
      } else {
        (void)value.emplace_back(const_value.at(i).get<T>());
      }
    }
  }
  if (memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), value.data(), value.size() * sizeof(T)) !=
      EOK) {
    GELOGE(ge::FAILED, "Call memcpy failed, total value size is %zu.", value.size() * sizeof(T));
    return false;
  }
  return true;
}

bool GetConstDataWithFloat16(const nlohmann::json &json_array, const size_t total_size,
                             std::unique_ptr<uint8_t[]> &tensor_holder) {
  std::vector<float> const_value = json_array["const_value"].get<std::vector<float>>();
  std::vector<uint16_t> const_data_vec;
  for (size_t i = 0UL; i < const_value.size(); ++i) {
    uint16_t const_data_uint16 = Float32ToFloat16(const_value[i]);
    (void)const_data_vec.emplace_back(const_data_uint16);
  }
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  if (memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), const_data_vec.data(),
               const_data_vec.size() * sizeof(uint16_t)) != EOK) {
    GELOGE(ge::FAILED, "Call memcpy failed, total value size is %zu.", const_data_vec.size() * sizeof(uint16_t));
    return false;
  }
  return true;
}

bool GetConstDataWithBF16(const nlohmann::json &json_array, const size_t total_size,
                          std::unique_ptr<uint8_t[]> &tensor_holder) {
  std::vector<float> const_value = json_array["const_value"].get<std::vector<float>>();
  std::vector<uint16_t> const_data_vec;
  for (size_t i = 0UL; i < const_value.size(); ++i) {
    uint16_t const_data_uint16 = Float32ToBfloat16(const_value[i]);
    (void)const_data_vec.emplace_back(const_data_uint16);
  }
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  GE_CHK_BOOL_RET_STATUS((memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), const_data_vec.data(),
                                   const_data_vec.size() * sizeof(uint16_t)) == EOK),
                         false, "Call memcpy failed, total value size is %zu.",
                         const_data_vec.size() * sizeof(uint16_t));
  return true;
}

const std::unordered_map<std::string, ParseAttrFunc> kDtypeToAttrFunc = {
    {"bool", ParseAndSetAttr<bool>},
    {"float", ParseAndSetFloatAttr},
    {"float32", ParseAndSetFloatAttr},
    {"int", ParseAndSetAttr<int64_t>},
    {"int32", ParseAndSetAttr<int64_t>},
    {"int64", ParseAndSetAttr<int64_t>},
    {"str", ParseAndSetAttr<std::string>},
    {"list_bool", ParseAndSetListAttr<bool>},
    {"list_float", ParseAndSetFloatListAttr},
    {"list_float32", ParseAndSetFloatListAttr},
    {"list_int", ParseAndSetListAttr<int64_t>},
    {"list_int32", ParseAndSetListAttr<int64_t>},
    {"list_int64", ParseAndSetListAttr<int64_t>},
    {"list_str", ParseAndSetListAttr<std::string>},
    {"list_list_int", ParseAndSetListListAttr},
    {"list_list_int32", ParseAndSetListListAttr},
    {"list_list_int64", ParseAndSetListListInt64Attr}};

const FuncTable kFuncTable = FuncTable()
                             .Init()
                             .Insert(ge::DT_INT8, GetConstData<int8_t>)
                             .Insert(ge::DT_UINT8, GetConstData<uint8_t>)
                             .Insert(ge::DT_INT16, GetConstData<int16_t>)
                             .Insert(ge::DT_UINT16, GetConstData<uint16_t>)
                             .Insert(ge::DT_INT32, GetConstData<int32_t>)
                             .Insert(ge::DT_UINT32, GetConstData<uint32_t>)
                             .Insert(ge::DT_INT64, GetConstData<int64_t>)
                             .Insert(ge::DT_UINT64, GetConstData<uint64_t>)
                             .Insert(ge::DT_FLOAT, GetConstData<float>)
                             .Insert(ge::DT_DOUBLE, &GetConstData<double>)
                             .Insert(ge::DT_FLOAT16, &GetConstDataWithFloat16)
                             .Insert(ge::DT_BF16, &GetConstDataWithBF16)
                             .Insert(ge::DT_BOOL, &GetConstData<int8_t>);

void ParseDtype(const nlohmann::json &json, ge::GeTensorDesc &tensor_desc) {
  if (json.contains("dtype")) {
    std::string dtype_str = json["dtype"].get<std::string>();
    (void)std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    const ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor_desc.SetDataType(ge_dtype);
  }
}

void ParseStorageShape(const nlohmann::json &json, gert::StorageShape &storage_shape,
                       std::vector<gert::StorageShape> &storage_shapes) {
  if (json.contains("shape")) {
    gert::Shape shape;
    const auto dims = json["shape"].get<std::vector<int64_t>>();
    for (const int64_t &dim : dims) {
      (void)shape.AppendDim(dim);
    }
    storage_shape.MutableStorageShape() = shape;
  }
  if (json.contains("ori_shape")) {
    gert::Shape shape;
    const auto dims = json["ori_shape"].get<std::vector<int64_t>>();
    for (const int64_t dim : dims) {
      (void)shape.AppendDim(dim);
    }
    storage_shape.MutableOriginShape() = shape;
  }
  (void)storage_shapes.emplace_back(storage_shape);
}

void ParseStorageFormat(const nlohmann::json &json, ge::GeTensorDesc &tensor_desc) {
  if (json.contains("format")) {
    std::string format_str = json["format"].get<std::string>();
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    if (json.contains("sub_format")) {
      const int32_t sub_format = json["sub_format"].get<std::int32_t>();
      GELOGD("Sub format: %d, Primary format: %d", sub_format, static_cast<int32_t>(ge_format));
      ge_format = static_cast<ge::Format>(ge::GetFormatFromSub(static_cast<int32_t>(ge_format), sub_format));
    }
    tensor_desc.SetFormat(ge_format);
  }
  if (json.contains("ori_format")) {
    std::string format_str = json["ori_format"].get<std::string>();
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    const ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor_desc.SetOriginFormat(ge_format);
  }
}

ge::graphStatus ParseConstValue(const nlohmann::json &input, const gert::StorageShape &storage_shape,
                                const ge::GeTensorDesc &tensor_desc, const uint32_t index,
                                std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> &index_to_tensor) {
  if (input.contains("const_value")) {
    size_t total_size = 0UL;
    const size_t tensor_size = static_cast<size_t>(ge::GetSizeInBytes(storage_shape.GetStorageShape().GetShapeSize(),
                                                                      tensor_desc.GetDataType()));
    auto tensor_holder = gert::Tensor::CreateFollowing(tensor_desc.GetDataType(), tensor_size, total_size);
    GE_CHECK_NOTNULL(tensor_holder);

    if (tensor_size != 0UL) {
      auto func = kFuncTable.Find(tensor_desc.GetDataType());
      GE_CHECK_NOTNULL(func);
      if (!func(input, total_size, tensor_holder)) {
        GELOGE(ge::GRAPH_FAILED, "Make tensor failed.");
        return ge::GRAPH_FAILED;
      }
    }
    auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
    tensor->MutableOriginShape() = storage_shape.GetOriginShape();
    tensor->MutableStorageShape() = storage_shape.GetStorageShape();
    tensor->SetDataType(tensor_desc.GetDataType());
    tensor->SetStorageFormat(tensor_desc.GetFormat());
    tensor->SetOriginFormat(tensor_desc.GetOriginFormat());
    (void)index_to_tensor.emplace_back(index, std::move(tensor_holder));
  } else {
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(gert::Tensor)]);
    GE_ASSERT_NOTNULL(tensor_holder);
    new (tensor_holder.get()) gert::Tensor({{}, {}}, {tensor_desc.GetOriginFormat(), tensor_desc.GetFormat(), {}},
                                           gert::kOnHost, tensor_desc.GetDataType(), nullptr);
    reinterpret_cast<gert::Tensor *>(tensor_holder.get())->MutableStorageShape() = storage_shape.GetStorageShape();
    reinterpret_cast<gert::Tensor *>(tensor_holder.get())->MutableOriginShape() = storage_shape.GetOriginShape();
    (void)index_to_tensor.emplace_back(index, std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseInput(const nlohmann::json &input, const uint32_t index, const ge::IrInputType input_type,
                           ContextComponent &context_com) {
  ge::GeTensorDesc tensor_desc;
  gert::StorageShape storage_shape;
  ParseDtype(input, tensor_desc);
  ParseStorageShape(input, storage_shape, context_com.storage_shapes);
  ParseStorageFormat(input, tensor_desc);
  const auto ret = ParseConstValue(input, storage_shape, tensor_desc, index, context_com.index_to_tensors);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  if (input_type == ge::kIrInputRequired) {
    (void) context_com.op_desc->AddInputDesc(std::to_string(index), tensor_desc);
  } else if (input_type == ge::kIrInputDynamic) {
    (void) context_com.op_desc->UpdateInputDesc(index, tensor_desc);
  } else {
    GELOGE(ge::GRAPH_FAILED, "Unsupported ir type.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseInputs(const char* inputs, ContextComponent& context_com) {
  nlohmann::json desc_list;
  try {
    desc_list = nlohmann::json::parse(inputs);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", inputs);
    return ge::GRAPH_FAILED;
  }
  uint32_t index = 0U;
  uint32_t optional_index = 0U;
  for (const auto &desc : desc_list) {
    if (desc.is_array()) {
      const auto input_num = desc.size();
      (void)context_com.op_desc->AddDynamicInputDesc("dynamic_" + std::to_string(index) + "_" +
                                                     std::to_string(input_num), static_cast<uint32_t>(input_num));
      context_com.op_desc->AppendIrInput("dynamic_" + std::to_string(index) + "_" + std::to_string(input_num),
                                         ge::kIrInputDynamic);
      for (const auto &ele : desc) {
        if (ele.is_null()) {
          GELOGW("Empty input at current index %u", index);
          continue;
        }
        if (ParseInput(ele, index, ge::kIrInputDynamic, context_com) != ge::GRAPH_SUCCESS) {
          return ge::GRAPH_FAILED;
        }
        ++index;
      }
    } else {
      if (desc.is_null()) {
        context_com.op_desc->AppendIrInput("optional" + std::to_string(optional_index), ge::kIrInputOptional);
        (void)context_com.op_desc->AddOptionalInputDesc(
            "optional" + std::to_string(optional_index),
            ge::GeTensorDesc(ge::GeShape(), ge::FORMAT_RESERVED, ge::DT_UNDEFINED));
        GELOGI("Optional input at index %u is null.", optional_index);
        ++optional_index;
        continue;
      }
      context_com.op_desc->AppendIrInput(std::to_string(index), ge::kIrInputRequired);
      if (ParseInput(desc, index, ge::kIrInputRequired, context_com) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
      ++index;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcNullOutput(const size_t ir_index, const ge::IrOutputType output_type,
                                uint32_t &index, ContextComponent &context_com) {
  const gert::OpImplKernelRegistry::OpImplFunctionsV2 *funcs;
  if (FindImplFuncs(context_com.op_desc->GetType().c_str(), funcs) && funcs->IsNullableOutput(ir_index)) {
    ge::GeTensorDesc tensor_desc;
    gert::StorageShape storage_shape;
    context_com.storage_shapes.emplace_back(storage_shape);
    (void)ge::AttrUtils::SetBool(tensor_desc, kIsNullOutput, true);
    if (output_type == ge::kIrOutputRequired) {
      context_com.op_desc->AppendIrOutput(std::to_string(index), ge::kIrOutputRequired);
      (void) context_com.op_desc->AddOutputDesc(std::to_string(index), tensor_desc);
    } else {
      GELOGE(ge::GRAPH_FAILED, "Unsupported ir type.");
      return ge::GRAPH_FAILED;
    }

    GELOGI("Set null output");
    index++;
  } else {
    GELOGW("Empty output, cur index %u, ir index %zu", index, ir_index);
  }
  return ge::GRAPH_SUCCESS;
}

void ParseIsNullableOutputExist(const nlohmann::json &json, ge::GeTensorDesc &tensor_desc) {
  if (json.contains("is_null_output")) {
    bool is_null_output = json["is_null_output"].get<bool>();
    (void)ge::AttrUtils::SetBool(tensor_desc, kIsNullOutput, is_null_output);
    GELOGI("Set null output");
  }
}

ge::graphStatus ParseOutput(const nlohmann::json &output, ge::IrOutputType output_type, const uint32_t index,
                            ContextComponent &context_com) {
  ge::GeTensorDesc tensor_desc;
  gert::StorageShape storage_shape;
  ParseDtype(output, tensor_desc);
  ParseStorageShape(output, storage_shape, context_com.storage_shapes);
  ParseStorageFormat(output, tensor_desc);
  ParseIsNullableOutputExist(output, tensor_desc);

  if (output_type == ge::kIrOutputRequired) {
    (void) context_com.op_desc->AddOutputDesc(std::to_string(index), tensor_desc);
  } else if (output_type == ge::kIrOutputDynamic) {
    (void) context_com.op_desc->UpdateOutputDesc(index, tensor_desc);
  } else {
    GELOGE(ge::GRAPH_FAILED, "Unsupported ir type.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

void ParseTopoInfo(const ascend_nlohmann::json &extra_info, ge::OpDescPtr &op_desc, const std::string &group_attr) {
  std::string group;
  if (!ge::AttrUtils::GetStr(op_desc, group_attr, group) || group.empty()) {
    GELOGW("hcom_topo_info need bind valid %s, which is needed to set on op %s %s", group_attr.c_str(),
           op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return;
  }
  if (ge::HcomTopoInfo::Instance().TopoInfoHasBeenSet(group.c_str())) {
    return;
  }
  ge::HcomTopoInfo::TopoInfo topo_info_parsed{};
  // 兼容老的场景
  if (extra_info.contains("rank_size")) {
    topo_info_parsed.rank_size = extra_info["rank_size"];
    GELOGD("Extra info contains rank size, rank size is %ld", topo_info_parsed.rank_size);
  } else {
    try {
      if (extra_info.contains("hcom_topo_info")) {
        const auto &json_hcom_topo_info = extra_info["hcom_topo_info"];
        GELOGD("Extra info contains topo info [%s]", json_hcom_topo_info.dump().c_str());
        (void)json_hcom_topo_info.get_to(topo_info_parsed);
      } else if (extra_info.contains(group_attr.c_str())) {
        const auto &json_hcom_topo_info = extra_info[group_attr.c_str()];
        GELOGD("Extra info [%s] contains topo for group[%s]", json_hcom_topo_info.dump().c_str(), group_attr.c_str());
        (void)json_hcom_topo_info.get_to(topo_info_parsed);
      } else {
        return;
      }
    } catch (const std::exception &e) {
      GELOGE(ge::GRAPH_FAILED, "Parse error %s", e.what());
      return;
    }
  }
  (void)ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topo_info_parsed);
  GELOGD("Set topo info for group %s successfully", group.c_str());
}

void ParseExtraInfo(const nlohmann::json &extra_info, ge::OpDescPtr &op_desc) {
  if (extra_info.contains("op_name")) {
    const std::string name = extra_info["op_name"];
    op_desc->SetName(name);
  }
  if (extra_info.contains("deterministic")) {
    const int32_t deterministic = extra_info["deterministic"];
    (void)ge::AttrUtils::SetInt(op_desc, "deterministic", deterministic);
  }
  if (extra_info.contains("deterministic_level")) {
    const int32_t deterministic_level = extra_info["deterministic_level"];
    (void)ge::AttrUtils::SetInt(op_desc, "deterministic_level", deterministic_level);
  }
  if (extra_info.contains(ge::public_attr::OP_AI_CORE_NUM)) {
    const std::string op_aicore_num = extra_info[ge::public_attr::OP_AI_CORE_NUM];
    GELOGI("Set op_aicore_num from extra info: %s", op_aicore_num.c_str());
    (void)ge::AttrUtils::SetStr(op_desc, ge::public_attr::OP_AI_CORE_NUM, op_aicore_num);
  }
  if (extra_info.contains(ge::public_attr::OP_VECTOR_CORE_NUM)) {
    const std::string op_vectorcore_num = extra_info[ge::public_attr::OP_VECTOR_CORE_NUM];
    GELOGI("Set op_vectorcore_num from extra info: %s", op_vectorcore_num.c_str());
    (void)ge::AttrUtils::SetStr(op_desc, ge::public_attr::OP_VECTOR_CORE_NUM, op_vectorcore_num);
  }
  for (const auto &group_attr: op_desc->GetAllAttrNames()) {
    if (group_attr.compare(0, kAttrGroup.length(), kAttrGroup) == 0) {
      ParseTopoInfo(extra_info, op_desc, group_attr);
    }
  }
}

ge::graphStatus ParseExtraInfos(const char *const extra_info, ge::OpDescPtr &op_desc) {
  if (extra_info == nullptr) {
    GELOGI("Extra info is nullptr.");
    return ge::GRAPH_SUCCESS;
  }
  nlohmann::json desc;
  try {
    desc = nlohmann::json::parse(extra_info);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", extra_info);
    return ge::GRAPH_FAILED;
  }
  if (desc.is_array()) {
    for (const auto &ele : desc) {
      ParseExtraInfo(ele, op_desc);
    }
  } else {
    ParseExtraInfo(desc, op_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseOutputs(const char *outputs, ContextComponent &context_com) {
  nlohmann::json desc_list;
  try {
    desc_list = nlohmann::json::parse(outputs);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", outputs);
    return ge::GRAPH_FAILED;
  }
  uint32_t index = 0;
  for (size_t ir_index = 0U; ir_index < desc_list.size(); ir_index++) {
    const auto &desc = desc_list[ir_index];
    if (desc.is_array()) {
      const size_t output_num = desc.size();
      //  可能传过来的输入没有指定名字,所有用"dynamic_"+"index"+"_"+"num"拼一个统一的假名字,输出也是相同的处理
      (void)context_com.op_desc->AddDynamicOutputDesc("dynamic_" + std::to_string(index) + "_" +
                                                      std::to_string(output_num), static_cast<uint32_t>(output_num));
      context_com.op_desc->AppendIrOutput("dynamic_" + std::to_string(index) + "_" + std::to_string(output_num),
                                          ge::kIrOutputDynamic);
      for (const auto &ele : desc) {
        if (ele.is_null()) {
          GELOGW("Empty output, cur index %u", index);
          continue;
        }
        GE_ASSERT_GRAPH_SUCCESS(ParseOutput(ele, ge::kIrOutputDynamic, index, context_com));
        ++index;
      }
    } else {
      if (desc.is_null()) {
        GE_ASSERT_GRAPH_SUCCESS(ProcNullOutput(ir_index, ge::kIrOutputRequired, index, context_com));
        continue;
      }
      context_com.op_desc->AppendIrOutput(std::to_string(index), ge::kIrOutputRequired);
      GE_ASSERT_GRAPH_SUCCESS(ParseOutput(desc, ge::kIrOutputRequired, index, context_com));
      ++index;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseAttrs(const char *attrs, ge::OpDescPtr &op_desc) {
  if (attrs == nullptr) {
    GELOGD("Attribute has not been set.");
  } else {
    nlohmann::json attrs_json;
    try {
      attrs_json = nlohmann::json::parse(attrs);
    } catch (const nlohmann::json::exception &e) {
      GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", attrs);
      return ge::GRAPH_FAILED;
    }
    for (const auto &attr : attrs_json) {
      if (!attr.contains("name") || !attr.contains("dtype") || !attr.contains("value")) {
        GELOGE(ge::GRAPH_FAILED, "cur attr does not contain name or dtype or value.");
        return ge::GRAPH_FAILED;
      }
      const std::string attr_name = attr["name"].get<std::string>();
      const std::string dtype = attr["dtype"].get<std::string>();
      const auto iter = kDtypeToAttrFunc.find(dtype);
      if (iter == kDtypeToAttrFunc.end()) {
        GELOGE(ge::GRAPH_FAILED, "Unknown dtype[%s], which is unsupported.", dtype.c_str());
        return ge::GRAPH_FAILED;
      }
      GE_ASSERT_TRUE((iter->second)(op_desc, attr, attr_name));
      GELOGD("Finished setting attribute [name: %s] for op.", attr_name.c_str());
    }
  }
  return ge::GRAPH_SUCCESS;
}

std::string DumpTilingData(gert::TilingData *tiling_data) {
  std::string output;
  if (tiling_data == nullptr) {
    return output;
  }
  if (tiling_data->GetDataSize() >= std::numeric_limits<size_t>::max() / kSize) {
    GELOGE(ge::GRAPH_FAILED, "Tiling data size overflow.");
    return output;
  }
  output.reserve(tiling_data->GetDataSize() * kSize);
  char *data = reinterpret_cast<char *>(tiling_data->GetData());
  for (size_t i = 0UL; i < tiling_data->GetDataSize(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(data[i]);
    output.push_back(kHexDigits[ch >> kRightShiftBits]);
    output.push_back(kHexDigits[ch & kAndBits]);
  }
  return output;
}

bool DumpRunInfo(gert::KernelContext *kernel_context, char *run_info_json, const size_t run_info_len) {
  GE_ASSERT_NOTNULL(run_info_json);
  nlohmann::json json_obj;
  auto ws = kernel_context->GetOutputPointer<gert::ContinuousVector>(gert::TilingContext::kOutputWorkspace);
  GE_ASSERT_NOTNULL(ws);
  std::vector<size_t> workspaces(reinterpret_cast<const size_t *>(ws->GetData()),
                                 reinterpret_cast<const size_t *>(ws->GetData()) + ws->GetSize());
  GE_ASSERT_NOTNULL(kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputBlockDim));
  GE_ASSERT_NOTNULL(kernel_context->GetOutputPointer<bool>(gert::TilingContext::kOutputAtomicCleanFlag));
  GE_ASSERT_NOTNULL(kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputTilingKey));
  GE_ASSERT_NOTNULL(kernel_context->GetOutputPointer<int32_t>(gert::TilingContext::kOutputTilingCond));
  GE_ASSERT_NOTNULL(kernel_context->GetOutputPointer<uint32_t>(gert::TilingContext::kOutputScheduleMode));
  json_obj["block_dim"] = *kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputBlockDim);
  json_obj["workspaces"] = workspaces;
  json_obj["tiling_data"] =
      DumpTilingData(kernel_context->GetOutputPointer<gert::TilingData>(gert::TilingContext::kOutputTilingData));
  json_obj["clear_atomic"] = *kernel_context->GetOutputPointer<bool>(gert::TilingContext::kOutputAtomicCleanFlag);
  json_obj["tiling_key"] = *kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputTilingKey);
  json_obj["tiling_cond"] = *kernel_context->GetOutputPointer<int32_t>(gert::TilingContext::kOutputTilingCond);
  json_obj["schedule_mode"] = *kernel_context->GetOutputPointer<uint32_t>(gert::TilingContext::kOutputScheduleMode);

  const auto local_mem_size = kernel_context->GetOutputPointer<uint32_t>(gert::TilingContext::kOutputLocalMemorySize);
  if (local_mem_size != nullptr) {
    json_obj["local_memory_size"] = *local_mem_size;
  }

  const auto aicpu_block_dim = kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputAicpuBlockDim);
  GE_ASSERT_NOTNULL(aicpu_block_dim);
  json_obj["aicpu_block_dim"] = *aicpu_block_dim;

  const std::string str = json_obj.dump();
  return memcpy_s(run_info_json, run_info_len, str.c_str(), str.size() + 1UL) == EOK;
}
}  // namespace

using ParseAndSetAttrValueFunc = std::function<void(ge::Operator &, const nlohmann::json &, const std::string &)>;
using ParseAndSetAttrValuePtr = std::shared_ptr<ParseAndSetAttrValueFunc>;

thread_local int64_t last_op_tiling_perf = -1;

template<typename T>
void ParseAndSetAttrValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  const T attr_value = attr["value"].get<T>();
  (void)op.SetAttr(attr_name.c_str(), attr_value);
}

template<typename T>
void ParseAndSetAttrListValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<T> attr_value = attr["value"].get<std::vector<T>>();
  (void)op.SetAttr(attr_name.c_str(), attr_value);
}
namespace {
thread_local std::string error_string;
constexpr int64_t ret_success = 0;
constexpr int64_t ret_fail = 1;
constexpr int64_t outter_error_type = 1;
constexpr int64_t inner_error_type = 2;

std::map<std::string, std::string> AssembleMap(const std::vector<error_message::unique_const_char_array>& args_key,
    const std::vector<error_message::unique_const_char_array>& args_value){
  std::map<std::string, std::string> result_map;

  // 第一步：校验两个向量长度一致，避免下标越界
  if (args_key.size() != args_value.size()) {
    GELOGE(ge::GRAPH_FAILED, "key length:%d is inconsistent with value length:%d", args_key.size(), args_value.size());
    return result_map; // 长度不一致返回空map
  }

  for (size_t i = 0; i < args_key.size(); ++i) {
    const char* key_ptr = args_key[i] ? args_key[i].get() : "";
    std::string key_str = key_ptr;

    const char* val_ptr = args_value[i] ? args_value[i].get() : "";
    std::string val_str = val_ptr;

    result_map[key_str] = val_str;
  }

  return result_map;
}

std::string GetRawErrorMessage() {
  try {
    nlohmann::json ret_json;
    const auto &error_messages = error_message::GetErrMgrRawErrorMessages();
    if (error_messages.empty()) {
      ret_json["ret_code"] = ret_success;
      return ret_json.dump();
    }
    ret_json["ret_code"] = ret_fail;
    nlohmann::json error_messages_json = {};
    for (const auto &item : error_messages) {
      nlohmann::json item_json;
      item_json["errorcode"] = item.error_id ? std::string(item.error_id.get()) : std::string("");
      if (item.args_key.empty() || item.args_value.empty()) {
        item_json["type"] = inner_error_type;
        item_json["errormsg"] = item.error_message ? std::string(item.error_message.get()) : std::string("");
      } else {
        item_json["type"] = outter_error_type;
        item_json["errormsg"] = AssembleMap(item.args_key, item.args_value);
      }
      error_messages_json.push_back(item_json);
    }
    ret_json["error_messages"] = error_messages_json;
    return ret_json.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "get failed when call json api, reason: %s", e.what());
    return "";
  }
}

void ParseAndSetAttrListListValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<std::vector<int32_t>> attr_value_int32 = attr["value"].get<std::vector<std::vector<int32_t>>>();
  std::vector<std::vector<int64_t>> attr_value_int64;
  std::vector<int64_t> temp_int64_vec;
  for (const auto &vec_int32 : attr_value_int32) {
    for (const auto &item : vec_int32) {
      int64_t tmp = static_cast<int64_t>(item);
      (void)temp_int64_vec.emplace_back(tmp);
    }
    (void)attr_value_int64.emplace_back(temp_int64_vec);
    temp_int64_vec.clear();
  }

  (void)op.SetAttr(attr_name.c_str(), attr_value_int64);
}

void ParseAndSetAttrListListInt64Value(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<std::vector<int64_t>> attr_value_int64 = attr["value"].get<std::vector<std::vector<int64_t>>>();
  (void)op.SetAttr(attr_name.c_str(), attr_value_int64);
}

const std::map<std::string, ParseAndSetAttrValuePtr> parse_attr_dtype_map = {
    {"bool", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<bool>)},
    {"float", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<float>)},
    {"float32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<float>)},
    {"int", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int32_t>)},
    {"int32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int32_t>)},
    {"int64", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int64_t>)},
    {"str", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<std::string>)},
    {"list_bool", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<bool>)},
    {"list_float", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<float>)},
    {"list_float32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<float>)},
    {"list_int", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int32_t>)},
    {"list_int32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int32_t>)},
    {"list_int64", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int64_t>)},
    {"list_str", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<std::string>)},
    {"list_list_int", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListValue)},
    {"list_list_int32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListValue)},
    {"list_list_int64", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListInt64Value)}};

void ParseShapeDesc(const nlohmann::json &shape, std::vector<TeOpTensor> &tensors) {
  TeOpTensor tensor;
  if (shape.contains("shape")) {
    tensor.shape = shape["shape"].get<std::vector< int64_t>>();
  }
  if (shape.contains("ori_shape")) {
    tensor.ori_shape = shape["ori_shape"].get<std::vector<int64_t>>();
  }
  if (shape.contains("format")) {
    tensor.format = shape["format"].get<std::string>();
  }
  if (shape.contains("ori_format")) {
    tensor.ori_format = shape["ori_format"].get<std::string>();
  }
  if (shape.contains("dtype")) {
    tensor.dtype = shape["dtype"].get<std::string>();
  }
  (void)tensors.emplace_back(tensor);
}

void ParseShapeDescList(const nlohmann::json &shape_list, std::vector<TeOpTensorArg> &op_args) {
  for (const auto &elem : shape_list) {
    TeOpTensorArg tensor_arg;
    tensor_arg.arg_type = TensorArgType::TA_NONE;

    if (elem.is_array()) {
      tensor_arg.arg_type = TensorArgType::TA_LIST;
      for (const auto &shape : elem) {
        ParseShapeDesc(shape, tensor_arg.tensor);
      }
    } else {
      tensor_arg.arg_type = TensorArgType::TA_SINGLE;
      ParseShapeDesc(elem, tensor_arg.tensor);
    }
    (void)op_args.emplace_back(tensor_arg);
  }
}

void ParseShapeDescV2(const nlohmann::json &shape, ge::OpDescPtr &op_desc, const bool &is_input) {
  ge::GeTensorDesc tensor;
  std::string name;
  if (shape.contains("shape")) {
    tensor.SetShape(ge::GeShape(shape["shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("ori_shape")) {
    tensor.SetOriginShape(ge::GeShape(shape["ori_shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("format")) {
    std::string format_str = shape["format"].get<std::string>();
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    const ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetFormat(ge_format);
  }
  if (shape.contains("ori_format")) {
    std::string format_str = shape["ori_format"].get<std::string>();
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    const ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetOriginFormat(ge_format);
  }
  if (shape.contains("dtype")) {
    std::string dtype_str = shape["dtype"].get<std::string>();
    (void)std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    const ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor.SetDataType(ge_dtype);
  }
  if (shape.contains("name")) {
    name = shape["name"];
    tensor.SetName(name);
    is_input ? op_desc->AddInputDesc(name, tensor) : op_desc->AddOutputDesc(name, tensor);
  } else {
    is_input ? op_desc->AddInputDesc(tensor) : op_desc->AddOutputDesc(tensor);
  }
}

void ParseAndSetOperatorAttr(const nlohmann::json &attr, ge::Operator &op) {
  if (!attr.contains("name") || !attr.contains("dtype") || !attr.contains("value")) {
    REPORT_INNER_ERR_MSG("E19999", "cur attr does not contain name or dtype or value.");
    return;
  }
  std::string attr_name;
  std::string dtype;
  attr_name = attr["name"].get<std::string>();
  dtype = attr["dtype"].get<std::string>();
  auto iter = parse_attr_dtype_map.find(dtype);
  if (iter == parse_attr_dtype_map.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Unknown dtype[%s], which is unsupported.", dtype.c_str());
    return;
  }
  ParseAndSetAttrValuePtr func_ptr = iter->second;
  if (func_ptr == nullptr) {
    GE_LOGE("ParseAndSetAttrValueFunc ptr cannot be null!");
    return;
  }
  (*func_ptr)(op, attr, attr_name);
  GELOGD("Finished setting attribute [name: %s] for op.", attr_name.c_str());
}

void ParseShapeDescListV2(const nlohmann::json &shape_list, ge::OpDescPtr &op_desc, const bool &is_input) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        if (shape.is_null()) {
          GELOGW("Empty input.");
          continue;
        }
        ParseShapeDescV2(shape, op_desc, is_input);
      }
    } else {
      if (elem.is_null()) {
        GELOGW("Empty input.");
        continue;
      }
      ParseShapeDescV2(elem, op_desc, is_input);
    }
  }
}

void ParseAndSetAttrsList(const nlohmann::json &attrs_list, ge::Operator &op) {
  for (const auto &attr : attrs_list) {
    ParseAndSetOperatorAttr(attr, op);
  }
}

template<typename T>
void GetConstDataPointer(const nlohmann::json &json_array, std::vector<uint8_t> &const_value) {
  std::vector<T> value = json_array.get<std::vector<T>>();
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(value.data());
  uint8_t *pv_end = pv_begin + (value.size() * sizeof(T));
  const_value = std::vector<uint8_t>(pv_begin, pv_end);
}

void CopyConstDataWithFloat16(const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  std::vector<float> const_value = json_array.get<std::vector<float>>();
  float *const_data_ptr = const_value.data();
  if (const_data_ptr == nullptr) {
    GE_LOGE("Failed to get constant data pointer");
    return;
  }
  std::vector<uint16_t> const_data_vec;
  const size_t size = sizeof(const_value)/sizeof(float);
  for (size_t i = 0; i < size; ++i) {
    const float const_data = const_data_ptr[i];
    uint16_t const_data_uint16 = optiling::Float32ToFloat16(const_data);
    (void)const_data_vec.emplace_back(const_data_uint16);
  }
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(const_data_vec.data());
  uint8_t *pv_end = pv_begin + (const_data_vec.size() * sizeof(uint16_t));
  value = std::vector<uint8_t>(pv_begin, pv_end);
}

bool CopyConstData(const std::string &dtype, const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  if (dtype == "int8") {
    GetConstDataPointer<int8_t>(json_array, value);
  } else if (dtype == "uint8") {
    GetConstDataPointer<uint8_t>(json_array, value);
  } else if (dtype == "int16") {
    GetConstDataPointer<int16_t>(json_array, value);
  } else if (dtype == "uint16") {
    GetConstDataPointer<uint16_t>(json_array, value);
  } else if (dtype == "int32") {
    GetConstDataPointer<int32_t>(json_array, value);
  } else if (dtype == "uint32") {
    GetConstDataPointer<uint32_t>(json_array, value);
  } else if (dtype == "int64") {
    GetConstDataPointer<int64_t>(json_array, value);
  } else if (dtype == "uint64") {
    GetConstDataPointer<uint64_t>(json_array, value);
  } else if (dtype == "float32") {
    GetConstDataPointer<float>(json_array, value);
  } else if (dtype == "double") {
    GetConstDataPointer<double>(json_array, value);
  } else if (dtype == "float16") {
    CopyConstDataWithFloat16(json_array, value);
  } else {
    GE_LOGE("Unknown dtype: %s", dtype.c_str());
    return false;
  }
  return true;
}

void ParseConstShapeDesc(const nlohmann::json &shape_json, std::map<std::string, TeConstTensorData> &const_tensors,
                         std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not constant tensor");
    return;
  }
  if (!shape_json.contains("name")) {
    GE_LOGE("const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  const bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    GE_LOGE("CopyConstData failed. Buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  ge::Shape ge_shape(shape);
  (void)std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  const ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
  const ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge_format, ge_dtype), res.first->second);
  (void)const_tensors.emplace(name, std::make_tuple(const_tensor.GetData(), const_tensor.GetSize(), const_tensor));
  return;
}

void ParseConstTensorList(const nlohmann::json &shape_list, std::map<std::string, TeConstTensorData> &const_tensors,
                          std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDesc(shape, const_tensors, const_values);
      }
    } else {
      ParseConstShapeDesc(elem, const_tensors, const_values);
    }
  }
}

void ParseConstShapeDescV2(const nlohmann::json &shape_json, ge::Operator &op_para,
                           std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not constant tensor");
    return;
  }
  if (!shape_json.contains("name")) {
    REPORT_INNER_ERR_MSG("E19999", "const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  const bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    REPORT_INNER_ERR_MSG("E19999", "CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  const ge::GeShape ge_shape(shape);
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  if (!dtype_str.empty()) {
    (void)std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  }
  ge::Format ge_format = ge::FORMAT_RESERVED;
  if (!format_str.empty()) {
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  }
  ge::GeTensorDesc ge_tensor(ge_shape, ge_format, ge_dtype);
  ge_tensor.SetName(name);
  ge::GeTensor const_tensor(ge_tensor, res.first->second);
  ge::GeTensorPtr const_tensor_ptr = std::make_shared<ge::GeTensor>(const_tensor);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  (void)op_para.SetInput(name.c_str(), const_op);
  return;
}

void ParseConstTensorListV2(const nlohmann::json &shape_list, ge::Operator &operator_para,
                            std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDescV2(shape, operator_para, const_values);
      }
    } else {
      ParseConstShapeDescV2(elem, operator_para, const_values);
    }
  }
}

std::string DumpByteBuffer(const ByteBuffer &buf) {
  static const std::string hex_digits = "0123456789ABCDEF";
  std::string str = buf.str();
  std::string output;
  const uint32_t num_two = 2;
  const uint32_t num_four = 4;
  const uint32_t num_fifteen = 15;
  output.reserve(str.size() * num_two);
  for (const unsigned char c : str) {
    output.push_back(hex_digits[c >> num_four]);
    output.push_back(hex_digits[c & num_fifteen]);
  }
  return output;
}

bool DumpOpRunInfo(const OpRunInfo &run_info, char *run_info_json, size_t &run_info_len) {
  if (run_info_json == nullptr) {
    GE_LOGE("run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  json_obj["block_dim"] = run_info.block_dim;
  json_obj["workspaces"] = run_info.workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.tiling_data);
  json_obj["clear_atomic"] = run_info.clear_atomic;
  json_obj["tiling_key"] = run_info.tiling_key;

  const std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    GE_LOGE("runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  return memcpy_s(run_info_json, str.size() + 1UL, str.c_str(), str.size() + 1UL) == EOK;
}

bool DumpRunInfoV2(const OpRunInfoV2 &run_info, char *run_info_json, size_t run_info_len) {
  if (run_info_json == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  std::vector<int64_t> workspaces;
  int64_t workspace;
  for (size_t i = 0; i < run_info.GetWorkspaceNum(); ++i) {
    (void) run_info.GetWorkspace(i, workspace);
    workspaces.push_back(workspace);
  }
  json_obj["block_dim"] = run_info.GetBlockDim();
  json_obj["workspaces"] = workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.GetAllTilingData());
  json_obj["clear_atomic"] = run_info.GetClearAtomic();
  json_obj["tiling_key"] = run_info.GetTilingKey();

  const std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    REPORT_INNER_ERR_MSG("E19999", "runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  return memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1) == EOK;
}

int TbeOpTilingPyInterfaceEx2BackUpInner(const char *const optype, const char *const compile_info,
                                         const char *const inputs, const char *const outputs, char *run_info_json,
                                         size_t run_info_len, const char *const compile_info_hash, uint64_t *elapse,
                                         const OpTilingFunc &tiling_func) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                         inputs, outputs);
    return 0;
  }

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  TeOpParas op_params;
  op_params.op_type = optype;
  std::map<std::string, std::vector<uint8_t>> const_values;
  try {
    const nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    const nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescList(inputs_json, op_params.inputs);
    ParseShapeDescList(outputs_json, op_params.outputs);
    ParseConstTensorList(inputs_json, op_params.const_inputs, const_values);
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func found, op_type:%s", optype);

  OpCompileInfo op_compile_info{compile_info, ""};
  if (compile_info_hash != nullptr) {
    op_compile_info.key = compile_info_hash;
  }

  OpRunInfo run_info;
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }

  const bool rc = (tiling_func)(op_params, op_compile_info, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type: %s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(\
        after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Optiling succeeded. op_type: %s", optype);
  (void)DumpOpRunInfo(run_info, run_info_json, run_info_len);
  return 1;
}

void CheckAndSetAttr(const char *attrs, ge::Operator &operator_param) {
  if (attrs != nullptr) {
    GELOGD("Attrs set from pyAPI is: %s", attrs);
    const nlohmann::json attrs_json = nlohmann::json::parse(attrs);
    ParseAndSetAttrsList(attrs_json, operator_param);
  } else {
    GELOGD("Attribute has not been set.");
  }
  return;
}

void ParseInputsAndOutputs(const char *inputs, const char *outputs, ge::OpDescPtr &op_desc,
    ge::Operator &operator_param, std::map<std::string, std::vector<uint8_t>> &const_values) {
  const nlohmann::json inputs_json = nlohmann::json::parse(inputs);
  const nlohmann::json outputs_json = nlohmann::json::parse(outputs);
  ParseShapeDescListV2(inputs_json, op_desc, true);
  ParseShapeDescListV2(outputs_json, op_desc, false);
  operator_param = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  ParseConstTensorListV2(inputs_json, operator_param, const_values);
}

int TbeOpTilingPyInterfaceEx2NewInner(const char *const optype, const char *const compile_info,
                                      const char *const inputs, const char *const outputs, char *run_info_json,
                                      size_t run_info_len, const char *const compile_info_hash, uint64_t *elapse,
                                      const OpTilingFuncV2 &tiling_func, const char *const attrs) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                         inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func v2 found, op_type: %s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  const std::string compile_info_str = compile_info;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }

  OpCompileInfoV2 op_compile_info{" ", compile_info_str};
  const ge::AscendString opCompileInfoHash(compile_info_hash);
  if (compile_info_hash != nullptr) {
    op_compile_info.SetKey(opCompileInfoHash);
  }

  OpRunInfoV2 run_info(static_cast<uint32_t>(0), false, static_cast<uint64_t>(0));
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }

  const bool rc = (tiling_func)(operator_param, op_compile_info, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type: %s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(\
        after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v2 succeeded. op_type: %s", optype);
  (void)DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

int TbeOpTilingPyInterfaceEx3Inner(const char *const optype, const char *const compile_info, const char *const inputs,
                                   const char *const outputs, char *run_info_json, size_t run_info_len,
                                   const char *const compile_info_hash, uint64_t *elapse,
                                   const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func,
                                   const char *const attrs) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                         inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func v3 found, op_type: %s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    GELOGE(ge::FAILED, "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    REPORT_INNER_ERR_MSG("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  if (compile_info_hash == nullptr) {
    return 0;
  }

  const ge::AscendString compile_info_json_str = compile_info;
  void* op_compile_json_ptr = (parse_func)(operator_param, compile_info_json_str);

  OpRunInfoV2 run_info(static_cast<uint32_t>(0), false, static_cast<uint64_t>(0));
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }
  const bool rc = (tiling_func)(operator_param, op_compile_json_ptr, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type: %s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>\
        (after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v3 succeeded. op_type: %s", optype);
  (void)DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

int TbeOpTilingPyInterfaceEx4Inner(const char *const optype, const char *const compile_info, const char *const inputs,
                                   const char *const outputs, char *run_info_json, size_t run_info_len,
                                   const char *const compile_info_hash, uint64_t *elapse,
                                   const OpTilingFuncV4 &tiling_func, const OpParseFuncV4 &parse_func,
                                   const char *const attrs) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                         inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func v4 found, op_type:%s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  std::string op_type_str = optype;
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("", op_type_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc_ptr, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to parse json during tiling v4. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  if (compile_info_hash == nullptr) {
    return 0;
  }

  const ge::AscendString compile_info_json = compile_info;
  const CompileInfoPtr op_compile_json_ptr = (parse_func)(operator_param, compile_info_json);

  OpRunInfoV2 run_info(static_cast<uint32_t>(0), false, static_cast<uint64_t>(0));
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }
  const bool rc = (tiling_func)(operator_param, op_compile_json_ptr, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type: %s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v4 succeed. op_type:%s", optype);
  (void) DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

gert::KernelContextHolder BuildTilingParseContextHolder(ge::OpDescPtr &op_desc, const char *compile_info,
                                                        const char *op_type, fe::PlatFormInfos &platform_infos,
                                                        const gert::OpImplKernelRegistry::OpImplFunctionsV2 *funcs) {
  std::vector<std::pair<void *, gert::Chain::Deleter>> tiling_parse_outputs(1, std::make_pair(nullptr, nullptr));
  if (op_desc->GetType() != OP_TYPE_AUTO_TILING) {
    tiling_parse_outputs[0].first = funcs->compile_info_creator();
    tiling_parse_outputs[0].second = funcs->compile_info_deleter;
  }

  return gert::KernelRunContextBuilder()
      .Inputs({std::make_pair(const_cast<char *>(compile_info), nullptr),
               std::make_pair(reinterpret_cast<void *>(&platform_infos), nullptr),
               std::make_pair(const_cast<char *>(op_type), nullptr)})
      .Outputs(tiling_parse_outputs)
      .Build(op_desc);
}

gert::KernelContextHolder BuildTilingContext(ContextComponent &context_com, gert::KernelContext *tiling_parse_context,
                                             fe::PlatFormInfos &platform_infos) {
  if (context_com.storage_shapes.size() >= std::numeric_limits<size_t>::max() - kTilingCtxFixedInputSize) {
    GELOGE(ge::GRAPH_FAILED, "Context storage size overflow.");
    return gert::KernelContextHolder();
  }
  std::vector<void *> tiling_context_inputs(context_com.storage_shapes.size() + kTilingCtxFixedInputSize, nullptr);
  for (size_t i = 0UL; i < context_com.index_to_tensors.size(); ++i) {
    tiling_context_inputs[context_com.index_to_tensors[i].first] =
        reinterpret_cast<gert::Tensor *>(context_com.index_to_tensors[i].second.get());
  }
  for (size_t i = 0UL; i < context_com.storage_shapes.size(); ++i) {
    if (tiling_context_inputs[i] == nullptr) {
      tiling_context_inputs[i] = &context_com.storage_shapes[i];
    }
  }
  if (tiling_parse_context->GetOutputPointer<void **>(0) == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Output Pointer is null.");
    return gert::KernelContextHolder();
  }
  tiling_context_inputs[context_com.storage_shapes.size()] = *tiling_parse_context->GetOutputPointer<void **>(0);
  tiling_context_inputs[context_com.storage_shapes.size() + 1UL] = reinterpret_cast<void *>(&platform_infos);
  int32_t deterministic = 0;
  (void)ge::AttrUtils::GetInt(context_com.op_desc, "deterministic", deterministic);
  GELOGI("Get deterministic: %d from node: %s", deterministic, context_com.op_desc->GetName().c_str());
  tiling_context_inputs[context_com.storage_shapes.size() + kDeterministicOffset] =
      reinterpret_cast<void *>(deterministic);
  int32_t deterministic_level = 0;
 	(void)ge::AttrUtils::GetInt(context_com.op_desc, "deterministic_level", deterministic_level);
  if (deterministic_level < 0 || deterministic_level > 2) {
    std::string readable_name = ge::GEThreadLocalContext().GetReadableName("ge.deterministicLevel");
    std::string error_msg =
        "Valid values for " + readable_name + " are {0,1,2}.";
    GELOGE(ge::FAILED, "Valid values for %s are {0,1,2}, given value is %d", readable_name.c_str(),
           deterministic_level);
    (void) REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                     std::vector<const char *>(
                                         {
                                           readable_name.c_str(), to_string(deterministic_level).c_str(),
                                               error_msg.c_str()
                                         }));
    return gert::KernelContextHolder();
  }
  GELOGI("Get deterministic level: %d from node: %s", deterministic_level, context_com.op_desc->GetName().c_str());
  tiling_context_inputs[context_com.storage_shapes.size() + kDeterministicLevelOffset] =
      reinterpret_cast<void *>(deterministic_level);
  return gert::KernelRunContextBuilder()
      .Inputs(tiling_context_inputs)
      .Outputs(
      {nullptr, nullptr, &context_com.atomic_flag, context_com.tiling_data.get(), context_com.workspace_size.get(),
      &context_com.tiling_cond, &context_com.schedule_mode, nullptr, nullptr})
      .Build(context_com.op_desc);
}

ge::graphStatus DoTilingParse(const gert::OpImplKernelRegistry::OpImplFunctionsV2 *funcs,
                              gert::KernelContextHolder &tiling_parse_context_holder) {
  GE_CHECK_NOTNULL(tiling_parse_context_holder.context_);
  return (funcs->tiling_parse)(tiling_parse_context_holder.context_);
}

ge::graphStatus DoTilingWithTiming(const gert::OpImplKernelRegistry::OpImplFunctionsV2 *funcs, uint64_t *elapse,
                                   gert::KernelContextHolder &tiling_context_holder) {
  GE_CHECK_NOTNULL(tiling_context_holder.context_);
  // calcu tiling cost time
  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }
  const auto ret = (funcs->tiling)(reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_));
  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseJson(const char *const inputs, const char *const outputs, const char *const attrs,
                          const char *const extra_info, ContextComponent &context_com) {
  if (ParseInputs(inputs, context_com) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse inputs failed.");
    REPORT_INNER_ERR_MSG("E19999", "Parse inputs failed.");
    return ge::GRAPH_FAILED;
  }
  if (ParseOutputs(outputs, context_com) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse outputs failed.");
    REPORT_INNER_ERR_MSG("E19999", "Parse outputs failed.");
    return ge::GRAPH_FAILED;
  }
  if (ParseAttrs(attrs, context_com.op_desc) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse attrs failed.");
    REPORT_INNER_ERR_MSG("E19999", "Parse attrs failed.");
    return ge::GRAPH_FAILED;
  }
  if (ParseExtraInfos(extra_info, context_com.op_desc) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse extra info failed.");
    REPORT_INNER_ERR_MSG("E19999", "Parse extra info failed.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseDeviceIdAndCoreType(const char *compile_info, uint32_t &device_id, std::string &core_type) {
  const std::string compile_str = compile_info;
  if (compile_str.empty()) {
    GELOGD("compile info is empty.");
    return ge::GRAPH_SUCCESS;
  }
  nlohmann::json info_list;
  try {
    info_list = nlohmann::json::parse(compile_info);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", compile_info);
    return ge::FAILED;
  }
  GELOGD("Parsing compile info: %s.", info_list.dump().c_str());

  if (info_list.contains("device_id")) {
    if (info_list["device_id"].is_null()) {
      GELOGD("device_id is null.");
    } else {
      device_id = std::atoi(info_list["device_id"].get<std::string>().c_str());
      GELOGI("Parse device id: %u.", device_id);
    }
  }
  if (info_list.contains(ge::ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE)) {
    if (info_list[ge::ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE].is_null()) {
      GELOGD("Attribute %s is null.", ge::ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE.c_str());
    } else {
      core_type = info_list[ge::ATTR_NAME_SGT_CUBE_VECTOR_CORE_TYPE].get<std::string>();
      GELOGI("Parsing core type: %s.", core_type.c_str());
    }
  } else {
    if (info_list.contains(ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE)) {
      if (info_list[ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE].is_null()) {
        GELOGD("Attribute %s is null.", ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE.c_str());
      } else {
        core_type = info_list[ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE].get<std::string>();
        GELOGI("Parsing core type: %s.", core_type.c_str());
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetPlatformInfos(uint32_t device_id, const std::string &core_type, fe::PlatFormInfos &platform_infos,
                         fe::PlatFormInfos &platform_infos_bak) {
  GE_ASSERT(fe::PlatformInfoManager::Instance().InitializePlatformInfo() == 0U, "InitializePlatformInfo failed.");

  GE_ASSERT(fe::PlatformInfoManager::Instance().GetPlatformInstanceByDevice(device_id, platform_infos) == 0,
            "GetPlatformInstanceByDevice failed.");

  platform_infos.SetCoreNumByCoreType(core_type);
  GELOGD("device id: %u, core type: %s, core num: %u.", device_id, core_type.c_str(), platform_infos.GetCoreNum());

  platform_infos_bak = platform_infos;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseSocVersion(fe::PlatFormInfos &platform_info, std::string &socVersionStr, std::string &shortSocVersionStr) {
  GE_ASSERT_TRUE(platform_info.GetPlatformResWithLock("version", "SoC_version", socVersionStr));
  GELOGI("SoC_version in platform_infos: %s", socVersionStr.c_str());

  GE_ASSERT_TRUE(platform_info.GetPlatformResWithLock("version", "Short_SoC_version", shortSocVersionStr));
  GELOGI("Short_SoC_version in platform_infos: %s", shortSocVersionStr.c_str());

  return ge::GRAPH_SUCCESS;
}

int64_t GetNewMaxTilingSize(const char *const attrs) {
  if (attrs == nullptr) {
    return 0;
  }
  nlohmann::json attr_json = nlohmann::json::parse(attrs);
    for (const auto &attr : attr_json) {
      if (attr.contains("name")  && attr.contains("value") &&
        attr["name"].get<std::string>() == "ascendc_op_para_size") { // new max tiling size
        return attr["value"].get<std::int64_t>();
      }
    }
  return 0;
}

int TbeOptilingPyInterfaceNew(const char *const op_type, const char *const compile_info, const char *const inputs,
                              const char *const outputs, char *run_info_json, size_t run_info_len, uint64_t *elapse,
                              const char *const attrs, const char *const extra_info) {
  if ((compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    GELOGE(ge::GRAPH_FAILED, "compile_info/inputs/outputs is null.");
    REPORT_INNER_ERR_MSG("E19999", "compile_info/inputs/outputs is null.");
    return 0;
  }

  const gert::OpImplKernelRegistry::OpImplFunctionsV2 *funcs;
  if (!FindImplFuncs(op_type, funcs)) {
    return 0;
  }
  ContextComponent context_com {};
  context_com.op_desc = std::make_shared<ge::OpDesc>("", op_type);
  if ((context_com.op_desc == nullptr) ||
      (ParseJson(inputs, outputs, attrs, extra_info, context_com) != ge::GRAPH_SUCCESS)) {
    return 0;
  }

  uint32_t device_id = 0U;
  std::string core_type;
  GE_ASSERT_SUCCESS(ParseDeviceIdAndCoreType(compile_info, device_id, core_type));

  fe::PlatFormInfos platform_infos;
  fe::PlatFormInfos platform_infos_bak;
  GE_ASSERT_SUCCESS(GetPlatformInfos(device_id, core_type, platform_infos, platform_infos_bak));

  std::string socVersionStr;
  std::string shortSocVersionStr;
  GE_ASSERT_SUCCESS(ParseSocVersion(platform_infos, socVersionStr, shortSocVersionStr));

  fe::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::CoreNumUtils::GetGeDefaultPlatformInfo(socVersionStr, platform_info));

  // 如果配置了算子级核数，更新到副本PlatformInfos中，后续用副本，防止影响其他算子
  bool is_op_core_num_set = false;
  GE_ASSERT_SUCCESS(ge::CoreNumUtils::UpdatePlatformInfosWithOpDesc(platform_info, context_com.op_desc, platform_infos_bak, is_op_core_num_set));

  // tiling parse
  gert::KernelContextHolder tiling_parse_context_holder;
  if (is_op_core_num_set) {
    tiling_parse_context_holder = BuildTilingParseContextHolder(context_com.op_desc, compile_info, op_type,
                                                                   platform_infos_bak, funcs);
  } else {
    tiling_parse_context_holder = BuildTilingParseContextHolder(context_com.op_desc, compile_info, op_type,
                                                                   platform_infos, funcs);
  }

  if (DoTilingParse(funcs, tiling_parse_context_holder) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Op %s tiling parse failed", op_type);
    REPORT_INNER_ERR_MSG("E19999", "Op %s tiling parse failed", op_type);
    return 0;
  }

  // tiling
  int64_t max_size = -1;
  const int64_t new_max_tiling_size = GetNewMaxTilingSize(attrs);
  if (!ge::AttrUtils::GetInt(context_com.op_desc, kMaxTilingSize, max_size)) {
    GELOGI("Missing maximum tiling size in opdesc.");
    if (new_max_tiling_size != 0) {
      max_size = new_max_tiling_size;
    }
  }
  if (max_size == -1) {
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }
  const auto aligned_max_size = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  context_com.tiling_data = gert::TilingData::CreateCap(aligned_max_size);
  context_com.workspace_size = gert::ContinuousVector::Create<size_t>(kWorkspaceHolerSize);
  gert::KernelContextHolder tiling_context_holder;
  if (is_op_core_num_set) {
    tiling_context_holder = BuildTilingContext(context_com, tiling_parse_context_holder.context_, platform_infos_bak);
  } else {
    tiling_context_holder = BuildTilingContext(context_com, tiling_parse_context_holder.context_, platform_infos);
  }

  if (tiling_context_holder.context_ == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Output build tiling context failed.");
    return 0;
  }
  if (tiling_context_holder.GetKernelContext()->GetOutputPointer<int32_t>(
                                                gert::TilingContext::kOutputTilingCond) == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Output tiling cond is null.");
    return 0;
  }
  if (tiling_context_holder.GetKernelContext()->GetOutputPointer<uint32_t>(
                                                gert::TilingContext::kOutputScheduleMode) == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Output tiling cond is null.");
    return 0;
  }

  // BuildTilingContext will not initialize schedule mode, initialize it here
  *tiling_context_holder.GetKernelContext()->GetOutputPointer<int32_t>(gert::TilingContext::kOutputTilingCond) = 0;
  *tiling_context_holder.GetKernelContext()->GetOutputPointer<uint32_t>(gert::TilingContext::kOutputScheduleMode) = 0;
  if (DoTilingWithTiming(funcs, elapse, tiling_context_holder) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Op %s tiling failed", op_type);
    REPORT_INNER_ERR_MSG("E19999", "Op %s tiling failed", op_type);
    return 0;
  }

  if (!DumpRunInfo(tiling_context_holder.context_, run_info_json, run_info_len)) {
    GELOGE(ge::GRAPH_FAILED, "Dump op %s tiling result failed", op_type);
    REPORT_INNER_ERR_MSG("E19999", "Dump op %s tiling result failed", op_type);
    return 0;
  }
  GELOGI("Op tiling succeed. op_type:%s", op_type);
  return 1;
}

int TbeOpTilingPyInterfaceOld(const char *const optype, const char *const compile_info,
                              const char *const compile_info_hash, const char *const inputs, const char *const outputs,
                              const char *const attrs, char *run_info_json, size_t run_info_len, uint64_t *elapse,
                              const char *const extra_info) {
  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(optype);
  if (iter == op_func_map.end()) {
    GELOGI("Op tiling function for op_type [%s] not found.", optype);
    return TbeOptilingPyInterfaceNew(optype, compile_info, inputs, outputs, run_info_json, run_info_len, elapse, attrs,
                                     extra_info);
  }
  OpTilingFuncInfo &op_func_info = iter->second;
  int ret = 0;
  if (op_func_info.IsFunctionV4()) {
    const OpTilingFuncV4 &tiling_func = op_func_info.GetOpTilingFuncV4();
    const OpParseFuncV4 &parse_func = op_func_info.GetOpParseFuncV4();
    ret = TbeOpTilingPyInterfaceEx4Inner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                         compile_info_hash, elapse, tiling_func, parse_func, attrs);
  } else if (op_func_info.IsFunctionV3()) {
    const OpTilingFuncV3 &tiling_func = op_func_info.GetOpTilingFuncV3();
    const OpParseFuncV3 &parse_func = op_func_info.GetOpParseFuncV3();
    ret = TbeOpTilingPyInterfaceEx3Inner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                         compile_info_hash, elapse, tiling_func, parse_func, attrs);
  } else if (op_func_info.IsFunctionV2()) {
    const OpTilingFuncV2 &tiling_func = op_func_info.GetOpTilingFuncV2();
    ret = TbeOpTilingPyInterfaceEx2NewInner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                            compile_info_hash, elapse, tiling_func, attrs);
  } else if (op_func_info.IsFunctionV1()) {
    const OpTilingFunc &tiling_func = op_func_info.GetOpTilingFunc();
    ret = TbeOpTilingPyInterfaceEx2BackUpInner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                               compile_info_hash, elapse, tiling_func);
  } else {
    GE_LOGE("Optiling func of op type [%s] is completely empty.", optype);
  }
  return ret;
}

extern "C" int OpTilingForCompile(const char *optype, const char *compile_info, const char *compile_info_hash,
                                  const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                  size_t run_info_len, uint64_t *elapse, const char *extra_info) {
  if (optype == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op type is null.");
    REPORT_INNER_ERR_MSG("E19999", "op type is null.");
    return 0;
  }

  if (optype == OP_TYPE_AUTO_TILING) {
    GELOGI("The tiling function is automatically enabled for tiling on rt2.");
    return TbeOptilingPyInterfaceNew(optype, compile_info, inputs, outputs, run_info_json, run_info_len, elapse, attrs,
                                     extra_info);
  }
  return TbeOpTilingPyInterfaceOld(optype, compile_info, compile_info_hash, inputs, outputs, attrs, run_info_json,
                                   run_info_len, elapse, extra_info);
}

extern "C" const char *DoOpTilingForCompile(const char *optype,
                                            const char *compile_info,
                                            const char *compile_info_hash,
                                            const char *inputs,
                                            const char *outputs,
                                            const char *attrs,
                                            char *run_info_json,
                                            size_t run_info_len,
                                            uint64_t *elapse,
                                            const char *extra_info) {
  if (optype == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op type is null.");
    REPORT_INNER_ERR_MSG("E19999", "op type is null.");
    error_string = GetRawErrorMessage();
    return error_string.data();
  }

  if (optype == OP_TYPE_AUTO_TILING) {
    GELOGI("The tiling function is automatically enabled for tiling on rt2.");
    if (TbeOptilingPyInterfaceNew(optype, compile_info, inputs, outputs, run_info_json, run_info_len, elapse, attrs,
                                  extra_info) == 0) {
      GELOGE(ge::GRAPH_FAILED, "TbeOptilingPyInterfaceNew failed.");
      REPORT_INNER_ERR_MSG("E19999", "TbeOptilingPyInterfaceNew failed.");
    }
    error_string = GetRawErrorMessage();
    return error_string.data();
  }
  if (TbeOpTilingPyInterfaceOld(optype, compile_info, compile_info_hash, inputs, outputs, attrs, run_info_json,
                                run_info_len, elapse, extra_info) == 0) {
    GELOGE(ge::GRAPH_FAILED, "TbeOpTilingPyInterfaceOld failed.");
    REPORT_INNER_ERR_MSG("E19999", "TbeOpTilingPyInterfaceOld failed.");
  }
  error_string = GetRawErrorMessage();
  return error_string.data();
}

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *compile_info_hash,
                                      const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                      size_t run_info_len, uint64_t *elapse) {
  GELOGW("Deprecated api, use OpTilingForCompile instead.");
  if (optype == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op type is null.");
    REPORT_INNER_ERR_MSG("E19999", "op type is null.");
    return 0;
  }

  if (optype == OP_TYPE_AUTO_TILING) {
    GELOGI("The tiling function is automatically enabled for tiling on rt2.");
    return TbeOptilingPyInterfaceNew(optype, compile_info, inputs, outputs, run_info_json, run_info_len, elapse, attrs,
                                     nullptr);
  }

  return TbeOpTilingPyInterfaceOld(optype, compile_info, compile_info_hash, inputs, outputs, attrs, run_info_json,
                                   run_info_len, elapse, nullptr);
}

extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse) {
  GELOGW("Deprecated api, use OpTilingForCompile instead.");
  return TbeOpTilingPyInterface(optype, compile_info, compile_info_hash, inputs, outputs, nullptr, run_info_json,
                                run_info_len, elapse);
}

extern "C" int TbeOpTilingPyInterfaceEx4(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV4 &tiling_func, const OpParseFuncV4 &parse_func,
                                         const char *attrs) {
  GELOGW("Deprecated api, use OpTilingForCompile instead.");
  return TbeOpTilingPyInterfaceEx4Inner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                        compile_info_hash, elapse, tiling_func, parse_func, attrs);
}

extern "C" int TbeOpTilingPyInterfaceEx3(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func,
                                         const char *attrs) {
  GELOGW("Deprecated api, use OpTilingForCompile instead.");
  return TbeOpTilingPyInterfaceEx3Inner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                        compile_info_hash, elapse, tiling_func, parse_func, attrs);
}

extern "C" int TbeOpTilingPyInterfaceEx2New(const char *optype, const char *compile_info, const char *inputs,
                                            const char *outputs, char *run_info_json, size_t run_info_len,
                                            const char *compile_info_hash, uint64_t *elapse,
                                            const OpTilingFuncV2 &tiling_func, const char *attrs) {
  GELOGW("Deprecated api, use OpTilingForCompile instead.");
  return TbeOpTilingPyInterfaceEx2NewInner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                           compile_info_hash, elapse, tiling_func, attrs);
}

extern "C" int TbeOpTilingPyInterfaceEx2BackUp(const char *optype, const char *compile_info, const char *inputs,
                                               const char *outputs, char *run_info_json, size_t run_info_len,
                                               const char *compile_info_hash, uint64_t *elapse,
                                               const OpTilingFunc &tiling_func) {
  GELOGW("Deprecated api, use OpTilingForCompile instead.");
  return TbeOpTilingPyInterfaceEx2BackUpInner(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                              compile_info_hash, elapse, tiling_func);
}

extern "C" Status TbeLoadSoAndSaveToRegistry(const char *so_path) {
  GE_ASSERT_NOTNULL(so_path);
  GELOGD("start TbeLoadSoAndSaveToRegistry, so path: %s, pid is %d", so_path, getpid());
  auto space_registry_v2 = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  if (space_registry_v2 == nullptr) {
    space_registry_v2 = std::make_shared<gert::OpImplSpaceRegistryV2>();
    GE_CHECK_NOTNULL(space_registry_v2);
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry_v2);
  }
  return gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->AddSoToRegistry(
      gert::OppSoDesc({ge::AscendString(so_path)}, ""));
}
}
}  // namespace optiling
