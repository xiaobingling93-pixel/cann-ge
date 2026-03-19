/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <map>
#include <regex>
#include <string>
#include <tuple>
#include <vector>
#include <sys/syscall.h>
#include "utils.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "dlog_pub.h"
#include "flow_graph/data_flow.h"
#include "dflow/compiler/session/dflow_api.h"
#include "ge/ge_api_v2.h"
#include "parser/onnx_parser.h"
#include "parser/tensorflow_parser.h"

namespace ge {
namespace {
namespace py = pybind11;
constexpr size_t kMaxUserDataSize = 64U;

struct ReturnMessage {
  uint32_t ret_code;
  std::string error_msg;
};

int64_t GetTid() {
  thread_local const int64_t tid = syscall(__NR_gettid);
  return tid;
}

#define DFLOW_MODULE_NAME static_cast<int32_t>(GE)

#define DFLOW_LOGE(fmt, ...) dlog_error(DFLOW_MODULE_NAME, "[%s][tid:%ld]: " fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)

class DFlowDataTypeManager {
 public:
  static DFlowDataTypeManager &GetInstance() {
    static DFlowDataTypeManager data_type_manager;
    return data_type_manager;
  }

  void Init(const std::map<ge::DataType, py::array> &type_map) {
    for (const auto &item : type_map) {
      auto const dtype = item.first;
      auto const array = item.second;
      numpy_dtype_to_ge_dtype[array.dtype().char_()] = dtype;
      const auto buff = array.request();
      ge_dtype_to_format_desc[dtype] = buff.format;
    }
  }

  const std::map<char, ge::DataType> &GetNumpyDtypeToGeDType() const {
    return numpy_dtype_to_ge_dtype;
  }

  const std::map<ge::DataType, std::string> &GetGeDtypeToFormatDesc() const {
    return ge_dtype_to_format_desc;
  }

 private:
  DFlowDataTypeManager() = default;
  std::map<char, ge::DataType> numpy_dtype_to_ge_dtype;
  std::map<ge::DataType, std::string> ge_dtype_to_format_desc;
};

std::string ConvertNumpyDataTypeToGeDataType(const py::dtype &np_data_dtype, ge::DataType &ge_data_type) {
  const auto &numpy_dtype_to_ge_type = DFlowDataTypeManager::GetInstance().GetNumpyDtypeToGeDType();
  const auto it = numpy_dtype_to_ge_type.find(np_data_dtype.char_());
  if (it != numpy_dtype_to_ge_type.cend()) {
    ge_data_type = it->second;
    return "";
  }
  return std::string("Unsupported data type:") + np_data_dtype.char_();
}

bool IsStringDataType(const std::string &data_type) {
  static const std::regex r("([^a-zA-Z])(S|U)[0-9]+");
  return std::regex_match(data_type, r);
};

const std::string ERR_MSG = "for details about the error information, see the ascend log.";

struct UserDataInfo {
  void *user_data_ptr = nullptr;
  size_t data_size = 0UL;
  size_t offset = 0UL;
};

struct FlowInfo {
  uint64_t start_time = 0UL;
  uint64_t end_time = 0UL;
  uint64_t transaction_id = 0UL;
  uint32_t flow_flags = 0U;
  UserDataInfo user_data;
};

struct DflowStringHead {
  int64_t addr;
  int64_t len;
};

std::vector<ge::AscendString> SplitToStrVector(const char *dataPtr, const size_t &data_size,
                                               const size_t &element_num) {
  std::vector<ge::AscendString> res;
  if (element_num == 0) {
    return res;
  }
  const size_t byte_num_per_element = data_size / element_num;
  if (byte_num_per_element == 0UL) {
    return res;
  }
  res.reserve(element_num);
  for (size_t i = 0UL; i < element_num; ++i) {
    res.emplace_back(dataPtr + i * byte_num_per_element);
  }
  return res;
}

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

class PyFlowMsg : public ge::FlowMsg {
 public:
  ge::MsgType GetMsgType() const override {
    PYBIND11_OVERRIDE_PURE(ge::MsgType, ge::FlowMsg, GetMsgType,);
  }

  void SetMsgType(ge::MsgType msg_type) override {
    PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetMsgType, msg_type);
  }

  ge::Tensor *GetTensor() const override {
    PYBIND11_OVERRIDE_PURE(ge::Tensor *, ge::FlowMsg, GetTensor,);
  }

  ge::Status GetRawData(void *&data_ptr, uint64_t &data_size) const override {
    (void)data_ptr;
    (void)data_size;
    PYBIND11_OVERRIDE_PURE(ge::Status, ge::FlowMsg, GetRawData,);
  }

  int32_t GetRetCode() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowMsg, GetRetCode,);
  }

  void SetRetCode(int32_t ret_code) override {
    PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetRetCode, ret_code);
  }

  void SetStartTime(uint64_t start_time) override {
    PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetStartTime, start_time);
  }

  uint64_t GetStartTime() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, ge::FlowMsg, GetStartTime,);
  }

  void SetEndTime(uint64_t end_time) override {
    PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetEndTime, end_time);
  }

  uint64_t GetEndTime() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, ge::FlowMsg, GetEndTime,);
  }

  void SetFlowFlags(uint32_t flags) override {
    PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetFlowFlags, flags);
  }

  uint32_t GetFlowFlags() const override {
    PYBIND11_OVERRIDE_PURE(uint32_t, ge::FlowMsg, GetFlowFlags,);
  }

  void SetTransactionId(uint64_t transaction_id) override {
    PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetTransactionId, transaction_id);
  }

  uint64_t GetTransactionId() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, ge::FlowMsg, GetTransactionId,);
  }

  ge::Status GetUserData(void *data, size_t size, size_t offset = 0U) const override {
    (void)data;
    (void)size;
    (void)offset;
    PYBIND11_OVERRIDE_PURE(ge::Status, FlowMsg, GetUserData,);
  }

  ge::Status SetUserData(const void *data, size_t size, size_t offset = 0U) override {
    PYBIND11_OVERRIDE_PURE(ge::Status, ge::FlowMsg, SetUserData, data, size, offset);
  }
};

std::map<ge::AscendString, ge::AscendString> ConvertToAscendString(const std::map<std::string, std::string> &str_map) {
  std::map<ge::AscendString, ge::AscendString> ascend_string_map;
  for (const auto &it : str_map) {
    AscendString key{it.first.c_str()};
    AscendString value{it.second.c_str()};
    ascend_string_map[key] = value;
  }
  return ascend_string_map;
}

void BindDflowAttr(py::module &m) {
  m.attr("PARAM_INVALID") = ACL_ERROR_GE_PARAM_INVALID;
  m.attr("SHAPE_INVALID") = ACL_ERROR_GE_SHAPE_INVALID;
  m.attr("DATATYPE_INVALID") = ACL_ERROR_GE_DATATYPE_INVALID;
  m.attr("NOT_INIT") = ACL_ERROR_GE_EXEC_NOT_INIT;
  m.attr("INNER_ERROR") = ACL_ERROR_GE_INTERNAL_ERROR;
  m.attr("SUBHEALTHY") = ACL_ERROR_GE_SUBHEALTHY;

  py::class_<dflow::TimeBatch>(m, "TimeBatch")
      .def(py::init())
      .def_readwrite("time_window", &dflow::TimeBatch::time_window)
      .def_readwrite("batch_dim", &dflow::TimeBatch::batch_dim)
      .def_readwrite("drop_remainder", &dflow::TimeBatch::drop_remainder);

  py::class_<dflow::CountBatch>(m, "CountBatch")
      .def(py::init())
      .def_readwrite("batch_size", &dflow::CountBatch::batch_size)
      .def_readwrite("slide_stride", &dflow::CountBatch::slide_stride)
      .def_readwrite("timeout", &dflow::CountBatch::timeout)
      .def_readwrite("padding", &dflow::CountBatch::padding);

  py::class_<dflow::DataFlowInputAttr>(m, "DataFlowInputAttr")
      .def(py::init())
      .def_readwrite("attr_type", &dflow::DataFlowInputAttr::attr_type)
      .def_readwrite("attr_value", &dflow::DataFlowInputAttr::attr_value);
}

void BindDflowEnum(py::module &m) {
  py::enum_<ge::MsgType>(m, "MsgType", py::arithmetic())
      .value("MSG_TYPE_TENSOR_DATA", ge::MsgType::MSG_TYPE_TENSOR_DATA)
      .value("MSG_TYPE_RAW_MSG", ge::MsgType::MSG_TYPE_RAW_MSG)
      .export_values();

  py::enum_<dflow::DataFlowAttrType>(m, "DataFlowAttrType")
      .value("COUNT_BATCH", dflow::DataFlowAttrType::COUNT_BATCH)
      .value("TIME_BATCH", dflow::DataFlowAttrType::TIME_BATCH)
      .export_values();
}

void BindDflowInitAndFinalize(py::module &m) {
  m.def("ge_initialize", [](const std::map<std::string, std::string> &options) {
          auto options_ascend_string = ConvertToAscendString(options);
          auto ret = ge::GEInitializeV2(options_ascend_string);
          if (ret != ge::SUCCESS) {
            DFLOW_LOGE("GEInitialize failed, ret=%u.", ret);
            return ret;
          }
          ret = dflow::DFlowInitialize(options_ascend_string);
          if (ret != ge::SUCCESS) {
            DFLOW_LOGE("DFlowInitialize failed, ret=%u.", ret);
            return ret;
          }
          return ret;
        },
        py::call_guard<py::gil_scoped_release>());

  m.def("ge_finalize", []() {
          auto ret = dflow::DFlowFinalize();
          if (ret != ge::SUCCESS) {
            DFLOW_LOGE("DFlowFinalize failed, ret=%u.", ret);
            return ret;
          }
          ret = ge::GEFinalizeV2();
          if (ret != ge::SUCCESS) {
            DFLOW_LOGE("GEFinalize failed, ret=%u.", ret);
            return ret;
          }
          return ret;
        },
        py::call_guard<py::gil_scoped_release>());
}

void BindReturnMessage(py::module &m) {
  py::class_<ReturnMessage>(m, "ReturnMessage")
      .def(py::init<uint32_t, std::string>())
      .def_readwrite("ret_code", &ReturnMessage::ret_code)
      .def_readwrite("error_msg", &ReturnMessage::error_msg);
}

void BindProcessPoint(py::module &m) {
  py::class_<dflow::ProcessPoint>(m, "ProcessPoint");
  py::class_<dflow::FunctionPp, dflow::ProcessPoint>(m, "FunctionPp")
      .def(py::init<const char *>())
      .def("set_compile_config", &dflow::FunctionPp::SetCompileConfig)
      // DataType和bool要定义在int64_前面，否则按照pybind11的匹配规则，会将DataType和bool匹配到int64_t
      .def("set_init_param", overload_cast_<const char *, const DataType &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param",
           overload_cast_<const char *, const std::vector<ge::DataType> &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param", overload_cast_<const char *, const char *>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param",
           [](dflow::FunctionPp &self, const char *attrName, const std::vector<std::string> &values) {
             std::vector<AscendString> strValues;
             strValues.reserve(values.size());
             for (auto &value : values) {
               strValues.emplace_back(value.c_str());
             }
             self.SetInitParam(attrName, strValues);
           })
      .def("set_init_param", overload_cast_<const char *, const bool &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param",
           overload_cast_<const char *, const std::vector<bool> &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param", overload_cast_<const char *, const int64_t &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param",
           overload_cast_<const char *, const std::vector<int64_t> &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param",
           overload_cast_<const char *, const std::vector<std::vector<int64_t>> &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param", overload_cast_<const char *, const float &>()(&dflow::FunctionPp::SetInitParam))
      .def("set_init_param",
           overload_cast_<const char *, const std::vector<float> &>()(&dflow::FunctionPp::SetInitParam))
      .def("add_invoked_closure",
           overload_cast_<const char *, const dflow::GraphPp &>()(&dflow::FunctionPp::AddInvokedClosure))
      .def("add_invoked_closure",
           overload_cast_<const char *, const dflow::FlowGraphPp &>()(&dflow::FunctionPp::AddInvokedClosure));

  py::class_<dflow::GraphPp, dflow::ProcessPoint>(m, "GraphPp")
      .def(py::init<const char *, const dflow::GraphBuilder>())
      .def("set_compile_config", &dflow::GraphPp::SetCompileConfig);

  py::class_<dflow::FlowGraphPp, dflow::ProcessPoint>(m, "FlowGraphPp")
      .def(py::init<const char *, const dflow::FlowGraphBuilder>())
      .def("set_compile_config", &dflow::FlowGraphPp::SetCompileConfig);
}

void BindLoadPp(py::module &m) {
  m.def("load_graph_pp", [](const std::string &framework, const std::string &graph_file,
                            const std::map<std::string, std::string> &load_params,
                            const std::string &compile_config_path, const std::string &name) {
    std::map<ge::AscendString, ge::AscendString> params = ConvertToAscendString(load_params);
    dflow::GraphPp err_graph_pp{name.data(), []() {
      return ge::Graph();
    }};

    static const std::set<std::string> support_frameworks = {"tensorflow", "onnx", "mindspore"};
    if (support_frameworks.find(framework) == support_frameworks.cend()) {
      ReturnMessage return_msg = {.ret_code = ACL_ERROR_GE_PARAM_INVALID,
                                  .error_msg = "Unsupported framework: " + framework};
      return std::make_tuple(return_msg, err_graph_pp);
    }

    dflow::GraphBuilder graph_build = [framework, graph_file, params]() {
      ge::Graph graph;
      if (framework == "tensorflow") {
        const auto ret = aclgrphParseTensorFlow(graph_file.data(), params, graph);
        if (ret != ge::GRAPH_SUCCESS) {
          DFLOW_LOGE("Failed to parse tensorflow model, file=%s, ret=%u", graph_file.c_str(), ret);
        }
      } else if (framework == "onnx") {
        const auto ret = aclgrphParseONNX(graph_file.data(), params, graph);
        if (ret != ge::GRAPH_SUCCESS) {
          DFLOW_LOGE("Failed to parse onnx model, file=%s, ret=%u", graph_file.c_str(), ret);
        }
      } else if (framework == "mindspore") {
        const auto ret = graph.LoadFromFile(graph_file.data());
        if (ret != ge::GRAPH_SUCCESS) {
          DFLOW_LOGE("Failed to parse mindspore model, file=%s, ret=%u", graph_file.c_str(), ret);
        }
      } else {
        DFLOW_LOGE("Unsupported framework, framework=%s, file=%s", framework.c_str(), graph_file.c_str());
      }
      return graph;
    };
    dflow::GraphPp graph_pp{name.data(), graph_build};
    (void)graph_pp.SetCompileConfig(compile_config_path.data());
    ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
    return std::make_tuple(return_msg, graph_pp);
  });

  m.def("load_flow_graph_pp",
        [](dflow::FlowGraph &flow_graph, const std::string &compile_config_path, const std::string &name) {
          dflow::FlowGraphPp flow_graph_pp{name.data(), [flow_graph]() {
            return flow_graph;
          }};
          (void)flow_graph_pp.SetCompileConfig(compile_config_path.data());
          ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
          return std::make_tuple(return_msg, flow_graph_pp);
        });
}

void BindFlowGraph(py::module &m) {
  py::class_<Operator>(m, "Operator")
      .def("set_attr", overload_cast_<const char *, bool>()(&Operator::SetAttr))
      .def("set_attr", overload_cast_<const char *, int64_t>()(&Operator::SetAttr))
      .def("set_attr", overload_cast_<const std::string &, const std::string &>()(&Operator::SetAttr));

  py::class_<dflow::FlowOperator, Operator>(m, "FlowOperator");

  py::class_<dflow::FlowData, dflow::FlowOperator>(m, "FlowData").def(py::init<const char *, int64_t>());

  py::class_<dflow::FlowNode, dflow::FlowOperator>(m, "FlowNode")
      .def(py::init<const char *, uint32_t, uint32_t>())
      .def("set_input", &dflow::FlowNode::SetInput)
      .def("add_pp", &dflow::FlowNode::AddPp)
      .def("map_input", &dflow::FlowNode::MapInput)
      .def("map_output", &dflow::FlowNode::MapOutput)
      .def("set_balance_scatter", &dflow::FlowNode::SetBalanceScatter)
      .def("set_balance_gather", &dflow::FlowNode::SetBalanceGather);

  py::class_<dflow::FlowGraph>(m, "FlowGraph")
      .def(py::init<const char *>())
      .def("set_inputs", &dflow::FlowGraph::SetInputs)
      .def("set_outputs", overload_cast_<const std::vector<dflow::FlowOperator> &>()(&dflow::FlowGraph::SetOutputs))
      .def("set_outputs", overload_cast_<const std::vector<std::pair<dflow::FlowOperator, std::vector<size_t>>> &>()(
               &dflow::FlowGraph::SetOutputs))
      .def("set_contains_n_mapping_node", &dflow::FlowGraph::SetContainsNMappingNode)
      .def("set_inputs_align_attrs", &dflow::FlowGraph::SetInputsAlignAttrs)
      .def("set_exception_catch", &dflow::FlowGraph::SetExceptionCatch)
      .def("set_graphpp_builder_async", &dflow::FlowGraph::SetGraphPpBuilderAsync);
}

void BindFlowInfo(py::module &m) {
  py::class_<FlowInfo>(m, "FlowInfo")
      .def(py::init())
      .def_readwrite("start_time", &FlowInfo::start_time)
      .def_readwrite("end_time", &FlowInfo::end_time)
      .def_readwrite("flow_flags", &FlowInfo::flow_flags)
      .def_readwrite("transaction_id", &FlowInfo::transaction_id)
      .def("set_user_data", [](FlowInfo &self, const py::buffer &user_data, size_t data_size, size_t offset) {
        self.user_data.user_data_ptr = user_data.request().ptr;
        self.user_data.data_size = data_size;
        self.user_data.offset = offset;
      });
}

ge::Tensor CreateTensorFromNumpyArray(const py::array &np_array) {
  auto flags = static_cast<unsigned int>(np_array.flags());
  if ((flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) == 0) {
    throw std::runtime_error("Numpy array is not C Contiguous");
  }
  ge::DataType dtype = ge::DataType::DT_FLOAT;
  if (IsStringDataType(py::str(np_array.dtype()))) {
    dtype = ge::DataType::DT_STRING;
  } else {
    const auto ret_msg = ConvertNumpyDataTypeToGeDataType(np_array.dtype(), dtype);
    if (!ret_msg.empty()) {
      throw std::runtime_error(ret_msg);
    }
  }
  std::vector<int64_t> dims;
  dims.reserve(np_array.ndim());
  for (ssize_t i = 0; i < np_array.ndim(); ++i) {
    dims.emplace_back(np_array.shape(i));
  }
  ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, dtype);
  ge::Tensor tensor;
  tensor.SetTensorDesc(desc);

  if (dtype == ge::DataType::DT_STRING) {
    const int64_t shape_size = desc.GetShape().GetShapeSize();
    const size_t element_number = shape_size <= 0L ? 1UL : static_cast<size_t>(shape_size);
    const auto string_vec =
        SplitToStrVector(static_cast<const char *>(np_array.data()), np_array.nbytes(), element_number);
    if (string_vec.empty()) {
      throw std::runtime_error("Split string to vector failed.");
    }
    tensor.SetData(string_vec);
  } else {
    tensor.SetData(static_cast<const uint8_t *>(np_array.data()), np_array.nbytes());
  }
  return tensor;
}

std::vector<std::string> DflowGetTensorStringData(const ge::Tensor &tensor) {
  const int64_t shape_size = tensor.GetTensorDesc().GetShape().GetShapeSize();
  const size_t element_number = shape_size <= 0L ? 1UL : static_cast<size_t>(shape_size);
  if (wrapper::CheckInt64MulOverflow(element_number, static_cast<int64_t>(sizeof(DflowStringHead)))) {
    throw std::runtime_error("element number " + std::to_string(element_number) +
                             " mul DflowStringHead size " + std::to_string(sizeof(DflowStringHead)) +
                             " is overflow.");
  }
  uint64_t total_header_size = element_number * sizeof(DflowStringHead);
  if (total_header_size > tensor.GetSize()) {
    throw std::runtime_error("Total ptr size " + std::to_string(total_header_size) +
                             " is greater than data size " + std::to_string(tensor.GetSize()));
  }
  if (tensor.GetData() == nullptr) {
    throw std::runtime_error("Data tensor nullptr is invalid.");
  }
  std::vector<std::string> tensor_strs;
  for (size_t i = 0; i < element_number; ++i) {
    auto header = reinterpret_cast<const DflowStringHead *>(tensor.GetData()) + i;
    tensor_strs.emplace_back(reinterpret_cast<const char *>(tensor.GetData() + header->addr));
  }
  return tensor_strs;
}

void BindGeTensor(py::module &m) {
  py::class_<ge::Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init(&CreateTensorFromNumpyArray))
      .def("get_dtype", [](const ge::Tensor &self) {
        return self.GetTensorDesc().GetDataType();
      })
      .def("get_shape", [](const ge::Tensor &self) {
        return self.GetTensorDesc().GetShape().GetDims();
      })
      .def("clone", [](const ge::Tensor &self) {
        return self.Clone();
      })
      .def("get_string_tensor", &DflowGetTensorStringData)
      .def_buffer([](ge::Tensor &tensor) -> py::buffer_info {
        const auto tensor_desc = tensor.GetTensorDesc();
        const auto dtype = tensor_desc.GetDataType();
        auto const &format_descs = DFlowDataTypeManager::GetInstance().GetGeDtypeToFormatDesc();
        auto it = format_descs.find(dtype);
        if (it == format_descs.cend()) {
          throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int32_t>(dtype)));
        }
        const auto item_size = static_cast<ssize_t>(ge::GetSizeByDataType(dtype));
        const auto shape = tensor_desc.GetShape();
        const auto dims = shape.GetDims();
        std::vector<ssize_t> strides;
        const std::string err_msg = wrapper::ComputeStrides(item_size, dims, strides);
        if (!err_msg.empty()) {
          throw std::runtime_error(err_msg);
        }
        return py::buffer_info(tensor.GetData(), item_size, it->second, static_cast<ssize_t>(shape.GetDimNum()), dims,
                               strides);
      });
}

void BindFlowMsg(py::module &m) {
  py::class_<ge::FlowMsg, std::shared_ptr<ge::FlowMsg>, PyFlowMsg>(m, "FlowMsg")
      .def(py::init<>())
      .def("get_msg_type", &ge::FlowMsg::GetMsgType)
      .def("set_msg_type", [](ge::FlowMsg &self, uint16_t msg_type) {
        return self.SetMsgType(static_cast<ge::MsgType>(msg_type));
      })
      .def("get_tensor", &ge::FlowMsg::GetTensor, py::return_value_policy::reference)
      .def("get_raw_data", [](const ge::FlowMsg &self) {
        void *data = nullptr;
        uint64_t data_size = 0U;
        (void)self.GetRawData(data, data_size);
        return py::memoryview::from_memory(data, static_cast<ssize_t>(data_size), false);
      })
      .def("get_ret_code", &ge::FlowMsg::GetRetCode)
      .def("set_ret_code", &ge::FlowMsg::SetRetCode)
      .def("get_start_time", &ge::FlowMsg::GetStartTime)
      .def("set_start_time", &ge::FlowMsg::SetStartTime)
      .def("get_end_time", &ge::FlowMsg::GetEndTime)
      .def("set_end_time", &ge::FlowMsg::SetEndTime)
      .def("get_flow_flags", &ge::FlowMsg::GetFlowFlags)
      .def("set_flow_flags", &ge::FlowMsg::SetFlowFlags)
      .def("get_transaction_id", &ge::FlowMsg::GetTransactionId)
      .def("set_transaction_id", &ge::FlowMsg::SetTransactionId)
      .def("__repr__", [](const ge::FlowMsg &self) {
        std::stringstream repr;
        repr << "FlowMsg(msg_type=" << static_cast<int32_t>(self.GetMsgType());
        repr << ", tensor=...";
        repr << ", ret_code=" << self.GetRetCode();
        repr << ", start_time=" << self.GetStartTime();
        repr << ", end_time=" << self.GetEndTime();
        repr << ", transaction_id=" << self.GetTransactionId();
        repr << ", flow_flags=" << self.GetFlowFlags() << ")";
        return repr.str();
      });
}

void BindFlowBufferFactory(py::module &m) {
  py::class_<ge::FlowBufferFactory>(m, "FlowBufferFactory")
      .def_static("alloc_tensor_msg", &ge::FlowBufferFactory::AllocTensorMsg)
      .def_static("alloc_raw_data_msg", &ge::FlowBufferFactory::AllocRawDataMsg)
      .def_static("alloc_empty_data_msg", &ge::FlowBufferFactory::AllocEmptyDataMsg)
      .def_static("to_tensor_flow_msg", [](const ge::Tensor &tensor) {
        return ge::FlowBufferFactory::ToFlowMsg(tensor);
      })
      .def_static("to_raw_data_flow_msg", [](const py::buffer &buffer) {
        py::buffer_info info = buffer.request();
        ge::RawData raw_data{};
        raw_data.addr = static_cast<const void *>(info.ptr);
        raw_data.len = info.size;
        return ge::FlowBufferFactory::ToFlowMsg(raw_data);
      });
}

Status SetFlowInfoFromWrapper(DataFlowInfo &flow_info, const FlowInfo &info) {
  flow_info.SetStartTime(info.start_time);
  flow_info.SetEndTime(info.end_time);
  flow_info.SetFlowFlags(info.flow_flags);
  flow_info.SetTransactionId(info.transaction_id);
  if (info.user_data.data_size != 0UL) {
    return flow_info.SetUserData(info.user_data.user_data_ptr, info.user_data.data_size, info.user_data.offset);
  }
  return SUCCESS;
}

void SetFlowInfoToWrapper(const DataFlowInfo &flow_info, FlowInfo &info) {
  info.start_time = flow_info.GetStartTime();
  info.end_time = flow_info.GetEndTime();
  info.flow_flags = flow_info.GetFlowFlags();
  info.transaction_id = flow_info.GetTransactionId();
}

ReturnMessage ConstructErrorReturnMessage(ge::Status ret, const std::string &operation) {
  ReturnMessage return_msg{.ret_code = ret, .error_msg = ""};
  if (ret == ACL_ERROR_GE_SUBHEALTHY) {
    return_msg.error_msg = "Current system is in subhealth status.";
  } else {
    return_msg.error_msg = "Failed to " + operation + ", " + ERR_MSG;
  }
  return return_msg;
}

auto DflowFeedData(dflow::DFlowSession &self, uint32_t graph_id, const std::vector<uint32_t> &indexes,
                   const std::vector<ge::Tensor> &inputs, const FlowInfo &info, int32_t timeout) {
  DataFlowInfo flow_info;
  const auto set_ret = SetFlowInfoFromWrapper(flow_info, info);
  if (set_ret != SUCCESS) {
    return ConstructErrorReturnMessage(set_ret, "set user data");
  }
  const auto ret = self.FeedDataFlowGraph(graph_id, indexes, inputs, flow_info, timeout);
  if ((ret != ge::SUCCESS)) {
    return ConstructErrorReturnMessage(ret, "feed data");
  }
  ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
  return return_msg;
}

auto DflowAddFlowGraph(dflow::DFlowSession &self, uint32_t graph_id,
                       const dflow::FlowGraph &flow_graph,
                       const std::map<std::string, std::string> &options) {
  auto options_ascend_string = ConvertToAscendString(options);
  const auto ret = self.AddGraph(graph_id, flow_graph, options_ascend_string);
  if (ret != SUCCESS) {
    return ConstructErrorReturnMessage(ret, "add flow graph");
  }
  ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
  return return_msg;
}

auto DflowFetchData(dflow::DFlowSession &self, uint32_t graph_id, const std::vector<uint32_t> &indexes, int32_t timeout,
                    const py::buffer &user_data) {
  const size_t user_data_size = user_data.request().size;
  ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
  std::vector<ge::Tensor> outputs;
  FlowInfo info;
  if (user_data_size > kMaxUserDataSize) {
    return_msg.ret_code = ACL_ERROR_GE_PARAM_INVALID;
    return_msg.error_msg = "The size of user data is greater than limit value." + ERR_MSG;
    return std::make_tuple(return_msg, outputs, info);
  }
  ge::DataFlowInfo flow_info;
  const auto ret = self.FetchDataFlowGraph(graph_id, indexes, outputs, flow_info, timeout);
  SetFlowInfoToWrapper(flow_info, info);
  if (user_data_size > 0) {
    (void)flow_info.GetUserData(user_data.request().ptr, user_data_size);
  }
  if ((ret != ge::SUCCESS)) {
    return_msg = ConstructErrorReturnMessage(ret, "fetch data");
  }
  return std::make_tuple(return_msg, outputs, info);
}

auto DflowFeedFlowMsg(dflow::DFlowSession &self, uint32_t graph_id, const std::vector<uint32_t> &indexes,
                      const std::vector<ge::FlowMsgPtr> &inputs, int32_t timeout) {
  const auto ret = self.FeedDataFlowGraph(graph_id, indexes, inputs, timeout);
  if ((ret != ge::SUCCESS)) {
    return ConstructErrorReturnMessage(ret, "feed flow msg");
  }
  ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
  return return_msg;
}

auto DflowFetchFlowMsg(dflow::DFlowSession &self, uint32_t graph_id, const std::vector<uint32_t> &indexes,
                       int32_t timeout) {
  ReturnMessage return_msg = {.ret_code = ge::SUCCESS, .error_msg = "success"};
  std::vector<ge::FlowMsgPtr> outputs;
  const auto ret = self.FetchDataFlowGraph(graph_id, indexes, outputs, timeout);
  if ((ret != ge::SUCCESS)) {
    return_msg = ConstructErrorReturnMessage(ret, "fetch flow msg");
  }
  return std::make_tuple(return_msg, outputs);
}

void BindDFlowSession(py::module &m) {
  py::class_<dflow::DFlowSession>(m, "DFlowSession")
      .def(py::init([](const std::map<std::string, std::string> &options) {
        auto options_ascend_string = ConvertToAscendString(options);
        // no need add std::nothrow, as it need raise exception to python
        return new dflow::DFlowSession(options_ascend_string);
      }), py::return_value_policy::take_ownership)
      .def("add_flow_graph", &DflowAddFlowGraph, py::call_guard<py::gil_scoped_release>())
      .def("feed_data", &DflowFeedData, py::call_guard<py::gil_scoped_release>())
      .def("feed_flow_msg", &DflowFeedFlowMsg, py::call_guard<py::gil_scoped_release>())
      .def("fetch_data", &DflowFetchData, py::call_guard<py::gil_scoped_release>())
      .def("fetch_flow_msg", &DflowFetchFlowMsg, py::call_guard<py::gil_scoped_release>());
}
} // namespace

PYBIND11_MODULE(dflow_wrapper, m) {
  BindDflowAttr(m);
  BindDflowEnum(m);
  BindDflowInitAndFinalize(m);
  BindReturnMessage(m);
  BindProcessPoint(m);
  BindLoadPp(m);
  BindFlowGraph(m);
  BindFlowInfo(m);
  BindGeTensor(m);
  BindFlowMsg(m);
  BindFlowBufferFactory(m);
  BindDFlowSession(m);

  m.def("init_datatype_manager",
        [](const std::map<ge::DataType, py::array> &type_map) {
          DFlowDataTypeManager::GetInstance().Init(type_map);
        });

  m.def("get_dflow_pybind11_build_abi", []() {
#ifdef PYBIND11_BUILD_ABI
    return PYBIND11_BUILD_ABI;
#else
    return "";
#endif
  });
}
}
