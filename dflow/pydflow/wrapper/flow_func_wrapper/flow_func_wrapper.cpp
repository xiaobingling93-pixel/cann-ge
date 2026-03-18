/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/detail/common.h>
#include <sys/syscall.h>
#include "utils.h"
#include "flow_func/ascend_string.h"
#include "flow_func/attr_value.h"
#include "flow_func/flow_func_defines.h"
#include "flow_func/flow_func_log.h"
#include "flow_func/flow_msg.h"
#include "flow_func/flow_msg_queue.h"
#include "flow_func/meta_multi_func.h"
#include "flow_func/meta_run_context.h"
#include "flow_func/meta_params.h"
#include "flow_func/tensor_data_type.h"
#include "graph/tensor.h"
#include "dlog_pub.h"

namespace {
namespace py = pybind11;

inline int64_t GetTid() {
  thread_local static const int64_t tid = static_cast<int64_t>(syscall(__NR_gettid));
  return tid;
}

constexpr int32_t kModuleIdUdf = static_cast<int32_t>(UDF);
#define UDF_LOG_ERROR(fmt, ...) dlog_error(kModuleIdUdf, "[%s][tid:%ld]: " fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

const std::map<FlowFunc::TensorDataType, ge::DataType> flow_func_dtype_to_ge_dtype{
    {FlowFunc::TensorDataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {FlowFunc::TensorDataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {FlowFunc::TensorDataType::DT_INT8, ge::DataType::DT_INT8},
    {FlowFunc::TensorDataType::DT_INT16, ge::DataType::DT_INT16},
    {FlowFunc::TensorDataType::DT_UINT16, ge::DataType::DT_UINT16},
    {FlowFunc::TensorDataType::DT_UINT8, ge::DataType::DT_UINT8},
    {FlowFunc::TensorDataType::DT_INT32, ge::DataType::DT_INT32},
    {FlowFunc::TensorDataType::DT_INT64, ge::DataType::DT_INT64},
    {FlowFunc::TensorDataType::DT_UINT32, ge::DataType::DT_UINT32},
    {FlowFunc::TensorDataType::DT_UINT64, ge::DataType::DT_UINT64},
    {FlowFunc::TensorDataType::DT_BOOL, ge::DataType::DT_BOOL},
    {FlowFunc::TensorDataType::DT_DOUBLE, ge::DataType::DT_DOUBLE},
    {FlowFunc::TensorDataType::DT_QINT8, ge::DataType::DT_QINT8},
    {FlowFunc::TensorDataType::DT_QINT16, ge::DataType::DT_QINT16},
    {FlowFunc::TensorDataType::DT_QINT32, ge::DataType::DT_QINT32},
    {FlowFunc::TensorDataType::DT_QUINT8, ge::DataType::DT_QUINT8},
    {FlowFunc::TensorDataType::DT_QUINT16, ge::DataType::DT_QUINT16},
    {FlowFunc::TensorDataType::DT_DUAL, ge::DataType::DT_DUAL},
    {FlowFunc::TensorDataType::DT_INT4, ge::DataType::DT_INT4},
    {FlowFunc::TensorDataType::DT_UINT1, ge::DataType::DT_UINT1},
    {FlowFunc::TensorDataType::DT_INT2, ge::DataType::DT_INT2},
    {FlowFunc::TensorDataType::DT_UINT2, ge::DataType::DT_UINT2},
    {FlowFunc::TensorDataType::DT_UNDEFINED, ge::DataType::DT_UNDEFINED}};

ge::DataType TransFuncDataTypeToGeDataType(const FlowFunc::TensorDataType &data_type) {
  const auto iter = flow_func_dtype_to_ge_dtype.find(data_type);
  if (iter == flow_func_dtype_to_ge_dtype.cend()) {
    return ge::DataType::DT_UNDEFINED;
  }
  return iter->second;
}

const std::map<ge::DataType, FlowFunc::TensorDataType> ge_dtype_to_flow_func_dtype{
    {ge::DataType::DT_FLOAT, FlowFunc::TensorDataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, FlowFunc::TensorDataType::DT_FLOAT16},
    {ge::DataType::DT_INT8, FlowFunc::TensorDataType::DT_INT8},
    {ge::DataType::DT_INT16, FlowFunc::TensorDataType::DT_INT16},
    {ge::DataType::DT_UINT16, FlowFunc::TensorDataType::DT_UINT16},
    {ge::DataType::DT_UINT8, FlowFunc::TensorDataType::DT_UINT8},
    {ge::DataType::DT_INT32, FlowFunc::TensorDataType::DT_INT32},
    {ge::DataType::DT_INT64, FlowFunc::TensorDataType::DT_INT64},
    {ge::DataType::DT_UINT32, FlowFunc::TensorDataType::DT_UINT32},
    {ge::DataType::DT_UINT64, FlowFunc::TensorDataType::DT_UINT64},
    {ge::DataType::DT_BOOL, FlowFunc::TensorDataType::DT_BOOL},
    {ge::DataType::DT_DOUBLE, FlowFunc::TensorDataType::DT_DOUBLE},
    {ge::DataType::DT_QINT8, FlowFunc::TensorDataType::DT_QINT8},
    {ge::DataType::DT_QINT16, FlowFunc::TensorDataType::DT_QINT16},
    {ge::DataType::DT_QINT32, FlowFunc::TensorDataType::DT_QINT32},
    {ge::DataType::DT_QUINT8, FlowFunc::TensorDataType::DT_QUINT8},
    {ge::DataType::DT_QUINT16, FlowFunc::TensorDataType::DT_QUINT16},
    {ge::DataType::DT_DUAL, FlowFunc::TensorDataType::DT_DUAL},
    {ge::DataType::DT_INT4, FlowFunc::TensorDataType::DT_INT4},
    {ge::DataType::DT_UINT1, FlowFunc::TensorDataType::DT_UINT1},
    {ge::DataType::DT_INT2, FlowFunc::TensorDataType::DT_INT2},
    {ge::DataType::DT_UINT2, FlowFunc::TensorDataType::DT_UINT2},
    {ge::DataType::DT_UNDEFINED, FlowFunc::TensorDataType::DT_UNDEFINED}};

FlowFunc::TensorDataType TransGeDataTypeToFuncDataType(const ge::DataType &data_type) {
  const auto iter = ge_dtype_to_flow_func_dtype.find(data_type);
  if (iter == ge_dtype_to_flow_func_dtype.cend()) {
    return FlowFunc::TensorDataType::DT_UNDEFINED;
  }
  return iter->second;
}

py::memoryview ToReadonlyMemoryView(const FlowFunc::Tensor &tensor) {
  const auto element_size = tensor.GetDataSize() / tensor.GetElementCnt();
  std::vector<uint64_t> strides;
  strides.push_back(element_size);
  auto stride_num = tensor.GetShape().size();
  if (stride_num > 1) {
    uint64_t stride = element_size;
    for (auto it = tensor.GetShape().rbegin(); it != tensor.GetShape().rend() && strides.size() < stride_num; ++it) {
      stride *= (*it);
      strides.push_back(stride);
    }
    std::reverse(strides.begin(), strides.end());
  }
  switch (tensor.GetDataType()) {
    case FlowFunc::TensorDataType::DT_BOOL: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<bool>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_INT8: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<int8_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_UINT8: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<uint8_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_INT16: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<int16_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_UINT16: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<uint16_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_INT32: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<int32_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_UINT32: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<uint32_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_INT64: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<int64_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_UINT64: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<uint64_t>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_FLOAT16:
    case FlowFunc::TensorDataType::DT_FLOAT: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<float>::value,
                                         tensor.GetShape(), strides, true);
    }
    case FlowFunc::TensorDataType::DT_DOUBLE: {
      return py::memoryview::from_buffer(tensor.GetData(), element_size, py::format_descriptor<double>::value,
                                         tensor.GetShape(), strides, true);
    }
    default: {
      return py::memoryview::from_memory(tensor.GetData(), tensor.GetDataSize(), true);
    }
  }
}

class PyFlowFuncLogger : public FlowFunc::FlowFuncLogger {
 public:
  bool IsLogEnable([[maybe_unused]] FlowFunc::FlowFuncLogLevel level) override {
    PYBIND11_OVERRIDE_PURE(bool, FlowFunc::FlowFuncLogger, IsLogEnable, );
  }
  void Error([[maybe_unused]] const char *fmt, ...) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Error, );
  }
  void Warn([[maybe_unused]] const char *fmt, ...) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Warn, );
  }
  void Info([[maybe_unused]] const char *fmt, ...) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Info, );
  }
  void Debug([[maybe_unused]] const char *fmt, ...) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Debug, );
  }
  void ErrorLog(const char *location_message, const char *user_message) const {
    auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
    logger.Error("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }
  void WarnLog(const char *location_message, const char *user_message) const {
    auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
    logger.Warn("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }
  void InfoLog(const char *location_message, const char *user_message) const {
    auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
    logger.Info("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }
  void DebugLog(const char *location_message, const char *user_message) const {
    auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
    logger.Debug("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }
  void RunTypeError(const char *location_message, const char *user_message) const {
    auto &logger = GetLogger(FlowFunc::FlowFuncLogType::RUN_LOG);
    logger.Error("%s%s[RUN]: %s", location_message, GetLogExtHeader(), user_message);
  }
  void RunTypeInfo(const char *location_message, const char *user_message) const {
    auto &logger = GetLogger(FlowFunc::FlowFuncLogType::RUN_LOG);
    logger.Info("%s%s[RUN]: %s", location_message, GetLogExtHeader(), user_message);
  }
};

class PyFlowMsg : public FlowFunc::FlowMsg {
 public:
  FlowFunc::MsgType GetMsgType() const override {
    PYBIND11_OVERRIDE_PURE(FlowFunc::MsgType, FlowFunc::FlowMsg, GetMsgType, );
  }

  FlowFunc::Tensor *GetTensor() const override {
    PYBIND11_OVERRIDE_PURE(FlowFunc::Tensor *, FlowFunc::FlowMsg, GetTensor, );
  }

  int32_t GetRetCode() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowMsg, GetRetCode, );
  }

  void SetRetCode(int32_t ret_code) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetRetCode, ret_code);
  }

  void SetStartTime(uint64_t start_time) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetStartTime, start_time);
  }

  uint64_t GetStartTime() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::FlowMsg, GetStartTime, );
  }

  void SetEndTime(uint64_t end_time) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetEndTime, end_time);
  }

  uint64_t GetEndTime() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::FlowMsg, GetEndTime, );
  }

  void SetFlowFlags(uint32_t flags) override {
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetFlowFlags, flags);
  }

  uint32_t GetFlowFlags() const override {
    PYBIND11_OVERRIDE_PURE(uint32_t, FlowFunc::FlowMsg, GetFlowFlags, );
  }

  void SetRouteLabel(uint32_t route_label) override {
    (void)route_label;
    PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetRouteLabel, );
  }

  uint64_t GetTransactionId() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::FlowMsg, GetTransactionId, );
  }
};

class PyFlowMsgQueue : public FlowFunc::FlowMsgQueue {
 public:
  int32_t Dequeue(std::shared_ptr<FlowFunc::FlowMsg> &flow_msg, [[maybe_unused]] int32_t timeout) override {
    (void)flow_msg;
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::FlowMsgQueue, Dequeue, );
  }

  int32_t Depth() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::FlowMsgQueue, Depth, );
  }

  int32_t Size() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::FlowMsgQueue, Size, );
  }
};

class PyTensor : public FlowFunc::Tensor {
 public:
  const std::vector<int64_t> &GetShape() const override {
    PYBIND11_OVERRIDE_PURE(const std::vector<int64_t> &, FlowFunc::Tensor, GetShape, );
  }

  FlowFunc::TensorDataType GetDataType() const override {
    PYBIND11_OVERRIDE_PURE(FlowFunc::TensorDataType, FlowFunc::Tensor, GetDataType, );
  }

  void *GetData() const override {
    PYBIND11_OVERRIDE_PURE(void *, FlowFunc::Tensor, GetData, );
  }

  uint64_t GetDataSize() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::Tensor, GetDataSize, );
  }

  int64_t GetElementCnt() const override {
    PYBIND11_OVERRIDE_PURE(int64_t, FlowFunc::Tensor, GetElementCnt, );
  }

  int32_t Reshape([[maybe_unused]] const std::vector<int64_t> &shape) override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::Tensor, Reshape, );
  }
};

class PyMetaParams : public FlowFunc::MetaParams {
 public:
  const char *GetName() const override {
    PYBIND11_OVERRIDE_PURE(char *, FlowFunc::MetaParams, GetName, );
  }

  std::shared_ptr<const FlowFunc::AttrValue> GetAttr(const char *attr_name) const override {
    (void)attr_name;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<const FlowFunc::AttrValue>, FlowFunc::MetaParams, GetAttr, );
  }

  size_t GetInputNum() const override {
    PYBIND11_OVERRIDE_PURE(size_t, FlowFunc::MetaParams, GetInputNum, );
  }

  size_t GetOutputNum() const override {
    PYBIND11_OVERRIDE_PURE(size_t, FlowFunc::MetaParams, GetOutputNum, );
  }
  const char *GetWorkPath() const override {
    PYBIND11_OVERRIDE_PURE(char *, FlowFunc::MetaParams, GetWorkPath, );
  }

  int32_t GetRunningDeviceId() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaParams, GetRunningDeviceId, );
  }

  int32_t GetRunningInstanceId() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaParams, GetRunningInstanceId, );
  }

  int32_t GetRunningInstanceNum() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaParams, GetRunningInstanceNum, );
  }
};

class PyMetaRunContext : public FlowFunc::MetaRunContext {
 public:
  std::shared_ptr<FlowFunc::FlowMsg> AllocTensorMsg([[maybe_unused]] const std::vector<int64_t> &shape,
                                                    FlowFunc::TensorDataType data_type) override {
    (void)data_type;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowFunc::FlowMsg>, FlowFunc::MetaRunContext, AllocTensorMsg, );
  }

  std::shared_ptr<FlowFunc::FlowMsg> AllocTensorMsgWithAlign([[maybe_unused]] const std::vector<int64_t> &shape,
                                                             [[maybe_unused]] FlowFunc::TensorDataType data_type,
                                                             [[maybe_unused]] uint32_t align) override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowFunc::FlowMsg>, FlowFunc::MetaRunContext, AllocTensorMsg, );
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowFunc::FlowMsg> out_msg) override {
    (void)out_idx;
    (void)out_msg;
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, SetOutput, );
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowFunc::FlowMsg> out_msg,
                    const FlowFunc::OutOptions &options) override {
    (void)out_idx;
    (void)out_msg;
    (void)options;
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, SetOutput, );
  }

  int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &out_msgs,
                          const FlowFunc::OutOptions &options) override {
    (void)out_msgs;
    (void)out_idx;
    (void)options;
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, SetMultiOutputs, );
  }

  std::shared_ptr<FlowFunc::FlowMsg> AllocEmptyDataMsg(FlowFunc::MsgType msg_type) override {
    (void)msg_type;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowFunc::FlowMsg>, FlowFunc::MetaRunContext, AllocEmptyDataMsg, );
  }

  int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &input_msgs,
                       std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &output_msgs,
                       [[maybe_unused]] int32_t timeout) override {
    (void)model_key;
    (void)input_msgs;
    (void)output_msgs;
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, RunFlowModel, );
  }

  int32_t GetUserData([[maybe_unused]] void *data, [[maybe_unused]] size_t size,
                      [[maybe_unused]] size_t offset = 0U) const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, GetUserData);
  }
};

class FuncDataTypeManager {
 public:
  static FuncDataTypeManager &GetInstance() {
    static FuncDataTypeManager data_type_manager;
    return data_type_manager;
  }

  void Init(const std::map<FlowFunc::TensorDataType, py::array> &type_map) {
    for (const auto &item : type_map) {
      auto const dtype = item.first;
      auto const array = item.second;
      const auto buff = array.request();
      func_dtype_to_format_desc_[dtype] = buff.format;
    }
  }

  const std::map<FlowFunc::TensorDataType, std::string> &GetFlowFuncDtypeToFormatDesc() const {
    return func_dtype_to_format_desc_;
  }

 private:
  FuncDataTypeManager() = default;
  std::map<FlowFunc::TensorDataType, std::string> func_dtype_to_format_desc_;
};

class PyBalanceConfig {
 public:
  void SetAffinityPolicy(FlowFunc::AffinityPolicy policy) {
    policy_ = policy;
  }

  void SetBalanceWeight(int32_t row_num, int32_t col_num) {
    row_num_ = row_num;
    col_num_ = col_num;
  }

  void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &data_pos) {
    data_pos_ = data_pos;
  }

  FlowFunc::AffinityPolicy GetAffinityPolicy() const {
    return policy_;
  }

  int32_t GetRowNum() const {
    return row_num_;
  }

  int32_t GetColNum() const {
    return col_num_;
  }

  std::vector<std::pair<int32_t, int32_t>> GetDataPos() const {
    return data_pos_;
  }

 private:
  FlowFunc::AffinityPolicy policy_;
  int32_t row_num_ = 0;
  int32_t col_num_ = 0;
  std::vector<std::pair<int32_t, int32_t>> data_pos_;
};
constexpr uint32_t kMaxDimSize = 32U;
struct RuntimeTensorDesc {
  uint64_t data_addr;
  int64_t data_offset_size;
  int64_t dtype;
  int64_t shape[kMaxDimSize + 1U];
  int64_t original_shape[kMaxDimSize + 1U];
  int64_t format;
  int64_t sub_format;
  uint64_t data_size;
  uint8_t reserved[448];  // padding to 1024 bytes.
};

class RuntimeTensorDescMsgProcessor {
 public:
  static int32_t GetRuntimeTensorDescs(const std::shared_ptr<FlowFunc::FlowMsg> &input_flow_msg,
                                       std::vector<RuntimeTensorDesc> &runtime_tensor_descs, int64_t input_num) {
    void *data_ptr = nullptr;
    uint64_t data_size = 0UL;
    auto ret = input_flow_msg->GetRawData(data_ptr, data_size);
    if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
      UDF_LOG_ERROR("Failed to get raw data, ret = %d.", ret);
      return ret;
    }
    uint64_t offset = 0UL;
    for (int64_t i = 0; i < input_num; ++i) {
      if (data_size - offset < sizeof(RuntimeTensorDesc)) {
        UDF_LOG_ERROR("Failed to check flow msg size, data size = %lu, input num = %ld.", data_size, input_num);
        return FlowFunc::FLOW_FUNC_ERR_PARAM_INVALID;
      }
      auto desc = reinterpret_cast<RuntimeTensorDesc *>(static_cast<uint8_t *>(data_ptr) + offset);
      runtime_tensor_descs.emplace_back(*desc);
      offset += sizeof(RuntimeTensorDesc);
    }
    return FlowFunc::FLOW_FUNC_SUCCESS;
  }

  static std::shared_ptr<FlowFunc::FlowMsg> CreateRuntimeTensorDescMsg(
      const std::shared_ptr<FlowFunc::MetaRunContext> &run_context,
      const std::vector<RuntimeTensorDesc> &runtime_tensor_descs) {
    size_t size = runtime_tensor_descs.size() * sizeof(RuntimeTensorDesc);
    auto msg = run_context->AllocRawDataMsg(size);
    if (msg == nullptr) {
      UDF_LOG_ERROR("Failed to allocate raw data msg, size=%zu.", size);
      return nullptr;
    }

    void *data_ptr = nullptr;
    uint64_t data_size = 0UL;
    auto ret = msg->GetRawData(data_ptr, data_size);
    if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
      UDF_LOG_ERROR("Failed to get raw data, ret = %d.", ret);
      return nullptr;
    }
    if (size > data_size) {
      UDF_LOG_ERROR("Failed to check flow msg size, alloc data size = %lu, but get data size = %lu.", size, data_size);
      return nullptr;
    }

    uint64_t offset = 0UL;
    for (const auto &runtime_tensor_desc : runtime_tensor_descs) {
      auto desc = reinterpret_cast<RuntimeTensorDesc *>(static_cast<uint8_t *>(data_ptr) + offset);
      *desc = runtime_tensor_desc;
      offset += sizeof(RuntimeTensorDesc);
    }
    return msg;
  }
};
}  // namespace

PYBIND11_MODULE(flowfunc_wrapper, m) {
  m.doc() = "pybind11 flowfunc_wrapper plugin";  // optional module docstring
  m.attr("FLOW_FUNC_SUCCESS") = FlowFunc::FLOW_FUNC_SUCCESS;
  m.attr("FLOW_FUNC_FAILED") = FlowFunc::FLOW_FUNC_FAILED;
  m.attr("FLOW_FUNC_ERR_PARAM_INVALID") = FlowFunc::FLOW_FUNC_ERR_PARAM_INVALID;
  m.attr("FLOW_FUNC_ERR_ATTR_NOT_EXITS") = FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS;
  m.attr("FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH") = FlowFunc::FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
  m.attr("FLOW_FUNC_ERR_TIME_OUT_ERROR") = FlowFunc::FLOW_FUNC_ERR_TIME_OUT_ERROR;
  m.attr("FLOW_FUNC_STATUS_REDEPLOYING") = FlowFunc::FLOW_FUNC_STATUS_REDEPLOYING;
  m.attr("FLOW_FUNC_STATUS_EXIT") = FlowFunc::FLOW_FUNC_STATUS_EXIT;
  m.attr("FLOW_FUNC_ERR_DRV_ERROR") = FlowFunc::FLOW_FUNC_ERR_DRV_ERROR;
  m.attr("FLOW_FUNC_ERR_QUEUE_ERROR") = FlowFunc::FLOW_FUNC_ERR_QUEUE_ERROR;
  m.attr("FLOW_FUNC_ERR_MEM_BUF_ERROR") = FlowFunc::FLOW_FUNC_ERR_MEM_BUF_ERROR;
  m.attr("FLOW_FUNC_ERR_EVENT_ERROR") = FlowFunc::FLOW_FUNC_ERR_EVENT_ERROR;
  m.attr("FLOW_FUNC_ERR_USER_DEFINE_START") = FlowFunc::FLOW_FUNC_ERR_USER_DEFINE_START;
  m.attr("FLOW_FUNC_ERR_USER_DEFINE_END") = FlowFunc::FLOW_FUNC_ERR_USER_DEFINE_END;
  py::enum_<FlowFunc::FlowFuncLogType>(m, "FlowFuncLogType", py::arithmetic())
      .value("DEBUG_LOG", FlowFunc::FlowFuncLogType::DEBUG_LOG)
      .value("RUN_LOG", FlowFunc::FlowFuncLogType::RUN_LOG)
      .export_values();
  py::enum_<FlowFunc::FlowFuncLogLevel>(m, "FlowFuncLogLevel", py::arithmetic())
      .value("DEBUG", FlowFunc::FlowFuncLogLevel::DEBUG)
      .value("INFO", FlowFunc::FlowFuncLogLevel::INFO)
      .value("WARN", FlowFunc::FlowFuncLogLevel::WARN)
      .value("ERROR", FlowFunc::FlowFuncLogLevel::ERROR)
      .export_values();
  py::enum_<FlowFunc::MsgType>(m, "MsgType", py::arithmetic())
      .value("MSG_TYPE_TENSOR_DATA", FlowFunc::MsgType::MSG_TYPE_TENSOR_DATA)
      .value("MSG_TYPE_RAW_MSG", FlowFunc::MsgType::MSG_TYPE_RAW_MSG)
      .value("MSG_TYPE_TORCH_TENSOR_MSG", static_cast<FlowFunc::MsgType>(1023))
      .value("MSG_TYPE_USER_DEFINE_START", static_cast<FlowFunc::MsgType>(1024))
      .value("MSG_TYPE_PICKLED_MSG", static_cast<FlowFunc::MsgType>(65535))
      .export_values();
  py::enum_<FlowFunc::FlowFlag>(m, "FlowFlag", py::arithmetic())
      .value("FLOW_FLAG_EOS", FlowFunc::FlowFlag::FLOW_FLAG_EOS)
      .value("FLOW_FLAG_SEG", FlowFunc::FlowFlag::FLOW_FLAG_SEG)
      .export_values();

  py::class_<FlowFunc::FlowFuncLogger, std::shared_ptr<FlowFunc::FlowFuncLogger>, PyFlowFuncLogger>(m, "FlowFuncLogger")
      .def(py::init())
      .def("get_log_header",
           [](FlowFunc::FlowFuncLogger &self) {
             const std::string log_header(self.GetLogExtHeader());
             return log_header;
           })
      .def("is_log_enable",
           [](FlowFunc::FlowFuncLogger &self, const FlowFunc::FlowFuncLogType &log_type,
              const FlowFunc::FlowFuncLogLevel &log_level) {
             FlowFunc::FlowFuncLogger &logger = self.GetLogger(log_type);
             return logger.IsLogEnable(log_level);
           })
      .def("debug_log_error", [](PyFlowFuncLogger &self, const char *location_message,
                                 const char *user_message) { self.ErrorLog(location_message, user_message); })
      .def("debug_log_info", [](PyFlowFuncLogger &self, const char *location_message,
                                const char *user_message) { self.InfoLog(location_message, user_message); })
      .def("debug_log_warn", [](PyFlowFuncLogger &self, const char *location_message,
                                const char *user_message) { self.WarnLog(location_message, user_message); })
      .def("debug_log_debug", [](PyFlowFuncLogger &self, const char *location_message,
                                 const char *user_message) { self.DebugLog(location_message, user_message); })
      .def("run_log_error", [](PyFlowFuncLogger &self, const char *location_message,
                               const char *user_message) { self.RunTypeError(location_message, user_message); })
      .def("run_log_info", [](PyFlowFuncLogger &self, const char *location_message,
                              const char *user_message) { self.RunTypeInfo(location_message, user_message); })
      .def("__repr__", [](FlowFunc::FlowFuncLogger &self) {
        std::stringstream repr;
        repr << "FlowFuncLogger(LogHeader=" << self.GetLogExtHeader() << ")";
        return repr.str();
      });

  py::class_<FlowFunc::FlowMsg, std::shared_ptr<FlowFunc::FlowMsg>, PyFlowMsg>(m, "FlowMsg")
      .def(py::init<>())
      .def("get_msg_type", &FlowFunc::FlowMsg::GetMsgType)
      .def("set_msg_type", [](FlowFunc::FlowMsg &self,
                              uint16_t msg_type) { return self.SetMsgType(static_cast<FlowFunc::MsgType>(msg_type)); })
      .def("get_tensor", &FlowFunc::FlowMsg::GetTensor, py::return_value_policy::reference)
      .def("get_raw_data",
           [](FlowFunc::FlowMsg &self) {
             void *data = nullptr;
             uint64_t data_size = 0U;
             (void)self.GetRawData(data, data_size);
             return py::memoryview::from_memory(data, data_size, false);
           })
      .def("get_ret_code", &FlowFunc::FlowMsg::GetRetCode)
      .def("set_ret_code", &FlowFunc::FlowMsg::SetRetCode)
      .def("get_start_time", &FlowFunc::FlowMsg::GetStartTime)
      .def("set_start_time", &FlowFunc::FlowMsg::SetStartTime)
      .def("get_end_time", &FlowFunc::FlowMsg::GetEndTime)
      .def("set_end_time", &FlowFunc::FlowMsg::SetEndTime)
      .def("get_flow_flags", &FlowFunc::FlowMsg::GetFlowFlags)
      .def("set_flow_flags", &FlowFunc::FlowMsg::SetFlowFlags)
      .def("set_route_label", &FlowFunc::FlowMsg::SetRouteLabel)
      .def("get_transaction_id", &FlowFunc::FlowMsg::GetTransactionId)
      .def("set_transaction_id", &FlowFunc::FlowMsg::SetTransactionId)
      .def("__repr__", [](FlowFunc::FlowMsg &self) {
        std::stringstream repr;
        repr << "FlowMsg(msg_type=" << static_cast<int32_t>(self.GetMsgType());
        repr << ", tensor=...";
        repr << ", ret_code=" << self.GetRetCode();
        repr << ", start_time=" << self.GetStartTime();
        repr << ", end_time=" << self.GetEndTime();
        repr << ", flow_flags=" << self.GetFlowFlags() << ")";
        return repr.str();
      });

  py::class_<FlowFunc::FlowMsgQueue, std::shared_ptr<FlowFunc::FlowMsgQueue>, PyFlowMsgQueue>(m, "FlowMsgQueue")
      .def(py::init<>())
      .def(
          "dequeue",
          [](FlowFunc::FlowMsgQueue &self, int32_t timeout) {
            std::shared_ptr<FlowFunc::FlowMsg> flow_msg;
            const auto ret = self.Dequeue(flow_msg, timeout);
            return std::make_tuple(ret, flow_msg);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("depth", &FlowFunc::FlowMsgQueue::Depth)
      .def("size", &FlowFunc::FlowMsgQueue::Size);

  py::class_<FlowFunc::FlowBufferFactory>(m, "FlowBufferFactory")
      .def_static(
          "alloc_tensor",
          [](const std::vector<int64_t> &shapes, const ge::DataType &dtype, uint32_t align) {
            const auto func_dtype = TransGeDataTypeToFuncDataType(dtype);
            return FlowFunc::FlowBufferFactory::AllocTensor(shapes, func_dtype, align);
          },
          py::return_value_policy::reference);

  py::class_<FlowFunc::Tensor, std::shared_ptr<FlowFunc::Tensor>, PyTensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<>())
      .def("get_shape", &FlowFunc::Tensor::GetShape)
      .def("get_dtype",
           [](FlowFunc::Tensor &self) {
             const auto f_dtype = self.GetDataType();
             const auto ge_dtype = TransFuncDataTypeToGeDataType(f_dtype);
             return ge_dtype;
           })
      .def("get_data", &ToReadonlyMemoryView)
      .def(
          "get_writable_data",
          [](FlowFunc::Tensor &self) { return py::memoryview::from_memory(self.GetData(), self.GetDataSize(), false); })
      .def("get_data_size", &FlowFunc::Tensor::GetDataSize)
      .def("get_element_cnt", &FlowFunc::Tensor::GetElementCnt)
      .def("reshape", &FlowFunc::Tensor::Reshape)
      .def("__repr__",
           [](FlowFunc::Tensor &self) {
             std::stringstream repr;
             repr << "Tensor(shape=[";
             for (auto shape_item : self.GetShape()) {
               repr << shape_item << ", ";
             }
             repr << "], data_type=" << static_cast<int32_t>(self.GetDataType());
             repr << ", data_size=" << self.GetDataSize();
             repr << ", element_cnt=" << self.GetElementCnt();
             repr << ", data=...)";
             return repr.str();
           })
      .def_buffer([](const FlowFunc::Tensor &tensor) -> py::buffer_info {
        const auto dtype = tensor.GetDataType();
        auto const &format_descs = FuncDataTypeManager::GetInstance().GetFlowFuncDtypeToFormatDesc();
        const auto it = format_descs.find(dtype);
        if (it == format_descs.cend()) {
          throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int32_t>(dtype)));
        }
        const auto item_size = static_cast<ssize_t>((tensor.GetDataSize() / tensor.GetElementCnt()));
        const auto shape = tensor.GetShape();
        std::vector<ssize_t> strides;
        const std::string err_msg = wrapper::ComputeStrides(item_size, shape, strides);
        if (!err_msg.empty()) {
          throw std::runtime_error(err_msg);
        }
        return py::buffer_info(tensor.GetData(), item_size, it->second, static_cast<ssize_t>(shape.size()), shape,
                               strides);
      });

  py::class_<FlowFunc::MetaParams, std::shared_ptr<FlowFunc::MetaParams>, PyMetaParams>(m, "MetaParams")
      .def(py::init<>())
      .def("get_name", &FlowFunc::MetaParams::GetName)
      .def("get_int64",
           [](FlowFunc::MetaParams &self, const char *name) {
             int64_t value = -1L;
             const auto ret = self.GetAttr<int64_t>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_int64_vector",
           [](FlowFunc::MetaParams &self, const char *name) {
             std::vector<int64_t> value;
             const auto ret = self.GetAttr<std::vector<int64_t>>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_int64_vector_vector",
           [](FlowFunc::MetaParams &self, const char *name) {
             std::vector<std::vector<int64_t>> value;
             const auto ret = self.GetAttr<std::vector<std::vector<int64_t>>>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_bool",
           [](FlowFunc::MetaParams &self, const char *name) {
             bool value;
             const auto ret = self.GetAttr<bool>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_bool_list",
           [](FlowFunc::MetaParams &self, const char *name) {
             std::vector<bool> value;
             const auto ret = self.GetAttr<std::vector<bool>>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_float",
           [](FlowFunc::MetaParams &self, const char *name) {
             float value;
             const auto ret = self.GetAttr<float>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_float_list",
           [](FlowFunc::MetaParams &self, const char *name) {
             std::vector<float> value;
             const auto ret = self.GetAttr<std::vector<float>>(name, value);
             if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
             }
             return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
           })
      .def("get_tensor_dtype",
           [](FlowFunc::MetaParams &self, const char *name) {
             FlowFunc::TensorDataType value;
             const auto ret = self.GetAttr<FlowFunc::TensorDataType>(name, value);
             if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(ret, ge::DataType::DT_UNDEFINED);
             }
             return std::make_tuple(ret, TransFuncDataTypeToGeDataType(value));
           })
      .def("get_tensor_dtype_list",
           [](FlowFunc::MetaParams &self, const char *name) {
             std::vector<FlowFunc::TensorDataType> value;
             std::vector<ge::DataType> GeDType;
             const auto ret = self.GetAttr<std::vector<FlowFunc::TensorDataType>>(name, value);
             if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(ret, GeDType);
             }
             for (const auto &f_dtype : value) {
               GeDType.emplace_back(TransFuncDataTypeToGeDataType(f_dtype));
             }
             return std::make_tuple(ret, GeDType);
           })
      .def("get_string",
           [](FlowFunc::MetaParams &self, const char *name) {
             FlowFunc::AscendString value;
             const auto ret = self.GetAttr<FlowFunc::AscendString>(name, value);
             if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(ret, "");
             }
             return std::make_tuple(ret, value.GetString());
           })
      .def("get_string_list",
           [](FlowFunc::MetaParams &self, const char *name) {
             std::vector<FlowFunc::AscendString> value;
             std::vector<std::string> StrList;
             const auto ret = self.GetAttr<std::vector<FlowFunc::AscendString>>(name, value);
             if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
               return std::make_tuple(ret, StrList);
             }
             for (const auto &str : value) {
               StrList.emplace_back(str.GetString());
             }
             return std::make_tuple(ret, StrList);
           })
      .def("get_input_number", &FlowFunc::MetaParams::GetInputNum)
      .def("get_output_number", &FlowFunc::MetaParams::GetOutputNum)
      .def("get_work_path", &FlowFunc::MetaParams::GetWorkPath)
      .def("get_running_device_id", &FlowFunc::MetaParams::GetRunningDeviceId)
      .def("get_running_instance_id", &FlowFunc::MetaParams::GetRunningInstanceId)
      .def("get_running_instance_num", &FlowFunc::MetaParams::GetRunningInstanceNum)
      .def("__repr__", [](FlowFunc::MetaParams &self) {
        std::stringstream repr;
        repr << "MetaParams(name= " << self.GetName();
        repr << " , input_number=" << self.GetInputNum();
        repr << " , output_number=" << self.GetOutputNum();
        repr << ", working_path=" << self.GetWorkPath();
        repr << ", running_device_id=" << self.GetRunningDeviceId();
        repr << ", running_instance_id=" << self.GetRunningInstanceId();
        repr << ", running_instance_num=" << self.GetRunningInstanceNum() << ")";
        return repr.str();
      });

  py::enum_<FlowFunc::AffinityPolicy>(m, "AffinityPolicy")
      .value("NO_AFFINITY", FlowFunc::AffinityPolicy::NO_AFFINITY)
      .value("ROW_AFFINITY", FlowFunc::AffinityPolicy::ROW_AFFINITY)
      .value("COL_AFFINITY", FlowFunc::AffinityPolicy::COL_AFFINITY)
      .export_values();

  py::class_<PyBalanceConfig>(m, "BalanceConfig")
      .def(py::init<>())
      .def("set_data_pos", &PyBalanceConfig::SetDataPos)
      .def("set_balance_weight", &PyBalanceConfig::SetBalanceWeight)
      .def("set_affinity_policy", &PyBalanceConfig::SetAffinityPolicy);

  py::class_<FlowFunc::MetaRunContext, std::shared_ptr<FlowFunc::MetaRunContext>, PyMetaRunContext>(m, "MetaRunContext")
      .def(py::init<>())
      .def(
          "alloc_tensor_msg",
          [](FlowFunc::MetaRunContext &self, const std::vector<int64_t> &shapes, const ge::DataType &dtype,
             uint32_t align) {
            const auto func_dtype = TransGeDataTypeToFuncDataType(dtype);
            return self.AllocTensorMsgWithAlign(shapes, func_dtype, align);
          },
          py::return_value_policy::reference)
      .def(
          "alloc_raw_data_msg",
          [](FlowFunc::MetaRunContext &self, int64_t size, uint32_t align) {
            return self.AllocRawDataMsg(size, align);
          },
          py::return_value_policy::reference)
      .def(
          "to_flow_msg",
          [](FlowFunc::MetaRunContext &self, std::shared_ptr<FlowFunc::Tensor> tensor) {
            return self.ToFlowMsg(tensor);
          },
          py::return_value_policy::reference)
      .def("set_output",
           overload_cast_<uint32_t, std::shared_ptr<FlowFunc::FlowMsg>>()(&FlowFunc::MetaRunContext::SetOutput))
      .def("set_output",
           [](FlowFunc::MetaRunContext &self, uint32_t out_idx, std::shared_ptr<FlowFunc::FlowMsg> out_msg,
              const PyBalanceConfig &config) {
             FlowFunc::OutOptions options;
             auto *balan_config = options.MutableBalanceConfig();
             balan_config->SetAffinityPolicy(config.GetAffinityPolicy());
             balan_config->SetDataPos(config.GetDataPos());
             FlowFunc::BalanceWeight balance_weight;
             balance_weight.rowNum = config.GetRowNum();
             balance_weight.colNum = config.GetColNum();
             balan_config->SetBalanceWeight(balance_weight);
             return self.SetOutput(out_idx, out_msg, options);
           })
      .def("set_multi_outputs",
           [](FlowFunc::MetaRunContext &self, uint32_t out_idx,
              const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &out_msg, const PyBalanceConfig &config) {
             FlowFunc::OutOptions options;
             auto *balan_config = options.MutableBalanceConfig();
             balan_config->SetAffinityPolicy(config.GetAffinityPolicy());
             balan_config->SetDataPos(config.GetDataPos());
             FlowFunc::BalanceWeight balance_weight;
             balance_weight.rowNum = config.GetRowNum();
             balance_weight.colNum = config.GetColNum();
             balan_config->SetBalanceWeight(balance_weight);
             return self.SetMultiOutputs(out_idx, out_msg, options);
           })
      .def("alloc_empty_msg", &FlowFunc::MetaRunContext::AllocEmptyDataMsg, py::return_value_policy::reference)
      .def(
          "run_flow_model",
          [](FlowFunc::MetaRunContext &self, const char *model_key,
             std::vector<std::shared_ptr<FlowFunc::FlowMsg>> input_msgs, int32_t timeout) {
            std::vector<std::shared_ptr<FlowFunc::FlowMsg>> outputMsgs;
            if (self.RunFlowModel(model_key, input_msgs, outputMsgs, timeout) == FlowFunc::FLOW_FUNC_SUCCESS) {
              return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, outputMsgs);
            }
            return std::make_tuple(FlowFunc::FLOW_FUNC_FAILED, std::vector<std::shared_ptr<FlowFunc::FlowMsg>>());
          },
          py::return_value_policy::reference_internal)
      .def("get_user_data",
           [](FlowFunc::MetaRunContext &self, py::buffer user_data, size_t size, size_t offset) {
             void *data = reinterpret_cast<void *>(user_data.request().ptr);
             return self.GetUserData(data, size, offset);
           })
      .def("raise_exception", &PyMetaRunContext::RaiseException)
      .def("get_exception",
           [](FlowFunc::MetaRunContext &self) {
             int32_t exp_code = 0;
             uint64_t usr_context_id = 0;
             bool ret = self.GetException(exp_code, usr_context_id);
             return std::make_tuple(ret, exp_code, usr_context_id);
           })
      .def("__repr__", [](FlowFunc::MetaRunContext &self) {
        (void)self;
        std::stringstream repr;
        repr << "MetaRunContext()";
        return repr.str();
      });

  m.def("init_func_datatype_manager", [](const std::map<FlowFunc::TensorDataType, py::array> &type_map) {
    FuncDataTypeManager::GetInstance().Init(type_map);
  });

  py::class_<RuntimeTensorDesc>(m, "RuntimeTensorDesc")
      .def(py::init<>())
      .def_static("from_memory",
                  [](py::buffer &buf) {
                    py::buffer_info info = buf.request();
                    if (static_cast<size_t>(info.size) < sizeof(RuntimeTensorDesc)) {
                      throw std::runtime_error("Buffer size is less than sizeof(RuntimeTensorDesc)");
                    }
                    auto desc_view = static_cast<RuntimeTensorDesc *>(info.ptr);
                    return *desc_view;
                  })
      .def_readwrite("address", &RuntimeTensorDesc::data_addr)
      .def_readwrite("dtype", &RuntimeTensorDesc::dtype)
      .def_readwrite("size", &RuntimeTensorDesc::data_size)
      .def_property(
          "shape", [](RuntimeTensorDesc &s) { return std::vector<int64_t>(&s.shape[1], &s.shape[1 + s.shape[0]]); },
          [](RuntimeTensorDesc &s, const std::vector<int64_t> &v) {
            s.shape[0] = v.size() > kMaxDimSize ? kMaxDimSize : v.size();
            for (size_t i = 0; i < v.size(); ++i) {
              s.shape[i + 1] = v[i];
            }
          })
      .def("to_bytes", [](RuntimeTensorDesc &desc) {
        return py::bytes(reinterpret_cast<char *>(&desc), sizeof(RuntimeTensorDesc));
      });

  py::class_<RuntimeTensorDescMsgProcessor>(m, "RuntimeTensorDescMsgProcessor")
      .def_static("get_runtime_tensor_descs",
                  [](const std::shared_ptr<FlowFunc::FlowMsg> &input_flow_msg, int64_t input_num) {
                    std::vector<RuntimeTensorDesc> runtime_tensor_descs;
                    auto ret = RuntimeTensorDescMsgProcessor::GetRuntimeTensorDescs(input_flow_msg,
                                                                                    runtime_tensor_descs, input_num);
                    return std::make_tuple(ret, runtime_tensor_descs);
                  })
      .def_static("create_runtime_tensor_desc_msg", &RuntimeTensorDescMsgProcessor::CreateRuntimeTensorDescMsg);
}