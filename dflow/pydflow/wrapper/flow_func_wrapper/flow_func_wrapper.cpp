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
#include "flow_func/meta_run_context.h"
#include "flow_func/meta_params.h"
#include "flow_func/tensor_data_type.h"
#include "graph/tensor.h"
#include "dlog_pub.h"

namespace FlowFunc {
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

const std::map<TensorDataType, ge::DataType> flow_func_dtype_to_ge_dtype{
    {TensorDataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {TensorDataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {TensorDataType::DT_INT8, ge::DataType::DT_INT8},
    {TensorDataType::DT_INT16, ge::DataType::DT_INT16},
    {TensorDataType::DT_UINT16, ge::DataType::DT_UINT16},
    {TensorDataType::DT_UINT8, ge::DataType::DT_UINT8},
    {TensorDataType::DT_INT32, ge::DataType::DT_INT32},
    {TensorDataType::DT_INT64, ge::DataType::DT_INT64},
    {TensorDataType::DT_UINT32, ge::DataType::DT_UINT32},
    {TensorDataType::DT_UINT64, ge::DataType::DT_UINT64},
    {TensorDataType::DT_BOOL, ge::DataType::DT_BOOL},
    {TensorDataType::DT_DOUBLE, ge::DataType::DT_DOUBLE},
    {TensorDataType::DT_QINT8, ge::DataType::DT_QINT8},
    {TensorDataType::DT_QINT16, ge::DataType::DT_QINT16},
    {TensorDataType::DT_QINT32, ge::DataType::DT_QINT32},
    {TensorDataType::DT_QUINT8, ge::DataType::DT_QUINT8},
    {TensorDataType::DT_QUINT16, ge::DataType::DT_QUINT16},
    {TensorDataType::DT_DUAL, ge::DataType::DT_DUAL},
    {TensorDataType::DT_INT4, ge::DataType::DT_INT4},
    {TensorDataType::DT_UINT1, ge::DataType::DT_UINT1},
    {TensorDataType::DT_INT2, ge::DataType::DT_INT2},
    {TensorDataType::DT_UINT2, ge::DataType::DT_UINT2},
    {TensorDataType::DT_UNDEFINED, ge::DataType::DT_UNDEFINED}};

ge::DataType TransFuncDataTypeToGeDataType(const TensorDataType &data_type) {
  const auto iter = flow_func_dtype_to_ge_dtype.find(data_type);
  if (iter == flow_func_dtype_to_ge_dtype.cend()) {
    return ge::DataType::DT_UNDEFINED;
  }
  return iter->second;
}

const std::map<ge::DataType, TensorDataType> ge_dtype_to_flow_func_dtype{
    {ge::DataType::DT_FLOAT, TensorDataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, TensorDataType::DT_FLOAT16},
    {ge::DataType::DT_INT8, TensorDataType::DT_INT8},
    {ge::DataType::DT_INT16, TensorDataType::DT_INT16},
    {ge::DataType::DT_UINT16, TensorDataType::DT_UINT16},
    {ge::DataType::DT_UINT8, TensorDataType::DT_UINT8},
    {ge::DataType::DT_INT32, TensorDataType::DT_INT32},
    {ge::DataType::DT_INT64, TensorDataType::DT_INT64},
    {ge::DataType::DT_UINT32, TensorDataType::DT_UINT32},
    {ge::DataType::DT_UINT64, TensorDataType::DT_UINT64},
    {ge::DataType::DT_BOOL, TensorDataType::DT_BOOL},
    {ge::DataType::DT_DOUBLE, TensorDataType::DT_DOUBLE},
    {ge::DataType::DT_QINT8, TensorDataType::DT_QINT8},
    {ge::DataType::DT_QINT16, TensorDataType::DT_QINT16},
    {ge::DataType::DT_QINT32, TensorDataType::DT_QINT32},
    {ge::DataType::DT_QUINT8, TensorDataType::DT_QUINT8},
    {ge::DataType::DT_QUINT16, TensorDataType::DT_QUINT16},
    {ge::DataType::DT_DUAL, TensorDataType::DT_DUAL},
    {ge::DataType::DT_INT4, TensorDataType::DT_INT4},
    {ge::DataType::DT_UINT1, TensorDataType::DT_UINT1},
    {ge::DataType::DT_INT2, TensorDataType::DT_INT2},
    {ge::DataType::DT_UINT2, TensorDataType::DT_UINT2},
    {ge::DataType::DT_UNDEFINED, TensorDataType::DT_UNDEFINED}};

TensorDataType TransGeDataTypeToFuncDataType(const ge::DataType &data_type) {
  const auto iter = ge_dtype_to_flow_func_dtype.find(data_type);
  if (iter == ge_dtype_to_flow_func_dtype.cend()) {
    return TensorDataType::DT_UNDEFINED;
  }
  return iter->second;
}

class PyFlowFuncLogger : public FlowFuncLogger {
 public:
  bool IsLogEnable(FlowFuncLogLevel level) override {
    (void)level;
    PYBIND11_OVERRIDE_PURE(bool, FlowFuncLogger, IsLogEnable,);
  }

  void Error(const char *fmt, ...) override {
    (void)fmt;
    PYBIND11_OVERRIDE_PURE(void, FlowFuncLogger, Error,);
  }

  void Warn(const char *fmt, ...) override {
    (void)fmt;
    PYBIND11_OVERRIDE_PURE(void, FlowFuncLogger, Warn,);
  }

  void Info(const char *fmt, ...) override {
    (void)fmt;
    PYBIND11_OVERRIDE_PURE(void, FlowFuncLogger, Info,);
  }

  void Debug(const char *fmt, ...) override {
    (void)fmt;
    PYBIND11_OVERRIDE_PURE(void, FlowFuncLogger, Debug,);
  }

  static void ErrorLog(const char *location_message, const char *user_message) {
    auto &logger = GetLogger(FlowFuncLogType::DEBUG_LOG);
    logger.Error("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }

  static void WarnLog(const char *location_message, const char *user_message) {
    auto &logger = GetLogger(FlowFuncLogType::DEBUG_LOG);
    logger.Warn("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }

  static void InfoLog(const char *location_message, const char *user_message) {
    auto &logger = GetLogger(FlowFuncLogType::DEBUG_LOG);
    logger.Info("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }

  static void DebugLog(const char *location_message, const char *user_message) {
    auto &logger = GetLogger(FlowFuncLogType::DEBUG_LOG);
    logger.Debug("%s%s: %s", location_message, GetLogExtHeader(), user_message);
  }

  static void RunTypeError(const char *location_message, const char *user_message) {
    auto &logger = GetLogger(FlowFuncLogType::RUN_LOG);
    logger.Error("%s%s[RUN]: %s", location_message, GetLogExtHeader(), user_message);
  }

  static void RunTypeInfo(const char *location_message, const char *user_message) {
    auto &logger = GetLogger(FlowFuncLogType::RUN_LOG);
    logger.Info("%s%s[RUN]: %s", location_message, GetLogExtHeader(), user_message);
  }
};

class PyFlowMsg : public FlowMsg {
 public:
  MsgType GetMsgType() const override {
    PYBIND11_OVERRIDE_PURE(MsgType, FlowMsg, GetMsgType,);
  }

  Tensor *GetTensor() const override {
    PYBIND11_OVERRIDE_PURE(Tensor *, FlowMsg, GetTensor,);
  }

  int32_t GetRetCode() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowMsg, GetRetCode,);
  }

  void SetRetCode(int32_t ret_code) override {
    PYBIND11_OVERRIDE_PURE(void, FlowMsg, SetRetCode, ret_code);
  }

  void SetStartTime(uint64_t start_time) override {
    PYBIND11_OVERRIDE_PURE(void, FlowMsg, SetStartTime, start_time);
  }

  uint64_t GetStartTime() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowMsg, GetStartTime,);
  }

  void SetEndTime(uint64_t end_time) override {
    PYBIND11_OVERRIDE_PURE(void, FlowMsg, SetEndTime, end_time);
  }

  uint64_t GetEndTime() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowMsg, GetEndTime,);
  }

  void SetFlowFlags(uint32_t flags) override {
    PYBIND11_OVERRIDE_PURE(void, FlowMsg, SetFlowFlags, flags);
  }

  uint32_t GetFlowFlags() const override {
    PYBIND11_OVERRIDE_PURE(uint32_t, FlowMsg, GetFlowFlags,);
  }

  void SetRouteLabel(uint32_t route_label) override {
    (void)route_label;
    PYBIND11_OVERRIDE_PURE(void, FlowMsg, SetRouteLabel,);
  }

  uint64_t GetTransactionId() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, FlowMsg, GetTransactionId,);
  }
};

class PyFlowMsgQueue : public FlowMsgQueue {
 public:
  int32_t Dequeue(std::shared_ptr<FlowMsg> &flow_msg, int32_t timeout) override {
    (void)flow_msg;
    (void)timeout;
    PYBIND11_OVERRIDE_PURE(int32_t, FlowMsgQueue, Dequeue,);
  }

  int32_t Depth() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowMsgQueue, Depth,);
  }

  int32_t Size() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, FlowMsgQueue, Size,);
  }
};

class PyTensor : public Tensor {
 public:
  const std::vector<int64_t> &GetShape() const override {
    PYBIND11_OVERRIDE_PURE(const std::vector<int64_t> &, Tensor, GetShape,);
  }

  TensorDataType GetDataType() const override {
    PYBIND11_OVERRIDE_PURE(TensorDataType, Tensor, GetDataType,);
  }

  void *GetData() const override {
    PYBIND11_OVERRIDE_PURE(void *, Tensor, GetData,);
  }

  uint64_t GetDataSize() const override {
    PYBIND11_OVERRIDE_PURE(uint64_t, Tensor, GetDataSize,);
  }

  int64_t GetElementCnt() const override {
    PYBIND11_OVERRIDE_PURE(int64_t, Tensor, GetElementCnt,);
  }

  int32_t Reshape(const std::vector<int64_t> &shape) override {
    (void)shape;
    PYBIND11_OVERRIDE_PURE(int32_t, Tensor, Reshape,);
  }
};

class PyMetaParams : public MetaParams {
 public:
  const char *GetName() const override {
    PYBIND11_OVERRIDE_PURE(char *, MetaParams, GetName,);
  }

  std::shared_ptr<const AttrValue> GetAttr(const char *attr_name) const override {
    (void)attr_name;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<const AttrValue>, MetaParams, GetAttr,);
  }

  size_t GetInputNum() const override {
    PYBIND11_OVERRIDE_PURE(size_t, MetaParams, GetInputNum,);
  }

  size_t GetOutputNum() const override {
    PYBIND11_OVERRIDE_PURE(size_t, MetaParams, GetOutputNum,);
  }

  const char *GetWorkPath() const override {
    PYBIND11_OVERRIDE_PURE(char *, MetaParams, GetWorkPath,);
  }

  int32_t GetRunningDeviceId() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, MetaParams, GetRunningDeviceId,);
  }

  int32_t GetRunningInstanceId() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, MetaParams, GetRunningInstanceId,);
  }

  int32_t GetRunningInstanceNum() const override {
    PYBIND11_OVERRIDE_PURE(int32_t, MetaParams, GetRunningInstanceNum,);
  }
};

class PyMetaRunContext : public MetaRunContext {
 public:
  std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape,
                                          TensorDataType data_type) override {
    (void)shape;
    (void)data_type;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowMsg>, MetaRunContext, AllocTensorMsg,);
  }

  std::shared_ptr<FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
                                                   TensorDataType data_type,
                                                   uint32_t align) override {
    (void)shape;
    (void)data_type;
    (void)align;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowMsg>, MetaRunContext, AllocTensorMsg,);
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) override {
    (void)out_idx;
    (void)out_msg;
    PYBIND11_OVERRIDE_PURE(int32_t, MetaRunContext, SetOutput,);
  }

  int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg,
                    const OutOptions &options) override {
    (void)out_idx;
    (void)out_msg;
    (void)options;
    PYBIND11_OVERRIDE_PURE(int32_t, MetaRunContext, SetOutput,);
  }

  int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
                          const OutOptions &options) override {
    (void)out_msgs;
    (void)out_idx;
    (void)options;
    PYBIND11_OVERRIDE_PURE(int32_t, MetaRunContext, SetMultiOutputs,);
  }

  std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msg_type) override {
    (void)msg_type;
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowMsg>, MetaRunContext, AllocEmptyDataMsg,);
  }

  int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
                       std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) override {
    (void)model_key;
    (void)input_msgs;
    (void)output_msgs;
    (void)timeout;
    PYBIND11_OVERRIDE_PURE(int32_t, MetaRunContext, RunFlowModel,);
  }

  int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const override {
    (void)data;
    (void)size;
    (void)offset;
    PYBIND11_OVERRIDE_PURE(int32_t, MetaRunContext, GetUserData,);
  }
};

class FuncDataTypeManager {
 public:
  static FuncDataTypeManager &GetInstance() {
    static FuncDataTypeManager data_type_manager;
    return data_type_manager;
  }

  void Init(const std::map<TensorDataType, py::array> &type_map) {
    for (const auto &item : type_map) {
      auto const dtype = item.first;
      auto const array = item.second;
      const auto buff = array.request();
      func_dtype_to_format_desc_[dtype] = buff.format;
    }
  }

  const std::map<TensorDataType, std::string> &GetFlowFuncDtypeToFormatDesc() const {
    return func_dtype_to_format_desc_;
  }

 private:
  FuncDataTypeManager() = default;
  std::map<TensorDataType, std::string> func_dtype_to_format_desc_;
};

py::memoryview ToReadonlyMemoryView(const Tensor &tensor) {
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
  auto const &format_descs = FuncDataTypeManager::GetInstance().GetFlowFuncDtypeToFormatDesc();
  const auto it = format_descs.find(tensor.GetDataType());
  if (it != format_descs.cend()) {
    return py::memoryview::from_buffer(tensor.GetData(), static_cast<ssize_t>(element_size), it->second.c_str(),
                                       tensor.GetShape(), strides, true);
  }
  return py::memoryview::from_memory(tensor.GetData(), static_cast<ssize_t>(tensor.GetDataSize()), true);
}

class PyBalanceConfig {
 public:
  void SetAffinityPolicy(AffinityPolicy policy) {
    policy_ = policy;
  }

  void SetBalanceWeight(int32_t row_num, int32_t col_num) {
    row_num_ = row_num;
    col_num_ = col_num;
  }

  void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &data_pos) {
    data_pos_ = data_pos;
  }

  AffinityPolicy GetAffinityPolicy() const {
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
  AffinityPolicy policy_ = AffinityPolicy::NO_AFFINITY;
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
  uint8_t reserved[448]; // padding to 1024 bytes.
};

class RuntimeTensorDescMsgProcessor {
 public:
  static int32_t GetRuntimeTensorDescs(const std::shared_ptr<FlowMsg> &input_flow_msg,
                                       std::vector<RuntimeTensorDesc> &runtime_tensor_descs, int64_t input_num) {
    void *data_ptr = nullptr;
    uint64_t data_size = 0UL;
    auto ret = input_flow_msg->GetRawData(data_ptr, data_size);
    if (ret != FLOW_FUNC_SUCCESS) {
      UDF_LOG_ERROR("Failed to get raw data, ret = %d.", ret);
      return ret;
    }
    uint64_t offset = 0UL;
    for (int64_t i = 0; i < input_num; ++i) {
      if (data_size - offset < sizeof(RuntimeTensorDesc)) {
        UDF_LOG_ERROR("Failed to check flow msg size, data size = %lu, input num = %ld.", data_size, input_num);
        return FLOW_FUNC_ERR_PARAM_INVALID;
      }
      auto desc = reinterpret_cast<RuntimeTensorDesc *>(static_cast<uint8_t *>(data_ptr) + offset);
      runtime_tensor_descs.emplace_back(*desc);
      offset += sizeof(RuntimeTensorDesc);
    }
    return FLOW_FUNC_SUCCESS;
  }

  static std::shared_ptr<FlowMsg> CreateRuntimeTensorDescMsg(
      const std::shared_ptr<MetaRunContext> &run_context,
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
    if (ret != FLOW_FUNC_SUCCESS) {
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

void BindFlowFuncAttr(py::module &m) {
  m.attr("FLOW_FUNC_SUCCESS") = FLOW_FUNC_SUCCESS;
  m.attr("FLOW_FUNC_FAILED") = FLOW_FUNC_FAILED;
  m.attr("FLOW_FUNC_ERR_PARAM_INVALID") = FLOW_FUNC_ERR_PARAM_INVALID;
  m.attr("FLOW_FUNC_ERR_ATTR_NOT_EXITS") = FLOW_FUNC_ERR_ATTR_NOT_EXITS;
  m.attr("FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH") = FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
  m.attr("FLOW_FUNC_ERR_TIME_OUT_ERROR") = FLOW_FUNC_ERR_TIME_OUT_ERROR;
  m.attr("FLOW_FUNC_STATUS_REDEPLOYING") = FLOW_FUNC_STATUS_REDEPLOYING;
  m.attr("FLOW_FUNC_STATUS_EXIT") = FLOW_FUNC_STATUS_EXIT;
  m.attr("FLOW_FUNC_ERR_DRV_ERROR") = FLOW_FUNC_ERR_DRV_ERROR;
  m.attr("FLOW_FUNC_ERR_QUEUE_ERROR") = FLOW_FUNC_ERR_QUEUE_ERROR;
  m.attr("FLOW_FUNC_ERR_MEM_BUF_ERROR") = FLOW_FUNC_ERR_MEM_BUF_ERROR;
  m.attr("FLOW_FUNC_ERR_EVENT_ERROR") = FLOW_FUNC_ERR_EVENT_ERROR;
  m.attr("FLOW_FUNC_ERR_USER_DEFINE_START") = FLOW_FUNC_ERR_USER_DEFINE_START;
  m.attr("FLOW_FUNC_ERR_USER_DEFINE_END") = FLOW_FUNC_ERR_USER_DEFINE_END;
}

void BindFlowFuncEnum(py::module &m) {
  py::enum_<FlowFuncLogType>(m, "FlowFuncLogType", py::arithmetic())
      .value("DEBUG_LOG", FlowFuncLogType::DEBUG_LOG)
      .value("RUN_LOG", FlowFuncLogType::RUN_LOG)
      .export_values();
  py::enum_<FlowFuncLogLevel>(m, "FlowFuncLogLevel", py::arithmetic())
      .value("DEBUG", FlowFuncLogLevel::DEBUG)
      .value("INFO", FlowFuncLogLevel::INFO)
      .value("WARN", FlowFuncLogLevel::WARN)
      .value("ERROR", FlowFuncLogLevel::ERROR)
      .export_values();
  py::enum_<MsgType>(m, "MsgType", py::arithmetic())
      .value("MSG_TYPE_TENSOR_DATA", MsgType::MSG_TYPE_TENSOR_DATA)
      .value("MSG_TYPE_RAW_MSG", MsgType::MSG_TYPE_RAW_MSG)
      .value("MSG_TYPE_TORCH_TENSOR_MSG", static_cast<MsgType>(1023))
      .value("MSG_TYPE_USER_DEFINE_START", static_cast<MsgType>(1024))
      .value("MSG_TYPE_PICKLED_MSG", static_cast<MsgType>(65535))
      .export_values();
  py::enum_<FlowFlag>(m, "FlowFlag", py::arithmetic())
      .value("FLOW_FLAG_EOS", FlowFlag::FLOW_FLAG_EOS)
      .value("FLOW_FLAG_SEG", FlowFlag::FLOW_FLAG_SEG)
      .export_values();
  py::enum_<AffinityPolicy>(m, "AffinityPolicy")
      .value("NO_AFFINITY", AffinityPolicy::NO_AFFINITY)
      .value("ROW_AFFINITY", AffinityPolicy::ROW_AFFINITY)
      .value("COL_AFFINITY", AffinityPolicy::COL_AFFINITY)
      .export_values();
}

void BindFlowFuncLogger(py::module &m) {
  py::class_<FlowFuncLogger, std::shared_ptr<FlowFuncLogger>, PyFlowFuncLogger>(m, "FlowFuncLogger")
      .def(py::init())
      .def("get_log_header", [](FlowFuncLogger &self) {
        const std::string log_header(self.GetLogExtHeader());
        return log_header;
      })
      .def("is_log_enable",
           [](FlowFuncLogger &self, const FlowFuncLogType &log_type, const FlowFuncLogLevel &log_level) {
             FlowFuncLogger &logger = self.GetLogger(log_type);
             return logger.IsLogEnable(log_level);
           })
      .def("debug_log_error", [](PyFlowFuncLogger &self, const char *location_message, const char *user_message) {
        (void)self;
        PyFlowFuncLogger::ErrorLog(location_message, user_message);
      })
      .def("debug_log_info", [](PyFlowFuncLogger &self, const char *location_message, const char *user_message) {
        (void)self;
        PyFlowFuncLogger::InfoLog(location_message, user_message);
      })
      .def("debug_log_warn", [](PyFlowFuncLogger &self, const char *location_message, const char *user_message) {
        (void)self;
        PyFlowFuncLogger::WarnLog(location_message, user_message);
      })
      .def("debug_log_debug", [](PyFlowFuncLogger &self, const char *location_message, const char *user_message) {
        (void)self;
        PyFlowFuncLogger::DebugLog(location_message, user_message);
      })
      .def("run_log_error", [](PyFlowFuncLogger &self, const char *location_message, const char *user_message) {
        (void)self;
        PyFlowFuncLogger::RunTypeError(location_message, user_message);
      })
      .def("run_log_info", [](PyFlowFuncLogger &self, const char *location_message, const char *user_message) {
        (void)self;
        PyFlowFuncLogger::RunTypeInfo(location_message, user_message);
      })
      .def("__repr__", [](FlowFuncLogger &self) {
        std::stringstream repr;
        repr << "FlowFuncLogger(LogHeader=" << self.GetLogExtHeader() << ")";
        return repr.str();
      });
}

void BindFlowMsg(py::module &m) {
  py::class_<FlowMsg, std::shared_ptr<FlowMsg>, PyFlowMsg>(m, "FlowMsg")
      .def(py::init<>())
      .def("get_msg_type", &FlowMsg::GetMsgType)
      .def("set_msg_type", [](FlowMsg &self, uint16_t msg_type) {
        return self.SetMsgType(static_cast<MsgType>(msg_type));
      })
      .def("get_tensor", &FlowMsg::GetTensor, py::return_value_policy::reference)
      .def("get_raw_data", [](FlowMsg &self) {
        void *data = nullptr;
        uint64_t data_size = 0U;
        (void)self.GetRawData(data, data_size);
        return py::memoryview::from_memory(data, data_size, false);
      })
      .def("get_ret_code", &FlowMsg::GetRetCode)
      .def("set_ret_code", &FlowMsg::SetRetCode)
      .def("get_start_time", &FlowMsg::GetStartTime)
      .def("set_start_time", &FlowMsg::SetStartTime)
      .def("get_end_time", &FlowMsg::GetEndTime)
      .def("set_end_time", &FlowMsg::SetEndTime)
      .def("get_flow_flags", &FlowMsg::GetFlowFlags)
      .def("set_flow_flags", &FlowMsg::SetFlowFlags)
      .def("set_route_label", &FlowMsg::SetRouteLabel)
      .def("get_transaction_id", &FlowMsg::GetTransactionId)
      .def("set_transaction_id", &FlowMsg::SetTransactionId)
      .def("__repr__", [](FlowMsg &self) {
        std::stringstream repr;
        repr << "FlowMsg(msg_type=" << static_cast<int32_t>(self.GetMsgType());
        repr << ", tensor=...";
        repr << ", ret_code=" << self.GetRetCode();
        repr << ", start_time=" << self.GetStartTime();
        repr << ", end_time=" << self.GetEndTime();
        repr << ", flow_flags=" << self.GetFlowFlags() << ")";
        return repr.str();
      });
}

void BindFlowMsgQueue(py::module &m) {
  py::class_<FlowMsgQueue, std::shared_ptr<FlowMsgQueue>, PyFlowMsgQueue>(m, "FlowMsgQueue")
      .def(py::init<>())
      .def("dequeue", [](FlowMsgQueue &self, int32_t timeout) {
             std::shared_ptr<FlowMsg> flow_msg;
             const auto ret = self.Dequeue(flow_msg, timeout);
             return std::make_tuple(ret, flow_msg);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("depth", &FlowMsgQueue::Depth)
      .def("size", &FlowMsgQueue::Size);
}

void BindFlowBufferFactory(py::module &m) {
  py::class_<FlowBufferFactory>(m, "FlowBufferFactory")
      .def_static(
          "alloc_tensor",
          [](const std::vector<int64_t> &shapes, const ge::DataType &dtype, uint32_t align) {
            const auto func_dtype = TransGeDataTypeToFuncDataType(dtype);
            return FlowBufferFactory::AllocTensor(shapes, func_dtype, align);
          },
          py::return_value_policy::reference);
}

void BindTensor(py::module &m) {
  py::class_<Tensor, std::shared_ptr<Tensor>, PyTensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<>())
      .def("get_shape", &Tensor::GetShape)
      .def("get_dtype", [](Tensor &self) {
        const auto f_dtype = self.GetDataType();
        const auto ge_dtype = TransFuncDataTypeToGeDataType(f_dtype);
        return ge_dtype;
      })
      .def("get_data", &ToReadonlyMemoryView)
      .def("get_writable_data", [](Tensor &self) {
        return py::memoryview::from_memory(self.GetData(), self.GetDataSize(), false);
      })
      .def("get_data_size", &Tensor::GetDataSize)
      .def("get_element_cnt", &Tensor::GetElementCnt)
      .def("reshape", &Tensor::Reshape)
      .def("__repr__", [](Tensor &self) {
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
      .def_buffer([](const Tensor &tensor) -> py::buffer_info {
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
}

template <typename T, typename Default = T>
auto GetAttrWrapper(MetaParams &self, const char *name, Default default_value = Default{}) {
  T value = default_value;
  const auto ret = self.GetAttr<T>(name, value);
  return std::make_tuple(ret, value);
}

auto GetTensorDtypeListWrapper(MetaParams &self, const char *name) {
  std::vector<TensorDataType> value;
  std::vector<ge::DataType> ge_dtype;
  const auto ret = self.GetAttr<std::vector<TensorDataType>>(name, value);
  if (ret == FLOW_FUNC_SUCCESS) {
    ge_dtype.reserve(value.size());
    std::transform(value.begin(), value.end(), std::back_inserter(ge_dtype),
                   TransFuncDataTypeToGeDataType);
  }
  return std::make_tuple(ret, ge_dtype);
}

auto GetStringListWrapper(MetaParams &self, const char *name) {
  std::vector<AscendString> value;
  std::vector<std::string> str_list;
  const auto ret = self.GetAttr<std::vector<AscendString>>(name, value);
  if (ret == FLOW_FUNC_SUCCESS) {
    str_list.reserve(value.size());
    std::transform(value.begin(), value.end(), std::back_inserter(str_list),
                   [](const AscendString &s) {
                     return s.GetString();
                   });
  }
  return std::make_tuple(ret, str_list);
}

std::string MetaParamsRepr(const MetaParams &self) {
  std::stringstream repr;
  repr << "MetaParams(name= " << self.GetName() << " , input_number=" << self.GetInputNum()
      << " , output_number=" << self.GetOutputNum() << ", working_path=" << self.GetWorkPath()
      << ", running_device_id=" << self.GetRunningDeviceId()
      << ", running_instance_id=" << self.GetRunningInstanceId()
      << ", running_instance_num=" << self.GetRunningInstanceNum() << ")";
  return repr.str();
}

void BindMetaParams(py::module &m) {
  py::class_<MetaParams, std::shared_ptr<MetaParams>, PyMetaParams>(m, "MetaParams")
      .def(py::init<>())
      .def("get_name", &MetaParams::GetName)
      .def("get_int64", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<int64_t>(self, name, -1L);
      })
      .def("get_int64_vector", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<std::vector<int64_t>>(self, name);
      })
      .def("get_int64_vector_vector", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<std::vector<std::vector<int64_t>>>(self, name);
      })
      .def("get_bool", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<bool>(self, name, false);
      })
      .def("get_bool_list", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<std::vector<bool>>(self, name);
      })
      .def("get_float", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<float>(self, name, 0.0f);
      })
      .def("get_float_list", [](MetaParams &self, const char *name) {
        return GetAttrWrapper<std::vector<float>>(self, name);
      })
      .def("get_tensor_dtype", [](MetaParams &self, const char *name) {
        TensorDataType value = TensorDataType::DT_UNDEFINED;
        const auto ret = self.GetAttr<TensorDataType>(name, value);
        return std::make_tuple(ret, TransFuncDataTypeToGeDataType(value));
      })
      .def("get_tensor_dtype_list", &GetTensorDtypeListWrapper)
      .def("get_string", [](MetaParams &self, const char *name) {
        AscendString value;
        const auto ret = self.GetAttr<AscendString>(name, value);
        return std::make_tuple(ret, value.GetString());
      })
      .def("get_string_list", &GetStringListWrapper)
      .def("get_input_number", &MetaParams::GetInputNum)
      .def("get_output_number", &MetaParams::GetOutputNum)
      .def("get_work_path", &MetaParams::GetWorkPath)
      .def("get_running_device_id", &MetaParams::GetRunningDeviceId)
      .def("get_running_instance_id", &MetaParams::GetRunningInstanceId)
      .def("get_running_instance_num", &MetaParams::GetRunningInstanceNum)
      .def("__repr__", &MetaParamsRepr);
}

void BindBalanceConfig(py::module &m) {
  py::class_<PyBalanceConfig>(m, "BalanceConfig")
      .def(py::init<>())
      .def("set_data_pos", &PyBalanceConfig::SetDataPos)
      .def("set_balance_weight", &PyBalanceConfig::SetBalanceWeight)
      .def("set_affinity_policy", &PyBalanceConfig::SetAffinityPolicy);
}

OutOptions CreateOutOptions(const PyBalanceConfig &config) {
  OutOptions options;
  auto *balance_config = options.MutableBalanceConfig();
  balance_config->SetAffinityPolicy(config.GetAffinityPolicy());
  balance_config->SetDataPos(config.GetDataPos());

  BalanceWeight balance_weight;
  balance_weight.rowNum = config.GetRowNum();
  balance_weight.colNum = config.GetColNum();
  balance_config->SetBalanceWeight(balance_weight);
  return options;
}

void BindMetaRunContext(py::module &m) {
  py::class_<MetaRunContext, std::shared_ptr<MetaRunContext>, PyMetaRunContext>(m, "MetaRunContext")
      .def(py::init<>())
      .def("alloc_tensor_msg",
           [](MetaRunContext &self, const std::vector<int64_t> &shapes, const ge::DataType &dtype, uint32_t align) {
             const auto func_dtype = TransGeDataTypeToFuncDataType(dtype);
             return self.AllocTensorMsgWithAlign(shapes, func_dtype, align);
           }, py::return_value_policy::reference)
      .def("alloc_raw_data_msg", &MetaRunContext::AllocRawDataMsg, py::return_value_policy::reference)
      .def("to_flow_msg", &MetaRunContext::ToFlowMsg, py::return_value_policy::reference)
      .def("set_output", overload_cast_<uint32_t, std::shared_ptr<FlowMsg>>()(&MetaRunContext::SetOutput))
      .def("set_output", [](MetaRunContext &self, uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg,
                            const PyBalanceConfig &config) {
        return self.SetOutput(out_idx, out_msg, CreateOutOptions(config));
      })
      .def("set_multi_outputs", [](MetaRunContext &self, uint32_t out_idx,
                                   const std::vector<std::shared_ptr<FlowMsg>> &out_msg,
                                   const PyBalanceConfig &config) {
        return self.SetMultiOutputs(out_idx, out_msg, CreateOutOptions(config));
      })
      .def("alloc_empty_msg", &MetaRunContext::AllocEmptyDataMsg, py::return_value_policy::reference)
      .def("run_flow_model", [](MetaRunContext &self, const char *model_key,
                                std::vector<std::shared_ptr<FlowMsg>> input_msgs, int32_t timeout) {
        std::vector<std::shared_ptr<FlowMsg>> outputMsgs;
        if (self.RunFlowModel(model_key, input_msgs, outputMsgs, timeout) == FLOW_FUNC_SUCCESS) {
          return std::make_tuple(FLOW_FUNC_SUCCESS, outputMsgs);
        }
        return std::make_tuple(FLOW_FUNC_FAILED, std::vector<std::shared_ptr<FlowMsg>>());
      }, py::return_value_policy::reference_internal)
      .def("get_user_data", [](MetaRunContext &self, py::buffer user_data, size_t size, size_t offset) {
        void *data = reinterpret_cast<void *>(user_data.request().ptr);
        return self.GetUserData(data, size, offset);
      })
      .def("raise_exception", &PyMetaRunContext::RaiseException)
      .def("get_exception", [](MetaRunContext &self) {
        int32_t exp_code = 0;
        uint64_t usr_context_id = 0;
        bool ret = self.GetException(exp_code, usr_context_id);
        return std::make_tuple(ret, exp_code, usr_context_id);
      })
      .def("__repr__", [](MetaRunContext &self) {
        (void)self;
        return std::string("MetaRunContext()");
      });
}

void BindRuntimeTensorDesc(py::module &m) {
  py::class_<RuntimeTensorDesc>(m, "RuntimeTensorDesc")
      .def(py::init<>())
      .def_static("from_memory", [](py::buffer &buf) {
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
      .def_property("shape", [](RuntimeTensorDesc &s) {
                      return std::vector<int64_t>(&s.shape[1], &s.shape[1 + s.shape[0]]);
                    },
                    [](RuntimeTensorDesc &s, const std::vector<int64_t> &v) {
                      s.shape[0] = v.size() > kMaxDimSize ? kMaxDimSize : v.size();
                      for (size_t i = 0; i < v.size(); ++i) {
                        s.shape[i + 1] = v[i];
                      }
                    })
      .def("to_bytes", [](RuntimeTensorDesc &desc) {
        return py::bytes(reinterpret_cast<char *>(&desc), sizeof(RuntimeTensorDesc));
      });
}

void BindRuntimeTensorDescMsgProcessor(py::module &m) {
  py::class_<RuntimeTensorDescMsgProcessor>(m, "RuntimeTensorDescMsgProcessor")
      .def_static("get_runtime_tensor_descs",
                  [](const std::shared_ptr<FlowMsg> &input_flow_msg, int64_t input_num) {
                    std::vector<RuntimeTensorDesc> runtime_tensor_descs;
                    auto ret = RuntimeTensorDescMsgProcessor::GetRuntimeTensorDescs(input_flow_msg,
                      runtime_tensor_descs, input_num);
                    return std::make_tuple(ret, runtime_tensor_descs);
                  })
      .def_static("create_runtime_tensor_desc_msg", &RuntimeTensorDescMsgProcessor::CreateRuntimeTensorDescMsg);
}
} // namespace

PYBIND11_MODULE(flowfunc_wrapper, m) {
  m.doc() = "pybind11 flowfunc_wrapper plugin"; // optional module docstring
  m.def("init_func_datatype_manager", [](const std::map<TensorDataType, py::array> &type_map) {
    FuncDataTypeManager::GetInstance().Init(type_map);
  });
  BindFlowFuncAttr(m);
  BindFlowFuncEnum(m);
  BindFlowFuncLogger(m);
  BindFlowMsg(m);
  BindFlowMsgQueue(m);
  BindFlowBufferFactory(m);
  BindTensor(m);
  BindMetaParams(m);
  BindBalanceConfig(m);
  BindMetaRunContext(m);
  BindRuntimeTensorDesc(m);
  BindRuntimeTensorDescMsgProcessor(m);
}
}
