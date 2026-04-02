/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "framework/common/ge_types.h"
#include "graph/ge_context.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/node_executor/node_executor.h"
#include "graph/manager/graph_manager_utils.h"
#include "hybrid/hybrid_davinci_model.h"

namespace ge {
namespace hybrid {
class HybridDavinciModel::Impl {
 public:
  explicit Impl(GeRootModelPtr ge_model) : model_(std::move(ge_model)), executor_(&model_), load_stream_(nullptr) {}

  ~Impl() {
    NodeExecutorManager::GetInstance().FinalizeExecutors();
  }

  Status ResolveStreamPolicy() {
    constexpr const char_t *kParallelModeMultiStreams = "0";
    constexpr const char_t *kParallelModeSerial = "1";
    constexpr const char_t *kParallelModeSingleStream = "2";
    const std::set<std::string>
        kValidValues = {"", kParallelModeMultiStreams, kParallelModeSerial, kParallelModeSingleStream};
    std::string parallel_mode;
    (void) GetContext().GetOption(OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE, parallel_mode);
    GE_CHK_BOOL_RET_STATUS(kValidValues.count(parallel_mode) > 0, PARAM_INVALID,
                           "Option %s is invalid, value = [%s]",
                           OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE, parallel_mode.c_str());
    use_default_stream_ = (parallel_mode == kParallelModeSingleStream);
    GELOGI("Option %s = [%s]", OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE, parallel_mode.c_str());
    return ge::SUCCESS;
  }

  Status Init() {
    GE_CHK_STATUS_RET(ResolveStreamPolicy(), "[Initialize][ResolveStreamPolicy] failed");
    GE_CHK_STATUS_RET(NodeExecutorManager::GetInstance().EnsureInitialized(),
                      "[Initialize][NodeExecutorManager] failed");
    GE_CHK_STATUS_RET(model_.Init(), "[Init][HybridModel] failed.");
    GE_CHK_STATUS_RET(executor_.Init(load_stream_), "[Init][HybridModelAsyncExecutor] failed.");
    return SUCCESS;
  }

  Status Execute(const std::vector<DataBuffer> &inputs,
                 const std::vector<GeTensorDesc> &input_desc,
                 std::vector<DataBuffer> &outputs,
                 std::vector<GeTensorDesc> &output_desc,
                 const aclrtStream stream) {
    const auto main_stream = use_default_stream_ ? nullptr : stream;
    return executor_.Execute(inputs, input_desc, outputs, output_desc, main_stream);
  }

  Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
    return executor_.Execute(inputs, outputs);
  }

  Status ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                               std::vector<gert::Tensor> &outputs,
                                               const aclrtStream stream) {
    return executor_.ExecuteWithStreamAsync(inputs, outputs, stream);
  }

  Status ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                                const aclrtStream stream) {
    return executor_.ExecuteWithStreamAsync(inputs, outputs, stream);
  }

  Status ModelRunStart() {
    return executor_.Start(listener_);
  }

  Status ModelRunStop() {
    return executor_.Stop();
  }

  Status EnqueueData(const std::shared_ptr<RunArgs> &args) {
    return executor_.EnqueueData(args);
  }

  void SetListener(const shared_ptr<ModelListener> &listener) {
    listener_ = listener;
  }

  void SetModelId(const uint32_t model_id) {
    executor_.SetModelId(model_id);
    model_.SetModelId(model_id);
  }

  void SetDeviceId(const uint32_t device_id) {
    model_.SetDeviceId(device_id);
    executor_.SetDeviceId(device_id);
  }

  void SetOmName(const std::string &model_name) {
    model_.SetOmName(model_name);
  }

  void SetLoadStream(const aclrtStream stream) {
    load_stream_ = stream;
  }

  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
    model_.SetFileConstantWeightDir(file_constant_weight_dir);
  }

  uint32_t GetDeviceId() const {
    return model_.GetDeviceId();
  }

  uint64_t GetGlobalStepAddr() const {
    return PtrToValue(model_.GetGlobalStep());
  }

  const GraphExecutionContext *GeContext() const { return executor_.GeContext(); }

  GraphExecutionContext *GeContext() { return executor_.GeContext(); }

  uint64_t GetSessionId() const {
    return model_.GetSessionId();
  }

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const {
    return model_.GetDynamicBatchInfo(batch_info, dynamic_type);
  }

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const {
    model_.GetUserDesignateShapeOrder(user_input_shape_order);
  }

  void GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) const {
    model_.GetModelAttr(dynamic_output_shape_info);
  }

  Status GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &output_formats) {
    return model_.GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats);
  }

  void SetModelDescVersion(const bool is_new_model_desc) {
    model_.SetModelDescVersion(is_new_model_desc);
  }

  uint32_t GetDataInputerSize() const { return executor_.GetDataInputerSize(); }

  bool GetRunningFlag() const { return executor_.GetRunningFlag(); }

  Status SetRunAsyncListenerCallback(const RunAsyncCallbackV2 &callback) const {
    const auto listener = dynamic_cast<RunAsyncListener *>(listener_.get());
    GE_CHECK_NOTNULL(listener);
    listener->SetCallback(callback);
    return SUCCESS;
  }

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_value) const {
    return model_.GetOpAttr(op_name, attr_name, attr_value);
  }

  Status GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const {
    return model_.GetAippInfo(index, aipp_info);
  }

  Status GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_index) const {
    return model_.GetAippType(index, aipp_type, aipp_index);
  }

  Status ReportProfilingData() const {
    return model_.ReportProfilingData();
  }

 private:
  std::shared_ptr<ModelListener> listener_;
  HybridModel model_;
  HybridModelAsyncExecutor executor_;
  aclrtStream load_stream_ = nullptr;
  bool use_default_stream_ = false;
};

HybridDavinciModel::~HybridDavinciModel() {
  delete impl_;
}

std::unique_ptr<HybridDavinciModel> HybridDavinciModel::Create(const GeRootModelPtr &ge_root_model) {
  auto instance = std::unique_ptr<HybridDavinciModel>(new (std::nothrow)HybridDavinciModel());
  if (instance != nullptr) {
    instance->impl_ = new (std::nothrow) HybridDavinciModel::Impl(ge_root_model);
    if (instance->impl_ != nullptr) {
      return instance;
    }
  }

  return nullptr;
}

Status HybridDavinciModel::Init() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Init();
}

Status HybridDavinciModel::Execute(const std::vector<DataBuffer> &inputs,
                                   const std::vector<GeTensorDesc> &input_desc,
                                   std::vector<DataBuffer> &outputs,
                                   std::vector<GeTensorDesc> &output_desc,
                                   const aclrtStream stream) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Execute(inputs, input_desc, outputs, output_desc, stream);
}

Status HybridDavinciModel::Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Execute(inputs, outputs);
}

Status HybridDavinciModel::ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                                                  const aclrtStream stream) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ExecuteWithStreamAsync(inputs, outputs, stream);
}

Status HybridDavinciModel::ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                                  std::vector<gert::Tensor> &outputs,
                                                  const aclrtStream stream) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ExecuteWithStreamAsync(inputs, outputs, stream);
}

Status HybridDavinciModel::ModelRunStart() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ModelRunStart();
}

Status HybridDavinciModel::ModelRunStop() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ModelRunStop();
}

Status HybridDavinciModel::EnqueueData(const shared_ptr<RunArgs> &args) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->EnqueueData(args);
}

void HybridDavinciModel::SetListener(const shared_ptr<ModelListener> &listener) {
  if (impl_ != nullptr) {
    impl_->SetListener(listener);
  }
}

void HybridDavinciModel::SetModelId(const uint32_t model_id) {
  if (impl_ != nullptr) {
    impl_->SetModelId(model_id);
  }
}

void HybridDavinciModel::SetDeviceId(const uint32_t device_id) {
  if (impl_ != nullptr) {
    impl_->SetDeviceId(device_id);
  }
}

void HybridDavinciModel::SetOmName(const std::string &om_name) {
  if (impl_ != nullptr) {
    impl_->SetOmName(om_name);
  }
}

void HybridDavinciModel::SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
  if (impl_ != nullptr) {
    impl_->SetFileConstantWeightDir(file_constant_weight_dir);
  }
}

void HybridDavinciModel::SetLoadStream(const aclrtStream stream) {
  if (impl_ != nullptr) {
    impl_->SetLoadStream(stream);
  }
}

uint32_t HybridDavinciModel::GetDeviceId() const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDeviceId();
}

uint64_t HybridDavinciModel::GetGlobalStepAddr() const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param: impl_ is nullptr, check invalid");
    GELOGE(ge::FAILED, "[Check][Param: impl_]null is invalid");
    return 0UL;
  }
  return impl_->GetGlobalStepAddr();
}

Status HybridDavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info,
                                               int32_t &dynamic_type) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDynamicBatchInfo(batch_info, dynamic_type);
}

void HybridDavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const {
  if (impl_ != nullptr) {
    impl_->GetUserDesignateShapeOrder(user_input_shape_order);
  }
}

void HybridDavinciModel::GetOutputShapeInfo(std::vector<std::string> &dynamic_output_shape_info) const {
  if (impl_ != nullptr) {
    impl_->GetModelAttr(dynamic_output_shape_info);
  }
}

Status HybridDavinciModel::GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                                  std::vector<InputOutputDescInfo> &output_desc,
                                                  std::vector<uint32_t> &input_formats,
                                                  std::vector<uint32_t> &output_formats) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats);
}

void HybridDavinciModel::SetModelDescVersion(const bool is_new_model_desc) {
  if (impl_ != nullptr) {
    impl_->SetModelDescVersion(is_new_model_desc);
  }
}

uint64_t HybridDavinciModel::GetSessionId() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetSessionId();
}

uint32_t HybridDavinciModel::GetDataInputerSize() const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDataInputerSize();
}

bool HybridDavinciModel::GetRunningFlag() const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->GetRunningFlag();
}

Status HybridDavinciModel::SetRunAsyncListenerCallback(
  const RunAsyncCallbackV2 &callback) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->SetRunAsyncListenerCallback(callback);
}

bool HybridDavinciModel::GetOpDescInfo(const uint32_t stream_id, const uint32_t task_id,
                                       OpDescInfo &op_desc_info) const {
  if (impl_ == nullptr) {
    return false;
  }
  auto context = impl_->GeContext();
  GE_RT_FALSE_CHECK_NOTNULL(context);
  const bool ret =
      context->exception_dumper.GetOpDescInfo(OpDescInfoId(stream_id, task_id, GetDeviceId()), op_desc_info);
  if (!ret) {
    for (const auto &iter : context->davinci_model) {
      if (iter->GetOpDescInfo(stream_id, task_id, op_desc_info)) {
        return true;
      }
    }
  }
  return ret;
}

Status HybridDavinciModel::GetOpAttr(const std::string &op_name, const std::string &attr_name,
                                     std::string &attr_value) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetOpAttr(op_name, attr_name, attr_value);
}

Status HybridDavinciModel::GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetAippInfo(index, aipp_info);
}

Status HybridDavinciModel::GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_data_index) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetAippType(index, aipp_type, aipp_data_index);
}

Status HybridDavinciModel::ReportProfilingData() const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ReportProfilingData();
}
}  // namespace hybrid
}  // namespace ge
