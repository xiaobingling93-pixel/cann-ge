/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HYBRID_HYBRID_DAVINCI_MODEL_H_
#define HYBRID_HYBRID_DAVINCI_MODEL_H_

#include <memory>
#include "ge/ge_api_error_codes.h"
#include "graph/load/model_manager/data_inputer.h"
#include "common/model/ge_root_model.h"
#include "exe_graph/runtime/tensor.h"
#include "acl/acl_rt.h"

namespace ge {
namespace hybrid {
class HybridDavinciModel {
 public:
  virtual ~HybridDavinciModel();

  HybridDavinciModel(const HybridDavinciModel &) = delete;
  HybridDavinciModel(HybridDavinciModel &&) = delete;
  HybridDavinciModel &operator=(const HybridDavinciModel &) = delete;
  HybridDavinciModel &operator=(HybridDavinciModel &&) = delete;

  static std::unique_ptr<HybridDavinciModel> Create(const GeRootModelPtr &ge_root_model);

  Status Init();

  virtual Status Execute(const std::vector<DataBuffer> &inputs,
                         const std::vector<GeTensorDesc> &input_desc,
                         std::vector<DataBuffer> &outputs,
                         std::vector<GeTensorDesc> &output_desc,
                         const aclrtStream stream);

  Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  Status ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                                const aclrtStream stream = nullptr);
  Status ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                                    std::vector<gert::Tensor> &outputs,
                                                    const aclrtStream stream = nullptr);

  Status ModelRunStart();

  Status ModelRunStop();

  Status EnqueueData(const std::shared_ptr<RunArgs> &args);

  void SetListener(const shared_ptr<ModelListener> &listener);

  void SetModelId(const uint32_t model_id);

  void SetDeviceId(const uint32_t device_id);

  void SetOmName(const std::string &om_name);

  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir);

  void SetLoadStream(const aclrtStream stream);

  uint64_t GetSessionId();

  uint32_t GetDeviceId() const;

  uint64_t GetGlobalStepAddr() const;

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const;

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const;

  void GetOutputShapeInfo(std::vector<std::string> &dynamic_output_shape_info) const;

  Status GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &output_formats);

  void SetModelDescVersion(const bool is_new_model_desc);

  uint32_t GetDataInputerSize() const;

  bool GetRunningFlag() const;

  Status SetRunAsyncListenerCallback(const RunAsyncCallbackV2 &callback);

  bool GetOpDescInfo(const uint32_t stream_id, const uint32_t task_id, OpDescInfo &op_desc_info) const;

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_value) const;

  Status GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const;

  Status GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_data_index) const;

  Status ReportProfilingData() const;

 private:
  HybridDavinciModel() = default;
  class Impl;
  Impl *impl_ = nullptr;
};
}  // namespace hybrid
}  // namespace ge
#endif // HYBRID_HYBRID_DAVINCI_MODEL_H_
