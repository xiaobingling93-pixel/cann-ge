/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_HETEROGENEOUS_MODEL_IO_HELPER_H_
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_HETEROGENEOUS_MODEL_IO_HELPER_H_

#include <thread>
#include <functional>
#include "common/thread_pool/thread_pool.h"
#include "ge/ge_data_flow_api.h"
#include "dflow/base/deploy/exchange_service.h"
#include "graph/ge_tensor.h"
#include "dflow/base/deploy/model_deployer.h"
#include "flow_msg.h"
#include "framework/common/runtime_tensor_desc.h"

namespace ge {
using EnqueueTask = std::function<Status(const DeployQueueAttr &)>;
class HeterogeneousModelIoHelper {
 public:
  HeterogeneousModelIoHelper(const std::vector<DeployQueueAttr> &input_queue_attrs,
                             const std::vector<std::vector<DeployQueueAttr>> &broadcast_input_queue_attrs);

  virtual ~HeterogeneousModelIoHelper() = default;

  Status Initialize();

  /// @ingroup ge
  /// @brief Feed tensor to queue
  /// @param [in]  indexes         key: tensor index, value: input index
  /// @param [in]  inputs          input tensors
  /// @param [in]  control_info    control info
  /// @return SUCCESS success / others failure
  Status Feed(const std::map<size_t, size_t> &indexes,
              const std::vector<GeTensor> &inputs,
              const ExchangeService::ControlInfo &control_info);

  /// @ingroup ge
  /// @brief Feed flow msg to queue
  /// @param [in]  indexes         key: tensor index, value: input index
  /// @param [in]  inputs          input flow msg
  /// @param [in]  control_info    control info
  /// @return SUCCESS success / others failure
  Status FeedFlowMsg(const std::map<size_t, size_t> &indexes,
                     const std::vector<FlowMsgBasePtr> &inputs,
                     const ExchangeService::ControlInfo &control_info);

  /// @ingroup ge
  /// @brief Fetch flow msg from queue
  /// @param [in]  queue_attr      queue attr
  /// @param [in]  control_info    control info
  /// @param [out] flow_msg        output flow msg
  /// @return SUCCESS success / others failure
  Status FetchFlowMsg(const DeployQueueAttr &queue_attr,
                      const ExchangeService::ControlInfo &control_info,
                      const GeTensorDescPtr &output_desc,
                      FlowMsgBasePtr &flow_msg) const;

  /// @ingroup ge
  /// @brief Feed tensor to queue
  /// @param [in]  raw_data_list   raw data list, can be 1 or n
  /// @param [in]  index           input index
  /// @param [in]  control_info    control info
  /// @return SUCCESS success / others failure
  Status FeedRawData(const std::vector<RawData> &raw_data_list, const uint32_t index,
                     const ExchangeService::ControlInfo &control_info);

 private:
  Status ExecuteEnqueueTask(const EnqueueTask &enqueue_task,
                            const DeployQueueAttr &queue_attr,
                            std::vector<std::future<Status>> &fut_rets,
                            bool execute_parallel = false);

  static Status FillBuffInfos(const GeTensor &tensor,
                              RuntimeTensorDesc &tensor_desc,
                              std::vector<ExchangeService::BuffInfo> &buffs);

  Status EnqueueFlowMsg(const FlowMsgBasePtr &flow_msg,
                        const DeployQueueAttr &queue_attr,
                        const ExchangeService::ControlInfo &control_info) const;

  std::vector<DeployQueueAttr> input_queue_attrs_;
  std::vector<std::vector<DeployQueueAttr>> broadcast_input_queue_attrs_;
  std::unique_ptr<ThreadPool> pool_;
  ExchangeService *exchange_service_ = nullptr;
};
}  // namespace ge
#endif  // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_HETEROGENEOUS_MODEL_IO_HELPER_H_
