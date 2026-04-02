/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_EXECUTE_GRAPH_EXECUTOR_H_
#define GE_GRAPH_EXECUTE_GRAPH_EXECUTOR_H_

#include <cstdarg>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "ge/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/model/executor.h"
#include "exe_graph/runtime/tensor.h"
#include "base/err_mgr.h"

namespace ge {
class GraphExecutor {
 public:
  GraphExecutor() = default;

  virtual ~GraphExecutor() = default;

  Status ExecuteGraph(const GraphId graph_id, const GeRootModelPtr &ge_root_model,
                      const std::vector<gert::Tensor> &input_tensor, std::vector<gert::Tensor> &output_tensor) const;

  Status ExecuteGraphAsync(const GeRootModelPtr &ge_root_model, const std::shared_ptr<RunArgs> &args) const;
  Status ExecuteGraphWithStream(aclrtStream const stream, const GraphNodePtr &graph_node,
                                const GeRootModelPtr &ge_root_model,
                                const std::vector<GeTensor> &input_tensor,
                                std::vector<GeTensor> &output_tensor) const;
  
  Status ExecuteGraphWithStream(aclrtStream const stream, const GraphNodePtr &graph_node,
                                const GeRootModelPtr &ge_root_model,
                                const std::vector<gert::Tensor> &input_tensor,
                                std::vector<gert::Tensor> &output_tensor) const;

  static Status SetDynamicSize(const uint32_t model_id, const std::vector<uint64_t> &batch_num,
                               const int32_t dynamic_type);

  static Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                       std::vector<InputOutputDescInfo> &output_desc);

  static Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                       std::vector<InputOutputDescInfo> &output_desc,
                                       std::vector<uint32_t> &input_formats,
                                       std::vector<uint32_t> &out_formats, const bool new_model_desc = false);

  static Status GetAippInfo(const uint32_t model_id, const uint32_t index, AippConfigInfo &aipp_info);

  static Status GetAippType(const uint32_t model_id, const uint32_t index, InputAippType &type, size_t &aipp_index);

  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  static Status GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                    int32_t &dynamic_type);

  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @return execute result
  static Status GetCombinedDynamicDims(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info);

  /// @ingroup ge
  /// @brief Get user designate shape order
  /// @param [in] model_id
  /// @param [out] user_input_shape_order
  /// @return execute result
  static Status GetUserDesignateShapeOrder(const uint32_t model_id, std::vector<std::string> &user_input_shape_order);

  static Status GetCurrentShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);

  static Status GetNodeAttr(const uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                            std::string &attr_value);

  static Status GetOutputShapeInfo(const uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info);

  static Status GetOrigInputInfo(const uint32_t model_id, const uint32_t index, OriginInputInfo &orig_input_info);
  static Status GetAllAippInputOutputDims(const uint32_t model_id, const uint32_t index,
                                          std::vector<InputOutputDims> &input_dims,
                                          std::vector<InputOutputDims> &output_dims);

  static Status GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                              OpDescInfo &op_desc_info);

  static uint32_t GetExecuteModelId(const GeRootModelPtr &ge_root_model);

 private:
  Status PrepareOutput(const std::vector<InputOutputDescInfo> &output_desc,
    std::vector<gert::Tensor> &output_tensor) const;

  Status SyncExecuteModel(const uint32_t model_id, const std::vector<gert::Tensor> &input_tensor,
                          std::vector<gert::Tensor> &output_tensor,
                          const error_message::ErrorManagerContext &error_context) const;
  Status AsyncExecuteModelArgsPtr(const GeRootModelPtr &ge_root_model, const uint32_t model_id,
                                  const std::shared_ptr<RunArgs> &args,
                                  const error_message::ErrorManagerContext &error_context) const;
};

using SyncExecuteModelFunc = std::function<Status(GraphExecutor *executor,
                                                  const uint32_t model_id,
                                                  const std::vector<GeTensor> &input_tensor,
                                                  std::vector<GeTensor> &output_tensor,
                                                  const error_message::ErrorManagerContext &error_context)>;

using AsyncExecuteModelFunc = std::function<Status(GraphExecutor *executor,
                                                   const GeRootModelPtr &ge_root_model,
                                                   const uint32_t model_id,
                                                   const std::vector<Tensor> &inputs,
                                                   const error_message::ErrorManagerContext &error_context,
                                                   const RunAsyncCallback &callback)>;
}  // namespace ge

#endif  // GE_GRAPH_EXECUTE_GRAPH_EXECUTOR_H_
