/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/execute/graph_executor.h"
#include "base/err_mgr.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/thread_pool.h"
#include "common/memory/tensor_trans_utils.h"

namespace {
constexpr size_t kMemAlignment = 64U;
}
namespace ge {
using Uint32Pair = std::pair<uint32_t, uint32_t>;

Status GraphExecutor::SetDynamicSize(const uint32_t model_id, const std::vector<uint64_t> &batch_num,
                                     const int32_t dynamic_type) {
  const auto ret = ModelManager::GetInstance().SetDynamicSize(model_id, batch_num, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][DynamicSize] failed, model_id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::PrepareOutput(const std::vector<InputOutputDescInfo> &output_desc,
  std::vector<gert::Tensor> &output_tensor) const {
  for (size_t i = 0UL; i < output_desc.size(); ++i) {
    GeTensor ge_tensor;
    ge_tensor.MutableTensorDesc().SetShape(GeShape(output_desc[i].shape_info.dims));
    ge_tensor.MutableTensorDesc().SetDataType(static_cast<DataType>(output_desc[i].data_type));
    const auto aligned_ptr = MakeShared<AlignedPtr>(output_desc[i].size, kMemAlignment);
    GE_ASSERT_NOTNULL(aligned_ptr);
    (void)ge_tensor.SetData(aligned_ptr, output_desc[i].size);
    ge_tensor.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);

    gert::Tensor gert_tensor;
    GE_ASSERT_SUCCESS(TensorTransUtils::GeTensor2GertTensor(ge_tensor, gert_tensor));
    output_tensor.emplace_back(std::move(gert_tensor));
  }
  return SUCCESS;
}

Status GraphExecutor::SyncExecuteModel(const uint32_t model_id, const std::vector<gert::Tensor> &input_tensor,
                                       std::vector<gert::Tensor> &output_tensor,
                                       const error_message::ErrorManagerContext &error_context) const {
  error_message::SetErrMgrContext(error_context);
  if (ModelManager::GetInstance().IsDynamicShape(model_id)) {
    GELOGI("[ExecuteGraph] SyncExecuteModel via dynamic shape model executor, modelId=%u", model_id);
    return ModelManager::GetInstance().SyncExecuteHybridModel(model_id, input_tensor, output_tensor);
  }

  // Prepare input and output
  std::vector<InputOutputDescInfo> inputs_desc;
  std::vector<InputOutputDescInfo> output_desc;

  GELOGI("[ExecuteGraph] GetInputOutputDescInfo via new ome begin, modelId=%u.", model_id);
  Status ret = GetInputOutputDescInfo(model_id, inputs_desc, output_desc);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_GET_IN_OUT_FAILED, "[Get][InputOutputDescInfo] failed, modelId=%u.", model_id);
    return GE_GRAPH_GET_IN_OUT_FAILED;
  }

  ret = PrepareOutput(output_desc, output_tensor);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_PREPARE_FAILED, "[Prepare][InputData] failed, modelId=%u.", model_id);
    return GE_GRAPH_PREPARE_FAILED;
  }
  // Run mode synchronize
  GELOGI("[ExecuteGraph] SyncExecuteModel via new ome begin.");
  GE_CHK_STATUS_RET_NOLOG(ModelManager::GetInstance().SyncExecuteModel(model_id, input_tensor, output_tensor));
  GELOGI("[GraphExecutor] execute model success, modelId=%u.", model_id);

  return SUCCESS;
}

Status GraphExecutor::ExecuteGraph(const GraphId graph_id, const GeRootModelPtr &ge_root_model,
                                   const std::vector<gert::Tensor> &input_tensor,
                                   std::vector<gert::Tensor> &output_tensor) const {
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHK_STATUS_RET_NOLOG(ModelManager::GetInstance().ModelSubscribe(graph_id));
  Status ret = SUCCESS;
  ret = SyncExecuteModel(ge_root_model->GetModelId(), input_tensor, output_tensor,
                         error_message::GetErrMgrContext());
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_SYNC_MODEL_FAILED, "[SyncExecute][Model] Error! graph id:%u", graph_id);
    return GE_GRAPH_SYNC_MODEL_FAILED;
  }
  return SUCCESS;
}

Status GraphExecutor::ExecuteGraphAsync(const GeRootModelPtr &ge_root_model, const std::shared_ptr<RunArgs> &args) const {
  GraphId graph_id = args->graph_id;
  GELOGI("[GraphExecutor] Start to async execute graph, graph_id=%u", graph_id);
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHK_STATUS_RET_NOLOG(ModelManager::GetInstance().ModelSubscribe(graph_id));
  Status ret = SUCCESS;
  ret = AsyncExecuteModelArgsPtr(ge_root_model, GetExecuteModelId(ge_root_model), args,
                                 error_message::GetErrMgrContext());
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_SYNC_MODEL_FAILED, "[AsyncExecute][Model] Error! graph id:%u", graph_id);
    return GE_GRAPH_SYNC_MODEL_FAILED;
  }

  GELOGI("[GraphExecutor] Async execute graph success, graph_id=%u", graph_id);
  return SUCCESS;
}

Status GraphExecutor::ExecuteGraphWithStream(rtStream_t const stream, const GraphNodePtr &graph_node,
                                             const GeRootModelPtr &ge_root_model,
                                             const std::vector<gert::Tensor> &input_tensor,
                                             std::vector<gert::Tensor> &output_tensor) const {
  GE_CHECK_NOTNULL(ge_root_model);
  const auto model_id = ge_root_model->GetModelId();
  const auto ret = ModelManager::GetInstance().ExecuteModelWithStreamAsync(model_id, graph_node, input_tensor,
                                                                           output_tensor, stream);
  if (ret != SUCCESS) {
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::ExecuteGraphWithStream(rtStream_t const stream, const GraphNodePtr &graph_node,
                                             const GeRootModelPtr &ge_root_model,
                                             const std::vector<GeTensor> &input_tensor,
                                             std::vector<GeTensor> &output_tensor) const {
  GE_CHECK_NOTNULL(ge_root_model);
  const auto model_id = ge_root_model->GetModelId();
  const auto ret = ModelManager::GetInstance().ExecuteModelWithStreamAsync(model_id, graph_node, input_tensor,
                                                                           output_tensor, stream);
  if (ret != SUCCESS) {
    return ret;
  }

  return SUCCESS;
}

uint32_t GraphExecutor::GetExecuteModelId(const GeRootModelPtr &ge_root_model) {
  const std::vector<uint32_t> &model_ids = ge_root_model->GetAllModelId();
  if (model_ids.empty()) {
    return kInvalidModelId;
  }
  if (model_ids.size() == 1U) {
    return ge_root_model->GetModelId();
  }

  uint32_t ret_model_id = kInvalidModelId;
  uint32_t min_load = std::numeric_limits<uint32_t>::max();
  for (uint32_t model_id : model_ids) {
    const uint32_t input_load = ModelManager::GetInstance().GetDataInputerSize(model_id);
    const uint32_t running_load = ModelManager::GetInstance().GetRunningFlag(model_id);
    const uint32_t load = input_load + running_load;
    if (load <= min_load) {
      min_load = load;
      ret_model_id = model_id;
    }
  }
  return ret_model_id;
}

Status GraphExecutor::AsyncExecuteModelArgsPtr(const GeRootModelPtr &ge_root_model, const uint32_t model_id,
                                               const std::shared_ptr<RunArgs> &args,
                                               const error_message::ErrorManagerContext &error_context) const {
  error_message::SetErrMgrContext(error_context);
  if (model_id == kInvalidModelId) {
    GELOGE(INTERNAL_ERROR, "No valid model id.");
    return INTERNAL_ERROR;
  }
  try {
    GELOGI("RunAsync begin.model_id %u", model_id);
    if (ModelManager::GetInstance().SetCallbackHybridLoad(model_id, ge_root_model, args->callback) != SUCCESS) {
      GELOGE(FAILED, "[Set][CallBack] for model fail, model_id %u", model_id);
      return FAILED;
    }

    const auto ret = ModelManager::GetInstance().DataInputTensor(model_id, args);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][DataInputTensor] RunAsync: DataInput fail, model_id %u", model_id);
      return ret;
    }

    GELOGI("RunAsync success.");
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERR_MSG("E19999", "Bad memory allocation exception occur failed, model_id %u", model_id);
    GELOGE(MEMALLOC_FAILED, "[Run][Async] failed, bad memory allocation occur, model_id %u", model_id);
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Some exceptions occur failed, model_id %u", model_id);
    GELOGE(FAILED, "[Run][Async] failed, some exceptions occur, model_id %u", model_id);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                             std::vector<InputOutputDescInfo> &output_desc) {
  try {
    const auto ret = ModelManager::GetInstance().GetInputOutputDescInfo(model_id, input_desc, output_desc);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][InputOutputDescInfo] failed, model_id:%u.", model_id);
      return ret;
    }
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERR_MSG("E19999", "Bad memory allocation exception occur failed, model_id:%u.", model_id);
    GELOGE(MEMALLOC_FAILED, "[Get][InputOutputDescInfo] failed, bad memory allocation occur, model_id:%u.", model_id);
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Some exceptions occur failed, model_id:%u.", model_id);
    GELOGE(FAILED, "[Get][InputOutputDescInfo] failed, some exceptions occur, model_id:%u.", model_id);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                             std::vector<InputOutputDescInfo> &output_desc,
                                             std::vector<uint32_t> &input_formats, std::vector<uint32_t> &out_formats,
                                             const bool new_model_desc) {
  try {
    const auto ret = ModelManager::GetInstance().GetInputOutputDescInfo(model_id, input_desc, output_desc,
                                                                        input_formats, out_formats, new_model_desc);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][InputOutputDescInfo] failed, model_id:%u.", model_id);
      return ret;
    }
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERR_MSG("E19999", "Bad memory allocation exception occur failed, model_id:%u.", model_id);
    GELOGE(MEMALLOC_FAILED, "[Get][InputOutputDescInfo] failed, bad memory allocation occur, model_id:%u.", model_id);
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "Some exceptions occur failed, model_id:%u.", model_id);
    GELOGE(FAILED, "[Get][InputOutputDescInfo] failed, some exceptions occur, model_id:%u.", model_id);
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @param [out] dynamic_type
/// @return execute result
Status GraphExecutor::GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                          int32_t &dynamic_type) {
  const auto ret = ModelManager::GetInstance().GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][DynamicBatchInfo] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
Status GraphExecutor::GetCombinedDynamicDims(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info) {
  const auto ret = ModelManager::GetInstance().GetCombinedDynamicDims(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GetCombinedDynamicDims] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

/// @ingroup ge
/// @brief Get user designate shape order
/// @param [in] model_id
/// @param [out] user_input_shape_order
/// @return execute result
Status GraphExecutor::GetUserDesignateShapeOrder(const uint32_t model_id,
                                                 std::vector<std::string> &user_input_shape_order) {
  const auto ret = ModelManager::GetInstance().GetUserDesignateShapeOrder(model_id, user_input_shape_order);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][UserDesignateShapeOrder] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetCurrentShape(const uint32_t model_id, std::vector<int64_t> &batch_info,
                                      int32_t &dynamic_type) {
  const auto ret = ModelManager::GetInstance().GetCurrentShape(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][CurShape] failed, model_id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetNodeAttr(const uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                                  std::string &attr_value) {
  const auto ret = ModelManager::GetInstance().GetNodeAttr(model_id, op_name, attr_name, attr_value);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OpAttr]Get op:%s attr:%s failed.", op_name.c_str(), attr_name.c_str());
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetOutputShapeInfo(const uint32_t model_id, std::vector<std::string> &dynamic_output_shape_info) {
  const auto ret = ModelManager::GetInstance().GetOutputShapeInfo(model_id, dynamic_output_shape_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelAttr] failed, model_id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetAippInfo(const uint32_t model_id, const uint32_t index, AippConfigInfo &aipp_info) {
  const auto ret = ModelManager::GetInstance().GetAippInfo(model_id, index, aipp_info);
  if (ret != SUCCESS) {
    GELOGW("GetAIPPInfo is not success.");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetAippType(const uint32_t model_id, const uint32_t index, InputAippType &type,
                                  size_t &aipp_index) {
  const auto ret = ModelManager::GetInstance().GetAippType(model_id, index, type, aipp_index);
  if (ret != SUCCESS) {
    GELOGW("Get aipp type is not success.");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetOrigInputInfo(const uint32_t model_id, const uint32_t index,
                                       OriginInputInfo &orig_input_info) {
  const auto ret = ModelManager::GetInstance().GetOrigInputInfo(model_id, index, orig_input_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OrigInputInfo] failed, model_id:%u, index:%u.", model_id, index);
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::GetAllAippInputOutputDims(const uint32_t model_id, const uint32_t index,
                                                std::vector<InputOutputDims> &input_dims,
                                                std::vector<InputOutputDims> &output_dims) {
  const auto ret = ModelManager::GetInstance().GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][AllAippInputOutputDims] failed, model_id:%u, index:%u.", model_id, index);
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::GetOpDescInfo(const uint32_t device_id, const uint32_t stream_id, const uint32_t task_id,
                                    OpDescInfo &op_desc_info) {
  const auto ret = ModelManager::GetInstance().GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OpDescInfo] failed, device_id:%u, stream_id:%u, task_id:%u.",
           device_id, stream_id, task_id);
    return ret;
  }
  return SUCCESS;
}
}  // namespace ge
