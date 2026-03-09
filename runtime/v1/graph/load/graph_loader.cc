/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/graph_loader.h"

#include "common/helper/model_parser_base.h"
#include "graph/ge_context.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/thread_pool/thread_pool.h"
#include "framework/common/types.h"
#include "base/err_mgr.h"

namespace ge {
Status GraphLoader::UnloadModel(const uint32_t model_id) {
  if (model_id == INVALID_MODEL_ID) {
    return SUCCESS;
  }
  GELOGI("UnLoad model begin, model id: %u.", model_id);
  GE_CHK_STATUS(ModelManager::GetInstance().Stop(model_id), "[Stop][Model] failed. model id:%u", model_id);

  GE_CHK_STATUS_RET(ModelManager::GetInstance().Unload(model_id), "[Unload][Model] failed. model id:%u", model_id);

  GELOGI("UnLoad model success, model id: %u.", model_id);
  return SUCCESS;
}

Status GraphLoader::LoadModelOnline(uint32_t &model_id, const GeRootModelPtr &ge_root_model,
                                    const GraphNodePtr &graph_node, const uint32_t device_id,
                                    const error_message::ErrorManagerContext &error_context,
                                    const rtStream_t stream) {
  error_message::SetErrMgrContext(error_context);
  GELOGI("Load model online begin.");
  if (ge_root_model == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Check param ge_root_model_ptr nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraph][Check][Param] GE load graph model_ptr is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  GE_CHK_STATUS_RET(ModelUtils::SetDevice(device_id), "[Call][SetDevice] failed, device_id:%u", device_id);
  GE_MAKE_GUARD(reset_device, [&device_id]() {
    GE_CHK_STATUS(ModelUtils::ResetDevice(device_id));
  });

  auto &model_mgr = ModelManager::GetInstance();
  GE_CHK_STATUS_RET_NOLOG(model_mgr.LoadModelOnline(model_id, ge_root_model, graph_node, device_id, stream));

  ge_root_model->SetModelId(model_id);
  if (ge_root_model->IsSpecificStream()) {
    GELOGI("No need to start a new thread to run model in specific scene.");
    return SUCCESS;
  }
  const auto ret = model_mgr.Start(model_id);
  if (ret != SUCCESS) {
    GE_CHK_STATUS(model_mgr.Unload(model_id), "[Unload][Model] failed after start failed, model_id:%u.", model_id);
    GELOGE(ret, "[Start][Model] failed, model_id:%u.", model_id);
    return ret;
  }
  GELOGI("Load model online success, model_id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::LoadDataFromFile(const std::string &path, const int32_t priority, ModelData &model_data) {
  if (!CheckInputPathValid(path, "model_file")) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param] model path is invalid:%s", path.c_str());
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  GELOGI("Load model begin, model path is: %s", path.c_str());

  const Status ret = ModelParserBase::LoadFromFile(path.c_str(), priority, model_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][LoadFromFile] failed. ret = %u, path:%s", ret, path.c_str());
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char_t *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  }
  return ret;
}

Status GraphLoader::LoadModelFromData(const ModelData &model_data, const ModelParam &model_param, uint32_t &model_id) {
  GELOGI("Load model begin, model_id:%u.", model_id);
  // For ACL, Open Device from App.
  const auto ret = ModelManager::GetInstance().LoadModelOffline(model_data, model_param, model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed, model_id:%u.", model_id);
    return ret;
  }
  GELOGI("Load model success, model_id:%u.", model_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Load task list from ModelData with queue.
/// @param [out] model_id: model id allocate from manager.
/// @param [in] model_data: Model data load from offline model.
/// @param [in] input_queue_ids: input queue ids create from user.
/// @param [in] output_queue_ids: input queue ids create from user.
/// @return: 0 for success / others for fail
///
Status GraphLoader::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data, const ModelQueueArg &arg) {
  GELOGI("Load model with queue begin, model_id:%u.", model_id);

  // For ACL, Open Device from App.
  const auto ret = ModelManager::GetInstance().LoadModelWithQ(model_id, model_data, arg);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] with queue failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGI("Load model with queue success, model_id:%u.", model_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Load task list from root model without input queue and output queue.
/// @param [out] model_id: model id allocate from manager.
/// @param [in] root_model: instance of GeRootModel.
/// @return: 0 for success / others for fail
///
Status GraphLoader::LoadModelWithoutQ(uint32_t &model_id, const GeRootModelPtr &root_model) {
  GELOGI("Load model without queue begin, model_id: %u.", model_id);

  const auto ret = ModelManager::GetInstance().LoadModelWithoutQ(model_id, root_model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] without queue failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGI("Load model without queue success, model_id: %u.", model_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Load task list from GeRootModelPtr with queue and params.
/// @param [out] model_id: model id allocate from manager.
/// @param [in] root_model: instance of GeRootModel.
/// @param [in] model_queue_param: params and queue ids and create from user.
/// @return: 0 for success / others for fail
///
Status GraphLoader::LoadModelWithQueueParam(uint32_t &model_id,
                                            const GeRootModelPtr &root_model,
                                            const ModelQueueParam &model_queue_param,
                                            const bool need_update_session_id) {
  GELOGI("Load model with queue and params begin, model_id:%u.", model_id);
  // For ACL, Open Device from App.
  const auto ret = ModelManager::GetInstance().LoadModelWithQueueParam(model_id, root_model, model_queue_param,
                                                                       0, need_update_session_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] with queue and params failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGI("Load model with queue and params success, model_id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::LoadModelWithQueueParam(uint32_t &model_id, const ModelData &model_data,
                                            const ModelQueueParam &model_queue_param) {
  return ModelManager::GetInstance().LoadModelWithQueueParam(model_id, model_data, model_queue_param);
}

///
/// @ingroup domi_ome
/// @brief  execute model
/// @param [in] model_id  model id
/// @param [in] stream   stream to execute model on
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data
/// @param [in] input_desc  description of model input data
/// @param [out] output_data  model output data
/// @param [out] output_desc  description of model output data
///
Status GraphLoader::ExecuteModel(const uint32_t model_id, rtStream_t const stream, const bool async_mode,
                                 const InputData &input_data, const std::vector<GeTensorDesc> &input_desc,
                                 OutputData &output_data, std::vector<GeTensorDesc> &output_desc) {
  const auto ret = ModelManager::GetInstance().ExecuteModel(model_id, stream, async_mode,
                                                            input_data, input_desc, output_data, output_desc,
                                                            {}, {});
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Model] failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGD("Execute model success, model_id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::GetModelDescInfoFromMem(const ModelData &model_data, ModelInOutInfo &info) {
  ModelPartition partition;
  partition.type = MODEL_INOUT_INFO;
  ModelHelper model_helper;
  GE_CHK_STATUS_RET_NOLOG(model_helper.LoadPartInfoFromModel(model_data, partition));
  std::vector<ModelDescTlvConfig> tlv_config;
  size_t offset = 0U;
  while (offset < static_cast<size_t>(partition.size)) {
    ModelDescTlvConfig config;
    GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t)),
                   "[Check][Param] offset:%" PRIu64 " is beyond the UINT64_MAX", offset);
    GE_CHECK_LE((offset + sizeof(uint32_t)), static_cast<size_t>(partition.size));
    const uint32_t type =
              *PtrToPtr<void, const uint32_t>(ValueToPtr(PtrToValue(partition.data) + static_cast<uint64_t>(offset)));
    
    config.type = static_cast<int32_t>(type);
    offset += sizeof(uint32_t);
    GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t)),
                   "[Check][Param] offset:%" PRIu64 " is beyond the UINT64_MAX", offset);
    GE_CHECK_LE((offset + sizeof(uint32_t)), static_cast<size_t>(partition.size));
    const uint32_t len =
              *PtrToPtr<void, const uint32_t>(ValueToPtr(PtrToValue(partition.data) + static_cast<uint64_t>(offset)));
    config.length = len;
    offset += sizeof(uint32_t);
    GELOGD("get current type %u, length is %u, total size is %zu, base ptr is %p",
           type, len, partition.size, partition.data);
    config.value = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(partition.data) + static_cast<uint64_t>(offset)));
    GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, static_cast<uint64_t>(len)),
                      "[Check][Param] offset:%" PRIu64 " is beyond the UINT64_MAX, len:%u", offset, len);
    GE_CHECK_LE((offset + len), static_cast<size_t>(partition.size));
    offset += len;
    tlv_config.emplace_back(config);
  }

  using GetModelInOutInfoFunc = ge::Status (*)(const uint8_t *data, size_t size, ge::ModelInOutInfo &info);
  std::map<ge::ModelDescType, GetModelInOutInfoFunc> GetModelInOutInfoFuncMap = {
      {ge::ModelDescType::MODEL_INPUT_DESC, &ge::ModelParserBase::GetModelInputDesc},
      {ge::ModelDescType::MODEL_OUTPUT_DESC, &ge::ModelParserBase::GetModelOutputDesc},
      {ge::ModelDescType::MODEL_DYNAMIC_BATCH, &ge::ModelParserBase::GetDynamicBatch},
      {ge::ModelDescType::MODEL_DYNAMIC_HW, &ge::ModelParserBase::GetDynamicHW},
      {ge::ModelDescType::MODEL_DYNAMIC_DIMS, &ge::ModelParserBase::GetDynamicDims},
      {ge::ModelDescType::MODEL_DYNAMIC_OUTPUT_SHAPE, &ge::ModelParserBase::GetDynamicOutShape},
      {ge::ModelDescType::MODEL_DESIGNATE_SHAPE_ORDER, &ge::ModelParserBase::GetDataNameOrder},
  };
  for (auto &cfg : tlv_config) {
    const auto it = GetModelInOutInfoFuncMap.find(static_cast<ModelDescType>(cfg.type));
    GE_IF_BOOL_EXEC(it == GetModelInOutInfoFuncMap.end(),
                  GELOGE(FAILED, "get type failed, type is %d", cfg.type);
                  return FAILED);
    GELOGD("start to analyze type is %d, len is %u", cfg.type, cfg.length);
    GE_CHK_STATUS_RET_NOLOG(it->second(cfg.value, static_cast<size_t>(cfg.length), info));
  }
  return SUCCESS;
}

Status GraphLoader::GetRuntimeModelId(const uint32_t model_id, uint32_t &model_runtime_id)
{
  return ModelManager::GetInstance().GetRuntimeModelId(model_id, model_runtime_id);
}
}  // namespace ge
