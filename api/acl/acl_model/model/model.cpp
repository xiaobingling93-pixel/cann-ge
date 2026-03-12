/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_mdl.h"
#include <vector>
#include <mutex>
#include <string>
#include <queue>
#include "securec.h"
#include "acl/acl_base.h"
#include "executor/ge_executor.h"
#include "common/log_inner.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/shape.h"
#include "model_desc_internal.h"
#include "error_codes_inner.h"
#include "common/prof_api_reg.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/om2_model_executor.h"
#include "framework/runtime/gert_api.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "model_config.h"
#include "acl_resource_manager.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph/def_types.h"
#include "types/tensor_desc_internal.h"
#include "types/data_buffer_internal.h"
#include "acl_model_impl.h"

namespace {
constexpr size_t MIN_OUTPUT_SHAPE_INFO_SIZE = 2U;
constexpr size_t MAX_OUTPUT_SHAPE_INFO_SIZE = MIN_OUTPUT_SHAPE_INFO_SIZE + static_cast<size_t>(ACL_MAX_DIM_CNT);
constexpr size_t DYNAMIC_BATCH_SIZE = 1U;
constexpr size_t DYNAMIC_HW_SIZE = 2U;
constexpr size_t TENSOR_NAME_ATTR_NUM = 5U;
constexpr int32_t DEFAULT_SYNC_TIMEOUT = -1;
constexpr uint16_t kStartTag = 0U;
constexpr uint16_t kEndTag = 1U;
constexpr const char_t *TENSOR_NAME_PREFIX = "acl";
constexpr const char_t *TENSOR_INPUT_STR = "input";
constexpr const char_t *TENSOR_OUTPUT_STR = "output";
constexpr const char_t *MODEL_ID_STR = "modelId";
constexpr const char_t *OPTION_EXEC_REUSE_ZERO_COPY_MEMORY = "ge.exec.reuseZeroCopyMemory";

std::mutex aclmdlGetOpAttrMutex;
std::mutex aclmdlBundleMutex;

enum class DimsType : std::uint8_t {
    DIMS_TYPE_V1 = 0,
    DIMS_TYPE_V2
};

enum class TensorType : std::uint8_t {
    INPUT_TENSOR_TYPE = 0,
    OUTPUT_TENSOR_TYPE
};

aclError aclmdlCheckQueueParam(const uint32_t *const inputQ, const size_t inputQNum, const uint32_t *const outputQ,
                                const size_t outputQNum) {
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(inputQ);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(outputQ);
    if ((inputQNum == 0U) || (outputQNum == 0U)) {
        ACL_LOG_INNER_ERROR("[Check][QNum]inputQNum[%zu] or outputQNum[%zu] is invalid, can't be zero",
                            inputQNum, outputQNum);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

ge::ModelLoadArg ConstructGeModelLoadArg(void *devPtr, size_t memSize, void *weightPtr, size_t weightSize,
    gert::RtSession *rtSession = nullptr, const std::vector<ge::FileConstantMem> &fileConstantMems = {},
    const bool need_clear_dfx_cache = false) {
    ge::ModelLoadArg loadArgs;
    loadArgs.dev_ptr = devPtr;
    loadArgs.mem_size = memSize;
    loadArgs.weight_ptr = weightPtr;
    loadArgs.weight_size = weightSize;
    loadArgs.rt_session = rtSession;
    loadArgs.file_constant_mems = fileConstantMems;
    loadArgs.need_clear_dfx_cache = need_clear_dfx_cache;
    return loadArgs;
}
}

aclmdlDesc *aclmdlCreateDescImpl()
{
    return new(std::nothrow) aclmdlDesc();
}

aclError aclmdlDestroyDescImpl(aclmdlDesc *modelDesc)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_DELETE_AND_SET_NULL(modelDesc);
    return ACL_SUCCESS;
}

namespace acl {
static gert::Shape ConstructRtShapeFromShape(const ge::Shape &geShape) {
    gert::Shape rtShape;
    rtShape.SetDimNum(geShape.GetDimNum());
    for (size_t i = 0U; i < geShape.GetDimNum(); ++i) {
        rtShape.SetDim(i, geShape.GetDim(i));
    }
    return rtShape;
}

static aclError ParseBatchInfo(aclmdlDesc * const modelDesc, const int32_t dynamicType,
    const std::vector<std::vector<int64_t>> &batchInfo)
{
    const uint32_t modelId = modelDesc->modelId;
    if (dynamicType == static_cast<int32_t>(ge::DYNAMIC_DIMS)) { // dynamic dims, size can be [1, 4]
        const size_t dimCount = batchInfo[0U].size();
        for (size_t i = 0U; i < batchInfo.size(); ++i) {
            if (batchInfo[i].size() != dimCount) {
                ACL_LOG_INNER_ERROR("[Check][Size]Get dynamic model info invalid, model id[%u], one dim count is %zu "
                    "while another is %zu", modelId, dimCount, batchInfo[i].size());
                modelDesc->dynamicDims.clear();
                return ACL_ERROR_GE_FAILURE;
            }
            std::vector<uint64_t> oneDims;
            for (size_t j = 0U; j < dimCount; ++j) {
                oneDims.push_back(static_cast<uint64_t>(batchInfo[i][j]));
            }
            modelDesc->dynamicDims.push_back(oneDims);
        }
    } else if (batchInfo[0U].size() == DYNAMIC_BATCH_SIZE) { // dynamic batch,size is 1
        for (size_t i = 0U; i < batchInfo.size(); ++i) {
            if (batchInfo[i].size() != DYNAMIC_BATCH_SIZE) {
                ACL_LOG_INNER_ERROR("[Check][Size]get dynamic model info invalid, model id[%u]", modelId);
                modelDesc->dynamicBatch.clear();
                return ACL_ERROR_GE_FAILURE;
            }
            modelDesc->dynamicBatch.push_back(static_cast<uint64_t>(batchInfo[i][0U]));
        }
    } else if (batchInfo[0U].size() == DYNAMIC_HW_SIZE) { // dynamic hw,size is 2
        for (size_t i = 0U; i < batchInfo.size(); ++i) {
            if (batchInfo[i].size() != DYNAMIC_HW_SIZE) { // dynamic hw,size is 2
                ACL_LOG_INNER_ERROR("[Check][Size]get dynamic model info invalid, model id[%u]", modelId);
                modelDesc->dynamicHW.clear();
                return ACL_ERROR_GE_FAILURE;
            }
            modelDesc->dynamicHW.push_back({static_cast<uint64_t>(batchInfo[i][0U]),
                                            static_cast<uint64_t>(batchInfo[i][1U])});
        }
    } else {
        ACL_LOG_INNER_ERROR("[Get][DynamicModel]get dynamic model info invalid, model id[%u]", modelId);
        return ACL_ERROR_GE_FAILURE;
    }

    return ACL_SUCCESS;
}

static aclError GetDynamicTensorInfoHelp(aclmdlDesc * const modelDesc, const int32_t dynamicType,
    const std::vector<std::vector<int64_t>> &batchInfo)
{
    if (batchInfo.empty()) {
        ACL_LOG_INFO("model is not dynamic, batchInfo is empty, modelId[%u]", modelDesc->modelId);
        return ACL_SUCCESS;
    }

    ACL_LOG_INFO("model is dynamic, modelId[%u]", modelDesc->modelId);
    const aclError retVal = acl::ParseBatchInfo(modelDesc, dynamicType, batchInfo);
    if (retVal != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Parse][BatchInfo]get model dynamic info failed, result[%d], model id[%u]",
            retVal, modelDesc->modelId);
        return retVal;
    }

    return ACL_SUCCESS;
}

static aclError GetDynamicTensorInfo(aclmdlDesc * const modelDesc)
{
    ACL_LOG_DEBUG("call ge interface executor.GetDynamicBatchInfo");
    const uint32_t modelId = modelDesc->modelId;
    const auto om2Executor = acl::AclResourceManager::GetInstance().GetOm2Executor(modelId);
    std::vector<std::vector<int64_t>> batchInfo;
    int32_t dynamicType = static_cast<int32_t>(ge::FIXED);
    std::vector<std::string> userDesignateShapeOrder;
    if (om2Executor != nullptr) {
      ge::Status ret = om2Executor->GetDynamicBatchInfo(batchInfo, dynamicType);
      if (ret != ge::SUCCESS) {
        ACL_LOG_WARN("can not get dynamic model info, ge result[%u], model id[%u]", ret, modelId);
      }
      ret = om2Executor->GetUserDesignateShapeOrder(userDesignateShapeOrder);
      if (ret != ge::SUCCESS) {
        ACL_LOG_WARN("can not get user designate shape order, ge result[%u], model id[%u]", ret, modelId);
      }
    } else {
      ge::GeExecutor executor;
      ge::Status ret = executor.GetDynamicBatchInfo(modelId, batchInfo, dynamicType);
      if (ret != ge::SUCCESS) {
        ACL_LOG_WARN("can not get dynamic model info, ge result[%u], model id[%u]", ret, modelId);
      }
      ret = executor.GetUserDesignateShapeOrder(modelId, userDesignateShapeOrder);
      if (ret != ge::SUCCESS) {
        ACL_LOG_WARN("can not get user designate shape order, ge result[%u], model id[%u]", ret, modelId);
      }
    }
    modelDesc->dataNameOrder = userDesignateShapeOrder;
    return GetDynamicTensorInfoHelp(modelDesc, dynamicType, batchInfo);
}

static aclError RuntimeV2GetDynamicTensorInfo(aclmdlDesc * const modelDesc, const gert::ModelDesc &geModelDesc)
{
    ACL_LOG_DEBUG("call ge interface executor.GetDynamicBatchInfo");
    std::vector<std::vector<int64_t>> batchInfo;
    const uint32_t modelId = modelDesc->modelId;
    int32_t dynamicType = static_cast<int32_t>(ge::FIXED);
    ge::Status ret = geModelDesc.GetDynamicBatchInfo(batchInfo, dynamicType);
    if (ret != ge::SUCCESS) {
        ACL_LOG_WARN("get dynamic model info failed, ge result[%u], model id[%u]", ret, modelId);
    }
    std::vector<std::string> userDesignateShapeOrder;
    ret = geModelDesc.GetUserDesignateShapeOrder(userDesignateShapeOrder);
    if (ret != ge::SUCCESS) {
        ACL_LOG_WARN("get user designate shape order failed, ge result[%u], model id[%u]", ret, modelId);
    }
    modelDesc->dataNameOrder = userDesignateShapeOrder;

    return GetDynamicTensorInfoHelp(modelDesc, dynamicType, batchInfo);
}

static aclError GetModelOutputShapeInfoHelp(aclmdlDesc * const modelDesc,
    std::vector<std::string> &geDynamicOutputShape)
{
    if (geDynamicOutputShape.empty()) {
        ACL_LOG_INFO("model is not dynamic, geDynamicOutputShape is empty, modelId[%u]", modelDesc->modelId);
        return ACL_SUCCESS;
    }

    std::vector<std::vector<int64_t>> &dynamicOutputShape = modelDesc->dynamicOutputShape;
    for (auto &it : geDynamicOutputShape) {
        int64_t val = 0;
        int64_t negativeFlag = 1;
        std::vector<int64_t> outputShape;
        // ge uses string like "0:0:1,3,224,224" to represent output shape info,
        // acl converts string like "0:0:1,3,224,224" to vector<int64_t>
        for (auto &strIt : it) {
            if ((strIt >= '0') && (strIt <= '9')) { // numeric character
                val = (val * 10) + static_cast<int64_t>(strIt - '0'); // character to number
            } else if (strIt == '-') { // '-' represents that dynamic model has static output
                negativeFlag = -1;
                ACL_LOG_DEBUG("dynamic model include static output");
            } else {
                val *= negativeFlag;
                outputShape.emplace_back(val);
                val = 0;
                negativeFlag = 1;
            }
        }
        val *= negativeFlag;
        outputShape.emplace_back(val); // last value
        dynamicOutputShape.emplace_back(outputShape);
    }

    return ACL_SUCCESS;
}

static aclError GetModelOutputShapeInfo(aclmdlDesc *const modelDesc)
{
    ACL_LOG_DEBUG("call ge interface executor.GetModelAttr");
    const uint32_t modelId = modelDesc->modelId;
    std::vector<std::string> geDynamicOutputShape;
    ge::Status ret = ge::SUCCESS;
    const auto om2Executor = acl::AclResourceManager::GetInstance().GetOm2Executor(modelId);
    if (om2Executor != nullptr) {
        ret = om2Executor->GetModelAttrs(geDynamicOutputShape);
    } else {
        ge::GeExecutor executor;
        ret = executor.GetModelAttr(modelId, geDynamicOutputShape);
    }
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][ModelAttr]get model attribute failed, ge result[%u], model id[%u]", ret, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    return GetModelOutputShapeInfoHelp(modelDesc, geDynamicOutputShape);
}

static aclError RuntimeV2GetModelOutputShapeInfo(aclmdlDesc *const modelDesc, const gert::ModelDesc &geModeDesc)
{
    ACL_LOG_DEBUG("call ge interface executor.GetModelAttr");
    std::vector<std::string> geDynamicOutputShape;
    const uint32_t modelId = modelDesc->modelId;
    const ge::Status ret = geModeDesc.GetModelAttrs(geDynamicOutputShape);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][ModelAttr]get model attribute failed, ge result[%u], model id[%u]", ret, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    return GetModelOutputShapeInfoHelp(modelDesc, geDynamicOutputShape);
}

static aclError ModelLoadFromFileWithMem(const char_t *const modelPath, uint32_t *const modelId,
    const ge::ModelLoadArg &loadArgs, const int32_t priority)
{
    ACL_LOG_INFO("start to execute ModelLoadFromFileWithMem, modelPath[%s], "
        "workSize[%zu], weightSize[%zu], priority[%d]", modelPath, loadArgs.mem_size, loadArgs.weight_size, priority);

    ge::GeExecutor executor;
    uint32_t id = 0U;
    ge::ModelData data;
    const std::string path(modelPath);
    data.om_path = path;
    ACL_LOG_INFO("call ge interface executor.LoadDataFromFile, workSize[%zu], weightSize[%zu]",
        loadArgs.mem_size, loadArgs.weight_size);
    ge::Status ret = executor.LoadDataFromFile(path, data);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromFile]load model from file[%s] failed, ge result[%u]", modelPath, ret);
        ACL_DELETE_ARRAY(data.model_data);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    data.priority = priority;
    ACL_LOG_INFO("call ge interface executor.LoadModelFromDataWithArgs, workSize[%zu], weightSize[%zu]",
                 loadArgs.mem_size, loadArgs.weight_size);
    ret = executor.LoadModelFromDataWithArgs(id, data, loadArgs);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromData]load model from data failed, ge result[%u]", ret);
        ACL_DELETE_ARRAY(data.model_data);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    *modelId = id;
    ACL_DELETE_ARRAY(data.model_data);
    ACL_LOG_INFO("successfully execute ModelLoadFromFileWithMem, workSize[%zu], weightSize[%zu], modelId[%u]",
        loadArgs.mem_size, loadArgs.weight_size, *modelId);
    return ACL_SUCCESS;
}

static aclError RuntimeV2ModelLoadCommon(ge::ModelData &modelData, uint32_t *const modelId,
    const void *const weightPtr, const size_t weightSize, std::vector<ge::FileConstantMem> file_constant_mems,
    const std::shared_ptr<gert::RtSession> &rtSessionExternal)
{
    ACL_LOG_INFO("call ge interface gert::LoadExecutorFromModelData, weightSize[%zu]", weightSize);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    std::unique_ptr<gert::ModelV2Executor> executor;
    auto rtSession = (rtSessionExternal != nullptr) ? rtSessionExternal :
            acl::AclResourceManager::GetInstance().CreateRtSession();
    ACL_REQUIRES_NOT_NULL(rtSession);
    gert::LoadExecutorArgs args = {
        .rt_session = rtSession.get(),
        .file_constant_mems = std::move(file_constant_mems)
    };
    executor = gert::LoadExecutorFromModelData(modelData, args, ret);
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromData]call gert::LoadExecutorFromModelDataWithMem load model from data failed, "
                           "ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    ACL_REQUIRES_NOT_NULL(executor);
    // 4. load rt2.0 executor
    ACL_LOG_DEBUG("call ge interface executorV2.Load");
    aclrtStream rtStream = nullptr;
    ACL_REQUIRES_CALL_RTS_OK(aclrtCreateStream(&rtStream), aclrtCreateStream);

    gert::ModelExecuteArg exeArg;
    exeArg.stream = rtStream;
    ret = executor->Load(exeArg, gert::ModelLoadArg(rtSession.get(), {weightPtr, weightSize}));
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromData]call load executorV2 failed, ge result[%u]", ret);
        (void)aclrtDestroyStream(rtStream);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    ret = aclrtSynchronizeStream(rtStream);
    if (ret != ACL_ERROR_NONE) {
        ACL_LOG_CALL_ERROR("synchronize stream failed, runtime result = %d", static_cast<int32_t>(ret));
        (void)aclrtDestroyStream(rtStream);
        return ACL_GET_ERRCODE_RTS(ret);
    }
    (void)aclrtDestroyStream(rtStream);

    // 5. get model-id
    acl::AclResourceManager::GetInstance().AddExecutor(*modelId , std::move(executor), rtSession);
    return ACL_SUCCESS;
}

static aclError RuntimeV2ModelLoadFromFileWithMem(const char_t *const modelPath, uint32_t *const modelId,
                                                  void *const weightPtr, const size_t weightSize,
                                                  const int32_t priority,
                                                  std::vector<ge::FileConstantMem> file_constant_mems = {},
                                                  const std::shared_ptr<gert::RtSession> &rtSessionExternal = nullptr)
{
    ACL_LOG_INFO("start to execute RuntimeV2ModelLoadFromFileWithMem, priority[%d], weightSize[%zu]",
                 priority, weightSize);

    // 1. load model data from file
    ge::ModelData modelData;
    modelData.om_path = modelPath;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ACL_LOG_INFO("call ge interface gert::LoadDataFromFile");
    ret = gert::LoadDataFromFile(modelPath, modelData);
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][Model]failed to load model from file by runtime2.0, ge errorCode is %u", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    std::shared_ptr<void> dataAuto;
    dataAuto.reset(modelData.model_data, [](const void * const p) { delete[] static_cast<const uint8_t *>(p); });
    // 2. config model data
    modelData.priority = priority;
    // 3. get rt2.0 executor
    ACL_REQUIRES_OK(RuntimeV2ModelLoadCommon(modelData, modelId, weightPtr, weightSize, std::move(file_constant_mems),
                                             rtSessionExternal));
    ACL_LOG_INFO("successfully execute RuntimeV2ModelLoadFromFileWithMem, modelSize[%lu], modelId[%u], weightSize[%zu]",
                 modelData.model_len, *modelId, weightSize);
    return ACL_SUCCESS;
}

static aclError Om2ModelLoadFromFileWithMem(const char_t *const modelPath, uint32_t *const modelId,
                                            const std::shared_ptr<gert::RtSession> &rtSessionExternal = nullptr) {
    auto rtSession = (rtSessionExternal != nullptr) ? rtSessionExternal :
                                                    acl::AclResourceManager::GetInstance().CreateRtSession();
    ACL_REQUIRES_NOT_NULL(rtSession);
    ge::ModelData modelData;
    auto ret = gert::LoadOm2DataFromFile(modelPath, modelData);
    std::shared_ptr<void> dataGuarder;
    dataGuarder.reset(modelData.model_data, [](const void *const p) {
        if (p != nullptr) {
            delete[] static_cast<const uint8_t *>(p);
        }
    });
    if (ret != ge::SUCCESS) {
      ACL_LOG_CALL_ERROR("[Model][FromData]call gert::LoadOm2DataFromFile failed, ge result[%u]", ret);
      return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    std::unique_ptr<gert::Om2ModelExecutor> executor = gert::LoadOm2ExecutorFromData(modelData, ret);
    if (ret != ge::SUCCESS) {
      ACL_LOG_CALL_ERROR("[Model][FromData]call gert::LoadOm2ExecutorFromData failed, ge result[%u]", ret);
      return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    ACL_REQUIRES_NOT_NULL(executor);
    acl::AclResourceManager::GetInstance().AddOm2Executor(*modelId , std::move(executor), rtSession);
    return ACL_SUCCESS;
}

static aclError Om2ModelLoadFromMemWithMem(const void *const model, const size_t modelSize, uint32_t *const modelId) {
    ge::ModelData modelData = {};
    modelData.model_data = const_cast<void *>(model);
    modelData.model_len = static_cast<uint64_t>(modelSize);
    ge::Status errorStatus = ge::SUCCESS;
    std::unique_ptr<gert::Om2ModelExecutor> executor = gert::LoadOm2ExecutorFromData(modelData, errorStatus);
    if (errorStatus != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromData]call gert::LoadOm2ExecutorFromData failed, ge result[%u]", errorStatus);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(errorStatus));
    }
    ACL_REQUIRES_NOT_NULL(executor);
    acl::AclResourceManager::GetInstance().AddOm2Executor(*modelId, std::move(executor), nullptr);
    return ACL_SUCCESS;
}

static aclError ModelLoadFromMemWithMem(const void *const model, const size_t modelSize, const std::string &modelPath,
                                        uint32_t *const modelId, const ge::ModelLoadArg &loadArgs,
                                        const char_t *const weightPath, const int32_t priority)
{
    ACL_LOG_INFO("start to execute ModelLoadFromMemWithMem, workSize[%zu], weightSize[%zu], priority[%d]",
        loadArgs.mem_size, loadArgs.weight_size, priority);
    if (modelSize == 0U) {
        ACL_LOG_INNER_ERROR("[Check][ModelSize]modelSize[%zu] is invalid, should not be zero", modelSize);
        return ACL_ERROR_INVALID_PARAM;
    }

    ge::GeExecutor geExecutor;
    uint32_t id = 0U;
    ge::ModelData modelData;
    modelData.model_data = const_cast<void *>(model);
    modelData.model_len = static_cast<uint64_t>(modelSize);
    modelData.priority = priority;
    modelData.om_path = modelPath;
    if (weightPath != nullptr) {
        modelData.weight_path = std::string(weightPath);
        ACL_LOG_INFO("Load weight path is [%s]", modelData.weight_path.c_str());
    }
    ACL_LOG_INFO("call ge interface executor.LoadModelFromDataWithArgs, modelSize[%zu], workSize[%zu], weightSize[%zu]",
        modelSize, loadArgs.mem_size, loadArgs.weight_size);
    const auto ret = geExecutor.LoadModelFromDataWithArgs(id, modelData, loadArgs);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromData]load model from data failed, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    *modelId = id;
    ACL_LOG_INFO("successfully execute ModelLoadFromMemWithMem, modelSize[%zu], workSize[%zu], "
        "weightSize[%zu], modelId[%u]", modelSize, loadArgs.mem_size, loadArgs.weight_size, *modelId);
    return ACL_SUCCESS;
}

static aclError IsSupportRuntimeV2WithModelPath(const char *filePath, bool &isSupportRuntimeV2)
{
    if (!acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) {
        isSupportRuntimeV2 = false;
        return ACL_SUCCESS;
    }
    const auto ret = gert::IsDynamicModel(filePath, isSupportRuntimeV2);
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_ERROR("[Check][Model] failed, modelPath is null, or file is invalid, ret[%u].", ret);
        return static_cast<aclError>(ret);
    }
    return ACL_SUCCESS;
}

static aclError GetBundleNumAndOffset(const void *const model, const size_t modelSize,
                                      size_t &varSize, std::vector<std::pair<size_t, size_t>> &subModelOffsetAndSize)
{
  varSize = 0U;
  size_t currentOffset = 0U;
  if (modelSize < (sizeof(ge::ModelFileHeader) + sizeof(ge::ModelPartitionTable))) {
    ACL_LOG_ERROR("[Check][Param] Invalid model size, Model data size %zu must be greater than or equal to %zu.",
                  modelSize, sizeof(ge::ModelFileHeader));
    return ACL_ERROR_INVALID_PARAM;
  }
  const auto *fileHeader = ge::PtrToPtr<void, ge::ModelFileHeader>(model);
  if (fileHeader->modeltype != ge::MODEL_TYPE_BUNDLE_MODEL) {
    ACL_LOG_ERROR("this is not bundle om, please check");
    return ACL_ERROR_INVALID_PARAM;
  }
  currentOffset += sizeof(ge::ModelFileHeader);
  const auto *partitionTable =
      ge::PtrToPtr<void, ge::ModelPartitionTable>(ge::ValueToPtr(ge::PtrToValue(model) + currentOffset));
  const size_t partitionTableSize = ge::SizeOfModelPartitionTable(*partitionTable);
  ACL_LOG_INFO("get offset %zu, partitionTableSize %zu", currentOffset, partitionTableSize);
  ACL_REQUIRES_OK(acl::CheckSizeTAddOverflow(currentOffset, partitionTableSize, currentOffset));
  ACL_REQUIRES_LE(currentOffset, modelSize);
  for (size_t i = 0; i < partitionTable->num; ++i) {
    ACL_LOG_INFO("get %zu om offset %zu, size %zu", i, currentOffset, partitionTable->partition[i].mem_size);
    if (partitionTable->partition[i].type == ge::BUNDLE_MODEL_VAR_INFO) {
      varSize = *ge::PtrToPtr<void, int64_t>(ge::ValueToPtr(ge::PtrToValue(model) + currentOffset));
      ACL_LOG_INFO("get var size %zu", varSize);
    }
    if (partitionTable->partition[i].type == ge::BUNDLE_MODEL_INFO) {
      subModelOffsetAndSize.emplace_back(currentOffset, partitionTable->partition[i].mem_size);
    }
    ACL_REQUIRES_OK(acl::CheckSizeTAddOverflow(currentOffset,
                                               partitionTable->partition[i].mem_size, currentOffset));
    ACL_REQUIRES_LE(currentOffset, modelSize);
  }
  return ACL_SUCCESS;
}

    static aclError IsSupportRuntimeV2WithModelData(const void *const model, const size_t modelSize,
                                                    bool &isSupportRuntimeV2)
    {
      if (!acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) {
        isSupportRuntimeV2 = false;
        return ACL_SUCCESS;
      }
      ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
      if (modelSize < sizeof(ge::ModelFileHeader)) {
        ACL_LOG_ERROR("[Check][Param] Invalid model size, Model data size %zu must be greater than or equal to %zu.",
                      modelSize, sizeof(ge::ModelFileHeader));
        return ACL_ERROR_INVALID_PARAM;
      }
      const auto *fileHeader = reinterpret_cast<const ge::ModelFileHeader *>(model);
      constexpr uint32_t kStaticOmFileModelNum = 1U;
      constexpr uint8_t kDynamicOmFlag = 1U;
      isSupportRuntimeV2 =
              ((fileHeader->version >= ge::MODEL_VERSION) &&
               ((fileHeader->model_num > kStaticOmFileModelNum) || (fileHeader->is_unknow_model == kDynamicOmFlag)));
      return ACL_SUCCESS;
    }

static aclError CheckIsRuntimeV2WithConfig(const aclmdlConfigHandle* handle, const char *filePath,
                                           const void *const model, const size_t modelSize, bool &isSupportRuntimeV2) {
  if (filePath != nullptr) {
    ACL_REQUIRES_OK(IsSupportRuntimeV2WithModelPath(filePath, isSupportRuntimeV2));
  } else {
    ACL_REQUIRES_OK(IsSupportRuntimeV2WithModelData(model, modelSize, isSupportRuntimeV2));
  }
  if (isSupportRuntimeV2) {
    ACL_REQUIRES_NOT_NULL(handle);
    if (handle->attrState.find(ACL_MDL_WITHOUT_GRAPH_INT32) != handle->attrState.end()) {
      ACL_LOG_ERROR("ACL_MDL_WITHOUT_GRAPH_INT32 can not be configured when model is dynamic");
      const std::string errMsg = "ACL_MDL_WITHOUT_GRAPH_INT32 can not be configured when model is dynamic";
      acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
          std::vector<const char *>({"param", "value", "reason"}),
          std::vector<const char *>({"aclmdlConfigHandle", "ACL_MDL_WITHOUT_GRAPH_INT32", errMsg.c_str()}));
      return ACL_ERROR_INVALID_PARAM;
    }
  }
  return ACL_SUCCESS;
}
static aclError RuntimeV2ModelLoadFromMemWithMem(const void *const model, const size_t modelSize,
                                                 const std::string &modelPath,
                                                 uint32_t *const modelId, void *const weightPtr,
                                                 const size_t weightSize, const char_t *const weightPath,
                                                 const int32_t priority,
                                                 std::vector<ge::FileConstantMem> file_constant_mems = {},
                                                 const std::shared_ptr<gert::RtSession> &rtSessionExternal = nullptr)
{
    ACL_LOG_INFO("start to execute RuntimeV2ModelLoadFromMemWithMem, modelSize[%zu], priority[%d], weightSize[%zu]",
                 modelSize, priority, weightSize);
    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(modelSize != 0U, ACL_ERROR_INVALID_PARAM,
        "[Check][ModelSize]modelSize[%zu] is invalid, should not be zero", modelSize);
    // 1. config model data
    ge::ModelData modelData;
    modelData.model_data = const_cast<void *>(model);
    modelData.model_len = static_cast<uint64_t>(modelSize);
    modelData.priority = priority;
    modelData.om_path = modelPath;
    if (weightPath != nullptr) {
        modelData.weight_path = std::string(weightPath);
        ACL_LOG_INFO("Load weight path is [%s]", modelData.weight_path.c_str());
    }

    // 2. get executorV2
    ACL_REQUIRES_OK(RuntimeV2ModelLoadCommon(modelData, modelId, weightPtr, weightSize, std::move(file_constant_mems),
                                             rtSessionExternal));
    ACL_LOG_INFO("successfully execute RuntimeV2ModelLoadFromMemWithMem, modelSize[%zu], modelId[%u], weightSize[%zu]",
                 modelSize, *modelId, weightSize);
    return ACL_SUCCESS;
}

static aclError ModelLoadFromFileWithQ(const char_t *const modelPath, uint32_t *const modelId,
    const ge::ModelQueueArg &args, const int32_t priority)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelPath);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    ACL_LOG_INFO("start to execute ModelLoadFromFileWithQ, modelPath[%s], inputQNum[%zu], "
        "outputQNum[%zu], priority[%d]", modelPath, args.input_queue_ids.size(), args.output_queue_ids.size(), priority);

    ge::GeExecutor geExecutor;
    uint32_t id = 0U;
    ge::ModelData modelData;
    const std::string path(modelPath);
    ACL_LOG_INFO("call ge interface executor.LoadDataFromFile");
    ge::Status ret = geExecutor.LoadDataFromFile(path, modelData);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][FromFile]load model from file[%s], ge result[%u], failed", modelPath, ret);
        ACL_DELETE_ARRAY(modelData.model_data);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    modelData.priority = priority;

    ACL_LOG_INFO("call ge interface executor.LoadModelWithQ");
    ret = geExecutor.LoadModelWithQ(id, modelData, args);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][WithQ]execute LoadModelWithQ failed, ge result[%u]", ret);
        ACL_DELETE_ARRAY(modelData.model_data);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    *modelId = id;
    ACL_DELETE_ARRAY(modelData.model_data);
    ACL_LOG_INFO("successfully execute ModelLoadFromFileWithQ, modelPath[%s], inputQNum[%zu], outputQNum[%zu], "
        "modelId[%u]", modelPath, args.input_queue_ids.size(), args.output_queue_ids.size(), *modelId);
    return ACL_SUCCESS;
}

static aclError ModelLoadFromMemWithQ(const void *const model, const size_t modelSize, uint32_t *const modelId,
    const ge::ModelQueueArg &args, const int32_t priority)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    ACL_LOG_INFO("start to execute ModelLoadFromMemWithQ, modelSize[%zu], inputQNum[%zu], outputQNum[%zu], "
        "priority[%d]", modelSize, args.input_queue_ids.size(), args.output_queue_ids.size(), priority);

    if (modelSize == 0U) {
        ACL_LOG_INNER_ERROR("[Check][Params]modelSize[%zu] is invalid, "
            "can't be zero", modelSize);
        return ACL_ERROR_INVALID_PARAM;
    }

    ge::GeExecutor executor;
    uint32_t id = 0U;
    ge::ModelData data;

    data.model_data = const_cast<void *>(model);
    data.model_len = static_cast<uint64_t>(modelSize);
    data.priority = priority;

    ACL_LOG_INFO("call ge interface executor.LoadModelWithQ, modelSize[%zu]", modelSize);
    const ge::Status ret = executor.LoadModelWithQ(id, data, args);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][WithQ]load model with Q, ge result[%u], failed", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    *modelId = id;
    ACL_LOG_INFO("successfully execute ModelLoadFromMemWithQ, modelSize[%zu], inputQNum[%zu], outputQNum[%zu], "
        "modelId[%u]", modelSize, args.input_queue_ids.size(), args.output_queue_ids.size(), *modelId);

    return ACL_SUCCESS;
}

static void SetInputData(const std::vector<AclModelTensor> &blobs,
    std::vector<ge::GeTensorDesc> &inputGeDesc, bool &isDynamic)
{
    for (size_t i = 0U; i < blobs.size(); ++i) {
        if (blobs[i].tensorDesc != nullptr) {
            isDynamic = true;
            break;
        }
    }
    if (isDynamic) {
        for (size_t i = 0U; i < blobs.size(); ++i) {
            if (blobs[i].tensorDesc != nullptr) {
                std::vector<int64_t> dims;
                ConvertSvecToVec(blobs[i].tensorDesc->dims, dims);
                const ge::GeShape shape(dims);
                ge::GeTensorDesc geTensorDescTmp(shape);
                geTensorDescTmp.SetOriginShape(shape);
                inputGeDesc.push_back(geTensorDescTmp);
            } else {
                const ge::GeTensorDesc geTensorDescTmp;
                inputGeDesc.push_back(geTensorDescTmp);
            }
        }
    }
    return;
}

static void WrapGeShape(gert::Shape &geShape,
    const ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> &dims)
{
    geShape.SetDimNum(dims.size());
    for (size_t i = 0U; i < dims.size(); ++i) {
        (void)geShape.SetDim(i, dims[i]);
    }
}

static void SetInputData(const std::vector<AclModelTensor> &blobs,
    std::vector<gert::Tensor> &inputTensor)
{
    for (size_t i = 0U; i < inputTensor.size(); ++i) {
        if (blobs[i].tensorDesc != nullptr) {
            WrapGeShape(inputTensor[i].MutableOriginShape(), blobs[i].tensorDesc->dims);
            WrapGeShape(inputTensor[i].MutableStorageShape(), blobs[i].tensorDesc->dims);
        }
    }
    return;
}

// runtime2.0 execute
static aclError RuntimeV2ModelExecute(const uint32_t modelId, const aclmdlDataset *const input,
    aclmdlDataset *const output, const bool isAsync, const aclrtStream stream)
{
    ACL_LOG_INFO("start to execute ModelExecute, modelId[%u]", modelId);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(input);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(output);

    auto const executor = acl::AclResourceManager::GetInstance().GetExecutor(modelId);
    if (executor == nullptr) {
        ACL_LOG_ERROR("input modelId[%u] is invalid, please make sure model has been loaed", modelId);
        return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
    }

    const gert::ModelDesc &desc = executor->GetModelDesc();
    const size_t inputNum = desc.GetInputNum();
    ACL_LOG_INFO("get model input num %zu, input num %zu", inputNum, input->blobs.size());
    if (input->blobs.size() < inputNum) {
        ACL_LOG_ERROR("intput blobs %zu can not be smaller than intput desc size %zu",
            input->blobs.size(), inputNum);
        return static_cast<aclError>(ACL_ERROR_INVALID_PARAM);
    }

    std::vector<gert::Tensor> inputTensor(inputNum);
    std::vector<gert::Tensor> outputTensor(output->blobs.size());
    std::vector<gert::Tensor *> inputVec(inputNum);
    std::vector<gert::Tensor *> outputVec(output->blobs.size());
    for (size_t i = 0UL; i < inputNum; ++i) {
        const auto dataBuffer = input->blobs[i].dataBuf;
        if (dataBuffer == nullptr) {
            ACL_LOG_ERROR("[Check][dataBuffer]input dataset blobs is null, "
                "modelId[%d], index[%zu]", modelId, i);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
                std::vector<const char *>({"param"}),
                std::vector<const char *>({"dataBuffer"}));
            return ACL_ERROR_INVALID_PARAM;
        }
        inputTensor[i].MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
        (void)inputTensor[i].MutableTensorData().SetAddr(dataBuffer->data, nullptr);
        inputTensor[i].MutableTensorData().SetSize(dataBuffer->length);
        inputTensor[i].MutableOriginShape() = desc.GetInputDesc(i)->GetOriginShape();
        inputTensor[i].MutableStorageShape() = desc.GetInputDesc(i)->GetStorageShape();
        inputVec[i] = &(inputTensor[i]);
    }

    SetInputData(input->blobs, inputTensor);
    ge::InputAippType type = ge::DATA_WITHOUT_AIPP;
    size_t aippIndex = 0U;
    for (size_t i = 0UL; i < inputNum; ++i) {
        (void)executor->GetAippType(static_cast<uint32_t>(i), type, aippIndex);
        if (type == ge::DATA_WITH_DYNAMIC_AIPP) {
            // dynamic aipp input no need to check range for compatibility
            continue;
        }
        if (!desc.GetInputDesc(i)->IsOriginShapeInRange(inputTensor[i].GetOriginShape())) {
            ACL_LOG_ERROR("[Check][InputShape] Input [%zu] shape out of shape range.", i);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    for (size_t i = 0UL; i < output->blobs.size(); ++i) {
        const auto dataBuffer = output->blobs[i].dataBuf;
        if (dataBuffer == nullptr) {
            ACL_LOG_ERROR("[Check][Databuffer]output dataset blobs is null, modelId[%d], index[%zu]",
                modelId, i);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
                std::vector<const char *>({"param"}),
                std::vector<const char *>({"dataBuffer"}));
            return ACL_ERROR_INVALID_PARAM;
        }
        if ((dataBuffer->data == nullptr) && (!isAsync)) {
            const size_t memSize = static_cast<size_t>(desc.GetOutputDesc(i)->GetSize());
            if (memSize > 0UL) {
                ACL_REQUIRES_CALL_RTS_OK(aclrtMalloc(reinterpret_cast<void **>(&dataBuffer->data), memSize,
                    ACL_MEM_TYPE_HIGH_BAND_WIDTH), aclrtMalloc);
                dataBuffer->length = memSize;
                ACL_LOG_DEBUG("ModelExecute, assign acl-malloced output addr to user-defined buffer, addr:[%p], "
                              "len:[%lu]", dataBuffer->data, dataBuffer->length);
            }
        }

        outputTensor[i].MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
        (void)outputTensor[i].MutableTensorData().SetAddr(dataBuffer->data, nullptr);
        outputTensor[i].MutableTensorData().SetSize(dataBuffer->length);
        outputVec[i] = &(outputTensor[i]);
    }

    ge::Status ret = ge::GRAPH_SUCCESS;
    const auto index_id = executor->GetIterationNum();
    MsprofEvent event{};
    CANN_PROFILING_EVENT_START(modelId, index_id, gert::GeProfInfoType::kModelExecute, event);
    CANN_PROFILING_STEP_TRACE(modelId, index_id, kStartTag, stream);
    if (isAsync) {
        gert::ModelExecuteArg arg;
        arg.stream = stream;
        arg.external_allocator = acl::AclResourceManager::GetInstance().GetAllocators(arg.stream, false).get();
        ret = executor->Execute(arg, inputVec.data(), inputTensor.size(), outputVec.data(), outputTensor.size());
    } else {
        ret = executor->ExecuteSync(inputVec.data(), inputTensor.size(), outputVec.data(), outputTensor.size());
    }
    CANN_PROFILING_STEP_TRACE(modelId, index_id, kEndTag, stream);
    CANN_PROFILING_EVENT_END(modelId, index_id, gert::GeProfInfoType::kModelExecute, event);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Exec][Model]Execute model failed, ge result[%u], modelId[%u]", ret, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    for (size_t i = 0UL; i < output->blobs.size(); ++i) {
        if (output->blobs[i].tensorDesc != nullptr) {
            aclDestroyTensorDescImpl(output->blobs[i].tensorDesc);
        }
        output->blobs[i].tensorDesc = aclCreateTensorDescImpl(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    }

    for (size_t i = 0UL; i < outputTensor.size(); ++i) {
        if (output->blobs[i].tensorDesc != nullptr) {
            output->blobs[i].tensorDesc->dims.clear();
            const ge::Format format = outputTensor[i].GetFormat().GetStorageFormat();
            const ge::DataType dataType = outputTensor[i].GetDataType();
            for (size_t idx = 0U; idx < outputTensor[i].MutableStorageShape().GetDimNum(); ++idx) {
                output->blobs[i].tensorDesc->dims.push_back(outputTensor[i].MutableStorageShape().GetDim(idx));
            }
            if (format != ge::FORMAT_RESERVED) {
                output->blobs[i].tensorDesc->format = static_cast<aclFormat>(format);
            }
            if (dataType != ge::DT_UNDEFINED) {
                output->blobs[i].tensorDesc->dataType = static_cast<aclDataType>(dataType);
            }
        }

        auto &dataBuffer = output->blobs[i].dataBuf;
        if ((dataBuffer->data == nullptr) && (!isAsync)) {
            ACL_REQUIRES_NOT_NULL(outputTensor[i].MutableTensorData().GetAddr());
            const auto outputSize = outputTensor[i].MutableTensorData().GetSize();
            ACL_REQUIRES_CALL_RTS_OK(aclrtMalloc(reinterpret_cast<void **>(&dataBuffer->data), outputSize,
                ACL_MEM_TYPE_HIGH_BAND_WIDTH), aclrtMalloc);
            ACL_REQUIRES_CALL_RTS_OK(aclrtMemcpy(dataBuffer->data, outputSize,
                outputTensor[i].MutableTensorData().GetAddr(), outputSize, ACL_MEMCPY_DEVICE_TO_DEVICE),
                aclrtMemcpy);
            dataBuffer->length = outputSize;
            ACL_LOG_DEBUG("ModelExecute, assign acl-malloced output addr to user-defined buffer, addr:[%p], len:[%lu]",
                          dataBuffer->data, dataBuffer->length);
        }
    }

    ACL_LOG_INFO("successfully execute ModelExecute, modelId[%u]", modelId);
    return ACL_SUCCESS;
}

static aclError Om2GetModelTensorDesc(std::vector<ge::TensorDesc> &inputDesc, std::vector<ge::TensorDesc> &outputDesc,
    size_t &inputNum, size_t &outputNum, const std::shared_ptr<gert::Om2ModelExecutor> &executor,
    const aclmdlDataset *const input, aclmdlDataset *const output, const uint32_t modelId)
{
    ge::Status getDescRet = executor->GetModelDescInfo(inputDesc, outputDesc);
    if (getDescRet != ge::SUCCESS) {
        ACL_LOG_ERROR("[Get][ModelDesc]Get model desc info failed, ge result[%u], modelId[%u]", getDescRet, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(getDescRet));
    }

    inputNum = inputDesc.size();
    outputNum = outputDesc.size();
    ACL_LOG_INFO("Om2GetModelTensorDesc get model input num %zu, output num %zu, input blobs num %zu, output blobs num %zu",
                 inputNum, outputNum, input->blobs.size(), output->blobs.size());
    if (input->blobs.size() != inputNum) {
        ACL_LOG_ERROR("Input blobs size mismatch. actual_size[%zu], expected_size[%zu]", input->blobs.size(), inputNum);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (output->blobs.size() != outputNum) {
        ACL_LOG_ERROR("Output blobs size mismatch. actual_size[%zu], expected_size[%zu]", output->blobs.size(),
                      outputNum);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

static aclError PrepareTensor(std::vector<gert::Tensor> &tensor, std::vector<gert::Tensor *> &vec,
    const size_t inputNum, const aclmdlDataset *const dataset, const std::vector<ge::TensorDesc> &tensorDesc,
    const uint32_t modelId)
{
    for (size_t i = 0UL; i < inputNum; ++i) {
        const auto dataBuffer = dataset->blobs[i].dataBuf;
        if (dataBuffer == nullptr) {
            ACL_LOG_ERROR("[Check][dataBuffer]input dataset blobs is null, modelId[%d], index[%zu]", modelId, i);
            return ACL_ERROR_INVALID_PARAM;
        }
        tensor[i].MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
        (void)tensor[i].MutableTensorData().SetAddr(dataBuffer->data, nullptr);
        tensor[i].MutableTensorData().SetSize(dataBuffer->length);
        tensor[i].MutableOriginShape() = ConstructRtShapeFromShape(tensorDesc[i].GetOriginShape());
        tensor[i].MutableStorageShape() = ConstructRtShapeFromShape(tensorDesc[i].GetShape());
        tensor[i].SetStorageFormat(tensorDesc[i].GetFormat());
        tensor[i].SetOriginFormat(tensorDesc[i].GetOriginFormat());
        tensor[i].SetDataType(tensorDesc[i].GetDataType());
        vec[i] = &(tensor[i]);
    }
    return ACL_SUCCESS;
}

static aclError Om2UpdateOutputTensorDesc(aclmdlDataset *const &output, std::vector<gert::Tensor> &outputTensor, const bool isAsync)
{
    for (size_t i = 0UL; i < output->blobs.size(); ++i) {
        if (output->blobs[i].tensorDesc != nullptr) {
            aclDestroyTensorDescImpl(output->blobs[i].tensorDesc);
        }
        output->blobs[i].tensorDesc = aclCreateTensorDescImpl(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    }

    for (size_t i = 0UL; i < outputTensor.size(); ++i) {
        if (output->blobs[i].tensorDesc != nullptr) {
            output->blobs[i].tensorDesc->dims.clear();
            const ge::Format format = outputTensor[i].GetFormat().GetStorageFormat();
            const ge::DataType dataType = outputTensor[i].GetDataType();
            for (size_t idx = 0U; idx < outputTensor[i].MutableStorageShape().GetDimNum(); ++idx) {
                output->blobs[i].tensorDesc->dims.push_back(outputTensor[i].MutableStorageShape().GetDim(idx));
            }
            if (format != ge::FORMAT_RESERVED) {
                output->blobs[i].tensorDesc->format = static_cast<aclFormat>(format);
            }
            if (dataType != ge::DT_UNDEFINED) {
                output->blobs[i].tensorDesc->dataType = static_cast<aclDataType>(dataType);
            }
        }

        auto &dataBuffer = output->blobs[i].dataBuf;
        if ((dataBuffer->data == nullptr) && (!isAsync)) {
            ACL_REQUIRES_NOT_NULL(outputTensor[i].MutableTensorData().GetAddr());
            const auto outputSize = outputTensor[i].MutableTensorData().GetSize();
            ACL_REQUIRES_CALL_RTS_OK(
                aclrtMalloc(reinterpret_cast<void **>(&dataBuffer->data), outputSize, ACL_MEM_TYPE_HIGH_BAND_WIDTH),
                aclrtMalloc);
            ACL_REQUIRES_CALL_RTS_OK(
                aclrtMemcpy(dataBuffer->data, outputSize, outputTensor[i].MutableTensorData().GetAddr(), outputSize,
                            ACL_MEMCPY_DEVICE_TO_DEVICE),
                aclrtMemcpy);
            dataBuffer->length = outputSize;
            ACL_LOG_DEBUG("ModelExecute, assign acl-malloced output addr to user-defined buffer, addr:[%p], len:[%lu]",
                          dataBuffer->data, dataBuffer->length);
        }
    }
    return ACL_SUCCESS;
}

static aclError Om2ModelExecute(const uint32_t modelId, const aclmdlDataset *const input, aclmdlDataset *const output,
                                const bool isAsync, const aclrtStream stream) {
    ACL_LOG_INFO("start to execute ModelExecute, modelId[%u]", modelId);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(input);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(output);

    auto const executor = acl::AclResourceManager::GetInstance().GetOm2Executor(modelId);
    if (executor == nullptr) {
        ACL_LOG_ERROR("input modelId[%u] is invalid, please make sure model has been loaded", modelId);
        return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
    }

    // Get model description info
    std::vector<ge::TensorDesc> inputDesc;
    std::vector<ge::TensorDesc> outputDesc;
    size_t inputNum = 0;
    size_t outputNum = 0;
    auto ret = Om2GetModelTensorDesc(inputDesc, outputDesc, inputNum, outputNum, executor, input, output, modelId);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("Get model TensorDesc failed.");
        return ret;
    }

    // Prepare input tensors
    std::vector<gert::Tensor> inputTensor(inputNum);
    std::vector<gert::Tensor *> inputVec(inputNum);
    ret = PrepareTensor(inputTensor, inputVec, inputNum, input, inputDesc, modelId);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("Prepare input tensors failed.");
        return ret;
    }
    
    // Prepare output tensors
    std::vector<gert::Tensor> outputTensor(outputNum);
    std::vector<gert::Tensor *> outputVec(outputNum);
    ret = PrepareTensor(outputTensor, outputVec, outputNum, output, outputDesc, modelId);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("Prepare input tensors failed.");
        return ret;
    }

    // Call executor run with tensor pointers
    ge::Status run_ret = ge::GRAPH_SUCCESS;
    if (isAsync) {
        run_ret = executor->RunAsync(stream, inputVec, outputVec);
    } else {
        run_ret = executor->Run(inputVec, outputVec);
    }
    if (run_ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Exec][Model]Execute model failed, ge result[%u], modelId[%u]", run_ret, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(run_ret));
    }

    // Update output tensor descriptions
    ret = Om2UpdateOutputTensorDesc(output, outputTensor, isAsync);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("Update output tensors descriptions failed.");
        return ret;
    }

    ACL_LOG_INFO("successfully execute Om2ModelExecute, modelId[%u]", modelId);
    return ACL_SUCCESS;
}

static aclError ModelExecute(const uint32_t modelId, const aclmdlDataset *const input,
    aclmdlDataset *const output, const bool basync, const aclrtStream stream)
{
    ACL_LOG_INFO("start to execute ModelExecute, modelId[%u]", modelId);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(input);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(output);

    ge::RunModelData inputData;
    inputData.timeout = 0U;
    inputData.timestamp = 0U;
    inputData.index = 0U;
    inputData.modelId = modelId;

    inputData.dynamic_batch_size = input->dynamicBatchSize;
    inputData.dynamic_image_height = input->dynamicResolutionHeight;
    inputData.dynamic_image_width = input->dynamicResolutionWidth;
    inputData.dynamic_dims = input->dynamicDims;
    ACL_LOG_DEBUG("ModelExecute dynamic param: batch_size[%lu], height[%lu], width[%lu], dim_num[%zu]",
                  input->dynamicBatchSize, input->dynamicResolutionHeight, input->dynamicResolutionWidth,
                  input->dynamicDims.size());
    inputData.blobs.resize(input->blobs.size());
    for (size_t i = 0UL; i < input->blobs.size(); ++i) {
        ge::DataBuffer &inputBuffer = inputData.blobs[i];
        const auto dataBuffer = input->blobs[i].dataBuf;
        if (dataBuffer == nullptr) {
            ACL_LOG_ERROR("[Check][dataBuffer]input dataset blobs is null, "
                "modelId[%d], index[%zu]", modelId, i);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
                std::vector<const char *>({"param"}),
                std::vector<const char *>({"dataBuffer"}));
            return ACL_ERROR_INVALID_PARAM;
        }
        inputBuffer.data = dataBuffer->data;
        inputBuffer.length = dataBuffer->length;
        inputBuffer.isDataSupportMemShare = false;
        inputBuffer.placement = ge::Placement::kPlacementDevice;
    }

    std::vector<ge::GeTensorDesc> inputGeDesc;
    bool dynamicFlag = false;
    SetInputData(input->blobs, inputGeDesc, dynamicFlag);

    ge::RunModelData outputData;
    outputData.modelId = modelId;

    std::vector<size_t> needMallocIndexes;
    outputData.blobs.resize(output->blobs.size());
    for (size_t i = 0UL; i < output->blobs.size(); ++i) {
        ge::DataBuffer &outputBuffer = outputData.blobs[i];
        const auto dataBuffer = output->blobs[i].dataBuf;
        if (dataBuffer == nullptr) {
            ACL_LOG_ERROR("[Check][Databuffer]output dataset blobs is null, modelId[%d], index[%zu]",
                modelId, i);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
                std::vector<const char *>({"param"}),
                std::vector<const char *>({"dataBuffer"}));
            return ACL_ERROR_INVALID_PARAM;
        }
        if (dataBuffer->data == nullptr) {
            needMallocIndexes.push_back(i);
        }
        outputBuffer.data = dataBuffer->data;
        outputBuffer.length = dataBuffer->length;
        outputBuffer.isDataSupportMemShare = false;
        outputBuffer.placement = ge::Placement::kPlacementDevice;
    }

    ge::GeExecutor executor;
    if ((!needMallocIndexes.empty()) && (!basync)) {
        std::vector<ge::TensorDesc> inputDesc;
        std::vector<ge::TensorDesc> outputDesc;
        (void)executor.GetModelDescInfo(modelId, inputDesc, outputDesc);
        if (outputDesc.size() < output->blobs.size()) {
            ACL_LOG_ERROR("[Check][OutputDesc] outputDesc size [%zu] mismatch with blobs size[%zu].",
                          outputDesc.size(), output->blobs.size());
            return ACL_ERROR_INVALID_PARAM;
        }

        for (auto &idx : needMallocIndexes) {
            auto &dataBuffer = output->blobs[idx].dataBuf;
            const size_t outputSize = static_cast<size_t>(outputDesc[idx].GetSize());
            if (outputSize > 0UL) {
                ACL_REQUIRES_CALL_RTS_OK(aclrtMalloc(reinterpret_cast<void **>(&dataBuffer->data), outputSize,
                    ACL_MEM_TYPE_HIGH_BAND_WIDTH), aclrtMalloc);
                dataBuffer->length = outputSize;
                outputData.blobs[idx].data = dataBuffer->data;
                outputData.blobs[idx].length = outputSize;
                ACL_LOG_DEBUG("ModelExecute, assign acl-malloced output addr to user-defined buffer, addr:[%p], "
                              "len:[%lu]", dataBuffer->data, dataBuffer->length);
            }
        }
    }

    ACL_LOG_INFO("call ge interface executor.ExecModel, modelId[%u], asyncMode[%d]",
        modelId, static_cast<int32_t>(basync));
    std::vector<ge::GeTensorDesc> outputGeDesc;
    const ge::Status ret = executor.ExecModel(modelId, stream, inputData, inputGeDesc,
        outputData, outputGeDesc, basync);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Exec][Model]Execute model failed, ge result[%u], modelId[%u]", ret, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    if (dynamicFlag) {
        for (size_t i = 0U; i < output->blobs.size(); ++i) {
            if (output->blobs[i].tensorDesc != nullptr) {
                aclDestroyTensorDescImpl(output->blobs[i].tensorDesc);
            }
            output->blobs[i].tensorDesc = aclCreateTensorDescImpl(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
        }
    }
    ACL_REQUIRES_LE(outputGeDesc.size(), output->blobs.size());
    for (size_t i = 0UL; i < outputGeDesc.size(); ++i) {
        if (output->blobs[i].tensorDesc != nullptr) {
            output->blobs[i].tensorDesc->dims.clear();
            const std::vector<int64_t> dims = outputGeDesc[i].GetShape().GetDims();
            const ge::Format format = outputGeDesc[i].GetFormat();
            const ge::DataType dataType = outputGeDesc[i].GetDataType();
            for (const auto &dim : dims) {
                output->blobs[i].tensorDesc->dims.push_back(dim);
            }
            if (format != ge::FORMAT_RESERVED) {
                output->blobs[i].tensorDesc->format = static_cast<aclFormat>(format);
            }
            if (dataType != ge::DT_UNDEFINED) {
                output->blobs[i].tensorDesc->dataType = static_cast<aclDataType>(dataType);
            }
        }

        auto &dataBuffer = output->blobs[i].dataBuf;
        if ((dataBuffer->data == nullptr) && (!basync)) {
            const auto outputSize = outputData.blobs[i].length;
            ACL_REQUIRES_CALL_RTS_OK(aclrtMalloc(reinterpret_cast<void **>(&dataBuffer->data), outputSize,
                ACL_MEM_TYPE_HIGH_BAND_WIDTH), aclrtMalloc);
            ACL_REQUIRES_CALL_RTS_OK(aclrtMemcpy(dataBuffer->data, outputSize, outputData.blobs[i].data,
                outputSize, ACL_MEMCPY_DEVICE_TO_DEVICE), aclrtMemcpy);
            dataBuffer->length = outputSize;
            ACL_LOG_DEBUG("ModelExecute, assign acl-malloced output addr to user-defined buffer, addr:[%p], "
                          "len:[%lu]", dataBuffer->data, dataBuffer->length);
        }
    }

    ACL_LOG_INFO("successfully execute ModelExecute, modelId[%u]", modelId);
    return ACL_SUCCESS;
}

// get real tensor name from modelDesc, it will return nullptr if tensorName isn't in modelDesc
static const char_t *GetRealTensorName(const aclmdlDesc *const modelDesc, const std::string &tensorName)
{
    for (size_t idx = 0U; idx < modelDesc->inputDesc.size(); ++idx) {
        if (modelDesc->inputDesc[idx].name == tensorName) {
            return modelDesc->inputDesc[idx].name.c_str();
        }
    }

    for (size_t idx = 0U; idx < modelDesc->outputDesc.size(); ++idx) {
        if (modelDesc->outputDesc[idx].name == tensorName) {
            return modelDesc->outputDesc[idx].name.c_str();
        }
    }
    return nullptr;
}

static bool IsConvertTensorNameLegal(const aclmdlDesc *const modelDesc, const std::string &tensorName)
{
    return (GetRealTensorName(modelDesc, tensorName) == nullptr);
}

// current conversion tensor name illegal needs to be transformed
static bool TransConvertTensorNameToLegal(const aclmdlDesc *const modelDesc, std::string &tensorName)
{
    size_t depth = 0U;
    tensorName = tensorName + "_";
    std::queue<std::string> q;
    q.push(tensorName);
    constexpr size_t maxDepth = 3U;
    while (!q.empty()) {
        if (depth == maxDepth) {
            ACL_LOG_INFO("reach max depth[%zu], cannot generate legal convert tensor name", maxDepth);
            tensorName = tensorName.substr(0U, tensorName.size() - 1U);
            return false;
        }
        const size_t len = q.size();
        for (size_t idx = 0U; idx < len; ++idx) {
            std::string curTensorName = q.front();
            q.pop();
            for (char_t c = 'a'; c <= 'z'; ++c) {
                curTensorName += c;
                if (IsConvertTensorNameLegal(modelDesc, curTensorName)) {
                    tensorName = curTensorName;
                    return true;
                }
                q.push(curTensorName);
                curTensorName = curTensorName.substr(0U, curTensorName.size() - 1U);
            }
        }
        depth++;
    }
    return false;
}

// convert params to convertName
static void GetConvertTensorName(const aclmdlDesc *const modelDesc, const size_t idx,
    const TensorType tensorType, std::string &convertName)
{
    convertName = std::string(TENSOR_NAME_PREFIX) + "_" +
        std::string(MODEL_ID_STR) + "_" + std::to_string(modelDesc->modelId);
    if (tensorType == TensorType::INPUT_TENSOR_TYPE) {
        convertName += ("_" + std::string(TENSOR_INPUT_STR));
    } else {
        convertName += ("_" + std::string(TENSOR_OUTPUT_STR));
    }
    convertName += ("_" + std::to_string(idx));
    ACL_LOG_INFO("convert realname of tensor success, conversion name = %s", convertName.c_str());
}

// get tensor name to dims with or without realname
static aclError GetTensorDescNameToDims(const aclmdlDesc *const modelDesc, const std::string &realName,
    const TensorType tensorType, const size_t idx, aclmdlIODims *const dims)
{
    const size_t dimsNameLen = sizeof(dims->name);
    std::string tensorName;
    if ((realName.size() + 1U) > dimsNameLen) {
        // use conversion name because realname is too long
        ACL_LOG_INFO("use conversion name because real tensor name is over than %zu", dimsNameLen);
        GetConvertTensorName(modelDesc, idx, tensorType, tensorName);
        if (!IsConvertTensorNameLegal(modelDesc, tensorName)) {
            if (!TransConvertTensorNameToLegal(modelDesc, tensorName)) {
                ACL_LOG_WARN("cannot generate legal tensor name, use conversion name %s may has conflict risk",
                    tensorName.c_str());
            }
        }
    } else {
        tensorName = realName;
    }

    const auto ret = strncpy_s(dims->name, dimsNameLen, tensorName.c_str(), tensorName.size());
    if (ret != EOK) {
        ACL_LOG_INNER_ERROR("[Copy][Str]call strncpy_s failed, result = %d", ret);
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

static aclError GetDims(const aclmdlDesc *const modelDesc, const TensorType tensorType, const DimsType dimsType,
    const size_t idx, aclmdlIODims *const dims)
{
    ACL_REQUIRES_NOT_NULL(dims);
    std::vector<aclmdlTensorDesc> desc;
    if (tensorType == TensorType::INPUT_TENSOR_TYPE) {
        desc = modelDesc->inputDesc;
    } else {
        desc = modelDesc->outputDesc;
    }

    const size_t descSize = desc.size();
    if (idx >= descSize) {
        ACL_LOG_INNER_ERROR("[Check][Params]GetDims failed, index[%zu] can not greater than or equal to tensor "
            "size[%zu]", idx, descSize);
        return ACL_ERROR_INVALID_PARAM;
    }

    const aclmdlTensorDesc &tensorDesc = desc[idx];
    const auto ret = GetTensorDescNameToDims(modelDesc, tensorDesc.name, tensorType, idx, dims);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][TensorDescName]get tensor desc name to dims failed, errorCode = %d", ret);
        return ret;
    }
    std::vector<int64_t> tensorDims;
    if (dimsType == DimsType::DIMS_TYPE_V1) {
        tensorDims = tensorDesc.dims;
    } else if (dimsType == DimsType::DIMS_TYPE_V2) {
        tensorDims = tensorDesc.dimsV2;
    } else {
        ACL_LOG_INNER_ERROR("[Check][dimsType]dims type[%d] is invalid", static_cast<int32_t>(dimsType));
        return ACL_ERROR_FAILURE;
    }

    const size_t dimSize = tensorDims.size();
    if (dimSize > static_cast<size_t>(ACL_MAX_DIM_CNT)) {
        ACL_LOG_INNER_ERROR("[Check][dimSize]get dims failed, dims count[%zu] can not larger than max[%d]",
            dims->dimCount, ACL_MAX_DIM_CNT);
        return ACL_ERROR_STORAGE_OVER_LIMIT;
    }
    dims->dimCount = dimSize;

    for (size_t i = 0U; i < dimSize; ++i) {
        dims->dims[i] = tensorDims[i];
    }

    return ACL_SUCCESS;
}

static aclError GetDimsRange(const aclmdlDesc *const modelDesc, const TensorType tensorType, const size_t idx,
                             aclmdlIODimsRange *const dimsRange)
{
    ACL_REQUIRES_NOT_NULL(dimsRange);
    const bool isDynamicGear = (!modelDesc->dynamicBatch.empty() || !modelDesc->dynamicDims.empty() ||
                          !modelDesc->dynamicHW.empty());
    if (isDynamicGear) {
        dimsRange->rangeCount = 0U;
        ACL_LOG_INFO("set rangeCount[%zu] in the dynamic gear model scenario success.", dimsRange->rangeCount);
        return ACL_SUCCESS;
    }

    std::vector<aclmdlTensorDesc> desc;
    if (tensorType == TensorType::INPUT_TENSOR_TYPE) {
        desc = modelDesc->inputDesc;
    } else {
        desc = modelDesc->outputDesc;
    }

    const size_t descSize = desc.size();
    if (idx >= descSize) {
        ACL_LOG_INNER_ERROR("[Check][Params]GetDimsRange failed, index[%zu] can not greater than or equal to tensor "
            "size[%zu]", idx, descSize);
        return ACL_ERROR_INVALID_PARAM;
    }

    const auto &tensorDesc = desc[idx];
    const auto &shapeRanges = tensorDesc.shapeRanges;
    const size_t dynamicDimSize = shapeRanges.size();
    constexpr size_t MIN = 0U;
    constexpr size_t MAX = 1U;
    if (dynamicDimSize > static_cast<size_t>(ACL_MAX_DIM_CNT)) {
        ACL_LOG_INNER_ERROR("[Check][dynamicDimSize]GetDimsRange failed, dim size[%zu] can not larger than max[%d]",
                            dynamicDimSize, ACL_MAX_DIM_CNT);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (dynamicDimSize == 0U) {
        const auto &staticDims = tensorDesc.dims;
        const size_t staticDimSize = staticDims.size();
        if (staticDimSize > static_cast<size_t>(ACL_MAX_DIM_CNT)) {
            ACL_LOG_INNER_ERROR("[Check][staticDimSize]GetDimsRange failed, dim size[%zu] can not larger than max[%d]",
                                staticDimSize, ACL_MAX_DIM_CNT);
            return ACL_ERROR_INTERNAL_ERROR;
        }
        dimsRange->rangeCount = staticDimSize;
        ACL_LOG_INFO("set rangeCount[%zu] in the static model scenario success.", dimsRange->rangeCount);
        for (size_t i = 0U; i < staticDimSize; ++i) {
            dimsRange->range[i][MIN] = staticDims[i];
            dimsRange->range[i][MAX] = staticDims[i];
            ACL_LOG_INFO("set range[%zu][%zu] to [%ld] and range[%zu][%zu] to [%ld] success.", i, MIN,
                         dimsRange->range[i][MIN], i, MAX, dimsRange->range[i][MAX]);
        }
    } else {
        if (dynamicDimSize != tensorDesc.dims.size()) {
            ACL_LOG_INNER_ERROR("[Check][dynamicDimSize]GetDimsRange failed, size of shapeRanges[%zu] does not equal "
                                "to size of dims[%zu]", dynamicDimSize, tensorDesc.dims.size());
            return ACL_ERROR_INTERNAL_ERROR;
        }
        dimsRange->rangeCount = dynamicDimSize;
        ACL_LOG_INFO("set rangeCount[%zu] in the dynamic model scenario success.", dimsRange->rangeCount);
        for (size_t i = 0U; i < dynamicDimSize; ++i) {
            dimsRange->range[i][MIN] = shapeRanges[i].first;
            dimsRange->range[i][MAX] = shapeRanges[i].second;
            ACL_LOG_INFO("set range[%zu][%zu] to [%ld] and range[%zu][%zu] to [%ld] success.", i, MIN,
                         dimsRange->range[i][MIN], i, MAX, dimsRange->range[i][MAX]);
        }
    }

    return ACL_SUCCESS;
}

static aclError GetCurGearIndex(const aclmdlDesc *const modelDesc, const std::vector<uint64_t> &shapeInfo,
                                const int32_t dynamicType, size_t &curGearIndex)
{
    if (dynamicType == static_cast<int32_t>(ge::DYNAMIC_DIMS)) { // dynamic dims, type is 3
        ACL_LOG_DEBUG("Get dynamic dims gear index, dynamicType[%d], modelId[%u]",
                      dynamicType, modelDesc->modelId);
        for (size_t i = 0U; i < modelDesc->dynamicDims.size(); ++i) {
            // shapeInfo is current dims
            if (shapeInfo == modelDesc->dynamicDims[i]) {
                curGearIndex = i;
                return ACL_SUCCESS;
            }
        }
    } else {
        const size_t shapeSize = shapeInfo.size();
        if (shapeSize == DYNAMIC_BATCH_SIZE) { // dynamic batch, type is 1
            ACL_LOG_DEBUG("Get dynamic batch gear index, dynamicType[%d], modelId[%u]", dynamicType,
                          modelDesc->modelId);
            for (size_t i = 0U; i < modelDesc->dynamicBatch.size(); ++i) {
                // shapeInfo[0] is current batch size
                if (shapeInfo[0U] == modelDesc->dynamicBatch[i]) {
                    curGearIndex = i;
                    return ACL_SUCCESS;
                }
            }
        } else if (shapeSize == DYNAMIC_HW_SIZE) { // dynamic hw, type is 2
            ACL_LOG_DEBUG("Get dynamic hw gear index, dynamicType[%d], modelId[%u]", dynamicType, modelDesc->modelId);
            for (size_t i = 0U; i < modelDesc->dynamicHW.size(); ++i) {
                // shapeInfo is current hw
                if (shapeInfo == modelDesc->dynamicHW[i]) {
                    curGearIndex = i;
                    return ACL_SUCCESS;
                }
            }
        } else {
            ACL_LOG_INNER_ERROR("[Check][dynamicType]dynamicType[%d] is invalid", dynamicType);
        }
    }

    return ACL_ERROR_FAILURE;
}

static aclError GetCurOuputShapeInfo(const aclmdlDesc *const modelDesc, const size_t idex,
                                     const size_t curGearIndex, aclmdlIODims *const dims)
{
    ACL_LOG_DEBUG("curGearIndex is %zu, dynamicOutputShapeInfoSize is %zu , modelId is %u",
        curGearIndex, modelDesc->dynamicOutputShape.size(), modelDesc->modelId);
    for (auto &it : modelDesc->dynamicOutputShape) {
        if ((it.size() < MIN_OUTPUT_SHAPE_INFO_SIZE) || (it.size() > MAX_OUTPUT_SHAPE_INFO_SIZE)) {
            ACL_LOG_INNER_ERROR("[Check][dynamicOutputShape]output shape info size[%zu] is invalid, range is "
                "[%zu, %zu]", it.size(), MIN_OUTPUT_SHAPE_INFO_SIZE, MAX_OUTPUT_SHAPE_INFO_SIZE);
            return ACL_ERROR_FAILURE;
        }
        // -1 represents static output gear index value
        // it[0] is gear index and it[1] is output index
        if (((static_cast<int64_t>(curGearIndex) == it[0U]) || (it[0U] == -1)) &&
            (static_cast<int64_t>(idex) == it[1U])) {
            int32_t idx = 0;
            for (size_t i = 2U; i < it.size(); ++i) { // from the third element is shape info
                dims->dims[idx] = it[i];
                idx++;
            }
            dims->dimCount = it.size() - 2U;
            const aclmdlTensorDesc &tensorDesc = modelDesc->outputDesc[idex];
            const auto ret = GetTensorDescNameToDims(modelDesc, tensorDesc.name, TensorType::OUTPUT_TENSOR_TYPE, idex,
                                                     dims);
            if (ret != ACL_SUCCESS) {
                ACL_LOG_INNER_ERROR("[Get][TensorDescName]get tensor desc name to dims failed, errorCode = %d", ret);
                return ret;
            }
            return ACL_SUCCESS;
        }
    }

    return ACL_ERROR_FAILURE;
}

static void UpdateGraphOptions(const std::string &key, const std::string &value)
{
    auto options = ge::GetThreadLocalContext().GetAllGraphOptions();
    options[key] = value;
    ge::GetThreadLocalContext().SetGraphOption(options);
}

static const char *aclmdlGetNameByIndex(const std::vector<aclmdlTensorDesc> &desc, const size_t idx)
{
    if (idx >= desc.size()) {
        ACL_LOG_ERROR("[Check][index]get name by index failed, index[%zu] is larger than or equal to desc size[%zu]",
            idx, desc.size());
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("cannot larger than or equal to desc size[%zu]",
            desc.size());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"index", std::to_string(idx).c_str(), errMsg.c_str()}));
        return "";
    }

    return desc[idx].name.c_str();
}

static aclFormat aclmdlGetFormat(const std::vector<aclmdlTensorDesc> &desc, const size_t idx)
{
    if (idx >= desc.size()) {
        ACL_LOG_INNER_ERROR("[Check][index]get data format by index failed, index[%zu] is larger "
            "than or equal to desc size[%zu]", idx, desc.size());
        return ACL_FORMAT_UNDEFINED;
    }

    return desc[idx].format;
}

static aclDataType aclmdlGetDataType(const std::vector<aclmdlTensorDesc> &desc, const size_t idx)
{
    if (idx >= desc.size()) {
        ACL_LOG_INNER_ERROR("[Check][Index]get data type by index failed, index[%zu] is larger than or "
            "equal to desc size[%zu]", idx, desc.size());
        return ACL_DT_UNDEFINED;
    }

    return desc[idx].dataType;
}

static aclError aclmdlGetIndexByName(const std::vector<aclmdlTensorDesc> &desc,
    const char_t *const name, size_t *const idx)
{
    ACL_REQUIRES_NOT_NULL(name);
    ACL_REQUIRES_NOT_NULL(idx);

    const std::string tensorName(name);
    for (size_t i = 0U; i < desc.size(); ++i) {
        if (desc[i].name == tensorName) {
            *idx = i;
            ACL_LOG_DEBUG("success to get tensor[%s] index[%zu]", name, *idx);
            return ACL_SUCCESS;
        }
    }

    ACL_LOG_INNER_ERROR("[Get][Index]get index by name failed, cannot find tensor name[%s]", name);
    return ACL_ERROR_INVALID_PARAM;
}

// try to transfer conversion name to real tensor name, it will return nullptr if conversion name isn't
// satisfy condition
static const char_t *TransTensorNameToReal(const aclmdlDesc *const modelDesc, const std::string &tensorName)
{
    std::vector<std::string> valArr;
    acl::StringUtils::Split(tensorName, '_', valArr);
    if ((valArr.size() != TENSOR_NAME_ATTR_NUM) && (valArr.size() != (TENSOR_NAME_ATTR_NUM + 1U))) {
        ACL_LOG_INNER_ERROR("[Check][Params]tensorName[%s] cannot be devided into %zu parts",
            tensorName.c_str(), TENSOR_NAME_ATTR_NUM);
        return nullptr;
    }
    if (valArr[0U] != TENSOR_NAME_PREFIX) {
        ACL_LOG_INNER_ERROR("[Check][Param]cannot find Attr[%s] in tensorName[%s]",
            TENSOR_NAME_PREFIX, tensorName.c_str());
        return nullptr;
    }
    const int32_t base = 10;
    if ((valArr[1U] == MODEL_ID_STR) && (acl::StringUtils::IsDigit(valArr[2U]))) {
        const auto modelId = strtoul(valArr[2U].c_str(), nullptr, base);
        if (modelId != modelDesc->modelId) {
            ACL_LOG_INNER_ERROR("[Check][modelId]modelId[%lu] is invalid, tensorName[%s]", modelId, tensorName.c_str());
            return nullptr;
        }
    } else {
        ACL_LOG_INNER_ERROR("[Check][modelId]cannot find attr[%s] or modelId in tensorName[%s]",
            MODEL_ID_STR, tensorName.c_str());
        return nullptr;
    }
    if (acl::StringUtils::IsDigit(valArr[4U])) {
        const auto idex = strtoul(valArr[4U].c_str(), nullptr, base);
        if (valArr[3U] == TENSOR_INPUT_STR) {
            if (idex >= modelDesc->inputDesc.size()) {
                ACL_LOG_INNER_ERROR("[Check][index]inputDesc index[%lu] should be in [0, %zu), tensorName[%s]",
                    idex, modelDesc->inputDesc.size(), tensorName.c_str());
                return nullptr;
            }
            return modelDesc->inputDesc[idex].name.c_str();
        }
        if (valArr[3U] == TENSOR_OUTPUT_STR) {
            if (idex >= modelDesc->outputDesc.size()) {
                ACL_LOG_INNER_ERROR("[Check][index]outputDesc index[%lu] should be in [0, %zu), tensorName[%s]", idex,
                    modelDesc->outputDesc.size(), tensorName.c_str());
                return nullptr;
            }
            return modelDesc->outputDesc[idex].name.c_str();
        }
    }

    ACL_LOG_INNER_ERROR("[Find][Attr]cannot find [input_%s] or [ouput_%s] in tensorName[%s]", valArr[4U].c_str(),
        valArr[4U].c_str(), tensorName.c_str());
    return nullptr;
}

static size_t aclmdlGetTensorSize(aclmdlTensorDesc &tensorDesc, size_t idx)
{
    std::vector<int64_t> &dims = tensorDesc.dims;
    bool needCalcByMaxShapeRange = false;
    for (size_t i = 0U; i < dims.size(); ++i) {
        if (dims[i] < 0) {
            needCalcByMaxShapeRange = true;
            break;
        }
    }

    if (tensorDesc.shapeRanges.empty()) {
        needCalcByMaxShapeRange = false;
    }

    if (needCalcByMaxShapeRange) {
        std::vector<std::pair<int64_t, int64_t>> &shapeRanges = tensorDesc.shapeRanges;
        const size_t elementTypeSize = aclDataTypeSize(tensorDesc.dataType);
        size_t outputSizeByMaxShapeRange = elementTypeSize;
        for (size_t i = 0U; i < shapeRanges.size(); ++i) {
            if (shapeRanges[i].second <= 0) {
                ACL_LOG_INFO("max shape of shapeRanges[%zu] is [%ld], index[%zu]",
                             i, shapeRanges[i].second, idx);
                return 0U;
            }
            if (acl::CheckSizeTMultiOverflow(outputSizeByMaxShapeRange, static_cast<size_t>(shapeRanges[i].second),
                                             outputSizeByMaxShapeRange) == ACL_ERROR_FAILURE) {
                return 0U;
            }
        }
        return outputSizeByMaxShapeRange;
    }

    return tensorDesc.size;
}

static void GetTensorInfo(std::vector<aclmdlTensorDesc> &inputDesc, const gert::ModelIoDesc *const inputs,
                          size_t &inputNum)
{
    for (size_t i = 0U; i < inputNum; ++i) {
        aclmdlTensorDesc tensorDesc;
        const auto tempPtr = inputs + i;
        tensorDesc.size = static_cast<size_t>(tempPtr->GetSize());
        std::string inputStr;
        const char_t *const inputName = tempPtr->GetName();
        if (inputName != nullptr) {
            inputStr = std::string(inputName);
        }
        tensorDesc.name = inputStr;
        tensorDesc.format = static_cast<aclFormat>(tempPtr->GetOriginFormat());
        tensorDesc.dataType = static_cast<aclDataType>(tempPtr->GetDataType());
        const gert::Shape &shape = tempPtr->GetOriginShape();
        for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
            tensorDesc.dims.emplace_back(shape.GetDim(j));
        }
        const auto &shapeRangeVector = tempPtr->GetOriginShapeRangeVector();
        (void)tensorDesc.shapeRanges.insert(tensorDesc.shapeRanges.cend(),
                                            shapeRangeVector.cbegin(), shapeRangeVector.cend());
        const gert::Shape &aippShapeV2 = tempPtr->GetAippShape();
        for (size_t j = 0U; j < aippShapeV2.GetDimNum(); ++j) {
            tensorDesc.dimsV2.emplace_back(aippShapeV2.GetDim(j));
        }
        inputDesc.push_back(tensorDesc);
    }
}

static aclError RuntimeV2GetDesc(aclmdlDesc * const modelDesc, const uint32_t modelId)
{
    const auto executor = acl::AclResourceManager::GetInstance().GetExecutor(modelId);
    if (executor == nullptr) {
        ACL_LOG_ERROR("input modelId[%u] is invalid while get executor", modelId);
        return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
    }
    const auto &geModelDesc = executor->GetModelDesc();
    size_t inputNum = 0U;
    size_t outputNum = 0U;
    const gert::ModelIoDesc *const inputs = geModelDesc.GetAllInputsDesc(inputNum);
    GetTensorInfo(modelDesc->inputDesc, inputs, inputNum);
    const gert::ModelIoDesc *const outputs = geModelDesc.GetAllOutputsDesc(outputNum);
    GetTensorInfo(modelDesc->outputDesc, outputs, outputNum);

    modelDesc->modelId = modelId;
    aclError retVal = acl::RuntimeV2GetDynamicTensorInfo(modelDesc, geModelDesc);
    if (retVal != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][DynamicTensorInfo]get model dynamic info failed, result[%d], model id[%u]",
            retVal, modelId);
        return retVal;
    }

    retVal = acl::RuntimeV2GetModelOutputShapeInfo(modelDesc, geModelDesc);
    if (retVal != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][ModelOutputShapeInfo]get model output shape info failed, result[%d], model id[%u]",
            retVal, modelId);
        return retVal;
    }

    ACL_LOG_INFO("successfully execute aclmdlGetDesc, model id[%u]", modelId);
    return ACL_SUCCESS;
}
}

aclError aclmdlGetDescImpl(aclmdlDesc *modelDesc, uint32_t modelId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlGetDesc);
    ACL_LOG_INFO("start to execute aclmdlGetDesc, model id[%u]", modelId);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    modelDesc->Clear();
    if ((acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) &&
        (acl::AclResourceManager::GetInstance().GetExecutor(modelId) != nullptr)) {
        return acl::RuntimeV2GetDesc(modelDesc, modelId);
    }

    std::vector<ge::TensorDesc> inputDesc;
    std::vector<ge::TensorDesc> outputDesc;
    std::vector<ge::TensorDesc> inputDescV2;
    std::vector<ge::TensorDesc> outputDescV2;
    auto om2Executor = acl::AclResourceManager::GetInstance().GetOm2Executor(modelId);
    if (om2Executor != nullptr) {
        ge::Status ret = om2Executor->GetModelDescInfo(inputDesc, outputDesc);
        if (ret != ge::SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][ModelDescInfo]get om2 model description failed, ge result[%u], model id[%u]", ret,
                               modelId);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
        }
        ret = om2Executor->GetModelDescInfo(inputDescV2, outputDescV2, true);
        if (ret != ge::SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][ModelDescInfo]get om2 model description v2 failed, ge result[%u], model id[%u]",
                               ret, modelId);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
        }
    } else {
        ge::GeExecutor executor;
        ge::Status ret = executor.GetModelDescInfo(modelId, inputDesc, outputDesc);
        if (ret != ge::SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][ModelDescInfo]get model description failed, ge result[%u], model id[%u]", ret,
                               modelId);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
        }

        ret = executor.GetModelDescInfo(modelId, inputDescV2, outputDescV2, true);
        if (ret != ge::SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][ModelDescInfo]get model description v2 failed, ge result[%u], model id[%u]", ret,
                               modelId);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
        }
    }

    if ((inputDescV2.size() < inputDesc.size()) || (outputDescV2.size() < outputDesc.size())) {
        ACL_LOG_CALL_ERROR("[Get][ModelDescInfo]description v2 size less than description v1 size, model id[%u]",
                           modelId);
        return ACL_ERROR_FAILURE;
    }

    for (size_t i = 0U; i < inputDesc.size(); ++i) {
        ge::AscendString inputName;
        aclmdlTensorDesc tensorDesc;
        tensorDesc.size = static_cast<size_t>(inputDesc[i].GetSize());
        const ge::graphStatus retStatus = inputDesc[i].GetName(inputName);
        if (retStatus != ge::GRAPH_SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][TensorName]the %zu input tensor GetName failed.", i);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(retStatus));
        }
        std::string inputStr;
        if (inputName.GetString() != nullptr) {
            inputStr = std::string(inputName.GetString());
        }
        tensorDesc.name = inputStr;
        tensorDesc.format = static_cast<aclFormat>(inputDesc[i].GetFormat());
        tensorDesc.dataType = static_cast<aclDataType>(inputDesc[i].GetDataType());
        const ge::Shape shape = inputDesc[i].GetShape();
        tensorDesc.dims = shape.GetDims();
        const ge::Shape shapeV2 = inputDescV2[i].GetShape();
        tensorDesc.dimsV2 = shapeV2.GetDims();
        (void)inputDesc[i].GetShapeRange(tensorDesc.shapeRanges);
        modelDesc->inputDesc.push_back(tensorDesc);
    }

    for (size_t i = 0U; i < outputDesc.size(); ++i) {
        ge::AscendString outputName;
        aclmdlTensorDesc tensorDesc;
        tensorDesc.size = static_cast<size_t>(outputDesc[i].GetSize());
        const ge::graphStatus retStatus = outputDesc[i].GetName(outputName);
        if (retStatus != ge::GRAPH_SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][TensorName]the %zu output tensor GetName failed.", i);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(retStatus));
        }
        std::string outputStr;
        if (outputName.GetString() != nullptr) {
            outputStr = std::string(outputName.GetString());
        }
        tensorDesc.name = outputStr;
        tensorDesc.format = static_cast<aclFormat>(outputDesc[i].GetFormat());
        tensorDesc.dataType = static_cast<aclDataType>(outputDesc[i].GetDataType());
        const ge::Shape shape = outputDesc[i].GetShape();
        tensorDesc.dims = shape.GetDims();
        const ge::Shape shapeV2 = outputDescV2[i].GetShape();
        tensorDesc.dimsV2 = shapeV2.GetDims();
        (void)outputDesc[i].GetShapeRange(tensorDesc.shapeRanges);
        modelDesc->outputDesc.push_back(tensorDesc);
    }

    modelDesc->modelId = modelId;
    aclError retVal = acl::GetDynamicTensorInfo(modelDesc);
    if (retVal != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][DynamicTensorInfo]get model dynamic info failed, result[%d], model id[%u]",
            retVal, modelId);
        return retVal;
    }

    retVal = acl::GetModelOutputShapeInfo(modelDesc);
    if (retVal != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][ModelOutputShapeInfo]get model output shape info failed, result[%d], model id[%u]",
            retVal, modelId);
        return retVal;
    }

    ACL_LOG_INFO("successfully execute aclmdlGetDesc, model id[%u]", modelId);
    return ACL_SUCCESS;
}

static void GetModelTensorDesc(const ge::ModelInOutInfo &info, aclmdlDesc *modelDesc, bool isInput)
{
    const auto &descs = isInput ? info.input_desc : info.output_desc;
    for (const auto &desc : descs) {
        aclmdlTensorDesc tmpDesc;
        tmpDesc.name = desc.name;
        tmpDesc.size = desc.size;
        tmpDesc.format = (desc.format != ge::FORMAT_RESERVED) ?
                static_cast<aclFormat>(desc.format) : ACL_FORMAT_UNDEFINED;
        tmpDesc.dataType = (desc.dataType != ge::DT_UNDEFINED) ?
                static_cast<aclDataType>(desc.dataType) : ACL_DT_UNDEFINED;
        tmpDesc.dims = desc.dims;
        tmpDesc.dimsV2 = desc.dimsV2;
        tmpDesc.shapeRanges = desc.shape_ranges;
        if (isInput) {
            modelDesc->inputDesc.emplace_back(tmpDesc);
        } else {
            modelDesc->outputDesc.emplace_back(tmpDesc);
        }
    }
}

static aclError GetDescFromMem(const ge::ModelData &modelData, const ge::GeExecutor &executor, aclmdlDesc *modelDesc)
{
    ACL_LOG_INFO("call ge interface GetModelDescInfoFromMem");
    ge::ModelInOutInfo info;
    const auto ret = executor.GetModelDescInfoFromMem(modelData, info);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("GetModelDescInfoFromMem failed");
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    GetModelTensorDesc(info, modelDesc, true);
    GetModelTensorDesc(info, modelDesc, false);
    modelDesc->dynamicBatch = info.dynamic_batch;
    modelDesc->dynamicHW = info.dynamic_hw;
    modelDesc->dynamicDims = info.dynamic_dims;
    ACL_REQUIRES_OK(acl::GetModelOutputShapeInfoHelp(modelDesc, info.dynamic_output_shape));
    modelDesc->dataNameOrder = info.data_name_order;
    return ACL_SUCCESS;
}

aclError aclmdlGetDescFromFileImpl(aclmdlDesc *modelDesc, const char *modelPath)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlGetDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelPath);
    ACL_LOG_INFO("start to execute aclmdlGetDescFromFile, path is %s", modelPath);
    modelDesc->Clear();
    ge::GeExecutor executor;
    ge::ModelData data;
    ACL_LOG_INFO("call ge interface executor.LoadDataFromFile, path is %s", modelPath);
    const ge::Status ret = executor.LoadDataFromFile(modelPath, data);
    std::shared_ptr<void> dataAuto;
    dataAuto.reset(data.model_data, [](const void * const p) { delete[] static_cast<const uint8_t *>(p); });
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Model][FromFile]load model from file[%s] failed, ge result[%u]", modelPath, ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    ACL_REQUIRES_OK(GetDescFromMem(data, executor, modelDesc));
    ACL_LOG_INFO("end to execute aclmdlGetDescFromFile, path is %s", modelPath);
    return ACL_SUCCESS;
}

aclError aclmdlGetDescFromMemImpl(aclmdlDesc *modelDesc, const void *model, size_t modelSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlGetDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    ACL_REQUIRES_POSITIVE_WITH_INPUT_REPORT(modelSize);
    ACL_LOG_INFO("start to execute aclmdlGetDescFromMem, modelSize is %zu", modelSize);
    modelDesc->Clear();
    ge::GeExecutor executor;
    ge::ModelData modelData;
    modelData.model_data = const_cast<void *>(model);
    modelData.model_len = static_cast<uint64_t>(modelSize);
    ACL_REQUIRES_OK(GetDescFromMem(modelData, executor, modelDesc));
    ACL_LOG_INFO("end to execute aclmdlGetDescFromMem, modelSize is %zu", modelSize);
    return ACL_SUCCESS;
}

size_t aclmdlGetNumInputsImpl(aclmdlDesc *modelDesc)
{
    if (modelDesc == nullptr) {
        return 0U;
    }

    return modelDesc->inputDesc.size();
}

size_t aclmdlGetNumOutputsImpl(aclmdlDesc *modelDesc)
{
    if (modelDesc == nullptr) {
        return 0U;
    }

    return modelDesc->outputDesc.size();
}

size_t aclmdlGetInputSizeByIndexImpl(aclmdlDesc *modelDesc, size_t index)
{
    if ((modelDesc == nullptr) || (index >= modelDesc->inputDesc.size())) {
        ACL_LOG_INNER_ERROR("input param is invalid, modelDesc[%p], index[%zu]", modelDesc, index);
        return 0U;
    }

    aclmdlTensorDesc &tensorDesc = modelDesc->inputDesc[index];
    return acl::aclmdlGetTensorSize(tensorDesc, index);
}

size_t aclmdlGetOutputSizeByIndexImpl(aclmdlDesc *modelDesc, size_t index)
{
    if ((modelDesc == nullptr) || (index >= modelDesc->outputDesc.size())) {
        ACL_LOG_INNER_ERROR("input param is invalid, index[%zu]", index);
        return 0U;
    }
    aclmdlTensorDesc &tensorDesc = modelDesc->outputDesc[index];
    return acl::aclmdlGetTensorSize(tensorDesc, index);
}

aclmdlExecConfigHandle *aclmdlCreateExecConfigHandleImpl()
{
    return new(std::nothrow) aclmdlExecConfigHandle();
}

aclError aclmdlDestroyExecConfigHandleImpl(const aclmdlExecConfigHandle *handle)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_DELETE_AND_SET_NULL(handle);
    return ACL_SUCCESS;
}

aclmdlDataset *aclmdlCreateDatasetImpl()
{
    return new(std::nothrow) aclmdlDataset();
}

aclError aclmdlDestroyDatasetImpl(const aclmdlDataset *dataset)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);
    for (size_t i = 0U; i < dataset->blobs.size(); ++i) {
        ACL_DELETE_ARRAY_AND_SET_NULL((const_cast<aclmdlDataset *>(dataset))->blobs[i].tensorDesc);
    }
    ACL_DELETE_AND_SET_NULL(dataset);
    return ACL_SUCCESS;
}

aclError aclmdlAddDatasetBufferImpl(aclmdlDataset *dataset, aclDataBuffer *dataBuffer)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataBuffer);
    const acl::AclModelTensor tensor = acl::AclModelTensor(dataBuffer, nullptr);
    dataset->blobs.push_back(tensor);
    return ACL_SUCCESS;
}

size_t aclmdlGetDatasetNumBuffersImpl(const aclmdlDataset *dataset)
{
    if (dataset == nullptr) {
        ACL_LOG_ERROR("[Check][Dataset]input param[dataset] is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"dataset"}));
        return 0U;
    }

    return dataset->blobs.size();
}

aclDataBuffer *aclmdlGetDatasetBufferImpl(const aclmdlDataset *dataset, size_t index)
{
    if ((dataset == nullptr) || (index >= dataset->blobs.size())) {
        ACL_LOG_ERROR("[Check][Params]input param is invalid, dataset[%p], index[%zu]", dataset, index);
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("dataset[%p], index[%zu]", dataset, index);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"input param", errMsg.c_str(), "check failed"}));
        return nullptr;
    }

    return dataset->blobs[index].dataBuf;
}

aclTensorDesc *aclmdlGetDatasetTensorDescImpl(const aclmdlDataset *dataset, size_t index)
{
    ACL_REQUIRES_NOT_NULL_RET_NULL_INPUT_REPORT(dataset);
    if (index >= dataset->blobs.size()) {
        ACL_LOG_ERROR("[Check][Index]input param index[%zu] must be smaller than output databuf size[%zu]",
                      index, dataset->blobs.size());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"index", std::to_string(index).c_str(), "must be smaller than output databuf size"}));
        return nullptr;
    }
    return dataset->blobs[index].tensorDesc;
}

aclError aclmdlSetDatasetTensorDescImpl(aclmdlDataset *dataset, aclTensorDesc *tensorDesc, size_t index)
{
    ACL_REQUIRES_NOT_NULL(dataset);
    if (index >= dataset->blobs.size()) {
        ACL_LOG_ERROR("[Check][Index]input param index[%zu] must be smaller than input databuf size[%zu]",
                      index, dataset->blobs.size());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"index", std::to_string(index).c_str(), "must be smaller than input databuf size"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if (tensorDesc == nullptr) {
        ACL_DELETE_AND_SET_NULL(dataset->blobs[index].tensorDesc);
        ACL_LOG_INFO("Set tensorDesc for tensor[%zu] successfully, tensorDesc is nullptr", index);
        return ACL_SUCCESS;
    }

    if (dataset->blobs[index].tensorDesc == nullptr) {
        dataset->blobs[index].tensorDesc = new(std::nothrow) aclTensorDesc[1]{*tensorDesc};
        ACL_CHECK_MALLOC_RESULT(dataset->blobs[index].tensorDesc);
    } else {
        *(dataset->blobs[index].tensorDesc) = *tensorDesc;
    }
    ACL_LOG_INFO("Set tensorDesc %s for tensor[%zu] successfully", tensorDesc->DebugString().c_str(), index);
    return ACL_SUCCESS;
}

static aclError UnloadRt2Model(const uint32_t modelId, const bool isSessionShared)
{
    const auto executor = acl::AclResourceManager::GetInstance().GetExecutor(modelId);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (executor != nullptr) {
        ret  = executor->UnLoad();
    } else {
        ACL_LOG_ERROR("modelId[%u] is invalid", modelId);
        return static_cast<aclError>(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
    }
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Unload][Model]failed to unload model, modelId is %u, errorCode is %u", modelId, ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    auto session = acl::AclResourceManager::GetInstance().GetRtSession(modelId);
    // shared session can not destroy
    if ((session != nullptr ) && !isSessionShared) {
      session->DestroyResources();
    }
    return acl::AclResourceManager::GetInstance().DeleteExecutor(modelId);
}

static aclError UnloadModelInner(const uint32_t modelId, const bool isSessionShared = false)
{
    const auto om2_executor = acl::AclResourceManager::GetInstance().GetOm2Executor(modelId);
    if (om2_executor != nullptr) {
        const auto ret = acl::AclResourceManager::GetInstance().DeleteOm2Executor(modelId);
        if (ret != ACL_SUCCESS) {
          ACL_LOG_ERROR("failed to unload model, modelId is %u, errorCode is %u", modelId, ret);
          return ret;
        }
    } else if ((acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) &&
        (acl::AclResourceManager::GetInstance().GetExecutor(modelId) != nullptr)) {
        const auto ret = UnloadRt2Model(modelId, isSessionShared);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_ERROR("failed to unload model, modelId is %u, errorCode is %u", modelId, ret);
            return ret;
        }
    } else {
        ge::GeExecutor executor;
        ACL_LOG_INFO("call ge interface executor.UnloadModel, modelId[%u]", modelId);
        const ge::Status ret = executor.UnloadModel(modelId);
        if (ret != ge::SUCCESS) {
            ACL_LOG_CALL_ERROR("[Unload][Model]model unload failed, ge result[%u], modelId[%u]", ret, modelId);
            return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
        }
    }
    return ACL_SUCCESS;
}

static aclError QueryBundleSubModelInfo(const void *data, size_t modelSize, aclmdlBundleQueryInfo *queryInfo) {
  std::vector<std::pair<size_t, size_t>> subModelOffsetAndSize;
  size_t varSize = 0U;
  ACL_REQUIRES_OK(acl::GetBundleNumAndOffset(data, modelSize, varSize, subModelOffsetAndSize));
  queryInfo->varSize = static_cast<size_t>(varSize);
  for (const auto &ele : subModelOffsetAndSize) {
    acl::BundleSubModelInfo tmpInfo;
    tmpInfo.offset = ele.first;
    tmpInfo.modelSize = ele.second;
    bool isSupportRT2 = false;
    void *currentModePtr = ge::ValueToPtr(ge::PtrToValue(data) + ele.first);
    ACL_REQUIRES_OK(acl::IsSupportRuntimeV2WithModelData(currentModePtr, ele.second, isSupportRT2));
    // only static shape model can query size
    if (!isSupportRT2) {
      ge::GeExecutor executor;
      ACL_REQUIRES_CALL_GE_OK(executor.GetMemAndWeightSize(currentModePtr, ele.second,
                                                           tmpInfo.workSize, tmpInfo.weightSize), "Query size failed");
    }
    ACL_LOG_INFO("get work size %zu, weight size %zu", tmpInfo.workSize, tmpInfo.weightSize);
    queryInfo->subModelInfos.emplace_back(tmpInfo);
  }
  return ACL_SUCCESS;
}

static aclError LoadBundleSubModelFromMem(const void *const currentModePtr, const size_t modelSize,
                                          const std::string &modelPath, std::shared_ptr<gert::RtSession> &rtSession,
                                          const ge::ModelLoadArg &loadArgs, uint32_t &currentModelId,
                                          const char_t *const weightPath = nullptr, const int32_t priority = 0)
{
  bool isSupportRT2 = false;
  ACL_REQUIRES_OK(acl::IsSupportRuntimeV2WithModelData(currentModePtr, modelSize, isSupportRT2));
  aclError ret = ACL_SUCCESS;
  if (isSupportRT2) {
    ret = acl::RuntimeV2ModelLoadFromMemWithMem(currentModePtr, modelSize, modelPath,
                                                &currentModelId, loadArgs.weight_ptr, loadArgs.weight_size,
                                                weightPath, priority, loadArgs.file_constant_mems, rtSession);
  } else {
    ret = acl::ModelLoadFromMemWithMem(currentModePtr, modelSize, modelPath, &currentModelId, loadArgs, weightPath, priority);
  }
  return ret;
}

static aclError BundleInitFromMem(std::shared_ptr<const uint8_t> model, size_t modelSize, const std::string &modelPath,
                                  void *varWeightPtr, size_t varWeightSize, uint32_t *bundleId)
{
  // get bundle num from file header
  std::vector<std::pair<size_t, size_t>> subModelOffsetAndSize;
  size_t varSize = 0U;
  ACL_REQUIRES_OK(acl::GetBundleNumAndOffset(model.get(), modelSize, varSize, subModelOffsetAndSize));
  if ((varWeightPtr != nullptr) && (varWeightSize < varSize)) {
    ACL_LOG_ERROR("varWeightSize %zu is invalid, it cannot be smaller than model required size %zu", varWeightSize, varSize);
    const std::string errMsg = "it cannot be smaller than model required size " + std::to_string(varSize);
    acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
        std::vector<const char *>({"param", "value", "reason"}),
        std::vector<const char *>({"varWeightSize", std::to_string(varWeightSize).c_str(), errMsg.c_str()}));
    return ACL_ERROR_INVALID_PARAM;
  }
  auto rtSession = acl::AclResourceManager::GetInstance().CreateRtSession();
  ACL_REQUIRES_NOT_NULL(rtSession);
  rtSession->SetExternalVar(varWeightPtr, varWeightSize);
  acl::AclResourceManager::GetInstance().AddExecutor(*bundleId, nullptr, rtSession);
  acl::BundleModelInfo info;
  info.isInit = true;
  info.varSize = varSize;
  info.fromFilePath = modelPath;
  info.bundleModelSize = modelSize;
  info.bundleModelData = model;
  info.rtSession = rtSession;
  for (const auto &offsetSize : subModelOffsetAndSize) {
    acl::BundleSubModelInfo subInfo;
    subInfo.offset = offsetSize.first;
    subInfo.modelSize = offsetSize.second;
    info.subModelInfos.emplace_back(subInfo);
  }
  acl::AclResourceManager::GetInstance().SetBundleInfo(*bundleId, info);
  return ACL_SUCCESS;
}

static aclError BundleLoadFromMem(std::shared_ptr<const uint8_t> model, size_t modelSize, const std::string &modelPath,
                                  uint32_t *bundleId)
{
  // BundleLoadFromMem = BundleInitFromMem + load sub model
  ACL_REQUIRES_OK(BundleInitFromMem(model, modelSize, modelPath, nullptr, 0, bundleId));
  acl::BundleModelInfo info;
  ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().GetBundleInfo(*bundleId, info));
  info.isInit = false;
  const auto loadArgs = ConstructGeModelLoadArg(nullptr, 0, nullptr, 0, info.rtSession.get(), {});
  for (const auto &subInfo : info.subModelInfos) {
    uint32_t currentModelId = 0U;
    void *currentModePtr = ge::ValueToPtr(ge::PtrToValue(model.get()) + subInfo.offset);
    aclError ret = LoadBundleSubModelFromMem(currentModePtr, subInfo.modelSize, modelPath, info.rtSession,
                                             loadArgs, currentModelId);
    if (ret != ACL_SUCCESS) {
      for (const auto &ele : info.loadedSubModelIdSet) {
        (void)UnloadModelInner(ele);
      }
      return ret;
    }
    info.loadedSubModelId.emplace_back(currentModelId); // only BundleLoadFromMem need set
    info.loadedSubModelIdSet.insert(currentModelId);
  }
  if (!info.fromFilePath.empty()) {
    info.bundleModelData = nullptr; // need reset nullptr when aclmdlBundleLoadFromFile to ensure compatibility
    info.bundleModelSize = 0U;
  }
  acl::AclResourceManager::GetInstance().SetBundleInfo(*bundleId, info);
  return ACL_SUCCESS;
}

aclError aclmdlBundleLoadFromFileImpl(const char *modelPath, uint32_t *bundleId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlBundleLoadFromFile);
    ACL_LOG_INFO("start to execute aclmdlBundleLoadFromFile");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelPath);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(bundleId);

    // 1. load model data from file
    ge::ModelData modelData;
    modelData.om_path = modelPath;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ACL_LOG_INFO("call ge interface gert::LoadDataFromFile");
    ret = gert::LoadDataFromFile(modelPath, modelData);
    std::shared_ptr<uint8_t> data;
    data.reset(ge::PtrToPtr<void, uint8_t>(modelData.model_data), std::default_delete<uint8_t[]>());
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][Model]failed to load model from file by runtime2.0, ge errorCode is %u", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    ACL_REQUIRES_OK(BundleLoadFromMem(data, modelData.model_len, modelPath, bundleId));
    ACL_LOG_INFO("successfully execute aclmdlBundleLoadFromFile, bundle id is %u", *bundleId);
    return ACL_SUCCESS;
}

aclError aclmdlBundleLoadFromMemImpl(const void *model,  size_t modelSize, uint32_t *bundleId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlBundleLoadFromMem);
    ACL_LOG_INFO("start to execute aclmdlBundleLoadFromMem, modelSize[%zu]", modelSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    ACL_REQUIRES_POSITIVE_WITH_INPUT_REPORT(modelSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(bundleId);

    std::shared_ptr<const uint8_t> data;
    // no delete func
    data.reset(ge::PtrToPtr<void, uint8_t>(model), [](const uint8_t* const p) { (void) p; });
    ACL_REQUIRES_OK(BundleLoadFromMem(data, modelSize, "", bundleId));

    ACL_LOG_INFO("execute aclmdlBundleLoadFromMem success, bundleId[%u]", *bundleId);
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromFileImpl(const char *modelPath, uint32_t *modelId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadFromFile);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelPath);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    bool isSupportOm2 = false;
    ACL_REQUIRES_OK(gert::IsOm2Model(modelPath, isSupportOm2));
    if (isSupportOm2) {
        const aclError ret = acl::Om2ModelLoadFromFileWithMem(modelPath, modelId);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_ERROR("Load OM2 model from file failed, path [%s], errorCode [%u]", modelPath, ret);
        }
        ACL_LOG_INFO("Successfully execute aclmdlLoadFromFile, modelId[%u], modelPath[%s]", *modelId, modelPath);
        return ret;
    }

    aclError ret = ACL_SUCCESS;
    bool isSupportRT2 = false;
    ACL_REQUIRES_OK(acl::IsSupportRuntimeV2WithModelPath(modelPath, isSupportRT2));
    if (isSupportRT2) {
        ret = acl::RuntimeV2ModelLoadFromFileWithMem(modelPath, modelId, nullptr, 0U, 0);
    } else {
        ge::ModelLoadArg loadArgs{};
        ret = acl::ModelLoadFromFileWithMem(modelPath, modelId, loadArgs, 0);
    }
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("Load model from file failed!");
        return ret;
    }
    ACL_LOG_INFO("successfully execute aclmdlLoadFromFile");
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromFileWithMemImpl(const char *modelPath, uint32_t *modelId,
                                   void *workPtr, size_t workSize,
                                   void *weightPtr, size_t weightSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadFromFileWithMem);
    ACL_LOG_INFO("start to execute aclmdlLoadFromFileWithMem, workSize[%zu], weightSize[%zu]", workSize, weightSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelPath);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    aclError ret = ACL_SUCCESS;
    bool isSupportRT2 = false;
    ACL_REQUIRES_OK(acl::IsSupportRuntimeV2WithModelPath(modelPath, isSupportRT2));
    if (isSupportRT2) {
        ret = acl::RuntimeV2ModelLoadFromFileWithMem(modelPath, modelId, weightPtr, weightSize, 0);
    } else {
        const auto loadArgs = ConstructGeModelLoadArg(workPtr, workSize, weightPtr, weightSize);
        ret = acl::ModelLoadFromFileWithMem(modelPath, modelId, loadArgs, 0);
    }
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ACL_LOG_INFO("Load model from file[%s] with memory success, modelId[%u]", modelPath, *modelId);
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromMemImpl(const void *model, size_t modelSize, uint32_t *modelId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadFromMem);
    ACL_LOG_INFO("start to execute aclmdlLoadFromMem, modelSize[%zu]", modelSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    ACL_REQUIRES_POSITIVE_WITH_INPUT_REPORT(modelSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    bool isSupportOm2 = false;
    ACL_REQUIRES_OK(gert::IsOm2Model(model, modelSize, isSupportOm2));
    if (isSupportOm2) {
        auto ret = acl::Om2ModelLoadFromMemWithMem(model, modelSize, modelId);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_ERROR("Load OM2 model from memory failed, errorCode is %u", ret);
        }
        ACL_LOG_INFO("Successfully execute aclmdlLoadFromMem, modelId[%u]", *modelId);
        return ret;
    }

    bool isSupportRT2 = false;
    ACL_REQUIRES_OK(acl::IsSupportRuntimeV2WithModelData(model, modelSize, isSupportRT2));
    aclError ret = ACL_SUCCESS;
    if (isSupportRT2) {
        ret = acl::RuntimeV2ModelLoadFromMemWithMem(model, modelSize, "", modelId, nullptr, 0U, nullptr, 0);
    } else {
        ge::ModelLoadArg loadArgs{};
        ret = acl::ModelLoadFromMemWithMem(model, modelSize, "", modelId, loadArgs, nullptr, 0);
    }
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    ACL_LOG_INFO("Load model from data success, modelId[%u]", *modelId);
    return ACL_SUCCESS;
}

// get some info from bundleId
aclError aclmdlBundleGetModelNumImpl(uint32_t bundleId, size_t *modelNum)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelNum);
  acl::BundleModelInfo bundleInfos;
  ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().GetBundleInfo(bundleId, bundleInfos));
  *modelNum = bundleInfos.subModelInfos.size();
  ACL_LOG_INFO("get bundleId %u model num %zu", bundleId, *modelNum);
  return ACL_SUCCESS;
}

aclError aclmdlBundleGetModelIdImpl(uint32_t bundleId, size_t index, uint32_t *modelId)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
  acl::BundleModelInfo bundleInfos;
  ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().GetBundleInfo(bundleId, bundleInfos));
  if (bundleInfos.isInit) {
    ACL_LOG_ERROR("aclmdlBundleGetModelId is supported only aclmdlBundleLoadFromFile or aclmdlBundleLoadFromMem is executed");
    return ACL_ERROR_API_NOT_SUPPORT;
  }
  if (index >= bundleInfos.loadedSubModelId.size()) {
    ACL_LOG_ERROR("bundleId %u input index %zu should be smaller than bundle size %zu",
                  bundleId, index, bundleInfos.loadedSubModelId.size());
    return ACL_ERROR_INVALID_PARAM;
  }
  *modelId = bundleInfos.loadedSubModelId[index];
  ACL_LOG_INFO("get bundleId %u index %zu model id %u", bundleId, index, *modelId);
  return ACL_SUCCESS;
}

aclmdlBundleQueryInfo *aclmdlBundleCreateQueryInfoImpl() {
  return new(std::nothrow) aclmdlBundleQueryInfo();
}

aclError aclmdlBundleDestroyQueryInfoImpl(aclmdlBundleQueryInfo *queryInfo)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(queryInfo);
  ACL_DELETE_AND_SET_NULL(queryInfo);
  return ACL_SUCCESS;
}

aclError aclmdlBundleQueryInfoFromFileImpl(const char* fileName, aclmdlBundleQueryInfo *queryInfo)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(fileName);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(queryInfo);
  ACL_LOG_INFO("start to execute aclmdlBundleQueryInfoFromFile %s", fileName);
  // 1. load model data from file
  ge::ModelData modelData;
  modelData.om_path = fileName;
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  ACL_LOG_INFO("call ge interface gert::LoadDataFromFile");
  ret = gert::LoadDataFromFile(fileName, modelData);
  std::shared_ptr<uint8_t> data;
  data.reset(ge::PtrToPtr<void, uint8_t>(modelData.model_data), std::default_delete<uint8_t[]>());
  if (ret != ge::GRAPH_SUCCESS) {
    ACL_LOG_CALL_ERROR("[Load][Model]failed to load model from file by runtime2.0, ge errorCode is %u", ret);
    return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
  }
  ACL_REQUIRES_OK(QueryBundleSubModelInfo(modelData.model_data, modelData.model_len, queryInfo));
  ACL_LOG_INFO("end to execute aclmdlBundleQueryInfoFromFile %s", fileName);
  return ACL_SUCCESS;
}

aclError aclmdlBundleQueryInfoFromMemImpl(const void *model, size_t modelSize, aclmdlBundleQueryInfo *queryInfo)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(queryInfo);
  ACL_LOG_INFO("start to execute aclmdlBundleQueryInfoFromMem");
  ACL_REQUIRES_OK(QueryBundleSubModelInfo(model, modelSize, queryInfo));
  ACL_LOG_INFO("end to execute aclmdlBundleQueryInfoFromMem");
  return ACL_SUCCESS;
}

aclError aclmdlBundleGetQueryModelNumImpl(const aclmdlBundleQueryInfo *queryInfo, size_t *modelNum)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(queryInfo);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelNum);
  *modelNum = queryInfo->subModelInfos.size();
  return ACL_SUCCESS;
}

aclError aclmdlBundleGetVarWeightSizeImpl(const aclmdlBundleQueryInfo *queryInfo, size_t *variableWeightSize)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(queryInfo);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(variableWeightSize);
  *variableWeightSize = queryInfo->varSize;
  return ACL_SUCCESS;
}

aclError aclmdlBundleGetSizeImpl(const aclmdlBundleQueryInfo *queryInfo, size_t index, size_t *workSize,
                             size_t *constWeightSize)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(queryInfo);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(workSize);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(constWeightSize);
  if (index >= queryInfo->subModelInfos.size()) {
    ACL_LOG_ERROR("index %zu should be less than %zu", index, queryInfo->subModelInfos.size());
    return ACL_ERROR_INVALID_PARAM;
  }
  *workSize = queryInfo->subModelInfos[index].workSize;
  *constWeightSize = queryInfo->subModelInfos[index].weightSize;
  return ACL_SUCCESS;
}

aclError aclmdlBundleInitFromFileImpl(const char* modelPath, void *varWeightPtr, size_t varWeightSize,
                                  uint32_t *bundleId)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelPath);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(bundleId);
  ACL_LOG_INFO("start to execute aclmdlBundleInitFromFile, model path %s, varWeightSize %zu",
               modelPath, varWeightSize);
  ge::ModelData modelData;
  modelData.om_path = modelPath;
  ACL_LOG_INFO("call ge interface gert::LoadDataFromFile");
  ACL_REQUIRES_CALL_GE_OK(gert::LoadDataFromFile(modelPath, modelData), "load data form file %s failed", modelPath);
  std::shared_ptr<uint8_t> tmpData;
  tmpData.reset(ge::PtrToPtr<void, uint8_t>(modelData.model_data), std::default_delete<uint8_t[]>());
  ACL_REQUIRES_NOT_NULL(tmpData);
  ACL_REQUIRES_OK(BundleInitFromMem(tmpData, modelData.model_len, modelPath, varWeightPtr, varWeightSize, bundleId));
  ACL_LOG_INFO("end to execute aclmdlBundleInitFromFile, model path %s, varWeightSize %zu, bundleId %u",
               modelPath, varWeightSize, *bundleId);
  return ACL_SUCCESS;
}

aclError aclmdlBundleInitFromMemImpl(const void* model, size_t modelSize, void *varWeightPtr,
                                 size_t varWeightSize, uint32_t *bundleId)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(bundleId);
  ACL_LOG_INFO("start to execute aclmdlBundleInitFromMem, model size %zu, varWeightSize %zu",
               modelSize, varWeightSize);
  std::shared_ptr<const uint8_t> tmpData;
  // no delete func
  tmpData.reset(ge::PtrToPtr<void, uint8_t>(model), [](const uint8_t* const p) { (void) p; });
  ACL_REQUIRES_OK(BundleInitFromMem(tmpData, modelSize, "", varWeightPtr, varWeightSize, bundleId));
  ACL_LOG_INFO("end to execute aclmdlBundleInitFromMem, model size %zu, varWeightSize %zu, bundleId %u",
               modelSize, varWeightSize, *bundleId);
  return ACL_SUCCESS;
}

aclError aclmdlBundleLoadModelImpl(uint32_t bundleId, size_t index, uint32_t *modelId)
{
  return aclmdlBundleLoadModelWithMemImpl(bundleId, index, nullptr, 0U, nullptr, 0U, modelId);
}

static aclError GetTargetBundleInfo(uint32_t bundleId, acl::BundleModelInfo &bundleInfos) {
  const std::unique_lock<std::mutex> lock(aclmdlBundleMutex);
  ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().GetBundleInfo(bundleId, bundleInfos));
  bool need_load_mem_from_file = !bundleInfos.isInit && !bundleInfos.fromFilePath.empty() &&
                                 (bundleInfos.bundleModelData == nullptr);
  if (need_load_mem_from_file) {
    ACL_LOG_INFO("bundle mem should allocated again when aclmdlBundleLoadFromFile called");
    ge::ModelData modelData;
    modelData.om_path = bundleInfos.fromFilePath;
    ACL_LOG_INFO("call ge interface gert::LoadDataFromFile");
    ACL_REQUIRES_CALL_GE_OK(gert::LoadDataFromFile(bundleInfos.fromFilePath.c_str(), modelData),
                            "load bundle om %s failed", bundleInfos.fromFilePath.c_str());
    std::shared_ptr<uint8_t> data;
    data.reset(ge::PtrToPtr<void, uint8_t>(modelData.model_data), std::default_delete<uint8_t[]>());
    bundleInfos.bundleModelData = data;
    bundleInfos.bundleModelSize = modelData.model_len;
    acl::AclResourceManager::GetInstance().SetBundleInfo(bundleId, bundleInfos);
  }
  return ACL_SUCCESS;
}

aclError aclmdlBundleLoadModelWithMemImpl(uint32_t bundleId, size_t index, void *workPtr, size_t workSize, void *weightPtr,
                                      size_t weightSize, uint32_t *modelId)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
  ACL_PROFILING_REG(acl::AclProfType::AclmdlBundleLoadModelWithMem);
  ACL_LOG_INFO("strat to execute aclmdlBundleLoadModelWithMem, bundleId %u, index %zu", bundleId, index);
  acl::BundleModelInfo bundleInfos;
  ACL_REQUIRES_OK(GetTargetBundleInfo(bundleId, bundleInfos));
  if (index >= bundleInfos.subModelInfos.size()) {
    ACL_LOG_ERROR("index %zu should be smaller than %zu", index, bundleInfos.subModelInfos.size());
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_REQUIRES_NOT_NULL(bundleInfos.bundleModelData);
  const size_t offset = bundleInfos.subModelInfos[index].offset;
  const size_t modelSize = bundleInfos.subModelInfos[index].modelSize;
  void *currentModePtr = ge::ValueToPtr(ge::PtrToValue(bundleInfos.bundleModelData.get()) + offset);
  const auto loadArgs = ConstructGeModelLoadArg(workPtr, workSize, weightPtr, weightSize,
                                                bundleInfos.rtSession.get(), {});
  ACL_REQUIRES_OK(LoadBundleSubModelFromMem(currentModePtr, modelSize, bundleInfos.fromFilePath,
                                            bundleInfos.rtSession, loadArgs, *modelId));
  acl::AclResourceManager::GetInstance().AddBundleSubmodelId(bundleId, *modelId);
  ACL_LOG_INFO("end to execute aclmdlBundleLoadModelWithMem, bundleId %u, index %zu", bundleId, index);
  return ACL_SUCCESS;
}

aclError aclmdlBundleLoadModelWithConfigImpl(uint32_t bundleId, size_t index, aclmdlConfigHandle *handle,
                                         uint32_t *modelId)
{
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
  ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
  ACL_PROFILING_REG(acl::AclProfType::AclmdlBundleLoadModelWithConfig);
  ACL_LOG_INFO("start to execute aclmdlBundleLoadModelWithConfig, bundleId %u, index %zu", bundleId, index);
  acl::BundleModelInfo bundleInfos;
  ACL_REQUIRES_OK(GetTargetBundleInfo(bundleId, bundleInfos));
  if (index >= bundleInfos.subModelInfos.size()) {
    ACL_LOG_ERROR("index %zu should be smaller than %zu", index, bundleInfos.subModelInfos.size());
    return ACL_ERROR_INVALID_PARAM;
  }
  acl::UpdateGraphOptions(OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, std::to_string(handle->reuseZeroCopy));
  ACL_REQUIRES_NOT_NULL(bundleInfos.bundleModelData);
  const size_t offset = bundleInfos.subModelInfos[index].offset;
  const size_t modelSize = bundleInfos.subModelInfos[index].modelSize;
  void *currentModePtr = ge::ValueToPtr(ge::PtrToValue(bundleInfos.bundleModelData.get()) + offset);
  const auto loadArgs = ConstructGeModelLoadArg(handle->workPtr, handle->workSize, handle->weightPtr, handle->weightSize,
                                                bundleInfos.rtSession.get(), handle->fileConstantMem, handle->withoutGraph);

  ACL_REQUIRES_OK(LoadBundleSubModelFromMem(currentModePtr, modelSize, bundleInfos.fromFilePath,
                                            bundleInfos.rtSession, loadArgs, *modelId, handle->weightPath.c_str(),
                                            handle->priority));
  acl::AclResourceManager::GetInstance().AddBundleSubmodelId(bundleId, *modelId);
  ACL_LOG_INFO("end to execute aclmdlBundleLoadModelWithConfig, bundleId %u, index %zu, model id is %u",
               bundleId, index, *modelId);
  return ACL_SUCCESS;
}

aclError aclmdlBundleUnloadModelImpl(uint32_t bundleId, uint32_t modelId)
{
  ACL_PROFILING_REG(acl::AclProfType::AclmdlBundleUnloadModel);
  ACL_LOG_INFO("start to execute aclmdlBundleUnloadModel bundleId %u, modelId %u", bundleId, modelId);
  acl::BundleModelInfo bundleInfos;
  ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().GetBundleInfo(bundleId, bundleInfos));
  if (bundleInfos.loadedSubModelIdSet.find(modelId) == bundleInfos.loadedSubModelIdSet.end()) {
    ACL_LOG_ERROR("current modelId %u is not bundleId %u sub model", modelId, bundleId);
    return ACL_ERROR_INVALID_PARAM;
  }
  ACL_REQUIRES_OK(UnloadModelInner(modelId, true));
  acl::AclResourceManager::GetInstance().DeleteBundleSubmodelId(bundleId, modelId);
  ACL_LOG_INFO("end to execute aclmdlBundleUnloadModel bundleId %u, modelId %u", bundleId, modelId);
  return ACL_SUCCESS;
}

aclError aclmdlLoadFromMemWithMemImpl(const void *model, size_t modelSize,
                                  uint32_t *modelId, void *workPtr, size_t workSize,
                                  void *weightPtr, size_t weightSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadFromMemWithMem);
    ACL_LOG_INFO("start to execute aclmdlLoadFromMemWithMem, modelSize[%zu], workSize[%zu], weightSize[%zu]",
        modelSize, workSize, weightSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    bool isSupportRT2 = false;
    ACL_REQUIRES_OK(acl::IsSupportRuntimeV2WithModelData(model, modelSize, isSupportRT2));
    aclError ret = ACL_SUCCESS;
    if (isSupportRT2) {
        ret = acl::RuntimeV2ModelLoadFromMemWithMem(model, modelSize, "", modelId,
                                                    weightPtr, weightSize, nullptr, 0);
    } else {
        const auto loadArgs = ConstructGeModelLoadArg(workPtr, workSize, weightPtr, weightSize);
        ret = acl::ModelLoadFromMemWithMem(model, modelSize, "", modelId, loadArgs, nullptr, 0);
    }
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ACL_LOG_INFO("successfully execute aclmdlLoadFromMemWithMem, modelSize[%zu], workSize[%zu], weightSize[%zu]",
        modelSize, workSize, weightSize);
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromFileWithQImpl(const char *modelPath, uint32_t *modelId, const uint32_t *inputQ,
                                 size_t inputQNum, const uint32_t *outputQ, size_t outputQNum)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadFromFileWithQ);
    if (aclmdlCheckQueueParam(inputQ, inputQNum, outputQ, outputQNum) != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }
    ACL_LOG_INFO("start to execute aclmdlLoadFromFileWithQ, inputQNum[%zu], outputQNum[%zu]", inputQNum, outputQNum);
    std::vector<uint32_t> inputQVec(inputQ, inputQ + inputQNum);
    std::vector<uint32_t> outputQVec(outputQ, outputQ + outputQNum);
    ge::ModelQueueArg args{.input_queue_ids = std::move(inputQVec), .output_queue_ids = std::move(outputQVec),
                        .file_constant_mems = {}, false};
    const aclError ret = acl::ModelLoadFromFileWithQ(modelPath, modelId, args, 0);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ACL_LOG_INFO("successfully execute aclmdlLoadFromFileWithQ, inputQNum[%zu], outputQNum[%zu]",
        inputQNum, outputQNum);
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromMemWithQImpl(const void *model, size_t modelSize, uint32_t *modelId,
    const uint32_t *inputQ, size_t inputQNum, const uint32_t *outputQ, size_t outputQNum)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadFromMemWithQ);
    if (aclmdlCheckQueueParam(inputQ, inputQNum, outputQ, outputQNum) != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }
    ACL_LOG_INFO("start to execute aclmdlLoadFromMemWithQ, modelSize[%zu], inputQNum[%zu], outputQNum[%zu]",
        modelSize, inputQNum, outputQNum);
    std::vector<uint32_t> inputQVec(inputQ, inputQ + inputQNum);
    std::vector<uint32_t> outputQVec(outputQ, outputQ + outputQNum);
    ge::ModelQueueArg args{.input_queue_ids = std::move(inputQVec), .output_queue_ids = std::move(outputQVec),
                        .file_constant_mems = {}, false};
    const aclError ret = acl::ModelLoadFromMemWithQ(model, modelSize, modelId, args, 0);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ACL_LOG_INFO("successfully execute aclmdlLoadFromMemWithQ, modelSize[%zu], inputQNum[%zu], outputQNum[%zu]",
        modelSize, inputQNum, outputQNum);
    return ACL_SUCCESS;
}

aclError aclmdlExecuteImpl(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlExecute);
    ACL_LOG_INFO("start to execute aclmdlExecute, modelId[%u]", modelId);
    aclError ret = ACL_SUCCESS;
    if (acl::AclResourceManager::GetInstance().GetOm2Executor(modelId) != nullptr) {
        ret = acl::Om2ModelExecute(modelId, input, output, false, nullptr);
    } else if ((acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) &&
        (acl::AclResourceManager::GetInstance().GetExecutor(modelId) != nullptr)) {
        ret = acl::RuntimeV2ModelExecute(modelId, input, output, false, nullptr);
    } else {
        ret = acl::ModelExecute(modelId, input, output, false, nullptr);
    }
    if (ret == ACL_SUCCESS) {
        ACL_LOG_INFO("aclmdlExecute success, modelId[%u]", modelId);
    } else {
        ACL_LOG_INNER_ERROR("[Exec][Model]modelId[%u] execute failed, result[%d]", modelId, ret);
    }
    return ret;
}

aclError aclmdlExecuteV2Impl(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream,
                         const aclmdlExecConfigHandle *handle)
{
    ACL_LOG_INFO("start to execute aclmdlExecuteV2, modelId[%u]", modelId);

    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    if (handle->streamSyncTimeout >= DEFAULT_SYNC_TIMEOUT) {
        ACL_LOG_INFO("stream synchronize timeout = %dms", handle->streamSyncTimeout);
        ge::GetContext().SetStreamSyncTimeout(handle->streamSyncTimeout);
    }
    if (handle->eventSyncTimeout >= DEFAULT_SYNC_TIMEOUT) {
        ACL_LOG_INFO("event synchronize timeout = %dms", handle->eventSyncTimeout);
        ge::GetContext().SetEventSyncTimeout(handle->eventSyncTimeout);
    }

    aclError ret = ACL_SUCCESS;
    if ((acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) &&
        (acl::AclResourceManager::GetInstance().GetExecutor(modelId) != nullptr)) {
        ret = acl::RuntimeV2ModelExecute(modelId, input, output, false, stream);
    } else {
        ret = acl::ModelExecute(modelId, input, output, false, stream);
        if (stream != nullptr) {
            const aclError rtErr = aclrtSynchronizeStreamWithTimeout(stream, handle->streamSyncTimeout);
            if (rtErr != ACL_ERROR_NONE) {
                ACL_LOG_CALL_ERROR("synchronize stream failed, runtime result = %d", static_cast<int32_t>(rtErr));
                return ACL_GET_ERRCODE_RTS(rtErr);
            }
        }
    }
    if (ret == ACL_SUCCESS) {
        ACL_LOG_INFO("aclmdlExecuteV2 success, modelId[%u]", modelId);
    } else {
        ACL_LOG_INNER_ERROR("[Exec][Model]modelId[%u] execute failed, result[%d]", modelId, ret);
    }
    return ret;
}

aclError aclmdlExecuteAsyncImpl(uint32_t modelId, const aclmdlDataset *input,
                            aclmdlDataset *output, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlExecuteAsync);
    ACL_LOG_INFO("start to execute aclmdlExecuteAsync, modelId[%u]", modelId);

    aclError ret = ACL_SUCCESS;
    if (acl::AclResourceManager::GetInstance().GetOm2Executor(modelId) != nullptr) {
        ret = acl::Om2ModelExecute(modelId, input, output, true, stream);
    } else if ((acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true)) &&
        (acl::AclResourceManager::GetInstance().GetExecutor(modelId) != nullptr)) {
        ret = acl::RuntimeV2ModelExecute(modelId, input, output, true, stream);
    } else {
        ret = acl::ModelExecute(modelId, input, output, true, stream);
    }
    if (ret == ACL_SUCCESS) {
        ACL_LOG_INFO("aclmdlExecuteAsync success, modelId[%u]", modelId);
    } else {
        ACL_LOG_INNER_ERROR("[Exec][Model]aclmdlExecuteAsync failed, result[%d], modelId[%u]", ret, modelId);
    }
    return ret;
}

aclError aclmdlUnloadImpl(uint32_t modelId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlUnload);
    ACL_LOG_INFO("start to execute aclmdlUnload, modelId[%u]", modelId);
    if (acl::AclResourceManager::GetInstance().IsBundleInnerId(modelId)) {
      ACL_LOG_ERROR("this modeId %u is bundle inner modelId, please ues aclmdlBundleUnload api instead", modelId);
      return ACL_ERROR_INVALID_PARAM;
    }
    ACL_REQUIRES_OK(UnloadModelInner(modelId));
    ACL_LOG_INFO("aclmdlUnload success, modelId[%u]", modelId);
    return ACL_SUCCESS;
}

aclError aclmdlBundleUnloadImpl(uint32_t bundleId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlBundleUnload);
    ACL_LOG_INFO("start to execute aclmdlBundleUnload %u", bundleId);
    acl::BundleModelInfo bundleInfos;
    ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().GetBundleInfo(bundleId, bundleInfos));
    aclError finalRet = ACL_SUCCESS;
    for (auto &modelId: bundleInfos.loadedSubModelIdSet) {
      // unload all and check result
      const auto ret = UnloadModelInner(modelId);
      if (ret != ACL_SUCCESS) {
        finalRet = ret;
      }
    }
    ACL_REQUIRES_OK(finalRet);
    acl::AclResourceManager::GetInstance().DeleteBundleInfo(bundleId);
    // release session var manager
    auto bundleSession = acl::AclResourceManager::GetInstance().GetRtSession(bundleId);
    ACL_REQUIRES_NOT_NULL(bundleSession);
    bundleSession->DestroyResources();
    ACL_REQUIRES_OK(acl::AclResourceManager::GetInstance().DeleteExecutor(bundleId));
    ACL_LOG_INFO("end to execute aclmdlBundleUnload %u", bundleId);
    return ACL_SUCCESS;
}

aclError aclmdlQuerySizeImpl(const char *fileName, size_t *workSize, size_t *weightSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlQuerySize);
    ACL_LOG_INFO("start to execute aclmdlQuerySize");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(fileName);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(workSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(weightSize);

    ge::GeExecutor executor;
    const std::string path(fileName);
    size_t work;
    size_t weight;
    ACL_LOG_DEBUG("call ge interface executor.GetMemAndWeightSize");
    const ge::Status ret = executor.GetMemAndWeightSize(path, work, weight);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][MemAndWeightSize]query size failed, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    *workSize = work;
    *weightSize = weight;
    ACL_LOG_INFO("success to get size from file[%s], work size[%zu] bytes, weight size[%zu] bytes",
        fileName, *workSize, *weightSize);

    return ACL_SUCCESS;
}

aclError aclmdlQuerySizeFromMemImpl(const void *model, size_t modelSize, size_t *workSize, size_t *weightSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlQuerySizeFromMem);
    ACL_LOG_INFO("start to execute ACL_QueryModelSizeFromMem, modelSize[%zu]", modelSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(workSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(weightSize);

    ge::GeExecutor executor;
    size_t work;
    size_t weight;
    ACL_LOG_DEBUG("call ge interface executor.GetMemAndWeightSize, modelSize[%zu]", modelSize);
    const ge::Status ret = executor.GetMemAndWeightSize(model, modelSize, work, weight);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][MemAndWeightSize]query size from mem failed, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    *workSize = work;
    *weightSize = weight;
    ACL_LOG_INFO("success to get size from mem, work size[%zu] bytes, weight size[%zu] bytes", *workSize,
        *weightSize);

    return ACL_SUCCESS;
}

aclError aclmdlSetDynamicBatchSizeImpl(uint32_t modelId, aclmdlDataset *dataset, size_t index, uint64_t batchSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlSetDynamicBatchSize);
    ACL_LOG_INFO("start to execute aclmdlSetDynamicBatchSize, modelId[%u], index[%zu], batchSize[%lu]",
        modelId, index, batchSize);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);

    if (batchSize == 0U) {
        ACL_LOG_INNER_ERROR("[Check][Batchsize]input param[batchSize] invalid, batchSize can't be zero");
        return ACL_ERROR_INVALID_PARAM;
    }

    const aclDataBuffer *const buf = aclmdlGetDatasetBufferImpl(dataset, index);
    if (buf == nullptr) {
        ACL_LOG_INNER_ERROR("[Check][buf]failed to get data buffer by index[%zu], dataset buffer is null", index);
        return ACL_ERROR_INVALID_PARAM;
    }

    void *const devPtr = aclGetDataBufferAddr(buf);
    if (devPtr == nullptr) {
        ACL_LOG_INNER_ERROR("[Check][devPtr]get addr by index[%zu] failed, data buffer addr can not be null", index);
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t memSize = aclGetDataBufferSizeV2(buf);

    dataset->dynamicBatchSize = batchSize;
    dataset->dynamicResolutionHeight = 0U;
    dataset->dynamicResolutionWidth = 0U;
    ACL_LOG_DEBUG("call ge interface executor.SetDynamicBatchSize, batchSize[%lu]", batchSize);
    ge::GeExecutor executor;
    const ge::Status ret = executor.SetDynamicBatchSize(modelId, devPtr, memSize, batchSize);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Set][DynamicBatchSize]set DynamicBatchSize failed, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    ACL_LOG_INFO("successfully execute aclmdlSetDynamicBatchSize, modelId[%u], index[%zu], batchSize[%lu]",
        modelId, index, batchSize);
    return ACL_SUCCESS;
}

aclError aclmdlSetDynamicHWSizeImpl(uint32_t modelId, aclmdlDataset *dataset, size_t index,
    uint64_t height, uint64_t width)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlSetDynamicHWSize);
    ACL_LOG_INFO("start to execute aclmdlSetDynamicHWSize, modelId[%u], index[%zu], height[%lu], width[%lu]",
        modelId, index, height, width);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);

    if ((height == 0U) || (width == 0U)) {
        ACL_LOG_INNER_ERROR("[Check][Params]height[%lu] or width[%lu] is invalid, can't be zero.", height, width);
        return ACL_ERROR_INVALID_PARAM;
    }

    aclDataBuffer *const buffer = aclmdlGetDatasetBufferImpl(dataset, index);
    if (buffer == nullptr) {
        ACL_LOG_INNER_ERROR("[Check][buffer]get data buffer by index[%zu] failed, dataset buffer can not be null",
            index);
        return ACL_ERROR_INVALID_PARAM;
    }

    void *const devPtr = aclGetDataBufferAddr(buffer);
    if (devPtr == nullptr) {
        ACL_LOG_INNER_ERROR("[Check][devPtr]get addr by index[%zu] failed, data buffer addr can not be nullptr", index);
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t memSize = aclGetDataBufferSizeV2(buffer);

    dataset->dynamicBatchSize = 0U;
    dataset->dynamicResolutionHeight = height;
    dataset->dynamicResolutionWidth = width;
    ACL_LOG_DEBUG("call ge interface executor.SetDynamicImageSize, height[%lu],width[%lu]", height, width);
    ge::GeExecutor executor;
    const ge::Status ret = executor.SetDynamicImageSize(modelId, devPtr, memSize, height, width);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Set][DynamicImageSize]Set dynamic image size, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    ACL_LOG_INFO("successfully execute aclmdlSetDynamicHWSize, modelId[%u], index[%zu], height[%lu], width[%lu]",
        modelId, index, height, width);
    return ACL_SUCCESS;
}

aclError aclmdlSetInputDynamicDimsImpl(uint32_t modelId, aclmdlDataset *dataset, size_t index, const aclmdlIODims *dims)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlSetInputDynamicDims);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataset);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);
    if (dims->dimCount == 0U) {
        ACL_LOG_ERROR("[Check][dimCount]dimCount[%zu] is invalid, can't be zero.", dims->dimCount);
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("dimCount[%u] can't be zero", dims->dimCount);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"dimCount", "0", errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
    }

    ACL_LOG_INFO("start to execute aclmdlSetInputDynamicDims, modelId[%u], index[%zu], dimCount[%zu]",
                 modelId, index, dims->dimCount);
    aclDataBuffer *const buffer = aclmdlGetDatasetBufferImpl(dataset, index);
    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(buffer != nullptr, ACL_ERROR_INVALID_PARAM,
        "[Check][buffer]get data buffer by index[%zu] failed, dataset buffer can not be null", index);

    void *const devPtr = aclGetDataBufferAddr(buffer);
    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(devPtr != nullptr, ACL_ERROR_INVALID_PARAM,
        "[Get][devPtr]get addr by index[%zu] failed, data buffer addr can not be null", index);
    const uint64_t memSize = aclGetDataBufferSizeV2(buffer);

    dataset->dynamicBatchSize = 0U;
    dataset->dynamicResolutionHeight = 0U;
    dataset->dynamicResolutionWidth = 0U;
    dataset->dynamicDims.clear();
    std::vector<uint64_t> curAllDims;
    for (size_t i = 0U; i < static_cast<std::size_t>(dims->dimCount); ++i) {
        curAllDims.push_back(static_cast<size_t>(dims->dims[i]));
    }
    ACL_LOG_DEBUG("Call ge interface executor.SetDynamicDims, dimCount[%zu]", dims->dimCount);
    ACL_LOG_DEBUG("Cur all dims size %zu", curAllDims.size());
    ge::GeExecutor executor;
    ge::Status ret = executor.SetDynamicDims(modelId, devPtr, memSize, curAllDims);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Set][DynamicDims]set dynamic dims failed, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    std::vector<uint64_t> curDynmaicDims;
    ret = executor.GetCurDynamicDims(modelId, curAllDims, curDynmaicDims);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][CurDynamicDims]get current dynamic dims failed, ge result[%u]", ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }
    ACL_LOG_DEBUG("current dynamic dims size %zu", curDynmaicDims.size());
    dataset->dynamicDims = curDynmaicDims;

    ACL_LOG_INFO("successfully execute aclmdlSetInputDynamicDims, modelId[%u], index[%zu], dimCount[%zu]",
                 modelId, index, dims->dimCount);
    return ACL_SUCCESS;
}

aclError aclmdlGetInputDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);

    const aclError ret = acl::GetDims(modelDesc, TensorType::INPUT_TENSOR_TYPE, DimsType::DIMS_TYPE_V1, index, dims);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][Dims]get input dims failed, result[%d], index[%zu], modelId[%u]", ret, index,
            modelDesc->modelId);
    }

    return ret;
}

aclError aclmdlGetOutputDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);

    const aclError ret = acl::GetDims(modelDesc, TensorType::OUTPUT_TENSOR_TYPE, DimsType::DIMS_TYPE_V1, index, dims);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][Dims]get output dims failed, result[%d], index[%zu], modelId[%u]",
            ret, index, modelDesc->modelId);
    }

    return ret;
}

aclError aclmdlGetInputDimsV2Impl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);

    const aclError ret = acl::GetDims(modelDesc, TensorType::INPUT_TENSOR_TYPE, DimsType::DIMS_TYPE_V2, index, dims);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][Dims]get input dims(v2) failed, result[%d], index[%zu], modelId[%u]",
            ret, index, modelDesc->modelId);
    }

    return ret;
}

aclError aclmdlGetInputDimsRangeImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODimsRange *dimsRange)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dimsRange);

    const aclError ret = acl::GetDimsRange(modelDesc, TensorType::INPUT_TENSOR_TYPE, index, dimsRange);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Get][DimsRange]get input dims range failed, result[%d], index[%zu], modelId[%u]",
            ret, index, modelDesc->modelId);
    }

    return ret;
}

aclError aclmdlGetCurOutputDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    if (acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(true) &&
        (acl::AclResourceManager::GetInstance().GetExecutor(modelDesc->modelId) != nullptr)) {
        ACL_LOG_WARN("This api does not support dynamic model, please check.");
        return ACL_ERROR_API_NOT_SUPPORT;
    }
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);
    const uint32_t modelId = modelDesc->modelId;
    const size_t descSize = modelDesc->outputDesc.size();
    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(index < descSize, ACL_ERROR_INVALID_PARAM,
        "[Check][descSize]aclmdlGetCurOutputDims failed, index[%zu] should be smaller "
        "than tensor size[%zu], modelId[%u]", index, descSize, modelId);

    std::vector<int64_t> geShapeInfo;
    ge::GeExecutor executor;
    int32_t dynamicType = static_cast<int32_t>(ge::FIXED);
    const ge::Status geRet = executor.GetCurShape(modelId, geShapeInfo, dynamicType);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][CurShape]can not get current shape, ge result[%d], modelId[%u]", geRet, modelId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(geRet));
    }
    // dynamic batch type is 1, dynamic hw type is 2, dynamic dims type is 3;
    // static model or not set dynamic shape info, dynamic type is 0, other value is invalid
    aclError aclRet;
    const size_t shapeSize = geShapeInfo.size();
    if ((dynamicType != static_cast<int32_t>(ge::DYNAMIC_DIMS)) && (shapeSize > 2U)) {
        ACL_LOG_INNER_ERROR("[Check][dynamicType]shapeSize[%zu] is invalid, modelId[%u]", shapeSize, modelId);
        return ACL_ERROR_GE_FAILURE;
    }
    if (shapeSize == 0U) {
        ACL_LOG_DEBUG("Dynamic type is 0, model[%u] is static or not set dynamic shape info", modelId);
        aclRet = acl::GetDims(modelDesc, TensorType::OUTPUT_TENSOR_TYPE, DimsType::DIMS_TYPE_V1, index, dims);
        ACL_REQUIRES_OK_WITH_INNER_MESSAGE(aclRet,
            "[Get][Dims]get current output dims failed, result[%d], index[%zu], modelId[%u]",
            aclRet, index, modelId);
        return ACL_SUCCESS;
    }

    size_t curGearIndex = 0U;
    std::vector<uint64_t> shapeInfo;
    for (auto &it : geShapeInfo) {
        shapeInfo.emplace_back(static_cast<uint64_t>(it));
    }
    aclRet = acl::GetCurGearIndex(modelDesc, shapeInfo, dynamicType, curGearIndex);
    ACL_REQUIRES_OK_WITH_INNER_MESSAGE(aclRet,
        "[Get][CurGearIndex]get current gear index failed, result[%d], index[%zu], "
        "modelId[%u], dynamicBatchSize[%zu], dynamicHWSize[%zu]", aclRet, index, modelId,
        modelDesc->dynamicBatch.size(), modelDesc->dynamicHW.size());

    aclRet = acl::GetCurOuputShapeInfo(modelDesc, index, curGearIndex, dims);
    ACL_REQUIRES_OK_WITH_INNER_MESSAGE(aclRet,
        "[Get][CurOuputShapeInfo]get current output shape info failed, result[%d], "
        "index[%zu], modelId[%u], the size of dynamicOutputShape[%zu]", aclRet, index, modelId,
        modelDesc->dynamicOutputShape.size());

    return ACL_SUCCESS;
}

const char *aclmdlGetOpAttrImpl(aclmdlDesc *modelDesc, const char *opName, const char *attr)
{
    ACL_LOG_INFO("start to execute aclmdlGetOpAttr");
    ACL_REQUIRES_NOT_NULL_RET_NULL(modelDesc);
    ACL_REQUIRES_NOT_NULL_RET_NULL(opName);
    ACL_REQUIRES_NOT_NULL_RET_NULL(attr);

    const std::string opNameStr(opName);
    const std::string attrStr(attr);
    if (attrStr != ACL_ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES) {
        ACL_LOG_INNER_ERROR("failed to execute aclmdlGetOpAttr, attr[%s] is invalid, only support "
            "ACL_ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES", attrStr.c_str());
        return nullptr;
    }

    const std::unique_lock<std::mutex> lock(aclmdlGetOpAttrMutex);
    std::map<std::string, std::map<std::string, std::string>>::const_iterator itOpName =
        modelDesc->opAttrValueMap.find(opName);
    if (itOpName != modelDesc->opAttrValueMap.cend()) {
        std::map<std::string, std::string>::const_iterator itAttr = itOpName->second.find(attr);
        if (itAttr != itOpName->second.cend()) {
            ACL_LOG_INFO("opName is [%s], the value of attr [%s] is %s", opName, attr, itAttr->second.c_str());
            return itAttr->second.c_str();
        }
    }

    ge::GeExecutor executor;
    const uint32_t modelId = modelDesc->modelId;
    ACL_LOG_INFO("Call ge interface executor.GetOpAttr, modelId is [%u], opName is [%s], attr is [%s]",
        modelId, opName, attr);
    std::string attrValue;
    const ge::Status ret = executor.GetOpAttr(modelId, opNameStr, attrStr, attrValue);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][Opattr]Execute GetOpAttr failed, ge result[%u], modelId[%u]", ret, modelId);
        return nullptr;
    }
    ACL_LOG_INFO("Execute aclmdlGetOpAttr successfully, opName is [%s], the value of attr[%s] is %s", opName, attr,
        attrValue.c_str());
    modelDesc->opAttrValueMap[opNameStr][attrStr] = attrValue;
    return modelDesc->opAttrValueMap[opNameStr][attrStr].c_str();
}

const char *aclmdlGetInputNameByIndexImpl(const aclmdlDesc *modelDesc, size_t index)
{
    ACL_LOG_INFO("start to execute aclmdlGetInputNameByIndex");
    if (modelDesc == nullptr) {
        ACL_LOG_ERROR("[Check][ModelDesc]modelDesc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"modelDesc"}));
        return "";
    }

    return acl::aclmdlGetNameByIndex(modelDesc->inputDesc, index);
}

const char *aclmdlGetOutputNameByIndexImpl(const aclmdlDesc *modelDesc, size_t index)
{
    ACL_LOG_INFO("start to execute aclmdlGetOutputNameByIndex");
    if (modelDesc == nullptr) {
        ACL_LOG_ERROR("[Check][ModelDesc]modelDesc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"modelDesc"}));
        return "";
    }

    return acl::aclmdlGetNameByIndex(modelDesc->outputDesc, index);
}

aclFormat aclmdlGetInputFormatImpl(const aclmdlDesc *modelDesc, size_t index)
{
    ACL_LOG_INFO("start to execute aclmdlGetInputFormat");
    if (modelDesc == nullptr) {
        ACL_LOG_ERROR("[Check][ModelDesc]modelDesc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"modelDesc"}));
        return ACL_FORMAT_UNDEFINED;
    }

    return acl::aclmdlGetFormat(modelDesc->inputDesc, index);
}

aclFormat aclmdlGetOutputFormatImpl(const aclmdlDesc *modelDesc, size_t index)
{
    ACL_LOG_INFO("start to execute aclmdlGetOutputFormat");
    if (modelDesc == nullptr) {
        ACL_LOG_ERROR("[Check][ModelDesc]modelDesc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"params"}),
            std::vector<const char *>({"modelDesc"}));
        return ACL_FORMAT_UNDEFINED;
    }

    return acl::aclmdlGetFormat(modelDesc->outputDesc, index);
}

aclDataType aclmdlGetInputDataTypeImpl(const aclmdlDesc *modelDesc, size_t index)
{
    ACL_LOG_INFO("start to execute aclmdlGetInputDataType");
    if (modelDesc == nullptr) {
        ACL_LOG_ERROR("[Check][ModelDesc]modelDesc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"modelDesc"}));
        return ACL_DT_UNDEFINED;
    }

    return acl::aclmdlGetDataType(modelDesc->inputDesc, index);
}

aclDataType aclmdlGetOutputDataTypeImpl(const aclmdlDesc *modelDesc, size_t index)
{
    ACL_LOG_INFO("start to execute aclmdlGetOutputDataType");
    if (modelDesc == nullptr) {
        ACL_LOG_ERROR("[Check][ModelDesc]modelDesc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"modelDesc"}));
        return ACL_DT_UNDEFINED;
    }

    return acl::aclmdlGetDataType(modelDesc->outputDesc, index);
}

aclError aclmdlGetInputIndexByNameImpl(const aclmdlDesc *modelDesc, const char *name, size_t *index)
{
    ACL_LOG_INFO("start to execute aclmdlGetInputIndexByName");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(name);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(index);

    ACL_LOG_INFO("successfully execute aclmdlGetInputIndexByName");
    return acl::aclmdlGetIndexByName(modelDesc->inputDesc, name, index);
}

aclError aclmdlGetOutputIndexByNameImpl(const aclmdlDesc *modelDesc, const char *name, size_t *index)
{
    ACL_LOG_INFO("start to execute aclmdlGetOutputIndexByName");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(name);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(index);

    ACL_LOG_INFO("successfully execute aclmdlGetOutputIndexByName");
    return acl::aclmdlGetIndexByName(modelDesc->outputDesc, name, index);
}

aclError aclmdlGetDynamicBatchImpl(const aclmdlDesc *modelDesc, aclmdlBatch *batch)
{
    ACL_LOG_INFO("start to execute aclmdlGetDynamicBatch");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(batch);

    const size_t batchCnt = modelDesc->dynamicBatch.size();
    if (batchCnt > static_cast<size_t>(ACL_MAX_BATCH_NUM)) {
        ACL_LOG_ERROR("[Check][batchCnt]aclmdlGetBatch failed, batch count[%zu] is larger than max batch num[%d]",
            batchCnt, ACL_MAX_BATCH_NUM);
        const std::string errMsg =
            acl::AclErrorLogManager::FormatStr("batch count[%zu] is larger than max batch num[%d]",
                batchCnt, ACL_MAX_BATCH_NUM);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"batch count", std::to_string(batchCnt).c_str(), errMsg.c_str()}));
        return ACL_ERROR_STORAGE_OVER_LIMIT;
    }

    batch->batchCount = batchCnt;
    if (batchCnt == 0U) {
        ACL_LOG_INFO("batch count is 0");
        return ACL_SUCCESS;
    }

    for (size_t i = 0U; i < batchCnt; ++i) {
        batch->batch[i] = modelDesc->dynamicBatch[i];
    }

    ACL_LOG_INFO("successfully execute aclmdlGetDynamicBatch");
    return ACL_SUCCESS;
}

aclError aclmdlGetDynamicHWImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw)
{
    (void)index;
    ACL_LOG_INFO("start to execute aclmdlGetDynamicHW");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(hw);

    const size_t hwCnt = modelDesc->dynamicHW.size();
    if (hwCnt > static_cast<size_t>(ACL_MAX_HW_NUM)) {
        ACL_LOG_INNER_ERROR("[Check][hwCnt]aclmdlGetHW failed, hw count[%zu] is larger than max[%d]",
            hwCnt, ACL_MAX_HW_NUM);
        return ACL_ERROR_STORAGE_OVER_LIMIT;
    }

    hw->hwCount = hwCnt;
    if (hwCnt == 0U) {
        ACL_LOG_INFO("hw count is 0");
        return ACL_SUCCESS;
    }

    for (size_t i = 0U; i < hwCnt; ++i) {
        for (size_t j = 0U; j < 2U; ++j) { // dynamic hw,size is 2
            hw->hw[i][j] = modelDesc->dynamicHW[i][j];
        }
    }
    ACL_LOG_INFO("successfully execute aclmdlGetDynamicHW");
    return ACL_SUCCESS;
}

aclError aclmdlGetInputDynamicGearCountImpl(const aclmdlDesc *modelDesc, size_t index, size_t *gearCount)
{
    ACL_LOG_INFO("start to execute aclmdlGetInputDynamicGearCount");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(gearCount);

    if (index != static_cast<size_t>(-1)) {
        ACL_LOG_INNER_ERROR("[Check][index]aclmdlGetInputDynamicGearCount failed, index must be -1 while input is %zu.",
            index);
        return ACL_ERROR_INVALID_PARAM;
    }

    const size_t dimCnt = modelDesc->dynamicDims.size();
    if (dimCnt > static_cast<size_t>(ACL_MAX_DIM_CNT)) {
        ACL_LOG_INNER_ERROR("[Check][dimCnt]aclmdlGetInputDynamicGearCount failed, dimCnt[%zu] is "
            "larger than max[%d]", dimCnt, ACL_MAX_DIM_CNT);
        return ACL_ERROR_STORAGE_OVER_LIMIT;
    }

    if (dimCnt == 0U) {
        *gearCount = 0U;
        ACL_LOG_INFO("Gear count is 0");
        return ACL_SUCCESS;
    }
    *gearCount = dimCnt;
    ACL_LOG_INFO("successfully execute aclmdlGetInputDynamicGearCount");
    return ACL_SUCCESS;
}

aclError aclmdlGetInputDynamicDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims, size_t gearCount)
{
    ACL_LOG_INFO("start to execute aclmdlGetInputDynamicDims");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);
    if (index != static_cast<std::size_t>(-1)) {
        ACL_LOG_INNER_ERROR("[Check][index]aclmdlGetInputDynamicDims failed, index must be -1 but it is %zu", index);
        return ACL_ERROR_INVALID_PARAM;
    }
    if (gearCount < modelDesc->dynamicDims.size()) {
        ACL_LOG_INNER_ERROR("[Check][gearCount]Gear count[%zu] can not less than model's dynamic gear count[%zu]",
            gearCount, modelDesc->dynamicDims.size());
        return ACL_ERROR_INVALID_PARAM;
    }
    std::vector<int64_t> allRawDims;
    for (auto &dataName : modelDesc->dataNameOrder) {
        for (auto &inputDesc : modelDesc->inputDesc) {
            if (inputDesc.name == dataName) {
                (void)allRawDims.insert(allRawDims.cend(), inputDesc.dims.cbegin(), inputDesc.dims.cend());
            }
        }
    }
    if (allRawDims.size() > ACL_MAX_DIM_CNT) {
        ACL_LOG_ERROR("current dynamic dims size %zu can not be larger than %d", allRawDims.size(), ACL_MAX_DIM_CNT);
        return ACL_ERROR_FAILURE;
    }

    for (size_t i = 0U; i < modelDesc->dynamicDims.size(); ++i) {
        size_t begIndex = 0U;
        for (size_t j = 0U; j < allRawDims.size(); ++j) {
            if (allRawDims[j] < 0) {
                if (begIndex >= modelDesc->dynamicDims[i].size()) {
                    ACL_LOG_INNER_ERROR("[Check][begIndex]User input data index[%zu] shape size overflow", index);
                    return ACL_ERROR_INVALID_PARAM;
                }
                dims[i].dims[j] = static_cast<int64_t>(modelDesc->dynamicDims[i][begIndex]);
                begIndex++;
            } else {
                dims[i].dims[j] = allRawDims[j];
            }
            dims[i].dimCount = allRawDims.size();
        }
    }
    ACL_LOG_INFO("successfully execute aclmdlGetInputDynamicDims");
    return ACL_SUCCESS;
}

aclError aclmdlCreateAndGetOpDescImpl(uint32_t deviceId, uint32_t streamId, uint32_t taskId, char *opName,
    size_t opNameLen, aclTensorDesc **inputDesc, size_t *numInputs, aclTensorDesc **outputDesc, size_t *numOutputs)
{
    ACL_LOG_INFO("start to execute aclmdlCreateAndGetOpDesc, deviceId[%u], streamId[%u], taskId[%u]",
        deviceId, streamId, taskId);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opName);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(inputDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(outputDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(numInputs);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(numOutputs);

    ge::GeExecutor executor;
    ge::OpDescInfo opDescInfo;
    ACL_LOG_DEBUG("call ge interface executor.GetOpDescInfo");
    const ge::Status geRet = executor.GetOpDescInfo(deviceId, streamId, taskId, opDescInfo);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][OpDescInfo]get op desc faild, ge result[%d], deviceId[%u], streamId[%u], taskId[%u]",
            geRet, deviceId, streamId, taskId);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(geRet));
    }

    if (opNameLen <= opDescInfo.op_name.length()) {
        ACL_LOG_INNER_ERROR("[Check][opNameLen]input length = %zu must be larger than op name real length = %zu",
            opNameLen, opDescInfo.op_name.length());
        return ACL_ERROR_INVALID_PARAM;
    }
    const auto ret = strncpy_s(opName, opNameLen, opDescInfo.op_name.c_str(),
        opDescInfo.op_name.length());
    if (ret != EOK) {
        ACL_LOG_INNER_ERROR("[Copy][OpName]copy op name failed, copy errorCode = %d, input opNameLen = %zu, "
            "real opNameLen = %zu", ret, opNameLen, opDescInfo.op_name.length());
        return ACL_ERROR_FAILURE;
    }

    const size_t inputNum = opDescInfo.input_format.size();
    const size_t outputNum = opDescInfo.output_format.size();

    ACL_REQUIRES_POSITIVE(inputNum);
    ACL_REQUIRES_POSITIVE(outputNum);
    *inputDesc = new(std::nothrow) aclTensorDesc[inputNum];
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(*inputDesc);
    *outputDesc = new(std::nothrow) aclTensorDesc[outputNum];
    if (*outputDesc == nullptr) {
        ACL_LOG_INNER_ERROR("[Check][outputDesc]alloc outputDesc memory failed");
        ACL_DELETE_ARRAY_AND_SET_NULL(*inputDesc);
        return ACL_ERROR_FAILURE;
    }
    ACL_REQUIRES_EQ(opDescInfo.input_data_type.size(), inputNum);
    ACL_REQUIRES_EQ(opDescInfo.input_shape.size(), inputNum);
    ACL_REQUIRES_EQ(opDescInfo.input_addrs.size(), inputNum);
    for (size_t idx = 0U; idx < inputNum; ++idx) {
        (*inputDesc)[idx].format = static_cast<aclFormat>(opDescInfo.input_format[idx]);
        (*inputDesc)[idx].dataType = static_cast<aclDataType>(opDescInfo.input_data_type[idx]);
        (*inputDesc)[idx].dims.assign(opDescInfo.input_shape[idx].begin(), opDescInfo.input_shape[idx].end());
        (*inputDesc)[idx].address = opDescInfo.input_addrs[idx];
    }
    ACL_REQUIRES_EQ(opDescInfo.output_data_type.size(), outputNum);
    ACL_REQUIRES_EQ(opDescInfo.output_shape.size(), outputNum);
    ACL_REQUIRES_EQ(opDescInfo.output_addrs.size(), outputNum);
    for (size_t idx = 0U; idx < outputNum; ++idx) {
        (*outputDesc)[idx].format = static_cast<aclFormat>(opDescInfo.output_format[idx]);
        (*outputDesc)[idx].dataType = static_cast<aclDataType>(opDescInfo.output_data_type[idx]);
        (*outputDesc)[idx].dims.assign(opDescInfo.output_shape[idx].begin(), opDescInfo.output_shape[idx].end());
        (*outputDesc)[idx].address = opDescInfo.output_addrs[idx];
    }
    *numInputs = inputNum;
    *numOutputs = outputNum;
    ACL_LOG_INFO("successfully execute aclmdlCreateAndGetOpDesc, deviceId[%u], streamId[%u], "
        "taskId[%u], numInputs[%zu], numOutputs[%zu]", deviceId, streamId, taskId, *numInputs, *numOutputs);
    return ACL_SUCCESS;
}

aclError aclmdlLoadWithConfigImpl(const aclmdlConfigHandle *handle, uint32_t *modelId)
{
    ACL_PROFILING_REG(acl::AclProfType::AclmdlLoadWithConfig);
    ACL_LOG_INFO("start to execute aclmdlLoadWithConfig");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelId);
    if (!acl::CheckMdlConfigHandle(handle)) {
        ACL_LOG_ERROR("[Check][ConfigHandle]model config is invalid because some params may not be set or invalid");
        return ACL_ERROR_INVALID_PARAM;
    }
    acl::UpdateGraphOptions(OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, std::to_string(handle->reuseZeroCopy));
    auto &file_constant_mems = handle->fileConstantMem;
    for (const auto &file_constant_mem : file_constant_mems) {
        ACL_LOG_INFO("file constant name[%s], device memory address[%p], device memory size[%zu]",
                     file_constant_mem.file_name.c_str(), file_constant_mem.device_mem, file_constant_mem.mem_size);
    }
    bool isSupportRT2 = false;
    switch (static_cast<int32_t>(handle->mdlLoadType)) {
        case ACL_MDL_LOAD_FROM_FILE:
        {
            ACL_REQUIRES_OK(acl::CheckIsRuntimeV2WithConfig(handle, handle->loadPath.c_str(), nullptr, 0, isSupportRT2));
            if (isSupportRT2) {
                return acl::RuntimeV2ModelLoadFromFileWithMem(handle->loadPath.c_str(), modelId, nullptr, 0U,
                                                              handle->priority, file_constant_mems);
            }
            ge::ModelLoadArg loadArgs{};
            loadArgs.file_constant_mems = file_constant_mems;
            loadArgs.need_clear_dfx_cache = handle->withoutGraph;
            return acl::ModelLoadFromFileWithMem(handle->loadPath.c_str(), modelId, loadArgs, handle->priority);
        }
        case ACL_MDL_LOAD_FROM_FILE_WITH_MEM:
        {
            ACL_REQUIRES_OK(acl::CheckIsRuntimeV2WithConfig(handle, handle->loadPath.c_str(), nullptr, 0, isSupportRT2));
            if (isSupportRT2) {
                return acl::RuntimeV2ModelLoadFromFileWithMem(handle->loadPath.c_str(), modelId, handle->weightPtr,
                                                              handle->weightSize, handle->priority,
                                                              file_constant_mems);
            }
            const auto loadArgs = ConstructGeModelLoadArg(handle->workPtr, handle->workSize, handle->weightPtr,
                handle->weightSize, nullptr, file_constant_mems, handle->withoutGraph);
            return acl::ModelLoadFromFileWithMem(handle->loadPath.c_str(), modelId, loadArgs, handle->priority);
        }
        case ACL_MDL_LOAD_FROM_MEM:
        {
            ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle->mdlAddr);
            ACL_REQUIRES_OK(acl::CheckIsRuntimeV2WithConfig(handle, nullptr, handle->mdlAddr, handle->mdlSize, isSupportRT2));
            if (isSupportRT2) {
                return acl::RuntimeV2ModelLoadFromMemWithMem(handle->mdlAddr, handle->mdlSize, handle->loadPath, modelId, nullptr, 0U,
                                                             handle->weightPath.c_str(), handle->priority,
                                                             file_constant_mems);
            }
            ge::ModelLoadArg loadArgs{};
            loadArgs.file_constant_mems = file_constant_mems;
            loadArgs.need_clear_dfx_cache = handle->withoutGraph;
            return acl::ModelLoadFromMemWithMem(handle->mdlAddr, handle->mdlSize, "", modelId, loadArgs,
                                                handle->weightPath.c_str(), handle->priority);
        }
        case ACL_MDL_LOAD_FROM_MEM_WITH_MEM:
        {
            ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle->mdlAddr);
            ACL_REQUIRES_OK(acl::CheckIsRuntimeV2WithConfig(handle, nullptr, handle->mdlAddr, handle->mdlSize, isSupportRT2));
            if (isSupportRT2) {
                return acl::RuntimeV2ModelLoadFromMemWithMem(handle->mdlAddr, handle->mdlSize, handle->loadPath, modelId,
                                                             handle->weightPtr, handle->weightSize,
                                                             handle->weightPath.c_str(), handle->priority, file_constant_mems);
            }
            const auto loadArgs = ConstructGeModelLoadArg(handle->workPtr, handle->workSize, handle->weightPtr,
                handle->weightSize, nullptr, file_constant_mems, handle->withoutGraph);
            return acl::ModelLoadFromMemWithMem(handle->mdlAddr, handle->mdlSize, handle->loadPath, modelId, loadArgs,
                                                handle->weightPath.c_str(), handle->priority);
        }
        case ACL_MDL_LOAD_FROM_FILE_WITH_Q:
        {
            if (aclmdlCheckQueueParam(handle->inputQ, handle->inputQNum, handle->outputQ, handle->outputQNum) !=
                ACL_SUCCESS) {
                return ACL_ERROR_INVALID_PARAM;
            }
            std::vector<uint32_t> inputQVec(handle->inputQ, handle->inputQ + handle->inputQNum);
            std::vector<uint32_t> outputQVec(handle->outputQ, handle->outputQ + handle->outputQNum);
            ge::ModelQueueArg args{.input_queue_ids = std::move(inputQVec), .output_queue_ids = std::move(outputQVec),
                                .file_constant_mems = file_constant_mems, handle->withoutGraph};
            return acl::ModelLoadFromFileWithQ(handle->loadPath.c_str(), modelId, args, handle->priority);
        }
        case ACL_MDL_LOAD_FROM_MEM_WITH_Q:
        {
            if (aclmdlCheckQueueParam(handle->inputQ, handle->inputQNum, handle->outputQ, handle->outputQNum) !=
                ACL_SUCCESS) {
                return ACL_ERROR_INVALID_PARAM;
            }
            std::vector<uint32_t> inputQVec(handle->inputQ, handle->inputQ + handle->inputQNum);
            std::vector<uint32_t> outputQVec(handle->outputQ, handle->outputQ + handle->outputQNum);
            ge::ModelQueueArg que_args{.input_queue_ids = std::move(inputQVec), .output_queue_ids = std::move(outputQVec),
                    .file_constant_mems = file_constant_mems, handle->withoutGraph};
            return acl::ModelLoadFromMemWithQ(handle->mdlAddr, handle->mdlSize, modelId, que_args, handle->priority);
        }
        default:
            ACL_LOG_INNER_ERROR("[Load][Model]model load type[%zu] is invalid, it should be in [%d, %d]",
                handle->mdlLoadType, ACL_MDL_LOAD_FROM_FILE, ACL_MDL_LOAD_FROM_MEM_WITH_Q);
            return ACL_ERROR_INVALID_PARAM;
    }
}

const char *aclmdlGetTensorRealNameImpl(const aclmdlDesc *modelDesc, const char *name)
{
    ACL_LOG_INFO("start to execute aclmdlGetTensorName");
    ACL_REQUIRES_NOT_NULL_RET_NULL_INPUT_REPORT(modelDesc);
    ACL_REQUIRES_NOT_NULL_RET_NULL_INPUT_REPORT(name);
    const char_t *realTensorName = acl::GetRealTensorName(modelDesc, name);
    if (realTensorName != nullptr) {
        ACL_LOG_INFO("successfully execute aclmdlGetTensorName, realTensorName = %s", realTensorName);
        return realTensorName;
    }
    realTensorName = acl::TransTensorNameToReal(modelDesc, name);
    if (realTensorName != nullptr) {
        ACL_LOG_INFO("successfully execute aclmdlGetTensorName, realTensorName = %s", realTensorName);
        return realTensorName;
    }
    ACL_LOG_INNER_ERROR("[Get][TensorName]execute aclmdlGetTensorName failed, name[%s] is invalid.", name);
    return nullptr;
}

aclError aclRecoverAllHcclTasksImpl(int32_t deviceId)
{
    ACL_LOG_INFO("start to execute aclRecoverAllHcclTasks in device %d", deviceId);
    ge::GeExecutor executor;
    const ge::Status ret = executor.RecoverAllModel(deviceId);
    ACL_CHECK_WITH_MESSAGE_AND_RETURN(ret == ge::SUCCESS, ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret)),
                                      "call RecoverAllModel fail in deviceid %d", deviceId);
    ACL_LOG_INFO("end to execute aclRecoverAllHcclTasks in device %d", deviceId);
    return ACL_SUCCESS;
}
