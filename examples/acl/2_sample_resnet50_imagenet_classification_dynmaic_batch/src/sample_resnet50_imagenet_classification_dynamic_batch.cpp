/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "../../common/sampleDevice.h"
#include "../../common/sampleModel.h"
#include "../../common/utils.h"

using namespace std;
bool g_isDevice = false;

class SampleRes50ImagenetClassification {
public:
    SampleRes50ImagenetClassification() = default;
    ~SampleRes50ImagenetClassification() = default;

    Result InitResource(const char *modelPath);
    Result Process(uint64_t batchSize);
    Result PrepareModel(const char *modelPath);
    void OutputModelResult(uint64_t batchSize) const;
private:
    // Device resource.
    std::shared_ptr<AclInstance> aclInstance_;
    std::shared_ptr<AclDevice> device_;
    std::shared_ptr<AclContext> context_;
    std::shared_ptr<AclStream> stream_;

    // Model.
    std::shared_ptr<AclModelWeight> modelWeight_;
    std::shared_ptr<AclModelWork> modelWork_;
    std::shared_ptr<AclModelDesc> modelDesc_;
    std::shared_ptr<AclModelInput> modelInput_;
    std::shared_ptr<AclModelOutput> modelOutput_;
    uint32_t modelId_{0U};
};

Result SampleRes50ImagenetClassification::InitResource(const char *modelPath)
{
    // ACL init.
    aclInstance_ = std::make_shared<AclInstance>(modelPath);
    INFO_LOG("acl init success");

    // Set device.
    device_ = std::make_shared<AclDevice>(0);
    INFO_LOG("set device success");

    // Create context (set current).
    context_ = std::make_shared<AclContext>(0);
    INFO_LOG("create context success");

    // Create stream.
    stream_ = std::make_shared<AclStream>();
    INFO_LOG("create stream success");

    // Get run mode
    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    auto ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_SUCCESS) {
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    return SUCCESS;
}

Result SampleRes50ImagenetClassification::PrepareModel(const char *modelPath)
{
    // Load model.
    size_t modelWorkSize, modelWeightSize;
    aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize, &modelWeightSize);
    if (ret != ACL_SUCCESS) {
        return FAILED;
    }
    modelWork_ = std::make_shared<AclModelWork>(modelWorkSize);
    modelWeight_ = std::make_shared<AclModelWeight>(modelWeightSize);

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWork_->GetModelWorkPtr(),
                        modelWork_->GetModelWorkSize(), modelWeight_->GetModelWeightPrt(), modelWeight_->GetModelWeightSize());
    if (ret != ACL_SUCCESS) {
        return FAILED;
    }
    INFO_LOG("load model %s success.", modelPath);

    // Create ModelDesc.
    modelDesc_ = std::make_shared<AclModelDesc>(modelId_);

    return SUCCESS;
}

void SampleRes50ImagenetClassification::OutputModelResult(uint64_t batchSize) const {
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(modelOutput_->GetDataSet()); ++i) {
        // Get model output data
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(modelOutput_->GetDataSet(), i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

        void *outHostData = nullptr;
        aclError ret = ACL_SUCCESS;
        float *outData = nullptr;
        if (!g_isDevice) {
            ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                    len, static_cast<int32_t>(ret));
                return;
            }

            // If app is running in host, need copy model output data from device to host
            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                (void)aclrtFreeHost(outHostData);
                return;
            }

            outData = static_cast<float*>(outHostData);
        } else {
            outData = static_cast<float*>(data);
        }

        for (size_t idx = 0; idx < batchSize; idx++) {
            INFO_LOG("Result of picture %zu:", idx + 1U);

            map<float, unsigned int, greater<>> resultMap;
            // resnet50 model output shape is [batchSize, 1000]
            for (unsigned int j = 0; j < 1000U; ++j) {
                resultMap[*outData] = j;
                outData++;
            }

            int cnt = 0;
            for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
                // Print top 5
                if (++cnt > 5) {
                    break;
                }

                INFO_LOG("top %d: index[%d] value[%lf]", cnt, it->second, it->first);
            }
        }

        if (!g_isDevice) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                return;
            }
        }
    }

    INFO_LOG("output data success");
}

Result SampleRes50ImagenetClassification::Process(uint64_t batchSize)
{
    std::vector<std::string> testFiles = {
        "../data/dog1_1024_683.bin",
       "../data/dog2_1024_683.bin"
    };

    void *picDevBuffer = nullptr;
    size_t devBufferSize = 0U;

    // Copy image data to device buffer
    const auto ret = Utils::MemcpyFilesToDeviceBuffer(testFiles, &picDevBuffer, devBufferSize, batchSize, g_isDevice);
    if (ret != SUCCESS) {
        aclrtFree(picDevBuffer);
        ERROR_LOG("memcpy device buffer failed");
        return FAILED;
    }

    modelInput_ = std::make_shared<AclModelInput>(picDevBuffer, devBufferSize, modelDesc_->GetModelDesc());
    modelOutput_ = std::make_shared<AclModelOutput>(modelDesc_->GetModelDesc());

    // Dynamic input default name is ACL_DYNAMIC_TENSOR_NAME
    size_t dynamic_input_index = 0U;
    aclError aclRet = aclmdlGetInputIndexByName(modelDesc_->GetModelDesc(), ACL_DYNAMIC_TENSOR_NAME, &dynamic_input_index);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("get input index failed");
        aclrtFree(picDevBuffer);
        return FAILED;
    }

    // Need to set batch size before execution
    aclRet = aclmdlSetDynamicBatchSize(modelId_, modelInput_->GetDataSet(), dynamic_input_index, batchSize);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("set dynamic batchSize failed");
        aclrtFree(picDevBuffer);
        return FAILED;
    }

    aclRet = aclmdlExecute(modelId_, modelInput_->GetDataSet(), modelOutput_->GetDataSet());
    if (aclRet != ACL_SUCCESS) {
        return FAILED;
    }
    INFO_LOG("model execute success");

    // Print the top 5 confidence values with indexes.
    // Use function [DumpModelOutputResult] if you want to dump results to file in the current directory.
    OutputModelResult(batchSize);

    aclrtFree(picDevBuffer);
    return SUCCESS;
}

int main()
{
    INFO_LOG("SAMPLE start to execute.");

    // To better demonstrate the core usage of the acl interface, the sample encapsulates
    // the resource management within SampleRes50ImagenetClassification, and the resource
    // release relies on the destructor of SampleRes50ImagenetClassification. Therefore,
    // using curly braces to define the scope ensures that it is destructed and resources
    // are released before the process exits.
    {
        SampleRes50ImagenetClassification sampleRes50;
        const char *aclConfigPath = "../src/acl.json";
        Result ret = sampleRes50.InitResource(aclConfigPath);
        if (ret != SUCCESS) {
            ERROR_LOG("SAMPLE NOT PASSED: sample init resource failed.");
            return FAILED;
        }

        ret = sampleRes50.PrepareModel("../model/resnet50_dynamic_batch.om");
        if (ret != SUCCESS) {
            ERROR_LOG("SAMPLE NOT PASSED: sample prepare model failed.");
            return FAILED;
        }

        constexpr uint64_t batchSize = 2U;
        ret = sampleRes50.Process(batchSize);
        if (ret != SUCCESS) {
            ERROR_LOG("SAMPLE NOT PASSED: sample process failed.");
            return FAILED;
        }
    }

    INFO_LOG("SAMPLE PASSED.");
    return SUCCESS;
}