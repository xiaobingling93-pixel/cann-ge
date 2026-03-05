/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../common/sampleDevice.h"
#include "../../common/sampleModel.h"
#include "../../common/utils.h"

using namespace std;

class SampleQwenLlm {
public:
    SampleQwenLlm() = default;
    ~SampleQwenLlm() = default;

    Result InitResource(const char *modelPath);
    Result Process();
    Result PrepareModel(const char *modelPath);
    void PrintInputs(size_t count);
    void PrintOutputs(size_t count, size_t lastPos);
private:
    // Device resource.
    std::shared_ptr<AclInstance> aclInstance_;
    std::shared_ptr<AclDevice> device_;
    std::shared_ptr<AclContext> context_;
    std::shared_ptr<AclStream> stream_;

    // Model.
    std::shared_ptr<AclModelDesc> modelDesc_;
    std::shared_ptr<AclLlmModelInput> modelInput_;
    std::shared_ptr<AclModelOutput> modelOutput_;
    uint32_t modelId_;
};

Result SampleQwenLlm::InitResource(const char *modelPath)
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

    return SUCCESS;
}

Result SampleQwenLlm::PrepareModel(const char *modelPath)
{
    // Load model.
    aclError ret = aclmdlLoadFromFile(modelPath, &modelId_);
    if (ret != ACL_SUCCESS) {
        return FAILED;
    }
    INFO_LOG("load model %s success.", modelPath);

    // Create ModelDesc.
    modelDesc_ = std::make_shared<AclModelDesc>(modelId_);

    return SUCCESS;
}

void SampleQwenLlm::PrintInputs(size_t count)
{
    size_t inputNum = aclmdlGetNumInputs(modelDesc_->GetModelDesc());
    for (size_t i = 0; i < inputNum && i < count; i++) {
        size_t tensorSize = aclmdlGetInputSizeByIndex(modelDesc_->GetModelDesc(), i);
        const char *tensorName = aclmdlGetInputNameByIndex(modelDesc_->GetModelDesc(), i);
        aclDataType dataType = aclmdlGetInputDataType(modelDesc_->GetModelDesc(), i);
        aclFormat format = aclmdlGetInputFormat(modelDesc_->GetModelDesc(), i);
        aclmdlIODims dims;
        aclmdlGetInputDims(modelDesc_->GetModelDesc(), i, &dims);
        std::ostringstream dimsOss;
        for (size_t d = 0; d < dims.dimCount; d++) {
            if (d > 0) {
                dimsOss << " ";
            }
            dimsOss << dims.dims[d];
        }
        INFO_LOG("  Input[%zu], tensorName=%s, size=%zu bytes, dtype=%d, format=%d, dims=%s",
                i, tensorName, tensorSize, dataType, format, dimsOss.str().c_str());
    }
}

void SampleQwenLlm::PrintOutputs(size_t count, size_t lastPos)
{
    size_t outputNum = aclmdlGetNumOutputs(modelDesc_->GetModelDesc());
    for (size_t i = 0; i < outputNum && i < count; ++i) {
        size_t tensorSize = aclmdlGetOutputSizeByIndex(modelDesc_->GetModelDesc(), i);
        const char *tensorName = aclmdlGetOutputNameByIndex(modelDesc_->GetModelDesc(), i);
        aclDataType dataType = aclmdlGetOutputDataType(modelDesc_->GetModelDesc(), i);
        aclFormat format = aclmdlGetOutputFormat(modelDesc_->GetModelDesc(), i);
        aclmdlIODims dims;
        aclmdlGetOutputDims(modelDesc_->GetModelDesc(), i, &dims);
        std::ostringstream dimsOss;
        for (size_t d = 0; d < dims.dimCount; d++) {
            if (d > 0) {
                dimsOss << " ";
            }
            dimsOss << dims.dims[d];
        }
        INFO_LOG("  Output[%zu], tensorName=%s, size=%zu bytes, dtype=%d, format=%d, dims=%s",
                i, tensorName, tensorSize, dataType, format, dimsOss.str().c_str());
    }
    // Parse the logits tensor to obtain the predicted_token_id. This sample mainly introduces
    // how to obtain the output tensor through the acl interface and calculate the predicted
    // token_id by indicating the maximum probability. In reality, an appropriate algorithm
    // needs to be selected based on the model to calculate the result.
    size_t outputSize = aclmdlGetOutputSizeByIndex(modelDesc_->GetModelDesc(), 0);
    aclDataBuffer* outputDataBuffer = aclmdlGetDatasetBuffer(modelOutput_->GetDataSet(), 0);
    void* outputBuffer = nullptr;
    if (outputDataBuffer != nullptr) {
        outputBuffer = aclGetDataBufferAddr(outputDataBuffer);
    }
    if (outputBuffer != nullptr) {
        aclDataType dataType = aclmdlGetOutputDataType(modelDesc_->GetModelDesc(), 0);
        size_t typeSize = aclDataTypeSize(dataType);
        if (typeSize == 0) {
            return;
        }
        size_t numElements = outputSize / typeSize;
        std::vector<float> logits(numElements);
        aclError copyRet = aclrtMemcpy(logits.data(), outputSize, outputBuffer, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet == ACL_SUCCESS) {
            // The length of the tokenizer table is 151936.
            size_t vocabSize = 151936;
            const float* lastLogits = logits.data() + lastPos * vocabSize;
            int maxIdx = 0;
            float maxVal = lastLogits[0];
            for (size_t j = 1; j < vocabSize; ++j) {
                float val = lastLogits[j];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            INFO_LOG("predicted_token_id: %d", maxIdx);
        }
    }
}

Result SampleQwenLlm::Process()
{
    INFO_LOG("Start to Process.");

    INFO_LOG("The first five inputs information:");
    PrintInputs(5);

    // Based on the information in inputs, assemble the input. If the accuracy of the model
    // is to be ensured, attention masks and other required inputs also need to be assembled.
    // This example focuses on the invocation of the acl interface to demonstrate how to call
    // the acl interface to execute an LLM model. To simplify the example, other inputs have
    // been ignored.
    std::vector<int64_t> inputIds;
    inputIds.push_back(14623);
    inputIds.push_back(525);
    inputIds.push_back(498);
    inputIds.push_back(30);
    // Remaining padding token.
    while (inputIds.size() < 512) {
        inputIds.push_back(151643);
    }
    std::vector<InputTensor> inputs;
    InputTensor input0{
        .data = inputIds.data(),
        .byte_size = inputIds.size() * sizeof(int64_t)
    };
    inputs.push_back(input0);
    modelInput_ = std::make_shared<AclLlmModelInput>(inputs, modelDesc_->GetModelDesc());
    modelOutput_ = std::make_shared<AclModelOutput>(modelDesc_->GetModelDesc());

    INFO_LOG("Start to execute model.");
    aclError ret = aclmdlExecute(modelId_, modelInput_->GetDataSet(), modelOutput_->GetDataSet());
    if (ret != SUCCESS) {
        return FAILED;
    }

    // This example merely prints the information of the output tensor. In a real usage scenario,
    // the predicted token ID obtained from the logits tensor during the model execution needs
    // to be calculated and then the token characters need to be parsed.
    INFO_LOG("The first five outputs information and the predicted token id:");
    // Print the fourth token ID of the prediction.
    PrintOutputs(5, 4);
    return SUCCESS;
}

int main()
{
    INFO_LOG("SAMPLE start to execute.");

    // To better demonstrate the core usage of the acl interface, the sample encapsulates
    // the resource management within SampleQwenLlm, and the resource
    // release relies on the destructor of SampleQwenLlm. Therefore,
    // using curly braces to define the scope ensures that it is destructed and resources
    // are released before the process exits.
    {
        SampleQwenLlm sampleQwen;
        const char *aclConfigPath = "../src/acl.json";
        Result ret = sampleQwen.InitResource(aclConfigPath);
        if (ret != SUCCESS) {
            ERROR_LOG("SAMPLE NOT PASSED: sample init resource failed.");
            return FAILED;
        }

        ret = sampleQwen.PrepareModel("../model/qwen.om");
        if (ret != SUCCESS) {
            ERROR_LOG("SAMPLE NOT PASSED: sample prepare model failed.");
            return FAILED;
        }

        ret = sampleQwen.Process();
        if (ret != SUCCESS) {
            ERROR_LOG("SAMPLE NOT PASSED: sample process failed.");
            return FAILED;
        }
    }

    INFO_LOG("SAMPLE PASSED.");
    return SUCCESS;
}