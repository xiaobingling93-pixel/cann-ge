/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sample_dynamic_batch.h"
#include <map>
#include <vector>

int main()
{
    INFO_LOG("SAMPLE start to execute.");

    const std::map<ge::AscendString, ge::AscendString> options{
            {"ge.graphRunMode", "0"}
    };
    std::unique_ptr<SampleDynamicBatch> sampleDynamicBatchPtr = nullptr;
    try {
        sampleDynamicBatchPtr = std::make_unique<SampleDynamicBatch>(options);
    } catch (std::runtime_error &e) {
        ERROR_LOG("SampleDynamicBatch creation failed");
        return FAILED;
    }
    if (!sampleDynamicBatchPtr) {
        ERROR_LOG("SampleDynamicBatch creation failed");
        return FAILED;
    }

    // resnet50_Opset16.onnx is static model, input node is x:[1, 3, 224, 224]
    const std::string modelPath = "../model/resnet50_Opset16.onnx";
    // parse onnx model and build ge graph
    auto ret = sampleDynamicBatchPtr->ParseModelAndBuildGraph(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("Parse onnx model failed, model:%s", modelPath.c_str());
        return FAILED;
    }
    INFO_LOG("Parse model %s success", modelPath.c_str());

    // enable dynamic batch
    const std::map<ge::AscendString, ge::AscendString> graph_options{
        {"ge.inputShape", "x:-1,3,224,224"},
        {"ge.dynamicDims", "1;2;4;8"},
        {"ge.dynamicNodeType", "1"}
    };
    // set specific input tensor
    std::vector<ge::Tensor> input_tensors;
    const std::initializer_list<int64_t> dims{2, 3, 224, 224};
    const ge::Shape input_shape(dims); // batchSize:2
    const ge::Tensor input_tensor(ge::TensorDesc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT));
    input_tensors.push_back(input_tensor);

    ret = sampleDynamicBatchPtr->CompileGraph(graph_options, input_tensors);
    if (ret != SUCCESS) {
        ERROR_LOG("SampleDynamicBatch compile graph failed");
        return FAILED;
    }

    const std::vector<std::string> testFiles = {"../data/dog1_1024_683.bin","../data/dog2_1024_683.bin"};
    ret = sampleDynamicBatchPtr->Process(testFiles, dims);
    if (ret != SUCCESS) {
        ERROR_LOG("SampleDynamicBatch process graph failed");
        return FAILED;
    }

    // print output result
    sampleDynamicBatchPtr->OutputModelResult();

    INFO_LOG("SAMPLE PASSED.");
    return SUCCESS;
}
