/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define protected public

#include <gtest/gtest.h>

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sys/file.h>
#include <vector>
#include <string>

#define private public

#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"
#include "python_adapter/python_api_call.h"
#include "common/common_utils.h"

#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "te_fusion_base.h"

using namespace ge;
using namespace te;

OpDescPtr CreateOpDesc(string name,
                       string type,
                       uint32_t id,
                       string backend,
                       uint32_t scopeId,
                       vector<string> &inputNameList,
                       uint32_t DescListSize,
                       vector<GeTensorDesc> &inputDescList,
                       vector<GeTensorDesc> &outputDescList) {
    OpDescPtr newOpDesc = std::make_shared<OpDesc>();

    newOpDesc->SetName(name);
    newOpDesc->SetType(type);
    newOpDesc->SetId(id);
    AttrUtils::SetStr(newOpDesc, ATTR_NAME_SESSION_GRAPH_ID, "session_id");
    AttrUtils::SetStr(newOpDesc, TVM_ATTR_NAME_MAGIC, backend);
    AttrUtils::SetInt(newOpDesc, ATTR_NAME_FUSION_SCOPE, scopeId);
    AttrUtils::SetStr(newOpDesc, "_pattern", "ElemWise");

    newOpDesc->SetInputName(inputNameList);

    for (uint32_t loop = 0; loop < inputDescList.size(); loop++) {
        newOpDesc->AddInputDesc(inputDescList[loop]);
    }

    for (uint32_t loop = 0; loop < outputDescList.size(); loop++) {
        newOpDesc->AddOutputDesc(outputDescList[loop]);
    }

    return newOpDesc;
}

void FillTensorDesc(GeTensorDesc &tensorDesc,
                    vector<int64_t> shapeDims,
                    DataType dataType)
{
    GeShape newShape(shapeDims);
    tensorDesc.SetShape(newShape);
    tensorDesc.SetDataType(dataType);
}

void CreateFile(std::string &realPath)
{
    FILE* fp = fopen(realPath.c_str(), "a+"); // create file first time
    if (fp == nullptr) {
        printf("Open file[%s] failed.", realPath.c_str());
        return;
    }
    fclose(fp);
}

void CreateDir(const std::string &kernelMetaTempDir)
{
    std::string real_path = te::fusion::RealPath(kernelMetaTempDir);
    if (real_path.empty()) {
        std::string command = "mkdir -p " + kernelMetaTempDir;
        system((char*) command.c_str());
    }
    return;
}

void ReplaceFileContent(std::string fileName, std::string oldStr, std::string newStr)
{
    std::string line;
    std::vector<std::string> lines;
    std::ifstream infile(fileName);
    if (!infile.is_open()){
        printf("file %s is not open", fileName.c_str());
    }
    std::string content((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();

    std::ofstream outfile(fileName, std::ios::out);
    size_t pos = content.find(oldStr);
    if (pos != string::npos) {
        content.replace(pos, oldStr.length(), newStr);
    }
    outfile << content.c_str();
    outfile.close();
}

bool GetJsonFromJsonFile(const std::string &realPath, nlohmann::json &jsonInfo)
{
    std::ifstream ifs(realPath);
    try {
        if (!ifs.is_open()) {
            printf("[GetJsonFromJsonFile] Open %s failed, file is already open.", realPath.c_str());
            return false;
        }
        ifs >> jsonInfo;
        ifs.close();
    } catch (const std::exception &e) {
        printf("Failed to convert file[%s] to Json. Error message is %s.", realPath.c_str(), e.what());
        ifs.close();
        return false;
    }
    return true;
}

bool WriteToJsonFile(const std::string &realPath, nlohmann::json &jsonInfo)
{
    std::ofstream ofs(realPath);
    try {
        if (!ofs.is_open()) {
            printf("[WriteToJsonFile] Open %s failed, file is already open.", realPath.c_str());
            return false;
        }
        ofs << jsonInfo.dump(4);
        ofs.close();
    } catch (const std::exception &e) {
        printf("Failed to convert file[%s] to Json. Error message is %s.", realPath.c_str(), e.what());
        ofs.close();
        return false;
    }
    return true;
}

void AddTensorToOpDesc(bool isInput, std::string name, vector<int64_t> shape, Format format,  DataType data_type,
                       std::vector<std::pair<int64_t, int64_t>> &range, OpDescPtr &opDescPtr)
{
    ge::GeTensorDesc tensorDesc;
    tensorDesc.SetShape(ge::GeShape(shape));
    tensorDesc.SetOriginShape(ge::GeShape(shape));
    tensorDesc.SetShapeRange(range);
    tensorDesc.SetOriginShapeRange(range);
    tensorDesc.SetDataType(data_type);
    tensorDesc.SetOriginDataType(data_type);
    tensorDesc.SetFormat(format);
    tensorDesc.SetOriginFormat(format);
    if (isInput) {
        opDescPtr->AddInputDesc(name, tensorDesc);
    } else {
        opDescPtr->AddOutputDesc(name, tensorDesc);
    }
}

void AddOpParamToTbeOpInfo(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                           std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfo &op_info)
{
    TbeOpParam opParam;
    std::vector<TbeOpTensor> tensors;
    TbeOpTensor tensor(name, shape, dtype, format);
    tensor.SetShapeRange(range);
    tensor.SetOriginShape(shape);
    tensor.SetOriginShapeRange(range);
    tensor.SetOriginFormat(format);

    if (name == "input12") {
        opParam.SetValueDepend(VALUE_DEPEND_OPTIONAL);
        TbeAttrValue attrValue("input12", (int32_t)1);
        tensor.SetConstValue(attrValue);
    }
    if (name == "input13") {
        opParam.SetValueDepend(VALUE_DEPEND_REQUIRED);
        TbeAttrValue attrValue("input13", (int32_t)1);
        tensor.SetConstValue(attrValue);
    }
    tensors.push_back(tensor);
    opParam.SetTensors(tensors);

    if (isInput) {
        op_info.AddInput(opParam);
    } else {
        op_info.AddOutput(opParam);
    }
}

void AddOpParamToTbeOpInfoV3(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                           const std::vector<float> &constValue, bool isInput, TbeOpInfo &op_info)
{
    TbeOpParam opParam;
    std::vector<TbeOpTensor> tensors;
    TbeOpTensor tensor(name, shape, dtype, format);
    tensor.SetOriginShape(shape);
    tensor.SetOriginFormat(format);

    if (! constValue.empty()) {
        opParam.SetValueDepend(VALUE_DEPEND_OPTIONAL);
        TbeAttrValue attrValue(name, constValue);
        tensor.SetConstValue(attrValue);
    }

    tensors.push_back(tensor);
    opParam.SetTensors(tensors);

    if (isInput) {
        op_info.AddInput(opParam);
    } else {
        op_info.AddOutput(opParam);
    }
}

void AddOpParamToTbeOpInfoPtr(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                              std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfoPtr op_info,
                              te::TensorType tType)
{
    TbeOpParam opParam;
    std::vector<TbeOpTensor> tensors;
    TbeOpTensor tensor(name, shape, dtype, format);
    tensor.SetShapeRange(range);
    tensor.SetOriginShape(shape);
    tensor.SetOriginShapeRange(range);
    tensor.SetOriginFormat(format);
    tensors.push_back(tensor);
    opParam.SetTensors(tensors);
    opParam.SetType(tType);

    if (isInput) {
        op_info->AddInput(opParam);
    } else {
        op_info->AddOutput(opParam);
    }
}

std::string GetCodeDir() {
  static std::string gCachedCodeDir;
  if (gCachedCodeDir.empty()) {
    const char *code_path_ptr = std::getenv("AIR_CODE_DIR");
    if (code_path_ptr != nullptr) {
      gCachedCodeDir = string(code_path_ptr);
    }
  }
  return gCachedCodeDir;
}