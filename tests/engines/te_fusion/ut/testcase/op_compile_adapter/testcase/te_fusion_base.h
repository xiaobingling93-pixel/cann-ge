/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_BASE_H
#define TE_FUSION_BASE_H

#include "graph/node.h"
#include "graph/op_desc.h"
#include "tensor_engine/fusion_types.h"
#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"

#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"

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
                       vector<GeTensorDesc> &outputDescList);
void FillTensorDesc(GeTensorDesc &tensorDesc,
                    vector<int64_t> shapeDims,
                    DataType dataType);

void CreateFile(std::string &realPath);

void CreateDir(const std::string &kernelMetaTempDir);

bool GetJsonFromJsonFile(const std::string &realPath, nlohmann::json &jsonInfo);

bool WriteToJsonFile(const std::string &realPath, nlohmann::json &jsonInfo);

void AddTensorToOpDesc(bool isInput, std::string name, vector<int64_t> shape, Format format,  DataType data_type,
                       std::vector<std::pair<int64_t, int64_t>> &range, OpDescPtr &opDescPtr);

void AddOpParamToTbeOpInfo(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                           std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfo &op_info);

void AddOpParamToTbeOpInfoV3(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                           const std::vector<float> &constValue, bool isInput, TbeOpInfo &op_info);

void AddOpParamToTbeOpInfoPtr(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                              std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfoPtr op_info,
                              te::TensorType tType);

void ReplaceFileContent(std::string fileName, std::string oldStr, std::string newStr);

std::string GetCodeDir();
#endif
