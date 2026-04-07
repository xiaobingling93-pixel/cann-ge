/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_engine/fusion_api.h"
#include "fe_llt_utils.h"
namespace te {

bool PreBuildTbeOp(TbeOpInfo &info) {
    info.SetPattern("mocker");
    return true;
}

bool TeFusion(std::vector<ge::NodePtr> teGraphNode, ge::NodePtr output_node)
{
    const std::string key_to_attr_json_file_path = "json_file_path";
    std::string json_file_path = fe::GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.json";
    ge::AttrUtils::SetStr(output_node->GetOpDesc(), key_to_attr_json_file_path, json_file_path);
    return true;
}

bool TeFusionEnd()
{
    return true;
}

}
