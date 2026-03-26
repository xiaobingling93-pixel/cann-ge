/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assemble_json/te_json_assemble.h"
#include <map>
#include <deque>
#include <algorithm>
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_config_info.h"
#include "common/te_context_utils.h"
#include "assemble_json/te_attr_utils.h"
#include "assemble_json/te_op_custom_utils.h"
#include "assemble_json/te_json_utils.h"
#include "compile/te_compile_task_cache.h"
#include "python_adapter/python_api_call.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/anchor.h"
#include "graph/tuning_utils.h"
#include "graph/utils/type_utils.h"
#include "cache/te_cache_manager.h"

namespace te {
namespace fusion {
using nlohmann::json;
using namespace ge;
namespace {
std::mutex gKernelNameMutex;
std::map<std::string, int> gKernelNameHash;

bool CompareNode(const ge::Node *NodeA, const ge::Node *NodeB)
{
    auto idA = NodeA->GetOpDesc()->GetId();
    auto idB = NodeB->GetOpDesc()->GetId();
    return idA < idB;
}

void TransDtypeToString(const ge::DataType &dtype, string &dtypeString) {
  dtypeString = ge::TypeUtils::DataTypeToSerialString(dtype);
  auto indx = dtypeString.find("_");
  if (indx == string::npos) {
    dtypeString = "";
    return;
  }
  dtypeString = dtypeString.substr(indx + 1);
  transform(dtypeString.begin(), dtypeString.end(), dtypeString.begin(), ::tolower);
}

const std::map<DataType, std::string> DTYPE_STR_MAP {{DT_FLOAT, "float32"},
                                                     {DT_DOUBLE, "float64"}};
/**
 * @brief: transform GE data type int to TBE op data type string
       enum DataType
        DT_FLOAT = 0,           // float type
        DT_FLOAT16 = 1,         // fp16 type
        DT_INT8 = 2,            // int8 type
        DT_INT32 = 3,           // int32 type
        DT_UINT8 = 4,           // uint8 type
        DT_INT16 = 6,           // int16 type
        DT_UINT16 = 7,          // uint16 type
        DT_UINT32 = 8,          // unsigned int32
        DT_INT64 = 9,           // int64 type
        DT_UINT64 = 10,         // unsigned int64
        DT_DOUBLE = 11,         // double type
        DT_BOOL = 12,           // bool type
 */
std::string GetDataTypeStr(const DataType dataType)
{
    std::string strType;
    auto iter = DTYPE_STR_MAP.find(dataType);
    if (iter != DTYPE_STR_MAP.end()) {
        strType = iter->second;
    } else {
       TransDtypeToString(dataType, strType);
    }
    return strType;
}
}

bool TeJsonAssemble::GenerateJsonAndKernelName(const std::vector<Node *> &nodes,
                                               const bool isAllowRepeated, std::string &jsonStr,
                                               std::string &kernelName)
{
    nlohmann::json jsonData;
    if (!GenerateJsonByNodes(nodes, jsonData)) {
        return false;
    }

    TE_DBGLOG("Try to generate unique json");
    std::string uniqueJsonStr;
    GetUniqueJsonStr(jsonData, uniqueJsonStr);
    if (uniqueJsonStr.empty()) {
        REPORT_TE_INNER_ERROR("Identity json[uniqueJsonStr] of the first node [%s] is empty.",
                              nodes[0]->GetName().c_str());
        return false;
    }

    TE_DBGLOGF("Unique json str is [%s].", uniqueJsonStr.c_str());
    if (kernelName.empty()) {
        if (nodes.size() == 1) {
            if (!GenerateKernelName("te_" + nodes.at(0)->GetType(), uniqueJsonStr, isAllowRepeated, kernelName)) {
                return false;
            }
        } else {
            std::string fusionName;
            TeJsonUtils::GenerateFusionName(jsonData, fusionName);
            if (!GenerateKernelName("te_fused_op" + fusionName, uniqueJsonStr, isAllowRepeated, kernelName)) {
                return false;
            }
        }
    }
    jsonData["fusion_op_name"] = kernelName;
    jsonStr = jsonData.dump();
    TE_DBGLOG("Generate json and kernel name [%s] successfully", kernelName.c_str());
    return true;
}

bool TeJsonAssemble::GeneratePrebuildJsonAndKernelName(const std::vector<ge::Node *> &nodes, std::string &jsonStr,
                                                       std::string &kernelName)
{
    nlohmann::json jsonData;
    if (!GenerateJsonByNodes(nodes, jsonData)) {
        return false;
    }

    TE_DBGLOG("Try to generate unique json.");
    std::string uniqueJsonStr;
    GetUniqueJsonStr(jsonData, uniqueJsonStr);
    if (uniqueJsonStr.empty()) {
        REPORT_TE_INNER_ERROR("Identity json[uniqueJsonStr] of the first node [%s] is empty.",
                              nodes[0]->GetName().c_str());
        return false;
    }
    if (kernelName.empty()) {
        if (!GeneratePreBuildKernelName("te_" + nodes.at(0)->GetType(), uniqueJsonStr, kernelName)) {
            return false;
        }
    }
    jsonData["fusion_op_name"] = kernelName;
    jsonStr = jsonData.dump();
    TE_DBGLOG("Generated op [%s] json data successfully.", kernelName.c_str());
    return true;
}

bool TeJsonAssemble::GenerateJsonByNodes(const std::vector<ge::Node *> &teGraphNode, nlohmann::json &jsonData)
{
    TE_FUSION_CHECK((teGraphNode.empty()), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Graph node list is empty.");
        return false;
    });

    CheckNodeInfo checkNodeInfo;
    std::vector<std::string> verifyOpTypeList;
    // check input params
    TE_FUSION_CHECK(!(CheckNodeList(teGraphNode, checkNodeInfo, verifyOpTypeList)), {
        REPORT_TE_INNER_ERROR("Failed to check tbe nodes in generating fusion json.");
        return false;
    });

    json opComputeList;
    std::map<string, int> allInputNames;
    std::unordered_set<ge::Node *> allNodes;
    bool isOnlyFusionCheck = false;
    bool isAutoFusionPattern = false;
    std::string enableSuperkernelPlus;
    for (auto &ele : teGraphNode) {
        allNodes.emplace(ele);
    }
    size_t nodeListIdx = 0;
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec;
    for (auto &currentNode : teGraphNode) {
        std::string keyName;
        bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(currentNode, keyName);
        TE_FUSION_CHECK(!bres || keyName.empty(), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to get node key name.");
            return false;
        });
        ConstTbeOpInfoPtr tbeOpInfo = nullptr;
        if (teGraphNode.size() == 1) {
            tbeOpInfo = TbeOpInfoCache::Instance().GetSecondTbeOpInfo(keyName);
        }
        if (tbeOpInfo == nullptr) {
            tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfo(keyName);
        }
        TE_FUSION_CHECK(tbeOpInfo == nullptr, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to check op is prebuild or not, op name is [%s].", keyName.c_str());
            return false;
        });
        tbeOpInfoVec.push_back(tbeOpInfo);
        TE_DBGLOG("Process node begin, current node name is [%s]", keyName.c_str());

        // record currentNode info in json
        json currentNodeJson;
        // get op desc
        auto currentNodeOpDesc = currentNode->GetOpDesc();
        // get basic params from currentNodeOpDesc
        currentNodeJson["name"] = keyName;
        currentNodeJson["type"] = currentNodeOpDesc->GetType();
        currentNodeJson["id"] = currentNodeOpDesc->GetId();
        bool isDynamicImpl = false;
        (void)AttrUtils::GetBool(currentNodeOpDesc, ATTR_NAME_IS_OP_DYNAMIC_IMPL, isDynamicImpl);
        currentNodeJson["is_dynamic_impl"] = isDynamicImpl;
        bool isCustomOp = false;
        (void)ge::AttrUtils::GetBool(currentNodeOpDesc, "_is_custom_op", isCustomOp);
        currentNodeJson["_is_custom_op"] = isCustomOp;
        bool isDynamicShape = false;
        (void)AttrUtils::GetBool(currentNodeOpDesc, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, isDynamicShape);
        currentNodeJson["dyn_flag"] = isDynamicShape;
        TE_DBGLOG("Op: %s, is_dy flag: %d, dyn_flag: %d.", currentNode->GetName().c_str(), isDynamicImpl, isDynamicShape);

        std::vector<std::string> originOpNames;
        if (!ge::AttrUtils::GetListStr(currentNodeOpDesc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, originOpNames)) {
            originOpNames.push_back(currentNode->GetName());
        }
        currentNodeJson["ori_name"] = originOpNames;

        std::string pattern = TeCompileTaskCache::Instance().GetOpPattern(keyName);
        string opPattern;
        if (ge::AttrUtils::GetStr(currentNodeOpDesc, ATTR_NAME_OP_PATTERN, opPattern)) {
            TE_DBGLOG("Op pattern attr of op[%s] is [%s]", currentNodeOpDesc->GetName().c_str(), opPattern.c_str());
            pattern = opPattern;
        }
        TE_DBGLOG("OpNode: %s, Pattern: %s.", keyName.c_str(), pattern.c_str());
        currentNodeJson["pattern"] = pattern;
        currentNodeJson["int64mode"] = tbeOpInfo->GetFlagUseInt64();
        currentNodeJson["op_impl_mode"] = tbeOpInfo->GetOpImplMode();
        currentNodeJson["extra_params"] = tbeOpInfo->GetExtraParams();
        if (!tbeOpInfo->GetExtraParams().empty()) {
            TE_DBGLOG("OpNode: %s has extra_params.", keyName.c_str());
        }
        std::string hasedExtraParams = tbeOpInfo->GetHashedExtraParams();
        if (!hasedExtraParams.empty()) {
            currentNodeJson["hashed_extra_params"] = hasedExtraParams;
            TE_DBGLOG("OpNode: %s, HashedExtraParams: %s.", keyName.c_str(), hasedExtraParams.c_str());
        }

        if (tbeOpInfo->GetUBSpaceSize() != -1) {
            currentNodeJson["UBSpaceSize"] = tbeOpInfo->GetUBSpaceSize();
            TE_DBGLOG("OpNode: %s, UBSpaceSize: %ld", keyName.c_str(), tbeOpInfo->GetUBSpaceSize());
        }
        int64_t spCnt = -1;
        (void)ge::AttrUtils::GetInt(currentNode->GetOpDesc(), ASCENDC_SPK_CNT, spCnt);
        int64_t spSubId = -1;
        (void)ge::AttrUtils::GetInt(currentNode->GetOpDesc(), ASCENDC_SPK_SUB_ID, spSubId);
        std::string subOpLoc;
        if (spCnt != -1 && spSubId != -1 && GetSubOpLoc(spCnt, spSubId, subOpLoc)) {
            currentNodeJson["super_kernel_sub_loc"] = subOpLoc;
            std::string spOptions;
            (void)ge::AttrUtils::GetStr(currentNodeOpDesc, SPK_OPTIONS, spOptions);
            currentNodeJson[SPK_OPTIONS] = spOptions;
            currentNodeJson[ENABLE_SPK] = true;
            TE_DBGLOG("OpNode: %s, spk_options: %s, sp_cnt: %d, sp_subid: %d.",
                      keyName.c_str(), spOptions.c_str(), spCnt, spSubId);
        }
        // get input desc json
        json inputDescJson;
        InOutToJsonParam inputPara(keyName, nodeListIdx, verifyOpTypeList, tbeOpInfo, currentNode);
        bres = TeJsonAssemble::GenBuildinIndescJson(inputPara, allNodes, inputDescJson, allInputNames, teGraphNode);
        TE_FUSION_CHECK(!bres, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to get builtin op input desc json, op name is [%s].", keyName.c_str());
            return false;
        });
        currentNodeJson["input_desc"] = inputDescJson;
        InOutToJsonParam outputJsonPara(keyName, nodeListIdx, verifyOpTypeList, tbeOpInfo, currentNode);
        TeJsonAssemble::GenBuildinOutdescJson(outputJsonPara, allNodes, currentNodeJson, teGraphNode);
        TeJsonAssemble::GenBuildinAttrsAndModuleJson(tbeOpInfo, currentNodeJson);
        // get func_name from TE API
        TE_DBGLOG("Start get func_name: OpName -> %s", keyName.c_str());
        std::string currentNodeFuncName;
        tbeOpInfo->GetFuncName(currentNodeFuncName);
        currentNodeJson["func_name"] = currentNodeFuncName;

        if (!isOnlyFusionCheck) {
            (void)AttrUtils::GetBool(currentNodeOpDesc, ONLY_FUSION_CHECK, isOnlyFusionCheck);
        }
        if (!isAutoFusionPattern) {
            (void)AttrUtils::GetBool(currentNodeOpDesc, AUTO_FUSION_PATTERN, isAutoFusionPattern);
        }
        if (enableSuperkernelPlus.empty()) {
            (void)tbeOpInfo->GetOption("enable_superkernel_plus", enableSuperkernelPlus);
        }

        opComputeList.push_back(currentNodeJson);
        TE_DBGLOG("Process node [%s] succeeded", keyName.c_str());
        ++nodeListIdx;
    }

    // soc info
    nlohmann::json socInfoJson;
    GenerateSocInfoJson(tbeOpInfoVec, socInfoJson);
    jsonData["SocInfo"] = socInfoJson;

    // write fusion op name
    jsonData["l1_size"] = PythonApiCall::Instance().GetL1SpaceSize();
    jsonData["scope_id"] = checkNodeInfo.scopeId;
    jsonData["graph_name"] = checkNodeInfo.graphName;
    jsonData["only_fusion_check"] = isOnlyFusionCheck;
    jsonData["auto_fusion_pattern"] = isAutoFusionPattern;
    jsonData["enable_superkernel_plus"] = enableSuperkernelPlus;
    if (TeContextUtils::GetBuildMode() == ge::BUILD_MODE_OPAT_RESULT) {
        jsonData[ge::BUILD_MODE] = ge::BUILD_MODE_OPAT_RESULT;
    }

    std::string opBuildOptions;
    (void)ge::AttrUtils::GetStr(teGraphNode[0]->GetOpDesc(), FUSION_OP_BUILD_OPTIONS, opBuildOptions);
    jsonData["op_build_options"] = opBuildOptions;

    // write and combine op_list: opInputList + opComputeList
    if (teGraphNode.size() > 1) {
        json dataDescJson;
        bool genRes = GenDatadescJson(teGraphNode, dataDescJson);
        TE_FUSION_CHECK(!genRes, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to generate data desc json.");
            return false;
        });
        for (size_t idx = 0; idx < dataDescJson.size(); idx++) {
            jsonData["op_list"].push_back(dataDescJson[idx]);
        }
    }

    for (size_t idx = 0; idx < opComputeList.size(); idx++) {
        jsonData["op_list"].push_back(opComputeList[idx]);
    }

    // check if output of node has muliple reference, only remain one.
    FilterOutputMultipleReference(jsonData);

    // check if output of node has optional output, fill with null if do not has data.
    FillOptionalOutputWithNull(teGraphNode, jsonData);
    return true;
}

bool TeJsonAssemble::GenerateKernelName(const std::string &prefix, const std::string &uniqueJsonStr,
                                        const bool isAllowRepeatedHash, std::string &kernelName)
{
    std::string kernelHash;
    if (!PythonApiCall::Instance().GenerateStrSha256HashValue(uniqueJsonStr, kernelHash)) {
        TE_WARNLOG("Failed to generate the kernelHash.");
        return false;
    };
    TE_DBGLOG("Kernel hash is [%s]", kernelHash.c_str());

    if (!isAllowRepeatedHash) {
        GetUnrepeatedHash(kernelHash);
    }
    kernelName = prefix + "_" + kernelHash;
    std::transform(kernelName.begin(), kernelName.end(), kernelName.begin(),
                   [](char c) { return c == '-' ? '_' : std::tolower(c); });
    TE_DBGLOG("Kernel for [%s] is [%s].", prefix.c_str(), kernelName.c_str());
    return true;
}

bool TeJsonAssemble::GeneratePreBuildKernelName(const std::string &prefix, const std::string &uniqueJsonStr,
                                                std::string &kernelName)
{
    std::string kernelHash;
    if (!PythonApiCall::Instance().GenerateStrSha256HashValue(uniqueJsonStr, kernelHash)) {
        TE_WARNLOG("Failed to generate the kernelHash.");
        return false;
    };
    TE_DBGLOG("Kernel hash is [%s].", kernelHash.c_str());

    kernelName = prefix + "_" + kernelHash + "_pre";
    std::transform(kernelName.begin(), kernelName.end(), kernelName.begin(),
                   [](char c) { return c == '-' ? '_' : std::tolower(c); });
    TE_DBGLOG("Kernel for [%s] is [%s]", prefix.c_str(), kernelName.c_str());
    return true;
}

void TeJsonAssemble::GetUnrepeatedHash(std::string &kernelHash)
{
    if (kernelHash.empty()) {
        return;
    }
    // op tune, one op may be compiled for a few times, each time may need a different kernel name
    // This is not thread safe, must be run under a mutex.
    std::lock_guard<std::mutex> lock_guard(gKernelNameMutex);
    auto res = gKernelNameHash.emplace(kernelHash, 0);
    if (!res.second) {
        // kernelHash is already existed
        res.first->second += 1;
    }
    kernelHash += "_" + std::to_string(res.first->second);
}

bool TeJsonAssemble::GenBuildinIndescJson(const InOutToJsonParam &inputPara,
                                          const std::unordered_set<ge::Node *> &allNodes,
                                          nlohmann::json &opJson, std::map<string, int> &allInputNames,
                                          const std::vector<ge::Node *> &teGraphNode)
{
    InputDescJsonParam inputs_json_param(inputPara.keyName, inputPara.node);
    inputPara.tbeOpInfo->GetInputs(inputs_json_param.inputs);

    bool bres = CheckInputsSize(inputs_json_param);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Check op[%s] inputs_json_param.inputs size failed.",
                           inputPara.keyName.c_str());
        return false;
    });
    TE_FUSION_CHECK(!GenInputsLinkKey(inputPara.node, inputPara.nodeListIdx, inputPara.verifyOpTypeList,
                                      inputs_json_param.inputsLinkKeyList, allNodes, teGraphNode), {
                        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Generate op[%s] inputs link key failed.",
                                           inputPara.node->GetName().c_str());
                        return false;
                    });
    bres = InputsDescToJsonProcess(allNodes, inputs_json_param, opJson, allInputNames);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Input desc to json of op[%s] failed.", inputPara.keyName.c_str());
        return false;
    });

    TE_DBGLOG("Input desc json data of node[%s] is generated successfully", inputPara.keyName.c_str());
    return true;
}

void TeJsonAssemble::GenBuildinOutdescJson(const InOutToJsonParam &outputPara,
                                           const std::unordered_set<ge::Node *> &allNodes, nlohmann::json &jsonStr,
                                           const std::vector<ge::Node *> &teGraphNode)
{
    TE_FUSION_CHECK(outputPara.node == nullptr, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Node is null.");
        return;
    });
    auto current_node_desc = outputPara.node->GetOpDesc();
    auto all_output_desc = current_node_desc->GetAllOutputsDesc();
    auto all_output_anchors = outputPara.node->GetAllOutDataAnchors();
    if (all_output_desc.size() != all_output_anchors.size()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Output description and out anchor size not match: %zu %zu",
                           all_output_desc.size(), all_output_anchors.size());
    }

    size_t peerDataNodesSize = 0;
    for (const auto &node : allNodes) {
        for (const auto &peer_node : node->GetOutDataNodes()) {
            if (allNodes.count(peer_node.get()) == 0) {
                ++peerDataNodesSize;
            }
        }
    }

    uint32_t idx = 0;
    for (auto &anchor : all_output_anchors) {
        OutputsDescToJsonProcess(jsonStr, allNodes, outputPara, idx, anchor, peerDataNodesSize, teGraphNode);
        ++idx;
    }
}

void TeJsonAssemble::GenBuildinAttrsAndModuleJson(const ConstTbeOpInfoPtr &tbeOpInfo, nlohmann::json &jsonStr)
{
    if (tbeOpInfo == nullptr) {
        return;
    }

    GenBuildinAttrsJson(*tbeOpInfo, jsonStr);

    std::string moduleName;
    (void)tbeOpInfo->GetModuleName(moduleName);

    std::string pyModulePath;
    if (PythonApiCall::Instance().UpdateSingleOpModule(*tbeOpInfo, moduleName, &pyModulePath)) {
        jsonStr["py_module_path"] = pyModulePath;
        jsonStr["module_name"] = moduleName;
    }
}

void TeJsonAssemble::GenBuildinPrivateAttrsJson(const TbeOpInfo &tbeOpInfo, json &jsonStr,
                                                const std::vector<std::string> &variableAttrs)
{
    json privateAttrDescJson;
    json privateAttrInfoDescJson;
    const std::vector<TbeAttrValue> privateAttrValues = tbeOpInfo.GetPrivateAttrValues();

    for (const auto &attr : privateAttrValues) {
        json attrInfoJson;
        json attrValueJson;
        std::string attrName = attr.GetName();
        attrInfoJson["name"] = attrName;
        SetAttrDtype2Json(attr, attrInfoJson);
        bool isVariableAttr = JudgeAttrIsVariableAttr(attr, variableAttrs);
        if (isVariableAttr) {
            json valueNone;
            TE_DBGLOGF("[%s] is in variableAttr's value, and set the attr value to None.", attrName.c_str());
            privateAttrDescJson.push_back(valueNone);
            attrInfoJson["value"] = valueNone;
        } else {
            NullDesc nullDesc;
            GetAttrValueToJson(attr, attrValueJson, nullDesc);
            privateAttrDescJson.push_back(attrValueJson);
            attrInfoJson["value"] = attrValueJson;
            privateAttrInfoDescJson.push_back(attrInfoJson);
        }
    }

    if (!privateAttrValues.empty()) {
        jsonStr["private_attr_desc"] = privateAttrDescJson;
        jsonStr["private_attr_info_desc"] = privateAttrInfoDescJson;
    }
}

void TeJsonAssemble::GenBuildinAttrsJson(const TbeOpInfo &tbeOpInfo, json &jsonStr)
{
    std::string opName;
    tbeOpInfo.GetName(opName);
    std::vector<TbeAttrValue> attrValues;
    // collect input tensors args
    tbeOpInfo.GetAttrValues(attrValues);
    std::vector<std::string> variableAttrs;
    GetVariableAttrValue(tbeOpInfo, variableAttrs);

    /**************************************************************************************
    Attention: null_desc comply with Python specifications.
    "attr_desc": [
        false,
        null,
        null,
        [
            1.0,
            null,
            null
        ]
    ],
    "attr_desc_null_desc": [
        null,
        "nan",
        "inf",
        [
            null,
            "inf",
            "-inf"
        ]
    ],
    "attr_info_desc": [
        {
            "dtype": "bool",
            "name": "adj_x1",
            "value": false
        },
        {
            "dtype": "float",
            "name": "adj_x2",
            "value": null
        },
        {
            "dtype": "list_float",
            "name": "adj_x2",
            "value": [
                1.0,
                null,
                null
            ],
            "value_null_desc": [
                null,
                "inf",
                "nan"
            ]
        },
        {
            "dtype": "float",
            "name": "offset_x",
            "value": null,
            "value_null_desc": "-inf"
        }
    ],
    **************************************************************************************/
    json attrDescJson;
    bool hasNullDesc = false;
    std::vector<json> attrDescNullDescJson;
    json attrInfoDescJson;

    for (const TbeAttrValue &attr : attrValues) {
        json attrInfoJson;
        json attrValueJson;
        json attrValueNullDescJson;

        std::string attrName = attr.GetName();
        // "name": "offset_x",
        attrInfoJson["name"] = attrName;

        // "dtype": "list_float",
        SetAttrDtype2Json(attr, attrInfoJson);

        // attr value is None, it means the attr is supporting any value
        bool isVariableAttr = JudgeAttrIsVariableAttr(attr, variableAttrs);
        if (isVariableAttr) {
            json valueNone;

            TE_DBGLOGF("[%s] is in variableAttr's value, and set the attr value to None", attrName.c_str());

            attrDescJson.push_back(valueNone);
            attrDescNullDescJson.push_back(valueNone);

            attrInfoJson["value"] = valueNone;
        } else {
            // get value and value_null_desc
            NullDesc nullDesc;
            GetAttrValueToJson(attr, attrValueJson, nullDesc);

            // value may be null, it could be nan/inf/-inf
            attrDescJson.push_back(attrValueJson);

            bool thisAttrHasNullDesc = false;
            TE_DBGLOG("nullDesc.nullType is %d", nullDesc.nullType);
            if (nullDesc.nullType == NullType::LIST_VALUE) {
                json nullDescJsonList = json::array({});
                for (auto iter = nullDesc.nullDesc.begin();
                     iter != nullDesc.nullDesc.end();
                     iter++) {
                    json itemValue;
                    if (*iter == KEY_INF || *iter == KEY_NEGTIVE_INF || *iter == KEY_NAN) {
                        itemValue = *iter;

                        thisAttrHasNullDesc = true;
                        hasNullDesc = true;
                    }
                    nullDescJsonList.push_back(itemValue);
                }
                attrDescNullDescJson.push_back(nullDescJsonList);
                attrValueNullDescJson = nullDescJsonList;
            } else if (nullDesc.nullType == NullType::SINGLE_VALUE) {
                std::string value = nullDesc.nullDesc[0];
                json nullDescJson;
                if (value == KEY_INF || value == KEY_NEGTIVE_INF || value == KEY_NAN) {
                    nullDescJson = value;

                    thisAttrHasNullDesc = true;
                    hasNullDesc = true;
                }

                attrDescNullDescJson.push_back(nullDescJson);
                attrValueNullDescJson = nullDescJson;
            } else {
                json valueNull;

                attrDescNullDescJson.push_back(valueNull);
            }

            attrInfoJson["value"] = attrValueJson;
            if (thisAttrHasNullDesc) {
                attrInfoJson["value_null_desc"] = attrValueNullDescJson;
                TE_DBGLOGF("attrInfoJson is %s", attrInfoJson.dump().c_str());
            }
        }

        attrInfoDescJson.push_back(attrInfoJson);
    }

    if (!attrValues.empty()) {
        jsonStr["attr_desc"] = attrDescJson;
        if (hasNullDesc) {
            jsonStr["attr_desc_null_desc"] = attrDescNullDescJson;
            TE_DBGLOGF("jsonStr is %s", jsonStr.dump().c_str());
        }
        jsonStr["attr_info_desc"] = attrInfoDescJson;
    }

    GenBuildinPrivateAttrsJson(tbeOpInfo, jsonStr, variableAttrs);
}

void TeJsonAssemble::GenerateOutputLinkKey(const std::unordered_set<ge::Node *> &allNodes,
                                           const ge::OutDataAnchorPtr &anchor,
                                           size_t i, const InOutToJsonParam &outputParam,
                                           const std::vector<ge::Node *> &teGraphNode,
                                           nlohmann::json &outputDescJson) {
    TE_FUSION_CHECK(anchor == nullptr, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Anchor is null.");
        return;
    });
    auto anchorIdx = anchor->GetIdx();
    auto curNode = anchor->GetOwnerNode();
    TE_FUSION_CHECK(curNode == nullptr, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Node is null.");
        return;
    });
    auto peers = anchor->GetPeerInDataAnchors();
    std::string srcTensorName = curNode->GetOpDesc()->GetOutputNameByIndex(anchorIdx);
    ge::InDataAnchorPtr peerInAnchor = (i < peers.size()) ? peers.at(i) : nullptr;
    std::string postFix;
    bool flag = (peerInAnchor != nullptr) && (allNodes.count(peerInAnchor->GetOwnerNode().get()) > 0);
    if (flag) {
        auto peerNode = peerInAnchor->GetOwnerNode();
        std::string dstType;
        for (size_t j = 0; j < teGraphNode.size(); j++) {
            if (teGraphNode[j] == peerNode.get()) {
                dstType = outputParam.verifyOpTypeList[j];
            }
        }
        std::string dstTensorName = peerNode->GetOpDesc()->GetInputNameByIndex(peerInAnchor->GetIdx());
        postFix = dstType + "_" + dstTensorName;
    }
    TE_FUSION_CHECK(outputParam.nodeListIdx >= outputParam.verifyOpTypeList.size(), {
        TE_WARNLOG("Index[%zu] is out of range [%zu].", outputParam.nodeListIdx, outputParam.verifyOpTypeList.size());
        return;
    });
    std::string linkKey = "o:" + outputParam.verifyOpTypeList[outputParam.nodeListIdx] + "_" + srcTensorName + "__" + postFix;
    outputDescJson["link_key"] = linkKey;
    TE_DBGLOG("Op [%s] output[%d] link key is [%s].", anchor->GetOwnerNode()->GetName().c_str(),
               anchorIdx, linkKey.c_str());
    return;
}

void TeJsonAssemble::OutputsDescToJsonProcess(nlohmann::json &json_str, const std::unordered_set<ge::Node *> &allNodes,
                                              const InOutToJsonParam &outputParam, uint32_t idx,
                                              const ge::OutDataAnchorPtr &anchor, size_t peerDataNodesSize,
                                              const std::vector<ge::Node *> &teGraphNode)
{
    std::vector<TbeOpParam> outputs = outputParam.tbeOpInfo->GetOutputs();
    nlohmann::json outputDescJson;
    auto desc = outputParam.node->GetOpDesc()->GetOutputDescPtr(idx);
    if (desc == nullptr) {
        TE_ERRLOG("Op_desc cannot be null.");
        return;
    }
    auto peers = anchor->GetPeerInDataAnchors();
    auto peersCrlAnchor = outputParam.node->GetOutControlAnchor()->GetPeerInControlAnchors();
    auto anchorIdx = anchor->GetIdx();
    auto dims = desc->GetShape().GetDims();
    auto dtype = GetDataTypeStr(desc->GetDataType());
    std::vector<int64_t> originShape;
    std::string originFormat;
    std::string format;
    int32_t subFormat = 0;
    size_t addrType = 0;
    size_t refCount = 0;
    bool hasOut = false;
    std::vector<int64_t> validShape;
    std::vector<int64_t> sliceOffset;
    std::vector<int64_t> sgtSliceShape;
    int64_t l1WorkspaceSize = 0;
    int32_t l1FusionType = 0;
    int64_t l1AddrFlag = -1;
    int64_t l1AddrOffset = 0;
    int64_t l1ValidSize = -1;
    int64_t cAxisValue = -1;
    DdrBaseType ddrBaseProp = DdrBaseType::WORKSPACE;
    std::vector<uint32_t> totalShape;
    uint32_t splitIndex = 0;
    std::vector<int64_t> currShape;
    std::string atomicType;
    std::string tensordtype;
    TensorType tensorType = TT_REQ;
    bool is_null_output = false;
    if (idx < outputs.size()) {
        const std::vector<TbeOpTensor> &tensors = outputs[idx].GetTensors();
        (void)outputs[idx].GetType(tensorType);
        if (tensors.size() > 0) {
            (void)tensors[0].GetOriginShape(originShape);
            (void)tensors[0].GetOriginFormat(originFormat);
            (void)tensors[0].GetFormat(format);
            (void)tensors[0].GetSubFormat(subFormat);
            (void)tensors[0].GetAddrType(addrType);
            (void)tensors[0].GetValidShape(validShape);
            (void)tensors[0].GetSgtSliceShape(sgtSliceShape);
            (void)tensors[0].GetSliceOffset(sliceOffset);
            (void)tensors[0].GetL1WorkspaceSize(l1WorkspaceSize);
            (void)tensors[0].GetL1FusionType(l1FusionType);
            (void)tensors[0].GetL1AddrFlag(l1AddrFlag);
            (void)tensors[0].GetAddrOffset(l1AddrOffset);
            (void)tensors[0].GetL1ValidSize(l1ValidSize);
            (void)tensors[0].GetTotalShape(totalShape);
            (void)tensors[0].GetSplitIndex(splitIndex);
            (void)tensors[0].GetShape(currShape);
            (void)tensors[0].GetCAxisValue(cAxisValue);
            (void)tensors[0].GetAtomicType(atomicType);
            (void)tensors[0].GetType(tensordtype);
            (void)tensors[0].GetIsNullOutput(is_null_output);
            ddrBaseProp = tensors[0].GetDdrBaseProp();
        }
    }
    std::vector<string> peerInputsOrder;
    GetPeerInputsOrder(anchor, idx, allNodes, peerInputsOrder);
    size_t peerSize = 0;
    bool hasDataAnchor = true;
    if (peers.size() > 0) {
        peerSize = peers.size();
    } else if (tensorType != TT_OPT) {
        if (peerDataNodesSize == 0 && peersCrlAnchor.size() > 0 &&
            outputParam.node->GetOutDataAnchor(0) != nullptr) {
            hasDataAnchor = true;
        } else {
            hasDataAnchor = false;
        }
        peerSize = 1;
    } else {
 	    if(is_null_output) {
 	    TE_DBGLOG("Op[%s] with peer size 0 and tensorType TT_OPT create outputDescJson with is_null_output %d",
 	        outputParam.node->GetOpDesc()->GetName().c_str(), is_null_output);
 	    outputDescJson["is_null_output"] = is_null_output;
 	    json_str["output_desc"].push_back(outputDescJson);
 	    return;
 	    }
    }
    
    TE_DBGLOG("Peers size: %d, node: %s", peerSize, outputParam.keyName.c_str());
    for (size_t i = 0; i < peerSize; ++i) {
        // get data_type
        outputDescJson["data_type"] = dtype;
        // get output_name: NodeName + "__" + output order
        std::string currentOutputName = outputParam.keyName + "__" + std::to_string(anchorIdx);
        // if the output is multi-refered, add an ":$seq" suffix
        if (peerSize > 1) {
            std::string suffix;
            GetOutputSuffix(peers.at(i), allNodes, suffix, refCount);
            if (hasOut && suffix == "out") {
                continue;
            }
            if (suffix == "out") {
                hasOut = true;
            }
            if (suffix != "0") {
                currentOutputName += ":" + suffix;
            }
        }
        TE_DBGLOG("currentOutputName is [%s].", currentOutputName.c_str());
        if (allNodes.size() > 1) {
            GenerateOutputLinkKey(allNodes, anchor, i, outputParam, teGraphNode, outputDescJson);
        }
        outputDescJson["shape"] = dims;
        outputDescJson["name"] = currentOutputName;
        outputDescJson["format"] = format;
        outputDescJson["sub_format"] = subFormat;
        outputDescJson["addr_type"] = addrType;
        outputDescJson["output_index"] = idx;
        outputDescJson["ori_shape"] = originShape;
        outputDescJson["ori_format"] = originFormat;
        outputDescJson["valid_shape"] = validShape;
        outputDescJson["sgt_slice_shape"] = sgtSliceShape;
        outputDescJson["slice_offset"] = sliceOffset;
        outputDescJson["L1_workspace_size"] = l1WorkspaceSize;
        outputDescJson["L1_fusion_type"] = l1FusionType;
        outputDescJson["total_shape"] = dims;
        outputDescJson["caxis_values"] = cAxisValue;
        outputDescJson["ddr_base_prop"] = ddrBaseProp;
        if (NotZero(validShape)) {
            outputDescJson["total_shape"] = validShape;
        }
        if (is_null_output) {
 	        TE_DBGLOG("Node[%s] get the null_output as %d", outputParam.keyName.c_str(), is_null_output);
 	        outputDescJson["is_null_output"] = is_null_output;
 	    }
        outputDescJson["split_index"] = splitIndex;
        outputDescJson["shape"] = currShape;
        if (NotZero(sgtSliceShape)) {
            outputDescJson["shape"] = sgtSliceShape;
        }
        if (!atomicType.empty()) {
            outputDescJson["atomic_type"] = atomicType + "." + tensordtype;
        }

        TE_FUSION_CHECK(l1AddrFlag != -1,
            outputDescJson["L1_addr_flag"] = l1AddrFlag);
        outputDescJson["L1_addr_offset"] = l1AddrOffset;
        TE_FUSION_CHECK(l1AddrFlag == 1,
            outputDescJson["L1_valid_size"] = l1ValidSize);
        if (!peerInputsOrder.empty()) {
            outputDescJson["peer_in_param_index"] = peerInputsOrder;
        }
        if (!hasDataAnchor) {
            outputDescJson["has_data_anchor"] = hasDataAnchor;
            TE_DBGLOG("has_data_anchor is %d, node: %s", hasDataAnchor, outputParam.keyName.c_str());
        }
        json_str["output_desc"].push_back(outputDescJson);
    }
    return;
}

void TeJsonAssemble::GetOutputSuffix(ge::InDataAnchorPtr anchor, const std::unordered_set<ge::Node *> &allNodes,
                                     std::string &suffix, size_t &refCount)
{
    if (anchor == nullptr || anchor->GetOwnerNode() == nullptr) {
        suffix = "out";
        return;
    }
    ge::NodePtr node = anchor->GetOwnerNode();
    if (allNodes.count(node.get()) == 0) {
        suffix = "out";
    } else {
        suffix = std::to_string(refCount);
        refCount++;
    }
    return;
}

bool TeJsonAssemble::CheckInputsSize(InputDescJsonParam &inputDescJsonPara)
{
    // get op desc
    auto op_desc = inputDescJsonPara.node->GetOpDesc();
    // get input name list and input desc size
    bool bres = GetOpInputKeyName(inputDescJsonPara.node, inputDescJsonPara.inputNameList);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get Node[%s] input key name.",
                           inputDescJsonPara.node->GetName().c_str());
        return false;
    });
    uint32_t inputNameListSize = inputDescJsonPara.inputNameList.size();
    uint32_t inputDescSize = op_desc->GetInputsSize();
    TE_FUSION_CHECK(inputNameListSize != inputDescSize, {
        REPORT_TE_INNER_ERROR("Node[%s] input name num: %u, does not match input_desc num: %u, and failed.",
                              inputDescJsonPara.node->GetName().c_str(), inputNameListSize, inputDescSize);
        return false;
    });

    for (size_t inputIndex = 0; inputIndex < inputDescJsonPara.inputs.size(); ++inputIndex) {
        TensorType type;
        (void)inputDescJsonPara.inputs[inputIndex].GetType(type);
        if (type == TT_REQ || type == TT_OPT) {
            inputDescJsonPara.inputOrder.insert(pair<int, int>(inputDescJsonPara.inputSize, inputIndex));
            inputDescJsonPara.inputSize++;
        } else {
            // TT_DYN input tensor is list, need to count all list members
            const std::vector<TbeOpTensor> &tensors = inputDescJsonPara.inputs[inputIndex].GetTensors();
            for (size_t idx = 0; idx < tensors.size(); ++idx) {
                inputDescJsonPara.inputOrder.insert(pair<int, int>(inputDescJsonPara.inputSize + idx, inputIndex));
            }
            inputDescJsonPara.inputSize += tensors.size();
        }
    }
    TE_FUSION_CHECK(inputDescSize > inputDescJsonPara.inputSize, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Graph input size[%d] should be <= input size=[%d].", inputDescSize,
                           inputDescJsonPara.inputSize);
        return false;
    });
    return true;
}

bool TeJsonAssemble::GenInputsLinkKey(const ge::Node *node, size_t nodeListIdx,
                                      std::vector<std::string> verifyOpTypeList,
                                      std::vector<std::string> &inputsLinkKeyList,
                                      const std::unordered_set<ge::Node *> &allNodes,
                                      const std::vector<ge::Node *> &teGraphNode)
{
    TE_FUSION_CHECK(node == nullptr, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Node is null.");
        return false;
    });
    TE_FUSION_CHECK(allNodes.size() == 1, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "SingleOpTask don't need link_key.");
        return true;
    });
    const std::string opName = node->GetName();
    const auto &allInAnchors = node->GetAllInDataAnchors();
    for (const auto &anchor : allInAnchors) {
        const auto &peerAnchor = anchor->GetPeerOutAnchor();
        if (peerAnchor == nullptr) {
            TE_DBGLOG("The anchor does not have a peer. Node: %s", opName.c_str());
            continue;
        }
        ge::NodePtr srcNode = peerAnchor->GetOwnerNode();
        if (srcNode == nullptr) {
            TE_WARNLOG("The anchor does not have an owner node. Node: %s", opName.c_str());
            continue;
        }
        std::string prefix = "i:";
        if (allNodes.count(srcNode.get()) > 0) {
            std::string srcOpType;
            for (size_t i = 0; i < teGraphNode.size(); i++) {
                if (teGraphNode[i] == srcNode.get()) {
                    srcOpType = verifyOpTypeList[i];
                }
            }
            auto src_idx = peerAnchor->GetIdx();
            std::string srcTensorName = srcNode->GetOpDesc()->GetOutputNameByIndex(src_idx);
            prefix = prefix + srcOpType + "_" + srcTensorName;
        }
        std::string dstTensorName = node->GetOpDesc()->GetInputNameByIndex(anchor->GetIdx());
        TE_FUSION_CHECK(nodeListIdx >= verifyOpTypeList.size(), {
            TE_WARNLOG("Index[%zu] is out of range [%zu].", nodeListIdx, verifyOpTypeList.size());
            return false;
        });
        inputsLinkKeyList.emplace_back(prefix + "__" + verifyOpTypeList[nodeListIdx] + "_" + dstTensorName);
    }
    return true;
}

bool TeJsonAssemble::InputsDescToJsonProcess(const std::unordered_set<ge::Node *> &allNodes,
                                             const InputDescJsonParam &const_para, nlohmann::json &jsonStr,
                                             std::map<string, int> &allInputNames)
{
    uint32_t nullCnt = 0;
    std::map<uint32_t, uint32_t> dyn_order;
    // process input desc
    std::map<size_t, int64_t> peerOutputsOrder;
    GetPeerOutputsOrder(const_para.node, allNodes, peerOutputsOrder);

    for (size_t inputIdx = 0; inputIdx < const_para.inputSize; ++inputIdx) {
        // record inputDescJson info in json
        nlohmann::json inputDescJson;
        bool isNull = false;

        auto iter = const_para.inputOrder.find(inputIdx);
        TE_FUSION_CHECK (iter == const_para.inputOrder.end(), {
            REPORT_TE_INNER_ERROR("Failed to get input index pair, index=[%zu].", inputIdx);
            return false;
        });

        bool bres = CheckInputsNullTensor(const_para.inputs[iter->second], isNull); // need to consider DYN inputs
        TE_FUSION_CHECK(!bres, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to check Inputs Tensor is Null or not.");
            return false;
        });

        if (isNull) {
            nullCnt++;
            inputDescJson["shape"] = "NULL";
            inputDescJson["data_type"] = 0;
            inputDescJson["name"] = const_para.keyName + "OPT";
        } else {
            // get tensor desc
            const std::vector<TbeOpTensor> &tensors = const_para.inputs.at(iter->second).GetTensors();
            TE_FUSION_CHECK(tensors.size() == 0, {
                REPORT_TE_INNER_ERROR("No input tensors found. Index: %zu", inputIdx);
                return false;
            });
            auto current_tensor_desc = const_para.node->GetOpDesc()->GetInputDesc(inputIdx);
            inputDescJson["shape"] = tensors.at(0).GetShape();
            inputDescJson["data_type"] = GetDataTypeStr(current_tensor_desc.GetDataType());
            inputDescJson["ddr_base_prop"] = tensors.at(0).GetDdrBaseProp();
            auto iter2 = peerOutputsOrder.find(inputIdx);
            if (iter2 == peerOutputsOrder.end()) {
                TE_DBGLOG("Cannot find the order of peer out of input %zu", inputIdx);
            } else {
                inputDescJson["peer_out_param_index"] = iter2->second;
                TE_DBGLOG("Set peer_out_param_index for index[%zu]", inputIdx);
            }
            TensorType type = const_para.inputs.at(iter->second).GetType();
            if (type == TT_REQ || type == TT_OPT) {
                InputsTensorDescToJson(tensors.at(0), inputDescJson);
            } else {
                if (dyn_order.find(iter->second) == dyn_order.end()) {
                    dyn_order[iter->second] = 0;
                } else {
                    dyn_order[iter->second] += 1;
                }
                InputsTensorDescToJson(tensors.at(dyn_order[iter->second]), inputDescJson);
                // record dyn input index in op
                inputDescJson["dyn_index"] = iter->second;
            }
            size_t realIdx = inputIdx - nullCnt;
            // get input_name
            if (allInputNames.find(const_para.inputNameList[realIdx]) == allInputNames.end()) {
                inputDescJson["name"] = const_para.inputNameList[realIdx];
                allInputNames.insert(std::make_pair(inputDescJson["name"], 1));
            } else {
                inputDescJson["name"] = const_para.inputNameList[realIdx] + ":" +
                                        std::to_string(allInputNames[const_para.inputNameList[realIdx]]);
                allInputNames[const_para.inputNameList[realIdx]] += 1;
            }
            if (!const_para.inputsLinkKeyList.empty() && realIdx < const_para.inputsLinkKeyList.size()) {
                inputDescJson["link_key"] = const_para.inputsLinkKeyList[realIdx];
            }
        }
        jsonStr.push_back(inputDescJson);
    }
    return true;
}

/* For the following case, we need to generate a unique
 * key to differenciate them:
 * A, B1, B2 are three nodes which are expected to fuse and
 * B1 and B2 have save type, format, dtype, shape and they
 * are both linked with same output of A.
 *           A                 A
 *     out1 / \  out2    out2 / \  out1
 *      \  /   \ /        \  /   \ /
 *       B1    B2    ->    B2    B1
 *             |           |
 *             B3          B3
 * The topological sequences are different and the final cce
 * files are different too. But the kernel name is the same.
 * So if we generate a kernel and cce for the left pattern,
 * they cannot be used for the right pattern.
 * As a result, we need to make their kernel name different.
 *
 * Here we use the topological id in b1 and b2 to generate a
 * unique id for this output.
 * For example, topo id of B1 and B2 are 5 and 6, the unique id
 * will be 0 and 1 and if topo ids are 100 and 99, the unique id
 * will be 1 and 0.
 * And the
 * */
void TeJsonAssemble::GetPeerInputsOrder(const ge::OutDataAnchorPtr &anchor, uint32_t idx,
                                        const std::unordered_set<ge::Node *> &allNodes,
                                        std::vector<std::string> &peerInputsOrder)
{
    if (idx > 0) {
        return;
    }
    const auto peerInAnchors = anchor->GetPeerInDataAnchors();
    std::vector<ge::NodePtr> peerInNodes;
    /* Use set to get rid of duplicated nodes.
     * Only inner nodes will be counted.
     * Outer nodes will not affect the cce and kernel name.
     * And if multiple references are from the same node,
     * we will only count once. Only references from
     * different nodes will lead to the problem above. */
    std::unordered_set<ge::NodePtr> innerNonDupliPeerInNodes;
    for (const auto &peerInAnchor : peerInAnchors) {
        auto peerInNode = peerInAnchor->GetOwnerNode();
        if (peerInNode == nullptr) {
            continue;
        }
        if (innerNonDupliPeerInNodes.count(peerInNode) == 0) {
            if (allNodes.count(peerInNode.get()) != 0) {
                peerInNodes.emplace_back(peerInNode);
                innerNonDupliPeerInNodes.emplace(peerInNode);
            }
        }
    }

    if (peerInNodes.size() <= 1) {
        TE_DBGLOG("peer in nodes size of out %d for node %s is [%zu]", anchor->GetIdx(),
                  anchor->GetOwnerNode()->GetName().c_str(), peerInNodes.size());
        return;
    }

    /* Outer nodes will not affect the cce and kernel name.
     * And if multiple references are from the same node,
     * we will only count once. Only references from
     * different nodes will lead to the problem above. */
    if (!ContainsDuplicatedType(allNodes)) {
        TE_DBGLOG("Do not contain duplicate nodes for out %d of node %s.", anchor->GetIdx(),
                  anchor->GetOwnerNode()->GetName().c_str());
        return;
    }

    TE_DBGLOG("Size of peerInNodes is %zu for the %dth out of node %s", peerInNodes.size(),
              anchor->GetIdx(), anchor->GetOwnerNode()->GetName().c_str());

    GetAllInnerSuccessorTypes(allNodes, peerInNodes, peerInputsOrder);
}

void TeJsonAssemble::GetAllInnerSuccessorTypes(const std::unordered_set<Node *> &allNodes,
                                               const std::vector<ge::NodePtr> &peerInNodes,
                                               std::vector<string> &successorTypes)
{
    std::deque<ge::NodePtr> allSuccessors;
    for (const auto &peerNode : peerInNodes) {
        allSuccessors.emplace_back(peerNode);
    }
    std::map<std::string, std::vector<Node*>> opMap;
    for (auto iter = allNodes.begin(); iter != allNodes.end(); ++iter) {
        std::string currType = (*iter)->GetType();
        auto it = opMap.find(currType);
        if (it == opMap.end()) {
            std::vector<Node*> newVec;
            newVec.push_back(*iter);
            opMap.insert(std::make_pair(currType, newVec));
        } else {
            it->second.push_back(*iter);
        }
    }

    for (auto iter = opMap.begin(); iter != opMap.end(); ++iter) {
        sort(iter->second.begin(), iter->second.end(), CompareNode);
    }

    size_t index = 0;
    size_t maxRecursive = 1000;
    while (!allSuccessors.empty() && index < maxRecursive) {
        const auto &peerNode = allSuccessors.front();
        TE_DBGLOG("id: %zu, node name: %s, type: %s, id: %ld", index, peerNode->GetName().c_str(),
                  peerNode->GetType().c_str(), peerNode->GetOpDesc()->GetId());

        std::string currType = peerNode->GetType();
        auto it = opMap.find(currType);
        if (it == opMap.end()) {
            TE_WARNLOG("Node [%s], currType: %s not found.", peerNode->GetName().c_str(), currType.c_str());
        } else {
            auto iter = std::find(it->second.begin(), it->second.end(), peerNode.get());
            int pos = std::distance(it->second.begin(), iter);
            currType = currType + ":" + std::to_string(pos);
        }
        TE_DBGLOG("node: %s, currType: %s", peerNode->GetName().c_str(), currType.c_str());
        successorTypes.emplace_back(currType);
        allSuccessors.pop_front();
        auto nextSuccessors = peerNode->GetOutDataNodes();
        size_t validCount = 0;
        for (const auto &ele : nextSuccessors) {
            if (allNodes.count(ele.get()) != 0) {
                allSuccessors.emplace_back(ele);
                validCount++;
            }
        }
        /* When reaching the end of a branch, add a empty string delimiter.
         * This operation is to differenciate the following two
         * cases:
         *           A                 A
         *     out1 / \  out2    out2 / \  out1
         *      \  /   \ /        \  /   \ /
         *       B      B    ->    B      B
         *              |          |
         *              B          B
         *
         * They are different and the successorTypes are:
         * left: {B, "", B, B, ""}
         * right: {B, B, "", B, ""} */
        if (validCount == 0) {
            successorTypes.emplace_back("");
        }
        index++;
    }
}

void TeJsonAssemble::GetPeerOutputsOrder(const ge::Node *node, const std::unordered_set<Node *> &allNodes,
                                         std::map<size_t, int64_t> &peerOutputsOrder)
{
    TE_DBGLOG("Get peer output order of node [%s, %s], total node count [%zu].",
              node->GetNamePtr(), node->GetTypePtr(), allNodes.size());
    /* If the all inputs of node are from outside nodes, we do not need set the order. */
    bool isAllInputsOuter = true;
    const auto inDataNodes = node->GetInDataNodes();
    for (auto &input_node : inDataNodes) {
        TE_DBGLOG("Input node: [%s, %s].", input_node->GetNamePtr(), input_node->GetTypePtr());
        if (allNodes.count(input_node.get()) != 0) {
            isAllInputsOuter = false;
            break;
        }
    }

    if (!ContainsDuplicatedType(inDataNodes)) {
        return;
    }

    auto inDataAnchors = node->GetAllInDataAnchors();
    /* We do not care the sequence of input when generating the json for single op. */
    if (allNodes.size() == 1 || isAllInputsOuter || node->GetType() == "Mul" || node->GetType() == "Add") {
        for (size_t inputIndex = 0; inputIndex < inDataAnchors.size(); inputIndex++) {
            peerOutputsOrder.emplace(std::make_pair(inputIndex, 0));
        }
        return;
    }
    std::vector<std::pair<size_t, int64_t>> allInputsId;
    for (size_t inputIndex = 0; inputIndex < inDataAnchors.size(); inputIndex++) {
        if (inDataAnchors.at(inputIndex) == nullptr ||
            inDataAnchors.at(inputIndex)->GetPeerOutAnchor() == nullptr ||
            inDataAnchors.at(inputIndex)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
            allInputsId.emplace_back(std::make_pair(inputIndex, -1));
        } else {
            auto inputNode = inDataAnchors.at(inputIndex)->GetPeerOutAnchor()->GetOwnerNode();
            auto peerOutAnchor = inDataAnchors.at(inputIndex)->GetPeerOutAnchor();
            int64_t id = inputNode->GetOpDesc()->GetId() * 100 + peerOutAnchor->GetIdx();
            allInputsId.emplace_back(std::make_pair(inputIndex, id));
        }
    }
    std::sort(allInputsId.begin(), allInputsId.end(), [](std::pair<size_t, int64_t> a, std::pair<size_t, int64_t> b) {
        return a.second < b.second;
    });

    int64_t peerOutId = 0;
    for (auto &ele : allInputsId) {
        peerOutputsOrder.emplace(std::make_pair(ele.first, peerOutId));
        peerOutId++;
    }
}

/**
 * @brief: check op input option tensor is null or not
 * @param [in] input: op input parameter
 * @param [out] isNull: op input option tensor is null or not
 * @return bool: check input tensor null result
 */
bool TeJsonAssemble::CheckInputsNullTensor(const TbeOpParam &input, bool &isNull)
{
    isNull = false;
    if (input.GetType() != TT_OPT) {
        return true;
    }
    // Null tensor situation 1: no tensor info
    const std::vector<TbeOpTensor> &tensors = input.GetTensors();
    if (tensors.size() == 0) {
        isNull = true;
        return true;
    }
    // option tensor size is 0 or 1
    TE_FUSION_CHECK(tensors.size() > 1, {
        REPORT_TE_INNER_ERROR("The current op tensor size is %zu; it should be either 0 or 1.", tensors.size());
        return false;
    });

    return true;
}

void TeJsonAssemble::InputsTensorDescToJson(const TbeOpTensor &tensor, json &inputDescJson)
{
    const std::vector<int64_t> &validShape = tensor.GetValidShape();
    if (NotZero(validShape)) {
        inputDescJson["shape"] = validShape;
    }

    const std::vector<int64_t> &sgtSliceShape = tensor.GetSgtSliceShape();
    if (NotZero(sgtSliceShape)) {
        inputDescJson["shape"] = sgtSliceShape;
    }

    const std::vector<int64_t> &currShape = tensor.GetShape();
    const std::vector<int64_t> &originShape = tensor.GetOriginShape();
    inputDescJson["ori_shape"] = originShape;
    inputDescJson["format"] = tensor.GetFormat();
    inputDescJson["sub_format"] = tensor.GetSubFormat();
    inputDescJson["ori_format"] = tensor.GetOriginFormat();
    inputDescJson["addr_type"] = tensor.GetAddrType();
    inputDescJson["valid_shape"] = validShape;
    inputDescJson["sgt_slice_shape"] = sgtSliceShape;
    inputDescJson["slice_offset"] = tensor.GetSliceOffset();
    inputDescJson["L1_workspace_size"] = tensor.GetL1WorkspaceSize();
    inputDescJson["L1_fusion_type"] = tensor.GetL1FusionType();
    inputDescJson["total_shape"] = currShape;
    inputDescJson["split_index"] = tensor.GetSplitIndex();
    inputDescJson["caxis_values"] = tensor.GetCAxisValue();

    int64_t l1AddrFlag = tensor.GetL1AddrFlag();
    TE_FUSION_CHECK(l1AddrFlag != -1, inputDescJson["L1_addr_flag"] = l1AddrFlag);
    inputDescJson["L1_addr_offset"] = tensor.GetAddrOffset();

    int64_t l1ValidSize = tensor.GetL1ValidSize();
    TE_FUSION_CHECK(l1AddrFlag == 1, inputDescJson["L1_valid_size"] = l1ValidSize);

    SetJsonRange(inputDescJson, tensor, currShape, originShape);
    bool isFirstLayer = false;
    bool hasSet = tensor.GetFirstLayer(isFirstLayer);
    TE_FUSION_CHECK(hasSet, inputDescJson["is_first_layer"] = isFirstLayer);
    if (tensor.HasConstValue()) {
        SetConstValueToJson(tensor, inputDescJson);
    }
}

void TeJsonAssemble::SetJsonRange(json &desc, const TbeOpTensor &tensor, const std::vector<int64_t> &currShape,
                                  const std::vector<int64_t> &oriShape)
{
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    bool hasSet = tensor.GetShapeRange(shapeRange);
    GetRealShapeRange(hasSet, currShape, shapeRange);
    desc["range"] = shapeRange;

    std::vector<std::pair<int64_t, int64_t>> oriShapeRange;
    hasSet = tensor.GetOriginShapeRange(oriShapeRange);
    GetRealShapeRange(hasSet, oriShape, oriShapeRange);
    desc["ori_range"] = oriShapeRange;

    std::vector<std::pair<int64_t, int64_t>> valueRange;
    hasSet = tensor.GetValueRange(valueRange);
    if (hasSet) {
        desc["value_range"] = valueRange;
    }
}

void TeJsonAssemble::GetRealShapeRange(const bool &hasSet, const std::vector<int64_t> &currShape,
                                       std::vector<std::pair<int64_t, int64_t>> &shapeRange)
{
    if (hasSet && shapeRange.size() != 0) {
        return;
    }
    for (const auto &shape : currShape) {
        if (shape > 0) {
            shapeRange.emplace_back(shape, shape);
        } else {
            shapeRange.emplace_back(0, -1);
        }
    }
}

void TeJsonAssemble::SetConstValueToJson(const TbeOpTensor &tensor, json &inputDescJson)
{
    bool isConstValueNone = false;
    (void)tensor.IsConstValueNone(isConstValueNone);
    if (isConstValueNone) {
        bool isConstValueRange = false;
        (void)tensor.GetConstValueRangeFlag(isConstValueRange);
        if (isConstValueRange) {
            TbeAttrValue constValueRange;
            (void)tensor.GetConstValueRange(constValueRange);
            json constValueRangeJson;
            NullDesc nullDesc;
            GetAttrValueToJson(constValueRange, constValueRangeJson, nullDesc);
            inputDescJson["const_value_range"] = constValueRangeJson;
        }
    } else {
        TbeAttrValue constValue;
        (void)tensor.GetConstValue(constValue);
        json constValueJson;
        NullDesc nullDesc;
        GetAttrValueToJson(constValue, constValueJson, nullDesc);
        inputDescJson["const_value"] = constValueJson;

        bool hasValidNullDesc = false;
        json nullDescJsonList = json::array({});
        TE_DBGLOG("nullDesc.nullType is: %d", nullDesc.nullType);
        for (auto iter = nullDesc.nullDesc.begin();
             iter != nullDesc.nullDesc.end();
             iter++) {
            json item;
            if (*iter == KEY_NAN || *iter == KEY_INF || *iter == KEY_NEGTIVE_INF) {
                item = *iter;
                hasValidNullDesc = true;
            }
            nullDescJsonList.push_back(item);
        }

        if (hasValidNullDesc) {
            inputDescJson["const_value_null_desc"] = nullDescJsonList;
            TE_DBGLOGF("inputDescJson is %s", inputDescJson.dump().c_str());
        }
    }
}

bool TeJsonAssemble::GenDatadescJson(const std::vector<ge::Node *> &teGraphNode, json &jsonStr)
{
    std::map<string, int> allInputNames;
    for (auto &currentNode : teGraphNode) {
        TE_FUSION_CHECK(currentNode == nullptr, continue);
        // get op desc
        auto currentNodeOpDesc = currentNode->GetOpDesc();

        // get input name list and input desc size
        vector<string> inputNameList;
        bool bres = GetOpInputKeyName(currentNode, inputNameList);
        TE_FUSION_CHECK(!bres, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to get Node[%s] input key name.", currentNode->GetName().c_str());
            return false;
        });
        uint32_t inputNameListSize = inputNameList.size();
        uint32_t inputDescSize = currentNodeOpDesc->GetInputsSize();
        TE_FUSION_CHECK(inputDescSize != inputNameListSize, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Node[%s] input name num: %u does not match input_desc num: %u, and failed.",
                               currentNode->GetName().c_str(), inputNameListSize, inputDescSize);
            return false;
        });

        std::string keyName;
        TE_FUSION_CHECK(!TbeOpInfoCache::Instance().GetOpKeyNameByNode(currentNode, keyName), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get node key name.");
            return false;
        });
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfo(keyName);
        TE_FUSION_CHECK(tbeOpInfo == nullptr, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get tbe op info.");
            return false;
        });

        std::vector<TbeOpParam> inputs;
        (void)tbeOpInfo->GetInputs(inputs);

        uint32_t inputTensorSize = 0;
        map<int, int> inputOrder;
        for (size_t inputIdx = 0; inputIdx < inputs.size(); ++inputIdx) {
            TensorType type;
            (void)inputs[inputIdx].GetType(type);
            if (type == TT_REQ || type == TT_OPT) {
                inputOrder.insert(pair<int, int>(inputTensorSize, inputIdx));
                inputTensorSize++;
            } else {
                // TT_DYN input tensor is list, need to count all list members
                const std::vector<TbeOpTensor> &tensors = inputs[inputIdx].GetTensors();
                for (size_t idx = 0; idx < tensors.size(); ++idx) {
                    inputOrder.insert(pair<int, int>(inputTensorSize + idx, inputIdx));
                }
                inputTensorSize += tensors.size();
            }
        }
        TE_FUSION_CHECK((inputDescSize > inputTensorSize), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Graph input size[%d] should be <= input size=[%d].",
                               inputDescSize, inputTensorSize);
            return false;
        });
        uint32_t nullCnt = 0;
        for (uint32_t idx = 0; idx < inputTensorSize; ++idx) {
            const map<int, int>::const_iterator iter = inputOrder.find(idx);
            TE_FUSION_CHECK(iter == inputOrder.end(), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                   "Failed to get input index pair, index=[%d].", idx);
                return false;
            });
            bool isNull = false;
            bres = CheckInputsNullTensor(inputs[iter->second], isNull); // need to consider DYN inputs
            TE_FUSION_CHECK(!bres, {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                   "Failed to check if inputs tensor is null.");
                return false;
            });

            std::string inputName = "";
            json inputDescJson;

            if (isNull) {
                nullCnt++;
                inputName = keyName + "OPT";
                inputDescJson["shape"] = "NULL";
                inputDescJson["data_type"] = 0;
                inputDescJson["name"] = inputName;
            } else {
                ge::Node *preNode = GetPreviousNode(currentNode, idx);
                TE_FUSION_CHECK(preNode == nullptr, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                       "Failed to get peer node point, curr keyName=[%s], index=[%d].",
                                       keyName.c_str(), idx);
                    return false;
                });
                if (std::find(teGraphNode.begin(), teGraphNode.end(), preNode) == teGraphNode.end()) {
                    bres = GenNodeDataJson(teGraphNode, currentNode, idx, iter->second, inputDescJson, true);
                    TE_FUSION_CHECK(!bres, {
                        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                                           "Failed to gen data json, index=[%d].", iter->second);
                        return false;
                    });

                    // get input_name
                    inputName = inputNameList[idx - nullCnt];
                    inputDescJson["name"] = inputName;
                }
            }
            TE_FUSION_CHECK(inputName != "", {
                json dataNodeJson;
                if (allInputNames.find(inputName) == allInputNames.end()) {
                    dataNodeJson["name"] = inputName;
                    if (!isNull) {
                        allInputNames.insert(std::make_pair(inputName, 1));
                    }
                } else {
                    dataNodeJson["name"] = inputName + ":" + std::to_string(allInputNames[inputName]);
                    inputDescJson["name"] = inputName + ":" + std::to_string(allInputNames[inputName]);
                    allInputNames[inputName] += 1;
                }

                dataNodeJson["type"] = "Data";
                dataNodeJson["output_desc"].push_back(inputDescJson);
                // add this data node to opInputList
                jsonStr.push_back(dataNodeJson);
            });
        }
    }
    // func end log
    TE_DBGLOG("Generate Data desc json data success.");

    return true;
}

bool TeJsonAssemble::GenNodeDataJson(const std::vector<ge::Node *> &teGraphNode, ge::Node *currNode,
                                     uint32_t inputTensorIdx, uint32_t inputGroupIdx, json &inputDescJson,
                                     bool needConvertBoolToInt8)
{
    ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(currNode);
    TE_FUSION_NOTNULL(tbeOpInfo);
    const std::vector<TbeOpParam> &inputs = tbeOpInfo->GetInputs();
    TE_FUSION_CHECK(inputGroupIdx >= inputs.size(), return false);

    const std::vector<TbeOpTensor> &tensors = inputs.at(inputGroupIdx).GetTensors();
    TE_FUSION_CHECK(tensors.size() == 0, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "There's no input tensors. idx:%u", inputGroupIdx);
        return false;
    });

    uint32_t tensorCount = 0;
    for (uint32_t inputIndex = 0; inputIndex < inputGroupIdx; ++inputIndex) {
        TensorType tensorType = inputs.at(inputIndex).GetType();
        if (tensorType == TT_REQ || tensorType == TT_OPT) {
            tensorCount++;
        } else {
            tensorCount += inputs.at(inputIndex).GetTensors().size();
        }
    }

    TensorType type = inputs.at(inputGroupIdx).GetType();
    uint32_t tensorIdx = 0;
    TE_FUSION_CHECK((type == TT_DYN), {
        TE_FUSION_CHECK ((inputTensorIdx < tensorCount || inputTensorIdx >= (tensorCount + tensors.size())), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                               "Failed to get input index error, index=[%d].", inputTensorIdx);
            return false;
        });
        tensorIdx = inputTensorIdx - tensorCount;
    });

    const std::vector<int64_t> &originShape = tensors.at(0).GetOriginShape();
    const std::vector<int64_t> &validShape = tensors.at(tensorIdx).GetValidShape();
    std::vector<int64_t> sgtSliceShape;
    inputDescJson["ori_shape"] = originShape;
    inputDescJson["ori_format"] = tensors.at(0).GetOriginFormat();
    inputDescJson["addr_type"] = tensors.at(tensorIdx).GetAddrType();
    inputDescJson["valid_shape"] = validShape;
    inputDescJson["sgt_slice_shape"] = sgtSliceShape;
    inputDescJson["slice_offset"] = tensors.at(tensorIdx).GetSliceOffset();
    inputDescJson["L1_workspace_size"] = tensors.at(tensorIdx).GetL1WorkspaceSize();
    inputDescJson["L1_fusion_type"] = tensors.at(tensorIdx).GetL1FusionType();
    inputDescJson["split_index"] = tensors.at(tensorIdx).GetSplitIndex();
    inputDescJson["caxis_values"] = tensors.at(tensorIdx).GetCAxisValue();
    TE_DBGLOG("cAxisValue [%ld], tensorIdx = %u.", tensors.at(tensorIdx).GetCAxisValue(), tensorIdx);
    inputDescJson["ddr_base_prop"] = tensors.at(tensorIdx).GetDdrBaseProp();

    bool isFirstLayer = false;
    bool hasSet = tensors.at(0).GetFirstLayer(isFirstLayer);
    TE_FUSION_CHECK(hasSet, {
        inputDescJson["is_first_layer"] = isFirstLayer;
    });

    int64_t l1AddrFlag = tensors.at(tensorIdx).GetL1AddrFlag();
    TE_FUSION_CHECK(l1AddrFlag != -1, {
        inputDescJson["L1_addr_flag"] = l1AddrFlag;
    });

    inputDescJson["L1_addr_offset"] = tensors.at(tensorIdx).GetAddrOffset();
    TE_FUSION_CHECK(l1AddrFlag == 1, {
        inputDescJson["L1_valid_size"] = tensors.at(tensorIdx).GetL1ValidSize();
    });

    TE_FUSION_CHECK(type == TT_DYN, {
        // record dyn input index in op
        inputDescJson["dyn_index"] = inputGroupIdx;
    });

    std::string dType = tensors.at(tensorIdx).GetType();
    if (needConvertBoolToInt8 && dType == "bool") {
        dType = "int8";
    }
    std::vector<int64_t> currShape = tensors.at(tensorIdx).GetShape();
    inputDescJson["shape"] = currShape;
    inputDescJson["format"] = tensors.at(tensorIdx).GetFormat();
    inputDescJson["sub_format"] = tensors.at(tensorIdx).GetSubFormat();
    inputDescJson["data_type"] = dType;
    inputDescJson["total_shape"] = currShape;

    if (NotZero(validShape)) {
        inputDescJson["shape"] = validShape;
        currShape = validShape;
    }
    SetJsonRange(inputDescJson, tensors.at(tensorIdx), currShape, originShape);

    bool bres = ChangeInputDescShape(teGraphNode, currNode, currShape, inputGroupIdx, inputDescJson);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to change input shape for fusion.");
        return false;
    });
    if (tensors.at(tensorIdx).HasConstValue()) {
        SetConstValueToJson(tensors.at(tensorIdx), inputDescJson);
    }
    return true;
}

void TeJsonAssemble::GenerateSocInfoJson(const std::vector<ConstTbeOpInfoPtr> &tbeOpInfoVec,
                                         nlohmann::json &socInfoJson)
{
    if (tbeOpInfoVec.empty()) {
        return;
    }

    std::string coreType;
    std::string coreNum;
    std::string opDebugConfig;
    bool hasEnableVectorCore = false;
    bool hasDisableVectorCore = false;
    bool isSingleOpFlag = false;
    ConstTbeOpInfoPtr firstOpInfo = nullptr;
    for (const ConstTbeOpInfoPtr &opInfo : tbeOpInfoVec) {
        if (opInfo == nullptr) {
            continue;
        }
        if (firstOpInfo == nullptr) {
            firstOpInfo = opInfo;
        }
        if (coreNum.empty()) {
            opInfo->GetOption(ge::AICORE_NUM, coreNum);
        }
        if (opDebugConfig.empty()) {
            opDebugConfig = opInfo->GetOpDebugConfig();
        }
        if (coreType.empty() || opInfo->GetOpCoreType() == AI_CORE) {
            coreType = opInfo->GetOpCoreType();
        }
        TE_DBGLOG("Vector core type for [%s] is [%d].", opInfo->GetName().c_str(),
                  static_cast<int32_t>(opInfo->GetVectorCoreType()));
        if (opInfo->IsOriSingleOpScene()) {
            isSingleOpFlag = true;
        }
        if (opInfo->GetVectorCoreType() == VectorCoreType::ENABLE) {
            hasEnableVectorCore = true;
        }
        if (opInfo->GetVectorCoreType() == VectorCoreType::DISABLE) {
            hasDisableVectorCore = true;
        }
    }
    TE_DBGLOG("The core type is [%s]", coreType.c_str());
    bool enableVectorCore = hasEnableVectorCore && !hasDisableVectorCore && !isSingleOpFlag;
    if (enableVectorCore) {
        coreType = MIX_VECTOR_CORE;  // for temp
        TE_DBGLOG("The core type is fixed to [%s].", MIX_VECTOR_CORE);
    }
    socInfoJson["device_id"] = TeConfigInfo::Instance().GetDeviceId();
    socInfoJson["coreType"] = coreType;
    socInfoJson["coreNum"] = coreNum;
    socInfoJson["socVersion"] = TeConfigInfo::Instance().GetSocVersion();
    socInfoJson["l1Fusion"] = TeConfigInfo::Instance().GetL1Fusion();
    socInfoJson["l2Fusion"] = TeConfigInfo::Instance().GetL2Fusion();
    socInfoJson["l2Mode"] = TeConfigInfo::Instance().GetL2Mode();

    socInfoJson["status_check"] = TeContextUtils::GetStatusCheck();
    socInfoJson["op_debug_level"] = TeConfigInfo::Instance().GetOpDebugLevelStr();
    socInfoJson["deterministic"] = TeContextUtils::GetDeterministic();
    socInfoJson["vector_fp_ceiling"] = TeConfigInfo::Instance().GetFpCeilingMode();
    if (TeContextUtils::GetDeterministicLevel() != "0") {
        TE_DBGLOG("Soc_info set deterministic_level [%s].", TeContextUtils::GetDeterministicLevel().c_str());
        socInfoJson["deterministic_level"] = TeContextUtils::GetDeterministicLevel();
    }
    socInfoJson[kEnableVectorCore] = enableVectorCore;
    if (!opDebugConfig.empty()) {
        socInfoJson["op_debug_config"] = opDebugConfig;
    }

    bool isValidOpInfo = (firstOpInfo != nullptr) && (firstOpInfo->GetNode() != nullptr) &&
                         (firstOpInfo->GetNode()->GetOpDesc() != nullptr);
    if (isValidOpInfo) {
        std::map<std::string, std::string> hardwareInfoMap;
        TeConfigInfo::Instance().GetHardwareInfoMap(hardwareInfoMap);
        if (hardwareInfoMap.find(kAiCoreCnt) != hardwareInfoMap.end()) {
            socInfoJson[kAiCoreCnt] = hardwareInfoMap[kAiCoreCnt];
            TE_DBGLOG("OpNode: %s, set ge ai_core_cnt: %s", firstOpInfo->GetName().c_str(), hardwareInfoMap[kAiCoreCnt].c_str());
        }
        if (hardwareInfoMap.find(kVectorCoreCnt) != hardwareInfoMap.end()) {
            socInfoJson[kVectorCoreCnt] = hardwareInfoMap[kVectorCoreCnt];
            TE_DBGLOG("OpNode: %s, set ge vector_core_cnt to %s", firstOpInfo->GetName().c_str(), hardwareInfoMap[kVectorCoreCnt].c_str());
        }
        std::string custAicNum = firstOpInfo->GetCustAicNum();
        if (custAicNum != "" && std::stoi(custAicNum) >= 0) {
            socInfoJson[kAiCoreCnt] = custAicNum;
            TE_DBGLOG("OpNode: %s, cust_aic_num: %s", firstOpInfo->GetName().c_str(), custAicNum.c_str());
        }
        std::string custAivNum = firstOpInfo->GetCustAivNum();
        if (custAivNum != "" && std::stoi(custAivNum) >= 0) {
            socInfoJson[kVectorCoreCnt] = custAivNum;
            TE_DBGLOG("OpNode: %s, cust_aiv_num: %s", firstOpInfo->GetName().c_str(), custAivNum.c_str());
        }
    }
}

void TeJsonAssemble::AssembleComipleParams(const std::vector<ge::Node *> &fusionNodes, nlohmann::json &jsonData)
{
    bool isOnlyFusionCheck = false;
    bool isAutoFusionPattern = false;
    int64_t scopeId = 0;
    std::string graphName;
    for (const Node *node : fusionNodes) {
        if (node == nullptr) {
            continue;
        }
        ge::OpDescPtr opDescPtr = node->GetOpDesc();
        if (!isOnlyFusionCheck) {
            (void)AttrUtils::GetBool(opDescPtr, ONLY_FUSION_CHECK, isOnlyFusionCheck);
        }
        if (!isAutoFusionPattern) {
            (void)AttrUtils::GetBool(opDescPtr, AUTO_FUSION_PATTERN, isAutoFusionPattern);
        }

        if (graphName.empty()) {
            (void)AttrUtils::GetStr(opDescPtr, ATTR_NAME_GRAPH_NAME, graphName);
        }

        if (opDescPtr->HasAttr(ATTR_NAME_L1_FUSION_SCOPE)) {
            (void)AttrUtils::GetInt(opDescPtr, ATTR_NAME_L1_FUSION_SCOPE, scopeId);
        } else if (opDescPtr->HasAttr(ge::ATTR_NAME_FUSION_SCOPE)) {
            (void)AttrUtils::GetInt(opDescPtr, ge::ATTR_NAME_FUSION_SCOPE, scopeId);
        }
    }
    jsonData["scope_id"] = scopeId;
    jsonData["graph_name"] = graphName;
    jsonData["only_fusion_check"] = isOnlyFusionCheck;
    jsonData["auto_fusion_pattern"] = isAutoFusionPattern;
}

void TeJsonAssemble::FillOptionalOutputWithNull(const std::vector<ge::Node *> &teGraphNode, nlohmann::json &jsonData)
{
    if (jsonData.find("op_list") == jsonData.end()) {
        TE_WARNLOG("Json data does not contains [op_list].");
        return;
    }

    for (auto &opNode : jsonData["op_list"]) {
        // only deal node whose type is not Data
        if (opNode.find("type") != opNode.end()) {
            std::string opType = opNode["type"];
            if (opType == "Data") {
                continue;
            }
        }

        if (opNode.find("output_desc") != opNode.end() && opNode.find("name") != opNode.end()) {
            nlohmann::json prebuildOutputDesc;
            std::string jsonName = opNode["name"];
            for (auto &currentNode : teGraphNode) {
                if (currentNode == nullptr) {
                    continue;
                }
                std::string keyName;
                bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(currentNode, keyName);
                TE_FUSION_CHECK(!bres, {
                    TE_WARNLOG("Failed to get node key name.");
                    return;
                });
                if (keyName == jsonName) {
                    GetPrebuildOutput(keyName, prebuildOutputDesc);
                    break;
                }
            }
            if (prebuildOutputDesc.find("list_args") != prebuildOutputDesc.end()) {
                opNode["output_data_desc"] = prebuildOutputDesc["list_args"];
                RefreshSgtSliceShape(opNode["output_desc"], opNode["output_data_desc"]);
            }
        }
    }
    TE_DBGLOG("Finished filling optional output with null");
}

void TeJsonAssemble::GetPrebuildOutput(const std::string &nodeName, nlohmann::json &jsonStr)
{
    std::string opParamStr;
    if (!TeCacheManager::Instance().GetOpArgsCache(nodeName, opParamStr)) {
        TE_ERRLOG("Failed to get op args info, nodeName:%s", nodeName.c_str());
        return;
    }
    TE_DBGLOG("GetOpParams of node[%s] is [%s]", nodeName.c_str(), opParamStr.c_str());

    try {
        jsonStr = json::parse(opParamStr);
    } catch (std::exception &e) {
        REPORT_TE_INNER_ERROR("Failed to parser json_str, the json_str is %s and the reason is %s",
                              opParamStr.c_str(), e.what());
        return;
    }
}

void TeJsonAssemble::RefreshSgtSliceShape(nlohmann::json &outputDesc, nlohmann::json &outputDataDesc)
{
    if (outputDesc.find("sgt_slice_shape") == outputDesc.end()) {
        return;
    }
    string sgtSliceShape = outputDesc["sgt_slice_shape"];
    TE_DBGLOG("Sgt slice shape is %s.", sgtSliceShape.c_str());
    if (sgtSliceShape != "[]") {
        outputDataDesc["sgt_slice_shape"] = outputDesc["sgt_slice_shape"];
        outputDataDesc["shape"] = outputDesc["shape"];
    }
}

void TeJsonAssemble::FilterOutputMultipleReference(nlohmann::json &jsonData)
{
    TE_DBGLOG("Begin to FilterOutputMultipleReference");
    if (jsonData.find("op_list") == jsonData.end()) {
        TE_WARNLOG("Json data does not contains [op_list].");
        return;
    }

    // collect all input desc name of nodes
    std::vector<std::string> inputNameVec;
    GetAllInputNameVec(jsonData, inputNameVec);

    for (auto &opNode : jsonData["op_list"]) {
        // only deal node whose type is not Data
        if (opNode.find("type") != opNode.end()) {
            std::string opType = opNode["type"];
            if (opType == "Data") {
                continue;
            }
        }

        if (opNode.find("output_desc") != opNode.end()) {
            nlohmann::json &outputDescVec = opNode["output_desc"];
            RemoveMultipleReferOutput(inputNameVec, outputDescVec);
        }
    }
    TE_DBGLOG("Finish to FilterOutputMultipleReference");
}

void TeJsonAssemble::RemoveMultipleReferOutput(const std::vector<std::string> &inputNameVec,
                                               nlohmann::json &outputDescVec)
{
    std::map<std::string, std::vector<uint32_t>> outputNameIndexMap;
    // collect output index whose name does not appear among the input names
    uint32_t outputIndex = 0;
    for (auto &outputDesc : outputDescVec) {
        if (outputDesc.find("name") != outputDesc.end()) {
            std::string outputDescName = outputDesc["name"];
            std::vector<std::string> nameSplitVec;
            size_t splitSize = SplitString(outputDescName, ":", nameSplitVec);
            if (splitSize > 1) {
                outputDescName = nameSplitVec[0];
            }
            if (std::find(inputNameVec.begin(), inputNameVec.end(), outputDescName) != inputNameVec.end()) {
                continue;
            }
            // if the output name has not been referred by other input
            const auto indexIter = outputNameIndexMap.find(outputDescName);
            if (indexIter == outputNameIndexMap.end()) {
                std::vector<uint32_t> outputIndexVec = {outputIndex};
                outputNameIndexMap.emplace(outputDescName, outputIndexVec);
            } else {
                indexIter->second.push_back(outputIndex);
            }
        }
        outputIndex++;
    }

    // collect the output index
    std::vector<uint32_t> removeOutputIndexVec;
    for (auto iter = outputNameIndexMap.begin(); iter != outputNameIndexMap.end(); ++iter) {
        if (iter->second.size() <= 1) {
            continue;
        }
        // if the size of output index is greater than one, remove them from output desc except the first one
        removeOutputIndexVec.insert(removeOutputIndexVec.end(), iter->second.begin() + 1, iter->second.end());
    }
    std::sort(removeOutputIndexVec.begin(), removeOutputIndexVec.end());
    if (!removeOutputIndexVec.empty()) {
        for (int32_t i = removeOutputIndexVec.size() -1; i >= 0; --i) {
            outputDescVec.erase(outputDescVec.begin() + removeOutputIndexVec[i]);
        }
    }
}

void TeJsonAssemble::GetAllInputNameVec(nlohmann::json &jsonData, std::vector<std::string> &inputNameVec)
{
    for (auto &opNode : jsonData["op_list"]) {
        if (opNode.find("type") != opNode.end()) {
            std::string opType = opNode["type"];
            if (opType == "Data") {
                continue;
            }
        }
        if (opNode.find("input_desc") != opNode.end()) {
            for (auto &inputDesc : opNode["input_desc"]) {
                if (inputDesc.find("name") != inputDesc.end()) {
                    inputNameVec.push_back(inputDesc["name"]);
                }
            }
        }
    }
}

void TeJsonAssemble::GetUniqueJsonStr(const json &oriJson, std::string &uniqueJsonStr)
{
    json jsonData = oriJson;
    jsonData.erase("fusion_op_name");
    jsonData.erase("scope_id");
    jsonData.erase("graph_name");

    if (jsonData.find("SocInfo") != jsonData.end()) {
        if (jsonData["SocInfo"].find("device_id") != jsonData["SocInfo"].end()) {
            jsonData["SocInfo"].erase("device_id");
        }
    }

    if (jsonData.find("op_list") == jsonData.end()) {
        TE_ERRLOG("Json data contains no 'op_list' \n%s", jsonData.dump().c_str());
        return;
    }

    if (!GetJsonDataWithOpList(oriJson, jsonData)) {
        return;
    }

    uniqueJsonStr = jsonData.dump();
}

bool TeJsonAssemble::GetJsonDataWithOpList(const json &oriJson, json &jsonData)
{
    for (auto &opnode : jsonData["op_list"]) {
        if (opnode.find("input_desc") != opnode.end()) {
            for (auto &desc : opnode["input_desc"]) {
                desc.erase("name");
                desc.erase("L1_addr_offset");
            }
        }

        if (opnode.find("output_desc") != opnode.end()) {
            std::set<uint32_t> outputIdx;
            for (auto it = opnode["output_desc"].begin(); it != opnode["output_desc"].end();) {
                if (it->is_null()) {
                    ++it;
                    continue;
                }
                std::string outputName = (*it)["name"];
                it->erase("name");
                it->erase("L1_addr_offset");
                if (opnode["type"] == "Data") {
                    ++it;
                    continue;
                }

                // In case of multi referenced output node, keep only one reference.
                if (TeJsonUtils::CheckIfOutputNode(outputName, oriJson)) {
                    auto res = outputIdx.emplace((*it)["output_index"]);
                    if (!res.second) {
                        it = opnode["output_desc"].erase(it);
                        continue;
                    }
                }
                ++it;
            }

            for (auto &desc : opnode["output_desc"]) {
                if (desc.is_null()) {
                    continue;
                }
                desc.erase("name");
            }
        }

        if (opnode.find("hashed_extra_params") != opnode.end()) {
            opnode.erase("extra_params");
        }

        opnode.erase("id");

        if (opnode.find("name") == opnode.end()) {
            TE_ERRLOG("Json data contains no 'name' \n%s", jsonData.dump().c_str());
            return false;
        }

        std::string opName = opnode["name"];
        opnode.erase("name");
        if (opnode["type"] == "Data") {
            continue;
        }

        opnode.erase("ori_name");
        opnode.erase("py_module_path");
    }
    return true;
}

bool TeJsonAssemble::GenerateOpJson(const ge::NodePtr &nodePtr, const TbeOpInfo &tbeOpInfo, std::string &jsonStr)
{
    TE_DBGLOG("Generate op json for node [%s, %s]", nodePtr->GetName().c_str(), nodePtr->GetType().c_str());

    std::unordered_set<ge::Node *> allNodes;
    allNodes.emplace(nodePtr.get());

    std::string keyName;
    bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(nodePtr.get(), keyName);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to get node key name.");
        return false;
    });

    TE_DBGLOG("Process node begin, current node name is [%s]", keyName.c_str());

    // record nodePtr info in json
    json currentNodeJson;
    // get op desc
    auto opDesc = nodePtr->GetOpDesc();
    // get basic params from currentNodeOpDesc
    currentNodeJson["name"] = keyName;
    currentNodeJson["type"] = opDesc->GetType();
    currentNodeJson["id"] = opDesc->GetId();
    currentNodeJson["is_dynamic_impl"] = tbeOpInfo.IsDynamicImpl();

    std::string extraParams;
    tbeOpInfo.GetExtraParams(extraParams);
    currentNodeJson["extra_params"] = extraParams;
    TE_DBGLOG("OpNode: %s, ExtraParams: %s.", keyName.c_str(), extraParams.c_str());

    // get input desc json
    json inputJson;
    std::map<string, int> allInputNames;
    ConstTbeOpInfoPtr tbeOpInfoPtr = std::make_shared<const TbeOpInfo>(tbeOpInfo);
    std::vector<std::string> opTypeList;
    opTypeList.emplace_back(opDesc->GetType());
    std::vector<ge::Node *> nodes;
    nodes.emplace_back(nodePtr.get());
    InOutToJsonParam inputPara(keyName, 0, opTypeList, tbeOpInfoPtr, nodePtr.get());
    bres = GenBuildinIndescJson(inputPara, allNodes, inputJson, allInputNames, nodes);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to get builtin op input desc json, op name is [%s].", keyName.c_str());
        return false;
    });
    TeJsonUtils::DeleteValuesFromJson("caxis_values", inputJson);
    TeJsonUtils::DeleteValuesFromJson("ddr_base_prop", inputJson);
    currentNodeJson["input_desc"] = inputJson;
    InOutToJsonParam outputJsonPara(keyName, 0, opTypeList, tbeOpInfoPtr, nodePtr.get());
    GenBuildinOutdescJson(outputJsonPara, allNodes, currentNodeJson, nodes);
    TeJsonUtils::DeleteValuesFromJson("atomic_type", currentNodeJson["output_desc"]);
    TeJsonUtils::DeleteValuesFromJson("caxis_values", currentNodeJson["output_desc"]);
    TeJsonUtils::DeleteValuesFromJson("ddr_base_prop", currentNodeJson["output_desc"]);
    GenBuildinAttrsJson(tbeOpInfo, currentNodeJson);
    std::string opModuleName;
    std::string opFuncName;
    (void)tbeOpInfo.GetModuleName(opModuleName);
    (void)tbeOpInfo.GetFuncName(opFuncName);
    currentNodeJson["func_name"] = opFuncName;

    std::string pyModulePath;
    TE_FUSION_CHECK((!PythonApiCall::Instance().UpdateSingleOpModule(tbeOpInfo, opModuleName, &pyModulePath)), {
        TE_ERRLOG("Failed to update and import single op module for %s.", keyName.c_str());
        return false;
    });
    currentNodeJson["module_name"] = opModuleName;
    currentNodeJson["py_module_path"] = pyModulePath;

    // create a json to save all info and write it into file
    json jsonData;
    jsonData["op_list"].push_back(currentNodeJson);
    jsonData["fusion_op_name"] = keyName;
    jsonStr = jsonData.dump();
    TE_DBGLOG("Generated op [%s] json data successfully.", nodePtr->GetName().c_str());
    return true;
}

bool TeJsonAssemble::GenerateOptionsMap(
    const std::vector<ConstTbeOpInfoPtr> &tbeOpInfoVec,
    std::map<std::string, std::string> &options)
{
    if (tbeOpInfoVec.empty()) {
        TE_WARNLOG("Tbe op info vector is empty.");
        return false;
    }
    std::string opDebugConfig;
    std::string coreType;
    for (ConstTbeOpInfoPtr tbeOpInfo : tbeOpInfoVec) {
        if (!tbeOpInfo->GetOpDebugConfig().empty() && opDebugConfig.empty()) {
            opDebugConfig = tbeOpInfo->GetOpDebugConfig();
        }
        if (coreType.empty() || tbeOpInfo->GetOpCoreType() == AI_CORE) {
            coreType = tbeOpInfo->GetOpCoreType();
        }
    }
    if (opDebugConfig.empty() && (TeConfigInfo::Instance().GetEnvSaveKernelMeta() == "1")) {
        opDebugConfig = "dump_cce,ccec_g";
    }
    TE_DBGLOG("Core type and op debug config are [%s] and [%s].", coreType.c_str(), opDebugConfig.c_str());
    ConstTbeOpInfoPtr firstTbeOpInfo = tbeOpInfoVec.at(0);
    std::string coreNum;
    (void)firstTbeOpInfo->GetOption(ge::AICORE_NUM, coreNum);

    options.emplace("socVersion", TeConfigInfo::Instance().GetSocVersion());
    options.emplace("coreType", coreType);
    options.emplace("coreNum", coreNum);
    options.emplace("l1Fusion", TeConfigInfo::Instance().GetL1Fusion());
    options.emplace("l2Fusion", TeConfigInfo::Instance().GetL2Fusion());
    options.emplace("l2Mode", TeConfigInfo::Instance().GetL2Mode());
    options.emplace("op_debug_level", TeConfigInfo::Instance().GetOpDebugLevelStr());
    options.emplace("op_debug_config", opDebugConfig);
    options.emplace("op_debug_dir", TeConfigInfo::Instance().GetKernelMetaParentDir());
    options.emplace("mdl_bank_path", TeConfigInfo::Instance().GetMdlBankPath());
    options.emplace("op_bank_path", TeConfigInfo::Instance().GetOpBankPath());
    options.emplace("status_check", TeContextUtils::GetStatusCheck());
    options.emplace("need_precompile", firstTbeOpInfo->IsNeedPreCompile() ? "true" : "false");
    options.emplace("deterministic", TeContextUtils::GetDeterministic());
    options.emplace("device_id", TeConfigInfo::Instance().GetDeviceId());
    options.emplace("deterministic_level", TeContextUtils::GetDeterministicLevel());
    TE_INFOLOG("Get deterministic_level as %s", TeContextUtils::GetDeterministicLevel().c_str());

    if (tbeOpInfoVec.size() == 1) {
        if (firstTbeOpInfo->GetNode() != nullptr) {
            std::string opBuildConfig;
            (void)ge::AttrUtils::GetStr(firstTbeOpInfo->GetNode()->GetOpDesc(), FUSION_OP_BUILD_OPTIONS, opBuildConfig);
            options.emplace("single_op_build_cfg", opBuildConfig);
        }
    }

    std::map<std::string, std::string> hardwareInfoMap;
    TeConfigInfo::Instance().GetHardwareInfoMap(hardwareInfoMap);
    for (const std::pair<const std::string, std::string> &hardwareItem : hardwareInfoMap) {
        options.emplace(hardwareItem.first, hardwareItem.second);
    }

    bool isValidOpInfo = (firstTbeOpInfo != nullptr) && (firstTbeOpInfo->GetNode() != nullptr) &&
                         (firstTbeOpInfo->GetNode()->GetOpDesc() != nullptr);
    if (isValidOpInfo) {
        std::string custAicNum;
        (void)ge::AttrUtils::GetStr(firstTbeOpInfo->GetNode()->GetOpDesc(), kAicCntKeyOp, custAicNum);
        if (custAicNum != "" && std::stoi(custAicNum) >= 0) {
            options[kAiCoreCnt] = custAicNum;
            options[kCubeCoreCnt] = custAicNum;
            TE_DBGLOG("OpNode: %s, set options.ai_core_cnt: %s", firstTbeOpInfo->GetName().c_str(), custAicNum.c_str());
        }
        std::string custAivNum;
        (void)ge::AttrUtils::GetStr(firstTbeOpInfo->GetNode()->GetOpDesc(), kAivCntKeyOp, custAivNum);
        if (custAivNum != "" && std::stoi(custAivNum) >= 0) {
            options[kVectorCoreCnt] = custAivNum;
            TE_DBGLOG("OpNode: %s, set options.vector_core_cnt: %s", firstTbeOpInfo->GetName().c_str(), custAivNum.c_str());
        }
    }
    return true;
}
}
}
