/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fusion_common.h"
#include <sstream>
#include <unistd.h>
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "register/register_base.h"

namespace te {
namespace fusion {
using namespace ge;
namespace {
constexpr const char *VARIABLE_ATTR = "variable_attr";
const std::string kCUSTOM_OPP_PATH = "custom_opp_path";
}
bool CheckMemoryL1MemoryL2(const std::string &opType, const std::vector<TbeOpParam> &inPutsOrOutPuts)
{
    for (size_t i = 0; i < inPutsOrOutPuts.size(); i++) {
        const std::vector<TbeOpTensor> &tensors = inPutsOrOutPuts.at(i).GetTensors();
        if (tensors.size() == 0) {
            continue;
        }
        for (const TbeOpTensor &tensor : tensors) {
            size_t addrType = tensor.GetAddrType();
            if (addrType == RT_MEMORY_L1 || addrType == RT_MEMORY_L2) {
                TE_INFOLOG("opType [%s], index[%d].addrType = [%d].", opType.c_str(), i, addrType);
                return true;
            }
        }
    }
    return false;
}

std::string GetInputNodesDesc(const ge::Node &node)
{
    std::ostringstream inNodesStream;
    uint32_t inputSize = node.GetAllInDataAnchorsSize();
    for (uint32_t i = 0; i < inputSize; ++i) {
        inNodesStream << "Input(" << i <<")" << " - (";
        ge::InDataAnchorPtr inDataAnchor = node.GetInDataAnchor(static_cast<int32_t>(i));
        if (inDataAnchor == nullptr || inDataAnchor->GetPeerOutAnchor() == nullptr ||
            inDataAnchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
            inNodesStream << "empty)";
            continue;
        }
        ge::NodePtr peerInNode = inDataAnchor->GetPeerOutAnchor()->GetOwnerNode();
        inNodesStream << peerInNode->GetName() << ", " << peerInNode->GetType() << ")";
    }
    return inNodesStream.str();
}

std::string GetNodesName(const std::vector<ge::Node *> &nodes)
{
    if (nodes.empty()) {
        return "None";
    }
    std::ostringstream nodesNameStream;
    for (const ge::Node *node : nodes) {
        if (node == nullptr) {
            continue;
        }
        nodesNameStream << "(" << node->GetName() << ", " << node->GetType() << ")";
    }
    return nodesNameStream.str();
}

PyObject* GenerateDictFromContext()
{
    PyObject *pyContextDict = HandleManager::Instance().TE_PyDict_New();
    if (pyContextDict == nullptr) {
        TE_ERRLOG("Failed to create py contextDict.");
        return nullptr;
    }

    TE_FUSION_CHECK(!SetContextItem(pyContextDict, OPT_MODULE_OP_TUNE, OPT_SWITCH_OP_TUNE, OPT_KEY_LIST_OP_TUNE), {
        TE_ERRLOG("Failed to set context value [%s].", OPT_MODULE_OP_TUNE.c_str());
        return nullptr;
    });

    TE_FUSION_CHECK(!SetContextItem(pyContextDict, OPT_MODULE_RL_TUNE, OPT_SWITCH_RL_TUNE, OPT_KEY_LIST_RL_TUNE), {
        TE_ERRLOG("Failed to set context value [%s].", OPT_MODULE_RL_TUNE.c_str());
        return nullptr;
    });

    const char* custom_opp_path = aclGetCustomOpLibPath();
    TE_FUSION_CHECK(!SetContextItem(pyContextDict, kCUSTOM_OPP_PATH, std::string(custom_opp_path)), {
        TE_ERRLOG("Failed to set context value [%s].", custom_opp_path);
        return nullptr;
    });
    return pyContextDict;
}

bool SetContextItem(PyObject *pyContextDict, const std::string &optKey, const std::string &custom_opp_path) {
    PyObject *pyOpTList = HandleManager::Instance()._Py_BuildValue("s", custom_opp_path.c_str());
    AUTO_PY_DECREF(pyOpTList);
    int ret = HandleManager::Instance().TE_PyDict_SetItemString(pyContextDict, optKey.c_str(), pyOpTList);
    TE_FUSION_CHECK((ret != 0), {
        TE_ERRLOG("Build PyDict_SetItemString[%s] failed.", optKey.c_str());
        return false;
    });
    return true;
}

void GenerateExtraInfoForFusionOpContext(const std::vector<ge::Node *> &oriNodes, PyObject *pyContextDict) {
    std::string maxCustAicNum = "-1";
    std::string maxCustAivNum = "-1";
    for (const auto &node : oriNodes) {
        const auto opDescPtr = node->GetOpDesc();
        std::string custAicNum;
        ge::AttrUtils::GetStr(opDescPtr, kAicCntKeyOp, custAicNum);
        if (custAicNum != "" && std::stoi(custAicNum) >= 0  && std::stoi(custAicNum) > std::stoi(maxCustAicNum)) {
            maxCustAicNum = custAicNum;
        }
        std::string custAivNum;
        ge::AttrUtils::GetStr(opDescPtr, kAivCntKeyOp, custAivNum);
        if (custAivNum != "" && std::stoi(custAivNum) >= 0 && std::stoi(custAivNum) > std::stoi(maxCustAivNum)) {
            maxCustAivNum = custAivNum;
        }
    }

    if (maxCustAicNum == "-1" && maxCustAivNum == "-1") {
        TE_INFOLOG("Both custAicNum and custAivNum attr are empty, skipping set pyContextDict");
        return;
    }

    TE_INFOLOG("maxCustAicNum is [%s], maxCustAivNum is [%s]", maxCustAicNum.c_str(), maxCustAivNum.c_str());

    TE_FUSION_CHECK(!SetContextItem(pyContextDict, kAicCntKeyOp, maxCustAicNum), {
        TE_ERRLOG("Failed to set context value [%s].", kAicCntKeyOp);
    });

    TE_FUSION_CHECK(!SetContextItem(pyContextDict, kAivCntKeyOp, maxCustAivNum), {
        TE_ERRLOG("Failed to set context value [%s].", kAivCntKeyOp);
    });
}

void GenerateExtraInfoForContext(const TbeOpInfo &opInfo, const ge::OpDescPtr &opDescPtr, PyObject *pyContextDict)
{
    std::string custAicNum;
    ge::AttrUtils::GetStr(opDescPtr, kAicCntKeyOp, custAicNum);
    if (custAicNum != "" && std::stoi(custAicNum) >= 0) {
        TE_INFOLOG("Get custAicNum attr op[%s, %s], custAicNum=%s", opDescPtr->GetNamePtr(), opDescPtr->GetTypePtr(),
                   custAicNum.c_str());
        TE_FUSION_CHECK(!SetContextItem(pyContextDict, kAicCntKeyOp, custAicNum), {
            TE_ERRLOG("Failed to set context value [%s].", kAicCntKeyOp);
        });
    }
    std::string custAivNum;
    ge::AttrUtils::GetStr(opDescPtr, kAivCntKeyOp, custAivNum);
    if (custAivNum != "" && std::stoi(custAivNum) >= 0) {
        TE_INFOLOG("Get custAivNum attr op%s(%s), custAivNum=%s", opDescPtr->GetNamePtr(), opDescPtr->GetTypePtr(),
                   custAivNum.c_str());
        TE_FUSION_CHECK(!SetContextItem(pyContextDict, kAivCntKeyOp, custAivNum), {
            TE_ERRLOG("Failed to set context value [%s].", kAivCntKeyOp);
        });
    }
    return;
}

bool SetContextItem(PyObject *pyContextDict, const std::string &contextKey,
                    const std::string &switchKey, const std::string &optKey)
{
    std::string optList;
    if (ge::GetContext().GetOption(contextKey, optList) == ge::GRAPH_SUCCESS) {
        int ret = 0;
        if (optList.empty()) {
            PyObject *pySwitchOff = HandleManager::Instance()._Py_BuildValue("s", SWITCH_OFF.c_str());
            AUTO_PY_DECREF(pySwitchOff);
            ret = HandleManager::Instance().TE_PyDict_SetItemString(pyContextDict, switchKey.c_str(), pySwitchOff);
            TE_FUSION_CHECK((ret != 0), {
                TE_ERRLOG("Build PyDict_SetItemString[%s] failed.", switchKey.c_str());
                return false;
            });
        } else {
            PyObject *pySwitchOn = HandleManager::Instance()._Py_BuildValue("s", SWITCH_ON.c_str());
            AUTO_PY_DECREF(pySwitchOn);
            ret = HandleManager::Instance().TE_PyDict_SetItemString(pyContextDict, switchKey.c_str(), pySwitchOn);
            TE_FUSION_CHECK((ret != 0), {
                TE_ERRLOG("Build PyDict_SetItemString[%s] failed.", switchKey.c_str());
                return false;
            });

            PyObject *pyOpTList = HandleManager::Instance()._Py_BuildValue("s", optList.c_str());
            AUTO_PY_DECREF(pyOpTList);
            ret = HandleManager::Instance().TE_PyDict_SetItemString(pyContextDict, optKey.c_str(), pyOpTList);
            TE_FUSION_CHECK((ret != 0), {
                TE_ERRLOG("Build PyDict_SetItemString[%s] failed.", optKey.c_str());
                return false;
            });
        }
    } else {
        TE_DBGLOG("Do not obtain [%s] param from ge context.", contextKey.c_str());
    }
    return true;
}

void SetContextItem(std::map<std::string, std::string> &contextDict, const std::string &contextKey,
                    const std::string &switchKey, const std::string &optKey)
{
    std::string optList;
    if (ge::GetContext().GetOption(contextKey, optList) == ge::GRAPH_SUCCESS) {
        if (optList.empty()) {
            contextDict.emplace(switchKey, SWITCH_OFF);
        } else {
            contextDict.emplace(switchKey, SWITCH_ON);
            contextDict.emplace(optKey, optList);
        }
    } else {
        TE_DBGLOG("Do not obtain [%s] param from ge context.", contextKey.c_str());
    }
}

void GenerateContextDict(std::map<std::string, std::string> &contextDict)
{
    SetContextItem(contextDict, OPT_MODULE_OP_TUNE, OPT_SWITCH_OP_TUNE, OPT_KEY_LIST_OP_TUNE);
    SetContextItem(contextDict, OPT_MODULE_RL_TUNE, OPT_SWITCH_RL_TUNE, OPT_KEY_LIST_RL_TUNE);
}

void SetLicensePassOptList(std::string &passOptList)
{
    if (ge::GetContext().GetOption(OPT_MODULE_PASS, passOptList) != ge::GRAPH_SUCCESS) {
        passOptList = "invalid";
    }

    TE_DBGLOG("passOptList[%s]", passOptList.c_str());
}

bool GetCheckNodeInfo(const ge::OpDescPtr &opDescPtr, CheckNodeInfo &checkNodeInfo)
{
    if (opDescPtr == nullptr) {
        TE_ERRLOG("CheckNodeList:Op desc is null.");
        return false;
    }

    if (opDescPtr->HasAttr(ATTR_NAME_L1_FUSION_SCOPE)) {
        TE_FUSION_CHECK(!(AttrUtils::GetInt(opDescPtr, ATTR_NAME_L1_FUSION_SCOPE, checkNodeInfo.scopeId)), {
            TE_ERRLOG("CheckNodeList:Get the attribute l1 scope id from op description [%s] failed.",
                      opDescPtr->GetName().c_str());
            return false;
        });
    } else if (opDescPtr->HasAttr(ATTR_NAME_FUSION_SCOPE)) {
        TE_FUSION_CHECK(!(AttrUtils::GetInt(opDescPtr, ATTR_NAME_FUSION_SCOPE, checkNodeInfo.scopeId)), {
            TE_ERRLOG("CheckNodeList:Get the attribute scope id from op description [%s] failed.",
                      opDescPtr->GetName().c_str());
            return false;
        });
    }

    if (opDescPtr->HasAttr(TVM_ATTR_NAME_MAGIC)) {
        TE_FUSION_CHECK(!(AttrUtils::GetStr(opDescPtr, TVM_ATTR_NAME_MAGIC, checkNodeInfo.backend)), {
            TE_ERRLOG("CheckNodeList:Get the attribute backend from op description [%s] failed.",
                      opDescPtr->GetName().c_str());
            return false;
        });
    }

    if (opDescPtr->HasAttr(ATTR_NAME_GRAPH_NAME)) {
        TE_FUSION_CHECK(!(AttrUtils::GetStr(opDescPtr, ATTR_NAME_GRAPH_NAME, checkNodeInfo.graphName)), {
            TE_ERRLOG("Get the attribute graph_name from op description [%s] failed.",
                      opDescPtr->GetName().c_str());
            return false;
        });
    }

    return true;
}

bool VerifyNodeInfo(const CheckNodeInfo &currentCheckNodeInfo, const CheckNodeInfo &checkNodeInfo)
{
    TE_FUSION_CHECK((currentCheckNodeInfo.backend != checkNodeInfo.backend), {
        REPORT_TE_INNER_ERROR("Op backend is not consistent, current is [%s], first node is [%s]",
                              currentCheckNodeInfo.backend.c_str(), checkNodeInfo.backend.c_str());
        return false;
    });

    TE_FUSION_CHECK((currentCheckNodeInfo.scopeId != checkNodeInfo.scopeId), {
        REPORT_TE_INNER_ERROR("Op scope id is not consistent, current is [%ld], first node is [%ld]",
                              currentCheckNodeInfo.scopeId, checkNodeInfo.scopeId);
        return false;
    });

    TE_FUSION_CHECK((currentCheckNodeInfo.graphName != checkNodeInfo.graphName), {
        REPORT_TE_INNER_ERROR("Op graph name is not consistent, current is [%s], first node is [%s]",
                              currentCheckNodeInfo.graphName.c_str(), checkNodeInfo.graphName.c_str());
        return false;
    });

    return true;
}

/**
 * @brief: check node list's backend and scope
 */
bool CheckNodeList(const std::vector<Node *> &teGraphNode)
{
    CheckNodeInfo checkNodeInfo;
    std::vector<std::string> verifyOpType;
    return CheckNodeList(teGraphNode, checkNodeInfo, verifyOpType);
}

bool CheckNodeList(const std::vector<Node *> &teGraphNode, CheckNodeInfo &checkNodeInfo,
                   std::vector<std::string> &verifyOpType)
{
    // func start log
    TE_DBGLOG("Start to check node list");

    // check NodeList
    // include:
    // 1. if scope is consistent
    // 2. if backend is consistent
    // 3. if graph name is consistent
    // 4. if is dynamic impl is consistent
    TE_FUSION_CHECK((teGraphNode.empty()), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Graph node list is empty.");
        return false;
    });
    std::unordered_map<std::string, uint32_t> opTypeCounter;
    bool isFirstNode = true;
    for (const Node *currentNode : teGraphNode) {
        // check whether current node is nullptr
        TE_FUSION_CHECK((currentNode == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get current node.");
            return false;
        });

        // get opDef
        auto currentNodeOpDesc = currentNode->GetOpDesc();
        CheckNodeInfo currentCheckNodeInfo;
        TE_FUSION_CHECK(!GetCheckNodeInfo(currentNodeOpDesc, currentCheckNodeInfo), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get op check info.");
            return false;
        });
        std::string currentOpType = currentNodeOpDesc->GetType();
        auto iter = opTypeCounter.find(currentOpType);
        if (iter == opTypeCounter.end()) {
            opTypeCounter[currentOpType] = 0;
            verifyOpType.emplace_back(currentOpType + "0");
        } else {
            ++opTypeCounter[currentOpType];
            verifyOpType.emplace_back(currentOpType + std::to_string(opTypeCounter[currentOpType]));
        }
        if (isFirstNode) {
            checkNodeInfo = currentCheckNodeInfo;
            isFirstNode = false;
        } else {
            TE_FUSION_CHECK(!VerifyNodeInfo(currentCheckNodeInfo, checkNodeInfo), {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Fail to verify node info.");
                return false;
            });
        }
    }

    // func end log
    TE_DBGLOG("Check node list success");
    return true;
}

void GetVariableAttrValue(const TbeOpInfo &opInfo, std::vector<std::string> &variableAttrs)
{
    ge::NodePtr node = opInfo.GetNode();
    if (node != nullptr) {
        auto opDesc = node->GetOpDesc();
        if (opDesc != nullptr) {
            (void)ge::AttrUtils::GetListStr(opDesc, VARIABLE_ATTR, variableAttrs);
            TE_DBGLOGF("Get node[%s] variableAttrValue:%s.", node->GetName().c_str(),
                       GetVectorValueToString(variableAttrs).c_str());
        }
    }
}

bool JudgeAttrIsVariableAttr(const TbeAttrValue &attrValue, const std::vector<std::string> &variableAttrs)
{
    if (variableAttrs.empty()) {
        return false;
    }
    auto it = std::find(variableAttrs.cbegin(), variableAttrs.cend(), attrValue.GetName());
    if (it == variableAttrs.cend()) {
        return false;
    }
    return true;
}

std::string GetSessionGraphId(const ge::Node *node)
{
    std::string keyStr = ATTR_NAME_SESSION_GRAPH_ID;
    std::string sessionGraphId = "";
    ge::ComputeGraphPtr ownerGraph = node->GetOwnerComputeGraph();
    if (ownerGraph == nullptr) {
        TE_WARNLOG("Owner graph of node[%s] is null.", node->GetName().c_str());
        return "";
    }
    if (ge::AttrUtils::GetStr(ownerGraph, keyStr, sessionGraphId) && !sessionGraphId.empty()) {
        TE_DBGLOG("Get sessionGraphId from ownerGraph");
        return sessionGraphId;
    }
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), keyStr, sessionGraphId) && !sessionGraphId.empty()) {
        TE_DBGLOG("Get sessionGraphId from OpDesc");
        return sessionGraphId;
    }
    return sessionGraphId;
}

/*
 * @brief: get op Node input key name from session_id + input Name
 * @param [in] Node: op node
 * @param [out] keyName: session_id + op input Name
 * @return bool: get input key name ok or not
 */
bool GetOpInputKeyName(const ge::Node *node, std::vector<std::string> &key_name)
{
    TE_FUSION_CHECK(node == nullptr, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Node is null.");
        return false;
    });

    std::string sessionGraphId = GetSessionGraphId(node);
    if (!sessionGraphId.empty()) {
        sessionGraphId += "_";
    }

    TE_DBGLOG("sessionGraphId is: [%s]", sessionGraphId.c_str());

    const std::string op_name = node->GetName();
    // get op desc
    const auto &all_in_anchors = node->GetAllInDataAnchors();
    for (const auto &anchor : all_in_anchors) {
        const auto &peer_anchor = anchor->GetPeerOutAnchor();
        if (peer_anchor == nullptr) {
            TE_DBGLOG("The anchor has no peer. node: %s", op_name.c_str());
            continue;
        }
        const ge::ConstNodePtr &src_node = peer_anchor->GetOwnerNode();
        if (src_node == nullptr) {
            TE_WARNLOG("The anchor has no owner node. node: %s", op_name.c_str());
            continue;
        }
        const std::string src_node_name = src_node->GetName();
        auto src_idx = peer_anchor->GetIdx();
        // The op input name should be matching connected op's output name
        key_name.push_back(sessionGraphId + src_node_name + "__" + std::to_string(src_idx));
    }
    return true;
}

bool TbeAttrDtypeToString(const ATTR_DTYPE &dtype, std::string &res)
{
    auto iter = TbeAttrDtypeToStringMap.find(dtype);
    if (iter == TbeAttrDtypeToStringMap.end()) {
        REPORT_TE_INNER_ERROR("Value dtype[%d] is invalid, not in enum ATTR_DTYPE.", dtype);
        return false;
    }
    res = iter->second;
    return true;
}

bool HasCustomOp(const std::vector<ge::Node *> &nodes)
{
    bool isCustomOp = false;
    for (const auto &currentNode : nodes) {
        TE_FUSION_CHECK(currentNode == nullptr, continue);
        if (ge::AttrUtils::GetBool(currentNode->GetOpDesc(), "_is_custom_op", isCustomOp)) {
            if (isCustomOp) {
                TE_DBGLOGF("Node[%s] is custom op.", currentNode->GetName().c_str());
                return true;
            }
        } else {
            TE_WARNLOGF("Node[%s] has no _is_custom_op attribute.", currentNode->GetName().c_str());
            return false;
        }
    }
    return false;
}

std::string GetTaskNodeName(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() == 1 && opTask->opNodes[0] != nullptr) {
        return opTask->opNodes[0]->GetName();
    }
    if (opTask->opNodes.size() > 1) {
        std::string fusionName("fusion");
        for (size_t index = 0; index < opTask->opNodes.size(); index++) {
            if (opTask->opNodes[index] != nullptr) {
                fusionName.append("_");
                fusionName.append(opTask->opNodes[index]->GetName());
            }
        }
        return fusionName;
    }
    return "";
}

void UpdateOpModulePath(const std::string &opsPathNamePrefix, std::string &opModule, size_t pos, bool flag) {
    std::string opModulePrefixPart = opModule.substr(0, pos);
    std::string opModuleSuffixPart = opModule.substr(pos + 1, opModule.size() - 1);
    if (flag) {
        if (opsPathNamePrefix == "") {
            opModule = opModulePrefixPart + "/dynamic/" + opModuleSuffixPart;
        } else {
            opModule = opModulePrefixPart + "/" + opsPathNamePrefix + "/dynamic/" + opModuleSuffixPart;
        }
    } else {
        if (opsPathNamePrefix == "") {
            opModule = opModulePrefixPart + ".dynamic." + opModuleSuffixPart;
        } else {
            opModule = opModulePrefixPart + "." + opsPathNamePrefix + ".dynamic." + opModuleSuffixPart;
        }
    }
}

void UpdateStaticOpModulePath(const std::string &opsPathNamePrefix, std::string &opModule, size_t pos, bool flag) {
    std::string opModulePrefixPart = opModule.substr(0, pos);
    std::string opModuleSuffixPart = opModule.substr(pos + 1, opModule.size() - 1);
    if (flag) {
        if (opsPathNamePrefix == "") {
            opModule = opModulePrefixPart + "/" + opModuleSuffixPart;
        } else {
            opModule = opModulePrefixPart + "/" + opsPathNamePrefix + "/" + opModuleSuffixPart;
        }
    } else {
        if (opsPathNamePrefix == "") {
            opModule = opModulePrefixPart + "." + opModuleSuffixPart;
        } else {
            opModule = opModulePrefixPart + "." + opsPathNamePrefix + "." + opModuleSuffixPart;
        }
    }
}

void UpdateOpModuleFromStaicToDynamicAndAddPrefix(const TbeOpInfo &tbeOpInfo, std::string &opModule)
{
    std::string opsPathNamePrefix = tbeOpInfo.GetOpsPathNamePrefix();
    // autofuse
    if (tbeOpInfo.GetOpFileName() == "asc_codegen_compile") {
        return;
    }

    size_t posLine = opModule.find_last_of('/');
    size_t posDot = opModule.find_last_of('.');
    if (tbeOpInfo.IsDynamicImpl() || tbeOpInfo.GetIsUnknownShape()) {
        if (posLine != string::npos && posDot != string::npos) {
            if (posLine > posDot) {
                UpdateOpModulePath(opsPathNamePrefix, opModule, posLine, true);
            } else {
                UpdateOpModulePath(opsPathNamePrefix, opModule, posDot, false);
            }
        } else {
            if (posLine != string::npos) {
                UpdateOpModulePath(opsPathNamePrefix, opModule, posLine, true);
                return;
            }
            if (posDot != string::npos) {
                UpdateOpModulePath(opsPathNamePrefix, opModule, posDot, false);
                return;
            }
        }
    } else {
        if (posLine != string::npos && posDot != string::npos) {
            if (posLine > posDot) {
                UpdateStaticOpModulePath(opsPathNamePrefix, opModule, posLine, true);
            } else {
                UpdateStaticOpModulePath(opsPathNamePrefix, opModule, posDot, false);
            }
        } else {
            if (posLine != string::npos) {
                UpdateStaticOpModulePath(opsPathNamePrefix, opModule, posLine, true);
                return;
            }
            if (posDot != string::npos) {
                UpdateStaticOpModulePath(opsPathNamePrefix, opModule, posDot, false);
                return;
            }
        }
    }
}

/**
 * @brief check op parameter
 * @param [in] opName         opname, need unique for fusion subgraph
 * @param [in] opModule       op module name
 * @param [in] opFuncName     op function name
 * @return [out] bool         the check result
 */
bool IsOpParameterValid(const std::string &opModule, const std::string &opFuncName)
{
    TE_FUSION_CHECK(!(IsNameValid(opModule, "/.")), {
        TE_ERRLOG("Check opModule failed.");
        return false;
    });
    TE_FUSION_CHECK(!(IsNameValid(opFuncName, "")), {
        TE_ERRLOG("Check opFuncName failed.");
        return false;
    });

    TE_FUSION_CHECK((opModule.find("//", 0) != opModule.npos), {
        REPORT_TE_INNER_ERROR("OpModule[%s] has 2 or more '/', check error.", opModule.c_str());
        return false;
    });
    TE_FUSION_CHECK((opModule.find("..", 0) != opModule.npos), {
        REPORT_TE_INNER_ERROR("OpModule[%s] is invalid, check error.", opModule.c_str());
        return false;
    });

    return true;
}

void OriginalOpNamesSplicing(const std::vector<std::string> &originalOpNames, std::string &opNameStr)
{
    uint32_t idx = 0;
    for (const auto &opName : originalOpNames) {
        if (!opName.empty()) {
            if (idx == 0) {
                opNameStr = opName;
            }
            opNameStr = opNameStr + "," + opName;
        }
        idx++;
    }
    if (opNameStr != "") {
        opNameStr = "[" + opNameStr + "]";
    }
}

/*
 * @brief: get Node's list string name of attr of _datadump_original_op_names
 * @param [in] teGraphNode: teGraphNode
 * @param [in] opNames: all opname in Model transformation
 */
void GetNodeListStr(const std::vector<ge::Node *> &teGraphNode, std::string &opNames)
{
    if (teGraphNode.size() < 1) {
        return;
    }
    uint32_t nodeIdx = 0;
    std::string opNameStr;
    for (auto pNode : teGraphNode) {
        std::string nodeName;
        std::vector<std::string> originalOpNames;
        TE_FUSION_CHECK(pNode == nullptr, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Node is null.");
            continue;
        });
        nodeName = pNode->GetName();
        if (nodeIdx != 0) {
            opNames = opNames + " | ";
        }
        opNames = opNames + nodeName;
        bool result = ge::AttrUtils::GetListStr(pNode->GetOpDesc(),
                                                ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, originalOpNames);
        TE_FUSION_CHECK(!result, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to get node originalOpNames.");
            continue;
        });
        opNameStr = "";
        OriginalOpNamesSplicing(originalOpNames, opNameStr);
        if (!opNameStr.empty()) {
            opNames = opNames + "," + opNameStr;
        }
        nodeIdx++;
    }
}

ge::Node *GetPreviousNode(const ge::Node *node, const uint32_t index)
{
    if (node == nullptr) {
        return nullptr;
    }
    ge::InDataAnchorPtr inAnchor = node->GetInDataAnchor(static_cast<int32_t>(index));
    if (node->GetInDataAnchor(index) == nullptr || node->GetInDataAnchor(index)->GetPeerOutAnchor() == nullptr ||
        node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
        return nullptr;
    }

    return node->GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode().get();
}

bool GetSubOpLoc(int64_t skCount, int64_t skSubId, std::string &locStr) {
    if (skSubId == 0) {
        locStr = "start";
    } else if (skSubId > 0 && skSubId < (skCount - 1)) {
        locStr = "middle";
    } else if (skSubId == (skCount - 1)) {
        locStr = "end";
    }
    return !locStr.empty();
}
} // namespace fusion
} // namespace te
