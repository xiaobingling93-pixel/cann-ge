/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <Python.h>
#include <set>
#include <map>
#include <functional>
#include <utility>
#include <sstream>
#include <climits>
#include <tuple>
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <random>
#include <ctime>
#include <iomanip>
#include <functional>
#include <cstdio>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include "compile/fusion_manager.h"

#include "inc/te_fusion_check.h"
#include "inc/te_fusion_error_code.h"
#include "python_adapter/py_decouple.h"
#include "python_adapter/python_api_call.h"
#include "python_adapter/python_adapter_manager.h"
#include "tensor_engine/fusion_api.h"
#include "binary/binary_manager.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_config_info.h"
#include "common/te_context_utils.h"
#include "common/te_file_utils.h"
#include "common/compile_result_utils.h"
#include "common/signal_manager.h"
#include "cache/te_cache_manager.h"
#include "assemble_json/te_json_utils.h"
#include "assemble_json/te_json_assemble.h"
#include "graph/anchor.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/tuning_utils.h"
#include "compile/te_compile_task_cache.h"
#include "dfxinfo_manager/trace_utils.h"
#include "dfxinfo_manager/dfxinfo_manager.h"
#include "file_handle/te_file_handle.h"
#include "python_adapter/py_object_utils.h"
#include "python_adapter/pyobj_assemble_utils.h"
#include "compile/superkernel_task.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;

// params for singleton mode
std::mutex TeFusionManager::mtx_;
constexpr int DEL_ALL_TASKS = -1;
constexpr int MAX_DFS_CNT = 1000;
constexpr int MAX_LOOP_CNT = 200;

constexpr const int PLAIN_PREFIX_SIZE = 180;

constexpr size_t MAX_LENGTH = 900;
constexpr const char *TUNE_PARAM = "TUNE_PARAM";
const std::string ATTR_NAME_SUPPORT_PARAM_REUSED = "_param_reused";
const std::string ATTR_NAME_SUPPORT_RELATION_REUSED = "_param_reused_relation";

namespace {
void PrintOp(const OpBuildTaskPtr &relBuildTaskPtr, const OpBuildTaskResultPtr &taskRes,
             string &opNames, string &opTypes, int loglevel)
{
    size_t index = 0;
    if (relBuildTaskPtr->pPrebuildOp == nullptr) {
        size_t sizeOfNodes = relBuildTaskPtr->opNodes.size();
        for (auto node : relBuildTaskPtr->opNodes) {
            opNames += node->GetName();

            opTypes += node->GetType();
            if (index < sizeOfNodes - 1) {
                opNames += ", ";
                opTypes += ", ";
            }
            index++;
        }
    } else {
        (void)relBuildTaskPtr->pPrebuildOp->GetRealName(opNames);
        relBuildTaskPtr->pPrebuildOp->GetOpType(opTypes);
    }

    if (taskRes->errArgs.empty()) {
        return;
    }

    if (taskRes->type == 0) {
        TE_FUSION_LOG_FULL(loglevel, "Failed to pre-compile node [opName:%s, opType: %s].",
                           opNames.c_str(), opTypes.c_str());
        if (relBuildTaskPtr->pPrebuildOp != nullptr) {
            ge::NodePtr nodePtr = nullptr;
            relBuildTaskPtr->pPrebuildOp->GetNode(nodePtr);
            if (nodePtr != nullptr) {
                TE_FUSION_LOG_FULL(loglevel, "The input nodes of node [opName:%s, opType: %s] is: [%s].",
                                   opNames.c_str(), opTypes.c_str(), GetInputNodesDesc(*nodePtr).c_str());
            }
        }
    } else if (taskRes->type == 1) {
        TE_FUSION_LOG_FULL(loglevel, "Failed to compile node [opName %s, opType: %s].",
                           opNames.c_str(), opTypes.c_str());
        if (!relBuildTaskPtr->opNodes.empty() && relBuildTaskPtr->opNodes[0] != nullptr) {
            TE_FUSION_LOG_FULL(loglevel, "The input nodes of node[opName:%s, opType: %s] is: [%s].",
                               opNames.c_str(), opTypes.c_str(),
                               GetInputNodesDesc(*relBuildTaskPtr->opNodes[0]).c_str());
        }
    } else if (taskRes->type == OP_TASK_TYPE_FUSION) {
        TE_INFOLOG("Compile fusion nodes [opName: %s, opType: %s] not successfully.", opNames.c_str(), opTypes.c_str());
    }
    if (taskRes->type == 0 || taskRes->type == 1) {
        TE_FUSION_LOG_FULL(loglevel, "Failed to compile node [opName:%s, opType: %s].",
                           opNames.c_str(), opTypes.c_str());
    }
}

string& ReplaceAll(string& str, const string& oldValue, const string& newValue)
{
    string::size_type pos = 0;
    while (pos != string::npos && pos < str.length()) {
        pos = str.find(oldValue, pos);
        if (pos != string::npos) {
            str.replace(pos, oldValue.length(), newValue);
            pos += newValue.length();
        }
    }
    return str;
}

void GetEachTensor(const string& allTensors, vector<string> &tensorList)
{
    auto leftBracketPos = allTensors.find('{');
    int loopMaxCount = 0;

    while (leftBracketPos != string::npos) {
        if (loopMaxCount > MAX_LOOP_CNT) {
            return;
        }
        loopMaxCount++;
        auto rightBracketPos = allTensors.find('}', leftBracketPos);
        if (rightBracketPos == string::npos) {
            TE_ERRLOG("No matching right bracket for tensors %s", allTensors.c_str());
            return;
        }
        auto len = rightBracketPos - leftBracketPos + 1;
        tensorList.emplace_back(allTensors.substr(leftBracketPos, len));
        leftBracketPos = allTensors.find('{', rightBracketPos);
    }
}

void GetStringByKey(const string &errorArgs, const string& key, string &str)
{
    size_t argSize = errorArgs.size();
    size_t sizeOfKey = key.size();

    size_t keyStartPos = errorArgs.find(key);
    uint32_t numOfParentheses = 1;
    for (size_t index = (keyStartPos + sizeOfKey + 2); index < argSize; ++index) {
        if (errorArgs.at(index) == '(') {
            numOfParentheses += 1;
        } else if (errorArgs.at(index) == ')') {
            numOfParentheses -= 1;
        }

        if (numOfParentheses == 0) {
            break;
        }

        str += errorArgs.at(index);
    }
}

void PrintTensors(const string &errorArgs, const string &key, int loglevel)
{
    string allTensorString;
    GetStringByKey(errorArgs, key, allTensorString);
    vector<string> list;
    GetEachTensor(allTensorString, list);
    uint32_t index = 0;
    for (auto &tensor : list) {
        ReplaceAll(tensor, "None", "{None}");
        string tensorName = key + std::to_string(index);
        TE_FUSION_LOG(loglevel, "%s: %s", tensorName.c_str(), tensor.c_str());
        index++;
    }
}

void ParseAndPrintArgsString(const string& opNames, const string& opTypes,
                             const OpBuildTaskResultPtr &taskRes, int loglevel)
{
    if (taskRes->errArgs.empty()) {
        TE_INFOLOG("errArgs is empty. opNames are [%s], opTypes are [%s].", opNames.c_str(), opTypes.c_str());
        return;
    }
    string errorArgs = taskRes->errArgs;
    string key;
    if (taskRes->statusCode == ERROR_DIED_PROCESS_STATUS_CODE) {
        TE_ERRLOG("===========COMPILER PROCESS DIED OF NODE %s, TYPE %s=============",
                  opNames.c_str(), opTypes.c_str());
    } else {
        TE_FUSION_LOG(loglevel, "===========FOLLOWING IS ARGUMENTS OF NODE %s, TYPE %s=============",
                      opNames.c_str(), opTypes.c_str());
    }
    key = "input";
    PrintTensors(errorArgs, key, loglevel);

    key = "outputs";
    PrintTensors(errorArgs, key, loglevel);

    key = "attrs";
    string attrString;
    GetStringByKey(errorArgs, key, attrString);
    TE_FUSION_LOG_FULL(loglevel, "Attributes are: {%s}", attrString.c_str());
}

void OutputOverstepLog(const string &tailLine, int loglevel)
{
    if (tailLine.length() > (MSG_LENGTH - PLAIN_PREFIX_SIZE)) {
        string spiltTailLine;
        size_t end = (MSG_LENGTH - PLAIN_PREFIX_SIZE);
        for (size_t start = 0;start < tailLine.length();) {
            if (end > tailLine.length()) {
                spiltTailLine = tailLine.substr(start, tailLine.length() - start);
                TE_FUSION_LOG(loglevel, "Stack: \n  %s", spiltTailLine.c_str());
            }
            spiltTailLine = tailLine.substr(start, end - start);
            start = end;
            end = end + (MSG_LENGTH - PLAIN_PREFIX_SIZE);
            TE_FUSION_LOG(loglevel, "Stack: \n  %s", spiltTailLine.c_str());
        }
    } else {
        TE_FUSION_LOG(loglevel, "Stack: \n  %s", tailLine.c_str());
    }
}

void ParseAndPrintArgsExceptionStack(const string &opNames, const OpBuildTaskResultPtr &taskRes,
                                     int loglevel, string &errorMsg)
{
    if (taskRes->pyExceptMsg.empty()) {
        return;
    }

    size_t pos = 0;
    TE_FUSION_LOG(loglevel, "===========FOLLOWING IS STACK INFO OF NODE %s=============",
                  opNames.c_str());
    /* Get prefix information. */
    string stackLine;
    size_t lastPos = string::npos;
    pos = taskRes->pyExceptMsg.find("File \"/", pos);
    if (pos != string::npos) {
        stackLine = taskRes->pyExceptMsg.substr(0, pos);
        lastPos = pos;
        pos = taskRes->pyExceptMsg.find("File \"/", lastPos + 1);
    }

    while (pos != string::npos) {
        string newLine = taskRes->pyExceptMsg.substr(lastPos, pos - lastPos);
        if (stackLine.length() + newLine.length() > (MSG_LENGTH - PLAIN_PREFIX_SIZE)) {
            ReplaceAll(stackLine, "\\n", "\n");
            TE_FUSION_LOG(loglevel, "Stack: \n  %s", stackLine.c_str());
            stackLine = newLine;
        } else {
            stackLine += newLine;
        }

        lastPos = pos;
        pos = taskRes->pyExceptMsg.find("File \"/", lastPos + 1);
    }

    /* Print tail information(Runtime error) */
    string tailLine = taskRes->pyExceptMsg.substr(lastPos);
    ReplaceAll(tailLine, "\\n", "\n");
    errorMsg = tailLine;
    if (stackLine.length() + tailLine.length() > (MSG_LENGTH - PLAIN_PREFIX_SIZE)) {
        ReplaceAll(stackLine, "\\n", "\n");
        TE_FUSION_LOG(loglevel, "Stack: \n  %s", stackLine.c_str());
        OutputOverstepLog(tailLine, loglevel);
    } else {
        auto line = stackLine + tailLine;
        ReplaceAll(line, "\\n", "\n");
        TE_FUSION_LOG(loglevel, "Stack: \n  %s", (line).c_str());
    }
    TE_FUSION_LOG(loglevel, "===========END OF DETAILED OP INFO OF NODE %s=============",
                        opNames.c_str());
}

int SetLogLevel(OpBuildTaskResultPtr taskRes)
{
    int loglevel = TE_FUSION_LOG_INFO;
    if (taskRes->statusCode != 0) {
        loglevel = TE_FUSION_LOG_WARNING;
        // 0: prebuild, 1:single build 2: fusion build, in python config
        // fusion build need warning log, other need error log
        TE_FUSION_CHECK(taskRes->type == 0 || taskRes->type == 1, {
            loglevel = TE_FUSION_LOG_ERROR;
        });
    }
    return loglevel;
}
}

void TeFusionManager::ReportBuildErrMessage(const OpBuildTaskPtr &relBuildTaskPtr, const string &opModuleNames,
    const OpBuildTaskResultPtr &taskRes, const string &errorMsg)
{
    if (taskRes->type == OP_TASK_TYPE_FUSION || taskRes->statusCode == 0) {
        return;
    }
    std::string opName;
    std::string opType;
    if (relBuildTaskPtr == nullptr) {
        return;
    }

    if (relBuildTaskPtr->pPrebuildOp == nullptr) {
        std::vector<ge::Node *> &teGraphNode = relBuildTaskPtr->opNodes;
        if (!teGraphNode.empty()) {
            opType = teGraphNode[0]->GetType();
        }
        GetNodeListStr(teGraphNode, opName);
    } else {
        relBuildTaskPtr->pPrebuildOp->GetRealName(opName);
        relBuildTaskPtr->pPrebuildOp->GetOpType(opType);
    }

    std::string compileType = taskRes->type == 0 ? "Pre-compile" : "Compile";

    std::string opPathInfo = opModuleNames + " " + compileType + " failed with errormsg/stack: " + errorMsg;
    std::map<std::string, std::string> mapArgs = {{"op_name", opName},
            {"opp_path", opPathInfo}, {"op_type", opType}};
    // Failed to compile Op [%s]. (oppath: [%s], optype: [%s]).
    TeErrMessageReport(EM_COMPILE_OP_FAILED_ERROR, mapArgs);
    size_t len = errorMsg.length();
    size_t startIndex = 0;
    size_t recursive_times = 0;
    constexpr int32_t kMaxTurnCount = 10;
    do {
        recursive_times++;
        int endIndex = std::min(startIndex + MAX_LENGTH, len);
        string subStr = errorMsg.substr(startIndex, endIndex - startIndex);
        REPORT_TE_INNER_WARN("%s exception message is:[%s].", compileType.c_str(), subStr.c_str());
        startIndex = endIndex;
    } while (startIndex < len && recursive_times < kMaxTurnCount);
    if (taskRes->statusCode == ERROR_DIED_PROCESS_STATUS_CODE) {
        std::string errMsgs = "op[" + opName + ", " + opType +
                              "], compiler process died, unknown reason, please check detail log.";
        TeInnerErrMessageReport(EM_UNKNOWN_PROCESS_DIED_ERROR, errMsgs);
    } else {
        REPORT_TE_INNER_ERROR(
            "%s op[%s] failed, oppath[%s], optype[%s], taskID[%lu]. Please check op's compilation error message.",
            compileType.c_str(), opName.c_str(), opModuleNames.c_str(), opType.c_str(), taskRes->taskId);
    }
}

/*
 * @brief: get TeFusionManager instance
 * @param [in] ddkVer: version
 * @return TeFusionManager*: TeFusionManager instance
 */
TeFusionManager *TeFusionManager::GetInstance()
{
    static TeFusionManager managerInstance;
    return &managerInstance;
}

/*
 * @brief: release TeFusionManager instance
 * @return void.
 */
void TeFusionManager::Finalize()
{
    // func begin log
    TE_DBGLOG("Start function: destroy tbe instance.");
    finComTaskList.clear();
    finishedTask_.clear();
    reportedErr_.clear();
    taskStatisticsTime_ = 0;
    lockFpHandle_.clear();
    reuseCount = 0;
    opKernelMap_.clear();
    fusionOpsKernel_.clear();
    taskFusionMap_.clear();
    dispatchedTask_.clear();
    lastPrintTime_ = 0;
    TE_INFOLOG("Destroy tbe instance success.");
}

/*
 * @brief: config op L1 info to python
 * @param [in] opinfo: op info, include op name, module name, parameters, and so on
 * @return bool: save L1 info parameter to python ok or not
 */
bool TeFusionManager::SetOpParamsL1Info(ge::Node *pNode)
{
    TbeOpInfoPtr tbeOpInfo;
    if (!TbeOpInfoCache::Instance().GetTbeOpInfoByNode(pNode, tbeOpInfo)) {
        TE_ERRLOGF("Node %s get tbeOpInfo by node failed.", pNode->GetName().c_str());
        return false;
    }

    return PythonApiCall::Instance().SetL1SpaceSize(tbeOpInfo->GetL1Space());
}

bool TeFusionManager::RefreshCacheAndSinalManager()
{
    if (!TeConfigInfo::Instance().RefreshConfigItems()) {
        TE_WARNLOG("Failed to refresh config params.")
        return false;
    }
    // init cache manager
    if (!TeCacheManager::Instance().Initialize()) {
        TE_WARNLOG("Failed to initialize cache manager.")
        return false;
    }
    SignalManager::Instance().SaveKernelTempDir(TeConfigInfo::Instance().GetKernelMetaTempDir());
    return true;
}

void TeFusionManager::UpdatePreops(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() == 1) {
        TbeOpInfoPtr pOpInfo;
        if (!TbeOpInfoCache::Instance().GetTbeOpInfoByNode(opTask->opNodes[0], pOpInfo)) {
            TE_WARNLOG("Node %s GetTbeOpInfoByName failed.", opTask->opNodes[0]->GetName().c_str());
            return;
        }
        const std::string &opStorePattern = pOpInfo->GetOpStorePattern();
        if ((opStorePattern.find("rangeAgnostic") != std::string::npos)) {
            std::string keyName;
            bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(opTask->opNodes[0], keyName);
            TE_FUSION_CHECK(!bres, {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to get node key name.");
                return;
            });

            TbeOpInfoPtr secondOpInfo = TbeOpInfoCache::Instance().MutableSecondTbeOpInfo(keyName);
            ConstTbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().GetTbeOpInfo(keyName);
            if (secondOpInfo != nullptr && opInfo != nullptr) {
                TbeOpInfoCache::Instance().UpdateTbeOpInfo(keyName, secondOpInfo);
            }
        }
    }
}

bool TeFusionManager::IsOpdebugCompile(const std::vector<ge::Node *> &nodes)
{
    for (auto currentNode : nodes) {
        TE_FUSION_CHECK(currentNode == nullptr, continue);
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(currentNode);
        if (tbeOpInfo == nullptr) {
            continue;
        }
        if (!tbeOpInfo->GetOpDebugConfig().empty()) {
            return true;
        }
    }
    return false;
}

bool TeFusionManager::CanReuseBuildDiskCache(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() == 0) {
        TE_WARNLOGF("There is no node in opTask.");
        return false;
    }

    DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::MATCH);
    if (IsOpdebugCompile(opTask->opNodes)) {
        DfxInfoManager::Instance().RecordStatistics(
            StatisticsType::DISK_CACHE, RecordEventType::REUSE_FAIL);
        TE_DBGLOGF("Node[%s] is can not reused DiskCache.", opTask->opNodes[0]->GetName().c_str());
        return false;
    }

    int skpSubId = -1;
    (void)ge::AttrUtils::GetInt(opTask->opNodes.at(0)->GetOpDesc(), ASCENDC_SPK_SUB_ID, skpSubId);
    CompileResultPtr compileRetPtr = TeCacheManager::Instance().MatchCompileCache(opTask->kernel, (skpSubId != -1));
    if (compileRetPtr != nullptr) {
        DfxInfoManager::Instance().RecordStatistics(
            StatisticsType::DISK_CACHE, RecordEventType::REUSE_SUCC);
        return UpdateOpTaskForCompileCache(opTask, compileRetPtr);
    }
    DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::REUSE_FAIL);
    return false;
}

FILE* TeFusionManager::FindLockFpHandle(const std::string &kernelName) const
{
    TeFusionManager *pInstance = TeFusionManager::GetInstance();
    if (pInstance == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Get tefusion manager instance failed.");
        return nullptr;
    }

    auto handleIter = pInstance->lockFpHandle_.find(kernelName);
    if (handleIter == pInstance->lockFpHandle_.end()) {
        return nullptr;
    }

    return handleIter->second;
}

void TeFusionManager::SaveLockFpHandle(const std::string &kernelName, FILE *fp) const
{
    TeFusionManager *pInstance = TeFusionManager::GetInstance();
    if (pInstance == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Get tefusion manager instance failed.");
        return;
    }

    auto handleIter = pInstance->lockFpHandle_.find(kernelName);
    if (handleIter != pInstance->lockFpHandle_.end()) {
        handleIter->second = fp;
        return;
    }

    pInstance->lockFpHandle_.emplace(kernelName, fp);
    return;
}

bool TeFusionManager::GetCompileFileFromCache(const std::string &kernelName,
        std::string &jsonPath, std::string &binPath)
{
    auto opIter = opKernelMap_.find(kernelName);
    if (opIter == opKernelMap_.end()) {
        TE_INFOLOG("Cannot find res by kernelName[%s].", kernelName.c_str());
        return false;
    }
    auto matchItem = opIter->second;
    auto res = matchItem.first;
    if (res == nullptr || res->compileRetPtr == nullptr) {
        TE_INFOLOG("Find resource by kernelName[%s], but build is nullptr.", kernelName.c_str());
        return false;
    }
    jsonPath = res->compileRetPtr->jsonPath;
    binPath = res->compileRetPtr->binPath;
    TE_INFOLOG("Get jsonpath[%s] and binpath[%s] by kernelName[%s].",
               jsonPath.c_str(), binPath.c_str(), kernelName.c_str());
    return true;
}

/*
 * @brief: build single op
 * @param [in] teGraphNode: node need to build
 * @return bool: build op ok or not
 */
bool TeFusionManager::BuildSingleOp(OpBuildTaskPtr &opTask)
{
    bool res = false;
    std::string opModule;
    std::string opFuncName;
    std::string extraParams;
    std::string opImplSwitch;
    std::vector<Node *> &teGraphNode = opTask->opNodes;
    std::string &kernelName = opTask->kernel;

    if (teGraphNode.size() != 1) {
        REPORT_TE_INNER_ERROR("The nodes of task[%lu] have the wrong size[%zu], which should be 1.",
                              opTask->taskId, teGraphNode.size());
        return false;
    }

    // create fusionOpName
    Node *pNode = teGraphNode[0];
    TE_FUSION_CHECK(pNode == nullptr, return false);
    TE_INFOLOG("Start to build real single op[%s], taskID[%lu:%lu].", pNode->GetName().c_str(), opTask->graphId,
               opTask->taskId);
    std::string keyName;
    bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(pNode, keyName);
    TE_FUSION_CHECK(!bres, {
        TE_ERRLOG("Node[%s]: failed to get op key name.", pNode->GetName().c_str());
        return false;
    });

    TbeOpInfoPtr pOpInfo;
    if (!TbeOpInfoCache::Instance().GetTbeOpInfoByNode(pNode, pOpInfo)) {
        TE_ERRLOG("Failed to get tbe op info of node %s.", pNode->GetName().c_str());
        return false;
    }

    TbeOpInfo &opInfo = *pOpInfo;
    opInfo.SetKernelName(kernelName);
    std::string opsPathNamePrefix;
    (void)ge::AttrUtils::GetStr(pNode->GetOpDesc(), OPS_PATH_NAME_PREFIX, opsPathNamePrefix);
    opInfo.SetOpsPathNamePrefix(opsPathNamePrefix);
    opTask->pTbeOpInfo = pOpInfo;
    // search for cached op kernel
    OpBuildTaskResultPtr opRes;
    TE_FUSION_CHECK(opTask->attrKernelName.empty(), {
        if (CanReuseTaskCache(opTask->kernel, opKernelMap_, opTask, opRes)) {
            TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                                 opTask->opNodes.at(0)->GetOpDesc()->GetType(), "reuse task cache");
            TE_INFOLOG("Cache op compile found, node name=[%s], kernel name=[%s], taskID[%lu:%lu].",
                       pNode->GetName().c_str(), opTask->kernel.c_str(), opTask->graphId, opTask->taskId);
            return true;
        }
    });
    res = PythonApiCall::Instance().SetL1SpaceSize(opInfo.GetL1Space());
    TE_FUSION_CHECK_WITH_DUMP_PYERR(!res, {
        TE_ERRLOG("Failed to set node[%s] L1 info.", pNode->GetName().c_str());
        return false;
    });

    PyLockGIL pyLockGIL;
    TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                         opTask->opNodes.at(0)->GetOpDesc()->GetType(), "get gil lock");
    // convert op all parameter from class to PyObject
    PyObject *pyInputs = nullptr;
    PyObject *pyOutputs = nullptr;
    PyObject *pyAttrs = nullptr;
    res = AssembleOpArgs(opInfo, kernelName, pyInputs, pyOutputs, pyAttrs, true);
    TE_INFOLOGF("Single op building: kernelName[%s], op inputs: %s, outputs: %s, attrs: %s.",
                kernelName.c_str(), PyObjectToStr(pyInputs).c_str(),
                PyObjectToStr(pyOutputs).c_str(), PyObjectToStr(pyAttrs).c_str());
    TE_FUSION_CHECK(!res, {
        REPORT_TE_INNER_ERROR("Failed to assemble op args with op[%s].", pNode->GetName().c_str());
        return false;
    });

    PyObject *pyPrivateAttrs = nullptr;
    res = AssembleOpPrivateAttrs(opInfo, pyPrivateAttrs, true);
    TE_INFOLOGF("Op[%s] Single op building, Pattrs: %s.",
                pNode->GetName().c_str(), PyObjectToStr(pyPrivateAttrs).c_str());
    TE_FUSION_CHECK(!res, {
        REPORT_TE_INNER_ERROR("Failed to assemble op private args with op[%s].", pNode->GetName().c_str());
        return false;
    });

    AUTO_PY_DECREF(pyInputs);
    AUTO_PY_DECREF(pyOutputs);
    AUTO_PY_DECREF(pyAttrs);

    (void)opInfo.GetModuleName(opModule);
    (void)opInfo.GetFuncName(opFuncName);
    (void)opInfo.GetExtraParams(extraParams);
    opInfo.GetOpImplSwitch(opImplSwitch);
    std::string opImplMode = opInfo.GetOpImplMode();

    TE_FUSION_CHECK(!PythonApiCall::Instance().UpdateSingleOpModule(opInfo, opModule), {
        TE_ERRLOG("Failed to update and import single op module.");
        return false;
    });

    bool ifUnknownShape = opInfo.GetIsUnknownShape();
    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    bool pyFlag = pyTrue == nullptr || pyFalse == nullptr;
    if (pyFlag) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse from class HandleManager.");
        return false;
    }
    PyObject *pyUnknownShape = ifUnknownShape ? pyTrue : pyFalse;
    PyObject *pyInt64Mode = pOpInfo->GetFlagUseInt64() ? pyTrue : pyFalse;

    std::string enableSuperkernelPlus;
    (void)opInfo.GetOption("enable_superkernel_plus", enableSuperkernelPlus);
    TE_DBGLOG("enable_superkernel_plus is [%s].", enableSuperkernelPlus.c_str());
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {pOpInfo};
    PyObject *optionValues = PyObjectUtils::GenPyOptionsInfo(tbeOpInfoVec);
    TE_FUSION_CHECK(optionValues == nullptr, {
        TE_ERRLOG("Failed to generate options info.");
        return false;
    });
    AUTO_PY_DECREF(optionValues);

    opTask->isHighPerformace = (TeContextUtils::GetPerformanceMode() == "high") ? true : false;
    PyObject *pyIsDynamicImpl = opInfo.IsDynamicImpl() ? pyTrue : pyFalse;
    std::string opPattern = TeCompileTaskCache::Instance().GetOpPattern(keyName);

    PyObject *contextDict = GenerateDictFromContext();
    GenerateExtraInfoForContext(opInfo, opTask->opNodes.at(0)->GetOpDesc(), contextDict);
    TE_FUSION_CHECK(contextDict == nullptr, {
        TE_ERRLOG("Failed to generate context dict.");
        return false;
    });
    AUTO_PY_DECREF(contextDict);

    std::string passOptList;
    SetLicensePassOptList(passOptList);

    // need update TUNE_PARAM
    std::string tuneParam;
    (void)ge::AttrUtils::GetStr(opTask->opNodes.at(0)->GetOpDesc(), TUNE_PARAM, tuneParam);
    if (tuneParam.empty()) {
        TE_INFOLOG("tik tuneParam from graph is empty, will not tune tik op.");
    }
    std::string relationStr;
    GenerateSingleOpRelationParam(opTask->opNodes.at(0), relationStr);
    auto node = opTask->opNodes.at(0);

    int64_t spSubId = -1;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ASCENDC_SPK_SUB_ID, spSubId);
    int64_t spCnt = -1;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ASCENDC_SPK_CNT, spCnt);
    std::string spOptions;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), SPK_OPTIONS, spOptions);
    json spKernelSubInfo;
    spKernelSubInfo[ENABLE_SPK] = (spSubId != -1);
    spKernelSubInfo[ASCENDC_SPK_OPTIONS] = spOptions;
    std::string subOpLoc;
    (void)GetSubOpLoc(spCnt, spSubId, subOpLoc);
    spKernelSubInfo["super_kernel_sub_loc"] = subOpLoc;
    TE_DBGLOG("SuperKernel sub op[%s, %s], spk_options[%s], spk_sub_loc[%d].",
              node->GetNamePtr(), node->GetTypePtr(), spOptions.c_str(), subOpLoc.c_str());
    std::string optionalInputMode;
    (void)ge::AttrUtils::GetStr(opTask->opNodes.at(0)->GetOpDesc(), OPTIONAL_INPUT_MODE, optionalInputMode);
    res = PythonAdapterManager::Instance().BuildOpAsync(opTask, "dispatch_single_op_compile_task", "sssssO(OOOOO)OssOssOsssssssss",
                       opModule.c_str(), pNode->GetName().c_str(), pNode->GetTypePtr(), opFuncName.c_str(),
                       kernelName.c_str(), pyUnknownShape, pyInputs, pyOutputs, pyAttrs, optionValues, pyPrivateAttrs,
                       pyInt64Mode, opTask->pre_task.empty() ? nullptr : opTask->pre_task.c_str(),
                       opTask->post_task.empty() ? nullptr : opTask->post_task.c_str(),
                       pyIsDynamicImpl, opPattern.c_str(), nullptr,
                       contextDict, passOptList.c_str(), tuneParam.empty() ? nullptr : tuneParam.c_str(),
                       extraParams.c_str(), relationStr.c_str(), opImplSwitch.c_str(), opImplMode.c_str(),
                       optionalInputMode.c_str(), enableSuperkernelPlus.c_str(), spKernelSubInfo.dump().c_str());

    if (res) {
        (void)opKernelMap_.emplace(opTask->kernel, std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTask})));
        TE_INFOLOG("Single op build task dispatched, taskID[%lu:%lu], node=[%s], kernel=[%s].",
                   opTask->graphId, opTask->taskId, pNode->GetName().c_str(), opTask->kernel.c_str());
        TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                             opTask->opNodes.at(0)->GetOpDesc()->GetType(), "dispatch online compile");
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::TASK_SUBMIT);
    }
    if (spSubId != -1) {
        TE_INFOLOG("Superkernel sub node[%s, %s] compile task dispatched.", node->GetNamePtr(), node->GetTypePtr());
    }
    return res;
}

/*
 * @brief: check op build is from PreBuildTbeOpInner or BuildSingleOp
 * @param [in] opName: op name
 * @param [out] isPreBuild: true is for PreBuildTbeOpInner, false is for BuildSingleOp
 * @return bool: check result
 */
bool TeFusionManager::CheckPreBuildOp(const std::string &opName, bool &isPreBuild)
{
    TE_FUSION_CHECK(opName.empty(), {
        TE_ERRLOG("Op name is null.");
        return false;
    });
    isPreBuild = TbeOpInfoCache::Instance().GetTbeOpInfo(opName) != nullptr;
    return true;
}

bool TeFusionManager::UpdateOpTaskForCompileCache(const OpBuildTaskPtr &opTask,
                                                  const CompileResultPtr &compileResultPtr)
{
    TE_FUSION_NOTNULL(compileResultPtr);
    opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;

    OpBuildTaskResultPtr opResPtr = nullptr;
    TE_FUSION_MAKE_SHARED(opResPtr = std::make_shared<OpBuildTaskResult>(), return false);
    if (compileResultPtr->jsonInfo->contains("compileInfo")) {
        std::string compileInfoKey;
        std::string compileInfoStr = compileResultPtr->jsonInfo->at("compileInfo").dump();
        if (!PythonApiCall::Instance().GenerateStrSha256HashValue(compileInfoStr, compileInfoKey)) {
            REPORT_TE_INNER_WARN("Failed to generate compile_info_key with info: %s.", compileInfoStr.c_str());
            return false;
        }
        TE_DBGLOG("compileInfoKey[%s], compileInfoStr[%s].", compileInfoKey.c_str(), compileInfoStr.c_str());
        opResPtr->compile_info_key = compileInfoKey;
        opResPtr->compile_info_str = compileInfoStr;
    }
    if (HasCustomOp(opTask->opNodes)) {
        TE_DBGLOGF("DiskCache reuse of the custom node[%s] succeeded.", opTask->opNodes[0]->GetName().c_str());
    }
    opResPtr->result = "success";
    opResPtr->statusCode = 0;
    opResPtr->jsonFilePath = compileResultPtr->jsonPath;
    opResPtr->compileRetType = CompileResultType::Cache;
    opResPtr->compileRetPtr = compileResultPtr;
    opTask->opRes = opResPtr;
    TE_INFOLOG("Op [kernelName: %s] can reuse cache of json file path[%s], don't need to compile.",
               opTask->kernel.c_str(), opTask->opRes->jsonFilePath.c_str());
    return true;
}

void TeFusionManager::FusionOpRelationGetInplacelist(const ge::OutDataAnchorPtr &anchor, const ge::Node *currentNode,
                                                     std::vector<int32_t> &inplaceOutIdxList, int32_t &realIdx,
                                                     std::unordered_set<ge::Node *> &allNodes)
{
    auto idx = anchor->GetIdx();
    bool isOuterOutput = false;
    auto peers = anchor->GetPeerInDataAnchors();
    for (size_t i = 0; i < peers.size(); ++i) {
        auto &peerAnchor = peers.at(i);
        if (peerAnchor == nullptr || peerAnchor->GetOwnerNode() == nullptr) {
            isOuterOutput = true;
        } else {
            ge::NodePtr peerNode = peerAnchor->GetOwnerNode();
            if (allNodes.count(peerNode.get()) == 0) {
                isOuterOutput = true;
            }
        }
        if (isOuterOutput) {
            TE_DBGLOG("Find the node [%s]'s Outer Output [%u].", currentNode->GetName().c_str(), idx);
            uint32_t dfsMaxCnt = 0;
            if (DfsFindOuterInput(currentNode, idx, allNodes, dfsMaxCnt)) {
                TE_DBGLOG("This node [%s]'s Outer Output[%u] has Outer input, So set to inplace list [%d].",
                          currentNode->GetName().c_str(), idx, realIdx);
                inplaceOutIdxList.emplace_back(realIdx);
            }
            realIdx++;
            break;
        }
    }
}

void TeFusionManager::GetFusionOpOuterInputCnt(std::unordered_set<ge::Node *> &allNodes, const ge::Node *currentNode,
                                               int32_t &outerInputCnt)
{
    const auto &allInDataAnchors = currentNode->GetAllInDataAnchors();
    for (const auto &anchor : allInDataAnchors) {
        const auto &peerAnchor = anchor->GetPeerOutAnchor();
        if (peerAnchor == nullptr) {
            TE_DBGLOG("The anchor has no peer. node: %s", currentNode->GetName().c_str());
            continue;
        }
        const auto &srcNode = peerAnchor->GetOwnerNode();
        if (srcNode == nullptr) {
            TE_DBGLOG("The anchor has no owner node. node: %s", currentNode->GetName().c_str());
            continue;
        }
        if (allNodes.count(srcNode.get()) == 0) {
            outerInputCnt++;
            continue;
        }
    }
}

void TeFusionManager::GenerateFusionOpRelationParam(const std::vector<ge::Node *> &teGraphNode,
                                                    std::string &relationJson)
{
    std::unordered_set<ge::Node *> allNodes;
    for (auto &ele : teGraphNode) {
        allNodes.emplace(ele);
    }
    TE_DBGLOG("GenerateFusionOpRelationParam function enter.");
    std::vector<int32_t> inplaceOutIdxList;
    int32_t realIdx = 0;
    int32_t outerInputCnt = 0;
    for (const auto &currentNode : teGraphNode) {
        TE_DBGLOG("GenerateFusionOpRelationParam currentNode:%s.", currentNode->GetName().c_str());
        auto allOutputAnchors = currentNode->GetAllOutDataAnchors();
        for (const auto &anchor : allOutputAnchors) {
            FusionOpRelationGetInplacelist(anchor, currentNode, inplaceOutIdxList, realIdx, allNodes);
        }
        GetFusionOpOuterInputCnt(allNodes, currentNode, outerInputCnt);
    }
    TE_DBGLOG("GenerateFusionOpRelationParam %zu.", inplaceOutIdxList.size());

    bool hasSet = false;
    for (size_t i = 0; i < inplaceOutIdxList.size(); i++) {
        relationJson += std::to_string(outerInputCnt + inplaceOutIdxList.at(static_cast<int32_t>(i)));
        if (i != inplaceOutIdxList.size() - 1) {
            relationJson += " ";
        }
        hasSet = true;
    }
    if (hasSet) {
        TE_INFOLOG("GenerateFusionOpRelationParam op:%s, relation:%s.", teGraphNode.at(0)->GetName().c_str(),
                   relationJson.c_str());
    }
    TE_DBGLOG("GenerateFusionOpRelationParam end.");
    return;
}

bool TeFusionManager::DfsFindOuterInputSingleAnchor(const ge::InDataAnchorPtr &anchor,
                                                    const std::unordered_set<ge::Node *> &allNodes,
                                                    const ge::Node *curNode, size_t outputIdx,
                                                    uint32_t &dfsMaxCnt)
{
    TE_FUSION_NOTNULL(anchor);
    const auto &peerAnchor = anchor->GetPeerOutAnchor();
    if (peerAnchor == nullptr) {
        TE_DBGLOG("The anchor has no peer. node: %s", curNode->GetName().c_str());
        return false;
    }
    auto srcIdx = peerAnchor->GetIdx();
    const auto &srcNode = peerAnchor->GetOwnerNode();
    if (srcNode == nullptr) {
        TE_WARNLOG("The anchor has no owner node. node: %s", curNode->GetName().c_str());
        return false;
    }
    if (allNodes.count(srcNode.get()) == 0) {
        TE_DBGLOG("DfsFindOuterInput node [%s]'s Output input [%u]", srcNode->GetName().c_str(), outputIdx);
        return true;
    }
    return DfsFindOuterInput(srcNode.get(), srcIdx, allNodes, dfsMaxCnt);
}

bool TeFusionManager::DfsFindOuterInput(const ge::Node * curNode, size_t outputIdx,
                                        const std::unordered_set<ge::Node *> &allNodes, uint32_t &dfsMaxCnt)
{
    dfsMaxCnt++;
    if (dfsMaxCnt > MAX_DFS_CNT) {
        TE_DBGLOG("DfsFindOuterInput from node [%s]'s Output [%u] has reached maxcnt",
                  curNode->GetName().c_str(), outputIdx);
        return false;
    }
    TE_DBGLOG("DfsFindOuterInput from node [%s]'s Output [%u]", curNode->GetName().c_str(), outputIdx);
    bool paramReused = false;

    auto opDesc = curNode->GetOpDesc();
    (void)ge::AttrUtils::GetBool(opDesc, ATTR_NAME_SUPPORT_PARAM_REUSED, paramReused);

    bool dfsRes = false;
    if (paramReused) {
        TE_DBGLOG("This node: %s has paramReused", curNode->GetName().c_str());
        const auto &allInDataAnchors = curNode->GetAllInDataAnchors();
        for (const auto &anchor : allInDataAnchors) {
            dfsRes = DfsFindOuterInputSingleAnchor(anchor, allNodes, curNode, outputIdx, dfsMaxCnt);
            if (dfsRes) {
                return true;
            }
        }
    } else {
        std::shared_ptr<std::map<int32_t, int32_t>> relationMapPtr = nullptr;
        relationMapPtr = opDesc->TryGetExtAttr(ATTR_NAME_SUPPORT_RELATION_REUSED, relationMapPtr);
        if (!relationMapPtr) {
            TE_DBGLOG("This node: %s has no relationMapPtr", curNode->GetName().c_str());
            return false;
        }
        TE_DBGLOG("This node: %s has relationMapPtr, outputIdx:%zu", curNode->GetName().c_str(), outputIdx);
        if (outputIdx >= relationMapPtr->size()) {
            return false;
        }
        auto iter = relationMapPtr->find(outputIdx);
        if (iter == relationMapPtr->end()) {
            return false;
        }
        auto inputIdx = iter->second;
        if (inputIdx < 0 || inputIdx >= static_cast<int32_t>(curNode->GetAllInDataAnchorsSize())) {
            TE_WARNLOG("The reused output-input relation [%zu-%zu] of node: %s is invalid", outputIdx,
                       inputIdx, curNode->GetName().c_str());
            return false;
        }
        TE_DBGLOG("This node: %s's relationMapPtr info:[%d-%d]", curNode->GetName().c_str(), outputIdx, inputIdx);
        const auto &inAnchor = curNode->GetInDataAnchor(inputIdx);
        dfsRes = DfsFindOuterInputSingleAnchor(inAnchor, allNodes, curNode, outputIdx, dfsMaxCnt);
        if (dfsRes) {
            return true;
        }
    }
    return dfsRes;
}

void TeFusionManager::GenerateSingleOpRelationParam(const ge::Node *curNode,
                                                    std::string &relationJson)
{
    TE_DBGLOG("GenerateSingleOpRelationParam enter op: %s", curNode->GetName().c_str());
    bool paramReused = false;
    auto opDesc = curNode->GetOpDesc();
    auto inputCnt = curNode->GetInDataNodes().size();
    (void)ge::AttrUtils::GetBool(opDesc, ATTR_NAME_SUPPORT_PARAM_REUSED, paramReused);

    TE_DBGLOG("op: %s, paramReused: %d", curNode->GetName().c_str(), paramReused);
    bool hasSet = false;
    if (paramReused) {
        for (size_t i = 0; i < curNode->GetAllOutDataAnchors().size(); i++) {
            relationJson += std::to_string(inputCnt + i);
            if (i != curNode->GetAllOutDataAnchors().size() - 1) {
                relationJson += " ";
            }
            hasSet = true;
        }
    } else {
        std::shared_ptr<std::map<int32_t, int32_t>> relationMapPtr = nullptr;
        relationMapPtr = opDesc->TryGetExtAttr(ATTR_NAME_SUPPORT_RELATION_REUSED, relationMapPtr);
        if (!relationMapPtr) {
            TE_DBGLOG("op: %s has no relationMapPtr", curNode->GetName().c_str());
            return;
        }
        size_t indexCnt = 0;
        for (auto iter = relationMapPtr->cbegin(); iter != relationMapPtr->cend(); ++iter) {
            relationJson += std::to_string(inputCnt + iter->first);
            if (indexCnt != relationMapPtr->size() - 1) {
                relationJson += " ";
            }
            indexCnt++;
            hasSet = true;
        }
    }
    if (hasSet) {
        TE_INFOLOG("op: %s, has set relation", curNode->GetName().c_str());
    }
    TE_DBGLOG("GenerateSingleOpRelationParam end op: %s, relation: %s", curNode->GetName().c_str(), relationJson.c_str());
    return;
}

bool TeFusionManager::ParallelCompilationProcess(const OpBuildTaskPtr &opTask)
{
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec;
    for (ge::Node *node : opTask->opNodes) {
        TE_FUSION_NOTNULL(node);
        ConstTbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        TE_FUSION_NOTNULL(opInfo);
        tbeOpInfoVec.push_back(opInfo);
    }

    PyLockGIL pyLockGIL;
    TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                         opTask->opNodes.at(0)->GetOpDesc()->GetType(), "get gil lock");
    PyObject *optionValues = PyObjectUtils::GenPyOptionsInfo(tbeOpInfoVec);
    TE_FUSION_NOTNULL(optionValues);
    AUTO_PY_DECREF(optionValues);

    PyObject *contextDict = GenerateDictFromContext();
    GenerateExtraInfoForFusionOpContext(opTask->opNodes, contextDict);
    TE_FUSION_NOTNULL(contextDict);
    AUTO_PY_DECREF(contextDict);

    std::string passOptList;
    SetLicensePassOptList(passOptList);
    std::string relationStr;
    GenerateFusionOpRelationParam(opTask->opNodes, relationStr);
    ge::Node *firstNode = opTask->opNodes[0];
    std::string fixpipeFusionFlag;
    (void)ge::AttrUtils::GetStr(firstNode->GetOpDesc(), FUSION_OP_BUILD_OPTIONS, fixpipeFusionFlag);
    std::string optionalInputMode;
    (void)ge::AttrUtils::GetStr(firstNode->GetOpDesc(), OPTIONAL_INPUT_MODE, optionalInputMode);
    std::string dynamicParamMode;
    (void)ge::AttrUtils::GetStr(firstNode->GetOpDesc(), DYNAMIC_PARAM_MODE, dynamicParamMode);
    bool res = PythonAdapterManager::Instance().BuildOpAsync(opTask, "dispatch_fusion_op_compile_task", "ssssOOssssss",
        opTask->jsonStr.c_str(), opTask->kernel.c_str(), opTask->pre_task.empty() ? nullptr : opTask->pre_task.c_str(),
        opTask->post_task.empty() ? nullptr : opTask->post_task.c_str(), optionValues, contextDict, passOptList.c_str(),
        firstNode->GetName().c_str(), relationStr.c_str(), fixpipeFusionFlag.c_str(), optionalInputMode.c_str(),
        dynamicParamMode.c_str());
    if (res && !opTask->kernel.empty()) {
        (void)fusionOpsKernel_.emplace(opTask->kernel,
                                       std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTask})));
        TE_INFOLOG("Fusion op build task dispatched, taskID[%lu:%lu]. kernel name=[%s].",
                   opTask->graphId, opTask->taskId, opTask->kernel.c_str());
        TE_INFOLOGF("Fusion op building first op_name is [%s], jsonstr is [%s].",
                    firstNode->GetName().c_str(), relationStr.c_str());
        TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                             opTask->opNodes.at(0)->GetOpDesc()->GetType(), "dispatch online compile");
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::ONLINE_COMPILE, RecordEventType::TASK_SUBMIT);
    }
    return res;
}

OpBuildResCode TeFusionManager::BuildTaskFusion(OpBuildTaskPtr &opTask)
{
    TE_DBGLOG("start to call BuildTaskFusion, taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
    TE_FUSION_TIMECOST_START(BuildTaskFusion);
    bool res = TaskFusion(opTask);
    TE_FUSION_TIMECOST_END(BuildTaskFusion, "BuildTaskFusion");
    if (!res) {
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    }

    res = SaveBuildTask(opTask);
    if (!res) {
        TE_ERRLOG("Save fusion build task failed. taskID: [%lu:%lu].", opTask->graphId, opTask->taskId);
        return OP_BUILD_FAIL;
    }
    TE_DBGLOG("Success to call BuildTaskFusion, taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
    return OP_BUILD_SUCC;
}

bool TeFusionManager::TaskFusion(OpBuildTaskPtr &opTask)
{
    auto teGraphNode = opTask->opNodes;
    json opComputeList;
    json kernelList;
    for (auto &currentNode : teGraphNode) {
        std::string keyName;
        bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(currentNode, keyName);
        TE_FUSION_CHECK(!bres || keyName.empty(), {
            TE_ERRLOG("Failed to get node key name.");
            return false;
        });

        // record currentNode info in json
        json currentNodeJson;
        // get op desc
        auto currentNodeOpDesc = currentNode->GetOpDesc();
        // get basic params from currentNodeOpDesc
        currentNodeJson["name"] = keyName;
        currentNodeJson["type"] = currentNodeOpDesc->GetType();
        opComputeList.push_back(currentNodeJson);

        json kernelInfo;
        std::string kernelName;
        (void)AttrUtils::GetStr(currentNodeOpDesc, "_kernelname", kernelName);
        kernelInfo["kernel_name"] = kernelName;

        vector<int64_t> taskArgs = {};
        (void)AttrUtils::GetListInt(currentNodeOpDesc, "_task_args", taskArgs);
        kernelInfo["kernel_args"] = taskArgs;

        std::string headFilePath;
        (void)AttrUtils::GetStr(currentNodeOpDesc, "_head_file_path", headFilePath);
        TE_DBGLOG("kernelName is [%s], _head_file_path is [%s]", kernelName.c_str(), headFilePath.c_str());
        kernelInfo["kernel_impl_header"] = headFilePath;

        kernelList.push_back(kernelInfo);
    }

    json jsonData;
    for (size_t idx = 0; idx < opComputeList.size(); idx++) {
        jsonData["op_list"].push_back(opComputeList[idx]);
    }

    json taskInfo;
    json kernelFusion;
    kernelFusion["kernel_fusion"] = true;
    taskInfo["compile_options"] = kernelFusion;
    taskInfo["kernel_list"] = kernelList;

    json fusionkernelName;
    std::string fusionkernelNameStr;
    std::string tmpTaskInfo = taskInfo.dump();
    if (!PythonApiCall::Instance().GenerateStrSha256HashValue(tmpTaskInfo, fusionkernelNameStr)) {
        TE_ERRLOGF("taskID[%lu:%lu] do not generate static key hash by json(%s).",
            opTask->graphId, opTask->taskId, tmpTaskInfo.c_str());
        return false;
    }
    opTask->taskFusionUniqueKey = fusionkernelNameStr;
    std::string kernelName = "te_fused_task_" + fusionkernelNameStr;
    opTask->kernel = kernelName;
    fusionkernelName["kernel_name"] = "te_fused_task_" + fusionkernelNameStr;
    taskInfo["fusion_kernel"] = fusionkernelName;

    jsonData["task_info"] = taskInfo;
    opTask->jsonStr = jsonData.dump();
    TE_DBGLOGF("taskID[%lu:%lu]: FusionTask jsonStr is [%s]",
               opTask->graphId, opTask->taskId, opTask->jsonStr.c_str());
    return TaskFusionProcess(opTask);
}

bool TeFusionManager::TaskFusionProcess(const OpBuildTaskPtr &opTask)
{
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec;
    for (ge::Node *node : opTask->opNodes) {
        ConstTbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (opInfo == nullptr) {
            TE_ERRLOG("Node[%s] get tbe op information by node failed.", node->GetName().c_str());
            return false;
        }
        tbeOpInfoVec.push_back(opInfo);
    }

    PyLockGIL pyLockGIL;
    PyObject *optionValues = PyObjectUtils::GenPyOptionsInfo(tbeOpInfoVec);
    TE_FUSION_CHECK(optionValues == nullptr, {
        TE_ERRLOG("Failed to generate fusion options info.");
        return false;
    });
    AUTO_PY_DECREF(optionValues);
    bool res = PythonAdapterManager::Instance().BuildOpAsync(opTask, "dispatch_task_fusion_task", "sO",
                                                             opTask->jsonStr.c_str(), optionValues);
    std::string taskFusionUniqueKey = opTask->taskFusionUniqueKey;
    if (res && !taskFusionUniqueKey.empty()) {
        (void)taskFusionMap_.emplace(taskFusionUniqueKey,
                                     std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTask})));
        TE_INFOLOG("Task Fusion task dispatched, taskID[%lu:%lu]. kernel name=[%s].",
                   opTask->graphId, opTask->taskId, opTask->kernel.c_str());
    }
    return res;
}

void TeFusionManager::DumpFusionOpInfoToJsonFile(const OpBuildTaskPtr &opTask)
{
    const std::string &paraDebugPath = TeConfigInfo::Instance().GetEnvParaDebugPath();
    if (paraDebugPath.empty()) {
        return;
    }

    json jsonData;
    try {
        jsonData = json::parse(opTask->jsonStr);
    } catch (std::exception &e) {
        TE_WARNLOG("Failed to parse jsonStr, the jsonStr is %s and the reason is %s",
                   opTask->jsonStr.c_str(), e.what());
        return;
    }

    json jsonGraphInfo;
    jsonGraphInfo["graph_id"] = opTask->graphId;
    jsonGraphInfo["task_id"] = opTask->taskId;
    for (const auto &node : opTask->opNodes) {
        TE_FUSION_CHECK(node == nullptr, continue);
        std::string opInfo = "node_name: " + node->GetName() + " , node_type: " + node->GetType();
        jsonGraphInfo["node_list"].push_back(opInfo);
    }
    jsonData["graph_info"] = jsonGraphInfo;
    if (PythonApiCall::Instance().DumpFusionJson(jsonData.dump(), paraDebugPath)) {
        TE_DBGLOG("Op json file dumped, kernel:%s", opTask->kernel.c_str());
    } else {
        TE_WARNLOG("Op json file dump failed, kernel:%s", opTask->kernel.c_str());
    }
}

bool TeFusionManager::IsFusionCheckTask(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() == 0 || opTask->opNodes[0] == nullptr) {
        return false;
    }
    bool isFusionCheck = false;
    return AttrUtils::GetBool(opTask->opNodes[0]->GetOpDesc(), ONLY_FUSION_CHECK, isFusionCheck) && isFusionCheck;
}

/**
 * @brief build fusion op
 * @param [in]  opTask      op build task
 * @return  bool            true: success, false: fail
 */
bool TeFusionManager::BuildFusionOp(OpBuildTaskPtr &opTask, const std::string &opCompileStrategyStr)
{
    TE_INFOLOG("Build taskID:[%lu:%lu]. Nodes is [%s]. CompileStrategy is [%s].",  opTask->graphId,
        opTask->taskId, GetNodesName(opTask->opNodes).c_str(), opCompileStrategyStr.c_str());

    opTask->opRes = nullptr;
    opTask->maxKernelId = 0;
    opTask->isHighPerformaceRes = false;

    // We use the strategy "keep compiling in op tune" to jump over the return above.
    // The actual strategy is empty string.
    string opCompileStrategyStrTmp = opCompileStrategyStr;
    bool isFESpecStrate = opCompileStrategyStr == COMPILE_STRATEGY_KEEP_COMPILING ||
                          opCompileStrategyStr == COMPILE_STRATEGY_NO_TUNE;
    if (isFESpecStrate) {
        opCompileStrategyStrTmp = "";
    }

    std::string opTuneMode = TeContextUtils::GetBuildMode();
    std::string opTuneStep = TeContextUtils::GetBuildStep();
    bool hasOpTuneStrategy = !opCompileStrategyStrTmp.empty();
    bool isOpTuneAfterBuildStage = opTuneMode == ge::BUILD_MODE_TUNING &&
        (opTuneStep == ge::BUILD_STEP_AFTER_BUILDER || opTuneStep == ge::BUILD_STEP_AFTER_BUILDER_SUB);
    bool onlyDumpJson = (opCompileStrategyStr != COMPILE_STRATEGY_KEEP_COMPILING) && isOpTuneAfterBuildStage;
    bool isOpTuneWithStrategy = (opTuneMode == ge::BUILD_MODE_TUNING) &&
                                (opTuneStep == ge::BUILD_STEP_AFTER_UB_MATCH) && hasOpTuneStrategy;
    if (!SetJsonDescAndKernelName(opTask, isOpTuneWithStrategy || onlyDumpJson)) {
        REPORT_TE_INNER_ERROR("Failed to set json desc and kernel name for task[%lu]", opTask->taskId);
        return false;
    }
    TE_INFOLOG("Set kernel name[%s] for task[%lu:%lu] while build mode[%s], build step[%s], compile strategy[%s].",
               opTask->kernel.c_str(), opTask->graphId, opTask->taskId,
               opTuneMode.c_str(), opTuneStep.c_str(), opCompileStrategyStr.c_str());

    // only fusion check
    if (IsFusionCheckTask(opTask)) {
        opTask->pre_task = opCompileStrategyStr;
        opTask->post_task = opCompileStrategyStr;
        TE_DBGLOGF("Fusion check taskID[%lu:%lu], kernel: %s, args: %s.",
                   opTask->graphId, opTask->taskId, opTask->kernel.c_str(), opTask->jsonStr.c_str());
        return ParallelCompilationProcess(opTask);
    }
    UpdatePreops(opTask);
    bool canReuseCache = (!(hasOpTuneStrategy || onlyDumpJson || opTuneMode == ge::BUILD_MODE_TUNING ||
        opTuneMode == ge::BUILD_MODE_OPAT_RESULT) && opTask->attrKernelName.empty());
    TE_DBGLOG("opTuneMode[%s], opTuneStep[%s], onlyDumpJson[%d], canReuseCache[%d], attrKernelName[%s].",
              opTuneMode.c_str(), opTuneStep.c_str(), onlyDumpJson,
              canReuseCache, opTask->attrKernelName.c_str());

    if (canReuseCache) {
        if (CanReuseBuildDiskCache(opTask)) {
            TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                                 opTask->opNodes.at(0)->GetOpDesc()->GetType(), "reuse disk cache");
            return true;
        }

        // update build option 'jit_compile' for cur task.
        if (!TeConfigInfo::Instance().IsBinaryInstalled()) {
            if (TeContextUtils::GetJitCompile() == JIT_MODE::JIT_USE_BINARY) {
                REPORT_TE_INNER_WARN("If you want to reuse the binary file, please download it and install it first!");
            }
        }

        // om reuse binary build files
        TE_FUSION_CHECK(BinaryManager::Instance().CanReuseBinaryKernel(opTask), {
            opTask->newCompile = false;
            if (opTask->status == OP_TASK_STATUS::OP_TASK_SUCC) {
                TE_INFOLOG("reuse binary result for node(%s). No need to compile.", GetTaskNodeName(opTask).c_str());
                TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                                     opTask->opNodes.at(0)->GetOpDesc()->GetType(), "reuse binary");
                reuseCount++;
                BinaryManager::Instance().SetBinaryReuseAttr(opTask);
                return true;
            } else if (!TeConfigInfo::Instance().IsDisableOpCompile()) { // for debug, rebuild binary kernel to get cce file
                TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                                     opTask->opNodes.at(0)->GetOpDesc()->GetType(), "build binary");
                if (opTask->opNodes.size() == 1) {
                    opTask->isBuildBinarySingleOp = true;
                    return BuildBinarySingleOp(opTask);
                } else {
                    opTask->isBuildBinaryFusionOp = true;
                    return BuildBinaryFusionOp(opTask);
                }
            }
        });

        // fusion op reuse bin res failed, back to single op bin reuse.
        TE_FUSION_CHECK(opTask->opRes != nullptr, {
            TE_FUSION_CHECK(opTask->opRes->backToSingleOpBinReuse,
                TE_INFOLOG("Fusion op[taskId:%d] try to reuse bin res not successfully, would back to single op bin reuse.",
                           opTask->taskId);
                return false
            );
        });
    }

    // when op_compile is turned off, return false
    if (TeConfigInfo::Instance().IsDisableOpCompile()) {
        TE_INFOLOG("Op compile is turned off, task[%lu:%lu] will not compile", opTask->graphId, opTask->taskId);
        return false;
    }
    // restore opRes to null
    opTask->opRes = nullptr;
    PythonApiCall::Instance().ResetL1SpaceSize();

    ge::Node *firstNode = nullptr;
    TE_FUSION_CHECK(opTask->opNodes.size() > 0, {
        TE_DBGLOG("Kernel name of node %s is %s.", opTask->opNodes[0]->GetName().c_str(), opTask->kernel.c_str());
        firstNode = opTask->opNodes[0];
        TE_FUSION_CHECK(firstNode == nullptr, {
            REPORT_TE_INNER_ERROR("First node of taskID[%lu:%lu] from fe is null.", opTask->graphId, opTask->taskId);
            return false;
        });
        bool isPrebuildOp = false;
        std::string keyName;
        bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(firstNode, keyName);
        TE_FUSION_CHECK(!bres, {
            TE_ERRLOG("Failed to get node key name.");
            return false;
        });
        bres = CheckPreBuildOp(keyName, isPrebuildOp);
        TE_FUSION_CHECK(!bres, {
            TE_ERRLOG("Failed to check op is prebuild or not, op name is [%s].", firstNode->GetName().c_str());
            return false;
        });
        TE_FUSION_CHECK(isPrebuildOp, {
            bres = SetOpParamsL1Info(firstNode);
            TE_FUSION_CHECK(!bres, {
                TE_ERRLOG("Failed to set op L1 info, op name is [%s].", firstNode->GetName().c_str());
                return false;
            });
        });
    });

    DumpFusionOpInfoToJsonFile(opTask);
    TE_FUSION_CHECK(onlyDumpJson, {
        TE_INFOLOG("Get option opTuneMode is tuning, dump json only, taskID[%lu:%lu]",
                   opTask->graphId, opTask->taskId);
        TE_FUSION_CHECK((opTask->opNodes.size() == 1), {
            bool setOPArgsStatus = SetOpArgsToNode(opTask);
            TE_FUSION_CHECK(!setOPArgsStatus, {
                TE_DBGLOGF("Unable to func exec SetOpArgsToNode.");
            });
        });
        opTask->status = OP_TASK_STATUS::OP_TASK_DO_NOT_SAVE_TASK;
        if (PythonApiCall::Instance().DumpFusionJson(opTask->jsonStr, TeContextUtils::GetTuningPath())) {
            TE_INFOLOG("Op json file dumped, kernel:%s", opTask->kernel.c_str());
        } else {
            TE_WARNLOG("Op json file dumping failed, kernel:%s", opTask->kernel.c_str());
        }
        return true;
    });

    opTask->pre_task = opCompileStrategyStrTmp;
    opTask->post_task = opCompileStrategyStrTmp;

    bool sRes = SyncOpTuneParams();
    TE_FUSION_CHECK((!sRes), {
        TE_ERRLOG("Failed to sync op tune params, taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
        return false;
    });

    if (opTask->opNodes.size() == 1) {
        // auto tune failed, do single op building if these's only one op
        if (!opTask->attrKernelName.empty()) {
            opTask->kernel = opTask->attrKernelName;
            TE_DBGLOG("Update kernel name of task[%lu] to [%s].", opTask->attrKernelName.c_str());
        }
        return BuildSingleOp(opTask);
    }

    TE_INFOLOG("Build fusion op: taskID[%lu:%lu]", opTask->graphId, opTask->taskId);

    OpBuildTaskResultPtr opRes;
    if (CanReuseTaskCache(opTask->kernel, fusionOpsKernel_, opTask, opRes)) {
        TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                             opTask->opNodes.at(0)->GetOpDesc()->GetType(), "reuse task cache");
        return true;
    }

    return ParallelCompilationProcess(opTask);
}

bool TeFusionManager::SetJsonDescAndKernelName(OpBuildTaskPtr &opTask, const bool isUnique)
{
    if (!TeJsonAssemble::GenerateJsonAndKernelName(opTask->opNodes, !isUnique, opTask->jsonStr, opTask->kernel)) {
        return false;
    }
    opTask->opUniqueKey = opTask->kernel;

    if (opTask->opNodes.size() == 1) {
        // optune may specify kernel name, using kernel name attr while compiling
        (void)AttrUtils::GetStr(opTask->opNodes.at(0)->GetOpDesc(), "_kernelname", opTask->attrKernelName);
        TE_DBGLOG("KernelName attr of node[%s, %s] is [%s].", opTask->opNodes.at(0)->GetOpDesc()->GetNamePtr(),
                  opTask->opNodes.at(0)->GetOpDesc()->GetTypePtr(), opTask->attrKernelName.c_str());
    }
    TE_DBGLOG("Set kernel name[%s] for op task[%lu].", opTask->kernel.c_str(), opTask->taskId);
    return true;
}

void TeFusionManager::GetOpModuleName(const OpBuildTaskPtr &relBuildTaskPtr, string &opModuleNames)
{
    if (relBuildTaskPtr->pPrebuildOp == nullptr) {
        std::string keyName = "";
        std::vector<ge::Node *> &teGraphNode = relBuildTaskPtr->opNodes;

        uint32_t idx = 0;
        for (auto pNode : teGraphNode) {
            ConstTbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(pNode);
            if (opInfo == nullptr) {
                TE_WARNLOG("Node[%s] get tbe op information by node failed.", pNode->GetName().c_str());
                continue;
            }

            std::string name = "";
            (void)opInfo->GetModuleName(name);
            const TbeOpInfo &tbeOpInfo = *opInfo;
            UpdateOpModuleFromStaicToDynamicAndAddPrefix(tbeOpInfo, name);
            if (idx != 0) {
                opModuleNames += + " | ";
            }
            opModuleNames += name;
            opModuleNames += ".py";
            idx++;
        }
    } else {
        TbeOpInfo &opInfo = *relBuildTaskPtr->pPrebuildOp;
        opInfo.GetModuleName(opModuleNames);
        UpdateOpModuleFromStaicToDynamicAndAddPrefix(opInfo, opModuleNames);
        opModuleNames += ".py";
    }
}

bool TeFusionManager::UpdateInhibitionInfoForLog()
{
    const std::time_t now = std::time(nullptr);
    const std::tm *const ptm = std::localtime(&now);
    if (ptm == nullptr) {
        TE_WARNLOG("Get local time failed.");
        return false;
    }

    if (std::difftime(now, taskStatisticsTime_) > TIME_INTERVAL) {
        taskStatisticsTime_ = now;
        return true;
    }
    return false;
}

bool TeFusionManager::IsTimeToPrintProgressHint()
{
    const std::time_t now = std::time(nullptr);
    const std::tm *const ptm = std::localtime(&now);
    if (ptm == nullptr) {
        TE_WARNLOG("Get local time failed.");
        return false;
    }

    if (lastPrintTime_ == 0) {
        lastPrintTime_ = now;
        return false;
    }
    if (std::difftime(now, lastPrintTime_) > PRINT_INTERVAL) {
        lastPrintTime_ = now;
        return true;
    }
    return false;
}

void TeFusionManager::PrintProgressHint()
{
    if (IsTimeToPrintProgressHint()) {
        printf(".");
        fflush(stdout);
    }
}

bool TeFusionManager::GetFinishedCompilationTask(uint64_t graphId)
{
    std::vector<OpBuildTaskResultPtr> taskRetVec;
    if (!PythonAdapterManager::Instance().GetFinishedCompilationTask(graphId, taskRetVec)) {
        return false;
    }
    // parser finished task list from python list to C++ list
    for (OpBuildTaskResultPtr &taskRes : taskRetVec) {
        if (taskRes == nullptr) {
            TE_ERRLOG("Parse build task result error");
            return false;
        }

        int loglevel = SetLogLevel(taskRes);
        std::pair<uint64_t, uint64_t> graphMapKey;
        graphMapKey.first = taskRes->graphId;
        graphMapKey.second = taskRes->taskId;

        const std::map<OpTaskKey, OpBuildTaskPtr>::const_iterator historyGraphItr = dispatchedTask_.find(graphMapKey);
        if (historyGraphItr == dispatchedTask_.end()) {
            TE_WARNLOG("Couldn't find related compilation task[%lu:%lu] in dispatchedTask_, maybe timed-out",
                       graphMapKey.first, graphMapKey.second);
            continue;
        }
        OpBuildTaskPtr relBuildTaskPtr = historyGraphItr->second;
        TE_FUSION_CHECK(relBuildTaskPtr == nullptr, return false);

        dispatchedTask_.erase(historyGraphItr);
        string opNames;
        string opTypes;

        PrintOp(relBuildTaskPtr, taskRes, opNames, opTypes, loglevel);
        string opModuleNames;
        GetOpModuleName(relBuildTaskPtr, opModuleNames);
        TE_FUSION_LOG_FULL(loglevel,
            "TaskID[%lu:%lu], opNames[%s], opType[%s], status[%d], result[%s], File[%s], compile result{%s},"
            " key[%s], pattern[%s], core type[%s], prebuilt_options[%s], json file path[%s].",
            taskRes->graphId, taskRes->taskId, opNames.c_str(), opTypes.c_str(), taskRes->statusCode,
            taskRes->result.c_str(), opModuleNames.c_str(),
            taskRes->infoMsg.c_str(), taskRes->compile_info_key.c_str(), taskRes->preCompileRetPtr->opPattern.c_str(),
            taskRes->preCompileRetPtr->coreType.c_str(), taskRes->preCompileRetPtr->prebuiltOptions.c_str(),
            taskRes->jsonFilePath.c_str());
        if (taskRes->result == COMPILER_PROCESS_DIED && taskRes->errArgs.empty()) {
            continue;
        }
        ParseAndPrintArgsString(opNames, opTypes, taskRes, loglevel);
        string errorMsg;
        ParseAndPrintArgsExceptionStack(opNames, taskRes, loglevel, errorMsg);
        ReportBuildErrMessage(relBuildTaskPtr, opModuleNames, taskRes, errorMsg);

        FinComTask finshedTaskItem;
        if (relBuildTaskPtr->pPrebuildOp == nullptr) {
            finshedTaskItem.teNodeOpDesc = relBuildTaskPtr->outNode;
            TE_FUSION_CHECK(!(SetBuildResult(relBuildTaskPtr, taskRes)), {
                TE_ERRLOG("Setting task[%lu:%lu] build result failed.",
                                taskRes->graphId, taskRes->taskId);
                return false;
            });
        } else {
            TE_FUSION_CHECK(!(SetPreBuildResult(relBuildTaskPtr, taskRes)), {
                TE_ERRLOG("Task[%lu:%lu] set prebuild result failed.",
                                taskRes->graphId, taskRes->taskId);
                return false;
            });
        }

        finshedTaskItem.graphId = taskRes->graphId;
        finshedTaskItem.taskId = taskRes->taskId;
        finshedTaskItem.status = taskRes->statusCode;
        finshedTaskItem.errMsg = taskRes->infoMsg;
        finComTaskList.push_back(finshedTaskItem);
        TE_INFOLOG("Get finished task[%lu:%lu] status[%s] from compile list.",
                   finshedTaskItem.graphId, finshedTaskItem.taskId,
                   (finshedTaskItem.status == 0) ? "success" : "unsuccess");

        std::time_t now = std::time(nullptr);
        TE_DBGLOG("Task[%lu:%lu] for [node:%s, type:%s] cost %lfs after dispatch.", finshedTaskItem.graphId,
                  finshedTaskItem.taskId, opNames.c_str(), opTypes.c_str(),
                  std::difftime(now, relBuildTaskPtr->start_time));
        // get task from cashed task list
        TE_FUSION_CHECK(!FinishPendingTask(relBuildTaskPtr, taskRes), {
            TE_ERRLOG("Unable to get cached task.");
            return false;
        });
    }

    return true;
}

bool TeFusionManager::SaveFinishedTask(const OpBuildTaskPtr &task)
{
    auto graphPair = finishedTask_.emplace(task->graphId, std::map<uint64_t, OpBuildTaskPtr>());
    const auto taskIter = graphPair.first;
    TE_FUSION_CHECK((taskIter == finishedTask_.end()), {
        REPORT_TE_INNER_ERROR("Dispatched task finished with failure, taskID[%lu:%lu].", task->graphId, task->taskId);
        return false;
    });
    auto taskPair = taskIter->second.emplace(task->taskId, task);
    if (!taskPair.second) {
        REPORT_TE_INNER_ERROR("Dispatched task finished; already exists, taskID[%lu:%lu].", task->graphId, task->taskId);
        return false;
    }

    TE_DBGLOG("Dispatched task finished taskID[%lu:%lu] status[%d].", task->graphId, task->taskId, task->status);
    return true;
}

void TeFusionManager::ReportPendingTaskInfo() const
{
    if (dispatchedTask_.empty()) {
        return;
    }

    for (const auto &taskItem : dispatchedTask_) {
        OpBuildTaskPtr task = taskItem.second;
        if (task == nullptr) {
            continue;
        }
        std::stringstream ss;
        ss << "Pending task:" << "Thread id:" << std::to_string(taskItem.second->graphId);
        ss << "|Task id:" << std::to_string(taskItem.second->taskId);
        ss << "|First op:" << taskItem.second->opNodes.at(0)->GetType() << "_";
        ss << std::to_string(taskItem.second->opNodes.at(0)->GetOpDesc()->GetId());
        TraceUtils::SubmitGlobalTrace(ss.str());
    }
}

bool TeFusionManager::SaveBuildTask(const OpBuildTaskPtr &task)
{
    if (task->status == OP_TASK_STATUS::OP_TASK_DO_NOT_SAVE_TASK) {
        return true;
    }

    if (task->status == OP_TASK_STATUS::OP_TASK_SUCC || task->status == OP_TASK_STATUS::OP_TASK_FAIL) {
        bool res = SaveFinishedTask(task);
        if (!res) {
            return false;
        }
        return true;
    }

    auto resPair = dispatchedTask_.emplace(OpTaskKey(task->graphId, task->taskId), task);
    if (!resPair.second) {
        TE_INFOLOG("Dispatched task already exists, taskID[%lu:%lu].", task->graphId, task->taskId);
    } else {
        TE_DBGLOG("Dispatched task saved. taskID [%lu:%lu].", task->graphId, task->taskId);
    }

    return true;
}

bool TeFusionManager::SetPreBuildResult(const OpBuildTaskPtr &relBuildTaskPtr,
                                        const OpBuildTaskResultPtr &opBuildResult)
{
    TE_FUSION_CHECK((relBuildTaskPtr == nullptr), {
        REPORT_TE_INNER_ERROR("Prebuild task is nullptr");
        return false;
    });

    TE_FUSION_CHECK((opBuildResult == nullptr || opBuildResult->preCompileRetPtr == nullptr), {
        REPORT_TE_INNER_ERROR("Prebuild result is nullptr");
        return false;
    });

    const string &preBuildPattern = relBuildTaskPtr->pPrebuildOp->GetPrebuildPattern();
    if (!preBuildPattern.empty() && preBuildPattern != STR_UNDEFINDED) {
        TE_DBGLOG("Modify preBuildPattern from [%s] to [%s]",
                  opBuildResult->preCompileRetPtr->opPattern.c_str(), preBuildPattern.c_str());
        opBuildResult->preCompileRetPtr->opPattern = preBuildPattern;
    }

    if (opBuildResult->preCompileRetType == PreCompileResultType::Online) {
        TE_DBGLOG("Cache prebuild result: build type [%d], kernel name [%s].",
                  relBuildTaskPtr->buildType, relBuildTaskPtr->kernel.c_str());
        bool ret = TeCacheManager::Instance().SetPreCompileResult(relBuildTaskPtr->kernel,
            opBuildResult->preCompileRetPtr);
        if (!ret) {
            TE_WARNLOG("Node[%s] failed to set PreBuildCacheResult.", relBuildTaskPtr->pPrebuildOp->GetName().c_str());
        }
    }

    relBuildTaskPtr->pPrebuildOp->SetPattern(opBuildResult->preCompileRetPtr->opPattern);
    relBuildTaskPtr->pPrebuildOp->SetOpCoreType(opBuildResult->preCompileRetPtr->coreType);
    relBuildTaskPtr->pTbeOpInfo->SetOpCoreType(opBuildResult->preCompileRetPtr->coreType);

    string extraParams = relBuildTaskPtr->pTbeOpInfo->GetExtraParams();
    if (!opBuildResult->preCompileRetPtr->prebuiltOptions.empty()) {
        if (extraParams.empty()) {
            json extraParamsTmp;
            extraParamsTmp["prebuilt_options"] = opBuildResult->preCompileRetPtr->prebuiltOptions;
            extraParams = extraParamsTmp.dump();
        } else {
            try {
                json extraParamsJson = json::parse(extraParams);
                extraParamsJson["prebuilt_options"] = opBuildResult->preCompileRetPtr->prebuiltOptions;
                extraParams = extraParamsJson.dump();
            } catch (std::exception &e) {
                TE_WARNLOG("Failed to parse jsonStr, the jsonStr is %s and the reason is %s",
                           extraParams.c_str(), e.what());
                return false;
            }
        }
    }
    relBuildTaskPtr->pTbeOpInfo->SetExtraParams(extraParams);

    TE_DBGLOG("Set node's %s core type to %s, with kernel name %s and pattern %s.",
              relBuildTaskPtr->pPrebuildOp->GetName().c_str(), opBuildResult->preCompileRetPtr->coreType.c_str(),
              relBuildTaskPtr->kernel.c_str(),
              opBuildResult->preCompileRetPtr->opPattern.c_str());
    return true;
}

void TeFusionManager::SetCompileResultAttr(const ge::OpDescPtr &opDesc, const CompileResultPtr &compileRetPtr)
{
    if (opDesc == nullptr || compileRetPtr == nullptr) {
        return;
    }
    (void)AttrUtils::SetStr(opDesc, "json_file_path", compileRetPtr->jsonPath);
    (void)AttrUtils::SetStr(opDesc, "bin_file_path", compileRetPtr->binPath);
    TE_DBGLOG("Setting json file path attribute [%s] for node [%s].", compileRetPtr->jsonPath.c_str(), opDesc->GetNamePtr());
    if (compileRetPtr->jsonInfo != nullptr && compileRetPtr->kernelBin != nullptr) {
        (void)opDesc->SetExtAttr("json_value_ptr", compileRetPtr->jsonInfo);
        (void)opDesc->SetExtAttr("bin_value_ptr", compileRetPtr->kernelBin);
        TE_INFOLOG("Set json value and kernel binary extension attribute for Node [%s].", opDesc->GetNamePtr());
    }
}

bool TeFusionManager::SetBuildResult(OpBuildTaskPtr &relBuildTaskPtr, OpBuildTaskResultPtr &opBuildResult)
{
    TE_FUSION_CHECK((relBuildTaskPtr == nullptr), {
        REPORT_TE_INNER_ERROR("Build task is nullptr");
        return false;
    });

    TE_FUSION_CHECK((opBuildResult == nullptr), {
        REPORT_TE_INNER_ERROR("Build result is nullptr");
        return false;
    });

    std::string opName = relBuildTaskPtr->outNode->GetName();
    // result from cache
    if (opBuildResult->compileRetPtr == nullptr && !opBuildResult->jsonFilePath.empty()) {
        TE_INFOLOG("Reuse binary json file for node[%s] is [%s].",
                   opName.c_str(), opBuildResult->jsonFilePath.c_str());
        opBuildResult->compileRetPtr = CompileResultUtils::ParseCompileResult(opBuildResult->jsonFilePath);
    }

    if (opBuildResult->compileRetType == CompileResultType::Online && relBuildTaskPtr->superKernelUniqueKey.empty()) {
        TE_DBGLOG("Begin to cache compile result, build type[%d], cacheKernelName[%s].",
                  relBuildTaskPtr->buildType, relBuildTaskPtr->kernel.c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::COPY);
        bool ret = TeCacheManager::Instance().SetCompileResult(opBuildResult->compileRetPtr);
        if (!ret) {
            TE_WARNLOG("Failed to cache compile result for node[%s] and kernel[%s].",
                       opName.c_str(), relBuildTaskPtr->kernel.c_str());
            DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::COPY_FAIL);
        } else {
            DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::COPY_SUCC);
        }
    }

    SetCompileResultAttr(relBuildTaskPtr->outNode, opBuildResult->compileRetPtr);
    TE_DBGLOG("Set compile result attr for node[%s].", opName.c_str());
    (void)AttrUtils::SetStr(relBuildTaskPtr->outNode, "op_unique_key", relBuildTaskPtr->opUniqueKey);
    TE_DBGLOG("op_unique_key for node [%s] is: %s.", opName.c_str(), relBuildTaskPtr->opUniqueKey.c_str());

    TE_FUSION_CHECK(!SetNodeCompileInfoAttr(relBuildTaskPtr->outNode, opBuildResult), {
        TE_ERRLOG("Failed to set compile_info for node [%s].", opName.c_str());
        return false;
    });

    return true;
}

void TeFusionManager::DelDispatchedTaskByKey(uint64_t graphId, uint64_t taskId)
{
    std::pair<uint64_t, uint64_t> graphMapKey;
    graphMapKey.first = graphId;
    graphMapKey.second = taskId;
    const std::map<OpTaskKey, OpBuildTaskPtr>::const_iterator historyGraphItr = dispatchedTask_.find(graphMapKey);
    if (historyGraphItr == dispatchedTask_.end()) {
        TE_INFOLOG("Couldn't find related compilation task[%lu:%lu] in dispatchedTask_, possibly timed-out.",
                   graphMapKey.first, graphMapKey.second);
        return;
    }
    dispatchedTask_.erase(historyGraphItr);
}

void TeFusionManager::DelDispatchedTask(std::vector<OpBuildTaskPtr> &vecOpBuildTask, int index)
{
    if (vecOpBuildTask.size() == 0) {
        return;
    }

    if (index == DEL_ALL_TASKS) {
        // del all
        for (size_t i = 0; i < vecOpBuildTask.size(); ++i) {
            auto &cashedBuildTaskPtr = vecOpBuildTask[i];
            TE_FUSION_CHECK(cashedBuildTaskPtr == nullptr, continue);
            DelDispatchedTaskByKey(cashedBuildTaskPtr->graphId, cashedBuildTaskPtr->taskId);
        }
        return;
    }

    OpBuildTaskPtr cashedBuildTaskPtr = vecOpBuildTask[index];
    TE_FUSION_CHECK(cashedBuildTaskPtr == nullptr, return);
    DelDispatchedTaskByKey(cashedBuildTaskPtr->graphId, cashedBuildTaskPtr->taskId);
}

bool TeFusionManager::CompilePendingTask(std::vector<OpBuildTaskPtr> &vecOpBuildTask,
                                         const OpBuildTaskPtr &finBuildTaskPtr,
                                         const OpBuildTaskResultPtr &opBuildResult)
{
    // the finsihed task may not be first task in the que when fuzzy cache building
    for (size_t i = 0; i < vecOpBuildTask.size(); ++i) {
        auto &cashedBuildTaskPtr = vecOpBuildTask[i];
        TE_FUSION_CHECK(cashedBuildTaskPtr == nullptr, continue);

        // cur task need to be delete
        if (finBuildTaskPtr != nullptr && cashedBuildTaskPtr == finBuildTaskPtr) {
            continue;
        }

        FinComTask cashedTaskItem;
        cashedTaskItem.graphId = cashedBuildTaskPtr->graphId;
        cashedTaskItem.taskId = cashedBuildTaskPtr->taskId;
        cashedTaskItem.status = (opBuildResult == nullptr) ? 0 : opBuildResult->statusCode;
        OpBuildTaskResultPtr res = (opBuildResult == nullptr) ? cashedBuildTaskPtr->opRes : opBuildResult;
        if (cashedBuildTaskPtr->pPrebuildOp != nullptr) {
            TE_INFOLOG("Finished prebuild tbeOpInfo, %s", res->preCompileRetPtr->opPattern.c_str());
            TE_FUSION_CHECK(!(SetPreBuildResult(cashedBuildTaskPtr, res)), {
                TE_ERRLOG("Setting queueTask build result failed.");
                return false;
            });
        } else {
            TE_DBGLOG("Setting task[%lu:%lu] build result.", cashedTaskItem.graphId, cashedTaskItem.taskId)
            cashedTaskItem.teNodeOpDesc = cashedBuildTaskPtr->outNode;
            cashedBuildTaskPtr->newCompile = false;
            TE_FUSION_CHECK(!(SetBuildResult(cashedBuildTaskPtr, res)), {
                TE_ERRLOG("Setting queueTask build result failed.");
                return false;
            });
        }

        finComTaskList.push_back(cashedTaskItem);
        if (cashedTaskItem.status == 0) {
            TE_INFOLOG(
                "Get finished task[%lu:%lu] from finished pending task: status[success], kernel[%s], taskIndex[%zu]",
                cashedTaskItem.graphId, cashedTaskItem.taskId, res->result.c_str(), i);
        } else {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING,
                "Get finished task[%lu:%lu] from finished pending task: status[failed], kernel[%s], taskIndex[%zu]",
                cashedTaskItem.graphId, cashedTaskItem.taskId, res->result.c_str(), i);
        }
    }

    DelDispatchedTask(vecOpBuildTask, DEL_ALL_TASKS);
    vecOpBuildTask.clear();
    return true;
}

std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>* TeFusionManager::FindBuildTaskList(
    const OpBuildTaskPtr &relBuildTaskPtr)
{
    std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>> *pMatch = nullptr;

    if (relBuildTaskPtr->pPrebuildOp != nullptr) {
        // prebuild
        TbeOpInfoPtr &pOpInfo = relBuildTaskPtr->pTbeOpInfo;
        pMatch = TeCompileTaskCache::Instance().GetOpBuildTaskPair(pOpInfo);
        if (pMatch != nullptr) {
            TE_INFOLOG("Get prebuild op finished task from cached task list, kernel [%s]",
                       relBuildTaskPtr->kernel.c_str());
        }
    } else if (relBuildTaskPtr->pTbeOpInfo != nullptr) {
        // single op
        auto opIter = opKernelMap_.find(relBuildTaskPtr->kernel);
        if (opIter != opKernelMap_.end()) {
            pMatch = &opIter->second;
            TE_INFOLOG("Retrieve single completed task from cached task list: [%s]", relBuildTaskPtr->kernel.c_str());
        }
    } else if (!relBuildTaskPtr->superKernelUniqueKey.empty()) {
        // super kernel
        const std::string &superKernelUniqueKey = relBuildTaskPtr->superKernelUniqueKey;
        TE_INFOLOG("Try to find pending task by key[%s]", superKernelUniqueKey.c_str());
        SuperKernelTaskManager::GetInstance().GetPendingTask(superKernelUniqueKey, pMatch);
    } else if (!relBuildTaskPtr->kernel.empty()) {
        auto opIter = fusionOpsKernel_.find(relBuildTaskPtr->kernel);
        if (opIter != fusionOpsKernel_.end()) {
            pMatch = &opIter->second;
            TE_INFOLOG("Retrieve completed fusion operation task from cached task list: [%s]", relBuildTaskPtr->kernel.c_str());
        }
    } else if (!relBuildTaskPtr->taskFusionUniqueKey.empty()) {
        // task fusion
        std::string &taskFusionUniqueKey = relBuildTaskPtr->taskFusionUniqueKey;
        auto opIter = taskFusionMap_.find(taskFusionUniqueKey);
        if (opIter != taskFusionMap_.end()) {
            pMatch = &opIter->second;
            TE_INFOLOG("Retrieve completed fused tasks from the cached task list: [%s]", relBuildTaskPtr->kernel.c_str());
        }
    }
    return pMatch;
}

bool TeFusionManager::FinishPendingTask(const OpBuildTaskPtr &relBuildTaskPtr,
                                        const OpBuildTaskResultPtr &opBuildResult)
{
    std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>> *pMatch = nullptr;
    pMatch = FindBuildTaskList(relBuildTaskPtr);
    TE_FUSION_NOTNULL(pMatch);
    pMatch->first = opBuildResult;
    std::vector<OpBuildTaskPtr> &vecOpBuildTask = pMatch->second;

    TE_DBGLOG("Find pending task list, kernel [%s], result size: [%lu].",
        relBuildTaskPtr->kernel.c_str(),
        vecOpBuildTask.size());

    if (!CompilePendingTask(vecOpBuildTask, relBuildTaskPtr, opBuildResult)) {
        return false;
    }

    return true;
}

bool SetNodeCompileInfoAttr(const OpDescPtr &opDesc, const OpBuildTaskResultPtr &opRes)
{
    TE_FUSION_CHECK((opDesc == nullptr), {
        REPORT_TE_INNER_ERROR("opDesc is nullptr");
        return false;
    });

    TE_FUSION_CHECK((opRes == nullptr), {
        REPORT_TE_INNER_ERROR("opRes is nullptr");
        return false;
    });

    std::string compileInfoKey = opRes->compile_info_key;
    std::string compileInfoStr = opRes->compile_info_str;
    std::string opName = opDesc->GetName();
    if (compileInfoKey.empty()) {
        TE_DBGLOG("Op[name=%s]: compile_info_key is empty, returning.", opName.c_str());
        return true;
    }

    TE_INFOLOGF("Op [name=%s]: save compile_info_key: %s, compile_info_json: %s.",
                opName.c_str(), compileInfoKey.c_str(), compileInfoStr.c_str());

    bool bres = AttrUtils::SetStr(opDesc, "compile_info_json", compileInfoStr);
    if (!bres) {
        REPORT_TE_INNER_ERROR("Node[%s] save compile_info_json failed. info: %s",
                              opName.c_str(), compileInfoStr.c_str());
        return false;
    }

    bres = AttrUtils::SetStr(opDesc, "compile_info_key", compileInfoKey);
    if (!bres) {
        REPORT_TE_INNER_ERROR("Node [%s] failed to save compile_info_key, key: %s",
                              opName.c_str(), compileInfoKey.c_str());
        return false;
    }

    return true;
}

OpBuildResCode TeFusionManager::BuildTbeOp(OpBuildTaskPtr &opTask,
                                           const std::string &opCompileStrategyStr)
{
    opTask->newCompile = true;
    opTask->maxKernelId = -1;
    opTask->isHighPerformace = false;
    opTask->isHighPerformaceRes = false;
    opTask->isBuildBinarySingleOp = false;
    opTask->isBuildBinaryFusionOp = false;
    TE_DBGLOG("Sgt slice shape index is [%lu], buildType is [%d]", opTask->sgt_slice_shape_index, opTask->buildType);
    if (!RefreshCacheAndSinalManager()) {
        TE_WARNLOG("Failed to init compile cache.");
        return OP_BUILD_FAIL;
    }
    TraceUtils::SubmitCompileDetailTrace(opTask->graphId, opTask->opNodes.at(0)->GetOpDesc()->GetId(),
                                         opTask->opNodes.at(0)->GetOpDesc()->GetType(), "init disk cache");
    bool res = BuildFusionOp(opTask, opCompileStrategyStr);
    if (!res) {
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    }

    res = SaveBuildTask(opTask);
    if (!res) {
        TE_ERRLOG("Save fusion build task failed. taskID: [%lu:%lu].", opTask->graphId, opTask->taskId);
        return OP_BUILD_FAIL;
    }
    opTask->start_time = std::time(nullptr);
    return OP_BUILD_SUCC;
}

bool TeFusionManager::SetOpArgsToNode(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.empty()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get teGraphNode.");
        return false;
    }
    ge::Node *node = opTask->opNodes[0];
    TE_FUSION_NOTNULL(node);
    TbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().MutableTbeOpInfoByNode(node);

    // remove caxis_values, atomic_type, input_c_values
    std::string newJsonStr;
    try {
        json jsonData = json::parse(opTask->jsonStr);
        newJsonStr = jsonData.dump();
        if (jsonData.contains("op_list")) {
            json jsonNode = jsonData["op_list"][0];
            if (jsonNode.contains("input_desc")) {
                json jsonInputDesc = jsonNode["input_desc"];
                TeJsonUtils::DeleteValuesFromJson("caxis_values", jsonInputDesc);
                TeJsonUtils::DeleteValuesFromJson("ddr_base_prop", jsonInputDesc);
                jsonNode["input_desc"] = jsonInputDesc;
            }
            if (jsonNode.contains("output_data_desc")) {
                json jsonOutputDesc = jsonNode["output_data_desc"];
                TeJsonUtils::DeleteValuesFromJson("atomic_type", jsonOutputDesc);
                TeJsonUtils::DeleteValuesFromJson("input_c_values", jsonOutputDesc);
                TeJsonUtils::DeleteValuesFromJson("ddr_base_prop", jsonOutputDesc);
                jsonNode["output_data_desc"] = jsonOutputDesc;
            }
            jsonData["op_list"][0] = jsonNode;
            newJsonStr = jsonData.dump();
        }
    } catch (std::exception &e) {
        REPORT_TE_INNER_ERROR("Failed to parse json, the reason is %s", e.what());
        return false;
    }

    std::string opArgsStr;
    if (!PythonApiCall::Instance().GetAttrInNode(tbeOpInfo, newJsonStr, opArgsStr)) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Node[%s] GetAttrInNode failed.", node->GetName().c_str());
        return false;
    }

    if (!ge::AttrUtils::SetStr(node->GetOpDesc(), "opArgs", opArgsStr)) {
        TE_ERRLOG("Failed to set attr opArgs for op[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
        return false;
    }

    TE_INFOLOG("Set op args[%s] for node[%s, %s].", opArgsStr.c_str(), node->GetNamePtr(), node->GetTypePtr());
    return true;
}

bool TeFusionManager::SyncOpTuneParams()
{
    if (TeContextUtils::EnableOpBankUpdate()) {
        TE_INFOLOG("op_bank_update is true");
        map<std::string, std::string> graphOptions = ge::GetThreadLocalContext().GetAllGraphOptions();
        graphOptions["ge.op_bank_update"] = "false";
        ge::GetThreadLocalContext().SetGraphOption(graphOptions);

        if (!PythonApiCall::Instance().SyncOpTuneParams()) {
            return false;
        }
    }
    return true;
}
} // namespace fusion
} // namespace te
