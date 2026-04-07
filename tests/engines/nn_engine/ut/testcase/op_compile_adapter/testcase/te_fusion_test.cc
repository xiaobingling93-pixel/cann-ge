/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include <gtest/gtest_pred_impl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <utime.h>
#include <chrono>
#include "graph/debug/ge_attr_define.h"
#include "assemble_json/te_json_utils.h"

#define private public
#define protected public
#include "adapter/common/get_attr_by_type.h"
#include "graph/tuning_utils.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"
#include "python_adapter/python_api_call.h"
#include "python_adapter/pyobj_assemble_utils.h"
#include "python_adapter/python_adapter_manager.h"
#include "cache/te_cache_space_manager.h"
#include "fusion_api.h"
#include "python_adapter/py_wrapper.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/common_utils.h"
#include "common/te_file_utils.h"
#include "common/te_config_info.h"
#include "file_handle/te_file_handle.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "register/op_check.h"
#include "te_fusion_base.h"
#include "graph_optimizer/op_compiler/op_format_tune.h"
#include "common/util/error_manager/error_manager.h"
#include "compile/prebuild_manager.h"
#include "cache/te_cache_space_manager.h"
#include "cache/te_cache_manager.h"
#include "assemble_json/te_json_assemble.h"
#include "assemble_json/te_op_custom_utils.h"
#include "../stub/Python_stub.h"
#include "../stub/Python.h"
#include "common_stub.h"
#include "common/signal_manager.h"
#include "binary/fusion_binary_info.h"
#include "binary/binary_manager.h"
#include "binary/shape_generalization.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
// #include "register/op_impl_registry_holder_manager.h"
#include "register/op_impl_kernel_registry.h"
#include "compile/superkernel_task.h"
#include "mmpa/mmpa_api.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace te;
using namespace te::fusion;
using nlohmann::json;

const std::string session_name = "session_id";
const std::string kParamReused = "_param_reused";
const std::string kParamReusedRelation = "_param_reused_relation";

using PythonApiCallPtr = shared_ptr<te::fusion::PythonApiCall>;
using OP_CHECK_FUNC = ge::graphStatus (*)(const ge::Operator &op, ge::AscendString &result);
class TeFusionUTest : public testing::Test
{
    public:
        TeFusionUTest(){}
    protected:
        virtual void SetUp()
        {
        }
        virtual void TearDown()
        {
        }
    protected:

};

bool GenerateStrSha256HashValue_stub(te::fusion::PythonApiCall *This,
    const std::string &str, std::string &res)
{
    res = "testSuccess";
    return true;
}

TEST(TeFusionUTest, CheckNodeList)
{
    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;

    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // first node
    inputnameList.clear();
    inputnameList.push_back("data");

    FillTensorDesc(input_desc1, { 16, 16, 16, 16 }, DT_FLOAT16);
    inputdescList.clear();

    FillTensorDesc(output_desc1, { 16, 16, 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);

    OpDescPtr relu_op = CreateOpDesc("relu", "Relu", 0, "RT_DEV_BINARY_MAGIC_ELF", 123, inputnameList, 1, inputdescList,
        outputdescList);
    OpDescPtr sqrt_op = CreateOpDesc("sqrt", "Sqrt", 1, "RT_DEV_BINARY_MAGIC_ELF", 123, inputnameList, 1, inputdescList,
        outputdescList);
    OpDescPtr round_op = CreateOpDesc("round", "Round", 2, "RT_DEV_BINARY_MAGIC_ELF", 123, inputnameList, 1,
        inputdescList, outputdescList);

    ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("test_graph");
    NodePtr relu_node_ptr = graph_ptr->AddNode(relu_op);
    NodePtr sqrt_node_ptr = graph_ptr->AddNode(sqrt_op);
    NodePtr round_node_ptr = graph_ptr->AddNode(sqrt_op);

    // add edge relation
    sqrt_node_ptr->AddLinkFrom(relu_node_ptr);
    round_node_ptr->AddLinkFrom(sqrt_node_ptr);

    std::vector<Node *> nodeList;
    nodeList.push_back(&(*relu_node_ptr));
    nodeList.push_back(&(*sqrt_node_ptr));
    nodeList.push_back(&(*round_node_ptr));

    CheckNodeInfo checkNodeInfo1;
    bool ret1 = GetCheckNodeInfo(relu_op, checkNodeInfo1);
    EXPECT_EQ(ret1, true);
    CheckNodeInfo checkNodeInfo2;
    bool ret2 = GetCheckNodeInfo(sqrt_op, checkNodeInfo2);
    EXPECT_EQ(ret2, true);

    bool ret3 = VerifyNodeInfo(checkNodeInfo1, checkNodeInfo2);
    EXPECT_EQ(ret3, true);
}

TEST(TeFusionUTest, UpdateOpModuleFromStaicToDynamicAndAddPrefix)
{
    string opModule4 = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/relu.py";
    TbeOpInfo tbeOpInfo("relu_op", opModule4, "Relu", "AICore");
    UpdateOpModuleFromStaicToDynamicAndAddPrefix(tbeOpInfo, opModule4);
    EXPECT_EQ(opModule4, "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/relu.py");

    tbeOpInfo.SetDynamicImpl(true);
    UpdateOpModuleFromStaicToDynamicAndAddPrefix(tbeOpInfo, opModule4);
    EXPECT_EQ(opModule4, "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/relu.dynamic.py");

    opModule4 = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/relu";
    tbeOpInfo.SetDynamicImpl(false);
    tbeOpInfo.SetIsUnknownShape(true);
    UpdateOpModuleFromStaicToDynamicAndAddPrefix(tbeOpInfo, opModule4);
    EXPECT_EQ(opModule4, "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/relu");
}

TEST(TeFusionUTest, PreBuildTbeOp_fail_case1)
{
    TbeOpInfo opinfo("051_Mul_1", "mul%", "Mul", "AICore");
    opinfo.SetRealName("Mul_1");
    bool ret = PreBuildTbeOp(opinfo, 10, 21);
    EXPECT_EQ(ret, false);
}

TEST(TeFusionUTest, PrintCompilingProcessPromot)
{
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(".");
    TeFusionManager *pInstance = TeFusionManager::GetInstance();
    vector<FinComTask> tasks;
    bool res = false;
    uint64_t graphId = 10;
    const int loop_times = 4;
    const int sleep_times = 4;

    for (int i = 0; i < loop_times; i++) {
        sleep(sleep_times);
        res = WaitAllFinished(graphId, tasks);
    }
}

TEST(TeFusionUTest, AssembleOpPrivateAttrs)
{
    TbeOpInfo tbeOpInfo("mul", "mul", "Mul", "AiCore");
    TbeAttrValue attr_v_int32("axis", (int32_t)1);
    vector<TbeAttrValue> private_attrs = {attr_v_int32};
    tbeOpInfo.SetPrivateAttrs(private_attrs);
    PyObject *pyPrivateAttrs = nullptr;
    bool res = AssembleOpPrivateAttrs(tbeOpInfo, pyPrivateAttrs, true);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, GetHardwareInfoMap)
{
    ge::GetContext().SetSessionId(1);
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::HardwareInfo)] = 
                            "ai_core_cnt:1;cube_core_cnt:1;vector_core_cnt:1;l2_size:201326592;memory_size:65452113920";
    TeConfigInfo::Instance().ParseHardwareInfo(true);
}

TEST(TeFusionUTest, OpCheckOpSupported)
{
    bool res = false;
    TbeOpInfo op_info("fill", "", "Fill", "AIcoreEngine");
    CheckSupportedInfo checkOpSupportedInfo;

    res = CheckOpSupported(op_info, checkOpSupportedInfo);
    EXPECT_EQ(res, true);
    EXPECT_EQ(checkOpSupportedInfo.isSupported, FULLY_SUPPORTED);
}

TEST(TeFusionUTest, OpCheckOpNotSupported)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    CheckSupportedInfo checkOpSupportedInfo;

    res = CheckOpSupported(op_info, checkOpSupportedInfo);
    EXPECT_EQ(res, true);
    EXPECT_EQ(checkOpSupportedInfo.isSupported, NOT_SUPPORTED);
}

TEST(TeFusionUTest, OpSelectTbeFormat)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    std::string opDtypeFormat;

    res = SelectTbeOpFormat(op_info, opDtypeFormat);
    EXPECT_EQ(res, true);
    EXPECT_EQ(opDtypeFormat, "FLOAT : NCHW");
}

TEST(TeFusionUTest, OpGetSpecificInfo)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    std::string opSpecificInfo;

    res = GetOpSpecificInfo(op_info, opSpecificInfo);
    EXPECT_EQ(res, true);
    EXPECT_EQ(opSpecificInfo, "SpecificInfo : Unsupported");
}

TEST(TeFusionUTest, QueryOpPattern)
{
    bool res = false;
    std::vector<std::pair<std::string, std::string>> opPatternVec;
    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;
    res = QueryOpPattern(opPatternVec);
    EXPECT_EQ(res, true);
    EXPECT_EQ(opPatternVec, opPatternVec);
    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;
}

TEST(TeFusionUTest, OpGetDynamicLxFusionInfo)
{
    LX_QUERY_STATUS res = LX_QUERY_FAIL;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    std::string result;

    res = GetOpInfo(op_info, result);
    EXPECT_EQ(res, LX_QUERY_SUCC);
    EXPECT_EQ(result, "support lxfusion");
}

TEST(TeFusionUTest, OpGetStaticLxFusionInfo)
{
    LX_QUERY_STATUS res = LX_QUERY_FAIL;
    std::string opModule = "/usr/local/get_op_support_info/impl/";
    TbeOpInfo op_info("fill", opModule, "Fill", "AIcoreEngine");
    op_info.SetDynamicImpl(true);
    op_info.SetIsUnknownShape(false);
    std::string result;

    res = GetOpInfo(op_info, result);
    EXPECT_EQ(res, LX_QUERY_SUCC);
    EXPECT_EQ(result, "not support lxfusion");
}

TEST(TeFusionUTest, OpDynamicShapeRangeSupported)
{
    bool res = false;
    TbeOpInfo op_info("fill", "", "Fill", "AIcoreEngine");
    bool isSupported = false;
    std::vector<size_t> upperLimitedInputIndexs;
    std::vector<size_t> lowerLimitedInputIndexs;

    res = DynamicShapeRangeCheck(op_info, isSupported, upperLimitedInputIndexs, lowerLimitedInputIndexs);
    EXPECT_EQ(res, true);
    EXPECT_EQ(isSupported, true);
}

TEST(TeFusionUTest, OpDynamicShapeRangeUnsupported)
{
    bool res = false;
    TbeOpInfo op_info("div", "", "Div", "AIcoreEngine");
    bool isSupported = false;
    std::vector<size_t> upperLimitedInputIndexs;
    std::vector<size_t> lowerLimitedInputIndexs;

    res = DynamicShapeRangeCheck(op_info, isSupported, upperLimitedInputIndexs, lowerLimitedInputIndexs);
    EXPECT_EQ(res, true);
    EXPECT_EQ(isSupported, false);

    bool index_res = true;
    std::vector<size_t> upper_index = { 1, 3 };
    std::vector<size_t> lower_index = { 2, 4 };
    for (size_t i = 0; i < upper_index.size(); i++) {
        if (upper_index[i] != upperLimitedInputIndexs[i]) {
            res = false;
        }
        if (lower_index[i] != lowerLimitedInputIndexs[i]) {
            res = false;
        }
    }
    EXPECT_EQ(res, true);
}

// const value
TEST(TeFusionUTest, InputsTensorDescToJson1)
{
    bool res = false;

    std::vector<int64_t> shape;
    shape.push_back(4);
    TbeOpTensor tensor("test1", shape, "float", "ND", ATTR_SHAPE_LIST);
    std::vector<float> const_value1;
    const_value1.push_back(1.0);
    const_value1.push_back(std::numeric_limits<float>::quiet_NaN());
    const_value1.push_back(std::numeric_limits<float>::infinity());
    const_value1.push_back(-std::numeric_limits<float>::infinity());
    TbeAttrValue attr_v_int32("axis", const_value1);
    tensor.SetConstValue(attr_v_int32);
    json input_desc_json;
    TeJsonAssemble::InputsTensorDescToJson(tensor, input_desc_json);
    printf("jsonRes is %s\n", input_desc_json.dump().c_str());
    EXPECT_EQ(input_desc_json.contains("const_value_null_desc"), true);
    EXPECT_EQ(input_desc_json["const_value_null_desc"][0].is_null(), true);
    EXPECT_EQ(input_desc_json["const_value_null_desc"][1], "nan");
    EXPECT_EQ(input_desc_json["const_value_null_desc"][2], "inf");
    EXPECT_EQ(input_desc_json["const_value_null_desc"][3], "-inf");

    std::vector<int64_t> shape2;
    TbeOpTensor tensor2("test1", shape2, "float", "ND", ATTR_SHAPE_LIST);
    std::vector<float> const_value2;
    TbeAttrValue attr_v_int32_2("axis", const_value2);
    tensor2.SetConstValue(attr_v_int32_2);

    json input_desc_json2;
    TeJsonAssemble::InputsTensorDescToJson(tensor2, input_desc_json2);
    printf("jsonRes is %s\n", input_desc_json2.dump().c_str());
    EXPECT_EQ(input_desc_json2.contains("const_value_null_desc"), false);
}

// attr value
TEST(TeFusionUTest, GenBuildinAttrsJson2)
{
    bool res = false;
 
    TbeOpInfo op_info("RangeD", "", "RangeD", "AIcoreEngine");

    std::vector<float> const_value1;
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "x", const_value1, true, op_info);
    AddOpParamToTbeOpInfoV3({3}, "float", "ND", "y", const_value1, false, op_info);

    TbeAttrValue attr_limit("limit", 1.0);
    op_info.AddAttrValue(attr_limit);
    TbeAttrValue attr_var_1("delta1", std::numeric_limits<float>::quiet_NaN());
    op_info.AddAttrValue(attr_var_1);
    TbeAttrValue attr_var_2("delta2", std::numeric_limits<float>::infinity());
    op_info.AddAttrValue(attr_var_2);
    TbeAttrValue attr_var_3("delta3", -std::numeric_limits<float>::infinity());
    op_info.AddAttrValue(attr_var_3);

    std::vector<float> attr_list_value;
    attr_list_value.push_back(2.0);
    attr_list_value.push_back(std::numeric_limits<float>::infinity());
    attr_list_value.push_back(-std::numeric_limits<float>::infinity());
    attr_list_value.push_back(std::numeric_limits<float>::quiet_NaN());
    TbeAttrValue attr_var_4("delta4", attr_list_value);
    op_info.AddAttrValue(attr_var_4);
    TbeAttrValue attr_v_int32("axis", (int32_t)1);
    vector<TbeAttrValue> private_attrs = {attr_v_int32};
    op_info.SetPrivateAttrs(private_attrs);

    nlohmann::json jsonRes;
    TeJsonAssemble::GenBuildinAttrsJson(op_info, jsonRes);
    printf("jsonRes is %s\n", jsonRes.dump().c_str());
    EXPECT_EQ(jsonRes.contains("attr_desc_null_desc"), true);
    EXPECT_EQ(jsonRes["attr_desc_null_desc"][0].is_null(), true);
    EXPECT_EQ(jsonRes["attr_desc_null_desc"][1], "nan");
    EXPECT_EQ(jsonRes["attr_desc_null_desc"][2], "inf");
    EXPECT_EQ(jsonRes["attr_desc_null_desc"][3], "-inf");

    EXPECT_EQ(jsonRes["attr_info_desc"][0].contains("value_null_desc"), false);
    EXPECT_EQ(jsonRes["attr_info_desc"][1].contains("value_null_desc"), true);
    EXPECT_EQ(jsonRes["attr_info_desc"][2].contains("value_null_desc"), true);
    EXPECT_EQ(jsonRes["attr_info_desc"][3].contains("value_null_desc"), true);
    EXPECT_EQ(jsonRes["attr_info_desc"][1]["value_null_desc"], "nan");
    EXPECT_EQ(jsonRes["attr_info_desc"][2]["value_null_desc"], "inf");
    EXPECT_EQ(jsonRes["attr_info_desc"][3]["value_null_desc"], "-inf");
    EXPECT_EQ(jsonRes["attr_info_desc"][4]["value_null_desc"][1], "inf");
    EXPECT_EQ(jsonRes["attr_info_desc"][4]["value_null_desc"][2], "-inf");
    EXPECT_EQ(jsonRes["attr_info_desc"][4]["value_null_desc"][3], "nan");
}

TEST(TeFusionUTest, OpSetArgsToNode)
{
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("accumulate", "Accumulate");
    (void)ge::AttrUtils::SetStr(opDescPtr, "_session_graph_id", "llt_test");
    vector<string> variableValue;
    variableValue.push_back("_input_name_key");
    variableValue.push_back("_output_name_key");
    ge::AttrUtils::SetListStr(opDescPtr, "variable_attr", variableValue);
    ge::NodePtr node = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(node.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask { graphId, taskId, sgtThreadIndex,
        teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC });
    TbeOpInfo op_info("accumulate", "", "Accumulate", "AIcoreEngine");
    TbeAttrValue attr_var("_input_name_key", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    op_info.AddAttrValue(attr_var);
    TbeAttrValue attr_var_1("_dst_type", (int8_t)1);
    attr_var.SetIsDefaultValue(true);
    op_info.AddAttrValue(attr_var_1);
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(op_info);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("llt_test_accumulate", pTbeOp);
    task->pTbeOpInfo = pTbeOp;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    currentFilePath = te::fusion::RealPath(currentFilePath);

    std::string jsonPath = currentFilePath + "/test_files/kernel_meta/accumulate.json";

    if (te::fusion::RealPath(jsonPath).empty()) {
        printf("jsonPath=%s is not exist!\n", jsonPath.c_str());
    } else {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            printf("Open %s failed!\n", jsonPath.c_str());
        } else {
            json json_obj;
            ifs >> json_obj;
            ifs.close();
            task->jsonStr = json_obj.dump();
            res = te::fusion::TeFusionManager::GetInstance()->SetOpArgsToNode(task);
        }
    }
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, ParseCheckResultFailed_01)
{
    bool res = false;
    PyObject temp;
    temp.func = "check_supported";
    temp.res = "isNotTuple";
    PyObject *resPtr = &temp;
    std::string opModule;
    te::CheckSupportedResult isSupport;
    std::string reason;
    PythonApiCallPtr pyApiCall = make_shared<te::fusion::PythonApiCall>();

    pyApiCall->ParseCheckResult(resPtr, opModule, isSupport, reason);
    EXPECT_EQ(isSupport, NOT_SUPPORTED);
    EXPECT_EQ(reason, "Null");
}

TEST(TeFusionUTest, ParseCheckResultFailed_02)
{
    bool res = false;
    PyObject temp;
    temp.func = "check_supported";
    temp.res = "parse_failed";
    PyObject *resPtr = &temp;
    std::string opModule;
    te::CheckSupportedResult isSupport;
    std::string reason;
    PythonApiCallPtr pyApiCall = make_shared<te::fusion::PythonApiCall>();

    pyApiCall->ParseCheckResult(resPtr, opModule, isSupport, reason);
    EXPECT_EQ(isSupport, NOT_SUPPORTED);
    EXPECT_EQ(reason, "Failed to Parse result from operator implementation");
}

TEST(TeFusionUTest, ParseCheckResultFailed_03)
{
    bool res = false;
    PyObject temp;
    temp.func = "check_supported";
    temp.res = "isTuple";
    PyObject *resPtr = &temp;
    std::string opModule;
    te::CheckSupportedResult isSupport;
    std::string reason;
    PythonApiCallPtr pyApiCall = make_shared<te::fusion::PythonApiCall>();

    pyApiCall->ParseCheckResult(resPtr, opModule, isSupport, reason);
    EXPECT_EQ(isSupport, PARTIALLY_SUPPORTED);
    EXPECT_EQ(reason, "Null");
}

TEST(TeFusionUTest, GenerateSocInfoJson)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);

    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    map<std::string, std::string> options;

    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    std::vector<ge::Node *> teGraphNode = { node.get() };
    nlohmann::json socInfoJson;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opInfo};
    te::fusion::TeJsonAssemble::GenerateSocInfoJson(tbeOpInfoVec, socInfoJson);

    EXPECT_EQ(socInfoJson["coreNum"], "5");
}

TEST(TeFusionUTest, GenerateSocInfoJson2)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    map<std::string, std::string> options;

    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    std::vector<ge::Node *> teGraphNode = { node.get() };
    nlohmann::json socInfoJson;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opInfo};
    te::fusion::TeJsonAssemble::GenerateSocInfoJson(tbeOpInfoVec, socInfoJson);

    EXPECT_EQ(socInfoJson["coreNum"], "5");
}

TEST(TeFusionUTest, GenerateSocInfoJson3)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    map<std::string, std::string> options;

    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    std::vector<ge::Node *> teGraphNode = { node.get() };
    nlohmann::json socInfoJson;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opInfo};
    te::fusion::TeJsonAssemble::GenerateSocInfoJson(tbeOpInfoVec, socInfoJson);

    EXPECT_EQ(socInfoJson["coreNum"], "5");
}

TEST(TeFusionUTest, GenerateSocInfoJson4)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    map<std::string, std::string> options;

    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    opInfo->SetVectorCoreType(VectorCoreType::ENABLE);
    std::vector<ge::Node *> teGraphNode = { node.get() };
    nlohmann::json socInfoJson;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opInfo};
    te::fusion::TeJsonAssemble::GenerateSocInfoJson(tbeOpInfoVec, socInfoJson);

    EXPECT_EQ(socInfoJson["coreNum"], "5");
}

TEST(TeFusionUTest, GenerateSocInfoJson5)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    map<std::string, std::string> options;
 
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_op_aicore_num", "5");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_op_vectorcore_num", "6");
    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    std::vector<ge::Node *> teGraphNode = {node.get()};
    nlohmann::json socInfoJson;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opInfo};
    te::fusion::TeJsonAssemble::GenerateSocInfoJson(tbeOpInfoVec, socInfoJson);
 
    EXPECT_EQ(socInfoJson["coreNum"], "5");
}

TEST(TeFusionUTest, GenerateOptionsMap)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    map<std::string, std::string> options;
 
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_op_aicore_num", "5");
    ge::AttrUtils::SetStr(node->GetOpDesc(), "_op_vectorcore_num", "6");
    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    std::vector<ge::Node *> teGraphNode = {node.get()};
    nlohmann::json socInfoJson;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opInfo};
    te::fusion::TeJsonAssemble::GenerateOptionsMap(tbeOpInfoVec, options);
 
    EXPECT_EQ(options["coreNum"], "5");
}

TEST(TeFusionUTest, GenNodeDataJson)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    std::string keyStr = "_pattern";
    ge::AttrUtils::SetStr(node->GetOpDesc(), keyStr, "x");

    TbeOpParam input = TbeOpParam();
    TbeOpTensor tensor = TbeOpTensor();
    auto value = TbeAttrValue();
    tensor.SetConstValue(value);
    tensor.SetType("bool");
    input.SetTensor(tensor);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    opInfo->AddInput(input);
    map<std::string, std::string> options;

    options["device_id"] = "1";
    options["ge.engineType"] = "2";
    options["fe.engineType"] = "VectorEngine";
    options[CORE_TYPE_NAME] = "4";
    options["ge.aicoreNum"] = "5";
    opInfo->SetOptions(options);
    opInfo->SetNode(node);
    std::vector<ge::Node *> teGraphNode = { node.get() };
    nlohmann::json data_json;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("TestOp", opInfo);
    bool res = te::fusion::TeJsonAssemble::GenNodeDataJson(teGraphNode, node.get(), 0, 0, data_json, true);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, RangeToString)
{
    const std::vector<std::pair<int64_t, int64_t>> ranges = { { 1, 10 }, { 2, 10 } };
    std::string res = "[{1,10},{2,10}]";
    std::string str = RangeToString(ranges);
    EXPECT_EQ(res, str);
}

TEST(TeFusionUTest, ShapeToString)
{
    const std::vector<int64_t> shapes = { 1, 2, 3, 4 };
    std::string res = "[1,2,3,4]";
    std::string str = ShapeToString(shapes);
    EXPECT_EQ(res, str);
}

TEST(TeFusionUTest, check_python_version)
{
    std::string cmd = "python3-config --prefix";
    const char* sysCammand = cmd.data();
    FILE *fp = popen(sysCammand, "r");
    char line[1024] = {0};
    bool flag = false;
    bool res = te::fusion::HandleManager::Instance().CheckPythonVersion(fp, line, flag);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, CheckCommandValid)
{
    std::string cmd = "config";
    char line[1024] = {0};
    bool res = te::fusion::HandleManager::Instance().CheckCommandValid(cmd, line);
    EXPECT_EQ(res, false);
}

TEST(TeFusionUTest, CheckCommandValid_fp_null)
{
    FILE* fp = fopen("test.txt", "r");
    MOCKER(popen)
        .expects(once())
        .will(returnValue(fp));
    std::string cmd = "python3-config --prefix";
    char line[1024] = {0};
    bool res = te::fusion::HandleManager::Instance().CheckCommandValid(cmd, line);
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

TEST(TeFusionUTest, LaunchDynamicLib_load_so_fail)
{
    std::string cmd = "python3 -V";
    const char* sysCammand = cmd.data();
    FILE *fp1 = popen(sysCammand, "r");
    FILE* fp2 = fopen("test.txt", "r");
    MOCKER(popen)
        .expects(exactly(2))
        .will(returnValue(fp1))
        .then(returnValue(fp2));
    void *libHandle = nullptr;
    MOCKER(mmDlopen)
        .expects(once())
        .will(returnValue(libHandle));
    bool res = te::fusion::HandleManager::Instance().LaunchDynamicLib();
    EXPECT_EQ(res, false);
    GlobalMockObject::verify();
}

TEST(TeFusionUTest, TbeInit_UT_fail)
{
    TeFusionManager *pInstance = nullptr;

    std::map<std::string, std::string> options;
    pInstance = TeFusionManager::GetInstance();
}

TEST(TEFUSION, SerialInit_UT_fail)
{
    TeFusionManager *pInstance = nullptr;
    std::map<std::string, std::string> options;
    pInstance = TeFusionManager::GetInstance();
    EXPECT_NE(pInstance, nullptr);

    GlobalMockObject::verify();
}

TEST(TeFusionUTest, TbeInitialize_UT0) {
    std::map<std::string, std::string> options = {{"abc", "dfg"}};
    MOCKER(setenv)
        .stubs()
        .will(returnValue(-1));
    bool parallelComp = true;
    bool *isSupportParallelCompilation = &parallelComp;
    TbeInitialize(options, isSupportParallelCompilation); 
}

TEST(TeFusionUTest, TbeInitialize_UT1) {
    std::map<std::string, std::string> options = {{"abc", "dfg"}};
    MOCKER(&TeConfigInfo::Initialize)
        .stubs()
        .will(returnValue(false));
    bool parallelComp = true;
    bool *isSupportParallelCompilation = &parallelComp;
    TbeInitialize(options, isSupportParallelCompilation);
}

TEST(TeFusionUTest, TbeInitialize_UT2) {
    std::map<std::string, std::string> options = {{"abc", "dfg"}};
    MOCKER(&HandleManager::Initialize)
        .stubs()
        .will(returnValue(false));
    bool parallelComp = true;
    bool *isSupportParallelCompilation = &parallelComp;
    TbeInitialize(options, isSupportParallelCompilation);
}

TEST(TeFusionUTest, TbeInit_UT)
{
    TeFusionManager *pInstance = nullptr;
    std::map<std::string, std::string> options;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(".");
    pInstance = TeFusionManager::GetInstance();
}

TEST(TeFusionUTest, TbeInitWithResourceCtrl)
{
    TeFusionManager *pInstance = nullptr;
    std::map<std::string, std::string> options;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(".");
    char *resCtrl = "MIN_COMPILE_RESOURCE_USAGE_CTRL=ub_fusion, coretype_check, op_compile, op_com";
    putenv(resCtrl);
    pInstance = TeFusionManager::GetInstance();
    resCtrl = "";
    putenv(resCtrl);
}

TEST(TeFusionUTest, RefreshSgtSliceShape_UT)
{
    nlohmann::json output_desc;
    nlohmann::json output_data_desc;
    output_desc["test0"] = "test0";
    TeJsonAssemble::RefreshSgtSliceShape(output_desc, output_data_desc);
    output_desc["sgt_slice_shape"] = "sgt_slice_shape";
    TeJsonAssemble::RefreshSgtSliceShape(output_desc, output_data_desc);
}

TEST(TeFusionUTest, PrebuildOpGenPyOptionFail_UT)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    ge::GetThreadLocalContext().graph_options_["ge.debugDir"] = "111111";
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);

    // call single op
    finalRes &= PreBuildTbeOp(*opInfoPtr, 0, 0);

    EXPECT_EQ(finalRes, false);


}

bool GetOpInputKeyName_stub(const ge::Node *node, std::vector<std::string> &key_name)
{
    key_name.push_back("11");
    key_name.push_back("22");
    printf("Get in GetOpInputKeyName_stub. key_name size is 2.\r\n");
    return true;
}

bool GetPreNodePtr_stub(te::fusion::TeFusionManager *This, ge::Node *currNode, uint32_t index, ge::Node *&preNode)
{
    printf("Get in GetPreNodePtr_stub. \r\n");
    return true;
}

/*
 * op: fused_mul_op
 * input_shape: (16, 16)
 * output_shape: (16, 16)
 * stype: float16
 * dtype: float16
 * description: test special states handling, including:
 * 1. mul-output, it will be named output, output_1, output_2...
 * 2. call prebuild and build process
 * 3. check task in quene
 */
TEST(TeFusionUTest, PrebuildAndBuildOp_UT)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);

    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr data1 = std::make_shared<ge::OpDesc>("Data1", "Data");
    OpDescPtr data2 = std::make_shared<ge::OpDesc>("Data2", "Data");
    data1->AddOutputDesc(output_desc1);
    data2->AddOutputDesc(output_desc2);
    OpDescPtr out = std::make_shared<ge::OpDesc>("output1", "Netoutput");
    out->AddInputDesc(output_desc1);

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);
    NodePtr data_node1 = graphPtr1->AddNode(data1);
    NodePtr data_node2 = graphPtr1->AddNode(data2);
    NodePtr out_node = graphPtr1->AddNode(out);
    nodeList.push_back(nodePoint1.get());
    ge::GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), nodePoint1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), nodePoint1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodePoint1->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    std::cout << "sjm curpath=" << GetCodeDir() << std::endl;
    std::cout << te::fusion::RealPath(".") << std::endl;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::cout << te::fusion::RealPath(currentFilePath) << std::endl;
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    }
    ge::GetThreadLocalContext().graph_options_["ge.debugDir"] = currentFilePath;
    options.emplace(std::make_pair("ge.opDebugLevel", "3"));
    options.emplace(std::make_pair("device_id", "3"));
    options.emplace(std::make_pair("op_precision_mode_str", "hell,ReduceSum:ad:,oqq,u"));
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);


    // 2. call funcs
    bool finalRes = PreBuildTbeOp(*opInfoPtr, 0, 0);

    std::vector<ge::NodePtr> empty;
    // call fusion op
    OpBuildResCode resCode;

    // TuneProc
    // create node
    OpDescPtr opdesc2;
    opdesc2 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr2 = std::make_shared<ComputeGraph>("4");
    NodePtr nodePoint2 = graphPtr2->AddNode(opdesc2);
    NodePtr data_node3 = graphPtr2->AddNode(data1);
    NodePtr data_node4 = graphPtr2->AddNode(data2);
    NodePtr out_node1 = graphPtr2->AddNode(out);
    ge::GraphUtils::AddEdge(data_node3->GetOutDataAnchor(0), nodePoint2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node4->GetOutDataAnchor(0), nodePoint2->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodePoint2->GetOutDataAnchor(0), out_node1->GetInDataAnchor(0));
    (void)ge::AttrUtils::SetStr(opdesc2, "_pattern",
        "Conv2d_backprop_input"); //   for GetDxDreluNode
    nodeList.push_back(nodePoint2.get());

    MOCKER_CPP(&te::fusion::GetOpInputKeyName, bool (*)(const ge::Node *, std::vector<std::string> &))
        .stubs()
        .will(invoke(GetOpInputKeyName_stub));

    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, "strage"); // GetPreNodePtr
    EXPECT_EQ(resCode, OP_BUILD_SUCC);

    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // TuneProc RL_TUNE RLTuneFusionOp
    EXPECT_EQ(resCode, OP_BUILD_SUCC);

    EXPECT_EQ(finalRes, true);

    GlobalMockObject::verify();


}

TEST(TeFusionUTest, PrebuildAndBuildOpForRL_UT)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    OpDescPtr data1 = std::make_shared<ge::OpDesc>("Data1", "Data");
    OpDescPtr data2 = std::make_shared<ge::OpDesc>("Data2", "Data");
    data1->AddOutputDesc(output_desc1);
    data2->AddOutputDesc(output_desc2);
    OpDescPtr out = std::make_shared<ge::OpDesc>("output1", "Netoutput");
    out->AddInputDesc(output_desc1);
    NodePtr data_node1 = graphPtr1->AddNode(data1);
    NodePtr data_node2 = graphPtr1->AddNode(data2);
    NodePtr out_node = graphPtr1->AddNode(out);

    ge::GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), nodePoint1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), nodePoint1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodePoint1->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    ge::GetThreadLocalContext().graph_options_["ge.debugDir"] = currentFilePath;
    options.emplace(std::make_pair("ge.opDebugLevel", "3"));
    options.emplace(std::make_pair("device_id", "3"));
    options.emplace(std::make_pair("op_precision_mode_str", "hell,ReduceSum:ad:,oqq,u"));
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);
opInfoPtr->SetUBSpaceSize(32768);

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub_ForRL;


    std::vector<ge::NodePtr> empty;
    // call fusion op
    OpBuildResCode resCode;

    // TuneProc
    // create node
    OpDescPtr opdesc2;
    opdesc2 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr2 = std::make_shared<ComputeGraph>("4");
    NodePtr nodePoint2 = graphPtr2->AddNode(opdesc2);
    (void)ge::AttrUtils::SetStr(opdesc2, "_pattern",
        "Conv2d_backprop_input"); //   for GetDxDreluNode

    MOCKER_CPP(&te::fusion::GetOpInputKeyName, bool (*)(const ge::Node *, std::vector<std::string> &))
        .stubs()
        .will(invoke(GetOpInputKeyName_stub));

    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, "strage"); // GetPreNodePtr

    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // TuneProc RL_TUNE RLTuneFusionOp

    GlobalMockObject::verify();


}

TEST(TeFusionUTest, PrebuildFailAndBuildFail_UT)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);


    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_PreBuild_Fail_Stub;

    // call single op
    finalRes &= PreBuildTbeOp(*opInfoPtr, 0, 0);

    EXPECT_EQ(finalRes, false);

    // call single op
    OpBuildResCode resCode;
    std::vector<ge::NodePtr> empty;
    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // BuildSingleOp fail
    EXPECT_EQ(resCode, OP_BUILD_SUCC);

    std::vector<std::string> original_names;
    original_names.push_back("matmul");
    original_names.push_back("matmul1");
    (void)ge::AttrUtils::SetListStr(opdesc1, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // RLTuneSingleOp fail
    EXPECT_EQ(resCode, OP_BUILD_FAIL);



    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
}

TEST(TeFusionUTest, BuildWithBuildModeOpatResult)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc output_desc1;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc1);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc1);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1,
        inputnameList, 2, inputdescList, outputdescList);
    OpDescPtr opdesc2;
    opdesc2 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1,
                           inputnameList, 2, inputdescList, outputdescList);
    OpDescPtr opdesc3;
    opdesc3 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1,
                           inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);
    NodePtr nodePoint2 = graphPtr1->AddNode(opdesc2);
    NodePtr nodePoint3 = graphPtr1->AddNode(opdesc3);
    ge::GraphUtils::AddEdge(nodePoint2->GetOutDataAnchor(0), nodePoint1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodePoint3->GetOutDataAnchor(0), nodePoint1->GetInDataAnchor(1));
    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);
    opInfoPtr->SetUBSpaceSize(32768);

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_PreBuild_Fail_Stub;

    // call single op
    finalRes &= PreBuildTbeOp(*opInfoPtr, 0, 0);

    EXPECT_EQ(finalRes, true);

    // call single op
    OpBuildResCode resCode;
    std::vector<ge::NodePtr> empty;
    string build_mode(ge::BUILD_MODE);
    string opat_result(ge::BUILD_MODE_OPAT_RESULT);
    std::map<std::string, std::string> options_map;
    options_map[build_mode] = opat_result;
    ge::GetThreadLocalContext().SetGraphOption(options_map);
    ge::GetThreadLocalContext().SetSessionOption(options_map);
    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // BuildSingleOp fail
    EXPECT_EQ(resCode, OP_BUILD_SUCC);
    options_map[build_mode] = "";
    ge::GetThreadLocalContext().SetGraphOption(options_map);
    ge::GetThreadLocalContext().SetSessionOption(options_map);
}

TEST(TeFusionUTest, BuildOpFail_UT)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);


    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_PreBuild_Fail_Stub;

    // call single op
    finalRes &= PreBuildTbeOp(*opInfoPtr, 0, 0);

    EXPECT_EQ(finalRes, false);

    // call single op
    OpBuildResCode resCode;
    std::vector<ge::NodePtr> empty;

    std::vector<std::string> original_names;
    original_names.push_back("matmul");
    original_names.push_back("matmul1");
    (void)ge::AttrUtils::SetListStr(opdesc1, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);

    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // RLTuneSingleOp fail GetNodeListStr

    EXPECT_EQ(resCode, OP_BUILD_FAIL);

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
}

TEST(TeFusionUTest, BuildOpFail_Parma_UB_Fusion_UT)
{
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
    CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    tensorop.SetAddrType(0x1U << 15U);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    tensorop2.SetAddrType(RT_MEMORY_HBM);
    tensorop2.SetAddrType(RT_MEMORY_L1);
    tensorop2.SetAddrType(RT_MEMORY_L2);
    tensorop2.SetAddrType(0);
    tensorop2.SetAddrType(0x1U << 15U);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;
    (void)opInfoPtr->SetOptions(options);
    opInfoPtr->SetNode(nodePoint1);
    opInfoPtr->SetUBSpaceSize(32768);

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_PreBuild_Fail_Stub;

    // call single op
    finalRes &= PreBuildTbeOp(*opInfoPtr, 0, 0);

    EXPECT_EQ(finalRes, false);

    // call single op
    OpBuildResCode resCode;
    std::vector<ge::NodePtr> empty;

    std::vector<std::string> original_names;
    original_names.push_back("matmul");
    original_names.push_back("matmul1");
    (void)ge::AttrUtils::SetListStr(opdesc1, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);

    resCode = TeFusion(nodeList, opdesc1, empty, 0, 0, ""); // RLTuneSingleOp fail GetNodeListStr

    EXPECT_EQ(resCode, OP_BUILD_FAIL);
    EXPECT_EQ(opinfo == *opInfoPtr, false);

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
}

te::fusion::OpBuildTaskResultPtr ParseOpTaskResult_stub(PyObject *pyRes)
{
    std::string compile_info_key;
    std::string compileInfo;

    printf("Get in ParseOpTaskResult_stub. \r\n");
    std::string pyExceptMsgValue = "\"\'errCode\': EB0000, \'message\': error occurs.\"";
    pyExceptMsgValue += "stub File \"/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc\"";
    pyExceptMsgValue += "File \"/usr/local/python3.7/lib/python3.7/site-packages/te_fusion/fusion_op.cc\"";
    std::string errArgsValue = "input0: {\"L1_addr_offset\":0,\"L1_fusion_type\":-1,\"data_type\":\"float16\"";
    errArgsValue +=
        "\"format\":\"FRACTAL_NZ\",\"name\":\"-1_0_trans_17__0\",\"ori_format\":\"ND\",\"ori_range\":[[1,1],[5,5]],}";
    errArgsValue += "output0: {\"L1_addr_offset\"";
    OpBuildTaskResultPtr ptr = std::make_shared<OpBuildTaskResult>();
    ptr->type = 0;
    ptr->graphId = 0;
    ptr->taskId = 0;
    ptr->statusCode = 1;
    ptr->errArgs = errArgsValue;
    ptr->pyExceptMsg = pyExceptMsgValue;
    ptr->compile_info_key = compile_info_key;
    ptr->compile_info_str = compileInfo;
    ptr->compileRetPtr = std::make_shared<CompileResult>();
    ptr->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    return ptr;
}

te::fusion::OpBuildTaskResultPtr ParseOpTaskResult_stub_1(PyObject *pyRes)
{
    std::string compile_info_key;
    std::string compileInfo;

    printf("Get in ParseOpTaskResult_stub_1. \r\n");
    std::string pyExceptMsgValue = "\"\'errCode\': None, \'message\': Error occurs.\"";
    pyExceptMsgValue += "stub File \"/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc\"";
    pyExceptMsgValue += "File \"/usr/local/python3.7/lib/python3.7/site-packages/te_fusion/fusion_op.cc\"";
    std::string errArgsValue = "input0: {\"L1_addr_offset\":0,\"L1_fusion_type\":-1,\"data_type\":\"float16\"";
    errArgsValue +=
        "\"format\":\"FRACTAL_NZ\",\"name\":\"-1_0_trans_17__0\",\"ori_format\":\"ND\",\"ori_range\":[[1,1],[5,5]],}";
    errArgsValue += "output0: {\"L1_addr_offset\"";
    OpBuildTaskResultPtr ptr = std::make_shared<OpBuildTaskResult>();
    ptr->type = 0;
    ptr->graphId = 0;
    ptr->taskId = 0;
    ptr->statusCode = 1;
    ptr->errArgs = errArgsValue;
    ptr->pyExceptMsg = pyExceptMsgValue;
    ptr->compile_info_key = compile_info_key;
    ptr->compile_info_str = compileInfo;
    ptr->compileRetPtr = std::make_shared<CompileResult>();
    ptr->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    return ptr;
}

te::fusion::OpBuildTaskResultPtr ParseOpTaskResult_stub3(PyObject *pyRes)
{
    std::string compile_info_key;
    std::string compileInfo;

    printf("Get in ParseOpTaskResult_stub3. \r\n");
    std::string pyExceptMsgValue = "\"\'errCode\': EC0000, \'message\': error occurs.\"";
    pyExceptMsgValue += "stub File \"/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc\"";
    for (size_t i = 0; i < 20; i++) {
      pyExceptMsgValue += "/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc";
    }
    pyExceptMsgValue += "stub File \"/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc\"";
    for (size_t i = 0; i < 5; i++) {
      pyExceptMsgValue += "/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc";
    }

    std::string errArgsValue = "input0: {\"L1_addr_offset\":0,\"L1_fusion_type\":-1,\"data_type\":\"float16\"";
    errArgsValue +=
        "\"format\":\"FRACTAL_NZ\",\"name\":\"-1_0_trans_17__0\",\"ori_format\":\"ND\",\"ori_range\":[[1,1],[5,5]],}";
    errArgsValue += "output0: {\"L1_addr_offset\"";
    OpBuildTaskResultPtr ptr = std::make_shared<OpBuildTaskResult>();
    ptr->type = 0;
    ptr->graphId = 0;
    ptr->taskId = 0;
    ptr->statusCode = 1;
    ptr->errArgs = errArgsValue;
    ptr->pyExceptMsg = pyExceptMsgValue;
    ptr->compile_info_key = compile_info_key;
    ptr->compile_info_str = compileInfo;
    ptr->compileRetPtr = std::make_shared<CompileResult>();
    ptr->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    return ptr;
}


te::fusion::OpBuildTaskResultPtr ParseOpTaskResult_stub4(PyObject *pyRes)
{
    std::string compile_info_key;
    std::string compileInfo;

    printf("Get in ParseOpTaskResult_stub4. \r\n");
    std::string pyExceptMsgValue = "\"\'errCode\': EC0000, \'message\': error occurs.\"";
    pyExceptMsgValue += "stub File \"/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc\"";
    for (size_t i = 0; i < 20; i++) {
      pyExceptMsgValue += "/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc";
    }
    pyExceptMsgValue += "stub File \"/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc\"";
    for (size_t i = 0; i < 5; i++) {
      pyExceptMsgValue += "/usr/local/python3.7/lib/site-packages/te_fusion/fusion_util.cc";
    }

    std::string errArgsValue = "input0: {\"L1_addr_offset\":0,\"L1_fusion_type\":-1,\"data_type\":\"float16\"";
    errArgsValue +=
        "\"format\":\"FRACTAL_NZ\",\"name\":\"-1_0_trans_17__0\",\"ori_format\":\"ND\",\"ori_range\":[[1,1],[5,5]],}";
    errArgsValue += "output0: {\"L1_addr_offset\"";
    std::string prebuiltOptions = "json_str";
    OpBuildTaskResultPtr ptr = std::make_shared<OpBuildTaskResult>();
    ptr->type = 0;
    ptr->graphId = 0;
    ptr->taskId = 0;
    ptr->statusCode = 1;
    ptr->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    ptr->preCompileRetPtr->prebuiltOptions = prebuiltOptions;
    ptr->errArgs = errArgsValue;
    ptr->pyExceptMsg = pyExceptMsgValue;
    ptr->compile_info_key = compile_info_key;
    ptr->compile_info_str = compileInfo;
    ptr->compileRetPtr = std::make_shared<CompileResult>();
    ptr->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    return ptr;
}

TEST(TeFusionUTest, GetFinishedCompilationTask_PreBuild_UT)
{
    ErrorManager::GetInstance().is_init_ = true;
    ErrorManager::GetInstance().error_message_per_work_id_.clear();
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;



    std::string path;
    std::string path_owner = "ut";
    (void)CheckPathValid(path, path_owner);

    // call single op
    finalRes = PreBuildTbeOp(*opInfoPtr, 0, 0);
    EXPECT_EQ(finalRes, false);

    // create another prebuild task
    TbeOpInfo opinfo1(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr1 = std::make_shared<TbeOpInfo>(opinfo1);
    opInfoPtr1->AddInput(opinput1);
    opInfoPtr1->AddInput(opinput2);
    opInfoPtr1->AddInput(opinput3);
    opInfoPtr1->SetNode(nodePoint1);
    finalRes = PreBuildTbeOp(*opInfoPtr1, 0, 0);
    EXPECT_EQ(finalRes, false);

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;

    PyObject_InitTruePyObj_Stub();
    MOCKER(te::fusion::PythonAdapterManager::ParseOpTaskResult)
        .stubs()
        .will(invoke(ParseOpTaskResult_stub));
    finalRes = te::fusion::TeFusionManager::GetInstance()->GetFinishedCompilationTask(0);
    EXPECT_EQ(finalRes, true);
    GlobalMockObject::verify();
    MOCKER(te::fusion::PythonAdapterManager::ParseOpTaskResult)
        .stubs()
        .will(invoke(ParseOpTaskResult_stub3));
    auto task = std::make_shared<te::fusion::OpBuildTask>(
        te::fusion::OpBuildTask { 0, 0, 0, nodeList, opdesc1, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC });
    task->kernel = "123";
    te::fusion::TeFusionManager::GetInstance()->dispatchedTask_.emplace(OpTaskKey(0, 0), task);
    te::fusion::TeFusionManager::GetInstance()->fusionOpsKernel_.emplace(task->kernel, std::make_pair(nullptr, std::vector<OpBuildTaskPtr>()));
    finalRes = te::fusion::TeFusionManager::GetInstance()->GetFinishedCompilationTask(0);
    EXPECT_EQ(finalRes, true);
    std::string errorMessage = ErrorManager::GetInstance().GetErrorMessage();
    cout << "Error message is: " << errorMessage << endl;
    size_t pos = errorMessage.find("E49999");
    EXPECT_NE(pos, -1);
    GlobalMockObject::verify();

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;


    ErrorManager::GetInstance().error_message_per_work_id_.clear();
}

TEST(TeFusionUTest, GetFinishedCompilationTask_PreBuild_ErrMsg_UT)
{
    ErrorManager::GetInstance().is_init_ = true;
    ErrorManager::GetInstance().error_message_per_work_id_.clear();
    // 1. create params
    std::vector<Node *> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(input_desc2, { 16, 16 }, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, { 16, 16 }, DT_FLOAT16);
    FillTensorDesc(output_desc2, { 16, 16 }, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer = "1.1.T6.B720";
    std::string opName = "reducenode";
    std::string opModule = "te_fusion.fusion_manager";
    std::string opType = "ReduceSum";
    std::string coreType = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 =
        CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;



    std::string path;
    for (size_t i = 0; i < 150; i++) {
        path += "1234567890abcdefghijklmnopqrstuvwxyz";
    }
    std::string path_owner = "ut";
    (void)CheckPathValid(path, path_owner);

    // call single op
    finalRes = PreBuildTbeOp(*opInfoPtr, 0, 0);
    EXPECT_EQ(finalRes, false);

    // create another prebuild task
    TbeOpInfo opinfo1(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr1 = std::make_shared<TbeOpInfo>(opinfo1);
    opInfoPtr1->AddInput(opinput1);
    opInfoPtr1->AddInput(opinput2);
    opInfoPtr1->AddInput(opinput3);
    opInfoPtr1->SetNode(nodePoint1);
    finalRes = PreBuildTbeOp(*opInfoPtr1, 0, 0);
    EXPECT_EQ(finalRes, false);

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;

    PyObject_InitTruePyObj_Stub();
    MOCKER(te::fusion::PythonAdapterManager::ParseOpTaskResult)
        .stubs()
        .will(invoke(ParseOpTaskResult_stub_1));

    auto task = std::make_shared<te::fusion::OpBuildTask>(
        te::fusion::OpBuildTask { 0, 0, 0, nodeList, opdesc1, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC });
    task->kernel = "123";
    te::fusion::TeFusionManager::GetInstance()->dispatchedTask_.emplace(OpTaskKey(0, 0), task);
    te::fusion::TeFusionManager::GetInstance()->fusionOpsKernel_.emplace(task->kernel, std::make_pair(nullptr, std::vector<OpBuildTaskPtr>()));
    finalRes = te::fusion::TeFusionManager::GetInstance()->GetFinishedCompilationTask(0);
    EXPECT_EQ(finalRes, true);

    std::string errorMessage = ErrorManager::GetInstance().GetErrorMessage();
    cout << "Error message is: " << errorMessage << endl;
    size_t pos = errorMessage.find("E49999");
    EXPECT_NE(pos, -1);
    GlobalMockObject::verify();

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;


    ErrorManager::GetInstance().error_message_per_work_id_.clear();
}

TEST(TeFusionUTest, GetFinishedCompilationTask_BuildOp_UT)
{
    bool finalRes = true;



    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul", "Matmul");
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());

    auto task = std::make_shared<te::fusion::OpBuildTask>(
        te::fusion::OpBuildTask { 0, 0, 0, teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC });

    te::fusion::TeFusionManager::GetInstance()->dispatchedTask_.emplace(OpTaskKey(0, 0), task);

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;

    MOCKER(te::fusion::PythonAdapterManager::ParseOpTaskResult)
        .stubs()
        .will(invoke(ParseOpTaskResult_stub));

    finalRes = te::fusion::TeFusionManager::GetInstance()->GetFinishedCompilationTask(0);
    EXPECT_EQ(finalRes, false);

    GlobalMockObject::verify();

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;


}

TEST(TeFusionUTest, test_upload_py_exception) {
    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;
    te::fusion::HandleManager::Instance().TE_PyTuple_Size = PyTuple_Size_2_Stub;
    PyObject *pyErrTuple = PyTuple_New(2); // ([], "x")
    PyObject *pyErrList = PyList_New(2);
    PyObject *pyErrDict1 = PyDict_New();
    PyObject *key1 = Py_BuildValue("s", "errCode");
    PyObject *value1 = Py_BuildValue("s", "E80001");
    PyDict_SetItem(pyErrDict1, key1, value1);
    PyObject *pyErrDict2 = PyDict_New();
    PyObject *key2 = Py_BuildValue("s", "errCode");
    PyObject *value2 = Py_BuildValue("s", "E70001");
    PyDict_SetItem(pyErrDict2, key2, value2);
    HandleManager::Instance().TE_PyList_SetItem(pyErrList, 0, pyErrDict1);
    HandleManager::Instance().TE_PyList_SetItem(pyErrList, 1, pyErrDict2);
    PyTuple_SetItem(pyErrTuple, 0, pyErrList);
    PyObject *pyOpName = Py_BuildValue("s", "node1");
    PyTuple_SetItem(pyErrTuple, 1, pyOpName);
    std::vector<std::map<std::string, std::string>> mapListArgs;
    te::fusion::PyWrapper::UploadPyException(pyErrTuple, mapListArgs);
}

void UploadPyExceptionUTStub(PyObject *pyObj, std::vector<std::map<std::string, std::string>> &mapListArgs) {
    std::map<std::string, std::string> mapArgs1;
    mapArgs1["type"] ="1";
    mapArgs1["errCode"] ="E90003";
    mapArgs1["detailed_cause"] ="aaa";
    mapListArgs.emplace_back(mapArgs1);
    std::map<std::string, std::string> mapArgs2;
    mapArgs2["type"] ="2";
    mapArgs2["errCode"] ="E68002";
    mapArgs2["keyword"] ="bbb";
    mapListArgs.emplace_back(mapArgs2);
    std::map<std::string, std::string> mapArgs3;
    mapArgs3["keyword"] ="ccc";
    mapListArgs.emplace_back(mapArgs3);
    std::map<std::string, std::string> mapArgs4;
    mapArgs4["errCode"] ="";
    mapArgs4["keyword"] ="ddd";
    mapListArgs.emplace_back(mapArgs4);
}

TEST(TeFusionUTest, GetUniqueJsonStr_UT) {
    nlohmann::json jsonData;
    std::string uniqueJson;
    jsonData["fusion_op_name"] = "";
    jsonData["scope_id"] = "";
    jsonData["graph_name"] = "";
    jsonData["is_dynamic_impl"] = "";
    nlohmann::json device_id;
    device_id["deviceId"] = 1;
    jsonData["SocInfo"] = device_id;
    te::fusion::TeJsonAssemble::GetUniqueJsonStr(jsonData, uniqueJson);
}

TEST(TeFusionUTest, CreatOpCacheIniFile_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string cache_file_real_path = TeConfigInfo::Instance().GetLibPath() + "../op_cache.ini";
    std::string cur_adk_version = "ADK";
    std::string cur_opp_version = "OPP";
    fusion_serial.CreatOpCacheIniFile(cache_file_real_path, cur_adk_version, cur_opp_version);
    if (!te::fusion::RealPath(cache_file_real_path).empty()) {
        EXPECT_EQ(remove(cache_file_real_path.c_str()), 0);
    }
}

TEST(TeFusionUTest, CreatOpCacheIniFile_002)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string cache_file_real_path = "";
    std::string cur_adk_version = "ADK";
    std::string cur_opp_version = "OPP";
    fusion_serial.CreatOpCacheIniFile(cache_file_real_path, cur_adk_version, cur_opp_version);
}

bool GetCurrentVersionInfo_stub1(std::string &adk, std::string &opp)
{
    return true;
}

TEST(TeFusionUTest, DelCachedFiles_002)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string cache_file_version_path = TeConfigInfo::Instance().GetLibPath() + "../version.info";
    int fd = open(cache_file_version_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value1 = "\n";
    int ret = write(fd, value1.c_str(), value1.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value2 = "Version=1\n";
    ret = write(fd, value2.c_str(), value2.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);

    if (!te::fusion::RealPath(cache_file_version_path).empty()) {
        EXPECT_EQ(remove(cache_file_version_path.c_str()), 0);
    }
}

TEST(TeFusionUTest, DelCachedFiles_003)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string cache_file_version_path = TeConfigInfo::Instance().GetLibPath() + "../version.info";
    int fd = open(cache_file_version_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value1 = "\n";
    int ret = write(fd, value1.c_str(), value1.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);

    if (!te::fusion::RealPath(cache_file_version_path).empty()) {
        EXPECT_EQ(remove(cache_file_version_path.c_str()), 0);
    }
}

TEST(TeFusionUTest, DelCachedFiles_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string cache_file_version_path = TeConfigInfo::Instance().GetLibPath() + "../version.info";
    int fd = open(cache_file_version_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value1 = "\n";
    int ret = write(fd, value1.c_str(), value1.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value2 = "Version=1\n";
    ret = write(fd, value2.c_str(), value2.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string op_cache_file_path = TeConfigInfo::Instance().GetLibPath() + "./op_cache.ini";
    fd = open(op_cache_file_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value3 = "[\n";
    ret = write(fd, value3.c_str(), value3.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value5 = "=\n";
    ret = write(fd, value5.c_str(), value5.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value4 = "[op_compiler_cache]\n";
    ret = write(fd, value4.c_str(), value4.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value6 = "adk_version=1\n";
    ret = write(fd, value6.c_str(), value6.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value7 = "ops_version=1\n";
    ret = write(fd, value7.c_str(), value7.length());
    std::string value8 = "max_op_cache_size=5000\n";
    ret = write(fd, value8.c_str(), value8.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value9 = "remain_cache_size_ratio=80\n";
    ret = write(fd, value9.c_str(), value9.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string opp_path = TeConfigInfo::Instance().GetLibPath() + "../";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    std::string cacheDir = TeConfigInfo::Instance().GetLibPath() + "./CMakeFiles/ut_utest_tefusion.dir/";
    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Enable);
    ReplaceFileContent(op_cache_file_path, "remain_cache_size_ratio=80", "remain_cache_size_ratio=101");
    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Enable);
    ReplaceFileContent(op_cache_file_path, "max_op_cache_size=5000", "max_op_cache_size=-1");
    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Force);
    setenv("ASCEND_OPP_PATH", "", 1);
    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Force);

    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Enable);
    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Disable);
    fusion_serial.DelCachedFiles(cacheDir, CompileCacheMode::Force);

    TeFileUtils::FcntlLockFileSet(fd, F_UNLCK, 0);

    if (!te::fusion::RealPath(op_cache_file_path).empty()) {
        EXPECT_EQ(remove(op_cache_file_path.c_str()), 0);
    }
    if (!te::fusion::RealPath(cache_file_version_path).empty()) {
        EXPECT_EQ(remove(cache_file_version_path.c_str()), 0);
    }
}

TEST(TeFusionUTest, CloseCacheFunction_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string cache_file_version_path = TeConfigInfo::Instance().GetLibPath() + "../version.info";
    int fd = open(cache_file_version_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value1 = "\n";
    int ret = write(fd, value1.c_str(), value1.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value2 = "Version=1\n";
    ret = write(fd, value2.c_str(), value2.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string op_cache_file_path = TeConfigInfo::Instance().GetLibPath() + "./op_cache.ini";
    fd = open(op_cache_file_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value3 = "[\n";
    ret = write(fd, value3.c_str(), value3.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value5 = "=\n";
    ret = write(fd, value5.c_str(), value5.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value4 = "[op_compiler_cache]\n";
    ret = write(fd, value4.c_str(), value4.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value6 = "adk_version=1\n";
    ret = write(fd, value6.c_str(), value6.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value7 = "ops_version=1\n";
    ret = write(fd, value7.c_str(), value7.length());
    std::string value8 = "max_op_cache_size=5000\n";
    ret = write(fd, value8.c_str(), value8.length());
    EXPECT_EQ(ret >= 0, true);
    std::string value9 = "remain_cache_size_ratio=80\n";
    ret = write(fd, value9.c_str(), value9.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string opp_path = TeConfigInfo::Instance().GetLibPath() + "../";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    std::string cacheDir = TeConfigInfo::Instance().GetLibPath() + "./CMakeFiles/ut_utest_tefusion.dir/";
    ReplaceFileContent(op_cache_file_path, "max_op_cache_size=5000", "max_op_cache_size=-1");
    std::string a = "-1";
    std::string b = "80";
    fusion_serial.UpdateOpCacheSizeCfg(a, b);
    EXPECT_EQ(fusion_serial.maxOpCacheSize_, -1);
}

// TEST(TeFusionUTest, CacheSpaceInitialize_001)
// {
//     TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
//     setenv("ASCEND_MAX_OP_CACHE_SIZE", "1", 1);
//     std::string cacheDir = TeConfigInfo::Instance().GetLibPath() + "../../";
//     bool res = fusion_serial.CacheSpaceInitialize(cacheDir);
//     EXPECT_EQ(res, true);
// }

TEST(TEFUSION, AgingCacheFileByAccessTime_001) {
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    uint64_t defaultTime = 1698927000;
    std::multimap<uint64_t, CacheFileSizeInfo> filesStatInfo;
    CacheFileSizeInfo ageFileInfo;
    ageFileInfo.jsonFilePath = TeConfigInfo::Instance().GetLibPath() + "../op_cache.json";
    FileInfo file1;
    file1.filePath = TeConfigInfo::Instance().GetLibPath() + "../op_cache.json";
    file1.fileSize = 100;
    FileInfo file2;
    file2.filePath = TeConfigInfo::Instance().GetLibPath() + "../op_cache.o";
    file2.fileSize = 100;
    ageFileInfo.totalFileSizeInfos.emplace_back(file1);
    ageFileInfo.totalFileSizeInfos.emplace_back(file2);
    filesStatInfo.insert(std::make_pair(defaultTime, ageFileInfo));
    fusion_serial.AgingCacheFileByAccessTime(filesStatInfo, 50);
    std::string checkres = RealPath(ageFileInfo.jsonFilePath);
    if (checkres != "") {
        fusion_serial.AgingCacheFileByAccessTime(filesStatInfo, 0);
        std::string checkres = RealPath(ageFileInfo.jsonFilePath);
    }
    EXPECT_EQ(checkres, "");
    // create json file
    int fd = open(ageFileInfo.jsonFilePath.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value = "[op_compiler_cache]\n";
    int ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    value = "adk_version=1\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    value = "ops_version=1\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    struct utimbuf newTimes;
    newTimes.actime = defaultTime;
    newTimes.modtime = defaultTime;
    utime(file1.filePath.c_str(), &newTimes);
    fusion_serial.AgingCacheFileByAccessTime(filesStatInfo, 0);
    checkres = RealPath(ageFileInfo.jsonFilePath);
    EXPECT_EQ(checkres, "");
}

TEST(TeFusionUTest, DelCacheFileByAccessTime_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string op_cache_file_path = TeConfigInfo::Instance().GetLibPath() + "../op_cache.o";
    int fd = open(op_cache_file_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value = "[op_compiler_cache]\n";
    int ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    value = "adk_version=1\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    value = "ops_version=1\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string cachePath = TeConfigInfo::Instance().GetLibPath() + "../";
    uint64_t sizeToDel = 0;
    fusion_serial.DelCacheFileByAccessTime(cachePath, sizeToDel);
    if (!te::fusion::RealPath(op_cache_file_path).empty()) {
        EXPECT_EQ(remove(op_cache_file_path.c_str()), 0);
    }
}

TEST(TeFusionUTest, DelCacheFileByAccessTime_002)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string op_cache_file_path = TeConfigInfo::Instance().GetLibPath() + "../op_cache.o";
    int fd = open(op_cache_file_path.c_str(), O_CREAT | O_RDWR, 0640);
    std::string value = "[op_compiler_cache]\n";
    int ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    value = "adk_version=1\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    value = "ops_version=1\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string op_cache_json_file_path = TeConfigInfo::Instance().GetLibPath() + "../op_cache.json";
    fd = open(op_cache_json_file_path.c_str(), O_CREAT | O_RDWR, 0640);
    value = "{\"adk_version\":1}\n";
    ret = write(fd, value.c_str(), value.length());
    EXPECT_EQ(ret >= 0, true);
    close(fd);
    std::string cachePath = TeConfigInfo::Instance().GetLibPath() + "../";
    uint64_t sizeToDel = 0;
    fusion_serial.DelCacheFileByAccessTime(cachePath, sizeToDel);
    if (!te::fusion::RealPath(op_cache_file_path).empty()) {
        EXPECT_EQ(remove(op_cache_file_path.c_str()), 0);
    }
    if (!te::fusion::RealPath(op_cache_json_file_path).empty()) {
        EXPECT_EQ(remove(op_cache_json_file_path.c_str()), 0);
    }
}

TEST(TeFusionUTest, GetFileSizeInfo_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::string jsonFilePath = currentFilePath + "/disk_cache/kernel_meta/matmul_add.json";

    FileInfo info;
    fusion_serial.GetFileSizeInfo(jsonFilePath, info);
}

TEST(TeFusionUTest, GetNpuCollectPath)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string npu_collect_dir = TeConfigInfo::Instance().GetLibPath() + "npuCollect";
    setenv("NPU_COLLECT_PATH", npu_collect_dir.c_str(), 1);
    TeConfigInfo::Instance().InitEnvItem();
    if (te::fusion::RealPath(npu_collect_dir).empty()) {
        EXPECT_EQ(mkdir(const_cast<char *>(npu_collect_dir.c_str()), S_IRWXU | S_IRGRP | S_IXGRP), 0);
    }
    EXPECT_EQ(te::fusion::RealPath(npu_collect_dir).empty(), false);
    std::string npuCollectPath = GetNpuCollectPath();
    EXPECT_EQ(npuCollectPath.empty(), false);
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::string kernelMetaPath = currentFilePath + "/stub/";
    bool res = TeFileUtils::CreateMultiLevelDir(npuCollectPath);
    EXPECT_EQ(res, true);
    res = TeFileUtils::CopyDirFileToNewDir(kernelMetaPath, npuCollectPath);
    EXPECT_EQ(res, true);
    if (!te::fusion::RealPath(npu_collect_dir).empty()) {
        te::fusion::TeFileUtils::DeleteFile(npu_collect_dir);
    }
}

TEST(TeFusionUTest, SetConstValueToJson_001)
{
    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensor("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    tensor.SetConstValueNone(true);
    TbeAttrValue attr_v_int32("axis", (int32_t)1);
    tensor.SetConstValueRange(attr_v_int32);
    tensor.SetConstValue(attr_v_int32);
    json inputDescJson;
    TeJsonAssemble::InputsTensorDescToJson(tensor, inputDescJson);
}

TEST(TeFusionUTest, SetConstValueToJson_002)
{
    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensor("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    TbeAttrValue attr_v_int32("axis", (int32_t)1);
    tensor.SetConstValue(attr_v_int32);
    json inputDescJson;
    TeJsonAssemble::InputsTensorDescToJson(tensor, inputDescJson);
}

TEST(TeFusionUTest, IsFilePathValid_001)
{
    std::string path = "";
    int res = IsFilePathValid(path);
    EXPECT_EQ(res, 0);
    path = "./test/";
    res = IsFilePathValid(path);
    EXPECT_EQ(res, 1);
}

TEST(TeFusionUTest, DirPath_001)
{
    std::string path = "";
    std::string realPath;
    std::string res = DirPath(path);
    EXPECT_EQ(res, "");
    path = "./test/test.json";
    res = DirPath(path);
    EXPECT_EQ(res, "./test");
}

TEST(TeFusionUTest, PrintPyException)
{
    PyWrapper::PrintPyException(false);
}

TEST(TeFusionUTest, Generate_Dict_FromContext_1)
{
    map<string, string> options;
    options.emplace("opt_module.op_tune", "ALL");
    ge::GetThreadLocalContext().SetGraphOption(options);
    PyObject *contextDict = GenerateDictFromContext();
    AUTO_PY_DECREF(contextDict);
    EXPECT_NE(contextDict, nullptr);
}

TEST(TeFusionUTest, Generate_Dict_FromContext_2)
{
    map<string, string> options;
    options.emplace("opt_module.op_tune", "");
    options.emplace("opt_module.rl_tune", "ALL");
    ge::GetThreadLocalContext().SetGraphOption(options);
    PyObject *contextDict = GenerateDictFromContext();
    AUTO_PY_DECREF(contextDict);
    EXPECT_NE(contextDict, nullptr);
}

bool GetConvElewise_stub(te::fusion::TeFusionManager *This, std::vector<ge::Node *> teGraphNode,
    std::vector<ge::Node *> &elewiseNode, uint32_t inputGroupIdx, bool &has_fixpipe)
{
    printf("Get in GetConvElewise_stub!!!! \r\n");
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("reducenode", "ReduceSum");
    ge::NodePtr node = std::make_shared<ge::Node>(opdesc, owner_graph);
    elewiseNode.push_back(node.get());
    has_fixpipe = true;
    return true;
}

bool GetDxDreluNode_stub(te::fusion::TeFusionManager *This, std::vector<ge::Node *> teGraphNode,
    std::vector<ge::Node *> &dReluNode)
{
    printf("Get in GetDxDreluNode_stub!!!! \r\n");
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("reducenode", "ReduceSum");
    ge::NodePtr node = std::make_shared<ge::Node>(opdesc, owner_graph);
    dReluNode.push_back(node.get());
    return true;
}

bool CheckCurrNode_stub(te::fusion::TeFusionManager *This, std::vector<ge::Node *> &elewiseNode,
    std::vector<ge::Node *> &dReluNode, const ge::Node *currNode, bool &transShape4hd, bool &isDRelu)
{
    printf("Get in CheckCurrNode_stub!!!! \r\n");
    transShape4hd = true;
    isDRelu = true;
    return true;
}

TEST(TeFusionUTest, ChangeInputDescShape_001)
{
    std::vector<ge::Node *> nodeList;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("reducenode", "ReduceSum");
    ge::AttrUtils::SetStr(opdesc, ATTR_NAME_OP_PATTERN, "CommonReduce");
    ge::NodePtr node = std::make_shared<ge::Node>(opdesc, owner_graph);
    nodeList.push_back(node.get());

    std::vector<int64_t> currShape;
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(-1);
    currShape.push_back(4);
    currShape.push_back(4);

    bool res = false;
    uint32_t inputGroupIdx = 0;
    nlohmann::json inputDescJson;
    std::pair<int64_t, int64_t> range1 = { 1, 10 };
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);

    res = te::fusion::ChangeInputDescShape(nodeList, node.get(), currShape, inputGroupIdx,
        inputDescJson);
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();
}

TEST(TeFusionUTest, ChangeInputDescShape_002)
{
    std::vector<ge::Node *> nodeList;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("reducenode", "ReduceSum");
    ge::AttrUtils::SetStr(opdesc, ATTR_NAME_OP_PATTERN, "CommonReduce");
    ge::NodePtr node = std::make_shared<ge::Node>(opdesc, owner_graph);
    nodeList.push_back(node.get());

    std::vector<int64_t> currShape;
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(-1);
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(4);

    bool res = false;
    uint32_t inputGroupIdx = 0;
    nlohmann::json inputDescJson;
    std::pair<int64_t, int64_t> range1 = { 1, 10 };
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);

    res = te::fusion::ChangeInputDescShape(nodeList, node.get(), currShape, inputGroupIdx,
        inputDescJson);
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();
}

TEST(TeFusionUTest, ChangeInputDescShape_003)
{
    std::vector<ge::Node *> nodeList;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("reducenode", "ReduceSum");
    ge::AttrUtils::SetStr(opdesc, ATTR_NAME_OP_PATTERN, "CommonReduce");
    ge::NodePtr node = std::make_shared<ge::Node>(opdesc, owner_graph);
    nodeList.push_back(node.get());

    std::vector<int64_t> currShape;
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(-1);
    currShape.push_back(4);

    bool res = false;
    uint32_t inputGroupIdx = 0;
    nlohmann::json inputDescJson;
    std::pair<int64_t, int64_t> range1 = { 1, 10 };
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);

    res = te::fusion::ChangeInputDescShape(nodeList, node.get(), currShape, inputGroupIdx,
        inputDescJson);
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();
}

TEST(TEFUSION, ChangeInputDescShape_004)
{
    std::vector<ge::Node *> nodeList;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>("reducenode","ReduceSum");
    ge::AttrUtils::SetStr(opdesc, ATTR_NAME_OP_PATTERN, "CommonReduce");
    ge::NodePtr node = std::make_shared<ge::Node>(opdesc, owner_graph);
    nodeList.push_back(node.get());

    std::vector<int64_t> currShape;
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(4);
    currShape.push_back(4);

    bool res = false;
    uint32_t inputGroupIdx = 1;
    nlohmann::json inputDescJson;
    inputDescJson["data_type"] = "uint1";
    std::pair<int64_t, int64_t> range1 = {1,10};
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);
    inputDescJson["range"].push_back(range1);

    res = te::fusion::ChangeInputDescShape(nodeList, node.get(), currShape, inputGroupIdx, inputDescJson);
    EXPECT_EQ(res, true);
    GlobalMockObject::verify();
}

TEST(TeFusionUTest, Get_String_dict_Py_obj)
{
    std::map<std::string, std::string> implMode;
    implMode.emplace(std::make_pair("AvgPoolV2", "high_performance"));
    PythonApiCallPtr pyApiCall = make_shared<te::fusion::PythonApiCall>();
    PyObject *pyImplMode = pyApiCall->GetStrDictPyObj(implMode);
    EXPECT_TRUE(pyImplMode != nullptr);
}

TEST(TeFusionUTest, FcntlLockFileUt)
{
    (void)TeFileUtils::FcntlLockFileSet(-1, F_UNLCK, 0); // F_UNLCK fail
    (void)TeFileUtils::FcntlLockFileSet(-1, F_RDLCK, 0); // F_RDLCK fail
    (void)TeFileUtils::IsFileFcntlLock(-1);              // F_GETLCK fail

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::string filePath = currentFilePath + "/test.txt";

    FILE *fp = fopen(filePath.c_str(), "a+");
    (void)TeFileUtils::IsFileFcntlLock(fileno(fp));

    bool res = TeFileUtils::FcntlLockFileSet(fileno(fp), F_RDLCK, 0);
    EXPECT_EQ(res, true);
    res = TeFileUtils::IsFileFcntlLock(fileno(fp));
    EXPECT_EQ(res, false); // locked process get lock will return unlock

    TeFileUtils::FcntlLockFileSet(fileno(fp), F_UNLCK, 0);

    fclose(fp);
    te::fusion::TeFileUtils::DeleteFile(filePath);
}

TEST(TeFusionUTest, InputsDescToJsonProcess_dynamicInput_Ut)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    std::string keyStr = "_pattern";
    ge::AttrUtils::SetStr(node->GetOpDesc(), keyStr, "x");

    TbeOpParam input1 = TbeOpParam();
    std::vector<int64_t> shape;
    TbeOpTensor tensor1 = TbeOpTensor("tensor1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    input1.SetTensor(tensor1);
    TbeOpInfoPtr opInfo = std::make_shared<TbeOpInfo>("TestOp", "", "TestOp", "");
    input1.SetType(static_cast<te::TensorType>(2)); // TT_DYM
    opInfo->AddInput(input1);
    TbeOpParam input2 = TbeOpParam();
    input2.SetTensor(tensor1);
    TbeOpTensor tensor2 = TbeOpTensor("tensor2", shape, "float16", "NCHW", ATTR_SHAPE_LIST);
    input2.SetTensor(tensor2);
    TbeOpTensor tensor3 = TbeOpTensor("tensor3", shape, "float16", "NHWC", ATTR_SHAPE_LIST);
    input2.SetTensor(tensor3);
    input2.SetType(static_cast<te::TensorType>(2)); // TT_DYM
    opInfo->AddInput(input2);
    TbeOpParam input3 = TbeOpParam();
    input3.SetTensor(tensor1);
    input3.SetType(static_cast<te::TensorType>(2)); // TT_DYM
    opInfo->AddInput(input3);

    ge::Node *teGraphNode = node.get();
    std::unordered_set<ge::Node *> allNodes;
    allNodes.emplace(teGraphNode);
    std::vector<TbeOpParam> inputs;
    opInfo->GetInputs(inputs);
    uint32_t inputSize = 3;
    std::string keyName = "testkeyname";
    std::vector<std::string> inputNameList;
    inputNameList.push_back("inputNameList1");
    inputNameList.push_back("inputNameList2");
    inputNameList.push_back("inputNameList3");
    std::map<int, int> inputOrder;
    inputOrder.insert(pair<int, int>(0, 1));
    inputOrder.insert(pair<int, int>(1, 1));
    inputOrder.insert(pair<int, int>(2, 1));
    nlohmann::json jsonStr;
    std::map<string, int> allInputNames;

    // test dyn_order[iter->second] += 1;
    InputDescJsonParam inputs_json_param(keyName, teGraphNode);
    inputs_json_param.inputs = inputs;
    inputs_json_param.inputSize = inputSize;
    inputs_json_param.inputNameList = inputNameList;
    inputs_json_param.inputOrder = inputOrder;
    inputs_json_param.inputsLinkKeyList = inputNameList;
    bool res = TeJsonAssemble::InputsDescToJsonProcess(allNodes, inputs_json_param, jsonStr, allInputNames);
    EXPECT_EQ(res, true);
    // check jsonStr dyn_order[1] is 2 jsonStr include tensor[0] tensor[1] tensor[2] inputDescJson["dyn_index"]=1
    if (!jsonStr.is_array() || jsonStr.size() != 3) {
        EXPECT_EQ(false, true);
    }
    int index = 0;
    for (auto it = jsonStr.begin(); it != jsonStr.end(); ++it) {
        json inputDesc = *it;
        if (index == 0) {
            EXPECT_EQ(inputDesc["format"], "ND");
            EXPECT_EQ(inputDesc["name"], "inputNameList1");
        }
        if (index == 1) {
            EXPECT_EQ(inputDesc["format"], "NCHW");
            EXPECT_EQ(inputDesc["name"], "inputNameList2");
        }
        if (index == 2) {
            EXPECT_EQ(inputDesc["format"], "NHWC");
            EXPECT_EQ(inputDesc["name"], "inputNameList3");
        }
        index++;
    }
}

bool CompileCacheReuseCheck_stub(te::fusion::TeFusionManager *This, const OpBuildTaskPtr &opTask) {
    return true;
}

TEST(TeFusionUTest, GetJsonValueFromJsonFileFailed) {
    std::string cacheFile = "./temp.json";
    nlohmann::json jsonValue;
    bool res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(cacheFile, jsonValue);
    EXPECT_EQ(res, false);

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    cacheFile = currentFilePath + "/disk_cache/kernel_meta" +
                "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd777_temp.json";
    res = te::fusion::CreateFile(cacheFile);
    EXPECT_EQ(res, true);
    res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(cacheFile, jsonValue);
    EXPECT_EQ(res, false);
    te::fusion::TeFileUtils::DeleteFile(cacheFile);
}

void AddTensorToOpDescV2(bool isInput, std::string name, vector<int64_t> shape, Format format,  DataType data_type,
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

void AddOpParamToTbeOpInfoV2(std::vector<int64_t> shape, std::string dtype, std::string format,std::string name,
                             std::vector<std::pair<int64_t, int64_t>> &range, bool isInput, TbeOpInfo &op_info)
{
    TbeOpParam opParam;
    std::vector<TbeOpTensor> tensors;
    TbeOpTensor tensor(name, shape, dtype, format);
    tensor.SetShapeRange(range);
    tensor.SetOriginShape(shape);
    tensor.SetOriginShapeRange(range);
    tensor.SetOriginFormat(format);

    if (name == "inputx") {
        int32_t val = 999;
        TbeAttrValue constVal("const_value", val);
        tensor.SetConstValue(constVal);
    }

    if (name == "inputy") {
        std::vector<std::vector<int64_t>> valueRange;
        std::vector<int64_t> tmp = {3, 18};
        valueRange.emplace_back(tmp);
        TbeAttrValue constVal("const_value_range", valueRange);
        tensor.SetConstValueRange(constVal);
    }

    tensors.push_back(tensor);
    opParam.SetTensors(tensors);
    if (name == "input1") {
        opParam.SetValueDepend(VALUE_DEPEND_OPTIONAL);
        opParam.SetType(TT_OPT);
    }

    if (name == "input2") {
        opParam.SetValueDepend(VALUE_DEPEND_REQUIRED);
        opParam.SetType(TT_REQ);
    }

    if (name == "input3") {
        opParam.SetValueDepend(VALUE_DEPEND_IGNORE);
        opParam.SetType(TT_DYN);
    }

    if (isInput) {
        op_info.AddInput(opParam);
    } else {
        op_info.AddOutput(opParam);
    }
}

TEST(TeFusionUTest, TaskFusion_001) {
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    (void)ge::AttrUtils::SetStr(opDescPtr, "_pattern", "ElemWise");
    (void)ge::AttrUtils::SetStr(opDescPtr, "_kernelname", "TaskFusion_001_add_kernelName");
    vector<int64_t> taskArgs = {11, 12, 13, 21, 31, 41};
    (void)ge::AttrUtils::SetListInt(opDescPtr, "_task_args", taskArgs);
    std::string headFilePath = "/home/xxx/kernel1.h";
    (void)ge::AttrUtils::SetStr(opDescPtr, "_head_file_path", headFilePath);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);
    tbeOpInfo->SetNode(nodeAdd);

    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    MOCKER_CPP(&te::fusion::PythonApiCall::GenerateStrSha256HashValue,
        bool(te::fusion::PythonApiCall::*)(const std::string &m, std::string &res) const)
        .stubs()
        .will(invoke(GenerateStrSha256HashValue_stub));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeAdd.get());

    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, INVALID_SGT_INDEX,
                                                   teGraphNode, nullptr, OP_TASK_STATUS::OP_TASK_PENDING});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;

    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("add", tbeOpInfo);
    te::fusion::TeFusionManager::GetInstance()->BuildTaskFusion(opTaskPtr);

    OpBuildResCode res = TaskFusion(teGraphNode, opDescPtr, taskId, graphId);
    EXPECT_EQ(res, OP_BUILD_SUCC);
}

TEST(TeFusionUTest, BuildSuperKernel_001) {
    std::vector<std::pair<int64_t, int64_t>> range;
 
    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    (void)ge::AttrUtils::SetStr(opDescPtr, "_pattern", "ElemWise");
    (void)ge::AttrUtils::SetStr(opDescPtr, "_kernelname", "TaskFusion_001_add_kernelName");
    vector<int64_t> taskArgs = {11, 12, 13, 21, 31, 41};
    (void)ge::AttrUtils::SetListInt(opDescPtr, "_task_args", taskArgs);
    std::string headFilePath = "/home/xxx/kernel1.h";
    (void)ge::AttrUtils::SetStr(opDescPtr, "_head_file_path", headFilePath);
    (void)ge::AttrUtils::SetStr(opDescPtr, "_op_aicore_num", "12");
    (void)ge::AttrUtils::SetStr(opDescPtr, "_op_vectorcore_num", "24");
    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
 
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);
 
    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));
 
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);
    tbeOpInfo->SetNode(nodeAdd);
 
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeAdd.get());
 
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    MOCKER_CPP(&te::fusion::PythonApiCall::GenerateStrSha256HashValue,
        bool(te::fusion::PythonApiCall::*)(const std::string &m, std::string &res) const)
        .stubs()
        .will(invoke(GenerateStrSha256HashValue_stub));
 
    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, INVALID_SGT_INDEX,
                                                   teGraphNode, nullptr, OP_TASK_STATUS::OP_TASK_PENDING});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;
 
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("add", tbeOpInfo);
    te::fusion::TeFusionManager::GetInstance()->BuildTaskFusion(opTaskPtr);
 
    OpBuildResCode res = BuildSuperKernel(teGraphNode, opDescPtr, taskId, graphId);
    EXPECT_EQ(res, OP_BUILD_SUCC);
 
}
 
TEST(TeFusionUTest, BuildSuperKernel_002) {
    std::vector<std::pair<int64_t, int64_t>> range;
 
    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    (void)ge::AttrUtils::SetStr(opDescPtr, "_pattern", "ElemWise");
    (void)ge::AttrUtils::SetStr(opDescPtr, "superkernel_kernelname", "TaskFusion_001_add_kernelName");
    vector<int64_t> taskArgs = {11, 12, 13, 21, 31, 41};
    (void)ge::AttrUtils::SetListInt(opDescPtr, "_task_args", taskArgs);
    std::string headFilePath = "/home/xxx/kernel1.h";
    (void)ge::AttrUtils::SetStr(opDescPtr, "_head_file_path", headFilePath);
    std::string jsonFilePath = "llt/atc/opcompiler/te_fusion/py_ut/json_files/add.json";
    (void)ge::AttrUtils::SetStr(opDescPtr, "json_file_path", jsonFilePath);
    std::string binFilePath = "llt/atc/opcompiler/te_fusion/py_ut/json_files/add.o";
    (void)ge::AttrUtils::SetStr(opDescPtr, "bin_file_path", binFilePath);
    std::vector<int64_t> receiveTasks = {1};
    (void)ge::AttrUtils::SetListInt(opDescPtr, "_sk_rcv_event_ids", receiveTasks);
    std::vector<int64_t> sendTasks = {1};
    (void)ge::AttrUtils::SetListInt(opDescPtr, "_sk_send_event_ids", sendTasks);
    (void)ge::AttrUtils::SetStr(opDescPtr, "_op_aicore_num", "12");
    (void)ge::AttrUtils::SetStr(opDescPtr, "_op_vectorcore_num", "24");
    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
 
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);
 
    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));
 
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);
    tbeOpInfo->SetNode(nodeAdd);
 
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeAdd.get());
 
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    MOCKER_CPP(&te::fusion::PythonApiCall::GenerateStrSha256HashValue,
        bool(te::fusion::PythonApiCall::*)(const std::string &m, std::string &res) const)
        .stubs()
        .will(invoke(GenerateStrSha256HashValue_stub));
 
    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, INVALID_SGT_INDEX,
                                                   teGraphNode, nullptr, OP_TASK_STATUS::OP_TASK_PENDING});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;
 
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("add", tbeOpInfo);
    te::fusion::TeFusionManager::GetInstance()->BuildTaskFusion(opTaskPtr);
    te::fusion::OpBuildTaskResultPtr opRes = std::make_shared<te::fusion::OpBuildTaskResult>();
    opRes->statusCode = 1;
    opRes->compileRetPtr = std::make_shared<CompileResult>();
    opTaskPtr->opRes = opRes;
 
    string kernel_name = "TaskFusion_001_add_kernelName";
    string super_kernel_name = "testSuccess";
    (void)te::fusion::SuperKernelTaskManager::GetInstance().superKernelMap_.emplace(super_kernel_name, std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTaskPtr})));
    (void)te::fusion::TeFusionManager::GetInstance()->opKernelMap_.emplace(kernel_name, std::make_pair(opRes, std::vector<OpBuildTaskPtr>({opTaskPtr})));
    OpBuildResCode res = BuildSuperKernel(teGraphNode, opDescPtr, taskId, graphId);
    EXPECT_EQ(res, OP_BUILD_SUCC);
 
}

TEST(TeFusionUTest, BinaryGeneralizeRegistered_001) {
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeAdd.get());

    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                   teGraphNode, opDescPtr, OP_TASK_STATUS::OP_TASK_SUCC});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;
    opTaskPtr->opRes = std::make_shared<OpBuildTaskResult>();

    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("add", tbeOpInfo);
    GeneralizedResult generalizedResult;
    te::fusion::ShapeGeneralization::GeneralizeSingleOp(opTaskPtr, generalizedResult);
}

TEST(TeFusionUTest, BinaryGeneralizeUnregistered_002) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul0","Mul0");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input2", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input3", {-1,-1,-1,3}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr);

    TbeOpInfo op_info("mul0", "", "Mul0", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input2", range, true, op_info);
    AddOpParamToTbeOpInfoV2({-1,-1,-1,3}, "int64", "NHWC", "input3", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);

    std::string strVal = "stringValue";
    TbeAttrValue attrVal("test_attr0", strVal);
    attrVal.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);

    TbeAttrValue attrVal1("test_attr1", true);
    attrVal1.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);
    std::vector<int64_t> val = {1,2,3,4};
    TbeAttrValue attrVal2("test_attr2", val);
    attrVal2.SetAttrSupAllFlag(true);

    TbeAttrValue attrVal3("test_attr3", val);
    attrVal3.SetAttrSupAllFlag(false);

    tbeOpInfo->SetPattern("formatAgnostic");
    tbeOpInfo->AddAttrValue(attrVal);
    tbeOpInfo->AddAttrValue(attrVal1);
    tbeOpInfo->AddAttrValue(attrVal2);
    tbeOpInfo->AddAttrValue(attrVal3);

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMul.get());

    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                   teGraphNode, opDescPtr, OP_TASK_STATUS::OP_TASK_SUCC});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub_ForBinary;

    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("mul0", tbeOpInfo);
    GeneralizedResult generalizedResult;
    te::fusion::ShapeGeneralization::GeneralizeSingleOp(opTaskPtr, generalizedResult);
    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
}

TEST(TeFusionUTest, BinaryGeneralizeUnregistered_003) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul0","Mul0");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input2", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "input3", {-1,-1,-1,3}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr);

    TbeOpInfo op_info("mul0", "", "Mul0", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "input2", range, true, op_info);
    AddOpParamToTbeOpInfoV2({-1,-1,-1,3}, "int64", "NHWC", "input3", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);

    std::string strVal = "stringValue";
    TbeAttrValue attrVal("test_attr0", strVal);
    attrVal.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);

    TbeAttrValue attrVal1("test_attr1", false);
    attrVal1.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);
    std::vector<int64_t> val = {1,2,3,4};
    TbeAttrValue attrVal2("test_attr2", val);
    attrVal2.SetAttrSupAllFlag(true);

    TbeAttrValue attrVal3("test_attr3", val);
    attrVal3.SetAttrSupAllFlag(false);

    tbeOpInfo->SetPattern("testOthers");
    tbeOpInfo->AddAttrValue(attrVal);
    tbeOpInfo->AddAttrValue(attrVal1);
    tbeOpInfo->AddAttrValue(attrVal2);
    tbeOpInfo->AddAttrValue(attrVal3);

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMul.get());

    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                   teGraphNode, opDescPtr, OP_TASK_STATUS::OP_TASK_SUCC});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub_ForBinary;

    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("mul0", tbeOpInfo);
    GeneralizedResult generalizedResult;
    te::fusion::ShapeGeneralization::GeneralizeSingleOp(opTaskPtr, generalizedResult);
    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
}

TEST(TeFusionUTest, BinaryGeneralizeUnregistered_004) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul0","Mul0");
    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr);

    TbeOpInfo op_info("mul0", "", "Mul0", "AIcoreEngine");
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "inputy", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "inputx", range, true, op_info);
    AddOpParamToTbeOpInfoV2({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>(op_info);

    std::string strVal = "stringValue";
    TbeAttrValue attrVal("test_attr0", strVal);
    attrVal.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);

    TbeAttrValue attrVal1("test_attr1", false);
    attrVal1.SetAttrSupAllFlag(true);
    //attrVal.SetAttrHasDefaultValFlag(true);
    std::vector<int64_t> val = {1,2,3,4};
    TbeAttrValue attrVal2("test_attr2", val);
    attrVal2.SetAttrSupAllFlag(true);

    TbeAttrValue attrVal3("test_attr3", val);
    attrVal3.SetAttrSupAllFlag(false);

    tbeOpInfo->SetPattern("testOthers");
    tbeOpInfo->AddAttrValue(attrVal);
    tbeOpInfo->AddAttrValue(attrVal1);
    tbeOpInfo->AddAttrValue(attrVal2);
    tbeOpInfo->AddAttrValue(attrVal3);

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMul.get());

    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    auto opTaskPtr = std::make_shared<OpBuildTask>(OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                   teGraphNode, opDescPtr, OP_TASK_STATUS::OP_TASK_SUCC});
    opTaskPtr->pTbeOpInfo = tbeOpInfo;

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub_ForBinary;

    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::L2Mode)] = "true";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::BufferOptimize)] = "l1_optimize";;
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("mul0", tbeOpInfo);
    GeneralizedResult generalizedResult;
    te::fusion::ShapeGeneralization::GeneralizeSingleOp(opTaskPtr, generalizedResult);
}

TEST(TeFusionUTest, BinaryGeneralize_001) {
    json nulljson0;
    json nulljson1;
    nulljson0["name"] = "test";
    nulljson0["type"] = "test";
    nulljson0["ori_format"] = "test";
    nulljson1["name"] = "test";
    nulljson1["type"] = "test";
    nulljson1["ori_format"] = "test";
    json tmpJson;
    tmpJson.emplace_back(nulljson0);
    tmpJson.emplace_back(nulljson1);
    json tmpJson1;
    tmpJson1.emplace_back(tmpJson);
    json::iterator iter = tmpJson1.begin();
    bool res = te::fusion::ShapeGeneralization::ParseInOrOutputJsonByTensor(tmpJson1, iter, tmpJson1);
    EXPECT_EQ(res, true);
    res = te::fusion::ShapeGeneralization::ParseGeneralizedResToBinaryJson(nulljson0, false, nulljson1, "test");
    EXPECT_EQ(res, false);
    nulljson0["result"] = "test";
    res = te::fusion::ShapeGeneralization::ParseGeneralizedResToBinaryJson(tmpJson, false, nulljson1, "test");
    EXPECT_EQ(res, false);
    json tmpJson3;
    tmpJson3.emplace_back(nulljson1);
    res = te::fusion::ShapeGeneralization::ParseGeneralizedResToBinaryJson(tmpJson3, false, nulljson1, "test");
    EXPECT_EQ(res, false);
    json tmpJson4;
    tmpJson4["inputs"] = tmpJson3;
    tmpJson4["outputs"] = tmpJson3;
    res = te::fusion::ShapeGeneralization::ParseGeneralizedResToBinaryJson(tmpJson1, false, tmpJson4, "test");
    EXPECT_EQ(res, true);

    json attrjson1;
    attrjson1["value"] = "1.0";
    attrjson1["dtype"] = "float";
    json attrjson2;
    attrjson2.emplace_back(attrjson1);
    tmpJson4["attrs"] = attrjson2;
    tmpJson1.clear();
    tmpJson.emplace_back(attrjson1);
    tmpJson1.emplace_back(tmpJson);
    res = te::fusion::ShapeGeneralization::ParseGeneralizedResToBinaryJson(tmpJson1, true, tmpJson4, "test");
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, BinaryGeneralize_002) {
    json nulljson0;
    json nulljson1;
    te::fusion::ShapeGeneralization::GetAttrGeneralizedResFromTensor(nulljson0, nulljson1);
    nulljson1["value"] = "test";
    te::fusion::ShapeGeneralization::GetAttrGeneralizedResFromTensor(nulljson0, nulljson1);
    json tmpJson0;
    tmpJson0.emplace_back(nulljson1);
    auto iter = tmpJson0.begin();
    json tmpJson1;
    tmpJson1.emplace_back(nulljson0);
    te::fusion::ShapeGeneralization::ParseAttrsJsonByTensor(tmpJson0, false, iter, tmpJson1);
}

void BuildOrderedGraph(ge::NodePtr &node0, ge::NodePtr &node1,
                       ge::NodePtr &node2, ge::NodePtr &node3) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::OpDescPtr opDescPtr0 = std::make_shared<ge::OpDesc>("mul0","Mul");
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("mul1","Mul");
    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("mul2","Mul");
    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("mul3","Mul1");

    std::vector<std::pair<int64_t, int64_t>> range;
    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr0);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr0);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr0);

    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);

    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);

    node0 = graph->AddNode(opDescPtr0);
    node1 = graph->AddNode(opDescPtr1);
    node2 = graph->AddNode(opDescPtr2);
    node3 = graph->AddNode(opDescPtr3);
}

void BuildOrderedGraph1(ge::NodePtr &node0, ge::NodePtr &node1,
                        ge::NodePtr &node2, ge::NodePtr &node3) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    ge::OpDescPtr opDescPtr0 = std::make_shared<ge::OpDesc>("mul0","Mul");
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("mul1","Mul");
    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("mul2","Mul1");
    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("mul3","Mul");

    std::vector<std::pair<int64_t, int64_t>> range;
    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr0);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr0);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr0);

    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);

    AddTensorToOpDescV2(true, "inputy", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "inputx", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    node0 = graph->AddNode(opDescPtr0);
    node1 = graph->AddNode(opDescPtr1);
    node2 = graph->AddNode(opDescPtr2);
    node3 = graph->AddNode(opDescPtr3);
}

TEST(TEFUSION, TestGetPeerInOrder0) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());
    allNodes.emplace(node2.get());
    allNodes.emplace(node3.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {"Mul:1", "", "Mul:2", "Mul1:0", ""};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);

    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestGetPeerInOrder1) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());
    allNodes.emplace(node2.get());
    allNodes.emplace(node3.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {"Mul:2", "Mul:1", "", "Mul1:0", ""};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);
    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestGetPeerInOrder1_1) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());
    allNodes.emplace(node2.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {"Mul:2", "", "Mul:1", ""};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);
    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestGetPeerInOrder2) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());
    allNodes.emplace(node2.get());
    allNodes.emplace(node3.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);
    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestGetPeerInOrder3) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);
    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestGetPeerInOrder4) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());
    allNodes.emplace(node2.get());
    allNodes.emplace(node3.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);
    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestGetPeerInOrder5) {
    ge::NodePtr node0;
    ge::NodePtr node1;
    ge::NodePtr node2;
    ge::NodePtr node3;
    BuildOrderedGraph1(node0, node1, node2, node3);

    std::unordered_set<Node *> allNodes;
    allNodes.emplace(node0.get());
    allNodes.emplace(node1.get());
    allNodes.emplace(node2.get());

    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
    std::vector<string> peerInputsOrder;
    std::vector<string> peerInputsOrderExpect = {"Mul:1", "", "Mul1:0", ""};
    TeJsonAssemble::GetPeerInputsOrder(node0->GetOutDataAnchor(0), 0, allNodes, peerInputsOrder);
    EXPECT_EQ(peerInputsOrder, peerInputsOrderExpect);
}

TEST(TEFUSION, TestFusionOpInplace1_NoResueMap) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    std::string keyStr1 = "_pattern";
    ge::AttrUtils::SetStr(nodeConv2D->GetOpDesc(), keyStr1, "x");
    std::string keyStr2 = "_pattern";
    ge::AttrUtils::SetStr(nodeRelu->GetOpDesc(), keyStr2, "x");
    std::string keyStr3 = "_pattern";
    ge::AttrUtils::SetStr(nodeMul->GetOpDesc(), keyStr3, "x");
    std::string keyStr4 = "_pattern";
    ge::AttrUtils::SetStr(nodeMinimum->GetOpDesc(), keyStr4, "x");
    std::string keyStr5 = "_pattern";
    ge::AttrUtils::SetStr(nodeAscendQuant->GetOpDesc(), keyStr5, "x");

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_EQ(relationParam, "");
}

TEST(TEFUSION, TestFusionOpInplace1_ReusedOnly_SUCC) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());
    ge::AttrUtils::SetBool(nodeConv2D->GetOpDesc(), kParamReused, true);
    ge::AttrUtils::SetBool(nodeRelu->GetOpDesc(), kParamReused, true);
    ge::AttrUtils::SetBool(nodeMul->GetOpDesc(), kParamReused, true);
    ge::AttrUtils::SetBool(nodeMinimum->GetOpDesc(), kParamReused, true);
    ge::AttrUtils::SetBool(nodeAscendQuant->GetOpDesc(), kParamReused, true);
    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_NE(relationParam, "");
}

TEST(TEFUSION, TestFusionOpInplace1_ReusedOnly_Fail) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);
    ge::AttrUtils::SetBool(nodeAscendQuant->GetOpDesc(), kParamReused, true);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_EQ(relationParam, "");
}

TEST(TEFUSION, TestFusionOpInplace1_RelationMapOnly_Succ) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap5 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap5->emplace(0, 0);
    nodeAscendQuant->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap5);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap4 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap4->emplace(0, 0);
    nodeMinimum->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap4);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap3 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap3->emplace(0, 1);
    nodeMul->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap3);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_NE(relationParam, "");
}

TEST(TEFUSION, TestFusionOpInplace1_RelationMapOnly_Fail) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap5 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap5->emplace(0, 0);
    nodeAscendQuant->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap5);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap4 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap4->emplace(0, 0);
    nodeMinimum->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap4);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap3 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap3->emplace(0, 0);
    nodeMul->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap3);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_EQ(relationParam, "");
}

TEST(TEFUSION, TestSingleOpInplace1_Reused_Succ) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);
    ge::AttrUtils::SetBool(nodeConv2D->GetOpDesc(), kParamReused, true);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);
    ge::NodePtr nodeEnd = ownerGraph->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateSingleOpRelationParam(nodeConv2D.get(), relationParam);
    EXPECT_NE(relationParam, "");
}

TEST(TEFUSION, TestSingleOpInplace1_Relation_Suss) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap3 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap3->emplace(0, 1);
    nodeConv2D->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap3);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);
    ge::NodePtr nodeEnd = ownerGraph->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateSingleOpRelationParam(nodeConv2D.get(), relationParam);
    EXPECT_NE(relationParam, "");
}

TEST(TEFUSION, TestFusionOpInplace1_ReusedMixRelation_SUCC) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap3 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap3->emplace(0, 1);
    nodeMinimum->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap3);
    ge::AttrUtils::SetBool(nodeAscendQuant->GetOpDesc(), kParamReused, true);
    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_NE(relationParam, "");
}

TEST(TEFUSION, TestFusionOpInplace1_RelationMapOnly_Fail1) {
    std::vector<std::pair<int64_t, int64_t>> range;
    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("Conv2D","Conv2D");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr1);

    ComputeGraphPtr ownerGraph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeConv2D = ownerGraph->AddNode(opDescPtr1);

    ge::OpDescPtr opDescPtr2 = std::make_shared<ge::OpDesc>("Relu","Relu");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr2);
    ge::NodePtr nodeRelu = ownerGraph->AddNode(opDescPtr2);

    ge::OpDescPtr opDescPtr3 = std::make_shared<ge::OpDesc>("Mul1","Mul1");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr3);
    ge::NodePtr nodeMul = ownerGraph->AddNode(opDescPtr3);

    ge::OpDescPtr opDescPtr4 = std::make_shared<ge::OpDesc>("Minimum","Minimum");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr4);
    ge::NodePtr nodeMinimum = ownerGraph->AddNode(opDescPtr4);

    ge::OpDescPtr opDescPtr5 = std::make_shared<ge::OpDesc>("AscendQuant","AscendQuant");
    AddTensorToOpDescV2(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    AddTensorToOpDescV2(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr5);
    ge::NodePtr nodeAscendQuant = ownerGraph->AddNode(opDescPtr5);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap5 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap5->emplace(1, 0);
    nodeAscendQuant->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap5);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap4 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap4->emplace(1, 0);
    nodeMinimum->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap4);

    std::shared_ptr<std::map<int32_t, int32_t>> reusedMap3 = std::make_shared<std::map<int32_t, int32_t>>();
    reusedMap3->emplace(1, 0);
    nodeMul->GetOpDesc()->SetExtAttr(kParamReusedRelation, reusedMap3);

    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");

    AddTensorToOpDescV2(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDescV2(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDescV2(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ge::NodePtr nodeData0 = ownerGraph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = ownerGraph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = ownerGraph->AddNode(input2OpDescPtr);

    ComputeGraphPtr ownerGraphOut = std::make_shared<ComputeGraph>("te1");
    ge::NodePtr nodeEnd = ownerGraphOut->AddNode(input0OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeConv2D->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeConv2D->GetOutDataAnchor(0), nodeRelu->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeRelu->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeData2->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(nodeMul->GetOutDataAnchor(0), nodeMinimum->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeMinimum->GetOutDataAnchor(0), nodeAscendQuant->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeAscendQuant->GetOutDataAnchor(0), nodeEnd->GetInDataAnchor(0));

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2D.get());
    teGraphNode.push_back(nodeRelu.get());
    teGraphNode.push_back(nodeMul.get());
    teGraphNode.push_back(nodeMinimum.get());
    teGraphNode.push_back(nodeAscendQuant.get());

    std::string relationParam;
    te::fusion::TeFusionManager::GetInstance()->GenerateFusionOpRelationParam(teGraphNode, relationParam);
    EXPECT_EQ(relationParam, "");
}

TEST(TeFusionUTest, GetOpModuleName_001) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                          te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    TbeOpInfo opinfo("matmul", "mul%", "Mul", "AICore");
    opinfo.SetRealName("matmul");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opinfo);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("matmul", pTbeOp);

    std::string opModuleNames;
    te::fusion::TeFusionManager::GetInstance()->GetOpModuleName(task, opModuleNames);
    EXPECT_EQ(opModuleNames, "mul%/mul.py");
}

TEST(TeFusionUTest, GenerateSingleNodeKernelName_002) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::AttrUtils::SetStr(opDescPtr, "conv2d_backprop_input_pattern", "Conv2d_backprop_input");
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());

    TbeOpInfo opinfo("conv2d_backprop_input", "conv2d", "Mul", "AICore");
    opinfo.SetRealName("conv2d_backprop_input");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opinfo);
    pTbeOp->SetOpFuncName("conv2d");
    pTbeOp->SetExtraParams("extra_params");
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("conv2d_backprop_input", pTbeOp);
    std::string kernelName;
    std::string jsonStr;
    te::fusion::TeJsonAssemble::GenerateJsonAndKernelName(teGraphNode, true, jsonStr, kernelName);
    std::cout << "jsonStr " << jsonStr << std::endl;
    nlohmann::json jsonFomatStr = json::parse(jsonStr);
    EXPECT_EQ(jsonFomatStr["op_list"][0]["extra_params"], "extra_params");
    std::cout << "kernelName " << kernelName << std::endl;
}

TEST(TeFusionUTest, GenerateSingleNodeKernelName_003) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::AttrUtils::SetStr(opDescPtr, "conv2d_backprop_input_pattern", "Conv2d_backprop_input");
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());

    TbeOpInfo opinfo("conv2d_backprop_input", "conv2d", "Mul", "AICore");
    opinfo.SetRealName("conv2d_backprop_input");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opinfo);
    pTbeOp->SetOpFuncName("conv2d");
    pTbeOp->SetExtraParams("extra_params");
    pTbeOp->SetHashedExtraParams("hashed_extra_params");
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("conv2d_backprop_input", pTbeOp);
    std::string kernelName;
    std::string jsonStr;
    te::fusion::TeJsonAssemble::GenerateJsonAndKernelName(teGraphNode, true, jsonStr, kernelName);
    std::cout << "jsonStr " << jsonStr << std::endl;
    nlohmann::json jsonFomatStr = json::parse(jsonStr);
    EXPECT_EQ(jsonFomatStr["op_list"][0]["extra_params"], "extra_params");
    EXPECT_EQ(jsonFomatStr["op_list"][0]["hashed_extra_params"], "hashed_extra_params");
    std::cout << "kernelName " << kernelName << std::endl;
}

TEST(TeFusionUTest, GetTbeOpInfo_001) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    bool res = te::fusion::TeFusionManager::GetInstance()->SetOpParamsL1Info(nodeMatmul.get());
    EXPECT_EQ(res, false);
}

TEST(TeFusionUTest, test_py_dict_to_map) {
    PyObject* pyObj = PyDict_New();
    PyObject* py_key_account_id = Py_BuildValue("i", 1);
    PyObject* py_value_account_id = Py_BuildValue("i", 1238);
    PyDict_SetItem(pyObj, py_key_account_id, py_value_account_id);
    std::map<std::string, std::string> mapArgs;
    te::fusion::PyWrapper::PydictToMap(pyObj, mapArgs);
}

TEST(TeFusionUTest, test_py_parse_str_list) {
    PyObjectPtr pyStrList;
    std::vector<std::string> vstr;
    te::fusion::PyWrapper::PyParseStrList(pyStrList, vstr);
}

TEST(TeFusionUTest, test_get_input_nodes_desc) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::GeShape shape({1,2,3,4});
    ge::GeTensorDesc input_desc(shape);
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("test","Test");
    opDescPtr->AddInputDesc(input_desc);
    opDescPtr->AddOutputDesc(input_desc);
    ge::NodePtr node = owner_graph->AddNode(opDescPtr);
    GetInputNodesDesc(*node);

    ge::OpDescPtr opDescPtr1 = std::make_shared<ge::OpDesc>("test1","Test");
    opDescPtr1->AddInputDesc(input_desc);
    opDescPtr1->AddOutputDesc(input_desc);
    ge::NodePtr node1 = owner_graph->AddNode(opDescPtr1);
    ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node->GetInDataAnchor(0));
    GetInputNodesDesc(*node);
}

TEST(TeFusionUTest, CacheSpaceInitialize_002)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    setenv("ASCEND_MAX_OP_CACHE_SIZE", "600", 1);
    setenv("ASCEND_REMAIN_CACHE_SIZE_RATIO", "120", 1);
    std::string cacheDir = TeConfigInfo::Instance().GetLibPath() + "../../";
    bool res = fusion_serial.CacheSpaceInitialize(cacheDir);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, CacheSpaceInitialize_003)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    setenv("ASCEND_MAX_OP_CACHE_SIZE", "-1", 1);
    std::string cacheDir = TeConfigInfo::Instance().GetLibPath() + "../../";
    bool res = fusion_serial.CacheSpaceInitialize(cacheDir);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, CacheSpaceInitialize_004)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string temp_env = TeConfigInfo::Instance().env_item_vec_[10];
    TeConfigInfo::Instance().env_item_vec_[10] = "-1";
    fusion_serial.maxOpCacheSize_ = INT64_MIN;
    bool res = fusion_serial.GetCacheSpaceMaxSizeCfg();
    EXPECT_EQ(fusion_serial.maxOpCacheSize_, -1);
    TeConfigInfo::Instance().env_item_vec_[10] = temp_env;
}
 
TEST(TeFusionUTest, CacheSpaceInitialize_005)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string temp_env = TeConfigInfo::Instance().env_item_vec_[10];
    TeConfigInfo::Instance().env_item_vec_[10] = "0";
    fusion_serial.maxOpCacheSize_ = INT64_MIN;
    uint64_t res = fusion_serial.GetCacheSpaceMaxSizeCfg();
    EXPECT_EQ(res, 524288000);
    TeConfigInfo::Instance().env_item_vec_[10] = temp_env;
}
 
TEST(TeFusionUTest, CacheSpaceInitialize_006)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    std::string temp_env = TeConfigInfo::Instance().env_item_vec_[10];
    TeConfigInfo::Instance().env_item_vec_[10] = "pptewr";
    fusion_serial.maxOpCacheSize_ = INT64_MIN;
    fusion_serial.GetCacheSpaceMaxSizeCfg();
    TeConfigInfo::Instance().env_item_vec_[10] = temp_env;
}

TEST(TeFusionUTest, GetNpuCollectPath_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    setenv("NPU_COLLECT_PATH", "op_cache_json_file_path", 1);
    TeConfigInfo::Instance().InitEnvItem();
    std::string npuPath = GetNpuCollectPath();
    EXPECT_EQ(npuPath.empty(), true);
}

bool CheckSupported_stub(te::fusion::PythonApiCall *This, const TbeOpInfo &opinfo, CheckSupportedResult &isSupport,
                         std::string &reason) {
    return false;
}

TEST(TeFusionUTest, OpCheckOpSupported_failed)
{
    bool res = false;
    TbeOpInfo op_info("aa", "", "AA", "AIcoreEngine");
    CheckSupportedInfo checkSupportedInfo;
    MOCKER_CPP(&te::fusion::PythonApiCall::CheckSupported,
               bool (te::fusion::PythonApiCall::*)(const TbeOpInfo &, CheckSupportedResult &, std::string &))
              .stubs()
              .will(invoke(CheckSupported_stub));

    res = CheckOpSupported(op_info, checkSupportedInfo);
    EXPECT_EQ(res, false);
    EXPECT_EQ(checkSupportedInfo.isSupported, NOT_SUPPORTED);
}

TEST(TeFusionUTest, OpCheckOpSupportedByCallingCppFunc)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    CheckSupportedInfo checkSupportedInfo;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("test","Test");
    ge::NodePtr node = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    op_info.SetNode(node); // set dummy nodeptr

    MOCKER(optiling::OpCheckFuncRegistry::GetOpCapability)
        .expects(exactly(3))
        .will(returnValue(OP_CHECK_FUNC(&CheckOpSupportedStubFalse)))
        .then(returnValue(OP_CHECK_FUNC(&CheckOpSupportedStubUnknown)))
        .then(returnValue(OP_CHECK_FUNC(&CheckOpSupportedStubInvalid)));
    // isSupported = False, dynamicCompileStatic = False
    res = CheckOpSupported(op_info, checkSupportedInfo);
    // isSupported = Unknown, dynamicCompileStatic = Invalid
    res = CheckOpSupported(op_info, checkSupportedInfo);
    // isSupported = Invalid
    res = CheckOpSupported(op_info, checkSupportedInfo);
    GlobalMockObject::reset();

    MOCKER(optiling::OpCheckFuncRegistry::GetOpCapability)
        .expects(exactly(4))
        .will(returnValue(OP_CHECK_FUNC(&CheckOpSupportedStub)))
        .then(returnValue(OP_CHECK_FUNC(&OpSelectTbeFormatStub)))
        .then(returnValue(OP_CHECK_FUNC(&OpSelectTbeFormatStub)))
        .then(returnValue(OP_CHECK_FUNC(&OpGetSpecificInfoStub)));
    // isSupported = True, dynamicCompileStatic = True
    res = CheckOpSupported(op_info, checkSupportedInfo);
    EXPECT_EQ(res, true);
    EXPECT_EQ(checkSupportedInfo.isSupported, FULLY_SUPPORTED);
    EXPECT_EQ(checkSupportedInfo.dynamicCompileStatic, true);
    EXPECT_EQ(checkSupportedInfo.allImplChecked, true);
}

TEST(TeFusionUTest, OpSelectTbeFormatByCallingCppFuncFailed)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    std::string opDtypeFormat;

    res = SelectTbeOpFormat(op_info, opDtypeFormat);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, OpSelectTbeFormatByCallingCppFuncSucc)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    std::string opDtypeFormat;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("test","Test");
    ge::NodePtr node = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    op_info.SetNode(node); // set dummy nodeptr
    res = SelectTbeOpFormat(op_info, opDtypeFormat);
    EXPECT_EQ(res, true);
    EXPECT_EQ(opDtypeFormat, "FLOAT : NCHW");
}

TEST(TeFusionUTest, OpGetSpecificInfoByCallingCppFunc)
{
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    std::string opSpecificInfo;

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("test","Test");
    ge::NodePtr node = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    op_info.SetNode(node); // set dummy nodeptr
    res = GetOpSpecificInfo(op_info, opSpecificInfo);
    EXPECT_EQ(res, true);
    EXPECT_EQ(opSpecificInfo, "SpecificInfo : Unsupported");
}

TEST(TeFusionUTest, DumpOpTuneJson_failed)
{
    std::string jsonStr = "atc/opcompiler/te_fusion/ut/testcase/te_fusion_test.cc";
    std::string tuningPath = "./dump_json";
    te::fusion::PythonApiCall::Instance().DumpFusionJson(jsonStr, tuningPath);
}

TEST(TeFusionUTest, get_op_unique_keys_success)
{
    OpDescPtr op = std::make_shared<ge::OpDesc>("TestOp", "TestOp");
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    NodePtr node = graphPtr->AddNode(op);
    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;
    TbeOpInfo op_info("conv", "conv_2d", "Conv2D", "AIcoreEngine");
    op_info.SetNode(node);
    std::vector<std::string> op_unique_keys;
    bool res = GetOpUniqueKeys(op_info, op_unique_keys);
    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;
    cout << "op_unique_keys.size() = " << op_unique_keys.size() << endl;
}

TEST(TeFusionUTest, check_op_impl_mode_supported)
{
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>("add", "add", "Add", "AiCore");
    tbeOpInfo->SetOpImplMode("high_perf");
    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
    te::fusion::PreBuildManager::GetInstance().CheckOpImplMode(tbeOpInfo);
    te::fusion::PreBuildManager::GetInstance().CheckOpImplMode(tbeOpInfo);
    auto iter = te::fusion::PreBuildManager::GetInstance().opImplModeSupportedMap_.find("Add");
    EXPECT_EQ(iter == te::fusion::PreBuildManager::GetInstance().opImplModeSupportedMap_.end(), false);
    EXPECT_EQ(iter->second, true);
}

TEST(TeFusionUTest, check_op_impl_mode_not_supported)
{
    TbeOpInfoPtr tbeOpInfo = std::make_shared<TbeOpInfo>("mul", "mul", "Mul", "AiCore");
    tbeOpInfo->SetOpImplMode("high_precision");
    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
    te::fusion::PreBuildManager::GetInstance().CheckOpImplMode(tbeOpInfo);
    auto iter = te::fusion::PreBuildManager::GetInstance().opImplModeSupportedMap_.find("Mul");
    EXPECT_EQ(iter == te::fusion::PreBuildManager::GetInstance().opImplModeSupportedMap_.end(), true);
}

TEST(TeFusionUTest, UpdateInhibitionInfoForLog)
{
    te::fusion::TeFusionManager::GetInstance()->taskStatisticsTime_ = 0;
    bool res = te::fusion::TeFusionManager::GetInstance()->UpdateInhibitionInfoForLog();
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, update_preops_info) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::AttrUtils::SetBool(opDescPtr, "_is_custom_op", false);
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                          te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    TbeOpInfo opinfo("matmul", "mul%", "Mul", "AICore");
    opinfo.SetRealName("matmul");
    opinfo.SetOpStorePattern("rangeAgnostic");
    opinfo.SetKernelName("secondaryGeneralize");
    task->pPrebuildOp = &opinfo;
    task->buildType = te::ACCURATELY_BUILD;
    TbeOpInfoPtr pTbeOp1 = std::make_shared<TbeOpInfo>(opinfo);
    TbeOpInfoCache::Instance().SetSecondTbeOpInfo("matmul", pTbeOp1);
    opinfo.SetKernelName("preops");
    TbeOpInfoPtr pTbeOp2 = std::make_shared<TbeOpInfo>(opinfo);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("matmul", pTbeOp2);

    te::fusion::TeFusionManager::GetInstance()->UpdatePreops(task);
    ConstTbeOpInfoPtr pTbeOp3 = TbeOpInfoCache::Instance().GetTbeOpInfo("matmul");

    std::string kernelName;
    (void)pTbeOp3->GetKernelName(kernelName);
    EXPECT_EQ(kernelName, "secondaryGeneralize");
}

TEST(TeFusionUTest, ReportErrMsgs)
{
    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                          te::fusion::OP_TASK_STATUS::OP_TASK_FAIL});
    OpBuildTaskResultPtr opResPtr = std::make_shared<OpBuildTaskResult>();
    opResPtr->statusCode = 3;
    task->opRes = opResPtr;
    std::string opModuleNames = "test.test";
    std::string errMsgs = "compiler process died.";
    te::fusion::TeFusionManager::GetInstance()->ReportBuildErrMessage(task, opModuleNames, opResPtr, errMsgs);
}

TEST(TeFusionUTest, IsFileUsed) {
    std::string fileName = "dummy.txt";
    bool res = TeFileUtils::IsFileUsed(fileName);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, Check_prebuilt_options_in_tbeopinfo) {
    // 1. create params
    std::vector<Node*> nodeList;

    // init
    vector<string> inputnameList;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> inputdescList;
    vector<GeTensorDesc> outputdescList;

    // State1: wrong node, mul-output
    // create node
    inputnameList.clear();
    inputnameList.push_back("data_0");
    inputnameList.push_back("data_0:1");

    FillTensorDesc(input_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(input_desc2, {16, 16}, DT_FLOAT16);
    inputdescList.clear();
    inputdescList.push_back(input_desc1);
    inputdescList.push_back(input_desc2);

    FillTensorDesc(output_desc1, {16, 16}, DT_FLOAT16);
    FillTensorDesc(output_desc2, {16, 16}, DT_FLOAT16);
    outputdescList.clear();
    outputdescList.push_back(output_desc1);
    outputdescList.push_back(output_desc2);
    // 2. call funcs
    bool finalRes = true;
    std::map<std::string, std::string> options;
    std::string ddkver_key = "DDK_version";
    std::string ddkVer     = "1.1.T6.B720";
    std::string opName     = "reducenode";
    std::string opModule   = "te_fusion.fusion_manager";
    std::string opType     = "ReduceSum";
    std::string coreType   = "AIcoreEngine";

    OpDescPtr opdesc1;
    opdesc1 = CreateOpDesc(opName, opType, 0, "RT_DEV_BINARY_MAGIC_ELF", 1, inputnameList, 2, inputdescList, outputdescList);

    ComputeGraphPtr graphPtr1 = std::make_shared<ComputeGraph>("3");
    NodePtr nodePoint1 = graphPtr1->AddNode(opdesc1);

    nodeList.push_back(nodePoint1.get());

    TbeOpInfo opinfo(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>(opinfo);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput1(TT_REQ, tensors);
    TbeOpTensor tensorop2("test2", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors2;
    tensors2.push_back(tensorop2);
    TbeOpParam opinput2(TT_OPT, tensors2);
    // set null tensor for op
    std::vector<TbeOpTensor> tensors3;
    TbeOpParam opinput3(TT_OPT, tensors3);

    opInfoPtr->AddInput(opinput1);
    opInfoPtr->AddInput(opinput2);
    opInfoPtr->AddInput(opinput3);

    options[ddkver_key] = ddkVer;

    std::string path;
    std::string path_owner = "ut";
    (void)CheckPathValid(path, path_owner);

    // call single op
    finalRes = PreBuildTbeOp(*opInfoPtr, 0, 0);
    EXPECT_EQ(finalRes, false);

    // create another prebuild task
    TbeOpInfo opinfo1(session_name + "_" + opName, opModule, opType, coreType);
    TbeOpInfoPtr opInfoPtr1 = std::make_shared<TbeOpInfo>(opinfo1);
    opInfoPtr1->AddInput(opinput1);
    opInfoPtr1->AddInput(opinput2);
    opInfoPtr1->AddInput(opinput3);
    opInfoPtr1->SetNode(nodePoint1);
    finalRes = PreBuildTbeOp(*opInfoPtr1, 1, 0);
    EXPECT_EQ(finalRes, false);

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_1_Stub;

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0,
                                                          nodeList, opdesc1, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    task->pPrebuildOp = &opinfo1;
    TeFusionManager *fusion_manager = te::fusion::TeFusionManager::GetInstance();
    fusion_manager->dispatchedTask_.emplace(OpTaskKey(0, 0), task);

    MOCKER(te::fusion::PythonAdapterManager::ParseOpTaskResult)
        .stubs()
        .will(invoke(ParseOpTaskResult_stub4));
    const std::map<OpTaskKey, OpBuildTaskPtr>::const_iterator historyGraphItr = fusion_manager->dispatchedTask_.find(OpTaskKey(0, 0));
    OpBuildTaskPtr relBuildTaskPtr = historyGraphItr->second;
    std::string extraParams = "{\"key1\":\"value1\"}";
    relBuildTaskPtr->pPrebuildOp->SetExtraParams(extraParams);
    relBuildTaskPtr->pTbeOpInfo = opInfoPtr;
    finalRes = fusion_manager->GetFinishedCompilationTask(0);
    EXPECT_EQ(finalRes, true);
    std::string extraParamsTmp;
    (void)relBuildTaskPtr->pPrebuildOp->GetExtraParams(extraParamsTmp);
    EXPECT_EQ(extraParams, extraParamsTmp);

    GlobalMockObject::verify();

    te::fusion::HandleManager::Instance().TE_PyList_Size = PyList_Size_Stub;
}

TEST(TeFusionUTest, Check_prebuilt_options_in_json) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::AttrUtils::SetBool(opDescPtr, "_is_custom_op", false);
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                          te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    TbeOpInfo opinfo("matmul", "mul%", "Mul", "AICore");
    opinfo.SetRealName("matmul");
    task->pPrebuildOp = &opinfo;
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opinfo);
    task->pTbeOpInfo = pTbeOp;
    task->buildType = te::ACCURATELY_BUILD;
    task->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd666_pre";

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;

    OpBuildTaskResultPtr opResPtr = std::make_shared<OpBuildTaskResult>();
    opResPtr->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    opResPtr->preCompileRetPtr->prebuiltOptions = "json_str";
    opResPtr->preCompileRetPtr->opPattern = "Opaque";
    opResPtr->preCompileRetPtr->coreType = "AiCore";

    bool res = te::fusion::TeFusionManager::GetInstance()->SetPreBuildResult(task, opResPtr);
    EXPECT_EQ(res, true);

    std::string cacheFile = TeCacheManager::Instance().cache_dir_path_ +
                                 "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd666_pre.json";
    std::string cacheFilePath = te::fusion::RealPath(cacheFile);
    EXPECT_EQ(cacheFilePath.empty(), false);

    nlohmann::json jsonValue;
    res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(cacheFilePath, jsonValue);
    EXPECT_EQ(jsonValue.at("pattern").get<std::string>(), "Opaque");
    EXPECT_EQ(jsonValue.at("coreType").get<std::string>(), "AiCore");
    EXPECT_EQ(jsonValue.at("prebuilt_options").get<std::string>(), "json_str");
    te::fusion::TeFileUtils::DeleteFile(cacheFile);
}

TEST(TeFusionUtil, FilterJsonValueFromJsonListByKey_will_filter)
{
    json testJson = {
        {{"key", "1"}, {"name", "a"}},
        {{"key", "2"}, {"name", "b"}},
        {{"key", "3"}, {"name", "c"}},
        {{"key", "1"}, {"name", "d"}}
    };

    auto ret = TeJsonUtils::FilterJsonValueFromJsonListByKey(string("key"), string("1"), testJson);
    EXPECT_TRUE(ret);
    for (auto n: testJson) {
        EXPECT_EQ(n["key"], string("1"))
            << "Json node(" << n.dump() << ") not match key.";
    }
}

TEST(TeFusionUtil, DeleteCaxisValuesFromJsonTest) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    //=====================one=========================
    std::string oneCaxisValueFile = currentFilePath +
                                 "/one_caxisc_values.json";
    std::string oneCaxisValueFilePath = te::fusion::RealPath(oneCaxisValueFile);
    EXPECT_EQ(oneCaxisValueFilePath.empty(), false);

    nlohmann::json oneCaxisjsonValue;
    bool res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(oneCaxisValueFilePath, oneCaxisjsonValue);
    te::fusion::TeJsonUtils::DeleteValuesFromJson("caxis_values", oneCaxisjsonValue);
    //=====================two=========================
    std::string twoCaxisValueFile = currentFilePath +
                                 "/two_caxisc_values.json";
    std::string twoCaxisValueFilePath = te::fusion::RealPath(twoCaxisValueFile);
    EXPECT_EQ(twoCaxisValueFilePath.empty(), false);

    nlohmann::json twoCaxisjsonValue;
    res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(twoCaxisValueFilePath, twoCaxisjsonValue);
    te::fusion::TeJsonUtils::DeleteValuesFromJson("caxis_values", twoCaxisjsonValue);
    //=====================no=========================
    std::string noCaxisValueFile = currentFilePath +
                                 "/two_has_no_caxisc_values.json";
    std::string noCaxisValueFilePath = te::fusion::RealPath(noCaxisValueFile);
    EXPECT_EQ(noCaxisValueFilePath.empty(), false);

    nlohmann::json noCaxisjsonValue;
    res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(noCaxisValueFilePath, noCaxisjsonValue);
    te::fusion::TeJsonUtils::DeleteValuesFromJson("caxis_values", noCaxisjsonValue);
    //=====================two-withone=========================
    std::string towWithOneCaxisValueFile = currentFilePath +
                                 "/two_has_one_caxisc_values.json";
    std::string towWithOneCaxisValueFilePath = te::fusion::RealPath(towWithOneCaxisValueFile);
    EXPECT_EQ(towWithOneCaxisValueFilePath.empty(), false);

    nlohmann::json twoWithOneCaxisjsonValue;
    res = te::fusion::TeFileUtils::GetJsonValueFromJsonFile(towWithOneCaxisValueFilePath, twoWithOneCaxisjsonValue);
    te::fusion::TeJsonUtils::DeleteValuesFromJson("caxis_values", twoWithOneCaxisjsonValue);
}

TEST(TeFusionUTest, BuildBinarySingleOp) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = 200;
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul", "Matmul");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddTensorToOpDesc(true, "input0", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(true, "input1", {-1,-1}, FORMAT_NCHW, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(true, "input12", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    teGraphNode.push_back(nodePtr.get());
    std::string opName = teGraphNode[0]->GetName();

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                            teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";
    opTask->buildType = FUZZILY_BUILD;
    opTask->newCompile = true;
    opTask->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd763";

    TbeOpInfo opInfo("matmul", "", "Matmul", "AIcoreEngine");
    AddOpParamToTbeOpInfo({-1,-1}, "int64", "NCHW", "input0", range1, true, opInfo);
    AddOpParamToTbeOpInfo({-1,-1}, "int64", "NCHW", "input1", range1, true, opInfo);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "input12", range, true, opInfo);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().tbeOpInfoMap_.clear();
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::map<std::string, std::string> options;
    options["coreNum"] = 32;
    options["coreType"] = "AiCore";
    options["all_ops_impl_path"] = "built-in/op_impl/ai_core/tbe/impl|customize/op_impl/ai_core/tbe/customize_impl|py_ut_op_stub/impl";
    (void)pTbeOp->SetOptions(options);

    nlohmann::json json_desc;
    nlohmann::json input;
    nlohmann::json output;
    nlohmann::json attr;
    nlohmann::json int64Mode;
    json_desc["kernelName"] = "matmul";
    input["name"] = "x";
    input["dtype"] = "float16";
    input["format"] = "ND";
    output["name"] = "y";
    output["dtype"] = "float16";
    output["format"] = "ND";
    attr["name"] = "offset_x";
    attr["dtype"] = "int";
    attr["value"] = "null";
    json_desc["supportInfo"]["inputs"].emplace_back(input);
    json_desc["supportInfo"]["outputs"].emplace_back(output);
    json_desc["supportInfo"]["attrs"].emplace_back(attr);
    json_desc["supportInfo"]["int64Mode"] = false;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter/stub/binary_stub";
    setenv("ASCEND_OPP_PATH", currentFilePath.c_str(), 1);

    te::fusion::TeConfigInfo::Instance().debugDirs_.push(".");
    TeFusionManager *pInstance = TeFusionManager::GetInstance();

    opTask->generalizedJson = json_desc;
    res = pInstance->BuildBinarySingleOp(opTask);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, BuildBinaryFusionOp) {
    bool res = false;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::AttrUtils::SetStr(opDescPtr, "conv2d_backprop_input_pattern", "Conv2d_backprop_input");
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());
    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                            te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});

    TbeOpInfo opinfo("conv2d_backprop_input", "mul%", "Mul", "AICore");
    opinfo.SetRealName("conv2d_backprop_input");
    TbeOpInfoPtr pTbeOp = std::make_shared<TbeOpInfo>(opinfo);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo("conv2d_backprop_input", pTbeOp);

    std::map<std::string, std::string> options;
    options["coreNum"] = 32;
    options["coreType"] = "AiCore";
    (void)pTbeOp->SetOptions(options);
    pTbeOp->SetNode(nodeConv2d);

    te::fusion::TeConfigInfo::Instance().debugDirs_.push(".");
    TeFusionManager *pInstance = TeFusionManager::GetInstance();

    nlohmann::json json_desc;
    json_desc["kernelName"] = "conv2d_backprop_input";
    opTask->generalizedJson = json_desc;
    res = pInstance->BuildBinaryFusionOp(opTask);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, Test_SetNodeCompileInfoAttr)
{
    MOCKER(AttrUtils::SetStr)
    .expects(exactly(6))
    .will(returnValue(false))
    .then(returnValue(true))
    .then(returnValue(false));

    OpDescPtr newOpDesc = std::make_shared<OpDesc>();
    newOpDesc->SetName("relu");

    OpBuildTaskResultPtr opRes = std::make_shared<OpBuildTaskResult>();

    opRes->compile_info_key = "compile_info_key";
    opRes->compile_info_str = "compile_info_str";

    // test for saving compile_info_json error
    SetNodeCompileInfoAttr(newOpDesc, opRes);

    // test for saving compile_info_key error
    SetNodeCompileInfoAttr(newOpDesc, opRes);
}

TEST(TeFusionUTest, Test_CanReuseBuildBinaryCache_001)
{
    bool res = false;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::AttrUtils::SetStr(opDescPtr, "conv2d_backprop_input_pattern", "Conv2d_backprop_input");
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());
    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                            te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    auto opTask1 = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 1, 0, teGraphNode, opDescPtr,
                                                             te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    opTask->kernel = "te_conv2d__kernel0";
    opTask->isBuildBinarySingleOp = true;
    opTask1->kernel = "te_conv2d__kernel0";
    opTask1->isBuildBinarySingleOp = true;

    bool hasError = false;
    res = te::fusion::TeFusionManager::GetInstance()->CanReuseBuildBinaryCache(opTask, hasError);
    EXPECT_EQ(res, false);

    PyObject *pyRes = nullptr;
    PyObjectPtr resPtr(pyRes, PyWrapper::PyObjectDecRef);
    te::fusion::TeFusionManager::GetInstance()->CheckBuildBinaryOpTaskResult(opTask);
    res = te::fusion::TeFusionManager::GetInstance()->CanReuseBuildBinaryCache(opTask1, hasError);
    EXPECT_EQ(res, true);

    PyObject *pyRes1 = PyDict_New();
    PyObjectPtr resPtr1(pyRes1, PyWrapper::PyObjectDecRef);
    te::fusion::TeFusionManager::GetInstance()->CheckBuildBinaryOpTaskResult(opTask);
    res = te::fusion::TeFusionManager::GetInstance()->CanReuseBuildBinaryCache(opTask1, hasError);
    EXPECT_EQ(res, true);
    res = te::fusion::TeFusionManager::GetInstance()->CanReuseBuildBinaryCache(opTask1, hasError);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, GetNewVersionString_0) {
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    char *tmpLineData = nullptr;
    uint32_t dataLength = 0;
    std::map<std::string, std::string> contentToUpdate = {{"abc", "111"}};
    std::string strFileData = "";
    fusion_serial.GetNewVersionString(tmpLineData, dataLength, contentToUpdate, strFileData); 
}


TEST(TeFusionUTest, Test_CanReuseBuildBinaryCache_002)
{
    bool res = false;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::AttrUtils::SetStr(opDescPtr, "conv2d_backprop_input_pattern", "Conv2d_backprop_input");
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());
    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                            te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    opTask->kernel = "te_conv2d";
    te::fusion::OpBuildTaskResultPtr opRes = std::make_shared<te::fusion::OpBuildTaskResult>();
    opTask->opRes = opRes;
    opTask->isBuildBinarySingleOp = true;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    std::string jsonFileName = currentFilePath + "/kernel_meta/" + opTask->kernel + ".json";
    std::string objFileName = currentFilePath + "/kernel_meta/" + opTask->kernel + ".o";
    res = te::fusion::CreateFile(jsonFileName);
    res = te::fusion::CreateFile(objFileName);

    bool hasError = false;
    res = te::fusion::TeFusionManager::GetInstance()->CanReuseBuildBinaryCache(opTask, hasError);

    te::fusion::TeFileUtils::DeleteFile(jsonFileName);
    te::fusion::TeFileUtils::DeleteFile(objFileName);

    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, Test_CanReuseBuildBinaryCache_003)
{
    bool res = false;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::AttrUtils::SetStr(opDescPtr, "conv2d_backprop_input_pattern", "Conv2d_backprop_input");
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());
    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                            te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    opTask->kernel = "te_conv2d__kernel0";
    opTask->isBuildBinaryFusionOp = true;

    bool hasError = false;
    res = te::fusion::TeFusionManager::GetInstance()->CanReuseBuildBinaryCache(opTask, hasError);

    EXPECT_EQ(res, false);
    ge::NodePtr nodeConv2d_null = std::make_shared<ge::Node>(opDescPtr, nullptr);
    std::string graph_id = GetSessionGraphId(nodeConv2d_null.get());
    EXPECT_EQ(graph_id, "");
}

TEST(TeFusionUTest, Test_JudgeBinKernelInstalled)
{
    auto te_instance = te::fusion::TeFusionManager::GetInstance();
    BinaryManager::Instance().binOmAdkVersionMap_[20] = "test";
    BinaryManager::Instance().binAdkVersionMap_[30] = "test";
    BinaryManager::Instance().binOmVersionMap_[20] = "test";
    BinaryManager::Instance().binOppVersionMap_[30] = "test";
    bool res = BinaryManager::Instance().JudgeBinKernelInstalled(true, 20);
    EXPECT_EQ(res, true);
    res = BinaryManager::Instance().JudgeBinKernelInstalled(true, 30);
    EXPECT_EQ(res, false);
    res = BinaryManager::Instance().JudgeBinKernelInstalled(false, 30);
    EXPECT_EQ(res, true);
    res = IsOppKernelInstalled(true, 20);
    EXPECT_EQ(res, true);
}

TEST(TeFusionUTest, Test_GetAllCompileStatistics)
{
    std::vector<std::string> compileStatistics;
    GetAllCompileStatistics(compileStatistics);
}

TEST(TeFusionUTest, SetMaxOpCahceSize_001)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    fusion_serial.maxOpCacheSize_ = 4;
    int64_t maxOpCacheSize = fusion_serial.GetMaxOpCacheSize();
    EXPECT_EQ(maxOpCacheSize, 4);
}

TEST(TeFusionUTest, SetMaxOpCahceSize_00)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    string str1 = "";
    string str2 = "";
    fusion_serial.UpdateOpCacheSizeCfg(str1, str2);
}

TEST(TeFusionUTest, SetMaxOpCahceSize_01)
{
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    string str1 = "0";
    string str2 = "50";
    fusion_serial.UpdateOpCacheSizeCfg(str1, str2);
}

TEST(TeFusionUTest, GetCacheRemainSizeRadio_001)
{
    unsetenv("ASCEND_REMAIN_CACHE_SIZE_RATIO");
    TeConfigInfo teConfigInfo;
    setenv("ASCEND_REMAIN_CACHE_SIZE_RATIO", "101", 1);
    teConfigInfo.InitEnvItem();
    map<string, string> options;
    teConfigInfo.Initialize(options);
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    fusion_serial.GetCacheRemainSizeRadio();
}

TEST(TeFusionUTest, Test_bin_version_match)
{
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te_hgl");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("conv2d_backprop_input","Conv2d_backprop_input");
    ge::NodePtr nodeConv2d = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeConv2d.get());
    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                            te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    bool bres = te::fusion::BinaryManager::Instance().CheckReuseBinaryCondition(opTask);
    EXPECT_EQ(true, bres);
}

void TestSigHandle(int signo)
{
    std::cout << "test signal handle" << std::endl;
}

TEST(TeFusionUTest, Test_signal_manager_init)
{
    SignalManager::Instance().Finalize();
    TeConfigInfo::Instance().env_item_vec_[14] = "none";
    SignalManager::Instance().Initialize();
    TeConfigInfo::Instance().env_item_vec_[14] = "all";

    struct sigaction testAct;
    testAct.sa_handler = TestSigHandle;
    sigemptyset(&testAct.sa_mask);
    testAct.sa_flags = 0;
    (void)sigaction(2, &testAct, nullptr);
    SignalManager::Instance().Initialize();
    SignalManager::Instance().kernelTempDirSet_.clear();
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::string kernelMetaTmepPath = currentFilePath + "/kernel_meta/kernel_meta_temp_123456";
    CreateDir(kernelMetaTmepPath);
    SignalManager::Instance().kernelTempDirSet_.emplace(kernelMetaTmepPath);
    ClearTEResource(2);
    SignalManager::Instance().Finalize();
}

TEST(TeFusionUTest, Test_get_Unrepeated_Hash)
{
    std::string kernelHash;
    TeJsonAssemble::GetUnrepeatedHash(kernelHash);
    EXPECT_EQ(kernelHash.empty(), true);
    const std::string someHash = "234h32jh58hgfsdiufn32u5t238regu";
    kernelHash = someHash;
    TeJsonAssemble::GetUnrepeatedHash(kernelHash);
    EXPECT_NE(kernelHash, someHash);
    TeJsonAssemble::GetUnrepeatedHash(kernelHash);
    EXPECT_NE(kernelHash, someHash);
}

TEST(TeFusionUTest, Test_Rm_KernelTempDir)
{
    SignalManager::Instance().kernelTempDirSet_.clear();
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    const std::string kernelMetaTmepPath = currentFilePath + "/kernel_meta/kernel_meta_temp_123456777777";
    CreateDir(kernelMetaTmepPath);
    SignalManager::Instance().SaveKernelTempDir(kernelMetaTmepPath);
    SignalManager::Instance().RmKernelTempDir(kernelMetaTmepPath.data());
}

TEST(TeFusionUTest, sync_optune_params_case)
{
    EXPECT_EQ(TeFusionManager::SyncOpTuneParams(), true);
    ge::GetThreadLocalContext().graph_options_[ge::OP_BANK_UPDATE_FLAG] = "true";
    EXPECT_EQ(TeFusionManager::SyncOpTuneParams(), true);
    ge::GetThreadLocalContext().graph_options_[ge::OP_BANK_UPDATE_FLAG] = "false";
}

// TEST(TEST_TEFUSION_ST, CheckFuncsCallingOpResApiSuccess)
// {
//     ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
//     ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
//     ge::AttrUtils::SetBool(opDescPtr, "_is_custom_op", false);
//     ge::NodePtr mm_node = std::make_shared<ge::Node>(opDescPtr, owner_graph);

//     gert::ExeResGenerationCtxBuilder exe_ctx_builder;
//     auto res_ptr_holder = exe_ctx_builder.CreateOpCheckContext(*mm_node);
//     EXPECT_NE(res_ptr_holder, nullptr);
//     auto op_check_ctx = reinterpret_cast<gert::OpCheckContext *>(res_ptr_holder->context_);
//     auto node_ptr = op_check_ctx->MutableInputPointer<ge::Node>(0);
//     EXPECT_NE(node_ptr, nullptr);

//     gert::OpImplSpaceRegistryV2Ptr spaceRegistryStub = std::make_shared<gert::OpImplSpaceRegistryV2>();
//     auto registry_holder = std::make_shared<gert::OpImplSpaceRegistry>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;

//     op_impl_func.check_support = CheckOpSupportedV2Stub;
//     op_impl_func.op_select_format = OpSelectTbeFormatV2Stub;
//     op_impl_func.get_op_support_info = OpGetSupportInfoV2Stub;
//     op_impl_func.get_op_specific_info = OpGetSpecificInfoV2Stub;
//     registry_holder->types_to_impl_[ge::AscendString(mm_node->GetType().c_str())] = op_impl_func;
//     spaceRegistryStub->AddRegistry(registry_holder);

//     gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceRegistryStub);

//     bool resCheckOpSupported = false;
//     bool resSelectTbeOpFormat = false;
//     bool resGetOpInfo = LX_QUERY_FAIL;
//     bool resGetOpSpecificInfo = false;
    
//     TbeOpInfo op_info("weightQuantBatchMatmulV2", "", "WeightQuantBatchMatmulV2", "AIcoreEngine");
//     op_info.SetNode(mm_node);

//     CheckSupportedInfo checkSupportedInfo;
//     resCheckOpSupported = CheckOpSupported(op_info, checkSupportedInfo);
//     EXPECT_EQ(resCheckOpSupported, true);
//     EXPECT_EQ(checkSupportedInfo.isSupported, FULLY_SUPPORTED);
//     EXPECT_EQ(checkSupportedInfo.dynamicCompileStatic, true);

//     std::string opDtypeFormat = "";
//     resSelectTbeOpFormat = SelectTbeOpFormat(op_info, opDtypeFormat);
//     EXPECT_EQ(resSelectTbeOpFormat, true);
//     EXPECT_EQ(opDtypeFormat, "FLOAT : NCHW");

//     std::string opSupportInfo = "";
//     resGetOpInfo = GetOpInfo(op_info, opSupportInfo);
//     EXPECT_EQ(resGetOpInfo, LX_QUERY_SUCC);
//     EXPECT_EQ(opSupportInfo, "SupportInfo : support lxfusion");

//     std::string opSpecificInfo = "";
//     resGetOpSpecificInfo = GetOpSpecificInfo(op_info, opSpecificInfo);
//     EXPECT_EQ(resGetOpSpecificInfo, true);
//     EXPECT_EQ(opSpecificInfo, "SpecificInfo : support lxfusion");
// }

// TEST(TEST_TEFUSION_ST, CheckFuncsCallingOpResApiFail)
// {
//     ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
//     ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
//     ge::AttrUtils::SetBool(opDescPtr, "_is_custom_op", false);
//     ge::NodePtr mm_node = std::make_shared<ge::Node>(opDescPtr, owner_graph);

//     gert::ExeResGenerationCtxBuilder exe_ctx_builder;
//     auto res_ptr_holder = exe_ctx_builder.CreateOpCheckContext(*mm_node);
//     EXPECT_NE(res_ptr_holder, nullptr);
//     auto op_check_ctx = reinterpret_cast<gert::OpCheckContext *>(res_ptr_holder->context_);
//     auto node_ptr = op_check_ctx->MutableInputPointer<ge::Node>(0);
//     EXPECT_NE(node_ptr, nullptr);

//     gert::OpImplSpaceRegistryV2Ptr spaceRegistryStub = std::make_shared<gert::OpImplSpaceRegistryV2>();
//     auto registry_holder = std::make_shared<gert::OpImplSpaceRegistry>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;

//     op_impl_func.check_support = CheckOpSupportedV2StubFail;
//     op_impl_func.op_select_format = OpSelectTbeFormatV2StubFail;
//     op_impl_func.get_op_support_info = OpGetSupportInfoV2StubFail;
//     op_impl_func.get_op_specific_info = OpGetSpecificInfoV2StubFail;
//     registry_holder->types_to_impl_[ge::AscendString(mm_node->GetType().c_str())] = op_impl_func;
//     spaceRegistryStub->AddRegistry(registry_holder);

//     gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceRegistryStub);

//     bool resCheckOpSupported = false;
//     bool resSelectTbeOpFormat = false;
//     LX_QUERY_STATUS resGetOpInfo = LX_QUERY_FAIL;
//     bool resGetOpSpecificInfo = false;
    
//     TbeOpInfo op_info("weightQuantBatchMatmulV2", "", "WeightQuantBatchMatmulV2", "AIcoreEngine");
//     op_info.SetNode(mm_node);

//     CheckSupportedInfo checkSupportedInfo;
//     resCheckOpSupported = CheckOpSupported(op_info, checkSupportedInfo);
//     EXPECT_EQ(resCheckOpSupported, false);

//     std::string opDtypeFormat = "";
//     resSelectTbeOpFormat = SelectTbeOpFormat(op_info, opDtypeFormat);
//     EXPECT_EQ(resSelectTbeOpFormat, false);

//     std::string opSupportInfo = "";
//     resGetOpInfo = GetOpInfo(op_info, opSupportInfo);
//     EXPECT_EQ(resGetOpInfo, LX_QUERY_FAIL);

//     std::string opSpecificInfo = "";
//     resGetOpSpecificInfo = GetOpSpecificInfo(op_info, opSpecificInfo);
//     EXPECT_EQ(resGetOpSpecificInfo, false);
// }

TEST(TeFusionUTest, check_prebuilt_is_custom_op) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::AttrUtils::SetBool(opDescPtr, "_is_custom_op", true);
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{0, 0, 0, teGraphNode, opDescPtr,
                                                          te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    OpBuildTaskResultPtr opResPtr = nullptr;
    opResPtr = std::make_shared<OpBuildTaskResult>();
    opResPtr->result = "success";
    opResPtr->preCompileRetType = PreCompileResultType::Cache;
    te::fusion::PreBuildManager::GetInstance().UpdateOpTaskForPreBuildCache(task, opResPtr);

    CompileResultPtr compileRetPtr = std::make_shared<CompileResult>();
    compileRetPtr->jsonInfo = std::make_shared<nlohmann::json>();
    te::fusion::TeFusionManager::GetInstance()->UpdateOpTaskForCompileCache(task, compileRetPtr);
}