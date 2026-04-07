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
#include "../stub/Python_stub.h"
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "ge_common/ge_api_types.h"

#define private public
#define protected public

#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "tensor_engine/fusion_api.h"
#include "compile/fusion_manager.h"
#include "cache/te_cache_space_manager.h"
#include "common/tbe_op_info_cache.h"
#include "common/common_utils.h"
#include "common/te_config_info.h"
#include "common/te_file_utils.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "te_fusion_base.h"
#include "graph_optimizer/op_compiler/op_format_tune.h"
#include "python_adapter/python_api_call.h"
#include "python_adapter/python_adapter_manager.h"
#include "cache/te_cache_manager.h"
#include "assemble_json/te_json_assemble.h"
#include "binary/binary_manager.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace te;
using namespace te::fusion;
using TbeOpInfoPtr = std::shared_ptr<te::TbeOpInfo>;
class FuzzyBuildUTest : public testing::Test
{
    public:
        FuzzyBuildUTest(){}
    protected:
        virtual void SetUp()
        {

        }
        virtual void TearDown()
        {
        }
    protected:

};

bool GenerateOFileSha256HashValue_stub(te::fusion::PythonApiCall *This,
                                       const char *binData, const size_t binSize, std::string &sha256Val)
{
    sha256Val = "b97559990204d88759446ca413be0189aa572e941ff87908a658d7044ead8854";
    return true;
}

TEST(FuzzyBuildUTest, GeneralizeFuncNotRegister) {
    bool res = false;
    TbeOpInfo op_info("fill", "", "Fill", "AIcoreEngine");
    bool hasRegisteredFunc = false;

    res = CheckIsTbeGeneralizeFuncRegistered(op_info, hasRegisteredFunc);
    EXPECT_EQ(res, true);
    EXPECT_EQ(hasRegisteredFunc, false);
}

TEST(FuzzyBuildUTest, GeneralizeFuncRegister) {
    bool res = false;
    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    bool hasRegisteredFunc = false;

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
    res = CheckIsTbeGeneralizeFuncRegistered(op_info, hasRegisteredFunc);
    EXPECT_EQ(res, true);
    EXPECT_EQ(hasRegisteredFunc, true);
}

TEST(FuzzyBuildUTest, GeneralizeFuncRegisterWithErrorPyAPI) {
    bool res = false;
    TbeOpInfo op_info("crop", "", "Crop", "AIcoreEngine");
    bool hasRegisteredFunc = false;

    te::fusion::HandleManager::Instance().TE_PyObject_CallMethod_SizeT = PyObject_CallMethod_Stub;
    res = CheckIsTbeGeneralizeFuncRegistered(op_info, hasRegisteredFunc);
    EXPECT_EQ(res, false);
}

void CompareGeneralizeShape(pair<vector<int64_t>, vector<int64_t>> &shapeGeneralize, ge::GeTensorDescPtr &tenosrDescPtr)
{
    std::vector<int64_t> shape = tenosrDescPtr->GetShape().GetDims();
    std::vector<int64_t> oriShape = tenosrDescPtr->GetOriginShape().GetDims();

    nlohmann::json temp1 = shape;
    nlohmann::json temp2 = oriShape;
    nlohmann::json temp3 = shapeGeneralize;
    std::cout << "shapeGeneralize is: " << temp3.dump(4) << std::endl;
    std::cout << "shape is: " << temp1.dump(4) << std::endl;
    std::cout << "oriShape is: " << temp2.dump(4) << std::endl;
    bool res = shapeGeneralize.first.size() == shape.size() ? true : false;
    EXPECT_EQ(res, true);
    res = shapeGeneralize.second.size() == oriShape.size() ? true : false;
    EXPECT_EQ(res, true);
    for (size_t i = 0; i < shape.size(); i++) {
        if (shapeGeneralize.first[i] != shape[i]) {
            res = false;
        }
    }
    EXPECT_EQ(res, true);

    for (size_t i = 0; i < oriShape.size(); i++) {
        if (shapeGeneralize.second[i] != oriShape[i]) {
            res = false;
        }
    }
    EXPECT_EQ(res, true);
}

void CompareGeneralizeRange(pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>> &rangeGeneralize,
                            ge::GeTensorDescPtr &tenosrDescPtr)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> oriRange;
    tenosrDescPtr->GetShapeRange(range);
    tenosrDescPtr->GetOriginShapeRange(oriRange);
    nlohmann::json temp1 = oriRange;
    nlohmann::json temp2 = range;
    nlohmann::json temp3 = rangeGeneralize;
    std::cout << "rangeGeneralize is: " << temp3.dump(4) << std::endl;
    std::cout << "range is: " << temp2.dump(4) << std::endl;
    std::cout << "oriRange is: " << temp1.dump(4) << std::endl;
    bool res = rangeGeneralize.first.size() == range.size() ? true : false;
    EXPECT_EQ(res, true);
    res = rangeGeneralize.second.size() == oriRange.size() ? true : false;
    EXPECT_EQ(res, true);
    for (size_t i = 0; i < range.size(); i++) {
        if (rangeGeneralize.first[i].first != range[i].first ||
            rangeGeneralize.first[i].second != range[i].second) {
            res = false;
        }
    }
    EXPECT_EQ(res, true);

    for (size_t i = 0; i < oriRange.size(); i++) {
        if (rangeGeneralize.second[i].first != oriRange[i].first ||
            rangeGeneralize.second[i].second != oriRange[i].second) {
            res = false;
        }
    }
    EXPECT_EQ(res, true);
}

void CompareGeneralizeRes(vector<pair<vector<int64_t>, vector<int64_t>>> &shapeGeneralizeRes,
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> &rangeGeneralizeRes,
    ge::NodePtr &nodePtr)
{
    bool res = false;
    res = shapeGeneralizeRes.size() == rangeGeneralizeRes.size() ? true : false;
    EXPECT_EQ(res, true);
    auto opDescPtr = nodePtr->GetOpDesc();
    size_t index = 0;
    // compare node inputs
    for (auto &tenosrDescPtr : opDescPtr->GetAllInputsDescPtr()) {
        std::cout << "input " << index << std::endl;
        std::cout << "CompareGeneralizeShape " << index << std::endl;
        CompareGeneralizeShape(shapeGeneralizeRes[index], tenosrDescPtr);
        std::cout << "CompareGeneralizeRange " << index << std::endl;
        CompareGeneralizeRange(rangeGeneralizeRes[index], tenosrDescPtr);
        index++;
    }
    // compare node outputs
    for (auto &tenosrDescPtr : opDescPtr->GetAllOutputsDescPtr()) {
        std::cout << "output " << index << std::endl;
        std::cout << "CompareGeneralizeShape " << index << std::endl;
        CompareGeneralizeShape(shapeGeneralizeRes[index], tenosrDescPtr);
        std::cout << "CompareGeneralizeRange " << index << std::endl;
        CompareGeneralizeRange(rangeGeneralizeRes[index], tenosrDescPtr);
        index++;
    }
    // compare inputs node
    for (auto &inputDataAnchor : nodePtr->GetAllInDataAnchors()) {
        std::cout << "output " << index << std::endl;
        if (inputDataAnchor == nullptr) {
            std::cout << "inputDataAnchor is nullptr " << index << std::endl;
            continue;
        }

        auto peerNodeOutAnchor = inputDataAnchor->GetPeerOutAnchor();
        if (peerNodeOutAnchor == nullptr) {
            std::cout << "peerNodeOutAnchor is nullptr " << index << std::endl;
            continue;
        }

        ge::NodePtr peerNode = peerNodeOutAnchor->GetOwnerNode();
        if (peerNode->GetType() != "Data") {
            std::cout << "peerNode[" << peerNode->GetOpDesc()->GetName() << "] is not data, optype is" << peerNode->GetType() << std::endl;
            continue;
        }

        auto peerInputDesc = peerNode->GetOpDesc()->MutableInputDesc(0);
        if (peerInputDesc == nullptr) {
            std::cout << "peerInputDesc is nullptr."<< std::endl;
            continue;
        }

        std::cout << "CompareGeneralizeShape " << index << std::endl;
        CompareGeneralizeShape(shapeGeneralizeRes[index], peerInputDesc);
        std::cout << "CompareGeneralizeRange " << index << std::endl;
        CompareGeneralizeRange(rangeGeneralizeRes[index], peerInputDesc);
        index++;
    }
}

TEST(FuzzyBuildUTest, GeneralizeFuncReturnNone) {
    bool res = false;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("lambNextMVWithDecay","LambNextMVWithDecay");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{10,20},{20,30},{30,40}};
    AddTensorToOpDesc(true, "input7", {9,2,0,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input8", {}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input9", {1}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input10", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input12", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output0", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    ge::NodePtr nodePtr1 = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::NodePtr nodePtr2 = std::make_shared<ge::Node>(opDescPtr, owner_graph);

    TbeOpInfo op_info("lambNextMVWithDecay", "", "LambNextMVWithDecay", "AIcoreEngine");
    AddOpParamToTbeOpInfo({9,2,0,4}, "int64", "NHWC", "input7", range, true, op_info);
    AddOpParamToTbeOpInfo({}, "int64", "NCHW", "input8", range, true, op_info);
    AddOpParamToTbeOpInfo({1}, "int64", "NCHW", "input9", range, true, op_info);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "input10", range, true, op_info);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "input12", range, true, op_info);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "input1", range1, true, op_info);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, REGISTER_FUNC, nodePtr1);
    EXPECT_EQ(res, true);

    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                ori_shape
    {{9,2,0,4},           {-1,-1,0,-1},},               // input7
    {{},                  {},},                         // input8
    {{1},                 {1},},                        // input9
    {{9,2,3,4},           {-1,-1,-1,-1},},              // input10
    {{9,2,3,4},           {9,2,3,4},},                  // input12
    {{9,2,3,4},           {-1,-1,-1,-1},},              // input13
    {{9,2,3,4},           {-1,-1,-1,-1},}               // output0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
                    // range                                              ori_range
    {{},                                               {{0,-1}, {0,-1}, {0,0}, {0,-1}}},                 // input7
    {{},                                               {}},                                              // input8
    {{},                                               {}},                                              // input9
    {{},                                               {{0,-1}, {0,-1}, {0,-1}, {0,-1}}},                // input10
    {{},                                               {}},                                              // input12
    {{{1,10},{10,20},{20,30},{30,40}},                 {{0,-1},{0,-1},{0,-1},{0,-1}}},                // input13
    {{},                                               {{0,-1}, {0,-1}, {0,-1}, {0,-1}}}                 // output0
    };

    CompareGeneralizeRes(shapeGeneralizeRes, rangeGeneralizeRes, nodePtr1);

    (void)ge::AttrUtils::SetBool(opDescPtr, ATTR_NAME_IS_LIMITED_GRAPH, true);
    res = TeGeneralize(op_info, REGISTER_FUNC, nodePtr2);
    EXPECT_EQ(res, true);

    shapeGeneralizeRes = {
        // shape                ori_shape
    {{9,2,0,4},           {-1,-1,0,4},},               // input7
    {{},                  {},},                         // input8
    {{1},                 {1},},                        // input9
    {{9,2,3,4},           {-1,-1,-1,4},},              // input10
    {{9,2,3,4},           {9,2,3,4},},                  // input12
    {{9,2,3,4},           {-1,-1,-1,4},},              // input13
    {{9,2,3,4},           {-1,-1,-1,4},}               // output0
    };
    rangeGeneralizeRes = {
                    // range                                              ori_range
    {{},                                               {{8,15}, {1,3}, {0,0}, {4,4}}},                 // input7
    {{},                                               {}},                                              // input8
    {{},                                               {}},                                              // input9
    {{},                                               {{8,15}, {1,3}, {1,3}, {4,4}}},                // input10
    {{},                                               {}},                                              // input12
    {{{1,10},{10,20},{20,30},{30,40}},                 {{8,15}, {1,3}, {1,3}, {4,4}}},                // input13
    {{},                                               {{8,15}, {1,3}, {1,3}, {4,4}}}                 // output0
    };

    CompareGeneralizeRes(shapeGeneralizeRes, rangeGeneralizeRes, nodePtr2);
}

TEST(FuzzyBuildUTest, GeneralizeFuncReturnSuccess) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    AddTensorToOpDesc(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, REGISTER_FUNC, nodeAdd);
    EXPECT_EQ(res, true);

    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                 ori_shape
    {{1,2,3,4},               {-1,2,-1,-1},},                // add input0
    {{1,2,3,4},               {1,-1,-1,-1},},                // add input1
    {{1,2,3,4},               {-1,-1,-1,-1},},               // add output0
    {{5,6,7,8},               {5,6,7,8},},                   // data0 input0
    {{6,7,8,9},               {6,7,8,9},}                    // data1 input0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
            // range                     ori_range
    {{},                      {{1,-1}, {2,2}, {1,100}, {50,90}}},    // add input0
    {{},                      {{1,1}, {1,-1}, {3,3},{50,90}}},       // add input1
    {{},                      {{1,-1}, {1,-1}, {1,-1}, {1,-1}}},     // add output0
    {{},                      {}},                                   // data0 input0
    {{},                      {}}                                    // data1 input0
    };

    CompareGeneralizeRes(shapeGeneralizeRes, rangeGeneralizeRes, nodeAdd);
}

bool SpreadSupportInfoInputs_stub(te::fusion::TeFusionManager *This, nlohmann::json &supportInfoJson,
    nlohmann::json &inputsJson)
{
    printf("Get in SpreadSupportInfoInputs_stub!!!! \r\n");
    return true;
}

bool FindCurrentInputJson_stub(te::fusion::TeFusionManager *This, nlohmann::json &spreadInputs,
    const int64_t &currentIndex, string &opName, nlohmann::json &currentInput)
{
    printf("Get in FindCurrentInputJson_stub!!!! \r\n");
    return true;
}

bool FindExistedInputAttr_stub(te::fusion::TeFusionManager *This, vector<ge::NamedAttrs> &existedInputAttrs,
    int64_t &existedIndex, int64_t &originalIndex, vector<ge::NamedAttrs> &supportedAttrs,
    vector<ge::NamedAttrs> &existedTensorAttrs)
{
    printf("Get in FindExistedInputAttr_stub!!!! \r\n");
    ge::NamedAttrs build_res;
    (void) build_res.SetAttr("_need_return_result", ge::GeAttrValue::CreateFrom<bool>(false));
    supportedAttrs.push_back(build_res);
    return true;
}

bool FindExistedInputAttr_null_attrs_stub(te::fusion::TeFusionManager *This,
    vector<ge::NamedAttrs> &existedInputAttrs,
    int64_t &existedIndex, int64_t &originalIndex, vector<ge::NamedAttrs> &supportedAttrs,
    vector<ge::NamedAttrs> &existedTensorAttrs)
{
    printf("Get in FindExistedInputAttr_null_attrs_stub!!!! \r\n");
    return true;
}

bool SetFuzzBuildAttrFromJsonForInputs_stub(te::fusion::TeFusionManager *This,
    const nlohmann::json &singleJsonTensor, std::string &keyName,
    ge::NamedAttrs &existedTensorAttr,
    bool isAttrExisted, ge::NamedAttrs &tensorAttr)
{
    printf("Get in SetFuzzBuildAttrFromJsonForInputs_stub!!!! \r\n");
    (void) tensorAttr.SetAttr("_need_return_result", ge::GeAttrValue::CreateFrom<bool>(false));
    return true;
}

TEST(FuzzyBuildUTest, GeneralizeTbeOpInfoReturnSuccess) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,0,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {1}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("const","Const");
    AddTensorToOpDesc(true, "input0", {10,20,30,40}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {10,20,30,40}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {32}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {32}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeConst = owner_graph->AddNode(input1OpDescPtr);
    ge::GraphUtils::AddEdge(nodeData->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeConst->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2,0,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfo({1}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, DEFAULT_TBE_OP_INFO, nodeAdd);
    EXPECT_EQ(res, true);

    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                 ori_shape
    {{1,2,0,4},             {1,-1,0,-1},},                 // add input0
    {{1},                     {1},},                           // add input1
    {{9,2,3,4},               {-1,-1,-1,-1},},                 // add output0
    {{10,20,30,40},        {10,20,30,40},}               // data input0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
             // range                    ori_range
    {{},                      {{1,1}, {0,-1}, {0,0}, {0,-1}}},        // add input0
    {{},                      {}},                                           // add input1
    {{},                      {{0,-1}, {0,-1}, {0,-1}, {0,-1}}},             // add output0
    {{},                      {}}                                            // data input0
    };

    CompareGeneralizeRes(shapeGeneralizeRes, rangeGeneralizeRes, nodeAdd);
}

TEST(FuzzyBuildUTest, GeneralizeTbeOpInfoReturnSuccess_001) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,0,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {1}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("const","Const");
    AddTensorToOpDesc(true, "input0", {10,20,30,40}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {10,20,30,40}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {32}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {32}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeConst = owner_graph->AddNode(input1OpDescPtr);
    ge::GraphUtils::AddEdge(nodeData->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeConst->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2,0,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfo({1}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, DEFAULT_LIMITED_TBE_OP_INFO, nodeAdd);
    EXPECT_EQ(res, true);

    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                 ori_shape
    {{1,2,0,4},               {1,-1,0,4},},                 // add input0
    {{1},                     {1},},                           // add input1
    {{9,2,3,4},               {-1,-1,-1,4},},                 // add output0
    {{10,20,30,40},           {10,20,30,40},}               // data input0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
             // range                    ori_range
    {{},                      {{1,1}, {1,3}, {0,0}, {4,4}}},        // add input0
    {{},                      {}},                                           // add input1
    {{},                      {{8,15}, {1,3}, {1,3}, {4,4}}},             // add output0
    {{},                      {}}                                            // data input0
    };

    CompareGeneralizeRes(shapeGeneralizeRes, rangeGeneralizeRes, nodeAdd);
}

TEST(FuzzyBuildUTest, GeneralizeNodeReturnSuccess) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("variable","Variable");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("const","Const");
    AddTensorToOpDesc(true, "input0", {32}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {32}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {16}, FORMAT_NCHW, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {16}, FORMAT_NCHW, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeVariable = owner_graph->AddNode(input1OpDescPtr);
    ge::GraphUtils::AddEdge(nodeData->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeVariable->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");

    res = TeGeneralize(op_info, DEFAULT_NODE, nodeAdd);
    EXPECT_EQ(res, true);

    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                 ori_shape
    {{1,2,3,4},           {1,-1,-1,-1},},                    // add input0
    {{},                  {},},                              // add input1
    {{9,2,3,4},           {-1,-1,-1,-1},}                    // add output0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
           // range                      ori_range
    {{},                  {{1,1}, {0,-1}, {0,-1}, {0,-1}}},            // add input0
    {{},                  {}},                                         // add input1
    {{},                  {{0,-1}, {0,-1}, {0,-1}, {0,-1}}}            // add output0
    };

    CompareGeneralizeRes(shapeGeneralizeRes, rangeGeneralizeRes, nodeAdd);
}

TEST(FuzzyBuildUTest, GeneralizeWithInvalidGeneralizeType) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("variable","Variable");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("const","Const");
    AddTensorToOpDesc(true, "input0", {32}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {32}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {16}, FORMAT_NCHW, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {16}, FORMAT_NCHW, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeVariable = owner_graph->AddNode(input1OpDescPtr);
    ge::GraphUtils::AddEdge(nodeData->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeVariable->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(1));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");

    res = TeGeneralize(op_info, GENERALIZE_TYPE_MAX, nodeAdd);
    EXPECT_EQ(res, false);
}

TEST(FuzzyBuildUTest, GeneralizeFuncReturnNotList) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;
    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("mul","Mul");
    AddTensorToOpDesc(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    AddTensorToOpDesc(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeMul = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);
    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData1->GetOutDataAnchor(0), nodeMul->GetInDataAnchor(1));

    TbeOpInfo op_info("mul", "", "Mul", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, REGISTER_FUNC, nodeMul);
    EXPECT_EQ(res, false);
}

TEST(FuzzyBuildUTest, TaskFirstCompileSuccess) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    task->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    task->buildType = ACCURATELY_BUILD;
    task->kernel = "te_matmul_cache";
    te::fusion::OpBuildTaskResultPtr opRes = std::make_shared<te::fusion::OpBuildTaskResult>();
    opRes->graphId = graphId;
    opRes->taskId = taskId;
    opRes->compile_info_key = "c775e7b757ede630cd0aa1113bd102661ab38829ca52a6422ab782862f268646";
    opRes->compile_info_str = "1234567890";
    opRes->result = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    std::string targetJsonPath = te::fusion::RealPath(currentFilePath + "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json");
    opRes->jsonFilePath = targetJsonPath;
    task->opRes = opRes;

    std::map<uint64_t, te::fusion::OpBuildTaskPtr> taskIdMap;
    taskIdMap[taskId] = task;
    std::map<uint64_t, std::map<uint64_t, te::fusion::OpBuildTaskPtr>> FinishedTask;
    FinishedTask[graphId] = taskIdMap;

    te::fusion::TeFusionManager::GetInstance()->finishedTask_ = FinishedTask;
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");

    res = WaitAllFinished(graphId, tasks);
    EXPECT_EQ(res, true);

    std::string cacheJsonFilePath = currentFilePath + "/disk_cache/kernel_meta/te_matmul_cache.json";
    std::string cacheOFilePath = currentFilePath + "/disk_cache/kernel_meta/te_matmul_cache.o";
    std::string path = te::fusion::RealPath(cacheJsonFilePath);
    EXPECT_EQ(path.empty(), true);

    path = te::fusion::RealPath(cacheOFilePath);
    EXPECT_EQ(path.empty(), true);

    std::string jsonFilePath;
    res = false;
    (void)ge::AttrUtils::SetStr(opDescPtr, "json_file_path", targetJsonPath);
    (void)ge::AttrUtils::GetStr(opDescPtr, "json_file_path", jsonFilePath);
    
    if (!jsonFilePath.empty()) {
        std::string path1 = te::fusion::RealPath(jsonFilePath);
        std::string path2 = te::fusion::RealPath(currentFilePath + "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json");
        if (path1 == path2) {
            res = true;
        }
    }
    EXPECT_EQ(res, true);

    std::string compile_info_str;
    (void)ge::AttrUtils::GetStr(opDescPtr, "compile_info_json", compile_info_str);
    EXPECT_EQ(compile_info_str, "1234567890");

    std::string compile_info_key;
    (void)ge::AttrUtils::GetStr(opDescPtr, "compile_info_key", compile_info_key);
    EXPECT_EQ(compile_info_key, "c775e7b757ede630cd0aa1113bd102661ab38829ca52a6422ab782862f268646");

    te::fusion::TeFileUtils::DeleteFile(cacheJsonFilePath);
    te::fusion::TeFileUtils::DeleteFile(cacheOFilePath);
}

TEST(FuzzyBuildUTest, CacheCompileSuccess) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());
    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    task->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    task->buildType = ACCURATELY_BUILD;
    task->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    te::fusion::OpBuildTaskResultPtr opRes = std::make_shared<te::fusion::OpBuildTaskResult>();
    opRes->graphId = graphId;
    opRes->taskId = taskId;
    opRes->result = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    opRes->preCompileRetPtr = std::make_shared<PreCompileResult>("");
    opRes->preCompileRetPtr->opPattern = "MatMul";
    opRes->compileRetPtr = nullptr;
    opRes->compile_info_key = "c775e7b757ede630cd0aa1113bd102661ab38829ca52a6422ab782862f268646";
    opRes->compile_info_str = "1234567890";
    task->opRes = opRes;

    std::map<uint64_t, te::fusion::OpBuildTaskPtr> taskIdMap;
    taskIdMap[taskId] = task;

    std::map<uint64_t, std::map<uint64_t, te::fusion::OpBuildTaskPtr>> FinishedTask;
    FinishedTask[graphId] = taskIdMap;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeFusionManager::GetInstance()->finishedTask_ = FinishedTask;
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    // make sure json file has compileInfo
    std::string cacheJson = TeCacheManager::Instance().cache_dir_path_ + "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    opRes->jsonFilePath = cacheJson;
    // nlohmann::json jsonInfo;
    // (void)GetJsonFromJsonFile(cacheJson, jsonInfo);
    // jsonInfo["compileInfo"] = "1234567890";
    // (void)WriteToJsonFile(cacheJson, jsonInfo);
    res = WaitAllFinished(graphId, tasks);
    EXPECT_EQ(res, true);

    std::string jsonFilePath;
    res = false;
    (void)ge::AttrUtils::GetStr(opDescPtr, "json_file_path", jsonFilePath);
    if (!jsonFilePath.empty()) {
        std::string path1 = te::fusion::RealPath(jsonFilePath);
        std::string path2 = te::fusion::RealPath(currentFilePath + "/disk_cache/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json");
        if (path1 == path2) {
            res = true;
        }
    }
    EXPECT_EQ(res, true);

    std::string compile_info_str;
    (void)ge::AttrUtils::GetStr(opDescPtr, "compile_info_json", compile_info_str);
    EXPECT_EQ(compile_info_str, "1234567890");

    std::string compile_info_key;
    (void)ge::AttrUtils::GetStr(opDescPtr, "compile_info_key", compile_info_key);
    EXPECT_EQ(compile_info_key, "c775e7b757ede630cd0aa1113bd102661ab38829ca52a6422ab782862f268646");
}

static void TeFusionCreateSingleNodeGraph(ComputeGraphPtr graph, std::vector<ge::Node *> &teGraphNode) {
    OpDescPtr relu_op = std::make_shared<OpDesc>("matmul", "Activation");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");

    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc1);
    data->AddOutputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_HWCN);
    out_desc1.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc1);

    vector<vector<int64_t>> sliceShapeAll; // for SetSgtTensorSliceShape
    vector<int64_t> sliceShape = {1};
    sliceShapeAll.push_back(sliceShape);
    (void)ge::AttrUtils::SetListListInt(relu_op->MutableInputDesc(0), "_sgt_slice_shape", sliceShapeAll);
    (void)ge::AttrUtils::SetListListInt(relu_op->MutableInputDesc(0), "_sgt_ori_slice_shape", sliceShapeAll);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_NCHW);
    in_desc3.SetDataType(DT_FLOAT16);
    data->AddInputDesc("z", in_desc3);

    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data_node = graph->AddNode(data);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

    ge::AttrUtils::SetBool(relu_node->GetOpDesc(), "_is_custom_op", false);

    teGraphNode.push_back(relu_node.get());
}

static bool FindOpKernelMap(OpBuildTaskPtr &opTask)
{
    auto opIter = te::fusion::TeFusionManager::GetInstance()->opKernelMap_.find(opTask->kernel);
    if (opIter == te::fusion::TeFusionManager::GetInstance()->opKernelMap_.end()) {
        printf("Task not in OpKernelMap. \r\n");
        return false;
    }

    auto &match = opIter->second;
    std::vector<OpBuildTaskPtr> &vecOpBuildTask = match.second;
    if (std::find(vecOpBuildTask.begin(), vecOpBuildTask.end(), opTask) == vecOpBuildTask.end()) {
        printf("Task not in OpKernelMap vecOpBuildTask. \r\n");
        return false;
    }
    return true;
}

static bool FindOpKernelBuildingMap(OpBuildTaskPtr &opTask, bool isBuilding)
{
    return true;
}

TEST(FuzzyBuildUTest, CacheFuzzyFlockingTaskCompileUt) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");

    CreateDir(te::fusion::TeConfigInfo::Instance().GetDebugDir() + "/kernel_meta_temp_test_id");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // make sure json file has compileInfo
    std::string cacheJson = TeCacheManager::Instance().cache_dir_path_ +
        "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    nlohmann::json jsonInfo;
    (void)GetJsonFromJsonFile(cacheJson, jsonInfo);
    jsonInfo["compileInfo"] = "1234567890";
    (void)WriteToJsonFile(cacheJson, jsonInfo);

    res = WaitAllFinished(graphId, tasks); // call flocking task, cache not find, need compile, save build result
    EXPECT_EQ(res, true);

    GlobalMockObject::verify();

    // check dispatched_task not deleted
    EXPECT_EQ(te::fusion::TeFusionManager::GetInstance()->dispatchedTask_.size(), 1);

    // create cache kernelName is te_Activation_success  kernelName is te_activation_success___fuzzy
    // start call build interface(BuildOpAsync) and save task to opKernelBuildingMap_ and OpKernelMap
    // check task in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelBuildingMap(task, true);
    EXPECT_EQ(res, true);
    res = FindOpKernelMap(task);
    EXPECT_EQ(res, false);

    // check build lock, lock file exist
    std::string lockFile = te::fusion::TeConfigInfo::Instance().GetDebugDir() + "/kernel_meta_temp_test_id/.lock";
    std::string lockFilePath = te::fusion::RealPath(lockFile);
    EXPECT_EQ(lockFilePath.empty(), true);

    // manually release lock
    te::fusion::TeFileUtils::DeleteFile(lockFilePath);
}

TEST(FuzzyBuildUTest, CacheFuzzyCompileSuccessUt) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("matmul","Matmul");
    ge::NodePtr nodeMatmul = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(nodeMatmul.get());

    std::string opName = nodeMatmul->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input1", range1, true, opInfo);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_SUCC});
    task->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";

    task->buildType = FUZZILY_BUILD;
    task->newCompile = true;
    task->kernel = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    te::fusion::OpBuildTaskResultPtr opRes = std::make_shared<te::fusion::OpBuildTaskResult>();
    opRes->graphId = graphId;
    opRes->taskId = taskId;
    opRes->result = "te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762";
    task->opRes = opRes;

    std::map<uint64_t, te::fusion::OpBuildTaskPtr> taskIdMap;
    taskIdMap[taskId] = task;

    std::map<uint64_t, std::map<uint64_t, te::fusion::OpBuildTaskPtr>> FinishedTask;
    FinishedTask[graphId] = taskIdMap;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeFusionManager::GetInstance()->finishedTask_ = FinishedTask;
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    std::string jsonFile = currentFilePath + "/kernel_meta/" + task->kernel + ".json";
    (void)GetJsonFromJsonFile(jsonFile, task->supportInfoJson);

    // make sure json file has compileInfo
    std::string cacheJson = TeCacheManager::Instance().cache_dir_path_ +
        "/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    nlohmann::json jsonInfo;
    (void)GetJsonFromJsonFile(cacheJson, jsonInfo);
    jsonInfo["compileInfo"] = "1234567890";
    (void)WriteToJsonFile(cacheJson, jsonInfo);

    opRes->jsonFilePath = cacheJson;

    res = WaitAllFinished(graphId, tasks);
    EXPECT_EQ(res, true);

    GlobalMockObject::verify();

    std::string jsonFilePath;
    res = false;
    (void)ge::AttrUtils::GetStr(opDescPtr, "json_file_path", jsonFilePath);
    if (!jsonFilePath.empty()) {
        std::string path1 = te::fusion::RealPath(jsonFilePath);
        std::string path2 = te::fusion::RealPath(currentFilePath + "/disk_cache/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json");
        if (path1 == path2) {
            res = true;
        }

        EXPECT_EQ(res, true);
        printf("path1=%s, path2=%s", path1.c_str(), path2.c_str());
    } else {
        printf("jsonFilePath is empty!");
    }
    // not really compile, copy json file from kernelMate to cache before build, the json file in kernelMeta does not
    // have compile info str and key, FuzzyIncrUpdateCache will del old cache file and copy from kernelMeta because cache
    // has files, so finally cache file has no compile info str and key
}

TEST(FuzzyBuildUTest, FuzzyBuildFindCacheUt) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string jsonPath = currentFilePath +
        "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    std::string dstPath = TeCacheManager::Instance().cache_dir_path_ + "/te_Activation_success.json";
    te::fusion::TeFileUtils::CopyFileToNewPath(jsonPath, dstPath);
    std::string dstRealPath = te::fusion::RealPath(dstPath);
    EXPECT_EQ(dstRealPath.empty(), false);

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);
    // 1127 temp for fuzzy call operater accurate build, EXPECT_EQ(task->new_cache_dir_flag, true);
    // check task SUCC
    // 1127 temp for fuzzy call operater accurate build, EXPECT_EQ(task->status, OP_TASK_STATUS::OP_TASK_SUCC);
    // create cache kernelName is te_Activation_success  kernelName is te_activation_success___fuzzy
    // check task not in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelBuildingMap(task, false);
    EXPECT_EQ(res, true);
    res = FindOpKernelMap(task);
    // 1127 temp for fuzzy call operater accurate build, EXPECT_EQ(res, false);

    GlobalMockObject::verify();

    // manually del cache json file
    te::fusion::TeFileUtils::DeleteFile(dstPath);
}

TEST(FuzzyBuildUTest, FuzzyBuildCompileLockFailUt) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    // fcntl can not manually create build lock file, so this case will lock and run
    std::string lockFile = te::fusion::TeConfigInfo::Instance().GetDebugDir() + "/kernel_meta/kernel_meta_temp/te_Activation_success.lock";

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);

    // check task pending
    EXPECT_EQ(task->status, OP_TASK_STATUS::OP_TASK_FAIL);
    // create cache kernelName is te_Activation_success  kernelName is te_activation_success___fuzzy
    // start call build interface(BuildOpAsync) and save task to opKernelBuildingMap_ and OpKernelMap
    // check task in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelBuildingMap(task, false);
    EXPECT_EQ(res, true);
    res = FindOpKernelMap(task);
    EXPECT_EQ(res, false);
    // manually release lock
    te::fusion::TeFileUtils::DeleteFile(lockFile);
}

TEST(FuzzyBuildUTest, FuzzyBuildFindNoCacheCompileUt) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strategy;
    task->sgt_slice_shape_index = 0; // for SetSgtTensorSliceShape
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);

    // create cache kernelName is te_Activation_success  kernelName is te_activation_success___fuzzy
    // start call build interface(BuildOpAsync) and save task to opKernelBuildingMap_ and OpKernelMap
    // check task in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelBuildingMap(task, true);
    EXPECT_EQ(res, true);
    res = FindOpKernelMap(task);
    EXPECT_EQ(res, false);

    // check build lock, lock file exist
    std::string lockFile = te::fusion::TeConfigInfo::Instance().GetDebugDir() + "/kernel_meta/kernel_meta_temp_test_id/.lock";
    std::string lockFilePath = te::fusion::RealPath(lockFile);
    EXPECT_EQ(lockFilePath.empty(), true);
    FILE *fp = te::fusion::TeFusionManager::GetInstance()->FindLockFpHandle(task->kernel);
    if (fp == nullptr) {
        printf("Lock file handle is null.");
        return;
    }
    (void)TeFileUtils::IsFileFcntlLock(fileno(fp));
    // manually release lock
    te::fusion::TeFileUtils::DeleteFile(lockFilePath);
}

TEST(FuzzyBuildUTest, FuncTeGeneralizeTest) {
    bool res = false;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("lambNextMVWithDecay","LambNextMVWithDecay");
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    TbeOpInfo op_info("lambNextMVWithDecay", "", "LambNextMVWithDecay", "AIcoreEngine");

    res = te::fusion::BinaryManager::Instance().TeGeneralize(op_info, GENERALIZE_TYPE_MAX, nodePtr);
    EXPECT_EQ(res, false);

    MOCKER_CPP(&te::fusion::BinaryManager::TeGeneralizeWithRegisterFunc).stubs().will(returnObjectList(false, true));
    res = te::fusion::BinaryManager::Instance().TeGeneralize(op_info, REGISTER_FUNC, nodePtr);

    GlobalMockObject::verify();

    EXPECT_EQ(res, false);

    res = te::fusion::BinaryManager::Instance().TeGeneralize(op_info, REGISTER_FUNC, nodePtr);
    EXPECT_EQ(res, true);
}

TEST(FuzzyBuildUTest, FuzzyBuildFindNoCacheOriShapeRangeCompileUt) {
    TbeOpInfoCache::Instance().secondTbeOpInfoMap_.clear();
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("fill", "", "Fill", "AIcoreEngine"); // unlimited range
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10}, {1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, false, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);

    std::string lockFile = te::fusion::TeConfigInfo::Instance().GetDebugDir() + "/kernel_meta/kernel_meta_temp_test_id/.lock";
    std::string lockFilePath = te::fusion::RealPath(lockFile);
    EXPECT_EQ(lockFilePath.empty(), true);
    // manually release lock
    te::fusion::TeFileUtils::DeleteFile(lockFilePath);
}

// 1127 temp for fuzzy call operater accurate build, writec compile info to json
void WriteCompileInfoTojson(std::string dstRealPath)
{
    json compileInfo;
    compileInfo["compileInfo"] = "tempcompileinfo";
    std::ofstream ofstream(dstRealPath, std::ios::out);
    try {
        if (!ofstream.is_open()) {
            std::cout << "Open file failed, file is already opened. /r/n" << std::endl;
            return;
        }

        ofstream << compileInfo.dump(4);
        ofstream.close();
    } catch (const std::exception &e) {
        std::cout << "Failed to write file. /r/n" << std::endl;
        ofstream.close();
        return;
    }
}

TEST(FuzzyBuildUTest, DynamicShapeAccurateBuildReuseFuzzyCacheMatchUt) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    (void)ge::AttrUtils::SetBool(teGraphNode[0]->GetOpDesc(), "support_dynamicshape", true);

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = ACCURATELY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string jsonPath = currentFilePath +
        "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    std::string dstPath = TeCacheManager::Instance().cache_dir_path_ + "/te_Activation_success.json";
    te::fusion::TeFileUtils::CopyFileToNewPath(jsonPath, dstPath);
    std::string dstRealPath = te::fusion::RealPath(dstPath);
    EXPECT_EQ(dstRealPath.empty(), false);
    WriteCompileInfoTojson(dstRealPath);
    std::string strategy;
    task->kernel = "te_Activation_success";
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    EXPECT_EQ(res, OP_BUILD_SUCC);
    res = FindOpKernelBuildingMap(task, false);
    EXPECT_EQ(res, true);
    res = FindOpKernelMap(task);
    // 1127 temp for fuzzy call operater accurate build, EXPECT_EQ(res, false);

    GlobalMockObject::verify();

    // manually del cache json file
    te::fusion::TeFileUtils::DeleteFile(dstPath);
}

TEST(FuzzyBuildUTest, DynamicShapeAccurateBuildReuseFuzzyCacheMatchNotHighPerformaceUt) {
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    MOCKER_CPP(&te::fusion::PythonApiCall::GetBinFileSha256Value,
        bool(te::fusion::PythonApiCall::*)(const char *binData, const size_t, std::string &res) const)
        .stubs()
        .will(invoke(GenerateOFileSha256HashValue_stub));
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    (void)ge::AttrUtils::SetBool(teGraphNode[0]->GetOpDesc(), "support_dynamicshape", true);

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = ACCURATELY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string jsonPath = currentFilePath +
        "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    std::string dstPath = TeCacheManager::Instance().cache_dir_path_ + "/te_Activation_success.json";
    te::fusion::TeFileUtils::CopyFileToNewPath(jsonPath, dstPath);
    std::string dstRealPath = te::fusion::RealPath(dstPath);
    EXPECT_EQ(dstRealPath.empty(), false);

    std::string strategy;
    task->kernel = "te_Activation_success";
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("End of BuildTbeOp!!!!!\r\n", opName.c_str());
    // no resuse cache, continue to accurate build
    EXPECT_EQ(res, OP_BUILD_SUCC);

    // check task SUCC
//    EXPECT_EQ(task->status, OP_TASK_STATUS::OP_TASK_SUCC);
    // check task in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelMap(task);
//    EXPECT_EQ(res, false);

    GlobalMockObject::verify();

    // manually del cache json file
    te::fusion::TeFileUtils::DeleteFile(dstPath);
}

TEST(FuzzyBuildUTest, DynamicShapeAccurateBuildReuseFuzzyCacheNoMatchUt) {
    TeCacheSpaceManager fusion_serial = TeCacheSpaceManager::Instance();
    MOCKER_CPP(&te::fusion::PythonApiCall::GetBinFileSha256Value,
               bool(te::fusion::PythonApiCall::*)(const char *binData, const size_t, std::string &res) const)
        .stubs()
        .will(invoke(GenerateOFileSha256HashValue_stub));
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    (void)ge::AttrUtils::SetBool(teGraphNode[0]->GetOpDesc(), "support_dynamicshape", true);

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = ACCURATELY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);

    TbeOpInfo opInfo1("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo1);
    (void)opInfo1.SetOptions(options);
    opInfo1.SetNode(nodePtr);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string jsonPath = currentFilePath +
        "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    std::string dstPath = TeCacheManager::Instance().cache_dir_path_ + "/te_Activation_success.json";
    te::fusion::TeFileUtils::CopyFileToNewPath(jsonPath, dstPath);
    std::string dstRealPath = te::fusion::RealPath(dstPath);
    EXPECT_EQ(dstRealPath.empty(), false);

    std::string strategy;
    task->kernel = "te_Activation_success";
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("End of BuildTbeOp!!!!!\r\n", opName.c_str());
    // no resuse cache, continue to accurate build
    EXPECT_EQ(res, OP_BUILD_SUCC);
    // check task SUCC
//    EXPECT_EQ(task->status, OP_TASK_STATUS::OP_TASK_SUCC);
    // check task in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelMap(task);
//    EXPECT_EQ(res, false);

    GlobalMockObject::verify();

    // manually del cache json file
    te::fusion::TeFileUtils::DeleteFile(dstPath);
}

TEST(FuzzyBuildUTest, DynamicShapeAccurateBuildReuseFuzzyCacheConditionDynamicFailUt) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = ACCURATELY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    opInfo.SetNode(nodePtr);

    TbeOpInfo opInfo1("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo1);
    (void)opInfo1.SetOptions(options);
    opInfo1.SetNode(nodePtr);

    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strategy;
    task->kernel = "te_Activation_success";
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("End of BuildTbeOp!!!!!\r\n", opName.c_str());
    // no resuse cache, continue to accurate build
    EXPECT_EQ(res, OP_BUILD_SUCC);
}

TEST(FuzzyBuildUTest, DynamicShapeAccurateBuildReuseFuzzyCacheConditionCacheFailUt) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = ACCURATELY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Disable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    opInfo.SetNode(nodePtr);
    TbeOpInfo opInfo1("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo1);
    (void)opInfo1.SetOptions(options);
    opInfo.SetNode(nodePtr);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string strategy;
    task->kernel = "te_Activation_success";
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    printf("End of BuildTbeOp!!!!!\r\n", opName.c_str());
    // no resuse cache, continue to accurate build
    EXPECT_EQ(res, OP_BUILD_SUCC);
}

TEST(FuzzyBuildUTest, AutoFusionBuildUt) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);
    (void)ge::AttrUtils::SetBool(teGraphNode[0]->GetOpDesc(), ONLY_FUSION_CHECK, true);

    std::string opName = teGraphNode[0]->GetName();
    printf("op_name is %s \r\n", opName.c_str());

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto task = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDescPtr, te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    task->buildType = FUZZILY_BUILD;

    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    te::fusion::TeConfigInfo::Instance().debugDirs_.push(currentFilePath + "/kernel_meta");
    te::fusion::TeConfigInfo::Instance().kernelMetaParentDir_ = currentFilePath;
    TeCacheManager::Instance().cache_mode_ = CompileCacheMode::Enable;
    TeCacheManager::Instance().cache_dir_path_ = currentFilePath + "/disk_cache/kernel_meta";
    TeConfigInfo::Instance().config_str_item_vec_[static_cast<size_t>(TeConfigInfo::ConfigStrItem::SocVersion)] = "../disk_cache/kernel_meta";

    map<std::string, string> options = {{ge::OP_COMPILER_CACHE_DIR, currentFilePath},
        {ge::OP_COMPILER_CACHE_MODE, COMPILE_CACHE_MODE_ENABLE}};
    TbeOpInfo opInfo("add", "", "Add", "AIcoreEngine");
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    AddOpParamToTbeOpInfo({1,2}, "int64", opName, "input0", range, true, opInfo);
    (void)opInfo.SetOptions(options);
    std::shared_ptr<TbeOpInfo> pTbeOp(new (std::nothrow) TbeOpInfo(opInfo));
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    pTbeOp->SetNode(nodePtr);
    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);

    std::string jsonPath = currentFilePath +
        "/kernel_meta/te_matmul_2ac1b9067af400eb161d75ccd6c6ca71bed7b97750d79c4476fb65ac9bedd762.json";
    std::string dstPath = TeCacheManager::Instance().cache_dir_path_ + "/te_Activation_success.json";
    te::fusion::TeFileUtils::CopyFileToNewPath(jsonPath, dstPath);
    std::string dstRealPath = te::fusion::RealPath(dstPath);
    EXPECT_EQ(dstRealPath.empty(), false);

    std::string strategy;
    res = te::fusion::TeFusionManager::GetInstance()->BuildTbeOp(task, strategy);
    // EXPECT_EQ(res, OP_BUILD_SUCC);
    // check task not in opKernelBuildingMap_ and OpKernelMap
    res = FindOpKernelBuildingMap(task, false);
    // EXPECT_EQ(res, false);

    GlobalMockObject::verify();
    // manually del cache json file
    te::fusion::TeFileUtils::DeleteFile(dstPath);
}

TEST(FuzzyBuildUTest, SetShapeAndValueToTensor_001) {
    nlohmann::json singleGeneralizeRes;
    std::vector<std::pair<int64_t, int64_t>> valueRange = {{1, -1}, {10 ,100}};
    std::vector<int64_t> value;
    singleGeneralizeRes["const_value_range"] = valueRange;
    singleGeneralizeRes["const_value"] = value;
    ge::GeTensorDescPtr tenosrDescPtr = std::make_shared<ge::GeTensorDesc>();
    ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>();
    ge::AttrUtils::SetTensor(tenosrDescPtr, ge::ATTR_NAME_VALUE, weight);

    te::fusion::BinaryManager::Instance().SetShapeAndValueToTensor(singleGeneralizeRes, tenosrDescPtr);
}

TEST(FuzzyBuildUTest, fusion_serial_GetFilesAccessTime) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    currentFilePath = te::fusion::RealPath(currentFilePath);
    printf("currentFilePath=%s.\n", currentFilePath.c_str());

    std::string jsonPathMixJson = currentFilePath +
        "/test_files/kernel_meta/matmul_add_1.json";
    std::string jsonPath = currentFilePath +
        "/test_files/kernel_meta/te_matmul_1.json";

    if (te::fusion::RealPath(jsonPath).empty()) {
        printf("jsonPath=%s is not exist!\n", jsonPath.c_str());
    }
    if (te::fusion::RealPath(jsonPathMixJson).empty()) {
        printf("jsonPathMixJson=%s is not exist!\n", jsonPathMixJson.c_str());
    }

    bool res  = TeFileUtils::IsObjFileExsit("MatMul", jsonPathMixJson);
    EXPECT_EQ(res, true);

    vector<std::string> cacheJsonFilePaths;
    cacheJsonFilePaths.push_back(jsonPath);
    cacheJsonFilePaths.push_back(jsonPathMixJson);
    std::multimap<uint64_t, CacheFileSizeInfo> filesStatInfo;

    te::fusion::TeCacheSpaceManager::Instance().GetFilesAccessTime(cacheJsonFilePaths, filesStatInfo);
}

TEST(FuzzyBuildUTest, fusion_serial_ParseOpBuildResult) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    currentFilePath = te::fusion::RealPath(currentFilePath);
    printf("currentFilePath=%s.\n", currentFilePath.c_str());

    std::string jsonPath = currentFilePath +
        "/test_files/kernel_meta/te_matmul_1.json";

    if (te::fusion::RealPath(jsonPath).empty()) {
        printf("jsonPath=%s is not exist!\n", jsonPath.c_str());
    }

    std::string compileInfo;
    std::string compileInfoKey;
    te::fusion::PythonAdapterManager::GetCompileInfo(jsonPath, compileInfo, compileInfoKey);
}

TEST(FuzzyBuildUTest, fusion_serial_ParseResult) {
    std::string currentFilePath = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/op_compile_adapter";
    currentFilePath = te::fusion::RealPath(currentFilePath);
    printf("currentFilePath=%s.\n", currentFilePath.c_str());

    std::string jsonPath = currentFilePath +
        "/test_files/kernel_meta/te_matmul_1.json";

    if (te::fusion::RealPath(jsonPath).empty()) {
        printf("jsonPath=%s is not exist!\n", jsonPath.c_str());
    }

    std::string opResCompile = "{\
        \"pattern\":\"Opaque\",\
        \"core_type\":\"AiCore\"\
        }";

    std::string jsonFilePath;
    std::string pattern;
    std::string coreType;
    std::string compileInfo;
    std::string compileInfoKey;
    int fusionCheckRes = -2;
    te::fusion::PythonAdapterManager::ParseResult(nullptr, 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);

    opResCompile = "{\
        \"core_type\":\"AiCore\"\
        }";
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 0, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);

    opResCompile = "{\
        \"json_file_path\":\"xxx\"\
        }";
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);

    opResCompile = "{\
        \"json_file_path\":\"xxx\", \"xxx\"\
        }";
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);

    currentFilePath = "./llt/atc/opcompiler/te_fusion/st";
    if (te::fusion::RealPath(currentFilePath).empty()) {
        currentFilePath = "../../../../../../../../../llt/atc/opcompiler/te_fusion/st";
    }
    currentFilePath = te::fusion::RealPath(currentFilePath);
    printf("currentFilePath=%s.\n", currentFilePath.c_str());

    jsonPath = currentFilePath +
        "/test_files/kernel_meta/te_matmul_1.json";

    opResCompile = "{\
        \"json_file_path\":\"" + jsonPath + "\"\
        }";
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);

    opResCompile = "{\
        \"json_file_path\":\"xxx\", \"compile_info\":{\"a\":1}\
        }";
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);

    opResCompile = "{\
    \"json_file_path\":\"xxx\", \"fusion_check_result\":0, \"compile_info\":{\"a\":1}\
    }";
    te::fusion::PythonAdapterManager::ParseResult(opResCompile.c_str(), 1, jsonFilePath, pattern, coreType, compileInfo, compileInfoKey, fusionCheckRes);
}

TEST(FuzzyBuildUTest, OutputsDescToJsonProcess_001) {
    ge::InDataAnchorPtr anchor = nullptr;
    std::unordered_set<ge::Node *> allNodes;
    std::string suffix;
    size_t ref_count = 0;
    TeJsonAssemble::GetOutputSuffix(anchor, allNodes, suffix, ref_count);
}

TEST(FuzzyBuildUTest, OutputsDescToJsonProcess_002) {
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("lambNextMVWithDecay","LambNextMVWithDecay");
    std::vector<std::pair<int64_t, int64_t>> range;
    std::vector<std::pair<int64_t, int64_t>> range1 = {{1,10},{10,20},{20,30},{30,40}};
    AddTensorToOpDesc(true, "input7", {9,2,0,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input8", {}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input9", {1}, FORMAT_NCHW, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input10", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input12", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input13", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range1, opDescPtr);
    AddTensorToOpDesc(false, "output0", {9,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    ge::NodePtr nodePtr = std::make_shared<ge::Node>(opDescPtr, owner_graph);
    ge::InDataAnchorPtr anchor = nodePtr->GetInDataAnchor(1);
    std::unordered_set<ge::Node *> allNodes;
    std::string suffix;
    size_t ref_count = 0;
    TeJsonAssemble::GetOutputSuffix(anchor, allNodes, suffix, ref_count);
}

TEST(FuzzyBuildUTest, OutputsDescToJsonProcess_003) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");
    AddTensorToOpDesc(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDesc(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = owner_graph->AddNode(input2OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeData1->GetInDataAnchor(0));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, REGISTER_FUNC, nodeAdd);
    EXPECT_EQ(res, true);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput(TT_DYN, tensors);
    std::vector<TbeOpParam> outputs;
    outputs.emplace_back(opinput);
    std::string key_name = "key_name";
    uint32_t idx = 0;
    json json_str;
    std::unordered_set<ge::Node *> allNodes;
    TbeOpInfo opInfo("test1", "", "Test", "AIcoreEngine");
    opInfo.SetOutputs(outputs);
    ConstTbeOpInfoPtr tbeOpInfoPtr = std::make_shared<const TbeOpInfo>(opInfo);
    size_t nodeListIdx = 0;
    std::vector<std::string> verifyOpTypeList = {"Add"};
    std::vector<ge::Node *> nodes;
    nodes.emplace_back(nodeAdd.get());
    InOutToJsonParam outputJsonPara(key_name, nodeListIdx, verifyOpTypeList, tbeOpInfoPtr, nodeData0.get());
    TeJsonAssemble::OutputsDescToJsonProcess(json_str, allNodes, outputJsonPara, idx, nodeData0->GetOutDataAnchor(0), 1, nodes);

    TbeOpTensor tensorop1("test1", shape, "bool", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors1;
    tensors1.push_back(tensorop1);
    TbeOpParam opinput1(TT_DYN, tensors1);
    std::vector<TbeOpParam> outputs1;
    outputs1.emplace_back(opinput1);
    json json_str1;
    std::unordered_set<ge::Node *> allNodes1;
    TbeOpInfo opInfo2("test1", "", "Test", "AIcoreEngine");
    opInfo2.SetOutputs(outputs1);
    ConstTbeOpInfoPtr tbeOpInfoPtr2 = std::make_shared<const TbeOpInfo>(opInfo2);
    TeJsonAssemble::OutputsDescToJsonProcess(json_str, allNodes1, outputJsonPara, idx, nodeData0->GetOutDataAnchor(0), 1, nodes);
}

TEST(FuzzyBuildUTest, OutputsDescToJsonProcess_004) {
    bool res = false;
    std::vector<std::pair<int64_t, int64_t>> range;

    // create node Add
    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>("add","Add");
    AddTensorToOpDesc(true, "input0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(true, "input1", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);
    AddTensorToOpDesc(false, "output0", {1,2,3,4}, FORMAT_NHWC, DT_INT64, range, opDescPtr);

    // create input nodes data0 and data1
    ge::OpDescPtr input0OpDescPtr = std::make_shared<ge::OpDesc>("data0","Data");
    ge::OpDescPtr input1OpDescPtr = std::make_shared<ge::OpDesc>("data1","Data");
    ge::OpDescPtr input2OpDescPtr = std::make_shared<ge::OpDesc>("data2","Data");
    AddTensorToOpDesc(true, "input0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(false, "output0", {5,6,7,8}, FORMAT_NHWC, DT_INT64, range, input0OpDescPtr);
    AddTensorToOpDesc(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input1OpDescPtr);
    AddTensorToOpDesc(true, "input0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);
    AddTensorToOpDesc(false, "output0", {6,7,8,9}, FORMAT_NHWC, DT_INT64, range, input2OpDescPtr);

    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");
    ge::NodePtr nodeAdd = owner_graph->AddNode(opDescPtr);
    ge::NodePtr nodeData0 = owner_graph->AddNode(input0OpDescPtr);
    ge::NodePtr nodeData1 = owner_graph->AddNode(input1OpDescPtr);
    ge::NodePtr nodeData2 = owner_graph->AddNode(input2OpDescPtr);

    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeAdd->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(nodeData0->GetOutDataAnchor(0), nodeData1->GetInDataAnchor(0));

    TbeOpInfo op_info("add", "", "Add", "AIcoreEngine");
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input0", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "input1", range, true, op_info);
    AddOpParamToTbeOpInfo({1,2,3,4}, "int64", "NHWC", "output0", range, false, op_info);

    res = TeGeneralize(op_info, REGISTER_FUNC, nodeAdd);
    EXPECT_EQ(res, true);

    std::vector<int64_t> shape;
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    shape.push_back(4);
    TbeOpTensor tensorop("test1", shape, "float16", "ND", ATTR_SHAPE_LIST);
    std::vector<TbeOpTensor> tensors;
    tensors.push_back(tensorop);
    TbeOpParam opinput(TT_DYN, tensors);
    std::vector<TbeOpParam> outputs;
    outputs.emplace_back(opinput);
    std::string key_name = "key_name";
    uint32_t idx = 0;
    json json_str;
    std::unordered_set<ge::Node *> allNodes;
    TbeOpInfo opInfo("test1", "", "Test", "AIcoreEngine");
    opInfo.SetOutputs(outputs);
    ConstTbeOpInfoPtr tbeOpInfoPtr = std::make_shared<const TbeOpInfo>(opInfo);
    size_t nodeListIdx = 0;
    std::vector<std::string> verifyOpTypeList = {"Add0"};
    std::vector<ge::Node *> nodes;
    nodes.emplace_back(nodeAdd.get());
    allNodes.insert(nodeAdd.get());
    allNodes.insert(nodeData0.get());
    allNodes.insert(nodeData1.get());
    allNodes.insert(nodeData2.get());
    InOutToJsonParam outputJsonPara(key_name, nodeListIdx, verifyOpTypeList, tbeOpInfoPtr, nodeData0.get());
    TeJsonAssemble::OutputsDescToJsonProcess(json_str, allNodes, outputJsonPara, idx, nodeData0->GetOutDataAnchor(0), 1, nodes);
}


TEST(FuzzyBuildUTest, FuzzyBuildSecondaryGeneralization_001) {
    bool res = false;
    uint64_t graphId = 100;
    uint64_t taskId = static_cast<uint64_t>(rand());
    uint64_t sgtThreadIndex = 300;
    vector<FinComTask> tasks;
    ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("te");

    std::vector<ge::Node *> teGraphNode;
    TeFusionCreateSingleNodeGraph(owner_graph, teGraphNode);

    std::string opName = teGraphNode[0]->GetName();

    std::shared_ptr<ge::OpDesc> opDescPtr = std::make_shared<ge::OpDesc>(teGraphNode[0]->GetName(), "");

    auto opTask = std::make_shared<te::fusion::OpBuildTask>(te::fusion::OpBuildTask{graphId, taskId, sgtThreadIndex,
                                                            teGraphNode, opDescPtr,
                                                            te::fusion::OP_TASK_STATUS::OP_TASK_PENDING});
    opTask->buildType = FUZZILY_BUILD;

    TbeOpInfoPtr opInfoPtr = std::make_shared<TbeOpInfo>("matmul", "", "Matmul", "AIcoreEngine");
    TbeOpInfo &opInfo = *opInfoPtr;
    std::vector<std::pair<int64_t, int64_t>> range = {{1,10},{1,10}};
    std::vector<std::pair<int64_t, int64_t>> range1 = {{0,0},{1,10},{1,10},{1,10}};
    std::vector<std::pair<int64_t, int64_t>> range2;
    AddOpParamToTbeOpInfo({-1,-1}, "int64", "NCHW", "input0", range, true, opInfo);
    AddOpParamToTbeOpInfo({1, 10, 20, 30}, "int64", "NCHW", "input13", range2, true, opInfo);
    AddOpParamToTbeOpInfo({1}, "int64", "NCHW", "input1", range2, true, opInfo);
    AddOpParamToTbeOpInfo({0, -1, -1, -1}, "int64", "NCHW", "input2", range1, true, opInfo);
    AddOpParamToTbeOpInfo({1, 10, 20, 30}, "int64", "NCHW", "input3", range2, true, opInfo);
    AddOpParamToTbeOpInfo({9,2,3,4}, "int64", "NHWC", "output0", range2, false, opInfo);
    opInfo.SetDynamicRankType(DynamicRankType::UPGRADE_TO_SUPPORT);

    opTask->pPrebuildOp = &opInfo;
    TbeOpInfoPtr tbe_op_info_ptr;
    res = te::fusion::BinaryManager::Instance().FuzzyBuildSecondaryGeneralization(opTask, "matmul", tbe_op_info_ptr);
    EXPECT_EQ(res, true);
    opInfo.SetOpStorePattern("formatAgnostic");
    res = te::fusion::BinaryManager::Instance().FuzzyBuildSecondaryGeneralization(opTask, "matmul", tbe_op_info_ptr);
    EXPECT_EQ(res, true);
    opInfo.SetIsUnknownShape(true);
    res = te::fusion::BinaryManager::Instance().FuzzyBuildSecondaryGeneralization(opTask, "matmul", tbe_op_info_ptr);
    EXPECT_EQ(res, true);
    std::vector<TbeOpParam> inputs = {};
    (void)tbe_op_info_ptr->GetInputs(inputs);
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<TbeOpTensor> tensors = {};
        (void)inputs[i].GetTensors(tensors);
        if (tensors.size() == 0) {
            continue;
        }
        for (size_t j = 0; j < tensors.size(); j++) {
            std::string format;
            std::string oriFormat;
            std::vector<int64_t> shape = {};
            std::vector<int64_t> oriShape = {};
            std::vector<std::pair<int64_t, int64_t>> shapeRange;
            std::vector<std::pair<int64_t, int64_t>> oriShapeRange;

            (void)tensors[j].GetFormat(format);
            (void)tensors[j].GetOriginFormat(oriFormat);
            (void)tensors[j].GetShape(shape);
            (void)tensors[j].GetOriginShape(oriShape);
            (void)tensors[j].GetShapeRange(shapeRange);
            (void)tensors[j].GetOriginShapeRange(oriShapeRange);

            bool shapeAndRangeRes = false;
            if (i == 0) {
                std::vector<int64_t> newShape = {-2};
                std::vector<std::pair<int64_t, int64_t>> newShapeRange = {};
                shapeAndRangeRes = (shape == newShape && oriShape == newShape &&
                                    shapeRange == newShapeRange && oriShapeRange == newShapeRange);
            }
            if (i == 1) {
                std::vector<int64_t> newShape = {1, 10, 20, 30};
                std::vector<std::pair<int64_t, int64_t>> newShapeRange = {};
                shapeAndRangeRes = (shape == newShape && oriShape == newShape &&
                                    shapeRange == newShapeRange && oriShapeRange == newShapeRange);
            }
            if (i == 2) {
                std::vector<int64_t> newShape = {1};
                std::vector<std::pair<int64_t, int64_t>> newShapeRange = {};
                shapeAndRangeRes = (shape == newShape && oriShape == newShape &&
                                    shapeRange == newShapeRange && oriShapeRange == newShapeRange);
            }
            if (i == 3) {
                std::vector<int64_t> newShape = {0, -1, -1, -1};
                std::vector<std::pair<int64_t, int64_t>> newShapeRange = {{0,0},{0,-1},{0,-1},{0,-1}};
                shapeAndRangeRes = (shape == newShape && oriShape == newShape &&
                                    shapeRange == newShapeRange && oriShapeRange == newShapeRange);
            }
            if (i == 4) {
                std::vector<int64_t> newShape = {1, 10, 20, 30};
                std::vector<std::pair<int64_t, int64_t>> newShapeRange = {};
                shapeAndRangeRes = (shape == newShape && oriShape == newShape &&
                                    shapeRange == newShapeRange && oriShapeRange == newShapeRange);
            }
            res = (format == "ND" && oriFormat == "ND" && shapeAndRangeRes);
            EXPECT_EQ(res, true);
        }
    }
}

TEST(FuzzyBuildUTest, ParseGeneralizedResultFromTbe_failed) {
    nlohmann::json generalizedRes;
    nlohmann::json generalizedInfo;
    bool res = te::fusion::BinaryManager::Instance().ParseGeneralizedResultFromTbe(generalizedRes, "matmul",
                                                                                     generalizedInfo);
    EXPECT_EQ(res, false);
    nlohmann::json temp;
    temp["result"] = "test";
    generalizedRes.push_back(temp);
    res = te::fusion::BinaryManager::Instance().ParseGeneralizedResultFromTbe(generalizedRes, "matmul",
                                                                                generalizedInfo);
    EXPECT_EQ(res, false);
}

TEST(FuzzyBuildUTest, SetShapeAndValueToTensor_failed) {
    nlohmann::json singleGeneralizeRes;
    nlohmann::json jsonNull;
    ge::GeTensorDescPtr tenosrDescPtr = nullptr;
    singleGeneralizeRes["ori_shape"] = jsonNull;
    te::fusion::BinaryManager::Instance().SetShapeAndValueToTensor(singleGeneralizeRes, tenosrDescPtr);
}

TEST(FuzzyBuildUTest, GeneralizeWitLimitedRule) {
    std::vector<int64_t> shape = {3, 16, 16};
    std::string format = "NHWC";
    std::vector<std::pair<int64_t, int64_t>> newShapeRange;
    te::fusion::BinaryManager::Instance().GeneralizeWitLimitedRule(shape, format, newShapeRange);
}
