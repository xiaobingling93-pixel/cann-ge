/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <iostream>
#include <cstring>

#include <nlohmann/json.hpp>
#include "common_stub.h"

using namespace nlohmann;
using namespace std;

void ParseJsonStrToGeneralizeResult(json res, char **generalizeResult)
{
    std::string stub = res.dump(4);
    char resCharStub[102400];
    strncpy(resCharStub, stub.c_str(), 102400);
    char *resStub = resCharStub;
    *generalizeResult = resStub;
}

void GetAscendQuantGeneralizeFuncReturn(char **generalizeResult)
{
    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
            // shape                 ori_shape
            {{-1,2,-1,-1},          {-1,2,-1,-1},},                // add input0
            {{-1,-1,-1,-1},         {-1,-1,-1,-1},}               // add output0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
            // range                                              ori_range
            {{{1,-1}, {2,2}, {1,100}, {50,90}},                {{1,-1}, {2,2}, {1,100}, {50,90}}},    // add input0
            {{{1,-1}, {1,-1}, {1,-1}, {1,-1}},                 {{1,-1}, {1,-1}, {1,-1}, {1,-1}}}     // add output0
    };

    json singleOp;
    for (size_t i = 0; i < shapeGeneralizeRes.size(); i++) {
        json singleRes;
        singleRes["shape"] = shapeGeneralizeRes[i].first;
        singleRes["ori_shape"] = shapeGeneralizeRes[i].second;
        singleRes["range"] = rangeGeneralizeRes[i].first;
        singleRes["ori_range"] = rangeGeneralizeRes[i].second;
        singleRes["format"] = "ND";
        singleOp.push_back(singleRes);
    }
    json res;
    res.push_back(singleOp);
    std::cout << "GetAscendQuantGeneralizeFuncReturn" << std::endl;
    ParseJsonStrToGeneralizeResult(res, generalizeResult);
    return;
}

void GetAddGeneralizeFuncReturn(char **generalizeResult)
{
    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                 ori_shape
    {{-1,2,-1,-1},          {-1,2,-1,-1},},                // add input0
    {{1,-1,-1,-1},          {1,-1,-1,-1},},                // add input1
    {{-1,-1,-1,-1},         {-1,-1,-1,-1},}               // add output0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
                    // range                                              ori_range
    {{{1,-1}, {2,2}, {1,100}, {50,90}},                {{1,-1}, {2,2}, {1,100}, {50,90}}},    // add input0
    {{{1,1}, {1,-1}, {3,3},{50,90}},                   {{1,1}, {1,-1}, {3,3},{50,90}}},       // add input1
    {{{1,-1}, {1,-1}, {1,-1}, {1,-1}},                 {{1,-1}, {1,-1}, {1,-1}, {1,-1}}}     // add output0
    };

    json singleOp;
    for (size_t i = 0; i < shapeGeneralizeRes.size(); i++) {
        json singleRes;
        singleRes["shape"] = shapeGeneralizeRes[i].first;
        singleRes["ori_shape"] = shapeGeneralizeRes[i].second;
        singleRes["range"] = rangeGeneralizeRes[i].first;
        singleRes["ori_range"] = rangeGeneralizeRes[i].second;
        singleRes["format"] = "ND";
        singleRes["dtype"] = "dtype";
        singleOp.push_back(singleRes);
    }
    json res;
    res.push_back(singleOp);
    std::cout << "GetAddGeneralizeFuncReturn" << std::endl;
    ParseJsonStrToGeneralizeResult(res, generalizeResult);
    return;
}

void GetMulGeneralizeFuncReturn(char **generalizeResult)
{
    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
        // shape                 ori_shape
    {{-1,2,-1,-1},          {-1,2,-1,-1},},                // add input0
    {{1,-1,-1,-1},          {1,-1,-1,-1},},                // add input1
    {{-1,-1,-1,-1},         {-1,-1,-1,-1},}               // add output0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
                    // range                                              ori_range
    {{{1,-1}, {2,2}, {1,100}, {50,90}},                {{1,-1}, {2,2}, {1,100}, {50,90}}},    // add input0
    {{{1,1}, {1,-1}, {3,3},{50,90}},                   {{1,1}, {1,-1}, {3,3},{50,90}}},       // add input1
    {{{1,-1}, {1,-1}, {1,-1}, {1,-1}},                 {{1,-1}, {1,-1}, {1,-1}, {1,-1}}}     // add output0
    };

    json singleOp;
    for (size_t i = 0; i < shapeGeneralizeRes.size(); i++) {
        json singleRes;
        singleRes["shape"] = shapeGeneralizeRes[i].first;
        singleRes["ori_shape"] = shapeGeneralizeRes[i].second;
        singleRes["range"] = rangeGeneralizeRes[i].first;
        singleRes["ori_range"] = rangeGeneralizeRes[i].second;
        singleOp.push_back(singleRes);
    }

    std::cout << "GetMulGeneralizeFuncReturn" << std::endl;
    ParseJsonStrToGeneralizeResult(singleOp, generalizeResult);
    return;
}

void GetMul1GeneralizeFuncReturn(char **generalizeResult)
{
    vector<pair<vector<int64_t>, vector<int64_t>>> shapeGeneralizeRes = {
            // shape                 ori_shape
            {{-1,2,-1,-1},          {-1,2,-1,-1},},                // add input0
            {{1,-1,-1,-1},          {1,-1,-1,-1},},                // add input1
            {{-1,-1,-1,-1},         {-1,-1,-1,-1},}               // add output0
    };
    vector<pair<vector<pair<int64_t, int64_t>>, vector<pair<int64_t, int64_t>>>> rangeGeneralizeRes = {
            // range                                              ori_range
            {{{1,-1}, {2,2}, {1,100}, {50,90}},                {{1,-1}, {2,2}, {1,100}, {50,90}}},    // add input0
            {{{1,1}, {1,-1}, {3,3},{50,90}},                   {{1,1}, {1,-1}, {3,3},{50,90}}},       // add input1
            {{{1,-1}, {1,-1}, {1,-1}, {1,-1}},                 {{1,-1}, {1,-1}, {1,-1}, {1,-1}}}     // add output0
    };

    json singleOp;
    for (size_t i = 0; i < shapeGeneralizeRes.size(); i++) {
        json singleRes;
        singleRes["shape"] = shapeGeneralizeRes[i].first;
        singleRes["ori_shape"] = shapeGeneralizeRes[i].second;
        singleRes["range"] = rangeGeneralizeRes[i].first;
        singleRes["ori_range"] = rangeGeneralizeRes[i].second;
        singleOp.push_back(singleRes);
    }
    json res;
    res.push_back(singleOp);
    std::cout << "GetMul1GeneralizeFuncReturn" << std::endl;
    ParseJsonStrToGeneralizeResult(res, generalizeResult);
    return;
}

void GetDivGeneralizeFuncReturn(char **generalizeResult)
{
    json result;
    result["result"] = "UNSUPPORTED";
    result["reason"]["param_index"] = {1,2,3,4};
    result["reason"]["type"].push_back("upper_limit");
    result["reason"]["type"].push_back("lower_limit");
    result["reason"]["type"].push_back("upper_limit");
    result["reason"]["type"].push_back("lower_limit");

    json res;
    res.push_back(result);
    std::cout << "GetDivGeneralizeFuncReturn" << std::endl;
    ParseJsonStrToGeneralizeResult(res, generalizeResult);
}

void GetOpSpecificInfoReturn(char **opSpecificInfo)
{
    json result;
    result["rangeLimit"] = "limited";
    ParseJsonStrToGeneralizeResult(result, opSpecificInfo);
}

ge::graphStatus CheckOpSupportedStub(ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "{\"isSupported\":\"True\", \"reason\":\"xxx\", \"dynamicCompileStatic\":\"True\"}";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOpSupportedStubFalse(ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "{\"isSupported\":\"False\", \"reason\":\"xxx\", \"dynamicCompileStatic\":\"False\"}";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOpSupportedStubUnknown(ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "{\"isSupported\":\"Unknown\", \"reason\":\"xxx\", \"dynamicCompileStatic\":\"Invalid\"}";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOpSupportedStubInvalid(ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "{\"isSupported\":\"Invalid\", \"reason\":\"xxx\", \"dynamicCompileStatic\":\"True\"}";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpSelectTbeFormatStub(ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "FLOAT : NCHW";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpGetSpecificInfoStub(ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "SpecificInfo : Unsupported";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOpSupportedV2Stub(const gert::OpCheckContext *context, ge::AscendString &result)
{
    std::string jsonStr = "{\"isSupported\":\"True\", \"reason\":\"xxx\", \"dynamicCompileStatic\":\"True\"}";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpSelectTbeFormatV2Stub(const gert::OpCheckContext *context, ge::AscendString &result)
{
    std::string jsonStr = "FLOAT : NCHW";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpGetSupportInfoV2Stub(const gert::OpCheckContext *context, ge::AscendString &result)
{
    std::string jsonStr = "SupportInfo : support lxfusion";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpGetSpecificInfoV2Stub(const gert::OpCheckContext *context, ge::AscendString &result)
{
    std::string jsonStr = "SpecificInfo : support lxfusion";
    result = ge::AscendString(jsonStr.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOpSupportedV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result)
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus OpSelectTbeFormatV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result)
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus OpGetSupportInfoV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result)
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus OpGetSpecificInfoV2StubFail(const gert::OpCheckContext *context, ge::AscendString &result)
{
    return ge::GRAPH_FAILED;
}

bool GenSimplifiedkeyStub(const ge::Operator &op, ge::AscendString &result)
{
    std::string jsonStr = "d=0,p=1/9,2/3,2/reflect";
    result = ge::AscendString(jsonStr.c_str());
    return true;
}