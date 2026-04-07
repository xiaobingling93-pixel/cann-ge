/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
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
            {{-1,200,-1,-1},         {-1,-1,-1,-1},}               // add output0
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
        singleRes["dtype"] = "int32";
        singleOp.push_back(singleRes);
    }
    json res;
    res.push_back(singleOp);
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
    std::cout << "GetMulGeneralizeFuncReturn" << std::endl;
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
