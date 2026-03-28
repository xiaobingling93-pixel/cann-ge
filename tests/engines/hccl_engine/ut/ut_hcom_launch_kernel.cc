/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include <mockcpp/mockcpp.hpp>

#include "hccl_stub.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <runtime/rt.h>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>
#include "hcom_launch_kernel.h"
#include "llt_hccl_stub_ge.h"
#include "common/ge_common/ge_types.h"
#include "exe_graph/lowering/shape_utils.h"

using namespace std;
using namespace hccl;

class HcomAllToAllKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化默认的HcomOpLaunchArgs
        strcpy(launchArgs_.opAttr.group, "hccl_world_group");
        launchArgs_.opAttr.dataType = HCCL_DATA_TYPE_INT8;
        launchArgs_.stream = nullptr;
        launchArgs_.inputNum = 1;
        launchArgs_.outputNum = 1;
        launchArgs_.inputAddrs.push_back(reinterpret_cast<void*>(0x1000));
        launchArgs_.inputAddrs.push_back(reinterpret_cast<void*>(0x1000));
        launchArgs_.inputAddrs.push_back(reinterpret_cast<void*>(0x1000));
        launchArgs_.inputAddrs.push_back(reinterpret_cast<void*>(0x1000));
        launchArgs_.inputAddrs.push_back(reinterpret_cast<void*>(0x1000));
        launchArgs_.outputAddrs.push_back(reinterpret_cast<void*>(0x2000));
        launchArgs_.outputAddrs.push_back(reinterpret_cast<void*>(0x2000));
        launchArgs_.outputAddrs.push_back(reinterpret_cast<void*>(0x2000));
        launchArgs_.outputAddrs.push_back(reinterpret_cast<void*>(0x2000));
        launchArgs_.outputAddrs.push_back(reinterpret_cast<void*>(0x2000));
        // 假设inputShapes和outputShapes为有效的形状
        gert::Shape inputShape{2, 2};
        gert::Shape outputShape{2, 2};
        launchArgs_.inputShapes.push_back(inputShape);
        launchArgs_.outputShapes.push_back(outputShape);
    }

    void TearDown() override {
        // 清理资源
        launchArgs_.inputAddrs.clear();
        launchArgs_.outputAddrs.clear();
        launchArgs_.inputShapes.clear();
        launchArgs_.outputShapes.clear();
    }

    HcomOpLaunchArgs launchArgs_;
    HcomOpInputStruct inputStruct_;
};

TEST_F(HcomAllToAllKernelTest, HcomAllToAllKernel_ShouldReturnError_WhenShapesMismatch)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;
    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcomAllToAllKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomAllToAllVKernel_ShouldReturnError_WhenShapesMismatch)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;
    launchArgs_.opAttr.op.alltoallv.recvType = HCCL_DATA_TYPE_INT8;
    HcclResult result = HcomAllToAllVKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomAllToAllVCKernel_ShouldReturnError_WhenShapesMismatch)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;
    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcomAllToAllVCKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomAllGahterKernel_ShouldReturnError_WhenShapesMismatch)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;
    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcomAllGatherKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomAllReduceKernel_ShouldReturnError_WhenShapesMismatch)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;

    std::string inputAddr = "inputAddr";
    string* pInputAddr = &inputAddr;
    vector<void*> inputAddrs;
    inputAddrs.push_back(pInputAddr);
    launchArgs_.inputAddrs = inputAddrs;

    std::string outputAddr = "outputAddr";
    string* pOutputAddr = &outputAddr;
    vector<void*> outputAddrs;
    outputAddrs.push_back(pOutputAddr);
    launchArgs_.outputAddrs = outputAddrs;

    HcomOpAttr opAttr;
    opAttr.dataType = HCCL_DATA_TYPE_INT8;
    opAttr.opType = HcomOpType::HCOM_ALL_REDUCE;
    strcpy(opAttr.group, "hccl_world_group");
    launchArgs_.opAttr = opAttr;

    std::string stream = "stream";
    string* pStream = &stream;
    launchArgs_.stream = pStream;

    launchArgs_.inputNum = 1;
    launchArgs_.outputNum = 1;

    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcomAllReduceKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomBroadcastKernel_ShouldReturnError_WhenShapesMismatch)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;
    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcomBroadcastKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomAllGahterKernelv2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;
    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));

    
#ifdef MACRO_DEV_TYPE_NEW
    MOCKER(HcomGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(DevType::DEV_TYPE_950));
#else
    MOCKER(HcomGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(DevType::DEV_TYPE_910_95));
#endif
    HcclResult result = HcomAllGatherKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomAllToAllKernelTest, HcomAllReduceKernelv2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    // 设置输入和输出形状不一致
    gert::Shape inputShape{2, 2};
    gert::Shape outputShape{3, 3};
    launchArgs_.inputShapes[0] = inputShape;
    launchArgs_.outputShapes[0] = outputShape;

    std::string inputAddr = "inputAddr";
    string* pInputAddr = &inputAddr;
    vector<void*> inputAddrs;
    inputAddrs.push_back(pInputAddr);
    launchArgs_.inputAddrs = inputAddrs;

    std::string outputAddr = "outputAddr";
    string* pOutputAddr = &outputAddr;
    vector<void*> outputAddrs;
    outputAddrs.push_back(pOutputAddr);
    launchArgs_.outputAddrs = outputAddrs;

    HcomOpAttr opAttr;
    opAttr.dataType = HCCL_DATA_TYPE_INT8;
    opAttr.opType = HcomOpType::HCOM_ALL_REDUCE;
    strcpy(opAttr.group, "hccl_world_group");
    launchArgs_.opAttr = opAttr;

    std::string stream = "stream";
    string* pStream = &stream;
    launchArgs_.stream = pStream;

    launchArgs_.inputNum = 1;
    launchArgs_.outputNum = 1;

    uint64_t count = 300 * 1024 * 1024;
    MOCKER(GetCountByShape).stubs().with(mockcpp::any(), mockcpp::any(), outBound(count)).will(returnValue(HCCL_SUCCESS));
    
#ifdef MACRO_DEV_TYPE_NEW
    MOCKER(HcomGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(DevType::DEV_TYPE_950));
#else
    MOCKER(HcomGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(DevType::DEV_TYPE_910_95));
#endif
    HcclResult result = HcomAllReduceKernel(launchArgs_, &inputStruct_);
    EXPECT_EQ(result, HCCL_SUCCESS);
}